//! Signal function interception for protecting SIGPROF.
//!
//! This module intercepts signal-related functions to prevent target programs
//! from interfering with the profiler's SIGPROF signal handler. Without this
//! protection, a program could:
//! - Replace our signal handler (breaking sampling)
//! - Block SIGPROF (stopping sample delivery)
//! - Wait for SIGPROF (potentially deadlocking)
//!
//! # Intercepted Functions
//!
//! - `sigaction`: Prevents replacing the SIGPROF handler
//! - `pthread_sigmask`: Prevents blocking SIGPROF
//! - `sigprocmask`: Prevents blocking SIGPROF
//!
//! # Behavior
//!
//! When a program tries to modify SIGPROF handling:
//! - `sigaction` on SIGPROF: Silently returns success without changing the handler
//! - Blocking SIGPROF: The block request is ignored (SIGPROF removed from mask)

use crate::posix::real::LazyFn;
use libc::{c_int, sigset_t};

// =============================================================================
// SIGACTION INTERCEPTION
// =============================================================================

type SigactionFn =
    unsafe extern "C" fn(c_int, *const libc::sigaction, *mut libc::sigaction) -> c_int;
static REAL_SIGACTION: LazyFn<SigactionFn> = LazyFn::new();

/// Intercepts `sigaction` to protect the SIGPROF handler.
///
/// If the program tries to change the SIGPROF handler, this function silently
/// returns success without actually modifying the handler. This prevents the
/// profiler's signal handler from being replaced.
///
/// # Safety
///
/// This function is called via `LD_PRELOAD` interception. The caller must ensure
/// that the `act` and `oldact` pointers (if non-null) are valid.
#[no_mangle]
#[allow(unreachable_pub)] // Exposed via C ABI
pub unsafe extern "C" fn sigaction(
    signum: c_int,
    act: *const libc::sigaction,
    oldact: *mut libc::sigaction,
) -> c_int {
    // Protect SIGPROF - our profiling signal
    if signum == libc::SIGPROF {
        // If just querying the current handler, allow it (act is null)
        if !act.is_null() {
            // Program is trying to change SIGPROF handler - pretend we did it
            if !oldact.is_null() {
                // Return a zeroed structure as the "old" handler
                // This is a small lie, but maintains program compatibility
                *oldact = core::mem::zeroed();
            }
            return 0; // Success (but we didn't actually do anything)
        }
    }

    // All other signals pass through to the real sigaction
    if let Some(real_fn) = REAL_SIGACTION.get(b"sigaction\0") {
        real_fn(signum, act, oldact)
    } else {
        // If we can't resolve the real function, set errno and return error
        *libc::__errno_location() = libc::ENOSYS;
        -1
    }
}

// =============================================================================
// PTHREAD_SIGMASK INTERCEPTION
// =============================================================================

type PthreadSigmaskFn = unsafe extern "C" fn(c_int, *const sigset_t, *mut sigset_t) -> c_int;
static REAL_PTHREAD_SIGMASK: LazyFn<PthreadSigmaskFn> = LazyFn::new();

/// Intercepts `pthread_sigmask` to prevent blocking SIGPROF.
///
/// If the program tries to block SIGPROF, this function removes SIGPROF from
/// the signal set before passing it to the real function.
///
/// # Safety
///
/// This function is called via `LD_PRELOAD` interception. The caller must ensure
/// that the `set` and `oldset` pointers (if non-null) are valid.
#[no_mangle]
#[allow(unreachable_pub)] // Exposed via C ABI
pub unsafe extern "C" fn pthread_sigmask(
    how: c_int,
    set: *const sigset_t,
    oldset: *mut sigset_t,
) -> c_int {
    // If blocking signals (SIG_BLOCK or SIG_SETMASK), ensure SIGPROF isn't blocked
    if !set.is_null() && (how == libc::SIG_BLOCK || how == libc::SIG_SETMASK) {
        // Create a modified copy of the signal set with SIGPROF removed
        let mut modified_set: sigset_t = *set;
        libc::sigdelset(&raw mut modified_set, libc::SIGPROF);

        // Call the real function with the modified set
        return match REAL_PTHREAD_SIGMASK.get(b"pthread_sigmask\0") {
            Some(real_fn) => real_fn(how, &raw const modified_set, oldset),
            None => libc::EINVAL,
        };
    }

    // For SIG_UNBLOCK or when set is null, pass through unchanged
    match REAL_PTHREAD_SIGMASK.get(b"pthread_sigmask\0") {
        Some(real_fn) => real_fn(how, set, oldset),
        None => libc::EINVAL,
    }
}

// =============================================================================
// SIGPROCMASK INTERCEPTION
// =============================================================================

type SigprocmaskFn = unsafe extern "C" fn(c_int, *const sigset_t, *mut sigset_t) -> c_int;
static REAL_SIGPROCMASK: LazyFn<SigprocmaskFn> = LazyFn::new();

/// Intercepts `sigprocmask` to prevent blocking SIGPROF.
///
/// Same behavior as `pthread_sigmask` interception.
///
/// # Safety
///
/// This function is called via `LD_PRELOAD` interception. The caller must ensure
/// that the `set` and `oldset` pointers (if non-null) are valid.
#[no_mangle]
#[allow(unreachable_pub)] // Exposed via C ABI
pub unsafe extern "C" fn sigprocmask(
    how: c_int,
    set: *const sigset_t,
    oldset: *mut sigset_t,
) -> c_int {
    // If blocking signals, ensure SIGPROF isn't blocked
    if !set.is_null() && (how == libc::SIG_BLOCK || how == libc::SIG_SETMASK) {
        let mut modified_set: sigset_t = *set;
        libc::sigdelset(&raw mut modified_set, libc::SIGPROF);

        // Call the real function with the modified set
        return if let Some(real_fn) = REAL_SIGPROCMASK.get(b"sigprocmask\0") {
            real_fn(how, &raw const modified_set, oldset)
        } else {
            *libc::__errno_location() = libc::ENOSYS;
            -1
        };
    }

    if let Some(real_fn) = REAL_SIGPROCMASK.get(b"sigprocmask\0") {
        real_fn(how, set, oldset)
    } else {
        *libc::__errno_location() = libc::ENOSYS;
        -1
    }
}
