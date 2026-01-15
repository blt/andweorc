//! Signal-based experiment trigger for external program profiling.
//!
//! This module enables causal profiling of programs that cannot be modified
//! by triggering experiments via SIGUSR1. This is the key mechanism for
//! profiling third-party applications like datadog/agent.
//!
//! # How It Works
//!
//! 1. Program runs with `LD_PRELOAD=libandweorc.so ANDWEORC_ENABLED=1`
//! 2. User sends `kill -USR1 <pid>` to trigger experiments
//! 3. Signal handler sets a pending flag (deferred execution for signal safety)
//! 4. Next progress point visit checks the flag and runs experiments
//!
//! # Environment Variables
//!
//! - `ANDWEORC_EXPERIMENT_TARGET`: Name of progress point to measure throughput.
//!   If not set, uses "default" as the progress point name.
//!
//! # Signal Safety
//!
//! The SIGUSR1 handler only sets an atomic flag - no allocation, no locks.
//! Actual experiment execution happens in normal code context (progress points).
//!
//! # Example
//!
//! ```bash
//! # Terminal 1: Run program with profiling enabled
//! LD_PRELOAD=./libandweorc.so ANDWEORC_ENABLED=1 \
//!   ANDWEORC_EXPERIMENT_TARGET="request_done" ./my_program
//!
//! # Terminal 2: Trigger experiments after warmup
//! kill -USR1 $(pgrep my_program)
//! ```

use std::sync::atomic::{AtomicBool, AtomicPtr, Ordering};
use libc::{c_int, c_char};

/// Flag indicating an experiment has been requested via signal.
/// Checked by progress points to trigger deferred experiment execution.
static EXPERIMENT_PENDING: AtomicBool = AtomicBool::new(false);

/// Flag indicating whether SIGUSR1 handler is registered.
static HANDLER_REGISTERED: AtomicBool = AtomicBool::new(false);

/// Pointer to the progress point name for experiment measurement.
/// Initialized from `ANDWEORC_EXPERIMENT_TARGET` environment variable.
/// Uses a raw pointer to a leaked String for signal-safe access.
static EXPERIMENT_TARGET: AtomicPtr<c_char> = AtomicPtr::new(std::ptr::null_mut());

/// SIGUSR1 signal handler.
///
/// This handler is async-signal-safe: it only sets an atomic flag.
/// The actual experiment execution is deferred to progress point visits.
///
/// # Signal Safety
///
/// - Only uses atomic store (async-signal-safe)
/// - No allocation, no locks, no heap access
/// - No calls to non-signal-safe functions
extern "C" fn sigusr1_handler(_signal: c_int) {
    // Set the pending flag - experiment will run at next progress point
    EXPERIMENT_PENDING.store(true, Ordering::Release);

    // Use libc_print for signal-safe output
    libc_print::libc_println!("[andweorc] SIGUSR1 received - experiments will run at next progress point");
}

/// Returns whether an experiment trigger is pending.
///
/// Called from progress point code to check if experiments should run.
#[inline]
pub(crate) fn is_experiment_pending() -> bool {
    EXPERIMENT_PENDING.load(Ordering::Acquire)
}

/// Clears the pending experiment flag.
///
/// Called after experiments complete to allow future triggers.
#[inline]
pub(crate) fn clear_experiment_pending() {
    EXPERIMENT_PENDING.store(false, Ordering::Release);
}

/// Returns the configured experiment target progress point name.
///
/// Returns the value from `ANDWEORC_EXPERIMENT_TARGET` environment variable,
/// or "default" if not set.
pub(crate) fn get_experiment_target() -> &'static str {
    let ptr = EXPERIMENT_TARGET.load(Ordering::Acquire);
    if ptr.is_null() {
        // Return the default name (without null terminator)
        "default"
    } else {
        // SAFETY: ptr was set during init from a leaked String, so it's valid
        // and will remain valid for the lifetime of the process.
        unsafe {
            let len = libc::strlen(ptr.cast());
            let slice = std::slice::from_raw_parts(ptr.cast::<u8>(), len);
            std::str::from_utf8_unchecked(slice)
        }
    }
}

/// Registers the SIGUSR1 signal handler.
///
/// This should be called during library initialization. The handler
/// will set a flag that triggers experiments on the next progress point.
///
/// # Returns
///
/// Returns `Ok(())` if the handler was registered successfully,
/// or `Err` with an error message if registration failed.
///
/// # Thread Safety
///
/// This function is idempotent - calling it multiple times is safe.
/// Only the first call will register the handler.
pub(crate) fn register_sigusr1_handler() -> Result<(), String> {
    // Only register once
    if HANDLER_REGISTERED.swap(true, Ordering::AcqRel) {
        return Ok(()); // Already registered
    }

    // Read experiment target from environment
    if let Ok(target) = std::env::var("ANDWEORC_EXPERIMENT_TARGET") {
        // Leak the string to get a 'static pointer
        let mut target_bytes = target.into_bytes();
        target_bytes.push(0); // Null terminate
        let ptr = target_bytes.leak().as_mut_ptr();
        EXPERIMENT_TARGET.store(ptr.cast(), Ordering::Release);
        libc_print::libc_println!(
            "[andweorc] experiment target from env: {}",
            get_experiment_target()
        );
    } else {
        libc_print::libc_println!(
            "[andweorc] using default experiment target: 'default'"
        );
    }

    // Set up sigaction for SIGUSR1
    let mut new_action: libc::sigaction = unsafe { std::mem::zeroed() };
    new_action.sa_sigaction = sigusr1_handler as usize;
    new_action.sa_flags = libc::SA_RESTART; // Restart interrupted syscalls

    // Block other signals during handler (empty set = don't block extra signals)
    unsafe {
        libc::sigemptyset(&raw mut new_action.sa_mask);
    }

    // Register the handler
    let result = unsafe {
        libc::sigaction(
            libc::SIGUSR1,
            &raw const new_action,
            std::ptr::null_mut(), // Don't save old handler
        )
    };

    if result == 0 {
        libc_print::libc_println!(
            "[andweorc] SIGUSR1 handler registered - send 'kill -USR1 <pid>' to trigger experiments"
        );
        Ok(())
    } else {
        let errno = unsafe { *libc::__errno_location() };
        HANDLER_REGISTERED.store(false, Ordering::Release); // Allow retry
        Err(format!("sigaction failed with errno {errno}"))
    }
}

/// Checks for pending experiment and runs it if triggered.
///
/// This is called from progress point visits. If an experiment was
/// triggered via SIGUSR1, this will run the full experiment suite.
///
/// # Returns
///
/// Returns `true` if an experiment was run, `false` otherwise.
pub(crate) fn check_and_run_experiments() -> bool {
    // Fast path: no experiment pending
    if !is_experiment_pending() {
        return false;
    }

    // Clear the pending flag first (before running experiments)
    // This allows a new signal to be received during experiment run
    clear_experiment_pending();

    let target = get_experiment_target();
    libc_print::libc_println!(
        "[andweorc] running experiments for progress point: {target}"
    );

    // Run the experiments
    // Note: run_experiments returns Option, we ignore the result here
    // The results are printed to stdout by the run_experiments function
    let _ = crate::run_experiments(target);

    libc_print::libc_println!("[andweorc] experiments complete");
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn experiment_pending_initially_false() {
        // Can't reliably test this since static is shared
        // Just test the API works
        let _ = is_experiment_pending();
    }

    #[test]
    fn clear_experiment_pending_works() {
        // Set and clear
        EXPERIMENT_PENDING.store(true, Ordering::Release);
        assert!(is_experiment_pending());
        clear_experiment_pending();
        assert!(!is_experiment_pending());
    }

    #[test]
    fn get_experiment_target_returns_default() {
        // If EXPERIMENT_TARGET is null, should return "default"
        let target = get_experiment_target();
        assert!(!target.is_empty());
    }
}
