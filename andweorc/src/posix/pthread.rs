//! Pthread function interception for causal profiling.
//!
//! This module intercepts pthread synchronization functions to inject delays
//! at "yield points" where threads naturally pause. This implements the Coz
//! virtual speedup mechanism.
//!
//! # Delay Injection Points
//!
//! Delays are consumed BEFORE acquiring locks (not after) because:
//! 1. The thread is about to potentially wait anyway
//! 2. Doesn't affect lock acquisition order
//! 3. Maintains program correctness
//!
//! # Functions Intercepted
//!
//! - Mutex: `pthread_mutex_lock`
//! - Condition variables: `pthread_cond_wait`, `pthread_cond_timedwait`
//! - Reader-writer locks: `pthread_rwlock_rdlock`, `pthread_rwlock_wrlock`
//! - Thread lifecycle: `pthread_create`, `pthread_exit`

use crate::posix::real::LazyFn;
use core::ffi::c_void;
use libc::{
    c_int, pthread_attr_t, pthread_cond_t, pthread_mutex_t, pthread_rwlock_t, pthread_t, timespec,
};

// =============================================================================
// THREAD LIFECYCLE
// =============================================================================

type PthreadCreateFn = unsafe extern "C" fn(
    *mut pthread_t,
    *const pthread_attr_t,
    extern "C" fn(*mut c_void) -> *mut c_void,
    *mut c_void,
) -> c_int;

static REAL_PTHREAD_CREATE: LazyFn<PthreadCreateFn> = LazyFn::new();

/// Wrapper data passed to the thread start routine.
struct ThreadWrapper {
    real_start: extern "C" fn(*mut c_void) -> *mut c_void,
    real_arg: *mut c_void,
}

/// Wrapper function that initializes profiling before calling the user's thread function.
extern "C" fn thread_start_wrapper(arg: *mut c_void) -> *mut c_void {
    // SAFETY: arg is a Box<ThreadWrapper> that we created in pthread_create
    let wrapper = unsafe { Box::from_raw(arg.cast::<ThreadWrapper>()) };

    // Register this thread with the profiler
    // This sets up per-thread sampling and delay accumulation
    crate::experiment::register_thread(unsafe { libc::pthread_self() });

    // Start profiling for this thread if profiling is active
    if crate::is_profiling_active() {
        // Ignore errors - profiling degradation is acceptable
        let _ = crate::start_profiling();
    }

    // Call the real thread function
    // Cleanup happens automatically via thread-local storage Drop
    (wrapper.real_start)(wrapper.real_arg)
}

/// Intercepts `pthread_create` to register new threads with the profiler.
///
/// The new thread will automatically be set up for profiling before the
/// user's start routine runs.
///
/// # Safety
///
/// All pointer parameters must be valid according to `pthread_create`'s contract.
#[no_mangle]
#[allow(unreachable_pub)] // Exposed via C ABI
pub unsafe extern "C" fn pthread_create(
    thread: *mut pthread_t,
    attr: *const pthread_attr_t,
    start_routine: extern "C" fn(*mut c_void) -> *mut c_void,
    arg: *mut c_void,
) -> c_int {
    let Some(real_fn) = REAL_PTHREAD_CREATE.get(b"pthread_create\0") else {
        // Can't resolve real function - return error
        return libc::EINVAL;
    };

    // Wrap the thread start function to intercept thread initialization
    let wrapper = Box::new(ThreadWrapper {
        real_start: start_routine,
        real_arg: arg,
    });

    real_fn(
        thread,
        attr,
        thread_start_wrapper,
        Box::into_raw(wrapper).cast::<c_void>(),
    )
}

type PthreadExitFn = unsafe extern "C" fn(*mut c_void) -> !;
static REAL_PTHREAD_EXIT: LazyFn<PthreadExitFn> = LazyFn::new();

/// Intercepts `pthread_exit` to deregister threads from the profiler.
///
/// # Safety
///
/// `retval` must be valid or null per `pthread_exit`'s contract.
#[no_mangle]
#[allow(unreachable_pub)] // Exposed via C ABI
pub unsafe extern "C" fn pthread_exit(retval: *mut c_void) -> ! {
    // Deregister this thread from the profiler
    crate::experiment::deregister_thread(libc::pthread_self());

    // Call the real pthread_exit
    match REAL_PTHREAD_EXIT.get(b"pthread_exit\0") {
        Some(real_fn) => real_fn(retval),
        None => {
            // Last resort - shouldn't happen but we need to exit somehow
            libc::_exit(1);
        }
    }
}

// =============================================================================
// MUTEX OPERATIONS
// =============================================================================

type PthreadMutexLockFn = unsafe extern "C" fn(*mut pthread_mutex_t) -> c_int;
static REAL_PTHREAD_MUTEX_LOCK: LazyFn<PthreadMutexLockFn> = LazyFn::new();

/// Intercepts `pthread_mutex_lock` to inject delays before acquiring.
///
/// Delays are consumed BEFORE acquiring the mutex to:
/// 1. Not affect lock acquisition order
/// 2. Maintain program correctness
/// 3. Simulate the thread being slower at reaching this point
///
/// # Safety
///
/// `mutex` must be a valid, initialized mutex.
#[no_mangle]
#[allow(unreachable_pub)] // Exposed via C ABI
pub unsafe extern "C" fn pthread_mutex_lock(mutex: *mut pthread_mutex_t) -> c_int {
    // Consume any pending delay BEFORE acquiring the lock
    crate::consume_pending_delay();

    match REAL_PTHREAD_MUTEX_LOCK.get(b"pthread_mutex_lock\0") {
        Some(real_fn) => real_fn(mutex),
        None => libc::EINVAL,
    }
}

// =============================================================================
// CONDITION VARIABLE OPERATIONS
// =============================================================================

type PthreadCondWaitFn = unsafe extern "C" fn(*mut pthread_cond_t, *mut pthread_mutex_t) -> c_int;
static REAL_PTHREAD_COND_WAIT: LazyFn<PthreadCondWaitFn> = LazyFn::new();

/// Intercepts `pthread_cond_wait` to inject delays before waiting.
///
/// # Safety
///
/// `cond` and `mutex` must be valid and properly initialized.
#[no_mangle]
#[allow(unreachable_pub)] // Exposed via C ABI
pub unsafe extern "C" fn pthread_cond_wait(
    cond: *mut pthread_cond_t,
    mutex: *mut pthread_mutex_t,
) -> c_int {
    // Consume delay before waiting
    crate::consume_pending_delay();

    match REAL_PTHREAD_COND_WAIT.get(b"pthread_cond_wait\0") {
        Some(real_fn) => real_fn(cond, mutex),
        None => libc::EINVAL,
    }
}

type PthreadCondTimedwaitFn =
    unsafe extern "C" fn(*mut pthread_cond_t, *mut pthread_mutex_t, *const timespec) -> c_int;
static REAL_PTHREAD_COND_TIMEDWAIT: LazyFn<PthreadCondTimedwaitFn> = LazyFn::new();

/// Intercepts `pthread_cond_timedwait` to inject delays before waiting.
///
/// # Safety
///
/// `cond`, `mutex`, and `abstime` must be valid and properly initialized.
#[no_mangle]
#[allow(unreachable_pub)] // Exposed via C ABI
pub unsafe extern "C" fn pthread_cond_timedwait(
    cond: *mut pthread_cond_t,
    mutex: *mut pthread_mutex_t,
    abstime: *const timespec,
) -> c_int {
    // Consume delay before waiting
    crate::consume_pending_delay();

    match REAL_PTHREAD_COND_TIMEDWAIT.get(b"pthread_cond_timedwait\0") {
        Some(real_fn) => real_fn(cond, mutex, abstime),
        None => libc::EINVAL,
    }
}

// =============================================================================
// READER-WRITER LOCK OPERATIONS
// =============================================================================

type PthreadRwlockRdlockFn = unsafe extern "C" fn(*mut pthread_rwlock_t) -> c_int;
static REAL_PTHREAD_RWLOCK_RDLOCK: LazyFn<PthreadRwlockRdlockFn> = LazyFn::new();

/// Intercepts `pthread_rwlock_rdlock` to inject delays before acquiring.
///
/// # Safety
///
/// `rwlock` must be a valid, initialized reader-writer lock.
#[no_mangle]
#[allow(unreachable_pub)] // Exposed via C ABI
pub unsafe extern "C" fn pthread_rwlock_rdlock(rwlock: *mut pthread_rwlock_t) -> c_int {
    crate::consume_pending_delay();

    match REAL_PTHREAD_RWLOCK_RDLOCK.get(b"pthread_rwlock_rdlock\0") {
        Some(real_fn) => real_fn(rwlock),
        None => libc::EINVAL,
    }
}

type PthreadRwlockWrlockFn = unsafe extern "C" fn(*mut pthread_rwlock_t) -> c_int;
static REAL_PTHREAD_RWLOCK_WRLOCK: LazyFn<PthreadRwlockWrlockFn> = LazyFn::new();

/// Intercepts `pthread_rwlock_wrlock` to inject delays before acquiring.
///
/// # Safety
///
/// `rwlock` must be a valid, initialized reader-writer lock.
#[no_mangle]
#[allow(unreachable_pub)] // Exposed via C ABI
pub unsafe extern "C" fn pthread_rwlock_wrlock(rwlock: *mut pthread_rwlock_t) -> c_int {
    crate::consume_pending_delay();

    match REAL_PTHREAD_RWLOCK_WRLOCK.get(b"pthread_rwlock_wrlock\0") {
        Some(real_fn) => real_fn(rwlock),
        None => libc::EINVAL,
    }
}

// =============================================================================
// BARRIER OPERATIONS (Linux only)
// =============================================================================

#[cfg(target_os = "linux")]
type PthreadBarrierWaitFn = unsafe extern "C" fn(*mut libc::pthread_barrier_t) -> c_int;
#[cfg(target_os = "linux")]
static REAL_PTHREAD_BARRIER_WAIT: LazyFn<PthreadBarrierWaitFn> = LazyFn::new();

/// Intercepts `pthread_barrier_wait` to inject delays before waiting.
///
/// # Safety
///
/// `barrier` must be a valid, initialized barrier.
#[cfg(target_os = "linux")]
#[no_mangle]
#[allow(unreachable_pub)] // Exposed via C ABI
pub unsafe extern "C" fn pthread_barrier_wait(barrier: *mut libc::pthread_barrier_t) -> c_int {
    crate::consume_pending_delay();

    match REAL_PTHREAD_BARRIER_WAIT.get(b"pthread_barrier_wait\0") {
        Some(real_fn) => real_fn(barrier),
        None => libc::EINVAL,
    }
}
