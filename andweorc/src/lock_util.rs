//! Utilities for handling lock poisoning in profiling contexts.
//!
//! Lock poisoning occurs when a thread panics while holding a lock. In profiling
//! contexts, we prefer to recover and continue rather than propagate the panic,
//! since profiling data isn't safety-critical and losing all data is worse than
//! having potentially-incomplete data.

use std::sync::{PoisonError, RwLockReadGuard, RwLockWriteGuard};

/// Recovers from a poisoned read lock.
///
/// This function is used when acquiring a read lock on profiling data.
/// If the lock is poisoned (indicating a previous panic), we recover
/// the guard and continue with a warning rather than propagating the error.
///
/// # Rationale
///
/// For profiling data, graceful degradation is preferred over hard failure:
/// - Profiling data is diagnostic, not safety-critical
/// - Losing all accumulated data is worse than having potentially-incomplete data
/// - The profiler should not crash the program being profiled
///
/// # Example
///
/// ```ignore
/// use std::sync::RwLock;
///
/// let lock = RwLock::new(HashMap::new());
/// let guard = recover_read(lock.read());
/// ```
pub(crate) fn recover_read<'a, T>(
    result: Result<RwLockReadGuard<'a, T>, PoisonError<RwLockReadGuard<'a, T>>>,
) -> RwLockReadGuard<'a, T> {
    result.unwrap_or_else(|poison| {
        libc_print::libc_eprintln!("[andweorc] warning: recovering from poisoned lock");
        poison.into_inner()
    })
}

/// Recovers from a poisoned write lock.
///
/// Similar to [`recover_read`], but for write locks.
///
/// # Example
///
/// ```ignore
/// use std::sync::RwLock;
///
/// let lock = RwLock::new(HashMap::new());
/// let mut guard = recover_write(lock.write());
/// guard.insert("key", "value");
/// ```
pub(crate) fn recover_write<'a, T>(
    result: Result<RwLockWriteGuard<'a, T>, PoisonError<RwLockWriteGuard<'a, T>>>,
) -> RwLockWriteGuard<'a, T> {
    result.unwrap_or_else(|poison| {
        libc_print::libc_eprintln!("[andweorc] warning: recovering from poisoned lock");
        poison.into_inner()
    })
}
