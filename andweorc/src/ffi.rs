//! C-compatible FFI for manual progress points.
//!
//! This module provides a C API for programs that want to explicitly mark
//! progress points in their code. While the profiler automatically intercepts
//! synchronization points via `LD_PRELOAD`, manual progress points allow more
//! precise throughput measurement.
//!
//! # Usage from C
//!
//! ```c
//! #include "andweorc.h"
//!
//! void process_request(void) {
//!     // ... do work ...
//!     andweorc_progress("request_done");
//! }
//! ```
//!
//! # Thread Safety
//!
//! All functions in this module are thread-safe and can be called from any thread.
//! They use lock-free operations internally.

use libc::{c_char, c_int};
use std::ffi::CStr;
use std::sync::atomic::{AtomicPtr, Ordering};

/// Maximum number of unique progress point names we can intern.
/// Names beyond this limit will be silently ignored.
const MAX_INTERNED_NAMES: usize = 1024;

/// Interned name storage. We leak memory intentionally - progress point names
/// are expected to be static strings or long-lived, and the profiler runs for
/// the lifetime of the process.
struct NameIntern {
    /// Array of interned name pointers. Each slot is either null or points to
    /// a heap-allocated, null-terminated string that lives forever.
    names: [AtomicPtr<u8>; MAX_INTERNED_NAMES],
    /// Next slot to try for insertion.
    next_slot: std::sync::atomic::AtomicUsize,
}

impl NameIntern {
    /// Creates a new empty name intern table.
    #[allow(clippy::declare_interior_mutable_const)] // Intentional const for array init
    const fn new() -> Self {
        // Initialize all slots to null
        // Using a macro because we can't use a loop in const context
        macro_rules! null_array {
            ($n:expr) => {{
                const NULL: AtomicPtr<u8> = AtomicPtr::new(std::ptr::null_mut());
                [NULL; $n]
            }};
        }

        Self {
            names: null_array!(MAX_INTERNED_NAMES),
            next_slot: std::sync::atomic::AtomicUsize::new(0),
        }
    }

    /// Interns a name, returning a static reference.
    ///
    /// If the name is already interned, returns the existing reference.
    /// If the table is full, returns `None`.
    ///
    /// # Safety
    ///
    /// The returned reference is valid for the lifetime of the process.
    fn intern(&self, name: &[u8]) -> Option<&'static str> {
        // First, check if already interned
        for i in 0..MAX_INTERNED_NAMES {
            let ptr = self.names[i].load(Ordering::Acquire);
            if ptr.is_null() {
                break; // No more entries
            }

            // SAFETY: ptr is either null (handled above) or points to a valid,
            // null-terminated string that we allocated.
            let existing = unsafe {
                let len = libc::strlen(ptr.cast::<i8>());
                std::slice::from_raw_parts(ptr, len)
            };

            if existing == name {
                // Already interned
                // SAFETY: The string is valid UTF-8 (we only intern valid UTF-8)
                // and lives forever.
                return Some(unsafe { std::str::from_utf8_unchecked(existing) });
            }
        }

        // Not found, try to intern
        let slot = self.next_slot.fetch_add(1, Ordering::Relaxed);
        if slot >= MAX_INTERNED_NAMES {
            // Table full
            return None;
        }

        // Allocate and copy the name (with null terminator)
        let mut owned = Vec::with_capacity(name.len() + 1);
        owned.extend_from_slice(name);
        owned.push(0); // Null terminator

        let ptr = owned.leak().as_mut_ptr();

        // Store in the slot
        self.names[slot].store(ptr, Ordering::Release);

        // SAFETY: We just allocated this, it's valid UTF-8, and it lives forever
        Some(unsafe { std::str::from_utf8_unchecked(std::slice::from_raw_parts(ptr, name.len())) })
    }
}

// SAFETY: NameIntern uses atomic operations for all shared state
unsafe impl Sync for NameIntern {}
unsafe impl Send for NameIntern {}

/// Global name intern table.
static NAME_INTERN: NameIntern = NameIntern::new();

/// Records a visit to a named progress point.
///
/// This function marks that one unit of useful work has completed. The profiler
/// uses these markers to measure throughput and correlate it with virtual speedups.
///
/// # Parameters
///
/// * `name` - A null-terminated C string identifying the progress point.
///   This should be a descriptive name like `requests_processed` or `items_completed`.
///   The name is interned, so the same string pointer doesn't need to be passed each time.
///
/// # Thread Safety
///
/// This function is thread-safe and can be called from any thread.
///
/// # Performance
///
/// When profiling is not active, this function returns immediately with minimal overhead.
/// When active, it performs atomic operations to record the visit.
///
/// # Example
///
/// ```c
/// void handle_request(void) {
///     // ... process request ...
///     andweorc_progress("request_done");
/// }
/// ```
#[no_mangle]
#[allow(unreachable_pub)] // Exposed via C ABI
pub unsafe extern "C" fn andweorc_progress(name: *const c_char) {
    if name.is_null() || !crate::is_profiling_active() {
        return;
    }

    // Convert C string to Rust slice
    let c_str = CStr::from_ptr(name);
    let bytes = c_str.to_bytes();

    // Validate UTF-8
    let Ok(name_str) = std::str::from_utf8(bytes) else {
        return; // Invalid UTF-8, silently ignore
    };

    // Intern the name to get a static reference
    let Some(interned) = NAME_INTERN.intern(name_str.as_bytes()) else {
        return; // Table full, silently ignore
    };

    // Get or create the progress point and record the visit
    if let Ok(exp) = crate::experiment::try_get_instance() {
        exp.progress(interned).note_visit();
    }
}

/// Records a visit to a progress point identified by file and line number.
///
/// This is a convenience function that creates a progress point name from
/// the source location. Useful with the `ANDWEORC_PROGRESS()` macro.
///
/// # Parameters
///
/// * `file` - A null-terminated C string containing the source file name.
/// * `line` - The line number in the source file.
///
/// # Thread Safety
///
/// This function is thread-safe and can be called from any thread.
///
/// # Example
///
/// ```c
/// // Usually called via macro:
/// #define ANDWEORC_PROGRESS() andweorc_progress_named(__FILE__, __LINE__)
///
/// void process(void) {
///     ANDWEORC_PROGRESS();  // Records visit at this source location
/// }
/// ```
#[no_mangle]
#[allow(unreachable_pub)] // Exposed via C ABI
pub unsafe extern "C" fn andweorc_progress_named(file: *const c_char, line: c_int) {
    if file.is_null() || !crate::is_profiling_active() {
        return;
    }

    // Convert C string to Rust
    let c_str = CStr::from_ptr(file);
    let Ok(file_str) = c_str.to_str() else {
        return; // Invalid UTF-8
    };

    // Create a name like "file.c:42"
    // Use a stack buffer to avoid allocation in the hot path
    let mut buf = [0u8; 256];
    let name = if file_str.len() + 12 < buf.len() {
        // Format into stack buffer
        let mut pos = 0;
        for &b in file_str.as_bytes() {
            if pos >= buf.len() - 12 {
                break;
            }
            buf[pos] = b;
            pos += 1;
        }
        buf[pos] = b':';
        pos += 1;

        // Format line number
        let line_str = itoa_simple(line, &mut buf[pos..]);
        pos += line_str.len();

        &buf[..pos]
    } else {
        return; // Name too long
    };

    // Intern and record
    let Some(interned) = NAME_INTERN.intern(name) else {
        return;
    };

    if let Ok(exp) = crate::experiment::try_get_instance() {
        exp.progress(interned).note_visit();
    }
}

/// Simple integer to ASCII conversion without allocation.
fn itoa_simple(mut n: c_int, buf: &mut [u8]) -> &[u8] {
    if buf.is_empty() {
        return &[];
    }

    let negative = n < 0;
    if negative {
        n = n.wrapping_neg();
    }

    let mut pos = buf.len();

    // Convert digits in reverse order
    loop {
        if pos == 0 {
            return &buf[..0]; // Buffer too small
        }
        pos -= 1;

        #[allow(clippy::cast_sign_loss)]
        let digit = (n % 10) as u8;
        buf[pos] = b'0' + digit;
        n /= 10;

        if n == 0 {
            break;
        }
    }

    // Add negative sign if needed
    if negative {
        if pos == 0 {
            return &buf[..0];
        }
        pos -= 1;
        buf[pos] = b'-';
    }

    &buf[pos..]
}

/// Begins a latency measurement section.
///
/// Call this at the start of an operation you want to measure, then call
/// `andweorc_end()` with the same name when the operation completes. The
/// profiler will track the distribution of latencies for this operation.
///
/// # Parameters
///
/// * `name` - A null-terminated C string identifying the operation.
///
/// # Thread Safety
///
/// Begin/end pairs must be called from the same thread. Nested pairs with
/// different names are allowed.
///
/// # Example
///
/// ```c
/// void database_query(void) {
///     andweorc_begin("db_query");
///     // ... perform query ...
///     andweorc_end("db_query");
/// }
/// ```
#[no_mangle]
#[allow(unreachable_pub)] // Exposed via C ABI
pub unsafe extern "C" fn andweorc_begin(name: *const c_char) {
    if name.is_null() || !crate::is_profiling_active() {
        return;
    }

    // Get current timestamp
    let now = monotonic_nanos();
    if now == 0 {
        return; // Clock failed
    }

    // Store in thread-local map
    LATENCY_STARTS.with(|starts| {
        let c_str = CStr::from_ptr(name);
        if let Ok(name_str) = c_str.to_str() {
            starts.borrow_mut().insert(name_str.to_owned(), now);
        }
    });
}

/// Ends a latency measurement section.
///
/// Call this at the end of an operation started with `andweorc_begin()`.
/// The elapsed time will be recorded for profiling analysis.
///
/// # Parameters
///
/// * `name` - A null-terminated C string identifying the operation.
///   Must match the name passed to `andweorc_begin()`.
///
/// # Thread Safety
///
/// Must be called from the same thread as the corresponding `andweorc_begin()`.
///
/// # Example
///
/// See `andweorc_begin()` for a complete example.
#[no_mangle]
#[allow(unreachable_pub)] // Exposed via C ABI
pub unsafe extern "C" fn andweorc_end(name: *const c_char) {
    if name.is_null() || !crate::is_profiling_active() {
        return;
    }

    let now = monotonic_nanos();
    if now == 0 {
        return;
    }

    LATENCY_STARTS.with(|starts| {
        let c_str = CStr::from_ptr(name);
        if let Ok(name_str) = c_str.to_str() {
            if let Some(start) = starts.borrow_mut().remove(name_str) {
                let elapsed = now.saturating_sub(start);
                // Record the latency sample
                // For now, we just consume any pending delay - future versions
                // could track latency distributions
                let _ = elapsed;
                crate::consume_pending_delay();
            }
        }
    });
}

// Thread-local storage for latency measurement start times.
thread_local! {
    static LATENCY_STARTS: std::cell::RefCell<std::collections::HashMap<String, u64>> =
        std::cell::RefCell::new(std::collections::HashMap::new());
}

/// Gets the current monotonic time in nanoseconds.
fn monotonic_nanos() -> u64 {
    let mut ts = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    // SAFETY: clock_gettime with CLOCK_MONOTONIC is async-signal-safe
    let result = unsafe { libc::clock_gettime(libc::CLOCK_MONOTONIC, &raw mut ts) };

    if result != 0 {
        return 0;
    }

    #[allow(clippy::cast_sign_loss)]
    let nanos = (ts.tv_sec as u64)
        .saturating_mul(1_000_000_000)
        .saturating_add(ts.tv_nsec as u64);
    nanos
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn name_intern_returns_same_reference() {
        let intern = NameIntern::new();
        let name1 = intern.intern(b"test").unwrap();
        let name2 = intern.intern(b"test").unwrap();
        assert_eq!(name1.as_ptr(), name2.as_ptr());
    }

    #[test]
    fn name_intern_different_names() {
        let intern = NameIntern::new();
        let name1 = intern.intern(b"foo").unwrap();
        let name2 = intern.intern(b"bar").unwrap();
        assert_ne!(name1, name2);
    }

    #[test]
    fn itoa_simple_positive() {
        let mut buf = [0u8; 16];
        let result = itoa_simple(42, &mut buf);
        assert_eq!(result, b"42");
    }

    #[test]
    fn itoa_simple_negative() {
        let mut buf = [0u8; 16];
        let result = itoa_simple(-123, &mut buf);
        assert_eq!(result, b"-123");
    }

    #[test]
    fn itoa_simple_zero() {
        let mut buf = [0u8; 16];
        let result = itoa_simple(0, &mut buf);
        assert_eq!(result, b"0");
    }
}
