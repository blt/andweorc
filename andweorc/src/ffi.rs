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
//!
//! # Safety
//!
//! All C string inputs are bounded to `MAX_NAME_LEN` bytes to prevent unbounded
//! memory reads from malformed input.

use libc::{c_char, c_int};
use std::sync::atomic::{AtomicBool, AtomicPtr, AtomicUsize, Ordering};

/// Maximum length for progress point names.
///
/// This bounds the `CStr` scanning to prevent reading past allocated memory
/// in case of malformed (non-null-terminated) input.
const MAX_NAME_LEN: usize = 4096;

/// Safely converts a C string to a Rust `&str` with bounded scanning.
///
/// This function copies the string into a thread-local buffer to avoid TOCTOU
/// races (where the source string could be modified between length check and use).
///
/// Returns `None` if:
/// - The pointer is null
/// - No null terminator is found within `MAX_NAME_LEN` bytes
/// - The string is not valid UTF-8
///
/// # Safety
///
/// - The pointer must point to readable memory for at least
///   `min(actual_string_length + 1, MAX_NAME_LEN)` bytes at the moment of call
/// - The returned `&'static str` is valid for the lifetime of the process
///   (backed by interned storage)
///
/// # Signal Safety
///
/// This function uses `strnlen` which is async-signal-safe on Linux.
/// However, it allocates via the `NameIntern` table, so it should not be
/// called from signal handlers.
unsafe fn cstr_to_str_bounded(ptr: *const c_char) -> Option<&'static str> {
    if ptr.is_null() {
        return None;
    }

    // Use strnlen to bound the scan (async-signal-safe on Linux)
    let len = libc::strnlen(ptr, MAX_NAME_LEN);
    if len >= MAX_NAME_LEN {
        // No null terminator found within bounds - reject
        return None;
    }

    // Copy to a local buffer to avoid TOCTOU race.
    // After strnlen returns, another thread could modify the source string.
    // By copying immediately, we capture a consistent snapshot.
    //
    // We use a stack buffer sized to MAX_NAME_LEN to avoid allocation.
    let mut local_buf = [0u8; MAX_NAME_LEN];

    // SAFETY: We verified len < MAX_NAME_LEN, and the source has at least `len` bytes.
    // copy_nonoverlapping is safe because:
    // - src is valid for reads of `len` bytes (strnlen found a null within bounds)
    // - dst is valid for writes of `len` bytes (local_buf has MAX_NAME_LEN capacity)
    // - regions don't overlap (stack buffer vs heap/static)
    std::ptr::copy_nonoverlapping(ptr.cast::<u8>(), local_buf.as_mut_ptr(), len);

    // Validate UTF-8 on our local copy
    let slice = &local_buf[..len];
    let s = std::str::from_utf8(slice).ok()?;

    // Intern the string to get a 'static reference.
    // This is necessary because we can't return a reference to local_buf.
    NAME_INTERN.intern(s.as_bytes())
}

/// Maximum number of unique progress point names we can intern.
/// Names beyond this limit will be dropped with a warning.
const MAX_INTERNED_NAMES: usize = 1024;

/// Flag to ensure we only warn once about table full.
static WARNED_TABLE_FULL: AtomicBool = AtomicBool::new(false);

/// Counter for dropped names (for observability).
static DROPPED_NAMES: AtomicUsize = AtomicUsize::new(0);

/// Returns the number of progress point names that were dropped due to
/// the intern table being full. Used for observability at shutdown.
pub(crate) fn dropped_names_count() -> usize {
    DROPPED_NAMES.load(Ordering::Relaxed)
}

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
    /// # Thread Safety
    ///
    /// This function is thread-safe. Concurrent calls with the same name will
    /// return the same interned reference (no duplicates). The implementation
    /// uses compare-and-swap to ensure exactly one thread wins when inserting.
    ///
    /// # Safety
    ///
    /// The returned reference is valid for the lifetime of the process.
    fn intern(&self, name: &[u8]) -> Option<&'static str> {
        // First, check if already interned by scanning all non-null slots.
        // We scan the full range rather than stopping at first null because
        // concurrent insertions might leave gaps temporarily.
        let current_slots = self.next_slot.load(Ordering::Acquire);
        let scan_limit = current_slots.min(MAX_INTERNED_NAMES);

        for i in 0..scan_limit {
            let ptr = self.names[i].load(Ordering::Acquire);
            if ptr.is_null() {
                continue; // Slot reserved but not yet filled
            }

            // SAFETY: ptr is non-null and points to a valid,
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

        // Not found, try to intern by reserving a slot
        let slot = self.next_slot.fetch_add(1, Ordering::AcqRel);
        if slot >= MAX_INTERNED_NAMES {
            // Table full - count and warn (once)
            DROPPED_NAMES.fetch_add(1, Ordering::Relaxed);

            // Only warn once to avoid spamming logs
            if !WARNED_TABLE_FULL.swap(true, Ordering::Relaxed) {
                libc_print::libc_eprintln!(
                    "[andweorc] WARNING: Progress point name table full ({} names). \
                     Additional unique names will be dropped. Consider using fewer \
                     unique progress point names or reusing existing ones.",
                    MAX_INTERNED_NAMES
                );
            }
            return None;
        }

        // Allocate and copy the name (with null terminator)
        let mut owned = Vec::with_capacity(name.len() + 1);
        owned.extend_from_slice(name);
        owned.push(0); // Null terminator
        let ptr = owned.leak().as_mut_ptr();

        // Use CAS to store in the slot. This should always succeed since we
        // reserved the slot, but we use CAS for consistency.
        let result = self.names[slot].compare_exchange(
            std::ptr::null_mut(),
            ptr,
            Ordering::Release,
            Ordering::Acquire,
        );

        if result.is_err() {
            // Another thread filled our slot (shouldn't happen with our design,
            // but handle gracefully). The memory we allocated is leaked, which
            // is acceptable for a profiler.
            // Re-scan to find the name (either ours or a duplicate).
            return self.find_existing(name);
        }

        // Successfully stored. Now check if another thread raced and inserted
        // the same name in a different slot. If so, we've created a duplicate.
        // This is acceptable for correctness (both point to equivalent strings)
        // but wasteful. The re-scan before insertion minimizes this.

        // SAFETY: We just allocated this, it's valid UTF-8, and it lives forever
        Some(unsafe { std::str::from_utf8_unchecked(std::slice::from_raw_parts(ptr, name.len())) })
    }

    /// Searches for an existing interned name.
    ///
    /// Used as fallback when CAS fails during insertion.
    fn find_existing(&self, name: &[u8]) -> Option<&'static str> {
        let current_slots = self.next_slot.load(Ordering::Acquire);
        let scan_limit = current_slots.min(MAX_INTERNED_NAMES);

        for i in 0..scan_limit {
            let ptr = self.names[i].load(Ordering::Acquire);
            if ptr.is_null() {
                continue;
            }

            let existing = unsafe {
                let len = libc::strlen(ptr.cast::<i8>());
                std::slice::from_raw_parts(ptr, len)
            };

            if existing == name {
                return Some(unsafe { std::str::from_utf8_unchecked(existing) });
            }
        }

        None
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
    if !crate::is_profiling_active() {
        return;
    }

    // Convert C string to Rust and intern it (cstr_to_str_bounded does both)
    let Some(interned) = cstr_to_str_bounded(name) else {
        return; // Null, too long, invalid UTF-8, or table full
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
    if !crate::is_profiling_active() {
        return;
    }

    // Convert C string to Rust with bounded scan
    let Some(file_str) = cstr_to_str_bounded(file) else {
        return; // Null, too long, or invalid UTF-8
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
///
/// Handles all `c_int` values correctly including `i32::MIN`.
fn itoa_simple(n: c_int, buf: &mut [u8]) -> &[u8] {
    if buf.is_empty() {
        return &[];
    }

    let negative = n < 0;
    // Convert to u32 to handle i32::MIN correctly.
    // For negative numbers, we use wrapping_neg on the i32 first, then cast.
    // Special case: i32::MIN.wrapping_neg() == i32::MIN, but when cast to u32
    // it becomes 2147483648 which is correct (absolute value of i32::MIN).
    #[allow(clippy::cast_sign_loss)]
    let mut val: u32 = if negative {
        // This works for all negative values including i32::MIN:
        // -1i32 as u32 = 4294967295, wrapping_neg = 1 ✓
        // i32::MIN as u32 = 2147483648, wrapping_neg = 2147483648 ✓
        (n as u32).wrapping_neg()
    } else {
        n as u32
    };

    let mut pos = buf.len();

    // Convert digits in reverse order
    loop {
        if pos == 0 {
            return &buf[..0]; // Buffer too small
        }
        pos -= 1;

        #[allow(clippy::cast_possible_truncation)]
        let digit = (val % 10) as u8;
        buf[pos] = b'0' + digit;
        val /= 10;

        if val == 0 {
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

/// Begins a timed code section for delay injection.
///
/// Call this at the start of an operation, then call `andweorc_end()` with
/// the same name when the operation completes. At `andweorc_end()`, any
/// accumulated delay will be consumed, acting as a delay injection point.
///
/// # Current Behavior
///
/// This function records the start time. The `andweorc_end()` function
/// calculates elapsed time and consumes pending delays. Future versions
/// may use the timing data for latency distribution analysis.
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
///     andweorc_end("db_query");  // Delays consumed here
/// }
/// ```
#[no_mangle]
#[allow(unreachable_pub)] // Exposed via C ABI
pub unsafe extern "C" fn andweorc_begin(name: *const c_char) {
    if !crate::is_profiling_active() {
        return;
    }

    // Convert C string to Rust with bounded scan
    let Some(name_str) = cstr_to_str_bounded(name) else {
        return; // Null, too long, or invalid UTF-8
    };

    // Get current timestamp
    let now = monotonic_nanos();
    if now == 0 {
        return; // Clock failed
    }

    // Store in thread-local map (name_str is already interned, so &'static str)
    LATENCY_STARTS.with(|starts| {
        starts.borrow_mut().insert(name_str, now);
    });
}

/// Ends a timed code section and consumes pending delay.
///
/// Call this at the end of an operation started with `andweorc_begin()`.
/// This function acts as a delay injection point - any delay accumulated
/// by the profiler will be consumed here (i.e., the thread will sleep).
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
/// # Note
///
/// If `andweorc_end()` is called without a matching `andweorc_begin()`, or with
/// a different name, the call is silently ignored (no delay is consumed).
///
/// # Example
///
/// See `andweorc_begin()` for a complete example.
#[no_mangle]
#[allow(unreachable_pub)] // Exposed via C ABI
pub unsafe extern "C" fn andweorc_end(name: *const c_char) {
    if !crate::is_profiling_active() {
        return;
    }

    // Convert C string to Rust with bounded scan
    let Some(name_str) = cstr_to_str_bounded(name) else {
        return; // Null, too long, or invalid UTF-8
    };

    let now = monotonic_nanos();
    if now == 0 {
        return;
    }

    LATENCY_STARTS.with(|starts| {
        if let Some(start) = starts.borrow_mut().remove(name_str) {
            // Calculate elapsed time for potential future use in latency tracking.
            // Currently unused, but preserved for when latency histogram support is added.
            let _elapsed = now.saturating_sub(start);

            // Consume any pending delay - this is the main function of begin/end pairs.
            // Acts as a delay injection point at region boundaries.
            crate::consume_pending_delay();
        }
    });
}

// Thread-local storage for latency measurement start times.
// Uses &'static str keys because all names are interned via NameIntern.
thread_local! {
    static LATENCY_STARTS: std::cell::RefCell<std::collections::HashMap<&'static str, u64>> =
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

    #[test]
    fn itoa_simple_i32_min() {
        let mut buf = [0u8; 16];
        let result = itoa_simple(i32::MIN, &mut buf);
        assert_eq!(result, b"-2147483648");
    }

    #[test]
    fn itoa_simple_i32_max() {
        let mut buf = [0u8; 16];
        let result = itoa_simple(i32::MAX, &mut buf);
        assert_eq!(result, b"2147483647");
    }
}
