//! Progress points for throughput measurement.
//!
//! Progress points mark locations in code where meaningful work completes.
//! The causal profiler uses these to measure throughput and correlate it
//! with virtual speedups at different code locations.
//!
//! # Example
//!
//! ```ignore
//! use andweorc::progress;
//!
//! fn handle_request() {
//!     // ... process request ...
//!     progress!("request_done");
//! }
//! ```

use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::Arc;

/// Gets the current monotonic time in nanoseconds.
///
/// # Returns
///
/// The current monotonic time in nanoseconds, or 0 if `clock_gettime` fails.
/// Failure should only occur due to programmer error (invalid `clock_id`),
/// not at runtime, so returning 0 is a reasonable fallback.
fn monotonic_nanos() -> u64 {
    let mut ts = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    // SAFETY: clock_gettime with CLOCK_MONOTONIC is async-signal-safe.
    // We pass a valid mutable pointer to ts.
    let result = unsafe { libc::clock_gettime(libc::CLOCK_MONOTONIC, &raw mut ts) };

    // clock_gettime only fails if the clock_id is invalid, which is a programmer
    // error. Return 0 as a safe fallback rather than panicking in production.
    if result != 0 {
        return 0;
    }

    // Convert to nanoseconds
    #[allow(clippy::cast_sign_loss)]
    let nanos = (ts.tv_sec as u64)
        .saturating_mul(1_000_000_000)
        .saturating_add(ts.tv_nsec as u64);
    nanos
}

/// A named progress point that tracks throughput.
///
/// Each visit to this progress point represents one unit of useful work
/// completed by the program.
pub struct Progress {
    /// The name identifying this progress point.
    name: &'static str,
    /// Number of times this progress point has been visited.
    visits: AtomicU32,
    /// Timestamp of the first visit (monotonic nanoseconds).
    first_visit_ns: AtomicU64,
    /// Timestamp of the most recent visit (monotonic nanoseconds).
    last_visit_ns: AtomicU64,
}

impl std::fmt::Debug for Progress {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Progress")
            .field("name", &self.name)
            .field("visits", &self.visits.load(Ordering::Relaxed))
            .field(
                "first_visit_ns",
                &self.first_visit_ns.load(Ordering::Relaxed),
            )
            .field("last_visit_ns", &self.last_visit_ns.load(Ordering::Relaxed))
            .finish()
    }
}

impl Progress {
    /// Creates a new progress point with the given name.
    pub(crate) fn new(name: &'static str) -> Self {
        Self {
            name,
            visits: AtomicU32::new(0),
            first_visit_ns: AtomicU64::new(0),
            last_visit_ns: AtomicU64::new(0),
        }
    }

    /// Returns a shared reference to the progress point with the given name.
    ///
    /// Creates the progress point if it doesn't exist.
    ///
    /// # Panics
    ///
    /// Panics if the experiment singleton failed to initialize (e.g., SIGPROF
    /// signal handler could not be registered).
    #[must_use]
    pub fn get_instance(name: &'static str) -> Arc<Self> {
        crate::experiment::get_instance().progress(name)
    }

    /// Records a visit to this progress point.
    ///
    /// This should be called each time the program completes one unit of
    /// meaningful work (e.g., processing one request, completing one iteration).
    pub fn note_visit(&self) {
        let now = monotonic_nanos();
        // Use Release to publish the increment to readers
        self.visits.fetch_add(1, Ordering::Release);

        // Set first visit timestamp if this is the first visit
        // Use compare_exchange to atomically set only if still 0
        // Release on success to publish the timestamp
        let _ = self
            .first_visit_ns
            .compare_exchange(0, now, Ordering::Release, Ordering::Relaxed);

        // Always update last visit timestamp
        // Use Release to publish to readers
        self.last_visit_ns.store(now, Ordering::Release);
    }

    /// Returns the total number of visits.
    #[must_use]
    pub fn visit_count(&self) -> u32 {
        self.visits.load(Ordering::Relaxed)
    }

    /// Returns the elapsed time in nanoseconds between first and last visit.
    ///
    /// Returns 0 if there have been fewer than 2 visits.
    #[must_use]
    pub fn elapsed_nanos(&self) -> u64 {
        // Use Acquire to synchronize with Release stores in note_visit/reset
        let first = self.first_visit_ns.load(Ordering::Acquire);
        let last = self.last_visit_ns.load(Ordering::Acquire);
        if first == 0 || last == 0 || last <= first {
            return 0;
        }
        last - first
    }

    /// Returns the throughput in visits per second.
    ///
    /// Returns 0.0 if there's not enough data to calculate throughput.
    #[must_use]
    pub fn throughput(&self) -> f64 {
        // Use Acquire to synchronize with Release stores in note_visit/reset
        let visits = self.visits.load(Ordering::Acquire);
        if visits < 2 {
            return 0.0;
        }

        let elapsed_ns = self.elapsed_nanos();
        if elapsed_ns == 0 {
            return 0.0;
        }

        // visits per nanosecond * 1e9 = visits per second
        let visits_f64 = f64::from(visits - 1); // -1 because first visit doesn't count
                                                // Allow precision loss - nanosecond precision isn't critical for throughput
        #[allow(clippy::cast_precision_loss)]
        let elapsed_secs = elapsed_ns as f64 / 1_000_000_000.0;
        visits_f64 / elapsed_secs
    }

    /// Resets the progress point counters and timestamps.
    ///
    /// # Concurrency Note
    ///
    /// This method is NOT atomic. If `note_visit()` is called concurrently
    /// with `reset()`, visits may be lost. This is acceptable for the profiler
    /// because:
    ///
    /// 1. Reset is only called between experiment rounds, not during active profiling
    /// 2. Lost visits during reset don't produce incorrect throughput values
    ///    (`throughput()` returns 0.0 for edge cases like zero timestamps)
    /// 3. The next experiment round will start fresh
    ///
    /// Callers should ensure the workload is quiesced before calling reset.
    pub fn reset(&self) {
        // Use Release ordering to ensure previous writes are visible before
        // any reader sees the reset state. Note: this doesn't make the reset
        // atomic - concurrent note_visit() calls may still race.
        self.visits.store(0, Ordering::Release);
        self.first_visit_ns.store(0, Ordering::Release);
        self.last_visit_ns.store(0, Ordering::Release);
    }
}

/// Marks a progress point in the program.
///
/// Each call to this macro represents one unit of useful work completed.
/// The causal profiler measures throughput by counting these visits.
///
/// # Example
///
/// ```ignore
/// use andweorc::progress;
///
/// fn process_item(item: Item) {
///     // ... do work ...
///     progress!("item_processed");
/// }
/// ```
#[macro_export]
macro_rules! progress {
    ($name:expr) => {
        $crate::progress_point::Progress::get_instance($name).note_visit();
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_progress_has_zero_visits() {
        let p = Progress::new("test");
        assert_eq!(p.visit_count(), 0);
    }

    #[test]
    fn new_progress_has_zero_elapsed() {
        let p = Progress::new("test");
        assert_eq!(p.elapsed_nanos(), 0);
    }

    #[test]
    fn new_progress_has_zero_throughput() {
        let p = Progress::new("test");
        assert!((p.throughput() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn single_visit_no_throughput() {
        let p = Progress::new("test");
        p.note_visit();
        assert_eq!(p.visit_count(), 1);
        // Throughput requires at least 2 visits
        assert!((p.throughput() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn multiple_visits_accumulate() {
        let p = Progress::new("test");
        for _ in 0..100 {
            p.note_visit();
        }
        assert_eq!(p.visit_count(), 100);
    }

    #[test]
    fn reset_clears_all_state() {
        let p = Progress::new("test");
        p.note_visit();
        p.note_visit();
        p.reset();
        assert_eq!(p.visit_count(), 0);
        assert_eq!(p.elapsed_nanos(), 0);
        assert!((p.throughput() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn elapsed_nanos_is_non_negative() {
        let p = Progress::new("test");
        p.note_visit();
        std::thread::sleep(std::time::Duration::from_millis(1));
        p.note_visit();
        // Elapsed should be at least 1ms = 1_000_000 ns
        assert!(p.elapsed_nanos() >= 1_000_000);
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    /// Test helper: calculate throughput from raw values
    fn calc_throughput(visits: u32, first_ns: u64, last_ns: u64) -> f64 {
        if visits < 2 {
            return 0.0;
        }
        if first_ns == 0 || last_ns == 0 || last_ns <= first_ns {
            return 0.0;
        }
        let elapsed_ns = last_ns - first_ns;
        let visits_f64 = f64::from(visits - 1);
        #[allow(clippy::cast_precision_loss)]
        let elapsed_secs = elapsed_ns as f64 / 1_000_000_000.0;
        visits_f64 / elapsed_secs
    }

    proptest! {
        /// Property: Throughput is always non-negative
        #[test]
        fn throughput_is_non_negative(visits in 0..1000_u32, elapsed_ns in 0..10_000_000_000_u64) {
            let first_ns = 1_000_000_000_u64; // 1 second
            let last_ns = first_ns + elapsed_ns;
            let throughput = calc_throughput(visits, first_ns, last_ns);
            prop_assert!(throughput >= 0.0, "throughput was {throughput}");
        }

        /// Property: Zero visits means zero throughput
        #[test]
        fn zero_visits_zero_throughput(first_ns in 1..1000_000_000_u64, last_ns in 1..1000_000_000_u64) {
            let throughput = calc_throughput(0, first_ns, last_ns.max(first_ns + 1));
            prop_assert!((throughput - 0.0).abs() < f64::EPSILON);
        }

        /// Property: One visit means zero throughput
        #[test]
        fn one_visit_zero_throughput(first_ns in 1..1000_000_000_u64, last_ns in 1..1000_000_000_u64) {
            let throughput = calc_throughput(1, first_ns, last_ns.max(first_ns + 1));
            prop_assert!((throughput - 0.0).abs() < f64::EPSILON);
        }

        /// Property: More visits in same time = higher throughput
        #[test]
        fn more_visits_higher_throughput(
            visits1 in 2..500_u32,
            visits2 in 501..1000_u32,
            elapsed_ns in 1_000_000..1_000_000_000_u64  // 1ms to 1s
        ) {
            let first_ns = 1_000_000_000_u64;
            let last_ns = first_ns + elapsed_ns;
            let t1 = calc_throughput(visits1, first_ns, last_ns);
            let t2 = calc_throughput(visits2, first_ns, last_ns);
            prop_assert!(t2 > t1, "t2 ({t2}) should be > t1 ({t1})");
        }

        /// Property: Same visits in less time = higher throughput
        #[test]
        fn less_time_higher_throughput(
            visits in 10..1000_u32,
            elapsed1 in 500_000_000..1_000_000_000_u64,  // 0.5s to 1s
            elapsed2 in 100_000_000..499_999_999_u64,    // 0.1s to 0.5s
        ) {
            let first_ns = 1_000_000_000_u64;
            let t1 = calc_throughput(visits, first_ns, first_ns + elapsed1);
            let t2 = calc_throughput(visits, first_ns, first_ns + elapsed2);
            prop_assert!(t2 > t1, "t2 ({t2}) should be > t1 ({t1})");
        }

        /// Property: Throughput is finite (not NaN or Inf)
        #[test]
        fn throughput_is_finite(
            visits in 2..10000_u32,
            elapsed_ns in 1..10_000_000_000_u64
        ) {
            let first_ns = 1_000_000_000_u64;
            let last_ns = first_ns + elapsed_ns;
            let throughput = calc_throughput(visits, first_ns, last_ns);
            prop_assert!(throughput.is_finite(), "throughput was {throughput}");
        }

        /// Property: Visit count accumulates correctly
        #[test]
        fn visit_count_accumulates(n in 1..100_usize) {
            let p = Progress::new("test");
            for _ in 0..n {
                p.note_visit();
            }
            prop_assert_eq!(p.visit_count() as usize, n);
        }
    }
}
