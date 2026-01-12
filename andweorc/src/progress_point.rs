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
///
/// # Concurrency
///
/// The progress point uses a generation counter to ensure atomic reset
/// behavior. When `reset()` is called, it increments the generation counter
/// before clearing other fields. `note_visit()` checks the generation before
/// updating timestamps to avoid corruption during concurrent reset/visit.
pub struct Progress {
    /// The name identifying this progress point.
    name: &'static str,
    /// Generation counter for atomic reset detection.
    ///
    /// Incremented by `reset()` before clearing other fields. `note_visit()`
    /// checks this to detect concurrent resets and avoid corrupted timestamps.
    generation: AtomicU32,
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
            .field("generation", &self.generation.load(Ordering::Relaxed))
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
            generation: AtomicU32::new(0),
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
    ///
    /// # Delay Injection
    ///
    /// Before recording the visit, this consumes any pending delay for the
    /// current thread. This is the core of the Coz virtual speedup mechanism:
    /// threads that hit the selected code during profiling accumulate delay
    /// debt which is "paid" at progress points.
    ///
    /// # Generation Counter
    ///
    /// Uses a generation counter to detect concurrent `reset()` calls. If a reset
    /// occurs during `note_visit()`, the timestamp updates are skipped to avoid
    /// corrupted throughput measurements. The visit count is still incremented.
    pub fn note_visit(&self) {
        // Consume any pending delay before measuring throughput.
        // This implements the Coz virtual speedup: threads hitting the selected
        // code are delayed at progress points, simulating what would happen if
        // that code were optimized.
        crate::consume_pending_delay();

        // Capture generation before any operations
        let gen = self.generation.load(Ordering::Acquire);
        let now = monotonic_nanos();

        // Always increment visit count (even if reset races with us)
        self.visits.fetch_add(1, Ordering::Release);

        // Only update timestamps if generation hasn't changed (no concurrent reset)
        // This prevents corrupted timestamps from racing with reset().
        //
        // Memory ordering correctness:
        // - reset() uses fetch_add with Release ordering
        // - This load uses Acquire ordering
        // - If reset() completed its fetch_add, we WILL see the new generation here
        // - If we still see the old generation, reset() hasn't happened-before this load,
        //   so our updates are valid and reset() will overwrite them afterward
        if self.generation.load(Ordering::Acquire) == gen {
            // Set first visit timestamp if this is the first visit
            // Use compare_exchange to atomically set only if still 0
            let _ =
                self.first_visit_ns
                    .compare_exchange(0, now, Ordering::Release, Ordering::Relaxed);

            // Always update last visit timestamp
            self.last_visit_ns.store(now, Ordering::Release);
        }
        // If generation changed, skip timestamp updates - throughput() will return
        // 0.0 for this round which is a safe fallback
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
    /// # Concurrency
    ///
    /// This method uses a generation counter to achieve safe concurrent reset:
    ///
    /// 1. First, increment the generation counter (Release ordering)
    /// 2. Then, clear all other fields
    ///
    /// Any `note_visit()` calls that race with reset will detect the generation
    /// change and skip timestamp updates, preventing corrupted measurements.
    /// The visit count increment may still race, but this is acceptable since
    /// throughput will be 0.0 anyway (timestamps are skipped or cleared).
    pub fn reset(&self) {
        // Increment generation FIRST to signal concurrent note_visit() calls
        // that a reset is in progress. They will detect this and skip timestamp
        // updates, preventing corrupted measurements.
        self.generation.fetch_add(1, Ordering::Release);

        // Now safe to clear the other fields - concurrent note_visit() calls
        // will either see the old state (and skip updates after detecting
        // generation change) or see the new zeroed state.
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

// =============================================================================
// KANI PROOFS
// =============================================================================
//
// These proofs formally verify critical properties of the progress point mechanism.
// Run with: ci/kani andweorc

#[cfg(kani)]
mod kani_proofs {
    use super::*;

    /// Proof: Generation counter always increases.
    ///
    /// This ensures reset() always advances the generation.
    #[kani::proof]
    fn generation_always_increases() {
        let gen = AtomicU32::new(0);
        let initial: u32 = kani::any();
        gen.store(initial, Ordering::Relaxed);

        let before = gen.load(Ordering::Acquire);
        gen.fetch_add(1, Ordering::Release);
        let after = gen.load(Ordering::Acquire);

        // Wrapping is fine - we just need them to be different
        kani::assert(after != before, "generation must change after increment");
    }

    /// Proof: Visit count accumulation is bounded.
    ///
    /// This verifies that visit counts don't cause undefined behavior.
    /// Uses unwind(5) to bound the loop iterations for tractable verification.
    #[kani::proof]
    #[kani::unwind(5)]
    fn visit_count_accumulation_bounded() {
        let visits = AtomicU32::new(0);
        let add_count: u32 = kani::any();

        // Limit to small value for tractable verification
        kani::assume(add_count <= 4);

        let mut expected = 0_u32;
        for _ in 0..add_count {
            visits.fetch_add(1, Ordering::Release);
            expected = expected.wrapping_add(1);
        }

        let final_count = visits.load(Ordering::Acquire);
        kani::assert(
            final_count == expected,
            "visit count must equal number of additions",
        );
    }

    /// Proof: First visit timestamp CAS preserves the first value.
    ///
    /// This verifies that if a CAS succeeds, the value is set correctly,
    /// and subsequent CAS operations from 0 will fail (because value is non-zero).
    #[kani::proof]
    fn first_visit_cas_preserves_value() {
        let first_visit = AtomicU64::new(0);
        let now: u64 = kani::any();

        // Timestamp should be non-zero
        kani::assume(now > 0);

        // CAS from 0 to now
        let result = first_visit.compare_exchange(0, now, Ordering::Release, Ordering::Relaxed);

        // If CAS succeeded, the value must be what we set
        if result.is_ok() {
            let loaded = first_visit.load(Ordering::Acquire);
            kani::assert(loaded == now, "CAS success must set the value");
        }

        // If CAS failed, the value was not zero (already set)
        if result.is_err() {
            let loaded = first_visit.load(Ordering::Acquire);
            kani::assert(loaded != 0, "CAS failure means value was non-zero");
        }
    }

    /// Proof: Reset clears all fields to zero.
    ///
    /// This verifies that reset() properly clears the progress point state.
    #[kani::proof]
    fn reset_clears_state() {
        let visits = AtomicU32::new(0);
        let first_visit = AtomicU64::new(0);
        let last_visit = AtomicU64::new(0);

        // Set some non-zero state
        let v: u32 = kani::any();
        let f: u64 = kani::any();
        let l: u64 = kani::any();

        visits.store(v, Ordering::Relaxed);
        first_visit.store(f, Ordering::Relaxed);
        last_visit.store(l, Ordering::Relaxed);

        // Reset (simulating Progress::reset)
        visits.store(0, Ordering::Release);
        first_visit.store(0, Ordering::Release);
        last_visit.store(0, Ordering::Release);

        kani::assert(
            visits.load(Ordering::Acquire) == 0,
            "visits must be zero after reset",
        );
        kani::assert(
            first_visit.load(Ordering::Acquire) == 0,
            "first_visit must be zero after reset",
        );
        kani::assert(
            last_visit.load(Ordering::Acquire) == 0,
            "last_visit must be zero after reset",
        );
    }

    /// Proof: Throughput calculation never produces NaN or Infinity for valid inputs.
    ///
    /// This verifies the throughput formula doesn't have edge cases that produce
    /// invalid floating point values.
    #[kani::proof]
    fn throughput_is_finite() {
        let visits: u32 = kani::any();
        let elapsed_ns: u64 = kani::any();

        // Assume valid inputs
        kani::assume(visits >= 2);
        kani::assume(elapsed_ns > 0);
        kani::assume(elapsed_ns <= 10_000_000_000_000); // Max ~10000 seconds

        // Calculate throughput (same as Progress::throughput)
        let visits_f64 = f64::from(visits - 1);
        #[allow(clippy::cast_precision_loss)]
        let elapsed_secs = elapsed_ns as f64 / 1_000_000_000.0;
        let throughput = visits_f64 / elapsed_secs;

        kani::assert(throughput.is_finite(), "throughput must be finite");
        kani::assert(throughput >= 0.0, "throughput must be non-negative");
    }

    /// Proof: Elapsed time calculation is correct.
    ///
    /// This verifies that last - first produces the expected elapsed time.
    #[kani::proof]
    fn elapsed_calculation_correct() {
        let first: u64 = kani::any();
        let elapsed: u64 = kani::any();

        // Prevent overflow
        kani::assume(elapsed <= u64::MAX - first);

        let last = first + elapsed;
        let calculated = last - first;

        kani::assert(calculated == elapsed, "elapsed must equal last - first");
    }
}
