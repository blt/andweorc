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
fn monotonic_nanos() -> u64 {
    let mut ts = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    // SAFETY: clock_gettime with CLOCK_MONOTONIC is async-signal-safe
    unsafe {
        libc::clock_gettime(libc::CLOCK_MONOTONIC, &raw mut ts);
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
        self.visits.fetch_add(1, Ordering::Relaxed);

        // Set first visit timestamp if this is the first visit
        // Use compare_exchange to atomically set only if still 0
        let _ = self
            .first_visit_ns
            .compare_exchange(0, now, Ordering::Relaxed, Ordering::Relaxed);

        // Always update last visit timestamp
        self.last_visit_ns.store(now, Ordering::Relaxed);
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
        let first = self.first_visit_ns.load(Ordering::Relaxed);
        let last = self.last_visit_ns.load(Ordering::Relaxed);
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
        let visits = self.visits.load(Ordering::Relaxed);
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
    pub fn reset(&self) {
        self.visits.store(0, Ordering::Relaxed);
        self.first_visit_ns.store(0, Ordering::Relaxed);
        self.last_visit_ns.store(0, Ordering::Relaxed);
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
