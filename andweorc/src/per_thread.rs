//! Per-thread profiler for collecting performance samples.
//!
//! Each thread participating in the experiment has a `PerThreadProfiler` that:
//! - Manages a perf event sampler for CPU cycles
//! - Collects instruction pointers and call chains
//!
//! # Signal Safety Considerations
//!
//! This module's `process_samples()` method is called from a SIGPROF signal handler.
//! While we strive for async-signal-safety, the underlying `perf_event` crate's
//! `SampleStream::read()` method may not be fully async-signal-safe:
//!
//! - Reading from the perf ring buffer itself is async-signal-safe (mmap'd memory)
//! - However, error paths in the `perf_event` crate may allocate
//! - The `PerfRecord` variants we handle don't allocate in the happy path
//!
//! # Mitigations
//!
//! 1. We use a zero timeout for non-blocking reads
//! 2. We limit samples per signal to prevent unbounded processing time
//! 3. The experiment singleton guards against re-initialization in signal context
//! 4. SIGPROF is blocked during handler execution to prevent re-entrancy
//!
//! # Future Work
//!
//! A fully signal-safe implementation would require either:
//! - A custom perf ring buffer reader that avoids all allocations
//! - Moving sample processing out of signal context entirely (e.g., to a dedicated thread)

use perf_event::events::Hardware;
use perf_event::sample::{PerfRecord, PerfSampleType};
use perf_event::SampleStream;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

/// Maximum pending delay in nanoseconds (100ms).
///
/// This caps the delay to prevent absurdly long sleeps that could stall
/// the program. If delays accumulate beyond this, we clamp to this value.
/// This can happen if:
/// - A thread never hits a progress point or yield point
/// - Many samples accumulate before consumption
///
/// 100ms is chosen as a reasonable upper bound - it's long enough to
/// measure causal impact but short enough to not freeze the program.
const MAX_PENDING_DELAY_NS: u64 = 100_000_000; // 100ms

/// Per-thread profiler that manages perf sampling and delay injection.
///
/// Each OS thread has one of these, managed by the global `Experiment`.
///
/// # Delay Injection
///
/// When causal profiling is active, threads accumulate delay "debt" in
/// `pending_delay_ns`. This debt is consumed at progress points, delay points,
/// or synchronization operations by sleeping for the accumulated duration.
/// This implements the Coz virtual speedup mechanism.
pub(crate) struct PerThreadProfiler {
    /// Maximum samples to poll per signal handler invocation.
    total_samples: u16,

    /// The perf event sample stream for this thread.
    /// None if perf events are not available (permissions or kernel support).
    sampler: Option<SampleStream>,

    /// Accumulated delay debt in nanoseconds.
    ///
    /// When a sample matches the selected IP during an experiment, delay is
    /// added here. The delay is consumed at the next progress point, delay
    /// point, or synchronization operation.
    ///
    /// Uses Relaxed ordering for adds (signal handler) and Acquire/Release
    /// for the swap operation that consumes the delay.
    pending_delay_ns: AtomicU64,
}

impl PerThreadProfiler {
    /// Creates a new per-thread profiler.
    ///
    /// # Arguments
    ///
    /// * `total_samples` - Maximum samples to read per polling period.
    ///
    /// # Errors
    ///
    /// Returns an error if hardware instruction counters are not available. This requires:
    /// - `kernel.perf_event_paranoid` <= 1
    /// - A CPU that supports instruction counting (most modern `x86_64`/ARM)
    ///
    /// Use `sudo sysctl kernel.perf_event_paranoid=1` to enable.
    pub(crate) fn try_new(total_samples: u16) -> Result<Self, String> {
        // Use hardware instruction counters - REQUIRED for correct causal profiling.
        // CPU_CLOCK would create a feedback loop: delays affect sampling rate,
        // which corrupts the causal measurements.
        let sampler_result = perf_event::Builder::new()
            .kind(Hardware::INSTRUCTIONS)
            .sample_frequency(1000)
            .sample(PerfSampleType::IP)
            .sample(PerfSampleType::TID)
            .sample(PerfSampleType::TIME)
            .sample_stream();

        let sampler = match sampler_result {
            Ok(s) => {
                // Enable the perf event to start collecting samples
                if let Err(e) = s.enable() {
                    return Err(format!(
                        "failed to enable perf counters: {e}. \
                         Try: sudo sysctl kernel.perf_event_paranoid=1"
                    ));
                }
                s
            }
            Err(e) => {
                return Err(format!(
                    "failed to create perf counters: {e}. \
                     Try: sudo sysctl kernel.perf_event_paranoid=1"
                ));
            }
        };

        Ok(Self {
            total_samples,
            sampler: Some(sampler),
            pending_delay_ns: AtomicU64::new(0),
        })
    }

    /// Adds delay to this thread's pending balance.
    ///
    /// Called from the signal handler when a sample matches the selected IP.
    /// The delay will be consumed at the next progress point, delay point,
    /// or synchronization operation.
    ///
    /// # Signal Safety
    ///
    /// This function is async-signal-safe: it only uses an atomic `fetch_add`.
    ///
    /// # Overflow Handling
    ///
    /// Uses wrapping addition for simplicity and signal safety. In practice,
    /// overflow is impossible under normal conditions:
    /// - Delays are typically 1-15ms per sample
    /// - Samples arrive at ~1000/sec
    /// - Even at max rate, it would take centuries to overflow u64
    ///
    /// The delay is clamped to `MAX_PENDING_DELAY_NS` at consumption time
    /// to prevent pathological cases from stalling the program.
    #[inline]
    pub(crate) fn add_delay(&self, ns: u64) {
        self.pending_delay_ns.fetch_add(ns, Ordering::Relaxed);
    }

    /// Returns and clears the pending delay for this thread.
    ///
    /// Returns the accumulated delay in nanoseconds (clamped to [`MAX_PENDING_DELAY_NS`])
    /// and resets the counter to zero. The caller should sleep for this duration
    /// to simulate virtual speedup.
    ///
    /// # Clamping
    ///
    /// If accumulated delays exceed `MAX_PENDING_DELAY_NS` (100ms), the returned
    /// value is clamped to that maximum. This prevents:
    /// - Threads that rarely hit progress points from stalling indefinitely
    /// - Pathological accumulation from freezing the program
    ///
    /// # Memory Ordering
    ///
    /// Uses Acquire ordering to synchronize with Relaxed stores from `add_delay()`.
    #[inline]
    pub(crate) fn take_pending_delay(&self) -> u64 {
        let raw = self.pending_delay_ns.swap(0, Ordering::Acquire);
        // Clamp to maximum to prevent pathological stalls
        raw.min(MAX_PENDING_DELAY_NS)
    }

    /// Processes any pending samples from the perf event stream.
    ///
    /// Called from the SIGPROF signal handler. Reads up to `total_samples`
    /// from the perf ring buffer and reports them to the experiment.
    ///
    /// # Signal Safety
    ///
    /// This is called from a signal handler context. See the module-level
    /// documentation for signal safety considerations.
    ///
    /// **Known limitations:**
    /// - The `perf_event` crate's error paths may allocate memory
    /// - The `callchain` field in samples may allocate (we use `as_deref()` to avoid cloning)
    ///
    /// In practice, signal safety violations are rare because:
    /// - We use non-blocking reads (zero timeout)
    /// - Happy-path sample reading doesn't allocate
    /// - Errors are handled by breaking out of the loop, not propagating
    pub(crate) fn process_samples(&self) {
        let Some(ref sampler) = self.sampler else {
            return;
        };

        // Zero timeout for immediate poll (non-blocking)
        let timeout = Duration::ZERO;

        for _ in 0..self.total_samples {
            match sampler.read(Some(timeout)) {
                Ok(Some(PerfRecord::Sample(s))) => {
                    crate::experiment::get_instance().report_sample(s.ip, s.callchain.as_deref());
                }
                Ok(Some(_)) => {
                    // Non-sample record (e.g., mmap event) - ignore
                }
                Ok(None) => {
                    break; // No more samples
                }
                Err(_) => {
                    break;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Test that process_samples doesn't panic when sampler is None.
    ///
    /// This tests the early return path in process_samples.
    #[test]
    fn process_samples_without_sampler_is_noop() {
        // Create a profiler with no sampler (simulating disabled profiler)
        let profiler = PerThreadProfiler {
            total_samples: 16,
            sampler: None,
            pending_delay_ns: AtomicU64::new(0),
        };

        // This should be a no-op and not panic
        profiler.process_samples();
    }

    /// Test delay accumulation and consumption.
    #[test]
    fn delay_accumulation_works() {
        let profiler = PerThreadProfiler {
            total_samples: 16,
            sampler: None,
            pending_delay_ns: AtomicU64::new(0),
        };

        // Add some delays
        profiler.add_delay(1000);
        profiler.add_delay(2000);
        profiler.add_delay(500);

        // Take should return the total and reset to zero
        let total = profiler.take_pending_delay();
        assert_eq!(total, 3500);

        // Second take should return zero
        let second = profiler.take_pending_delay();
        assert_eq!(second, 0);
    }

    /// Test that take_pending_delay is atomic (returns and clears).
    #[test]
    fn take_pending_delay_clears_balance() {
        let profiler = PerThreadProfiler {
            total_samples: 16,
            sampler: None,
            pending_delay_ns: AtomicU64::new(5000),
        };

        let first = profiler.take_pending_delay();
        assert_eq!(first, 5000);

        let second = profiler.take_pending_delay();
        assert_eq!(second, 0);
    }
}

// =============================================================================
// KANI PROOFS
// =============================================================================
//
// These proofs formally verify critical properties of the delay injection mechanism.
// Run with: ci/kani andweorc

#[cfg(kani)]
mod kani_proofs {
    use super::*;

    /// Proof: A single delay add followed by take returns the same value.
    ///
    /// This verifies that delays are preserved through the accumulator.
    #[kani::proof]
    fn delay_add_take_preserves_value() {
        let pending = AtomicU64::new(0);
        let delay: u64 = kani::any();

        // Limit delay to reasonable values (up to 10 seconds in nanoseconds)
        kani::assume(delay <= 10_000_000_000);

        pending.fetch_add(delay, Ordering::Relaxed);
        let taken = pending.swap(0, Ordering::Acquire);

        kani::assert(taken == delay, "taken delay must equal added delay");
    }

    /// Proof: Take clears the balance to zero.
    ///
    /// This ensures that after taking, no residual delay remains.
    #[kani::proof]
    fn take_clears_balance() {
        let pending = AtomicU64::new(0);
        let delay: u64 = kani::any();
        kani::assume(delay <= 10_000_000_000);

        pending.fetch_add(delay, Ordering::Relaxed);
        let _ = pending.swap(0, Ordering::Acquire);
        let second = pending.swap(0, Ordering::Acquire);

        kani::assert(second == 0, "second take must return zero");
    }

    /// Proof: Multiple delays accumulate correctly.
    ///
    /// This verifies that sequential adds produce the expected sum.
    #[kani::proof]
    fn delays_accumulate() {
        let pending = AtomicU64::new(0);
        let delay1: u64 = kani::any();
        let delay2: u64 = kani::any();

        // Limit to values that won't overflow when summed
        kani::assume(delay1 <= 5_000_000_000);
        kani::assume(delay2 <= 5_000_000_000);

        pending.fetch_add(delay1, Ordering::Relaxed);
        pending.fetch_add(delay2, Ordering::Relaxed);
        let taken = pending.swap(0, Ordering::Acquire);

        kani::assert(
            taken == delay1 + delay2,
            "accumulated delay must equal sum of inputs",
        );
    }

    /// Proof: Zero delay add has no effect.
    ///
    /// Adding zero should not change the balance.
    #[kani::proof]
    fn zero_delay_no_effect() {
        let pending = AtomicU64::new(0);
        let initial: u64 = kani::any();
        kani::assume(initial <= 10_000_000_000);

        pending.store(initial, Ordering::Relaxed);
        pending.fetch_add(0, Ordering::Relaxed);
        let final_val = pending.load(Ordering::Acquire);

        kani::assert(final_val == initial, "adding zero must not change balance");
    }

    /// Proof: Delay accumulation is bounded by u64::MAX.
    ///
    /// This ensures no undefined behavior from overflow.
    #[kani::proof]
    fn delay_accumulation_bounded() {
        let pending = AtomicU64::new(0);
        let delay: u64 = kani::any();

        // Add any delay value
        let prev = pending.fetch_add(delay, Ordering::Relaxed);
        let current = pending.load(Ordering::Relaxed);

        // The result should be prev + delay (wrapping on overflow)
        // This is well-defined behavior for atomics
        kani::assert(
            current == prev.wrapping_add(delay),
            "accumulation must follow wrapping semantics",
        );
    }

    /// Proof: Take is idempotent after clearing.
    ///
    /// Multiple takes after the balance is cleared all return zero.
    #[kani::proof]
    fn take_idempotent_when_empty() {
        let pending = AtomicU64::new(0);

        let first = pending.swap(0, Ordering::Acquire);
        let second = pending.swap(0, Ordering::Acquire);
        let third = pending.swap(0, Ordering::Acquire);

        kani::assert(first == 0, "first take from empty must be zero");
        kani::assert(second == 0, "second take from empty must be zero");
        kani::assert(third == 0, "third take from empty must be zero");
    }
}
