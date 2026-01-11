//! Per-thread profiler for collecting performance samples.
//!
//! Each thread participating in the experiment has a `PerThreadProfiler` that:
//! - Manages a perf event sampler for CPU cycles
//! - Collects instruction pointers and call chains

use perf_event::events::Software;
use perf_event::sample::{PerfRecord, PerfSampleType};
use perf_event::SampleStream;
use std::time::Duration;

/// Per-thread profiler that manages perf sampling.
///
/// Each OS thread has one of these, managed by the global `Experiment`.
pub(crate) struct PerThreadProfiler {
    /// Maximum samples to poll per signal handler invocation.
    total_samples: u16,

    /// The perf event sample stream for this thread.
    /// None if perf events are not available (permissions or kernel support).
    sampler: Option<SampleStream>,
}

impl PerThreadProfiler {
    /// Creates a new per-thread profiler.
    ///
    /// # Arguments
    ///
    /// * `total_samples` - Maximum samples to read per polling period.
    ///
    /// If perf events are not available (missing permissions or kernel support),
    /// the profiler will be created without sampling capability.
    pub(crate) fn new(total_samples: u16) -> Self {
        // Use SOFTWARE CPU_CLOCK for sampling - works in more environments than hardware counters
        let sampler_result = perf_event::Builder::new()
            .kind(Software::CPU_CLOCK)
            .sample_frequency(1000)
            .sample(PerfSampleType::IP)
            .sample(PerfSampleType::TID)
            .sample(PerfSampleType::TIME)
            .sample_stream();

        let sampler = match sampler_result {
            Ok(s) => {
                // Enable the perf event to start collecting samples
                if let Err(e) = s.enable() {
                    libc_print::libc_eprintln!("[andweorc] failed to enable perf sampling: {}", e);
                    None
                } else {
                    Some(s)
                }
            }
            Err(e) => {
                libc_print::libc_eprintln!("[andweorc] perf sampling not available: {}", e);
                libc_print::libc_eprintln!(
                    "[andweorc] Try: sudo sysctl kernel.perf_event_paranoid=-1"
                );
                None
            }
        };

        Self {
            total_samples,
            sampler,
        }
    }

    /// Processes any pending samples from the perf event stream.
    ///
    /// Called from the SIGPROF signal handler. Reads up to `total_samples`
    /// from the perf ring buffer and reports them to the experiment.
    ///
    /// # Safety
    ///
    /// This is called from a signal handler context. The current implementation
    /// may not be fully async-signal-safe due to potential allocations.
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
