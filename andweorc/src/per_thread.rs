//! Per-thread profiler for collecting performance samples.
//!
//! Each thread participating in the experiment has a `PerThreadProfiler` that:
//! - Manages a perf event sampler for CPU cycles
//! - Collects instruction pointers and call chains
//! - Injects delays for virtual speedup experiments

use crate::timer::{nanosleep, Interval, IntervalError};
use perf_event::events::Software;
use perf_event::sample::{PerfRecord, PerfSampleType};
use perf_event::SampleStream;
use std::time::Duration;

/// Default sampling interval for the profiler timer.
const DEFAULT_SAMPLE_INTERVAL: Duration = Duration::from_millis(10);

/// Per-thread profiler that manages perf sampling and delay injection.
///
/// Each OS thread has one of these, managed by the global `Experiment`.
pub(crate) struct PerThreadProfiler {
    /// Total number of delays this thread has observed.
    delay_count: u32,

    /// Maximum samples to poll per signal handler invocation.
    total_samples: u16,

    /// The perf event sample stream for this thread.
    /// None if perf events are not available (permissions or kernel support).
    sampler: Option<SampleStream>,

    /// The interval timer for this thread.
    /// None if profiling is not active.
    timer: Option<Interval>,
}

impl PerThreadProfiler {
    /// Creates a new per-thread profiler.
    ///
    /// # Arguments
    ///
    /// * `total_samples` - Maximum samples to read per polling period.
    ///
    /// If perf events are not available (missing permissions or kernel support),
    /// the profiler will be created without sampling capability. Delay injection
    /// will still work.
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
                    libc_print::libc_eprintln!(
                        "[andweorc] failed to enable perf sampling: {}",
                        e
                    );
                    None
                } else {
                    Some(s)
                }
            }
            Err(e) => {
                libc_print::libc_eprintln!(
                    "[andweorc] perf sampling not available: {}",
                    e
                );
                libc_print::libc_eprintln!(
                    "[andweorc] Try: sudo sysctl kernel.perf_event_paranoid=-1"
                );
                None
            }
        };

        Self {
            sampler,
            total_samples,
            delay_count: 0,
            timer: None,
        }
    }

    /// Returns true if perf sampling is available for this profiler.
    #[allow(dead_code)]
    pub(crate) fn has_sampler(&self) -> bool {
        self.sampler.is_some()
    }

    /// Returns true if profiling is currently active (timer is running).
    #[allow(dead_code)]
    pub(crate) fn is_profiling(&self) -> bool {
        self.timer.is_some()
    }

    /// Starts profiling for this thread.
    ///
    /// Creates an interval timer that delivers SIGPROF to this thread
    /// at the default sampling interval.
    ///
    /// # Errors
    ///
    /// Returns an error if the timer cannot be created or started.
    #[allow(dead_code)]
    pub(crate) fn start_profiling(&mut self) -> Result<(), IntervalError> {
        self.start_profiling_with_interval(DEFAULT_SAMPLE_INTERVAL)
    }

    /// Starts profiling for this thread with a custom interval.
    ///
    /// # Arguments
    ///
    /// * `interval` - How often to deliver SIGPROF signals.
    ///
    /// # Errors
    ///
    /// Returns an error if the timer cannot be created or started.
    #[allow(dead_code)]
    pub(crate) fn start_profiling_with_interval(
        &mut self,
        interval: Duration,
    ) -> Result<(), IntervalError> {
        // Get the current thread ID
        // gettid returns pid_t which fits in i32 on Linux
        #[allow(clippy::cast_possible_truncation)]
        let tid = unsafe { libc::syscall(libc::SYS_gettid) as libc::pid_t };

        // Create and start the timer
        let timer = Interval::new(tid, libc::SIGPROF)?;
        timer.start(interval)?;

        self.timer = Some(timer);
        Ok(())
    }

    /// Stops profiling for this thread.
    ///
    /// # Errors
    ///
    /// Returns an error if the timer cannot be stopped.
    #[allow(dead_code)]
    pub(crate) fn stop_profiling(&mut self) -> Result<(), IntervalError> {
        if let Some(timer) = self.timer.take() {
            timer.stop()?;
        }
        Ok(())
    }

    /// Injects a delay based on the current experiment state.
    ///
    /// Calculates the difference between global and local delay counts and
    /// sleeps proportionally to simulate virtual speedup of other code.
    #[allow(dead_code)]
    pub(crate) fn delay(&mut self) {
        let details = crate::experiment::get_instance().delay_details();
        let delay_diff = details.global_delay.saturating_sub(self.delay_count);

        if delay_diff > 0 {
            let duration = details.current_delay.mul_f64(f64::from(delay_diff));
            // Ignore errors for now - delay injection is best-effort
            let _ = nanosleep(duration);
        }

        self.delay_count = details.global_delay;
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
