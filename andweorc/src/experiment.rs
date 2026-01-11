//! Central coordinator for causal profiling experiments.
//!
//! The `Experiment` singleton manages global state across all threads including:
//! - Signal handler registration for periodic sampling
//! - Per-thread profiler state
//! - Progress point tracking
//! - Delay table for virtual speedup experiments

use crate::per_thread::PerThreadProfiler;
use crate::progress_point::Progress;
use dashmap::DashMap;
use libc::{c_int, c_void};
use nix::sys::pthread::{pthread_self, Pthread};
use nix::sys::signal::{signal, SigHandler, Signal};
use rand::rngs::SmallRng;
use rand::SeedableRng;
use std::sync::atomic::{AtomicBool, AtomicPtr, AtomicU32, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::Duration;

/// Percentage multipliers for the delay table.
///
/// The first 25 entries are 0% (baseline measurements), followed by
/// increasing percentages from 5% to 150% of the baseline delay.
const DELAY_PRCNT: [f64; 55] = [
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55,
    0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15, 1.20, 1.25, 1.30, 1.35,
    1.40, 1.45, 1.50,
];

/// The baseline delay duration used to compute actual delays.
const DELAY_BASELINE: Duration = Duration::from_millis(10);

/// Flag indicating whether the experiment singleton is fully initialized.
/// This is used by the signal handler to avoid accessing uninitialized data.
static EXPERIMENT_INITIALIZED: AtomicBool = AtomicBool::new(false);

/// Signal handler that processes perf samples when SIGPROF fires.
///
/// This handler is called on whichever thread receives the signal. It looks up
/// the thread's profiler and triggers sample processing.
///
/// # Safety
///
/// This is a signal handler and must only call async-signal-safe functions.
/// The current implementation may violate this by acquiring locks in `DashMap`.
/// TODO: Make this fully signal-safe.
extern "C" fn process_samples(_signal: c_int) {
    // Block SIGPROF during handler to prevent re-entrancy
    // This prevents the race condition where a nested signal handler
    // drains the perf buffer before the original handler reads it.
    let mut oldset: libc::sigset_t = unsafe { std::mem::zeroed() };
    let mut newset: libc::sigset_t = unsafe { std::mem::zeroed() };
    unsafe {
        libc::sigemptyset(&raw mut newset);
        libc::sigaddset(&raw mut newset, libc::SIGPROF);
        libc::pthread_sigmask(libc::SIG_BLOCK, &raw const newset, &raw mut oldset);
    }

    // Check if experiment is initialized before accessing it.
    // This avoids calling Once::call_once from signal handler context.
    if !EXPERIMENT_INITIALIZED.load(Ordering::Acquire) {
        unsafe {
            libc::pthread_sigmask(libc::SIG_SETMASK, &raw const oldset, std::ptr::null_mut());
        }
        return;
    }

    // SAFETY: We've confirmed the experiment is initialized via the atomic flag.
    // The EXPERIMENT static is guaranteed to be valid after EXPERIMENT_INITIALIZED is set.
    // OnceLock::get() is safe to call here since we've already checked the flag.
    let Some(experiment) = EXPERIMENT.get() else {
        unsafe {
            libc::pthread_sigmask(libc::SIG_SETMASK, &raw const oldset, std::ptr::null_mut());
        }
        return;
    };
    let tid = pthread_self();
    let ts = experiment.thread_state(tid);
    ts.process_samples();

    // Restore original signal mask
    unsafe {
        libc::pthread_sigmask(libc::SIG_SETMASK, &raw const oldset, std::ptr::null_mut());
    }
}

static EXPERIMENT: OnceLock<Experiment> = OnceLock::new();

/// Returns a reference to the global experiment singleton.
///
/// Initializes the experiment on first call, including registering the
/// SIGPROF signal handler.
pub(crate) fn get_instance() -> &'static Experiment {
    EXPERIMENT.get_or_init(|| {
        let exp = Experiment::new(DELAY_BASELINE);
        // Mark as initialized AFTER the experiment is fully constructed
        EXPERIMENT_INITIALIZED.store(true, Ordering::Release);
        exp
    })
}

/// Central coordinator for causal profiling experiments.
///
/// Manages all threads participating in the experiment, tracks progress points,
/// and coordinates delay injection for virtual speedup measurements.
pub(crate) struct Experiment {
    /// The signal handler registered for SIGPROF.
    #[allow(dead_code)]
    timer_signal_handle: SigHandler,

    /// Per-thread profiler state, keyed by pthread ID.
    thread_states: DashMap<Pthread, Arc<PerThreadProfiler>>,

    /// Named progress points for throughput measurement.
    progress_points: DashMap<&'static str, Arc<Progress>>,

    /// Sample counts by instruction pointer (for finding hot code).
    sample_counts: DashMap<usize, AtomicU64>,

    /// Pre-computed delay durations for each speedup percentage.
    delay_table: Vec<Duration>,

    /// Random number generator for experiment scheduling.
    #[allow(dead_code)]
    rng: SmallRng,

    /// Total number of delays issued across all threads.
    /// Incremented when the selected line is sampled.
    global_delay_count: AtomicU32,

    /// Current index into the delay table.
    delay_index: AtomicUsize,

    /// The instruction pointer currently being virtually sped up.
    /// When null, no virtual speedup is active (baseline measurement).
    selected_ip: AtomicPtr<c_void>,

    /// Total samples collected (for statistics).
    total_samples: AtomicU64,

    /// Samples matching the selected IP.
    selected_samples: AtomicU64,

    /// Whether an experiment round is currently active.
    is_active: AtomicBool,
}

/// Information about the current delay configuration.
#[derive(Debug, Clone, Copy)]
pub(crate) struct DelayDetails {
    /// The global delay count at the time of query.
    pub global_delay: u32,
    /// The current delay duration to apply.
    pub current_delay: Duration,
}

/// Statistics about the current experiment.
#[allow(dead_code)]
#[derive(Debug, Clone, Copy)]
pub(crate) struct ExperimentStats {
    /// Total samples collected.
    pub total_samples: u64,
    /// Samples matching the selected IP.
    pub selected_samples: u64,
    /// Whether an experiment is active.
    pub is_active: bool,
    /// Current speedup percentage index.
    pub speedup_index: usize,
}

impl Experiment {
    /// Creates a new experiment with the given baseline delay duration.
    ///
    /// # Panics
    ///
    /// Panics if the SIGPROF signal handler cannot be registered. This is
    /// considered a fatal error as the profiler cannot function without it.
    fn new(delay_baseline: Duration) -> Self {
        let handler = SigHandler::Handler(process_samples);

        // Register the signal handler for SIGPROF.
        // SAFETY: The handler function is valid for the lifetime of the program.
        // TODO: Implement Drop to restore the previous handler.
        let result = unsafe { signal(Signal::SIGPROF, handler) };
        match result {
            Ok(_) => {}
            Err(e) => {
                // This is a fatal error - we cannot profile without the signal handler
                panic!("failed to register SIGPROF handler: {e}");
            }
        }

        Self {
            timer_signal_handle: handler,
            thread_states: DashMap::new(),
            progress_points: DashMap::new(),
            sample_counts: DashMap::new(),
            global_delay_count: AtomicU32::new(0),
            delay_index: AtomicUsize::new(0),
            selected_ip: AtomicPtr::new(std::ptr::null_mut()),
            total_samples: AtomicU64::new(0),
            selected_samples: AtomicU64::new(0),
            is_active: AtomicBool::new(false),
            // TODO: Use a better seed (e.g., from /dev/urandom)
            rng: SmallRng::seed_from_u64(123_456_789),
            delay_table: DELAY_PRCNT
                .iter()
                .map(|prct| delay_baseline.mul_f64(*prct))
                .collect(),
        }
    }

    /// Returns the current delay configuration.
    pub(crate) fn delay_details(&self) -> DelayDetails {
        let global_delay = self.global_delay_count.load(Ordering::Relaxed);
        let current_delay = self.delay_table[self.delay_index.load(Ordering::Relaxed)];
        DelayDetails {
            global_delay,
            current_delay,
        }
    }

    /// Returns statistics about the current experiment.
    #[allow(dead_code)]
    pub(crate) fn stats(&self) -> ExperimentStats {
        ExperimentStats {
            total_samples: self.total_samples.load(Ordering::Relaxed),
            selected_samples: self.selected_samples.load(Ordering::Relaxed),
            is_active: self.is_active.load(Ordering::Relaxed),
            speedup_index: self.delay_index.load(Ordering::Relaxed),
        }
    }

    /// Starts a new experiment round with the given IP and speedup index.
    ///
    /// # Arguments
    ///
    /// * `ip` - The instruction pointer to virtually speed up.
    /// * `speedup_index` - Index into the delay table for speedup percentage.
    #[allow(dead_code)]
    pub(crate) fn start_experiment(&self, ip: *const c_void, speedup_index: usize) {
        // Reset counters
        self.total_samples.store(0, Ordering::Relaxed);
        self.selected_samples.store(0, Ordering::Relaxed);
        self.global_delay_count.store(0, Ordering::Relaxed);

        // Set the selected IP and speedup
        self.delay_index.store(speedup_index, Ordering::Relaxed);
        self.selected_ip.store(ip.cast_mut(), Ordering::Release);
        self.is_active.store(true, Ordering::Release);
    }

    /// Stops the current experiment round.
    #[allow(dead_code)]
    pub(crate) fn stop_experiment(&self) {
        self.is_active.store(false, Ordering::Release);
        self.selected_ip
            .store(std::ptr::null_mut(), Ordering::Release);
    }

    /// Registers a thread with the experiment, creating a profiler for it.
    ///
    /// Returns an Arc to the thread's profiler for external use.
    #[allow(dead_code)]
    pub(crate) fn register_thread(&self, thread_id: Pthread) -> Arc<PerThreadProfiler> {
        let refmut = self
            .thread_states
            .entry(thread_id)
            .or_insert_with(|| Arc::new(PerThreadProfiler::new(100)));
        Arc::clone(refmut.value())
    }

    /// Returns the profiler for the given thread, creating one if necessary.
    pub(crate) fn thread_state(&self, thread_id: Pthread) -> Arc<PerThreadProfiler> {
        let refmut = self
            .thread_states
            .entry(thread_id)
            .or_insert_with(|| Arc::new(PerThreadProfiler::new(100)));
        Arc::clone(refmut.value())
    }

    /// Returns the progress point with the given name, creating one if necessary.
    pub(crate) fn progress(&self, name: &'static str) -> Arc<Progress> {
        let refmut = self
            .progress_points
            .entry(name)
            .or_insert_with(|| Arc::new(Progress::new(name)));
        Arc::clone(refmut.value())
    }

    /// Records a sample from a thread's profiler.
    ///
    /// This is called from the signal handler context with the instruction
    /// pointer and optional call chain from the perf sample.
    ///
    /// If an experiment is active and the sample IP matches the selected IP,
    /// the global delay count is incremented, causing other threads to delay.
    pub(crate) fn report_sample(
        &self,
        ip: Option<*const c_void>,
        _callchain: Option<&[*const c_void]>,
    ) {
        // Always count total samples
        self.total_samples.fetch_add(1, Ordering::Relaxed);

        // Track sample counts by IP for finding hot code
        if let Some(sample_ip) = ip {
            let ip_usize = sample_ip as usize;
            self.sample_counts
                .entry(ip_usize)
                .or_insert_with(|| AtomicU64::new(0))
                .fetch_add(1, Ordering::Relaxed);
        }

        // Check if we have an experiment running
        if !self.is_active.load(Ordering::Acquire) {
            return;
        }

        let selected = self.selected_ip.load(Ordering::Acquire);
        if selected.is_null() {
            // Baseline measurement (no speedup)
            return;
        }

        // Check if this sample matches the selected IP
        if let Some(sample_ip) = ip {
            if sample_ip == selected {
                // This sample is at the selected line - increment delay count
                // to trigger delays in other threads (virtual speedup)
                self.selected_samples.fetch_add(1, Ordering::Relaxed);
                self.global_delay_count.fetch_add(1, Ordering::Relaxed);
            }
        }
    }

    /// Returns the top N most frequently sampled IPs.
    pub(crate) fn top_ips(&self, n: usize) -> Vec<usize> {
        let mut counts: Vec<(usize, u64)> = self
            .sample_counts
            .iter()
            .map(|entry| (*entry.key(), entry.value().load(Ordering::Relaxed)))
            .collect();

        counts.sort_by(|a, b| b.1.cmp(&a.1));
        counts.into_iter().take(n).map(|(ip, _)| ip).collect()
    }
}
