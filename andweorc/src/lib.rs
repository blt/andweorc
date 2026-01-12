//! Andweorc: A causal profiler for Rust programs.
//!
//! This crate implements causal profiling using the Coz approach: insert virtual
//! speedups via delays to non-sampled threads, measure throughput change, and
//! identify which source lines have causal impact on performance.
//!
//! # Overview
//!
//! Causal profiling answers the question "if I optimize this code, how much will
//! my program speed up?" rather than just "where does my program spend time?"
//!
//! # Usage
//!
//! Mark progress points in your code using the [`progress!`] macro:
//!
//! ```ignore
//! use andweorc::progress;
//!
//! fn process_request() {
//!     // ... do work ...
//!     progress!("request_complete");
//! }
//! ```
//!
//! Or use the `#[profile]` attribute to automatically add a progress point:
//!
//! ```ignore
//! use andweorc::profile;
//!
//! #[profile]
//! fn process_request() {
//!     // ... do work ...
//! }
//! ```

pub mod experiment;
mod ffi;
pub mod json_output;
mod lock_util;
mod per_thread;
mod posix;
pub mod progress_point;
pub mod runner;
mod timer;
pub mod validate;

use per_thread::PerThreadProfiler;
use std::cell::RefCell;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

/// Global flag indicating whether profiling is currently active.
/// Used to short-circuit delay consumption checks when profiling is disabled.
static PROFILING_ACTIVE: AtomicBool = AtomicBool::new(false);

/// Errors that can occur during profiling operations.
///
/// This enum provides structured error information that allows users to
/// programmatically handle different failure modes.
#[derive(Debug, Clone)]
pub enum ProfilingError {
    /// Profiling is already active for the current thread.
    ///
    /// Call `stop_profiling()` first before starting again.
    AlreadyActive,

    /// Failed to get the kernel thread ID.
    ///
    /// This should never happen on Linux systems but is handled gracefully.
    ThreadIdResolution,

    /// Failed to create the interval timer.
    ///
    /// This typically indicates resource limits (check `ulimit -a`) or
    /// insufficient permissions.
    TimerCreation(String),

    /// Failed to start the interval timer.
    ///
    /// The timer was created but could not be armed.
    TimerStart(String),

    /// Failed to initialize the experiment singleton.
    ///
    /// This typically means the SIGPROF signal handler could not be registered.
    ExperimentInit(String),

    /// Failed to create hardware performance counters.
    ///
    /// This typically indicates:
    /// - `kernel.perf_event_paranoid` is too restrictive (> 1)
    /// - Hardware counters are not available on this CPU
    /// - Insufficient permissions
    ///
    /// Try: `sudo sysctl kernel.perf_event_paranoid=1`
    PerfCounterCreation(String),
}

impl std::fmt::Display for ProfilingError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::AlreadyActive => write!(
                f,
                "profiling already active for this thread; call stop_profiling() first"
            ),
            Self::ThreadIdResolution => write!(f, "failed to get kernel thread ID (gettid)"),
            Self::TimerCreation(msg) => write!(f, "failed to create timer: {msg}"),
            Self::TimerStart(msg) => write!(f, "failed to start timer: {msg}"),
            Self::ExperimentInit(msg) => write!(f, "failed to initialize experiment: {msg}"),
            Self::PerfCounterCreation(msg) => {
                write!(f, "failed to create hardware performance counters: {msg}")
            }
        }
    }
}

impl std::error::Error for ProfilingError {}

// Thread-local storage for the profiling timer.
// Each thread has its own timer that delivers SIGPROF signals.
thread_local! {
    static PROFILING_TIMER: RefCell<Option<timer::Interval>> = const { RefCell::new(None) };
}

/// Resolves an instruction pointer to a source location string.
fn resolve_symbol(ip: usize) -> String {
    if ip == 0 {
        return "<baseline>".to_string();
    }

    let mut result = format!("0x{ip:x}");

    // Use backtrace to resolve the symbol
    backtrace::resolve(ip as *mut std::ffi::c_void, |symbol| {
        if let Some(name) = symbol.name() {
            result = format!("{name}");
        }
        if let Some(filename) = symbol.filename() {
            if let Some(lineno) = symbol.lineno() {
                result = format!("{}:{lineno}", filename.display());
            }
        }
    });

    result
}

/// Starts profiling for the current thread.
///
/// This creates an interval timer that delivers SIGPROF signals to the current
/// thread at regular intervals. The signal handler will collect perf samples
/// and report them to the experiment.
///
/// # Errors
///
/// Returns a [`ProfilingError`] if:
/// - Profiling is already active for this thread
/// - The timer cannot be created or started
/// - The experiment singleton failed to initialize
pub fn start_profiling() -> Result<(), ProfilingError> {
    start_profiling_with_interval(Duration::from_millis(10))
}

/// Starts profiling for the current thread with a custom interval.
///
/// # Arguments
///
/// * `interval` - How often to deliver SIGPROF signals and collect samples.
///
/// # Errors
///
/// Returns a [`ProfilingError`] if:
/// - Profiling is already active for this thread ([`ProfilingError::AlreadyActive`])
/// - The kernel thread ID cannot be resolved ([`ProfilingError::ThreadIdResolution`])
/// - The timer cannot be created ([`ProfilingError::TimerCreation`])
/// - The timer cannot be started ([`ProfilingError::TimerStart`])
pub fn start_profiling_with_interval(interval: Duration) -> Result<(), ProfilingError> {
    // Check if profiling is already active for this thread
    let already_profiling = PROFILING_TIMER.with(|cell| cell.borrow().is_some());
    if already_profiling {
        return Err(ProfilingError::AlreadyActive);
    }

    // Ensure the experiment singleton is initialized (this registers the signal handler)
    let _ = experiment::get_instance();

    // IMPORTANT: Create the per-thread profiler BEFORE starting the timer.
    // This avoids deadlock where the signal handler tries to access the profiler
    // while we're in the middle of creating it. The profiler is stored in
    // thread-local storage and accessed from the signal handler without locks.
    let profiler = PerThreadProfiler::try_new(16).map_err(ProfilingError::PerfCounterCreation)?;
    experiment::set_thread_profiler(Arc::new(profiler));

    // Get the current thread ID (Linux specific)
    // SAFETY: gettid is a simple syscall that requires no special privileges
    let kernel_tid_raw = unsafe { libc::syscall(libc::SYS_gettid) };

    // gettid returns -1 on error (though this should never happen for gettid)
    if kernel_tid_raw < 0 {
        return Err(ProfilingError::ThreadIdResolution);
    }

    // gettid returns pid_t which fits in i32 on Linux
    #[allow(clippy::cast_possible_truncation)]
    let kernel_tid = kernel_tid_raw as libc::pid_t;

    let timer = timer::Interval::new(kernel_tid, libc::SIGPROF)
        .map_err(|e| ProfilingError::TimerCreation(e.to_string()))?;
    timer
        .start(interval)
        .map_err(|e| ProfilingError::TimerStart(e.to_string()))?;

    // Store the timer in thread-local storage
    // The timer's Drop implementation will call timer_delete when the thread exits
    // or when stop_profiling() is called
    PROFILING_TIMER.with(|cell| {
        *cell.borrow_mut() = Some(timer);
    });

    Ok(())
}

/// Stops profiling for the current thread.
///
/// This stops the interval timer and prevents further SIGPROF signals from being
/// delivered to this thread. The timer resources are cleaned up via the timer's
/// Drop implementation (which calls `timer_delete`).
///
/// It is safe to call this function even if profiling was never started for this
/// thread - it will simply do nothing in that case.
///
/// # Note
///
/// This only stops profiling for the calling thread. Other threads that have
/// started profiling will continue to receive signals.
pub fn stop_profiling() {
    PROFILING_TIMER.with(|cell| {
        if let Some(timer) = cell.borrow_mut().take() {
            // Stop the timer before dropping to ensure no more signals are delivered
            // Ignore errors since we're stopping anyway
            let _ = timer.stop();
            // Timer is dropped here, which calls timer_delete via Drop
        }
    });
}

/// Returns whether profiling is currently active for any thread.
///
/// This is used to short-circuit delay consumption checks when profiling
/// is disabled, avoiding unnecessary overhead.
#[inline]
pub fn is_profiling_active() -> bool {
    PROFILING_ACTIVE.load(Ordering::Relaxed)
}

/// Enables the profiling active flag.
///
/// Called when profiling experiments start.
pub(crate) fn set_profiling_active(active: bool) {
    PROFILING_ACTIVE.store(active, Ordering::Release);
}

/// Consumes any pending delay for the current thread.
///
/// This function is the core of the Coz virtual speedup mechanism. When called,
/// it checks if the current thread has accumulated delay debt (from samples
/// hitting the selected code during an experiment). If so, it sleeps for
/// that duration to simulate the virtual speedup.
///
/// # Usage
///
/// This is called automatically from:
/// - Progress points (`progress!()` macro)
/// - Delay points (`delay_point!()` macro)
/// - Profiled synchronization wrappers
///
/// # Performance
///
/// When profiling is not active, this function returns immediately with minimal
/// overhead (a single atomic load).
#[inline]
pub fn consume_pending_delay() {
    // Fast path: skip if profiling is not active
    if !is_profiling_active() {
        return;
    }

    // Get the thread profiler from thread-local storage
    if let Some(profiler) = experiment::get_thread_profiler_public() {
        let pending_ns = profiler.take_pending_delay();
        if pending_ns > 0 {
            let delay = Duration::from_nanos(pending_ns);
            // Use nanosleep to apply the delay
            // Ignore errors - we're simulating delays, not guaranteeing them
            let _ = timer::nanosleep(delay);
        }
    }
}

/// Insert a delay consumption point in tight loops.
///
/// Use this macro in loops that don't have progress points but where delay
/// injection is needed for accurate causal profiling.
///
/// # Example
///
/// ```ignore
/// use andweorc::delay_point;
///
/// fn tight_loop() {
///     for i in 0..1_000_000 {
///         // ... do work ...
///         delay_point!();  // Allow delays to be injected
///     }
/// }
/// ```
///
/// # Performance
///
/// When profiling is not active, this expands to a single atomic load check.
#[macro_export]
macro_rules! delay_point {
    () => {
        $crate::consume_pending_delay()
    };
}

/// Re-export the profile attribute macro.
pub use andweorc_macros::profile;

/// Initializes the profiler if the `ANDWEORC_ENABLED` environment variable is set.
///
/// Call this at the start of your program's `main()` function. The profiler will
/// only activate if `ANDWEORC_ENABLED=1` is set (typically done by `cargo andweorc`).
///
/// # Example
///
/// ```ignore
/// fn main() {
///     andweorc::init();
///     // ... your program ...
/// }
/// ```
///
/// # Errors
///
/// Returns a [`ProfilingError`] if profiling was requested but could not be started.
/// See [`start_profiling`] for the specific error conditions.
pub fn init() -> Result<(), ProfilingError> {
    if std::env::var("ANDWEORC_ENABLED").is_ok_and(|v| v == "1") {
        start_profiling()?;
    }
    Ok(())
}

/// Runs the profiler experiment loop and returns results.
///
/// This function orchestrates the causal profiling experiments:
/// 1. Collects baseline measurements
/// 2. Tests virtual speedups at each hot code location
/// 3. Calculates causal impact
///
/// Call this after your program has been running long enough to collect
/// meaningful samples (typically at the end of main or at a checkpoint).
///
/// # Arguments
///
/// * `progress_point` - Name of the progress point to measure throughput against.
///
/// # Returns
///
/// Returns `Some` with a static reference to the profiling results if profiling
/// is enabled, `None` otherwise. The reference is valid for the lifetime of the
/// program (the results are intentionally leaked to provide a stable reference).
///
/// # Example
///
/// ```ignore
/// fn main() {
///     andweorc::init();
///     // ... your workload ...
///     if let Some(results) = andweorc::run_experiments("request_done") {
///         // results contains causal impact data
///     }
/// }
/// ```
#[must_use]
pub fn run_experiments(progress_point: &'static str) -> Option<&'static runner::ProfilingResults> {
    if std::env::var("ANDWEORC_ENABLED").is_ok_and(|v| v == "1") {
        let runner = runner::Runner::new();
        let _ = runner.run(progress_point);

        // Leak the runner to get a static reference to results
        // This is intentional - results need to outlive the function call
        let leaked_runner = Box::leak(Box::new(runner));
        let results = leaked_runner.results();

        // Output in machine-parseable format for CLI
        for ip in results.profiled_ips() {
            if let Some(experiments) = results.results_for_ip(ip) {
                let symbol = resolve_symbol(ip);
                for exp in experiments {
                    libc_print::libc_println!(
                        "EXPERIMENT: ip=0x{:x} symbol={} speedup={:.2} throughput={:.2} duration={} samples={}",
                        exp.ip,
                        symbol,
                        exp.speedup_pct,
                        exp.throughput,
                        exp.duration_ms,
                        exp.matching_samples
                    );
                }
            }
        }

        // Print human-readable summary with resolved symbols
        let impacts = results.calculate_impacts();
        libc_print::libc_println!("\n=== Causal Profiling Results ===");
        libc_print::libc_println!("Top optimization opportunities:\n");

        for (i, (ip, impact)) in impacts.iter().take(10).enumerate() {
            let symbol = resolve_symbol(*ip);
            libc_print::libc_println!("{}. {} (impact = {:.4})", i + 1, symbol, impact);
        }

        if impacts.is_empty() {
            libc_print::libc_println!("No optimization opportunities found.");
            libc_print::libc_println!("Make sure the program ran long enough to collect samples.");
        }

        Some(results)
    } else {
        None
    }
}
