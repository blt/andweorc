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
mod per_thread;
// NOTE: posix module disabled for now - pthread interceptors require LD_PRELOAD
// and don't work when directly linked (dlsym returns NULL for RTLD_NEXT)
// mod posix;
pub mod progress_point;
pub mod runner;
mod timer;

use per_thread::PerThreadProfiler;
use std::cell::RefCell;
use std::sync::Arc;
use std::time::Duration;

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
/// Returns an error string if the timer cannot be created or started.
///
/// # Panics
///
/// Panics if the experiment singleton failed to initialize (e.g., SIGPROF
/// signal handler could not be registered).
pub fn start_profiling() -> Result<(), String> {
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
/// Returns an error string if the timer cannot be created or started, or if
/// profiling is already active for this thread.
///
/// # Panics
///
/// Panics if the experiment singleton failed to initialize (e.g., SIGPROF
/// signal handler could not be registered).
pub fn start_profiling_with_interval(interval: Duration) -> Result<(), String> {
    // Check if profiling is already active for this thread
    let already_profiling = PROFILING_TIMER.with(|cell| cell.borrow().is_some());
    if already_profiling {
        return Err("profiling already active for this thread".to_string());
    }

    // Ensure the experiment singleton is initialized (this registers the signal handler)
    let _ = experiment::get_instance();

    // IMPORTANT: Create the per-thread profiler BEFORE starting the timer.
    // This avoids deadlock where the signal handler tries to access the profiler
    // while we're in the middle of creating it. The profiler is stored in
    // thread-local storage and accessed from the signal handler without locks.
    let profiler = Arc::new(PerThreadProfiler::new(16));
    experiment::set_thread_profiler(profiler);

    // Get the current thread ID (Linux specific)
    // SAFETY: gettid is a simple syscall that requires no special privileges
    let kernel_tid_raw = unsafe { libc::syscall(libc::SYS_gettid) };

    // gettid returns -1 on error (though this should never happen for gettid)
    if kernel_tid_raw < 0 {
        return Err("gettid syscall failed".to_string());
    }

    // gettid returns pid_t which fits in i32 on Linux
    #[allow(clippy::cast_possible_truncation)]
    let kernel_tid = kernel_tid_raw as libc::pid_t;

    let timer = timer::Interval::new(kernel_tid, libc::SIGPROF)
        .map_err(|e| format!("failed to create timer: {e}"))?;
    timer
        .start(interval)
        .map_err(|e| format!("failed to start timer: {e}"))?;

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
/// Returns an error if profiling was requested but could not be started.
///
/// # Panics
///
/// Panics if the experiment singleton failed to initialize (e.g., SIGPROF
/// signal handler could not be registered) while `ANDWEORC_ENABLED=1` is set.
pub fn init() -> Result<(), String> {
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
