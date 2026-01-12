//! Central coordinator for causal profiling experiments.
//!
//! The `Experiment` singleton manages global state across all threads including:
//! - Signal handler registration for periodic sampling
//! - Per-thread profiler state (via thread-local storage)
//! - Progress point tracking
//! - Delay table for virtual speedup experiments
//!
//! # Signal Safety
//!
//! This module is designed with signal safety as a primary concern. All data
//! structures accessed from the SIGPROF signal handler are either:
//! - Atomic primitives (async-signal-safe)
//! - Thread-local storage (accessed without locks)
//! - Pre-allocated fixed-size arrays with lock-free access

use crate::lock_util::{recover_read, recover_write};
use crate::per_thread::PerThreadProfiler;
use crate::progress_point::Progress;
use libc::{c_int, c_void};
use nix::errno::Errno;
use nix::sys::signal::{signal, SigHandler, Signal};
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicPtr, AtomicU32, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock, RwLock};
use std::time::Duration;

/// Percentage multipliers for the delay table.
///
/// The first 25 entries are 0% (baseline measurements), followed by
/// increasing percentages from 5% to 150% of the baseline delay.
///
/// # Invariants
///
/// - All values are non-negative
/// - First 25 entries are 0.0 (baseline)
/// - Remaining entries increase monotonically by 0.05
/// - Maximum value is 1.50 (150% speedup)
const DELAY_PRCNT: [f64; 55] = [
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55,
    0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15, 1.20, 1.25, 1.30, 1.35,
    1.40, 1.45, 1.50,
];

/// The baseline delay duration used to compute actual delays.
const DELAY_BASELINE: Duration = Duration::from_millis(10);

/// Number of buckets for the sample count hash table.
/// Must be a power of 2 for efficient modulo via bitwise AND.
const SAMPLE_COUNT_BUCKETS: usize = 8192;

/// Maximum linear probing iterations before giving up.
///
/// This bounds the worst-case time spent in the signal handler.
/// With 8192 buckets and good hash distribution, 32 probes should
/// be sufficient for finding or inserting most IPs. If we exhaust
/// probes, the sample is silently dropped (acceptable for profiling).
const MAX_PROBE_ITERATIONS: usize = 32;

/// Flag indicating whether the experiment singleton is fully initialized.
/// This is used by the signal handler to avoid accessing uninitialized data.
static EXPERIMENT_INITIALIZED: AtomicBool = AtomicBool::new(false);

/// Errors that can occur during experiment initialization.
#[derive(Debug, Clone, Copy)]
pub enum ExperimentInitError {
    /// Failed to register the SIGPROF signal handler.
    SignalHandlerRegistration(Errno),
}

impl std::fmt::Display for ExperimentInitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::SignalHandlerRegistration(e) => {
                write!(f, "failed to register SIGPROF signal handler: {e}")
            }
        }
    }
}

impl std::error::Error for ExperimentInitError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::SignalHandlerRegistration(e) => Some(e),
        }
    }
}

// Thread-local storage for the per-thread profiler.
//
// This is accessed from the signal handler, so we use a raw pointer
// to avoid any locking. The profiler is created before the timer starts
// and lives for the duration of profiling.
thread_local! {
    static THREAD_PROFILER: std::cell::Cell<Option<Arc<PerThreadProfiler>>> = const { std::cell::Cell::new(None) };
}

/// Sets the per-thread profiler for the current thread.
///
/// Must be called before starting the profiling timer.
pub(crate) fn set_thread_profiler(profiler: Arc<PerThreadProfiler>) {
    THREAD_PROFILER.with(|cell| {
        cell.set(Some(profiler));
    });
}

/// Gets the per-thread profiler for the current thread.
///
/// Returns None if no profiler has been set.
fn get_thread_profiler() -> Option<Arc<PerThreadProfiler>> {
    THREAD_PROFILER.with(|cell| {
        // We need to temporarily take the value to clone it
        let profiler = cell.take();
        let result = profiler.clone();
        cell.set(profiler);
        result
    })
}

/// Public accessor for the per-thread profiler.
///
/// Used by `consume_pending_delay()` to access delay balances.
pub(crate) fn get_thread_profiler_public() -> Option<Arc<PerThreadProfiler>> {
    get_thread_profiler()
}

/// Signal handler that processes perf samples when SIGPROF fires.
///
/// This handler is called on whichever thread receives the signal. It looks up
/// the thread's profiler from thread-local storage and triggers sample processing.
///
/// # Safety
///
/// This is a signal handler and only calls async-signal-safe functions:
/// - Atomic loads/stores (async-signal-safe)
/// - Thread-local storage access (async-signal-safe on Linux)
/// - `pthread_sigmask` (async-signal-safe per POSIX)
///
/// The per-thread profiler is pre-allocated before the timer starts, ensuring
/// no memory allocation occurs in signal context.
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
    if !EXPERIMENT_INITIALIZED.load(Ordering::Acquire) {
        unsafe {
            libc::pthread_sigmask(libc::SIG_SETMASK, &raw const oldset, std::ptr::null_mut());
        }
        return;
    }

    // Get the profiler from thread-local storage (no locks needed)
    if let Some(profiler) = get_thread_profiler() {
        profiler.process_samples();
    }

    // Restore original signal mask
    unsafe {
        libc::pthread_sigmask(libc::SIG_SETMASK, &raw const oldset, std::ptr::null_mut());
    }
}

static EXPERIMENT: OnceLock<Result<Experiment, ExperimentInitError>> = OnceLock::new();

/// Returns a reference to the global experiment singleton.
///
/// Initializes the experiment on first call, including registering the
/// SIGPROF signal handler.
///
/// # Errors
///
/// Returns an error if the SIGPROF signal handler cannot be registered.
/// This typically happens when:
/// - Running in an environment that doesn't support SIGPROF
/// - The process doesn't have permission to set signal handlers
pub(crate) fn try_get_instance() -> Result<&'static Experiment, &'static ExperimentInitError> {
    let result = EXPERIMENT.get_or_init(|| {
        let exp_result = Experiment::new(DELAY_BASELINE);
        if exp_result.is_ok() {
            // Mark as initialized AFTER the experiment is fully constructed
            EXPERIMENT_INITIALIZED.store(true, Ordering::Release);
        }
        exp_result
    });
    match result {
        Ok(exp) => Ok(exp),
        Err(e) => Err(e),
    }
}

/// Returns a reference to the global experiment singleton.
///
/// This is a convenience function that assumes initialization succeeded.
/// Use `try_get_instance()` if you need to handle initialization errors.
///
/// # Panics
///
/// Panics if the experiment failed to initialize. This can happen if
/// the SIGPROF signal handler cannot be registered.
///
/// # Signal Safety Note
///
/// Although this function can panic, it is safe to call from the signal handler
/// code path because:
/// 1. The signal handler checks `EXPERIMENT_INITIALIZED` before calling code that
///    uses this function
/// 2. `EXPERIMENT_INITIALIZED` is only set to true AFTER successful initialization
/// 3. Therefore, if reached via signal handler, initialization has already succeeded
///
/// This panic can only occur during initial setup (outside signal context), not
/// during production profiling operations.
pub(crate) fn get_instance() -> &'static Experiment {
    match try_get_instance() {
        Ok(exp) => exp,
        Err(e) => panic!("experiment initialization failed: {e}"),
    }
}

/// Initializes the experiment singleton for `LD_PRELOAD` mode.
///
/// This is similar to `try_get_instance()` but returns the error type directly
/// for easier error handling in the library constructor.
///
/// # Errors
///
/// Returns an error if the SIGPROF signal handler cannot be registered.
pub fn try_init() -> Result<&'static Experiment, ExperimentInitError> {
    try_get_instance().map_err(|e| *e)
}

/// Shuts down the profiler and outputs results.
///
/// Called from the library destructor or explicitly when profiling should stop.
/// This function:
/// 1. Stops any active experiment
/// 2. Sets profiling inactive
/// 3. Outputs accumulated results
pub fn shutdown() {
    // Mark profiling as inactive first
    crate::set_profiling_active(false);

    // Stop any active experiment
    if let Ok(exp) = try_get_instance() {
        exp.stop_experiment();

        // Output basic statistics
        let total = exp.total_samples.load(Ordering::Acquire);
        let selected = exp.selected_samples.load(Ordering::Acquire);
        libc_print::libc_println!(
            "[andweorc] stats: total_samples={total}, selected_samples={selected}"
        );

        // Output top sampled locations
        let top = exp.top_ips(10);
        if !top.is_empty() {
            libc_print::libc_println!("[andweorc] top sampled locations:");
            for (i, ip) in top.iter().enumerate() {
                libc_print::libc_println!("  {}: 0x{:x}", i + 1, ip);
            }
        }
    }
}

/// Registers a thread with the profiler.
///
/// Called from the `pthread_create` interceptor to set up profiling for a new thread.
/// This sets up the per-thread profiler and registers it with the experiment.
///
/// # Arguments
///
/// * `_tid` - The pthread ID of the thread being registered (currently unused,
///   but reserved for future thread tracking).
pub fn register_thread(_tid: libc::pthread_t) {
    // Only create profiler if profiling is actually active
    // This prevents crashes when hardware counters aren't available
    if !crate::is_profiling_active() {
        return;
    }

    // Create and set up the per-thread profiler
    let profiler = std::sync::Arc::new(crate::per_thread::PerThreadProfiler::new(16));
    set_thread_profiler(profiler);
}

/// Deregisters a thread from the profiler.
///
/// Called from the `pthread_exit` interceptor to clean up profiling state.
/// Currently a no-op as thread-local storage handles cleanup automatically.
///
/// # Arguments
///
/// * `_tid` - The pthread ID of the thread being deregistered.
pub fn deregister_thread(_tid: libc::pthread_t) {
    // Thread-local storage handles cleanup automatically via Drop
    // No explicit cleanup needed currently
}

/// A single sample entry containing an IP and its count.
///
/// IP and count are stored together to ensure they share a cache line,
/// avoiding false sharing when different threads access different entries.
#[repr(C)]
struct SampleEntry {
    /// Instruction pointer (0 = empty slot)
    ip: AtomicUsize,
    /// Number of times this IP was sampled
    count: AtomicU64,
}

impl SampleEntry {
    /// Creates a new empty sample entry.
    const fn new() -> Self {
        Self {
            ip: AtomicUsize::new(0),
            count: AtomicU64::new(0),
        }
    }
}

/// Lock-free sample counter using open addressing with linear probing.
///
/// This data structure is designed for use in signal handlers where
/// acquiring locks is forbidden. It uses atomic operations exclusively.
///
/// # Design
///
/// - Fixed-size array of interleaved (IP, count) entries
/// - Uses linear probing for collision resolution
/// - IPs are stored as `AtomicUsize`, counts as `AtomicU64`
/// - An IP of 0 indicates an empty slot
/// - IP and count are colocated to avoid false sharing
///
/// # Limitations
///
/// - Cannot track more than `SAMPLE_COUNT_BUCKETS` unique IPs
/// - Once a slot is used, it's never freed (acceptable for profiling)
/// - Hash collisions may cause some IPs to be missed if table is full
struct SampleCounts {
    /// Interleaved IP/count entries for cache efficiency
    entries: Box<[SampleEntry; SAMPLE_COUNT_BUCKETS]>,
}

impl std::fmt::Debug for SampleCounts {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SampleCounts")
            .field("buckets", &SAMPLE_COUNT_BUCKETS)
            .finish_non_exhaustive()
    }
}

impl SampleCounts {
    /// Creates a new empty sample counter.
    fn new() -> Self {
        // Initialize entries array with empty entries
        // Using Box to avoid stack overflow with large arrays
        let entries = Box::new(std::array::from_fn(|_| SampleEntry::new()));
        Self { entries }
    }

    /// Hashes an IP address to a bucket index.
    ///
    /// Uses FNV-1a hash for good distribution.
    #[inline]
    fn hash(ip: usize) -> usize {
        // FNV-1a hash
        const FNV_OFFSET: usize = 0xcbf2_9ce4_8422_2325_usize;
        const FNV_PRIME: usize = 0x0100_0000_01b3_usize;

        let mut hash = FNV_OFFSET;
        // Hash each byte of the IP
        for i in 0..std::mem::size_of::<usize>() {
            let byte = (ip >> (i * 8)) & 0xFF;
            hash ^= byte;
            hash = hash.wrapping_mul(FNV_PRIME);
        }
        hash & (SAMPLE_COUNT_BUCKETS - 1)
    }

    /// Increments the count for the given IP.
    ///
    /// Uses lock-free linear probing with bounded iterations. If the probe
    /// limit is reached without finding a slot, the increment is silently dropped.
    ///
    /// # Signal Safety
    ///
    /// This function is async-signal-safe: it only uses atomic operations.
    /// The probe count is bounded by `MAX_PROBE_ITERATIONS` to ensure
    /// bounded execution time in signal handler context.
    ///
    /// # Memory Ordering
    ///
    /// - Relaxed loads for checking existing entries (hot path)
    /// - Release on successful slot claim to publish the IP
    /// - Relaxed for count increments (counts are always read with Acquire)
    fn increment(&self, ip: usize) {
        if ip == 0 {
            return; // 0 is reserved for empty slots
        }

        let start = Self::hash(ip);

        // Linear probing with bounded iterations for signal safety
        for i in 0..MAX_PROBE_ITERATIONS {
            let idx = (start + i) & (SAMPLE_COUNT_BUCKETS - 1);
            let entry = &self.entries[idx];

            let current = entry.ip.load(Ordering::Relaxed);

            if current == ip {
                // Found existing entry, increment count
                entry.count.fetch_add(1, Ordering::Relaxed);
                return;
            }

            if current == 0 {
                // Empty slot - try to claim it
                // Use Release on success to publish the IP to readers
                match entry
                    .ip
                    .compare_exchange(0, ip, Ordering::Release, Ordering::Relaxed)
                {
                    Ok(_) => {
                        // Successfully claimed the slot
                        entry.count.fetch_add(1, Ordering::Relaxed);
                        return;
                    }
                    Err(actual) => {
                        if actual == ip {
                            // Another thread/signal just inserted this IP
                            entry.count.fetch_add(1, Ordering::Relaxed);
                            return;
                        }
                        // Someone else claimed it with a different IP, continue probing
                    }
                }
            }
            // Slot occupied by different IP, continue probing
        }
        // Probe limit reached, silently drop this sample (acceptable for profiling)
    }

    /// Returns all (IP, count) pairs with non-zero counts.
    ///
    /// # Memory Ordering
    ///
    /// Uses Acquire loads to synchronize with Release stores in `increment()`.
    /// This ensures we see all counts that were incremented before we read.
    /// Call this only after all profiling threads have stopped.
    fn entries(&self) -> Vec<(usize, u64)> {
        let mut result = Vec::new();
        for entry in self.entries.iter() {
            // Acquire synchronizes with Release in increment() when slot was claimed
            let ip = entry.ip.load(Ordering::Acquire);
            if ip != 0 {
                // Acquire to see all count updates
                let count = entry.count.load(Ordering::Acquire);
                if count > 0 {
                    result.push((ip, count));
                }
            }
        }
        result
    }
}

/// Central coordinator for causal profiling experiments.
///
/// Manages all threads participating in the experiment, tracks progress points,
/// and coordinates delay injection for virtual speedup measurements.
///
/// # Signal Safety
///
/// Fields accessed from signal handlers use only atomic operations:
/// - `sample_counts`: Lock-free hash table
/// - `global_delay_count`, `delay_index`, etc.: Atomic primitives
/// - `delay_table`: Read-only after initialization
///
/// Fields NOT accessed from signal handlers use standard synchronization:
/// - `progress_points`: `RwLock`-protected `HashMap`
pub struct Experiment {
    /// The previous signal handler for SIGPROF, saved for restoration on Drop.
    previous_handler: SigHandler,

    /// Named progress points for throughput measurement.
    /// Protected by `RwLock` since not accessed from signal handlers.
    progress_points: RwLock<HashMap<&'static str, Arc<Progress>>>,

    /// Sample counts by instruction pointer (for finding hot code).
    /// Lock-free structure safe for signal handler access.
    sample_counts: SampleCounts,

    /// Pre-computed delay durations for each speedup percentage.
    /// Read-only after initialization.
    delay_table: Vec<Duration>,

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

// SAFETY: Experiment uses only atomic operations for cross-thread access
// from signal handlers. The RwLock-protected fields are only accessed
// from normal (non-signal) context.
unsafe impl Sync for Experiment {}

impl std::fmt::Debug for Experiment {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Experiment")
            .field("delay_table_len", &self.delay_table.len())
            .field("is_active", &self.is_active.load(Ordering::Relaxed))
            .field("total_samples", &self.total_samples.load(Ordering::Relaxed))
            .finish_non_exhaustive()
    }
}

/// Statistics about the current experiment.
#[derive(Debug, Clone, Copy)]
pub(crate) struct ExperimentStats {
    /// Samples matching the selected IP.
    pub selected_samples: u64,
}

impl Experiment {
    /// Creates a new experiment with the given baseline delay duration.
    ///
    /// # Errors
    ///
    /// Returns an error if the SIGPROF signal handler cannot be registered.
    fn new(delay_baseline: Duration) -> Result<Self, ExperimentInitError> {
        let handler = SigHandler::Handler(process_samples);

        // Register the signal handler for SIGPROF.
        // SAFETY: The handler function is valid for the lifetime of the program.
        // The previous handler is saved and will be restored in Drop.
        let previous_handler = unsafe { signal(Signal::SIGPROF, handler) }
            .map_err(ExperimentInitError::SignalHandlerRegistration)?;

        Ok(Self {
            previous_handler,
            progress_points: RwLock::new(HashMap::new()),
            sample_counts: SampleCounts::new(),
            global_delay_count: AtomicU32::new(0),
            delay_index: AtomicUsize::new(0),
            selected_ip: AtomicPtr::new(std::ptr::null_mut()),
            total_samples: AtomicU64::new(0),
            selected_samples: AtomicU64::new(0),
            is_active: AtomicBool::new(false),
            delay_table: DELAY_PRCNT
                .iter()
                .map(|prct| delay_baseline.mul_f64(*prct))
                .collect(),
        })
    }

    /// Returns statistics about the current experiment.
    pub(crate) fn stats(&self) -> ExperimentStats {
        ExperimentStats {
            selected_samples: self.selected_samples.load(Ordering::Relaxed),
        }
    }

    /// Starts a new experiment round with the given IP and speedup index.
    ///
    /// # Arguments
    ///
    /// * `ip` - The instruction pointer to virtually speed up.
    /// * `speedup_index` - Index into the delay table for speedup percentage.
    ///   If out of bounds, it will be clamped to the maximum valid index.
    pub(crate) fn start_experiment(&self, ip: *const c_void, speedup_index: usize) {
        // Reset counters
        self.total_samples.store(0, Ordering::Relaxed);
        self.selected_samples.store(0, Ordering::Relaxed);
        self.global_delay_count.store(0, Ordering::Relaxed);

        // Clamp speedup_index to valid range
        let safe_index = speedup_index.min(self.delay_table.len().saturating_sub(1));

        // Set the selected IP and speedup
        self.delay_index.store(safe_index, Ordering::Relaxed);
        self.selected_ip.store(ip.cast_mut(), Ordering::Release);
        self.is_active.store(true, Ordering::Release);
    }

    /// Stops the current experiment round.
    pub(crate) fn stop_experiment(&self) {
        self.is_active.store(false, Ordering::Release);
        self.selected_ip
            .store(std::ptr::null_mut(), Ordering::Release);
    }

    /// Returns the progress point with the given name, creating one if necessary.
    ///
    /// This function acquires a lock and should NOT be called from signal handlers.
    pub(crate) fn progress(&self, name: &'static str) -> Arc<Progress> {
        // Fast path: check if it exists with a read lock
        {
            let read_guard = recover_read(self.progress_points.read());
            if let Some(progress) = read_guard.get(name) {
                return Arc::clone(progress);
            }
        }

        // Slow path: acquire write lock and insert
        let mut write_guard = recover_write(self.progress_points.write());

        // Double-check after acquiring write lock
        if let Some(progress) = write_guard.get(name) {
            return Arc::clone(progress);
        }

        let progress = Arc::new(Progress::new(name));
        write_guard.insert(name, Arc::clone(&progress));
        progress
    }

    /// Records a sample from a thread's profiler.
    ///
    /// This is called from the signal handler context with the instruction
    /// pointer and optional call chain from the perf sample.
    ///
    /// If an experiment is active and the sample IP matches the selected IP,
    /// the global delay count is incremented, causing other threads to delay.
    ///
    /// # Signal Safety
    ///
    /// This function is async-signal-safe: it only uses atomic operations
    /// and the lock-free `SampleCounts` data structure.
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
            self.sample_counts.increment(ip_usize);
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
                // This sample is at the selected line - this thread should be delayed
                // to simulate virtual speedup (Coz algorithm)
                self.selected_samples.fetch_add(1, Ordering::Relaxed);
                self.global_delay_count.fetch_add(1, Ordering::Relaxed);

                // Add delay to this thread's pending balance
                // The delay will be consumed at the next progress point
                if let Some(profiler) = get_thread_profiler() {
                    let delay_index = self.delay_index.load(Ordering::Relaxed);
                    // Safety: delay_index is clamped to valid range in start_experiment()
                    if delay_index < self.delay_table.len() {
                        // Allow truncation: delay values are bounded to max 15ms (< u64::MAX)
                        #[allow(clippy::cast_possible_truncation)]
                        let delay_ns = self.delay_table[delay_index].as_nanos() as u64;
                        profiler.add_delay(delay_ns);
                    }
                }
            }
        }
    }

    /// Returns the top N most frequently sampled IPs.
    pub(crate) fn top_ips(&self, n: usize) -> Vec<usize> {
        let mut counts = self.sample_counts.entries();
        counts.sort_by(|a, b| b.1.cmp(&a.1));
        counts.into_iter().take(n).map(|(ip, _)| ip).collect()
    }
}

impl Drop for Experiment {
    fn drop(&mut self) {
        // Restore the previous SIGPROF signal handler.
        // SAFETY: We're restoring the handler that was saved during initialization.
        // This is safe because the previous handler was valid when we saved it.
        //
        // Note: Since Experiment is stored in a OnceLock static, this Drop will
        // only run during program shutdown (if at all). However, implementing it
        // correctly ensures proper cleanup semantics.
        let _ = unsafe { signal(Signal::SIGPROF, self.previous_handler) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ==========================================================================
    // DELAY TABLE INVARIANT TESTS
    // ==========================================================================

    #[test]
    fn delay_prcnt_has_correct_length() {
        assert_eq!(DELAY_PRCNT.len(), 55);
    }

    #[test]
    fn delay_prcnt_first_25_are_baseline() {
        for (i, &prcnt) in DELAY_PRCNT.iter().take(25).enumerate() {
            assert!(
                prcnt.abs() < f64::EPSILON,
                "DELAY_PRCNT[{i}] should be 0.0, got {prcnt}"
            );
        }
    }

    #[test]
    fn delay_prcnt_values_are_non_negative() {
        for (i, &prcnt) in DELAY_PRCNT.iter().enumerate() {
            assert!(
                prcnt >= 0.0,
                "DELAY_PRCNT[{i}] should be >= 0.0, got {prcnt}"
            );
        }
    }

    #[test]
    fn delay_prcnt_speedup_values_increase_monotonically() {
        for i in 26..DELAY_PRCNT.len() {
            let prev = DELAY_PRCNT[i - 1];
            let curr = DELAY_PRCNT[i];
            assert!(
                curr > prev || (i <= 25),
                "DELAY_PRCNT[{i}] ({curr}) should be > DELAY_PRCNT[{}] ({prev})",
                i - 1
            );
        }
    }

    #[test]
    fn delay_prcnt_max_is_150_percent() {
        let max = DELAY_PRCNT.iter().copied().fold(0.0_f64, f64::max);
        assert!(
            (max - 1.5).abs() < f64::EPSILON,
            "Maximum delay percent should be 1.5, got {max}"
        );
    }

    // ==========================================================================
    // SAMPLE COUNTS TESTS
    // ==========================================================================

    #[test]
    fn sample_counts_empty_initially() {
        let counts = SampleCounts::new();
        let entries = counts.entries();
        assert!(entries.is_empty(), "New SampleCounts should be empty");
    }

    #[test]
    fn sample_counts_increment_single() {
        let counts = SampleCounts::new();
        counts.increment(0x1234);
        let entries = counts.entries();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0], (0x1234, 1));
    }

    #[test]
    fn sample_counts_increment_multiple_same_ip() {
        let counts = SampleCounts::new();
        for _ in 0..100 {
            counts.increment(0xABCD);
        }
        let entries = counts.entries();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0], (0xABCD, 100));
    }

    #[test]
    fn sample_counts_increment_different_ips() {
        let counts = SampleCounts::new();
        counts.increment(0x1000);
        counts.increment(0x2000);
        counts.increment(0x3000);
        counts.increment(0x1000); // Again

        let mut entries = counts.entries();
        entries.sort_by_key(|&(ip, _)| ip);

        assert_eq!(entries.len(), 3);
        assert!(entries.contains(&(0x1000, 2)));
        assert!(entries.contains(&(0x2000, 1)));
        assert!(entries.contains(&(0x3000, 1)));
    }

    #[test]
    fn sample_counts_ignores_zero_ip() {
        let counts = SampleCounts::new();
        counts.increment(0); // Should be ignored
        counts.increment(0x1234);
        let entries = counts.entries();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].0, 0x1234);
    }

    #[test]
    fn sample_counts_hash_distributes_well() {
        // Test that hash function distributes values across buckets
        let mut buckets_used = std::collections::HashSet::new();
        for i in 0..1000 {
            let ip = 0x4000_0000 + i * 0x1000;
            let bucket = SampleCounts::hash(ip);
            buckets_used.insert(bucket);
        }
        // With 1000 different IPs and 8192 buckets, we should use many buckets
        assert!(
            buckets_used.len() > 500,
            "Hash should distribute well, got {} buckets",
            buckets_used.len()
        );
    }

    // ==========================================================================
    // DELAY DETAILS TESTS
    // ==========================================================================

    #[test]
    fn delay_table_matches_percentages() {
        // Verify the delay table is correctly computed from percentages
        let delay_table: Vec<Duration> = DELAY_PRCNT
            .iter()
            .map(|prct| DELAY_BASELINE.mul_f64(*prct))
            .collect();

        // First 25 should be zero (baseline)
        for (i, delay) in delay_table.iter().take(25).enumerate() {
            assert_eq!(*delay, Duration::ZERO, "delay_table[{i}] should be ZERO");
        }

        // Entry 25 should be 5% of baseline = 0.5ms
        let expected_5pct = Duration::from_micros(500);
        assert_eq!(delay_table[25], expected_5pct);

        // Entry 54 (last) should be 150% of baseline = 15ms
        let expected_150pct = Duration::from_millis(15);
        assert_eq!(delay_table[54], expected_150pct);
    }
}

// =============================================================================
// KANI PROOFS
// =============================================================================
//
// These proofs formally verify critical invariants that the profiler depends on.
// Run with: ci/kani andweorc

#[cfg(kani)]
mod kani_proofs {
    use super::*;

    /// Proof: Hash function always produces a valid bucket index.
    ///
    /// This is critical for signal safety - an out-of-bounds index would cause
    /// undefined behavior in the signal handler.
    #[kani::proof]
    fn hash_produces_valid_index() {
        let ip: usize = kani::any();
        let bucket = SampleCounts::hash(ip);
        kani::assert(
            bucket < SAMPLE_COUNT_BUCKETS,
            "hash must produce valid bucket index",
        );
    }

    /// Proof: All delay percentages are non-negative.
    ///
    /// Negative delays would cause incorrect virtual speedup calculations.
    #[kani::proof]
    fn delay_prcnt_all_non_negative() {
        let idx: usize = kani::any();
        kani::assume(idx < DELAY_PRCNT.len());
        kani::assert(
            DELAY_PRCNT[idx] >= 0.0,
            "delay percentage must be non-negative",
        );
    }

    /// Proof: Baseline indices (0-24) have zero delay percentage.
    ///
    /// These are used for baseline measurements where no virtual speedup is applied.
    #[kani::proof]
    fn delay_prcnt_baseline_is_zero() {
        let idx: usize = kani::any();
        kani::assume(idx < 25);
        kani::assert(
            DELAY_PRCNT[idx] == 0.0,
            "baseline indices must have zero delay",
        );
    }

    /// Proof: Speedup indices (25-54) have monotonically increasing values.
    ///
    /// This ensures higher speedup indices always result in more delay.
    #[kani::proof]
    fn delay_prcnt_speedup_monotonic() {
        let idx: usize = kani::any();
        kani::assume(idx >= 26 && idx < DELAY_PRCNT.len());
        kani::assert(
            DELAY_PRCNT[idx] > DELAY_PRCNT[idx - 1],
            "speedup delays must be monotonically increasing",
        );
    }

    /// Proof: Maximum delay percentage is 1.5 (150%).
    ///
    /// This bounds the maximum virtual speedup that can be simulated.
    #[kani::proof]
    fn delay_prcnt_max_bounded() {
        let idx: usize = kani::any();
        kani::assume(idx < DELAY_PRCNT.len());
        kani::assert(
            DELAY_PRCNT[idx] <= 1.5,
            "delay percentage must not exceed 150%",
        );
    }

    /// Proof: Delay table has exactly 55 entries.
    ///
    /// The experiment runner depends on this size for index calculations.
    #[kani::proof]
    fn delay_table_size() {
        kani::assert(
            DELAY_PRCNT.len() == 55,
            "delay table must have exactly 55 entries",
        );
    }

    /// Proof: Index clamping always produces a valid index.
    ///
    /// The delay_details function uses this pattern to safely access the delay table.
    #[kani::proof]
    fn delay_index_clamp_valid() {
        let idx: usize = kani::any();
        let table_len = DELAY_PRCNT.len();

        // This is the clamping logic from delay_details()
        let safe_index = idx.min(table_len.saturating_sub(1));

        kani::assert(safe_index < table_len, "clamped index must be valid");
    }

    /// Proof: FNV-1a hash bucket mask is correct for power-of-2 size.
    ///
    /// SAMPLE_COUNT_BUCKETS must be a power of 2 for the bitwise AND mask to work.
    #[kani::proof]
    fn bucket_count_is_power_of_two() {
        // A number is a power of 2 if it has exactly one bit set
        // This is equivalent to: n > 0 && (n & (n - 1)) == 0
        let n = SAMPLE_COUNT_BUCKETS;
        kani::assert(n > 0, "bucket count must be positive");
        kani::assert(n & (n - 1) == 0, "bucket count must be power of 2");
    }
}
