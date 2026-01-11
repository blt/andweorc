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

use crate::per_thread::PerThreadProfiler;
use crate::progress_point::Progress;
use libc::{c_int, c_void};
use nix::errno::Errno;
use nix::sys::signal::{signal, SigHandler, Signal};
use rand::rngs::SmallRng;
use rand::SeedableRng;
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

/// Gets a random seed from the system entropy source.
///
/// Uses `/dev/urandom` on Linux for high-quality entropy without blocking.
/// Falls back to a combination of process ID and high-resolution timestamp
/// if the entropy source is unavailable.
fn get_random_seed() -> u64 {
    // Try to read from /dev/urandom
    let mut buf = [0u8; 8];
    // SAFETY: We're reading from a special file that provides random bytes.
    // This is safe and the file always exists on Linux.
    let result = unsafe {
        let fd = libc::open(c"/dev/urandom".as_ptr(), libc::O_RDONLY);
        if fd >= 0 {
            let bytes_read = libc::read(fd, buf.as_mut_ptr().cast(), 8);
            libc::close(fd);
            bytes_read == 8
        } else {
            false
        }
    };

    if result {
        u64::from_ne_bytes(buf)
    } else {
        // Fallback: combine PID and high-resolution time for reasonable entropy
        // Allow sign loss: PID is always positive on Linux
        #[allow(clippy::cast_sign_loss)]
        let pid = unsafe { libc::getpid() as u64 };
        let mut ts = libc::timespec {
            tv_sec: 0,
            tv_nsec: 0,
        };
        unsafe {
            libc::clock_gettime(libc::CLOCK_MONOTONIC, &raw mut ts);
        }
        #[allow(clippy::cast_sign_loss)]
        let time_component = (ts.tv_sec as u64)
            .wrapping_mul(1_000_000_000)
            .wrapping_add(ts.tv_nsec as u64);
        pid.wrapping_mul(0x517c_c1b7_2722_0a95) ^ time_component
    }
}

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
pub(crate) fn get_instance() -> &'static Experiment {
    match try_get_instance() {
        Ok(exp) => exp,
        Err(e) => panic!("experiment initialization failed: {e}"),
    }
}

/// Lock-free sample counter using open addressing with linear probing.
///
/// This data structure is designed for use in signal handlers where
/// acquiring locks is forbidden. It uses atomic operations exclusively.
///
/// # Design
///
/// - Fixed-size array of (IP, count) pairs
/// - Uses linear probing for collision resolution
/// - IPs are stored as `AtomicUsize`, counts as `AtomicU64`
/// - An IP of 0 indicates an empty slot
///
/// # Limitations
///
/// - Cannot track more than `SAMPLE_COUNT_BUCKETS` unique IPs
/// - Once a slot is used, it's never freed (acceptable for profiling)
/// - Hash collisions may cause some IPs to be missed if table is full
struct SampleCounts {
    /// Instruction pointers (0 = empty slot)
    ips: Box<[AtomicUsize; SAMPLE_COUNT_BUCKETS]>,
    /// Corresponding counts
    counts: Box<[AtomicU64; SAMPLE_COUNT_BUCKETS]>,
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
        // Initialize arrays with zeros
        // Using Box to avoid stack overflow with large arrays
        let ips = Box::new(std::array::from_fn(|_| AtomicUsize::new(0)));
        let counts = Box::new(std::array::from_fn(|_| AtomicU64::new(0)));
        Self { ips, counts }
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
    /// Uses lock-free linear probing. If the table is full and the IP
    /// is not found, the increment is silently dropped.
    ///
    /// # Signal Safety
    ///
    /// This function is async-signal-safe: it only uses atomic operations.
    fn increment(&self, ip: usize) {
        if ip == 0 {
            return; // 0 is reserved for empty slots
        }

        let start = Self::hash(ip);

        // Linear probing with a maximum of SAMPLE_COUNT_BUCKETS attempts
        for i in 0..SAMPLE_COUNT_BUCKETS {
            let idx = (start + i) & (SAMPLE_COUNT_BUCKETS - 1);

            let current = self.ips[idx].load(Ordering::Relaxed);

            if current == ip {
                // Found existing entry, increment count
                self.counts[idx].fetch_add(1, Ordering::Relaxed);
                return;
            }

            if current == 0 {
                // Empty slot - try to claim it
                match self.ips[idx].compare_exchange(0, ip, Ordering::Relaxed, Ordering::Relaxed) {
                    Ok(_) => {
                        // Successfully claimed the slot
                        self.counts[idx].fetch_add(1, Ordering::Relaxed);
                        return;
                    }
                    Err(actual) => {
                        if actual == ip {
                            // Another thread/signal just inserted this IP
                            self.counts[idx].fetch_add(1, Ordering::Relaxed);
                            return;
                        }
                        // Someone else claimed it with a different IP, continue probing
                    }
                }
            }
            // Slot occupied by different IP, continue probing
        }
        // Table full, silently drop this sample (acceptable for profiling)
    }

    /// Returns all (IP, count) pairs with non-zero counts.
    fn entries(&self) -> Vec<(usize, u64)> {
        let mut result = Vec::new();
        for i in 0..SAMPLE_COUNT_BUCKETS {
            let ip = self.ips[i].load(Ordering::Relaxed);
            if ip != 0 {
                let count = self.counts[i].load(Ordering::Relaxed);
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

        // Seed RNG from system entropy source for non-deterministic experiment scheduling
        let rng_seed = get_random_seed();

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
            rng: SmallRng::seed_from_u64(rng_seed),
            delay_table: DELAY_PRCNT
                .iter()
                .map(|prct| delay_baseline.mul_f64(*prct))
                .collect(),
        })
    }

    /// Returns the current delay configuration.
    ///
    /// Uses safe bounds checking to prevent panics even if the delay index
    /// is somehow out of bounds (which would indicate a bug). Returns zero
    /// delay if the delay table is empty or index is invalid.
    ///
    /// # Signal Safety
    ///
    /// This function is async-signal-safe: it only uses atomic loads and
    /// array indexing on a pre-allocated, read-only table.
    pub(crate) fn delay_details(&self) -> DelayDetails {
        let global_delay = self.global_delay_count.load(Ordering::Relaxed);
        let index = self.delay_index.load(Ordering::Relaxed);

        // Use safe index access to prevent panic in signal handler context
        // Falls back to zero delay if table is empty or index invalid
        let current_delay = self
            .delay_table
            .get(index)
            .or_else(|| self.delay_table.last())
            .copied()
            .unwrap_or(Duration::ZERO);

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
    ///   If out of bounds, it will be clamped to the maximum valid index.
    #[allow(dead_code)]
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
    #[allow(dead_code)]
    pub(crate) fn stop_experiment(&self) {
        self.is_active.store(false, Ordering::Release);
        self.selected_ip
            .store(std::ptr::null_mut(), Ordering::Release);
    }

    /// Returns the progress point with the given name, creating one if necessary.
    ///
    /// This function acquires a lock and should NOT be called from signal handlers.
    ///
    /// # Panics
    ///
    /// Panics if the `progress_points` lock is poisoned. This can only occur if
    /// another thread panicked while holding the lock, which indicates a bug.
    #[allow(clippy::expect_used)]
    pub(crate) fn progress(&self, name: &'static str) -> Arc<Progress> {
        // Fast path: check if it exists with a read lock
        {
            let read_guard = self
                .progress_points
                .read()
                .expect("progress_points lock poisoned");
            if let Some(progress) = read_guard.get(name) {
                return Arc::clone(progress);
            }
        }

        // Slow path: acquire write lock and insert
        let mut write_guard = self
            .progress_points
            .write()
            .expect("progress_points lock poisoned");

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
                // This sample is at the selected line - increment delay count
                // to trigger delays in other threads (virtual speedup)
                self.selected_samples.fetch_add(1, Ordering::Relaxed);
                self.global_delay_count.fetch_add(1, Ordering::Relaxed);
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

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        /// Property: Any valid index into DELAY_PRCNT produces a non-negative value
        #[test]
        fn delay_prcnt_all_values_non_negative(idx in 0..DELAY_PRCNT.len()) {
            prop_assert!(DELAY_PRCNT[idx] >= 0.0);
        }

        /// Property: Indices 0-24 always produce 0.0
        #[test]
        fn delay_prcnt_baseline_indices_are_zero(idx in 0..25_usize) {
            prop_assert!((DELAY_PRCNT[idx]).abs() < f64::EPSILON);
        }

        /// Property: Indices 25-54 produce increasing values
        #[test]
        fn delay_prcnt_speedup_indices_increase(idx in 26..55_usize) {
            prop_assert!(DELAY_PRCNT[idx] > DELAY_PRCNT[idx - 1]);
        }

        /// Property: Sample counts increment correctly for any IP
        #[test]
        fn sample_counts_increment_any_ip(ip in 1..usize::MAX) {
            let counts = SampleCounts::new();
            counts.increment(ip);
            let entries = counts.entries();
            prop_assert!(entries.iter().any(|&(stored_ip, count)| stored_ip == ip && count >= 1));
        }

        /// Property: Multiple increments to same IP accumulate
        #[test]
        fn sample_counts_accumulate(ip in 1..usize::MAX, n in 1..100_u64) {
            let counts = SampleCounts::new();
            for _ in 0..n {
                counts.increment(ip);
            }
            let entries = counts.entries();
            let count = entries.iter()
                .find(|&&(stored_ip, _)| stored_ip == ip)
                .map(|&(_, c)| c)
                .unwrap_or(0);
            prop_assert_eq!(count, n);
        }

        /// Property: Hash function always produces valid bucket index
        #[test]
        fn hash_produces_valid_index(ip in 0..usize::MAX) {
            let bucket = SampleCounts::hash(ip);
            prop_assert!(bucket < SAMPLE_COUNT_BUCKETS);
        }

        /// Property: Delay table index clamping works correctly
        #[test]
        fn delay_index_clamps_correctly(idx in 0..1000_usize) {
            let delay_table: Vec<Duration> = DELAY_PRCNT
                .iter()
                .map(|prct| DELAY_BASELINE.mul_f64(*prct))
                .collect();

            let safe_index = idx.min(delay_table.len().saturating_sub(1));
            let delay = delay_table.get(safe_index)
                .or_else(|| delay_table.last())
                .copied()
                .unwrap_or(Duration::ZERO);

            // Should never panic and always return a valid duration
            prop_assert!(delay <= Duration::from_millis(15)); // Max is 150% of 10ms
        }
    }
}
