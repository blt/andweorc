//! Micro-benchmarks for core andweorc operations.
//!
//! These benchmarks measure the performance of critical operations that
//! affect profiling overhead. The goal is to keep overhead minimal so
//! the profiler doesn't significantly perturb the program being measured.

#![allow(missing_docs)]

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

/// Benchmark the hash function used for sample counting.
///
/// This is called on every sample, so it needs to be fast.
fn bench_hash(c: &mut Criterion) {
    // FNV-1a hash implementation (same as in experiment.rs)
    fn hash(ip: usize) -> usize {
        const FNV_OFFSET: usize = 0xcbf2_9ce4_8422_2325_usize;
        const FNV_PRIME: usize = 0x0100_0000_01b3_usize;

        let mut hash = FNV_OFFSET;
        for i in 0..std::mem::size_of::<usize>() {
            let byte = (ip >> (i * 8)) & 0xFF;
            hash ^= byte;
            hash = hash.wrapping_mul(FNV_PRIME);
        }
        hash & 8191 // Mask for 8192 buckets
    }

    c.bench_function("hash_ip", |b| {
        b.iter(|| hash(black_box(0x7fff_beef_cafe_u64 as usize)))
    });
}

/// Benchmark atomic compare-exchange operations.
///
/// This is the core operation for lock-free sample counting.
fn bench_atomic_cas(c: &mut Criterion) {
    let atomic = AtomicU64::new(0);

    c.bench_function("atomic_cas_success", |b| {
        b.iter(|| {
            let old = atomic.load(Ordering::Relaxed);
            let _ = atomic.compare_exchange(old, old + 1, Ordering::Relaxed, Ordering::Relaxed);
        })
    });

    c.bench_function("atomic_fetch_add", |b| {
        b.iter(|| atomic.fetch_add(black_box(1), Ordering::Relaxed))
    });
}

/// Benchmark clock_gettime for monotonic time.
///
/// This is called on every progress point visit.
fn bench_monotonic_time(c: &mut Criterion) {
    c.bench_function("clock_gettime_monotonic", |b| {
        b.iter(|| {
            let mut ts = libc::timespec {
                tv_sec: 0,
                tv_nsec: 0,
            };
            unsafe {
                libc::clock_gettime(libc::CLOCK_MONOTONIC, &raw mut ts);
            }
            black_box(ts.tv_sec as u64 * 1_000_000_000 + ts.tv_nsec as u64)
        })
    });
}

/// Benchmark linear regression calculation.
///
/// This is called when calculating causal impact after experiments complete.
fn bench_linear_regression(c: &mut Criterion) {
    // Simulate experiment results with various data set sizes
    fn calculate_regression(xs: &[f64], ys: &[f64]) -> f64 {
        let n = xs.len() as f64;
        if n < 2.0 {
            return 0.0;
        }

        let sum_x: f64 = xs.iter().sum();
        let sum_y: f64 = ys.iter().sum();
        let mean_x = sum_x / n;
        let mean_y = sum_y / n;

        let mut numerator = 0.0;
        let mut denominator = 0.0;
        for (x, y) in xs.iter().zip(ys.iter()) {
            let dx = x - mean_x;
            let dy = y - mean_y;
            numerator += dx * dy;
            denominator += dx * dx;
        }

        if denominator.abs() < f64::EPSILON {
            return 0.0;
        }

        numerator / denominator
    }

    let mut group = c.benchmark_group("linear_regression");

    for size in [5, 10, 25, 50, 100] {
        // Generate test data: throughput increases linearly with speedup
        let xs: Vec<f64> = (0..size).map(|i| i as f64 * 0.05).collect();
        let ys: Vec<f64> = xs.iter().map(|x| 1000.0 + x * 50.0).collect();

        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(size),
            &(xs, ys),
            |b, (xs, ys)| b.iter(|| calculate_regression(black_box(xs), black_box(ys))),
        );
    }

    group.finish();
}

/// Benchmark delay calculation with table lookup.
///
/// This is called when determining how much to delay non-selected threads.
fn bench_delay_lookup(c: &mut Criterion) {
    // Pre-computed delay table (same structure as in experiment.rs)
    const DELAY_PRCNT: [f64; 55] = [
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45,
        0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15, 1.20,
        1.25, 1.30, 1.35, 1.40, 1.45, 1.50,
    ];

    let baseline = Duration::from_millis(10);
    let delay_table: Vec<Duration> = DELAY_PRCNT
        .iter()
        .map(|prct| baseline.mul_f64(*prct))
        .collect();

    c.bench_function("delay_table_lookup", |b| {
        b.iter(|| {
            let index = black_box(30_usize);
            let safe_index = index.min(delay_table.len().saturating_sub(1));
            delay_table
                .get(safe_index)
                .or_else(|| delay_table.last())
                .copied()
                .unwrap_or(Duration::ZERO)
        })
    });
}

/// Benchmark nanosleep for small durations.
///
/// This is used for delay injection in virtual speedup experiments.
fn bench_nanosleep(c: &mut Criterion) {
    let mut group = c.benchmark_group("nanosleep");

    // Reduce sample count for sleep benchmarks since they're slow
    group.sample_size(10);

    for nanos in [100, 1_000, 10_000, 100_000] {
        let duration = Duration::from_nanos(nanos);
        group.bench_with_input(BenchmarkId::from_parameter(nanos), &duration, |b, dur| {
            b.iter(|| {
                let ts = libc::timespec {
                    tv_sec: dur.as_secs() as libc::time_t,
                    tv_nsec: dur.subsec_nanos() as libc::c_long,
                };
                unsafe {
                    libc::nanosleep(&raw const ts, std::ptr::null_mut());
                }
            })
        });
    }

    group.finish();
}

/// Benchmark thread-local storage access.
///
/// This is used to access per-thread profiler state from signal handlers.
fn bench_thread_local(c: &mut Criterion) {
    use std::cell::Cell;

    thread_local! {
        static TLS_VALUE: Cell<u64> = const { Cell::new(0) };
    }

    c.bench_function("tls_read_write", |b| {
        b.iter(|| {
            TLS_VALUE.with(|cell| {
                let val = cell.get();
                cell.set(val + 1);
                black_box(val)
            })
        })
    });
}

/// Benchmark progress point visit overhead.
///
/// This measures the full cost of a progress point increment, which includes:
/// - Atomic counter increment
/// - Clock_gettime for first/last timestamp tracking
fn bench_progress_point_overhead(c: &mut Criterion) {
    // Simulate progress point state
    struct SimulatedProgress {
        visits: AtomicU64,
        first_visit_ns: AtomicU64,
        last_visit_ns: AtomicU64,
    }

    impl SimulatedProgress {
        fn new() -> Self {
            Self {
                visits: AtomicU64::new(0),
                first_visit_ns: AtomicU64::new(0),
                last_visit_ns: AtomicU64::new(0),
            }
        }

        fn visit(&self) {
            let mut ts = libc::timespec {
                tv_sec: 0,
                tv_nsec: 0,
            };
            unsafe {
                libc::clock_gettime(libc::CLOCK_MONOTONIC, &raw mut ts);
            }
            let now = ts.tv_sec as u64 * 1_000_000_000 + ts.tv_nsec as u64;

            self.visits.fetch_add(1, Ordering::Release);
            let _ =
                self.first_visit_ns
                    .compare_exchange(0, now, Ordering::Release, Ordering::Relaxed);
            self.last_visit_ns.store(now, Ordering::Release);
        }
    }

    let progress = SimulatedProgress::new();

    c.bench_function("progress_point_visit", |b| b.iter(|| progress.visit()));
}

/// Benchmark sample counting overhead (SampleEntry increment).
///
/// This measures the cost of the lock-free hash table increment that
/// happens on every sample in the signal handler.
fn bench_sample_increment_overhead(c: &mut Criterion) {
    use std::sync::atomic::AtomicUsize;

    const BUCKETS: usize = 8192;

    struct SampleEntry {
        ip: AtomicUsize,
        count: AtomicU64,
    }

    impl SampleEntry {
        const fn new() -> Self {
            Self {
                ip: AtomicUsize::new(0),
                count: AtomicU64::new(0),
            }
        }
    }

    // Simplified hash
    fn hash(ip: usize) -> usize {
        const FNV_OFFSET: usize = 0xcbf2_9ce4_8422_2325_usize;
        const FNV_PRIME: usize = 0x0100_0000_01b3_usize;
        let mut h = FNV_OFFSET;
        for i in 0..std::mem::size_of::<usize>() {
            h ^= (ip >> (i * 8)) & 0xFF;
            h = h.wrapping_mul(FNV_PRIME);
        }
        h & (BUCKETS - 1)
    }

    // Create a sample counter
    let entries: Vec<SampleEntry> = (0..BUCKETS).map(|_| SampleEntry::new()).collect();

    // Pre-populate with a known IP to test the "found existing" fast path
    let known_ip = 0xDEAD_BEEF_usize;
    let idx = hash(known_ip);
    entries[idx].ip.store(known_ip, Ordering::Relaxed);

    c.bench_function("sample_increment_existing", |b| {
        b.iter(|| {
            let ip = black_box(known_ip);
            let start = hash(ip);
            for i in 0..BUCKETS {
                let idx = (start + i) & (BUCKETS - 1);
                let current = entries[idx].ip.load(Ordering::Relaxed);
                if current == ip {
                    entries[idx].count.fetch_add(1, Ordering::Relaxed);
                    return;
                }
            }
        })
    });

    // Test the "claim new slot" path
    let new_ip = 0xCAFE_BABE_usize;
    c.bench_function("sample_increment_new", |b| {
        b.iter(|| {
            let ip = black_box(new_ip);
            let start = hash(ip);
            for i in 0..BUCKETS {
                let idx = (start + i) & (BUCKETS - 1);
                let current = entries[idx].ip.load(Ordering::Relaxed);
                if current == ip {
                    entries[idx].count.fetch_add(1, Ordering::Relaxed);
                    return;
                }
                if current == 0 {
                    match entries[idx].ip.compare_exchange(
                        0,
                        ip,
                        Ordering::Release,
                        Ordering::Relaxed,
                    ) {
                        Ok(_) => {
                            entries[idx].count.fetch_add(1, Ordering::Relaxed);
                            return;
                        }
                        Err(actual) if actual == ip => {
                            entries[idx].count.fetch_add(1, Ordering::Relaxed);
                            return;
                        }
                        Err(_) => {}
                    }
                }
            }
        })
    });
}

/// Benchmark combined profiler overhead per sample.
///
/// This estimates the total overhead the profiler adds when a sample is taken:
/// - Hash calculation
/// - Sample table increment
/// - TLS access pattern
fn bench_combined_sample_overhead(c: &mut Criterion) {
    use std::cell::RefCell;
    use std::sync::atomic::AtomicUsize;

    const BUCKETS: usize = 8192;

    struct SampleEntry {
        ip: AtomicUsize,
        count: AtomicU64,
    }

    thread_local! {
        static PROFILER_STATE: RefCell<u64> = const { RefCell::new(0) };
    }

    // Simulate the full sample processing path
    let entries: Vec<SampleEntry> = (0..BUCKETS)
        .map(|_| SampleEntry {
            ip: AtomicUsize::new(0),
            count: AtomicU64::new(0),
        })
        .collect();

    // Pre-populate some IPs
    for i in 0..100 {
        let ip = 0x1000_0000 + i * 0x1000;
        let idx = {
            const FNV_OFFSET: usize = 0xcbf2_9ce4_8422_2325_usize;
            const FNV_PRIME: usize = 0x0100_0000_01b3_usize;
            let mut h = FNV_OFFSET;
            for j in 0..std::mem::size_of::<usize>() {
                h ^= (ip >> (j * 8)) & 0xFF;
                h = h.wrapping_mul(FNV_PRIME);
            }
            h & (BUCKETS - 1)
        };
        entries[idx].ip.store(ip, Ordering::Relaxed);
    }

    c.bench_function("combined_sample_overhead", |b| {
        let mut ip_counter = 0_usize;
        b.iter(|| {
            // Simulate getting an IP from a sample (cycle through known IPs)
            let ip = 0x1000_0000 + (ip_counter % 100) * 0x1000;
            ip_counter += 1;

            // Hash the IP
            let start = {
                const FNV_OFFSET: usize = 0xcbf2_9ce4_8422_2325_usize;
                const FNV_PRIME: usize = 0x0100_0000_01b3_usize;
                let mut h = FNV_OFFSET;
                for i in 0..std::mem::size_of::<usize>() {
                    h ^= (ip >> (i * 8)) & 0xFF;
                    h = h.wrapping_mul(FNV_PRIME);
                }
                h & (BUCKETS - 1)
            };

            // Increment sample count (lock-free)
            for i in 0..BUCKETS {
                let idx = (start + i) & (BUCKETS - 1);
                let current = entries[idx].ip.load(Ordering::Relaxed);
                if current == ip {
                    entries[idx].count.fetch_add(1, Ordering::Relaxed);
                    break;
                }
            }

            // TLS access (simulating per-thread state check)
            PROFILER_STATE.with(|state| {
                *state.borrow_mut() += 1;
            });
        })
    });
}

criterion_group!(
    benches,
    bench_hash,
    bench_atomic_cas,
    bench_monotonic_time,
    bench_linear_regression,
    bench_delay_lookup,
    bench_nanosleep,
    bench_thread_local,
    bench_progress_point_overhead,
    bench_sample_increment_overhead,
    bench_combined_sample_overhead,
);

criterion_main!(benches);
