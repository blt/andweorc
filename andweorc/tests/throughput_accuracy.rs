//! Integration tests for throughput measurement accuracy.
//!
//! These tests verify that the core measurement mechanisms produce
//! accurate results, which is essential for causal profiling correctness.
//!
//! Run with: cargo test --test throughput_accuracy

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Simple progress counter for testing throughput measurement.
struct TestProgress {
    visits: AtomicU64,
    first_visit_ns: AtomicU64,
    last_visit_ns: AtomicU64,
}

impl TestProgress {
    fn new() -> Self {
        Self {
            visits: AtomicU64::new(0),
            first_visit_ns: AtomicU64::new(0),
            last_visit_ns: AtomicU64::new(0),
        }
    }

    fn monotonic_nanos() -> u64 {
        let mut ts = libc::timespec {
            tv_sec: 0,
            tv_nsec: 0,
        };
        unsafe {
            libc::clock_gettime(libc::CLOCK_MONOTONIC, &mut ts as *mut _);
        }
        #[allow(clippy::cast_sign_loss)]
        {
            (ts.tv_sec as u64) * 1_000_000_000 + (ts.tv_nsec as u64)
        }
    }

    fn note_visit(&self) {
        let now = Self::monotonic_nanos();
        self.visits.fetch_add(1, Ordering::Release);
        let _ = self
            .first_visit_ns
            .compare_exchange(0, now, Ordering::Release, Ordering::Relaxed);
        self.last_visit_ns.store(now, Ordering::Release);
    }

    fn throughput(&self) -> f64 {
        let visits = self.visits.load(Ordering::Acquire);
        if visits < 2 {
            return 0.0;
        }

        let first = self.first_visit_ns.load(Ordering::Acquire);
        let last = self.last_visit_ns.load(Ordering::Acquire);
        if first == 0 || last == 0 || last <= first {
            return 0.0;
        }

        let elapsed_ns = last - first;
        #[allow(clippy::cast_precision_loss)]
        {
            f64::from((visits - 1) as u32) / (elapsed_ns as f64 / 1_000_000_000.0)
        }
    }

    fn reset(&self) {
        self.visits.store(0, Ordering::Release);
        self.first_visit_ns.store(0, Ordering::Release);
        self.last_visit_ns.store(0, Ordering::Release);
    }
}

/// Test: Throughput measurement with known rate.
///
/// This test creates a workload with a known iteration rate and verifies
/// that the measured throughput is within acceptable bounds.
#[test]
fn throughput_matches_known_rate() {
    let progress = TestProgress::new();
    let target_ops_per_sec = 1000.0;
    let delay_per_op = Duration::from_secs_f64(1.0 / target_ops_per_sec);
    let iterations = 100;

    // Warm up
    for _ in 0..10 {
        progress.note_visit();
        std::thread::sleep(delay_per_op);
    }
    progress.reset();

    // Run measured iterations
    let start = Instant::now();
    for _ in 0..iterations {
        progress.note_visit();
        std::thread::sleep(delay_per_op);
    }
    let elapsed = start.elapsed();

    let measured_throughput = progress.throughput();
    let expected_throughput = f64::from(iterations - 1) / elapsed.as_secs_f64();

    // Throughput should be within 20% of expected (accounting for timing jitter)
    let relative_error = ((measured_throughput - expected_throughput) / expected_throughput).abs();
    assert!(
        relative_error < 0.2,
        "Throughput error too high: measured={measured_throughput:.2}, expected={expected_throughput:.2}, error={:.1}%",
        relative_error * 100.0
    );
}

/// Test: Concurrent progress point updates don't lose counts.
///
/// This tests the correctness of lock-free progress point updates
/// under concurrent access.
#[test]
fn concurrent_updates_no_lost_counts() {
    let progress = Arc::new(TestProgress::new());
    let iterations_per_thread = 10_000;
    let num_threads = 4;

    let handles: Vec<_> = (0..num_threads)
        .map(|_| {
            let p = Arc::clone(&progress);
            std::thread::spawn(move || {
                for _ in 0..iterations_per_thread {
                    p.note_visit();
                }
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }

    let expected_visits = num_threads * iterations_per_thread;
    let actual_visits = progress.visits.load(Ordering::Acquire);
    assert_eq!(
        actual_visits, expected_visits as u64,
        "Lost counts: expected={expected_visits}, got={actual_visits}"
    );
}

/// Test: Throughput is monotonically related to work rate.
///
/// This verifies the fundamental property that more work per unit time
/// produces higher measured throughput.
#[test]
fn faster_work_higher_throughput() {
    let slow_progress = TestProgress::new();
    let fast_progress = TestProgress::new();

    let slow_delay = Duration::from_millis(10);
    let fast_delay = Duration::from_millis(2);
    let iterations = 50;

    // Measure slow rate
    for _ in 0..iterations {
        slow_progress.note_visit();
        std::thread::sleep(slow_delay);
    }
    let slow_throughput = slow_progress.throughput();

    // Measure fast rate
    for _ in 0..iterations {
        fast_progress.note_visit();
        std::thread::sleep(fast_delay);
    }
    let fast_throughput = fast_progress.throughput();

    assert!(
        fast_throughput > slow_throughput,
        "Fast throughput ({fast_throughput:.2}) should be > slow throughput ({slow_throughput:.2})"
    );

    // Fast should be roughly 5x slow (10ms/2ms = 5)
    let ratio = fast_throughput / slow_throughput;
    assert!(
        ratio > 3.0 && ratio < 7.0,
        "Throughput ratio should be ~5x, got {ratio:.2}x"
    );
}

/// Test: Reset actually clears all state.
///
/// Verifies that after reset, the progress point behaves as if newly created.
#[test]
fn reset_clears_state_completely() {
    let progress = TestProgress::new();

    // Accumulate some state
    for _ in 0..100 {
        progress.note_visit();
    }
    assert!(progress.visits.load(Ordering::Acquire) > 0);

    // Reset
    progress.reset();

    // State should be cleared
    assert_eq!(progress.visits.load(Ordering::Acquire), 0);
    assert_eq!(progress.first_visit_ns.load(Ordering::Acquire), 0);
    assert_eq!(progress.last_visit_ns.load(Ordering::Acquire), 0);
    assert!((progress.throughput() - 0.0).abs() < f64::EPSILON);

    // New visits should work correctly
    progress.note_visit();
    std::thread::sleep(Duration::from_millis(10));
    progress.note_visit();

    assert_eq!(progress.visits.load(Ordering::Acquire), 2);
    assert!(progress.throughput() > 0.0);
}

/// Test: Throughput stability over time.
///
/// Verifies that repeated measurements of the same workload produce
/// consistent results (low variance).
#[test]
fn throughput_is_stable() {
    let measurements: Vec<f64> = (0..5)
        .map(|_| {
            let progress = TestProgress::new();
            let delay = Duration::from_millis(5);

            for _ in 0..50 {
                progress.note_visit();
                std::thread::sleep(delay);
            }
            progress.throughput()
        })
        .collect();

    // Calculate mean and standard deviation
    let mean: f64 = measurements.iter().sum::<f64>() / measurements.len() as f64;
    let variance: f64 =
        measurements.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / measurements.len() as f64;
    let stddev = variance.sqrt();
    let cv = stddev / mean; // Coefficient of variation

    assert!(
        cv < 0.15,
        "Throughput too variable: mean={mean:.2}, stddev={stddev:.2}, CV={cv:.2}"
    );
}

/// Test: Synthetic bottleneck detection.
///
/// Creates a workload with a known "bottleneck" (slow operation) and verifies
/// that throughput increases when the bottleneck is removed.
#[test]
fn bottleneck_detection_synthetic() {
    let progress = TestProgress::new();
    let running = Arc::new(AtomicBool::new(true));
    let iterations = 100;

    // Workload with bottleneck (fast + slow operations)
    let bottleneck_delay = Duration::from_millis(10);
    let fast_delay = Duration::from_millis(1);

    for _ in 0..iterations {
        // Fast operation
        std::thread::sleep(fast_delay);
        // Bottleneck operation
        std::thread::sleep(bottleneck_delay);
        progress.note_visit();
    }
    let throughput_with_bottleneck = progress.throughput();

    progress.reset();

    // Same workload without bottleneck
    for _ in 0..iterations {
        // Fast operation (same)
        std::thread::sleep(fast_delay);
        // No bottleneck
        progress.note_visit();
    }
    let throughput_without_bottleneck = progress.throughput();

    // Removing the bottleneck should increase throughput
    assert!(
        throughput_without_bottleneck > throughput_with_bottleneck,
        "Removing bottleneck should increase throughput: with={throughput_with_bottleneck:.2}, without={throughput_without_bottleneck:.2}"
    );

    // The speedup should be significant (roughly proportional to bottleneck size)
    let speedup = throughput_without_bottleneck / throughput_with_bottleneck;
    let _ = running; // Prevent unused warning
    assert!(
        speedup > 5.0,
        "Speedup should be significant (>5x), got {speedup:.2}x"
    );
}
