//! End-to-end validation tests for causal profiling.
//!
//! These tests verify that the core causal profiling mechanism works correctly:
//! - Delay injection at progress points
//! - Throughput measurement under varying delays
//! - Causal impact calculation
//!
//! Run with: cargo test --test causal_validation
//!
//! Note: Some tests require hardware performance counters to be available.
//! If you see test failures related to perf events, ensure:
//! - kernel.perf_event_paranoid <= 1 (sudo sysctl kernel.perf_event_paranoid=1)
//! - You have read access to /proc/sys/kernel/perf_event_paranoid

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Simple thread-local delay accumulator for testing.
/// This mimics the PerThreadProfiler's delay mechanism.
struct DelayAccumulator {
    pending_ns: AtomicU64,
}

impl DelayAccumulator {
    fn new() -> Self {
        Self {
            pending_ns: AtomicU64::new(0),
        }
    }

    fn add_delay(&self, ns: u64) {
        self.pending_ns.fetch_add(ns, Ordering::Relaxed);
    }

    fn take_pending_delay(&self) -> u64 {
        self.pending_ns.swap(0, Ordering::Acquire)
    }
}

/// Progress counter that optionally consumes accumulated delays.
struct TestProgress {
    visits: AtomicU64,
    first_visit_ns: AtomicU64,
    last_visit_ns: AtomicU64,
    delay_accumulator: Option<Arc<DelayAccumulator>>,
}

impl TestProgress {
    fn new() -> Self {
        Self {
            visits: AtomicU64::new(0),
            first_visit_ns: AtomicU64::new(0),
            last_visit_ns: AtomicU64::new(0),
            delay_accumulator: None,
        }
    }

    fn with_delay(delay_accumulator: Arc<DelayAccumulator>) -> Self {
        Self {
            visits: AtomicU64::new(0),
            first_visit_ns: AtomicU64::new(0),
            last_visit_ns: AtomicU64::new(0),
            delay_accumulator: Some(delay_accumulator),
        }
    }

    fn monotonic_nanos() -> u64 {
        let mut ts = libc::timespec {
            tv_sec: 0,
            tv_nsec: 0,
        };
        unsafe {
            libc::clock_gettime(libc::CLOCK_MONOTONIC, &raw mut ts);
        }
        #[allow(clippy::cast_sign_loss)]
        {
            (ts.tv_sec as u64) * 1_000_000_000 + (ts.tv_nsec as u64)
        }
    }

    fn nanosleep(duration: Duration) {
        let ts = libc::timespec {
            #[allow(clippy::cast_possible_wrap)]
            tv_sec: duration.as_secs() as libc::time_t,
            #[allow(clippy::cast_possible_wrap)]
            tv_nsec: i64::from(duration.subsec_nanos()),
        };
        unsafe {
            libc::nanosleep(&ts, std::ptr::null_mut());
        }
    }

    fn note_visit(&self) {
        // Consume pending delay (Coz mechanism)
        if let Some(ref acc) = self.delay_accumulator {
            let pending = acc.take_pending_delay();
            if pending > 0 {
                Self::nanosleep(Duration::from_nanos(pending));
            }
        }

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

    #[allow(dead_code)]
    fn reset(&self) {
        self.visits.store(0, Ordering::Release);
        self.first_visit_ns.store(0, Ordering::Release);
        self.last_visit_ns.store(0, Ordering::Release);
    }
}

// =============================================================================
// DELAY INJECTION MECHANISM TESTS
// =============================================================================

/// Test: Delay accumulation works correctly.
///
/// Verifies that delays can be added and consumed atomically.
#[test]
fn delay_accumulation_basic() {
    let acc = DelayAccumulator::new();

    // Add multiple delays
    acc.add_delay(1_000_000); // 1ms
    acc.add_delay(2_000_000); // 2ms
    acc.add_delay(500_000); // 0.5ms

    // Take should return total
    let total = acc.take_pending_delay();
    assert_eq!(total, 3_500_000, "expected 3.5ms total delay");

    // Second take should return zero
    let second = acc.take_pending_delay();
    assert_eq!(second, 0, "balance should be cleared");
}

/// Test: Delay injection affects throughput.
///
/// This is the core Coz mechanism: injecting delays should reduce throughput.
#[test]
fn delay_injection_reduces_throughput() {
    let iterations = 100;
    let work_delay = Duration::from_millis(2);

    // Baseline: no injected delays
    let baseline_progress = TestProgress::new();
    for _ in 0..iterations {
        std::thread::sleep(work_delay);
        baseline_progress.note_visit();
    }
    let baseline_throughput = baseline_progress.throughput();

    // With injected delays: 5ms delay per visit
    let acc = Arc::new(DelayAccumulator::new());
    let delayed_progress = TestProgress::with_delay(Arc::clone(&acc));

    for _ in 0..iterations {
        std::thread::sleep(work_delay);
        acc.add_delay(5_000_000); // 5ms before each visit
        delayed_progress.note_visit();
    }
    let delayed_throughput = delayed_progress.throughput();

    // Delayed throughput should be significantly lower
    assert!(
        delayed_throughput < baseline_throughput * 0.6,
        "delayed ({delayed_throughput:.2}) should be < 60% of baseline ({baseline_throughput:.2})"
    );

    // Expected ratio: work/(work+delay) = 2ms/(2ms+5ms) ≈ 0.29
    let ratio = delayed_throughput / baseline_throughput;
    assert!(
        ratio > 0.2 && ratio < 0.5,
        "ratio {ratio:.2} should be between 0.2 and 0.5"
    );
}

/// Test: Variable delay amounts produce proportional throughput changes.
///
/// More delay should mean proportionally lower throughput.
#[test]
fn delay_amount_proportional_to_throughput() {
    let iterations = 50;
    let work_delay = Duration::from_millis(5);

    let mut throughputs = Vec::new();

    for delay_ms in [0, 2, 5, 10] {
        let acc = Arc::new(DelayAccumulator::new());
        let progress = TestProgress::with_delay(Arc::clone(&acc));

        for _ in 0..iterations {
            std::thread::sleep(work_delay);
            if delay_ms > 0 {
                acc.add_delay(delay_ms * 1_000_000);
            }
            progress.note_visit();
        }

        throughputs.push((delay_ms, progress.throughput()));
    }

    // Throughput should decrease with more delay
    for i in 1..throughputs.len() {
        let (prev_delay, prev_throughput) = throughputs[i - 1];
        let (curr_delay, curr_throughput) = throughputs[i];
        assert!(
            curr_throughput < prev_throughput,
            "delay {curr_delay}ms throughput ({curr_throughput:.2}) should be < delay {prev_delay}ms throughput ({prev_throughput:.2})"
        );
    }

    // Check approximate proportionality
    // work/(work+delay) ratio should roughly match throughput ratio
    let t_0 = throughputs[0].1;
    let t_10 = throughputs[3].1;

    // Expected: 5ms/(5ms+0ms) / 5ms/(5ms+10ms) = 1.0 / 0.33 ≈ 3.0
    let actual_ratio = t_0 / t_10;
    assert!(
        actual_ratio > 2.0 && actual_ratio < 4.0,
        "throughput ratio {actual_ratio:.2} should be between 2 and 4"
    );
}

// =============================================================================
// CONCURRENT DELAY INJECTION TESTS
// =============================================================================

/// Test: Delay injection works correctly under concurrent access.
///
/// Multiple threads adding delays should not lose any.
#[test]
fn concurrent_delay_accumulation() {
    let acc = Arc::new(DelayAccumulator::new());
    let num_threads = 4;
    let adds_per_thread = 10_000;
    let delay_per_add = 100_u64;

    let handles: Vec<_> = (0..num_threads)
        .map(|_| {
            let acc = Arc::clone(&acc);
            std::thread::spawn(move || {
                for _ in 0..adds_per_thread {
                    acc.add_delay(delay_per_add);
                }
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }

    let total = acc.take_pending_delay();
    let expected = num_threads * adds_per_thread * delay_per_add;
    assert_eq!(
        total,
        expected,
        "expected {expected}, got {total} (lost {} ns)",
        expected - total
    );
}

/// Test: Concurrent delay consumption and accumulation.
///
/// One thread adds delays while another consumes them.
#[test]
fn concurrent_add_and_consume() {
    let acc = Arc::new(DelayAccumulator::new());
    let running = Arc::new(AtomicBool::new(true));

    let total_added = Arc::new(AtomicU64::new(0));
    let total_consumed = Arc::new(AtomicU64::new(0));

    // Producer thread
    let acc_producer = Arc::clone(&acc);
    let running_producer = Arc::clone(&running);
    let total_added_clone = Arc::clone(&total_added);
    let producer = std::thread::spawn(move || {
        let mut added = 0_u64;
        while running_producer.load(Ordering::Relaxed) {
            acc_producer.add_delay(1000);
            added += 1000;
            std::thread::yield_now();
        }
        total_added_clone.store(added, Ordering::Release);
    });

    // Consumer thread
    let acc_consumer = Arc::clone(&acc);
    let running_consumer = Arc::clone(&running);
    let total_consumed_clone = Arc::clone(&total_consumed);
    let consumer = std::thread::spawn(move || {
        let mut consumed = 0_u64;
        while running_consumer.load(Ordering::Relaxed) {
            consumed += acc_consumer.take_pending_delay();
            std::thread::yield_now();
        }
        // Drain any remaining
        consumed += acc_consumer.take_pending_delay();
        total_consumed_clone.store(consumed, Ordering::Release);
    });

    // Let them run for a bit
    std::thread::sleep(Duration::from_millis(100));
    running.store(false, Ordering::Release);

    producer.join().unwrap();
    consumer.join().unwrap();

    let added = total_added.load(Ordering::Acquire);
    let consumed = total_consumed.load(Ordering::Acquire);

    // Consumed should equal added (nothing lost)
    assert_eq!(
        added,
        consumed,
        "added {added} != consumed {consumed}, lost {} ns",
        added.saturating_sub(consumed)
    );
}

// =============================================================================
// THROUGHPUT MEASUREMENT ACCURACY TESTS
// =============================================================================

/// Test: Throughput measurement is accurate with delays.
///
/// When we inject known delays, the measured throughput should match
/// the expected theoretical throughput.
#[test]
fn throughput_accuracy_with_delays() {
    let work_ns = 5_000_000_u64; // 5ms work per iteration
    let delay_ns = 10_000_000_u64; // 10ms delay per iteration
    let iterations = 50;

    let acc = Arc::new(DelayAccumulator::new());
    let progress = TestProgress::with_delay(Arc::clone(&acc));

    let start = Instant::now();
    for _ in 0..iterations {
        TestProgress::nanosleep(Duration::from_nanos(work_ns));
        acc.add_delay(delay_ns);
        progress.note_visit();
    }
    let wall_time = start.elapsed();

    let measured_throughput = progress.throughput();

    // Expected throughput based on work + delay per iteration
    let expected_time_per_iter_ns = work_ns + delay_ns;
    #[allow(clippy::cast_precision_loss)]
    let expected_throughput = 1_000_000_000.0 / expected_time_per_iter_ns as f64;

    // Should be within 15% (accounting for scheduling variance)
    let error = ((measured_throughput - expected_throughput) / expected_throughput).abs();
    assert!(
        error < 0.15,
        "measured {measured_throughput:.2} vs expected {expected_throughput:.2}, error {:.1}%, wall time {wall_time:?}",
        error * 100.0
    );
}

// =============================================================================
// VIRTUAL SPEEDUP SIMULATION TESTS
// =============================================================================

/// Test: Virtual speedup simulation matches Coz model.
///
/// When we "virtually speed up" code by NOT delaying threads that hit it,
/// throughput should increase proportionally to how much time that code
/// represents of the total.
#[test]
fn virtual_speedup_model() {
    let iterations = 100;
    let fast_work = Duration::from_millis(2);
    let slow_work = Duration::from_millis(8); // The "bottleneck"

    // Baseline: no virtual speedup
    // Total time per iteration: fast + slow = 10ms
    // Expected throughput: ~100 ops/sec
    let baseline_progress = TestProgress::new();
    for _ in 0..iterations {
        std::thread::sleep(fast_work);
        std::thread::sleep(slow_work);
        baseline_progress.note_visit();
    }
    let baseline_throughput = baseline_progress.throughput();

    // Simulate 50% speedup of slow_work by delaying threads that DON'T hit it
    // If slow_work were 50% faster (4ms instead of 8ms), total would be 6ms
    // To simulate, we ADD delay to non-bottleneck code
    // Delay amount = speedup_pct * bottleneck_time = 0.5 * 8ms = 4ms
    let speedup_pct = 0.5;
    #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
    let delay_ns = (speedup_pct * slow_work.as_nanos() as f64) as u64;

    let acc = Arc::new(DelayAccumulator::new());
    let speedup_progress = TestProgress::with_delay(Arc::clone(&acc));

    for _ in 0..iterations {
        // This represents "fast" code that should be delayed
        std::thread::sleep(fast_work);
        acc.add_delay(delay_ns);
        speedup_progress.note_visit();

        // Slow work is NOT delayed (it's the "selected" code being sped up)
        std::thread::sleep(slow_work);
    }
    let speedup_throughput = speedup_progress.throughput();

    // With virtual speedup, throughput should be LOWER because we're
    // adding delays to simulate what would happen if bottleneck were faster
    // (non-bottleneck code would take relatively more time)
    assert!(
        speedup_throughput < baseline_throughput,
        "speedup {speedup_throughput:.2} should be < baseline {baseline_throughput:.2}"
    );

    // The ratio tells us the "causal impact" of the slow_work
    // Lower ratio = higher impact (optimizing it helps more)
    let ratio = speedup_throughput / baseline_throughput;

    // slow_work is 80% of total time (8ms out of 10ms)
    // A 50% speedup of slow_work should reduce total time by 40%
    // So throughput ratio should be around 60%
    assert!(
        ratio > 0.45 && ratio < 0.75,
        "ratio {ratio:.2} should be between 0.45 and 0.75"
    );
}

/// Test: Causal impact correlates with bottleneck severity.
///
/// Code that represents more of the runtime should have higher causal impact.
#[test]
fn causal_impact_correlation() {
    let iterations = 50;

    // Test different bottleneck severities
    // bottleneck_pct: what fraction of work is in the "bottleneck"
    let test_cases = vec![
        (0.2, 2, 8), // 20% bottleneck: 2ms bottleneck, 8ms other
        (0.5, 5, 5), // 50% bottleneck: 5ms each
        (0.8, 8, 2), // 80% bottleneck: 8ms bottleneck, 2ms other
    ];

    let mut results = Vec::new();

    for (bottleneck_pct, bottleneck_ms, other_ms) in test_cases {
        let bottleneck_work = Duration::from_millis(bottleneck_ms);
        let other_work = Duration::from_millis(other_ms);

        // Measure baseline throughput
        let baseline_progress = TestProgress::new();
        for _ in 0..iterations {
            std::thread::sleep(bottleneck_work);
            std::thread::sleep(other_work);
            baseline_progress.note_visit();
        }
        let baseline = baseline_progress.throughput();

        // Simulate 50% speedup of bottleneck
        // Delay = 0.5 * bottleneck_time added to OTHER code
        #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
        let delay_ns = (0.5 * bottleneck_work.as_nanos() as f64) as u64;

        let acc = Arc::new(DelayAccumulator::new());
        let speedup_progress = TestProgress::with_delay(Arc::clone(&acc));

        for _ in 0..iterations {
            std::thread::sleep(bottleneck_work);
            // Other code gets delayed
            std::thread::sleep(other_work);
            acc.add_delay(delay_ns);
            speedup_progress.note_visit();
        }
        let speedup = speedup_progress.throughput();

        // Impact = how much throughput changed
        let impact = (baseline - speedup) / baseline;
        results.push((bottleneck_pct, impact));
    }

    // Higher bottleneck percentage should correlate with higher impact
    // (more throughput reduction when we virtually speed it up)
    for i in 1..results.len() {
        let (prev_pct, prev_impact) = results[i - 1];
        let (curr_pct, curr_impact) = results[i];
        assert!(
            curr_impact > prev_impact,
            "{:.0}% bottleneck impact ({:.3}) should be > {:.0}% impact ({:.3})",
            curr_pct * 100.0,
            curr_impact,
            prev_pct * 100.0,
            prev_impact
        );
    }
}

// =============================================================================
// EDGE CASE TESTS
// =============================================================================

/// Test: Zero delay has no effect.
#[test]
fn zero_delay_no_effect() {
    let iterations = 50;
    let work = Duration::from_millis(5);

    // Without delay accumulator
    let no_acc = TestProgress::new();
    for _ in 0..iterations {
        std::thread::sleep(work);
        no_acc.note_visit();
    }

    // With accumulator but zero delays
    let acc = Arc::new(DelayAccumulator::new());
    let with_acc = TestProgress::with_delay(Arc::clone(&acc));
    for _ in 0..iterations {
        std::thread::sleep(work);
        acc.add_delay(0);
        with_acc.note_visit();
    }

    let t1 = no_acc.throughput();
    let t2 = with_acc.throughput();

    // Should be essentially the same (within 5%)
    let diff = ((t1 - t2) / t1).abs();
    assert!(
        diff < 0.05,
        "zero delay should have no effect: {t1:.2} vs {t2:.2}"
    );
}

/// Test: Very small delays are still applied.
#[test]
fn small_delays_are_applied() {
    let iterations = 100;
    let work = Duration::from_millis(1);

    // Without delays
    let baseline = TestProgress::new();
    for _ in 0..iterations {
        std::thread::sleep(work);
        baseline.note_visit();
    }

    // With 1ms delays
    let acc = Arc::new(DelayAccumulator::new());
    let delayed = TestProgress::with_delay(Arc::clone(&acc));
    for _ in 0..iterations {
        std::thread::sleep(work);
        acc.add_delay(1_000_000); // 1ms
        delayed.note_visit();
    }

    let t_base = baseline.throughput();
    let t_delayed = delayed.throughput();

    // Delayed should be noticeably slower
    // Work=1ms, delay=1ms, so throughput should be roughly halved
    let ratio = t_delayed / t_base;
    assert!(
        ratio < 0.7,
        "1ms delay should reduce throughput: ratio {ratio:.2}"
    );
}

/// Test: Rapid delay accumulation and consumption.
#[test]
fn rapid_delay_churn() {
    let acc = Arc::new(DelayAccumulator::new());
    let iterations = 10_000;

    for i in 0..iterations {
        let delay = (i % 100 + 1) as u64 * 1000;
        acc.add_delay(delay);
        let consumed = acc.take_pending_delay();
        assert_eq!(
            consumed, delay,
            "iteration {i}: expected {delay}, got {consumed}"
        );
    }
}

// =============================================================================
// STATISTICAL PROPERTY TESTS
// =============================================================================

/// Test: Multiple measurements are consistent.
///
/// Running the same workload multiple times should give similar results.
#[test]
fn measurement_stability() {
    let measurements: Vec<f64> = (0..5)
        .map(|_| {
            let work = Duration::from_millis(5);
            let delay_ns = 5_000_000_u64;

            let acc = Arc::new(DelayAccumulator::new());
            let progress = TestProgress::with_delay(Arc::clone(&acc));

            for _ in 0..50 {
                std::thread::sleep(work);
                acc.add_delay(delay_ns);
                progress.note_visit();
            }

            progress.throughput()
        })
        .collect();

    // Calculate coefficient of variation
    let mean: f64 = measurements.iter().sum::<f64>() / measurements.len() as f64;
    let variance: f64 =
        measurements.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / measurements.len() as f64;
    let cv = variance.sqrt() / mean;

    assert!(
        cv < 0.10,
        "throughput variance too high: CV={cv:.2}, measurements={measurements:?}"
    );
}
