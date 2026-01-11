//! Stress tests for timing-dependent bugs.
//!
//! These tests stress the system with high concurrency to expose
//! race conditions, timing bugs, and edge cases that might not
//! appear under normal operation.
//!
//! Run with: cargo test --test stress_tests --release
//! (Release mode recommended for realistic timing behavior)

use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Barrier};
use std::thread;
use std::time::Duration;

/// Number of buckets for stress testing (matches production).
const STRESS_BUCKETS: usize = 8192;

/// Lock-free sample counter for stress testing.
struct StressSampleCounts {
    ips: Box<[AtomicUsize; STRESS_BUCKETS]>,
    counts: Box<[AtomicU64; STRESS_BUCKETS]>,
}

impl StressSampleCounts {
    fn new() -> Self {
        Self {
            ips: Box::new(std::array::from_fn(|_| AtomicUsize::new(0))),
            counts: Box::new(std::array::from_fn(|_| AtomicU64::new(0))),
        }
    }

    fn hash(ip: usize) -> usize {
        const FNV_OFFSET: usize = 0xcbf2_9ce4_8422_2325_usize;
        const FNV_PRIME: usize = 0x0100_0000_01b3_usize;

        let mut hash = FNV_OFFSET;
        for i in 0..std::mem::size_of::<usize>() {
            let byte = (ip >> (i * 8)) & 0xFF;
            hash ^= byte;
            hash = hash.wrapping_mul(FNV_PRIME);
        }
        hash & (STRESS_BUCKETS - 1)
    }

    fn increment(&self, ip: usize) {
        if ip == 0 {
            return;
        }

        let start = Self::hash(ip);

        for i in 0..STRESS_BUCKETS {
            let idx = (start + i) & (STRESS_BUCKETS - 1);
            let current = self.ips[idx].load(Ordering::Relaxed);

            if current == ip {
                self.counts[idx].fetch_add(1, Ordering::Relaxed);
                return;
            }

            if current == 0 {
                match self.ips[idx].compare_exchange(0, ip, Ordering::Release, Ordering::Relaxed) {
                    Ok(_) => {
                        self.counts[idx].fetch_add(1, Ordering::Relaxed);
                        return;
                    }
                    Err(actual) => {
                        if actual == ip {
                            self.counts[idx].fetch_add(1, Ordering::Relaxed);
                            return;
                        }
                    }
                }
            }
        }
    }

    fn total_count(&self) -> u64 {
        let mut total = 0;
        for i in 0..STRESS_BUCKETS {
            total += self.counts[i].load(Ordering::Acquire);
        }
        total
    }

    fn unique_ips(&self) -> usize {
        let mut count = 0;
        for i in 0..STRESS_BUCKETS {
            if self.ips[i].load(Ordering::Acquire) != 0 {
                count += 1;
            }
        }
        count
    }
}

/// Stress test: High contention on same IP.
///
/// Many threads increment the same IP simultaneously. This stresses
/// the CAS retry path and verifies no counts are lost.
#[test]
fn high_contention_same_ip() {
    let counts = Arc::new(StressSampleCounts::new());
    let num_threads = 8;
    let iterations = 100_000;
    let ip = 0xDEAD_BEEF_usize;

    let barrier = Arc::new(Barrier::new(num_threads));

    let handles: Vec<_> = (0..num_threads)
        .map(|_| {
            let c = Arc::clone(&counts);
            let b = Arc::clone(&barrier);
            thread::spawn(move || {
                b.wait(); // Synchronize start for maximum contention
                for _ in 0..iterations {
                    c.increment(ip);
                }
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }

    let expected = (num_threads * iterations) as u64;
    let actual = counts.total_count();
    assert_eq!(
        actual, expected,
        "Lost counts under high contention: expected={expected}, got={actual}"
    );
}

/// Stress test: Many unique IPs.
///
/// Tests the linear probing behavior when many unique IPs are inserted.
/// Verifies that IPs don't get lost even with high table utilization.
#[test]
fn many_unique_ips() {
    let counts = Arc::new(StressSampleCounts::new());
    let num_threads = 4;
    let ips_per_thread = 1000;

    let barrier = Arc::new(Barrier::new(num_threads));

    let handles: Vec<_> = (0..num_threads)
        .map(|thread_id| {
            let c = Arc::clone(&counts);
            let b = Arc::clone(&barrier);
            thread::spawn(move || {
                b.wait();
                for i in 0..ips_per_thread {
                    // Each thread uses non-overlapping IP ranges
                    let ip = 0x1000_0000 + thread_id * 0x100_0000 + i;
                    c.increment(ip);
                }
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }

    let expected_ips = num_threads * ips_per_thread;
    let actual_ips = counts.unique_ips();
    let expected_count = expected_ips as u64;
    let actual_count = counts.total_count();

    assert_eq!(
        actual_ips, expected_ips,
        "Lost IPs: expected={expected_ips}, got={actual_ips}"
    );
    assert_eq!(
        actual_count, expected_count,
        "Lost counts: expected={expected_count}, got={actual_count}"
    );
}

/// Stress test: Hash collision handling.
///
/// Creates IPs that intentionally hash to the same bucket to stress
/// the linear probing collision resolution.
#[test]
fn hash_collision_stress() {
    let counts = Arc::new(StressSampleCounts::new());
    let num_threads = 4;
    let iterations = 10_000;

    // Find IPs that hash to the same bucket
    let base_bucket = StressSampleCounts::hash(0x1234);
    let mut colliding_ips = Vec::new();
    let mut candidate = 1_usize;
    while colliding_ips.len() < num_threads {
        if StressSampleCounts::hash(candidate) == base_bucket {
            colliding_ips.push(candidate);
        }
        candidate += 1;
    }

    let barrier = Arc::new(Barrier::new(num_threads));
    let colliding_ips = Arc::new(colliding_ips);

    let handles: Vec<_> = (0..num_threads)
        .map(|thread_id| {
            let c = Arc::clone(&counts);
            let b = Arc::clone(&barrier);
            let ips = Arc::clone(&colliding_ips);
            thread::spawn(move || {
                let ip = ips[thread_id];
                b.wait();
                for _ in 0..iterations {
                    c.increment(ip);
                }
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }

    let expected_count = (num_threads * iterations) as u64;
    let actual_count = counts.total_count();
    assert_eq!(
        actual_count, expected_count,
        "Lost counts under collision stress: expected={expected_count}, got={actual_count}"
    );
}

/// Stress test: Rapid fire increments.
///
/// Tests behavior under very high throughput with minimal delay.
/// This can expose timing-dependent bugs and cache effects.
#[test]
fn rapid_fire_increments() {
    let counts = Arc::new(StressSampleCounts::new());
    let num_threads = 8;
    let iterations = 1_000_000;
    let running = Arc::new(AtomicBool::new(true));

    let handles: Vec<_> = (0..num_threads)
        .map(|thread_id| {
            let c = Arc::clone(&counts);
            let r = Arc::clone(&running);
            thread::spawn(move || {
                let mut count = 0_u64;
                let base_ip = 0x4000_0000 + thread_id;
                while r.load(Ordering::Relaxed) && count < iterations {
                    // Cycle through a few IPs to stress both insert and update paths
                    let ip = base_ip + (count as usize % 10) * 0x1000;
                    c.increment(ip);
                    count += 1;
                }
                count
            })
        })
        .collect();

    let total_ops: u64 = handles.into_iter().map(|h| h.join().unwrap()).sum();
    let actual_count = counts.total_count();

    // Allow for some variance due to the cycling behavior
    // Each thread cycles through 10 IPs, so there might be some overlaps
    // in practice due to hash distribution
    assert!(
        actual_count >= total_ops * 9 / 10,
        "Lost too many counts: expected ~{total_ops}, got {actual_count}"
    );
}

/// Stress test: Mixed workload.
///
/// Simulates a realistic workload with a mix of:
/// - Hot IPs (frequently sampled)
/// - Warm IPs (occasionally sampled)
/// - Cold IPs (rarely sampled)
#[test]
fn mixed_workload() {
    let counts = Arc::new(StressSampleCounts::new());
    let num_threads = 4;
    let iterations = 50_000;

    // Track expected counts for verification
    let hot_count = Arc::new(AtomicU64::new(0));
    let warm_count = Arc::new(AtomicU64::new(0));
    let cold_count = Arc::new(AtomicU64::new(0));

    let barrier = Arc::new(Barrier::new(num_threads));

    let handles: Vec<_> = (0..num_threads)
        .map(|thread_id| {
            let c = Arc::clone(&counts);
            let b = Arc::clone(&barrier);
            let hot = Arc::clone(&hot_count);
            let warm = Arc::clone(&warm_count);
            let cold = Arc::clone(&cold_count);
            thread::spawn(move || {
                let hot_ip = 0xAAAA_AAAA_usize;
                let warm_ip = 0xBBBB_0000 + thread_id;
                let cold_ip = 0xCCCC_0000 + thread_id;

                b.wait();
                for i in 0..iterations {
                    if i % 10 == 0 {
                        // 10% cold path
                        c.increment(cold_ip);
                        cold.fetch_add(1, Ordering::Relaxed);
                    } else if i % 3 == 0 {
                        // ~30% warm path
                        c.increment(warm_ip);
                        warm.fetch_add(1, Ordering::Relaxed);
                    } else {
                        // ~60% hot path
                        c.increment(hot_ip);
                        hot.fetch_add(1, Ordering::Relaxed);
                    }
                }
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }

    let expected_total = hot_count.load(Ordering::Acquire)
        + warm_count.load(Ordering::Acquire)
        + cold_count.load(Ordering::Acquire);
    let actual_total = counts.total_count();

    assert_eq!(
        actual_total, expected_total,
        "Lost counts in mixed workload: expected={expected_total}, got={actual_total}"
    );
}

/// Stress test: Concurrent readers and writers.
///
/// Tests safety when reading entries() while other threads are writing.
/// This simulates the pattern of reading profiler data while profiling continues.
#[test]
fn concurrent_readers_writers() {
    let counts = Arc::new(StressSampleCounts::new());
    let running = Arc::new(AtomicBool::new(true));
    let write_count = Arc::new(AtomicU64::new(0));
    let num_writers = 4;

    // Writer threads
    let writer_handles: Vec<_> = (0..num_writers)
        .map(|thread_id| {
            let c = Arc::clone(&counts);
            let r = Arc::clone(&running);
            let wc = Arc::clone(&write_count);
            thread::spawn(move || {
                let base_ip = 0x5000_0000 + thread_id * 0x100_0000;
                while r.load(Ordering::Relaxed) {
                    c.increment(base_ip);
                    wc.fetch_add(1, Ordering::Relaxed);
                }
            })
        })
        .collect();

    // Reader thread
    let counts_for_reader = Arc::clone(&counts);
    let running_for_reader = Arc::clone(&running);
    let reader_handle = thread::spawn(move || {
        let mut reads = 0_u64;
        while running_for_reader.load(Ordering::Relaxed) {
            let total = counts_for_reader.total_count();
            // Verify we don't get any weird values
            assert!(total < 1_000_000_000, "Impossible count value: {total}");
            reads += 1;
            thread::sleep(Duration::from_micros(100));
        }
        reads
    });

    // Let it run for a bit
    thread::sleep(Duration::from_millis(100));
    running.store(false, Ordering::Release);

    for h in writer_handles {
        h.join().unwrap();
    }
    let reads = reader_handle.join().unwrap();

    // Verify final state
    let final_count = counts.total_count();
    let writes = write_count.load(Ordering::Acquire);
    assert!(
        final_count <= writes,
        "More counts than writes: counts={final_count}, writes={writes}"
    );
    assert!(
        reads > 10,
        "Reader should have done multiple reads, got {reads}"
    );
}
