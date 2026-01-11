//! Loom exhaustive concurrency tests for lock-free sample counting.
//!
//! These tests use the Loom model checker to exhaustively explore all possible
//! interleavings of concurrent operations, proving the absence of data races
//! and verifying linearizability of the lock-free hash table.
//!
//! Run with: cargo test --test loom_sample_counts --release
//!
//! Loom tests are computationally expensive. The tests here use small bucket
//! counts and limited thread counts to keep the state space manageable.

use loom::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use loom::sync::Arc;
use loom::thread;

/// Number of buckets for loom testing (small to keep state space manageable).
const LOOM_BUCKETS: usize = 4;

/// Simplified SampleCounts for Loom testing.
///
/// This mirrors the production SampleCounts structure but uses loom atomics
/// and a smaller bucket count for exhaustive testing.
struct LoomSampleCounts {
    ips: [AtomicUsize; LOOM_BUCKETS],
    counts: [AtomicU64; LOOM_BUCKETS],
}

impl LoomSampleCounts {
    fn new() -> Self {
        Self {
            ips: std::array::from_fn(|_| AtomicUsize::new(0)),
            counts: std::array::from_fn(|_| AtomicU64::new(0)),
        }
    }

    /// Simple hash function for testing (just modulo bucket count).
    fn hash(ip: usize) -> usize {
        ip % LOOM_BUCKETS
    }

    /// Increments the count for the given IP using lock-free linear probing.
    ///
    /// This is the exact algorithm from production, just with loom atomics.
    fn increment(&self, ip: usize) {
        if ip == 0 {
            return;
        }

        let start = Self::hash(ip);

        for i in 0..LOOM_BUCKETS {
            let idx = (start + i) % LOOM_BUCKETS;

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
                            // Another thread just inserted this IP
                            self.counts[idx].fetch_add(1, Ordering::Relaxed);
                            return;
                        }
                        // Someone else claimed it with a different IP, continue probing
                    }
                }
            }
        }
        // Table full, silently drop
    }

    /// Returns the total count for a specific IP (for verification).
    fn get_count(&self, ip: usize) -> u64 {
        if ip == 0 {
            return 0;
        }

        let start = Self::hash(ip);

        for i in 0..LOOM_BUCKETS {
            let idx = (start + i) % LOOM_BUCKETS;
            let stored = self.ips[idx].load(Ordering::Acquire);

            if stored == ip {
                return self.counts[idx].load(Ordering::Acquire);
            }
            if stored == 0 {
                return 0;
            }
        }
        0
    }

    /// Returns total count across all IPs (for verification).
    fn total_count(&self) -> u64 {
        let mut total = 0;
        for i in 0..LOOM_BUCKETS {
            total += self.counts[i].load(Ordering::Acquire);
        }
        total
    }
}

/// Test: Concurrent increments of the same IP should not lose counts.
///
/// This is the most critical property - when two threads both call
/// increment(IP), the final count should be exactly 2.
#[test]
fn same_ip_no_lost_increments() {
    loom::model(|| {
        let counts = Arc::new(LoomSampleCounts::new());
        let ip = 0x1234_usize;

        let c1 = Arc::clone(&counts);
        let c2 = Arc::clone(&counts);

        let t1 = thread::spawn(move || {
            c1.increment(ip);
        });

        let t2 = thread::spawn(move || {
            c2.increment(ip);
        });

        t1.join().unwrap();
        t2.join().unwrap();

        // Both increments must be counted
        let final_count = counts.get_count(ip);
        assert_eq!(final_count, 2, "Lost increment! Count was {final_count}");
    });
}

/// Test: Concurrent increments of different IPs should all succeed.
///
/// When two threads increment different IPs, both should be recorded.
#[test]
fn different_ips_both_recorded() {
    loom::model(|| {
        let counts = Arc::new(LoomSampleCounts::new());
        let ip1 = 0x1000_usize;
        let ip2 = 0x2000_usize;

        let c1 = Arc::clone(&counts);
        let c2 = Arc::clone(&counts);

        let t1 = thread::spawn(move || {
            c1.increment(ip1);
        });

        let t2 = thread::spawn(move || {
            c2.increment(ip2);
        });

        t1.join().unwrap();
        t2.join().unwrap();

        // Both IPs should have count 1
        let count1 = counts.get_count(ip1);
        let count2 = counts.get_count(ip2);
        assert_eq!(count1, 1, "IP1 count wrong: {count1}");
        assert_eq!(count2, 1, "IP2 count wrong: {count2}");
    });
}

/// Test: Slot contention - two IPs that hash to the same bucket.
///
/// This tests the linear probing behavior under contention.
#[test]
fn slot_contention_both_succeed() {
    loom::model(|| {
        let counts = Arc::new(LoomSampleCounts::new());
        // IPs that hash to the same bucket (with 4 buckets, 1 and 5 both hash to 1)
        let ip1 = 1_usize;
        let ip2 = LOOM_BUCKETS + 1; // Also hashes to bucket 1

        let c1 = Arc::clone(&counts);
        let c2 = Arc::clone(&counts);

        let t1 = thread::spawn(move || {
            c1.increment(ip1);
        });

        let t2 = thread::spawn(move || {
            c2.increment(ip2);
        });

        t1.join().unwrap();
        t2.join().unwrap();

        // Total should be 2 (one for each IP)
        let total = counts.total_count();
        assert_eq!(total, 2, "Total count wrong: {total}");
    });
}

/// Test: Multiple increments from multiple threads.
///
/// Three threads each increment the same IP once. Final count should be 3.
#[test]
fn three_threads_same_ip() {
    loom::model(|| {
        let counts = Arc::new(LoomSampleCounts::new());
        let ip = 0x5678_usize;

        let handles: Vec<_> = (0..3)
            .map(|_| {
                let c = Arc::clone(&counts);
                thread::spawn(move || {
                    c.increment(ip);
                })
            })
            .collect();

        for h in handles {
            h.join().unwrap();
        }

        let final_count = counts.get_count(ip);
        assert_eq!(final_count, 3, "Lost increment! Count was {final_count}");
    });
}

/// Test: Race between check and CAS in increment.
///
/// This specifically tests the TOCTOU race where:
/// 1. Thread A loads IP as 0
/// 2. Thread B successfully CASes IP from 0 to their value
/// 3. Thread A's CAS fails, but needs to handle correctly
#[test]
fn cas_race_handling() {
    loom::model(|| {
        let counts = Arc::new(LoomSampleCounts::new());
        // Use IPs that hash to the same bucket
        let ip = 1_usize;

        let c1 = Arc::clone(&counts);
        let c2 = Arc::clone(&counts);

        // Both threads try to claim the same slot for the same IP
        let t1 = thread::spawn(move || {
            c1.increment(ip);
            c1.increment(ip); // Two increments
        });

        let t2 = thread::spawn(move || {
            c2.increment(ip);
        });

        t1.join().unwrap();
        t2.join().unwrap();

        // Total should be 3
        let final_count = counts.get_count(ip);
        assert_eq!(final_count, 3, "Lost increment! Count was {final_count}");
    });
}

/// Test: Zero IP is ignored.
///
/// Zero is reserved as the empty slot marker and should never be stored.
#[test]
fn zero_ip_ignored() {
    loom::model(|| {
        let counts = Arc::new(LoomSampleCounts::new());

        let c1 = Arc::clone(&counts);
        let c2 = Arc::clone(&counts);

        let t1 = thread::spawn(move || {
            c1.increment(0); // Should be ignored
            c1.increment(1);
        });

        let t2 = thread::spawn(move || {
            c2.increment(0); // Should be ignored
            c2.increment(2);
        });

        t1.join().unwrap();
        t2.join().unwrap();

        // Zero should have no count
        let zero_count = counts.get_count(0);
        assert_eq!(zero_count, 0, "Zero IP should be ignored");

        // Total should be 2 (one each for IP 1 and 2)
        let total = counts.total_count();
        assert_eq!(total, 2, "Total count wrong: {total}");
    });
}
