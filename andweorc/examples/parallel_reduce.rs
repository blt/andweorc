//! Parallel reduction with barrier synchronization.
//!
//! This example demonstrates causal profiling on a parallel reduction pattern,
//! common in numerical computing and data processing (MapReduce, aggregations).
//!
//! The pattern:
//! 1. Each thread computes a local sum over its partition (MAP phase)
//! 2. Barrier sync
//! 3. Thread 0 reduces all partial sums (REDUCE phase)
//! 4. Barrier sync before next iteration
//!
//! The bottleneck is typically in the reduction phase which is serialized.
//! This pattern shows how causal profiling can identify whether the
//! bottleneck is in parallel work or sequential reduction.
//!
//! Expected results:
//! - If data partition is large: compute_partial_sum has HIGH impact
//! - If data partition is small: reduce_all has HIGH impact
//!
//! Run with:
//!   ANDWEORC_ENABLED=1 cargo run --example parallel_reduce --release

#![allow(clippy::print_stdout)]
#![allow(clippy::unwrap_used)]

use andweorc::{init, progress, run_experiments};
use std::hint::black_box;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Barrier};
use std::thread;

/// Number of worker threads
const NUM_THREADS: usize = 4;

/// Size of each thread's data partition (adjust to change bottleneck)
const PARTITION_SIZE: usize = 100_000;

/// Number of reduction iterations per round (simulates complex aggregation)
const REDUCE_ITERATIONS: usize = 50_000;

/// Shared partial results array
struct SharedState {
    partials: [AtomicU64; NUM_THREADS],
    running: AtomicBool,
    iterations: AtomicU64,
}

impl SharedState {
    fn new() -> Self {
        Self {
            partials: std::array::from_fn(|_| AtomicU64::new(0)),
            running: AtomicBool::new(true),
            iterations: AtomicU64::new(0),
        }
    }
}

/// Compute partial sum over a data partition (MAP phase).
/// Each thread does this independently.
#[inline(never)]
fn compute_partial_sum(thread_id: usize, iteration: u64) -> u64 {
    let mut sum: u64 = 0;
    let base = (thread_id as u64).wrapping_mul(PARTITION_SIZE as u64);

    // Simulate reading and processing data partition
    for i in 0..PARTITION_SIZE as u64 {
        // Mix in iteration to vary the data
        let value = (base + i).wrapping_mul(iteration.wrapping_add(1));

        // Some computation on each element
        let processed = value ^ (value >> 17);
        let processed = processed.wrapping_mul(0x0100_0000_01b3);

        sum = sum.wrapping_add(processed);
    }

    black_box(sum)
}

/// Reduce all partial sums into a final result (REDUCE phase).
/// Only thread 0 does this, creating a potential bottleneck.
#[inline(never)]
fn reduce_all(partials: &[AtomicU64; NUM_THREADS]) -> u64 {
    // Gather partial results
    let mut total: u64 = 0;
    for partial in partials {
        total = total.wrapping_add(partial.load(Ordering::Acquire));
    }

    // Simulate complex aggregation logic
    let mut hash = total;
    for i in 0..REDUCE_ITERATIONS as u64 {
        hash ^= i;
        hash = hash.wrapping_mul(0x9e37_79b9_7f4a_7c15);
        hash = hash.rotate_left(23);
    }

    black_box(hash)
}

/// Worker thread function
fn worker(thread_id: usize, barrier: Arc<Barrier>, state: Arc<SharedState>) {
    let mut local_iteration: u64 = 0;

    loop {
        // Check if we should stop
        if !state.running.load(Ordering::Relaxed) {
            break;
        }

        // === MAP PHASE ===
        // Each thread computes its partial sum independently
        let partial = compute_partial_sum(thread_id, local_iteration);
        state.partials[thread_id].store(partial, Ordering::Release);

        // Barrier: Wait for all threads to complete MAP phase
        barrier.wait();

        // === REDUCE PHASE ===
        // Only thread 0 performs the reduction
        if thread_id == 0 {
            let _result = reduce_all(&state.partials);
            progress!("reduce_done");
            state.iterations.fetch_add(1, Ordering::Relaxed);
        }

        // Barrier: Wait for reduction to complete before next iteration
        barrier.wait();

        local_iteration += 1;
    }
}

fn main() {
    println!("=== Parallel Reduction with Barrier Synchronization ===\n");
    println!("This example demonstrates causal profiling on a MapReduce pattern.");
    println!("{} threads each process a partition, then reduce results.\n", NUM_THREADS);
    println!("Configuration:");
    println!("  Threads: {}", NUM_THREADS);
    println!("  Partition size: {} elements per thread", PARTITION_SIZE);
    println!("  Reduce iterations: {} (simulated aggregation work)\n", REDUCE_ITERATIONS);
    println!("Expected profiling results:");
    println!("  - compute_partial_sum: Impact depends on partition size");
    println!("  - reduce_all: Impact depends on reduction complexity\n");

    // Initialize profiler
    if let Err(e) = init() {
        eprintln!("Failed to initialize profiler: {e}");
    }

    let state = Arc::new(SharedState::new());
    let barrier = Arc::new(Barrier::new(NUM_THREADS));

    // Brief warmup
    println!("Phase 1: Brief warmup...");
    {
        let warmup_state = Arc::new(SharedState::new());
        let warmup_barrier = Arc::new(Barrier::new(NUM_THREADS));

        let handles: Vec<_> = (0..NUM_THREADS)
            .map(|id| {
                let s = Arc::clone(&warmup_state);
                let b = Arc::clone(&warmup_barrier);
                thread::spawn(move || worker(id, b, s))
            })
            .collect();

        // Run for a fixed number of iterations
        while warmup_state.iterations.load(Ordering::Relaxed) < 50 {
            thread::yield_now();
        }
        warmup_state.running.store(false, Ordering::Release);

        for h in handles {
            let _ = h.join();
        }

        let its = warmup_state.iterations.load(Ordering::Relaxed);
        println!("Warmup complete: {} iterations\n", its);
    }

    // Phase 2: Run with experiments
    println!("Phase 2: Running with causal profiling experiments...");
    println!("Workers will continue until experiments complete.\n");

    // Spawn worker threads
    let handles: Vec<_> = (0..NUM_THREADS)
        .map(|id| {
            let s = Arc::clone(&state);
            let b = Arc::clone(&barrier);
            thread::spawn(move || worker(id, b, s))
        })
        .collect();

    // Run experiments on the main thread
    let _ = run_experiments("reduce_done");

    // Signal workers to stop
    state.running.store(false, Ordering::Release);

    // Wait for all workers
    for h in handles {
        let _ = h.join();
    }

    let total_iterations = state.iterations.load(Ordering::Relaxed);
    println!("\n=== Results ===");
    println!("Total iterations completed: {}", total_iterations);

    println!("\n=== Causal Profiling Analysis ===");
    println!("The profiling results show which phase is the bottleneck:\n");
    println!("Scenario 1 - Large partitions, simple reduce:");
    println!("  compute_partial_sum: HIGH impact (parallel work dominates)");
    println!("  reduce_all: LOW impact (fast aggregation)\n");
    println!("Scenario 2 - Small partitions, complex reduce:");
    println!("  compute_partial_sum: LOW impact (fast parallel work)");
    println!("  reduce_all: HIGH impact (serialized reduction dominates)\n");
    println!("This pattern is common in data processing pipelines:");
    println!("  - Spark/MapReduce aggregations");
    println!("  - Parallel numerical algorithms");
    println!("  - Machine learning gradient averaging");
}
