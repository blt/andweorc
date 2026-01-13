//! Barrier-synchronized parallel workload (coz-style benchmark).
//!
//! This example demonstrates classic causal profiling on a barrier-synchronized
//! workload, similar to PARSEC benchmarks (fluidanimate, streamcluster).
//!
//! Unlike queue-based pipelines, barrier synchronization means ALL threads must
//! complete each phase before ANY thread can proceed. This creates a clear
//! bottleneck pattern where speeding up the slowest phase directly improves
//! overall throughput.
//!
//! The workload has two phases per iteration:
//! - Phase A: Light work (10K iterations) - NOT the bottleneck
//! - Phase B: Heavy work (100K iterations) - THE BOTTLENECK
//!
//! All threads must complete Phase A before any can start Phase B.
//! All threads must complete Phase B before any can start the next iteration.
//!
//! Expected causal profiling results:
//! - phase_b_compute: HIGH positive impact (~0.8-1.0)
//! - phase_a_compute: LOW or zero impact (~0.0-0.1)
//!
//! This pattern matches the coz paper's findings where barrier optimization
//! yielded 37.5% (fluidanimate) to 68.4% (streamcluster) improvements.
//!
//! Run with:
//!   ANDWEORC_ENABLED=1 cargo run --example barrier_phases --release

#![allow(clippy::print_stdout)]
#![allow(clippy::unwrap_used)]

use andweorc::{init, progress, run_experiments};
use std::hint::black_box;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Barrier};
use std::thread;

/// Number of worker threads (adjust based on CPU cores)
const NUM_THREADS: usize = 4;

/// Phase A computation - light work (NOT the bottleneck)
#[inline(never)]
fn phase_a_compute() -> u64 {
    let mut hash: u64 = 0x517c_c1b7_2722_0a95;
    for i in 0..10_000_u64 {
        hash ^= i;
        hash = hash.wrapping_mul(0x0100_0000_01b3);
        hash = hash.rotate_left(17);
    }
    black_box(hash)
}

/// Phase B computation - heavy work (THIS IS THE BOTTLENECK)
/// 10x more iterations than Phase A
#[inline(never)]
fn phase_b_compute() -> u64 {
    let mut hash: u64 = 0x9e37_79b9_7f4a_7c15;
    for i in 0..100_000_u64 {
        // This is where 90%+ of CPU time should be spent
        hash ^= i;
        hash = hash.wrapping_mul(0x0100_0000_01b3);
        hash = hash.rotate_left(17);
    }
    black_box(hash)
}

/// Worker thread function
fn worker(
    thread_id: usize,
    barrier: Arc<Barrier>,
    running: Arc<AtomicBool>,
    iterations: Arc<AtomicU64>,
) {
    // Each thread runs iterations until signaled to stop
    // Check running flag at the START of each iteration only
    // (can't check mid-iteration due to barrier synchronization)
    loop {
        // Check if we should stop BEFORE starting a new iteration
        if !running.load(Ordering::Relaxed) {
            break;
        }

        // Phase A: Light work
        let _ = phase_a_compute();

        // Barrier: Wait for all threads to complete Phase A
        barrier.wait();

        // Only thread 0 records the progress point
        if thread_id == 0 {
            progress!("phase_a_done");
        }

        // Phase B: Heavy work (THE BOTTLENECK)
        let _ = phase_b_compute();

        // Barrier: Wait for all threads to complete Phase B
        barrier.wait();

        // Only thread 0 records the progress point and counts iterations
        if thread_id == 0 {
            progress!("iteration_done");
            iterations.fetch_add(1, Ordering::Relaxed);
        }
    }
}

fn main() {
    println!("=== Barrier-Synchronized Parallel Workload ===\n");
    println!("This example demonstrates coz-style causal profiling.");
    println!("All {} threads must complete each phase before proceeding.\n", NUM_THREADS);
    println!("Phase workloads:");
    println!("  Phase A: 10K hash iterations (light work)");
    println!("  Phase B: 100K hash iterations (BOTTLENECK - 10x more work)\n");
    println!("Expected results:");
    println!("  - phase_b_compute: HIGH causal impact");
    println!("  - phase_a_compute: LOW causal impact\n");

    // Initialize profiler
    if let Err(e) = init() {
        eprintln!("Failed to initialize profiler: {e}");
    }

    let running = Arc::new(AtomicBool::new(true));
    let iterations = Arc::new(AtomicU64::new(0));

    // Create barrier for synchronization (used twice per iteration)
    let barrier = Arc::new(Barrier::new(NUM_THREADS));

    // Brief warmup
    println!("Phase 1: Brief warmup...");
    {
        let warmup_running = Arc::new(AtomicBool::new(true));
        let warmup_iterations = Arc::new(AtomicU64::new(0));
        let warmup_barrier = Arc::new(Barrier::new(NUM_THREADS));

        let handles: Vec<_> = (0..NUM_THREADS)
            .map(|id| {
                let r = Arc::clone(&warmup_running);
                let it = Arc::clone(&warmup_iterations);
                let b = Arc::clone(&warmup_barrier);
                thread::spawn(move || worker(id, b, r, it))
            })
            .collect();

        // Run for a fixed number of iterations
        while warmup_iterations.load(Ordering::Relaxed) < 100 {
            thread::yield_now();
        }
        warmup_running.store(false, Ordering::Release);

        for h in handles {
            let _ = h.join();
        }

        let its = warmup_iterations.load(Ordering::Relaxed);
        println!("Warmup complete: {} iterations\n", its);
    }

    // Phase 2: Run with experiments
    println!("Phase 2: Running with causal profiling experiments...");
    println!("This will take several minutes - workers run continuously\n");

    // Spawn worker threads
    let handles: Vec<_> = (0..NUM_THREADS)
        .map(|id| {
            let r = Arc::clone(&running);
            let it = Arc::clone(&iterations);
            let b = Arc::clone(&barrier);
            thread::spawn(move || worker(id, b, r, it))
        })
        .collect();

    // Run experiments on the main thread
    // Measure throughput at iteration_done (after both phases complete)
    let _ = run_experiments("iteration_done");

    // Signal workers to stop
    running.store(false, Ordering::Release);

    // Wait for all workers to finish
    for h in handles {
        let _ = h.join();
    }

    let total_iterations = iterations.load(Ordering::Relaxed);
    println!("\n=== Results ===");
    println!("Total iterations completed: {}", total_iterations);

    println!("\n=== Expected Causal Profiling Analysis ===");
    println!("If profiling is enabled (ANDWEORC_ENABLED=1), you should see:");
    println!();
    println!("  phase_b_compute: HIGH causal impact (~0.8-1.0)");
    println!("    - This is the bottleneck phase");
    println!("    - All threads wait for the slowest thread in Phase B");
    println!("    - Speeding it up directly improves iteration throughput");
    println!();
    println!("  phase_a_compute: LOW causal impact (~0.0-0.1)");
    println!("    - Phase A completes quickly");
    println!("    - Even if sped up, threads still wait for Phase B");
    println!();
    println!("This pattern matches the coz paper's barrier optimization findings.");
}
