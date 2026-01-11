//! Example: Lock contention analysis.
//!
//! This example demonstrates how causal profiling can identify lock
//! contention as a bottleneck. It simulates a web server with:
//!
//! 1. A shared counter protected by a mutex (high contention)
//! 2. Thread-local counters (low contention)
//!
//! The profiler should show that speeding up the mutex-protected
//! critical section would have significant impact on throughput.

// Examples are demonstration code - allow more relaxed rules
#![allow(clippy::unwrap_used)]
#![allow(clippy::expect_used)]
#![allow(clippy::print_stdout)]
#![allow(dead_code)]

use andweorc::progress;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Instant;

/// Shared state protected by a mutex - this creates contention.
struct SharedCounter {
    value: Mutex<u64>,
}

impl SharedCounter {
    fn new() -> Self {
        Self {
            value: Mutex::new(0),
        }
    }

    /// Increment with lock - this is the contention point.
    fn increment(&self) {
        let mut guard = self.value.lock().unwrap();
        // Simulate some work while holding the lock
        let v = *guard;
        // Busy wait to exaggerate the critical section
        for _ in 0..100 {
            std::hint::black_box(v + 1);
        }
        *guard = v + 1;
    }

    fn get(&self) -> u64 {
        *self.value.lock().unwrap()
    }
}

/// Process a "request" using shared state - high contention.
fn process_request_shared(counter: &SharedCounter) {
    // Simulate request processing
    let mut checksum = 0u64;
    for i in 0..1000 {
        checksum = checksum.wrapping_add(i);
    }
    std::hint::black_box(checksum);

    // Update shared counter - THIS IS THE BOTTLENECK
    counter.increment();

    progress!("shared_request_done");
}

/// Process a "request" using thread-local state - no contention.
fn process_request_local(local_counter: &mut u64) {
    // Simulate request processing
    let mut checksum = 0u64;
    for i in 0..1000 {
        checksum = checksum.wrapping_add(i);
    }
    std::hint::black_box(checksum);

    // Update local counter - no contention
    *local_counter += 1;

    progress!("local_request_done");
}

fn main() {
    // Initialize profiler (only active if ANDWEORC_ENABLED=1)
    if let Err(e) = andweorc::init() {
        eprintln!("Failed to initialize profiler: {}", e);
    }

    println!("Lock Contention Example");
    println!("=======================");
    println!();
    println!("This example demonstrates how causal profiling identifies");
    println!("mutex contention as a performance bottleneck.");
    println!();

    // Phase 1: Warmup on main thread only to collect samples safely
    println!("Phase 1: Warmup (collecting samples on main thread)...");
    let shared_counter = Arc::new(SharedCounter::new());
    let start = Instant::now();

    for _ in 0..50_000 {
        process_request_shared(&shared_counter);
    }

    let warmup_elapsed = start.elapsed();
    println!(
        "Warmup: {} requests in {:?} ({:.2} req/sec)",
        shared_counter.get(),
        warmup_elapsed,
        50_000.0 / warmup_elapsed.as_secs_f64()
    );

    // Phase 2: Run experiments with multiple worker threads
    // Testing multi-threaded profiling with SIGPROF blocking fix
    let num_workers = 4;
    println!(
        "\nPhase 2: Running causal profiling experiments with {} workers...",
        num_workers
    );

    let running = Arc::new(AtomicBool::new(true));
    let experiment_counter = Arc::new(SharedCounter::new());

    let mut workers = Vec::with_capacity(num_workers);
    for worker_id in 0..num_workers {
        let running_clone = Arc::clone(&running);
        let counter_clone = Arc::clone(&experiment_counter);

        workers.push(thread::spawn(move || {
            // Start profiling on this thread
            if let Err(e) = andweorc::start_profiling() {
                eprintln!("Worker {}: Failed to start profiling: {}", worker_id, e);
            }
            let mut iterations = 0u64;
            while running_clone.load(Ordering::Relaxed) {
                process_request_shared(&counter_clone);
                iterations += 1;
            }
            iterations
        }));
    }

    // Run experiments on the main thread
    let _ = andweorc::run_experiments("shared_request_done");

    // Stop workers
    running.store(false, Ordering::Relaxed);
    let mut total_iterations = 0u64;
    for (i, worker) in workers.into_iter().enumerate() {
        let iters = worker.join().unwrap();
        println!("Worker {} completed {} iterations", i, iters);
        total_iterations += iters;
    }
    println!(
        "Total: {} iterations across {} workers",
        total_iterations, num_workers
    );
}
