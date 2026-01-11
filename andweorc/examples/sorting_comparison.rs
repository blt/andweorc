//! Example: Sorting algorithm comparison.
//!
//! This example compares quicksort vs bubble sort to demonstrate how causal
//! profiling identifies the algorithm as the bottleneck rather than just
//! showing "sort takes time".
//!
//! The profiler should identify that speeding up bubble_sort would have
//! a much larger impact on throughput than speeding up quicksort.

// Examples are demonstration code - allow more relaxed rules
#![allow(clippy::unwrap_used)]
#![allow(clippy::expect_used)]
#![allow(clippy::print_stdout)]
#![allow(clippy::print_stderr)]
#![allow(clippy::uninlined_format_args)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::doc_markdown)]
#![allow(clippy::ptr_arg)]
#![allow(clippy::manual_is_multiple_of)]

use andweorc::progress;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;

/// Bubble sort implementation - deliberately slow for comparison.
fn bubble_sort<T: Ord>(arr: &mut [T]) {
    let n = arr.len();
    for i in 0..n {
        for j in 0..n - 1 - i {
            if arr[j] > arr[j + 1] {
                arr.swap(j, j + 1);
            }
        }
    }
}

/// Quicksort implementation - efficient comparison baseline.
fn quicksort<T: Ord>(arr: &mut [T]) {
    if arr.len() <= 1 {
        return;
    }

    let pivot_idx = partition(arr);
    let (left, right) = arr.split_at_mut(pivot_idx);
    quicksort(left);
    quicksort(&mut right[1..]);
}

fn partition<T: Ord>(arr: &mut [T]) -> usize {
    let len = arr.len();
    let pivot_idx = len / 2;
    arr.swap(pivot_idx, len - 1);

    let mut store_idx = 0;
    for i in 0..len - 1 {
        if arr[i] <= arr[len - 1] {
            arr.swap(i, store_idx);
            store_idx += 1;
        }
    }
    arr.swap(store_idx, len - 1);
    store_idx
}

/// Process a batch using bubble sort - this is the bottleneck.
fn process_batch_bubble(data: &mut Vec<i32>) {
    bubble_sort(data);
    progress!("bubble_sort_done");
}

/// Process a batch using quicksort - this is fast.
fn process_batch_quick(data: &mut Vec<i32>) {
    quicksort(data);
    progress!("quicksort_done");
}

/// Generate random-ish data for sorting.
fn generate_data(size: usize, seed: u64) -> Vec<i32> {
    let mut data = Vec::with_capacity(size);
    let mut state = seed;
    for _ in 0..size {
        // Simple LCG for deterministic "random" numbers
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        #[allow(clippy::cast_possible_truncation)]
        data.push((state >> 33) as i32);
    }
    data
}

fn main() {
    // Initialize profiler (only active if ANDWEORC_ENABLED=1)
    if let Err(e) = andweorc::init() {
        eprintln!("Failed to initialize profiler: {}", e);
    }

    println!("Sorting Comparison Example");
    println!("==========================");
    println!();
    println!("This example demonstrates causal profiling by comparing");
    println!("bubble sort (O(n^2)) vs quicksort (O(n log n)).");
    println!();

    let bubble_size = 500; // Small array for bubble sort (it's slow!)
    let quick_size = 10_000; // Larger array for quicksort

    // Phase 1: Warmup to collect samples
    println!("Phase 1: Warmup (collecting samples)...");
    let warmup_iterations = 500;
    let start = Instant::now();

    for i in 0..warmup_iterations {
        if i % 2 == 0 {
            let mut data = generate_data(bubble_size, i as u64);
            process_batch_bubble(&mut data);
        } else {
            let mut data = generate_data(quick_size, i as u64);
            process_batch_quick(&mut data);
        }
    }

    let warmup_elapsed = start.elapsed();
    println!(
        "Warmup: {} iterations in {:?}",
        warmup_iterations, warmup_elapsed
    );
    println!();

    // Phase 2: Run experiments with a worker thread doing the actual work
    println!("Phase 2: Running causal profiling experiments...");
    println!("(Worker thread runs sorting while main thread runs experiments)");
    println!();

    // Flag to stop the worker
    let running = Arc::new(AtomicBool::new(true));
    let running_clone = Arc::clone(&running);

    // Spawn worker thread to continuously run the workload
    let worker = std::thread::spawn(move || {
        // Initialize profiling on this thread too
        if let Err(e) = andweorc::start_profiling() {
            eprintln!("Worker: Failed to start profiling: {}", e);
        }

        let mut iterations = 0u64;
        while running_clone.load(Ordering::Relaxed) {
            if iterations % 2 == 0 {
                let mut data = generate_data(bubble_size, iterations);
                process_batch_bubble(&mut data);
            } else {
                let mut data = generate_data(quick_size, iterations);
                process_batch_quick(&mut data);
            }
            iterations += 1;
        }
        iterations
    });

    // Run experiments on the main thread
    // This will measure throughput from the worker's progress points
    let _ = andweorc::run_experiments("bubble_sort_done");

    // Stop the worker
    running.store(false, Ordering::Relaxed);
    let worker_iterations = worker.join().unwrap();
    println!("Worker completed {} iterations", worker_iterations);
}
