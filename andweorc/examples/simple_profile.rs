//! Simple example demonstrating andweorc profiler functionality.
//!
//! This example shows how the profiler collects samples and reports them.
//! Run with: `cargo run --example simple_profile`
//!
//! Note: Perf events require appropriate permissions. You may need to:
//! - Run as root, OR
//! - Set `/proc/sys/kernel/perf_event_paranoid` to 1 or lower, OR
//! - Give the binary `CAP_PERFMON` capability

// Examples are allowed to use println
#![allow(clippy::print_stdout)]

use andweorc::progress_point::Progress;
use andweorc::{progress, start_profiling};
use std::hint::black_box;

fn expensive_computation(iterations: u64) -> u64 {
    let mut sum = 0u64;
    for i in 0..iterations {
        sum = sum.wrapping_add(i.wrapping_mul(i));
    }
    black_box(sum)
}

fn main() {
    println!("Andweorc Profiler Example");
    println!("=========================\n");

    // Start profiling for this thread
    match start_profiling() {
        Ok(()) => println!("Profiling started successfully"),
        Err(e) => println!("Failed to start profiling: {e}"),
    }

    println!("\nRunning expensive computation...");
    println!("(If perf events are available, you'll see IP/CALLCHAIN output)\n");

    // Do some work and mark progress
    for i in 0..10 {
        let result = expensive_computation(1_000_000);
        progress!("iteration_complete");
        println!("Iteration {i}: result = {result}");
    }

    // Get the progress point and show throughput statistics
    let progress_point = Progress::get_instance("iteration_complete");
    println!("\nThroughput Statistics:");
    println!("  Visits: {}", progress_point.visit_count());
    println!("  Elapsed: {} ns", progress_point.elapsed_nanos());
    println!(
        "  Throughput: {:.2} iterations/sec",
        progress_point.throughput()
    );

    println!("\nDone!");
}
