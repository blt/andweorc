//! Debug example to test perf-event configuration.

// Examples are not production code - allow relaxed linting
#![allow(clippy::unwrap_used)]
#![allow(clippy::print_stdout)]
#![allow(clippy::print_stderr)]
#![allow(clippy::too_many_lines)]
#![allow(clippy::uninlined_format_args)]
#![allow(clippy::cast_precision_loss)]

use perf_event::events::Hardware;
use perf_event::sample::PerfSampleType;
use perf_event::Builder;
use std::time::Duration;

fn main() {
    println!("Testing perf-event configurations...\n");

    // Test 1: Basic counter (no sampling)
    println!("Test 1: Basic CPU_CYCLES counter");
    match Builder::new().kind(Hardware::CPU_CYCLES).counter() {
        Ok(counter) => {
            println!("  SUCCESS: Basic counter created");
            drop(counter);
        }
        Err(e) => {
            println!("  FAILED: {}", e);
        }
    }

    // Test 2: Counter with sampling (no IP)
    println!("\nTest 2: Counter with sample_frequency");
    match Builder::new()
        .kind(Hardware::CPU_CYCLES)
        .sample_frequency(1000)
        .counter()
    {
        Ok(counter) => {
            println!("  SUCCESS: Sampling counter created");
            drop(counter);
        }
        Err(e) => {
            println!("  FAILED: {}", e);
        }
    }

    // Test 3: Sample stream (minimal)
    println!("\nTest 3: Sample stream (minimal - IP only)");
    match Builder::new()
        .kind(Hardware::CPU_CYCLES)
        .sample_frequency(1000)
        .sample(PerfSampleType::IP)
        .sample_stream()
    {
        Ok(stream) => {
            println!("  SUCCESS: Sample stream created");
            drop(stream);
        }
        Err(e) => {
            println!("  FAILED: {}", e);
        }
    }

    // Test 4: Sample stream with CALLCHAIN
    println!("\nTest 4: Sample stream with CALLCHAIN");
    match Builder::new()
        .kind(Hardware::CPU_CYCLES)
        .sample_frequency(1000)
        .sample(PerfSampleType::IP)
        .sample(PerfSampleType::CALLCHAIN)
        .sample_stream()
    {
        Ok(stream) => {
            println!("  SUCCESS: Sample stream with callchain created");
            drop(stream);
        }
        Err(e) => {
            println!("  FAILED: {}", e);
        }
    }

    // Test 5: Full config from per_thread.rs
    println!("\nTest 5: Full configuration from per_thread.rs");
    match Builder::new()
        .kind(Hardware::CPU_CYCLES)
        .sample_frequency(1000)
        .sample(PerfSampleType::ADDR)
        .sample(PerfSampleType::CALLCHAIN)
        .sample(PerfSampleType::CPU)
        .sample(PerfSampleType::IP)
        .sample(PerfSampleType::PERIOD)
        .sample(PerfSampleType::STREAM_ID)
        .sample(PerfSampleType::TID)
        .sample(PerfSampleType::TIME)
        .sample_stream()
    {
        Ok(stream) => {
            println!("  SUCCESS: Full sample stream created");

            // Try to actually read samples
            println!("  Reading samples...");
            let timeout = Duration::from_millis(100);
            match stream.read(Some(timeout)) {
                Ok(Some(sample)) => println!("  Got sample: {:?}", sample),
                Ok(None) => println!("  No samples available (expected without workload)"),
                Err(e) => println!("  Read error: {}", e),
            }
            drop(stream);
        }
        Err(e) => {
            println!("  FAILED: {}", e);
            eprintln!("\n  Full error: {:?}", e);
        }
    }

    // Test 6: Try different event types
    println!("\nTest 6: Try INSTRUCTIONS instead of CPU_CYCLES");
    match Builder::new()
        .kind(Hardware::INSTRUCTIONS)
        .sample_frequency(1000)
        .sample(PerfSampleType::IP)
        .sample_stream()
    {
        Ok(stream) => {
            println!("  SUCCESS: Instructions sample stream created");
            drop(stream);
        }
        Err(e) => {
            println!("  FAILED: {}", e);
        }
    }

    // Test 7: Software event
    println!("\nTest 7: Try SOFTWARE CPU_CLOCK");
    match Builder::new()
        .kind(perf_event::events::Software::CPU_CLOCK)
        .sample_frequency(1000)
        .sample(PerfSampleType::IP)
        .sample_stream()
    {
        Ok(stream) => {
            println!("  SUCCESS: Software CPU_CLOCK stream created");
            drop(stream);
        }
        Err(e) => {
            println!("  FAILED: {}", e);
        }
    }

    println!("\nDone.");
}
