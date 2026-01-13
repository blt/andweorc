//! Pipeline stage bottleneck example.
//!
//! This example demonstrates how causal profiling identifies the limiting stage
//! in a multi-stage processing pipeline. Unlike traditional profilers that show
//! where time is spent, causal profiling shows which stage actually limits throughput.
//!
//! The pipeline has three stages:
//! 1. Parse: Fast stage (~1ms per item)
//! 2. Transform: SLOW stage (~10ms per item) - THE BOTTLENECK
//! 3. Serialize: Fast stage (~1ms per item)
//!
//! Traditional profiling would show Transform takes most time (obvious).
//! Causal profiling shows that optimizing Transform would yield ~10x throughput gain,
//! while optimizing Parse or Serialize would yield almost no improvement.
//!
//! Run with:
//!   ANDWEORC_ENABLED=1 cargo run --example pipeline_stages --release
//!
//! Expected results:
//!   - Transform stage shows high causal impact (~0.8-1.0)
//!   - Parse and Serialize show low causal impact (~0.0-0.1)

// Examples are allowed to use println and unwrap
#![allow(clippy::print_stdout)]
#![allow(clippy::unwrap_used)]
#![allow(clippy::expect_used)]

use andweorc::{init, progress, run_experiments, start_profiling};
use std::hint::black_box;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

/// Simulated work item flowing through the pipeline
#[derive(Clone)]
struct WorkItem {
    id: u64,
    data: Vec<u8>,
}

/// Stage 1: Parse - Fast (~1ms)
fn parse_stage(item_id: u64) -> WorkItem {
    // Simulate parsing work
    let mut data = Vec::with_capacity(1000);
    for i in 0..1000 {
        data.push((i as u8).wrapping_add(item_id as u8));
    }

    // Small busy-wait to simulate ~1ms of work
    busy_wait_micros(1000);

    progress!("parse_done");

    WorkItem { id: item_id, data }
}

/// Stage 2: Transform - SLOW (~10ms) - THIS IS THE BOTTLENECK
fn transform_stage(mut item: WorkItem) -> WorkItem {
    // Simulate heavy transformation work
    for _ in 0..10 {
        for byte in &mut item.data {
            *byte = byte.wrapping_mul(31).wrapping_add(17);
        }
    }

    // Busy-wait to simulate ~10ms of work (THE BOTTLENECK)
    busy_wait_micros(10_000);

    progress!("transform_done");

    item
}

/// Stage 3: Serialize - Fast (~1ms)
fn serialize_stage(item: WorkItem) -> Vec<u8> {
    // Simulate serialization
    let mut output = Vec::with_capacity(item.data.len() + 8);
    output.extend_from_slice(&item.id.to_le_bytes());
    output.extend_from_slice(&item.data);

    // Small busy-wait to simulate ~1ms of work
    busy_wait_micros(1000);

    progress!("serialize_done");

    output
}

/// Busy-wait for approximately the given number of microseconds
fn busy_wait_micros(micros: u64) {
    let start = Instant::now();
    let target = Duration::from_micros(micros);
    while start.elapsed() < target {
        black_box(0u64);
    }
}

/// Run the pipeline with separate threads for each stage
fn run_pipeline(num_items: u64, running: Arc<AtomicBool>, items_processed: Arc<AtomicU64>) {
    // Create channels between stages
    let (parse_tx, parse_rx): (Sender<WorkItem>, Receiver<WorkItem>) = channel();
    let (transform_tx, transform_rx): (Sender<WorkItem>, Receiver<WorkItem>) = channel();
    let (serialize_tx, serialize_rx): (Sender<Vec<u8>>, Receiver<Vec<u8>>) = channel();

    let running_clone = Arc::clone(&running);
    let items_clone = Arc::clone(&items_processed);

    // Stage 1: Parser thread
    let parser = thread::spawn(move || {
        if let Err(e) = start_profiling() {
            eprintln!("Parser: failed to start profiling: {e}");
        }

        let mut item_id = 0u64;
        while running_clone.load(Ordering::Relaxed) && item_id < num_items {
            let item = parse_stage(item_id);
            if parse_tx.send(item).is_err() {
                break;
            }
            item_id += 1;
        }
        drop(parse_tx);
        println!("Parser: processed {item_id} items");
    });

    let running_clone = Arc::clone(&running);

    // Stage 2: Transformer thread (THE BOTTLENECK)
    let transformer = thread::spawn(move || {
        if let Err(e) = start_profiling() {
            eprintln!("Transformer: failed to start profiling: {e}");
        }

        let mut count = 0u64;
        while let Ok(item) = parse_rx.recv() {
            if !running_clone.load(Ordering::Relaxed) {
                break;
            }
            let transformed = transform_stage(item);
            if transform_tx.send(transformed).is_err() {
                break;
            }
            count += 1;
        }
        drop(transform_tx);
        println!("Transformer: processed {count} items");
    });

    // Stage 3: Serializer thread
    let serializer = thread::spawn(move || {
        if let Err(e) = start_profiling() {
            eprintln!("Serializer: failed to start profiling: {e}");
        }

        let mut count = 0u64;
        while let Ok(item) = transform_rx.recv() {
            let serialized = serialize_stage(item);
            if serialize_tx.send(serialized).is_err() {
                break;
            }
            count += 1;
            items_clone.fetch_add(1, Ordering::Relaxed);
        }
        drop(serialize_tx);
        println!("Serializer: processed {count} items");
    });

    // Collector thread (drains output)
    let collector = thread::spawn(move || {
        let mut total_bytes = 0usize;
        while let Ok(data) = serialize_rx.recv() {
            total_bytes += data.len();
            black_box(&data);
        }
        println!("Collector: received {total_bytes} bytes");
    });

    // Wait for all stages to complete
    parser.join().unwrap();
    transformer.join().unwrap();
    serializer.join().unwrap();
    collector.join().unwrap();
}

fn main() {
    println!("=== Pipeline Stage Bottleneck Example ===\n");
    println!("This example demonstrates causal profiling on a multi-stage pipeline.");
    println!("The Transform stage is intentionally 10x slower than other stages.\n");
    println!("Stage timings:");
    println!("  Parse:     ~1ms per item");
    println!("  Transform: ~10ms per item (BOTTLENECK)");
    println!("  Serialize: ~1ms per item\n");

    // Initialize profiler
    if let Err(e) = init() {
        eprintln!("Failed to initialize profiler: {e}");
    }

    let num_items = 200u64;
    let running = Arc::new(AtomicBool::new(true));
    let items_processed = Arc::new(AtomicU64::new(0));

    // Phase 1: Warmup
    println!("Phase 1: Warmup run (no profiling experiments)...");
    let start = Instant::now();

    run_pipeline(50, Arc::clone(&running), Arc::clone(&items_processed));

    let warmup_elapsed = start.elapsed();
    let warmup_items = items_processed.load(Ordering::Relaxed);
    println!(
        "Warmup: {} items in {:?} ({:.1} items/sec)\n",
        warmup_items,
        warmup_elapsed,
        warmup_items as f64 / warmup_elapsed.as_secs_f64()
    );

    // Reset for profiling run
    items_processed.store(0, Ordering::Relaxed);

    // Phase 2: Run with experiments
    println!("Phase 2: Running with causal profiling experiments...");
    println!("(Main thread orchestrates experiments while workers process items)\n");

    let running_clone = Arc::clone(&running);
    let items_clone = Arc::clone(&items_processed);

    // Start pipeline in background
    let pipeline = thread::spawn(move || {
        run_pipeline(num_items, running_clone, items_clone);
    });

    // Run experiments on the main thread, measuring transform_done throughput
    // (this is the bottleneck stage, so its throughput = overall throughput)
    let _ = run_experiments("transform_done");

    // Wait for pipeline to finish
    pipeline.join().unwrap();

    let final_items = items_processed.load(Ordering::Relaxed);
    println!("\n=== Results ===");
    println!("Total items processed: {final_items}");

    println!("\n=== Expected Causal Profiling Analysis ===");
    println!("If profiling is enabled (ANDWEORC_ENABLED=1), you should see:");
    println!();
    println!("  transform_done: HIGH causal impact (~0.8-1.0)");
    println!("    - This is the bottleneck stage");
    println!("    - Optimizing it would directly improve overall throughput");
    println!();
    println!("  parse_done: LOW causal impact (~0.0-0.1)");
    println!("    - Parse is fast but blocked waiting for Transform");
    println!("    - Making it faster won't help overall throughput");
    println!();
    println!("  serialize_done: LOW causal impact (~0.0-0.1)");
    println!("    - Serialize is fast but starved by slow Transform");
    println!("    - Making it faster won't help overall throughput");
    println!();
    println!("Traditional profiling would show Transform takes 83% of time (10/(1+10+1)).");
    println!("Causal profiling reveals the causal relationship: only Transform matters.");
}
