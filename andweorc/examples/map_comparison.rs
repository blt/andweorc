//! Example: `HashMap` vs `BTreeMap` comparison.
//!
//! This example compares `HashMap` and `BTreeMap` performance patterns
//! to demonstrate how causal profiling can identify which data
//! structure is the bottleneck for different access patterns.
//!
//! `HashMap`: O(1) average lookup, O(n) worst case
//! `BTreeMap`: O(log n) lookup, better cache locality for range queries

// Examples are demonstration code - allow more relaxed rules
#![allow(clippy::unwrap_used)]
#![allow(clippy::expect_used)]
#![allow(clippy::print_stdout)]
#![allow(clippy::uninlined_format_args)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::doc_markdown)]
#![allow(clippy::explicit_iter_loop)]
#![allow(dead_code)]
#![allow(unused_assignments)]

use andweorc::progress;
use std::collections::{BTreeMap, HashMap};
use std::time::Instant;

/// Simple hash function that can cause collisions.
fn bad_hash(key: u64) -> u64 {
    // This creates clustering for demonstration
    key % 1000
}

/// Workload: Random lookups (HashMap should be faster).
fn random_lookup_hashmap(map: &HashMap<u64, String>, keys: &[u64]) -> usize {
    let mut found = 0;
    for &key in keys {
        if map.contains_key(&key) {
            found += 1;
        }
    }
    progress!("hashmap_lookup_done");
    found
}

/// Workload: Random lookups (BTreeMap is slower for this).
fn random_lookup_btreemap(map: &BTreeMap<u64, String>, keys: &[u64]) -> usize {
    let mut found = 0;
    for &key in keys {
        if map.contains_key(&key) {
            found += 1;
        }
    }
    progress!("btreemap_lookup_done");
    found
}

/// Workload: Range iteration (BTreeMap should be faster).
fn range_iteration_btreemap(map: &BTreeMap<u64, String>, start: u64, end: u64) -> usize {
    let mut count = 0;
    for (_k, v) in map.range(start..end) {
        count += v.len();
    }
    progress!("btreemap_range_done");
    count
}

/// Workload: Range iteration simulation for HashMap (have to check all keys).
fn range_iteration_hashmap(map: &HashMap<u64, String>, start: u64, end: u64) -> usize {
    let mut count = 0;
    for (k, v) in map.iter() {
        if *k >= start && *k < end {
            count += v.len();
        }
    }
    progress!("hashmap_range_done");
    count
}

/// Generate lookup keys using simple LCG.
fn generate_keys(count: usize, max_key: u64, seed: u64) -> Vec<u64> {
    let mut keys = Vec::with_capacity(count);
    let mut state = seed;
    for _ in 0..count {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        keys.push(state % max_key);
    }
    keys
}

fn main() {
    println!("Map Comparison Example");
    println!("======================");
    println!();
    println!("Comparing HashMap vs BTreeMap for different access patterns.");
    println!();

    let map_size = 100_000;
    let lookup_count = 50_000;
    let iterations = 100;

    // Build the maps
    println!("Building maps with {} entries...", map_size);
    let mut hashmap: HashMap<u64, String> = HashMap::with_capacity(map_size);
    let mut btreemap: BTreeMap<u64, String> = BTreeMap::new();

    for i in 0..map_size as u64 {
        let value = format!("value_{}", i);
        hashmap.insert(i, value.clone());
        btreemap.insert(i, value);
    }

    // Generate lookup keys
    let lookup_keys = generate_keys(lookup_count, map_size as u64 * 2, 12345);

    // Test 1: Random lookups
    println!();
    println!(
        "Test 1: Random Lookups ({} lookups x {} iterations)",
        lookup_count, iterations
    );

    let start = Instant::now();
    let mut total_found = 0;
    for _ in 0..iterations {
        total_found += random_lookup_hashmap(&hashmap, &lookup_keys);
    }
    let hashmap_lookup_time = start.elapsed();
    println!(
        "  HashMap: {:?} ({:.2}M lookups/sec)",
        hashmap_lookup_time,
        (lookup_count * iterations) as f64 / hashmap_lookup_time.as_secs_f64() / 1_000_000.0
    );

    let start = Instant::now();
    total_found = 0;
    for _ in 0..iterations {
        total_found += random_lookup_btreemap(&btreemap, &lookup_keys);
    }
    let btreemap_lookup_time = start.elapsed();
    println!(
        "  BTreeMap: {:?} ({:.2}M lookups/sec)",
        btreemap_lookup_time,
        (lookup_count * iterations) as f64 / btreemap_lookup_time.as_secs_f64() / 1_000_000.0
    );

    // Test 2: Range queries
    println!();
    println!(
        "Test 2: Range Queries (10% range x {} iterations)",
        iterations
    );

    let range_start = (map_size as u64) / 4;
    let range_end = range_start + (map_size as u64) / 10;

    let start = Instant::now();
    let mut total_chars = 0;
    for _ in 0..iterations {
        total_chars += range_iteration_btreemap(&btreemap, range_start, range_end);
    }
    let btreemap_range_time = start.elapsed();
    println!("  BTreeMap range: {:?}", btreemap_range_time);

    let start = Instant::now();
    total_chars = 0;
    for _ in 0..iterations {
        total_chars += range_iteration_hashmap(&hashmap, range_start, range_end);
    }
    let hashmap_range_time = start.elapsed();
    println!("  HashMap iterate: {:?}", hashmap_range_time);

    println!();
    println!("Summary:");
    println!(
        "  Random lookup: HashMap is {:.2}x faster",
        btreemap_lookup_time.as_secs_f64() / hashmap_lookup_time.as_secs_f64()
    );
    println!(
        "  Range query: BTreeMap is {:.2}x faster",
        hashmap_range_time.as_secs_f64() / btreemap_range_time.as_secs_f64()
    );
    println!();
    println!("Causal profiling would help identify which pattern dominates");
    println!("your actual workload and where optimization effort should focus.");

    // Prevent optimizing away
    std::hint::black_box(total_found);
    std::hint::black_box(total_chars);
}
