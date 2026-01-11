//! Self-profiling example: use andweorc to find optimization opportunities in its own code.
//!
//! This example exercises the core algorithms used by the profiler:
//! - Hash function for sample counting
//! - Linear probing lookups
//! - Linear regression calculation
//! - Clock access patterns
//!
//! Run with: ANDWEORC_ENABLED=1 cargo run --example self_profile --release

// Replicate the core algorithms from andweorc for profiling

/// FNV-1a hash (same as in experiment.rs)
#[inline(never)]
fn hash_ip(ip: usize) -> usize {
    andweorc::progress!("hash_start");
    const FNV_OFFSET: usize = 0xcbf2_9ce4_8422_2325_usize;
    const FNV_PRIME: usize = 0x0100_0000_01b3_usize;

    let mut hash = FNV_OFFSET;
    for i in 0..std::mem::size_of::<usize>() {
        let byte = (ip >> (i * 8)) & 0xFF;
        hash ^= byte;
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    let result = hash & 8191;
    andweorc::progress!("hash_done");
    result
}

/// Simulate linear probing lookup (core of lock-free sample counting)
#[inline(never)]
fn linear_probe_lookup(table: &[(usize, u64)], target_ip: usize) -> Option<u64> {
    andweorc::progress!("probe_start");
    let start = hash_ip(target_ip);
    let len = table.len();

    for i in 0..len {
        let idx = (start + i) & (len - 1);
        let (stored_ip, count) = table[idx];

        if stored_ip == target_ip {
            andweorc::progress!("probe_found");
            return Some(count);
        }
        if stored_ip == 0 {
            andweorc::progress!("probe_miss");
            return None;
        }
    }
    andweorc::progress!("probe_full");
    None
}

/// Linear regression calculation (same algorithm as runner.rs)
#[inline(never)]
fn calculate_linear_regression(xs: &[f64], ys: &[f64]) -> f64 {
    andweorc::progress!("regression_start");

    #[allow(clippy::cast_precision_loss)]
    let n = xs.len() as f64;
    if n < 2.0 {
        return 0.0;
    }

    // Calculate means
    let sum_x: f64 = xs.iter().sum();
    let sum_y: f64 = ys.iter().sum();
    let mean_x = sum_x / n;
    let mean_y = sum_y / n;

    // Calculate slope
    let mut numerator = 0.0;
    let mut denominator = 0.0;
    for (&x, &y) in xs.iter().zip(ys.iter()) {
        let dx = x - mean_x;
        let dy = y - mean_y;
        numerator += dx * dy;
        denominator += dx * dx;
    }

    let result = if denominator.abs() < f64::EPSILON {
        0.0
    } else {
        numerator / denominator
    };

    andweorc::progress!("regression_done");
    result
}

/// Simulate clock access pattern (called on every progress point visit)
#[inline(never)]
fn get_monotonic_time() -> u64 {
    let mut ts = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    unsafe {
        libc::clock_gettime(libc::CLOCK_MONOTONIC, &raw mut ts);
    }
    #[allow(clippy::cast_sign_loss)]
    let nanos = (ts.tv_sec as u64) * 1_000_000_000 + (ts.tv_nsec as u64);
    nanos
}

/// Simulate the throughput calculation
#[inline(never)]
fn calculate_throughput(visits: u32, first_ns: u64, last_ns: u64) -> f64 {
    andweorc::progress!("throughput_start");

    if visits < 2 {
        return 0.0;
    }
    if first_ns == 0 || last_ns == 0 || last_ns <= first_ns {
        return 0.0;
    }

    let elapsed_ns = last_ns - first_ns;
    let visits_f64 = f64::from(visits - 1);
    #[allow(clippy::cast_precision_loss)]
    let elapsed_secs = elapsed_ns as f64 / 1_000_000_000.0;
    let result = visits_f64 / elapsed_secs;

    andweorc::progress!("throughput_done");
    result
}

fn main() {
    // Initialize the profiler
    if let Err(e) = andweorc::init() {
        eprintln!("Failed to initialize profiler: {e}");
        return;
    }

    println!("Running self-profiling workload...");
    println!("This exercises the core algorithms used by andweorc.");

    // Create a simulated sample count table
    let mut sample_table: Vec<(usize, u64)> = vec![(0, 0); 8192];

    // Pre-populate with some entries
    for i in 0..1000 {
        let ip = 0x4000_0000 + i * 0x1000;
        let bucket = hash_ip(ip);
        sample_table[bucket] = (ip, (i % 100) as u64);
    }

    // Create test data for linear regression
    let xs: Vec<f64> = (0..50).map(|i| i as f64 * 0.05).collect();
    let ys: Vec<f64> = xs
        .iter()
        .map(|x| 1000.0 + x * 50.0 + (x * 10.0).sin())
        .collect();

    let iterations = 100_000;
    let mut total_hash = 0usize;
    let mut total_probes = 0u64;
    let mut total_slopes = 0.0;
    let mut total_throughput = 0.0;

    for i in 0..iterations {
        // Exercise hash function
        let ip = 0x4000_0000 + (i * 0x1234) % 0x1000_0000;
        total_hash += hash_ip(ip);

        // Exercise linear probing
        if let Some(count) = linear_probe_lookup(&sample_table, ip) {
            total_probes += count;
        }

        // Exercise linear regression (less frequently - it's slower)
        if i % 100 == 0 {
            total_slopes += calculate_linear_regression(&xs, &ys);
        }

        // Exercise clock access and throughput calculation
        let now = get_monotonic_time();
        let first = now.saturating_sub(1_000_000_000); // 1 second ago
        total_throughput += calculate_throughput((i % 1000 + 2) as u32, first, now);

        // Main progress point for throughput measurement
        andweorc::progress!("iteration_done");
    }

    // Prevent dead code elimination
    println!(
        "Completed {iterations} iterations. hash_sum={total_hash}, probes={total_probes}, slopes={total_slopes:.2}, throughput_sum={total_throughput:.2}"
    );

    // Run experiments
    if let Some(results) = andweorc::run_experiments("iteration_done") {
        let impacts = results.calculate_impacts();
        println!("\n=== Self-Profile Results ===");
        println!("Found {} optimization candidates", impacts.len());
    }
}
