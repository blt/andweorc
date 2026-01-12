# andweorc - A Causal Profiler for Rust

**andweorc** (Old English for "cause", [Wiktionary](https://en.wiktionary.org/wiki/andweorc#Old_English)) is a causal profiler for Rust programs. Unlike traditional profilers that show where your program spends time, andweorc answers a more useful question: **"If I optimize this code, how much faster will my program run?"**

Inspired by [Coz](https://github.com/plasma-umass/coz) and designed for Rust's ecosystem.

> **Note:** This is a work-in-progress. The commit history may be rebased. Please be aware of this if you fork.

## Why Causal Profiling?

Traditional profilers tell you where your program spends time, but this can be misleading:

- **Hot code isn't always the bottleneck.** A function might consume 30% of CPU time, but optimizing it may yield zero improvement if it runs in parallel with slower code.
- **Amdahl's Law is hard to apply.** Even if you know a function takes 50% of runtime, the actual speedup from optimizing it depends on whether it's on the critical path.
- **Sampling doesn't show causation.** Just because code runs frequently doesn't mean it's limiting throughput.

Causal profiling answers: *"What would happen to my program's throughput if this line of code ran X% faster?"*

It does this by virtually speeding up code regions using a clever trick: instead of making target code faster (impossible without changing it), it **slows down everything else** proportionally. This simulates what would happen if that code were optimized.

## Quick Start

### 1. Add andweorc to your project

```toml
[dependencies]
andweorc = { git = "https://github.com/troutwine/andweorc" }
```

### 2. Mark progress points in your code

Progress points define what "throughput" means for your program. Place them where meaningful work completes:

```rust
use andweorc::progress;

fn process_request(req: Request) -> Response {
    // ... handle the request ...

    progress!("request_complete");  // Mark one unit of work done
    response
}
```

### 3. Initialize the profiler

```rust
fn main() {
    // Initialize profiler (only active when ANDWEORC_ENABLED=1)
    andweorc::init().expect("Failed to initialize profiler");

    // ... your program runs here ...

    // Run experiments and print results
    andweorc::run_experiments("request_complete");
}
```

### 4. Run with profiling enabled

```bash
# Set permissions for perf events (one-time setup)
sudo sysctl kernel.perf_event_paranoid=-1

# Run your program with profiling
ANDWEORC_ENABLED=1 cargo run --release
```

## Understanding Results

After running experiments, andweorc outputs results like:

```
=== Causal Profiling Results ===
Top optimization opportunities:

1. src/parser.rs:142 (impact = 0.8234)
2. src/db/query.rs:89 (impact = 0.3156)
3. src/serialize.rs:201 (impact = 0.0892)
```

The **impact score** indicates the causal relationship between code speed and program throughput:

- **Impact > 0.5**: High-value optimization target. Speeding up this code will meaningfully improve throughput.
- **Impact 0.1 - 0.5**: Moderate value. Worth optimizing if it's straightforward.
- **Impact < 0.1**: Low value. Optimizing this code will have minimal effect on overall performance.
- **Impact near 0 or negative**: Not a bottleneck. Don't waste time here.

## Detailed Usage Guide

### Choosing Progress Points

Progress points should mark **meaningful work completion**, not arbitrary locations. Good progress points:

```rust
// Good: Marks completion of a logical unit of work
fn handle_connection(conn: TcpStream) {
    // ... process connection ...
    progress!("connection_handled");
}

// Good: Marks iteration completion in a batch processor
fn process_batch(items: &[Item]) {
    for item in items {
        process_item(item);
        progress!("item_processed");
    }
}
```

Bad progress points:

```rust
// Bad: Too fine-grained, doesn't represent meaningful work
fn compute(data: &[f64]) -> f64 {
    let mut sum = 0.0;
    for &x in data {
        sum += x;
        progress!("addition");  // Don't do this!
    }
    sum
}
```

### Multi-threaded Programs

For multi-threaded programs, each thread that participates in the workload should start profiling:

```rust
fn main() {
    andweorc::init().expect("Failed to initialize profiler");

    let handles: Vec<_> = (0..num_threads)
        .map(|_| {
            std::thread::spawn(|| {
                // Start profiling on this worker thread
                andweorc::start_profiling().expect("Failed to start profiling");

                loop {
                    let work = get_work();
                    process(work);
                    progress!("work_done");
                }
            })
        })
        .collect();

    // Run experiments while workers are active
    andweorc::run_experiments("work_done");

    // Signal workers to stop and join
    // ...
}
```

### Using the Runner API for Custom Experiments

For more control, use the `Runner` API directly:

```rust
use andweorc::runner::{Runner, RunnerConfig};

fn main() {
    andweorc::start_profiling().expect("Failed to start profiling");

    // Start your workload in background threads...

    let config = RunnerConfig {
        round_duration: Duration::from_secs(2),
        baseline_rounds: 10,
        speedup_percentages: vec![0.0, 0.25, 0.5, 0.75, 1.0],
        max_ips: 20,
    };

    let runner = Runner::with_config(config);
    let results = runner.run("my_progress_point");

    // Analyze results programmatically
    for (ip, impact) in results.calculate_impacts() {
        println!("IP 0x{:x}: impact = {:.4}", ip, impact);
    }

    // Get statistically significant results only
    for (ip, stats) in results.significant_impacts() {
        println!(
            "IP 0x{:x}: impact = {:.4} (R² = {:.2}, 95% CI: [{:.4}, {:.4}])",
            ip,
            stats.slope,
            stats.r_squared,
            stats.slope_ci_lower,
            stats.slope_ci_upper
        );
    }
}
```

### Interpreting Statistical Output

The full statistical output includes:

- **slope**: The causal impact (throughput change per unit speedup)
- **R²**: Goodness of fit (0-1). Higher means the linear model fits well.
- **95% CI**: Confidence interval for the slope. If it doesn't include zero, the result is statistically significant.

```rust
let stats = results.calculate_impacts_with_stats();
for (ip, reg) in stats {
    if reg.is_significant() && reg.has_good_fit() {
        println!("High-confidence optimization target: 0x{:x}", ip);
        println!("  Expected speedup: {:.1}% per 10% code improvement",
                 reg.slope * 0.1 * 100.0);
    }
}
```

## Example: Finding the Real Bottleneck

Consider a program that processes data through multiple stages:

```rust
fn process_pipeline(data: &[Record]) {
    let parsed = parse_records(data);           // 10% of CPU time
    let validated = validate_records(&parsed);   // 20% of CPU time
    let transformed = transform_records(&validated); // 40% of CPU time
    let result = aggregate_results(&transformed);    // 30% of CPU time

    progress!("pipeline_complete");
}
```

A traditional profiler would say "optimize `transform_records` - it's 40% of runtime!"

But causal profiling might reveal:

```
1. validate_records (impact = 0.72)
2. transform_records (impact = 0.15)
3. aggregate_results (impact = 0.08)
4. parse_records (impact = 0.03)
```

The validation step, despite being only 20% of CPU time, is the actual bottleneck! Perhaps it's doing synchronous I/O, holding a lock, or blocking on something that serializes the pipeline.

## Requirements

- **Linux only**: Requires `perf_event_open` syscall for hardware performance counters
- **Permissions**: Either run as root, or set `kernel.perf_event_paranoid` to -1 or 1
- **Frame pointers**: For accurate stack traces, compile with frame pointers:

```toml
# In Cargo.toml
[profile.release]
debug = true  # Keep debug info for symbol resolution
```

Or in `.cargo/config.toml`:
```toml
[build]
rustflags = ["-C", "force-frame-pointers=yes"]
```

## How It Works

1. **Sample collection**: The profiler periodically samples all threads to find where they're executing (instruction pointers).

2. **Virtual speedup**: For each hot code location, the profiler runs experiments. When a sample hits the selected location, other threads are delayed proportionally - simulating that code running faster.

3. **Throughput measurement**: Progress points track how many units of work complete per second during each experiment.

4. **Causal attribution**: Linear regression correlates virtual speedup percentage with throughput change. A strong positive correlation means that code location causally affects performance.

## Troubleshooting

### "perf sampling not available"

Set kernel permissions:
```bash
sudo sysctl kernel.perf_event_paranoid=-1
```

Or for a permanent fix, add to `/etc/sysctl.conf`:
```
kernel.perf_event_paranoid=-1
```

### No optimization opportunities found

- Ensure your program runs long enough to collect samples (several seconds minimum)
- Check that progress points are being hit (they should be in the hot path)
- Verify the workload is CPU-bound; I/O-bound programs need different analysis

### Results seem random

- Increase `baseline_rounds` for more statistical power
- Run experiments longer with `round_duration`
- Look at R² values; low R² means high noise
- Check that the workload is consistent across experiment rounds

## Limitations

- **Linux only**: Uses Linux-specific perf events and timers
- **No fork support**: Child processes aren't automatically profiled
- **Frame pointers needed**: For accurate stack unwinding

## License

See LICENSE file.

## References

- [Coz: Finding Code that Counts with Causal Profiling](https://arxiv.org/abs/1608.03676) - The original paper
- [Coz Profiler](https://github.com/plasma-umass/coz) - C/C++ implementation
