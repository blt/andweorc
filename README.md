# andweorc - A Causal Profiler for Linux Programs

**andweorc** (Old English for "cause", [Wiktionary](https://en.wiktionary.org/wiki/andweorc#Old_English)) is a causal profiler for Linux programs. Unlike traditional profilers that show where your program spends time, andweorc answers a more useful question: **"If I optimize this code, how much faster will my program run?"**

Inspired by [Coz](https://github.com/plasma-umass/coz) and designed for production use.

> **Note:** This is a work-in-progress. The commit history may be rebased.

## Why Causal Profiling?

Traditional profilers tell you where your program spends time, but this can be misleading:

- **Hot code isn't always the bottleneck.** A function might consume 30% of CPU time, but optimizing it may yield zero improvement if it runs in parallel with slower code.
- **Amdahl's Law is hard to apply.** Even if you know a function takes 50% of runtime, the actual speedup from optimizing it depends on whether it's on the critical path.
- **Sampling doesn't show causation.** Just because code runs frequently doesn't mean it's limiting throughput.

Causal profiling answers: *"What would happen to my program's throughput if this line of code ran X% faster?"*

It does this by virtually speeding up code regions using a clever trick: instead of making target code faster (impossible without changing it), it **slows down everything else** proportionally. This simulates what would happen if that code were optimized.

## Repository Structure

```
andweorc/
├── andweorc/           # Core profiling library (Rust lib + dylib for LD_PRELOAD)
│   ├── src/
│   │   ├── experiment.rs      # Central experiment coordinator
│   │   ├── per_thread.rs      # Per-thread sampling via perf events
│   │   ├── progress_point.rs  # Throughput measurement
│   │   ├── ffi.rs             # C-compatible API
│   │   ├── runner.rs          # Experiment orchestration
│   │   └── posix/             # pthread interception for LD_PRELOAD
│   └── include/
│       └── andweorc.h         # C header for FFI
├── andweorc-macros/    # Proc-macros for #[profile] and progress!()
├── cargo-andweorc/     # Cargo subcommand for easy profiling
└── ci/                 # CI scripts (validate, test, kani proofs)
```

## Quick Start

### Option 1: Rust Programs with Instrumentation

Add andweorc to your project:

```toml
[dependencies]
andweorc = { git = "https://github.com/blt/andweorc" }
```

Mark progress points where meaningful work completes:

```rust
use andweorc::progress;

fn process_request(req: Request) -> Response {
    // ... handle the request ...
    progress!("request_complete");  // Mark one unit of work done
    response
}

fn main() {
    andweorc::init().expect("Failed to initialize profiler");
    // ... your program runs here ...
    andweorc::run_experiments("request_complete");
}
```

Run with profiling enabled:

```bash
sudo sysctl kernel.perf_event_paranoid=1
ANDWEORC_ENABLED=1 cargo run --release
```

### Option 2: Any Linux Program via LD_PRELOAD

Profile any program without source changes:

```bash
# Build the shared library
cargo build --release

# Profile any binary (C, C++, Go, Rust, etc.)
LD_PRELOAD=target/release/libandweorc.so \
ANDWEORC_ENABLED=1 \
./your_program
```

For throughput measurement, add progress points using the C API:

```c
#include "andweorc.h"

void process_item(void) {
    // ... do work ...
    andweorc_progress("item_processed");
}
```

### Option 3: Cargo Subcommand

Install and use the cargo subcommand:

```bash
cargo install --path cargo-andweorc

# Profile a binary
cargo andweorc run --bin myapp --release

# Generate a report from saved data
cargo andweorc report profile.json --format csv
```

## C API Reference

The C API allows profiling programs written in any language. Include `andweorc/include/andweorc.h` in your project.

### Functions

```c
// Mark completion of one unit of work
void andweorc_progress(const char* name);

// Mark progress using source location as name
void andweorc_progress_named(const char* file, int line);

// Begin/end a timed section (delay injection points)
void andweorc_begin(const char* name);
void andweorc_end(const char* name);
```

### Macros

```c
// Progress point at current file:line
ANDWEORC_PROGRESS();

// Named progress point
ANDWEORC_PROGRESS_NAMED("requests_done");

// Timed section
ANDWEORC_BEGIN("db_query");
// ... do query ...
ANDWEORC_END("db_query");
```

### Example: C Program

```c
#include <stdio.h>
#include "andweorc.h"

void process_batch(int* items, int count) {
    for (int i = 0; i < count; i++) {
        // ... process item ...
        andweorc_progress("item_processed");
    }
}

int main() {
    int items[1000];
    for (int i = 0; i < 100; i++) {
        process_batch(items, 1000);
    }
    return 0;
}
```

Compile and run:

```bash
gcc -o myapp myapp.c -I/path/to/andweorc/include

LD_PRELOAD=/path/to/libandweorc.so \
ANDWEORC_ENABLED=1 \
./myapp
```

## Understanding Results

After profiling, andweorc outputs results like:

```
=== Causal Profiling Results ===
Top optimization opportunities:

1. src/parser.rs:142 (impact = 0.8234)
2. src/db/query.rs:89 (impact = 0.3156)
3. src/serialize.rs:201 (impact = 0.0892)
```

The **impact score** indicates the causal relationship between code speed and program throughput:

| Impact | Interpretation |
|--------|----------------|
| > 0.5 | High-value target. Optimize this for meaningful speedup. |
| 0.1 - 0.5 | Moderate value. Worth optimizing if straightforward. |
| < 0.1 | Low value. Minimal effect on overall performance. |
| ~ 0 or negative | Not a bottleneck. Don't waste time here. |

## Detailed Usage

### Choosing Progress Points

Progress points should mark **meaningful work completion**, not arbitrary locations:

```rust
// GOOD: Marks completion of a logical unit of work
fn handle_connection(conn: TcpStream) {
    // ... process connection ...
    progress!("connection_handled");
}

// GOOD: Marks iteration completion in a batch processor
fn process_batch(items: &[Item]) {
    for item in items {
        process_item(item);
        progress!("item_processed");
    }
}

// BAD: Too fine-grained
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

Each worker thread should start profiling:

```rust
fn main() {
    andweorc::init().expect("Failed to initialize profiler");

    let handles: Vec<_> = (0..num_threads)
        .map(|_| {
            std::thread::spawn(|| {
                andweorc::start_profiling().expect("Failed to start profiling");

                loop {
                    let work = get_work();
                    process(work);
                    progress!("work_done");
                }
            })
        })
        .collect();

    andweorc::run_experiments("work_done");
}
```

### Using the Runner API

For programmatic control over experiments:

```rust
use andweorc::runner::Runner;
use std::time::Duration;

fn main() {
    andweorc::start_profiling().expect("Failed to start profiling");
    // ... start workload in background threads ...

    let runner = Runner::new();
    let results = runner.run("my_progress_point");

    // Analyze results programmatically
    for (ip, impact) in results.calculate_impacts() {
        println!("IP 0x{:x}: impact = {:.4}", ip, impact);
    }
}
```

### JSON Output

Results can be output in JSON format for integration with other tools:

```rust
use andweorc::json_output::JsonOutput;

let output = JsonOutput::new();
output.add_experiment(ip, speedup, throughput, duration, samples);
let json = output.to_json();
```

## How It Works

1. **Sample Collection**: Hardware instruction counters trigger at regular intervals, capturing which instruction each thread is executing.

2. **Virtual Speedup**: For each hot code location, the profiler runs experiments. When a sample hits the selected location, that thread proceeds normally while other threads are delayed proportionally.

3. **Delay Injection**: Delays are consumed at "yield points":
   - Progress points (`progress!()` macro)
   - Delay points (`delay_point!()` macro)
   - Synchronization operations (pthread mutex, condvar, rwlock)

4. **Throughput Measurement**: Progress points track work units completed per second during each experiment.

5. **Causal Attribution**: Linear regression correlates virtual speedup percentage with throughput change. A strong positive correlation indicates causal impact.

### Mathematical Basis

If code location X takes time `t` and we want to simulate it running `s%` faster:
- Delay all threads NOT at X by `t * s` nanoseconds
- This makes X's relative execution time decrease by approximately `s%`

Speedup percentages range from 5% to 150% across experiments.

## Requirements

- **Linux only**: Requires `perf_event_open` syscall for hardware performance counters
- **Permissions**: `kernel.perf_event_paranoid` must be ≤ 1

```bash
# Temporary (until reboot)
sudo sysctl kernel.perf_event_paranoid=1

# Permanent
echo 'kernel.perf_event_paranoid=1' | sudo tee /etc/sysctl.d/99-perf.conf
```

- **Debug info**: For symbol resolution, compile with debug info:

```toml
[profile.release]
debug = true
```

- **Frame pointers** (recommended): For accurate stack traces:

```toml
# .cargo/config.toml
[build]
rustflags = ["-C", "force-frame-pointers=yes"]
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `ANDWEORC_ENABLED` | Set to `1` to enable profiling | (disabled) |
| `ANDWEORC_ROUND_DURATION_MS` | Duration of each experiment round | `1000` |
| `ANDWEORC_BASELINE_ROUNDS` | Number of baseline measurements | `5` |

## Troubleshooting

### "failed to create hardware performance counters"

Set kernel permissions:

```bash
sudo sysctl kernel.perf_event_paranoid=1
```

Or grant capability to the binary:

```bash
sudo setcap cap_perfmon=ep ./your_binary
```

### No optimization opportunities found

- Ensure the program runs long enough (several seconds minimum)
- Check that progress points are being hit
- Verify the workload is CPU-bound; I/O-bound programs need different analysis

### Results seem random

- Increase baseline rounds for more statistical power
- Run experiments longer
- Check that the workload is consistent across experiment rounds
- Look at R² values; low R² indicates high noise

### Container/Docker Usage

Profiling in containers requires:

```bash
docker run --cap-add=SYS_PTRACE --security-opt seccomp=unconfined ...
```

Or set perf_event_paranoid inside the container:

```bash
docker run --privileged ...
```

## Verification

The codebase is extensively verified:

- **Kani formal proofs** for critical algorithms (delay injection, atomic operations)
- **Loom exhaustive tests** for concurrency correctness
- **Property-based tests** for non-trivial logic
- **Integration tests** for throughput accuracy

Run all checks:

```bash
ci/validate
```

## Limitations

- **Linux only**: Uses Linux-specific perf events and timers
- **No fork support**: Child processes aren't automatically profiled
- **Single progress point**: Current version measures throughput at one progress point per run

## License

MIT License. See LICENSE file.

## References

- [Coz: Finding Code that Counts with Causal Profiling](https://arxiv.org/abs/1608.03676) - The original paper
- [Coz Profiler](https://github.com/plasma-umass/coz) - C/C++ implementation
