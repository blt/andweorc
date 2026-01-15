# Andweorc Examples

Comprehensive examples demonstrating **causal profiling** effectiveness, inspired by the
[Coz paper](https://sigops.org/s/conferences/sosp/2015/current/2015-Monterey/printable/090-curtsinger.pdf)
(SOSP 2015, Best Paper Award).

## What is Causal Profiling?

Traditional profilers show where time is spent. Causal profiling answers a different question:
**"If I optimize this code, how much faster will my program run?"**

The key mechanism:
1. Select a code location (instruction pointer)
2. Inject **virtual speedups** by delaying threads NOT at that location
3. Measure throughput change
4. Calculate **causal impact** via linear regression

Every example in this directory calls `run_experiments()` to perform full causal profiling.
We do not include sampling-only examples - that's not what andweorc is for.

## Requirements

- Linux with `kernel.perf_event_paranoid <= 1` (or CAP_PERFMON capability)
- GCC for C examples
- Hardware performance counter support (most modern x86_64 CPUs)

```bash
# Enable perf events (temporary)
sudo sysctl kernel.perf_event_paranoid=1

# Enable perf events (permanent)
echo 'kernel.perf_event_paranoid=1' | sudo tee /etc/sysctl.d/99-perf.conf
```

## C Examples

Located in `examples/c/`. Build with:

```bash
cd examples/c
make all
```

### 1. Cache Contention (Memcached-inspired)

**File:** `cache_contention.c`

Multi-threaded hash table cache with per-bucket mutex locks. Inspired by the Coz
paper's Memcached analysis which achieved **9% improvement**.

**How it works:**
- 8 worker threads perform cache get/set operations
- Each bucket has its own mutex lock (contention point)
- Main thread calls `andweorc_run_experiments("cache_op_done")`
- Profiler tests virtual speedups and calculates causal impact

**Run:**
```bash
LD_PRELOAD=../../target/release/libandweorc.so ANDWEORC_ENABLED=1 ./cache_contention
```

### 2. Dedup Hash Table (coz paper inspired)

**File:** `dedup_hash.c`

Deduplication workload with intentionally bad hash function creating hot buckets.
From the coz paper, fixing hash distribution in dedup yielded **8% improvement**.

**How it works:**
- Bad hash function creates uneven bucket distribution
- Long chain traversals become the bottleneck
- Main thread calls `andweorc_run_experiments("dedup_done")`
- Profiler identifies chain traversal as high-impact code

**Run:**
```bash
LD_PRELOAD=../../target/release/libandweorc.so ANDWEORC_ENABLED=1 ./dedup_hash
```

## Rust Examples

Located in `andweorc/examples/`. Build and run with cargo:

### 1. Pipeline Stages

**File:** `andweorc/examples/pipeline_stages.rs`

Multi-stage processing pipeline demonstrating stage bottleneck identification.

**Stage timings:**
- Parse: ~1ms per item (fast)
- Transform: ~10ms per item (BOTTLENECK)
- Serialize: ~1ms per item (fast)

**How it works:**
- 4-thread pipeline runs continuously in background
- Main thread calls `run_experiments("transform_done")`
- Profiler identifies Transform stage as high-impact

**Run:**
```bash
ANDWEORC_ENABLED=1 cargo run --example pipeline_stages --release
```

### 2. Barrier Phases (PARSEC-inspired)

**File:** `andweorc/examples/barrier_phases.rs`

Barrier-synchronized workload similar to PARSEC benchmarks. Demonstrates the
counterintuitive insight that barrier code can show **negative causal impact**.

**How it works:**
- 4 threads synchronized by barriers
- Phase A: light work (10K iterations)
- Phase B: heavy work (100K iterations) - BOTTLENECK
- Main thread calls `run_experiments("iteration_done")`

**Run:**
```bash
ANDWEORC_ENABLED=1 cargo run --example barrier_phases --release
```

### 3. Parallel Reduce (MapReduce pattern)

**File:** `andweorc/examples/parallel_reduce.rs`

MapReduce-style parallel reduction with barrier synchronization.

**How it works:**
- 4 threads compute partial sums (MAP phase)
- Thread 0 reduces all partials (REDUCE phase)
- Main thread calls `run_experiments("reduce_done")`
- Profiler shows whether MAP or REDUCE is the bottleneck

**Run:**
```bash
ANDWEORC_ENABLED=1 cargo run --example parallel_reduce --release
```

### 4. Lock Contention

**File:** `andweorc/examples/lock_contention.rs`

Identifies mutex lock contention as a bottleneck.

**Run:**
```bash
ANDWEORC_ENABLED=1 cargo run --example lock_contention --release
```

### 5. Sorting Comparison

**File:** `andweorc/examples/sorting_comparison.rs`

Compares causal impact of different sorting algorithms.

**Run:**
```bash
ANDWEORC_ENABLED=1 cargo run --example sorting_comparison --release
```

### 6. Self Profile

**File:** `andweorc/examples/self_profile.rs`

Profiles andweorc's own algorithms to validate the profiler.

**Run:**
```bash
ANDWEORC_ENABLED=1 cargo run --example self_profile --release
```

## Understanding Causal Impact Values

- **Impact > 0**: Speeding up this code improves throughput
- **Impact ~0**: Not on critical path
- **Impact < 0**: Optimizing would INCREASE contention (key insight!)

### Negative Impact and Barriers

Barrier-synchronized workloads show counterintuitive patterns. When you "virtually
speed up" code by delaying other threads:
- The selected thread finishes faster
- But it waits at the barrier for delayed threads
- Net result: iteration takes LONGER (negative correlation)

**What negative impact tells you:**
- "Optimizing this specific code won't improve performance"
- "The problem is the synchronization pattern, not the code efficiency"
- The fix is to REPLACE the barrier, not optimize the hot code

**Real-world example from Coz paper:**
- PARSEC's fluidanimate and streamcluster used custom spin barriers
- Traditional profilers showed the spin-wait as a hot spot
- **Wrong fix:** Optimize the spin-wait loop
- **Right fix:** Replace spin barrier with pthread_barrier
- Result: 37.5% and 68.4% speedups respectively

## Signal-Based Experiment Triggering

For profiling programs you can't modify, andweorc supports triggering experiments
via SIGUSR1. This enables causal profiling of third-party applications.

### How It Works

1. Run the target program with `LD_PRELOAD` and `ANDWEORC_ENABLED=1`
2. The program must have progress points (via intercepted pthread functions or explicit calls)
3. Send `SIGUSR1` to trigger experiments: `kill -USR1 <pid>`
4. Experiments run at the next progress point visit

### Environment Variables

- `ANDWEORC_EXPERIMENT_TARGET`: Name of progress point to measure throughput (default: "default")

### Example: Profiling Any Program

```bash
# Terminal 1: Start the program with profiling enabled
LD_PRELOAD=/path/to/libandweorc.so ANDWEORC_ENABLED=1 \
  ANDWEORC_EXPERIMENT_TARGET="request_done" ./my_server

# Terminal 2: Let it warm up, then trigger experiments
sleep 30  # Warmup period
kill -USR1 $(pgrep my_server)
# Watch terminal 1 for profiling output
```

### Example: Profiling a Rust Benchmark

```bash
# Terminal 1: Run the dedup hash example
LD_PRELOAD=../../target/release/libandweorc.so ANDWEORC_ENABLED=1 \
  ANDWEORC_EXPERIMENT_TARGET="dedup_done" ./dedup_hash &
PID=$!

# Wait for warmup, then trigger experiments via signal
sleep 5
kill -USR1 $PID

# Wait for experiments to complete
wait $PID
```

### Progress Points from C Code

Programs can explicitly mark progress points even if they're being profiled
via LD_PRELOAD:

```c
#include <andweorc.h>

void process_item(Item* item) {
    // ... do work ...
    andweorc_progress("item_done");  // Mark completion
}
```

### Programmatic Triggering

You can also trigger experiments from code:

```c
// After warmup period
andweorc_trigger_experiments("request_done");
```

## Troubleshooting

### "Operation not supported" errors

Hardware performance counters may not be available in:
- Virtual machines without nested virtualization
- Containers without proper permissions
- Systems with restricted perf_event_paranoid

**Fix:**
```bash
sudo sysctl kernel.perf_event_paranoid=1
# or grant capability:
sudo setcap cap_perfmon=ep ./your_binary
```

### Examples run but no profiling output

1. Verify `ANDWEORC_ENABLED=1` is set
2. Check that `libandweorc.so` path is correct for `LD_PRELOAD`
3. Ensure `kernel.perf_event_paranoid <= 1`

## References

- [Coz Paper (SOSP 2015)](https://sigops.org/s/conferences/sosp/2015/current/2015-Monterey/printable/090-curtsinger.pdf)
- [The Morning Paper - Coz analysis](https://blog.acolyer.org/2015/10/14/coz-finding-code-that-counts-with-causal-profling/)
- [Original Coz implementation](https://github.com/plasma-umass/coz)
