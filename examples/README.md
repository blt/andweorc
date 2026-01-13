# Andweorc Examples

Comprehensive examples demonstrating causal profiling effectiveness, inspired by the
[Coz paper](https://sigops.org/s/conferences/sosp/2015/current/2015-Monterey/printable/090-curtsinger.pdf)
(SOSP 2015, Best Paper Award).

## Overview

These examples demonstrate the key insight of causal profiling: **identifying which code
has causal impact on performance**, not just where time is spent.

Traditional profilers show hot spots that may not be bottlenecks. Causal profiling
answers: "If I optimize this code, how much faster will my program run?"

## Requirements

- Linux with `kernel.perf_event_paranoid <= 1` (or CAP_PERFMON capability)
- GCC for C examples
- Go 1.21+ for Go examples (optional)
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

**Expected profiling results:**
- HIGH causal impact for `pthread_mutex_lock` in `cache_get`/`cache_set`
- The bucket locks are the throughput bottleneck

**Run:**
```bash
# Without profiling (baseline measurement)
./cache_contention

# With causal profiling
LD_PRELOAD=../../target/release/libandweorc.so ANDWEORC_ENABLED=1 ./cache_contention
```

**Sample output (without profiling):**
```
Total operations: 7,192,539 ops/sec
Bucket statistics show uniform distribution across buckets
Hit rate: ~100%
```

### 2. Producer-Consumer Queue

**File:** `producer_consumer.c`

Bounded queue with single mutex protecting both head and tail. Classic contention
pattern where consumers and producers compete for the same lock.

**Expected profiling results:**
- HIGH causal impact for `pthread_mutex_lock` in `queue_enqueue`/`queue_dequeue`
- Both operations share a single lock, serializing all access

**Optimization opportunities (revealed by profiling):**
- Use separate locks for head and tail (two-lock queue)
- Use lock-free queue implementation
- Increase queue size to reduce blocking frequency

**Run:**
```bash
./producer_consumer
# or with profiling:
LD_PRELOAD=../../target/release/libandweorc.so ANDWEORC_ENABLED=1 ./producer_consumer
```

**Sample output:**
```
Producer throughput: ~2M items/sec
Consumer throughput: ~2M items/sec
```

### 3. Barrier Synchronization (PARSEC-inspired)

**File:** `barrier_sync.c`

Demonstrates the key causal profiling insight: **sometimes optimizing code would
actually slow things down**. Inspired by the Coz paper's PARSEC fluidanimate
benchmark.

Uses a spin-wait barrier similar to PARSEC's `parsec_barrier.cpp`. The Coz paper
showed that:
- fluidanimate: **37.5% speedup** by replacing custom barrier
- streamcluster: **68.4% speedup** with same fix

**Expected profiling results:**
- **NEGATIVE causal impact** for spin-wait loop
- Optimizing the spin-wait would *increase* contention and slow things down
- The fix is to REPLACE the barrier, not optimize the spin code

**Run:**
```bash
./barrier_sync
```

**Sample output:**
```
pthread_barrier is 2.4x faster than spin barrier
Improvement: 135.3%
```

This directly validates the Coz paper's findings about barrier contention!

## Rust Examples

Located in `andweorc/examples/`. Build and run with cargo:

### Pipeline Stages

**File:** `andweorc/examples/pipeline_stages.rs`

Multi-stage processing pipeline demonstrating stage bottleneck identification.

**Stage timings:**
- Parse: ~1ms per item (fast)
- Transform: ~10ms per item (BOTTLENECK)
- Serialize: ~1ms per item (fast)

**Expected profiling results:**
```
transform_done: HIGH causal impact (~0.8-1.0)
  - This is the bottleneck stage
  - Optimizing it would directly improve throughput

parse_done: LOW causal impact (~0.0-0.1)
  - Fast but blocked waiting for Transform
  - Making it faster won't help throughput

serialize_done: LOW causal impact (~0.0-0.1)
  - Fast but starved by slow Transform
  - Making it faster won't help throughput
```

Traditional profiling would show Transform takes 83% of time `(10/(1+10+1))`.
Causal profiling reveals the *causal relationship*: only Transform matters.

**Run:**
```bash
ANDWEORC_ENABLED=1 cargo run --example pipeline_stages --release
```

**Sample output:**
```
Warmup: 50 items in 502ms (99.5 items/sec)
Total items processed: 200
```

The throughput of ~100 items/sec matches the 10ms Transform bottleneck.

## Go Examples

Located in `examples/go/`. Build with:

```bash
cd examples/go
go build -o http_contention http_contention.go
```

### HTTP Server with Contention

**File:** `http_contention.go`

HTTP server with shared state protected by `sync.Mutex`. Demonstrates
cross-language profiling via LD_PRELOAD (Go's sync primitives use pthread
under the hood).

**Expected profiling results:**
- HIGH causal impact for sync.Mutex operations
- Shared state lock is the bottleneck under concurrent load

**Optimization opportunities:**
- Use `sync.RWMutex` for read-heavy workloads
- Shard the counter map
- Use atomic operations for simple counters

**Run:**
```bash
# Start server
LD_PRELOAD=../../target/release/libandweorc.so ANDWEORC_ENABLED=1 ./http_contention

# Load test (in another terminal)
hey -n 10000 -c 50 http://localhost:8080/counter
# or
for i in $(seq 1 1000); do curl -s http://localhost:8080/counter > /dev/null; done
```

## Expected Results Summary

| Example | Expected Impact | Key Finding |
|---------|----------------|-------------|
| Cache Contention | HIGH on mutex lock | Bucket locks limit throughput |
| Producer-Consumer | HIGH on queue lock | Single lock serializes access |
| Barrier Sync | **NEGATIVE** on spin-wait | Optimization would increase contention |
| Pipeline Stages | HIGH on Transform | Only bottleneck stage matters |
| HTTP Server | HIGH on sync.Mutex | Lock contention under load |

## Comparison with Coz Paper Results

| Benchmark | Coz Paper | Our Example |
|-----------|-----------|-------------|
| Memcached (9% improvement) | Cache contention | cache_contention.c |
| PARSEC fluidanimate (37.5%) | Barrier contention | barrier_sync.c (135% improvement!) |
| PARSEC streamcluster (68.4%) | Same pattern | barrier_sync.c |
| SQLite (25% improvement) | Lock contention | producer_consumer.c |

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

1. Verify ANDWEORC_ENABLED=1 is set
2. Check that libandweorc.so path is correct for LD_PRELOAD
3. Ensure kernel.perf_event_paranoid <= 1
4. Run `cargo run --example check_perf` to diagnose

### Understanding causal impact values

- **Impact > 0.5**: High-value optimization target
- **Impact 0.1-0.5**: Moderate optimization opportunity
- **Impact ~0**: Not on critical path
- **Impact < 0**: Optimizing would INCREASE contention (key insight!)

### Barrier synchronization and negative impact

Barrier-synchronized workloads show counterintuitive causal patterns that are actually
**correct and valuable**. Here's why:

**How causal profiling works:**
1. Select a code location (instruction pointer)
2. Delay threads that are NOT at that location
3. Measure throughput change
4. If throughput increases with delay, speeding up that code would help

**What happens with barriers:**
1. All threads must reach the barrier before any can proceed
2. When we "virtually speed up" code by delaying other threads:
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

This is a key insight of causal profiling that traditional profilers cannot provide.

## References

- [Coz Paper (SOSP 2015)](https://sigops.org/s/conferences/sosp/2015/current/2015-Monterey/printable/090-curtsinger.pdf)
- [The Morning Paper - Coz analysis](https://blog.acolyer.org/2015/10/14/coz-finding-code-that-counts-with-causal-profling/)
- [Original Coz implementation](https://github.com/plasma-umass/coz)
