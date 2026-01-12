# Andweorc Development Guide

Andweorc (Old English for "cause") is a causal profiler for Rust programs. It
implements the Coz approach: insert virtual speedups via delays to non-sampled
threads, measure throughput change, and identify which source lines have causal
impact on performance.

**This program MUST be correct.** Incorrect profiling data is worse than no
data—it leads developers to optimize the wrong code. Every component must be
verified to the extent possible.

## Non-Negotiable Goals

1. **Correctness First**: All measurements must be accurate. Incorrect causal
   attribution wastes developer time on wrong optimizations.
2. **Minimal Overhead**: The profiler must not significantly perturb the
   program being measured. Pre-compute everything possible at initialization.
3. **Deterministic**: Given the same program and inputs, profiling results
   must be reproducible.

## Correctness Strategy

### Testing Hierarchy (in order of preference)

1. **Formal Verification (Kani)**: For critical algorithms—delay injection,
   experiment scheduling, throughput calculation. If it can be proven, prove it.
2. **Property-Based Testing (proptest)**: For all non-trivial logic. Unit tests
   are insufficient. Test invariants, not examples.
3. **Fuzz Testing**: For parsing, serialization, and any code handling external
   input.
4. **Unit Tests**: Only for simple, pure functions where property tests add no
   value.

### Critical Components Requiring Formal Verification

- **Delay injection logic**: Incorrect delays invalidate all causal claims
- **Throughput measurement**: Must accurately track progress point visits
- **Experiment scheduling**: Must correctly cycle through speedup percentages
- **Virtual speedup calculation**: Core causal profiling math

### Property Test Configuration

```bash
# Local development (fast feedback)
PROPTEST_CASES=64

# CI (thorough verification)
PROPTEST_CASES=512
PROPTEST_MAX_SHRINK_ITERS=10000
PROPTEST_MAX_SHRINK_TIME=60000
```

All property tests use environment variables—no inline case count overrides.

## Code Standards

### Error Handling

- **Andweorc MUST NOT panic** in production code paths
- Panics acceptable only for fundamental invariant violations (bugs)
- Always use `Result<T, E>` for fallible operations
- Propagate errors to where they can be meaningfully handled
- Design for graceful degradation

### Performance Discipline

- Pre-allocate when size is known
- Avoid allocations in hot paths (signal handlers, sample processing)
- Prefer algorithms with good worst-case behavior over average-case
- Prefer bounded operations over unbounded
- **Do NOT optimize without profiling data**—ironic for a profiler, but true

### Signal Handler Safety

Signal handlers have severe restrictions. Code in signal handlers must:
- Never allocate memory
- Never acquire locks (use lock-free data structures)
- Only call async-signal-safe functions
- Be as short as possible—set a flag and return

### Code Style

- All warnings treated as errors
- No `mod.rs` files—name modules directly (`foo.rs` not `foo/mod.rs`)
- All imports at top of file—NEVER inside functions
- Use named format arguments: `"{value}"` not `"{}"`
- Naive, obvious code preferred over clever abstractions
- **"Shape rule"**: 3+ repetitions = create abstraction; fewer = duplicate

### Documentation

- **Document WHY, not WHAT**—code shows what; comments explain why
- Every public item must have doc comments
- Include `# Errors` section for fallible functions
- Include `# Panics` section if function can panic
- Include `# Safety` section for unsafe functions

## Development Workflow

### Required Before Every Commit

```bash
ci/validate
```

This runs all checks. Do not run individual cargo commands—use CI scripts:

- `ci/validate` - Full validation (REQUIRED after all changes)
- `ci/fmt` - Format code
- `ci/check` - Type check
- `ci/clippy` - Lint
- `ci/test` - Run tests
- `ci/kani` - Formal verification (when applicable)

### Commit Hygiene

- Small, focused commits
- Descriptive commit messages explaining WHY
- Reference issues where applicable

## Architecture

### Crate Structure

```
andweorc/           Workspace root
├── andweorc/       Core profiling library (cdylib + lib)
├── andweorc-macros/ Proc-macro for #[profile] instrumentation
└── cargo-andweorc/ Cargo subcommand for running profiler
```

### Core Components

1. **Experiment** (`experiment.rs`): Central coordinator. Manages global state,
   signal handlers, delay table, and experiment progression.

2. **PerThreadProfiler** (`per_thread.rs`): Per-thread sampling via perf
   events. Collects instruction pointers and call chains.

3. **Progress Points** (`progress_point.rs`): Atomic counters marking
   throughput checkpoints. Auto-instrumented via proc-macro.

4. **Timer** (`timer.rs`): Per-thread interval timers for periodic sampling.

### Causal Profiling Algorithm (Coz Approach)

1. Continuously sample all threads to find where they're executing
2. Select a "line" (instruction address range) to virtually speed up
3. For each sample:
   - If the sampled thread IS at the selected line: no delay (fast code path)
   - If the sampled thread is NOT at the selected line: add delay to that thread
4. This effectively makes all code *except* the selected line slower, simulating
   that the selected line runs faster
5. Measure throughput via progress points during the experiment
6. Run experiments at multiple speedup percentages (5% to 150%)
7. Calculate impact: the slope of throughput vs speedup percentage indicates
   how much optimizing this line would improve overall performance

### Output Format

JSON output with causal impact per source line:

```json
{
  "experiments": [
    {
      "location": "src/main.rs:42",
      "speedup_percent": 10,
      "throughput_delta_percent": 5.2
    }
  ]
}
```

## Dependencies

### Allowed Licenses

MIT, Apache-2.0, BSD-2-Clause, BSD-3-Clause, ISC, Zlib, Unlicense, CC0-1.0

### Critical Dependencies

- `perf-event`: Hardware performance counter access (custom fork)
- `nix`: Safe POSIX bindings for signals, timers
- `libc`: Raw FFI when nix doesn't suffice

### Dependency Principles

- Minimize dependencies—each is an attack surface and maintenance burden
- `default-features = false` always; enable only what's needed
- Audit new dependencies for correctness and maintenance status
- No dependencies with known vulnerabilities (enforced by cargo-deny)

## Platform Support

Linux only. Causal profiling requires:
- `perf_event_open` syscall for hardware counters
- `timer_create` with `SIGEV_THREAD_ID` for per-thread timers
- Frame pointers for accurate stack unwinding

## Verification Checklist for New Code

Before submitting any code, verify:

- [ ] `ci/validate` passes
- [ ] No new warnings (warnings are errors)
- [ ] No `unwrap()` or `expect()` in non-test code (use `?` or handle errors)
- [ ] No allocations in signal handlers
- [ ] Property tests for non-trivial logic
- [ ] Kani proofs for critical algorithms (if applicable)
- [ ] Documentation for public items
- [ ] No increase in unsafe code without justification

## Glossary

- **Causal Profiling**: Technique that determines which code optimizations
  would improve overall performance, not just which code is hot.
- **Virtual Speedup**: Simulating faster execution of a code region by delaying
  all other threads proportionally.
- **Progress Point**: A location in code that represents one unit of useful
  work completed (e.g., one request processed).
- **Throughput**: Progress points per unit time—the metric being optimized.
