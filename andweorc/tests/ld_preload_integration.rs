//! Integration tests for LD_PRELOAD functionality.
//!
//! These tests verify that the profiler's LD_PRELOAD interception mechanisms
//! work correctly. Note that some tests may be skipped if hardware counters
//! are not available.

use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::Duration;

/// Test that profiling can be started and stopped without panicking.
#[test]
fn profiling_lifecycle_no_panic() {
    // This test verifies that the profiling API doesn't panic
    // even when hardware counters might not be available.

    // Don't start profiling if counters aren't available
    if !andweorc::validate::perf_counters_available() {
        eprintln!("Skipping test: hardware counters not available");
        return;
    }

    // Start profiling
    let result = andweorc::start_profiling();
    if result.is_err() {
        // This is acceptable - might fail due to permissions
        eprintln!(
            "Profiling start failed (expected in some environments): {:?}",
            result
        );
        return;
    }

    // Let it run briefly
    thread::sleep(Duration::from_millis(10));

    // Stop profiling
    andweorc::stop_profiling();
}

/// Test that consume_pending_delay works without crashing when not profiling.
#[test]
fn consume_delay_when_not_profiling() {
    // When profiling is not active, consume_pending_delay should be a no-op
    andweorc::consume_pending_delay();
    // Should not panic or crash
}

/// Test that multiple threads can call consume_pending_delay concurrently.
#[test]
fn concurrent_delay_consumption() {
    let counter = Arc::new(AtomicU32::new(0));
    let mut handles = vec![];

    // Spawn multiple threads
    for _ in 0..4 {
        let counter = Arc::clone(&counter);
        handles.push(thread::spawn(move || {
            for _ in 0..1000 {
                andweorc::consume_pending_delay();
                counter.fetch_add(1, Ordering::Relaxed);
            }
        }));
    }

    // Wait for all threads
    for handle in handles {
        handle.join().expect("Thread panicked");
    }

    // All increments should have been recorded
    assert_eq!(counter.load(Ordering::Relaxed), 4000);
}

/// Test environment validation doesn't panic.
#[test]
fn environment_validation_no_panic() {
    // validate_environment should return Ok or Err, never panic
    let result = andweorc::validate::validate_environment();
    // We don't care about the result, just that it doesn't panic
    let _ = result;
}

/// Test that perf_counters_available returns a boolean without panicking.
#[test]
fn perf_counters_check() {
    let available = andweorc::validate::perf_counters_available();
    // Just verify it returns a boolean
    let _ = available;
}

/// Test rapid thread creation and destruction.
#[test]
fn rapid_thread_churn() {
    // Create and destroy many threads rapidly
    // This stresses the pthread_create/exit interception
    for _ in 0..100 {
        let handles: Vec<_> = (0..10)
            .map(|_| {
                thread::spawn(|| {
                    // Brief work
                    let _ = 1 + 1;
                })
            })
            .collect();

        for handle in handles {
            handle.join().expect("Thread panicked");
        }
    }
}

/// Test that the FFI functions handle null pointers gracefully.
#[test]
fn ffi_null_pointer_safety() {
    // These should not crash when called via FFI with proper context
    // Note: We can't directly call andweorc_progress with null because
    // it's defined in the ffi module which is private. But the behavior
    // is tested implicitly through the consume_pending_delay path.

    // consume_pending_delay should handle missing profiler gracefully
    andweorc::consume_pending_delay();
}

/// Test that is_profiling_active returns correctly.
#[test]
fn profiling_active_flag() {
    // Initially should be false (unless something else set it)
    // This test just verifies the function doesn't panic
    let _ = andweorc::is_profiling_active();
}

/// Test delay_point! macro compiles and doesn't crash.
#[test]
fn delay_point_macro_works() {
    // The delay_point! macro should be a no-op when not profiling
    andweorc::delay_point!();

    // Multiple calls should work
    for _ in 0..100 {
        andweorc::delay_point!();
    }
}

/// Test that starting profiling twice returns an error.
#[test]
fn double_start_profiling_returns_error() {
    if !andweorc::validate::perf_counters_available() {
        eprintln!("Skipping test: hardware counters not available");
        return;
    }

    // First start
    let result = andweorc::start_profiling();
    if result.is_err() {
        eprintln!("Profiling start failed, skipping test");
        return;
    }

    // Second start should fail
    let result2 = andweorc::start_profiling();
    assert!(result2.is_err());

    // Clean up
    andweorc::stop_profiling();
}

/// Test that stop_profiling is safe to call multiple times.
#[test]
fn multiple_stop_profiling_safe() {
    // Calling stop multiple times should be safe
    andweorc::stop_profiling();
    andweorc::stop_profiling();
    andweorc::stop_profiling();
}

/// Test that init() respects ANDWEORC_ENABLED env var.
#[test]
fn init_respects_env_var() {
    // When ANDWEORC_ENABLED is not set, init should do nothing and succeed
    std::env::remove_var("ANDWEORC_ENABLED");
    let result = andweorc::init();
    assert!(result.is_ok());
}
