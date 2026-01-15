//! POSIX function interception for LD_PRELOAD-based profiling.
//!
//! This module provides automatic profiling of any Linux program via `LD_PRELOAD`.
//! When loaded, it intercepts pthread synchronization functions and injects delays
//! to implement the Coz causal profiling approach.
//!
//! # Usage
//!
//! ```bash
//! LD_PRELOAD=/path/to/libandweorc.so ANDWEORC_ENABLED=1 ./target_program
//! ```
//!
//! # Architecture
//!
//! The interception works by:
//! 1. Library constructor runs before `main()`, initializing the profiler
//! 2. Synchronization functions (`pthread_mutex_lock`, etc.) are intercepted
//! 3. Delays are injected at sync points to simulate virtual speedups
//! 4. Library destructor outputs results when the program exits
//!
//! # Signal Safety
//!
//! The interceptors are designed to be as safe as possible:
//! - No allocation in hot paths
//! - Lock-free dlsym resolution
//! - Fast path for disabled profiling (single atomic load)

mod pthread;
mod real;
mod signal;
pub(crate) mod trigger;

/// Checks if the `ANDWEORC_ENABLED` environment variable is set to "1".
///
/// Uses raw libc getenv to avoid Rust std initialization issues in constructors.
///
/// # Safety
///
/// This function is safe to call from library constructors before Rust std
/// is fully initialized.
fn env_enabled() -> bool {
    // SAFETY: getenv is async-signal-safe and works before std initialization
    unsafe {
        let key = b"ANDWEORC_ENABLED\0";
        let val = libc::getenv(key.as_ptr().cast::<i8>());
        if val.is_null() {
            return false;
        }
        // Check if value is "1"
        *val == i8::from_ne_bytes([b'1'])
    }
}

/// Library constructor - runs before `main()`.
///
/// This function is called by the dynamic linker when the library is loaded
/// via `LD_PRELOAD`. It initializes the profiler before the target program starts.
///
/// # Behavior
///
/// - If `ANDWEORC_ENABLED=1` is not set, does nothing (profiler disabled)
/// - Validates the environment (perf counters, permissions)
/// - Initializes the experiment singleton and signal handler
/// - On failure, prints an error and calls `_exit(1)`
#[used]
#[link_section = ".init_array"]
static INIT: extern "C" fn() = {
    extern "C" fn init() {
        // Only initialize if explicitly enabled
        if !env_enabled() {
            return;
        }

        // Validate environment before initializing
        if let Err(e) = crate::validate::validate_environment() {
            libc_print::libc_eprintln!("{e}");
            // SAFETY: _exit is async-signal-safe and does not run destructors
            unsafe { libc::_exit(1) };
        }

        // Initialize the experiment singleton
        match crate::experiment::try_init() {
            Ok(_) => {
                crate::set_profiling_active(true);
                libc_print::libc_println!("[andweorc] profiler initialized via LD_PRELOAD");
            }
            Err(e) => {
                libc_print::libc_eprintln!("[andweorc] FATAL: initialization failed: {e}");
                // SAFETY: _exit is async-signal-safe
                unsafe { libc::_exit(1) };
            }
        }

        // Register SIGUSR1 handler for signal-based experiment triggering
        // This allows external programs to trigger experiments via `kill -USR1 <pid>`
        if let Err(e) = trigger::register_sigusr1_handler() {
            libc_print::libc_eprintln!("[andweorc] WARNING: failed to register SIGUSR1 handler: {e}");
            // Continue anyway - the profiler can still work via explicit API calls
        }
    }
    init
};

/// Library destructor - runs when the program exits.
///
/// This function outputs profiling results before the process terminates.
/// It may not run if the program calls `_exit()` directly.
#[used]
#[link_section = ".fini_array"]
static FINI: extern "C" fn() = {
    extern "C" fn fini() {
        if crate::is_profiling_active() {
            // Shutdown the profiler and output results
            crate::experiment::shutdown();
            libc_print::libc_println!("[andweorc] profiler shutdown complete");
        }
    }
    fini
};
