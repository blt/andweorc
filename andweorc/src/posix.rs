//! POSIX function interception for profiler integration.
//!
//! This module contains (currently disabled) code for intercepting POSIX
//! functions via `LD_PRELOAD`. This approach is not needed for the Rust-only
//! cargo subcommand workflow, but the code is preserved for potential
//! future use with C/C++ programs.
//!
//! The interception uses `dlsym(RTLD_NEXT, ...)` to forward calls to the
//! real libc functions after performing profiler bookkeeping.

mod pthread;
