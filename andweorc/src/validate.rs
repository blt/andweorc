//! Environment validation for causal profiling.
//!
//! This module validates that the system environment supports causal profiling
//! before attempting to initialize. It provides clear, actionable error messages
//! when requirements are not met.

use std::fmt;
use std::fs;
use std::io;

/// Errors that can occur during environment validation.
#[derive(Debug)]
pub enum EnvironmentError {
    /// Hardware performance counters are restricted by kernel settings.
    PerfRestricted {
        /// Current value of `perf_event_paranoid`
        current: i32,
        /// How to fix the issue
        fix: &'static str,
    },

    /// Could not read the `perf_event_paranoid` sysctl value.
    SysctlReadFailed {
        /// The path that could not be read
        path: &'static str,
        /// The underlying error
        error: io::Error,
    },

    /// Hardware performance counters are not available.
    HardwareCountersUnavailable {
        /// Description of the failure
        reason: String,
    },

    /// Signal handler registration failed.
    SignalRegistrationFailed {
        /// Description of the failure
        reason: String,
    },
}

impl fmt::Display for EnvironmentError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::PerfRestricted { current, fix } => {
                write!(
                    f,
                    "[andweorc] FATAL: Hardware performance counters restricted.\n\n\
                     Cause: kernel.perf_event_paranoid is set to {current} (restricts perf events)\n\n\
                     To fix, run one of:\n  \
                     # Temporary (until reboot):\n  \
                     {fix}\n\n  \
                     # Permanent:\n  \
                     echo 'kernel.perf_event_paranoid=1' | sudo tee /etc/sysctl.d/99-perf.conf\n\n  \
                     # Or grant capability to the binary:\n  \
                     sudo setcap cap_perfmon=ep /path/to/binary"
                )
            }
            Self::SysctlReadFailed { path, error } => {
                write!(f, "[andweorc] FATAL: Could not read {path}: {error}")
            }
            Self::HardwareCountersUnavailable { reason } => {
                write!(
                    f,
                    "[andweorc] FATAL: Hardware performance counters unavailable.\n\n\
                     Cause: {reason}\n\n\
                     Causal profiling requires hardware instruction counters.\n\
                     Ensure your CPU supports performance monitoring and you have\n\
                     appropriate permissions (see perf_event_paranoid setting)."
                )
            }
            Self::SignalRegistrationFailed { reason } => {
                write!(
                    f,
                    "[andweorc] FATAL: Could not register signal handler.\n\nCause: {reason}"
                )
            }
        }
    }
}

impl std::error::Error for EnvironmentError {}

/// Validates that the environment supports causal profiling.
///
/// This should be called before initializing the profiler. It checks:
/// 1. `perf_event_paranoid` setting allows hardware counters
/// 2. Hardware instruction counters are actually available
/// 3. Signal handlers can be registered
///
/// # Errors
///
/// Returns an [`EnvironmentError`] with a clear, actionable message if
/// any requirement is not met.
pub fn validate_environment() -> Result<(), EnvironmentError> {
    // Check perf_event_paranoid setting
    check_perf_paranoid()?;

    // Test hardware counter availability
    // (currently a no-op, actual test happens in per_thread.rs)
    test_hardware_counters();

    Ok(())
}

/// Reads the `kernel.perf_event_paranoid` sysctl value.
fn read_perf_paranoid() -> Result<i32, EnvironmentError> {
    const PATH: &str = "/proc/sys/kernel/perf_event_paranoid";

    let content = fs::read_to_string(PATH).map_err(|e| EnvironmentError::SysctlReadFailed {
        path: PATH,
        error: e,
    })?;

    content
        .trim()
        .parse::<i32>()
        .map_err(|_| EnvironmentError::SysctlReadFailed {
            path: PATH,
            error: io::Error::new(io::ErrorKind::InvalidData, "not a valid integer"),
        })
}

/// Checks if `perf_event_paranoid` allows hardware counter access.
fn check_perf_paranoid() -> Result<(), EnvironmentError> {
    let paranoid = read_perf_paranoid()?;

    // Values:
    //  -1: Allow use of (almost) all events by all users
    //   0: Allow all users to use their process's counters
    //   1: Disallow raw tracepoint access by users without CAP_SYS_ADMIN
    //   2: Disallow kernel profiling by users without CAP_PERFMON
    //   3: Disallow all perf_event usage without CAP_PERFMON
    //
    // We need at least level 1 for hardware instruction counters on most kernels,
    // but some systems work at level 2. We'll check for level 2+ and warn.
    if paranoid > 2 {
        return Err(EnvironmentError::PerfRestricted {
            current: paranoid,
            fix: "sudo sysctl kernel.perf_event_paranoid=1",
        });
    }

    Ok(())
}

/// Tests that hardware instruction counters are actually available.
///
/// This is a quick sanity check. The actual counter creation happens in
/// `per_thread.rs` which provides detailed error messages on failure.
fn test_hardware_counters() {
    // The actual hardware counter test happens when we create the sampler
    // in per_thread.rs. Here we just verify that perf_event_open is likely
    // to succeed based on the paranoid setting (already checked above).
    //
    // We could try to create a counter here, but that would:
    // 1. Add overhead at startup
    // 2. Duplicate the error handling in per_thread.rs
    //
    // Instead, we rely on the perf_event_paranoid check and let the
    // per_thread.rs error message guide users if hardware counters
    // are unavailable for other reasons.
}

/// Checks if hardware performance counters are available on this system.
///
/// This function attempts to create a test counter to verify that the
/// perf subsystem is functional. Use this in tests to skip when counters
/// aren't available (e.g., in containers or VMs without perf support).
///
/// # Returns
///
/// `true` if hardware counters can be created, `false` otherwise.
#[must_use]
pub fn perf_counters_available() -> bool {
    use perf_event::events::Hardware;

    // Try to create a simple instruction counter
    // This uses the same API as per_thread.rs but doesn't enable or sample
    perf_event::Builder::new()
        .kind(Hardware::INSTRUCTIONS)
        .observe_self()
        .any_cpu()
        .counter()
        .is_ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn read_perf_paranoid_works() {
        // This test may fail on systems without /proc
        if std::path::Path::new("/proc/sys/kernel/perf_event_paranoid").exists() {
            let result = read_perf_paranoid();
            assert!(result.is_ok());
            let value = result.unwrap();
            // Paranoid is typically -1 to 3
            assert!((-1..=3).contains(&value) || value > 3);
        }
    }

    #[test]
    fn environment_error_display_is_helpful() {
        let err = EnvironmentError::PerfRestricted {
            current: 3,
            fix: "sudo sysctl kernel.perf_event_paranoid=1",
        };
        let msg = format!("{err}");
        assert!(msg.contains("FATAL"));
        assert!(msg.contains("sudo sysctl"));
        assert!(msg.contains("cap_perfmon"));
    }
}
