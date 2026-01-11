//! Per-thread interval timer for periodic signal delivery.
//!
//! This module provides interval timers that deliver signals to specific threads.
//! It uses POSIX `timer_create` with `SIGEV_THREAD_ID` to target signals at individual
//! threads rather than the process as a whole.

use libc::{
    c_int, itimerspec, pid_t, sigevent, sigval, timer_t, timespec, CLOCK_MONOTONIC, SIGEV_THREAD_ID,
};
use std::mem::MaybeUninit;
use std::ptr;
use std::time::Duration;

/// A periodic timer that delivers signals to a specific thread.
///
/// Uses POSIX `timer_create` with `SIGEV_THREAD_ID` for thread-targeted delivery.
/// The timer is automatically deleted when dropped.
pub(crate) struct Interval {
    /// The POSIX timer handle.
    timer_id: timer_t,
    /// Whether the timer has been created successfully.
    created: bool,
}

// SAFETY: The timer_t is an opaque handle that can be safely sent between threads.
// The timer itself is associated with a specific thread (via SIGEV_THREAD_ID),
// but the handle can be held by any thread.
unsafe impl Send for Interval {}

// SAFETY: The Interval methods that mutate state (&self on start/stop) are internally
// synchronized by the kernel. Multiple threads can call timer_settime on the same
// timer safely.
unsafe impl Sync for Interval {}

impl Interval {
    /// Creates a new interval timer targeting a specific thread.
    ///
    /// # Arguments
    ///
    /// * `thread_id` - The thread ID (from `gettid()`) to receive signals.
    /// * `signal` - The signal number to deliver (e.g., SIGPROF).
    ///
    /// # Errors
    ///
    /// Returns an error if the timer cannot be created.
    pub(crate) fn new(thread_id: pid_t, signal: c_int) -> Result<Self, IntervalError> {
        // Initialize sigevent structure for thread-targeted signal delivery
        // SAFETY: We're initializing a C struct that will be passed to timer_create
        let mut sev: sigevent = unsafe { MaybeUninit::zeroed().assume_init() };
        sev.sigev_notify = SIGEV_THREAD_ID;
        sev.sigev_signo = signal;
        sev.sigev_value = sigval {
            sival_ptr: ptr::null_mut(),
        };

        // Set the thread ID to receive the signal
        // SAFETY: sigev_notify_thread_id is a Linux-specific extension
        // On Linux, this is accessed via the _sigev_un union
        #[cfg(target_os = "linux")]
        {
            // The sigev_notify_thread_id field is in a union, accessed differently
            // depending on glibc version. We use the _tid field directly.
            sev.sigev_notify_thread_id = thread_id;
        }

        let mut timer_id: timer_t = ptr::null_mut();

        // SAFETY: timer_create is called with valid pointers
        let result =
            unsafe { libc::timer_create(CLOCK_MONOTONIC, &raw mut sev, &raw mut timer_id) };

        if result == -1 {
            return Err(IntervalError::CreateFailed(std::io::Error::last_os_error()));
        }

        Ok(Self {
            timer_id,
            created: true,
        })
    }

    /// Starts the timer with the given interval.
    ///
    /// The timer will fire periodically at the specified interval, delivering
    /// the configured signal to the target thread.
    ///
    /// # Errors
    ///
    /// Returns an error if the timer cannot be started.
    pub(crate) fn start(&self, interval: Duration) -> Result<(), IntervalError> {
        if !self.created {
            return Err(IntervalError::NotCreated);
        }

        let ts = duration_to_timespec(interval);
        let its = itimerspec {
            it_interval: ts, // Repeat interval
            it_value: ts,    // Initial expiration
        };

        // SAFETY: timer_settime is called with a valid timer_id
        let result =
            unsafe { libc::timer_settime(self.timer_id, 0, &raw const its, ptr::null_mut()) };

        if result == -1 {
            return Err(IntervalError::SetTimeFailed(std::io::Error::last_os_error()));
        }

        Ok(())
    }

    /// Stops the timer.
    ///
    /// The timer will no longer fire until started again.
    ///
    /// # Errors
    ///
    /// Returns an error if the timer cannot be stopped.
    pub(crate) fn stop(&self) -> Result<(), IntervalError> {
        if !self.created {
            return Err(IntervalError::NotCreated);
        }

        let its = itimerspec {
            it_interval: timespec {
                tv_sec: 0,
                tv_nsec: 0,
            },
            it_value: timespec {
                tv_sec: 0,
                tv_nsec: 0,
            },
        };

        // SAFETY: timer_settime is called with a valid timer_id
        let result =
            unsafe { libc::timer_settime(self.timer_id, 0, &raw const its, ptr::null_mut()) };

        if result == -1 {
            return Err(IntervalError::SetTimeFailed(std::io::Error::last_os_error()));
        }

        Ok(())
    }
}

impl Drop for Interval {
    fn drop(&mut self) {
        if self.created {
            // SAFETY: timer_delete is called with a valid timer_id
            unsafe {
                libc::timer_delete(self.timer_id);
            }
        }
    }
}

/// Converts a Duration to a libc timespec.
fn duration_to_timespec(duration: Duration) -> timespec {
    timespec {
        tv_sec: duration.as_secs().cast_signed(),
        tv_nsec: i64::from(duration.subsec_nanos()),
    }
}

/// Errors that can occur when working with interval timers.
#[derive(Debug)]
pub(crate) enum IntervalError {
    /// The timer was not created successfully.
    NotCreated,
    /// Failed to create the timer.
    CreateFailed(std::io::Error),
    /// Failed to set the timer interval.
    SetTimeFailed(std::io::Error),
}

impl std::fmt::Display for IntervalError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotCreated => write!(f, "timer was not created"),
            Self::CreateFailed(e) => write!(f, "failed to create timer: {e}"),
            Self::SetTimeFailed(e) => write!(f, "failed to set timer: {e}"),
        }
    }
}

impl std::error::Error for IntervalError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::NotCreated => None,
            Self::CreateFailed(e) | Self::SetTimeFailed(e) => Some(e),
        }
    }
}

/// Sleeps for the specified duration using POSIX nanosleep.
///
/// This is a simple wrapper around `libc::nanosleep` that handles interruptions.
///
/// # Errors
///
/// Returns an error if nanosleep fails for a reason other than being interrupted.
pub(crate) fn nanosleep(duration: Duration) -> Result<(), NanosleepError> {
    let mut request = duration_to_timespec(duration);
    let mut remaining = timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };

    loop {
        // SAFETY: nanosleep is called with valid pointers
        let result = unsafe { libc::nanosleep(&raw const request, &raw mut remaining) };

        if result == 0 {
            return Ok(());
        }

        let err = std::io::Error::last_os_error();
        if err.raw_os_error() == Some(libc::EINTR) {
            // Interrupted by signal, continue with remaining time
            request = remaining;
        } else {
            return Err(NanosleepError::Failed(err));
        }
    }
}

/// Errors that can occur during nanosleep.
#[derive(Debug)]
pub(crate) enum NanosleepError {
    /// nanosleep failed.
    Failed(std::io::Error),
}

impl std::fmt::Display for NanosleepError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Failed(e) => write!(f, "nanosleep failed: {e}"),
        }
    }
}

impl std::error::Error for NanosleepError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Failed(e) => Some(e),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nanosleep_short() {
        let start = std::time::Instant::now();
        let result = nanosleep(Duration::from_millis(10));
        let elapsed = start.elapsed();

        assert!(result.is_ok());
        // Should have slept for at least 10ms (allow some tolerance)
        assert!(elapsed >= Duration::from_millis(9));
        // But not too long (allow 50ms tolerance for scheduling)
        assert!(elapsed < Duration::from_millis(60));
    }

    #[test]
    fn test_duration_to_timespec() {
        let duration = Duration::new(5, 123_456_789);
        let ts = duration_to_timespec(duration);

        assert_eq!(ts.tv_sec, 5);
        assert_eq!(ts.tv_nsec, 123_456_789);
    }
}
