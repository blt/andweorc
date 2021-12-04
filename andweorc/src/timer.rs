use core::ptr::null;

use libc::{c_int, gettid, pid_t, sigevent, sigval, SIGEV_SIGNAL, SIGEV_THREAD_ID};

/// Notify a thread on a periodic interval
pub(crate) struct Interval {}

impl Interval {
    pub fn new(thread_id: pid_t, signal: c_int) -> Self {
        unsafe {
            let sigev: sigevent = Default::default();
            // sigev_notify: SIGEV_THREAD_ID,
            // sigev_value: sigval { sival_ptr: null() },
            // sigev_signo: signal,
            // sigev_notify_thread_id: thread_id,
            //            };
            unimplemented!()
        }
    }
}
