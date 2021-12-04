/// A progress point demarcates progress through a program, associated with a
/// name. The variant of the progress point determines the goal. I will document
/// this better.
use alloc::sync::Arc;
use core::sync::atomic::{AtomicU32, Ordering};
use libc_print::libc_println;

pub struct Progress {
    name: &'static str,
    visits: AtomicU32,
}

impl Progress {
    pub(crate) fn new(name: &'static str) -> Self {
        Self {
            name,
            visits: AtomicU32::new(0),
        }
    }

    pub fn get_instance(name: &'static str) -> Arc<Self> {
        crate::experiment::get_instance().progress(name)
    }

    pub fn note_visit(&self) {
        let visits = self.visits.fetch_add(1, Ordering::Relaxed);
        libc_println!("PROGRESS: {} -> {}", self.name, visits);
    }
}

#[macro_export]
macro_rules! progress {
    ($name:expr) => {
        andweorc::progress_point::Progress::get_instance($name).note_visit();
    };
}
