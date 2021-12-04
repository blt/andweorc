use crate::{per_thread::PerThreadProfiler, progress_point::Progress};
use alloc::sync::Arc;
use alloc::vec::Vec;
use core::sync::atomic::{AtomicU32, AtomicUsize, Ordering};
use core::time::Duration;
use dashmap::DashMap;
use libc::{c_int, c_void};
use libc_print::{libc_eprint, libc_println};
use nix::sys::pthread::{pthread_self, Pthread};
use nix::sys::signal::{self, SigHandler, Signal};
use rand::{rngs::SmallRng, SeedableRng};
use spin::Once;

// The percentage of delays to be used in experiments. Ba
const DELAY_BASELINE: Duration = Duration::from_millis(10);
const DELAY_PRCNT: [f64; 55] = [
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55,
    0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15, 1.20, 1.25, 1.30, 1.35,
    1.40, 1.45, 1.50,
];

/// The function that is periodically called to process perf samples
///
/// The timer signal will be caught by any running thread of the library. This
/// means that we must look up which thread we are, map that to an appropriate
/// profiler and then....
extern "C" fn process_samples(signal: c_int) {
    // TODO check that this is the correct signal
    let tid = pthread_self();
    let ts = crate::experiment::get_instance().thread_state(tid);
    ts.process_samples();
}

static EXPERIMENT: Once<Experiment> = Once::new();

pub(crate) fn get_instance() -> &'static Experiment {
    EXPERIMENT.call_once(|| Experiment::new(DELAY_BASELINE))
}

pub(crate) struct Experiment {
    timer_signal_handle: SigHandler,

    thread_states: DashMap<Pthread, Arc<PerThreadProfiler>>,
    progress_points: DashMap<&'static str, Arc<Progress>>,

    delay_table: Vec<Duration>,
    rng: SmallRng,

    /// The total number of delays that this experiment has managed.
    global_delay_count: AtomicU32,
    /// The index to use for the delay_table
    delay_index: AtomicUsize,
}

pub(crate) struct DelayDetails {
    pub global_delay: u32,
    pub current_delay: Duration,
}

impl Experiment {
    pub fn new(delay_baseline: Duration) -> Self {
        // Setup the timer signal
        let handler = SigHandler::Handler(process_samples);
        // SAFETY The handler lasts as long as this struct does. We'll need to
        // implement a drop to disable the signal? TODO
        unsafe { signal::signal(Signal::SIGPROF, handler) }.unwrap();

        Self {
            timer_signal_handle: handler,

            thread_states: DashMap::new(),
            progress_points: DashMap::new(),

            global_delay_count: AtomicU32::new(0),
            delay_index: AtomicUsize::new(0),
            rng: SmallRng::seed_from_u64(123456789), // TODO make seeding this better
            delay_table: DELAY_PRCNT
                .iter()
                .map(|prct| delay_baseline.mul_f64(*prct))
                .collect(),
        }
    }

    pub fn delay_details(&self) -> DelayDetails {
        let global_delay = self.global_delay_count.load(Ordering::Relaxed);
        let current_delay = self.delay_table[self.delay_index.load(Ordering::Relaxed)];
        DelayDetails {
            global_delay,
            current_delay,
        }
    }

    pub fn register_thread(&self, thread_id: Pthread) -> Arc<PerThreadProfiler> {
        let refmut = self
            .thread_states
            .entry(thread_id)
            .or_insert_with(|| Arc::new(PerThreadProfiler::new(100)));
        Arc::clone(refmut.value())
    }

    pub fn thread_state(&self, thread_id: Pthread) -> Arc<PerThreadProfiler> {
        let refmut = self
            .thread_states
            .entry(thread_id)
            .or_insert_with(|| Arc::new(PerThreadProfiler::new(100)));
        Arc::clone(refmut.value())
    }

    pub fn progress(&self, name: &'static str) -> Arc<Progress> {
        let refmut = self
            .progress_points
            .entry(name)
            .or_insert_with(|| Arc::new(Progress::new(name)));
        Arc::clone(refmut.value())
    }

    /// Submit a callstack to be included in the experiment
    pub fn report_sample(&self, ip: Option<*const c_void>, callchain: Option<Vec<*const c_void>>) {
        libc_println!("IP: {:?} | CALLCHAIN: {:?}", ip, callchain);
    }
}
