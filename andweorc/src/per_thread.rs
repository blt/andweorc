use alloc::vec::Vec;
use core::time::Duration;
use nix::sys::pthread::pthread_self;
use perf_event::events::Hardware;
use perf_event::sample::{PerfRecord, PerfSampleType};
use perf_event::{Builder, SampleStream};

// thread_local!(pub(crate) static PER_THREAD: PerThreadProfiler = PerThreadProfiler::new(100));

fn nanosleep(duration: Duration) {
    unimplemented!();
}

/// A profiler for each thread
///
/// This struct manages the perf loop that sits at the "top" of every operating
/// system thread.
pub(crate) struct PerThreadProfiler {
    /// The total number of delays that this thread has observed
    delay_count: u32,

    /// Maximum number of samples that will be polled for with each pulse
    total_samples: u16,
    sampler: SampleStream,
}

impl PerThreadProfiler {
    pub fn new(total_samples: u16) -> Self {
        let sampler = Builder::new()
            .kind(Hardware::CPU_CYCLES)
            .sample_frequency(1000)
            .sample(PerfSampleType::ADDR)
            .sample(PerfSampleType::CALLCHAIN)
            .sample(PerfSampleType::CPU)
            .sample(PerfSampleType::IP)
            .sample(PerfSampleType::PERIOD)
            .sample(PerfSampleType::STREAM_ID)
            .sample(PerfSampleType::TID)
            .sample(PerfSampleType::TIME)
            .sample_stream()
            .unwrap();

        Self {
            sampler,
            total_samples,
            delay_count: 0,
        }
    }

    pub fn delay(&self) {
        let details = crate::experiment::get_instance().delay_details();
        let delay_diff = details.global_delay - self.delay_count;

        let duration = details.current_delay.mul_f64(delay_diff as f64);
        nanosleep(duration);
    }

    /// Process samples collected since the last time the sampler was polled
    ///
    /// This function collects any of the samples this thread's sampler has
    /// collected since the last polling period. Caller here is a signal handler
    /// on a timer, interrupting
    pub fn process_samples(&self) {
        let timeout = Duration::from_millis(1);
        for _ in 0..self.total_samples {
            if let Some(PerfRecord::Sample(sample)) = self.sampler.read(Some(timeout)).unwrap() {
                let ip = sample.ip;
                let callchain = sample.callchain;
                crate::experiment::get_instance().report_sample(ip, callchain);
                // sample.ip
                // //                println!("{:?}", sample);
                // let callchain: Vec<u64> = sample
                //     .callchain
                //     .unwrap()
                //     .into_iter()
                //     .map(|ptr| ptr as u64)
                //     .collect();
                // let mut buf = Vec::with_capacity(callchain.len());
                // sr.enrich(&callchain, &mut buf).unwrap();
                //                println!("{:?}", callchain);
            } else {
                break;
            }
        }
    }
}
