//! Experiment runner for causal profiling.
//!
//! This module implements the main experiment loop that:
//! 1. Collects samples to identify hot code locations
//! 2. Runs experiments with different virtual speedups
//! 3. Measures throughput changes
//! 4. Records results for analysis

use crate::experiment::get_instance;
use crate::progress_point::Progress;
use crate::timer::nanosleep;
use dashmap::DashMap;
use libc::c_void;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

/// Results from a single experiment round.
#[derive(Debug, Clone, Copy)]
pub struct ExperimentResult {
    /// The instruction pointer that was virtually sped up.
    pub ip: usize,
    /// The speedup percentage applied (0.0 to 1.5).
    pub speedup_pct: f64,
    /// Throughput observed during this experiment (operations/sec).
    pub throughput: f64,
    /// Duration of the experiment in milliseconds.
    pub duration_ms: u64,
    /// Number of samples that matched the selected IP.
    pub matching_samples: u64,
}

/// Accumulated results from all experiments.
#[derive(Debug, Default)]
pub struct ProfilingResults {
    /// All experiment results, keyed by IP.
    results: DashMap<usize, Vec<ExperimentResult>>,
}

impl ProfilingResults {
    /// Creates a new results container.
    #[must_use]
    pub fn new() -> Self {
        Self {
            results: DashMap::new(),
        }
    }

    /// Adds an experiment result.
    pub fn add_result(&self, result: ExperimentResult) {
        self.results.entry(result.ip).or_default().push(result);
    }

    /// Returns all results for a given IP.
    #[must_use]
    pub fn results_for_ip(&self, ip: usize) -> Option<Vec<ExperimentResult>> {
        self.results.get(&ip).map(|r| r.clone())
    }

    /// Returns all unique IPs that have been profiled.
    #[must_use]
    pub fn profiled_ips(&self) -> Vec<usize> {
        self.results.iter().map(|r| *r.key()).collect()
    }

    /// Calculates the causal impact for each IP.
    ///
    /// Returns a list of (IP, impact) pairs sorted by impact descending.
    /// Impact is calculated as the slope of throughput vs speedup percentage.
    #[must_use]
    pub fn calculate_impacts(&self) -> Vec<(usize, f64)> {
        let mut impacts: Vec<(usize, f64)> = self
            .results
            .iter()
            .filter_map(|entry| {
                let results = entry.value();
                if results.len() < 2 {
                    return None;
                }

                // Calculate linear regression of throughput vs speedup
                let impact = calculate_linear_regression(results);
                Some((*entry.key(), impact))
            })
            .collect();

        // Sort by impact descending (higher impact = more important)
        impacts.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        impacts
    }
}

/// Calculates the linear regression slope of throughput vs speedup.
fn calculate_linear_regression(results: &[ExperimentResult]) -> f64 {
    // Allow precision loss - this is statistical calculation
    #[allow(clippy::cast_precision_loss)]
    let n = results.len() as f64;
    if n < 2.0 {
        return 0.0;
    }

    // Calculate means
    let sum_x: f64 = results.iter().map(|r| r.speedup_pct).sum();
    let sum_y: f64 = results.iter().map(|r| r.throughput).sum();
    let mean_x = sum_x / n;
    let mean_y = sum_y / n;

    // Calculate slope
    let mut numerator = 0.0;
    let mut denominator = 0.0;
    for r in results {
        let dx = r.speedup_pct - mean_x;
        let dy = r.throughput - mean_y;
        numerator += dx * dy;
        denominator += dx * dx;
    }

    if denominator.abs() < f64::EPSILON {
        return 0.0;
    }

    numerator / denominator
}

/// Configuration for the experiment runner.
#[derive(Debug, Clone)]
pub struct RunnerConfig {
    /// Duration of each experiment round.
    pub round_duration: Duration,
    /// Number of baseline rounds (0% speedup) to run.
    pub baseline_rounds: usize,
    /// Speedup percentages to test (0.0 to 1.5).
    pub speedup_percentages: Vec<f64>,
    /// Maximum number of IPs to test.
    pub max_ips: usize,
}

impl Default for RunnerConfig {
    fn default() -> Self {
        Self {
            round_duration: Duration::from_secs(1),
            baseline_rounds: 5,
            speedup_percentages: vec![0.0, 0.25, 0.5, 0.75, 1.0],
            max_ips: 10,
        }
    }
}

/// The experiment runner that orchestrates causal profiling.
pub struct Runner {
    /// Configuration for experiments.
    config: RunnerConfig,
    /// Results from all experiments.
    results: ProfilingResults,
    /// Whether the runner is currently active.
    active: AtomicBool,
}

impl std::fmt::Debug for Runner {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Runner")
            .field("config", &self.config)
            .field("active", &self.active.load(Ordering::Relaxed))
            .finish_non_exhaustive()
    }
}

impl Runner {
    /// Creates a new experiment runner with default configuration.
    #[must_use]
    pub fn new() -> Self {
        Self::with_config(RunnerConfig::default())
    }

    /// Creates a new experiment runner with the given configuration.
    #[must_use]
    pub fn with_config(config: RunnerConfig) -> Self {
        Self {
            config,
            results: ProfilingResults::new(),
            active: AtomicBool::new(false),
        }
    }

    /// Runs a single experiment round with the given IP and speedup index.
    fn run_experiment_round(
        &self,
        ip: Option<usize>,
        speedup_index: usize,
        progress_point: &Arc<Progress>,
    ) -> ExperimentResult {
        let experiment = get_instance();

        // Reset progress point for this round
        progress_point.reset();

        // Start the experiment
        if let Some(ip_val) = ip {
            experiment.start_experiment(ip_val as *const c_void, speedup_index);
        }

        // Wait for the round duration
        let _ = nanosleep(self.config.round_duration);

        // Stop the experiment
        experiment.stop_experiment();

        // Get results
        let stats = experiment.stats();
        let throughput = progress_point.throughput();
        let elapsed_ns = progress_point.elapsed_nanos();

        // Determine speedup percentage from index
        // Allow precision loss - index is small (max ~55)
        #[allow(clippy::cast_precision_loss)]
        let speedup_pct = match speedup_index {
            0..=24 => 0.0, // First 25 entries are baseline
            i => (i - 25) as f64 * 0.05 + 0.05,
        };

        ExperimentResult {
            ip: ip.unwrap_or(0),
            speedup_pct,
            throughput,
            duration_ms: elapsed_ns / 1_000_000,
            matching_samples: stats.selected_samples,
        }
    }

    /// Runs the full experiment suite.
    ///
    /// This will:
    /// 1. Run baseline measurements
    /// 2. Test each hot IP with various speedup percentages
    /// 3. Record results
    ///
    /// Returns the profiling results.
    pub fn run(&self, progress_point_name: &'static str) -> &ProfilingResults {
        self.active.store(true, Ordering::Release);

        let progress_point = Progress::get_instance(progress_point_name);

        // Phase 1: Run baseline measurements
        for _ in 0..self.config.baseline_rounds {
            let result = self.run_experiment_round(None, 0, &progress_point);
            self.results.add_result(result);
        }

        // Phase 2: Get top IPs to test from the experiment
        let top_ips = get_instance().top_ips(self.config.max_ips);

        // Phase 3: Run experiments for each IP and speedup percentage
        for ip in top_ips {
            for (idx, _speedup) in self.config.speedup_percentages.iter().enumerate() {
                // Map speedup percentage to delay table index
                // 0.0 -> index 0, 0.05 -> index 25, 0.10 -> index 26, etc.
                let speedup_index = if idx == 0 {
                    0 // Baseline
                } else {
                    24 + idx // Speedup entries start at index 25
                };

                let result = self.run_experiment_round(Some(ip), speedup_index, &progress_point);
                self.results.add_result(result);
            }
        }

        self.active.store(false, Ordering::Release);
        &self.results
    }

    /// Returns whether the runner is currently active.
    #[must_use]
    pub fn is_active(&self) -> bool {
        self.active.load(Ordering::Acquire)
    }

    /// Returns the profiling results.
    #[must_use]
    pub fn results(&self) -> &ProfilingResults {
        &self.results
    }
}

impl Default for Runner {
    fn default() -> Self {
        Self::new()
    }
}
