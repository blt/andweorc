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
use libc::c_void;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, RwLock};
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
///
/// This structure is NOT accessed from signal handlers, so standard
/// synchronization (`RwLock`) is safe to use.
#[derive(Debug)]
pub struct ProfilingResults {
    /// All experiment results, keyed by IP.
    results: RwLock<HashMap<usize, Vec<ExperimentResult>>>,
}

impl Default for ProfilingResults {
    fn default() -> Self {
        Self::new()
    }
}

impl ProfilingResults {
    /// Creates a new results container.
    #[must_use]
    pub fn new() -> Self {
        Self {
            results: RwLock::new(HashMap::new()),
        }
    }

    /// Adds an experiment result.
    ///
    /// # Panics
    ///
    /// Panics if the results lock is poisoned (indicates a bug).
    #[allow(clippy::expect_used)]
    pub fn add_result(&self, result: ExperimentResult) {
        let mut guard = self.results.write().expect("results lock poisoned");
        guard.entry(result.ip).or_default().push(result);
    }

    /// Returns all results for a given IP.
    ///
    /// # Panics
    ///
    /// Panics if the results lock is poisoned (indicates a bug).
    #[must_use]
    #[allow(clippy::expect_used)]
    pub fn results_for_ip(&self, ip: usize) -> Option<Vec<ExperimentResult>> {
        let guard = self.results.read().expect("results lock poisoned");
        guard.get(&ip).cloned()
    }

    /// Returns all unique IPs that have been profiled.
    ///
    /// # Panics
    ///
    /// Panics if the results lock is poisoned (indicates a bug).
    #[must_use]
    #[allow(clippy::expect_used)]
    pub fn profiled_ips(&self) -> Vec<usize> {
        let guard = self.results.read().expect("results lock poisoned");
        guard.keys().copied().collect()
    }

    /// Calculates the causal impact for each IP.
    ///
    /// Returns a list of (IP, impact) pairs sorted by impact descending.
    /// Impact is calculated as the slope of throughput vs speedup percentage.
    ///
    /// # Panics
    ///
    /// Panics if the results lock is poisoned (indicates a bug).
    #[must_use]
    #[allow(clippy::expect_used)]
    pub fn calculate_impacts(&self) -> Vec<(usize, f64)> {
        let guard = self.results.read().expect("results lock poisoned");
        let mut impacts: Vec<(usize, f64)> = guard
            .iter()
            .filter_map(|(&ip, results)| {
                if results.len() < 2 {
                    return None;
                }

                // Calculate linear regression of throughput vs speedup
                let impact = calculate_linear_regression(results);
                Some((ip, impact))
            })
            .collect();

        // Sort by impact descending (higher impact = more important)
        // NaN values (from degenerate data) compare as Equal, placing them last in the sorted list
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
            // IP of 0 indicates a baseline measurement (no specific code location targeted)
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
    ///
    /// # Panics
    ///
    /// Panics if the experiment singleton failed to initialize (e.g., SIGPROF
    /// signal handler could not be registered).
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper to create an ExperimentResult for testing
    fn make_result(speedup_pct: f64, throughput: f64) -> ExperimentResult {
        ExperimentResult {
            ip: 0x1000,
            speedup_pct,
            throughput,
            duration_ms: 1000,
            matching_samples: 10,
        }
    }

    #[test]
    fn linear_regression_empty_returns_zero() {
        let results: Vec<ExperimentResult> = vec![];
        assert!((calculate_linear_regression(&results) - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn linear_regression_single_point_returns_zero() {
        let results = vec![make_result(0.0, 100.0)];
        assert!((calculate_linear_regression(&results) - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn linear_regression_constant_y_returns_zero() {
        // All same throughput regardless of speedup = zero slope
        let results = vec![
            make_result(0.0, 100.0),
            make_result(0.5, 100.0),
            make_result(1.0, 100.0),
        ];
        let slope = calculate_linear_regression(&results);
        assert!(slope.abs() < f64::EPSILON, "slope was {slope}");
    }

    #[test]
    fn linear_regression_positive_slope() {
        // Throughput increases with speedup
        let results = vec![
            make_result(0.0, 100.0),
            make_result(0.5, 125.0),
            make_result(1.0, 150.0),
        ];
        let slope = calculate_linear_regression(&results);
        // Expected slope: (150-100)/(1.0-0.0) = 50
        assert!((slope - 50.0).abs() < 0.001, "slope was {slope}");
    }

    #[test]
    fn linear_regression_negative_slope() {
        // Throughput decreases with speedup (shouldn't happen in practice, but test it)
        let results = vec![
            make_result(0.0, 150.0),
            make_result(0.5, 125.0),
            make_result(1.0, 100.0),
        ];
        let slope = calculate_linear_regression(&results);
        assert!((slope - (-50.0)).abs() < 0.001, "slope was {slope}");
    }

    #[test]
    fn linear_regression_same_x_returns_zero() {
        // All same speedup = can't compute slope
        let results = vec![
            make_result(0.5, 100.0),
            make_result(0.5, 150.0),
            make_result(0.5, 200.0),
        ];
        let slope = calculate_linear_regression(&results);
        assert!((slope - 0.0).abs() < f64::EPSILON, "slope was {slope}");
    }

    #[test]
    fn profiling_results_empty_initially() {
        let results = ProfilingResults::new();
        assert!(results.profiled_ips().is_empty());
    }

    #[test]
    fn profiling_results_add_and_retrieve() {
        let results = ProfilingResults::new();
        let exp = make_result(0.5, 100.0);
        results.add_result(exp);

        let ips = results.profiled_ips();
        assert_eq!(ips.len(), 1);
        assert!(ips.contains(&0x1000));

        let ip_results = results.results_for_ip(0x1000).unwrap();
        assert_eq!(ip_results.len(), 1);
        assert!((ip_results[0].throughput - 100.0).abs() < f64::EPSILON);
    }

    #[test]
    fn calculate_impacts_needs_at_least_two_results() {
        let results = ProfilingResults::new();
        results.add_result(make_result(0.0, 100.0));
        // Only one result for IP 0x1000
        let impacts = results.calculate_impacts();
        assert!(impacts.is_empty());
    }

    #[test]
    fn calculate_impacts_sorts_descending() {
        let results = ProfilingResults::new();

        // IP 0x1000: positive slope (good optimization target)
        results.add_result(ExperimentResult {
            ip: 0x1000,
            speedup_pct: 0.0,
            throughput: 100.0,
            duration_ms: 1000,
            matching_samples: 10,
        });
        results.add_result(ExperimentResult {
            ip: 0x1000,
            speedup_pct: 1.0,
            throughput: 150.0,
            duration_ms: 1000,
            matching_samples: 10,
        });

        // IP 0x2000: smaller positive slope
        results.add_result(ExperimentResult {
            ip: 0x2000,
            speedup_pct: 0.0,
            throughput: 100.0,
            duration_ms: 1000,
            matching_samples: 10,
        });
        results.add_result(ExperimentResult {
            ip: 0x2000,
            speedup_pct: 1.0,
            throughput: 110.0,
            duration_ms: 1000,
            matching_samples: 10,
        });

        let impacts = results.calculate_impacts();
        assert_eq!(impacts.len(), 2);
        // First should be 0x1000 with higher impact
        assert_eq!(impacts[0].0, 0x1000);
        assert_eq!(impacts[1].0, 0x2000);
        assert!(impacts[0].1 > impacts[1].1);
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        /// Property: Linear regression always returns finite values
        #[test]
        fn linear_regression_is_finite(
            n in 2..50_usize,
            base_throughput in 1.0..10000.0_f64,
            slope in -1000.0..1000.0_f64
        ) {
            let results: Vec<ExperimentResult> = (0..n)
                .map(|i| {
                    #[allow(clippy::cast_precision_loss)]
                    let speedup = i as f64 * 0.05;
                    let throughput = base_throughput + speedup * slope;
                    ExperimentResult {
                        ip: 0x1000,
                        speedup_pct: speedup,
                        throughput,
                        duration_ms: 1000,
                        matching_samples: 10,
                    }
                })
                .collect();

            let computed_slope = calculate_linear_regression(&results);
            prop_assert!(computed_slope.is_finite(), "slope was {computed_slope}");
        }

        /// Property: Perfect linear data recovers the original slope
        #[test]
        fn linear_regression_recovers_slope(
            n in 3..20_usize,
            base_throughput in 100.0..1000.0_f64,
            slope in 10.0..100.0_f64  // positive slopes only for easier testing
        ) {
            let results: Vec<ExperimentResult> = (0..n)
                .map(|i| {
                    #[allow(clippy::cast_precision_loss)]
                    let speedup = i as f64 * 0.1;
                    let throughput = base_throughput + speedup * slope;
                    ExperimentResult {
                        ip: 0x1000,
                        speedup_pct: speedup,
                        throughput,
                        duration_ms: 1000,
                        matching_samples: 10,
                    }
                })
                .collect();

            let computed_slope = calculate_linear_regression(&results);
            // Allow 0.01% error due to floating point
            let relative_error = ((computed_slope - slope) / slope).abs();
            prop_assert!(relative_error < 0.0001, "computed {computed_slope}, expected {slope}");
        }

        /// Property: Adding a constant to all y values doesn't change slope
        #[test]
        fn linear_regression_invariant_to_y_shift(
            n in 3..20_usize,
            base_throughput in 100.0..1000.0_f64,
            slope in 10.0..100.0_f64,
            shift in -500.0..500.0_f64
        ) {
            let results1: Vec<ExperimentResult> = (0..n)
                .map(|i| {
                    #[allow(clippy::cast_precision_loss)]
                    let speedup = i as f64 * 0.1;
                    let throughput = base_throughput + speedup * slope;
                    ExperimentResult {
                        ip: 0x1000,
                        speedup_pct: speedup,
                        throughput,
                        duration_ms: 1000,
                        matching_samples: 10,
                    }
                })
                .collect();

            let results2: Vec<ExperimentResult> = results1
                .iter()
                .map(|r| ExperimentResult {
                    throughput: r.throughput + shift,
                    ..*r
                })
                .collect();

            let slope1 = calculate_linear_regression(&results1);
            let slope2 = calculate_linear_regression(&results2);

            prop_assert!(
                (slope1 - slope2).abs() < 0.001,
                "slopes should be equal: {slope1} vs {slope2}"
            );
        }

        /// Property: Scaling all x values by a constant scales slope inversely
        #[test]
        fn linear_regression_scales_with_x(
            n in 3..20_usize,
            base_throughput in 100.0..1000.0_f64,
            slope in 10.0..100.0_f64,
            scale in 0.5..2.0_f64
        ) {
            let results1: Vec<ExperimentResult> = (0..n)
                .map(|i| {
                    #[allow(clippy::cast_precision_loss)]
                    let speedup = i as f64 * 0.1;
                    let throughput = base_throughput + speedup * slope;
                    ExperimentResult {
                        ip: 0x1000,
                        speedup_pct: speedup,
                        throughput,
                        duration_ms: 1000,
                        matching_samples: 10,
                    }
                })
                .collect();

            let results2: Vec<ExperimentResult> = results1
                .iter()
                .map(|r| ExperimentResult {
                    speedup_pct: r.speedup_pct * scale,
                    ..*r
                })
                .collect();

            let slope1 = calculate_linear_regression(&results1);
            let slope2 = calculate_linear_regression(&results2);

            // slope2 should be slope1 / scale
            let expected_slope2 = slope1 / scale;
            let relative_error = ((slope2 - expected_slope2) / expected_slope2).abs();
            prop_assert!(
                relative_error < 0.001,
                "slope2 ({slope2}) should be slope1/scale ({expected_slope2})"
            );
        }

        /// Property: ProfilingResults accumulates results correctly
        #[test]
        fn profiling_results_accumulates(n in 1..50_usize) {
            let results = ProfilingResults::new();
            for i in 0..n {
                results.add_result(ExperimentResult {
                    ip: 0x1000,
                    #[allow(clippy::cast_precision_loss)]
                    speedup_pct: i as f64 * 0.1,
                    throughput: 100.0,
                    duration_ms: 1000,
                    matching_samples: 10,
                });
            }

            let ip_results = results.results_for_ip(0x1000).unwrap();
            prop_assert_eq!(ip_results.len(), n);
        }
    }
}
