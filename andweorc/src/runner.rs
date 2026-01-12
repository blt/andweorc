//! Experiment runner for causal profiling.
//!
//! This module implements the main experiment loop that:
//! 1. Collects samples to identify hot code locations
//! 2. Runs experiments with different virtual speedups
//! 3. Measures throughput changes
//! 4. Records results for analysis

use crate::experiment::get_instance;
use crate::lock_util::{recover_read, recover_write};
use crate::progress_point::Progress;
use crate::timer::nanosleep;
use libc::c_void;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, RwLock};
use std::time::Duration;

/// Number of baseline entries at the start of the delay table.
const BASELINE_ENTRIES: usize = 25;

/// Converts a delay table index to a speedup percentage.
///
/// The delay table has 25 baseline entries (0%) followed by
/// speedup entries from 5% to 150% in 5% increments.
#[allow(clippy::cast_precision_loss)]
fn delay_index_to_speedup_pct(index: usize) -> f64 {
    if index < BASELINE_ENTRIES {
        0.0
    } else {
        (index - BASELINE_ENTRIES + 1) as f64 * 0.05
    }
}

/// Converts a configuration index to a delay table index.
///
/// The first entry (index 0) maps to baseline (delay index 0).
/// Subsequent entries map to speedup indices starting at 25.
fn config_index_to_delay_index(config_index: usize) -> usize {
    if config_index == 0 {
        0 // Baseline
    } else {
        BASELINE_ENTRIES - 1 + config_index // Speedup entries start at index 25
    }
}

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
    pub fn add_result(&self, result: ExperimentResult) {
        let mut guard = recover_write(self.results.write());
        guard.entry(result.ip).or_default().push(result);
    }

    /// Returns all results for a given IP.
    #[must_use]
    pub fn results_for_ip(&self, ip: usize) -> Option<Vec<ExperimentResult>> {
        let guard = recover_read(self.results.read());
        guard.get(&ip).cloned()
    }

    /// Returns all unique IPs that have been profiled.
    #[must_use]
    pub fn profiled_ips(&self) -> Vec<usize> {
        let guard = recover_read(self.results.read());
        guard.keys().copied().collect()
    }

    /// Calculates the causal impact for each IP.
    ///
    /// Returns a list of (IP, impact) pairs sorted by impact descending.
    /// Impact is calculated as the slope of throughput vs speedup percentage.
    #[must_use]
    pub fn calculate_impacts(&self) -> Vec<(usize, f64)> {
        let guard = recover_read(self.results.read());
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

    /// Calculates the causal impact for each IP with full statistical analysis.
    ///
    /// Returns a list of (IP, `RegressionResult`) pairs sorted by impact descending.
    /// Only includes IPs with at least 3 data points (required for confidence intervals).
    ///
    /// The `RegressionResult` includes:
    /// - Slope (causal impact)
    /// - R² (goodness of fit)
    /// - 95% confidence intervals
    /// - Statistical significance indicator
    #[must_use]
    pub fn calculate_impacts_with_stats(&self) -> Vec<(usize, RegressionResult)> {
        let guard = recover_read(self.results.read());
        let mut impacts: Vec<(usize, RegressionResult)> = guard
            .iter()
            .filter_map(|(&ip, results)| {
                // Need at least 3 points for confidence intervals
                calculate_linear_regression_full(results).map(|reg| (ip, reg))
            })
            .collect();

        // Sort by impact (slope) descending
        impacts.sort_by(|a, b| {
            b.1.slope
                .partial_cmp(&a.1.slope)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        impacts
    }

    /// Returns only statistically significant impacts.
    ///
    /// Filters to IPs where:
    /// - The 95% confidence interval doesn't include zero
    /// - The R² is above 0.5 (reasonable fit)
    #[must_use]
    pub fn significant_impacts(&self) -> Vec<(usize, RegressionResult)> {
        self.calculate_impacts_with_stats()
            .into_iter()
            .filter(|(_, reg)| reg.is_significant() && reg.r_squared > 0.5)
            .collect()
    }
}

/// Results from linear regression analysis.
///
/// Provides slope, intercept, confidence intervals, and goodness-of-fit metrics.
/// These statistics allow users to assess the reliability of causal impact claims.
#[derive(Debug, Clone, Copy)]
pub struct RegressionResult {
    /// Slope of the regression line (throughput change per unit speedup).
    /// This is the primary "causal impact" metric.
    pub slope: f64,
    /// Y-intercept of the regression line (baseline throughput).
    pub intercept: f64,
    /// Coefficient of determination (R²). Ranges from 0 to 1.
    /// Higher values indicate the speedup explains more variance in throughput.
    pub r_squared: f64,
    /// Standard error of the slope estimate.
    pub slope_std_error: f64,
    /// Lower bound of 95% confidence interval for slope.
    pub slope_ci_lower: f64,
    /// Upper bound of 95% confidence interval for slope.
    pub slope_ci_upper: f64,
    /// Number of data points used in the regression.
    pub n: usize,
}

impl RegressionResult {
    /// Returns true if the slope is statistically significant at 95% confidence.
    ///
    /// A significant slope means the confidence interval doesn't include zero,
    /// indicating the speedup has a measurable effect on throughput.
    #[must_use]
    pub fn is_significant(&self) -> bool {
        // Significant if CI doesn't cross zero
        (self.slope_ci_lower > 0.0 && self.slope_ci_upper > 0.0)
            || (self.slope_ci_lower < 0.0 && self.slope_ci_upper < 0.0)
    }

    /// Returns true if the regression fit is good (R² > 0.7).
    #[must_use]
    pub fn has_good_fit(&self) -> bool {
        self.r_squared > 0.7
    }
}

/// Calculates linear regression with confidence intervals and R².
///
/// Uses least squares regression with t-distribution confidence intervals.
/// Returns `None` if there are fewer than 3 data points (need df >= 1 for CI).
fn calculate_linear_regression_full(results: &[ExperimentResult]) -> Option<RegressionResult> {
    let n = results.len();
    if n < 3 {
        return None; // Need at least 3 points for meaningful CI
    }

    #[allow(clippy::cast_precision_loss)]
    let n_f64 = n as f64;

    // Calculate means
    let sum_x: f64 = results.iter().map(|r| r.speedup_pct).sum();
    let sum_y: f64 = results.iter().map(|r| r.throughput).sum();
    let mean_x = sum_x / n_f64;
    let mean_y = sum_y / n_f64;

    // Calculate sums of squares and cross products
    let mut sum_sq_x = 0.0; // Sum of (x - mean_x)^2
    let mut sum_sq_y = 0.0; // Sum of (y - mean_y)^2
    let mut sum_cross = 0.0; // Sum of (x - mean_x)(y - mean_y)

    for r in results {
        let dx = r.speedup_pct - mean_x;
        let dy = r.throughput - mean_y;
        sum_sq_x += dx * dx;
        sum_sq_y += dy * dy;
        sum_cross += dx * dy;
    }

    // Check for degenerate cases
    if sum_sq_x.abs() < f64::EPSILON {
        return None; // All x values are the same
    }

    // Calculate slope and intercept
    let slope = sum_cross / sum_sq_x;
    let intercept = mean_y - slope * mean_x;

    // Calculate R² (coefficient of determination)
    let r_squared = if sum_sq_y.abs() < f64::EPSILON {
        1.0 // All y values are the same, perfect fit to horizontal line
    } else {
        let reg_sum_sq = slope * slope * sum_sq_x; // Regression sum of squares
        reg_sum_sq / sum_sq_y
    };

    // Calculate residual sum of squares for standard error
    let mut resid_sum_sq = 0.0;
    for r in results {
        let predicted = intercept + slope * r.speedup_pct;
        let residual = r.throughput - predicted;
        resid_sum_sq += residual * residual;
    }

    // Standard error of regression (residual standard deviation)
    let df = n_f64 - 2.0; // Degrees of freedom
    let s_squared = resid_sum_sq / df;
    let s = s_squared.sqrt();

    // Standard error of slope
    let slope_std_error = s / sum_sq_x.sqrt();

    // t-value for 95% CI with (n-2) degrees of freedom
    // Using approximation for t-distribution critical value
    let t_critical = t_critical_value_95(n - 2);

    // Confidence interval for slope
    let margin = t_critical * slope_std_error;
    let slope_ci_lower = slope - margin;
    let slope_ci_upper = slope + margin;

    Some(RegressionResult {
        slope,
        intercept,
        r_squared,
        slope_std_error,
        slope_ci_lower,
        slope_ci_upper,
        n,
    })
}

/// Returns the critical t-value for 95% confidence interval.
///
/// Uses a lookup table for common degrees of freedom, with interpolation
/// for values not in the table.
fn t_critical_value_95(df: usize) -> f64 {
    // t-critical values for 95% CI (two-tailed, alpha = 0.05)
    // From standard t-distribution tables
    match df {
        0 => f64::INFINITY,
        1 => 12.706,
        2 => 4.303,
        3 => 3.182,
        4 => 2.776,
        5 => 2.571,
        6 => 2.447,
        7 => 2.365,
        8 => 2.306,
        9 => 2.262,
        10 => 2.228,
        11 => 2.201,
        12 => 2.179,
        13 => 2.160,
        14 => 2.145,
        15 => 2.131,
        16 => 2.120,
        17 => 2.110,
        18 => 2.101,
        19 => 2.093,
        20 => 2.086,
        21..=25 => 2.060,
        26..=30 => 2.042,
        31..=40 => 2.021,
        41..=60 => 2.000,
        61..=120 => 1.980,
        _ => 1.960, // Approaches z-value for large df
    }
}

/// Calculates the linear regression slope of throughput vs speedup.
///
/// This is the simple version for backwards compatibility.
/// For full statistics including confidence intervals, use
/// `calculate_linear_regression_full`.
fn calculate_linear_regression(results: &[ExperimentResult]) -> f64 {
    calculate_linear_regression_full(results).map_or(0.0, |r| r.slope)
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

        // Start the experiment - enable delay injection
        crate::set_profiling_active(true);
        if let Some(ip_val) = ip {
            experiment.start_experiment(ip_val as *const c_void, speedup_index);
        }

        // Wait for the round duration
        let _ = nanosleep(self.config.round_duration);

        // Stop the experiment - disable delay injection
        experiment.stop_experiment();
        crate::set_profiling_active(false);

        // Get results
        let stats = experiment.stats();
        let throughput = progress_point.throughput();
        let elapsed_ns = progress_point.elapsed_nanos();

        // Determine speedup percentage from index
        let speedup_pct = delay_index_to_speedup_pct(speedup_index);

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
                // Map configuration index to delay table index
                let speedup_index = config_index_to_delay_index(idx);

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

        // IP 0x1000: positive slope of 50 (good optimization target)
        results.add_result(ExperimentResult {
            ip: 0x1000,
            speedup_pct: 0.0,
            throughput: 100.0,
            duration_ms: 1000,
            matching_samples: 10,
        });
        results.add_result(ExperimentResult {
            ip: 0x1000,
            speedup_pct: 0.5,
            throughput: 125.0,
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

        // IP 0x2000: smaller positive slope of 10
        results.add_result(ExperimentResult {
            ip: 0x2000,
            speedup_pct: 0.0,
            throughput: 100.0,
            duration_ms: 1000,
            matching_samples: 10,
        });
        results.add_result(ExperimentResult {
            ip: 0x2000,
            speedup_pct: 0.5,
            throughput: 105.0,
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
        // First should be 0x1000 with higher impact (slope 50 > slope 10)
        assert_eq!(impacts[0].0, 0x1000);
        assert_eq!(impacts[1].0, 0x2000);
        assert!(
            impacts[0].1 > impacts[1].1,
            "slope {} should be > {}",
            impacts[0].1,
            impacts[1].1
        );
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

        /// Property: R² is between 0 and 1 for all valid data
        #[test]
        fn r_squared_in_valid_range(
            n in 3..20_usize,
            base_throughput in 100.0..1000.0_f64,
            slope in 10.0..100.0_f64,
            noise_scale in 0.0..50.0_f64
        ) {
            let results: Vec<ExperimentResult> = (0..n)
                .map(|i| {
                    #[allow(clippy::cast_precision_loss)]
                    let speedup = i as f64 * 0.1;
                    // Add some noise based on index (deterministic for reproducibility)
                    #[allow(clippy::cast_precision_loss)]
                    let noise = ((i as f64 * 7.3).sin()) * noise_scale;
                    let throughput = base_throughput + speedup * slope + noise;
                    ExperimentResult {
                        ip: 0x1000,
                        speedup_pct: speedup,
                        throughput,
                        duration_ms: 1000,
                        matching_samples: 10,
                    }
                })
                .collect();

            if let Some(reg) = calculate_linear_regression_full(&results) {
                prop_assert!(reg.r_squared >= 0.0, "R² should be >= 0, got {}", reg.r_squared);
                prop_assert!(reg.r_squared <= 1.0 + f64::EPSILON, "R² should be <= 1, got {}", reg.r_squared);
            }
        }

        /// Property: Confidence interval contains true slope for perfect linear data
        #[test]
        fn ci_contains_true_slope_for_perfect_data(
            n in 5..20_usize,
            base_throughput in 100.0..1000.0_f64,
            slope in 10.0..100.0_f64
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

            if let Some(reg) = calculate_linear_regression_full(&results) {
                // For perfect data, slope should equal input and CI should be very tight
                let slope_error = (reg.slope - slope).abs();
                prop_assert!(slope_error < 0.001, "slope error {} too large", slope_error);
                // R² should be 1.0 for perfect linear data
                prop_assert!(reg.r_squared > 0.999, "R² should be ~1, got {}", reg.r_squared);
            }
        }
    }
}

#[cfg(test)]
mod regression_stats_tests {
    use super::*;

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
    fn regression_full_needs_three_points() {
        let results = vec![make_result(0.0, 100.0), make_result(1.0, 150.0)];
        assert!(calculate_linear_regression_full(&results).is_none());

        let results = vec![
            make_result(0.0, 100.0),
            make_result(0.5, 125.0),
            make_result(1.0, 150.0),
        ];
        assert!(calculate_linear_regression_full(&results).is_some());
    }

    #[test]
    fn perfect_linear_data_has_r_squared_one() {
        let results = vec![
            make_result(0.0, 100.0),
            make_result(0.5, 125.0),
            make_result(1.0, 150.0),
        ];
        let reg = calculate_linear_regression_full(&results).unwrap();
        assert!(
            (reg.r_squared - 1.0).abs() < 0.001,
            "R² should be 1.0, got {}",
            reg.r_squared
        );
    }

    #[test]
    fn constant_y_has_r_squared_one() {
        // All same throughput = horizontal line = perfect fit to that line
        let results = vec![
            make_result(0.0, 100.0),
            make_result(0.5, 100.0),
            make_result(1.0, 100.0),
        ];
        let reg = calculate_linear_regression_full(&results).unwrap();
        // R² = 1 because there's no variance to explain
        assert!(
            (reg.r_squared - 1.0).abs() < 0.001,
            "R² should be 1.0 for constant y, got {}",
            reg.r_squared
        );
        // Slope should be 0
        assert!(
            reg.slope.abs() < 0.001,
            "slope should be 0 for constant y, got {}",
            reg.slope
        );
    }

    #[test]
    fn noisy_data_has_lower_r_squared() {
        // Perfect data
        let perfect = vec![
            make_result(0.0, 100.0),
            make_result(0.5, 125.0),
            make_result(1.0, 150.0),
        ];
        let reg_perfect = calculate_linear_regression_full(&perfect).unwrap();

        // Noisy data (same x values, scattered y)
        let noisy = vec![
            make_result(0.0, 100.0),
            make_result(0.5, 110.0), // Not on the line
            make_result(1.0, 150.0),
        ];
        let reg_noisy = calculate_linear_regression_full(&noisy).unwrap();

        assert!(
            reg_noisy.r_squared < reg_perfect.r_squared,
            "noisy R² ({}) should be less than perfect R² ({})",
            reg_noisy.r_squared,
            reg_perfect.r_squared
        );
    }

    #[test]
    fn significant_positive_slope() {
        // Clear positive trend
        let results = vec![
            make_result(0.0, 100.0),
            make_result(0.25, 112.0),
            make_result(0.5, 125.0),
            make_result(0.75, 137.0),
            make_result(1.0, 150.0),
        ];
        let reg = calculate_linear_regression_full(&results).unwrap();

        assert!(reg.slope > 0.0, "slope should be positive");
        assert!(
            reg.is_significant(),
            "should be significant: CI [{}, {}]",
            reg.slope_ci_lower,
            reg.slope_ci_upper
        );
        assert!(reg.slope_ci_lower > 0.0, "CI lower bound should be > 0");
    }

    #[test]
    fn non_significant_flat_data() {
        // Essentially flat with tiny variation
        let results = vec![
            make_result(0.0, 100.0),
            make_result(0.25, 100.1),
            make_result(0.5, 99.9),
            make_result(0.75, 100.2),
            make_result(1.0, 100.0),
        ];
        let reg = calculate_linear_regression_full(&results).unwrap();

        // Slope should be near zero
        assert!(reg.slope.abs() < 1.0, "slope should be near zero");
        // CI should include zero (not significant)
        assert!(
            !reg.is_significant() || reg.slope.abs() < 0.5,
            "should not be significant or have very small slope"
        );
    }

    #[test]
    fn t_critical_decreases_with_df() {
        // More data points = narrower CI = smaller t-critical
        assert!(t_critical_value_95(5) > t_critical_value_95(10));
        assert!(t_critical_value_95(10) > t_critical_value_95(20));
        assert!(t_critical_value_95(20) > t_critical_value_95(100));
    }

    #[test]
    fn ci_narrows_with_more_data() {
        // 5 points
        let small = vec![
            make_result(0.0, 100.0),
            make_result(0.25, 112.0),
            make_result(0.5, 125.0),
            make_result(0.75, 137.0),
            make_result(1.0, 150.0),
        ];
        let reg_small = calculate_linear_regression_full(&small).unwrap();

        // 10 points (same slope, interpolated)
        let large: Vec<_> = (0..10)
            .map(|i| {
                #[allow(clippy::cast_precision_loss)]
                let x = i as f64 * 0.1;
                make_result(x, 100.0 + x * 50.0)
            })
            .collect();
        let reg_large = calculate_linear_regression_full(&large).unwrap();

        let ci_width_small = reg_small.slope_ci_upper - reg_small.slope_ci_lower;
        let ci_width_large = reg_large.slope_ci_upper - reg_large.slope_ci_lower;

        assert!(
            ci_width_large < ci_width_small,
            "larger sample should have narrower CI: {} vs {}",
            ci_width_large,
            ci_width_small
        );
    }
}
