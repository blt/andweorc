//! JSON output format with schema versioning.
//!
//! This module provides structured JSON output for profiling results,
//! enabling machine-readable analysis and integration with other tools.
//!
//! # Schema Versioning
//!
//! The output includes a schema version to allow consumers to handle
//! format changes gracefully. The current schema version is 1.
//!
//! # Example Output
//!
//! ```json
//! {
//!   "schema_version": 1,
//!   "profiler": "andweorc",
//!   "profiler_version": "0.1.0",
//!   "experiments": [...],
//!   "impacts": [...]
//! }
//! ```

use crate::runner::{ExperimentResult, ProfilingResults, RegressionResult};
use serde::Serialize;

/// Current schema version for the JSON output format.
///
/// Increment this when making breaking changes to the output structure.
/// Consumers can use this to handle format evolution.
pub const SCHEMA_VERSION: u32 = 1;

/// Complete profiling output in JSON format.
///
/// This is the top-level structure serialized to JSON when
/// outputting profiling results.
#[derive(Debug, Clone, Serialize)]
pub struct JsonOutput {
    /// Schema version for format compatibility checking.
    pub schema_version: u32,
    /// Name of the profiler.
    pub profiler: &'static str,
    /// Version of the profiler.
    pub profiler_version: &'static str,
    /// All raw experiment results.
    pub experiments: Vec<ExperimentEntry>,
    /// Calculated causal impacts with statistical analysis.
    pub impacts: Vec<ImpactEntry>,
    /// Metadata about the profiling session.
    pub metadata: Metadata,
}

/// Metadata about the profiling session.
#[derive(Debug, Clone, Copy, Serialize)]
pub struct Metadata {
    /// Total number of experiments run.
    pub total_experiments: usize,
    /// Number of unique instruction pointers tested.
    pub unique_ips: usize,
    /// Number of statistically significant impacts found.
    pub significant_impacts: usize,
}

/// A single experiment entry with resolved symbol information.
#[derive(Debug, Clone, Serialize)]
pub struct ExperimentEntry {
    /// Instruction pointer (hex string for readability).
    pub ip: String,
    /// Resolved symbol or source location (if available).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub symbol: Option<String>,
    /// Speedup percentage applied (0.0 to 1.5).
    pub speedup_pct: f64,
    /// Throughput observed (operations/sec).
    pub throughput: f64,
    /// Duration of the experiment in milliseconds.
    pub duration_ms: u64,
    /// Number of samples that matched the selected IP.
    pub matching_samples: u64,
}

impl ExperimentEntry {
    /// Creates an experiment entry from raw result with optional symbol resolution.
    #[must_use]
    pub fn from_result(result: &ExperimentResult, symbol: Option<String>) -> Self {
        let ip_val = result.ip;
        Self {
            ip: format!("0x{ip_val:x}"),
            symbol,
            speedup_pct: result.speedup_pct,
            throughput: result.throughput,
            duration_ms: result.duration_ms,
            matching_samples: result.matching_samples,
        }
    }
}

/// A causal impact entry with statistical analysis.
#[derive(Debug, Clone, Serialize)]
pub struct ImpactEntry {
    /// Instruction pointer (hex string for readability).
    pub ip: String,
    /// Resolved symbol or source location (if available).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub symbol: Option<String>,
    /// Causal impact (slope of throughput vs speedup).
    pub impact: f64,
    /// Statistical analysis of the impact.
    pub statistics: StatisticsEntry,
}

/// Statistical analysis of a causal impact.
#[derive(Debug, Clone, Copy, Serialize)]
pub struct StatisticsEntry {
    /// Coefficient of determination (R²). 0-1, higher is better fit.
    pub r_squared: f64,
    /// Standard error of the slope estimate.
    pub slope_std_error: f64,
    /// Lower bound of 95% confidence interval for slope.
    pub ci_lower: f64,
    /// Upper bound of 95% confidence interval for slope.
    pub ci_upper: f64,
    /// Number of data points used in the regression.
    pub n: usize,
    /// Whether the impact is statistically significant at 95% confidence.
    pub is_significant: bool,
    /// Whether the regression has a good fit (R² > 0.7).
    pub has_good_fit: bool,
}

impl StatisticsEntry {
    /// Creates a statistics entry from a regression result.
    #[must_use]
    pub fn from_regression(reg: &RegressionResult) -> Self {
        Self {
            r_squared: reg.r_squared,
            slope_std_error: reg.slope_std_error,
            ci_lower: reg.slope_ci_lower,
            ci_upper: reg.slope_ci_upper,
            n: reg.n,
            is_significant: reg.is_significant(),
            has_good_fit: reg.has_good_fit(),
        }
    }
}

impl ImpactEntry {
    /// Creates an impact entry from raw results with optional symbol resolution.
    #[must_use]
    pub fn from_regression(ip: usize, reg: &RegressionResult, symbol: Option<String>) -> Self {
        Self {
            ip: format!("0x{ip:x}"),
            symbol,
            impact: reg.slope,
            statistics: StatisticsEntry::from_regression(reg),
        }
    }
}

/// Generates JSON output from profiling results.
///
/// # Arguments
///
/// * `results` - The profiling results to serialize.
/// * `symbol_resolver` - Optional reference to a function to resolve IP addresses to symbols.
///
/// # Returns
///
/// A `JsonOutput` struct that can be serialized to JSON.
#[must_use]
pub fn generate_output<F>(results: &ProfilingResults, symbol_resolver: Option<&F>) -> JsonOutput
where
    F: Fn(usize) -> Option<String>,
{
    let resolve = |ip: usize| -> Option<String> { symbol_resolver.and_then(|f| f(ip)) };

    // Collect all experiments
    let mut experiments = Vec::new();
    for ip in results.profiled_ips() {
        if let Some(ip_results) = results.results_for_ip(ip) {
            for result in &ip_results {
                experiments.push(ExperimentEntry::from_result(result, resolve(result.ip)));
            }
        }
    }

    // Calculate impacts with statistics
    let impacts_with_stats = results.calculate_impacts_with_stats();
    let impacts: Vec<ImpactEntry> = impacts_with_stats
        .iter()
        .map(|(ip, reg)| ImpactEntry::from_regression(*ip, reg, resolve(*ip)))
        .collect();

    // Count significant impacts
    let significant_count = impacts
        .iter()
        .filter(|i| i.statistics.is_significant && i.statistics.has_good_fit)
        .count();

    // Compute metadata before moving experiments
    let total_experiments = experiments.len();
    let unique_ips = results.profiled_ips().len();

    JsonOutput {
        schema_version: SCHEMA_VERSION,
        profiler: "andweorc",
        profiler_version: env!("CARGO_PKG_VERSION"),
        experiments,
        impacts,
        metadata: Metadata {
            total_experiments,
            unique_ips,
            significant_impacts: significant_count,
        },
    }
}

/// Serializes profiling results to a JSON string.
///
/// # Arguments
///
/// * `results` - The profiling results to serialize.
/// * `symbol_resolver` - Optional reference to a function to resolve IP addresses to symbols.
/// * `pretty` - Whether to use pretty-printed (indented) JSON.
///
/// # Returns
///
/// A JSON string representation of the results.
///
/// # Errors
///
/// Returns an error if JSON serialization fails (should not happen in practice).
pub fn to_json_string<F>(
    results: &ProfilingResults,
    symbol_resolver: Option<&F>,
    pretty: bool,
) -> Result<String, serde_json::Error>
where
    F: Fn(usize) -> Option<String>,
{
    let output = generate_output(results, symbol_resolver);
    if pretty {
        serde_json::to_string_pretty(&output)
    } else {
        serde_json::to_string(&output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runner::ExperimentResult;

    fn make_result(ip: usize, speedup_pct: f64, throughput: f64) -> ExperimentResult {
        ExperimentResult {
            ip,
            speedup_pct,
            throughput,
            duration_ms: 1000,
            matching_samples: 10,
        }
    }

    #[test]
    fn json_output_has_correct_schema_version() {
        let results = ProfilingResults::new();
        let output = generate_output::<fn(usize) -> Option<String>>(&results, None);
        assert_eq!(output.schema_version, SCHEMA_VERSION);
    }

    #[test]
    fn json_output_has_profiler_info() {
        let results = ProfilingResults::new();
        let output = generate_output::<fn(usize) -> Option<String>>(&results, None);
        assert_eq!(output.profiler, "andweorc");
        assert!(!output.profiler_version.is_empty());
    }

    #[test]
    fn json_output_includes_experiments() {
        let results = ProfilingResults::new();
        results.add_result(make_result(0x1000, 0.0, 100.0));
        results.add_result(make_result(0x1000, 0.5, 125.0));
        results.add_result(make_result(0x1000, 1.0, 150.0));

        let output = generate_output::<fn(usize) -> Option<String>>(&results, None);
        assert_eq!(output.experiments.len(), 3);
        assert_eq!(output.metadata.total_experiments, 3);
    }

    #[test]
    fn json_output_resolves_symbols() {
        let results = ProfilingResults::new();
        results.add_result(make_result(0x1000, 0.0, 100.0));
        results.add_result(make_result(0x1000, 0.5, 125.0));
        results.add_result(make_result(0x1000, 1.0, 150.0));

        let resolver = |ip: usize| -> Option<String> {
            if ip == 0x1000 {
                Some("main::process".to_string())
            } else {
                None
            }
        };

        let output = generate_output(&results, Some(&resolver));
        assert!(output.experiments[0].symbol.is_some());
        assert_eq!(
            output.experiments[0].symbol.as_ref().unwrap(),
            "main::process"
        );
    }

    #[test]
    fn json_serialization_works() {
        let results = ProfilingResults::new();
        results.add_result(make_result(0x1000, 0.0, 100.0));
        results.add_result(make_result(0x1000, 0.5, 125.0));
        results.add_result(make_result(0x1000, 1.0, 150.0));

        let json = to_json_string::<fn(usize) -> Option<String>>(&results, None, false).unwrap();
        assert!(json.contains("\"schema_version\":1"));
        assert!(json.contains("\"profiler\":\"andweorc\""));
    }

    #[test]
    fn json_pretty_print_works() {
        let results = ProfilingResults::new();
        results.add_result(make_result(0x1000, 0.0, 100.0));

        let json = to_json_string::<fn(usize) -> Option<String>>(&results, None, true).unwrap();
        // Pretty print should have newlines
        assert!(json.contains('\n'));
    }

    #[test]
    fn ip_format_is_hex() {
        let entry = ExperimentEntry::from_result(&make_result(0x1234, 0.0, 100.0), None);
        assert_eq!(entry.ip, "0x1234");
    }

    #[test]
    fn statistics_entry_captures_significance() {
        let reg = RegressionResult {
            slope: 50.0,
            intercept: 100.0,
            r_squared: 0.95,
            slope_std_error: 5.0,
            slope_ci_lower: 40.0,
            slope_ci_upper: 60.0,
            n: 10,
        };

        let stats = StatisticsEntry::from_regression(&reg);
        assert!(stats.is_significant); // CI doesn't cross zero
        assert!(stats.has_good_fit); // R² > 0.7
    }
}
