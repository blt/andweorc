//! Cargo subcommand for the andweorc causal profiler.
//!
//! This binary provides the `cargo andweorc` command for running Rust programs
//! with causal profiling enabled.
//!
//! # Usage
//!
//! ```bash
//! cargo andweorc run --bin myapp
//! cargo andweorc report profile.json
//! ```

// CLI tools need to print to stdout/stderr
#![allow(clippy::print_stdout, clippy::print_stderr)]

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::BufReader;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

/// Cargo subcommand for causal profiling.
#[derive(Parser, Debug)]
#[command(name = "cargo")]
#[command(bin_name = "cargo")]
struct Cargo {
    #[command(subcommand)]
    command: CargoCommand,
}

/// Cargo andweorc subcommand.
#[derive(Subcommand, Debug)]
enum CargoCommand {
    /// Causal profiler for Rust programs.
    Andweorc(AndweorcArgs),
}

/// Andweorc command line arguments.
#[derive(Parser, Debug)]
#[command(version, about)]
struct AndweorcArgs {
    #[command(subcommand)]
    command: AndweorcCommand,
}

/// Andweorc subcommands.
#[derive(Subcommand, Debug)]
enum AndweorcCommand {
    /// Run a binary with causal profiling enabled.
    Run(RunArgs),
    /// Generate a report from profiling data.
    Report(ReportArgs),
}

/// Arguments for the run subcommand.
#[derive(Parser, Debug)]
struct RunArgs {
    /// Name of the binary to run.
    #[arg(long)]
    bin: Option<String>,

    /// Name of the example to run.
    #[arg(long)]
    example: Option<String>,

    /// Build and run in release mode.
    #[arg(long)]
    release: bool,

    /// Output file for profiling results (JSON).
    #[arg(long, short, default_value = "profile.json")]
    output: PathBuf,

    /// Duration of each experiment round in milliseconds.
    #[arg(long, default_value = "1000")]
    round_duration_ms: u64,

    /// Number of baseline rounds to run.
    #[arg(long, default_value = "5")]
    baseline_rounds: usize,

    /// Additional arguments to pass to the binary.
    #[arg(last = true)]
    args: Vec<String>,
}

/// Arguments for the report subcommand.
#[derive(Parser, Debug)]
struct ReportArgs {
    /// Path to the profiling data file.
    input: PathBuf,

    /// Output format (text, json, csv).
    #[arg(long, short, default_value = "text")]
    format: String,
}

/// A single experiment result from the profiler.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct ExperimentResult {
    /// The instruction pointer address.
    ip: u64,
    /// Speedup percentage applied.
    speedup_pct: f64,
    /// Throughput observed.
    throughput: f64,
    /// Duration in milliseconds.
    duration_ms: u64,
    /// Number of matching samples.
    matching_samples: u64,
}

/// Source location information.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
struct SourceLocation {
    /// Source file path.
    file: String,
    /// Line number.
    line: u32,
    /// Function name (if known).
    function: Option<String>,
}

/// Causal impact result for a source location.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct CausalImpact {
    /// Source location.
    location: SourceLocation,
    /// Linear regression slope (throughput change per speedup %).
    impact: f64,
    /// Number of experiments at this location.
    experiment_count: usize,
    /// Average throughput at baseline.
    baseline_throughput: f64,
    /// Predicted throughput at 100% speedup.
    predicted_max_throughput: f64,
}

/// Full profiling report.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct ProfileReport {
    /// Binary that was profiled.
    binary: String,
    /// Total profiling duration in seconds.
    duration_secs: f64,
    /// Causal impacts sorted by magnitude.
    impacts: Vec<CausalImpact>,
}

fn main() -> Result<()> {
    let args = Cargo::parse();

    match args.command {
        CargoCommand::Andweorc(andweorc) => match andweorc.command {
            AndweorcCommand::Run(run_args) => run_profiler(&run_args),
            AndweorcCommand::Report(report_args) => generate_report(&report_args),
        },
    }
}

/// Runs the profiler on the target binary.
fn run_profiler(args: &RunArgs) -> Result<()> {
    // Step 1: Build the target with debug info and frame pointers
    let build_result = build_target(args)?;
    let binary_path = build_result.binary_path;

    eprintln!("Built: {}", binary_path.display());

    // Step 2: Run the binary with profiling
    eprintln!("Running with causal profiling...");
    eprintln!(
        "  Round duration: {}ms, Baseline rounds: {}",
        args.round_duration_ms, args.baseline_rounds
    );

    let output = run_binary(&binary_path, args)?;

    // Step 3: Parse output and resolve symbols
    let results = parse_profiler_output(&output);
    let resolved = resolve_symbols(&binary_path, &results)?;

    // Step 4: Calculate causal impacts
    let impacts = calculate_impacts(&resolved);

    // Step 5: Write results
    let report = ProfileReport {
        binary: binary_path
            .file_name()
            .map_or_else(|| "unknown".to_string(), |n| n.to_string_lossy().to_string()),
        duration_secs: results
            .iter()
            .map(|r| f64::from(u32::try_from(r.duration_ms).unwrap_or(u32::MAX)) / 1000.0)
            .sum(),
        impacts,
    };

    let json = serde_json::to_string_pretty(&report)?;
    std::fs::write(&args.output, &json)?;
    eprintln!("Results written to: {}", args.output.display());

    // Print summary to stdout
    print_summary(&report);

    Ok(())
}

/// Build result containing the path to the built binary.
struct BuildResult {
    binary_path: PathBuf,
}

/// Builds the target binary with appropriate flags.
fn build_target(args: &RunArgs) -> Result<BuildResult> {
    let mut cmd = Command::new("cargo");
    cmd.arg("build");

    if args.release {
        cmd.arg("--release");
    }

    if let Some(ref bin) = args.bin {
        cmd.arg("--bin").arg(bin);
    } else if let Some(ref example) = args.example {
        cmd.arg("--example").arg(example);
    }

    // Add RUSTFLAGS for debug info and frame pointers
    let rustflags = std::env::var("RUSTFLAGS").unwrap_or_default();
    let new_rustflags = format!("{rustflags} -C debuginfo=2 -C force-frame-pointers=yes");
    cmd.env("RUSTFLAGS", new_rustflags);

    cmd.stdout(Stdio::inherit());
    cmd.stderr(Stdio::inherit());

    let status = cmd.status().context("Failed to run cargo build")?;
    if !status.success() {
        anyhow::bail!("cargo build failed");
    }

    // Determine the binary path
    let target_dir = find_target_dir()?;
    let profile_dir = if args.release { "release" } else { "debug" };

    let binary_name = if let Some(ref bin) = args.bin {
        bin.clone()
    } else if let Some(ref example) = args.example {
        example.clone()
    } else {
        // Try to get package name from Cargo.toml
        get_package_name()?
    };

    let mut binary_path = target_dir.join(profile_dir);
    if args.example.is_some() {
        binary_path = binary_path.join("examples");
    }
    binary_path = binary_path.join(&binary_name);

    if !binary_path.exists() {
        anyhow::bail!("Binary not found at: {}", binary_path.display());
    }

    Ok(BuildResult { binary_path })
}

/// Finds the target directory.
fn find_target_dir() -> Result<PathBuf> {
    // Use cargo metadata to find target dir
    let output = Command::new("cargo")
        .args(["metadata", "--format-version=1", "--no-deps"])
        .output()
        .context("Failed to run cargo metadata")?;

    if !output.status.success() {
        anyhow::bail!("cargo metadata failed");
    }

    let metadata: serde_json::Value =
        serde_json::from_slice(&output.stdout).context("Failed to parse cargo metadata")?;

    metadata["target_directory"]
        .as_str()
        .map(PathBuf::from)
        .context("target_directory not found in metadata")
}

/// Gets the package name from Cargo.toml.
fn get_package_name() -> Result<String> {
    let output = Command::new("cargo")
        .args(["metadata", "--format-version=1", "--no-deps"])
        .output()
        .context("Failed to run cargo metadata")?;

    if !output.status.success() {
        anyhow::bail!("cargo metadata failed");
    }

    let metadata: serde_json::Value =
        serde_json::from_slice(&output.stdout).context("Failed to parse cargo metadata")?;

    metadata["packages"][0]["name"]
        .as_str()
        .map(String::from)
        .context("package name not found")
}

/// Runs the binary and captures profiler output.
fn run_binary(binary_path: &Path, args: &RunArgs) -> Result<String> {
    let mut cmd = Command::new(binary_path);
    cmd.args(&args.args);

    // Set environment for profiling
    cmd.env("ANDWEORC_ENABLED", "1");
    cmd.env(
        "ANDWEORC_ROUND_DURATION_MS",
        args.round_duration_ms.to_string(),
    );
    cmd.env("ANDWEORC_BASELINE_ROUNDS", args.baseline_rounds.to_string());

    cmd.stdout(Stdio::piped());
    cmd.stderr(Stdio::piped());

    let child = cmd.spawn().context("Failed to start binary")?;
    let output = child
        .wait_with_output()
        .context("Failed to wait for binary")?;

    // Combine stdout and stderr for profiler output
    let mut result = String::from_utf8_lossy(&output.stdout).to_string();
    result.push_str(&String::from_utf8_lossy(&output.stderr));

    Ok(result)
}

/// Parses profiler output to extract experiment results.
fn parse_profiler_output(output: &str) -> Vec<ExperimentResult> {
    let mut results = Vec::new();

    for line in output.lines() {
        // Look for PROGRESS lines with format: PROGRESS: name -> count
        if line.starts_with("PROGRESS:") {
            // This is a progress point notification, not an experiment result
            continue;
        }

        // Look for EXPERIMENT lines with format:
        // EXPERIMENT: ip=0x... speedup=0.XX throughput=XXX duration=XXX samples=XXX
        if let Some(rest) = line.strip_prefix("EXPERIMENT:") {
            results.push(parse_experiment_line(rest));
        }
    }

    results
}

/// Parses a single experiment output line.
fn parse_experiment_line(line: &str) -> ExperimentResult {
    let mut ip = 0u64;
    let mut speedup_pct = 0.0;
    let mut throughput = 0.0;
    let mut duration_ms = 0u64;
    let mut matching_samples = 0u64;

    for part in line.split_whitespace() {
        if let Some(val) = part.strip_prefix("ip=") {
            ip = val.strip_prefix("0x").map_or_else(
                || val.parse().unwrap_or(0),
                |hex| u64::from_str_radix(hex, 16).unwrap_or(0),
            );
        } else if let Some(val) = part.strip_prefix("speedup=") {
            speedup_pct = val.parse().unwrap_or(0.0);
        } else if let Some(val) = part.strip_prefix("throughput=") {
            throughput = val.parse().unwrap_or(0.0);
        } else if let Some(val) = part.strip_prefix("duration=") {
            duration_ms = val.parse().unwrap_or(0);
        } else if let Some(val) = part.strip_prefix("samples=") {
            matching_samples = val.parse().unwrap_or(0);
        }
    }

    ExperimentResult {
        ip,
        speedup_pct,
        throughput,
        duration_ms,
        matching_samples,
    }
}

/// Resolves IP addresses to source locations using debug info.
fn resolve_symbols(
    binary_path: &Path,
    results: &[ExperimentResult],
) -> Result<HashMap<SourceLocation, Vec<ExperimentResult>>> {
    let file = File::open(binary_path)?;
    let mmap = unsafe { memmap2::Mmap::map(&file)? };
    let object = object::File::parse(&*mmap)?;
    let ctx = addr2line::Context::new(&object)?;

    let mut resolved: HashMap<SourceLocation, Vec<ExperimentResult>> = HashMap::new();

    for result in results {
        let location = if result.ip == 0 {
            // Baseline result (no specific IP)
            SourceLocation {
                file: "<baseline>".to_string(),
                line: 0,
                function: None,
            }
        } else if let Ok(Some(loc)) = ctx.find_location(result.ip) {
            SourceLocation {
                file: loc.file.unwrap_or("<unknown>").to_string(),
                line: loc.line.unwrap_or(0),
                function: None,
            }
        } else {
            SourceLocation {
                file: format!("<unknown:0x{:x}>", result.ip),
                line: 0,
                function: None,
            }
        };

        resolved.entry(location).or_default().push(result.clone());
    }

    Ok(resolved)
}

/// Calculates causal impacts from resolved experiment results.
fn calculate_impacts(resolved: &HashMap<SourceLocation, Vec<ExperimentResult>>) -> Vec<CausalImpact> {
    let mut impacts: Vec<CausalImpact> = resolved
        .iter()
        .filter_map(|(location, results)| {
            if results.len() < 2 {
                return None;
            }

            // Calculate linear regression of throughput vs speedup
            #[allow(clippy::cast_precision_loss)]
            let n = results.len() as f64;

            let sum_x: f64 = results.iter().map(|r| r.speedup_pct).sum();
            let sum_y: f64 = results.iter().map(|r| r.throughput).sum();
            let mean_x = sum_x / n;
            let mean_y = sum_y / n;

            let mut numerator = 0.0;
            let mut denominator = 0.0;
            for r in results {
                let dx = r.speedup_pct - mean_x;
                let dy = r.throughput - mean_y;
                numerator += dx * dy;
                denominator += dx * dx;
            }

            if denominator.abs() < f64::EPSILON {
                return None;
            }

            let slope = numerator / denominator;
            let intercept = mean_y - slope * mean_x;

            Some(CausalImpact {
                location: location.clone(),
                impact: slope,
                experiment_count: results.len(),
                baseline_throughput: intercept,
                predicted_max_throughput: intercept + slope * 1.0,
            })
        })
        .collect();

    // Sort by impact magnitude (descending)
    impacts.sort_by(|a, b| {
        b.impact
            .abs()
            .partial_cmp(&a.impact.abs())
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    impacts
}

/// Prints a summary of the profiling results.
fn print_summary(report: &ProfileReport) {
    println!("\n=== Causal Profiling Results ===");
    println!("Binary: {}", report.binary);
    println!("Total duration: {:.2}s", report.duration_secs);
    println!("\nTop Optimization Opportunities:");
    println!("{:-<80}", "");

    for (i, impact) in report.impacts.iter().take(10).enumerate() {
        let improvement = if impact.baseline_throughput > 0.0 {
            (impact.predicted_max_throughput - impact.baseline_throughput)
                / impact.baseline_throughput
                * 100.0
        } else {
            0.0
        };

        println!(
            "{}. {}:{} (impact: {:.2}, potential: {:.1}% improvement)",
            i + 1,
            impact.location.file,
            impact.location.line,
            impact.impact,
            improvement
        );
    }

    if report.impacts.is_empty() {
        println!("  No optimization opportunities identified.");
        println!("  Make sure the binary uses andweorc::progress! macros.");
    }
}

/// Generates a report from saved profiling data.
fn generate_report(args: &ReportArgs) -> Result<()> {
    let file = File::open(&args.input).context("Failed to open input file")?;
    let report: ProfileReport =
        serde_json::from_reader(BufReader::new(file)).context("Failed to parse profile data")?;

    match args.format.as_str() {
        "json" => {
            let json = serde_json::to_string_pretty(&report)?;
            println!("{json}");
        }
        "csv" => {
            println!("file,line,impact,experiment_count,baseline_throughput,predicted_max_throughput");
            for impact in &report.impacts {
                println!(
                    "{},{},{},{},{},{}",
                    impact.location.file,
                    impact.location.line,
                    impact.impact,
                    impact.experiment_count,
                    impact.baseline_throughput,
                    impact.predicted_max_throughput
                );
            }
        }
        _ => {
            print_summary(&report);
        }
    }

    Ok(())
}
