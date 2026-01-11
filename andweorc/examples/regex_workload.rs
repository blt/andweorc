//! Example: Regex-heavy workload analysis.
//!
//! This example simulates a log parser that uses multiple regex patterns
//! to extract structured data. This is a very common pattern in:
//! - Log analysis tools
//! - Web scrapers
//! - Data extraction pipelines
//!
//! The profiler should identify which regex patterns are the bottleneck
//! and how much speedup could be gained by optimizing them.

// Examples are demonstration code - allow more relaxed rules
#![allow(clippy::unwrap_used)]
#![allow(clippy::expect_used)]
#![allow(clippy::print_stdout)]
#![allow(clippy::uninlined_format_args)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::struct_field_names)]
#![allow(clippy::map_unwrap_or)]

use andweorc::progress;
use regex::Regex;
use std::time::Instant;

/// Log entry structure extracted from log lines.
#[derive(Debug)]
#[allow(dead_code)]
struct LogEntry {
    timestamp: String,
    level: String,
    message: String,
    ip_addresses: Vec<String>,
    urls: Vec<String>,
}

/// Simulates parsing log lines with multiple regex patterns.
struct LogParser {
    timestamp_re: Regex,
    level_re: Regex,
    ip_re: Regex,
    url_re: Regex,
}

impl LogParser {
    fn new() -> Self {
        Self {
            // Timestamp pattern: 2024-01-15T10:30:45.123Z
            timestamp_re: Regex::new(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z?").unwrap(),
            // Log level: INFO, WARN, ERROR, DEBUG
            level_re: Regex::new(r"\b(INFO|WARN|ERROR|DEBUG|TRACE)\b").unwrap(),
            // IP addresses (both IPv4 and simple patterns)
            ip_re: Regex::new(r"\b(?:\d{1,3}\.){3}\d{1,3}\b").unwrap(),
            // URLs - this is the expensive pattern
            url_re: Regex::new(r#"https?://[^\s<>"']+"#).unwrap(),
        }
    }

    /// Parse a single log line - extract all patterns.
    fn parse_line(&self, line: &str) -> LogEntry {
        // Extract timestamp - usually fast
        let timestamp = self
            .timestamp_re
            .find(line)
            .map(|m| m.as_str().to_string())
            .unwrap_or_default();
        progress!("parse_timestamp");

        // Extract log level - very fast
        let level = self
            .level_re
            .find(line)
            .map(|m| m.as_str().to_string())
            .unwrap_or_else(|| "UNKNOWN".to_string());
        progress!("parse_level");

        // Extract all IP addresses - can be slow with many matches
        let ip_addresses: Vec<String> = self
            .ip_re
            .find_iter(line)
            .map(|m| m.as_str().to_string())
            .collect();
        progress!("parse_ips");

        // Extract URLs - this is often the bottleneck
        let urls: Vec<String> = self
            .url_re
            .find_iter(line)
            .map(|m| m.as_str().to_string())
            .collect();
        progress!("parse_urls");

        // Everything after the log level is the message
        let message = line.to_string();

        LogEntry {
            timestamp,
            level,
            message,
            ip_addresses,
            urls,
        }
    }
}

/// Generate synthetic log lines for testing.
fn generate_log_lines(count: usize) -> Vec<String> {
    let levels = ["INFO", "WARN", "ERROR", "DEBUG"];
    let mut lines = Vec::with_capacity(count);

    for i in 0..count {
        let level = levels[i % 4];
        let hour = i % 24;
        let minute = i % 60;
        let ip1 = (i % 256) as u8;
        let ip2 = ((i / 256) % 256) as u8;

        // Generate increasingly complex log lines
        let line = match i % 5 {
            0 => format!(
                "2024-01-15T{:02}:{:02}:45.123Z {} Simple message from 192.168.{}.{}",
                hour, minute, level, ip1, ip2
            ),
            1 => format!(
                "2024-01-15T{:02}:{:02}:45.123Z {} Request from 10.0.{}.{} to https://api.example.com/v1/users",
                hour, minute, level, ip1, ip2
            ),
            2 => format!(
                "2024-01-15T{:02}:{:02}:45.123Z {} Connection 172.16.{}.{} -> 192.168.{}.{} via https://proxy.internal.net/forward",
                hour, minute, level, ip1, ip2, ip2, ip1
            ),
            3 => format!(
                "2024-01-15T{:02}:{:02}:45.123Z {} External API call to https://api.github.com/repos/rust-lang/rust/pulls from client 8.8.{}.{}",
                hour, minute, level, ip1, ip2
            ),
            _ => format!(
                "2024-01-15T{:02}:{:02}:45.123Z {} Multi-hop: 10.0.{}.{} -> 172.16.{}.{} -> 192.168.{}.{} URLs: https://service-a.local/api https://service-b.local/api https://external.com/callback",
                hour, minute, level, ip1, ip2, ip2, ip1, ip1, ip2
            ),
        };
        lines.push(line);
    }
    lines
}

fn main() {
    println!("Regex Workload Example");
    println!("======================");
    println!();
    println!("This example simulates a log parser with multiple regex patterns.");
    println!("Causal profiling can identify which regex patterns are bottlenecks.");
    println!();

    let log_lines = generate_log_lines(10_000);
    let parser = LogParser::new();

    println!("Parsing {} log lines...", log_lines.len());
    let start = Instant::now();

    let mut total_ips = 0;
    let mut total_urls = 0;

    for line in &log_lines {
        let entry = parser.parse_line(line);
        total_ips += entry.ip_addresses.len();
        total_urls += entry.urls.len();
    }

    let elapsed = start.elapsed();

    println!();
    println!("Results:");
    println!("  Lines parsed: {}", log_lines.len());
    println!("  Total IPs found: {}", total_ips);
    println!("  Total URLs found: {}", total_urls);
    println!("  Time: {:?}", elapsed);
    println!(
        "  Throughput: {:.2} lines/sec",
        log_lines.len() as f64 / elapsed.as_secs_f64()
    );
    println!();
    println!("Use causal profiling to find which regex pattern(s) would");
    println!("benefit most from optimization (or caching).");
}
