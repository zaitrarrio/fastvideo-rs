//! Host memory high-water marks for load-time logging.
//!
//! A pod's container limit kills the process (exit 137) with no message, so
//! loaders log the peak they reached: `VmHWM` from `/proc/self/status` (the
//! process's resident-set high-water mark). Unknown off Linux.

/// Peak resident set of this process in bytes, if the OS reports it.
pub fn peak_rss_bytes() -> Option<u64> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    parse_kib_field(&status, "VmHWM:").map(|k| k * 1024)
}

/// [`peak_rss_bytes`] as `"12.3 GiB"`, or `"unknown"`.
pub fn peak_rss_human() -> String {
    peak_rss_bytes().map_or_else(
        || "unknown".to_string(),
        |b| format!("{:.1} GiB", b as f64 / f64::from(1u32 << 30)),
    )
}

fn parse_kib_field(status: &str, field: &str) -> Option<u64> {
    status
        .lines()
        .find_map(|l| l.strip_prefix(field))
        .and_then(|rest| rest.split_whitespace().next())
        .and_then(|n| n.parse().ok())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_vmhwm() {
        let s = "Name:\tfv\nVmPeak:\t 100 kB\nVmHWM:\t  2048 kB\nVmRSS:\t 1024 kB\n";
        assert_eq!(parse_kib_field(s, "VmHWM:"), Some(2048));
        assert_eq!(parse_kib_field(s, "VmSwap:"), None);
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn this_process_has_a_peak() {
        assert!(peak_rss_bytes().is_some_and(|b| b > 0));
    }
}
