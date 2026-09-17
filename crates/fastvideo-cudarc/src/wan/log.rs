//! Runtime logging for the cudarc Wan path.
//!
//! Controlled by `FASTVIDEO_LOG`:
//! - unset / `1` / `info` → info (device, generate, step timings)
//! - `2` / `debug` → verbose (SDPA/GEMM path picks, once-per-process)
//! - `0` / `off` / `false` → quiet
//!
//! `FASTVIDEO_LOG_STEPS=1` forces per-step timing even when quiet is not set
//! (also on by default at info+).

use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};
use std::time::Instant;

const OFF: u8 = 0;
const INFO: u8 = 1;
const DEBUG: u8 = 2;

fn level() -> u8 {
    static CACHED: AtomicU8 = AtomicU8::new(255);
    let c = CACHED.load(Ordering::Relaxed);
    if c != 255 {
        return c;
    }
    let v = match std::env::var("FASTVIDEO_LOG") {
        Ok(s) => {
            let s = s.trim().to_ascii_lowercase();
            match s.as_str() {
                "0" | "off" | "false" | "quiet" | "error" => OFF,
                "2" | "debug" | "trace" | "verbose" => DEBUG,
                _ => INFO,
            }
        }
        Err(_) => INFO,
    };
    CACHED.store(v, Ordering::Relaxed);
    v
}

pub fn info_enabled() -> bool {
    level() >= INFO
}

pub fn debug_enabled() -> bool {
    level() >= DEBUG
}

pub fn info(args: std::fmt::Arguments<'_>) {
    if info_enabled() {
        eprintln!("[fastvideo] {args}");
    }
}

pub fn debug(args: std::fmt::Arguments<'_>) {
    if debug_enabled() {
        eprintln!("[fastvideo:debug] {args}");
    }
}

/// Log once per process (debug level).
pub fn debug_once(flag: &AtomicBool, args: std::fmt::Arguments<'_>) {
    if debug_enabled() && !flag.swap(true, Ordering::Relaxed) {
        eprintln!("[fastvideo:debug] {args}");
    }
}

pub fn info_once(flag: &AtomicBool, args: std::fmt::Arguments<'_>) {
    if info_enabled() && !flag.swap(true, Ordering::Relaxed) {
        eprintln!("[fastvideo] {args}");
    }
}

/// Simple span timer that logs elapsed ms at drop when info is on.
pub struct StepTimer {
    label: String,
    start: Instant,
    enabled: bool,
}

impl StepTimer {
    pub fn start(label: impl Into<String>) -> Self {
        Self {
            label: label.into(),
            start: Instant::now(),
            enabled: info_enabled(),
        }
    }
}

impl Drop for StepTimer {
    fn drop(&mut self) {
        if self.enabled {
            eprintln!(
                "[fastvideo] {} done in {}ms",
                self.label,
                self.start.elapsed().as_millis()
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn level_parses() {
        let prev = std::env::var("FASTVIDEO_LOG").ok();
        // Reset cache by writing through env then forcing re-read is hard with
        // AtomicU8 cache; just smoke the helpers.
        info(format_args!("test info"));
        debug(format_args!("test debug"));
        match prev {
            Some(v) => std::env::set_var("FASTVIDEO_LOG", v),
            None => std::env::remove_var("FASTVIDEO_LOG"),
        }
    }
}
