//! External tool discovery.
//!
//! `FV_FFMPEG` / `FV_FFPROBE` override the binaries; otherwise `ffmpeg` and
//! `ffprobe` are looked up on `PATH`. The serve image ships ffmpeg (§6.1);
//! CPU CI may not, so every ffmpeg-backed test checks [`ffmpeg_available`]
//! first and skips when it is false.

use std::ffi::OsString;
use std::path::PathBuf;
use std::process::{Command, Stdio};

use crate::error::{MediaError, Result};

fn bin(env: &str, default: &str) -> PathBuf {
    std::env::var_os(env)
        .filter(|v| !v.is_empty())
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(OsString::from(default)))
}

pub fn ffmpeg_bin() -> PathBuf {
    bin("FV_FFMPEG", "ffmpeg")
}

pub fn ffprobe_bin() -> PathBuf {
    bin("FV_FFPROBE", "ffprobe")
}

fn runs(path: PathBuf) -> bool {
    Command::new(path)
        .arg("-version")
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

pub fn ffmpeg_available() -> bool {
    runs(ffmpeg_bin())
}

pub fn ffprobe_available() -> bool {
    runs(ffprobe_bin())
}

/// A quiet ffmpeg command (`-hide_banner -loglevel error -nostats -y`).
pub fn ffmpeg_command() -> Command {
    let mut c = Command::new(ffmpeg_bin());
    c.args(["-hide_banner", "-loglevel", "error", "-nostats", "-y"]);
    c
}

/// Run a command to completion; a non-zero exit becomes a `Tool` error that
/// carries the stderr tail.
pub fn run_checked(mut cmd: Command, tool: &str) -> Result<std::process::Output> {
    let out = cmd
        .stdin(Stdio::null())
        .output()
        .map_err(|e| MediaError::tool(tool, format!("not available: {e}")))?;
    if !out.status.success() {
        let err = String::from_utf8_lossy(&out.stderr);
        let tail: String = err.lines().rev().take(8).collect::<Vec<_>>().into_iter().rev().collect::<Vec<_>>().join("\n");
        return Err(MediaError::tool(tool, format!("exited with {}: {tail}", out.status)));
    }
    Ok(out)
}
