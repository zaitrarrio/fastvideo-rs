//! Inter-frame VP8 through ffmpeg's `libvpx` encoder (design §5.9).
//!
//! VP8 serves WebRTC peers whose stack offers no H.264 (the Python and C++
//! Reactor SDKs' libwebrtc, open-source Chromium). The encoder is an ffmpeg
//! subprocess, like the NVENC path in [`crate::video`]: rgb24 on stdin, IVF
//! on stdout, one VP8 frame per IVF record. Settings are libvpx's real-time
//! profile: `-deadline realtime`, CBR at the target bitrate, no alt-ref and
//! no lag (one frame out per frame in), error resilient, a keyframe every
//! `gop_seconds`.
//!
//! [`Vp8Encoder::encode`] does not wait for its frame: frames come back in
//! order with a later `encode` or [`Vp8Encoder::poll`] (a few ms of pipe
//! latency; waiting per frame would cap throughput at 1/latency).
//!
//! Forced keyframes (PLI/FIR, a new peer) restart the ffmpeg process: the
//! first frame of a new process is a keyframe. Callers rate-limit requests
//! (`IdrLimiter` / `fastvideo_webrtc::writer::KeyframeLimiter`). The old
//! process gets EOF and flushes the frame libvpx still holds; a streaming
//! session's warm spare ([`crate::pipe`]) takes over without an ffmpeg start.
//!
//! [`libvpx_available`] probes once per process whether ffmpeg can encode
//! `libvpx` here; the serve image's ffmpeg has it (docker/gpucheck.Dockerfile
//! checks at build time).

use std::process::Stdio;

use bytes::Bytes;

use crate::av::{AvCheck, RgbFrame};
use crate::error::{MediaError, Result};
use crate::pipe::{PipeProcs, RestartStats, SparePool, SpareSpec};
use crate::tools;
use crate::video::EncodedFrame;

/// VP8 encoder settings.
#[derive(Debug, Clone, PartialEq)]
pub struct Vp8Config {
    pub width: u32,
    pub height: u32,
    pub fps: u32,
    /// CBR target.
    pub bitrate_bps: u32,
    /// Seconds between periodic keyframes.
    pub gop_seconds: f32,
    /// libvpx threads (0: ffmpeg's default).
    pub threads: u16,
    /// libvpx real-time speed, 0..=16 (higher is faster and larger).
    pub cpu_used: u8,
}

impl Vp8Config {
    /// Real-time defaults: [`crate::video::default_bitrate`] for the canvas,
    /// a keyframe every 2 s, speed 8, one thread (1344x768 at ~80 fps on one
    /// core; libvpx's VP8 worker threads spin-wait, which measured 25x
    /// slower than one thread on a CPU-contended 4-core host).
    pub fn new(width: u32, height: u32, fps: u32) -> Self {
        Self {
            width,
            height,
            fps,
            bitrate_bps: crate::video::default_bitrate(width, height),
            gop_seconds: 2.0,
            threads: 1,
            cpu_used: 8,
        }
    }

    pub fn gop_frames(&self) -> u32 {
        ((self.gop_seconds * self.fps as f32).round() as u32).max(1)
    }

    pub fn validate(&self) -> Result<()> {
        if self.width == 0 || self.height == 0 || self.width > 16383 || self.height > 16383 {
            return Err(MediaError::invalid(format!("VP8 cannot carry {}x{}", self.width, self.height)));
        }
        if self.fps == 0 || self.fps > 240 {
            return Err(MediaError::invalid(format!("fps {} out of range", self.fps)));
        }
        if self.bitrate_bps < 50_000 {
            return Err(MediaError::invalid(format!("bitrate {} b/s is too low", self.bitrate_bps)));
        }
        if self.cpu_used > 16 {
            return Err(MediaError::invalid("cpu_used must be 0..=16"));
        }
        Ok(())
    }
}

/// The ffmpeg arguments (after the common quiet flags) of the VP8 encoder.
pub fn ffmpeg_args(cfg: &Vp8Config) -> Vec<String> {
    let b = cfg.bitrate_bps.to_string();
    let g = cfg.gop_frames().to_string();
    let mut a: Vec<String> = ["-f", "rawvideo", "-pix_fmt", "rgb24", "-s"].map(String::from).to_vec();
    a.push(format!("{}x{}", cfg.width, cfg.height));
    a.extend(["-r".into(), cfg.fps.to_string(), "-i".into(), "pipe:0".into(), "-an".into()]);
    let v: &[&str] = &[
        "-c:v", "libvpx", "-deadline", "realtime", "-error-resilient", "1", "-lag-in-frames", "0", "-auto-alt-ref", "0",
        "-pix_fmt", "yuv420p", "-qmin", "4", "-qmax", "56", "-undershoot-pct", "95", "-overshoot-pct", "15",
    ];
    a.extend(v.iter().map(|s| s.to_string()));
    a.extend(["-cpu-used".into(), cfg.cpu_used.to_string()]);
    a.extend(["-b:v".into(), b.clone(), "-minrate".into(), b.clone(), "-maxrate".into(), b]);
    // One second of buffer (libvpx's `rc_buf_sz` default is 6 s, too slow
    // to react for live video).
    a.extend(["-bufsize".into(), cfg.bitrate_bps.to_string()]);
    a.extend(["-g".into(), g.clone(), "-keyint_min".into(), g]);
    if cfg.threads > 0 {
        a.extend(["-threads".into(), cfg.threads.to_string()]);
    }
    a.extend(["-flush_packets", "1", "-f", "ivf", "pipe:1"].map(String::from));
    a
}

/// Whether a VP8 frame is a key frame (RFC 6386 §9.1: bit 0 clear).
pub fn is_keyframe(frame: &[u8]) -> bool {
    frame.first().is_some_and(|b| b & 1 == 0)
}

/// Pops the complete VP8 frames from an IVF byte stream. `header_done`
/// tracks whether the 32-byte file header was consumed.
pub fn take_ivf_frames(buf: &mut Vec<u8>, header_done: &mut bool) -> Result<Vec<Vec<u8>>> {
    let mut out = Vec::new();
    let mut at = 0usize;
    if !*header_done {
        if buf.len() < 32 {
            return Ok(out);
        }
        if &buf[0..4] != b"DKIF" {
            return Err(MediaError::Parse { path: None, message: "ffmpeg output is not IVF".into() });
        }
        let hlen = u16::from_le_bytes([buf[6], buf[7]]) as usize;
        if hlen < 32 {
            return Err(MediaError::Parse { path: None, message: format!("IVF header length {hlen}") });
        }
        if buf.len() < hlen {
            return Ok(out);
        }
        at = hlen;
        *header_done = true;
    }
    while buf.len() >= at + 12 {
        let len = u32::from_le_bytes([buf[at], buf[at + 1], buf[at + 2], buf[at + 3]]) as usize;
        if buf.len() < at + 12 + len {
            break;
        }
        out.push(buf[at + 12..at + 12 + len].to_vec());
        at += 12 + len;
    }
    buf.drain(..at);
    Ok(out)
}

/// Inter-frame VP8 through an ffmpeg `libvpx` subprocess.
pub struct Vp8Encoder {
    cfg: Vp8Config,
    procs: PipeProcs,
    out_index: u64,
    key_pending: bool,
}

impl std::fmt::Debug for Vp8Encoder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Vp8Encoder").field("cfg", &self.cfg).field("restarts", &self.restarts()).finish()
    }
}

impl Vp8Encoder {
    /// A standalone encoder: no spare.
    pub fn new(cfg: Vp8Config) -> Result<Self> {
        Self::with_pool(cfg, None)
    }

    /// A streaming session's encoder: starts from `pool`'s warm spare when
    /// it holds this profile, and restarts into it ([`crate::pipe`]).
    pub fn with_pool(cfg: Vp8Config, pool: Option<SparePool>) -> Result<Self> {
        let procs = PipeProcs::new(SpareSpec::vp8(&cfg)?, pool)?;
        Ok(Self { cfg, procs, out_index: 0, key_pending: false })
    }

    pub fn config(&self) -> &Vp8Config {
        &self.cfg
    }

    /// ffmpeg restarts done to honour forced keyframes.
    pub fn restarts(&self) -> u64 {
        self.procs.stats().restarts
    }

    /// Restart counts by spare use and their latencies.
    pub fn restart_stats(&self) -> &RestartStats {
        self.procs.stats()
    }

    /// ffmpeg ids: current, then flushing ones.
    pub fn pids(&self) -> Vec<u32> {
        self.procs.pids()
    }

    /// Make the next encoded frame a keyframe.
    pub fn force_keyframe(&mut self) {
        self.key_pending = true;
    }

    fn wrap(&mut self, frames: impl IntoIterator<Item = Vec<u8>>) -> Vec<EncodedFrame> {
        frames
            .into_iter()
            .map(|f| {
                let e = EncodedFrame { keyframe: is_keyframe(&f), data: Bytes::from(f), index: self.out_index };
                self.out_index += 1;
                e
            })
            .collect()
    }

    /// Encode one frame; returns the frames that became ready, in order.
    /// It does not wait: ffmpeg hands a frame back a few ms later, so the
    /// frame usually comes back with the next call or [`Self::poll`].
    pub fn encode(&mut self, frame: &RgbFrame) -> Result<Vec<EncodedFrame>> {
        frame.check()?;
        if frame.width != self.cfg.width || frame.height != self.cfg.height {
            return Err(MediaError::invalid(format!(
                "frame {}x{} does not match the encoder {}x{}",
                frame.width, frame.height, self.cfg.width, self.cfg.height
            )));
        }
        if self.key_pending {
            // A fresh process starts with a keyframe.
            self.key_pending = false;
            self.procs.restart()?;
        }
        self.procs.write(&frame.data)?;
        self.poll()
    }

    /// The frames that became ready since the last call, in order.
    pub fn poll(&mut self) -> Result<Vec<EncodedFrame>> {
        let raw = self.procs.ready()?;
        Ok(self.wrap(raw))
    }

    /// Flush everything still inside the encoder.
    pub fn finish(&mut self) -> Result<Vec<EncodedFrame>> {
        let frames = self.procs.finish()?;
        Ok(self.wrap(frames))
    }
}

/// One `libvpx` encode probe: `Err` says why (ffmpeg missing, or built
/// without libvpx).
pub fn libvpx_probe() -> std::result::Result<(), String> {
    let ffmpeg = tools::ffmpeg_bin();
    let out = std::process::Command::new(&ffmpeg)
        .args(["-hide_banner", "-loglevel", "error", "-nostats"])
        .args(["-f", "lavfi", "-i", "color=c=black:s=64x48:r=24:d=0.1", "-c:v", "libvpx", "-f", "null", "-"])
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .output()
        .map_err(|e| format!("cannot run {}: {e}", ffmpeg.display()))?;
    if out.status.success() {
        return Ok(());
    }
    let err = String::from_utf8_lossy(&out.stderr);
    let tail: Vec<&str> = err.lines().map(str::trim).filter(|l| !l.is_empty()).collect();
    Err(tail.last().map_or_else(|| format!("ffmpeg libvpx exited with {}", out.status), |l| l.to_string()))
}

/// Whether ffmpeg can encode `libvpx` here (probed once per process).
pub fn libvpx_available() -> bool {
    static PROBE: std::sync::OnceLock<std::result::Result<(), String>> = std::sync::OnceLock::new();
    let r = PROBE.get_or_init(|| {
        let r = libvpx_probe();
        if let Err(e) = &r {
            tracing::info!(error = %e, "ffmpeg libvpx unavailable: VP8 peers get intra-only frames");
        }
        r
    });
    r.is_ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ivf_frames_are_split_across_reads() {
        let mut ivf = b"DKIF\0\0\x20\0VP80".to_vec();
        ivf.resize(32, 0);
        for (i, f) in [&[0x10u8, 1, 2][..], &[0x11, 9]].iter().enumerate() {
            ivf.extend_from_slice(&(f.len() as u32).to_le_bytes());
            ivf.extend_from_slice(&(i as u64).to_le_bytes());
            ivf.extend_from_slice(f);
        }
        let mut buf = Vec::new();
        let mut done = false;
        let mut got = Vec::new();
        for b in ivf.chunks(5) {
            buf.extend_from_slice(b);
            got.extend(take_ivf_frames(&mut buf, &mut done).unwrap());
        }
        assert_eq!(got, vec![vec![0x10, 1, 2], vec![0x11, 9]]);
        assert!(buf.is_empty());
        assert!(is_keyframe(&got[0]) && !is_keyframe(&got[1]));
        let mut bad = b"RIFF".to_vec();
        bad.resize(40, 0);
        assert!(take_ivf_frames(&mut bad, &mut false).is_err());
    }

    #[test]
    fn realtime_cbr_args() {
        let c = Vp8Config::new(1344, 768, 24);
        c.validate().unwrap();
        let a = ffmpeg_args(&c).join(" ");
        for want in ["-c:v libvpx", "-deadline realtime", "-lag-in-frames 0", "-b:v 6000000", "-g 48", "-f ivf pipe:1"] {
            assert!(a.contains(want), "{want} missing from {a}");
        }
        assert!(Vp8Config { bitrate_bps: 10, ..c.clone() }.validate().is_err());
        assert!(Vp8Config { width: 0, ..c }.validate().is_err());
    }
}
