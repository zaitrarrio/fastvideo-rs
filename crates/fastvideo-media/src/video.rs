//! H.264 encoding behind one trait (design §0 decisions 1-2, §5.9).
//!
//! | Backend | `encoder =` | Where | Notes |
//! |---|---|---|---|
//! | **NVENC** (production) | `nvenc` | always built; needs ffmpeg with `h264_nvenc`, an NVIDIA GPU and the `video` driver capability at run time | ffmpeg subprocess: rgb24 on stdin, scaler in the filter graph, Annex-B with AUDs on stdout. Constrained Baseline, CBR, no B-frames, IDR every `gop_seconds`, no scene-cut IDRs, zero-latency |
//! | OpenH264 | `openh264` | `openh264` feature; **CPU-only tests/CI, never deployed** (no patent licence for a source build) | In process; scaler in Rust |
//!
//! `auto` (fv-serve's default for every encoder setting) resolves once per
//! process through [`auto_encoder`]: NVENC when an NVENC encode probe
//! succeeds, else OpenH264.
//!
//! x264 is not a backend (owner decision). The ffmpeg pipe encoder can run
//! libx264 only as [`FfmpegH264::Libx264CpuTest`], which exists so the
//! ffmpeg plumbing is testable on GPU-less CI; nothing selects it by default
//! and it is not reachable from config.
//!
//! The encoder config carries the **publish profile** (decision 2): Cloudflare
//! WHIP scales H3 streams to 1280×720 at level 3.1; MediaMTX and peer WebRTC
//! keep the native canvas at level 4.0. See [`H264Config::for_publish`].
//!
//! Forced IDRs (PLI/FIR, rate-limited by [`IdrLimiter`]) are native in
//! OpenH264. Through the ffmpeg pipe there is no per-frame control, so the
//! pipe encoder restarts its ffmpeg process: the first frame of a new process
//! is an IDR with SPS/PPS. Frame indices continue across the restart.

use std::io::{Read, Write};
use std::process::{Child, ChildStdin, Stdio};
use std::sync::mpsc;

use bytes::Bytes;

use crate::av::{AvCheck, RgbFrame};
use crate::error::{MediaError, Result};
use crate::h264::{self, H264Level};
use crate::scale::{self, ScaleMode};
use crate::tools;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum EncoderBackend {
    /// NVIDIA NVENC through ffmpeg `h264_nvenc` (production).
    Nvenc,
    /// OpenH264 in process (CPU test/CI only).
    #[serde(rename = "openh264")]
    OpenH264,
    /// ffmpeg libx264, CPU test plumbing only; not configurable.
    #[serde(skip)]
    CpuTestX264,
}

impl std::str::FromStr for EncoderBackend {
    type Err = MediaError;
    fn from_str(s: &str) -> Result<Self> {
        match s {
            "nvenc" => Ok(Self::Nvenc),
            "openh264" => Ok(Self::OpenH264),
            other => Err(MediaError::invalid(format!("unknown encoder {other:?} (nvenc | openh264)"))),
        }
    }
}

impl EncoderBackend {
    /// Whether this build can construct the backend (NVENC still needs a GPU
    /// and ffmpeg with `h264_nvenc` at run time; see [`nvenc_available`]).
    pub fn compiled(self) -> bool {
        match self {
            Self::Nvenc | Self::CpuTestX264 => true,
            Self::OpenH264 => cfg!(feature = "openh264"),
        }
    }
}

/// Default bitrate by encoded canvas (§5.1): 6 Mb/s at 720p and up, 2.5 Mb/s at 480p.
pub fn default_bitrate(width: u32, height: u32) -> u32 {
    if width.min(height) >= 720 { 6_000_000 } else { 2_500_000 }
}

/// Where a stream is published; selects canvas cap and level (decision 2).
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PublishTarget {
    /// WHIP to Cloudflare Stream: at most 1280×720, level 3.1.
    Cloudflare,
    /// WHIP to a self-hosted MediaMTX: native canvas, level 4.0.
    Mediamtx,
    /// Peer WebRTC (Reactor, fal director): native canvas, level 4.0.
    Peer,
}

impl PublishTarget {
    /// `(canvas cap, level)`; `None` cap means native.
    pub fn policy(self) -> (Option<(u32, u32)>, H264Level) {
        match self {
            PublishTarget::Cloudflare => (Some((1280, 720)), H264Level::L3_1),
            PublishTarget::Mediamtx | PublishTarget::Peer => (None, H264Level::L4_0),
        }
    }
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct H264Config {
    /// Frames handed to `encode` (the model canvas).
    pub input_width: u32,
    pub input_height: u32,
    /// Encoded canvas (after the scaler).
    pub width: u32,
    pub height: u32,
    pub scale: ScaleMode,
    pub fps: u32,
    pub bitrate_bps: u32,
    /// Seconds between IDRs (2 s for WHIP/browsers, 1 s for HLS).
    pub gop_seconds: f32,
    /// `None` picks the smallest level that fits.
    pub level: Option<H264Level>,
    /// Encoder threads (OpenH264; 0 = default).
    pub threads: u16,
}

impl H264Config {
    /// Native canvas, no scaling, automatic level.
    pub fn new(width: u32, height: u32, fps: u32) -> Self {
        Self {
            input_width: width,
            input_height: height,
            width,
            height,
            scale: ScaleMode::Fit,
            fps,
            bitrate_bps: default_bitrate(width, height),
            gop_seconds: 2.0,
            level: None,
            threads: 0,
        }
    }

    /// The config for publishing a `w×h` model canvas to `target`:
    /// Cloudflare → capped at 1280×720 (orientation kept), level 3.1;
    /// MediaMTX / peer → native, level 4.0.
    pub fn for_publish(target: PublishTarget, width: u32, height: u32, fps: u32) -> Self {
        let (cap, level) = target.policy();
        let (w, h) = cap.map(|c| scale::capped_canvas(width, height, c)).unwrap_or((width, height));
        let mut c = Self::new(width, height, fps);
        c.width = w;
        c.height = h;
        c.bitrate_bps = default_bitrate(w, h);
        c.level = Some(level);
        c
    }

    pub fn scales(&self) -> bool {
        (self.input_width, self.input_height) != (self.width, self.height)
    }

    pub fn gop_frames(&self) -> u32 {
        ((self.fps as f32 * self.gop_seconds).round() as u32).max(1)
    }

    pub fn validate(&self) -> Result<()> {
        for (w, h) in [(self.width, self.height), (self.input_width, self.input_height)] {
            if w == 0 || h == 0 || w % 2 != 0 || h % 2 != 0 {
                return Err(MediaError::invalid(format!("H.264 4:2:0 needs even dimensions, got {w}x{h}")));
            }
        }
        if self.fps == 0 {
            return Err(MediaError::invalid("fps must be positive"));
        }
        if self.bitrate_bps == 0 {
            return Err(MediaError::invalid("bitrate must be positive"));
        }
        self.resolved_level().map(|_| ())
    }

    /// The configured level, or the minimum that fits. A configured level
    /// too small for the encoded canvas is an error, not a non-conforming stream.
    pub fn resolved_level(&self) -> Result<H264Level> {
        let min = H264Level::min_for(self.width, self.height, self.fps).ok_or_else(|| {
            MediaError::invalid(format!("{}x{}@{} exceeds H.264 level 5.2", self.width, self.height, self.fps))
        })?;
        match self.level {
            None => Ok(min),
            Some(l) if l >= min => Ok(l),
            Some(l) => Err(MediaError::invalid(format!(
                "H.264 level {l} cannot carry {}x{}@{}; the minimum is {min}",
                self.width, self.height, self.fps
            ))),
        }
    }

    /// SDP `profile-level-id` (Constrained Baseline at the resolved level).
    pub fn profile_level_id(&self) -> Result<String> {
        Ok(self.resolved_level()?.cb_profile_level_id())
    }
}

/// One encoded access unit (Annex-B, start codes included).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EncodedFrame {
    pub data: Bytes,
    pub keyframe: bool,
    /// Output order index (no B-frames, so also presentation order).
    pub index: u64,
}

pub trait VideoEncoder: Send {
    fn backend(&self) -> EncoderBackend;
    fn config(&self) -> &H264Config;
    /// Encode one input-canvas frame; returns the access units that became
    /// ready (zero or more: the pipe encoder lags by about one frame).
    fn encode(&mut self, frame: &RgbFrame) -> Result<Vec<EncodedFrame>>;
    /// Make the next encoded frame an IDR (PLI/FIR, or after an input drop).
    fn force_idr(&mut self);
    /// Flush everything still inside the encoder.
    fn finish(&mut self) -> Result<Vec<EncodedFrame>>;
}

/// Construct a backend.
pub fn create_encoder(backend: EncoderBackend, cfg: H264Config) -> Result<Box<dyn VideoEncoder>> {
    cfg.validate()?;
    match backend {
        EncoderBackend::Nvenc => Ok(Box::new(PipeEncoder::new(cfg, FfmpegH264::Nvenc)?)),
        EncoderBackend::CpuTestX264 => Ok(Box::new(PipeEncoder::new(cfg, FfmpegH264::Libx264CpuTest)?)),
        #[cfg(feature = "openh264")]
        EncoderBackend::OpenH264 => Ok(Box::new(openh264_backend::OpenH264Encoder::new(cfg)?)),
        #[cfg(not(feature = "openh264"))]
        EncoderBackend::OpenH264 => {
            Err(MediaError::Unsupported("built without the `openh264` feature of fastvideo-media".into()))
        }
    }
}

/// Rate limit for forced IDRs: at most one per `min_interval` seconds (§5.1).
#[derive(Debug, Clone)]
pub struct IdrLimiter {
    min_interval: f64,
    last: Option<f64>,
    pending: bool,
}

impl IdrLimiter {
    pub fn new(min_interval: f64) -> Self {
        Self { min_interval, last: None, pending: false }
    }

    /// Record a request (PLI/FIR from any peer).
    pub fn request(&mut self) {
        self.pending = true;
    }

    /// Whether to force an IDR now; clears the pending request when it fires.
    pub fn poll(&mut self, now: f64) -> bool {
        if !self.pending {
            return false;
        }
        if let Some(t) = self.last {
            if now - t < self.min_interval {
                return false;
            }
        }
        self.pending = false;
        self.last = Some(now);
        true
    }
}

// ---------------------------------------------------------------------------
// ffmpeg H.264 encoder arguments (shared by the pipe encoder, sinks, mp4)
// ---------------------------------------------------------------------------

/// The H.264 encoder ffmpeg runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FfmpegH264 {
    /// `h264_nvenc` (production).
    #[default]
    Nvenc,
    /// `libx264`: CPU-only tests of the ffmpeg plumbing. Never deployed.
    #[serde(skip)]
    Libx264CpuTest,
}

impl FfmpegH264 {
    /// Real-time streaming arguments: Constrained Baseline, CBR, fixed GOP,
    /// no B-frames, no scene-cut IDRs, zero latency, AUD NALs, and SPS/PPS
    /// repeated at every IDR (raw Annex-B output has no global header).
    pub fn stream_args(self, cfg: &H264Config) -> Result<Vec<String>> {
        let lvl = cfg.resolved_level()?;
        let g = cfg.gop_frames().to_string();
        let b = cfg.bitrate_bps.to_string();
        let s = |v: &[&str]| v.iter().map(|x| x.to_string()).collect::<Vec<_>>();
        let mut a = match self {
            FfmpegH264::Nvenc => {
                let mut a = s(&["-c:v", "h264_nvenc", "-preset", "p4", "-tune", "ll", "-profile:v", "baseline"]);
                a.extend(s(&["-level:v", &lvl.to_string(), "-rc", "cbr", "-b:v", &b, "-maxrate", &b]));
                a.extend(["-bufsize".into(), (u64::from(cfg.bitrate_bps) / 2).to_string()]);
                a.extend(s(&["-g", &g, "-bf", "0", "-forced-idr", "1", "-no-scenecut", "1", "-strict_gop", "1"]));
                a.extend(s(&["-zerolatency", "1", "-rc-lookahead", "0", "-delay", "0", "-aud", "1"]));
                a
            }
            FfmpegH264::Libx264CpuTest => {
                let mut a = s(&["-c:v", "libx264", "-preset", "veryfast", "-tune", "zerolatency", "-profile:v", "baseline"]);
                a.extend(s(&["-level:v", &lvl.to_string(), "-g", &g, "-keyint_min", &g, "-sc_threshold", "0"]));
                a.extend(s(&["-b:v", &b, "-maxrate", &b]));
                a.extend(["-bufsize".into(), (u64::from(cfg.bitrate_bps) / 2).to_string()]);
                a.extend(s(&["-x264-params", "aud=1:repeat-headers=1"]));
                a
            }
        };
        a.extend(s(&["-pix_fmt", "yuv420p"]));
        Ok(a)
    }

    /// File (batch MP4) arguments at a constant-quality target.
    pub fn file_args(self, quality: u8) -> Vec<String> {
        let q = quality.to_string();
        let v: Vec<&str> = match self {
            FfmpegH264::Nvenc => vec![
                "-c:v", "h264_nvenc", "-preset", "p5", "-tune", "hq", "-profile:v", "high", "-rc", "vbr", "-cq", &q, "-b:v",
                "0", "-bf", "0", "-pix_fmt", "yuv420p",
            ],
            FfmpegH264::Libx264CpuTest => vec!["-c:v", "libx264", "-preset", "veryfast", "-crf", &q, "-pix_fmt", "yuv420p"],
        };
        v.into_iter().map(String::from).collect()
    }
}

/// Whether ffmpeg can open `h264_nvenc` here (GPU, driver `video`
/// capability, and an ffmpeg built with nvenc). Encodes a few black frames.
pub fn nvenc_available() -> bool {
    nvenc_probe().is_ok()
}

/// One NVENC encode probe: `Err` says why (ffmpeg missing, or the tail of
/// ffmpeg's error output).
pub fn nvenc_probe() -> std::result::Result<(), String> {
    nvenc_probe_with(&tools::ffmpeg_bin())
}

fn nvenc_probe_with(ffmpeg: &std::path::Path) -> std::result::Result<(), String> {
    let mut c = std::process::Command::new(ffmpeg);
    c.args(["-hide_banner", "-loglevel", "error", "-nostats"])
        .args(["-f", "lavfi", "-i", "color=c=black:s=256x144:r=24:d=0.2", "-c:v", "h264_nvenc", "-f", "null", "-"])
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::piped());
    let out = c.output().map_err(|e| format!("cannot run {}: {e}", ffmpeg.display()))?;
    if out.status.success() {
        return Ok(());
    }
    let err = String::from_utf8_lossy(&out.stderr);
    let lines: Vec<&str> = err.lines().map(str::trim).filter(|l| !l.is_empty()).collect();
    let tail = lines[lines.len().saturating_sub(3)..].join(" | ");
    Err(if tail.is_empty() { format!("ffmpeg h264_nvenc exited with {}", out.status) } else { tail })
}

/// What an `auto` encoder setting resolves to (see [`auto_encoder`]).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AutoEncoder {
    /// NVENC when the probe encoded, else OpenH264.
    pub backend: EncoderBackend,
    /// Why NVENC was not chosen (the last probe's error).
    pub nvenc_error: Option<String>,
    /// Probes run (1, or 2 after a retry).
    pub attempts: u32,
}

impl AutoEncoder {
    /// From one probe outcome.
    pub fn from_probe(probe: std::result::Result<(), String>) -> Self {
        match probe {
            Ok(()) => Self { backend: EncoderBackend::Nvenc, nvenc_error: None, attempts: 1 },
            Err(e) => Self { backend: EncoderBackend::OpenH264, nvenc_error: Some(e), attempts: 1 },
        }
    }

    /// Probes, and when the probe fails in a way that can be transient
    /// (ffmpeg ran and has `h264_nvenc`, but opening the encoder failed:
    /// a serverless GPU whose driver or NVENC sessions are not ready yet),
    /// waits `delay` and probes once more.
    pub fn probe_with_retry(mut probe: impl FnMut() -> std::result::Result<(), String>, delay: std::time::Duration) -> Self {
        let first = probe();
        match first {
            Err(e) if probe_error_is_transient(&e) => {
                std::thread::sleep(delay);
                Self { attempts: 2, ..Self::from_probe(probe()) }
            }
            other => Self::from_probe(other),
        }
    }
}

/// Whether a probe error is worth a retry: not a missing ffmpeg and not an
/// ffmpeg built without `h264_nvenc`, which a retry cannot change.
pub fn probe_error_is_transient(e: &str) -> bool {
    !(e.starts_with("cannot run ") || e.contains("Unknown encoder"))
}

/// Resolves `encoder = "auto"`: NVENC when an NVENC encode probe succeeds
/// (retried once after 2 s when the failure may be transient), else
/// OpenH264. The probe runs once per process (it spawns ffmpeg), so every
/// `auto` setting and every later session agree.
pub fn auto_encoder() -> &'static AutoEncoder {
    AUTO.get_or_init(|| AutoEncoder::probe_with_retry(nvenc_probe, std::time::Duration::from_secs(2)))
}

/// The [`auto_encoder`] outcome when the probe already ran (no probing).
pub fn auto_encoder_if_probed() -> Option<&'static AutoEncoder> {
    AUTO.get()
}

static AUTO: std::sync::OnceLock<AutoEncoder> = std::sync::OnceLock::new();

// ---------------------------------------------------------------------------
// The ffmpeg pipe encoder (NVENC in production)
// ---------------------------------------------------------------------------

struct Proc {
    child: Child,
    stdin: Option<ChildStdin>,
    rx: mpsc::Receiver<Vec<u8>>,
    reader: Option<std::thread::JoinHandle<()>>,
}

impl Proc {
    fn spawn(cfg: &H264Config, codec: FfmpegH264) -> Result<Self> {
        let mut cmd = tools::ffmpeg_command();
        cmd.args(pipe_encoder_args(cfg, codec)?).stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::null());
        let mut child = cmd.spawn().map_err(|e| MediaError::tool("ffmpeg", format!("not available: {e}")))?;
        let stdin = child.stdin.take();
        let mut stdout = child.stdout.take().ok_or_else(|| MediaError::tool("ffmpeg", "no stdout"))?;
        let (tx, rx) = mpsc::channel();
        let reader = std::thread::Builder::new().name("h264-ffmpeg-out".into()).spawn(move || {
            let mut buf = Vec::new();
            let mut chunk = vec![0u8; 1 << 16];
            loop {
                match stdout.read(&mut chunk) {
                    Ok(0) | Err(_) => break,
                    Ok(n) => {
                        buf.extend_from_slice(&chunk[..n]);
                        for au in take_complete_aus(&mut buf) {
                            if tx.send(au).is_err() {
                                return;
                            }
                        }
                    }
                }
            }
            if !buf.is_empty() {
                let _ = tx.send(buf);
            }
        })?;
        Ok(Self { child, stdin, rx, reader: Some(reader) })
    }

    /// Close stdin, drain every AU, reap the process.
    fn close(mut self) -> Result<Vec<Vec<u8>>> {
        drop(self.stdin.take());
        let aus: Vec<Vec<u8>> = self.rx.iter().collect();
        if let Some(r) = self.reader.take() {
            let _ = r.join();
        }
        let status = self.child.wait()?;
        if !status.success() {
            return Err(MediaError::tool("ffmpeg", format!("h264 encode exited with {status}")));
        }
        Ok(aus)
    }
}

impl Drop for Proc {
    fn drop(&mut self) {
        drop(self.stdin.take());
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// The full ffmpeg argument list (after the common flags) of the pipe encoder.
pub fn pipe_encoder_args(cfg: &H264Config, codec: FfmpegH264) -> Result<Vec<String>> {
    let mut a: Vec<String> = ["-f", "rawvideo", "-pix_fmt", "rgb24", "-s"].map(String::from).to_vec();
    a.push(format!("{}x{}", cfg.input_width, cfg.input_height));
    a.extend(["-r".into(), cfg.fps.to_string(), "-i".into(), "pipe:0".into(), "-an".into()]);
    if let Some(vf) = scale::ffmpeg_filter(cfg.input_width, cfg.input_height, cfg.width, cfg.height, cfg.scale) {
        a.extend(["-vf".into(), vf]);
    }
    a.extend(codec.stream_args(cfg)?);
    a.extend(["-f", "h264", "pipe:1"].map(String::from));
    Ok(a)
}

/// H.264 through an ffmpeg subprocess: rgb24 in, Annex-B access units out.
pub struct PipeEncoder {
    cfg: H264Config,
    codec: FfmpegH264,
    proc: Option<Proc>,
    out_index: u64,
    idr_pending: bool,
    restarts: u64,
}

impl PipeEncoder {
    pub fn new(cfg: H264Config, codec: FfmpegH264) -> Result<Self> {
        cfg.validate()?;
        let proc = Proc::spawn(&cfg, codec)?;
        Ok(Self { cfg, codec, proc: Some(proc), out_index: 0, idr_pending: false, restarts: 0 })
    }

    /// ffmpeg restarts done to honour forced IDRs.
    pub fn restarts(&self) -> u64 {
        self.restarts
    }

    fn wrap(&mut self, aus: impl IntoIterator<Item = Vec<u8>>) -> Vec<EncodedFrame> {
        aus.into_iter()
            .map(|au| {
                let f = EncodedFrame { keyframe: h264::is_idr(&au), data: Bytes::from(au), index: self.out_index };
                self.out_index += 1;
                f
            })
            .collect()
    }

    fn ready(&mut self) -> Vec<EncodedFrame> {
        let aus: Vec<Vec<u8>> = match &self.proc {
            Some(p) => p.rx.try_iter().collect(),
            None => Vec::new(),
        };
        self.wrap(aus)
    }
}

/// Pop every access unit that is followed by the next AUD from `buf`.
fn take_complete_aus(buf: &mut Vec<u8>) -> Vec<Vec<u8>> {
    let mut starts = Vec::new();
    let mut i = 0;
    while i + 4 <= buf.len() {
        if buf[i] == 0 && buf[i + 1] == 0 && buf[i + 2] == 1 && buf[i + 3] & 0x1f == h264::nal::AUD {
            // Include the leading zero of a 4-byte start code.
            let s = if i > 0 && buf[i - 1] == 0 { i - 1 } else { i };
            starts.push(s);
            i += 4;
        } else {
            i += 1;
        }
    }
    let mut out = Vec::new();
    if starts.len() < 2 {
        return out;
    }
    let last = *starts.last().expect("len >= 2");
    for w in starts.windows(2) {
        out.push(buf[w[0]..w[1]].to_vec());
    }
    buf.drain(..last);
    out
}

impl VideoEncoder for PipeEncoder {
    fn backend(&self) -> EncoderBackend {
        match self.codec {
            FfmpegH264::Nvenc => EncoderBackend::Nvenc,
            FfmpegH264::Libx264CpuTest => EncoderBackend::CpuTestX264,
        }
    }

    fn config(&self) -> &H264Config {
        &self.cfg
    }

    fn encode(&mut self, frame: &RgbFrame) -> Result<Vec<EncodedFrame>> {
        frame.check()?;
        if frame.width != self.cfg.input_width || frame.height != self.cfg.input_height {
            return Err(MediaError::invalid(format!(
                "frame {}x{} does not match the encoder input {}x{}",
                frame.width, frame.height, self.cfg.input_width, self.cfg.input_height
            )));
        }
        let mut out = Vec::new();
        if self.idr_pending {
            // A fresh process starts with an IDR (plus SPS/PPS).
            self.idr_pending = false;
            if let Some(p) = self.proc.take() {
                let aus = p.close()?;
                out.extend(self.wrap(aus));
            }
            self.proc = Some(Proc::spawn(&self.cfg, self.codec)?);
            self.restarts += 1;
        }
        let p = self.proc.as_mut().ok_or_else(|| MediaError::Encode("encoder finished".into()))?;
        let stdin = p.stdin.as_mut().ok_or_else(|| MediaError::Encode("encoder finished".into()))?;
        stdin.write_all(&frame.data).map_err(|e| MediaError::tool("ffmpeg", format!("stdin: {e}")))?;
        out.extend(self.ready());
        Ok(out)
    }

    fn force_idr(&mut self) {
        self.idr_pending = true;
    }

    fn finish(&mut self) -> Result<Vec<EncodedFrame>> {
        match self.proc.take() {
            Some(p) => {
                let aus = p.close()?;
                Ok(self.wrap(aus))
            }
            None => Ok(Vec::new()),
        }
    }
}

// ---------------------------------------------------------------------------
// OpenH264 (CPU test/CI backend)
// ---------------------------------------------------------------------------

#[cfg(feature = "openh264")]
pub mod openh264_backend {
    use super::*;
    use openh264::encoder::{
        BitRate, Complexity, Encoder, EncoderConfig, FrameRate, FrameType, IntraFramePeriod, Level, Profile,
        RateControlMode, SpsPpsStrategy, UsageType,
    };
    use openh264::formats::{RgbSliceU8, YUVBuffer, YUVSource};
    use openh264::{OpenH264API, Timestamp};

    fn level(l: H264Level) -> Level {
        match l {
            H264Level::L3_0 => Level::Level_3_0,
            H264Level::L3_1 => Level::Level_3_1,
            H264Level::L3_2 => Level::Level_3_2,
            H264Level::L4_0 => Level::Level_4_0,
            H264Level::L4_1 => Level::Level_4_1,
            H264Level::L4_2 => Level::Level_4_2,
            H264Level::L5_0 => Level::Level_5_0,
            H264Level::L5_1 => Level::Level_5_1,
            H264Level::L5_2 => Level::Level_5_2,
        }
    }

    pub struct OpenH264Encoder {
        cfg: H264Config,
        enc: Encoder,
        yuv: YUVBuffer,
        frames_in: u64,
        frames_out: u64,
    }

    impl OpenH264Encoder {
        pub fn new(cfg: H264Config) -> Result<Self> {
            cfg.validate()?;
            let lvl = cfg.resolved_level()?;
            let oc = EncoderConfig::new()
                .bitrate(BitRate::from_bps(cfg.bitrate_bps))
                .max_frame_rate(FrameRate::from_hz(cfg.fps as f32))
                .rate_control_mode(RateControlMode::Bitrate)
                .usage_type(UsageType::CameraVideoRealTime)
                .profile(Profile::Baseline)
                .level(level(lvl))
                .complexity(Complexity::Low)
                .intra_frame_period(IntraFramePeriod::from_num_frames(cfg.gop_frames()))
                .scene_change_detect(false)
                .skip_frames(false)
                .sps_pps_strategy(SpsPpsStrategy::ConstantId)
                .num_threads(cfg.threads);
            let enc = Encoder::with_api_config(OpenH264API::from_source(), oc)
                .map_err(|e| MediaError::Encode(format!("openh264 init: {e}")))?;
            let yuv = YUVBuffer::new(cfg.width as usize, cfg.height as usize);
            Ok(Self { cfg, enc, yuv, frames_in: 0, frames_out: 0 })
        }
    }

    impl VideoEncoder for OpenH264Encoder {
        fn backend(&self) -> EncoderBackend {
            EncoderBackend::OpenH264
        }

        fn config(&self) -> &H264Config {
            &self.cfg
        }

        fn encode(&mut self, frame: &RgbFrame) -> Result<Vec<EncodedFrame>> {
            frame.check()?;
            if frame.width != self.cfg.input_width || frame.height != self.cfg.input_height {
                return Err(MediaError::invalid(format!(
                    "frame {}x{} does not match the encoder input {}x{}",
                    frame.width, frame.height, self.cfg.input_width, self.cfg.input_height
                )));
            }
            let scaled;
            let frame = if self.cfg.scales() {
                scaled = scale::scale_rgb(frame, self.cfg.width, self.cfg.height, self.cfg.scale);
                &scaled
            } else {
                frame
            };
            let dims = (frame.width as usize, frame.height as usize);
            self.yuv.read_rgb8(RgbSliceU8::new(&frame.data, dims));
            let ts = Timestamp::from_millis(self.frames_in * 1000 / u64::from(self.cfg.fps));
            self.frames_in += 1;
            let bs = self.enc.encode_at(&self.yuv, ts).map_err(|e| MediaError::Encode(format!("openh264: {e}")))?;
            let ft = bs.frame_type();
            if matches!(ft, FrameType::Skip | FrameType::Invalid) {
                return Ok(Vec::new());
            }
            let data = bs.to_vec();
            if data.is_empty() {
                return Ok(Vec::new());
            }
            let out = EncodedFrame { keyframe: ft == FrameType::IDR, data: Bytes::from(data), index: self.frames_out };
            self.frames_out += 1;
            Ok(vec![out])
        }

        fn force_idr(&mut self) {
            self.enc.force_intra_frame();
        }

        fn finish(&mut self) -> Result<Vec<EncodedFrame>> {
            // No lookahead and no B-frames: nothing is held back.
            Ok(Vec::new())
        }
    }

    /// Decode an Annex-B stream with the OpenH264 decoder into RGB24 frames.
    pub fn decode_rgb(access_units: &[&[u8]]) -> Result<Vec<RgbFrame>> {
        let mut dec = openh264::decoder::Decoder::new().map_err(|e| MediaError::Decode(e.to_string()))?;
        let mut out = Vec::new();
        let push = |y: &openh264::decoder::DecodedYUV<'_>, out: &mut Vec<RgbFrame>| {
            let (w, h) = y.dimensions();
            let mut rgb = vec![0u8; w * h * 3];
            y.write_rgb8(&mut rgb);
            let idx = out.len() as u64;
            out.push(RgbFrame { width: w as u32, height: h as u32, data: Bytes::from(rgb), index: idx });
        };
        for au in access_units {
            if let Some(y) = dec.decode(au).map_err(|e| MediaError::Decode(e.to_string()))? {
                push(&y, &mut out);
            }
        }
        for y in dec.flush_remaining().map_err(|e| MediaError::Decode(e.to_string()))? {
            push(&y, &mut out);
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_levels() {
        let c = H264Config::new(1344, 768, 24);
        assert_eq!(c.resolved_level().unwrap(), H264Level::L3_2);
        assert_eq!(c.gop_frames(), 48);
        assert_eq!(c.bitrate_bps, 6_000_000);
        let mut c31 = c.clone();
        c31.level = Some(H264Level::L3_1);
        assert!(c31.validate().is_err());
        assert!(H264Config::new(833, 480, 16).validate().is_err());
        assert_eq!(H264Config::new(832, 480, 16).gop_frames(), 32);
        assert_eq!(H264Config::new(832, 480, 16).bitrate_bps, 2_500_000);
    }

    #[test]
    fn publish_profiles() {
        // Cloudflare WHIP: H3 1344x768 -> 1280x720 at level 3.1.
        let cf = H264Config::for_publish(PublishTarget::Cloudflare, 1344, 768, 24);
        assert_eq!((cf.input_width, cf.input_height, cf.width, cf.height), (1344, 768, 1280, 720));
        assert_eq!(cf.resolved_level().unwrap(), H264Level::L3_1);
        assert_eq!(cf.profile_level_id().unwrap(), "42e01f");
        assert!(cf.scales());
        // Portrait H3 keeps its orientation.
        let cfp = H264Config::for_publish(PublishTarget::Cloudflare, 768, 1344, 24);
        assert_eq!((cfp.width, cfp.height), (720, 1280));
        // Already within the cap: untouched.
        let wan = H264Config::for_publish(PublishTarget::Cloudflare, 832, 480, 16);
        assert_eq!((wan.width, wan.height), (832, 480));
        assert!(!wan.scales());
        // MediaMTX / peer: native at level 4.0.
        for t in [PublishTarget::Mediamtx, PublishTarget::Peer] {
            let c = H264Config::for_publish(t, 1344, 768, 24);
            assert_eq!((c.width, c.height), (1344, 768));
            assert_eq!(c.resolved_level().unwrap(), H264Level::L4_0);
            assert_eq!(c.profile_level_id().unwrap(), "42e028");
        }
    }

    #[test]
    fn nvenc_args() {
        let cf = H264Config::for_publish(PublishTarget::Cloudflare, 1344, 768, 24);
        let a = pipe_encoder_args(&cf, FfmpegH264::Nvenc).unwrap().join(" ");
        assert!(a.contains("-s 1344x768 -r 24 -i pipe:0"), "{a}");
        assert!(a.contains("-vf scale=1260:720:flags=bicubic,pad=1280:720:10:0:black,setsar=1"), "{a}");
        assert!(a.contains("-c:v h264_nvenc"));
        assert!(a.contains("-profile:v baseline -level:v 3.1 -rc cbr -b:v 6000000"));
        assert!(a.contains("-g 48 -bf 0 -forced-idr 1 -no-scenecut 1"));
        assert!(a.contains("-zerolatency 1") && a.contains("-aud 1"));
        assert!(a.ends_with("-f h264 pipe:1"));
        assert!(!a.contains("libx264"));
        let peer = H264Config::for_publish(PublishTarget::Peer, 1344, 768, 24);
        let a = pipe_encoder_args(&peer, FfmpegH264::Nvenc).unwrap().join(" ");
        assert!(!a.contains("-vf"));
        assert!(a.contains("-level:v 4.0"));
        assert!(FfmpegH264::Nvenc.file_args(19).join(" ").contains("h264_nvenc -preset p5 -tune hq -profile:v high -rc vbr -cq 19"));
        assert_eq!(FfmpegH264::default(), FfmpegH264::Nvenc);
    }

    #[test]
    fn backend_names() {
        assert_eq!("nvenc".parse::<EncoderBackend>().unwrap(), EncoderBackend::Nvenc);
        assert_eq!("openh264".parse::<EncoderBackend>().unwrap(), EncoderBackend::OpenH264);
        assert!("x264-ffmpeg".parse::<EncoderBackend>().is_err());
        assert!("x264".parse::<EncoderBackend>().is_err());
        assert!(EncoderBackend::Nvenc.compiled());
        assert_eq!(serde_json::to_string(&EncoderBackend::Nvenc).unwrap(), "\"nvenc\"");
        assert_eq!(serde_json::to_string(&EncoderBackend::OpenH264).unwrap(), "\"openh264\"");
        assert!(serde_json::from_str::<EncoderBackend>("\"cpu-test-x264\"").is_err());
    }

    #[test]
    fn auto_picks_nvenc_only_when_the_probe_encodes() {
        let a = AutoEncoder::from_probe(Ok(()));
        assert_eq!((a.backend, a.nvenc_error), (EncoderBackend::Nvenc, None));
        let a = AutoEncoder::from_probe(Err("No NVENC capable devices found".into()));
        assert_eq!(a.backend, EncoderBackend::OpenH264);
        assert_eq!(a.nvenc_error.as_deref(), Some("No NVENC capable devices found"));
        // A missing ffmpeg is a probe failure, not a panic.
        let e = nvenc_probe_with(std::path::Path::new("/nonexistent/ffmpeg")).unwrap_err();
        assert!(e.contains("cannot run /nonexistent/ffmpeg"), "{e}");
        assert!(!probe_error_is_transient(&e));
    }

    #[test]
    fn auto_probe_retries_once_on_a_transient_failure() {
        use std::time::Duration;
        // Fails once (NVENC not ready), then encodes: NVENC after 2 probes.
        let mut n = 0;
        let a = AutoEncoder::probe_with_retry(
            || {
                n += 1;
                if n == 1 { Err("OpenEncodeSessionEx failed: out of memory (10)".into()) } else { Ok(()) }
            },
            Duration::ZERO,
        );
        assert_eq!((a.backend, a.attempts, a.nvenc_error), (EncoderBackend::Nvenc, 2, None));
        // Fails twice: OpenH264 with the second error.
        let mut n = 0;
        let a = AutoEncoder::probe_with_retry(
            || {
                n += 1;
                Err(format!("Cannot load libcuda.so.1 ({n})"))
            },
            Duration::ZERO,
        );
        assert_eq!((a.backend, a.attempts), (EncoderBackend::OpenH264, 2));
        assert_eq!(a.nvenc_error.as_deref(), Some("Cannot load libcuda.so.1 (2)"));
        // No retry when a retry cannot help.
        for e in ["cannot run ffmpeg: No such file or directory", "Unknown encoder 'h264_nvenc'"] {
            let mut n = 0;
            let a = AutoEncoder::probe_with_retry(
                || {
                    n += 1;
                    Err(e.to_owned())
                },
                Duration::ZERO,
            );
            assert_eq!((a.attempts, n), (1, 1), "{e}");
        }
        // Success first time: one probe.
        let a = AutoEncoder::probe_with_retry(|| Ok(()), Duration::ZERO);
        assert_eq!((a.backend, a.attempts), (EncoderBackend::Nvenc, 1));
    }

    #[test]
    fn idr_limiter_one_per_second() {
        let mut l = IdrLimiter::new(1.0);
        assert!(!l.poll(0.0));
        l.request();
        assert!(l.poll(0.0));
        l.request();
        assert!(!l.poll(0.5)); // too soon; stays pending
        assert!(l.poll(1.0));
        assert!(!l.poll(5.0));
    }

    #[test]
    fn aud_splitter() {
        let mut buf = vec![0, 0, 0, 1, 9, 0xf0, 0, 0, 0, 1, 0x65, 1, 0, 0, 0, 1, 9, 0xf0, 0, 0, 1, 0x41, 2];
        let aus = take_complete_aus(&mut buf);
        assert_eq!(aus.len(), 1);
        assert_eq!(aus[0], vec![0, 0, 0, 1, 9, 0xf0, 0, 0, 0, 1, 0x65, 1]);
        assert_eq!(buf, vec![0, 0, 0, 1, 9, 0xf0, 0, 0, 1, 0x41, 2]);
    }
}
