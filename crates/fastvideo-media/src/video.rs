//! H.264 encoding behind one trait (design §5.9).
//!
//! | Backend | `encoder =` | Build | Notes |
//! |---|---|---|---|
//! | OpenH264, in process | `openh264` | `openh264` feature | Constrained Baseline, IDR every `gop_seconds` (2 s), no scene-cut IDRs, forced IDR on PLI. The streaming default |
//! | libx264 via an ffmpeg subprocess | `x264-ffmpeg` | always (needs ffmpeg at run time) | Same GOP/profile; forced IDR is not available through the pipe. The switch if the OpenH264 source build is not cleared legally (risk R17) |
//! | NVENC | `nvenc` | **slot only** | [`nvenc::NvencEncoder::new`] returns `Unsupported`. Reserved so a hardware (or other ffmpeg-based) backend can be added behind the same trait once the encoder licensing path is chosen |
//!
//! Every backend takes [`RgbFrame`]s and returns Annex-B access units.

use std::io::{Read, Write};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::mpsc;

use bytes::Bytes;

use crate::av::{AvCheck, RgbFrame};
use crate::error::{MediaError, Result};
use crate::h264::{self, H264Level};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum EncoderBackend {
    #[serde(rename = "openh264")]
    OpenH264,
    X264Ffmpeg,
    Nvenc,
}

impl std::str::FromStr for EncoderBackend {
    type Err = MediaError;
    fn from_str(s: &str) -> Result<Self> {
        match s {
            "openh264" => Ok(Self::OpenH264),
            "x264-ffmpeg" | "x264" => Ok(Self::X264Ffmpeg),
            "nvenc" => Ok(Self::Nvenc),
            other => Err(MediaError::invalid(format!("unknown encoder {other:?}"))),
        }
    }
}

impl EncoderBackend {
    /// Whether this build can construct the backend at all (ffmpeg presence
    /// is checked at construction).
    pub fn compiled(self) -> bool {
        match self {
            Self::OpenH264 => cfg!(feature = "openh264"),
            Self::X264Ffmpeg => true,
            Self::Nvenc => false,
        }
    }
}

/// Default bitrate by canvas (§5.1): 6 Mb/s at 768p, 2.5 Mb/s at 480p.
pub fn default_bitrate(width: u32, height: u32) -> u32 {
    if width.min(height) >= 720 { 6_000_000 } else { 2_500_000 }
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct H264Config {
    pub width: u32,
    pub height: u32,
    pub fps: u32,
    pub bitrate_bps: u32,
    /// Seconds between IDRs (2 s for WHIP/Cloudflare/browsers, 1 s for HLS).
    pub gop_seconds: f32,
    /// `None` picks the smallest level that fits (3.2 for 1344×768@24).
    pub level: Option<H264Level>,
    /// Encoder threads (0 = backend default).
    pub threads: u16,
}

impl H264Config {
    pub fn new(width: u32, height: u32, fps: u32) -> Self {
        Self { width, height, fps, bitrate_bps: default_bitrate(width, height), gop_seconds: 2.0, level: None, threads: 0 }
    }

    pub fn gop_frames(&self) -> u32 {
        ((self.fps as f32 * self.gop_seconds).round() as u32).max(1)
    }

    pub fn validate(&self) -> Result<()> {
        if self.width == 0 || self.height == 0 || self.width % 2 != 0 || self.height % 2 != 0 {
            return Err(MediaError::invalid(format!("H.264 4:2:0 needs even dimensions, got {}x{}", self.width, self.height)));
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
    /// that is too small for the canvas is an error rather than a silently
    /// non-conforming stream.
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
    /// Encode one frame; returns the access units that became ready (zero or
    /// more: a subprocess backend lags by up to one frame).
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
        #[cfg(feature = "openh264")]
        EncoderBackend::OpenH264 => Ok(Box::new(openh264_backend::OpenH264Encoder::new(cfg)?)),
        #[cfg(not(feature = "openh264"))]
        EncoderBackend::OpenH264 => {
            Err(MediaError::Unsupported("built without the `openh264` feature of fastvideo-media".into()))
        }
        EncoderBackend::X264Ffmpeg => Ok(Box::new(X264FfmpegEncoder::new(cfg)?)),
        EncoderBackend::Nvenc => Ok(Box::new(nvenc::NvencEncoder::new(cfg)?)),
    }
}

/// Rate limit for forced IDRs: at most one per `min_interval` seconds (§5.1:
/// one per second).
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
// OpenH264
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
            if frame.width != self.cfg.width || frame.height != self.cfg.height {
                return Err(MediaError::invalid(format!(
                    "frame {}x{} does not match the encoder's {}x{}",
                    frame.width, frame.height, self.cfg.width, self.cfg.height
                )));
            }
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
            // OpenH264 has no lookahead and no B-frames: nothing is held back.
            Ok(Vec::new())
        }
    }

    /// Decode an Annex-B stream with the OpenH264 decoder into RGB24 frames
    /// (used by the round-trip tests and for thumbnails).
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

// ---------------------------------------------------------------------------
// x264 through an ffmpeg subprocess
// ---------------------------------------------------------------------------

/// libx264 behind ffmpeg: rgb24 on stdin, Annex-B with AUDs on stdout.
pub struct X264FfmpegEncoder {
    cfg: H264Config,
    child: Child,
    stdin: Option<ChildStdin>,
    rx: mpsc::Receiver<Vec<u8>>,
    reader: Option<std::thread::JoinHandle<()>>,
    out_index: u64,
    warned_idr: bool,
}

impl X264FfmpegEncoder {
    /// ffmpeg arguments after the input (shared with the RTMP/HLS sinks).
    pub fn x264_args(cfg: &H264Config) -> Result<Vec<String>> {
        let lvl = cfg.resolved_level()?;
        let g = cfg.gop_frames().to_string();
        let b = cfg.bitrate_bps;
        Ok(vec![
            "-c:v".into(),
            "libx264".into(),
            "-preset".into(),
            "veryfast".into(),
            "-tune".into(),
            "zerolatency".into(),
            "-profile:v".into(),
            "baseline".into(),
            "-level:v".into(),
            lvl.to_string(),
            "-pix_fmt".into(),
            "yuv420p".into(),
            "-g".into(),
            g.clone(),
            "-keyint_min".into(),
            g,
            "-sc_threshold".into(),
            "0".into(),
            "-b:v".into(),
            b.to_string(),
            "-maxrate".into(),
            (u64::from(b) * 6 / 5).to_string(),
            "-bufsize".into(),
            (u64::from(b) * 2).to_string(),
        ])
    }

    pub fn new(cfg: H264Config) -> Result<Self> {
        cfg.validate()?;
        let mut cmd = crate::tools::ffmpeg_command();
        cmd.args(["-f", "rawvideo", "-pix_fmt", "rgb24", "-s"])
            .arg(format!("{}x{}", cfg.width, cfg.height))
            .args(["-r", &cfg.fps.to_string(), "-i", "pipe:0", "-an"])
            .args(Self::x264_args(&cfg)?)
            .args(["-x264-params", "aud=1:repeat-headers=1", "-f", "h264", "pipe:1"])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null());
        let mut child = spawn(cmd)?;
        let stdin = child.stdin.take();
        let mut stdout = child.stdout.take().ok_or_else(|| MediaError::tool("ffmpeg", "no stdout"))?;
        let (tx, rx) = mpsc::channel();
        let reader = std::thread::Builder::new().name("x264-ffmpeg-out".into()).spawn(move || {
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
        Ok(Self { cfg, child, stdin, rx, reader: Some(reader), out_index: 0, warned_idr: false })
    }

    fn collect(&mut self, block: bool) -> Vec<EncodedFrame> {
        let mut out = Vec::new();
        loop {
            let au = if block { self.rx.recv().ok() } else { self.rx.try_recv().ok() };
            let Some(au) = au else { break };
            out.push(EncodedFrame { keyframe: h264::is_idr(&au), data: Bytes::from(au), index: self.out_index });
            self.out_index += 1;
        }
        out
    }
}

fn spawn(mut cmd: Command) -> Result<Child> {
    cmd.spawn().map_err(|e| MediaError::tool("ffmpeg", format!("not available: {e}")))
}

/// Pop every access unit that is followed by the next AUD from `buf`.
fn take_complete_aus(buf: &mut Vec<u8>) -> Vec<Vec<u8>> {
    // AUD NAL: start code then 0x09.
    let mut starts = Vec::new();
    let mut i = 0;
    while i + 4 <= buf.len() {
        if buf[i] == 0 && buf[i + 1] == 0 && buf[i + 2] == 1 && buf[i + 3] & 0x1f == h264::nal::AUD {
            // Include a leading zero of a 4-byte start code.
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

impl VideoEncoder for X264FfmpegEncoder {
    fn backend(&self) -> EncoderBackend {
        EncoderBackend::X264Ffmpeg
    }

    fn config(&self) -> &H264Config {
        &self.cfg
    }

    fn encode(&mut self, frame: &RgbFrame) -> Result<Vec<EncodedFrame>> {
        frame.check()?;
        if frame.width != self.cfg.width || frame.height != self.cfg.height {
            return Err(MediaError::invalid("frame size does not match the encoder"));
        }
        let stdin = self.stdin.as_mut().ok_or_else(|| MediaError::Encode("encoder finished".into()))?;
        stdin.write_all(&frame.data).map_err(|e| MediaError::tool("ffmpeg", format!("stdin: {e}")))?;
        Ok(self.collect(false))
    }

    fn force_idr(&mut self) {
        // Not reachable through the pipe; the fixed GOP bounds recovery time.
        if !self.warned_idr {
            tracing::debug!("x264-ffmpeg: forced IDR not supported, waiting for the next GOP");
            self.warned_idr = true;
        }
    }

    fn finish(&mut self) -> Result<Vec<EncodedFrame>> {
        drop(self.stdin.take());
        let out = self.collect(true);
        if let Some(r) = self.reader.take() {
            let _ = r.join();
        }
        let status = self.child.wait()?;
        if !status.success() {
            return Err(MediaError::tool("ffmpeg", format!("x264 encode exited with {status}")));
        }
        Ok(out)
    }
}

impl Drop for X264FfmpegEncoder {
    fn drop(&mut self) {
        drop(self.stdin.take());
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

// ---------------------------------------------------------------------------
// NVENC slot
// ---------------------------------------------------------------------------

pub mod nvenc {
    //! Reserved backend slot. The encoder licensing path (OpenH264 source
    //! build, x264, NVENC via the driver's video capability, or an ffmpeg
    //! build with a hardware encoder) is not decided yet (§5.9, risk R17).
    //! A real implementation implements [`VideoEncoder`](super::VideoEncoder)
    //! here and flips [`EncoderBackend::compiled`](super::EncoderBackend::compiled).

    use super::*;

    pub struct NvencEncoder {
        cfg: H264Config,
    }

    impl NvencEncoder {
        pub fn new(cfg: H264Config) -> Result<Self> {
            let _ = cfg;
            Err(MediaError::Unsupported(
                "the NVENC backend is a reserved slot and is not implemented; use openh264 or x264-ffmpeg".into(),
            ))
        }
    }

    impl VideoEncoder for NvencEncoder {
        fn backend(&self) -> EncoderBackend {
            EncoderBackend::Nvenc
        }
        fn config(&self) -> &H264Config {
            &self.cfg
        }
        fn encode(&mut self, _frame: &RgbFrame) -> Result<Vec<EncodedFrame>> {
            Err(MediaError::Unsupported("nvenc".into()))
        }
        fn force_idr(&mut self) {}
        fn finish(&mut self) -> Result<Vec<EncodedFrame>> {
            Ok(Vec::new())
        }
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
        let mut c40 = c.clone();
        c40.level = Some(H264Level::L4_0);
        assert_eq!(c40.resolved_level().unwrap(), H264Level::L4_0);
        assert!(H264Config::new(833, 480, 16).validate().is_err());
        assert_eq!(H264Config::new(832, 480, 16).gop_frames(), 32);
        assert_eq!(H264Config::new(832, 480, 16).bitrate_bps, 2_500_000);
    }

    #[test]
    fn backend_names_and_nvenc_slot() {
        assert_eq!("openh264".parse::<EncoderBackend>().unwrap(), EncoderBackend::OpenH264);
        assert_eq!("x264-ffmpeg".parse::<EncoderBackend>().unwrap(), EncoderBackend::X264Ffmpeg);
        assert_eq!("nvenc".parse::<EncoderBackend>().unwrap(), EncoderBackend::Nvenc);
        assert!(!EncoderBackend::Nvenc.compiled());
        let e = create_encoder(EncoderBackend::Nvenc, H264Config::new(64, 64, 24)).err().unwrap();
        assert!(matches!(e, MediaError::Unsupported(_)));
        let j = serde_json::to_string(&EncoderBackend::X264Ffmpeg).unwrap();
        assert_eq!(j, "\"x264-ffmpeg\"");
        assert_eq!(serde_json::to_string(&EncoderBackend::OpenH264).unwrap(), "\"openh264\"");
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
