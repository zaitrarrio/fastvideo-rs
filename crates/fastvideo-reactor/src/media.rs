//! Per-session media: encode once per negotiated codec and fan out to the
//! peers (design §5.1, §5.3; reactor §4.2, §4.5).
//!
//! Pacing is the engine's (`fastvideo_engine_service::stream::pace`, WP-12 /
//! WP-15): the clip pacer (`AvPacer`, idle policy **Black**) or the causal
//! pacer (`FramePacer`, adaptive fps) yields one [`Tick`] per frame, each with
//! exactly `48000/fps` mono samples for an audio session. Here:
//!
//! - **Video thread**: a fresh frame is encoded once per codec in use
//!   ([`VideoCodec::H264`] through `fastvideo-media`, [`VideoCodec::Vp8`]
//!   intra-only through libwebp) and sent to every peer of that codec at the
//!   tick's 90 kHz RTP time. A repeated frame is not re-sent (the client holds
//!   its last frame, as RT's pacer), unless the held picture changed (the
//!   pacer's flush to black after a clip ends with nothing armed, or on a cut).
//! - **Audio thread**: RT's feeder. Tick audio goes into a ≤200 ms buffer;
//!   exactly one 10 ms / 480-sample Opus frame per 10 ms goes out, silence
//!   when the buffer is short, so the audio RTP clock stays on wall time.
//! - **Black frame at connection start** (reactor §4.2): a new peer, a
//!   resumed video track, a PLI/FIR or a dropped tick ([`MediaPipeline::kick`])
//!   re-sends the current picture as a keyframe; before the first frame the
//!   current picture is black.

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

use bytes::Bytes;
use fastvideo_engine_service::{Tick, TickReceiver};
use fastvideo_media::pacer::VideoOut;
use fastvideo_protocol::RgbFrame;
use fastvideo_webrtc::host::PeerHandle;
use fastvideo_webrtc::writer::{AudioPacket, TrackKind, VideoCodec, VideoFrame};

/// 10 ms at 48 kHz (RT `_push_audio_frame`).
pub const AUDIO_FRAME_SAMPLES: usize = 480;
/// RT's per-track audio buffer cap (200 ms).
pub const AUDIO_BUFFER_SAMPLES: usize = 9600;

/// Which H.264 encoder serves H.264 peers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum H264Backend {
    /// Production (NVENC through ffmpeg `h264_nvenc`).
    Nvenc,
    /// OpenH264 in process (CPU test backend, `openh264` feature).
    OpenH264,
    /// No H.264: only VP8 is offered in answers.
    Off,
}

impl H264Backend {
    /// Whether this build (and machine) can encode H.264 with it.
    pub fn usable(self) -> bool {
        match self {
            H264Backend::Nvenc => fastvideo_media::video::nvenc_available(),
            H264Backend::OpenH264 => fastvideo_media::video::EncoderBackend::OpenH264.compiled(),
            H264Backend::Off => false,
        }
    }
}

/// Codecs this build can send, in answer-preference order: H.264 first when
/// its backend is usable, then VP8 (feature `vp8`).
pub fn sendable_codecs(h264: H264Backend) -> Vec<VideoCodec> {
    let mut v = Vec::new();
    if h264.usable() {
        v.push(VideoCodec::H264);
    }
    if cfg!(feature = "vp8") {
        v.push(VideoCodec::Vp8);
    }
    v
}

#[derive(Clone, Debug)]
pub struct MediaConfig {
    /// Nominal fps (encoder rate control; RTP comes from the ticks).
    pub fps: u32,
    /// The session has an audio track.
    pub audio: bool,
    /// Canvas of the black picture before the first frame.
    pub canvas: (u32, u32),
    pub h264: H264Backend,
    /// H.264 target bitrate (`None`: `fastvideo-media` default per canvas).
    pub bitrate_bps: Option<u32>,
    /// VP8 (libwebp) quality 0..100.
    pub vp8_quality: f32,
}

impl MediaConfig {
    pub fn new(fps: u32, audio: bool, canvas: (u32, u32)) -> Self {
        Self { fps, audio, canvas, h264: H264Backend::Nvenc, bitrate_bps: None, vp8_quality: 70.0 }
    }
}

/// Counters.
#[derive(Debug, Default)]
pub struct MediaStats {
    pub ticks: AtomicU64,
    pub frames_encoded: AtomicU64,
    pub frames_sent: AtomicU64,
    pub black_frames: AtomicU64,
    pub keyframes_forced: AtomicU64,
    pub holds: AtomicU64,
    pub encode_errors: AtomicU64,
    pub audio_frames: AtomicU64,
    pub audio_silence_frames: AtomicU64,
    pub audio_samples_in: AtomicU64,
}

struct PeerSink {
    handle: PeerHandle,
    codec: Option<VideoCodec>,
    audio: bool,
}

struct Shared {
    cfg: MediaConfig,
    peers: Mutex<HashMap<u64, PeerSink>>,
    kicks: Mutex<HashSet<VideoCodec>>,
    audio: Mutex<VecDeque<f32>>,
    closed: AtomicBool,
    wake: (Mutex<()>, Condvar),
    stats: MediaStats,
    on_first_frame: Mutex<Option<Box<dyn FnOnce() + Send>>>,
}

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

/// One session's encode and fan-out. Dropping it stops the threads.
pub struct MediaPipeline {
    sh: Arc<Shared>,
}

impl std::fmt::Debug for MediaPipeline {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MediaPipeline").field("fps", &self.sh.cfg.fps).finish()
    }
}

impl MediaPipeline {
    /// Starts consuming `ticks` (must be called inside a tokio runtime).
    pub fn start(cfg: MediaConfig, ticks: TickReceiver) -> Self {
        let sh = Arc::new(Shared {
            cfg,
            peers: Mutex::new(HashMap::new()),
            kicks: Mutex::new(HashSet::new()),
            audio: Mutex::new(VecDeque::new()),
            closed: AtomicBool::new(false),
            wake: (Mutex::new(()), Condvar::new()),
            stats: MediaStats::default(),
            on_first_frame: Mutex::new(None),
        });
        let v = sh.clone();
        let rt = tokio::runtime::Handle::current();
        std::thread::Builder::new()
            .name("reactor-video".into())
            .spawn(move || video_loop(v, ticks, rt))
            .expect("spawning the reactor video thread");
        if sh.cfg.audio {
            let a = sh.clone();
            std::thread::Builder::new()
                .name("reactor-audio".into())
                .spawn(move || audio_loop(a))
                .expect("spawning the reactor audio thread");
        }
        Self { sh }
    }

    pub fn stats(&self) -> &MediaStats {
        &self.sh.stats
    }

    /// Called once when the first real frame goes out (TTFF `transport`).
    pub fn on_first_frame(&self, f: impl FnOnce() + Send + 'static) {
        *lock(&self.sh.on_first_frame) = Some(Box::new(f));
    }

    /// Adds a peer; its first video is a keyframe of the current picture
    /// (black before the first frame).
    pub fn add_peer(&self, handle: PeerHandle) {
        let codec = handle.video_codec();
        let audio = handle.sends(TrackKind::Audio);
        lock(&self.sh.peers).insert(handle.id(), PeerSink { handle, codec, audio });
        if let Some(c) = codec {
            self.kick(c);
        }
    }

    pub fn remove_peer(&self, id: u64) {
        lock(&self.sh.peers).remove(&id);
    }

    pub fn peer_count(&self) -> usize {
        lock(&self.sh.peers).len()
    }

    /// Re-send the current picture as a keyframe for `codec` (PLI/FIR,
    /// resumed track, new peer). Served within 50 ms even when no tick flows.
    pub fn kick(&self, codec: VideoCodec) {
        lock(&self.sh.kicks).insert(codec);
    }

    pub fn close(&self) {
        self.sh.closed.store(true, Ordering::Relaxed);
        let (_, cv) = &self.sh.wake;
        cv.notify_all();
    }
}

impl Drop for MediaPipeline {
    fn drop(&mut self) {
        self.close();
    }
}

fn same_picture(a: &RgbFrame, b: &RgbFrame) -> bool {
    a.width == b.width && a.height == b.height && a.data.as_ptr() == b.data.as_ptr() && a.data.len() == b.data.len()
}

struct VideoState {
    encoders: HashMap<VideoCodec, Box<dyn FrameEncoder>>,
    /// The picture on screen (last sent), and whether it was ever real.
    current: Option<RgbFrame>,
    last_rtp: u64,
    first_sent: bool,
}

fn video_loop(sh: Arc<Shared>, mut ticks: TickReceiver, rt: tokio::runtime::Handle) {
    let step = 90_000 / u64::from(sh.cfg.fps.max(1));
    let mut st = VideoState { encoders: HashMap::new(), current: None, last_rtp: 0, first_sent: false };
    while !sh.closed.load(Ordering::Relaxed) {
        let next = rt.block_on(async { tokio::time::timeout(Duration::from_millis(50), ticks.recv()).await });
        let tick: Option<Tick> = match next {
            Ok(Some(t)) => Some(t),
            // The pacer ended (session closed).
            Ok(None) => break,
            Err(_) => None,
        };
        let mut kicks: HashSet<VideoCodec> = std::mem::take(&mut *lock(&sh.kicks));
        if ticks.take_dropped() {
            // Frames were lost before the encoder: re-sync every codec.
            kicks.extend(lock(&sh.peers).values().filter_map(|p| p.codec));
        }
        let Some(t) = tick else {
            if !kicks.is_empty() {
                let rtp = st.last_rtp + step;
                let pic = st.current.clone();
                send_picture(&sh, &mut st, pic, rtp, &kicks, true);
            }
            continue;
        };
        sh.stats.ticks.fetch_add(1, Ordering::Relaxed);
        if let Some(a) = t.audio {
            sh.stats.audio_samples_in.fetch_add(a.len() as u64, Ordering::Relaxed);
            let mut buf = lock(&sh.audio);
            buf.extend(a);
            let over = buf.len().saturating_sub(AUDIO_BUFFER_SAMPLES);
            buf.drain(..over);
        }
        let rtp = u64::from(t.video_rtp).max(st.last_rtp + 1);
        match t.video {
            VideoOut::Fresh(f) => {
                send_picture(&sh, &mut st, Some(f), rtp, &kicks, false);
                if !st.first_sent {
                    st.first_sent = true;
                    if let Some(cb) = lock(&sh.on_first_frame).take() {
                        cb();
                    }
                }
            }
            VideoOut::Repeat(f) => {
                let changed = st.current.as_ref().is_none_or(|c| !same_picture(c, &f));
                if changed {
                    // The pacer now holds another picture (flush to black).
                    send_picture(&sh, &mut st, Some(f), rtp, &kicks, true);
                } else if !kicks.is_empty() {
                    send_picture(&sh, &mut st, Some(f), rtp, &kicks, true);
                } else {
                    sh.stats.holds.fetch_add(1, Ordering::Relaxed);
                }
            }
            VideoOut::Nothing => {
                if !kicks.is_empty() {
                    let pic = st.current.clone();
                    send_picture(&sh, &mut st, pic, rtp, &kicks, true);
                }
            }
        }
    }
}

/// Encodes `pic` (black when `None`) for every codec in use and sends it.
/// `all_key`: a keyframe for every codec; else only for codecs in `kicks`.
fn send_picture(
    sh: &Shared,
    st: &mut VideoState,
    pic: Option<RgbFrame>,
    rtp: u64,
    kicks: &HashSet<VideoCodec>,
    all_key: bool,
) {
    let frame = pic.unwrap_or_else(|| {
        let (w, h) = st.current.as_ref().map_or(sh.cfg.canvas, |c| (c.width, c.height));
        RgbFrame::black(w, h, 0)
    });
    let peers: Vec<(PeerHandle, VideoCodec)> = lock(&sh.peers)
        .values()
        .filter_map(|p| p.codec.map(|c| (p.handle.clone(), c)))
        .collect();
    let codecs: HashSet<VideoCodec> = peers.iter().map(|(_, c)| *c).collect();
    let black = frame.data.iter().step_by(97).all(|b| *b == 0);
    for codec in codecs {
        let key = all_key || kicks.contains(&codec);
        if kicks.contains(&codec) {
            sh.stats.keyframes_forced.fetch_add(1, Ordering::Relaxed);
        }
        let enc = match st.encoders.get_mut(&codec) {
            Some(e) if e.dims() == (frame.width, frame.height) => e,
            _ => match new_encoder(codec, &sh.cfg, frame.width, frame.height) {
                Ok(e) => st.encoders.entry(codec).insert_entry(e).into_mut(),
                Err(e) => {
                    tracing::warn!(?codec, error = %e, "video encoder unavailable");
                    sh.stats.encode_errors.fetch_add(1, Ordering::Relaxed);
                    continue;
                }
            },
        };
        match enc.encode(&frame, key) {
            Ok(Some(data)) => {
                sh.stats.frames_encoded.fetch_add(1, Ordering::Relaxed);
                if black {
                    sh.stats.black_frames.fetch_add(1, Ordering::Relaxed);
                }
                for (p, c) in &peers {
                    if *c == codec && p.send_video(VideoFrame::new(data.clone(), rtp)).is_ok() {
                        sh.stats.frames_sent.fetch_add(1, Ordering::Relaxed);
                    }
                }
            }
            Ok(None) => {}
            Err(e) => {
                sh.stats.encode_errors.fetch_add(1, Ordering::Relaxed);
                tracing::warn!(?codec, error = %e, "video encode failed");
                st.encoders.remove(&codec);
            }
        }
    }
    st.current = Some(frame);
    st.last_rtp = rtp;
}

fn sleep_until(sh: &Shared, t: Instant) {
    let (m, cv) = &sh.wake;
    let mut g = lock(m);
    loop {
        if sh.closed.load(Ordering::Relaxed) {
            return;
        }
        let now = Instant::now();
        if now >= t {
            return;
        }
        g = cv.wait_timeout(g, t - now).map(|r| r.0).unwrap_or_else(|e| e.into_inner().0);
    }
}

fn audio_loop(sh: Arc<Shared>) {
    let period = Duration::from_millis(10);
    let mut enc = match AudioEncoder::new() {
        Ok(e) => Some(e),
        Err(e) => {
            tracing::warn!(error = %e, "no Opus encoder in this build: audio tracks stay silent");
            None
        }
    };
    let mut samples: u64 = 0;
    let mut next = Instant::now();
    while !sh.closed.load(Ordering::Relaxed) {
        sleep_until(&sh, next);
        // Catch up at most 5 frames after a stall (RT feeder).
        let now = Instant::now();
        let mut due = 0;
        while next <= now && due < 5 {
            next += period;
            due += 1;
        }
        if next <= now {
            next = now + period;
        }
        for _ in 0..due {
            let frame: Vec<f32> = {
                let mut buf = lock(&sh.audio);
                if buf.len() >= AUDIO_FRAME_SAMPLES {
                    buf.drain(..AUDIO_FRAME_SAMPLES).collect()
                } else {
                    sh.stats.audio_silence_frames.fetch_add(1, Ordering::Relaxed);
                    vec![0.0; AUDIO_FRAME_SAMPLES]
                }
            };
            let rtp = samples;
            samples += AUDIO_FRAME_SAMPLES as u64;
            sh.stats.audio_frames.fetch_add(1, Ordering::Relaxed);
            let Some(enc) = enc.as_mut() else { continue };
            let peers: Vec<PeerHandle> =
                lock(&sh.peers).values().filter(|p| p.audio).map(|p| p.handle.clone()).collect();
            if peers.is_empty() {
                continue;
            }
            match enc.encode(&frame) {
                Ok(data) => {
                    for p in &peers {
                        let _ = p.send_audio(AudioPacket::new(data.clone(), rtp));
                    }
                }
                Err(e) => tracing::warn!(error = %e, "opus encode failed"),
            }
        }
    }
}

// ---- encoders ----------------------------------------------------------------

/// One video encoder for one codec at one canvas.
pub trait FrameEncoder: Send {
    fn dims(&self) -> (u32, u32);
    /// Encode `f`; `key` forces a keyframe. `None`: nothing ready yet.
    fn encode(&mut self, f: &RgbFrame, key: bool) -> Result<Option<Bytes>, String>;
}

fn new_encoder(codec: VideoCodec, cfg: &MediaConfig, w: u32, h: u32) -> Result<Box<dyn FrameEncoder>, String> {
    match codec {
        VideoCodec::H264 => H264Encoder::new(cfg, w, h).map(|e| Box::new(e) as Box<dyn FrameEncoder>),
        VideoCodec::Vp8 => vp8::Vp8Encoder::new(w, h, cfg.vp8_quality).map(|e| Box::new(e) as Box<dyn FrameEncoder>),
    }
}

struct H264Encoder {
    dims: (u32, u32),
    inner: Box<dyn fastvideo_media::video::VideoEncoder>,
}

impl H264Encoder {
    fn new(cfg: &MediaConfig, w: u32, h: u32) -> Result<Self, String> {
        use fastvideo_media::video::{create_encoder, EncoderBackend, H264Config, PublishTarget};
        let backend = match cfg.h264 {
            H264Backend::Nvenc => EncoderBackend::Nvenc,
            H264Backend::OpenH264 => EncoderBackend::OpenH264,
            H264Backend::Off => return Err("H.264 is off".into()),
        };
        let mut c = H264Config::for_publish(PublishTarget::Peer, w, h, cfg.fps.max(1));
        if let Some(b) = cfg.bitrate_bps {
            c.bitrate_bps = b;
        }
        let inner = create_encoder(backend, c).map_err(|e| e.to_string())?;
        Ok(Self { dims: (w, h), inner })
    }
}

impl FrameEncoder for H264Encoder {
    fn dims(&self) -> (u32, u32) {
        self.dims
    }
    fn encode(&mut self, f: &RgbFrame, key: bool) -> Result<Option<Bytes>, String> {
        if key {
            self.inner.force_idr();
        }
        let out = self.inner.encode(f).map_err(|e| e.to_string())?;
        // No B-frames: at most one AU per input; a lagging pipe encoder may
        // hand back an earlier one, which is still in order.
        Ok(out.into_iter().last().map(|e| e.data))
    }
}

pub mod vp8 {
    //! Intra-only VP8: every frame is a keyframe, encoded by libwebp (a
    //! lossy WebP image *is* one VP8 key frame in a RIFF container). Heavier
    //! on bitrate than an inter-frame encoder, but it needs no system library
    //! and every WebRTC stack decodes VP8, including the Python reactor_sdk's
    //! libwebrtc, which offers no H.264.

    use bytes::Bytes;
    use fastvideo_protocol::RgbFrame;

    pub struct Vp8Encoder {
        dims: (u32, u32),
        #[cfg_attr(not(feature = "vp8"), allow(dead_code))]
        quality: f32,
    }

    impl Vp8Encoder {
        pub fn new(w: u32, h: u32, quality: f32) -> Result<Self, String> {
            if !cfg!(feature = "vp8") {
                return Err("built without the `vp8` feature of fastvideo-reactor".into());
            }
            if w == 0 || h == 0 || w > 16383 || h > 16383 {
                return Err(format!("VP8 cannot carry {w}x{h}"));
            }
            Ok(Self { dims: (w, h), quality })
        }
    }

    /// The VP8 frame inside a simple-format lossy WebP file
    /// (`RIFF....WEBPVP8 <len><frame>`).
    pub fn webp_to_vp8(webp: &[u8]) -> Result<&[u8], String> {
        if webp.len() < 20 || &webp[0..4] != b"RIFF" || &webp[8..12] != b"WEBP" {
            return Err("not a WebP file".into());
        }
        if &webp[12..16] != b"VP8 " {
            return Err(format!(
                "unexpected WebP chunk {:?} (want lossy `VP8 `)",
                String::from_utf8_lossy(&webp[12..16])
            ));
        }
        let len = u32::from_le_bytes([webp[16], webp[17], webp[18], webp[19]]) as usize;
        webp.get(20..20 + len).ok_or_else(|| "truncated VP8 chunk".to_string())
    }

    /// Whether a VP8 frame is a key frame (RFC 6386 §9.1: bit 0 clear).
    pub fn is_keyframe(frame: &[u8]) -> bool {
        frame.first().is_some_and(|b| b & 1 == 0)
    }

    impl super::FrameEncoder for Vp8Encoder {
        fn dims(&self) -> (u32, u32) {
            self.dims
        }

        #[cfg(feature = "vp8")]
        fn encode(&mut self, f: &RgbFrame, _key: bool) -> Result<Option<Bytes>, String> {
            let mut cfg = webp::WebPConfig::new().map_err(|_| "libwebp config".to_string())?;
            cfg.lossless = 0;
            cfg.quality = self.quality;
            // Fastest method: real-time pacing matters more than size.
            cfg.method = 0;
            let mem = webp::Encoder::from_rgb(&f.data, f.width, f.height)
                .encode_advanced(&cfg)
                .map_err(|e| format!("libwebp: {e:?}"))?;
            let frame = webp_to_vp8(&mem)?;
            Ok(Some(Bytes::copy_from_slice(frame)))
        }

        #[cfg(not(feature = "vp8"))]
        fn encode(&mut self, _f: &RgbFrame, _key: bool) -> Result<Option<Bytes>, String> {
            Err("built without the `vp8` feature".into())
        }
    }
}

/// The 10 ms mono Opus encoder (feature `opus`).
struct AudioEncoder {
    #[cfg(feature = "opus")]
    inner: fastvideo_media::opus::OpusEncoder,
}

impl AudioEncoder {
    fn new() -> Result<Self, String> {
        #[cfg(feature = "opus")]
        {
            let inner =
                fastvideo_media::opus::OpusEncoder::new(fastvideo_media::opus::OpusConfig::reactor(), 0)
                    .map_err(|e| e.to_string())?;
            Ok(Self { inner })
        }
        #[cfg(not(feature = "opus"))]
        {
            Err("built without the `opus` feature of fastvideo-reactor".into())
        }
    }

    #[allow(unused_variables)]
    fn encode(&mut self, frame: &[f32]) -> Result<Bytes, String> {
        #[cfg(feature = "opus")]
        {
            self.inner.encode_frame(frame).map_err(|e| e.to_string())
        }
        #[cfg(not(feature = "opus"))]
        {
            Err("no opus".into())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn webp_container_is_unwrapped() {
        let mut f = b"RIFF\0\0\0\0WEBPVP8 ".to_vec();
        f.extend_from_slice(&3u32.to_le_bytes());
        f.extend_from_slice(&[0x10, 0x02, 0x00, 0xff]);
        assert_eq!(vp8::webp_to_vp8(&f).unwrap(), &[0x10, 0x02, 0x00]);
        assert!(vp8::is_keyframe(&[0x10]));
        assert!(!vp8::is_keyframe(&[0x11]));
        assert!(vp8::webp_to_vp8(b"RIFF\0\0\0\0WEBPVP8L\0\0\0\0").is_err());
        assert!(vp8::webp_to_vp8(b"nope").is_err());
    }

    #[cfg(feature = "vp8")]
    #[test]
    fn vp8_frames_are_keyframes_with_the_right_size() {
        let mut e = vp8::Vp8Encoder::new(64, 48, 60.0).unwrap();
        let data = e.encode(&RgbFrame::black(64, 48, 0), false).unwrap().unwrap();
        assert!(vp8::is_keyframe(&data));
        // RFC 6386 §9.1: start code 9d 01 2a, then 14-bit width and height.
        assert_eq!(&data[3..6], &[0x9d, 0x01, 0x2a]);
        let w = u16::from_le_bytes([data[6], data[7]]) & 0x3fff;
        let h = u16::from_le_bytes([data[8], data[9]]) & 0x3fff;
        assert_eq!((w, h), (64, 48));
    }

    #[test]
    fn sendable_codecs_follow_the_build() {
        let c = sendable_codecs(H264Backend::Off);
        assert_eq!(c.contains(&VideoCodec::Vp8), cfg!(feature = "vp8"));
        assert!(!c.contains(&VideoCodec::H264));
    }
}
