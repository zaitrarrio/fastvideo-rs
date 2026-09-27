//! Per-session media: pacing, encode-once and fan-out to peers (design §5.1,
//! §5.3; reactor §4.2, §4.5).
//!
//! RT's model: the model emits bundles (one video frame plus that frame's
//! audio slice); a pacer releases **one bundle per `1/fps` tick**; the
//! bundle's audio is appended to a ≤200 ms per-track buffer; a separate
//! feeder pushes **exactly one 10 ms / 480-sample Opus frame per 10 ms
//! tick** for the life of the track, substituting silence when nothing is
//! buffered, so the audio clock stays locked to wall time. A/V sync is
//! co-transport: a frame's audio enters the buffer when the frame is sent.
//!
//! Here:
//!
//! - [`MediaPipeline::push`] queues bundles (bounded, `buffer_frames`, the
//!   producer waits: backpressure, never drops).
//! - The **video thread** ticks at `fps`, pops one item per tick, encodes it
//!   once per negotiated codec ([`VideoCodec::H264`] via `fastvideo-media`,
//!   [`VideoCodec::Vp8`] intra-only via libwebp) and fans the bitstream out.
//!   The 90 kHz RTP clock advances every tick, sent or not; on underrun
//!   nothing is sent and the client holds its last frame.
//! - The **audio thread** runs the 10 ms Opus feeder (48 kHz mono, design
//!   §5.3 Reactor row) when the session has an audio track.
//! - A **black frame** is sent at the start of each connection (when its
//!   video is resumed) and after a flush (reactor §4.2), as an IDR/keyframe.
//! - A PLI/FIR or a resumed track forces a keyframe of the current picture
//!   for that codec ([`MediaPipeline::kick`]).

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

use bytes::Bytes;
use fastvideo_protocol::{RgbFrame, WIRE_AUDIO_RATE};
use fastvideo_webrtc::host::PeerHandle;
use fastvideo_webrtc::writer::{AudioPacket, VideoCodec, VideoFrame};
use tokio::sync::{oneshot, Notify};

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
    /// Whether this build can encode H.264 with it.
    pub fn usable(self) -> bool {
        match self {
            H264Backend::Nvenc => fastvideo_media::video::nvenc_available(),
            H264Backend::OpenH264 => {
                fastvideo_media::video::EncoderBackend::OpenH264.compiled()
            }
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
    pub fps: u32,
    /// The session has an audio track.
    pub audio: bool,
    /// Initial canvas (black frames before the first real frame).
    pub canvas: (u32, u32),
    /// Pacer queue depth in frames (2 s by default).
    pub buffer_frames: usize,
    pub h264: H264Backend,
    /// H.264 target bitrate (`None`: `fastvideo-media` default per canvas).
    pub bitrate_bps: Option<u32>,
    /// VP8 (libwebp) quality 0..100.
    pub vp8_quality: f32,
}

impl MediaConfig {
    pub fn new(fps: u32, audio: bool, canvas: (u32, u32)) -> Self {
        Self {
            fps,
            audio,
            canvas,
            buffer_frames: (fps as usize * 2).max(1),
            h264: H264Backend::Nvenc,
            bitrate_bps: None,
            vp8_quality: 70.0,
        }
    }
}

/// Counters (tests and `/metrics`-style introspection).
#[derive(Debug, Default)]
pub struct MediaStats {
    pub ticks: AtomicU64,
    pub frames_sent: AtomicU64,
    pub black_frames: AtomicU64,
    pub keyframes_forced: AtomicU64,
    pub underruns: AtomicU64,
    pub encode_errors: AtomicU64,
    pub audio_frames: AtomicU64,
    pub audio_silence_frames: AtomicU64,
    pub audio_samples_in: AtomicU64,
}

enum Item {
    Frame(RgbFrame, Option<Vec<f32>>),
    Black,
    Marker(oneshot::Sender<()>),
}

struct PeerSink {
    handle: PeerHandle,
    codec: Option<VideoCodec>,
    audio: bool,
}

struct Shared {
    cfg: MediaConfig,
    queue: Mutex<VecDeque<Item>>,
    space: Notify,
    peers: Mutex<HashMap<u64, PeerSink>>,
    kicks: Mutex<HashSet<VideoCodec>>,
    audio: Mutex<VecDeque<f32>>,
    closed: AtomicBool,
    wake: (Mutex<()>, Condvar),
    stats: MediaStats,
}

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

/// One session's media pipeline. Dropping it stops the threads.
pub struct MediaPipeline {
    sh: Arc<Shared>,
}

impl std::fmt::Debug for MediaPipeline {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MediaPipeline").field("fps", &self.sh.cfg.fps).finish()
    }
}

impl MediaPipeline {
    pub fn start(cfg: MediaConfig) -> Self {
        let sh = Arc::new(Shared {
            cfg,
            queue: Mutex::new(VecDeque::new()),
            space: Notify::new(),
            peers: Mutex::new(HashMap::new()),
            kicks: Mutex::new(HashSet::new()),
            audio: Mutex::new(VecDeque::new()),
            closed: AtomicBool::new(false),
            wake: (Mutex::new(()), Condvar::new()),
            stats: MediaStats::default(),
        });
        let v = sh.clone();
        std::thread::Builder::new()
            .name("reactor-video".into())
            .spawn(move || video_loop(v))
            .expect("spawning the video pacer thread");
        if sh.cfg.audio {
            let a = sh.clone();
            std::thread::Builder::new()
                .name("reactor-audio".into())
                .spawn(move || audio_loop(a))
                .expect("spawning the audio feeder thread");
        }
        Self { sh }
    }

    pub fn fps(&self) -> u32 {
        self.sh.cfg.fps
    }

    pub fn stats(&self) -> &MediaStats {
        &self.sh.stats
    }

    /// Bundles still queued (not yet released by the pacer).
    pub fn queued(&self) -> usize {
        lock(&self.sh.queue).len()
    }

    /// Queue frames with their audio (48 kHz mono; split evenly across the
    /// frames, `48000/fps` each). Waits while the queue is full.
    pub async fn push(&self, frames: Vec<RgbFrame>, audio: Option<Vec<f32>>) {
        let n = frames.len().max(1);
        let spf = WIRE_AUDIO_RATE as usize / self.sh.cfg.fps.max(1) as usize;
        let mut audio = audio.map(VecDeque::from);
        for f in frames {
            let slice = audio.as_mut().map(|a| {
                let take = spf.min(a.len());
                a.drain(..take).collect::<Vec<f32>>()
            });
            self.push_item(Item::Frame(f, slice)).await;
        }
        let _ = n;
    }

    async fn push_item(&self, item: Item) {
        let mut item = Some(item);
        loop {
            // Register for the wakeup before checking, so a pop between the
            // check and the await is not missed.
            let notified = self.sh.space.notified();
            {
                let mut q = lock(&self.sh.queue);
                let data_items = q.iter().filter(|i| matches!(i, Item::Frame(..))).count();
                let is_frame = matches!(item, Some(Item::Frame(..)));
                if !is_frame || data_items < self.sh.cfg.buffer_frames || self.sh.closed.load(Ordering::Relaxed) {
                    q.push_back(item.take().expect("pushed once"));
                    return;
                }
            }
            notified.await;
        }
    }

    /// Resolves once everything queued before it has been released
    /// (clip end: `clip_finished` is sent when playback really ended).
    pub async fn drained(&self) {
        let (tx, rx) = oneshot::channel();
        self.push_item(Item::Marker(tx)).await;
        let _ = rx.await;
    }

    /// Queue one black frame after what is queued (clip end, fast-h3's
    /// `output.flush()` hold on black).
    pub async fn black(&self) {
        self.push_item(Item::Black).await;
    }

    /// Drop everything queued (video and buffered audio) and show black now
    /// (stop, reset).
    pub fn flush(&self) {
        {
            let mut q = lock(&self.sh.queue);
            // Markers still resolve (their waiters see the clip end).
            let markers: Vec<Item> = q.drain(..).filter(|i| matches!(i, Item::Marker(_))).collect();
            q.push_back(Item::Black);
            q.extend(markers);
        }
        lock(&self.sh.audio).clear();
        self.sh.space.notify_waiters();
    }

    /// Adds a peer. Its first video is a keyframe of the current picture
    /// (black before the first frame): the start-of-connection black frame.
    pub fn add_peer(&self, handle: PeerHandle) {
        let codec = handle.video_codec();
        let audio = handle.sends(fastvideo_webrtc::writer::TrackKind::Audio);
        let id = handle.id();
        lock(&self.sh.peers).insert(id, PeerSink { handle, codec, audio });
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

    /// Re-send the current picture as a keyframe for `codec` on the next
    /// tick (PLI/FIR, resumed track, new peer).
    pub fn kick(&self, codec: VideoCodec) {
        lock(&self.sh.kicks).insert(codec);
    }

    pub fn close(&self) {
        self.sh.closed.store(true, Ordering::Relaxed);
        self.sh.space.notify_waiters();
        let (_, cv) = &self.sh.wake;
        cv.notify_all();
        // Resolve pending markers so no waiter hangs.
        lock(&self.sh.queue).clear();
    }
}

impl Drop for MediaPipeline {
    fn drop(&mut self) {
        self.close();
    }
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

fn video_loop(sh: Arc<Shared>) {
    let fps = sh.cfg.fps.max(1);
    let period = Duration::from_secs_f64(1.0 / f64::from(fps));
    let mut encoders: HashMap<VideoCodec, Box<dyn FrameEncoder>> = HashMap::new();
    let mut current: Option<RgbFrame> = None;
    let mut tick: u64 = 0;
    let mut next = Instant::now();
    while !sh.closed.load(Ordering::Relaxed) {
        sleep_until(&sh, next);
        if sh.closed.load(Ordering::Relaxed) {
            break;
        }
        next += period;
        let now = Instant::now();
        if now > next + period * 8 {
            // Far behind (suspended, overloaded): re-anchor, never burst.
            next = now + period;
        }
        let rtp = tick * 90_000 / u64::from(fps);
        tick += 1;
        sh.stats.ticks.fetch_add(1, Ordering::Relaxed);

        // Pop one frame; markers resolve and black frames count as frames.
        let mut out: Option<(RgbFrame, bool)> = None;
        {
            let mut q = lock(&sh.queue);
            while let Some(item) = q.pop_front() {
                match item {
                    Item::Marker(tx) => {
                        let _ = tx.send(());
                    }
                    Item::Black => {
                        let (w, h) = current
                            .as_ref()
                            .map(|f| (f.width, f.height))
                            .unwrap_or(sh.cfg.canvas);
                        out = Some((RgbFrame::black(w, h, 0), true));
                        break;
                    }
                    Item::Frame(f, a) => {
                        if let Some(a) = a {
                            sh.stats.audio_samples_in.fetch_add(a.len() as u64, Ordering::Relaxed);
                            let mut buf = lock(&sh.audio);
                            buf.extend(a);
                            let over = buf.len().saturating_sub(AUDIO_BUFFER_SAMPLES);
                            buf.drain(..over);
                        }
                        out = Some((f, false));
                        break;
                    }
                }
            }
        }
        sh.space.notify_waiters();
        let kicks: HashSet<VideoCodec> = std::mem::take(&mut *lock(&sh.kicks));
        let (frame, black) = match out {
            Some((f, black)) => {
                if black {
                    sh.stats.black_frames.fetch_add(1, Ordering::Relaxed);
                }
                current = Some(f.clone());
                (f, black)
            }
            None if !kicks.is_empty() => {
                let f = current.clone().unwrap_or_else(|| {
                    sh.stats.black_frames.fetch_add(1, Ordering::Relaxed);
                    RgbFrame::black(sh.cfg.canvas.0, sh.cfg.canvas.1, 0)
                });
                (f, true)
            }
            None => {
                sh.stats.underruns.fetch_add(1, Ordering::Relaxed);
                continue;
            }
        };
        // Encode once per codec in use and fan out.
        let peers: Vec<(PeerHandle, VideoCodec)> = lock(&sh.peers)
            .values()
            .filter_map(|p| p.codec.map(|c| (p.handle.clone(), c)))
            .collect();
        let codecs: HashSet<VideoCodec> = peers.iter().map(|(_, c)| *c).collect();
        for codec in codecs {
            let key = black || kicks.contains(&codec);
            if kicks.contains(&codec) {
                sh.stats.keyframes_forced.fetch_add(1, Ordering::Relaxed);
            }
            let enc = match encoders.get_mut(&codec) {
                Some(e) if e.dims() == (frame.width, frame.height) => e,
                _ => match new_encoder(codec, &sh.cfg, frame.width, frame.height) {
                    Ok(e) => {
                        encoders.insert(codec, e);
                        encoders.get_mut(&codec).expect("inserted")
                    }
                    Err(e) => {
                        tracing::warn!(?codec, error = %e, "video encoder unavailable");
                        sh.stats.encode_errors.fetch_add(1, Ordering::Relaxed);
                        continue;
                    }
                },
            };
            match enc.encode(&frame, key) {
                Ok(Some(data)) => {
                    for (p, c) in &peers {
                        if *c == codec {
                            let _ = p.send_video(VideoFrame::new(data.clone(), rtp));
                        }
                    }
                    sh.stats.frames_sent.fetch_add(1, Ordering::Relaxed);
                }
                Ok(None) => {}
                Err(e) => {
                    sh.stats.encode_errors.fetch_add(1, Ordering::Relaxed);
                    tracing::warn!(?codec, error = %e, "video encode failed");
                    encoders.remove(&codec);
                }
            }
        }
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
        let mut due = 0;
        let now = Instant::now();
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
            let peers: Vec<PeerHandle> = lock(&sh.peers)
                .values()
                .filter(|p| p.audio)
                .map(|p| p.handle.clone())
                .collect();
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

fn new_encoder(
    codec: VideoCodec,
    cfg: &MediaConfig,
    w: u32,
    h: u32,
) -> Result<Box<dyn FrameEncoder>, String> {
    match codec {
        VideoCodec::H264 => H264Encoder::new(cfg, w, h).map(|e| Box::new(e) as Box<dyn FrameEncoder>),
        VideoCodec::Vp8 => vp8::Vp8Encoder::new(w, h, cfg.vp8_quality)
            .map(|e| Box::new(e) as Box<dyn FrameEncoder>),
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
        let mut c = H264Config::for_publish(PublishTarget::Peer, w, h, cfg.fps);
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
    //! on bitrate than a real inter-frame encoder, but it needs no system
    //! library and every WebRTC stack decodes VP8, including the Python
    //! reactor_sdk's libwebrtc, which offers no H.264.

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
            let inner = fastvideo_media::opus::OpusEncoder::new(
                fastvideo_media::opus::OpusConfig::reactor(),
                0,
            )
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
        let f = RgbFrame::black(64, 48, 0);
        let data = e.encode(&f, false).unwrap().unwrap();
        assert!(vp8::is_keyframe(&data));
        // RFC 6386 §9.1: start code 9d 01 2a then 14-bit width/height.
        assert_eq!(&data[3..6], &[0x9d, 0x01, 0x2a]);
        let w = u16::from_le_bytes([data[6], data[7]]) & 0x3fff;
        let h = u16::from_le_bytes([data[8], data[9]]) & 0x3fff;
        assert_eq!((w, h), (64, 48));
    }

    #[tokio::test]
    async fn pacer_releases_one_frame_per_tick_and_audio_in_lockstep() {
        let cfg = MediaConfig { h264: H264Backend::Off, ..MediaConfig::new(50, true, (16, 16)) };
        let m = MediaPipeline::start(cfg);
        let frames: Vec<RgbFrame> = (0..10).map(|i| RgbFrame::black(16, 16, i)).collect();
        // 960 samples per frame at 50 fps.
        m.push(frames, Some(vec![0.25; 9600])).await;
        let t0 = Instant::now();
        m.drained().await;
        let el = t0.elapsed();
        // 10 frames at 20 ms: about 200 ms, never a burst.
        assert!(el >= Duration::from_millis(150), "{el:?}");
        assert_eq!(m.stats().audio_samples_in.load(Ordering::Relaxed), 9600);
        m.flush();
        tokio::time::sleep(Duration::from_millis(60)).await;
        assert!(m.stats().black_frames.load(Ordering::Relaxed) >= 1);
        assert!(m.stats().audio_frames.load(Ordering::Relaxed) >= 10);
    }

    #[tokio::test]
    async fn push_waits_when_the_queue_is_full() {
        let cfg = MediaConfig { h264: H264Backend::Off, buffer_frames: 2, ..MediaConfig::new(20, false, (8, 8)) };
        let m = Arc::new(MediaPipeline::start(cfg));
        let t0 = Instant::now();
        m.push((0..6).map(|i| RgbFrame::black(8, 8, i)).collect(), None).await;
        // 6 frames with room for 2 at 50 ms per tick: at least ~150 ms.
        assert!(t0.elapsed() >= Duration::from_millis(120), "{:?}", t0.elapsed());
    }
}
