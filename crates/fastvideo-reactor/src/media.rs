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
//!   inter-frame through ffmpeg `libvpx` ([`fastvideo_media::vp8`]), or
//!   intra-only through libwebp when ffmpeg has no libvpx) and sent to every
//!   peer of that codec at the tick's 90 kHz RTP time (a pipe encoder that
//!   hands a frame back late keeps that frame's own timestamp). A repeated frame is not re-sent (the client holds
//!   its last frame, as RT's pacer), unless the held picture changed (the
//!   pacer's flush to black after a clip ends with nothing armed, or on a cut).
//! - **Audio thread**: RT's feeder. Tick audio goes into a ≤200 ms buffer;
//!   exactly one 10 ms / 480-sample Opus frame per 10 ms goes out, silence
//!   when the buffer is short, so the audio RTP clock stays on wall time.
//! - **Black frame at connection start** (reactor §4.2): a new peer, a
//!   resumed video track, a PLI/FIR or a dropped tick ([`MediaPipeline::kick`])
//!   re-sends the current picture as a keyframe; before the first frame the
//!   current picture is black. PLI/FIR ([`MediaPipeline::request_keyframe`])
//!   is rate-limited to one keyframe per [`KEYFRAME_MIN_INTERVAL`] per codec
//!   (a later request waits for the window, it is not dropped).
//! - **Warm spare**: a keyframe restarts a pipe encoder's ffmpeg (NVENC,
//!   libvpx). The session keeps one pre-started, primed spare process
//!   ([`fastvideo_media::pipe::SparePool`]): pre-warmed at session start for
//!   the canvas in the preferred codec (the first encoder adopts it), then
//!   refilled after every (re)start. A restart swaps it in while the old
//!   process flushes its last frame after EOF; the spare ends with the
//!   session.

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

use bytes::Bytes;
use fastvideo_engine_service::{Tick, TickReceiver};
use fastvideo_media::pacer::VideoOut;
use fastvideo_media::pipe::{SparePool, SpareSpec};
use fastvideo_media::queue::GapKeyframes;
use fastvideo_protocol::RgbFrame;
use fastvideo_webrtc::host::PeerHandle;
use fastvideo_webrtc::writer::{AudioPacket, TrackKind, VideoCodec, VideoFrame};

/// 10 ms at 48 kHz (RT `_push_audio_frame`).
pub const AUDIO_FRAME_SAMPLES: usize = 480;
/// RT's per-track audio buffer cap (200 ms).
pub const AUDIO_BUFFER_SAMPLES: usize = 9600;
/// PLI/FIR keyframes: at most one per codec per second (design §5.1).
pub const KEYFRAME_MIN_INTERVAL: Duration = Duration::from_secs(1);

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
/// its backend is usable, then VP8 (ffmpeg `libvpx`, else libwebp intra-only
/// with feature `vp8`).
pub fn sendable_codecs(h264: H264Backend) -> Vec<VideoCodec> {
    let mut v = Vec::new();
    if h264.usable() {
        v.push(VideoCodec::H264);
    }
    if cfg!(feature = "vp8") || fastvideo_media::vp8::libvpx_available() {
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
    /// H.264 and VP8 (libvpx) target bitrate (`None`: `fastvideo-media`'s
    /// default per canvas).
    pub bitrate_bps: Option<u32>,
    /// Quality 0..100 of the intra-only libwebp VP8 fallback (used only
    /// when ffmpeg has no `libvpx`).
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
    /// PLI/FIR requests held back by the one-per-second limit.
    pub keyframes_limited: AtomicU64,
    /// Encoded video bytes (once per codec, before fan-out).
    pub video_bytes: AtomicU64,
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
    /// PLI/FIR: pending request and the last one served, per codec.
    requests: Mutex<HashMap<VideoCodec, (bool, Option<Instant>)>>,
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
        Self::start_with_spares(cfg, ticks, SparePool::per_session())
    }

    /// [`Self::start`] with the session's encoder spare pool (possibly
    /// warmed before the session started); pre-warms it unless it already
    /// holds a spare. The pool's spare ends with the session.
    pub fn start_with_spares(cfg: MediaConfig, ticks: TickReceiver, spares: SparePool) -> Self {
        let sh = Arc::new(Shared {
            cfg,
            peers: Mutex::new(HashMap::new()),
            kicks: Mutex::new(HashSet::new()),
            requests: Mutex::new(HashMap::new()),
            audio: Mutex::new(VecDeque::new()),
            closed: AtomicBool::new(false),
            wake: (Mutex::new(()), Condvar::new()),
            stats: MediaStats::default(),
            on_first_frame: Mutex::new(None),
        });
        let v = sh.clone();
        let rt = tokio::runtime::Handle::current();
        // Pre-warm an encoder for the session's canvas in the codec answers
        // prefer, while the peer connects (codec probes included), so the
        // first video frame does not wait for ffmpeg to start.
        if spares.spare_state().is_none() {
            prewarm(&spares, &sh.cfg);
        }
        std::thread::Builder::new()
            .name("reactor-video".into())
            .spawn(move || video_loop(v, ticks, rt, spares))
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

    /// A PLI/FIR from a peer of `codec`: like [`Self::kick`], but at most one
    /// keyframe per [`KEYFRAME_MIN_INTERVAL`] per codec; a request inside the
    /// window is served when the window ends.
    pub fn request_keyframe(&self, codec: VideoCodec) {
        lock(&self.sh.requests).entry(codec).or_insert((false, None)).0 = true;
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
    /// Per codec: whether a gap in the ticks still needs a keyframe.
    gaps: HashMap<VideoCodec, GapKeyframes>,
    /// The session's warm ffmpeg spare (one, shared by the pipe encoders):
    /// the first encoder and keyframe restarts take it instead of starting
    /// ffmpeg. Killed when the video thread ends (session end).
    spares: SparePool,
}

fn video_loop(sh: Arc<Shared>, mut ticks: TickReceiver, rt: tokio::runtime::Handle, spares: SparePool) {
    let step = 90_000 / u64::from(sh.cfg.fps.max(1));
    let mut st = VideoState {
        encoders: HashMap::new(),
        current: None,
        last_rtp: 0,
        first_sent: false,
        gaps: HashMap::new(),
        spares,
    };
    while !sh.closed.load(Ordering::Relaxed) {
        let next = rt.block_on(async { tokio::time::timeout(Duration::from_millis(50), ticks.recv()).await });
        let tick: Option<Tick> = match next {
            Ok(Some(t)) => Some(t),
            // The pacer ended (session closed).
            Ok(None) => break,
            Err(_) => None,
        };
        flush_ready(&sh, &mut st);
        let mut kicks: HashSet<VideoCodec> = std::mem::take(&mut *lock(&sh.kicks));
        kicks.extend(due_requests(&sh, Instant::now()));
        if ticks.take_dropped() {
            // Frames were lost before the encoder: re-sync every codec,
            // unless a keyframe is on its way or just went out (a pipe
            // encoder restarts ffmpeg for a keyframe; frames dropped while
            // it starts must not restart it again).
            let now = Instant::now();
            let gaps = &st.gaps;
            kicks.extend(
                lock(&sh.peers)
                    .values()
                    .filter_map(|p| p.codec)
                    .filter(|c| gaps.get(c).is_none_or(|g| g.gap_needs_keyframe(now))),
            );
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

/// The codecs whose pending PLI/FIR may be served at `now` (marks them
/// served).
fn due_requests(sh: &Shared, now: Instant) -> Vec<VideoCodec> {
    let mut due = Vec::new();
    for (codec, (pending, last)) in lock(&sh.requests).iter_mut() {
        if !*pending {
            continue;
        }
        if last.is_some_and(|t| now.duration_since(t) < KEYFRAME_MIN_INTERVAL) {
            sh.stats.keyframes_limited.fetch_add(1, Ordering::Relaxed);
            continue;
        }
        *pending = false;
        *last = Some(now);
        due.push(*codec);
    }
    due
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
        let gaps = st.gaps.entry(codec).or_default();
        if st.encoders.get(&codec).is_some_and(|e| e.dims() != (frame.width, frame.height)) {
            // Dropped first: its ffmpeg processes (and spare) end here.
            st.encoders.remove(&codec);
        }
        let enc = match st.encoders.get_mut(&codec) {
            Some(e) => e,
            None => match new_encoder(codec, &sh.cfg, frame.width, frame.height, &st.spares) {
                Ok(e) => {
                    // A new encoder starts with a keyframe.
                    gaps.forced();
                    st.encoders.entry(codec).insert_entry(e).into_mut()
                }
                Err(e) => {
                    tracing::warn!(?codec, error = %e, "video encoder unavailable");
                    sh.stats.encode_errors.fetch_add(1, Ordering::Relaxed);
                    continue;
                }
            },
        };
        let key = all_key || kicks.contains(&codec);
        if key {
            gaps.forced();
        }
        if kicks.contains(&codec) {
            sh.stats.keyframes_forced.fetch_add(1, Ordering::Relaxed);
        }
        match enc.encode(&frame, key, rtp) {
            Ok(out) => {
                if fan_out(sh, &peers, codec, out, black) {
                    gaps.sent(Instant::now());
                }
            }
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

/// Sends encoded frames to every peer of `codec`; `true` if one was a
/// keyframe.
fn fan_out(sh: &Shared, peers: &[(PeerHandle, VideoCodec)], codec: VideoCodec, out: Vec<(u64, Bytes)>, black: bool) -> bool {
    let mut key = false;
    for (t, data) in out {
        key |= match codec {
            VideoCodec::H264 => fastvideo_media::h264::is_idr(&data),
            VideoCodec::Vp8 => fastvideo_media::vp8::is_keyframe(&data),
        };
        sh.stats.frames_encoded.fetch_add(1, Ordering::Relaxed);
        sh.stats.video_bytes.fetch_add(data.len() as u64, Ordering::Relaxed);
        if black {
            sh.stats.black_frames.fetch_add(1, Ordering::Relaxed);
        }
        for (p, c) in peers {
            if *c == codec && p.send_video(VideoFrame::new(data.clone(), t)).is_ok() {
                sh.stats.frames_sent.fetch_add(1, Ordering::Relaxed);
            }
        }
    }
    key
}

/// Sends what the pipe encoders finished since the last encode, so the last
/// frame before a hold does not wait for the next fresh frame.
fn flush_ready(sh: &Shared, st: &mut VideoState) {
    if st.encoders.is_empty() {
        return;
    }
    let peers: Vec<(PeerHandle, VideoCodec)> = lock(&sh.peers)
        .values()
        .filter_map(|p| p.codec.map(|c| (p.handle.clone(), c)))
        .collect();
    let black = st.current.as_ref().is_some_and(|f| f.data.iter().step_by(97).all(|b| *b == 0));
    let mut failed = Vec::new();
    for (codec, enc) in st.encoders.iter_mut() {
        match enc.poll() {
            Ok(out) => {
                if fan_out(sh, &peers, *codec, out, black) {
                    st.gaps.entry(*codec).or_default().sent(Instant::now());
                }
            }
            Err(e) => {
                sh.stats.encode_errors.fetch_add(1, Ordering::Relaxed);
                tracing::warn!(?codec, error = %e, "video encode failed");
                failed.push(*codec);
            }
        }
    }
    for c in failed {
        st.encoders.remove(&c);
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
    /// Encode `f` (RTP time `rtp`); `key` forces a keyframe. Returns the
    /// frames that became ready with their own RTP times, in order (none
    /// yet, this one, or an earlier one first from a lagging pipe encoder).
    fn encode(&mut self, f: &RgbFrame, key: bool, rtp: u64) -> Result<Vec<(u64, Bytes)>, String>;
    /// Frames that became ready since the last call (pipe encoders).
    fn poll(&mut self) -> Result<Vec<(u64, Bytes)>, String> {
        Ok(Vec::new())
    }
}

/// Pre-warms `pool` for a session with `cfg`, on a background thread.
pub(crate) fn prewarm(pool: &SparePool, cfg: &MediaConfig) {
    let c = cfg.clone();
    pool.prewarm_with(move || prewarm_spec(&c));
}

/// The pipe encoder the session most likely needs first (the codec answers
/// prefer, at the canvas), or `None` for an in-process one.
fn prewarm_spec(cfg: &MediaConfig) -> Option<SpareSpec> {
    let (w, h) = cfg.canvas;
    if cfg.h264 == H264Backend::Nvenc && cfg.h264.usable() {
        let c = h264_config(cfg, w, h);
        return SpareSpec::h264(fastvideo_media::video::FfmpegH264::Nvenc, &c).ok();
    }
    if cfg.h264 == H264Backend::OpenH264 && cfg.h264.usable() {
        return None;
    }
    fastvideo_media::vp8::libvpx_available().then(|| SpareSpec::vp8(&vp8_config(cfg, w, h)).ok()).flatten()
}

fn h264_config(cfg: &MediaConfig, w: u32, h: u32) -> fastvideo_media::video::H264Config {
    use fastvideo_media::video::{H264Config, PublishTarget};
    let mut c = H264Config::for_publish(PublishTarget::Peer, w, h, cfg.fps.max(1));
    if let Some(b) = cfg.bitrate_bps {
        c.bitrate_bps = b;
    }
    c
}

fn vp8_config(cfg: &MediaConfig, w: u32, h: u32) -> fastvideo_media::vp8::Vp8Config {
    let mut c = fastvideo_media::vp8::Vp8Config::new(w, h, cfg.fps.max(1));
    if let Some(b) = cfg.bitrate_bps {
        c.bitrate_bps = b;
    }
    c
}

fn new_encoder(
    codec: VideoCodec,
    cfg: &MediaConfig,
    w: u32,
    h: u32,
    spares: &SparePool,
) -> Result<Box<dyn FrameEncoder>, String> {
    match codec {
        VideoCodec::H264 => H264Encoder::new(cfg, w, h, spares).map(|e| Box::new(e) as Box<dyn FrameEncoder>),
        VideoCodec::Vp8 if fastvideo_media::vp8::libvpx_available() => {
            LibvpxEncoder::new(cfg, w, h, spares).map(|e| Box::new(e) as Box<dyn FrameEncoder>)
        }
        VideoCodec::Vp8 => vp8::Vp8Encoder::new(w, h, cfg.vp8_quality).map(|e| Box::new(e) as Box<dyn FrameEncoder>),
    }
}

/// Pairs the frames an encoder hands back with the RTP times they went in
/// with (no B-frames: one frame out per frame in, in order).
#[derive(Default)]
struct RtpFifo(VecDeque<u64>);

impl RtpFifo {
    /// Frame `rtp` went in and `out` came back.
    fn stamp(&mut self, rtp: u64, out: impl IntoIterator<Item = Bytes>) -> Vec<(u64, Bytes)> {
        self.0.push_back(rtp);
        self.take(out)
    }

    /// `out` came back (nothing went in).
    fn take(&mut self, out: impl IntoIterator<Item = Bytes>) -> Vec<(u64, Bytes)> {
        out.into_iter().filter_map(|d| self.0.pop_front().map(|t| (t, d))).collect()
    }
}

/// Inter-frame VP8 through ffmpeg `libvpx`.
struct LibvpxEncoder {
    inner: fastvideo_media::vp8::Vp8Encoder,
    rtps: RtpFifo,
}

impl LibvpxEncoder {
    fn new(cfg: &MediaConfig, w: u32, h: u32, spares: &SparePool) -> Result<Self, String> {
        let c = vp8_config(cfg, w, h);
        let inner = fastvideo_media::vp8::Vp8Encoder::with_pool(c, Some(spares.clone())).map_err(|e| e.to_string())?;
        Ok(Self { inner, rtps: RtpFifo::default() })
    }
}

impl FrameEncoder for LibvpxEncoder {
    fn dims(&self) -> (u32, u32) {
        (self.inner.config().width, self.inner.config().height)
    }
    fn encode(&mut self, f: &RgbFrame, key: bool, rtp: u64) -> Result<Vec<(u64, Bytes)>, String> {
        if key {
            self.inner.force_keyframe();
        }
        let out = self.inner.encode(f).map_err(|e| e.to_string())?;
        Ok(self.rtps.stamp(rtp, out.into_iter().map(|e| e.data)))
    }
    fn poll(&mut self) -> Result<Vec<(u64, Bytes)>, String> {
        let out = self.inner.poll().map_err(|e| e.to_string())?;
        Ok(self.rtps.take(out.into_iter().map(|e| e.data)))
    }
}

struct H264Encoder {
    dims: (u32, u32),
    inner: Box<dyn fastvideo_media::video::VideoEncoder>,
    rtps: RtpFifo,
}

impl H264Encoder {
    fn new(cfg: &MediaConfig, w: u32, h: u32, spares: &SparePool) -> Result<Self, String> {
        use fastvideo_media::video::{create_stream_encoder, EncoderBackend};
        let backend = match cfg.h264 {
            H264Backend::Nvenc => EncoderBackend::Nvenc,
            H264Backend::OpenH264 => EncoderBackend::OpenH264,
            H264Backend::Off => return Err("H.264 is off".into()),
        };
        let inner = create_stream_encoder(backend, h264_config(cfg, w, h), spares).map_err(|e| e.to_string())?;
        Ok(Self { dims: (w, h), inner, rtps: RtpFifo::default() })
    }
}

impl FrameEncoder for H264Encoder {
    fn dims(&self) -> (u32, u32) {
        self.dims
    }
    fn encode(&mut self, f: &RgbFrame, key: bool, rtp: u64) -> Result<Vec<(u64, Bytes)>, String> {
        if key {
            self.inner.force_idr();
        }
        let out = self.inner.encode(f).map_err(|e| e.to_string())?;
        // Every AU goes out: dropping one would break the references of the
        // P-frames after it.
        Ok(self.rtps.stamp(rtp, out.into_iter().map(|e| e.data)))
    }
    fn poll(&mut self) -> Result<Vec<(u64, Bytes)>, String> {
        let out = self.inner.poll().map_err(|e| e.to_string())?;
        Ok(self.rtps.take(out.into_iter().map(|e| e.data)))
    }
}

pub mod vp8 {
    //! Intra-only VP8, the fallback when ffmpeg has no `libvpx`: every frame
    //! is a keyframe, encoded by libwebp (a lossy WebP image *is* one VP8 key
    //! frame in a RIFF container). Much heavier on bitrate than the
    //! inter-frame libvpx encoder, but it needs no external tool, and every
    //! WebRTC stack decodes VP8, including the Python reactor_sdk's
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
        fn encode(&mut self, f: &RgbFrame, _key: bool, rtp: u64) -> Result<Vec<(u64, Bytes)>, String> {
            let mut cfg = webp::WebPConfig::new().map_err(|_| "libwebp config".to_string())?;
            cfg.lossless = 0;
            cfg.quality = self.quality;
            // Fastest method: real-time pacing matters more than size.
            cfg.method = 0;
            let mem = webp::Encoder::from_rgb(&f.data, f.width, f.height)
                .encode_advanced(&cfg)
                .map_err(|e| format!("libwebp: {e:?}"))?;
            let frame = webp_to_vp8(&mem)?;
            Ok(vec![(rtp, Bytes::copy_from_slice(frame))])
        }

        #[cfg(not(feature = "vp8"))]
        fn encode(&mut self, _f: &RgbFrame, _key: bool, _rtp: u64) -> Result<Vec<(u64, Bytes)>, String> {
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
        let (_, data) = e.encode(&RgbFrame::black(64, 48, 0), false, 0).unwrap().remove(0);
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
        assert_eq!(c.contains(&VideoCodec::Vp8), cfg!(feature = "vp8") || fastvideo_media::vp8::libvpx_available());
        assert!(!c.contains(&VideoCodec::H264));
    }

    #[test]
    fn pli_keyframes_are_limited_to_one_per_second_per_codec() {
        let sh = Shared {
            cfg: MediaConfig::new(24, false, (64, 48)),
            peers: Mutex::new(HashMap::new()),
            kicks: Mutex::new(HashSet::new()),
            requests: Mutex::new(HashMap::new()),
            audio: Mutex::new(VecDeque::new()),
            closed: AtomicBool::new(false),
            wake: (Mutex::new(()), Condvar::new()),
            stats: MediaStats::default(),
            on_first_frame: Mutex::new(None),
        };
        let request = |c| lock(&sh.requests).entry(c).or_insert((false, None)).0 = true;
        let t0 = Instant::now();
        assert!(due_requests(&sh, t0).is_empty());
        request(VideoCodec::Vp8);
        assert_eq!(due_requests(&sh, t0), vec![VideoCodec::Vp8]);
        assert!(due_requests(&sh, t0).is_empty(), "served once");
        // A burst inside the window waits for it, then is served once.
        request(VideoCodec::Vp8);
        request(VideoCodec::Vp8);
        request(VideoCodec::H264);
        assert_eq!(due_requests(&sh, t0 + Duration::from_millis(500)), vec![VideoCodec::H264]);
        assert!(due_requests(&sh, t0 + Duration::from_millis(999)).is_empty());
        assert_eq!(due_requests(&sh, t0 + KEYFRAME_MIN_INTERVAL), vec![VideoCodec::Vp8]);
        assert!(due_requests(&sh, t0 + Duration::from_secs(5)).is_empty());
        assert_eq!(sh.stats.keyframes_limited.load(Ordering::Relaxed), 2);
    }
}
