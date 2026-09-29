//! Director playout: built chunks → lockstep A/V ticks → encoders → the
//! peer (design §5.1, §5.5).
//!
//! One OS thread per session (encoding never runs on the async runtime or
//! the engine executor). It owns:
//!
//! - the chunk queue, fed by the session in [`PreparedChunk`]s whose audio
//!   is already 48 kHz stereo, exactly `frames · 48000/fps` samples long;
//! - an [`AvPacer`]: every tick is one video frame plus exactly `48000/fps`
//!   audio samples, refilled in 3-frame lockstep slices; on underrun video
//!   holds the last frame and audio is silence, so the audio clock never
//!   stops (freeze-and-silence, the WMA `deadline_missed` behaviour);
//! - a re-anchoring [`Metronome`] at the model fps (never bursts);
//! - Opus, with RTP times from the sample counter.
//!
//! Video is encoded on a second thread (H.264, or VP8 for offers without
//! H.264: inter-frame through ffmpeg `libvpx`, intra-only libwebp when
//! ffmpeg has no libvpx) fed through a 10-tick drop-oldest queue, so a slow or
//! restarting encoder never stalls the clock or the audio (design §5.10:
//! "encoder input 10 ticks, drop-oldest, then force IDR"). RTP times come
//! from the tick counter; PLI/FIR is rate limited to one keyframe a second.
//!
//! It reports chunk starts (with lateness after an underrun), underruns and
//! emitted video time back to the session.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{Receiver, RecvTimeoutError, Sender};
use std::sync::Arc;
use std::time::{Duration, Instant};

use bytes::Bytes;
use fastvideo_media::lockstep::{self, Slice, EMIT_FRAMES, WIRE_RATE};
use fastvideo_media::opus::{OpusConfig, OpusEncoder};
use fastvideo_media::pacer::{AvPacer, AvPacerConfig, Metronome, VideoOut};
use fastvideo_media::queue::{DropOldest, GapKeyframes};
use fastvideo_media::video::{create_encoder, EncoderBackend, H264Config, PublishTarget, VideoEncoder};
use fastvideo_media::RgbFrame;
use fastvideo_webrtc::writer::KeyframeLimiter;

use super::vp8::{quality_for, Vp8Encoder};

/// The negotiated video codec.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VideoCodec {
    H264,
    /// VP8 for offers without H.264 (`vp8_fallback`): ffmpeg `libvpx`, else
    /// intra-only libwebp.
    Vp8,
}

/// Where encoded media goes (the WebRTC peer; a recorder in tests). Cloned
/// once: one copy per encoder thread.
pub trait MediaSink: Clone + Send + 'static {
    /// One encoded access unit / frame with its 90 kHz RTP time.
    fn video(&mut self, data: Bytes, rtp_time: u64, keyframe: bool);
    /// One Opus packet with its 48 kHz RTP time.
    fn audio(&mut self, data: Bytes, rtp_time: u64);
}

/// Media settings of one session.
#[derive(Clone, Debug)]
pub struct MediaConfig {
    pub fps: u32,
    /// Stereo 48 kHz audio lane; `false` for a video-only model.
    pub audio: bool,
    pub codec: VideoCodec,
    /// H.264 backend (NVENC in production; OpenH264 or x264 in tests).
    pub h264: EncoderBackend,
    /// Video bitrate; `None`: by canvas (§5.1).
    pub video_bitrate: Option<u32>,
    pub gop_seconds: f32,
}

/// A built chunk ready for playout.
#[derive(Clone, Debug)]
pub struct PreparedChunk {
    pub index: u32,
    pub frames: Vec<RgbFrame>,
    /// Interleaved stereo at 48 kHz, `frames.len() · 48000/fps · 2` long.
    pub audio: Option<Vec<f32>>,
}

pub enum MediaCmd {
    Chunk(PreparedChunk),
    /// PLI/FIR or a new connection: force a keyframe (rate limited).
    Keyframe,
    /// Opus target bitrate (`configure.audio_bitrate`).
    AudioBitrate(u32),
    Stop,
}

#[derive(Clone, Debug, PartialEq)]
pub enum MediaEvent {
    /// The chunk's first frame entered playout; `late_by` after an underrun.
    ChunkStarted { index: u32, late_by: Option<f64> },
    /// Playout ran dry after a chunk ended (freeze + silence).
    Underrun { after: u32 },
    /// Video time emitted so far (fresh frames / fps), about once a second.
    Emitted { video_seconds: f64 },
    /// The encoder failed; the session ends.
    Failed(String),
}

/// Counters the session reads for `chunk` metrics.
#[derive(Debug, Default)]
pub struct MediaGauges {
    /// Frames queued for playout (not yet emitted), across chunks.
    pub buffered_frames: AtomicU64,
    /// Chunks queued whose first frame has not played yet.
    pub pending_chunks: AtomicU64,
    pub fresh_frames: AtomicU64,
    pub ticks: AtomicU64,
    pub video_packets: AtomicU64,
    pub audio_packets: AtomicU64,
    pub encode_us: AtomicU64,
    /// Frames the video encoder fell behind on (dropped oldest).
    pub video_dropped: AtomicU64,
}

/// Encoder input depth in ticks (design §5.10).
const VIDEO_QUEUE: usize = 10;

struct Playing {
    chunk: PreparedChunk,
    slices: Vec<Slice>,
    next: usize,
    started: bool,
}

enum Video {
    None,
    H264(Box<dyn VideoEncoder>),
    Vp8(Vp8Encoder),
    /// Inter-frame VP8 through ffmpeg `libvpx` (when ffmpeg has it).
    Libvpx(fastvideo_media::vp8::Vp8Encoder),
}

/// The playout thread's state.
pub struct MediaLoop<S: MediaSink> {
    cfg: MediaConfig,
    sink: S,
    rx: Receiver<MediaCmd>,
    events: tokio::sync::mpsc::UnboundedSender<MediaEvent>,
    gauges: Arc<MediaGauges>,
    queue: VecDeque<Playing>,
    pacer: AvPacer<RgbFrame>,
    video_q: Arc<DropOldest<(RgbFrame, u64)>>,
    keyframe: Arc<AtomicBool>,
    video_failed: Arc<AtomicBool>,
    opus: Option<OpusEncoder>,
    audio_rtp: u64,
    keyframes: KeyframeLimiter,
    underrun_since: Option<Instant>,
    played_any: bool,
    last_index: u32,
}

impl<S: MediaSink> MediaLoop<S> {
    pub fn new(
        cfg: MediaConfig,
        sink: S,
        rx: Receiver<MediaCmd>,
        events: tokio::sync::mpsc::UnboundedSender<MediaEvent>,
        gauges: Arc<MediaGauges>,
    ) -> Result<Self, String> {
        let pacer = AvPacer::new(AvPacerConfig::wire(cfg.fps, cfg.audio.then_some(2))).map_err(|e| e.to_string())?;
        let opus = if cfg.audio {
            Some(OpusEncoder::new(OpusConfig::wma(None), 0).map_err(|e| e.to_string())?)
        } else {
            None
        };
        Ok(Self {
            cfg,
            sink,
            rx,
            events,
            gauges,
            queue: VecDeque::new(),
            pacer,
            video_q: Arc::new(DropOldest::new(VIDEO_QUEUE)),
            keyframe: Arc::new(AtomicBool::new(false)),
            video_failed: Arc::new(AtomicBool::new(false)),
            opus,
            audio_rtp: 0,
            keyframes: KeyframeLimiter::default(),
            underrun_since: None,
            played_any: false,
            last_index: 0,
        })
    }

    /// Spawns the playout thread and its video encoder thread.
    pub fn spawn(self) -> std::io::Result<std::thread::JoinHandle<()>> {
        let enc = VideoThread {
            cfg: self.cfg.clone(),
            sink: self.sink.clone(),
            q: self.video_q.clone(),
            keyframe: self.keyframe.clone(),
            failed: self.video_failed.clone(),
            events: self.events.clone(),
            gauges: self.gauges.clone(),
            video: Video::None,
            pending_rtp: VecDeque::new(),
            gaps: GapKeyframes::default(),
        };
        std::thread::Builder::new().name("wma-video-enc".into()).spawn(move || enc.run())?;
        std::thread::Builder::new().name("wma-playout".into()).spawn(move || self.run())
    }

    fn emit(&self, e: MediaEvent) {
        let _ = self.events.send(e);
    }

    /// Returns `false` on `Stop` or when the session is gone.
    fn handle(&mut self, cmd: MediaCmd) -> bool {
        match cmd {
            MediaCmd::Chunk(c) => {
                let slices = lockstep::slices(c.frames.len() as u32, EMIT_FRAMES, self.cfg.fps, WIRE_RATE).collect();
                self.gauges.buffered_frames.fetch_add(c.frames.len() as u64, Ordering::Relaxed);
                self.gauges.pending_chunks.fetch_add(1, Ordering::Relaxed);
                self.queue.push_back(Playing { chunk: c, slices, next: 0, started: false });
            }
            MediaCmd::Keyframe => {
                if self.keyframes.request(Instant::now()) {
                    self.force_keyframe();
                }
            }
            MediaCmd::AudioBitrate(b) => {
                if self.cfg.audio {
                    match OpusEncoder::new(OpusConfig::wma(Some(b)), 0) {
                        Ok(o) => self.opus = Some(o),
                        Err(e) => tracing::warn!(error = %e, "opus bitrate change failed"),
                    }
                }
            }
            MediaCmd::Stop => return false,
        }
        true
    }

    fn force_keyframe(&mut self) {
        self.keyframe.store(true, Ordering::Relaxed);
    }

    fn run(mut self) {
        struct CloseOnExit(Arc<DropOldest<(RgbFrame, u64)>>);
        impl Drop for CloseOnExit {
            fn drop(&mut self) {
                self.0.close();
            }
        }
        let _close = CloseOnExit(self.video_q.clone());
        let t0 = Instant::now();
        let mut met = Metronome::new(f64::from(self.cfg.fps));
        let mut since_report = 0u32;
        loop {
            loop {
                match self.rx.try_recv() {
                    Ok(cmd) => {
                        if !self.handle(cmd) {
                            return;
                        }
                    }
                    Err(std::sync::mpsc::TryRecvError::Empty) => break,
                    Err(std::sync::mpsc::TryRecvError::Disconnected) => return,
                }
            }
            if self.keyframes.poll(Instant::now()) {
                self.force_keyframe();
            }
            if let Err(dt) = met.poll(t0.elapsed().as_secs_f64()) {
                match self.rx.recv_timeout(Duration::from_secs_f64(dt.min(0.01))) {
                    Ok(cmd) => {
                        if !self.handle(cmd) {
                            return;
                        }
                    }
                    Err(RecvTimeoutError::Timeout) => {}
                    Err(RecvTimeoutError::Disconnected) => return,
                }
                continue;
            }
            self.refill();
            let tick = self.pacer.tick();
            self.gauges.ticks.fetch_add(1, Ordering::Relaxed);
            let enc_t = Instant::now();
            let fresh = tick.video.is_fresh();
            if fresh {
                self.gauges.fresh_frames.fetch_add(1, Ordering::Relaxed);
                let _ = self.gauges.buffered_frames.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |v| Some(v.saturating_sub(1)));
            }
            let frame = match &tick.video {
                VideoOut::Nothing => None,
                VideoOut::Fresh(f) | VideoOut::Repeat(f) => Some(f),
            };
            if let Some(f) = frame {
                let rtp = fastvideo_webrtc::writer::video_rtp_time(tick.index, self.cfg.fps);
                if self.video_q.push((f.clone(), rtp)) {
                    self.gauges.video_dropped.fetch_add(1, Ordering::Relaxed);
                }
            }
            if self.video_failed.load(Ordering::Relaxed) {
                return;
            }
            if let (Some(samples), Some(opus)) = (&tick.audio, &mut self.opus) {
                match opus.push(samples) {
                    Ok(packets) => {
                        for p in packets {
                            self.sink.audio(p.data, self.audio_rtp);
                            self.audio_rtp += u64::from(p.samples);
                            self.gauges.audio_packets.fetch_add(1, Ordering::Relaxed);
                        }
                    }
                    Err(e) => {
                        self.emit(MediaEvent::Failed(format!("opus: {e}")));
                        return;
                    }
                }
            }
            self.gauges.encode_us.fetch_add(enc_t.elapsed().as_micros() as u64, Ordering::Relaxed);
            since_report += 1;
            if since_report >= self.cfg.fps {
                since_report = 0;
                let fresh = self.gauges.fresh_frames.load(Ordering::Relaxed);
                self.emit(MediaEvent::Emitted { video_seconds: fresh as f64 / f64::from(self.cfg.fps) });
            }
        }
    }

    /// Keeps two lockstep slices in the pacer; starts chunks and detects
    /// underruns.
    fn refill(&mut self) {
        let spf = WIRE_RATE / self.cfg.fps.max(1);
        while self.pacer.buffered_frames() < (2 * EMIT_FRAMES) as usize {
            let Some(p) = self.queue.front_mut() else {
                if self.played_any && self.pacer.buffered_frames() == 0 && self.underrun_since.is_none() {
                    self.underrun_since = Some(Instant::now());
                    let _ = self.events.send(MediaEvent::Underrun { after: self.last_index });
                }
                return;
            };
            if !p.started {
                p.started = true;
                let late_by = self.underrun_since.take().map(|t| t.elapsed().as_secs_f64());
                let index = p.chunk.index;
                self.last_index = index;
                self.played_any = true;
                let _ = self.gauges.pending_chunks.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |v| Some(v.saturating_sub(1)));
                let _ = self.events.send(MediaEvent::ChunkStarted { index, late_by });
            }
            let Some(s) = p.slices.get(p.next).copied() else {
                self.queue.pop_front();
                continue;
            };
            p.next += 1;
            let frames = p.chunk.frames[s.frame_lo as usize..s.frame_hi as usize].to_vec();
            let audio: Vec<f32> = match &p.chunk.audio {
                Some(a) => {
                    let lo = (s.frame_lo * spf * 2) as usize;
                    let hi = ((s.frame_hi * spf * 2) as usize).min(a.len());
                    a[lo.min(hi)..hi].to_vec()
                }
                None => Vec::new(),
            };
            if let Err(e) = self.pacer.push_slice(frames, &audio) {
                tracing::warn!(error = %e, "dropping a malformed slice");
            }
        }
    }

}

/// The video encoder thread: pops ticks' frames, encodes, sends.
struct VideoThread<S: MediaSink> {
    cfg: MediaConfig,
    sink: S,
    q: Arc<DropOldest<(RgbFrame, u64)>>,
    keyframe: Arc<AtomicBool>,
    failed: Arc<AtomicBool>,
    events: tokio::sync::mpsc::UnboundedSender<MediaEvent>,
    gauges: Arc<MediaGauges>,
    video: Video,
    pending_rtp: VecDeque<u64>,
    gaps: GapKeyframes,
}

impl<S: MediaSink> VideoThread<S> {
    fn send(&mut self, data: Bytes, rtp: u64, keyframe: bool) {
        if keyframe {
            self.gaps.sent(Instant::now());
        }
        self.sink.video(data, rtp, keyframe);
        self.gauges.video_packets.fetch_add(1, Ordering::Relaxed);
    }

    fn run(mut self) {
        loop {
            let Some((f, rtp)) = self.q.pop_timeout(Duration::from_millis(50)) else {
                if self.q.is_closed() {
                    return;
                }
                // The libvpx pipe hands frames back a few ms after the
                // input: send the last one before a pause now.
                if let Video::Libvpx(e) = &mut self.video {
                    match e.poll() {
                        Ok(frames) => {
                            for x in frames {
                                let t = self.pending_rtp.pop_front().unwrap_or_default();
                                self.send(x.data, t, x.keyframe);
                            }
                        }
                        Err(e) => {
                            self.failed.store(true, Ordering::Relaxed);
                            let _ = self.events.send(MediaEvent::Failed(format!("vp8: {e}")));
                            return;
                        }
                    }
                }
                continue;
            };
            // A gap in the sequence (drop-oldest) needs a keyframe, unless
            // one is on its way or just went out. (A pipe encoder restarts
            // ffmpeg for a keyframe; frames dropped while it starts must
            // not restart it again.)
            if self.q.take_dropped() && self.gaps.gap_needs_keyframe(Instant::now()) {
                self.keyframe.store(true, Ordering::Relaxed);
            }
            if self.keyframe.swap(false, Ordering::Relaxed) {
                match &mut self.video {
                    Video::H264(e) => {
                        e.force_idr();
                        self.gaps.forced();
                    }
                    Video::Libvpx(e) => {
                        e.force_keyframe();
                        self.gaps.forced();
                    }
                    // Intra-only: every frame is a keyframe.
                    Video::Vp8(_) => {}
                    // Opened with the next frame, a keyframe.
                    Video::None => {}
                }
            }
            let t = Instant::now();
            if let Err(e) = self.encode_video(&f, rtp) {
                self.failed.store(true, Ordering::Relaxed);
                let _ = self.events.send(MediaEvent::Failed(e));
                return;
            }
            self.gauges.encode_us.fetch_add(t.elapsed().as_micros() as u64, Ordering::Relaxed);
        }
    }

    fn open_video(&mut self, f: &RgbFrame) -> Result<(), String> {
        let bitrate = self.cfg.video_bitrate.unwrap_or_else(|| fastvideo_media::video::default_bitrate(f.width, f.height));
        self.video = match self.cfg.codec {
            VideoCodec::H264 => {
                let mut c = H264Config::for_publish(PublishTarget::Peer, f.width, f.height, self.cfg.fps);
                c.bitrate_bps = bitrate;
                c.gop_seconds = self.cfg.gop_seconds;
                Video::H264(create_encoder(self.cfg.h264, c).map_err(|e| format!("h264 encoder: {e}"))?)
            }
            VideoCodec::Vp8 if fastvideo_media::vp8::libvpx_available() => {
                let mut c = fastvideo_media::vp8::Vp8Config::new(f.width, f.height, self.cfg.fps);
                c.bitrate_bps = bitrate;
                c.gop_seconds = self.cfg.gop_seconds;
                Video::Libvpx(fastvideo_media::vp8::Vp8Encoder::new(c).map_err(|e| format!("vp8 encoder: {e}"))?)
            }
            VideoCodec::Vp8 => Video::Vp8(
                Vp8Encoder::new(f.width, f.height, quality_for(bitrate, f.width, f.height, self.cfg.fps))
                    .map_err(|e| format!("vp8 encoder: {e}"))?,
            ),
        };
        Ok(())
    }

    fn encode_video(&mut self, f: &RgbFrame, rtp: u64) -> Result<(), String> {
        if matches!(self.video, Video::None) {
            self.open_video(f)?;
            self.gaps.forced();
        }
        let out: Vec<(u64, Bytes, bool)> = match &mut self.video {
            Video::H264(e) => {
                // One output per input, in order: timestamps by FIFO.
                self.pending_rtp.push_back(rtp);
                let aus = e.encode(f).map_err(|e| format!("h264: {e}"))?;
                aus.into_iter().map(|x| (self.pending_rtp.pop_front().unwrap_or(rtp), x.data, x.keyframe)).collect()
            }
            Video::Libvpx(e) => {
                self.pending_rtp.push_back(rtp);
                let frames = e.encode(f).map_err(|e| format!("vp8: {e}"))?;
                frames.into_iter().map(|x| (self.pending_rtp.pop_front().unwrap_or(rtp), x.data, x.keyframe)).collect()
            }
            Video::Vp8(e) => vec![(rtp, e.encode(f).map_err(|e| format!("vp8: {e}"))?, true)],
            Video::None => Vec::new(),
        };
        for (t, data, key) in out {
            self.send(data, t, key);
        }
        Ok(())
    }
}

/// Starts a playout thread; returns its command sender.
pub fn start<S: MediaSink>(
    cfg: MediaConfig,
    sink: S,
    events: tokio::sync::mpsc::UnboundedSender<MediaEvent>,
    gauges: Arc<MediaGauges>,
) -> Result<Sender<MediaCmd>, String> {
    let (tx, rx) = std::sync::mpsc::channel();
    let l = MediaLoop::new(cfg, sink, rx, events, gauges)?;
    l.spawn().map_err(|e| e.to_string())?;
    Ok(tx)
}
