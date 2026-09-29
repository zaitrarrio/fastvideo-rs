//! Media hand-off from the streaming cores to a transport (design §5.1,
//! §5.4, §5.5, §5.10).
//!
//! ```text
//! ClipPlayer ──MediaItem (bounded)──► clip pacer  (AvPacer, 2 s A/V, Metronome @ fps) ──┐
//! CausalSession ──blocks (depth 4)──► causal pacer (FramePacer 48, adaptive fps)  ──────┴─► Tick queue (10, drop-oldest) ─► encoder
//! ```
//!
//! - [`MediaItem`] is what a [`ClipPlayer`](super::player::ClipPlayer) emits:
//!   3-frame lockstep slices on a re-anchoring metronome, clip boundaries,
//!   and `Clear` when a clip is cut.
//! - [`spawn_clip_pacer`] runs `fastvideo_media`'s [`AvPacer`]: every tick is
//!   one video frame (fresh, or the last one held) plus exactly `48000/fps`
//!   audio sample frames (or silence), so the audio clock never stops.
//! - [`spawn_causal_pacer`] runs strobe's [`FramePacer`] port over a
//!   [`CausalSession`]: drop-oldest over 48 frames, freeze on underrun, and
//!   an adaptive playout rate; the RTP clock advances by
//!   `90000/effective_fps`.
//! - Both emit [`Tick`]s into a [`TickReceiver`]: 10 ticks, drop-oldest
//!   (the encoder-input row of §5.10). After a drop the consumer should
//!   force an IDR ([`TickReceiver::take_dropped`]).
//!
//! Everything here runs on tokio tasks; nothing touches the executor.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use fastvideo_media::clock::{MonotonicClock, VideoRtpClock};
use fastvideo_media::pacer::{
    AvPacer, AvPacerConfig, FramePacer, Metronome, VideoOut, CAUSAL_BUFFER_FRAMES, CAUSAL_MIN_FPS,
};
use fastvideo_protocol::{ApiError, CausalLimits, EndReason, RgbFrame, SessionSpec};
use tokio::sync::{mpsc, watch, Notify};
use tokio::task::JoinHandle;
use uuid::Uuid;

use crate::cancel::lock;
use super::causal::{CausalControl, CausalSession};

/// A slice of one playing clip: `frames` plus exactly their audio.
#[derive(Clone, Debug, PartialEq)]
pub struct MediaSlice {
    pub clip_id: Uuid,
    /// Index of `frames[0]` within the clip.
    pub first_frame: u32,
    pub frames: Vec<RgbFrame>,
    /// Interleaved 48 kHz at the session's channel count,
    /// `round(hi·spf) − round(lo·spf)` sample frames; `None` for video-only.
    pub audio: Option<Vec<f32>>,
}

/// How a clip's playout ended.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PlayOutcome {
    /// Every frame went out.
    Finished,
    /// `stop` or `reset` cut it.
    Stopped,
    /// The audience left mid-clip (nobody to tell).
    Gone,
}

/// What a clip player hands to its pacer.
#[derive(Clone, Debug, PartialEq)]
pub enum MediaItem {
    Slice(MediaSlice),
    /// The clip's last slice has been sent (or it was cut). `armed`: another
    /// clip starts right away (a pending `play` or autoplay with a ready
    /// clip). With nothing armed, fast-h3 flushes to black (Reactor) and the
    /// fal director holds the last frame.
    ClipEnd {
        clip_id: Uuid,
        outcome: PlayOutcome,
        armed: bool,
    },
    /// Drop whatever the pacer still buffers (a cut: `stop`, `reset`).
    Clear,
}

/// What the video shows while nothing plays.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum IdlePolicy {
    /// Hold the last frame (fal director, WHIP).
    #[default]
    Hold,
    /// Flush to black after a clip ends with nothing armed, and on a cut
    /// (Reactor fast-h3 parity).
    Black,
}

/// When the tick loop starts.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum TickStart {
    /// At the first video frame (transports we initiate, e.g. WHIP, wait for
    /// it before the handshake; design §5.2).
    #[default]
    OnFirstFrame,
    /// At once: ticks carry no video and silent audio until the first frame.
    Immediately,
}

/// One pacer tick: one video slot and (with audio) exactly `48000/fps`
/// sample frames.
#[derive(Clone, Debug, PartialEq)]
pub struct Tick {
    /// 0-based since the tick loop started.
    pub index: u64,
    pub video: VideoOut<RgbFrame>,
    /// Interleaved wire-rate audio, `None` for a video-only session.
    pub audio: Option<Vec<f32>>,
    /// 90 kHz RTP timestamp of this frame (base 0): fixed `90000/fps` steps
    /// for clips, `90000/effective_fps` for causal playout.
    pub video_rtp: u32,
    /// The playout rate this tick was paced at.
    pub fps: f64,
}

struct TickInner {
    q: Mutex<VecDeque<Tick>>,
    cap: usize,
    notify: Notify,
    closed: AtomicBool,
    dropped: AtomicU64,
    dropped_flag: AtomicBool,
}

/// The encoder side of the tick queue (bounded, drop-oldest).
pub struct TickReceiver {
    inner: Arc<TickInner>,
}

impl std::fmt::Debug for TickReceiver {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TickReceiver")
            .field("len", &self.len())
            .field("dropped", &self.dropped())
            .finish()
    }
}

#[derive(Clone)]
pub(crate) struct TickSender {
    inner: Arc<TickInner>,
}

pub(crate) fn tick_queue(cap: usize) -> (TickSender, TickReceiver) {
    let inner = Arc::new(TickInner {
        q: Mutex::new(VecDeque::new()),
        cap: cap.max(1),
        notify: Notify::new(),
        closed: AtomicBool::new(false),
        dropped: AtomicU64::new(0),
        dropped_flag: AtomicBool::new(false),
    });
    (TickSender { inner: inner.clone() }, TickReceiver { inner })
}

impl TickSender {
    pub(crate) fn send(&self, t: Tick) {
        {
            let mut q = lock(&self.inner.q);
            q.push_back(t);
            while q.len() > self.inner.cap {
                q.pop_front();
                self.inner.dropped.fetch_add(1, Ordering::Relaxed);
                self.inner.dropped_flag.store(true, Ordering::Relaxed);
            }
        }
        self.inner.notify.notify_one();
    }

    pub(crate) fn close(&self) {
        self.inner.closed.store(true, Ordering::Release);
        self.inner.notify.notify_one();
    }
}

impl Drop for TickSender {
    fn drop(&mut self) {
        // The pacer task holds the only sender.
        self.close();
    }
}

impl TickReceiver {
    /// The next tick; `None` once the pacer has ended and the queue is empty.
    pub async fn recv(&mut self) -> Option<Tick> {
        loop {
            let n = self.inner.notify.notified();
            if let Some(t) = lock(&self.inner.q).pop_front() {
                return Some(t);
            }
            if self.inner.closed.load(Ordering::Acquire) {
                return lock(&self.inner.q).pop_front();
            }
            n.await;
        }
    }

    pub fn try_recv(&mut self) -> Option<Tick> {
        lock(&self.inner.q).pop_front()
    }

    pub fn len(&self) -> usize {
        lock(&self.inner.q).len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Ticks dropped because the consumer fell behind.
    pub fn dropped(&self) -> u64 {
        self.inner.dropped.load(Ordering::Relaxed)
    }

    /// Whether ticks were dropped since the last call (force an IDR).
    pub fn take_dropped(&self) -> bool {
        self.inner.dropped_flag.swap(false, Ordering::Relaxed)
    }
}

/// Pacer counters, published on every tick.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct PaceStats {
    pub ticks: u64,
    pub fresh_frames: u64,
    pub repeated_frames: u64,
    /// Frames the pacer's own buffer dropped (drop-oldest).
    pub pacer_dropped: u64,
    /// Causal: ticks served by repeating the last frame.
    pub underruns: u64,
    /// Unique frames per second of playout since the first frame.
    pub unique_fps: f64,
    /// Current playout rate.
    pub effective_fps: f64,
    /// Seconds of video emitted since the first frame.
    pub video_seconds: f64,
    /// Seconds counted against `max_seconds`: since the first frame (clip),
    /// or since the first frame or the last `reset` (causal).
    pub limit_seconds: f64,
    /// Set when the pacer ended on its own (`max_seconds`, the source ended).
    pub ended: Option<EndReason>,
}

/// A running pacer.
#[derive(Debug)]
pub struct PacedStream {
    pub ticks: TickReceiver,
    /// Turns `true` at the first video frame.
    pub first_frame: watch::Receiver<bool>,
    pub stats: watch::Receiver<PaceStats>,
    pub task: JoinHandle<()>,
}

/// Encoder-input depth (design §5.10).
pub const TICK_DEPTH: usize = 10;

/// Clip pacer settings.
#[derive(Clone, Debug, PartialEq)]
pub struct ClipPacerConfig {
    pub fps: u32,
    /// Wire audio channels; `None` for a video-only session.
    pub channels: Option<u8>,
    /// For the black frame of [`IdlePolicy::Black`].
    pub canvas: (u32, u32),
    pub idle: IdlePolicy,
    pub start: TickStart,
    pub tick_depth: usize,
    /// Enforced in video time from the first frame (`SessionSpec::max_seconds`).
    pub max_seconds: Option<u32>,
    /// Playout speed (1.0 = real time; tests only).
    pub speed: f64,
}

impl ClipPacerConfig {
    /// From a session spec: its fps, canvas, audio channels and limit.
    pub fn for_spec(spec: &SessionSpec) -> Self {
        Self {
            fps: spec.fps,
            channels: spec.tracks.audio.as_ref().map(|a| a.channels),
            canvas: spec.canvas,
            idle: IdlePolicy::Hold,
            start: TickStart::OnFirstFrame,
            tick_depth: TICK_DEPTH,
            max_seconds: spec.max_seconds,
            speed: 1.0,
        }
    }
}

fn media_err(e: impl std::fmt::Display) -> ApiError {
    ApiError::internal(format!("pacer: {e}"))
}

/// Starts the clip pacer over a player's media stream (needs a tokio
/// runtime). It ends when the media channel closes and its buffers are
/// drained, or at `max_seconds`.
pub fn spawn_clip_pacer(
    mut media: mpsc::Receiver<MediaItem>,
    cfg: ClipPacerConfig,
) -> Result<PacedStream, ApiError> {
    let mut pacer: AvPacer<RgbFrame> =
        AvPacer::new(AvPacerConfig::wire(cfg.fps, cfg.channels)).map_err(media_err)?;
    let first_frame = pacer.first_frame();
    let (tx, rx) = tick_queue(cfg.tick_depth);
    let (stats_tx, stats_rx) = watch::channel(PaceStats {
        effective_fps: cfg.fps as f64,
        ..PaceStats::default()
    });
    let task = tokio::spawn(async move {
        let fps = cfg.fps.max(1);
        let speed = if cfg.speed > 0.0 { cfg.speed } else { 1.0 };
        let mut met = Metronome::new(fps as f64 * speed);
        let t0 = Instant::now();
        let mut rtp = VideoRtpClock::fixed(fps, 0);
        let mut closed = false;
        let mut pending_black = false;
        let mut started = cfg.start == TickStart::Immediately;
        let mut stats = PaceStats {
            effective_fps: fps as f64,
            ..PaceStats::default()
        };
        let mut first_at: Option<u64> = None;
        let black = || RgbFrame::black(cfg.canvas.0.max(1), cfg.canvas.1.max(1), 0);
        let spt = pacer.samples_per_tick() as usize * cfg.channels.unwrap_or(1) as usize;
        let apply = |pacer: &mut AvPacer<RgbFrame>, item: MediaItem, pending_black: &mut bool| match item {
            MediaItem::Slice(s) => {
                *pending_black = false;
                let n = s.frames.len();
                let r = match &s.audio {
                    Some(a) => pacer.push_slice(s.frames, a),
                    None if cfg.channels.is_some() => {
                        pacer.push_slice(s.frames, &vec![0.0; n * spt])
                    }
                    None => {
                        pacer.push_frames(s.frames);
                        Ok(())
                    }
                };
                if let Err(e) = r {
                    tracing::warn!(error = %e, "clip pacer dropped a malformed slice");
                }
            }
            MediaItem::ClipEnd { armed, .. } => {
                if !armed && cfg.idle == IdlePolicy::Black {
                    *pending_black = true;
                }
            }
            MediaItem::Clear => {
                pacer.clear();
                if cfg.idle == IdlePolicy::Black {
                    pacer.hold(black());
                    *pending_black = false;
                }
            }
        };
        loop {
            // Take everything that has arrived.
            loop {
                match media.try_recv() {
                    Ok(item) => apply(&mut pacer, item, &mut pending_black),
                    Err(mpsc::error::TryRecvError::Empty) => break,
                    Err(mpsc::error::TryRecvError::Disconnected) => {
                        closed = true;
                        break;
                    }
                }
            }
            if !started && pacer.has_first_frame() {
                started = true;
                met.reanchor();
            }
            if closed && (!started || pacer.buffered_frames() == 0) {
                stats.ended.get_or_insert(EndReason::Stopped);
                stats_tx.send_replace(stats.clone());
                break;
            }
            let now = t0.elapsed().as_secs_f64();
            let wait = if started { met.poll(now).err() } else { Some(1.0) };
            if let Some(dt) = wait {
                if closed {
                    tokio::time::sleep(Duration::from_secs_f64(dt.min(1.0))).await;
                    continue;
                }
                tokio::select! {
                    item = media.recv() => match item {
                        Some(item) => apply(&mut pacer, item, &mut pending_black),
                        None => closed = true,
                    },
                    _ = tokio::time::sleep(Duration::from_secs_f64(dt)) => {}
                }
                continue;
            }
            let t = pacer.tick();
            if pending_black && pacer.buffered_frames() == 0 {
                pacer.hold(black());
                pending_black = false;
            }
            match &t.video {
                VideoOut::Fresh(_) => stats.fresh_frames += 1,
                VideoOut::Repeat(_) => stats.repeated_frames += 1,
                VideoOut::Nothing => {}
            }
            if first_at.is_none() && !matches!(t.video, VideoOut::Nothing) {
                first_at = Some(t.index);
            }
            stats.ticks += 1;
            stats.pacer_dropped = pacer.stats().dropped_frames;
            let shown = first_at.map_or(0, |f| t.index + 1 - f);
            stats.video_seconds = shown as f64 / fps as f64;
            stats.limit_seconds = stats.video_seconds;
            stats.unique_fps = if shown > 0 {
                stats.fresh_frames as f64 / (shown as f64 / fps as f64)
            } else {
                0.0
            };
            tx.send(Tick {
                index: t.index,
                video: t.video,
                audio: t.audio,
                video_rtp: rtp.advance(),
                fps: fps as f64,
            });
            let limit = cfg
                .max_seconds
                .is_some_and(|m| shown >= u64::from(m) * u64::from(fps));
            if limit {
                stats.ended = Some(EndReason::SessionLimit);
            }
            stats_tx.send_replace(stats.clone());
            if limit {
                break;
            }
        }
        tx.close();
    });
    Ok(PacedStream {
        ticks: rx,
        first_frame,
        stats: stats_rx,
        task,
    })
}

/// Causal pacer settings (design §5.4).
#[derive(Clone, Debug, PartialEq)]
pub struct CausalPacerConfig {
    /// Playout ceiling (the model fps).
    pub fps: u32,
    /// Drop-oldest bound (48).
    pub buffer_frames: usize,
    /// Adaptive floor (4).
    pub min_fps: u32,
    /// Adaptive playout from the generation-rate EMA (on by default).
    pub adaptive: bool,
    pub tick_depth: usize,
    /// Video seconds from the first frame, or from the first frame after the
    /// last `reset` (a reset restarts this clock).
    pub max_seconds: Option<u32>,
    /// Video seconds of the whole session, resets included.
    pub total_max_seconds: Option<u32>,
}

impl CausalPacerConfig {
    /// Settings for `spec`: resets do not extend the session
    /// (`total_max_seconds = max_seconds`).
    pub fn for_spec(spec: &SessionSpec) -> Self {
        Self {
            fps: spec.fps,
            buffer_frames: CAUSAL_BUFFER_FRAMES,
            min_fps: CAUSAL_MIN_FPS,
            adaptive: true,
            tick_depth: TICK_DEPTH,
            max_seconds: spec.max_seconds,
            total_max_seconds: spec.max_seconds,
        }
    }

    /// Settings for a live session under `limits` (design §5.2): the clock
    /// restarts at each `reset`, up to `limits.hard_max_s` of video in all.
    pub fn with_limits(spec: &SessionSpec, limits: &CausalLimits) -> Self {
        let total = spec.max_seconds.map(|m| m.max(limits.hard_max_s)).unwrap_or(limits.hard_max_s);
        Self {
            total_max_seconds: Some(total),
            ..Self::for_spec(spec)
        }
    }

    /// Buffered frames above which the pacer stops taking blocks: half the
    /// drop-oldest bound, so a hot generator waits on the session channel
    /// (depth `EngineConfig::causal_depth`) and the frames queued ahead of a
    /// prompt switch stay near the old 48-frame bound.
    pub fn pull_mark(&self) -> usize {
        (self.buffer_frames / 2).max(1)
    }
}

/// Starts the causal pacer: it owns `session` (dropping it at the end closes
/// the session) and reports `unique_fps` back through `control`. Ticks carry
/// no audio (SF-Wan is video-only). The loop starts at the first frame and
/// ends when the session ends, on a backend error, or at `max_seconds`.
pub fn spawn_causal_pacer(
    mut session: CausalSession,
    cfg: CausalPacerConfig,
) -> Result<PacedStream, ApiError> {
    let control: CausalControl = session.control();
    let mut pacer: FramePacer<RgbFrame, MonotonicClock> = FramePacer::new(
        cfg.fps,
        cfg.buffer_frames,
        cfg.adaptive,
        cfg.min_fps,
        MonotonicClock::new(),
    )
    .map_err(media_err)?;
    let (tx, rx) = tick_queue(cfg.tick_depth);
    let (first_tx, first_rx) = watch::channel(false);
    let (stats_tx, stats_rx) = watch::channel(PaceStats::default());
    let task = tokio::spawn(async move {
        // tokio's clock (not std's), so a paused-time test can run the
        // session limits in virtual time.
        let t0 = tokio::time::Instant::now();
        let mut met = Metronome::new(pacer.effective_fps());
        let mut rtp = VideoRtpClock::adaptive(0);
        let mut stats = PaceStats::default();
        let mut index = 0u64;
        let ended: Option<EndReason>;
        let mut started = false;
        let mut video_s = 0.0f64;
        // Video seconds since the first frame or the last reset.
        let mut window_s = 0.0f64;
        let mut last_len = 0usize;
        loop {
            let now = t0.elapsed().as_secs_f64();
            let wait = if started { met.poll(now).err() } else { Some(1.0) };
            if let Some(dt) = wait {
                // Backpressure: take the next block only when it fits under
                // the pull mark, so a generator faster than playout blocks on
                // the session channel (the executor waits) instead of
                // overflowing the drop-oldest buffer (skipped frames). A reset
                // is taken promptly by the next pull (the buffer drains).
                let room = pacer.buffered() == 0 || pacer.buffered() + last_len <= cfg.pull_mark();
                if !room {
                    tokio::time::sleep(Duration::from_secs_f64(dt)).await;
                    continue;
                }
                tokio::select! {
                    b = session.next_block() => match b {
                        Some(Ok(block)) => {
                            if block.reset {
                                pacer.clear();
                                // A reset renews the anchor: the limit clock
                                // restarts at its first block.
                                window_s = 0.0;
                            }
                            last_len = block.frames.len();
                            // The generation rate from the block's own time:
                            // arrivals are paced by the pull mark.
                            let gen_fps = (block.stats.block_ms > 0.0)
                                .then(|| last_len as f64 * 1e3 / block.stats.block_ms);
                            pacer.push_chunk_rate(block.frames, gen_fps);
                            if !started {
                                started = true;
                                first_tx.send_replace(true);
                                met.reanchor();
                            }
                        }
                        Some(Err(e)) => {
                            ended = Some(EndReason::Error(e));
                            break;
                        }
                        None => {
                            ended = Some(EndReason::Stopped);
                            break;
                        }
                    },
                    _ = tokio::time::sleep(Duration::from_secs_f64(dt)) => {}
                }
                continue;
            }
            let eff = pacer.effective_fps();
            let under = pacer.stats().underruns;
            let Ok(frame) = pacer.next_frame() else {
                continue;
            };
            let fresh = pacer.stats().underruns == under;
            let video = if fresh {
                stats.fresh_frames += 1;
                VideoOut::Fresh(frame)
            } else {
                stats.repeated_frames += 1;
                VideoOut::Repeat(frame)
            };
            let ps = pacer.stats();
            stats.ticks += 1;
            stats.underruns = ps.underruns;
            stats.pacer_dropped = ps.dropped;
            stats.effective_fps = eff;
            stats.unique_fps = pacer.unique_fps();
            video_s += 1.0 / eff.max(1e-3);
            window_s += 1.0 / eff.max(1e-3);
            stats.video_seconds = video_s;
            stats.limit_seconds = window_s;
            control.report_playout(stats.unique_fps, eff);
            tx.send(Tick {
                index,
                video,
                audio: None,
                video_rtp: rtp.advance_adaptive(eff),
                fps: eff,
            });
            index += 1;
            // The next tick comes at the (possibly new) effective rate.
            met.set_fps(pacer.effective_fps());
            stats_tx.send_replace(stats.clone());
            if cfg.max_seconds.is_some_and(|m| window_s >= f64::from(m))
                || cfg.total_max_seconds.is_some_and(|m| video_s >= f64::from(m))
            {
                ended = Some(EndReason::SessionLimit);
                break;
            }
        }
        stats.ended = ended;
        stats_tx.send_replace(stats);
        tx.close();
        drop(session);
    });
    Ok(PacedStream {
        ticks: rx,
        first_frame: first_rx,
        stats: stats_rx,
        task,
    })
}
