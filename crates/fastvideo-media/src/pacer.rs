//! Pacing between bursty generation and a steady real-time transport.
//!
//! Three pieces (design §5.4, §5.5, §5.10):
//!
//! - [`FramePacer`]: the causal (SF-Wan) video jitter buffer. **Ported from
//!   strobe-core `src/pacing.rs`** (strobe, MIT License; the copyright and
//!   permission notice are in `NOTICE` in this crate). Drop-oldest over
//!   `max_buffer`, freeze (repeat the last frame) on underrun, adaptive
//!   playout fps from an EMA of the generation rate (0.3 new / 0.7 old),
//!   clamped to `[min_fps, fps]`. The semantics, constants and order of
//!   operations are kept exactly, and strobe's tests are ported verbatim.
//! - [`AvPacer`]: the clip-session A/V pacer. Original to fastvideo-rs,
//!   following the infinite-livestream pacer semantics (streaming-refs §4.2):
//!   one tick is one video frame (or a repeat) plus **exactly** `rate/fps`
//!   audio sample frames (or silence), with video and audio FIFOs sharing
//!   one shallow cap (2 s). It adds a first-frame notification.
//! - [`Metronome`]: a re-anchoring tick scheduler that never bursts to
//!   catch up.
//!
//! strobe-core's notes on its two intentional differences from the Python
//! pacer still apply to `FramePacer`: it is generic over the frame type
//! (the pacer only clones on underrun), and it has no async signalling of its
//! own (callers poll [`FramePacer::has_frames`] or wrap it).

use std::collections::VecDeque;

use tokio::sync::watch;

use crate::clock::Clock;
use crate::error::{MediaError, Result};
use crate::lockstep::samples_per_frame;

// ---------------------------------------------------------------------------
// FramePacer: strobe-core port (MIT). Keep in step with strobe `pacing.rs`.
// ---------------------------------------------------------------------------

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum PacerError {
    #[error("fps and max_buffer must be positive")]
    InvalidParams,
    /// Equivalent to Python's `LookupError("pacer has no frames yet")`.
    #[error("pacer has no frames yet")]
    Empty,
}

/// Counters behind the `pacer:` log line. `served - underruns` is the number
/// of unique frames shown; per second that is `unique_fps`, the smoothness
/// metric (not the RTP rate).
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct PacerStats {
    pub pushed: u64,
    pub served: u64,
    pub dropped: u64,
    pub underruns: u64,
}

impl PacerStats {
    pub fn unique_frames(&self) -> u64 {
        self.served - self.underruns
    }
}

/// Defaults from design §5.4 for causal SF-Wan streaming.
pub const CAUSAL_BUFFER_FRAMES: usize = 48;
pub const CAUSAL_MIN_FPS: u32 = 4;

pub struct FramePacer<T, C: Clock> {
    /// Playout ceiling; the fixed rate when `adaptive` is false.
    fps: u32,
    max_buffer: usize,
    adaptive: bool,
    min_fps: u32,
    clock: C,
    buf: VecDeque<T>,
    last: Option<T>,
    closed: bool,
    stats: PacerStats,
    /// Generation-rate estimate (frames/sec of arrival), EMA over block bursts.
    /// Starts at the floor so playout eases in (buffer builds) rather than racing
    /// ahead and underrunning before the estimate converges.
    gen_ema: f64,
    last_push_t: Option<f64>,
    /// fastvideo-rs addition: when the first frame was served, for `unique_fps`.
    first_serve_t: Option<f64>,
}

impl<T: Clone, C: Clock> FramePacer<T, C> {
    pub fn new(fps: u32, max_buffer: usize, adaptive: bool, min_fps: u32, clock: C) -> Result<Self, PacerError> {
        if fps == 0 || max_buffer == 0 {
            return Err(PacerError::InvalidParams);
        }
        let min_fps = min_fps.min(fps).max(1);
        Ok(Self {
            fps,
            max_buffer,
            adaptive,
            min_fps,
            clock,
            buf: VecDeque::new(),
            last: None,
            closed: false,
            stats: PacerStats::default(),
            gen_ema: f64::from(min_fps),
            last_push_t: None,
            first_serve_t: None,
        })
    }

    /// The causal defaults of design §5.4: buffer 48, adaptive, floor 4.
    pub fn causal(fps: u32, clock: C) -> Result<Self, PacerError> {
        Self::new(fps, CAUSAL_BUFFER_FRAMES, true, CAUSAL_MIN_FPS, clock)
    }

    /// Add a burst of frames. Drops oldest buffered frames on overflow, so
    /// latency stays bounded instead of growing without limit when generation
    /// runs hot.
    pub fn push_chunk<I: IntoIterator<Item = T>>(&mut self, chunk: I) {
        let mut n: u64 = 0;
        for frame in chunk {
            self.buf.push_back(frame);
            self.stats.pushed += 1;
            n += 1;
        }
        // Update the generation-rate estimate from this burst's arrival timing.
        // A block of n frames arriving dt after the previous burst == n/dt fps.
        // Ordering matters: this runs BEFORE the overflow drop, so the estimate
        // reflects what the model produced, not what survived the buffer.
        let t = self.clock.now();
        if let (Some(prev), true) = (self.last_push_t, n > 0) {
            let dt = t - prev;
            if dt > 1e-3 {
                let inst = n as f64 / dt;
                self.gen_ema = 0.3 * inst + 0.7 * self.gen_ema;
            }
        }
        self.last_push_t = Some(t);
        while self.buf.len() > self.max_buffer {
            self.buf.pop_front();
            self.stats.dropped += 1;
        }
    }

    /// Playout rate: the fixed `fps`, or (adaptive) the measured generation rate
    /// clamped to `[min_fps, fps]` so playout tracks what the model produces.
    pub fn effective_fps(&self) -> f64 {
        if !self.adaptive {
            return f64::from(self.fps);
        }
        f64::from(self.min_fps).max(f64::from(self.fps).min(self.gen_ema))
    }

    /// Non-blocking: buffered frame, or a repeat of the last one on underrun.
    ///
    /// Freezing beats stalling — the track must keep producing or the peer
    /// connection dies. Returns [`PacerError::Empty`] only if nothing has ever
    /// been pushed.
    pub fn next_frame(&mut self) -> Result<T, PacerError> {
        if self.first_serve_t.is_none() && (self.last.is_some() || !self.buf.is_empty()) {
            self.first_serve_t = Some(self.clock.now());
        }
        if let Some(frame) = self.buf.pop_front() {
            self.last = Some(frame.clone());
            self.stats.served += 1;
            return Ok(frame);
        }
        if let Some(last) = &self.last {
            self.stats.underruns += 1;
            self.stats.served += 1;
            return Ok(last.clone());
        }
        Err(PacerError::Empty)
    }

    pub fn buffered(&self) -> usize {
        self.buf.len()
    }

    pub fn has_frames(&self) -> bool {
        !self.buf.is_empty()
    }

    pub fn stats(&self) -> PacerStats {
        self.stats
    }

    /// Unique frames served per second of playout since the first serve.
    pub fn unique_fps(&self) -> f64 {
        match self.first_serve_t {
            Some(t0) => {
                let dt = self.clock.now() - t0;
                if dt > 1e-6 { self.stats.unique_frames() as f64 / dt } else { 0.0 }
            }
            None => 0.0,
        }
    }

    /// Drop everything buffered (causal `reset`); the last frame stays held.
    pub fn clear(&mut self) {
        self.buf.clear();
    }

    pub fn close(&mut self) {
        self.closed = true;
    }

    pub fn is_closed(&self) -> bool {
        self.closed
    }
}

// ---------------------------------------------------------------------------
// AvPacer: clip-session A/V lockstep pacer (original).
// ---------------------------------------------------------------------------

/// Shallow shared cap for the clip pacer's FIFOs (§5.5).
pub const AV_BUFFER_SECONDS: f64 = 2.0;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AvPacerConfig {
    pub fps: u32,
    /// Audio rate on the wire; 48000 on every WebRTC path.
    pub rate: u32,
    /// Audio channels on the wire (Reactor 1, WMA/WHIP 2).
    pub channels: u8,
    /// `false` for a video-only track set: ticks carry no audio at all.
    pub audio: bool,
    pub buffer_seconds: f64,
}

impl AvPacerConfig {
    /// 48 kHz wire audio with `channels`, or video only when `None`.
    pub fn wire(fps: u32, channels: Option<u8>) -> Self {
        Self {
            fps,
            rate: crate::lockstep::WIRE_RATE,
            channels: channels.unwrap_or(1),
            audio: channels.is_some(),
            buffer_seconds: AV_BUFFER_SECONDS,
        }
    }
}

/// The video half of a tick.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VideoOut<T> {
    /// Nothing has ever been pushed. The video track stays silent (no fake
    /// black encode, §5.2), except where a protocol sends its own black frame.
    Nothing,
    /// A new frame from the FIFO.
    Fresh(T),
    /// Underrun: the last frame again (hold).
    Repeat(T),
}

impl<T> VideoOut<T> {
    pub fn frame(&self) -> Option<&T> {
        match self {
            VideoOut::Nothing => None,
            VideoOut::Fresh(f) | VideoOut::Repeat(f) => Some(f),
        }
    }
    pub fn is_fresh(&self) -> bool {
        matches!(self, VideoOut::Fresh(_))
    }
}

/// One pacer tick: exactly one video slot and exactly `rate/fps` audio frames.
#[derive(Debug, Clone, PartialEq)]
pub struct AvTick<T> {
    /// 0-based tick number; the RTP clocks derive from it.
    pub index: u64,
    pub video: VideoOut<T>,
    /// Interleaved, exactly `samples_per_tick · channels` long, or `None` for
    /// a video-only session.
    pub audio: Option<Vec<f32>>,
    /// How many of this tick's sample frames were silence padding.
    pub silent_frames: u32,
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct AvStats {
    pub ticks: u64,
    pub fresh_frames: u64,
    pub repeated_frames: u64,
    /// Ticks before any frame existed.
    pub empty_ticks: u64,
    /// Ticks whose audio was entirely silence.
    pub silent_ticks: u64,
    /// Ticks whose audio was partly silence (the end of a clip's audio).
    pub partial_ticks: u64,
    pub dropped_frames: u64,
    pub dropped_samples: u64,
    /// Total audio sample frames (per channel) emitted; always `ticks·spt`.
    pub audio_frames_out: u64,
    pub pushed_frames: u64,
    pub pushed_samples: u64,
}

/// Constant-rate A/V clock between clip playout and the encoders.
pub struct AvPacer<T> {
    cfg: AvPacerConfig,
    spt: u32,
    max_frames: usize,
    max_audio: usize,
    frames: VecDeque<T>,
    audio: VecDeque<f32>,
    last: Option<T>,
    stats: AvStats,
    first_frame: watch::Sender<bool>,
}

impl<T: Clone> AvPacer<T> {
    /// Fails unless `rate % fps == 0` (the §5.5 admission check).
    pub fn new(cfg: AvPacerConfig) -> Result<Self> {
        let spt = samples_per_frame(cfg.rate, cfg.fps)?;
        if cfg.channels == 0 {
            return Err(MediaError::invalid("channels must be positive"));
        }
        if cfg.buffer_seconds.is_nan() || cfg.buffer_seconds <= 0.0 {
            return Err(MediaError::invalid("buffer_seconds must be positive"));
        }
        let max_frames = ((f64::from(cfg.fps) * cfg.buffer_seconds).round() as usize).max(1);
        let max_audio = ((f64::from(cfg.rate) * cfg.buffer_seconds).round() as usize).max(1) * cfg.channels as usize;
        let (tx, _) = watch::channel(false);
        Ok(Self {
            cfg,
            spt,
            max_frames,
            max_audio,
            frames: VecDeque::new(),
            audio: VecDeque::new(),
            last: None,
            stats: AvStats::default(),
            first_frame: tx,
        })
    }

    pub fn config(&self) -> &AvPacerConfig {
        &self.cfg
    }

    /// Audio sample frames per tick (`rate/fps`: 2000 at 24 fps, 3000 at 16).
    pub fn samples_per_tick(&self) -> u32 {
        self.spt
    }

    /// A receiver that turns `true` when the first video frame is pushed.
    /// Transports we initiate (WHIP) wait on it before the handshake (§5.2).
    pub fn first_frame(&self) -> watch::Receiver<bool> {
        self.first_frame.subscribe()
    }

    pub fn has_first_frame(&self) -> bool {
        *self.first_frame.borrow()
    }

    /// Buffer one video frame; drops the oldest over the cap.
    pub fn push_frame(&mut self, frame: T) {
        self.frames.push_back(frame);
        self.stats.pushed_frames += 1;
        while self.frames.len() > self.max_frames {
            self.frames.pop_front();
            self.stats.dropped_frames += 1;
        }
        if !*self.first_frame.borrow() {
            self.first_frame.send_replace(true);
        }
    }

    pub fn push_frames<I: IntoIterator<Item = T>>(&mut self, frames: I) {
        for f in frames {
            self.push_frame(f);
        }
    }

    /// Buffer interleaved wire-rate audio; drops the oldest sample frames over
    /// the cap. Ignored for a video-only session. A trailing partial sample
    /// frame is rejected.
    pub fn push_audio(&mut self, samples: &[f32]) -> Result<()> {
        let ch = self.cfg.channels as usize;
        if samples.len() % ch != 0 {
            return Err(MediaError::invalid("audio is not a whole number of sample frames"));
        }
        if !self.cfg.audio {
            return Ok(());
        }
        self.audio.extend(samples.iter().copied());
        self.stats.pushed_samples += (samples.len() / ch) as u64;
        if self.audio.len() > self.max_audio {
            let excess = self.audio.len() - self.max_audio;
            self.audio.drain(..excess);
            self.stats.dropped_samples += (excess / ch) as u64;
        }
        Ok(())
    }

    /// Push one lockstep slice (frames plus their audio).
    pub fn push_slice<I: IntoIterator<Item = T>>(&mut self, frames: I, audio: &[f32]) -> Result<()> {
        self.push_audio(audio)?;
        self.push_frames(frames);
        Ok(())
    }

    /// Emit one tick. Never fails, never blocks.
    pub fn tick(&mut self) -> AvTick<T> {
        let index = self.stats.ticks;
        self.stats.ticks += 1;
        let video = if let Some(f) = self.frames.pop_front() {
            self.last = Some(f.clone());
            self.stats.fresh_frames += 1;
            VideoOut::Fresh(f)
        } else if let Some(l) = &self.last {
            self.stats.repeated_frames += 1;
            VideoOut::Repeat(l.clone())
        } else {
            self.stats.empty_ticks += 1;
            VideoOut::Nothing
        };
        let (audio, silent_frames) = if self.cfg.audio {
            let ch = self.cfg.channels as usize;
            let want = self.spt as usize * ch;
            let take = want.min(self.audio.len());
            let mut v: Vec<f32> = self.audio.drain(..take).collect();
            let silent = ((want - take) / ch) as u32;
            v.resize(want, 0.0);
            if silent == self.spt {
                self.stats.silent_ticks += 1;
            } else if silent > 0 {
                self.stats.partial_ticks += 1;
            }
            self.stats.audio_frames_out += u64::from(self.spt);
            (Some(v), silent)
        } else {
            (None, 0)
        };
        AvTick { index, video, audio, silent_frames }
    }

    /// Hold `frame` from now on (Reactor `flush` to black, §5.5), dropping
    /// anything buffered.
    pub fn hold(&mut self, frame: T) {
        self.frames.clear();
        self.audio.clear();
        self.last = Some(frame);
    }

    /// Drop everything buffered (stop/reset); the last frame stays held.
    pub fn clear(&mut self) {
        self.frames.clear();
        self.audio.clear();
    }

    pub fn buffered_frames(&self) -> usize {
        self.frames.len()
    }

    /// Buffered audio in sample frames per channel.
    pub fn buffered_audio(&self) -> usize {
        self.audio.len() / self.cfg.channels as usize
    }

    pub fn stats(&self) -> AvStats {
        self.stats
    }
}

// ---------------------------------------------------------------------------
// Metronome (original).
// ---------------------------------------------------------------------------

/// Periods behind schedule before the metronome resnaps instead of bursting.
pub const RESNAP_PERIODS: f64 = 8.0;

/// A drift-free, re-anchoring tick schedule.
///
/// Deadlines advance by exactly one period per tick, so there is no drift
/// while the loop keeps up. A short stall is caught up (a few back-to-back
/// ticks); a stall longer than [`RESNAP_PERIODS`] re-anchors the schedule to
/// "now" instead of machine-gunning catch-up ticks.
#[derive(Debug, Clone)]
pub struct Metronome {
    period: f64,
    next: Option<f64>,
    resnaps: u64,
}

impl Metronome {
    pub fn new(fps: f64) -> Self {
        Self { period: 1.0 / fps.max(1e-3), next: None, resnaps: 0 }
    }

    /// Change the rate (adaptive causal playout); takes effect from the next tick.
    pub fn set_fps(&mut self, fps: f64) {
        self.period = 1.0 / fps.max(1e-3);
    }

    pub fn period(&self) -> f64 {
        self.period
    }

    /// `Ok(())` means tick now (the schedule has advanced); `Err(dt)` means
    /// sleep `dt` seconds and poll again. The first poll ticks immediately.
    pub fn poll(&mut self, now: f64) -> std::result::Result<(), f64> {
        let next = *self.next.get_or_insert(now);
        if now < next {
            return Err(next - now);
        }
        let mut due = next;
        if now - due > self.period * RESNAP_PERIODS {
            due = now;
            self.resnaps += 1;
        }
        self.next = Some(due + self.period);
        Ok(())
    }

    /// Re-anchor so the next poll ticks immediately (clip start).
    pub fn reanchor(&mut self) {
        self.next = None;
    }

    pub fn resnaps(&self) -> u64 {
        self.resnaps
    }
}

#[cfg(test)]
impl<T: Clone> FramePacer<T, crate::clock::ManualClock> {
    /// Test-only helper mirroring the `clock["t"] += dt` idiom in test_pacing.py.
    fn clock_advance(&mut self, dt: f64) {
        self.clock.advance(dt);
    }
}

/// strobe-core's `pacing.rs` tests, ported verbatim (strobe, MIT).
#[cfg(test)]
mod strobe_tests {
    use super::*;
    use crate::clock::{ManualClock, MonotonicClock};

    /// Stand-in for a frame. The Python tests compare pixel buffers; here a
    /// distinct integer per frame serves the same "is this the frame I expect"
    /// purpose without pulling in an array crate.
    fn frames(n: u32) -> Vec<u32> {
        (0..n).collect()
    }

    fn pacer(fps: u32, max_buffer: usize) -> FramePacer<u32, MonotonicClock> {
        FramePacer::new(fps, max_buffer, false, 4, MonotonicClock::new()).unwrap()
    }

    #[test]
    fn push_and_serve_in_order() {
        let mut p = pacer(16, 10);
        p.push_chunk(frames(3));
        let a = p.next_frame().unwrap();
        let b = p.next_frame().unwrap();
        assert_ne!(a, b);
        assert_eq!(p.stats().served, 2);
        assert_eq!(p.buffered(), 1);
    }

    #[test]
    fn overflow_drops_oldest() {
        let mut p = pacer(16, 4);
        p.push_chunk(frames(6));
        assert_eq!(p.buffered(), 4);
        assert_eq!(p.stats().dropped, 2);
        // First served frame should be index 2 (0 and 1 dropped).
        assert_eq!(p.next_frame().unwrap(), 2);
    }

    #[test]
    fn underrun_repeats_last_frame() {
        let mut p = pacer(16, 4);
        p.push_chunk(frames(1));
        let first = p.next_frame().unwrap();
        let again = p.next_frame().unwrap();
        assert_eq!(first, again);
        assert_eq!(p.stats().underruns, 1);
    }

    #[test]
    fn empty_pacer_errors() {
        let mut p = pacer(16, 4);
        assert_eq!(p.next_frame().unwrap_err(), PacerError::Empty);
    }

    #[test]
    fn invalid_params() {
        let r = FramePacer::<u32, _>::new(0, 4, false, 4, MonotonicClock::new());
        assert_eq!(r.err(), Some(PacerError::InvalidParams));
        let r = FramePacer::<u32, _>::new(16, 0, false, 4, MonotonicClock::new());
        assert_eq!(r.err(), Some(PacerError::InvalidParams));
    }

    #[test]
    fn effective_fps_fixed_when_not_adaptive() {
        let mut p = pacer(16, 100);
        assert_eq!(p.effective_fps(), 16.0);
        p.push_chunk(frames(3));
        assert_eq!(p.effective_fps(), 16.0); // never moves off the fixed rate
    }

    #[test]
    fn adaptive_tracks_generation_rate() {
        let clock = ManualClock::new(0.0);
        let mut p = FramePacer::<u32, _>::new(16, 1000, true, 4, clock).unwrap();
        // ~8 fps generation: an 8-frame block arrives every 1.0s.
        for _ in 0..25 {
            p.clock_advance(1.0);
            p.push_chunk(frames(8));
        }
        let fps = p.effective_fps();
        assert!((7.5..=8.5).contains(&fps), "converged to {fps}, want ~8");
    }

    #[test]
    fn adaptive_clamps_to_cap() {
        let clock = ManualClock::new(0.0);
        let mut p = FramePacer::<u32, _>::new(10, 5000, true, 3, clock).unwrap();
        // ~200 fps generation — far above the cap.
        for _ in 0..25 {
            p.clock_advance(0.1);
            p.push_chunk(frames(20));
        }
        assert_eq!(p.effective_fps(), 10.0); // never exceeds the fps ceiling
    }

    #[test]
    fn adaptive_clamps_to_floor() {
        let clock = ManualClock::new(0.0);
        let mut p = FramePacer::<u32, _>::new(16, 1000, true, 3, clock).unwrap();
        // ~0.2 fps generation — far below the floor.
        for _ in 0..10 {
            p.clock_advance(10.0);
            p.push_chunk(frames(2));
        }
        assert_eq!(p.effective_fps(), 3.0); // never drops below min_fps
    }

    #[test]
    fn close_is_observable() {
        let mut p = pacer(16, 4);
        assert!(!p.is_closed());
        p.close();
        assert!(p.is_closed());
    }
}

/// fastvideo-rs tests for the causal defaults, `AvPacer` and `Metronome`.
#[cfg(test)]
mod tests {
    use super::*;
    use crate::clock::ManualClock;
    use crate::lockstep::{clip_samples, slices, EMIT_FRAMES};
    use std::sync::Arc;

    #[test]
    fn causal_defaults_match_design() {
        let clock = Arc::new(ManualClock::new(0.0));
        let mut p = FramePacer::<u32, _>::causal(16, clock.clone()).unwrap();
        // SF-Wan emits 12-frame blocks; 5 blocks overflow the 48-frame buffer.
        for b in 0..5u32 {
            clock.advance(0.5);
            p.push_chunk((0..12).map(|i| b * 12 + i));
        }
        assert_eq!(p.buffered(), 48);
        assert_eq!(p.stats().dropped, 12);
        assert_eq!(p.next_frame().unwrap(), 12); // oldest block dropped
        // 12 frames per 0.5 s is 24 fps of generation: clamped to the 16 cap.
        assert_eq!(p.effective_fps(), 16.0);
        // Starve it: freeze on the last frame, counted as underruns.
        let mut q = FramePacer::<u32, _>::causal(16, clock.clone()).unwrap();
        q.push_chunk([7]);
        assert_eq!(q.next_frame().unwrap(), 7);
        for _ in 0..10 {
            assert_eq!(q.next_frame().unwrap(), 7);
        }
        assert_eq!(q.stats().underruns, 10);
        assert_eq!(q.stats().unique_frames(), 1);
    }

    #[test]
    fn adaptive_slow_generation_lowers_playout_rate() {
        // ~547 ms per 12-frame block (our pre-E7 SF-Wan): ~21.9 fps of
        // generation clamps to 16; at ~1.6 s per block (7.5 fps) playout follows.
        let clock = Arc::new(ManualClock::new(0.0));
        let mut p = FramePacer::<u32, _>::causal(16, clock.clone()).unwrap();
        for _ in 0..30 {
            clock.advance(1.6);
            p.push_chunk(0..12);
        }
        let f = p.effective_fps();
        assert!((7.0..=8.0).contains(&f), "{f}");
        // Unique fps over a period of steady 1-per-period serving.
        let mut u = FramePacer::<u32, _>::causal(16, clock.clone()).unwrap();
        u.push_chunk(0..16);
        for _ in 0..16 {
            u.next_frame().unwrap();
            clock.advance(1.0 / 16.0);
        }
        assert!((u.unique_fps() - 16.0).abs() < 1e-6);
    }

    #[test]
    fn av_pacer_admission_and_counts() {
        assert!(AvPacer::<u32>::new(AvPacerConfig::wire(7, Some(1))).is_err());
        let p = AvPacer::<u32>::new(AvPacerConfig::wire(24, Some(2))).unwrap();
        assert_eq!(p.samples_per_tick(), 2000);
        let p = AvPacer::<u32>::new(AvPacerConfig::wire(16, None)).unwrap();
        assert_eq!(p.samples_per_tick(), 3000);
    }

    #[test]
    fn av_pacer_freeze_and_silence_on_underrun() {
        let mut p = AvPacer::<u32>::new(AvPacerConfig::wire(24, Some(1))).unwrap();
        // Before any frame: no video, silent audio of the right size.
        let t = p.tick();
        assert_eq!(t.video, VideoOut::Nothing);
        assert_eq!(t.audio.as_ref().unwrap().len(), 2000);
        assert_eq!(t.silent_frames, 2000);
        let mut rx = p.first_frame();
        assert!(!*rx.borrow_and_update());
        p.push_slice([1, 2], &[0.5; 3000]).unwrap();
        assert!(rx.has_changed().unwrap());
        assert!(p.has_first_frame());
        let a = p.tick();
        assert_eq!(a.video, VideoOut::Fresh(1));
        assert_eq!(a.silent_frames, 0);
        let b = p.tick();
        assert_eq!(b.video, VideoOut::Fresh(2));
        assert_eq!(b.silent_frames, 1000); // only 1000 samples were left
        assert!(b.audio.as_ref().unwrap()[..1000].iter().all(|&s| s == 0.5));
        assert!(b.audio.as_ref().unwrap()[1000..].iter().all(|&s| s == 0.0));
        let c = p.tick();
        assert_eq!(c.video, VideoOut::Repeat(2));
        assert_eq!(c.silent_frames, 2000);
        let s = p.stats();
        assert_eq!((s.ticks, s.fresh_frames, s.repeated_frames, s.empty_ticks), (4, 2, 1, 1));
        assert_eq!((s.silent_ticks, s.partial_ticks), (2, 1));
        assert_eq!(s.audio_frames_out, 8000);
    }

    #[test]
    fn av_pacer_drop_oldest_shared_cap() {
        let mut p = AvPacer::<u32>::new(AvPacerConfig::wire(24, Some(2))).unwrap();
        // 3 s of A/V into a 2 s cap.
        for i in 0..72u32 {
            p.push_slice([i], &[i as f32; 4000]).unwrap();
        }
        assert_eq!(p.buffered_frames(), 48);
        assert_eq!(p.buffered_audio(), 96_000);
        assert_eq!(p.stats().dropped_frames, 24);
        assert_eq!(p.stats().dropped_samples, 48_000);
        // Oldest dropped on both lanes, so they are still aligned.
        let t = p.tick();
        assert_eq!(t.video, VideoOut::Fresh(24));
        assert!(t.audio.unwrap().iter().all(|&s| s == 24.0));
    }

    #[test]
    fn av_pacer_video_only_has_no_audio_lane() {
        let mut p = AvPacer::<u32>::new(AvPacerConfig::wire(16, None)).unwrap();
        p.push_slice([9], &[1.0; 3000]).unwrap(); // ignored
        let t = p.tick();
        assert_eq!(t.audio, None);
        assert_eq!(t.video, VideoOut::Fresh(9));
        assert_eq!(p.stats().audio_frames_out, 0);
    }

    #[test]
    fn av_pacer_hold_black() {
        let mut p = AvPacer::<u32>::new(AvPacerConfig::wire(24, Some(1))).unwrap();
        p.push_slice([1, 2, 3], &[0.1; 6000]).unwrap();
        p.tick();
        p.hold(0);
        assert_eq!(p.tick().video, VideoOut::Repeat(0));
        assert_eq!(p.buffered_frames(), 0);
    }

    /// Deterministic xorshift for the long-run tests.
    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
        fn below(&mut self, n: u64) -> u64 {
            self.next() % n
        }
    }

    /// §7.4: after N ticks the audio sent is exactly `N·48000/fps` sample
    /// frames, across underruns, idle gaps, clip boundaries and overflow.
    fn long_run(fps: u32, channels: u8, hours: f64, seed: u64) {
        let mut p = AvPacer::<u64>::new(AvPacerConfig::wire(fps, Some(channels))).unwrap();
        let spt = u64::from(p.samples_per_tick());
        let ticks = (hours * 3600.0 * f64::from(fps)) as u64;
        let mut rng = Rng(seed);
        let mut sent_audio = 0u64;
        let mut ticks_done = 0u64;
        let mut next_frame_id = 0u64;
        let mut fresh_ids = Vec::new();
        while ticks_done < ticks {
            // A clip of random legal-ish length (H3 17n+5; SF-Wan blocks of 12).
            let frames = if fps == 24 { 17 * (7 + rng.below(14)) as u32 + 5 } else { 12 * (1 + rng.below(8)) as u32 };
            // Wire audio already fitted to the frame count (`prepare_clip_audio`).
            let n = clip_samples(u64::from(frames), fps, 48_000) as usize;
            let audio = vec![0.25f32; n * channels as usize];
            for sl in slices(frames, EMIT_FRAMES, fps, 48_000) {
                let a = &audio[sl.sample_lo as usize * channels as usize..sl.sample_hi as usize * channels as usize];
                let ids: Vec<u64> = (0..sl.frames()).map(|k| next_frame_id + u64::from(k)).collect();
                next_frame_id += u64::from(sl.frames());
                p.push_slice(ids, a).unwrap();
                // Emit about one slice worth of ticks, sometimes fewer (build
                // ahead) or more (underrun).
                let emit = match rng.below(10) {
                    0 => 0,
                    1 => sl.frames() + 2,
                    _ => sl.frames(),
                };
                for _ in 0..emit {
                    let t = p.tick();
                    sent_audio += t.audio.as_ref().unwrap().len() as u64 / u64::from(channels);
                    if let VideoOut::Fresh(id) = t.video {
                        fresh_ids.push(id);
                    }
                    ticks_done += 1;
                }
            }
            // Idle gap between clips: freeze + silence.
            for _ in 0..rng.below(u64::from(fps) * 3) {
                let t = p.tick();
                sent_audio += t.audio.as_ref().unwrap().len() as u64 / u64::from(channels);
                if let VideoOut::Fresh(id) = t.video {
                    fresh_ids.push(id);
                }
                ticks_done += 1;
            }
            assert_eq!(sent_audio, ticks_done * spt, "audio drifted from the tick count");
        }
        let s = p.stats();
        assert_eq!(s.audio_frames_out, s.ticks * spt);
        assert_eq!(s.ticks, ticks_done);
        // Frames come out in order even with drops.
        assert!(fresh_ids.windows(2).all(|w| w[0] < w[1]));
        // RTP clocks derived from the same counter agree exactly.
        let mut v = crate::clock::VideoRtpClock::fixed(fps, 0);
        let mut a = crate::clock::AudioRtpClock::new(0);
        for _ in 0..s.ticks {
            v.advance();
            a.advance(spt);
        }
        assert_eq!(u64::from(v.current()), (s.ticks * 90_000 / u64::from(fps)) % (1 << 32));
        assert_eq!(a.samples(), s.ticks * spt);
        // Audio time equals video time exactly.
        assert_eq!(a.samples() * u64::from(fps), s.ticks * 48_000);
    }

    #[test]
    fn lockstep_long_run_h3_24fps_stereo() {
        long_run(24, 2, 1.0, 0x9e37_79b9_7f4a_7c15);
    }

    #[test]
    fn lockstep_long_run_h3_24fps_mono() {
        long_run(24, 1, 0.5, 42);
    }

    #[test]
    fn lockstep_long_run_sfwan_16fps() {
        long_run(16, 2, 1.0, 7);
    }

    #[test]
    fn metronome_no_drift_no_burst() {
        let mut m = Metronome::new(24.0);
        let mut now = 0.0;
        let mut ticks = 0u64;
        // Poll every 1 ms for 60 s: exactly 24·60 ticks (+1 for t=0), no drift.
        while now < 60.0 - 1e-9 {
            if m.poll(now).is_ok() {
                ticks += 1;
            }
            now += 0.001;
        }
        assert_eq!(ticks, 24 * 60);
        // A 100 ms stall is caught up (2 extra ticks back to back).
        let mut m = Metronome::new(24.0);
        assert!(m.poll(0.0).is_ok());
        let mut burst = 0;
        while m.poll(0.1).is_ok() {
            burst += 1;
        }
        assert_eq!(burst, 2);
        assert_eq!(m.resnaps(), 0);
        // A 5 s stall resnaps: one tick, then back to the period.
        let mut m = Metronome::new(24.0);
        assert!(m.poll(0.0).is_ok());
        assert!(m.poll(5.0).is_ok());
        let wait = m.poll(5.0).unwrap_err();
        assert!((wait - 1.0 / 24.0).abs() < 1e-9);
        assert_eq!(m.resnaps(), 1);
    }
}
