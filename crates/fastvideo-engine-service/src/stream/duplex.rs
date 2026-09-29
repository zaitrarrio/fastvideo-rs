//! `DuplexSession`: a model that reads the client's camera and microphone
//! while it streams (design §5.11).
//!
//! Admission is the same as for every stream (one session per executor, the
//! model resident). The session owns the model's [`InputBuffers`]: a
//! transport's ingest ([`fastvideo_webrtc::ingest`] in the front-ends)
//! pushes decoded [`InputFrame`](fastvideo_protocol::InputFrame)s and
//! [`InputAudio`](fastvideo_protocol::InputAudio) into them, and the model
//! worker reads them. [`DuplexSession::start`] starts the worker and returns
//! a [`DuplexControl`] and the same [`PacedStream`] of ticks every other
//! stream produces, so the egress side (encoders, WebRTC peers, WHIP) is
//! unchanged.
//!
//! The only duplex model today is the **loopback echo** ([`ECHO_MODEL`],
//! [`EchoBackend`]): every output tick takes the newest input frame
//! (skipping any backlog: a live model shows the present), scales it to the
//! session canvas and draws a visible overlay ([`draw_echo_overlay`]: a
//! magenta border and the tick index in 32 black/white cells), and plays
//! the microphone back (48 kHz, the session's channel count, silence when
//! there is none). Before the first input frame it shows a grey card with
//! the same overlay, so answer-side transports have a picture at once. It
//! exercises the whole ingest path end to end without a GPU. Real-time V2V
//! and avatar models plug in here as further workers.
//!
//! Limits: the session's `max_seconds` bounds the output in video time
//! (`EndReason::SessionLimit`, as causal sessions, design §5.2).

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use fastvideo_media::pacer::VideoOut;
use fastvideo_media::ring::InputBuffers;
use fastvideo_media::scale::{scale_rgb, ScaleMode};
use fastvideo_protocol::{
    ApiError, AudioCaps, AudioInputCaps, CanvasCaps, DuplexCaps, DuplexSpec, EndReason, ErrorKind, Family, FpsCaps,
    FrameGrid, InputCaps, InputVideoCodec, KnobCaps, ModelCaps, ModelId, RefLimits, ResolvedJob,
    RgbFrame, SessionContext, StreamCaps, VideoInputCaps, WIRE_AUDIO_RATE,
};
use serde::{Deserialize, Serialize};
use tokio::sync::{watch, Notify};

use super::pace::{tick_queue, PaceStats, PacedStream, Tick, TICK_DEPTH};
use crate::backend::{BlockInput, BlockStats, CausalSpec, ClipOutput, ClipSink, DeviceInfo, EngineBackend, LoadEvent, SessionId};
use crate::cancel::{lock, StepControl};
use crate::service::Shared;

/// The loopback echo model id.
pub const ECHO_MODEL: &str = "fv-echo";

/// Caps of the loopback echo: 640x360 input (VP8/H.264 up to 1280x720 at
/// 30 fps, 4 Mbit/s), mono microphone, output at 24/25/30 fps with audio,
/// one frame per unit, a session context accepted (and echoed in its
/// state).
pub fn echo_caps() -> ModelCaps {
    let duplex = DuplexCaps {
        unit_ms: 33,
        target_fps: 30,
        input: InputCaps {
            video: Some(VideoInputCaps {
                width: 640,
                height: 360,
                max_width: 1280,
                max_height: 720,
                max_fps: 30,
                codecs: vec![InputVideoCodec::Vp8, InputVideoCodec::H264],
            }),
            audio: Some(AudioInputCaps { rate: WIRE_AUDIO_RATE, channels: 1 }),
            max_bitrate_kbps: 4000,
            buffer_ms: 500,
        },
        audio_out: true,
        context: true,
    };
    ModelCaps {
        id: ModelId::new(ECHO_MODEL),
        family: Family::Loopback,
        served_names: vec![ECHO_MODEL.to_owned()],
        tasks: Default::default(),
        audio: Some(AudioCaps { native_rate: WIRE_AUDIO_RATE, channels: 1, via_sidecar: false }),
        fps: FpsCaps { allowed: vec![24, 25, 30], default: 30, container_only: false },
        frames: FrameGrid::new(1, 0, 1, 1, 1),
        canvas: CanvasCaps {
            multiple: 2,
            max_area: 1280 * 720,
            aspect: (0.25, 4.0),
            short_edges: vec![360, 480, 720],
            pad_and_crop: false,
            hd: None,
        },
        refs: RefLimits::none(),
        stream: Some(StreamCaps::Duplex(duplex)),
        knobs: KnobCaps::default(),
        resident: true,
        tier: None,
        recipe: None,
    }
}

/// The executor behind [`ECHO_MODEL`]: no weights, no GPU; it only exists
/// so the echo is a served model (caps, residency, one session at a time).
/// Batch generation is refused.
#[derive(Debug, Default)]
pub struct EchoBackend;

fn echo_refusal() -> ApiError {
    ApiError::new(ErrorKind::InvalidRequest, format!("`{ECHO_MODEL}` only streams (duplex sessions)"))
}

impl EngineBackend for EchoBackend {
    fn device(&self) -> DeviceInfo {
        DeviceInfo { index: 0, name: "loopback (no device)".into(), total_memory_mb: 0 }
    }
    fn caps(&self) -> Vec<ModelCaps> {
        vec![echo_caps()]
    }
    fn load(&mut self, _model: &ModelId, _obs: &mut dyn FnMut(LoadEvent)) -> Result<(), ApiError> {
        Ok(())
    }
    fn generate(&mut self, _job: &ResolvedJob, _out: &mut dyn ClipSink, _ctl: &StepControl) -> Result<ClipOutput, ApiError> {
        Err(echo_refusal())
    }
    fn causal_open(&mut self, _s: SessionId, _spec: &CausalSpec) -> Result<(), ApiError> {
        Err(echo_refusal())
    }
    fn causal_block(
        &mut self,
        _s: SessionId,
        _input: &BlockInput,
        _out: &mut dyn ClipSink,
        _ctl: &StepControl,
    ) -> Result<BlockStats, ApiError> {
        Err(echo_refusal())
    }
    fn causal_close(&mut self, _s: SessionId) {}
}

/// Border colour of the echo overlay.
pub const ECHO_BORDER: [u8; 3] = [255, 0, 255];

/// Border width of the echo overlay for a `height`-pixel picture.
pub fn echo_border(height: u32) -> u32 {
    (height / 40).max(4) & !1
}

/// The echo overlay: a magenta border and, inside it along the top, the
/// tick `index` in 32 cells (MSB first, white = 1), each `2 * border` high.
pub fn draw_echo_overlay(src: &RgbFrame, index: u64) -> RgbFrame {
    let (w, h) = (src.width as usize, src.height as usize);
    let b = echo_border(src.height) as usize;
    let mut d = src.data.to_vec();
    let mut put = |x: usize, y: usize, c: [u8; 3]| {
        let o = (y * w + x) * 3;
        if let Some(p) = d.get_mut(o..o + 3) {
            p.copy_from_slice(&c);
        }
    };
    for y in 0..h {
        for x in 0..w {
            if x < b || y < b || x >= w.saturating_sub(b) || y >= h.saturating_sub(b) {
                put(x, y, ECHO_BORDER);
            }
        }
    }
    let inner = w.saturating_sub(2 * b);
    let cell_h = (2 * b).min(h.saturating_sub(2 * b));
    if inner >= 32 {
        for bit in 0..32usize {
            let on = (index >> (31 - bit)) & 1 == 1;
            let c = if on { [255, 255, 255] } else { [0, 0, 0] };
            let (x0, x1) = (b + bit * inner / 32, b + (bit + 1) * inner / 32);
            for y in b..b + cell_h {
                for x in x0..x1 {
                    put(x, y, c);
                }
            }
        }
    }
    RgbFrame { width: src.width, height: src.height, data: d.into(), index: src.index }
}

/// Reads the index an echo overlay carries (tests, clients): the 32 cells,
/// thresholded at mid grey.
pub fn read_echo_index(f: &RgbFrame) -> Option<u64> {
    let b = echo_border(f.height);
    let inner = f.width.checked_sub(2 * b)?;
    let y = b + b.min(f.height.saturating_sub(2 * b) / 2);
    let mut v = 0u64;
    for bit in 0..32u32 {
        let x = b + bit * inner / 32 + inner / 64;
        let p = f.pixel(x, y)?;
        let lum = (u32::from(p[0]) + u32::from(p[1]) + u32::from(p[2])) / 3;
        v = (v << 1) | u64::from(lum > 127);
    }
    Some(v)
}

/// The duplex command set (Reactor duplex mode, native `commands`).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", content = "data", rename_all = "snake_case")]
pub enum DuplexCommand {
    SetPaused { paused: bool },
    GetState,
}

/// `state_update` of a duplex session.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct DuplexState {
    pub paused: bool,
    pub context: SessionContext,
    pub unit_ms: u32,
    pub fps: u32,
    /// Output ticks so far.
    pub frames_out: u64,
    /// Input frames shown (each at most once).
    pub input_frames: u64,
    /// Input frames skipped because newer ones were already buffered.
    pub input_skipped: u64,
    /// Input frames dropped by a full ring.
    pub input_dropped: u64,
    pub input_audio_chunks: u64,
    /// Arrival in the ring to output, for the last input frame shown.
    pub input_latency_ms: Option<f64>,
    pub has_input: bool,
    /// The centre pixel of the last input frame shown (a quick "is my
    /// camera arriving" check for clients and tests).
    pub last_input_centre: Option<[u8; 3]>,
    /// Source size of the last input frame shown.
    pub last_input_source: Option<(u32, u32)>,
}

/// The answer to a [`DuplexCommand`].
#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(tag = "type", content = "data", rename_all = "snake_case")]
pub enum DuplexReply {
    StateUpdate(DuplexState),
    CommandError { command: String, reason: String },
}

#[derive(Debug, Default)]
struct Ctl {
    state: Mutex<DuplexState>,
    closed: Mutex<bool>,
    wake: Notify,
}

/// Cloneable control handle of a running duplex session.
#[derive(Clone, Debug)]
pub struct DuplexControl {
    ctl: Arc<Ctl>,
    input: Arc<InputBuffers>,
}

impl DuplexControl {
    pub fn state(&self) -> DuplexState {
        let mut s = lock(&self.ctl.state).clone();
        let v = self.input.video.stats();
        s.input_dropped = v.dropped_full;
        s.input_skipped = v.skipped_behind;
        s
    }

    pub fn set_paused(&self, paused: bool) {
        lock(&self.ctl.state).paused = paused;
    }

    /// Runs one command.
    pub fn apply(&self, cmd: DuplexCommand) -> DuplexReply {
        match cmd {
            DuplexCommand::SetPaused { paused } => self.set_paused(paused),
            DuplexCommand::GetState => {}
        }
        DuplexReply::StateUpdate(self.state())
    }

    /// The input rings (the transport's ingest writes here).
    pub fn input(&self) -> &Arc<InputBuffers> {
        &self.input
    }

    /// Stops the worker; the pacer ends (`Stopped`) and the session is
    /// released.
    pub fn close(&self) {
        *lock(&self.ctl.closed) = true;
        self.input.close();
        self.ctl.wake.notify_waiters();
    }

    pub fn is_closed(&self) -> bool {
        *lock(&self.ctl.closed)
    }
}

/// An admitted duplex session. Dropping it releases the executor slot and
/// closes the input.
pub struct DuplexSession {
    engine: Arc<Shared>,
    id: SessionId,
    executor: usize,
    spec: DuplexSpec,
    caps: ModelCaps,
    duplex: DuplexCaps,
    input: Arc<InputBuffers>,
}

impl std::fmt::Debug for DuplexSession {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DuplexSession")
            .field("id", &self.id)
            .field("model", &self.spec.session.model)
            .field("executor", &self.executor)
            .finish()
    }
}

impl DuplexSession {
    pub(crate) fn new(
        engine: Arc<Shared>,
        id: SessionId,
        executor: usize,
        spec: DuplexSpec,
        caps: ModelCaps,
    ) -> Result<Self, ApiError> {
        let duplex = match caps.stream.as_ref().and_then(StreamCaps::duplex) {
            Some(d) => d.clone(),
            None => {
                engine.close_clip_session(id);
                return Err(ApiError::invalid_param("model", format!("`{}` is not a duplex model", caps.id)));
            }
        };
        let input = InputBuffers::for_caps(&duplex.input);
        Ok(Self { engine, id, executor, spec, caps, duplex, input })
    }

    pub fn id(&self) -> SessionId {
        self.id
    }

    pub fn spec(&self) -> &DuplexSpec {
        &self.spec
    }

    pub fn caps(&self) -> &ModelCaps {
        &self.caps
    }

    pub fn duplex_caps(&self) -> &DuplexCaps {
        &self.duplex
    }

    /// The input rings the transport's ingest writes into.
    pub fn input(&self) -> &Arc<InputBuffers> {
        &self.input
    }

    /// Starts the model worker (needs a tokio runtime): its output ticks
    /// and the control handle. The worker owns the session; it ends at
    /// [`DuplexControl::close`], at `max_seconds` of output, or when the
    /// tick receiver is dropped.
    pub fn start(self) -> Result<(DuplexControl, PacedStream), ApiError> {
        if self.caps.family != Family::Loopback {
            return Err(ApiError::new(
                ErrorKind::InvalidRequest,
                format!("no duplex worker for `{}` yet (only the loopback echo streams duplex)", self.caps.id),
            ));
        }
        let ctl = Arc::new(Ctl::default());
        {
            let mut s = lock(&ctl.state);
            s.context = self.spec.context.clone();
            s.unit_ms = self.duplex.unit_ms;
            s.fps = self.spec.session.fps;
        }
        let control = DuplexControl { ctl: ctl.clone(), input: self.input.clone() };
        let paced = spawn_echo(self, ctl)?;
        Ok((control, paced))
    }
}

impl Drop for DuplexSession {
    fn drop(&mut self) {
        self.input.close();
        self.engine.close_clip_session(self.id);
    }
}

/// Microphone audio held for playback at most (drop oldest).
const ECHO_AUDIO_MAX_MS: u64 = 250;

fn spawn_echo(session: DuplexSession, ctl: Arc<Ctl>) -> Result<PacedStream, ApiError> {
    let spec = session.spec.session.clone();
    let (w, h) = spec.canvas;
    let fps = spec.fps;
    if fps == 0 || w == 0 || h == 0 {
        return Err(ApiError::invalid_param("fps", "a duplex session needs a canvas and an fps"));
    }
    let out_ch = spec.tracks.audio.as_ref().map(|a| a.channels);
    let spf = match out_ch {
        Some(_) => spec.tracks.samples_per_frame()? as usize,
        None => 0,
    };
    let (tx, rx) = tick_queue(TICK_DEPTH);
    let (first_tx, first_rx) = watch::channel(false);
    let (stats_tx, stats_rx) = watch::channel(PaceStats::default());
    let task = tokio::spawn(async move {
        let input = session.input.clone();
        let mut interval = tokio::time::interval(Duration::from_secs_f64(1.0 / f64::from(fps)));
        interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
        let waiting = RgbFrame::solid(w, h, [64, 64, 64], 0);
        let mut last: Option<RgbFrame> = None;
        let mut fifo: VecDeque<f32> = VecDeque::new();
        let fifo_max = out_ch.map_or(0, |c| (WIRE_AUDIO_RATE as u64 * ECHO_AUDIO_MAX_MS / 1000) as usize * usize::from(c));
        let mut stats = PaceStats::default();
        let mut index = 0u64;
        let ended = loop {
            tokio::select! {
                _ = interval.tick() => {}
                _ = ctl.wake.notified() => {}
            }
            if *lock(&ctl.closed) {
                break EndReason::Stopped;
            }
            let paused = lock(&ctl.state).paused;
            // Microphone in: always drained, so it never goes stale.
            for a in input.audio.drain() {
                let pcm = &a.item.pcm;
                lock(&ctl.state).input_audio_chunks += 1;
                if paused {
                    continue;
                }
                if let Some(c) = out_ch {
                    let pcm = if pcm.channels == c {
                        pcm.clone()
                    } else if c == 1 {
                        pcm.to_mono()
                    } else {
                        pcm.to_mono().upmix(c)
                    };
                    fifo.extend(pcm.samples.iter().copied());
                }
            }
            while fifo.len() > fifo_max {
                fifo.pop_front();
            }
            let fresh_input = if paused { None } else { input.video.take_latest() };
            let video = match fresh_input {
                Some(t) => {
                    let src = scale_rgb(&t.item.frame, w, h, ScaleMode::Fit);
                    let f = draw_echo_overlay(&src, index);
                    {
                        let mut s = lock(&ctl.state);
                        s.input_frames += 1;
                        s.has_input = true;
                        s.input_latency_ms = Some(t.arrived.elapsed().as_secs_f64() * 1e3);
                        s.last_input_centre = t.item.frame.pixel(t.item.frame.width / 2, t.item.frame.height / 2);
                        s.last_input_source = Some(t.item.source);
                    }
                    last = Some(f.clone());
                    VideoOut::Fresh(f)
                }
                None => match &last {
                    Some(l) => VideoOut::Repeat(l.clone()),
                    None => {
                        // The waiting card, with a live counter.
                        VideoOut::Fresh(draw_echo_overlay(&waiting, index))
                    }
                },
            };
            let audio = out_ch.map(|c| {
                let n = spf * usize::from(c);
                let take = n.min(fifo.len());
                let mut v: Vec<f32> = fifo.drain(..take).collect();
                v.resize(n, 0.0);
                v
            });
            match &video {
                VideoOut::Fresh(_) => stats.fresh_frames += 1,
                _ => stats.repeated_frames += 1,
            }
            tx.send(Tick {
                index,
                video,
                audio,
                video_rtp: (index * 90_000 / u64::from(fps)) as u32,
                fps: f64::from(fps),
            });
            if index == 0 {
                first_tx.send_replace(true);
            }
            index += 1;
            lock(&ctl.state).frames_out = index;
            stats.ticks = index;
            stats.effective_fps = f64::from(fps);
            stats.unique_fps = f64::from(fps);
            stats.video_seconds = index as f64 / f64::from(fps);
            stats.limit_seconds = stats.video_seconds;
            stats_tx.send_replace(stats.clone());
            if spec.max_seconds.is_some_and(|m| stats.video_seconds >= f64::from(m)) {
                break EndReason::SessionLimit;
            }
        };
        stats.ended = Some(ended);
        stats_tx.send_replace(stats);
        tx.close();
        drop(session);
    });
    Ok(PacedStream { ticks: rx, first_frame: first_rx, stats: stats_rx, task })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn overlay_round_trips_the_index_and_keeps_the_middle() {
        let f = RgbFrame::solid(640, 360, [10, 200, 30], 0);
        for i in [0u64, 1, 77, 0xdead_beef] {
            let o = draw_echo_overlay(&f, i);
            assert_eq!(read_echo_index(&o), Some(i & 0xffff_ffff));
            assert_eq!(o.pixel(0, 0), Some(ECHO_BORDER));
            assert_eq!(o.pixel(639, 359), Some(ECHO_BORDER));
            assert_eq!(o.pixel(320, 200), Some([10, 200, 30]));
        }
        assert_eq!(echo_border(360), 8);
        assert_eq!(echo_border(96), 4);
    }

    #[test]
    fn echo_caps_are_duplex() {
        let c = echo_caps();
        let d = c.stream.as_ref().and_then(StreamCaps::duplex).unwrap();
        assert!(d.audio_out && d.context);
        let v = d.input.video.as_ref().unwrap();
        assert_eq!((v.width, v.height, v.max_fps), (640, 360, 30));
        assert!(c.fps.allows(30));
        assert_eq!(c.family, Family::Loopback);
    }
}
