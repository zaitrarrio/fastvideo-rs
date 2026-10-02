//! One director session (design §5.2 lifecycle, §5.6 mapping).
//!
//! `Starting` (engine clip session opened for admission, answer returned)
//! → `Ready` (control channel open, `session_info` sent) → `Streaming`
//! (first chunk playing) → `Closing` (farewell: `stream_exhausted` for
//! `stop` / `session_limit`, a session-failure `error` otherwise) →
//! `Closed(reason)`: the engine session is released, the peer closed.
//!
//! The session task runs the [`Control`] state machine, dispatches one
//! chunk build at a time (continuity `AnchorLastFrame`: chunk N+1 starts
//! from chunk N's last frame, written as a PNG in the session directory;
//! the duplicated anchor frame is trimmed from playout), prepares each
//! built chunk for lockstep playout (48 kHz stereo, exact sample count,
//! 20 ms crossfades) and hands it to the playout thread ([`super::media`]).
//! It keeps at most `buffer_chunks` built chunks queued ahead of the one
//! playing.
//!
//! On a causal model (LongLive, SF-Wan) there are no chunk builds: one
//! rollout ([`DirectorStream`]) runs for the session, every block goes to
//! playout as it arrives, a `chunk` is reported every `causal_chunk_blocks`
//! blocks, and the control's direction is handed to the engine block by
//! block (`set_prompt`, applied at the next block boundary). The rollout is
//! paused while more than `causal_lead_seconds` of video is queued
//! (docs/serve/director-causal.md).

use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use fastvideo_media::crossfade::apply_clip_fades;
use fastvideo_media::lockstep::{prepare_clip_audio, WIRE_RATE};
use fastvideo_protocol::{ApiError, Continuity, EndReason, MediaKind, MediaRef, ModelCaps, SessionSpec, SessionState};
use fastvideo_serve_kit::ServeCtx;
use fastvideo_webrtc::channel::{ChannelMessage, FAL_CONTROL};
use fastvideo_webrtc::host::{CloseReason, PeerEvent, PeerHandle};
use serde_json::{json, Value};
use tokio::sync::mpsc;

use super::control::{CausalLimits, ChunkPlan, Control};
use super::engine::{ChunkBuild, DirectorClips, DirectorStream, StreamBlock};
use super::info::{session_info, InfoFacts};
use super::media::{self, MediaCmd, MediaConfig, MediaEvent, MediaGauges, MediaSink, PreparedChunk, VideoCodec};
use super::messages::{self as m, ClientMessage, ErrorCode, RejectReason, ScriptBeat};
use super::service::{canvas_for, DirectorConfig};

/// Shared, observable state of a session (the registry keeps it).
#[derive(Debug)]
pub struct SessionHandle {
    pub id: String,
    /// Bridge sessions expect `/wma/session/heartbeat`; runner sessions
    /// (`/start-session`) live as long as their SSE response.
    pub heartbeats: bool,
    last_beat: Mutex<Instant>,
    state: Mutex<SessionState>,
    end: tokio::sync::Notify,
    closed: tokio::sync::watch::Sender<bool>,
}

impl SessionHandle {
    pub fn new(id: String, heartbeats: bool) -> Arc<Self> {
        let (closed, _) = tokio::sync::watch::channel(false);
        Arc::new(Self {
            id,
            heartbeats,
            last_beat: Mutex::new(Instant::now()),
            state: Mutex::new(SessionState::Starting),
            end: tokio::sync::Notify::new(),
            closed,
        })
    }

    /// A heartbeat: `false` once the session is closing or closed.
    pub fn beat(&self) -> bool {
        if !self.is_open() {
            return false;
        }
        *self.last_beat.lock().expect("beat lock") = Instant::now();
        true
    }

    pub fn since_beat(&self) -> Duration {
        self.last_beat.lock().expect("beat lock").elapsed()
    }

    pub fn state(&self) -> SessionState {
        self.state.lock().expect("state lock").clone()
    }

    pub fn is_open(&self) -> bool {
        !matches!(self.state(), SessionState::Closing | SessionState::Closed(_))
    }

    fn set_state(&self, s: SessionState) {
        let mut cur = self.state.lock().expect("state lock");
        if cur.can_transition_to(&s) {
            *cur = s;
        }
    }

    /// Ask the session to end as `ClientGone` (the runner's SSE dropped).
    pub fn end(&self) {
        self.end.notify_one();
    }

    /// Resolves when the session has closed.
    pub fn closed(&self) -> tokio::sync::watch::Receiver<bool> {
        self.closed.subscribe()
    }
}

/// Sends encoded media to the peer.
#[derive(Clone)]
struct PeerSink(PeerHandle);

impl MediaSink for PeerSink {
    fn video(&mut self, data: bytes::Bytes, rtp_time: u64, _keyframe: bool) {
        let _ = self.0.send_video(fastvideo_webrtc::writer::VideoFrame { data, rtp_time });
    }
    fn audio(&mut self, data: bytes::Bytes, rtp_time: u64) {
        let _ = self.0.send_audio(fastvideo_webrtc::writer::AudioPacket { data, rtp_time });
    }
}

/// Everything a new session needs.
pub struct SessionInit {
    pub cfg: Arc<DirectorConfig>,
    pub ctx: ServeCtx,
    pub clips: Arc<dyn DirectorClips>,
    /// The causal rollout of a causal model (`clips` is then [`NoClips`]).
    pub stream: Option<Arc<dyn DirectorStream>>,
    pub peer: PeerHandle,
    pub events: mpsc::UnboundedReceiver<PeerEvent>,
    pub codec: VideoCodec,
    pub handle: Arc<SessionHandle>,
    pub facts: InfoFacts,
    pub continuity: Continuity,
}

/// A finished build, prepared for playout.
struct Built {
    plan: ChunkPlan,
    chunk: PreparedChunk,
    generated: u32,
    trimmed: u32,
    anchor: Option<PathBuf>,
    gen_s: f64,
    prep_s: f64,
    ready_at: Instant,
}

/// Rolling phase history for `session_metrics`.
#[derive(Default)]
struct Phase {
    samples: Vec<f64>,
}

impl Phase {
    const LIMIT: usize = 64;
    fn push(&mut self, ms: f64) {
        if self.samples.len() == Self::LIMIT {
            self.samples.remove(0);
        }
        self.samples.push(ms);
    }
    fn summary(&self) -> Value {
        let mut v = self.samples.clone();
        v.sort_by(|a, b| a.total_cmp(b));
        let q = |p: f64| if v.is_empty() { 0.0 } else { v[((v.len() - 1) as f64 * p).round() as usize] };
        json!({
            "total_ms": v.iter().sum::<f64>(),
            "p50_ms": q(0.5),
            "p95_ms": q(0.95),
            "max_ms": v.last().copied().unwrap_or(0.0),
            "count": v.len(),
        })
    }
}

/// The blocks of the director chunk being reported.
#[derive(Default)]
struct ChunkAcc {
    blocks: u32,
    first_block: u64,
    frames: u32,
    gen_ms: f64,
    recache_ms: f64,
    recaches: u32,
    prompt_version: u64,
}

/// A causal session's rollout state.
struct Causal {
    stream: Arc<dyn DirectorStream>,
    limits: CausalLimits,
    /// The prompt last handed to the engine (`None` before the first).
    engine_prompt: Option<String>,
    /// `(engine version, client prompt_version)`, ascending.
    versions: VecDeque<(u64, u64)>,
    /// `prompt_applied` owed once a block of that engine version arrives.
    owed: VecDeque<(u64, Vec<Value>)>,
    /// Blocks received.
    blocks: u64,
    /// Frames handed to playout.
    pushed_frames: u64,
    acc: ChunkAcc,
    paused: bool,
    recaches: u64,
    recache_ms: f64,
}

struct Session {
    init: SessionInit,
    control: Control,
    clips: Arc<dyn DirectorClips>,
    media: std::sync::mpsc::Sender<MediaCmd>,
    media_rx: mpsc::UnboundedReceiver<MediaEvent>,
    gauges: Arc<MediaGauges>,
    build_tx: mpsc::UnboundedSender<Result<Built, ApiError>>,
    build_rx: mpsc::UnboundedReceiver<Result<Built, ApiError>>,
    building: bool,
    anchor: Option<PathBuf>,
    /// The configured generation canvas (`None` until `configure`).
    canvas: Option<(u32, u32)>,
    /// The delivered size when the generation canvas is padded (LTX):
    /// frames are centre-cropped to it.
    crop: Option<(u32, u32)>,
    dir: PathBuf,
    started: Instant,
    info_sent: bool,
    channel_open_at: Option<Instant>,
    chunks_ready: u32,
    /// Chunks whose first frame has entered playout.
    chunks_started: u32,
    last_ready: Option<Instant>,
    gen_ema: Option<f64>,
    phases: [(&'static str, Phase); 3],
    last_metrics: Instant,
    video_seconds: f64,
    underruns: u64,
    ending: Option<EndReason>,
    causal: Option<Causal>,
    block_rx: mpsc::UnboundedReceiver<Result<StreamBlock, ApiError>>,
}

fn ms(d: f64) -> f64 {
    (d * 1_000_000.0).round() / 1000.0
}

/// Runs a session to completion (spawned by the service).
pub async fn run(init: SessionInit) {
    let gauges = Arc::new(MediaGauges::default());
    let (ev_tx, media_rx) = mpsc::unbounded_channel();
    let audio = init.clips.spec().tracks.has_audio();
    let mcfg = MediaConfig {
        fps: init.clips.spec().fps,
        audio,
        codec: init.codec,
        h264: init.cfg.h264,
        video_bitrate: init.cfg.video_bitrate,
        gop_seconds: 2.0,
    };
    let dir = init.cfg.work_dir.join(format!("wma-{}", init.handle.id));
    let handle = init.handle.clone();
    let media = match media::start(mcfg, PeerSink(init.peer.clone()), ev_tx, gauges.clone()) {
        Ok(tx) => tx,
        Err(e) => {
            tracing::error!(error = %e, "director playout failed to start");
            init.peer.close();
            handle.set_state(SessionState::Closed(EndReason::Error(ApiError::internal(e))));
            handle.closed.send_replace(true);
            return;
        }
    };
    let (build_tx, build_rx) = mpsc::unbounded_channel();
    let control = Control::new(init.facts.limits.clone());
    let clips = init.clips.clone();
    let (block_tx, block_rx) = mpsc::unbounded_channel();
    let causal = match (&init.stream, &init.facts.limits.causal) {
        (Some(st), Some(limits)) => {
            // The rollout's blocks, forwarded to the session loop.
            let st2 = st.clone();
            tokio::spawn(async move {
                loop {
                    match st2.next_block().await {
                        Some(r) => {
                            if block_tx.send(r).is_err() {
                                return;
                            }
                        }
                        None => {
                            let _ = block_tx.send(Err(ApiError::engine_failed("the causal rollout ended")));
                            return;
                        }
                    }
                }
            });
            Some(Causal {
                stream: st.clone(),
                limits: limits.clone(),
                engine_prompt: None,
                versions: VecDeque::new(),
                owed: VecDeque::new(),
                blocks: 0,
                pushed_frames: 0,
                acc: ChunkAcc::default(),
                paused: false,
                recaches: 0,
                recache_ms: 0.0,
            })
        }
        _ => {
            drop(block_tx);
            None
        }
    };
    let mut s = Session {
        init,
        control,
        clips,
        media,
        media_rx,
        gauges,
        build_tx,
        build_rx,
        building: false,
        anchor: None,
        canvas: None,
        crop: None,
        dir,
        started: Instant::now(),
        info_sent: false,
        channel_open_at: None,
        chunks_ready: 0,
        chunks_started: 0,
        last_ready: None,
        gen_ema: None,
        phases: [("generation", Phase::default()), ("prepare", Phase::default()), ("delivery", Phase::default())],
        last_metrics: Instant::now(),
        video_seconds: 0.0,
        underruns: 0,
        ending: None,
        causal,
        block_rx,
    };
    handle.set_state(SessionState::Ready);
    let reason = s.run_loop().await;
    s.finish(reason).await;
}

impl Session {
    fn send(&self, v: Value) {
        let text = v.to_string();
        if let Err(e) = self.init.peer.post_message(ChannelMessage::text(FAL_CONTROL, text)) {
            tracing::debug!(error = %e, "control message dropped");
        }
    }

    fn fail(&mut self, code: ErrorCode, msg: impl Into<String>, version: Option<u64>) {
        let msg = msg.into();
        self.send(m::error(code, msg.clone(), version));
        if code.is_session_failure() && self.ending.is_none() {
            self.ending = Some(EndReason::Error(ApiError::engine_failed(msg)));
        }
    }

    async fn run_loop(&mut self) -> EndReason {
        let mut tick = tokio::time::interval(Duration::from_millis(250));
        tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        let handle = self.init.handle.clone();
        loop {
            tokio::select! {
                ev = self.init.events.recv() => match ev {
                    Some(e) => self.on_peer(e).await,
                    None => self.end(EndReason::ClientGone),
                },
                Some(b) = self.build_rx.recv() => self.on_built(b),
                Some(b) = self.block_rx.recv() => self.on_block(b),
                Some(e) = self.media_rx.recv() => self.on_media(e),
                _ = tick.tick() => self.on_tick(),
                _ = handle.end.notified() => self.end(EndReason::ClientGone),
            }
            if let Some(r) = self.ending.take() {
                return r;
            }
            self.maybe_dispatch();
        }
    }

    fn end(&mut self, r: EndReason) {
        if self.ending.is_none() {
            self.ending = Some(r);
        }
    }

    fn on_tick(&mut self) {
        let h = &self.init.handle;
        if h.heartbeats && h.since_beat() > self.init.cfg.heartbeat_timeout {
            tracing::info!(session = %h.id, "three heartbeats missed: closing");
            return self.end(EndReason::ClientGone);
        }
        if !self.control.is_configured() {
            if let Some(t) = self.channel_open_at {
                if t.elapsed() > self.init.cfg.configure_timeout {
                    return self.fail(ErrorCode::ConfigurationTimeout, "no `configure` arrived in time", None);
                }
            }
        }
        self.throttle();
        if self.last_metrics.elapsed() >= Duration::from_secs(10) && self.control.is_configured() {
            let v = self.session_metrics(false);
            self.send(v);
        }
    }

    async fn on_peer(&mut self, e: PeerEvent) {
        match e {
            PeerEvent::Connected | PeerEvent::KeyframeRequest { .. } => {
                let _ = self.media.send(MediaCmd::Keyframe);
            }
            PeerEvent::ChannelOpen { label } if label == FAL_CONTROL => {
                if !self.info_sent {
                    self.info_sent = true;
                    self.channel_open_at = Some(Instant::now());
                    self.send(session_info(&self.init.facts));
                }
            }
            PeerEvent::ChannelClose { label } if label == FAL_CONTROL => self.end(EndReason::ClientGone),
            PeerEvent::Message(msg) if msg.label == FAL_CONTROL => match msg.as_text() {
                Some(t) => {
                    let t = t.to_owned();
                    self.on_text(&t).await
                }
                None => self.send(m::error(ErrorCode::InvalidMessage, "control messages are JSON text", None)),
            },
            PeerEvent::Closed(r) => {
                tracing::info!(session = %self.init.handle.id, reason = ?r, "director peer closed");
                self.end(match r {
                    CloseReason::HostShutdown => EndReason::Evicted,
                    _ => EndReason::ClientGone,
                });
            }
            _ => {}
        }
    }

    async fn stage(&self, url: &str, param: &str) -> Result<PathBuf, String> {
        self.stage_sized(url, param).await.map(|(p, _)| p)
    }

    /// [`Self::stage`], plus the image's upright size when known.
    async fn stage_sized(&self, url: &str, param: &str) -> Result<(PathBuf, Option<(u32, u32)>), String> {
        let r = MediaRef::parse(url, param).map_err(|e| e.message)?;
        let dir = self.dir.join("inputs");
        let ctx = &self.init.ctx;
        ctx.ingestor()
            .stage_one(&r, MediaKind::Image, &self.init.cfg.ingest, &dir, param, ctx.now())
            .await
            .map(|m| (m.path, m.probe.dims()))
            .map_err(|e| e.message)
    }

    async fn stage_script(&self, beats: &[ScriptBeat]) -> Result<Vec<Option<PathBuf>>, String> {
        let mut out = Vec::with_capacity(beats.len());
        for (i, b) in beats.iter().enumerate() {
            out.push(match &b.end_image_url {
                Some(u) => Some(self.stage(u, &format!("script[{i}].end_image_url")).await?),
                None => None,
            });
        }
        Ok(out)
    }

    async fn on_text(&mut self, text: &str) {
        let msg = match m::parse(text) {
            Ok(x) => x,
            Err(inv) => return self.send(m::error(ErrorCode::InvalidMessage, inv.error, inv.prompt_version)),
        };
        match msg {
            ClientMessage::Ping(ts) => self.send(m::pong(ts)),
            ClientMessage::NetworkInfo { request_id } => {
                let st = self.init.peer.stats();
                let path = json!({
                    "session_id": self.init.handle.id,
                    "observed_at_ms": time::OffsetDateTime::now_utc().unix_timestamp_nanos() / 1_000_000,
                    "available": false,
                    "reason": "this runner does not report its selected candidate pair",
                    "connection_state": if st.connected { "connected" } else { "connecting" },
                    "ice_connection_state": if st.connected { "connected" } else { "checking" },
                });
                self.send(m::network_info(&request_id, path));
            }
            ClientMessage::Stop => {
                if self.control.stop() {
                    self.send(m::stream_exhausted(self.chunks_ready, "stopped"));
                    self.end(EndReason::Stopped);
                }
            }
            ClientMessage::Configure(c) => {
                if let Err((reply, fatal)) = self.control.precheck_configure(&c) {
                    self.send(reply.clone());
                    if fatal {
                        self.end(EndReason::Error(ApiError::invalid(reply["error"].as_str().unwrap_or_default())));
                    }
                    return;
                }
                let v = Some(c.prompt_version);
                let (image, image_dims) = match &c.image_url {
                    Some(u) => match self.stage_sized(u, "image_url").await {
                        Ok((p, d)) => (Some(p), d),
                        Err(e) => return self.fail(ErrorCode::InvalidInitialImage, e, v),
                    },
                    None => (None, None),
                };
                let (end_image, end_dims) = match &c.end_image_url {
                    Some(u) => match self.stage_sized(u, "end_image_url").await {
                        Ok((p, d)) => (Some(p), d),
                        Err(e) => return self.fail(ErrorCode::InvalidInitialImage, e, v),
                    },
                    None => (None, None),
                };
                let script_images = match &c.script {
                    Some(b) => match self.stage_script(b).await {
                        Ok(x) => x,
                        Err(e) => return self.fail(ErrorCode::InvalidInitialScript, e, v),
                    },
                    None => Vec::new(),
                };
                // Every chunk is built on the configured canvas.
                let res = c.resolution.unwrap_or(self.control.default_resolution());
                // No `aspect_ratio`: the session follows the opening image
                // (else the end image), at the nearest aspect the director
                // serves; with no image, 16:9 (fal's default).
                let mut c = c;
                if c.aspect_ratio.is_none() {
                    if let Some((w, h)) = image_dims.or(end_dims) {
                        let a = m::Aspect::nearest(f64::from(w) / f64::from(h));
                        tracing::info!(session = %self.init.handle.id, "director canvas: aspect {} follows the {w}x{h} image", a.as_str());
                        c.aspect_ratio = Some(a);
                    }
                }
                let aspect = c.aspect_ratio.unwrap_or(m::Aspect::Landscape);
                let (w, h, crop) = canvas_for(self.clips.caps(), res, aspect);
                tracing::info!(session = %self.init.handle.id, res = res.as_str(), aspect = aspect.as_str(), "director canvas: generate {w}x{h}, deliver {:?}", crop.unwrap_or((w, h)));
                self.canvas = Some((w, h));
                self.crop = crop;
                if let Some(b) = c.audio_bitrate {
                    let _ = self.media.send(MediaCmd::AudioBitrate(b));
                }
                let reply = self.control.configure(&c, image, end_image, script_images);
                self.send(reply);
                if let Some(cz) = &self.causal {
                    // Seed before the first prompt (it restarts the rollout),
                    // then block 0's direction starts generation.
                    if let Some(seed) = self.control.settings().and_then(|s| s.seed) {
                        cz.stream.set_seed(seed);
                    }
                    self.causal_next();
                }
            }
            ClientMessage::Prompt(p) => {
                if let Err(out) = self.control.precheck_prompt(&p) {
                    for o in out {
                        self.send(o);
                    }
                    return;
                }
                let v = p.prompt_version;
                let end_image = match &p.end_image_url {
                    Some(u) => match self.stage(u, "end_image_url").await {
                        Ok(x) => Some(x),
                        Err(e) => return self.send(m::prompt_rejected(v, RejectReason::InvalidImage, e)),
                    },
                    None => None,
                };
                let script_images = match &p.script {
                    Some(b) => match self.stage_script(b).await {
                        Ok(x) => x,
                        Err(e) => return self.send(m::prompt_rejected(v, RejectReason::InvalidImage, e)),
                    },
                    None => Vec::new(),
                };
                for o in self.control.prompt(&p, end_image, script_images) {
                    self.send(o);
                }
                // Causal: a replanning update goes to the engine now (the
                // next block it starts); an appended one waits its turn.
                let append = match &p.script {
                    Some(_) => p.script_mode == Some(m::ScriptMode::Append),
                    None => p.replan == Some(false),
                };
                if self.causal.is_some() && !append {
                    self.causal_next();
                }
            }
        }
    }

    fn anchoring(&self) -> bool {
        matches!(self.init.continuity, Continuity::AnchorLastFrame { .. })
    }

    fn maybe_dispatch(&mut self) {
        if self.causal.is_some() || self.building || self.ending.is_some() || !self.control.is_configured() || self.control.is_stopped() {
            return;
        }
        // Built chunks queued behind the one playing.
        let queued = self.chunks_ready.saturating_sub(self.chunks_started) as usize;
        if queued >= self.init.cfg.buffer_chunks.max(1) {
            return;
        }
        let Some((plan, applied)) = self.control.next_chunk() else { return };
        for a in applied {
            self.send(a);
        }
        let anchor = if self.anchoring() && plan.first_image.is_none() { self.anchor.clone() } else { None };
        let trimmed = u32::from(anchor.is_some());
        let first = plan.first_image.clone().or(anchor);
        let build = ChunkBuild {
            prompt: plan.prompt.clone(),
            seed: self.control.settings().and_then(|s| s.seed).map(|s| s.wrapping_add(u64::from(plan.index))),
            seconds: plan.seconds,
            canvas: self.canvas,
            first_frame: first,
            last_frame: plan.end_image.clone(),
        };
        self.building = true;
        let clips = self.clips.clone();
        let tx = self.build_tx.clone();
        let fps = self.clips.spec().fps;
        let audio = self.clips.spec().tracks.has_audio();
        let crossfade = match self.init.continuity {
            Continuity::HardCut => None,
            Continuity::Crossfade { ms } | Continuity::AnchorLastFrame { crossfade_ms: ms } => Some(ms),
        };
        let anchor_path = self.anchoring().then(|| self.dir.join(format!("anchor-{}.png", plan.index)));
        let dir = self.dir.clone();
        let crop = self.crop;
        tokio::spawn(async move {
            let t0 = Instant::now();
            let out = clips.build(build).await;
            let gen_s = t0.elapsed().as_secs_f64();
            let res = match out {
                Err(e) => Err(e),
                Ok(out) => {
                    let t1 = Instant::now();
                    let prep = Prep { fps, audio, trimmed, crossfade, anchor_path, dir, crop };
                    let prepared = tokio::task::spawn_blocking(move || prepare(plan, out, prep))
                        .await
                        .map_err(|e| ApiError::internal(format!("chunk preparation panicked: {e}")))
                        .and_then(|r| r);
                    prepared.map(|mut b| {
                        b.gen_s = gen_s;
                        b.prep_s = t1.elapsed().as_secs_f64();
                        b.ready_at = Instant::now();
                        b
                    })
                }
            };
            let _ = tx.send(res);
        });
    }

    fn on_built(&mut self, b: Result<Built, ApiError>) {
        self.building = false;
        let b = match b {
            Ok(b) => b,
            Err(e) => return self.fail(ErrorCode::GenerationFailed, e.message, None),
        };
        if self.ending.is_some() || self.control.is_stopped() {
            return;
        }
        let fps = f64::from(self.clips.spec().fps.max(1));
        let buffered_frames = self.gauges.buffered_frames.load(Ordering::Relaxed) as f64;
        let lead_s = buffered_frames / fps;
        let presented = b.chunk.frames.len() as u32;
        let playback = f64::from(presented) / fps;
        self.gen_ema = Some(match self.gen_ema {
            None => b.gen_s,
            Some(e) => 0.7 * e + 0.3 * b.gen_s,
        });
        let est = self.gen_ema.unwrap_or(b.gen_s);
        let depth_chunks = self.chunks_ready.saturating_sub(self.chunks_started) + 1;
        let interval = self.last_ready.map(|t| ms(b.ready_at.duration_since(t).as_secs_f64()));
        self.last_ready = Some(b.ready_at);
        let gen_ms = ms(b.gen_s);
        let prep_ms = ms(b.prep_s);
        self.phases[0].1.push(gen_ms);
        self.phases[1].1.push(prep_ms);
        let chunk = json!({
            "type": "chunk",
            "chunk_index": b.plan.index,
            "prompt_version": b.plan.prompt_version,
            "requested_duration_seconds": b.plan.seconds.clamp(5.0, 15.0),
            "generated_frame_count": b.generated,
            "trimmed_context_frames": b.trimmed,
            "presented_frame_count": presented,
            "native_playable_frame_count": presented,
            "playback_seconds": playback,
            "generation_seconds": b.gen_s,
            "next_generation_estimate_seconds": est,
            "buffer_depth_seconds": lead_s + playback,
            "buffer_depth_chunks": depth_chunks,
            "scheduling_lead_ms": ms(lead_s),
            "scheduling_slack_ms": ms(lead_s + playback - est),
            "route": "unknown",
            "hard_cut": b.plan.index > 0 && b.trimmed == 0 && matches!(self.init.continuity, Continuity::HardCut),
            "dispatch": {
                "overhead_ms": prep_ms,
                "wall_ms": ms(b.gen_s + b.prep_s),
                "phases_ms": {"generate": gen_ms, "prepare": prep_ms},
                "classified_ms": ms(b.gen_s + b.prep_s),
            },
        });
        self.send(chunk);
        self.send(json!({
            "type": "chunk_metrics",
            "chunk_index": b.plan.index,
            "route": "unknown",
            "units": "ms",
            "gauges": {
                "buffer_depth_ms": ms(lead_s + playback),
                "buffer_depth_chunks": depth_chunks,
                "generated_frames": b.generated,
                "presented_frames": presented,
            },
            "phases_ms": {"generate": gen_ms, "prepare": prep_ms},
            "chunk_consumable_ready_ms": ms(b.ready_at.duration_since(self.started).as_secs_f64()),
            "chunk_consumable_interval_ms": interval,
        }));
        self.chunks_ready += 1;
        self.anchor = b.anchor;
        let _ = self.media.send(MediaCmd::Chunk(b.chunk));
    }

    /// Causal: the control's direction for the next block the engine
    /// starts; a changed prompt goes to the engine (applied at its next
    /// block boundary), and the `prompt_applied` it carries is owed until a
    /// block generated with it arrives.
    fn causal_next(&mut self) {
        if self.ending.is_some() {
            return;
        }
        let Some((plan, applied)) = self.control.next_chunk() else { return };
        let Some(cz) = self.causal.as_mut() else { return };
        if cz.engine_prompt.as_deref() != Some(plan.prompt.as_str()) {
            let v = cz.stream.set_prompt(&plan.prompt);
            tracing::info!(session = %self.init.handle.id, engine_version = v, prompt_version = plan.prompt_version, "director: prompt to the causal engine");
            cz.engine_prompt = Some(plan.prompt);
            cz.versions.push_back((v, plan.prompt_version));
            if !applied.is_empty() {
                cz.owed.push_back((v, applied));
            }
        } else {
            // The same text: nothing changes in the picture, applied now.
            if let Some(last) = cz.versions.back_mut() {
                last.1 = last.1.max(plan.prompt_version);
            }
            for a in applied {
                self.send(a);
            }
        }
    }

    /// Causal: pause the rollout while `causal_lead_seconds` of video is
    /// queued for playout, resume below it.
    fn throttle(&mut self) {
        let fps = f64::from(self.clips.spec().fps.max(1));
        let fresh = self.gauges.fresh_frames.load(Ordering::Relaxed);
        let lead_target = self.init.cfg.causal_lead_seconds;
        let Some(cz) = self.causal.as_mut() else { return };
        let lead = cz.pushed_frames.saturating_sub(fresh) as f64 / fps;
        let pause = lead >= lead_target;
        if pause != cz.paused {
            cz.paused = pause;
            cz.stream.set_paused(pause);
        }
    }

    /// Causal: one block of the rollout, straight to playout.
    fn on_block(&mut self, r: Result<StreamBlock, ApiError>) {
        let b = match r {
            Ok(b) => b,
            Err(e) => {
                if self.ending.is_none() && !self.control.is_stopped() {
                    self.fail(ErrorCode::GenerationFailed, e.message, None);
                }
                return;
            }
        };
        if self.ending.is_some() || self.control.is_stopped() || b.frames.is_empty() {
            return;
        }
        let fps = f64::from(self.clips.spec().fps.max(1));
        let id = self.init.handle.id.clone();
        let Some(cz) = self.causal.as_mut() else { return };
        // `prompt_applied` for every version this block carries.
        let mut out = Vec::new();
        while cz.owed.front().is_some_and(|(v, _)| *v <= b.prompt_version) {
            out.extend(cz.owed.pop_front().map(|x| x.1).unwrap_or_default());
        }
        while cz.versions.len() > 1 && cz.versions[1].0 <= b.prompt_version {
            cz.versions.pop_front();
        }
        let version = cz.versions.front().filter(|(v, _)| *v <= b.prompt_version).map_or(1, |x| x.1);
        let block = cz.blocks;
        cz.blocks += 1;
        let n = b.frames.len() as u32;
        if cz.acc.blocks == 0 {
            cz.acc.first_block = block;
        }
        cz.acc.blocks += 1;
        cz.acc.frames += n;
        cz.acc.gen_ms += b.block_ms;
        cz.acc.prompt_version = cz.acc.prompt_version.max(version);
        if b.recache_ms > 0.0 {
            cz.acc.recache_ms += b.recache_ms;
            cz.acc.recaches += 1;
            cz.recaches += 1;
            cz.recache_ms += b.recache_ms;
            tracing::info!(session = %id, block, prompt_version = version, recache_ms = b.recache_ms, "director: KV re-cache before block");
        }
        cz.pushed_frames += u64::from(n);
        let chunk_blocks = cz.limits.chunk_blocks.max(1);
        let done = cz.acc.blocks >= chunk_blocks;
        let acc = if done { Some(std::mem::take(&mut cz.acc)) } else { None };
        for o in out {
            self.send(o);
        }
        let _ = self.media.send(MediaCmd::Chunk(PreparedChunk { index: block as u32, frames: b.frames, audio: None }));
        if let Some(acc) = acc {
            self.causal_chunk(acc, fps);
        }
        // The next block's direction, then the pacing.
        self.causal_next();
        self.throttle();
    }

    /// Causal: the `chunk` / `chunk_metrics` of a full director chunk.
    fn causal_chunk(&mut self, acc: ChunkAcc, fps: f64) {
        let Some(cz) = self.causal.as_ref() else { return };
        let fresh = self.gauges.fresh_frames.load(Ordering::Relaxed);
        let lead_s = cz.pushed_frames.saturating_sub(fresh) as f64 / fps;
        let chunk_s = f64::from(cz.limits.block_frames * cz.limits.chunk_blocks.max(1)) / fps;
        let paused = cz.paused;
        let gen_s = acc.gen_ms / 1000.0;
        let playback = f64::from(acc.frames) / fps;
        self.gen_ema = Some(match self.gen_ema {
            None => gen_s,
            Some(e) => 0.7 * e + 0.3 * gen_s,
        });
        let est = self.gen_ema.unwrap_or(gen_s);
        let now = Instant::now();
        let interval = self.last_ready.map(|t| ms(now.duration_since(t).as_secs_f64()));
        self.last_ready = Some(now);
        let gen_ms = ms(gen_s);
        self.phases[0].1.push(gen_ms);
        self.phases[1].1.push(0.0);
        let index = self.chunks_ready;
        let depth_chunks = (lead_s / chunk_s).ceil() as u32;
        let pace = if gen_s > 0.0 { f64::from(acc.frames) / gen_s } else { 0.0 };
        let causal = json!({
            "blocks": acc.blocks,
            "first_block": acc.first_block,
            "block_ms_mean": ms(gen_s / f64::from(acc.blocks.max(1))),
            "generation_fps": (pace * 100.0).round() / 100.0,
            "recaches": acc.recaches,
            "recache_ms": ms(acc.recache_ms / 1000.0),
            "paused": paused,
        });
        self.send(json!({
            "type": "chunk",
            "chunk_index": index,
            "prompt_version": acc.prompt_version,
            "requested_duration_seconds": chunk_s,
            "generated_frame_count": acc.frames,
            "trimmed_context_frames": 0,
            "presented_frame_count": acc.frames,
            "native_playable_frame_count": acc.frames,
            "playback_seconds": playback,
            "generation_seconds": gen_s,
            "next_generation_estimate_seconds": est,
            "buffer_depth_seconds": lead_s,
            "buffer_depth_chunks": depth_chunks,
            "scheduling_lead_ms": ms(lead_s),
            "scheduling_slack_ms": ms(lead_s - est),
            "route": "unknown",
            "hard_cut": false,
            "dispatch": {
                "overhead_ms": 0.0,
                "wall_ms": gen_ms,
                "phases_ms": {"generate": gen_ms, "prepare": 0.0},
                "classified_ms": gen_ms,
            },
            "causal": causal,
        }));
        self.send(json!({
            "type": "chunk_metrics",
            "chunk_index": index,
            "route": "unknown",
            "units": "ms",
            "gauges": {
                "buffer_depth_ms": ms(lead_s),
                "buffer_depth_chunks": depth_chunks,
                "generated_frames": acc.frames,
                "presented_frames": acc.frames,
            },
            "phases_ms": {"generate": gen_ms, "prepare": 0.0},
            "chunk_consumable_ready_ms": ms(now.duration_since(self.started).as_secs_f64()),
            "chunk_consumable_interval_ms": interval,
        }));
        self.chunks_ready += 1;
    }

    fn on_media(&mut self, e: MediaEvent) {
        match e {
            MediaEvent::ChunkStarted { index, late_by } => {
                self.chunks_started += 1;
                self.init.handle.set_state(SessionState::Streaming);
                if let Some(late) = late_by {
                    // Causal playout units are blocks: name their chunk.
                    let chunk = match &self.causal {
                        Some(cz) => index / cz.limits.chunk_blocks.max(1),
                        None => index,
                    };
                    if index > 0 {
                        tracing::info!(session = %self.init.handle.id, chunk, playout_unit = index, late_by = late, "director: deadline missed");
                        self.send(m::deadline_missed(chunk, late));
                    }
                }
            }
            MediaEvent::Underrun { .. } => self.underruns += 1,
            MediaEvent::Emitted { video_seconds } => {
                self.video_seconds = video_seconds;
                if let Some(max) = self.init.facts.max_session_seconds {
                    if video_seconds >= max as f64 && self.control.stop() {
                        self.send(m::stream_exhausted(self.chunks_ready, "session_limit"));
                        self.end(EndReason::SessionLimit);
                    }
                }
            }
            MediaEvent::Failed(err) => self.fail(ErrorCode::GenerationFailed, format!("encoding failed: {err}"), None),
        }
    }

    fn session_metrics(&mut self, final_: bool) -> Value {
        self.last_metrics = Instant::now();
        let g = &self.gauges;
        let phases: serde_json::Map<String, Value> =
            self.phases.iter().map(|(n, p)| ((*n).to_owned(), p.summary())).collect();
        let causal = self.causal.as_ref().map(|cz| {
            json!({
                "blocks": cz.blocks,
                "recaches": cz.recaches,
                "recache_ms_total": ms(cz.recache_ms / 1000.0),
                "paused": cz.paused,
                "video_dropped": g.video_dropped.load(Ordering::Relaxed),
            })
        });
        let mut v = json!({
            "type": "session_metrics",
            "units": "ms",
            "history_limit": Phase::LIMIT,
            "history_size": self.phases[0].1.samples.len(),
            "session_wall_ms": ms(self.started.elapsed().as_secs_f64()),
            "gauges": {
                "chunks": self.chunks_ready,
                "video_seconds": self.video_seconds,
                "buffered_frames": g.buffered_frames.load(Ordering::Relaxed),
                "underruns": self.underruns,
                "video_packets": g.video_packets.load(Ordering::Relaxed),
                "audio_packets": g.audio_packets.load(Ordering::Relaxed),
                "encode_ms_total": g.encode_us.load(Ordering::Relaxed) as f64 / 1000.0,
            },
            "phases": phases,
            "final": final_,
        });
        if let Some(c) = causal {
            v["causal"] = c;
        }
        v
    }

    async fn finish(&mut self, reason: EndReason) {
        let h = self.init.handle.clone();
        h.set_state(SessionState::Closing);
        tracing::info!(session = %h.id, ?reason, chunks = self.chunks_ready, "director session closing");
        if !matches!(reason, EndReason::ClientGone | EndReason::Evicted) {
            let fm = self.session_metrics(true);
            self.send(fm);
            // Let the farewell leave before the transport closes.
            tokio::time::sleep(Duration::from_millis(300)).await;
        }
        let _ = self.media.send(MediaCmd::Stop);
        self.init.peer.close();
        // Release the engine session now, even if a build is in flight.
        self.clips.close().await;
        if let Some(cz) = &self.causal {
            tracing::info!(session = %h.id, blocks = cz.blocks, recaches = cz.recaches, underruns = self.underruns, "director causal rollout closing");
            cz.stream.close().await;
        }
        self.clips = Arc::new(NoClips(self.clips.caps().clone(), self.clips.spec().clone()));
        self.init.clips = self.clips.clone();
        let _ = tokio::fs::remove_dir_all(&self.dir).await;
        h.set_state(SessionState::Closed(reason));
        h.closed.send_replace(true);
    }
}

/// No clip session: left behind once the engine session is released, and
/// what a causal session holds instead (its builds always fail).
pub(super) struct NoClips(pub(super) ModelCaps, pub(super) SessionSpec);

#[async_trait::async_trait]
impl DirectorClips for NoClips {
    fn caps(&self) -> &ModelCaps {
        &self.0
    }
    fn spec(&self) -> &SessionSpec {
        &self.1
    }
    async fn build(&self, _: ChunkBuild) -> Result<super::engine::ChunkOutput, ApiError> {
        Err(ApiError::internal("the engine session is closed"))
    }
    async fn close(&self) {}
}

/// How to prepare one built chunk.
struct Prep {
    fps: u32,
    /// Stereo 48 kHz wire audio (else video only).
    audio: bool,
    /// Leading frames to drop (the duplicated anchor).
    trimmed: u32,
    crossfade: Option<u16>,
    /// Where to write the last frame for the next chunk's anchor.
    anchor_path: Option<PathBuf>,
    dir: PathBuf,
    /// Centre-crop every frame to this size (pad-and-crop canvases).
    crop: Option<(u32, u32)>,
}

/// Chunk preparation (blocking): anchor PNG, trim, wire audio, fades.
fn prepare(plan: ChunkPlan, out: super::engine::ChunkOutput, p: Prep) -> Result<Built, ApiError> {
    let Prep { fps, audio, trimmed, crossfade, anchor_path, dir, crop } = p;
    let mut frames = out.frames;
    let generated = frames.len() as u32;
    if generated == 0 {
        return Err(ApiError::engine_failed("the engine produced no frames"));
    }
    // The anchor is the uncropped last frame: the next chunk is generated on
    // the same padded canvas, so its first frame matches pixel for pixel
    // (a cropped anchor would be scaled up to cover the canvas: a zoom at
    // every join).
    let anchor = match anchor_path {
        Some(p) => {
            std::fs::create_dir_all(&dir).map_err(|e| ApiError::internal(format!("session dir: {e}")))?;
            let last = frames.last().expect("non-empty");
            let img = image::RgbImage::from_raw(last.width, last.height, last.data.to_vec())
                .ok_or_else(|| ApiError::internal("last frame has the wrong size"))?;
            img.save(&p).map_err(|e| ApiError::internal(format!("writing the anchor frame: {e}")))?;
            Some(p)
        }
        None => None,
    };
    let trimmed = trimmed.min(generated - 1);
    let audio = if audio {
        let wire = prepare_clip_audio(out.audio.as_ref(), generated, fps, WIRE_RATE, 2)
            .map_err(|e| ApiError::internal(format!("chunk audio: {e}")))?;
        let wire = match crossfade {
            Some(ms) if ms > 0 => apply_clip_fades(&wire, ms, plan.index > 0, true),
            _ => wire,
        };
        let spf = (WIRE_RATE / fps) as usize;
        let skip = trimmed as usize * spf * 2;
        Some(wire.samples[skip.min(wire.samples.len())..].to_vec())
    } else {
        None
    };
    frames.drain(..trimmed as usize);
    if let Some((w, h)) = crop {
        for f in &mut frames {
            *f = f.crop_center(w, h);
        }
    }
    let index = plan.index;
    Ok(Built {
        plan,
        chunk: PreparedChunk { index, frames, audio },
        generated,
        trimmed,
        anchor,
        gen_s: 0.0,
        prep_s: 0.0,
        ready_at: Instant::now(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use fastvideo_protocol::RgbFrame;

    /// A pad-and-crop chunk: frames delivered centre-cropped, the anchor
    /// saved uncropped (the next chunk generates on the padded canvas).
    #[test]
    fn prepare_crops_frames_and_keeps_the_anchor_uncropped() {
        let dir = std::env::temp_dir().join(format!("fv-director-prep-{}", uuid::Uuid::new_v4()));
        let frame = |i: u64| {
            let mut data = Vec::with_capacity(8 * 6 * 3);
            for y in 0..6u8 {
                for x in 0..8u8 {
                    data.extend_from_slice(&[x, y, i as u8]);
                }
            }
            RgbFrame::new(8, 6, data.into(), i).unwrap()
        };
        let plan = ChunkPlan { index: 1, prompt: "p".into(), seconds: 5.0, first_image: None, end_image: None, prompt_version: 1 };
        let out = super::super::engine::ChunkOutput { frames: (0..3).map(frame).collect(), audio: None };
        let anchor_path = dir.join("anchor-1.png");
        let prep = Prep { fps: 24, audio: false, trimmed: 1, crossfade: None, anchor_path: Some(anchor_path.clone()), dir: dir.clone(), crop: Some((6, 4)) };
        let b = prepare(plan, out, prep).unwrap();
        assert_eq!((b.generated, b.trimmed), (3, 1));
        assert_eq!(b.chunk.frames.len(), 2);
        for f in &b.chunk.frames {
            assert_eq!((f.width, f.height), (6, 4));
            assert_eq!(f.pixel(0, 0), Some([1, 1, f.index as u8]));
        }
        let anchor = image::open(b.anchor.as_ref().unwrap()).unwrap().to_rgb8();
        assert_eq!(anchor.dimensions(), (8, 6));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
