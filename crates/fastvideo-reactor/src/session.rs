//! The local-runtime state machine and the one live session (reactor §3.2,
//! §3.4; design §5.2, §5.7).
//!
//! States: `CREATED` (model loading) → `READY` → `WAITING` (on
//! `/start_session`) → `STREAMING` (first connection) ↔ `ORPHANED` (last
//! connection gone) → `CLOSING` (on `/stop_session`, at a causal session's
//! length limit (design §5.2; `session_ended` with
//! [`session_limit_reason`]), or after
//! `orphan_timeout` in WAITING/ORPHANED) → `READY`. A failed model load is
//! `TERMINATED`. One process hosts exactly one session, with the fixed id
//! [`SESSION_ID`].
//!
//! On entering CLOSING the farewell goes out first (`moderation` for a
//! moderated stop, else `session_ended{reason}` when a reason was given),
//! then every connection closes and the engine session is released.

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use fastvideo_engine_service::{CausalControl, PacedStream};
use fastvideo_protocol::{
    draw_seed, ApiError, CausalLimits, Continuity, DuplexSpec, EndReason, ModelCaps, SessionContext, SessionSpec,
    StreamCaps, TrackSet,
};
use fastvideo_webrtc::host::{PeerHandle, RtcHost};
use fastvideo_webrtc::writer::VideoCodec;
use serde_json::{json, Value};

use crate::avatar::{AvatarDriver, AvatarSettings};
use crate::causal::CausalDriver;
use crate::clip::{aspect_canvas, ClipDriver};
use crate::commands::{ClipBounds, CommandTable};
use crate::driver::{Driver, Outbox};
use crate::duplex::{DuplexDriver, Inputs};
use crate::engine::{LoadState, Mode, StreamEngine};
use crate::journal::Journal;
use crate::media::{sendable_codecs, H264Backend, MediaConfig, MediaPipeline};
use fastvideo_media::pipe::SparePool;
use crate::schema;
use crate::uploads::Uploads;
use crate::wire::ServerMsg;

/// RT's fixed local session id (`runner.py:SESSION_ID`).
pub const SESSION_ID: &str = "00000000-0000-0000-0000-000000000000";
/// Track names (reactor §4.5: reusing them keeps existing frontends working).
pub const VIDEO_TRACK: &str = "main_video";
pub const AUDIO_TRACK: &str = "main_audio";
/// RT's drain reason (`_DRAIN_CLOSE_REASON`).
pub const DRAIN_REASON: &str = "Session ended: the server is shutting down.";
/// `session_ended.reason` when a causal session reaches its length limit.
pub fn session_limit_reason(seconds: u32) -> String {
    format!("Session ended: the {seconds} s session length limit was reached.")
}
/// How long a closing session waits for its driver (engine release).
pub const DRIVER_CLOSE_TIMEOUT: Duration = Duration::from_secs(10);
/// RT's moderated-stop notice.
pub const MODERATION_MESSAGE: &str = "Session terminated due to policy violation.";

/// Who may start and join a duplex session (design §5.11: client media
/// goes to authenticated sessions only). Called with the request headers;
/// `Err(reason)` answers 401. RT has no auth, so clip and causal sessions
/// never call it.
#[derive(Clone)]
pub struct IngestAuth(pub Arc<IngestCheck>);

/// The check behind an [`IngestAuth`].
pub type IngestCheck = dyn Fn(&axum::http::HeaderMap) -> Result<(), String> + Send + Sync;

impl std::fmt::Debug for IngestAuth {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("IngestAuth(..)")
    }
}

/// Runtime settings (RT env equivalents in brackets).
#[derive(Clone, Debug)]
pub struct ReactorConfig {
    /// Model id/served name to stream; `None`: the first resident model
    /// with stream caps.
    pub model: Option<String>,
    /// Short edge the canvas resolves at (default: the model's default tier).
    pub short_edge: Option<u32>,
    /// Initial aspect (`16:9`).
    pub aspect: String,
    /// Session seed; `None`: `params.seed`, else drawn.
    pub seed: Option<u64>,
    /// WAITING/ORPHANED → CLOSING after this (`ORPHAN_TIMEOUT_SECONDS`, 60).
    pub orphan_timeout: Duration,
    /// Watchdog: a connection silent this long is lost
    /// (`WEBRTC_CLIENT_PING_TIMEOUT_SECONDS`, 20), polled every
    /// `watchdog_interval` (2 s).
    pub ping_timeout: Duration,
    pub watchdog_interval: Duration,
    /// Concurrent peer connections (`max_connections`, 64).
    pub max_connections: usize,
    /// Registered connection ids (RT buffers candidates for at most 128).
    pub max_registered: usize,
    /// Buffered trickle candidates per connection (256).
    pub max_candidates: usize,
    /// H.264 encoder for H.264 peers.
    pub h264: H264Backend,
    pub h264_bitrate_bps: Option<u32>,
    pub vp8_quality: f32,
    /// How long outbound messages wait for the first inbound frame to latch
    /// the wire version before the offer header's seed is used.
    pub latch_grace: Duration,
    /// `server_info.server_version`.
    pub server_version: String,
    /// Causal-mode session length (design §5.2): `/start_session`
    /// `max_seconds`, else the default, at most the hard ceiling; a `reset`
    /// restarts the clock up to the ceiling in all. Duplex sessions follow
    /// the same rule.
    pub causal_limits: CausalLimits,
    /// Duplex sessions: the check on `/start_session`, `/stop_session`,
    /// `POST connections` and `sdp_params` (`None`: open, as RT).
    pub ingest_auth: Option<IngestAuth>,
    /// Command set override: `None` follows the model's stream caps (clip
    /// or causal); `Some(Mode::Avatar)` serves the script avatar (Reactor
    /// `ltx`) on an image-to-video model with audio.
    pub mode: Option<Mode>,
    /// Avatar-mode settings.
    pub avatar: AvatarSettings,
}

impl Default for ReactorConfig {
    fn default() -> Self {
        Self {
            model: None,
            short_edge: None,
            aspect: "16:9".into(),
            seed: None,
            orphan_timeout: Duration::from_secs(60),
            ping_timeout: Duration::from_secs(20),
            watchdog_interval: Duration::from_secs(2),
            max_connections: 64,
            max_registered: 128,
            max_candidates: 256,
            h264: H264Backend::Nvenc,
            h264_bitrate_bps: None,
            vp8_quality: 70.0,
            latch_grace: Duration::from_secs(2),
            server_version: format!("fastvideo-rs {}", env!("CARGO_PKG_VERSION")),
            causal_limits: CausalLimits::default(),
            ingest_auth: None,
            mode: None,
            avatar: AvatarSettings::default(),
        }
    }
}

/// The RT session states (lower-cased on the wire).
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "lowercase")]
pub enum RtState {
    Created,
    Ready,
    Waiting,
    Streaming,
    Orphaned,
    Closing,
    Terminated,
}

impl RtState {
    pub fn as_str(self) -> &'static str {
        match self {
            RtState::Created => "created",
            RtState::Ready => "ready",
            RtState::Waiting => "waiting",
            RtState::Streaming => "streaming",
            RtState::Orphaned => "orphaned",
            RtState::Closing => "closing",
            RtState::Terminated => "terminated",
        }
    }
    /// A session is running (the signalling routes are open).
    pub fn is_running(self) -> bool {
        matches!(self, RtState::Waiting | RtState::Streaming | RtState::Orphaned)
    }
}

/// An answer's life on `GET …/sdp_params`.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum Answer {
    None,
    Pending,
    Ready(String),
    Taken,
    Failed(String),
}

pub(crate) struct Conn {
    /// Bumped by every offer; a stale negotiation is discarded.
    pub gen: u64,
    pub answer: Answer,
    pub candidates: Vec<String>,
    pub peer: Option<PeerHandle>,
    /// Track name → mid, from the offer's `track_mapping`.
    pub mapping: HashMap<String, String>,
}

/// The live session.
pub(crate) struct Live {
    pub epoch: u64,
    pub caps: ModelCaps,
    pub tracks: TrackSet,
    pub table: CommandTable,
    pub openapi: Value,
    pub driver: Arc<dyn Driver>,
    pub media: Arc<MediaPipeline>,
    pub out: Outbox,
    pub codecs: Vec<VideoCodec>,
    pub conns: Mutex<HashMap<u32, Conn>>,
    pub connected: Mutex<HashSet<u32>>,
    /// Duplex mode: the input tracks and their publishers.
    pub inputs: Option<Arc<Inputs>>,
}

pub(crate) struct St {
    pub phase: RtState,
    pub live: Option<Arc<Live>>,
    pub epoch: u64,
    /// Bumped on every WAITING/STREAMING/ORPHANED transition; an orphan
    /// timer fires only if nothing moved since it was armed.
    pub orphan_gen: u64,
}

pub(crate) struct Inner {
    pub cfg: ReactorConfig,
    pub engine: Arc<dyn StreamEngine>,
    pub host: RtcHost,
    pub st: Mutex<St>,
    pub op: tokio::sync::Mutex<()>,
    pub journal: Journal,
    /// The next session's warm encoder spare, started while READY (the
    /// session's canvas, fps and codec follow from the loaded model), so
    /// the first video frame does not wait for ffmpeg to start. Taken by
    /// `start_session`; a new one is started once the session ended.
    pub next_spares: Mutex<Option<SparePool>>,
    /// Client uploads (reactor §3.5), cleared when a session ends.
    pub uploads: Arc<Uploads>,
}

pub(crate) fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

/// An HTTP-level refusal: status and `{"detail": …}`.
#[derive(Clone, Debug, PartialEq)]
pub struct Refusal {
    pub status: u16,
    pub detail: String,
    pub retry_after: Option<u32>,
}

impl Refusal {
    pub fn new(status: u16, detail: impl Into<String>) -> Self {
        Self { status, detail: detail.into(), retry_after: None }
    }
}

/// The Reactor local runtime. Cheap to clone.
#[derive(Clone)]
pub struct Reactor {
    pub(crate) inner: Arc<Inner>,
}

impl std::fmt::Debug for Reactor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Reactor").field("state", &self.state()).finish()
    }
}

/// Clip-length bounds from caps (design §5.7).
pub fn clip_bounds(caps: &ModelCaps, fps: u32) -> ClipBounds {
    let fps = f64::from(fps.max(1));
    let (min_s, max_s) = match caps.stream {
        Some(StreamCaps::Clip { min_s, max_s }) => (f64::from(min_s), f64::from(max_s)),
        _ => (f64::from(caps.frames.min) / fps, f64::from(caps.frames.max) / fps),
    };
    ClipBounds { min_s, max_s, default_s: f64::from(caps.frames.default) / fps }
}

impl Reactor {
    /// A runtime over `engine`, answering offers on `host`.
    pub fn new(cfg: ReactorConfig, engine: Arc<dyn StreamEngine>, host: RtcHost) -> Self {
        let me = Self {
            inner: Arc::new(Inner {
                cfg,
                engine,
                host,
                st: Mutex::new(St { phase: RtState::Ready, live: None, epoch: 0, orphan_gen: 0 }),
                op: tokio::sync::Mutex::new(()),
                journal: Journal::default(),
                next_spares: Mutex::new(None),
                uploads: Arc::new(Uploads::new(
                    std::env::temp_dir().join("fv-reactor-uploads").join(uuid::Uuid::new_v4().simple().to_string()),
                )),
            }),
        };
        // Warm the first session's encoder once the model has loaded.
        if me.inner.engine.load_state() == LoadState::Ready {
            me.warm_next_session();
        } else if let Ok(rt) = tokio::runtime::Handle::try_current() {
            let weak = Arc::downgrade(&me.inner);
            rt.spawn(async move {
                loop {
                    let Some(inner) = weak.upgrade() else { return };
                    match inner.engine.load_state() {
                        LoadState::Ready => {
                            Reactor { inner }.warm_next_session();
                            return;
                        }
                        LoadState::Failed(_) => return,
                        _ => {}
                    }
                    drop(inner);
                    tokio::time::sleep(Duration::from_millis(100)).await;
                }
            });
        }
        me
    }

    /// The media settings of a session of `caps`.
    fn media_config(&self, caps: &ModelCaps, audio: bool) -> MediaConfig {
        let cfg = &self.inner.cfg;
        MediaConfig {
            h264: cfg.h264,
            bitrate_bps: cfg.h264_bitrate_bps,
            vp8_quality: cfg.vp8_quality,
            ..MediaConfig::new(Self::fps(caps), audio, self.canvas(caps))
        }
    }

    /// Starts the next session's warm encoder spare (READY, model loaded).
    fn warm_next_session(&self) {
        let Some(caps) = self.model() else { return };
        if self.table(&caps).is_none() {
            return;
        }
        let pool = SparePool::per_session();
        if !pool.enabled() {
            return;
        }
        crate::media::prewarm(&pool, &self.media_config(&caps, false));
        *lock(&self.inner.next_spares) = Some(pool);
    }

    pub fn config(&self) -> &ReactorConfig {
        &self.inner.cfg
    }

    pub fn host(&self) -> &RtcHost {
        &self.inner.host
    }

    pub fn journal(&self) -> &Journal {
        &self.inner.journal
    }

    /// The current state (CREATED/TERMINATED come from the engine).
    pub fn state(&self) -> RtState {
        let phase = lock(&self.inner.st).phase;
        match self.inner.engine.load_state() {
            LoadState::Failed(_) => RtState::Terminated,
            LoadState::Loading if phase == RtState::Ready => RtState::Created,
            _ => phase,
        }
    }

    /// The streamed model's caps, once loaded.
    pub fn model(&self) -> Option<ModelCaps> {
        if self.inner.engine.load_state() != LoadState::Ready {
            return None;
        }
        self.inner.engine.stream_model(self.inner.cfg.model.as_deref())
    }

    fn fps(caps: &ModelCaps) -> u32 {
        caps.fps.default
    }

    /// The delivered canvas (the tracks' size).
    fn canvas(&self, caps: &ModelCaps) -> (u32, u32) {
        if self.mode(caps) == Some(Mode::Avatar) {
            return self.inner.cfg.avatar.size;
        }
        let short = self
            .inner
            .cfg
            .short_edge
            .or_else(|| caps.canvas.short_edges.first().copied())
            .unwrap_or(480);
        aspect_canvas(&caps.canvas, &self.inner.cfg.aspect, short)
            .unwrap_or((short * 16 / 9 / 2 * 2, short))
    }

    /// The tracks a session of `caps` carries (mono audio, design §5.3).
    pub fn tracks(&self, caps: &ModelCaps) -> TrackSet {
        TrackSet::for_model(caps, self.canvas(caps), Self::fps(caps), (VIDEO_TRACK, AUDIO_TRACK), 1, false)
    }

    /// The mode a session of `caps` runs: the configured override when the
    /// model can serve it, else the model's own.
    pub fn mode(&self, caps: &ModelCaps) -> Option<Mode> {
        match self.inner.cfg.mode {
            Some(Mode::Avatar) if Mode::avatar_capable(caps) => Some(Mode::Avatar),
            Some(Mode::Avatar) => {
                tracing::warn!(model = %caps.id, "reactor mode avatar: the model has no image-to-video with audio");
                None
            }
            _ => Mode::of(caps),
        }
    }

    /// The command table for `caps`.
    pub fn table(&self, caps: &ModelCaps) -> Option<CommandTable> {
        let mode = self.mode(caps)?;
        Some(CommandTable::for_mode(mode, clip_bounds(caps, Self::fps(caps))))
    }

    /// The upload store.
    pub fn uploads(&self) -> &Arc<Uploads> {
        &self.inner.uploads
    }

    /// The OpenAPI document of a session (avatar mode adds its limits).
    fn openapi(&self, caps: &ModelCaps, table: &CommandTable, tracks: &TrackSet) -> Value {
        let cfg = &self.inner.cfg;
        // Duplex models add their input tracks and caps (design §5.11).
        let inputs = (table.mode == Mode::Duplex).then(|| Self::input_caps(caps).map(crate::duplex::input_schema)).flatten();
        let mut doc =
            schema::openapi_with_inputs(&caps.id.0, &cfg.server_version, table, tracks, &cfg.causal_limits, inputs.as_ref());
        if table.mode == Mode::Avatar {
            use fastvideo_engine_service::stream::avatar::{TAKE_MAX_S, TAKE_MIN_S};
            doc["x-reactor"]["session_limits"] = json!({
                "take_min_s": TAKE_MIN_S, "take_max_s": TAKE_MAX_S,
                "session_max_s": cfg.avatar.session_max_s, "clock": "video"
            });
            doc["x-reactor"]["avatar"] = json!({
                "window_s": cfg.avatar.window_s, "width": cfg.avatar.size.0, "height": cfg.avatar.size.1,
                "fps": Self::fps(caps), "speech": "native (the model speaks the script); set_voice_audio drives it with a file"
            });
        }
        doc
    }

    /// `GET /schema`: the OpenAPI document (`{}` before the model loads).
    pub fn schema(&self) -> Value {
        if let Some(l) = &lock(&self.inner.st).live {
            return l.openapi.clone();
        }
        match self.model() {
            Some(caps) => match self.table(&caps) {
                Some(t) => self.openapi(&caps, &t, &self.tracks(&caps)),
                None => json!({}),
            },
            None => json!({}),
        }
    }

    /// A duplex model's input caps.
    pub fn input_caps(caps: &ModelCaps) -> Option<&fastvideo_protocol::InputCaps> {
        caps.stream.as_ref().and_then(StreamCaps::duplex).map(|d| &d.input)
    }

    /// Whether the live session (else the model to stream) is duplex.
    pub fn is_duplex(&self) -> bool {
        match self.live() {
            Some(l) => l.inputs.is_some(),
            None => self.model().is_some_and(|c| Self::input_caps(&c).is_some()),
        }
    }

    /// The [`IngestAuth`] check, for duplex sessions only.
    pub fn authorize(&self, headers: &axum::http::HeaderMap) -> Result<(), Refusal> {
        let Some(auth) = &self.inner.cfg.ingest_auth else { return Ok(()) };
        if !self.is_duplex() {
            return Ok(());
        }
        (auth.0)(headers).map_err(|e| Refusal::new(401, e))
    }

    /// `capabilities.tracks`, client perspective (`recvonly` = model out).
    pub fn client_tracks(tracks: &TrackSet) -> Value {
        let mut v = vec![json!({"name": tracks.video.name, "kind": "video", "direction": "recvonly"})];
        if let Some(a) = &tracks.audio {
            v.push(json!({"name": a.name, "kind": "audio", "direction": "recvonly"}));
        }
        Value::Array(v)
    }

    /// `track_map` of `POST /connections`, model perspective.
    pub fn track_map(tracks: &TrackSet) -> Value {
        let mut m = serde_json::Map::new();
        m.insert(tracks.video.name.clone(), json!({"kind": "video", "direction": "out", "rate": 0.0}));
        if let Some(a) = &tracks.audio {
            m.insert(a.name.clone(), json!({"kind": "audio", "direction": "out", "rate": f64::from(a.rate)}));
        }
        Value::Object(m)
    }

    /// `track_map` of a live session: its output tracks and, in duplex
    /// mode, its input tracks (`in`).
    pub(crate) fn live_track_map(live: &Live) -> Value {
        let mut m = Self::track_map(&live.tracks);
        if let (Some(i), Value::Object(o)) = (&live.inputs, &mut m) {
            i.track_map(o);
        }
        m
    }

    /// The session descriptor (`runner.py:descriptor`).
    pub fn descriptor(&self) -> Value {
        let state = self.state();
        let (caps, tracks) = match &lock(&self.inner.st).live {
            Some(l) => (Some(l.caps.clone()), Some(l.tracks.clone())),
            None => (None, None),
        };
        let caps = caps.or_else(|| self.model());
        let tracks = tracks.or_else(|| caps.as_ref().map(|c| self.tracks(c)));
        let inputs = caps.as_ref().and_then(Self::input_caps).map(crate::duplex::client_input_tracks);
        let mut d = json!({
            "session_id": SESSION_ID,
            "state": state.as_str(),
            "cluster": "local",
            "model": {"name": caps.as_ref().map(|c| c.id.0.clone()).unwrap_or_default()},
            "server_info": {"server_version": self.inner.cfg.server_version},
            "selected_transport": {"protocol": "webrtc", "version": "1.0"},
            "recording": {"enabled": false, "chunk_seconds": 4},
        });
        if let Some(t) = tracks {
            let mut all = Self::client_tracks(&t);
            if let (Value::Array(a), Some(i)) = (&mut all, inputs) {
                a.extend(i);
            }
            d["capabilities"] = json!({"protocol_version": "v0", "tracks": all, "commands": []});
        }
        d
    }

    fn transition(&self, st: &mut St, to: RtState, event: &str, detail: Value) {
        let from = st.phase;
        st.phase = to;
        if matches!(to, RtState::Waiting | RtState::Streaming | RtState::Orphaned) {
            st.orphan_gen += 1;
        }
        self.inner.journal.record(event, from.as_str(), to.as_str(), detail);
    }

    fn arm_orphan_timer(&self, epoch: u64, gen: u64) {
        let me = self.clone();
        let after = self.inner.cfg.orphan_timeout;
        tokio::spawn(async move {
            tokio::time::sleep(after).await;
            let fire = {
                let st = lock(&me.inner.st);
                st.epoch == epoch
                    && st.orphan_gen == gen
                    && matches!(st.phase, RtState::Waiting | RtState::Orphaned)
            };
            if fire {
                tracing::info!("reactor session orphaned for {after:?}: closing");
                me.close(None, false, "timeout").await;
            }
        });
    }

    /// `POST /start_session`.
    pub async fn start_session(&self, params: Value) -> Result<Value, Refusal> {
        let _op = self.inner.op.lock().await;
        let state = self.state();
        match state {
            RtState::Ready => {}
            RtState::Created => {
                return Err(Refusal {
                    retry_after: Some(1),
                    ..Refusal::new(503, "cannot start session while created: the model is loading")
                })
            }
            RtState::Terminated => return Err(Refusal::new(503, "cannot start session while terminated")),
            s => return Err(Refusal::new(409, format!("cannot start session while {}", s.as_str()))),
        }
        let caps = self
            .model()
            .ok_or_else(|| Refusal::new(503, "cannot start session: no streamable model is loaded"))?;
        let table = self.table(&caps).ok_or_else(|| Refusal::new(503, "the model has no stream caps"))?;
        let fps = Self::fps(&caps);
        let canvas = self.canvas(&caps);
        let tracks = self.tracks(&caps);
        if tracks.has_audio() {
            tracks.samples_per_frame().map_err(|e| Refusal::new(503, e.message))?;
        }
        let seed = self
            .inner
            .cfg
            .seed
            .or_else(|| params.get("seed").and_then(Value::as_u64))
            .unwrap_or_else(draw_seed);
        // Causal sessions always have a length (design §5.2); clip sessions
        // run until stopped.
        let max_seconds = match table.mode {
            Mode::Causal | Mode::Duplex => {
                let asked = match params.get("max_seconds") {
                    None | Some(Value::Null) => None,
                    Some(v) => Some(
                        v.as_u64()
                            .and_then(|n| u32::try_from(n).ok())
                            .ok_or_else(|| Refusal::new(400, "max_seconds must be a whole number of seconds"))?,
                    ),
                };
                Some(self.inner.cfg.causal_limits.resolve(asked).map_err(|e| Refusal::new(400, e.message))?)
            }
            Mode::Avatar => {
                let cap = self.inner.cfg.avatar.session_max_s;
                match params.get("max_seconds") {
                    None | Some(Value::Null) => Some(cap),
                    Some(v) => Some(
                        v.as_u64()
                            .and_then(|n| u32::try_from(n).ok())
                            .filter(|n| (1..=cap).contains(n))
                            .ok_or_else(|| Refusal::new(400, format!("max_seconds must be 1..={cap}")))?,
                    ),
                }
            }
            Mode::Clip => None,
        };
        // Avatar: generated on the model's padded canvas, delivered cropped.
        let gen_canvas = match table.mode {
            Mode::Avatar => {
                let (w, h) = canvas;
                fastvideo_protocol::resolve_canvas(&fastvideo_protocol::CanvasSpec::Exact { width: w, height: h }, &caps, None)
                    .map(|(gw, gh, _)| (gw, gh))
                    .map_err(|e| Refusal::new(503, format!("avatar canvas {w}x{h}: {}", e.message)))?
            }
            _ => canvas,
        };
        let spec = SessionSpec {
            model: caps.id.clone(),
            tracks: tracks.clone(),
            canvas: gen_canvas,
            fps,
            continuity: Continuity::HardCut,
            max_seconds,
            seed: Some(seed),
        };
        let cfg = &self.inner.cfg;
        let out = Outbox::default();
        let refuse = |e: ApiError| {
            let status = e.kind.http_status();
            let mut r = Refusal::new(
                if status == 429 { 409 } else { status },
                format!("cannot start session: {}", e.message),
            );
            if status == 503 {
                r.retry_after = Some(1);
            }
            r
        };
        let context: SessionContext = match params.get("context") {
            None | Some(Value::Null) => SessionContext::default(),
            Some(v) => serde_json::from_value(v.clone()).map_err(|e| Refusal::new(400, format!("context: {e}")))?,
        };
        context.validate().map_err(|e| Refusal::new(400, format!("{}: {}", e.param.unwrap_or_default(), e.message)))?;
        let mut inputs = None;
        let (driver, paced, causal): (Arc<dyn Driver>, PacedStream, Option<CausalControl>) = match table.mode {
            Mode::Clip => {
                let session = self.inner.engine.open_clip(spec).await.map_err(refuse)?;
                let (d, p) = ClipDriver::start(session, &cfg.aspect, out.clone()).map_err(refuse)?;
                (Arc::new(d), p, None)
            }
            Mode::Causal => {
                let session = self.inner.engine.open_causal(spec).await.map_err(refuse)?;
                let (d, p) =
                    CausalDriver::start(session, seed, &cfg.causal_limits, out.clone()).map_err(refuse)?;
                let c = d.control().clone();
                (Arc::new(d), p, Some(c))
            }
            Mode::Duplex => {
                let session = self.inner.engine.open_duplex(DuplexSpec { session: spec, context }).await.map_err(refuse)?;
                let i = Arc::new(Inputs::new(&session.duplex_caps().input, session.input().clone(), max_seconds));
                i.prewarm();
                let (d, p) = DuplexDriver::start(session, out.clone(), i.clone()).map_err(refuse)?;
                inputs = Some(i);
                (Arc::new(d), p, None)
            }
            Mode::Avatar => {
                let session = self.inner.engine.open_clip(spec).await.map_err(refuse)?;
                let (d, p) = AvatarDriver::start(session, &cfg.avatar, self.inner.uploads.clone(), seed, out.clone())
                    .map_err(refuse)?;
                (Arc::new(d), p, None)
            }
        };
        let mut pace_stats = paced.stats.clone();
        let spares = lock(&self.inner.next_spares).take().unwrap_or_else(SparePool::per_session);
        let media =
            Arc::new(MediaPipeline::start_with_spares(self.media_config(&caps, tracks.has_audio()), paced.ticks, spares));
        if let Some(c) = causal {
            media.on_first_frame(move || c.mark_first_frame_sent());
        }
        let openapi = self.openapi(&caps, &table, &tracks);
        let codecs = sendable_codecs(cfg.h264);
        let (epoch, gen) = {
            let mut st = lock(&self.inner.st);
            st.epoch += 1;
            let live = Arc::new(Live {
                epoch: st.epoch,
                caps,
                tracks,
                table,
                openapi,
                driver,
                media,
                out,
                codecs,
                conns: Mutex::new(HashMap::new()),
                connected: Mutex::new(HashSet::new()),
                inputs,
            });
            st.live = Some(live);
            self.transition(&mut st, RtState::Waiting, "start_session", json!({}));
            (st.epoch, st.orphan_gen)
        };
        self.arm_orphan_timer(epoch, gen);
        // The pacer ends the session at its length limit: session_ended.
        if let Some(limit) = max_seconds {
            let me = self.clone();
            tokio::spawn(async move {
                let hit = pace_stats.wait_for(|s| s.ended.is_some()).await.is_ok_and(|s| s.ended == Some(EndReason::SessionLimit));
                if hit {
                    tracing::info!(limit, "reactor session reached its length limit: closing");
                    me.close_epoch(Some(epoch), Some(session_limit_reason(limit)), false, "session_limit").await;
                }
            });
        }
        Ok(self.descriptor())
    }

    /// `POST /stop_session`.
    pub async fn stop_session(&self, moderate: bool, reason: String) -> Result<(), Refusal> {
        match self.state() {
            s if s.is_running() => {}
            RtState::Created | RtState::Terminated => {
                return Err(Refusal::new(503, format!("cannot stop session while {}", self.state().as_str())))
            }
            s => return Err(Refusal::new(409, format!("cannot stop session while {}", s.as_str()))),
        }
        let reason = (!reason.is_empty()).then_some(reason);
        self.close(reason, moderate, "stop_session").await;
        Ok(())
    }

    /// Server shutdown: `session_ended` with RT's drain reason.
    pub async fn drain(&self) {
        if self.state().is_running() {
            self.close(Some(DRAIN_REASON.into()), false, "drain").await;
        }
    }

    /// CLOSING: farewell, close every connection, release the engine, READY.
    async fn close(&self, reason: Option<String>, moderated: bool, event: &str) {
        self.close_epoch(None, reason, moderated, event).await;
    }

    /// [`Self::close`], only if the live session is still `epoch` (when set).
    async fn close_epoch(&self, epoch: Option<u64>, reason: Option<String>, moderated: bool, event: &str) {
        let _op = self.inner.op.lock().await;
        let live = {
            let mut st = lock(&self.inner.st);
            if epoch.is_some_and(|e| st.live.as_ref().is_none_or(|l| l.epoch != e)) {
                return;
            }
            let Some(live) = st.live.take() else { return };
            self.transition(&mut st, RtState::Closing, event, json!({"reason": reason, "moderated": moderated}));
            live
        };
        if moderated {
            live.out.broadcast(ServerMsg::Moderation { action: "terminate".into(), message: MODERATION_MESSAGE.into() });
        } else if let Some(r) = &reason {
            live.out.broadcast(ServerMsg::SessionEnded { reason: r.clone() });
        }
        // Let the gateways put the farewell on the ordered channels first.
        if !live.out.is_empty() {
            tokio::time::sleep(Duration::from_millis(150)).await;
        }
        let peers: Vec<PeerHandle> = lock(&live.conns).values_mut().filter_map(|c| c.peer.take()).collect();
        let n_peers = peers.len();
        for p in peers {
            p.close();
        }
        tracing::info!(event, peers = n_peers, "reactor session closing: peers closed");
        // The driver releases the engine session; bounded, so a stuck
        // driver cannot leave the runtime in CLOSING for good.
        if tokio::time::timeout(DRIVER_CLOSE_TIMEOUT, live.driver.close()).await.is_err() {
            tracing::warn!(event, "reactor session closing: the driver did not close within {DRIVER_CLOSE_TIMEOUT:?}; going on");
        }
        tracing::info!(event, "reactor session closing: driver closed");
        live.media.close();
        tracing::info!(event, "reactor session closing: media closed");
        self.inner.uploads.clear();
        {
            let mut st = lock(&self.inner.st);
            self.transition(&mut st, RtState::Ready, "cleanup_complete", json!({}));
        }
        // The session's spare ends with its media thread; warm one for the
        // next session.
        self.warm_next_session();
    }

    /// A peer reached connected (gateway).
    pub(crate) fn connection_opened(&self, live: &Arc<Live>, conn: u32) {
        let n = {
            let mut c = lock(&live.connected);
            c.insert(conn);
            c.len()
        };
        {
            let mut st = lock(&self.inner.st);
            if st.epoch == live.epoch && matches!(st.phase, RtState::Waiting | RtState::Orphaned) {
                self.transition(&mut st, RtState::Streaming, "connection_opened", json!({"connection_id": conn}));
            }
        }
        live.driver.peers_changed(n);
    }

    /// A peer went away (gateway).
    pub(crate) fn connection_closed(&self, live: &Arc<Live>, conn: u32) {
        let n = {
            let mut c = lock(&live.connected);
            if !c.remove(&conn) {
                return;
            }
            c.len()
        };
        live.driver.peers_changed(n);
        if n == 0 {
            let arm = {
                let mut st = lock(&self.inner.st);
                if st.epoch == live.epoch && st.phase == RtState::Streaming {
                    self.transition(&mut st, RtState::Orphaned, "connection_closed", json!({"connection_id": conn}));
                    Some((st.epoch, st.orphan_gen))
                } else {
                    None
                }
            };
            if let Some((e, g)) = arm {
                self.arm_orphan_timer(e, g);
            }
        }
    }

    pub(crate) fn live(&self) -> Option<Arc<Live>> {
        lock(&self.inner.st).live.clone()
    }

    /// Whether `live` is still the current session.
    pub(crate) fn is_current(&self, live: &Live) -> bool {
        lock(&self.inner.st).live.as_ref().is_some_and(|l| l.epoch == live.epoch)
    }
}
