//! Native `/fv/v1/streams` (design §5.1-5.2, §5.8, §9; **native** shapes,
//! WP-15): run an engine streaming session and publish it with WHIP.
//!
//! | Route | Behaviour |
//! |---|---|
//! | `POST /fv/v1/streams` | Open a session (`causal` for SF-Wan, `clip` for H3/LTX/Wan clip models), start the pacer and publish via WHIP → 201 stream object. Busy → 429 + `Retry-After`; model not resident → 503 + `Retry-After` |
//! | `GET /fv/v1/streams` | Live and recently ended streams |
//! | `GET /fv/v1/streams/{id}` | State, WHIP resource, pacer stats, TTFF phases, recent session events |
//! | `POST /fv/v1/streams/{id}/commands` | One session command, `{"type","data"}`: the clip set ([`ClipCommand`]) or the causal set ([`CausalCommand`]) → `{"reply": …}` |
//! | `DELETE /fv/v1/streams/{id}` | Stop: WHIP `DELETE`, then the session is released |
//!
//! Pipeline per stream (§5.1): session → pacer ([`spawn_causal_pacer`] /
//! [`spawn_clip_pacer`]) → **wait for the first frame** (WHIP is a
//! transport we initiate: never offer an empty track, §5.2) → WHIP offer
//! (H.264 first, video-only sessions offer no audio m-line, §5.3) → one
//! encoder per session (H.264 from `[webrtc] encoder`, `auto` = NVENC when
//! the startup probe encodes, else OpenH264, else the CPU test x264; Opus
//! 48 kHz stereo) → the published peer. PLI/FIR and tick drops
//! force an IDR (at most one per second).
//!
//! The publisher needs the `webrtc` and `http-client` features (and
//! `encoders` for Opus/OpenH264); without them `POST` answers 501.

// The registry is only driven by the publisher.
#![cfg_attr(not(all(feature = "webrtc", feature = "http-client")), allow(dead_code))]

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use axum::extract::{Path, State};
use axum::http::{HeaderMap, HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use fastvideo_engine_service::stream::{
    CausalCommand, CausalControl, ClipCommand, ClipPlayer, PaceStats, Ttff,
};
use fastvideo_engine_service::EngineService;
use fastvideo_protocol::{
    canvas_for_aspect, ApiError, Continuity, ErrorKind, ModelCaps, ModelId, ProtocolId, SessionSpec,
    StreamCaps, TrackSet,
};
use fastvideo_serve_kit::ServeCtx;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tokio::sync::{watch, Notify};

use crate::gate::ServiceGate;

/// Stream settings (from `[webrtc]` and the environment).
#[derive(Clone, Debug)]
pub struct StreamsConfig {
    /// Default WHIP target kind when a stream does not say.
    pub whip_target: Option<String>,
    /// Default WHIP bearer token (`FV_WHIP_TOKEN`).
    pub whip_token: Option<String>,
    /// `auto` | `nvenc` | `openh264` | `x264-test` (`[webrtc] encoder`,
    /// `FV_STREAM_ENCODER`; `auto` is normally resolved at startup by
    /// [`crate::encoders::resolve`]).
    pub encoder: String,
    /// STUN servers for the publisher's srflx candidate (empty: host only).
    pub stun: Vec<String>,
    /// UDP bind for the publisher's str0m host.
    pub udp_bind: std::net::SocketAddr,
    pub first_frame_timeout: Duration,
    pub connect_timeout: Duration,
    /// Ended streams kept for `GET`.
    pub keep_ended: usize,
}

impl Default for StreamsConfig {
    fn default() -> Self {
        Self {
            whip_target: None,
            whip_token: None,
            encoder: "auto".into(),
            stun: vec!["stun:stun.l.google.com:19302".into()],
            udp_bind: "0.0.0.0:0".parse().expect("static addr"),
            first_frame_timeout: Duration::from_secs(300),
            connect_timeout: Duration::from_secs(30),
            keep_ended: 32,
        }
    }
}

impl StreamsConfig {
    /// From the serve config plus `FV_STREAM_STUN` (comma-separated;
    /// `none` disables the probe).
    pub fn from_config(c: &crate::config::Config) -> Self {
        let mut s = Self {
            whip_target: c.webrtc.whip_target.clone(),
            whip_token: (!c.webrtc.whip_token.is_empty()).then(|| c.webrtc.whip_token.expose().to_owned()),
            encoder: c.webrtc.encoder.clone(),
            ..Self::default()
        };
        if let Ok(v) = std::env::var("FV_STREAM_STUN") {
            s.stun = if v.trim() == "none" {
                Vec::new()
            } else {
                v.split(',').map(|x| x.trim().to_owned()).filter(|x| !x.is_empty()).collect()
            };
        }
        s
    }
}

/// `POST /fv/v1/streams` body.
#[derive(Clone, Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StreamBody {
    pub model: String,
    /// The WHIP endpoint to publish to.
    pub whip_url: String,
    /// Bearer token (or the Basic password with `whip_user`).
    #[serde(default)]
    pub whip_token: Option<String>,
    #[serde(default)]
    pub whip_user: Option<String>,
    /// `cloudflare` | `mediamtx` (default: guessed from the URL).
    #[serde(default)]
    pub whip_target: Option<String>,
    /// Causal: the initial prompt (required). Clip: a first clip.
    #[serde(default)]
    pub prompt: Option<String>,
    /// Clip mode: clips enqueued at start.
    #[serde(default)]
    pub clips: Vec<ClipItem>,
    #[serde(default)]
    pub width: Option<u32>,
    #[serde(default)]
    pub height: Option<u32>,
    #[serde(default)]
    pub fps: Option<u32>,
    #[serde(default)]
    pub seed: Option<u64>,
    #[serde(default)]
    pub max_seconds: Option<u32>,
    /// Publish audio when the model has it (default true).
    #[serde(default)]
    pub audio: Option<bool>,
    #[serde(default)]
    pub continuity: Option<Continuity>,
    /// Clip mode: autoplay (default true).
    #[serde(default)]
    pub autoplay: Option<bool>,
}

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ClipItem {
    pub prompt: String,
    #[serde(default)]
    pub seconds: Option<f64>,
    #[serde(default)]
    pub seed: Option<u64>,
}

/// Where a stream is.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum StreamState {
    /// Session open, waiting for the first frame.
    Starting,
    /// First frame exists; WHIP offer / ICE in progress.
    Publishing,
    Streaming,
    Closed,
}

/// The status a stream reports.
#[derive(Clone, Debug, Serialize)]
pub struct StreamStatus {
    pub state: StreamState,
    pub end_reason: Option<String>,
    pub error: Option<String>,
    pub whip_resource: Option<String>,
    pub video_codec: Option<String>,
    pub encoder: Option<String>,
    pub frames_sent: u64,
    pub keyframes_sent: u64,
    pub audio_packets_sent: u64,
    /// IDRs forced on request (a forced NVENC IDR restarts ffmpeg).
    pub forced_idrs: u64,
    /// PLI/FIR received from the WHIP endpoint.
    pub keyframe_requests: u64,
    /// Requests answered by the periodic IDR or a keyframe that had just
    /// gone out, without forcing one.
    pub keyframe_requests_covered: u64,
    pub video_send_errors: u64,
    pub started_ms: u64,
    pub first_frame_ms: Option<u64>,
    pub streaming_ms: Option<u64>,
}

enum Control {
    Clip(ClipPlayer),
    Causal(CausalControl),
}

struct Entry {
    id: String,
    model: String,
    mode: &'static str,
    created_ms: u64,
    status: Mutex<StreamStatus>,
    pace: Mutex<Option<watch::Receiver<PaceStats>>>,
    events: Mutex<VecDeque<Value>>,
    control: Mutex<Option<Control>>,
    stop: Notify,
}

impl Entry {
    fn view(&self) -> Value {
        let st = self.status.lock().map(|s| s.clone()).ok();
        let pace = self
            .pace
            .lock()
            .ok()
            .and_then(|p| p.as_ref().map(|r| r.borrow().clone()));
        let (ttff, session_state) = match self.control.lock().ok().as_deref() {
            Some(Some(Control::Causal(c))) => (Some(c.ttff()), serde_json::to_value(c.state()).ok()),
            Some(Some(Control::Clip(p))) => (None, serde_json::to_value(p.state()).ok()),
            _ => (None, None),
        };
        json!({
            "id": self.id,
            "object": "fv.stream",
            "model": self.model,
            "mode": self.mode,
            "created_at": self.created_ms / 1000,
            "status": st,
            "pacer": pace.map(pace_json),
            "ttff": ttff.map(|t: Ttff| serde_json::to_value(t).unwrap_or(Value::Null)),
            "session": session_state,
            "events": self.events.lock().map(|e| e.iter().cloned().collect::<Vec<_>>()).unwrap_or_default(),
        })
    }

    fn set(&self, f: impl FnOnce(&mut StreamStatus)) {
        if let Ok(mut s) = self.status.lock() {
            f(&mut s);
        }
    }

    fn push_event(&self, v: Value) {
        if let Ok(mut e) = self.events.lock() {
            e.push_back(v);
            while e.len() > 64 {
                e.pop_front();
            }
        }
    }
}

fn pace_json(p: PaceStats) -> Value {
    json!({
        "ticks": p.ticks,
        "fresh_frames": p.fresh_frames,
        "repeated_frames": p.repeated_frames,
        "pacer_dropped": p.pacer_dropped,
        "underruns": p.underruns,
        "unique_fps": (p.unique_fps * 100.0).round() / 100.0,
        "effective_fps": (p.effective_fps * 100.0).round() / 100.0,
        "video_seconds": (p.video_seconds * 1000.0).round() / 1000.0,
        "ended": p.ended.map(|e| serde_json::to_value(e).unwrap_or(Value::Null)),
    })
}

/// The stream registry behind the routes.
pub struct Streams {
    engine: EngineService,
    cfg: StreamsConfig,
    map: Mutex<HashMap<String, Arc<Entry>>>,
    order: Mutex<VecDeque<String>>,
}

fn now_ms() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_millis() as u64).unwrap_or(0)
}

fn err_reply(e: &ApiError) -> Response {
    // Design §5.2: `/fv/v1/streams` answers busy with 429.
    let status = match e.kind {
        ErrorKind::Conflict => StatusCode::TOO_MANY_REQUESTS,
        _ => StatusCode::from_u16(e.http_status()).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR),
    };
    let kind = serde_json::to_value(e.kind).unwrap_or(Value::Null);
    let mut r = (status, Json(json!({"error": {"kind": kind, "message": e.message, "param": e.param}}))).into_response();
    if let Some(s) = e.retry_after_s {
        if let Ok(v) = HeaderValue::from_str(&s.to_string()) {
            r.headers_mut().insert("retry-after", v);
        }
    }
    r
}

fn not_found(id: &str) -> Response {
    err_reply(&ApiError::new(ErrorKind::NotFound, format!("no stream {id}")))
}

impl Streams {
    pub fn new(engine: EngineService, cfg: StreamsConfig) -> Self {
        Self {
            engine,
            cfg,
            map: Mutex::new(HashMap::new()),
            order: Mutex::new(VecDeque::new()),
        }
    }

    fn get(&self, id: &str) -> Option<Arc<Entry>> {
        self.map.lock().ok()?.get(id).cloned()
    }

    fn insert(&self, e: Arc<Entry>) {
        let id = e.id.clone();
        if let Ok(mut m) = self.map.lock() {
            m.insert(id.clone(), e);
        }
        if let Ok(mut o) = self.order.lock() {
            o.push_back(id);
        }
        self.prune();
    }

    /// Forgets the oldest ended streams beyond `keep_ended`.
    fn prune(&self) {
        let (Ok(mut m), Ok(mut o)) = (self.map.lock(), self.order.lock()) else {
            return;
        };
        let ended: Vec<String> = o
            .iter()
            .filter(|id| {
                m.get(*id)
                    .and_then(|e| e.status.lock().ok().map(|s| s.state == StreamState::Closed))
                    .unwrap_or(true)
            })
            .cloned()
            .collect();
        let excess = ended.len().saturating_sub(self.cfg.keep_ended);
        for id in ended.into_iter().take(excess) {
            m.remove(&id);
            o.retain(|x| *x != id);
        }
    }

    fn list(&self) -> Vec<Value> {
        let ids: Vec<String> = self.order.lock().map(|o| o.iter().rev().cloned().collect()).unwrap_or_default();
        ids.iter().filter_map(|id| self.get(id)).map(|e| e.view()).collect()
    }
}

/// The session spec for a body (pure; exposed for tests).
pub fn session_spec(caps: &ModelCaps, b: &StreamBody) -> Result<SessionSpec, ApiError> {
    let fps = b.fps.unwrap_or(match caps.stream {
        Some(StreamCaps::Causal { target_fps, .. }) => target_fps,
        _ => caps.fps.default,
    });
    let canvas = match (b.width, b.height) {
        (Some(w), Some(h)) => (w, h),
        (None, None) => {
            let short = caps.canvas.short_edges.first().copied().unwrap_or(480);
            canvas_for_aspect(&caps.canvas, 16.0 / 9.0, short)
        }
        _ => return Err(ApiError::invalid_param("width", "give both width and height, or neither")),
    };
    if canvas.0 % 2 != 0 || canvas.1 % 2 != 0 || canvas.0 < 16 || canvas.1 < 16 {
        return Err(ApiError::invalid_param("width", "the canvas needs even sides of at least 16"));
    }
    // WHIP publishes stereo Opus (design §5.3).
    let tracks = TrackSet::for_model(caps, canvas, fps, ("video", "audio"), 2, false);
    let tracks = if b.audio == Some(false) {
        TrackSet { audio: None, ..tracks }
    } else {
        tracks
    };
    let continuity = b.continuity.unwrap_or_default();
    Ok(SessionSpec {
        model: caps.id.clone(),
        tracks,
        canvas,
        fps,
        continuity,
        max_seconds: b.max_seconds,
        seed: b.seed,
    })
}

async fn create(State(ctx): State<ServeCtx>, st: Arc<Streams>, headers: HeaderMap, body: Value) -> Response {
    if let Err(e) = ctx.auth().authenticate(ProtocolId::Native, &headers) {
        return err_reply(&e);
    }
    if !PUBLISHER {
        return (
            StatusCode::NOT_IMPLEMENTED,
            Json(json!({"error": {"kind": "not_implemented", "message": "native WHIP streams need fv-serve built with the `webrtc` and `http-client` features", "param": null}})),
        )
            .into_response();
    }
    let b: StreamBody = match serde_json::from_value(body) {
        Ok(b) => b,
        Err(e) => return err_reply(&ApiError::invalid_param("body", e.to_string())),
    };
    match start(&st, b).await {
        Ok(entry) => (StatusCode::CREATED, Json(entry.view())).into_response(),
        Err(e) => err_reply(&e),
    }
}

#[cfg(not(all(feature = "webrtc", feature = "http-client")))]
async fn start(_st: &Arc<Streams>, _b: StreamBody) -> Result<Arc<Entry>, ApiError> {
    Err(ApiError::internal("no WHIP publisher in this build"))
}

/// Whether this build can publish (features `webrtc` + `http-client`).
pub const PUBLISHER: bool = cfg!(all(feature = "webrtc", feature = "http-client"));

#[cfg(all(feature = "webrtc", feature = "http-client"))]
async fn start(st: &Arc<Streams>, b: StreamBody) -> Result<Arc<Entry>, ApiError> {
    publish::start(st, b).await
}

async fn status(State(ctx): State<ServeCtx>, st: Arc<Streams>, headers: HeaderMap, id: String) -> Response {
    if let Err(e) = ctx.auth().authenticate(ProtocolId::Native, &headers) {
        return err_reply(&e);
    }
    match st.get(&id) {
        Some(e) => Json(e.view()).into_response(),
        None => not_found(&id),
    }
}

async fn stop(State(ctx): State<ServeCtx>, st: Arc<Streams>, headers: HeaderMap, id: String) -> Response {
    if let Err(e) = ctx.auth().authenticate(ProtocolId::Native, &headers) {
        return err_reply(&e);
    }
    let Some(e) = st.get(&id) else {
        return not_found(&id);
    };
    e.stop.notify_one();
    // Wait (briefly) for the WHIP DELETE and the session release.
    for _ in 0..200 {
        if e.status.lock().map(|s| s.state == StreamState::Closed).unwrap_or(true) {
            break;
        }
        tokio::time::sleep(Duration::from_millis(25)).await;
    }
    Json(e.view()).into_response()
}

async fn command(State(ctx): State<ServeCtx>, st: Arc<Streams>, headers: HeaderMap, id: String, body: Value) -> Response {
    if let Err(e) = ctx.auth().authenticate(ProtocolId::Native, &headers) {
        return err_reply(&e);
    }
    let Some(e) = st.get(&id) else {
        return not_found(&id);
    };
    let ctl = match e.control.lock() {
        Ok(g) => match g.as_ref() {
            Some(Control::Clip(p)) => Some(Control::Clip(p.clone())),
            Some(Control::Causal(c)) => Some(Control::Causal(c.clone())),
            None => None,
        },
        Err(_) => None,
    };
    let r = match ctl {
        Some(Control::Clip(p)) => match serde_json::from_value::<ClipCommand>(body) {
            Ok(c) => p
                .command(c)
                .await
                .map(|reply| json!({"reply": reply.map(|r| json!({"type": r.type_name(), "data": r.data()}))})),
            Err(err) => Err(ApiError::invalid_param("type", err.to_string())),
        },
        Some(Control::Causal(c)) => match serde_json::from_value::<CausalCommand>(body) {
            Ok(cmd) => Ok(json!({"reply": c.apply(cmd)})),
            Err(err) => Err(ApiError::invalid_param("type", err.to_string())),
        },
        None => Err(ApiError::conflict("the stream has ended")),
    };
    match r {
        Ok(v) => Json(v).into_response(),
        Err(err) => err_reply(&err),
    }
}

/// The `/fv/v1/streams` routes.
pub fn routes(gate: Arc<ServiceGate>, cfg: StreamsConfig) -> Router<ServeCtx> {
    let st = Arc::new(Streams::new(gate.engine().clone(), cfg));
    let (s1, s2, s3, s4, s5) = (st.clone(), st.clone(), st.clone(), st.clone(), st);
    Router::new()
        .route(
            "/fv/v1/streams",
            get(move |State(ctx): State<ServeCtx>, headers: HeaderMap| {
                let st = s1.clone();
                async move {
                    if let Err(e) = ctx.auth().authenticate(ProtocolId::Native, &headers) {
                        return err_reply(&e);
                    }
                    Json(json!({"object": "list", "data": st.list()})).into_response()
                }
            })
            .post(move |ctx: State<ServeCtx>, headers: HeaderMap, Json(body): Json<Value>| {
                let st = s2.clone();
                async move { create(ctx, st, headers, body).await }
            }),
        )
        .route(
            "/fv/v1/streams/{id}",
            get(move |ctx: State<ServeCtx>, headers: HeaderMap, Path(id): Path<String>| {
                let st = s3.clone();
                async move { status(ctx, st, headers, id).await }
            })
            .delete(move |ctx: State<ServeCtx>, headers: HeaderMap, Path(id): Path<String>| {
                let st = s4.clone();
                async move { stop(ctx, st, headers, id).await }
            }),
        )
        .route(
            "/fv/v1/streams/{id}/commands",
            post(move |ctx: State<ServeCtx>, headers: HeaderMap, Path(id): Path<String>, Json(body): Json<Value>| {
                let st = s5.clone();
                async move { command(ctx, st, headers, id, body).await }
            }),
        )
}

fn new_entry(model: &ModelId, mode: &'static str) -> Arc<Entry> {
    let id = format!("fvstream_{}", uuid_simple());
    Arc::new(Entry {
        id,
        model: model.to_string(),
        mode,
        created_ms: now_ms(),
        status: Mutex::new(StreamStatus {
            state: StreamState::Starting,
            end_reason: None,
            error: None,
            whip_resource: None,
            video_codec: None,
            encoder: None,
            frames_sent: 0,
            keyframes_sent: 0,
            audio_packets_sent: 0,
            forced_idrs: 0,
            keyframe_requests: 0,
            keyframe_requests_covered: 0,
            video_send_errors: 0,
            started_ms: now_ms(),
            first_frame_ms: None,
            streaming_ms: None,
        }),
        pace: Mutex::new(None),
        events: Mutex::new(VecDeque::new()),
        control: Mutex::new(None),
        stop: Notify::new(),
    })
}

fn uuid_simple() -> String {
    fastvideo_serve_kit::random_token()
}

#[cfg(all(feature = "webrtc", feature = "http-client"))]
mod publish {
    //! The WHIP publisher loop.

    use std::collections::VecDeque;
    use std::sync::Arc;
    use std::time::{Duration, Instant};

    use fastvideo_engine_service::stream::{
        spawn_causal_pacer, spawn_clip_pacer, CausalPacerConfig, ClipCommand, ClipPacerConfig,
        ClipPlayerConfig, PacedStream,
    };
    use fastvideo_media::pacer::VideoOut;
    use fastvideo_media::video::{create_encoder, EncoderBackend, VideoEncoder};
    use fastvideo_protocol::{ApiError, ErrorKind, StreamCaps};
    use fastvideo_webrtc::host::{AudioLayout, HostConfig, PeerEvent, RtcHost};
    use fastvideo_webrtc::ice::IceServer;
    use fastvideo_webrtc::whip::{WhipAuth, WhipConfig, WhipPublishOptions, WhipPublisher, WhipTarget};
    use fastvideo_webrtc::writer::{AudioPacket, VideoFrame};
    use serde_json::json;
    use url::Url;

    use super::{new_entry, now_ms, session_spec, Control, Entry, StreamBody, StreamState, Streams};

    fn internal(e: impl std::fmt::Display) -> ApiError {
        ApiError::internal(e.to_string())
    }

    fn pick_encoder(choice: &str) -> Result<EncoderBackend, ApiError> {
        match choice {
            "nvenc" => Ok(EncoderBackend::Nvenc),
            "openh264" => Ok(EncoderBackend::OpenH264),
            "x264-test" => Ok(EncoderBackend::CpuTestX264),
            _ => {
                // `auto` left unresolved (a router built without
                // `App::build`): the same per-process probe.
                let (_, sel) = crate::encoders::auto_selection();
                pick_encoder(&sel.streams)
            }
        }
    }

    pub(super) async fn start(st: &Arc<Streams>, b: StreamBody) -> Result<Arc<Entry>, ApiError> {
        let url = Url::parse(&b.whip_url).map_err(|e| ApiError::invalid_param("whip_url", e.to_string()))?;
        let target = match b.whip_target.clone().or_else(|| st.cfg.whip_target.clone()) {
            Some(t) => t.parse::<WhipTarget>().map_err(|e| ApiError::invalid_param("whip_target", e.to_string()))?,
            None => WhipTarget::guess(&url),
        };
        let token = b.whip_token.clone().or_else(|| st.cfg.whip_token.clone()).unwrap_or_default();
        let auth = match &b.whip_user {
            Some(u) => WhipAuth::from_user_token(u, &token),
            None if token.is_empty() => WhipAuth::None,
            None => WhipAuth::Bearer(token),
        };
        let caps = st
            .engine
            .caps()
            .get(&fastvideo_protocol::ModelId::new(&b.model))
            .cloned()
            .ok_or_else(|| ApiError::invalid_param("model", format!("model `{}` is not served here", b.model)))?;
        let spec = session_spec(&caps, &b)?;
        let causal = matches!(caps.stream, Some(StreamCaps::Causal { .. }));
        let entry = new_entry(&caps.id, if causal { "causal" } else { "clip" });
        let paced: PacedStream = if causal {
            let prompt = b
                .prompt
                .clone()
                .filter(|p| !p.trim().is_empty())
                .ok_or_else(|| ApiError::invalid_param("prompt", "a causal stream needs a prompt"))?;
            let session = st.engine.open_causal_session(spec.clone()).await?;
            let control = session.control();
            control.set_prompt(prompt.trim());
            *entry.control.lock().map_err(internal)? = Some(Control::Causal(control));
            spawn_causal_pacer(session, CausalPacerConfig::for_spec(&spec))?
        } else {
            let session = st.engine.open_clip_session(spec.clone()).await?;
            let (player, out) = session.into_player(ClipPlayerConfig {
                autoplay: b.autoplay.unwrap_or(true),
                ..ClipPlayerConfig::default()
            })?;
            let mut events = out.events;
            let e2 = entry.clone();
            tokio::spawn(async move {
                while let Some(ev) = events.recv().await {
                    e2.push_event(json!({"type": ev.type_name(), "data": ev.data()}));
                }
            });
            let mut clips: Vec<(String, Option<f64>, Option<u64>)> =
                b.clips.iter().map(|c| (c.prompt.clone(), c.seconds, c.seed)).collect();
            if let Some(p) = &b.prompt {
                clips.insert(0, (p.clone(), None, None));
            }
            for (prompt, seconds, seed) in clips {
                player
                    .command(ClipCommand::Enqueue {
                        prompt,
                        metadata: String::new(),
                        seed,
                        seconds,
                        position: None,
                    })
                    .await?;
            }
            *entry.control.lock().map_err(internal)? = Some(Control::Clip(player));
            spawn_clip_pacer(out.media, ClipPacerConfig::for_spec(&spec))?
        };
        *entry.pace.lock().map_err(internal)? = Some(paced.stats.clone());
        st.insert(entry.clone());
        let cfg = st.cfg.clone();
        let e2 = entry.clone();
        tokio::spawn(async move {
            let r = run(&e2, paced, spec.tracks.has_audio(), (spec.canvas, spec.fps), url, auth, target, &cfg).await;
            let reason = match &r {
                Ok(reason) => Some(reason.clone()),
                Err(_) => Some("error".to_owned()),
            };
            if let Err(e) = &r {
                tracing::warn!(stream = %e2.id, error = %e, "stream ended with an error");
            }
            let ctl = e2.control.lock().ok().and_then(|mut c| c.take());
            match ctl {
                Some(Control::Clip(p)) => p.close().await,
                Some(Control::Causal(c)) => c.close(),
                None => {}
            }
            e2.set(|s| {
                s.state = StreamState::Closed;
                s.end_reason = reason;
                if let Err(e) = &r {
                    s.error = Some(e.message.clone());
                }
            });
        });
        Ok(entry)
    }

    #[allow(clippy::too_many_arguments)]
    async fn run(
        entry: &Arc<Entry>,
        mut paced: PacedStream,
        audio: bool,
        (canvas, fps): ((u32, u32), u32),
        url: Url,
        auth: WhipAuth,
        target: WhipTarget,
        cfg: &super::StreamsConfig,
    ) -> Result<String, ApiError> {
        let t0 = Instant::now();
        // 1. The first frame (never offer an empty track).
        let mut first = paced.first_frame.clone();
        tokio::select! {
            r = tokio::time::timeout(cfg.first_frame_timeout, first.wait_for(|v| *v)) => match r {
                Ok(Ok(_)) => {}
                Ok(Err(_)) => return Err(ApiError::engine_failed("the session ended before its first frame")),
                Err(_) => return Err(ApiError::new(ErrorKind::Timeout, "no first frame in time")),
            },
            _ = entry.stop.notified() => return Ok("stopped".into()),
        }
        entry.set(|s| {
            s.state = StreamState::Publishing;
            s.first_frame_ms = Some(t0.elapsed().as_millis() as u64);
        });
        // 2. WHIP offer / answer.
        let stun: Vec<IceServer> = cfg.stun.iter().map(IceServer::stun).collect();
        let host = RtcHost::bind(HostConfig {
            udp_bind: Some(cfg.udp_bind),
            ice_servers: stun.clone(),
            ..HostConfig::default()
        })
        .await
        .map_err(internal)?;
        let mut publisher = WhipPublisher::publish(
            &host,
            WhipConfig::new(url, auth).with_target(target),
            WhipPublishOptions {
                audio: audio.then_some(AudioLayout::Stereo),
                stun,
                stun_timeout: Duration::from_secs(2),
            },
        )
        .await
        .map_err(|e| ApiError::new(ErrorKind::Internal, format!("whip: {e}")))?;
        let resource = publisher.resource_url().map(|u| u.to_string());
        let codec = publisher.negotiated_video_codec().map(str::to_owned);
        entry.set(|s| {
            s.whip_resource = resource;
            s.video_codec = codec;
        });
        let r = stream_loop(entry, &mut paced, &mut publisher, audio, canvas, fps, target, cfg).await;
        let _ = publisher.teardown().await;
        host.shutdown().await;
        paced.task.abort();
        r
    }

    #[allow(clippy::too_many_arguments)]
    async fn stream_loop(
        entry: &Arc<Entry>,
        paced: &mut PacedStream,
        publisher: &mut WhipPublisher,
        audio: bool,
        canvas: (u32, u32),
        fps: u32,
        target: WhipTarget,
        cfg: &super::StreamsConfig,
    ) -> Result<String, ApiError> {
        let handle = publisher.peer().handle().clone();
        // 3. ICE + DTLS.
        let deadline = tokio::time::Instant::now() + cfg.connect_timeout;
        loop {
            tokio::select! {
                ev = publisher.peer().next_event() => match ev {
                    Some(PeerEvent::Connected) => break,
                    Some(PeerEvent::Closed(r)) => return Err(ApiError::new(ErrorKind::Internal, format!("whip peer closed: {r:?}"))),
                    Some(_) => {}
                    None => return Err(ApiError::new(ErrorKind::Internal, "whip peer ended")),
                },
                _ = tokio::time::sleep_until(deadline) => return Err(ApiError::new(ErrorKind::Timeout, "whip ICE/DTLS did not connect")),
                _ = entry.stop.notified() => return Ok("stopped".into()),
            }
        }
        // 4. One encoder per session.
        let backend = {
            let c = cfg.encoder.clone();
            tokio::task::spawn_blocking(move || pick_encoder(&c)).await.map_err(internal)??
        };
        let hcfg = crate::whip::whip_h264(target, canvas.0, canvas.1, fps);
        let gop_frames = hcfg.gop_frames();
        let mut enc: Option<Box<dyn VideoEncoder>> = Some(
            tokio::task::spawn_blocking(move || create_encoder(backend, hcfg))
                .await
                .map_err(internal)?
                .map_err(|e| ApiError::internal(format!("encoder: {e}")))?,
        );
        entry.set(|s| {
            s.encoder = Some(format!("{backend:?}"));
            s.state = StreamState::Streaming;
            s.streaming_ms = Some(now_ms().saturating_sub(s.started_ms));
        });
        #[cfg(feature = "encoders")]
        let mut opus = if audio {
            Some(
                fastvideo_media::opus::OpusEncoder::new(fastvideo_media::opus::OpusConfig::whip(), 0)
                    .map_err(|e| ApiError::internal(format!("opus: {e}")))?,
            )
        } else {
            None
        };
        #[cfg(not(feature = "encoders"))]
        let _ = audio;
        // A new encoder starts on an IDR, so nothing is requested up front.
        // PLI/FIR (MediaMTX sends one every 2 s), a gap in the ticks or a
        // send error ask for a keyframe; the policy answers with the
        // periodic IDR when it is due within 1 s and forces one otherwise.
        let t0 = Instant::now();
        let mut idr = fastvideo_media::video::KeyframePolicy::new(1.0, gop_frames);
        let mut rtps: VecDeque<u32> = VecDeque::new();
        let mut first_sent = false;
        let reason = loop {
            tokio::select! {
                _ = entry.stop.notified() => break "stopped".to_owned(),
                ev = publisher.peer().next_event() => match ev {
                    Some(PeerEvent::KeyframeRequest { .. }) => {
                        idr.request(t0.elapsed().as_secs_f64());
                        entry.set(|s| s.keyframe_requests += 1);
                    }
                    Some(PeerEvent::Closed(r)) => return Err(ApiError::new(ErrorKind::Internal, format!("whip peer closed: {r:?}"))),
                    Some(_) => {}
                    None => return Err(ApiError::new(ErrorKind::Internal, "whip peer ended")),
                },
                t = paced.ticks.recv() => {
                    let Some(t) = t else {
                        let end = paced.stats.borrow().ended.clone();
                        break match end {
                            Some(e) => serde_json::to_value(e).ok().and_then(|v| v.as_str().map(str::to_owned).or_else(|| Some(v.to_string()))).unwrap_or_else(|| "ended".into()),
                            None => "ended".into(),
                        };
                    };
                    if paced.ticks.take_dropped() {
                        idr.request(t0.elapsed().as_secs_f64());
                    }
                    #[cfg(feature = "encoders")]
                    if let (Some(op), Some(a)) = (opus.as_mut(), t.audio.as_ref()) {
                        match op.push(a) {
                            Ok(pkts) => {
                                for p in pkts {
                                    if handle.send_audio(AudioPacket::new(p.data, u64::from(p.rtp_ts))).is_ok() {
                                        entry.set(|s| s.audio_packets_sent += 1);
                                    }
                                }
                            }
                            Err(e) => tracing::warn!(error = %e, "opus encode failed"),
                        }
                    }
                    let frame = match &t.video {
                        VideoOut::Fresh(f) | VideoOut::Repeat(f) => f.clone(),
                        VideoOut::Nothing => continue,
                    };
                    let force = idr.next_frame(t0.elapsed().as_secs_f64());
                    rtps.push_back(t.video_rtp);
                    let mut e = enc.take().expect("encoder present");
                    let (e, out) = tokio::task::spawn_blocking(move || {
                        if force {
                            e.force_idr();
                        }
                        let r = e.encode(&frame);
                        (e, r)
                    })
                    .await
                    .map_err(internal)?;
                    enc = Some(e);
                    if force {
                        entry.set(|s| s.forced_idrs += 1);
                    }
                    let covered = idr.covered();
                    entry.set(|s| s.keyframe_requests_covered = covered);
                    let aus = out.map_err(|e| ApiError::internal(format!("encode: {e}")))?;
                    for au in aus {
                        let rtp = rtps.pop_front().unwrap_or(t.video_rtp);
                        let key = au.keyframe;
                        if key {
                            idr.keyframe_out(t0.elapsed().as_secs_f64());
                        }
                        match handle.send_video(VideoFrame::new(au.data, u64::from(rtp))) {
                            Ok(()) => {
                                entry.set(|s| {
                                    s.frames_sent += 1;
                                    if key {
                                        s.keyframes_sent += 1;
                                    }
                                });
                                if !first_sent {
                                    first_sent = true;
                                    if let Ok(g) = entry.control.lock() {
                                        if let Some(Control::Causal(c)) = g.as_ref() {
                                            c.mark_first_frame_sent();
                                        }
                                    }
                                }
                            }
                            Err(_) => {
                                entry.set(|s| s.video_send_errors += 1);
                                idr.request(t0.elapsed().as_secs_f64());
                            }
                        }
                    }
                }
            }
        };
        if let Some(mut e) = enc.take() {
            let _ = tokio::task::spawn_blocking(move || e.finish()).await;
        }
        Ok(reason)
    }
}
