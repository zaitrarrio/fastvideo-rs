//! `/v1/segments`: a pull-based segment API over causal sessions, for a
//! control plane that drives an infinite live stream (docs/serve/segments.md).
//!
//! | Route | Behaviour |
//! |---|---|
//! | `POST /v1/segments` | Render the next segment of a stream: `{stream_id, segment_index, prompt, duration_sec, …}` → 202 segment object. The first segment of a `stream_id` opens a causal session (exclusive executor lease); later ones continue it, so the KV cache, the sink and the decoder state carry across segments and the seams are invisible |
//! | `GET /v1/segments/{id}` | Segment object: `queued` \| `running` \| `succeeded` (with `video_url`) \| `failed` |
//! | `GET /v1/segments/streams/{stream_id}` | The stream's session: model, canvas, blocks generated, queue depth, last use |
//! | `DELETE /v1/segments/streams/{stream_id}` | Close the session now (it also closes after `idle_ttl`) |
//!
//! The same routes are mounted under `/fv/v1/segments`. Auth: the native
//! API's (`Authorization: Bearer <key>`).
//!
//! Semantics the control plane relies on:
//! - **Segments are scheduling units.** A segment is `round(duration_sec ×
//!   fps)` frames of one continuous rollout. Frames beyond a segment's share
//!   of the last block carry over into the next segment, so nothing is
//!   dropped and clip N+1 starts exactly where N ended. `trim_head_sec` is
//!   always 0.
//! - **Prompts land at block boundaries.** Each segment's prompt is set
//!   before its first block; the model's prompt-switch policy (keep or KV
//!   re-cache) applies. `reset: true` clears the KV cache first (a scene
//!   cut). `recache` is accepted for the contract and currently means the
//!   same as a prompt change (the policy is the model's).
//! - **One stream per session, one session per executor.** A second stream
//!   on a busy executor gets 429 + `Retry-After`; the control plane routes
//!   streams to backends.
//! - **Session length.** Causal sessions are capped (`[streams]`
//!   `causal_hard_max_s`, 300 s of video by default). When a session ends
//!   under a segment, the next segment opens a new one from text: the
//!   segment object says `session.restarted: true`, and the control plane
//!   should treat that seam as a cut.
//! - **Not supported on the causal path** (refused with `invalid_param`):
//!   `context.image_url`, `context.video_url` (the clip models' chaining
//!   inputs). `reference_image_urls` are accepted and ignored (`warnings`).

use std::collections::{HashMap, VecDeque};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use axum::extract::{Path, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use fastvideo_engine_service::{CausalControl, CausalSession, EngineService};
use fastvideo_media::mp4::{write_mp4, Mp4Spec};
use fastvideo_media::video::FfmpegH264;
use fastvideo_protocol::{
    canvas_for_aspect, ApiError, CausalLimits, Continuity, ErrorKind, ModelCaps, ModelId,
    ProtocolId, RgbFrame, SessionSpec, StreamCaps, TrackSet,
};
use fastvideo_serve_kit::{ArtifactMeta, ServeCtx};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tokio::sync::{mpsc, Notify};

use crate::gate::ServiceGate;

/// Longest accepted prompt (the causal control's own limit).
const MAX_PROMPT_CHARS: usize = 800;

/// `[segments]` resolved.
#[derive(Clone, Debug)]
pub struct SegmentsConfig {
    /// The causal model a request without `model` uses; `None`: the first
    /// causal model the engine serves.
    pub model: Option<String>,
    /// Close a stream's session after this long without a segment request.
    pub idle_ttl: Duration,
    /// Session length in video seconds (resolved through `causal`).
    pub max_seconds: Option<u32>,
    pub causal: CausalLimits,
    /// Signed `video_url` lifetime.
    pub url_ttl: Duration,
    /// H.264 encoder for the segment MP4s.
    pub encoder: FfmpegH264,
    /// Finished segments kept for `GET` (per process).
    pub keep: usize,
    /// Segment jobs a stream may have waiting.
    pub queue: usize,
}

impl Default for SegmentsConfig {
    fn default() -> Self {
        Self {
            model: None,
            idle_ttl: Duration::from_secs(120),
            max_seconds: None,
            causal: CausalLimits::default(),
            url_ttl: Duration::from_secs(3600),
            encoder: FfmpegH264::Nvenc,
            keep: 256,
            queue: 8,
        }
    }
}

impl SegmentsConfig {
    pub fn from_config(c: &crate::config::Config) -> Self {
        let s = &c.segments;
        Self {
            model: s.model.clone().filter(|m| !m.trim().is_empty()),
            idle_ttl: Duration::from_secs(s.idle_ttl_s.max(1)),
            max_seconds: s.max_seconds,
            causal: c.streams.causal_limits(),
            url_ttl: Duration::from_secs(s.url_ttl_s.max(60)),
            encoder: match s.encoder.as_str() {
                "libx264" => FfmpegH264::Libx264CpuTest,
                _ => FfmpegH264::Nvenc,
            },
            keep: s.keep.max(16),
            queue: s.queue.max(1),
        }
    }
}

/// `POST /v1/segments` body (docs/render-backend contract of the control plane).
#[derive(Clone, Debug, Default, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SegmentRequest {
    pub stream_id: String,
    pub segment_index: u64,
    #[serde(default)]
    pub model: Option<String>,
    pub prompt: String,
    pub duration_sec: f64,
    /// `480p` (the causal canvas tier); other tiers only if the model serves them.
    #[serde(default)]
    pub resolution: Option<String>,
    /// `16:9` (default), `9:16`, `1:1`, …
    #[serde(default)]
    pub aspect_ratio: Option<String>,
    #[serde(default)]
    pub seed: Option<u64>,
    /// The direction changed: apply the model's prompt-switch policy.
    #[serde(default)]
    pub recache: bool,
    /// A scene cut: clear the KV cache before this segment.
    #[serde(default)]
    pub reset: bool,
    /// Accepted and ignored on Wan (no reference conditioning).
    #[serde(default)]
    pub reference_image_urls: Vec<String>,
    /// Keyframe planning (clip models): accepted and ignored on the causal path.
    #[serde(default)]
    pub end_image_url: Option<String>,
    #[serde(default)]
    pub context: Option<SegmentContext>,
    /// POSTed the finished segment object (feature `http-client`).
    #[serde(default)]
    pub callback_url: Option<String>,
    #[serde(default)]
    pub extra: Option<Value>,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SegmentContext {
    #[serde(default)]
    pub image_url: Option<String>,
    #[serde(default)]
    pub video_url: Option<String>,
    #[serde(default)]
    pub overlap_sec: Option<f64>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SegmentState {
    Queued,
    Running,
    Succeeded,
    Failed,
}

#[derive(Clone, Debug, Serialize)]
struct SegmentStatus {
    status: SegmentState,
    video_url: Option<String>,
    frames: u32,
    fps: u32,
    width: u32,
    height: u32,
    /// Blocks of the rollout this segment spans (first, last).
    blocks: Option<(u64, u64)>,
    /// The session had to be reopened for this segment (length limit or
    /// failure): the seam before it is a cut.
    restarted: bool,
    /// Blocks at the head of this segment generated before its prompt
    /// landed (the engine's lookahead): the direction shows after them.
    stale_blocks: u32,
    /// Blocks discarded after a `reset` (generated before it).
    discarded_blocks: u32,
    generation_ms: Option<f64>,
    encode_ms: Option<f64>,
    error: Option<String>,
    finished_ms: Option<u64>,
}

/// One segment request and its outcome.
pub struct SegmentJob {
    id: String,
    req: SegmentRequest,
    created_ms: u64,
    warnings: Vec<String>,
    status: Mutex<SegmentStatus>,
    done: Notify,
}

impl SegmentJob {
    fn set(&self, f: impl FnOnce(&mut SegmentStatus)) {
        if let Ok(mut s) = self.status.lock() {
            f(&mut s);
        }
    }

    pub fn id(&self) -> &str {
        &self.id
    }

    pub fn view(&self, ttl_note: bool) -> Value {
        let s = self
            .status
            .lock()
            .map(|s| s.clone())
            .unwrap_or_else(|e| e.into_inner().clone());
        let mut v = json!({
            "id": self.id,
            "object": "fv.segment",
            "stream_id": self.req.stream_id,
            "segment_index": self.req.segment_index,
            "status": s.status,
            "video_url": s.video_url,
            "trim_head_sec": 0.0,
            "frames": s.frames,
            "fps": s.fps,
            "width": s.width,
            "height": s.height,
            "duration_sec": if s.fps > 0 { f64::from(s.frames) / f64::from(s.fps) } else { 0.0 },
            "blocks": s.blocks,
            "session": {"restarted": s.restarted, "stale_blocks": s.stale_blocks, "discarded_blocks": s.discarded_blocks},
            "metrics": {"generation_ms": s.generation_ms, "encode_ms": s.encode_ms},
            "error": s.error,
            "created_at": self.created_ms / 1000,
            "finished_at": s.finished_ms.map(|m| m / 1000),
        });
        if !self.warnings.is_empty() {
            v["warnings"] = json!(self.warnings);
        }
        if ttl_note {
            v["video_url_note"] = json!("signed; fetch it soon");
        }
        v
    }
}

/// One stream's session worker.
struct Slot {
    stream_id: String,
    model: ModelId,
    tx: mpsc::Sender<Arc<SegmentJob>>,
    control: Mutex<Option<CausalControl>>,
    last_used: Mutex<Instant>,
    closed: AtomicBool,
    /// Sessions opened for this stream (1 = never restarted).
    sessions: AtomicU64,
    blocks: AtomicU64,
    queued: AtomicU64,
    canvas: (u32, u32),
    fps: u32,
    created_ms: u64,
}

impl Slot {
    fn view(&self) -> Value {
        let state = self
            .control
            .lock()
            .ok()
            .and_then(|c| c.as_ref().map(|c| serde_json::to_value(c.state()).ok()))
            .flatten();
        json!({
            "object": "fv.segment_stream",
            "stream_id": self.stream_id,
            "model": self.model.to_string(),
            "width": self.canvas.0,
            "height": self.canvas.1,
            "fps": self.fps,
            "sessions": self.sessions.load(Ordering::Relaxed),
            "blocks": self.blocks.load(Ordering::Relaxed),
            "queued": self.queued.load(Ordering::Relaxed),
            "closed": self.closed.load(Ordering::Relaxed),
            "idle_s": self.last_used.lock().map(|t| t.elapsed().as_secs_f64()).unwrap_or(0.0),
            "created_at": self.created_ms / 1000,
            "session": state,
        })
    }
}

/// The registry behind the routes.
pub struct Segments {
    engine: EngineService,
    cfg: SegmentsConfig,
    work_dir: PathBuf,
    /// The serving context (artifact store, signer), attached by the first request.
    ctx: Mutex<Option<ServeCtx>>,
    slots: Mutex<HashMap<String, Arc<Slot>>>,
    jobs: Mutex<HashMap<String, Arc<SegmentJob>>>,
    /// `(stream_id, segment_index)` → job id, for idempotent resubmits.
    by_index: Mutex<HashMap<(String, u64), String>>,
    order: Mutex<VecDeque<String>>,
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

fn err_reply(e: &ApiError) -> Response {
    // Like `/fv/v1/streams`: a busy executor is 429, not 409.
    let status = match e.kind {
        ErrorKind::Conflict => StatusCode::TOO_MANY_REQUESTS,
        _ => StatusCode::from_u16(e.http_status()).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR),
    };
    let kind = serde_json::to_value(e.kind).unwrap_or(Value::Null);
    let mut r = (
        status,
        Json(json!({"error": {"kind": kind, "message": e.message, "param": e.param}})),
    )
        .into_response();
    if let Some(s) = e
        .retry_after_s
        .or((status == StatusCode::TOO_MANY_REQUESTS).then_some(2))
    {
        if let Ok(v) = s.to_string().parse() {
            r.headers_mut().insert("retry-after", v);
        }
    }
    r
}

fn parse_aspect(s: &str) -> Option<f64> {
    let (w, h) = s.split_once(':')?;
    let (w, h): (f64, f64) = (w.trim().parse().ok()?, h.trim().parse().ok()?);
    (w > 0.0 && h > 0.0).then_some(w / h)
}

fn parse_short_edge(s: &str) -> Option<u32> {
    s.trim().trim_end_matches(['p', 'P']).parse().ok()
}

/// The session spec for a stream's first segment (pure; tests).
pub fn session_spec(
    caps: &ModelCaps,
    r: &SegmentRequest,
    cfg: &SegmentsConfig,
) -> Result<SessionSpec, ApiError> {
    let Some(StreamCaps::Causal { target_fps, .. }) = caps.stream else {
        return Err(ApiError::invalid_param(
            "model",
            format!("`{}` is not a causal (streaming) model", caps.id),
        ));
    };
    let short = match &r.resolution {
        None => caps.canvas.short_edges.first().copied().unwrap_or(480),
        Some(s) => {
            let n = parse_short_edge(s).ok_or_else(|| {
                ApiError::invalid_param(
                    "resolution",
                    format!("`{s}` is not a resolution like `480p`"),
                )
            })?;
            if !caps.canvas.short_edges.contains(&n) {
                return Err(ApiError::invalid_param(
                    "resolution",
                    format!(
                        "`{}` serves {}",
                        caps.id,
                        caps.canvas
                            .short_edges
                            .iter()
                            .map(|e| format!("{e}p"))
                            .collect::<Vec<_>>()
                            .join(", ")
                    ),
                ));
            }
            n
        }
    };
    let ratio = match &r.aspect_ratio {
        None => 16.0 / 9.0,
        Some(a) => parse_aspect(a).ok_or_else(|| {
            ApiError::invalid_param(
                "aspect_ratio",
                format!("`{a}` is not an aspect ratio like `16:9`"),
            )
        })?,
    };
    let (lo, hi) = caps.canvas.aspect;
    if ratio < f64::from(lo) || ratio > f64::from(hi) {
        return Err(ApiError::invalid_param(
            "aspect_ratio",
            format!("`{}` serves aspect ratios {lo}..{hi}", caps.id),
        ));
    }
    let canvas = canvas_for_aspect(&caps.canvas, ratio, short);
    let tracks = TrackSet::for_model(caps, canvas, target_fps, ("video", "audio"), 2, false);
    Ok(SessionSpec {
        model: caps.id.clone(),
        tracks: TrackSet {
            audio: None,
            ..tracks
        },
        canvas,
        fps: target_fps,
        continuity: Continuity::default(),
        max_seconds: Some(cfg.causal.resolve(cfg.max_seconds)?),
        seed: r.seed,
    })
}

fn validate(r: &SegmentRequest) -> Result<Vec<String>, ApiError> {
    if r.stream_id.trim().is_empty()
        || r.stream_id.len() > 128
        || !r
            .stream_id
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
    {
        return Err(ApiError::invalid_param(
            "stream_id",
            "stream_id is 1–128 characters of [A-Za-z0-9_-]",
        ));
    }
    if r.prompt.trim().is_empty() {
        return Err(ApiError::invalid_param("prompt", "the prompt is empty"));
    }
    if r.prompt.chars().count() > MAX_PROMPT_CHARS {
        return Err(ApiError::invalid_param(
            "prompt",
            "the prompt is longer than 800 characters",
        ));
    }
    if !(r.duration_sec.is_finite() && (0.5..=30.0).contains(&r.duration_sec)) {
        return Err(ApiError::invalid_param(
            "duration_sec",
            "duration_sec is 0.5 to 30 seconds",
        ));
    }
    if let Some(c) = &r.context {
        if c.image_url.is_some() {
            return Err(ApiError::invalid_param("context.image_url", "the causal path keeps its own state: image conditioning is not supported (use continuity \"native\")"));
        }
        if c.video_url.is_some() {
            return Err(ApiError::invalid_param("context.video_url", "the causal path keeps its own state: video context is not supported (use continuity \"native\")"));
        }
    }
    let mut warnings = Vec::new();
    if !r.reference_image_urls.is_empty() {
        warnings.push(
            "reference_image_urls ignored: the model has no reference conditioning".to_owned(),
        );
    }
    if r.end_image_url.is_some() {
        warnings.push(
            "end_image_url ignored: the causal path renders toward no planned frame".to_owned(),
        );
    }
    Ok(warnings)
}

impl Segments {
    pub fn new(engine: EngineService, cfg: SegmentsConfig, work_dir: PathBuf) -> Self {
        Self {
            engine,
            cfg,
            work_dir,
            ctx: Mutex::new(None),
            slots: Mutex::new(HashMap::new()),
            jobs: Mutex::new(HashMap::new()),
            by_index: Mutex::new(HashMap::new()),
            order: Mutex::new(VecDeque::new()),
        }
    }

    fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
        m.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn job(&self, id: &str) -> Option<Arc<SegmentJob>> {
        Self::lock(&self.jobs).get(id).cloned()
    }

    fn slot(&self, stream_id: &str) -> Option<Arc<Slot>> {
        Self::lock(&self.slots)
            .get(stream_id)
            .cloned()
            .filter(|s| !s.closed.load(Ordering::Relaxed))
    }

    /// The model for a request: its own, the configured default, else the
    /// first causal model served.
    fn resolve_model(&self, r: &SegmentRequest) -> Result<ModelCaps, ApiError> {
        let caps = self.engine.caps();
        if let Some(m) = r
            .model
            .as_deref()
            .filter(|m| !m.trim().is_empty())
            .or(self.cfg.model.as_deref())
        {
            let id = ModelId::new(m);
            return caps
                .get(&id)
                .cloned()
                .or_else(|| caps.models().find(|c| c.answers_to(m)).cloned())
                .ok_or_else(|| {
                    ApiError::invalid_param("model", format!("model `{m}` is not served here"))
                });
        }
        caps.models()
            .find(|c| matches!(c.stream, Some(StreamCaps::Causal { .. })))
            .cloned()
            .ok_or_else(|| {
                ApiError::invalid_param("model", "no causal model is served here; give `model`")
            })
    }

    fn remember(&self, job: Arc<SegmentJob>) {
        let key = (job.req.stream_id.clone(), job.req.segment_index);
        Self::lock(&self.by_index).insert(key, job.id.clone());
        let id = job.id.clone();
        Self::lock(&self.jobs).insert(id.clone(), job);
        let mut order = Self::lock(&self.order);
        order.push_back(id);
        while order.len() > self.cfg.keep {
            if let Some(old) = order.pop_front() {
                if let Some(j) = Self::lock(&self.jobs).remove(&old) {
                    Self::lock(&self.by_index)
                        .remove(&(j.req.stream_id.clone(), j.req.segment_index));
                }
            }
        }
    }

    /// Submits a segment: idempotent per `(stream_id, segment_index)`.
    pub async fn submit(self: &Arc<Self>, r: SegmentRequest) -> Result<Arc<SegmentJob>, ApiError> {
        let warnings = validate(&r)?;
        if let Some(id) = Self::lock(&self.by_index)
            .get(&(r.stream_id.clone(), r.segment_index))
            .cloned()
        {
            if let Some(j) = self.job(&id) {
                return Ok(j);
            }
        }
        let caps = self.resolve_model(&r)?;
        let spec = session_spec(&caps, &r, &self.cfg)?;
        let slot = match self.slot(&r.stream_id) {
            Some(s) => {
                if s.model != caps.id {
                    return Err(ApiError::invalid_param(
                        "model",
                        format!("stream `{}` runs `{}`", r.stream_id, s.model),
                    ));
                }
                s
            }
            None => self.open_slot(&r.stream_id, caps, spec)?,
        };
        let fps = slot.fps;
        let job = Arc::new(SegmentJob {
            id: format!("fvseg_{}", fastvideo_serve_kit::random_token()),
            req: r,
            created_ms: now_ms(),
            warnings,
            status: Mutex::new(SegmentStatus {
                status: SegmentState::Queued,
                video_url: None,
                frames: 0,
                fps,
                width: slot.canvas.0,
                height: slot.canvas.1,
                blocks: None,
                restarted: false,
                stale_blocks: 0,
                discarded_blocks: 0,
                generation_ms: None,
                encode_ms: None,
                error: None,
                finished_ms: None,
            }),
            done: Notify::new(),
        });
        *Self::lock(&slot.last_used) = Instant::now();
        match slot.tx.try_send(job.clone()) {
            Ok(()) => {
                slot.queued.fetch_add(1, Ordering::Relaxed);
            }
            Err(mpsc::error::TrySendError::Full(_)) => {
                return Err(ApiError::conflict(format!(
                    "stream `{}` has {} segments waiting",
                    slot.stream_id, self.cfg.queue
                )));
            }
            Err(mpsc::error::TrySendError::Closed(_)) => {
                return Err(ApiError::conflict(format!(
                    "stream `{}` is closing; retry",
                    slot.stream_id
                )));
            }
        }
        self.remember(job.clone());
        Ok(job)
    }

    /// Opens a stream's slot and spawns its worker. The causal session itself
    /// opens inside the worker on the first segment (so an executor busy with
    /// another stream fails that segment, not the submit).
    fn open_slot(
        self: &Arc<Self>,
        stream_id: &str,
        caps: ModelCaps,
        spec: SessionSpec,
    ) -> Result<Arc<Slot>, ApiError> {
        let (tx, rx) = mpsc::channel(self.cfg.queue);
        let slot = Arc::new(Slot {
            stream_id: stream_id.to_owned(),
            model: caps.id.clone(),
            tx,
            control: Mutex::new(None),
            last_used: Mutex::new(Instant::now()),
            closed: AtomicBool::new(false),
            sessions: AtomicU64::new(0),
            blocks: AtomicU64::new(0),
            queued: AtomicU64::new(0),
            canvas: spec.canvas,
            fps: spec.fps,
            created_ms: now_ms(),
        });
        Self::lock(&self.slots).insert(stream_id.to_owned(), slot.clone());
        let me = self.clone();
        let s2 = slot.clone();
        tokio::spawn(async move {
            me.run_slot(s2, caps, spec, rx).await;
        });
        Ok(slot)
    }

    /// Closes a stream's session (idempotent).
    pub fn close_stream(&self, stream_id: &str) -> bool {
        let Some(slot) = Self::lock(&self.slots).remove(stream_id) else {
            return false;
        };
        slot.closed.store(true, Ordering::Relaxed);
        if let Some(c) = Self::lock(&slot.control).take() {
            c.close();
        }
        true
    }

    pub fn stream_view(&self, stream_id: &str) -> Option<Value> {
        Self::lock(&self.slots).get(stream_id).map(|s| s.view())
    }

    pub fn list_streams(&self) -> Vec<Value> {
        Self::lock(&self.slots).values().map(|s| s.view()).collect()
    }

    /// The worker: one causal session at a time for the stream, a FIFO of
    /// segments, a carry-over of frames between them.
    async fn run_slot(
        self: Arc<Self>,
        slot: Arc<Slot>,
        caps: ModelCaps,
        spec: SessionSpec,
        mut rx: mpsc::Receiver<Arc<SegmentJob>>,
    ) {
        let _ = &caps;
        let mut session: Option<CausalSession> = None;
        let mut carry: Vec<RgbFrame> = Vec::new();
        let mut last_prompt: Option<String> = None;
        let mut next_block: u64 = 0;
        loop {
            let job = match tokio::time::timeout(self.cfg.idle_ttl, rx.recv()).await {
                Ok(Some(j)) => j,
                Ok(None) => break,
                Err(_) => {
                    tracing::info!(stream = %slot.stream_id, idle_s = self.cfg.idle_ttl.as_secs(), "segment stream idle: closing its session");
                    break;
                }
            };
            slot.queued.fetch_sub(1, Ordering::Relaxed);
            if slot.closed.load(Ordering::Relaxed) {
                job.set(|s| {
                    s.status = SegmentState::Failed;
                    s.error = Some("the stream was closed".into());
                    s.finished_ms = Some(now_ms());
                });
                job.done.notify_waiters();
                continue;
            }
            job.set(|s| s.status = SegmentState::Running);
            let t0 = Instant::now();
            let need = ((job.req.duration_sec * f64::from(spec.fps)).round() as usize).max(1);
            let mut restarted = false;
            let mut first_block: Option<u64> = None;
            let mut failed: Option<ApiError> = None;
            // Blocks generated under the previous prompt that this segment
            // still carries (the engine's lookahead), and blocks thrown
            // away after a reset.
            let mut stale_blocks: u32 = 0;
            let mut discarded: u32 = 0;

            // Open (or reopen) the session.
            if session.is_none() {
                match self.engine.open_causal_session(spec.clone()).await {
                    Ok(s) => {
                        *Self::lock(&slot.control) = Some(s.control());
                        if slot.sessions.fetch_add(1, Ordering::Relaxed) > 0 {
                            restarted = true;
                        }
                        carry.clear();
                        last_prompt = None;
                        next_block = 0;
                        session = Some(s);
                    }
                    Err(e) => failed = Some(e),
                }
            }

            if failed.is_none() {
                let s = session.as_mut().expect("opened above");
                if job.req.reset {
                    s.reset();
                    carry.clear();
                }
                if let Some(seed) = job.req.seed {
                    s.set_seed(seed);
                }
                let prompt = job.req.prompt.trim();
                let mut version: Option<u64> = None;
                if job.req.reset || job.req.recache || last_prompt.as_deref() != Some(prompt) {
                    version = Some(s.set_prompt(prompt));
                    last_prompt = Some(prompt.to_owned());
                }
                // The engine runs up to `causal_depth` blocks ahead. After a
                // reset those blocks belong to the old scene: skip to the
                // first block generated after the reset.
                let mut await_reset = job.req.reset;
                let mut reopened = false;
                while carry.len() < need {
                    let s = session.as_mut().expect("open while collecting");
                    match s.next_block().await {
                        Some(Ok(b)) => {
                            if await_reset {
                                if !b.reset {
                                    discarded += 1;
                                    continue;
                                }
                                await_reset = false;
                            }
                            if version.is_some_and(|v| b.prompt_version < v) {
                                stale_blocks += 1;
                            }
                            first_block.get_or_insert(b.index);
                            next_block = b.index + 1;
                            slot.blocks.fetch_add(1, Ordering::Relaxed);
                            carry.extend(b.frames);
                        }
                        Some(Err(e)) => {
                            failed = Some(e);
                            break;
                        }
                        None if reopened => {
                            failed = Some(ApiError::engine_failed(
                                "the causal session ended twice under one segment",
                            ));
                            break;
                        }
                        None => {
                            // The session ended (its length limit, or the
                            // engine closed it): finish this segment from a
                            // fresh session. The seam is a cut.
                            tracing::info!(stream = %slot.stream_id, frames = carry.len(), need, "causal session ended under a segment: reopening");
                            *Self::lock(&slot.control) = None;
                            session = None;
                            match self.engine.open_causal_session(spec.clone()).await {
                                Ok(ns) => {
                                    *Self::lock(&slot.control) = Some(ns.control());
                                    slot.sessions.fetch_add(1, Ordering::Relaxed);
                                    restarted = true;
                                    reopened = true;
                                    ns.set_prompt(prompt);
                                    session = Some(ns);
                                }
                                Err(e) => {
                                    failed = Some(e);
                                    break;
                                }
                            }
                        }
                    }
                }
            }

            if let Some(e) = failed {
                tracing::warn!(stream = %slot.stream_id, segment = job.req.segment_index, error = %e.message, "segment failed");
                *Self::lock(&slot.control) = None;
                session = None;
                carry.clear();
                job.set(|s| {
                    s.status = SegmentState::Failed;
                    s.error = Some(e.message.clone());
                    s.restarted = restarted;
                    s.finished_ms = Some(now_ms());
                });
                self.callback(&job).await;
                job.done.notify_waiters();
                continue;
            }

            let frames: Vec<RgbFrame> = carry.drain(..need).collect();
            let generation_ms = t0.elapsed().as_secs_f64() * 1e3;
            let last_block = next_block.saturating_sub(1);
            let blocks = first_block.map(|f| (f, last_block));

            // Encode and store off the runtime.
            let t1 = Instant::now();
            let ctx = self.engine_ctx();
            let out = self.work_dir.join(format!("{}.mp4", job.id));
            let mp4 = Mp4Spec {
                width: spec.canvas.0,
                height: spec.canvas.1,
                fps: spec.fps,
                quality: 19,
                encoder: self.cfg.encoder,
                audio: None,
                faststart: true,
                fragmented: false,
            };
            let out2 = out.clone();
            let n = frames.len() as u32;
            let encoded = tokio::task::spawn_blocking(move || write_mp4(&out2, mp4, &frames, None))
                .await
                .map_err(|e| ApiError::internal(format!("encode task: {e}")))
                .and_then(|r| {
                    r.map_err(|e| ApiError::internal(format!("encoding the segment: {e}")))
                });
            let stored = match (encoded, ctx) {
                (Ok(path), Some(ctx)) => {
                    let meta = ArtifactMeta {
                        file_name: format!("{}-{}.mp4", job.req.stream_id, job.req.segment_index),
                        mime: "video/mp4".into(),
                        width: spec.canvas.0,
                        height: spec.canvas.1,
                        frames: n,
                        fps: spec.fps,
                        audio: None,
                    };
                    match ctx.artifacts().put(&path, meta).await {
                        Ok(a) => Ok(ctx
                            .artifacts()
                            .signer()
                            .url_for(&a, self.cfg.url_ttl)
                            .to_string()),
                        Err(e) => Err(e),
                    }
                }
                (Ok(path), None) => Ok(format!("file://{}", path.display())),
                (Err(e), _) => Err(e),
            };
            let encode_ms = t1.elapsed().as_secs_f64() * 1e3;
            match stored {
                Ok(url) => job.set(|s| {
                    s.status = SegmentState::Succeeded;
                    s.video_url = Some(url);
                    s.frames = n;
                    s.blocks = blocks;
                    s.restarted = restarted;
                    s.stale_blocks = stale_blocks;
                    s.discarded_blocks = discarded;
                    s.generation_ms = Some(generation_ms);
                    s.encode_ms = Some(encode_ms);
                    s.finished_ms = Some(now_ms());
                }),
                Err(e) => job.set(|s| {
                    s.status = SegmentState::Failed;
                    s.error = Some(e.message.clone());
                    s.restarted = restarted;
                    s.finished_ms = Some(now_ms());
                }),
            }
            self.callback(&job).await;
            job.done.notify_waiters();
        }
        // Leaving the loop: close the session and forget the slot.
        slot.closed.store(true, Ordering::Relaxed);
        if let Some(c) = Self::lock(&slot.control).take() {
            c.close();
        }
        drop(session);
        let mut slots = Self::lock(&self.slots);
        if slots
            .get(&slot.stream_id)
            .is_some_and(|s| Arc::ptr_eq(s, &slot))
        {
            slots.remove(&slot.stream_id);
        }
    }

    fn engine_ctx(&self) -> Option<ServeCtx> {
        Self::lock(&self.ctx).clone()
    }

    #[cfg(feature = "http-client")]
    async fn callback(&self, job: &SegmentJob) {
        let Some(url) = job.req.callback_url.as_deref() else {
            return;
        };
        let body = job.view(false);
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(10))
            .build();
        let Ok(client) = client else { return };
        for attempt in 0..3u32 {
            match client.post(url).json(&body).send().await {
                Ok(r) if r.status().is_success() => return,
                Ok(r) => {
                    tracing::warn!(segment = %job.id, status = %r.status(), attempt, "segment callback refused")
                }
                Err(e) => {
                    tracing::warn!(segment = %job.id, error = %e, attempt, "segment callback failed")
                }
            }
            tokio::time::sleep(Duration::from_millis(500 * u64::from(attempt + 1))).await;
        }
    }

    #[cfg(not(feature = "http-client"))]
    async fn callback(&self, job: &SegmentJob) {
        if job.req.callback_url.is_some() {
            tracing::debug!(segment = %job.id, "segment callback skipped: built without `http-client`");
        }
    }
}

// The artifact store comes from the `ServeCtx`, which the routes receive
// as axum state after the registry is built: it is attached on first use.
impl Segments {
    fn attach(&self, ctx: &ServeCtx) {
        let mut g = Self::lock(&self.ctx);
        if g.is_none() {
            *g = Some(ctx.clone());
        }
    }
}

async fn create(
    State(ctx): State<ServeCtx>,
    st: Arc<Segments>,
    headers: HeaderMap,
    body: Value,
) -> Response {
    if let Err(e) = ctx.auth().authenticate(ProtocolId::Native, &headers) {
        return err_reply(&e);
    }
    st.attach(&ctx);
    let r: SegmentRequest = match serde_json::from_value(body) {
        Ok(r) => r,
        Err(e) => return err_reply(&ApiError::invalid_param("body", e.to_string())),
    };
    match st.submit(r).await {
        Ok(job) => (StatusCode::ACCEPTED, Json(job.view(false))).into_response(),
        Err(e) => err_reply(&e),
    }
}

async fn status(
    State(ctx): State<ServeCtx>,
    st: Arc<Segments>,
    headers: HeaderMap,
    id: String,
) -> Response {
    if let Err(e) = ctx.auth().authenticate(ProtocolId::Native, &headers) {
        return err_reply(&e);
    }
    match st.job(&id) {
        Some(j) => Json(j.view(false)).into_response(),
        None => err_reply(&ApiError::not_found(format!("segment `{id}` not found"))),
    }
}

async fn stream_status(
    State(ctx): State<ServeCtx>,
    st: Arc<Segments>,
    headers: HeaderMap,
    id: String,
) -> Response {
    if let Err(e) = ctx.auth().authenticate(ProtocolId::Native, &headers) {
        return err_reply(&e);
    }
    match st.stream_view(&id) {
        Some(v) => Json(v).into_response(),
        None => err_reply(&ApiError::not_found(format!(
            "stream `{id}` has no session here"
        ))),
    }
}

async fn stream_close(
    State(ctx): State<ServeCtx>,
    st: Arc<Segments>,
    headers: HeaderMap,
    id: String,
) -> Response {
    if let Err(e) = ctx.auth().authenticate(ProtocolId::Native, &headers) {
        return err_reply(&e);
    }
    let closed = st.close_stream(&id);
    Json(json!({"stream_id": id, "closed": closed})).into_response()
}

async fn list(State(ctx): State<ServeCtx>, st: Arc<Segments>, headers: HeaderMap) -> Response {
    if let Err(e) = ctx.auth().authenticate(ProtocolId::Native, &headers) {
        return err_reply(&e);
    }
    Json(json!({"object": "list", "data": st.list_streams()})).into_response()
}

/// The `/v1/segments` routes (also under `/fv/v1/segments`).
pub fn routes(gate: Arc<ServiceGate>, cfg: SegmentsConfig, work_dir: PathBuf) -> Router<ServeCtx> {
    let st = Arc::new(Segments::new(gate.engine().clone(), cfg, work_dir));
    let _ = std::fs::create_dir_all(&st.work_dir);
    let mut r = Router::new();
    for prefix in ["/v1/segments", "/fv/v1/segments"] {
        let (s1, s2, s3, s4, s5) = (st.clone(), st.clone(), st.clone(), st.clone(), st.clone());
        r = r
            .route(
                prefix,
                post(
                    move |ctx: State<ServeCtx>, headers: HeaderMap, Json(body): Json<Value>| {
                        let st = s1.clone();
                        async move { create(ctx, st, headers, body).await }
                    },
                ),
            )
            .route(
                &format!("{prefix}/streams"),
                get(move |ctx: State<ServeCtx>, headers: HeaderMap| {
                    let st = s2.clone();
                    async move { list(ctx, st, headers).await }
                }),
            )
            .route(
                &format!("{prefix}/streams/{{id}}"),
                get(
                    move |ctx: State<ServeCtx>, headers: HeaderMap, Path(id): Path<String>| {
                        let st = s3.clone();
                        async move { stream_status(ctx, st, headers, id).await }
                    },
                )
                .delete(
                    move |ctx: State<ServeCtx>, headers: HeaderMap, Path(id): Path<String>| {
                        let st = s4.clone();
                        async move { stream_close(ctx, st, headers, id).await }
                    },
                ),
            )
            .route(
                &format!("{prefix}/{{id}}"),
                get(
                    move |ctx: State<ServeCtx>, headers: HeaderMap, Path(id): Path<String>| {
                        let st = s5.clone();
                        async move { status(ctx, st, headers, id).await }
                    },
                ),
            );
    }
    r
}

#[cfg(test)]
mod tests {
    use fastvideo_engine_service::FakeModel;

    use super::*;

    fn req(duration: f64) -> SegmentRequest {
        SegmentRequest {
            stream_id: "str_1".into(),
            segment_index: 0,
            prompt: "a harbor at dusk".into(),
            duration_sec: duration,
            ..SegmentRequest::default()
        }
    }

    #[test]
    fn spec_follows_the_causal_canvas_and_clamps_the_session_length() {
        let sf = FakeModel::sf_wan().caps;
        let cfg = SegmentsConfig::default();
        let s = session_spec(&sf, &req(6.0), &cfg).unwrap();
        assert_eq!(s.fps, 16);
        assert_eq!(s.canvas.1, 480);
        assert!(s.tracks.audio.is_none(), "segments are video-only on Wan");
        assert_eq!(s.max_seconds, Some(120));
        let tight = SegmentsConfig {
            max_seconds: Some(900),
            ..SegmentsConfig::default()
        };
        assert_eq!(
            session_spec(&sf, &req(6.0), &tight).unwrap().max_seconds,
            Some(300)
        );
        let mut r = req(6.0);
        r.resolution = Some("720p".into());
        assert_eq!(
            session_spec(&sf, &r, &cfg).unwrap_err().param.as_deref(),
            Some("resolution")
        );
        r.resolution = Some("480p".into());
        r.aspect_ratio = Some("9:16".into());
        let s = session_spec(&sf, &r, &cfg).unwrap();
        assert!(s.canvas.0 < s.canvas.1);
        let h3 = FakeModel::h3_turbo().caps;
        assert_eq!(
            session_spec(&h3, &req(6.0), &cfg)
                .unwrap_err()
                .param
                .as_deref(),
            Some("model")
        );
    }

    #[test]
    fn requests_are_validated_and_clip_inputs_refused() {
        assert!(validate(&req(6.0)).unwrap().is_empty());
        let mut r = req(6.0);
        r.context = Some(SegmentContext {
            image_url: Some("https://x/0.png".into()),
            ..SegmentContext::default()
        });
        assert_eq!(
            validate(&r).unwrap_err().param.as_deref(),
            Some("context.image_url")
        );
        let mut r = req(6.0);
        r.reference_image_urls = vec!["https://x/m.jpg".into()];
        assert_eq!(validate(&r).unwrap().len(), 1);
        assert_eq!(
            validate(&req(0.0)).unwrap_err().param.as_deref(),
            Some("duration_sec")
        );
        let mut r = req(6.0);
        r.stream_id = "bad id".into();
        assert_eq!(
            validate(&r).unwrap_err().param.as_deref(),
            Some("stream_id")
        );
    }
}
