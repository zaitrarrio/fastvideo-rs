//! `/v1/segments`: a pull-based segment API over engine sessions, for a
//! control plane that drives an infinite live stream (docs/serve/segments.md).
//!
//! | Route | Behaviour |
//! |---|---|
//! | `POST /v1/segments` | Render the next segment of a stream: `{stream_id, segment_index, prompt, duration_sec, …}` → 202 segment object. The first segment of a `stream_id` opens a session on an executor (exclusive lease); later ones continue it. On a **causal** model (SF-Wan) the session is one rollout: the KV cache, the sink and the decoder state carry across segments and the seams are invisible. On a **clip** model (H3, LTX, Wan) each segment is one clip build anchored on the previous segment's last frame (image-to-video), the duplicated frame dropped |
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
//! - **Clip models** (`StreamCaps::Clip`, `Task::I2V`): the worker keeps the
//!   last frame of the previous segment and conditions the next build on it
//!   (`continuity: "anchor"`); the control plane may instead send its own
//!   `context.image_url` (`"image"`); `reset` or a first segment starts from
//!   text (`"text"`). `end_image_url` anchors the last frame when the model
//!   has `Task::Keyframes`. The clip length snaps up onto the model's frame
//!   grid: the 202 object says how many frames the segment will have.
//! - **Not supported** (refused with `invalid_param`): `context.*` on the
//!   causal path, `context.video_url` everywhere (no extend on this path).
//!   `reference_image_urls` are accepted and ignored (`warnings`).

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
use fastvideo_engine_service::{
    CausalControl, CausalSession, ClipBuild, ClipSession, EngineService,
};
use fastvideo_media::mp4::{write_mp4, AudioTarget, Mp4Spec};
use fastvideo_media::video::FfmpegH264;
use fastvideo_protocol::{
    canvas_for_aspect, resolve_canvas, ApiError, CanvasSpec, CausalLimits, Continuity, ErrorKind,
    ModelCaps, ModelId, Pcm, ProtocolId, Ratio, RgbFrame, SessionSpec, StreamCaps, Task, TrackSet,
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
    /// The model a request without `model` uses; `None`: the first causal
    /// model the engine serves, else the first clip model with image-to-video.
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
    /// A short-edge tier (`480p`, `768p`, …); the model's default when
    /// absent, the nearest served tier (with a warning) when not served.
    #[serde(default)]
    pub resolution: Option<String>,
    /// `16:9` (default), `9:16`, `1:1`, …
    #[serde(default)]
    pub aspect_ratio: Option<String>,
    #[serde(default)]
    pub seed: Option<u64>,
    /// The direction changed: apply the model's prompt-switch policy
    /// (causal); nothing on clip models (every build is conditioned afresh).
    #[serde(default)]
    pub recache: bool,
    /// A scene cut: clear the KV cache (causal) / start from text instead
    /// of the previous segment's last frame (clip).
    #[serde(default)]
    pub reset: bool,
    /// Accepted and ignored (no reference conditioning on this path).
    #[serde(default)]
    pub reference_image_urls: Vec<String>,
    /// Keyframe planning: the frame the segment must end on (clip models
    /// with `Task::Keyframes`); ignored on the causal path.
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
    /// How the segment was conditioned: `native` (one causal rollout),
    /// `anchor` (the previous segment's last frame), `image` (the request's
    /// `context.image_url`), `text` (a first segment, a `reset`, or an
    /// anchor that was lost).
    continuity: &'static str,
    /// Frames the segment is planned to have (set at submit from the model's
    /// frame grid), then the frames it has.
    planned_frames: u32,
    /// The MP4 carries an audio track.
    audio: bool,
    /// Clip continuations: how well the build reproduced its anchor
    /// (`anchor_diff`: mean luma difference between the anchor and the
    /// build's first frame, the one dropped) and how far the first
    /// delivered frame is from the anchor (`step_diff`), both 0–255. A
    /// model that keeps the scene gives a `step_diff` near the clip's own
    /// inter-frame difference; a model that only paints the anchor and
    /// then re-imagines the scene gives a large one.
    seam: Option<(f64, f64)>,
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
            "planned_frames": s.planned_frames,
            "planned_duration_sec": if s.fps > 0 { f64::from(s.planned_frames) / f64::from(s.fps) } else { 0.0 },
            "continuity": s.continuity,
            "audio": s.audio,
            "seam": s.seam.map(|(a, b)| json!({"anchor_diff": a, "step_diff": b})),
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
    /// Wakes the worker on `DELETE` so a clip session is released at once.
    close: Notify,
    last_used: Mutex<Instant>,
    closed: AtomicBool,
    /// Sessions opened for this stream (1 = never restarted).
    sessions: AtomicU64,
    /// Causal: blocks consumed. Clip: builds finished.
    blocks: AtomicU64,
    queued: AtomicU64,
    canvas: (u32, u32),
    fps: u32,
    path: SegPath,
    created_ms: u64,
}

/// Which engine session a stream runs on.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SegPath {
    /// One causal rollout (SF-Wan): segments cut from it.
    Causal,
    /// One clip build per segment, anchored on the previous last frame.
    Clip,
}

impl SegPath {
    /// The path a model's caps allow; `None` when it can neither stream
    /// causally nor build image-anchored clips.
    pub fn of(caps: &ModelCaps) -> Option<Self> {
        match caps.stream {
            Some(StreamCaps::Causal { .. }) => Some(Self::Causal),
            Some(StreamCaps::Clip { .. }) if caps.supports(Task::I2V) => Some(Self::Clip),
            _ => None,
        }
    }
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
            "path": self.path,
            "sessions": self.sessions.load(Ordering::Relaxed),
            "blocks": self.blocks.load(Ordering::Relaxed),
            "clips": self.blocks.load(Ordering::Relaxed),
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

fn parse_ratio(s: &str) -> Option<(u32, u32)> {
    let (w, h) = s.split_once(':')?;
    let (w, h): (u32, u32) = (w.trim().parse().ok()?, h.trim().parse().ok()?);
    (w > 0 && h > 0 && w <= 64 && h <= 64).then_some((w, h))
}

fn parse_short_edge(s: &str) -> Option<u32> {
    s.trim().trim_end_matches(['p', 'P']).parse().ok()
}

/// A stream's session plan: the engine session, the output crop (models
/// that generate padded and crop back, LTX) and the warnings a snapped
/// resolution leaves.
#[derive(Clone, Debug)]
pub struct SegPlan {
    pub spec: SessionSpec,
    /// `(width, height)` of the delivered frames when smaller than the
    /// generated canvas (centre crop); the anchor stays uncropped.
    pub crop: Option<(u32, u32)>,
    pub warnings: Vec<String>,
}

impl SegPlan {
    /// The size the control plane receives.
    pub fn output_size(&self) -> (u32, u32) {
        self.crop.unwrap_or(self.spec.canvas)
    }
}

/// The session plan for a stream's first segment (pure; tests).
pub fn session_spec(
    caps: &ModelCaps,
    r: &SegmentRequest,
    cfg: &SegmentsConfig,
) -> Result<SegPlan, ApiError> {
    let path = SegPath::of(caps).ok_or_else(|| {
        ApiError::invalid_param(
            "model",
            format!(
                "`{}` can neither stream causally nor build image-anchored clips",
                caps.id
            ),
        )
    })?;
    let fps = match (path, &caps.stream) {
        (SegPath::Causal, Some(StreamCaps::Causal { target_fps, .. })) => *target_fps,
        _ => caps.fps.default,
    };
    let mut warnings = Vec::new();
    let tiers = &caps.canvas.short_edges;
    let short = match &r.resolution {
        None => tiers.first().copied().unwrap_or(480),
        Some(s) => {
            let n = parse_short_edge(s).ok_or_else(|| {
                ApiError::invalid_param(
                    "resolution",
                    format!("`{s}` is not a resolution like `480p`"),
                )
            })?;
            if tiers.contains(&n) {
                n
            } else {
                // The nearest served tier: a stream must not fail on a tier
                // name; the control plane learns the canvas from the reply.
                let near = tiers
                    .iter()
                    .copied()
                    .min_by_key(|t| t.abs_diff(n))
                    .unwrap_or(480);
                warnings.push(format!(
                    "resolution {n}p is not served by `{}` (serves {}): using {near}p",
                    caps.id,
                    tiers
                        .iter()
                        .map(|e| format!("{e}p"))
                        .collect::<Vec<_>>()
                        .join(", ")
                ));
                near
            }
        }
    };
    let (rw, rh) = match &r.aspect_ratio {
        None => (16, 9),
        Some(a) => parse_ratio(a).ok_or_else(|| {
            ApiError::invalid_param(
                "aspect_ratio",
                format!("`{a}` is not an aspect ratio like `16:9`"),
            )
        })?,
    };
    let ratio = f64::from(rw) / f64::from(rh);
    let (lo, hi) = caps.canvas.aspect;
    if ratio < f64::from(lo) || ratio > f64::from(hi) {
        return Err(ApiError::invalid_param(
            "aspect_ratio",
            format!("`{}` serves aspect ratios {lo}..{hi}", caps.id),
        ));
    }
    // Like the director: a pad-and-crop model (LTX) generates the padded
    // canvas and the output is centre-cropped to the tier; the anchor
    // frame stays uncropped so continuity does not zoom.
    let (canvas, crop) = if path == SegPath::Clip && caps.canvas.pad_and_crop {
        let spec = CanvasSpec::Aspect {
            ratio: Ratio::new(rw, rh),
            short_edge: short,
        };
        match resolve_canvas(&spec, caps, None) {
            Ok((w, h, c)) => ((w, h), c.filter(|c| *c != (w, h))),
            Err(_) => (canvas_for_aspect(&caps.canvas, ratio, short), None),
        }
    } else {
        (canvas_for_aspect(&caps.canvas, ratio, short), None)
    };
    let tracks = TrackSet::for_model(caps, canvas, fps, ("video", "audio"), 2, false);
    let spec = match path {
        SegPath::Causal => SessionSpec {
            model: caps.id.clone(),
            // Video-only: the causal models are silent.
            tracks: TrackSet {
                audio: None,
                ..tracks
            },
            canvas,
            fps,
            continuity: Continuity::default(),
            max_seconds: Some(cfg.causal.resolve(cfg.max_seconds)?),
            seed: r.seed,
        },
        SegPath::Clip => SessionSpec {
            model: caps.id.clone(),
            tracks,
            canvas,
            fps,
            continuity: Continuity::default(),
            max_seconds: None,
            seed: r.seed,
        },
    };
    Ok(SegPlan {
        spec,
        crop,
        warnings,
    })
}

/// Frames a segment gets: `round(duration × fps)` on the causal path (the
/// rollout is cut anywhere); snapped up onto the model's grid, within its
/// range, on the clip path.
pub fn planned_frames(caps: &ModelCaps, path: SegPath, fps: u32, duration_sec: f64) -> u32 {
    let n = ((duration_sec * f64::from(fps)).round() as u32).max(1);
    match path {
        SegPath::Causal => n,
        SegPath::Clip => {
            let g = &caps.frames;
            g.align_up(n.clamp(g.min, g.max)).unwrap_or(g.max)
        }
    }
}

/// Whether URLs in a request can be fetched by this build.
const CAN_FETCH: bool = cfg!(feature = "http-client");

fn validate(r: &SegmentRequest, caps: &ModelCaps, path: SegPath) -> Result<Vec<String>, ApiError> {
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
    let mut warnings = Vec::new();
    if let Some(c) = &r.context {
        if c.video_url.is_some() {
            return Err(ApiError::invalid_param(
                "context.video_url",
                match path {
                    SegPath::Causal => "the causal path keeps its own state: video context is not supported (use continuity \"native\")",
                    SegPath::Clip => "video context (extend) is not supported on this path: the backend chains segments on the last frame itself (use continuity \"native\" or \"last_frame\")",
                },
            ));
        }
        if let Some(u) = &c.image_url {
            match path {
                SegPath::Causal => {
                    return Err(ApiError::invalid_param("context.image_url", "the causal path keeps its own state: image conditioning is not supported (use continuity \"native\")"));
                }
                SegPath::Clip => check_url("context.image_url", u)?,
            }
        }
    }
    if let Some(u) = &r.end_image_url {
        match path {
            SegPath::Clip if caps.supports(Task::Keyframes) => check_url("end_image_url", u)?,
            SegPath::Clip => warnings.push(format!(
                "end_image_url ignored: `{}` cannot render toward a last frame",
                caps.id
            )),
            SegPath::Causal => warnings.push(
                "end_image_url ignored: the causal path renders toward no planned frame".to_owned(),
            ),
        }
    }
    if !r.reference_image_urls.is_empty() {
        warnings.push(
            "reference_image_urls ignored: no reference conditioning on this path".to_owned(),
        );
    }
    Ok(warnings)
}

/// A fetchable `http(s)` URL (or `file://` in builds without the HTTP
/// client, for tests).
fn check_url(param: &str, u: &str) -> Result<(), ApiError> {
    let ok = u.starts_with("https://") || u.starts_with("http://");
    if ok && !CAN_FETCH {
        return Err(ApiError::invalid_param(
            param,
            "this build cannot fetch URLs (feature `http-client`)",
        ));
    }
    if ok || u.starts_with("file://") {
        return Ok(());
    }
    Err(ApiError::invalid_param(
        param,
        format!("`{u}` is not an http(s) URL"),
    ))
}

/// Fetches an image URL (`context.image_url`, `end_image_url`), checks that
/// it decodes and writes it as a PNG at `path`.
async fn fetch_image(url: &str, path: &std::path::Path) -> Result<(), ApiError> {
    const MAX_BYTES: usize = 32 << 20;
    let bytes: Vec<u8> = if let Some(p) = url.strip_prefix("file://") {
        let b = tokio::fs::read(p)
            .await
            .map_err(|e| ApiError::invalid_param("image", format!("reading {p}: {e}")))?;
        if b.len() > MAX_BYTES {
            return Err(ApiError::invalid_param(
                "image",
                "the image is larger than 32 MB",
            ));
        }
        b
    } else {
        #[cfg(feature = "http-client")]
        {
            let client = reqwest::Client::builder()
                .timeout(Duration::from_secs(20))
                .build()
                .map_err(|e| ApiError::internal(format!("http client: {e}")))?;
            let resp =
                client.get(url).send().await.map_err(|e| {
                    ApiError::invalid_param("image", format!("fetching {url}: {e}"))
                })?;
            if !resp.status().is_success() {
                return Err(ApiError::invalid_param(
                    "image",
                    format!("fetching {url}: HTTP {}", resp.status()),
                ));
            }
            let b = resp
                .bytes()
                .await
                .map_err(|e| ApiError::invalid_param("image", format!("reading {url}: {e}")))?;
            if b.len() > MAX_BYTES {
                return Err(ApiError::invalid_param(
                    "image",
                    "the image is larger than 32 MB",
                ));
            }
            b.to_vec()
        }
        #[cfg(not(feature = "http-client"))]
        {
            return Err(ApiError::invalid_param(
                "image",
                "this build cannot fetch URLs (feature `http-client`)",
            ));
        }
    };
    let path = path.to_path_buf();
    tokio::task::spawn_blocking(move || {
        let img = image::load_from_memory(&bytes)
            .map_err(|e| ApiError::invalid_param("image", format!("not an image: {e}")))?;
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir)
                .map_err(|e| ApiError::internal(format!("creating {}: {e}", dir.display())))?;
        }
        img.to_rgb8()
            .save(&path)
            .map_err(|e| ApiError::internal(format!("writing {}: {e}", path.display())))
    })
    .await
    .map_err(|e| ApiError::internal(format!("image task: {e}")))?
}

/// A stored segment MP4.
struct Stored {
    url: String,
    frames: u32,
    audio: bool,
    encode_ms: f64,
}

/// Writes an RGB24 frame as a PNG (the next build's first-frame anchor).
fn write_png(f: &RgbFrame, path: &PathBuf) -> Result<(), ApiError> {
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)
            .map_err(|e| ApiError::internal(format!("creating {}: {e}", dir.display())))?;
    }
    let img = image::RgbImage::from_raw(f.width, f.height, f.data.to_vec())
        .ok_or_else(|| ApiError::internal("anchor frame has the wrong size"))?;
    img.save(path)
        .map_err(|e| ApiError::internal(format!("writing {}: {e}", path.display())))
}

/// Mean absolute luma difference (0–255) between two frames of one size
/// (sampled every 4th pixel); 255 when the sizes differ.
fn luma_diff(a: &RgbFrame, b: &RgbFrame) -> f64 {
    if a.width != b.width || a.height != b.height || a.data.len() != b.data.len() {
        return 255.0;
    }
    let luma = |p: &[u8]| 0.299 * f64::from(p[0]) + 0.587 * f64::from(p[1]) + 0.114 * f64::from(p[2]);
    let (mut sum, mut n) = (0.0, 0u64);
    for (pa, pb) in a.data.chunks_exact(3).zip(b.data.chunks_exact(3)).step_by(4) {
        sum += (luma(pa) - luma(pb)).abs();
        n += 1;
    }
    if n == 0 {
        0.0
    } else {
        (sum / n as f64 * 1000.0).round() / 1000.0
    }
}

/// Drops one video frame's worth of samples from the head of `pcm` (the
/// anchor frame a continuation duplicates).
fn drop_audio_head(pcm: &Pcm, fps: u32) -> Pcm {
    let per_frame = (pcm.rate / fps.max(1)) as usize * usize::from(pcm.channels);
    let skip = per_frame.min(pcm.samples.len());
    Pcm::new(pcm.rate, pcm.channels, pcm.samples[skip..].to_vec())
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
                .resolve(m)
                .cloned()
                .or_else(|| caps.get(&id).cloned())
                .or_else(|| caps.models().find(|c| c.answers_to(m)).cloned())
                .ok_or_else(|| {
                    ApiError::invalid_param("model", format!("model `{m}` is not served here"))
                });
        }
        caps.models()
            .find(|c| SegPath::of(c) == Some(SegPath::Causal))
            .or_else(|| {
                caps.models()
                    .find(|c| SegPath::of(c) == Some(SegPath::Clip))
            })
            .cloned()
            .ok_or_else(|| {
                ApiError::invalid_param(
                    "model",
                    "no causal or image-to-video clip model is served here; give `model`",
                )
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
        if let Some(id) = Self::lock(&self.by_index)
            .get(&(r.stream_id.clone(), r.segment_index))
            .cloned()
        {
            if let Some(j) = self.job(&id) {
                return Ok(j);
            }
        }
        let caps = self.resolve_model(&r)?;
        let path = SegPath::of(&caps).ok_or_else(|| {
            ApiError::invalid_param(
                "model",
                format!(
                    "`{}` can neither stream causally nor build image-anchored clips",
                    caps.id
                ),
            )
        })?;
        let mut warnings = validate(&r, &caps, path)?;
        let plan = session_spec(&caps, &r, &self.cfg)?;
        warnings.extend(plan.warnings.iter().cloned());
        let planned = planned_frames(&caps, path, plan.spec.fps, r.duration_sec);
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
            None => self.open_slot(&r.stream_id, caps, plan)?,
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
                continuity: match path {
                    SegPath::Causal => "native",
                    SegPath::Clip => "text",
                },
                planned_frames: planned,
                audio: false,
                seam: None,
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

    /// Opens a stream's slot and spawns its worker. The engine session itself
    /// opens inside the worker on the first segment (so an executor busy with
    /// another stream fails that segment, not the submit).
    fn open_slot(
        self: &Arc<Self>,
        stream_id: &str,
        caps: ModelCaps,
        plan: SegPlan,
    ) -> Result<Arc<Slot>, ApiError> {
        let (tx, rx) = mpsc::channel(self.cfg.queue);
        let path = SegPath::of(&caps).unwrap_or(SegPath::Causal);
        let canvas = plan.output_size();
        let spec = plan.spec.clone();
        let slot = Arc::new(Slot {
            stream_id: stream_id.to_owned(),
            model: caps.id.clone(),
            tx,
            control: Mutex::new(None),
            close: Notify::new(),
            last_used: Mutex::new(Instant::now()),
            closed: AtomicBool::new(false),
            sessions: AtomicU64::new(0),
            blocks: AtomicU64::new(0),
            queued: AtomicU64::new(0),
            canvas,
            fps: spec.fps,
            path,
            created_ms: now_ms(),
        });
        Self::lock(&self.slots).insert(stream_id.to_owned(), slot.clone());
        let me = self.clone();
        let s2 = slot.clone();
        tokio::spawn(async move {
            match path {
                SegPath::Causal => me.run_slot(s2, caps, spec, rx).await,
                SegPath::Clip => me.run_clip_slot(s2, caps, plan, rx).await,
            }
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
        slot.close.notify_one();
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
                job.set(|s| s.restarted = restarted);
                self.finish(&job, Err(e)).await;
                continue;
            }

            let frames: Vec<RgbFrame> = carry.drain(..need).collect();
            let generation_ms = t0.elapsed().as_secs_f64() * 1e3;
            let last_block = next_block.saturating_sub(1);
            let blocks = first_block.map(|f| (f, last_block));
            let outcome = self
                .encode_and_store(&job, &spec, spec.canvas, frames, None)
                .await;
            job.set(|s| {
                s.restarted = restarted;
                s.stale_blocks = stale_blocks;
                s.discarded_blocks = discarded;
                s.blocks = blocks;
                s.generation_ms = Some(generation_ms);
            });
            self.finish(&job, outcome).await;
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

    /// The worker for a clip model: one `ClipSession` (an executor lease)
    /// per stream, one build per segment, the previous segment's last frame
    /// as the next build's first frame.
    async fn run_clip_slot(
        self: Arc<Self>,
        slot: Arc<Slot>,
        caps: ModelCaps,
        plan: SegPlan,
        mut rx: mpsc::Receiver<Arc<SegmentJob>>,
    ) {
        let spec = plan.spec.clone();
        let size = plan.output_size();
        let mut session: Option<ClipSession> = None;
        // The last frame of the previous segment, as a PNG (the build's
        // first-frame input) and in memory (the seam measurement).
        let mut anchor: Option<PathBuf> = None;
        let mut anchor_frame: Option<RgbFrame> = None;
        let audio_channels = spec.tracks.audio.as_ref().map(|a| a.channels);
        loop {
            let next = tokio::select! {
                j = tokio::time::timeout(self.cfg.idle_ttl, rx.recv()) => j,
                () = slot.close.notified() => break,
            };
            let job = match next {
                Ok(Some(j)) => j,
                Ok(None) => break,
                Err(_) => {
                    tracing::info!(stream = %slot.stream_id, idle_s = self.cfg.idle_ttl.as_secs(), "segment stream idle: closing its session");
                    break;
                }
            };
            slot.queued.fetch_sub(1, Ordering::Relaxed);
            if slot.closed.load(Ordering::Relaxed) {
                self.finish(&job, Err(ApiError::conflict("the stream was closed")))
                    .await;
                continue;
            }
            job.set(|s| s.status = SegmentState::Running);
            let t0 = Instant::now();
            let planned = planned_frames(&caps, SegPath::Clip, spec.fps, job.req.duration_sec);

            // Conditioning: the request's image, else the anchor, else text.
            let mut first_frame: Option<PathBuf> = None;
            let mut continuity = "text";
            let mut failed: Option<ApiError> = None;
            if let Some(u) = job
                .req
                .context
                .as_ref()
                .and_then(|c| c.image_url.as_deref())
            {
                let p = self.work_dir.join(format!("{}-first.png", job.id));
                match fetch_image(u, &p).await {
                    Ok(()) => {
                        first_frame = Some(p);
                        continuity = "image";
                    }
                    Err(e) => failed = Some(e),
                }
            } else if !job.req.reset {
                if let Some(a) = anchor.clone() {
                    first_frame = Some(a);
                    continuity = "anchor";
                }
            }
            let mut last_frame: Option<PathBuf> = None;
            if failed.is_none() && caps.supports(Task::Keyframes) {
                if let Some(u) = job.req.end_image_url.as_deref() {
                    let p = self.work_dir.join(format!("{}-last.png", job.id));
                    match fetch_image(u, &p).await {
                        Ok(()) => last_frame = Some(p),
                        Err(e) => failed = Some(e),
                    }
                }
            }

            // Open (or reopen) the session.
            let mut restarted = false;
            if failed.is_none() && session.is_none() {
                match self.engine.open_clip_session(spec.clone()).await {
                    Ok(s) => {
                        if slot.sessions.fetch_add(1, Ordering::Relaxed) > 0 {
                            restarted = true;
                        }
                        session = Some(s);
                    }
                    Err(e) => failed = Some(e),
                }
            }

            let mut built: Option<(Vec<RgbFrame>, Option<Pcm>)> = None;
            // An engine failure (a busy executor, a failed build) drops the
            // session so the next segment reopens; a bad input does not.
            let mut engine_failed = false;
            if failed.is_none() {
                let s = session.as_ref().expect("opened above");
                let build = ClipBuild {
                    prompt: job.req.prompt.trim().to_owned(),
                    negative_prompt: None,
                    // Like the director: the stream seed plus the segment
                    // index, so consecutive builds differ.
                    seed: job.req.seed.map(|s| s.wrapping_add(job.req.segment_index)),
                    seconds: None,
                    frames: Some(planned),
                    canvas: None,
                    first_frame: first_frame.clone(),
                    last_frame: last_frame.clone(),
                    audio_drive: None,
                };
                let res = match s.build(build).await {
                    Ok(h) => h.wait().await,
                    Err(e) => Err(e),
                };
                match res {
                    Ok(out) => {
                        let frames = out.frames.unwrap_or_default();
                        if frames.is_empty() {
                            failed = Some(ApiError::engine_failed("the build returned no frames"));
                            engine_failed = true;
                        } else {
                            built = Some((frames, out.audio));
                        }
                    }
                    Err(e) => {
                        failed = Some(e);
                        engine_failed = true;
                    }
                }
            }
            if let Some(p) = first_frame.as_ref().filter(|p| Some(*p) != anchor.as_ref()) {
                let _ = tokio::fs::remove_file(p).await;
            }
            if let Some(p) = &last_frame {
                let _ = tokio::fs::remove_file(p).await;
            }

            let Some((mut frames, mut audio)) = built else {
                let e = failed.unwrap_or_else(|| ApiError::internal("no build"));
                tracing::warn!(stream = %slot.stream_id, segment = job.req.segment_index, error = %e.message, "segment failed");
                // After an engine error the next segment reopens the
                // session; the anchor survives, so the picture continues.
                if engine_failed || session.is_none() {
                    session = None;
                }
                job.set(|s| {
                    s.restarted = restarted;
                    s.continuity = continuity;
                });
                self.finish(&job, Err(e)).await;
                continue;
            };
            let generation_ms = t0.elapsed().as_secs_f64() * 1e3;
            slot.blocks.fetch_add(1, Ordering::Relaxed);

            // A continuation's first frame is the anchor itself: drop it (and
            // its audio) so segment N+1 starts on the frame after N's last.
            let mut seam = None;
            if continuity == "anchor" && frames.len() > 1 {
                if let Some(a) = &anchor_frame {
                    let m = (luma_diff(a, &frames[0]), luma_diff(a, &frames[1]));
                    tracing::info!(stream = %slot.stream_id, segment = job.req.segment_index, anchor_diff = m.0, step_diff = m.1, "clip seam");
                    seam = Some(m);
                }
                frames.remove(0);
                if let Some(a) = audio.take() {
                    audio = Some(drop_audio_head(&a, spec.fps));
                }
            }
            // The next segment's anchor: this one's last frame.
            let next = self.work_dir.join(format!(
                "anchor-{}-{}.png",
                slot.stream_id, job.req.segment_index
            ));
            let last = frames.last().cloned().expect("non-empty");
            let n2 = next.clone();
            let wrote = tokio::task::spawn_blocking(move || write_png(&last, &n2))
                .await
                .map_err(|e| ApiError::internal(format!("anchor task: {e}")))
                .and_then(|r| r);
            match wrote {
                Ok(()) => {
                    if let Some(old) = anchor.replace(next) {
                        let _ = tokio::fs::remove_file(old).await;
                    }
                    anchor_frame = frames.last().cloned();
                }
                Err(e) => {
                    tracing::warn!(stream = %slot.stream_id, error = %e.message, "anchor frame not written: the next segment starts from text");
                    anchor = None;
                    anchor_frame = None;
                }
            }

            // The delivered frames: centre-cropped to the tier on a
            // pad-and-crop model (the anchor above stays uncropped).
            if let Some((w, h)) = plan.crop {
                frames = tokio::task::spawn_blocking(move || {
                    frames
                        .iter()
                        .map(|f| f.crop_center(w, h))
                        .collect::<Vec<_>>()
                })
                .await
                .unwrap_or_default();
            }
            let outcome = self
                .encode_and_store(
                    &job,
                    &spec,
                    size,
                    frames,
                    audio.as_ref().map(|a| (a, audio_channels)),
                )
                .await;
            job.set(|s| {
                s.restarted = restarted;
                s.continuity = continuity;
                s.seam = seam;
                s.generation_ms = Some(generation_ms);
            });
            self.finish(&job, outcome).await;
        }
        slot.closed.store(true, Ordering::Relaxed);
        if let Some(s) = session.take() {
            s.close();
        }
        if let Some(a) = anchor.take() {
            let _ = tokio::fs::remove_file(a).await;
        }
        let mut slots = Self::lock(&self.slots);
        if slots
            .get(&slot.stream_id)
            .is_some_and(|s| Arc::ptr_eq(s, &slot))
        {
            slots.remove(&slot.stream_id);
        }
    }

    /// Encodes the segment's frames (and audio) to an MP4 off the runtime
    /// and stores it: the signed URL, the frame count and the encode time.
    async fn encode_and_store(
        &self,
        job: &SegmentJob,
        spec: &SessionSpec,
        size: (u32, u32),
        frames: Vec<RgbFrame>,
        audio: Option<(&Pcm, Option<u8>)>,
    ) -> Result<Stored, ApiError> {
        let t1 = Instant::now();
        let ctx = self.engine_ctx();
        let out = self.work_dir.join(format!("{}.mp4", job.id));
        let target = audio.map(|(pcm, ch)| AudioTarget {
            rate: pcm.rate,
            channels: ch.unwrap_or(pcm.channels),
            bitrate_bps: 128_000,
        });
        let mp4 = Mp4Spec {
            width: size.0,
            height: size.1,
            fps: spec.fps,
            quality: 19,
            encoder: self.cfg.encoder,
            audio: target,
            faststart: true,
            fragmented: false,
        };
        let out2 = out.clone();
        let n = frames.len() as u32;
        let pcm = audio.map(|(p, _)| p.clone());
        let encoded =
            tokio::task::spawn_blocking(move || write_mp4(&out2, mp4, &frames, pcm.as_ref()))
                .await
                .map_err(|e| ApiError::internal(format!("encode task: {e}")))
                .and_then(|r| {
                    r.map_err(|e| ApiError::internal(format!("encoding the segment: {e}")))
                })?;
        let url = match ctx {
            Some(ctx) => {
                let meta = ArtifactMeta {
                    file_name: format!("{}-{}.mp4", job.req.stream_id, job.req.segment_index),
                    mime: "video/mp4".into(),
                    width: size.0,
                    height: size.1,
                    frames: n,
                    fps: spec.fps,
                    audio: target.map(|t| (t.rate, t.channels)),
                };
                let a = ctx.artifacts().put(&encoded, meta).await?;
                ctx.artifacts()
                    .signer()
                    .url_for(&a, self.cfg.url_ttl)
                    .to_string()
            }
            None => format!("file://{}", encoded.display()),
        };
        Ok(Stored {
            url,
            frames: n,
            audio: target.is_some(),
            encode_ms: t1.elapsed().as_secs_f64() * 1e3,
        })
    }

    /// Records a segment's outcome, calls back and wakes waiters.
    async fn finish(&self, job: &SegmentJob, outcome: Result<Stored, ApiError>) {
        match outcome {
            Ok(st) => job.set(|s| {
                s.status = SegmentState::Succeeded;
                s.video_url = Some(st.url);
                s.frames = st.frames;
                s.planned_frames = st.frames;
                s.audio = st.audio;
                s.encode_ms = Some(st.encode_ms);
                s.finished_ms = Some(now_ms());
            }),
            Err(e) => job.set(|s| {
                s.status = SegmentState::Failed;
                s.error = Some(e.message.clone());
                s.finished_ms = Some(now_ms());
            }),
        }
        self.callback(job).await;
        job.done.notify_waiters();
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
        let SegPlan {
            spec: s,
            warnings: w,
            ..
        } = session_spec(&sf, &req(6.0), &cfg).unwrap();
        assert!(w.is_empty());
        assert_eq!(s.fps, 16);
        assert_eq!(s.canvas.1, 480);
        assert!(s.tracks.audio.is_none(), "segments are video-only on Wan");
        assert_eq!(s.max_seconds, Some(120));
        assert_eq!(planned_frames(&sf, SegPath::Causal, 16, 6.0), 96);
        let tight = SegmentsConfig {
            max_seconds: Some(900),
            ..SegmentsConfig::default()
        };
        assert_eq!(
            session_spec(&sf, &req(6.0), &tight)
                .unwrap()
                .spec
                .max_seconds,
            Some(300)
        );
        // A tier the model does not serve snaps to the nearest one, with a warning.
        let mut r = req(6.0);
        r.resolution = Some("720p".into());
        let SegPlan {
            spec: s,
            warnings: w,
            ..
        } = session_spec(&sf, &r, &cfg).unwrap();
        assert_eq!(s.canvas.1, 480);
        assert_eq!(w.len(), 1, "{w:?}");
        r.resolution = Some("480p".into());
        r.aspect_ratio = Some("9:16".into());
        let s = session_spec(&sf, &r, &cfg).unwrap().spec;
        assert!(s.canvas.0 < s.canvas.1);
        r.resolution = Some("tall".into());
        assert_eq!(
            session_spec(&sf, &r, &cfg).unwrap_err().param.as_deref(),
            Some("resolution")
        );
    }

    #[test]
    fn clip_models_get_a_clip_session_on_their_frame_grid() {
        let h3 = FakeModel::h3_turbo().caps;
        let cfg = SegmentsConfig::default();
        assert_eq!(SegPath::of(&h3), Some(SegPath::Clip));
        let SegPlan {
            spec: s,
            warnings: w,
            crop,
        } = session_spec(&h3, &req(6.0), &cfg).unwrap();
        assert!(w.is_empty());
        assert_eq!(crop, None);
        assert_eq!(s.fps, 24);
        assert_eq!(s.canvas.1, 768, "H3's default tier");
        assert!(s.tracks.audio.is_some(), "H3 segments carry audio");
        assert_eq!(s.max_seconds, None);
        // 6 s = 144 frames, up onto 17n+5: 158.
        assert_eq!(planned_frames(&h3, SegPath::Clip, 24, 6.0), 158);
        // Below the grid's minimum: the minimum.
        assert_eq!(planned_frames(&h3, SegPath::Clip, 24, 1.0), h3.frames.min);
        assert_eq!(planned_frames(&h3, SegPath::Clip, 24, 30.0), h3.frames.max);
        // LTX pads to its 64 multiple and crops back to the tier.
        let ltx = FakeModel::ltx_turbo().caps;
        let mut r = req(6.0);
        r.resolution = Some("480p".into());
        let p = session_spec(&ltx, &r, &cfg).unwrap();
        assert_eq!(p.output_size().1, 480);
        assert!(p.spec.canvas.1 >= 480, "{:?}", p.spec.canvas);
        assert_eq!(planned_frames(&ltx, SegPath::Clip, 24, 6.0), 145);
        let wan = FakeModel::wan().caps;
        assert_eq!(
            SegPath::of(&wan),
            None,
            "text-only clip models cannot chain"
        );
        assert_eq!(
            session_spec(&wan, &req(6.0), &cfg)
                .unwrap_err()
                .param
                .as_deref(),
            Some("model")
        );
    }

    #[test]
    fn requests_are_validated_and_clip_inputs_refused() {
        let sf = FakeModel::sf_wan().caps;
        let h3 = FakeModel::h3_turbo().caps;
        assert!(validate(&req(6.0), &sf, SegPath::Causal)
            .unwrap()
            .is_empty());
        let mut r = req(6.0);
        r.context = Some(SegmentContext {
            image_url: Some("file:///x/0.png".into()),
            ..SegmentContext::default()
        });
        assert_eq!(
            validate(&r, &sf, SegPath::Causal)
                .unwrap_err()
                .param
                .as_deref(),
            Some("context.image_url")
        );
        // The clip path takes an image, never a video context.
        assert!(validate(&r, &h3, SegPath::Clip).unwrap().is_empty());
        r.context = Some(SegmentContext {
            video_url: Some("file:///x/0.mp4".into()),
            ..SegmentContext::default()
        });
        assert_eq!(
            validate(&r, &h3, SegPath::Clip)
                .unwrap_err()
                .param
                .as_deref(),
            Some("context.video_url")
        );
        let mut r = req(6.0);
        r.reference_image_urls = vec!["https://x/m.jpg".into()];
        assert_eq!(validate(&r, &sf, SegPath::Causal).unwrap().len(), 1);
        // end_image_url: a keyframe on H3, a warning on the causal path.
        let mut r = req(6.0);
        r.end_image_url = Some("file:///x/end.png".into());
        assert!(validate(&r, &h3, SegPath::Clip).unwrap().is_empty());
        assert_eq!(validate(&r, &sf, SegPath::Causal).unwrap().len(), 1);
        r.end_image_url = Some("ftp://x/end.png".into());
        assert_eq!(
            validate(&r, &h3, SegPath::Clip)
                .unwrap_err()
                .param
                .as_deref(),
            Some("end_image_url")
        );
        assert_eq!(
            validate(&req(0.0), &sf, SegPath::Causal)
                .unwrap_err()
                .param
                .as_deref(),
            Some("duration_sec")
        );
        let mut r = req(6.0);
        r.stream_id = "bad id".into();
        assert_eq!(
            validate(&r, &sf, SegPath::Causal)
                .unwrap_err()
                .param
                .as_deref(),
            Some("stream_id")
        );
    }

    #[test]
    fn seam_measure_is_zero_on_equal_frames_and_grows_with_the_change() {
        let f = |v: u8| RgbFrame::new(8, 4, bytes::Bytes::from(vec![v; 8 * 4 * 3]), 0).unwrap();
        assert_eq!(luma_diff(&f(10), &f(10)), 0.0);
        assert_eq!(luma_diff(&f(10), &f(30)), 20.0);
        let other = RgbFrame::new(4, 4, bytes::Bytes::from(vec![0; 4 * 4 * 3]), 0).unwrap();
        assert_eq!(luma_diff(&f(10), &other), 255.0);
    }

    #[test]
    fn audio_head_follows_the_dropped_anchor_frame() {
        let pcm = Pcm::new(48_000, 2, vec![1.0f32; 48_000 * 2]);
        let cut = drop_audio_head(&pcm, 24);
        assert_eq!(cut.samples.len(), 48_000 * 2 - 2000 * 2);
        let short = Pcm::new(48_000, 2, vec![1.0f32; 10]);
        assert!(drop_audio_head(&short, 24).samples.is_empty());
    }
}
