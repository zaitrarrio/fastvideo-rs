//! Native `/fv/v1/*` API (design §2.1, §6.2; **native** shapes, not an
//! external protocol).
//!
//! | Route | Behaviour |
//! |---|---|
//! | `GET /fv/v1/capabilities` | Models (caps, recipe, tier; causal models also `stream_limits`, design §5.2), tier bindings and aliases: what every public id maps to (risk R10); `auth.mode` (`none` \| `keys` \| `trust-gateway`, no secrets) |
//! | `POST /fv/v1/jobs` | Submit `{model, prompt, ...}` → 202 job object |
//! | `GET /fv/v1/jobs` | Caller's jobs, newest first (`status`, `model`, `limit`, `after`, `order`, `protocol`) |
//! | `GET /fv/v1/jobs/{id}` | Job object (with `protocol` and `metrics`: stage timings from the engine) |
//! | `GET /fv/v1/jobs/{id}/content` | 302 to the signed output URL; 409 until done |
//! | `DELETE /fv/v1/jobs/{id}` | Cancels an unfinished job, deletes a finished one |
//! | `/fv/v1/streams*` | Native WHIP streams: [`crate::streams`] (WP-15), mounted by `app::assemble` |
//!
//! Auth: `Authorization: Bearer <key>` (serve-kit `ProtocolId::Native`).
//!
//! **List scope.** `GET /fv/v1/jobs` lists native jobs by default, because
//! a job's `id` is its API's own id and `/fv/v1/jobs/{id}` resolves native
//! ids only. `?protocol=all` lists the caller's jobs from every API (fal,
//! `/v1/videos`, MiniMax, LTX, …) in the native shape, with `protocol`
//! naming the API that owns each `id` (fetch or cancel it there);
//! `?protocol=<name>` lists one API. Owner scoping is unchanged: with keys,
//! a caller lists only the jobs its key created.

use std::sync::Arc;
use std::time::Duration;

use axum::extract::{Path, Query, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::get;
use axum::{Json, Router};
use fastvideo_protocol::{EditOp, ExtendAt, RetakeMode, VideoEdit};
use fastvideo_protocol::{
    ApiError, AudioInput, AudioRole, BatchProtocol, CanvasSpec, CausalLimits, ErrorCtx, GenerationRequest, HttpReply, Job, JobId, JobState, JobStatus,
    JobView, Keyframe, Length, ListQuery, MediaKind, MediaRef, Reference, NormalizeCtx, ProtocolId, Ratio, SamplingOverrides, Snap,
    SortOrder, StreamCaps, SubmitEndpoint, Task, TimingSpec, ViewCtx,
};
use fastvideo_serve_kit::handlers::{self, error_reply, find_job, SubmitOpts};
use fastvideo_serve_kit::{IngestPolicy, ServeCtx};
use serde::Deserialize;
use serde_json::{json, Value};

use crate::gate::ServiceGate;

/// The native API's protocol identity and error envelope
/// `{"error":{"kind","message","param"}}`.
#[derive(Clone, Copy, Debug, Default)]
pub struct Native;

impl BatchProtocol for Native {
    fn id(&self) -> ProtocolId {
        ProtocolId::Native
    }
    fn new_external_id(&self, job: JobId) -> String {
        format!("fvjob_{}", job.0.simple())
    }
    fn render_error(&self, err: &ApiError, cx: &ErrorCtx) -> HttpReply {
        let kind = serde_json::to_value(err.kind).unwrap_or(Value::Null);
        let mut r = HttpReply::json(
            err.http_status(),
            json!({"error": {"kind": kind, "message": err.message, "param": err.param}}),
        );
        if let Some(id) = &cx.request_id {
            r.push_header("x-request-id", id.clone());
        }
        r
    }
}

/// `POST /fv/v1/jobs` body.
#[derive(Clone, Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NativeBody {
    pub model: String,
    /// Required, except for audio-to-video with an `image_url`.
    #[serde(default)]
    pub prompt: String,
    #[serde(default)]
    pub negative_prompt: Option<String>,
    #[serde(default)]
    pub seed: Option<u64>,
    /// `"WxH"`.
    #[serde(default)]
    pub size: Option<String>,
    #[serde(default)]
    pub width: Option<u32>,
    #[serde(default)]
    pub height: Option<u32>,
    /// `"16:9"` with `short_edge`. With no size or aspect, an image-to-video,
    /// keyframes or H3 reference-to-video job follows its image's aspect
    /// (after EXIF orientation), clamped to the model's aspect range.
    #[serde(default)]
    pub aspect_ratio: Option<String>,
    /// The short-edge tier: with `aspect_ratio`, or alone on an
    /// image-conditioned job (the canvas follows the image at this tier).
    #[serde(default)]
    pub short_edge: Option<u32>,
    #[serde(default)]
    pub seconds: Option<f64>,
    #[serde(default)]
    pub num_frames: Option<u32>,
    #[serde(default)]
    pub fps: Option<u32>,
    #[serde(default)]
    pub steps: Option<u32>,
    #[serde(default)]
    pub guidance: Option<f32>,
    /// First-frame image (https URL or data URI): image-to-video.
    #[serde(default)]
    pub image_url: Option<String>,
    /// Last-frame image: keyframes (with or without `image_url`).
    #[serde(default)]
    pub last_image_url: Option<String>,
    /// Reference images (https URLs or data URIs): reference-to-video
    /// (`Task::Ref2V`). LTX-2.5 (`ltx-pro`, served by its IC-LoRA companion
    /// `ltx25-ref2v`) takes one: a reference sheet of the characters, props
    /// and location, with a "Reference sheet: … Generated video: …" prompt.
    #[serde(default)]
    pub reference_urls: Vec<String>,
    /// Reference conditioning strength, 0 to 1 (default 1: the reference is
    /// kept clean).
    #[serde(default)]
    pub reference_strength: Option<f32>,
    /// Reference LoRA strength, 0 to 2 (LTX: the IC-LoRA's stage-1 strength;
    /// default 1).
    #[serde(default)]
    pub reference_lora_strength: Option<f32>,
    /// Driving audio (https URL or data URI): audio-to-video (`Task::A2V`,
    /// LTX-2.5). The audio sets the length (2 to 20 s; `seconds` /
    /// `num_frames` may ask for less), the output carries it, and
    /// `image_url` (optional) is the first frame.
    #[serde(default)]
    pub audio_url: Option<String>,
    /// Source video (https URL or data URI) of a retake or extend (LTX-2.5;
    /// MP4/MOV/MKV/WebM, 8 to 60 fps, at most 60 s). The output keeps its
    /// size (snapped down to multiples of 32, at most 1920x1088 worth of
    /// pixels) and frame rate.
    #[serde(default)]
    pub video_url: Option<String>,
    /// Retake: regenerate `[start_s, end_s)` seconds of `video_url` (2 to
    /// 20 s; clamped to the video) with `prompt`; the rest is kept.
    #[serde(default)]
    pub start_s: Option<f64>,
    #[serde(default)]
    pub end_s: Option<f64>,
    /// Retake: `replace_audio_and_video` (default), `replace_video` or
    /// `replace_audio`. With `audio_url`, `replace_video`: the window gets
    /// that audio and the picture is regenerated to match it.
    #[serde(default)]
    pub retake_mode: Option<String>,
    /// Extend: add `extend_s` seconds (2 to 20) to `video_url`.
    #[serde(default)]
    pub extend_s: Option<f64>,
    /// Extend: `end` (default) or `start`.
    #[serde(default)]
    pub extend_at: Option<String>,
    /// Extend: seconds of the source the model continues from (1 to 20;
    /// default as many as fit). The rest of the source is copied unchanged.
    #[serde(default)]
    pub context_s: Option<f64>,
}

fn media(s: &str, field: &str) -> Result<MediaRef, ApiError> {
    if s.starts_with("data:") {
        return Ok(MediaRef::DataUri(s.to_owned()));
    }
    url::Url::parse(s)
        .map(MediaRef::Http)
        .map_err(|e| ApiError::invalid_param(field, format!("`{field}` is not a URL: {e}")))
}

fn parse_size(s: &str) -> Result<(u32, u32), ApiError> {
    let bad = || ApiError::invalid_param("size", format!("size `{s}` must be WxH"));
    let (w, h) = s.split_once(['x', 'X', '*']).ok_or_else(bad)?;
    Ok((w.trim().parse().map_err(|_| bad())?, h.trim().parse().map_err(|_| bad())?))
}

/// Submit endpoint for native jobs.
#[derive(Clone, Copy, Debug, Default)]
pub struct NativeSubmit;

impl SubmitEndpoint for NativeSubmit {
    type Body = NativeBody;

    fn normalize(&self, b: NativeBody, cx: &NormalizeCtx) -> Result<GenerationRequest, ApiError> {
        let _ = cx;
        // Audio-to-video with a first frame may leave the prompt empty; so
        // may a retake or an extend (the LTX API's `prompt` is optional).
        if b.prompt.trim().is_empty() && !(b.audio_url.is_some() && b.image_url.is_some()) && b.video_url.is_none() {
            return Err(ApiError::invalid_param("prompt", "`prompt` must not be empty"));
        }
        if b.video_url.is_some() {
            return normalize_edit(b);
        }
        for (set, field) in [
            (b.start_s.is_some(), "start_s"),
            (b.end_s.is_some(), "end_s"),
            (b.retake_mode.is_some(), "retake_mode"),
            (b.extend_s.is_some(), "extend_s"),
            (b.extend_at.is_some(), "extend_at"),
            (b.context_s.is_some(), "context_s"),
        ] {
            if set {
                return Err(ApiError::invalid_param(field, format!("`{field}` needs `video_url` (retake / extend)")));
            }
        }
        let mut r = GenerationRequest::text(ProtocolId::Native, b.model, b.prompt);
        r.negative_prompt = b.negative_prompt;
        r.seed = b.seed;
        r.canvas = match (b.size, b.width, b.height, b.aspect_ratio) {
            (Some(s), None, None, None) => {
                let (width, height) = parse_size(&s)?;
                CanvasSpec::Exact { width, height }
            }
            (None, Some(width), Some(height), None) => CanvasSpec::Exact { width, height },
            (None, None, None, Some(a)) => CanvasSpec::Aspect {
                ratio: a.parse::<Ratio>()?,
                short_edge: b
                    .short_edge
                    .ok_or_else(|| ApiError::invalid_param("short_edge", "`aspect_ratio` needs `short_edge`"))?,
            },
            // No size: an image-conditioned job follows its image
            // (`negotiate`), at `short_edge` when given.
            (None, None, None, None) => match b.short_edge {
                Some(short_edge) => CanvasSpec::FollowImage { short_edge },
                None => CanvasSpec::ModelDefault,
            },
            _ => {
                return Err(ApiError::invalid_param(
                    "size",
                    "give one of `size`, `width`+`height`, or `aspect_ratio`+`short_edge`",
                ))
            }
        };
        let length = match (b.seconds, b.num_frames) {
            (Some(_), Some(_)) => return Err(ApiError::invalid_param("num_frames", "give `seconds` or `num_frames`, not both")),
            (Some(value), None) => Length::Seconds { value, snap: Snap::AlignUp },
            (None, Some(value)) => Length::Frames { value, snap: Snap::Exact },
            (None, None) => Length::ModelDefault,
        };
        r.timing = TimingSpec { length, fps: b.fps };
        r.sampling = SamplingOverrides {
            steps: b.steps,
            guidance: b.guidance,
            reference_strength: b.reference_strength,
            reference_lora_strength: b.reference_lora_strength,
            ..Default::default()
        };
        if let Some(u) = b.image_url {
            r.task = Task::I2V;
            r.keyframes.push(Keyframe { at: fastvideo_protocol::Anchor::First, image: media(&u, "image_url")? });
        }
        if let Some(u) = b.last_image_url {
            r.task = Task::Keyframes;
            r.keyframes.push(Keyframe { at: fastvideo_protocol::Anchor::Last, image: media(&u, "last_image_url")? });
        }
        if !b.reference_urls.is_empty() {
            // `negotiate` refuses references mixed with first/last frames.
            r.task = Task::Ref2V;
            for (i, u) in b.reference_urls.iter().enumerate() {
                r.references.push(Reference {
                    kind: MediaKind::Image,
                    media: media(u, &format!("reference_urls[{i}]"))?,
                });
            }
        }
        if let Some(u) = b.audio_url {
            // Audio-to-video: the images (if any) stay its first/last frames;
            // `negotiate` refuses references mixed in.
            r.task = Task::A2V;
            r.audio_in = Some(AudioInput { media: media(&u, "audio_url")?, role: AudioRole::Drive, max_s: None });
        }
        if r.task == Task::T2V && matches!(r.canvas, CanvasSpec::FollowImage { .. }) {
            return Err(ApiError::invalid_param(
                "short_edge",
                "`short_edge` without `aspect_ratio` follows an input image; text-to-video needs `aspect_ratio`",
            ));
        }
        Ok(r)
    }

    fn submit_reply(&self, job: &Job, cx: &ViewCtx) -> HttpReply {
        HttpReply::json(202, job_json(job, cx))
    }
}

/// A retake (`start_s` + `end_s`) or an extend (`extend_s`) of `video_url`.
fn normalize_edit(b: NativeBody) -> Result<GenerationRequest, ApiError> {
    let video = media(b.video_url.as_deref().unwrap_or_default(), "video_url")?;
    let op = match (b.start_s, b.end_s, b.extend_s) {
        (Some(start_s), Some(end_s), None) => {
            if end_s.partial_cmp(&start_s) != Some(std::cmp::Ordering::Greater) {
                return Err(ApiError::invalid_param("end_s", "`end_s` must be greater than `start_s`"));
            }
            let mode = match b.retake_mode.as_deref() {
                None if b.audio_url.is_some() => RetakeMode::ReplaceVideo,
                None => RetakeMode::default(),
                Some(m) => RetakeMode::parse(m).ok_or_else(|| {
                    ApiError::invalid_param("retake_mode", "`retake_mode` must be one of replace_audio, replace_video, replace_audio_and_video")
                })?,
            };
            if b.extend_at.is_some() || b.context_s.is_some() {
                return Err(ApiError::invalid_param("extend_at", "`extend_at` / `context_s` are extend fields; a retake takes `start_s`, `end_s`"));
            }
            EditOp::Retake { start_s, duration_s: end_s - start_s, mode }
        }
        (None, None, Some(duration_s)) => {
            let at = match b.extend_at.as_deref() {
                None => ExtendAt::End,
                Some(a) => ExtendAt::parse(a).ok_or_else(|| ApiError::invalid_param("extend_at", "`extend_at` must be one of start, end"))?,
            };
            if b.retake_mode.is_some() || b.audio_url.is_some() {
                return Err(ApiError::invalid_param("retake_mode", "an extend takes no `retake_mode` or `audio_url`"));
            }
            EditOp::Extend { duration_s, at, context_s: b.context_s }
        }
        _ => {
            return Err(ApiError::invalid_param(
                "video_url",
                "with `video_url`, give `start_s` and `end_s` (retake) or `extend_s` (extend)",
            ))
        }
    };
    for (set, field) in [
        (b.image_url.is_some(), "image_url"),
        (b.last_image_url.is_some(), "last_image_url"),
        (!b.reference_urls.is_empty(), "reference_urls"),
        (b.seconds.is_some(), "seconds"),
        (b.num_frames.is_some(), "num_frames"),
        (b.fps.is_some(), "fps"),
        (b.aspect_ratio.is_some() || b.short_edge.is_some(), "aspect_ratio"),
    ] {
        if set {
            return Err(ApiError::invalid_param(field, format!("a retake or extend takes no `{field}`: it follows the source video")));
        }
    }
    let mut r = GenerationRequest::text(ProtocolId::Native, b.model, b.prompt);
    r.task = match op {
        EditOp::Retake { .. } => Task::Retake,
        EditOp::Extend { .. } => Task::Extend,
    };
    r.negative_prompt = b.negative_prompt;
    r.seed = b.seed;
    r.canvas = match (b.size, b.width, b.height) {
        (Some(s), None, None) => {
            let (width, height) = parse_size(&s)?;
            CanvasSpec::Exact { width, height }
        }
        (None, Some(width), Some(height)) => CanvasSpec::Exact { width, height },
        (None, None, None) => CanvasSpec::ModelDefault,
        _ => return Err(ApiError::invalid_param("size", "give `size` or `width`+`height` (at most the source's), or neither")),
    };
    r.sampling = SamplingOverrides { steps: b.steps, guidance: b.guidance, ..Default::default() };
    if let Some(u) = b.audio_url {
        r.audio_in = Some(AudioInput { media: media(&u, "audio_url")?, role: AudioRole::Dub, max_s: None });
    }
    r.edit = Some(VideoEdit { video, op });
    Ok(r)
}

/// The native job object.
pub fn job_json(job: &Job, cx: &ViewCtx) -> Value {
    let rfc = |t: time::OffsetDateTime| t.format(&time::format_description::well_known::Rfc3339).ok();
    let error = match &job.state {
        JobState::Failed(e) => json!({"kind": serde_json::to_value(e.kind).unwrap_or(Value::Null), "message": e.message}),
        _ => Value::Null,
    };
    let output = job.artifacts.first().map(|a| {
        json!({
            "url": cx.urls.url_for(a, Duration::from_secs(24 * 3600)).to_string(),
            "mime": a.mime, "bytes": a.bytes, "width": a.width, "height": a.height,
            "frames": a.frames, "fps": a.fps,
            "audio": a.audio.map(|(rate, channels)| json!({"rate": rate, "channels": channels})),
        })
    });
    json!({
        "id": job.external_id,
        "object": "fv.job",
        "status": job.status().as_str(),
        "progress": (job.progress * 100.0).round() / 100.0,
        "queue_position": job.queue_position,
        "model": job.requested_model(),
        "resolved_model": job.resolved.model.0,
        "tier": job.resolved.tier,
        "recipe": job.resolved.recipe,
        "task": job.task(),
        "width": job.resolved.width,
        "height": job.resolved.height,
        "num_frames": job.resolved.num_frames,
        "fps": job.resolved.fps,
        "seed": job.resolved.seed,
        "notes": job.logs.iter().filter(|l| l.message.starts_with("canvas:")).map(|l| l.message.as_str()).collect::<Vec<_>>(),
        "created_at": rfc(job.created_at),
        "started_at": job.started_at.and_then(rfc),
        "completed_at": job.completed_at.and_then(rfc),
        "expires_at": rfc(job.expires_at),
        "error": error,
        "output": output,
        "protocol": job.protocol.as_str(),
        "metrics": metrics_json(job),
    })
}

/// The job's measurements: the engine's `Finished` event metrics
/// (`inference_s` = denoise, `stage_durations` = per-stage seconds as in
/// FastVideo `X-Stage-Durations` and fal `timings`, `peak_memory_mb`,
/// `build_rtf`), plus `queue_s` (created → started) and `run_s` (started →
/// completed) from the job's own timestamps. Unmeasured values are `null`
/// (the stages stay `{}` until the job finishes).
pub fn metrics_json(job: &Job) -> Value {
    let secs = |d: time::Duration| d.as_seconds_f64().max(0.0);
    let m = &job.metrics;
    json!({
        "inference_s": m.inference_s,
        "stage_durations": m.stage_durations,
        "peak_memory_mb": m.peak_memory_mb,
        "build_rtf": m.build_rtf,
        "queue_s": job.started_at.map(|s| secs(s - job.created_at)),
        "run_s": job.started_at.zip(job.completed_at).map(|(s, e)| secs(e - s)),
    })
}

/// `?protocol=` of `GET /fv/v1/jobs`: `native` (default), `all`, or one
/// API's name (`fal`, `openai_videos`, `minimax_v2`, `ltx_v1`, `ltx_v2`,
/// `fastwan`, …). `None` is every API.
fn protocol_filter(p: Option<&str>) -> Result<Option<ProtocolId>, ApiError> {
    match p.map(str::trim) {
        None | Some("") | Some("native") => Ok(Some(ProtocolId::Native)),
        Some("all") => Ok(None),
        Some(name) => ProtocolId::ALL
            .iter()
            .find(|p| p.as_str() == name)
            .map(|p| Some(*p))
            .ok_or_else(|| ApiError::invalid_param("protocol", format!("unknown protocol `{name}` (native, all, or an API name)"))),
    }
}

/// Status and content views.
#[derive(Clone, Copy, Debug, Default)]
pub struct NativeView;

impl JobView for NativeView {
    fn status_reply(&self, job: &Job, cx: &ViewCtx) -> HttpReply {
        HttpReply::json(200, job_json(job, cx))
    }
    fn result_reply(&self, job: &Job, cx: &ViewCtx) -> HttpReply {
        match (job.status(), job.artifacts.first()) {
            (JobStatus::Succeeded, Some(a)) => {
                let u = cx.urls.url_for(a, Duration::from_secs(3600));
                HttpReply::empty(302).with_header("location", u.to_string())
            }
            (s, _) if s.is_terminal() => Native.render_error(
                &ApiError::conflict(format!("job is {} and has no output", s.as_str())),
                &ErrorCtx::default(),
            ),
            _ => Native.render_error(&ApiError::conflict("job is still in progress"), &ErrorCtx::default()),
        }
    }
}

#[derive(Debug, Default, Deserialize)]
struct ListParams {
    status: Option<String>,
    model: Option<String>,
    limit: Option<usize>,
    after: Option<String>,
    order: Option<String>,
    protocol: Option<String>,
}

fn reply_err(e: ApiError) -> Response {
    let r = error_reply(&Native, &e, &ErrorCtx::default());
    let body = r.json_body().cloned().unwrap_or(Value::Null);
    (StatusCode::from_u16(r.status).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR), Json(body)).into_response()
}

async fn list(State(ctx): State<ServeCtx>, headers: HeaderMap, Query(p): Query<ListParams>) -> Response {
    let owner = match ctx.auth().authenticate(ProtocolId::Native, &headers) {
        Ok(o) => o,
        Err(e) => return reply_err(e),
    };
    let protocol = match protocol_filter(p.protocol.as_deref()) {
        Ok(p) => p,
        Err(e) => return reply_err(e),
    };
    let mut q = ListQuery {
        owner,
        protocol,
        model: p.model,
        after: p.after,
        limit: p.limit.unwrap_or(20).clamp(1, 100),
        ..Default::default()
    };
    if let Some(s) = p.status {
        for part in s.split(',') {
            let st = match part.trim() {
                "queued" => JobStatus::Queued,
                "running" => JobStatus::Running,
                "succeeded" => JobStatus::Succeeded,
                "failed" => JobStatus::Failed,
                "cancelled" => JobStatus::Cancelled,
                other => return reply_err(ApiError::invalid_param("status", format!("unknown status `{other}`"))),
            };
            q.statuses.push(st);
        }
    }
    q.order = match p.order.as_deref() {
        None | Some("desc") => SortOrder::Desc,
        Some("asc") => SortOrder::Asc,
        Some(o) => return reply_err(ApiError::invalid_param("order", format!("unknown order `{o}`"))),
    };
    let page = ctx.jobs().list(q).await;
    let cx = ctx.view_ctx(false);
    let data: Vec<Value> = page.items.iter().map(|j| job_json(j, &cx)).collect();
    Json(json!({
        "object": "list",
        "data": data,
        "total": page.total,
        "has_more": page.has_more,
        "first_id": page.first().map(|j| j.external_id.clone()),
        "last_id": page.last().map(|j| j.external_id.clone()),
    }))
    .into_response()
}

async fn delete(State(ctx): State<ServeCtx>, headers: HeaderMap, Path(id): Path<String>) -> Response {
    let res = async {
        let owner = ctx.auth().authenticate(ProtocolId::Native, &headers)?;
        let job = find_job(&ctx, &Native, &id, owner.as_ref()).await?;
        if job.is_terminal() {
            ctx.jobs().remove(job.id).await;
            Ok(json!({"id": id, "object": "fv.job.deleted", "deleted": true}))
        } else {
            let j = fastvideo_serve_kit::events::cancel_job(&ctx, job.id).await?;
            Ok(json!({"id": id, "object": "fv.job", "status": j.status().as_str(), "cancel_requested": true}))
        }
    }
    .await;
    match res {
        Ok(v) => Json(v).into_response(),
        Err(e) => reply_err(e),
    }
}

fn capabilities(gate: &ServiceGate, causal: &CausalLimits) -> Value {
    let caps = gate.engine().caps();
    let models: Vec<Value> = caps
        .entries()
        .map(|e| {
            let mut m = json!({"caps": e.caps, "recipe": e.recipe, "executors": e.executors});
            // Live causal sessions are length-limited (design §5.2).
            if matches!(e.caps.stream, Some(StreamCaps::Causal { .. })) {
                m["stream_limits"] = causal.advertised();
            }
            m
        })
        .collect();
    let tiers: Vec<Value> = caps.tier_bindings().map(|b| serde_json::to_value(b).unwrap_or(Value::Null)).collect();
    json!({
        "object": "fv.capabilities",
        "models": models,
        "tiers": tiers,
        "aliases": gate.aliases(),
        "readiness": gate.engine().readiness(),
    })
}

/// The `auth` object of `/fv/v1/capabilities`: the server's auth mode
/// (`none`, `keys` or `trust-gateway`) only, never keys or tokens. The
/// console reads it to drop its API-key prompts when `mode` is `none`.
pub fn auth_info(mode: fastvideo_serve_kit::AuthMode) -> Value {
    json!({ "mode": mode })
}

/// The `/fv/v1/capabilities` body.
pub type CapsFn = Arc<dyn Fn() -> Value + Send + Sync>;

/// The native routes; `causal` is advertised as each causal model's
/// `stream_limits`.
pub fn routes(gate: Arc<ServiceGate>, body_max: usize, sync_timeout: Duration, causal: CausalLimits) -> Router<ServeCtx> {
    let _ = sync_timeout;
    routes_with(Arc::new(move || capabilities(&gate, &causal)), body_max)
}

/// The native routes with `/fv/v1/capabilities` from `caps` (the gateway
/// aggregates its pools, docs/serve/gateway.md §4).
pub fn routes_with(caps: CapsFn, body_max: usize) -> Router<ServeCtx> {
    let proto = Arc::new(Native);
    let view = Arc::new(NativeView);
    let mut opts = SubmitOpts::new(IngestPolicy::default());
    opts.body_max = body_max;
    Router::new()
        .route(
            "/fv/v1/capabilities",
            get(move |State(ctx): State<ServeCtx>, headers: HeaderMap| {
                let caps = caps.clone();
                async move {
                    match ctx.auth().authenticate(ProtocolId::Native, &headers) {
                        Ok(_) => {
                            let mut body = caps();
                            body["auth"] = auth_info(ctx.auth().mode);
                            Json(body).into_response()
                        }
                        Err(e) => reply_err(e),
                    }
                }
            }),
        )
        .route(
            "/fv/v1/jobs",
            handlers::submit(proto.clone(), Arc::new(NativeSubmit), view.clone(), opts).get(list),
        )
        .route("/fv/v1/jobs/{id}", handlers::status(proto.clone(), view.clone(), "id").delete(delete))
        .route("/fv/v1/jobs/{id}/content", handlers::result(proto, view, "id"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn body(v: Value) -> NativeBody {
        serde_json::from_value(v).unwrap()
    }

    #[test]
    fn reference_urls_make_a_reference_to_video_request() {
        let cx = NormalizeCtx::new(time::OffsetDateTime::UNIX_EPOCH);
        let r = NativeSubmit
            .normalize(
                body(json!({
                    "model": "ltx-pro",
                    "prompt": "Reference sheet: a crab. Generated video: the crab walks.",
                    "size": "1536x896",
                    "reference_urls": ["https://e.x/sheet.png"],
                    "reference_strength": 0.9,
                    "reference_lora_strength": 1.2,
                })),
                &cx,
            )
            .unwrap();
        assert_eq!(r.task, Task::Ref2V);
        assert_eq!(r.references.len(), 1);
        assert_eq!(r.references[0].kind, MediaKind::Image);
        assert_eq!(r.sampling.reference_strength, Some(0.9));
        assert_eq!(r.sampling.reference_lora_strength, Some(1.2));
        assert_eq!(r.canvas, CanvasSpec::Exact { width: 1536, height: 896 });
        let bad = NativeSubmit.normalize(
            body(json!({"model": "ltx-pro", "prompt": "p", "reference_urls": ["not a url"]})),
            &cx,
        );
        assert_eq!(bad.unwrap_err().param.as_deref(), Some("reference_urls[0]"));
        // Without references the task and knobs are untouched.
        let t2v = NativeSubmit
            .normalize(body(json!({"model": "ltx-pro", "prompt": "p"})), &cx)
            .unwrap();
        assert_eq!(t2v.task, Task::T2V);
        assert!(t2v.references.is_empty() && t2v.sampling.is_empty());
    }

    #[test]
    fn audio_url_makes_an_audio_to_video_request() {
        let cx = NormalizeCtx::new(time::OffsetDateTime::UNIX_EPOCH);
        let r = NativeSubmit
            .normalize(
                body(json!({"model": "ltx-turbo", "prompt": "a man talks", "audio_url": "https://e.x/speech.mp3"})),
                &cx,
            )
            .unwrap();
        assert_eq!(r.task, Task::A2V);
        assert_eq!(r.audio_in.as_ref().map(|a| a.role), Some(AudioRole::Drive));
        assert!(r.keyframes.is_empty());
        assert_eq!(r.timing.length, Length::ModelDefault);
        // With a first frame the prompt may be empty; the image stays frame 0.
        let i = NativeSubmit
            .normalize(
                body(json!({"model": "ltx-turbo", "image_url": "data:image/png;base64,AA", "audio_url": "data:audio/wav;base64,AA"})),
                &cx,
            )
            .unwrap();
        assert_eq!((i.task, i.keyframes.len()), (Task::A2V, 1));
        assert!(i.prompt.is_empty());
        // Without an image the prompt is required; a bad URL names its field.
        let e = NativeSubmit.normalize(body(json!({"model": "m", "audio_url": "https://e.x/a.wav"})), &cx);
        assert_eq!(e.unwrap_err().param.as_deref(), Some("prompt"));
        let e = NativeSubmit.normalize(body(json!({"model": "m", "prompt": "p", "audio_url": "nope"})), &cx);
        assert_eq!(e.unwrap_err().param.as_deref(), Some("audio_url"));
    }

    #[test]
    fn video_url_makes_a_retake_or_an_extend() {
        let cx = NormalizeCtx::new(time::OffsetDateTime::UNIX_EPOCH);
        let n = |v: Value| NativeSubmit.normalize(body(v), &cx);
        let r = n(json!({"model": "ltx-pro", "prompt": "it rains", "video_url": "https://e.x/v.mp4", "start_s": 1.0, "end_s": 3.5})).unwrap();
        assert_eq!(r.task, Task::Retake);
        let e = r.edit.unwrap();
        assert_eq!(e.op, EditOp::Retake { start_s: 1.0, duration_s: 2.5, mode: RetakeMode::ReplaceAudioAndVideo });
        assert!(r.audio_in.is_none());
        // New window audio: replace_video by default, the audio a dub.
        let r = n(json!({"model": "ltx-pro", "prompt": "", "video_url": "https://e.x/v.mp4", "start_s": 0, "end_s": 2, "audio_url": "https://e.x/a.wav"})).unwrap();
        assert_eq!(r.edit.unwrap().op, EditOp::Retake { start_s: 0.0, duration_s: 2.0, mode: RetakeMode::ReplaceVideo });
        assert_eq!(r.audio_in.map(|a| a.role), Some(AudioRole::Dub));
        let x = n(json!({"model": "ltx-pro", "prompt": "the road", "video_url": "https://e.x/v.mp4", "extend_s": 4, "extend_at": "start", "context_s": 3})).unwrap();
        assert_eq!(x.task, Task::Extend);
        assert_eq!(x.edit.unwrap().op, EditOp::Extend { duration_s: 4.0, at: ExtendAt::Start, context_s: Some(3.0) });
        for (b, param) in [
            (json!({"model": "m", "prompt": "p", "video_url": "https://e.x/v.mp4"}), "video_url"),
            (json!({"model": "m", "prompt": "p", "video_url": "https://e.x/v.mp4", "start_s": 2, "end_s": 1}), "end_s"),
            (json!({"model": "m", "prompt": "p", "video_url": "https://e.x/v.mp4", "start_s": 0, "end_s": 2, "retake_mode": "x"}), "retake_mode"),
            (json!({"model": "m", "prompt": "p", "video_url": "https://e.x/v.mp4", "extend_s": 2, "extend_at": "middle"}), "extend_at"),
            (json!({"model": "m", "prompt": "p", "video_url": "https://e.x/v.mp4", "extend_s": 2, "seconds": 4}), "seconds"),
            (json!({"model": "m", "prompt": "p", "video_url": "https://e.x/v.mp4", "extend_s": 2, "image_url": "https://e.x/i.png"}), "image_url"),
            (json!({"model": "m", "prompt": "p", "start_s": 0, "end_s": 2}), "start_s"),
            (json!({"model": "m", "prompt": "p", "extend_s": 2}), "extend_s"),
        ] {
            assert_eq!(n(b.clone()).unwrap_err().param.as_deref(), Some(param), "{b}");
        }
    }
}
