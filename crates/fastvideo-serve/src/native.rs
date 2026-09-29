//! Native `/fv/v1/*` API (design §2.1, §6.2; **native** shapes, not an
//! external protocol).
//!
//! | Route | Behaviour |
//! |---|---|
//! | `GET /fv/v1/capabilities` | Models (caps, recipe, tier; causal models also `stream_limits`, design §5.2), tier bindings and aliases: what every public id maps to (risk R10) |
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
use fastvideo_protocol::{
    ApiError, BatchProtocol, CanvasSpec, CausalLimits, ErrorCtx, GenerationRequest, HttpReply, Job, JobId, JobState, JobStatus,
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
        if b.prompt.trim().is_empty() {
            return Err(ApiError::invalid_param("prompt", "`prompt` must not be empty"));
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
                        Ok(_) => Json(caps()).into_response(),
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
}
