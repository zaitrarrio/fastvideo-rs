//! The FastWan Video API (design §4.2; minimax-fastvideo §2.2, §2.4), exactly
//! as `streaming-client/fastwan_link.py` calls it:
//!
//! | Call | Reply |
//! |---|---|
//! | `POST /generate` `{prompt,width,height,num_frames,fps,seed}` | 200 `{prompt_id, status}` |
//! | `GET /status/{prompt_id}` | 200 `{status, error?}`; `status` in `queued\|processing\|completed\|failed`, `error` a string |
//! | `GET /video/{prompt_id}` | 200 raw MP4 bytes |
//! | `DELETE /video/{prompt_id}` | 200 `{}`: cancels a running job, removes it and its MP4 |
//! | `GET /health` (binary) | `model_loaded` must be truthy ([`health_body`]) |
//! | `GET /` (binary) | `{"model": <served name>}` ([`root_body`]) |
//!
//! Errors are FastAPI `{"detail": "<string>"}`. The client fails a clip on
//! 400/413/415/422 and treats anything else as "server unreachable, retry",
//! so `Loading` is 503 and a full queue 429. `num_frames` must be on the Wan
//! `4k+1` grid.
//!
//! The API has no model field: every job runs one configured model
//! ([`crate::VideosConfig::fastwan_model`], default the first Wan clip model).

use std::sync::Arc;

use axum::extract::{Path, State};
use axum::http::HeaderMap;
use axum::response::Response;
use axum::routing::get;
use axum::Router;
use fastvideo_protocol::{
    ApiError, BatchProtocol, CanvasSpec, ErrorCtx, Family, GenerationRequest, HttpReply, Job,
    JobId, JobState, JobStatus, JobView, Length, ModelCaps, NormalizeCtx, ProtocolId, Snap,
    StreamCaps, SubmitEndpoint, Task, TimingSpec, ViewCtx,
};
use fastvideo_serve_kit::handlers::{self, find_job, SubmitOpts};
use fastvideo_serve_kit::{into_response, EngineGate, ServeCtx};
use serde::Deserialize;
use serde_json::{json, Map, Value};

use crate::error::{fastapi_error, fastapi_error_status};
use crate::videos::{artifact_reply, delete_job, metadata};
use crate::VideosConfig;

/// The FastWan API (`ProtocolId::FastWan`).
#[derive(Clone, Copy, Debug, Default)]
pub struct FastWanApi;

impl BatchProtocol for FastWanApi {
    fn id(&self) -> ProtocolId {
        ProtocolId::FastWan
    }
    /// A hyphenated UUID (`prompt_id`).
    fn new_external_id(&self, job: JobId) -> String {
        job.0.to_string()
    }
    fn render_error(&self, err: &ApiError, _cx: &ErrorCtx) -> HttpReply {
        fastapi_error(err)
    }
}

/// `POST /generate` body (`fastwan_link.py:458-465`). Unknown fields are
/// ignored, as a FastAPI model does by default.
#[derive(Clone, Debug, Default, PartialEq, Deserialize)]
pub struct FastWanGenerate {
    pub prompt: String,
    pub width: Option<u32>,
    pub height: Option<u32>,
    pub num_frames: Option<u32>,
    pub fps: Option<u32>,
    pub seed: Option<i64>,
}

/// `POST /generate` normalization; `model` is the configured FastWan model.
#[derive(Clone, Debug, Default)]
pub struct FastWanSubmit {
    pub model: Option<String>,
}

impl SubmitEndpoint for FastWanSubmit {
    type Body = FastWanGenerate;

    fn normalize(
        &self,
        b: FastWanGenerate,
        _cx: &NormalizeCtx,
    ) -> Result<GenerationRequest, ApiError> {
        let model = self
            .model
            .clone()
            .ok_or_else(|| ApiError::invalid_param("model", "no FastWan model is served here"))?;
        if b.prompt.trim().is_empty() {
            return Err(ApiError::invalid_param(
                "prompt",
                "prompt must not be empty",
            ));
        }
        let mut req = GenerationRequest::text(ProtocolId::FastWan, model, b.prompt);
        req.task = Task::T2V;
        req.canvas = match (b.width, b.height) {
            (Some(width), Some(height)) if width > 0 && height > 0 => {
                CanvasSpec::Exact { width, height }
            }
            (None, None) => CanvasSpec::ModelDefault,
            _ => {
                return Err(ApiError::invalid_param(
                    "width",
                    "width and height must both be positive",
                ))
            }
        };
        let length = match b.num_frames {
            Some(n) => Length::Frames {
                value: n,
                snap: Snap::Exact,
            },
            None => Length::ModelDefault,
        };
        if b.fps == Some(0) {
            return Err(ApiError::invalid_param("fps", "fps must be positive"));
        }
        req.timing = TimingSpec { length, fps: b.fps };
        req.seed = match b.seed {
            Some(s) if s < 0 => {
                return Err(ApiError::invalid_param("seed", "seed must be non-negative"))
            }
            s => s.map(|s| s as u64),
        };
        Ok(req)
    }

    fn submit_reply(&self, job: &Job, _cx: &ViewCtx) -> HttpReply {
        HttpReply::json(
            200,
            json!({ "prompt_id": job.external_id, "status": status_name(job.status()) }),
        )
    }
}

/// FastWan status names: `Running` is `processing`; `Failed`/`Cancelled` are
/// `failed` (`fastwan_link.py:92`).
pub fn status_name(s: JobStatus) -> &'static str {
    match s {
        JobStatus::Queued => "queued",
        JobStatus::Running => "processing",
        JobStatus::Succeeded => "completed",
        JobStatus::Failed | JobStatus::Cancelled => "failed",
    }
}

/// `GET /status/{prompt_id}` body: `{status, error?}` (plus `metadata` with
/// the tier and recipe when known; the client ignores extra keys).
pub fn status_body(job: &Job) -> Value {
    let mut m = Map::new();
    m.insert("status".into(), status_name(job.status()).into());
    match &job.state {
        JobState::Failed(e) => {
            m.insert("error".into(), e.message.clone().into());
        }
        JobState::Cancelled => {
            m.insert("error".into(), "the generation was cancelled".into());
        }
        _ => {}
    }
    if let Some(md) = metadata(job) {
        m.insert("metadata".into(), md);
    }
    Value::Object(m)
}

/// `/status` and `/video` rendering.
#[derive(Clone, Copy, Debug, Default)]
pub struct FastWanView;

impl JobView for FastWanView {
    fn status_reply(&self, job: &Job, _cx: &ViewCtx) -> HttpReply {
        HttpReply::json(200, status_body(job))
    }
    /// The MP4 when completed; 422 (rejected) when failed; 404 while pending.
    fn result_reply(&self, job: &Job, cx: &ViewCtx) -> HttpReply {
        match &job.state {
            JobState::Succeeded => artifact_reply(job, cx).unwrap_or_else(|| {
                fastapi_error_status(404, &ApiError::not_found("video file not found"))
            }),
            JobState::Failed(e) => {
                fastapi_error_status(422, &ApiError::engine_failed(e.message.clone()))
            }
            JobState::Cancelled => {
                fastapi_error_status(422, &ApiError::cancelled("the generation was cancelled"))
            }
            _ => fastapi_error_status(404, &ApiError::not_found("the video is not ready yet")),
        }
    }
}

/// The model FastWan jobs run: the configured one, else the first Wan model
/// that is not causal-only, else the first Wan model.
pub fn fastwan_model(models: &[ModelCaps], configured: Option<&str>) -> Option<String> {
    if let Some(c) = configured {
        return Some(c.to_owned());
    }
    let wan: Vec<&ModelCaps> = models.iter().filter(|m| m.family == Family::Wan).collect();
    wan.iter()
        .find(|m| !matches!(m.stream, Some(StreamCaps::Causal { .. })))
        .or_else(|| wan.first())
        .map(|m| {
            m.served_names
                .first()
                .cloned()
                .unwrap_or_else(|| m.id.0.clone())
        })
}

/// `GET /health` body: `{"status","model_loaded"}` and its status (503 while
/// the engine is not ready). The binary merges this into its `/health`.
pub fn health_body(engine: &dyn EngineGate) -> (u16, Value) {
    let loaded = engine.admit().is_ok() && !engine.models().is_empty();
    let status = if loaded { "ok" } else { "loading" };
    (
        if loaded { 200 } else { 503 },
        json!({ "status": status, "model_loaded": loaded }),
    )
}

/// `GET /` body: `{"model": <served name>}`.
pub fn root_body(model: Option<&str>) -> Value {
    json!({ "model": model })
}

/// `GET /health` and `GET /` for a standalone FastWan server. `fv-serve`
/// owns both paths (design §9) and merges [`health_body`] / [`root_body`].
pub fn service_routes(ctx: &ServeCtx, cfg: &VideosConfig) -> Router<ServeCtx> {
    let model = fastwan_model(&ctx.engine().models(), cfg.fastwan_model.as_deref());
    Router::new()
        .route(
            "/health",
            get(|State(ctx): State<ServeCtx>| async move {
                let (s, v) = health_body(ctx.engine().as_ref());
                into_response(HttpReply::json(s, v), &ctx, None).await
            }),
        )
        .route(
            "/",
            get(move |State(ctx): State<ServeCtx>| async move {
                into_response(
                    HttpReply::json(200, root_body(model.as_deref())),
                    &ctx,
                    None,
                )
                .await
            }),
        )
}

async fn delete(ctx: ServeCtx, id: String, headers: HeaderMap) -> Response {
    let r = async {
        let owner = ctx.auth().authenticate(ProtocolId::FastWan, &headers)?;
        let job = find_job(&ctx, &FastWanApi, &id, owner.as_ref()).await?;
        delete_job(&ctx, &job).await;
        Ok::<_, ApiError>(HttpReply::json(200, json!({})))
    }
    .await;
    into_response(r.unwrap_or_else(|e| fastapi_error(&e)), &ctx, None).await
}

/// `/generate`, `/status/{prompt_id}`, `/video/{prompt_id}` over the
/// serve-kit generic handlers.
pub fn routes(ctx: &ServeCtx, cfg: &VideosConfig) -> Router<ServeCtx> {
    let model = fastwan_model(&ctx.engine().models(), cfg.fastwan_model.as_deref());
    let (proto, view) = (Arc::new(FastWanApi), Arc::new(FastWanView));
    let endpoint = Arc::new(FastWanSubmit { model });
    let mut opts = SubmitOpts::new(cfg.ingest.clone());
    opts.body_max = ctx.config().body_max_bytes;
    Router::new()
        .route(
            "/generate",
            handlers::submit(proto.clone(), endpoint, view.clone(), opts),
        )
        .route(
            "/status/{prompt_id}",
            handlers::status(proto.clone(), view.clone(), "prompt_id"),
        )
        .route(
            "/video/{prompt_id}",
            handlers::result(proto, view, "prompt_id").delete(
                |State(ctx): State<ServeCtx>, Path(id): Path<String>, headers: HeaderMap| {
                    delete(ctx, id, headers)
                },
            ),
        )
}
