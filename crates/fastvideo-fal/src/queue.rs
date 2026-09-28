//! Queue submit, status, result, cancel and the SSE status stream
//! (design §4.4; fal §9).
//!
//! Every route exists under both path forms (fal §9.1): the app-only form
//! `/{owner}/{alias}/requests/{id}[…]` that `@fal-ai/client` builds and the
//! submit response advertises, and the full endpoint form
//! `/{owner}/{alias}/{sub}/requests/{id}[…]` the OpenAPI lists. The result
//! also answers at `…/requests/{id}/response` (D-QUEUE's example URL).
//! Request ids are unique, so either form finds the job, as long as it was
//! submitted under the same app.
//!
//! **Cancel answers 202** `{"status":"CANCELLATION_REQUESTED"}` (D-QUEUE's
//! table), not the OpenAPI's `200 {"success": bool}`: the docs table is the
//! newer and more specific contract, and every client accepts any 2xx (JS
//! checks `ok`, Python `raise_for_status`, the user's FL client ignores the
//! body). A queued job is cancelled at once (its status turns `COMPLETED`
//! with `error_type: "client_cancelled"`); a running one gets its engine
//! cancel token tripped and ends at the next denoise step (it "may still
//! complete if mid-processing", D-QUEUE). A finished job answers 400
//! `{"status":"ALREADY_COMPLETED"}`, an unknown id 404 `{"status":"NOT_FOUND"}`.

use std::sync::Arc;
use std::time::Duration;

use axum::body::Bytes;
use axum::extract::{Path as UrlPath, Query, State};
use axum::http::HeaderMap;
use axum::response::Response;
use base64::Engine as _;
use fastvideo_protocol::{
    ApiError, ArtifactLocation, BatchProtocol, ErrorCtx, ErrorKind, GenerationRequest, HttpReply,
    Job, JobState, JobStatus, JobView, KeyId, ListQuery, LogLevel, NormalizeCtx, ProtocolId,
    ReplyBody, SseEvent, SseFollow, SseSpec, SubmitEndpoint, ViewCtx,
};
use fastvideo_serve_kit::events::cancel_job;
use fastvideo_serve_kit::handlers::{error_reply, find_job, submit_request};
use fastvideo_serve_kit::{into_response, ServeCtx};
use serde_json::{json, Value};
use time::OffsetDateTime;

use crate::error::{error_body, error_message, not_found_request, FalProtocol};
use crate::schema::{fal_param, wants_inline, Endpoint, FalInput, File, VideoOutput};
use crate::{FalApp, FalConfig};

// ---------------------------------------------------------------- views

/// The app id a job was submitted under (`request_echo.model` is the full
/// endpoint id, e.g. `minimax/h3-max/text-to-video` or
/// `lightricks/ltx-2.5/text-to-video/fast`).
pub fn app_of(job: &Job) -> &str {
    crate::schema::app_id(job.requested_model())
}

/// `(response_url, status_url, cancel_url)`: app-only, no `/response`
/// suffix (fal §9.1 recommendation).
pub fn request_urls(app: &str, request_id: &str, cx: &ViewCtx) -> (String, String, String) {
    let base = cx.public_url(&format!("{app}/requests/{request_id}")).to_string();
    (base.clone(), format!("{base}/status"), format!("{base}/cancel"))
}

/// A timestamp as fal writes it: `2026-02-17T10:30:01.123Z`.
pub fn fal_timestamp(t: OffsetDateTime) -> String {
    let t = t.to_offset(time::UtcOffset::UTC);
    format!(
        "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}.{:03}Z",
        t.year(),
        t.month() as u8,
        t.day(),
        t.hour(),
        t.minute(),
        t.second(),
        t.millisecond()
    )
}

/// One log entry: `{message, level, source:"USER", timestamp}` (JS
/// `RequestLog`, fal §9.3).
pub fn log_entry(l: &fastvideo_protocol::LogLine) -> Value {
    let level = match l.level {
        LogLevel::Debug => "DEBUG",
        LogLevel::Info => "INFO",
        LogLevel::Warn => "WARN",
        LogLevel::Error => "ERROR",
    };
    json!({"message": l.message, "level": level, "source": "USER", "timestamp": fal_timestamp(l.timestamp)})
}

/// The queue status body (fal §9.3). Invariants the clients rely on:
/// `IN_QUEUE` always has `queue_position`; `IN_PROGRESS` and `COMPLETED`
/// always have `logs` (`[]` unless `?logs=1`); every body has `request_id`,
/// `response_url`, `status_url` and `cancel_url`. Only the three standard
/// status strings are ever used (Python raises on anything else).
pub fn status_json(job: &Job, cx: &ViewCtx) -> Value {
    let rid = job.external_id.as_str();
    let (response_url, status_url, cancel_url) = request_urls(app_of(job), rid, cx);
    let mut v = json!({
        "request_id": rid,
        "response_url": response_url,
        "status_url": status_url,
        "cancel_url": cancel_url,
    });
    let logs = || -> Value {
        if cx.with_logs {
            Value::Array(job.logs.iter().map(log_entry).collect())
        } else {
            Value::Array(Vec::new())
        }
    };
    match &job.state {
        JobState::Queued => {
            v["status"] = "IN_QUEUE".into();
            v["queue_position"] = job.queue_position.unwrap_or(0).into();
        }
        JobState::Running => {
            v["status"] = "IN_PROGRESS".into();
            v["logs"] = logs();
        }
        state => {
            v["status"] = "COMPLETED".into();
            v["logs"] = logs();
            let mut metrics = serde_json::Map::new();
            if let Some(s) = job.metrics.inference_s {
                metrics.insert("inference_time".into(), s.into());
            }
            v["metrics"] = Value::Object(metrics);
            match state {
                JobState::Failed(e) => {
                    v["error"] = error_message(e).into();
                    v["error_type"] = error_body(e).1.into();
                }
                JobState::Cancelled => v["error_type"] = "client_cancelled".into(),
                _ => {}
            }
        }
    }
    v
}

/// The bare output JSON of a succeeded job (fal §3.4, §5.2), or `None`.
pub fn output_json(job: &Job, cx: &ViewCtx, url_ttl: Duration) -> Option<Value> {
    if job.status() != JobStatus::Succeeded {
        return None;
    }
    let a = job.artifacts.first()?;
    let out = VideoOutput {
        video: File {
            url: cx.urls.url_for(a, url_ttl).to_string(),
            content_type: Some(a.mime.clone()),
            file_name: Some(a.file_name.clone()),
            file_size: Some(a.bytes),
        },
        expanded_prompt: None,
        // The effective seed (the request's, or the one the server drew),
        // on every task: fal's r2v schema requires it, and on t2v/i2v it
        // is an extra key clients ignore but callers need to reproduce a
        // clip.
        seed: Some(job.resolved.seed),
        timings: timings(job),
    };
    serde_json::to_value(out).ok()
}

/// The output `timings` (fal: `object<string, number>`, "'inference' is the
/// DiT denoising time"). `inference` keeps fal's meaning (denoise only);
/// the other keys are our breakdown, all in seconds:
///
/// - one key per engine stage, named as in `X-Stage-Durations` (H3:
///   `text`, `refine`, `denoise`, `audio_decode`, `video_decode`, `encode`;
///   `text` includes the I2V multimodal text encoder);
/// - `queue`: submit to start; `total`: start to completion (the whole
///   engine run, so `total - inference` is the time outside the denoise).
///
/// `None` when nothing was measured (fal: "Null on routes that do not
/// report backend timings").
pub fn timings(job: &Job) -> Option<serde_json::Map<String, Value>> {
    let mut m = serde_json::Map::new();
    let secs = |d: time::Duration| Value::from(d.as_seconds_f64().max(0.0));
    if let Some(s) = job.metrics.inference_s {
        m.insert("inference".into(), s.into());
    }
    for (stage, s) in &job.metrics.stage_durations {
        m.entry(stage.clone()).or_insert_with(|| (*s).into());
    }
    if m.is_empty() {
        return None;
    }
    if let Some(start) = job.started_at {
        m.entry("queue").or_insert_with(|| secs(start - job.created_at));
        if let Some(end) = job.completed_at {
            m.entry("total").or_insert_with(|| secs(end - start));
        }
    }
    Some(m)
}

/// Headers that carry our tier/recipe metadata (design §0.3, §0.6: the fal
/// output schema has no metadata field, so it goes in headers).
fn meta_headers(job: &Job, r: &mut HttpReply) {
    if let Some(t) = job.resolved.tier {
        r.push_header("x-fv-tier", t.as_str());
        if !t.passes_quality_gate() {
            r.push_header("x-fv-quality", "draft");
        }
    }
    if let Some(rec) = &job.resolved.recipe {
        r.push_header("x-fv-recipe", rec.clone());
    }
}

/// Status and result rendering for fal jobs (pure).
#[derive(Clone, Debug)]
pub struct FalView {
    /// Lifetime of the signed output URL.
    pub url_ttl: Duration,
}

impl Default for FalView {
    fn default() -> Self {
        Self { url_ttl: Duration::from_secs(24 * 3600) }
    }
}

impl JobView for FalView {
    fn status_reply(&self, job: &Job, cx: &ViewCtx) -> HttpReply {
        HttpReply::json(200, status_json(job, cx)).with_header("x-fal-request-id", job.external_id.clone())
    }

    /// Succeeded: 200 bare output. Queued/running: 400
    /// `{"detail":"Request is still in progress"}`. Failed: the error's own
    /// status and `detail` (fal §13 recommendation). Cancelled: 499
    /// `client_cancelled`.
    fn result_reply(&self, job: &Job, cx: &ViewCtx) -> HttpReply {
        let rid = job.external_id.clone();
        let ecx = ErrorCtx { request_id: Some(rid.clone()), route: None, external_id: Some(rid.clone()) };
        let mut r = match &job.state {
            JobState::Queued | JobState::Running => HttpReply::json(400, json!({"detail": "Request is still in progress"}))
                .with_header("x-fal-error-type", "bad_request")
                .with_header("x-fal-request-id", rid),
            JobState::Succeeded => match output_json(job, cx, self.url_ttl) {
                Some(v) => HttpReply::json(200, v).with_header("x-fal-request-id", rid),
                None => FalProtocol.render_error(&ApiError::internal("the job has no output"), &ecx),
            },
            JobState::Failed(e) => FalProtocol.render_error(e, &ecx),
            JobState::Cancelled => FalProtocol.render_error(&ApiError::cancelled("Request was cancelled"), &ecx),
        };
        meta_headers(job, &mut r);
        r
    }
}

/// A view that renders status with or without logs regardless of `cx`
/// (the SSE follower renders with `with_logs: false`).
struct LogsView {
    inner: Arc<FalView>,
    with_logs: bool,
}

impl JobView for LogsView {
    fn status_reply(&self, job: &Job, cx: &ViewCtx) -> HttpReply {
        let mut cx = *cx;
        cx.with_logs = self.with_logs;
        self.inner.status_reply(job, &cx)
    }
    fn result_reply(&self, job: &Job, cx: &ViewCtx) -> HttpReply {
        self.inner.result_reply(job, cx)
    }
}

/// Replaces `video.url` with a `data:video/mp4;base64,…` URI for
/// `sync_mode: true` (fal §3.3, INFERRED shape). Only local artifacts up to
/// `max_bytes` are inlined; others keep their URL.
pub async fn inline_video(mut reply: HttpReply, job: &Job, max_bytes: u64) -> HttpReply {
    if reply.status != 200 || !wants_inline(job) {
        return reply;
    }
    let Some(a) = job.artifacts.first() else { return reply };
    let ArtifactLocation::Local(path) = &a.location else { return reply };
    if a.bytes > max_bytes {
        return reply;
    }
    let Ok(bytes) = tokio::fs::read(path).await else { return reply };
    if let ReplyBody::Json(v) = &mut reply.body {
        let b64 = base64::engine::general_purpose::STANDARD.encode(&bytes);
        v["video"]["url"] = format!("data:{};base64,{b64}", a.mime).into();
    }
    reply
}

// ---------------------------------------------------------------- submit

/// One submit endpoint of one app (pure parts, for golden tests).
#[derive(Clone, Debug)]
pub struct FalEndpoint {
    pub app: FalApp,
    pub endpoint: Endpoint,
}

impl FalEndpoint {
    /// `minimax/h3-max/text-to-video`.
    pub fn endpoint_id(&self) -> String {
        format!("{}/{}", self.app.id, self.endpoint.sub())
    }
}

impl SubmitEndpoint for FalEndpoint {
    type Body = Value;
    fn normalize(&self, body: Value, cx: &NormalizeCtx) -> Result<GenerationRequest, ApiError> {
        FalInput::parse_for(self.app.kind(), self.endpoint, &body)?.normalize(&self.app.target(self.endpoint).0, cx)
    }
    /// HTTP 200 `{request_id, response_url, status_url, cancel_url,
    /// queue_position}` with app-only URLs (design §4.4), plus
    /// `x-fal-request-id`.
    fn submit_reply(&self, job: &Job, cx: &ViewCtx) -> HttpReply {
        let rid = job.external_id.as_str();
        let (response_url, status_url, cancel_url) = request_urls(&self.app.id, rid, cx);
        HttpReply::json(
            200,
            json!({
                "request_id": rid,
                "response_url": response_url,
                "status_url": status_url,
                "cancel_url": cancel_url,
                "queue_position": job.queue_position.unwrap_or(0),
            }),
        )
        .with_header("x-fal-request-id", rid.to_owned())
    }
}

/// Fal jobs of `app` that are still waiting.
async fn queued_for(ctx: &ServeCtx, app: &str) -> usize {
    let q = ListQuery {
        protocol: Some(ProtocolId::Fal),
        statuses: vec![JobStatus::Queued],
        limit: usize::MAX,
        ..Default::default()
    };
    ctx.jobs().list(q).await.items.iter().filter(|j| app_of(j) == app).count()
}

/// Resolves an endpoint's model: its name (the app's model, or the LTX /
/// Wan endpoint's tier alias) through the engine's aliases and served
/// names, else its tier (design §0.3).
pub(crate) fn resolve_app_model(ctx: &ServeCtx, app: &FalApp, endpoint: Endpoint) -> Result<String, ApiError> {
    let (name, tier) = app.target(endpoint);
    let models = ctx.engine().models();
    let engine = ctx.engine().clone();
    let by_name = fastvideo_protocol::resolve_model(&name, |n| engine.alias(n), &models).map(|c| c.id.0.clone());
    by_name
        .or_else(|e| match tier {
            Some((family, tier)) => fastvideo_protocol::resolve_tier(family, tier, &models).map(|c| c.id.0.clone()),
            None => Err(e),
        })
        .map_err(|_| match endpoint.target() {
            Some(_) => ApiError::not_found(format!(
                "Application \"{}/{}\" not found (no `{name}` model is served here)",
                app.id,
                endpoint.sub()
            )),
            None => ApiError::not_found(format!("Application \"{}\" not found", app.id)),
        })
}

/// Auth, validation, normalization, queue limit, then serve-kit's submit
/// pipeline. Errors name fal fields.
pub(crate) async fn submit_job(
    ctx: &ServeCtx,
    cfg: &FalConfig,
    ep: &FalEndpoint,
    query: Vec<(String, String)>,
    headers: &HeaderMap,
    body: &Bytes,
) -> Result<Job, ApiError> {
    let owner: Option<KeyId> = ctx.auth().authenticate(ProtocolId::Fal, headers)?;
    let mut echo: Value = if body.is_empty() {
        Value::Object(Default::default())
    } else {
        serde_json::from_slice(body).map_err(|e| ApiError::invalid(format!("JSON decode error: {e}")))?
    };
    let mut ncx = NormalizeCtx::new(ctx.now());
    ncx.owner = owner.clone();
    ncx.query = query;
    ncx.headers = headers
        .iter()
        .filter_map(|(k, v)| Some((k.as_str().to_owned(), v.to_str().ok()?.to_owned())))
        .collect();
    let mut req = ep.normalize(echo.clone(), &ncx)?;
    if let Some(max) = ncx.query_param("fal_max_queue_length") {
        let max: usize = max
            .trim()
            .parse()
            .map_err(|_| ApiError::invalid_param("fal_max_queue_length", "fal_max_queue_length must be a non-negative integer"))?;
        // "Reject the request with 429 if the endpoint's queue already has
        // more than this many requests waiting" (D-HDR).
        let waiting = queued_for(ctx, &ep.app.id).await;
        if waiting > max {
            return Err(ApiError::queue_full(format!(
                "The queue has {waiting} requests waiting, more than fal_max_queue_length={max}"
            )));
        }
    }
    crate::storage::rewrite_own_uploads(ctx, &mut req);
    req.model = resolve_app_model(ctx, &ep.app, ep.endpoint)?;
    // The H3 schema's omitted `resolution` (768P) follows the model's tiers;
    // the LTX and Wan schemas have their own defaults.
    if echo.get("resolution").is_none() && ep.endpoint.target().is_none() {
        if let Some(caps) = ctx.engine().models().into_iter().find(|m| m.id.0 == req.model) {
            crate::schema::default_resolution_for(&mut req, &caps.canvas.short_edges);
        }
    }
    if let Value::Object(m) = &mut echo {
        m.insert("model".into(), ep.endpoint_id().into());
    }
    let snapshot = req.clone();
    submit_request(ctx, &FalProtocol, req, owner, echo, &cfg.ingest)
        .await
        .map_err(|mut e| {
            if let Some(p) = e.param.take() {
                e.param = Some(fal_param(&p, &snapshot));
            }
            e
        })
}

fn fresh_request_id() -> String {
    uuid::Uuid::new_v4().to_string()
}

/// `POST /{app}/{sub}`.
pub(crate) async fn submit_handler(
    ctx: ServeCtx,
    cfg: Arc<FalConfig>,
    ep: Arc<FalEndpoint>,
    query: Vec<(String, String)>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    let reply = match submit_job(&ctx, &cfg, &ep, query, &headers, &body).await {
        Ok(job) => ep.submit_reply(&job, &ctx.view_ctx(false)),
        Err(e) => {
            let ecx = ErrorCtx { request_id: Some(fresh_request_id()), route: Some(format!("/{}", ep.endpoint_id())), external_id: None };
            error_reply(&FalProtocol, &e, &ecx)
        }
    };
    into_response(reply, &ctx, None).await
}

// ---------------------------------------------------------------- lookups

/// Auth, then the job by request id under `app`; errors are ready replies.
pub(crate) async fn lookup(ctx: &ServeCtx, app: &str, id: &str, headers: &HeaderMap) -> Result<Job, HttpReply> {
    let ecx = ErrorCtx { request_id: None, route: None, external_id: Some(id.to_owned()) };
    let owner = ctx
        .auth()
        .authenticate(ProtocolId::Fal, headers)
        .map_err(|e| error_reply(&FalProtocol, &e, &ecx))?;
    match find_job(ctx, &FalProtocol, id, owner.as_ref()).await {
        Ok(j) if app_of(&j) == app => Ok(j),
        Ok(_) => Err(not_found_request(None)),
        Err(e) if e.kind == ErrorKind::NotFound => Err(not_found_request(None)),
        Err(e) => Err(error_reply(&FalProtocol, &e, &ecx)),
    }
}

/// `?logs=1|0|true|false` (JS sends `1`/`0`, Python an httpx bool).
pub fn logs_param(query: &[(String, String)]) -> bool {
    query
        .iter()
        .rev()
        .find(|(k, _)| k == "logs")
        .is_some_and(|(_, v)| matches!(v.trim().to_ascii_lowercase().as_str(), "1" | "true"))
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum Op {
    Status,
    Stream,
    Result,
    Cancel,
}

pub(crate) async fn request_handler(
    ctx: ServeCtx,
    cfg: Arc<FalConfig>,
    app: Arc<str>,
    op: Op,
    id: String,
    query: Vec<(String, String)>,
    headers: HeaderMap,
) -> Response {
    let view = Arc::new(FalView { url_ttl: cfg.url_ttl });
    let job = match lookup(&ctx, &app, &id, &headers).await {
        Ok(j) => j,
        Err(r) => return into_response(r, &ctx, None).await,
    };
    let with_logs = logs_param(&query);
    match op {
        Op::Status => into_response(view.status_reply(&job, &ctx.view_ctx(with_logs)), &ctx, None).await,
        Op::Result => {
            let r = view.result_reply(&job, &ctx.view_ctx(false));
            let r = inline_video(r, &job, cfg.inline_max_bytes).await;
            into_response(r, &ctx, None).await
        }
        Op::Stream => {
            let initial = view.status_reply(&job, &ctx.view_ctx(with_logs));
            let data = match initial.body {
                ReplyBody::Json(v) => v.to_string(),
                _ => String::new(),
            };
            let spec = SseSpec {
                initial: vec![SseEvent::data(data)],
                follow: (!job.is_terminal()).then_some(SseFollow::JobStatus { job: job.id, close_on_terminal: true, with_logs }),
                keepalive: Some(Duration::from_secs(15)),
            };
            let reply = HttpReply::sse(spec).with_header("x-fal-request-id", job.external_id.clone());
            let v: Arc<dyn JobView> = Arc::new(LogsView { inner: view, with_logs });
            into_response(reply, &ctx, Some(v)).await
        }
        Op::Cancel => {
            let rid = job.external_id.clone();
            let reply = if job.is_terminal() {
                already_completed(&rid)
            } else {
                match cancel_job(&ctx, job.id).await {
                    Ok(_) => HttpReply::json(202, json!({"status": "CANCELLATION_REQUESTED"})).with_header("x-fal-request-id", rid),
                    Err(e) if e.kind == ErrorKind::AlreadyCompleted => already_completed(&rid),
                    Err(e) if e.kind == ErrorKind::NotFound => not_found_request(Some(&rid)),
                    Err(e) => error_reply(
                        &FalProtocol,
                        &e,
                        &ErrorCtx { request_id: Some(rid.clone()), route: None, external_id: Some(rid) },
                    ),
                }
            };
            into_response(reply, &ctx, None).await
        }
    }
}

fn already_completed(rid: &str) -> HttpReply {
    HttpReply::json(400, json!({"status": "ALREADY_COMPLETED"}))
        .with_header("x-fal-error-type", "bad_request")
        .with_header("x-fal-request-id", rid.to_owned())
}

/// Adds the queue routes of one app to `router`.
pub(crate) fn app_routes(
    mut router: axum::Router<ServeCtx>,
    cfg: &Arc<FalConfig>,
    app: &FalApp,
) -> axum::Router<ServeCtx> {
    use axum::routing::{get, post, put};
    let body_max = cfg.body_max;
    for &endpoint in app.endpoints() {
        let ep = Arc::new(FalEndpoint { app: app.clone(), endpoint });
        let path = format!("/{}", ep.endpoint_id());
        let (c, e) = (cfg.clone(), ep.clone());
        router = router.route(
            &path,
            post(move |State(ctx): State<ServeCtx>, Query(q): Query<Vec<(String, String)>>, headers: HeaderMap, body: Bytes| {
                submit_handler(ctx, c.clone(), e.clone(), q, headers, body)
            })
            .layer(axum::extract::DefaultBodyLimit::max(body_max)),
        );
    }
    let app_id: Arc<str> = Arc::from(app.id.as_str());
    let prefixes: Vec<String> = std::iter::once(format!("/{}", app.id))
        .chain(app.endpoints().iter().map(|e| format!("/{}/{}", app.id, e.sub())))
        .collect();
    for prefix in prefixes {
        let mk = |op: Op| {
            let (c, a) = (cfg.clone(), app_id.clone());
            move |State(ctx): State<ServeCtx>, UrlPath(id): UrlPath<String>, Query(q): Query<Vec<(String, String)>>, headers: HeaderMap| {
                request_handler(ctx, c.clone(), a.clone(), op, id, q, headers)
            }
        };
        router = router
            .route(&format!("{prefix}/requests/{{id}}"), get(mk(Op::Result)))
            .route(&format!("{prefix}/requests/{{id}}/response"), get(mk(Op::Result)))
            .route(&format!("{prefix}/requests/{{id}}/status"), get(mk(Op::Status)))
            .route(&format!("{prefix}/requests/{{id}}/status/stream"), get(mk(Op::Stream)))
            .route(&format!("{prefix}/requests/{{id}}/cancel"), put(mk(Op::Cancel)));
    }
    router
}
