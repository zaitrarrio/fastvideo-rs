//! `HttpReply` -> axum, and the generic handlers (design §3.5): `submit`
//! (auth -> ingestion -> `negotiate` -> `engine.submit` -> `JobStore`),
//! `status` and `result`.
//!
//! Adapters mount them per route:
//!
//! ```ignore
//! Router::new()
//!     .route("/v2/text-to-video", handlers::submit(proto.clone(), T2v, view.clone(), SubmitOpts::new(IngestPolicy::ltx())))
//!     .route("/v2/text-to-video/{id}", handlers::status(proto.clone(), view.clone(), "id"))
//! ```
//!
//! Every error goes through the adapter's `BatchProtocol::render_error`.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use axum::body::{Body, Bytes};
use axum::extract::{DefaultBodyLimit, MatchedPath, Path as UrlPath, Query, State};
use axum::http::{header, HeaderMap, HeaderName, HeaderValue, Request, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post, MethodRouter};
use fastvideo_protocol::{
    negotiate_noted, precheck, resolve_model, ApiError, BatchProtocol, ErrorCtx, GenerationRequest,
    HttpReply, Job, JobId, JobView, KeyId, MediaKind, MediaRef, NormalizeCtx, ReplyBody, SubmitEndpoint,
};
use tower_http::services::ServeFile;

use crate::ctx::ServeCtx;
use crate::events::wait_terminal;
use crate::ingest::IngestPolicy;

/// Converts a framework-free reply. `view` renders `SseFollow::JobStatus`.
pub async fn into_response(reply: HttpReply, ctx: &ServeCtx, view: Option<Arc<dyn JobView>>) -> Response {
    let status = StatusCode::from_u16(reply.status).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
    let mut resp = match reply.body {
        ReplyBody::Json(v) => (status, axum::Json(v)).into_response(),
        ReplyBody::Bytes { mime, data } => {
            let mut r = (status, data).into_response();
            set(&mut r, header::CONTENT_TYPE, &mime);
            r
        }
        ReplyBody::File { path, mime } => {
            let req = Request::get("/").body(Body::empty()).expect("static request");
            let mut r = match tower::ServiceExt::oneshot(ServeFile::new(&path), req).await {
                Ok(r) => r.map(Body::new).into_response(),
                Err(e) => match e {},
            };
            if r.status().is_success() {
                *r.status_mut() = status;
                set(&mut r, header::CONTENT_TYPE, &mime);
            }
            r
        }
        ReplyBody::Sse(spec) => {
            let mut r = crate::sse::sse_response(ctx.clone(), spec, view);
            *r.status_mut() = status;
            r
        }
        ReplyBody::Empty => status.into_response(),
    };
    for (k, v) in reply.headers {
        if let (Ok(k), Ok(v)) = (HeaderName::from_bytes(k.as_bytes()), HeaderValue::from_str(&v)) {
            resp.headers_mut().append(k, v);
        }
    }
    resp
}

fn set(r: &mut Response, k: HeaderName, v: &str) {
    if let Ok(v) = HeaderValue::from_str(v) {
        r.headers_mut().insert(k, v);
    }
}

/// Renders `err` through the adapter, adding `Retry-After` when set.
pub fn error_reply<P: BatchProtocol + ?Sized>(proto: &P, err: &ApiError, cx: &ErrorCtx) -> HttpReply {
    let mut r = proto.render_error(err, cx);
    if let Some(s) = err.retry_after_s {
        if r.header("retry-after").is_none() {
            r.push_header("retry-after", s.to_string());
        }
    }
    r
}

/// Options for [`submit`].
#[derive(Clone, Debug)]
pub struct SubmitOpts {
    pub ingest: IngestPolicy,
    /// Sync endpoint: wait for the job and answer `JobView::result_reply`
    /// (LTX `/v1/*`, fal `/run`, FastVideo `/v1/videos/sync`).
    pub wait: Option<Duration>,
    /// Request body cap.
    pub body_max: usize,
    /// Header whose value becomes the request id (e.g. `x-fal-request-id`
    /// echoes); default a fresh 32-hex id.
    pub request_id_header: Option<&'static str>,
}

impl SubmitOpts {
    pub fn new(ingest: IngestPolicy) -> Self {
        Self { ingest, wait: None, body_max: 64 * 1024 * 1024, request_id_header: None }
    }
    pub fn sync(mut self, timeout: Duration) -> Self {
        self.wait = Some(timeout);
        self
    }
}

fn normalize_ctx(ctx: &ServeCtx, owner: Option<KeyId>, headers: &HeaderMap, query: Vec<(String, String)>, request_id: String) -> NormalizeCtx {
    let mut n = NormalizeCtx::new(ctx.now());
    n.owner = owner;
    n.query = query;
    n.headers = headers
        .iter()
        .filter_map(|(k, v)| Some((k.as_str().to_owned(), v.to_str().ok()?.to_owned())))
        .collect();
    n.request_id = request_id;
    n
}

/// The submit pipeline after `normalize`: admission, safety, model
/// resolution, precheck, ingestion, `negotiate`, store insert, engine submit.
/// On an engine refusal the job is removed again.
pub async fn submit_request<P: BatchProtocol + ?Sized>(
    ctx: &ServeCtx,
    proto: &P,
    req: GenerationRequest,
    owner: Option<KeyId>,
    request_echo: serde_json::Value,
    policy: &IngestPolicy,
) -> Result<Job, ApiError> {
    // docs/serve/tracing.md: the request's trace (set by the HTTP layer).
    let trace = fastvideo_trace::current();
    let mark = || trace.map(|_| fastvideo_trace::now_ns()).unwrap_or(0);
    let span = |name: &'static str, comp: fastvideo_trace::Comp, since: u64| {
        if let Some(t) = trace {
            t.span_since(comp, name, since, 0);
        }
    };
    let t_validate = mark();
    ctx.engine().admit()?;
    ctx.safety().check_request(&req)?;
    let models = ctx.engine().models();
    let engine = ctx.engine().clone();
    let caps = resolve_model(&req.model, |n| engine.alias(n), &models)?;
    // A tier alias names the tier's base model; a task it does not serve
    // (H3 reference-to-video) goes to the tier's companion for that task.
    let caps = fastvideo_protocol::route_task(caps, req.task, &models);
    precheck(&req, caps)?;
    span("validate", fastvideo_trace::Comp::Adapter, t_validate);
    let id = JobId::new();
    let dir = ctx.inputs_dir(id);
    let t_ingest = mark();
    let resolved = async {
        let staged = ctx.ingestor().stage(&req, policy, &dir, ctx.now()).await?;
        negotiate_noted(&req, caps, &staged).map(|(r, n)| (r, n, passthrough_sources(&req, &staged)))
    }
    .await;
    let (resolved, notes, sources) = match resolved {
        Ok(r) => r,
        Err(e) => {
            let _ = tokio::fs::remove_dir_all(&dir).await;
            return Err(e);
        }
    };
    span("ingest_negotiate", fastvideo_trace::Comp::Adapter, t_ingest);
    let now = ctx.now();
    let pid = proto.id();
    let mut job = Job::new(id, pid, proto.new_external_id(id), resolved, now, ctx.config().retention(pid));
    job.owner = owner;
    job.request_echo = request_echo;
    job.callback = req.callback.clone();
    job.input_sources = sources;
    job.trace = trace.map(|t| t.traceparent(t.parent));
    for n in notes {
        tracing::info!(job = %id, model = %caps.id, "{n}");
        job.logs.push(fastvideo_protocol::LogLine::info(n, now));
    }
    tracing::debug!(job = %id, "job: submit accepted, recording");
    let t_insert = mark();
    if let Err(e) = ctx.jobs().insert(job.clone()).await {
        let _ = tokio::fs::remove_dir_all(&dir).await;
        return Err(e.into());
    }
    span("insert", fastvideo_trace::Comp::Store, t_insert);
    tracing::debug!(job = %id, "job: recorded");
    let t_submit = mark();
    if let Err(e) = ctx.engine().submit(&job).await {
        ctx.jobs().remove(id).await;
        return Err(e);
    }
    span("submit", fastvideo_trace::Comp::Queue, t_submit);
    tracing::debug!(job = %id, "job: submitted to the engine");
    let t_reread = mark();
    let job = ctx.jobs().get(id).await.unwrap_or(job);
    span("get", fastvideo_trace::Comp::Store, t_reread);
    ctx.notify(&job);
    Ok(job)
}

/// Staged inputs a worker may fetch from the client's own URL (see
/// `Job::input_sources`): video and audio given as `http(s)` URLs, stored
/// byte for byte as fetched. Images are left out: ingestion may re-encode
/// them upright (EXIF orientation).
fn passthrough_sources(req: &GenerationRequest, staged: &fastvideo_protocol::StagedInputs) -> Vec<(std::path::PathBuf, String)> {
    let mut out = Vec::new();
    for (r, (kind, m)) in req.references.iter().zip(&staged.references) {
        if let (MediaRef::Http(u), MediaKind::Video | MediaKind::Audio) = (&r.media, kind) {
            out.push((m.path.clone(), u.to_string()));
        }
    }
    if let (Some(a), Some(m)) = (&req.audio_in, &staged.audio_in) {
        if let MediaRef::Http(u) = &a.media {
            out.push((m.path.clone(), u.to_string()));
        }
    }
    out
}

/// Looks a job up by its wire id for `owner` (other owners' jobs are 404).
///
/// A traced request (docs/serve/tracing.md) records the lookup as
/// `store.lookup` with the job's status as its argument (0 queued ..
/// 4 cancelled, -1 unknown), so the trace shows which poll first saw the
/// job terminal.
pub async fn find_job<P: BatchProtocol + ?Sized>(ctx: &ServeCtx, proto: &P, external_id: &str, owner: Option<&KeyId>) -> Result<Job, ApiError> {
    let trace = fastvideo_trace::current();
    let t0 = trace.map(|_| fastvideo_trace::now_ns());
    let found = ctx.jobs().by_external(proto.id(), external_id).await;
    if let (Some(t), Some(s)) = (trace, t0) {
        t.span_since(fastvideo_trace::Comp::Store, "lookup", s, found.as_ref().map_or(-1, status_code));
    }
    match found {
        Some(j) if j.owner.is_none() || j.owner.as_ref() == owner => Ok(j),
        _ => Err(ApiError::not_found(format!("`{external_id}` was not found"))),
    }
}

fn request_id(headers: &HeaderMap, from: Option<&'static str>) -> String {
    from.and_then(|h| headers.get(h))
        .and_then(|v| v.to_str().ok())
        .map(str::to_owned)
        .unwrap_or_else(crate::random_token)
}

/// `POST` handler for one submit endpoint.
pub fn submit<P, E, V>(proto: Arc<P>, endpoint: Arc<E>, view: Arc<V>, opts: SubmitOpts) -> MethodRouter<ServeCtx>
where
    P: BatchProtocol,
    E: SubmitEndpoint,
    V: JobView,
{
    let body_max = opts.body_max;
    let opts = Arc::new(opts);
    post(
        move |State(ctx): State<ServeCtx>,
              matched: Option<MatchedPath>,
              Query(query): Query<Vec<(String, String)>>,
              headers: HeaderMap,
              body: Bytes| {
            let (proto, endpoint, view, opts) = (proto.clone(), endpoint.clone(), view.clone(), opts.clone());
            async move {
                let rid = request_id(&headers, opts.request_id_header);
                let ecx = ErrorCtx {
                    request_id: Some(rid.clone()),
                    route: matched.map(|m| m.as_str().to_owned()),
                    external_id: None,
                };
                let fail = |e: ApiError| error_reply(&*proto, &e, &ecx);
                let trace = fastvideo_trace::current();
                let t_parse = trace.map(|_| fastvideo_trace::now_ns());
                let reply = async {
                    let owner = ctx.auth().authenticate(proto.id(), &headers)?;
                    let echo: serde_json::Value = if body.is_empty() {
                        serde_json::Value::Object(Default::default())
                    } else {
                        serde_json::from_slice(&body)
                            .map_err(|e| ApiError::invalid(format!("invalid JSON body: {e}")))?
                    };
                    let parsed: E::Body = serde_json::from_value(echo.clone())
                        .map_err(|e| ApiError::invalid(e.to_string()))?;
                    let ncx = normalize_ctx(&ctx, owner.clone(), &headers, query, rid.clone());
                    let req = endpoint.normalize(parsed, &ncx)?;
                    if let (Some(t), Some(s)) = (trace, t_parse) {
                        t.span_since(fastvideo_trace::Comp::Adapter, "parse", s, body.len() as i64);
                    }
                    let job = submit_request(&ctx, &*proto, req, owner, echo, &opts.ingest).await?;
                    match opts.wait {
                        None => Ok(endpoint.submit_reply(&job, &ctx.view_ctx(false))),
                        Some(t) => {
                            let j = wait_terminal(&ctx, job.id, t).await.unwrap_or(job);
                            if !j.is_terminal() {
                                let _ = crate::events::cancel_job(&ctx, j.id).await;
                                return Err(ApiError::timeout("generation did not finish in time"));
                            }
                            Ok(view.result_reply(&j, &ctx.view_ctx(false)))
                        }
                    }
                }
                .await
                .unwrap_or_else(fail);
                let v: Arc<dyn JobView> = view;
                into_response(reply, &ctx, Some(v)).await
            }
        },
    )
    .layer(DefaultBodyLimit::max(body_max))
}

enum Which {
    Status,
    Result,
}

fn lookup<P: BatchProtocol, V: JobView>(proto: Arc<P>, view: Arc<V>, param: &'static str, which: Which) -> MethodRouter<ServeCtx> {
    let which = Arc::new(which);
    get(
        move |State(ctx): State<ServeCtx>,
              matched: Option<MatchedPath>,
              UrlPath(params): UrlPath<HashMap<String, String>>,
              Query(query): Query<Vec<(String, String)>>,
              headers: HeaderMap| {
            let (proto, view, which) = (proto.clone(), view.clone(), which.clone());
            async move {
                let ext = params.get(param).cloned().unwrap_or_default();
                let ecx = ErrorCtx {
                    request_id: Some(crate::random_token()),
                    route: matched.map(|m| m.as_str().to_owned()),
                    external_id: Some(ext.clone()),
                };
                let with_logs = query
                    .iter()
                    .any(|(k, v)| k == "logs" && matches!(v.as_str(), "1" | "true"));
                let reply = async {
                    let owner = ctx.auth().authenticate(proto.id(), &headers)?;
                    let job = find_job(&ctx, &*proto, &ext, owner.as_ref()).await?;
                    let cx = ctx.view_ctx(with_logs);
                    Ok::<_, ApiError>(match *which {
                        Which::Status => view.status_reply(&job, &cx),
                        Which::Result => view.result_reply(&job, &cx),
                    })
                }
                .await
                .unwrap_or_else(|e| error_reply(&*proto, &e, &ecx));
                let v: Arc<dyn JobView> = view;
                into_response(reply, &ctx, Some(v)).await
            }
        },
    )
}

/// A job's status as a trace argument: 0 queued, 1 running, 2 succeeded,
/// 3 failed, 4 cancelled.
pub fn status_code(job: &Job) -> i64 {
    use fastvideo_protocol::JobStatus::*;
    match job.status() {
        Queued => 0,
        Running => 1,
        Succeeded => 2,
        Failed => 3,
        Cancelled => 4,
    }
}

/// `GET` status handler; the job's wire id is the path parameter `param`.
/// `?logs=1|true` sets `ViewCtx::with_logs`.
pub fn status<P: BatchProtocol, V: JobView>(proto: Arc<P>, view: Arc<V>, param: &'static str) -> MethodRouter<ServeCtx> {
    lookup(proto, view, param, Which::Status)
}

/// `GET` result handler (see [`status`]).
pub fn result<P: BatchProtocol, V: JobView>(proto: Arc<P>, view: Arc<V>, param: &'static str) -> MethodRouter<ServeCtx> {
    lookup(proto, view, param, Which::Result)
}

#[cfg(test)]
mod tests;
