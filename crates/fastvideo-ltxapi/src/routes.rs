//! The axum router for every LTX route (design §4.5, §9).
//!
//! | Route | Behaviour |
//! |---|---|
//! | `POST /v2/{text,image}-to-video` | `202 {id, created_at}` |
//! | `GET /v2/{text,image}-to-video/{id}` | `V2JobStatusResponse`; other endpoint's job → 404 |
//! | `POST /v1/{text,image}-to-video` | sync; `200 video/mp4` bytes; over the timeout → 504 |
//! | `POST /v1/upload` | `200 {upload_url, storage_uri, expires_at, required_headers}` |
//! | `POST /v1\|v2/{audio-to-video,retake,extend,video-to-video-hdr,video-to-video-reframe}` | `403 permission_error` |
//! | `GET /v2/{those}/{id}` | `404 not_found_error` |
//!
//! Every reply carries `x-request-id` (32 hex characters). Auth is
//! `Authorization: Bearer <key>` (serve-kit `Auth`, protocols `LtxV1` /
//! `LtxV2`). `PUT /uploads/{token}` and `GET /files/...` are serve-kit's
//! shared routes (`ServeCtx::routes`), mounted by the binary.

use std::sync::Arc;
use std::time::Duration;

use axum::body::Bytes;
use axum::extract::{DefaultBodyLimit, Path as UrlPath, State};
use axum::http::{HeaderMap, HeaderValue};
use axum::response::Response;
use axum::routing::{get, post};
use axum::Router;
use fastvideo_protocol::{
    resolve_model, resolve_tier, ApiError, ErrorCtx, Family, HttpReply, JobView, NormalizeCtx,
    SubmitEndpoint,
};
use fastvideo_serve_kit::events::{cancel_job, wait_terminal};
use fastvideo_serve_kit::handlers::{find_job, into_response, submit_request};
use fastvideo_serve_kit::{IngestPolicy, ServeCtx};
use serde_json::Value;

use crate::error::{self, Api, LtxProtocol};
use crate::models::{parse_tier_alias, LtxModels};
use crate::request::{Endpoint, Submit};
use crate::stubs::{refused, STUB_ENDPOINTS};
use crate::upload::upload_reply;
use crate::v1::ConcurrencyLimit;
use crate::v2::{not_found, V2View};

/// `[ltx]` settings.
#[derive(Clone, Debug)]
pub struct LtxConfig {
    /// `model` ids and their engine targets.
    pub models: LtxModels,
    /// `ltx.sync_timeout` for `/v1/*`; `None` uses `ServeConfig::sync_timeout`.
    pub sync_timeout: Option<Duration>,
    /// Lifetime of `result.video_url`.
    pub url_ttl: Duration,
    /// Concurrent `/v1/*` generations per API key (upstream default 2);
    /// `None` disables the limit.
    pub v1_concurrency: Option<usize>,
    /// Media ingestion limits (ltx §2.0).
    pub ingest: IngestPolicy,
    /// Request body cap (data URIs up to 15 MB encoded).
    pub body_max_bytes: usize,
}

impl Default for LtxConfig {
    fn default() -> Self {
        Self {
            models: LtxModels::default(),
            sync_timeout: None,
            url_ttl: Duration::from_secs(24 * 3600),
            v1_concurrency: Some(2),
            ingest: IngestPolicy::ltx(),
            body_max_bytes: 64 * 1024 * 1024,
        }
    }
}

struct Shared {
    cfg: LtxConfig,
    models: Arc<LtxModels>,
    limit: Option<Arc<ConcurrencyLimit>>,
}

/// Every LTX route, for `Router::merge` into the serve binary.
pub fn router(cfg: LtxConfig) -> Router<ServeCtx> {
    let body_max = cfg.body_max_bytes;
    let st = Arc::new(Shared {
        models: Arc::new(cfg.models.clone()),
        limit: cfg.v1_concurrency.map(ConcurrencyLimit::new),
        cfg,
    });
    let mut r = Router::new();
    for api in [Api::V1, Api::V2] {
        for ep in Endpoint::ALL {
            let path = format!("{}/{}", api.prefix(), ep.segment());
            let route: Arc<str> = path.clone().into();
            let s = st.clone();
            r = r.route(
                &path,
                post(move |State(ctx): State<ServeCtx>, headers: HeaderMap, body: Bytes| {
                    let (s, route) = (s.clone(), route.clone());
                    async move { generate(ctx, s, api, ep, route, headers, body).await }
                }),
            );
            if api == Api::V2 {
                let s = st.clone();
                r = r.route(
                    &format!("{path}/{{id}}"),
                    get(move |State(ctx): State<ServeCtx>, UrlPath(id): UrlPath<String>, headers: HeaderMap| {
                        let s = s.clone();
                        async move { status(ctx, s, Some(ep), id, headers).await }
                    }),
                );
            }
        }
        for stub in STUB_ENDPOINTS {
            let path = format!("{}/{stub}", api.prefix());
            r = r.route(
                &path,
                post(move |State(ctx): State<ServeCtx>, headers: HeaderMap| async move {
                    let rid = fastvideo_serve_kit::random_token();
                    let reply = ctx
                        .auth()
                        .authenticate(api.protocol(), &headers)
                        .and_then(|_| Err::<HttpReply, _>(refused()))
                        .unwrap_or_else(|e| error::render(&e, api, &ecx(&rid, None)));
                    finish(reply, &rid, &ctx).await
                }),
            );
            if api == Api::V2 {
                let s = st.clone();
                r = r.route(
                    &format!("{path}/{{id}}"),
                    get(move |State(ctx): State<ServeCtx>, UrlPath(id): UrlPath<String>, headers: HeaderMap| {
                        let s = s.clone();
                        async move { status(ctx, s, None, id, headers).await }
                    }),
                );
            }
        }
    }
    r.route(
        "/v1/upload",
        post(|State(ctx): State<ServeCtx>, headers: HeaderMap| async move {
            let rid = fastvideo_serve_kit::random_token();
            let reply = async {
                ctx.auth().authenticate(Api::V1.protocol(), &headers)?;
                let t = ctx
                    .uploads()
                    .create(None, None, ctx.now())
                    .map_err(|e| ApiError::internal(format!("creating the upload failed: {e}")))?;
                Ok::<_, ApiError>(upload_reply(&t))
            }
            .await
            .unwrap_or_else(|e| error::render(&e, Api::V1, &ecx(&rid, None)));
            finish(reply, &rid, &ctx).await
        }),
    )
    .layer(DefaultBodyLimit::max(body_max))
    .layer(axum::middleware::map_response(ensure_request_id))
}

fn ecx(rid: &str, external_id: Option<&str>) -> ErrorCtx {
    ErrorCtx {
        request_id: Some(rid.to_owned()),
        route: None,
        external_id: external_id.map(str::to_owned),
    }
}

/// Stamps `rid` as the reply's only `x-request-id` and converts it.
async fn finish(mut reply: HttpReply, rid: &str, ctx: &ServeCtx) -> Response {
    reply.headers.retain(|(k, _)| !k.eq_ignore_ascii_case("x-request-id"));
    reply.push_header("x-request-id", rid);
    into_response(reply, ctx, None).await
}

/// Any response without `x-request-id` (e.g. axum's own 405) gets one.
async fn ensure_request_id(mut res: Response) -> Response {
    if !res.headers().contains_key("x-request-id") {
        if let Ok(v) = HeaderValue::from_str(&fastvideo_serve_kit::random_token()) {
            res.headers_mut().insert("x-request-id", v);
        }
    }
    res
}

/// Binds the normalized model name (a tier alias or an engine id) to a
/// served model; a known LTX id with nothing behind it is `403`.
fn bind_model(ctx: &ServeCtx, name: &str, requested: &str) -> Result<String, ApiError> {
    let engine = ctx.engine().clone();
    let models = engine.models();
    if let Ok(c) = resolve_model(name, |n| engine.alias(n), &models) {
        if c.family == Family::Ltx2 {
            return Ok(c.id.0.clone());
        }
    }
    if let Some(t) = parse_tier_alias(name) {
        if let Ok(c) = resolve_tier(Family::Ltx2, t, &models) {
            return Ok(c.id.0.clone());
        }
    }
    Err(ApiError::forbidden(format!(
        "model `{requested}` is not available for the account"
    ))
    .with_param("model"))
}

async fn generate(
    ctx: ServeCtx,
    st: Arc<Shared>,
    api: Api,
    ep: Endpoint,
    route: Arc<str>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    let rid = fastvideo_serve_kit::random_token();
    let proto = LtxProtocol { api };
    let reply = async {
        let owner = ctx.auth().authenticate(api.protocol(), &headers)?;
        let echo: Value = if body.is_empty() {
            Value::Object(Default::default())
        } else {
            serde_json::from_slice(&body)
                .map_err(|e| ApiError::invalid(format!("invalid JSON body: {e}")))?
        };
        let mut ncx = NormalizeCtx::new(ctx.now());
        ncx.owner = owner.clone();
        ncx.request_id = rid.clone();
        let submit = Submit::new(ep, api, st.models.clone());
        let mut req = submit.normalize(echo.clone(), &ncx)?;
        let requested = echo.get("model").and_then(Value::as_str).unwrap_or_default();
        req.model = bind_model(&ctx, &req.model, requested)?;
        let _slot = match (api, &st.limit) {
            (Api::V1, Some(l)) => Some(l.acquire(owner.as_ref().map(|k| k.0.as_str()).unwrap_or(""))?),
            _ => None,
        };
        let job = submit_request(&ctx, &proto, req, owner, echo, &st.cfg.ingest).await?;
        match api {
            Api::V2 => Ok(submit.submit_reply(&job, &ctx.view_ctx(false))),
            Api::V1 => {
                let timeout = st.cfg.sync_timeout.unwrap_or(ctx.config().sync_timeout);
                let j = wait_terminal(&ctx, job.id, timeout).await.unwrap_or(job);
                if !j.is_terminal() {
                    let _ = cancel_job(&ctx, j.id).await;
                    return Err(ApiError::timeout("generation did not finish in time"));
                }
                Ok(crate::v1::sync_reply(&ctx, &j).await)
            }
        }
    }
    .await
    .unwrap_or_else(|e| {
        let mut cx = ecx(&rid, None);
        cx.route = Some(route.to_string());
        error::render(&e, api, &cx)
    });
    finish(reply, &rid, &ctx).await
}

/// `GET /v2/{endpoint}/{id}`; `ep` is `None` for the refused endpoints.
async fn status(
    ctx: ServeCtx,
    st: Arc<Shared>,
    ep: Option<Endpoint>,
    id: String,
    headers: HeaderMap,
) -> Response {
    let rid = fastvideo_serve_kit::random_token();
    let reply = async {
        let owner = ctx.auth().authenticate(Api::V2.protocol(), &headers)?;
        let ep = ep.ok_or_else(|| not_found(&id))?;
        let job = find_job(&ctx, &LtxProtocol::V2, &id, owner.as_ref())
            .await
            .map_err(|_| not_found(&id))?;
        Ok::<_, ApiError>(V2View::new(ep, st.cfg.url_ttl).status_reply(&job, &ctx.view_ctx(false)))
    }
    .await
    .unwrap_or_else(|e| error::render(&e, Api::V2, &ecx(&rid, Some(&id))));
    finish(reply, &rid, &ctx).await
}
