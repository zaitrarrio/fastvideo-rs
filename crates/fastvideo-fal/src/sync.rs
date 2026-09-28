//! `POST /run/{app}/{sub}`: the synchronous endpoint (design §4.4; fal §10.1).
//!
//! The body is the model input; the response is the model output on the
//! same connection, with `x-fal-request-id` (JS reads it into
//! `Result.requestId`). The job goes through the same queue as a queued
//! submit, so it is also visible at `/{app}/requests/{id}/status`. If it does
//! not finish within `ServeConfig::sync_timeout`, it is cancelled and the
//! answer is 504 `request_timeout`. Python reaches this route through
//! `FAL_RUN_HOST=<host>/run` (fal §12.2).

use std::sync::Arc;

use axum::body::Bytes;
use axum::extract::{Query, State};
use axum::http::HeaderMap;
use axum::response::Response;
use axum::routing::post;
use axum::Router;
use fastvideo_protocol::{ApiError, ErrorCtx, JobView};
use fastvideo_serve_kit::events::{cancel_job, wait_terminal};
use fastvideo_serve_kit::handlers::error_reply;
use fastvideo_serve_kit::{into_response, ServeCtx};

use crate::error::FalProtocol;
use crate::queue::{inline_video, submit_job, FalEndpoint, FalView};
use crate::{FalApp, FalConfig};

async fn run_handler(
    ctx: ServeCtx,
    cfg: Arc<FalConfig>,
    ep: Arc<FalEndpoint>,
    query: Vec<(String, String)>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    let route = Some(format!("/run/{}", ep.endpoint_id()));
    let job = match submit_job(&ctx, &cfg, &ep, query, &headers, &body).await {
        Ok(j) => j,
        Err(e) => {
            let ecx = ErrorCtx { request_id: Some(uuid::Uuid::new_v4().to_string()), route, external_id: None };
            return into_response(error_reply(&FalProtocol, &e, &ecx), &ctx, None).await;
        }
    };
    let rid = job.external_id.clone();
    let job = wait_terminal(&ctx, job.id, ctx.config().sync_timeout).await.unwrap_or(job);
    if !job.is_terminal() {
        let _ = cancel_job(&ctx, job.id).await;
        let ecx = ErrorCtx { request_id: Some(rid.clone()), route, external_id: Some(rid) };
        let e = ApiError::timeout("The request did not finish within the server's sync timeout");
        return into_response(error_reply(&FalProtocol, &e, &ecx), &ctx, None).await;
    }
    let view = FalView { url_ttl: cfg.url_ttl };
    let reply = view.result_reply(&job, &ctx.view_ctx(false));
    let reply = inline_video(reply, &job, cfg.inline_max_bytes).await;
    into_response(reply, &ctx, None).await
}

/// Adds `POST /run/{app}/{sub}` for every endpoint of `app`.
pub(crate) fn app_routes(mut router: Router<ServeCtx>, cfg: &Arc<FalConfig>, app: &FalApp) -> Router<ServeCtx> {
    for &endpoint in app.endpoints() {
        let ep = Arc::new(FalEndpoint { app: app.clone(), endpoint });
        let (c, e) = (cfg.clone(), ep.clone());
        router = router.route(
            &format!("/run/{}", ep.endpoint_id()),
            post(move |State(ctx): State<ServeCtx>, Query(q): Query<Vec<(String, String)>>, headers: HeaderMap, body: Bytes| {
                run_handler(ctx, c.clone(), e.clone(), q, headers, body)
            })
            .layer(axum::extract::DefaultBodyLimit::max(cfg.body_max)),
        );
    }
    router
}
