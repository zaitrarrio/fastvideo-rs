//! Webhooks (`?fal_webhook=`) signed with our Ed25519 key, and our JWKS
//! (design §4.4; fal §9.7; risk R9).
//!
//! On completion the job's webhook gets one POST
//! `{request_id, gateway_request_id, status: "OK"|"ERROR", payload}`:
//!
//! - success: `payload` is the output JSON (the same body as the result
//!   route; `sync_mode` does not inline the video here);
//! - failure: `status: "ERROR"`, `error: "Invalid status code: <http>"` and
//!   the error body the result route answers as `payload`;
//! - cancelled: `status: "ERROR"` with a `client_cancelled` payload (499).
//!
//! `gateway_request_id` equals `request_id` (we have no separate gateway).
//! Delivery, retries (fal schedule) and signing live in serve-kit's
//! `CallbackSender`: headers `X-Fal-Webhook-{Request-Id,User-Id,Timestamp,
//! Signature}`, signature = hex Ed25519 over
//! `request_id\nuser_id\ntimestamp\nhex(sha256(body))`. Receivers verify with
//! our key from `GET /.well-known/jwks.json`, since fal's keys cannot sign
//! for us.

use std::time::Duration;

use axum::response::{IntoResponse, Response};
use axum::routing::get;
use axum::Router;
use fastvideo_protocol::{ApiError, Job, JobState, ViewCtx};
use fastvideo_serve_kit::callback::fal_webhook_body;
use fastvideo_serve_kit::{CallbackRender, ServeCtx};
use serde_json::json;

use crate::error::error_body;
use crate::queue::output_json;

/// Renders fal webhook bodies; register it for `ProtocolId::Fal` on the
/// `ServeCtx` builder.
#[derive(Clone, Debug)]
pub struct FalWebhook {
    pub url_ttl: Duration,
}

impl Default for FalWebhook {
    fn default() -> Self {
        Self { url_ttl: Duration::from_secs(24 * 3600) }
    }
}

/// The webhook body for `job`'s current state; `None` until it is terminal.
pub fn webhook_body(job: &Job, cx: &ViewCtx, url_ttl: Duration) -> Option<serde_json::Value> {
    let rid = job.external_id.as_str();
    let fail = |e: &ApiError| {
        let (status, _, body) = error_body(e);
        fal_webhook_body(rid, rid, false, body, Some(&format!("Invalid status code: {status}")))
    };
    match &job.state {
        JobState::Queued | JobState::Running => None,
        JobState::Succeeded => Some(match output_json(job, cx, url_ttl) {
            Some(out) => fal_webhook_body(rid, rid, true, out, None),
            None => fail(&ApiError::internal("the job has no output")),
        }),
        JobState::Failed(e) => Some(fail(e)),
        JobState::Cancelled => Some(fail(&ApiError::cancelled("Request was cancelled"))),
    }
}

impl CallbackRender for FalWebhook {
    fn callback_body(&self, job: &Job, cx: &ViewCtx) -> Option<serde_json::Value> {
        webhook_body(job, cx, self.url_ttl)
    }
}

async fn jwks(ctx: axum::extract::State<ServeCtx>) -> Response {
    match ctx.callbacks().signer() {
        Some(s) => axum::Json(s.jwks()).into_response(),
        None => (
            axum::http::StatusCode::NOT_FOUND,
            axum::Json(json!({"detail": "webhook signing is not configured", "error_type": "not_found"})),
        )
            .into_response(),
    }
}

/// `GET /.well-known/jwks.json`.
pub(crate) fn routes(router: Router<ServeCtx>) -> Router<ServeCtx> {
    router.route("/.well-known/jwks.json", get(jwks))
}
