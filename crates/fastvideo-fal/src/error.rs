//! fal error envelopes (design §4.4 "Errors", §4.6; fal §9.6, §13).
//!
//! | Kind | HTTP | Body | `X-Fal-Error-Type` |
//! |---|---|---|---|
//! | InvalidRequest / Unsupported / UnsupportedMedia | 422 | `{"detail":[{loc,msg,type:"value_error"}]}` | `value_error` |
//! | ContentFiltered | 422 | `{"detail":[{loc,msg,type:"content_policy_violation"}]}` | `content_policy_violation` |
//! | Unauthorized | 401 | `{"detail":"invalid key credentials"}` | `unauthorized` |
//! | Forbidden | 403 | `{"detail","error_type":"forbidden"}` | `forbidden` |
//! | NotFound | 404 | `{"detail","error_type":"not_found"}` (unknown request ids: `{"status":"NOT_FOUND"}`, see [`not_found_request`]) | `not_found` |
//! | AlreadyCompleted | 400 | `{"status":"ALREADY_COMPLETED"}` | `bad_request` |
//! | Conflict | 409 | `{"detail","error_type":"bad_request"}` | `bad_request` |
//! | PayloadTooLarge | 413 | `{"detail","error_type":"bad_request"}` | `bad_request` |
//! | RateLimited / QueueFull | 429 | `{"detail","error_type":"concurrent_requests_limit"}` + `X-Fal-Needs-Retry: 1` | `concurrent_requests_limit` |
//! | Loading | 503 | `{"detail","error_type":"runner_scheduling_failure"}` (+ `Retry-After`) | `runner_scheduling_failure` |
//! | Timeout | 504 | `{"detail","error_type":"request_timeout"}` | `request_timeout` |
//! | Cancelled | 499 | `{"detail","error_type":"client_cancelled"}` | `client_cancelled` |
//! | EngineFailed | 500 | `{"detail":[{loc,msg,type:"internal_server_error"}]}` (a model error, fal §9.6) | `internal_server_error` |
//! | Internal | 500 | `{"detail","error_type":"internal_error"}` | `internal_error` |
//!
//! Every error carries `X-Fal-Error-Type` and, when known, `x-fal-request-id`.

use fastvideo_protocol::{ApiError, BatchProtocol, ErrorCtx, ErrorKind, GapId, HttpReply, JobId, ProtocolId};
use serde_json::json;

use crate::schema::loc;

/// The fal batch API (queue and sync).
#[derive(Clone, Copy, Debug, Default)]
pub struct FalProtocol;

/// `(status, error_type, detail-list form)` for an error kind.
pub fn classify(kind: ErrorKind) -> (u16, &'static str, bool) {
    match kind {
        ErrorKind::InvalidRequest | ErrorKind::Unsupported(_) | ErrorKind::UnsupportedMedia => (422, "value_error", true),
        ErrorKind::ContentFiltered => (422, "content_policy_violation", true),
        ErrorKind::Unauthorized => (401, "unauthorized", false),
        ErrorKind::Forbidden => (403, "forbidden", false),
        ErrorKind::NotFound => (404, "not_found", false),
        ErrorKind::AlreadyCompleted => (400, "bad_request", false),
        ErrorKind::Conflict => (409, "bad_request", false),
        ErrorKind::PayloadTooLarge => (413, "bad_request", false),
        ErrorKind::RateLimited | ErrorKind::QueueFull => (429, "concurrent_requests_limit", false),
        ErrorKind::Loading => (503, "runner_scheduling_failure", false),
        ErrorKind::Timeout => (504, "request_timeout", false),
        ErrorKind::Cancelled => (499, "client_cancelled", false),
        ErrorKind::EngineFailed => (500, "internal_server_error", true),
        ErrorKind::Internal => (500, "internal_error", false),
    }
}

/// The body and status of `err` (without headers).
pub fn error_body(err: &ApiError) -> (u16, &'static str, serde_json::Value) {
    let (status, ty, list) = classify(err.kind);
    let body = match err.kind {
        ErrorKind::Unauthorized => json!({"detail": "invalid key credentials"}),
        ErrorKind::AlreadyCompleted => json!({"status": "ALREADY_COMPLETED"}),
        _ if list => json!({"detail": [{
            "loc": loc(err.param.as_deref().or_else(|| gap_param(err.kind))),
            "msg": err.message,
            "type": ty,
        }]}),
        _ => json!({"detail": err.message, "error_type": ty}),
    };
    (status, ty, body)
}

/// The fal field an engine gap is about, when the error names none.
pub fn gap_param(kind: ErrorKind) -> Option<&'static str> {
    match kind {
        ErrorKind::Unsupported(g) => match g {
            GapId::H3TargetAudio => Some("target_audio_url"),
            GapId::H3Refine1080P | GapId::H3Resolution2K => Some("resolution"),
            GapId::H3FourSeconds => Some("duration"),
            _ => None,
        },
        _ => None,
    }
}

/// The `error` string a failed job's status carries (and a webhook's `error`).
pub fn error_message(err: &ApiError) -> String {
    match &err.param {
        Some(p) if error_body(err).2["detail"].is_array() => format!("{p}: {}", err.message),
        _ => err.message.clone(),
    }
}

impl BatchProtocol for FalProtocol {
    fn id(&self) -> ProtocolId {
        ProtocolId::Fal
    }
    /// fal request ids are UUIDs; a fresh one, unrelated to the internal id.
    fn new_external_id(&self, _job: JobId) -> String {
        uuid::Uuid::new_v4().to_string()
    }
    fn render_error(&self, err: &ApiError, cx: &ErrorCtx) -> HttpReply {
        let (status, ty, body) = error_body(err);
        let mut r = HttpReply::json(status, body).with_header("x-fal-error-type", ty);
        if matches!(err.kind, ErrorKind::RateLimited | ErrorKind::QueueFull) {
            r.push_header("x-fal-needs-retry", "1");
        }
        if let Some(id) = &cx.request_id {
            r.push_header("x-fal-request-id", id.clone());
        }
        r
    }
}

/// 404 `{"status":"NOT_FOUND"}` for an unknown request id (fal §9.3 PROBE).
pub fn not_found_request(request_id: Option<&str>) -> HttpReply {
    let mut r = HttpReply::json(404, json!({"status": "NOT_FOUND"})).with_header("x-fal-error-type", "not_found");
    if let Some(id) = request_id {
        r.push_header("x-fal-request-id", id.to_owned());
    }
    r
}
