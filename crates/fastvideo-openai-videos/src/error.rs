//! OpenAI-style and FastAPI-style error rendering (design §4.1, §4.2, §4.6).
//!
//! - FastVideo `/v1/*`: `{"error":{"message","type","param","code"}}` where
//!   `type` is `invalid_request_error` (4xx) or `server_error` (5xx) and
//!   `code` is the HTTP status as an integer. Validation errors are 400,
//!   never 422 (`api_server.py:124-151`).
//! - FastWan: FastAPI's `{"detail": "<string>"}`. `Loading` is 503 because
//!   the client treats every non-rejecting status as "unreachable, retry".

use fastvideo_protocol::{ApiError, ErrorKind, HttpReply};
use serde_json::json;

/// The HTTP status of `kind` on both APIs (design §4.6 rows "FastVideo" and
/// "FastWan"). Only `ContentFiltered` differs from the canonical table: both
/// APIs answer validation-style refusals with 400, never 422.
pub fn status_of(kind: ErrorKind) -> u16 {
    match kind {
        ErrorKind::InvalidRequest | ErrorKind::Unsupported(_) | ErrorKind::ContentFiltered => 400,
        ErrorKind::Unauthorized => 401,
        ErrorKind::Forbidden => 403,
        ErrorKind::NotFound => 404,
        ErrorKind::AlreadyCompleted | ErrorKind::Conflict | ErrorKind::Cancelled => 409,
        ErrorKind::PayloadTooLarge => 413,
        ErrorKind::UnsupportedMedia => 415,
        ErrorKind::RateLimited | ErrorKind::QueueFull => 429,
        ErrorKind::Loading => 503,
        ErrorKind::Timeout => 504,
        ErrorKind::EngineFailed | ErrorKind::Internal => 500,
    }
}

/// Our generic parameter names (from `negotiate` / ingestion) in FastVideo's
/// request naming.
pub fn fastvideo_param(p: &str) -> String {
    let base = p.split('[').next().unwrap_or(p);
    match base {
        "duration" => "seconds".into(),
        "resolution" => "size".into(),
        "keyframes" | "image_url" => "image_reference".into(),
        "references" => "references".into(),
        "audio" | "audio_url" => "audio_reference".into(),
        _ => base.into(),
    }
}

/// `{"error":{...}}` with the status from [`status_of`], plus `Retry-After`
/// for retryable kinds.
pub fn openai_error(err: &ApiError) -> HttpReply {
    openai_error_status(status_of(err.kind), err)
}

/// Like [`openai_error`] with an explicit status (e.g. 422 for the content of
/// a failed job, 404 for one still in progress).
pub fn openai_error_status(status: u16, err: &ApiError) -> HttpReply {
    let kind = if status >= 500 {
        "server_error"
    } else {
        "invalid_request_error"
    };
    let body = json!({
        "error": {
            "message": err.message,
            "type": kind,
            "param": err.param.as_deref().map(fastvideo_param),
            "code": status,
        }
    });
    with_retry(HttpReply::json(status, body), err)
}

/// FastAPI `{"detail": "<message>"}` with the status from [`status_of`].
pub fn fastapi_error(err: &ApiError) -> HttpReply {
    fastapi_error_status(status_of(err.kind), err)
}

/// Like [`fastapi_error`] with an explicit status.
pub fn fastapi_error_status(status: u16, err: &ApiError) -> HttpReply {
    with_retry(
        HttpReply::json(status, json!({ "detail": err.message })),
        err,
    )
}

fn with_retry(mut r: HttpReply, err: &ApiError) -> HttpReply {
    if let Some(s) = err.retry_after_s {
        r.push_header("retry-after", s.to_string());
    }
    r
}

/// A serde error on the request body as an `InvalidRequest`, naming the
/// field when serde does (`unknown field `x``, `missing field `x``).
pub fn body_error(e: &serde_json::Error) -> ApiError {
    let msg = e.to_string();
    let field = ["unknown field `", "missing field `"]
        .iter()
        .find_map(|p| msg.split_once(p))
        .and_then(|(_, rest)| rest.split_once('`'))
        .map(|(f, _)| f.to_owned());
    let msg = match msg.split_once(" at line ") {
        Some((m, _)) => m.to_owned(),
        None => msg,
    };
    let err = ApiError::invalid(msg);
    match field {
        Some(f) => err.with_param(f),
        None => err,
    }
}
