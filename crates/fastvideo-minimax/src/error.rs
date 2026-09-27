//! `OaiError` envelope rendering (design §4.3, research §1.5).
//!
//! ```json
//! {"type":"error","error":{"type":"bad_request_error",
//!  "message":"invalid params, … (2013)","http_code":"400"},"request_id":"<32 hex>"}
//! ```
//!
//! The HTTP status is the real one:
//!
//! | Kind | HTTP | `error.type` | Code |
//! |---|---|---|---|
//! | InvalidRequest / Unsupported / PayloadTooLarge / UnsupportedMedia / Conflict / AlreadyCompleted / Cancelled | 400 | `bad_request_error` | 2013 |
//! | NotFound (unknown `task_id`) | 404 | `bad_request_error` | `invalid task_id (2013)` |
//! | Unauthorized / Forbidden | 401 | `authorized_error` | 1004 |
//! | ContentFiltered | 422 | `unprocessable_entity_error` | 1026 |
//! | RateLimited / QueueFull | 429 | `rate_limit_error` | 1002 |
//! | Loading | 529 | `overloaded_error` | — |
//! | Timeout | 500 | `server_error` | 1001 |
//! | EngineFailed / Internal | 500 | `server_error` | 1000 |

use fastvideo_protocol::{ApiError, ErrorKind, HttpReply};
use serde_json::json;

/// How one [`ErrorKind`] renders.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OaiClass {
    pub http: u16,
    pub error_type: &'static str,
    /// The platform code in the message's trailing parentheses.
    pub code: Option<&'static str>,
}

/// The table in the module docs.
pub fn classify(kind: ErrorKind) -> OaiClass {
    let c = |http, error_type, code| OaiClass { http, error_type, code };
    match kind {
        ErrorKind::InvalidRequest
        | ErrorKind::Unsupported(_)
        | ErrorKind::PayloadTooLarge
        | ErrorKind::UnsupportedMedia
        | ErrorKind::Conflict
        | ErrorKind::AlreadyCompleted
        | ErrorKind::Cancelled => c(400, "bad_request_error", Some("2013")),
        ErrorKind::NotFound => c(404, "bad_request_error", Some("2013")),
        ErrorKind::Unauthorized | ErrorKind::Forbidden => c(401, "authorized_error", Some("1004")),
        ErrorKind::ContentFiltered => c(422, "unprocessable_entity_error", Some("1026")),
        ErrorKind::RateLimited | ErrorKind::QueueFull => c(429, "rate_limit_error", Some("1002")),
        ErrorKind::Loading => c(529, "overloaded_error", None),
        ErrorKind::Timeout => c(500, "server_error", Some("1001")),
        ErrorKind::EngineFailed | ErrorKind::Internal => c(500, "server_error", Some("1000")),
    }
}

/// The platform code a failed task reports in `task.error.code`
/// (design §4.3: `"1000"`; content filtering `"1026"`, timeouts `"1001"`).
pub fn task_error_code(kind: ErrorKind) -> &'static str {
    match kind {
        ErrorKind::ContentFiltered => "1026",
        ErrorKind::Timeout => "1001",
        _ => "1000",
    }
}

/// MiniMax's own wording for a request without a key (research §1.5).
pub const MISSING_KEY_MESSAGE: &str =
    "Please carry the API secret key in the 'Authorization' field of the request header";

/// The `error.message` text: MiniMax's prefix per class, then ` (<code>)`.
pub fn message_for(err: &ApiError) -> String {
    let class = classify(err.kind);
    let body = match err.kind {
        ErrorKind::NotFound => "invalid task_id".to_owned(),
        ErrorKind::Unauthorized | ErrorKind::Forbidden => {
            let m = if err.message == "missing credentials" {
                MISSING_KEY_MESSAGE
            } else {
                err.message.as_str()
            };
            format!("login fail: {m}")
        }
        ErrorKind::RateLimited | ErrorKind::QueueFull => format!("rate limit, {}", err.message),
        ErrorKind::EngineFailed | ErrorKind::Internal | ErrorKind::Timeout => {
            format!("internal error, {}", err.message)
        }
        ErrorKind::Loading => format!("overloaded, {}", err.message),
        ErrorKind::ContentFiltered => err.message.clone(),
        _ => format!("invalid params, {}", err.message),
    };
    match class.code {
        Some(code) => format!("{body} ({code})"),
        None => body,
    }
}

/// Renders `err` as an `OaiError` reply with its real HTTP status.
pub fn render_oai_error(err: &ApiError, request_id: Option<&str>) -> HttpReply {
    let class = classify(err.kind);
    let rid = request_id
        .map(str::to_owned)
        .unwrap_or_else(fastvideo_serve_kit::random_token);
    HttpReply::json(
        class.http,
        json!({
            "type": "error",
            "error": {
                "type": class.error_type,
                "message": message_for(err),
                "http_code": class.http.to_string(),
            },
            "request_id": rid,
        }),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use fastvideo_protocol::GapId;

    #[test]
    fn envelope_and_codes() {
        let r = render_oai_error(
            &ApiError::invalid("content must include a non-empty text item (prompt is required)"),
            Some("021785229015510a2c883cf675b9804d"),
        );
        assert_eq!(r.status, 400);
        assert_eq!(
            r.json_body().unwrap(),
            &json!({"type":"error","error":{"type":"bad_request_error",
                "message":"invalid params, content must include a non-empty text item (prompt is required) (2013)",
                "http_code":"400"},"request_id":"021785229015510a2c883cf675b9804d"})
        );
        let s = |e: ApiError| render_oai_error(&e, None).status;
        assert_eq!(s(ApiError::unsupported(GapId::H3Resolution2K)), 400);
        assert_eq!(s(ApiError::not_found("x")), 404);
        assert_eq!(s(ApiError::unauthorized("x")), 401);
        assert_eq!(s(ApiError::content_filtered("x")), 422);
        assert_eq!(s(ApiError::queue_full("x")), 429);
        assert_eq!(s(ApiError::loading("x")), 529);
        assert_eq!(s(ApiError::internal("x")), 500);
        assert_eq!(message_for(&ApiError::not_found("x")), "invalid task_id (2013)");
        assert_eq!(message_for(&ApiError::loading("models are loading")), "overloaded, models are loading");
        let rid = render_oai_error(&ApiError::internal("x"), None);
        assert_eq!(rid.json_body().unwrap()["request_id"].as_str().unwrap().len(), 32);
    }
}
