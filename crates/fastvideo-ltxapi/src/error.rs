//! LTX error envelope rendering (design §4.5, ltx §1.5).
//!
//! Every error is `{"type":"error","error":{"type":<error_type>,"message":..}}`
//! with one of the eleven documented error types. The same `{type, message}`
//! object is the `error` of a failed v2 job.
//!
//! | HTTP | `error.type` | produced for |
//! |---|---|---|
//! | 400 | `invalid_request_error` | `InvalidRequest`, `Unsupported(gap)` (except `LtxEndpoint`), bad media, conflicts |
//! | 401 | `authentication_error` | `Unauthorized` |
//! | 402 | `insufficient_funds_error` | never (no billing); renderable |
//! | 403 | `permission_error` | `Forbidden`, `Unsupported(LtxEndpoint)` |
//! | 404 | `not_found_error` | `NotFound` |
//! | 422 | `content_filtered_error` | `ContentFiltered` |
//! | 429 | `concurrency_limit_error` | `RateLimited` / `QueueFull` on `/v1/*` (sync) |
//! | 429 | `rate_limit_error` | `RateLimited` / `QueueFull` on `/v2/*` (async) |
//! | 500 | `api_error` | `EngineFailed`, `Internal`, `Cancelled` |
//! | 503 | `service_unavailable_error` | `Loading` |
//! | 529 | `overloaded_error` | never produced today; renderable |
//!
//! `Timeout` (a `/v1/*` generation over the sync timeout) answers `504`, the
//! status the OAS declares for v1 ("Request timeout"). The OAS gives it no
//! error type of its own, so the body uses `api_error` (retryable).
//!
//! Every error reply carries `x-request-id`; every 429 carries `Retry-After`.

use fastvideo_protocol::{
    ApiError, BatchProtocol, ErrorCtx, ErrorKind, GapId, HttpReply, JobId, ProtocolId,
};
use serde_json::{json, Value};

/// `Retry-After` seconds on a 429 when the error carries none.
pub const DEFAULT_RETRY_AFTER_S: u32 = 5;

/// Which LTX API surface a route belongs to; selects the 429 flavour.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Api {
    /// `/v1/*`: sync generation and `/v1/upload`.
    V1,
    /// `/v2/*`: async jobs.
    V2,
}

impl Api {
    pub fn protocol(&self) -> ProtocolId {
        match self {
            Api::V1 => ProtocolId::LtxV1,
            Api::V2 => ProtocolId::LtxV2,
        }
    }
    /// `/v1` or `/v2`.
    pub fn prefix(&self) -> &'static str {
        match self {
            Api::V1 => "/v1",
            Api::V2 => "/v2",
        }
    }
    /// The surface a matched route belongs to (`None` for other routes).
    pub fn of_route(route: &str) -> Option<Api> {
        if route.starts_with("/v1/") {
            Some(Api::V1)
        } else if route.starts_with("/v2/") {
            Some(Api::V2)
        } else {
            None
        }
    }
}

/// The eleven documented LTX error types (ltx §1.5).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum LtxErrorType {
    InvalidRequest,
    Authentication,
    InsufficientFunds,
    Permission,
    NotFound,
    ContentFiltered,
    ConcurrencyLimit,
    RateLimit,
    Api,
    ServiceUnavailable,
    Overloaded,
}

impl LtxErrorType {
    pub const ALL: [LtxErrorType; 11] = [
        LtxErrorType::InvalidRequest,
        LtxErrorType::Authentication,
        LtxErrorType::InsufficientFunds,
        LtxErrorType::Permission,
        LtxErrorType::NotFound,
        LtxErrorType::ContentFiltered,
        LtxErrorType::ConcurrencyLimit,
        LtxErrorType::RateLimit,
        LtxErrorType::Api,
        LtxErrorType::ServiceUnavailable,
        LtxErrorType::Overloaded,
    ];

    /// The wire name (`error.type`).
    pub fn as_str(&self) -> &'static str {
        match self {
            LtxErrorType::InvalidRequest => "invalid_request_error",
            LtxErrorType::Authentication => "authentication_error",
            LtxErrorType::InsufficientFunds => "insufficient_funds_error",
            LtxErrorType::Permission => "permission_error",
            LtxErrorType::NotFound => "not_found_error",
            LtxErrorType::ContentFiltered => "content_filtered_error",
            LtxErrorType::ConcurrencyLimit => "concurrency_limit_error",
            LtxErrorType::RateLimit => "rate_limit_error",
            LtxErrorType::Api => "api_error",
            LtxErrorType::ServiceUnavailable => "service_unavailable_error",
            LtxErrorType::Overloaded => "overloaded_error",
        }
    }

    /// The HTTP status documented for the type.
    pub fn status(&self) -> u16 {
        match self {
            LtxErrorType::InvalidRequest => 400,
            LtxErrorType::Authentication => 401,
            LtxErrorType::InsufficientFunds => 402,
            LtxErrorType::Permission => 403,
            LtxErrorType::NotFound => 404,
            LtxErrorType::ContentFiltered => 422,
            LtxErrorType::ConcurrencyLimit | LtxErrorType::RateLimit => 429,
            LtxErrorType::Api => 500,
            LtxErrorType::ServiceUnavailable => 503,
            LtxErrorType::Overloaded => 529,
        }
    }

    /// Whether the docs say the client may retry.
    pub fn retryable(&self) -> bool {
        matches!(
            self,
            LtxErrorType::ConcurrencyLimit
                | LtxErrorType::RateLimit
                | LtxErrorType::Api
                | LtxErrorType::ServiceUnavailable
                | LtxErrorType::Overloaded
        )
    }

    /// The error type for an [`ErrorKind`] on `api`.
    pub fn of(kind: ErrorKind, api: Api) -> Self {
        match kind {
            ErrorKind::InvalidRequest
            | ErrorKind::PayloadTooLarge
            | ErrorKind::UnsupportedMedia
            | ErrorKind::AlreadyCompleted
            | ErrorKind::Conflict => LtxErrorType::InvalidRequest,
            ErrorKind::Unsupported(GapId::LtxEndpoint) => LtxErrorType::Permission,
            ErrorKind::Unsupported(_) => LtxErrorType::InvalidRequest,
            ErrorKind::Unauthorized => LtxErrorType::Authentication,
            ErrorKind::Forbidden => LtxErrorType::Permission,
            ErrorKind::NotFound => LtxErrorType::NotFound,
            ErrorKind::ContentFiltered => LtxErrorType::ContentFiltered,
            ErrorKind::RateLimited | ErrorKind::QueueFull => match api {
                Api::V1 => LtxErrorType::ConcurrencyLimit,
                Api::V2 => LtxErrorType::RateLimit,
            },
            ErrorKind::Loading => LtxErrorType::ServiceUnavailable,
            ErrorKind::Timeout
            | ErrorKind::Cancelled
            | ErrorKind::EngineFailed
            | ErrorKind::Internal => LtxErrorType::Api,
        }
    }
}

impl std::fmt::Display for LtxErrorType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// The inner `{type, message}` object (also a failed job's `error`).
pub fn error_object(t: LtxErrorType, message: &str) -> Value {
    json!({ "type": t.as_str(), "message": message })
}

/// The full envelope `{"type":"error","error":{..}}`.
pub fn error_body(t: LtxErrorType, message: &str) -> Value {
    json!({ "type": "error", "error": error_object(t, message) })
}

/// The HTTP status for `err` on `api` (the type's status; `Timeout` -> 504).
pub fn status_of(err: &ApiError, api: Api) -> u16 {
    if err.kind == ErrorKind::Timeout {
        return 504;
    }
    LtxErrorType::of(err.kind, api).status()
}

/// Renders `err` as an LTX error reply on `api`, with `x-request-id` from
/// `cx` (fresh when absent) and `Retry-After` on 429/503.
pub fn render(err: &ApiError, api: Api, cx: &ErrorCtx) -> HttpReply {
    let t = LtxErrorType::of(err.kind, api);
    let mut r = HttpReply::json(status_of(err, api), error_body(t, &err.message));
    let rid = cx
        .request_id
        .clone()
        .unwrap_or_else(fastvideo_serve_kit::random_token);
    r.push_header("x-request-id", rid);
    let retry = match t {
        LtxErrorType::ConcurrencyLimit | LtxErrorType::RateLimit => {
            Some(err.retry_after_s.unwrap_or(DEFAULT_RETRY_AFTER_S))
        }
        _ => err.retry_after_s,
    };
    if let Some(s) = retry {
        r.push_header("retry-after", s.to_string());
    }
    r
}

/// The LTX `BatchProtocol` for one surface. Wire ids are the job uuid
/// (`a1b2c3d4-e5f6-7890-abcd-ef1234567890` shape, ltx §2.1).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LtxProtocol {
    pub api: Api,
}

impl LtxProtocol {
    pub const V1: LtxProtocol = LtxProtocol { api: Api::V1 };
    pub const V2: LtxProtocol = LtxProtocol { api: Api::V2 };
}

impl BatchProtocol for LtxProtocol {
    fn id(&self) -> ProtocolId {
        self.api.protocol()
    }
    fn new_external_id(&self, job: JobId) -> String {
        job.0.hyphenated().to_string()
    }
    fn render_error(&self, err: &ApiError, cx: &ErrorCtx) -> HttpReply {
        // The route decides the 429 flavour when known (upload is v1).
        let api = cx
            .route
            .as_deref()
            .and_then(Api::of_route)
            .unwrap_or(self.api);
        render(err, api, cx)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eleven_distinct_types() {
        let names: std::collections::BTreeSet<_> =
            LtxErrorType::ALL.iter().map(|t| t.as_str()).collect();
        assert_eq!(names.len(), 11);
        for t in LtxErrorType::ALL {
            assert!(t.as_str().ends_with("_error"));
            assert_eq!(t.retryable(), t.status() >= 429, "{t}");
        }
    }

    #[test]
    fn kind_table() {
        use ErrorKind as K;
        let v1 = |k| LtxErrorType::of(k, Api::V1);
        let v2 = |k| LtxErrorType::of(k, Api::V2);
        assert_eq!(v2(K::QueueFull), LtxErrorType::RateLimit);
        assert_eq!(v1(K::QueueFull), LtxErrorType::ConcurrencyLimit);
        assert_eq!(v1(K::RateLimited), LtxErrorType::ConcurrencyLimit);
        assert_eq!(v2(K::Unsupported(GapId::LtxFps)), LtxErrorType::InvalidRequest);
        assert_eq!(v2(K::Unsupported(GapId::LtxEndpoint)), LtxErrorType::Permission);
        assert_eq!(v2(K::Loading), LtxErrorType::ServiceUnavailable);
        assert_eq!(v2(K::Cancelled), LtxErrorType::Api);
        assert_eq!(status_of(&ApiError::timeout("t"), Api::V1), 504);
    }

    #[test]
    fn render_headers() {
        let cx = ErrorCtx {
            request_id: Some("1234567890abcdef1234567890abcdef".into()),
            ..Default::default()
        };
        let r = render(&ApiError::queue_full("full"), Api::V2, &cx);
        assert_eq!(r.status, 429);
        assert_eq!(r.header("x-request-id"), Some("1234567890abcdef1234567890abcdef"));
        assert_eq!(r.header("retry-after"), Some("5"));
        let r = render(&ApiError::loading("warming"), Api::V1, &ErrorCtx::default());
        assert_eq!(r.status, 503);
        assert_eq!(r.header("retry-after"), Some("1"));
        assert_eq!(r.header("x-request-id").map(str::len), Some(32));
        // The route decides the surface.
        let cx = ErrorCtx {
            route: Some("/v1/text-to-video".into()),
            ..Default::default()
        };
        let r = LtxProtocol::V2.render_error(&ApiError::queue_full("x"), &cx);
        assert_eq!(
            r.json_body().unwrap()["error"]["type"],
            "concurrency_limit_error"
        );
    }
}
