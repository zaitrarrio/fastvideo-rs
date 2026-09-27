//! `ANY /fal/proxy`: fal's proxy protocol, routed by `x-fal-target-url`
//! (design §4.4; fal §12.1, D-PROXY).
//!
//! `@fal-ai/client` with `proxyUrl: {url: "<base>/fal/proxy", when:
//! "always"}` sends every request to the proxy and moves the real target into
//! the `x-fal-target-url` header. We ignore the fal host's identity and
//! route on its role and path, into this crate's own router:
//!
//! | target host | routed to |
//! |---|---|
//! | `queue.fal.run` | `<path>` (queue submit / status / result / cancel) |
//! | `fal.run` | `/run<path>` (sync; `…/ice` for the director) |
//! | `wma.fal.run` | `/wma<path>` (director bridge, WP-14) |
//! | `rest.fal.ai`, `rest.alpha.fal.ai` | `<path>` (`/storage/upload/initiate`) |
//!
//! The query string is kept; the method, headers (including the caller's
//! `Authorization`) and body pass through unchanged. A missing header is 400,
//! a non-fal host 400, and an unrouted path answers whatever the inner router
//! answers (404).

use axum::body::Body;
use axum::extract::Request;
use axum::http::{HeaderValue, StatusCode, Uri};
use axum::response::{IntoResponse, Response};
use axum::routing::any;
use axum::Router;
use serde_json::json;
use tower::ServiceExt;

/// The header carrying the real target URL.
pub const TARGET_HEADER: &str = "x-fal-target-url";

/// Maps a fal target URL to a path (with query) on this server.
pub fn map_target(target: &str) -> Result<String, String> {
    let u = url::Url::parse(target.trim()).map_err(|e| format!("invalid {TARGET_HEADER}: {e}"))?;
    if !matches!(u.scheme(), "https" | "http") {
        return Err(format!("{TARGET_HEADER} must be an http(s) URL"));
    }
    let host = u.host_str().unwrap_or_default().to_ascii_lowercase();
    let path = u.path();
    let mapped = match host.as_str() {
        "queue.fal.run" => path.to_owned(),
        "fal.run" => format!("/run{path}"),
        "wma.fal.run" => format!("/wma{path}"),
        "rest.fal.ai" | "rest.alpha.fal.ai" => path.to_owned(),
        other => return Err(format!("{TARGET_HEADER} host `{other}` is not routed by this server")),
    };
    Ok(match u.query() {
        Some(q) => format!("{mapped}?{q}"),
        None => mapped,
    })
}

fn bad(detail: String) -> Response {
    let mut r = (StatusCode::BAD_REQUEST, axum::Json(json!({"detail": detail, "error_type": "bad_request"}))).into_response();
    r.headers_mut().insert("x-fal-error-type", HeaderValue::from_static("bad_request"));
    r
}

async fn proxy(inner: Router, req: Request) -> Response {
    let Some(target) = req.headers().get(TARGET_HEADER).and_then(|v| v.to_str().ok()).map(str::to_owned) else {
        return bad(format!("Missing the {TARGET_HEADER} header"));
    };
    let path = match map_target(&target) {
        Ok(p) => p,
        Err(e) => return bad(e),
    };
    let Ok(uri) = path.parse::<Uri>() else {
        return bad(format!("{TARGET_HEADER} has an unusable path"));
    };
    let (mut parts, body) = req.into_parts();
    parts.uri = uri;
    parts.headers.remove(TARGET_HEADER);
    let req = Request::from_parts(parts, body);
    match inner.oneshot(req).await {
        Ok(r) => r,
        Err(e) => match e {},
    }
}

/// `ANY /fal/proxy`, dispatching into `inner` (a router with its state).
pub fn proxy_router(inner: Router) -> Router {
    Router::new().route("/fal/proxy", any(move |req: Request<Body>| proxy(inner.clone(), req)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn targets() {
        let m = |s: &str| map_target(s);
        assert_eq!(m("https://queue.fal.run/minimax/h3-max/text-to-video?fal_webhook=x").unwrap(), "/minimax/h3-max/text-to-video?fal_webhook=x");
        assert_eq!(m("https://queue.fal.run/minimax/h3-max/requests/r1/status?logs=1").unwrap(), "/minimax/h3-max/requests/r1/status?logs=1");
        assert_eq!(m("https://fal.run/minimax/h3-max/text-to-video").unwrap(), "/run/minimax/h3-max/text-to-video");
        assert_eq!(m("https://fal.run/minimax/h3-max/director/ice").unwrap(), "/run/minimax/h3-max/director/ice");
        assert_eq!(m("https://wma.fal.run/session").unwrap(), "/wma/session");
        assert_eq!(m("https://rest.fal.ai/storage/upload/initiate?storage_type=fal-cdn-v3").unwrap(), "/storage/upload/initiate?storage_type=fal-cdn-v3");
        assert!(m("https://evil.example.com/x").is_err());
        assert!(m("not a url").is_err());
        assert!(m("ftp://queue.fal.run/x").is_err());
    }
}
