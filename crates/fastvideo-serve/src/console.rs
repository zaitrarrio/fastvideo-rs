//! The browser console (docs/serve/console.md): static pages embedded in
//! the binary (`include_str!`, no build step, no external assets).
//!
//! | Route | Page |
//! |---|---|
//! | `GET /console` | Connect (server URL, API key in local storage) and the model list (`GET /fal/schema`) |
//! | `GET /console/admin` | Admin token (session storage), mint / list / revoke API keys (`/fv/v1/admin/keys`) |
//! | `GET /console/deployments` | Admin: release channels, live builds and drift, the deployment registry, history; Promote / Rollback (`/fv/v1/admin/releases*`, `/fv/v1/admin/deployments`; gateway only) |
//! | `GET /console/live` | Live input: publish the camera and microphone (getUserMedia) to a duplex model over native WHIP ingest or the Reactor runtime, and watch its output |
//! | `GET /console/avatar` | The script avatar (Reactor `ltx` over the Reactor runtime in avatar mode): photo, script, scene, speech rate, duration, seed; the WebRTC stream and per-window timings |
//! | `GET /console/models/{owner}/{alias}/{task}` | A fal model page: Playground (schema-driven form, uploads, result, logs, history) and API snippets; `task = director` is the live WebRTC page |
//! | `GET /console/assets/{file}` | CSS and JS modules |
//!
//! The pages only call the public APIs (fal queue, storage, schema, admin
//! keys) from the browser, so they hold no server state. Responses carry a
//! strict CSP (scripts from this origin only).

use axum::extract::Path;
use axum::http::{header, HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::get;
use axum::Router;

/// `(file name, content type, body)` of every embedded asset.
pub const ASSETS: &[(&str, &str, &str)] = &[
    ("console.css", "text/css; charset=utf-8", include_str!("../console/console.css")),
    ("common.js", "text/javascript; charset=utf-8", include_str!("../console/common.js")),
    ("home.js", "text/javascript; charset=utf-8", include_str!("../console/home.js")),
    ("admin.js", "text/javascript; charset=utf-8", include_str!("../console/admin.js")),
    ("deployments.js", "text/javascript; charset=utf-8", include_str!("../console/deployments.js")),
    ("model.js", "text/javascript; charset=utf-8", include_str!("../console/model.js")),
    ("form.js", "text/javascript; charset=utf-8", include_str!("../console/form.js")),
    ("snippets.js", "text/javascript; charset=utf-8", include_str!("../console/snippets.js")),
    ("director.js", "text/javascript; charset=utf-8", include_str!("../console/director.js")),
    ("avatar.js", "text/javascript; charset=utf-8", include_str!("../console/avatar.js")),
    ("live.js", "text/javascript; charset=utf-8", include_str!("../console/live.js")),
];

const INDEX: &str = include_str!("../console/index.html");
const ADMIN: &str = include_str!("../console/admin.html");
const DEPLOYMENTS: &str = include_str!("../console/deployments.html");
const MODEL: &str = include_str!("../console/model.html");
const AVATAR: &str = include_str!("../console/avatar.html");
const LIVE: &str = include_str!("../console/live.html");

/// Scripts and styles from this origin only; media, images and API calls may
/// go to other origins (signed file URLs, a configured server URL).
pub const CSP: &str = "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; \
img-src 'self' data: blob: http: https:; media-src 'self' data: blob: http: https:; \
connect-src 'self' http: https:; object-src 'none'; base-uri 'none'; frame-ancestors 'none'; form-action 'none'";

fn with_headers(content_type: &'static str, body: &'static str, html: bool) -> Response {
    let mut r = (StatusCode::OK, body).into_response();
    let h = r.headers_mut();
    h.insert(header::CONTENT_TYPE, HeaderValue::from_static(content_type));
    h.insert(header::CACHE_CONTROL, HeaderValue::from_static("no-cache"));
    h.insert(header::X_CONTENT_TYPE_OPTIONS, HeaderValue::from_static("nosniff"));
    h.insert(header::REFERRER_POLICY, HeaderValue::from_static("no-referrer"));
    if html {
        h.insert(header::CONTENT_SECURITY_POLICY, HeaderValue::from_static(CSP));
        h.insert(header::X_FRAME_OPTIONS, HeaderValue::from_static("DENY"));
    }
    r
}

fn page(body: &'static str) -> Response {
    with_headers("text/html; charset=utf-8", body, true)
}

async fn asset(Path(file): Path<String>) -> Response {
    match ASSETS.iter().find(|(name, _, _)| *name == file) {
        Some((_, ct, body)) => with_headers(ct, body, false),
        None => (StatusCode::NOT_FOUND, "no such console asset").into_response(),
    }
}

/// The console routes.
pub fn routes() -> Router {
    Router::new()
        .route("/console", get(|| async { page(INDEX) }))
        .route("/console/admin", get(|| async { page(ADMIN) }))
        .route("/console/deployments", get(|| async { page(DEPLOYMENTS) }))
        .route("/console/avatar", get(|| async { page(AVATAR) }))
        .route("/console/live", get(|| async { page(LIVE) }))
        .route("/console/models/{owner}/{alias}/{*task}", get(|| async { page(MODEL) }))
        .route("/console/assets/{file}", get(asset))
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::http::Request;
    use tower::ServiceExt;

    async fn get(uri: &str) -> (StatusCode, String, String) {
        let r = routes().oneshot(Request::get(uri).body(Body::empty()).unwrap()).await.unwrap();
        let ct = r.headers().get(header::CONTENT_TYPE).map(|v| v.to_str().unwrap().to_owned()).unwrap_or_default();
        let s = r.status();
        let b = axum::body::to_bytes(r.into_body(), 1 << 20).await.unwrap();
        (s, ct, String::from_utf8(b.to_vec()).unwrap())
    }

    #[tokio::test]
    async fn pages_and_assets_serve_with_their_types() {
        for (uri, marker) in [
            ("/console", "home.js"),
            ("/console/admin", "admin.js"),
            ("/console/deployments", "deployments.js"),
            ("/console/avatar", "avatar.js"),
            ("/console/models/minimax/h3-max/text-to-video", "model.js"),
            ("/console/models/minimax/h3-turbo/director", "model.js"),
        ] {
            let (s, ct, body) = get(uri).await;
            assert_eq!(s, 200, "{uri}");
            assert_eq!(ct, "text/html; charset=utf-8", "{uri}");
            assert!(body.contains(marker), "{uri}");
        }
        for (name, ct, _) in ASSETS {
            let (s, got, body) = get(&format!("/console/assets/{name}")).await;
            assert_eq!((s, got.as_str()), (StatusCode::OK, *ct), "{name}");
            assert!(!body.is_empty());
        }
        assert_eq!(get("/console/assets/nope.js").await.0, 404);
    }

    /// Every asset a page or module references is embedded, and nothing is
    /// loaded from another origin.
    #[test]
    fn references_resolve_and_stay_local() {
        let mut texts: Vec<&str> = vec![INDEX, ADMIN, DEPLOYMENTS, MODEL];
        texts.extend(ASSETS.iter().map(|a| a.2));
        for t in &texts {
            for part in t.split("/console/assets/").skip(1) {
                let name: String = part.chars().take_while(|c| c.is_ascii_alphanumeric() || *c == '.' || *c == '-').collect();
                assert!(ASSETS.iter().any(|a| a.0 == name), "missing asset {name}");
            }
            for part in t.split("from './").skip(1).chain(t.split("import('./").skip(1)) {
                let name: String = part.chars().take_while(|c| *c != '\'').collect();
                assert!(ASSETS.iter().any(|a| a.0 == name), "missing module {name}");
            }
            assert!(!t.contains("<script src=\"http"), "no external scripts");
        }
        for t in [INDEX, ADMIN, DEPLOYMENTS, MODEL] {
            for attr in ["onclick=\"", "onsubmit=\"", "onload=\"", "<script>"] {
                assert!(!t.contains(attr), "inline script `{attr}` is blocked by the CSP");
            }
        }
    }
}
