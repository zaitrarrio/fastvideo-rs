//! Director signalling routes (design §5.6, fal §8.2).
//!
//! | Route | Behaviour |
//! |---|---|
//! | `POST /wma/ice` `{app_id}` | `{"ice_servers":[…]}` from config |
//! | `POST /{app}/director/ice`, `POST /run/{app}/director/ice` | the same body: the JS fallback (`context.run`, direct or through `/fal/proxy`) |
//! | `POST /wma/session` `{app_id,sdp,type:"offer"}` | non-trickle answer `{session_id,sdp,type:"answer"}`; busy → 429 |
//! | `POST /wma/session/heartbeat` `{session_id}` | `{alive}`; 15 s without one closes the session (ClientGone) |
//! | `POST /start-session` | runner side: SSE, first event `data:{sdp,type:"answer",session_id}`, `: keepalive` every 15 s, open for the session |
//! | `GET`/`POST /info` | runner side: `DirectorInfo` |
//!
//! Every route authenticates like the rest of fal (`Authorization: Key`).
//! Bridge errors are `{"error": "..."}` (what the JS client reads first);
//! a body that does not parse is 422 `text/plain`, as the bridge answers.
//! Model metadata (tier, recipe) rides on `x-fv-*` response headers of
//! `/wma/session`, since the WMA bodies have closed schemas.

use std::convert::Infallible;
use std::sync::Arc;
use std::time::Duration;

use axum::body::Bytes;
use axum::extract::State;
use axum::http::{HeaderMap, HeaderValue, StatusCode};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Extension, Json, Router};
use fastvideo_protocol::{ApiError, ErrorKind, ProtocolId};
use fastvideo_serve_kit::ServeCtx;
use futures::StreamExt;
use serde::Deserialize;
use serde_json::json;

use super::service::DirectorService;

type Svc = Extension<Arc<DirectorService>>;

/// The bridge's HTTP status for an error.
fn status_of(e: &ApiError) -> StatusCode {
    match e.kind {
        ErrorKind::Unauthorized => StatusCode::UNAUTHORIZED,
        ErrorKind::Forbidden => StatusCode::FORBIDDEN,
        ErrorKind::NotFound => StatusCode::NOT_FOUND,
        ErrorKind::Conflict | ErrorKind::RateLimited | ErrorKind::QueueFull => StatusCode::TOO_MANY_REQUESTS,
        ErrorKind::Loading => StatusCode::SERVICE_UNAVAILABLE,
        ErrorKind::InvalidRequest | ErrorKind::Unsupported(_) | ErrorKind::UnsupportedMedia => StatusCode::UNPROCESSABLE_ENTITY,
        ErrorKind::PayloadTooLarge => StatusCode::PAYLOAD_TOO_LARGE,
        ErrorKind::Timeout => StatusCode::GATEWAY_TIMEOUT,
        _ => StatusCode::INTERNAL_SERVER_ERROR,
    }
}

/// `{"error": message}` with the bridge's status (and `Retry-After`).
fn bridge_error(e: ApiError) -> Response {
    let mut r = (status_of(&e), Json(json!({"error": e.message}))).into_response();
    if let Some(s) = e.retry_after_s {
        r.headers_mut().insert("retry-after", HeaderValue::from(s));
    }
    r
}

/// Runner routes speak FastAPI: `{"detail": message}`.
fn runner_error(e: ApiError) -> Response {
    let mut r = (status_of(&e), Json(json!({"detail": e.message}))).into_response();
    if let Some(s) = e.retry_after_s {
        r.headers_mut().insert("retry-after", HeaderValue::from(s));
    }
    r
}

fn unprocessable(msg: String) -> Response {
    (StatusCode::UNPROCESSABLE_ENTITY, [("content-type", "text/plain; charset=utf-8")], msg).into_response()
}

fn auth(ctx: &ServeCtx, headers: &HeaderMap) -> Result<(), ApiError> {
    ctx.auth().authenticate(ProtocolId::Fal, headers).map(|_| ())
}

fn body<T: for<'de> Deserialize<'de>>(bytes: &Bytes) -> Result<T, Box<Response>> {
    serde_json::from_slice(bytes).map_err(|e| {
        // serde's messages ("missing field `app_id` at line 1 column 2")
        // without the position, as the bridge answers.
        let msg = e.to_string();
        let msg = msg.split(" at line ").next().unwrap_or(&msg).to_owned();
        Box::new(unprocessable(msg))
    })
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct IceBody {
    app_id: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SessionBody {
    app_id: String,
    sdp: String,
    #[serde(rename = "type")]
    ty: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct HeartbeatBody {
    session_id: String,
}

/// `StartSessionRequest` (API-DIR): the offer forwarded by the bridge.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct StartSessionBody {
    sdp: String,
    #[serde(rename = "type", default = "offer")]
    ty: String,
    #[serde(default)]
    session_id: Option<String>,
    #[serde(default)]
    #[allow(dead_code)]
    ice_servers: Option<Vec<serde_json::Value>>,
    #[serde(default)]
    #[allow(dead_code)]
    ice_status: Option<String>,
    #[serde(default)]
    #[allow(dead_code)]
    credential_age_seconds: Option<f64>,
}

fn offer() -> String {
    "offer".into()
}

fn ice_reply(svc: &DirectorService) -> Response {
    Json(json!({"ice_servers": svc.config().ice_servers})).into_response()
}

async fn wma_ice(State(ctx): State<ServeCtx>, Extension(svc): Svc, headers: HeaderMap, raw: Bytes) -> Response {
    if let Err(e) = auth(&ctx, &headers) {
        return bridge_error(e);
    }
    let b: IceBody = match body(&raw) {
        Ok(b) => b,
        Err(r) => return *r,
    };
    if svc.app_for(&b.app_id).is_none() {
        return bridge_error(ApiError::not_found(format!("unknown app `{}`", b.app_id)));
    }
    ice_reply(&svc)
}

async fn app_ice(State(ctx): State<ServeCtx>, Extension(svc): Svc, headers: HeaderMap) -> Response {
    if let Err(e) = auth(&ctx, &headers) {
        return runner_error(e);
    }
    ice_reply(&svc)
}

async fn wma_session(State(ctx): State<ServeCtx>, Extension(svc): Svc, headers: HeaderMap, raw: Bytes) -> Response {
    if let Err(e) = auth(&ctx, &headers) {
        return bridge_error(e);
    }
    let b: SessionBody = match body(&raw) {
        Ok(b) => b,
        Err(r) => return *r,
    };
    if b.ty != "offer" {
        return unprocessable(format!("`type` must be \"offer\", got {:?}", b.ty));
    }
    let Some(app) = svc.app_for(&b.app_id).cloned() else {
        return bridge_error(ApiError::not_found(format!("unknown app `{}`", b.app_id)));
    };
    match svc.create(&ctx, &app, &b.sdp, None, true).await {
        Ok(c) => {
            let mut r = Json(json!({"session_id": c.session_id, "sdp": c.sdp, "type": "answer"})).into_response();
            let h = r.headers_mut();
            if let Ok(v) = HeaderValue::from_str(c.caps.id.as_str()) {
                h.insert("x-fv-model", v);
            }
            if let Some(t) = c.caps.tier {
                h.insert("x-fv-tier", HeaderValue::from_static(t.as_str()));
            }
            if let Some(v) = c.caps.recipe.as_deref().and_then(|r| HeaderValue::from_str(r).ok()) {
                h.insert("x-fv-recipe", v);
            }
            r
        }
        Err(e) => bridge_error(e),
    }
}

async fn wma_heartbeat(State(ctx): State<ServeCtx>, Extension(svc): Svc, headers: HeaderMap, raw: Bytes) -> Response {
    if let Err(e) = auth(&ctx, &headers) {
        return bridge_error(e);
    }
    let b: HeartbeatBody = match body(&raw) {
        Ok(b) => b,
        Err(r) => return *r,
    };
    let alive = svc.session(&b.session_id).is_some_and(|s| s.heartbeats && s.beat());
    Json(json!({"alive": alive})).into_response()
}

/// Ends the session when the runner's SSE response is dropped.
struct EndOnDrop(Arc<super::session::SessionHandle>);

impl Drop for EndOnDrop {
    fn drop(&mut self) {
        self.0.end();
    }
}

async fn start_session(State(ctx): State<ServeCtx>, Extension(svc): Svc, headers: HeaderMap, raw: Bytes) -> Response {
    if let Err(e) = auth(&ctx, &headers) {
        return runner_error(e);
    }
    let b: StartSessionBody = match body(&raw) {
        Ok(b) => b,
        Err(r) => return *r,
    };
    if b.ty != "offer" {
        return unprocessable(format!("`type` must be \"offer\", got {:?}", b.ty));
    }
    let Some(app) = svc.runner_app().cloned() else {
        return runner_error(ApiError::not_found("no director app is configured"));
    };
    let c = match svc.create(&ctx, &app, &b.sdp, b.session_id, false).await {
        Ok(c) => c,
        Err(e) => return runner_error(e),
    };
    let first = Event::default().data(json!({"sdp": c.sdp, "type": "answer", "session_id": c.session_id}).to_string());
    let closed = c.handle.closed();
    let guard = EndOnDrop(c.handle.clone());
    let rest = futures::stream::unfold((closed, guard), |(mut closed, guard)| async move {
        let _ = closed.wait_for(|c| *c).await;
        drop(guard);
        None::<(Result<Event, Infallible>, _)>
    });
    let stream = futures::stream::once(async move { Ok::<_, Infallible>(first) }).chain(rest);
    Sse::new(stream).keep_alive(KeepAlive::new().interval(Duration::from_secs(15)).text("keepalive")).into_response()
}

async fn info(State(ctx): State<ServeCtx>, Extension(svc): Svc, headers: HeaderMap) -> Response {
    if let Err(e) = auth(&ctx, &headers) {
        return runner_error(e);
    }
    match svc.info(&ctx) {
        Ok(v) => Json(v).into_response(),
        Err(e) => runner_error(e),
    }
}

/// Every director route, still needing the `ServeCtx` state. Merge it into
/// the fal router with [`crate::router_with`] so `/fal/proxy` reaches it.
pub fn routes(svc: Arc<DirectorService>) -> Router<ServeCtx> {
    let mut r = Router::new()
        .route("/wma/ice", post(wma_ice))
        .route("/wma/session", post(wma_session))
        .route("/wma/session/heartbeat", post(wma_heartbeat))
        .route("/start-session", post(start_session))
        .route("/info", get(info).post(info));
    for app in &svc.config().apps {
        if !app.is_valid() {
            continue;
        }
        r = r
            .route(&format!("/{}/director/ice", app.id), post(app_ice))
            .route(&format!("/run/{}/director/ice", app.id), post(app_ice));
    }
    r.layer(Extension(svc))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn statuses() {
        assert_eq!(status_of(&ApiError::conflict("busy")), StatusCode::TOO_MANY_REQUESTS);
        assert_eq!(status_of(&ApiError::loading("warming")), StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(status_of(&ApiError::unauthorized("key")), StatusCode::UNAUTHORIZED);
        assert_eq!(status_of(&ApiError::invalid("bad")), StatusCode::UNPROCESSABLE_ENTITY);
        let r = bridge_error(ApiError::conflict("busy").with_retry_after(5));
        assert_eq!(r.status(), 429);
        assert_eq!(r.headers()["retry-after"], "5");
    }
}
