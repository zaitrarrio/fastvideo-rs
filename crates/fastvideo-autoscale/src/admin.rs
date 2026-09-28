//! `GET /fv/v1/admin/autoscale` (status and recent decisions) and
//! `POST /fv/v1/admin/autoscale` (`{"dry_run": bool}` and/or `{"tick": true}`),
//! both behind the gateway's admin token.

use std::sync::Arc;

use axum::extract::State;
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::get;
use axum::{Json, Router};
use serde::Deserialize;
use serde_json::json;

use crate::controller::{unix_now, Controller};

/// Returns true when the request carries the admin credential.
pub type Authorize = Arc<dyn Fn(&HeaderMap) -> bool + Send + Sync>;

pub const ADMIN_PATH: &str = "/fv/v1/admin/autoscale";

#[derive(Clone)]
struct AdminState {
    ctrl: Arc<Controller>,
    authorize: Authorize,
}

pub fn admin_router(ctrl: Arc<Controller>, authorize: Authorize) -> Router {
    Router::new().route(ADMIN_PATH, get(status).post(update)).with_state(AdminState { ctrl, authorize })
}

/// `Authorization: Bearer <token>` compared in constant time.
pub fn bearer(token: impl Into<String>) -> Authorize {
    let token = token.into();
    Arc::new(move |h: &HeaderMap| {
        let Some(v) = h.get(axum::http::header::AUTHORIZATION).and_then(|v| v.to_str().ok()) else { return false };
        let Some(got) = v.strip_prefix("Bearer ") else { return false };
        let (a, b) = (got.as_bytes(), token.as_bytes());
        !b.is_empty() && a.len() == b.len() && a.iter().zip(b).fold(0u8, |acc, (x, y)| acc | (x ^ y)) == 0
    })
}

fn unauthorized() -> Response {
    (StatusCode::UNAUTHORIZED, Json(json!({"error": "admin token required"}))).into_response()
}

async fn status(State(s): State<AdminState>, headers: HeaderMap) -> Response {
    if !(s.authorize)(&headers) {
        return unauthorized();
    }
    Json(s.ctrl.status()).into_response()
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Update {
    #[serde(default)]
    dry_run: Option<bool>,
    /// Run one tick now and return its report.
    #[serde(default)]
    tick: bool,
}

async fn update(State(s): State<AdminState>, headers: HeaderMap, body: Option<Json<Update>>) -> Response {
    if !(s.authorize)(&headers) {
        return unauthorized();
    }
    let Some(Json(u)) = body else {
        return (StatusCode::BAD_REQUEST, Json(json!({"error": "body: {\"dry_run\": bool, \"tick\": bool}"}))).into_response();
    };
    if let Some(d) = u.dry_run {
        s.ctrl.set_dry_run(d);
    }
    if u.tick {
        let r = s.ctrl.tick(unix_now()).await;
        return Json(json!({"dry_run": s.ctrl.dry_run(), "tick": r})).into_response();
    }
    Json(json!({"dry_run": s.ctrl.dry_run()})).into_response()
}
