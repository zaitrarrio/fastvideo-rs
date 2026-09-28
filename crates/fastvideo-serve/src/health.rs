//! `/health`, `/healthz`, `/ping`, `/` and `/metrics` (design §6.1-6.3, §9).
//!
//! - `GET /ping` (Runpod load balancer): **204** while models load, **200**
//!   `{"status":"healthy"}` when ready, **503** when loading failed or the
//!   server is draining (the balancer stops routing to it). The binary
//!   binds before building the app, and answers 204 from
//!   [`crate::app::serve_while_building`] until these routes exist.
//! - `GET /health`: the merged FastVideo / FastWan / Reactor body
//!   `{"status":"ok","model_loaded":true,"state":"AVAILABLE"}`; 503 with
//!   `model_loaded:false` until ready.
//! - `GET /healthz`: `{state, loaded, loading, models, engine, jobs}`; 200 when
//!   ready, else 503 (pods and Vast probe it).
//! - `GET /`: `{"model": <first served name>, "server": "fv-serve", ...}`
//!   (FastWan reads `model`).
//! - `GET /metrics`: Prometheus text.

use std::sync::Arc;

use axum::extract::State;
use axum::http::{header, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::get;
use axum::{Json, Router};
use fastvideo_engine_service::{Readiness, Residency};
use serde_json::json;

use crate::gate::ServiceGate;

/// State for the health routes.
#[derive(Clone)]
pub struct Health {
    pub gate: Arc<ServiceGate>,
    pub metrics: Option<metrics_exporter_prometheus::PrometheusHandle>,
    pub jobs_backend: &'static str,
    pub artifacts_backend: &'static str,
    /// `GET /` model (FastWan's, when mounted); else the first resident
    /// model's served name.
    pub root_model: Option<String>,
}

/// Server state as the probes see it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Phase {
    Loading,
    Ready,
    Failed,
    Draining,
}

impl Health {
    pub fn phase(&self) -> Phase {
        if !self.gate.admitting() {
            return Phase::Draining;
        }
        match self.gate.engine().readiness() {
            Readiness::Ready => Phase::Ready,
            Readiness::Loading { .. } => Phase::Loading,
            Readiness::Failed(_) => Phase::Failed,
        }
    }

    fn served_name(&self) -> Option<String> {
        if let Some(m) = &self.root_model {
            return Some(m.clone());
        }
        let caps = self.gate.engine().caps();
        caps.models()
            .find(|m| m.resident)
            .or_else(|| caps.models().next())
            .map(|m| m.served_names.first().cloned().unwrap_or_else(|| m.id.0.clone()))
    }
}

pub fn routes(h: Health) -> Router {
    Router::new()
        .route("/ping", get(ping))
        .route("/health", get(health))
        .route("/healthz", get(healthz))
        .route("/", get(root))
        .route("/metrics", get(metrics_text))
        .with_state(h)
}

async fn ping(State(h): State<Health>) -> Response {
    match h.phase() {
        Phase::Loading => StatusCode::NO_CONTENT.into_response(),
        Phase::Ready => (StatusCode::OK, Json(json!({"status": "healthy"}))).into_response(),
        Phase::Failed => (StatusCode::SERVICE_UNAVAILABLE, Json(json!({"status": "failed"}))).into_response(),
        Phase::Draining => (StatusCode::SERVICE_UNAVAILABLE, Json(json!({"status": "draining"}))).into_response(),
    }
}

async fn health(State(h): State<Health>) -> Response {
    let (code, state) = match h.phase() {
        Phase::Ready => (StatusCode::OK, "AVAILABLE"),
        Phase::Loading => (StatusCode::SERVICE_UNAVAILABLE, "LOADING"),
        Phase::Failed => (StatusCode::SERVICE_UNAVAILABLE, "FAILED"),
        Phase::Draining => (StatusCode::SERVICE_UNAVAILABLE, "DRAINING"),
    };
    let ready = code == StatusCode::OK;
    // `status` as FastWan's health body (`ok` / `loading`), plus our states.
    let status = match h.phase() {
        Phase::Ready => "ok",
        Phase::Loading => "loading",
        Phase::Failed => "failed",
        Phase::Draining => "draining",
    };
    (code, Json(json!({"status": status, "model_loaded": ready, "state": state}))).into_response()
}

async fn healthz(State(h): State<Health>) -> Response {
    let engine = h.gate.engine();
    let pool = engine.pool();
    let mut loaded = Vec::new();
    let mut loading = Vec::new();
    for e in pool.entries() {
        match e.state {
            Residency::Resident => loaded.push(e.model.0.clone()),
            Residency::Loading { .. } => loading.push(e.model.0.clone()),
            _ => {}
        }
    }
    loaded.sort();
    loaded.dedup();
    let phase = h.phase();
    let state = match phase {
        Phase::Loading => "loading",
        Phase::Ready => "ready",
        Phase::Failed => "failed",
        Phase::Draining => "draining",
    };
    let stats = engine.stats();
    let failed = match engine.readiness() {
        Readiness::Failed(m) => Some(m),
        _ => None,
    };
    let body = json!({
        "state": state,
        "loaded": loaded,
        "loading": loading,
        "error": failed,
        "models": engine.caps().models().map(|m| m.id.0.clone()).collect::<Vec<_>>(),
        "engine": {
            "queued_batch": stats.queued_batch,
            "queued_stream": stats.queued_stream,
            "running": stats.running,
            "sessions": stats.sessions,
            "draining": stats.draining,
        },
        "stores": {"jobs": h.jobs_backend, "artifacts": h.artifacts_backend},
        "version": env!("CARGO_PKG_VERSION"),
    });
    let code = if phase == Phase::Ready { StatusCode::OK } else { StatusCode::SERVICE_UNAVAILABLE };
    (code, Json(body)).into_response()
}

async fn root(State(h): State<Health>) -> Response {
    Json(json!({
        "model": h.served_name(),
        "server": "fv-serve",
        "version": env!("CARGO_PKG_VERSION"),
    }))
    .into_response()
}

async fn metrics_text(State(h): State<Health>) -> Response {
    let Some(handle) = &h.metrics else {
        return (StatusCode::NOT_FOUND, "metrics are disabled\n").into_response();
    };
    let engine = h.gate.engine();
    let s = engine.stats();
    metrics::gauge!("fv_engine_queued", "priority" => "batch").set(s.queued_batch as f64);
    metrics::gauge!("fv_engine_queued", "priority" => "stream").set(s.queued_stream as f64);
    metrics::gauge!("fv_engine_running").set(s.running as f64);
    metrics::gauge!("fv_engine_sessions").set(s.sessions as f64);
    metrics::gauge!("fv_ready").set(if h.phase() == Phase::Ready { 1.0 } else { 0.0 });
    handle.run_upkeep();
    ([(header::CONTENT_TYPE, "text/plain; version=0.0.4")], handle.render()).into_response()
}
