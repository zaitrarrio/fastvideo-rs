//! Gateway-only HTTP: the aggregated `/fv/v1/capabilities` body, the pool
//! metrics `GET /fv/v1/gateway/pools` (admin token), and `/ping`,
//! `/health`, `/healthz`, `/`, `/metrics` reflecting pool health
//! (docs/serve/gateway.md §4, §6, §7).

use std::sync::Arc;

use axum::extract::State;
use axum::http::{header, HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::get;
use axum::{Json, Router};
use fastvideo_serve_kit::AdminToken;
use serde_json::{json, Value};

use super::Gateway;

/// The `/fv/v1/capabilities` body in gateway mode: the native shape
/// (`models`, `tiers`, `aliases`, `readiness`) plus `pools` and each
/// model's `pools`.
pub fn capabilities(gw: &Gateway) -> Value {
    let cat = gw.catalog();
    let models: Vec<Value> = cat
        .table
        .entries()
        .map(|e| {
            let pools: Vec<&str> = cat.pools_of.get(&e.caps.id).map(|v| v.iter().filter_map(|i| gw.pools.get(*i)).map(|p| p.id()).collect()).unwrap_or_default();
            json!({"caps": e.caps, "recipe": e.recipe, "executors": e.executors, "pools": pools})
        })
        .collect();
    let tiers: Vec<Value> = cat.table.tier_bindings().map(|b| serde_json::to_value(b).unwrap_or(Value::Null)).collect();
    json!({
        "object": "fv.capabilities",
        "models": models,
        "tiers": tiers,
        "aliases": cat.aliases,
        "readiness": if gw.any_available() { "ready" } else { "unavailable" },
        "gateway": true,
        "pools": pools(gw),
    })
}

/// Per-pool state for `/fv/v1/capabilities` and `/healthz`.
pub fn pools(gw: &Gateway) -> Vec<Value> {
    gw.pools
        .iter()
        .map(|p| {
            let available = p.available();
            let st = p.lock();
            let models: Vec<String> = st.live_caps.as_ref().unwrap_or(&p.static_caps).iter().map(|(c, _)| c.id.0.clone()).collect();
            json!({
                "id": p.id(),
                "kind": p.cfg.kind,
                "endpoint_id": p.cfg.endpoint_id,
                "available": available,
                "caps": if st.live_caps.is_some() { "live" } else { "static" },
                "models": models,
                "workers": st.workers.values().collect::<Vec<_>>(),
                "queued": st.queued,
                "running": st.running,
                "streams": st.streams,
                "error": st.last_error,
            })
        })
        .collect()
}

#[derive(Clone)]
struct HealthState {
    gw: Arc<Gateway>,
    admin: Arc<AdminToken>,
    metrics: Option<metrics_exporter_prometheus::PrometheusHandle>,
    root_model: Option<String>,
}

/// `/ping`, `/health`, `/healthz`, `/`, `/metrics` and `/fv/v1/gateway/pools`.
/// `root_model`: what `GET /` names (FastWan's model when FastWan is mounted).
pub fn routes(
    gw: Arc<Gateway>,
    admin: Arc<AdminToken>,
    metrics: Option<metrics_exporter_prometheus::PrometheusHandle>,
    root_model: Option<String>,
) -> Router {
    Router::new()
        .route("/ping", get(ping))
        .route("/health", get(health))
        .route("/healthz", get(healthz))
        .route("/", get(root))
        .route("/metrics", get(metrics_text))
        .route("/fv/v1/gateway/pools", get(pools_route))
        .with_state(HealthState { gw, admin, metrics, root_model })
}

fn phase(gw: &Gateway) -> (&'static str, StatusCode) {
    if !gw.admitting() {
        ("draining", StatusCode::SERVICE_UNAVAILABLE)
    } else if gw.any_available() {
        ("ready", StatusCode::OK)
    } else {
        ("unavailable", StatusCode::SERVICE_UNAVAILABLE)
    }
}

async fn ping(State(h): State<HealthState>) -> Response {
    match phase(&h.gw) {
        ("ready", _) => (StatusCode::OK, Json(json!({"status": "healthy"}))).into_response(),
        (s, c) => (c, Json(json!({"status": s}))).into_response(),
    }
}

async fn health(State(h): State<HealthState>) -> Response {
    let (s, c) = phase(&h.gw);
    let (status, state) = match s {
        "ready" => ("ok", "AVAILABLE"),
        "draining" => ("draining", "DRAINING"),
        _ => ("unavailable", "UNAVAILABLE"),
    };
    (c, Json(json!({"status": status, "model_loaded": c == StatusCode::OK, "state": state, "gateway": true}))).into_response()
}

async fn healthz(State(h): State<HealthState>) -> Response {
    let (s, c) = phase(&h.gw);
    let cat = h.gw.catalog();
    (
        c,
        Json(json!({
            "state": s,
            "gateway": true,
            "models": cat.table.models().map(|m| m.id.0.clone()).collect::<Vec<_>>(),
            "pools": pools(&h.gw),
            "stores": {"jobs": "d1"},
            "version": env!("CARGO_PKG_VERSION"),
        })),
    )
        .into_response()
}

async fn root(State(h): State<HealthState>) -> Response {
    let cat = h.gw.catalog();
    let model = h.root_model.clone().or_else(|| cat.table.models().next().map(|m| m.served_names.first().cloned().unwrap_or_else(|| m.id.0.clone())));
    Json(json!({"model": model, "server": "fv-serve", "role": "gateway", "version": env!("CARGO_PKG_VERSION")})).into_response()
}

async fn metrics_text(State(h): State<HealthState>) -> Response {
    let Some(handle) = &h.metrics else {
        return (StatusCode::NOT_FOUND, "metrics are disabled\n").into_response();
    };
    metrics::gauge!("fv_ready").set(if phase(&h.gw).0 == "ready" { 1.0 } else { 0.0 });
    handle.run_upkeep();
    ([(header::CONTENT_TYPE, "text/plain; version=0.0.4")], handle.render()).into_response()
}

async fn pools_route(State(h): State<HealthState>, headers: HeaderMap) -> Response {
    if !h.admin.check_headers(&headers) {
        return (StatusCode::UNAUTHORIZED, Json(json!({"error": {"kind": "unauthorized", "message": "the admin token is required"}}))).into_response();
    }
    Json(json!({"object": "fv.gateway.pools", "pools": h.gw.metrics(), "state": pools(&h.gw)})).into_response()
}
