//! Worker role (`server.role = "worker"`, docs/serve/gateway.md §3, §5.3):
//! a GPU fv-serve behind the gateway.
//!
//! - [`token_layer`]: every route but health, `/metrics` and the signed
//!   `/files` / `/uploads` needs `x-fv-internal-token` (the gateway's
//!   shared secret); API auth itself is the gateway's (`trust-gateway`).
//! - [`routes`]: `/fv/v1/internal/*`:
//!
//! | Route | Behaviour |
//! |---|---|
//! | `POST /fv/v1/internal/jobs` | the dispatch envelope: fetch inputs, adopt the D1 row, submit to the engine → 202 `{id, status, worker}`; a job already held here answers 200; one held by another live worker 409 |
//! | `GET /fv/v1/internal/jobs/{id}` | `{id, status, progress}` (what a queue `wait` polls) |
//! | `DELETE /fv/v1/internal/jobs/{id}` | cancel |
//! | `GET /fv/v1/internal/status` | worker id, pool, readiness, draining, load, caps (gateway probes) |
//!
//! - [`spawn_registration`]: pod workers upsert `gw_workers` every 10 s.

// Handler helpers return a ready `Response` as their error (early return).
#![allow(clippy::result_large_err)]

use std::collections::{BTreeMap, HashSet};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use axum::body::Body;
use axum::extract::{Path, State};
use axum::http::{Request, StatusCode};
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use fastvideo_engine_service::Readiness;
use fastvideo_protocol::{ApiError, Job, JobId, StoreError};
use fastvideo_serve_kit::d1::{D1Client, Stmt};
use fastvideo_serve_kit::events::{apply_event, cancel_job, JobEvent};
use fastvideo_serve_kit::{D1JobStore, ServeCtx};
use serde_json::{json, Value};

use crate::gate::ServiceGate;
use crate::gateway::dispatch::Envelope;
use crate::gateway::schema::now_ms;
use crate::gateway::TOKEN_HEADER;

/// Paths a worker serves without the internal token.
fn open_path(p: &str) -> bool {
    matches!(p, "/ping" | "/health" | "/healthz" | "/" | "/metrics") || p.starts_with("/files/") || p.starts_with("/uploads/")
}

fn ct_eq(a: &[u8], b: &[u8]) -> bool {
    a.len() == b.len() && a.iter().zip(b).fold(0u8, |acc, (x, y)| acc | (x ^ y)) == 0
}

/// Requires the internal token on every non-open route.
pub fn token_layer(router: Router, token: Arc<str>) -> Router {
    router.layer(axum::middleware::from_fn(move |req: Request<Body>, next: Next| {
        let token = token.clone();
        async move {
            if req.method() == axum::http::Method::OPTIONS || open_path(req.uri().path()) {
                return next.run(req).await;
            }
            let ok = req.headers().get(TOKEN_HEADER).is_some_and(|v| ct_eq(v.as_bytes(), token.as_bytes()));
            if ok {
                next.run(req).await
            } else {
                (
                    StatusCode::UNAUTHORIZED,
                    Json(json!({"error": {"kind": "unauthorized", "message": "this is a gateway worker: requests need the internal token"}})),
                )
                    .into_response()
            }
        }
    }))
}

/// What the internal routes need.
pub struct WorkerState {
    pub ctx: ServeCtx,
    pub gate: Arc<ServiceGate>,
    pub d1: Option<Arc<D1JobStore>>,
    pub http: reqwest::Client,
    pub worker_id: String,
    pub pool: Option<String>,
    pub queue_max: usize,
    submitted: Mutex<HashSet<JobId>>,
}

impl WorkerState {
    pub fn new(ctx: ServeCtx, gate: Arc<ServiceGate>, d1: Option<Arc<D1JobStore>>, worker_id: String, pool: Option<String>, queue_max: usize) -> Arc<Self> {
        let http = reqwest::Client::builder().connect_timeout(Duration::from_secs(10)).build().unwrap_or_default();
        Arc::new(Self { ctx, gate, d1, http, worker_id, pool, queue_max, submitted: Mutex::new(HashSet::new()) })
    }
}

fn err(e: &ApiError) -> Response {
    let kind = serde_json::to_value(e.kind).unwrap_or(Value::Null);
    let mut r = (StatusCode::from_u16(e.http_status()).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR), Json(json!({"error": {"kind": kind, "message": e.message}}))).into_response();
    if let Some(s) = e.retry_after_s {
        r.headers_mut().insert("retry-after", s.into());
    }
    r
}

/// `/fv/v1/internal/*`.
pub fn routes(st: Arc<WorkerState>) -> Router {
    Router::new()
        .route("/fv/v1/internal/jobs", post(take))
        .route("/fv/v1/internal/jobs/{id}", get(job_status).delete(job_cancel))
        .route("/fv/v1/internal/status", get(status))
        .with_state(st)
}

fn readiness_word(r: &Readiness) -> &'static str {
    match r {
        Readiness::Ready => "ready",
        Readiness::Loading { .. } => "loading",
        Readiness::Failed(_) => "failed",
    }
}

async fn status(State(st): State<Arc<WorkerState>>) -> Response {
    let engine = st.gate.engine();
    let s = engine.stats();
    let models: Vec<Value> = engine.caps().entries().map(|e| json!({"caps": e.caps, "recipe": e.recipe})).collect();
    Json(json!({
        "object": "fv.worker",
        "worker_id": st.worker_id,
        "pool": st.pool,
        "readiness": readiness_word(&engine.readiness()),
        "draining": !st.gate.admitting(),
        "stats": {"queued_batch": s.queued_batch, "queued_stream": s.queued_stream, "running": s.running, "sessions": s.sessions},
        "models": models,
        "version": env!("CARGO_PKG_VERSION"),
    }))
    .into_response()
}

fn parse_id(id: &str) -> Result<JobId, Response> {
    id.parse::<JobId>().map_err(|_| err(&ApiError::not_found(format!("job `{id}` was not found"))))
}

async fn job_status(State(st): State<Arc<WorkerState>>, Path(id): Path<String>) -> Response {
    let id = match parse_id(&id) {
        Ok(i) => i,
        Err(r) => return r,
    };
    match st.ctx.jobs().get(id).await {
        Some(j) => Json(json!({"id": id.to_string(), "status": j.status().as_str(), "progress": j.progress, "worker": st.worker_id})).into_response(),
        None => err(&ApiError::not_found(format!("job `{id}` was not found"))),
    }
}

async fn job_cancel(State(st): State<Arc<WorkerState>>, Path(id): Path<String>) -> Response {
    let id = match parse_id(&id) {
        Ok(i) => i,
        Err(r) => return r,
    };
    match cancel_job(&st.ctx, id).await {
        Ok(j) => Json(json!({"id": id.to_string(), "status": j.status().as_str(), "cancel_requested": j.cancel_requested})).into_response(),
        Err(e) if e.kind == fastvideo_protocol::ErrorKind::AlreadyCompleted => {
            let s = st.ctx.jobs().get(id).await.map(|j| j.status().as_str().to_owned()).unwrap_or_default();
            Json(json!({"id": id.to_string(), "status": s})).into_response()
        }
        Err(e) => err(&e),
    }
}

/// Downloads the envelope's inputs and points `job.resolved` at them.
async fn fetch_inputs(st: &WorkerState, env: &Envelope, job: &mut Job) -> Result<(), ApiError> {
    if env.inputs.is_empty() {
        return Ok(());
    }
    let dir = st.ctx.inputs_dir(job.id);
    tokio::fs::create_dir_all(&dir).await.map_err(|e| ApiError::internal(format!("inputs dir: {e}")))?;
    let mut map: BTreeMap<PathBuf, PathBuf> = BTreeMap::new();
    for (i, input) in env.inputs.iter().enumerate() {
        let name = input.path.file_name().and_then(|n| n.to_str()).map(str::to_owned).unwrap_or_else(|| format!("input-{i}"));
        let dst = dir.join(format!("{i}-{name}"));
        let resp = st
            .http
            .get(&input.url)
            .timeout(Duration::from_secs(300))
            .send()
            .await
            .map_err(|e| ApiError::internal(format!("fetching a dispatched input: {}", e.without_url())))?;
        if !resp.status().is_success() {
            return Err(ApiError::internal(format!("fetching a dispatched input answered {}", resp.status())));
        }
        let bytes = resp.bytes().await.map_err(|e| ApiError::internal(format!("reading a dispatched input: {}", e.without_url())))?;
        tokio::fs::write(&dst, &bytes).await.map_err(|e| ApiError::internal(format!("writing a dispatched input: {e}")))?;
        map.insert(input.path.clone(), dst);
    }
    let r = &mut job.resolved;
    let swap = |p: &mut PathBuf| {
        if let Some(n) = map.get(p) {
            *p = n.clone();
        }
    };
    r.keyframes.iter_mut().for_each(|(_, p)| swap(p));
    r.references.iter_mut().for_each(|(_, p)| swap(p));
    if let Some((_, p)) = r.audio_in.as_mut() {
        swap(p);
    }
    Ok(())
}

/// Deletes the dispatched input artifacts once the job ends.
fn cleanup_inputs(st: &Arc<WorkerState>, id: JobId, env: &Envelope) {
    let arts: Vec<_> = env.inputs.iter().filter_map(|i| i.artifact.clone()).collect();
    if arts.is_empty() {
        return;
    }
    let st = st.clone();
    tokio::spawn(async move {
        if let Some(mut rx) = st.ctx.jobs().watch(id) {
            while !rx.borrow_and_update().state.is_terminal() {
                if rx.changed().await.is_err() {
                    break;
                }
            }
        }
        for a in &arts {
            st.ctx.artifacts().delete(a).await;
        }
    });
}

async fn take(State(st): State<Arc<WorkerState>>, Json(env): Json<Envelope>) -> Response {
    let id = env.job.id;
    if st.submitted.lock().unwrap_or_else(|p| p.into_inner()).contains(&id) {
        let s = st.ctx.jobs().get(id).await.map(|j| j.status().as_str().to_owned()).unwrap_or_default();
        return (StatusCode::OK, Json(json!({"id": id.to_string(), "status": s, "worker": st.worker_id, "duplicate": true}))).into_response();
    }
    if let Err(e) = st.ctx.engine().admit() {
        return err(&e);
    }
    let s = st.gate.engine().stats();
    if st.queue_max > 0 && s.queued_batch >= st.queue_max {
        return err(&ApiError::queue_full(format!("worker queue is full ({} queued)", s.queued_batch)).with_retry_after(5));
    }
    let mut job = env.job.clone();
    if let Err(e) = fetch_inputs(&st, &env, &mut job).await {
        let _ = tokio::fs::remove_dir_all(st.ctx.inputs_dir(id)).await;
        return err(&e);
    }
    let adopted = match &st.d1 {
        Some(d1) => d1.adopt(job).await,
        None => st.ctx.jobs().insert(job.clone()).await.map(|_| job),
    };
    let job = match adopted {
        Ok(j) => j,
        Err(StoreError::AlreadyExists(_)) => {
            let _ = tokio::fs::remove_dir_all(st.ctx.inputs_dir(id)).await;
            return err(&ApiError::conflict("the job is finished, cancelled, or held by another worker"));
        }
        Err(e) => return err(&e.into()),
    };
    st.submitted.lock().unwrap_or_else(|p| p.into_inner()).insert(id);
    if let Err(e) = st.ctx.engine().submit(&job).await {
        tracing::warn!(job = %id, error = %e.message, "worker: engine refused a dispatched job");
        let _ = apply_event(&st.ctx, id, JobEvent::Failed(e.clone())).await;
        return err(&e);
    }
    cleanup_inputs(&st, id, &env);
    tracing::info!(job = %id, attempt = env.attempt, pool = ?env.pool, "worker: took a dispatched job");
    let status = st.ctx.jobs().get(id).await.map(|j| j.status().as_str().to_owned()).unwrap_or_else(|| "queued".into());
    (StatusCode::ACCEPTED, Json(json!({"id": id.to_string(), "status": status, "worker": st.worker_id}))).into_response()
}

/// A pod worker's registration in `gw_workers`.
#[derive(Clone)]
pub struct Registration {
    pub db: D1Client,
    pub pool: String,
    pub worker_id: String,
    pub url: String,
    pub gate: Arc<ServiceGate>,
}

impl Registration {
    /// Upserts the row now (`draining` overrides the state).
    pub async fn write(&self, draining: bool) {
        let engine = self.gate.engine();
        let state = if draining || !self.gate.admitting() {
            "draining"
        } else {
            readiness_word(&engine.readiness())
        };
        let s = engine.stats();
        let r = self
            .db
            .query(Stmt::new(
                "INSERT INTO gw_workers (pool, worker_id, url, state, running, sessions, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?) \
                 ON CONFLICT(pool, worker_id) DO UPDATE SET url = excluded.url, state = excluded.state, running = excluded.running, \
                 sessions = excluded.sessions, updated_at = excluded.updated_at",
                vec![
                    json!(self.pool),
                    json!(self.worker_id),
                    json!(self.url),
                    json!(state),
                    json!(s.running),
                    json!(s.sessions),
                    json!(now_ms()),
                ],
            ))
            .await;
        if let Err(e) = r {
            tracing::warn!(error = %e, "worker: gateway registration failed");
        }
    }
}

/// Registers every 10 s until the handle is aborted.
pub fn spawn_registration(reg: Registration) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        if let Err(e) = crate::gateway::schema::migrate(&reg.db).await {
            tracing::warn!(error = %e, "worker: gateway tables");
        }
        tracing::info!(pool = %reg.pool, url = %reg.url, "worker: registering with the gateway pool");
        loop {
            reg.write(false).await;
            tokio::time::sleep(Duration::from_secs(10)).await;
        }
    })
}
