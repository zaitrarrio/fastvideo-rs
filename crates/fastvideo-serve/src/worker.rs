//! Worker role (`server.role = "worker"`, docs/serve/gateway.md §3, §5.3):
//! a GPU fv-serve behind the gateway.
//!
//! - [`token_layer`]: every route but health, `/metrics` and the signed
//!   `/files` / `/uploads` needs `x-fv-internal-token` (the gateway's
//!   shared secret); API auth itself is the gateway's (`trust-gateway`).
//!   A direct worker (`gateway.direct`, no gateway in front) needs the
//!   token only on `/fv/v1/internal/*`; its APIs use its own `auth.mode`
//!   (docs/control/gateway-less-auth.md).
//! - [`routes`][]: `/fv/v1/internal/*`:
//!
//! | Route | Behaviour |
//! |---|---|
//! | `POST /fv/v1/internal/jobs` | the dispatch envelope: fetch inputs, adopt the D1 row, submit to the engine → 202 `{id, status, worker, load}` (`load`: running, queued, capacity, queue_max after taking it); a job already held here answers 200; one held by another live worker 409 |
//! | `GET /fv/v1/internal/jobs/{id}` | `{id, status, progress}` (what a queue `wait` polls); with `?wait_s=N&since=<status>` it answers once the status differs from `since` (or after `N` s, at most 60) and D1 has it, adding the `job` (the dispatching gateway's notification, gateway.md §3.5) |
//! | `DELETE /fv/v1/internal/jobs/{id}` | cancel |
//! | `GET /fv/v1/internal/status` | worker id, pool, readiness, draining, load, `capacity` (executors), `queue_max`, caps, `build` (git sha, variant, image digest, channel) (gateway probes) |
//! | `POST /fv/v1/internal/drain`, `…/undrain` | stop / resume taking new jobs and sessions (running work finishes); the `gw_workers` row says `draining` (the autoscaler, gateway.md §8.5) |
//!
//! - [`spawn_registration`]: pod workers upsert `gw_workers` every 10 s.

// Handler helpers return a ready `Response` as their error (early return).
#![allow(clippy::result_large_err)]

use std::collections::{BTreeMap, HashSet};
use std::sync::atomic::{AtomicBool, Ordering};
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
use base64::Engine as _;
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

/// Whether a direct worker needs the internal token for `p`.
fn internal_path(p: &str) -> bool {
    p == "/fv/v1/internal" || p.starts_with("/fv/v1/internal/")
}

/// Requires the internal token on every non-open route, or with `direct`
/// only on `/fv/v1/internal/*` (the other routes authenticate clients
/// themselves).
pub fn token_layer(router: Router, token: Arc<str>, direct: bool) -> Router {
    router.layer(axum::middleware::from_fn(move |req: Request<Body>, next: Next| {
        let token = token.clone();
        async move {
            let path = req.uri().path();
            if req.method() == axum::http::Method::OPTIONS || open_path(path) || (direct && !internal_path(path)) {
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
    /// SSRF guard and limits for inputs passed through as client URLs.
    pub ingest: fastvideo_serve_kit::IngestPolicy,
    submitted: Mutex<HashSet<JobId>>,
    /// Set by `POST /fv/v1/internal/drain` (the autoscaler, docs/serve/gateway.md §8.5).
    drained: Arc<AtomicBool>,
    registration: Option<Registration>,
}

impl WorkerState {
    pub fn new(ctx: ServeCtx, gate: Arc<ServiceGate>, d1: Option<Arc<D1JobStore>>, worker_id: String, pool: Option<String>, queue_max: usize) -> Self {
        let http = reqwest::Client::builder().connect_timeout(Duration::from_secs(10)).build().unwrap_or_default();
        Self {
            ctx,
            gate,
            d1,
            http,
            worker_id,
            pool,
            queue_max,
            ingest: fastvideo_serve_kit::IngestPolicy::default(),
            submitted: Mutex::new(HashSet::new()),
            drained: Arc::default(),
            registration: None,
        }
    }

    /// Shares the drain flag with the registration (which reports it).
    pub fn with_drain(mut self, drained: Arc<AtomicBool>, registration: Option<Registration>) -> Self {
        self.drained = drained;
        self.registration = registration;
        self
    }

    /// The drain flag (shared with the registration).
    pub(crate) fn drained_flag(&self) -> &AtomicBool {
        &self.drained
    }

    /// Drained (autoscaler) or shutting down: no new jobs or sessions.
    pub fn draining(&self) -> bool {
        self.drained.load(Ordering::SeqCst) || !self.gate.admitting()
    }
}

async fn set_drain(st: &WorkerState, on: bool) -> Response {
    st.drained.store(on, Ordering::SeqCst);
    if let Some(r) = &st.registration {
        r.write(false).await;
    }
    tracing::info!(drained = on, "worker: drain state changed");
    let s = st.gate.engine().stats();
    Json(json!({"worker_id": st.worker_id, "draining": st.draining(), "running": s.running, "queued": s.queued_batch, "sessions": s.sessions}))
        .into_response()
}

/// `POST /fv/v1/internal/drain`: take no new jobs or sessions; running
/// work finishes. The `gw_workers` row reports `draining`.
async fn drain(State(st): State<Arc<WorkerState>>) -> Response {
    set_drain(&st, true).await
}

/// `POST /fv/v1/internal/undrain`.
async fn undrain(State(st): State<Arc<WorkerState>>) -> Response {
    set_drain(&st, false).await
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
/// Envelopes carry small inputs inline (base64; the gateway's
/// `inline_inputs_max_bytes`, 8 MiB by default): well above axum's 2 MB.
const ENVELOPE_MAX: usize = 64 * 1024 * 1024;

pub fn routes(st: WorkerState) -> Router {
    routes_shared(Arc::new(st))
}

/// [`routes`] over a state shared with the dispatcher socket
/// ([`crate::edge_link`]).
pub fn routes_shared(st: Arc<WorkerState>) -> Router {
    Router::new()
        .route("/fv/v1/internal/jobs", post(take).layer(axum::extract::DefaultBodyLimit::max(ENVELOPE_MAX)))
        .route("/fv/v1/internal/jobs/{id}", get(job_status).delete(job_cancel))
        .route("/fv/v1/internal/status", get(status))
        .route("/fv/v1/internal/drain", post(drain))
        .route("/fv/v1/internal/undrain", post(undrain))
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
    Json(status_json(&st)).into_response()
}

/// The body of `GET /fv/v1/internal/status` (also a dispatcher hello's caps).
pub fn status_json(st: &WorkerState) -> Value {
    let engine = st.gate.engine();
    let s = engine.stats();
    let models: Vec<Value> = engine.caps().entries().map(|e| json!({"caps": e.caps, "recipe": e.recipe})).collect();
    json!({
        "object": "fv.worker",
        "worker_id": st.worker_id,
        "pool": st.pool,
        "readiness": readiness_word(&engine.readiness()),
        // Models this worker failed (the startup capability check: a GPU
        // that cannot run the model) and why; the gateway shows them in
        // `/fv/v1/status` and dispatches nothing here.
        "failed_models": crate::health::failed_models(&engine.pool()),
        "draining": st.draining(),
        "stats": {"queued_batch": s.queued_batch, "queued_stream": s.queued_stream, "running": s.running, "sessions": s.sessions},
        // What the gateway places against: jobs run at once (executors)
        // and the batch queue limit (0: none).
        "capacity": s.executors.max(1),
        "queue_max": st.queue_max,
        "models": models,
        "version": env!("CARGO_PKG_VERSION"),
        // Git sha, build time, variant, image and channel (docs/serve/releases.md).
        "build": crate::build_info::BuildInfo::current().json(),
    })
}

fn parse_id(id: &str) -> Result<JobId, Response> {
    id.parse::<JobId>().map_err(|_| err(&ApiError::not_found(format!("job `{id}` was not found"))))
}

/// `?wait_s=N&since=<status>`: a gateway waiting for the job's next status.
#[derive(Debug, Default, serde::Deserialize)]
struct StatusWait {
    wait_s: Option<u64>,
    since: Option<String>,
}

/// The longest a status wait holds the request.
const STATUS_WAIT_MAX: Duration = Duration::from_secs(60);

async fn job_status(State(st): State<Arc<WorkerState>>, Path(id): Path<String>, axum::extract::Query(q): axum::extract::Query<StatusWait>) -> Response {
    let id = match parse_id(&id) {
        Ok(i) => i,
        Err(r) => return r,
    };
    let Some(wait) = q.wait_s else {
        return match st.ctx.jobs().get(id).await {
            Some(j) => Json(json!({"id": id.to_string(), "status": j.status().as_str(), "progress": j.progress, "worker": st.worker_id})).into_response(),
            None => err(&ApiError::not_found(format!("job `{id}` was not found"))),
        };
    };
    // A wait (the dispatching gateway, docs/serve/gateway.md §3.5): answer
    // when the status differs from `since` (at once if it already does),
    // or after `wait_s`, with the job itself, once D1 has that version (so
    // the gateway never shows a state the other replicas cannot read).
    let since = q.since.unwrap_or_default();
    let changed = |j: &Job| j.status().as_str() != since;
    let Some(mut job) = st.ctx.jobs().get(id).await else {
        return err(&ApiError::not_found(format!("job `{id}` was not found")));
    };
    if !changed(&job) {
        if let Some(mut rx) = st.ctx.jobs().watch(id) {
            let limit = Duration::from_secs(wait).min(STATUS_WAIT_MAX);
            let _ = tokio::time::timeout(limit, async {
                while rx.changed().await.is_ok() {
                    rx.borrow_and_update();
                    match st.ctx.jobs().get(id).await {
                        Some(j) if changed(&j) => return,
                        Some(_) => {}
                        None => return,
                    }
                }
            })
            .await;
        }
        match st.ctx.jobs().get(id).await {
            Some(j) => job = j,
            None => return err(&ApiError::not_found(format!("job `{id}` was not found"))),
        }
    }
    if changed(&job) {
        if let Some(d1) = &st.d1 {
            if let Err(e) = d1.settle(id).await {
                tracing::debug!(job = %id, error = %e, "worker: status wait: D1 write pending");
            }
        }
        job = st.ctx.jobs().get(id).await.unwrap_or(job);
    }
    Json(json!({"id": id.to_string(), "status": job.status().as_str(), "progress": job.progress, "worker": st.worker_id, "job": job})).into_response()
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

/// Why an input could not be put in place: a passed-through client URL
/// (the gateway sends it through the store instead: HTTP 424), or anything
/// else.
enum FetchError {
    Source(ApiError),
    Other(ApiError),
}

/// Puts one envelope input at `dst`: inline bytes, the client's URL (SSRF
/// guard of ingestion, checked against the gateway's SHA-256), the shared
/// store, or the signed URL.
async fn fetch_one(st: &WorkerState, input: &crate::gateway::dispatch::InputRef, dst: &std::path::Path) -> Result<(), FetchError> {
    let other = FetchError::Other;
    if let Some(b64) = &input.inline {
        let data = base64::engine::general_purpose::STANDARD
            .decode(b64)
            .map_err(|e| other(ApiError::invalid(format!("a dispatched input is not valid base64: {e}"))))?;
        return tokio::fs::write(dst, &data).await.map_err(|e| other(ApiError::internal(format!("writing a dispatched input: {e}"))));
    }
    if let Some(src) = &input.source {
        let r = async {
            let url = url::Url::parse(src).map_err(|e| ApiError::invalid(format!("a passed-through input URL: {e}")))?;
            let kind = input.kind.unwrap_or(fastvideo_protocol::MediaKind::Video);
            fastvideo_serve_kit::ingest::fetch_public(&url, kind, &st.ingest, dst, "input").await?;
            if let Some(want) = &input.sha256 {
                let got = crate::gateway::dispatch::sha256_file(dst).await?;
                if !got.eq_ignore_ascii_case(want) {
                    return Err(ApiError::invalid("a passed-through input changed since the gateway fetched it"));
                }
            }
            Ok(())
        }
        .await;
        match r {
            Ok(()) => return Ok(()),
            Err(e) if input.artifact.is_none() && input.url.is_empty() => {
                let _ = tokio::fs::remove_file(dst).await;
                return Err(FetchError::Source(e));
            }
            Err(e) => tracing::info!(error = %e.message, "worker: a passed-through input failed; using the store copy"),
        }
    }
    // The shared store first (R2, or a directory shared on one host):
    // no round trip through the gateway's public URL.
    if let Some(a) = &input.artifact {
        match st.ctx.artifacts().open(a).await {
            Ok(fastvideo_serve_kit::artifacts::ArtifactBody::File(p)) if tokio::fs::copy(&p, dst).await.is_ok() => return Ok(()),
            Ok(fastvideo_serve_kit::artifacts::ArtifactBody::Bytes(b)) if tokio::fs::write(dst, &b).await.is_ok() => return Ok(()),
            _ => {}
        }
    }
    if input.url.is_empty() {
        return Err(other(ApiError::internal("a dispatched input has no source")));
    }
    let resp = st
        .http
        .get(&input.url)
        .timeout(Duration::from_secs(300))
        .send()
        .await
        .map_err(|e| other(ApiError::internal(format!("fetching a dispatched input: {}", e.without_url()))))?;
    if !resp.status().is_success() {
        return Err(other(ApiError::internal(format!("fetching a dispatched input answered {}", resp.status()))));
    }
    let bytes = resp.bytes().await.map_err(|e| other(ApiError::internal(format!("reading a dispatched input: {}", e.without_url()))))?;
    tokio::fs::write(dst, &bytes).await.map_err(|e| other(ApiError::internal(format!("writing a dispatched input: {e}"))))
}

/// Puts the envelope's inputs in place (all at once: one store round trip
/// in wall time, not one per input) and points `job.resolved` at them.
async fn fetch_inputs(st: &WorkerState, env: &Envelope, job: &mut Job) -> Result<(), FetchError> {
    if env.inputs.is_empty() {
        return Ok(());
    }
    let dir = st.ctx.inputs_dir(job.id);
    tokio::fs::create_dir_all(&dir).await.map_err(|e| FetchError::Other(ApiError::internal(format!("inputs dir: {e}"))))?;
    let t0 = std::time::Instant::now();
    let fetches = env.inputs.iter().enumerate().map(|(i, input)| {
        let dir = &dir;
        async move {
            let name = input.path.file_name().and_then(|n| n.to_str()).map(str::to_owned).unwrap_or_else(|| format!("input-{i}"));
            let dst = dir.join(format!("{i}-{name}"));
            fetch_one(st, input, &dst).await.map(|()| (input.path.clone(), dst))
        }
    });
    let map: BTreeMap<PathBuf, PathBuf> = futures::future::try_join_all(fetches).await?.into_iter().collect();
    let fetch_s = t0.elapsed().as_secs_f64();
    metrics::histogram!("fv_worker_input_fetch_seconds").record(fetch_s);
    let via = |w: &str| env.inputs.iter().filter(|i| i.via() == w).count();
    tracing::info!(job = %job.id, inputs = map.len(), inline = via("inline"), source = via("source"), store = via("store"),
        fetch_ms = (fetch_s * 1e3) as u64, "worker: dispatched inputs in place");
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
    take_envelope(&st, env, false, None).await
}

/// Takes a dispatched job (the gateway's `POST /fv/v1/internal/jobs`, or a
/// dispatcher push): 202 taken, 200 already held here, else the refusal.
/// `takeover`: adopt even if another worker's heartbeat on the row is fresh
/// (a dispatcher re-dispatch after it declared that worker lost). `lease`:
/// the dispatcher's fencing token (docs/serve/gateway-cloudflare.md, phase
/// 2): the job is held and started at once and its row written behind,
/// every write conditional on the lease (`D1JobStore::adopt_leased`).
pub async fn take_envelope(st: &Arc<WorkerState>, env: Envelope, takeover: bool, lease: Option<u64>) -> Response {
    let id = env.job.id;
    tracing::debug!(job = %id, "worker: envelope received");
    if st.submitted.lock().unwrap_or_else(|p| p.into_inner()).contains(&id) {
        let s = st.ctx.jobs().get(id).await.map(|j| j.status().as_str().to_owned()).unwrap_or_default();
        return (StatusCode::OK, Json(json!({"id": id.to_string(), "status": s, "worker": st.worker_id, "duplicate": true}))).into_response();
    }
    if st.draining() {
        return err(&ApiError::loading("this worker is draining").with_retry_after(5));
    }
    if let Err(e) = st.ctx.engine().admit() {
        return err(&e);
    }
    let s = st.gate.engine().stats();
    if st.queue_max > 0 && s.queued_batch >= st.queue_max {
        return err(&ApiError::queue_full(format!("worker queue is full ({} queued)", s.queued_batch)).with_retry_after(5));
    }
    let mut job = env.job.clone();
    if let Err(e) = fetch_inputs(st, &env, &mut job).await {
        let _ = tokio::fs::remove_dir_all(st.ctx.inputs_dir(id)).await;
        return match e {
            // The gateway retries with the input in the store.
            FetchError::Source(e) => (
                StatusCode::FAILED_DEPENDENCY,
                Json(json!({"error": {"kind": "invalid_request", "message": format!("a passed-through input: {}", e.message)}})),
            )
                .into_response(),
            FetchError::Other(e) => err(&e),
        };
    }
    // Inputs are in place: the rest is the store write and the GPU queue
    // (fal `timings.dispatch` ends here, `timings.wait` starts).
    job.dispatched_at = Some(st.ctx.now());
    let t_adopt = std::time::Instant::now();
    let adopted = match (&st.d1, lease) {
        (Some(d1), Some(l)) if l > 0 => d1.adopt_leased(job, l),
        (Some(d1), _) => d1.adopt_with(job, takeover).await,
        (None, _) => st.ctx.jobs().insert(job.clone()).await.map(|_| job),
    };
    metrics::histogram!("fv_worker_adopt_seconds").record(t_adopt.elapsed().as_secs_f64());
    tracing::debug!(job = %id, "worker: adopted");
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
    cleanup_inputs(st, id, &env);
    tracing::info!(job = %id, attempt = env.attempt, pool = ?env.pool, "worker: took a dispatched job");
    let status = st.ctx.jobs().get(id).await.map(|j| j.status().as_str().to_owned()).unwrap_or_else(|| "queued".into());
    // The load right after taking it (this job included): the gateway
    // places the next job of a burst on it without waiting for a probe.
    let s = st.gate.engine().stats();
    let load = json!({"running": s.running, "queued": s.queued_batch + s.queued_stream, "capacity": s.executors.max(1), "queue_max": st.queue_max});
    (StatusCode::ACCEPTED, Json(json!({"id": id.to_string(), "status": status, "worker": st.worker_id, "load": load}))).into_response()
}

/// A pod worker's registration in `gw_workers`.
#[derive(Clone)]
pub struct Registration {
    pub db: D1Client,
    pub pool: String,
    pub worker_id: String,
    pub url: String,
    pub gate: Arc<ServiceGate>,
    /// The drain flag of `POST /fv/v1/internal/drain`.
    pub drained: Arc<AtomicBool>,
}

impl Registration {
    /// Upserts the row now (`draining` overrides the state).
    pub async fn write(&self, draining: bool) {
        let engine = self.gate.engine();
        let state = if draining || !self.gate.admitting() || self.drained.load(Ordering::SeqCst) {
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
