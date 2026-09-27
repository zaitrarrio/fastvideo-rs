//! A local Runpod queue simulator: the worker-side URL shapes of
//! `runpod-python`'s `tests/test_serverless/local_sim/localhost.py`
//! (research-deploy §1.1) plus the public `/run`, `/status`, `/stream`,
//! `/cancel` API (§1.2), all in memory.
//!
//! Worker side (auth: `Authorization: <api key>`, raw):
//!
//! | Route | Behaviour |
//! |---|---|
//! | `GET /v2/{ep}/job-take/{worker}?job_in_progress=` | 200 `{id,input}` or 204 after a short long-poll |
//! | `GET /v2/{ep}/job-take-batch/{worker}?batch_size=` | 200 `[{id,input},…]` or 204 |
//! | `GET /v2/{ep}/job-stop/{worker}` | 200 `{"jobsToStop":[…]}` or 204 |
//! | `POST /v2/{ep}/job-done/{worker}/{job}?isStream=` | progress (`status: IN_PROGRESS`) or the result |
//! | `POST /v2/{ep}/job-stream/{worker}/{job}` | a stream chunk |
//! | `GET /v2/{ep}/ping/{worker}?job_id=&runpod_version=` | heartbeat |
//!
//! Client side (no auth): `POST /v2/{ep}/run`, `GET /v2/{ep}/status/{id}`,
//! `GET /v2/{ep}/stream/{id}`, `POST /v2/{ep}/cancel/{id}`, `GET /v2/{ep}/health`.
//!
//! Faults can be scripted ([`Sim::script_take`], [`Sim::fail_next_done`]) to
//! exercise 429 / 400 back-off and job-done retries.

use std::collections::{BTreeMap, VecDeque};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use axum::body::Bytes;
use axum::extract::{Path, Query, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::Serialize;
use serde_json::{json, Value};
use tokio::sync::Notify;

/// A job's state in the simulator.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum SimStatus {
    InQueue,
    InProgress,
    Completed,
    Failed,
    Cancelled,
}

/// Everything the simulator saw of one job.
#[derive(Clone, Debug, Serialize)]
pub struct SimJob {
    pub id: String,
    pub input: Value,
    pub status: SimStatus,
    pub worker: Option<String>,
    pub progress: Vec<Value>,
    pub stream: Vec<Value>,
    pub output: Option<Value>,
    /// The raw `error` string (JSON text of the error report).
    pub error: Option<String>,
    /// `isStream` of the final job-done.
    pub final_is_stream: Option<bool>,
    /// job-done POSTs received (including rejected ones).
    pub done_posts: u32,
    pub x_request_ids: Vec<String>,
}

impl SimJob {
    /// The parsed error report.
    pub fn error_report(&self) -> Option<Value> {
        self.error.as_deref().and_then(|e| serde_json::from_str(e).ok())
    }
    pub fn is_terminal(&self) -> bool {
        matches!(self.status, SimStatus::Completed | SimStatus::Failed | SimStatus::Cancelled)
    }
}

/// One heartbeat.
#[derive(Clone, Debug, Serialize)]
pub struct Ping {
    pub worker: String,
    pub job_ids: Vec<String>,
    pub version: String,
}

#[derive(Default)]
struct State_ {
    queue: VecDeque<String>,
    jobs: BTreeMap<String, SimJob>,
    to_stop: Vec<String>,
    pings: Vec<Ping>,
    take_script: VecDeque<u16>,
    done_failures: u32,
    takes: Vec<bool>,
    unauthorized: u32,
    next_id: u64,
}

/// The simulator. Cheap to clone.
#[derive(Clone)]
pub struct Sim {
    st: Arc<Mutex<State_>>,
    notify: Arc<Notify>,
    api_key: String,
    endpoint: String,
    long_poll: Duration,
}

impl std::fmt::Debug for Sim {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Sim").field("endpoint", &self.endpoint).finish_non_exhaustive()
    }
}

type Q = Query<BTreeMap<String, String>>;

impl Sim {
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            st: Arc::default(),
            notify: Arc::default(),
            api_key: api_key.into(),
            endpoint: "sim-ep".into(),
            long_poll: Duration::from_millis(100),
        }
    }

    /// How long job-take / job-stop hold an empty poll open.
    pub fn with_long_poll(mut self, d: Duration) -> Self {
        self.long_poll = d;
        self
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, State_> {
        self.st.lock().unwrap_or_else(|p| p.into_inner())
    }

    /// The `RUNPOD_*` variables a worker needs to talk to this simulator at
    /// `base` (e.g. `http://127.0.0.1:PORT`; any host for `RouterTransport`).
    pub fn env(&self, base: &str, worker: &str, ping_ms: u64) -> Vec<(String, String)> {
        let ep = &self.endpoint;
        [
            ("RUNPOD_WEBHOOK_GET_JOB", format!("{base}/v2/{ep}/job-take/$ID?gpu=SIM")),
            ("RUNPOD_WEBHOOK_POST_OUTPUT", format!("{base}/v2/{ep}/job-done/$RUNPOD_POD_ID/$ID?gpu=SIM")),
            ("RUNPOD_WEBHOOK_POST_STREAM", format!("{base}/v2/{ep}/job-stream/$RUNPOD_POD_ID/$ID?gpu=SIM")),
            ("RUNPOD_WEBHOOK_PING", format!("{base}/v2/{ep}/ping/$RUNPOD_POD_ID")),
            ("RUNPOD_PING_INTERVAL", ping_ms.to_string()),
            ("RUNPOD_AI_API_KEY", self.api_key.clone()),
            ("RUNPOD_POD_ID", worker.to_owned()),
            ("RUNPOD_ENDPOINT_ID", ep.clone()),
        ]
        .into_iter()
        .map(|(k, v)| (k.to_owned(), v))
        .collect()
    }

    /// Queues a job; returns its id.
    pub fn submit(&self, input: Value) -> String {
        let id = {
            let mut st = self.lock();
            st.next_id += 1;
            let id = format!("sim-{}", st.next_id);
            st.jobs.insert(
                id.clone(),
                SimJob {
                    id: id.clone(),
                    input,
                    status: SimStatus::InQueue,
                    worker: None,
                    progress: Vec::new(),
                    stream: Vec::new(),
                    output: None,
                    error: None,
                    final_is_stream: None,
                    done_posts: 0,
                    x_request_ids: Vec::new(),
                },
            );
            st.queue.push_back(id.clone());
            id
        };
        self.notify.notify_waiters();
        id
    }

    /// Client cancel: the job is CANCELLED and its worker told to stop.
    pub fn cancel(&self, id: &str) {
        {
            let mut st = self.lock();
            st.queue.retain(|q| q != id);
            if let Some(j) = st.jobs.get_mut(id) {
                if !j.is_terminal() {
                    let was_running = j.status == SimStatus::InProgress;
                    j.status = SimStatus::Cancelled;
                    if was_running {
                        st.to_stop.push(id.to_owned());
                    }
                }
            }
        }
        self.notify.notify_waiters();
    }

    /// Forces the next job-take answers (e.g. `[429, 400]`).
    pub fn script_take(&self, statuses: &[u16]) {
        self.lock().take_script.extend(statuses);
    }

    /// The next `n` job-done / progress POSTs answer 500.
    pub fn fail_next_done(&self, n: u32) {
        self.lock().done_failures += n;
    }

    pub fn job(&self, id: &str) -> Option<SimJob> {
        self.lock().jobs.get(id).cloned()
    }

    pub fn pings(&self) -> Vec<Ping> {
        self.lock().pings.clone()
    }

    /// `job_in_progress` of every take so far.
    pub fn takes(&self) -> Vec<bool> {
        self.lock().takes.clone()
    }

    pub fn unauthorized(&self) -> u32 {
        self.lock().unauthorized
    }

    /// Waits until the worker posted `id`'s final job-done (or `timeout`).
    pub async fn wait_terminal(&self, id: &str, timeout: Duration) -> Option<SimJob> {
        let end = tokio::time::Instant::now() + timeout;
        loop {
            let n = self.notify.notified();
            if let Some(j) = self.job(id).filter(|j| j.final_is_stream.is_some()) {
                return Some(j);
            }
            if tokio::time::timeout_at(end, n).await.is_err() {
                return self.job(id);
            }
        }
    }

    fn authorized(&self, h: &HeaderMap) -> bool {
        let ok = h.get("authorization").and_then(|v| v.to_str().ok()) == Some(self.api_key.as_str());
        if !ok {
            self.lock().unauthorized += 1;
        }
        ok
    }

    /// The router (serve it on TCP, or drive it with `RouterTransport`).
    pub fn router(&self) -> Router {
        Router::new()
            .route("/v2/{ep}/job-take/{worker}", get(take))
            .route("/v2/{ep}/job-take-batch/{worker}", get(take_batch))
            .route("/v2/{ep}/job-stop/{worker}", get(stop))
            .route("/v2/{ep}/job-done/{worker}/{job}", post(done))
            .route("/v2/{ep}/job-stream/{worker}/{job}", post(stream))
            .route("/v2/{ep}/ping/{worker}", get(ping))
            .route("/v2/{ep}/run", post(run))
            .route("/v2/{ep}/status/{id}", get(status))
            .route("/v2/{ep}/stream/{id}", get(client_stream))
            .route("/v2/{ep}/cancel/{id}", post(client_cancel))
            .route("/v2/{ep}/health", get(health))
            .with_state(self.clone())
    }

    /// Serves on `addr` (e.g. `127.0.0.1:0`); returns the base URL.
    pub async fn serve(&self, addr: &str) -> std::io::Result<(String, tokio::task::JoinHandle<()>)> {
        let l = tokio::net::TcpListener::bind(addr).await?;
        let base = format!("http://{}", l.local_addr()?);
        let r = self.router();
        let h = tokio::spawn(async move {
            let _ = axum::serve(l, r).await;
        });
        Ok((base, h))
    }

    fn pop(&self, worker: &str, n: usize) -> Vec<Value> {
        let mut st = self.lock();
        let mut out = Vec::new();
        while out.len() < n {
            let Some(id) = st.queue.pop_front() else { break };
            if let Some(j) = st.jobs.get_mut(&id) {
                j.status = SimStatus::InProgress;
                j.worker = Some(worker.to_owned());
                out.push(json!({"id": j.id, "input": j.input, "webhook": null}));
            }
        }
        out
    }

    async fn take_n(&self, worker: &str, q: &BTreeMap<String, String>, n: usize, batch: bool) -> Response {
        {
            let mut st = self.lock();
            st.takes.push(q.get("job_in_progress").map(String::as_str) == Some("1"));
            if let Some(code) = st.take_script.pop_front() {
                return StatusCode::from_u16(code).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR).into_response();
            }
        }
        let end = tokio::time::Instant::now() + self.long_poll;
        loop {
            let notified = self.notify.notified();
            let jobs = self.pop(worker, n);
            if !jobs.is_empty() {
                return if batch { Json(Value::Array(jobs)).into_response() } else { Json(jobs[0].clone()).into_response() };
            }
            if tokio::time::timeout_at(end, notified).await.is_err() {
                return StatusCode::NO_CONTENT.into_response();
            }
        }
    }
}

async fn take(State(s): State<Sim>, Path((_ep, worker)): Path<(String, String)>, Query(q): Q, h: HeaderMap) -> Response {
    if !s.authorized(&h) {
        return StatusCode::UNAUTHORIZED.into_response();
    }
    s.take_n(&worker, &q, 1, false).await
}

async fn take_batch(State(s): State<Sim>, Path((_ep, worker)): Path<(String, String)>, Query(q): Q, h: HeaderMap) -> Response {
    if !s.authorized(&h) {
        return StatusCode::UNAUTHORIZED.into_response();
    }
    let n = q.get("batch_size").and_then(|v| v.parse().ok()).unwrap_or(1usize).max(1);
    s.take_n(&worker, &q, n, true).await
}

async fn stop(State(s): State<Sim>, h: HeaderMap) -> Response {
    if !s.authorized(&h) {
        return StatusCode::UNAUTHORIZED.into_response();
    }
    let end = tokio::time::Instant::now() + s.long_poll;
    loop {
        let notified = s.notify.notified();
        let ids: Vec<String> = std::mem::take(&mut s.lock().to_stop);
        if !ids.is_empty() {
            return Json(json!({"jobsToStop": ids})).into_response();
        }
        if tokio::time::timeout_at(end, notified).await.is_err() {
            return StatusCode::NO_CONTENT.into_response();
        }
    }
}

async fn done(
    State(s): State<Sim>,
    Path((_ep, _worker, job)): Path<(String, String, String)>,
    Query(q): Q,
    h: HeaderMap,
    body: Bytes,
) -> Response {
    if !s.authorized(&h) {
        return StatusCode::UNAUTHORIZED.into_response();
    }
    let rid = h.get("x-request-id").and_then(|v| v.to_str().ok()).unwrap_or_default().to_owned();
    let Ok(v) = serde_json::from_slice::<Value>(&body) else {
        return (StatusCode::BAD_REQUEST, "body is not JSON").into_response();
    };
    let resp = {
        let mut st = s.lock();
        let fail = st.done_failures > 0;
        if fail {
            st.done_failures -= 1;
        }
        let Some(j) = st.jobs.get_mut(&job) else { return StatusCode::NOT_FOUND.into_response() };
        j.done_posts += 1;
        j.x_request_ids.push(rid);
        if fail {
            StatusCode::INTERNAL_SERVER_ERROR.into_response()
        } else if v.get("status").and_then(Value::as_str) == Some("IN_PROGRESS") {
            j.progress.push(v.get("output").cloned().unwrap_or(Value::Null));
            StatusCode::OK.into_response()
        } else {
            j.final_is_stream = Some(q.get("isStream").map(String::as_str) == Some("true"));
            j.output = v.get("output").cloned();
            j.error = v.get("error").and_then(Value::as_str).map(str::to_owned);
            if j.status != SimStatus::Cancelled {
                j.status = if j.error.is_some() { SimStatus::Failed } else { SimStatus::Completed };
            }
            StatusCode::OK.into_response()
        }
    };
    s.notify.notify_waiters();
    resp
}

async fn stream(State(s): State<Sim>, Path((_ep, _worker, job)): Path<(String, String, String)>, h: HeaderMap, body: Bytes) -> Response {
    if !s.authorized(&h) {
        return StatusCode::UNAUTHORIZED.into_response();
    }
    let Ok(v) = serde_json::from_slice::<Value>(&body) else {
        return (StatusCode::BAD_REQUEST, "body is not JSON").into_response();
    };
    {
        let mut st = s.lock();
        let Some(j) = st.jobs.get_mut(&job) else { return StatusCode::NOT_FOUND.into_response() };
        j.stream.push(v.get("output").cloned().unwrap_or(Value::Null));
    }
    s.notify.notify_waiters();
    StatusCode::OK.into_response()
}

async fn ping(State(s): State<Sim>, Path((_ep, worker)): Path<(String, String)>, Query(q): Q, h: HeaderMap) -> Response {
    if !s.authorized(&h) {
        return StatusCode::UNAUTHORIZED.into_response();
    }
    let job_ids = q.get("job_id").map(|v| v.split(',').filter(|x| !x.is_empty()).map(str::to_owned).collect()).unwrap_or_default();
    s.lock().pings.push(Ping { worker, job_ids, version: q.get("runpod_version").cloned().unwrap_or_default() });
    StatusCode::OK.into_response()
}

async fn run(State(s): State<Sim>, Json(body): Json<Value>) -> Response {
    let Some(input) = body.get("input").cloned() else {
        return (StatusCode::BAD_REQUEST, Json(json!({"error": "input is required"}))).into_response();
    };
    let id = s.submit(input);
    Json(json!({"id": id, "status": "IN_QUEUE"})).into_response()
}

fn client_view(j: &SimJob) -> Value {
    let mut v = json!({"id": j.id, "status": j.status, "workerId": j.worker});
    if let Some(o) = j.output.clone().or_else(|| j.progress.last().cloned()) {
        v["output"] = o;
    }
    if let Some(e) = &j.error {
        v["error"] = json!(e);
    }
    v
}

async fn status(State(s): State<Sim>, Path((_ep, id)): Path<(String, String)>) -> Response {
    match s.job(&id) {
        Some(j) => Json(client_view(&j)).into_response(),
        None => StatusCode::NOT_FOUND.into_response(),
    }
}

async fn client_stream(State(s): State<Sim>, Path((_ep, id)): Path<(String, String)>) -> Response {
    match s.job(&id) {
        Some(j) => Json(json!({
            "status": j.status,
            "stream": j.stream.iter().map(|c| json!({"output": c})).collect::<Vec<_>>(),
        }))
        .into_response(),
        None => StatusCode::NOT_FOUND.into_response(),
    }
}

async fn client_cancel(State(s): State<Sim>, Path((_ep, id)): Path<(String, String)>) -> Response {
    s.cancel(&id);
    match s.job(&id) {
        Some(j) => Json(json!({"id": id, "status": j.status})).into_response(),
        None => StatusCode::NOT_FOUND.into_response(),
    }
}

async fn health(State(s): State<Sim>) -> Response {
    let st = s.lock();
    let count = |x: SimStatus| st.jobs.values().filter(|j| j.status == x).count();
    Json(json!({
        "jobs": {"inQueue": count(SimStatus::InQueue), "inProgress": count(SimStatus::InProgress),
                 "completed": count(SimStatus::Completed), "failed": count(SimStatus::Failed)},
        "workers": {"running": st.pings.iter().map(|p| p.worker.clone()).collect::<std::collections::BTreeSet<_>>().len()},
    }))
    .into_response()
}
