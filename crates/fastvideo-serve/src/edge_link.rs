//! A worker's socket to its pool's Durable Object (`[dispatch] do_url`,
//! docs/serve/gateway-cloudflare.md §3.2): the push path, next to the
//! gateway's `POST /fv/v1/internal/jobs`, which keeps working.
//!
//! - Dials `{do_url}/pools/{pool}/connect` (WebSocket, `x-fv-internal-token`,
//!   `x-fv-worker-id`), reconnects with backoff (0.25 s doubling to 2 s) and
//!   announces itself with a hello: caps (the `/fv/v1/internal/status` body),
//!   version, capacity, draining, and every job it took from the dispatcher
//!   that it has not reported done (reconcile after a Worker redeploy).
//! - A pushed job goes through the same take as the gateway path
//!   ([`crate::worker::take_envelope`]: inputs, engine), but with the
//!   push's lease: the job is held and started at once and its D1 row is
//!   written behind, fenced by the lease (phase 2). `ack` when taken,
//!   `nack` otherwise (`retry` for busy / draining; 424 for a client URL it
//!   could not fetch). A finished job is reported `done`; a job a newer
//!   lease took is stopped here.
//! - `cancel` and `drain` frames act as the internal routes do; a status
//!   frame every `dispatch.status_s` is the heartbeat.

use std::collections::BTreeMap;
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use fastvideo_dispatch_proto::{self as proto, DoMsg, Held, Hello, WorkerMsg};
use fastvideo_protocol::JobId;
use futures::{SinkExt, StreamExt};
use tokio::sync::mpsc;
use tokio_tungstenite::tungstenite::client::IntoClientRequest;
use tokio_tungstenite::tungstenite::http::HeaderValue;
use tokio_tungstenite::tungstenite::Message;

use crate::gateway::dispatch::Envelope;
use crate::worker::WorkerState;

/// The link's settings.
#[derive(Clone, Debug)]
pub struct LinkCfg {
    pub do_url: String,
    pub pool: String,
    pub token: String,
    pub capacity: u32,
    pub status_every: Duration,
}

/// Jobs taken from the dispatcher (id → (attempt, lease)) not yet reported done.
type Taken = Arc<Mutex<BTreeMap<JobId, (u32, u64)>>>;

/// `http(s)://host/…` → `ws(s)://host/…/pools/{pool}/connect`.
pub fn connect_url(base: &str, pool: &str) -> String {
    let base = base.trim_end_matches('/');
    let ws = if let Some(r) = base.strip_prefix("https://") {
        format!("wss://{r}")
    } else if let Some(r) = base.strip_prefix("http://") {
        format!("ws://{r}")
    } else {
        base.to_owned()
    };
    format!("{ws}{}", proto::connect_path(pool))
}

/// Starts the link (runs until aborted).
pub fn spawn(st: Arc<WorkerState>, cfg: LinkCfg) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        let taken: Taken = Arc::default();
        let (tx, mut rx) = mpsc::unbounded_channel::<WorkerMsg>();
        // A newer lease took one of our jobs (the dispatcher gave up on us
        // meanwhile): stop it here; its writes are already refused.
        if let Some(d1) = st.d1.clone() {
            let mut fenced = d1.subscribe_fenced();
            let st2 = st.clone();
            tokio::spawn(async move {
                while let Some(id) = fenced.recv().await {
                    tracing::warn!(job = %id, "worker: a newer lease holds this job; stopping it here");
                    let _ = st2.gate.engine().cancel(id);
                }
            });
        }
        let mut backoff = Duration::from_millis(250);
        let url = connect_url(&cfg.do_url, &cfg.pool);
        tracing::info!(pool = %cfg.pool, url = %url, capacity = cfg.capacity, "worker: connecting to the pool's dispatcher");
        loop {
            let t0 = Instant::now();
            match session(&st, &cfg, &url, &taken, &tx, &mut rx).await {
                Ok(()) => tracing::info!(pool = %cfg.pool, "worker: dispatcher socket closed; reconnecting"),
                Err(e) => tracing::warn!(pool = %cfg.pool, error = %e, "worker: dispatcher socket failed; reconnecting"),
            }
            if t0.elapsed() > Duration::from_secs(30) {
                backoff = Duration::from_millis(250);
            }
            tokio::time::sleep(backoff).await;
            backoff = (backoff * 2).min(Duration::from_secs(2));
        }
    })
}

fn state_word(j: Option<&fastvideo_protocol::Job>) -> String {
    j.map(|j| j.status().as_str().to_owned()).unwrap_or_else(|| "failed".into())
}

async fn hello(st: &WorkerState, cfg: &LinkCfg, taken: &Taken) -> (Hello, Vec<JobId>) {
    let held: Vec<(JobId, (u32, u64))> = taken.lock().unwrap_or_else(|p| p.into_inner()).iter().map(|(k, v)| (*k, *v)).collect();
    let mut jobs = Vec::new();
    let mut finished = Vec::new();
    for (id, (attempt, lease)) in held {
        let j = st.ctx.jobs().get(id).await;
        let h = Held { job_id: id.to_string(), attempt, lease, state: state_word(j.as_ref()) };
        if h.finished() {
            finished.push(id);
        }
        jobs.push(h);
    }
    let build = crate::build_info::BuildInfo::current().json();
    let sha = build.get("git_sha").and_then(|v| v.as_str()).unwrap_or("unknown").chars().take(7).collect();
    let h = Hello {
        worker_id: st.worker_id.clone(),
        pool: cfg.pool.clone(),
        proto: proto::PROTO_VERSION,
        version: env!("CARGO_PKG_VERSION").to_owned(),
        sha,
        capacity: cfg.capacity.max(1),
        draining: st.draining(),
        models: Vec::new(),
        caps: crate::worker::status_json(st),
        jobs,
    };
    (h, finished)
}

async fn session(
    st: &Arc<WorkerState>,
    cfg: &LinkCfg,
    url: &str,
    taken: &Taken,
    tx: &mpsc::UnboundedSender<WorkerMsg>,
    rx: &mut mpsc::UnboundedReceiver<WorkerMsg>,
) -> Result<(), String> {
    let mut req = url.into_client_request().map_err(|e| format!("dispatcher URL: {e}"))?;
    let h = req.headers_mut();
    h.insert(proto::TOKEN_HEADER, HeaderValue::from_str(&cfg.token).map_err(|_| "internal token is not a valid header value".to_owned())?);
    h.insert(proto::WORKER_HEADER, HeaderValue::from_str(&st.worker_id).map_err(|_| "worker id is not a valid header value".to_owned())?);
    let t0 = Instant::now();
    let (ws, _) = tokio::time::timeout(Duration::from_secs(15), tokio_tungstenite::connect_async(req))
        .await
        .map_err(|_| "connect timed out".to_owned())?
        .map_err(|e| format!("connect: {e}"))?;
    let (mut sink, mut stream) = ws.split();
    let (h, finished) = hello(st, cfg, taken).await;
    let n = h.jobs.len();
    let text = serde_json::to_string(&WorkerMsg::Hello(h)).map_err(|e| e.to_string())?;
    sink.send(Message::text(text)).await.map_err(|e| format!("hello: {e}"))?;
    {
        let mut g = taken.lock().unwrap_or_else(|p| p.into_inner());
        for id in finished {
            g.remove(&id);
        }
    }
    tracing::info!(pool = %cfg.pool, held = n, connect_ms = t0.elapsed().as_millis() as u64, "worker: connected to the pool's dispatcher");
    let mut status = tokio::time::interval(cfg.status_every.max(Duration::from_secs(1)));
    status.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    status.tick().await;
    loop {
        tokio::select! {
            frame = stream.next() => {
                let Some(frame) = frame else { return Ok(()) };
                match frame.map_err(|e| format!("read: {e}"))? {
                    Message::Text(t) => {
                        if t.as_str() == proto::PONG {
                            continue;
                        }
                        match serde_json::from_str::<DoMsg>(t.as_str()) {
                            Ok(m) => on_frame(st, taken, tx, m),
                            Err(e) => tracing::warn!(error = %e, "worker: unreadable dispatcher frame"),
                        }
                    }
                    Message::Close(_) => return Ok(()),
                    _ => {}
                }
            }
            Some(m) = rx.recv() => {
                let done = match &m {
                    WorkerMsg::Done { job_id, .. } => job_id.parse::<JobId>().ok(),
                    _ => None,
                };
                let text = serde_json::to_string(&m).map_err(|e| e.to_string())?;
                if let Err(e) = sink.send(Message::text(text)).await {
                    // Kept for the next hello (the job is still in `taken`).
                    return Err(format!("send: {e}"));
                }
                if let Some(id) = done {
                    taken.lock().unwrap_or_else(|p| p.into_inner()).remove(&id);
                }
            }
            _ = status.tick() => {
                let s = st.gate.engine().stats();
                let m = WorkerMsg::Status { running: u32::try_from(s.running).unwrap_or(u32::MAX), draining: st.draining(), capacity: cfg.capacity.max(1) };
                let text = serde_json::to_string(&m).map_err(|e| e.to_string())?;
                sink.send(Message::text(text)).await.map_err(|e| format!("status: {e}"))?;
            }
        }
    }
}

fn on_frame(st: &Arc<WorkerState>, taken: &Taken, tx: &mpsc::UnboundedSender<WorkerMsg>, m: DoMsg) {
    match m {
        DoMsg::Job { job_id, attempt, envelope, lease, takeover } => {
            let st = st.clone();
            let taken = taken.clone();
            let tx = tx.clone();
            tokio::spawn(async move { take(st, taken, tx, Push { job_id, attempt, lease, takeover }, envelope).await });
        }
        DoMsg::Cancel { job_id } => cancel(st, job_id),
        DoMsg::Welcome { cancel: drop, .. } => {
            for j in drop {
                tracing::warn!(job = %j, "worker: the dispatcher gave this job to another worker; cancelling the copy here");
                cancel(st, j);
            }
        }
        DoMsg::Drain { on } => {
            st.set_drained(on);
        }
    }
}

fn cancel(st: &Arc<WorkerState>, job_id: String) {
    let Ok(id) = job_id.parse::<JobId>() else { return };
    let st = st.clone();
    tokio::spawn(async move {
        if let Err(e) = fastvideo_serve_kit::events::cancel_job(&st.ctx, id).await {
            tracing::debug!(job = %id, error = %e.message, "worker: dispatcher cancel");
        }
    });
}

/// A push's identity.
struct Push {
    job_id: String,
    attempt: u32,
    lease: u64,
    takeover: bool,
}

async fn take(st: Arc<WorkerState>, taken: Taken, tx: mpsc::UnboundedSender<WorkerMsg>, push: Push, envelope: serde_json::Value) {
    let Push { job_id, attempt, lease, takeover } = push;
    let t0 = Instant::now();
    let nack = |retry: bool, code: u16, message: String| WorkerMsg::Nack { job_id: job_id.clone(), attempt, retry, code, message };
    let env: Envelope = match serde_json::from_value(envelope) {
        Ok(e) => e,
        Err(e) => {
            let _ = tx.send(nack(false, 400, format!("unreadable envelope: {e}")));
            return;
        }
    };
    let id = env.job.id;
    if id.to_string() != job_id {
        let _ = tx.send(nack(false, 400, "the envelope is for another job".into()));
        return;
    }
    let resp = crate::worker::take_envelope(&st, env, takeover, Some(lease)).await;
    let code = resp.status().as_u16();
    if resp.status().is_success() {
        let worker_ms = t0.elapsed().as_millis() as u64;
        taken.lock().unwrap_or_else(|p| p.into_inner()).insert(id, (attempt, lease));
        let _ = tx.send(WorkerMsg::Ack { job_id: job_id.clone(), attempt, lease, worker_ms });
        metrics::histogram!("fv_worker_edge_take_seconds").record(worker_ms as f64 / 1e3);
        tracing::info!(job = %id, attempt, lease, takeover, worker_ms, "worker: took a pushed job");
        // Report the end (frees the slot on the dispatcher).
        if let Some(mut w) = st.ctx.jobs().watch(id) {
            while !w.borrow_and_update().state.is_terminal() {
                if w.changed().await.is_err() {
                    break;
                }
            }
        }
        let state = state_word(st.ctx.jobs().get(id).await.as_ref());
        let _ = tx.send(WorkerMsg::Done { job_id, attempt, state });
        return;
    }
    let body = axum::body::to_bytes(resp.into_body(), 64 * 1024).await.unwrap_or_default();
    let message = serde_json::from_slice::<serde_json::Value>(&body)
        .ok()
        .and_then(|v| v.pointer("/error/message").and_then(|m| m.as_str()).map(str::to_owned))
        .unwrap_or_default();
    // Busy, draining, loading, held elsewhere, or a server-side failure:
    // another worker (or this one later) may take it. A bad job is final.
    let retry = matches!(code, 409 | 429 | 503) || code >= 500;
    tracing::info!(job = %id, attempt, code, retry, %message, "worker: refused a pushed job");
    let _ = tx.send(nack(retry, code, message));
}

impl WorkerState {
    /// Drain on / off (the dispatcher's `drain` frame, as the internal
    /// `drain` / `undrain` routes).
    pub fn set_drained(&self, on: bool) {
        self.drained_flag().store(on, Ordering::SeqCst);
    }
}
