//! A worker's sockets to Durable Object dispatchers: its pool's object
//! (`[dispatch] do_url` + `[gateway] pool`, docs/serve/gateway-cloudflare.md
//! §3.2) or one object per model family it serves (`[dispatch] families`,
//! docs/serve/dispatch-do-family.md). The gateway's `POST
//! /fv/v1/internal/jobs` keeps working next to them.
//!
//! - Dials `{do_url}{scope}/connect` (WebSocket, `x-fv-internal-token`,
//!   `x-fv-worker-id`), reconnects with backoff (0.25 s doubling to 2 s) and
//!   announces itself with a hello: caps (the `/fv/v1/internal/status` body),
//!   version, capacity, draining, and every job it took from that object
//!   that it has not reported done (reconcile after a Worker redeploy).
//! - A pushed job goes through the same take as the gateway path
//!   ([`crate::worker::take_envelope`]: inputs, engine), but with the
//!   push's lease: the job is held and started at once and its D1 row is
//!   written behind, fenced by the lease. `ack` when taken, `nack`
//!   otherwise (`retry` for busy / draining; 424 for a client URL it could
//!   not fetch). A finished job is reported `done`; a job a newer lease took
//!   is stopped here.
//! - `cancel` and `drain` frames act as the internal routes do; a status
//!   frame every `dispatch.status_s` is the heartbeat.
//!
//! Family sockets (protocol 2) add:
//!
//! - **the arbiter** ([`crate::arbiter`]): every offer is taken or refused
//!   against the one slot budget of this GPU, shared by all its family
//!   sockets; a refusal is a 429 nack (the object offers the job elsewhere at
//!   once); every change of the budget is reported as `slots` (credits) on
//!   every family socket;
//! - **sessions**: `session_offer` reserves the GPU through the arbiter and
//!   answers `session_ack` with this worker's public endpoint, or
//!   `session_nack`; `session_revoke` frees it; held sessions are
//!   re-announced in the hello;
//! - **direct uploads** (`dispatch.direct_upload`, [`crate::upload`]): after
//!   the ack the job's output is uploaded through part URLs the object mints.

use std::collections::{BTreeMap, HashMap};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use fastvideo_dispatch_proto::{self as proto, DoMsg, Held, HeldSession, Hello, Scope, Slots, WorkerMsg};
use fastvideo_protocol::JobId;
use futures::{SinkExt, StreamExt};
use tokio::sync::{mpsc, oneshot};
use tokio_tungstenite::tungstenite::client::IntoClientRequest;
use tokio_tungstenite::tungstenite::http::HeaderValue;
use tokio_tungstenite::tungstenite::Message;

use crate::arbiter::Arbiter;
use crate::gateway::dispatch::Envelope;
use crate::upload::{UploadSpec, Uploads};
use crate::worker::WorkerState;
use fastvideo_serve_kit::EngineGate;

/// One link's settings.
#[derive(Clone, Debug)]
pub struct LinkCfg {
    pub do_url: String,
    /// The pool (protocol 1) or the family (protocol 2) this socket serves.
    pub scope: Scope,
    pub token: String,
    /// Jobs taken at once (pool sockets; family sockets use the arbiter's).
    pub capacity: u32,
    pub status_every: Duration,
    /// This worker's public base URL (family sockets: sessions).
    pub endpoint: String,
    /// The engine's output directory (`<dir>/<job>/output.mp4`), followed by
    /// direct uploads.
    pub engine_out: PathBuf,
}

/// What the family links of one process share.
#[derive(Clone, Debug)]
pub struct Family {
    pub arbiter: Arc<Arbiter>,
    /// Direct uploads (`None`: outputs go through the artifact store).
    pub uploads: Option<Arc<Uploads>>,
}

/// Jobs taken from the dispatcher (id → (attempt, lease)) not yet reported done.
type Taken = Arc<Mutex<BTreeMap<JobId, (u32, u64)>>>;

/// The sending half of a link, with request/response matching for the
/// frames that carry a `req` (upload grants and commits).
pub struct LinkHandle {
    pub scope: Scope,
    tx: mpsc::UnboundedSender<WorkerMsg>,
    pending: Mutex<HashMap<u64, oneshot::Sender<DoMsg>>>,
    next: AtomicU64,
}

impl std::fmt::Debug for LinkHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LinkHandle").field("scope", &self.scope).finish_non_exhaustive()
    }
}

impl LinkHandle {
    /// A handle over `tx` (the link task reads the other end).
    pub fn new(scope: Scope, tx: mpsc::UnboundedSender<WorkerMsg>) -> Arc<Self> {
        Arc::new(Self { scope, tx, pending: Mutex::default(), next: AtomicU64::new(0) })
    }

    /// Queues a frame (sent once the socket is up).
    pub fn send(&self, m: WorkerMsg) {
        let _ = self.tx.send(m);
    }

    /// Sends the frame `build(req)` and waits for the answer with that `req`.
    pub async fn request(&self, build: impl FnOnce(u64) -> WorkerMsg, timeout: Duration) -> Result<DoMsg, String> {
        let req = self.next.fetch_add(1, Ordering::Relaxed) + 1;
        let (tx, rx) = oneshot::channel();
        self.pending.lock().unwrap_or_else(std::sync::PoisonError::into_inner).insert(req, tx);
        self.send(build(req));
        match tokio::time::timeout(timeout, rx).await {
            Ok(Ok(m)) => Ok(m),
            Ok(Err(_)) => Err("the link was dropped".into()),
            Err(_) => {
                self.pending.lock().unwrap_or_else(std::sync::PoisonError::into_inner).remove(&req);
                Err(format!("no answer from the dispatcher within {} s", timeout.as_secs()))
            }
        }
    }

    /// Hands an answer frame to its waiting request; other frames come back.
    pub fn resolve(&self, m: DoMsg) -> Option<DoMsg> {
        let req = match &m {
            DoMsg::UploadGrant { req, .. } | DoMsg::UploadCommitted { req, .. } => *req,
            _ => return Some(m),
        };
        let w = self.pending.lock().unwrap_or_else(std::sync::PoisonError::into_inner).remove(&req);
        match w {
            Some(w) => {
                let _ = w.send(m);
                None
            }
            None => None,
        }
    }
}

/// `http(s)://host/…` → `ws(s)://host/…/pools/{pool}/connect`.
pub fn connect_url(base: &str, pool: &str) -> String {
    scope_url(base, &Scope::Pool(pool.to_owned()))
}

/// `http(s)://host/…` → `ws(s)://host/…{scope}/connect`.
pub fn scope_url(base: &str, scope: &Scope) -> String {
    let base = base.trim_end_matches('/');
    let ws = if let Some(r) = base.strip_prefix("https://") {
        format!("wss://{r}")
    } else if let Some(r) = base.strip_prefix("http://") {
        format!("ws://{r}")
    } else {
        base.to_owned()
    };
    format!("{ws}{}", scope.connect_path())
}

/// Per-link state.
struct Link {
    st: Arc<WorkerState>,
    cfg: LinkCfg,
    fam: Option<Family>,
    handle: Arc<LinkHandle>,
    taken: Taken,
    /// Sessions held from this object (id → lease).
    sessions: Mutex<BTreeMap<String, u64>>,
    /// Offers received on the current socket.
    offers_seen: AtomicU64,
}

impl Link {
    fn family(&self) -> &str {
        self.cfg.scope.id()
    }
    fn slots(&self) -> Slots {
        let (free, session_free) = match &self.fam {
            Some(f) if !self.st.draining() => {
                let x = f.arbiter.free();
                (x.jobs, x.sessions)
            }
            _ => (0, 0),
        };
        Slots { free, session_free, offers_seen: self.offers_seen.load(Ordering::SeqCst) }
    }
}

/// Starts the link (runs until aborted). `fam`: the family links' shared
/// arbiter and uploads (required for a family scope).
pub fn spawn(st: Arc<WorkerState>, cfg: LinkCfg, fam: Option<Family>) -> tokio::task::JoinHandle<()> {
    let (tx, rx) = mpsc::unbounded_channel::<WorkerMsg>();
    let handle = LinkHandle::new(cfg.scope.clone(), tx);
    let link = Arc::new(Link {
        st,
        cfg,
        fam,
        handle,
        taken: Arc::default(),
        sessions: Mutex::default(),
        offers_seen: AtomicU64::new(0),
    });
    tokio::spawn(run(link, rx))
}

async fn run(link: Arc<Link>, mut rx: mpsc::UnboundedReceiver<WorkerMsg>) {
    let st = &link.st;
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
    let url = scope_url(&link.cfg.do_url, &link.cfg.scope);
    let scope = link.cfg.scope.object_name();
    tracing::info!(scope = %scope, url = %url, capacity = link.cfg.capacity, family = link.fam.is_some(), "worker: connecting to the dispatcher");
    loop {
        let t0 = Instant::now();
        match session(&link, &url, &mut rx).await {
            Ok(()) => tracing::info!(scope = %scope, "worker: dispatcher socket closed; reconnecting"),
            Err(e) => tracing::warn!(scope = %scope, error = %e, "worker: dispatcher socket failed; reconnecting"),
        }
        if t0.elapsed() > Duration::from_secs(30) {
            backoff = Duration::from_millis(250);
        }
        tokio::time::sleep(backoff).await;
        backoff = (backoff * 2).min(Duration::from_secs(2));
    }
}

fn state_word(j: Option<&fastvideo_protocol::Job>) -> String {
    j.map(|j| j.status().as_str().to_owned()).unwrap_or_else(|| "failed".into())
}

async fn hello(link: &Link) -> (Hello, Vec<JobId>) {
    let st = &link.st;
    let held: Vec<(JobId, (u32, u64))> = link.taken.lock().unwrap_or_else(|p| p.into_inner()).iter().map(|(k, v)| (*k, *v)).collect();
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
    let family = link.fam.is_some();
    let sessions = if family {
        link.sessions.lock().unwrap_or_else(|p| p.into_inner()).iter().map(|(k, v)| HeldSession { session_id: k.clone(), lease: *v }).collect()
    } else {
        Vec::new()
    };
    let capacity = match &link.fam {
        Some(f) => f.arbiter.capacity(),
        None => link.cfg.capacity.max(1),
    };
    let h = Hello {
        worker_id: st.worker_id.clone(),
        pool: link.cfg.scope.object_name(),
        proto: proto::PROTO_VERSION,
        version: env!("CARGO_PKG_VERSION").to_owned(),
        sha,
        capacity,
        draining: st.draining(),
        // Family objects place by model; pool objects take any of the pool's.
        models: if family { st.gate.models().iter().map(|m| m.id.to_string()).collect() } else { Vec::new() },
        caps: crate::worker::status_json(st),
        jobs,
        sessions,
        endpoint: if family { link.cfg.endpoint.clone() } else { String::new() },
        slots: family.then(|| link.slots()),
    };
    (h, finished)
}

async fn session(link: &Arc<Link>, url: &str, rx: &mut mpsc::UnboundedReceiver<WorkerMsg>) -> Result<(), String> {
    let st = &link.st;
    let mut req = url.into_client_request().map_err(|e| format!("dispatcher URL: {e}"))?;
    let h = req.headers_mut();
    h.insert(proto::TOKEN_HEADER, HeaderValue::from_str(&link.cfg.token).map_err(|_| "internal token is not a valid header value".to_owned())?);
    h.insert(proto::WORKER_HEADER, HeaderValue::from_str(&st.worker_id).map_err(|_| "worker id is not a valid header value".to_owned())?);
    let t0 = Instant::now();
    let (ws, _) = tokio::time::timeout(Duration::from_secs(15), tokio_tungstenite::connect_async(req))
        .await
        .map_err(|_| "connect timed out".to_owned())?
        .map_err(|e| format!("connect: {e}"))?;
    let (mut sink, mut stream) = ws.split();
    // A new socket: the offer count starts again (the hello resets it there too).
    link.offers_seen.store(0, Ordering::SeqCst);
    let (h, finished) = hello(link).await;
    let n = h.jobs.len();
    let text = serde_json::to_string(&WorkerMsg::Hello(h)).map_err(|e| e.to_string())?;
    sink.send(Message::text(text)).await.map_err(|e| format!("hello: {e}"))?;
    {
        let mut g = link.taken.lock().unwrap_or_else(|p| p.into_inner());
        for id in finished {
            g.remove(&id);
        }
    }
    tracing::info!(scope = %link.cfg.scope.object_name(), held = n, connect_ms = t0.elapsed().as_millis() as u64, "worker: connected to the dispatcher");
    let mut status = tokio::time::interval(link.cfg.status_every.max(Duration::from_secs(1)));
    status.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    status.tick().await;
    let mut budget = link.fam.as_ref().map(|f| f.arbiter.subscribe());
    let mut last_slots = link.fam.as_ref().map(|_| link.slots());
    loop {
        let changed = async {
            match budget.as_mut() {
                Some(b) => b.changed().await.is_ok(),
                None => std::future::pending().await,
            }
        };
        tokio::select! {
            frame = stream.next() => {
                let Some(frame) = frame else { return Ok(()) };
                match frame.map_err(|e| format!("read: {e}"))? {
                    Message::Text(t) => {
                        if t.as_str() == proto::PONG {
                            continue;
                        }
                        match serde_json::from_str::<DoMsg>(t.as_str()) {
                            Ok(m) => on_frame(link, m),
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
                    link.taken.lock().unwrap_or_else(|p| p.into_inner()).remove(&id);
                }
            }
            true = changed => {
                let s = link.slots();
                if last_slots != Some(s) {
                    last_slots = Some(s);
                    let text = serde_json::to_string(&WorkerMsg::Slots(s)).map_err(|e| e.to_string())?;
                    sink.send(Message::text(text)).await.map_err(|e| format!("slots: {e}"))?;
                }
            }
            _ = status.tick() => {
                let s = st.gate.engine().stats();
                let capacity = link.fam.as_ref().map_or(link.cfg.capacity, |f| f.arbiter.capacity()).max(1);
                let m = WorkerMsg::Status { running: u32::try_from(s.running).unwrap_or(u32::MAX), draining: st.draining(), capacity };
                let text = serde_json::to_string(&m).map_err(|e| e.to_string())?;
                sink.send(Message::text(text)).await.map_err(|e| format!("status: {e}"))?;
                if link.fam.is_some() {
                    // Credits again with the heartbeat (draining changes them).
                    let s = link.slots();
                    last_slots = Some(s);
                    let text = serde_json::to_string(&WorkerMsg::Slots(s)).map_err(|e| e.to_string())?;
                    sink.send(Message::text(text)).await.map_err(|e| format!("slots: {e}"))?;
                }
            }
        }
    }
}

fn on_frame(link: &Arc<Link>, m: DoMsg) {
    let Some(m) = link.handle.resolve(m) else { return };
    let st = &link.st;
    match m {
        DoMsg::Job { job_id, attempt, envelope, lease, takeover } => {
            link.offers_seen.fetch_add(1, Ordering::SeqCst);
            if let Some(f) = &link.fam {
                // The arbiter decides now, before anything else can.
                if let Err(r) = f.arbiter.try_take_job(link.family(), &job_id) {
                    tracing::info!(job = %job_id, family = link.family(), reason = %r, "worker: offer refused by the arbiter");
                    metrics::counter!("fv_worker_arbiter_refused_total", "family" => link.family().to_owned()).increment(1);
                    link.handle.send(WorkerMsg::Nack { job_id, attempt, retry: true, code: 429, message: r.to_string() });
                    return;
                }
            }
            let link = link.clone();
            tokio::spawn(async move { take(link, Push { job_id, attempt, lease, takeover }, envelope).await });
        }
        DoMsg::Cancel { job_id } => cancel(st, job_id),
        DoMsg::Welcome { cancel: drop, end_sessions, .. } => {
            for j in drop {
                tracing::warn!(job = %j, "worker: the dispatcher gave this job to another worker; cancelling the copy here");
                cancel(st, j);
            }
            for s in end_sessions {
                end_session(link, &s, "ended by the dispatcher while disconnected");
            }
        }
        DoMsg::Drain { on } => {
            st.set_drained(on);
        }
        DoMsg::SessionOffer { session_id, lease, kind, .. } => {
            let Some(f) = &link.fam else {
                link.handle.send(WorkerMsg::SessionNack { session_id, lease, code: 501, message: "no sessions on a pool socket".into() });
                return;
            };
            let r = if st.draining() { Err("draining".to_owned()) } else { f.arbiter.try_take_session(link.family(), &session_id).map_err(|e| e.to_string()) };
            match r {
                Ok(()) => {
                    link.sessions.lock().unwrap_or_else(|p| p.into_inner()).insert(session_id.clone(), lease);
                    tracing::info!(session = %session_id, lease, kind = %kind, family = link.family(), "worker: session admitted");
                    link.handle.send(WorkerMsg::SessionAck { session_id, lease, endpoint: link.cfg.endpoint.clone() });
                }
                Err(e) => {
                    tracing::info!(session = %session_id, reason = %e, "worker: session refused");
                    link.handle.send(WorkerMsg::SessionNack { session_id, lease, code: 429, message: e });
                }
            }
        }
        DoMsg::SessionRevoke { session_id, reason } => end_session(link, &session_id, &reason),
        // Answers without a waiting request (it timed out): nothing to do.
        DoMsg::UploadGrant { .. } | DoMsg::UploadCommitted { .. } => {}
    }
}

fn end_session(link: &Link, id: &str, why: &str) {
    let had = link.sessions.lock().unwrap_or_else(|p| p.into_inner()).remove(id).is_some();
    if let Some(f) = &link.fam {
        f.arbiter.release_session(id);
    }
    if had {
        tracing::info!(session = %id, reason = %why, "worker: session ended; slot free");
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

async fn take(link: Arc<Link>, push: Push, envelope: serde_json::Value) {
    let Push { job_id, attempt, lease, takeover } = push;
    let st = link.st.clone();
    let t0 = Instant::now();
    let release = |link: &Link| {
        if let Some(f) = &link.fam {
            f.arbiter.release_job(&job_id);
        }
    };
    let nack = |retry: bool, code: u16, message: String| WorkerMsg::Nack { job_id: job_id.clone(), attempt, retry, code, message };
    let env: Envelope = match serde_json::from_value(envelope) {
        Ok(e) => e,
        Err(e) => {
            release(&link);
            link.handle.send(nack(false, 400, format!("unreadable envelope: {e}")));
            return;
        }
    };
    let id = env.job.id;
    if id.to_string() != job_id {
        release(&link);
        link.handle.send(nack(false, 400, "the envelope is for another job".into()));
        return;
    }
    let file_name = crate::adapters::artifact_file_name(&env.job);
    let resp = crate::worker::take_envelope(&st, env, takeover, Some(lease)).await;
    let code = resp.status().as_u16();
    if resp.status().is_success() {
        let worker_ms = t0.elapsed().as_millis() as u64;
        link.taken.lock().unwrap_or_else(|p| p.into_inner()).insert(id, (attempt, lease));
        link.handle.send(WorkerMsg::Ack { job_id: job_id.clone(), attempt, lease, worker_ms });
        metrics::histogram!("fv_worker_edge_take_seconds").record(worker_ms as f64 / 1e3);
        tracing::info!(job = %id, attempt, lease, takeover, worker_ms, scope = %link.cfg.scope.object_name(), "worker: took a pushed job");
        let uploads = link.fam.as_ref().and_then(|f| f.uploads.clone());
        if let Some(u) = &uploads {
            let name = if proto::valid_id(&file_name) { file_name } else { "output.mp4".to_owned() };
            u.start(UploadSpec {
                job: id,
                attempt,
                lease,
                tail: link.cfg.engine_out.join(id.to_string()).join("output.mp4"),
                name,
                content_type: "video/mp4".into(),
                link: link.handle.clone(),
            });
        }
        // Report the end (frees the slot on the dispatcher).
        if let Some(mut w) = st.ctx.jobs().watch(id) {
            while !w.borrow_and_update().state.is_terminal() {
                if w.changed().await.is_err() {
                    break;
                }
            }
        }
        if let Some(u) = &uploads {
            // No output was stored (failed, cancelled): abort the upload.
            u.end(id);
        }
        let state = state_word(st.ctx.jobs().get(id).await.as_ref());
        release(&link);
        link.handle.send(WorkerMsg::Done { job_id, attempt, state });
        return;
    }
    release(&link);
    let body = axum::body::to_bytes(resp.into_body(), 64 * 1024).await.unwrap_or_default();
    let message = serde_json::from_slice::<serde_json::Value>(&body)
        .ok()
        .and_then(|v| v.pointer("/error/message").and_then(|m| m.as_str()).map(str::to_owned))
        .unwrap_or_default();
    // Busy, draining, loading, held elsewhere, or a server-side failure:
    // another worker (or this one later) may take it. A bad job is final.
    let retry = matches!(code, 409 | 429 | 503) || code >= 500;
    tracing::info!(job = %id, attempt, code, retry, %message, "worker: refused a pushed job");
    link.handle.send(nack(retry, code, message));
}

impl WorkerState {
    /// Drain on / off (the dispatcher's `drain` frame, as the internal
    /// `drain` / `undrain` routes).
    pub fn set_drained(&self, on: bool) {
        self.drained_flag().store(on, Ordering::SeqCst);
    }
}
