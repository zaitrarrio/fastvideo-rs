//! The pool scheduler: a pure, deterministic state machine (time comes in
//! as `now` in Unix milliseconds; effects go out as [`Out`]). The Durable
//! Object in `crates/fastvideo-edge` hosts it over its SQLite storage and
//! WebSockets; a native host (tests, or fv-serve later) can do the same.
//!
//! - **Queue**: FIFO by enqueue order. A job is pushed to the connected,
//!   non-draining worker with the fewest held jobs below its capacity.
//! - **Ack**: the worker's ack is the adopt; no ack within
//!   [`Cfg::ack_timeout_ms`] puts the job back in the queue (same attempt).
//!   A retryable nack (busy, draining, held elsewhere) does too, after a
//!   short backoff; a final nack fails the job.
//! - **Worker loss**: a socket that closes keeps its jobs for
//!   [`Cfg::reconnect_grace_ms`] (a Worker deploy drops every socket and the
//!   workers come back within seconds). After that, or when a connected
//!   worker is silent for [`Cfg::stale_after_ms`], its pushed jobs go back to
//!   the queue and its running jobs are **re-dispatched once** (attempt + 1,
//!   `takeover`), then failed ([`EnqueueReq::retries`]). A re-dispatch no
//!   worker takes within [`Cfg::redispatch_wait_ms`] fails too.
//! - **Leases** (fencing tokens): every push carries the job's next lease.
//!   The worker writes the job's D1 row only while no newer lease holds it,
//!   and an ack or a reconnect with an older lease gets a cancel: a worker
//!   the dispatcher gave up on cannot run on in parallel with its successor.
//! - **Reconnect** ([`Hello::jobs`]): a job the worker still holds with the
//!   current lease stays (or becomes) running there; a job the dispatcher
//!   pushed to it that it does not hold is queued again; one it ran that it
//!   lost is a loss; a job it holds under an older lease is cancelled there
//!   ([`DoMsg::Welcome`]).
//! - **Restage**: a worker that cannot fetch a client URL nacks 424; the job
//!   waits until the gateway sends it again with the input in the store
//!   ([`EnqueueReq::replace`]).
//! - **Spill**: the host may keep a large envelope outside ([`Sched::enqueue_spilled`],
//!   e.g. in R2); the push then asks the host to load it ([`Out::PushSpilled`]).
//!
//! Protocol 2 (docs/serve/dispatch-do-family.md):
//!
//! - **Credits**: a worker that reports [`Slots`] is offered jobs while
//!   `free − (offers sent − offers seen) > 0`; its own arbiter has the last
//!   word. A **429 nack** (the arbiter is full: another family holds the GPU)
//!   puts the job back at once, with no backoff and no bounce limit, and the
//!   worker gets nothing more until its next `slots` frame.
//! - **Sessions** ([`Sched::admit`]): offered to a connected worker with a
//!   free session slot; its ack makes the lease live (the host answers the
//!   waiting HTTP call on [`Out::SessionReady`]); a nack or an ack timeout
//!   tries the next worker. Live leases expire without [`Sched::renew`];
//!   [`Sched::release`] and the worker's `session_end` end them.
//! - **Uploads**: `upload_init` from the job's current holder (lease-checked)
//!   asks the host to create a multipart upload ([`Out::CreateUpload`]), then
//!   to mint part URLs ([`Out::Grant`]); `upload_done` asks it to complete
//!   the upload ([`Out::CompleteUpload`]) and records the job's result. Uploads
//!   of jobs that moved on (lost, fenced, failed, cancelled) or that expired
//!   are aborted ([`Out::AbortUpload`]).

use std::collections::{BTreeMap, BTreeSet, VecDeque};

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::{
    DispatchTimings, DoMsg, EnqueueReq, EnqueueResp, FailedJob, FamilyMetrics, Hello, Part, PoolStatus, SessionGrant, SessionInfo, SessionReq, Slots, WorkerInfo,
    WorkerMsg,
};

/// Timeouts, milliseconds.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Cfg {
    /// Push → ack before the job goes back to the queue.
    pub ack_timeout_ms: i64,
    /// A closed socket keeps its jobs this long (redeploys, network blips).
    pub reconnect_grace_ms: i64,
    /// A connected worker that sent nothing for this long is lost.
    pub stale_after_ms: i64,
    /// Backoff after a retryable nack.
    pub nack_backoff_ms: i64,
    /// Backoff after a 409 (the D1 row is held by another worker).
    pub conflict_backoff_ms: i64,
    /// A job no worker takes for this long (per attempt) fails.
    pub max_bounce_ms: i64,
    /// Finished jobs are kept this long (status, duplicate enqueues).
    pub keep_finished_ms: i64,
    /// A job re-dispatched after a loss that no worker takes for this long
    /// fails (the gateway path fails it at once when no worker is there).
    pub redispatch_wait_ms: i64,
    /// A session offer without an ack for this long goes to the next worker.
    #[serde(default = "d_session_offer")]
    pub session_offer_timeout_ms: i64,
    /// Default session lease without a renew.
    #[serde(default = "d_session_ttl")]
    pub session_ttl_ms: i64,
    /// Workers a session admission tries before it answers "no capacity".
    #[serde(default = "d_session_tries")]
    pub session_max_tries: u32,
    /// Part URLs and an open upload live this long.
    #[serde(default = "d_upload_ttl")]
    pub upload_ttl_ms: i64,
    /// Object key prefix of uploads (`{prefix}{scope id}/{job}/{attempt}-{lease}/{name}`).
    #[serde(default = "d_upload_prefix")]
    pub upload_prefix: String,
}

fn d_session_offer() -> i64 {
    5_000
}
fn d_session_ttl() -> i64 {
    60_000
}
fn d_session_tries() -> u32 {
    3
}
fn d_upload_ttl() -> i64 {
    3_600_000
}
fn d_upload_prefix() -> String {
    "outputs/".into()
}

/// Ended sessions are kept this long (status, idempotent admits).
const KEEP_SESSIONS_MS: i64 = 600_000;

impl Default for Cfg {
    fn default() -> Self {
        Self {
            ack_timeout_ms: 10_000,
            reconnect_grace_ms: 20_000,
            stale_after_ms: 60_000,
            nack_backoff_ms: 1_000,
            conflict_backoff_ms: 3_000,
            max_bounce_ms: 180_000,
            keep_finished_ms: 3_600_000,
            redispatch_wait_ms: 120_000,
            session_offer_timeout_ms: d_session_offer(),
            session_ttl_ms: d_session_ttl(),
            session_max_tries: d_session_tries(),
            upload_ttl_ms: d_upload_ttl(),
            upload_prefix: d_upload_prefix(),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Phase {
    Queued,
    Pushed,
    Running,
    Succeeded,
    Failed,
    Cancelled,
}

impl Phase {
    pub fn finished(self) -> bool {
        matches!(self, Phase::Succeeded | Phase::Failed | Phase::Cancelled)
    }
    pub fn as_str(self) -> &'static str {
        match self {
            Phase::Queued => "queued",
            Phase::Pushed => "pushed",
            Phase::Running => "running",
            Phase::Succeeded => "succeeded",
            Phase::Failed => "failed",
            Phase::Cancelled => "cancelled",
        }
    }
    fn from_worker(s: &str) -> Phase {
        match s {
            "succeeded" => Phase::Succeeded,
            "cancelled" => Phase::Cancelled,
            _ => Phase::Failed,
        }
    }
}

/// One job (persisted as JSON; the envelope is stored on its own).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct JobRec {
    pub job_id: String,
    pub attempt: u32,
    pub max_attempts: u32,
    pub phase: Phase,
    pub worker: Option<String>,
    pub model: Option<String>,
    pub seq: u64,
    pub enqueued_at: i64,
    pub pushed_at: Option<i64>,
    pub acked_at: Option<i64>,
    pub finished_at: Option<i64>,
    /// Ack deadline while pushed.
    pub deadline: Option<i64>,
    /// Not pushed before this (backoff).
    pub not_before: i64,
    /// First push of the current attempt (bounce limit).
    pub first_push_at: Option<i64>,
    pub error: Option<String>,
    pub cancel_requested: bool,
    /// Re-dispatch after a loss: the worker may take the row over.
    pub takeover: bool,
    /// Failed here (not by its worker): the gateway fails the D1 row.
    pub failed_here: bool,
    /// Whether the timings of its first dispatch were recorded.
    pub timed: bool,
    /// When a loss put it back in the queue (the re-dispatch wait).
    #[serde(default)]
    pub requeued_at: Option<i64>,
    /// The lease of the last push (0: never pushed).
    #[serde(default)]
    pub lease: u64,
    /// Waiting for the gateway to restage an input (a worker nacked 424).
    #[serde(default)]
    pub restage: bool,
    /// Where the host keeps the envelope when it is not held here.
    #[serde(default)]
    pub spill: Option<String>,
    /// Protocol 2: the committed output upload.
    #[serde(default)]
    pub result: Option<JobResult>,
    /// The API key it runs for (the edge's in-flight quota).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub owner: Option<String>,
    #[serde(skip)]
    pub envelope: Option<Value>,
}

/// A job's committed output object.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct JobResult {
    pub key: String,
    pub bytes: u64,
    pub sha256: String,
}

impl JobRec {
    fn has_envelope(&self) -> bool {
        self.envelope.is_some() || self.spill.is_some()
    }
}

/// One worker (persisted as JSON).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct WorkerRec {
    pub worker_id: String,
    pub connected: bool,
    pub draining: bool,
    pub capacity: u32,
    pub version: String,
    pub sha: String,
    pub models: Vec<String>,
    pub caps: Value,
    pub last_seen: i64,
    pub disconnected_at: Option<i64>,
    /// Skipped until then (it answered busy, or missed an ack).
    pub full_until: i64,
    /// Protocol of its last hello.
    #[serde(default)]
    pub proto: u32,
    /// Protocol 2: free job slots it last reported (`None`: protocol 1,
    /// placed by `capacity`).
    #[serde(default)]
    pub credits: Option<u32>,
    /// Offers sent on its current socket, and how many it had seen when it
    /// reported `credits`.
    #[serde(default)]
    pub offers_sent: u64,
    #[serde(default)]
    pub offers_seen: u64,
    /// Free session slots it last reported.
    #[serde(default)]
    pub session_free: u32,
    /// Its public base URL (sessions).
    #[serde(default)]
    pub endpoint: String,
    /// Its API front (what the edge routes to it).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub front: Option<crate::FrontInfo>,
    /// Its models are loaded (its last status frame; absent: ready).
    #[serde(default = "yes")]
    pub ready: bool,
}

fn yes() -> bool {
    true
}

/// A session lease's state.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SessionState {
    Offered,
    Live,
    Ended,
}

impl SessionState {
    pub fn as_str(self) -> &'static str {
        match self {
            SessionState::Offered => "offered",
            SessionState::Live => "live",
            SessionState::Ended => "ended",
        }
    }
}

/// One streaming session (persisted as JSON).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SessionRec {
    pub session_id: String,
    pub model: Option<String>,
    pub kind: String,
    pub owner: Option<String>,
    pub worker: Option<String>,
    /// Fencing token: +1 per offer.
    pub lease: u64,
    pub state: SessionState,
    pub created_at: i64,
    /// Offer ack deadline.
    pub deadline: Option<i64>,
    pub ttl_ms: i64,
    pub expires_at: Option<i64>,
    /// Workers that refused or timed out.
    pub tried: Vec<String>,
    pub endpoint: String,
    pub end_reason: Option<String>,
    pub ended_at: Option<i64>,
}

impl SessionRec {
    fn grant(&self) -> SessionGrant {
        SessionGrant {
            session_id: self.session_id.clone(),
            lease: self.lease,
            worker_id: self.worker.clone().unwrap_or_default(),
            endpoint: self.endpoint.clone(),
            expires_ms: self.expires_at.unwrap_or(0),
        }
    }
}

/// An output upload's state.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum UploadState {
    /// The host is creating the multipart upload.
    Creating,
    Open,
    /// The host is completing it.
    Completing,
    Done,
}

/// One output upload (persisted as JSON), keyed by its object key.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct UploadRec {
    pub key: String,
    /// Empty while creating.
    pub upload_id: String,
    pub job_id: String,
    pub attempt: u32,
    pub lease: u64,
    pub worker: String,
    pub state: UploadState,
    pub content_type: String,
    pub created_at: i64,
    pub expires_at: i64,
    pub bytes: u64,
    pub sha256: String,
}

/// What [`Sched::admit`] decided.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Admit {
    /// Offered to a worker: wait for [`Out::SessionReady`].
    Pending(String),
    /// Already live (an idempotent admit).
    Granted(SessionGrant),
    /// No worker can take it.
    Refused(String),
}

/// An effect for the host.
#[derive(Clone, Debug, PartialEq)]
pub enum Out {
    /// Send a frame on the worker's socket.
    Send { worker: String, msg: DoMsg },
    /// Close the worker's socket (declared lost while connected).
    Close { worker: String },
    /// Load the envelope kept at `key` into `msg` (a [`DoMsg::Job`] with a
    /// null envelope), then send it to the worker.
    PushSpilled { worker: String, key: String, msg: DoMsg },
    /// Create a multipart upload of `key`, then call
    /// [`Sched::upload_created`] (with `req` and `parts`).
    CreateUpload { worker: String, req: u64, key: String, content_type: String, parts: u16 },
    /// Mint URLs for parts `from .. from + count` of `upload_id`, valid until
    /// `expires_ms`, and send them as a [`DoMsg::UploadGrant`] for `req`.
    Grant { worker: String, req: u64, job_id: String, key: String, upload_id: String, from: u16, count: u16, expires_ms: i64 },
    /// Complete the upload with `parts`, then call [`Sched::upload_completed`]
    /// with the object's size.
    CompleteUpload { worker: String, req: u64, key: String, upload_id: String, parts: Vec<Part>, bytes: u64 },
    /// Abort a multipart upload (best effort).
    AbortUpload { key: String, upload_id: String },
    /// Answer the admission call waiting for `session_id`.
    SessionReady { session_id: String, result: Result<SessionGrant, String> },
}

/// What changed since the last [`Sched::take_dirty`]: rows to write.
#[derive(Debug, Default)]
pub struct Dirty {
    pub jobs: Vec<JobRec>,
    pub workers: Vec<WorkerRec>,
    pub removed_jobs: Vec<String>,
    pub removed_workers: Vec<String>,
    /// Jobs whose envelope must be written (new) ...
    pub new_envelopes: Vec<(String, Value)>,
    /// ... or deleted (finished).
    pub dropped_envelopes: Vec<String>,
    /// Spilled envelopes (host keys) no longer needed.
    pub dropped_spills: Vec<String>,
    pub sessions: Vec<SessionRec>,
    pub removed_sessions: Vec<String>,
    pub uploads: Vec<UploadRec>,
    pub removed_uploads: Vec<String>,
}

impl Dirty {
    pub fn is_empty(&self) -> bool {
        self.jobs.is_empty()
            && self.workers.is_empty()
            && self.removed_jobs.is_empty()
            && self.removed_workers.is_empty()
            && self.new_envelopes.is_empty()
            && self.dropped_envelopes.is_empty()
            && self.dropped_spills.is_empty()
            && self.sessions.is_empty()
            && self.removed_sessions.is_empty()
            && self.uploads.is_empty()
            && self.removed_uploads.is_empty()
    }
}

/// The scheduler of one pool.
#[derive(Debug)]
pub struct Sched {
    pub pool: String,
    pub cfg: Cfg,
    jobs: BTreeMap<String, JobRec>,
    workers: BTreeMap<String, WorkerRec>,
    seq: u64,
    dirty_jobs: BTreeSet<String>,
    dirty_workers: BTreeSet<String>,
    removed_jobs: BTreeSet<String>,
    removed_workers: BTreeSet<String>,
    new_envelopes: BTreeSet<String>,
    dropped_envelopes: BTreeSet<String>,
    dropped_spills: BTreeSet<String>,
    /// (queue_ms, ack_ms, worker_ms) of recent first dispatches.
    timings: VecDeque<(f64, f64, f64)>,
    sessions: BTreeMap<String, SessionRec>,
    uploads: BTreeMap<String, UploadRec>,
    dirty_sessions: BTreeSet<String>,
    removed_sessions: BTreeSet<String>,
    dirty_uploads: BTreeSet<String>,
    removed_uploads: BTreeSet<String>,
}

const TIMINGS_KEEP: usize = 256;
/// Disconnected workers holding nothing are forgotten after a day.
const FORGET_WORKER_MS: i64 = 86_400_000;

impl Sched {
    pub fn new(pool: impl Into<String>, cfg: Cfg) -> Self {
        Self {
            pool: pool.into(),
            cfg,
            jobs: BTreeMap::new(),
            workers: BTreeMap::new(),
            seq: 0,
            dirty_jobs: BTreeSet::new(),
            dirty_workers: BTreeSet::new(),
            removed_jobs: BTreeSet::new(),
            removed_workers: BTreeSet::new(),
            new_envelopes: BTreeSet::new(),
            dropped_envelopes: BTreeSet::new(),
            dropped_spills: BTreeSet::new(),
            timings: VecDeque::new(),
            sessions: BTreeMap::new(),
            uploads: BTreeMap::new(),
            dirty_sessions: BTreeSet::new(),
            removed_sessions: BTreeSet::new(),
            dirty_uploads: BTreeSet::new(),
            removed_uploads: BTreeSet::new(),
        }
    }

    /// Rebuilds from persisted rows (envelopes already set on the jobs).
    pub fn restore(pool: impl Into<String>, cfg: Cfg, jobs: Vec<JobRec>, workers: Vec<WorkerRec>) -> Self {
        Self::restore_all(pool, cfg, jobs, workers, Vec::new(), Vec::new())
    }

    /// [`Sched::restore`] with protocol-2 sessions and uploads.
    pub fn restore_all(pool: impl Into<String>, cfg: Cfg, jobs: Vec<JobRec>, workers: Vec<WorkerRec>, sessions: Vec<SessionRec>, uploads: Vec<UploadRec>) -> Self {
        let mut s = Self::new(pool, cfg);
        for j in jobs {
            s.seq = s.seq.max(j.seq);
            s.jobs.insert(j.job_id.clone(), j);
        }
        for w in workers {
            s.workers.insert(w.worker_id.clone(), w);
        }
        for x in sessions {
            s.sessions.insert(x.session_id.clone(), x);
        }
        for u in uploads {
            s.uploads.insert(u.key.clone(), u);
        }
        s
    }

    pub fn sessions(&self) -> impl Iterator<Item = &SessionRec> {
        self.sessions.values()
    }
    pub fn session(&self, id: &str) -> Option<&SessionRec> {
        self.sessions.get(id)
    }
    pub fn uploads(&self) -> impl Iterator<Item = &UploadRec> {
        self.uploads.values()
    }

    pub fn job(&self, id: &str) -> Option<&JobRec> {
        self.jobs.get(id)
    }
    pub fn worker(&self, id: &str) -> Option<&WorkerRec> {
        self.workers.get(id)
    }
    pub fn workers(&self) -> impl Iterator<Item = &WorkerRec> {
        self.workers.values()
    }
    pub fn jobs(&self) -> impl Iterator<Item = &JobRec> {
        self.jobs.values()
    }

    /// Changed rows since the last call.
    pub fn take_dirty(&mut self) -> Dirty {
        let jobs = std::mem::take(&mut self.dirty_jobs).into_iter().filter_map(|id| self.jobs.get(&id).cloned()).collect();
        let workers = std::mem::take(&mut self.dirty_workers).into_iter().filter_map(|id| self.workers.get(&id).cloned()).collect();
        let new_envelopes = std::mem::take(&mut self.new_envelopes)
            .into_iter()
            .filter_map(|id| self.jobs.get(&id).and_then(|j| j.envelope.clone()).map(|e| (id, e)))
            .collect();
        let sessions = std::mem::take(&mut self.dirty_sessions).into_iter().filter_map(|id| self.sessions.get(&id).cloned()).collect();
        let uploads = std::mem::take(&mut self.dirty_uploads).into_iter().filter_map(|id| self.uploads.get(&id).cloned()).collect();
        Dirty {
            jobs,
            workers,
            removed_jobs: std::mem::take(&mut self.removed_jobs).into_iter().collect(),
            removed_workers: std::mem::take(&mut self.removed_workers).into_iter().collect(),
            new_envelopes,
            dropped_envelopes: std::mem::take(&mut self.dropped_envelopes).into_iter().collect(),
            dropped_spills: std::mem::take(&mut self.dropped_spills).into_iter().collect(),
            sessions,
            removed_sessions: std::mem::take(&mut self.removed_sessions).into_iter().collect(),
            uploads,
            removed_uploads: std::mem::take(&mut self.removed_uploads).into_iter().collect(),
        }
    }

    fn touch_job(&mut self, id: &str) {
        self.dirty_jobs.insert(id.to_owned());
    }
    fn touch_worker(&mut self, id: &str) {
        self.dirty_workers.insert(id.to_owned());
    }

    fn held(&self, worker: &str) -> u32 {
        self.jobs.values().filter(|j| matches!(j.phase, Phase::Pushed | Phase::Running) && j.worker.as_deref() == Some(worker)).count() as u32
    }

    /// Free job slots of a worker as the dispatcher estimates them
    /// (protocol 2: credits minus offers in flight; protocol 1: capacity
    /// minus held).
    fn free_estimate(&self, w: &WorkerRec) -> u32 {
        match w.credits {
            Some(c) => {
                let in_flight = w.offers_sent.saturating_sub(w.offers_seen);
                c.saturating_sub(u32::try_from(in_flight).unwrap_or(u32::MAX))
            }
            None => w.capacity.saturating_sub(self.held(&w.worker_id)),
        }
    }

    /// Free session slots of a worker (offers in flight subtracted).
    fn session_free_estimate(&self, w: &WorkerRec) -> u32 {
        if w.proto < 2 {
            return 0;
        }
        let offered = self.sessions.values().filter(|x| x.state == SessionState::Offered && x.worker.as_deref() == Some(w.worker_id.as_str())).count() as u32;
        w.session_free.saturating_sub(offered)
    }

    fn position(&self, j: &JobRec) -> u32 {
        self.jobs.values().filter(|o| o.phase == Phase::Queued && o.seq < j.seq).count() as u32
    }

    /// `POST /enqueue`. A job id seen before answers its state (idempotent),
    /// unless it is a `replace` of a job waiting for a restage.
    pub fn enqueue(&mut self, req: EnqueueReq, now: i64) -> (EnqueueResp, Vec<Out>) {
        self.enqueue_spilled(req, None, now)
    }

    /// [`Sched::enqueue`] with the envelope kept by the host at `spill`
    /// (then `req.envelope` is ignored).
    pub fn enqueue_spilled(&mut self, req: EnqueueReq, spill: Option<String>, now: i64) -> (EnqueueResp, Vec<Out>) {
        if req.replace && self.jobs.get(&req.job_id).is_some_and(|j| j.restage && j.phase == Phase::Queued) {
            let id = req.job_id.clone();
            let j = self.jobs.get_mut(&id).expect("checked");
            if let Some(old) = j.spill.take() {
                self.dropped_spills.insert(old);
            }
            j.restage = false;
            j.not_before = now;
            j.first_push_at = None;
            j.envelope = if spill.is_some() { None } else { Some(req.envelope) };
            j.spill = spill;
            if j.envelope.is_some() {
                self.new_envelopes.insert(id.clone());
            }
            self.touch_job(&id);
            let out = self.pump(now);
            let j = &self.jobs[&id];
            let resp = EnqueueResp { job_id: id, state: j.phase.as_str().into(), worker: j.worker.clone(), position: self.position(j), duplicate: false };
            return (resp, out);
        }
        if req.replace && self.jobs.get(&req.job_id).is_some_and(|j| !j.phase.finished()) {
            // Not waiting for a restage: keep the new envelope (inputs now
            // in the store) for a later restage or re-dispatch.
            let id = req.job_id.clone();
            let j = self.jobs.get_mut(&id).expect("checked");
            if let Some(old) = j.spill.take() {
                self.dropped_spills.insert(old);
            }
            j.envelope = if spill.is_some() { None } else { Some(req.envelope) };
            j.spill = spill;
            if j.envelope.is_some() {
                self.new_envelopes.insert(id.clone());
            }
            self.touch_job(&id);
            let j = &self.jobs[&id];
            let resp = EnqueueResp { job_id: id, state: j.phase.as_str().into(), worker: j.worker.clone(), position: self.position(j), duplicate: true };
            return (resp, Vec::new());
        }
        if let Some(j) = self.jobs.get(&req.job_id) {
            let resp = EnqueueResp { job_id: j.job_id.clone(), state: j.phase.as_str().into(), worker: j.worker.clone(), position: self.position(j), duplicate: true };
            return (resp, Vec::new());
        }
        if req.max_queued > 0 {
            let queued = self.jobs.values().filter(|j| j.phase == Phase::Queued && j.model == req.model).count();
            if queued >= req.max_queued as usize {
                let resp = EnqueueResp { job_id: req.job_id, state: "refused".into(), worker: None, position: queued as u32, duplicate: false };
                return (resp, Vec::new());
            }
        }
        self.seq += 1;
        let id = req.job_id.clone();
        let j = JobRec {
            job_id: id.clone(),
            attempt: 1,
            max_attempts: req.retries.saturating_add(1),
            phase: Phase::Queued,
            worker: None,
            model: req.model,
            seq: self.seq,
            enqueued_at: now,
            pushed_at: None,
            acked_at: None,
            finished_at: None,
            deadline: None,
            not_before: now,
            first_push_at: None,
            error: None,
            cancel_requested: false,
            takeover: false,
            failed_here: false,
            timed: false,
            requeued_at: None,
            lease: 0,
            restage: false,
            envelope: if spill.is_some() { None } else { Some(req.envelope) },
            spill,
            result: None,
            owner: req.owner,
        };
        let held_here = j.envelope.is_some();
        self.jobs.insert(id.clone(), j);
        self.touch_job(&id);
        if held_here {
            self.new_envelopes.insert(id.clone());
        }
        let out = self.pump(now);
        let j = &self.jobs[&id];
        let resp = EnqueueResp { job_id: id, state: j.phase.as_str().into(), worker: j.worker.clone(), position: self.position(j), duplicate: false };
        (resp, out)
    }

    /// Client cancel: `None` for an unknown job, else its state after.
    pub fn cancel(&mut self, job_id: &str, now: i64) -> (Option<&'static str>, Vec<Out>) {
        let Some(j) = self.jobs.get_mut(job_id) else { return (None, Vec::new()) };
        let mut out = Vec::new();
        match j.phase {
            Phase::Queued => {
                j.phase = Phase::Cancelled;
                j.finished_at = Some(now);
            }
            Phase::Pushed | Phase::Running => {
                j.cancel_requested = true;
                if let Some(w) = j.worker.clone() {
                    out.push(Out::Send { worker: w, msg: DoMsg::Cancel { job_id: job_id.to_owned() } });
                }
            }
            _ => {}
        }
        let phase = j.phase.as_str();
        if phase == "cancelled" {
            self.drop_envelope(job_id);
        }
        self.touch_job(job_id);
        out.extend(self.reap_uploads(now));
        (Some(phase), out)
    }

    fn drop_envelope(&mut self, id: &str) {
        if let Some(j) = self.jobs.get_mut(id) {
            if j.envelope.take().is_some() {
                self.dropped_envelopes.insert(id.to_owned());
            }
            if let Some(k) = j.spill.take() {
                self.dropped_spills.insert(k);
            }
        }
    }

    /// A frame from `worker` (the socket's identity, not the frame's).
    pub fn on_msg(&mut self, worker: &str, msg: WorkerMsg, now: i64) -> Vec<Out> {
        let mut out = Vec::new();
        if let Some(w) = self.workers.get_mut(worker) {
            w.last_seen = now;
        }
        match msg {
            WorkerMsg::Hello(h) => return self.hello(worker, h, now),
            WorkerMsg::Ack { job_id, attempt, lease, worker_ms } => self.ack(worker, &job_id, attempt, lease, worker_ms, now, &mut out),
            WorkerMsg::Nack { job_id, attempt, retry, code, message } => self.nack(worker, &job_id, attempt, retry, code, &message, now),
            WorkerMsg::Done { job_id, attempt: _, state } => {
                if let Some(j) = self.jobs.get_mut(&job_id) {
                    if !j.phase.finished() && j.worker.as_deref() == Some(worker) {
                        j.phase = Phase::from_worker(&state);
                        j.finished_at = Some(now);
                        j.deadline = None;
                        self.touch_job(&job_id);
                        self.drop_envelope(&job_id);
                    }
                }
            }
            WorkerMsg::Status { running: _, draining, capacity, ready } => {
                if let Some(w) = self.workers.get_mut(worker) {
                    let ready = ready.unwrap_or(true);
                    if w.draining != draining || w.capacity != capacity.max(1) || w.ready != ready {
                        w.draining = draining;
                        w.capacity = capacity.max(1);
                        w.ready = ready;
                        self.dirty_workers.insert(worker.to_owned());
                    }
                }
            }
            WorkerMsg::Slots(sl) => self.slots(worker, sl),
            WorkerMsg::Front(f) => {
                if let Some(w) = self.workers.get_mut(worker) {
                    w.ready = f.ready;
                    w.front = Some(f);
                    self.dirty_workers.insert(worker.to_owned());
                }
            }
            WorkerMsg::UploadInit { req, job_id, attempt, lease, name, content_type, parts } => {
                self.upload_init(worker, req, &job_id, attempt, lease, &name, content_type, parts, now, &mut out)
            }
            WorkerMsg::UploadMore { req, job_id, upload_id, from, count } => self.upload_more(worker, req, &job_id, &upload_id, from, count, &mut out),
            WorkerMsg::UploadDone { req, job_id, attempt: _, lease, upload_id, parts, bytes, sha256 } => {
                self.upload_done(worker, req, &job_id, lease, &upload_id, parts, bytes, sha256, &mut out)
            }
            WorkerMsg::UploadAbort { job_id: _, upload_id } => {
                let key = self.uploads.values().find(|u| u.upload_id == upload_id && u.worker == worker && u.state != UploadState::Done).map(|u| u.key.clone());
                if let Some(k) = key {
                    out.extend(self.drop_upload(&k));
                }
            }
            WorkerMsg::SessionAck { session_id, lease, endpoint } => self.session_ack(worker, &session_id, lease, endpoint, now, &mut out),
            WorkerMsg::SessionNack { session_id, lease, code: _, message } => {
                let offered = self.sessions.get(&session_id).is_some_and(|x| x.state == SessionState::Offered && x.worker.as_deref() == Some(worker) && x.lease == lease);
                if offered {
                    out.extend(self.session_next(&session_id, Some(&message), now));
                }
            }
            WorkerMsg::SessionEnd { session_id, lease } => {
                let mine = self.sessions.get(&session_id).is_some_and(|x| x.state != SessionState::Ended && x.worker.as_deref() == Some(worker) && x.lease == lease);
                if mine {
                    self.end_session(&session_id, "ended by the worker", now, &mut out, false);
                }
            }
        }
        out.extend(self.reap_uploads(now));
        out.extend(self.pump(now));
        out
    }

    fn slots(&mut self, worker: &str, sl: Slots) {
        if let Some(w) = self.workers.get_mut(worker) {
            w.credits = Some(sl.free);
            w.session_free = sl.session_free;
            w.offers_seen = sl.offers_seen;
            w.full_until = 0;
            self.dirty_workers.insert(worker.to_owned());
        }
    }

    fn hello(&mut self, worker: &str, h: Hello, now: i64) -> Vec<Out> {
        let mut out = Vec::new();
        let w = self.workers.entry(worker.to_owned()).or_insert_with(|| WorkerRec {
            worker_id: worker.to_owned(),
            connected: true,
            draining: false,
            capacity: 1,
            version: String::new(),
            sha: String::new(),
            models: Vec::new(),
            caps: Value::Null,
            last_seen: now,
            disconnected_at: None,
            full_until: 0,
            proto: 0,
            credits: None,
            offers_sent: 0,
            offers_seen: 0,
            session_free: 0,
            endpoint: String::new(),
            front: None,
            ready: true,
        });
        w.connected = true;
        w.disconnected_at = None;
        w.draining = h.draining;
        w.capacity = h.capacity.max(1);
        w.version = h.version;
        w.sha = h.sha;
        w.models = h.models;
        w.caps = h.caps;
        w.last_seen = now;
        w.full_until = 0;
        w.proto = h.proto;
        w.endpoint = h.endpoint;
        w.ready = h.front.as_ref().is_none_or(|f| f.ready);
        w.front = h.front;
        // A new socket: offer counters start again on both sides.
        w.offers_sent = 0;
        w.offers_seen = 0;
        w.credits = h.slots.map(|s| s.free);
        w.session_free = h.slots.map_or(0, |s| s.session_free);
        self.touch_worker(worker);

        let reported: BTreeMap<String, crate::Held> = h.jobs.into_iter().map(|x| (x.job_id.clone(), x)).collect();
        let mut cancel = Vec::new();
        for (id, r) in &reported {
            let Some(j) = self.jobs.get_mut(id) else {
                if !r.finished() {
                    // Unknown here (storage lost): it runs there, without an
                    // envelope to re-dispatch it.
                    self.seq += 1;
                    self.jobs.insert(
                        id.clone(),
                        JobRec {
                            job_id: id.clone(),
                            attempt: r.attempt,
                            max_attempts: r.attempt,
                            phase: Phase::Running,
                            worker: Some(worker.to_owned()),
                            model: None,
                            seq: self.seq,
                            enqueued_at: now,
                            pushed_at: None,
                            acked_at: Some(now),
                            finished_at: None,
                            deadline: None,
                            not_before: now,
                            first_push_at: None,
                            error: None,
                            cancel_requested: false,
                            takeover: false,
                            failed_here: false,
                            timed: true,
                            requeued_at: None,
                            lease: r.lease,
                            restage: false,
                            spill: None,
                            result: None,
                            owner: None,
                            envelope: None,
                        },
                    );
                    self.touch_job(id);
                }
                continue;
            };
            // Its copy is current when it holds the last lease (0: a worker
            // without leases, matched by assignment only).
            let current = r.lease == 0 || r.lease == j.lease;
            if j.phase.finished() {
                if !r.finished() && (j.phase == Phase::Cancelled || j.worker.as_deref() != Some(worker) || !current) {
                    cancel.push(id.clone());
                }
                continue;
            }
            if r.finished() {
                if current && (j.worker.as_deref() == Some(worker) || j.phase == Phase::Queued) {
                    j.phase = Phase::from_worker(&r.state);
                    j.worker = Some(worker.to_owned());
                    j.finished_at = Some(now);
                    j.deadline = None;
                    self.touch_job(id);
                    self.drop_envelope(id);
                }
                continue;
            }
            let mine = j.worker.as_deref() == Some(worker);
            let keep = current && (mine || j.phase == Phase::Queued);
            if keep {
                // Pushed or running here, or queued again (ack timeout,
                // grace period) with no newer push: it runs here.
                j.worker = Some(worker.to_owned());
                j.phase = Phase::Running;
                j.acked_at.get_or_insert(now);
                j.deadline = None;
                j.requeued_at = None;
                j.attempt = j.attempt.max(r.attempt);
                self.touch_job(id);
            } else {
                // Given to another worker under a newer lease: drop this copy.
                cancel.push(id.clone());
            }
        }
        // What we gave it that it does not report.
        let missing: Vec<(String, Phase)> = self
            .jobs
            .values()
            .filter(|j| j.worker.as_deref() == Some(worker) && matches!(j.phase, Phase::Pushed | Phase::Running) && !reported.contains_key(&j.job_id))
            .map(|j| (j.job_id.clone(), j.phase))
            .collect();
        for (id, phase) in missing {
            if phase == Phase::Pushed {
                self.requeue(&id, now, 0);
            } else {
                self.lose(&id, &format!("worker {worker} reconnected without the job"), now);
            }
        }
        // Sessions it re-announces: kept when current, else ended there.
        let held: BTreeMap<String, u64> = h.sessions.into_iter().map(|x| (x.session_id, x.lease)).collect();
        let mut end_sessions = Vec::new();
        for (id, lease) in &held {
            let current = self.sessions.get(id).is_some_and(|x| x.state != SessionState::Ended && x.worker.as_deref() == Some(worker) && x.lease == *lease);
            if !current {
                end_sessions.push(id.clone());
                continue;
            }
            if self.sessions.get(id).is_some_and(|x| x.state == SessionState::Offered) {
                // Its ack was lost with the old socket: it holds the GPU.
                self.session_ack(worker, id, *lease, String::new(), now, &mut out);
            }
        }
        let lost: Vec<String> = self
            .sessions
            .values()
            .filter(|x| x.state != SessionState::Ended && x.worker.as_deref() == Some(worker) && !held.contains_key(&x.session_id))
            .map(|x| x.session_id.clone())
            .collect();
        for id in lost {
            self.end_session(&id, "the worker reconnected without it", now, &mut out, false);
        }
        out.push(Out::Send { worker: worker.to_owned(), msg: DoMsg::Welcome { worker_id: worker.to_owned(), pool: self.pool.clone(), cancel, end_sessions } });
        out.extend(self.reap_uploads(now));
        out.extend(self.pump(now));
        out
    }

    #[allow(clippy::too_many_arguments)]
    fn ack(&mut self, worker: &str, id: &str, attempt: u32, lease: u64, worker_ms: u64, now: i64, out: &mut Vec<Out>) {
        let Some(j) = self.jobs.get_mut(id) else { return };
        let cancel = |out: &mut Vec<Out>| out.push(Out::Send { worker: worker.to_owned(), msg: DoMsg::Cancel { job_id: id.to_owned() } });
        let current = lease == 0 || lease == j.lease;
        if j.phase.finished() {
            if j.worker.as_deref() != Some(worker) || j.phase == Phase::Cancelled || !current {
                cancel(out);
            }
            return;
        }
        if !current {
            // An older push (the job went to another worker meanwhile).
            cancel(out);
            return;
        }
        let mine = j.worker.as_deref() == Some(worker);
        match j.phase {
            Phase::Pushed if mine && j.attempt == attempt => {}
            // Late ack after an ack timeout, before any newer push.
            Phase::Queued => {}
            Phase::Running if mine => return,
            _ => {
                cancel(out);
                return;
            }
        }
        j.phase = Phase::Running;
        j.worker = Some(worker.to_owned());
        j.acked_at = Some(now);
        j.deadline = None;
        j.requeued_at = None;
        j.attempt = j.attempt.max(attempt);
        let cancel_requested = j.cancel_requested;
        if !j.timed && j.attempt == 1 {
            if let Some(p) = j.pushed_at {
                j.timed = true;
                self.timings.push_back(((p - j.enqueued_at) as f64, (now - p) as f64, worker_ms as f64));
                if self.timings.len() > TIMINGS_KEEP {
                    self.timings.pop_front();
                }
            }
        }
        if cancel_requested {
            cancel(out);
        }
        self.touch_job(id);
    }

    #[allow(clippy::too_many_arguments)]
    fn nack(&mut self, worker: &str, id: &str, attempt: u32, retry: bool, code: u16, message: &str, now: i64) {
        let Some(j) = self.jobs.get(id) else { return };
        if !(j.phase == Phase::Pushed && j.worker.as_deref() == Some(worker) && j.attempt == attempt) {
            return;
        }
        if code == 424 {
            // A client URL the worker could not fetch: the gateway sends
            // the job again with that input in the store.
            if let Some(j) = self.jobs.get_mut(id) {
                j.phase = Phase::Queued;
                j.worker = None;
                j.deadline = None;
                j.restage = true;
                j.error = Some(format!("an input needs the store: {message}"));
            }
            self.touch_job(id);
            return;
        }
        if !retry {
            self.fail(id, &format!("a worker refused the job: {message}"), now);
            return;
        }
        if code == 429 && self.workers.get(worker).is_some_and(|w| w.credits.is_some()) {
            // Its arbiter is full (another family holds the GPU): offer it
            // elsewhere now; this worker waits for its next `slots`.
            if let Some(w) = self.workers.get_mut(worker) {
                w.credits = Some(0);
                self.dirty_workers.insert(worker.to_owned());
            }
            self.requeue(id, now, 0);
            return;
        }
        let bounced_out = j.first_push_at.is_some_and(|t| now - t > self.cfg.max_bounce_ms);
        if bounced_out {
            self.fail(id, &format!("no worker took the job for {} s (last: {message})", self.cfg.max_bounce_ms / 1000), now);
            return;
        }
        let backoff = if code == 409 { self.cfg.conflict_backoff_ms } else { self.cfg.nack_backoff_ms };
        if matches!(code, 429 | 503) {
            if let Some(w) = self.workers.get_mut(worker) {
                w.full_until = now + 2 * self.cfg.nack_backoff_ms;
                self.dirty_workers.insert(worker.to_owned());
            }
        }
        self.requeue(id, now, backoff);
    }

    /// Back to the queue, same attempt.
    fn requeue(&mut self, id: &str, now: i64, backoff: i64) {
        if let Some(j) = self.jobs.get_mut(id) {
            j.phase = Phase::Queued;
            j.worker = None;
            j.deadline = None;
            j.not_before = now + backoff;
            self.touch_job(id);
        }
    }

    /// The job's worker is lost: re-dispatch (attempt + 1) or fail.
    fn lose(&mut self, id: &str, why: &str, now: i64) {
        let Some(j) = self.jobs.get_mut(id) else { return };
        if j.attempt < j.max_attempts && j.has_envelope() && !j.cancel_requested {
            j.attempt += 1;
            j.phase = Phase::Queued;
            j.worker = None;
            j.takeover = true;
            j.deadline = None;
            j.pushed_at = None;
            j.acked_at = None;
            j.first_push_at = None;
            j.not_before = now;
            j.requeued_at = Some(now);
            j.error = Some(format!("{why}; dispatching again (attempt {})", j.attempt));
            self.touch_job(id);
        } else if j.cancel_requested {
            j.phase = Phase::Cancelled;
            j.finished_at = Some(now);
            self.touch_job(id);
            self.drop_envelope(id);
        } else {
            let why = format!("the worker running this job was lost ({why})");
            self.fail(id, &why, now);
        }
    }

    fn fail(&mut self, id: &str, why: &str, now: i64) {
        if let Some(j) = self.jobs.get_mut(id) {
            j.phase = Phase::Failed;
            j.error = Some(why.to_owned());
            j.finished_at = Some(now);
            j.deadline = None;
            j.failed_here = true;
            self.touch_job(id);
            self.drop_envelope(id);
        }
    }

    /// The worker's last socket closed.
    pub fn disconnect(&mut self, worker: &str, now: i64) -> Vec<Out> {
        if let Some(w) = self.workers.get_mut(worker) {
            if w.connected {
                w.connected = false;
                w.disconnected_at = Some(now);
                self.touch_worker(worker);
            }
        }
        Vec::new()
    }

    /// Timers (the alarm): ack deadlines, lost workers, backoffs, cleanup.
    pub fn tick(&mut self, now: i64) -> Vec<Out> {
        let mut out = Vec::new();
        // Connected but silent: lost.
        let silent: Vec<String> =
            self.workers.values().filter(|w| w.connected && now - w.last_seen > self.cfg.stale_after_ms).map(|w| w.worker_id.clone()).collect();
        for id in silent {
            if let Some(w) = self.workers.get_mut(&id) {
                w.connected = false;
                w.disconnected_at = Some(w.last_seen);
            }
            self.touch_worker(&id);
            out.push(Out::Close { worker: id });
        }
        // Missed acks.
        let late: Vec<(String, Option<String>)> = self
            .jobs
            .values()
            .filter(|j| j.phase == Phase::Pushed && j.deadline.is_some_and(|d| d <= now))
            .map(|j| (j.job_id.clone(), j.worker.clone()))
            .collect();
        for (id, w) in late {
            let connected = w.as_deref().and_then(|w| self.workers.get(w)).is_some_and(|w| w.connected);
            if connected || w.is_none() {
                if let Some(wr) = w.as_deref().and_then(|w| self.workers.get_mut(w)) {
                    wr.full_until = now + self.cfg.ack_timeout_ms;
                }
                if let Some(w) = w {
                    self.touch_worker(&w);
                }
                self.requeue(&id, now, 0);
            }
        }
        // Gone past the grace period.
        let gone: Vec<(String, i64)> = self
            .workers
            .values()
            .filter(|w| !w.connected && w.disconnected_at.is_some_and(|t| now - t >= self.cfg.reconnect_grace_ms))
            .map(|w| (w.worker_id.clone(), now - w.disconnected_at.unwrap_or(now)))
            .collect();
        for (wid, away) in &gone {
            let held: Vec<(String, Phase)> = self
                .jobs
                .values()
                .filter(|j| j.worker.as_deref() == Some(wid.as_str()) && matches!(j.phase, Phase::Pushed | Phase::Running))
                .map(|j| (j.job_id.clone(), j.phase))
                .collect();
            for (id, phase) in held {
                if phase == Phase::Pushed {
                    self.requeue(&id, now, 0);
                } else {
                    self.lose(&id, &format!("worker {wid} disconnected {} s ago", away / 1000), now);
                }
            }
            let sess: Vec<String> =
                self.sessions.values().filter(|x| x.state != SessionState::Ended && x.worker.as_deref() == Some(wid.as_str())).map(|x| x.session_id.clone()).collect();
            for id in sess {
                self.end_session(&id, &format!("worker {wid} disconnected {} s ago", away / 1000), now, &mut out, false);
            }
        }
        // Session offers without an answer, and leases not renewed.
        let late: Vec<String> =
            self.sessions.values().filter(|x| x.state == SessionState::Offered && x.deadline.is_some_and(|d| d <= now)).map(|x| x.session_id.clone()).collect();
        for id in late {
            out.extend(self.session_next(&id, Some("no answer to the offer"), now));
        }
        let expired: Vec<String> =
            self.sessions.values().filter(|x| x.state == SessionState::Live && x.expires_at.is_some_and(|t| t <= now)).map(|x| x.session_id.clone()).collect();
        for id in expired {
            self.end_session(&id, "the lease expired (no renew)", now, &mut out, true);
        }
        let old: Vec<String> =
            self.sessions.values().filter(|x| x.state == SessionState::Ended && x.ended_at.is_some_and(|t| now - t > KEEP_SESSIONS_MS)).map(|x| x.session_id.clone()).collect();
        for id in old {
            self.sessions.remove(&id);
            self.dirty_sessions.remove(&id);
            self.removed_sessions.insert(id);
        }
        // Re-dispatches no worker took.
        let stuck: Vec<(String, String)> = self
            .jobs
            .values()
            .filter(|j| j.phase == Phase::Queued && j.requeued_at.is_some_and(|t| now - t >= self.cfg.redispatch_wait_ms))
            .map(|j| (j.job_id.clone(), j.error.clone().unwrap_or_default()))
            .collect();
        for (id, prev) in stuck {
            let prev = prev.split("; dispatching again").next().unwrap_or_default().to_owned();
            let why = format!("the worker running this job was lost ({prev}); no worker took it again within {} s", self.cfg.redispatch_wait_ms / 1000);
            self.fail(&id, &why, now);
        }
        // Restages the gateway never sent.
        let unstaged: Vec<String> = self
            .jobs
            .values()
            .filter(|j| j.phase == Phase::Queued && j.restage && j.first_push_at.is_some_and(|t| now - t > self.cfg.max_bounce_ms))
            .map(|j| j.job_id.clone())
            .collect();
        for id in unstaged {
            let why = format!("an input could not be fetched and was not restaged within {} s", self.cfg.max_bounce_ms / 1000);
            self.fail(&id, &why, now);
        }
        // Cleanup.
        let old: Vec<String> =
            self.jobs.values().filter(|j| j.phase.finished() && j.finished_at.is_some_and(|t| now - t > self.cfg.keep_finished_ms)).map(|j| j.job_id.clone()).collect();
        for id in old {
            self.jobs.remove(&id);
            self.dirty_jobs.remove(&id);
            self.removed_jobs.insert(id.clone());
            self.dropped_envelopes.insert(id);
        }
        out.extend(self.reap_uploads(now));
        let forget: Vec<String> = self
            .workers
            .values()
            .filter(|w| !w.connected && w.disconnected_at.is_some_and(|t| now - t > FORGET_WORKER_MS))
            .map(|w| w.worker_id.clone())
            .filter(|id| self.held(id) == 0)
            .collect();
        for id in forget {
            self.workers.remove(&id);
            self.dirty_workers.remove(&id);
            self.removed_workers.insert(id);
        }
        out.extend(self.pump(now));
        out
    }

    /// When [`Sched::tick`] next has something to do (may be in the past:
    /// tick now).
    pub fn next_wake(&self, now: i64) -> Option<i64> {
        let mut t: Option<i64> = None;
        let mut at = |x: i64| t = Some(t.map_or(x, |t| t.min(x)));
        for j in self.jobs.values() {
            match j.phase {
                Phase::Pushed => {
                    // A disconnected worker's jobs wait for its grace period.
                    let connected = j.worker.as_deref().and_then(|w| self.workers.get(w)).is_none_or(|w| w.connected);
                    if let (Some(d), true) = (j.deadline, connected) {
                        at(d);
                    }
                }
                Phase::Queued if j.not_before > now => at(j.not_before),
                _ => {}
            }
            if j.phase == Phase::Queued {
                if let Some(t) = j.requeued_at {
                    at(t + self.cfg.redispatch_wait_ms);
                }
                if let (true, Some(t)) = (j.restage, j.first_push_at) {
                    at(t + self.cfg.max_bounce_ms + 1);
                }
            }
            if j.phase.finished() {
                if let Some(f) = j.finished_at {
                    at(f + self.cfg.keep_finished_ms + 1);
                }
            }
        }
        for w in self.workers.values() {
            if w.connected {
                if self.held(&w.worker_id) > 0 {
                    at(w.last_seen + self.cfg.stale_after_ms + 1);
                }
            } else if let Some(d) = w.disconnected_at {
                if self.held(&w.worker_id) > 0 {
                    at(d + self.cfg.reconnect_grace_ms);
                } else {
                    at(d + FORGET_WORKER_MS + 1);
                }
            }
            if w.full_until > now && self.jobs.values().any(|j| j.phase == Phase::Queued) {
                at(w.full_until);
            }
        }
        for x in self.sessions.values() {
            match x.state {
                SessionState::Offered => {
                    if let Some(d) = x.deadline {
                        at(d);
                    }
                }
                SessionState::Live => {
                    if let Some(e) = x.expires_at {
                        at(e);
                    }
                }
                SessionState::Ended => {
                    if let Some(e) = x.ended_at {
                        at(e + KEEP_SESSIONS_MS + 1);
                    }
                }
            }
        }
        for u in self.uploads.values() {
            if u.state != UploadState::Done {
                at(u.expires_at);
            }
        }
        // Queued jobs waiting for a worker are pushed on the next event
        // (a hello, an ack, a done, a slots frame); no timer needed for them.
        t
    }

    /// Pushes queued jobs to free workers.
    pub fn pump(&mut self, now: i64) -> Vec<Out> {
        let mut out = Vec::new();
        let mut queued: Vec<(u64, String)> = self
            .jobs
            .values()
            .filter(|j| j.phase == Phase::Queued && j.not_before <= now && j.has_envelope() && !j.restage)
            .map(|j| (j.seq, j.job_id.clone()))
            .collect();
        queued.sort();
        for (_, id) in queued {
            let model = self.jobs[&id].model.clone();
            let pick = self
                .workers
                .values()
                .filter(|w| w.connected && !w.draining && w.full_until <= now)
                .filter(|w| w.models.is_empty() || model.as_ref().is_none_or(|m| w.models.contains(m)))
                .filter(|w| self.free_estimate(w) > 0)
                .map(|w| (self.held(&w.worker_id), w.worker_id.clone()))
                .min();
            let Some((_, w)) = pick else { continue };
            if let Some(wr) = self.workers.get_mut(&w) {
                wr.offers_sent += 1;
                self.dirty_workers.insert(w.clone());
            }
            let Some(j) = self.jobs.get_mut(&id) else { continue };
            j.phase = Phase::Pushed;
            j.worker = Some(w.clone());
            j.pushed_at = Some(now);
            j.deadline = Some(now + self.cfg.ack_timeout_ms);
            j.first_push_at.get_or_insert(now);
            j.lease += 1;
            let msg = DoMsg::Job { job_id: id.clone(), attempt: j.attempt, envelope: j.envelope.clone().unwrap_or(Value::Null), lease: j.lease, takeover: j.takeover };
            let spill = if j.envelope.is_none() { j.spill.clone() } else { None };
            self.touch_job(&id);
            match spill {
                Some(key) => out.push(Out::PushSpilled { worker: w, key, msg }),
                None => out.push(Out::Send { worker: w, msg }),
            }
        }
        out
    }

    /// `GET /status`.
    pub fn status(&self, now: i64) -> PoolStatus {
        let count = |p: Phase| self.jobs.values().filter(|j| j.phase == p).count() as u32;
        let workers = self
            .workers
            .values()
            .map(|w| WorkerInfo {
                worker_id: w.worker_id.clone(),
                connected: w.connected,
                draining: w.draining,
                capacity: w.capacity,
                held: self.held(&w.worker_id),
                version: w.version.clone(),
                sha: w.sha.clone(),
                last_seen_ms: w.last_seen,
                caps: w.caps.clone(),
                free: self.free_estimate(w),
                session_free: self.session_free_estimate(w),
                endpoint: w.endpoint.clone(),
                proto: w.proto,
                front: w.front.clone(),
                ready: w.ready,
            })
            .collect();
        let mut owners = BTreeMap::new();
        for j in self.jobs.values().filter(|j| !j.phase.finished()) {
            if let Some(o) = &j.owner {
                *owners.entry(o.clone()).or_insert(0u32) += 1;
            }
        }
        let failed = self
            .jobs
            .values()
            .filter(|j| j.failed_here && j.phase == Phase::Failed)
            .map(|j| FailedJob { job_id: j.job_id.clone(), attempt: j.attempt, error: j.error.clone().unwrap_or_default(), at_ms: j.finished_at.unwrap_or(now) })
            .collect();
        let restage = self.jobs.values().filter(|j| j.restage && j.phase == Phase::Queued).map(|j| j.job_id.clone()).collect();
        PoolStatus {
            pool: self.pool.clone(),
            now_ms: now,
            restage,
            queued: count(Phase::Queued),
            pushed: count(Phase::Pushed),
            running: count(Phase::Running),
            workers,
            failed,
            timings: self.timings(),
            dispatcher: String::new(),
            sessions: self
                .sessions
                .values()
                .filter(|x| x.state != SessionState::Ended)
                .map(|x| SessionInfo {
                    session_id: x.session_id.clone(),
                    state: x.state.as_str().into(),
                    worker: x.worker.clone(),
                    lease: x.lease,
                    kind: x.kind.clone(),
                    expires_ms: x.expires_at.unwrap_or(0),
                })
                .collect(),
            owners,
        }
    }

    /// A job's phase for a [`crate::JobWait`] (`unknown` when not here).
    pub fn job_phase(&self, id: &str) -> &'static str {
        self.jobs.get(id).map_or("unknown", |j| j.phase.as_str())
    }

    /// `GET /metrics`: the demand signal of this dispatcher.
    pub fn metrics(&self, now: i64) -> FamilyMetrics {
        let count = |p: Phase| self.jobs.values().filter(|j| j.phase == p).count() as u32;
        let oldest = self.jobs.values().filter(|j| j.phase == Phase::Queued).map(|j| j.enqueued_at).min().map_or(0, |t| (now - t).max(0));
        let conn: Vec<&WorkerRec> = self.workers.values().filter(|w| w.connected).collect();
        let t = self.timings();
        FamilyMetrics {
            family: self.pool.strip_prefix("family:").unwrap_or(&self.pool).to_owned(),
            now_ms: now,
            queued: count(Phase::Queued),
            oldest_queued_ms: oldest,
            pushed: count(Phase::Pushed),
            running: count(Phase::Running),
            workers: conn.len() as u32,
            slots_total: conn.iter().filter(|w| !w.draining).map(|w| w.capacity).sum(),
            slots_free: conn.iter().filter(|w| !w.draining).map(|w| self.free_estimate(w)).sum(),
            sessions_live: self.sessions.values().filter(|x| x.state == SessionState::Live).count() as u32,
            session_capacity: conn.iter().filter(|w| !w.draining).map(|w| self.session_free_estimate(w)).sum(),
            failed_1h: self.jobs.values().filter(|j| j.failed_here && j.phase == Phase::Failed && j.finished_at.is_some_and(|f| now - f <= 3_600_000)).count() as u32,
            queue_p50_ms: t.queue_p50_ms,
            ack_p50_ms: t.ack_p50_ms,
            uploads_open: self.uploads.values().filter(|u| u.state != UploadState::Done).count() as u32,
        }
    }

    // ------------------------------------------------------------ sessions

    /// `POST /sessions`: offer a session to a worker with a free session
    /// slot. `Pending`: the host waits for [`Out::SessionReady`].
    pub fn admit(&mut self, req: SessionReq, now: i64) -> (Admit, Vec<Out>) {
        self.seq += 1;
        let id = req.session_id.clone().filter(|s| crate::valid_id(s)).unwrap_or_else(|| format!("ses-{now:x}-{}", self.seq));
        if let Some(x) = self.sessions.get(&id) {
            match x.state {
                SessionState::Live => return (Admit::Granted(x.grant()), Vec::new()),
                SessionState::Offered => return (Admit::Pending(id), Vec::new()),
                SessionState::Ended => {}
            }
        }
        let ttl = if req.ttl_ms > 0 { req.ttl_ms } else { self.cfg.session_ttl_ms };
        self.sessions.insert(
            id.clone(),
            SessionRec {
                session_id: id.clone(),
                model: req.model,
                kind: req.kind,
                owner: req.owner,
                worker: None,
                lease: 0,
                state: SessionState::Offered,
                created_at: now,
                deadline: None,
                ttl_ms: ttl,
                expires_at: None,
                tried: Vec::new(),
                endpoint: String::new(),
                end_reason: None,
                ended_at: None,
            },
        );
        self.dirty_sessions.insert(id.clone());
        let mut out = Vec::new();
        match self.offer_session(&id, now, &mut out) {
            Ok(()) => (Admit::Pending(id), out),
            Err(why) => {
                self.finish_session(&id, &why, now);
                (Admit::Refused(why), out)
            }
        }
    }

    /// Offers session `id` to the next worker; `Err` when none is left.
    fn offer_session(&mut self, id: &str, now: i64, out: &mut Vec<Out>) -> Result<(), String> {
        let Some(x) = self.sessions.get(id) else { return Err("unknown session".into()) };
        if x.tried.len() as u32 >= self.cfg.session_max_tries {
            return Err(format!("no worker took the session ({} tried)", x.tried.len()));
        }
        let (model, tried) = (x.model.clone(), x.tried.clone());
        let pick = self
            .workers
            .values()
            .filter(|w| w.connected && !w.draining && !tried.contains(&w.worker_id))
            .filter(|w| w.models.is_empty() || model.as_ref().is_none_or(|m| w.models.contains(m)))
            .filter(|w| self.session_free_estimate(w) > 0)
            .map(|w| (self.held(&w.worker_id), w.worker_id.clone()))
            .min();
        let Some((_, w)) = pick else {
            return Err(if tried.is_empty() { "no worker has a free session slot".into() } else { format!("no other worker has a free session slot ({} refused)", tried.len()) });
        };
        let x = self.sessions.get_mut(id).expect("checked");
        x.lease += 1;
        x.worker = Some(w.clone());
        x.state = SessionState::Offered;
        x.deadline = Some(now + self.cfg.session_offer_timeout_ms);
        let msg = DoMsg::SessionOffer { session_id: id.to_owned(), lease: x.lease, model: x.model.clone(), kind: x.kind.clone(), ttl_ms: x.ttl_ms };
        self.dirty_sessions.insert(id.to_owned());
        out.push(Out::Send { worker: w, msg });
        Ok(())
    }

    /// The offered worker refused or did not answer: try the next one.
    fn session_next(&mut self, id: &str, why: Option<&str>, now: i64) -> Vec<Out> {
        let mut out = Vec::new();
        if let Some(x) = self.sessions.get_mut(id) {
            if let Some(w) = x.worker.take() {
                x.tried.push(w);
            }
            x.deadline = None;
        }
        self.dirty_sessions.insert(id.to_owned());
        if let Err(e) = self.offer_session(id, now, &mut out) {
            let e = match why {
                Some(w) => format!("{e}; last: {w}"),
                None => e,
            };
            self.finish_session(id, &e, now);
            out.push(Out::SessionReady { session_id: id.to_owned(), result: Err(e) });
        }
        out
    }

    fn session_ack(&mut self, worker: &str, id: &str, lease: u64, endpoint: String, now: i64, out: &mut Vec<Out>) {
        let fallback = self.workers.get(worker).map(|w| w.endpoint.clone()).unwrap_or_default();
        let Some(x) = self.sessions.get_mut(id) else {
            out.push(Out::Send { worker: worker.to_owned(), msg: DoMsg::SessionRevoke { session_id: id.to_owned(), reason: "unknown session".into() } });
            return;
        };
        let mine = x.worker.as_deref() == Some(worker) && x.lease == lease;
        match x.state {
            SessionState::Offered if mine => {
                x.state = SessionState::Live;
                x.deadline = None;
                x.expires_at = Some(now + x.ttl_ms);
                x.endpoint = if endpoint.is_empty() { fallback } else { endpoint };
                let g = x.grant();
                self.dirty_sessions.insert(id.to_owned());
                out.push(Out::SessionReady { session_id: id.to_owned(), result: Ok(g) });
            }
            SessionState::Live if mine => {}
            _ => out.push(Out::Send { worker: worker.to_owned(), msg: DoMsg::SessionRevoke { session_id: id.to_owned(), reason: "the session is not offered to this worker".into() } }),
        }
    }

    fn finish_session(&mut self, id: &str, why: &str, now: i64) {
        if let Some(x) = self.sessions.get_mut(id) {
            x.state = SessionState::Ended;
            x.deadline = None;
            x.ended_at = Some(now);
            x.end_reason = Some(why.to_owned());
            self.dirty_sessions.insert(id.to_owned());
        }
    }

    /// Ends a session: the worker is told (`revoke`) unless it ended it, and
    /// a pending admission is answered.
    fn end_session(&mut self, id: &str, why: &str, now: i64, out: &mut Vec<Out>, revoke: bool) {
        let Some(x) = self.sessions.get(id) else { return };
        if x.state == SessionState::Ended {
            return;
        }
        let (state, worker) = (x.state, x.worker.clone());
        self.finish_session(id, why, now);
        if revoke {
            if let Some(w) = worker {
                out.push(Out::Send { worker: w, msg: DoMsg::SessionRevoke { session_id: id.to_owned(), reason: why.to_owned() } });
            }
        }
        if state == SessionState::Offered {
            out.push(Out::SessionReady { session_id: id.to_owned(), result: Err(why.to_owned()) });
        }
    }

    /// Extends a live session's lease (`lease` 0: any).
    pub fn renew(&mut self, id: &str, lease: u64, now: i64) -> Option<SessionGrant> {
        let x = self.sessions.get_mut(id)?;
        if x.state != SessionState::Live || (lease != 0 && lease != x.lease) {
            return None;
        }
        x.expires_at = Some(now + x.ttl_ms);
        self.dirty_sessions.insert(id.to_owned());
        Some(self.sessions[id].grant())
    }

    /// Ends a session on the caller's request; `None` for an unknown one.
    pub fn release(&mut self, id: &str, now: i64) -> (Option<&'static str>, Vec<Out>) {
        let mut out = Vec::new();
        if !self.sessions.contains_key(id) {
            return (None, out);
        }
        self.end_session(id, "released", now, &mut out, true);
        out.extend(self.pump(now));
        (Some("ended"), out)
    }

    // ------------------------------------------------------------- uploads

    /// Whether `worker` holds `job_id` under `lease` now.
    fn holds(&self, worker: &str, job_id: &str, lease: u64) -> bool {
        self.jobs.get(job_id).is_some_and(|j| !j.phase.finished() && j.worker.as_deref() == Some(worker) && j.lease == lease)
    }

    fn grant_err(worker: &str, req: u64, job_id: &str, e: &str) -> Out {
        Out::Send {
            worker: worker.to_owned(),
            msg: DoMsg::UploadGrant {
                req,
                job_id: job_id.to_owned(),
                upload_id: String::new(),
                key: String::new(),
                bucket: String::new(),
                part_urls: Vec::new(),
                expires_ms: 0,
                error: Some(e.to_owned()),
            },
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn upload_init(&mut self, worker: &str, req: u64, job_id: &str, attempt: u32, lease: u64, name: &str, content_type: String, parts: u16, now: i64, out: &mut Vec<Out>) {
        if !crate::valid_id(name) {
            out.push(Self::grant_err(worker, req, job_id, "invalid object name"));
            return;
        }
        if !self.holds(worker, job_id, lease) {
            out.push(Self::grant_err(worker, req, job_id, "this worker does not hold the job under this lease"));
            return;
        }
        let scope = self.pool.strip_prefix("family:").unwrap_or(&self.pool).to_owned();
        let key = format!("{}{scope}/{job_id}/{attempt}-{lease}/{name}", self.cfg.upload_prefix);
        let parts = parts.clamp(1, 10_000);
        match self.uploads.get(&key).map(|u| (u.state, u.upload_id.clone(), u.expires_at)) {
            Some((UploadState::Open, upload_id, _)) => {
                // A repeated init (after a reconnect): fresh URLs.
                let exp = now + self.cfg.upload_ttl_ms;
                if let Some(u) = self.uploads.get_mut(&key) {
                    u.expires_at = exp;
                }
                self.dirty_uploads.insert(key.clone());
                out.push(Out::Grant { worker: worker.to_owned(), req, job_id: job_id.to_owned(), key, upload_id, from: 1, count: parts, expires_ms: exp });
            }
            Some(_) => out.push(Self::grant_err(worker, req, job_id, "the upload is being created or completed")),
            None => {
                self.uploads.insert(
                    key.clone(),
                    UploadRec {
                        key: key.clone(),
                        upload_id: String::new(),
                        job_id: job_id.to_owned(),
                        attempt,
                        lease,
                        worker: worker.to_owned(),
                        state: UploadState::Creating,
                        content_type: content_type.clone(),
                        created_at: now,
                        expires_at: now + self.cfg.upload_ttl_ms,
                        bytes: 0,
                        sha256: String::new(),
                    },
                );
                self.dirty_uploads.insert(key.clone());
                out.push(Out::CreateUpload { worker: worker.to_owned(), req, key, content_type, parts });
            }
        }
    }

    /// The host created (or failed to create) the multipart upload of `key`.
    pub fn upload_created(&mut self, key: &str, req: u64, parts: u16, result: Result<String, String>, now: i64) -> Vec<Out> {
        let mut out = Vec::new();
        let Some(u) = self.uploads.get(key).cloned() else {
            if let Ok(id) = result {
                out.push(Out::AbortUpload { key: key.to_owned(), upload_id: id });
            }
            return out;
        };
        if u.state != UploadState::Creating {
            return out;
        }
        match result {
            Ok(upload_id) if self.holds(&u.worker, &u.job_id, u.lease) => {
                let exp = now + self.cfg.upload_ttl_ms;
                let rec = self.uploads.get_mut(key).expect("checked");
                rec.upload_id = upload_id.clone();
                rec.state = UploadState::Open;
                rec.expires_at = exp;
                self.dirty_uploads.insert(key.to_owned());
                out.push(Out::Grant { worker: u.worker, req, job_id: u.job_id, key: key.to_owned(), upload_id, from: 1, count: parts.clamp(1, 10_000), expires_ms: exp });
            }
            Ok(upload_id) => {
                // The job moved on while the upload was being created.
                self.uploads.remove(key);
                self.removed_uploads.insert(key.to_owned());
                out.push(Out::AbortUpload { key: key.to_owned(), upload_id });
                out.push(Self::grant_err(&u.worker, req, &u.job_id, "the job moved on (lease changed)"));
            }
            Err(e) => {
                self.uploads.remove(key);
                self.removed_uploads.insert(key.to_owned());
                out.push(Self::grant_err(&u.worker, req, &u.job_id, &format!("creating the upload failed: {e}")));
            }
        }
        out
    }

    #[allow(clippy::too_many_arguments)]
    fn upload_more(&mut self, worker: &str, req: u64, job_id: &str, upload_id: &str, from: u16, count: u16, out: &mut Vec<Out>) {
        let found = self.uploads.values().find(|u| u.upload_id == upload_id && u.state == UploadState::Open).map(|u| (u.key.clone(), u.worker.clone(), u.lease, u.expires_at));
        match found {
            Some((key, w, lease, exp)) if w == worker && self.holds(worker, job_id, lease) => {
                out.push(Out::Grant { worker: worker.to_owned(), req, job_id: job_id.to_owned(), key, upload_id: upload_id.to_owned(), from: from.max(1), count: count.clamp(1, 10_000), expires_ms: exp });
            }
            _ => out.push(Self::grant_err(worker, req, job_id, "no open upload of this job for this worker")),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn upload_done(&mut self, worker: &str, req: u64, job_id: &str, lease: u64, upload_id: &str, parts: Vec<Part>, bytes: u64, sha256: String, out: &mut Vec<Out>) {
        let committed = |key: String, ok: bool, bytes: u64, error: Option<String>| Out::Send {
            worker: worker.to_owned(),
            msg: DoMsg::UploadCommitted { req, job_id: job_id.to_owned(), key, bucket: String::new(), bytes, ok, error },
        };
        let Some(u) = self.uploads.values().find(|u| u.upload_id == upload_id && u.job_id == job_id).cloned() else {
            out.push(committed(String::new(), false, 0, Some("no such upload".into())));
            return;
        };
        if u.state == UploadState::Done {
            out.push(committed(u.key, true, u.bytes, None));
            return;
        }
        if u.worker != worker || u.lease != lease || !self.holds(worker, job_id, lease) {
            out.push(committed(u.key.clone(), false, 0, Some("fenced: the job is held under a newer lease".into())));
            out.extend(self.drop_upload(&u.key));
            return;
        }
        if u.state != UploadState::Open {
            return;
        }
        let rec = self.uploads.get_mut(&u.key).expect("found");
        rec.state = UploadState::Completing;
        rec.bytes = bytes;
        rec.sha256 = sha256;
        self.dirty_uploads.insert(u.key.clone());
        out.push(Out::CompleteUpload { worker: worker.to_owned(), req, key: u.key, upload_id: upload_id.to_owned(), parts, bytes });
    }

    /// The host completed (`Ok(object size)`) or failed to complete `key`.
    pub fn upload_completed(&mut self, key: &str, req: u64, result: Result<u64, String>, _now: i64) -> Vec<Out> {
        let mut out = Vec::new();
        let Some(u) = self.uploads.get(key).cloned() else { return out };
        if u.state != UploadState::Completing {
            return out;
        }
        let msg = |ok: bool, bytes: u64, error: Option<String>| DoMsg::UploadCommitted { req, job_id: u.job_id.clone(), key: key.to_owned(), bucket: String::new(), bytes, ok, error };
        match result {
            Ok(size) if size == u.bytes => {
                let rec = self.uploads.get_mut(key).expect("checked");
                rec.state = UploadState::Done;
                self.dirty_uploads.insert(key.to_owned());
                if let Some(j) = self.jobs.get_mut(&u.job_id) {
                    j.result = Some(JobResult { key: key.to_owned(), bytes: size, sha256: u.sha256.clone() });
                    self.dirty_jobs.insert(u.job_id.clone());
                }
                out.push(Out::Send { worker: u.worker.clone(), msg: msg(true, size, None) });
            }
            other => {
                let e = match other {
                    Ok(size) => format!("the object has {size} bytes, the worker sent {}", u.bytes),
                    Err(e) => format!("completing the upload failed: {e}"),
                };
                out.push(Out::Send { worker: u.worker.clone(), msg: msg(false, 0, Some(e)) });
                out.extend(self.drop_upload(key));
            }
        }
        out
    }

    /// Forgets an unfinished upload and aborts it at the store.
    fn drop_upload(&mut self, key: &str) -> Vec<Out> {
        let mut out = Vec::new();
        if let Some(u) = self.uploads.remove(key) {
            self.dirty_uploads.remove(key);
            self.removed_uploads.insert(key.to_owned());
            if !u.upload_id.is_empty() && u.state != UploadState::Done {
                out.push(Out::AbortUpload { key: u.key, upload_id: u.upload_id });
            }
        }
        out
    }

    /// Aborts uploads nobody will complete: expired, or of a job that moved
    /// to a newer lease or ended without them; forgets finished ones whose
    /// job is gone.
    fn reap_uploads(&mut self, now: i64) -> Vec<Out> {
        let mut out = Vec::new();
        let stale: Vec<String> = self
            .uploads
            .values()
            .filter(|u| match u.state {
                UploadState::Done => !self.jobs.contains_key(&u.job_id),
                UploadState::Creating | UploadState::Completing => u.expires_at <= now,
                UploadState::Open => {
                    u.expires_at <= now
                        || self.jobs.get(&u.job_id).is_none_or(|j| j.lease != u.lease || j.worker.as_deref() != Some(u.worker.as_str()) || j.phase.finished())
                }
            })
            .map(|u| u.key.clone())
            .collect();
        for k in stale {
            out.extend(self.drop_upload(&k));
        }
        out
    }

    fn timings(&self) -> DispatchTimings {
        fn p50(mut v: Vec<f64>) -> (f64, f64) {
            if v.is_empty() {
                return (0.0, 0.0);
            }
            v.sort_by(|a, b| a.total_cmp(b));
            (v[v.len() / 2], v[v.len() - 1])
        }
        let (q50, qmax) = p50(self.timings.iter().map(|t| t.0).collect());
        let (a50, amax) = p50(self.timings.iter().map(|t| t.1).collect());
        let (w50, _) = p50(self.timings.iter().map(|t| t.2).collect());
        DispatchTimings { count: self.timings.len() as u32, queue_p50_ms: q50, queue_max_ms: qmax, ack_p50_ms: a50, ack_max_ms: amax, worker_p50_ms: w50 }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Held;
    use serde_json::json;

    fn hello(id: &str, cap: u32, jobs: Vec<Held>) -> WorkerMsg {
        WorkerMsg::Hello(Hello { worker_id: id.into(), pool: "p".into(), capacity: cap, jobs, ..Hello::default() })
    }
    fn enq(s: &mut Sched, id: &str, now: i64) -> (EnqueueResp, Vec<Out>) {
        s.enqueue(EnqueueReq { job_id: id.into(), envelope: json!({"job": {"id": id}}), retries: 1, model: None, replace: false, owner: None, max_queued: 0 }, now)
    }
    fn pushes(out: &[Out]) -> Vec<(String, String, u32, bool)> {
        out.iter()
            .filter_map(|o| match o {
                Out::Send { worker, msg: DoMsg::Job { job_id, attempt, takeover, .. } } => Some((worker.clone(), job_id.clone(), *attempt, *takeover)),
                _ => None,
            })
            .collect()
    }
    fn cancels(out: &[Out]) -> Vec<(String, String)> {
        out.iter()
            .filter_map(|o| match o {
                Out::Send { worker, msg: DoMsg::Cancel { job_id } } => Some((worker.clone(), job_id.clone())),
                _ => None,
            })
            .collect()
    }
    fn ack(s: &mut Sched, w: &str, j: &str, attempt: u32, now: i64) -> Vec<Out> {
        s.on_msg(w, WorkerMsg::Ack { job_id: j.into(), attempt, lease: 0, worker_ms: 3 }, now)
    }

    #[test]
    fn push_on_enqueue_and_on_free_slot() {
        let mut s = Sched::new("p", Cfg::default());
        let (r, out) = enq(&mut s, "a", 0);
        assert_eq!(r.state, "queued");
        assert!(out.is_empty());
        // A worker connects: the queued job is pushed at once.
        let out = s.on_msg("w1", hello("w1", 1, vec![]), 10);
        assert_eq!(pushes(&out), vec![("w1".into(), "a".into(), 1, false)]);
        ack(&mut s, "w1", "a", 1, 15);
        assert_eq!(s.job("a").unwrap().phase, Phase::Running);
        let (r, out) = enq(&mut s, "b", 20);
        assert_eq!((r.state.as_str(), r.position), ("queued", 0));
        assert!(pushes(&out).is_empty());
        let out = s.on_msg("w1", WorkerMsg::Done { job_id: "a".into(), attempt: 1, state: "succeeded".into() }, 30);
        assert_eq!(pushes(&out), vec![("w1".into(), "b".into(), 1, false)]);
        let st = s.status(40);
        assert_eq!((st.queued, st.pushed, st.running), (0, 1, 0));
        assert_eq!(st.timings.count, 1);
        assert_eq!(st.timings.queue_p50_ms, 10.0);
        // Duplicate enqueue: idempotent.
        let (r, out) = enq(&mut s, "b", 50);
        assert!(r.duplicate && out.is_empty());
    }

    #[test]
    fn least_loaded_worker_and_capacity() {
        let mut s = Sched::new("p", Cfg::default());
        s.on_msg("w1", hello("w1", 2, vec![]), 0);
        s.on_msg("w2", hello("w2", 2, vec![]), 0);
        let mut to = Vec::new();
        for i in 0..5 {
            let (_, out) = enq(&mut s, &format!("j{i}"), 1);
            to.extend(pushes(&out).into_iter().map(|p| p.0));
        }
        assert_eq!(to, vec!["w1", "w2", "w1", "w2"]);
        assert_eq!(s.status(2).queued, 1);
    }

    #[test]
    fn ack_timeout_requeues() {
        let mut s = Sched::new("p", Cfg::default());
        s.on_msg("w1", hello("w1", 1, vec![]), 0);
        s.on_msg("w2", hello("w2", 1, vec![]), 0);
        let (_, out) = enq(&mut s, "a", 0);
        assert_eq!(pushes(&out)[0].0, "w1");
        let wake = s.next_wake(0).unwrap();
        assert_eq!(wake, s.cfg.ack_timeout_ms);
        let out = s.tick(wake);
        // w1 is skipped for a while: w2 gets it, same attempt, next lease.
        assert_eq!(pushes(&out), vec![("w2".into(), "a".into(), 1, false)]);
        assert_eq!(s.job("a").unwrap().lease, 2);
        // w1's late ack (lease 1) is stale: w1 is told to drop it.
        let late = WorkerMsg::Ack { job_id: "a".into(), attempt: 1, lease: 1, worker_ms: 3 };
        let out = s.on_msg("w1", late, wake + 5);
        assert_eq!(cancels(&out), vec![("w1".into(), "a".into())]);
        let out = s.on_msg("w2", WorkerMsg::Ack { job_id: "a".into(), attempt: 1, lease: 2, worker_ms: 3 }, wake + 6);
        assert!(cancels(&out).is_empty());
        assert_eq!((s.job("a").unwrap().phase, s.job("a").unwrap().worker.as_deref()), (Phase::Running, Some("w2")));
    }

    #[test]
    fn nack_retry_and_final() {
        let mut s = Sched::new("p", Cfg::default());
        s.on_msg("w1", hello("w1", 1, vec![]), 0);
        enq(&mut s, "a", 0);
        let out = s.on_msg("w1", WorkerMsg::Nack { job_id: "a".into(), attempt: 1, retry: true, code: 409, message: "held".into() }, 1);
        assert!(pushes(&out).is_empty());
        let t = s.next_wake(1).unwrap();
        assert_eq!(t, 1 + s.cfg.conflict_backoff_ms);
        let out = s.tick(t);
        assert_eq!(pushes(&out).len(), 1);
        let out = s.on_msg("w1", WorkerMsg::Nack { job_id: "a".into(), attempt: 1, retry: false, code: 400, message: "bad".into() }, t + 1);
        assert!(pushes(&out).is_empty());
        let j = s.job("a").unwrap();
        assert_eq!(j.phase, Phase::Failed);
        assert!(j.failed_here);
        assert_eq!(s.status(t + 2).failed.len(), 1);
    }

    #[test]
    fn worker_loss_redispatches_once_then_fails() {
        let mut s = Sched::new("p", Cfg::default());
        s.on_msg("w1", hello("w1", 1, vec![]), 0);
        s.on_msg("w2", hello("w2", 1, vec![]), 0);
        enq(&mut s, "a", 0);
        ack(&mut s, "w1", "a", 1, 1);
        s.disconnect("w1", 100);
        // Within the grace period nothing moves.
        assert!(pushes(&s.tick(100 + s.cfg.reconnect_grace_ms - 1)).is_empty());
        let grace = s.cfg.reconnect_grace_ms;
        let out = s.tick(100 + grace);
        assert_eq!(pushes(&out), vec![("w2".into(), "a".into(), 2, true)]);
        ack(&mut s, "w2", "a", 2, 100 + grace + 1);
        // w2 goes silent (connected, no frames): lost for good.
        let t = 100 + grace + 1 + s.cfg.stale_after_ms + 1;
        let out = s.tick(t);
        assert!(out.contains(&Out::Close { worker: "w2".into() }));
        let out = s.tick(t + grace);
        assert!(pushes(&out).is_empty());
        let j = s.job("a").unwrap();
        assert_eq!(j.phase, Phase::Failed);
        assert!(j.error.as_deref().unwrap().contains("lost"));
    }

    #[test]
    fn redeploy_reconnect_keeps_running_jobs() {
        let mut s = Sched::new("p", Cfg::default());
        s.on_msg("w1", hello("w1", 2, vec![]), 0);
        enq(&mut s, "run", 0);
        enq(&mut s, "pushed", 0);
        ack(&mut s, "w1", "run", 1, 1);
        // Redeploy: the state is restored from storage, the socket is gone.
        let d = s.take_dirty();
        let jobs: Vec<JobRec> = s.jobs().cloned().collect();
        let workers: Vec<WorkerRec> = s.workers().cloned().collect();
        assert!(!d.jobs.is_empty());
        let mut s = Sched::restore("p", Cfg::default(), jobs, workers);
        s.disconnect("w1", 10);
        // It comes back holding `run` (never got `pushed`) plus one it
        // finished meanwhile that we never heard of.
        let held = vec![
            Held { job_id: "run".into(), attempt: 1, lease: 1, state: "running".into() },
            Held { job_id: "ghost".into(), attempt: 1, lease: 1, state: "succeeded".into() },
        ];
        let out = s.on_msg("w1", hello("w1", 2, held), 2_000);
        assert_eq!(s.job("run").unwrap().phase, Phase::Running);
        assert_eq!(s.job("run").unwrap().attempt, 1);
        // `pushed` was lost in flight: pushed again, same attempt.
        assert_eq!(pushes(&out), vec![("w1".into(), "pushed".into(), 1, false)]);
        assert!(s.job("ghost").is_none());
        // Past the grace period nothing is lost.
        assert!(pushes(&s.tick(2_000 + s.cfg.reconnect_grace_ms + 5)).is_empty());
        assert_eq!(s.job("run").unwrap().phase, Phase::Running);
    }

    #[test]
    fn redispatch_without_workers_fails() {
        let mut s = Sched::new("p", Cfg::default());
        s.on_msg("w1", hello("w1", 1, vec![]), 0);
        enq(&mut s, "a", 0);
        ack(&mut s, "w1", "a", 1, 1);
        s.disconnect("w1", 10);
        let t = 10 + s.cfg.reconnect_grace_ms;
        s.tick(t);
        assert_eq!(s.job("a").unwrap().phase, Phase::Queued);
        let wake = s.next_wake(t).unwrap();
        assert_eq!(wake, t + s.cfg.redispatch_wait_ms);
        s.tick(wake);
        let j = s.job("a").unwrap();
        assert_eq!(j.phase, Phase::Failed);
        assert!(j.error.as_deref().unwrap().starts_with("the worker running this job was lost (worker w1 disconnected"), "{:?}", j.error);
        assert_eq!(s.status(wake).failed.len(), 1);
    }

    #[test]
    fn reconnect_after_loss_cancels_the_zombie() {
        let mut s = Sched::new("p", Cfg::default());
        s.on_msg("w1", hello("w1", 1, vec![]), 0);
        s.on_msg("w2", hello("w2", 1, vec![]), 0);
        enq(&mut s, "a", 0);
        ack(&mut s, "w1", "a", 1, 1);
        s.disconnect("w1", 10);
        let t = 10 + s.cfg.reconnect_grace_ms;
        s.tick(t);
        ack(&mut s, "w2", "a", 2, t + 1);
        // w1 comes back, still running attempt 1 under lease 1: told to drop it.
        let out = s.on_msg("w1", hello("w1", 1, vec![Held { job_id: "a".into(), attempt: 1, lease: 1, state: "running".into() }]), t + 2);
        let welcome = out.iter().find_map(|o| match o {
            Out::Send { msg: DoMsg::Welcome { cancel, .. }, .. } => Some(cancel.clone()),
            _ => None,
        });
        assert_eq!(welcome, Some(vec!["a".to_string()]));
        assert_eq!(s.job("a").unwrap().worker.as_deref(), Some("w2"));
    }

    #[test]
    fn cancel_paths() {
        let mut s = Sched::new("p", Cfg::default());
        enq(&mut s, "q", 0);
        assert_eq!(s.cancel("q", 1).0, Some("cancelled"));
        s.on_msg("w1", hello("w1", 1, vec![]), 2);
        enq(&mut s, "r", 3);
        let (st, out) = s.cancel("r", 4);
        assert_eq!(st, Some("pushed"));
        assert_eq!(cancels(&out), vec![("w1".into(), "r".into())]);
        assert_eq!(s.cancel("nope", 5).0, None);
        let d = s.take_dirty();
        assert!(d.dropped_envelopes.contains(&"q".to_string()));
    }

    #[test]
    fn draining_and_models() {
        let mut s = Sched::new("p", Cfg::default());
        let mut h = Hello { worker_id: "w1".into(), pool: "p".into(), capacity: 4, models: vec!["m1".into()], ..Hello::default() };
        s.on_msg("w1", WorkerMsg::Hello(h.clone()), 0);
        let (_, out) = s.enqueue(EnqueueReq { job_id: "x".into(), envelope: json!({}), retries: 0, model: Some("m2".into()), replace: false, owner: None, max_queued: 0 }, 1);
        assert!(pushes(&out).is_empty());
        let (_, out) = s.enqueue(EnqueueReq { job_id: "y".into(), envelope: json!({}), retries: 0, model: Some("m1".into()), replace: false, owner: None, max_queued: 0 }, 2);
        assert_eq!(pushes(&out).len(), 1);
        h.draining = true;
        h.jobs = vec![Held { job_id: "y".into(), attempt: 1, lease: 1, state: "queued".into() }];
        s.on_msg("w1", WorkerMsg::Hello(h), 3);
        assert_eq!(s.job("y").unwrap().phase, Phase::Running);
        let (_, out) = s.enqueue(EnqueueReq { job_id: "z".into(), envelope: json!({}), retries: 0, model: None, replace: false, owner: None, max_queued: 0 }, 4);
        assert!(pushes(&out).is_empty());
        let out = s.on_msg("w1", WorkerMsg::Status { running: 1, draining: false, capacity: 4, ready: None }, 5);
        assert_eq!(pushes(&out).len(), 1);
    }

    #[test]
    fn finished_jobs_are_forgotten() {
        let mut s = Sched::new("p", Cfg::default());
        s.on_msg("w1", hello("w1", 1, vec![]), 0);
        enq(&mut s, "a", 0);
        ack(&mut s, "w1", "a", 1, 1);
        s.on_msg("w1", WorkerMsg::Done { job_id: "a".into(), attempt: 1, state: "failed".into() }, 2);
        assert!(!s.job("a").unwrap().failed_here);
        let t = s.next_wake(3).unwrap();
        s.tick(t);
        assert!(s.job("a").is_none());
        assert!(s.take_dirty().removed_jobs.contains(&"a".to_string()));
    }

    #[test]
    fn restage_after_424_and_spilled_envelopes() {
        let mut s = Sched::new("p", Cfg::default());
        s.on_msg("w1", hello("w1", 1, vec![]), 0);
        let (_, out) = s.enqueue_spilled(EnqueueReq { job_id: "a".into(), envelope: json!(null), retries: 1, model: None, replace: false, owner: None, max_queued: 0 }, Some("env/a".into()), 0);
        // A spilled envelope is pushed through the host.
        assert!(matches!(&out[..], [Out::PushSpilled { key, .. }] if key == "env/a"));
        let out = s.on_msg("w1", WorkerMsg::Nack { job_id: "a".into(), attempt: 1, retry: true, code: 424, message: "gone".into() }, 1);
        assert!(out.iter().all(|o| !matches!(o, Out::Send { msg: DoMsg::Job { .. }, .. } | Out::PushSpilled { .. })));
        assert_eq!(s.status(2).restage, vec!["a".to_string()]);
        // Not pushed again until the gateway replaces the envelope.
        assert!(pushes(&s.tick(5_000)).is_empty());
        let (r, out) = s.enqueue(EnqueueReq { job_id: "a".into(), envelope: json!({"stored": true}), retries: 1, model: None, replace: true, owner: None, max_queued: 0 }, 6_000);
        assert!(!r.duplicate);
        assert_eq!(pushes(&out), vec![("w1".into(), "a".into(), 1, false)]);
        assert_eq!(s.job("a").unwrap().lease, 2);
        let d = s.take_dirty();
        assert_eq!(d.dropped_spills, vec!["env/a".to_string()]);
        assert!(s.status(7_000).restage.is_empty());
        // A restage that never comes fails after the bounce limit.
        s.enqueue(EnqueueReq { job_id: "b".into(), envelope: json!({}), retries: 1, model: None, replace: false, owner: None, max_queued: 0 }, 7_000);
        s.on_msg("w1", WorkerMsg::Done { job_id: "a".into(), attempt: 1, state: "succeeded".into() }, 7_001);
        s.on_msg("w1", WorkerMsg::Nack { job_id: "b".into(), attempt: 1, retry: true, code: 424, message: "gone".into() }, 7_002);
        let t = s.next_wake(7_003).unwrap();
        s.tick(t.max(7_000 + s.cfg.max_bounce_ms + 2));
        assert_eq!(s.job("b").unwrap().phase, Phase::Failed);
    }

    // ------------------------------------------------------- protocol 2

    fn hello2(id: &str, cap: u32, free: u32, session_free: u32, jobs: Vec<Held>) -> WorkerMsg {
        WorkerMsg::Hello(Hello {
            worker_id: id.into(),
            pool: "family:wan".into(),
            proto: 2,
            capacity: cap,
            jobs,
            endpoint: format!("https://{id}.example"),
            slots: Some(Slots { free, session_free, offers_seen: 0 }),
            ..Hello::default()
        })
    }
    fn slots(s: &mut Sched, w: &str, free: u32, session_free: u32, seen: u64, now: i64) -> Vec<Out> {
        s.on_msg(w, WorkerMsg::Slots(Slots { free, session_free, offers_seen: seen }), now)
    }
    fn ready(out: &[Out]) -> Vec<(String, Result<SessionGrant, String>)> {
        out.iter()
            .filter_map(|o| match o {
                Out::SessionReady { session_id, result } => Some((session_id.clone(), result.clone())),
                _ => None,
            })
            .collect()
    }
    fn session_offers(out: &[Out]) -> Vec<(String, String, u64)> {
        out.iter()
            .filter_map(|o| match o {
                Out::Send { worker, msg: DoMsg::SessionOffer { session_id, lease, .. } } => Some((worker.clone(), session_id.clone(), *lease)),
                _ => None,
            })
            .collect()
    }
    fn revokes(out: &[Out]) -> Vec<(String, String)> {
        out.iter()
            .filter_map(|o| match o {
                Out::Send { worker, msg: DoMsg::SessionRevoke { session_id, .. } } => Some((worker.clone(), session_id.clone())),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn credits_bound_offers_in_flight() {
        let mut s = Sched::new("family:wan", Cfg::default());
        s.on_msg("w1", hello2("w1", 2, 1, 0, vec![]), 0);
        let (_, out) = enq(&mut s, "a", 1);
        assert_eq!(pushes(&out).len(), 1);
        // One credit, one offer in flight: nothing more although the
        // capacity is 2.
        let (_, out) = enq(&mut s, "b", 2);
        assert!(pushes(&out).is_empty());
        // A stale report (computed before the offer arrived) changes nothing.
        let out = slots(&mut s, "w1", 1, 0, 0, 3);
        assert!(pushes(&out).is_empty());
        // The worker took `a` and still has one slot (`seen` = 1): `b` goes.
        ack(&mut s, "w1", "a", 1, 4);
        let out = slots(&mut s, "w1", 1, 0, 1, 5);
        assert_eq!(pushes(&out), vec![("w1".into(), "b".into(), 1, false)]);
        assert_eq!(s.metrics(6).slots_free, 0);
    }

    #[test]
    fn arbiter_nack_reoffers_elsewhere_at_once() {
        // (No stale-worker detection in this test: it spans a long wait.)
        let cfg = Cfg { stale_after_ms: i64::MAX / 4, ..Cfg::default() };
        let mut s = Sched::new("family:wan", cfg);
        s.on_msg("w1", hello2("w1", 1, 1, 0, vec![]), 0);
        s.on_msg("w2", hello2("w2", 1, 1, 0, vec![]), 0);
        let (_, out) = enq(&mut s, "a", 0);
        assert_eq!(pushes(&out)[0].0, "w1");
        // w1's GPU was taken by another family's job meanwhile.
        let nack = WorkerMsg::Nack { job_id: "a".into(), attempt: 1, retry: true, code: 429, message: "busy with family ltx".into() };
        let out = s.on_msg("w1", nack.clone(), 5);
        assert_eq!(pushes(&out), vec![("w2".into(), "a".into(), 1, false)], "no backoff");
        assert_eq!(s.job("a").unwrap().lease, 2);
        // w2 is busy too: the job waits (not failed) until a slot frees,
        // however long that takes.
        let nack2 = WorkerMsg::Nack { job_id: "a".into(), attempt: 1, retry: true, code: 429, message: "busy".into() };
        let out = s.on_msg("w2", nack2, 6);
        assert!(pushes(&out).is_empty());
        let late = 6 + s.cfg.max_bounce_ms * 3;
        assert!(pushes(&s.tick(late)).is_empty());
        assert_eq!(s.job("a").unwrap().phase, Phase::Queued);
        let out = slots(&mut s, "w2", 1, 0, 1, late + 1);
        assert_eq!(pushes(&out), vec![("w2".into(), "a".into(), 1, false)]);
        // A late 429 from w2 again: still not failed (no bounce limit).
        let out = s.on_msg("w2", WorkerMsg::Nack { job_id: "a".into(), attempt: 1, retry: true, code: 429, message: "busy".into() }, late + 2);
        assert!(pushes(&out).is_empty());
        assert_eq!(s.job("a").unwrap().phase, Phase::Queued);
    }

    #[test]
    fn ack_timeout_requeues_and_fences_the_late_ack_v2() {
        let mut s = Sched::new("family:wan", Cfg::default());
        s.on_msg("w1", hello2("w1", 1, 1, 0, vec![]), 0);
        s.on_msg("w2", hello2("w2", 1, 1, 0, vec![]), 0);
        enq(&mut s, "a", 0);
        let out = s.tick(s.cfg.ack_timeout_ms);
        assert_eq!(pushes(&out), vec![("w2".into(), "a".into(), 1, false)]);
        let out = s.on_msg("w1", WorkerMsg::Ack { job_id: "a".into(), attempt: 1, lease: 1, worker_ms: 1 }, s.cfg.ack_timeout_ms + 1);
        assert_eq!(cancels(&out), vec![("w1".into(), "a".into())]);
    }

    #[test]
    fn redeploy_reannounce_v2_runs_nothing_twice() {
        let mut s = Sched::new("family:wan", Cfg::default());
        s.on_msg("w1", hello2("w1", 2, 2, 0, vec![]), 0);
        enq(&mut s, "run", 0);
        enq(&mut s, "inflight", 0);
        ack(&mut s, "w1", "run", 1, 1);
        let lease_run = s.job("run").unwrap().lease;
        let jobs: Vec<JobRec> = s.jobs().cloned().collect();
        let workers: Vec<WorkerRec> = s.workers().cloned().collect();
        let mut s = Sched::restore_all("family:wan", Cfg::default(), jobs, workers, Vec::new(), Vec::new());
        s.disconnect("w1", 10);
        let held = vec![Held { job_id: "run".into(), attempt: 1, lease: lease_run, state: "running".into() }];
        // Back with one slot left (it runs `run`).
        let out = s.on_msg("w1", hello2("w1", 2, 1, 0, held), 500);
        assert_eq!(pushes(&out), vec![("w1".into(), "inflight".into(), 1, false)], "only the offer lost in flight goes again");
        assert_eq!(s.job("run").unwrap().attempt, 1);
        assert!(cancels(&out).is_empty());
    }

    #[test]
    fn session_admission_ack_renew_expire() {
        let mut s = Sched::new("family:sfwan", Cfg::default());
        s.on_msg("w1", hello2("w1", 1, 1, 1, vec![]), 0);
        let (a, out) = s.admit(SessionReq { session_id: Some("s1".into()), kind: "director".into(), ..SessionReq::default() }, 10);
        assert_eq!(a, Admit::Pending("s1".into()));
        assert_eq!(session_offers(&out), vec![("w1".into(), "s1".into(), 1)]);
        let out = s.on_msg("w1", WorkerMsg::SessionAck { session_id: "s1".into(), lease: 1, endpoint: String::new() }, 20);
        let r = ready(&out);
        let g = r[0].1.clone().unwrap();
        assert_eq!((g.worker_id.as_str(), g.endpoint.as_str(), g.expires_ms), ("w1", "https://w1.example", 20 + s.cfg.session_ttl_ms));
        // Idempotent admit of the same id.
        assert!(matches!(s.admit(SessionReq { session_id: Some("s1".into()), ..SessionReq::default() }, 21).0, Admit::Granted(_)));
        assert_eq!(s.metrics(22).sessions_live, 1);
        let g = s.renew("s1", 1, 1_000).unwrap();
        s.on_msg("w1", WorkerMsg::Status { running: 0, draining: false, capacity: 1, ready: None }, 50_000);
        assert_eq!(g.expires_ms, 1_000 + s.cfg.session_ttl_ms);
        assert!(s.renew("s1", 7, 1_001).is_none(), "wrong lease");
        let out = s.tick(g.expires_ms);
        assert_eq!(revokes(&out), vec![("w1".into(), "s1".into())]);
        assert_eq!(s.session("s1").unwrap().state, SessionState::Ended);
        assert!(s.renew("s1", 0, g.expires_ms + 1).is_none());
    }

    #[test]
    fn session_nack_timeout_and_no_capacity() {
        let mut s = Sched::new("family:sfwan", Cfg::default());
        // A protocol-1 worker never gets sessions.
        s.on_msg("v1", hello("v1", 1, vec![]), 0);
        let (a, _) = s.admit(SessionReq::default(), 1);
        assert!(matches!(a, Admit::Refused(_)), "{a:?}");
        s.on_msg("w1", hello2("w1", 1, 1, 1, vec![]), 2);
        s.on_msg("w2", hello2("w2", 1, 1, 1, vec![]), 2);
        let (a, out) = s.admit(SessionReq { session_id: Some("s".into()), ..SessionReq::default() }, 3);
        assert_eq!(a, Admit::Pending("s".into()));
        assert_eq!(session_offers(&out)[0].0, "w1");
        let out = s.on_msg("w1", WorkerMsg::SessionNack { session_id: "s".into(), lease: 1, code: 429, message: "busy".into() }, 4);
        assert_eq!(session_offers(&out), vec![("w2".into(), "s".into(), 2)]);
        // w2 never answers: the offer times out and nobody is left.
        let out = s.tick(4 + s.cfg.session_offer_timeout_ms);
        let r = ready(&out);
        assert_eq!(r.len(), 1);
        assert!(r[0].1.as_ref().unwrap_err().contains("no other worker"), "{r:?}");
        // A late ack from w2 is revoked.
        let out = s.on_msg("w2", WorkerMsg::SessionAck { session_id: "s".into(), lease: 2, endpoint: String::new() }, 9_000);
        assert_eq!(revokes(&out), vec![("w2".into(), "s".into())]);
    }

    #[test]
    fn sessions_survive_a_reconnect_and_release_frees_the_slot() {
        let mut s = Sched::new("family:sfwan", Cfg::default());
        s.on_msg("w1", hello2("w1", 1, 1, 1, vec![]), 0);
        let (_, _) = s.admit(SessionReq { session_id: Some("a".into()), ..SessionReq::default() }, 1);
        s.on_msg("w1", WorkerMsg::SessionAck { session_id: "a".into(), lease: 1, endpoint: "https://pod-a".into() }, 2);
        s.disconnect("w1", 3);
        // Back holding `a` plus a stale session it should drop.
        let h = WorkerMsg::Hello(Hello {
            worker_id: "w1".into(),
            pool: "family:sfwan".into(),
            proto: 2,
            sessions: vec![crate::HeldSession { session_id: "a".into(), lease: 1 }, crate::HeldSession { session_id: "old".into(), lease: 4 }],
            slots: Some(Slots { free: 0, session_free: 0, offers_seen: 0 }),
            ..Hello::default()
        });
        let out = s.on_msg("w1", h, 500);
        let w = out.iter().find_map(|o| match o {
            Out::Send { msg: DoMsg::Welcome { end_sessions, .. }, .. } => Some(end_sessions.clone()),
            _ => None,
        });
        assert_eq!(w, Some(vec!["old".to_string()]));
        assert_eq!(s.session("a").unwrap().state, SessionState::Live);
        let (st, out) = s.release("a", 600);
        assert_eq!(st, Some("ended"));
        assert_eq!(revokes(&out), vec![("w1".into(), "a".into())]);
        // A worker that is gone past the grace period loses its sessions.
        s.on_msg("w1", WorkerMsg::Slots(Slots { free: 1, session_free: 1, offers_seen: 0 }), 700);
        s.admit(SessionReq { session_id: Some("b".into()), ..SessionReq::default() }, 701);
        s.on_msg("w1", WorkerMsg::SessionAck { session_id: "b".into(), lease: 1, endpoint: String::new() }, 702);
        s.disconnect("w1", 800);
        s.tick(800 + s.cfg.reconnect_grace_ms);
        assert_eq!(s.session("b").unwrap().state, SessionState::Ended);
    }

    fn creates(out: &[Out]) -> Vec<(String, u64, String)> {
        out.iter()
            .filter_map(|o| match o {
                Out::CreateUpload { worker, req, key, .. } => Some((worker.clone(), *req, key.clone())),
                _ => None,
            })
            .collect()
    }
    fn grants(out: &[Out]) -> Vec<(String, u64, String, u16, u16)> {
        out.iter()
            .filter_map(|o| match o {
                Out::Grant { worker, req, upload_id, from, count, .. } => Some((worker.clone(), *req, upload_id.clone(), *from, *count)),
                _ => None,
            })
            .collect()
    }
    fn grant_errors(out: &[Out]) -> Vec<String> {
        out.iter()
            .filter_map(|o| match o {
                Out::Send { msg: DoMsg::UploadGrant { error: Some(e), .. }, .. } => Some(e.clone()),
                _ => None,
            })
            .collect()
    }
    fn aborts(out: &[Out]) -> Vec<String> {
        out.iter()
            .filter_map(|o| match o {
                Out::AbortUpload { upload_id, .. } => Some(upload_id.clone()),
                _ => None,
            })
            .collect()
    }
    fn init(job: &str, lease: u64, req: u64) -> WorkerMsg {
        WorkerMsg::UploadInit { req, job_id: job.into(), attempt: 1, lease, name: "output.mp4".into(), content_type: "video/mp4".into(), parts: 4 }
    }

    #[test]
    fn upload_grant_complete_and_result() {
        let mut s = Sched::new("family:wan", Cfg::default());
        s.on_msg("w1", hello2("w1", 1, 1, 0, vec![]), 0);
        enq(&mut s, "a", 0);
        ack(&mut s, "w1", "a", 1, 1);
        let out = s.on_msg("w1", init("a", 1, 7), 2);
        let c = creates(&out);
        assert_eq!(c, vec![("w1".into(), 7, "outputs/wan/a/1-1/output.mp4".into())]);
        let key = c[0].2.clone();
        let out = s.upload_created(&key, 7, 4, Ok("U1".into()), 3);
        assert_eq!(grants(&out), vec![("w1".into(), 7, "U1".into(), 1, 4)]);
        let out = s.on_msg("w1", WorkerMsg::UploadMore { req: 8, job_id: "a".into(), upload_id: "U1".into(), from: 5, count: 4 }, 4);
        assert_eq!(grants(&out), vec![("w1".into(), 8, "U1".into(), 5, 4)]);
        let parts = vec![Part { n: 1, etag: "e1".into() }, Part { n: 2, etag: "e2".into() }];
        let done = WorkerMsg::UploadDone { req: 9, job_id: "a".into(), attempt: 1, lease: 1, upload_id: "U1".into(), parts: parts.clone(), bytes: 100, sha256: "ab".into() };
        let out = s.on_msg("w1", done.clone(), 5);
        assert!(matches!(&out[..], [Out::CompleteUpload { req: 9, bytes: 100, .. }]), "{out:?}");
        // A size mismatch would fail; the right size commits.
        let out = s.upload_completed(&key, 9, Ok(100), 6);
        assert!(out.iter().any(|o| matches!(o, Out::Send { msg: DoMsg::UploadCommitted { ok: true, bytes: 100, .. }, .. })));
        assert_eq!(s.job("a").unwrap().result.as_ref().unwrap().sha256, "ab");
        // A repeated done is answered from the record (idempotent).
        let out = s.on_msg("w1", done, 7);
        assert!(out.iter().any(|o| matches!(o, Out::Send { msg: DoMsg::UploadCommitted { ok: true, .. }, .. })));
        // The job ends: nothing is aborted.
        let out = s.on_msg("w1", WorkerMsg::Done { job_id: "a".into(), attempt: 1, state: "succeeded".into() }, 8);
        assert!(aborts(&out).is_empty());
    }

    #[test]
    fn uploads_are_fenced_and_aborted_when_the_job_moves() {
        let mut s = Sched::new("family:wan", Cfg::default());
        s.on_msg("w1", hello2("w1", 1, 1, 0, vec![]), 0);
        s.on_msg("w2", hello2("w2", 1, 1, 0, vec![]), 0);
        enq(&mut s, "a", 0);
        ack(&mut s, "w1", "a", 1, 1);
        // Not the holder, or a stale lease: refused.
        assert_eq!(grant_errors(&s.on_msg("w2", init("a", 1, 1), 2)).len(), 1);
        assert_eq!(grant_errors(&s.on_msg("w1", init("a", 9, 2), 2)).len(), 1);
        let out = s.on_msg("w1", init("a", 1, 3), 3);
        let key = creates(&out)[0].2.clone();
        s.upload_created(&key, 3, 2, Ok("U".into()), 4);
        // w1 is lost; the job goes to w2 under lease 2.
        s.disconnect("w1", 10);
        let out = s.tick(10 + s.cfg.reconnect_grace_ms);
        assert_eq!(pushes(&out), vec![("w2".into(), "a".into(), 2, true)]);
        assert_eq!(aborts(&out), vec!["U".to_string()], "the old holder's upload is aborted");
        // The old holder's done is fenced.
        let done = WorkerMsg::UploadDone { req: 5, job_id: "a".into(), attempt: 1, lease: 1, upload_id: "U".into(), parts: vec![], bytes: 1, sha256: String::new() };
        let out = s.on_msg("w1", done, 20_100);
        assert!(out.iter().any(|o| matches!(o, Out::Send { msg: DoMsg::UploadCommitted { ok: false, .. }, .. })), "{out:?}");
        // An upload created after its job moved on is aborted at once.
        ack(&mut s, "w2", "a", 2, 20_200);
        let out = s.on_msg("w2", init("a", 2, 6), 20_300);
        let k2 = creates(&out)[0].2.clone();
        s.cancel("a", 20_400);
        s.on_msg("w2", WorkerMsg::Done { job_id: "a".into(), attempt: 2, state: "cancelled".into() }, 20_500);
        let out = s.upload_created(&k2, 6, 2, Ok("U2".into()), 20_600);
        assert_eq!(aborts(&out), vec!["U2".to_string()]);
        // An open upload that outlives its TTL is aborted by the alarm.
        slots(&mut s, "w2", 1, 0, 1, 29_999);
        enq(&mut s, "b", 30_000);
        let who = s.job("b").unwrap().worker.clone().unwrap();
        ack(&mut s, &who, "b", 1, 30_001);
        let out = s.on_msg(&who, init("b", 1, 7), 30_002);
        let k3 = creates(&out)[0].2.clone();
        s.upload_created(&k3, 7, 1, Ok("U3".into()), 30_003);
        let wake = s.next_wake(30_004).unwrap();
        assert!(wake <= 30_003 + s.cfg.upload_ttl_ms);
        let out = s.tick(30_003 + s.cfg.upload_ttl_ms);
        assert_eq!(aborts(&out), vec!["U3".to_string()]);
    }

    #[test]
    fn metrics_report_depth_age_and_slots() {
        let mut s = Sched::new("family:wan", Cfg::default());
        s.on_msg("w1", hello2("w1", 2, 1, 1, vec![]), 0);
        enq(&mut s, "a", 100);
        enq(&mut s, "b", 200);
        enq(&mut s, "c", 300);
        let m = s.metrics(1_300);
        assert_eq!(m.family, "wan");
        assert_eq!((m.queued, m.pushed, m.workers, m.slots_total, m.slots_free, m.session_capacity), (2, 1, 1, 2, 0, 1));
        assert_eq!(m.oldest_queued_ms, 1_100);
    }

    /// Edge control plane: per-model admission, owners in flight, the
    /// envelope update of a front's background input copy, job phases.
    #[test]
    fn admission_owners_and_envelope_updates() {
        let mut s = Sched::new("family:h3", Cfg::default());
        let req = |id: &str, owner: &str, model: &str| EnqueueReq {
            job_id: id.into(),
            envelope: json!({"v": 1}),
            retries: 1,
            model: Some(model.into()),
            replace: false,
            owner: Some(owner.into()),
            max_queued: 2,
        };
        assert_eq!(s.enqueue(req("a", "key_1", "m"), 1).0.state, "queued");
        assert_eq!(s.enqueue(req("b", "key_1", "m"), 2).0.state, "queued");
        // A third of the same model is refused; another model is not.
        assert_eq!(s.enqueue(req("c", "key_2", "m"), 3).0.state, "refused");
        assert!(s.job("c").is_none());
        assert_eq!(s.enqueue(req("d", "key_2", "other"), 4).0.state, "queued");
        let st = s.status(5);
        assert_eq!(st.owners.get("key_1"), Some(&2));
        assert_eq!(st.owners.get("key_2"), Some(&1));
        // `replace` on a queued job keeps the new envelope, no new job.
        let (r, out) = s.enqueue(EnqueueReq { envelope: json!({"v": 2}), replace: true, ..req("a", "key_1", "m") }, 6);
        assert!(r.duplicate && out.is_empty());
        assert_eq!(s.job("a").unwrap().envelope, Some(json!({"v": 2})));
        assert_eq!(s.job_phase("a"), "queued");
        assert_eq!(s.job_phase("zz"), "unknown");
        s.cancel("a", 7);
        assert_eq!(s.job_phase("a"), "cancelled");
        assert_eq!(s.status(8).owners.get("key_1"), Some(&1));
    }

    #[test]
    fn fronts_and_readiness_ride_on_the_worker() {
        let mut s = Sched::new("family:h3", Cfg::default());
        let f = crate::FrontInfo { url: "https://w1".into(), names: [("fasth3".to_owned(), "fasth3".to_owned())].into(), ready: false, ..Default::default() };
        s.on_msg("w1", WorkerMsg::Hello(Hello { worker_id: "w1".into(), pool: "family:h3".into(), front: Some(f), ..Hello::default() }), 1);
        let w = &s.status(2).workers[0];
        assert!(!w.ready && w.front.as_ref().unwrap().names.contains_key("fasth3"));
        s.on_msg("w1", WorkerMsg::Status { running: 0, draining: false, capacity: 1, ready: Some(true) }, 3);
        assert!(s.status(4).workers[0].ready);
    }
}
