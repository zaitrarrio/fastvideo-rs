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

use std::collections::{BTreeMap, BTreeSet, VecDeque};

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::{DispatchTimings, DoMsg, EnqueueReq, EnqueueResp, FailedJob, Hello, PoolStatus, WorkerInfo, WorkerMsg};

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
}

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
    #[serde(skip)]
    pub envelope: Option<Value>,
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
        }
    }

    /// Rebuilds from persisted rows (envelopes already set on the jobs).
    pub fn restore(pool: impl Into<String>, cfg: Cfg, jobs: Vec<JobRec>, workers: Vec<WorkerRec>) -> Self {
        let mut s = Self::new(pool, cfg);
        for j in jobs {
            s.seq = s.seq.max(j.seq);
            s.jobs.insert(j.job_id.clone(), j);
        }
        for w in workers {
            s.workers.insert(w.worker_id.clone(), w);
        }
        s
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
        Dirty {
            jobs,
            workers,
            removed_jobs: std::mem::take(&mut self.removed_jobs).into_iter().collect(),
            removed_workers: std::mem::take(&mut self.removed_workers).into_iter().collect(),
            new_envelopes,
            dropped_envelopes: std::mem::take(&mut self.dropped_envelopes).into_iter().collect(),
            dropped_spills: std::mem::take(&mut self.dropped_spills).into_iter().collect(),
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
        if let Some(j) = self.jobs.get(&req.job_id) {
            let resp = EnqueueResp { job_id: j.job_id.clone(), state: j.phase.as_str().into(), worker: j.worker.clone(), position: self.position(j), duplicate: true };
            return (resp, Vec::new());
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
            WorkerMsg::Status { running: _, draining, capacity } => {
                if let Some(w) = self.workers.get_mut(worker) {
                    if w.draining != draining || w.capacity != capacity.max(1) {
                        w.draining = draining;
                        w.capacity = capacity.max(1);
                        self.dirty_workers.insert(worker.to_owned());
                    }
                }
            }
        }
        out.extend(self.pump(now));
        out
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
        out.push(Out::Send { worker: worker.to_owned(), msg: DoMsg::Welcome { worker_id: worker.to_owned(), pool: self.pool.clone(), cancel } });
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
        // Queued jobs waiting for a worker are pushed on the next event
        // (a hello, an ack, a done); no timer needed for them.
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
                .map(|w| (self.held(&w.worker_id), w.capacity, w.worker_id.clone()))
                .filter(|(held, cap, _)| held < cap)
                .min_by_key(|(held, _, id)| (*held, id.clone()))
                .map(|(_, _, id)| id);
            let Some(w) = pick else { continue };
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
            })
            .collect();
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
        }
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
        s.enqueue(EnqueueReq { job_id: id.into(), envelope: json!({"job": {"id": id}}), retries: 1, model: None, replace: false }, now)
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
        let (_, out) = s.enqueue(EnqueueReq { job_id: "x".into(), envelope: json!({}), retries: 0, model: Some("m2".into()), replace: false }, 1);
        assert!(pushes(&out).is_empty());
        let (_, out) = s.enqueue(EnqueueReq { job_id: "y".into(), envelope: json!({}), retries: 0, model: Some("m1".into()), replace: false }, 2);
        assert_eq!(pushes(&out).len(), 1);
        h.draining = true;
        h.jobs = vec![Held { job_id: "y".into(), attempt: 1, lease: 1, state: "queued".into() }];
        s.on_msg("w1", WorkerMsg::Hello(h), 3);
        assert_eq!(s.job("y").unwrap().phase, Phase::Running);
        let (_, out) = s.enqueue(EnqueueReq { job_id: "z".into(), envelope: json!({}), retries: 0, model: None, replace: false }, 4);
        assert!(pushes(&out).is_empty());
        let out = s.on_msg("w1", WorkerMsg::Status { running: 1, draining: false, capacity: 4 }, 5);
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
        let (_, out) = s.enqueue_spilled(EnqueueReq { job_id: "a".into(), envelope: json!(null), retries: 1, model: None, replace: false }, Some("env/a".into()), 0);
        // A spilled envelope is pushed through the host.
        assert!(matches!(&out[..], [Out::PushSpilled { key, .. }] if key == "env/a"));
        let out = s.on_msg("w1", WorkerMsg::Nack { job_id: "a".into(), attempt: 1, retry: true, code: 424, message: "gone".into() }, 1);
        assert!(out.iter().all(|o| !matches!(o, Out::Send { msg: DoMsg::Job { .. }, .. } | Out::PushSpilled { .. })));
        assert_eq!(s.status(2).restage, vec!["a".to_string()]);
        // Not pushed again until the gateway replaces the envelope.
        assert!(pushes(&s.tick(5_000)).is_empty());
        let (r, out) = s.enqueue(EnqueueReq { job_id: "a".into(), envelope: json!({"stored": true}), retries: 1, model: None, replace: true }, 6_000);
        assert!(!r.duplicate);
        assert_eq!(pushes(&out), vec![("w1".into(), "a".into(), 1, false)]);
        assert_eq!(s.job("a").unwrap().lease, 2);
        let d = s.take_dirty();
        assert_eq!(d.dropped_spills, vec!["env/a".to_string()]);
        assert!(s.status(7_000).restage.is_empty());
        // A restage that never comes fails after the bounce limit.
        s.enqueue(EnqueueReq { job_id: "b".into(), envelope: json!({}), retries: 1, model: None, replace: false }, 7_000);
        s.on_msg("w1", WorkerMsg::Done { job_id: "a".into(), attempt: 1, state: "succeeded".into() }, 7_001);
        s.on_msg("w1", WorkerMsg::Nack { job_id: "b".into(), attempt: 1, retry: true, code: 424, message: "gone".into() }, 7_002);
        let t = s.next_wake(7_003).unwrap();
        s.tick(t.max(7_000 + s.cfg.max_bounce_ms + 2));
        assert_eq!(s.job("b").unwrap().phase, Phase::Failed);
    }
}
