//! `EngineService`: start, caps, readiness, submit, cancel, sessions, drain
//! (design §3.6, §6.3).
//!
//! The async API every protocol adapter uses. Work runs on one executor
//! thread per backend ([`crate::executor`]); this side only books jobs into
//! the [`Scheduler`] and hands out event streams.
//!
//! Job lifecycle as seen on a [`JobHandle`]'s event stream:
//!
//! ```text
//! Queued{position}+  ->  Started  ->  (Stage | Progress | Log)*  ->  Finished | Failed | Cancelled
//!        └──────────────── cancel while queued ────────────────────────────────────┘ Cancelled
//! ```
//!
//! `Queued` is re-sent whenever the position changes. Exactly one terminal
//! event is sent, and nothing after it.

use std::collections::{BTreeMap, HashMap};
use std::path::PathBuf;
use std::sync::{Arc, Condvar, Mutex, Weak};
use std::time::{Duration, Instant};

use fastvideo_protocol::{
    ApiError, JobId, LogLine, ModelCaps, ModelId, ResolvedJob, SessionSpec, StreamCaps,
};
use tokio::sync::{mpsc, watch, Notify};

use crate::backend::{ClipOutput, EngineBackend, SessionId};
use crate::cancel::{lock, CancelToken, OutputMode};
use crate::caps::{CapabilityTable, Recipe};
use crate::executor;
use crate::pool::{ModelPool, Readiness, Residency};
use crate::scheduler::{busy, ExecState, Priority, QueueItem, Scheduler};
use crate::stream::causal::{CausalSession, CausalShared};
use crate::stream::clip::ClipSession;

/// Engine configuration (the `[engine]` part of the serve config).
#[derive(Clone, Debug)]
pub struct EngineConfig {
    /// Maximum queued batch jobs (`[limits] queue_max`); more -> `QueueFull`.
    pub queue_max: usize,
    /// Load non-resident models on demand, evicting others (risk R18).
    pub swap: bool,
    /// Tier alias (`h3-max`, `ltx-turbo`, …) -> model id, overriding the
    /// backends' own recipe tiers.
    pub tier_overrides: BTreeMap<String, ModelId>,
    /// Batch MP4s go to `<output_dir>/<job id>/`.
    pub output_dir: PathBuf,
    /// Executor -> causal consumer channel depth, in blocks. 2 (design §5.4
    /// said 4): the causal pacer applies backpressure, so a generator faster
    /// than playout waits here, and every queued block delays a prompt switch.
    pub causal_depth: usize,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            queue_max: 32,
            swap: false,
            tier_overrides: BTreeMap::new(),
            output_dir: std::env::temp_dir().join("fv-engine"),
            causal_depth: 2,
        }
    }
}

/// Engine-side job events (design §3.6).
#[derive(Clone, Debug, PartialEq)]
pub enum EngineEvent {
    /// 0-based position; re-sent when it changes.
    Queued { position: u32 },
    Started,
    Stage { name: &'static str },
    Progress { step: u32, total: u32 },
    Log(LogLine),
    Finished(ClipOutput),
    Failed(ApiError),
    Cancelled,
}

impl EngineEvent {
    pub fn is_terminal(&self) -> bool {
        matches!(
            self,
            EngineEvent::Finished(_) | EngineEvent::Failed(_) | EngineEvent::Cancelled
        )
    }
}

/// A submitted job.
///
/// **Deviation from design §3.6:** `events` is unbounded, so the executor
/// thread never blocks on a slow consumer; progress events are small and a
/// job emits a bounded number of them. `id` is added for convenience.
#[derive(Debug)]
pub struct JobHandle {
    pub id: JobId,
    pub events: mpsc::UnboundedReceiver<EngineEvent>,
    pub cancel: CancelToken,
}

impl JobHandle {
    /// Waits for the terminal event: `Ok` on `Finished`, the error on
    /// `Failed`, `Cancelled` on cancel (or if the engine went away).
    pub async fn wait(mut self) -> Result<ClipOutput, ApiError> {
        while let Some(ev) = self.events.recv().await {
            match ev {
                EngineEvent::Finished(out) => return Ok(out),
                EngineEvent::Failed(e) => return Err(e),
                EngineEvent::Cancelled => return Err(crate::cancel::cancelled_error()),
                _ => {}
            }
        }
        Err(ApiError::internal("the engine dropped the job"))
    }
}

/// What `EngineService::cancel` did.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CancelOutcome {
    /// It was queued and is now cancelled (event sent).
    Dequeued,
    /// It is running; it stops at its next step.
    Requested,
    /// Not known to the engine (never submitted, or already finished).
    Unknown,
}

/// Counters for health / metrics routes.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct EngineStats {
    pub queued_batch: usize,
    pub queued_stream: usize,
    pub running: usize,
    pub sessions: usize,
    pub draining: bool,
    /// Jobs it can run at once (one per executor).
    pub executors: usize,
}

pub(crate) struct JobEntry {
    pub tx: mpsc::UnboundedSender<EngineEvent>,
    pub cancel: CancelToken,
    pub job: Arc<ResolvedJob>,
    pub mode: OutputMode,
    pub last_pos: Option<u32>,
    pub running: bool,
}

pub(crate) struct State {
    pub sched: Scheduler,
    pub pool: ModelPool,
    pub jobs: HashMap<JobId, JobEntry>,
    pub causal: HashMap<SessionId, Arc<CausalShared>>,
    /// Causal sessions closed by their owner, for the executor to release
    /// on the backend (`causal_close`).
    pub to_close: Vec<Vec<SessionId>>,
    pub draining: bool,
    pub shutdown: bool,
    pub alive: usize,
    /// Per executor: the cancel token of the background warm-up run on its
    /// GPU now (fast boot B). Any arriving work trips it.
    pub warmup_cancel: Vec<Option<CancelToken>>,
}

impl State {
    /// A job or session arrived: every background warm-up run yields (it is
    /// cancelled at its next step and retried when the executor is idle).
    /// Warm-up tokens carry no callbacks, so tripping them under the lock is safe.
    pub fn yield_warmups(&mut self) {
        for t in self.warmup_cancel.iter().flatten() {
            t.cancel();
        }
    }

    /// Sends `Queued{position}` to every queued job whose position changed.
    pub fn publish_positions(&mut self) {
        for (job, pos) in self.sched.positions() {
            if let Some(e) = self.jobs.get_mut(&job) {
                if e.last_pos != Some(pos) {
                    e.last_pos = Some(pos);
                    let _ = e.tx.send(EngineEvent::Queued { position: pos });
                }
            }
        }
    }
}

pub(crate) struct Shared {
    pub state: Mutex<State>,
    /// Executors park here waiting for work.
    pub work_cv: Condvar,
    /// Async waiters (drain, readiness) re-check state on every change.
    pub changed: Notify,
    pub readiness: watch::Sender<Readiness>,
    pub caps: CapabilityTable,
    pub cfg: EngineConfig,
}

impl Shared {
    pub fn lock(&self) -> std::sync::MutexGuard<'_, State> {
        lock(&self.state)
    }

    /// Wakes executors and async waiters, and republishes readiness.
    pub fn changed(&self, st: &State) {
        self.readiness.send_if_modified(|r| {
            let now = st.pool.readiness();
            if *r != now {
                *r = now;
                true
            } else {
                false
            }
        });
        self.work_cv.notify_all();
        self.changed.notify_waiters();
    }

    /// Admission for a model: resident (or loadable in swap mode).
    fn admit_model(&self, st: &State, model: &ModelId) -> Result<(), ApiError> {
        if self.caps.get(model).is_none() {
            return Err(ApiError::invalid_param(
                "model",
                format!("model `{model}` is not served here ({})", self.caps.served_summary()),
            ));
        }
        match st.pool.model_state(model) {
            Some(Residency::Resident) => Ok(()),
            Some(Residency::Pending | Residency::Loading { .. }) => Err(ApiError::loading(
                format!("model `{model}` is still loading"),
            )
            .with_retry_after(1)),
            Some(Residency::Failed(e)) => Err(ApiError::engine_failed(format!(
                "model `{model}` failed to load: {}",
                e.message
            ))),
            Some(Residency::Unloaded) if self.cfg.swap => Ok(()),
            Some(Residency::Unloaded) | None => Err(ApiError::invalid_param(
                "model",
                format!("model `{model}` is not resident on this server"),
            )),
        }
    }

    pub fn submit(
        self: &Arc<Self>,
        id: JobId,
        job: ResolvedJob,
        prio: Priority,
        pin: Option<usize>,
        mode: Option<OutputMode>,
    ) -> Result<JobHandle, ApiError> {
        let cancel = CancelToken::new();
        let (tx, rx) = mpsc::unbounded_channel();
        {
            let mut st = self.lock();
            if st.draining {
                return Err(draining());
            }
            self.admit_model(&st, &job.model)?;
            if st.jobs.contains_key(&id) {
                return Err(ApiError::conflict(format!("job {id} was already submitted")));
            }
            st.sched.enqueue(QueueItem {
                job: id,
                prio,
                model: job.model.clone(),
                pin,
            })?;
            let mode = mode.unwrap_or_else(|| OutputMode::File {
                dir: self.cfg.output_dir.join(id.to_string()),
            });
            st.jobs.insert(
                id,
                JobEntry {
                    tx,
                    cancel: cancel.clone(),
                    job: Arc::new(job),
                    mode,
                    last_pos: None,
                    running: false,
                },
            );
            st.publish_positions();
            st.yield_warmups();
            self.changed(&st);
        }
        let weak: Weak<Shared> = Arc::downgrade(self);
        cancel.on_cancel(move || {
            if let Some(sh) = weak.upgrade() {
                sh.dequeue_cancelled(id);
            }
        });
        Ok(JobHandle {
            id,
            events: rx,
            cancel,
        })
    }

    /// On-cancel callback: a queued job leaves the queue at once.
    fn dequeue_cancelled(&self, id: JobId) {
        let mut st = self.lock();
        if st.sched.remove(id).is_some() {
            if let Some(e) = st.jobs.remove(&id) {
                let _ = e.tx.send(EngineEvent::Cancelled);
            }
            st.publish_positions();
            self.changed(&st);
        }
    }

    /// Resolves once `cond` holds or `timeout` passes; returns whether it held.
    pub async fn wait_until(&self, timeout: Duration, cond: impl Fn(&State) -> bool) -> bool {
        let deadline = Instant::now() + timeout;
        loop {
            let n = self.changed.notified();
            if cond(&self.lock()) {
                return true;
            }
            let left = deadline.saturating_duration_since(Instant::now());
            if left.is_zero() {
                return false;
            }
            let _ = tokio::time::timeout(left.min(Duration::from_millis(50)), n).await;
        }
    }

    /// Picks an executor for a new session on `model` and occupies it.
    /// A causal session's shared state is registered under the same lock
    /// that grants the lease, so the executor never sees a lease without it.
    pub fn open_session(
        &self,
        spec: &SessionSpec,
        sid: SessionId,
        causal: Option<Arc<CausalShared>>,
    ) -> Result<(usize, ModelCaps), ApiError> {
        self.open_session_of(spec, sid, causal, false)
    }

    /// [`Self::open_session`] for any session kind: `duplex` for a duplex
    /// model (client input tracks, design §5.11).
    pub fn open_session_of(
        &self,
        spec: &SessionSpec,
        sid: SessionId,
        causal: Option<Arc<CausalShared>>,
        duplex: bool,
    ) -> Result<(usize, ModelCaps), ApiError> {
        let is_causal = causal.is_some();
        let caps = self
            .caps
            .get(&spec.model)
            .cloned()
            .ok_or_else(|| {
                ApiError::invalid_param(
                    "model",
                    format!("model `{}` is not served here ({})", spec.model, self.caps.served_summary()),
                )
            })?;
        let ok = match &caps.stream {
            Some(StreamCaps::Causal { .. }) => is_causal && !duplex,
            Some(StreamCaps::Clip { .. }) => !is_causal && !duplex,
            Some(StreamCaps::Duplex(_)) => duplex,
            None => false,
        };
        if !ok {
            let kind = if duplex {
                "duplex"
            } else if is_causal {
                "causal"
            } else {
                "clip"
            };
            return Err(ApiError::invalid_param(
                "model",
                format!("model `{}` does not support {kind} streaming", spec.model),
            ));
        }
        if spec.tracks.has_audio() {
            spec.tracks.samples_per_frame()?;
        }
        if spec.fps == 0 || !caps.fps.allows(spec.fps) {
            return Err(ApiError::invalid_param(
                "fps",
                format!("fps {} is not supported by `{}`", spec.fps, spec.model),
            ));
        }
        let mut st = self.lock();
        if st.draining {
            return Err(draining());
        }
        match st.pool.model_state(&spec.model) {
            Some(Residency::Resident) => {}
            Some(Residency::Pending | Residency::Loading { .. }) => {
                return Err(ApiError::loading(format!("model `{}` is still loading", spec.model))
                    .with_retry_after(1))
            }
            _ => {
                return Err(ApiError::loading(format!(
                    "model `{}` is not resident",
                    spec.model
                ))
                .with_retry_after(1))
            }
        }
        let exec = st.sched.session_executor(&spec.model).ok_or_else(busy)?;
        st.sched.open_session(exec, sid, is_causal)?;
        if let Some(cs) = causal {
            st.causal.insert(sid, cs);
        }
        st.yield_warmups();
        self.changed(&st);
        Ok((exec, caps))
    }

    /// Releases a clip session's executor slot.
    pub fn close_clip_session(&self, sid: SessionId) {
        let mut st = self.lock();
        st.sched.close_session(sid);
        self.changed(&st);
    }

    /// Releases a causal session: frees the lease now and queues the backend
    /// `causal_close` for the executor.
    pub fn close_causal_session(&self, sid: SessionId) {
        let mut st = self.lock();
        if let Some(cs) = st.causal.remove(&sid) {
            if let Some(i) = st.sched.close_session(sid) {
                st.to_close[i].push(sid);
            }
            drop(cs);
        }
        self.changed(&st);
    }
}

fn draining() -> ApiError {
    ApiError::loading("the engine is shutting down").with_retry_after(5)
}

struct Inner {
    shared: Arc<Shared>,
    threads: Mutex<Vec<std::thread::JoinHandle<()>>>,
}

impl Drop for Inner {
    fn drop(&mut self) {
        let mut st = self.shared.lock();
        st.shutdown = true;
        st.yield_warmups();
        self.shared.changed(&st);
    }
}

/// The engine (design §3.6). Cheap to clone; the executors shut down when the
/// last clone is dropped (or on [`EngineService::drain`]).
#[derive(Clone)]
pub struct EngineService {
    inner: Arc<Inner>,
}

impl std::fmt::Debug for EngineService {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EngineService")
            .field("models", &self.inner.shared.caps.len())
            .field("readiness", &self.readiness())
            .finish()
    }
}

impl EngineService {
    /// Starts one executor thread per backend (one per GPU). Each executor
    /// loads its backend's `resident` models; [`readiness`] reports progress.
    ///
    /// **Deviation from design §3.6:** returns `Result`, since two backends
    /// declaring one model id with different caps (or a bad tier override)
    /// is a configuration error. Needs no tokio runtime.
    ///
    /// [`readiness`]: EngineService::readiness
    pub fn start(
        cfg: EngineConfig,
        backends: Vec<Box<dyn EngineBackend>>,
    ) -> Result<Self, ApiError> {
        if backends.is_empty() {
            return Err(ApiError::internal("the engine needs at least one backend"));
        }
        let mut per_exec = Vec::new();
        let mut execs = Vec::new();
        let mut pool = ModelPool::default();
        for (i, b) in backends.iter().enumerate() {
            let caps = b.caps();
            let mut ex = ExecState::default();
            let mut list: Vec<(ModelCaps, Recipe)> = Vec::new();
            for c in caps {
                ex.serves.insert(c.id.clone());
                if c.resident {
                    ex.resident.insert(c.id.clone());
                }
                pool.declare(i, c.id.clone(), c.resident);
                let r = b.recipe(&c.id);
                list.push((c, r));
            }
            execs.push(ex);
            per_exec.push(list);
        }
        let caps = CapabilityTable::build(per_exec, &cfg.tier_overrides)?;
        let n = backends.len();
        let readiness = watch::Sender::new(pool.readiness());
        let shared = Arc::new(Shared {
            state: Mutex::new(State {
                sched: Scheduler::new(execs, cfg.queue_max, cfg.swap),
                pool,
                jobs: HashMap::new(),
                causal: HashMap::new(),
                to_close: vec![Vec::new(); n],
                draining: false,
                shutdown: false,
                alive: n,
                warmup_cancel: vec![None; n],
            }),
            work_cv: Condvar::new(),
            changed: Notify::new(),
            readiness,
            caps,
            cfg,
        });
        let mut threads = Vec::new();
        for (i, b) in backends.into_iter().enumerate() {
            let sh = shared.clone();
            let t = std::thread::Builder::new()
                .name(format!("fv-exec-{i}"))
                .spawn(move || executor::run(sh, i, b))
                .map_err(|e| ApiError::internal(format!("spawning executor {i}: {e}")))?;
            threads.push(t);
        }
        Ok(Self {
            inner: Arc::new(Inner {
                shared,
                threads: Mutex::new(threads),
            }),
        })
    }

    fn shared(&self) -> &Arc<Shared> {
        &self.inner.shared
    }

    /// Every served model (design §3.6 `caps()`).
    pub fn caps(&self) -> &CapabilityTable {
        &self.shared().caps
    }

    pub fn readiness(&self) -> Readiness {
        self.shared().readiness.borrow().clone()
    }

    /// A receiver that sees every readiness change.
    pub fn watch_readiness(&self) -> watch::Receiver<Readiness> {
        self.shared().readiness.subscribe()
    }

    /// Waits until loading finishes (`Ready` or `Failed`).
    pub async fn wait_ready(&self) -> Readiness {
        let mut rx = self.watch_readiness();
        let r = match rx
            .wait_for(|r| !matches!(r, Readiness::Loading { .. }))
            .await
        {
            Ok(r) => Some(r.clone()),
            Err(_) => None,
        };
        r.unwrap_or_else(|| self.readiness())
    }

    /// `warming` while a resident model still warms up in the background,
    /// `warm` once it has, `off` when nothing warms up in the background
    /// (fast boot B; readiness does not wait for it).
    pub fn warmup(&self) -> &'static str {
        self.shared().lock().pool.warmup_summary()
    }

    /// Residency of every (executor, model).
    pub fn pool(&self) -> ModelPool {
        self.shared().lock().pool.clone()
    }

    /// Queues a job (design §3.6). Errors: `InvalidRequest` (model not
    /// served / not resident without swap), `Loading` (model still loading,
    /// or draining), `EngineFailed` (its load failed), `QueueFull`.
    pub async fn submit(
        &self,
        job: JobId,
        r: ResolvedJob,
        prio: Priority,
    ) -> Result<JobHandle, ApiError> {
        self.shared().submit(job, r, prio, None, None)
    }

    /// Like [`submit`](Self::submit) with the output collected in memory
    /// (`ClipOutput::frames` / `audio`) instead of an MP4.
    pub async fn submit_frames(
        &self,
        job: JobId,
        r: ResolvedJob,
        prio: Priority,
    ) -> Result<JobHandle, ApiError> {
        self.shared()
            .submit(job, r, prio, None, Some(OutputMode::Frames))
    }

    /// Cancels by id (for routes that only know the job id).
    pub fn cancel(&self, job: JobId) -> CancelOutcome {
        let (token, queued) = {
            let st = self.shared().lock();
            match st.jobs.get(&job) {
                Some(e) => (e.cancel.clone(), !e.running),
                None => return CancelOutcome::Unknown,
            }
        };
        token.cancel();
        if queued {
            CancelOutcome::Dequeued
        } else {
            CancelOutcome::Requested
        }
    }

    /// 0-based queue position, `None` when not queued.
    pub fn queue_position(&self, job: JobId) -> Option<u32> {
        self.shared().lock().sched.position_of(job)
    }

    pub fn stats(&self) -> EngineStats {
        let st = self.shared().lock();
        let queued_batch = st.sched.queued_batch();
        EngineStats {
            queued_batch,
            queued_stream: st.sched.queued() - queued_batch,
            running: st.jobs.values().filter(|e| e.running).count(),
            sessions: (0..st.sched.executors())
                .filter(|&i| st.sched.exec(i).session.is_some())
                .count(),
            draining: st.draining,
            executors: st.sched.executors(),
        }
    }

    /// Opens a clip-queue session (H3, LTX, FastWan; design §5.5). One
    /// session per executor; the model must be resident.
    pub async fn open_clip_session(&self, spec: SessionSpec) -> Result<ClipSession, ApiError> {
        let sid = SessionId::new();
        let (exec, caps) = self.shared().open_session(&spec, sid, None)?;
        Ok(ClipSession::new(self.shared().clone(), sid, exec, spec, caps))
    }

    /// Opens a duplex session (design §5.11): admission as for any stream
    /// (one session per executor, resident model), the model's input rings,
    /// and — for the loopback echo — the model worker, started by
    /// [`DuplexSession::start`](crate::stream::duplex::DuplexSession::start).
    pub async fn open_duplex_session(
        &self,
        spec: fastvideo_protocol::DuplexSpec,
    ) -> Result<crate::stream::duplex::DuplexSession, ApiError> {
        spec.context.validate()?;
        let sid = SessionId::new();
        let (exec, caps) = self.shared().open_session_of(&spec.session, sid, None, true)?;
        crate::stream::duplex::DuplexSession::new(self.shared().clone(), sid, exec, spec, caps)
    }

    /// Opens a causal SF-Wan session under an exclusive executor lease
    /// (design §5.4). Batch jobs for that executor wait until it closes.
    pub async fn open_causal_session(&self, spec: SessionSpec) -> Result<CausalSession, ApiError> {
        let sid = SessionId::new();
        let (mut session, cs) = CausalSession::new(self.shared().clone(), sid, spec.clone());
        let (exec, caps) = self.shared().open_session(&spec, sid, Some(cs))?;
        session.attach(exec, caps);
        Ok(session)
    }

    /// Graceful shutdown (design §6.3): stop admission, cancel queued jobs,
    /// let running ones finish within `grace`, then cancel them, close
    /// sessions and stop the executors.
    pub async fn drain(&self, grace: Duration) {
        let sh = self.shared().clone();
        let sessions: Vec<Arc<CausalShared>> = {
            let mut st = sh.lock();
            st.draining = true;
            st.yield_warmups();
            for it in st.sched.clear() {
                if let Some(e) = st.jobs.remove(&it.job) {
                    let _ = e.tx.send(EngineEvent::Cancelled);
                }
            }
            sh.changed(&st);
            st.causal.values().cloned().collect()
        };
        for s in &sessions {
            s.close();
        }
        let idle = |st: &State| st.jobs.values().all(|e| !e.running);
        if !sh.wait_until(grace, idle).await {
            let tokens: Vec<CancelToken> = sh
                .lock()
                .jobs
                .values()
                .filter(|e| e.running)
                .map(|e| e.cancel.clone())
                .collect();
            for t in tokens {
                t.cancel();
            }
            sh.wait_until(Duration::from_secs(30), idle).await;
        }
        {
            let mut st = sh.lock();
            st.shutdown = true;
            st.yield_warmups();
            sh.changed(&st);
        }
        if sh.wait_until(Duration::from_secs(10), |st| st.alive == 0).await {
            for t in lock(&self.inner.threads).drain(..) {
                let _ = t.join();
            }
        }
    }
}
