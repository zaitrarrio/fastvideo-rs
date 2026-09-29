//! The gateway's job store (docs/serve/gateway.md §3): D1 read-through.
//!
//! The gateway inserts rows (`hold_inserts = false`: no cache, no
//! `worker`), workers adopt and update them, and every gateway read goes to
//! D1. `watch()` has no in-process source of changes here, so it polls D1
//! (one poller per watched job, shared by its receivers, stopped when the
//! last receiver goes or the job ends): fal `status/stream` SSE and the
//! sync endpoints' waits work unchanged. A version conflict on an update
//! (a worker wrote the row meanwhile) is not an error for the gateway: the
//! worker's copy is authoritative, so the fresh row is returned.
//!
//! Status reads (a fal client polls `status` every few hundred ms; each D1
//! read costs ~0.3 s) are served from an in-memory view where the state is
//! known: a job this replica inserted, updated or read within the watch
//! poll interval (`gateway.watch_poll_ms`, the latency SSE already has),
//! or a finished job read within the last minute (finished jobs change only
//! by deletion). Anything else reads D1 and refreshes the view; the
//! pollers behind `watch()` refresh it too.
//!
//! **Insert behind** (pools with `dispatch = "durable-object"`,
//! docs/serve/gateway-cloudflare.md phase 2): for a job the predicate set
//! with [`GatewayJobStore::set_insert_behind`] accepts, `insert` returns at
//! once and the D1 insert runs in the background, so the dispatch does not
//! wait for it. Until it lands the job is answered from memory; updates and
//! removals wait for it. The worker's own write may land first: the late
//! insert then keeps the worker's row (`D1JobStore::insert` treats a row
//! with the same external id as its own).

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use fastvideo_protocol::{Job, JobId, JobSnapshot, JobState, JobStore, JobUpdate, ListQuery, Page, ProtocolId, StoreError};
use fastvideo_serve_kit::D1JobStore;
use time::OffsetDateTime;
use tokio::sync::watch;

/// See the module docs.
pub struct GatewayJobStore {
    inner: Arc<D1JobStore>,
    poll: Duration,
    watchers: Arc<Mutex<HashMap<JobId, Arc<watch::Sender<JobSnapshot>>>>>,
    view: Arc<View>,
    behind: std::sync::OnceLock<InsertBehind>,
    pending: Pending,
}

/// Which jobs are inserted behind (see the module docs).
pub type InsertBehind = Arc<dyn Fn(&Job) -> bool + Send + Sync>;

/// Jobs whose D1 insert is in flight: the job, and a flag set when it lands.
type Pending = Arc<Mutex<HashMap<JobId, (Job, watch::Receiver<bool>)>>>;

/// How long an update waits for an insert in flight.
const PENDING_WAIT: Duration = Duration::from_secs(15);

/// How long a finished job read from D1 is served from memory.
const TERMINAL_FRESH: Duration = Duration::from_secs(60);
/// Jobs the view holds before it drops the stale ones.
const VIEW_MAX: usize = 4096;

/// The in-memory view of recently seen jobs (see the module docs).
#[derive(Default)]
struct View {
    fresh: Duration,
    jobs: Mutex<ViewInner>,
}

#[derive(Default)]
struct ViewInner {
    by_id: HashMap<JobId, (Job, Instant)>,
    by_ext: HashMap<(ProtocolId, String), JobId>,
}

impl View {
    fn lock(&self) -> std::sync::MutexGuard<'_, ViewInner> {
        self.jobs.lock().unwrap_or_else(|p| p.into_inner())
    }
    fn remember(&self, j: &Job) {
        let mut g = self.lock();
        if g.by_id.len() >= VIEW_MAX {
            let fresh = self.fresh;
            g.by_id.retain(|_, (j, at)| at.elapsed() < if j.is_terminal() { TERMINAL_FRESH } else { fresh });
            if g.by_id.len() >= VIEW_MAX {
                g.by_id.clear();
            }
            let ViewInner { by_id, by_ext } = &mut *g;
            by_ext.retain(|_, id| by_id.contains_key(id));
        }
        g.by_ext.insert((j.protocol, j.external_id.clone()), j.id);
        g.by_id.insert(j.id, (j.clone(), Instant::now()));
    }
    fn forget(&self, id: JobId) {
        let mut g = self.lock();
        if let Some((j, _)) = g.by_id.remove(&id) {
            g.by_ext.remove(&(j.protocol, j.external_id));
        }
    }
    fn get(&self, id: JobId) -> Option<Job> {
        let g = self.lock();
        let (j, at) = g.by_id.get(&id)?;
        let ttl = if j.is_terminal() { TERMINAL_FRESH } else { self.fresh };
        (at.elapsed() < ttl).then(|| j.clone())
    }
    fn by_external(&self, p: ProtocolId, ext: &str) -> Option<Job> {
        let id = *self.lock().by_ext.get(&(p, ext.to_owned()))?;
        self.get(id)
    }
}

fn counted(source: &'static str) {
    metrics::counter!("fv_gateway_job_reads_total", "source" => source).increment(1);
}

impl std::fmt::Debug for GatewayJobStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GatewayJobStore").field("poll", &self.poll).finish_non_exhaustive()
    }
}

impl GatewayJobStore {
    pub fn new(inner: Arc<D1JobStore>, poll: Duration) -> Self {
        let poll = poll.max(Duration::from_millis(20));
        Self {
            inner,
            poll,
            watchers: Arc::default(),
            view: Arc::new(View { fresh: poll, ..View::default() }),
            behind: std::sync::OnceLock::new(),
            pending: Arc::default(),
        }
    }

    /// Inserts the jobs `f` accepts behind the response (once; see the
    /// module docs).
    pub fn set_insert_behind(&self, f: InsertBehind) {
        let _ = self.behind.set(f);
    }

    fn pending_job(&self, id: JobId) -> Option<Job> {
        self.pending.lock().unwrap_or_else(|p| p.into_inner()).get(&id).map(|(j, _)| j.clone())
    }

    /// Waits until an insert in flight for `id` landed (or gave up).
    async fn settled(&self, id: JobId) {
        let rx = self.pending.lock().unwrap_or_else(|p| p.into_inner()).get(&id).map(|(_, rx)| rx.clone());
        if let Some(mut rx) = rx {
            let _ = tokio::time::timeout(PENDING_WAIT, rx.wait_for(|done| *done)).await;
        }
    }

    pub fn d1(&self) -> &Arc<D1JobStore> {
        &self.inner
    }
}

/// What a poll compares (the parts a status view shows).
fn fingerprint(j: &Job) -> (String, u32, Option<u32>, usize, usize, bool) {
    (
        j.status().as_str().to_owned(),
        (j.progress * 1000.0) as u32,
        j.queue_position,
        j.logs.len(),
        j.artifacts.len(),
        j.cancel_requested,
    )
}

#[allow(clippy::too_many_arguments)]
async fn poller(
    store: Arc<D1JobStore>,
    view: Arc<View>,
    pending: Pending,
    id: JobId,
    tx: Arc<watch::Sender<JobSnapshot>>,
    watchers: Arc<Mutex<HashMap<JobId, Arc<watch::Sender<JobSnapshot>>>>>,
    every: Duration,
) {
    let mut seq = 0u64;
    let mut last = None;
    loop {
        if tx.receiver_count() == 0 {
            break;
        }
        let j = match store.get(id).await {
            Some(j) => j,
            None => match pending.lock().unwrap_or_else(|p| p.into_inner()).get(&id).map(|(j, _)| j.clone()) {
                // Inserted behind, not in D1 yet.
                Some(j) => j,
                None => break,
            },
        };
        view.remember(&j);
        let fp = fingerprint(&j);
        if last.as_ref() != Some(&fp) {
            seq += 1;
            tx.send_replace(j.snapshot(seq));
            last = Some(fp);
        }
        if j.is_terminal() {
            break;
        }
        tokio::time::sleep(every).await;
    }
    let mut g = watchers.lock().unwrap_or_else(|p| p.into_inner());
    if g.get(&id).is_some_and(|t| Arc::ptr_eq(t, &tx)) {
        g.remove(&id);
    }
}

#[async_trait::async_trait]
impl JobStore for GatewayJobStore {
    async fn insert(&self, job: Job) -> Result<(), StoreError> {
        if self.behind.get().is_some_and(|f| f(&job)) {
            let id = job.id;
            let (tx, rx) = watch::channel(false);
            self.pending.lock().unwrap_or_else(|p| p.into_inner()).insert(id, (job.clone(), rx));
            self.view.remember(&job);
            let inner = self.inner.clone();
            let pending = self.pending.clone();
            tokio::spawn(async move {
                let t0 = std::time::Instant::now();
                match inner.insert(job).await {
                    Ok(()) => metrics::histogram!("fv_gateway_insert_behind_seconds").record(t0.elapsed().as_secs_f64()),
                    Err(e) => tracing::warn!(job = %id, error = %e, "gateway: a job row inserted behind the dispatch failed"),
                }
                pending.lock().unwrap_or_else(|p| p.into_inner()).remove(&id);
                let _ = tx.send(true);
            });
            return Ok(());
        }
        let j = job.clone();
        self.inner.insert(job).await?;
        self.view.remember(&j);
        Ok(())
    }

    async fn get(&self, id: JobId) -> Option<Job> {
        if let Some(j) = self.view.get(id) {
            counted("memory");
            return Some(j);
        }
        if let Some(j) = self.pending_job(id) {
            counted("memory");
            return Some(j);
        }
        counted("d1");
        let j = self.inner.get(id).await?;
        self.view.remember(&j);
        Some(j)
    }

    async fn by_external(&self, p: ProtocolId, external_id: &str) -> Option<Job> {
        if let Some(j) = self.view.by_external(p, external_id) {
            counted("memory");
            return Some(j);
        }
        let pend = self.pending.lock().unwrap_or_else(|p| p.into_inner()).values().find(|(j, _)| j.protocol == p && j.external_id == external_id).map(|(j, _)| j.clone());
        if let Some(j) = pend {
            counted("memory");
            return Some(j);
        }
        counted("d1");
        let j = self.inner.by_external(p, external_id).await?;
        self.view.remember(&j);
        Some(j)
    }

    async fn update(&self, id: JobId, f: JobUpdate) -> Result<Job, StoreError> {
        self.settled(id).await;
        let r = match self.inner.update(id, f).await {
            Err(StoreError::Io(m)) if m.contains("changed concurrently") => {
                tracing::debug!(job = %id, "gateway update lost to a worker write; the worker's copy stands");
                self.inner.get(id).await.ok_or(StoreError::NotFound(id))
            }
            r => r,
        };
        match &r {
            Ok(j) => self.view.remember(j),
            Err(_) => self.view.forget(id),
        }
        r
    }

    async fn list(&self, q: ListQuery) -> Page<Job> {
        self.inner.list(q).await
    }

    async fn remove(&self, id: JobId) -> Option<Job> {
        self.settled(id).await;
        self.view.forget(id);
        self.inner.remove(id).await
    }

    fn watch(&self, id: JobId) -> Option<watch::Receiver<JobSnapshot>> {
        let mut g = self.watchers.lock().unwrap_or_else(|p| p.into_inner());
        if let Some(tx) = g.get(&id) {
            return Some(tx.subscribe());
        }
        // A placeholder until the first poll (which runs at once).
        let (tx, rx) = watch::channel(JobSnapshot {
            id,
            seq: 0,
            state: JobState::Queued,
            progress: 0.0,
            queue_position: None,
            log_count: 0,
        });
        let tx = Arc::new(tx);
        g.insert(id, tx.clone());
        tokio::spawn(poller(self.inner.clone(), self.view.clone(), self.pending.clone(), id, tx, self.watchers.clone(), self.poll));
        Some(rx)
    }

    async fn sweep_expired(&self, now: OffsetDateTime) -> usize {
        self.inner.sweep_expired(now).await
    }
}
