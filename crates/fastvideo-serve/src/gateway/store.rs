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
}

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
        Self { inner, poll, watchers: Arc::default(), view: Arc::new(View { fresh: poll, ..View::default() }) }
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

async fn poller(
    store: Arc<D1JobStore>,
    view: Arc<View>,
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
        let Some(j) = store.get(id).await else { break };
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
        counted("d1");
        let j = self.inner.by_external(p, external_id).await?;
        self.view.remember(&j);
        Some(j)
    }

    async fn update(&self, id: JobId, f: JobUpdate) -> Result<Job, StoreError> {
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
        tokio::spawn(poller(self.inner.clone(), self.view.clone(), id, tx, self.watchers.clone(), self.poll));
        Some(rx)
    }

    async fn sweep_expired(&self, now: OffsetDateTime) -> usize {
        self.inner.sweep_expired(now).await
    }
}
