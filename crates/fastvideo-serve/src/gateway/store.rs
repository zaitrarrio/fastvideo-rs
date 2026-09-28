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

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use fastvideo_protocol::{Job, JobId, JobSnapshot, JobState, JobStore, JobUpdate, ListQuery, Page, ProtocolId, StoreError};
use fastvideo_serve_kit::D1JobStore;
use time::OffsetDateTime;
use tokio::sync::watch;

/// See the module docs.
pub struct GatewayJobStore {
    inner: Arc<D1JobStore>,
    poll: Duration,
    watchers: Arc<Mutex<HashMap<JobId, Arc<watch::Sender<JobSnapshot>>>>>,
}

impl std::fmt::Debug for GatewayJobStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GatewayJobStore").field("poll", &self.poll).finish_non_exhaustive()
    }
}

impl GatewayJobStore {
    pub fn new(inner: Arc<D1JobStore>, poll: Duration) -> Self {
        Self { inner, poll: poll.max(Duration::from_millis(20)), watchers: Arc::default() }
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
        self.inner.insert(job).await
    }

    async fn get(&self, id: JobId) -> Option<Job> {
        self.inner.get(id).await
    }

    async fn by_external(&self, p: ProtocolId, external_id: &str) -> Option<Job> {
        self.inner.by_external(p, external_id).await
    }

    async fn update(&self, id: JobId, f: JobUpdate) -> Result<Job, StoreError> {
        match self.inner.update(id, f).await {
            Err(StoreError::Io(m)) if m.contains("changed concurrently") => {
                tracing::debug!(job = %id, "gateway update lost to a worker write; the worker's copy stands");
                self.inner.get(id).await.ok_or(StoreError::NotFound(id))
            }
            r => r,
        }
    }

    async fn list(&self, q: ListQuery) -> Page<Job> {
        self.inner.list(q).await
    }

    async fn remove(&self, id: JobId) -> Option<Job> {
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
        tokio::spawn(poller(self.inner.clone(), id, tx, self.watchers.clone(), self.poll));
        Some(rx)
    }

    async fn sweep_expired(&self, now: OffsetDateTime) -> usize {
        self.inner.sweep_expired(now).await
    }
}
