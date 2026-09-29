//! `MemJobStore`: an in-memory map plus, optionally, one JSON manifest per job
//! under `<state_dir>/jobs/` (design §3.4, the h3fast pattern).
//!
//! - [`MemJobStore::open`] reloads manifests; `Queued`/`Running` jobs become
//!   `Failed(Internal, "interrupted by restart")` and are rewritten.
//! - Every `insert`/`update` rewrites the job's manifest (write to a temp file,
//!   then rename), so a crash never leaves a half-written manifest.
//! - `sweep_expired` and `remove` delete the manifest, the job's artifacts
//!   (through the configured [`ArtifactStore`]) and its staged inputs dir.
//!
//! No Redis: the store is per process (design §6.3).

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use fastvideo_protocol::{
    Job, JobId, JobSnapshot, JobStore, JobUpdate, ListQuery, Page, ProtocolId, StoreError,
};
use time::OffsetDateTime;
use tokio::sync::watch;

use crate::artifacts::ArtifactStore;

struct Entry {
    job: Job,
    tx: watch::Sender<JobSnapshot>,
}

#[derive(Default)]
struct Inner {
    jobs: HashMap<JobId, Entry>,
    by_ext: HashMap<(ProtocolId, String), JobId>,
    seq: u64,
}

/// In-memory [`JobStore`] with optional durable manifests.
pub struct MemJobStore {
    inner: Mutex<Inner>,
    /// Serializes mutations with their manifest writes, so manifests land in
    /// the same order as the in-memory changes.
    io: tokio::sync::Mutex<()>,
    manifests: Option<PathBuf>,
    artifacts: Option<Arc<dyn ArtifactStore>>,
    inputs_root: Option<PathBuf>,
    max_jobs: Option<usize>,
}

impl Default for MemJobStore {
    fn default() -> Self {
        Self::memory()
    }
}

impl MemJobStore {
    /// A purely in-memory store (tests, Runpod LB sync-only).
    pub fn memory() -> Self {
        Self {
            inner: Mutex::new(Inner::default()),
            io: tokio::sync::Mutex::new(()),
            manifests: None,
            artifacts: None,
            inputs_root: None,
            max_jobs: None,
        }
    }

    /// A durable store with manifests in `dir` (created if missing). Reloads
    /// existing manifests and applies restart recovery at `now`. Unreadable
    /// manifests are skipped with a warning.
    pub async fn open(dir: impl Into<PathBuf>, now: OffsetDateTime) -> std::io::Result<Self> {
        let dir = dir.into();
        tokio::fs::create_dir_all(&dir).await?;
        let mut store = Self::memory();
        store.manifests = Some(dir.clone());
        let mut rd = tokio::fs::read_dir(&dir).await?;
        let mut loaded = Vec::new();
        while let Some(e) = rd.next_entry().await? {
            let p = e.path();
            if p.extension().and_then(|x| x.to_str()) != Some("json") {
                continue;
            }
            let job = match tokio::fs::read(&p).await.map(|b| serde_json::from_slice::<Job>(&b)) {
                Ok(Ok(j)) => j,
                Ok(Err(err)) => {
                    tracing::warn!(path = %p.display(), %err, "skipping unreadable job manifest");
                    continue;
                }
                Err(err) => {
                    tracing::warn!(path = %p.display(), %err, "skipping unreadable job manifest");
                    continue;
                }
            };
            loaded.push(job);
        }
        for mut job in loaded {
            if job.recover_after_restart(now) {
                write_manifest(&dir, &job).await?;
            }
            let inner = store.inner.get_mut().expect("fresh mutex");
            inner.seq += 1;
            let (tx, _) = watch::channel(job.snapshot(inner.seq));
            inner.by_ext.insert((job.protocol, job.external_id.clone()), job.id);
            inner.jobs.insert(job.id, Entry { job, tx });
        }
        Ok(store)
    }

    /// Deletes artifacts through `a` when jobs are removed or expire.
    pub fn with_artifacts(mut self, a: Arc<dyn ArtifactStore>) -> Self {
        self.artifacts = Some(a);
        self
    }
    /// Deletes `<root>/<job_id>/` (staged inputs) when jobs are removed or expire.
    pub fn with_inputs_root(mut self, root: impl Into<PathBuf>) -> Self {
        self.inputs_root = Some(root.into());
        self
    }
    /// Refuse inserts (`StoreError::Full`) beyond `n` live jobs.
    pub fn with_max_jobs(mut self, n: usize) -> Self {
        self.max_jobs = Some(n);
        self
    }

    pub fn len(&self) -> usize {
        self.lock().jobs.len()
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Inner> {
        self.inner.lock().unwrap_or_else(|p| p.into_inner())
    }

    async fn persist(&self, job: &Job) -> Result<(), StoreError> {
        if let Some(dir) = &self.manifests {
            write_manifest(dir, job)
                .await
                .map_err(|e| StoreError::Io(e.to_string()))?;
        }
        Ok(())
    }

    async fn cleanup(&self, job: &Job) {
        if let Some(dir) = &self.manifests {
            let _ = tokio::fs::remove_file(manifest_path(dir, job.id)).await;
        }
        if let Some(a) = &self.artifacts {
            for art in &job.artifacts {
                a.delete(art).await;
            }
        }
        if let Some(root) = &self.inputs_root {
            let _ = tokio::fs::remove_dir_all(root.join(job.id.to_string())).await;
        }
    }
}

fn manifest_path(dir: &Path, id: JobId) -> PathBuf {
    dir.join(format!("{id}.json"))
}

async fn write_manifest(dir: &Path, job: &Job) -> std::io::Result<()> {
    let bytes = serde_json::to_vec_pretty(job).map_err(std::io::Error::other)?;
    let tmp = dir.join(format!(".{}.json.tmp", job.id));
    tokio::fs::write(&tmp, bytes).await?;
    tokio::fs::rename(&tmp, manifest_path(dir, job.id)).await
}

#[async_trait::async_trait]
impl JobStore for MemJobStore {
    async fn insert(&self, job: Job) -> Result<(), StoreError> {
        let _io = self.io.lock().await;
        {
            let mut g = self.lock();
            if g.jobs.contains_key(&job.id) {
                return Err(StoreError::AlreadyExists(job.id));
            }
            let key = (job.protocol, job.external_id.clone());
            if g.by_ext.contains_key(&key) {
                return Err(StoreError::DuplicateExternal(job.protocol, job.external_id));
            }
            if self.max_jobs.is_some_and(|m| g.jobs.len() >= m) {
                return Err(StoreError::Full);
            }
            g.seq += 1;
            let (tx, _) = watch::channel(job.snapshot(g.seq));
            g.by_ext.insert(key, job.id);
            g.jobs.insert(job.id, Entry { job: job.clone(), tx });
        }
        if let Err(e) = self.persist(&job).await {
            let mut g = self.lock();
            g.jobs.remove(&job.id);
            g.by_ext.remove(&(job.protocol, job.external_id.clone()));
            return Err(e);
        }
        Ok(())
    }

    async fn get(&self, id: JobId) -> Option<Job> {
        self.lock().jobs.get(&id).map(|e| e.job.clone())
    }

    async fn by_external(&self, p: ProtocolId, external_id: &str) -> Option<Job> {
        let g = self.lock();
        let id = g.by_ext.get(&(p, external_id.to_owned()))?;
        g.jobs.get(id).map(|e| e.job.clone())
    }

    async fn update(
        &self,
        id: JobId,
        f: JobUpdate,
    ) -> Result<Job, StoreError> {
        let _io = self.io.lock().await;
        let job = {
            let mut g = self.lock();
            g.seq += 1;
            let seq = g.seq;
            let e = g.jobs.get_mut(&id).ok_or(StoreError::NotFound(id))?;
            let mut next = e.job.clone();
            f(&mut next);
            // Identity fields are fixed.
            next.id = e.job.id;
            next.protocol = e.job.protocol;
            next.external_id = e.job.external_id.clone();
            e.job = next.clone();
            e.tx.send_replace(next.snapshot(seq));
            next
        };
        self.persist(&job).await?;
        Ok(job)
    }

    async fn list(&self, q: ListQuery) -> Page<Job> {
        let g = self.lock();
        q.apply(g.jobs.values().map(|e| &e.job))
    }

    async fn remove(&self, id: JobId) -> Option<Job> {
        let _io = self.io.lock().await;
        let job = {
            let mut g = self.lock();
            let e = g.jobs.remove(&id)?;
            g.by_ext.remove(&(e.job.protocol, e.job.external_id.clone()));
            e.job
        };
        self.cleanup(&job).await;
        Some(job)
    }

    fn watch(&self, id: JobId) -> Option<watch::Receiver<JobSnapshot>> {
        self.lock().jobs.get(&id).map(|e| e.tx.subscribe())
    }

    async fn sweep_expired(&self, now: OffsetDateTime) -> usize {
        let _io = self.io.lock().await;
        let expired: Vec<Job> = {
            let mut g = self.lock();
            let ids: Vec<JobId> = g
                .jobs
                .values()
                .filter(|e| e.job.is_expired(now))
                .map(|e| e.job.id)
                .collect();
            ids.into_iter()
                .filter_map(|id| {
                    let e = g.jobs.remove(&id)?;
                    g.by_ext.remove(&(e.job.protocol, e.job.external_id.clone()));
                    Some(e.job)
                })
                .collect()
        };
        for j in &expired {
            self.cleanup(j).await;
        }
        expired.len()
    }
}

/// Runs `store.sweep_expired` every `every` until the returned handle is aborted.
pub fn spawn_sweeper(
    store: Arc<dyn JobStore>,
    every: std::time::Duration,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        let mut t = tokio::time::interval(every);
        t.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        loop {
            t.tick().await;
            let n = store.sweep_expired(OffsetDateTime::now_utc()).await;
            if n > 0 {
                tracing::info!(removed = n, "expired jobs swept");
            }
        }
    })
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use fastvideo_protocol::{
        ApiError, AudioPlan, ErrorKind, JobMetrics, JobState, JobStatus, ModelId, PostProcess,
        ResolvedJob, SamplingOverrides, SortOrder, Task,
    };
    use std::time::Duration;

    pub fn resolved() -> ResolvedJob {
        ResolvedJob {
            model: ModelId::new("fasth3"),
            task: Task::T2V,
            prompt: "a cat".into(),
            negative_prompt: String::new(),
            seed: 7,
            width: 1344,
            height: 768,
            num_frames: 124,
            fps: 24,
            keyframes: vec![],
            references: vec![],
            audio_in: None,
            audio: AudioPlan::Native { rate: 32000, channels: 2 },
            post: PostProcess::default(),
            sampling: SamplingOverrides::default(),
            tier: None,
            recipe: None,
            edit: None,
        }
    }

    pub fn job(p: ProtocolId, ext: &str, at: OffsetDateTime) -> Job {
        Job::new(JobId::new(), p, ext, resolved(), at, Duration::from_secs(3600))
    }

    fn t0() -> OffsetDateTime {
        time::macros::datetime!(2026-09-27 12:00 UTC)
    }

    #[tokio::test]
    async fn insert_get_external_and_duplicates() {
        let s = MemJobStore::memory();
        let j = job(ProtocolId::Fal, "abc", t0());
        s.insert(j.clone()).await.unwrap();
        assert_eq!(s.get(j.id).await.unwrap().external_id, "abc");
        assert!(s.by_external(ProtocolId::Fal, "abc").await.is_some());
        assert!(s.by_external(ProtocolId::LtxV2, "abc").await.is_none());
        assert_eq!(s.insert(j.clone()).await, Err(StoreError::AlreadyExists(j.id)));
        let dup = job(ProtocolId::Fal, "abc", t0());
        assert!(matches!(s.insert(dup).await, Err(StoreError::DuplicateExternal(..))));
        // Same external id under another protocol is fine.
        s.insert(job(ProtocolId::LtxV2, "abc", t0())).await.unwrap();
    }

    #[tokio::test]
    async fn update_is_atomic_and_watched() {
        let s = MemJobStore::memory();
        let j = job(ProtocolId::MiniMaxV2, "1", t0());
        s.insert(j.clone()).await.unwrap();
        let mut rx = s.watch(j.id).unwrap();
        let seq0 = rx.borrow().seq;
        let out = s
            .update(j.id, Box::new(move |j| {
                j.mark_running(t0()).unwrap();
                j.set_step(5, 10);
                j.external_id = "tamper".into(); // identity fields are restored
            }))
            .await
            .unwrap();
        assert_eq!(out.external_id, "1");
        assert!(rx.has_changed().unwrap());
        let snap = rx.borrow_and_update().clone();
        assert!(snap.seq > seq0);
        assert_eq!(snap.state, JobState::Running);
        assert_eq!(snap.progress, 0.5);
        let missing = JobId::new();
        assert_eq!(
            s.update(missing, Box::new(|_| {})).await.unwrap_err(),
            StoreError::NotFound(missing)
        );
    }

    #[tokio::test]
    async fn list_filters_and_pages() {
        let s = MemJobStore::memory();
        for i in 0..5 {
            let mut j = job(ProtocolId::OpenAiVideos, &format!("v{i}"), t0() + Duration::from_secs(i));
            if i % 2 == 0 {
                j.mark_cancelled(t0()).unwrap();
            }
            s.insert(j).await.unwrap();
        }
        s.insert(job(ProtocolId::Fal, "f", t0())).await.unwrap();
        let q = ListQuery { protocol: Some(ProtocolId::OpenAiVideos), limit: 2, ..Default::default() };
        let p = s.list(q.clone()).await;
        assert_eq!(p.total, 5);
        assert!(p.has_more);
        assert_eq!(p.items.iter().map(|j| j.external_id.as_str()).collect::<Vec<_>>(), ["v4", "v3"]);
        let p2 = s.list(ListQuery { after: Some("v3".into()), ..q.clone() }).await;
        assert_eq!(p2.items[0].external_id, "v2");
        let asc = s
            .list(ListQuery { order: SortOrder::Asc, statuses: vec![JobStatus::Queued], ..q })
            .await;
        assert_eq!(asc.items.iter().map(|j| j.external_id.as_str()).collect::<Vec<_>>(), ["v1", "v3"]);
    }

    #[tokio::test]
    async fn max_jobs_is_full() {
        let s = MemJobStore::memory().with_max_jobs(1);
        s.insert(job(ProtocolId::Fal, "a", t0())).await.unwrap();
        assert_eq!(s.insert(job(ProtocolId::Fal, "b", t0())).await, Err(StoreError::Full));
        let e: ApiError = StoreError::Full.into();
        assert_eq!(e.kind, ErrorKind::QueueFull);
    }

    #[tokio::test]
    async fn manifests_restart_recovery_and_sweep() {
        let dir = std::env::temp_dir().join(format!("fvkit-store-{}", crate::random_token()));
        let inputs = dir.join("inputs");
        let (running, done, queued);
        {
            let s = MemJobStore::open(dir.join("jobs"), t0()).await.unwrap();
            let a = job(ProtocolId::Fal, "run", t0());
            let b = job(ProtocolId::Fal, "done", t0());
            let c = job(ProtocolId::MiniMaxV2, "q", t0());
            running = a.id;
            done = b.id;
            queued = c.id;
            for j in [a, b, c] {
                s.insert(j).await.unwrap();
            }
            s.update(running, Box::new(|j| j.mark_running(t0()).unwrap())).await.unwrap();
            s.update(done, Box::new(|j| {
                j.mark_running(t0()).unwrap();
                j.mark_succeeded(t0(), vec![], JobMetrics::default()).unwrap();
            }))
            .await
            .unwrap();
            // A corrupt manifest is skipped.
            tokio::fs::write(dir.join("jobs/junk.json"), b"{not json").await.unwrap();
        }
        let later = t0() + Duration::from_secs(60);
        let s = MemJobStore::open(dir.join("jobs"), later)
            .await
            .unwrap()
            .with_inputs_root(&inputs);
        assert_eq!(s.len(), 3);
        for id in [running, queued] {
            let j = s.get(id).await.unwrap();
            let e = j.state.error().expect("failed after restart");
            assert_eq!(e.kind, ErrorKind::Internal);
            assert_eq!(e.message, "interrupted by restart");
            assert_eq!(j.completed_at, Some(later));
        }
        assert_eq!(s.get(done).await.unwrap().status(), JobStatus::Succeeded);
        assert!(s.by_external(ProtocolId::MiniMaxV2, "q").await.is_some());
        // Recovery was persisted: a third open sees the same.
        let s3 = MemJobStore::open(dir.join("jobs"), later + Duration::from_secs(1)).await.unwrap();
        assert_eq!(
            s3.get(running).await.unwrap().completed_at,
            Some(later),
            "recovery rewrote the manifest"
        );
        drop(s3);

        tokio::fs::create_dir_all(inputs.join(done.to_string())).await.unwrap();
        assert_eq!(s.sweep_expired(t0() + Duration::from_secs(10)).await, 0);
        assert_eq!(s.sweep_expired(t0() + Duration::from_secs(3600)).await, 3);
        assert!(s.is_empty());
        assert!(!inputs.join(done.to_string()).exists());
        assert!(!dir.join(format!("jobs/{done}.json")).exists());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn remove_closes_watch() {
        let s = MemJobStore::memory();
        let j = job(ProtocolId::Fal, "x", t0());
        s.insert(j.clone()).await.unwrap();
        let mut rx = s.watch(j.id).unwrap();
        assert!(s.remove(j.id).await.is_some());
        assert!(rx.changed().await.is_err());
        assert!(s.watch(j.id).is_none());
        assert!(s.by_external(ProtocolId::Fal, "x").await.is_none());
    }
}
