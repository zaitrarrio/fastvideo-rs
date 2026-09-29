//! `D1JobStore`: a [`JobStore`] whose durable, shared copy lives in
//! Cloudflare D1 (design §0 decision 7).
//!
//! - **Authoritative memory.** Jobs this worker inserts stay in an in-memory
//!   cache (with `watch` channels) while they run and for
//!   [`D1Options::terminal_ttl`] after they finish. Reads of cached jobs never
//!   touch D1. Jobs from other workers (or from before a restart or
//!   scale-to-zero) are read from D1.
//! - **Writes.** `insert` is write-through (the wire id handed to the client
//!   must be durable). State changes (status, cancel request, artifacts) are
//!   written immediately. Progress, queue position and log lines are
//!   coalesced: at most one write per [`D1Options::progress_interval`]
//!   (1 s) per job, by a background flusher. Writes of one job are ordered
//!   (a per-job lock plus a sequence number); a failed write stays dirty and
//!   the flusher retries it, on top of the client's own retry/backoff.
//! - **Jobs owned elsewhere** are updated read-modify-write with an
//!   optimistic `version` check; a concurrent change is reported as
//!   `StoreError::Io` (the caller may retry). A job running on another
//!   worker is authoritative there: that worker's next write wins.
//! - **Liveness.** The flusher bumps `updated_at` of this worker's unfinished
//!   jobs every [`D1Options::heartbeat`]. `open` fails this worker's own
//!   unfinished jobs ("interrupted by restart"); `sweep_expired` also fails
//!   other workers' unfinished jobs whose heartbeat is older than
//!   [`D1Options::stale_after`] ("worker lost").
//! - `list` runs in D1 (indexes on owner/status/created) and overlays the
//!   fresher cached copies.
//!
//! Removal and expiry delete the row and, like `MemJobStore`, the job's
//! artifacts and staged inputs.

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::{Arc, Mutex, Weak};
use std::time::{Duration, Instant};

use fastvideo_protocol::{
    ApiError, Job, JobId, JobSnapshot, JobStore, JobUpdate, ListQuery, Page, ProtocolId,
    SortOrder, StoreError,
};
use serde_json::{json, Map, Value};
use time::OffsetDateTime;
use tokio::sync::watch;

use super::client::{D1Client, D1Error, Stmt};
use super::schema;
use crate::artifacts::ArtifactStore;

/// Tuning for [`D1JobStore`].
#[derive(Clone, Debug)]
pub struct D1Options {
    /// This worker's id (`worker` column): Runpod pod id, hostname, ...
    pub worker_id: String,
    /// Minimum spacing of progress-only writes per job (≤ 1 write/s/job).
    pub progress_interval: Duration,
    /// Flusher tick.
    pub flush_tick: Duration,
    /// How long finished jobs stay cached (and watchable) before eviction.
    pub terminal_ttl: Duration,
    /// `updated_at` heartbeat for this worker's unfinished jobs.
    pub heartbeat: Duration,
    /// Other workers' unfinished jobs without a heartbeat for this long are
    /// failed by `sweep_expired` (`None`: never).
    pub stale_after: Option<Duration>,
    /// Refuse inserts (`StoreError::Full`) beyond this many unfinished jobs.
    pub max_active: Option<usize>,
    /// Keep inserted jobs in memory as this worker's (the default). A
    /// gateway (docs/serve/gateway.md) sets `false`: it inserts rows for
    /// jobs a worker then [`adopt`](D1JobStore::adopt)s, so the row is
    /// written with no `worker` and every read goes to D1.
    pub hold_inserts: bool,
}

impl D1Options {
    pub fn new(worker_id: impl Into<String>) -> Self {
        Self {
            worker_id: worker_id.into(),
            progress_interval: Duration::from_secs(1),
            flush_tick: Duration::from_millis(200),
            terminal_ttl: Duration::from_secs(600),
            heartbeat: Duration::from_secs(60),
            stale_after: Some(Duration::from_secs(900)),
            max_active: None,
            hold_inserts: true,
        }
    }
}

struct Entry {
    job: Job,
    tx: watch::Sender<JobSnapshot>,
    seq: u64,
    /// Sequence number of the last version written to D1 (per-job write lock).
    flushed: Arc<tokio::sync::Mutex<u64>>,
    last_write: Option<Instant>,
    dirty: bool,
    terminal_since: Option<Instant>,
}

#[derive(Default)]
struct Cache {
    jobs: HashMap<JobId, Entry>,
    by_ext: HashMap<(ProtocolId, String), JobId>,
    seq: u64,
}

/// Counters for tests and metrics.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct D1Stats {
    /// Row writes (insert, upsert, conditional update) sent to D1.
    pub writes: u64,
    /// Writes that failed after the client's retries.
    pub failed_writes: u64,
    /// Reads that went to D1 (cache misses and lists).
    pub reads: u64,
}

/// The D1-backed job store. Build with [`D1JobStore::new`], then
/// [`D1JobStore::open`].
pub struct D1JobStore {
    db: D1Client,
    opts: D1Options,
    cache: Mutex<Cache>,
    stats: Mutex<D1Stats>,
    artifacts: Option<Arc<dyn ArtifactStore>>,
    inputs_root: Option<PathBuf>,
    last_heartbeat: Mutex<Option<Instant>>,
}

impl std::fmt::Debug for D1JobStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("D1JobStore")
            .field("worker", &self.opts.worker_id)
            .field("cached", &self.lock().jobs.len())
            .finish_non_exhaustive()
    }
}

const COLS: &str = "id, protocol, external_id, owner, status, model, resolved_model, task, progress, created_at, updated_at, completed_at, expires_at, worker, version, job";
const UNFINISHED: &str = "('queued', 'running')";

fn ms(t: OffsetDateTime) -> i64 {
    (t.unix_timestamp_nanos() / 1_000_000) as i64
}

fn now_ms() -> i64 {
    ms(OffsetDateTime::now_utc())
}

fn task_str(job: &Job) -> String {
    serde_json::to_value(job.task())
        .ok()
        .and_then(|v| v.as_str().map(str::to_owned))
        .unwrap_or_default()
}

fn opt_str(s: Option<&str>) -> Value {
    s.map_or(Value::Null, |s| json!(s))
}

/// The 16 column values of [`COLS`] for `job` (`version` 0).
fn row_params(job: &Job, worker: Option<&str>) -> Result<Vec<Value>, StoreError> {
    let body = serde_json::to_string(job).map_err(|e| StoreError::Io(format!("encoding job: {e}")))?;
    Ok(vec![
        json!(job.id.to_string()),
        json!(job.protocol.as_str()),
        json!(job.external_id),
        opt_str(job.owner.as_ref().map(|o| o.0.as_str())),
        json!(job.status().as_str()),
        json!(job.requested_model()),
        json!(job.resolved.model.0),
        json!(task_str(job)),
        json!(job.progress as f64),
        json!(ms(job.created_at)),
        json!(now_ms()),
        job.completed_at.map_or(Value::Null, |t| json!(ms(t))),
        json!(ms(job.expires_at)),
        opt_str(worker),
        json!(0),
        json!(body),
    ])
}

fn insert_stmt(job: &Job, worker: Option<&str>) -> Result<Stmt, StoreError> {
    Ok(Stmt::new(
        format!("INSERT INTO jobs ({COLS}) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"),
        row_params(job, worker)?,
    ))
}

fn upsert_stmt(job: &Job, worker: Option<&str>) -> Result<Stmt, StoreError> {
    Ok(Stmt::new(
        format!(
            "INSERT INTO jobs ({COLS}) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) \
             ON CONFLICT(id) DO UPDATE SET owner = excluded.owner, status = excluded.status, \
             model = excluded.model, resolved_model = excluded.resolved_model, task = excluded.task, \
             progress = excluded.progress, updated_at = excluded.updated_at, \
             completed_at = excluded.completed_at, expires_at = excluded.expires_at, \
             worker = excluded.worker, version = jobs.version + 1, job = excluded.job"
        ),
        row_params(job, worker)?,
    ))
}

/// `UPDATE ... WHERE id = ? AND version = ?` (jobs owned elsewhere).
fn cas_stmt(job: &Job, version: i64) -> Result<Stmt, StoreError> {
    let body = serde_json::to_string(job).map_err(|e| StoreError::Io(format!("encoding job: {e}")))?;
    Ok(Stmt::new(
        "UPDATE jobs SET owner = ?, status = ?, progress = ?, updated_at = ?, completed_at = ?, \
         expires_at = ?, job = ?, version = version + 1 WHERE id = ? AND version = ?",
        vec![
            opt_str(job.owner.as_ref().map(|o| o.0.as_str())),
            json!(job.status().as_str()),
            json!(job.progress as f64),
            json!(now_ms()),
            job.completed_at.map_or(Value::Null, |t| json!(ms(t))),
            json!(ms(job.expires_at)),
            json!(body),
            json!(job.id.to_string()),
            json!(version),
        ],
    ))
}

fn decode_job(row: &Map<String, Value>) -> Option<Job> {
    let s = row.get("job")?.as_str()?;
    match serde_json::from_str::<Job>(s) {
        Ok(j) => Some(j),
        Err(e) => {
            tracing::warn!(error = %e, "D1 job store: undecodable job row skipped");
            None
        }
    }
}

fn version_of(row: &Map<String, Value>) -> i64 {
    row.get("version").and_then(Value::as_f64).unwrap_or(0.0) as i64
}

fn io(e: D1Error) -> StoreError {
    StoreError::Io(e.to_string())
}

/// Fixed identity fields survive any update.
fn keep_identity(next: &mut Job, prev: &Job) {
    next.id = prev.id;
    next.protocol = prev.protocol;
    next.external_id = prev.external_id.clone();
}

/// A change that must reach D1 now (not coalesced).
fn is_state_change(a: &Job, b: &Job) -> bool {
    a.status() != b.status()
        || a.cancel_requested != b.cancel_requested
        || a.artifacts != b.artifacts
        || a.expires_at != b.expires_at
        || a.owner != b.owner
}

impl D1JobStore {
    /// A store over `db`; call [`open`](Self::open) to migrate and start.
    pub fn new(db: D1Client, opts: D1Options) -> Self {
        Self {
            db,
            opts,
            cache: Mutex::new(Cache::default()),
            stats: Mutex::new(D1Stats::default()),
            artifacts: None,
            inputs_root: None,
            last_heartbeat: Mutex::new(None),
        }
    }
    /// Deletes artifacts through `a` when jobs are removed or expire.
    pub fn with_artifacts(mut self, a: Arc<dyn ArtifactStore>) -> Self {
        self.artifacts = Some(a);
        self
    }
    /// Deletes `<root>/<job_id>/` when jobs are removed or expire.
    pub fn with_inputs_root(mut self, root: impl Into<PathBuf>) -> Self {
        self.inputs_root = Some(root.into());
        self
    }

    /// Applies migrations, fails this worker's unfinished jobs left by a
    /// previous run, and starts the flusher (needs a tokio runtime).
    pub async fn open(self, now: OffsetDateTime) -> Result<Arc<Self>, D1Error> {
        schema::migrate(&self.db).await?;
        let me = Arc::new(self);
        let recovered = me.recover(now).await?;
        if recovered > 0 {
            tracing::warn!(jobs = recovered, "D1 job store: unfinished jobs of a previous run marked failed");
        }
        let weak = Arc::downgrade(&me);
        let tick = me.opts.flush_tick;
        tokio::spawn(flusher(weak, tick));
        Ok(me)
    }

    pub fn options(&self) -> &D1Options {
        &self.opts
    }
    /// The D1 client (for tables other than `jobs`).
    pub fn client(&self) -> &D1Client {
        &self.db
    }

    /// Takes over a job another process inserted (a gateway, see
    /// docs/serve/gateway.md): from now on this worker's memory is
    /// authoritative for it and the row's `worker` is this worker. `job` is
    /// the dispatched copy; the row's fields win except `resolved` (whose
    /// input paths are this worker's) and `dispatched_at`. Idempotent for a
    /// job already held here. Refused (`AlreadyExists`) when the row is
    /// finished, cancel was requested, or another worker holds it with a
    /// fresh heartbeat (younger than `stale_after`).
    ///
    /// One D1 round trip: an upsert whose `DO UPDATE` only applies when the
    /// row may be taken over, returning the row it wrote. No row back (zero
    /// changes) means refused; the conditions are checked by SQLite inside
    /// the statement, so two workers racing for a row cannot both win.
    pub async fn adopt(&self, job: Job) -> Result<Job, StoreError> {
        if let Some(e) = self.lock().jobs.get(&job.id) {
            return Ok(e.job.clone());
        }
        let enc = |e: serde_json::Error| StoreError::Io(format!("encoding job: {e}"));
        let resolved = serde_json::to_string(&job.resolved).map_err(enc)?;
        let dispatched = serde_json::to_value(&job).map_err(enc)?.get("dispatched_at").cloned().unwrap_or(Value::Null);
        let stale_ms = self.opts.stale_after.unwrap_or(Duration::from_secs(900)).as_millis() as i64;
        let mut params = row_params(&job, Some(&self.opts.worker_id))?;
        params.extend([json!(resolved), dispatched, json!(self.opts.worker_id), json!(now_ms() - stale_ms)]);
        let stmt = Stmt::new(
            format!(
                "INSERT INTO jobs ({COLS}) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) \
                 ON CONFLICT(id) DO UPDATE SET worker = excluded.worker, updated_at = excluded.updated_at, \
                 version = jobs.version + 1, \
                 job = json_set(jobs.job, '$.resolved', json(?), '$.dispatched_at', ?) \
                 WHERE jobs.status IN ('queued', 'running') \
                 AND COALESCE(json_extract(jobs.job, '$.cancel_requested'), 0) = 0 \
                 AND (jobs.worker IS NULL OR jobs.worker = ? OR jobs.updated_at <= ?) \
                 RETURNING job"
            ),
            params,
        );
        self.count(|s| s.writes += 1);
        let r = match self.db.query(stmt).await {
            Ok(r) => r,
            Err(e) => {
                self.count(|s| s.failed_writes += 1);
                return Err(io(e));
            }
        };
        let Some(next) = r.rows.first().and_then(decode_job) else {
            return Err(StoreError::AlreadyExists(job.id));
        };
        let mut g = self.lock();
        g.seq += 1;
        let seq = g.seq;
        let (tx, _) = watch::channel(next.snapshot(seq));
        g.by_ext.insert((next.protocol, next.external_id.clone()), next.id);
        g.jobs.insert(
            next.id,
            Entry {
                job: next.clone(),
                tx,
                seq,
                flushed: Arc::new(tokio::sync::Mutex::new(seq)),
                last_write: Some(Instant::now()),
                dirty: false,
                terminal_since: None,
            },
        );
        Ok(next)
    }
    pub fn stats(&self) -> D1Stats {
        self.stats.lock().unwrap_or_else(|p| p.into_inner()).clone()
    }
    /// Jobs held in memory.
    pub fn cached(&self) -> usize {
        self.lock().jobs.len()
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Cache> {
        self.cache.lock().unwrap_or_else(|p| p.into_inner())
    }
    fn count(&self, f: impl FnOnce(&mut D1Stats)) {
        f(&mut self.stats.lock().unwrap_or_else(|p| p.into_inner()));
    }

    async fn write(&self, stmt: Stmt) -> Result<u64, D1Error> {
        self.count(|s| s.writes += 1);
        match self.db.query(stmt).await {
            Ok(r) => Ok(r.changes),
            Err(e) => {
                self.count(|s| s.failed_writes += 1);
                Err(e)
            }
        }
    }

    async fn read(&self, stmt: Stmt) -> Result<Vec<Map<String, Value>>, D1Error> {
        self.count(|s| s.reads += 1);
        Ok(self.db.query(stmt).await?.rows)
    }

    async fn fetch_row(&self, id: JobId) -> Result<Option<(Job, i64)>, D1Error> {
        let rows = self
            .read(Stmt::new("SELECT job, version FROM jobs WHERE id = ?", vec![json!(id.to_string())]))
            .await?;
        Ok(rows.first().and_then(|r| Some((decode_job(r)?, version_of(r)))))
    }

    /// Fails this worker's unfinished jobs (restart recovery).
    async fn recover(&self, now: OffsetDateTime) -> Result<usize, D1Error> {
        let rows = self
            .read(Stmt::new(
                format!("SELECT job, version FROM jobs WHERE worker = ? AND status IN {UNFINISHED}"),
                vec![json!(self.opts.worker_id)],
            ))
            .await?;
        let mut n = 0;
        for r in &rows {
            let Some(mut job) = decode_job(r) else { continue };
            if job.recover_after_restart(now) {
                if let Ok(s) = cas_stmt(&job, version_of(r)) {
                    if self.write(s).await? > 0 {
                        n += 1;
                    }
                }
            }
        }
        Ok(n)
    }

    /// Fails other workers' unfinished jobs whose heartbeat is older than
    /// `stale_after`.
    async fn reap_stale(&self, now: OffsetDateTime) -> usize {
        let Some(age) = self.opts.stale_after else { return 0 };
        let cutoff = now_ms() - age.as_millis() as i64;
        let rows = match self
            .read(Stmt::new(
                format!(
                    "SELECT job, version FROM jobs WHERE status IN {UNFINISHED} AND updated_at < ? \
                     AND (worker IS NULL OR worker != ?) LIMIT 100"
                ),
                vec![json!(cutoff), json!(self.opts.worker_id)],
            ))
            .await
        {
            Ok(r) => r,
            Err(e) => {
                tracing::warn!(error = %e, "D1 job store: stale scan failed");
                return 0;
            }
        };
        let mut n = 0;
        for r in &rows {
            let Some(mut job) = decode_job(r) else { continue };
            if job.mark_failed(now, ApiError::internal("the worker running this job was lost")).is_ok() {
                if let Ok(s) = cas_stmt(&job, version_of(r)) {
                    if matches!(self.write(s).await, Ok(c) if c > 0) {
                        n += 1;
                    }
                }
            }
        }
        n
    }

    /// Writes the newest cached version of `id` if it is newer than the last
    /// one written. A failure leaves the entry dirty.
    async fn flush(&self, id: JobId) -> Result<(), D1Error> {
        let Some(lock) = self.lock().jobs.get(&id).map(|e| e.flushed.clone()) else {
            return Ok(());
        };
        let mut done = lock.lock().await;
        let (job, seq) = {
            let g = self.lock();
            match g.jobs.get(&id) {
                Some(e) if e.seq > *done => (e.job.clone(), e.seq),
                _ => return Ok(()),
            }
        };
        let stmt = match upsert_stmt(&job, Some(&self.opts.worker_id)) {
            Ok(s) => s,
            Err(e) => return Err(D1Error::Decode(e.to_string())),
        };
        let res = self.write(stmt).await;
        let mut g = self.lock();
        if let Some(e) = g.jobs.get_mut(&id) {
            e.last_write = Some(Instant::now());
            match &res {
                Ok(_) => e.dirty = e.seq > seq,
                Err(_) => e.dirty = true,
            }
        }
        drop(g);
        if res.is_ok() {
            *done = seq;
        }
        res.map(|_| ())
    }

    /// Writes every dirty cached job now (shutdown).
    pub async fn flush_all(&self) {
        let ids: Vec<JobId> = self.lock().jobs.iter().filter(|(_, e)| e.dirty).map(|(id, _)| *id).collect();
        for id in ids {
            if let Err(e) = self.flush(id).await {
                tracing::warn!(job = %id, error = %e, "D1 job store: flush failed");
            }
        }
    }

    /// One flusher tick: coalesced writes, eviction, heartbeat.
    pub async fn tick(&self) {
        let due: Vec<JobId> = {
            let g = self.lock();
            g.jobs
                .iter()
                .filter(|(_, e)| e.dirty && e.last_write.is_none_or(|t| t.elapsed() >= self.opts.progress_interval))
                .map(|(id, _)| *id)
                .collect()
        };
        for id in due {
            if let Err(e) = self.flush(id).await {
                tracing::warn!(job = %id, error = %e, "D1 job store: deferred write failed; will retry");
            }
        }
        {
            let mut g = self.lock();
            let ttl = self.opts.terminal_ttl;
            let evict: Vec<JobId> = g
                .jobs
                .iter()
                .filter(|(_, e)| !e.dirty && e.terminal_since.is_some_and(|t| t.elapsed() >= ttl))
                .map(|(id, _)| *id)
                .collect();
            for id in evict {
                if let Some(e) = g.jobs.remove(&id) {
                    g.by_ext.remove(&(e.job.protocol, e.job.external_id));
                }
            }
        }
        let beat = {
            let mut hb = self.last_heartbeat.lock().unwrap_or_else(|p| p.into_inner());
            let due = hb.is_none_or(|t| t.elapsed() >= self.opts.heartbeat);
            if due {
                *hb = Some(Instant::now());
            }
            due && self.lock().jobs.values().any(|e| !e.job.is_terminal())
        };
        if beat {
            let s = Stmt::new(
                format!("UPDATE jobs SET updated_at = ? WHERE worker = ? AND status IN {UNFINISHED}"),
                vec![json!(now_ms()), json!(self.opts.worker_id)],
            );
            if let Err(e) = self.write(s).await {
                tracing::warn!(error = %e, "D1 job store: heartbeat failed");
            }
        }
    }

    async fn cleanup(&self, job: &Job) {
        if let Some(a) = &self.artifacts {
            for art in &job.artifacts {
                a.delete(art).await;
            }
        }
        if let Some(root) = &self.inputs_root {
            let _ = tokio::fs::remove_dir_all(root.join(job.id.to_string())).await;
        }
    }

    /// `WHERE` clause and params for the filters of `q` (not paging).
    fn filters(q: &ListQuery, ext_ids: &[String]) -> (String, Vec<Value>) {
        let mut w: Vec<String> = Vec::new();
        let mut p: Vec<Value> = Vec::new();
        if let Some(o) = &q.owner {
            w.push("owner = ?".into());
            p.push(json!(o.0));
        }
        if let Some(pr) = q.protocol {
            w.push("protocol = ?".into());
            p.push(json!(pr.as_str()));
        }
        if !q.statuses.is_empty() {
            w.push(format!("status IN ({})", vec!["?"; q.statuses.len()].join(", ")));
            p.extend(q.statuses.iter().map(|s| json!(s.as_str())));
        }
        if let Some(m) = &q.model {
            w.push("(model = ? OR resolved_model = ?)".into());
            p.push(json!(m));
            p.push(json!(m));
        }
        if let Some(t) = q.task {
            if let Ok(Value::String(s)) = serde_json::to_value(t) {
                w.push("task = ?".into());
                p.push(json!(s));
            }
        }
        if !ext_ids.is_empty() {
            w.push(format!("external_id IN ({})", vec!["?"; ext_ids.len()].join(", ")));
            p.extend(ext_ids.iter().map(|s| json!(s)));
        }
        let clause = if w.is_empty() { "1 = 1".to_owned() } else { w.join(" AND ") };
        (clause, p)
    }

    async fn list_d1(&self, q: &ListQuery) -> Result<Page<Job>, D1Error> {
        // D1 binds at most 100 parameters per statement: long id filters
        // fetch candidates in chunks and page locally.
        if q.external_ids.len() > 60 {
            let mut all: HashMap<JobId, Job> = HashMap::new();
            for chunk in q.external_ids.chunks(60) {
                let (w, p) = Self::filters(q, chunk);
                for r in self.read(Stmt::new(format!("SELECT job FROM jobs WHERE {w}"), p)).await? {
                    if let Some(j) = decode_job(&r) {
                        all.insert(j.id, j);
                    }
                }
            }
            let jobs: Vec<Job> = all.into_values().map(|j| self.fresher(j)).collect();
            return Ok(q.apply(jobs.iter()));
        }
        let (w, p) = Self::filters(q, &q.external_ids);
        let (cmp, dir) = match q.order {
            SortOrder::Desc => ("<", "DESC"),
            SortOrder::Asc => (">", "ASC"),
        };
        let mut items_sql = format!("SELECT job FROM jobs WHERE {w}");
        let mut ip = p.clone();
        if let Some(after) = &q.after {
            // Rows strictly after the cursor row in (created_at, id) order;
            // the cursor must itself pass the filters, else nothing matches
            // (as `ListQuery::apply`). Bare columns in `w` bind to `c` here.
            items_sql.push_str(&format!(
                " AND EXISTS (SELECT 1 FROM jobs c WHERE {w} AND c.external_id = ? \
                 AND (jobs.created_at {cmp} c.created_at OR (jobs.created_at = c.created_at AND jobs.id {cmp} c.id)))"
            ));
            ip.extend(p.iter().cloned());
            ip.push(json!(after));
        }
        items_sql.push_str(&format!(" ORDER BY created_at {dir}, id {dir} LIMIT ? OFFSET ?"));
        // SQLite: LIMIT -1 is "no limit" (callers pass usize::MAX for all).
        ip.push(json!(i64::try_from(q.limit).ok().and_then(|l| l.checked_add(1)).unwrap_or(-1)));
        ip.push(json!(q.offset as i64));
        self.count(|s| s.reads += 1);
        let res = self
            .db
            .batch(vec![
                Stmt::new(format!("SELECT COUNT(*) AS n FROM jobs WHERE {w}"), p),
                Stmt::new(items_sql, ip),
            ])
            .await?;
        let total = res[0]
            .rows
            .first()
            .and_then(|r| r.get("n"))
            .and_then(Value::as_f64)
            .unwrap_or(0.0) as usize;
        let mut items: Vec<Job> = res[1].rows.iter().filter_map(decode_job).map(|j| self.fresher(j)).collect();
        let has_more = items.len() > q.limit;
        items.truncate(q.limit);
        Ok(Page { items, total, has_more })
    }

    /// The cached copy when it is held (fresher progress), else `j`.
    fn fresher(&self, j: Job) -> Job {
        self.lock().jobs.get(&j.id).map(|e| e.job.clone()).unwrap_or(j)
    }
}

async fn flusher(store: Weak<D1JobStore>, tick: Duration) {
    let mut t = tokio::time::interval(tick);
    t.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    loop {
        t.tick().await;
        let Some(s) = store.upgrade() else { return };
        s.tick().await;
    }
}

#[async_trait::async_trait]
impl JobStore for D1JobStore {
    async fn insert(&self, job: Job) -> Result<(), StoreError> {
        {
            let g = self.lock();
            if g.jobs.contains_key(&job.id) {
                return Err(StoreError::AlreadyExists(job.id));
            }
            if g.by_ext.contains_key(&(job.protocol, job.external_id.clone())) {
                return Err(StoreError::DuplicateExternal(job.protocol, job.external_id));
            }
            if let Some(m) = self.opts.max_active {
                if g.jobs.values().filter(|e| !e.job.is_terminal()).count() >= m {
                    return Err(StoreError::Full);
                }
            }
        }
        let hold = self.opts.hold_inserts;
        let stmt = insert_stmt(&job, hold.then_some(self.opts.worker_id.as_str()))?;
        match self.write(stmt).await {
            Ok(_) => {}
            Err(e) if e.is_constraint() => {
                // A retried insert whose first attempt landed looks like a
                // conflict: the row with our id is ours.
                match self.fetch_row(job.id).await {
                    Ok(Some((j, _))) if j.external_id == job.external_id => {}
                    Ok(Some(_)) => return Err(StoreError::AlreadyExists(job.id)),
                    _ => return Err(StoreError::DuplicateExternal(job.protocol, job.external_id)),
                }
            }
            Err(e) => return Err(io(e)),
        }
        if !hold {
            return Ok(());
        }
        let mut g = self.lock();
        g.seq += 1;
        let seq = g.seq;
        let (tx, _) = watch::channel(job.snapshot(seq));
        g.by_ext.insert((job.protocol, job.external_id.clone()), job.id);
        let terminal = job.is_terminal();
        g.jobs.insert(
            job.id,
            Entry {
                job,
                tx,
                seq,
                flushed: Arc::new(tokio::sync::Mutex::new(seq)),
                last_write: Some(Instant::now()),
                dirty: false,
                terminal_since: terminal.then(Instant::now),
            },
        );
        Ok(())
    }

    async fn get(&self, id: JobId) -> Option<Job> {
        if let Some(e) = self.lock().jobs.get(&id) {
            return Some(e.job.clone());
        }
        match self.fetch_row(id).await {
            Ok(r) => r.map(|(j, _)| j),
            Err(e) => {
                tracing::warn!(job = %id, error = %e, "D1 job store: read failed");
                None
            }
        }
    }

    async fn by_external(&self, p: ProtocolId, external_id: &str) -> Option<Job> {
        {
            let g = self.lock();
            if let Some(id) = g.by_ext.get(&(p, external_id.to_owned())) {
                return g.jobs.get(id).map(|e| e.job.clone());
            }
        }
        let r = self
            .read(Stmt::new(
                "SELECT job FROM jobs WHERE protocol = ? AND external_id = ?",
                vec![json!(p.as_str()), json!(external_id)],
            ))
            .await;
        match r {
            Ok(rows) => rows.first().and_then(decode_job),
            Err(e) => {
                tracing::warn!(error = %e, "D1 job store: read failed");
                None
            }
        }
    }

    async fn update(&self, id: JobId, f: JobUpdate) -> Result<Job, StoreError> {
        let mut f = Some(f);
        let cached = {
            let mut g = self.lock();
            g.seq += 1;
            let seq = g.seq;
            g.jobs.get_mut(&id).map(|e| {
                let prev = e.job.clone();
                let mut next = prev.clone();
                (f.take().expect("unused"))(&mut next);
                keep_identity(&mut next, &prev);
                let urgent = is_state_change(&prev, &next);
                if next != prev {
                    e.job = next.clone();
                    e.seq = seq;
                    e.dirty = true;
                    if next.is_terminal() && e.terminal_since.is_none() {
                        e.terminal_since = Some(Instant::now());
                    }
                    e.tx.send_replace(next.snapshot(seq));
                }
                (next, urgent)
            })
        };
        let Some((job, urgent)) = cached else {
            // Not held here: read-modify-write with a version check.
            let f = f.take().expect("unused");
            let (mut job, version) = self
                .fetch_row(id)
                .await
                .map_err(io)?
                .ok_or(StoreError::NotFound(id))?;
            let prev = job.clone();
            f(&mut job);
            keep_identity(&mut job, &prev);
            if job == prev {
                return Ok(job);
            }
            let changed = self.write(cas_stmt(&job, version)?).await.map_err(io)?;
            if changed == 0 {
                return Err(StoreError::Io(format!("job {id} changed concurrently; retry")));
            }
            return Ok(job);
        };
        if urgent {
            if let Err(e) = self.flush(id).await {
                // Memory stays authoritative; the flusher retries.
                tracing::warn!(job = %id, error = %e, "D1 job store: state write failed; will retry");
            }
        }
        Ok(job)
    }

    async fn list(&self, q: ListQuery) -> Page<Job> {
        match self.list_d1(&q).await {
            Ok(p) => p,
            Err(e) => {
                tracing::warn!(error = %e, "D1 job store: list failed; answering from memory");
                let g = self.lock();
                q.apply(g.jobs.values().map(|e| &e.job))
            }
        }
    }

    async fn remove(&self, id: JobId) -> Option<Job> {
        let lock = self.lock().jobs.get(&id).map(|e| e.flushed.clone());
        let _guard = match &lock {
            Some(l) => Some(l.lock().await),
            None => None,
        };
        let cached = {
            let mut g = self.lock();
            let e = g.jobs.remove(&id);
            if let Some(e) = &e {
                g.by_ext.remove(&(e.job.protocol, e.job.external_id.clone()));
            }
            e.map(|e| e.job)
        };
        self.count(|s| s.writes += 1);
        let deleted = match self
            .db
            .query(Stmt::new("DELETE FROM jobs WHERE id = ? RETURNING job", vec![json!(id.to_string())]))
            .await
        {
            Ok(r) => r.rows.first().and_then(decode_job),
            Err(e) => {
                self.count(|s| s.failed_writes += 1);
                tracing::warn!(job = %id, error = %e, "D1 job store: delete failed");
                None
            }
        };
        let job = cached.or(deleted)?;
        self.cleanup(&job).await;
        Some(job)
    }

    fn watch(&self, id: JobId) -> Option<watch::Receiver<JobSnapshot>> {
        self.lock().jobs.get(&id).map(|e| e.tx.subscribe())
    }

    async fn sweep_expired(&self, now: OffsetDateTime) -> usize {
        let mut gone: HashMap<JobId, Job> = HashMap::new();
        {
            let mut g = self.lock();
            let ids: Vec<JobId> = g.jobs.values().filter(|e| e.job.is_expired(now)).map(|e| e.job.id).collect();
            for id in ids {
                if let Some(e) = g.jobs.remove(&id) {
                    g.by_ext.remove(&(e.job.protocol, e.job.external_id.clone()));
                    gone.insert(id, e.job);
                }
            }
        }
        self.count(|s| s.writes += 1);
        match self
            .db
            .query(Stmt::new("DELETE FROM jobs WHERE expires_at <= ? RETURNING job", vec![json!(ms(now))]))
            .await
        {
            Ok(r) => {
                for j in r.rows.iter().filter_map(decode_job) {
                    gone.entry(j.id).or_insert(j);
                }
            }
            Err(e) => {
                self.count(|s| s.failed_writes += 1);
                tracing::warn!(error = %e, "D1 job store: expiry sweep failed");
            }
        }
        let mut seen = HashSet::new();
        for j in gone.values() {
            if seen.insert(j.id) {
                self.cleanup(j).await;
            }
        }
        let stale = self.reap_stale(now).await;
        if stale > 0 {
            tracing::warn!(jobs = stale, "D1 job store: jobs of lost workers marked failed");
        }
        gone.len()
    }
}
