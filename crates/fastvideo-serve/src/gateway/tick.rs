//! The gateway tick (`gateway.tick_s`, docs/serve/gateway.md §3-§4, §7):
//!
//! 1. **Probes**: every pod worker (static `urls` plus `gw_workers`
//!    registrations of the last 45 s) answers `GET /fv/v1/internal/status`
//!    (health, load, sessions, draining, caps); serverless pools answer
//!    Runpod `/health`. Live caps refresh every `caps_refresh_s`; the
//!    catalog is rebuilt from them.
//! 2. **Reaper** over the `active` `gw_dispatch` rows joined with their job
//!    rows: finished jobs close the row (with durations for the metrics);
//!    jobs whose worker heartbeat is older than the pool's `stale_after_s`,
//!    or whose Runpod job ended while the row is unfinished, are lost and
//!    re-dispatched (`retries`) or failed. Queued serverless jobs still in
//!    Runpod's queue get their heartbeat touched.
//! 3. **Metrics** per pool ([`PoolMetrics`]), Prometheus gauges, and the
//!    [`PoolScaler`](super::scale::PoolScaler) hooks.
//!
//! Every replica ticks; row claims (`attempt`/`state` guards) keep the
//! reaper's actions single.

use std::collections::BTreeMap;
use std::time::{Duration, Instant};

use fastvideo_engine_service::Recipe;
use fastvideo_protocol::{JobId, JobStore, ModelCaps};
use fastvideo_serve_kit::d1::Stmt;
use serde_json::{json, Value};

use super::dispatch::DispatchRow;
use super::scale::{DurationStats, PoolMetrics, WorkerCounts};
use super::schema::now_ms;
use super::{Gateway, Pool, WorkerView};

/// A worker's `GET /fv/v1/internal/status` body, parsed.
#[derive(Clone, Debug, Default)]
pub struct WorkerStatus {
    pub id: Option<String>,
    pub ready: bool,
    pub draining: bool,
    pub running: u32,
    pub queued: u32,
    pub sessions: u32,
    pub caps: Vec<(ModelCaps, Recipe)>,
    pub build: Option<super::WorkerBuild>,
    /// `readiness: failed` (a model failed to load or cannot run on its GPU).
    pub failed: bool,
    /// Its failed models and why.
    pub failed_models: BTreeMap<String, String>,
}

impl WorkerStatus {
    pub fn parse(v: &Value) -> Self {
        let n = |p: &str| v.pointer(p).and_then(Value::as_u64).unwrap_or(0) as u32;
        let caps = v
            .get("models")
            .and_then(Value::as_array)
            .map(|a| {
                a.iter()
                    .filter_map(|e| {
                        let c: ModelCaps = serde_json::from_value(e.get("caps")?.clone()).ok()?;
                        let r: Recipe = e.get("recipe").cloned().and_then(|r| serde_json::from_value(r).ok()).unwrap_or_default();
                        Some((c, r))
                    })
                    .collect()
            })
            .unwrap_or_default();
        Self {
            id: v.get("worker_id").and_then(Value::as_str).map(str::to_owned),
            ready: v.get("readiness").and_then(Value::as_str) == Some("ready"),
            draining: v.get("draining").and_then(Value::as_bool).unwrap_or(false),
            running: n("/stats/running"),
            queued: n("/stats/queued_batch") + n("/stats/queued_stream"),
            sessions: n("/stats/sessions"),
            caps,
            build: super::WorkerBuild::parse(v),
            failed: v.get("readiness").and_then(Value::as_str) == Some("failed"),
            failed_models: v
                .get("failed_models")
                .and_then(Value::as_object)
                .map(|o| o.iter().filter_map(|(k, v)| v.as_str().map(|s| (k.clone(), s.chars().take(400).collect()))).collect())
                .unwrap_or_default(),
        }
    }
}

/// A failed worker probe.
#[derive(Debug)]
struct ProbeErr {
    msg: String,
    /// An HTTP answer (not a connection error).
    answered: bool,
    /// fv-serve still starting (503 `{"error":{"kind":"loading"}}`).
    starting: bool,
}

impl ProbeErr {
    fn net(msg: String) -> Self {
        Self { msg, answered: false, starting: false }
    }
}

/// A job's row as the reaper sees it.
#[derive(Debug)]
struct Active {
    row: DispatchRow,
    status: Option<String>,
    job_updated: i64,
    job_created: i64,
    worker: Option<String>,
}

const REGISTERED_FRESH_MS: i64 = 45_000;

impl Gateway {
    /// One tick (see the module docs).
    pub async fn tick(&self) {
        self.probe_pools().await;
        self.rebuild_catalog();
        let active = self.active_rows().await;
        let mut counts: BTreeMap<String, (u32, u32, i64)> = BTreeMap::new();
        let now = now_ms();
        for a in active {
            match a.status.as_deref() {
                None => {
                    if self.close_row(&a.row.job_id, None).await {
                        self.drop_inputs(&a.row.inputs).await;
                    }
                }
                Some("queued") | Some("running") => {
                    let e = counts.entry(a.row.pool.clone()).or_insert((0, 0, i64::MAX));
                    if a.status.as_deref() == Some("queued") {
                        e.0 += 1;
                        e.2 = e.2.min(a.job_created);
                    } else {
                        e.1 += 1;
                    }
                    self.check_alive(&a, now).await;
                }
                Some(_) => {
                    let id = a.row.job_id.parse::<JobId>().ok();
                    let job = match id {
                        Some(id) => self.jobs.get(id).await,
                        None => None,
                    };
                    // The store copies of its inputs go with the row (the
                    // worker deletes those it was sent; the background
                    // copies for a re-dispatch are only in the row).
                    if self.close_row(&a.row.job_id, job.as_ref()).await {
                        self.drop_inputs(&a.row.inputs).await;
                    }
                }
            }
        }
        self.expire_leases().await;
        let metrics = self.compute_metrics(&counts, now).await;
        for s in self.scalers() {
            s.observe(&metrics).await;
        }
    }

    /// Probes every pool (concurrently).
    async fn probe_pools(&self) {
        let registered = self.registered_workers().await;
        let futs = self.pools.iter().map(|p| {
            let reg = registered.get(p.id()).cloned().unwrap_or_default();
            async move {
                if p.is_pod() {
                    self.probe_pods(p, reg).await;
                } else {
                    self.probe_serverless(p).await;
                }
            }
        });
        futures::future::join_all(futs).await;
    }

    /// `gw_workers` rows seen in the last 45 s, by pool: (url, state).
    async fn registered_workers(&self) -> BTreeMap<String, Vec<(String, String)>> {
        let mut out: BTreeMap<String, Vec<(String, String)>> = BTreeMap::new();
        let r = self
            .db
            .query(Stmt::new(
                "SELECT pool, url, state FROM gw_workers WHERE updated_at > ?",
                vec![json!(now_ms() - REGISTERED_FRESH_MS)],
            ))
            .await;
        if let Ok(r) = r {
            for row in r.rows {
                let s = |k: &str| row.get(k).and_then(Value::as_str).unwrap_or_default().to_owned();
                out.entry(s("pool")).or_default().push((s("url").trim_end_matches('/').to_owned(), s("state")));
            }
        }
        out
    }

    async fn probe_pods(&self, p: &Pool, registered: Vec<(String, String)>) {
        let static_urls: Vec<String> = p.cfg.urls.iter().map(|u| u.trim_end_matches('/').to_owned()).collect();
        let mut urls = static_urls.clone();
        for (u, _) in &registered {
            if !urls.contains(u) {
                urls.push(u.clone());
            }
        }
        let probes = urls.iter().map(|u| async move {
            let r = self
                .worker_req(reqwest::Method::GET, &format!("{u}/fv/v1/internal/status"))
                .timeout(Duration::from_secs(4))
                .send()
                .await;
            let res = match r {
                Ok(resp) if resp.status().is_success() => {
                    resp.json::<Value>().await.map(|v| WorkerStatus::parse(&v)).map_err(|e| ProbeErr::net(e.without_url().to_string()))
                }
                Ok(resp) => {
                    let code = resp.status().as_u16();
                    let body = resp.json::<Value>().await.unwrap_or(Value::Null);
                    let starting = code == 503 && body.pointer("/error/kind").and_then(Value::as_str) == Some("loading");
                    Err(ProbeErr { msg: format!("status probe answered {code}"), answered: true, starting })
                }
                Err(e) => Err(ProbeErr::net(e.without_url().to_string())),
            };
            (u.clone(), res)
        });
        let results = futures::future::join_all(probes).await;
        let refresh = Duration::from_secs(self.cfg.caps_refresh_s.max(1));
        let mut st = p.lock();
        let mut next: BTreeMap<String, WorkerView> = BTreeMap::new();
        for (url, res) in results {
            let reg_state = registered.iter().find(|(u, _)| *u == url).map(|(_, s)| s.clone());
            let last_ok = st.workers.get(&url).and_then(|w| w.last_ok);
            let mut w = WorkerView { url: url.clone(), registered: reg_state.is_some(), last_ok, ..WorkerView::default() };
            match res {
                Ok(s) => {
                    w.healthy = true;
                    w.last_ok = Some(Instant::now());
                    w.ready = s.ready;
                    w.draining = s.draining || reg_state.as_deref() == Some("draining");
                    w.running = s.running;
                    w.queued = s.queued;
                    w.sessions = s.sessions;
                    w.id = s.id.clone();
                    w.build = s.build.clone();
                    w.failed = s.failed;
                    w.failed_models = s.failed_models.clone();
                    if s.ready && !s.caps.is_empty() && st.caps_at.is_none_or(|t| t.elapsed() >= refresh) {
                        st.live_caps = Some(s.caps);
                        st.caps_at = Some(Instant::now());
                    }
                }
                Err(e) => {
                    w.healthy = false;
                    w.answered = e.answered;
                    w.starting = e.starting;
                    w.last_error = Some(e.msg);
                }
            }
            next.insert(url, w);
        }
        st.available = next.values().any(|w| w.healthy && !w.draining);
        st.last_error = if st.available {
            None
        } else if next.is_empty() {
            Some("no worker is configured or registered".into())
        } else {
            next.values().find_map(|w| w.last_error.clone()).or_else(|| Some("every worker is draining".into()))
        };
        st.workers = next;
    }

    async fn probe_serverless(&self, p: &Pool) {
        let ep = p.cfg.endpoint_id.clone().unwrap_or_default();
        let r = self.runpod.health(&ep).await;
        let mut st = p.lock();
        match r {
            Ok(v) => {
                st.available = true;
                st.last_error = None;
                st.health = Some(v);
                st.health_at = Some(Instant::now());
            }
            Err(e) => {
                st.available = false;
                st.last_error = Some(e);
            }
        }
    }

    /// The `active` dispatch rows with their job's state.
    async fn active_rows(&self) -> Vec<Active> {
        let r = self
            .db
            .query(Stmt::raw(
                "SELECT d.job_id AS job_id, d.pool AS pool, d.kind AS kind, d.target AS target, d.ref AS ref, \
                 d.attempt AS attempt, d.inputs AS inputs, j.status AS status, j.updated_at AS j_updated, \
                 j.created_at AS j_created, j.worker AS worker \
                 FROM gw_dispatch d LEFT JOIN jobs j ON j.id = d.job_id WHERE d.state = 'active' LIMIT 1000",
            ))
            .await;
        let rows = match r {
            Ok(r) => r.rows,
            Err(e) => {
                tracing::warn!(error = %e, "gateway: reading the dispatch table failed");
                return Vec::new();
            }
        };
        rows.iter()
            .map(|r| Active {
                row: DispatchRow::from_row(r),
                status: r.get("status").and_then(Value::as_str).map(str::to_owned),
                job_updated: r.get("j_updated").and_then(Value::as_f64).unwrap_or(0.0) as i64,
                job_created: r.get("j_created").and_then(Value::as_f64).unwrap_or(0.0) as i64,
                worker: r.get("worker").and_then(Value::as_str).map(str::to_owned),
            })
            .collect()
    }

    /// Closes a dispatch row (job finished or gone), keeping durations;
    /// `true` when this call closed it.
    async fn close_row(&self, job_id: &str, job: Option<&fastvideo_protocol::Job>) -> bool {
        let secs = |a: time::OffsetDateTime, b: time::OffsetDateTime| (b - a).as_seconds_f64();
        let (run_s, wait_s) = match job {
            Some(j) => (
                j.started_at.zip(j.completed_at).map(|(s, c)| secs(s, c)),
                j.started_at.map(|s| secs(j.created_at, s)),
            ),
            None => (None, None),
        };
        let now = now_ms();
        let r = self
            .db
            .query(Stmt::new(
                "UPDATE gw_dispatch SET state = 'done', updated_at = ?, finished_at = ?, run_s = ?, wait_s = ? WHERE job_id = ? AND state = 'active'",
                vec![json!(now), json!(now), run_s.map_or(Value::Null, |v| json!(v)), wait_s.map_or(Value::Null, |v| json!(v)), json!(job_id)],
            ))
            .await;
        match r {
            Ok(r) => r.changes > 0,
            Err(e) => {
                tracing::warn!(job = %job_id, error = %e, "gateway: closing a dispatch row failed");
                false
            }
        }
    }

    /// Worker-loss checks for one unfinished job.
    async fn check_alive(&self, a: &Active, now: i64) {
        let Some(pool) = self.pool(&a.row.pool) else { return };
        let stale_ms = (pool.cfg.stale_after_s.max(1) * 1000) as i64;
        let age = now - a.job_updated;
        if a.row.kind == "pod" {
            if age > stale_ms {
                self.lost(pool, a, &format!("no heartbeat for {} s", age / 1000)).await;
            }
            return;
        }
        // Serverless: ask Runpod when the row is quiet (not adopted yet, or stale).
        let unadopted = a.worker.is_none() && a.status.as_deref() == Some("queued");
        if !(age > stale_ms || (unadopted && age > stale_ms / 3)) {
            return;
        }
        let Some(r) = &a.row.r#ref else { return };
        match self.runpod.status(&a.row.target, r).await {
            Ok(Some(s)) if s.is_alive() => {
                if unadopted || s.status == "IN_QUEUE" {
                    // Still in Runpod's queue (or starting): heartbeat on its behalf.
                    let _ = self
                        .db
                        .query(Stmt::new(
                            "UPDATE jobs SET updated_at = ? WHERE id = ? AND status IN ('queued', 'running')",
                            vec![json!(now), json!(a.row.job_id)],
                        ))
                        .await;
                } else if age > 2 * stale_ms {
                    self.lost(pool, a, &format!("Runpod job {} is {} but the worker wrote nothing for {} s", r, s.status, age / 1000)).await;
                }
            }
            Ok(Some(s)) => {
                // Re-read: the worker may have finished just now.
                if let Ok(id) = a.row.job_id.parse::<JobId>() {
                    if self.jobs.get(id).await.is_some_and(|j| j.is_terminal()) {
                        return;
                    }
                }
                let err = s.error.map(|e| e.to_string()).unwrap_or_default();
                self.lost(pool, a, &format!("Runpod job {r} ended {} while the job was unfinished {err}", s.status)).await;
            }
            Ok(None) => self.lost(pool, a, &format!("Runpod does not know job {r}")).await,
            Err(e) => tracing::debug!(job = %a.row.job_id, error = %e, "gateway: Runpod status failed (retrying next tick)"),
        }
    }

    async fn lost(&self, pool: &Pool, a: &Active, why: &str) {
        if a.row.attempt <= pool.cfg.retries {
            match self.redispatch(&a.row, &format!("worker lost: {why}")).await {
                Ok(_) => {}
                Err(e) => tracing::warn!(job = %a.row.job_id, error = %e.message, "gateway: re-dispatch failed"),
            }
        } else {
            self.fail_lost(&a.row, why).await;
        }
    }

    /// Ends leases whose worker is gone or which saw no activity.
    async fn expire_leases(&self) {
        // Director leases need heartbeats (15 s on the worker); others end
        // on stop. Anything idle for 10 min is closed.
        let cutoff_director = now_ms() - 60_000;
        let cutoff_any = now_ms() - 600_000;
        let _ = self
            .db
            .query(Stmt::new(
                "UPDATE gw_sessions SET state = 'ended' WHERE state = 'live' AND ((kind = 'director' AND updated_at < ?) OR updated_at < ?)",
                vec![json!(cutoff_director), json!(cutoff_any)],
            ))
            .await;
        let mut g = self.leases.lock().unwrap_or_else(|p| p.into_inner());
        g.retain(|_, l| l.created_at > cutoff_any);
    }

    async fn compute_metrics(&self, counts: &BTreeMap<String, (u32, u32, i64)>, now: i64) -> Vec<PoolMetrics> {
        let window = self.cfg.metrics_window_s.max(1);
        let mut run: BTreeMap<String, Vec<f64>> = BTreeMap::new();
        let mut wait: BTreeMap<String, Vec<f64>> = BTreeMap::new();
        if let Ok(r) = self
            .db
            .query(Stmt::new(
                "SELECT pool, run_s, wait_s FROM gw_dispatch WHERE state = 'done' AND finished_at > ?",
                vec![json!(now - (window * 1000) as i64)],
            ))
            .await
        {
            for row in r.rows {
                let pool = row.get("pool").and_then(Value::as_str).unwrap_or_default().to_owned();
                if let Some(v) = row.get("run_s").and_then(Value::as_f64) {
                    run.entry(pool.clone()).or_default().push(v);
                }
                if let Some(v) = row.get("wait_s").and_then(Value::as_f64) {
                    wait.entry(pool).or_default().push(v);
                }
            }
        }
        let mut submitted: BTreeMap<String, u64> = BTreeMap::new();
        if let Ok(r) = self.db.query(Stmt::raw("SELECT pool, COUNT(*) AS n FROM gw_dispatch GROUP BY pool")).await {
            for row in r.rows {
                let pool = row.get("pool").and_then(Value::as_str).unwrap_or_default().to_owned();
                submitted.insert(pool, row.get("n").and_then(Value::as_f64).unwrap_or(0.0) as u64);
            }
        }
        let mut streams: BTreeMap<String, u32> = BTreeMap::new();
        if let Ok(r) = self.db.query(Stmt::raw("SELECT pool, COUNT(*) AS n FROM gw_sessions WHERE state = 'live' GROUP BY pool")).await {
            for row in r.rows {
                let pool = row.get("pool").and_then(Value::as_str).unwrap_or_default().to_owned();
                streams.insert(pool, row.get("n").and_then(Value::as_f64).unwrap_or(0.0) as u32);
            }
        }
        let mut out = Vec::new();
        for p in &self.pools {
            let (queued, running, oldest) = counts.get(p.id()).copied().unwrap_or((0, 0, i64::MAX));
            let available = p.available();
            let mut st = p.lock();
            let workers = if p.is_pod() { pod_counts(st.workers.values()) } else { runpod_counts(st.health.as_ref()) };
            let m = PoolMetrics {
                pool: p.id().to_owned(),
                kind: p.cfg.kind,
                endpoint_id: p.cfg.endpoint_id.clone(),
                at_unix_ms: now,
                queued,
                running,
                oldest_queued_age_s: if oldest == i64::MAX { 0.0 } else { ((now - oldest).max(0) as f64) / 1000.0 },
                streams: streams.get(p.id()).copied().unwrap_or(0),
                run_time: DurationStats::of(run.get(p.id()).map(Vec::as_slice).unwrap_or(&[])),
                queue_wait: DurationStats::of(wait.get(p.id()).map(Vec::as_slice).unwrap_or(&[])),
                window_s: window,
                workers,
                available,
                max_queued: p.cfg.max_queued,
                max_streams: p.cfg.max_streams,
                submitted_total: submitted.get(p.id()).copied().unwrap_or(0),
            };
            st.queued = queued;
            st.running = running;
            st.streams = m.streams;
            st.pending = 0;
            for w in st.workers.values_mut() {
                w.inflight = 0;
            }
            st.metrics = Some(m.clone());
            let pool = p.id().to_owned();
            metrics::gauge!("fv_pool_queued", "pool" => pool.clone()).set(m.queued as f64);
            metrics::gauge!("fv_pool_running", "pool" => pool.clone()).set(m.running as f64);
            metrics::gauge!("fv_pool_oldest_queued_seconds", "pool" => pool.clone()).set(m.oldest_queued_age_s);
            metrics::gauge!("fv_pool_streams", "pool" => pool.clone()).set(m.streams as f64);
            metrics::gauge!("fv_pool_available", "pool" => pool.clone()).set(if m.available { 1.0 } else { 0.0 });
            metrics::gauge!("fv_pool_submitted_total", "pool" => pool.clone()).set(m.submitted_total as f64);
            for (state, n) in [
                ("total", m.workers.total),
                ("ready", m.workers.ready),
                ("busy", m.workers.busy),
                ("idle", m.workers.idle),
                ("initializing", m.workers.initializing),
                ("unhealthy", m.workers.unhealthy),
            ] {
                metrics::gauge!("fv_pool_workers", "pool" => pool.clone(), "state" => state).set(n as f64);
            }
            out.push(m);
        }
        out
    }
}

fn pod_counts<'a>(ws: impl Iterator<Item = &'a WorkerView>) -> WorkerCounts {
    let mut c = WorkerCounts::default();
    for w in ws {
        c.total += 1;
        if !w.healthy {
            c.unhealthy += 1;
        } else if !w.ready {
            c.initializing += 1;
        } else {
            c.ready += 1;
            if w.running + w.queued + w.sessions > 0 {
                c.busy += 1;
            } else {
                c.idle += 1;
            }
        }
    }
    c
}

/// Runpod `/health` `workers`: `idle`, `initializing`, `ready`, `running`,
/// `throttled`, `unhealthy`.
fn runpod_counts(h: Option<&Value>) -> WorkerCounts {
    let Some(w) = h.and_then(|h| h.get("workers")) else { return WorkerCounts::default() };
    let n = |k: &str| w.get(k).and_then(Value::as_u64).unwrap_or(0) as u32;
    let (idle, running, init, unhealthy, throttled) = (n("idle"), n("running"), n("initializing"), n("unhealthy"), n("throttled"));
    WorkerCounts {
        total: idle + running + init + unhealthy + throttled,
        ready: n("ready").max(idle + running),
        busy: running,
        idle,
        initializing: init,
        unhealthy,
    }
}
