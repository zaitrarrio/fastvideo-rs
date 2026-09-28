//! Job dispatch (docs/serve/gateway.md §3): pool choice, inputs, the
//! envelope, the `gw_dispatch` record, cancel, and the re-dispatch / fail
//! steps of the worker-loss reaper.
//!
//! Envelope (**native**), the body of the worker's
//! `POST /fv/v1/internal/jobs`:
//! `{"job": <Job>, "inputs": [{"path","url","artifact"}], "attempt": n, "pool": id}`.
//! Serverless pools get it inside the queue envelope
//! `{"kind":"http","method":"POST","path":"/fv/v1/internal/jobs","body":…,
//! "wait":true,"poll_path":"/fv/v1/internal/jobs/<id>","cancel_path":…}`.

use std::path::PathBuf;
use std::time::Duration;

use fastvideo_protocol::{ApiError, Artifact, Job, JobId, JobState, LogLine};
use fastvideo_serve_kit::artifacts::valid_file_name;
use fastvideo_serve_kit::d1::Stmt;
use fastvideo_serve_kit::ArtifactMeta;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use super::schema::now_ms;
use super::{Gateway, Pool};

/// One staged input shipped to the worker.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct InputRef {
    /// The path in the dispatched job's `resolved` (the gateway's).
    pub path: PathBuf,
    /// Signed URL the worker downloads it from.
    pub url: String,
    /// The artifact holding it (the worker deletes it when the job ends).
    #[serde(default)]
    pub artifact: Option<Artifact>,
}

/// The dispatch envelope (see the module docs).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Envelope {
    pub job: Job,
    #[serde(default)]
    pub inputs: Vec<InputRef>,
    #[serde(default = "one")]
    pub attempt: u32,
    #[serde(default)]
    pub pool: Option<String>,
}

fn one() -> u32 {
    1
}

/// Where a dispatch went.
#[derive(Clone, Debug, PartialEq)]
pub struct Placed {
    /// Pod URL or endpoint id.
    pub target: String,
    /// Runpod job id or worker id.
    pub r#ref: Option<String>,
}

/// The internal paths of a job on a worker.
pub fn job_path(id: JobId) -> String {
    format!("/fv/v1/internal/jobs/{id}")
}

impl Gateway {
    /// `EngineGate::submit`: choose a pool, ship inputs, dispatch, record.
    pub async fn submit_job(&self, job: &Job) -> Result<(), ApiError> {
        let cat = self.catalog();
        let pools = cat.pools_of.get(&job.resolved.model).cloned().unwrap_or_default();
        if pools.is_empty() {
            return Err(ApiError::not_found(format!("model `{}` is not served by any pool", job.resolved.model)));
        }
        let mut order: Vec<&Pool> = pools.iter().filter_map(|i| self.pools.get(*i)).collect();
        // Available pools first, then fewest queued.
        order.sort_by_key(|p| (!p.available(), { let s = p.lock(); s.queued + s.pending }));
        let inputs = self.stage_inputs(job).await?;
        let mut last: Option<ApiError> = None;
        for p in order {
            if !p.available() {
                last.get_or_insert_with(|| unavailable(p));
                continue;
            }
            let max = p.cfg.max_queued;
            let queued = {
                let s = p.lock();
                s.queued + s.pending
            };
            if max > 0 && queued >= max {
                last = Some(ApiError::queue_full(format!("pool `{}` has {queued} queued jobs (max {max})", p.id())).with_retry_after(5));
                continue;
            }
            match self.dispatch_to(p, job, &inputs, 1, None).await {
                Ok(placed) => {
                    p.lock().pending += 1;
                    if let Err(e) = self.record(p, job.id, &placed, 1, &inputs).await {
                        tracing::warn!(job = %job.id, error = %e, "gateway: recording the dispatch failed (the job runs; reaping and metrics miss it)");
                    }
                    metrics::counter!("fv_gateway_dispatched_total", "pool" => p.id().to_owned()).increment(1);
                    return Ok(());
                }
                Err(e) => {
                    tracing::warn!(job = %job.id, pool = p.id(), error = %e.message, "gateway: dispatch failed");
                    last = Some(e);
                }
            }
        }
        self.drop_inputs(&inputs).await;
        Err(last.unwrap_or_else(|| ApiError::loading("no pool can take the job").with_retry_after(10)))
    }

    /// Puts the job's staged inputs into the artifact store (shared: R2) and
    /// signs URLs the worker can fetch.
    async fn stage_inputs(&self, job: &Job) -> Result<Vec<InputRef>, ApiError> {
        let r = &job.resolved;
        let mut paths: Vec<PathBuf> = r.keyframes.iter().map(|(_, p)| p.clone()).collect();
        paths.extend(r.references.iter().map(|(_, p)| p.clone()));
        paths.extend(r.audio_in.iter().map(|(_, p)| p.clone()));
        if paths.is_empty() {
            return Ok(Vec::new());
        }
        let ctx = self.ctx()?;
        let mut out = Vec::new();
        for (i, path) in paths.into_iter().enumerate() {
            let name = path.file_name().and_then(|n| n.to_str()).filter(|n| valid_file_name(n)).map(str::to_owned).unwrap_or_else(|| format!("input-{i}.bin"));
            // The store moves the file: keep the gateway copy (a re-dispatch
            // after a store failure still has it) by copying first.
            let tmp = path.with_extension(format!("gw{i}.tmp"));
            tokio::fs::copy(&path, &tmp).await.map_err(|e| ApiError::internal(format!("staging input for dispatch: {e}")))?;
            let meta = ArtifactMeta { file_name: name, mime: "application/octet-stream".into(), ..ArtifactMeta::default() };
            let art = ctx.artifacts().put(&tmp, meta).await?;
            let url = ctx.url_signer().url_for(&art, self.input_url_ttl).to_string();
            out.push(InputRef { path, url, artifact: Some(art) });
        }
        Ok(out)
    }

    async fn drop_inputs(&self, inputs: &[InputRef]) {
        let Ok(ctx) = self.ctx() else { return };
        for i in inputs {
            if let Some(a) = &i.artifact {
                ctx.artifacts().delete(a).await;
            }
        }
    }

    /// Sends the envelope to `pool` (`exclude`: a pod URL not to use).
    pub(crate) async fn dispatch_to(&self, pool: &Pool, job: &Job, inputs: &[InputRef], attempt: u32, exclude: Option<&str>) -> Result<Placed, ApiError> {
        let env = Envelope { job: job.clone(), inputs: inputs.to_vec(), attempt, pool: Some(pool.id().to_owned()) };
        let timeout = Duration::from_secs(pool.cfg.dispatch_timeout_s.max(1));
        if pool.is_pod() {
            let mut cands: Vec<(String, u32)> = {
                let st = pool.lock();
                st.workers.values().filter(|w| w.usable() && Some(w.url.as_str()) != exclude).map(|w| (w.url.clone(), w.load())).collect()
            };
            cands.sort_by_key(|(_, l)| *l);
            let mut last = None;
            for (url, _) in cands {
                let r = self
                    .worker_req(reqwest::Method::POST, &format!("{url}/fv/v1/internal/jobs"))
                    .timeout(timeout)
                    .json(&env)
                    .send()
                    .await;
                match r {
                    Ok(resp) if resp.status().is_success() => {
                        let v: Value = resp.json().await.unwrap_or(Value::Null);
                        if let Some(w) = pool.lock().workers.get_mut(&url) {
                            w.inflight += 1;
                        }
                        let worker = v.get("worker").and_then(Value::as_str).map(str::to_owned);
                        return Ok(Placed { target: url, r#ref: worker });
                    }
                    Ok(resp) => {
                        let s = resp.status().as_u16();
                        let v: Value = resp.json().await.unwrap_or(Value::Null);
                        let msg = v.pointer("/error/message").and_then(Value::as_str).unwrap_or_default().to_owned();
                        tracing::info!(job = %job.id, worker = %url, status = s, %msg, "gateway: worker refused the job");
                        if s == 409 {
                            // Already held by a worker (a duplicate delivery).
                            return Err(ApiError::conflict(format!("the job is already running on a worker: {msg}")));
                        }
                        if (400..500).contains(&s) && s != 429 {
                            // The job itself is bad for the worker: no point trying others.
                            let kind = v.pointer("/error/kind").cloned().and_then(|k| serde_json::from_value(k).ok());
                            return Err(match kind {
                                Some(k) => ApiError::new(k, msg),
                                None => ApiError::invalid(msg),
                            });
                        }
                        last = Some(format!("worker {url} answered {s}: {msg}"));
                    }
                    Err(e) => {
                        let msg = e.without_url().to_string();
                        if let Some(w) = pool.lock().workers.get_mut(&url) {
                            w.healthy = false;
                            w.last_error = Some(msg.clone());
                        }
                        last = Some(format!("worker {url}: {msg}"));
                    }
                }
            }
            return Err(ApiError::loading(format!(
                "pool `{}` has no worker that can take the job now{}",
                pool.id(),
                last.map(|l| format!(" ({l})")).unwrap_or_default()
            ))
            .with_retry_after(10));
        }
        let ep = pool.cfg.endpoint_id.clone().unwrap_or_default();
        let path = job_path(job.id);
        let input = json!({
            "kind": "http",
            "method": "POST",
            "path": "/fv/v1/internal/jobs",
            "body": env,
            "wait": true,
            "poll_path": path,
            "cancel_path": path,
            "poll_interval_ms": 1000,
            "timeout_s": pool.cfg.job_timeout_s,
        });
        match self.runpod.run(&ep, &input, timeout).await {
            Ok(id) => Ok(Placed { target: ep, r#ref: Some(id) }),
            Err(e) => {
                let mut st = pool.lock();
                st.last_error = Some(e.clone());
                Err(ApiError::loading(format!("pool `{}` (Runpod endpoint) did not take the job: {e}", pool.id())).with_retry_after(10))
            }
        }
    }

    /// Writes (or replaces) the `gw_dispatch` row.
    async fn record(&self, pool: &Pool, id: JobId, placed: &Placed, attempt: u32, inputs: &[InputRef]) -> Result<(), String> {
        let now = now_ms();
        let kind = if pool.is_pod() { "pod" } else { "runpod-serverless" };
        let inputs = serde_json::to_string(inputs).unwrap_or_else(|_| "[]".into());
        self.db
            .query(Stmt::new(
                "INSERT INTO gw_dispatch (job_id, pool, kind, target, ref, attempt, state, inputs, created_at, updated_at) \
                 VALUES (?, ?, ?, ?, ?, ?, 'active', ?, ?, ?) \
                 ON CONFLICT(job_id) DO UPDATE SET pool = excluded.pool, kind = excluded.kind, target = excluded.target, \
                 ref = excluded.ref, attempt = excluded.attempt, state = 'active', updated_at = excluded.updated_at",
                vec![
                    json!(id.to_string()),
                    json!(pool.id()),
                    json!(kind),
                    json!(placed.target),
                    placed.r#ref.as_ref().map_or(Value::Null, |r| json!(r)),
                    json!(attempt),
                    json!(inputs),
                    json!(now),
                    json!(now),
                ],
            ))
            .await
            .map(|_| ())
            .map_err(|e| e.to_string())
    }

    /// `EngineGate::cancel`: forwards to the job's worker or Runpod job.
    pub async fn cancel_job(&self, id: JobId) -> bool {
        let row = match self.dispatch_row(id).await {
            Some(r) => r,
            None => return false,
        };
        let pool = self.pool(&row.pool);
        if row.kind == "pod" {
            let r = self.worker_req(reqwest::Method::DELETE, &format!("{}{}", row.target, job_path(id))).timeout(Duration::from_secs(15)).send().await;
            if let Err(e) = r {
                tracing::warn!(job = %id, error = %e.without_url(), "gateway: forwarding cancel to the worker failed");
            }
        } else if let Some(r) = &row.r#ref {
            if let Err(e) = self.runpod.cancel(&row.target, r).await {
                tracing::warn!(job = %id, error = %e, "gateway: Runpod cancel failed");
            }
        }
        let _ = pool;
        true
    }

    /// The `gw_dispatch` row of a job.
    pub async fn dispatch_row(&self, id: JobId) -> Option<DispatchRow> {
        let rows = self
            .db
            .query(Stmt::new("SELECT * FROM gw_dispatch WHERE job_id = ?", vec![json!(id.to_string())]))
            .await
            .ok()?
            .rows;
        rows.first().map(DispatchRow::from_row)
    }

    /// Puts a lost job back to `queued` (worker cleared) and dispatches it
    /// again; `false` when another replica claimed the retry first or the
    /// job changed.
    pub(crate) async fn redispatch(&self, row: &DispatchRow, why: &str) -> Result<bool, ApiError> {
        let claimed = self
            .db
            .query(Stmt::new(
                "UPDATE gw_dispatch SET attempt = attempt + 1, updated_at = ? WHERE job_id = ? AND attempt = ? AND state = 'active'",
                vec![json!(now_ms()), json!(row.job_id), json!(row.attempt)],
            ))
            .await
            .map_err(|e| ApiError::internal(e.to_string()))?
            .changes;
        if claimed == 0 {
            return Ok(false);
        }
        let Some(pool) = self.pool(&row.pool) else { return Ok(false) };
        let attempt = row.attempt + 1;
        let msg = format!("{why}; dispatching again (attempt {attempt})");
        let Some(job) = self
            .rewrite_job(&row.job_id, true, |j, now| {
                if j.is_terminal() {
                    return false;
                }
                j.state = JobState::Queued;
                j.started_at = None;
                j.progress = 0.0;
                j.queue_position = None;
                j.logs.push(LogLine::info(msg.clone(), now));
                true
            })
            .await?
        else {
            return Ok(false);
        };
        // The old placement, best effort (normally gone already).
        if row.kind == "pod" {
            let _ = self.worker_req(reqwest::Method::DELETE, &format!("{}{}", row.target, job_path(job.id))).timeout(Duration::from_secs(5)).send().await;
        } else if let Some(r) = &row.r#ref {
            let _ = self.runpod.cancel(&row.target, r).await;
        }
        let exclude = (row.kind == "pod").then_some(row.target.as_str());
        match self.dispatch_to(pool, &job, &row.inputs, attempt, exclude).await {
            Ok(placed) => {
                self.record(pool, job.id, &placed, attempt, &row.inputs).await.map_err(ApiError::internal)?;
                metrics::counter!("fv_gateway_redispatched_total", "pool" => pool.id().to_owned()).increment(1);
                tracing::warn!(job = %job.id, pool = pool.id(), attempt, "gateway: job re-dispatched after a worker loss");
                Ok(true)
            }
            Err(e) => {
                let row2 = DispatchRow { attempt, ..row.clone() };
                self.fail_lost(&row2, &format!("{why}; re-dispatch failed: {}", e.message)).await;
                Ok(true)
            }
        }
    }

    /// Fails a lost job (claims the row first; idempotent across replicas).
    pub(crate) async fn fail_lost(&self, row: &DispatchRow, why: &str) {
        let claimed = self
            .db
            .query(Stmt::new(
                "UPDATE gw_dispatch SET state = 'lost', updated_at = ?, finished_at = ? WHERE job_id = ? AND attempt = ? AND state = 'active'",
                vec![json!(now_ms()), json!(now_ms()), json!(row.job_id), json!(row.attempt)],
            ))
            .await
            .map(|r| r.changes)
            .unwrap_or(0);
        if claimed == 0 {
            return;
        }
        let err = ApiError::internal(format!("the worker running this job was lost ({why})"));
        match self.rewrite_job(&row.job_id, false, |j, now| j.mark_failed(now, err.clone()).is_ok()).await {
            Ok(Some(job)) => {
                metrics::counter!("fv_gateway_lost_total", "pool" => row.pool.clone()).increment(1);
                tracing::warn!(job = %job.id, pool = %row.pool, %why, "gateway: job failed, its worker was lost");
                if let Ok(ctx) = self.ctx() {
                    ctx.notify(&job);
                }
            }
            Ok(None) => {}
            Err(e) => tracing::warn!(job = %row.job_id, error = %e.message, "gateway: failing a lost job"),
        }
    }

    /// Read-modify-write of a job row with a version check; `f` returns
    /// whether it changed the job. `clear_worker` also clears the row's
    /// `worker` (a re-dispatch). `Ok(None)`: nothing to do or lost a race.
    pub(crate) async fn rewrite_job(
        &self,
        id: &str,
        clear_worker: bool,
        mut f: impl FnMut(&mut Job, time::OffsetDateTime) -> bool,
    ) -> Result<Option<Job>, ApiError> {
        let rows = self
            .db
            .query(Stmt::new("SELECT job, version FROM jobs WHERE id = ?", vec![json!(id)]))
            .await
            .map_err(|e| ApiError::internal(e.to_string()))?
            .rows;
        let Some(r) = rows.first() else { return Ok(None) };
        let Some(mut job) = r.get("job").and_then(Value::as_str).and_then(|s| serde_json::from_str::<Job>(s).ok()) else { return Ok(None) };
        let version = r.get("version").and_then(Value::as_f64).unwrap_or(0.0) as i64;
        let now = time::OffsetDateTime::now_utc();
        if !f(&mut job, now) {
            return Ok(None);
        }
        let body = serde_json::to_string(&job).map_err(|e| ApiError::internal(e.to_string()))?;
        let worker = if clear_worker { ", worker = NULL" } else { "" };
        let changed = self
            .db
            .query(Stmt::new(
                format!(
                    "UPDATE jobs SET status = ?, progress = ?, updated_at = ?, completed_at = ?, job = ?, version = version + 1{worker} \
                     WHERE id = ? AND version = ?"
                ),
                vec![
                    json!(job.status().as_str()),
                    json!(job.progress as f64),
                    json!(now_ms()),
                    job.completed_at.map_or(Value::Null, |t| json!((t.unix_timestamp_nanos() / 1_000_000) as i64)),
                    json!(body),
                    json!(id),
                    json!(version),
                ],
            ))
            .await
            .map_err(|e| ApiError::internal(e.to_string()))?
            .changes;
        Ok((changed > 0).then_some(job))
    }
}

/// `503` for a pool that cannot take work.
fn unavailable(p: &Pool) -> ApiError {
    let why = p.lock().last_error.clone().unwrap_or_else(|| "no worker is reachable".into());
    ApiError::loading(format!("the pool serving this model (`{}`) is unavailable: {why}", p.id())).with_retry_after(15)
}

/// A `gw_dispatch` row.
#[derive(Clone, Debug)]
pub struct DispatchRow {
    pub job_id: String,
    pub pool: String,
    pub kind: String,
    pub target: String,
    pub r#ref: Option<String>,
    pub attempt: u32,
    pub inputs: Vec<InputRef>,
}

impl DispatchRow {
    pub fn from_row(r: &serde_json::Map<String, Value>) -> Self {
        let s = |k: &str| r.get(k).and_then(Value::as_str).unwrap_or_default().to_owned();
        Self {
            job_id: s("job_id"),
            pool: s("pool"),
            kind: s("kind"),
            target: s("target"),
            r#ref: r.get("ref").and_then(Value::as_str).map(str::to_owned),
            attempt: r.get("attempt").and_then(Value::as_f64).unwrap_or(1.0) as u32,
            inputs: r.get("inputs").and_then(Value::as_str).and_then(|t| serde_json::from_str(t).ok()).unwrap_or_default(),
        }
    }
}
