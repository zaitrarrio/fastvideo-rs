//! Job dispatch (docs/serve/gateway.md §3): pool choice, inputs, the
//! envelope, the `gw_dispatch` record, cancel, and the re-dispatch / fail
//! steps of the worker-loss reaper.
//!
//! Envelope (**native**), the body of the worker's
//! `POST /fv/v1/internal/jobs`:
//! `{"job": <Job>, "inputs": [{"path", "inline" | "source" | "url"+"artifact", …}], "attempt": n, "pool": id}`.
//!
//! Inputs travel the cheapest way that works (docs/serve/gateway.md §3.1):
//! inline in the envelope (base64) up to `gateway.inline_inputs_max_bytes`
//! per job; large video/audio the client gave as a public URL are fetched
//! by the worker from that URL; the rest go through the artifact store (R2)
//! with a signed URL. Inputs that skipped the store are copied into it
//! after the dispatch, in the background, for a re-dispatch after a worker
//! loss.
//! Serverless pools get it inside the queue envelope
//! `{"kind":"http","method":"POST","path":"/fv/v1/internal/jobs","body":…,
//! "wait":true,"poll_path":"/fv/v1/internal/jobs/<id>","cancel_path":…}`.

use std::time::{Duration, Instant};

use fastvideo_protocol::{ApiError, Job, JobId, JobState, LogLine};
use fastvideo_serve_kit::d1::Stmt;
use serde_json::{json, Value};

use super::schema::now_ms;
use super::{Gateway, NoWorker, Pool, Reservation, TakeLoad};

pub use crate::front::envelope::{sha256_file, Envelope, InputRef, INPUT_FETCH_FAILED};
use crate::front::envelope::{stage_for_retry, store_many, SERVERLESS_INLINE_MAX};

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
        order.sort_by_key(|p| (!p.available(), { let s = p.lock(); s.queued + s.pending() }));
        let budget = order.iter().map(|p| self.inline_budget(p)).min().unwrap_or(0);
        let t_stage = Instant::now();
        let mut inputs = self.plan_inputs(job, budget).await?;
        let mut stage_s = t_stage.elapsed().as_secs_f64();
        let mut last: Option<ApiError> = None;
        for p in order {
            if !p.available() {
                last.get_or_insert_with(|| unavailable(p));
                continue;
            }
            let max = p.cfg.max_queued;
            let queued = {
                let s = p.lock();
                s.queued + s.pending()
            };
            if max > 0 && queued >= max {
                last = Some(ApiError::queue_full(format!("pool `{}` has {queued} queued jobs (max {max})", p.id())).with_retry_after(5));
                continue;
            }
            let t_dispatch = Instant::now();
            tracing::debug!(job = %job.id, pool = p.id(), "gateway: dispatch call");
            let mut r = self.dispatch_to(p, job, &inputs, 1, None).await;
            if matches!(&r, Err(e) if e.param.as_deref() == Some(INPUT_FETCH_FAILED)) {
                // The worker could not fetch a client URL (gone, refused by
                // the SSRF guard, changed since ingestion): through the store.
                tracing::info!(job = %job.id, pool = p.id(), "gateway: a worker could not fetch a passed-through input; sending it through the store");
                let t = Instant::now();
                let idx: Vec<usize> = (0..inputs.len()).filter(|&i| inputs[i].source.is_some()).collect();
                store_many(self.ctx()?, self.input_url_ttl, &mut inputs, &idx).await?;
                stage_s += t.elapsed().as_secs_f64();
                r = self.dispatch_to(p, job, &inputs, 1, None).await;
            }
            match r {
                Ok(placed) => {
                    tracing::debug!(job = %job.id, "gateway: the worker took the job");
                    let dispatch_s = t_dispatch.elapsed().as_secs_f64();
                    p.lock().recording += 1;
                    let t_record = Instant::now();
                    if let Err(e) = self.record(p, job.id, &placed, 1, &inputs).await {
                        tracing::warn!(job = %job.id, error = %e, "gateway: recording the dispatch failed (the job runs; reaping and metrics miss it)");
                    }
                    {
                        // Counted as pending until a tick's D1 count read
                        // after this write (see `tick`).
                        let mut st = p.lock();
                        st.recording = st.recording.saturating_sub(1);
                        st.recorded.push(Instant::now());
                    }
                    let record_s = t_record.elapsed().as_secs_f64();
                    let pool = p.id().to_owned();
                    metrics::counter!("fv_gateway_dispatched_total", "pool" => pool.clone()).increment(1);
                    // Where the submit side of fal `timings.dispatch` goes
                    // (docs/serve/gateway.md §3.2).
                    for (phase, v) in [("stage_inputs", stage_s), ("dispatch", dispatch_s), ("record", record_s)] {
                        metrics::histogram!("fv_gateway_submit_phase_seconds", "pool" => pool.clone(), "phase" => phase).record(v);
                    }
                    let via = |w: &str| inputs.iter().filter(|i| i.via() == w).count();
                    tracing::info!(
                        job = %job.id, pool = %pool, inputs = inputs.len(), inline = via("inline"), source = via("source"), store = via("store"),
                        stage_inputs_ms = (stage_s * 1e3) as u64, dispatch_ms = (dispatch_s * 1e3) as u64, record_ms = (record_s * 1e3) as u64,
                        "gateway: dispatched"
                    );
                    self.follow(p, job.id, &placed);
                    if self.cfg.stage_inputs_for_retry && p.cfg.retries > 0 && !p.is_edge() && inputs.iter().any(|i| i.artifact.is_none()) {
                        if let Ok(ctx) = self.ctx() {
                            tokio::spawn(stage_for_retry(self.db.clone(), ctx.clone(), self.input_url_ttl, job.id, 1, inputs));
                        }
                    }
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

    /// Bytes of input a dispatch to `p` may carry inline.
    fn inline_budget(&self, p: &Pool) -> u64 {
        let max = self.cfg.inline_inputs_max_bytes;
        if p.is_pod() {
            max
        } else {
            max.min(SERVERLESS_INLINE_MAX)
        }
    }

    /// How each input travels ([`crate::front::envelope::plan_inputs`]).
    async fn plan_inputs(&self, job: &Job, budget: u64) -> Result<Vec<InputRef>, ApiError> {
        crate::front::envelope::plan_inputs(self.ctx()?, job, budget, self.cfg.input_passthrough, self.input_url_ttl).await
    }

    /// Inputs of a `gw_dispatch` row for a re-dispatch: an input whose store
    /// copy was not made yet goes through the store now, from this replica's
    /// staged file (off the client's path).
    pub(crate) async fn revive_inputs(&self, inputs: &[InputRef]) -> Result<Vec<InputRef>, ApiError> {
        let mut out = inputs.to_vec();
        let idx: Vec<usize> = (0..out.len()).filter(|&i| out[i].via() == "none").collect();
        if !idx.is_empty() {
            store_many(self.ctx()?, self.input_url_ttl, &mut out, &idx).await?;
        }
        Ok(out)
    }

    pub(crate) async fn drop_inputs(&self, inputs: &[InputRef]) {
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
        if pool.is_edge() {
            return self.edge_enqueue(pool, &env, timeout).await;
        }
        if pool.is_pod() {
            // One worker at a time, each reserved before its call (so the
            // concurrent submits of a burst spread instead of all picking
            // the worker that looked emptiest at the last probe).
            let mut tried: Vec<String> = exclude.map(|e| vec![e.to_owned()]).unwrap_or_default();
            let mut last = None;
            let mut full = false;
            loop {
                let slot = match Reservation::take(pool, &tried) {
                    Ok(r) => r,
                    Err(NoWorker::QueueFull) => {
                        full = true;
                        break;
                    }
                    Err(NoWorker::None) => break,
                };
                let url = slot.url.clone();
                tried.push(url.clone());
                let sent = Instant::now();
                let r = self
                    .worker_req(reqwest::Method::POST, &format!("{url}/fv/v1/internal/jobs"))
                    .timeout(timeout)
                    .json(&env)
                    .send()
                    .await;
                match r {
                    Ok(resp) if resp.status().is_success() => {
                        let v: Value = resp.json().await.unwrap_or(Value::Null);
                        slot.placed(job.id, sent, TakeLoad::parse(&v));
                        let worker = v.get("worker").and_then(Value::as_str).map(str::to_owned);
                        return Ok(Placed { target: url, r#ref: worker });
                    }
                    Ok(resp) => {
                        let s = resp.status().as_u16();
                        let v: Value = resp.json().await.unwrap_or(Value::Null);
                        let msg = v.pointer("/error/message").and_then(Value::as_str).unwrap_or_default().to_owned();
                        tracing::info!(job = %job.id, worker = %url, status = s, %msg, "gateway: worker refused the job");
                        if s == 424 {
                            // A passed-through input the worker could not fetch.
                            return Err(ApiError::internal(format!("the worker could not fetch an input: {msg}")).with_param(INPUT_FETCH_FAILED));
                        }
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
                        if s == 429 {
                            full = true;
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
                // `slot` dropped: the reservation is released.
            }
            // The details (worker URLs, errors) are for the log and the
            // admin route, not for API clients.
            if let Some(l) = &last {
                tracing::warn!(job = %job.id, pool = pool.id(), last = %l, "gateway: no worker took the job");
            }
            if full {
                return Err(ApiError::queue_full(format!("every worker of pool `{}` has a full queue", pool.id())).with_retry_after(5));
            }
            return Err(ApiError::loading(format!("pool `{}` has no worker that can take the job now", pool.id())).with_retry_after(10));
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
                tracing::warn!(job = %job.id, pool = pool.id(), error = %e, "gateway: the Runpod endpoint did not take the job");
                Err(ApiError::loading(format!("pool `{}` did not take the job; retry later", pool.id())).with_retry_after(10))
            }
        }
    }

    /// Follows a job dispatched to a pod worker (docs/serve/gateway.md
    /// §3.5): a status wait on the worker answers at each status change,
    /// once D1 has it, and the job goes into this replica's view and wakes
    /// its `watch()` poller, so a status change shows here without waiting
    /// for a poll tick (`watch_poll_ms`) and a D1 read. Stops at a terminal
    /// status, when the worker no longer knows the job (404), after three
    /// failed calls in a row, or with a worker that does not wait (an older
    /// build: no `job` in the answer). The poller and the reaper stay as
    /// they are; this only makes the dispatching replica see changes sooner.
    fn follow(&self, pool: &Pool, id: JobId, placed: &Placed) {
        if !pool.is_pod() || pool.is_edge() {
            return;
        }
        self.jobs.dispatched(id);
        let (http, token, jobs) = (self.http.clone(), self.token.clone(), self.jobs.clone());
        let url = format!("{}{}", placed.target, job_path(id));
        tokio::spawn(async move {
            const WAIT_S: u64 = 25;
            let mut since = "queued".to_owned();
            let mut errors = 0;
            loop {
                // `since` is a status word (`queued`, `running`, ...).
                let r = http
                    .get(format!("{url}?wait_s={WAIT_S}&since={since}"))
                    .header(super::TOKEN_HEADER, &token)
                    .timeout(Duration::from_secs(WAIT_S + 15))
                    .send()
                    .await;
                let v: Value = match r {
                    Ok(resp) if resp.status().is_success() => resp.json().await.unwrap_or(Value::Null),
                    Ok(resp) if resp.status() == reqwest::StatusCode::NOT_FOUND => return,
                    _ => {
                        errors += 1;
                        if errors >= 3 {
                            return;
                        }
                        tokio::time::sleep(Duration::from_secs(1)).await;
                        continue;
                    }
                };
                errors = 0;
                let Some(job) = v.get("job").and_then(|j| serde_json::from_value::<Job>(j.clone()).ok()) else {
                    return;
                };
                let done = job.is_terminal();
                since = job.status().as_str().to_owned();
                tracing::debug!(job = %id, status = %since, "gateway: the worker reported a status change");
                jobs.observe(job);
                if done {
                    return;
                }
            }
        });
    }

    /// Writes (or replaces) the `gw_dispatch` row; never over a later
    /// attempt's (a replica that lost the re-dispatch claim race).
    async fn record(&self, pool: &Pool, id: JobId, placed: &Placed, attempt: u32, inputs: &[InputRef]) -> Result<(), String> {
        let now = now_ms();
        let kind = if pool.is_edge() {
            super::edge::KIND
        } else if pool.is_pod() {
            "pod"
        } else {
            "runpod-serverless"
        };
        let rec: Vec<InputRef> = inputs.iter().map(InputRef::for_record).collect();
        let inputs = serde_json::to_string(&rec).unwrap_or_else(|_| "[]".into());
        self.db
            .query(Stmt::new(
                "INSERT INTO gw_dispatch (job_id, pool, kind, target, ref, attempt, state, inputs, created_at, updated_at) \
                 VALUES (?, ?, ?, ?, ?, ?, 'active', ?, ?, ?) \
                 ON CONFLICT(job_id) DO UPDATE SET pool = excluded.pool, kind = excluded.kind, target = excluded.target, \
                 ref = excluded.ref, attempt = excluded.attempt, state = 'active', updated_at = excluded.updated_at \
                 WHERE gw_dispatch.attempt <= excluded.attempt",
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
        if row.kind == super::edge::KIND {
            self.edge_cancel(&row, id).await;
        } else if row.kind == "pod" {
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
        let inputs = match self.revive_inputs(&row.inputs).await {
            Ok(i) => i,
            Err(e) => {
                let row2 = DispatchRow { attempt, ..row.clone() };
                self.fail_lost(&row2, &format!("{why}; its inputs are gone: {}", e.message)).await;
                return Ok(true);
            }
        };
        match self.dispatch_to(pool, &job, &inputs, attempt, exclude).await {
            Ok(placed) => {
                self.record(pool, job.id, &placed, attempt, &inputs).await.map_err(ApiError::internal)?;
                self.follow(pool, job.id, &placed);
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
        self.drop_inputs(&row.inputs).await;
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

/// `503` for a pool that cannot take work (the probe's error text stays
/// in the log and on the admin route).
fn unavailable(p: &Pool) -> ApiError {
    ApiError::loading(format!("the pool serving this model (`{}`) is unavailable: no worker is reachable", p.id())).with_retry_after(15)
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
