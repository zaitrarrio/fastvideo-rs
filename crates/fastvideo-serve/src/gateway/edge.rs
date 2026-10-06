//! Pod pools with `dispatch = "durable-object"` (docs/serve/gateway-cloudflare.md,
//! phase 1): the gateway keeps auth, the API adapters, the D1 job rows and
//! the console, and hands each job to the pool's Durable Object
//! (`crates/fastvideo-edge`) instead of posting it to a worker. The DO
//! pushes it over the socket of a connected worker, whose ack is the adopt
//! (as on the gateway path, the worker adopts the D1 row and writes
//! progress and results to D1/R2).
//!
//! - submit: `POST {do_url}/pools/{pool}/enqueue` with the same envelope;
//!   the `gw_dispatch` row gets kind `durable-object`.
//! - cancel: `POST {do_url}/pools/{pool}/cancel/{job}`.
//! - tick: `GET {do_url}/pools/{pool}/status` replaces the worker probes
//!   (workers, load, caps and builds come from the workers' hellos). Worker
//!   loss is the DO's (re-dispatch once, then fail): the reaper leaves these
//!   rows alone, and jobs the DO failed are failed in D1 here.
//! - phase 2: the job row of a job only DO pools serve is inserted behind
//!   the dispatch ([`install`]); the worker starts at once and writes its
//!   row behind too, fenced by the push's lease. A job a worker could not
//!   fetch a client URL for comes back in the status (`restage`): its inputs
//!   go through the store and the envelope is replaced on the DO.
//! - family objects (`[[pools]] family = "…"`, docs/serve/dispatch-do-family.md):
//!   the same calls go to `/families/{family}/…`, several pools can share one
//!   object, and **streaming sessions** are admitted by the object
//!   ([`Gateway::edge_admit`]): it reserves a GPU through the worker's
//!   arbiter and returns the worker's public endpoint, to which the
//!   signalling is then proxied (media flows client ↔ GPU). The gateway's
//!   session lease carries the object's session id; the tick renews the
//!   object's lease while the gateway's is live and releases it when the
//!   gateway's ends.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use fastvideo_dispatch_proto::{self as proto, EnqueueReq, EnqueueResp, PoolStatus, Scope, SessionGrant, SessionReq};
use fastvideo_protocol::{ApiError, JobId};

use super::dispatch::{DispatchRow, Envelope, InputRef, Placed};
use super::tick::WorkerStatus;
use super::{Gateway, Pool, WorkerBuild, WorkerView, TOKEN_HEADER};
use crate::config::DispatchMode;

/// `gw_dispatch.kind` of a job handed to a Durable Object.
pub const KIND: &str = "durable-object";

/// Restages in progress on this replica (job ids).
static RESTAGING: Mutex<BTreeSet<String>> = Mutex::new(BTreeSet::new());

/// Phase 2: job rows of jobs only Durable Object pools serve are inserted
/// behind the dispatch (see the module docs).
pub(crate) fn install(gw: &Arc<Gateway>) {
    if !gw.pools.iter().any(Pool::is_edge) {
        return;
    }
    let weak = Arc::downgrade(gw);
    gw.jobs.set_insert_behind(Arc::new(move |job| {
        let Some(gw) = weak.upgrade() else { return false };
        let cat = gw.catalog();
        cat.pools_of.get(&job.resolved.model).is_some_and(|ps| !ps.is_empty() && ps.iter().all(|i| gw.pools.get(*i).is_some_and(Pool::is_edge)))
    }));
}

impl Pool {
    /// Jobs go through the pool's Durable Object.
    pub fn is_edge(&self) -> bool {
        self.cfg.dispatch == DispatchMode::DurableObject
    }

    /// The fv-edge base URL (validated at config time).
    pub fn do_base(&self) -> &str {
        self.cfg.do_url.as_deref().unwrap_or("").trim_end_matches('/')
    }

    /// The object this pool's jobs go to: its family's, else its own.
    pub fn scope(&self) -> Scope {
        match &self.cfg.family {
            Some(f) => Scope::Family(f.clone()),
            None => Scope::Pool(self.id().to_owned()),
        }
    }

    /// Sessions are admitted by the pool's family object.
    pub fn edge_sessions(&self) -> bool {
        self.is_edge() && self.cfg.family.is_some()
    }
}

/// Why an admission failed: status, message, retry-after.
#[derive(Debug)]
pub struct AdmitError {
    pub status: u16,
    pub message: String,
    pub retry_after: Option<u32>,
}

impl Gateway {
    fn edge_req(&self, m: reqwest::Method, url: &str) -> reqwest::RequestBuilder {
        self.http.request(m, url).header(TOKEN_HEADER, self.token())
    }

    /// Hands the envelope to the pool's Durable Object.
    pub(crate) async fn edge_enqueue(&self, pool: &Pool, env: &Envelope, timeout: Duration) -> Result<Placed, ApiError> {
        let base = pool.do_base();
        let body = EnqueueReq {
            job_id: env.job.id.to_string(),
            envelope: serde_json::to_value(env).map_err(|e| ApiError::internal(format!("encoding the envelope: {e}")))?,
            retries: pool.cfg.retries,
            model: Some(env.job.resolved.model.to_string()),
            replace: false,
            owner: env.job.owner.as_ref().map(|o| o.0.clone()),
            max_queued: 0,
        };
        let t0 = Instant::now();
        let r = self.edge_req(reqwest::Method::POST, &format!("{base}{}", pool.scope().enqueue_path())).timeout(timeout).json(&body).send().await;
        let ms = t0.elapsed().as_secs_f64() * 1e3;
        match r {
            Ok(resp) if resp.status().is_success() => {
                let v: EnqueueResp = resp.json().await.map_err(|e| ApiError::internal(format!("the dispatcher's answer: {}", e.without_url())))?;
                tracing::debug!(job = %env.job.id, pool = pool.id(), state = %v.state, worker = ?v.worker, position = v.position, enqueue_ms = ms as u64, "gateway: enqueued on the Durable Object");
                metrics::histogram!("fv_gateway_edge_enqueue_seconds", "pool" => pool.id().to_owned()).record(ms / 1e3);
                Ok(Placed { target: base.to_owned(), r#ref: v.worker })
            }
            Ok(resp) => {
                let s = resp.status().as_u16();
                let msg = resp.text().await.unwrap_or_default();
                tracing::warn!(job = %env.job.id, pool = pool.id(), status = s, body = %msg.chars().take(300).collect::<String>(), "gateway: the Durable Object refused the job");
                pool.lock().last_error = Some(format!("dispatcher answered {s}"));
                Err(ApiError::loading(format!("pool `{}` did not take the job; retry later", pool.id())).with_retry_after(10))
            }
            Err(e) => {
                let msg = e.without_url().to_string();
                tracing::warn!(job = %env.job.id, pool = pool.id(), error = %msg, "gateway: the Durable Object is unreachable");
                pool.lock().last_error = Some(msg);
                Err(ApiError::loading(format!("pool `{}` did not take the job; retry later", pool.id())).with_retry_after(10))
            }
        }
    }

    /// Forwards a cancel to the pool's Durable Object (best effort).
    pub(crate) async fn edge_cancel(&self, row: &DispatchRow, id: JobId) {
        let Some(pool) = self.pool(&row.pool) else { return };
        let url = format!("{}{}", pool.do_base(), pool.scope().cancel_path(&id.to_string()));
        match self.edge_req(reqwest::Method::POST, &url).timeout(Duration::from_secs(15)).send().await {
            Ok(r) if r.status().is_success() || r.status().as_u16() == 404 => {}
            Ok(r) => tracing::warn!(job = %id, status = r.status().as_u16(), "gateway: the Durable Object refused a cancel"),
            Err(e) => tracing::warn!(job = %id, error = %e.without_url(), "gateway: forwarding cancel to the Durable Object failed"),
        }
    }

    /// The tick's probe of a Durable Object pool: its workers, their caps
    /// and builds, and the jobs it failed (marked failed in D1 here).
    pub(crate) async fn probe_edge(&self, p: &Pool) {
        let url = format!("{}{}", p.do_base(), p.scope().status_path());
        let r = self.edge_req(reqwest::Method::GET, &url).timeout(Duration::from_secs(5)).send().await;
        let status = match r {
            Ok(resp) if resp.status().is_success() => resp.json::<PoolStatus>().await.map_err(|e| e.without_url().to_string()),
            Ok(resp) => Err(format!("dispatcher status answered {}", resp.status().as_u16())),
            Err(e) => Err(e.without_url().to_string()),
        };
        let st_ = match status {
            Ok(s) => s,
            Err(e) => {
                let mut st = p.lock();
                st.available = false;
                st.last_error = Some(e);
                st.workers.clear();
                return;
            }
        };
        let refresh = Duration::from_secs(self.cfg.caps_refresh_s.max(1));
        {
            let mut st = p.lock();
            let mut next: BTreeMap<String, WorkerView> = BTreeMap::new();
            // A family object's worker may serve several families: this
            // pool keeps the live caps of its own models only (else it would
            // claim, and take, another family's jobs).
            let own: BTreeSet<&str> = p.static_caps.iter().map(|(c, _)| c.id.0.as_str()).collect();
            for w in &st_.workers {
                let mut parsed = WorkerStatus::parse(&w.caps);
                if p.cfg.family.is_some() {
                    parsed.caps.retain(|(c, _)| own.contains(c.id.0.as_str()));
                }
                // A family object's workers announce their public endpoint
                // (sessions are proxied there); others are reached only
                // through the object.
                let key = if w.endpoint.is_empty() { format!("do:{}", w.worker_id) } else { w.endpoint.trim_end_matches('/').to_owned() };
                let build = parsed.build.clone().or_else(|| {
                    Some(WorkerBuild { version: Some(w.version.clone()), git_sha: Some(w.sha.clone()), ..WorkerBuild::default() })
                });
                let view = WorkerView {
                    url: key.clone(),
                    id: Some(w.worker_id.clone()),
                    healthy: w.connected,
                    ready: w.connected,
                    draining: w.draining,
                    running: w.held,
                    registered: true,
                    last_ok: w.connected.then(Instant::now),
                    build,
                    ..WorkerView::default()
                };
                if w.connected && !parsed.caps.is_empty() && st.caps_at.is_none_or(|t| t.elapsed() >= refresh) {
                    st.live_caps = Some(parsed.caps);
                    st.caps_at = Some(Instant::now());
                }
                next.insert(key, view);
            }
            st.available = st_.usable_workers() > 0;
            st.last_error = if st.available { None } else { Some("no worker is connected to the pool's Durable Object".into()) };
            st.workers = next;
        }
        for id in &st_.restage {
            self.edge_restage(p, id).await;
        }
        // Failed in the last 10 min (older ones were handled by earlier ticks).
        let recent = st_.now_ms - 600_000;
        for f in st_.failed.iter().filter(|f| f.at_ms > recent) {
            self.edge_failed(p, f).await;
        }
    }

    /// A worker could not fetch a client URL: send the job again with its
    /// inputs in the store (the envelope is replaced on the DO).
    async fn edge_restage(&self, p: &Pool, job_id: &str) {
        if !RESTAGING.lock().unwrap_or_else(|e| e.into_inner()).insert(job_id.to_owned()) {
            return;
        }
        let r = async {
            let id = job_id.parse::<JobId>().map_err(|_| "bad job id".to_owned())?;
            let row = self.dispatch_row(id).await.ok_or("no dispatch row")?;
            let job = fastvideo_protocol::JobStore::get(self.jobs.as_ref(), id).await.ok_or("no job row")?;
            // Every input through the store (from this replica's staged files).
            let bare: Vec<InputRef> = row.inputs.iter().map(|i| InputRef { path: i.path.clone(), kind: i.kind, bytes: i.bytes, ..InputRef::default() }).collect();
            let inputs = self.revive_inputs(&bare).await.map_err(|e| e.message)?;
            let env = Envelope { job, inputs, attempt: row.attempt, pool: Some(p.id().to_owned()) };
            let body = EnqueueReq {
                job_id: job_id.to_owned(),
                envelope: serde_json::to_value(&env).map_err(|e| e.to_string())?,
                retries: p.cfg.retries,
                model: Some(env.job.resolved.model.to_string()),
                replace: true,
                owner: env.job.owner.as_ref().map(|o| o.0.clone()),
                max_queued: 0,
            };
            let url = format!("{}{}", p.do_base(), p.scope().enqueue_path());
            let resp = self.edge_req(reqwest::Method::POST, &url).timeout(Duration::from_secs(30)).json(&body).send().await.map_err(|e| e.without_url().to_string())?;
            if !resp.status().is_success() {
                return Err(format!("dispatcher answered {}", resp.status()));
            }
            Ok::<usize, String>(env.inputs.len())
        }
        .await;
        match r {
            Ok(n) => tracing::info!(job = %job_id, pool = p.id(), inputs = n, "gateway: inputs restaged through the store for a worker that could not fetch them"),
            Err(e) => tracing::warn!(job = %job_id, pool = p.id(), error = %e, "gateway: restaging a job's inputs failed (the next tick retries)"),
        }
        RESTAGING.lock().unwrap_or_else(|e| e.into_inner()).remove(job_id);
    }

    /// A job the Durable Object failed: fail its D1 row (idempotent: the
    /// row claim in `fail_lost` makes it once).
    async fn edge_failed(&self, p: &Pool, f: &proto::FailedJob) {
        let Ok(id) = f.job_id.parse::<JobId>() else { return };
        let Some(row) = self.dispatch_row(id).await else { return };
        if row.kind != KIND || row.pool != p.id() {
            return;
        }
        if row.attempt < f.attempt {
            // The DO re-dispatched it before failing; keep the row's attempt in step.
            let _ = self
                .db
                .query(fastvideo_serve_kit::d1::Stmt::new(
                    "UPDATE gw_dispatch SET attempt = ? WHERE job_id = ? AND state = 'active'",
                    vec![serde_json::json!(f.attempt), serde_json::json!(f.job_id)],
                ))
                .await;
        }
        let row = DispatchRow { attempt: f.attempt.max(row.attempt), ..row };
        let why = f.error.strip_prefix("the worker running this job was lost (").and_then(|s| s.strip_suffix(')')).unwrap_or(&f.error).to_owned();
        self.fail_lost(&row, &format!("{why} [dispatcher]")).await;
    }

    /// Admits a streaming session through the pool's family object: a GPU
    /// with a free session slot is reserved and its endpoint returned.
    pub(crate) async fn edge_admit(&self, pool: &Pool, kind: &str, model: Option<&str>, owner: Option<&str>) -> Result<SessionGrant, AdmitError> {
        let body = SessionReq {
            session_id: Some(format!("gws-{}", uuid::Uuid::new_v4().simple())),
            model: model.map(str::to_owned),
            kind: kind.to_owned(),
            owner: owner.map(str::to_owned),
            ttl_ms: 0,
        };
        let url = format!("{}{}", pool.do_base(), pool.scope().sessions_path());
        let t0 = Instant::now();
        let r = self.edge_req(reqwest::Method::POST, &url).timeout(Duration::from_secs(30)).json(&body).send().await;
        let err = |status: u16, message: String| AdmitError { status, message, retry_after: Some(10) };
        match r {
            Ok(resp) if resp.status().is_success() => {
                let g: SessionGrant = resp.json().await.map_err(|e| err(502, format!("the dispatcher's answer: {}", e.without_url())))?;
                tracing::info!(pool = pool.id(), session = %g.session_id, worker = %g.worker_id, endpoint = %g.endpoint, admit_ms = t0.elapsed().as_millis() as u64, "gateway: session admitted by the family object");
                metrics::histogram!("fv_gateway_edge_admit_seconds", "pool" => pool.id().to_owned()).record(t0.elapsed().as_secs_f64());
                Ok(g)
            }
            Ok(resp) => {
                let s = resp.status().as_u16();
                let msg = resp.json::<serde_json::Value>().await.ok().and_then(|v| v.pointer("/error/message").and_then(|m| m.as_str()).map(str::to_owned)).unwrap_or_default();
                Err(err(if s == 429 { 429 } else { 503 }, format!("pool `{}` has no free worker for a session: {msg}", pool.id())))
            }
            Err(e) => Err(err(503, format!("pool `{}`: the dispatcher is unreachable: {}", pool.id(), e.without_url()))),
        }
    }

    /// Releases a family object's session (best effort).
    pub(crate) async fn edge_release(&self, pool_id: &str, session: &str) {
        let Some(pool) = self.pool(pool_id) else { return };
        let url = format!("{}{}", pool.do_base(), pool.scope().session_release_path(session));
        match self.edge_req(reqwest::Method::POST, &url).timeout(Duration::from_secs(15)).send().await {
            Ok(r) if r.status().is_success() || r.status().as_u16() == 404 => tracing::debug!(session, "gateway: released the object's session"),
            Ok(r) => tracing::warn!(session, status = r.status().as_u16(), "gateway: the object refused a session release"),
            Err(e) => tracing::warn!(session, error = %e.without_url(), "gateway: releasing a session on the object failed"),
        }
    }

    /// Renews a family object's session; `false` when the object no longer
    /// has it (expired, its worker lost): the gateway's lease ends too.
    pub(crate) async fn edge_renew(&self, pool_id: &str, session: &str) -> bool {
        let Some(pool) = self.pool(pool_id) else { return true };
        let url = format!("{}{}", pool.do_base(), pool.scope().session_renew_path(session));
        match self.edge_req(reqwest::Method::POST, &url).timeout(Duration::from_secs(10)).send().await {
            Ok(r) => r.status().as_u16() != 404,
            Err(_) => true,
        }
    }

    /// The tick's side of session leases: every live gateway lease that
    /// holds a family object's session renews it there; one the object lost
    /// ends here.
    pub(crate) async fn edge_sessions_sync(&self) {
        if !self.pools.iter().any(Pool::edge_sessions) {
            return;
        }
        let rows = match self.db.query(fastvideo_serve_kit::d1::Stmt::raw("SELECT id, pool, body FROM gw_sessions WHERE state = 'live' AND body IS NOT NULL")).await {
            Ok(r) => r.rows,
            Err(_) => return,
        };
        let live: Vec<(String, String, String)> = rows
            .iter()
            .filter_map(|r| {
                let s = |k: &str| r.get(k).and_then(serde_json::Value::as_str).map(str::to_owned);
                Some((s("id")?, s("pool")?, do_session_of(s("body").as_deref())?))
            })
            .collect();
        let futs = live.iter().map(|(id, pool, sid)| async move {
            if !self.edge_renew(pool, sid).await {
                tracing::info!(lease = %id, session = %sid, "gateway: the family object ended this session; ending the lease");
                self.lease_end(id).await;
            }
        });
        futures::future::join_all(futs).await;
    }
}

/// The family object's session id kept in a lease row's `body`.
pub(crate) fn do_session_of(body: Option<&str>) -> Option<String> {
    let v: serde_json::Value = serde_json::from_str(body?).ok()?;
    v.get("do_session").and_then(serde_json::Value::as_str).map(str::to_owned)
}

/// A lease row's `body` for a family object's session.
pub(crate) fn do_session_body(sid: Option<&str>) -> Option<String> {
    sid.map(|s| serde_json::json!({"do_session": s}).to_string())

}
