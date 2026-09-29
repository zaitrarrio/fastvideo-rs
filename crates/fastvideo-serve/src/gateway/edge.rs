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

use std::collections::BTreeMap;
use std::time::{Duration, Instant};

use fastvideo_dispatch_proto::{self as proto, EnqueueReq, EnqueueResp, PoolStatus};
use fastvideo_protocol::{ApiError, JobId};

use super::dispatch::{DispatchRow, Envelope, Placed};
use super::tick::WorkerStatus;
use super::{Gateway, Pool, WorkerBuild, WorkerView, TOKEN_HEADER};
use crate::config::DispatchMode;

/// `gw_dispatch.kind` of a job handed to a Durable Object.
pub const KIND: &str = "durable-object";

impl Pool {
    /// Jobs go through the pool's Durable Object.
    pub fn is_edge(&self) -> bool {
        self.cfg.dispatch == DispatchMode::DurableObject
    }

    /// The fv-edge base URL (validated at config time).
    pub fn do_base(&self) -> &str {
        self.cfg.do_url.as_deref().unwrap_or("").trim_end_matches('/')
    }
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
        };
        let t0 = Instant::now();
        let r = self.edge_req(reqwest::Method::POST, &format!("{base}{}", proto::enqueue_path(pool.id()))).timeout(timeout).json(&body).send().await;
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
        let url = format!("{}{}", pool.do_base(), proto::cancel_path(pool.id(), &id.to_string()));
        match self.edge_req(reqwest::Method::POST, &url).timeout(Duration::from_secs(15)).send().await {
            Ok(r) if r.status().is_success() || r.status().as_u16() == 404 => {}
            Ok(r) => tracing::warn!(job = %id, status = r.status().as_u16(), "gateway: the Durable Object refused a cancel"),
            Err(e) => tracing::warn!(job = %id, error = %e.without_url(), "gateway: forwarding cancel to the Durable Object failed"),
        }
    }

    /// The tick's probe of a Durable Object pool: its workers, their caps
    /// and builds, and the jobs it failed (marked failed in D1 here).
    pub(crate) async fn probe_edge(&self, p: &Pool) {
        let url = format!("{}{}", p.do_base(), proto::status_path(p.id()));
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
            for w in &st_.workers {
                let parsed = WorkerStatus::parse(&w.caps);
                let key = format!("do:{}", w.worker_id);
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
        // Failed in the last 10 min (older ones were handled by earlier ticks).
        let recent = st_.now_ms - 600_000;
        for f in st_.failed.iter().filter(|f| f.at_ms > recent) {
            self.edge_failed(p, f).await;
        }
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
}
