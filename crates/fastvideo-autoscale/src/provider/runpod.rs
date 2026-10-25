//! Runpod over its public APIs (feature `runpod`).
//!
//! - [`RunpodServerless`]: steers a serverless endpoint. Observes REST v1
//!   `GET /endpoints/{id}` (settings) and the queue API `GET /{id}/health`
//!   (workers idle / running / initializing); applies `PATCH /endpoints/{id}`
//!   with only the changed fields (`workersMin`, `workersMax`,
//!   `idleTimeout`, `scalerType`, `scalerValue`). Runpod starts and reaps
//!   the workers; lowering `workersMin` never stops a busy one, and the
//!   policy never sets `workersMax` below the busy workers.
//! - [`RunpodPods`]: pods from a template (REST v1 `POST /pods`), named
//!   `fv-as-<pool>-<n>`, across the pool's GPU types and region+volume
//!   placements in preference order (out of stock → next; no
//!   `allowedCudaVersions` filter). Ready = the worker URL answers the
//!   ready path with 200; the pod is then announced to the
//!   [`WorkerRegistry`]. Deletion only after drain and with zero in-flight
//!   work (checked again right before `DELETE`).
//! - [`RunpodBalance`]: GraphQL `myself { clientBalance }`.
//! - [`RunpodHealthSignals`]: queue signals from `/health` alone, for
//!   running the controller without a gateway (provider validation).
//!
//! The API key is only ever sent as a header; errors never echo it.
//!
//! **One Runpod implementation** (docs/adr/0018-fv-deploy.md §9): the
//! requests (`fastvideo_deploy::provider::runpod::Requests`), the answer
//! mapping (`decode`, `delete_answer`, `graphql_data`: no capacity, 4xx,
//! 5xx, the key scrubbed), the balance (`account_of`), the billed price
//! (`parse_usd`) and the pod-from-template body (`template_pod_body`,
//! `with_worker_config`) are fv-deploy's; this module only sends them on its
//! async reqwest client ([`RunpodApi::send`]) and keeps the pool steering
//! (observe / apply, the endpoint patch diff, `/health` counts).

use std::collections::{BTreeMap, BTreeSet};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use fastvideo_deploy::boot::sys::Request;
use fastvideo_deploy::provider::runpod::{
    self as shared, template_pod_body, with_worker_config, Requests, RunpodUrls, TemplatePod,
};
use fastvideo_deploy::provider::{parse_usd, ProviderKind};
use serde_json::{json, Map, Value};

use super::{ApplyReport, BalanceSource, Provider, ProviderError};
use crate::config::{PoolConfig, PoolKind};
use crate::gateway::{SignalSource, WorkerRegistry};
use crate::types::{
    EndpointSettings, PoolDecision, PoolObservation, PoolSignals, Worker, WorkerState,
};

/// API endpoints and the key.
#[derive(Clone)]
pub struct RunpodApi {
    http: reqwest::Client,
    key: String,
    pub urls: RunpodUrls,
}

impl std::fmt::Debug for RunpodApi {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RunpodApi")
            .field("urls", &self.urls)
            .finish_non_exhaustive()
    }
}

impl From<fastvideo_deploy::provider::ProviderError> for ProviderError {
    fn from(e: fastvideo_deploy::provider::ProviderError) -> Self {
        match e {
            fastvideo_deploy::provider::ProviderError::NoCapacity(m) => {
                ProviderError::NoCapacity(m)
            }
            other => ProviderError::Api(other.to_string()),
        }
    }
}

/// `/health` counts of a serverless endpoint.
#[derive(Clone, Debug, Default, PartialEq, serde::Deserialize, serde::Serialize)]
#[serde(default)]
pub struct Health {
    pub jobs: HealthJobs,
    pub workers: HealthWorkers,
}

#[derive(Clone, Debug, Default, PartialEq, serde::Deserialize, serde::Serialize)]
#[serde(default, rename_all = "camelCase")]
pub struct HealthJobs {
    pub completed: u64,
    pub failed: u64,
    pub in_progress: u64,
    pub in_queue: u64,
    pub retried: u64,
}

#[derive(Clone, Debug, Default, PartialEq, serde::Deserialize, serde::Serialize)]
#[serde(default)]
pub struct HealthWorkers {
    pub idle: u32,
    pub initializing: u32,
    pub ready: u32,
    pub running: u32,
    pub throttled: u32,
    pub unhealthy: u32,
}

fn err(ctx: &str, e: impl std::fmt::Display) -> ProviderError {
    ProviderError::Api(format!("{ctx}: {e}"))
}

impl RunpodApi {
    pub fn new(api_key: impl Into<String>) -> Result<Self, ProviderError> {
        let http = reqwest::Client::builder()
            .timeout(Duration::from_secs(60))
            .build()
            .map_err(|e| err("http client", e))?;
        Ok(Self::with_client(http, api_key, RunpodUrls::default()))
    }

    /// On a given client and URLs (tests point it at a local fake).
    pub fn with_client(
        http: reqwest::Client,
        api_key: impl Into<String>,
        urls: RunpodUrls,
    ) -> Self {
        Self {
            http,
            key: api_key.into(),
            urls,
        }
    }

    /// fv-deploy's requests for this key and these URLs.
    pub fn requests(&self) -> Requests<'_> {
        Requests::new(&self.urls, &self.key)
    }

    /// From `RUNPOD_API_KEY` (or `FV_RUNPOD_API_KEY`).
    pub fn from_env() -> Result<Self, ProviderError> {
        let key = std::env::var("FV_RUNPOD_API_KEY")
            .or_else(|_| std::env::var("RUNPOD_API_KEY"))
            .map_err(|_| ProviderError::Other("RUNPOD_API_KEY is not set".into()))?;
        Self::new(key)
    }

    /// Sends one of fv-deploy's requests on the async client: its status and
    /// body (the transport's error never carries the URL or the key).
    pub async fn send(&self, req: &Request, what: &str) -> Result<(u16, Vec<u8>), ProviderError> {
        use fastvideo_deploy::boot::sys::Method;
        let m = match req.method {
            Method::Get => reqwest::Method::GET,
            Method::Post => reqwest::Method::POST,
            Method::Delete => reqwest::Method::DELETE,
            Method::Patch => reqwest::Method::PATCH,
            Method::Put => reqwest::Method::PUT,
        };
        let mut b = self.http.request(m, &req.url).timeout(req.timeout);
        for (k, v) in &req.headers {
            b = b.header(k, v);
        }
        if let Some(body) = &req.body {
            b = b.body(body.clone());
        }
        let resp = b.send().await.map_err(|e| err(what, e.without_url()))?;
        let status = resp.status().as_u16();
        let body = resp.bytes().await.map_err(|e| err(what, e.without_url()))?;
        Ok((status, body.to_vec()))
    }

    /// A REST answer's JSON (fv-deploy's `decode`).
    async fn json(&self, req: &Request, what: &str) -> Result<Value, ProviderError> {
        let (status, body) = self.send(req, what).await?;
        Ok(shared::decode(what, status, &body, &self.key)?)
    }

    pub async fn endpoint(&self, id: &str) -> Result<Value, ProviderError> {
        self.json(&self.requests().endpoint(id), "GET endpoint")
            .await
    }

    pub async fn patch_endpoint(&self, id: &str, body: &Value) -> Result<Value, ProviderError> {
        self.json(&self.requests().patch_endpoint(id, body), "PATCH endpoint")
            .await
    }

    pub async fn health(&self, id: &str) -> Result<Health, ProviderError> {
        let v = self.json(&self.requests().health(id), "GET health").await?;
        serde_json::from_value(v).map_err(|e| err("health", e))
    }

    pub async fn balance(&self) -> Result<f64, ProviderError> {
        let what = "GraphQL balance";
        let (status, body) = self.send(&self.requests().balance(), what).await?;
        let data = shared::graphql_data(what, status, &body, &self.key)?;
        Ok(shared::account_of(ProviderKind::RunpodPod, &data)?.balance_usd)
    }

    pub async fn pods(&self) -> Result<Vec<Value>, ProviderError> {
        match self.json(&self.requests().pods(), "GET pods").await? {
            Value::Array(v) => Ok(v),
            _ => Ok(Vec::new()),
        }
    }

    pub async fn create_pod(&self, body: &Value) -> Result<Value, ProviderError> {
        self.json(&self.requests().create_pod(body), "POST pods")
            .await
    }

    /// A template's `env` (REST v1 `GET /templates/{templateId}`; an absent `env` is empty).
    pub async fn template_env(&self, id: &str) -> Result<Map<String, Value>, ProviderError> {
        let v = self
            .json(&self.requests().template(id), "GET template")
            .await?;
        Ok(template_env_of(&v))
    }

    /// Gone (404) is deleted (fv-deploy's `delete_answer`).
    pub async fn delete_pod(&self, id: &str) -> Result<(), ProviderError> {
        let what = "DELETE pod";
        let (status, body) = self.send(&self.requests().delete_pod(id), what).await?;
        Ok(shared::delete_answer(what, status, &body, &self.key)?)
    }
}

/// Shared with fv-deploy (`fastvideo_deploy::provider::runpod`).
pub use fastvideo_deploy::provider::runpod::template_env_of;

/// Endpoint settings from a REST v1 endpoint object.
pub fn endpoint_settings(v: &Value) -> EndpointSettings {
    let u = |k: &str| v.get(k).and_then(Value::as_u64).unwrap_or(0) as u32;
    EndpointSettings {
        workers_min: u("workersMin"),
        workers_max: u("workersMax"),
        idle_timeout_s: u("idleTimeout"),
        scaler_type: v
            .get("scalerType")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_owned(),
        scaler_value: u("scalerValue"),
    }
}

/// The PATCH body: only fields that differ.
pub fn endpoint_patch(
    cur: Option<&EndpointSettings>,
    want: &EndpointSettings,
) -> Map<String, Value> {
    let mut m = Map::new();
    let differs = |f: &dyn Fn(&EndpointSettings) -> Value| cur.is_none_or(|c| f(c) != f(want));
    if differs(&|e| json!(e.workers_min)) {
        m.insert("workersMin".into(), json!(want.workers_min));
    }
    if differs(&|e| json!(e.workers_max)) {
        m.insert("workersMax".into(), json!(want.workers_max));
    }
    if want.idle_timeout_s > 0 && differs(&|e| json!(e.idle_timeout_s)) {
        m.insert("idleTimeout".into(), json!(want.idle_timeout_s));
    }
    if !want.scaler_type.is_empty() && differs(&|e| json!(e.scaler_type)) {
        m.insert("scalerType".into(), json!(want.scaler_type));
    }
    if want.scaler_value > 0 && differs(&|e| json!(e.scaler_value)) {
        m.insert("scalerValue".into(), json!(want.scaler_value));
    }
    m
}

/// Workers of a serverless endpoint as `/health` counts them (ids are
/// positional: Runpod does not list serverless workers by id here).
/// `workers.running` stays up for a while after a worker's last job ends
/// (seen live: `running: 2` with `inProgress: 0`), so only
/// `min(running, jobs.inProgress)` of them count as busy.
pub fn health_workers(health: &Health, now_s: f64) -> Vec<Worker> {
    let h = &health.workers;
    let busy_n = u64::from(h.running).min(health.jobs.in_progress) as u32;
    let mk = |kind: &str, i: u32, state: WorkerState, busy: u32| Worker {
        id: format!("{kind}-{i}"),
        state,
        busy,
        created_at_s: now_s,
        ready_at_s: None,
        url: None,
        gpu_type: None,
        usd_per_hr: None,
    };
    let mut v = Vec::new();
    v.extend((0..h.running).map(|i| mk("running", i, WorkerState::Ready, u32::from(i < busy_n))));
    v.extend((0..h.idle).map(|i| mk("idle", i, WorkerState::Ready, 0)));
    v.extend((0..h.initializing).map(|i| mk("initializing", i, WorkerState::Starting, 0)));
    v
}

/// Steers Runpod serverless endpoints.
pub struct RunpodServerless {
    api: RunpodApi,
}

impl RunpodServerless {
    pub fn new(api: RunpodApi) -> Self {
        Self { api }
    }
}

#[async_trait::async_trait]
impl Provider for RunpodServerless {
    fn name(&self) -> &'static str {
        "runpod-serverless"
    }

    async fn observe(
        &self,
        pool: &PoolConfig,
        now_s: f64,
    ) -> Result<PoolObservation, ProviderError> {
        let id = &pool.serverless.endpoint_id;
        let ep = self.api.endpoint(id).await?;
        let h = self.api.health(id).await?;
        Ok(PoolObservation {
            workers: health_workers(&h, now_s),
            endpoint: Some(endpoint_settings(&ep)),
        })
    }

    async fn apply(
        &self,
        pool: &PoolConfig,
        d: &PoolDecision,
        _now_s: f64,
    ) -> Result<ApplyReport, ProviderError> {
        let mut r = ApplyReport::default();
        let Some(want) = d.endpoint.as_ref() else {
            return Ok(r);
        };
        let id = &pool.serverless.endpoint_id;
        // Re-read right before writing: another tool may have changed it.
        let cur = endpoint_settings(&self.api.endpoint(id).await?);
        let patch = endpoint_patch(Some(&cur), want);
        if patch.is_empty() {
            return Ok(r);
        }
        let body = Value::Object(patch);
        self.api.patch_endpoint(id, &body).await?;
        tracing::info!(pool = %pool.name, endpoint = %id, patch = %body, "autoscale: endpoint patched");
        r.endpoint_patched = true;
        r.notes.push(format!("PATCH {body}"));
        Ok(r)
    }
}

/// Account balance over GraphQL.
pub struct RunpodBalance(pub RunpodApi);

#[async_trait::async_trait]
impl BalanceSource for RunpodBalance {
    async fn balance_usd(&self) -> Result<f64, ProviderError> {
        self.0.balance().await
    }
}

/// Queue signals from Runpod `/health` (no gateway): queued = `inQueue`,
/// running = `inProgress`, the oldest job's age is the time since the
/// queue was last empty, arrivals = the sum of all job counters.
pub struct RunpodHealthSignals {
    api: RunpodApi,
    /// pool → endpoint id.
    endpoints: BTreeMap<String, String>,
    /// pool → time the queue became non-empty.
    since: Mutex<BTreeMap<String, f64>>,
}

impl RunpodHealthSignals {
    pub fn new(api: RunpodApi, pools: &[PoolConfig]) -> Self {
        let endpoints = pools
            .iter()
            .filter(|p| p.kind == PoolKind::Serverless)
            .map(|p| (p.name.clone(), p.serverless.endpoint_id.clone()))
            .collect();
        Self {
            api,
            endpoints,
            since: Mutex::new(BTreeMap::new()),
        }
    }
}

#[async_trait::async_trait]
impl SignalSource for RunpodHealthSignals {
    async fn signals(&self, pools: &[String]) -> Vec<PoolSignals> {
        let now = crate::controller::unix_now();
        let mut out = Vec::new();
        for pool in pools {
            let Some(ep) = self.endpoints.get(pool) else {
                continue;
            };
            let h = match self.api.health(ep).await {
                Ok(h) => h,
                Err(e) => {
                    tracing::warn!(pool = %pool, error = %e, "autoscale: /health failed");
                    continue;
                }
            };
            let oldest = {
                let mut g = self.since.lock().unwrap_or_else(|p| p.into_inner());
                if h.jobs.in_queue > 0 {
                    now - *g.entry(pool.clone()).or_insert(now)
                } else {
                    g.remove(pool);
                    0.0
                }
            };
            out.push(PoolSignals {
                pool: pool.clone(),
                queued: h.jobs.in_queue as u32,
                oldest_queued_s: oldest,
                running_jobs: h.jobs.in_progress as u32,
                live_streams: 0,
                arrivals_total: h.jobs.completed
                    + h.jobs.failed
                    + h.jobs.in_progress
                    + h.jobs.in_queue,
                recent_job_s: None,
                arrival_rate_per_s: None,
            });
        }
        out
    }
}

// ---- The gateway's pod workers ----------------------------------------

/// [`WorkerRegistry`] over the gateway's own bookkeeping (docs/serve/gateway.md
/// §5.3): pod workers register themselves in D1 `gw_workers`, so
/// `register`/`deregister` only log; in-flight work per worker is
/// `running + sessions` of its row (seen in the last 45 s); `drain` asks
/// the worker itself (`POST {url}/fv/v1/internal/drain` with the internal
/// token) to report `draining`, best effort. Deletion never relies on the
/// drain alone: the pods provider re-reads the in-flight count right before
/// `DELETE`.
pub struct GatewayWorkers {
    d1: fastvideo_serve_kit::d1::D1Client,
    http: reqwest::Client,
    internal_token: Option<String>,
    urls: Mutex<BTreeMap<String, String>>,
}

impl GatewayWorkers {
    pub fn new(
        d1: fastvideo_serve_kit::d1::D1Client,
        internal_token: Option<String>,
    ) -> Result<Self, ProviderError> {
        let http = reqwest::Client::builder()
            .timeout(Duration::from_secs(10))
            .build()
            .map_err(|e| err("http client", e))?;
        Ok(Self {
            d1,
            http,
            internal_token,
            urls: Mutex::new(BTreeMap::new()),
        })
    }

    async fn post(&self, worker_id: &str, path: &str) {
        let Some(url) = self
            .urls
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .get(worker_id)
            .cloned()
        else {
            return;
        };
        let Some(tok) = &self.internal_token else {
            return;
        };
        match self
            .http
            .post(format!("{url}{path}"))
            .header("x-fv-internal-token", tok)
            .send()
            .await
        {
            Ok(r) if r.status().is_success() => {}
            Ok(r) => {
                tracing::warn!(worker = %worker_id, status = %r.status(), path, "autoscale: worker drain request refused")
            }
            Err(e) => {
                tracing::warn!(worker = %worker_id, error = %e.without_url(), path, "autoscale: worker unreachable")
            }
        }
    }
}

#[async_trait::async_trait]
impl WorkerRegistry for GatewayWorkers {
    async fn register(&self, pool: &str, worker_id: &str, url: &str) {
        self.urls
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .insert(worker_id.to_owned(), url.to_owned());
        tracing::info!(
            pool,
            worker = worker_id,
            url,
            "autoscale: pod ready (it registers itself with the gateway)"
        );
    }
    async fn drain(&self, _pool: &str, worker_id: &str) {
        self.post(worker_id, "/fv/v1/internal/drain").await;
    }
    async fn undrain(&self, _pool: &str, worker_id: &str) {
        self.post(worker_id, "/fv/v1/internal/undrain").await;
    }
    async fn deregister(&self, _pool: &str, worker_id: &str) {
        self.urls
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .remove(worker_id);
    }
    async fn in_flight(&self, pool: &str) -> BTreeMap<String, u32> {
        let now = crate::controller::unix_now();
        let q = fastvideo_serve_kit::d1::Stmt::new(
            "SELECT worker_id, url, running, sessions, updated_at FROM gw_workers WHERE pool = ?",
            vec![json!(pool)],
        );
        let rows = match self.d1.query(q).await {
            Ok(r) => r.rows,
            Err(e) => {
                tracing::warn!(pool, error = %e, "autoscale: gw_workers unreadable");
                return BTreeMap::new();
            }
        };
        let mut m = BTreeMap::new();
        for r in rows {
            // Seen in the last 45 s (`updated_at` in ms or s).
            let at = r.get("updated_at").and_then(Value::as_f64).unwrap_or(0.0);
            let at = if at > 1e11 { at / 1000.0 } else { at };
            if now - at > 45.0 {
                continue;
            }
            let n = r.get("running").and_then(Value::as_u64).unwrap_or(0)
                + r.get("sessions").and_then(Value::as_u64).unwrap_or(0);
            for k in ["worker_id", "url"] {
                if let Some(s) = r.get(k).and_then(Value::as_str) {
                    m.insert(s.trim_end_matches('/').to_owned(), n as u32);
                }
            }
        }
        m
    }
}

// ---- Pods -------------------------------------------------------------

/// Shared with fv-deploy (`fastvideo_deploy::provider::runpod`).
pub use fastvideo_deploy::provider::runpod::parse_runpod_time;

#[derive(Default)]
struct PodBook {
    ready_at: BTreeMap<String, f64>,
    created_at: BTreeMap<String, f64>,
    draining: BTreeSet<String>,
    seq: u64,
}

/// Pods from a template, one pool per name prefix `fv-as-<pool>-`.
pub struct RunpodPods {
    api: RunpodApi,
    http: reqwest::Client,
    registry: Arc<dyn WorkerRegistry>,
    prices: BTreeMap<String, f64>,
    book: Mutex<PodBook>,
}

impl RunpodPods {
    pub fn new(
        api: RunpodApi,
        registry: Arc<dyn WorkerRegistry>,
        prices: BTreeMap<String, f64>,
    ) -> Result<Self, ProviderError> {
        let http = reqwest::Client::builder()
            .timeout(Duration::from_secs(10))
            .build()
            .map_err(|e| err("http client", e))?;
        Ok(Self {
            api,
            http,
            registry,
            prices,
            book: Mutex::new(PodBook::default()),
        })
    }

    fn book(&self) -> std::sync::MutexGuard<'_, PodBook> {
        self.book.lock().unwrap_or_else(|p| p.into_inner())
    }

    pub fn prefix(pool: &str) -> String {
        format!("fv-as-{pool}-")
    }

    fn url(pool: &PoolConfig, id: &str) -> String {
        pool.pod.url_template.replace("{id}", id)
    }

    /// Pod create payloads in preference order: every GPU type in every placement. `template_env` is the
    /// template's own env: REST v1 `POST /pods` documents no rule for a body `env` next to a `templateId`
    /// (the v2 create documents a per-key merge, body values winning), so a body that sends the worker
    /// config carries the template's env too (the secret references, the image identity), with the config
    /// on top: the same pod whichever rule v1 applies.
    pub fn payloads(
        pool: &PoolConfig,
        name: &str,
        template_env: &Map<String, Value>,
    ) -> Vec<(String, Value)> {
        // The template's image bakes no worker config (ADR-0011): the pool's
        // rides in the pod's env (fv-deploy's `with_worker_config`).
        let env = (!pool.pod.worker_toml_b64.is_empty())
            .then(|| with_worker_config(template_env, &pool.pod.worker_toml_b64));
        let mut v = Vec::new();
        for pl in &pool.pod.placements {
            for gpu in &pool.pod.gpu_types {
                let body = template_pod_body(&TemplatePod {
                    name,
                    template_id: &pool.pod.template_id,
                    cloud_type: &pool.pod.cloud_type,
                    gpu,
                    volume_id: &pl.volume_id,
                    data_center: &pl.data_center,
                    env: env.as_ref(),
                });
                v.push((format!("{gpu} in {}", pl.data_center), body));
            }
        }
        v
    }

    async fn is_ready(&self, url: &str, path: &str) -> bool {
        matches!(self.http.get(format!("{url}{path}")).send().await, Ok(r) if r.status().as_u16() == 200)
    }

    async fn create_one(
        &self,
        pool: &PoolConfig,
        now_s: f64,
        r: &mut ApplyReport,
    ) -> Result<String, ProviderError> {
        let name = {
            let mut b = self.book();
            b.seq += 1;
            format!(
                "{}{}{}",
                Self::prefix(&pool.name),
                now_s as u64 % 100_000_000,
                b.seq
            )
        };
        // The template's env rides along with the worker config (see `payloads`); without it the pod
        // could boot without the template's secret references, so a failed read creates nothing.
        let template_env = if pool.pod.worker_toml_b64.is_empty() {
            Map::new()
        } else {
            self.api
                .template_env(&pool.pod.template_id)
                .await
                .map_err(|e| {
                    ProviderError::Api(format!(
                        "template {}: its env could not be read ({e}); no pod created",
                        pool.pod.template_id
                    ))
                })?
        };
        let mut last = None;
        for (label, body) in Self::payloads(pool, &name, &template_env) {
            match self.api.create_pod(&body).await {
                Ok(v) => {
                    let Some(id) = v.get("id").and_then(Value::as_str).map(str::to_owned) else {
                        last = Some(ProviderError::Api(format!("create on {label}: no id")));
                        continue;
                    };
                    // A number or a numeric string (fv-deploy's `parse_usd`; Runpod's
                    // REST docs show `'0.74'`); missing stays 0 here (the pool's cap
                    // applies to a reported price; ADR-0018 §9 open decision).
                    let dph = parse_usd(v.get("costPerHr")).unwrap_or(0.0);
                    let gpu = body["gpuTypeIds"][0].as_str().unwrap_or("");
                    let cap = if pool.pod.max_usd_per_hr > 0.0 {
                        pool.pod.max_usd_per_hr
                    } else {
                        self.prices.get(gpu).map_or(f64::INFINITY, |p| p * 1.25)
                    };
                    if dph > cap {
                        r.notes.push(format!(
                            "{id} on {label} costs ${dph}/h > ${cap:.2}/h: deleted"
                        ));
                        let _ = self.api.delete_pod(&id).await;
                        continue;
                    }
                    self.book().created_at.insert(id.clone(), now_s);
                    tracing::info!(pool = %pool.name, pod = %id, on = %label, usd_per_hr = dph, "autoscale: pod created");
                    return Ok(id);
                }
                Err(ProviderError::NoCapacity(m)) => {
                    r.notes.push(format!("no stock: {label}"));
                    last = Some(ProviderError::NoCapacity(m));
                }
                Err(e) => {
                    r.notes.push(format!("create on {label} failed: {e}"));
                    last = Some(e);
                }
            }
        }
        Err(last.unwrap_or_else(|| {
            ProviderError::NoCapacity("no GPU type / placement configured".into())
        }))
    }
}

#[async_trait::async_trait]
impl Provider for RunpodPods {
    fn name(&self) -> &'static str {
        "runpod-pods"
    }

    async fn observe(
        &self,
        pool: &PoolConfig,
        now_s: f64,
    ) -> Result<PoolObservation, ProviderError> {
        let prefix = Self::prefix(&pool.name);
        let pods: Vec<Value> = self
            .api
            .pods()
            .await?
            .into_iter()
            .filter(|p| {
                p.get("name")
                    .and_then(Value::as_str)
                    .is_some_and(|n| n.starts_with(&prefix))
            })
            .filter(|p| p.get("desiredStatus").and_then(Value::as_str) != Some("TERMINATED"))
            .collect();
        let load = self.registry.in_flight(&pool.name).await;
        let mut workers = Vec::new();
        for p in pods {
            let Some(id) = p.get("id").and_then(Value::as_str).map(str::to_owned) else {
                continue;
            };
            let url = Self::url(pool, &id);
            let created = self
                .book()
                .created_at
                .get(&id)
                .copied()
                .or_else(|| {
                    p.get("createdAt")
                        .and_then(Value::as_str)
                        .and_then(parse_runpod_time)
                })
                .unwrap_or(now_s);
            let mut ready_at = self.book().ready_at.get(&id).copied();
            if ready_at.is_none() && self.is_ready(&url, &pool.pod.ready_path).await {
                ready_at = Some(now_s);
                self.book().ready_at.insert(id.clone(), now_s);
                self.registry.register(&pool.name, &id, &url).await;
                tracing::info!(pool = %pool.name, pod = %id, after_s = now_s - created, "autoscale: pod ready");
            }
            let busy = load
                .get(&id)
                .or_else(|| load.get(&url))
                .copied()
                .unwrap_or(0);
            let state = if self.book().draining.contains(&id) {
                WorkerState::Draining
            } else if ready_at.is_some() {
                WorkerState::Ready
            } else {
                WorkerState::Starting
            };
            workers.push(Worker {
                id,
                state,
                busy,
                created_at_s: created,
                ready_at_s: ready_at,
                url: Some(url),
                gpu_type: p
                    .pointer("/machine/gpuTypeId")
                    .and_then(Value::as_str)
                    .map(str::to_owned),
                usd_per_hr: parse_usd(p.get("costPerHr")),
            });
        }
        Ok(PoolObservation {
            workers,
            endpoint: None,
        })
    }

    async fn apply(
        &self,
        pool: &PoolConfig,
        d: &PoolDecision,
        now_s: f64,
    ) -> Result<ApplyReport, ProviderError> {
        let mut r = ApplyReport::default();
        for id in &d.drain {
            self.book().draining.insert(id.clone());
            self.registry.drain(&pool.name, id).await;
            r.drained.push(id.clone());
        }
        for id in &d.undrain {
            self.book().draining.remove(id);
            self.registry.undrain(&pool.name, id).await;
            r.undrained.push(id.clone());
        }
        if !d.delete.is_empty() {
            let load = self.registry.in_flight(&pool.name).await;
            for id in &d.delete {
                let url = Self::url(pool, id);
                let busy = load
                    .get(id)
                    .or_else(|| load.get(&url))
                    .copied()
                    .unwrap_or(0);
                if busy > 0 {
                    r.notes
                        .push(format!("{id} got work since the decision: not deleted"));
                    continue;
                }
                match self.api.delete_pod(id).await {
                    Ok(()) => {
                        self.registry.deregister(&pool.name, id).await;
                        let mut b = self.book();
                        b.draining.remove(id);
                        b.ready_at.remove(id);
                        b.created_at.remove(id);
                        r.deleted.push(id.clone());
                        tracing::info!(pool = %pool.name, pod = %id, "autoscale: pod deleted");
                    }
                    Err(e) => r.notes.push(format!("delete {id}: {e}")),
                }
            }
        }
        for _ in 0..d.create {
            match self.create_one(pool, now_s, &mut r).await {
                Ok(id) => r.created.push(id),
                Err(e) => {
                    r.notes.push(format!("create: {e}"));
                    break;
                }
            }
        }
        Ok(r)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{Placement, PodConfig};

    #[test]
    fn patch_has_only_changed_fields() {
        let cur = EndpointSettings {
            workers_min: 0,
            workers_max: 1,
            idle_timeout_s: 5,
            scaler_type: "QUEUE_DELAY".into(),
            scaler_value: 1,
        };
        let want = EndpointSettings {
            workers_min: 2,
            idle_timeout_s: 60,
            ..cur.clone()
        };
        let p = endpoint_patch(Some(&cur), &want);
        assert_eq!(
            Value::Object(p),
            json!({"workersMin": 2, "idleTimeout": 60})
        );
        assert!(endpoint_patch(Some(&cur), &cur).is_empty());
        let v = json!({"workersMin": 1, "workersMax": 3, "idleTimeout": 300, "scalerType": "QUEUE_DELAY", "scalerValue": 4});
        assert_eq!(endpoint_settings(&v).workers_max, 3);
    }

    #[test]
    fn health_counts_become_workers() {
        let h: Health = serde_json::from_str(
            r#"{"jobs":{"completed":69,"failed":31,"inProgress":1,"inQueue":2,"retried":2},
                "workers":{"idle":1,"initializing":1,"ready":1,"running":1,"throttled":0,"unhealthy":0}}"#,
        )
        .unwrap();
        let w = health_workers(&h, 10.0);
        assert_eq!(w.len(), 3);
        assert_eq!(w.iter().filter(|w| w.busy > 0).count(), 1);
        // Running workers with no job in progress are not busy.
        let idle: Health =
            serde_json::from_str(r#"{"jobs":{"inProgress":0},"workers":{"running":2}}"#).unwrap();
        assert!(health_workers(&idle, 0.0).iter().all(|w| w.busy == 0));
        assert_eq!(
            w.iter()
                .filter(|w| w.state == WorkerState::Starting)
                .count(),
            1
        );
    }

    #[test]
    fn runpod_times() {
        assert_eq!(
            parse_runpod_time("2026-09-28 00:00:00.000 +0000 UTC"),
            Some(1_790_553_600.0)
        );
        assert_eq!(
            parse_runpod_time("2026-09-27T20:31:01Z"),
            Some(1_790_553_600.0 - 12_539.0)
        );
        assert_eq!(parse_runpod_time("garbage"), None);
    }

    #[test]
    fn pod_payloads_fall_back_across_types_and_regions() {
        let pool = PoolConfig {
            name: "sfwan".into(),
            kind: PoolKind::Pod,
            pod: PodConfig {
                template_id: "tpl".into(),
                gpu_types: vec!["NVIDIA H100 80GB HBM3".into(), "NVIDIA H200".into()],
                // Two regions (a hypothetical rebuilt US volume first; the old
                // US volume s2k01690bi was deleted 2026-10).
                placements: vec![
                    Placement {
                        data_center: "US-CA-2".into(),
                        volume_id: "usrebuilt0".into(),
                    },
                    Placement {
                        data_center: "EU-RO-1".into(),
                        volume_id: "bz0ofkfx45".into(),
                    },
                ],
                ..PodConfig::default()
            },
            ..PoolConfig::default()
        };
        let p = RunpodPods::payloads(&pool, "fv-as-sfwan-1", &Map::new());
        let order: Vec<&str> = p.iter().map(|(l, _)| l.as_str()).collect();
        assert_eq!(
            order,
            [
                "NVIDIA H100 80GB HBM3 in US-CA-2",
                "NVIDIA H200 in US-CA-2",
                "NVIDIA H100 80GB HBM3 in EU-RO-1",
                "NVIDIA H200 in EU-RO-1"
            ]
        );
        let b = &p[2].1;
        assert_eq!(b["networkVolumeId"], "bz0ofkfx45");
        assert_eq!(b["templateId"], "tpl");
        assert!(b.get("allowedCudaVersions").is_none());
        // No resolved config: no env (the binary always resolves one).
        assert!(b.get("env").is_none());
        let mut with = pool.clone();
        with.pod.worker_toml_b64 = "W3NlcnZlcl0K".into();
        // The template's env rides along (REST v1 documents no merge rule), the config on top.
        let tpl = template_env_of(&json!({"env": {
            "FV_CF_ACCOUNT_ID": "{{ RUNPOD_SECRET_fv_cf_account_id }}",
            "FV_WORKER_TOML_B64": "b2xk",
            "N": 3
        }}));
        let b = &RunpodPods::payloads(&with, "fv-as-sfwan-2", &tpl)[0].1;
        assert_eq!(b["env"]["FV_WORKER_TOML_B64"], "W3NlcnZlcl0K");
        assert_eq!(
            b["env"]["FV_CF_ACCOUNT_ID"],
            "{{ RUNPOD_SECRET_fv_cf_account_id }}"
        );
        assert!(b["env"].get("N").is_none());
        // Without a config the template's env applies as it is (no body env).
        assert!(RunpodPods::payloads(&pool, "x", &tpl)[0]
            .1
            .get("env")
            .is_none());
    }

    /// A local fake Runpod: `POST /v1/pods` says no stock, `DELETE` of a
    /// gone pod is a 404, a 500 echoes the key, GraphQL answers a balance,
    /// a template has an env. Every request's method, path and
    /// authorization are recorded.
    async fn fake_runpod() -> (RunpodUrls, Arc<Mutex<Vec<String>>>) {
        use axum::extract::{OriginalUri, Request as AxRequest};
        use axum::http::StatusCode;
        let seen: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
        let s = Arc::clone(&seen);
        let app = axum::Router::new().fallback(move |uri: OriginalUri, req: AxRequest| {
            let s = Arc::clone(&s);
            async move {
                let auth = req
                    .headers()
                    .get("authorization")
                    .and_then(|v| v.to_str().ok())
                    .unwrap_or("")
                    .to_owned();
                let line = format!("{} {} {auth}", req.method(), uri.0.path());
                s.lock()
                    .unwrap_or_else(|p| p.into_inner())
                    .push(line.clone());
                match line.split(' ').take(2).collect::<Vec<_>>().as_slice() {
                    ["POST", "/v1/pods"] => (
                        StatusCode::INTERNAL_SERVER_ERROR,
                        "There are no longer any instances available".to_owned(),
                    ),
                    ["DELETE", "/v1/pods/gone"] => (StatusCode::NOT_FOUND, String::new()),
                    ["DELETE", _] => (StatusCode::OK, "{}".to_owned()),
                    ["GET", "/v1/endpoints/boom"] => {
                        (StatusCode::INTERNAL_SERVER_ERROR, format!("echo {auth}"))
                    }
                    ["GET", "/v1/templates/tpl"] => (
                        StatusCode::OK,
                        r#"{"env":{"FV_CF_ACCOUNT_ID":"{{ RUNPOD_SECRET_fv_cf_account_id }}"}}"#
                            .to_owned(),
                    ),
                    ["POST", "/graphql"] => (
                        StatusCode::OK,
                        r#"{"data":{"myself":{"clientBalance":21.5,"currentSpendPerHr":0}}}"#
                            .to_owned(),
                    ),
                    _ => (StatusCode::NOT_FOUND, "{}".to_owned()),
                }
            }
        });
        let l = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .unwrap_or_else(|e| panic!("bind: {e}"));
        let base = format!(
            "http://{}",
            l.local_addr().map(|a| a.to_string()).unwrap_or_default()
        );
        tokio::spawn(async move {
            let _ = axum::serve(l, app).await;
        });
        let urls = RunpodUrls {
            rest: format!("{base}/v1"),
            queue: format!("{base}/v2"),
            graphql: format!("{base}/graphql"),
            ..RunpodUrls::default()
        };
        (urls, seen)
    }

    /// The autoscaler sends fv-deploy's requests and reads the answers with
    /// fv-deploy's decoders: no stock is `NoCapacity`, a gone pod's delete
    /// is done, the balance parses, a template's env is read, and an error
    /// never carries the key.
    #[tokio::test(flavor = "multi_thread")]
    async fn the_async_client_sends_the_shared_requests() {
        const KEY: &str = "rpa_AUTOSCALE_SENTINEL";
        let (urls, seen) = fake_runpod().await;
        let http = reqwest::Client::builder()
            .no_proxy()
            .build()
            .unwrap_or_else(|e| panic!("{e}"));
        let api = RunpodApi::with_client(http, KEY, urls.clone());
        let e = api.create_pod(&json!({"name": "x"})).await;
        assert!(matches!(e, Err(ProviderError::NoCapacity(_))), "{e:?}");
        assert!(api.delete_pod("gone").await.is_ok());
        assert!(api.delete_pod("p1").await.is_ok());
        assert_eq!(api.balance().await.ok(), Some(21.5));
        let env = api.template_env("tpl").await.unwrap_or_default();
        assert_eq!(
            env.get("FV_CF_ACCOUNT_ID").and_then(Value::as_str),
            Some("{{ RUNPOD_SECRET_fv_cf_account_id }}")
        );
        let e = api
            .endpoint("boom")
            .await
            .err()
            .map(|e| e.to_string())
            .unwrap_or_default();
        assert!(e.contains("HTTP 500") && !e.contains(KEY), "{e}");
        let got = seen.lock().map(|g| g.clone()).unwrap_or_default();
        let bearer = format!("Bearer {KEY}");
        assert!(got.iter().all(|l| l.ends_with(&bearer)), "{got:?}");
        // The same requests fv-deploy's blocking client sends.
        let r = Requests::new(&urls, KEY);
        let want: Vec<String> = [
            r.create_pod(&json!({"name": "x"})),
            r.delete_pod("gone"),
            r.delete_pod("p1"),
            r.balance(),
            r.template("tpl"),
            r.endpoint("boom"),
        ]
        .iter()
        .map(|q| {
            let path = q.url.split_once("://").map_or("", |x| x.1);
            let path = &path[path.find('/').unwrap_or(0)..];
            format!("{} {path} {bearer}", q.method.as_str())
        })
        .collect();
        assert_eq!(got, want);
    }
}
