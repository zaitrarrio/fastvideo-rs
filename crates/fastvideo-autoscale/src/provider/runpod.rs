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
//! - [`HttpGatewayPools`]: signals from a gateway's
//!   `GET /fv/v1/gateway/pools` (an autoscaler outside the gateway process).
//!
//! The API key is only ever sent as a header; errors never echo it.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use serde_json::{json, Map, Value};

use super::{ApplyReport, BalanceSource, Provider, ProviderError};
use crate::config::{PoolConfig, PoolKind};
use crate::gateway::{GatewayPoolMetrics, GatewaySignals, SignalSource, WorkerRegistry};
use crate::types::{EndpointSettings, PoolDecision, PoolObservation, PoolSignals, Worker, WorkerState};

/// API endpoints and the key.
#[derive(Clone)]
pub struct RunpodApi {
    http: reqwest::Client,
    key: String,
    pub rest: String,
    pub queue: String,
    pub graphql: String,
}

impl std::fmt::Debug for RunpodApi {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RunpodApi").field("rest", &self.rest).field("queue", &self.queue).finish_non_exhaustive()
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
            .timeout(Duration::from_secs(30))
            .build()
            .map_err(|e| err("http client", e))?;
        Ok(Self {
            http,
            key: api_key.into(),
            rest: "https://rest.runpod.io/v1".into(),
            queue: "https://api.runpod.ai/v2".into(),
            graphql: "https://api.runpod.io/graphql".into(),
        })
    }

    /// From `RUNPOD_API_KEY` (or `FV_RUNPOD_API_KEY`).
    pub fn from_env() -> Result<Self, ProviderError> {
        let key = std::env::var("FV_RUNPOD_API_KEY")
            .or_else(|_| std::env::var("RUNPOD_API_KEY"))
            .map_err(|_| ProviderError::Other("RUNPOD_API_KEY is not set".into()))?;
        Self::new(key)
    }

    async fn send(&self, req: reqwest::RequestBuilder, ctx: &str) -> Result<Value, ProviderError> {
        let resp = req.bearer_auth(&self.key).send().await.map_err(|e| err(ctx, e.without_url()))?;
        let status = resp.status();
        let text = resp.text().await.map_err(|e| err(ctx, e.without_url()))?;
        if !status.is_success() {
            let snippet: String = text.chars().take(300).collect();
            if snippet.contains("no instances currently available") || snippet.contains("no longer any instances") {
                return Err(ProviderError::NoCapacity(format!("{ctx}: {snippet}")));
            }
            return Err(err(ctx, format!("HTTP {status}: {snippet}")));
        }
        if text.trim().is_empty() {
            return Ok(Value::Null);
        }
        serde_json::from_str(&text).map_err(|e| err(ctx, e))
    }

    pub async fn endpoint(&self, id: &str) -> Result<Value, ProviderError> {
        self.send(self.http.get(format!("{}/endpoints/{id}", self.rest)), "GET endpoint").await
    }

    pub async fn patch_endpoint(&self, id: &str, body: &Value) -> Result<Value, ProviderError> {
        self.send(self.http.patch(format!("{}/endpoints/{id}", self.rest)).json(body), "PATCH endpoint").await
    }

    pub async fn health(&self, id: &str) -> Result<Health, ProviderError> {
        let v = self.send(self.http.get(format!("{}/{id}/health", self.queue)), "GET health").await?;
        serde_json::from_value(v).map_err(|e| err("health", e))
    }

    pub async fn balance(&self) -> Result<f64, ProviderError> {
        let v = self
            .send(
                self.http.post(&self.graphql).json(&json!({"query": "{ myself { clientBalance } }"})),
                "GraphQL balance",
            )
            .await?;
        v.pointer("/data/myself/clientBalance")
            .and_then(Value::as_f64)
            .ok_or_else(|| ProviderError::Api("GraphQL balance: no clientBalance".into()))
    }

    pub async fn pods(&self) -> Result<Vec<Value>, ProviderError> {
        match self.send(self.http.get(format!("{}/pods", self.rest)), "GET pods").await? {
            Value::Array(v) => Ok(v),
            _ => Ok(Vec::new()),
        }
    }

    pub async fn create_pod(&self, body: &Value) -> Result<Value, ProviderError> {
        self.send(self.http.post(format!("{}/pods", self.rest)).json(body), "POST pods").await
    }

    pub async fn delete_pod(&self, id: &str) -> Result<(), ProviderError> {
        self.send(self.http.delete(format!("{}/pods/{id}", self.rest)), "DELETE pod").await.map(|_| ())
    }
}

/// Endpoint settings from a REST v1 endpoint object.
pub fn endpoint_settings(v: &Value) -> EndpointSettings {
    let u = |k: &str| v.get(k).and_then(Value::as_u64).unwrap_or(0) as u32;
    EndpointSettings {
        workers_min: u("workersMin"),
        workers_max: u("workersMax"),
        idle_timeout_s: u("idleTimeout"),
        scaler_type: v.get("scalerType").and_then(Value::as_str).unwrap_or("").to_owned(),
        scaler_value: u("scalerValue"),
    }
}

/// The PATCH body: only fields that differ.
pub fn endpoint_patch(cur: Option<&EndpointSettings>, want: &EndpointSettings) -> Map<String, Value> {
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

    async fn observe(&self, pool: &PoolConfig, now_s: f64) -> Result<PoolObservation, ProviderError> {
        let id = &pool.serverless.endpoint_id;
        let ep = self.api.endpoint(id).await?;
        let h = self.api.health(id).await?;
        Ok(PoolObservation { workers: health_workers(&h, now_s), endpoint: Some(endpoint_settings(&ep)) })
    }

    async fn apply(&self, pool: &PoolConfig, d: &PoolDecision, _now_s: f64) -> Result<ApplyReport, ProviderError> {
        let mut r = ApplyReport::default();
        let Some(want) = d.endpoint.as_ref() else { return Ok(r) };
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
        Self { api, endpoints, since: Mutex::new(BTreeMap::new()) }
    }
}

#[async_trait::async_trait]
impl SignalSource for RunpodHealthSignals {
    async fn signals(&self, pools: &[String]) -> Vec<PoolSignals> {
        let now = crate::controller::unix_now();
        let mut out = Vec::new();
        for pool in pools {
            let Some(ep) = self.endpoints.get(pool) else { continue };
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
                arrivals_total: h.jobs.completed + h.jobs.failed + h.jobs.in_progress + h.jobs.in_queue,
                recent_job_s: None,
                arrival_rate_per_s: None,
            });
        }
        out
    }
}

/// Signals from a gateway over HTTP (`GET /fv/v1/gateway/pools`, admin token).
pub struct HttpGatewayPools {
    http: reqwest::Client,
    base: String,
    admin_token: String,
    sink: GatewaySignals,
}

impl HttpGatewayPools {
    pub fn new(base: impl Into<String>, admin_token: impl Into<String>) -> Result<Self, ProviderError> {
        let http = reqwest::Client::builder().timeout(Duration::from_secs(15)).build().map_err(|e| err("http client", e))?;
        Ok(Self { http, base: base.into().trim_end_matches('/').to_owned(), admin_token: admin_token.into(), sink: GatewaySignals::default() })
    }
}

#[async_trait::async_trait]
impl SignalSource for HttpGatewayPools {
    async fn signals(&self, pools: &[String]) -> Vec<PoolSignals> {
        #[derive(serde::Deserialize)]
        struct Pools {
            pools: Vec<GatewayPoolMetrics>,
        }
        let r = self.http.get(format!("{}/fv/v1/gateway/pools", self.base)).bearer_auth(&self.admin_token).send().await;
        match r {
            Ok(resp) if resp.status().is_success() => match resp.json::<Pools>().await {
                Ok(p) => self.sink.update(&p.pools),
                Err(e) => tracing::warn!(error = %e.without_url(), "autoscale: gateway pools body"),
            },
            Ok(resp) => tracing::warn!(status = %resp.status(), "autoscale: gateway pools"),
            Err(e) => tracing::warn!(error = %e.without_url(), "autoscale: gateway pools unreachable"),
        }
        self.sink.signals(pools).await
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
    pub fn new(d1: fastvideo_serve_kit::d1::D1Client, internal_token: Option<String>) -> Result<Self, ProviderError> {
        let http = reqwest::Client::builder().timeout(Duration::from_secs(10)).build().map_err(|e| err("http client", e))?;
        Ok(Self { d1, http, internal_token, urls: Mutex::new(BTreeMap::new()) })
    }

    async fn post(&self, worker_id: &str, path: &str) {
        let Some(url) = self.urls.lock().unwrap_or_else(|p| p.into_inner()).get(worker_id).cloned() else { return };
        let Some(tok) = &self.internal_token else { return };
        match self.http.post(format!("{url}{path}")).header("x-fv-internal-token", tok).send().await {
            Ok(r) if r.status().is_success() => {}
            Ok(r) => tracing::warn!(worker = %worker_id, status = %r.status(), path, "autoscale: worker drain request refused"),
            Err(e) => tracing::warn!(worker = %worker_id, error = %e.without_url(), path, "autoscale: worker unreachable"),
        }
    }
}

#[async_trait::async_trait]
impl WorkerRegistry for GatewayWorkers {
    async fn register(&self, pool: &str, worker_id: &str, url: &str) {
        self.urls.lock().unwrap_or_else(|p| p.into_inner()).insert(worker_id.to_owned(), url.to_owned());
        tracing::info!(pool, worker = worker_id, url, "autoscale: pod ready (it registers itself with the gateway)");
    }
    async fn drain(&self, _pool: &str, worker_id: &str) {
        self.post(worker_id, "/fv/v1/internal/drain").await;
    }
    async fn undrain(&self, _pool: &str, worker_id: &str) {
        self.post(worker_id, "/fv/v1/internal/undrain").await;
    }
    async fn deregister(&self, _pool: &str, worker_id: &str) {
        self.urls.lock().unwrap_or_else(|p| p.into_inner()).remove(worker_id);
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
            let n = r.get("running").and_then(Value::as_u64).unwrap_or(0) + r.get("sessions").and_then(Value::as_u64).unwrap_or(0);
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

/// Parses Runpod's `"2026-09-27 20:31:01.321 +0000 UTC"` (and RFC 3339
/// `2026-09-27T20:31:01Z`) into unix seconds.
pub fn parse_runpod_time(s: &str) -> Option<f64> {
    let s = s.trim();
    let (date, rest) = s.split_at_checked(10)?;
    let mut d = date.split('-').map(|x| x.parse::<i64>().ok());
    let (y, mo, da) = (d.next()??, d.next()??, d.next()??);
    let time = rest.trim_start_matches([' ', 'T']);
    let hms: String = time.chars().take_while(|c| c.is_ascii_digit() || *c == ':' || *c == '.').collect();
    let mut t = hms.split(':');
    let (h, mi) = (t.next()?.parse::<i64>().ok()?, t.next()?.parse::<i64>().ok()?);
    let sec: f64 = t.next()?.parse().ok()?;
    // Days from civil (Howard Hinnant).
    let y2 = if mo <= 2 { y - 1 } else { y };
    let era = y2.div_euclid(400);
    let yoe = y2 - era * 400;
    let doy = (153 * (mo + if mo > 2 { -3 } else { 9 }) + 2) / 5 + da - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    let days = era * 146_097 + doe - 719_468;
    Some((days * 86_400 + h * 3600 + mi * 60) as f64 + sec)
}

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
    pub fn new(api: RunpodApi, registry: Arc<dyn WorkerRegistry>, prices: BTreeMap<String, f64>) -> Result<Self, ProviderError> {
        let http = reqwest::Client::builder().timeout(Duration::from_secs(10)).build().map_err(|e| err("http client", e))?;
        Ok(Self { api, http, registry, prices, book: Mutex::new(PodBook::default()) })
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

    /// Pod create payloads in preference order: every GPU type in every placement.
    pub fn payloads(pool: &PoolConfig, name: &str) -> Vec<(String, Value)> {
        let mut v = Vec::new();
        for pl in &pool.pod.placements {
            for gpu in &pool.pod.gpu_types {
                v.push((
                    format!("{gpu} in {}", pl.data_center),
                    json!({
                        "name": name,
                        "templateId": pool.pod.template_id,
                        "computeType": "GPU",
                        "cloudType": pool.pod.cloud_type,
                        "gpuTypeIds": [gpu],
                        "gpuCount": 1,
                        "networkVolumeId": pl.volume_id,
                        "volumeMountPath": "/workspace",
                        "dataCenterIds": [pl.data_center],
                    }),
                ));
            }
        }
        v
    }

    async fn is_ready(&self, url: &str, path: &str) -> bool {
        matches!(self.http.get(format!("{url}{path}")).send().await, Ok(r) if r.status().as_u16() == 200)
    }

    async fn create_one(&self, pool: &PoolConfig, now_s: f64, r: &mut ApplyReport) -> Result<String, ProviderError> {
        let name = {
            let mut b = self.book();
            b.seq += 1;
            format!("{}{}{}", Self::prefix(&pool.name), now_s as u64 % 100_000_000, b.seq)
        };
        let mut last = None;
        for (label, body) in Self::payloads(pool, &name) {
            match self.api.create_pod(&body).await {
                Ok(v) => {
                    let Some(id) = v.get("id").and_then(Value::as_str).map(str::to_owned) else {
                        last = Some(ProviderError::Api(format!("create on {label}: no id")));
                        continue;
                    };
                    let dph = v.get("costPerHr").and_then(Value::as_f64).unwrap_or(0.0);
                    let gpu = body["gpuTypeIds"][0].as_str().unwrap_or("");
                    let cap = if pool.pod.max_usd_per_hr > 0.0 {
                        pool.pod.max_usd_per_hr
                    } else {
                        self.prices.get(gpu).map_or(f64::INFINITY, |p| p * 1.25)
                    };
                    if dph > cap {
                        r.notes.push(format!("{id} on {label} costs ${dph}/h > ${cap:.2}/h: deleted"));
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
        Err(last.unwrap_or_else(|| ProviderError::NoCapacity("no GPU type / placement configured".into())))
    }
}

#[async_trait::async_trait]
impl Provider for RunpodPods {
    fn name(&self) -> &'static str {
        "runpod-pods"
    }

    async fn observe(&self, pool: &PoolConfig, now_s: f64) -> Result<PoolObservation, ProviderError> {
        let prefix = Self::prefix(&pool.name);
        let pods: Vec<Value> = self
            .api
            .pods()
            .await?
            .into_iter()
            .filter(|p| p.get("name").and_then(Value::as_str).is_some_and(|n| n.starts_with(&prefix)))
            .filter(|p| p.get("desiredStatus").and_then(Value::as_str) != Some("TERMINATED"))
            .collect();
        let load = self.registry.in_flight(&pool.name).await;
        let mut workers = Vec::new();
        for p in pods {
            let Some(id) = p.get("id").and_then(Value::as_str).map(str::to_owned) else { continue };
            let url = Self::url(pool, &id);
            let created = self
                .book()
                .created_at
                .get(&id)
                .copied()
                .or_else(|| p.get("createdAt").and_then(Value::as_str).and_then(parse_runpod_time))
                .unwrap_or(now_s);
            let mut ready_at = self.book().ready_at.get(&id).copied();
            if ready_at.is_none() && self.is_ready(&url, &pool.pod.ready_path).await {
                ready_at = Some(now_s);
                self.book().ready_at.insert(id.clone(), now_s);
                self.registry.register(&pool.name, &id, &url).await;
                tracing::info!(pool = %pool.name, pod = %id, after_s = now_s - created, "autoscale: pod ready");
            }
            let busy = load.get(&id).or_else(|| load.get(&url)).copied().unwrap_or(0);
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
                gpu_type: p.pointer("/machine/gpuTypeId").and_then(Value::as_str).map(str::to_owned),
                usd_per_hr: p.get("costPerHr").and_then(Value::as_f64),
            });
        }
        Ok(PoolObservation { workers, endpoint: None })
    }

    async fn apply(&self, pool: &PoolConfig, d: &PoolDecision, now_s: f64) -> Result<ApplyReport, ProviderError> {
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
                let busy = load.get(id).or_else(|| load.get(&url)).copied().unwrap_or(0);
                if busy > 0 {
                    r.notes.push(format!("{id} got work since the decision: not deleted"));
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
        let cur = EndpointSettings { workers_min: 0, workers_max: 1, idle_timeout_s: 5, scaler_type: "QUEUE_DELAY".into(), scaler_value: 1 };
        let want = EndpointSettings { workers_min: 2, idle_timeout_s: 60, ..cur.clone() };
        let p = endpoint_patch(Some(&cur), &want);
        assert_eq!(Value::Object(p), json!({"workersMin": 2, "idleTimeout": 60}));
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
        let idle: Health = serde_json::from_str(r#"{"jobs":{"inProgress":0},"workers":{"running":2}}"#).unwrap();
        assert!(health_workers(&idle, 0.0).iter().all(|w| w.busy == 0));
        assert_eq!(w.iter().filter(|w| w.state == WorkerState::Starting).count(), 1);
    }

    #[test]
    fn runpod_times() {
        assert_eq!(parse_runpod_time("2026-09-28 00:00:00.000 +0000 UTC"), Some(1_790_553_600.0));
        assert_eq!(parse_runpod_time("2026-09-27T20:31:01Z"), Some(1_790_553_600.0 - 12_539.0));
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
                    Placement { data_center: "US-CA-2".into(), volume_id: "usrebuilt0".into() },
                    Placement { data_center: "EUR-IS-1".into(), volume_id: "jg48s6o1w0".into() },
                ],
                ..PodConfig::default()
            },
            ..PoolConfig::default()
        };
        let p = RunpodPods::payloads(&pool, "fv-as-sfwan-1");
        let order: Vec<&str> = p.iter().map(|(l, _)| l.as_str()).collect();
        assert_eq!(
            order,
            ["NVIDIA H100 80GB HBM3 in US-CA-2", "NVIDIA H200 in US-CA-2", "NVIDIA H100 80GB HBM3 in EUR-IS-1", "NVIDIA H200 in EUR-IS-1"]
        );
        let b = &p[2].1;
        assert_eq!(b["networkVolumeId"], "jg48s6o1w0");
        assert_eq!(b["templateId"], "tpl");
        assert!(b.get("allowedCudaVersions").is_none());
    }
}
