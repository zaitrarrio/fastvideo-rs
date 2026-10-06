//! Gateway mode (`[engine] backend = "remote"`, docs/serve/gateway.md): one
//! entry point for every API in front of one GPU worker pool per model
//! family.
//!
//! - [`Gateway`] implements serve-kit's `EngineGate` (the engine seam every
//!   adapter uses): `models()` are the pools' aggregated caps, `submit`
//!   dispatches the stored job to a pool ([`dispatch`]), `cancel` forwards.
//! - [`store::GatewayJobStore`]: D1 read-through job store (workers write).
//! - [`tick`]: pool probes, live caps, the worker-loss reaper, metrics and
//!   the [`scale::PoolScaler`] hooks.
//! - [`proxy`]: stream and peer-session signalling (fal director, Reactor,
//!   `/fv/v1/streams`) proxied to a worker; media flows client ↔ worker.
//! - [`routes`]: `/fv/v1/capabilities`, `/fv/v1/gateway/pools`, health.
//! - [`schema`]: the gateway's D1 tables; [`runpod`]: the queue API client.

pub mod dispatch;
pub mod edge;
pub mod proxy;
pub mod routes;
pub mod runpod;
pub mod scale;
pub mod schema;
pub use crate::front::store;
pub mod tick;

use std::collections::{BTreeMap, HashMap};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock, RwLock};
use std::time::{Duration, Instant};

use anyhow::anyhow;
use fastvideo_engine_service::{CapabilityTable, FakeBackend, FakeConfig, Recipe};
use fastvideo_engine_service::EngineBackend;
use fastvideo_protocol::{ApiError, Job, JobId, ModelCaps, ModelId};
use fastvideo_serve_kit::d1::D1Client;
use fastvideo_serve_kit::{EngineGate, ServeCtx};
use serde::Serialize;
use serde_json::Value;

use crate::config::{Config, GatewayCfg, PoolCfg, PoolKind};
use scale::{PoolMetrics, PoolScaler};
use store::GatewayJobStore;

/// Header carrying the internal token on gateway → worker requests.
pub const TOKEN_HEADER: &str = "x-fv-internal-token";

/// A worker's build as its `/fv/v1/internal/status` reports it
/// ([`crate::build_info`]; docs/serve/releases.md). Old workers report
/// none.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize)]
pub struct WorkerBuild {
    pub version: Option<String>,
    pub git_sha: Option<String>,
    pub build_time: Option<String>,
    pub variant: Option<String>,
    pub channel: Option<String>,
    pub image_digest: Option<String>,
    pub image_tag: Option<String>,
}

impl WorkerBuild {
    /// From the status body's `build` object (and `version`).
    pub fn parse(status: &Value) -> Option<Self> {
        let b = status.get("build").filter(|b| b.is_object())?;
        let s = |p: &str| b.pointer(p).and_then(Value::as_str).filter(|v| !v.is_empty()).map(str::to_owned);
        Some(Self {
            version: s("/version").or_else(|| status.get("version").and_then(Value::as_str).map(str::to_owned)),
            git_sha: s("/git_sha"),
            build_time: s("/build_time"),
            variant: s("/variant"),
            channel: s("/channel"),
            image_digest: s("/image/digest"),
            image_tag: s("/image/tag"),
        })
    }

    /// The short sha the public views show (`unknown` when not reported).
    pub fn short_sha(b: Option<&Self>) -> String {
        b.and_then(|b| b.git_sha.as_deref()).filter(|s| *s != "unknown").map(|s| s.chars().take(7).collect()).unwrap_or_else(|| "unknown".into())
    }
}

/// One worker of a pod pool as last seen.
#[derive(Clone, Debug, Default, Serialize)]
pub struct WorkerView {
    pub url: String,
    pub id: Option<String>,
    /// Answered its last probe.
    pub healthy: bool,
    /// Models loaded (`readiness: ready`).
    pub ready: bool,
    pub draining: bool,
    pub running: u32,
    pub queued: u32,
    pub sessions: u32,
    /// From `gw_workers` (self-registered) rather than `urls`.
    pub registered: bool,
    /// Jobs it runs at once (its executors; 1 when it does not say).
    pub capacity: u32,
    /// Its batch queue limit (0: none, or not reported).
    pub queue_max: u32,
    /// Dispatch calls from this replica to it now in progress: reserved
    /// when the worker is picked, released when the call returns.
    pub reserved: u32,
    /// Jobs this replica placed here that `running` / `queued` may not
    /// count yet (see [`WorkerView::unreported`]).
    pub unreported: u32,
    /// The placements behind `unreported`: (job, when the worker took it).
    #[serde(skip)]
    pub placed: Vec<(JobId, Instant)>,
    /// `running` / `queued` were computed by the worker after this instant
    /// (the probe or dispatch call that brought them was sent then).
    #[serde(skip)]
    pub load_at: Option<Instant>,
    /// The job whose dispatch answer brought `running` / `queued` (it
    /// counts it although it was taken after `load_at`).
    #[serde(skip)]
    pub load_includes: Option<JobId>,
    pub last_error: Option<String>,
    /// The last failed probe got an HTTP answer (not a connection error).
    pub answered: bool,
    /// The last failed probe hit a worker still starting (503 `loading`
    /// before its routes exist; its `/ping` answers 204).
    pub starting: bool,
    /// When a probe last succeeded.
    #[serde(skip)]
    pub last_ok: Option<Instant>,
    /// Its build, from the last successful probe (admin view only; the
    /// public views show the short sha and channel per pool).
    pub build: Option<WorkerBuild>,
    /// Reports `readiness: failed` (e.g. a GPU that cannot run its model):
    /// shown as `failed` and never dispatched to.
    pub failed: bool,
    /// Its failed models and why (from its internal status).
    pub failed_models: BTreeMap<String, String>,
}

impl WorkerView {
    fn usable(&self) -> bool {
        self.healthy && !self.draining && !self.failed
    }
    /// Has a base URL the gateway can call (not a worker known only
    /// through a Durable Object, `do:<id>`).
    pub fn dialable(&self) -> bool {
        !self.url.starts_with("do:")
    }
    /// Its load as the gateway places against it: what the worker last
    /// reported (running + queued), the jobs placed here since that report,
    /// and the dispatch calls in progress.
    pub fn load(&self) -> u32 {
        self.running + self.queued + self.unreported_now() + self.reserved
    }
    /// Placements its last report cannot include: taken after the report
    /// was computed, other than the job whose answer brought it.
    fn unreported_now(&self) -> u32 {
        self.placed.iter().filter(|(j, at)| self.load_at.is_none_or(|t| *at > t) && Some(*j) != self.load_includes).count() as u32
    }
    /// Takes a load report computed after `at` (older reports are ignored:
    /// a probe answered before a later dispatch's answer); placements the
    /// report surely counts are forgotten.
    pub(crate) fn report_load(&mut self, at: Instant, running: u32, queued: u32, includes: Option<JobId>) {
        if self.load_at.is_some_and(|t| t > at) {
            return;
        }
        self.running = running;
        self.queued = queued;
        self.load_at = Some(at);
        self.load_includes = includes;
        self.placed.retain(|(_, took)| *took > at);
        self.unreported = self.unreported_now();
    }
    /// Job-times a new job would wait here: full rounds of its capacity
    /// ahead of it (0: a free slot).
    fn rounds(&self) -> u32 {
        self.load() / self.capacity.max(1)
    }
    /// Its queue is full (jobs beyond its run slots reach `queue_max`).
    fn queue_full(&self) -> bool {
        self.queue_max > 0 && self.load().saturating_sub(self.capacity.max(1)) >= self.queue_max
    }
    /// The public status word (see [`crate::status`]).
    pub fn state(&self) -> crate::status::State {
        use crate::status::State as S;
        if self.healthy {
            if self.draining {
                S::Draining
            } else if self.failed {
                S::Failed
            } else if !self.ready {
                S::Loading
            } else if self.load() > 0 {
                S::Busy
            } else {
                S::Ready
            }
        } else if self.starting {
            S::Loading
        } else if self.answered {
            S::Unhealthy
        } else {
            S::Down
        }
    }
}

/// Mutable view of a pool.
#[derive(Debug, Default)]
pub struct PoolState {
    pub live_caps: Option<Vec<(ModelCaps, Recipe)>>,
    pub caps_at: Option<Instant>,
    pub workers: BTreeMap<String, WorkerView>,
    /// Serverless: the last Runpod `/health` body.
    pub health: Option<Value>,
    /// Serverless: when Runpod `/health` last answered.
    pub health_at: Option<Instant>,
    pub available: bool,
    pub last_error: Option<String>,
    /// From the last tick (D1).
    pub queued: u32,
    pub running: u32,
    pub streams: u32,
    /// Dispatches from this replica the last tick's D1 count may miss:
    /// being recorded now, and recorded (at these instants) after it.
    pub recording: u32,
    pub recorded: Vec<Instant>,
    pub metrics: Option<PoolMetrics>,
    /// Orders equally loaded workers differently on each replica (so two
    /// replicas' bursts do not start on the same worker).
    pub tie_seed: u64,
}

impl PoolState {
    /// Jobs dispatched from this replica that `queued` does not count yet.
    pub fn pending(&self) -> u32 {
        self.recording + self.recorded.len() as u32
    }

    /// Picks the worker for a job and reserves a slot on it, under the pool
    /// lock (concurrent submits see each other's picks): among usable,
    /// dialable workers not in `skip` whose queue is not full, ready ones
    /// first, then the fewest job-times of wait (free slot first), then the
    /// lowest load. With no free slot anywhere the job queues on the worker
    /// where it starts soonest; the gateway holds no queue of its own.
    pub(crate) fn pick(&mut self, skip: &[String]) -> Result<String, NoWorker> {
        let seed = self.tie_seed;
        let tie = |url: &str| {
            use std::hash::{Hash, Hasher};
            let mut h = std::collections::hash_map::DefaultHasher::new();
            (seed, url).hash(&mut h);
            h.finish()
        };
        let mut full = false;
        let mut best: Option<((bool, u32, u32, u64), String)> = None;
        for w in self.workers.values() {
            if !w.usable() || !w.dialable() || skip.contains(&w.url) {
                continue;
            }
            if w.queue_full() {
                full = true;
                continue;
            }
            let key = (!w.ready, w.rounds(), w.load(), tie(&w.url));
            if best.as_ref().is_none_or(|(k, _)| key < *k) {
                best = Some((key, w.url.clone()));
            }
        }
        let Some((_, url)) = best else {
            return Err(if full { NoWorker::QueueFull } else { NoWorker::None });
        };
        if let Some(w) = self.workers.get_mut(&url) {
            w.reserved += 1;
        }
        Ok(url)
    }
}

/// Why [`PoolState::pick`] found no worker.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NoWorker {
    /// None is usable (down, draining, failed, or already tried).
    None,
    /// Every usable one has a full queue.
    QueueFull,
}

/// A slot reserved on a pod worker for one dispatch call; dropping it
/// (a refusal, an error, a cancelled submit) releases it, [`Reservation::placed`]
/// turns it into a placement.
pub(crate) struct Reservation<'a> {
    pool: &'a Pool,
    pub url: String,
    open: bool,
}

/// The worker's load right after it took a job (the dispatch answer).
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct TakeLoad {
    pub running: u32,
    pub queued: u32,
    pub capacity: u32,
    pub queue_max: u32,
}

impl TakeLoad {
    pub fn parse(v: &Value) -> Option<Self> {
        let l = v.get("load").filter(|l| l.is_object())?;
        let n = |k: &str| l.get(k).and_then(Value::as_u64).map(|x| x as u32);
        Some(Self { running: n("running")?, queued: n("queued")?, capacity: n("capacity").unwrap_or(1), queue_max: n("queue_max").unwrap_or(0) })
    }
}

impl<'a> Reservation<'a> {
    /// Reserves a slot on the best worker of `pool` (see [`PoolState::pick`]).
    pub fn take(pool: &'a Pool, skip: &[String]) -> Result<Self, NoWorker> {
        let url = pool.lock().pick(skip)?;
        Ok(Self { pool, url, open: true })
    }

    /// The worker took `job` (the call was sent at `sent`): the slot
    /// becomes a placement, counted until a report includes it; `load` is
    /// the worker's own count from its answer.
    pub fn placed(mut self, job: JobId, sent: Instant, load: Option<TakeLoad>) {
        self.open = false;
        let mut st = self.pool.lock();
        if let Some(w) = st.workers.get_mut(&self.url) {
            w.reserved = w.reserved.saturating_sub(1);
            w.placed.push((job, Instant::now()));
            if let Some(l) = load {
                w.capacity = l.capacity.max(1);
                w.queue_max = l.queue_max;
                w.report_load(sent, l.running, l.queued, Some(job));
            }
            w.unreported = w.unreported_now();
        }
    }
}

impl Drop for Reservation<'_> {
    fn drop(&mut self) {
        if self.open {
            if let Some(w) = self.pool.lock().workers.get_mut(&self.url) {
                w.reserved = w.reserved.saturating_sub(1);
            }
        }
    }
}

/// One `[[pools]]` entry at run time.
#[derive(Debug)]
pub struct Pool {
    pub idx: usize,
    pub cfg: PoolCfg,
    pub static_caps: Vec<(ModelCaps, Recipe)>,
    pub st: Mutex<PoolState>,
}

impl Pool {
    pub fn lock(&self) -> std::sync::MutexGuard<'_, PoolState> {
        self.st.lock().unwrap_or_else(|p| p.into_inner())
    }
    pub fn id(&self) -> &str {
        &self.cfg.id
    }
    pub fn is_pod(&self) -> bool {
        self.cfg.kind == PoolKind::Pod
    }
    /// The caps this pool advertises: live when fetched, else static.
    pub fn caps(&self) -> Vec<(ModelCaps, Recipe)> {
        self.lock().live_caps.clone().unwrap_or_else(|| self.static_caps.clone())
    }
    /// Whether a job would be dispatched now (serverless: the endpoint
    /// answers; pods: a usable worker).
    pub fn available(&self) -> bool {
        let st = self.lock();
        if self.is_pod() {
            st.workers.values().any(WorkerView::usable)
        } else {
            st.available
        }
    }
}

/// Every model of every pool: the gateway's `CapabilityTable` (one
/// "executor" per pool), which pools serve each model, and the alias map.
#[derive(Debug, Default)]
pub struct Catalog {
    pub table: CapabilityTable,
    pub pools_of: BTreeMap<ModelId, Vec<usize>>,
    pub aliases: BTreeMap<String, String>,
}

impl Catalog {
    /// First pool wins for a model two pools declare differently.
    pub fn build(pools: &[Pool], aliases: &BTreeMap<String, String>, overrides: &BTreeMap<String, ModelId>) -> Result<Self, ApiError> {
        let mut seen: BTreeMap<ModelId, (ModelCaps, Recipe)> = BTreeMap::new();
        let mut pools_of: BTreeMap<ModelId, Vec<usize>> = BTreeMap::new();
        let mut order = Vec::new();
        for p in pools {
            for (mut caps, recipe) in p.caps() {
                // The gateway view: every served model is reachable.
                caps.resident = true;
                pools_of.entry(caps.id.clone()).or_default().push(p.idx);
                if !seen.contains_key(&caps.id) {
                    order.push(caps.id.clone());
                    seen.insert(caps.id.clone(), (caps, recipe));
                }
            }
        }
        let list: Vec<(ModelCaps, Recipe)> = order.iter().filter_map(|id| seen.remove(id)).collect();
        let overrides: BTreeMap<String, ModelId> = overrides.iter().filter(|(_, m)| pools_of.contains_key(*m)).map(|(a, m)| (a.clone(), m.clone())).collect();
        let table = CapabilityTable::build(vec![list], &overrides)?;
        let mut all = aliases.clone();
        for (alias, model) in table.tier_aliases() {
            all.entry(alias).or_insert(model.0);
        }
        Ok(Self { table, pools_of, aliases: all })
    }
}

/// A stream / peer-session lease (docs/serve/gateway.md §5).
#[derive(Clone, Debug, Serialize)]
pub struct Lease {
    pub id: String,
    pub pool: String,
    /// `director` | `reactor` | `stream`.
    pub kind: String,
    /// Pod: the worker base URL; serverless: the endpoint id.
    pub target: String,
    /// Serverless: the Runpod job id.
    pub r#ref: Option<String>,
    pub owner: Option<String>,
    pub lease_key: Option<String>,
    pub state: String,
    pub created_at: i64,
    /// A family object's session id (its admission reserved the GPU;
    /// docs/serve/dispatch-do-family.md §8); kept in `gw_sessions.body`.
    pub do_session: Option<String>,
}

/// The gateway (see the module docs).
pub struct Gateway {
    pub cfg: GatewayCfg,
    pub pools: Vec<Pool>,
    pub http: reqwest::Client,
    pub db: D1Client,
    pub jobs: Arc<GatewayJobStore>,
    pub runpod: runpod::RunpodApi,
    token: String,
    catalog: RwLock<Arc<Catalog>>,
    base_aliases: BTreeMap<String, String>,
    tier_overrides: BTreeMap<String, ModelId>,
    ctx: OnceLock<ServeCtx>,
    admitting: AtomicBool,
    scalers: Mutex<Vec<Arc<dyn PoolScaler>>>,
    /// fal app id → model name (a tier alias or model id), for the director.
    pub fal_apps: Vec<(String, String)>,
    pub(crate) leases: Mutex<HashMap<String, Lease>>,
    /// Signed-URL lifetime for dispatched inputs.
    pub input_url_ttl: Duration,
}

impl std::fmt::Debug for Gateway {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Gateway").field("pools", &self.pools.iter().map(|p| p.id()).collect::<Vec<_>>()).finish_non_exhaustive()
    }
}

/// A pool's static caps from its config.
pub fn static_caps(p: &PoolCfg) -> anyhow::Result<Vec<(ModelCaps, Recipe)>> {
    let mut out = Vec::new();
    if !p.fake_models.is_empty() {
        let ids: Vec<&str> = p.fake_models.iter().map(String::as_str).collect();
        let fc = FakeConfig::default().with_models(&ids);
        if fc.models.len() != ids.len() {
            return Err(anyhow!("pools.{}: fake_models {:?} names an unknown fake model", p.id, p.fake_models));
        }
        let b = FakeBackend::new(fc);
        for c in b.caps() {
            let r = b.recipe(&c.id);
            out.push((c, r));
        }
    }
    if !p.models.is_empty() {
        for m in crate::app::catalog_models(&p.models).map_err(|e| anyhow!("pools.{}: {e}", p.id))? {
            out.push((m.caps(), m.describe()));
        }
    }
    Ok(out)
}

impl Gateway {
    /// Builds the gateway over `db` (the D1 of the job store) and `jobs`.
    pub async fn build(config: &Config, db: D1Client, jobs: Arc<GatewayJobStore>) -> anyhow::Result<Arc<Self>> {
        schema::migrate(&db).await.map_err(|e| anyhow!("gateway tables: {e}"))?;
        let http = reqwest::Client::builder()
            .connect_timeout(Duration::from_secs(5))
            .build()
            .map_err(|e| anyhow!("HTTP client: {e}"))?;
        let mut pools = Vec::new();
        for (idx, p) in config.pools.iter().enumerate() {
            let caps = static_caps(p)?;
            let tie_seed = u64::from_str_radix(&fastvideo_serve_kit::random_token()[..16], 16).unwrap_or(idx as u64);
            let mut st = PoolState { available: true, tie_seed, ..PoolState::default() };
            for u in &p.urls {
                let url = u.trim_end_matches('/').to_owned();
                st.workers.insert(url.clone(), WorkerView { url, healthy: true, ready: true, capacity: 1, ..WorkerView::default() });
            }
            pools.push(Pool { idx, cfg: p.clone(), static_caps: caps, st: Mutex::new(st) });
        }
        let mut base_aliases = config.aliases.clone();
        for p in &config.pools {
            for (a, m) in &p.aliases {
                base_aliases.entry(a.clone()).or_insert_with(|| m.clone());
            }
        }
        let tier_overrides: BTreeMap<String, ModelId> =
            config.engine.tier_overrides.iter().map(|(k, v)| (k.clone(), ModelId::new(v))).collect();
        let catalog = Catalog::build(&pools, &base_aliases, &tier_overrides).map_err(|e| anyhow!("pool caps: {e}"))?;
        let fal_apps = crate::adapters::fal_app_models(&config.protocols.fal_apps);
        let mut cfg = config.gateway.clone();
        // The Reactor routes' model: `[gateway] reactor_model`, else `[reactor] model`
        // (FV_REACTOR_MODEL), else the first stream-capable pod-pool model.
        if cfg.reactor_model.is_none() {
            cfg.reactor_model = config.reactor.model.clone();
        }
        let gw = Arc::new(Self {
            cfg,
            runpod: runpod::RunpodApi::new(http.clone(), &config.gateway.runpod_api_base, config.gateway.runpod_api_key.expose()),
            http,
            pools,
            db,
            jobs,
            token: config.gateway.internal_token.expose().to_owned(),
            catalog: RwLock::new(Arc::new(catalog)),
            base_aliases,
            tier_overrides,
            ctx: OnceLock::new(),
            admitting: AtomicBool::new(true),
            scalers: Mutex::new(Vec::new()),
            fal_apps,
            leases: Mutex::new(HashMap::new()),
            input_url_ttl: Duration::from_secs(6 * 3600),
        });
        tracing::info!(pools = ?gw.pools.iter().map(|p| (p.id().to_owned(), p.cfg.kind)).collect::<Vec<_>>(), models = gw.catalog().table.len(), "gateway: pools configured");
        edge::install(&gw);
        Ok(gw)
    }

    /// Connects the gateway to the context built around it (once).
    pub fn attach(&self, ctx: ServeCtx) {
        let _ = self.ctx.set(ctx);
    }

    pub(crate) fn ctx(&self) -> Result<&ServeCtx, ApiError> {
        self.ctx.get().ok_or_else(|| ApiError::internal("gateway is not attached"))
    }

    pub fn catalog(&self) -> Arc<Catalog> {
        self.catalog.read().unwrap_or_else(|p| p.into_inner()).clone()
    }

    /// Rebuilds the catalog from the pools' current caps.
    pub fn rebuild_catalog(&self) {
        match Catalog::build(&self.pools, &self.base_aliases, &self.tier_overrides) {
            Ok(c) => *self.catalog.write().unwrap_or_else(|p| p.into_inner()) = Arc::new(c),
            Err(e) => tracing::warn!(error = %e, "gateway: pool caps conflict; keeping the previous catalog"),
        }
    }

    pub fn token(&self) -> &str {
        &self.token
    }

    pub fn pool(&self, id: &str) -> Option<&Pool> {
        self.pools.iter().find(|p| p.id() == id)
    }

    /// Pools serving the model `name` resolves to (alias, tier alias, id or
    /// served name), in config order.
    pub fn pools_for_name(&self, name: &str) -> Option<(ModelId, Vec<usize>)> {
        let cat = self.catalog();
        let resolved = cat.aliases.get(name).cloned().unwrap_or_else(|| name.to_owned());
        let caps = cat.table.resolve(&resolved).or_else(|| cat.table.resolve(name))?;
        let pools = cat.pools_of.get(&caps.id).cloned().unwrap_or_default();
        Some((caps.id.clone(), pools))
    }

    pub fn stop_admission(&self) {
        self.admitting.store(false, Ordering::SeqCst);
    }

    pub fn admitting(&self) -> bool {
        self.admitting.load(Ordering::SeqCst)
    }

    /// Registers an autoscaler hook (docs/serve/gateway.md §7).
    pub fn add_scaler(&self, s: Arc<dyn PoolScaler>) {
        self.scalers.lock().unwrap_or_else(|p| p.into_inner()).push(s);
    }

    pub(crate) fn scalers(&self) -> Vec<Arc<dyn PoolScaler>> {
        self.scalers.lock().unwrap_or_else(|p| p.into_inner()).clone()
    }

    /// The last metrics of every pool (one per pool once the first tick ran).
    pub fn metrics(&self) -> Vec<PoolMetrics> {
        self.pools.iter().filter_map(|p| p.lock().metrics.clone()).collect()
    }

    /// Whether any pool can take work now.
    pub fn any_available(&self) -> bool {
        self.pools.iter().any(Pool::available)
    }

    /// A request to a worker with the internal token.
    pub(crate) fn worker_req(&self, m: reqwest::Method, url: &str) -> reqwest::RequestBuilder {
        self.http.request(m, url).header(TOKEN_HEADER, &self.token)
    }

    /// Starts the tick loop (probes, reaper, metrics).
    pub fn spawn_tick(self: &Arc<Self>) -> tokio::task::JoinHandle<()> {
        let weak = Arc::downgrade(self);
        let every = Duration::from_secs(self.cfg.tick_s.max(1));
        tokio::spawn(async move {
            let mut t = tokio::time::interval(every);
            t.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
            loop {
                t.tick().await;
                let Some(gw) = weak.upgrade() else { return };
                gw.tick().await;
            }
        })
    }
}

/// The gateway's router: serve-kit files/uploads, the native API (with the
/// aggregated capabilities and the stream proxy), every adapter over the
/// gateway's `ServeCtx`, the director and Reactor signalling proxies, the
/// admin key routes, health and pool metrics, the console; then metrics,
/// the multi-worker filter (gateway policy) and tracing.
pub fn assemble(
    config: &Config,
    ctx: &ServeCtx,
    gw: &Arc<Gateway>,
    admin: Arc<fastvideo_serve_kit::AdminToken>,
    keys: Arc<fastvideo_serve_kit::KeyStore>,
) -> axum::Router {
    use axum::Router;
    let mcfg = crate::app::mount_cfg(config);
    let mut kit: Router<ServeCtx> = ctx.routes();
    if config.protocols.native {
        let g = gw.clone();
        kit = kit.merge(crate::native::routes_with(Arc::new(move || routes::capabilities(&g)), mcfg.body_max));
        kit = kit.merge(proxy::stream_routes(gw.clone()));
    }
    let fal_extra: Router<ServeCtx> = if config.protocols.fal && config.protocols.fal_director {
        proxy::director_routes(gw.clone())
    } else {
        Router::new()
    };
    let (adapters, stateful) = crate::adapters::mount(&mcfg, ctx, fal_extra);
    kit = kit.merge(adapters);
    let mut r = kit
        .with_state(ctx.clone())
        .merge(stateful)
        .merge(fastvideo_serve_kit::admin_routes(keys, admin.clone()))
        .merge(crate::releases::routes(gw.clone(), admin.clone(), crate::releases::ReleasesCfg::from_env(&crate::config::ProcessEnv)))
        .merge(routes::routes(gw.clone(), admin, crate::metrics::install(), crate::adapters::root_model(&mcfg, ctx)));
    if config.protocols.reactor {
        r = r.merge(proxy::reactor_routes(gw.clone(), ctx.clone()));
    }
    if config.server.console {
        r = r.merge(crate::console::routes());
    }
    let r = r.route_layer(axum::middleware::from_fn(crate::metrics::track));
    let policy = crate::multiworker::Policy::from_config(config, "d1");
    if policy.multi() {
        tracing::info!(workers_max = policy.workers_max, "gateway replicas: every route but uploads is served by any replica");
    }
    let r = crate::multiworker::layer(r, policy, &config.protocols.fal_apps).layer(tower_http::trace::TraceLayer::new_for_http());
    // The gateway is the public front: the same CORS as a standalone server.
    match crate::app::cors_layer(&config.server.cors_origins) {
        Some(cors) => r.layer(cors),
        None => r,
    }
}

#[async_trait::async_trait]
impl EngineGate for Gateway {
    fn models(&self) -> Vec<ModelCaps> {
        self.catalog().table.models().cloned().collect()
    }

    fn alias(&self, name: &str) -> Option<String> {
        self.catalog().aliases.get(name).cloned()
    }

    fn admit(&self) -> Result<(), ApiError> {
        if self.admitting() {
            Ok(())
        } else {
            Err(ApiError::loading("the gateway is shutting down"))
        }
    }

    async fn submit(&self, job: &Job) -> Result<(), ApiError> {
        self.submit_job(job).await
    }

    async fn cancel(&self, id: JobId) -> bool {
        self.cancel_job(id).await
    }
}
