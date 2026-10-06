//! A worker that is also an API front behind the edge Worker
//! (`dispatch.front`, docs/serve/edge-control-plane.md §2.4).
//!
//! The edge authenticates the caller, picks a front of the request's model
//! family and forwards the request with its verdict (`x-fv-edge-auth`); the
//! front runs the API adapters as any fv-serve does (`auth.mode =
//! trust-edge`), and its engine seam is a [`FrontGate`]:
//!
//! - `submit` builds the dispatch envelope ([`envelope`]: inputs inline,
//!   passed through as the client's URL, or through the store) and enqueues
//!   it on the model's family object, which offers it to whichever worker
//!   has a slot (this one too); a background copy of the inputs that
//!   skipped the store replaces the envelope there, so a re-dispatch after a
//!   worker loss or a restage (a worker that cannot fetch a client URL) has
//!   them;
//! - `cancel` stops the job here when it runs here, else asks the family
//!   object (which tells the worker that holds it);
//! - the job store is the read-through [`store::FrontJobStore`]: the job
//!   row is inserted behind the enqueue, jobs this worker runs are answered
//!   from its own cache, others from D1, and `watch()` (SSE, sync waits) is
//!   woken by the family object's job wait ([`DoWaiter`]).
//!
//! [`front_info`] is what the worker announces to the edge (its hello's
//! `front`): the names that route to it, its fal apps, its protocols.
//! [`EdgeUploads`] fetches an upload another front holds (its token's tag
//! routes the edge to it).

pub mod envelope;
pub mod store;

use std::collections::{BTreeMap, HashMap};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

use fastvideo_dispatch_proto::{EnqueueReq, EnqueueResp, FrontInfo, JobWait, Scope, TOKEN_HEADER};
use fastvideo_protocol::{ApiError, Job, JobId, ModelCaps};
use fastvideo_serve_kit::{EngineGate, ServeCtx};

use crate::config::{Config, DispatchCfg};
use crate::gate::ServiceGate;
use envelope::Envelope;

/// The family of any model of the fleet: this front's own map for its
/// models, the edge's registry for the others (a cancel or a status wait
/// of a job another family runs).
pub struct Families {
    http: reqwest::Client,
    do_url: String,
    token: String,
    dispatch: DispatchCfg,
    /// This front's own models.
    local: Vec<String>,
    cache: Mutex<(Option<Instant>, BTreeMap<String, String>)>,
}

impl Families {
    pub fn new(cfg: &FrontCfg, local: Vec<String>) -> Arc<Self> {
        let http = reqwest::Client::builder().connect_timeout(Duration::from_secs(5)).build().unwrap_or_default();
        Arc::new(Self { http, do_url: cfg.do_url.clone(), token: cfg.token.clone(), dispatch: cfg.dispatch.clone(), local, cache: Mutex::default() })
    }

    /// The family of `model` (an id, or any name the fronts announce).
    pub async fn of(&self, model: &str) -> Option<String> {
        if self.local.iter().any(|m| m == model) || self.dispatch.model_families.contains_key(model) {
            return self.dispatch.family_of(model).map(str::to_owned);
        }
        {
            let g = self.cache.lock().unwrap_or_else(|p| p.into_inner());
            if let Some(f) = g.1.get(model) {
                return Some(f.clone());
            }
            if g.0.is_some_and(|t| t.elapsed() < Duration::from_secs(5)) {
                return None;
            }
        }
        let r = self.http.get(format!("{}/registry", self.do_url)).header(TOKEN_HEADER, &self.token).timeout(Duration::from_secs(10)).send().await.ok()?;
        let reg: fastvideo_dispatch_proto::front::Registry = r.json().await.ok()?;
        let mut map = BTreeMap::new();
        for x in reg.fronts() {
            for (name, id) in &x.info.names {
                map.entry(id.clone()).or_insert_with(|| x.family.to_owned());
                map.entry(name.clone()).or_insert_with(|| x.family.to_owned());
            }
        }
        let f = map.get(model).cloned();
        *self.cache.lock().unwrap_or_else(|p| p.into_inner()) = (Some(Instant::now()), map);
        f
    }
}

/// The settings a front needs.
#[derive(Clone, Debug)]
pub struct FrontCfg {
    pub do_url: String,
    pub token: String,
    pub dispatch: DispatchCfg,
    /// Signed-URL lifetime of inputs that go through the store.
    pub input_url_ttl: Duration,
}

impl FrontCfg {
    pub fn from_config(c: &Config) -> Option<Self> {
        let do_url = c.dispatch.do_url.clone()?;
        c.dispatch.front.then(|| Self {
            do_url: do_url.trim_end_matches('/').to_owned(),
            token: c.gateway.internal_token.expose().to_owned(),
            dispatch: c.dispatch.clone(),
            input_url_ttl: Duration::from_secs(6 * 3600),
        })
    }
}

/// The front's engine seam (see the module docs).
pub struct FrontGate {
    local: Arc<ServiceGate>,
    cfg: FrontCfg,
    http: reqwest::Client,
    ctx: OnceLock<ServeCtx>,
    me: std::sync::Weak<FrontGate>,
    families: Arc<Families>,
}

impl std::fmt::Debug for FrontGate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FrontGate").field("do_url", &self.cfg.do_url).finish_non_exhaustive()
    }
}

impl FrontGate {
    /// The family resolver (shared with the job store's waiter).
    pub fn families(&self) -> Arc<Families> {
        self.families.clone()
    }

    pub fn new(local: Arc<ServiceGate>, cfg: FrontCfg) -> Arc<Self> {
        let http = reqwest::Client::builder().connect_timeout(Duration::from_secs(5)).build().unwrap_or_default();
        let own: Vec<String> = local.models().iter().map(|m| m.id.to_string()).collect();
        let families = Families::new(&cfg, own);
        Arc::new_cyclic(|me| Self { local, cfg, http, ctx: OnceLock::new(), me: me.clone(), families })
    }

    pub fn attach(&self, ctx: ServeCtx) {
        let _ = self.ctx.set(ctx);
    }

    fn ctx(&self) -> Result<&ServeCtx, ApiError> {
        self.ctx.get().ok_or_else(|| ApiError::internal("the front is not attached"))
    }

    fn scope_of(&self, model: &str) -> Result<Scope, ApiError> {
        self.cfg
            .dispatch
            .family_of(model)
            .map(|f| Scope::Family(f.to_owned()))
            .ok_or_else(|| ApiError::not_found(format!("model `{model}` has no dispatch family on this front (dispatch.model_families)")))
    }

    fn req(&self, m: reqwest::Method, path: &str) -> reqwest::RequestBuilder {
        self.http.request(m, format!("{}{path}", self.cfg.do_url)).header(TOKEN_HEADER, &self.cfg.token)
    }

    async fn enqueue(&self, scope: &Scope, body: &EnqueueReq) -> Result<EnqueueResp, ApiError> {
        let r = self
            .req(reqwest::Method::POST, &scope.enqueue_path())
            .timeout(Duration::from_secs(30))
            .json(body)
            .send()
            .await
            .map_err(|e| ApiError::loading(format!("the dispatcher is unreachable: {}", e.without_url())).with_retry_after(5))?;
        let status = r.status();
        if !status.is_success() {
            let text = r.text().await.unwrap_or_default();
            tracing::warn!(job = %body.job_id, %status, body = %text.chars().take(300).collect::<String>(), "front: enqueue refused");
            return Err(match status.as_u16() {
                429 => ApiError::queue_full("the queue for this model is full").with_retry_after(10),
                400 => ApiError::invalid("the dispatcher refused the job"),
                _ => ApiError::loading("the dispatcher could not take the job").with_retry_after(5),
            });
        }
        let resp: EnqueueResp = r.json().await.map_err(|e| ApiError::internal(format!("the dispatcher's answer: {}", e.without_url())))?;
        if resp.state == "refused" {
            return Err(ApiError::queue_full(format!("{} jobs of this model are queued (max {})", resp.position, body.max_queued)).with_retry_after(10));
        }
        Ok(resp)
    }

    /// Copies the inputs that skipped the store into it and replaces the
    /// envelope on the family object (for a re-dispatch or a restage).
    async fn stage_copies(self: Arc<Self>, scope: Scope, mut req: EnqueueReq, env: Envelope) {
        let Ok(ctx) = self.ctx().cloned() else { return };
        let t0 = Instant::now();
        let inputs = match envelope::store_copies(&ctx, self.cfg.input_url_ttl, &env.inputs).await {
            Ok(i) => i,
            Err(e) => {
                tracing::warn!(job = %env.job.id, error = %e.message, "front: copying inputs to the store failed (a re-dispatch would lack them)");
                return;
            }
        };
        let env = Envelope { inputs, ..env };
        let Ok(v) = serde_json::to_value(&env) else { return };
        req.envelope = v;
        req.replace = true;
        match self.enqueue(&scope, &req).await {
            Ok(_) => tracing::debug!(job = %env.job.id, ms = t0.elapsed().as_millis() as u64, "front: envelope with stored inputs on the dispatcher"),
            Err(e) => tracing::warn!(job = %env.job.id, error = %e.message, "front: replacing the envelope failed"),
        }
    }
}

#[async_trait::async_trait]
impl EngineGate for FrontGate {
    fn models(&self) -> Vec<ModelCaps> {
        self.local.models()
    }

    fn alias(&self, name: &str) -> Option<String> {
        self.local.alias(name)
    }

    fn admit(&self) -> Result<(), ApiError> {
        // A front takes requests while its own models load: the job runs
        // wherever the family object places it.
        if self.local.admitting() {
            Ok(())
        } else {
            Err(ApiError::loading("this front is shutting down"))
        }
    }

    async fn submit(&self, job: &Job) -> Result<(), ApiError> {
        let t0 = Instant::now();
        let ctx = self.ctx()?;
        let model = job.resolved.model.to_string();
        let scope = self.scope_of(&model)?;
        let d = &self.cfg.dispatch;
        let inputs = envelope::plan_inputs(ctx, job, d.inline_inputs_max_bytes, d.input_passthrough, self.cfg.input_url_ttl).await?;
        let stage_s = t0.elapsed().as_secs_f64();
        let env = Envelope { job: job.clone(), inputs, attempt: 1, pool: Some(scope.id().to_owned()) };
        let body = EnqueueReq {
            job_id: job.id.to_string(),
            envelope: serde_json::to_value(&env).map_err(|e| ApiError::internal(format!("encoding the envelope: {e}")))?,
            retries: d.retries,
            model: Some(model),
            replace: false,
            owner: job.owner.as_ref().map(|o| o.0.clone()),
            max_queued: d.max_queued,
        };
        let t1 = Instant::now();
        let resp = self.enqueue(&scope, &body).await?;
        let enqueue_s = t1.elapsed().as_secs_f64();
        let family = scope.id().to_owned();
        for (phase, v) in [("stage_inputs", stage_s), ("enqueue", enqueue_s)] {
            metrics::histogram!("fv_front_submit_phase_seconds", "family" => family.clone(), "phase" => phase).record(v);
        }
        let via = |w: &str| env.inputs.iter().filter(|i| i.via() == w).count();
        tracing::info!(job = %job.id, family = %family, state = %resp.state, worker = ?resp.worker, inline = via("inline"), source = via("source"),
            store = via("store"), stage_inputs_ms = (stage_s * 1e3) as u64, enqueue_ms = (enqueue_s * 1e3) as u64, "front: enqueued");
        if d.retries > 0 && env.inputs.iter().any(|i| i.artifact.is_none()) {
            if let Some(me) = self.me.upgrade() {
                tokio::spawn(me.stage_copies(scope, body, env));
            }
        }
        Ok(())
    }

    async fn cancel(&self, id: JobId) -> bool {
        // Running here: stop it here (a cancel from the family object
        // arrives the same way, so no loop).
        if self.local.cancel(id).await {
            return true;
        }
        let Ok(ctx) = self.ctx() else { return false };
        let Some(job) = ctx.jobs().get(id).await else { return false };
        let Some(scope) = self.families.of(job.resolved.model.as_str()).await.map(Scope::Family) else { return false };
        match self.req(reqwest::Method::POST, &scope.cancel_path(&id.to_string())).timeout(Duration::from_secs(10)).send().await {
            Ok(r) if r.status().is_success() => true,
            Ok(r) => {
                tracing::info!(job = %id, status = %r.status(), "front: the dispatcher does not know the job");
                false
            }
            Err(e) => {
                tracing::warn!(job = %id, error = %e.without_url(), "front: cancel through the dispatcher failed");
                false
            }
        }
    }
}

/// Wakes a front's `watch()` pollers on the family object's job changes
/// (`GET {family}/jobs/{id}/wait`).
pub struct DoWaiter {
    http: reqwest::Client,
    do_url: String,
    token: String,
    families: Arc<Families>,
    /// The phase each job last had there.
    seen: Mutex<HashMap<JobId, String>>,
}

impl DoWaiter {
    pub fn new(cfg: &FrontCfg, families: Arc<Families>) -> Arc<Self> {
        let http = reqwest::Client::builder().connect_timeout(Duration::from_secs(5)).build().unwrap_or_default();
        Arc::new(Self { http, do_url: cfg.do_url.clone(), token: cfg.token.clone(), families, seen: Mutex::default() })
    }
}

#[async_trait::async_trait]
impl store::JobWaiter for DoWaiter {
    async fn wait(&self, job: &Job) {
        let Some(f) = self.families.of(job.resolved.model.as_str()).await else {
            return std::future::pending().await;
        };
        let since = self.seen.lock().unwrap_or_else(|p| p.into_inner()).get(&job.id).cloned().unwrap_or_default();
        let url = format!("{}{}?since={since}&wait_ms=25000", self.do_url, Scope::Family(f).job_wait_path(&job.id.to_string()));
        let r = self.http.get(url).header(TOKEN_HEADER, &self.token).timeout(Duration::from_secs(35)).send().await;
        match r {
            Ok(r) if r.status().is_success() => {
                if let Ok(w) = r.json::<JobWait>().await {
                    let unknown = {
                        let mut g = self.seen.lock().unwrap_or_else(|p| p.into_inner());
                        if matches!(w.phase.as_str(), "succeeded" | "failed" | "cancelled" | "unknown") {
                            g.remove(&job.id);
                        } else {
                            g.insert(job.id, w.phase.clone());
                        }
                        w.phase == "unknown"
                    };
                    if unknown {
                        // Not there (finished long ago, or not yet): the
                        // regular poll covers it.
                        std::future::pending::<()>().await;
                    }
                }
            }
            // An older dispatcher, or unreachable: polling only.
            _ => std::future::pending().await,
        }
    }
}

/// Fetches an upload another front holds through the edge
/// (`GET {edge}/fv/v1/internal/uploads/{token}`, routed by the token's tag).
pub struct EdgeUploads {
    http: reqwest::Client,
    do_url: String,
    token: String,
}

impl EdgeUploads {
    pub fn new(cfg: &FrontCfg) -> Arc<Self> {
        let http = reqwest::Client::builder().connect_timeout(Duration::from_secs(5)).build().unwrap_or_default();
        Arc::new(Self { http, do_url: cfg.do_url.clone(), token: cfg.token.clone() })
    }
}

#[async_trait::async_trait]
impl fastvideo_serve_kit::RemoteUploads for EdgeUploads {
    async fn fetch(&self, token: &str, dst: &std::path::Path, max_bytes: u64) -> Result<(u64, Option<String>), String> {
        use futures::StreamExt;
        use tokio::io::AsyncWriteExt;
        if fastvideo_dispatch_proto::front::token_tag(token).is_none() {
            return Err("not an upload of another front".into());
        }
        let r = self
            .http
            .get(format!("{}/fv/v1/internal/uploads/{token}", self.do_url))
            .header(TOKEN_HEADER, &self.token)
            .timeout(Duration::from_secs(300))
            .send()
            .await
            .map_err(|e| e.without_url().to_string())?;
        if !r.status().is_success() {
            return Err(format!("the edge answered {}", r.status()));
        }
        let mime = r.headers().get("content-type").and_then(|v| v.to_str().ok()).map(str::to_owned);
        let mut f = tokio::fs::File::create(dst).await.map_err(|e| e.to_string())?;
        let mut n = 0u64;
        let mut body = r.bytes_stream();
        while let Some(chunk) = body.next().await {
            let chunk = chunk.map_err(|e| e.without_url().to_string())?;
            n += chunk.len() as u64;
            if n > max_bytes {
                return Err(format!("the upload exceeds {max_bytes} bytes"));
            }
            f.write_all(&chunk).await.map_err(|e| e.to_string())?;
        }
        f.flush().await.map_err(|e| e.to_string())?;
        Ok((n, mime))
    }
}

/// What this worker announces to the edge (its hello's `front`).
pub fn front_info(config: &Config, gate: &ServiceGate, url: &str, worker_id: &str) -> FrontInfo {
    let engine = gate.engine();
    let caps = engine.caps();
    let failed = crate::health::failed_models(&engine.pool());
    let local: Vec<&ModelCaps> = caps.models().filter(|m| !failed.contains_key(m.id.as_str())).collect();
    let mut names: BTreeMap<String, String> = BTreeMap::new();
    for m in &local {
        names.insert(m.id.to_string(), m.id.to_string());
        for n in &m.served_names {
            names.entry(n.clone()).or_insert_with(|| m.id.to_string());
        }
    }
    for (alias, model) in caps.tier_aliases() {
        if local.iter().any(|m| m.id == model) {
            names.entry(alias).or_insert_with(|| model.to_string());
        }
    }
    for (alias, target) in &config.aliases {
        if let Some(m) = caps.resolve(target).filter(|m| local.iter().any(|l| l.id == m.id)) {
            names.entry(alias.clone()).or_insert_with(|| m.id.to_string());
        }
    }
    // The APIs' own model names (MiniMax, LTX) by the tier they select:
    // routed here when this worker serves that tier.
    for (api, tiers) in [
        ("MiniMax-H3", &["h3-max", "h3-turbo", "h3-draft"][..]),
        ("MiniMax-H3-Max", &["h3-max"][..]),
        ("MiniMax-H3-Turbo", &["h3-turbo"][..]),
        ("MiniMax-H3-Draft", &["h3-draft"][..]),
        ("ltx-2-5-pro", &["ltx-pro"][..]),
        ("ltx-2-3-pro", &["ltx-pro"][..]),
        ("ltx-2-5-fast", &["ltx-turbo"][..]),
        ("ltx-2-3-fast", &["ltx-turbo"][..]),
    ] {
        if let Some(m) = tiers.iter().find_map(|t| names.get(*t).cloned()) {
            names.entry(api.to_owned()).or_insert(m);
        }
    }
    let p = &config.protocols;
    let mut protocols: Vec<&str> = vec!["serve"];
    for (on, name) in [
        (p.openai_videos && cfg!(feature = "openai-videos"), "openai_videos"),
        (p.fastwan && cfg!(feature = "openai-videos"), "fastwan"),
        (p.minimax && cfg!(feature = "minimax"), "minimax"),
        (p.fal && cfg!(feature = "fal"), "fal"),
        (p.fal && p.fal_director && cfg!(all(feature = "fal", feature = "webrtc")), "fal_director"),
        (p.ltx && cfg!(feature = "ltxapi"), "ltx"),
        (p.reactor && cfg!(feature = "reactor"), "reactor"),
        (p.native, "native"),
        (config.server.console, "console"),
    ] {
        if on {
            protocols.push(name);
        }
    }
    let fal_apps: Vec<String> = if protocols.contains(&"fal") {
        crate::adapters::fal_app_models(&p.fal_apps)
            .into_iter()
            .filter(|(_, model)| {
                let target = config.aliases.get(model).map(String::as_str).unwrap_or(model);
                caps.resolve(target).or_else(|| caps.resolve(model)).is_some_and(|m| local.iter().any(|l| l.id == m.id))
            })
            .map(|(app, _)| app)
            .collect()
    } else {
        Vec::new()
    };
    let first = local.first().map(|m| m.id.to_string());
    let defaults: BTreeMap<String, String> = match &first {
        Some(m) => protocols.iter().filter(|p| !matches!(**p, "serve" | "console")).map(|p| ((*p).to_owned(), m.clone())).collect(),
        None => BTreeMap::new(),
    };
    let reactor = if protocols.contains(&"reactor") {
        config
            .reactor
            .model
            .clone()
            .filter(|m| caps.resolve(m).is_some())
            .or_else(|| local.iter().find(|m| m.stream.is_some()).map(|m| m.id.to_string()))
    } else {
        None
    };
    FrontInfo {
        url: url.trim_end_matches('/').to_owned(),
        protocols: protocols.into_iter().map(str::to_owned).collect(),
        names,
        fal_apps,
        defaults,
        reactor,
        failed_models: failed,
        ready: matches!(engine.readiness(), fastvideo_engine_service::Readiness::Ready),
        tag: fastvideo_dispatch_proto::front::worker_tag(worker_id),
    }
}
