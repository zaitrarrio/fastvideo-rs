//! Assembly: engine, stores, `ServeCtx`, routers; serving and draining.

use std::collections::BTreeMap;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{anyhow, Context};
use axum::Router;
use fastvideo_engine_service::{
    EngineBackend, EngineConfig, EngineService, FakeBackend, FakeConfig, FakeTiming, Mp4Mode, Readiness,
};
use fastvideo_media::video::FfmpegH264;
use fastvideo_protocol::{JobStore, ModelId, ProtocolId};
use fastvideo_serve_kit::store::spawn_sweeper;
use fastvideo_serve_kit::{
    Auth, CallbackSender, KeyRing, KeyStore, ServeConfig, ServeCtx, UrlKey, WebhookSigner,
};
use tokio::net::TcpListener;
use tower_http::trace::TraceLayer;
use url::Url;

use crate::adapters::{self, MountCfg};
use crate::config::{ArtifactBackend, Config, EngineBackendKind, JobBackend, Mode};
use crate::gate::{OutputPolicy, ServiceGate};
use crate::health::{self, Health};
use crate::{console, metrics, native, storage};

/// Parts tests (or other front ends) substitute.
#[derive(Default)]
pub struct Overrides {
    /// Use this job store instead of the configured one.
    pub jobs: Option<Arc<dyn JobStore>>,
    /// Use this engine instead of building one from `[engine]`.
    pub engine: Option<EngineService>,
    /// Use this D1 connection for the D1 job store (and the gateway's
    /// tables) instead of the HTTP API from `[jobs.d1]` (tests: the mock).
    pub d1: Option<fastvideo_serve_kit::D1Client>,
    /// Use this artifact store instead of the configured one (tests: a
    /// store with R2-like latency).
    pub artifacts: Option<Arc<dyn fastvideo_serve_kit::ArtifactStore>>,
    /// Autoscaler hooks for gateway mode (docs/serve/gateway.md §7).
    #[cfg(feature = "http-client")]
    pub scalers: Vec<Arc<dyn crate::gateway::scale::PoolScaler>>,
}

/// A built server.
pub struct App {
    pub config: Config,
    pub ctx: ServeCtx,
    pub gate: Arc<ServiceGate>,
    pub router: Router,
    pub d1: Option<Arc<fastvideo_serve_kit::D1JobStore>>,
    /// Minted API keys (`/fv/v1/admin/keys`).
    pub keys: Arc<KeyStore>,
    /// Whether the admin token is the server's own (`<state_dir>/admin_token`
    /// or, on a worker, random) rather than `FV_ADMIN_TOKEN`.
    pub admin_token_generated: bool,
    /// Where the admin token came from.
    pub admin_token_source: crate::admin_token::Source,
    /// The Reactor local runtime, when mounted (feature `reactor`).
    #[cfg(feature = "reactor")]
    pub reactor: Option<fastvideo_reactor::Reactor>,
    /// Gateway mode (`engine.backend = "remote"`): the pools.
    #[cfg(feature = "http-client")]
    pub gateway: Option<Arc<crate::gateway::Gateway>>,
    /// This process's worker id (D1 `worker` column).
    pub worker_id: String,
    /// Experimental feature flags (`/fv/v1/admin/flags`, crate::flags).
    pub flags: Arc<crate::flags::FeatureFlags>,
    #[cfg(feature = "http-client")]
    registration: Option<crate::worker::Registration>,
    sweeper: tokio::task::JoinHandle<()>,
    key_maintenance: tokio::task::JoinHandle<()>,
    background: Vec<tokio::task::JoinHandle<()>>,
}

impl std::fmt::Debug for App {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("App").field("ctx", &self.ctx).finish_non_exhaustive()
    }
}

impl Drop for App {
    fn drop(&mut self) {
        self.sweeper.abort();
        self.key_maintenance.abort();
        for h in &self.background {
            h.abort();
        }
    }
}

/// The engine from `[engine]`.
pub fn build_engine(c: &Config) -> anyhow::Result<EngineService> {
    build_engine_with_clock(c, None)
}

/// [`build_engine`], with the fake backend's load and step time on `clock`
/// (tests pass a `ManualClock` to decide when loading and steps end).
pub fn build_engine_with_clock(
    c: &Config,
    clock: Option<std::sync::Arc<dyn fastvideo_engine_service::Clock>>,
) -> anyhow::Result<EngineService> {
    let backends: Vec<Box<dyn EngineBackend>> = match c.engine.backend {
        EngineBackendKind::Fake => {
            let f = &c.engine.fake;
            let mut fc = FakeConfig::default();
            if !f.models.is_empty() {
                let ids: Vec<&str> = f.models.iter().map(String::as_str).collect();
                fc = fc.with_models(&ids);
                if fc.models.is_empty() {
                    return Err(anyhow!("engine.fake.models matches no fake model"));
                }
            }
            if f.all_resident {
                for m in &mut fc.models {
                    m.caps.resident = true;
                }
            }
            if f.h3_1080p {
                for m in &mut fc.models {
                    if m.caps.family == fastvideo_protocol::Family::H3
                        && matches!(m.caps.tier, Some(fastvideo_protocol::Tier::Max | fastvideo_protocol::Tier::Turbo))
                    {
                        m.caps.canvas = m.caps.canvas.clone().with_h3_1080p();
                    }
                }
            }
            if let Some(d) = &f.device {
                fc.device_profile = Some(fastvideo_engine_service::device::DeviceProfile::parse(d).map_err(|e| anyhow!("engine.fake.device: {e}"))?);
            }
            fc.timing = FakeTiming {
                load: Duration::from_millis(f.load_ms),
                step: Duration::from_millis(f.step_ms),
                rtf: f.rtf,
                ..FakeTiming::default()
            };
            fc.mp4 = Mp4Mode::Auto;
            if let Some(clock) = clock {
                fc.clock = clock;
            }
            vec![Box::new(FakeBackend::new(fc))]
        }
        EngineBackendKind::Remote => {
            // Gateway mode: no local models. An idle engine keeps the
            // drain/shutdown paths uniform; the gateway is the ServeCtx's
            // engine gate.
            vec![Box::new(FakeBackend::new(FakeConfig::default().with_models(&[])))]
        }
        EngineBackendKind::Cuda => {
            // WP-11: the CudaBackend from `[[models]]` (one GPU).
            #[cfg(feature = "cuda")]
            if !c.models.is_empty() {
                vec![cuda_backend(c)?]
            } else {
                // Without `[[models]]`: the SF-Wan-only causal backend from
                // `FV_SFWAN_WEIGHTS` (WP-15).
                if let Some(b) = fastvideo_engine_service::cuda::causal::CausalCudaBackend::from_env() {
                    let cfg = EngineConfig {
                        queue_max: c.limits.queue_max,
                        output_dir: c.server.state_dir.join("engine-out"),
                        ..EngineConfig::default()
                    };
                    return EngineService::start(cfg, vec![Box::new(b)])
                        .map_err(|e| anyhow!("starting the engine: {e}"));
                }
                return Err(anyhow!("engine.backend = cuda needs [[models]] (or FV_SFWAN_WEIGHTS)"));
            }
            #[cfg(not(feature = "cuda"))]
            return Err(anyhow!(
                "engine.backend = cuda: this fv-serve was built without the `cuda` feature"
            ));
        }
    };
    let cfg = EngineConfig {
        queue_max: c.limits.queue_max,
        swap: c.engine.swap,
        tier_overrides: c.engine.tier_overrides.iter().map(|(k, v)| (k.clone(), ModelId::new(v))).collect(),
        output_dir: c.server.state_dir.join("engine-out"),
        ..EngineConfig::default()
    };
    EngineService::start(cfg, backends).map_err(|e| anyhow!("starting the engine: {e}"))
}

/// The CUDA catalog models for `[[models]]`: each entry resolves against
/// the catalog (`recipe` = tier alias, catalog id or H3 recipe name, empty =
/// the family's turbo tier); weights default under `FV_WEIGHTS`
/// (`/workspace/weights`) and the tiny decoders under `FV_TAE_DIR`
/// (`$FV_WEIGHTS/auxiliary/tae`). Also checks that the set can share one
/// process (process-wide technique settings). Built without `cuda` so the
/// shipped configs are tested on CPU.
pub fn cuda_models(c: &Config) -> anyhow::Result<Vec<fastvideo_engine_service::cuda::CudaModel>> {
    use fastvideo_engine_service::cuda::ProcessPlan;
    let models = catalog_models(&c.models)?;
    if !models.iter().any(|m| m.resident) {
        return Err(anyhow!("[[models]]: no model is `resident = true` (nothing would load before readiness)"));
    }
    ProcessPlan::for_models(&models).map_err(|e| anyhow!("[[models]]: {e}"))?;
    Ok(models)
}

/// `[[models]]`-style entries resolved against the CUDA catalog (CPU-only:
/// no weights are read). The gateway uses it for a pool's static caps.
pub fn catalog_models(entries: &[crate::config::ModelCfg]) -> anyhow::Result<Vec<fastvideo_engine_service::cuda::CudaModel>> {
    use fastvideo_engine_service::cuda::{model_from_config, ModelEntryCfg, WeightLayout};
    let root = std::env::var("FV_WEIGHTS").unwrap_or_else(|_| "/workspace/weights".into());
    let mut layout = WeightLayout::new(&root);
    if let Ok(t) = std::env::var("FV_TAE_DIR") {
        layout = layout.with_tae_dir(t);
    }
    entries
        .iter()
        .map(|m| {
            let extra = m
                .extra
                .iter()
                // Strings and booleans (`warmup = true`) reach the backend as text.
                .filter_map(|(k, v)| {
                    v.as_str()
                        .map(str::to_owned)
                        .or_else(|| v.as_bool().map(|b| b.to_string()))
                        .map(|s| (k.clone(), s))
                })
                .collect();
            model_from_config(
                &layout,
                &ModelEntryCfg {
                    id: m.id.clone(),
                    family: m.family.clone(),
                    recipe: m.recipe.clone().unwrap_or_default(),
                    weights: m.weights.as_ref().map(Into::into),
                    resident: m.resident,
                    served_names: m.served_names.clone(),
                    extra,
                },
            )
            .map_err(|e| anyhow!("[[models]]: {e}"))
        })
        .collect::<anyhow::Result<Vec<_>>>()
}

/// The WP-11 `CudaBackend` (GPU 0) for `[[models]]` ([`cuda_models`]).
#[cfg(feature = "cuda")]
fn cuda_backend(c: &Config) -> anyhow::Result<Box<dyn EngineBackend>> {
    use fastvideo_engine_service::cuda::{CudaBackend, CudaBackendConfig, Mp4Encoder};
    let mut cfg = CudaBackendConfig::new(0, cuda_models(c)?);
    cfg.work_dir = c.server.state_dir.join("engine-work");
    cfg.text_cache = Some(c.server.state_dir.join("text-cache"));
    // `auto` was resolved at startup (encoders::resolve) to nvenc or the
    // CPU test encoder.
    cfg.encoder = if c.engine.post_encoder == "cpu-test-x264" {
        Mp4Encoder::Libx264CpuTest
    } else {
        Mp4Encoder::Nvenc
    };
    let b = CudaBackend::new(cfg).map_err(|e| anyhow!("cuda backend: {e}"))?;
    Ok(Box::new(b))
}

fn public_base(c: &Config) -> anyhow::Result<Url> {
    if let Some(u) = &c.server.public_base_url {
        return Url::parse(u).context("server.public_base_url");
    }
    let addr: SocketAddr = c.bind_addr()?;
    // Runpod pod proxy / LB URL, Vast public IP:port (WP-16 discovery).
    if c.server.mode == Mode::Http {
        if let Some(u) = fastvideo_deploy::env::Discovery::from_process().public_base_url(addr.port()) {
            return Url::parse(&u).context("discovered public base URL");
        }
    }
    let host = if addr.ip().is_unspecified() { "127.0.0.1".to_owned() } else { addr.ip().to_string() };
    Url::parse(&format!("http://{host}:{}", addr.port())).context("public base")
}

fn worker_id(c: &Config) -> String {
    c.server.worker_id.clone().unwrap_or_else(|| format!("fv-{}", fastvideo_serve_kit::random_token()))
}

impl App {
    /// Builds everything (starts the engine and background tasks; needs a
    /// tokio runtime).
    pub async fn build(mut config: Config, ov: Overrides) -> anyhow::Result<App> {
        // `auto` encoder settings → NVENC or OpenH264, probed once.
        crate::encoders::resolve(&mut config).await;
        tokio::fs::create_dir_all(&config.server.state_dir)
            .await
            .with_context(|| format!("creating {}", config.server.state_dir.display()))?;
        let base = public_base(&config)?;
        let worker = worker_id(&config);
        let key = if config.artifacts.signing_key.is_empty() {
            tracing::warn!("FV_URL_SIGNING_KEY is not set: /files URLs die with this process");
            UrlKey::random()
        } else {
            UrlKey::new(config.artifacts.signing_key.expose())
        };
        let artifacts = match ov.artifacts.clone() {
            Some(a) => a,
            None => storage::build_artifacts(&config, &base, &key).map_err(|e| anyhow!(e))?,
        };
        let gateway_mode = config.engine.backend == EngineBackendKind::Remote;
        let worker_role = config.server.role == crate::config::Role::Worker;
        if worker_role && config.auth.mode != fastvideo_serve_kit::AuthMode::TrustGateway {
            tracing::info!("server.role = worker: API auth is the gateway's (trust-gateway); every route needs the internal token");
            config.auth.mode = fastvideo_serve_kit::AuthMode::TrustGateway;
        }
        let (jobs, d1, jobs_kind) = match ov.jobs {
            Some(j) => (j, None, "custom"),
            None => {
                let j = storage::build_jobs(&config, &worker, artifacts.clone(), ov.d1.clone()).await.map_err(|e| anyhow!(e))?;
                let kind = match j.kind {
                    JobBackend::D1 => "d1",
                    JobBackend::Memory => "memory",
                    _ => "file",
                };
                (j.store, j.d1, kind)
            }
        };
        let engine = match ov.engine {
            Some(e) => e,
            None => build_engine(&config)?,
        };
        let encoder = match config.engine.post_encoder.as_str() {
            "cpu-test-x264" => FfmpegH264::Libx264CpuTest,
            "auto" if crate::encoders::auto_selection().1.post == "cpu-test-x264" => FfmpegH264::Libx264CpuTest,
            _ => FfmpegH264::Nvenc,
        };
        let gate = ServiceGate::new(
            engine,
            config.aliases.clone(),
            OutputPolicy {
                placeholder: config.engine.backend == EngineBackendKind::Fake && config.engine.fake.placeholder_output,
                encoder,
                scratch: config.server.state_dir.join("outputs"),
            },
        );

        let mut sc = ServeConfig::new(base.clone(), &config.server.state_dir);
        sc.url_ttl = Duration::from_secs(config.artifacts.url_ttl_s);
        sc.body_max_bytes = config.limits.body_max_mb * 1024 * 1024;
        sc.sync_timeout = Duration::from_secs(config.server.sync_timeout_s);
        let mut retention = BTreeMap::new();
        for (name, secs) in &config.jobs.retention_s {
            if let Some(p) = ProtocolId::ALL.iter().find(|p| p.as_str() == name) {
                retention.insert(*p, Duration::from_secs(*secs));
            }
        }
        sc.retention = retention;
        let keys = if config.auth.keys.is_empty() {
            KeyRing::default()
        } else {
            KeyRing::from_hash_list(config.auth.keys.expose()).map_err(|e| anyhow!(e))?
        };
        let key_store = storage::build_key_store(&config).await.map_err(|e| anyhow!(e))?;
        if config.auth.mode == fastvideo_serve_kit::AuthMode::Keys && keys.is_empty() && key_store.list().is_empty() {
            tracing::warn!("auth.mode = keys with no FV_API_KEYS and no minted keys: keyed APIs answer 401 until a key is minted (/console/admin)");
        }
        let admin_resolved = crate::admin_token::resolve(&config, !worker_role)?;
        let admin_token_generated = admin_resolved.source != crate::admin_token::Source::Configured;
        let admin_token_source = admin_resolved.source.clone();
        let sealed_admin = admin_resolved.sealed;
        let admin = admin_resolved.token;
        // fal webhooks are Ed25519-signed with a key published at
        // /.well-known/jwks.json; without FV_WEBHOOK_ED25519_KEY the key is
        // per process (receivers must re-fetch the JWKS after restarts).
        let signer = if config.webhook_key.is_empty() {
            if config.protocols.fal {
                tracing::warn!("FV_WEBHOOK_ED25519_KEY is not set: fal webhooks use a per-process key");
            }
            WebhookSigner::random("fv-serve")
        } else {
            WebhookSigner::from_seed_str(config.webhook_key.expose(), "fv-serve").map_err(|e| anyhow!(e))?
        };
        let mut callbacks = CallbackSender::new(CallbackSender::default_transport(), Some(signer));
        if config.server.callbacks_allow_private {
            tracing::warn!("server.callbacks_allow_private: webhooks may target loopback/private hosts (tests only)");
            callbacks.target.allow_private = true;
        }
        // A worker's jobs came through the gateway, which ran the MiniMax
        // callback challenge when it took the request.
        callbacks.challenge_done_elsewhere = worker_role;
        let callbacks = Arc::new(callbacks);
        let mcfg = mount_cfg(&config);
        // Gateway mode: the pools behind a `RemoteGate` (docs/serve/gateway.md)
        // and a D1 read-through job store.
        #[cfg(feature = "http-client")]
        let (gw, jobs) = if gateway_mode {
            let d1s = d1.clone().ok_or_else(|| anyhow!("engine.backend = remote needs the D1 job store"))?;
            let store = Arc::new(crate::gateway::store::GatewayJobStore::new(
                d1s.clone(),
                Duration::from_millis(config.gateway.watch_poll_ms.max(50)),
            ));
            let gw = crate::gateway::Gateway::build(&config, d1s.client().clone(), store.clone()).await?;
            for s in &ov.scalers {
                gw.add_scaler(s.clone());
            }
            (Some(gw), store as Arc<dyn JobStore>)
        } else {
            (None, jobs)
        };
        #[cfg(not(feature = "http-client"))]
        if gateway_mode || worker_role {
            return Err(anyhow!("gateway mode and the worker role need fv-serve built with `http-client`"));
        }
        #[cfg(feature = "http-client")]
        let engine_gate: Arc<dyn fastvideo_serve_kit::EngineGate> = match &gw {
            Some(g) => g.clone(),
            None => gate.clone(),
        };
        #[cfg(not(feature = "http-client"))]
        let engine_gate: Arc<dyn fastvideo_serve_kit::EngineGate> = gate.clone();
        // Experimental feature flags: in D1 beside the jobs when there is
        // D1, else in the state dir; applied to the caps every API
        // negotiates against (crate::flags).
        let flags = match &d1 {
            Some(d) => crate::flags::FeatureFlags::d1(d.client().clone()).await,
            None => crate::flags::FeatureFlags::file(config.server.state_dir.join("feature_flags.json")).await,
        };
        let flags_refresh = flags.spawn_refresh(Duration::from_secs(30));
        let engine_gate: Arc<dyn fastvideo_serve_kit::EngineGate> =
            Arc::new(crate::flags::FlaggedGate { inner: engine_gate, flags: flags.clone() });
        let mut builder = ServeCtx::builder(sc, engine_gate)
            .auth(Auth::new(config.auth.mode, keys).with_key_store(key_store.clone()))
            .url_key(key)
            .jobs(jobs.clone())
            .artifacts(artifacts)
            .callbacks(callbacks);
        for (p, r) in adapters::renderers(&mcfg) {
            builder = builder.renderer(p, r);
        }
        if let Some(dir) = &config.artifacts.local_dir {
            builder = builder.artifacts_root(dir);
        }
        let ctx = builder.build().await.context("building the serve context")?;
        gate.attach(ctx.clone());
        let admin = Arc::new(admin);
        #[cfg(feature = "http-client")]
        if let Some(g) = &gw {
            g.attach(ctx.clone());
            let mut router = crate::gateway::assemble(&config, &ctx, g, admin.clone(), key_store.clone())
                .merge(crate::admin_token::routes(sealed_admin.clone()))
                .merge(crate::flags::routes(flags.clone(), admin.clone()));
            let d1_client = d1.as_ref().map(|d| d.client().clone());
            let autoscale = crate::autoscale::start(&config, g, d1_client, admin.clone(), &worker)?;
            let sweeper = spawn_sweeper(jobs, Duration::from_secs(config.jobs.sweep_interval_s.max(1)));
            let refresh = (key_store.backend_kind() == "d1").then_some(Duration::from_secs(30));
            let key_maintenance = key_store.spawn_maintenance(refresh, Duration::from_secs(30));
            // First probes before serving, then the tick loop.
            g.tick().await;
            let tick = g.spawn_tick();
            let mut background = vec![tick, flags_refresh];
            if let Some((admin_routes, handle)) = autoscale {
                router = router.merge(admin_routes);
                background.push(handle);
            }
            return Ok(App {
                config,
                ctx,
                gate,
                router,
                d1,
                keys: key_store,
                admin_token_generated,
                admin_token_source: admin_token_source.clone(),
                #[cfg(feature = "reactor")]
                reactor: None,
                gateway: Some(g.clone()),
                worker_id: worker,
                flags,
                registration: None,
                sweeper,
                key_maintenance,
                background,
            });
        }

        // Streaming front-ends that own sockets answer offers on one shared
        // WebRTC host (the Reactor runtime, the fal director): they share
        // the `[webrtc]` ports.
        #[cfg(any(feature = "reactor", all(feature = "fal", feature = "webrtc")))]
        let rtc_host = {
            let reactor_on = cfg!(feature = "reactor") && config.protocols.reactor;
            let director_on = cfg!(all(feature = "fal", feature = "webrtc")) && config.protocols.fal && config.protocols.fal_director;
            if reactor_on || director_on {
                match crate::rtc::bind(&config).await {
                    Ok(h) => Some(h),
                    Err(e) => {
                        tracing::error!(error = %format!("{e:#}"), "no WebRTC host: the Reactor runtime and the fal director are not mounted");
                        None
                    }
                }
            } else {
                None
            }
        };
        #[allow(unused_mut)]
        let mut streams = Router::new();
        #[cfg(feature = "reactor")]
        let reactor = match (&rtc_host, config.protocols.reactor) {
            (Some(host), true) => match crate::reactor::build_on(&config, gate.engine(), host.clone()) {
                Ok(r) => {
                    streams = streams.merge(fastvideo_reactor::router(r.clone()));
                    Some(r)
                }
                Err(e) => {
                    tracing::error!(error = %format!("{e:#}"), "the Reactor runtime is not mounted");
                    None
                }
            },
            _ => None,
        };
        // Minted API keys: /fv/v1/admin/keys behind the admin token.
        streams = streams
            .merge(fastvideo_serve_kit::admin_routes(key_store.clone(), admin.clone()))
            .merge(crate::flags::routes(flags.clone(), admin.clone()))
            .merge(crate::admin_token::routes(sealed_admin));
        #[allow(unused_mut)]
        let mut fal_extra: Router<ServeCtx> = Router::new();
        #[cfg(all(feature = "fal", feature = "webrtc"))]
        if let (Some(host), true) = (&rtc_host, config.protocols.fal && config.protocols.fal_director) {
            let svc = crate::director::build(&config, &mcfg, gate.engine().clone(), host.clone());
            fal_extra = fastvideo_fal::director::routes(svc);
        }
        #[allow(unused_mut)]
        let mut background = vec![flags_refresh];
        #[cfg(feature = "http-client")]
        let mut registration = None;
        // Worker role: the gateway's internal routes (before the route
        // layers, so metrics and the multi-worker filter see them).
        #[cfg(feature = "http-client")]
        if worker_role {
            let drained = Arc::new(std::sync::atomic::AtomicBool::new(false));
            if let (Some(pool), Some(d), Some(url), true, Mode::Http) =
                (&config.gateway.pool, &d1, &config.server.public_base_url, config.gateway.register, config.server.mode)
            {
                let reg = crate::worker::Registration {
                    db: d.client().clone(),
                    pool: pool.clone(),
                    worker_id: worker.clone(),
                    url: url.trim_end_matches('/').to_owned(),
                    gate: gate.clone(),
                    drained: drained.clone(),
                };
                background.push(crate::worker::spawn_registration(reg.clone()));
                registration = Some(reg);
            }
            let st = crate::worker::WorkerState::new(
                ctx.clone(),
                gate.clone(),
                d1.clone(),
                worker.clone(),
                config.gateway.pool.clone(),
                config.limits.queue_max,
            )
            .with_drain(drained, registration.clone());
            streams = streams.merge(crate::worker::routes(st));
        }
        let mut router = assemble(&config, &ctx, &gate, jobs_kind, streams, fal_extra);
        #[cfg(feature = "http-client")]
        if worker_role {
            router = crate::worker::token_layer(router, Arc::from(config.gateway.internal_token.expose()));
        }
        if config.server.forward {
            let ready = Arc::new(std::sync::OnceLock::new());
            let h = fastvideo_deploy::dispatch::RouterHandler::new(router.clone()).with_info(crate::deploy::info_fn(
                &config,
                gate.clone(),
                crate::deploy::Boot::now(),
                ready,
            ));
            router = crate::deploy::with_forward(router, h);
        }
        let sweeper = spawn_sweeper(jobs, Duration::from_secs(config.jobs.sweep_interval_s.max(1)));
        // D1 is shared between workers: reload it so keys minted or revoked
        // elsewhere apply here within 30 s.
        let refresh = (key_store.backend_kind() == "d1").then_some(Duration::from_secs(30));
        let key_maintenance = key_store.spawn_maintenance(refresh, Duration::from_secs(30));
        Ok(App {
            config,
            ctx,
            gate,
            router,
            d1,
            keys: key_store,
            admin_token_generated,
            admin_token_source,
            #[cfg(feature = "reactor")]
            reactor,
            #[cfg(feature = "http-client")]
            gateway: None,
            worker_id: worker,
            flags,
            #[cfg(feature = "http-client")]
            registration,
            sweeper,
            key_maintenance,
            background,
        })
    }

    /// Serves on `listener` until `stop` resolves, then drains (§6.3):
    /// admission stops (503), queued jobs are cancelled, the running one
    /// gets `shutdown_grace`, pending D1 writes are flushed, HTTP closes.
    pub async fn serve(self, listener: TcpListener, stop: impl std::future::Future<Output = ()> + Send + 'static) -> anyhow::Result<()> {
        self.announce();
        let drained = self.drain_after(stop);
        axum::serve(listener, self.router.clone())
            .with_graceful_shutdown(drained)
            .await
            .context("HTTP server")?;
        Ok(())
    }

    /// `FV-SERVE READY …` on stdout once ready: the gateway's pools, or the
    /// engine's models ([`announce_ready`]).
    fn announce(&self) {
        #[cfg(feature = "http-client")]
        if let Some(g) = &self.gateway {
            println!("FV-SERVE READY gateway pools={}", g.pools.iter().map(|p| p.id().to_owned()).collect::<Vec<_>>().join(","));
            return;
        }
        announce_ready(self.gate.clone());
    }

    /// Waits for `stop`, then drains (see [`App::serve`]).
    fn drain_after(&self, stop: impl std::future::Future<Output = ()> + Send + 'static) -> impl std::future::Future<Output = ()> + Send + 'static {
        let gate = self.gate.clone();
        let grace = self.config.shutdown_grace();
        let d1 = self.d1.clone();
        let keys = self.keys.clone();
        #[cfg(feature = "reactor")]
        let reactor = self.reactor.clone();
        #[cfg(feature = "http-client")]
        let gw = self.gateway.clone();
        #[cfg(feature = "http-client")]
        let registration = self.registration.clone();
        async move {
            stop.await;
            tracing::info!("shutdown requested: draining");
            #[cfg(feature = "http-client")]
            if let Some(g) = &gw {
                g.stop_admission();
            }
            #[cfg(feature = "http-client")]
            if let Some(r) = &registration {
                gate.stop_admission();
                r.write(true).await;
            }
            // Reactor: `session_ended` with RT's drain reason, then close.
            #[cfg(feature = "reactor")]
            if let Some(r) = &reactor {
                r.drain().await;
            }
            drain(&gate, grace, d1.as_deref()).await;
            if let Err(e) = keys.flush().await {
                tracing::warn!(error = %e, "api keys: writing last_used_at failed");
            }
            tracing::info!("drained");
        }
    }
}

/// Serves on `listener` from the start while `build` runs (the encoder
/// probe, stores, D1, the engine), then hands every request to the built
/// app, which serves and drains as [`App::serve`]. Until the app exists
/// the probes answer without blocking: `/ping` **204** (Runpod load
/// balancer: initializing), `/health` and `/healthz` 503 `loading`, every
/// other route 503 with `Retry-After`. Model loading itself runs in the
/// background after `build` (`/ping` stays 204 until ready), so no probe
/// waits for weights. A failed `build` stops the listener and returns
/// the error; `stop` during `build` exits without draining.
pub async fn serve_while_building(
    listener: TcpListener,
    build: impl std::future::Future<Output = anyhow::Result<App>> + Send,
    stop: impl std::future::Future<Output = ()> + Send + 'static,
) -> anyhow::Result<()> {
    let slot: Arc<std::sync::OnceLock<Router>> = Arc::default();
    let (tx, rx) = tokio::sync::oneshot::channel::<()>();
    let boot = booting_router(slot.clone());
    let server = tokio::spawn(async move {
        axum::serve(listener, boot)
            .with_graceful_shutdown(async {
                let _ = rx.await;
            })
            .await
    });
    // `stop` fires once; a watch lets both the build race and the drain
    // wait on it.
    let (stop_tx, stop_rx) = tokio::sync::watch::channel(false);
    let stop_task = tokio::spawn(async move {
        stop.await;
        let _ = stop_tx.send(true);
    });
    let stopped = |mut rx: tokio::sync::watch::Receiver<bool>| async move {
        let _ = rx.wait_for(|s| *s).await;
    };
    let built = tokio::select! {
        r = build => r,
        _ = stopped(stop_rx.clone()) => {
            let _ = tx.send(());
            let _ = server.await;
            return Ok(());
        }
    };
    let app = match built {
        Ok(a) => a,
        Err(e) => {
            stop_task.abort();
            let _ = tx.send(());
            let _ = server.await;
            return Err(e);
        }
    };
    let _ = slot.set(app.router.clone());
    app.announce();
    tracing::info!("app built: serving every route");
    app.drain_after(stopped(stop_rx)).await;
    let _ = tx.send(());
    server.await.context("HTTP server task")?.context("HTTP server")?;
    Ok(())
}

/// The router while the app builds: [`booting_reply`] until `slot` holds
/// the built router, then that router.
fn booting_router(slot: Arc<std::sync::OnceLock<Router>>) -> Router {
    Router::new().fallback(move |req: axum::extract::Request| {
        let slot = slot.clone();
        async move {
            use tower::ServiceExt;
            match slot.get() {
                Some(r) => r.clone().oneshot(req).await.unwrap_or_else(|e| match e {}),
                None => booting_reply(req.uri().path()),
            }
        }
    })
}

/// A probe's answer before the app is built (see [`serve_while_building`]).
fn booting_reply(path: &str) -> axum::response::Response {
    use axum::http::{header, StatusCode};
    use axum::response::IntoResponse;
    use axum::Json;
    use serde_json::json;
    match path {
        "/ping" => StatusCode::NO_CONTENT.into_response(),
        "/health" => (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(json!({"status": "loading", "model_loaded": false, "state": "LOADING"})),
        )
            .into_response(),
        "/healthz" => (StatusCode::SERVICE_UNAVAILABLE, Json(json!({"state": "starting"}))).into_response(),
        _ => (
            StatusCode::SERVICE_UNAVAILABLE,
            [(header::RETRY_AFTER, "5")],
            Json(json!({"error": {"kind": "loading", "message": "fv-serve is starting"}})),
        )
            .into_response(),
    }
}

/// Stops admission, drains the engine, waits for the job pumps, flushes D1.
pub async fn drain(gate: &ServiceGate, grace: Duration, d1: Option<&fastvideo_serve_kit::D1JobStore>) {
    gate.stop_admission();
    gate.engine().drain(grace).await;
    // Let the per-job pumps record the terminal events (and fire callbacks).
    tokio::time::sleep(Duration::from_millis(200)).await;
    if let Some(d) = d1 {
        d.flush_all().await;
    }
}

/// Prints `FV-SERVE READY models=<ids>` once the engine is ready (the Vast
/// PyWorker tails stdout for it, design §6.1).
fn announce_ready(gate: Arc<ServiceGate>) {
    tokio::spawn(async move {
        match gate.engine().wait_ready().await {
            Readiness::Ready => {
                let ids: Vec<String> = gate.engine().caps().models().map(|m| m.id.0.clone()).collect();
                println!("FV-SERVE READY models={}", ids.join(","));
                tracing::info!(models = ids.len(), "ready");
            }
            Readiness::Failed(e) => tracing::error!(error = %e, "model loading failed"),
            Readiness::Loading { .. } => {}
        }
    });
}

/// The adapters' view of the config.
pub fn mount_cfg(config: &Config) -> MountCfg {
    MountCfg {
        protocols: config.protocols.clone(),
        body_max: config.limits.body_max_mb * 1024 * 1024,
        sync_timeout: Duration::from_secs(config.server.sync_timeout_s),
        url_ttl: Duration::from_secs(config.artifacts.url_ttl_s),
        ltx: config.ltx.clone(),
    }
}

/// The full router: health + serve-kit files/uploads + native + adapters,
/// with request metrics and tracing, and the `/console` pages.
/// `streams` carries the streaming front-ends that are built with their own
/// state (the Reactor runtime) and the admin key routes; `fal_extra` goes
/// into the fal router (the
/// director's routes, so `/fal/proxy` reaches them).
pub fn assemble(
    config: &Config,
    ctx: &ServeCtx,
    gate: &Arc<ServiceGate>,
    jobs_kind: &'static str,
    streams: Router,
    fal_extra: Router<ServeCtx>,
) -> Router {
    let mcfg = mount_cfg(config);
    let mut kit: Router<ServeCtx> = ctx.routes();
    if config.protocols.native {
        kit = kit.merge(native::routes(gate.clone(), mcfg.body_max, mcfg.sync_timeout, config.streams.causal_limits()));
        kit = kit.merge(crate::streams::routes(gate.clone(), crate::streams::StreamsConfig::from_config(config)));
    }
    let (adapters, stateful) = adapters::mount(&mcfg, ctx, fal_extra);
    kit = kit.merge(adapters);
    let artifacts_kind = match config.artifact_backend() {
        ArtifactBackend::S3 => "s3",
        _ => "local",
    };
    let h = Health {
        gate: gate.clone(),
        metrics: metrics::install(),
        jobs_backend: jobs_kind,
        artifacts_backend: artifacts_kind,
        root_model: adapters::root_model(&mcfg, ctx),
    };
    let mut r = kit
        .with_state(ctx.clone())
        .merge(stateful)
        .merge(streams)
        .merge(health::routes(h));
    if config.server.console {
        r = r.merge(console::routes());
    }
    let r = r.route_layer(axum::middleware::from_fn(metrics::track));
    // Behind a load balancer with several workers: only the routes every
    // worker can answer (design §6.2).
    let policy = crate::multiworker::Policy::from_config(config, jobs_kind);
    if policy.multi() {
        tracing::warn!(
            policy = %crate::multiworker::summary(&policy),
            "server.workers_max > 1: serving only routes any worker can answer"
        );
    }
    let r = crate::multiworker::layer(r, policy, &config.protocols.fal_apps).layer(TraceLayer::new_for_http());
    match cors_layer(&config.server.cors_origins) {
        Some(cors) => r.layer(cors),
        None => r,
    }
}

/// Response headers a cross-origin page may read: the FastVideo metric
/// headers, our tier/recipe metadata and the fal request id.
const CORS_EXPOSE: [&str; 11] = [
    "x-request-id",
    "x-model",
    "x-inference-time-s",
    "x-stage-durations",
    "x-peak-memory-mb",
    "x-fv-tier",
    "x-fv-recipe",
    "x-fv-quality",
    "x-fal-request-id",
    "content-disposition",
    "content-length",
];

/// CORS for every route (`server.cors_origins`, config validated): answers
/// preflights (so `PUT /uploads/{token}` and `POST /storage/upload/initiate`
/// work from a page on another origin, as fal's storage does) with the
/// request's method and headers mirrored (`Authorization` is never covered
/// by a `*` allow-list, so it must be echoed), and exposes the metric
/// headers. `["*"]` allows any origin; `[]` is `None` (no CORS). Never with
/// credentials.
pub fn cors_layer(origins: &[String]) -> Option<tower_http::cors::CorsLayer> {
    use axum::http::{HeaderName, HeaderValue};
    use tower_http::cors::{AllowHeaders, AllowMethods, AllowOrigin, CorsLayer};
    if origins.is_empty() {
        return None;
    }
    let allow = if origins.iter().any(|o| o == "*") {
        AllowOrigin::any()
    } else {
        AllowOrigin::list(origins.iter().filter_map(|o| HeaderValue::from_str(o).ok()))
    };
    Some(
        CorsLayer::new()
            .allow_origin(allow)
            .allow_methods(AllowMethods::mirror_request())
            .allow_headers(AllowHeaders::mirror_request())
            .expose_headers(CORS_EXPOSE.map(HeaderName::from_static))
            .max_age(Duration::from_secs(3600)),
    )
}
