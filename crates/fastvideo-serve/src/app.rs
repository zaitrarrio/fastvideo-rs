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
use fastvideo_serve_kit::{Auth, CallbackSender, KeyRing, ServeConfig, ServeCtx, UrlKey, WebhookSigner};
use tokio::net::TcpListener;
use tower_http::trace::TraceLayer;
use url::Url;

use crate::adapters::{self, MountCfg};
use crate::config::{ArtifactBackend, Config, EngineBackendKind, JobBackend, Mode};
use crate::gate::{OutputPolicy, ServiceGate};
use crate::health::{self, Health};
use crate::{metrics, native, storage};

/// Parts tests (or other front ends) substitute.
#[derive(Default)]
pub struct Overrides {
    /// Use this job store instead of the configured one.
    pub jobs: Option<Arc<dyn JobStore>>,
    /// Use this engine instead of building one from `[engine]`.
    pub engine: Option<EngineService>,
}

/// A built server.
pub struct App {
    pub config: Config,
    pub ctx: ServeCtx,
    pub gate: Arc<ServiceGate>,
    pub router: Router,
    pub d1: Option<Arc<fastvideo_serve_kit::D1JobStore>>,
    /// The Reactor local runtime, when mounted (feature `reactor`).
    #[cfg(feature = "reactor")]
    pub reactor: Option<fastvideo_reactor::Reactor>,
    sweeper: tokio::task::JoinHandle<()>,
}

impl std::fmt::Debug for App {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("App").field("ctx", &self.ctx).finish_non_exhaustive()
    }
}

impl Drop for App {
    fn drop(&mut self) {
        self.sweeper.abort();
    }
}

/// The engine from `[engine]`.
pub fn build_engine(c: &Config) -> anyhow::Result<EngineService> {
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
            fc.timing = FakeTiming {
                load: Duration::from_millis(f.load_ms),
                step: Duration::from_millis(f.step_ms),
                rtf: f.rtf,
                ..FakeTiming::default()
            };
            fc.mp4 = Mp4Mode::Auto;
            vec![Box::new(FakeBackend::new(fc))]
        }
        EngineBackendKind::Cuda => {
            // WP-15 stopgap until the full CudaBackend (WP-11) lands: an
            // SF-Wan-only causal streaming backend from `FV_SFWAN_WEIGHTS`.
            #[cfg(feature = "cuda")]
            if let Some(b) = fastvideo_engine_service::cuda::causal::CausalCudaBackend::from_env() {
                let cfg = EngineConfig {
                    queue_max: c.limits.queue_max,
                    output_dir: c.server.state_dir.join("engine-out"),
                    ..EngineConfig::default()
                };
                return EngineService::start(cfg, vec![Box::new(b)]).map_err(|e| anyhow!("starting the engine: {e}"));
            }
            // Mount point for WP-11 (`fastvideo_engine_service::cuda::CudaBackend`
            // built from `[[models]]`, one backend per GPU).
            return Err(anyhow!(
                "engine.backend = cuda: the CUDA backend (WP-11) is not available in this build{}",
                if cfg!(feature = "cuda") { "" } else { " (built without `cuda`)" }
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
    pub async fn build(config: Config, ov: Overrides) -> anyhow::Result<App> {
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
        let artifacts = storage::build_artifacts(&config, &base, &key).map_err(|e| anyhow!(e))?;
        let (jobs, d1, jobs_kind) = match ov.jobs {
            Some(j) => (j, None, "custom"),
            None => {
                let j = storage::build_jobs(&config, &worker, artifacts.clone()).await.map_err(|e| anyhow!(e))?;
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

        let mut sc = ServeConfig::new(base, &config.server.state_dir);
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
        if config.auth.mode == fastvideo_serve_kit::AuthMode::Keys && keys.is_empty() {
            tracing::warn!("auth.mode = keys with no FV_API_KEYS: every keyed API answers 401");
        }
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
        let callbacks = Arc::new(CallbackSender::new(CallbackSender::default_transport(), Some(signer)));
        let mcfg = mount_cfg(&config);
        let mut builder = ServeCtx::builder(sc, gate.clone())
            .auth(Auth::new(config.auth.mode, keys))
            .url_key(key)
            .jobs(jobs.clone())
            .artifacts(artifacts)
            .callbacks(callbacks);
        for (p, r) in adapters::renderers(&mcfg) {
            builder = builder.renderer(p, r);
        }
        let ctx = builder.build().await.context("building the serve context")?;
        gate.attach(ctx.clone());

        // Streaming front-ends that own sockets (Reactor: the WebRTC host).
        #[allow(unused_mut)]
        let mut streams = Router::new();
        #[cfg(feature = "reactor")]
        let reactor = if config.protocols.reactor {
            match crate::reactor::build(&config, gate.engine()).await {
                Ok(r) => {
                    streams = streams.merge(fastvideo_reactor::router(r.clone()));
                    Some(r)
                }
                Err(e) => {
                    tracing::error!(error = %format!("{e:#}"), "the Reactor runtime is not mounted");
                    None
                }
            }
        } else {
            None
        };
        let mut router = assemble(&config, &ctx, &gate, jobs_kind, streams);
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
        Ok(App {
            config,
            ctx,
            gate,
            router,
            d1,
            #[cfg(feature = "reactor")]
            reactor,
            sweeper,
        })
    }

    /// Serves on `listener` until `stop` resolves, then drains (§6.3):
    /// admission stops (503), queued jobs are cancelled, the running one
    /// gets `shutdown_grace`, pending D1 writes are flushed, HTTP closes.
    pub async fn serve(self, listener: TcpListener, stop: impl std::future::Future<Output = ()> + Send + 'static) -> anyhow::Result<()> {
        let gate = self.gate.clone();
        let grace = self.config.shutdown_grace();
        let d1 = self.d1.clone();
        #[cfg(feature = "reactor")]
        let reactor = self.reactor.clone();
        announce_ready(gate.clone());
        let drained = async move {
            stop.await;
            tracing::info!("shutdown requested: draining");
            // Reactor: `session_ended` with RT's drain reason, then close.
            #[cfg(feature = "reactor")]
            if let Some(r) = &reactor {
                r.drain().await;
            }
            drain(&gate, grace, d1.as_deref()).await;
            tracing::info!("drained");
        };
        axum::serve(listener, self.router.clone())
            .with_graceful_shutdown(drained)
            .await
            .context("HTTP server")?;
        Ok(())
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
/// with request metrics and tracing.
/// `streams` carries the streaming front-ends that are built with their own
/// state (the Reactor runtime).
pub fn assemble(
    config: &Config,
    ctx: &ServeCtx,
    gate: &Arc<ServiceGate>,
    jobs_kind: &'static str,
    streams: Router,
) -> Router {
    let mcfg = mount_cfg(config);
    let mut kit: Router<ServeCtx> = ctx.routes();
    if config.protocols.native {
        kit = kit.merge(native::routes(gate.clone(), mcfg.body_max, mcfg.sync_timeout));
        kit = kit.merge(crate::streams::routes(gate.clone(), crate::streams::StreamsConfig::from_config(config)));
    }
    let (adapters, stateful) = adapters::mount(&mcfg, ctx);
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
    kit.with_state(ctx.clone())
        .merge(stateful)
        .merge(streams)
        .merge(health::routes(h))
        .route_layer(axum::middleware::from_fn(metrics::track))
        .layer(TraceLayer::new_for_http())
}
