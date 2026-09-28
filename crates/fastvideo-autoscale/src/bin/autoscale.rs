//! `fv-autoscale`: the controller as its own process, next to (or without)
//! a gateway. Inside the gateway the same controller runs in-process
//! (docs/serve/gateway.md "Autoscaling").
//!
//! ```text
//! fv-autoscale --config configs/serve/autoscale.toml
//!     [--gateway https://gw.example]   signals from GET /fv/v1/gateway/pools (FV_ADMIN_TOKEN);
//!                                      without it: Runpod /health of each serverless endpoint
//!     [--live | --dry-run]             override `dry_run`
//!     [--once]                         one tick, print the report as JSON, exit
//!     [--admin-bind 127.0.0.1:9090]    serve /fv/v1/admin/autoscale (FV_ADMIN_TOKEN)
//!     [--holder NAME]                  lease holder id (default: hostname-pid)
//! ```
//!
//! Env: `RUNPOD_API_KEY`; `FV_ADMIN_TOKEN`; for `lease.backend = "d1"`
//! (feature `d1-http`) `FV_CF_ACCOUNT_ID`, `FV_CF_API_TOKEN`,
//! `FV_D1_DATABASE_ID`; for pod pools also `FV_INTERNAL_TOKEN` (drain
//! requests) and the D1 values (`gw_workers`). Secrets are never logged.

use std::sync::Arc;

use fastvideo_autoscale::admin::{admin_router, bearer};
use fastvideo_autoscale::config::{AutoscaleConfig, LeaseBackend, PoolKind};
use fastvideo_autoscale::controller::{unix_now, Controller};
use fastvideo_autoscale::gateway::{MemoryRegistry, SignalSource, WorkerRegistry};
use fastvideo_autoscale::lease::{Lease, MemoryLease};
use fastvideo_autoscale::provider::runpod::{
    HttpGatewayPools, RunpodApi, RunpodBalance, RunpodHealthSignals, RunpodPods, RunpodServerless,
};
use fastvideo_autoscale::provider::Providers;

fn arg(args: &[String], name: &str) -> Option<String> {
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).cloned()
}

fn die(msg: impl std::fmt::Display) -> ! {
    eprintln!("fv-autoscale: {msg}");
    std::process::exit(2)
}

#[cfg(feature = "d1-http")]
fn d1_client() -> Option<fastvideo_serve_kit::d1::D1Client> {
    let v = |k: &str| std::env::var(k).ok().filter(|s| !s.is_empty());
    let cfg = fastvideo_serve_kit::d1::D1Config::new(v("FV_CF_ACCOUNT_ID")?, v("FV_CF_API_TOKEN")?, v("FV_D1_DATABASE_ID")?);
    fastvideo_serve_kit::d1::D1Client::http(cfg).ok()
}

#[cfg(not(feature = "d1-http"))]
fn d1_client() -> Option<fastvideo_serve_kit::d1::D1Client> {
    None
}

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt().with_env_filter(
        tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
    ).init();
    let args: Vec<String> = std::env::args().skip(1).collect();
    let path = arg(&args, "--config").unwrap_or_else(|| die("--config <file> is required"));
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| die(format!("{path}: {e}")));
    let mut cfg = AutoscaleConfig::from_toml_document(&text).unwrap_or_else(|e| die(e));
    if args.iter().any(|a| a == "--live") {
        cfg.dry_run = false;
    }
    if args.iter().any(|a| a == "--dry-run") {
        cfg.dry_run = true;
    }
    if !cfg.enabled {
        die("autoscale.enabled is false");
    }
    if let Some(p) = cfg.pools.iter().find(|p| p.kind == PoolKind::Serverless && p.serverless.endpoint_id.is_empty()) {
        die(format!("pool {}: serverless.endpoint_id is required outside the gateway", p.name));
    }
    let api = RunpodApi::from_env().unwrap_or_else(|e| die(e));
    let admin_token = std::env::var("FV_ADMIN_TOKEN").ok().filter(|s| !s.is_empty());

    let signals: Arc<dyn SignalSource> = match arg(&args, "--gateway") {
        Some(url) => {
            let tok = admin_token.clone().unwrap_or_else(|| die("--gateway needs FV_ADMIN_TOKEN"));
            Arc::new(HttpGatewayPools::new(url, tok).unwrap_or_else(|e| die(e)))
        }
        None => Arc::new(RunpodHealthSignals::new(api.clone(), &cfg.pools)),
    };

    let d1 = d1_client();
    let lease: Arc<dyn Lease> = match cfg.lease.backend {
        LeaseBackend::Memory => Arc::new(MemoryLease::default()),
        LeaseBackend::D1 => Arc::new(fastvideo_autoscale::lease::D1Lease::new(
            d1.clone().unwrap_or_else(|| die("lease.backend = d1 needs feature d1-http and FV_CF_ACCOUNT_ID/FV_CF_API_TOKEN/FV_D1_DATABASE_ID")),
        )),
    };

    let mut providers = Providers::new(Arc::new(RunpodBalance(api.clone())))
        .with(PoolKind::Serverless, Arc::new(RunpodServerless::new(api.clone())));
    if cfg.pools.iter().any(|p| p.kind == PoolKind::Pod) {
        let registry: Arc<dyn WorkerRegistry> = match d1.clone() {
            Some(c) => Arc::new(
                fastvideo_autoscale::provider::runpod::GatewayWorkers::new(c, std::env::var("FV_INTERNAL_TOKEN").ok())
                    .unwrap_or_else(|e| die(e)),
            ),
            None => {
                tracing::warn!("pod pools without D1: in-flight work per pod is unknown (treated as idle); use only for tests");
                Arc::new(MemoryRegistry::default())
            }
        };
        providers = providers.with(
            PoolKind::Pod,
            Arc::new(RunpodPods::new(api.clone(), registry, cfg.prices.clone()).unwrap_or_else(|e| die(e))),
        );
    }

    let holder = arg(&args, "--holder").unwrap_or_else(|| {
        format!("{}-{}", std::env::var("HOSTNAME").unwrap_or_else(|_| "fv-autoscale".into()), std::process::id())
    });
    let ctrl = Arc::new(Controller::new(cfg, providers, signals, lease, holder));

    if args.iter().any(|a| a == "--once") {
        let r = ctrl.tick(unix_now()).await;
        println!("{}", serde_json::to_string_pretty(&r).unwrap_or_default());
        return;
    }

    if let Some(bind) = arg(&args, "--admin-bind") {
        let tok = admin_token.unwrap_or_else(|| die("--admin-bind needs FV_ADMIN_TOKEN"));
        let app = admin_router(ctrl.clone(), bearer(tok));
        let l = tokio::net::TcpListener::bind(&bind).await.unwrap_or_else(|e| die(format!("{bind}: {e}")));
        tracing::info!(%bind, "admin route up");
        tokio::spawn(async move {
            let _ = axum::serve(l, app).await;
        });
    }

    let (tx, rx) = tokio::sync::watch::channel(false);
    let h = ctrl.clone().spawn(rx);
    let _ = tokio::signal::ctrl_c().await;
    let _ = tx.send(true);
    let _ = h.await;
}
