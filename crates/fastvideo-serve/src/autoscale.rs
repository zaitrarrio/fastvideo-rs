//! The autoscaler inside the gateway (docs/serve/gateway.md §8; the code
//! is `crates/fastvideo-autoscale`). With `[autoscale] enabled = true` in
//! gateway mode: the gateway's `PoolScaler` hook feeds the controller's
//! signals, Runpod serverless endpoints and template pods are steered with
//! `FV_RUNPOD_API_KEY`/`RUNPOD_API_KEY` (`gateway.runpod_api_key`), one
//! replica acts at a time (D1 lease row), and `/fv/v1/admin/autoscale`
//! answers behind the admin token.

use std::sync::Arc;

use anyhow::anyhow;
use axum::Router;
use fastvideo_autoscale::gateway::{GatewayPoolMetrics, GatewaySignals, WorkerRegistry};
use fastvideo_autoscale::lease::{D1Lease, Lease, MemoryLease};
use fastvideo_autoscale::provider::runpod::{GatewayWorkers, RunpodApi, RunpodBalance, RunpodPods, RunpodServerless};
use fastvideo_autoscale::provider::Providers;
use fastvideo_autoscale::{AutoscaleConfig, Controller, PoolKind};
use fastvideo_serve_kit::keys::AdminToken;
use fastvideo_serve_kit::D1Client;

use crate::config::Config;
use crate::gateway::scale::{PoolMetrics, PoolScaler};
use crate::gateway::Gateway;

/// Hands every metrics tick to the controller's signals.
struct Hook(Arc<GatewaySignals>);

#[async_trait::async_trait]
impl PoolScaler for Hook {
    async fn observe(&self, pools: &[PoolMetrics]) {
        // Same JSON on both sides (the §7 contract).
        let v: Vec<GatewayPoolMetrics> = pools
            .iter()
            .filter_map(|m| serde_json::to_value(m).ok().and_then(|v| serde_json::from_value(v).ok()))
            .collect();
        self.0.update(&v);
    }
}

/// The `[autoscale]` config with gaps filled from `[[pools]]` (serverless
/// endpoint ids), validated.
pub fn resolve(config: &Config) -> anyhow::Result<AutoscaleConfig> {
    let mut a = config.autoscale.clone();
    for p in &mut a.pools {
        let Some(gp) = config.pools.iter().find(|g| g.id == p.name) else {
            return Err(anyhow!("[autoscale] pool {:?} is not a gateway [[pools]] id", p.name));
        };
        if p.kind == PoolKind::Serverless && p.serverless.endpoint_id.is_empty() {
            p.serverless.endpoint_id = gp.endpoint_id.clone().unwrap_or_default();
        }
    }
    a.validate().map_err(|e| anyhow!("{e}"))?;
    Ok(a)
}

/// Starts the controller when `[autoscale] enabled`; returns the admin
/// router to merge and the loop's handle.
pub fn start(
    config: &Config,
    gw: &Arc<Gateway>,
    d1: Option<D1Client>,
    admin: Arc<AdminToken>,
    worker_id: &str,
) -> anyhow::Result<Option<(Router, tokio::task::JoinHandle<()>)>> {
    if !config.autoscale.enabled {
        return Ok(None);
    }
    let cfg = resolve(config)?;
    let key = if config.gateway.runpod_api_key.is_empty() {
        None
    } else {
        Some(config.gateway.runpod_api_key.expose().to_owned())
    };
    let api = match key {
        Some(k) => RunpodApi::new(k),
        None => RunpodApi::from_env(),
    }
    .map_err(|e| anyhow!("[autoscale]: {e}"))?;

    let signals = Arc::new(GatewaySignals::default());
    gw.add_scaler(Arc::new(Hook(signals.clone())));

    let lease: Arc<dyn Lease> = match (cfg.lease.backend, d1.clone()) {
        (fastvideo_autoscale::config::LeaseBackend::D1, Some(c)) => Arc::new(D1Lease::new(c)),
        (fastvideo_autoscale::config::LeaseBackend::D1, None) => {
            return Err(anyhow!("[autoscale.lease] backend = d1 needs the D1 job store"));
        }
        _ => Arc::new(MemoryLease::default()),
    };
    let mut providers = Providers::new(Arc::new(RunpodBalance(api.clone())))
        .with(PoolKind::Serverless, Arc::new(RunpodServerless::new(api.clone())));
    if cfg.pools.iter().any(|p| p.kind == PoolKind::Pod) {
        let d1 = d1.ok_or_else(|| anyhow!("[autoscale] pod pools need D1 (gw_workers)"))?;
        let tok = config.gateway.internal_token.expose();
        let registry: Arc<dyn WorkerRegistry> = Arc::new(
            GatewayWorkers::new(d1, (!tok.is_empty()).then(|| tok.to_owned())).map_err(|e| anyhow!("{e}"))?,
        );
        providers = providers.with(
            PoolKind::Pod,
            Arc::new(RunpodPods::new(api, registry, cfg.prices.clone()).map_err(|e| anyhow!("{e}"))?),
        );
    }
    let holder = format!("{worker_id}-{}", std::process::id());
    let ctrl = Arc::new(Controller::new(cfg, providers, signals, lease, holder));
    let router = fastvideo_autoscale::admin::admin_router(ctrl.clone(), Arc::new(move |h| admin.check_headers(h)));
    // No stop signal: the loop lives as long as the process (the lease
    // expires after `ttl_s` when it is gone).
    let (_tx, rx) = tokio::sync::watch::channel(false);
    let handle = ctrl.spawn(rx);
    Ok(Some((router, handle)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gateway::scale::{DurationStats, WorkerCounts};
    use fastvideo_autoscale::gateway::SignalSource;

    fn gateway_config() -> Config {
        let text = format!(
            "{}\n{}",
            include_str!("../../../configs/serve/gateway.toml")
                .split("# ---- BEGIN [autoscale]")
                .next()
                .unwrap_or_default(),
            include_str!("../../../configs/serve/autoscale.toml")
        );
        let mut c: Config = toml::from_str(&text).expect("gateway.toml + autoscale.toml parse as one config");
        for p in &mut c.pools {
            if p.id == "wan" {
                p.endpoint_id = Some("ep-wan".into());
            }
        }
        c
    }

    #[test]
    fn autoscale_include_resolves_against_the_gateway_pools() {
        let c = gateway_config();
        assert!(c.autoscale.enabled);
        let a = resolve(&c).unwrap();
        let wan = a.pools.iter().find(|p| p.name == "wan").unwrap();
        assert_eq!(wan.serverless.endpoint_id, "ep-wan", "filled from [[pools]]");
        let mut bad = c.clone();
        bad.autoscale.pools[0].name = "nope".into();
        assert!(resolve(&bad).is_err());
    }

    #[tokio::test]
    async fn the_hook_feeds_the_signals() {
        let s = Arc::new(GatewaySignals::default());
        let m = PoolMetrics {
            pool: "wan".into(),
            kind: crate::config::PoolKind::RunpodServerless,
            endpoint_id: Some("ep".into()),
            at_unix_ms: 1_000,
            queued: 3,
            running: 1,
            oldest_queued_age_s: 9.0,
            streams: 0,
            run_time: DurationStats::of(&[6.0, 7.0]),
            queue_wait: DurationStats::default(),
            window_s: 600,
            workers: WorkerCounts::default(),
            available: true,
            max_queued: 32,
            max_streams: 0,
            submitted_total: 4,
        };
        Hook(s.clone()).observe(&[m]).await;
        let v = s.signals(&["wan".into()]).await;
        assert_eq!((v[0].queued, v[0].running_jobs, v[0].recent_job_s), (3, 1, Some(6.5)));
    }
}
