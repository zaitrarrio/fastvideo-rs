//! The gateway↔autoscaler interface (docs/serve/gateway.md, "Autoscaling").
//!
//! The gateway implements both traits and hands them to the controller:
//!
//! - [`SignalSource`]: per-pool queue and load, read every tick.
//! - [`WorkerRegistry`]: pod workers the controller creates are
//!   registered with the gateway pool once ready, drained before deletion,
//!   and removed after. The registry also reports in-flight work per worker
//!   so the controller never deletes a busy one.
//!
//! [`StaticSignals`] and [`MemoryRegistry`] are in-process versions for
//! tests and for running the controller without a gateway (provider
//! validation).

use std::collections::BTreeMap;
use std::sync::Mutex;

use crate::types::PoolSignals;

#[async_trait::async_trait]
pub trait SignalSource: Send + Sync {
    /// Signals for the named pools (missing pools count as idle).
    async fn signals(&self, pools: &[String]) -> Vec<PoolSignals>;
}

#[async_trait::async_trait]
pub trait WorkerRegistry: Send + Sync {
    /// A pod is ready: route `pool` work to `url`.
    async fn register(&self, pool: &str, worker_id: &str, url: &str);
    /// Stop routing new work to the worker (in-flight work continues).
    async fn drain(&self, pool: &str, worker_id: &str);
    /// Route to a draining worker again.
    async fn undrain(&self, pool: &str, worker_id: &str);
    /// The worker is gone.
    async fn deregister(&self, pool: &str, worker_id: &str);
    /// In-flight jobs + streams per registered worker.
    async fn in_flight(&self, pool: &str) -> BTreeMap<String, u32>;
}

/// Signals set by hand.
#[derive(Default)]
pub struct StaticSignals {
    inner: Mutex<BTreeMap<String, PoolSignals>>,
}

impl StaticSignals {
    pub fn set(&self, s: PoolSignals) {
        self.inner.lock().unwrap_or_else(|p| p.into_inner()).insert(s.pool.clone(), s);
    }
}

#[async_trait::async_trait]
impl SignalSource for StaticSignals {
    async fn signals(&self, pools: &[String]) -> Vec<PoolSignals> {
        let g = self.inner.lock().unwrap_or_else(|p| p.into_inner());
        pools.iter().filter_map(|p| g.get(p).cloned()).collect()
    }
}

/// One registered worker.
#[derive(Clone, Debug, PartialEq, serde::Serialize)]
pub struct Registered {
    pub url: String,
    pub draining: bool,
    pub in_flight: u32,
}

/// An in-memory registry.
#[derive(Default)]
pub struct MemoryRegistry {
    inner: Mutex<BTreeMap<String, BTreeMap<String, Registered>>>,
}

impl MemoryRegistry {
    pub fn snapshot(&self, pool: &str) -> BTreeMap<String, Registered> {
        self.inner.lock().unwrap_or_else(|p| p.into_inner()).get(pool).cloned().unwrap_or_default()
    }
    pub fn set_in_flight(&self, pool: &str, worker: &str, n: u32) {
        if let Some(w) = self.inner.lock().unwrap_or_else(|p| p.into_inner()).get_mut(pool).and_then(|m| m.get_mut(worker)) {
            w.in_flight = n;
        }
    }
}

#[async_trait::async_trait]
impl WorkerRegistry for MemoryRegistry {
    async fn register(&self, pool: &str, worker_id: &str, url: &str) {
        let mut g = self.inner.lock().unwrap_or_else(|p| p.into_inner());
        g.entry(pool.to_owned())
            .or_default()
            .entry(worker_id.to_owned())
            .or_insert(Registered { url: url.to_owned(), draining: false, in_flight: 0 });
    }
    async fn drain(&self, pool: &str, worker_id: &str) {
        if let Some(w) = self.inner.lock().unwrap_or_else(|p| p.into_inner()).get_mut(pool).and_then(|m| m.get_mut(worker_id)) {
            w.draining = true;
        }
    }
    async fn undrain(&self, pool: &str, worker_id: &str) {
        if let Some(w) = self.inner.lock().unwrap_or_else(|p| p.into_inner()).get_mut(pool).and_then(|m| m.get_mut(worker_id)) {
            w.draining = false;
        }
    }
    async fn deregister(&self, pool: &str, worker_id: &str) {
        if let Some(m) = self.inner.lock().unwrap_or_else(|p| p.into_inner()).get_mut(pool) {
            m.remove(worker_id);
        }
    }
    async fn in_flight(&self, pool: &str) -> BTreeMap<String, u32> {
        self.snapshot(pool).into_iter().map(|(k, v)| (k, v.in_flight)).collect()
    }
}

// ---- The gateway's metrics (docs/serve/gateway.md §7) ------------------

/// Mirror of `fastvideo_serve::gateway::scale::PoolKind` (same JSON).
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum GatewayPoolKind {
    RunpodServerless,
    Pod,
}

/// Mirror of `DurationStats` (seconds over a window).
#[derive(Clone, Debug, Default, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct DurationStats {
    pub count: u32,
    pub mean_s: f64,
    pub p50_s: f64,
    pub p90_s: f64,
    pub max_s: f64,
}

/// Mirror of `WorkerCounts`.
#[derive(Clone, Debug, Default, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct WorkerCounts {
    pub total: u32,
    pub ready: u32,
    pub busy: u32,
    pub idle: u32,
    pub initializing: u32,
    pub unhealthy: u32,
}

/// Mirror of `fastvideo_serve::gateway::scale::PoolMetrics`: what the
/// gateway's `PoolScaler::observe` hands over every tick, and what
/// `GET /fv/v1/gateway/pools` returns (`{"pools": [..]}`).
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct GatewayPoolMetrics {
    pub pool: String,
    pub kind: GatewayPoolKind,
    #[serde(default)]
    pub endpoint_id: Option<String>,
    pub at_unix_ms: i64,
    pub queued: u32,
    pub running: u32,
    pub oldest_queued_age_s: f64,
    pub streams: u32,
    #[serde(default)]
    pub run_time: DurationStats,
    #[serde(default)]
    pub queue_wait: DurationStats,
    #[serde(default)]
    pub window_s: u64,
    #[serde(default)]
    pub workers: WorkerCounts,
    #[serde(default)]
    pub available: bool,
    #[serde(default)]
    pub max_queued: u32,
    #[serde(default)]
    pub max_streams: u32,
}

struct Latest {
    m: GatewayPoolMetrics,
    /// (seconds, queued + running + streams) of the previous update.
    prev: Option<(f64, u32)>,
    rate: Option<f64>,
}

/// Holds the gateway's latest [`GatewayPoolMetrics`] per pool and serves
/// them as [`PoolSignals`]. The gateway's `PoolScaler::observe` (or the
/// HTTP poller) calls [`GatewaySignals::update`].
///
/// Arrival rate: throughput over the gateway's window
/// (`run_time.count / window_s`) plus the growth of queued + running
/// work since the last update, smoothed.
#[derive(Default)]
pub struct GatewaySignals {
    inner: Mutex<BTreeMap<String, Latest>>,
}

impl GatewaySignals {
    pub fn update(&self, pools: &[GatewayPoolMetrics]) {
        let mut g = self.inner.lock().unwrap_or_else(|p| p.into_inner());
        for m in pools {
            let t = m.at_unix_ms as f64 / 1000.0;
            let load = m.queued + m.running + m.streams;
            let e = g.entry(m.pool.clone()).or_insert(Latest { m: m.clone(), prev: None, rate: None });
            let throughput = if m.window_s > 0 { f64::from(m.run_time.count) / m.window_s as f64 } else { 0.0 };
            let growth = match e.prev {
                Some((t0, l0)) if t > t0 => (f64::from(load) - f64::from(l0)) / (t - t0),
                _ => 0.0,
            };
            let now_rate = (throughput + growth).max(0.0);
            e.rate = Some(match e.rate {
                Some(r) => r + 0.3 * (now_rate - r),
                None => now_rate,
            });
            e.prev = Some((t, load));
            e.m = m.clone();
        }
    }

    /// The latest metrics of a pool.
    pub fn latest(&self, pool: &str) -> Option<GatewayPoolMetrics> {
        self.inner.lock().unwrap_or_else(|p| p.into_inner()).get(pool).map(|e| e.m.clone())
    }
}

#[async_trait::async_trait]
impl SignalSource for GatewaySignals {
    async fn signals(&self, pools: &[String]) -> Vec<PoolSignals> {
        let g = self.inner.lock().unwrap_or_else(|p| p.into_inner());
        pools
            .iter()
            .filter_map(|p| g.get(p))
            .map(|e| PoolSignals {
                pool: e.m.pool.clone(),
                queued: e.m.queued,
                oldest_queued_s: e.m.oldest_queued_age_s,
                running_jobs: e.m.running,
                live_streams: e.m.streams,
                arrivals_total: 0,
                recent_job_s: (e.m.run_time.count > 0 && e.m.run_time.mean_s > 0.0).then_some(e.m.run_time.mean_s),
                arrival_rate_per_s: e.rate,
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn metrics(t_ms: i64, queued: u32, running: u32, finished: u32) -> GatewayPoolMetrics {
        GatewayPoolMetrics {
            pool: "h3-turbo".into(),
            kind: GatewayPoolKind::RunpodServerless,
            endpoint_id: Some("ep".into()),
            at_unix_ms: t_ms,
            queued,
            running,
            oldest_queued_age_s: if queued > 0 { 12.0 } else { 0.0 },
            streams: 0,
            run_time: DurationStats { count: finished, mean_s: 24.0, p50_s: 24.0, p90_s: 30.0, max_s: 31.0 },
            queue_wait: DurationStats::default(),
            window_s: 300,
            workers: WorkerCounts::default(),
            available: true,
            max_queued: 32,
            max_streams: 0,
        }
    }

    #[test]
    fn parses_the_gateway_json() {
        let j = r#"{"object":"fv.gateway.pools","pools":[{"pool":"wan-turbo","kind":"runpod-serverless",
            "endpoint_id":"abc","at_unix_ms":1790553600000,"queued":2,"running":1,"oldest_queued_age_s":3.5,
            "streams":0,"run_time":{"count":4,"mean_s":7.0,"p50_s":6.9,"p90_s":7.5,"max_s":8.0},
            "queue_wait":{"count":4,"mean_s":1.0,"p50_s":1.0,"p90_s":2.0,"max_s":2.0},"window_s":300,
            "workers":{"total":1,"ready":1,"busy":1,"idle":0,"initializing":0,"unhealthy":0},
            "available":true,"max_queued":32,"max_streams":0}]}"#;
        #[derive(serde::Deserialize)]
        struct Pools {
            pools: Vec<GatewayPoolMetrics>,
        }
        let p: Pools = serde_json::from_str(j).unwrap();
        assert_eq!(p.pools[0].kind, GatewayPoolKind::RunpodServerless);
        assert_eq!(p.pools[0].workers.busy, 1);
    }

    #[tokio::test]
    async fn metrics_become_signals_with_a_rate() {
        let s = GatewaySignals::default();
        s.update(&[metrics(1_000_000, 0, 1, 30)]); // 30 finished in 300 s: 0.1/s
        let v = s.signals(&["h3-turbo".into(), "other".into()]).await;
        assert_eq!(v.len(), 1);
        assert_eq!(v[0].running_jobs, 1);
        assert_eq!(v[0].recent_job_s, Some(24.0));
        assert!((v[0].arrival_rate_per_s.unwrap() - 0.1).abs() < 1e-9);
        // 10 s later 5 more jobs are waiting: growth 0.5/s on top, smoothed.
        s.update(&[metrics(1_010_000, 5, 1, 30)]);
        let v = s.signals(&["h3-turbo".into()]).await;
        let r = v[0].arrival_rate_per_s.unwrap();
        assert!((r - (0.1 + 0.3 * 0.5)).abs() < 1e-9, "{r}");
        assert_eq!((v[0].queued, v[0].oldest_queued_s), (5, 12.0));
    }
}
