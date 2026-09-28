//! The autoscaler interface (docs/serve/gateway.md §7): what the gateway
//! measures per pool and the hook an autoscaler implements
//! (`crates/fastvideo-autoscale`). These types are the contract; the
//! gateway fills them from D1 (queue depth, durations, leases) and from
//! the pools (Runpod `/health`, pod probes and registrations).

use serde::{Deserialize, Serialize};

pub use crate::config::PoolKind;

/// Summary of durations (seconds) over a window.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct DurationStats {
    pub count: u32,
    pub mean_s: f64,
    pub p50_s: f64,
    pub p90_s: f64,
    pub max_s: f64,
}

impl DurationStats {
    /// Stats of `v` (any order; non-finite values are skipped).
    pub fn of(v: &[f64]) -> Self {
        let mut v: Vec<f64> = v.iter().copied().filter(|x| x.is_finite()).collect();
        if v.is_empty() {
            return Self::default();
        }
        v.sort_by(|a, b| a.total_cmp(b));
        let n = v.len();
        let pick = |q: f64| v[((q * (n - 1) as f64).round() as usize).min(n - 1)];
        Self {
            count: n as u32,
            mean_s: v.iter().sum::<f64>() / n as f64,
            p50_s: pick(0.5),
            p90_s: pick(0.9),
            max_s: v[n - 1],
        }
    }
}

/// Workers of a pool as the gateway sees them (serverless: Runpod
/// `/health`; pods: static and registered workers and their probes).
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorkerCounts {
    pub total: u32,
    pub ready: u32,
    pub busy: u32,
    pub idle: u32,
    pub initializing: u32,
    pub unhealthy: u32,
}

/// One pool at one instant.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PoolMetrics {
    pub pool: String,
    pub kind: PoolKind,
    /// Serverless: the Runpod endpoint id.
    pub endpoint_id: Option<String>,
    pub at_unix_ms: i64,
    /// Jobs dispatched to this pool and not started (D1).
    pub queued: u32,
    /// Jobs started and not finished (D1).
    pub running: u32,
    /// Age of the oldest queued job (0 when none).
    pub oldest_queued_age_s: f64,
    /// Live stream / peer-session leases (D1).
    pub streams: u32,
    /// started → finished of the jobs finished within `window_s`.
    pub run_time: DurationStats,
    /// created → started of the same jobs.
    pub queue_wait: DurationStats,
    pub window_s: u64,
    pub workers: WorkerCounts,
    /// Whether the gateway would dispatch to this pool now.
    pub available: bool,
    /// Admission limit on queued jobs (0: none).
    pub max_queued: u32,
    /// Stream sessions limit (0: one per pod worker; serverless: none).
    pub max_streams: u32,
    /// Jobs ever dispatched to this pool (monotonic, from D1; a re-dispatch
    /// after a worker loss does not count again): the arrival rate is its
    /// difference over time.
    #[serde(default)]
    pub submitted_total: u64,
}

/// Implemented by the autoscaler. Each gateway replica calls its hooks
/// after every metrics tick (`gateway.tick_s`) with every pool.
#[async_trait::async_trait]
pub trait PoolScaler: Send + Sync + 'static {
    async fn observe(&self, pools: &[PoolMetrics]);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn duration_stats() {
        assert_eq!(DurationStats::of(&[]), DurationStats::default());
        let s = DurationStats::of(&[3.0, 1.0, 2.0, f64::NAN, 10.0]);
        assert_eq!(s.count, 4);
        assert_eq!(s.max_s, 10.0);
        assert_eq!(s.mean_s, 4.0);
        assert_eq!(s.p50_s, 3.0);
        assert_eq!(s.p90_s, 10.0);
    }

    #[test]
    fn metrics_json_shape() {
        let m = PoolMetrics {
            pool: "h3-turbo".into(),
            kind: PoolKind::RunpodServerless,
            endpoint_id: Some("ep".into()),
            at_unix_ms: 1,
            queued: 2,
            running: 1,
            oldest_queued_age_s: 4.5,
            streams: 0,
            run_time: DurationStats::default(),
            queue_wait: DurationStats::default(),
            window_s: 600,
            workers: WorkerCounts::default(),
            available: true,
            max_queued: 32,
            max_streams: 0,
            submitted_total: 9,
        };
        let v = serde_json::to_value(&m).unwrap();
        assert_eq!(v["kind"], "runpod-serverless");
        assert_eq!(v["workers"]["idle"], 0);
        assert_eq!(serde_json::from_value::<PoolMetrics>(v).unwrap(), m);
    }
}
