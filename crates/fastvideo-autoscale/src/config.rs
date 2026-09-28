//! `[autoscale]` configuration (docs/serve/gateway.md, "Autoscaling").
//!
//! Every field has a default so a pool needs only `name`, `family` and
//! its provider section. Durations are seconds, money is US dollars.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

/// Top-level `[autoscale]` table.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct AutoscaleConfig {
    /// Master switch: without it the controller never starts.
    pub enabled: bool,
    /// Decide and log, never call a provider's write API.
    pub dry_run: bool,
    /// Seconds between controller ticks.
    pub interval_s: f64,
    /// Global $/hr cap across every pool (0 = none).
    pub budget_usd_per_hr: f64,
    /// Account balance floor: below it the controller is in hard stop
    /// (no new workers; serverless min 0, max = busy workers).
    pub balance_floor_usd: f64,
    /// Seconds between balance reads (the Runpod GraphQL `clientBalance`).
    pub balance_interval_s: f64,
    /// $/hr per GPU type id (Runpod ids, e.g. "NVIDIA H100 80GB HBM3").
    /// Serverless pools may use a separate id such as "serverless:NVIDIA H100 80GB HBM3".
    pub prices: BTreeMap<String, f64>,
    /// Leader election.
    pub lease: LeaseConfig,
    /// Number of decisions kept for `/fv/v1/admin/autoscale`.
    pub history: usize,
    #[serde(rename = "pools")]
    pub pools: Vec<PoolConfig>,
}

impl Default for AutoscaleConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            dry_run: true,
            interval_s: 15.0,
            budget_usd_per_hr: 0.0,
            balance_floor_usd: 8.0,
            balance_interval_s: 300.0,
            prices: default_prices(),
            lease: LeaseConfig::default(),
            history: 200,
            pools: Vec::new(),
        }
    }
}

/// Prices as of 2026-09-28. Pods: Runpod GraphQL `gpuTypes.securePrice`.
/// Serverless (`serverless:` prefix): flex per-second price × 3600; the
/// H100 value matches the WP-18 bill (13 min ≈ $0.9).
pub fn default_prices() -> BTreeMap<String, f64> {
    [
        ("NVIDIA H200", 4.59),
        ("NVIDIA H100 80GB HBM3", 3.49),
        ("NVIDIA H100 NVL", 3.19),
        ("NVIDIA H100 PCIe", 2.89),
        ("NVIDIA RTX PRO 6000 Blackwell Server Edition", 2.09),
        ("NVIDIA RTX PRO 6000 Blackwell Workstation Edition", 2.19),
        ("NVIDIA L40S", 1.09),
        ("NVIDIA RTX 6000 Ada Generation", 0.84),
        ("NVIDIA A100-SXM4-80GB", 1.59),
        ("NVIDIA GeForce RTX 4090", 0.74),
        ("serverless:NVIDIA H200", 5.58),
        ("serverless:NVIDIA H100 80GB HBM3", 4.18),
        ("serverless:NVIDIA H100 NVL", 4.18),
        ("serverless:NVIDIA RTX PRO 6000 Blackwell Server Edition", 3.24),
        ("serverless:NVIDIA L40S", 1.91),
        ("serverless:NVIDIA GeForce RTX 4090", 1.12),
    ]
    .into_iter()
    .map(|(k, v)| (k.to_owned(), v))
    .collect()
}

impl AutoscaleConfig {
    /// Parses the `[autoscale]` table of a TOML document (the whole
    /// document may be a gateway config; other tables are ignored).
    pub fn from_toml_document(text: &str) -> Result<Self, ConfigError> {
        #[derive(Deserialize)]
        struct Doc {
            #[serde(default)]
            autoscale: Option<toml::Value>,
        }
        let doc: Doc = toml::from_str(text).map_err(|e| ConfigError::Parse(e.to_string()))?;
        let Some(v) = doc.autoscale else { return Ok(Self::default()) };
        let cfg: Self = v.try_into().map_err(|e: toml::de::Error| ConfigError::Parse(e.to_string()))?;
        cfg.validate()?;
        Ok(cfg)
    }

    /// Price of one worker of `pool` in $/hr (0 when unknown, which the
    /// validation rejects for pools with a budget).
    pub fn price(&self, pool: &PoolConfig) -> f64 {
        if let Some(p) = pool.price_usd_per_hr {
            return p;
        }
        let key = match pool.kind {
            PoolKind::Serverless => format!("serverless:{}", pool.gpu_type()),
            PoolKind::Pod => pool.gpu_type().to_owned(),
        };
        self.prices.get(&key).or_else(|| self.prices.get(pool.gpu_type())).copied().unwrap_or(0.0)
    }

    pub fn validate(&self) -> Result<(), ConfigError> {
        let bad = |m: String| Err(ConfigError::Invalid(m));
        if self.interval_s.is_nan() || self.interval_s <= 0.0 {
            return bad("autoscale.interval_s must be > 0".into());
        }
        let mut names = std::collections::BTreeSet::new();
        for p in &self.pools {
            if !names.insert(p.name.as_str()) {
                return bad(format!("duplicate pool {}", p.name));
            }
            if p.name.is_empty() || !p.name.chars().all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_') {
                return bad(format!("pool name {:?} must be [A-Za-z0-9_-]+", p.name));
            }
            if p.min_workers > p.max_workers {
                return bad(format!("pool {}: min_workers > max_workers", p.name));
            }
            if p.jobs_per_worker == 0 || p.streams_per_worker == 0 {
                return bad(format!("pool {}: jobs_per_worker and streams_per_worker must be >= 1", p.name));
            }
            if p.slo_queue_wait_s.is_nan() || p.slo_queue_wait_s <= 0.0 {
                return bad(format!("pool {}: slo_queue_wait_s must be > 0", p.name));
            }
            if !(0.0..1.0).contains(&p.hysteresis) {
                return bad(format!("pool {}: hysteresis must be in [0, 1)", p.name));
            }
            for s in &p.schedule {
                if s.start_hour > 24 || s.end_hour > 24 || s.days.iter().any(|d| *d > 6) {
                    return bad(format!("pool {}: schedule hours are 0-24, days 0 (Mon)-6 (Sun)", p.name));
                }
            }
            if (p.budget_usd_per_hr > 0.0 || self.budget_usd_per_hr > 0.0) && self.price(p) <= 0.0 {
                return bad(format!("pool {}: no price for {:?} (set prices or price_usd_per_hr)", p.name, p.gpu_type()));
            }
            match p.kind {
                // The gateway fills an empty endpoint id from its `[[pools]]`.
                PoolKind::Serverless => {}
                PoolKind::Pod => {
                    if p.pod.template_id.is_empty() || p.pod.gpu_types.is_empty() || p.pod.placements.is_empty() {
                        return bad(format!("pool {}: pod.template_id, pod.gpu_types and pod.placements are required", p.name));
                    }
                }
            }
        }
        Ok(())
    }
}

#[derive(Debug, thiserror::Error, PartialEq)]
pub enum ConfigError {
    #[error("autoscale config: {0}")]
    Parse(String),
    #[error("autoscale config: {0}")]
    Invalid(String),
}

/// Leader lease (one row per `name` in a D1 table).
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct LeaseConfig {
    /// `memory` (single replica, tests) or `d1`.
    pub backend: LeaseBackend,
    /// Lease row name (one controller per name).
    pub name: String,
    /// A holder keeps the lease this long without renewing.
    pub ttl_s: f64,
}

impl Default for LeaseConfig {
    fn default() -> Self {
        Self { backend: LeaseBackend::Memory, name: "fv-autoscale".into(), ttl_s: 60.0 }
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum LeaseBackend {
    Memory,
    D1,
}

/// How workers of a pool come to exist.
#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum PoolKind {
    /// A Runpod serverless endpoint: Runpod scales, we steer min/max.
    /// (`runpod-serverless` as in the gateway's `[[pools]] kind`.)
    #[default]
    #[serde(alias = "runpod-serverless")]
    Serverless,
    /// Runpod pods created and deleted by the controller.
    Pod,
}

/// One GPU worker pool (one model family).
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct PoolConfig {
    pub name: String,
    /// Model family (`wan`, `h3`, `ltx`, ...); selects simulator defaults.
    pub family: String,
    pub kind: PoolKind,
    pub min_workers: u32,
    pub max_workers: u32,
    /// Workers kept warm even when idle (a floor above `min_workers`).
    pub warm_min: u32,
    /// Time-of-day floors.
    pub schedule: Vec<ScheduleRule>,
    /// Offset of the schedule's clock from UTC, minutes (e.g. -420 for PDT).
    pub schedule_utc_offset_min: i32,
    /// Batch jobs one worker runs at once (1 per GPU).
    pub jobs_per_worker: u32,
    /// Live streams one worker holds at once (1 per GPU).
    pub streams_per_worker: u32,
    /// Target queue wait (the p95 goal).
    pub slo_queue_wait_s: f64,
    /// Scale up when the oldest queued job is older than this share of the SLO.
    pub slo_breach_fraction: f64,
    /// Max workers added per decision (0 = no limit).
    pub scale_up_step: u32,
    /// Min seconds between scale-ups (a queue-age breach bypasses it).
    pub scale_up_cooldown_s: f64,
    /// Max workers removed per decision.
    pub scale_down_step: u32,
    /// Demand must stay below capacity this long before the first
    /// scale-down of an idle period.
    pub idle_timeout_s: f64,
    /// Min seconds after any scale change before a (further) scale-down.
    pub scale_down_cooldown_s: f64,
    /// Scale down only when demand ≤ (workers - 1) × (1 - hysteresis).
    pub hysteresis: f64,
    /// Expected worker start + model load (seconds). Measured values
    /// replace it once the controller has seen cold starts.
    pub cold_start_s: f64,
    /// Job duration used until real samples arrive.
    pub default_job_s: f64,
    /// Target utilization for the steady-state (Little's law) term.
    pub target_utilization: f64,
    /// Seconds of arrivals used for the rate and growth estimates.
    pub rate_window_s: f64,
    /// Pool $/hr cap (0 = none).
    pub budget_usd_per_hr: f64,
    /// Price override in $/hr per worker.
    pub price_usd_per_hr: Option<f64>,
    /// Higher wins when the global budget cannot fund every pool.
    pub priority: i32,
    pub serverless: ServerlessConfig,
    pub pod: PodConfig,
}

impl Default for PoolConfig {
    fn default() -> Self {
        Self {
            name: String::new(),
            family: String::new(),
            kind: PoolKind::Serverless,
            min_workers: 0,
            max_workers: 2,
            warm_min: 0,
            schedule: Vec::new(),
            schedule_utc_offset_min: 0,
            jobs_per_worker: 1,
            streams_per_worker: 1,
            slo_queue_wait_s: 60.0,
            slo_breach_fraction: 0.5,
            scale_up_step: 2,
            scale_up_cooldown_s: 30.0,
            scale_down_step: 1,
            idle_timeout_s: 300.0,
            scale_down_cooldown_s: 120.0,
            hysteresis: 0.2,
            cold_start_s: 130.0,
            default_job_s: 30.0,
            target_utilization: 0.8,
            rate_window_s: 300.0,
            budget_usd_per_hr: 0.0,
            price_usd_per_hr: None,
            priority: 0,
            serverless: ServerlessConfig::default(),
            pod: PodConfig::default(),
        }
    }
}

impl PoolConfig {
    /// The GPU type used for pricing: the first preference.
    pub fn gpu_type(&self) -> &str {
        match self.kind {
            PoolKind::Serverless => self.serverless.gpu_type.as_str(),
            PoolKind::Pod => self.pod.gpu_types.first().map(String::as_str).unwrap_or(""),
        }
    }

    /// The schedule floor at `now_unix_s`.
    pub fn scheduled_min(&self, now_unix_s: f64) -> u32 {
        let local = now_unix_s + f64::from(self.schedule_utc_offset_min) * 60.0;
        let day_index = (local / 86_400.0).floor() as i64;
        // 1970-01-01 was a Thursday: Monday = 0.
        let weekday = (day_index + 3).rem_euclid(7) as u8;
        let hour = (local.rem_euclid(86_400.0) / 3600.0).floor() as u8;
        self.schedule.iter().filter(|r| r.matches(weekday, hour)).map(|r| r.min_workers).max().unwrap_or(0)
    }
}

/// `min_workers` from `start_hour` (inclusive) to `end_hour` (exclusive)
/// in the pool's schedule clock; `end_hour < start_hour` wraps midnight.
/// Empty `days` = every day (0 = Monday … 6 = Sunday).
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ScheduleRule {
    #[serde(default)]
    pub days: Vec<u8>,
    pub start_hour: u8,
    pub end_hour: u8,
    pub min_workers: u32,
}

impl ScheduleRule {
    fn matches(&self, weekday: u8, hour: u8) -> bool {
        let day_ok = |d: u8| self.days.is_empty() || self.days.contains(&d);
        if self.start_hour <= self.end_hour {
            day_ok(weekday) && hour >= self.start_hour && hour < self.end_hour
        } else if hour >= self.start_hour {
            day_ok(weekday)
        } else {
            // After midnight: the rule started the previous day.
            hour < self.end_hour && day_ok((weekday + 6) % 7)
        }
    }
}

/// Runpod serverless steering (REST v1 `PATCH /endpoints/{id}`).
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct ServerlessConfig {
    pub endpoint_id: String,
    /// GPU type for pricing.
    pub gpu_type: String,
    /// `QUEUE_DELAY` (seconds a job waits before Runpod adds a worker) or
    /// `REQUEST_COUNT` (jobs per worker).
    pub scaler_type: String,
    pub scaler_value: u32,
    /// Runpod's own idle timeout (seconds) for workers above `workersMin`.
    pub idle_timeout_s: u32,
}

impl Default for ServerlessConfig {
    fn default() -> Self {
        Self {
            endpoint_id: String::new(),
            gpu_type: "NVIDIA H100 80GB HBM3".into(),
            scaler_type: "QUEUE_DELAY".into(),
            scaler_value: 2,
            idle_timeout_s: 60,
        }
    }
}

/// Runpod pods from a template (REST v1 `POST /pods`).
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct PodConfig {
    pub template_id: String,
    /// Preference order; the controller falls back along it when a type
    /// is out of stock.
    pub gpu_types: Vec<String>,
    /// Region + volume pairs, tried in order.
    pub placements: Vec<Placement>,
    pub cloud_type: String,
    /// Pods older than this are drained and replaced.
    pub max_lifetime_s: f64,
    /// A pod that is not ready this long after create is deleted.
    pub boot_timeout_s: f64,
    /// Worker URL; `{id}` is the pod id.
    pub url_template: String,
    /// Health path answering 200 when ready (204 while loading).
    pub ready_path: String,
    /// Refuse a pod whose `costPerHr` exceeds this (0 = the price table × 1.25).
    pub max_usd_per_hr: f64,
}

impl Default for PodConfig {
    fn default() -> Self {
        Self {
            template_id: String::new(),
            gpu_types: Vec::new(),
            placements: vec![
                Placement { data_center: "US-CA-2".into(), volume_id: "s2k01690bi".into() },
                Placement { data_center: "EUR-IS-1".into(), volume_id: "jg48s6o1w0".into() },
            ],
            cloud_type: "SECURE".into(),
            max_lifetime_s: 12.0 * 3600.0,
            boot_timeout_s: 1200.0,
            url_template: "https://{id}-8000.proxy.runpod.net".into(),
            ready_path: "/ping".into(),
            max_usd_per_hr: 0.0,
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Placement {
    pub data_center: String,
    pub volume_id: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_example_and_validates() {
        let text = r#"
            [server]
            bind = "0.0.0.0:8000"

            [autoscale]
            enabled = true
            dry_run = false
            budget_usd_per_hr = 20.0

            [[autoscale.pools]]
            name = "wan"
            family = "wan"
            max_workers = 3
            warm_min = 1
            schedule = [{ start_hour = 9, end_hour = 18, min_workers = 2, days = [0,1,2,3,4] }]
            serverless = { endpoint_id = "abc", gpu_type = "NVIDIA H100 80GB HBM3" }

            [[autoscale.pools]]
            name = "h3"
            family = "h3"
            kind = "pod"
            pod = { template_id = "t1", gpu_types = ["NVIDIA H200"] }
        "#;
        let c = AutoscaleConfig::from_toml_document(text).unwrap();
        assert!(c.enabled && !c.dry_run);
        assert_eq!(c.pools.len(), 2);
        assert_eq!(c.price(&c.pools[0]), 4.18);
        assert_eq!(c.price(&c.pools[1]), 4.59);
        assert_eq!(c.pools[1].pod.placements.len(), 2);
    }

    #[test]
    fn committed_example_parses() {
        let c = AutoscaleConfig::from_toml_document(include_str!("../../../configs/serve/autoscale.toml")).unwrap();
        assert!(c.enabled && c.dry_run);
        assert_eq!(c.lease.backend, LeaseBackend::D1);
        let names: Vec<&str> = c.pools.iter().map(|p| p.name.as_str()).collect();
        assert_eq!(names, ["h3-turbo", "wan", "ltx", "sfwan-live"]);
        let live = &c.pools[3];
        assert_eq!(live.kind, PoolKind::Pod);
        assert_eq!(live.pod.placements[1].volume_id, "jg48s6o1w0");
        assert_eq!(c.price(live), 2.09);
        // 10:00 PDT on a Monday = 17:00 UTC.
        assert_eq!(c.pools[0].scheduled_min(1_790_553_600.0 + 17.0 * 3600.0), 1);
    }

    #[test]
    fn missing_table_is_disabled() {
        let c = AutoscaleConfig::from_toml_document("[server]\nbind='x'\n").unwrap();
        assert!(!c.enabled);
    }

    #[test]
    fn rejects_bad_pools() {
        let t = "[autoscale]\n[[autoscale.pools]]\nname='a'\nmin_workers=3\nmax_workers=1\nserverless={endpoint_id='e'}\n";
        assert!(AutoscaleConfig::from_toml_document(t).is_err());
        let t = "[autoscale]\n[[autoscale.pools]]\nname='a'\nkind='pod'\n";
        assert!(AutoscaleConfig::from_toml_document(t).is_err(), "pod template required");
        let t = "[autoscale]\nbogus=1\n";
        assert!(AutoscaleConfig::from_toml_document(t).is_err());
    }

    #[test]
    fn schedule_floor_by_weekday_and_hour() {
        let p = PoolConfig {
            schedule: vec![
                ScheduleRule { days: vec![0, 1, 2, 3, 4], start_hour: 9, end_hour: 18, min_workers: 2 },
                ScheduleRule { days: vec![4], start_hour: 22, end_hour: 2, min_workers: 1 },
            ],
            ..PoolConfig::default()
        };
        // 2026-09-28 is a Monday. 00:00 UTC:
        let monday = 1_790_553_600.0;
        assert_eq!(p.scheduled_min(monday + 10.0 * 3600.0), 2);
        assert_eq!(p.scheduled_min(monday + 8.0 * 3600.0), 0);
        assert_eq!(p.scheduled_min(monday + 5.0 * 86_400.0 + 10.0 * 3600.0), 0, "Saturday");
        // Friday 23:00 and Saturday 01:00 (wrapped rule of Friday).
        assert_eq!(p.scheduled_min(monday + 4.0 * 86_400.0 + 23.0 * 3600.0), 1);
        assert_eq!(p.scheduled_min(monday + 5.0 * 86_400.0 + 3600.0), 1);
        assert_eq!(p.scheduled_min(monday + 6.0 * 86_400.0 + 3600.0), 0);
        // Offset clock: 10:00 local at UTC-7 is 17:00 UTC.
        let q = PoolConfig { schedule_utc_offset_min: -420, ..p.clone() };
        assert_eq!(q.scheduled_min(monday + 17.0 * 3600.0), 2);
    }
}
