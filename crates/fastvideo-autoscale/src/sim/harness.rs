//! Replays traces against [`SimWorld`] under a strategy and reports queue
//! wait, GPU-hours, cost and cold starts (`fv-autoscale-sim` prints the
//! table committed in docs/serve/gateway.md).
//!
//! Strategies:
//! - `autoscaler`: the real [`Controller`] (policy + sim provider + memory
//!   lease) steering a serverless endpoint every `interval_s`;
//! - `runpod-only`: the endpoint left at min 0 / max N with Runpod's scaler
//!   and idle timeout alone (what `runpod-endpoint.sh` deploys);
//! - `always-on`: min = max = N.

use std::sync::{Arc, Mutex};

use serde::Serialize;

use super::trace::{self, Arrival, Rng};
use super::{FamilyProfile, SimProvider, SimSignals, SimWorld};
use crate::config::{AutoscaleConfig, PoolConfig, PoolKind, ScheduleRule, ServerlessConfig};
use crate::controller::Controller;
use crate::lease::MemoryLease;
use crate::provider::Providers;
use crate::types::EndpointSettings;

/// 2026-09-28 00:00 UTC, a Monday (schedules see real weekdays).
pub const SIM_EPOCH: f64 = 1_790_553_600.0;

#[derive(Clone, Debug, PartialEq, Serialize)]
pub enum Strategy {
    Autoscaler,
    /// The autoscaler with `warm_min` workers kept warm.
    AutoscalerWarm(u32),
    RunpodOnly,
    AlwaysOn(u32),
}

impl Strategy {
    pub fn label(&self) -> String {
        match self {
            Strategy::Autoscaler => "autoscaler".into(),
            Strategy::AutoscalerWarm(n) => format!("autoscaler, warm {n}"),
            Strategy::RunpodOnly => "runpod-only".into(),
            Strategy::AlwaysOn(n) => format!("always-on {n}"),
        }
    }
}

#[derive(Clone, Debug)]
pub struct Scenario {
    pub trace: String,
    pub profile: FamilyProfile,
    pub pool: PoolConfig,
    pub arrivals: Vec<Arrival>,
    /// Trace length; the run continues until the work drains (≤ 1 h more).
    pub hours: f64,
    pub strategy: Strategy,
    pub seed: u64,
    pub price_usd_per_hr: f64,
}

#[derive(Clone, Debug, Serialize)]
pub struct Report {
    pub trace: String,
    pub family: String,
    pub strategy: String,
    pub jobs: u64,
    pub wait_p50_s: f64,
    pub wait_p95_s: f64,
    pub wait_max_s: f64,
    pub slo_s: f64,
    /// Share of jobs that waited at most the SLO.
    pub within_slo: f64,
    pub gpu_hours: f64,
    pub cost_usd: f64,
    pub cold_starts: u64,
    pub uncached_starts: u64,
    pub max_workers: u32,
    pub busy_delete_attempts: u64,
    pub unfinished: u64,
    pub endpoint_patches: u64,
    /// Per minute: (minutes, queued, workers, endpoint min).
    #[serde(skip)]
    pub timeline: Vec<(f64, usize, usize, u32)>,
    /// Every job's wait, sorted.
    #[serde(skip)]
    pub waits: Vec<f64>,
    /// Runs merged into this report.
    pub runs: u32,
}

pub fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let rank = (p / 100.0 * (sorted.len() as f64 - 1.0)).round() as usize;
    sorted[rank.min(sorted.len() - 1)]
}

/// The recommended serverless pool for a family (the simulated config).
pub fn pool_for(profile: &FamilyProfile, max_workers: u32) -> PoolConfig {
    let slo = match profile.family.as_str() {
        "wan" => 60.0,
        _ => 120.0,
    };
    PoolConfig {
        name: profile.family.clone(),
        family: profile.family.clone(),
        kind: PoolKind::Serverless,
        max_workers,
        slo_queue_wait_s: slo,
        cold_start_s: profile.cold_start_s(),
        default_job_s: profile.job_s,
        idle_timeout_s: 300.0,
        scale_down_cooldown_s: 120.0,
        scale_up_cooldown_s: 30.0,
        scale_up_step: 2,
        serverless: ServerlessConfig {
            endpoint_id: format!("sim-{}", profile.family),
            gpu_type: "NVIDIA H100 80GB HBM3".into(),
            // Runpod's own scaler as a backstop only: it adds a worker for
            // a job that waited half the SLO; the policy leads.
            scaler_type: "QUEUE_DELAY".into(),
            scaler_value: (slo * 0.5) as u32,
            idle_timeout_s: 60,
        },
        ..PoolConfig::default()
    }
}

pub fn run(s: &Scenario) -> Report {
    let rt = tokio::runtime::Builder::new_current_thread().build().expect("tokio runtime");
    rt.block_on(run_async(s))
}

async fn run_async(s: &Scenario) -> Report {
    let world = Arc::new(Mutex::new(SimWorld::new(SIM_EPOCH, s.seed, 1_000.0)));
    let endpoint = match &s.strategy {
        Strategy::Autoscaler | Strategy::AutoscalerWarm(_) => None,
        // Runpod console defaults.
        Strategy::RunpodOnly => Some(EndpointSettings {
            workers_min: 0,
            workers_max: s.pool.max_workers,
            idle_timeout_s: 5,
            scaler_type: "QUEUE_DELAY".into(),
            scaler_value: 4,
        }),
        Strategy::AlwaysOn(n) => Some(EndpointSettings {
            workers_min: *n,
            workers_max: *n,
            idle_timeout_s: s.pool.serverless.idle_timeout_s,
            scaler_type: s.pool.serverless.scaler_type.clone(),
            scaler_value: s.pool.serverless.scaler_value,
        }),
    };
    {
        let mut w = world.lock().unwrap_or_else(|p| p.into_inner());
        w.add_pool(s.pool.clone(), s.profile.clone(), s.price_usd_per_hr, endpoint);
        w.load_trace(&s.pool.name, s.arrivals.clone());
    }
    let warm = match s.strategy {
        Strategy::AutoscalerWarm(n) => Some(n),
        Strategy::Autoscaler => Some(0),
        _ => None,
    };
    let ctrl = if let Some(warm_min) = warm {
        let cfg = AutoscaleConfig {
            enabled: true,
            dry_run: false,
            interval_s: 15.0,
            balance_interval_s: 60.0,
            pools: vec![PoolConfig { price_usd_per_hr: Some(s.price_usd_per_hr), warm_min, ..s.pool.clone() }],
            ..AutoscaleConfig::default()
        };
        let sim = Arc::new(SimProvider(world.clone()));
        let providers = Providers::new(sim.clone()).with(PoolKind::Serverless, sim.clone()).with(PoolKind::Pod, sim);
        Some(Controller::new(cfg, providers, Arc::new(SimSignals(world.clone())), Arc::new(MemoryLease::default()), "sim"))
    } else {
        None
    };
    let end = SIM_EPOCH + s.hours * 3600.0;
    let hard_end = end + 3600.0;
    let mut t = SIM_EPOCH;
    let mut timeline = Vec::new();
    loop {
        if let Some(c) = &ctrl {
            c.tick(t).await;
        }
        t += 15.0;
        let mut w = world.lock().unwrap_or_else(|p| p.into_inner());
        w.advance_to(t);
        if ((t - SIM_EPOCH) % 60.0).abs() < 1e-6 {
            let q = w.queued(&s.pool.name);
            let min = w.endpoint(&s.pool.name).map_or(0, |e| e.workers_min);
            timeline.push(((t - SIM_EPOCH) / 60.0, q, w.worker_count(&s.pool.name), min));
        }
        if t >= hard_end || (t >= end && w.outstanding(&s.pool.name) == 0) {
            w.finish();
            break;
        }
    }
    let st = world.lock().unwrap_or_else(|p| p.into_inner()).stats(&s.pool.name);
    let mut waits = st.waits.clone();
    waits.sort_by(f64::total_cmp);
    let within = if waits.is_empty() {
        1.0
    } else {
        waits.iter().filter(|w| **w <= s.pool.slo_queue_wait_s).count() as f64 / waits.len() as f64
    };
    Report {
        trace: s.trace.clone(),
        family: s.profile.family.clone(),
        strategy: s.strategy.label(),
        jobs: st.jobs + st.streams,
        wait_p50_s: percentile(&waits, 50.0),
        wait_p95_s: percentile(&waits, 95.0),
        wait_max_s: waits.last().copied().unwrap_or(0.0),
        slo_s: s.pool.slo_queue_wait_s,
        within_slo: within,
        gpu_hours: st.gpu_seconds / 3600.0,
        cost_usd: st.cost_usd,
        cold_starts: st.cold_starts,
        uncached_starts: st.uncached_starts,
        max_workers: st.max_workers,
        busy_delete_attempts: st.busy_delete_attempts,
        unfinished: st.unfinished,
        endpoint_patches: st.endpoint_patches,
        timeline,
        waits,
        runs: 1,
    }
}

/// Average jobs per hour per family in the simulated traces (an
/// assumption: enough load that the busiest hour needs 1-3 GPUs).
pub fn avg_per_hr(family: &str) -> f64 {
    match family {
        "wan" => 120.0,
        "ltx" => 40.0,
        _ => 60.0,
    }
}

/// The four traces for one family: (name, arrivals, hours).
pub fn traces(family: &str, seed: u64) -> Vec<(String, Vec<Arrival>, f64)> {
    let avg = avg_per_hr(family);
    let mut rng = Rng::new(seed);
    let mut out = vec![
        ("steady".to_string(), trace::steady(&mut rng, SIM_EPOCH, 6.0, avg), 6.0),
        (
            "diurnal 2.5x".to_string(),
            trace::diurnal(&mut rng, SIM_EPOCH, 24.0, avg, 2.5, 14.0),
            24.0,
        ),
        ("spike 10x/10min".to_string(), trace::spike(&mut rng, SIM_EPOCH, 4.0, avg, 10.0, 2.0, 10.0), 4.0),
    ];
    let (b, h) = trace::idle_burst(&mut rng, SIM_EPOCH, 2.0, 20, 120.0, 1.0);
    out.push(("idle→burst 20".to_string(), b, h));
    for (_, v, _) in &mut out {
        trace::jitter(&mut rng, v, 0.2);
    }
    out
}

/// The suite behind the docs table.
pub fn standard_suite(seed: u64) -> Vec<Scenario> {
    let price = 4.18; // serverless H100 80GB, $/hr
    let mut v = Vec::new();
    for fam in ["wan", "h3", "ltx"] {
        let profile = FamilyProfile::by_family(fam);
        for (name, arrivals, hours) in traces(fam, seed) {
            let peak_need = peak_workers(&profile, avg_per_hr(fam));
            let mut strategies = vec![Strategy::Autoscaler, Strategy::RunpodOnly];
            if fam == "h3" {
                strategies.push(Strategy::AlwaysOn(peak_need));
                if name.starts_with("idle") {
                    strategies.insert(1, Strategy::AutoscalerWarm(1));
                }
            }
            for strategy in strategies {
                let mut pool = pool_for(&profile, 4);
                if name.starts_with("diurnal") && strategy == Strategy::Autoscaler {
                    // Business-hours floor ahead of the 14:00 peak.
                    pool.schedule = vec![ScheduleRule { days: vec![], start_hour: 10, end_hour: 19, min_workers: 1 }];
                }
                v.push(Scenario {
                    trace: name.clone(),
                    profile: profile.clone(),
                    pool,
                    arrivals: arrivals.clone(),
                    hours,
                    strategy,
                    seed,
                    price_usd_per_hr: price,
                });
            }
        }
    }
    v
}

/// Workers the diurnal busiest hour keeps busy (rounded up).
pub fn peak_workers(p: &FamilyProfile, avg_per_hr: f64) -> u32 {
    ((avg_per_hr * 2.5 * p.job_s / 3600.0).ceil() as u32).max(1)
}

/// Merges runs of the same (trace, family, strategy) over several seeds:
/// waits pooled, totals averaged per run.
pub fn merge(reports: Vec<Report>) -> Vec<Report> {
    let mut out: Vec<Report> = Vec::new();
    for r in reports {
        if let Some(m) = out.iter_mut().find(|m| m.trace == r.trace && m.family == r.family && m.strategy == r.strategy) {
            let n = f64::from(m.runs);
            let avg = |a: f64, b: f64| (a * n + b) / (n + 1.0);
            m.gpu_hours = avg(m.gpu_hours, r.gpu_hours);
            m.cost_usd = avg(m.cost_usd, r.cost_usd);
            m.jobs += r.jobs;
            m.cold_starts += r.cold_starts;
            m.uncached_starts += r.uncached_starts;
            m.max_workers = m.max_workers.max(r.max_workers);
            m.busy_delete_attempts += r.busy_delete_attempts;
            m.unfinished += r.unfinished;
            m.endpoint_patches += r.endpoint_patches;
            m.waits.extend(r.waits);
            m.runs += 1;
        } else {
            out.push(r);
        }
    }
    for m in &mut out {
        m.waits.sort_by(f64::total_cmp);
        m.wait_p50_s = percentile(&m.waits, 50.0);
        m.wait_p95_s = percentile(&m.waits, 95.0);
        m.wait_max_s = m.waits.last().copied().unwrap_or(0.0);
        m.within_slo = if m.waits.is_empty() {
            1.0
        } else {
            m.waits.iter().filter(|w| **w <= m.slo_s).count() as f64 / m.waits.len() as f64
        };
    }
    out
}

pub fn markdown(reports: &[Report]) -> String {
    let mut s = String::from(
        "| trace | family | strategy | jobs/run | wait p50 s | wait p95 s | ≤ SLO | GPU-h/run | $/run | cold starts/run (uncached) | max workers |\n\
         |---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|\n",
    );
    for r in reports {
        let n = f64::from(r.runs.max(1));
        s.push_str(&format!(
            "| {} | {} | {} | {:.0} | {:.0} | {:.0} | {:.0}% (SLO {:.0}s) | {:.2} | {:.2} | {:.1} ({:.1}) | {} |\n",
            r.trace,
            r.family,
            r.strategy,
            r.jobs as f64 / n,
            r.wait_p50_s,
            r.wait_p95_s,
            r.within_slo * 100.0,
            r.slo_s,
            r.gpu_hours,
            r.cost_usd,
            r.cold_starts as f64 / n,
            r.uncached_starts as f64 / n,
            r.max_workers
        ));
    }
    s
}
