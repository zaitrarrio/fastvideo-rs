//! The scaling policy: pure and deterministic. The caller passes the time,
//! the gateway's signals, the provider's observation and the balance; the
//! policy keeps only its own estimates (arrival rate, job duration, cold
//! start) and timers (cooldowns, idle hysteresis) between calls.
//!
//! Per pool, each call:
//!
//! 1. **Demand** (workers): the queue need, or the steady-state
//!    `λ·D / (utilization·jobs_per_worker)` plus streams, whichever is
//!    larger. The queue need is the workers that start every queued job
//!    within the SLO of capacity being ready: busy workers take
//!    `floor(SLO/D)` jobs each after their current one, every other worker
//!    `floor(SLO/D) + 1`.
//! 2. **Cold-start prediction**: the queue expected when a worker started
//!    now is ready, `Q + max(λ - μ, dQ/dt) · cold_start` (μ = service rate
//!    of the active workers). If waiting it out would miss the SLO, the
//!    demand covers that projected queue now.
//! 3. **Queue-age breach**: the oldest job waited longer than
//!    `slo · slo_breach_fraction` and no idle or starting worker will take
//!    it: at least one more worker, bypassing the cooldown.
//! 4. **Floors and caps**: `max(min_workers, warm_min, schedule)` up to
//!    `min(max_workers, pool budget / price)`.
//! 5. **Steps**: scale-up by at most `scale_up_step` per `scale_up_cooldown_s`;
//!    scale-down only after demand stayed ≤ `(current - 1)(1 - hysteresis)`
//!    for `idle_timeout_s`, `scale_down_cooldown_s` after the last change,
//!    `scale_down_step` at a time, never below the busy workers.
//! 6. **Global budget** across pools (busy workers first, then floors, then
//!    the rest by priority and queue urgency), and the **balance floor**:
//!    below it only busy workers survive (hard stop).
//! 7. **Actions**: serverless → endpoint `workersMin = target`,
//!    `workersMax = cap` (never below busy); pods → create / drain / undrain
//!    / delete, with drain-before-delete, a boot timeout and a max lifetime
//!    (expired workers are replaced first, drained once the replacement is
//!    ready).

use std::collections::{BTreeMap, BTreeSet, VecDeque};

use serde::Serialize;

use crate::config::{AutoscaleConfig, PoolConfig, PoolKind};
use crate::types::{
    Action, EndpointSettings, Estimates, PoolDecision, PoolObservation, PoolSignals, WorkerState,
};

/// Weight of a new sample in the duration EWMAs.
const EWMA_ALPHA: f64 = 0.3;

#[derive(Clone, Copy, Debug)]
struct Sample {
    t: f64,
    arrivals: u64,
    queued: u32,
}

/// Timers and estimates of one pool.
#[derive(Clone, Debug, Default, Serialize)]
pub struct PoolState {
    pub last_up_s: Option<f64>,
    pub last_down_s: Option<f64>,
    pub below_since_s: Option<f64>,
    pub job_s: Option<f64>,
    pub cold_start_s: Option<f64>,
    pub cold_starts: u64,
    #[serde(skip)]
    samples: VecDeque<Sample>,
    #[serde(skip)]
    measured: BTreeSet<String>,
}

impl PoolState {
    fn record(&mut self, now: f64, sig: &PoolSignals, obs: &PoolObservation, window: f64) {
        self.samples.push_back(Sample { t: now, arrivals: sig.arrivals_total, queued: sig.queued });
        while self.samples.len() > 2 && self.samples.front().is_some_and(|s| now - s.t > window) {
            self.samples.pop_front();
        }
        if let Some(d) = sig.recent_job_s.filter(|d| d.is_finite() && *d > 0.0) {
            self.job_s = Some(match self.job_s {
                Some(prev) => prev + EWMA_ALPHA * (d - prev),
                None => d,
            });
        }
        // Cold starts: every worker's first observed ready time.
        let present: BTreeSet<&str> = obs.workers.iter().map(|w| w.id.as_str()).collect();
        self.measured.retain(|id| present.contains(id.as_str()));
        for w in &obs.workers {
            if let Some(r) = w.ready_at_s {
                if self.measured.insert(w.id.clone()) && r >= w.created_at_s {
                    let d = r - w.created_at_s;
                    self.cold_starts += 1;
                    self.cold_start_s = Some(match self.cold_start_s {
                        Some(prev) => prev + EWMA_ALPHA * (d - prev),
                        None => d,
                    });
                }
            }
        }
    }

    /// (arrivals per second, queue growth per second) over the window.
    fn rates(&self) -> (f64, f64) {
        let (Some(a), Some(b)) = (self.samples.front(), self.samples.back()) else { return (0.0, 0.0) };
        let dt = b.t - a.t;
        if dt <= 0.0 {
            return (0.0, 0.0);
        }
        let arrivals = b.arrivals.saturating_sub(a.arrivals) as f64 / dt;
        let growth = (f64::from(b.queued) - f64::from(a.queued)) / dt;
        (arrivals, growth)
    }
}

/// Everything the policy needs for one call.
#[derive(Clone, Debug, Default)]
pub struct Inputs {
    pub now_s: f64,
    pub signals: BTreeMap<String, PoolSignals>,
    pub observations: BTreeMap<String, PoolObservation>,
    /// Account balance in $; `None` when it could not be read yet.
    pub balance_usd: Option<f64>,
}

/// The policy with its per-pool state.
#[derive(Clone, Debug, Default)]
pub struct Policy {
    pub states: BTreeMap<String, PoolState>,
}

/// Per-pool intermediate result before the global budget.
struct Want {
    target: u32,
    busy: u32,
    floor: u32,
    cap: u32,
    current: u32,
    demand: f64,
    urgency: f64,
    reasons: Vec<String>,
    est: Estimates,
}

fn ceil_u32(x: f64) -> u32 {
    if x.is_finite() && x > 0.0 {
        x.ceil().min(f64::from(u32::MAX)) as u32
    } else {
        0
    }
}

/// Workers (busy ones included) needed so that `q` queued jobs of `d`
/// seconds each start within `slo` of a worker being available for them:
/// the `busy_ready` workers start `floor(slo/d)` of them each after their
/// current job, every further worker `floor(slo/d) + 1`. A booting or new
/// worker counts from when it is ready, so a queue that meets a cold start
/// waits about one cold start more than the SLO (instead of a worker per
/// job, which costs a cold start and an idle timeout per job).
fn queue_need(q: f64, slo: f64, d: f64, jobs_per_worker: u32, busy_ready: f64, busy: f64) -> f64 {
    if q <= 0.0 {
        return busy;
    }
    let d = d.max(0.001);
    let c = f64::from(jobs_per_worker);
    let busy_take = busy_ready * (slo / d).floor() * c;
    if q <= busy_take {
        return busy;
    }
    busy + ((q - busy_take) / (((slo / d).floor() + 1.0) * c)).ceil()
}

impl Policy {
    pub fn new() -> Self {
        Self::default()
    }

    /// One decision per configured pool, in config order.
    pub fn decide(&mut self, cfg: &AutoscaleConfig, inp: &Inputs) -> Vec<PoolDecision> {
        let hard_stop = inp.balance_usd.is_some_and(|b| b < cfg.balance_floor_usd);
        let balance_unknown = inp.balance_usd.is_none();
        let mut wants: Vec<Want> = Vec::with_capacity(cfg.pools.len());
        for pool in &cfg.pools {
            let sig = inp.signals.get(&pool.name).cloned().unwrap_or_else(|| PoolSignals {
                pool: pool.name.clone(),
                ..PoolSignals::default()
            });
            let obs = inp.observations.get(&pool.name).cloned().unwrap_or_default();
            let st = self.states.entry(pool.name.clone()).or_default();
            st.record(inp.now_s, &sig, &obs, pool.rate_window_s);
            wants.push(want(cfg, pool, st, inp.now_s, &sig, &obs, hard_stop, balance_unknown));
        }
        if !hard_stop && cfg.budget_usd_per_hr > 0.0 {
            apply_global_budget(cfg, &mut wants);
        }
        cfg.pools
            .iter()
            .zip(wants)
            .map(|(pool, w)| {
                let obs = inp.observations.get(&pool.name).cloned().unwrap_or_default();
                let st = self.states.get_mut(&pool.name).expect("state created above");
                commit(cfg, pool, st, inp.now_s, &obs, w, hard_stop)
            })
            .collect()
    }
}

#[allow(clippy::too_many_arguments)]
fn want(
    cfg: &AutoscaleConfig,
    pool: &PoolConfig,
    st: &mut PoolState,
    now: f64,
    sig: &PoolSignals,
    obs: &PoolObservation,
    hard_stop: bool,
    balance_unknown: bool,
) -> Want {
    let mut reasons = Vec::new();
    let c = f64::from(pool.jobs_per_worker);
    let slo = pool.slo_queue_wait_s;
    let job_s = st.job_s.unwrap_or(pool.default_job_s).max(0.001);
    let cold = st.cold_start_s.unwrap_or(pool.cold_start_s).max(pool.cold_start_s.min(1.0));
    let (slope, growth) = st.rates();
    let lambda = sig.arrival_rate_per_s.filter(|r| r.is_finite() && *r >= 0.0).unwrap_or(slope);
    let q = f64::from(sig.queued);

    // Workers holding work: the provider's busy workers, or the gateway's
    // running jobs and streams, whichever is larger.
    let obs_busy = obs.workers.iter().filter(|w| w.busy > 0).count() as u32;
    let sig_busy = ceil_u32(f64::from(sig.running_jobs) / c)
        + ceil_u32(f64::from(sig.live_streams) / f64::from(pool.streams_per_worker));
    let busy = obs_busy.max(sig_busy);

    let expired = expired_ids(pool, obs, now);
    let active_now = obs
        .workers
        .iter()
        .filter(|w| w.state != WorkerState::Draining && !expired.contains(&w.id) && !boot_timed_out(pool, w, now))
        .count() as u32;
    // Serverless: the control variable is the endpoint's `workersMin`
    // (Runpod adds and reaps workers above it itself).
    let current = match (pool.kind, obs.endpoint.as_ref()) {
        (PoolKind::Serverless, Some(e)) => e.workers_min,
        _ => active_now,
    };
    let serving = |w: &&crate::types::Worker| w.state == WorkerState::Ready && !expired.contains(&w.id);
    let starting = obs
        .workers
        .iter()
        .filter(|w| w.state == WorkerState::Starting && !boot_timed_out(pool, w, now))
        .count() as u32;
    let idle_ready = obs.workers.iter().filter(serving).filter(|w| w.busy == 0).count() as u32;
    let busy_ready = obs.workers.iter().filter(serving).filter(|w| w.busy > 0).count() as u32;
    // 1. Demand.
    let stream_workers = ceil_u32(f64::from(sig.live_streams) / f64::from(pool.streams_per_worker));
    let need_now = queue_need(q, slo, job_s, pool.jobs_per_worker, f64::from(busy_ready.min(busy)), f64::from(busy));
    let steady_raw = lambda * job_s / (pool.target_utilization.clamp(0.05, 1.0) * c) + f64::from(stream_workers);
    let need_steady = steady_raw.ceil();
    let mut demand = need_now.max(need_steady);
    if need_steady > need_now {
        reasons.push(format!("steady: {:.2}/min x {:.0}s", lambda * 60.0, job_s));
    } else if q > 0.0 {
        reasons.push(format!("queue: {} queued, oldest {:.0}s", sig.queued, sig.oldest_queued_s));
    }

    // 2. Cold-start prediction.
    let mu = f64::from(current.saturating_sub(stream_workers)) * c / job_s;
    let rise = (lambda - mu).max(growth).max(0.0);
    let projected = q + rise * cold;
    let projected_wait = if current > 0 { projected * job_s / (f64::from(current) * c) } else { f64::INFINITY };
    if rise > 0.0 && projected > q && projected_wait > slo {
        let pred = queue_need(projected, slo, job_s, pool.jobs_per_worker, f64::from(busy_ready.min(busy)), f64::from(busy));
        if pred > demand {
            demand = pred;
            reasons.push(format!("predictive: queue {projected:.1} in {cold:.0}s cold start"));
        }
    }

    // 3. Queue-age breach with nobody about to take the oldest job.
    let urgent = sig.queued > 0
        && sig.oldest_queued_s > slo * pool.slo_breach_fraction
        && sig.queued > (starting + idle_ready) * pool.jobs_per_worker;
    if urgent {
        demand = demand.max(f64::from(current + 1));
        reasons.push(format!("queue age {:.0}s > {:.0}s", sig.oldest_queued_s, slo * pool.slo_breach_fraction));
    }

    // 4. Floors and caps.
    let floor = pool.min_workers.max(pool.warm_min).max(pool.scheduled_min(now));
    let price = cfg.price(pool);
    let mut cap = pool.max_workers;
    if pool.budget_usd_per_hr > 0.0 && price > 0.0 {
        let by_budget = (pool.budget_usd_per_hr / price).floor() as u32;
        if by_budget < cap {
            cap = by_budget;
            reasons.push(format!("pool budget ${:.2}/h caps {cap}", pool.budget_usd_per_hr));
        }
    }
    let cap = cap.max(busy);
    let desired = ceil_u32(demand).max(floor).min(cap);

    // 5. Steps, cooldowns, hysteresis.
    let mut target = current;
    if hard_stop {
        target = busy;
        reasons.push(format!("balance below ${:.2}: hard stop", cfg.balance_floor_usd));
        st.below_since_s = None;
    } else if desired > current {
        st.below_since_s = None;
        let cooled = st.last_up_s.is_none_or(|t| now - t >= pool.scale_up_cooldown_s);
        let from_zero = current == 0 && (sig.queued > 0 || floor > 0);
        if balance_unknown {
            reasons.push("balance unknown: no scale-up".into());
        } else if cooled || urgent || from_zero || current < floor || ceil_u32(need_now) > current {
            // Jobs already queued (or floors) are not held by the cooldown.
            let stepped = if pool.scale_up_step == 0 { desired } else { desired.min(current + pool.scale_up_step) };
            // Floors (warm, schedule) and jobs already waiting are met at
            // once; the step limits only rate- and prediction-driven growth.
            target = stepped.max(floor.min(cap)).max(ceil_u32(need_now).min(cap));
        } else {
            reasons.push("scale-up cooldown".into());
        }
    } else if desired < current {
        // Hysteresis: the work in hand fits one worker fewer, and the
        // rate-based demand leaves `hysteresis` headroom on it.
        let fewer = f64::from(current - 1);
        let low = need_now <= fewer && steady_raw <= fewer * (1.0 - pool.hysteresis);
        if !low {
            st.below_since_s = None;
        } else {
            let since = *st.below_since_s.get_or_insert(now);
            let last_change = st.last_up_s.into_iter().chain(st.last_down_s).fold(f64::NEG_INFINITY, f64::max);
            if now - since >= pool.idle_timeout_s && now - last_change >= pool.scale_down_cooldown_s {
                target = desired.max(current.saturating_sub(pool.scale_down_step.max(1))).max(busy);
            } else if current > cap {
                // Over a cap (budget or max): no idle wait.
                target = cap.max(busy);
            } else {
                reasons.push(format!("idle {:.0}s of {:.0}s", now - since, pool.idle_timeout_s));
            }
        }
    } else {
        st.below_since_s = None;
    }
    if current > cap && target > cap {
        target = cap.max(busy);
    }

    let urgency = if slo > 0.0 { sig.oldest_queued_s / slo } else { 0.0 };
    Want {
        target,
        busy,
        floor,
        cap,
        current,
        demand,
        urgency,
        reasons,
        est: Estimates {
            arrivals_per_min: lambda * 60.0,
            queue_growth_per_min: growth * 60.0,
            job_s,
            cold_start_s: cold,
            projected_queue: projected,
        },
    }
}

/// Busy workers first, then floors, then the remaining targets, by priority
/// (then queue urgency, then config order) until the global $/hr runs out.
fn apply_global_budget(cfg: &AutoscaleConfig, wants: &mut [Want]) {
    let prices: Vec<f64> = cfg.pools.iter().map(|p| cfg.price(p)).collect();
    let mut order: Vec<usize> = (0..wants.len()).collect();
    order.sort_by(|&a, &b| {
        cfg.pools[b]
            .priority
            .cmp(&cfg.pools[a].priority)
            .then(wants[b].urgency.total_cmp(&wants[a].urgency))
            .then(a.cmp(&b))
    });
    let mut grant: Vec<u32> = wants.iter().map(|w| w.busy.min(w.target)).collect();
    let mut left = cfg.budget_usd_per_hr - grant.iter().zip(&prices).map(|(g, p)| f64::from(*g) * p).sum::<f64>();
    for pass in 0..2 {
        for &i in &order {
            let goal = if pass == 0 { wants[i].target.min(wants[i].floor.max(wants[i].busy)) } else { wants[i].target };
            let need = goal.saturating_sub(grant[i]);
            if need == 0 {
                continue;
            }
            let afford = if prices[i] <= 0.0 { need } else { ((left / prices[i]).floor().max(0.0) as u32).min(need) };
            grant[i] += afford;
            left -= f64::from(afford) * prices[i];
        }
    }
    for (i, w) in wants.iter_mut().enumerate() {
        if grant[i] < w.target {
            w.reasons.push(format!("global budget ${:.2}/h: {} of {}", cfg.budget_usd_per_hr, grant[i], w.target));
            w.target = grant[i];
        }
    }
}

fn expired_ids(pool: &PoolConfig, obs: &PoolObservation, now: f64) -> BTreeSet<String> {
    if pool.kind != PoolKind::Pod || pool.pod.max_lifetime_s <= 0.0 {
        return BTreeSet::new();
    }
    obs.workers
        .iter()
        .filter(|w| w.state == WorkerState::Ready && now - w.created_at_s > pool.pod.max_lifetime_s)
        .map(|w| w.id.clone())
        .collect()
}

fn boot_timed_out(pool: &PoolConfig, w: &crate::types::Worker, now: f64) -> bool {
    pool.kind == PoolKind::Pod
        && pool.pod.boot_timeout_s > 0.0
        && w.state == WorkerState::Starting
        && now - w.created_at_s > pool.pod.boot_timeout_s
}

fn commit(
    cfg: &AutoscaleConfig,
    pool: &PoolConfig,
    st: &mut PoolState,
    now: f64,
    obs: &PoolObservation,
    w: Want,
    hard_stop: bool,
) -> PoolDecision {
    let Want { target, busy, floor, cap, current, demand, reasons, est, .. } = w;
    let mut reasons = reasons;
    let action = if hard_stop {
        Action::HardStop
    } else if target > current {
        st.last_up_s = Some(now);
        Action::ScaleUp
    } else if target < current {
        // The idle period continues: the next step down waits only for
        // `scale_down_cooldown_s`.
        st.last_down_s = Some(now);
        Action::ScaleDown
    } else {
        Action::Hold
    };
    let price = cfg.price(pool);
    let mut d = PoolDecision {
        pool: pool.name.clone(),
        at_s: now,
        action,
        current,
        target,
        busy,
        demand,
        floor,
        cap,
        reasons: Vec::new(),
        endpoint: None,
        create: 0,
        drain: Vec::new(),
        undrain: Vec::new(),
        delete: Vec::new(),
        target_usd_per_hr: f64::from(target) * price,
        estimates: est,
    };
    match pool.kind {
        PoolKind::Serverless => {
            let (min, max) = if hard_stop { (0, busy) } else { (target, cap.max(target).max(busy)) };
            d.endpoint = Some(EndpointSettings {
                workers_min: min,
                workers_max: max,
                idle_timeout_s: pool.serverless.idle_timeout_s,
                scaler_type: pool.serverless.scaler_type.clone(),
                scaler_value: pool.serverless.scaler_value,
            });
        }
        PoolKind::Pod => pod_actions(pool, obs, now, target, hard_stop, &mut d, &mut reasons),
    }
    d.reasons = reasons;
    d
}

fn pod_actions(
    pool: &PoolConfig,
    obs: &PoolObservation,
    now: f64,
    target: u32,
    hard_stop: bool,
    d: &mut PoolDecision,
    reasons: &mut Vec<String>,
) {
    let expired = expired_ids(pool, obs, now);
    let mut workers: Vec<&crate::types::Worker> = obs.workers.iter().collect();
    // Deterministic order: oldest first, then id.
    workers.sort_by(|a, b| a.created_at_s.total_cmp(&b.created_at_s).then(a.id.cmp(&b.id)));

    // Deletes: idle draining workers; starting workers past the boot timeout.
    for w in &workers {
        if w.state == WorkerState::Draining && w.busy == 0 {
            d.delete.push(w.id.clone());
        } else if boot_timed_out(pool, w, now) {
            d.delete.push(w.id.clone());
            reasons.push(format!("{} not ready after {:.0}s", w.id, pool.pod.boot_timeout_s));
        }
    }
    let gone = |id: &str, d: &PoolDecision| d.delete.iter().any(|x| x == id);
    let mut active: Vec<&crate::types::Worker> = workers
        .iter()
        .copied()
        .filter(|w| w.state != WorkerState::Draining && !expired.contains(&w.id) && !gone(&w.id, d))
        .collect();
    let n = active.len() as u32;

    if hard_stop {
        for w in &active {
            if w.busy == 0 {
                if w.state == WorkerState::Starting {
                    d.delete.push(w.id.clone());
                } else {
                    d.drain.push(w.id.clone());
                }
            }
        }
        for id in &expired {
            if !d.drain.contains(id) {
                d.drain.push(id.clone());
            }
        }
        return;
    }

    if target > n {
        let mut need = target - n;
        // Take back draining workers that still hold work (not expired).
        for w in &workers {
            if need == 0 {
                break;
            }
            if w.state == WorkerState::Draining && w.busy > 0 && now - w.created_at_s <= pool.pod.max_lifetime_s {
                d.undrain.push(w.id.clone());
                need -= 1;
            }
        }
        d.create = need;
    } else if target < n {
        let mut remove = n - target;
        // Newest starting workers first (deleted: not routed yet), then the
        // oldest idle ready workers (drained, deleted once observed idle).
        active.sort_by(|a, b| {
            let rank = |w: &crate::types::Worker| match (w.state, w.busy) {
                (WorkerState::Starting, _) => 0,
                (_, 0) => 1,
                _ => 2,
            };
            rank(a).cmp(&rank(b)).then_with(|| {
                if a.state == WorkerState::Starting {
                    b.created_at_s.total_cmp(&a.created_at_s)
                } else {
                    a.created_at_s.total_cmp(&b.created_at_s)
                }
            })
        });
        for w in &active {
            if remove == 0 || w.busy > 0 {
                break;
            }
            if w.state == WorkerState::Starting {
                d.delete.push(w.id.clone());
            } else {
                d.drain.push(w.id.clone());
            }
            remove -= 1;
        }
    }

    // Expired workers: replaced first (they are not counted above), drained
    // once the fresh ready workers cover the target.
    if !expired.is_empty() {
        let ready_fresh = active
            .iter()
            .filter(|w| w.state == WorkerState::Ready && !d.drain.contains(&w.id))
            .count() as u32;
        if ready_fresh >= target {
            for w in workers.iter().filter(|w| expired.contains(&w.id)) {
                d.drain.push(w.id.clone());
                reasons.push(format!("{} past max lifetime {:.0}s", w.id, pool.pod.max_lifetime_s));
            }
        } else {
            reasons.push(format!("{} worker(s) past max lifetime: replacing first", expired.len()));
        }
    }
}

#[cfg(test)]
mod tests;
