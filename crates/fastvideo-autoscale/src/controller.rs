//! The controller loop: lease → signals → observe → decide → apply.
//!
//! [`Controller::tick`] takes the time as an argument (the simulator drives
//! it with a simulated clock); [`Controller::spawn`] runs it every
//! `interval_s` on the wall clock inside the gateway process.

use std::collections::{BTreeMap, VecDeque};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use serde::Serialize;

use crate::config::AutoscaleConfig;
use crate::gateway::SignalSource;
use crate::lease::Lease;
use crate::policy::{Inputs, Policy, PoolState};
use crate::provider::{ApplyReport, Providers};
use crate::types::{Action, PoolDecision, PoolObservation, WorkerState};

/// One pool's result of one tick.
#[derive(Clone, Debug, Serialize)]
pub struct PoolTick {
    pub decision: PoolDecision,
    pub observed: PoolObservation,
    /// Set when the leader applied it (not in dry run).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub applied: Option<ApplyReport>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// One tick.
#[derive(Clone, Debug, Serialize)]
pub struct TickReport {
    pub at_s: f64,
    pub leader: bool,
    pub dry_run: bool,
    pub balance_usd: Option<f64>,
    pub pools: Vec<PoolTick>,
}

/// `/fv/v1/admin/autoscale`.
#[derive(Clone, Debug, Serialize)]
pub struct Status {
    pub enabled: bool,
    pub dry_run: bool,
    pub holder: String,
    pub leader: bool,
    pub interval_s: f64,
    pub budget_usd_per_hr: f64,
    pub balance_floor_usd: f64,
    pub balance_usd: Option<f64>,
    pub last: Option<TickReport>,
    pub states: BTreeMap<String, PoolState>,
    /// Recent decisions that changed something or were not `hold`, newest last.
    pub history: Vec<PoolDecision>,
}

struct Inner {
    policy: Policy,
    last: Option<TickReport>,
    history: VecDeque<PoolDecision>,
    balance: Option<f64>,
    balance_at: Option<f64>,
    leader: bool,
}

pub struct Controller {
    cfg: AutoscaleConfig,
    providers: Providers,
    signals: Arc<dyn SignalSource>,
    lease: Arc<dyn Lease>,
    holder: String,
    dry_run: AtomicBool,
    inner: Mutex<Inner>,
    /// Serializes ticks (the loop and a manual tick).
    tick_lock: tokio::sync::Mutex<()>,
}

impl Controller {
    pub fn new(
        cfg: AutoscaleConfig,
        providers: Providers,
        signals: Arc<dyn SignalSource>,
        lease: Arc<dyn Lease>,
        holder: impl Into<String>,
    ) -> Self {
        let dry = cfg.dry_run;
        Self {
            cfg,
            providers,
            signals,
            lease,
            holder: holder.into(),
            dry_run: AtomicBool::new(dry),
            inner: Mutex::new(Inner {
                policy: Policy::new(),
                last: None,
                history: VecDeque::new(),
                balance: None,
                balance_at: None,
                leader: false,
            }),
            tick_lock: tokio::sync::Mutex::new(()),
        }
    }

    pub fn config(&self) -> &AutoscaleConfig {
        &self.cfg
    }

    pub fn set_dry_run(&self, on: bool) {
        self.dry_run.store(on, Ordering::SeqCst);
        tracing::info!(dry_run = on, "autoscale: dry run changed");
    }

    pub fn dry_run(&self) -> bool {
        self.dry_run.load(Ordering::SeqCst)
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Inner> {
        self.inner.lock().unwrap_or_else(|p| p.into_inner())
    }

    pub fn status(&self) -> Status {
        let g = self.lock();
        Status {
            enabled: self.cfg.enabled,
            dry_run: self.dry_run(),
            holder: self.holder.clone(),
            leader: g.leader,
            interval_s: self.cfg.interval_s,
            budget_usd_per_hr: self.cfg.budget_usd_per_hr,
            balance_floor_usd: self.cfg.balance_floor_usd,
            balance_usd: g.balance,
            last: g.last.clone(),
            states: g.policy.states.clone(),
            history: g.history.iter().cloned().collect(),
        }
    }

    /// One pass at time `now_s` (unix seconds, or simulated seconds).
    pub async fn tick(&self, now_s: f64) -> TickReport {
        let _serial = self.tick_lock.lock().await;
        let pools = &self.cfg.pools;
        let names: Vec<String> = pools.iter().map(|p| p.name.clone()).collect();

        let leader = match self.lease.acquire(&self.cfg.lease.name, &self.holder, now_s, self.cfg.lease.ttl_s).await {
            Ok(l) => l,
            Err(e) => {
                tracing::warn!(error = %e, "autoscale: lease unavailable; not leading this tick");
                false
            }
        };

        let signals: BTreeMap<String, _> =
            self.signals.signals(&names).await.into_iter().map(|s| (s.pool.clone(), s)).collect();

        let mut observations = BTreeMap::new();
        let mut errors: BTreeMap<String, String> = BTreeMap::new();
        for pool in pools {
            let Some(p) = self.providers.get(pool.kind) else {
                errors.insert(pool.name.clone(), format!("no provider for {:?} pools", pool.kind));
                continue;
            };
            match p.observe(pool, now_s).await {
                Ok(o) => {
                    observations.insert(pool.name.clone(), o);
                }
                Err(e) => {
                    errors.insert(pool.name.clone(), format!("observe: {e}"));
                }
            }
        }

        let balance = self.refresh_balance(now_s).await;
        let decisions = {
            let mut g = self.lock();
            g.leader = leader;
            let inp = Inputs { now_s, signals: signals.clone(), observations: observations.clone(), balance_usd: balance };
            g.policy.decide(&self.cfg, &inp)
        };

        let dry = self.dry_run();
        let mut out = Vec::with_capacity(decisions.len());
        for (pool, d) in pools.iter().zip(decisions) {
            let observed = observations.get(&pool.name).cloned().unwrap_or_default();
            let mut error = errors.get(&pool.name).cloned();
            let mut applied = None;
            // A pool whose observation failed is left alone.
            let can_apply = error.is_none() && leader && !dry && !d.is_noop(&observed);
            if can_apply {
                if let Some(p) = self.providers.get(pool.kind) {
                    match p.apply(pool, &d, now_s).await {
                        Ok(r) => applied = Some(r),
                        Err(e) => {
                            metrics::counter!("fv_autoscale_apply_errors_total", "pool" => pool.name.clone()).increment(1);
                            error = Some(format!("apply: {e}"));
                        }
                    }
                }
            }
            log_decision(&d, &observed, leader, dry, applied.as_ref(), error.as_deref());
            record_metrics(&d, &observed, signals.get(&pool.name));
            out.push(PoolTick { decision: d, observed, applied, error });
        }
        metrics::gauge!("fv_autoscale_leader").set(if leader { 1.0 } else { 0.0 });
        metrics::gauge!("fv_autoscale_dry_run").set(if dry { 1.0 } else { 0.0 });

        let report = TickReport { at_s: now_s, leader, dry_run: dry, balance_usd: balance, pools: out };
        let mut g = self.lock();
        for t in &report.pools {
            if t.decision.action != Action::Hold || !t.decision.is_noop(&t.observed) {
                g.history.push_back(t.decision.clone());
            }
        }
        while g.history.len() > self.cfg.history.max(1) {
            g.history.pop_front();
        }
        g.last = Some(report.clone());
        report
    }

    async fn refresh_balance(&self, now_s: f64) -> Option<f64> {
        let (cached, at) = {
            let g = self.lock();
            (g.balance, g.balance_at)
        };
        if at.is_some_and(|t| now_s - t < self.cfg.balance_interval_s) {
            return cached;
        }
        match self.providers.balance.balance_usd().await {
            Ok(b) => {
                let mut g = self.lock();
                g.balance = Some(b);
                g.balance_at = Some(now_s);
                metrics::gauge!("fv_autoscale_balance_usd").set(b);
                Some(b)
            }
            Err(e) => {
                tracing::warn!(error = %e, "autoscale: balance read failed; keeping the last value");
                cached
            }
        }
    }

    /// Runs `tick` every `interval_s` on the wall clock until `stop` fires;
    /// releases the lease on the way out.
    pub fn spawn(self: Arc<Self>, mut stop: tokio::sync::watch::Receiver<bool>) -> tokio::task::JoinHandle<()> {
        tokio::spawn(async move {
            let mut every = tokio::time::interval(Duration::from_secs_f64(self.cfg.interval_s));
            every.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
            tracing::info!(
                pools = self.cfg.pools.len(),
                dry_run = self.dry_run(),
                holder = %self.holder,
                "autoscale: controller started"
            );
            loop {
                tokio::select! {
                    _ = every.tick() => { self.tick(unix_now()).await; }
                    r = stop.changed() => { if r.is_err() || *stop.borrow() { break; } }
                }
            }
            if let Err(e) = self.lease.release(&self.cfg.lease.name, &self.holder).await {
                tracing::warn!(error = %e, "autoscale: lease release failed");
            }
            tracing::info!("autoscale: controller stopped");
        })
    }
}

pub fn unix_now() -> f64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs_f64()).unwrap_or(0.0)
}

fn log_decision(
    d: &PoolDecision,
    obs: &PoolObservation,
    leader: bool,
    dry: bool,
    applied: Option<&ApplyReport>,
    error: Option<&str>,
) {
    let changed = !d.is_noop(obs);
    let reasons = d.reasons.join("; ");
    let endpoint = d.endpoint.as_ref().map(|e| format!("min={} max={}", e.workers_min, e.workers_max));
    if let Some(e) = error {
        tracing::warn!(pool = %d.pool, action = ?d.action, current = d.current, target = d.target, error = e, "autoscale decision failed");
    } else if changed || d.action != Action::Hold {
        tracing::info!(
            pool = %d.pool,
            action = ?d.action,
            current = d.current,
            target = d.target,
            busy = d.busy,
            demand = d.demand,
            floor = d.floor,
            cap = d.cap,
            endpoint = endpoint.as_deref().unwrap_or("-"),
            create = d.create,
            drain = ?d.drain,
            delete = ?d.delete,
            leader,
            dry_run = dry,
            applied = applied.is_some(),
            reasons = %reasons,
            "autoscale decision"
        );
    } else {
        tracing::debug!(pool = %d.pool, current = d.current, target = d.target, reasons = %reasons, "autoscale hold");
    }
}

fn record_metrics(d: &PoolDecision, obs: &PoolObservation, sig: Option<&crate::types::PoolSignals>) {
    let pool = d.pool.clone();
    metrics::gauge!("fv_autoscale_target_workers", "pool" => pool.clone()).set(f64::from(d.target));
    metrics::gauge!("fv_autoscale_demand_workers", "pool" => pool.clone()).set(d.demand);
    metrics::gauge!("fv_autoscale_busy_workers", "pool" => pool.clone()).set(f64::from(d.busy));
    metrics::gauge!("fv_autoscale_floor_workers", "pool" => pool.clone()).set(f64::from(d.floor));
    metrics::gauge!("fv_autoscale_cap_workers", "pool" => pool.clone()).set(f64::from(d.cap));
    metrics::gauge!("fv_autoscale_target_usd_per_hour", "pool" => pool.clone()).set(d.target_usd_per_hr);
    for (state, name) in
        [(WorkerState::Starting, "starting"), (WorkerState::Ready, "ready"), (WorkerState::Draining, "draining")]
    {
        metrics::gauge!("fv_autoscale_workers", "pool" => pool.clone(), "state" => name).set(f64::from(obs.count(state)));
    }
    if let Some(s) = sig {
        metrics::gauge!("fv_autoscale_queued", "pool" => pool.clone()).set(f64::from(s.queued));
        metrics::gauge!("fv_autoscale_oldest_queued_seconds", "pool" => pool.clone()).set(s.oldest_queued_s);
    }
    let action = match d.action {
        Action::Hold => "hold",
        Action::ScaleUp => "scale_up",
        Action::ScaleDown => "scale_down",
        Action::HardStop => "hard_stop",
    };
    metrics::counter!("fv_autoscale_decisions_total", "pool" => pool, "action" => action).increment(1);
}
