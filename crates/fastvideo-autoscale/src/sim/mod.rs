//! A deterministic simulation of GPU worker pools, for tests and the
//! simulation harness ([`harness`]).
//!
//! [`SimWorld`] advances in 1 s steps: arrivals join a FIFO queue, workers
//! boot (worker start + weight load, sometimes an uncached image pull),
//! take one job (or one stream) each, and finish. A serverless pool also
//! emulates Runpod's own scaler (`workersMin` always kept, `QUEUE_DELAY` /
//! `REQUEST_COUNT` scale-up to `workersMax`, idle reaping after the idle
//! timeout; busy workers are never reaped). [`SimProvider`] and
//! [`SimSignals`] put the controller in front of it.

pub mod harness;
pub mod trace;

use std::collections::{BTreeMap, VecDeque};
use std::sync::{Arc, Mutex};

use crate::config::{PoolConfig, PoolKind};
use crate::gateway::SignalSource;
use crate::provider::{ApplyReport, BalanceSource, Provider, ProviderError};
use crate::types::{EndpointSettings, PoolDecision, PoolObservation, PoolSignals, Worker, WorkerState};

pub use trace::{Arrival, Rng};

/// Timings of one model family (seconds), from the WP-18/WP-19 runs
/// (docs/serve/e2e/serverless.md, docs/gaps/2026-09-27-cold-start.md).
#[derive(Clone, Debug, PartialEq, serde::Serialize)]
pub struct FamilyProfile {
    pub family: String,
    /// Submit → process start with the image cached on the host.
    pub worker_start_s: f64,
    /// Process start → models resident.
    pub load_s: f64,
    /// Warm job, mean (traces scale it per job).
    pub job_s: f64,
    /// Extra time of a worker's first job (no `warmup`).
    pub first_job_extra_s: f64,
    /// Submit → process start on a host without the image.
    pub uncached_start_s: f64,
    /// Share of cold starts that land on such a host.
    pub uncached_share: f64,
}

impl FamilyProfile {
    /// wan-turbo: start 49 s, load 70 s, warm job 6.4 s (H100, WP-18 test E).
    pub fn wan() -> Self {
        Self::new("wan", 49.3, 70.4, 6.4, 0.0)
    }
    /// h3-turbo: start 80 s, load 52 s, warm job 24.4 s, first job +45 s.
    pub fn h3() -> Self {
        Self::new("h3", 80.3, 52.3, 24.4, 45.1)
    }
    /// ltx-turbo: start 79 s, load 33 s, 1080p job ~40 s (pod C),
    /// first job +15 s (the warm-up case).
    pub fn ltx() -> Self {
        Self::new("ltx", 79.0, 33.0, 40.0, 15.0)
    }
    pub fn by_family(f: &str) -> Self {
        match f {
            "wan" => Self::wan(),
            "ltx" => Self::ltx(),
            _ => Self::h3(),
        }
    }
    fn new(family: &str, start: f64, load: f64, job: f64, first: f64) -> Self {
        Self {
            family: family.into(),
            worker_start_s: start,
            load_s: load,
            job_s: job,
            first_job_extra_s: first,
            uncached_start_s: 458.0,
            uncached_share: 0.1,
        }
    }
    pub fn cold_start_s(&self) -> f64 {
        self.worker_start_s + self.load_s
    }
}

#[derive(Clone, Debug)]
struct Job {
    arrived: f64,
    /// Warm duration.
    dur: f64,
    stream: bool,
}

#[derive(Clone, Debug)]
struct SimWorker {
    id: String,
    created: f64,
    ready_at: f64,
    ready: bool,
    draining: bool,
    /// (ends at, is stream)
    job: Option<(f64, bool)>,
    /// Start of the current job.
    job_started: f64,
    jobs_done: u32,
    idle_since: f64,
}

/// Counters of one pool.
#[derive(Clone, Debug, Default, serde::Serialize)]
pub struct PoolStats {
    pub jobs: u64,
    pub streams: u64,
    /// Queue wait of every started job (seconds).
    #[serde(skip)]
    pub waits: Vec<f64>,
    pub gpu_seconds: f64,
    pub cost_usd: f64,
    pub cold_starts: u64,
    pub uncached_starts: u64,
    pub max_workers: u32,
    /// Deletes refused because the worker was busy (the policy must never ask).
    pub busy_delete_attempts: u64,
    pub endpoint_patches: u64,
    pub unfinished: u64,
}

struct SimPool {
    cfg: PoolConfig,
    profile: FamilyProfile,
    price: f64,
    /// Serverless endpoint settings (Runpod emulation); `None` for pods.
    endpoint: Option<EndpointSettings>,
    queue: VecDeque<Job>,
    workers: Vec<SimWorker>,
    next_id: u64,
    arrivals_total: u64,
    /// (finished at, duration) of recent jobs.
    recent: VecDeque<(f64, f64)>,
    stats: PoolStats,
}

/// The simulated world. Share it as `Arc<Mutex<SimWorld>>`.
pub struct SimWorld {
    pub now: f64,
    pools: BTreeMap<String, SimPool>,
    pending: BTreeMap<String, VecDeque<Arrival>>,
    rng: Rng,
    pub balance_usd: f64,
}

impl SimWorld {
    pub fn new(start_s: f64, seed: u64, balance_usd: f64) -> Self {
        Self { now: start_s, pools: BTreeMap::new(), pending: BTreeMap::new(), rng: Rng::new(seed), balance_usd }
    }

    /// Adds a pool. Serverless pools start from `endpoint` (min 0 / max
    /// `max_workers` when `None`).
    pub fn add_pool(&mut self, cfg: PoolConfig, profile: FamilyProfile, price: f64, endpoint: Option<EndpointSettings>) {
        let endpoint = match cfg.kind {
            PoolKind::Serverless => Some(endpoint.unwrap_or(EndpointSettings {
                workers_min: 0,
                workers_max: cfg.max_workers,
                idle_timeout_s: cfg.serverless.idle_timeout_s,
                scaler_type: cfg.serverless.scaler_type.clone(),
                scaler_value: cfg.serverless.scaler_value,
            })),
            PoolKind::Pod => None,
        };
        self.pending.insert(cfg.name.clone(), VecDeque::new());
        self.pools.insert(
            cfg.name.clone(),
            SimPool {
                cfg,
                profile,
                price,
                endpoint,
                queue: VecDeque::new(),
                workers: Vec::new(),
                next_id: 0,
                arrivals_total: 0,
                recent: VecDeque::new(),
                stats: PoolStats::default(),
            },
        );
    }

    /// Schedules arrivals (times ≥ now, sorted).
    pub fn load_trace(&mut self, pool: &str, mut arrivals: Vec<Arrival>) {
        arrivals.sort_by(|a, b| a.t.total_cmp(&b.t));
        self.pending.entry(pool.to_owned()).or_default().extend(arrivals);
    }

    pub fn stats(&self, pool: &str) -> PoolStats {
        self.pools.get(pool).map(|p| p.stats.clone()).unwrap_or_default()
    }

    pub fn endpoint(&self, pool: &str) -> Option<EndpointSettings> {
        self.pools.get(pool).and_then(|p| p.endpoint.clone())
    }

    pub fn worker_count(&self, pool: &str) -> usize {
        self.pools.get(pool).map_or(0, |p| p.workers.len())
    }

    /// Jobs waiting for a worker.
    pub fn queued(&self, pool: &str) -> usize {
        self.pools.get(pool).map_or(0, |p| p.queue.len())
    }

    /// Queued + running work left.
    pub fn outstanding(&self, pool: &str) -> usize {
        self.pools.get(pool).map_or(0, |p| p.queue.len() + p.workers.iter().filter(|w| w.job.is_some()).count())
            + self.pending.get(pool).map_or(0, VecDeque::len)
    }

    /// Advances to `until` in 1 s steps.
    pub fn advance_to(&mut self, until: f64) {
        while self.now + 1.0 <= until + 1e-9 {
            self.now += 1.0;
            self.step();
        }
    }

    fn step(&mut self) {
        let now = self.now;
        let names: Vec<String> = self.pools.keys().cloned().collect();
        for name in names {
            // Arrivals.
            let mut arrived = Vec::new();
            if let Some(q) = self.pending.get_mut(&name) {
                while q.front().is_some_and(|a| a.t <= now) {
                    arrived.extend(q.pop_front());
                }
            }
            let mut spawn_n = 0u32;
            let rng = &mut self.rng;
            let p = self.pools.get_mut(&name).expect("pool");
            for a in arrived {
                let dur = if a.stream { a.duration_s } else { p.profile.job_s * a.duration_scale };
                p.arrivals_total += 1;
                p.queue.push_back(Job { arrived: a.t, dur: dur.max(1.0), stream: a.stream });
            }
            // Boot and finish.
            for w in &mut p.workers {
                if !w.ready && w.ready_at <= now {
                    w.ready = true;
                    w.idle_since = now;
                }
                if let Some((end, stream)) = w.job {
                    if end <= now {
                        w.job = None;
                        w.jobs_done += 1;
                        w.idle_since = now;
                        if !stream {
                            p.recent.push_back((end, end - w.job_started));
                        }
                    }
                }
            }
            // Runpod's scaler.
            if let Some(ep) = p.endpoint.clone() {
                let active = p.workers.len() as u32;
                let starting = p.workers.iter().filter(|w| !w.ready).count() as u32;
                let mut want = ep.workers_min.max(active.min(ep.workers_max));
                let waiting = match ep.scaler_type.as_str() {
                    "REQUEST_COUNT" => {
                        let work = p.queue.len() as u32 + p.workers.iter().filter(|w| w.job.is_some()).count() as u32;
                        work.div_ceil(ep.scaler_value.max(1))
                    }
                    _ => {
                        // QUEUE_DELAY: one more worker per job that waited
                        // longer than the delay and no booting or idle
                        // worker will take.
                        let over =
                            p.queue.iter().filter(|j| now - j.arrived >= f64::from(ep.scaler_value)).count() as u32;
                        let idle = p.workers.iter().filter(|w| w.ready && !w.draining && w.job.is_none()).count() as u32;
                        active + over.saturating_sub(starting + idle)
                    }
                };
                want = want.max(waiting.min(ep.workers_max));
                if want > active {
                    spawn_n = want - active;
                }
                // Reap idle workers above min after the idle timeout, and
                // idle workers above max at once.
                let mut n = active;
                let idle_to = f64::from(ep.idle_timeout_s);
                p.workers.retain(|w| {
                    let idle = w.ready && w.job.is_none();
                    let reap = idle && ((n > ep.workers_min && now - w.idle_since >= idle_to) || n > ep.workers_max);
                    if reap {
                        n -= 1;
                    }
                    !reap
                });
            }
            for _ in 0..spawn_n {
                Self::spawn(p, rng, now);
            }
            // Dispatch FIFO.
            for w in p.workers.iter_mut().filter(|w| w.ready && !w.draining && w.job.is_none()) {
                let Some(j) = p.queue.pop_front() else { break };
                let extra = if w.jobs_done == 0 && !j.stream { p.profile.first_job_extra_s } else { 0.0 };
                w.job = Some((now + j.dur + extra, j.stream));
                w.job_started = now;
                p.stats.waits.push(now - j.arrived);
                if j.stream {
                    p.stats.streams += 1;
                } else {
                    p.stats.jobs += 1;
                }
            }
            while p.recent.front().is_some_and(|(t, _)| now - *t > 300.0) {
                p.recent.pop_front();
            }
            // Billing: every worker, booting or not.
            let n = p.workers.len() as f64;
            p.stats.gpu_seconds += n;
            let c = n * p.price / 3600.0;
            p.stats.cost_usd += c;
            self.balance_usd -= c;
            p.stats.max_workers = p.stats.max_workers.max(p.workers.len() as u32);
        }
    }

    fn spawn(p: &mut SimPool, rng: &mut Rng, now: f64) -> String {
        p.next_id += 1;
        let id = format!("{}-{}", p.cfg.name, p.next_id);
        let uncached = rng.next_f64() < p.profile.uncached_share;
        let start = if uncached { p.profile.uncached_start_s } else { p.profile.worker_start_s };
        p.stats.cold_starts += 1;
        if uncached {
            p.stats.uncached_starts += 1;
        }
        p.workers.push(SimWorker {
            id: id.clone(),
            created: now,
            ready_at: now + start + p.profile.load_s,
            ready: false,
            draining: false,
            job: None,
            job_started: now,
            jobs_done: 0,
            idle_since: now,
        });
        id
    }

    /// Finishes the run: counts work never started or finished.
    pub fn finish(&mut self) {
        for p in self.pools.values_mut() {
            p.stats.unfinished = (p.queue.len() + p.workers.iter().filter(|w| w.job.is_some()).count()) as u64;
            // Waits of jobs never started count as the time they waited.
            let now = self.now;
            let waited: Vec<f64> = p.queue.iter().map(|j| now - j.arrived).collect();
            p.stats.waits.extend(waited);
        }
    }

    fn signals(&self, pool: &str) -> PoolSignals {
        let Some(p) = self.pools.get(pool) else { return PoolSignals { pool: pool.into(), ..Default::default() } };
        PoolSignals {
            pool: pool.into(),
            queued: p.queue.len() as u32,
            oldest_queued_s: p.queue.front().map_or(0.0, |j| self.now - j.arrived),
            running_jobs: p.workers.iter().filter(|w| matches!(w.job, Some((_, false)))).count() as u32,
            live_streams: p.workers.iter().filter(|w| matches!(w.job, Some((_, true)))).count() as u32,
            arrivals_total: p.arrivals_total,
            recent_job_s: if p.recent.is_empty() {
                None
            } else {
                Some(p.recent.iter().map(|(_, d)| d).sum::<f64>() / p.recent.len() as f64)
            },
            arrival_rate_per_s: None,
        }
    }

    fn observe(&self, pool: &str) -> PoolObservation {
        let Some(p) = self.pools.get(pool) else { return PoolObservation::default() };
        PoolObservation {
            workers: p
                .workers
                .iter()
                .map(|w| Worker {
                    id: w.id.clone(),
                    state: if w.draining {
                        WorkerState::Draining
                    } else if w.ready {
                        WorkerState::Ready
                    } else {
                        WorkerState::Starting
                    },
                    busy: u32::from(w.job.is_some()),
                    created_at_s: w.created,
                    ready_at_s: w.ready.then_some(w.ready_at),
                    url: None,
                    gpu_type: None,
                    usd_per_hr: Some(p.price),
                })
                .collect(),
            endpoint: p.endpoint.clone(),
        }
    }

    fn apply(&mut self, pool: &str, d: &PoolDecision) -> ApplyReport {
        let now = self.now;
        let rng = &mut self.rng;
        let Some(p) = self.pools.get_mut(pool) else { return ApplyReport::default() };
        let mut r = ApplyReport::default();
        if let (Some(want), Some(cur)) = (d.endpoint.as_ref(), p.endpoint.as_mut()) {
            if want != cur {
                *cur = want.clone();
                p.stats.endpoint_patches += 1;
                r.endpoint_patched = true;
            }
        }
        if p.cfg.kind == PoolKind::Pod {
            for _ in 0..d.create {
                r.created.push(Self::spawn(p, rng, now));
            }
            for id in &d.drain {
                if let Some(w) = p.workers.iter_mut().find(|w| &w.id == id) {
                    w.draining = true;
                    r.drained.push(id.clone());
                }
            }
            for id in &d.undrain {
                if let Some(w) = p.workers.iter_mut().find(|w| &w.id == id) {
                    w.draining = false;
                    r.undrained.push(id.clone());
                }
            }
            for id in &d.delete {
                if let Some(i) = p.workers.iter().position(|w| &w.id == id) {
                    if p.workers[i].job.is_some() {
                        p.stats.busy_delete_attempts += 1;
                        r.notes.push(format!("refused: {id} is busy"));
                    } else {
                        p.workers.remove(i);
                        r.deleted.push(id.clone());
                    }
                }
            }
        }
        r
    }
}

/// Shared handle.
pub type SharedWorld = Arc<Mutex<SimWorld>>;

fn lock(w: &SharedWorld) -> std::sync::MutexGuard<'_, SimWorld> {
    w.lock().unwrap_or_else(|p| p.into_inner())
}

/// The simulator as a [`Provider`] (both pool kinds) and [`BalanceSource`].
pub struct SimProvider(pub SharedWorld);

#[async_trait::async_trait]
impl Provider for SimProvider {
    fn name(&self) -> &'static str {
        "sim"
    }
    async fn observe(&self, pool: &PoolConfig, _now_s: f64) -> Result<PoolObservation, ProviderError> {
        Ok(lock(&self.0).observe(&pool.name))
    }
    async fn apply(&self, pool: &PoolConfig, d: &PoolDecision, _now_s: f64) -> Result<ApplyReport, ProviderError> {
        Ok(lock(&self.0).apply(&pool.name, d))
    }
}

#[async_trait::async_trait]
impl BalanceSource for SimProvider {
    async fn balance_usd(&self) -> Result<f64, ProviderError> {
        Ok(lock(&self.0).balance_usd)
    }
}

/// The simulator as the gateway's [`SignalSource`].
pub struct SimSignals(pub SharedWorld);

#[async_trait::async_trait]
impl SignalSource for SimSignals {
    async fn signals(&self, pools: &[String]) -> Vec<PoolSignals> {
        let w = lock(&self.0);
        pools.iter().map(|p| w.signals(p)).collect()
    }
}
