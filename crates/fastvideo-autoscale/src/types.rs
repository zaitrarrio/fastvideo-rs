//! Inputs and outputs of the policy: what the gateway reports per pool,
//! what a provider observes, and what the policy decides.

use serde::{Deserialize, Serialize};

/// Load of one pool as the gateway sees it (the gateway↔autoscaler
/// interface, docs/serve/gateway.md "Autoscaling").
#[derive(Clone, Debug, Default, Deserialize, Serialize, PartialEq)]
pub struct PoolSignals {
    pub pool: String,
    /// Jobs admitted but not yet running on a worker.
    pub queued: u32,
    /// Age of the oldest queued job (seconds; 0 when the queue is empty).
    pub oldest_queued_s: f64,
    /// Batch jobs running on a worker.
    pub running_jobs: u32,
    /// Live streams holding a worker.
    pub live_streams: u32,
    /// Monotonic count of jobs and streams admitted since start (the
    /// arrival rate is its slope).
    pub arrivals_total: u64,
    /// Mean duration of recently finished jobs (seconds), if any finished.
    pub recent_job_s: Option<f64>,
    /// Arrival rate (per second) when the source measures it directly;
    /// otherwise the policy uses the slope of `arrivals_total`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub arrival_rate_per_s: Option<f64>,
}

/// Lifecycle of one worker as the provider reports it.
#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub enum WorkerState {
    /// Created, booting or loading weights; takes no work yet.
    Starting,
    /// Serving (registered with the gateway).
    Ready,
    /// No new work; deleted once `busy` is 0.
    Draining,
}

/// One GPU worker.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct Worker {
    pub id: String,
    pub state: WorkerState,
    /// Jobs + streams on it right now.
    pub busy: u32,
    /// Seconds (same clock as `now`).
    pub created_at_s: f64,
    pub ready_at_s: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gpu_type: Option<String>,
    /// Billed $/hr when the provider reports it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usd_per_hr: Option<f64>,
}

/// Runpod serverless endpoint scaling fields.
#[derive(Clone, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
pub struct EndpointSettings {
    pub workers_min: u32,
    pub workers_max: u32,
    pub idle_timeout_s: u32,
    pub scaler_type: String,
    pub scaler_value: u32,
}

/// What a provider sees for one pool.
#[derive(Clone, Debug, Default, Deserialize, Serialize, PartialEq)]
pub struct PoolObservation {
    pub workers: Vec<Worker>,
    /// Serverless pools: the endpoint's current settings.
    pub endpoint: Option<EndpointSettings>,
}

impl PoolObservation {
    pub fn count(&self, s: WorkerState) -> u32 {
        self.workers.iter().filter(|w| w.state == s).count() as u32
    }
    /// Starting + ready (not draining).
    pub fn active(&self) -> u32 {
        self.workers.iter().filter(|w| w.state != WorkerState::Draining).count() as u32
    }
}

/// The direction of a decision.
#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Action {
    Hold,
    ScaleUp,
    ScaleDown,
    /// Balance below the floor: only busy workers survive.
    HardStop,
}

/// What the controller should do to one pool now.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct PoolDecision {
    pub pool: String,
    pub at_s: f64,
    pub action: Action,
    /// Active (not draining) workers observed.
    pub current: u32,
    /// Active workers wanted after this decision.
    pub target: u32,
    /// Workers holding a job or stream (never removed).
    pub busy: u32,
    /// Workers the load needs (before floors, caps and steps).
    pub demand: f64,
    /// max(min_workers, warm_min, schedule).
    pub floor: u32,
    /// min(max_workers, budget).
    pub cap: u32,
    pub reasons: Vec<String>,
    /// Serverless: the endpoint settings to apply (None = leave as is).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub endpoint: Option<EndpointSettings>,
    /// Pods: new workers to create.
    pub create: u32,
    /// Pods: workers to stop routing to (deleted once idle).
    pub drain: Vec<String>,
    /// Pods: draining workers to take back instead of creating new ones.
    pub undrain: Vec<String>,
    /// Pods: workers to delete now (idle draining, boot timeout).
    pub delete: Vec<String>,
    /// $/hr of `target` workers.
    pub target_usd_per_hr: f64,
    pub estimates: Estimates,
}

/// The policy's running estimates for one pool.
#[derive(Clone, Debug, Default, Deserialize, Serialize, PartialEq)]
pub struct Estimates {
    /// Arrivals per minute over the rate window.
    pub arrivals_per_min: f64,
    /// Queue growth per minute over the rate window.
    pub queue_growth_per_min: f64,
    /// Job duration (EWMA of reported means, else the configured default).
    pub job_s: f64,
    /// Worker start + load (EWMA of measured starts, else the configured value).
    pub cold_start_s: f64,
    /// Queue expected when a worker started now becomes ready.
    pub projected_queue: f64,
}

impl PoolDecision {
    /// True when applying it changes nothing.
    pub fn is_noop(&self, obs: &PoolObservation) -> bool {
        self.create == 0
            && self.drain.is_empty()
            && self.undrain.is_empty()
            && self.delete.is_empty()
            && self.endpoint.as_ref().is_none_or(|e| obs.endpoint.as_ref() == Some(e))
    }
}
