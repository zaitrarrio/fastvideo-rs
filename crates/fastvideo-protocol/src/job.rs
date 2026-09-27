//! `Job`, `JobState`, `Artifact`, `JobStore` trait (design §3.4).
//!
//! The job model is protocol-neutral: every adapter's status view renders a
//! [`Job`]. State changes go through the checked transition methods on
//! [`Job`] so every store and view sees the same state machine:
//!
//! ```text
//! Queued ──► Running ──► Succeeded
//!   │           ├──────► Failed(ApiError)
//!   │           └──────► Cancelled
//!   ├──────────────────► Failed(ApiError)   (submit error, restart recovery)
//!   └──────────────────► Cancelled
//! ```
//!
//! Terminal states (`Succeeded`, `Failed`, `Cancelled`) never change.

use std::collections::BTreeMap;
use std::fmt;
use std::path::PathBuf;
use std::str::FromStr;
use std::time::Duration;

use serde::{Deserialize, Serialize};
use time::OffsetDateTime;
use uuid::Uuid;

use crate::error::{ApiError, ErrorKind};
use crate::negotiate::ResolvedJob;
use crate::request::{ProtocolId, Task};

/// Internal job id (all protocols); the wire id is `Job::external_id`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct JobId(pub Uuid);

impl JobId {
    /// A fresh random (v4) id.
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }
}

impl Default for JobId {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for JobId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.0, f)
    }
}

impl FromStr for JobId {
    type Err = uuid::Error;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Uuid::parse_str(s).map(Self)
    }
}

/// Id of the API key that created a job (for owner-scoped listing).
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct KeyId(pub String);

impl fmt::Display for KeyId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// Id of an output artifact; the first path segment of
/// `GET /files/{artifact_id}/{file_name}`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ArtifactId(pub Uuid);

impl ArtifactId {
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }
}

impl Default for ArtifactId {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for ArtifactId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.0, f)
    }
}

impl FromStr for ArtifactId {
    type Err = uuid::Error;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Uuid::parse_str(s).map(Self)
    }
}

/// Lifecycle state of a job.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "status", content = "error", rename_all = "snake_case")]
pub enum JobState {
    Queued,
    Running,
    Succeeded,
    Failed(ApiError),
    Cancelled,
}

impl JobState {
    pub fn status(&self) -> JobStatus {
        match self {
            JobState::Queued => JobStatus::Queued,
            JobState::Running => JobStatus::Running,
            JobState::Succeeded => JobStatus::Succeeded,
            JobState::Failed(_) => JobStatus::Failed,
            JobState::Cancelled => JobStatus::Cancelled,
        }
    }
    pub fn is_terminal(&self) -> bool {
        self.status().is_terminal()
    }
    /// The failure, when `Failed`.
    pub fn error(&self) -> Option<&ApiError> {
        match self {
            JobState::Failed(e) => Some(e),
            _ => None,
        }
    }
    /// Whether the state machine allows `self -> next`.
    pub fn can_transition_to(&self, next: &JobState) -> bool {
        self.status().can_transition_to(next.status())
    }
}

/// [`JobState`] without the payload: filters, metrics, transition checks.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum JobStatus {
    Queued,
    Running,
    Succeeded,
    Failed,
    Cancelled,
}

impl JobStatus {
    pub fn is_terminal(&self) -> bool {
        matches!(
            self,
            JobStatus::Succeeded | JobStatus::Failed | JobStatus::Cancelled
        )
    }
    /// The state machine in the module docs.
    pub fn can_transition_to(&self, next: JobStatus) -> bool {
        use JobStatus::*;
        matches!(
            (self, next),
            (Queued, Running)
                | (Queued, Failed)
                | (Queued, Cancelled)
                | (Running, Succeeded)
                | (Running, Failed)
                | (Running, Cancelled)
        )
    }
    pub fn as_str(&self) -> &'static str {
        match self {
            JobStatus::Queued => "queued",
            JobStatus::Running => "running",
            JobStatus::Succeeded => "succeeded",
            JobStatus::Failed => "failed",
            JobStatus::Cancelled => "cancelled",
        }
    }
}

/// A refused state change.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
#[error("job {job}: illegal transition {from:?} -> {to:?}")]
pub struct TransitionError {
    pub job: JobId,
    pub from: JobStatus,
    pub to: JobStatus,
}

impl From<TransitionError> for ApiError {
    /// Leaving a terminal state is `AlreadyCompleted`; anything else is `Conflict`.
    fn from(e: TransitionError) -> Self {
        let kind = if e.from.is_terminal() {
            ErrorKind::AlreadyCompleted
        } else {
            ErrorKind::Conflict
        };
        ApiError::new(kind, e.to_string())
    }
}

/// One log line (fal `logs`).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LogLine {
    pub message: String,
    pub level: LogLevel,
    #[serde(with = "time::serde::rfc3339")]
    pub timestamp: OffsetDateTime,
}

impl LogLine {
    pub fn info(message: impl Into<String>, timestamp: OffsetDateTime) -> Self {
        Self {
            message: message.into(),
            level: LogLevel::Info,
            timestamp,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LogLevel {
    Debug,
    Info,
    Warn,
    Error,
}

/// Measurements of one generation (FastVideo `X-*` headers, fal `timings`).
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct JobMetrics {
    /// Denoise wall time, seconds (fal `timings.inference`).
    pub inference_s: Option<f64>,
    /// Stage name -> seconds (FastVideo `X-Stage-Durations`).
    #[serde(default)]
    pub stage_durations: BTreeMap<String, f64>,
    pub peak_memory_mb: Option<f64>,
    /// Build time / clip time.
    pub build_rtf: Option<f64>,
}

/// Where a status callback goes.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum CallbackSpec {
    /// fal `?fal_webhook=`: Ed25519-signed POST on completion.
    FalWebhook { url: url::Url },
    /// MiniMax `callback_url`: challenge, then POST `{task}` on every change.
    MiniMax { url: url::Url },
}

impl CallbackSpec {
    pub fn url(&self) -> &url::Url {
        match self {
            CallbackSpec::FalWebhook { url } | CallbackSpec::MiniMax { url } => url,
        }
    }
}

/// Where an artifact's bytes live.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ArtifactLocation {
    Local(PathBuf),
    Object { bucket: String, key: String },
}

/// One output file.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Artifact {
    pub id: ArtifactId,
    pub mime: String,
    pub file_name: String,
    pub bytes: u64,
    pub location: ArtifactLocation,
    pub width: u32,
    pub height: u32,
    pub frames: u32,
    pub fps: u32,
    /// `(rate, channels)` of the audio stream, if any.
    pub audio: Option<(u32, u8)>,
}

impl Artifact {
    /// Duration in seconds (`frames / fps`).
    pub fn duration_s(&self) -> f64 {
        if self.fps == 0 {
            0.0
        } else {
            self.frames as f64 / self.fps as f64
        }
    }
}

/// A generation job.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Job {
    pub id: JobId,
    pub protocol: ProtocolId,
    /// fal uuid | MiniMax 18-digit numeric | LTX uuid | `video_gen_<32hex>` | FastWan uuid.
    pub external_id: String,
    pub owner: Option<KeyId>,
    /// Original fields some views echo (prompt, size, seconds, ratio, model).
    pub request_echo: serde_json::Value,
    pub resolved: ResolvedJob,
    pub state: JobState,
    /// `0..=1` from engine step events.
    pub progress: f32,
    pub queue_position: Option<u32>,
    #[serde(with = "time::serde::rfc3339")]
    pub created_at: OffsetDateTime,
    #[serde(with = "time::serde::rfc3339::option")]
    pub started_at: Option<OffsetDateTime>,
    #[serde(with = "time::serde::rfc3339::option")]
    pub completed_at: Option<OffsetDateTime>,
    #[serde(with = "time::serde::rfc3339")]
    pub expires_at: OffsetDateTime,
    pub logs: Vec<LogLine>,
    pub metrics: JobMetrics,
    pub artifacts: Vec<Artifact>,
    pub callback: Option<CallbackSpec>,
    pub cancel_requested: bool,
}

impl Job {
    /// A `Queued` job created `now`, expiring after `retention`.
    pub fn new(
        id: JobId,
        protocol: ProtocolId,
        external_id: impl Into<String>,
        resolved: ResolvedJob,
        now: OffsetDateTime,
        retention: Duration,
    ) -> Self {
        Self {
            id,
            protocol,
            external_id: external_id.into(),
            owner: None,
            request_echo: serde_json::Value::Null,
            resolved,
            state: JobState::Queued,
            progress: 0.0,
            queue_position: None,
            created_at: now,
            started_at: None,
            completed_at: None,
            expires_at: now + retention,
            logs: Vec::new(),
            metrics: JobMetrics::default(),
            artifacts: Vec::new(),
            callback: None,
            cancel_requested: false,
        }
    }

    pub fn status(&self) -> JobStatus {
        self.state.status()
    }
    pub fn is_terminal(&self) -> bool {
        self.state.is_terminal()
    }
    /// The model name the client sent (`request_echo.model`), else the resolved id.
    pub fn requested_model(&self) -> &str {
        self.request_echo
            .get("model")
            .and_then(|m| m.as_str())
            .unwrap_or(&self.resolved.model.0)
    }
    pub fn task(&self) -> Task {
        self.resolved.task
    }

    fn transition(&mut self, next: JobState) -> Result<(), TransitionError> {
        if !self.state.can_transition_to(&next) {
            return Err(TransitionError {
                job: self.id,
                from: self.status(),
                to: next.status(),
            });
        }
        self.state = next;
        Ok(())
    }

    /// `Queued -> Running`; clears the queue position.
    pub fn mark_running(&mut self, now: OffsetDateTime) -> Result<(), TransitionError> {
        self.transition(JobState::Running)?;
        self.started_at = Some(now);
        self.queue_position = None;
        Ok(())
    }

    /// `Running -> Succeeded` with the outputs; progress becomes 1.
    pub fn mark_succeeded(
        &mut self,
        now: OffsetDateTime,
        artifacts: Vec<Artifact>,
        metrics: JobMetrics,
    ) -> Result<(), TransitionError> {
        self.transition(JobState::Succeeded)?;
        self.finish(now);
        self.progress = 1.0;
        self.artifacts = artifacts;
        self.metrics = metrics;
        Ok(())
    }

    /// `Queued | Running -> Failed(err)`.
    pub fn mark_failed(
        &mut self,
        now: OffsetDateTime,
        err: ApiError,
    ) -> Result<(), TransitionError> {
        self.transition(JobState::Failed(err))?;
        self.finish(now);
        Ok(())
    }

    /// `Queued | Running -> Cancelled`.
    pub fn mark_cancelled(&mut self, now: OffsetDateTime) -> Result<(), TransitionError> {
        self.transition(JobState::Cancelled)?;
        self.cancel_requested = true;
        self.finish(now);
        Ok(())
    }

    fn finish(&mut self, now: OffsetDateTime) {
        self.completed_at = Some(now);
        self.queue_position = None;
    }

    /// Records a progress report, clamped to `0..=1` and never moving
    /// backwards. Ignored once terminal.
    pub fn set_progress(&mut self, p: f32) {
        if self.is_terminal() || !p.is_finite() {
            return;
        }
        self.progress = self.progress.max(p.clamp(0.0, 1.0));
    }

    /// `step / total` as progress.
    pub fn set_step(&mut self, step: u32, total: u32) {
        if total > 0 {
            self.set_progress(step as f32 / total as f32);
        }
    }

    /// Restart recovery (design §3.4): unfinished jobs become
    /// `Failed(Internal, "interrupted by restart")`. Returns whether it changed.
    pub fn recover_after_restart(&mut self, now: OffsetDateTime) -> bool {
        if self.is_terminal() {
            return false;
        }
        self.mark_failed(now, ApiError::internal("interrupted by restart"))
            .is_ok()
    }

    pub fn is_expired(&self, now: OffsetDateTime) -> bool {
        now >= self.expires_at
    }

    /// The snapshot `JobStore::watch` publishes; `seq` is the store's change counter.
    pub fn snapshot(&self, seq: u64) -> JobSnapshot {
        JobSnapshot {
            id: self.id,
            seq,
            state: self.state.clone(),
            progress: self.progress,
            queue_position: self.queue_position,
            log_count: self.logs.len(),
        }
    }
}

/// A compact view of a job published on every change (`JobStore::watch`).
/// Watchers that need more (SSE status bodies) re-read the job from the store.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct JobSnapshot {
    pub id: JobId,
    /// Increases on every store update of this job.
    pub seq: u64,
    pub state: JobState,
    pub progress: f32,
    pub queue_position: Option<u32>,
    pub log_count: usize,
}

/// Job-store failures.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum StoreError {
    #[error("job {0} already exists")]
    AlreadyExists(JobId),
    #[error("external id {1} already exists for {0}")]
    DuplicateExternal(ProtocolId, String),
    #[error("job {0} not found")]
    NotFound(JobId),
    #[error("job store is full")]
    Full,
    #[error("job store I/O: {0}")]
    Io(String),
}

impl From<StoreError> for ApiError {
    fn from(e: StoreError) -> Self {
        match e {
            StoreError::NotFound(_) => ApiError::not_found(e.to_string()),
            StoreError::Full => ApiError::queue_full(e.to_string()),
            StoreError::AlreadyExists(_) | StoreError::DuplicateExternal(..) => {
                ApiError::conflict(e.to_string())
            }
            StoreError::Io(_) => ApiError::internal(e.to_string()),
        }
    }
}

/// Sort order for listings.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SortOrder {
    /// Newest first (FastVideo default).
    #[default]
    Desc,
    Asc,
}

/// Listing filters and paging. Cursor paging (FastVideo `after`) and offset
/// paging (MiniMax `page_num`/`page_size`) are both supported.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ListQuery {
    pub owner: Option<KeyId>,
    pub protocol: Option<ProtocolId>,
    /// Empty = any status.
    pub statuses: Vec<JobStatus>,
    /// Matches the requested model name or the resolved model id.
    pub model: Option<String>,
    pub task: Option<Task>,
    /// Empty = any; otherwise only these external ids.
    pub external_ids: Vec<String>,
    pub order: SortOrder,
    /// Return items strictly after this external id (in `order`).
    pub after: Option<String>,
    /// Items to skip after filtering and `after`.
    pub offset: usize,
    pub limit: usize,
}

impl Default for ListQuery {
    fn default() -> Self {
        Self {
            owner: None,
            protocol: None,
            statuses: Vec::new(),
            model: None,
            task: None,
            external_ids: Vec::new(),
            order: SortOrder::Desc,
            after: None,
            offset: 0,
            limit: 20,
        }
    }
}

impl ListQuery {
    /// Whether `job` passes the filters (not the paging).
    pub fn matches(&self, job: &Job) -> bool {
        self.owner
            .as_ref()
            .is_none_or(|o| job.owner.as_ref() == Some(o))
            && self.protocol.is_none_or(|p| job.protocol == p)
            && (self.statuses.is_empty() || self.statuses.contains(&job.status()))
            && self
                .model
                .as_deref()
                .is_none_or(|m| job.requested_model() == m || job.resolved.model.0 == m)
            && self.task.is_none_or(|t| job.resolved.task == t)
            && (self.external_ids.is_empty() || self.external_ids.contains(&job.external_id))
    }

    /// Applies filters, ordering (by `created_at`, ties by id) and paging to
    /// any set of jobs. `MemJobStore::list` can be exactly this.
    pub fn apply<'a>(&self, jobs: impl IntoIterator<Item = &'a Job>) -> Page<Job> {
        let mut v: Vec<&Job> = jobs.into_iter().filter(|j| self.matches(j)).collect();
        v.sort_by(|a, b| a.created_at.cmp(&b.created_at).then(a.id.cmp(&b.id)));
        if self.order == SortOrder::Desc {
            v.reverse();
        }
        let total = v.len();
        let start = match &self.after {
            Some(a) => v
                .iter()
                .position(|j| &j.external_id == a)
                .map_or(v.len(), |i| i + 1),
            None => 0,
        };
        let rest = &v[start.min(v.len())..];
        let rest = &rest[self.offset.min(rest.len())..];
        let n = self.limit.min(rest.len());
        let items: Vec<Job> = rest[..n].iter().map(|j| (*j).clone()).collect();
        Page {
            has_more: rest.len() > n,
            total,
            items,
        }
    }
}

/// One page of a listing.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Page<T> {
    pub items: Vec<T>,
    /// Matching items before paging.
    pub total: usize,
    /// More items follow this page.
    pub has_more: bool,
}

impl<T> Default for Page<T> {
    fn default() -> Self {
        Self {
            items: Vec::new(),
            total: 0,
            has_more: false,
        }
    }
}

impl<T> Page<T> {
    pub fn first(&self) -> Option<&T> {
        self.items.first()
    }
    pub fn last(&self) -> Option<&T> {
        self.items.last()
    }
}

/// A job mutation for [`JobStore::update`].
pub type JobUpdate = Box<dyn FnOnce(&mut Job) + Send>;

/// Job persistence. `MemJobStore` (serve-kit) keeps an in-memory map plus one
/// JSON manifest per job.
#[async_trait::async_trait]
pub trait JobStore: Send + Sync + 'static {
    async fn insert(&self, job: Job) -> Result<(), StoreError>;
    async fn get(&self, id: JobId) -> Option<Job>;
    async fn by_external(&self, p: ProtocolId, external_id: &str) -> Option<Job>;
    /// Applies `f` atomically and returns the updated job; watchers see a new snapshot.
    async fn update(
        &self,
        id: JobId,
        f: Box<dyn FnOnce(&mut Job) + Send>,
    ) -> Result<Job, StoreError>;
    /// Owner/protocol/status/model filters, cursor.
    async fn list(&self, q: ListQuery) -> Page<Job>;
    async fn remove(&self, id: JobId) -> Option<Job>;
    fn watch(&self, id: JobId) -> Option<tokio::sync::watch::Receiver<JobSnapshot>>;
    /// Removes jobs (and their artifacts) whose `expires_at <= now`; returns the count.
    async fn sweep_expired(&self, now: OffsetDateTime) -> usize;
}
