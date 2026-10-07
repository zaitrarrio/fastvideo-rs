//! The `EngineBackend` trait every executor drives (design §3.6), and the
//! value types that cross it.
//!
//! A backend runs on its executor thread only: never async, never touches
//! tokio. `FakeBackend` (always built) and `CudaBackend` (feature `cuda`,
//! WP-11) implement it.

use std::fmt;
use std::path::PathBuf;

use fastvideo_protocol::{ApiError, JobMetrics, ModelCaps, ModelId, Pcm, ResolvedJob, RgbFrame};
use serde::{Deserialize, Serialize};

use crate::cancel::StepControl;
use crate::caps::Recipe;

/// A streaming session id (clip or causal).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct SessionId(pub uuid::Uuid);

impl SessionId {
    pub fn new() -> Self {
        Self(uuid::Uuid::new_v4())
    }
}

impl Default for SessionId {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for SessionId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.0, f)
    }
}

/// The device an executor owns.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeviceInfo {
    /// CUDA ordinal (fake: its index).
    pub index: u32,
    pub name: String,
    pub total_memory_mb: u64,
}

/// Weight-loading progress reported by `EngineBackend::load`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LoadEvent {
    /// Entering a load stage (`text_encoder`, `dit`, `vae`, …).
    Stage(&'static str),
    /// Bytes (or shards) done out of total.
    Progress { done: u64, total: u64 },
}

/// What one generation produced.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ClipOutput {
    /// Batch: the finished MP4 (`OutputMode::File`).
    pub mp4: Option<PathBuf>,
    /// Streaming builds: the frames in order (`OutputMode::Frames`).
    pub frames: Option<Vec<RgbFrame>>,
    /// Native-rate PCM (streaming builds), when the model has audio.
    pub audio: Option<Pcm>,
    pub metrics: JobMetrics,
}

/// Receives frames and audio as a pipeline produces them (E2 in-memory sinks).
pub trait ClipSink {
    fn frames(&mut self, f: &[RgbFrame]);
    fn audio(&mut self, pcm: &Pcm);
}

/// Drops everything (batch jobs: the backend writes the file itself).
#[derive(Debug, Default)]
pub struct NullSink;

impl ClipSink for NullSink {
    fn frames(&mut self, _: &[RgbFrame]) {}
    fn audio(&mut self, _: &Pcm) {}
}

/// Collects everything in memory.
#[derive(Debug, Default)]
pub struct CollectSink {
    pub frames: Vec<RgbFrame>,
    pub audio: Vec<Pcm>,
}

impl CollectSink {
    /// All collected audio concatenated (`None` if none arrived).
    pub fn joined_audio(&self) -> Option<Pcm> {
        let first = self.audio.first()?;
        let mut v = Vec::new();
        for p in &self.audio {
            v.extend_from_slice(&p.samples);
        }
        Some(Pcm::new(first.rate, first.channels, v))
    }
}

impl ClipSink for CollectSink {
    fn frames(&mut self, f: &[RgbFrame]) {
        self.frames.extend_from_slice(f);
    }
    fn audio(&mut self, pcm: &Pcm) {
        self.audio.push(pcm.clone());
    }
}

/// Opening a causal (SF-Wan) session on the backend.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CausalSpec {
    pub model: ModelId,
    pub width: u32,
    pub height: u32,
    pub fps: u32,
    pub seed: u64,
    pub prompt: String,
}

/// Input for one causal block (one executor turn).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BlockInput {
    /// 0-based block index since open or the last reset.
    pub block_index: u64,
    /// Applies from this block on (prompt changes land at block boundaries).
    pub prompt: String,
    pub prompt_version: u64,
    pub seed: u64,
    /// Clear the KV cache and restart at block 0 before this block.
    pub reset: bool,
}

/// What one causal block took.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct BlockStats {
    pub block_index: u64,
    pub frames: u32,
    /// Wall (or fake-clock) milliseconds for denoise + decode.
    pub block_ms: f64,
    /// Milliseconds of `block_ms` spent on a prompt-switch KV re-cache
    /// before this block (LongLive); 0 when there was none.
    #[serde(default)]
    pub recache_ms: f64,
}

/// One GPU's worth of models (design §3.6). Runs on the executor thread.
pub trait EngineBackend: Send + 'static {
    fn device(&self) -> DeviceInfo;
    /// Every model this backend can serve (`resident` ones load at start).
    fn caps(&self) -> Vec<ModelCaps>;
    /// The recipe behind a model id (tier, steps, routes). **Addition to
    /// design §3.6** for the §0.3 tiers.
    fn recipe(&self, _model: &ModelId) -> Recipe {
        Recipe::default()
    }
    fn load(&mut self, model: &ModelId, obs: &mut dyn FnMut(LoadEvent)) -> Result<(), ApiError>;
    /// Background warm-up (fast boot B): the warm-up runs a resident model
    /// still wants after [`Self::load`] returned, in order. The executor
    /// runs them with [`Self::warmup_run`] only while it has no other work,
    /// one at a time on its own thread (never beside a job), and cancels
    /// the one running when a job arrives; a cancelled run is retried
    /// later. **Addition to design §3.6**; the default has none (a backend
    /// that warms up inside `load` reports ready after it, as before).
    fn warmup_pending(&self, _model: &ModelId) -> Vec<String> {
        Vec::new()
    }
    /// Runs warm-up `name` of `model` (one of [`Self::warmup_pending`]),
    /// observing `cancel` at its steps (a cancelled run returns the
    /// `Cancelled` error). Returns a short description for the log.
    fn warmup_run(
        &mut self,
        model: &ModelId,
        name: &str,
        _cancel: &crate::cancel::CancelToken,
    ) -> Result<String, ApiError> {
        Err(ApiError::internal(format!("model `{model}` has no warm-up `{name}`")))
    }
    /// Frees a model (swap mode, R18). **Addition to design §3.6**; the
    /// default does nothing.
    fn unload(&mut self, _model: &ModelId) {}
    /// Device timing marks for a traced run (docs/serve/tracing.md): `cap`
    /// preallocated marks recorded on the device's work queue (CUDA events
    /// on the compute stream). `None`: host times only.
    fn marks(&mut self, _cap: usize) -> Option<Box<dyn fastvideo_trace::MarkPool>> {
        None
    }
    fn generate(
        &mut self,
        job: &ResolvedJob,
        out: &mut dyn ClipSink,
        ctl: &StepControl,
    ) -> Result<ClipOutput, ApiError>;
    fn causal_open(&mut self, s: SessionId, spec: &CausalSpec) -> Result<(), ApiError>;
    fn causal_block(
        &mut self,
        s: SessionId,
        input: &BlockInput,
        out: &mut dyn ClipSink,
        ctl: &StepControl,
    ) -> Result<BlockStats, ApiError>;
    fn causal_close(&mut self, s: SessionId);
}
