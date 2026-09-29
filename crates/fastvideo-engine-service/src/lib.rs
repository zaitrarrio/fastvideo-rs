//! Engine service: warm model pool, one executor thread per GPU, scheduler,
//! cancellation, progress events and the streaming cores (design §3.6, §5).
//!
//! - [`EngineService`] ([`service`]): the async API every protocol adapter
//!   uses: `start`, `caps`, `readiness`, `submit`, `cancel`, sessions, `drain`.
//! - [`executor`]: one OS thread per [`EngineBackend`]; never async.
//! - [`scheduler`]: `Priority::Stream` before `Priority::Batch`, queue
//!   positions, the exclusive causal lease.
//! - [`pool`]: residency and [`Readiness`].
//! - [`caps`]: the [`CapabilityTable`], recipes, technique profiles and the
//!   max/turbo [`Tier`]s.
//! - [`cancel`]: [`CancelToken`], [`StepControl`] and the [`StepHook`] seam the
//!   CUDA pipelines' step observers plug into (package E1).
//! - [`backend`]: the [`EngineBackend`] trait and its value types.
//! - [`fake`]: [`FakeBackend`], deterministic synthetic A/V (always built).
//! - [`clock`]: [`ManualClock`] for deterministic tests.
//! - [`stream`]: [`ClipSession`] (clip-queue builds) and [`CausalSession`]
//!   (SF-Wan block rollout).
//!
//! - [`cuda`]: the CUDA model catalog (tiers, recipes, caps; always built)
//!   and, with the `cuda` feature, `CudaBackend` over the fastvideo-cudarc
//!   pipelines (WP-11).
//!
//! Owned by WP-02 (docs/serve/design.md §8).

pub mod backend;
pub mod cancel;
pub mod caps;
pub mod clock;
pub(crate) mod executor;
pub mod fake;
pub mod pool;
pub mod scheduler;
pub mod service;
pub mod stream;
pub mod cuda;

pub use backend::{
    BlockInput, BlockStats, CausalSpec, ClipOutput, ClipSink, CollectSink, DeviceInfo,
    EngineBackend, LoadEvent, NullSink, SessionId,
};
pub use cancel::{CancelToken, OutputMode, StepControl, StepEvent, StepHook};
pub use caps::{
    default_profile, ltx_fps_caps, parse_tier_alias, tier_alias, CapabilityTable, ModelEntry, Recipe,
    TierBinding, LTX_FPS, SOL_H3_4STEP_DENSE_PROFILE, SOL_H3_4STEP_PROFILE,
};
pub use fastvideo_protocol::Tier;
pub use clock::{Clock, ManualClock, SystemClock};
pub use fake::{FakeBackend, FakeConfig, FakeFaults, FakeModel, FakeTiming, JobValidator, Mp4Mode};
pub use pool::{ModelPool, Readiness, Residency};
pub use scheduler::Priority;
pub use service::{
    CancelOutcome, EngineConfig, EngineEvent, EngineService, EngineStats, JobHandle,
};
pub use stream::avatar::{
    AvatarConfig, AvatarEvent, AvatarOutputs, AvatarPlan, AvatarPlayer, AvatarStatus, AvatarTake,
    AvatarWindow, PlanInput, WindowKind, WindowReport,
};
pub use stream::causal::{
    CausalBlock, CausalCommand, CausalControl, CausalReply, CausalSession, CausalState,
    CausalStats, Ttff,
};
pub use stream::clip::{ClipBuild, ClipSession};
pub use stream::pace::{
    spawn_causal_pacer, spawn_clip_pacer, CausalPacerConfig, ClipPacerConfig, IdlePolicy,
    MediaItem, MediaSlice, PaceStats, PacedStream, PlayOutcome, Tick, TickReceiver, TickStart,
};
pub use stream::player::{
    BuildReport, ClipCommand, ClipEvent, ClipOutputs, ClipPlayer, ClipPlayerConfig, ClipState,
};
pub use stream::queue::{ClipInfo, QueueName};
