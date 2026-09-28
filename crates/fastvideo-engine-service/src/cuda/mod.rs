//! The CUDA backend (WP-11, design §3.6): `CudaBackend` implements
//! `EngineBackend` over the fastvideo-cudarc pipelines, one backend per GPU.
//!
//! - [`caps`] (always built): the model catalog, tiers, recipes, technique
//!   profiles and the `ModelCaps` derived from the model configs, plus the
//!   [`ProcessPlan`] check for which models may share a process.
//! - `h3`, `ltx2`, `wan` (feature `cuda`): one resident pipeline per model
//!   (`H3Pipeline`, `Ltx2Pipeline`, `WanPipeline` incl. TI2V-5B), loaded
//!   once and kept (the warm pool), and `generate` mapping a `ResolvedJob`
//!   onto `H3Request` / `Ltx2Request` / `GenerateConfig`.
//! - `causal` (feature `cuda`, WP-15): `CausalDriver`, the SF-Wan open-ended
//!   block rollout (E6, `wan::stream::CausalRollout`); `CudaBackend`
//!   delegates `causal_open/block/close` to it for its SF-Wan model.
//!
//! **Cancel and progress** (E1): every generate runs with
//! `fastvideo_cudarc::Hooks`. The job's `CancelToken` trips a cudarc
//! `CancelToken` through an on-cancel callback, so the pipeline stops within
//! one denoise step (or decode chunk) with `PipelineError::Cancelled`,
//! mapped to `ApiError` kind `Cancelled`. Stage boundaries and denoise steps
//! are reported through the `StepControl` (stages `text`, `denoise`,
//! `upsample`, `refine`, `audio_decode`, `video_decode`, then `encode`);
//! steps count across the denoise stages (LTX two-stage: 8 + 3 = 11).
//!
//! **Output** (E2 in-memory sinks): every generate attaches a
//! `fastvideo_cudarc::sink::FrameSink`, so the decoded RGB8 frames and the
//! PCM arrive in memory (the same bytes the CLI writes as `frame-NNN.png`):
//! - `OutputMode::File{dir}`: frames stream into `<dir>/output.mp4`, encoded
//!   on NVENC (design §0.1; `fastvideo-media::mp4`), AAC audio at the
//!   model's native rate unless the plan drops it, crop applied for
//!   pad-and-crop canvases. With `CudaBackendConfig::keep_frames` the
//!   pipelines also write their PNGs to `<dir>/frames/` (identity checks
//!   against the CLI).
//! - `OutputMode::Frames`: RGB frames and the PCM go to the `ClipSink` and
//!   into `ClipOutput::{frames, audio}`; nothing touches the disk but the
//!   pipeline's `audio.wav` in a scratch directory that is removed.
//!
//! **Technique profiles.** `CudaBackend::new` installs the process's
//! profile (the models' shared load-time settings, [`ProcessPlan`]) before
//! any pipeline reads a `FASTVIDEO_*` setting, and sets the process env the
//! recipes need (`FASTVIDEO_VSA` for FastWan). Per model, an H3 pipeline
//! whose profile is not the installed one switches to it with `set_arm`
//! after load; LTX resolves its stage-2 route from its own profile per
//! request.

pub mod caps;

#[cfg(feature = "cuda")]
mod backend;
#[cfg(feature = "cuda")]
pub mod causal;
#[cfg(feature = "cuda")]
pub mod h3;
#[cfg(feature = "cuda")]
pub mod ltx2;
#[cfg(feature = "cuda")]
mod output;
#[cfg(feature = "cuda")]
pub mod wan;

#[cfg(feature = "cuda")]
pub use backend::{install_process_plan, CudaBackend, CudaBackendConfig, Mp4Encoder};
pub use caps::{
    catalog, find, model_from_config, CudaModel, ModelEntryCfg, CudaRecipe, H3Recipe, Ltx2Recipe, LtxStage2, LtxVersion, ProcessPlan,
    SfWanRecipe, WanDecoder, WanRecipe, WanSampler, WeightLayout,
};
