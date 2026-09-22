//! LTX-2 model family (Lightricks, 19B asymmetric dual-stream audio+video):
//! configuration structs and host-side math the device graph in
//! `fastvideo-cudarc::ltx2` consumes. Reference: diffusers `transformer_ltx2.py`
//! and `pipelines/ltx2`. See docs/ports/ltx2.md.

pub mod config;
pub mod rope;
pub mod schedule;

pub use config::{
    ltx2_19b, ltx2_19b_distilled, ltx2_5_22b_distilled, Gemma3TextConfig, Gemma4TextConfig, Ltx2AudioVaeConfig,
    Ltx2BweConfig, Ltx2ConnectorsConfig, Ltx2DiffusionDecoderConfig, Ltx2LatentUpsamplerConfig, Ltx2ModelVersion,
    Ltx2PipelineDefaults, Ltx2RopeType, Ltx2SchedulerConfig, Ltx2TransformerConfig, Ltx2VaeDecoderStage,
    Ltx2VaeDecoderUpsampler, Ltx2VaeUpsampleKind, Ltx2VideoVaeConfig, Ltx2VocoderConfig, Ltx2Config,
};
pub use rope::{Ltx2RopeTables, ScalarDivision, SplitRope};
pub use schedule::{AncestralOpts, Ltx2Schedule};
