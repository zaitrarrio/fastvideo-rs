//! LTX-2 model family (Lightricks, 19B asymmetric dual-stream audio+video):
//! configuration structs and host-side math the device graph in
//! `fastvideo-cudarc::ltx2` consumes. Reference: diffusers `transformer_ltx2.py`
//! and `pipelines/ltx2`. See docs/ports/ltx2.md.

pub mod config;
pub mod fbcache;
pub mod hq;
pub mod lora;
pub mod pisa;
pub mod rope;
pub mod schedule;
pub mod sol;

pub use config::{
    ltx2_19b, ltx2_19b_distilled, ltx2_23_22b, ltx2_23_22b_distilled, ltx2_5_22b_distilled,
    Gemma3TextConfig, Gemma4TextConfig, Ltx2AudioVaeConfig, Ltx2BweConfig, Ltx2Config,
    Ltx2ConnectorsConfig, Ltx2DiffusionDecoderConfig, Ltx2LatentUpsamplerConfig, Ltx2ModelVersion,
    Ltx2PipelineDefaults, Ltx2RopeType, Ltx2SchedulerConfig, Ltx2TransformerConfig,
    Ltx2VaeDecoderStage, Ltx2VaeDecoderUpsampler, Ltx2VaeUpsampleKind, Ltx2VideoVaeConfig,
    Ltx2VocoderConfig,
};
pub use fbcache::{
    requested as fbcache_requested, FbCache, FbDecision, APPLIED as FBCACHE_APPLIED,
    THRESHOLD as FBCACHE_THRESHOLD,
};
pub use hq::{requested as hq_requested, STAGE1_STEPS as HQ_STAGE1_STEPS};
pub use pisa::{
    feat_norm_keep_indices, midpoint_prune_requested, prunes_step, route as pisa_route,
    scatter_prev, stage1_cache_requested, stage1_skips_step, Ltx23PisaRoute,
    BLOCK_SIZE as PISA_BLOCK_SIZE, PRUNE_APPLIED, SPARSITY as PISA_SPARSITY, STAGE1_CACHE_APPLIED,
    STAGE1_CACHE_PRESET,
};
pub use rope::{Ltx2RopeTables, ScalarDivision, SplitRope};
pub use schedule::{AncestralOpts, Ltx2Schedule};
pub use sol::{route, route_for_call, Ltx25SolRoute, LORA_STRENGTH, STAGE2_SIGMAS, STAGE2_TAUS};
