//! Cosmos3-Super host math for `fastvideo-cudarc::cosmos3`.
//! See docs/ports/cosmos3.md.
//!
//! TeaCache and the official canvas live in [`crate::cosmos::sol`]. This
//! module re-exports them; it does not invent a second cache.

pub mod config;
pub mod schedule;

pub use crate::cosmos::sol::{
    fp4_linear, official_requested, teacache_requested, SolCosmosTea, TeaCacheWindow,
    FP4_SKIP_FIRST, FP4_SKIP_LAST, OFFICIAL_FLOW_SHIFT, OFFICIAL_FPS, OFFICIAL_FRAMES,
    OFFICIAL_GUIDANCE, OFFICIAL_HEIGHT, OFFICIAL_STEPS, OFFICIAL_WIDTH, TEACACHE_MAX_CONSECUTIVE,
    TEACACHE_START_STEP, TEACACHE_THRESHOLD,
};
pub use config::{Cosmos3Preset, Cosmos3TransformerConfig};
pub use schedule::Cosmos3Schedule;
