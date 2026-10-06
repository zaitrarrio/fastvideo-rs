//! SANA-Video (2B, 480p) host configs and math for `fastvideo-cudarc::sana_video`.
//! Spec and parity plan: docs/ports/sana-video.md.
//!
//! Reference: Diffusers `SanaVideoPipeline` + `SanaVideoTransformer3DModel`
//! on `Efficient-Large-Model/SANA-Video_2B_480p_diffusers` (Apache-2.0):
//! linear-attention DiT, Gemma-2-2B text encoder, the Wan 2.1 VAE (the file is
//! byte-identical to `Wan-AI/Wan2.1-T2V-1.3B-Diffusers/vae`), flow DPM-Solver++.

pub mod config;
pub mod keys;
pub mod rope;
pub mod schedule;
pub mod sol;
pub mod text;

pub use config::{Gemma2TextConfig, SanaVideoTransformerConfig};
pub use schedule::{DpmStepPlan, SanaDpmSolver, SANA_FLOW_SHIFT};
pub use sol::SanaOptimizations;
pub use text::{PromptLayout, SANA_MAX_SEQUENCE_LENGTH};
