//! MiniMax-H3 / FastH3 device graph: Qwen3-VL text encoder (layer-50 hidden
//! states), the single-stream audio+video DiT, the f16t4d24 video VAE and the
//! DAC-style audio VAE. Shares tensor/ops/nn/attention with [`crate::wan`].
//! See docs/ports/h3.md.

pub mod text;
pub mod audio_vae;
pub mod vae;
pub mod vsa;
pub mod transformer;
pub mod pipeline;

#[cfg(test)]
mod manifest_tests;
