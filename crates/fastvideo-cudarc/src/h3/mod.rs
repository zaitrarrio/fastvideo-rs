//! MiniMax-H3 / FastH3 device graph: Qwen3-VL text encoder (layer-50 hidden
//! states), the single-stream audio+video DiT, the f16t4d24 video VAE and the
//! DAC-style audio VAE. Shares tensor/ops/nn/attention with [`crate::wan`].
//! See docs/ports/h3.md.

pub mod audio_vae;
pub mod fused16;
pub mod lora;
pub mod media;
pub mod mlx;
pub mod pipeline;
pub mod recovered_8b;
pub mod spark;
pub mod text;
pub mod transformer;
pub mod vae;
pub mod vae_encoder;
pub mod vision;
pub mod vsa;

#[cfg(test)]
mod manifest_tests;
#[cfg(test)]
mod reference_tests;
pub mod slim;
pub mod text_cache;
