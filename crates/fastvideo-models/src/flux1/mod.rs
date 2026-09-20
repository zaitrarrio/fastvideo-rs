//! FLUX.1 Candle oracle: DiT (3-axis RoPE), CLIP-L + T5-XXL, SD3 VAE, T2I pipeline.

pub mod config;
pub mod family;
pub mod pipeline;
pub mod text;
pub mod transformer;
pub mod weights;

pub use config::{
    Flux1ArchConfig, FLUX1_CLIP_REQUIRED_KEYS, FLUX1_T5_REQUIRED_KEYS, FLUX1_TRANSFORMER_REQUIRED_KEYS,
};
pub use family::{
    calculate_shift, calculate_shift_flux1, image_ids, pack_latents_flux1, packed_hw, text_ids,
    unpack_latents_flux1,
};
pub use pipeline::{Flux1Pipeline, GenerateConfig};
pub use text::{
    flux1_dummy_text, flux1_t5_len, pad_token_ids, tokenize_flux1, ClipTextConfig, ClipTextEncoder,
    Flux1TextEncoder, T5Config, T5Encoder,
};
pub use transformer::Flux1Transformer2D;
pub use weights::arch_from_transformer_config;
