//! Flux2 Candle oracle: DiT, 2D VAE, Mistral3/Qwen3 text postprocess, T2I pipeline.

pub mod config;
pub mod family;
pub mod pipeline;
pub mod text;
pub mod transformer;
pub mod vae;
pub mod weights;

pub use config::{
    Flux2ArchConfig, Flux2VaeConfig, FLUX2_TRANSFORMER_REQUIRED_KEYS, FLUX2_VAE_REQUIRED_KEYS,
    PARAM_NAMES_MAPPING,
};
pub use family::{
    compute_empirical_mu, flux2_time_shift, image_ids, pack_latents_2x2, packed_hw,
    stack_hidden_layers, text_ids, unpatchify_2x2,
};
pub use pipeline::{Flux2Pipeline, GenerateConfig};
pub use text::{
    stack_layers_host, Flux2TextEncoder, Flux2TextKind, Qwen3Config, Qwen3Encoder, FLUX2_SYSTEM_MESSAGE,
};
pub use transformer::Flux2Transformer2D;
pub use vae::AutoencoderKlFlux2;
pub use weights::arch_from_transformer_config;
