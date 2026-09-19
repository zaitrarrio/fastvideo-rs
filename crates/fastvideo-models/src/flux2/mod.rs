//! Flux2 Candle oracle: DiT, 2D VAE, Mistral3/Qwen3 text postprocess, T2I pipeline.

pub mod config;
pub mod family;
pub mod parity;
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
    flux2_dummy_text, flux2_text_len, format_flux2_prompt, format_mistral3_chat, format_qwen3_chat,
    pad_token_ids, stack_layers_host, tokenize_flux2, Flux2TextEncoder, Flux2TextKind, Mistral3Config,
    Mistral3Encoder, Qwen3Config, Qwen3Encoder, FLUX2_SYSTEM_MESSAGE,
};
pub use vae::group_norm;
pub use transformer::Flux2Transformer2D;
pub use vae::AutoencoderKlFlux2;
pub use parity::{klein_1024_mu, mse, psnr};
pub use weights::arch_from_transformer_config;
