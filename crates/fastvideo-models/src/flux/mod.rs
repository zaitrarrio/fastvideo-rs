//! FLUX.1 host configs. Spec: docs/ports/flux.md.

pub mod config;
pub mod family;
pub mod text;
pub mod weights;

use crate::schedulers::FlowMatchEulerDiscreteScheduler;
use crate::vae::AutoencoderKlConfig;

pub use config::{
    Flux1ArchConfig, FLUX1_CLIP_REQUIRED_KEYS, FLUX1_T5_REQUIRED_KEYS, FLUX1_TRANSFORMER_REQUIRED_KEYS,
};
pub use family::{
    calculate_shift, calculate_shift_flux1, image_ids, pack_latents_flux1, packed_hw, text_ids,
    unpack_latents_flux1,
};
pub use text::{
    flux1_dummy_text, flux1_t5_len, pad_token_ids, tokenize_flux1, ClipTextConfig, T5Config,
};
pub use weights::{arch_from_transformer_config, local_flux1, looks_like_flux1};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FluxPreset {
    Dev,
    Schnell,
}

impl FluxPreset {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Dev => "flux1_dev",
            Self::Schnell => "flux1_schnell",
        }
    }
    pub fn default_height(self) -> usize {
        1024
    }
    pub fn default_width(self) -> usize {
        1024
    }
    pub fn default_steps(self) -> usize {
        match self {
            Self::Dev => 28,
            Self::Schnell => 4,
        }
    }
    pub fn flow_shift(self) -> f64 {
        1.0
    }
    pub fn guidance_scale(self) -> f32 {
        match self {
            Self::Dev => 3.5,
            Self::Schnell => 0.0,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct FluxTransformerConfig {
    pub in_channels: usize,
    pub out_channels: usize,
    pub num_layers: usize,
    pub num_single_layers: usize,
    pub num_attention_heads: usize,
    pub attention_head_dim: usize,
    pub joint_attention_dim: usize,
    pub pooled_projection_dim: usize,
    pub axes_dims_rope: [usize; 3],
    pub guidance_embeds: bool,
    pub patch_size: usize,
    pub timestep_guidance_channels: usize,
    pub mlp_ratio: f32,
    pub rope_theta: f32,
    pub eps: f32,
}

impl FluxTransformerConfig {
    pub fn flux1_dev() -> Self {
        Self {
            in_channels: 64,
            out_channels: 64,
            num_layers: 19,
            num_single_layers: 38,
            num_attention_heads: 24,
            attention_head_dim: 128,
            joint_attention_dim: 4096,
            pooled_projection_dim: 768,
            axes_dims_rope: [16, 56, 56],
            guidance_embeds: true,
            patch_size: 1,
            timestep_guidance_channels: 256,
            mlp_ratio: 4.0,
            rope_theta: 10_000.0,
            eps: 1e-6,
        }
    }

    pub fn tiny() -> Self {
        Self {
            in_channels: 16,
            out_channels: 16,
            num_layers: 2,
            num_single_layers: 2,
            num_attention_heads: 4,
            attention_head_dim: 8,
            joint_attention_dim: 32,
            pooled_projection_dim: 16,
            axes_dims_rope: [2, 2, 4],
            guidance_embeds: true,
            patch_size: 1,
            timestep_guidance_channels: 16,
            mlp_ratio: 2.0,
            rope_theta: 10_000.0,
            eps: 1e-6,
        }
    }

    pub fn inner_dim(&self) -> usize {
        self.num_attention_heads * self.attention_head_dim
    }

    /// Unpacked VAE latent spatial (16-ch before 2×2 pack into 64).
    pub fn vae_latent_spatial(&self, height: usize, width: usize) -> (usize, usize) {
        (height / 8, width / 8)
    }

    /// Packed DiT spatial (half of VAE spatial).
    pub fn packed_spatial(&self, height: usize, width: usize) -> (usize, usize) {
        let (h, w) = self.vae_latent_spatial(height, width);
        (h / 2, w / 2)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct FluxConfig {
    pub dit: FluxTransformerConfig,
    pub vae: AutoencoderKlConfig,
    pub flow_shift: f64,
}

impl FluxConfig {
    pub fn for_preset(preset: FluxPreset) -> Self {
        Self {
            dit: FluxTransformerConfig::from_preset(preset.as_str()),
            vae: AutoencoderKlConfig::flux(),
            flow_shift: preset.flow_shift(),
        }
    }

    pub fn tiny() -> Self {
        Self {
            dit: FluxTransformerConfig::tiny(),
            vae: AutoencoderKlConfig::tiny(16),
            flow_shift: 1.0,
        }
    }

    pub fn schedule(&self, steps: usize) -> FlowMatchEulerDiscreteScheduler {
        let mut s = FlowMatchEulerDiscreteScheduler::new(1000, self.flow_shift);
        s.set_timesteps(steps);
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flux1_dims() {
        let c = FluxTransformerConfig::flux1_dev();
        assert_eq!(c.inner_dim(), 3072);
        assert_eq!(c.num_layers + c.num_single_layers, 57);
        assert_eq!(c.axes_dims_rope.iter().sum::<usize>(), 128);
    }
}
