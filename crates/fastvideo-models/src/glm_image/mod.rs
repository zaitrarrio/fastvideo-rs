//! GLM-Image host configs. Spec: docs/ports/glm-image.md.

use crate::schedulers::FlowMatchEulerDiscreteScheduler;
use crate::vae::AutoencoderKlConfig;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GlmImagePreset {
    Base,
}

impl GlmImagePreset {
    pub fn as_str(self) -> &'static str { "glm_image" }
    pub fn default_height(self) -> usize { 1024 }
    pub fn default_width(self) -> usize { 1024 }
    pub fn default_steps(self) -> usize { 30 }
    pub fn flow_shift(self) -> f64 { 3.0 }
    pub fn guidance_scale(self) -> f32 { 3.5 }
}

#[derive(Debug, Clone, PartialEq)]
pub struct GlmImageTransformerConfig {
    pub in_channels: usize,
    pub out_channels: usize,
    pub num_layers: usize,
    pub num_attention_heads: usize,
    pub attention_head_dim: usize,
    pub text_embed_dim: usize,
    pub condition_dim: usize,
    pub time_embed_dim: usize,
    pub patch_size: usize,
    pub prior_vq_quantizer_codebook_size: usize,
}

impl GlmImageTransformerConfig {
    pub fn base() -> Self {
        Self {
            in_channels: 16,
            out_channels: 16,
            num_layers: 30,
            num_attention_heads: 32,
            attention_head_dim: 128,
            text_embed_dim: 1472,
            condition_dim: 256,
            time_embed_dim: 512,
            patch_size: 2,
            prior_vq_quantizer_codebook_size: 16384,
        }
    }

    pub fn tiny() -> Self {
        Self {
            in_channels: 4,
            out_channels: 4,
            num_layers: 2,
            num_attention_heads: 4,
            attention_head_dim: 8,
            text_embed_dim: 32,
            condition_dim: 16,
            time_embed_dim: 16,
            patch_size: 2,
            prior_vq_quantizer_codebook_size: 64,
        }
    }

    pub fn inner_dim(&self) -> usize {
        self.num_attention_heads * self.attention_head_dim
    }

    pub fn latent_spatial(&self, height: usize, width: usize) -> (usize, usize) {
        (height / 8, width / 8)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct GlmImageConfig {
    pub dit: GlmImageTransformerConfig,
    pub vae: AutoencoderKlConfig,
    pub flow_shift: f64,
}

impl GlmImageConfig {
    pub fn for_preset(preset: GlmImagePreset) -> Self {
        Self {
            dit: GlmImageTransformerConfig::base(),
            vae: AutoencoderKlConfig::sd3(),
            flow_shift: preset.flow_shift(),
        }
    }

    pub fn tiny() -> Self {
        Self {
            dit: GlmImageTransformerConfig::tiny(),
            vae: AutoencoderKlConfig::tiny(4),
            flow_shift: 3.0,
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
    fn base_dims() {
        let c = GlmImageTransformerConfig::base();
        assert_eq!(c.inner_dim(), 4096);
        assert_eq!(c.num_layers, 30);
        assert_eq!(c.text_embed_dim, 1472);
    }
}
