//! FLUX.1 host configs. Spec: docs/ports/flux.md.

use crate::schedulers::FlowMatchEulerDiscreteScheduler;
use crate::vae::AutoencoderKlConfig;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FluxPreset {
    Dev,
}

impl FluxPreset {
    pub fn as_str(self) -> &'static str { "flux1_dev" }
    pub fn default_height(self) -> usize { 1024 }
    pub fn default_width(self) -> usize { 1024 }
    pub fn default_steps(self) -> usize { 28 }
    pub fn flow_shift(self) -> f64 { 1.0 }
    pub fn guidance_scale(self) -> f32 { 3.5 }
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
            axes_dims_rope: [4, 4, 4],
            guidance_embeds: true,
            patch_size: 1,
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
            dit: FluxTransformerConfig::flux1_dev(),
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
