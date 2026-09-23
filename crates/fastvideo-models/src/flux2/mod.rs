//! FLUX.2 host configs. Spec: docs/ports/flux2.md.

use crate::schedulers::FlowMatchEulerDiscreteScheduler;
use crate::vae::AutoencoderKlConfig;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Flux2Preset {
    Klein4b,
    Klein9b,
    Dev,
}

impl Flux2Preset {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Klein4b => "flux2_klein_4b",
            Self::Klein9b => "flux2_klein_9b",
            Self::Dev => "flux2_dev",
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
            Self::Klein4b | Self::Klein9b => 4,
            Self::Dev => 50,
        }
    }

    pub fn flow_shift(self) -> f64 {
        1.0
    }

    pub fn guidance_scale(self) -> f32 {
        match self {
            Self::Klein4b | Self::Klein9b => 1.0,
            Self::Dev => 4.0,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Flux2TransformerConfig {
    pub in_channels: usize,
    pub out_channels: usize,
    pub num_layers: usize,
    pub num_single_layers: usize,
    pub num_attention_heads: usize,
    pub attention_head_dim: usize,
    pub joint_attention_dim: usize,
    pub timestep_guidance_channels: usize,
    pub mlp_ratio: f32,
    pub axes_dims_rope: [usize; 4],
    pub rope_theta: f32,
    pub guidance_embeds: bool,
    pub patch_size: usize,
}

impl Flux2TransformerConfig {
    pub fn klein_4b() -> Self {
        Self {
            in_channels: 128,
            out_channels: 128,
            num_layers: 5,
            num_single_layers: 20,
            num_attention_heads: 24,
            attention_head_dim: 128,
            joint_attention_dim: 7680,
            timestep_guidance_channels: 256,
            mlp_ratio: 3.0,
            axes_dims_rope: [32, 32, 32, 32],
            rope_theta: 2000.0,
            guidance_embeds: false,
            patch_size: 1,
        }
    }

    pub fn klein_9b() -> Self {
        Self {
            in_channels: 128,
            out_channels: 128,
            num_layers: 8,
            num_single_layers: 24,
            num_attention_heads: 32,
            attention_head_dim: 128,
            joint_attention_dim: 10240,
            timestep_guidance_channels: 256,
            mlp_ratio: 3.0,
            axes_dims_rope: [32, 32, 32, 32],
            rope_theta: 2000.0,
            guidance_embeds: false,
            patch_size: 1,
        }
    }

    pub fn flux2_dev() -> Self {
        Self {
            in_channels: 128,
            out_channels: 128,
            num_layers: 8,
            num_single_layers: 48,
            num_attention_heads: 48,
            attention_head_dim: 128,
            joint_attention_dim: 15360,
            timestep_guidance_channels: 256,
            mlp_ratio: 3.0,
            axes_dims_rope: [32, 32, 32, 32],
            rope_theta: 2000.0,
            guidance_embeds: true,
            patch_size: 1,
        }
    }

    pub fn for_preset(preset: Flux2Preset) -> Self {
        match preset {
            Flux2Preset::Klein4b => Self::klein_4b(),
            Flux2Preset::Klein9b => Self::klein_9b(),
            Flux2Preset::Dev => Self::flux2_dev(),
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
            timestep_guidance_channels: 16,
            mlp_ratio: 3.0,
            axes_dims_rope: [4, 4, 4, 4],
            rope_theta: 2000.0,
            guidance_embeds: false,
            patch_size: 1,
        }
    }

    pub fn inner_dim(&self) -> usize {
        self.num_attention_heads * self.attention_head_dim
    }

    pub fn packed_spatial(&self, height: usize, width: usize) -> (usize, usize) {
        // VAE 8× then 2×2 pack from 16→128 channels → /2 spatial.
        ((height / 8) / 2, (width / 8) / 2)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Flux2Config {
    pub dit: Flux2TransformerConfig,
    pub vae: AutoencoderKlConfig,
    pub flow_shift: f64,
}

impl Flux2Config {
    pub fn for_preset(preset: Flux2Preset) -> Self {
        Self {
            dit: Flux2TransformerConfig::for_preset(preset),
            vae: AutoencoderKlConfig::flux2(),
            flow_shift: preset.flow_shift(),
        }
    }

    pub fn tiny() -> Self {
        Self {
            dit: Flux2TransformerConfig::tiny(),
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
    fn klein4b_and_dev() {
        let k = Flux2TransformerConfig::klein_4b();
        assert_eq!(k.num_layers, 5);
        assert_eq!(k.inner_dim(), 3072);
        let d = Flux2TransformerConfig::flux2_dev();
        assert_eq!(d.num_single_layers, 48);
        assert_eq!(d.inner_dim(), 6144);
        assert!(d.guidance_embeds);
    }
}
