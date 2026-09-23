//! SD 3.5 host configs (MMDiT T2I). Spec: docs/ports/sd35.md.

use crate::schedulers::FlowMatchEulerDiscreteScheduler;
use crate::vae::AutoencoderKlConfig;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Sd35Preset {
    Medium,
}

impl Sd35Preset {
    pub fn as_str(self) -> &'static str {
        "sd35_medium"
    }

    pub fn default_height(self) -> usize {
        1024
    }

    pub fn default_width(self) -> usize {
        1024
    }

    pub fn default_steps(self) -> usize {
        40
    }

    pub fn flow_shift(self) -> f64 {
        3.0
    }

    pub fn guidance_scale(self) -> f32 {
        4.5
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Sd35TransformerConfig {
    pub in_channels: usize,
    pub out_channels: usize,
    pub num_layers: usize,
    pub num_attention_heads: usize,
    pub attention_head_dim: usize,
    pub joint_attention_dim: usize,
    pub caption_projection_dim: usize,
    pub pooled_projection_dim: usize,
    pub patch_size: usize,
    pub sample_size: usize,
}

impl Sd35TransformerConfig {
    pub fn medium() -> Self {
        Self {
            in_channels: 16,
            out_channels: 16,
            num_layers: 24,
            num_attention_heads: 24,
            attention_head_dim: 64,
            joint_attention_dim: 4096,
            caption_projection_dim: 1536,
            pooled_projection_dim: 2048,
            patch_size: 2,
            sample_size: 128,
        }
    }

    pub fn tiny() -> Self {
        Self {
            in_channels: 4,
            out_channels: 4,
            num_layers: 2,
            num_attention_heads: 4,
            attention_head_dim: 8,
            joint_attention_dim: 32,
            caption_projection_dim: 32,
            pooled_projection_dim: 16,
            patch_size: 2,
            sample_size: 8,
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
pub struct Sd35Config {
    pub dit: Sd35TransformerConfig,
    pub vae: AutoencoderKlConfig,
    pub flow_shift: f64,
}

impl Sd35Config {
    pub fn for_preset(preset: Sd35Preset) -> Self {
        Self {
            dit: Sd35TransformerConfig::medium(),
            vae: AutoencoderKlConfig::sd3(),
            flow_shift: preset.flow_shift(),
        }
    }

    pub fn tiny() -> Self {
        Self {
            dit: Sd35TransformerConfig::tiny(),
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
    fn medium_dims() {
        let c = Sd35TransformerConfig::medium();
        assert_eq!(c.inner_dim(), 1536);
        assert_eq!(c.num_layers, 24);
        assert_eq!(c.in_channels, 16);
    }

    #[test]
    fn schedule_steps() {
        let cfg = Sd35Config::for_preset(Sd35Preset::Medium);
        assert_eq!(cfg.schedule(40).inference_timesteps().len(), 40);
    }
}
