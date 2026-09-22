//! Z-Image host configs (2D DiT T2I). Spec: docs/ports/z-image.md.

use crate::schedulers::FlowMatchEulerDiscreteScheduler;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ZImagePreset {
    Turbo,
}

impl ZImagePreset {
    pub fn as_str(self) -> &'static str {
        "zimage_turbo"
    }

    pub fn default_height(self) -> usize {
        1024
    }

    pub fn default_width(self) -> usize {
        1024
    }

    pub fn default_steps(self) -> usize {
        8
    }

    pub fn flow_shift(self) -> f64 {
        3.0
    }

    pub fn guidance_scale(self) -> f32 {
        0.0
    }

    pub fn max_sequence_length(self) -> usize {
        512
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ZImageTransformerConfig {
    pub in_channels: usize,
    pub out_channels: usize,
    pub dim: usize,
    pub n_layers: usize,
    pub n_refiner_layers: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub cap_feat_dim: usize,
    pub rope_theta: f32,
    pub t_scale: f32,
    pub axes_dims: [usize; 3],
    pub patch_size: usize,
    pub adaln_embed_dim: usize,
}

impl ZImageTransformerConfig {
    /// FastVideo `ZImageDiTArchConfig` / Z-Image-Turbo.
    pub fn turbo() -> Self {
        Self {
            in_channels: 16,
            out_channels: 16,
            dim: 3840,
            n_layers: 30,
            n_refiner_layers: 2,
            n_heads: 30,
            n_kv_heads: 30,
            cap_feat_dim: 2560,
            rope_theta: 256.0,
            t_scale: 1000.0,
            axes_dims: [32, 48, 48],
            patch_size: 2,
            adaln_embed_dim: 256,
        }
    }

    pub fn tiny() -> Self {
        Self {
            in_channels: 4,
            out_channels: 4,
            dim: 64,
            n_layers: 2,
            n_refiner_layers: 1,
            n_heads: 4,
            n_kv_heads: 4,
            cap_feat_dim: 32,
            rope_theta: 256.0,
            t_scale: 1000.0,
            axes_dims: [8, 4, 4],
            patch_size: 2,
            adaln_embed_dim: 16,
        }
    }

    pub fn head_dim(&self) -> usize {
        self.dim / self.n_heads
    }

    pub fn latent_spatial(&self, height: usize, width: usize) -> (usize, usize) {
        // AutoencoderKL 8× downsample, then DiT patch 2.
        let lh = height / 8;
        let lw = width / 8;
        (lh, lw)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ZImageConfig {
    pub dit: ZImageTransformerConfig,
    pub flow_shift: f64,
}

impl ZImageConfig {
    pub fn for_preset(preset: ZImagePreset) -> Self {
        Self {
            dit: ZImageTransformerConfig::turbo(),
            flow_shift: preset.flow_shift(),
        }
    }

    pub fn tiny() -> Self {
        Self {
            dit: ZImageTransformerConfig::tiny(),
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
    fn turbo_dims() {
        let c = ZImageTransformerConfig::turbo();
        assert_eq!(c.dim, 3840);
        assert_eq!(c.head_dim(), 128);
        assert_eq!(c.axes_dims.iter().sum::<usize>(), 128);
        assert_eq!(c.n_layers, 30);
    }

    #[test]
    fn schedule_steps() {
        let cfg = ZImageConfig::for_preset(ZImagePreset::Turbo);
        let s = cfg.schedule(8);
        assert_eq!(s.inference_timesteps().len(), 8);
    }
}
