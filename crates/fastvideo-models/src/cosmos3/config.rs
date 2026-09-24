//! Cosmos3-Super host configs. Spec: docs/ports/cosmos3.md.
//!
//! Canvas, TeaCache, and NVFP4 step windows are published in-tree
//! ([`crate::cosmos::sol`]). Super 64B DiT widths are not — `super_64b()`
//! stays `None` until a Hub `config.json` is vendored.

use crate::cosmos::sol::{
    OFFICIAL_FLOW_SHIFT, OFFICIAL_FPS, OFFICIAL_FRAMES, OFFICIAL_GUIDANCE, OFFICIAL_HEIGHT,
    OFFICIAL_STEPS, OFFICIAL_WIDTH,
};

/// 64B text-to-video Super. Hub id is `TODO(upstream)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Cosmos3Preset {
    Super64bT2v,
}

impl Cosmos3Preset {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Super64bT2v => "cosmos3_super_64b_t2v",
        }
    }

    /// `(height, width, num_frames)` from `models/cosmos3.toml`.
    pub fn canvas(self) -> (usize, usize, usize) {
        (OFFICIAL_HEIGHT, OFFICIAL_WIDTH, OFFICIAL_FRAMES)
    }

    pub fn default_steps(self) -> usize {
        OFFICIAL_STEPS
    }

    pub fn guidance(self) -> f32 {
        OFFICIAL_GUIDANCE
    }

    /// Recorded Super flow-shift. The Cosmos3 FlowMatch schedule applies it.
    pub fn flow_shift(self) -> f64 {
        OFFICIAL_FLOW_SHIFT
    }

    pub fn fps(self) -> u32 {
        OFFICIAL_FPS
    }
}

/// DiT sizes. Only [`Self::tiny`] is concrete; Super 64B is `TODO(upstream)`.
#[derive(Debug, Clone, PartialEq)]
pub struct Cosmos3TransformerConfig {
    pub in_channels: usize,
    pub out_channels: usize,
    pub num_attention_heads: usize,
    pub attention_head_dim: usize,
    pub num_layers: usize,
    pub mlp_ratio: f32,
    pub text_embed_dim: usize,
    pub adaln_lora_dim: usize,
    pub patch_size: [usize; 3],
}

impl Cosmos3TransformerConfig {
    /// Unit-test graph. Not Super 64B.
    pub fn tiny() -> Self {
        Self {
            in_channels: 4,
            out_channels: 4,
            num_attention_heads: 2,
            attention_head_dim: 16,
            num_layers: 2,
            mlp_ratio: 2.0,
            text_embed_dim: 32,
            adaln_lora_dim: 8,
            patch_size: [1, 2, 2],
        }
    }

    /// Super 64B DiT. `None` until Hub `config.json` is vendored.
    pub fn super_64b() -> Option<Self> {
        None
    }

    pub fn hidden_size(&self) -> usize {
        self.num_attention_heads * self.attention_head_dim
    }

    pub fn mlp_hidden(&self) -> usize {
        (self.hidden_size() as f32 * self.mlp_ratio) as usize
    }

    pub fn patch_volume(&self) -> usize {
        self.patch_size.iter().product()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn official_canvas_matches_sol() {
        let p = Cosmos3Preset::Super64bT2v;
        assert_eq!(p.as_str(), "cosmos3_super_64b_t2v");
        assert_eq!(p.canvas(), (720, 1280, 189));
        assert_eq!(p.default_steps(), 35);
        assert_eq!(p.guidance(), 6.0);
        assert_eq!(p.flow_shift(), 10.0);
        assert_eq!(p.fps(), 24);
    }

    #[test]
    fn super_64b_dims_are_upstream() {
        assert!(Cosmos3TransformerConfig::super_64b().is_none());
    }

    #[test]
    fn tiny_dims() {
        let c = Cosmos3TransformerConfig::tiny();
        assert_eq!(c.hidden_size(), 32);
        assert_eq!(c.mlp_hidden(), 64);
        assert_eq!(c.patch_volume(), 4);
        assert_eq!(c.in_channels, c.out_channels);
    }
}
