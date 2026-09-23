//! GEN3C DiT sizes. Spec: docs/ports/gen3c.md.
//!
//! Patch layout reuses [`crate::cosmos::CosmosTransformerConfig`] so the
//! cudarc Cosmos DiT graph loads GEN3C weights without a second block graph.

use crate::cosmos::{CosmosTransformerConfig, ExtraPosEmbed};

/// Hub recipe.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Gen3CPreset {
    Cosmos7b,
}

impl Gen3CPreset {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Cosmos7b => "gen3c_cosmos_7b",
        }
    }

    pub fn sigma_max(self) -> f64 {
        80.0
    }

    pub fn sigma_min(self) -> f64 {
        0.002
    }

    /// Official GEN3C / FastVideo `sigma_data` (Cosmos Predict2 uses 1.0).
    pub fn sigma_data(self) -> f64 {
        0.5
    }

    pub fn sigma_conditional(self) -> f64 {
        0.001
    }

    pub fn default_height(self) -> usize {
        704
    }

    pub fn default_width(self) -> usize {
        1280
    }

    pub fn default_num_frames(self) -> usize {
        121
    }

    pub fn default_fps(self) -> u32 {
        24
    }

    pub fn default_steps(self) -> usize {
        35
    }

    pub fn frame_buffer_max(self) -> usize {
        2
    }

    pub fn channels_per_buffer(self) -> usize {
        32
    }
}

/// GEN3C arch + helpers for Cosmos DiT reuse.
#[derive(Debug, Clone, PartialEq)]
pub struct Gen3CTransformerConfig {
    pub latent_channels: usize,
    pub out_channels: usize,
    pub frame_buffer_max: usize,
    pub channels_per_buffer: usize,
    pub num_attention_heads: usize,
    pub attention_head_dim: usize,
    pub num_layers: usize,
    pub mlp_ratio: f32,
    pub text_embed_dim: usize,
    pub adaln_lora_dim: usize,
    pub max_size: [usize; 3],
    pub patch_size: [usize; 3],
    pub rope_scale: [f32; 3],
    pub concat_padding_mask: bool,
}

impl Gen3CTransformerConfig {
    /// FastVideo `Gen3CArchConfig` / GEN3C-Cosmos-7B.
    pub fn cosmos_7b() -> Self {
        Self {
            latent_channels: 16,
            out_channels: 16,
            frame_buffer_max: 2,
            channels_per_buffer: 32,
            num_attention_heads: 32,
            attention_head_dim: 128,
            num_layers: 28,
            mlp_ratio: 4.0,
            text_embed_dim: 1024,
            adaln_lora_dim: 256,
            max_size: [128, 240, 240],
            patch_size: [1, 2, 2],
            rope_scale: [2.0, 1.0, 1.0],
            concat_padding_mask: true,
        }
    }

    pub fn tiny() -> Self {
        Self {
            latent_channels: 4,
            out_channels: 4,
            frame_buffer_max: 1,
            channels_per_buffer: 8,
            num_attention_heads: 2,
            attention_head_dim: 16,
            num_layers: 2,
            mlp_ratio: 2.0,
            text_embed_dim: 32,
            adaln_lora_dim: 8,
            max_size: [4, 32, 32],
            patch_size: [1, 2, 2],
            rope_scale: [2.0, 1.0, 1.0],
            concat_padding_mask: true,
        }
    }

    pub fn hidden_size(&self) -> usize {
        self.num_attention_heads * self.attention_head_dim
    }

    /// Warped buffer channels: `frame_buffer_max * 32`.
    pub fn buffer_channels(&self) -> usize {
        self.frame_buffer_max * self.channels_per_buffer
    }

    /// Channels before optional padding mask:
    /// latent + condition mask + 3D cache buffers.
    pub fn in_channels(&self) -> usize {
        self.latent_channels + 1 + self.buffer_channels()
    }

    pub fn patch_in_channels(&self) -> usize {
        if self.concat_padding_mask {
            self.in_channels() + 1
        } else {
            self.in_channels()
        }
    }

    /// Map onto the shared Cosmos DiT config (same AdaLN / RoPE graph).
    pub fn to_cosmos(&self) -> CosmosTransformerConfig {
        CosmosTransformerConfig {
            in_channels: self.in_channels(),
            out_channels: self.out_channels,
            num_attention_heads: self.num_attention_heads,
            attention_head_dim: self.attention_head_dim,
            num_layers: self.num_layers,
            mlp_ratio: self.mlp_ratio,
            text_embed_dim: self.text_embed_dim,
            adaln_lora_dim: self.adaln_lora_dim,
            max_size: self.max_size,
            patch_size: self.patch_size,
            rope_scale: self.rope_scale,
            concat_padding_mask: self.concat_padding_mask,
            extra_pos_embed_type: ExtraPosEmbed::Learnable,
            use_crossattn_projection: false,
            crossattn_proj_in_channels: self.text_embed_dim,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cosmos_7b_dims() {
        let c = Gen3CTransformerConfig::cosmos_7b();
        assert_eq!(c.hidden_size(), 4096);
        assert_eq!(c.buffer_channels(), 64);
        assert_eq!(c.in_channels(), 81);
        assert_eq!(c.patch_in_channels(), 82);
        let cosmos = c.to_cosmos();
        assert_eq!(cosmos.in_channels, 81);
        assert_eq!(cosmos.out_channels, 16);
        assert_eq!(cosmos.hidden_size(), 4096);
        assert_eq!(cosmos.num_layers, 28);
    }

    #[test]
    fn tiny_maps_to_cosmos() {
        let c = Gen3CTransformerConfig::tiny();
        assert_eq!(c.in_channels(), 4 + 1 + 8);
        assert_eq!(c.to_cosmos().patch_in_channels(), c.patch_in_channels());
    }
}
