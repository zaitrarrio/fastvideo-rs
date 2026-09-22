//! Cosmos Predict2 host configs. Spec: docs/ports/cosmos.md.

/// Hub recipe.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CosmosPreset {
    V2w2b,
    V2w14b,
}

impl CosmosPreset {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::V2w2b => "cosmos2_v2w_2b",
            Self::V2w14b => "cosmos2_v2w_14b",
        }
    }

    pub fn sigma_max(self) -> f64 {
        80.0
    }

    pub fn sigma_min(self) -> f64 {
        0.002
    }

    pub fn sigma_data(self) -> f64 {
        1.0
    }

    pub fn default_fps(self) -> u32 {
        16
    }
}

/// Diffusers `CosmosTransformer3DModel` sizes.
#[derive(Debug, Clone, PartialEq)]
pub struct CosmosTransformerConfig {
    /// Includes condition channel for Video2World (`latents + 1`).
    pub in_channels: usize,
    pub out_channels: usize,
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
    pub extra_pos_embed_type: ExtraPosEmbed,
    pub use_crossattn_projection: bool,
    pub crossattn_proj_in_channels: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExtraPosEmbed {
    None,
    Learnable,
}

impl CosmosTransformerConfig {
    /// Predict2-2B Video2World (NVIDIA net → Diffusers layout).
    pub fn predict2_2b() -> Self {
        Self {
            in_channels: 17,
            out_channels: 16,
            num_attention_heads: 16,
            attention_head_dim: 128,
            num_layers: 28,
            mlp_ratio: 4.0,
            text_embed_dim: 1024,
            adaln_lora_dim: 256,
            max_size: [128, 240, 240],
            patch_size: [1, 2, 2],
            rope_scale: [1.0, 3.0, 3.0],
            concat_padding_mask: true,
            extra_pos_embed_type: ExtraPosEmbed::Learnable,
            use_crossattn_projection: false,
            crossattn_proj_in_channels: 1024,
        }
    }

    /// Predict2-14B Video2World.
    pub fn predict2_14b() -> Self {
        Self {
            in_channels: 17,
            out_channels: 16,
            num_attention_heads: 40,
            attention_head_dim: 128,
            num_layers: 36,
            mlp_ratio: 4.0,
            text_embed_dim: 1024,
            adaln_lora_dim: 256,
            max_size: [128, 240, 240],
            patch_size: [1, 2, 2],
            rope_scale: [0.833_333_3, 2.0, 2.0],
            concat_padding_mask: true,
            extra_pos_embed_type: ExtraPosEmbed::Learnable,
            use_crossattn_projection: false,
            crossattn_proj_in_channels: 1024,
        }
    }

    pub fn tiny() -> Self {
        Self {
            in_channels: 5, // 4 latent + 1 cond
            out_channels: 4,
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
            extra_pos_embed_type: ExtraPosEmbed::Learnable,
            use_crossattn_projection: false,
            crossattn_proj_in_channels: 32,
        }
    }

    pub fn hidden_size(&self) -> usize {
        self.num_attention_heads * self.attention_head_dim
    }

    pub fn mlp_hidden(&self) -> usize {
        (self.hidden_size() as f32 * self.mlp_ratio) as usize
    }

    /// Latent channels excluding the Video2World condition channel.
    pub fn latent_channels(&self) -> usize {
        self.in_channels.saturating_sub(1)
    }

    /// Channels into `CosmosPatchEmbed` after optional padding mask.
    pub fn patch_in_channels(&self) -> usize {
        if self.concat_padding_mask {
            self.in_channels + 1
        } else {
            self.in_channels
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn predict2_2b_dims() {
        let c = CosmosTransformerConfig::predict2_2b();
        assert_eq!(c.hidden_size(), 2048);
        assert_eq!(c.num_layers, 28);
        assert_eq!(c.latent_channels(), 16);
        assert_eq!(c.patch_in_channels(), 18);
    }

    #[test]
    fn predict2_14b_dims() {
        let c = CosmosTransformerConfig::predict2_14b();
        assert_eq!(c.hidden_size(), 5120);
        assert_eq!(c.num_layers, 36);
    }
}
