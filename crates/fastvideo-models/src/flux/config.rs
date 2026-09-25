//! FLUX.1 DiT architecture extras: required keys, schnell, and weight-key inference.
//!
//! Numbers follow FastVideo `fastvideo/configs/models/dits/flux.py` and
//! published Diffusers `transformer/config.json` for
//! `black-forest-labs/FLUX.1-dev` / `FLUX.1-schnell`.

use super::{FluxPreset, FluxTransformerConfig};

impl FluxTransformerConfig {
    pub fn hidden_size(&self) -> usize {
        self.inner_dim()
    }

    pub fn mlp_hidden(&self) -> usize {
        (self.hidden_size() as f32 * self.mlp_ratio) as usize
    }

    pub fn flux1_schnell() -> Self {
        Self::flux1_dev()
    }

    pub fn from_preset(preset: &str) -> Self {
        match preset {
            "flux1_schnell" => Self::flux1_schnell(),
            _ => Self::flux1_dev(),
        }
    }

    /// Infer block counts from Diffusers / FastVideo weight keys.
    pub fn update_from_weight_keys(&mut self, keys: impl IntoIterator<Item = impl AsRef<str>>) {
        let mut num_layers = 0usize;
        let mut num_single = 0usize;
        for key in keys {
            let k = key.as_ref();
            if !k.contains("single_transformer_blocks.") && k.contains("transformer_blocks.") {
                if let Some(idx) = block_index_after(k, "transformer_blocks.") {
                    num_layers = num_layers.max(idx + 1);
                }
            }
            if k.contains("single_transformer_blocks.") {
                if let Some(idx) = block_index_after(k, "single_transformer_blocks.") {
                    num_single = num_single.max(idx + 1);
                }
            }
        }
        if num_layers > 0 {
            self.num_layers = num_layers;
        }
        if num_single > 0 {
            self.num_single_layers = num_single;
        }
    }
}

impl FluxPreset {
    pub fn from_name(preset: &str) -> Self {
        match preset {
            "flux1_schnell" => Self::Schnell,
            _ => Self::Dev,
        }
    }
}

fn block_index_after(key: &str, prefix: &str) -> Option<usize> {
    let rest = key.split(prefix).nth(1)?;
    rest.split('.').next()?.parse().ok()
}

/// Diffusers FLUX.1 `transformer/` keys that must exist.
pub const FLUX1_TRANSFORMER_REQUIRED_KEYS: &[&str] = &[
    "x_embedder.weight",
    "context_embedder.weight",
    "time_text_embed.timestep_embedder.linear_1.weight",
    "time_text_embed.timestep_embedder.linear_2.weight",
    "time_text_embed.text_embedder.linear_1.weight",
    "norm_out.linear.weight",
    "proj_out.weight",
    "transformer_blocks.0.norm1.linear.weight",
    "transformer_blocks.0.attn.to_q.weight",
    "single_transformer_blocks.0.norm.linear.weight",
];

/// CLIP-L `text_encoder/` keys.
pub const FLUX1_CLIP_REQUIRED_KEYS: &[&str] = &[
    "text_model.embeddings.token_embedding.weight",
    "text_model.embeddings.position_embedding.weight",
    "text_model.encoder.layers.0.self_attn.q_proj.weight",
    "text_model.final_layer_norm.weight",
];

/// T5-XXL `text_encoder_2/` keys.
pub const FLUX1_T5_REQUIRED_KEYS: &[&str] = &[
    "shared.weight",
    "encoder.block.0.layer.0.SelfAttention.q.weight",
    "encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight",
    "encoder.final_layer_norm.weight",
];

/// Alias used by the Flux1 port docs / weight maps.
pub type Flux1ArchConfig = FluxTransformerConfig;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn published_widths() {
        let dev = FluxTransformerConfig::flux1_dev();
        assert_eq!(dev.hidden_size(), 3072);
        assert_eq!(dev.in_channels, 64);
        assert_eq!(dev.joint_attention_dim, 4096);
        assert_eq!(dev.pooled_projection_dim, 768);
        assert_eq!(dev.axes_dims_rope, [16, 56, 56]);
        assert!(dev.guidance_embeds);
        assert_eq!(dev.mlp_hidden(), 12288);
        let schnell = FluxTransformerConfig::flux1_schnell();
        assert_eq!(schnell.num_layers, 19);
        assert_eq!(schnell.num_single_layers, 38);
    }

    #[test]
    fn published_rope_dims_sum_to_head() {
        let t = FluxTransformerConfig::flux1_dev();
        assert_eq!(t.axes_dims_rope.iter().sum::<usize>(), t.attention_head_dim);
    }
}
