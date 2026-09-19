//! Flux2 DiT / VAE architecture defaults.
//!
//! Numbers follow FastVideo `fastvideo/configs/models/dits/flux_2.py`,
//! `vaes/flux2vae.py`, and published Diffusers `transformer/config.json`
//! (Klein 4B from `black-forest-labs/FLUX.2-klein-4B`).

#[derive(Debug, Clone, PartialEq)]
pub struct Flux2ArchConfig {
    pub patch_size: usize,
    pub in_channels: usize,
    pub out_channels: usize,
    /// Double-stream transformer blocks.
    pub num_layers: usize,
    /// Single-stream transformer blocks.
    pub num_single_layers: usize,
    pub attention_head_dim: usize,
    pub num_attention_heads: usize,
    /// Text encoder output dim after layer-stack concat.
    pub joint_attention_dim: usize,
    pub timestep_guidance_channels: usize,
    pub mlp_ratio: f32,
    pub axes_dims_rope: [usize; 4],
    pub rope_theta: f32,
    pub eps: f32,
    pub guidance_embeds: bool,
}

impl Flux2ArchConfig {
    pub fn hidden_size(&self) -> usize {
        self.num_attention_heads * self.attention_head_dim
    }

    pub fn mlp_hidden(&self) -> usize {
        (self.hidden_size() as f32 * self.mlp_ratio) as usize
    }

    fn base() -> Self {
        Self {
            patch_size: 1,
            in_channels: 128,
            out_channels: 128,
            num_layers: 19,
            num_single_layers: 38,
            attention_head_dim: 128,
            num_attention_heads: 24,
            joint_attention_dim: 15360,
            timestep_guidance_channels: 256,
            mlp_ratio: 3.0,
            axes_dims_rope: [32, 32, 32, 32],
            rope_theta: 2000.0,
            eps: 1e-6,
            guidance_embeds: true,
        }
    }

    /// Full FLUX.2-dev (Mistral3 text, embedded guidance). FastVideo defaults.
    pub fn flux2_dev() -> Self {
        Self::base()
    }

    /// FLUX.2 Klein 4B distilled (Qwen3 text, no guidance embeds).
    /// HF `transformer/config.json` for `black-forest-labs/FLUX.2-klein-4B`.
    pub fn flux2_klein_4b() -> Self {
        Self {
            num_layers: 5,
            num_single_layers: 20,
            joint_attention_dim: 7680,
            guidance_embeds: false,
            ..Self::base()
        }
    }

    /// FLUX.2 Klein 9B: same Klein pipeline; layer counts inferred from
    /// `transformer/config.json` or checkpoint keys when present.
    pub fn flux2_klein_9b() -> Self {
        Self {
            guidance_embeds: false,
            joint_attention_dim: 7680,
            ..Self::base()
        }
    }

    /// Zero-weight CI graph (one double + one single block).
    pub fn tiny() -> Self {
        Self {
            patch_size: 1,
            in_channels: 8,
            out_channels: 8,
            num_layers: 1,
            num_single_layers: 1,
            attention_head_dim: 8,
            num_attention_heads: 2,
            joint_attention_dim: 16,
            timestep_guidance_channels: 16,
            mlp_ratio: 2.0,
            axes_dims_rope: [2, 2, 2, 2],
            rope_theta: 2000.0,
            eps: 1e-6,
            guidance_embeds: true,
        }
    }

    /// Tiny Klein: no guidance embeds, stacked-layer text dim.
    /// Three Qwen3 hidden states (tiny encoder width 16) concat to 48.
    pub fn tiny_klein() -> Self {
        Self {
            guidance_embeds: false,
            joint_attention_dim: 48,
            ..Self::tiny()
        }
    }

    pub fn from_preset(preset: &str) -> Self {
        match preset {
            "flux2_dev" => Self::flux2_dev(),
            "flux2_klein_4b" => Self::flux2_klein_4b(),
            "flux2_klein_9b" => Self::flux2_klein_9b(),
            "tiny_klein" => Self::tiny_klein(),
            _ => Self::flux2_dev(),
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

fn block_index_after(key: &str, prefix: &str) -> Option<usize> {
    let rest = key.split(prefix).nth(1)?;
    rest.split('.').next()?.parse().ok()
}

#[derive(Debug, Clone, PartialEq)]
pub struct Flux2VaeConfig {
    pub in_channels: usize,
    pub out_channels: usize,
    pub block_out_channels: Vec<usize>,
    pub layers_per_block: usize,
    pub latent_channels: usize,
    pub norm_num_groups: usize,
    pub scaling_factor: f32,
    pub spatial_compression_ratio: usize,
}

impl Flux2VaeConfig {
    pub fn flux2() -> Self {
        Self {
            in_channels: 3,
            out_channels: 3,
            block_out_channels: vec![128, 256, 512, 512],
            layers_per_block: 2,
            latent_channels: 32,
            norm_num_groups: 32,
            scaling_factor: 0.13025,
            spatial_compression_ratio: 8,
        }
    }

    pub fn tiny() -> Self {
        Self {
            in_channels: 3,
            out_channels: 3,
            block_out_channels: vec![8, 16],
            layers_per_block: 1,
            latent_channels: 8,
            norm_num_groups: 2,
            scaling_factor: 0.13025,
            spatial_compression_ratio: 4,
        }
    }

    /// Small full-path decoder (3 up blocks) for Candle↔cudarc parity tests.
    pub fn small() -> Self {
        Self {
            in_channels: 3,
            out_channels: 3,
            block_out_channels: vec![8, 8, 8],
            layers_per_block: 1,
            latent_channels: 4,
            norm_num_groups: 2,
            scaling_factor: 0.13025,
            spatial_compression_ratio: 4,
        }
    }
}

/// Diffusers → FastVideo transformer key rewrite (identity for Diffusers).
pub const PARAM_NAMES_MAPPING: &[(&str, &str)] = &[(r"^transformer\.(\w*)\.(.*)$", r"$1.$2")];

/// Keys that must exist on a Flux2 Diffusers `transformer/` weight map.
pub const FLUX2_TRANSFORMER_REQUIRED_KEYS: &[&str] = &[
    "x_embedder.weight",
    "context_embedder.weight",
    "time_guidance_embed.timestep_embedder.linear_1.weight",
    "time_guidance_embed.timestep_embedder.linear_2.weight",
    "double_stream_modulation_img.linear.weight",
    "double_stream_modulation_txt.linear.weight",
    "single_stream_modulation.linear.weight",
    "norm_out.linear.weight",
    "proj_out.weight",
];

/// Keys that must exist on a Flux2 Diffusers `vae/` weight map.
pub const FLUX2_VAE_REQUIRED_KEYS: &[&str] = &[
    "encoder.conv_in.weight",
    "decoder.conv_out.weight",
    "quant_conv.weight",
    "post_quant_conv.weight",
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hidden_size_matches_published_configs() {
        let dev = Flux2ArchConfig::flux2_dev();
        assert_eq!(dev.hidden_size(), 3072);
        assert_eq!(dev.joint_attention_dim, 15360);
        assert!(dev.guidance_embeds);
        assert_eq!(dev.in_channels, 128);
        let klein = Flux2ArchConfig::flux2_klein_4b();
        assert_eq!(klein.num_layers, 5);
        assert_eq!(klein.num_single_layers, 20);
        assert_eq!(klein.joint_attention_dim, 7680);
        assert!(!klein.guidance_embeds);
        assert_eq!(klein.hidden_size(), 3072);
    }

    #[test]
    fn infers_block_counts_from_keys() {
        let mut cfg = Flux2ArchConfig::flux2_dev();
        cfg.update_from_weight_keys([
            "transformer_blocks.0.attn.to_q.weight",
            "transformer_blocks.4.ff.linear_in.weight",
            "single_transformer_blocks.0.attn.to_out.weight",
            "single_transformer_blocks.19.attn.norm_q.weight",
        ]);
        assert_eq!(cfg.num_layers, 5);
        assert_eq!(cfg.num_single_layers, 20);
    }

    #[test]
    fn preset_dispatch() {
        assert_eq!(
            Flux2ArchConfig::from_preset("flux2_klein_4b").num_layers,
            5
        );
        assert!(Flux2ArchConfig::from_preset("flux2_dev").guidance_embeds);
    }
}
