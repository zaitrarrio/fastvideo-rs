//! Wan transformer architecture defaults.
//!
//! Numbers follow FastVideo `fastvideo/models/wan/config.py` and the published
//! Diffusers `transformer/config.json` files for each family.

#[derive(Debug, Clone, PartialEq)]
pub struct WanVideoArchConfig {
    pub patch_size: [usize; 3],
    pub text_len: usize,
    pub num_attention_heads: usize,
    pub attention_head_dim: usize,
    pub in_channels: usize,
    pub out_channels: usize,
    pub text_dim: usize,
    pub freq_dim: usize,
    pub ffn_dim: usize,
    pub num_layers: usize,
    pub eps: f32,
    pub rope_max_seq_len: usize,
    pub image_dim: Option<usize>,
    pub added_kv_proj_dim: Option<usize>,
    /// Wan2.2 MoE: `t >= boundary_ratio * num_train` uses high-noise expert.
    pub boundary_ratio: Option<f32>,
    /// Causal / Self-Forcing temporal window. `-1` is global attention.
    pub local_attn_size: i32,
    pub sink_size: usize,
    pub causal: bool,
}

impl WanVideoArchConfig {
    pub fn hidden_size(&self) -> usize {
        self.num_attention_heads * self.attention_head_dim
    }

    pub fn is_i2v(&self) -> bool {
        self.image_dim.is_some() || self.in_channels > self.out_channels
    }

    pub fn is_moe(&self) -> bool {
        self.boundary_ratio.is_some()
    }

    fn base() -> Self {
        Self {
            patch_size: [1, 2, 2],
            text_len: 512,
            num_attention_heads: 40,
            attention_head_dim: 128,
            in_channels: 16,
            out_channels: 16,
            text_dim: 4096,
            freq_dim: 256,
            ffn_dim: 13824,
            num_layers: 40,
            eps: 1e-6,
            rope_max_seq_len: 1024,
            image_dim: None,
            added_kv_proj_dim: None,
            boundary_ratio: None,
            local_attn_size: -1,
            sink_size: 0,
            causal: false,
        }
    }

    /// Wan 2.1 T2V 1.3B / FastWan 1.3B. HF `transformer/config.json`.
    pub fn wan_t2v_1_3b() -> Self {
        Self {
            num_attention_heads: 12,
            ffn_dim: 8960,
            num_layers: 30,
            ..Self::base()
        }
    }

    /// FastVideo `WanVideoArchConfig` defaults (14B T2V).
    pub fn wan_t2v_14b() -> Self {
        Self::base()
    }

    /// Wan 2.1 I2V 14B (480p/720p share the DiT; 36-channel concat).
    pub fn wan_i2v_14b() -> Self {
        Self {
            in_channels: 36,
            image_dim: Some(1280),
            added_kv_proj_dim: Some(5120),
            ..Self::base()
        }
    }

    /// Wan 2.2 TI2V 5B (48-channel VAE).
    pub fn wan_2_2_ti2v_5b() -> Self {
        Self {
            num_attention_heads: 24,
            ffn_dim: 14336,
            num_layers: 30,
            in_channels: 48,
            out_channels: 48,
            ..Self::base()
        }
    }

    /// Wan 2.2 T2V A14B high/low-noise experts (same 14B DiT, MoE routing).
    pub fn wan_2_2_t2v_a14b() -> Self {
        Self {
            boundary_ratio: Some(0.875),
            ..Self::base()
        }
    }

    /// Wan 2.2 I2V A14B.
    pub fn wan_2_2_i2v_a14b() -> Self {
        Self {
            in_channels: 36,
            image_dim: Some(1280),
            added_kv_proj_dim: Some(5120),
            boundary_ratio: Some(0.875),
            ..Self::base()
        }
    }

    /// Self-Forcing causal Wan 2.1 1.3B.
    pub fn sf_wan_t2v_1_3b() -> Self {
        Self {
            causal: true,
            local_attn_size: 21,
            sink_size: 0,
            ..Self::wan_t2v_1_3b()
        }
    }

    pub fn tiny() -> Self {
        Self {
            patch_size: [1, 2, 2],
            text_len: 8,
            num_attention_heads: 2,
            attention_head_dim: 8,
            in_channels: 4,
            out_channels: 4,
            text_dim: 16,
            freq_dim: 16,
            ffn_dim: 32,
            num_layers: 1,
            eps: 1e-6,
            rope_max_seq_len: 64,
            image_dim: None,
            added_kv_proj_dim: None,
            boundary_ratio: None,
            local_attn_size: -1,
            sink_size: 0,
            causal: false,
        }
    }

    /// Small I2V graph for tests (real 36-channel concat + added KV, not 14B).
    pub fn i2v_block() -> Self {
        Self {
            in_channels: 36,
            out_channels: 16,
            image_dim: Some(32),
            added_kv_proj_dim: Some(32),
            num_layers: 1,
            num_attention_heads: 2,
            attention_head_dim: 8,
            ffn_dim: 32,
            text_len: 8,
            text_dim: 16,
            freq_dim: 16,
            rope_max_seq_len: 64,
            ..Self::tiny()
        }
    }

    pub fn from_preset(preset: &str) -> Self {
        match preset {
            "wan_t2v_1_3b" | "fast_wan_t2v_480p" | "wan_fun_1_3b_inp" | "wan_fun_1_3b_control" => {
                Self::wan_t2v_1_3b()
            }
            "wan_t2v_14b" => Self::wan_t2v_14b(),
            "wan_i2v_14b_480p" | "wan_i2v_14b_720p" => Self::wan_i2v_14b(),
            "wan_2_2_ti2v_5b" | "fast_wan_2_2_ti2v_5b" | "lucy_edit_dev" => Self::wan_2_2_ti2v_5b(),
            "wan_2_2_t2v_a14b" | "sf_wan_2_2_t2v_a14b" => Self::wan_2_2_t2v_a14b(),
            "wan_2_2_i2v_a14b" | "sf_wan_2_2_i2v_a14b" => Self::wan_2_2_i2v_a14b(),
            "sf_wan_t2v_1_3b" => Self::sf_wan_t2v_1_3b(),
            _ => Self::wan_t2v_1_3b(),
        }
    }
}

/// Diffusers → FastVideo weight-name rewrite rules (regex pairs).
pub const PARAM_NAMES_MAPPING: &[(&str, &str)] = &[
    (r"^patch_embedding\.(.*)$", r"patch_embedding.proj.$1"),
    (
        r"^condition_embedder\.text_embedder\.linear_1\.(.*)$",
        r"condition_embedder.text_embedder.fc_in.$1",
    ),
    (
        r"^condition_embedder\.text_embedder\.linear_2\.(.*)$",
        r"condition_embedder.text_embedder.fc_out.$1",
    ),
    (
        r"^condition_embedder\.time_embedder\.linear_1\.(.*)$",
        r"condition_embedder.time_embedder.mlp.fc_in.$1",
    ),
    (
        r"^condition_embedder\.time_embedder\.linear_2\.(.*)$",
        r"condition_embedder.time_embedder.mlp.fc_out.$1",
    ),
    (
        r"^condition_embedder\.time_proj\.(.*)$",
        r"condition_embedder.time_modulation.linear.$1",
    ),
];

/// Keys that must exist in Wan 2.1 T2V 1.3B Diffusers `weight_map`.
pub const WAN_T2V_1_3B_REQUIRED_KEYS: &[&str] = &[
    "patch_embedding.weight",
    "patch_embedding.bias",
    "condition_embedder.time_embedder.linear_1.weight",
    "condition_embedder.time_embedder.linear_2.weight",
    "condition_embedder.time_proj.weight",
    "condition_embedder.text_embedder.linear_1.weight",
    "condition_embedder.text_embedder.linear_2.weight",
    "blocks.0.attn1.to_q.weight",
    "blocks.0.attn1.to_out.0.weight",
    "blocks.0.attn2.to_q.weight",
    "blocks.0.ffn.net.0.proj.weight",
    "blocks.0.ffn.net.2.weight",
    "blocks.29.attn1.to_q.weight",
    "proj_out.weight",
    "scale_shift_table",
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hidden_size_matches_hf_configs() {
        let b = WanVideoArchConfig::wan_t2v_1_3b();
        assert_eq!(b.hidden_size(), 1536);
        assert_eq!(b.num_layers, 30);
        assert_eq!(b.ffn_dim, 8960);
        let f = WanVideoArchConfig::wan_t2v_14b();
        assert_eq!(f.hidden_size(), 5120);
        assert_eq!(f.num_layers, 40);
        let i = WanVideoArchConfig::wan_i2v_14b();
        assert_eq!(i.in_channels, 36);
        assert_eq!(i.out_channels, 16);
        assert_eq!(i.image_dim, Some(1280));
        assert_eq!(i.added_kv_proj_dim, Some(5120));
        let t = WanVideoArchConfig::wan_2_2_ti2v_5b();
        assert_eq!(t.hidden_size(), 3072);
        assert_eq!(t.in_channels, 48);
        assert_eq!(t.out_channels, 48);
        let moe = WanVideoArchConfig::wan_2_2_t2v_a14b();
        assert_eq!(moe.boundary_ratio, Some(0.875));
        assert!(moe.is_moe());
        let causal = WanVideoArchConfig::sf_wan_t2v_1_3b();
        assert!(causal.causal);
        assert_eq!(causal.local_attn_size, 21);
    }
}
