//! Wan transformer architecture defaults.
//!
//! Numbers follow FastVideo `fastvideo/models/wan/config.py` (14B-class
//! defaults) and the Wan2.1-T2V-1.3B Diffusers `config.json`.

#[derive(Debug, Clone)]
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
}

impl WanVideoArchConfig {
    pub fn hidden_size(&self) -> usize {
        self.num_attention_heads * self.attention_head_dim
    }

    /// Wan 2.1 T2V 1.3B / FastWan 1.3B.
    pub fn wan_t2v_1_3b() -> Self {
        Self {
            patch_size: [1, 2, 2],
            text_len: 512,
            num_attention_heads: 12,
            attention_head_dim: 128,
            in_channels: 16,
            out_channels: 16,
            text_dim: 4096,
            freq_dim: 256,
            ffn_dim: 8960,
            num_layers: 30,
            eps: 1e-6,
            rope_max_seq_len: 1024,
        }
    }

    /// FastVideo `WanVideoArchConfig` defaults (14B-class).
    pub fn wan_t2v_14b() -> Self {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hidden_size_1_3b() {
        let cfg = WanVideoArchConfig::wan_t2v_1_3b();
        assert_eq!(cfg.hidden_size(), 1536);
        assert_eq!(cfg.num_layers, 30);
    }
}
