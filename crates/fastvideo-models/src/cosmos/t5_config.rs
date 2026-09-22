//! Diffusers `T5EncoderModel` sizes for Cosmos Predict2 (`t5-11b` layout).

#[derive(Debug, Clone)]
pub struct T5Config {
    pub vocab_size: usize,
    pub d_model: usize,
    pub d_kv: usize,
    pub d_ff: usize,
    pub num_heads: usize,
    pub num_layers: usize,
    pub relative_attention_num_buckets: usize,
    pub relative_attention_max_distance: usize,
    pub eps: f64,
    /// `feed_forward_proj == "gated-gelu"` (T5 v1.1 / UMT5) vs classic Relu.
    pub is_gated: bool,
    pub max_sequence_length: usize,
}

impl T5Config {
    /// Classic `google-t5/t5-11b` encoder half — Diffusers Cosmos T5 default.
    pub fn t5_11b() -> Self {
        Self {
            vocab_size: 32_128,
            d_model: 1024,
            d_kv: 128,
            d_ff: 65_536,
            num_heads: 128,
            num_layers: 24,
            relative_attention_num_buckets: 32,
            relative_attention_max_distance: 128,
            eps: 1e-6,
            is_gated: false,
            max_sequence_length: 512,
        }
    }

    /// `google/t5-v1_1-xxl` encoder half — FLUX.1 / SD3.5 `text_encoder_2` / `_3`.
    ///
    /// Diffusers keys: `encoder.embed_tokens|shared`, `encoder.block.{i}.*`,
    /// `encoder.final_layer_norm.weight`. Gated-GELU (`wi_0` / `wi_1` / `wo`).
    pub fn t5_xxl() -> Self {
        Self {
            vocab_size: 32_128,
            d_model: 4096,
            d_kv: 64,
            d_ff: 10_240,
            num_heads: 64,
            num_layers: 24,
            relative_attention_num_buckets: 32,
            relative_attention_max_distance: 128,
            eps: 1e-6,
            is_gated: true,
            max_sequence_length: 512,
        }
    }

    pub fn tiny() -> Self {
        Self {
            vocab_size: 128,
            d_model: 32,
            d_kv: 8,
            d_ff: 64,
            num_heads: 2,
            num_layers: 1,
            relative_attention_num_buckets: 8,
            relative_attention_max_distance: 16,
            eps: 1e-6,
            is_gated: false,
            max_sequence_length: 16,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn t5_11b_matches_cosmos_text_dim() {
        let c = T5Config::t5_11b();
        assert_eq!(c.d_model, 1024);
        assert_eq!(c.num_layers, 24);
        assert!(!c.is_gated);
        assert_eq!(c.max_sequence_length, 512);
    }

    #[test]
    fn t5_xxl_matches_flux_joint_dim() {
        let c = T5Config::t5_xxl();
        assert_eq!(c.d_model, 4096);
        assert!(c.is_gated);
        assert_eq!(c.num_layers, 24);
    }
}
