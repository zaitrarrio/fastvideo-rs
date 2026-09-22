//! UMT5 encoder architecture constants (host-only; device graph lives in cudarc).

#[derive(Debug, Clone)]
pub struct Umt5Config {
    pub vocab_size: usize,
    pub d_model: usize,
    pub d_kv: usize,
    pub d_ff: usize,
    pub num_heads: usize,
    pub num_layers: usize,
    pub relative_attention_num_buckets: usize,
    pub relative_attention_max_distance: usize,
    pub dropout: f64,
    pub eps: f64,
}

impl Umt5Config {
    pub fn xxl() -> Self {
        Self {
            vocab_size: 256_384,
            d_model: 4096,
            d_kv: 64,
            d_ff: 10240,
            num_heads: 64,
            num_layers: 24,
            relative_attention_num_buckets: 32,
            relative_attention_max_distance: 128,
            dropout: 0.0,
            eps: 1e-6,
        }
    }

    pub fn tiny() -> Self {
        Self {
            vocab_size: 128,
            d_model: 16,
            d_kv: 8,
            d_ff: 32,
            num_heads: 2,
            num_layers: 1,
            relative_attention_num_buckets: 8,
            relative_attention_max_distance: 16,
            dropout: 0.0,
            eps: 1e-6,
        }
    }

    /// `google/byt5-small` encoder half (HunyuanVideo 1.5 `text_encoder_2`).
    pub fn byt5_small() -> Self {
        Self {
            vocab_size: 384,
            d_model: 1472,
            d_kv: 64,
            d_ff: 3584,
            num_heads: 6,
            num_layers: 12,
            relative_attention_num_buckets: 32,
            relative_attention_max_distance: 128,
            dropout: 0.0,
            eps: 1e-6,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn byt5_small_matches_hy15_dim() {
        let c = Umt5Config::byt5_small();
        assert_eq!(c.d_model, 1472);
        assert_eq!(c.num_layers, 12);
        assert_eq!(c.vocab_size, 384);
    }
}
