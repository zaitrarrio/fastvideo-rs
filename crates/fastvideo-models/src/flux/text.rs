//! FLUX.1 text configs and tokenize helpers (CLIP-L pooled + T5-XXL tokens).
//!
//! Encoders live in `fastvideo-cudarc`. This module is host-only (no Candle).

use crate::wan::Umt5Config;

#[derive(Debug, Clone, PartialEq)]
pub struct ClipTextConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub max_position_embeddings: usize,
    pub layer_norm_eps: f64,
    pub pad_token_id: u32,
    pub eos_token_id: u32,
    pub text_len: usize,
}

impl ClipTextConfig {
    /// `openai/clip-vit-large-patch14` (FLUX.1 `text_encoder/`).
    pub fn clip_l() -> Self {
        Self {
            vocab_size: 49408,
            hidden_size: 768,
            intermediate_size: 3072,
            num_hidden_layers: 12,
            num_attention_heads: 12,
            max_position_embeddings: 77,
            layer_norm_eps: 1e-5,
            pad_token_id: 0,
            eos_token_id: 49407,
            text_len: 77,
        }
    }

    pub fn tiny() -> Self {
        Self {
            vocab_size: 32,
            hidden_size: 16,
            intermediate_size: 32,
            num_hidden_layers: 1,
            num_attention_heads: 2,
            max_position_embeddings: 8,
            layer_norm_eps: 1e-5,
            pad_token_id: 0,
            eos_token_id: 2,
            text_len: 8,
        }
    }
}

#[derive(Debug, Clone)]
pub struct T5Config {
    pub inner: Umt5Config,
    pub text_len: usize,
    pub pad_token_id: u32,
}

impl T5Config {
    /// T5-v1.1-XXL (`google/t5-v1_1-xxl`) used by FLUX.1 `text_encoder_2/`.
    pub fn xxl() -> Self {
        Self {
            inner: Umt5Config {
                vocab_size: 32_128,
                d_model: 4096,
                d_kv: 64,
                d_ff: 10240,
                num_heads: 64,
                num_layers: 24,
                relative_attention_num_buckets: 32,
                relative_attention_max_distance: 128,
                dropout: 0.0,
                eps: 1e-6,
            },
            text_len: 512,
            pad_token_id: 0,
        }
    }

    pub fn tiny() -> Self {
        Self {
            inner: Umt5Config::tiny(),
            text_len: 8,
            pad_token_id: 0,
        }
    }

    /// Schnell Diffusers default `max_sequence_length=256`.
    pub fn xxl_schnell() -> Self {
        Self {
            text_len: 256,
            ..Self::xxl()
        }
    }
}

pub fn flux1_dummy_text() -> bool {
    matches!(
        std::env::var("FASTVIDEO_FLUX1_DUMMY_TEXT").as_deref(),
        Ok("1") | Ok("true") | Ok("TRUE")
    )
}

pub fn flux1_t5_len(default: usize) -> usize {
    std::env::var("FASTVIDEO_FLUX1_TEXT_LEN")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
        .max(1)
}

pub fn pad_token_ids(ids: &[u32], text_len: usize, pad_id: u32) -> (Vec<u32>, usize) {
    crate::flux2::pad_token_ids(ids, text_len, pad_id)
}

pub fn tokenize_flux1(path: &str, prompt: &str, max_len: usize) -> Result<(Vec<u32>, usize), String> {
    crate::flux2::tokenize_flux2(path, prompt, max_len)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clip_l_and_t5_xxl_widths() {
        let clip = ClipTextConfig::clip_l();
        assert_eq!(clip.hidden_size, 768);
        assert_eq!(clip.text_len, 77);
        let t5 = T5Config::xxl();
        assert_eq!(t5.inner.d_model, 4096);
        assert_eq!(t5.text_len, 512);
        assert_eq!(T5Config::xxl_schnell().text_len, 256);
    }
}
