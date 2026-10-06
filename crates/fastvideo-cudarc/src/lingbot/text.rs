//! LingBot Qwen3-VL text encode (`LingBotVideoPipeline.encode_prompt`).
//!
//! `PROMPT_TEMPLATE.format(prompt)` → Qwen3-VL processor (single prompt: no
//! padding) → `outputs.hidden_states[-1]`, which transformers ≥ 5 ties to
//! `last_hidden_state`, i.e. the output **after** the final RMSNorm (tap
//! `num_layers` here) → drop the first `crop_start` template tokens.
//! Text-only Qwen3-VL positions are `0..L` on all three M-RoPE axes, which
//! makes its interleaved M-RoPE an ordinary rotary.

use std::path::Path;

use fastvideo_models::lingbot::{tokenize_lingbot_prompt, PROMPT_CROP_START};

use crate::llm::{DecoderConfig, ResidentDecoder};
use crate::wan::tensor::{CudaTensor, Result, TensorError};
use crate::wan::weights::WeightMap;

fn msg(s: impl Into<String>) -> TensorError {
    TensorError::Message(s.into())
}

/// Tokens kept after the crop (`TOKEN_LENGTH` truncation is far above any
/// prompt; this bounds a runaway input).
pub const QWEN_MAX_LENGTH: usize = fastvideo_models::lingbot::config::TOKEN_LENGTH;

fn resolve_qwen_cfg(map: &WeightMap) -> DecoderConfig {
    let mut cfg = DecoderConfig::qwen3_vl_4b_text();
    if map.contains(&cfg.embed_key) {
        return cfg;
    }
    // Text-only packs that drop `language_model.`.
    cfg.layer_prefix = "model.layers".into();
    cfg.embed_key = "model.embed_tokens.weight".into();
    cfg.final_norm_key = "model.norm.weight".into();
    cfg
}

/// The resident Qwen3-VL text tower of a LingBot tree.
pub struct LingBotTextEncoder {
    decoder: ResidentDecoder,
    root: std::path::PathBuf,
}

impl LingBotTextEncoder {
    pub fn load(root: &Path, text_dim: usize) -> Result<Self> {
        let map = WeightMap::open(&root.join("text_encoder")).map_err(|e| msg(e.to_string()))?;
        let cfg = resolve_qwen_cfg(&map);
        if cfg.hidden != text_dim {
            return Err(msg(format!(
                "lingbot Qwen3-VL hidden {} vs DiT text_dim {text_dim}",
                cfg.hidden
            )));
        }
        let n = cfg.num_layers();
        Ok(Self {
            decoder: ResidentDecoder::load(&map, &cfg, n)?,
            root: root.to_path_buf(),
        })
    }

    /// `[1, L - crop, text_dim]` hidden states of `prompt`.
    pub fn encode(&self, prompt: &str) -> Result<CudaTensor> {
        let (ids, crop) = tokenize_lingbot_prompt(&self.root, prompt, QWEN_MAX_LENGTH).map_err(msg)?;
        if crop != PROMPT_CROP_START {
            crate::wan::log::info(format_args!(
                "lingbot text: template crop {crop} (reference processor: {PROMPT_CROP_START})"
            ));
        }
        if ids.len() <= crop {
            return Err(msg(format!("lingbot qwen: {} tokens, crop {crop}", ids.len())));
        }
        let attend = vec![true; ids.len()];
        let positions: Vec<u32> = (0..ids.len() as u32).collect();
        let tap = self.decoder.config().num_layers();
        let hidden = self
            .decoder
            .hidden_states(&ids, &positions, &attend, &[tap])?
            .pop()
            .ok_or_else(|| msg("lingbot qwen: no hidden state"))?;
        let seq = hidden.shape[1];
        hidden.narrow(1, crop, seq - crop)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qwen4b_dims() {
        let cfg = DecoderConfig::qwen3_vl_4b_text();
        assert_eq!(cfg.hidden, 2560);
        assert_eq!(cfg.num_layers(), 36);
        assert_eq!(cfg.heads, 32);
        assert_eq!(cfg.kv_heads, 8);
        assert_eq!(cfg.layer_prefix, "model.language_model.layers");
    }
}
