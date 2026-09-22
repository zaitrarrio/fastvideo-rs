//! LingBot Qwen3-VL text encode (chat template + crop 140).

use std::path::Path;

use fastvideo_models::lingbot::{tokenize_lingbot_prompt, PROMPT_CROP_START};

use crate::llm::{self, DecoderConfig};
use crate::wan::tensor::{CudaTensor, Result, TensorError};
use crate::wan::weights::WeightMap;

fn msg(s: impl Into<String>) -> TensorError {
    TensorError::Message(s.into())
}

/// Max tokenized length including the system/user template (crop + body).
pub const QWEN_MAX_LENGTH: usize = PROMPT_CROP_START + 512;

fn resolve_qwen_cfg(map: &WeightMap) -> DecoderConfig {
    let mut cfg = DecoderConfig::qwen3_vl_4b_text();
    if map.contains(&cfg.embed_key) {
        return cfg;
    }
    // Diffusers text-only packs often use `model.layers` without language_model.
    cfg.layer_prefix = "model.layers".into();
    cfg.embed_key = "model.embed_tokens.weight".into();
    cfg.final_norm_key = "model.norm.weight".into();
    cfg
}

/// Encode prompt → final hidden state cropped at [`PROMPT_CROP_START`], trimmed
/// to non-pad length (FastVideo `postprocess_lingbot_video_text`).
pub fn encode_prompt(root: &Path, prompt: &str, text_dim: usize) -> Result<CudaTensor> {
    let ids = tokenize_lingbot_prompt(root, prompt, QWEN_MAX_LENGTH).map_err(msg)?;
    if ids.is_empty() {
        return Err(msg("lingbot qwen: empty token ids"));
    }
    let map = WeightMap::open(&root.join("text_encoder")).map_err(|e| msg(e.to_string()))?;
    let cfg = resolve_qwen_cfg(&map);
    if cfg.hidden != text_dim {
        return Err(msg(format!(
            "lingbot Qwen3-VL hidden {} vs DiT text_dim {text_dim}",
            cfg.hidden
        )));
    }
    let attend = vec![true; ids.len()];
    let positions: Vec<u32> = (0..ids.len() as u32).collect();
    // Final layer residual (tap = num_layers - 1) ≡ Diffusers `hidden_states[-1]`
    // before the final RMSNorm when we read the post-block residual; our
    // `hidden_states` taps post-layer outputs. Tap last layer index.
    let tap = cfg.num_layers().saturating_sub(1);
    let mut taps = llm::hidden_states(&map, &cfg, &ids, &positions, &attend, &[tap])?;
    let hidden = taps
        .pop()
        .ok_or_else(|| msg("lingbot qwen: decoder returned no hidden state"))?;
    let seq = hidden.shape.get(1).copied().unwrap_or(0);
    if PROMPT_CROP_START >= seq {
        return Err(msg(format!(
            "lingbot qwen: crop {PROMPT_CROP_START} >= seq {seq}"
        )));
    }
    let cropped = hidden.narrow(1, PROMPT_CROP_START, seq - PROMPT_CROP_START)?;
    Ok(cropped)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crop_and_max_len() {
        assert_eq!(PROMPT_CROP_START, 140);
        assert_eq!(QWEN_MAX_LENGTH, 652);
    }

    #[test]
    fn qwen4b_dims() {
        let cfg = DecoderConfig::qwen3_vl_4b_text();
        assert_eq!(cfg.hidden, 2560);
        assert_eq!(cfg.num_layers(), 36);
        assert_eq!(cfg.heads, 32);
        assert_eq!(cfg.kv_heads, 8);
    }
}
