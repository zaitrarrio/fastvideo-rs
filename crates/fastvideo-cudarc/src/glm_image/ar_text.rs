//! GLM-Image autoregressive text tower (`vision_language_encoder/`).
//!
//! Diffusers `GlmImagePipeline` uses:
//! - `text_encoder/` — ByT5 glyph embeds (`T5EncoderModel`, d_model=1472)
//! - `vision_language_encoder/` — `GlmImageForConditionalGeneration` AR tower
//!   (GLM text LM, hidden=4096, 40 layers) that produces prior visual tokens
//!
//! This module runs the AR **text** path (embed → decoder layers → hidden) when
//! the Hub pack is present. Full VQ prior `generate()` remains a follow-on.

use std::path::Path;

use fastvideo_models::hunyuan15::tokenize_byt5;

use crate::llm::{self, Act, DecoderConfig, LayerAttn};
use crate::wan::pipeline::{PipelineError, Result};
use crate::wan::tensor::CudaTensor;
use crate::wan::umt5::Umt5Encoder;
use crate::wan::weights::WeightMap;
use fastvideo_models::wan::Umt5Config;

fn msg(s: impl Into<String>) -> PipelineError {
    PipelineError::Message(s.into())
}

/// Hub probes for `vision_language_encoder/` (GlmImage text LM).
pub mod probes {
    pub const PROBES: &[&str] = &[
        "model.language_model.embed_tokens.weight",
        "language_model.embed_tokens.weight",
        "model.embed_tokens.weight",
        "model.language_model.layers.0.self_attn.q_proj.weight",
        "language_model.layers.0.self_attn.q_proj.weight",
    ];
}

/// ByT5 glyph probes under `text_encoder/`.
pub mod byt5_probes {
    pub const PROBES: &[&str] = &[
        "encoder.block.0.layer.0.SelfAttention.q.weight",
        "encoder.embed_tokens.weight",
        "shared.weight",
        "encoder.final_layer_norm.weight",
    ];
}

impl DecoderConfig {
    /// `zai-org/GLM-Image` `vision_language_encoder` text config (GLM-4-9B family).
    pub fn glm_image_ar_text() -> Self {
        let mut layer = LayerAttn::global(10_000.0, 1.0);
        layer.partial_rotary = Some(0.5);
        let layers = vec![layer; 40];
        Self {
            vocab: 168_064,
            hidden: 4096,
            heads: 32,
            kv_heads: 2,
            head_dim: 128,
            intermediate: 13_696,
            rms_eps: 1e-5,
            norm_offset: 0.0,
            act: Act::Silu,
            qk_norm: false,
            sandwich_norms: false,
            embed_scale: 1.0,
            attn_scale: (128f32).powf(-0.5),
            layers,
            layer_prefix: "model.language_model.layers".into(),
            embed_key: "model.language_model.embed_tokens.weight".into(),
            final_norm_key: "model.language_model.norm.weight".into(),
            attention_k_eq_v: false,
        }
    }
}

fn remap_ar_keys(cfg: &mut DecoderConfig, map: &WeightMap) {
    if map.contains("model.language_model.embed_tokens.weight") {
        cfg.embed_key = "model.language_model.embed_tokens.weight".into();
        cfg.final_norm_key = "model.language_model.norm.weight".into();
        cfg.layer_prefix = "model.language_model.layers".into();
    } else if map.contains("language_model.embed_tokens.weight") {
        cfg.embed_key = "language_model.embed_tokens.weight".into();
        cfg.final_norm_key = "language_model.norm.weight".into();
        cfg.layer_prefix = "language_model.layers".into();
    } else if map.contains("model.embed_tokens.weight") {
        cfg.embed_key = "model.embed_tokens.weight".into();
        cfg.final_norm_key = "model.norm.weight".into();
        cfg.layer_prefix = "model.layers".into();
    }
}

/// Encode prompt through the AR text LM → `[1, S, hidden]`.
pub fn encode_ar_hidden(
    root: &Path,
    subdir: &str,
    prompt: &str,
    max_length: usize,
) -> Result<CudaTensor> {
    let dir = root.join(subdir);
    let map = WeightMap::open(&dir).map_err(|e| msg(e.to_string()))?;
    let hit = probes::PROBES.iter().find(|k| map.contains(k));
    if hit.is_none() {
        return Err(msg(format!(
            "glm_image AR: {} present but no language_model embed keys {:?}",
            dir.display(),
            probes::PROBES.iter().take(3).collect::<Vec<_>>()
        )));
    }
    let mut cfg = DecoderConfig::glm_image_ar_text();
    remap_ar_keys(&mut cfg, &map);
    // Byte-level ids are fine for a conditioning scaffold; Hub processor uses a
    // chat template — we still run the real weight graph on these ids.
    let mut ids = tokenize_byt5(prompt, max_length);
    if ids.is_empty() {
        ids.push(0);
    }
    let attend = vec![true; ids.len()];
    let positions: Vec<u32> = (0..ids.len() as u32).collect();
    let tap = cfg.num_layers().saturating_sub(1);
    let mut taps = llm::hidden_states(&map, &cfg, &ids, &positions, &attend, &[tap])
        .map_err(|e| msg(e.to_string()))?;
    taps.pop()
        .ok_or_else(|| msg("glm_image AR: decoder returned no hidden state"))
}

/// ByT5 glyph encode matching Hub `text_encoder/` (d_model=1472).
pub fn encode_byt5_glyphs(
    root: &Path,
    encoder_subdir: &str,
    prompt: &str,
    max_length: usize,
) -> Result<CudaTensor> {
    let dir = root.join(encoder_subdir);
    let map = WeightMap::open(&dir).map_err(|e| msg(e.to_string()))?;
    if byt5_probes::PROBES.iter().all(|k| !map.contains(k)) {
        return Err(msg(format!(
            "glm_image ByT5: {} present but missing encoder.block / shared.weight",
            dir.display()
        )));
    }
    let cfg = Umt5Config::byt5_small();
    let enc = Umt5Encoder::load(cfg, &map).map_err(|e| msg(e.to_string()))?;
    let ids = tokenize_byt5(prompt, max_length);
    enc.forward(&ids, 1, max_length)
        .map_err(|e| msg(e.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ar_config_matches_hub() {
        let c = DecoderConfig::glm_image_ar_text();
        assert_eq!(c.hidden, 4096);
        assert_eq!(c.num_layers(), 40);
        assert_eq!(c.kv_heads, 2);
        assert_eq!(c.vocab, 168_064);
    }
}
