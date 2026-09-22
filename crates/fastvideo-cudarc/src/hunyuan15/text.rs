//! HunyuanVideo 1.5 dual text encode: Qwen2.5-VL mid-layer + ByT5 glyphs.
//!
//! Qwen path reuses [`crate::llm`]. ByT5 is still a stub: zero embeds when the
//! prompt has no quoted glyph text; otherwise refuses until the T5 encoder
//! lands.

use std::path::Path;

use fastvideo_models::hunyuan15::{
    extract_glyph_texts, qwen_hidden_tap, tokenize_byt5, tokenize_qwen, Hunyuan15PipelineDefaults,
    Hunyuan15TransformerConfig, QWEN_CROP_START,
};
use fastvideo_models::wan::Umt5Config;

use crate::llm::{self, DecoderConfig};
use crate::wan::tensor::{CudaTensor, Result, TensorError};
use crate::wan::umt5::Umt5Encoder;
use crate::wan::weights::WeightMap;

fn msg(s: impl Into<String>) -> TensorError {
    TensorError::Message(s.into())
}

/// Dual-stream conditioning for one prompt.
#[derive(Debug)]
pub struct Hunyuan15TextConditioning {
    /// Qwen mid-layer `[1, S, text_embed_dim]` after crop.
    pub qwen: CudaTensor,
    /// ByT5 `[1, L, text_embed_2_dim]` (zeros when no glyphs).
    pub byt5: CudaTensor,
    pub qwen_attend: Vec<bool>,
    pub byt5_attend: Vec<bool>,
}

impl Hunyuan15TextConditioning {
    pub fn qwen_cfg() -> DecoderConfig {
        DecoderConfig::qwen25_vl_7b_text()
    }

    /// Tap matching Diffusers `hidden_states[-3]` on the 7B tower.
    pub fn qwen_tap() -> usize {
        qwen_hidden_tap(Self::qwen_cfg().num_layers())
    }
}

/// Encode already-tokenized Qwen ids; crop the leading template tokens.
pub fn encode_qwen_ids(
    map: &WeightMap,
    cfg: &DecoderConfig,
    ids: &[u32],
    attend: &[bool],
    crop_start: usize,
    tap: usize,
) -> Result<(CudaTensor, Vec<bool>)> {
    if ids.is_empty() {
        return Err(msg("hy15 qwen: empty token ids"));
    }
    if tap >= cfg.num_layers() {
        return Err(msg(format!(
            "hy15 qwen: tap {tap} of {}-layer tower would be post-norm",
            cfg.num_layers()
        )));
    }
    let positions: Vec<u32> = (0..ids.len() as u32).collect();
    let mut taps = llm::hidden_states(map, cfg, ids, &positions, attend, &[tap])?;
    let hidden = taps
        .pop()
        .ok_or_else(|| msg("hy15 qwen: decoder returned no hidden state"))?;
    let seq = hidden.shape.get(1).copied().unwrap_or(0);
    if crop_start >= seq {
        return Err(msg(format!(
            "hy15 qwen: crop_start {crop_start} >= seq {seq}"
        )));
    }
    let cropped = hidden.narrow(1, crop_start, seq - crop_start)?;
    let attend = attend.get(crop_start..).unwrap_or(&[]).to_vec();
    let attend = if attend.len() == cropped.shape[1] {
        attend
    } else {
        vec![true; cropped.shape[1]]
    };
    Ok((cropped, attend))
}

/// Zero ByT5 embeds (no quoted glyph text, or encoder not loaded yet).
pub fn byt5_zeros(max_length: usize, dim: usize) -> Result<(CudaTensor, Vec<bool>)> {
    let t = CudaTensor::zeros(&[1, max_length, dim]);
    Ok((t, vec![false; max_length]))
}

/// Encode ByT5 glyph text (or zeros when `glyph` is empty).
pub fn encode_byt5(
    root: &Path,
    glyph: Option<&str>,
    max_length: usize,
    dim: usize,
) -> Result<(CudaTensor, Vec<bool>)> {
    let Some(text) = glyph else {
        return byt5_zeros(max_length, dim);
    };
    let enc_dir = root.join("text_encoder_2");
    if !enc_dir.is_dir() {
        return Err(msg(format!(
            "hy15 ByT5: prompt has glyph text but {} missing",
            enc_dir.display()
        )));
    }
    let cfg = Umt5Config::byt5_small();
    if cfg.d_model != dim {
        return Err(msg(format!(
            "hy15 ByT5: d_model {} vs DiT text_embed_2_dim {dim}",
            cfg.d_model
        )));
    }
    let map = WeightMap::open(&enc_dir).map_err(|e| msg(e.to_string()))?;
    let enc = Umt5Encoder::load(cfg, &map)?;
    let ids = tokenize_byt5(text, max_length);
    let attend: Vec<bool> = ids.iter().map(|&id| id != 0).collect();
    let hidden = enc.forward(&ids, 1, max_length)?;
    Ok((hidden, attend))
}

/// Encode one prompt from a Diffusers pack root (`tokenizer/` + `text_encoder/`).
///
/// ByT5: zeros when `extract_glyph_texts` is `None`; otherwise runs the
/// `text_encoder_2` UMT5/ByT5 graph.
pub fn encode_prompt(
    root: &Path,
    prompt: &str,
    defaults: &Hunyuan15PipelineDefaults,
    dit_cfg: &Hunyuan15TransformerConfig,
) -> Result<Hunyuan15TextConditioning> {
    let qwen_cfg = Hunyuan15TextConditioning::qwen_cfg();
    let tap = Hunyuan15TextConditioning::qwen_tap();
    let ids = tokenize_qwen(root, prompt, defaults.qwen_max_length).map_err(msg)?;
    let attend = vec![true; ids.len()];
    let map = WeightMap::open(&root.join("text_encoder")).map_err(|e| msg(e.to_string()))?;
    let (qwen, qwen_attend) =
        encode_qwen_ids(&map, &qwen_cfg, &ids, &attend, defaults.text_crop_start, tap)?;

    let glyph = extract_glyph_texts(prompt);
    let (byt5, byt5_attend) = encode_byt5(
        root,
        glyph.as_deref(),
        defaults.byt5_max_length,
        dit_cfg.text_embed_2_dim,
    )?;

    debug_assert_eq!(defaults.text_crop_start, QWEN_CROP_START);
    Ok(Hunyuan15TextConditioning {
        qwen,
        byt5,
        qwen_attend,
        byt5_attend,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use fastvideo_models::hunyuan15::Hunyuan15TransformerConfig;

    #[test]
    fn byt5_zeros_shape() {
        let cfg = Hunyuan15TransformerConfig::tiny();
        let (t, mask) = byt5_zeros(8, cfg.text_embed_2_dim).unwrap();
        assert_eq!(t.shape, vec![1, 8, cfg.text_embed_2_dim]);
        assert_eq!(mask, vec![false; 8]);
    }

    #[test]
    fn qwen_tap_is_26() {
        assert_eq!(Hunyuan15TextConditioning::qwen_tap(), 26);
        assert_eq!(Hunyuan15TextConditioning::qwen_cfg().hidden, 3584);
    }
}
