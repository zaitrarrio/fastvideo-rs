//! Kandinsky 5 dual text: Qwen2.5-VL (crop 129) + CLIP-L pooled (768).

use std::path::Path;

use fastvideo_models::hunyuan15::tokenize_qwen;
use fastvideo_models::kandinsky5::Kandinsky5TransformerConfig;

use crate::hunyuan15::text::{encode_qwen_ids, Hunyuan15TextConditioning};
use crate::llm::DecoderConfig;
use crate::wan::tensor::{CudaTensor, Result, TensorError};
use crate::wan::weights::WeightMap;

use super::clip_text::{ClipTextConfig, ClipTextModel};

fn msg(s: impl Into<String>) -> TensorError {
    TensorError::Message(s.into())
}

/// Diffusers Kandinsky5 `prompt_template_encode_start_idx`.
pub const QWEN_CROP_START: usize = 129;

/// Qwen max length including template (FV: crop + 512).
pub const QWEN_MAX_LENGTH: usize = QWEN_CROP_START + 512;

#[derive(Debug)]
pub struct Kandinsky5TextConditioning {
    /// Qwen mid-layer `[1, S, 3584]` after crop.
    pub qwen: CudaTensor,
    /// CLIP pooled `[1, 768]`.
    pub clip_pooled: CudaTensor,
}

pub fn encode_prompt(
    root: &Path,
    prompt: &str,
    cfg: &Kandinsky5TransformerConfig,
) -> Result<Kandinsky5TextConditioning> {
    let qwen_cfg = DecoderConfig::qwen25_vl_7b_text();
    let tap = Hunyuan15TextConditioning::qwen_tap();
    let ids = tokenize_qwen(root, prompt, QWEN_MAX_LENGTH).map_err(msg)?;
    let attend = vec![true; ids.len()];
    let map = WeightMap::open(&root.join("text_encoder")).map_err(|e| msg(e.to_string()))?;
    let crop = cfg.qwen_crop_start;
    let (qwen, _) = encode_qwen_ids(&map, &qwen_cfg, &ids, &attend, crop, tap)?;

    let clip_pooled = if root.join("text_encoder_2").is_dir() {
        encode_clip_pooled(root, prompt, cfg.in_text_dim2)?
    } else {
        CudaTensor::zeros(&[1, cfg.in_text_dim2])
    };
    Ok(Kandinsky5TextConditioning { qwen, clip_pooled })
}

fn encode_clip_pooled(root: &Path, prompt: &str, dim: usize) -> Result<CudaTensor> {
    let clip_cfg = ClipTextConfig::vit_l_14();
    if clip_cfg.hidden_size != dim {
        return Err(msg(format!(
            "k5 CLIP hidden {} vs DiT in_text_dim2 {dim}",
            clip_cfg.hidden_size
        )));
    }
    let ids =
        fastvideo_models::kandinsky5::tokenize_clip(root, prompt, clip_cfg.max_position_embeddings)
            .map_err(msg)?;
    let map = WeightMap::open(&root.join("text_encoder_2")).map_err(|e| msg(e.to_string()))?;
    let model = ClipTextModel::load(&map, clip_cfg)?;
    model.encode_pooled(&ids)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crop_constant() {
        assert_eq!(QWEN_CROP_START, 129);
    }
}
