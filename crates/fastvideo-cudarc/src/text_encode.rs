//! Shared text-encode paths for image/audio Diffusers packs.
//!
//! Policy: when the encoder weight dir is present, run the real graph (CLIP /
//! T5 / Qwen) or return a clear load error — never silent zeros. Tiny/zeros
//! scaffolds may pass `allow_zeros=true` when no encoder dir exists.

use std::path::Path;

use fastvideo_models::cosmos::{tokenize_t5_at, T5Config};
use fastvideo_models::hunyuan15::{qwen_hidden_tap, tokenize_qwen};
use fastvideo_models::kandinsky5::tokenize_clip_at;

use crate::cosmos::t5::T5Encoder;
use crate::kandinsky5::clip_text::{ClipTextConfig, ClipTextModel};
use crate::llm::{self, DecoderConfig};
use crate::wan::pipeline::{PipelineError, Result};
use crate::wan::tensor::CudaTensor;
use crate::wan::weights::WeightMap;

fn msg(s: impl Into<String>) -> PipelineError {
    PipelineError::Message(s.into())
}

/// Resolve zeros-vs-encode: missing dir + `allow_zeros` → zeros; missing dir
/// without allow → error; present → encode.
pub fn zeros_or_encode(
    allow_zeros: bool,
    shape: &[usize],
    encoder_dir: &Path,
    encode: impl FnOnce() -> Result<CudaTensor>,
) -> Result<CudaTensor> {
    if encoder_dir.is_dir() {
        return encode();
    }
    if allow_zeros {
        return Ok(CudaTensor::zeros(shape));
    }
    Err(msg(format!(
        "text encode: missing {} (place Diffusers encoder weights, or use tiny zeros scaffold)",
        encoder_dir.display()
    )))
}

/// T5-XXL sequence embeds `[1, S, 4096]` from `text_encoder_2` (FLUX) or
/// `text_encoder_3` (SD3.5).
pub fn encode_t5_xxl(
    root: &Path,
    encoder_subdir: &str,
    tokenizer_subdir: &str,
    prompt: &str,
    max_length: usize,
) -> Result<CudaTensor> {
    let cfg = T5Config::t5_xxl();
    let max_length = max_length.min(cfg.max_sequence_length);
    let (ids, mask) = tokenize_t5_at(root, tokenizer_subdir, prompt, max_length).map_err(msg)?;
    let map = WeightMap::open(&root.join(encoder_subdir)).map_err(|e| msg(e.to_string()))?;
    let enc = T5Encoder::load(cfg.clone(), &map).map_err(|e| msg(e.to_string()))?;
    let mut embeds = enc
        .forward(&ids, 1, ids.len(), Some(&mask))
        .map_err(|e| msg(e.to_string()))?;
    let mut host = embeds.host_cow()?.to_vec();
    let d = cfg.d_model;
    for (i, &keep) in mask.iter().enumerate() {
        if !keep {
            for c in 0..d {
                host[i * d + c] = 0.0;
            }
        }
    }
    embeds = CudaTensor::from_vec(host, embeds.shape.clone())?;
    Ok(embeds)
}

/// CLIP ViT-L/14 last-layer `[1, 77, 768]`.
pub fn encode_clip_l_hidden(
    root: &Path,
    encoder_subdir: &str,
    tokenizer_subdir: &str,
    prompt: &str,
) -> Result<CudaTensor> {
    let cfg = ClipTextConfig::vit_l_14();
    let ids = tokenize_clip_at(root, tokenizer_subdir, prompt, cfg.max_position_embeddings)
        .map_err(msg)?;
    let map = WeightMap::open(&root.join(encoder_subdir)).map_err(|e| msg(e.to_string()))?;
    let model = ClipTextModel::load(&map, cfg).map_err(|e| msg(e.to_string()))?;
    model.encode_hidden(&ids).map_err(|e| msg(e.to_string()))
}

/// CLIP pooled `[1, 768]`.
pub fn encode_clip_l_pooled(
    root: &Path,
    encoder_subdir: &str,
    tokenizer_subdir: &str,
    prompt: &str,
) -> Result<CudaTensor> {
    let cfg = ClipTextConfig::vit_l_14();
    let ids = tokenize_clip_at(root, tokenizer_subdir, prompt, cfg.max_position_embeddings)
        .map_err(msg)?;
    let map = WeightMap::open(&root.join(encoder_subdir)).map_err(|e| msg(e.to_string()))?;
    let model = ClipTextModel::load(&map, cfg).map_err(|e| msg(e.to_string()))?;
    model.encode_pooled(&ids).map_err(|e| msg(e.to_string()))
}

/// Cosmos-style T5-11B `[1, S, 1024]` (MMAudio / Stable-Audio project path).
pub fn encode_t5_11b(
    root: &Path,
    encoder_subdir: &str,
    tokenizer_subdir: &str,
    prompt: &str,
    max_length: usize,
) -> Result<CudaTensor> {
    let cfg = T5Config::t5_11b();
    let max_length = max_length.min(cfg.max_sequence_length);
    let (ids, mask) = tokenize_t5_at(root, tokenizer_subdir, prompt, max_length).map_err(msg)?;
    let map = WeightMap::open(&root.join(encoder_subdir)).map_err(|e| msg(e.to_string()))?;
    let enc = T5Encoder::load(cfg.clone(), &map).map_err(|e| msg(e.to_string()))?;
    let mut embeds = enc
        .forward(&ids, 1, ids.len(), Some(&mask))
        .map_err(|e| msg(e.to_string()))?;
    let mut host = embeds.host_cow()?.to_vec();
    let d = cfg.d_model;
    for (i, &keep) in mask.iter().enumerate() {
        if !keep {
            for c in 0..d {
                host[i * d + c] = 0.0;
            }
        }
    }
    embeds = CudaTensor::from_vec(host, embeds.shape.clone())?;
    Ok(embeds)
}

/// Z-Image / Qwen3-VL-4B text tower: mid-layer tap → `[1, S, 2560]`.
pub fn encode_qwen3_cap(
    root: &Path,
    prompt: &str,
    max_length: usize,
    out_dim: usize,
) -> Result<CudaTensor> {
    let mut cfg = DecoderConfig::qwen3_vl_4b_text();
    // Diffusers text-only packs often drop `language_model.`.
    let map = WeightMap::open(&root.join("text_encoder")).map_err(|e| msg(e.to_string()))?;
    if map.contains("model.embed_tokens.weight") {
        cfg.embed_key = "model.embed_tokens.weight".into();
        cfg.final_norm_key = "model.norm.weight".into();
        cfg.layer_prefix = "model.layers".into();
    }
    if cfg.hidden != out_dim {
        return Err(msg(format!(
            "zimage Qwen3: hidden {} vs DiT cap_feat_dim {out_dim}",
            cfg.hidden
        )));
    }
    let tap = qwen_hidden_tap(cfg.num_layers());
    let ids = tokenize_qwen(root, prompt, max_length).map_err(msg)?;
    let attend = vec![true; ids.len()];
    let positions: Vec<u32> = (0..ids.len() as u32).collect();
    let mut taps = llm::hidden_states(&map, &cfg, &ids, &positions, &attend, &[tap])
        .map_err(|e| msg(e.to_string()))?;
    taps.pop()
        .ok_or_else(|| msg("zimage Qwen3: decoder returned no hidden state"))
}

/// Project or pad CLIP `[1,S,768]` into `out_dim` by repeat/truncate channels.
pub fn broadcast_to_dim(src: &CudaTensor, out_dim: usize) -> Result<CudaTensor> {
    let [b, s, d] = match src.shape[..] {
        [b, s, d] => [b, s, d],
        _ => return Err(msg(format!("broadcast_to_dim want [B,S,D], got {:?}", src.shape))),
    };
    if d == out_dim {
        return Ok(src.clone());
    }
    let host = src.host_cow()?;
    let mut out = vec![0f32; b * s * out_dim];
    for bi in 0..b {
        for si in 0..s {
            for od in 0..out_dim {
                out[(bi * s + si) * out_dim + od] = host[(bi * s + si) * d + (od % d)];
            }
        }
    }
    CudaTensor::from_vec(out, vec![b, s, out_dim]).map_err(Into::into)
}

/// Pad/truncate sequence length to `seq`.
pub fn pad_seq(src: &CudaTensor, seq: usize) -> Result<CudaTensor> {
    let [b, s, d] = match src.shape[..] {
        [b, s, d] => [b, s, d],
        _ => return Err(msg(format!("pad_seq want [B,S,D], got {:?}", src.shape))),
    };
    if s == seq {
        return Ok(src.clone());
    }
    let host = src.host_cow()?;
    let mut out = vec![0f32; b * seq * d];
    let copy_s = s.min(seq);
    for bi in 0..b {
        for si in 0..copy_s {
            let src_off = (bi * s + si) * d;
            let dst_off = (bi * seq + si) * d;
            out[dst_off..dst_off + d].copy_from_slice(&host[src_off..src_off + d]);
        }
    }
    CudaTensor::from_vec(out, vec![b, seq, d]).map_err(Into::into)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zeros_or_encode_allow() {
        let t = zeros_or_encode(true, &[1, 4, 8], Path::new("/tmp/no-such-te-xyz"), || {
            unreachable!()
        })
        .unwrap();
        assert_eq!(t.shape, vec![1, 4, 8]);
    }

    #[test]
    fn zeros_or_encode_refuse() {
        let err = zeros_or_encode(false, &[1, 4, 8], Path::new("/tmp/no-such-te-xyz"), || {
            unreachable!()
        })
        .unwrap_err();
        assert!(err.to_string().contains("missing"));
    }

    #[test]
    fn broadcast_and_pad() {
        let src = CudaTensor::zeros(&[1, 3, 4]);
        let b = broadcast_to_dim(&src, 8).unwrap();
        assert_eq!(b.shape, vec![1, 3, 8]);
        let p = pad_seq(&b, 5).unwrap();
        assert_eq!(p.shape, vec![1, 5, 8]);
    }
}
