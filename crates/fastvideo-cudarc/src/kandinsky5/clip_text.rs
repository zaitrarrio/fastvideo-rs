//! CLIP ViT-L/14 text encoder (`text_encoder_2`) for Kandinsky 5 pooled embeds.

use std::path::Path;

use fastvideo_models::kandinsky5::tokenize_clip;

use crate::wan::nn::{self, Linear};
use crate::wan::tensor::{CudaTensor, Result, TensorError};
use crate::wan::weights::{self, WeightMap};

fn msg(s: impl Into<String>) -> TensorError {
    TensorError::Message(s.into())
}

#[derive(Debug, Clone)]
pub struct ClipTextConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub max_position_embeddings: usize,
    pub vocab_size: usize,
    pub layer_norm_eps: f32,
    pub eos_token_id: u32,
}

impl ClipTextConfig {
    /// `openai/clip-vit-large-patch14` text tower.
    pub fn vit_l_14() -> Self {
        Self {
            hidden_size: 768,
            intermediate_size: 3072,
            num_hidden_layers: 12,
            num_attention_heads: 12,
            max_position_embeddings: 77,
            vocab_size: 49408,
            layer_norm_eps: 1e-5,
            eos_token_id: 49407,
        }
    }

    pub fn tiny() -> Self {
        Self {
            hidden_size: 32,
            intermediate_size: 64,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            max_position_embeddings: 8,
            vocab_size: 100,
            layer_norm_eps: 1e-5,
            eos_token_id: 2,
        }
    }
}

/// OpenAI CLIP `quick_gelu`: `x * sigmoid(1.702 x)`.
fn quick_gelu(x: &CudaTensor) -> Result<CudaTensor> {
    let host = x.host_cow()?;
    let out: Vec<f32> = host
        .iter()
        .map(|&v| {
            let s = 1.0 / (1.0 + (-1.702 * v).exp());
            v * s
        })
        .collect();
    Ok(CudaTensor::from_vec(out, x.shape.clone())?.to_device()?)
}

fn causal_mask(seq: usize) -> Result<CudaTensor> {
    let mut m = vec![0f32; seq * seq];
    for i in 0..seq {
        for j in (i + 1)..seq {
            m[i * seq + j] = f32::NEG_INFINITY;
        }
    }
    Ok(CudaTensor::from_vec(m, vec![1, 1, seq, seq])?)
}

struct ClipTextAttention {
    q: Linear,
    k: Linear,
    v: Linear,
    out: Linear,
    heads: usize,
    dim_head: usize,
}

impl ClipTextAttention {
    fn zeros(dim: usize, heads: usize) -> Result<Self> {
        let w = || {
            Linear::from_tensors(
                CudaTensor::zeros(&[dim, dim]),
                Some(CudaTensor::zeros(&[dim])),
            )
        };
        Ok(Self {
            q: w()?,
            k: w()?,
            v: w()?,
            out: w()?,
            heads,
            dim_head: dim / heads,
        })
    }

    fn load(map: &WeightMap, prefix: &str, dim: usize, heads: usize) -> Result<Self> {
        Ok(Self {
            q: Linear::load(map, &weights::join_key(prefix, "q_proj"), dim, dim, true)?,
            k: Linear::load(map, &weights::join_key(prefix, "k_proj"), dim, dim, true)?,
            v: Linear::load(map, &weights::join_key(prefix, "v_proj"), dim, dim, true)?,
            out: Linear::load(map, &weights::join_key(prefix, "out_proj"), dim, dim, true)?,
            heads,
            dim_head: dim / heads,
        })
    }

    fn forward(&self, xs: &CudaTensor, mask: &CudaTensor) -> Result<CudaTensor> {
        let (b, s, _) = (xs.shape[0], xs.shape[1], xs.shape[2]);
        let reshape = |t: CudaTensor| -> Result<CudaTensor> {
            t.reshape(vec![b, s, self.heads, self.dim_head])?
                .transpose(1, 2)
        };
        let q = reshape(self.q.forward(xs)?)?;
        let k = reshape(self.k.forward(xs)?)?;
        let v = reshape(self.v.forward(xs)?)?;
        let attn = nn::scaled_dot_product_attention_masked(&q, &k, &v, None, Some(mask))?;
        let attn = attn
            .transpose(1, 2)?
            .reshape(vec![b, s, self.heads * self.dim_head])?;
        self.out.forward(&attn)
    }
}

struct ClipTextMlp {
    fc1: Linear,
    fc2: Linear,
}

impl ClipTextMlp {
    fn zeros(dim: usize, inter: usize) -> Result<Self> {
        Ok(Self {
            fc1: Linear::from_tensors(
                CudaTensor::zeros(&[inter, dim]),
                Some(CudaTensor::zeros(&[inter])),
            )?,
            fc2: Linear::from_tensors(
                CudaTensor::zeros(&[dim, inter]),
                Some(CudaTensor::zeros(&[dim])),
            )?,
        })
    }

    fn load(map: &WeightMap, prefix: &str, dim: usize, inter: usize) -> Result<Self> {
        Ok(Self {
            fc1: Linear::load(map, &weights::join_key(prefix, "fc1"), dim, inter, true)?,
            fc2: Linear::load(map, &weights::join_key(prefix, "fc2"), inter, dim, true)?,
        })
    }

    fn forward(&self, xs: &CudaTensor) -> Result<CudaTensor> {
        let h = quick_gelu(&self.fc1.forward(xs)?)?;
        self.fc2.forward(&h)
    }
}

struct ClipEncoderLayer {
    ln1_w: CudaTensor,
    ln1_b: CudaTensor,
    attn: ClipTextAttention,
    ln2_w: CudaTensor,
    ln2_b: CudaTensor,
    mlp: ClipTextMlp,
    eps: f32,
}

impl ClipEncoderLayer {
    fn zeros(cfg: &ClipTextConfig) -> Result<Self> {
        let d = cfg.hidden_size;
        Ok(Self {
            ln1_w: CudaTensor::ones(&[d]),
            ln1_b: CudaTensor::zeros(&[d]),
            attn: ClipTextAttention::zeros(d, cfg.num_attention_heads)?,
            ln2_w: CudaTensor::ones(&[d]),
            ln2_b: CudaTensor::zeros(&[d]),
            mlp: ClipTextMlp::zeros(d, cfg.intermediate_size)?,
            eps: cfg.layer_norm_eps,
        })
    }

    fn load(map: &WeightMap, prefix: &str, cfg: &ClipTextConfig) -> Result<Self> {
        let d = cfg.hidden_size;
        let key = |n: &str| weights::join_key(prefix, n);
        Ok(Self {
            ln1_w: weights::cuda_tensor_shaped(map, &key("layer_norm1.weight"), &[d])?,
            ln1_b: weights::cuda_tensor_shaped(map, &key("layer_norm1.bias"), &[d])?,
            attn: ClipTextAttention::load(map, &key("self_attn"), d, cfg.num_attention_heads)?,
            ln2_w: weights::cuda_tensor_shaped(map, &key("layer_norm2.weight"), &[d])?,
            ln2_b: weights::cuda_tensor_shaped(map, &key("layer_norm2.bias"), &[d])?,
            mlp: ClipTextMlp::load(map, &key("mlp"), d, cfg.intermediate_size)?,
            eps: cfg.layer_norm_eps,
        })
    }

    fn forward(&self, xs: &CudaTensor, mask: &CudaTensor) -> Result<CudaTensor> {
        let n = nn::layer_norm(xs, self.eps, Some(&self.ln1_w), Some(&self.ln1_b))?;
        let h = xs.add(&self.attn.forward(&n, mask)?)?;
        let n = nn::layer_norm(&h, self.eps, Some(&self.ln2_w), Some(&self.ln2_b))?;
        h.add(&self.mlp.forward(&n)?)
    }
}

/// CLIP text tower → pooled EOS embedding `[1, hidden]`.
pub struct ClipTextModel {
    pub cfg: ClipTextConfig,
    token_embed: CudaTensor,
    pos_embed: CudaTensor,
    layers: Vec<ClipEncoderLayer>,
    final_ln_w: CudaTensor,
    final_ln_b: CudaTensor,
}

impl ClipTextModel {
    pub fn zeros(cfg: ClipTextConfig) -> Result<Self> {
        let d = cfg.hidden_size;
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for _ in 0..cfg.num_hidden_layers {
            layers.push(ClipEncoderLayer::zeros(&cfg)?);
        }
        Ok(Self {
            token_embed: CudaTensor::zeros(&[cfg.vocab_size, d]),
            pos_embed: CudaTensor::zeros(&[cfg.max_position_embeddings, d]),
            layers,
            final_ln_w: CudaTensor::ones(&[d]),
            final_ln_b: CudaTensor::zeros(&[d]),
            cfg,
        })
    }

    pub fn load(map: &WeightMap, cfg: ClipTextConfig) -> Result<Self> {
        let d = cfg.hidden_size;
        let root = if map.contains("text_model.embeddings.token_embedding.weight") {
            "text_model."
        } else {
            ""
        };
        let te = format!("{root}embeddings.token_embedding.weight");
        let pe = format!("{root}embeddings.position_embedding.weight");
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            layers.push(ClipEncoderLayer::load(
                map,
                &format!("{root}encoder.layers.{i}"),
                &cfg,
            )?);
        }
        Ok(Self {
            token_embed: weights::cuda_tensor_shaped(map, &te, &[cfg.vocab_size, d])?,
            pos_embed: weights::cuda_tensor_shaped(map, &pe, &[cfg.max_position_embeddings, d])?,
            layers,
            final_ln_w: weights::cuda_tensor_shaped(
                map,
                &format!("{root}final_layer_norm.weight"),
                &[d],
            )?,
            final_ln_b: weights::cuda_tensor_shaped(
                map,
                &format!("{root}final_layer_norm.bias"),
                &[d],
            )?,
            cfg,
        })
    }

    /// Full last-layer hidden states `[1, S, D]` (pre-pool).
    pub fn encode_hidden(&self, input_ids: &[u32]) -> Result<CudaTensor> {
        let s = input_ids.len();
        if s == 0 || s > self.cfg.max_position_embeddings {
            return Err(msg(format!(
                "clip text: seq {s} vs max {}",
                self.cfg.max_position_embeddings
            )));
        }
        let idxs: Vec<usize> = input_ids.iter().map(|&i| i as usize).collect();
        let tok = self.token_embed.embedding_rows(&idxs)?;
        let pos_idxs: Vec<usize> = (0..s).collect();
        let pos = self.pos_embed.embedding_rows(&pos_idxs)?;
        let mut h = tok.add(&pos)?.unsqueeze(0)?; // [1,S,D]
        let mask = causal_mask(s)?;
        for layer in &self.layers {
            h = layer.forward(&h, &mask)?;
        }
        nn::layer_norm(
            &h,
            self.cfg.layer_norm_eps,
            Some(&self.final_ln_w),
            Some(&self.final_ln_b),
        )
    }

    /// `input_ids` length ≤ `max_position_embeddings` → pooled `[1, hidden]`.
    pub fn encode_pooled(&self, input_ids: &[u32]) -> Result<CudaTensor> {
        let h = self.encode_hidden(input_ids)?;
        let eos_pos = pool_eos_index(input_ids, self.cfg.eos_token_id);
        h.narrow(1, eos_pos, 1)?.squeeze(1) // [1, D]
    }
}

fn pool_eos_index(ids: &[u32], eos: u32) -> usize {
    if eos == 2 {
        ids.iter()
            .enumerate()
            .max_by_key(|(_, &v)| v)
            .map(|(i, _)| i)
            .unwrap_or(0)
    } else {
        ids.iter()
            .position(|&id| id == eos)
            .unwrap_or(ids.len().saturating_sub(1))
    }
}

pub fn encode_pooled_from_dir(root: &Path, prompt: &str) -> Result<CudaTensor> {
    let cfg = ClipTextConfig::vit_l_14();
    let ids = tokenize_clip(root, prompt, cfg.max_position_embeddings).map_err(msg)?;
    let map = WeightMap::open(&root.join("text_encoder_2")).map_err(|e| msg(e.to_string()))?;
    let model = ClipTextModel::load(&map, cfg)?;
    model.encode_pooled(&ids)
}

/// Encode CLIP-L sequence embeds from `text_encoder/` + `tokenizer/` (FLUX/SD3).
pub fn encode_hidden_from_dir(
    root: &Path,
    text_encoder_subdir: &str,
    tokenizer_subdir: &str,
    prompt: &str,
) -> Result<CudaTensor> {
    use fastvideo_models::kandinsky5::tokenize_clip_at;
    let cfg = ClipTextConfig::vit_l_14();
    let ids = tokenize_clip_at(root, tokenizer_subdir, prompt, cfg.max_position_embeddings)
        .map_err(msg)?;
    let map = WeightMap::open(&root.join(text_encoder_subdir)).map_err(|e| msg(e.to_string()))?;
    let model = ClipTextModel::load(&map, cfg)?;
    model.encode_hidden(&ids)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tiny_pooled_shape() {
        let m = ClipTextModel::zeros(ClipTextConfig::tiny()).unwrap();
        let ids = vec![1u32, 2, 0, 0, 0, 0, 0, 0];
        let p = m.encode_pooled(&ids).unwrap();
        assert_eq!(p.shape, vec![1, 32]);
    }

    #[test]
    fn eos_pool_legacy() {
        assert_eq!(pool_eos_index(&[1, 5, 3, 0], 2), 1);
        assert_eq!(pool_eos_index(&[49406, 100, 49407, 0], 49407), 2);
    }
}
