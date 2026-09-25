//! CLIP-L + T5-XXL on `CudaTensor` (mirrors `fastvideo_models::flux::text`).

use fastvideo_models::flux::{flux1_dummy_text, flux1_t5_len, ClipTextConfig, T5Config};
pub use fastvideo_models::flux::{pad_token_ids, tokenize_flux1};

use crate::wan::nn::{self, Linear};
use crate::wan::tensor::{CudaTensor, Result};
use crate::wan::weights::{self, WeightMap};

pub fn dummy_text() -> bool {
    flux1_dummy_text()
}

pub fn t5_len(default: usize) -> usize {
    flux1_t5_len(default)
}

fn quick_gelu(xs: &CudaTensor) -> CudaTensor {
    // x * sigmoid(1.702 x) = silu(1.702 x) / 1.702
    xs.mul_scalar(1.702).silu().mul_scalar(1.0 / 1.702)
}

fn causal_mask(seq: usize) -> Result<CudaTensor> {
    let mut data = vec![0.0f32; seq * seq];
    for q in 0..seq {
        for k in 0..seq {
            if k > q {
                data[q * seq + k] = -1e9;
            }
        }
    }
    CudaTensor::from_vec(data, vec![1, 1, seq, seq])
}

struct ClipAttn {
    q: Linear,
    k: Linear,
    v: Linear,
    out: Linear,
    heads: usize,
    dim_head: usize,
}

impl ClipAttn {
    fn zeros(dim: usize, heads: usize) -> Self {
        Self {
            q: Linear::zeros(dim, dim, true),
            k: Linear::zeros(dim, dim, true),
            v: Linear::zeros(dim, dim, true),
            out: Linear::zeros(dim, dim, true),
            heads,
            dim_head: dim / heads,
        }
    }

    fn load(map: &WeightMap, prefix: &str, dim: usize, heads: usize) -> Result<Self> {
        Ok(Self {
            q: Linear::load(map, &format!("{prefix}.q_proj"), dim, dim, true)?,
            k: Linear::load(map, &format!("{prefix}.k_proj"), dim, dim, true)?,
            v: Linear::load(map, &format!("{prefix}.v_proj"), dim, dim, true)?,
            out: Linear::load(map, &format!("{prefix}.out_proj"), dim, dim, true)?,
            heads,
            dim_head: dim / heads,
        })
    }

    fn forward(&self, xs: &CudaTensor, mask: &CudaTensor) -> Result<CudaTensor> {
        let b = xs.shape[0];
        let s = xs.shape[1];
        let q = self.q.forward(xs)?.reshape(vec![b, s, self.heads, self.dim_head])?.transpose(1, 2)?;
        let k = self.k.forward(xs)?.reshape(vec![b, s, self.heads, self.dim_head])?.transpose(1, 2)?;
        let v = self.v.forward(xs)?.reshape(vec![b, s, self.heads, self.dim_head])?.transpose(1, 2)?;
        let attn = nn::scaled_dot_product_attention_masked(&q, &k, &v, None, Some(mask))?;
        let attn = attn.transpose(1, 2)?.reshape(vec![b, s, self.heads * self.dim_head])?;
        self.out.forward(&attn)
    }
}

struct ClipLayer {
    attn: ClipAttn,
    ln1_w: CudaTensor,
    ln1_b: CudaTensor,
    fc1: Linear,
    fc2: Linear,
    ln2_w: CudaTensor,
    ln2_b: CudaTensor,
    eps: f32,
}

impl ClipLayer {
    fn zeros(cfg: &ClipTextConfig) -> Self {
        let dim = cfg.hidden_size;
        Self {
            attn: ClipAttn::zeros(dim, cfg.num_attention_heads),
            ln1_w: CudaTensor::from_vec(vec![1.0; dim], vec![dim]).expect("ln"),
            ln1_b: CudaTensor::zeros(&[dim]),
            fc1: Linear::zeros(dim, cfg.intermediate_size, true),
            fc2: Linear::zeros(cfg.intermediate_size, dim, true),
            ln2_w: CudaTensor::from_vec(vec![1.0; dim], vec![dim]).expect("ln"),
            ln2_b: CudaTensor::zeros(&[dim]),
            eps: cfg.layer_norm_eps as f32,
        }
    }

    fn load(map: &WeightMap, prefix: &str, cfg: &ClipTextConfig) -> Result<Self> {
        let dim = cfg.hidden_size;
        Ok(Self {
            attn: ClipAttn::load(map, &format!("{prefix}.self_attn"), dim, cfg.num_attention_heads)?,
            ln1_w: weights::cuda_tensor_shaped(map, &format!("{prefix}.layer_norm1.weight"), &[dim])?,
            ln1_b: weights::cuda_tensor_shaped(map, &format!("{prefix}.layer_norm1.bias"), &[dim])?,
            fc1: Linear::load(map, &format!("{prefix}.mlp.fc1"), dim, cfg.intermediate_size, true)?,
            fc2: Linear::load(map, &format!("{prefix}.mlp.fc2"), cfg.intermediate_size, dim, true)?,
            ln2_w: weights::cuda_tensor_shaped(map, &format!("{prefix}.layer_norm2.weight"), &[dim])?,
            ln2_b: weights::cuda_tensor_shaped(map, &format!("{prefix}.layer_norm2.bias"), &[dim])?,
            eps: cfg.layer_norm_eps as f32,
        })
    }

    fn forward(&self, xs: &CudaTensor, mask: &CudaTensor) -> Result<CudaTensor> {
        let n1 = xs.layer_norm(self.eps, Some(&self.ln1_w), Some(&self.ln1_b))?;
        let xs = xs.add(&self.attn.forward(&n1, mask)?)?;
        let n2 = xs.layer_norm(self.eps, Some(&self.ln2_w), Some(&self.ln2_b))?;
        xs.add(&self.fc2.forward(&quick_gelu(&self.fc1.forward(&n2)?))?)
    }
}

pub struct ClipTextEncoder {
    pub cfg: ClipTextConfig,
    token_emb: CudaTensor,
    pos_emb: CudaTensor,
    layers: Vec<ClipLayer>,
    final_ln_w: CudaTensor,
    final_ln_b: CudaTensor,
}

impl ClipTextEncoder {
    pub fn zeros(cfg: ClipTextConfig) -> Self {
        let dim = cfg.hidden_size;
        let layers = (0..cfg.num_hidden_layers).map(|_| ClipLayer::zeros(&cfg)).collect();
        Self {
            token_emb: CudaTensor::zeros(&[cfg.vocab_size, dim]),
            pos_emb: CudaTensor::zeros(&[cfg.max_position_embeddings, dim]),
            final_ln_w: CudaTensor::from_vec(vec![1.0; dim], vec![dim]).expect("ln"),
            final_ln_b: CudaTensor::zeros(&[dim]),
            layers,
            cfg,
        }
    }

    pub fn load(cfg: ClipTextConfig, map: &WeightMap) -> Result<Self> {
        let prefix = if map.contains("text_model.embeddings.token_embedding.weight") {
            "text_model"
        } else {
            ""
        };
        let te = if prefix.is_empty() {
            "embeddings.token_embedding.weight".into()
        } else {
            format!("{prefix}.embeddings.token_embedding.weight")
        };
        let pe = if prefix.is_empty() {
            "embeddings.position_embedding.weight".into()
        } else {
            format!("{prefix}.embeddings.position_embedding.weight")
        };
        let enc = if prefix.is_empty() {
            "encoder.layers".into()
        } else {
            format!("{prefix}.encoder.layers")
        };
        let fln = if prefix.is_empty() {
            "final_layer_norm".into()
        } else {
            format!("{prefix}.final_layer_norm")
        };
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            layers.push(ClipLayer::load(map, &format!("{enc}.{i}"), &cfg)?);
        }
        Ok(Self {
            token_emb: weights::cuda_tensor_shaped(map, &te, &[cfg.vocab_size, cfg.hidden_size])?,
            pos_emb: weights::cuda_tensor_shaped(
                map,
                &pe,
                &[cfg.max_position_embeddings, cfg.hidden_size],
            )?,
            final_ln_w: weights::cuda_tensor_shaped(map, &format!("{fln}.weight"), &[cfg.hidden_size])?,
            final_ln_b: weights::cuda_tensor_shaped(map, &format!("{fln}.bias"), &[cfg.hidden_size])?,
            layers,
            cfg,
        })
    }

    pub fn forward(&self, ids: &[u32]) -> Result<(CudaTensor, CudaTensor)> {
        let s = ids.len().min(self.cfg.max_position_embeddings).max(1);
        let ids = &ids[..s];
        let table = self.token_emb.host_cow()?;
        let pos = self.pos_emb.host_cow()?;
        let dim = self.cfg.hidden_size;
        let mut hidden = vec![0.0f32; s * dim];
        for (i, &id) in ids.iter().enumerate() {
            let row = (id as usize).min(self.cfg.vocab_size.saturating_sub(1));
            for d in 0..dim {
                hidden[i * dim + d] = table[row * dim + d] + pos[i * dim + d];
            }
        }
        let mut x = CudaTensor::from_vec(hidden, vec![1, s, dim])?;
        let mask = causal_mask(s)?;
        for layer in &self.layers {
            x = layer.forward(&x, &mask)?;
        }
        x = x.layer_norm(self.cfg.layer_norm_eps as f32, Some(&self.final_ln_w), Some(&self.final_ln_b))?;
        let eos = ids
            .iter()
            .enumerate()
            .max_by_key(|(_, t)| *t)
            .map(|(i, _)| i)
            .unwrap_or(0);
        let pooled = x.narrow(1, eos, 1)?.reshape(vec![1, dim])?;
        Ok((x, pooled))
    }
}

struct T5Attn {
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    n_heads: usize,
    d_kv: usize,
}

impl T5Attn {
    fn zeros(cfg: &T5Config) -> Self {
        let inner = cfg.inner.num_heads * cfg.inner.d_kv;
        Self {
            q: Linear::zeros(cfg.inner.d_model, inner, false),
            k: Linear::zeros(cfg.inner.d_model, inner, false),
            v: Linear::zeros(cfg.inner.d_model, inner, false),
            o: Linear::zeros(inner, cfg.inner.d_model, false),
            n_heads: cfg.inner.num_heads,
            d_kv: cfg.inner.d_kv,
        }
    }

    fn load(map: &WeightMap, prefix: &str, cfg: &T5Config) -> Result<Self> {
        let inner = cfg.inner.num_heads * cfg.inner.d_kv;
        Ok(Self {
            q: Linear::load(map, &format!("{prefix}.q"), cfg.inner.d_model, inner, false)?,
            k: Linear::load(map, &format!("{prefix}.k"), cfg.inner.d_model, inner, false)?,
            v: Linear::load(map, &format!("{prefix}.v"), cfg.inner.d_model, inner, false)?,
            o: Linear::load(map, &format!("{prefix}.o"), inner, cfg.inner.d_model, false)?,
            n_heads: cfg.inner.num_heads,
            d_kv: cfg.inner.d_kv,
        })
    }

    fn forward(&self, xs: &CudaTensor, bias: &CudaTensor) -> Result<CudaTensor> {
        let b = xs.shape[0];
        let s = xs.shape[1];
        let q = self.q.forward(xs)?.reshape(vec![b, s, self.n_heads, self.d_kv])?.transpose(1, 2)?;
        let k = self.k.forward(xs)?.reshape(vec![b, s, self.n_heads, self.d_kv])?.transpose(1, 2)?;
        let v = self.v.forward(xs)?.reshape(vec![b, s, self.n_heads, self.d_kv])?.transpose(1, 2)?;
        let scores = q.matmul(&k.transpose(2, 3)?)?;
        let scores = scores.add(bias)?;
        let attn = scores.softmax(-1)?;
        let ctx = attn.matmul(&v)?.transpose(1, 2)?.reshape(vec![b, s, self.n_heads * self.d_kv])?;
        self.o.forward(&ctx)
    }
}

struct T5Layer {
    attn: T5Attn,
    wi_0: Linear,
    wi_1: Linear,
    wo: Linear,
    ln1: CudaTensor,
    ln2: CudaTensor,
    eps: f32,
}

impl T5Layer {
    fn zeros(cfg: &T5Config) -> Self {
        let d = cfg.inner.d_model;
        Self {
            attn: T5Attn::zeros(cfg),
            wi_0: Linear::zeros(d, cfg.inner.d_ff, false),
            wi_1: Linear::zeros(d, cfg.inner.d_ff, false),
            wo: Linear::zeros(cfg.inner.d_ff, d, false),
            ln1: CudaTensor::from_vec(vec![1.0; d], vec![d]).expect("ln"),
            ln2: CudaTensor::from_vec(vec![1.0; d], vec![d]).expect("ln"),
            eps: cfg.inner.eps as f32,
        }
    }

    fn load(map: &WeightMap, prefix: &str, cfg: &T5Config) -> Result<Self> {
        let d = cfg.inner.d_model;
        Ok(Self {
            attn: T5Attn::load(map, &format!("{prefix}.layer.0.SelfAttention"), cfg)?,
            wi_0: Linear::load(
                map,
                &format!("{prefix}.layer.1.DenseReluDense.wi_0"),
                d,
                cfg.inner.d_ff,
                false,
            )?,
            wi_1: Linear::load(
                map,
                &format!("{prefix}.layer.1.DenseReluDense.wi_1"),
                d,
                cfg.inner.d_ff,
                false,
            )?,
            wo: Linear::load(
                map,
                &format!("{prefix}.layer.1.DenseReluDense.wo"),
                cfg.inner.d_ff,
                d,
                false,
            )?,
            ln1: weights::cuda_tensor_shaped(map, &format!("{prefix}.layer.0.layer_norm.weight"), &[d])?,
            ln2: weights::cuda_tensor_shaped(map, &format!("{prefix}.layer.1.layer_norm.weight"), &[d])?,
            eps: cfg.inner.eps as f32,
        })
    }

    fn forward(&self, xs: &CudaTensor, bias: &CudaTensor) -> Result<CudaTensor> {
        let n1 = xs.rms_norm(&self.ln1, self.eps)?;
        let xs = xs.add(&self.attn.forward(&n1, bias)?)?;
        let n2 = xs.rms_norm(&self.ln2, self.eps)?;
        let gelu = nn::gelu_tanh(&self.wi_0.forward(&n2)?);
        let lin = self.wi_1.forward(&n2)?;
        xs.add(&self.wo.forward(&gelu.mul(&lin)?)?)
    }
}

pub struct T5Encoder {
    pub cfg: T5Config,
    embed: CudaTensor,
    layers: Vec<T5Layer>,
    relative_bias: CudaTensor,
    final_ln: CudaTensor,
}

impl T5Encoder {
    pub fn zeros(cfg: T5Config) -> Self {
        let d = cfg.inner.d_model;
        let layers = (0..cfg.inner.num_layers).map(|_| T5Layer::zeros(&cfg)).collect();
        Self {
            embed: CudaTensor::zeros(&[cfg.inner.vocab_size, d]),
            relative_bias: CudaTensor::zeros(&[cfg.inner.relative_attention_num_buckets, cfg.inner.num_heads]),
            final_ln: CudaTensor::from_vec(vec![1.0; d], vec![d]).expect("ln"),
            layers,
            cfg,
        }
    }

    pub fn load(cfg: T5Config, map: &WeightMap) -> Result<Self> {
        let d = cfg.inner.d_model;
        let embed = weights::cuda_tensor_shaped(map, "shared.weight", &[cfg.inner.vocab_size, d])
            .or_else(|_| {
                weights::cuda_tensor_shaped(map, "encoder.embed_tokens.weight", &[cfg.inner.vocab_size, d])
            })?;
        let relative_bias = weights::cuda_tensor_shaped(
            map,
            "encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight",
            &[cfg.inner.relative_attention_num_buckets, cfg.inner.num_heads],
        )?;
        let mut layers = Vec::with_capacity(cfg.inner.num_layers);
        for i in 0..cfg.inner.num_layers {
            layers.push(T5Layer::load(map, &format!("encoder.block.{i}"), &cfg)?);
        }
        Ok(Self {
            final_ln: weights::cuda_tensor_shaped(map, "encoder.final_layer_norm.weight", &[d])?,
            embed,
            layers,
            relative_bias,
            cfg,
        })
    }

    pub fn forward(&self, ids: &[u32]) -> Result<CudaTensor> {
        let s = ids.len().max(1);
        let table = self.embed.host_cow()?;
        let d = self.cfg.inner.d_model;
        let mut hidden = vec![0.0f32; s * d];
        for (i, &id) in ids.iter().enumerate().take(s) {
            let row = (id as usize).min(self.cfg.inner.vocab_size.saturating_sub(1));
            hidden[i * d..(i + 1) * d].copy_from_slice(&table[row * d..(row + 1) * d]);
        }
        let mut x = CudaTensor::from_vec(hidden, vec![1, s, d])?;
        let bias = t5_rel_bias(
            s,
            self.cfg.inner.relative_attention_num_buckets,
            self.cfg.inner.relative_attention_max_distance,
            self.cfg.inner.num_heads,
            &self.relative_bias,
        )?;
        for layer in &self.layers {
            x = layer.forward(&x, &bias)?;
        }
        x.rms_norm(&self.final_ln, self.cfg.inner.eps as f32)
    }
}

fn t5_rel_bias(
    seq: usize,
    num_buckets: usize,
    max_distance: usize,
    n_heads: usize,
    table: &CudaTensor,
) -> Result<CudaTensor> {
    let host = table.host_cow()?;
    let mut buckets = vec![0usize; seq * seq];
    let num_buckets = num_buckets as i64;
    let max_exact = num_buckets / 4;
    let max_distance = max_distance as i64;
    for i in 0..seq as i64 {
        for j in 0..seq as i64 {
            let mut relative = j - i;
            let mut bucket = 0i64;
            let n_buckets = num_buckets / 2;
            if relative < 0 {
                bucket += n_buckets;
                relative = -relative;
            }
            let is_small = relative < max_exact;
            let relative_log = ((relative as f64 / max_exact as f64).ln()
                / (max_distance as f64 / max_exact as f64).ln()
                * (n_buckets - max_exact) as f64)
                .floor() as i64
                + max_exact;
            let relative_bucket = if is_small {
                relative
            } else {
                relative_log.min(n_buckets - 1)
            };
            buckets[(i as usize) * seq + j as usize] = (bucket + relative_bucket) as usize;
        }
    }
    let mut out = vec![0.0f32; n_heads * seq * seq];
    for (pos, &bkt) in buckets.iter().enumerate() {
        for h in 0..n_heads {
            out[h * seq * seq + pos] = host[bkt * n_heads + h];
        }
    }
    CudaTensor::from_vec(out, vec![1, n_heads, seq, seq])
}

pub struct Flux1TextEncoder {
    clip: Option<ClipTextEncoder>,
    t5: Option<T5Encoder>,
    dummy_joint: usize,
    dummy_pooled: usize,
}

impl Flux1TextEncoder {
    pub fn dummy(joint: usize, pooled: usize) -> Self {
        Self {
            clip: None,
            t5: None,
            dummy_joint: joint,
            dummy_pooled: pooled,
        }
    }

    pub fn zeros_tiny() -> Self {
        Self {
            clip: Some(ClipTextEncoder::zeros(ClipTextConfig::tiny())),
            t5: Some(T5Encoder::zeros(T5Config::tiny())),
            dummy_joint: 16,
            dummy_pooled: 8,
        }
    }

    pub fn load(root: &std::path::Path, joint: usize, pooled: usize, t5_max: usize) -> Result<Self> {
        if dummy_text() {
            return Ok(Self::dummy(joint, pooled));
        }
        let clip = WeightMap::from_dir(&root.join("text_encoder"))
            .ok()
            .and_then(|map| ClipTextEncoder::load(ClipTextConfig::clip_l(), &map).ok());
        let mut t5_cfg = T5Config::xxl();
        t5_cfg.text_len = t5_max.max(1);
        let t5 = WeightMap::from_dir(&root.join("text_encoder_2"))
            .ok()
            .and_then(|map| T5Encoder::load(t5_cfg, &map).ok());
        if clip.is_none() && t5.is_none() {
            return Ok(Self::dummy(joint, pooled));
        }
        Ok(Self {
            clip,
            t5,
            dummy_joint: joint,
            dummy_pooled: pooled,
        })
    }

    pub fn clip_len(&self) -> usize {
        self.clip.as_ref().map(|c| c.cfg.text_len).unwrap_or(77)
    }

    pub fn t5_len(&self) -> usize {
        self.t5.as_ref().map(|c| c.cfg.text_len).unwrap_or(512)
    }

    pub fn encode(&self, clip_ids: &[u32], t5_ids: &[u32]) -> Result<(CudaTensor, CudaTensor)> {
        if dummy_text() || (self.clip.is_none() && self.t5.is_none()) {
            return dummy_encode(clip_ids, t5_ids, self.dummy_joint, self.dummy_pooled);
        }
        let pooled = if let Some(clip) = &self.clip {
            clip.forward(clip_ids)?.1
        } else {
            dummy_encode(clip_ids, t5_ids, self.dummy_joint, self.dummy_pooled)?.1
        };
        let tokens = if let Some(t5) = &self.t5 {
            t5.forward(t5_ids)?
        } else {
            dummy_encode(clip_ids, t5_ids, self.dummy_joint, self.dummy_pooled)?.0
        };
        Ok((tokens, pooled))
    }
}

fn dummy_encode(
    clip_ids: &[u32],
    t5_ids: &[u32],
    joint: usize,
    pooled: usize,
) -> Result<(CudaTensor, CudaTensor)> {
    let seq = t5_ids.len().max(1);
    let mut tok = vec![0.0f32; seq * joint];
    for (i, v) in tok.iter_mut().enumerate() {
        let id = t5_ids.get(i % t5_ids.len().max(1)).copied().unwrap_or(0) as f32;
        *v = ((id * 0.01) + (i as f32) * 0.001) % 1.0;
    }
    let mut pool = vec![0.0f32; pooled];
    for (i, v) in pool.iter_mut().enumerate() {
        let id = clip_ids.get(i % clip_ids.len().max(1)).copied().unwrap_or(0) as f32;
        *v = ((id * 0.01) + (i as f32) * 0.001) % 1.0;
    }
    Ok((
        CudaTensor::from_vec(tok, vec![1, seq, joint])?,
        CudaTensor::from_vec(pool, vec![1, pooled])?,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tiny_clip_and_t5_shapes() {
        let clip = ClipTextEncoder::zeros(ClipTextConfig::tiny());
        let (_h, p) = clip.forward(&[1, 2, 3, 2]).unwrap();
        assert_eq!(p.shape, vec![1, 16]);
        let t5 = T5Encoder::zeros(T5Config::tiny());
        let out = t5.forward(&[1, 2, 3, 4]).unwrap();
        assert_eq!(out.shape, vec![1, 4, 16]);
    }
}
