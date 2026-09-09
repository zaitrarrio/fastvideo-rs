//! UMT5 encoder: T5 encoder with per-layer relative attention bias.

use candle_core::{DType, Device, Result, Tensor, D};
use candle_nn::VarBuilder;

use crate::nn::Linear;

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
}

fn t5_layer_norm(xs: &Tensor, weight: &Tensor, eps: f64) -> Result<Tensor> {
    let x = xs.to_dtype(DType::F32)?;
    let mean_sq = x.sqr()?.mean_keepdim(D::Minus1)?;
    let y = x.broadcast_div(&(mean_sq + eps)?.sqrt()?)?;
    y.to_dtype(xs.dtype())?.broadcast_mul(&weight.to_dtype(xs.dtype())?)
}

fn relative_position_bucket(
    seq_len: usize,
    num_buckets: usize,
    max_distance: usize,
    device: &Device,
) -> Result<Tensor> {
    // Encoder bidirectional buckets (Hugging Face T5).
    let mut buckets = vec![0i64; seq_len * seq_len];
    let num_buckets = num_buckets as i64;
    let max_exact = num_buckets / 4;
    let max_distance = max_distance as i64;
    for i in 0..seq_len as i64 {
        for j in 0..seq_len as i64 {
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
            buckets[(i as usize) * seq_len + j as usize] = bucket + relative_bucket;
        }
    }
    let buckets: Vec<u32> = buckets.into_iter().map(|b| b as u32).collect();
    Tensor::from_vec(buckets, (seq_len, seq_len), device)
}

#[derive(Debug, Clone)]
struct DenseGated {
    wi_0: Linear,
    wi_1: Linear,
    wo: Linear,
}

impl DenseGated {
    fn load(cfg: &Umt5Config, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            wi_0: Linear::load(cfg.d_model, cfg.d_ff, vb.pp("wi_0"))?,
            wi_1: Linear::load(cfg.d_model, cfg.d_ff, vb.pp("wi_1"))?,
            wo: Linear::load(cfg.d_ff, cfg.d_model, vb.pp("wo"))?,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let dtype = xs.dtype();
        let gelu = crate::nn::gelu_tanh(&self.wi_0.forward(xs)?)?.to_dtype(DType::F32)?;
        let linear = self.wi_1.forward(xs)?.to_dtype(DType::F32)?;
        self.wo.forward(&(gelu * linear)?.to_dtype(dtype)?)
    }
}

#[derive(Debug, Clone)]
struct SelfAttention {
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    relative_bias: Tensor,
    n_heads: usize,
    d_kv: usize,
}

impl SelfAttention {
    fn load(cfg: &Umt5Config, vb: VarBuilder) -> Result<Self> {
        let inner = cfg.num_heads * cfg.d_kv;
        Ok(Self {
            q: Linear::load(cfg.d_model, inner, vb.pp("q"))?,
            k: Linear::load(cfg.d_model, inner, vb.pp("k"))?,
            v: Linear::load(cfg.d_model, inner, vb.pp("v"))?,
            o: Linear::load(inner, cfg.d_model, vb.pp("o"))?,
            relative_bias: vb.get(
                (cfg.relative_attention_num_buckets, cfg.num_heads),
                "relative_attention_bias.weight",
            )?,
            n_heads: cfg.num_heads,
            d_kv: cfg.d_kv,
        })
    }

    fn forward(&self, xs: &Tensor, mask: Option<&Tensor>, buckets: &Tensor) -> Result<Tensor> {
        let (b, s, _) = xs.dims3()?;
        let q = self.q.forward(xs)?.reshape((b, s, self.n_heads, self.d_kv))?.transpose(1, 2)?;
        let k = self.k.forward(xs)?.reshape((b, s, self.n_heads, self.d_kv))?.transpose(1, 2)?;
        let v = self.v.forward(xs)?.reshape((b, s, self.n_heads, self.d_kv))?.transpose(1, 2)?;
        let q = q.contiguous()?.to_dtype(DType::F32)?;
        let k = k.contiguous()?.to_dtype(DType::F32)?;
        let v = v.contiguous()?.to_dtype(DType::F32)?;
        let scores = q.matmul(&k.transpose(D::Minus1, D::Minus2)?)?;
        // relative bias: embedding lookup [S,S] -> [H,S,S]
        let bias = self
            .relative_bias
            .index_select(&buckets.flatten_all()?, 0)?;
        let bias = bias
            .reshape((s, s, self.n_heads))?
            .permute((2, 0, 1))?
            .unsqueeze(0)?
            .to_dtype(DType::F32)?;
        let mut scores = scores.broadcast_add(&bias)?;
        if let Some(mask) = mask {
            let neg = Tensor::new(f32::NEG_INFINITY, xs.device())?.broadcast_as(scores.shape())?;
            let keep = Tensor::new(0f32, xs.device())?.broadcast_as(scores.shape())?;
            let m = mask.to_dtype(DType::F32)?.broadcast_as(scores.shape())?;
            scores = m.where_cond(&keep, &neg)?;
        }
        let attn = candle_nn::ops::softmax_last_dim(&scores)?;
        let ctx = attn.matmul(&v)?.transpose(1, 2)?.contiguous()?.reshape((b, s, self.n_heads * self.d_kv))?;
        self.o.forward(&ctx.to_dtype(xs.dtype())?)
    }
}

#[derive(Debug, Clone)]
struct EncoderLayer {
    attn: SelfAttention,
    ff: DenseGated,
    ln1: Tensor,
    ln2: Tensor,
    eps: f64,
}

impl EncoderLayer {
    fn load(cfg: &Umt5Config, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            attn: SelfAttention::load(cfg, vb.pp("layer").pp("0").pp("SelfAttention"))?,
            ln1: vb.pp("layer").pp("0").pp("layer_norm").get(cfg.d_model, "weight")?,
            ff: DenseGated::load(cfg, vb.pp("layer").pp("1").pp("DenseReluDense"))?,
            ln2: vb.pp("layer").pp("1").pp("layer_norm").get(cfg.d_model, "weight")?,
            eps: cfg.eps,
        })
    }

    fn forward(&self, xs: &Tensor, mask: Option<&Tensor>, buckets: &Tensor) -> Result<Tensor> {
        let normed = t5_layer_norm(xs, &self.ln1, self.eps)?;
        let xs = (xs + self.attn.forward(&normed, mask, buckets)?)?;
        let normed = t5_layer_norm(&xs, &self.ln2, self.eps)?;
        xs + self.ff.forward(&normed)?
    }
}

#[derive(Debug, Clone)]
pub struct Umt5Encoder {
    pub cfg: Umt5Config,
    embed: Tensor,
    layers: Vec<EncoderLayer>,
    final_ln: Tensor,
}

impl Umt5Encoder {
    pub fn load(cfg: Umt5Config, vb: VarBuilder) -> Result<Self> {
        let embed = vb
            .pp("encoder")
            .pp("embed_tokens")
            .get((cfg.vocab_size, cfg.d_model), "weight")
            .or_else(|_| vb.pp("shared").get((cfg.vocab_size, cfg.d_model), "weight"))?;
        let mut layers = Vec::with_capacity(cfg.num_layers);
        for i in 0..cfg.num_layers {
            layers.push(EncoderLayer::load(
                &cfg,
                vb.pp("encoder").pp("block").pp(&i.to_string()),
            )?);
        }
        Ok(Self {
            final_ln: vb.pp("encoder").pp("final_layer_norm").get(cfg.d_model, "weight")?,
            cfg,
            embed,
            layers,
        })
    }

    pub fn forward(&self, input_ids: &Tensor, attention_mask: Option<&Tensor>) -> Result<Tensor> {
        let (_b, s) = input_ids.dims2()?;
        let mut hidden = self
            .embed
            .index_select(&input_ids.flatten_all()?, 0)?
            .reshape((input_ids.dim(0)?, s, self.cfg.d_model))?;
        let buckets = relative_position_bucket(
            s,
            self.cfg.relative_attention_num_buckets,
            self.cfg.relative_attention_max_distance,
            input_ids.device(),
        )?;
        for layer in &self.layers {
            hidden = layer.forward(&hidden, attention_mask, &buckets)?;
        }
        t5_layer_norm(&hidden, &self.final_ln, self.cfg.eps)
    }
}

/// Trim true token length then right-pad to `text_len` (FastVideo T5 postprocess).
pub fn pad_prompt_embeds(embeds: &Tensor, seq_lens: &[usize], text_len: usize) -> Result<Tensor> {
    let (b, _s, d) = embeds.dims3()?;
    let device = embeds.device();
    let mut rows = Vec::with_capacity(b);
    for (i, &len) in seq_lens.iter().enumerate() {
        let len = len.min(text_len);
        let row = embeds.narrow(0, i, 1)?.narrow(1, 0, len)?;
        let pad = Tensor::zeros((1, text_len - len, d), embeds.dtype(), device)?;
        rows.push(Tensor::cat(&[&row, &pad], 1)?);
    }
    Tensor::cat(&rows, 0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tiny_umt5_forward_shape() {
        let device = Device::Cpu;
        let cfg = Umt5Config::tiny();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let enc = Umt5Encoder::load(cfg.clone(), vb).unwrap();
        let ids = Tensor::zeros((1, 6), DType::U32, &device).unwrap();
        let out = enc.forward(&ids, None).unwrap();
        assert_eq!(out.dims(), &[1, 6, 16]);
        let padded = pad_prompt_embeds(&out, &[4], 8).unwrap();
        assert_eq!(padded.dims(), &[1, 8, 16]);
    }
}
