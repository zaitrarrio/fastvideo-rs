//! FLUX.1 text: CLIP-L pooled + T5-XXL tokens.
//!
//! Diffusers / FastVideo: `CLIPTextModel` (77 tokens, pooled EOS) and
//! `T5EncoderModel` (T5-v1.1-XXL, last hidden). Tiny / dummy embeds stay
//! available for CI.

use candle_core::{DType, Device, IndexOp, Result, Tensor};
use candle_nn::VarBuilder;

use crate::nn::{self, Linear};
use crate::wan::umt5::Umt5Config;

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

#[derive(Debug, Clone, PartialEq)]
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

pub fn tokenize_flux1(path: &str, prompt: &str, max_len: usize) -> Result<(Vec<u32>, usize)> {
    crate::flux2::tokenize_flux2(path, prompt, max_len)
}

fn t5_layer_norm(xs: &Tensor, weight: &Tensor, eps: f64) -> Result<Tensor> {
    let x = xs.to_dtype(DType::F32)?;
    let mean_sq = x.sqr()?.mean_keepdim(D::Minus1)?;
    let y = x.broadcast_div(&(mean_sq + eps)?.sqrt()?)?;
    y.to_dtype(xs.dtype())?
        .broadcast_mul(&weight.to_dtype(xs.dtype())?)
}

fn relative_position_bucket(
    seq_len: usize,
    num_buckets: usize,
    max_distance: usize,
    device: &Device,
) -> Result<Tensor> {
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
        let gelu = nn::gelu_tanh(&self.wi_0.forward(xs)?)?.to_dtype(DType::F32)?;
        let linear = self.wi_1.forward(xs)?.to_dtype(DType::F32)?;
        self.wo.forward(&(gelu * linear)?.to_dtype(dtype)?)
    }
}

struct T5Attention {
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    n_heads: usize,
    d_kv: usize,
}

impl T5Attention {
    fn load(cfg: &Umt5Config, vb: VarBuilder) -> Result<Self> {
        let inner = cfg.num_heads * cfg.d_kv;
        Ok(Self {
            q: Linear::load(cfg.d_model, inner, vb.pp("q"))?,
            k: Linear::load(cfg.d_model, inner, vb.pp("k"))?,
            v: Linear::load(cfg.d_model, inner, vb.pp("v"))?,
            o: Linear::load(inner, cfg.d_model, vb.pp("o"))?,
            n_heads: cfg.num_heads,
            d_kv: cfg.d_kv,
        })
    }

    fn forward(&self, xs: &Tensor, bias: &Tensor) -> Result<Tensor> {
        let (b, s, _) = xs.dims3()?;
        let q = self
            .q
            .forward(xs)?
            .reshape((b, s, self.n_heads, self.d_kv))?
            .transpose(1, 2)?;
        let k = self
            .k
            .forward(xs)?
            .reshape((b, s, self.n_heads, self.d_kv))?
            .transpose(1, 2)?;
        let v = self
            .v
            .forward(xs)?
            .reshape((b, s, self.n_heads, self.d_kv))?
            .transpose(1, 2)?;
        let q = q.contiguous()?.to_dtype(DType::F32)?;
        let k = k.contiguous()?.to_dtype(DType::F32)?;
        let v = v.contiguous()?.to_dtype(DType::F32)?;
        let scores = q.matmul(&k.transpose(candle_core::D::Minus1, candle_core::D::Minus2)?)?;
        let scores = scores.broadcast_add(&bias.to_dtype(DType::F32)?)?;
        let attn = candle_nn::ops::softmax_last_dim(&scores)?;
        let ctx = attn
            .matmul(&v)?
            .transpose(1, 2)?
            .contiguous()?
            .reshape((b, s, self.n_heads * self.d_kv))?;
        self.o.forward(&ctx.to_dtype(xs.dtype())?)
    }
}

struct T5Layer {
    attn: T5Attention,
    ff: DenseGated,
    ln1: Tensor,
    ln2: Tensor,
    eps: f64,
}

impl T5Layer {
    fn load(cfg: &Umt5Config, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            attn: T5Attention::load(cfg, vb.pp("layer").pp("0").pp("SelfAttention"))?,
            ln1: vb
                .pp("layer")
                .pp("0")
                .pp("layer_norm")
                .get(cfg.d_model, "weight")?,
            ff: DenseGated::load(cfg, vb.pp("layer").pp("1").pp("DenseReluDense"))?,
            ln2: vb
                .pp("layer")
                .pp("1")
                .pp("layer_norm")
                .get(cfg.d_model, "weight")?,
            eps: cfg.eps,
        })
    }

    fn forward(&self, xs: &Tensor, bias: &Tensor) -> Result<Tensor> {
        let normed = t5_layer_norm(xs, &self.ln1, self.eps)?;
        let xs = (xs + self.attn.forward(&normed, bias)?)?;
        let normed = t5_layer_norm(&xs, &self.ln2, self.eps)?;
        xs + self.ff.forward(&normed)?
    }
}

/// T5 encoder that shares `relative_attention_bias` from block 0 (HF T5).
pub struct T5Encoder {
    pub cfg: T5Config,
    embed: Tensor,
    layers: Vec<T5Layer>,
    relative_bias: Tensor,
    final_ln: Tensor,
}

impl T5Encoder {
    pub fn load(cfg: T5Config, vb: VarBuilder) -> Result<Self> {
        let inner = &cfg.inner;
        let embed = vb
            .pp("shared")
            .get((inner.vocab_size, inner.d_model), "weight")
            .or_else(|_| {
                vb.pp("encoder")
                    .pp("embed_tokens")
                    .get((inner.vocab_size, inner.d_model), "weight")
            })?;
        let relative_bias = vb
            .pp("encoder")
            .pp("block")
            .pp("0")
            .pp("layer")
            .pp("0")
            .pp("SelfAttention")
            .get(
                (inner.relative_attention_num_buckets, inner.num_heads),
                "relative_attention_bias.weight",
            )?;
        let mut layers = Vec::with_capacity(inner.num_layers);
        for i in 0..inner.num_layers {
            layers.push(T5Layer::load(
                inner,
                vb.pp("encoder").pp("block").pp(&i.to_string()),
            )?);
        }
        Ok(Self {
            final_ln: vb
                .pp("encoder")
                .pp("final_layer_norm")
                .get(inner.d_model, "weight")?,
            cfg,
            embed,
            layers,
            relative_bias,
        })
    }

    pub fn device(&self) -> &Device {
        self.embed.device()
    }

    pub fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        let (_b, s) = input_ids.dims2()?;
        let mut hidden = self
            .embed
            .index_select(&input_ids.flatten_all()?, 0)?
            .reshape((input_ids.dim(0)?, s, self.cfg.inner.d_model))?;
        let buckets = relative_position_bucket(
            s,
            self.cfg.inner.relative_attention_num_buckets,
            self.cfg.inner.relative_attention_max_distance,
            input_ids.device(),
        )?;
        let bias = self
            .relative_bias
            .index_select(&buckets.flatten_all()?, 0)?
            .reshape((s, s, self.cfg.inner.num_heads))?
            .permute((2, 0, 1))?
            .unsqueeze(0)?;
        for layer in &self.layers {
            hidden = layer.forward(&hidden, &bias)?;
        }
        t5_layer_norm(&hidden, &self.final_ln, self.cfg.inner.eps)
    }
}

struct ClipAttention {
    q: Linear,
    k: Linear,
    v: Linear,
    out: Linear,
    heads: usize,
    dim_head: usize,
}

impl ClipAttention {
    fn load(dim: usize, heads: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            q: Linear::load(dim, dim, vb.pp("q_proj"))?,
            k: Linear::load(dim, dim, vb.pp("k_proj"))?,
            v: Linear::load(dim, dim, vb.pp("v_proj"))?,
            out: Linear::load(dim, dim, vb.pp("out_proj"))?,
            heads,
            dim_head: dim / heads,
        })
    }

    fn forward(&self, xs: &Tensor, mask: Option<&Tensor>) -> Result<Tensor> {
        let (b, s, _) = xs.dims3()?;
        let q = self
            .q
            .forward(xs)?
            .reshape((b, s, self.heads, self.dim_head))?
            .transpose(1, 2)?
            .contiguous()?;
        let k = self
            .k
            .forward(xs)?
            .reshape((b, s, self.heads, self.dim_head))?
            .transpose(1, 2)?
            .contiguous()?;
        let v = self
            .v
            .forward(xs)?
            .reshape((b, s, self.heads, self.dim_head))?
            .transpose(1, 2)?
            .contiguous()?;
        let attn = nn::scaled_dot_product_attention(&q, &k, &v, mask)?;
        let attn = attn
            .transpose(1, 2)?
            .contiguous()?
            .reshape((b, s, self.heads * self.dim_head))?;
        self.out.forward(&attn)
    }
}

struct ClipMlp {
    fc1: Linear,
    fc2: Linear,
}

impl ClipMlp {
    fn load(dim: usize, inner: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            fc1: Linear::load(dim, inner, vb.pp("fc1"))?,
            fc2: Linear::load(inner, dim, vb.pp("fc2"))?,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.fc2.forward(&nn::quick_gelu(&self.fc1.forward(xs)?)?)
    }
}

struct ClipLayer {
    attn: ClipAttention,
    ln1_w: Tensor,
    ln1_b: Tensor,
    mlp: ClipMlp,
    ln2_w: Tensor,
    ln2_b: Tensor,
    eps: f64,
}

impl ClipLayer {
    fn load(cfg: &ClipTextConfig, vb: VarBuilder) -> Result<Self> {
        let dim = cfg.hidden_size;
        Ok(Self {
            attn: ClipAttention::load(dim, cfg.num_attention_heads, vb.pp("self_attn"))?,
            ln1_w: vb.pp("layer_norm1").get(dim, "weight")?,
            ln1_b: vb.pp("layer_norm1").get(dim, "bias")?,
            mlp: ClipMlp::load(dim, cfg.intermediate_size, vb.pp("mlp"))?,
            ln2_w: vb.pp("layer_norm2").get(dim, "weight")?,
            ln2_b: vb.pp("layer_norm2").get(dim, "bias")?,
            eps: cfg.layer_norm_eps,
        })
    }

    fn forward(&self, xs: &Tensor, mask: Option<&Tensor>) -> Result<Tensor> {
        let n1 = nn::layer_norm(&xs.to_dtype(DType::F32)?, self.eps, Some(&self.ln1_w), Some(&self.ln1_b))?
            .to_dtype(xs.dtype())?;
        let xs = (xs + self.attn.forward(&n1, mask)?)?;
        let n2 = nn::layer_norm(&xs.to_dtype(DType::F32)?, self.eps, Some(&self.ln2_w), Some(&self.ln2_b))?
            .to_dtype(xs.dtype())?;
        xs + self.mlp.forward(&n2)?
    }
}

fn causal_mask(seq: usize, device: &Device, dtype: DType) -> Result<Tensor> {
    let mut data = vec![0.0f32; seq * seq];
    for q in 0..seq {
        for k in 0..seq {
            if k > q {
                data[q * seq + k] = -1e9;
            }
        }
    }
    Tensor::from_vec(data, (1, 1, seq, seq), device)?.to_dtype(dtype)
}

/// CLIP text encoder (pooled EOS hidden after final LayerNorm).
pub struct ClipTextEncoder {
    pub cfg: ClipTextConfig,
    token_emb: Tensor,
    pos_emb: Tensor,
    layers: Vec<ClipLayer>,
    final_ln_w: Tensor,
    final_ln_b: Tensor,
}

impl ClipTextEncoder {
    pub fn load(cfg: ClipTextConfig, vb: VarBuilder) -> Result<Self> {
        let root = if vb
            .pp("text_model")
            .pp("embeddings")
            .pp("token_embedding")
            .get((cfg.vocab_size, cfg.hidden_size), "weight")
            .is_ok()
        {
            vb.pp("text_model")
        } else {
            vb
        };
        let emb = root.pp("embeddings");
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            layers.push(ClipLayer::load(
                &cfg,
                root.pp("encoder").pp("layers").pp(i),
            )?);
        }
        Ok(Self {
            token_emb: emb
                .pp("token_embedding")
                .get((cfg.vocab_size, cfg.hidden_size), "weight")?,
            pos_emb: emb
                .pp("position_embedding")
                .get((cfg.max_position_embeddings, cfg.hidden_size), "weight")?,
            final_ln_w: root.pp("final_layer_norm").get(cfg.hidden_size, "weight")?,
            final_ln_b: root.pp("final_layer_norm").get(cfg.hidden_size, "bias")?,
            layers,
            cfg,
        })
    }

    pub fn device(&self) -> &Device {
        self.token_emb.device()
    }

    /// Returns `(last_hidden [B,S,D], pooled [B,D])`.
    pub fn forward(&self, input_ids: &Tensor) -> Result<(Tensor, Tensor)> {
        let (b, s) = input_ids.dims2()?;
        let s = s.min(self.cfg.max_position_embeddings);
        let ids = input_ids.narrow(1, 0, s)?;
        let tok = self
            .token_emb
            .index_select(&ids.flatten_all()?, 0)?
            .reshape((b, s, self.cfg.hidden_size))?;
        let pos = self.pos_emb.narrow(0, 0, s)?.reshape((1, s, self.cfg.hidden_size))?;
        let mut hidden = (tok + pos.broadcast_as(tok.shape())?)?;
        let mask = causal_mask(s, hidden.device(), hidden.dtype())?;
        for layer in &self.layers {
            hidden = layer.forward(&hidden, Some(&mask))?;
        }
        hidden = nn::layer_norm(
            &hidden.to_dtype(DType::F32)?,
            self.cfg.layer_norm_eps,
            Some(&self.final_ln_w),
            Some(&self.final_ln_b),
        )?
        .to_dtype(hidden.dtype())?;
        let pooled = pool_eos(&hidden, &ids)?;
        Ok((hidden, pooled))
    }
}

fn pool_eos(hidden: &Tensor, ids: &Tensor) -> Result<Tensor> {
    let (b, _s, d) = hidden.dims3()?;
    let ids_u32 = ids.to_dtype(DType::U32)?.to_vec2::<u32>()?;
    let mut rows = Vec::with_capacity(b);
    for (bi, row) in ids_u32.iter().enumerate() {
        let eos = row
            .iter()
            .enumerate()
            .max_by_key(|(_, t)| *t)
            .map(|(i, _)| i)
            .unwrap_or(0);
        rows.push(hidden.i((bi, eos))?);
    }
    Tensor::stack(&rows, 0)?.reshape((b, d))
}

/// CLIP pooled + T5 sequence (or dummy hashes for CI).
pub struct Flux1TextEncoder {
    clip: Option<ClipTextEncoder>,
    t5: Option<T5Encoder>,
    dummy_joint: usize,
    dummy_pooled: usize,
    device: Device,
    dtype: DType,
}

impl Flux1TextEncoder {
    pub fn dummy(joint: usize, pooled: usize, device: &Device, dtype: DType) -> Self {
        Self {
            clip: None,
            t5: None,
            dummy_joint: joint,
            dummy_pooled: pooled,
            device: device.clone(),
            dtype,
        }
    }

    pub fn load(
        clip: Option<ClipTextEncoder>,
        t5: Option<T5Encoder>,
        joint: usize,
        pooled: usize,
        device: Device,
        dtype: DType,
    ) -> Self {
        if clip.is_none() && t5.is_none() {
            return Self::dummy(joint, pooled, &device, dtype);
        }
        Self {
            clip,
            t5,
            dummy_joint: joint,
            dummy_pooled: pooled,
            device,
            dtype,
        }
    }

    pub fn t5_config(&self) -> Option<&T5Config> {
        self.t5.as_ref().map(|e| &e.cfg)
    }

    pub fn clip_config(&self) -> Option<&ClipTextConfig> {
        self.clip.as_ref().map(|e| &e.cfg)
    }

    /// Encode token ids. Returns `(t5_tokens [1,S,joint], clip_pooled [1,pooled])`.
    pub fn encode(&self, clip_ids: &[u32], t5_ids: &[u32]) -> Result<(Tensor, Tensor)> {
        if flux1_dummy_text() || (self.clip.is_none() && self.t5.is_none()) {
            return self.dummy_encode(clip_ids, t5_ids);
        }
        let pooled = if let Some(clip) = &self.clip {
            let input = Tensor::new(clip_ids, clip.device())?.unsqueeze(0)?;
            let (_h, p) = clip.forward(&input)?;
            p.to_dtype(self.dtype)?
        } else {
            dummy_vec(clip_ids, 1, self.dummy_pooled, &self.device, self.dtype)?
                .reshape((1, self.dummy_pooled))?
        };
        let tokens = if let Some(t5) = &self.t5 {
            let input = Tensor::new(t5_ids, t5.device())?.unsqueeze(0)?;
            t5.forward(&input)?.to_dtype(self.dtype)?
        } else {
            dummy_vec(t5_ids, t5_ids.len().max(1), self.dummy_joint, &self.device, self.dtype)?
        };
        Ok((tokens, pooled))
    }

    fn dummy_encode(&self, clip_ids: &[u32], t5_ids: &[u32]) -> Result<(Tensor, Tensor)> {
        let seq = t5_ids.len().max(1);
        let tokens = dummy_vec(t5_ids, seq, self.dummy_joint, &self.device, self.dtype)?;
        let pooled = dummy_vec(clip_ids, 1, self.dummy_pooled, &self.device, self.dtype)?
            .reshape((1, self.dummy_pooled))?;
        Ok((tokens, pooled))
    }
}

fn dummy_vec(ids: &[u32], seq: usize, dim: usize, device: &Device, dtype: DType) -> Result<Tensor> {
    let data: Vec<f32> = (0..seq * dim)
        .map(|i| {
            let id = ids.get(i % ids.len().max(1)).copied().unwrap_or(0) as f32;
            ((id * 0.01) + (i as f32) * 0.001) % 1.0
        })
        .collect();
    Tensor::from_vec(data, (1, seq, dim), device)?.to_dtype(dtype)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tiny_clip_pools() {
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let enc = ClipTextEncoder::load(ClipTextConfig::tiny(), vb).unwrap();
        let ids = Tensor::from_vec(vec![1u32, 2, 3, 2], (1, 4), &device).unwrap();
        let (h, p) = enc.forward(&ids).unwrap();
        assert_eq!(h.dims(), &[1, 4, 16]);
        assert_eq!(p.dims(), &[1, 16]);
    }

    #[test]
    fn tiny_t5_encodes() {
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let enc = T5Encoder::load(T5Config::tiny(), vb).unwrap();
        let ids = Tensor::from_vec(vec![1u32, 2, 3, 4], (1, 4), &device).unwrap();
        let out = enc.forward(&ids).unwrap();
        assert_eq!(out.dims(), &[1, 4, 16]);
    }

    #[test]
    fn dummy_shapes() {
        let device = Device::Cpu;
        let text = Flux1TextEncoder::dummy(16, 8, &device, DType::F32);
        let (t, p) = text.encode(&[1, 2], &[3, 4, 5]).unwrap();
        assert_eq!(t.dims(), &[1, 3, 16]);
        assert_eq!(p.dims(), &[1, 8]);
    }
}
