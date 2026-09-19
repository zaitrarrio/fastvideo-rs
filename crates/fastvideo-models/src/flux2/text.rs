//! Flux2 text encoders: Mistral3 (dev) and Qwen3 (Klein).
//!
//! Upstream FastVideo loads both through Transformers `from_pretrained_local`.
//! This crate ports the **postprocess** (layer-stack concat) and a Candle
//! Qwen3-style encoder so Klein can run without Python. Full Mistral3 remains
//! a documented HF-backed follow-up; tiny / dummy embeds cover CI smoke.

use candle_core::{DType, Device, IndexOp, Result, Tensor};
use candle_nn::VarBuilder;

use crate::nn::{self, Linear};

use super::family::stack_hidden_layers;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Flux2TextKind {
    /// FLUX.2-dev: Mistral3, layers (10, 20, 30).
    Mistral3,
    /// Klein: Qwen3, layers (9, 18, 27).
    Qwen3,
}

impl Flux2TextKind {
    pub fn from_preset(preset: &str) -> Self {
        if preset.contains("klein") {
            Self::Qwen3
        } else {
            Self::Mistral3
        }
    }

    pub fn out_layers(self) -> &'static [usize] {
        match self {
            Self::Mistral3 => &[10, 20, 30],
            Self::Qwen3 => &[9, 18, 27],
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Mistral3 => "mistral3",
            Self::Qwen3 => "qwen3",
        }
    }
}

pub const FLUX2_SYSTEM_MESSAGE: &str =
    "You are an AI that reasons about image descriptions. You give structured \
     responses focusing on object relationships, object\nattribution and actions \
     without speculation.";

#[derive(Debug, Clone, PartialEq)]
pub struct Qwen3Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f32,
    pub max_position_embeddings: usize,
    pub text_len: usize,
}

impl Qwen3Config {
    /// FastVideo `Qwen3TextArchConfig` (Klein 4B).
    pub fn klein_4b() -> Self {
        Self {
            vocab_size: 151936,
            hidden_size: 2560,
            intermediate_size: 9728,
            num_hidden_layers: 36,
            num_attention_heads: 32,
            num_key_value_heads: 8,
            head_dim: 128,
            rms_norm_eps: 1e-6,
            rope_theta: 1_000_000.0,
            max_position_embeddings: 40960,
            text_len: 512,
        }
    }

    pub fn tiny() -> Self {
        Self {
            vocab_size: 32,
            hidden_size: 16,
            intermediate_size: 32,
            num_hidden_layers: 2,
            num_attention_heads: 2,
            num_key_value_heads: 2,
            head_dim: 8,
            rms_norm_eps: 1e-6,
            rope_theta: 10_000.0,
            max_position_embeddings: 64,
            text_len: 8,
        }
    }
}

#[derive(Debug, Clone)]
struct RmsNorm {
    weight: Tensor,
    eps: f64,
}

impl RmsNorm {
    fn load(dim: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            weight: vb.get(dim, "weight")?,
            eps,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        nn::rms_norm(xs, &self.weight, self.eps)
    }
}

struct Qwen3Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    rope_theta: f32,
}

impl Qwen3Attention {
    fn load(cfg: &Qwen3Config, vb: VarBuilder) -> Result<Self> {
        let q = cfg.num_attention_heads * cfg.head_dim;
        let kv = cfg.num_key_value_heads * cfg.head_dim;
        Ok(Self {
            q_proj: Linear::load(cfg.hidden_size, q, vb.pp("q_proj"))?,
            k_proj: Linear::load(cfg.hidden_size, kv, vb.pp("k_proj"))?,
            v_proj: Linear::load(cfg.hidden_size, kv, vb.pp("v_proj"))?,
            o_proj: Linear::load(q, cfg.hidden_size, vb.pp("o_proj"))?,
            q_norm: RmsNorm::load(cfg.head_dim, cfg.rms_norm_eps, vb.pp("q_norm"))?,
            k_norm: RmsNorm::load(cfg.head_dim, cfg.rms_norm_eps, vb.pp("k_norm"))?,
            heads: cfg.num_attention_heads,
            kv_heads: cfg.num_key_value_heads,
            head_dim: cfg.head_dim,
            rope_theta: cfg.rope_theta,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (b, s, _) = xs.dims3()?;
        let q = self.q_proj.forward(xs)?.reshape((b, s, self.heads, self.head_dim))?;
        let k = self.k_proj.forward(xs)?.reshape((b, s, self.kv_heads, self.head_dim))?;
        let v = self.v_proj.forward(xs)?.reshape((b, s, self.kv_heads, self.head_dim))?;
        let q = apply_head_rms(&self.q_norm, &q)?;
        let k = apply_head_rms(&self.k_norm, &k)?;
        let q = apply_rope_neox(&q, self.rope_theta)?;
        let k = apply_rope_neox(&k, self.rope_theta)?;
        let k = repeat_kv(&k, self.heads / self.kv_heads)?;
        let v = repeat_kv(&v, self.heads / self.kv_heads)?;
        let q = q.transpose(1, 2)?.contiguous()?;
        let k = k.transpose(1, 2)?.contiguous()?;
        let v = v.transpose(1, 2)?.contiguous()?;
        let mask = causal_mask(s, xs.device(), xs.dtype())?;
        let attn = nn::scaled_dot_product_attention(&q, &k, &v, Some(&mask))?;
        let attn = attn
            .transpose(1, 2)?
            .contiguous()?
            .reshape((b, s, self.heads * self.head_dim))?;
        self.o_proj.forward(&attn)
    }
}

fn apply_head_rms(norm: &RmsNorm, xs: &Tensor) -> Result<Tensor> {
    let dims = xs.dims().to_vec();
    let last = *dims.last().unwrap();
    let flat = xs.reshape(((), last))?;
    let y = norm.forward(&flat)?;
    y.reshape(dims)
}

fn repeat_kv(xs: &Tensor, reps: usize) -> Result<Tensor> {
    if reps == 1 {
        return Ok(xs.clone());
    }
    let (b, s, h, d) = xs.dims4()?;
    xs.unsqueeze(3)?
        .expand((b, s, h, reps, d))?
        .reshape((b, s, h * reps, d))
}

fn apply_rope_neox(xs: &Tensor, theta: f32) -> Result<Tensor> {
    let (b, s, h, d) = xs.dims4()?;
    let half = d / 2;
    let device = xs.device();
    let dtype = xs.dtype();
    let inv: Vec<f32> = (0..half)
        .map(|i| 1.0 / theta.powf(i as f32 / half as f32))
        .collect();
    let freqs = Tensor::from_vec(inv, (half,), device)?.to_dtype(DType::F32)?;
    let pos = Tensor::arange(0f32, s as f32, device)?;
    let angles = pos.reshape((s, 1))?.broadcast_mul(&freqs.reshape((1, half))?)?;
    let cos = angles.cos()?.to_dtype(dtype)?;
    let sin = angles.sin()?.to_dtype(dtype)?;
    let x = xs.to_dtype(DType::F32)?.reshape((b * s * h, d))?;
    let x1 = x.narrow(1, 0, half)?;
    let x2 = x.narrow(1, half, half)?;
    let cos = cos
        .reshape((1, s, 1, half))?
        .expand((b, s, h, half))?
        .reshape((b * s * h, half))?
        .to_dtype(DType::F32)?;
    let sin = sin
        .reshape((1, s, 1, half))?
        .expand((b, s, h, half))?
        .reshape((b * s * h, half))?
        .to_dtype(DType::F32)?;
    let y1 = (x1.broadcast_mul(&cos)? - x2.broadcast_mul(&sin)?)?;
    let y2 = (x2.broadcast_mul(&cos)? + x1.broadcast_mul(&sin)?)?;
    Tensor::cat(&[&y1, &y2], 1)?
        .to_dtype(dtype)?
        .reshape((b, s, h, d))
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

struct Qwen3Mlp {
    gate: Linear,
    up: Linear,
    down: Linear,
}

impl Qwen3Mlp {
    fn load(cfg: &Qwen3Config, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            gate: Linear::load(cfg.hidden_size, cfg.intermediate_size, vb.pp("gate_proj"))?,
            up: Linear::load(cfg.hidden_size, cfg.intermediate_size, vb.pp("up_proj"))?,
            down: Linear::load(cfg.intermediate_size, cfg.hidden_size, vb.pp("down_proj"))?,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let gated = nn::silu(&self.gate.forward(xs)?)?;
        self.down.forward(&(gated * self.up.forward(xs)?)?)
    }
}

struct Qwen3Layer {
    input_norm: RmsNorm,
    attn: Qwen3Attention,
    post_norm: RmsNorm,
    mlp: Qwen3Mlp,
}

impl Qwen3Layer {
    fn load(cfg: &Qwen3Config, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            input_norm: RmsNorm::load(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?,
            attn: Qwen3Attention::load(cfg, vb.pp("self_attn"))?,
            post_norm: RmsNorm::load(
                cfg.hidden_size,
                cfg.rms_norm_eps,
                vb.pp("post_attention_layernorm"),
            )?,
            mlp: Qwen3Mlp::load(cfg, vb.pp("mlp"))?,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let h = (xs + self.attn.forward(&self.input_norm.forward(xs)?)?)?;
        &h + self.mlp.forward(&self.post_norm.forward(&h)?)?
    }
}

pub struct Qwen3Encoder {
    embed: Tensor,
    layers: Vec<Qwen3Layer>,
    norm: RmsNorm,
    pub cfg: Qwen3Config,
    device: Device,
}

impl Qwen3Encoder {
    pub fn load(cfg: Qwen3Config, vb: VarBuilder) -> Result<Self> {
        let embed = match vb.get((cfg.vocab_size, cfg.hidden_size), "embed_tokens.weight") {
            Ok(t) => t,
            Err(_) => vb.pp("embed_tokens").get((cfg.vocab_size, cfg.hidden_size), "weight")?,
        };
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            layers.push(Qwen3Layer::load(&cfg, vb.pp("layers").pp(i))?);
        }
        Ok(Self {
            embed,
            layers,
            norm: RmsNorm::load(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("norm"))?,
            device: vb.device().clone(),
            cfg,
        })
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Returns `(last_hidden, all_hidden_including_embedding)`.
    pub fn forward_hidden(&self, input_ids: &Tensor) -> Result<(Tensor, Vec<Tensor>)> {
        let mut hidden = {
            let ids = input_ids.flatten_all()?;
            let n = ids.dims1()?;
            let table = self.embed.to_dtype(DType::F32)?;
            let mut rows = Vec::with_capacity(n);
            let ids_u32 = ids.to_dtype(DType::U32)?.to_vec1::<u32>()?;
            for id in ids_u32 {
                let idx = (id as usize).min(self.cfg.vocab_size.saturating_sub(1));
                rows.push(table.i(idx)?);
            }
            let stacked = Tensor::stack(&rows, 0)?;
            let (b, s) = match input_ids.dims() {
                [b, s] => (*b, *s),
                [s] => (1, *s),
                other => candle_core::bail!("input_ids want [B,S] got {other:?}"),
            };
            stacked.reshape((b, s, self.cfg.hidden_size))?
        };
        let mut all = vec![hidden.clone()];
        for layer in &self.layers {
            hidden = layer.forward(&hidden)?;
            all.push(hidden.clone());
        }
        hidden = self.norm.forward(&hidden)?;
        *all.last_mut().unwrap() = hidden.clone();
        Ok((hidden, all))
    }
}

/// Flux2 text front-end: dummy (tiny), Qwen3 (Klein), or stacked hidden states.
pub struct Flux2TextEncoder {
    pub kind: Flux2TextKind,
    qwen: Option<Qwen3Encoder>,
    dummy_dim: usize,
    device: Device,
    dtype: DType,
}

impl Flux2TextEncoder {
    pub fn dummy(kind: Flux2TextKind, dim: usize, device: &Device, dtype: DType) -> Self {
        Self {
            kind,
            qwen: None,
            dummy_dim: dim,
            device: device.clone(),
            dtype,
        }
    }

    pub fn qwen3(enc: Qwen3Encoder, dtype: DType) -> Self {
        Self {
            kind: Flux2TextKind::Qwen3,
            device: enc.device().clone(),
            qwen: Some(enc),
            dummy_dim: 0,
            dtype,
        }
    }

    pub fn encode_ids(&self, ids: &[u32]) -> Result<Tensor> {
        if let Some(enc) = &self.qwen {
            let input = Tensor::new(ids, enc.device())?.unsqueeze(0)?;
            let (_last, all) = enc.forward_hidden(&input)?;
            return stack_selected(&all, self.kind.out_layers(), self.dtype);
        }
        let seq = ids.len().max(1);
        let data: Vec<f32> = (0..seq * self.dummy_dim)
            .map(|i| ((ids[i % ids.len()] as f32) * 0.01 + (i as f32) * 0.001) % 1.0)
            .collect();
        Tensor::from_vec(data, (1, seq, self.dummy_dim), &self.device)?.to_dtype(self.dtype)
    }
}

fn stack_selected(all: &[Tensor], layers: &[usize], dtype: DType) -> Result<Tensor> {
    let mut chosen = Vec::new();
    for &idx in layers {
        let t = all.get(idx.min(all.len().saturating_sub(1))).ok_or_else(|| {
            candle_core::Error::Msg(format!("missing hidden layer {idx}"))
        })?;
        chosen.push(t.to_dtype(DType::F32)?);
    }
    let (_b, seq, hidden) = chosen[0].dims3()?;
    let mut host_layers = Vec::new();
    let mut owned = Vec::new();
    for t in &chosen {
        owned.push(t.flatten_all()?.to_vec1::<f32>()?);
    }
    for row in &owned {
        host_layers.push(row.as_slice());
    }
    let stacked = stack_hidden_layers(&host_layers, seq, hidden)
        .map_err(|e| candle_core::Error::Msg(e))?;
    Tensor::from_vec(stacked, (1, seq, chosen.len() * hidden), chosen[0].device())?.to_dtype(dtype)
}

/// Host helper used by tests and the cudarc port.
pub fn stack_layers_host(layers: &[&[f32]], seq: usize, hidden: usize) -> std::result::Result<Vec<f32>, String> {
    stack_hidden_layers(layers, seq, hidden)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kinds_match_upstream_layers() {
        assert_eq!(Flux2TextKind::Mistral3.out_layers(), &[10, 20, 30]);
        assert_eq!(Flux2TextKind::Qwen3.out_layers(), &[9, 18, 27]);
        assert_eq!(Flux2TextKind::from_preset("flux2_klein_4b"), Flux2TextKind::Qwen3);
        assert_eq!(Flux2TextKind::from_preset("flux2_dev"), Flux2TextKind::Mistral3);
    }

    #[test]
    fn tiny_qwen3_encodes() {
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let enc = Qwen3Encoder::load(Qwen3Config::tiny(), vb).unwrap();
        let ids = Tensor::from_vec(vec![1u32, 2, 3, 4], (1, 4), &device).unwrap();
        let (last, all) = enc.forward_hidden(&ids).unwrap();
        assert_eq!(last.dims(), &[1, 4, 16]);
        assert_eq!(all.len(), 3);
    }
}
