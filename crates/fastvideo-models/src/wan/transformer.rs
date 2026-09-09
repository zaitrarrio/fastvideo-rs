//! WanTransformer3D matching Hugging Face Diffusers weight names.

use candle_core::{DType, Device, Result, Tensor, D};
use candle_nn::VarBuilder;

use crate::nn::{self, Linear};
use crate::wan::config::WanVideoArchConfig;

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

#[derive(Debug, Clone)]
struct WanAttention {
    to_q: Linear,
    to_k: Linear,
    to_v: Linear,
    to_out: Linear,
    norm_q: RmsNorm,
    norm_k: RmsNorm,
    heads: usize,
    dim_head: usize,
}

impl WanAttention {
    fn load(dim: usize, heads: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        let dim_head = dim / heads;
        Ok(Self {
            to_q: Linear::load(dim, dim, vb.pp("to_q"))?,
            to_k: Linear::load(dim, dim, vb.pp("to_k"))?,
            to_v: Linear::load(dim, dim, vb.pp("to_v"))?,
            to_out: Linear::load(dim, dim, vb.pp("to_out").pp("0"))?,
            norm_q: RmsNorm::load(dim, eps, vb.pp("norm_q"))?,
            norm_k: RmsNorm::load(dim, eps, vb.pp("norm_k"))?,
            heads,
            dim_head,
        })
    }

    fn forward(
        &self,
        hidden: &Tensor,
        encoder: Option<&Tensor>,
        rotary: Option<&(Tensor, Tensor)>,
    ) -> Result<Tensor> {
        let ctx = encoder.unwrap_or(hidden);
        let q = self.norm_q.forward(&self.to_q.forward(hidden)?)?;
        let k = self.norm_k.forward(&self.to_k.forward(ctx)?)?;
        let v = self.to_v.forward(ctx)?;
        let (b, sq, _) = q.dims3()?;
        let sk = k.dim(1)?;
        let q = q.reshape((b, sq, self.heads, self.dim_head))?;
        let k = k.reshape((b, sk, self.heads, self.dim_head))?;
        let v = v.reshape((b, sk, self.heads, self.dim_head))?;
        let (mut q, mut k) = (q, k);
        if let Some((cos, sin)) = rotary {
            q = apply_rotary(&q, cos, sin)?;
            k = apply_rotary(&k, cos, sin)?;
        }
        let q = q.transpose(1, 2)?.contiguous()?;
        let k = k.transpose(1, 2)?.contiguous()?;
        let v = v.transpose(1, 2)?.contiguous()?;
        let attn = nn::scaled_dot_product_attention(&q, &k, &v)?;
        let attn = attn.transpose(1, 2)?.contiguous()?.reshape((b, sq, self.heads * self.dim_head))?;
        self.to_out.forward(&attn)
    }
}

fn apply_rotary(xs: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    // xs: [B, S, H, D], cos/sin: [1, S, 1, D] with interleaved pairs.
    let d = xs.dim(D::Minus1)?;
    let mut x1s = Vec::with_capacity(d / 2);
    let mut x2s = Vec::with_capacity(d / 2);
    let mut cos_e = Vec::with_capacity(d / 2);
    let mut sin_o = Vec::with_capacity(d / 2);
    for i in 0..(d / 2) {
        x1s.push(xs.narrow(D::Minus1, i * 2, 1)?);
        x2s.push(xs.narrow(D::Minus1, i * 2 + 1, 1)?);
        cos_e.push(cos.narrow(D::Minus1, i * 2, 1)?);
        sin_o.push(sin.narrow(D::Minus1, i * 2 + 1, 1)?);
    }
    let x1 = Tensor::cat(&x1s, D::Minus1)?;
    let x2 = Tensor::cat(&x2s, D::Minus1)?;
    let cos = Tensor::cat(&cos_e, D::Minus1)?;
    let sin = Tensor::cat(&sin_o, D::Minus1)?;
    let out1 = (x1.broadcast_mul(&cos)? - x2.broadcast_mul(&sin)?)?;
    let out2 = (x1.broadcast_mul(&sin)? + x2.broadcast_mul(&cos)?)?;
    let mut parts = Vec::with_capacity(d);
    let x1s = out1.chunk(d / 2, D::Minus1)?;
    let x2s = out2.chunk(d / 2, D::Minus1)?;
    for (a, b) in x1s.into_iter().zip(x2s) {
        parts.push(a);
        parts.push(b);
    }
    Tensor::cat(&parts, D::Minus1)
}

fn rotary_1d(dim: usize, seq: usize, theta: f64, device: &Device) -> Result<(Tensor, Tensor)> {
    let half = dim / 2;
    let pos: Vec<f32> = (0..seq).map(|i| i as f32).collect();
    let freqs: Vec<f32> = (0..half)
        .map(|i| 1.0 / (theta.powf(2.0 * i as f64 / dim as f64) as f32))
        .collect();
    let pos = Tensor::from_vec(pos, (seq,), device)?;
    let freqs = Tensor::from_vec(freqs, (half,), device)?;
    let args = pos.reshape((seq, 1))?.broadcast_mul(&freqs.reshape((1, half))?)?;
    let cos = args.cos()?;
    let sin = args.sin()?;
    // repeat_interleave 2 on last dim
    let mut cos_parts = Vec::new();
    let mut sin_parts = Vec::new();
    for i in 0..half {
        let c = cos.narrow(1, i, 1)?;
        let s = sin.narrow(1, i, 1)?;
        cos_parts.push(c.clone());
        cos_parts.push(c);
        sin_parts.push(s.clone());
        sin_parts.push(s);
    }
    Ok((Tensor::cat(&cos_parts, 1)?, Tensor::cat(&sin_parts, 1)?))
}

fn wan_rope(
    cfg: &WanVideoArchConfig,
    frames: usize,
    height: usize,
    width: usize,
    device: &Device,
) -> Result<(Tensor, Tensor)> {
    let d = cfg.attention_head_dim;
    let h_dim = 2 * (d / 6);
    let w_dim = h_dim;
    let t_dim = d - h_dim - w_dim;
    let (cos_t, sin_t) = rotary_1d(t_dim, cfg.rope_max_seq_len, 10000.0, device)?;
    let (cos_h, sin_h) = rotary_1d(h_dim, cfg.rope_max_seq_len, 10000.0, device)?;
    let (cos_w, sin_w) = rotary_1d(w_dim, cfg.rope_max_seq_len, 10000.0, device)?;
    let ppf = frames / cfg.patch_size[0];
    let pph = height / cfg.patch_size[1];
    let ppw = width / cfg.patch_size[2];
    let cos_f = cos_t.narrow(0, 0, ppf)?.reshape((ppf, 1, 1, t_dim))?.broadcast_as((ppf, pph, ppw, t_dim))?;
    let cos_hh = cos_h.narrow(0, 0, pph)?.reshape((1, pph, 1, h_dim))?.broadcast_as((ppf, pph, ppw, h_dim))?;
    let cos_ww = cos_w.narrow(0, 0, ppw)?.reshape((1, 1, ppw, w_dim))?.broadcast_as((ppf, pph, ppw, w_dim))?;
    let sin_f = sin_t.narrow(0, 0, ppf)?.reshape((ppf, 1, 1, t_dim))?.broadcast_as((ppf, pph, ppw, t_dim))?;
    let sin_hh = sin_h.narrow(0, 0, pph)?.reshape((1, pph, 1, h_dim))?.broadcast_as((ppf, pph, ppw, h_dim))?;
    let sin_ww = sin_w.narrow(0, 0, ppw)?.reshape((1, 1, ppw, w_dim))?.broadcast_as((ppf, pph, ppw, w_dim))?;
    let seq = ppf * pph * ppw;
    let cos = Tensor::cat(&[&cos_f, &cos_hh, &cos_ww], D::Minus1)?.reshape((1, seq, 1, d))?;
    let sin = Tensor::cat(&[&sin_f, &sin_hh, &sin_ww], D::Minus1)?.reshape((1, seq, 1, d))?;
    Ok((cos.to_dtype(DType::F32)?, sin.to_dtype(DType::F32)?))
}

#[derive(Debug, Clone)]
struct FeedForward {
    proj: Linear,
    out: Linear,
}

impl FeedForward {
    fn load(dim: usize, ffn_dim: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            proj: Linear::load(dim, ffn_dim, vb.pp("net").pp("0").pp("proj"))?,
            out: Linear::load(ffn_dim, dim, vb.pp("net").pp("2"))?,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let h = nn::gelu_tanh(&self.proj.forward(xs)?)?;
        self.out.forward(&h)
    }
}

#[derive(Debug, Clone)]
struct TextProjection {
    linear_1: Linear,
    linear_2: Linear,
}

impl TextProjection {
    fn load(in_dim: usize, dim: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            linear_1: Linear::load(in_dim, dim, vb.pp("linear_1"))?,
            linear_2: Linear::load(dim, dim, vb.pp("linear_2"))?,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let h = nn::gelu_tanh(&self.linear_1.forward(xs)?)?;
        self.linear_2.forward(&h)
    }
}

#[derive(Debug, Clone)]
struct TimestepEmbedding {
    linear_1: Linear,
    linear_2: Linear,
}

impl TimestepEmbedding {
    fn load(in_dim: usize, dim: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            linear_1: Linear::load(in_dim, dim, vb.pp("linear_1"))?,
            linear_2: Linear::load(dim, dim, vb.pp("linear_2"))?,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let h = nn::silu(&self.linear_1.forward(xs)?)?;
        self.linear_2.forward(&h)
    }
}

#[derive(Debug, Clone)]
struct WanBlock {
    norm1_eps: f64,
    attn1: WanAttention,
    attn2: WanAttention,
    norm2_weight: Tensor,
    norm2_bias: Tensor,
    ffn: FeedForward,
    scale_shift_table: Tensor,
}

impl WanBlock {
    fn load(cfg: &WanVideoArchConfig, vb: VarBuilder) -> Result<Self> {
        let dim = cfg.hidden_size();
        Ok(Self {
            norm1_eps: cfg.eps as f64,
            attn1: WanAttention::load(dim, cfg.num_attention_heads, cfg.eps as f64, vb.pp("attn1"))?,
            attn2: WanAttention::load(dim, cfg.num_attention_heads, cfg.eps as f64, vb.pp("attn2"))?,
            norm2_weight: vb.pp("norm2").get(dim, "weight")?,
            norm2_bias: vb.pp("norm2").get(dim, "bias")?,
            ffn: FeedForward::load(dim, cfg.ffn_dim, vb.pp("ffn"))?,
            scale_shift_table: vb.get((1, 6, dim), "scale_shift_table")?,
        })
    }

    fn forward(
        &self,
        hidden: &Tensor,
        encoder: &Tensor,
        temb: &Tensor,
        rotary: &(Tensor, Tensor),
    ) -> Result<Tensor> {
        let e = self
            .scale_shift_table
            .to_dtype(DType::F32)?
            .broadcast_add(&temb.to_dtype(DType::F32)?)?;
        let chunks = e.chunk(6, 1)?;
        let shift_msa = &chunks[0];
        let scale_msa = &chunks[1];
        let gate_msa = &chunks[2];
        let c_shift = &chunks[3];
        let c_scale = &chunks[4];
        let c_gate = &chunks[5];

        let normed = nn::layer_norm(&hidden.to_dtype(DType::F32)?, self.norm1_eps, None, None)?;
        let normed = (normed.broadcast_mul(&(scale_msa.to_dtype(DType::F32)? + 1.0)?)?.broadcast_add(&shift_msa.to_dtype(DType::F32)?)?)
            .to_dtype(hidden.dtype())?;
        let attn = self.attn1.forward(&normed, None, Some(rotary))?;
        let hidden = (hidden.to_dtype(DType::F32)?.broadcast_add(&attn.to_dtype(DType::F32)?.broadcast_mul(gate_msa)?)?)
            .to_dtype(hidden.dtype())?;

        let normed = nn::layer_norm(
            &hidden.to_dtype(DType::F32)?,
            self.norm1_eps,
            Some(&self.norm2_weight),
            Some(&self.norm2_bias),
        )?
        .to_dtype(hidden.dtype())?;
        let attn = self.attn2.forward(&normed, Some(encoder), None)?;
        let hidden = (hidden + attn)?;

        let normed = nn::layer_norm(&hidden.to_dtype(DType::F32)?, self.norm1_eps, None, None)?;
        let normed = (normed.broadcast_mul(&(c_scale.to_dtype(DType::F32)? + 1.0)?)?.broadcast_add(&c_shift.to_dtype(DType::F32)?)?)
            .to_dtype(hidden.dtype())?;
        let ff = self.ffn.forward(&normed)?;
        (hidden.to_dtype(DType::F32)?.broadcast_add(&ff.to_dtype(DType::F32)?.broadcast_mul(c_gate)?)?)
            .to_dtype(hidden.dtype())
    }
}

#[derive(Debug, Clone)]
pub struct WanTransformer3D {
    pub cfg: WanVideoArchConfig,
    patch_weight: Tensor,
    patch_bias: Tensor,
    time_embedder: TimestepEmbedding,
    time_proj: Linear,
    text_embedder: TextProjection,
    blocks: Vec<WanBlock>,
    proj_out: Linear,
    scale_shift_table: Tensor,
    freq_dim: usize,
}

impl WanTransformer3D {
    pub fn load(cfg: WanVideoArchConfig, vb: VarBuilder) -> Result<Self> {
        let dim = cfg.hidden_size();
        let p = cfg.patch_size;
        let patch_weight = vb
            .pp("patch_embedding")
            .get((dim, cfg.in_channels, p[0], p[1], p[2]), "weight")?;
        let patch_bias = vb.pp("patch_embedding").get(dim, "bias")?;
        let cond = vb.pp("condition_embedder");
        let mut blocks = Vec::with_capacity(cfg.num_layers);
        for i in 0..cfg.num_layers {
            blocks.push(WanBlock::load(&cfg, vb.pp("blocks").pp(&i.to_string()))?);
        }
        Ok(Self {
            time_embedder: TimestepEmbedding::load(cfg.freq_dim, dim, cond.pp("time_embedder"))?,
            time_proj: Linear::load(dim, dim * 6, cond.pp("time_proj"))?,
            text_embedder: TextProjection::load(cfg.text_dim, dim, cond.pp("text_embedder"))?,
            freq_dim: cfg.freq_dim,
            proj_out: Linear::load(dim, cfg.out_channels * p.iter().product::<usize>(), vb.pp("proj_out"))?,
            scale_shift_table: vb.get((1, 2, dim), "scale_shift_table")?,
            cfg,
            patch_weight,
            patch_bias,
            blocks,
        })
    }

    fn patch_embed(&self, xs: &Tensor) -> Result<Tensor> {
        // xs: [B, C, T, H, W], kernel/stride (1,2,2) → conv2d per frame.
        let (b, c, t, h, w) = xs.dims5()?;
        let p = self.cfg.patch_size;
        let x = xs.transpose(1, 2)?.contiguous()?.reshape((b * t, c, h, w))?;
        let k = self.patch_weight.reshape((self.cfg.hidden_size(), c * p[0], p[1], p[2]))?;
        let y = nn::conv2d(&x, &k.to_dtype(xs.dtype())?, 0, p[1])?;
        let y = y.broadcast_add(&self.patch_bias.to_dtype(xs.dtype())?.reshape((1, self.cfg.hidden_size(), 1, 1))?)?;
        let (_, dim, hh, ww) = y.dims4()?;
        y.reshape((b, t, dim, hh, ww))?
            .permute((0, 2, 1, 3, 4))?
            .flatten_from(2)?
            .transpose(1, 2)?
            .contiguous()
    }

    pub fn forward(&self, latents: &Tensor, timestep: &Tensor, encoder: &Tensor) -> Result<Tensor> {
        let (b, _c, t, h, w) = latents.dims5()?;
        let device = latents.device();
        let rotary = wan_rope(&self.cfg, t, h, w, device)?;
        let mut hidden = self.patch_embed(latents)?;
        let temb_in = nn::sinusoidal_timesteps(timestep, self.freq_dim, device)?;
        let temb = self.time_embedder.forward(&temb_in.to_dtype(latents.dtype())?)?;
        let timestep_proj = self.time_proj.forward(&nn::silu(&temb)?)?;
        let timestep_proj = timestep_proj.reshape((b, 6, self.cfg.hidden_size()))?;
        let encoder = self.text_embedder.forward(encoder)?;
        for block in &self.blocks {
            hidden = block.forward(&hidden, &encoder, &timestep_proj, &rotary)?;
        }
        let table = self.scale_shift_table.to_dtype(DType::F32)?;
        let temb_f = temb.to_dtype(DType::F32)?.unsqueeze(1)?;
        let ss = table.broadcast_add(&temb_f)?;
        let chunks = ss.chunk(2, 1)?;
        let shift = &chunks[0];
        let scale = &chunks[1];
        hidden = nn::layer_norm(&hidden.to_dtype(DType::F32)?, self.cfg.eps as f64, None, None)?;
        hidden = (hidden.broadcast_mul(&(scale + 1.0)?)?.broadcast_add(shift)?).to_dtype(latents.dtype())?;
        hidden = self.proj_out.forward(&hidden)?;
        let p = self.cfg.patch_size;
        let ppf = t / p[0];
        let pph = h / p[1];
        let ppw = w / p[2];
        let hidden = hidden.reshape(vec![
            b,
            ppf,
            pph,
            ppw,
            p[0],
            p[1],
            p[2],
            self.cfg.out_channels,
        ])?;
        hidden
            .permute(vec![0, 7, 1, 4, 2, 5, 3, 6])?
            .contiguous()?
            .reshape((b, self.cfg.out_channels, ppf * p[0], pph * p[1], ppw * p[2]))
    }
}

impl WanVideoArchConfig {
    pub fn tiny() -> Self {
        Self {
            patch_size: [1, 2, 2],
            text_len: 8,
            num_attention_heads: 2,
            attention_head_dim: 8,
            in_channels: 4,
            out_channels: 4,
            text_dim: 16,
            freq_dim: 16,
            ffn_dim: 32,
            num_layers: 1,
            eps: 1e-6,
            rope_max_seq_len: 64,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_nn::VarBuilder;

    #[test]
    fn tiny_dit_forward_shape() {
        let device = Device::Cpu;
        let cfg = WanVideoArchConfig::tiny();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = WanTransformer3D::load(cfg.clone(), vb).unwrap();
        let latents = Tensor::zeros((1, 4, 2, 8, 8), DType::F32, &device).unwrap();
        let t = Tensor::from_vec(vec![500f32], (1,), &device).unwrap();
        let enc = Tensor::zeros((1, 8, 16), DType::F32, &device).unwrap();
        let out = model.forward(&latents, &t, &enc).unwrap();
        assert_eq!(out.dims(), &[1, 4, 2, 8, 8]);
    }
}
