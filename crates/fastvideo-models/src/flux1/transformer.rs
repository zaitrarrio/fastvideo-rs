//! FluxTransformer2DModel matching Diffusers / FastVideo FLUX.1 weight names.

use candle_core::{DType, Device, Result, Tensor, D};
use candle_nn::VarBuilder;

use crate::nn::{self, Linear};

use super::config::Flux1ArchConfig;
use super::family::{image_ids, text_ids};

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

fn apply_rotary(xs: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let (b, s, h, d) = xs.dims4()?;
    let dtype = xs.dtype();
    let x = xs.to_dtype(DType::F32)?.contiguous()?.reshape((b, s, h, d / 2, 2))?;
    let even = x.narrow(4, 0, 1)?.squeeze(4)?;
    let odd = x.narrow(4, 1, 1)?.squeeze(4)?;
    let cos = broadcast_rope(cos, b, s, h, d)?;
    let sin = broadcast_rope(sin, b, s, h, d)?;
    let cos_e = cos
        .narrow(3, 0, d)?
        .reshape((b, s, h, d / 2, 2))?
        .narrow(4, 0, 1)?
        .squeeze(4)?;
    let sin_e = sin
        .narrow(3, 0, d)?
        .reshape((b, s, h, d / 2, 2))?
        .narrow(4, 0, 1)?
        .squeeze(4)?;
    let y1 = (even.broadcast_mul(&cos_e)? - odd.broadcast_mul(&sin_e)?)?;
    let y2 = (even.broadcast_mul(&sin_e)? + odd.broadcast_mul(&cos_e)?)?;
    let y = Tensor::stack(&[&y1, &y2], 4)?.reshape((b, s, h, d))?;
    y.to_dtype(dtype)
}

fn broadcast_rope(freq: &Tensor, b: usize, s: usize, h: usize, d: usize) -> Result<Tensor> {
    let freq = freq.to_dtype(DType::F32)?;
    let freq = match freq.dims() {
        [seq, dim] if *seq == s && *dim == d => freq,
        [dim] if *dim == d => freq.reshape((1, d))?.expand((s, d))?.contiguous()?,
        [seq, dim] if *dim == d => freq.narrow(0, 0, s.min(*seq))?,
        other => candle_core::bail!("rope freq shape {other:?} want [{s}, {d}]"),
    };
    freq.reshape((1, s, 1, d))?.expand((b, s, h, d))?.contiguous()
}

struct Flux1Attention {
    to_q: Linear,
    to_k: Linear,
    to_v: Linear,
    to_out: Option<Linear>,
    norm_q: RmsNorm,
    norm_k: RmsNorm,
    add_q: Option<Linear>,
    add_k: Option<Linear>,
    add_v: Option<Linear>,
    to_add_out: Option<Linear>,
    norm_added_q: Option<RmsNorm>,
    norm_added_k: Option<RmsNorm>,
    heads: usize,
    dim_head: usize,
}

impl Flux1Attention {
    fn load(
        dim: usize,
        heads: usize,
        dim_head: usize,
        eps: f64,
        joint: bool,
        pre_only: bool,
        vb: VarBuilder,
    ) -> Result<Self> {
        let inner = heads * dim_head;
        let (add_q, add_k, add_v, to_add_out, norm_added_q, norm_added_k) = if joint {
            (
                Some(Linear::load(dim, inner, vb.pp("add_q_proj"))?),
                Some(Linear::load(dim, inner, vb.pp("add_k_proj"))?),
                Some(Linear::load(dim, inner, vb.pp("add_v_proj"))?),
                Some(Linear::load(inner, dim, vb.pp("to_add_out"))?),
                Some(RmsNorm::load(dim_head, eps, vb.pp("norm_added_q"))?),
                Some(RmsNorm::load(dim_head, eps, vb.pp("norm_added_k"))?),
            )
        } else {
            (None, None, None, None, None, None)
        };
        Ok(Self {
            to_q: Linear::load(dim, inner, vb.pp("to_q"))?,
            to_k: Linear::load(dim, inner, vb.pp("to_k"))?,
            to_v: Linear::load(dim, inner, vb.pp("to_v"))?,
            to_out: if pre_only {
                None
            } else {
                Some(Linear::load(inner, dim, vb.pp("to_out").pp("0"))?)
            },
            norm_q: RmsNorm::load(dim_head, eps, vb.pp("norm_q"))?,
            norm_k: RmsNorm::load(dim_head, eps, vb.pp("norm_k"))?,
            add_q,
            add_k,
            add_v,
            to_add_out,
            norm_added_q,
            norm_added_k,
            heads,
            dim_head,
        })
    }

    fn reshape_heads(&self, xs: &Tensor) -> Result<Tensor> {
        let (b, s, _) = xs.dims3()?;
        xs.reshape((b, s, self.heads, self.dim_head))
    }

    fn rms_heads(&self, xs: &Tensor, norm: &RmsNorm) -> Result<Tensor> {
        let dims = xs.dims().to_vec();
        let last = *dims.last().unwrap();
        let y = norm.forward(&xs.reshape(((), last))?)?;
        y.reshape(dims)
    }

    fn forward(
        &self,
        hidden: &Tensor,
        encoder: Option<&Tensor>,
        rope: Option<&(Tensor, Tensor)>,
    ) -> Result<(Tensor, Option<Tensor>)> {
        let q = self.rms_heads(&self.reshape_heads(&self.to_q.forward(hidden)?)?, &self.norm_q)?;
        let k = self.rms_heads(&self.reshape_heads(&self.to_k.forward(hidden)?)?, &self.norm_k)?;
        let v = self.reshape_heads(&self.to_v.forward(hidden)?)?;
        let (q, k, v, text_len) = if let (Some(enc), Some(aq), Some(ak), Some(av), Some(nq), Some(nk)) = (
            encoder,
            &self.add_q,
            &self.add_k,
            &self.add_v,
            &self.norm_added_q,
            &self.norm_added_k,
        ) {
            let eq = self.rms_heads(&self.reshape_heads(&aq.forward(enc)?)?, nq)?;
            let ek = self.rms_heads(&self.reshape_heads(&ak.forward(enc)?)?, nk)?;
            let ev = self.reshape_heads(&av.forward(enc)?)?;
            let text_len = enc.dim(1)?;
            (
                Tensor::cat(&[&eq, &q], 1)?,
                Tensor::cat(&[&ek, &k], 1)?,
                Tensor::cat(&[&ev, &v], 1)?,
                Some(text_len),
            )
        } else {
            (q, k, v, None)
        };
        let (mut q, mut k) = (q, k);
        if let Some((cos, sin)) = rope {
            q = apply_rotary(&q, cos, sin)?;
            k = apply_rotary(&k, cos, sin)?;
        }
        let attn = nn::scaled_dot_product_attention(
            &q.transpose(1, 2)?.contiguous()?,
            &k.transpose(1, 2)?.contiguous()?,
            &v.transpose(1, 2)?.contiguous()?,
            None,
        )?;
        let (b, _, s, _) = attn.dims4()?;
        let attn = attn
            .transpose(1, 2)?
            .contiguous()?
            .reshape((b, s, self.heads * self.dim_head))?;
        if let (Some(text_len), Some(to_add)) = (text_len, &self.to_add_out) {
            let text = attn.narrow(1, 0, text_len)?;
            let img = attn.narrow(1, text_len, s - text_len)?;
            let img = match &self.to_out {
                Some(proj) => proj.forward(&img)?,
                None => img,
            };
            let text = to_add.forward(&text)?;
            Ok((img, Some(text)))
        } else if let Some(proj) = &self.to_out {
            Ok((proj.forward(&attn)?, None))
        } else {
            Ok((attn, None))
        }
    }
}

struct GeluFeedForward {
    proj: Linear,
    out: Linear,
}

impl GeluFeedForward {
    fn load(dim: usize, inner: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            proj: Linear::load(dim, inner, vb.pp("net").pp("0").pp("proj"))?,
            out: Linear::load(inner, dim, vb.pp("net").pp("2"))?,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.out.forward(&nn::gelu_tanh(&self.proj.forward(xs)?)?)
    }
}

struct AdaLayerNormZero {
    linear: Linear,
    eps: f64,
}

impl AdaLayerNormZero {
    fn load(dim: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            linear: Linear::load(dim, dim * 6, vb.pp("linear"))?,
            eps,
        })
    }

    fn forward(&self, xs: &Tensor, emb: &Tensor) -> Result<(Tensor, Tensor, Tensor, Tensor, Tensor)> {
        let e = self.linear.forward(&nn::silu(emb)?)?;
        let last = e.dim(D::Minus1)?;
        let chunk = last / 6;
        let shift_msa = e.narrow(D::Minus1, 0, chunk)?;
        let scale_msa = e.narrow(D::Minus1, chunk, chunk)?;
        let gate_msa = e.narrow(D::Minus1, 2 * chunk, chunk)?;
        let shift_mlp = e.narrow(D::Minus1, 3 * chunk, chunk)?;
        let scale_mlp = e.narrow(D::Minus1, 4 * chunk, chunk)?;
        let gate_mlp = e.narrow(D::Minus1, 5 * chunk, chunk)?;
        let n = nn::layer_norm(xs, self.eps, None, None)?;
        let n = n
            .broadcast_mul(&(scale_msa.unsqueeze(1)? + 1.0)?)?
            .broadcast_add(&shift_msa.unsqueeze(1)?)?;
        Ok((n, gate_msa, shift_mlp, scale_mlp, gate_mlp))
    }
}

struct AdaLayerNormZeroSingle {
    linear: Linear,
    eps: f64,
}

impl AdaLayerNormZeroSingle {
    fn load(dim: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            linear: Linear::load(dim, dim * 3, vb.pp("linear"))?,
            eps,
        })
    }

    fn forward(&self, xs: &Tensor, emb: &Tensor) -> Result<(Tensor, Tensor)> {
        let e = self.linear.forward(&nn::silu(emb)?)?;
        let last = e.dim(D::Minus1)?;
        let chunk = last / 3;
        let shift = e.narrow(D::Minus1, 0, chunk)?;
        let scale = e.narrow(D::Minus1, chunk, chunk)?;
        let gate = e.narrow(D::Minus1, 2 * chunk, chunk)?;
        let n = nn::layer_norm(xs, self.eps, None, None)?;
        let n = n
            .broadcast_mul(&(scale.unsqueeze(1)? + 1.0)?)?
            .broadcast_add(&shift.unsqueeze(1)?)?;
        Ok((n, gate))
    }
}

struct AdaLayerNormContinuous {
    linear: Linear,
    eps: f64,
}

impl AdaLayerNormContinuous {
    fn load(dim: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            linear: Linear::load(dim, dim * 2, vb.pp("linear"))?,
            eps,
        })
    }

    fn forward(&self, xs: &Tensor, cond: &Tensor) -> Result<Tensor> {
        let emb = self.linear.forward(&nn::silu(cond)?)?;
        let last = emb.dim(D::Minus1)?;
        let scale = emb.narrow(D::Minus1, 0, last / 2)?;
        let shift = emb.narrow(D::Minus1, last / 2, last / 2)?;
        let n = nn::layer_norm(xs, self.eps, None, None)?;
        n.broadcast_mul(&(scale.unsqueeze(1)? + 1.0)?)?
            .broadcast_add(&shift.unsqueeze(1)?)
    }
}

struct TimestepEmbedding {
    linear_1: Linear,
    linear_2: Linear,
}

impl TimestepEmbedding {
    fn load(in_dim: usize, hidden: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            linear_1: Linear::load(in_dim, hidden, vb.pp("linear_1"))?,
            linear_2: Linear::load(hidden, hidden, vb.pp("linear_2"))?,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.linear_2.forward(&nn::silu(&self.linear_1.forward(xs)?)?)
    }
}

struct CombinedTimestepGuidanceTextProj {
    timestep: TimestepEmbedding,
    guidance: Option<TimestepEmbedding>,
    text: TimestepEmbedding,
    channels: usize,
}

impl CombinedTimestepGuidanceTextProj {
    fn load(
        channels: usize,
        hidden: usize,
        pooled: usize,
        with_guidance: bool,
        vb: VarBuilder,
    ) -> Result<Self> {
        Ok(Self {
            timestep: TimestepEmbedding::load(channels, hidden, vb.pp("timestep_embedder"))?,
            guidance: if with_guidance {
                Some(TimestepEmbedding::load(channels, hidden, vb.pp("guidance_embedder"))?)
            } else {
                None
            },
            text: TimestepEmbedding::load(pooled, hidden, vb.pp("text_embedder"))?,
            channels,
        })
    }

    fn forward(&self, timestep: &Tensor, guidance: Option<&Tensor>, pooled: &Tensor) -> Result<Tensor> {
        let proj = nn::sinusoidal_timesteps(timestep, self.channels, timestep.device())?;
        let mut emb = self.timestep.forward(&proj.to_dtype(timestep.dtype())?)?;
        if let (Some(g), Some(ge)) = (guidance, &self.guidance) {
            let gp = nn::sinusoidal_timesteps(g, self.channels, g.device())?;
            emb = (emb + ge.forward(&gp.to_dtype(g.dtype())?)?)?;
        }
        Ok((emb + self.text.forward(pooled)?)?)
    }
}

struct FluxTransformerBlock {
    norm1: AdaLayerNormZero,
    norm1_context: AdaLayerNormZero,
    attn: Flux1Attention,
    ff: GeluFeedForward,
    ff_context: GeluFeedForward,
    eps: f64,
}

impl FluxTransformerBlock {
    fn load(cfg: &Flux1ArchConfig, vb: VarBuilder) -> Result<Self> {
        let dim = cfg.hidden_size();
        Ok(Self {
            norm1: AdaLayerNormZero::load(dim, cfg.eps as f64, vb.pp("norm1"))?,
            norm1_context: AdaLayerNormZero::load(dim, cfg.eps as f64, vb.pp("norm1_context"))?,
            attn: Flux1Attention::load(
                dim,
                cfg.num_attention_heads,
                cfg.attention_head_dim,
                cfg.eps as f64,
                true,
                false,
                vb.pp("attn"),
            )?,
            ff: GeluFeedForward::load(dim, cfg.mlp_hidden(), vb.pp("ff"))?,
            ff_context: GeluFeedForward::load(dim, cfg.mlp_hidden(), vb.pp("ff_context"))?,
            eps: cfg.eps as f64,
        })
    }

    fn forward(
        &self,
        hidden: &Tensor,
        encoder: &Tensor,
        temb: &Tensor,
        rope: Option<&(Tensor, Tensor)>,
    ) -> Result<(Tensor, Tensor)> {
        let (img_n, gate_msa, shift_mlp, scale_mlp, gate_mlp) = self.norm1.forward(hidden, temb)?;
        let (txt_n, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp) =
            self.norm1_context.forward(encoder, temb)?;
        let (img_a, txt_a) = self.attn.forward(&img_n, Some(&txt_n), rope)?;
        let txt_a = txt_a.ok_or_else(|| candle_core::Error::Msg("joint attn dropped text".into()))?;
        let hidden = (hidden + img_a.broadcast_mul(&gate_msa.unsqueeze(1)?)?)?;
        let encoder = (encoder + txt_a.broadcast_mul(&c_gate_msa.unsqueeze(1)?)?)?;
        let img_n = nn::layer_norm(&hidden, self.eps, None, None)?;
        let img_n = img_n
            .broadcast_mul(&(scale_mlp.unsqueeze(1)? + 1.0)?)?
            .broadcast_add(&shift_mlp.unsqueeze(1)?)?;
        let hidden = (hidden + self.ff.forward(&img_n)?.broadcast_mul(&gate_mlp.unsqueeze(1)?)?)?;
        let txt_n = nn::layer_norm(&encoder, self.eps, None, None)?;
        let txt_n = txt_n
            .broadcast_mul(&(c_scale_mlp.unsqueeze(1)? + 1.0)?)?
            .broadcast_add(&c_shift_mlp.unsqueeze(1)?)?;
        let encoder = (encoder + self.ff_context.forward(&txt_n)?.broadcast_mul(&c_gate_mlp.unsqueeze(1)?)?)?;
        Ok((encoder, hidden))
    }
}

struct FluxSingleTransformerBlock {
    norm: AdaLayerNormZeroSingle,
    attn: Flux1Attention,
    proj_mlp: Linear,
    proj_out: Linear,
}

impl FluxSingleTransformerBlock {
    fn load(cfg: &Flux1ArchConfig, vb: VarBuilder) -> Result<Self> {
        let dim = cfg.hidden_size();
        let mlp = cfg.mlp_hidden();
        Ok(Self {
            norm: AdaLayerNormZeroSingle::load(dim, cfg.eps as f64, vb.pp("norm"))?,
            attn: Flux1Attention::load(
                dim,
                cfg.num_attention_heads,
                cfg.attention_head_dim,
                cfg.eps as f64,
                false,
                true,
                vb.pp("attn"),
            )?,
            proj_mlp: Linear::load(dim, mlp, vb.pp("proj_mlp"))?,
            proj_out: Linear::load(dim + mlp, dim, vb.pp("proj_out"))?,
        })
    }

    fn forward(
        &self,
        hidden: &Tensor,
        encoder: &Tensor,
        temb: &Tensor,
        rope: Option<&(Tensor, Tensor)>,
    ) -> Result<(Tensor, Tensor)> {
        let text_len = encoder.dim(1)?;
        let cat = Tensor::cat(&[encoder, hidden], 1)?;
        let (normed, gate) = self.norm.forward(&cat, temb)?;
        let mlp = nn::gelu_tanh(&self.proj_mlp.forward(&normed)?)?;
        let (attn, _) = self.attn.forward(&normed, None, rope)?;
        let y = self.proj_out.forward(&Tensor::cat(&[&attn, &mlp], D::Minus1)?)?;
        let y = (cat + y.broadcast_mul(&gate.unsqueeze(1)?)?)?;
        let encoder = y.narrow(1, 0, text_len)?;
        let hidden = y.narrow(1, text_len, y.dim(1)? - text_len)?;
        Ok((encoder, hidden))
    }
}

pub struct Flux1Transformer2D {
    pub cfg: Flux1ArchConfig,
    time_text: CombinedTimestepGuidanceTextProj,
    x_embedder: Linear,
    context_embedder: Linear,
    blocks: Vec<FluxTransformerBlock>,
    single_blocks: Vec<FluxSingleTransformerBlock>,
    norm_out: AdaLayerNormContinuous,
    proj_out: Linear,
}

impl Flux1Transformer2D {
    pub fn load(cfg: Flux1ArchConfig, vb: VarBuilder) -> Result<Self> {
        let dim = cfg.hidden_size();
        let mut blocks = Vec::with_capacity(cfg.num_layers);
        for i in 0..cfg.num_layers {
            blocks.push(FluxTransformerBlock::load(&cfg, vb.pp("transformer_blocks").pp(i))?);
        }
        let mut single_blocks = Vec::with_capacity(cfg.num_single_layers);
        for i in 0..cfg.num_single_layers {
            single_blocks.push(FluxSingleTransformerBlock::load(
                &cfg,
                vb.pp("single_transformer_blocks").pp(i),
            )?);
        }
        Ok(Self {
            time_text: CombinedTimestepGuidanceTextProj::load(
                cfg.timestep_guidance_channels,
                dim,
                cfg.pooled_projection_dim,
                cfg.guidance_embeds,
                vb.pp("time_text_embed"),
            )?,
            x_embedder: Linear::load(cfg.in_channels, dim, vb.pp("x_embedder"))?,
            context_embedder: Linear::load(cfg.joint_attention_dim, dim, vb.pp("context_embedder"))?,
            blocks,
            single_blocks,
            norm_out: AdaLayerNormContinuous::load(dim, cfg.eps as f64, vb.pp("norm_out"))?,
            proj_out: Linear::load(dim, cfg.patch_size * cfg.patch_size * cfg.out_channels, vb.pp("proj_out"))?,
            cfg,
        })
    }

    pub fn forward(
        &self,
        hidden: &Tensor,
        encoder: &Tensor,
        pooled: &Tensor,
        timestep: &Tensor,
        guidance: Option<&Tensor>,
        img_h: usize,
        img_w: usize,
    ) -> Result<Tensor> {
        let mut hidden = hidden.clone();
        let mut input_5d = None;
        if hidden.rank() == 5 {
            let (b, c, t, h, w) = hidden.dims5()?;
            input_5d = Some((b, t, h, w));
            hidden = hidden.permute((0, 2, 3, 4, 1))?.reshape((b, t * h * w, c))?;
        } else if hidden.rank() == 4 {
            let (b, c, h, w) = hidden.dims4()?;
            input_5d = Some((b, 1, h, w));
            hidden = hidden.permute((0, 2, 3, 1))?.reshape((b, h * w, c))?;
        }
        let timestep = (timestep.to_dtype(hidden.dtype())? * 1000.0)?;
        let guidance = match guidance {
            Some(g) => Some((g.to_dtype(hidden.dtype())? * 1000.0)?),
            None => None,
        };
        hidden = self.x_embedder.forward(&hidden)?;
        let temb = self
            .time_text
            .forward(&timestep, guidance.as_ref(), &pooled.to_dtype(hidden.dtype())?)?;
        let mut encoder = self.context_embedder.forward(encoder)?;
        let text_len = encoder.dim(1)?;
        let rope = flux1_rope(
            text_len,
            img_h,
            img_w,
            &self.cfg.axes_dims_rope,
            self.cfg.rope_theta,
            hidden.device(),
        )?;
        for block in &self.blocks {
            let (e, h) = block.forward(&hidden, &encoder, &temb, Some(&rope))?;
            encoder = e;
            hidden = h;
        }
        for block in &self.single_blocks {
            let (e, h) = block.forward(&hidden, &encoder, &temb, Some(&rope))?;
            encoder = e;
            hidden = h;
        }
        hidden = self.norm_out.forward(&hidden, &temb)?;
        let mut output = self.proj_out.forward(&hidden)?;
        if let Some((b, t, h, w)) = input_5d {
            output = output
                .reshape((b, t, h, w, self.cfg.out_channels))?
                .permute((0, 4, 1, 2, 3))?;
        }
        Ok(output)
    }
}

fn flux1_rope(
    text_len: usize,
    img_h: usize,
    img_w: usize,
    axes: &[usize; 3],
    theta: f32,
    device: &Device,
) -> Result<(Tensor, Tensor)> {
    let mut ids = text_ids(text_len);
    ids.extend(image_ids(img_h, img_w));
    let seq = ids.len() / 3;
    nd_rope(&ids, seq, axes, theta, device)
}

fn nd_rope(ids: &[f32], seq: usize, axes: &[usize; 3], theta: f32, device: &Device) -> Result<(Tensor, Tensor)> {
    let mut cos_parts = Vec::new();
    let mut sin_parts = Vec::new();
    for (axis, &dim) in axes.iter().enumerate() {
        let pos: Vec<f32> = (0..seq).map(|i| ids[i * 3 + axis]).collect();
        let (c, s) = rope_1d(&pos, dim, theta, device)?;
        cos_parts.push(c);
        sin_parts.push(s);
    }
    Ok((
        Tensor::cat(&cos_parts.iter().collect::<Vec<_>>(), D::Minus1)?,
        Tensor::cat(&sin_parts.iter().collect::<Vec<_>>(), D::Minus1)?,
    ))
}

fn rope_1d(pos: &[f32], dim: usize, theta: f32, device: &Device) -> Result<(Tensor, Tensor)> {
    let half = dim / 2;
    let inv: Vec<f32> = (0..half)
        .map(|i| 1.0 / theta.powf((2 * i) as f32 / dim as f32))
        .collect();
    let mut cos = vec![0.0f32; pos.len() * dim];
    let mut sin = vec![0.0f32; pos.len() * dim];
    for (p, &x) in pos.iter().enumerate() {
        for i in 0..half {
            let freq = x * inv[i];
            let c = freq.cos();
            let s = freq.sin();
            cos[p * dim + 2 * i] = c;
            cos[p * dim + 2 * i + 1] = c;
            sin[p * dim + 2 * i] = s;
            sin[p * dim + 2 * i + 1] = s;
        }
    }
    Ok((
        Tensor::from_vec(cos, (pos.len(), dim), device)?,
        Tensor::from_vec(sin, (pos.len(), dim), device)?,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tiny_forward_shapes() {
        let device = Device::Cpu;
        let cfg = Flux1ArchConfig::tiny();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let dit = Flux1Transformer2D::load(cfg.clone(), vb).unwrap();
        let hidden = Tensor::zeros((1, cfg.in_channels, 1, 2, 2), DType::F32, &device).unwrap();
        let enc = Tensor::zeros((1, 4, cfg.joint_attention_dim), DType::F32, &device).unwrap();
        let pooled = Tensor::zeros((1, cfg.pooled_projection_dim), DType::F32, &device).unwrap();
        let t = Tensor::from_vec(vec![0.5f32], (1,), &device).unwrap();
        let g = Tensor::from_vec(vec![3.5f32], (1,), &device).unwrap();
        let out = dit.forward(&hidden, &enc, &pooled, &t, Some(&g), 2, 2).unwrap();
        assert_eq!(out.dims(), &[1, cfg.out_channels, 1, 2, 2]);
    }
}
