//! Flux2Transformer2DModel matching Diffusers / FastVideo weight names.

use candle_core::{DType, Device, Result, Tensor, D};
use candle_nn::VarBuilder;

use crate::nn::{self, Linear};

use super::config::Flux2ArchConfig;
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

fn swiglu(xs: &Tensor) -> Result<Tensor> {
    let last = xs.dim(D::Minus1)?;
    if last % 2 != 0 {
        candle_core::bail!("swiglu last dim must be even, got {last}");
    }
    let x1 = xs.narrow(D::Minus1, 0, last / 2)?;
    let x2 = xs.narrow(D::Minus1, last / 2, last / 2)?;
    Ok((nn::silu(&x1)? * x2)?)
}

struct Flux2FeedForward {
    linear_in: Linear,
    linear_out: Linear,
}

impl Flux2FeedForward {
    fn load(dim: usize, inner: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            linear_in: Linear::load(dim, inner * 2, vb.pp("linear_in"))?,
            linear_out: Linear::load(inner, dim, vb.pp("linear_out"))?,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.linear_out.forward(&swiglu(&self.linear_in.forward(xs)?)?)
    }
}

struct Flux2Attention {
    to_q: Linear,
    to_k: Linear,
    to_v: Linear,
    to_out: Linear,
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

impl Flux2Attention {
    fn load(dim: usize, heads: usize, dim_head: usize, eps: f64, joint: bool, vb: VarBuilder) -> Result<Self> {
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
            to_out: Linear::load(inner, dim, vb.pp("to_out").pp("0"))?,
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
        let q = q.transpose(1, 2)?.contiguous()?;
        let k = k.transpose(1, 2)?.contiguous()?;
        let v = v.transpose(1, 2)?.contiguous()?;
        let attn = nn::scaled_dot_product_attention(&q, &k, &v, None)?;
        let (b, _, s, _) = attn.dims4()?;
        let attn = attn
            .transpose(1, 2)?
            .contiguous()?
            .reshape((b, s, self.heads * self.dim_head))?;
        if let (Some(text_len), Some(to_add)) = (text_len, &self.to_add_out) {
            let text = attn.narrow(1, 0, text_len)?;
            let img = attn.narrow(1, text_len, s - text_len)?;
            let img = self.to_out.forward(&img)?;
            let text = to_add.forward(&text)?;
            Ok((img, Some(text)))
        } else {
            Ok((self.to_out.forward(&attn)?, None))
        }
    }
}

fn apply_rotary(xs: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    // xs: [B, S, H, D], cos/sin: [S, D] fp32, pair-rotate last dim.
    let (b, s, h, d) = xs.dims4()?;
    let dtype = xs.dtype();
    let x = xs.to_dtype(DType::F32)?.contiguous()?.reshape((b, s, h, d / 2, 2))?;
    let even = x.narrow(4, 0, 1)?.squeeze(4)?;
    let odd = x.narrow(4, 1, 1)?.squeeze(4)?;
    let cos = broadcast_rope(cos, b, s, h, d)?;
    let sin = broadcast_rope(sin, b, s, h, d)?;
    let cos_e = cos.narrow(3, 0, d)?.reshape((b, s, h, d / 2, 2))?.narrow(4, 0, 1)?.squeeze(4)?;
    let sin_e = sin.narrow(3, 0, d)?.reshape((b, s, h, d / 2, 2))?.narrow(4, 0, 1)?.squeeze(4)?;
    // Diffusers use_real_unbind_dim=-1: rotate (x1, x2) by (cos, sin) with
    // interleave: y1 = x1*cos - x2*sin, y2 = x1*sin + x2*cos.
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

struct Flux2ParallelSelfAttention {
    to_qkv_mlp: Linear,
    to_out: Linear,
    norm_q: RmsNorm,
    norm_k: RmsNorm,
    heads: usize,
    dim_head: usize,
    inner: usize,
    _mlp_hidden: usize,
}

impl Flux2ParallelSelfAttention {
    fn load(dim: usize, heads: usize, dim_head: usize, mlp_hidden: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        let inner = heads * dim_head;
        Ok(Self {
            to_qkv_mlp: Linear::load(dim, inner * 3 + mlp_hidden * 2, vb.pp("to_qkv_mlp_proj"))?,
            to_out: Linear::load(inner + mlp_hidden, dim, vb.pp("to_out"))?,
            norm_q: RmsNorm::load(dim_head, eps, vb.pp("norm_q"))?,
            norm_k: RmsNorm::load(dim_head, eps, vb.pp("norm_k"))?,
            heads,
            dim_head,
            inner,
            _mlp_hidden: mlp_hidden,
        })
    }

    fn rms_heads(&self, xs: &Tensor, norm: &RmsNorm) -> Result<Tensor> {
        let dims = xs.dims().to_vec();
        let last = *dims.last().unwrap();
        norm.forward(&xs.reshape(((), last))?)?.reshape(dims)
    }

    fn forward(&self, hidden: &Tensor, rope: Option<&(Tensor, Tensor)>) -> Result<Tensor> {
        let proj = self.to_qkv_mlp.forward(hidden)?;
        let last = proj.dim(D::Minus1)?;
        let qkv = proj.narrow(D::Minus1, 0, self.inner * 3)?;
        let mlp = proj.narrow(D::Minus1, self.inner * 3, last - self.inner * 3)?;
        let q = qkv.narrow(D::Minus1, 0, self.inner)?;
        let k = qkv.narrow(D::Minus1, self.inner, self.inner)?;
        let v = qkv.narrow(D::Minus1, self.inner * 2, self.inner)?;
        let (b, s, _) = q.dims3()?;
        let mut q = self.rms_heads(&q.reshape((b, s, self.heads, self.dim_head))?, &self.norm_q)?;
        let mut k = self.rms_heads(&k.reshape((b, s, self.heads, self.dim_head))?, &self.norm_k)?;
        let v = v.reshape((b, s, self.heads, self.dim_head))?;
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
        let attn = attn
            .transpose(1, 2)?
            .contiguous()?
            .reshape((b, s, self.inner))?;
        let mlp = swiglu(&mlp)?;
        let cat = Tensor::cat(&[&attn, &mlp], D::Minus1)?;
        self.to_out.forward(&cat)
    }
}

struct Flux2TransformerBlock {
    attn: Flux2Attention,
    ff: Flux2FeedForward,
    ff_context: Flux2FeedForward,
    eps: f64,
}

impl Flux2TransformerBlock {
    fn load(cfg: &Flux2ArchConfig, vb: VarBuilder) -> Result<Self> {
        let dim = cfg.hidden_size();
        Ok(Self {
            attn: Flux2Attention::load(
                dim,
                cfg.num_attention_heads,
                cfg.attention_head_dim,
                cfg.eps as f64,
                true,
                vb.pp("attn"),
            )?,
            ff: Flux2FeedForward::load(dim, cfg.mlp_hidden(), vb.pp("ff"))?,
            ff_context: Flux2FeedForward::load(dim, cfg.mlp_hidden(), vb.pp("ff_context"))?,
            eps: cfg.eps as f64,
        })
    }

    fn modulate(xs: &Tensor, shift: &Tensor, scale: &Tensor, eps: f64) -> Result<Tensor> {
        let n = nn::layer_norm(xs, eps, None, None)?;
        n.broadcast_mul(&(scale + 1.0)?)?.broadcast_add(shift)
    }

    fn forward(
        &self,
        hidden: &Tensor,
        encoder: &Tensor,
        img_mod: &[(Tensor, Tensor, Tensor); 2],
        txt_mod: &[(Tensor, Tensor, Tensor); 2],
        rope: Option<&(Tensor, Tensor)>,
    ) -> Result<(Tensor, Tensor)> {
        let (shift_msa, scale_msa, gate_msa) = &img_mod[0];
        let (shift_mlp, scale_mlp, gate_mlp) = &img_mod[1];
        let (c_shift_msa, c_scale_msa, c_gate_msa) = &txt_mod[0];
        let (c_shift_mlp, c_scale_mlp, c_gate_mlp) = &txt_mod[1];
        let img_n = Self::modulate(hidden, shift_msa, scale_msa, self.eps)?;
        let txt_n = Self::modulate(encoder, c_shift_msa, c_scale_msa, self.eps)?;
        let (img_a, txt_a) = self.attn.forward(&img_n, Some(&txt_n), rope)?;
        let txt_a = txt_a.ok_or_else(|| candle_core::Error::Msg("joint attn dropped text".into()))?;
        let hidden = (hidden + img_a.broadcast_mul(gate_msa)?)?;
        let encoder = (encoder + txt_a.broadcast_mul(c_gate_msa)?)?;
        let img_n = Self::modulate(&hidden, shift_mlp, scale_mlp, self.eps)?;
        let hidden = (hidden + self.ff.forward(&img_n)?.broadcast_mul(gate_mlp)?)?;
        let txt_n = Self::modulate(&encoder, c_shift_mlp, c_scale_mlp, self.eps)?;
        let encoder = (encoder + self.ff_context.forward(&txt_n)?.broadcast_mul(c_gate_mlp)?)?;
        Ok((encoder, hidden))
    }
}

struct Flux2SingleTransformerBlock {
    attn: Flux2ParallelSelfAttention,
    eps: f64,
}

impl Flux2SingleTransformerBlock {
    fn load(cfg: &Flux2ArchConfig, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            attn: Flux2ParallelSelfAttention::load(
                cfg.hidden_size(),
                cfg.num_attention_heads,
                cfg.attention_head_dim,
                cfg.mlp_hidden(),
                cfg.eps as f64,
                vb.pp("attn"),
            )?,
            eps: cfg.eps as f64,
        })
    }

    fn forward(
        &self,
        hidden: &Tensor,
        shift: &Tensor,
        scale: &Tensor,
        gate: &Tensor,
        rope: Option<&(Tensor, Tensor)>,
    ) -> Result<Tensor> {
        let n = nn::layer_norm(hidden, self.eps, None, None)?;
        let n = n.broadcast_mul(&(scale + 1.0)?)?.broadcast_add(shift)?;
        hidden + self.attn.forward(&n, rope)?.broadcast_mul(gate)?
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

struct Flux2TimestepGuidanceEmbeddings {
    timestep: TimestepEmbedding,
    guidance: Option<TimestepEmbedding>,
    channels: usize,
}

impl Flux2TimestepGuidanceEmbeddings {
    fn load(channels: usize, hidden: usize, with_guidance: bool, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            timestep: TimestepEmbedding::load(channels, hidden, vb.pp("timestep_embedder"))?,
            guidance: if with_guidance {
                Some(TimestepEmbedding::load(channels, hidden, vb.pp("guidance_embedder"))?)
            } else {
                None
            },
            channels,
        })
    }

    fn forward(&self, timestep: &Tensor, guidance: Option<&Tensor>) -> Result<Tensor> {
        let proj = nn::sinusoidal_timesteps(timestep, self.channels, timestep.device())?;
        let mut emb = self.timestep.forward(&proj.to_dtype(timestep.dtype())?)?;
        if let (Some(g), Some(ge)) = (guidance, &self.guidance) {
            let gp = nn::sinusoidal_timesteps(g, self.channels, g.device())?;
            emb = (emb + ge.forward(&gp.to_dtype(g.dtype())?)?)?;
        }
        Ok(emb)
    }
}

struct Flux2Modulation {
    linear: Linear,
    sets: usize,
}

impl Flux2Modulation {
    fn load(dim: usize, sets: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            linear: Linear::load(dim, dim * 3 * sets, vb.pp("linear"))?,
            sets,
        })
    }

    fn forward(&self, temb: &Tensor) -> Result<Vec<(Tensor, Tensor, Tensor)>> {
        let mut modu = self.linear.forward(&nn::silu(temb)?)?;
        if modu.rank() == 2 {
            modu = modu.unsqueeze(1)?;
        }
        let last = modu.dim(D::Minus1)?;
        let chunk = last / (3 * self.sets);
        let mut out = Vec::with_capacity(self.sets);
        for i in 0..self.sets {
            let base = i * 3 * chunk;
            out.push((
                modu.narrow(D::Minus1, base, chunk)?,
                modu.narrow(D::Minus1, base + chunk, chunk)?,
                modu.narrow(D::Minus1, base + 2 * chunk, chunk)?,
            ));
        }
        Ok(out)
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

pub struct Flux2Transformer2D {
    pub cfg: Flux2ArchConfig,
    time_guidance: Flux2TimestepGuidanceEmbeddings,
    double_mod_img: Flux2Modulation,
    double_mod_txt: Flux2Modulation,
    single_mod: Flux2Modulation,
    x_embedder: Linear,
    context_embedder: Linear,
    blocks: Vec<Flux2TransformerBlock>,
    single_blocks: Vec<Flux2SingleTransformerBlock>,
    norm_out: AdaLayerNormContinuous,
    proj_out: Linear,
}

impl Flux2Transformer2D {
    pub fn load(cfg: Flux2ArchConfig, vb: VarBuilder) -> Result<Self> {
        let dim = cfg.hidden_size();
        let mut blocks = Vec::with_capacity(cfg.num_layers);
        for i in 0..cfg.num_layers {
            blocks.push(Flux2TransformerBlock::load(&cfg, vb.pp("transformer_blocks").pp(i))?);
        }
        let mut single_blocks = Vec::with_capacity(cfg.num_single_layers);
        for i in 0..cfg.num_single_layers {
            single_blocks.push(Flux2SingleTransformerBlock::load(
                &cfg,
                vb.pp("single_transformer_blocks").pp(i),
            )?);
        }
        Ok(Self {
            time_guidance: Flux2TimestepGuidanceEmbeddings::load(
                cfg.timestep_guidance_channels,
                dim,
                cfg.guidance_embeds,
                vb.pp("time_guidance_embed"),
            )?,
            double_mod_img: Flux2Modulation::load(dim, 2, vb.pp("double_stream_modulation_img"))?,
            double_mod_txt: Flux2Modulation::load(dim, 2, vb.pp("double_stream_modulation_txt"))?,
            single_mod: Flux2Modulation::load(dim, 1, vb.pp("single_stream_modulation"))?,
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
        }
        let timestep = (timestep.to_dtype(hidden.dtype())? * 1000.0)?;
        let guidance = match guidance {
            Some(g) => Some((g.to_dtype(hidden.dtype())? * 1000.0)?),
            None => None,
        };
        let temb = self.time_guidance.forward(&timestep, guidance.as_ref())?;
        let img_mod = self.double_mod_img.forward(&temb)?;
        let txt_mod = self.double_mod_txt.forward(&temb)?;
        let single_mod = self.single_mod.forward(&temb)?;
        let img_mod_arr = [
            img_mod[0].clone(),
            img_mod[1].clone(),
        ];
        let txt_mod_arr = [
            txt_mod[0].clone(),
            txt_mod[1].clone(),
        ];
        hidden = self.x_embedder.forward(&hidden)?;
        let mut encoder = self.context_embedder.forward(encoder)?;
        let text_len = encoder.dim(1)?;
        let img_len = hidden.dim(1)?;
        let rope = flux2_rope(
            text_len,
            img_h,
            img_w,
            &self.cfg.axes_dims_rope,
            self.cfg.rope_theta,
            hidden.device(),
        )?;
        for block in &self.blocks {
            let (e, h) = block.forward(&hidden, &encoder, &img_mod_arr, &txt_mod_arr, Some(&rope))?;
            encoder = e;
            hidden = h;
        }
        hidden = Tensor::cat(&[&encoder, &hidden], 1)?;
        let (shift, scale, gate) = &single_mod[0];
        for block in &self.single_blocks {
            hidden = block.forward(&hidden, shift, scale, gate, Some(&rope))?;
        }
        hidden = hidden.narrow(1, text_len, img_len)?;
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

fn flux2_rope(
    text_len: usize,
    img_h: usize,
    img_w: usize,
    axes: &[usize; 4],
    theta: f32,
    device: &Device,
) -> Result<(Tensor, Tensor)> {
    let txt = text_ids(text_len);
    let img = image_ids(1, img_h, img_w);
    let mut ids = txt;
    ids.extend(img);
    let seq = ids.len() / 4;
    let (cos, sin) = nd_rope(&ids, seq, axes, theta, device)?;
    Ok((cos, sin))
}

fn nd_rope(ids: &[f32], seq: usize, axes: &[usize; 4], theta: f32, device: &Device) -> Result<(Tensor, Tensor)> {
    let mut cos_parts = Vec::new();
    let mut sin_parts = Vec::new();
    for (axis, &dim) in axes.iter().enumerate() {
        let pos: Vec<f32> = (0..seq).map(|i| ids[i * 4 + axis]).collect();
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
            // Diffusers Flux2: repeat_interleave(2) so pairs share the same freq.
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
        let cfg = Flux2ArchConfig::tiny();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let dit = Flux2Transformer2D::load(cfg.clone(), vb).unwrap();
        let hidden = Tensor::zeros((1, cfg.in_channels, 1, 2, 2), DType::F32, &device).unwrap();
        let enc = Tensor::zeros((1, 4, cfg.joint_attention_dim), DType::F32, &device).unwrap();
        let t = Tensor::from_vec(vec![0.5f32], (1,), &device).unwrap();
        let g = Tensor::from_vec(vec![4.0f32], (1,), &device).unwrap();
        let out = dit.forward(&hidden, &enc, &t, Some(&g), 2, 2).unwrap();
        assert_eq!(out.dims(), &[1, cfg.out_channels, 1, 2, 2]);
    }
}
