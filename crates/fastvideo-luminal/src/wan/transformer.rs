//! WanTransformer3D on host NdTensor (matches Candle tiny shapes).

use fastvideo_models::wan::WanVideoArchConfig;

use super::nn::{self, Linear};
use super::tensor::{NdTensor, Result, TensorError};
use super::weights::{self, WeightMap};

#[derive(Debug, Clone)]
struct RmsNorm {
    weight: NdTensor,
    eps: f32,
}

impl RmsNorm {
    fn zeros(dim: usize, eps: f32) -> Self {
        Self {
            weight: NdTensor::ones(&[dim]),
            eps,
        }
    }

    fn load(map: &WeightMap, prefix: &str, dim: usize, eps: f32) -> Result<Self> {
        Ok(Self {
            weight: weights::nd_tensor_shaped(map, &weights::join_key(prefix, "weight"), &[dim])?,
            eps,
        })
    }

    fn forward(&self, xs: &NdTensor) -> Result<NdTensor> {
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
    fn zeros(dim: usize, heads: usize, eps: f32) -> Self {
        Self {
            to_q: Linear::zeros(dim, dim, true),
            to_k: Linear::zeros(dim, dim, true),
            to_v: Linear::zeros(dim, dim, true),
            to_out: Linear::zeros(dim, dim, true),
            norm_q: RmsNorm::zeros(dim, eps),
            norm_k: RmsNorm::zeros(dim, eps),
            heads,
            dim_head: dim / heads,
        }
    }

    fn load(map: &WeightMap, prefix: &str, dim: usize, heads: usize, eps: f32) -> Result<Self> {
        Ok(Self {
            to_q: Linear::load(map, &weights::join_key(prefix, "to_q"), dim, dim, true)?,
            to_k: Linear::load(map, &weights::join_key(prefix, "to_k"), dim, dim, true)?,
            to_v: Linear::load(map, &weights::join_key(prefix, "to_v"), dim, dim, true)?,
            to_out: Linear::load(map, &weights::join_key(prefix, "to_out.0"), dim, dim, true)?,
            norm_q: RmsNorm::load(map, &weights::join_key(prefix, "norm_q"), dim, eps)?,
            norm_k: RmsNorm::load(map, &weights::join_key(prefix, "norm_k"), dim, eps)?,
            heads,
            dim_head: dim / heads,
        })
    }

    fn forward(
        &self,
        hidden: &NdTensor,
        encoder: Option<&NdTensor>,
        rotary: Option<&(NdTensor, NdTensor)>,
    ) -> Result<NdTensor> {
        let ctx = encoder.unwrap_or(hidden);
        let q = self.norm_q.forward(&self.to_q.forward(hidden)?)?;
        let mut k = self.norm_k.forward(&self.to_k.forward(ctx)?)?;
        let mut v = self.to_v.forward(ctx)?;
        let (b, sq, _) = (q.shape[0], q.shape[1], q.shape[2]);
        let sk = k.shape[1];
        let mut q = q.reshape(vec![b, sq, self.heads, self.dim_head])?;
        k = k.reshape(vec![b, sk, self.heads, self.dim_head])?;
        v = v.reshape(vec![b, sk, self.heads, self.dim_head])?;
        if let Some((cos, sin)) = rotary {
            q = apply_rotary(&q, cos, sin)?;
            k = apply_rotary(&k, cos, sin)?;
        }
        let q = q.transpose(1, 2)?;
        let k = k.transpose(1, 2)?;
        let v = v.transpose(1, 2)?;
        let attn = nn::scaled_dot_product_attention(&q, &k, &v, None)?;
        let attn = attn
            .transpose(1, 2)?
            .reshape(vec![b, sq, self.heads * self.dim_head])?;
        self.to_out.forward(&attn)
    }
}

fn pair_last_dim(xs: &NdTensor) -> Result<(NdTensor, NdTensor)> {
    let mut dims = xs.shape.clone();
    let d = dims.pop().ok_or_else(|| TensorError::Message("empty rotary".into()))?;
    if d % 2 != 0 {
        return Err(TensorError::Message("rotary last dim must be even".into()));
    }
    dims.push(d / 2);
    dims.push(2);
    let xs = xs.reshape(dims)?;
    let rank = xs.rank();
    let even = xs.narrow(rank - 1, 0, 1)?.squeeze(rank - 1)?;
    let odd = xs.narrow(rank - 1, 1, 1)?.squeeze(rank - 1)?;
    Ok((even, odd))
}

fn apply_rotary(xs: &NdTensor, cos: &NdTensor, sin: &NdTensor) -> Result<NdTensor> {
    let (x1, x2) = pair_last_dim(xs)?;
    let (cos_e, _) = pair_last_dim(cos)?;
    let (_, sin_o) = pair_last_dim(sin)?;
    let out1 = x1.mul(&cos_e)?.sub(&x2.mul(&sin_o)?)?;
    let out2 = x1.mul(&sin_o)?.add(&x2.mul(&cos_e)?)?;
    let out1 = out1.unsqueeze(out1.rank())?;
    let out2 = out2.unsqueeze(out2.rank())?;
    let stacked = NdTensor::cat(&[&out1, &out2], out1.rank() - 1)?;
    let mut out_dims = stacked.shape.clone();
    let pair = out_dims.pop().unwrap_or(2);
    let half = out_dims.pop().unwrap_or(0);
    out_dims.push(half * pair);
    stacked.reshape(out_dims)
}

fn rotary_1d(dim: usize, seq: usize, theta: f64) -> Result<(NdTensor, NdTensor)> {
    let half = dim / 2;
    let mut cos = vec![0.0f32; seq * dim];
    let mut sin = vec![0.0f32; seq * dim];
    for p in 0..seq {
        for i in 0..half {
            let freq = 1.0 / theta.powf(2.0 * i as f64 / dim as f64) as f32;
            let arg = p as f32 * freq;
            // repeat_interleave 2
            cos[p * dim + 2 * i] = arg.cos();
            cos[p * dim + 2 * i + 1] = arg.cos();
            sin[p * dim + 2 * i] = arg.sin();
            sin[p * dim + 2 * i + 1] = arg.sin();
        }
    }
    Ok((
        NdTensor::from_vec(cos, vec![seq, dim])?,
        NdTensor::from_vec(sin, vec![seq, dim])?,
    ))
}

fn wan_rope(
    cfg: &WanVideoArchConfig,
    frames: usize,
    height: usize,
    width: usize,
) -> Result<(NdTensor, NdTensor)> {
    let d = cfg.attention_head_dim;
    let h_dim = 2 * (d / 6);
    let w_dim = h_dim;
    let t_dim = d - h_dim - w_dim;
    let (cos_t, sin_t) = rotary_1d(t_dim, cfg.rope_max_seq_len, 10000.0)?;
    let (cos_h, sin_h) = rotary_1d(h_dim, cfg.rope_max_seq_len, 10000.0)?;
    let (cos_w, sin_w) = rotary_1d(w_dim, cfg.rope_max_seq_len, 10000.0)?;
    let ppf = frames / cfg.patch_size[0];
    let pph = height / cfg.patch_size[1];
    let ppw = width / cfg.patch_size[2];
    let seq = ppf * pph * ppw;
    let mut cos = vec![0.0f32; seq * d];
    let mut sin = vec![0.0f32; seq * d];
    let mut idx = 0usize;
    for ft in 0..ppf {
        for fh in 0..pph {
            for fw in 0..ppw {
                let mut o = 0usize;
                for i in 0..t_dim {
                    cos[idx * d + o] = cos_t.data[ft * t_dim + i];
                    sin[idx * d + o] = sin_t.data[ft * t_dim + i];
                    o += 1;
                }
                for i in 0..h_dim {
                    cos[idx * d + o] = cos_h.data[fh * h_dim + i];
                    sin[idx * d + o] = sin_h.data[fh * h_dim + i];
                    o += 1;
                }
                for i in 0..w_dim {
                    cos[idx * d + o] = cos_w.data[fw * w_dim + i];
                    sin[idx * d + o] = sin_w.data[fw * w_dim + i];
                    o += 1;
                }
                idx += 1;
            }
        }
    }
    Ok((
        NdTensor::from_vec(cos, vec![1, seq, 1, d])?,
        NdTensor::from_vec(sin, vec![1, seq, 1, d])?,
    ))
}

#[derive(Debug, Clone)]
struct FeedForward {
    proj: Linear,
    out: Linear,
}

impl FeedForward {
    fn zeros(dim: usize, ffn_dim: usize) -> Self {
        Self {
            proj: Linear::zeros(dim, ffn_dim, true),
            out: Linear::zeros(ffn_dim, dim, true),
        }
    }

    fn load(map: &WeightMap, prefix: &str, dim: usize, ffn_dim: usize) -> Result<Self> {
        Ok(Self {
            proj: Linear::load(map, &weights::join_key(prefix, "net.0.proj"), dim, ffn_dim, true)?,
            out: Linear::load(map, &weights::join_key(prefix, "net.2"), ffn_dim, dim, true)?,
        })
    }

    fn forward(&self, xs: &NdTensor) -> Result<NdTensor> {
        let h = nn::gelu_tanh(&self.proj.forward(xs)?);
        self.out.forward(&h)
    }
}

#[derive(Debug, Clone)]
struct TextProjection {
    linear_1: Linear,
    linear_2: Linear,
}

impl TextProjection {
    fn zeros(in_dim: usize, dim: usize) -> Self {
        Self {
            linear_1: Linear::zeros(in_dim, dim, true),
            linear_2: Linear::zeros(dim, dim, true),
        }
    }

    fn load(map: &WeightMap, prefix: &str, in_dim: usize, dim: usize) -> Result<Self> {
        Ok(Self {
            linear_1: Linear::load(map, &weights::join_key(prefix, "linear_1"), in_dim, dim, true)?,
            linear_2: Linear::load(map, &weights::join_key(prefix, "linear_2"), dim, dim, true)?,
        })
    }

    fn forward(&self, xs: &NdTensor) -> Result<NdTensor> {
        let h = nn::gelu_tanh(&self.linear_1.forward(xs)?);
        self.linear_2.forward(&h)
    }
}

#[derive(Debug, Clone)]
struct TimestepEmbedding {
    linear_1: Linear,
    linear_2: Linear,
}

impl TimestepEmbedding {
    fn zeros(in_dim: usize, dim: usize) -> Self {
        Self {
            linear_1: Linear::zeros(in_dim, dim, true),
            linear_2: Linear::zeros(dim, dim, true),
        }
    }

    fn load(map: &WeightMap, prefix: &str, in_dim: usize, dim: usize) -> Result<Self> {
        Ok(Self {
            linear_1: Linear::load(map, &weights::join_key(prefix, "linear_1"), in_dim, dim, true)?,
            linear_2: Linear::load(map, &weights::join_key(prefix, "linear_2"), dim, dim, true)?,
        })
    }

    fn forward(&self, xs: &NdTensor) -> Result<NdTensor> {
        let h = nn::silu(&self.linear_1.forward(xs)?);
        self.linear_2.forward(&h)
    }
}

#[derive(Debug, Clone)]
struct WanBlock {
    norm1_eps: f32,
    attn1: WanAttention,
    attn2: WanAttention,
    norm2_weight: NdTensor,
    norm2_bias: NdTensor,
    ffn: FeedForward,
    scale_shift_table: NdTensor, // [1, 6, dim]
}

impl WanBlock {
    fn zeros(cfg: &WanVideoArchConfig) -> Self {
        let dim = cfg.hidden_size();
        Self {
            norm1_eps: cfg.eps,
            attn1: WanAttention::zeros(dim, cfg.num_attention_heads, cfg.eps),
            attn2: WanAttention::zeros(dim, cfg.num_attention_heads, cfg.eps),
            norm2_weight: NdTensor::ones(&[dim]),
            norm2_bias: NdTensor::zeros(&[dim]),
            ffn: FeedForward::zeros(dim, cfg.ffn_dim),
            scale_shift_table: NdTensor::zeros(&[1, 6, dim]),
        }
    }

    fn load(map: &WeightMap, prefix: &str, cfg: &WanVideoArchConfig) -> Result<Self> {
        let dim = cfg.hidden_size();
        Ok(Self {
            norm1_eps: cfg.eps,
            attn1: WanAttention::load(
                map,
                &weights::join_key(prefix, "attn1"),
                dim,
                cfg.num_attention_heads,
                cfg.eps,
            )?,
            attn2: WanAttention::load(
                map,
                &weights::join_key(prefix, "attn2"),
                dim,
                cfg.num_attention_heads,
                cfg.eps,
            )?,
            norm2_weight: weights::nd_tensor_shaped(
                map,
                &weights::join_key(prefix, "norm2.weight"),
                &[dim],
            )?,
            norm2_bias: weights::nd_tensor_shaped(
                map,
                &weights::join_key(prefix, "norm2.bias"),
                &[dim],
            )?,
            ffn: FeedForward::load(map, &weights::join_key(prefix, "ffn"), dim, cfg.ffn_dim)?,
            scale_shift_table: weights::nd_tensor_shaped(
                map,
                &weights::join_key(prefix, "scale_shift_table"),
                &[1, 6, dim],
            )?,
        })
    }

    fn forward(
        &self,
        hidden: &NdTensor,
        encoder: &NdTensor,
        temb: &NdTensor,
        rotary: &(NdTensor, NdTensor),
    ) -> Result<NdTensor> {
        let e = self.scale_shift_table.add(temb)?;
        let chunks = e.chunk(6, 1)?;
        let shift_msa = &chunks[0];
        let scale_msa = &chunks[1];
        let gate_msa = &chunks[2];
        let c_shift = &chunks[3];
        let c_scale = &chunks[4];
        let c_gate = &chunks[5];

        let normed = nn::layer_norm(hidden, self.norm1_eps, None, None)?;
        let normed = normed
            .mul(&scale_msa.add_scalar(1.0))?
            .add(shift_msa)?;
        let attn = self.attn1.forward(&normed, None, Some(rotary))?;
        let hidden = hidden.add(&attn.mul(gate_msa)?)?;

        let normed = nn::layer_norm(
            &hidden,
            self.norm1_eps,
            Some(&self.norm2_weight),
            Some(&self.norm2_bias),
        )?;
        let attn = self.attn2.forward(&normed, Some(encoder), None)?;
        let hidden = hidden.add(&attn)?;

        let normed = nn::layer_norm(&hidden, self.norm1_eps, None, None)?;
        let normed = normed.mul(&c_scale.add_scalar(1.0))?.add(c_shift)?;
        let ff = self.ffn.forward(&normed)?;
        hidden.add(&ff.mul(c_gate)?)
    }
}

#[derive(Debug, Clone)]
pub struct WanTransformer3D {
    pub cfg: WanVideoArchConfig,
    patch_weight: NdTensor, // [dim, in_c, pt, ph, pw]
    patch_bias: NdTensor,
    time_embedder: TimestepEmbedding,
    time_proj: Linear,
    text_embedder: TextProjection,
    blocks: Vec<WanBlock>,
    proj_out: Linear,
    scale_shift_table: NdTensor, // [1, 2, dim]
    freq_dim: usize,
}

impl WanTransformer3D {
    pub fn zeros(cfg: WanVideoArchConfig) -> Self {
        let dim = cfg.hidden_size();
        let p = cfg.patch_size;
        let mut blocks = Vec::with_capacity(cfg.num_layers);
        for _ in 0..cfg.num_layers {
            blocks.push(WanBlock::zeros(&cfg));
        }
        Self {
            patch_weight: NdTensor::zeros(&[dim, cfg.in_channels, p[0], p[1], p[2]]),
            patch_bias: NdTensor::zeros(&[dim]),
            time_embedder: TimestepEmbedding::zeros(cfg.freq_dim, dim),
            time_proj: Linear::zeros(dim, dim * 6, true),
            text_embedder: TextProjection::zeros(cfg.text_dim, dim),
            proj_out: Linear::zeros(dim, cfg.out_channels * p.iter().product::<usize>(), true),
            scale_shift_table: NdTensor::zeros(&[1, 2, dim]),
            freq_dim: cfg.freq_dim,
            blocks,
            cfg,
        }
    }

    pub fn load(cfg: WanVideoArchConfig, map: &WeightMap) -> Result<Self> {
        Self::from_map(cfg, map)
    }

    pub fn from_map(cfg: WanVideoArchConfig, map: &WeightMap) -> Result<Self> {
        let dim = cfg.hidden_size();
        let p = cfg.patch_size;
        let mut blocks = Vec::with_capacity(cfg.num_layers);
        for i in 0..cfg.num_layers {
            blocks.push(WanBlock::load(map, &format!("blocks.{i}"), &cfg)?);
        }
        Ok(Self {
            patch_weight: weights::nd_tensor_shaped(
                map,
                "patch_embedding.weight",
                &[dim, cfg.in_channels, p[0], p[1], p[2]],
            )?,
            patch_bias: weights::nd_tensor_shaped(map, "patch_embedding.bias", &[dim])?,
            time_embedder: TimestepEmbedding::load(
                map,
                "condition_embedder.time_embedder",
                cfg.freq_dim,
                dim,
            )?,
            time_proj: Linear::load(map, "condition_embedder.time_proj", dim, dim * 6, true)?,
            text_embedder: TextProjection::load(
                map,
                "condition_embedder.text_embedder",
                cfg.text_dim,
                dim,
            )?,
            proj_out: Linear::load(
                map,
                "proj_out",
                dim,
                cfg.out_channels * p.iter().product::<usize>(),
                true,
            )?,
            scale_shift_table: weights::nd_tensor_shaped(map, "scale_shift_table", &[1, 2, dim])?,
            freq_dim: cfg.freq_dim,
            blocks,
            cfg,
        })
    }

    fn patch_embed(&self, xs: &NdTensor) -> Result<NdTensor> {
        // xs: [B, C, T, H, W]
        let (b, c, t, h, w) = (
            xs.shape[0],
            xs.shape[1],
            xs.shape[2],
            xs.shape[3],
            xs.shape[4],
        );
        let p = self.cfg.patch_size;
        let x = xs
            .permute(&[0, 2, 1, 3, 4])?
            .reshape(vec![b * t, c, h, w])?;
        let k = self
            .patch_weight
            .reshape(vec![self.cfg.hidden_size(), c * p[0], p[1], p[2]])?;
        let y = nn::conv2d(&x, &k, 0, p[1])?;
        let bias = self
            .patch_bias
            .reshape(vec![1, self.cfg.hidden_size(), 1, 1])?;
        let y = y.add(&bias)?;
        let (_, dim, hh, ww) = (y.shape[0], y.shape[1], y.shape[2], y.shape[3]);
        y.reshape(vec![b, t, dim, hh, ww])?
            .permute(&[0, 2, 1, 3, 4])?
            .flatten_from(2)?
            .transpose(1, 2)
    }

    pub fn forward(
        &self,
        latents: &NdTensor,
        timestep: &NdTensor,
        encoder: &NdTensor,
    ) -> Result<NdTensor> {
        let (b, _c, t, h, w) = (
            latents.shape[0],
            latents.shape[1],
            latents.shape[2],
            latents.shape[3],
            latents.shape[4],
        );
        let rotary = wan_rope(&self.cfg, t, h, w)?;
        let mut hidden = self.patch_embed(latents)?;
        let temb_in = nn::sinusoidal_timesteps(timestep, self.freq_dim)?;
        let temb = self.time_embedder.forward(&temb_in)?;
        let timestep_proj = self
            .time_proj
            .forward(&nn::silu(&temb))?
            .reshape(vec![b, 6, self.cfg.hidden_size()])?;
        let encoder = self.text_embedder.forward(encoder)?;
        for block in &self.blocks {
            hidden = block.forward(&hidden, &encoder, &timestep_proj, &rotary)?;
        }
        let temb_f = temb.unsqueeze(1)?;
        let ss = self.scale_shift_table.add(&temb_f)?;
        let chunks = ss.chunk(2, 1)?;
        let shift = &chunks[0];
        let scale = &chunks[1];
        hidden = nn::layer_norm(&hidden, self.cfg.eps, None, None)?;
        hidden = hidden.mul(&scale.add_scalar(1.0))?.add(shift)?;
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
            .permute(&[0, 7, 1, 4, 2, 5, 3, 6])?
            .reshape(vec![
                b,
                self.cfg.out_channels,
                ppf * p[0],
                pph * p[1],
                ppw * p[2],
            ])
    }
}
