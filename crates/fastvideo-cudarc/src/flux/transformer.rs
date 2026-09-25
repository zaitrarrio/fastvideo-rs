//! FluxTransformer2DModel on `CudaTensor` (FLUX.1: AdaLN-Zero, GELU, 3-axis RoPE).

use fastvideo_models::flux::Flux1ArchConfig;

use crate::wan::nn::{self, Linear};
use crate::wan::tensor::{CudaTensor, Result, TensorError};
use crate::wan::weights::{cuda_tensor_shaped, join_key, WeightMap};

fn msg(s: impl Into<String>) -> TensorError {
    TensorError::Message(s.into())
}

struct RmsNorm {
    weight: CudaTensor,
    eps: f32,
}

impl RmsNorm {
    fn zeros(dim: usize, eps: f32) -> Self {
        Self {
            weight: CudaTensor::from_vec(vec![1.0; dim], vec![dim]).expect("rms"),
            eps,
        }
    }

    fn load(map: &WeightMap, prefix: &str, dim: usize, eps: f32) -> Result<Self> {
        Ok(Self {
            weight: cuda_tensor_shaped(map, &join_key(prefix, "weight"), &[dim])?,
            eps,
        })
    }

    fn forward(&self, xs: &CudaTensor) -> Result<CudaTensor> {
        nn::rms_norm(xs, &self.weight, self.eps)
    }
}

fn rms_heads(xs: &CudaTensor, norm: &RmsNorm, heads: usize, dim_head: usize) -> Result<CudaTensor> {
    let b = xs.shape[0];
    let s = xs.shape[1];
    let shaped = xs.reshape(vec![b, s, heads, dim_head])?;
    let flat = shaped.reshape(vec![b * s * heads, dim_head])?;
    norm.forward(&flat)?.reshape(vec![b, s, heads, dim_head])
}

fn apply_rotary(xs: &CudaTensor, cos: &CudaTensor, sin: &CudaTensor) -> Result<CudaTensor> {
    xs.apply_rotary_bshd(cos, sin)
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
    fn zeros(dim: usize, heads: usize, dim_head: usize, eps: f32, joint: bool, pre_only: bool) -> Self {
        let inner = heads * dim_head;
        Self {
            to_q: Linear::zeros(dim, inner, true),
            to_k: Linear::zeros(dim, inner, true),
            to_v: Linear::zeros(dim, inner, true),
            to_out: (!pre_only).then(|| Linear::zeros(inner, dim, true)),
            norm_q: RmsNorm::zeros(dim_head, eps),
            norm_k: RmsNorm::zeros(dim_head, eps),
            add_q: joint.then(|| Linear::zeros(dim, inner, true)),
            add_k: joint.then(|| Linear::zeros(dim, inner, true)),
            add_v: joint.then(|| Linear::zeros(dim, inner, true)),
            to_add_out: joint.then(|| Linear::zeros(inner, dim, true)),
            norm_added_q: joint.then(|| RmsNorm::zeros(dim_head, eps)),
            norm_added_k: joint.then(|| RmsNorm::zeros(dim_head, eps)),
            heads,
            dim_head,
        }
    }

    fn load(
        map: &WeightMap,
        prefix: &str,
        dim: usize,
        heads: usize,
        dim_head: usize,
        eps: f32,
        joint: bool,
        pre_only: bool,
    ) -> Result<Self> {
        let inner = heads * dim_head;
        Ok(Self {
            to_q: Linear::load(map, &format!("{prefix}.to_q"), dim, inner, true)?,
            to_k: Linear::load(map, &format!("{prefix}.to_k"), dim, inner, true)?,
            to_v: Linear::load(map, &format!("{prefix}.to_v"), dim, inner, true)?,
            to_out: if pre_only {
                None
            } else {
                Some(Linear::load(map, &format!("{prefix}.to_out.0"), inner, dim, true)?)
            },
            norm_q: RmsNorm::load(map, &format!("{prefix}.norm_q"), dim_head, eps)?,
            norm_k: RmsNorm::load(map, &format!("{prefix}.norm_k"), dim_head, eps)?,
            add_q: if joint {
                Some(Linear::load(map, &format!("{prefix}.add_q_proj"), dim, inner, true)?)
            } else {
                None
            },
            add_k: if joint {
                Some(Linear::load(map, &format!("{prefix}.add_k_proj"), dim, inner, true)?)
            } else {
                None
            },
            add_v: if joint {
                Some(Linear::load(map, &format!("{prefix}.add_v_proj"), dim, inner, true)?)
            } else {
                None
            },
            to_add_out: if joint {
                Some(Linear::load(map, &format!("{prefix}.to_add_out"), inner, dim, true)?)
            } else {
                None
            },
            norm_added_q: if joint {
                Some(RmsNorm::load(map, &format!("{prefix}.norm_added_q"), dim_head, eps)?)
            } else {
                None
            },
            norm_added_k: if joint {
                Some(RmsNorm::load(map, &format!("{prefix}.norm_added_k"), dim_head, eps)?)
            } else {
                None
            },
            heads,
            dim_head,
        })
    }

    fn forward(
        &self,
        hidden: &CudaTensor,
        encoder: Option<&CudaTensor>,
        rope: &(CudaTensor, CudaTensor),
    ) -> Result<(CudaTensor, Option<CudaTensor>)> {
        let q = rms_heads(&self.to_q.forward(hidden)?, &self.norm_q, self.heads, self.dim_head)?;
        let k = rms_heads(&self.to_k.forward(hidden)?, &self.norm_k, self.heads, self.dim_head)?;
        let v = {
            let t = self.to_v.forward(hidden)?;
            t.reshape(vec![t.shape[0], t.shape[1], self.heads, self.dim_head])?
        };
        let (mut q, mut k, v, text_len) = if let (Some(enc), Some(aq), Some(ak), Some(av), Some(nq), Some(nk)) = (
            encoder,
            &self.add_q,
            &self.add_k,
            &self.add_v,
            &self.norm_added_q,
            &self.norm_added_k,
        ) {
            let eq = rms_heads(&aq.forward(enc)?, nq, self.heads, self.dim_head)?;
            let ek = rms_heads(&ak.forward(enc)?, nk, self.heads, self.dim_head)?;
            let ev = {
                let t = av.forward(enc)?;
                t.reshape(vec![t.shape[0], t.shape[1], self.heads, self.dim_head])?
            };
            (
                CudaTensor::cat(&[&eq, &q], 1)?,
                CudaTensor::cat(&[&ek, &k], 1)?,
                CudaTensor::cat(&[&ev, &v], 1)?,
                Some(enc.shape[1]),
            )
        } else {
            (q, k, v, None)
        };
        q = apply_rotary(&q, &rope.0, &rope.1)?;
        k = apply_rotary(&k, &rope.0, &rope.1)?;
        let attn = nn::scaled_dot_product_attention(
            &q.transpose(1, 2)?,
            &k.transpose(1, 2)?,
            &v.transpose(1, 2)?,
            None,
        )?;
        let attn = attn.transpose(1, 2)?;
        let b = attn.shape[0];
        let s = attn.shape[1];
        let attn = attn.reshape(vec![b, s, self.heads * self.dim_head])?;
        if let (Some(text_len), Some(to_add)) = (text_len, &self.to_add_out) {
            let text = attn.narrow(1, 0, text_len)?;
            let img = attn.narrow(1, text_len, s - text_len)?;
            let img = match &self.to_out {
                Some(proj) => proj.forward(&img)?,
                None => img,
            };
            Ok((img, Some(to_add.forward(&text)?)))
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
    fn zeros(dim: usize, inner: usize) -> Self {
        Self {
            proj: Linear::zeros(dim, inner, true),
            out: Linear::zeros(inner, dim, true),
        }
    }

    fn load(map: &WeightMap, prefix: &str, dim: usize, inner: usize) -> Result<Self> {
        Ok(Self {
            proj: Linear::load(map, &format!("{prefix}.net.0.proj"), dim, inner, true)?,
            out: Linear::load(map, &format!("{prefix}.net.2"), inner, dim, true)?,
        })
    }

    fn forward(&self, xs: &CudaTensor) -> Result<CudaTensor> {
        self.out.forward(&nn::gelu_tanh(&self.proj.forward(xs)?))
    }
}

/// Diffusers `AdaLayerNormZero`: 6 chunks (msa + mlp).
struct AdaLayerNormZero {
    linear: Linear,
    eps: f32,
}

impl AdaLayerNormZero {
    fn zeros(dim: usize, eps: f32) -> Self {
        Self {
            linear: Linear::zeros(dim, dim * 6, true),
            eps,
        }
    }

    fn load(map: &WeightMap, prefix: &str, dim: usize, eps: f32) -> Result<Self> {
        Ok(Self {
            linear: Linear::load(map, &format!("{prefix}.linear"), dim, dim * 6, true)?,
            eps,
        })
    }

    fn forward(
        &self,
        xs: &CudaTensor,
        emb: &CudaTensor,
    ) -> Result<(CudaTensor, CudaTensor, CudaTensor, CudaTensor, CudaTensor)> {
        let mut e = self.linear.forward(&emb.silu())?;
        if e.rank() == 2 {
            e = e.unsqueeze(1)?;
        }
        let last = *e.shape.last().unwrap();
        let chunk = last / 6;
        let shift_msa = e.narrow(e.rank() - 1, 0, chunk)?;
        let scale_msa = e.narrow(e.rank() - 1, chunk, chunk)?;
        let gate_msa = e.narrow(e.rank() - 1, 2 * chunk, chunk)?;
        let shift_mlp = e.narrow(e.rank() - 1, 3 * chunk, chunk)?;
        let scale_mlp = e.narrow(e.rank() - 1, 4 * chunk, chunk)?;
        let gate_mlp = e.narrow(e.rank() - 1, 5 * chunk, chunk)?;
        let n = xs.layer_norm(self.eps, None, None)?;
        let n = n.mul(&scale_msa.add_scalar(1.0))?.add(&shift_msa)?;
        Ok((n, gate_msa, shift_mlp, scale_mlp, gate_mlp))
    }
}

struct AdaLayerNormZeroSingle {
    linear: Linear,
    eps: f32,
}

impl AdaLayerNormZeroSingle {
    fn zeros(dim: usize, eps: f32) -> Self {
        Self {
            linear: Linear::zeros(dim, dim * 3, true),
            eps,
        }
    }

    fn load(map: &WeightMap, prefix: &str, dim: usize, eps: f32) -> Result<Self> {
        Ok(Self {
            linear: Linear::load(map, &format!("{prefix}.linear"), dim, dim * 3, true)?,
            eps,
        })
    }

    fn forward(&self, xs: &CudaTensor, emb: &CudaTensor) -> Result<(CudaTensor, CudaTensor)> {
        let mut e = self.linear.forward(&emb.silu())?;
        if e.rank() == 2 {
            e = e.unsqueeze(1)?;
        }
        let last = *e.shape.last().unwrap();
        let chunk = last / 3;
        let shift = e.narrow(e.rank() - 1, 0, chunk)?;
        let scale = e.narrow(e.rank() - 1, chunk, chunk)?;
        let gate = e.narrow(e.rank() - 1, 2 * chunk, chunk)?;
        let n = xs.layer_norm(self.eps, None, None)?;
        let n = n.mul(&scale.add_scalar(1.0))?.add(&shift)?;
        Ok((n, gate))
    }
}

struct AdaLayerNormContinuous {
    linear: Linear,
    eps: f32,
}

impl AdaLayerNormContinuous {
    fn zeros(dim: usize, eps: f32) -> Self {
        Self {
            linear: Linear::zeros(dim, dim * 2, true),
            eps,
        }
    }

    fn load(map: &WeightMap, prefix: &str, dim: usize, eps: f32) -> Result<Self> {
        Ok(Self {
            linear: Linear::load(map, &format!("{prefix}.linear"), dim, dim * 2, true)?,
            eps,
        })
    }

    fn forward(&self, xs: &CudaTensor, cond: &CudaTensor) -> Result<CudaTensor> {
        let mut emb = self.linear.forward(&cond.silu())?;
        if emb.rank() == 2 {
            emb = emb.unsqueeze(1)?;
        }
        let last = *emb.shape.last().unwrap();
        let scale = emb.narrow(emb.rank() - 1, 0, last / 2)?;
        let shift = emb.narrow(emb.rank() - 1, last / 2, last / 2)?;
        let n = xs.layer_norm(self.eps, None, None)?;
        n.mul(&scale.add_scalar(1.0))?.add(&shift)
    }
}

struct TimestepEmbedding {
    linear_1: Linear,
    linear_2: Linear,
}

impl TimestepEmbedding {
    fn zeros(in_dim: usize, hidden: usize) -> Self {
        Self {
            linear_1: Linear::zeros(in_dim, hidden, true),
            linear_2: Linear::zeros(hidden, hidden, true),
        }
    }

    fn load(map: &WeightMap, prefix: &str, in_dim: usize, hidden: usize) -> Result<Self> {
        Ok(Self {
            linear_1: Linear::load(map, &format!("{prefix}.linear_1"), in_dim, hidden, true)?,
            linear_2: Linear::load(map, &format!("{prefix}.linear_2"), hidden, hidden, true)?,
        })
    }

    fn forward(&self, xs: &CudaTensor) -> Result<CudaTensor> {
        self.linear_2.forward(&self.linear_1.forward(xs)?.silu())
    }
}

struct CombinedTimestepGuidanceTextProj {
    timestep: TimestepEmbedding,
    guidance: Option<TimestepEmbedding>,
    text: TimestepEmbedding,
    channels: usize,
}

impl CombinedTimestepGuidanceTextProj {
    fn zeros(channels: usize, hidden: usize, pooled: usize, with_guidance: bool) -> Self {
        Self {
            timestep: TimestepEmbedding::zeros(channels, hidden),
            guidance: with_guidance.then(|| TimestepEmbedding::zeros(channels, hidden)),
            text: TimestepEmbedding::zeros(pooled, hidden),
            channels,
        }
    }

    fn load(
        map: &WeightMap,
        prefix: &str,
        channels: usize,
        hidden: usize,
        pooled: usize,
        with_guidance: bool,
    ) -> Result<Self> {
        Ok(Self {
            timestep: TimestepEmbedding::load(
                map,
                &format!("{prefix}.timestep_embedder"),
                channels,
                hidden,
            )?,
            guidance: if with_guidance {
                Some(TimestepEmbedding::load(
                    map,
                    &format!("{prefix}.guidance_embedder"),
                    channels,
                    hidden,
                )?)
            } else {
                None
            },
            text: TimestepEmbedding::load(map, &format!("{prefix}.text_embedder"), pooled, hidden)?,
            channels,
        })
    }

    fn forward(&self, timestep: f32, guidance: Option<f32>, pooled: &CudaTensor) -> Result<CudaTensor> {
        let t = CudaTensor::from_vec(vec![timestep], vec![1])?;
        let proj = nn::sinusoidal_timesteps(&t, self.channels)?;
        let mut emb = self.timestep.forward(&proj)?;
        if let (Some(g), Some(ge)) = (guidance, &self.guidance) {
            let gp = nn::sinusoidal_timesteps(&CudaTensor::from_vec(vec![g], vec![1])?, self.channels)?;
            emb = emb.add(&ge.forward(&gp)?)?;
        }
        emb.add(&self.text.forward(pooled)?)
    }
}

struct FluxTransformerBlock {
    norm1: AdaLayerNormZero,
    norm1_context: AdaLayerNormZero,
    attn: Flux1Attention,
    ff: GeluFeedForward,
    ff_context: GeluFeedForward,
    eps: f32,
}

impl FluxTransformerBlock {
    fn zeros(cfg: &Flux1ArchConfig) -> Self {
        let dim = cfg.hidden_size();
        Self {
            norm1: AdaLayerNormZero::zeros(dim, cfg.eps),
            norm1_context: AdaLayerNormZero::zeros(dim, cfg.eps),
            attn: Flux1Attention::zeros(
                dim,
                cfg.num_attention_heads,
                cfg.attention_head_dim,
                cfg.eps,
                true,
                false,
            ),
            ff: GeluFeedForward::zeros(dim, cfg.mlp_hidden()),
            ff_context: GeluFeedForward::zeros(dim, cfg.mlp_hidden()),
            eps: cfg.eps,
        }
    }

    fn load(map: &WeightMap, i: usize, cfg: &Flux1ArchConfig) -> Result<Self> {
        let dim = cfg.hidden_size();
        let p = format!("transformer_blocks.{i}");
        Ok(Self {
            norm1: AdaLayerNormZero::load(map, &format!("{p}.norm1"), dim, cfg.eps)?,
            norm1_context: AdaLayerNormZero::load(map, &format!("{p}.norm1_context"), dim, cfg.eps)?,
            attn: Flux1Attention::load(
                map,
                &format!("{p}.attn"),
                dim,
                cfg.num_attention_heads,
                cfg.attention_head_dim,
                cfg.eps,
                true,
                false,
            )?,
            ff: GeluFeedForward::load(map, &format!("{p}.ff"), dim, cfg.mlp_hidden())?,
            ff_context: GeluFeedForward::load(map, &format!("{p}.ff_context"), dim, cfg.mlp_hidden())?,
            eps: cfg.eps,
        })
    }

    fn modulate_mlp(
        xs: &CudaTensor,
        shift: &CudaTensor,
        scale: &CudaTensor,
        eps: f32,
    ) -> Result<CudaTensor> {
        let n = xs.layer_norm(eps, None, None)?;
        n.mul(&scale.add_scalar(1.0))?.add(shift)
    }

    fn forward(
        &self,
        hidden: &CudaTensor,
        encoder: &CudaTensor,
        temb: &CudaTensor,
        rope: &(CudaTensor, CudaTensor),
    ) -> Result<(CudaTensor, CudaTensor)> {
        let (img_n, gate_msa, shift_mlp, scale_mlp, gate_mlp) = self.norm1.forward(hidden, temb)?;
        let (txt_n, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp) =
            self.norm1_context.forward(encoder, temb)?;
        let (img_a, txt_a) = self.attn.forward(&img_n, Some(&txt_n), rope)?;
        let txt_a = txt_a.ok_or_else(|| msg("joint attn dropped text"))?;
        let hidden = hidden.add(&img_a.mul(&gate_msa)?)?;
        let encoder = encoder.add(&txt_a.mul(&c_gate_msa)?)?;
        let img_n = Self::modulate_mlp(&hidden, &shift_mlp, &scale_mlp, self.eps)?;
        let hidden = hidden.add(&self.ff.forward(&img_n)?.mul(&gate_mlp)?)?;
        let txt_n = Self::modulate_mlp(&encoder, &c_shift_mlp, &c_scale_mlp, self.eps)?;
        let encoder = encoder.add(&self.ff_context.forward(&txt_n)?.mul(&c_gate_mlp)?)?;
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
    fn zeros(cfg: &Flux1ArchConfig) -> Self {
        let dim = cfg.hidden_size();
        let mlp = cfg.mlp_hidden();
        Self {
            norm: AdaLayerNormZeroSingle::zeros(dim, cfg.eps),
            attn: Flux1Attention::zeros(
                dim,
                cfg.num_attention_heads,
                cfg.attention_head_dim,
                cfg.eps,
                false,
                true,
            ),
            proj_mlp: Linear::zeros(dim, mlp, true),
            proj_out: Linear::zeros(dim + mlp, dim, true),
        }
    }

    fn load(map: &WeightMap, i: usize, cfg: &Flux1ArchConfig) -> Result<Self> {
        let dim = cfg.hidden_size();
        let mlp = cfg.mlp_hidden();
        let p = format!("single_transformer_blocks.{i}");
        Ok(Self {
            norm: AdaLayerNormZeroSingle::load(map, &format!("{p}.norm"), dim, cfg.eps)?,
            attn: Flux1Attention::load(
                map,
                &format!("{p}.attn"),
                dim,
                cfg.num_attention_heads,
                cfg.attention_head_dim,
                cfg.eps,
                false,
                true,
            )?,
            proj_mlp: Linear::load(map, &format!("{p}.proj_mlp"), dim, mlp, true)?,
            proj_out: Linear::load(map, &format!("{p}.proj_out"), dim + mlp, dim, true)?,
        })
    }

    fn forward(
        &self,
        hidden: &CudaTensor,
        encoder: &CudaTensor,
        temb: &CudaTensor,
        rope: &(CudaTensor, CudaTensor),
    ) -> Result<(CudaTensor, CudaTensor)> {
        let text_len = encoder.shape[1];
        let cat = CudaTensor::cat(&[encoder, hidden], 1)?;
        let (normed, gate) = self.norm.forward(&cat, temb)?;
        let mlp = nn::gelu_tanh(&self.proj_mlp.forward(&normed)?);
        let (attn, _) = self.attn.forward(&normed, None, rope)?;
        let y = self.proj_out.forward(&CudaTensor::cat(&[&attn, &mlp], attn.rank() - 1)?)?;
        let y = cat.add(&y.mul(&gate)?)?;
        let encoder = y.narrow(1, 0, text_len)?;
        let hidden = y.narrow(1, text_len, y.shape[1] - text_len)?;
        Ok((encoder, hidden))
    }
}

pub struct Flux1Transformer2D {
    pub cfg: Flux1ArchConfig,
    time_text: CombinedTimestepGuidanceTextProj,
    x_embed: Linear,
    ctx_embed: Linear,
    blocks: Vec<FluxTransformerBlock>,
    singles: Vec<FluxSingleTransformerBlock>,
    norm_out: AdaLayerNormContinuous,
    proj_out: Linear,
    rope_cache: Option<((usize, usize, usize), (CudaTensor, CudaTensor))>,
    rope_builds: u64,
}

impl Flux1Transformer2D {
    pub fn zeros(cfg: Flux1ArchConfig) -> Self {
        let dim = cfg.hidden_size();
        Self {
            time_text: CombinedTimestepGuidanceTextProj::zeros(
                cfg.timestep_guidance_channels,
                dim,
                cfg.pooled_projection_dim,
                cfg.guidance_embeds,
            ),
            x_embed: Linear::zeros(cfg.in_channels, dim, true),
            ctx_embed: Linear::zeros(cfg.joint_attention_dim, dim, true),
            blocks: (0..cfg.num_layers).map(|_| FluxTransformerBlock::zeros(&cfg)).collect(),
            singles: (0..cfg.num_single_layers)
                .map(|_| FluxSingleTransformerBlock::zeros(&cfg))
                .collect(),
            norm_out: AdaLayerNormContinuous::zeros(dim, cfg.eps),
            proj_out: Linear::zeros(dim, cfg.patch_size * cfg.patch_size * cfg.out_channels, true),
            rope_cache: None,
            rope_builds: 0,
            cfg,
        }
    }

    pub fn load(cfg: Flux1ArchConfig, map: &WeightMap) -> Result<Self> {
        let dim = cfg.hidden_size();
        let mut blocks = Vec::new();
        for i in 0..cfg.num_layers {
            blocks.push(FluxTransformerBlock::load(map, i, &cfg)?);
        }
        let mut singles = Vec::new();
        for i in 0..cfg.num_single_layers {
            singles.push(FluxSingleTransformerBlock::load(map, i, &cfg)?);
        }
        Ok(Self {
            time_text: CombinedTimestepGuidanceTextProj::load(
                map,
                "time_text_embed",
                cfg.timestep_guidance_channels,
                dim,
                cfg.pooled_projection_dim,
                cfg.guidance_embeds,
            )?,
            x_embed: Linear::load(map, "x_embedder", cfg.in_channels, dim, true)?,
            ctx_embed: Linear::load(map, "context_embedder", cfg.joint_attention_dim, dim, true)?,
            blocks,
            singles,
            norm_out: AdaLayerNormContinuous::load(map, "norm_out", dim, cfg.eps)?,
            proj_out: Linear::load(
                map,
                "proj_out",
                dim,
                cfg.patch_size * cfg.patch_size * cfg.out_channels,
                true,
            )?,
            rope_cache: None,
            rope_builds: 0,
            cfg,
        })
    }

    fn cached_rope(&mut self, text_len: usize, img_h: usize, img_w: usize) -> Result<(CudaTensor, CudaTensor)> {
        let key = (text_len, img_h, img_w);
        if let Some((k, tables)) = &self.rope_cache {
            if *k == key {
                return Ok(tables.clone());
            }
        }
        let (mut cos, mut sin) = flux1_rope(text_len, img_h, img_w, &self.cfg.axes_dims_rope, self.cfg.rope_theta)?;
        cos.pin_device()?;
        sin.pin_device()?;
        let tables = (cos, sin);
        self.rope_builds += 1;
        self.rope_cache = Some((key, tables.clone()));
        Ok(tables)
    }

    pub fn forward(
        &mut self,
        hidden: &CudaTensor,
        encoder: &CudaTensor,
        pooled: &CudaTensor,
        timestep: f32,
        guidance: Option<f32>,
        img_h: usize,
        img_w: usize,
    ) -> Result<CudaTensor> {
        let mut x = hidden.clone();
        let mut five = None;
        if x.rank() == 5 {
            let (b, c, t, h, w) = (x.shape[0], x.shape[1], x.shape[2], x.shape[3], x.shape[4]);
            five = Some((b, t, h, w));
            x = x.permute(&[0, 2, 3, 4, 1])?.reshape(vec![b, t * h * w, c])?;
        } else if x.rank() == 4 {
            let (b, c, h, w) = (x.shape[0], x.shape[1], x.shape[2], x.shape[3]);
            five = Some((b, 1, h, w));
            x = x.permute(&[0, 2, 3, 1])?.reshape(vec![b, h * w, c])?;
        }
        x = self.x_embed.forward(&x)?;
        let temb = self.time_text.forward(timestep * 1000.0, guidance.map(|g| g * 1000.0), pooled)?;
        let mut enc = self.ctx_embed.forward(encoder)?;
        let text_len = enc.shape[1];
        let rope = self.cached_rope(text_len, img_h, img_w)?;
        for block in &self.blocks {
            let (e, h) = block.forward(&x, &enc, &temb, &rope)?;
            enc = e;
            x = h;
        }
        for block in &self.singles {
            let (e, h) = block.forward(&x, &enc, &temb, &rope)?;
            enc = e;
            x = h;
        }
        x = self.norm_out.forward(&x, &temb)?;
        let mut out = self.proj_out.forward(&x)?;
        if let Some((b, t, h, w)) = five {
            out = out
                .reshape(vec![b, t, h, w, self.cfg.out_channels])?
                .permute(&[0, 4, 1, 2, 3])?;
        }
        Ok(out)
    }
}

fn flux1_rope(
    text_len: usize,
    img_h: usize,
    img_w: usize,
    axes: &[usize; 3],
    theta: f32,
) -> Result<(CudaTensor, CudaTensor)> {
    let mut ids = fastvideo_models::flux::text_ids(text_len);
    ids.extend(fastvideo_models::flux::image_ids(img_h, img_w));
    let seq = ids.len() / 3;
    let mut cos = Vec::new();
    let mut sin = Vec::new();
    for (axis, &dim) in axes.iter().enumerate() {
        let half = dim / 2;
        for p in 0..seq {
            let pos = ids[p * 3 + axis];
            for i in 0..half {
                let freq = pos * (1.0 / theta.powf((2 * i) as f32 / dim as f32));
                let c = freq.cos();
                let s = freq.sin();
                cos.push(c);
                cos.push(c);
                sin.push(s);
                sin.push(s);
            }
        }
    }
    let rope_dim: usize = axes.iter().sum();
    Ok((
        CudaTensor::from_vec(cos, vec![seq, rope_dim])?,
        CudaTensor::from_vec(sin, vec![seq, rope_dim])?,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tiny_forward_shapes() {
        let mut dit = Flux1Transformer2D::zeros(Flux1ArchConfig::tiny());
        let hidden = CudaTensor::zeros(&[1, dit.cfg.in_channels, 1, 2, 2]);
        let enc = CudaTensor::zeros(&[1, 4, dit.cfg.joint_attention_dim]);
        let pooled = CudaTensor::zeros(&[1, dit.cfg.pooled_projection_dim]);
        let out = dit.forward(&hidden, &enc, &pooled, 0.5, Some(3.5), 2, 2).unwrap();
        assert_eq!(out.shape, vec![1, dit.cfg.out_channels, 1, 2, 2]);
    }

    #[test]
    fn flux1_rope_is_3_axis() {
        let (cos, sin) = flux1_rope(2, 1, 1, &[2, 2, 4], 10_000.0).unwrap();
        assert_eq!(cos.shape, sin.shape);
        // text_len=2 + 1×1 image tokens; axes 2+2+4.
        assert_eq!(cos.shape, vec![3, 8]);
    }

    #[test]
    fn rope_tables_cached_across_forwards() {
        let mut dit = Flux1Transformer2D::zeros(Flux1ArchConfig::tiny());
        let hidden = CudaTensor::zeros(&[1, dit.cfg.in_channels, 1, 2, 2]);
        let enc = CudaTensor::zeros(&[1, 4, dit.cfg.joint_attention_dim]);
        let pooled = CudaTensor::zeros(&[1, dit.cfg.pooled_projection_dim]);
        dit.forward(&hidden, &enc, &pooled, 0.5, Some(1.0), 2, 2).unwrap();
        dit.forward(&hidden, &enc, &pooled, 0.4, Some(1.0), 2, 2).unwrap();
        assert_eq!(dit.rope_builds, 1, "same packed size reuses tables");
        dit.cached_rope(4, 3, 3).unwrap();
        assert_eq!(dit.rope_builds, 2);
        dit.cached_rope(4, 3, 3).unwrap();
        assert_eq!(dit.rope_builds, 2);
    }
}
