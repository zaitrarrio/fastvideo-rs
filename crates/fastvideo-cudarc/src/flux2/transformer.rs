//! Flux2 DiT on `CudaTensor` (cudarc primary path).

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use fastvideo_models::flux2::Flux2ArchConfig;

use crate::wan::nn::{self, Linear};
use crate::wan::tensor::{CudaTensor, Result, TensorError};
use crate::wan::weights::{cuda_tensor_shaped, join_key, WeightMap};

/// RoPE accounting. Host apply is the old Q/K D2H/H2D path; on a live CUDA
/// device [`apply_rotary`] launches `apply_rotary_bshd` and host counters stay
/// at zero. Table rebuilds are cached on [`Flux2Transformer2D`] across steps.
static ROPE_HOST_CALLS: AtomicU64 = AtomicU64::new(0);
static ROPE_HOST_MS: AtomicU64 = AtomicU64::new(0);
static ROPE_HOST_ELEMS: AtomicU64 = AtomicU64::new(0);
static ROPE_DEVICE_CALLS: AtomicU64 = AtomicU64::new(0);
static ROPE_DEVICE_ELEMS: AtomicU64 = AtomicU64::new(0);
static ROPE_TABLE_CALLS: AtomicU64 = AtomicU64::new(0);
static ROPE_TABLE_MS: AtomicU64 = AtomicU64::new(0);

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RopeHostStats {
    pub apply_calls: u64,
    pub apply_ms: u64,
    pub apply_elems: u64,
    pub device_apply_calls: u64,
    pub device_apply_elems: u64,
    pub table_calls: u64,
    pub table_ms: u64,
}

pub fn rope_host_stats() -> RopeHostStats {
    RopeHostStats {
        apply_calls: ROPE_HOST_CALLS.load(Ordering::Relaxed),
        apply_ms: ROPE_HOST_MS.load(Ordering::Relaxed),
        apply_elems: ROPE_HOST_ELEMS.load(Ordering::Relaxed),
        device_apply_calls: ROPE_DEVICE_CALLS.load(Ordering::Relaxed),
        device_apply_elems: ROPE_DEVICE_ELEMS.load(Ordering::Relaxed),
        table_calls: ROPE_TABLE_CALLS.load(Ordering::Relaxed),
        table_ms: ROPE_TABLE_MS.load(Ordering::Relaxed),
    }
}

pub fn reset_rope_host_stats() {
    for c in [
        &ROPE_HOST_CALLS,
        &ROPE_HOST_MS,
        &ROPE_HOST_ELEMS,
        &ROPE_DEVICE_CALLS,
        &ROPE_DEVICE_ELEMS,
        &ROPE_TABLE_CALLS,
        &ROPE_TABLE_MS,
    ] {
        c.store(0, Ordering::Relaxed);
    }
}

fn record_apply_rotary(elems: u64, ms: u64) {
    ROPE_HOST_CALLS.fetch_add(1, Ordering::Relaxed);
    ROPE_HOST_ELEMS.fetch_add(elems, Ordering::Relaxed);
    ROPE_HOST_MS.fetch_add(ms, Ordering::Relaxed);
}

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

fn swiglu(xs: &CudaTensor) -> Result<CudaTensor> {
    let last = *xs.shape.last().ok_or_else(|| msg("empty swiglu"))?;
    if last % 2 != 0 {
        return Err(msg(format!("swiglu last dim {last}")));
    }
    let x1 = xs.narrow(xs.rank() - 1, 0, last / 2)?;
    let x2 = xs.narrow(xs.rank() - 1, last / 2, last / 2)?;
    Ok(x1.silu().mul(&x2)?)
}

struct FeedForward {
    linear_in: Linear,
    linear_out: Linear,
}

impl FeedForward {
    fn zeros(dim: usize, inner: usize) -> Self {
        Self {
            linear_in: Linear::zeros(dim, inner * 2, false),
            linear_out: Linear::zeros(inner, dim, false),
        }
    }

    fn load(map: &WeightMap, prefix: &str, dim: usize, inner: usize) -> Result<Self> {
        Ok(Self {
            linear_in: Linear::load(map, &format!("{prefix}.linear_in"), dim, inner * 2, false)?,
            linear_out: Linear::load(map, &format!("{prefix}.linear_out"), inner, dim, false)?,
        })
    }

    fn forward(&self, xs: &CudaTensor) -> Result<CudaTensor> {
        self.linear_out.forward(&swiglu(&self.linear_in.forward(xs)?)?)
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
    let elems = xs.numel() as u64;
    if crate::wan::stats::device_expected() {
        let y = xs.apply_rotary_bshd(cos, sin)?;
        ROPE_DEVICE_CALLS.fetch_add(1, Ordering::Relaxed);
        ROPE_DEVICE_ELEMS.fetch_add(elems, Ordering::Relaxed);
        return Ok(y);
    }
    let t0 = Instant::now();
    let y = xs.apply_rotary_bshd(cos, sin)?;
    record_apply_rotary(elems, t0.elapsed().as_millis() as u64);
    Ok(y)
}

struct JointAttn {
    to_q: Linear,
    to_k: Linear,
    to_v: Linear,
    to_out: Linear,
    add_q: Linear,
    add_k: Linear,
    add_v: Linear,
    to_add_out: Linear,
    norm_q: RmsNorm,
    norm_k: RmsNorm,
    norm_aq: RmsNorm,
    norm_ak: RmsNorm,
    heads: usize,
    dim_head: usize,
}

impl JointAttn {
    fn zeros(dim: usize, heads: usize, dim_head: usize, eps: f32) -> Self {
        let inner = heads * dim_head;
        Self {
            to_q: Linear::zeros(dim, inner, false),
            to_k: Linear::zeros(dim, inner, false),
            to_v: Linear::zeros(dim, inner, false),
            to_out: Linear::zeros(inner, dim, false),
            add_q: Linear::zeros(dim, inner, false),
            add_k: Linear::zeros(dim, inner, false),
            add_v: Linear::zeros(dim, inner, false),
            to_add_out: Linear::zeros(inner, dim, false),
            norm_q: RmsNorm::zeros(dim_head, eps),
            norm_k: RmsNorm::zeros(dim_head, eps),
            norm_aq: RmsNorm::zeros(dim_head, eps),
            norm_ak: RmsNorm::zeros(dim_head, eps),
            heads,
            dim_head,
        }
    }

    fn load(map: &WeightMap, prefix: &str, dim: usize, heads: usize, dim_head: usize, eps: f32) -> Result<Self> {
        let inner = heads * dim_head;
        Ok(Self {
            to_q: Linear::load(map, &format!("{prefix}.to_q"), dim, inner, false)?,
            to_k: Linear::load(map, &format!("{prefix}.to_k"), dim, inner, false)?,
            to_v: Linear::load(map, &format!("{prefix}.to_v"), dim, inner, false)?,
            to_out: Linear::load(map, &format!("{prefix}.to_out.0"), inner, dim, false)?,
            add_q: Linear::load(map, &format!("{prefix}.add_q_proj"), dim, inner, false)?,
            add_k: Linear::load(map, &format!("{prefix}.add_k_proj"), dim, inner, false)?,
            add_v: Linear::load(map, &format!("{prefix}.add_v_proj"), dim, inner, false)?,
            to_add_out: Linear::load(map, &format!("{prefix}.to_add_out"), inner, dim, false)?,
            norm_q: RmsNorm::load(map, &format!("{prefix}.norm_q"), dim_head, eps)?,
            norm_k: RmsNorm::load(map, &format!("{prefix}.norm_k"), dim_head, eps)?,
            norm_aq: RmsNorm::load(map, &format!("{prefix}.norm_added_q"), dim_head, eps)?,
            norm_ak: RmsNorm::load(map, &format!("{prefix}.norm_added_k"), dim_head, eps)?,
            heads,
            dim_head,
        })
    }

    fn forward(
        &self,
        hidden: &CudaTensor,
        encoder: &CudaTensor,
        rope: &(CudaTensor, CudaTensor),
    ) -> Result<(CudaTensor, CudaTensor)> {
        let q = rms_heads(&self.to_q.forward(hidden)?, &self.norm_q, self.heads, self.dim_head)?;
        let k = rms_heads(&self.to_k.forward(hidden)?, &self.norm_k, self.heads, self.dim_head)?;
        let v = {
            let t = self.to_v.forward(hidden)?;
            t.reshape(vec![t.shape[0], t.shape[1], self.heads, self.dim_head])?
        };
        let eq = rms_heads(&self.add_q.forward(encoder)?, &self.norm_aq, self.heads, self.dim_head)?;
        let ek = rms_heads(&self.add_k.forward(encoder)?, &self.norm_ak, self.heads, self.dim_head)?;
        let ev = {
            let t = self.add_v.forward(encoder)?;
            t.reshape(vec![t.shape[0], t.shape[1], self.heads, self.dim_head])?
        };
        let text_len = encoder.shape[1];
        let mut q = CudaTensor::cat(&[&eq, &q], 1)?;
        let mut k = CudaTensor::cat(&[&ek, &k], 1)?;
        let v = CudaTensor::cat(&[&ev, &v], 1)?;
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
        let text = attn.narrow(1, 0, text_len)?;
        let img = attn.narrow(1, text_len, s - text_len)?;
        Ok((self.to_out.forward(&img)?, self.to_add_out.forward(&text)?))
    }
}

struct DoubleBlock {
    attn: JointAttn,
    ff: FeedForward,
    ff_ctx: FeedForward,
    eps: f32,
}

impl DoubleBlock {
    fn zeros(cfg: &Flux2ArchConfig) -> Self {
        let dim = cfg.hidden_size();
        Self {
            attn: JointAttn::zeros(dim, cfg.num_attention_heads, cfg.attention_head_dim, cfg.eps),
            ff: FeedForward::zeros(dim, cfg.mlp_hidden()),
            ff_ctx: FeedForward::zeros(dim, cfg.mlp_hidden()),
            eps: cfg.eps,
        }
    }

    fn load(map: &WeightMap, i: usize, cfg: &Flux2ArchConfig) -> Result<Self> {
        let dim = cfg.hidden_size();
        let p = format!("transformer_blocks.{i}");
        Ok(Self {
            attn: JointAttn::load(map, &format!("{p}.attn"), dim, cfg.num_attention_heads, cfg.attention_head_dim, cfg.eps)?,
            ff: FeedForward::load(map, &format!("{p}.ff"), dim, cfg.mlp_hidden())?,
            ff_ctx: FeedForward::load(map, &format!("{p}.ff_context"), dim, cfg.mlp_hidden())?,
            eps: cfg.eps,
        })
    }

    fn modulate(xs: &CudaTensor, shift: &CudaTensor, scale: &CudaTensor, eps: f32) -> Result<CudaTensor> {
        let n = xs.layer_norm(eps, None, None)?;
        n.mul(&scale.add_scalar(1.0))?.add(shift)
    }

    fn forward(
        &self,
        hidden: &CudaTensor,
        encoder: &CudaTensor,
        img: &[(CudaTensor, CudaTensor, CudaTensor); 2],
        txt: &[(CudaTensor, CudaTensor, CudaTensor); 2],
        rope: &(CudaTensor, CudaTensor),
    ) -> Result<(CudaTensor, CudaTensor)> {
        let img_n = Self::modulate(hidden, &img[0].0, &img[0].1, self.eps)?;
        let txt_n = Self::modulate(encoder, &txt[0].0, &txt[0].1, self.eps)?;
        let (ia, ta) = self.attn.forward(&img_n, &txt_n, rope)?;
        let hidden = hidden.add(&ia.mul(&img[0].2)?)?;
        let encoder = encoder.add(&ta.mul(&txt[0].2)?)?;
        let img_n = Self::modulate(&hidden, &img[1].0, &img[1].1, self.eps)?;
        let hidden = hidden.add(&self.ff.forward(&img_n)?.mul(&img[1].2)?)?;
        let txt_n = Self::modulate(&encoder, &txt[1].0, &txt[1].1, self.eps)?;
        let encoder = encoder.add(&self.ff_ctx.forward(&txt_n)?.mul(&txt[1].2)?)?;
        Ok((encoder, hidden))
    }
}

struct ParallelAttn {
    to_qkv_mlp: Linear,
    to_out: Linear,
    norm_q: RmsNorm,
    norm_k: RmsNorm,
    heads: usize,
    dim_head: usize,
    inner: usize,
}

impl ParallelAttn {
    fn zeros(dim: usize, heads: usize, dim_head: usize, mlp: usize, eps: f32) -> Self {
        let inner = heads * dim_head;
        Self {
            to_qkv_mlp: Linear::zeros(dim, inner * 3 + mlp * 2, false),
            to_out: Linear::zeros(inner + mlp, dim, false),
            norm_q: RmsNorm::zeros(dim_head, eps),
            norm_k: RmsNorm::zeros(dim_head, eps),
            heads,
            dim_head,
            inner,
        }
    }

    fn load(map: &WeightMap, prefix: &str, dim: usize, heads: usize, dim_head: usize, mlp: usize, eps: f32) -> Result<Self> {
        let inner = heads * dim_head;
        Ok(Self {
            to_qkv_mlp: Linear::load(map, &format!("{prefix}.to_qkv_mlp_proj"), dim, inner * 3 + mlp * 2, false)?,
            to_out: Linear::load(map, &format!("{prefix}.to_out"), inner + mlp, dim, false)?,
            norm_q: RmsNorm::load(map, &format!("{prefix}.norm_q"), dim_head, eps)?,
            norm_k: RmsNorm::load(map, &format!("{prefix}.norm_k"), dim_head, eps)?,
            heads,
            dim_head,
            inner,
        })
    }

    fn forward(&self, hidden: &CudaTensor, rope: &(CudaTensor, CudaTensor)) -> Result<CudaTensor> {
        let proj = self.to_qkv_mlp.forward(hidden)?;
        let last = *proj.shape.last().unwrap();
        let qkv = proj.narrow(proj.rank() - 1, 0, self.inner * 3)?;
        let mlp = proj.narrow(proj.rank() - 1, self.inner * 3, last - self.inner * 3)?;
        let q = qkv.narrow(qkv.rank() - 1, 0, self.inner)?;
        let k = qkv.narrow(qkv.rank() - 1, self.inner, self.inner)?;
        let v = qkv.narrow(qkv.rank() - 1, self.inner * 2, self.inner)?;
        let q = rms_heads(&q, &self.norm_q, self.heads, self.dim_head)?;
        let k = rms_heads(&k, &self.norm_k, self.heads, self.dim_head)?;
        let v = v.reshape(vec![v.shape[0], v.shape[1], self.heads, self.dim_head])?;
        let q = apply_rotary(&q, &rope.0, &rope.1)?;
        let k = apply_rotary(&k, &rope.0, &rope.1)?;
        let attn = nn::scaled_dot_product_attention(
            &q.transpose(1, 2)?,
            &k.transpose(1, 2)?,
            &v.transpose(1, 2)?,
            None,
        )?
        .transpose(1, 2)?;
        let attn = attn.reshape(vec![attn.shape[0], attn.shape[1], self.inner])?;
        let mlp = swiglu(&mlp)?;
        self.to_out.forward(&CudaTensor::cat(&[&attn, &mlp], attn.rank() - 1)?)
    }
}

struct SingleBlock {
    attn: ParallelAttn,
    eps: f32,
}

impl SingleBlock {
    fn zeros(cfg: &Flux2ArchConfig) -> Self {
        Self {
            attn: ParallelAttn::zeros(
                cfg.hidden_size(),
                cfg.num_attention_heads,
                cfg.attention_head_dim,
                cfg.mlp_hidden(),
                cfg.eps,
            ),
            eps: cfg.eps,
        }
    }

    fn load(map: &WeightMap, i: usize, cfg: &Flux2ArchConfig) -> Result<Self> {
        Ok(Self {
            attn: ParallelAttn::load(
                map,
                &format!("single_transformer_blocks.{i}.attn"),
                cfg.hidden_size(),
                cfg.num_attention_heads,
                cfg.attention_head_dim,
                cfg.mlp_hidden(),
                cfg.eps,
            )?,
            eps: cfg.eps,
        })
    }

    fn forward(
        &self,
        hidden: &CudaTensor,
        shift: &CudaTensor,
        scale: &CudaTensor,
        gate: &CudaTensor,
        rope: &(CudaTensor, CudaTensor),
    ) -> Result<CudaTensor> {
        let n = hidden.layer_norm(self.eps, None, None)?;
        let n = n.mul(&scale.add_scalar(1.0))?.add(shift)?;
        hidden.add(&self.attn.forward(&n, rope)?.mul(gate)?)
    }
}

struct TimeEmbed {
    linear_1: Linear,
    linear_2: Linear,
    channels: usize,
}

impl TimeEmbed {
    fn zeros(channels: usize, hidden: usize) -> Self {
        Self {
            linear_1: Linear::zeros(channels, hidden, false),
            linear_2: Linear::zeros(hidden, hidden, false),
            channels,
        }
    }

    fn load(map: &WeightMap, prefix: &str, channels: usize, hidden: usize) -> Result<Self> {
        Ok(Self {
            linear_1: Linear::load(map, &format!("{prefix}.linear_1"), channels, hidden, false)?,
            linear_2: Linear::load(map, &format!("{prefix}.linear_2"), hidden, hidden, false)?,
            channels,
        })
    }

    fn forward(&self, t: &CudaTensor) -> Result<CudaTensor> {
        let proj = nn::sinusoidal_timesteps(t, self.channels)?;
        self.linear_2.forward(&self.linear_1.forward(&proj)?.silu())
    }
}

struct Modulation {
    linear: Linear,
    sets: usize,
}

impl Modulation {
    fn zeros(dim: usize, sets: usize) -> Self {
        Self {
            linear: Linear::zeros(dim, dim * 3 * sets, false),
            sets,
        }
    }

    fn load(map: &WeightMap, prefix: &str, dim: usize, sets: usize) -> Result<Self> {
        Ok(Self {
            linear: Linear::load(map, prefix, dim, dim * 3 * sets, false)?,
            sets,
        })
    }

    fn forward(&self, temb: &CudaTensor) -> Result<Vec<(CudaTensor, CudaTensor, CudaTensor)>> {
        let mut m = self.linear.forward(&temb.silu())?;
        if m.rank() == 2 {
            m = m.unsqueeze(1)?;
        }
        let last = *m.shape.last().unwrap();
        let chunk = last / (3 * self.sets);
        let mut out = Vec::new();
        for i in 0..self.sets {
            let base = i * 3 * chunk;
            out.push((
                m.narrow(m.rank() - 1, base, chunk)?,
                m.narrow(m.rank() - 1, base + chunk, chunk)?,
                m.narrow(m.rank() - 1, base + 2 * chunk, chunk)?,
            ));
        }
        Ok(out)
    }
}

pub struct Flux2Transformer2D {
    pub cfg: Flux2ArchConfig,
    time: TimeEmbed,
    guidance: Option<TimeEmbed>,
    double_img: Modulation,
    double_txt: Modulation,
    single: Modulation,
    x_embed: Linear,
    ctx_embed: Linear,
    blocks: Vec<DoubleBlock>,
    singles: Vec<SingleBlock>,
    norm_out: Linear,
    proj_out: Linear,
    /// `(text_len, img_h, img_w)` → pinned sin/cos. Reused across denoise steps.
    rope_cache: Option<((usize, usize, usize), (CudaTensor, CudaTensor))>,
    /// Times `flux2_rope` ran for this instance (cache misses).
    rope_builds: u64,
}

impl Flux2Transformer2D {
    pub fn zeros(cfg: Flux2ArchConfig) -> Self {
        let dim = cfg.hidden_size();
        Self {
            time: TimeEmbed::zeros(cfg.timestep_guidance_channels, dim),
            guidance: cfg
                .guidance_embeds
                .then(|| TimeEmbed::zeros(cfg.timestep_guidance_channels, dim)),
            double_img: Modulation::zeros(dim, 2),
            double_txt: Modulation::zeros(dim, 2),
            single: Modulation::zeros(dim, 1),
            x_embed: Linear::zeros(cfg.in_channels, dim, false),
            ctx_embed: Linear::zeros(cfg.joint_attention_dim, dim, false),
            blocks: (0..cfg.num_layers).map(|_| DoubleBlock::zeros(&cfg)).collect(),
            singles: (0..cfg.num_single_layers).map(|_| SingleBlock::zeros(&cfg)).collect(),
            norm_out: Linear::zeros(dim, dim * 2, false),
            proj_out: Linear::zeros(dim, cfg.out_channels, false),
            rope_cache: None,
            rope_builds: 0,
            cfg,
        }
    }

    pub fn load(cfg: Flux2ArchConfig, map: &WeightMap) -> Result<Self> {
        let dim = cfg.hidden_size();
        let mut blocks = Vec::new();
        for i in 0..cfg.num_layers {
            blocks.push(DoubleBlock::load(map, i, &cfg)?);
        }
        let mut singles = Vec::new();
        for i in 0..cfg.num_single_layers {
            singles.push(SingleBlock::load(map, i, &cfg)?);
        }
        Ok(Self {
            time: TimeEmbed::load(map, "time_guidance_embed.timestep_embedder", cfg.timestep_guidance_channels, dim)?,
            guidance: if cfg.guidance_embeds {
                Some(TimeEmbed::load(
                    map,
                    "time_guidance_embed.guidance_embedder",
                    cfg.timestep_guidance_channels,
                    dim,
                )?)
            } else {
                None
            },
            double_img: Modulation::load(map, "double_stream_modulation_img.linear", dim, 2)?,
            double_txt: Modulation::load(map, "double_stream_modulation_txt.linear", dim, 2)?,
            single: Modulation::load(map, "single_stream_modulation.linear", dim, 1)?,
            x_embed: Linear::load(map, "x_embedder", cfg.in_channels, dim, false)?,
            ctx_embed: Linear::load(map, "context_embedder", cfg.joint_attention_dim, dim, false)?,
            blocks,
            singles,
            norm_out: Linear::load(map, "norm_out.linear", dim, dim * 2, false)?,
            proj_out: Linear::load(map, "proj_out", dim, cfg.out_channels, false)?,
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
        let (mut cos, mut sin) = flux2_rope(text_len, img_h, img_w, &self.cfg.axes_dims_rope, self.cfg.rope_theta)?;
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
        timestep: f32,
        guidance: Option<f32>,
        img_h: usize,
        img_w: usize,
    ) -> Result<CudaTensor> {
        let mut x = hidden.clone();
        let mut five = None;
        if x.rank() == 5 {
            let (b, c, t, h, w) = (x.shape[0], x.shape[1], x.shape[2], x.shape[3], x.shape[4]);
            five = Some((b, t, h, w, c));
            x = x.permute(&[0, 2, 3, 4, 1])?.reshape(vec![b, t * h * w, c])?;
        }
        let t = CudaTensor::from_vec(vec![timestep * 1000.0], vec![1])?;
        let mut temb = self.time.forward(&t)?;
        if let (Some(g), Some(ge)) = (guidance, &self.guidance) {
            temb = temb.add(&ge.forward(&CudaTensor::from_vec(vec![g * 1000.0], vec![1])?)?)?;
        }
        let img_mod = self.double_img.forward(&temb)?;
        let txt_mod = self.double_txt.forward(&temb)?;
        let single_mod = self.single.forward(&temb)?;
        let img_arr = [img_mod[0].clone(), img_mod[1].clone()];
        let txt_arr = [txt_mod[0].clone(), txt_mod[1].clone()];
        x = self.x_embed.forward(&x)?;
        let mut enc = self.ctx_embed.forward(encoder)?;
        let text_len = enc.shape[1];
        let img_len = x.shape[1];
        let rope = self.cached_rope(text_len, img_h, img_w)?;
        for block in &self.blocks {
            let (e, h) = block.forward(&x, &enc, &img_arr, &txt_arr, &rope)?;
            enc = e;
            x = h;
        }
        x = CudaTensor::cat(&[&enc, &x], 1)?;
        let (shift, scale, gate) = &single_mod[0];
        for block in &self.singles {
            x = block.forward(&x, shift, scale, gate, &rope)?;
        }
        x = x.narrow(1, text_len, img_len)?;
        let n = x.layer_norm(self.cfg.eps, None, None)?;
        let ada = self.norm_out.forward(&temb.silu())?;
        let half = ada.shape[ada.rank() - 1] / 2;
        let scale = ada.narrow(ada.rank() - 1, 0, half)?.unsqueeze(1)?;
        let shift = ada.narrow(ada.rank() - 1, half, half)?.unsqueeze(1)?;
        x = n.mul(&scale.add_scalar(1.0))?.add(&shift)?;
        let mut out = self.proj_out.forward(&x)?;
        if let Some((b, t, h, w, _)) = five {
            out = out
                .reshape(vec![b, t, h, w, self.cfg.out_channels])?
                .permute(&[0, 4, 1, 2, 3])?;
        }
        Ok(out)
    }
}

fn flux2_rope(text_len: usize, img_h: usize, img_w: usize, axes: &[usize; 4], theta: f32) -> Result<(CudaTensor, CudaTensor)> {
    let t0 = Instant::now();
    let txt = fastvideo_models::flux2::text_ids(text_len);
    let img = fastvideo_models::flux2::image_ids(1, img_h, img_w);
    let mut ids = txt;
    ids.extend(img);
    let seq = ids.len() / 4;
    let mut cos = Vec::new();
    let mut sin = Vec::new();
    for (axis, &dim) in axes.iter().enumerate() {
        let half = dim / 2;
        for p in 0..seq {
            let pos = ids[p * 4 + axis];
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
    let tables = (
        CudaTensor::from_vec(cos, vec![seq, rope_dim])?,
        CudaTensor::from_vec(sin, vec![seq, rope_dim])?,
    );
    ROPE_TABLE_CALLS.fetch_add(1, Ordering::Relaxed);
    ROPE_TABLE_MS.fetch_add(t0.elapsed().as_millis() as u64, Ordering::Relaxed);
    Ok(tables)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    static ROPE_STATS_LOCK: Mutex<()> = Mutex::new(());

    #[test]
    fn apply_rotary_records_host_stats() {
        let _guard = ROPE_STATS_LOCK.lock().unwrap();
        reset_rope_host_stats();
        let xs = CudaTensor::from_vec((0..16).map(|i| i as f32 * 0.1).collect(), vec![1, 2, 1, 8]).unwrap();
        let ones = vec![1.0f32; 16];
        let zeros = vec![0.0f32; 16];
        let cos = CudaTensor::from_vec(ones, vec![2, 8]).unwrap();
        let sin = CudaTensor::from_vec(zeros, vec![2, 8]).unwrap();
        let out = apply_rotary(&xs, &cos, &sin).unwrap();
        assert_eq!(out.shape, vec![1, 2, 1, 8]);
        assert_eq!(
            &*out.host_cow().unwrap(),
            &*xs.host_cow().unwrap(),
            "cos=1 sin=0 is identity"
        );
        let stats = rope_host_stats();
        if crate::wan::stats::device_expected() {
            assert_eq!(stats.device_apply_calls, 1);
            assert_eq!(stats.device_apply_elems, 16);
            assert_eq!(stats.apply_calls, 0);
        } else {
            assert_eq!(stats.apply_calls, 1);
            assert_eq!(stats.apply_elems, 16);
            assert_eq!(stats.device_apply_calls, 0);
        }
    }

    #[test]
    fn apply_rotary_quarter_turn() {
        let _guard = ROPE_STATS_LOCK.lock().unwrap();
        reset_rope_host_stats();
        let xs = CudaTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 1, 1, 4]).unwrap();
        let cos = CudaTensor::from_vec(vec![0.0, 0.0, 0.0, 0.0], vec![1, 4]).unwrap();
        let sin = CudaTensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![1, 4]).unwrap();
        let out = apply_rotary(&xs, &cos, &sin).unwrap();
        // (x1, x2) → (-x2, x1)
        let got = out.host_cow().unwrap();
        assert!((got[0] + 2.0).abs() < 1e-6);
        assert!((got[1] - 1.0).abs() < 1e-6);
        assert!((got[2] + 4.0).abs() < 1e-6);
        assert!((got[3] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn flux2_rope_records_table_stats() {
        let (cos, sin) = flux2_rope(2, 1, 1, &[2, 2, 2, 2], 2000.0).unwrap();
        assert_eq!(cos.shape, sin.shape);
        // text_len=2 + 1×1 image tokens; axes 2+2+2+2.
        assert_eq!(cos.shape, vec![3, 8]);
        assert!(!cos.data.is_empty());
    }

    #[test]
    fn rope_tables_cached_across_forwards() {
        let _guard = ROPE_STATS_LOCK.lock().unwrap();
        let mut dit = Flux2Transformer2D::zeros(Flux2ArchConfig::tiny());
        let hidden = CudaTensor::zeros(&[1, dit.cfg.in_channels, 1, 2, 2]);
        let enc = CudaTensor::zeros(&[1, 4, dit.cfg.joint_attention_dim]);
        dit.forward(&hidden, &enc, 0.5, None, 2, 2).unwrap();
        dit.forward(&hidden, &enc, 0.4, None, 2, 2).unwrap();
        assert_eq!(dit.rope_builds, 1, "same packed size reuses tables");
        assert_eq!(dit.rope_cache.as_ref().map(|(k, _)| *k), Some((4, 2, 2)));
        dit.cached_rope(4, 3, 3).unwrap();
        assert_eq!(dit.rope_builds, 2, "size change rebuilds tables");
        dit.cached_rope(4, 3, 3).unwrap();
        assert_eq!(dit.rope_builds, 2, "repeat of new size stays cached");
    }
}
