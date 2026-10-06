//! LingBot-Video DiT (`LingBotVideoTransformer3DModel`, Dense 1.3B and MoE
//! 30B-A3B, base and refiner).
//!
//! Single-stream joint transformer over `[video; text]` tokens:
//!
//! * patchify `(1, 2, 2)` → `patch_embedder`; text: RMSNorm → Linear-SiLU-Linear;
//! * timestep: sinusoid (cos, sin) → `time_embedder` MLP → `time_modulation`
//!   (SiLU, Linear → 6·D), shared by every token; each block adds its
//!   `scale_shift_table` and splits shift/scale/gate (gates `tanh`);
//! * block: `x += tanh(g1) · RMSNorm_post(attn(RMSNorm(x)·(1+s1)+b1))`,
//!   `x += tanh(g2) · RMSNorm_post(ffn(RMSNorm(x)·(1+s2)+b2))`;
//!   attention has per-head RMS q/k norm and 3D complex RoPE
//!   (`fastvideo_models::lingbot::rope`), full (non-causal) over the joint
//!   sequence;
//! * FFN: SwiGLU (dense layers) or MoE: group-limited sigmoid router
//!   (`fastvideo_models::lingbot::routing`), 128 SwiGLU experts stored stacked
//!   (`experts.w1/w2/w3` `[E, ·, ·]`), plus a shared SwiGLU expert;
//! * out: LayerNorm (no affine) · (1+scale) + shift (`norm_out_modulation`),
//!   `proj_out` on the video tokens, unpatchify.
//!
//! The time path (`time_embedder`, `time_modulation`, `norm_out_modulation`,
//! `scale_shift_table`) runs in f32 on the host, as the reference keeps those
//! modules in fp32. The router GEMM is f32 too (routing is discrete).

use std::sync::Mutex;

use fastvideo_models::lingbot::sol::PisaPolicy;
use fastvideo_models::lingbot::{joint_positions, rope_tables, LingBotTransformerConfig, RouterSpec};

use crate::wan::nn::{self, Linear};
use crate::wan::tensor::{CudaTensor, Result, TensorError};
use crate::wan::weights::{self, WeightMap};

fn msg(s: impl Into<String>) -> TensorError {
    TensorError::Message(s.into())
}

/// A small f32 linear evaluated on the host (`[out, in]` row-major).
#[derive(Debug, Clone)]
struct HostLinear {
    w: Vec<f32>,
    b: Vec<f32>,
    out_dim: usize,
    in_dim: usize,
}

impl HostLinear {
    fn load(map: &WeightMap, prefix: &str, in_dim: usize, out_dim: usize) -> Result<Self> {
        let w = weights::cuda_tensor_shaped(map, &weights::join_key(prefix, "weight"), &[out_dim, in_dim])?;
        let b = weights::cuda_tensor_shaped(map, &weights::join_key(prefix, "bias"), &[out_dim])?;
        Ok(Self {
            w: w.host_cow()?.into_owned(),
            b: b.host_cow()?.into_owned(),
            out_dim,
            in_dim,
        })
    }

    fn zeros(in_dim: usize, out_dim: usize) -> Self {
        Self {
            w: vec![0.0; in_dim * out_dim],
            b: vec![0.0; out_dim],
            out_dim,
            in_dim,
        }
    }

    fn apply(&self, x: &[f32]) -> Vec<f32> {
        use rayon::prelude::*;
        (0..self.out_dim)
            .into_par_iter()
            .map(|o| {
                let row = &self.w[o * self.in_dim..(o + 1) * self.in_dim];
                row.iter().zip(x).map(|(a, b)| a * b).sum::<f32>() + self.b[o]
            })
            .collect()
    }
}

fn silu_host(x: &[f32]) -> Vec<f32> {
    x.iter().map(|&v| v / (1.0 + (-v).exp())).collect()
}

/// `Timesteps(dim, flip_sin_to_cos=True, downscale_freq_shift=0)`: `[cos | sin]`.
pub fn timestep_sinusoid(t: f32, dim: usize) -> Vec<f32> {
    let half = dim / 2;
    let mut out = vec![0f32; dim];
    for i in 0..half {
        let freq = (-(10000f32.ln()) * i as f32 / half as f32).exp();
        let arg = t * freq;
        out[i] = arg.cos();
        out[half + i] = arg.sin();
    }
    out
}

fn vec_tensor(v: &[f32], dims: &[usize]) -> Result<CudaTensor> {
    let mut t = CudaTensor::from_vec(v.to_vec(), dims.to_vec())?;
    t.pin_device()?;
    Ok(t)
}

fn norm_weight(map: &WeightMap, key: &str, dim: usize) -> Result<CudaTensor> {
    let mut t = weights::cuda_tensor_shaped(map, key, &[dim])?;
    t.pin_device()?;
    Ok(t)
}

/// SwiGLU MLP `down(silu(gate(x)) * up(x))`, bias-free.
struct Mlp {
    gate: Linear,
    up: Linear,
    down: Linear,
}

impl Mlp {
    fn zeros(dim: usize, mid: usize) -> Self {
        Self {
            gate: Linear::zeros(dim, mid, false),
            up: Linear::zeros(dim, mid, false),
            down: Linear::zeros(mid, dim, false),
        }
    }

    fn load(map: &WeightMap, prefix: &str, dim: usize, mid: usize) -> Result<Self> {
        let key = |n: &str| weights::join_key(prefix, n);
        Ok(Self {
            gate: Linear::load(map, &key("gate_proj"), dim, mid, false)?,
            up: Linear::load(map, &key("up_proj"), dim, mid, false)?,
            down: Linear::load(map, &key("down_proj"), mid, dim, false)?,
        })
    }

    fn forward(&self, x: &CudaTensor) -> Result<CudaTensor> {
        let g = self.gate.forward(x)?.silu();
        let u = self.up.forward(x)?;
        self.down.forward(&g.mul(&u)?)
    }
}

/// `[E, out, in]` stacked expert weights → one bias-free linear per expert.
fn stacked_linears(
    map: &WeightMap,
    key: &str,
    experts: usize,
    out_dim: usize,
    in_dim: usize,
) -> Result<Vec<Linear>> {
    let per = out_dim * in_dim;
    #[cfg(feature = "cuda")]
    if Linear::device_bf16_route() && map.has_tensor(key) {
        if let Some((shape, values)) = map.lazy_bf16(key)? {
            if shape != [experts, out_dim, in_dim] {
                return Err(msg(format!(
                    "{key}: shape {shape:?} != [{experts}, {out_dim}, {in_dim}]"
                )));
            }
            let dev = crate::wan::device::global_device().ok_or_else(|| msg("no device"))?;
            let mut out = Vec::with_capacity(experts);
            for e in 0..experts {
                let slice = dev
                    .stream
                    .memcpy_stod(&values[e * per..(e + 1) * per])
                    .map_err(|err| msg(err.to_string()))?;
                crate::wan::stats::record_h2d(per / 2);
                out.push(Linear::from_device_bf16(slice, in_dim, out_dim)?);
            }
            return Ok(out);
        }
    }
    let t = weights::cuda_tensor_shaped(map, key, &[experts, out_dim, in_dim])?;
    let host = t.host_cow()?;
    (0..experts)
        .map(|e| {
            Linear::from_tensors(
                CudaTensor::from_vec(host[e * per..(e + 1) * per].to_vec(), vec![out_dim, in_dim])?,
                None,
            )
        })
        .collect()
}

/// Which router implementation runs (`FASTVIDEO_LINGBOT_ROUTER=host|device`).
fn router_on_host() -> bool {
    static FLAG: crate::wan::envflag::CachedString = crate::wan::envflag::CachedString::new();
    FLAG.get_or_init(|| crate::wan::envflag::string_flag("FASTVIDEO_LINGBOT_ROUTER", "device")) == "host"
}

/// Tokens per MoE dispatch chunk (`FASTVIDEO_LINGBOT_MOE_CHUNK`, default 32768):
/// bounds the `[tokens · k, D]` gathered activations.
fn moe_chunk() -> usize {
    crate::wan::envflag::usize_flag("FASTVIDEO_LINGBOT_MOE_CHUNK", 32_768).max(1)
}

struct Moe {
    /// `[D, E]` f32 (the router weight transposed for `x @ W^T`).
    router_t: CudaTensor,
    bias: Vec<f32>,
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    bias_dev: CudaTensor,
    experts: Vec<Mlp>,
    shared: Option<Mlp>,
    spec: RouterSpec,
}

/// Router parity counters (`FASTVIDEO_LINGBOT_ROUTER_CHECK=1`): device vs host
/// decisions on the first MoE call of the process.
#[cfg(feature = "cuda")]
static ROUTER_CHECKED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

impl Moe {
    fn zeros(cfg: &LingBotTransformerConfig) -> Result<Self> {
        let (d, e, m) = (cfg.hidden_size, cfg.num_experts, cfg.moe_intermediate_size);
        Ok(Self {
            router_t: vec_tensor(&vec![0.0; d * e], &[d, e])?,
            bias: vec![0.0; e],
            bias_dev: vec_tensor(&vec![0.0; e], &[e])?,
            experts: (0..e).map(|_| Mlp::zeros(d, m)).collect(),
            shared: (cfg.n_shared_experts > 0).then(|| Mlp::zeros(d, m * cfg.n_shared_experts)),
            spec: RouterSpec::from_config(cfg),
        })
    }

    fn load(map: &WeightMap, prefix: &str, cfg: &LingBotTransformerConfig) -> Result<Self> {
        let (d, e, m) = (cfg.hidden_size, cfg.num_experts, cfg.moe_intermediate_size);
        let key = |n: &str| weights::join_key(prefix, n);
        let w = weights::cuda_tensor_shaped(map, &key("router.weight"), &[e, d])?;
        let wh = w.host_cow()?;
        let mut wt = vec![0f32; d * e];
        for r in 0..e {
            for c in 0..d {
                wt[c * e + r] = wh[r * d + c];
            }
        }
        let bias = weights::cuda_tensor_shaped(map, &key("router.e_score_correction_bias"), &[e])?
            .host_cow()?
            .into_owned();
        let w1 = stacked_linears(map, &key("experts.w1"), e, m, d)?;
        let w3 = stacked_linears(map, &key("experts.w3"), e, m, d)?;
        let w2 = stacked_linears(map, &key("experts.w2"), e, d, m)?;
        let experts = w1
            .into_iter()
            .zip(w3)
            .zip(w2)
            .map(|((gate, up), down)| Mlp { gate, up, down })
            .collect();
        let shared = if cfg.n_shared_experts > 0 {
            Some(Mlp::load(map, &key("shared_experts"), d, m * cfg.n_shared_experts)?)
        } else {
            None
        };
        Ok(Self {
            router_t: vec_tensor(&wt, &[d, e])?,
            bias_dev: vec_tensor(&bias, &[e])?,
            bias,
            experts,
            shared,
            spec: RouterSpec::from_config(cfg),
        })
    }

    fn route(&self, x: &CudaTensor) -> Result<(Vec<u32>, Vec<f32>)> {
        let logits = x.to_f32_act()?.matmul(&self.router_t)?;
        let host = |l: &CudaTensor| -> Result<(Vec<u32>, Vec<f32>)> {
            Ok(fastvideo_models::lingbot::route(&l.host_cow()?, &self.bias, &self.spec))
        };
        if router_on_host() {
            return host(&logits);
        }
        #[cfg(feature = "cuda")]
        if let (Some(l), Some(b)) = (logits.dev()?, self.bias_dev.dev()?) {
            let p = crate::wan::ops::GroupTopk {
                experts: self.spec.num_experts,
                top_k: self.spec.top_k,
                n_group: self.spec.n_group.unwrap_or(1),
                topk_group: self.spec.topk_group,
                softmax: self.spec.score_func == fastvideo_models::lingbot::ScoreFunc::Softmax,
                norm: self.spec.norm_topk_prob,
                scale: self.spec.route_scale,
                round_bf16: self.spec.round_bf16,
            };
            let got = crate::wan::ops::moe_group_topk_device(&l, &b, p)?;
            if crate::wan::envflag::bool_flag("FASTVIDEO_LINGBOT_ROUTER_CHECK", false)
                && !ROUTER_CHECKED.swap(true, std::sync::atomic::Ordering::Relaxed)
            {
                let want = host(&logits)?;
                let k = self.spec.top_k;
                let rows = got.0.len() / k.max(1);
                let mut set_diff = 0usize;
                let mut w_err = 0f32;
                for r in 0..rows {
                    let mut a: Vec<u32> = got.0[r * k..(r + 1) * k].to_vec();
                    let mut b: Vec<u32> = want.0[r * k..(r + 1) * k].to_vec();
                    a.sort_unstable();
                    b.sort_unstable();
                    if a != b {
                        set_diff += 1;
                    } else {
                        for s in 0..k {
                            w_err = w_err.max((got.1[r * k + s] - want.1[r * k + s]).abs());
                        }
                    }
                }
                crate::wan::log::info(format_args!(
                    "lingbot router check: {set_diff}/{rows} rows chose a different expert set; \
                     max |w| diff on matching rows {w_err:.3e}"
                ));
            }
            return Ok(got);
        }
        host(&logits)
    }

    fn forward(&self, x: &CudaTensor) -> Result<CudaTensor> {
        let [b, s, d] = match x.shape[..] {
            [b, s, d] => [b, s, d],
            _ => return Err(msg(format!("lingbot moe: {:?}", x.shape))),
        };
        let n_tok = b * s;
        let flat = x.reshape(vec![n_tok, d])?;
        let k = self.spec.top_k;
        let chunk = moe_chunk();
        let mut parts = Vec::with_capacity(n_tok.div_ceil(chunk));
        let mut start = 0usize;
        while start < n_tok {
            let len = chunk.min(n_tok - start);
            let xc = flat.narrow(0, start, len)?;
            let (idx, w) = self.route(&xc)?;
            let disp = fastvideo_models::lingbot::dispatch(&idx, self.spec.num_experts);
            let rows: Vec<usize> = disp.order.iter().map(|&f| f / k).collect();
            let gathered = xc.index_select_rows(&rows)?;
            let mut outs = Vec::with_capacity(self.experts.len());
            let mut off = 0usize;
            for (e, &cnt) in disp.counts.iter().enumerate() {
                if cnt == 0 {
                    continue;
                }
                let xe = gathered.narrow(0, off, cnt)?;
                outs.push(self.experts[e].forward(&xe)?.to_f32_act()?);
                off += cnt;
            }
            let refs: Vec<&CudaTensor> = outs.iter().collect();
            let ys = CudaTensor::cat(&refs, 0)?;
            let mut routed = combine(&ys, &disp.pos, &w, len, k, d)?;
            if let Some(shared) = &self.shared {
                routed = routed.add(&shared.forward(&xc)?.to_f32_act()?)?;
            }
            parts.push(routed);
            start += len;
        }
        let refs: Vec<&CudaTensor> = parts.iter().collect();
        CudaTensor::cat(&refs, 0)?.reshape(vec![b, s, d])
    }
}

fn combine(ys: &CudaTensor, pos: &[u32], w: &[f32], n: usize, k: usize, d: usize) -> Result<CudaTensor> {
    #[cfg(feature = "cuda")]
    if let Some(y) = ys.dev()? {
        return CudaTensor::from_device_slice(
            crate::wan::ops::moe_combine_device(&y, pos, w, n, k, d)?,
            vec![n, d],
        );
    }
    CudaTensor::from_vec(
        crate::wan::ops::moe_combine_host(&ys.host_cow()?, pos, w, n, k, d),
        vec![n, d],
    )
}

enum Ffn {
    Dense(Mlp),
    Moe(Moe),
}

impl Ffn {
    fn forward(&self, x: &CudaTensor) -> Result<CudaTensor> {
        match self {
            Self::Dense(f) => f.forward(x),
            Self::Moe(f) => f.forward(x),
        }
    }
}

struct Attention {
    to_q: Linear,
    to_k: Linear,
    to_v: Linear,
    to_out: Linear,
    norm_q: CudaTensor,
    norm_k: CudaTensor,
    heads: usize,
    head_dim: usize,
    eps: f32,
}

/// Attention route of one forward: dense, or PISA at `sparsity` on the
/// layers the policy leaves sparse.
#[derive(Debug, Clone, Copy, Default)]
pub struct AttnRoute<'a> {
    pub pisa: Option<&'a PisaPolicy>,
}

impl Attention {
    fn zeros(cfg: &LingBotTransformerConfig) -> Result<Self> {
        let d = cfg.hidden_size;
        let hd = cfg.head_dim();
        Ok(Self {
            to_q: Linear::zeros(d, d, cfg.qkv_bias),
            to_k: Linear::zeros(d, d, cfg.qkv_bias),
            to_v: Linear::zeros(d, d, cfg.qkv_bias),
            to_out: Linear::zeros(d, d, cfg.out_bias),
            norm_q: vec_tensor(&vec![1.0; hd], &[hd])?,
            norm_k: vec_tensor(&vec![1.0; hd], &[hd])?,
            heads: cfg.num_attention_heads,
            head_dim: hd,
            eps: cfg.norm_eps,
        })
    }

    fn load(map: &WeightMap, prefix: &str, cfg: &LingBotTransformerConfig) -> Result<Self> {
        let d = cfg.hidden_size;
        let hd = cfg.head_dim();
        let key = |n: &str| weights::join_key(prefix, n);
        Ok(Self {
            to_q: Linear::load(map, &key("to_q"), d, d, cfg.qkv_bias)?,
            to_k: Linear::load(map, &key("to_k"), d, d, cfg.qkv_bias)?,
            to_v: Linear::load(map, &key("to_v"), d, d, cfg.qkv_bias)?,
            to_out: Linear::load(map, &key("to_out"), d, d, cfg.out_bias)?,
            norm_q: norm_weight(map, &key("norm_q.weight"), hd)?,
            norm_k: norm_weight(map, &key("norm_k.weight"), hd)?,
            heads: cfg.num_attention_heads,
            head_dim: hd,
            eps: cfg.norm_eps,
        })
    }

    fn forward(
        &self,
        x: &CudaTensor,
        cos: &CudaTensor,
        sin: &CudaTensor,
        pisa_sparsity: Option<f64>,
    ) -> Result<CudaTensor> {
        let [b, s, _] = match x.shape[..] {
            [b, s, d] => [b, s, d],
            _ => return Err(msg(format!("lingbot attn: {:?}", x.shape))),
        };
        let shape = vec![b, s, self.heads, self.head_dim];
        let qk = |lin: &Linear, norm: &CudaTensor| -> Result<CudaTensor> {
            lin.forward(x)?
                .reshape(shape.clone())?
                .rms_norm(norm, self.eps)?
                .apply_rotary_bshd(cos, sin)?
                .transpose(1, 2)
        };
        let q = qk(&self.to_q, &self.norm_q)?;
        let k = qk(&self.to_k, &self.norm_k)?;
        let v = self.to_v.forward(x)?.reshape(shape.clone())?.transpose(1, 2)?;
        let out = match pisa_sparsity {
            Some(sp) => crate::pisa_attn::pisa_attn(&q, &k, &v, sp, None)?,
            None => nn::scaled_dot_product_attention(&q, &k, &v, None)?,
        };
        self.to_out.forward(&out.merge_heads()?)
    }
}

struct Block {
    scale_shift_table: Vec<f32>,
    norm1: CudaTensor,
    norm2: CudaTensor,
    norm_post_attn: CudaTensor,
    norm_post_ffn: CudaTensor,
    attn: Attention,
    ffn: Ffn,
    eps: f32,
}

impl Block {
    fn zeros(cfg: &LingBotTransformerConfig, layer: usize) -> Result<Self> {
        let d = cfg.hidden_size;
        let ones = || vec_tensor(&vec![1.0; d], &[d]);
        Ok(Self {
            scale_shift_table: vec![0.0; 6 * d],
            norm1: ones()?,
            norm2: ones()?,
            norm_post_attn: ones()?,
            norm_post_ffn: ones()?,
            attn: Attention::zeros(cfg)?,
            ffn: if cfg.layer_is_moe(layer) {
                Ffn::Moe(Moe::zeros(cfg)?)
            } else {
                Ffn::Dense(Mlp::zeros(d, cfg.intermediate_size))
            },
            eps: cfg.norm_eps,
        })
    }

    fn load(map: &WeightMap, layer: usize, cfg: &LingBotTransformerConfig) -> Result<Self> {
        let d = cfg.hidden_size;
        let p = format!("blocks.{layer}");
        let key = |n: &str| weights::join_key(&p, n);
        let sst = weights::cuda_tensor_shaped(map, &key("scale_shift_table"), &[1, 6 * d])?;
        Ok(Self {
            scale_shift_table: sst.host_cow()?.into_owned(),
            norm1: norm_weight(map, &key("norm1.weight"), d)?,
            norm2: norm_weight(map, &key("norm2.weight"), d)?,
            norm_post_attn: norm_weight(map, &key("norm_post_attn.weight"), d)?,
            norm_post_ffn: norm_weight(map, &key("norm_post_ffn.weight"), d)?,
            attn: Attention::load(map, &key("attn"), cfg)?,
            ffn: if cfg.layer_is_moe(layer) {
                Ffn::Moe(Moe::load(map, &key("ffn"), cfg)?)
            } else {
                Ffn::Dense(Mlp::load(map, &key("ffn"), d, cfg.intermediate_size)?)
            },
            eps: cfg.norm_eps,
        })
    }

    /// `[shift_msa, 1 + scale_msa, tanh(gate_msa), shift_mlp, 1 + scale_mlp, tanh(gate_mlp)]`
    /// as `[1, 1, D]` device tensors.
    fn modulation(&self, temb6: &[f32], d: usize) -> Result<Vec<CudaTensor>> {
        let m: Vec<f32> = temb6
            .iter()
            .zip(&self.scale_shift_table)
            .map(|(a, b)| a + b)
            .collect();
        (0..6)
            .map(|c| {
                let part = &m[c * d..(c + 1) * d];
                let v: Vec<f32> = match c {
                    1 | 4 => part.iter().map(|x| 1.0 + x).collect(),
                    2 | 5 => part.iter().map(|x| x.tanh()).collect(),
                    _ => part.to_vec(),
                };
                vec_tensor(&v, &[1, 1, d])
            })
            .collect()
    }

    fn forward(
        &self,
        x: &CudaTensor,
        temb6: &[f32],
        cos: &CudaTensor,
        sin: &CudaTensor,
        pisa_sparsity: Option<f64>,
    ) -> Result<CudaTensor> {
        let d = *x.shape.last().ok_or_else(|| msg("lingbot block: scalar"))?;
        let m = self.modulation(temb6, d)?;
        let a_in = x.rms_norm(&self.norm1, self.eps)?.mul(&m[1])?.add(&m[0])?;
        let a = self.attn.forward(&a_in, cos, sin, pisa_sparsity)?;
        let x = x.add(&a.rms_norm(&self.norm_post_attn, self.eps)?.mul(&m[2])?)?;
        let f_in = x.rms_norm(&self.norm2, self.eps)?.mul(&m[4])?.add(&m[3])?;
        let f = self.ffn.forward(&f_in)?;
        x.add(&f.rms_norm(&self.norm_post_ffn, self.eps)?.mul(&m[5])?)
    }
}

/// Cached RoPE tables of one `(text_len, grid)` shape.
type RopeCache = Option<((usize, usize, usize, usize), CudaTensor, CudaTensor)>;

pub struct LingBotTransformer {
    pub cfg: LingBotTransformerConfig,
    patch: Linear,
    time_1: HostLinear,
    time_2: HostLinear,
    time_mod: HostLinear,
    out_mod: HostLinear,
    text_norm: CudaTensor,
    text_1: Linear,
    text_2: Linear,
    blocks: Vec<Block>,
    proj_out: Linear,
    rope: Mutex<RopeCache>,
}

impl LingBotTransformer {
    pub fn zeros(cfg: LingBotTransformerConfig) -> Result<Self> {
        cfg.validate().map_err(msg)?;
        let d = cfg.hidden_size;
        let blocks = (0..cfg.depth)
            .map(|i| Block::zeros(&cfg, i))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            patch: Linear::zeros(cfg.patch_dim(), d, true),
            time_1: HostLinear::zeros(cfg.freq_dim, d),
            time_2: HostLinear::zeros(d, d),
            time_mod: HostLinear::zeros(d, 6 * d),
            out_mod: HostLinear::zeros(d, 2 * d),
            text_norm: vec_tensor(&vec![1.0; cfg.text_dim], &[cfg.text_dim])?,
            text_1: Linear::zeros(cfg.text_dim, d, true),
            text_2: Linear::zeros(d, d, true),
            blocks,
            proj_out: Linear::zeros(d, cfg.out_patch_dim(), true),
            rope: Mutex::new(None),
            cfg,
        })
    }

    pub fn load(cfg: LingBotTransformerConfig, map: &WeightMap) -> Result<Self> {
        cfg.validate().map_err(msg)?;
        let d = cfg.hidden_size;
        let mut blocks = Vec::with_capacity(cfg.depth);
        for i in 0..cfg.depth {
            blocks.push(Block::load(map, i, &cfg)?);
        }
        Ok(Self {
            patch: Linear::load(map, "patch_embedder", cfg.patch_dim(), d, true)?,
            time_1: HostLinear::load(map, "time_embedder.linear_1", cfg.freq_dim, d)?,
            time_2: HostLinear::load(map, "time_embedder.linear_2", d, d)?,
            time_mod: HostLinear::load(map, "time_modulation.1", d, 6 * d)?,
            out_mod: HostLinear::load(map, "norm_out_modulation.1", d, 2 * d)?,
            text_norm: norm_weight(map, "text_embedder.norm.weight", cfg.text_dim)?,
            text_1: Linear::load(map, "text_embedder.linear_1", cfg.text_dim, d, true)?,
            text_2: Linear::load(map, "text_embedder.linear_2", d, d, true)?,
            blocks,
            proj_out: Linear::load(map, "proj_out", d, cfg.out_patch_dim(), true)?,
            rope: Mutex::new(None),
            cfg,
        })
    }

    /// `text_embedder` on Qwen hidden states `[1, L, text_dim]` → `[1, L, D]`.
    /// Step-invariant: compute once per prompt and stage.
    pub fn embed_text(&self, encoder: &CudaTensor) -> Result<CudaTensor> {
        let n = encoder.rms_norm(&self.text_norm, 1e-6)?;
        self.text_2.forward(&self.text_1.forward(&n)?.silu())
    }

    /// `t_emb` (`time_embedder(time_proj(t))`), `temb6` and the final
    /// `(shift, scale)` for scalar timestep `t` (in `[0, 1000]`).
    fn time_path(&self, t: f32) -> (Vec<f32>, Vec<f32>) {
        let f = timestep_sinusoid(t, self.cfg.freq_dim);
        let h = silu_host(&self.time_1.apply(&f));
        let temb = self.time_2.apply(&h);
        let st = silu_host(&temb);
        (self.time_mod.apply(&st), self.out_mod.apply(&st))
    }

    fn rope_for(&self, text_len: usize, gt: usize, gh: usize, gw: usize) -> Result<(CudaTensor, CudaTensor)> {
        let key = (text_len, gt, gh, gw);
        let mut slot = self.rope.lock().map_err(|_| msg("lingbot rope cache poisoned"))?;
        if let Some((k, c, s)) = slot.as_ref() {
            if *k == key {
                return Ok((c.clone(), s.clone()));
            }
        }
        let pos = joint_positions(text_len, gt, gh, gw);
        let (c, s) = rope_tables(&self.cfg, &pos);
        let hd = self.cfg.head_dim();
        let c = vec_tensor(&c, &[pos.len(), hd])?;
        let s = vec_tensor(&s, &[pos.len(), hd])?;
        *slot = Some((key, c.clone(), s.clone()));
        Ok((c, s))
    }

    /// Velocity for `latents` `[1, C, T, H, W]` at transformer timestep `t`
    /// (`sigma · 1000`), given `text` from [`Self::embed_text`] (`[1, L, D]`).
    pub fn forward(
        &self,
        latents: &CudaTensor,
        text: &CudaTensor,
        timestep: f32,
        route: AttnRoute<'_>,
    ) -> Result<CudaTensor> {
        let [b, c, t, h, w] = match latents.shape[..] {
            [b, c, t, h, w] => [b, c, t, h, w],
            _ => return Err(msg(format!("lingbot dit: {:?}", latents.shape))),
        };
        if b != 1 {
            return Err(msg("lingbot dit: batch 1 (CFG branches run as separate passes)"));
        }
        if c != self.cfg.in_channels {
            return Err(msg(format!("lingbot dit: {c} channels, want {}", self.cfg.in_channels)));
        }
        let [pt, ph, pw] = self.cfg.patch_size;
        if t % pt != 0 || h % ph != 0 || w % pw != 0 {
            return Err(msg(format!("lingbot dit: [{t},{h},{w}] vs patch {:?}", self.cfg.patch_size)));
        }
        let d = self.cfg.hidden_size;
        let (gt, gh, gw) = (t / pt, h / ph, w / pw);
        let n_video = gt * gh * gw;
        let text_len = text.shape.get(1).copied().unwrap_or(0);
        if text.shape != [1, text_len, d] {
            return Err(msg(format!("lingbot dit: text {:?}, want [1, L, {d}]", text.shape)));
        }

        let tokens = patchify(&latents.host_cow()?, c, t, h, w, self.cfg.patch_size);
        let x = self
            .patch
            .forward(&CudaTensor::from_vec(tokens, vec![n_video, self.cfg.patch_dim()])?)?
            .reshape(vec![1, n_video, d])?;
        let mut joint = CudaTensor::cat(&[&x, text], 1)?;
        let (cos, sin) = self.rope_for(text_len, gt, gh, gw)?;
        let (temb6, final_mod) = self.time_path(timestep);

        for (i, block) in self.blocks.iter().enumerate() {
            let sparsity = route
                .pisa
                .filter(|p| p.sparse_layer(i))
                .map(|p| p.sparsity());
            joint = block.forward(&joint, &temb6, &cos, &sin, sparsity)?;
        }

        let video = joint.narrow(1, 0, n_video)?;
        let shift = vec_tensor(&final_mod[..d], &[1, 1, d])?;
        let scale: Vec<f32> = final_mod[d..2 * d].iter().map(|v| 1.0 + v).collect();
        let scale = vec_tensor(&scale, &[1, 1, d])?;
        let y = nn::layer_norm(&video, self.cfg.norm_eps, None, None)?
            .mul(&scale)?
            .add(&shift)?;
        let out = self.proj_out.forward(&y)?;
        let pixels = unpatchify(&out.host_cow()?, self.cfg.out_channels, t, h, w, self.cfg.patch_size);
        CudaTensor::from_vec(pixels, vec![1, self.cfg.out_channels, t, h, w])
    }
}

/// `[C, T, H, W]` → `[(t h w), (pt ph pw c)]` (`patchify_and_embed` order).
pub fn patchify(x: &[f32], c: usize, t: usize, h: usize, w: usize, p: [usize; 3]) -> Vec<f32> {
    let [pt, ph, pw] = p;
    let (gt, gh, gw) = (t / pt, h / ph, w / pw);
    let dim = c * pt * ph * pw;
    let mut out = vec![0f32; gt * gh * gw * dim];
    for ti in 0..gt {
        for yi in 0..gh {
            for xi in 0..gw {
                let tok = (ti * gh + yi) * gw + xi;
                let mut o = tok * dim;
                for a in 0..pt {
                    for bb in 0..ph {
                        for cc in 0..pw {
                            for ci in 0..c {
                                out[o] = x[((ci * t + ti * pt + a) * h + yi * ph + bb) * w + xi * pw + cc];
                                o += 1;
                            }
                        }
                    }
                }
            }
        }
    }
    out
}

/// Inverse of [`patchify`] for `c` output channels.
pub fn unpatchify(y: &[f32], c: usize, t: usize, h: usize, w: usize, p: [usize; 3]) -> Vec<f32> {
    let [pt, ph, pw] = p;
    let (gt, gh, gw) = (t / pt, h / ph, w / pw);
    let dim = c * pt * ph * pw;
    let mut out = vec![0f32; c * t * h * w];
    for ti in 0..gt {
        for yi in 0..gh {
            for xi in 0..gw {
                let tok = (ti * gh + yi) * gw + xi;
                let mut o = tok * dim;
                for a in 0..pt {
                    for bb in 0..ph {
                        for cc in 0..pw {
                            for ci in 0..c {
                                out[((ci * t + ti * pt + a) * h + yi * ph + bb) * w + xi * pw + cc] = y[o];
                                o += 1;
                            }
                        }
                    }
                }
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn seeded(key: &str, shape: &[usize]) -> Vec<f32> {
        // Deterministic small weights from the key; norms near 1.
        let h = key.bytes().fold(1469598103934665603u64, |a, b| (a ^ b as u64).wrapping_mul(1099511628211));
        let n: usize = shape.iter().product();
        let is_norm = key.contains("norm") && key.ends_with(".weight");
        (0..n)
            .map(|i| {
                let v = (((h.wrapping_add((i as u64).wrapping_mul(0x9e3779b97f4a7c15))) >> 11) as f64 / (1u64 << 53) as f64) as f32;
                let u = (v - 0.5) * 0.4;
                if is_norm {
                    1.0 + u
                } else {
                    u
                }
            })
            .collect()
    }

    fn tiny(cfg: &LingBotTransformerConfig) -> LingBotTransformer {
        let map = WeightMap::generated(seeded);
        LingBotTransformer::load(cfg.clone(), &map).unwrap()
    }

    #[test]
    fn patchify_roundtrips() {
        let (c, t, h, w) = (3, 2, 4, 6);
        let x: Vec<f32> = (0..c * t * h * w).map(|i| i as f32).collect();
        let y = patchify(&x, c, t, h, w, [1, 2, 2]);
        assert_eq!(y.len(), x.len());
        // token 0 = (t0, y0, x0): features (ph, pw, c) with c innermost.
        assert_eq!(&y[..3], &[0.0, (t * h * w) as f32, (2 * t * h * w) as f32]);
        assert_eq!(unpatchify(&y, c, t, h, w, [1, 2, 2]), x);
    }

    #[test]
    fn timestep_embedding_is_cos_then_sin() {
        let e = timestep_sinusoid(0.0, 8);
        assert_eq!(e, vec![1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]);
        let e = timestep_sinusoid(500.0, 8);
        assert!((e[0] - 500f32.cos()).abs() < 1e-6 && (e[4] - 500f32.sin()).abs() < 1e-6);
    }

    #[test]
    fn tiny_dense_forward_shapes_and_finite() {
        let cfg = LingBotTransformerConfig::tiny();
        let dit = tiny(&cfg);
        let x: Vec<f32> = (0..cfg.in_channels * 2 * 4 * 4).map(|i| (i as f32 * 0.37).sin()).collect();
        let x = CudaTensor::from_vec(x, vec![1, cfg.in_channels, 2, 4, 4]).unwrap();
        let enc: Vec<f32> = (0..3 * cfg.text_dim).map(|i| (i as f32 * 0.11).cos()).collect();
        let enc = CudaTensor::from_vec(enc, vec![1, 3, cfg.text_dim]).unwrap();
        let text = dit.embed_text(&enc).unwrap();
        let out = dit.forward(&x, &text, 500.0, AttnRoute::default()).unwrap();
        assert_eq!(out.shape, vec![1, cfg.out_channels, 2, 4, 4]);
        assert!(out.host_cow().unwrap().iter().all(|v| v.is_finite()));
    }

    #[test]
    fn tiny_moe_forward_matches_dense_expert_reference() {
        // MoE layer output equals Σ_k w_k · expert_k(x) + shared(x) computed
        // token by token with the host router: the dispatch / gather /
        // combine path must be a pure permutation.
        let cfg = LingBotTransformerConfig::tiny_moe();
        let map = WeightMap::generated(seeded);
        let moe = Moe::load(&map, "blocks.1.ffn", &cfg).unwrap();
        let (n, d) = (7usize, cfg.hidden_size);
        let x: Vec<f32> = (0..n * d).map(|i| (i as f32 * 0.173).sin()).collect();
        let xt = CudaTensor::from_vec(x.clone(), vec![1, n, d]).unwrap();
        let got = moe.forward(&xt).unwrap().host_cow().unwrap().into_owned();

        let logits = CudaTensor::from_vec(x.clone(), vec![n, d])
            .unwrap()
            .matmul(&moe.router_t)
            .unwrap();
        let (idx, w) = fastvideo_models::lingbot::route(&logits.host_cow().unwrap(), &moe.bias, &moe.spec);
        let k = cfg.num_experts_per_tok;
        for t in 0..n {
            let row = CudaTensor::from_vec(x[t * d..(t + 1) * d].to_vec(), vec![1, d]).unwrap();
            let mut want = moe.shared.as_ref().unwrap().forward(&row).unwrap().host_cow().unwrap().into_owned();
            for s in 0..k {
                let e = idx[t * k + s] as usize;
                let y = moe.experts[e].forward(&row).unwrap();
                for (a, b) in want.iter_mut().zip(y.host_cow().unwrap().iter()) {
                    *a += w[t * k + s] * b;
                }
            }
            for j in 0..d {
                assert!((got[t * d + j] - want[j]).abs() < 1e-5, "t{t} j{j}");
            }
        }
    }

    #[test]
    fn moe_chunking_does_not_change_the_result() {
        let cfg = LingBotTransformerConfig::tiny_moe();
        let map = WeightMap::generated(seeded);
        let moe = Moe::load(&map, "blocks.1.ffn", &cfg).unwrap();
        let (n, d) = (9usize, cfg.hidden_size);
        let x: Vec<f32> = (0..n * d).map(|i| (i as f32 * 0.29).cos()).collect();
        let xt = CudaTensor::from_vec(x, vec![1, n, d]).unwrap();
        let whole = moe.forward(&xt).unwrap().host_cow().unwrap().into_owned();
        std::env::set_var("FASTVIDEO_LINGBOT_MOE_CHUNK", "4");
        let chunked = moe.forward(&xt).unwrap().host_cow().unwrap().into_owned();
        std::env::remove_var("FASTVIDEO_LINGBOT_MOE_CHUNK");
        for (a, b) in whole.iter().zip(&chunked) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn tiny_moe_full_forward_and_pisa_route() {
        let cfg = LingBotTransformerConfig::tiny_moe();
        let dit = tiny(&cfg);
        let x: Vec<f32> = (0..cfg.in_channels * 2 * 4 * 4).map(|i| (i as f32 * 0.21).cos()).collect();
        let x = CudaTensor::from_vec(x, vec![1, cfg.in_channels, 2, 4, 4]).unwrap();
        let enc = CudaTensor::from_vec(vec![0.1; 2 * cfg.text_dim], vec![1, 2, cfg.text_dim]).unwrap();
        let text = dit.embed_text(&enc).unwrap();
        let dense = dit.forward(&x, &text, 700.0, AttnRoute::default()).unwrap();
        let mut policy = PisaPolicy::published();
        policy.dense_layers = 0..=0;
        policy.density = 1.0; // keep every block: PISA must equal dense
        let pisa = dit.forward(&x, &text, 700.0, AttnRoute { pisa: Some(&policy) }).unwrap();
        let (a, b) = (dense.host_cow().unwrap(), pisa.host_cow().unwrap());
        for (u, v) in a.iter().zip(b.iter()) {
            assert!((u - v).abs() < 1e-4, "{u} vs {v}");
        }
    }

    /// Golden: `scripts/ref/lingbot_reference.py`, a NumPy transcription of
    /// the upstream `LingBotVideoTransformer3DModel.forward`, with random
    /// tiny-MoE weights (group routing, shared expert, dense layer 0).
    #[test]
    fn golden_tiny_moe_matches_numpy_reference() {
        let dir = std::env::temp_dir().join(format!("lingbot-golden-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("model.safetensors"),
            include_bytes!("fixtures/tiny_moe_golden.st"),
        )
        .unwrap();
        let map = WeightMap::load_dir(&dir).unwrap();
        let cfg = LingBotTransformerConfig::tiny_moe();
        let dit = LingBotTransformer::load(cfg, &map).unwrap();
        let get = |k: &str| {
            let (shape, v) = map.get_f32(k).unwrap();
            CudaTensor::from_vec(v, shape).unwrap()
        };
        let lat = get("test.latents");
        let text = dit.embed_text(&get("test.text")).unwrap();
        let t = get("test.timestep").host_cow().unwrap()[0];
        let want = get("test.expected");
        let got = dit.forward(&lat, &text, t, AttnRoute::default()).unwrap();
        assert_eq!(got.shape, want.shape);
        let (g, w) = (got.host_cow().unwrap(), want.host_cow().unwrap());
        let mut max = 0f32;
        for (a, b) in g.iter().zip(w.iter()) {
            max = max.max((a - b).abs());
        }
        let _ = std::fs::remove_dir_all(&dir);
        assert!(max < 2e-4, "max |rust - numpy| = {max}");
    }

    #[test]
    fn zeros_graph_runs() {
        let cfg = LingBotTransformerConfig::tiny_moe();
        let dit = LingBotTransformer::zeros(cfg.clone()).unwrap();
        let x = CudaTensor::zeros(&[1, cfg.in_channels, 1, 2, 2]);
        let text = CudaTensor::zeros(&[1, 2, cfg.hidden_size]);
        let out = dit.forward(&x, &text, 10.0, AttnRoute::default()).unwrap();
        assert_eq!(out.shape, vec![1, cfg.out_channels, 1, 2, 2]);
    }
}
