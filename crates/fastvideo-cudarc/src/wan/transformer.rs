//! WanTransformer3D on CudaTensor.
//!
//! Per block on the device: one fused QKV GEMM, one q/k RMSNorm+RoPE kernel
//! each (written straight into BHSD), dense cuBLAS attention, AdaLN and gated
//! residuals read from the `[b, 6, dim]` modulation table without chunk
//! copies, and a bias+GELU fused FFN.

use fastvideo_models::wan::sol::{
    morton3d_on_route, pisa_requested, route, sol_attn_requested, WanAttnProfile, WanAttnRoute,
};
use fastvideo_models::wan::sol_cache::{A14bCacheController, A14bExpert};
use fastvideo_models::wan::WanVideoArchConfig;

use super::fused::Rope;
#[cfg(feature = "cuda")]
use super::vsa::VsaCtx;
/// Placeholder so block signatures are the same shape on CPU builds.
#[cfg(not(feature = "cuda"))]
pub struct VsaCtx;
use super::nn::{self, Linear};
use super::tensor::{CudaTensor, Result, TensorError};
use super::weights::{self, WeightMap};

fn pinned(mut t: CudaTensor) -> Result<CudaTensor> {
    t.pin_device()?;
    Ok(t)
}

#[derive(Debug, Clone)]
struct WanAttention {
    /// Self-attention: [q; k; v] rows. Cross-attention: q only.
    q_or_qkv: Linear,
    /// Cross-attention [k; v] rows over the encoder states.
    kv: Option<Linear>,
    to_out: Linear,
    norm_q: CudaTensor,
    norm_k: CudaTensor,
    add_k: Option<Linear>,
    add_v: Option<Linear>,
    /// TurboWan SLA linear-branch projection (`head_dim → head_dim`); self-attn only.
    proj_l: Option<Linear>,
    heads: usize,
    dim_head: usize,
    eps: f32,
}

impl WanAttention {
    fn zeros(
        dim: usize,
        heads: usize,
        eps: f32,
        cross: bool,
        added_kv: Option<usize>,
    ) -> Result<Self> {
        let (add_k, add_v) = match added_kv {
            Some(extra) => (
                Some(Linear::zeros(extra, dim, true)),
                Some(Linear::zeros(extra, dim, true)),
            ),
            None => (None, None),
        };
        Ok(Self {
            q_or_qkv: Linear::zeros(dim, if cross { dim } else { 3 * dim }, true),
            kv: cross.then(|| Linear::zeros(dim, 2 * dim, true)),
            to_out: Linear::zeros(dim, dim, true),
            norm_q: pinned(CudaTensor::ones(&[dim]))?,
            norm_k: pinned(CudaTensor::ones(&[dim]))?,
            add_k,
            add_v,
            proj_l: if cross || !super::sla::sla_enabled() {
                None
            } else {
                Some(Linear::zeros(dim / heads, dim / heads, true))
            },
            heads,
            dim_head: dim / heads,
            eps,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn load(
        map: &WeightMap,
        prefix: &str,
        dim: usize,
        heads: usize,
        eps: f32,
        cross: bool,
        added_kv: Option<usize>,
    ) -> Result<Self> {
        let key = |name: &str| weights::join_key(prefix, name);
        let (add_k, add_v) = match added_kv {
            Some(extra) => (
                Some(Linear::load(map, &key("add_k_proj"), extra, dim, true)?),
                Some(Linear::load(map, &key("add_v_proj"), extra, dim, true)?),
            ),
            None => (None, None),
        };
        let (q_or_qkv, kv) = if cross {
            (
                Linear::load(map, &key("to_q"), dim, dim, true)?,
                Some(Linear::load_fused(
                    map,
                    &[&key("to_k"), &key("to_v")],
                    dim,
                    dim,
                    true,
                )?),
            )
        } else {
            (
                Linear::load_fused(
                    map,
                    &[&key("to_q"), &key("to_k"), &key("to_v")],
                    dim,
                    dim,
                    true,
                )?,
                None,
            )
        };
        let dim_head = dim / heads;
        let proj_l = if cross {
            None
        } else {
            // Prefer checkpoint proj_l; otherwise zeros so SLA still runs (o_l→0).
            match super::sla::load_proj_l(map, prefix, dim_head)? {
                Some(p) => Some(p),
                None if super::sla::sla_enabled() => Some(Linear::zeros(dim_head, dim_head, true)),
                None => None,
            }
        };
        Ok(Self {
            q_or_qkv,
            kv,
            to_out: Linear::load(map, &key("to_out.0"), dim, dim, true)?,
            norm_q: pinned(weights::cuda_tensor_shaped(
                map,
                &key("norm_q.weight"),
                &[dim],
            )?)?,
            norm_k: pinned(weights::cuda_tensor_shaped(
                map,
                &key("norm_k.weight"),
                &[dim],
            )?)?,
            add_k,
            add_v,
            proj_l,
            heads,
            dim_head,
            eps,
        })
    }

    fn attend(
        &self,
        q: &CudaTensor,
        k: &CudaTensor,
        v: &CudaTensor,
        mask: Option<&CudaTensor>,
    ) -> Result<CudaTensor> {
        let attn = nn::scaled_dot_product_attention_masked(q, k, v, None, mask)?;
        self.to_out.forward(&attn.merge_heads()?)
    }

    fn attend_routed(
        &self,
        q: &CudaTensor,
        k: &CudaTensor,
        v: &CudaTensor,
        plan: AttnPlan,
    ) -> Result<Option<CudaTensor>> {
        let scale = Some((self.dim_head as f32).sqrt().recip());
        match route(plan.profile, plan.step, plan.layer) {
            WanAttnRoute::Dense => Ok(None),
            WanAttnRoute::Sol { tau } => {
                let (q, k, v, inverse) = if morton3d_on_route(plan.profile, plan.step, plan.layer) {
                    if let Some((frames, height, width)) = plan.morton_grid {
                        if frames.saturating_mul(height).saturating_mul(width) == q.shape[2] {
                            let (perm, inverse) =
                                super::sol_cache::morton3d_pair(frames, height, width);
                            (
                                super::sol_cache::gather_bhsd_seq(q, &perm)?,
                                super::sol_cache::gather_bhsd_seq(k, &perm)?,
                                super::sol_cache::gather_bhsd_seq(v, &perm)?,
                                Some(inverse),
                            )
                        } else {
                            (q.clone(), k.clone(), v.clone(), None)
                        }
                    } else {
                        (q.clone(), k.clone(), v.clone(), None)
                    }
                } else {
                    (q.clone(), k.clone(), v.clone(), None)
                };
                let mut out = crate::sol_attn::sol_attn(&q, &k, &v, tau, scale, None, 0)?;
                if let Some(inverse) = inverse {
                    out = super::sol_cache::gather_bhsd_seq(&out, &inverse)?;
                }
                Ok(Some(out))
            }
            WanAttnRoute::Pisa { sparsity } => {
                Ok(Some(crate::pisa_attn::pisa_attn(q, k, v, sparsity, scale)?))
            }
        }
    }

    fn forward_self(
        &self,
        hidden: &CudaTensor,
        rope: &(CudaTensor, CudaTensor),
        mask: Option<&CudaTensor>,
        gate: Option<&Linear>,
        vsa: Option<&VsaCtx>,
        ar: Option<&super::ar_cache::ArFrame>,
        plan: AttnPlan,
    ) -> Result<CudaTensor> {
        let dim = self.heads * self.dim_head;
        let qkv = self.q_or_qkv.forward(hidden)?;
        let rope = || {
            Some(Rope {
                cos: &rope.0,
                sin: &rope.1,
            })
        };
        let q = qkv.qk_norm_rope_bhsd(0, self.heads, &self.norm_q, rope(), self.eps)?;
        let k = qkv.qk_norm_rope_bhsd(dim, self.heads, &self.norm_k, rope(), self.eps)?;
        let v = qkv.split_heads_bhsd(2 * dim, self.heads, self.dim_head)?;
        // Causal + NVFP4 writes each frame into the rolling cache and attends
        // the dequantized span. RoPE is already on Q/K. AdaLN stays on the
        // hidden `[B,S,C]` tensor, which does not share that layout.
        if let Some(spec) = ar {
            let out = super::ar_cache::attend_cached(&q, &k, &v, spec)?;
            return self.to_out.forward(&out.merge_heads()?);
        }
        let (k, v) = super::nvfp4::maybe_kv(k, v)?;
        if mask.is_none() {
            if let Some(out) = self.attend_routed(&q, &k, &v, plan)? {
                return self.to_out.forward(&out.merge_heads()?);
            }
        }
        // TurboWan SLA: block top-k sparse + linear attention (self-attn only).
        if super::sla::sla_enabled() && mask.is_none() {
            let cfg = super::sla::SlaConfig::from_env();
            let out = super::sla::sla_attention(&q, &k, &v, self.proj_l.as_ref(), &cfg)?;
            return self.to_out.forward(&out.merge_heads()?);
        }
        // VSA needs both the tiling for this grid and the checkpoint's gate;
        // without either it is not the configuration the weights were trained
        // for, so fall back to the dense path rather than approximate it.
        if let (Some(ctx), Some(gate)) = (vsa, gate) {
            // The gate shares q/k/v's projection input but takes no RoPE and
            // no norm, matching upstream.
            let g = gate
                .forward(hidden)?
                .split_heads_bhsd(0, self.heads, self.dim_head)?;
            if let Some(out) = self.attend_vsa(&q, &k, &v, &g, ctx)? {
                // Same tail as the dense path: VSA returns BHSD attention, which
                // still has to be merged back to [b, seq, dim] and projected.
                return self.to_out.forward(&out.merge_heads()?);
            }
        }
        self.attend(&q, &k, &v, mask)
    }

    /// VSA over `[b, heads, seq, dim]` inputs; `None` when the device path is
    /// unavailable, so the caller can use dense attention instead.
    #[cfg(feature = "cuda")]
    fn attend_vsa(
        &self,
        q: &CudaTensor,
        k: &CudaTensor,
        v: &CudaTensor,
        gate: &CudaTensor,
        ctx: &VsaCtx,
    ) -> Result<Option<CudaTensor>> {
        let [b, heads, seq, dim] = q.shape[..] else {
            return Ok(None);
        };
        if seq != ctx.seq {
            return Ok(None);
        }
        let (Some(qd), Some(kd), Some(vd), Some(gd)) = (q.dev()?, k.dev()?, v.dev()?, gate.dev()?)
        else {
            return Ok(None);
        };
        let scale = 1.0 / (dim as f32).sqrt();
        let out = super::vsa::vsa_attention_device(
            &qd,
            &kd,
            &vd,
            Some(&gd),
            &ctx.plan,
            ctx.topk,
            b * heads,
            seq,
            dim,
            scale,
            ctx.group,
        )?;
        Ok(Some(CudaTensor::from_device_slice(
            out,
            vec![b, heads, seq, dim],
        )?))
    }

    #[cfg(not(feature = "cuda"))]
    fn attend_vsa(
        &self,
        _q: &CudaTensor,
        _k: &CudaTensor,
        _v: &CudaTensor,
        _gate: &CudaTensor,
        _ctx: &VsaCtx,
    ) -> Result<Option<CudaTensor>> {
        Ok(None)
    }

    fn forward_cross(
        &self,
        hidden: &CudaTensor,
        encoder: &CudaTensor,
        image: Option<&CudaTensor>,
    ) -> Result<CudaTensor> {
        let dim = self.heads * self.dim_head;
        let kv_proj = self
            .kv
            .as_ref()
            .ok_or_else(|| TensorError::Message("cross attention without kv".into()))?;
        let q = self.q_or_qkv.forward(hidden)?.qk_norm_rope_bhsd(
            0,
            self.heads,
            &self.norm_q,
            None,
            self.eps,
        )?;
        let kv = kv_proj.forward(encoder)?;
        let mut k = kv.qk_norm_rope_bhsd(0, self.heads, &self.norm_k, None, self.eps)?;
        let mut v = kv.split_heads_bhsd(dim, self.heads, self.dim_head)?;
        if let (Some(add_k), Some(add_v), Some(img)) = (&self.add_k, &self.add_v, image) {
            let ik = add_k
                .forward(img)?
                .split_heads_bhsd(0, self.heads, self.dim_head)?;
            let iv = add_v
                .forward(img)?
                .split_heads_bhsd(0, self.heads, self.dim_head)?;
            k = CudaTensor::cat(&[&ik, &k], 2)?;
            v = CudaTensor::cat(&[&iv, &v], 2)?;
        }
        let (k, v) = super::nvfp4::maybe_kv(k, v)?;
        self.attend(&q, &k, &v, None)
    }
}

fn rotary_1d(dim: usize, seq: usize, theta: f64) -> (Vec<f32>, Vec<f32>) {
    let half = dim / 2;
    let mut cos = vec![0.0f32; seq * dim];
    let mut sin = vec![0.0f32; seq * dim];
    for p in 0..seq {
        for i in 0..half {
            let freq = 1.0 / theta.powf(2.0 * i as f64 / dim as f64) as f32;
            let arg = p as f32 * freq;
            // repeat_interleave 2
            for o in [p * dim + 2 * i, p * dim + 2 * i + 1] {
                cos[o] = arg.cos();
                sin[o] = arg.sin();
            }
        }
    }
    (cos, sin)
}

/// 3-D RoPE tables `[seq, head_dim]` (time, height, width split of the head).
fn wan_rope(
    cfg: &WanVideoArchConfig,
    frames: usize,
    height: usize,
    width: usize,
) -> Result<(CudaTensor, CudaTensor)> {
    let d = cfg.attention_head_dim;
    let h_dim = 2 * (d / 6);
    let w_dim = h_dim;
    let t_dim = d - h_dim - w_dim;
    let axes = [
        (t_dim, rotary_1d(t_dim, cfg.rope_max_seq_len, 10000.0)),
        (h_dim, rotary_1d(h_dim, cfg.rope_max_seq_len, 10000.0)),
        (w_dim, rotary_1d(w_dim, cfg.rope_max_seq_len, 10000.0)),
    ];
    let (ppf, pph, ppw) = (
        frames / cfg.patch_size[0],
        height / cfg.patch_size[1],
        width / cfg.patch_size[2],
    );
    let seq = ppf * pph * ppw;
    let mut cos = Vec::with_capacity(seq * d);
    let mut sin = Vec::with_capacity(seq * d);
    for ft in 0..ppf {
        for fh in 0..pph {
            for fw in 0..ppw {
                for ((ad, (c, s)), pos) in axes.iter().zip([ft, fh, fw]) {
                    cos.extend_from_slice(&c[pos * ad..(pos + 1) * ad]);
                    sin.extend_from_slice(&s[pos * ad..(pos + 1) * ad]);
                }
            }
        }
    }
    Ok((
        CudaTensor::from_vec(cos, vec![seq, d])?,
        CudaTensor::from_vec(sin, vec![seq, d])?,
    ))
}

#[derive(Debug, Clone)]
struct FeedForward {
    proj: Linear,
    out: Linear,
}

impl FeedForward {
    fn forward(&self, xs: &CudaTensor) -> Result<CudaTensor> {
        self.out.forward(&self.proj.forward_gelu(xs)?)
    }
}

#[derive(Debug, Clone)]
struct MlpEmbed {
    linear_1: Linear,
    linear_2: Linear,
}

impl MlpEmbed {
    fn zeros(in_dim: usize, dim: usize) -> Self {
        Self {
            linear_1: Linear::zeros(in_dim, dim, true),
            linear_2: Linear::zeros(dim, dim, true),
        }
    }

    fn load(map: &WeightMap, prefix: &str, in_dim: usize, dim: usize) -> Result<Self> {
        Ok(Self {
            linear_1: Linear::load(
                map,
                &weights::join_key(prefix, "linear_1"),
                in_dim,
                dim,
                true,
            )?,
            linear_2: Linear::load(map, &weights::join_key(prefix, "linear_2"), dim, dim, true)?,
        })
    }

    /// Text projection: GELU-tanh MLP.
    fn forward_gelu(&self, xs: &CudaTensor) -> Result<CudaTensor> {
        self.linear_2.forward(&self.linear_1.forward_gelu(xs)?)
    }

    /// Timestep embedding: SiLU MLP.
    fn forward_silu(&self, xs: &CudaTensor) -> Result<CudaTensor> {
        self.linear_2.forward(&self.linear_1.forward(xs)?.silu())
    }
}

#[derive(Debug, Clone)]
struct ImageEmbedder {
    norm1_w: CudaTensor,
    norm1_b: CudaTensor,
    proj: Linear,
    out: Linear,
    norm2_w: CudaTensor,
    norm2_b: CudaTensor,
}

impl ImageEmbedder {
    fn load(map: &WeightMap, prefix: &str, in_dim: usize, out_dim: usize) -> Result<Self> {
        let key = |name: &str| weights::join_key(prefix, name);
        Ok(Self {
            norm1_w: pinned(weights::cuda_tensor_shaped(
                map,
                &key("norm1.weight"),
                &[in_dim],
            )?)?,
            norm1_b: pinned(weights::cuda_tensor_shaped(
                map,
                &key("norm1.bias"),
                &[in_dim],
            )?)?,
            proj: Linear::load(map, &key("ff.net.0.proj"), in_dim, in_dim, true)?,
            out: Linear::load(map, &key("ff.net.2"), in_dim, out_dim, true)?,
            norm2_w: pinned(weights::cuda_tensor_shaped(
                map,
                &key("norm2.weight"),
                &[out_dim],
            )?)?,
            norm2_b: pinned(weights::cuda_tensor_shaped(
                map,
                &key("norm2.bias"),
                &[out_dim],
            )?)?,
        })
    }

    fn forward(&self, xs: &CudaTensor) -> Result<CudaTensor> {
        let x = xs.layer_norm(1e-5, Some(&self.norm1_w), Some(&self.norm1_b))?;
        let x = self.out.forward(&self.proj.forward_gelu(&x)?)?;
        x.layer_norm(1e-5, Some(&self.norm2_w), Some(&self.norm2_b))
    }
}

#[derive(Debug, Clone, Copy)]
struct AttnPlan {
    step: usize,
    layer: usize,
    profile: WanAttnProfile,
    morton_grid: Option<(usize, usize, usize)>,
}

impl Default for AttnPlan {
    fn default() -> Self {
        Self {
            step: 0,
            layer: 0,
            profile: WanAttnProfile::Off,
            morton_grid: None,
        }
    }
}

/// Modulation table slots of [`WanBlock::scale_shift_table`] + time projection.
const SHIFT_MSA: usize = 0;
const SCALE_MSA: usize = 1;
const GATE_MSA: usize = 2;
const SHIFT_FFN: usize = 3;
const SCALE_FFN: usize = 4;
const GATE_FFN: usize = 5;

#[derive(Debug, Clone)]
struct WanBlock {
    eps: f32,
    attn1: WanAttention,
    /// `to_gate_compress`, present only on VSA checkpoints.
    gate: Option<Linear>,
    attn2: WanAttention,
    norm2_weight: CudaTensor,
    norm2_bias: CudaTensor,
    ffn: FeedForward,
    scale_shift_table: CudaTensor, // [1, 6, dim]
}

impl WanBlock {
    fn zeros(cfg: &WanVideoArchConfig) -> Result<Self> {
        let dim = cfg.hidden_size();
        let heads = cfg.num_attention_heads;
        Ok(Self {
            eps: cfg.eps,
            attn1: WanAttention::zeros(dim, heads, cfg.eps, false, None)?,
            gate: None,
            attn2: WanAttention::zeros(dim, heads, cfg.eps, true, cfg.added_kv_proj_dim)?,
            norm2_weight: pinned(CudaTensor::ones(&[dim]))?,
            norm2_bias: pinned(CudaTensor::zeros(&[dim]))?,
            ffn: FeedForward {
                proj: Linear::zeros(dim, cfg.ffn_dim, true),
                out: Linear::zeros(cfg.ffn_dim, dim, true),
            },
            scale_shift_table: pinned(CudaTensor::zeros(&[1, 6, dim]))?,
        })
    }

    fn load(map: &WeightMap, prefix: &str, cfg: &WanVideoArchConfig) -> Result<Self> {
        let dim = cfg.hidden_size();
        let heads = cfg.num_attention_heads;
        let key = |name: &str| weights::join_key(prefix, name);
        Ok(Self {
            eps: cfg.eps,
            attn1: WanAttention::load(map, &key("attn1"), dim, heads, cfg.eps, false, None)?,
            // VSA checkpoints carry a per-block gate for the coarse branch;
            // dense checkpoints simply do not have it.
            gate: map
                .contains(&key("to_gate_compress.weight"))
                .then(|| Linear::load(map, &key("to_gate_compress"), dim, dim, true))
                .transpose()?,
            attn2: WanAttention::load(
                map,
                &key("attn2"),
                dim,
                heads,
                cfg.eps,
                true,
                cfg.added_kv_proj_dim,
            )?,
            norm2_weight: pinned(weights::cuda_tensor_shaped(
                map,
                &key("norm2.weight"),
                &[dim],
            )?)?,
            norm2_bias: pinned(weights::cuda_tensor_shaped(
                map,
                &key("norm2.bias"),
                &[dim],
            )?)?,
            ffn: FeedForward {
                proj: Linear::load(map, &key("ffn.net.0.proj"), dim, cfg.ffn_dim, true)?,
                out: Linear::load(map, &key("ffn.net.2"), cfg.ffn_dim, dim, true)?,
            },
            scale_shift_table: pinned(weights::cuda_tensor_shaped(
                map,
                &key("scale_shift_table"),
                &[1, 6, dim],
            )?)?,
        })
    }

    fn forward(
        &self,
        hidden: &CudaTensor,
        encoder: &CudaTensor,
        timestep_proj: &CudaTensor,
        rope: &(CudaTensor, CudaTensor),
        image: Option<&CudaTensor>,
        mask: Option<&CudaTensor>,
        vsa: Option<&VsaCtx>,
        ar: Option<&super::ar_cache::ArFrame>,
        plan: AttnPlan,
    ) -> Result<CudaTensor> {
        use super::stats::phase;
        let e = timestep_proj.add(&self.scale_shift_table)?;
        let normed = phase("1_norm_msa", || {
            hidden.ln_adaln_e(&e, SCALE_MSA, SHIFT_MSA, self.eps)
        })?;
        let attn = phase("2_self_attn", || {
            self.attn1
                .forward_self(&normed, rope, mask, self.gate.as_ref(), vsa, ar, plan)
        })?;
        let hidden = phase("3_residual_msa", || {
            hidden.residual_gate_add_e(&attn, &e, GATE_MSA)
        })?;

        let normed = phase("4_norm_cross", || {
            hidden.layer_norm(self.eps, Some(&self.norm2_weight), Some(&self.norm2_bias))
        })?;
        let hidden = phase("5_cross_attn", || {
            Ok::<_, TensorError>(hidden.add(&self.attn2.forward_cross(&normed, encoder, image)?)?)
        })?;

        let normed = phase("6_norm_ffn", || {
            hidden.ln_adaln_e(&e, SCALE_FFN, SHIFT_FFN, self.eps)
        })?;
        let ff = phase("7_ffn", || self.ffn.forward(&normed))?;
        phase("8_residual_ffn", || {
            hidden.residual_gate_add_e(&ff, &e, GATE_FFN)
        })
    }
}

#[derive(Debug, Clone)]
pub struct WanTransformer3D {
    pub cfg: WanVideoArchConfig,
    patch_weight: CudaTensor, // [dim, in_c, pt, ph, pw]
    patch_bias: CudaTensor,
    time_embedder: MlpEmbed,
    time_proj: Linear,
    text_embedder: MlpEmbed,
    image_embedder: Option<ImageEmbedder>,
    blocks: Vec<WanBlock>,
    proj_out: Linear,
    scale_shift_table: CudaTensor, // [1, 2, dim]
    freq_dim: usize,
    /// RoPE tables keyed by `(seq_len, head_dim)`: built and uploaded once.
    rotary_cache: std::sync::Arc<
        std::sync::Mutex<std::collections::HashMap<(usize, usize), (CudaTensor, CudaTensor)>>,
    >,
    /// VSA tiling per latent grid, built once and shared by every layer.
    #[cfg(feature = "cuda")]
    vsa_cache: std::sync::Arc<
        std::sync::Mutex<std::collections::HashMap<(usize, usize, usize), std::sync::Arc<VsaCtx>>>,
    >,
    /// Sol TeaCache block residual. Empty unless `FASTVIDEO_WAN_SOL_CACHE=teacache`.
    sol_tea: std::sync::Arc<std::sync::Mutex<Option<SolTeaRuntime>>>,
    /// Sol TaylorSeer lite. Empty unless `FASTVIDEO_WAN_SOL_CACHE=taylorseer`.
    sol_taylor: std::sync::Arc<std::sync::Mutex<Option<SolTaylorRuntime>>>,
    /// A14B EasyCache: block-0 fresh + blocks 1..=39 residual.
    sol_a14b: std::sync::Arc<std::sync::Mutex<Option<A14bRuntime>>>,
    attn_step: std::sync::Arc<std::sync::Mutex<Option<usize>>>,
    attn_profile: std::sync::Arc<std::sync::Mutex<WanAttnProfile>>,
}

use fastvideo_models::wan::sol_cache::{SolTeaCache, TaylorSchedule, TeaBranch};

/// Per-branch block residual for Sol TeaCache. Armed by the denoise loop.
#[derive(Debug)]
struct SolTeaRuntime {
    state: SolTeaCache,
    pending: Option<TeaArm>,
    active: Option<TeaActive>,
    cond_signal: Option<CudaTensor>,
    uncond_signal: Option<CudaTensor>,
    cond_residual: Option<CudaTensor>,
    uncond_residual: Option<CudaTensor>,
    /// Both CFG rows, used when cond and uncond share one forward.
    batch_residual: Option<CudaTensor>,
}

#[derive(Debug, Clone, Copy)]
enum TeaArm {
    One(TeaBranch, usize),
    /// Row 0 is uncond, row 1 is cond.
    Batch(usize),
}

#[derive(Debug, Clone, Copy)]
enum TeaActive {
    One(TeaBranch),
    Batch,
}

/// TaylorSeer lite: forecast `proj_out`, skip the blocks on a forecast step.
#[derive(Debug)]
struct SolTaylorRuntime {
    schedule: TaylorSchedule,
    compute: bool,
    last_update: Option<isize>,
    factors: Vec<CudaTensor>,
}

/// A14B EasyCache runtime: block 0 always runs; tail residual is per CFG branch.
#[derive(Debug)]
struct A14bRuntime {
    state: A14bCacheController,
    pending: Option<(bool, usize)>,
    active_cond: bool,
    reuse: bool,
    previous_input: Option<CudaTensor>,
    last_compute_input: Option<CudaTensor>,
    last_compute_output: Option<CudaTensor>,
    cond_residual: Option<CudaTensor>,
    uncond_residual: Option<CudaTensor>,
}

fn mean_abs(t: &CudaTensor) -> Result<f64> {
    let host = t.host_cow()?;
    let n = host.len().max(1) as f64;
    Ok(host.iter().map(|v| f64::from(*v).abs()).sum::<f64>() / n)
}

fn mean_abs_delta(a: &CudaTensor, b: &CudaTensor) -> Result<f64> {
    let left = a.host_cow()?;
    let right = b.host_cow()?;
    let n = left.len().max(1) as f64;
    let mut sum = 0.0;
    for (x, y) in left.iter().zip(right.iter()) {
        sum += (f64::from(*x) - f64::from(*y)).abs();
    }
    Ok(sum / n)
}

fn relative_l1(current: &CudaTensor, previous: &CudaTensor) -> Result<f64> {
    let cur = current.host_cow()?;
    let prev = previous.host_cow()?;
    let n = cur.len().max(1) as f64;
    let mut num = 0.0;
    let mut den = 0.0;
    for (a, b) in cur.iter().zip(prev.iter()) {
        num += (f64::from(*a) - f64::from(*b)).abs();
        den += f64::from(*b).abs();
    }
    Ok((num / n) / (den / n).max(1e-8))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TaylorPhase {
    Off,
    Compute,
    Forecast,
}

fn env_usize_alt(primary: &str, fallback: &str, default: usize) -> usize {
    std::env::var(primary)
        .ok()
        .and_then(|s| s.parse().ok())
        .or_else(|| std::env::var(fallback).ok().and_then(|s| s.parse().ok()))
        .unwrap_or(default)
}

fn decide_one_branch(
    runtime: &mut SolTeaRuntime,
    signal: &CudaTensor,
    branch: TeaBranch,
    step: usize,
) -> Result<Option<bool>> {
    if signal.shape.first().copied().unwrap_or(0) != 1 {
        return Err(TensorError::Message(
            "wan sol teacache single-branch forward has a batch other than 1".into(),
        ));
    }
    let rel = if runtime.state.needs_signal(branch, step) {
        let previous = match branch {
            TeaBranch::Cond => runtime.cond_signal.as_ref(),
            TeaBranch::Uncond => runtime.uncond_signal.as_ref(),
        }
        .expect("teacache signal");
        relative_l1(signal, previous)?
    } else {
        0.0
    };
    let decision = runtime.state.decide(branch, step, rel);
    match branch {
        TeaBranch::Cond => runtime.cond_signal = Some(signal.clone()),
        TeaBranch::Uncond => runtime.uncond_signal = Some(signal.clone()),
    }
    runtime.active = Some(TeaActive::One(branch));
    if decision.compute {
        Ok(Some(false))
    } else {
        runtime.state.note_reused(branch);
        super::log::debug(format_args!(
            "wan sol teacache reuse step {step} reason {}",
            decision.reason
        ));
        Ok(Some(true))
    }
}

fn decide_cfg_batch(
    runtime: &mut SolTeaRuntime,
    signal: &CudaTensor,
    step: usize,
) -> Result<Option<bool>> {
    if signal.shape.first().copied().unwrap_or(0) != 2 {
        return Err(TensorError::Message(
            "wan sol teacache batch forward wants uncond at row 0 and cond at row 1".into(),
        ));
    }
    let mut compute = false;
    for (row, branch) in [(0, TeaBranch::Uncond), (1, TeaBranch::Cond)] {
        let row_signal = signal.narrow(0, row, 1)?;
        let rel = if runtime.state.needs_signal(branch, step) {
            let previous = match branch {
                TeaBranch::Cond => runtime.cond_signal.as_ref(),
                TeaBranch::Uncond => runtime.uncond_signal.as_ref(),
            }
            .expect("teacache signal");
            relative_l1(&row_signal, previous)?
        } else {
            0.0
        };
        let decision = runtime.state.decide(branch, step, rel);
        match branch {
            TeaBranch::Cond => runtime.cond_signal = Some(row_signal),
            TeaBranch::Uncond => runtime.uncond_signal = Some(row_signal),
        }
        compute |= decision.compute;
    }
    runtime.active = Some(TeaActive::Batch);
    if compute {
        Ok(Some(false))
    } else {
        runtime.state.note_reused(TeaBranch::Uncond);
        runtime.state.note_reused(TeaBranch::Cond);
        super::log::debug(format_args!(
            "wan sol teacache reuse step {step} both cfg rows"
        ));
        Ok(Some(true))
    }
}

fn update_factors(
    previous: &[CudaTensor],
    features: &CudaTensor,
    delta: isize,
    max_order: usize,
) -> Result<Vec<CudaTensor>> {
    let mut factors = vec![features.clone()];
    if previous.is_empty() {
        return Ok(factors);
    }
    let inv = 1.0 / delta as f32;
    for j in 0..max_order {
        let Some(prev) = previous.get(j) else {
            break;
        };
        factors.push(factors[j].sub(prev)?.mul_scalar(inv));
    }
    Ok(factors)
}

fn predict_factors(factors: &[CudaTensor], step_offset: isize) -> Result<CudaTensor> {
    let base = factors.first().ok_or_else(|| {
        TensorError::Message("wan taylorseer forecast before the first proj_out".into())
    })?;
    let mut output = base.mul_scalar(1.0);
    let mut pow = 1.0f64;
    let mut fact = 1.0f64;
    for (order, factor) in factors.iter().enumerate().skip(1) {
        pow *= step_offset as f64;
        fact *= order as f64;
        output = output.add(&factor.mul_scalar((pow / fact) as f32))?;
    }
    Ok(output)
}

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

fn env_f64(name: &str, default: f64) -> f64 {
    std::env::var(name)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

impl WanTransformer3D {
    pub fn zeros(cfg: WanVideoArchConfig) -> Self {
        Self::try_zeros(cfg).expect("zero transformer")
    }

    fn try_zeros(cfg: WanVideoArchConfig) -> Result<Self> {
        let dim = cfg.hidden_size();
        let p = cfg.patch_size;
        let blocks = (0..cfg.num_layers)
            .map(|_| WanBlock::zeros(&cfg))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            patch_weight: pinned(CudaTensor::zeros(&[dim, cfg.in_channels, p[0], p[1], p[2]]))?,
            patch_bias: pinned(CudaTensor::zeros(&[dim]))?,
            time_embedder: MlpEmbed::zeros(cfg.freq_dim, dim),
            time_proj: Linear::zeros(dim, dim * 6, true),
            text_embedder: MlpEmbed::zeros(cfg.text_dim, dim),
            image_embedder: None,
            freq_dim: cfg.freq_dim,
            blocks,
            proj_out: Linear::zeros(dim, cfg.out_channels * p.iter().product::<usize>(), true),
            scale_shift_table: pinned(CudaTensor::zeros(&[1, 2, dim]))?,
            rotary_cache: Default::default(),
            #[cfg(feature = "cuda")]
            vsa_cache: Default::default(),
            sol_tea: Default::default(),
            sol_taylor: Default::default(),
            sol_a14b: Default::default(),
            attn_step: Default::default(),
            attn_profile: Default::default(),
            cfg,
        })
    }

    pub fn load(cfg: WanVideoArchConfig, map: &WeightMap) -> Result<Self> {
        Self::from_map(cfg, map)
    }

    pub fn from_map(cfg: WanVideoArchConfig, map: &WeightMap) -> Result<Self> {
        let dim = cfg.hidden_size();
        let p = cfg.patch_size;
        let blocks = (0..cfg.num_layers)
            .map(|i| WanBlock::load(map, &format!("blocks.{i}"), &cfg))
            .collect::<Result<Vec<_>>>()?;
        let image_embedder = match (cfg.image_dim, cfg.added_kv_proj_dim) {
            (Some(in_dim), Some(out_dim)) => Some(ImageEmbedder::load(
                map,
                "condition_embedder.image_embedder",
                in_dim,
                out_dim,
            )?),
            _ => None,
        };
        Ok(Self {
            patch_weight: pinned(weights::cuda_tensor_shaped(
                map,
                "patch_embedding.weight",
                &[dim, cfg.in_channels, p[0], p[1], p[2]],
            )?)?,
            patch_bias: pinned(weights::cuda_tensor_shaped(
                map,
                "patch_embedding.bias",
                &[dim],
            )?)?,
            time_embedder: MlpEmbed::load(
                map,
                "condition_embedder.time_embedder",
                cfg.freq_dim,
                dim,
            )?,
            time_proj: Linear::load(map, "condition_embedder.time_proj", dim, dim * 6, true)?,
            text_embedder: MlpEmbed::load(
                map,
                "condition_embedder.text_embedder",
                cfg.text_dim,
                dim,
            )?,
            image_embedder,
            proj_out: Linear::load(
                map,
                "proj_out",
                dim,
                cfg.out_channels * p.iter().product::<usize>(),
                true,
            )?,
            scale_shift_table: pinned(weights::cuda_tensor_shaped(
                map,
                "scale_shift_table",
                &[1, 2, dim],
            )?)?,
            freq_dim: cfg.freq_dim,
            blocks,
            rotary_cache: Default::default(),
            #[cfg(feature = "cuda")]
            vsa_cache: Default::default(),
            sol_tea: Default::default(),
            sol_taylor: Default::default(),
            sol_a14b: Default::default(),
            attn_step: Default::default(),
            attn_profile: Default::default(),
            cfg,
        })
    }

    /// `[B, C, T, H, W]` → `[B, seq, dim]` via a stride-`p` conv per frame.
    fn patch_embed(&self, xs: &CudaTensor) -> Result<CudaTensor> {
        let [b, c, t, h, w] = xs.shape[..] else {
            return Err(TensorError::Message(format!(
                "patch_embed expects BCTHW, got {:?}",
                xs.shape
            )));
        };
        let p = self.cfg.patch_size;
        let dim = self.cfg.hidden_size();
        if p[0] != 1 {
            return Err(TensorError::Message(
                "patch_embed supports temporal patch size 1 (Wan)".into(),
            ));
        }
        let x = xs
            .permute(&[0, 2, 1, 3, 4])?
            .reshape(vec![b * t, c, h, w])?;
        let k = self.patch_weight.reshape(vec![dim, c, p[1], p[2]])?;
        let y = x.conv2d(&k, Some(&self.patch_bias), 0, p[1])?;
        let (hh, ww) = (y.shape[2], y.shape[3]);
        y.reshape(vec![b, t, dim, hh, ww])?
            .permute(&[0, 1, 3, 4, 2])?
            .reshape(vec![b, t * hh * ww, dim])
    }

    /// Install or clear Sol TeaCache for this denoise. `teacache` reads the
    /// published defaults (threshold 0.12, warmup 2, cooldown the last 2).
    pub fn configure_sol_teacache(&self, num_steps: usize) -> Result<()> {
        let family = std::env::var("FASTVIDEO_WAN_SOL_CACHE").unwrap_or_default();
        let family = family.trim().to_ascii_lowercase();
        let mut slot = self.sol_tea.lock().expect("sol tea");
        if family != "teacache" {
            *slot = None;
            return Ok(());
        }
        let coefficients =
            std::env::var("FASTVIDEO_WAN_TEACACHE_COEFFS").unwrap_or_else(|_| "1.0,0.0".into());
        let coefficients: Vec<f64> = coefficients
            .split(',')
            .filter_map(|part| part.trim().parse().ok())
            .collect();
        let state = SolTeaCache::new(
            env_f64("FASTVIDEO_WAN_TEACACHE_THRESH", 0.12),
            env_usize("FASTVIDEO_WAN_TEACACHE_START", 2),
            env_usize("FASTVIDEO_WAN_TEACACHE_END", num_steps.saturating_sub(2)),
            env_usize("FASTVIDEO_WAN_TEACACHE_MAX_HITS", 0),
            env_usize("FASTVIDEO_WAN_TEACACHE_PERIODIC", 0),
            coefficients,
        )
        .map_err(TensorError::Message)?;
        super::log::info(format_args!(
            "wan sol teacache: threshold {} start {} end {} (block residual, dense head)",
            state.threshold, state.start_step, state.end_step
        ));
        *slot = Some(SolTeaRuntime {
            state,
            pending: None,
            active: None,
            cond_signal: None,
            uncond_signal: None,
            cond_residual: None,
            uncond_residual: None,
            batch_residual: None,
        });
        Ok(())
    }

    pub fn sol_teacache_enabled(&self) -> bool {
        self.sol_tea.lock().expect("sol tea").is_some()
    }

    /// Mark the next forward as one CFG branch of `step`. No-op when TeaCache is off.
    pub fn arm_sol_teacache(&self, cond: bool, step: usize) {
        let mut slot = self.sol_tea.lock().expect("sol tea");
        if let Some(runtime) = slot.as_mut() {
            runtime.pending = Some(TeaArm::One(
                if cond {
                    TeaBranch::Cond
                } else {
                    TeaBranch::Uncond
                },
                step,
            ));
        }
    }

    /// Next forward holds uncond in row 0 and cond in row 1.
    pub fn arm_sol_teacache_batch(&self, step: usize) {
        let mut slot = self.sol_tea.lock().expect("sol tea");
        if let Some(runtime) = slot.as_mut() {
            runtime.pending = Some(TeaArm::Batch(step));
        }
    }

    /// `Some(true)` reuses the block residual. `Some(false)` runs the blocks.
    /// `None` leaves the forward dense and does not touch the cache.
    fn begin_sol_tea(&self, signal: &CudaTensor) -> Result<Option<bool>> {
        let mut slot = self.sol_tea.lock().expect("sol tea");
        let Some(runtime) = slot.as_mut() else {
            return Ok(None);
        };
        let Some(arm) = runtime.pending.take() else {
            return Ok(None);
        };
        match arm {
            TeaArm::One(branch, step) => decide_one_branch(runtime, signal, branch, step),
            TeaArm::Batch(step) => decide_cfg_batch(runtime, signal, step),
        }
    }

    fn add_sol_tea_residual(&self, hidden: CudaTensor) -> Result<CudaTensor> {
        let residual = {
            let mut slot = self.sol_tea.lock().expect("sol tea");
            let runtime = slot.as_mut().expect("sol tea");
            match runtime.active.take().expect("sol tea branch") {
                TeaActive::One(TeaBranch::Cond) => runtime.cond_residual.clone(),
                TeaActive::One(TeaBranch::Uncond) => runtime.uncond_residual.clone(),
                TeaActive::Batch => runtime.batch_residual.clone(),
            }
            .expect("sol tea residual")
        };
        hidden.add(&residual)
    }

    fn finish_sol_tea(&self, before: &CudaTensor, after: &CudaTensor) -> Result<()> {
        let mut slot = self.sol_tea.lock().expect("sol tea");
        let runtime = slot.as_mut().expect("sol tea");
        let residual = after.sub(before)?;
        match runtime.active.take().expect("sol tea branch") {
            TeaActive::One(branch) => {
                runtime.state.note_computed(branch);
                match branch {
                    TeaBranch::Cond => runtime.cond_residual = Some(residual),
                    TeaBranch::Uncond => runtime.uncond_residual = Some(residual),
                }
            }
            TeaActive::Batch => {
                runtime.state.note_computed(TeaBranch::Uncond);
                runtime.state.note_computed(TeaBranch::Cond);
                runtime.batch_residual = Some(residual);
            }
        }
        Ok(())
    }

    /// Install or clear TaylorSeer lite. Off unless the cache family is `taylorseer`.
    pub fn configure_sol_taylor(&self, num_steps: usize) -> Result<()> {
        let family = std::env::var("FASTVIDEO_WAN_SOL_CACHE").unwrap_or_default();
        let family = family.trim().to_ascii_lowercase();
        let mut slot = self.sol_taylor.lock().expect("sol taylor");
        if family != "taylorseer" {
            *slot = None;
            return Ok(());
        }
        let lite = std::env::var("FASTVIDEO_WAN_TAYLOR_LITE")
            .or_else(|_| std::env::var("WAN22_TAYLOR_LITE"))
            .unwrap_or_else(|_| "1".into());
        if matches!(lite.trim(), "0" | "false" | "off") {
            return Err(TensorError::Message(
                "wan taylorseer lite is the published mode (WAN22_TAYLOR_LITE); full per-block factors are not this path".into(),
            ));
        }
        let schedule = TaylorSchedule::new(
            env_usize_alt("FASTVIDEO_WAN_TAYLOR_INTERVAL", "WAN22_TAYLOR_INTERVAL", 3),
            env_usize_alt("FASTVIDEO_WAN_TAYLOR_WARMUP", "WAN22_TAYLOR_WARMUP", 3),
            env_usize_alt(
                "FASTVIDEO_WAN_TAYLOR_COOLDOWN",
                "WAN22_TAYLOR_COOLDOWN_START",
                num_steps.saturating_sub(2),
            ),
            env_usize_alt("FASTVIDEO_WAN_TAYLOR_ORDER", "WAN22_TAYLOR_ORDER", 1),
        )
        .map_err(TensorError::Message)?;
        super::log::info(format_args!(
            "wan sol taylorseer: interval {} warmup {} cooldown {} order {} (proj_out forecast, blocks skipped)",
            schedule.interval, schedule.warmup, schedule.cooldown_start, schedule.max_order
        ));
        *slot = Some(SolTaylorRuntime {
            schedule,
            compute: true,
            last_update: None,
            factors: Vec::new(),
        });
        Ok(())
    }

    pub fn sol_taylor_enabled(&self) -> bool {
        self.sol_taylor.lock().expect("sol taylor").is_some()
    }

    /// A14B EasyCache only. 5B / 14B keep the whole-stack controller.
    pub fn configure_a14b_cache(&self, num_steps: usize) -> Result<()> {
        let family = std::env::var("FASTVIDEO_WAN_SOL_CACHE").unwrap_or_default();
        let family = family.trim().to_ascii_lowercase();
        let mut slot = self.sol_a14b.lock().expect("sol a14b");
        if family != "easycache" || !self.cfg.is_moe() {
            *slot = None;
            return Ok(());
        }
        if self.blocks.len() < 2 {
            return Err(TensorError::Message(
                "wan a14b cache needs at least two transformer blocks".into(),
            ));
        }
        let state = A14bCacheController::official(num_steps).map_err(TensorError::Message)?;
        super::log::info(format_args!(
            "wan a14b easycache: threshold {} start {} tail {} max_reuse {} (block-0 fresh, blocks 1-{} residual)",
            state.threshold, state.start_step, state.tail_steps, state.max_reuse,
            self.blocks.len().saturating_sub(1)
        ));
        *slot = Some(A14bRuntime {
            state,
            pending: None,
            active_cond: true,
            reuse: false,
            previous_input: None,
            last_compute_input: None,
            last_compute_output: None,
            cond_residual: None,
            uncond_residual: None,
        });
        Ok(())
    }

    pub fn sol_a14b_enabled(&self) -> bool {
        self.sol_a14b.lock().expect("sol a14b").is_some()
    }

    pub fn arm_a14b_cache(&self, cond: bool, step: usize) {
        let mut slot = self.sol_a14b.lock().expect("sol a14b");
        if let Some(runtime) = slot.as_mut() {
            runtime.pending = Some((cond, step));
        }
    }

    pub fn configure_attn_route(&self) {
        let sol = sol_attn_requested(
            std::env::var("FASTVIDEO_WAN_SOL_ATTN")
                .ok()
                .or_else(|| std::env::var("WAN22_SOL_ATTN").ok())
                .as_deref(),
        );
        let pisa = pisa_requested(
            std::env::var("FASTVIDEO_WAN_PISA")
                .ok()
                .or_else(|| std::env::var("WAN22_PISA").ok())
                .as_deref(),
        );
        let profile = if self.cfg.is_moe() && (pisa || sol) {
            WanAttnProfile::PisaA14b
        } else if self.cfg.num_layers == 30 && self.cfg.in_channels == 48 && pisa {
            WanAttnProfile::Pisa5b
        } else if self.cfg.num_layers >= 40 && !self.cfg.is_moe() && sol {
            WanAttnProfile::Sol14b
        } else {
            WanAttnProfile::Off
        };
        if profile != WanAttnProfile::Off {
            super::log::info(format_args!("wan attn route: {profile:?}"));
        }
        *self.attn_profile.lock().expect("attn profile") = profile;
    }

    pub fn sol_attn_enabled(&self) -> bool {
        *self.attn_profile.lock().expect("attn profile") != WanAttnProfile::Off
    }

    pub fn arm_attn_step(&self, step: usize) {
        *self.attn_step.lock().expect("attn step") = Some(step);
    }

    fn attn_plan(&self, layer: usize, morton_grid: Option<(usize, usize, usize)>) -> AttnPlan {
        AttnPlan {
            step: self.attn_step.lock().expect("attn step").unwrap_or(0),
            layer,
            profile: *self.attn_profile.lock().expect("attn profile"),
            morton_grid,
        }
    }

    fn begin_a14b_tail(&self, prefix: &CudaTensor) -> Result<Option<bool>> {
        let mut slot = self.sol_a14b.lock().expect("sol a14b");
        let Some(runtime) = slot.as_mut() else {
            return Ok(None);
        };
        let Some((cond, step)) = runtime.pending.take() else {
            return Ok(None);
        };
        if cond {
            let change = if runtime
                .state
                .needs_input_signal(A14bExpert::HighNoise, step)
            {
                mean_abs_delta(prefix, runtime.previous_input.as_ref().expect("a14b prev"))?
            } else {
                0.0
            };
            runtime.previous_input = Some(prefix.clone());
            let decision = runtime.state.decide(A14bExpert::HighNoise, step, change);
            runtime.reuse = !decision.compute;
        }
        runtime.active_cond = cond;
        let residual = if cond {
            runtime.cond_residual.as_ref()
        } else {
            runtime.uncond_residual.as_ref()
        };
        if runtime.reuse {
            if residual.is_none() {
                runtime.reuse = false;
                return Ok(Some(false));
            }
            if cond {
                runtime.state.note_reused(A14bExpert::HighNoise);
            }
            Ok(Some(true))
        } else {
            Ok(Some(false))
        }
    }

    fn add_a14b_residual(&self, prefix: CudaTensor) -> Result<CudaTensor> {
        let residual = {
            let slot = self.sol_a14b.lock().expect("sol a14b");
            let runtime = slot.as_ref().expect("sol a14b");
            if runtime.active_cond {
                runtime.cond_residual.clone()
            } else {
                runtime.uncond_residual.clone()
            }
            .expect("a14b residual")
        };
        prefix.add(&residual)
    }

    fn finish_a14b_tail(&self, prefix: &CudaTensor, hidden: &CudaTensor) -> Result<()> {
        let mut slot = self.sol_a14b.lock().expect("sol a14b");
        let runtime = slot.as_mut().expect("sol a14b");
        let residual = hidden.sub(prefix)?;
        if runtime.active_cond {
            let (full_in, out_change) =
                match (&runtime.last_compute_input, &runtime.last_compute_output) {
                    (Some(prev_in), Some(prev_out)) => (
                        mean_abs_delta(prefix, prev_in)?,
                        mean_abs_delta(&residual, prev_out)?,
                    ),
                    _ => (0.0, 0.0),
                };
            let norm = mean_abs(&residual)?;
            runtime
                .state
                .note_computed(A14bExpert::HighNoise, full_in, out_change, norm);
            runtime.last_compute_input = Some(prefix.clone());
            runtime.last_compute_output = Some(residual.clone());
            runtime.cond_residual = Some(residual);
        } else {
            runtime.uncond_residual = Some(residual);
        }
        Ok(())
    }

    /// `Forecast` skips the blocks and the head. `Compute` runs them and
    /// refreshes the `proj_out` factors afterwards.
    fn begin_taylor(&self) -> Result<TaylorPhase> {
        let mut slot = self.sol_taylor.lock().expect("sol taylor");
        let Some(runtime) = slot.as_mut() else {
            return Ok(TaylorPhase::Off);
        };
        let decision = runtime.schedule.begin_forward();
        runtime.compute = decision.compute;
        if decision.compute {
            return Ok(TaylorPhase::Compute);
        }
        if runtime.factors.is_empty() || runtime.last_update.is_none() {
            return Err(TensorError::Message(
                "wan taylorseer forecast before the first proj_out".into(),
            ));
        }
        super::log::debug(format_args!(
            "wan sol taylorseer forecast step {}",
            decision.step
        ));
        Ok(TaylorPhase::Forecast)
    }

    fn predict_taylor(&self) -> Result<CudaTensor> {
        let slot = self.sol_taylor.lock().expect("sol taylor");
        let runtime = slot.as_ref().expect("sol taylor");
        let last = runtime.last_update.expect("taylor last update");
        let offset = runtime.schedule.current_step() - last;
        predict_factors(&runtime.factors, offset)
    }

    fn update_taylor(&self, features: &CudaTensor) -> Result<()> {
        let mut slot = self.sol_taylor.lock().expect("sol taylor");
        let runtime = slot.as_mut().expect("sol taylor");
        if !runtime.compute {
            return Ok(());
        }
        let current = runtime.schedule.current_step();
        let delta = runtime.last_update.map(|last| current - last).unwrap_or(1);
        if runtime.last_update.is_some() && delta == 0 {
            return Err(TensorError::Message(
                "wan taylorseer delta step cannot be zero".into(),
            ));
        }
        runtime.factors = update_factors(
            &runtime.factors,
            features,
            delta,
            runtime.schedule.max_order,
        )?;
        runtime.last_update = Some(current);
        Ok(())
    }

    /// Causal self-attention cache for `FASTVIDEO_NVFP4`. `None` keeps the
    /// dense causal mask.
    fn causal_ar_frame(&self, t: usize, h: usize, w: usize) -> Option<super::ar_cache::ArFrame> {
        let rule = fastvideo_models::nvfp4::from_env()?;
        if !self.cfg.causal {
            return None;
        }
        let p = self.cfg.patch_size;
        if p[0] == 0 || p[1] == 0 || p[2] == 0 {
            return None;
        }
        let frames = t / p[0];
        let spatial = (h / p[1]) * (w / p[2]);
        if frames == 0
            || spatial == 0
            || !self
                .cfg
                .attention_head_dim
                .is_multiple_of(fastvideo_models::nvfp4::BLOCK)
        {
            return None;
        }
        let sink = self.cfg.sink_size.saturating_mul(spatial);
        let seq = frames * spatial;
        let (capacity, max_attention) = if self.cfg.local_attn_size > 0 {
            let window = self.cfg.local_attn_size as usize * spatial;
            (sink + window.max(spatial), window)
        } else {
            (seq, 0)
        };
        static SAID: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        super::log::info_once(
            &SAID,
            format_args!(
                "wan ar kv cache: frames {frames} spatial {spatial} capacity {capacity} sink {sink} window {max_attention}"
            ),
        );
        Some(super::ar_cache::ArFrame {
            heads: self.cfg.num_attention_heads,
            dim: self.cfg.attention_head_dim,
            capacity,
            sink_tokens: sink,
            max_attention,
            frame_seqlen: spatial,
            rule,
        })
    }

    pub fn forward(
        &self,
        latents: &CudaTensor,
        timestep: &CudaTensor,
        encoder: &CudaTensor,
    ) -> Result<CudaTensor> {
        self.forward_ctx(latents, timestep, encoder, None)
    }

    /// Get-or-build the VSA context for a latent grid, or `None` when VSA is
    /// off, unavailable, or the checkpoint has no gates.
    #[cfg(feature = "cuda")]
    fn vsa_for(&self, t: usize, h: usize, w: usize) -> Result<Option<std::sync::Arc<VsaCtx>>> {
        if !super::nn::vsa_enabled() {
            return Ok(None);
        }
        // No live device means a CPU run, where VSA has no device path: fall
        // back to dense rather than failing to upload a tiling. Say so, rather
        // than silently running dense while the caller believes VSA is on.
        if super::device::global_device().is_none() || !self.blocks.iter().any(|b| b.gate.is_some())
        {
            static ONCE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
            super::log::info_once(
                &ONCE,
                format_args!("vsa: requested but unavailable (no device, or checkpoint has no to_gate_compress); using dense attention"),
            );
            return Ok(None);
        }
        let p = self.cfg.patch_size;
        let grid = (t / p[0], h / p[1], w / p[2]);
        let mut map = self.vsa_cache.lock().expect("vsa cache lock");
        if let Some(ctx) = map.get(&grid) {
            return Ok(Some(ctx.clone()));
        }
        let sparsity = super::envflag::f64_flag("FASTVIDEO_VSA_SPARSITY", 0.8);
        let group = super::envflag::usize_flag("FASTVIDEO_VSA_GROUP", 32);
        let ctx = std::sync::Arc::new(VsaCtx::new(grid, sparsity, group)?);
        super::log::info(format_args!(
            "vsa: grid {grid:?} tiles {} topk {} group {group}",
            ctx.plan.num_tiles, ctx.topk
        ));
        map.insert(grid, ctx.clone());
        Ok(Some(ctx))
    }

    #[cfg(not(feature = "cuda"))]
    fn vsa_for(&self, _t: usize, _h: usize, _w: usize) -> Result<Option<std::sync::Arc<VsaCtx>>> {
        Ok(None)
    }

    /// Get-or-build the `[seq, head_dim]` RoPE tables for a latent grid.
    pub fn rotary_for(&self, t: usize, h: usize, w: usize) -> Result<(CudaTensor, CudaTensor)> {
        let p = self.cfg.patch_size;
        let seq = (t / p[0]) * (h / p[1]) * (w / p[2]);
        let key = (seq, self.cfg.attention_head_dim);
        let mut map = self.rotary_cache.lock().expect("rotary cache lock");
        if let Some(pair) = map.get(&key) {
            return Ok(pair.clone());
        }
        let (cos, sin) = wan_rope(&self.cfg, t, h, w)?;
        let pair = (pinned(cos)?, pinned(sin)?);
        map.insert(key, pair.clone());
        Ok(pair)
    }

    pub fn forward_ctx(
        &self,
        latents: &CudaTensor,
        timestep: &CudaTensor,
        encoder: &CudaTensor,
        image: Option<&CudaTensor>,
    ) -> Result<CudaTensor> {
        let [b, _c, t, h, w] = latents.shape[..] else {
            return Err(TensorError::Message(format!(
                "forward expects BCTHW latents, got {:?}",
                latents.shape
            )));
        };
        let taylor = self.begin_taylor()?;
        if taylor == TaylorPhase::Forecast {
            return self.unpatchify(self.predict_taylor()?, b, t, h, w);
        }
        let dim = self.cfg.hidden_size();
        let rope = self.rotary_for(t, h, w)?;
        let ar = self.causal_ar_frame(t, h, w);
        let mask = if self.cfg.causal && ar.is_none() {
            let mask = fastvideo_models::wan::causal_temporal_mask(&self.cfg, t, h, w);
            let seq = (mask.len() as f64).sqrt() as usize;
            Some(CudaTensor::from_vec(mask, vec![1, 1, seq, seq])?.to_device()?)
        } else {
            None
        };
        let mut hidden = self.patch_embed(latents)?;
        let temb = self
            .time_embedder
            .forward_silu(&nn::sinusoidal_timesteps(timestep, self.freq_dim)?)?;
        let timestep_proj = self
            .time_proj
            .forward(&temb.silu())?
            .reshape(vec![b, 6, dim])?;
        let encoder = self.text_embedder.forward_gelu(encoder)?;
        let image = match (image, &self.image_embedder) {
            (Some(img), Some(emb)) => Some(emb.forward(img)?),
            (Some(img), None) => Some(img.clone()),
            _ => None,
        };
        // VSA applies to the self-attention grid only, and only when the
        // checkpoint carries the gates it was trained with.
        let vsa = self.vsa_for(t, h, w)?;
        let p = self.cfg.patch_size;
        let morton_grid = Some((t / p[0], h / p[1], w / p[2]));
        let tea = self.begin_sol_tea(&timestep_proj)?;
        let block_in = if tea == Some(false) {
            Some(hidden.clone())
        } else {
            None
        };
        if tea == Some(true) {
            hidden = self.add_sol_tea_residual(hidden)?;
        } else if self.sol_a14b_enabled() {
            hidden = self.blocks[0].forward(
                &hidden,
                &encoder,
                &timestep_proj,
                &rope,
                image.as_ref(),
                mask.as_ref(),
                vsa.as_deref(),
                ar.as_ref(),
                self.attn_plan(0, morton_grid),
            )?;
            match self.begin_a14b_tail(&hidden)? {
                Some(true) => hidden = self.add_a14b_residual(hidden)?,
                reuse => {
                    let prefix = hidden.clone();
                    for (layer, block) in self.blocks.iter().enumerate().skip(1) {
                        hidden = block.forward(
                            &hidden,
                            &encoder,
                            &timestep_proj,
                            &rope,
                            image.as_ref(),
                            mask.as_ref(),
                            vsa.as_deref(),
                            ar.as_ref(),
                            self.attn_plan(layer, morton_grid),
                        )?;
                    }
                    if reuse == Some(false) {
                        self.finish_a14b_tail(&prefix, &hidden)?;
                    }
                }
            }
        } else {
            for (layer, block) in self.blocks.iter().enumerate() {
                hidden = block.forward(
                    &hidden,
                    &encoder,
                    &timestep_proj,
                    &rope,
                    image.as_ref(),
                    mask.as_ref(),
                    vsa.as_deref(),
                    ar.as_ref(),
                    self.attn_plan(layer, morton_grid),
                )?;
            }
            if let Some(before) = block_in.as_ref() {
                self.finish_sol_tea(before, &hidden)?;
            }
        }
        // Output head: table [1, 2, dim] + temb broadcast over both rows.
        let temb_rows = temb.unsqueeze(1)?;
        let e = CudaTensor::cat(&[&temb_rows, &temb_rows], 1)?.add(&self.scale_shift_table)?;
        hidden = hidden.ln_adaln_e(&e, 1, 0, self.cfg.eps)?;
        hidden = self.proj_out.forward(&hidden)?;
        if taylor == TaylorPhase::Compute {
            self.update_taylor(&hidden)?;
        }
        self.unpatchify(hidden, b, t, h, w)
    }

    /// `[B, seq, oc*pt*ph*pw]` → `[B, oc, T, H, W]` (temporal patch size 1).
    fn unpatchify(
        &self,
        hidden: CudaTensor,
        b: usize,
        t: usize,
        h: usize,
        w: usize,
    ) -> Result<CudaTensor> {
        let p = self.cfg.patch_size;
        let oc = self.cfg.out_channels;
        let (ppf, pph, ppw) = (t / p[0], h / p[1], w / p[2]);
        if p[0] != 1 {
            return Err(TensorError::Message(
                "unpatchify supports temporal patch size 1 (Wan)".into(),
            ));
        }
        // [b*ppf, pph, ppw, ph, pw, oc] → [b*ppf, oc, pph, ph, ppw, pw]
        let x = hidden
            .reshape(vec![b * ppf, pph, ppw, p[1], p[2], oc])?
            .permute(&[0, 5, 1, 3, 2, 4])?
            .reshape(vec![b, ppf, oc, pph * p[1], ppw * p[2]])?;
        x.permute(&[0, 2, 1, 3, 4])
    }
}
