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

use super::attn::BlockCausal;
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

    /// q/k RMSNorm (+ RoPE) into BHSD: FastVideo's bf16 rounding points on
    /// the bf16 path ([`super::fuse::qk_norm_rope`]), the f32 op otherwise.
    fn qk(
        &self,
        proj: &CudaTensor,
        col_off: usize,
        weight: &CudaTensor,
        rope: Option<Rope<'_>>,
    ) -> Result<CudaTensor> {
        let again = rope.as_ref().map(|r| Rope {
            cos: r.cos,
            sin: r.sin,
        });
        match super::fuse::qk_norm_rope(proj, col_off, self.heads, weight, again, self.eps)? {
            Some(t) => Ok(t),
            None => proj.qk_norm_rope_bhsd(col_off, self.heads, weight, rope, self.eps),
        }
    }

    fn attend(
        &self,
        q: &CudaTensor,
        k: &CudaTensor,
        v: &CudaTensor,
        mask: Option<&BlockCausal>,
    ) -> Result<CudaTensor> {
        let attn = match mask {
            Some(m) => nn::sdpa_block_causal(q, k, v, None, *m)?,
            None => nn::scaled_dot_product_attention_masked(q, k, v, None, None)?,
        };
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
                let grid = plan
                    .morton_grid
                    .filter(|&(f, h, w)| f.saturating_mul(h).saturating_mul(w) == q.shape[2])
                    .filter(|_| morton3d_on_route(plan.profile, plan.step, plan.layer));
                use super::sol_cache::morton_gather;
                let (q, k, v) = match grid {
                    Some(g) => (
                        morton_gather(q, g, false)?,
                        morton_gather(k, g, false)?,
                        morton_gather(v, g, false)?,
                    ),
                    None => (q.clone(), k.clone(), v.clone()),
                };
                let mut out = crate::sol_attn::sol_attn(&q, &k, &v, tau, scale, None, 0)?;
                if let Some(g) = grid {
                    out = morton_gather(&out, g, true)?;
                }
                Ok(Some(out))
            }
            WanAttnRoute::Pisa { sparsity } => {
                Ok(Some(crate::pisa_attn::pisa_attn(q, k, v, sparsity, scale)?))
            }
        }
    }

    /// FastVideo `rope_cache_policy = "relativistic"`
    /// (`CausalWanSelfAttention.forward`, `relativistic_window_offsets`): the
    /// normed, un-roped K goes into the cache; the window is roped at
    /// positions `[0, window)` and the queries at its tail. `table` is the
    /// window's table (`[window, d]`, see `WanTransformer3D::forward_kv`).
    fn forward_self_relativistic(
        &self,
        qkv: &CudaTensor,
        table: &(CudaTensor, CudaTensor),
        at: super::causal::KvAt<'_>,
    ) -> Result<CudaTensor> {
        let dim = self.heads * self.dim_head;
        let q = self.qk(qkv, 0, &self.norm_q, None)?;
        let k = self.qk(qkv, dim, &self.norm_k, None)?;
        let v = qkv.split_heads_bhsd(2 * dim, self.heads, self.dim_head)?;
        let n = q.shape[2];
        let (kw, vw) = at.cache.update(at.layer, &k, &v, at.current_start)?;
        let window = kw.shape[2];
        if window < n || table.0.shape[0] != window {
            return Err(TensorError::Message(format!(
                "relativistic rope: window {window} tokens, {n} queries, table {:?}",
                table.0.shape
            )));
        }
        let q = super::causal::rope_bhsd(
            &q,
            &table.0.narrow(0, window - n, n)?,
            &table.1.narrow(0, window - n, n)?,
        )?;
        let kw = super::causal::rope_bhsd(&kw, &table.0, &table.1)?;
        let out = nn::sdpa_kv_window(&q, &kw, &vw)?.merge_heads()?;
        self.to_out.forward(&out)
    }

    fn forward_self(
        &self,
        hidden: &CudaTensor,
        rope: &(CudaTensor, CudaTensor),
        mask: Option<&BlockCausal>,
        gate: Option<&Linear>,
        vsa: Option<&VsaCtx>,
        ar: Option<&super::ar_cache::ArFrame>,
        kv: Option<super::causal::KvAt<'_>>,
        plan: AttnPlan,
    ) -> Result<CudaTensor> {
        let dim = self.heads * self.dim_head;
        let qkv = self.q_or_qkv.forward(hidden)?;
        if let Some(at) = kv.filter(|a| a.cache.spec.relativistic) {
            return self.forward_self_relativistic(&qkv, rope, at);
        }
        let rope = || {
            Some(Rope {
                cos: &rope.0,
                sin: &rope.1,
            })
        };
        let q = self.qk(&qkv, 0, &self.norm_q, rope())?;
        let k = self.qk(&qkv, dim, &self.norm_k, rope())?;
        let v = qkv.split_heads_bhsd(2 * dim, self.heads, self.dim_head)?;
        // Autoregressive causal Wan: this block's K/V into the cache, the
        // queries against the cache window, unmasked (super::causal).
        if let Some(at) = kv {
            let (kw, vw) = at.cache.update(at.layer, &k, &v, at.current_start)?;
            let out = nn::sdpa_kv_window(&q, &kw, &vw)?.merge_heads()?;
            if super::dump::op_block() == Some(at.layer) {
                let [b, h, sk, d] = kw.shape[..] else {
                    return Err(TensorError::Message("kv window rank".into()));
                };
                let rows = kw.permute(&[0, 2, 1, 3])?.reshape(vec![b * sk, h * d])?;
                let stride = super::dump::BLOCK_ROW_STRIDE;
                super::dump::rows_strided(
                    &super::dump::named(&format!("b{}_kwin", at.layer)),
                    &rows,
                    stride,
                )?;
                super::dump::rows_strided(
                    &super::dump::named(&format!("b{}_attn_x", at.layer)),
                    &out,
                    stride,
                )?;
            }
            return self.to_out.forward(&out);
        }
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

    /// Cross-attention K (RMS-normed) and V over the text tokens, BHSD. A
    /// function of the text alone: the same for every step of a denoise.
    fn cross_kv(&self, encoder: &CudaTensor) -> Result<(CudaTensor, CudaTensor)> {
        let dim = self.heads * self.dim_head;
        let kv_proj = self
            .kv
            .as_ref()
            .ok_or_else(|| TensorError::Message("cross attention without kv".into()))?;
        let kv = kv_proj.forward(encoder)?;
        let k = self.qk(&kv, 0, &self.norm_k, None)?;
        let v = kv.split_heads_bhsd(dim, self.heads, self.dim_head)?;
        Ok((k, v))
    }

    /// `cached`: this block's [`Self::cross_kv`] for `encoder`, computed once
    /// per denoise (see [`WanTransformer3D::begin_text_cache`]).
    fn forward_cross(
        &self,
        hidden: &CudaTensor,
        encoder: &CudaTensor,
        image: Option<&CudaTensor>,
        cached: Option<&(CudaTensor, CudaTensor)>,
    ) -> Result<CudaTensor> {
        let q = self.qk(&self.q_or_qkv.forward(hidden)?, 0, &self.norm_q, None)?;
        let (mut k, mut v) = match cached {
            Some((k, v)) => (k.clone(), v.clone()),
            None => self.cross_kv(encoder)?,
        };
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

/// [`rotary_1d`] with at least `seq` rows, memoized per `(dim, theta)`: row
/// `p` does not depend on the length, so a longer table serves every
/// shorter request (an open-ended stream asks for a new offset every block).
fn rotary_1d_cached(dim: usize, seq: usize, theta: f64) -> std::sync::Arc<(Vec<f32>, Vec<f32>)> {
    type Memo = std::collections::HashMap<(usize, u64), std::sync::Arc<(Vec<f32>, Vec<f32>)>>;
    static MEMO: std::sync::OnceLock<std::sync::Mutex<Memo>> = std::sync::OnceLock::new();
    let key = (dim, theta.to_bits());
    let mut memo = MEMO.get_or_init(Default::default).lock().expect("rope memo");
    if let Some(t) = memo.get(&key) {
        if t.0.len() >= seq * dim {
            return t.clone();
        }
    }
    let t = std::sync::Arc::new(rotary_1d(dim, seq.next_power_of_two(), theta));
    memo.insert(key, t.clone());
    t
}

/// 3-D RoPE tables `[seq, head_dim]` (time, height, width split of the head).
///
/// Latent frames `start_frame ..` (FastVideo `get_rotary_pos_embed(...,
/// start_frame=start_frame)`, the causal blocks; 0 otherwise).
fn wan_rope_at(
    cfg: &WanVideoArchConfig,
    frames: usize,
    height: usize,
    width: usize,
    start_frame: usize,
) -> Result<(CudaTensor, CudaTensor)> {
    let d = cfg.attention_head_dim;
    let h_dim = 2 * (d / 6);
    let w_dim = h_dim;
    let t_dim = d - h_dim - w_dim;
    let axes = [
        // Past the checkpoint's table (an open-ended stream, `wan::stream`) the
        // temporal rows continue with the same formula.
        (t_dim, rotary_1d_cached(t_dim, cfg.rope_max_seq_len.max(start_frame + frames), 10000.0)),
        (h_dim, rotary_1d_cached(h_dim, cfg.rope_max_seq_len, 10000.0)),
        (w_dim, rotary_1d_cached(w_dim, cfg.rope_max_seq_len, 10000.0)),
    ];
    let (ppf, pph, ppw) = (
        frames / cfg.patch_size[0],
        height / cfg.patch_size[1],
        width / cfg.patch_size[2],
    );
    let seq = ppf * pph * ppw;
    let mut cos = Vec::with_capacity(seq * d);
    let mut sin = Vec::with_capacity(seq * d);
    for ft in start_frame..start_frame + ppf {
        for fh in 0..pph {
            for fw in 0..ppw {
                for ((ad, t), pos) in axes.iter().zip([ft, fh, fw]) {
                    let (c, s) = &**t;
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

/// The step-invariant inputs of one block forward.
#[derive(Clone, Copy)]
struct BlockCond<'a> {
    /// `timestep_proj + scale_shift_table`, `[b, 6, dim]`.
    e: &'a CudaTensor,
    /// Cross-attention K/V over the text, when cached for this denoise.
    cross_kv: Option<&'a (CudaTensor, CudaTensor)>,
}

/// Text conditioning prepared once per denoise: the text embedder's output
/// and every block's cross-attention K/V. Holds the encoder tensor it was
/// computed from, so its storage identity cannot be reused while cached.
#[derive(Debug)]
struct PreparedText {
    encoder: CudaTensor,
    embedded: CudaTensor,
    kv: Vec<(CudaTensor, CudaTensor)>,
}

/// Which RoPE rows a causal block forward reads
/// ([`WanTransformer3D::kv_table_spec`]): `rows` rows of the table of
/// `frames` latent frames from `start_frame`, and for a rebased sink
/// (`sink`: target frame, sink frames) the sink's tables at 0 and at the
/// target.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KvTableSpec {
    pub frames: usize,
    pub start_frame: usize,
    pub rows: usize,
    pub sink: Option<(usize, usize)>,
}

/// The RoPE tables of one causal block forward: `(cos, sin)` of the rows
/// the block reads, and for a rebased sink `(target, at 0, at target)`.
#[derive(Debug, Clone)]
pub struct KvTables {
    pub rope: (CudaTensor, CudaTensor),
    pub sink: Option<(usize, (CudaTensor, CudaTensor), (CudaTensor, CudaTensor))>,
}

impl KvTables {
    /// Every table, in a fixed order (for copying one set into another).
    pub fn tensors(&self) -> Vec<&CudaTensor> {
        let mut v = vec![&self.rope.0, &self.rope.1];
        if let Some((_, a, b)) = &self.sink {
            v.extend([&a.0, &a.1, &b.0, &b.1]);
        }
        v
    }

    /// The same shapes: one set can be copied into the other.
    pub fn same_layout(&self, other: &KvTables) -> bool {
        let (a, b) = (self.tensors(), other.tensors());
        a.len() == b.len() && a.iter().zip(&b).all(|(x, y)| x.shape == y.shape)
    }
}

/// The time conditioning of one causal block forward (a timestep's
/// embedding and every block's modulation), held on the device.
#[derive(Debug, Clone)]
pub struct KvTime(std::sync::Arc<PreparedTime>);

/// The text conditioning of one causal block forward: the text embedding
/// and, when cached, every block's cross-attention K/V.
#[derive(Debug, Clone)]
pub struct KvText {
    embedded: CudaTensor,
    kv: Option<Vec<(CudaTensor, CudaTensor)>>,
}

impl KvText {
    fn tensors(&self) -> Vec<&CudaTensor> {
        let mut v = vec![&self.embedded];
        for (k, vv) in self.kv.iter().flatten() {
            v.extend([k, vv]);
        }
        v
    }

    /// A copy in buffers of its own: a slot a CUDA graph reads by address
    /// while [`Self::assign`] refills it for a new prompt.
    pub fn duplicate(&self) -> Result<KvText> {
        Ok(KvText {
            embedded: super::graph::duplicate(&self.embedded)?,
            kv: match &self.kv {
                Some(kv) => Some(
                    kv.iter()
                        .map(|(k, v)| Ok((super::graph::duplicate(k)?, super::graph::duplicate(v)?)))
                        .collect::<Result<Vec<_>>>()?,
                ),
                None => None,
            },
        })
    }

    /// Overwrite this slot's buffers with `src`'s values, in place.
    pub fn assign(&self, src: &KvText) -> Result<()> {
        let (d, s) = (self.tensors(), src.tensors());
        if d.len() != s.len() || d.iter().zip(&s).any(|(a, b)| a.shape != b.shape) {
            return Err(TensorError::Message(
                "KvText::assign: the conditioning layouts differ".into(),
            ));
        }
        for (d, s) in d.into_iter().zip(s) {
            super::graph::assign(d, s)?;
        }
        Ok(())
    }
}

/// Time conditioning for one timestep vector: the time embedding (the
/// output head's), `timestep_proj` (`[b, 6, dim]`) and every block's
/// modulation. A function of the timestep values and the weights only.
#[derive(Debug)]
struct PreparedTime {
    key: Vec<u32>,
    temb: CudaTensor,
    timestep_proj: CudaTensor,
    e: Vec<CudaTensor>,
}

/// Distinct timestep vectors kept: a 50-step UniPC run with a CFG batch is
/// 50; DMD / rCM runs are 3-4 per request and repeat across requests.
const TIME_CACHE_ENTRIES: usize = 64;
/// Encoder tensors kept per denoise (the CFG pair, its rows, a margin).
const TEXT_CACHE_ENTRIES: usize = 4;

/// `FASTVIDEO_WAN_COND_CACHE=0` recomputes the text K/V, text embedding and
/// time modulation in every forward (the byte-identical reference path).
fn cond_cache_enabled() -> bool {
    static FLAG: super::envflag::CachedBool = super::envflag::CachedBool::new();
    FLAG.get_or_init(|| super::envflag::bool_flag("FASTVIDEO_WAN_COND_CACHE", true))
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

    /// A reference FP8 recipe on the block's attention and FFN linears — the
    /// ones FastVideo's `fp8_config._FP8_SUFFIXES` tag for Wan: `to_q`,
    /// `to_k`, `to_v`, `to_out` of both attentions and `ffn.fc_in/fc_out`.
    /// W8A8 keeps one tensor scale per original linear (the fused QKV / KV
    /// stacks are sections); MXFP8 scales per 32 values and takes each stack
    /// whole. The I2V image K/V and the VSA gate stay bf16. A linear that
    /// already carries another recipe (`FASTVIDEO_FP8`, NVFP4, affine, LoRA)
    /// is left as it is.
    fn quantize(&mut self, kind: super::quant::QuantKind, cfg: &WanVideoArchConfig) -> Result<()> {
        use super::quant::{QuantKind, Section};
        let dim = cfg.hidden_size();
        let q = |rows| Section {
            rows,
            quantized: true,
        };
        let stack = |n: usize| match kind {
            QuantKind::W8A8 => (0..n).map(|_| q(dim)).collect(),
            QuantKind::Mxfp8 => vec![q(n * dim)],
        };
        let plain = |l: &Linear| {
            !(l.is_fp8_gemm() || l.is_nvfp4() || l.is_affine() || l.is_fp8_rows() || l.has_lora())
        };
        let apply = |l: &mut Linear, sections: Vec<Section>| -> Result<()> {
            if plain(l) {
                l.quantize(kind, sections)?;
            }
            Ok(())
        };
        apply(&mut self.attn1.q_or_qkv, stack(3))?;
        apply(&mut self.attn1.to_out, vec![q(dim)])?;
        apply(&mut self.attn2.q_or_qkv, vec![q(dim)])?;
        if let Some(kv) = self.attn2.kv.as_mut() {
            apply(kv, stack(2))?;
        }
        apply(&mut self.attn2.to_out, vec![q(dim)])?;
        apply(&mut self.ffn.proj, vec![q(cfg.ffn_dim)])?;
        apply(&mut self.ffn.out, vec![q(dim)])
    }

    /// `cond.e`: this block's modulation, `timestep_proj + scale_shift_table`
    /// ([`WanBlock::modulation`]); `cond.cross_kv`: its cached cross K/V.
    fn forward(
        &self,
        hidden: &CudaTensor,
        encoder: &CudaTensor,
        cond: BlockCond<'_>,
        rope: &(CudaTensor, CudaTensor),
        image: Option<&CudaTensor>,
        mask: Option<&BlockCausal>,
        vsa: Option<&VsaCtx>,
        ar: Option<&super::ar_cache::ArFrame>,
        kv: Option<super::causal::KvAt<'_>>,
        plan: AttnPlan,
    ) -> Result<CudaTensor> {
        use super::stats::phase;
        let e = cond.e;
        // Per-frame modulation (Wan 2.2 TI2V: the first latent frame at
        // timestep 0): `e` has one row per (batch, latent frame), and every
        // modulated op sees the tokens as `[b·T, S/T, dim]` — time-major
        // tokens make each frame a contiguous row block. Attention and the
        // FFN keep `[b, S, dim]`. Both are the same buffer.
        let b = hidden.shape[0];
        let v = |x: &CudaTensor| mod_view(x, e);
        let u = |x: CudaTensor| unview(x, b);
        let normed = phase("1_norm_msa", || {
            v(hidden)?
                .ln_adaln_e(e, SCALE_MSA, SHIFT_MSA, self.eps)
                .and_then(u)
        })?;
        let attn = phase("2_self_attn", || {
            self.attn1
                .forward_self(&normed, rope, mask, self.gate.as_ref(), vsa, ar, kv, plan)
        })?;
        // bf16 activations: FastVideo's residual + norm rounding points
        // (wan::fuse); f32 activations keep the op chain.
        let (w2, b2) = (&self.norm2_weight, &self.norm2_bias);
        let (hidden, normed) = phase("3_residual_norm_cross", || {
            let (hv, av) = (v(hidden)?, v(&attn)?);
            if let Some((h, n)) =
                super::fuse::self_residual_norm(&hv, &av, e, GATE_MSA, w2, b2, self.eps)?
            {
                return Ok((u(h)?, u(n)?));
            }
            let hidden = u(hv.residual_gate_add_e(&av, e, GATE_MSA)?)?;
            let normed = hidden.layer_norm(self.eps, Some(w2), Some(b2))?;
            Ok::<_, TensorError>((hidden, normed))
        })?;
        let cross = phase("5_cross_attn", || {
            self.attn2
                .forward_cross(&normed, encoder, image, cond.cross_kv)
        })?;
        let (hidden, normed) = phase("6_residual_norm_ffn", || {
            let (hv, cv) = (v(&hidden)?, v(&cross)?);
            if let Some((h, n)) =
                super::fuse::cross_residual_norm_mod(&hv, &cv, e, SCALE_FFN, SHIFT_FFN, self.eps)?
            {
                return Ok((u(h)?, u(n)?));
            }
            let hidden = hv.add(&cv)?;
            let normed = hidden.ln_adaln_e(e, SCALE_FFN, SHIFT_FFN, self.eps)?;
            Ok::<_, TensorError>((u(hidden)?, u(normed)?))
        })?;
        let ff = phase("7_ffn", || self.ffn.forward(&normed))?;
        phase("8_residual_ffn", || {
            let (hv, fv) = (v(&hidden)?, v(&ff)?);
            match super::fuse::gate_residual(&hv, &fv, e, GATE_FFN)? {
                Some(h) => u(h),
                None => u(hv.residual_gate_add_e(&fv, e, GATE_FFN)?),
            }
        })
    }
}

/// `x` (`[b, S, dim]`) as `[rows, b·S/rows, dim]` for a modulation table with
/// `rows` rows: one per batch row (the usual case: `x` itself), or one per
/// (batch row, latent frame) under per-frame timesteps.
fn mod_view(x: &CudaTensor, e: &CudaTensor) -> Result<CudaTensor> {
    let [b, s, d] = x.shape[..] else {
        return Err(TensorError::Message(format!(
            "modulated op expects [b, S, dim], got {:?}",
            x.shape
        )));
    };
    let rows = e.shape[0];
    if rows == b {
        return Ok(x.clone());
    }
    if rows % b != 0 || (b * s) % rows != 0 {
        return Err(TensorError::Message(format!(
            "modulation rows {rows} do not tile {:?}",
            x.shape
        )));
    }
    x.reshape(vec![rows, b * s / rows, d])
}

/// Undo [`mod_view`]: back to `[b, S, dim]`.
fn unview(x: CudaTensor, b: usize) -> Result<CudaTensor> {
    if x.shape[0] == b {
        return Ok(x);
    }
    let d = *x.shape.last().unwrap_or(&1);
    let n = x.numel() / (b * d).max(1);
    x.reshape_owned(vec![b, n, d])
}

impl super::offload::OffloadBlock for WanBlock {
    /// Every linear, in one fixed order. Quantized, LoRA and NVFP4 linears
    /// (and absent optional ones) stay in the skeleton.
    fn for_each_linear_mut(&mut self, f: &mut dyn FnMut(&mut Linear) -> Result<()>) -> Result<()> {
        for attn in [&mut self.attn1, &mut self.attn2] {
            f(&mut attn.q_or_qkv)?;
            if let Some(kv) = attn.kv.as_mut() {
                f(kv)?;
            }
            f(&mut attn.to_out)?;
            for extra in [&mut attn.add_k, &mut attn.add_v, &mut attn.proj_l] {
                if let Some(l) = extra.as_mut() {
                    f(l)?;
                }
            }
        }
        if let Some(g) = self.gate.as_mut() {
            f(g)?;
        }
        f(&mut self.ffn.proj)?;
        f(&mut self.ffn.out)
    }
}

/// The DiT blocks, shared by the clones of one model (the device ring and
/// the pinned host copy are per model, not per clone).
#[derive(Clone)]
struct WanBlocks(std::sync::Arc<super::offload::BlockWeights<WanBlock>>);

impl std::fmt::Debug for WanBlocks {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0.describe())
    }
}

impl std::ops::Deref for WanBlocks {
    type Target = super::offload::BlockWeights<WanBlock>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl WanBlocks {
    fn resident(blocks: Vec<WanBlock>) -> Self {
        Self(std::sync::Arc::new(super::offload::BlockWeights::resident(
            "wan dit", blocks,
        )))
    }
}

/// Where a Wan DiT's blocks live ([`WanTransformer3D::load_with_residency`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum WanBlockResidency {
    /// On the device for the whole run (every model before the A14B swap).
    #[default]
    Resident,
    /// `FASTVIDEO_DIT_OFFLOAD=streamed`: pinned host memory, copied one
    /// block ahead (`FASTVIDEO_DIT_OFFLOAD_LOOKAHEAD`) of the computing one.
    Streamed,
    /// Pinned host memory between uses, the whole model on the device while
    /// in use: the A14B expert that is not denoising stays parked
    /// ([`WanTransformer3D::park`]).
    Parked,
}

impl WanBlockResidency {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Resident => "resident",
            Self::Streamed => "streamed",
            Self::Parked => "parked",
        }
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
    /// Resident, layerwise-streamed, or parked (the A14B expert swap):
    /// [`super::offload::BlockWeights`].
    blocks: WanBlocks,
    proj_out: Linear,
    scale_shift_table: CudaTensor, // [1, 2, dim]
    freq_dim: usize,
    /// RoPE tables keyed by `(seq_len, head_dim, start_frame)`: built and
    /// uploaded once.
    rotary_cache: std::sync::Arc<
        std::sync::Mutex<
            std::collections::HashMap<(usize, usize, usize), (CudaTensor, CudaTensor)>,
        >,
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
    /// Text conditioning per encoder tensor, between [`Self::begin_text_cache`]
    /// and [`Self::end_text_cache`] (`None` outside: nothing cached).
    text_cache: std::sync::Arc<std::sync::Mutex<Option<Vec<std::sync::Arc<PreparedText>>>>>,
    /// Time conditioning per timestep vector (weights-only function; kept).
    time_cache: std::sync::Arc<std::sync::Mutex<Vec<std::sync::Arc<PreparedTime>>>>,
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

use super::sol_cache::{mean_abs, mean_abs_delta};

/// `mean|current - previous| / max(mean|previous|, 1e-8)`, reduced on the
/// device (two scalars come back) — see [`super::sol_cache::abs_sums`].
fn relative_l1(current: &CudaTensor, previous: &CudaTensor) -> Result<f64> {
    let (num, den, n) = super::sol_cache::abs_sums(current, previous)?;
    let n = n.max(1) as f64;
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
        let blocks = WanBlocks::resident(
            (0..cfg.num_layers)
                .map(|_| WanBlock::zeros(&cfg))
                .collect::<Result<Vec<_>>>()?,
        );
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
            text_cache: Default::default(),
            time_cache: Default::default(),
            cfg,
        })
    }

    pub fn load(cfg: WanVideoArchConfig, map: &WeightMap) -> Result<Self> {
        Self::from_map(cfg, map)
    }

    pub fn from_map(cfg: WanVideoArchConfig, map: &WeightMap) -> Result<Self> {
        Self::load_with_residency(cfg, map, WanBlockResidency::Resident)
    }

    /// [`Self::load`] with the blocks resident, streamed or parked
    /// ([`WanBlockResidency`]). Streamed and parked blocks leave the device
    /// as each one loads, so loading holds one block there at a time.
    pub fn load_with_residency(
        cfg: WanVideoArchConfig,
        map: &WeightMap,
        residency: WanBlockResidency,
    ) -> Result<Self> {
        let dim = cfg.hidden_size();
        let p = cfg.patch_size;
        let plan =
            super::quant::WanQuantPlan::from_env(cfg.num_layers).map_err(TensorError::Message)?;
        plan.announce();
        let mut weights = match residency {
            WanBlockResidency::Resident => {
                super::offload::BlockWeights::new("wan dit", super::offload::Residency::Resident)
            }
            WanBlockResidency::Streamed => {
                super::offload::BlockWeights::new("wan dit", super::offload::Residency::Streamed)
            }
            WanBlockResidency::Parked => {
                super::offload::BlockWeights::new("wan dit", super::offload::Residency::Streamed)
                    .with_whole_ring()
            }
        };
        // Each block is quantized as it loads, so its bf16 weights are freed
        // before the next block's arrive.
        for i in 0..cfg.num_layers {
            let mut block = WanBlock::load(map, &format!("blocks.{i}"), &cfg)?;
            if let Some(kind) = plan.block(i) {
                block.quantize(kind, &cfg)?;
            }
            weights.push(block)?;
        }
        if residency != WanBlockResidency::Resident {
            super::log::info(format_args!("{}", weights.describe()));
        }
        let blocks = WanBlocks(std::sync::Arc::new(weights));
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
            text_cache: Default::default(),
            time_cache: Default::default(),
            cfg,
        })
    }

    /// Where the blocks live.
    pub fn block_residency(&self) -> WanBlockResidency {
        if self.blocks.is_whole_ring() {
            WanBlockResidency::Parked
        } else if self.blocks.residency().is_streamed() {
            WanBlockResidency::Streamed
        } else {
            WanBlockResidency::Resident
        }
    }

    /// A parked model ([`WanBlockResidency::Parked`]): let its blocks' device
    /// copy go (they stay in pinned host memory; the next forward brings
    /// them back, block by block behind the compute). Waits for the kernels
    /// that read them. A no-op for resident and streamed models.
    pub fn park(&self) {
        if self.blocks.is_whole_ring() {
            self.blocks.release_device();
        }
    }

    /// Whether a parked model's blocks are on the device now.
    pub fn is_on_device(&self) -> bool {
        !self.blocks.residency().is_streamed() || self.blocks.has_device_ring()
    }

    /// Bytes of block weight in pinned host memory (0 when resident).
    pub fn host_block_bytes(&self) -> u64 {
        self.blocks.host_bytes()
    }

    /// Log and reset the block-copy statistics (streamed and parked models).
    pub fn report_offload(&self, what: &str) -> super::offload::OffloadStats {
        self.blocks.report(what)
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
        let sol_value = std::env::var("FASTVIDEO_WAN_SOL_ATTN")
            .ok()
            .or_else(|| std::env::var("WAN22_SOL_ATTN").ok());
        let sol = sol_attn_requested(sol_value.as_deref());
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
        } else if self.cfg.num_layers == 30
            && self.cfg.out_channels == 16
            && !self.cfg.is_moe()
            && !self.cfg.causal
            && sol
        {
            // Wan 2.1 1.3B (T2V / FastWan / TurboWan): 30 layers, 16-channel
            // latents. `fullstack`: the base model's dense guards.
            fastvideo_models::wan::sol::sol_13b_profile(sol_value.as_deref())
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

    /// From here until [`Self::end_text_cache`], the text embedding and every
    /// block's cross-attention K/V are computed once per encoder tensor (by
    /// storage identity: pass the same tensor, or clones of it, each step)
    /// and reused. Off with `FASTVIDEO_WAN_COND_CACHE=0`.
    pub fn begin_text_cache(&self) {
        let mut slot = self.text_cache.lock().expect("text cache");
        *slot = cond_cache_enabled().then(Vec::new);
    }

    /// Drop the text conditioning cached since [`Self::begin_text_cache`].
    pub fn end_text_cache(&self) {
        *self.text_cache.lock().expect("text cache") = None;
    }

    fn prepare_text(
        &self,
        encoder: &CudaTensor,
    ) -> Result<(CudaTensor, Option<std::sync::Arc<PreparedText>>)> {
        let id = encoder.storage_id();
        {
            let slot = self.text_cache.lock().expect("text cache");
            let (Some(cache), Some(id)) = (slot.as_ref(), id) else {
                drop(slot);
                return Ok((self.text_embedder.forward_gelu(encoder)?, None));
            };
            if let Some(hit) = cache
                .iter()
                .find(|p| p.encoder.storage_id() == Some(id) && p.encoder.shape == encoder.shape)
            {
                return Ok((hit.embedded.clone(), Some(hit.clone())));
            }
        }
        let embedded = self.text_embedder.forward_gelu(encoder)?;
        let kv = (0..self.blocks.len())
            .map(|i| self.blocks.with(i, |b| b.attn2.cross_kv(&embedded)))
            .collect::<Result<Vec<_>>>()?;
        let prepared = std::sync::Arc::new(PreparedText {
            encoder: encoder.clone(),
            embedded: embedded.clone(),
            kv,
        });
        let mut slot = self.text_cache.lock().expect("text cache");
        if let Some(cache) = slot.as_mut() {
            if cache.len() >= TEXT_CACHE_ENTRIES {
                cache.remove(0);
            }
            cache.push(prepared.clone());
        }
        Ok((embedded, Some(prepared)))
    }

    /// Time embedding, `timestep_proj` and each block's modulation for
    /// `timestep` (`[b]`, or `[b·T]` per latent frame), from the cache when
    /// the same values were seen. `b` is the number of timestep values.
    fn prepare_time(
        &self,
        timestep: &CudaTensor,
        b: usize,
    ) -> Result<std::sync::Arc<PreparedTime>> {
        let key: Option<Vec<u32>> = cond_cache_enabled()
            .then(|| {
                timestep
                    .host_cow()
                    .map(|h| h.iter().map(|v| v.to_bits()).collect())
            })
            .transpose()?;
        if let Some(key) = &key {
            let cache = self.time_cache.lock().expect("time cache");
            if let Some(hit) = cache
                .iter()
                .find(|p| &p.key == key && p.timestep_proj.shape[0] == b)
            {
                return Ok(hit.clone());
            }
        }
        let dim = self.cfg.hidden_size();
        let temb = self
            .time_embedder
            .forward_silu(&nn::sinusoidal_timesteps(timestep, self.freq_dim)?)?;
        let timestep_proj = self
            .time_proj
            .forward(&temb.silu())?
            .reshape(vec![b, 6, dim])?;
        let e = (0..self.blocks.len())
            .map(|i| timestep_proj.add(&self.blocks.skeleton(i).scale_shift_table))
            .collect::<Result<Vec<_>>>()?;
        let prepared = std::sync::Arc::new(PreparedTime {
            key: key.clone().unwrap_or_default(),
            temb,
            timestep_proj,
            e,
        });
        if key.is_some() {
            let mut cache = self.time_cache.lock().expect("time cache");
            if cache.len() >= TIME_CACHE_ENTRIES {
                cache.remove(0);
            }
            cache.push(prepared.clone());
        }
        Ok(prepared)
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
        if super::device::global_device().is_none()
            || !(0..self.blocks.len()).any(|i| self.blocks.skeleton(i).gate.is_some())
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

    /// Drop the RoPE tables of block offsets before `start_frame` (an
    /// open-ended absolute-RoPE rollout would otherwise keep one table per
    /// block for ever). Tables at offset 0 stay.
    pub fn forget_rotary_before(&self, start_frame: usize) {
        self.rotary_cache
            .lock()
            .expect("rotary cache lock")
            .retain(|&(_, _, s), _| s == 0 || s >= start_frame);
    }

    /// Get-or-build the `[seq, head_dim]` RoPE tables for a latent grid.
    pub fn rotary_for(&self, t: usize, h: usize, w: usize) -> Result<(CudaTensor, CudaTensor)> {
        self.rotary_at(t, h, w, 0)
    }

    /// [`Self::rotary_for`] for latent frames `start_frame ..`.
    pub fn rotary_at(
        &self,
        t: usize,
        h: usize,
        w: usize,
        start_frame: usize,
    ) -> Result<(CudaTensor, CudaTensor)> {
        let p = self.cfg.patch_size;
        let seq = (t / p[0]) * (h / p[1]) * (w / p[2]);
        let key = (seq, self.cfg.attention_head_dim, start_frame);
        let mut map = self.rotary_cache.lock().expect("rotary cache lock");
        if let Some(pair) = map.get(&key) {
            return Ok(pair.clone());
        }
        let (cos, sin) = wan_rope_at(&self.cfg, t, h, w, start_frame)?;
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
        // Block-causal self-attention over the whole clip (FastVideo's
        // full-sequence causal forward, `_prepare_blockwise_causal_attn_mask`;
        // its inference is `forward_kv`) as a kernel parameter: the flash
        // kernel skips invisible key tiles and never builds the [S, S] mask
        // (`nn::sdpa_block_causal` materializes it only on its fallback).
        let mask = if self.cfg.causal && ar.is_none() {
            let p = self.cfg.patch_size;
            let frame_tokens = (h / p[1].max(1)) * (w / p[2].max(1));
            Some(
                super::attn::block_causal_for(&self.cfg, frame_tokens).ok_or_else(|| {
                    TensorError::Message(format!(
                        "causal Wan: local_attn_size {} is not whole {}-frame blocks; the \
                         full-sequence mask takes whole blocks only",
                        self.cfg.local_attn_size, self.cfg.num_frames_per_block
                    ))
                })?,
            )
        } else {
            None
        };
        let mut hidden = self.patch_embed(latents)?;
        if super::tensor::bf16_residual() {
            // The reference's patch embedding is a bf16 conv: the residual
            // stream starts bf16 (FASTVIDEO_BF16_ACT).
            hidden = hidden.quantize_bf16()?;
        }
        let _ = dim;
        // Step-invariant conditioning: cached by timestep values / encoder
        // tensor when enabled, recomputed with the same ops otherwise.
        // One timestep per batch row, or (Wan 2.2 TI2V, `expand_timesteps`)
        // one per (batch row, latent frame): the modulation then goes per
        // frame (see `mod_view`).
        let p0 = self.cfg.patch_size[0].max(1);
        let n_t = timestep.numel();
        if n_t != b && n_t != b * (t / p0) {
            return Err(TensorError::Message(format!(
                "timestep has {n_t} values for batch {b} and {} latent frames",
                t / p0
            )));
        }
        let time = self.prepare_time(timestep, n_t)?;
        let (temb, timestep_proj) = (&time.temb, &time.timestep_proj);
        let (encoder, text) = self.prepare_text(encoder)?;
        let cond = |layer: usize| BlockCond {
            e: &time.e[layer],
            cross_kv: text.as_ref().map(|t| &t.kv[layer]),
        };
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
            hidden = self.blocks.with(0, |block| {
                block.forward(
                    &hidden,
                    &encoder,
                    cond(0),
                    &rope,
                    image.as_ref(),
                    mask.as_ref(),
                    vsa.as_deref(),
                    ar.as_ref(),
                    None,
                    self.attn_plan(0, morton_grid),
                )
            })?;
            match self.begin_a14b_tail(&hidden)? {
                Some(true) => hidden = self.add_a14b_residual(hidden)?,
                reuse => {
                    let prefix = hidden.clone();
                    for layer in 1..self.blocks.len() {
                        hidden = self.blocks.with(layer, |block| {
                            block.forward(
                                &hidden,
                                &encoder,
                                cond(layer),
                                &rope,
                                image.as_ref(),
                                mask.as_ref(),
                                vsa.as_deref(),
                                ar.as_ref(),
                                None,
                                self.attn_plan(layer, morton_grid),
                            )
                        })?;
                    }
                    if reuse == Some(false) {
                        self.finish_a14b_tail(&prefix, &hidden)?;
                    }
                }
            }
        } else {
            let dump_blocks = super::dump::blocks();
            if dump_blocks {
                super::dump::rows_strided(
                    &super::dump::named("patch_embed"),
                    &hidden,
                    super::dump::BLOCK_ROW_STRIDE,
                )?;
                super::dump::tensor(&super::dump::named("timestep_proj"), &timestep_proj)?;
            }
            for layer in 0..self.blocks.len() {
                hidden = self.blocks.with(layer, |block| {
                    block.forward(
                        &hidden,
                        &encoder,
                        cond(layer),
                        &rope,
                        image.as_ref(),
                        mask.as_ref(),
                        vsa.as_deref(),
                        ar.as_ref(),
                        None,
                        self.attn_plan(layer, morton_grid),
                    )
                })?;
                if dump_blocks {
                    super::dump::rows_strided(
                        &super::dump::named(&format!("block_{layer}")),
                        &hidden,
                        super::dump::BLOCK_ROW_STRIDE,
                    )?;
                }
            }
            if let Some(before) = block_in.as_ref() {
                self.finish_sol_tea(before, &hidden)?;
            }
        }
        // Output head: table [1, 2, dim] + temb broadcast over both rows.
        let temb_rows = temb.unsqueeze(1)?;
        let e = CudaTensor::cat(&[&temb_rows, &temb_rows], 1)?.add(&self.scale_shift_table)?;
        hidden = unview(
            mod_view(&hidden, &e)?.ln_adaln_e(&e, 1, 0, self.cfg.eps)?,
            b,
        )?;
        hidden = self.proj_out.forward(&hidden)?;
        if taylor == TaylorPhase::Compute {
            self.update_taylor(&hidden)?;
        }
        self.unpatchify(hidden, b, t, h, w)
    }

    /// One causal block through the KV cache: FastVideo
    /// `CausalWanTransformer3DModel._forward_inference`. `latents` are the
    /// block's `[B, C, F, H, W]` (`F` = frames in the block), `start_frame`
    /// its first latent frame (RoPE offset and cache position). No sparse
    /// routes, caches or VSA: the reference has none on this path. With
    /// dumping on and [`super::dump::blocks`] set, each block output is
    /// written as `<prefix>block_<i>`; [`super::dump::op_blocks`] layers write
    /// their key window and attention output.
    pub fn forward_kv(
        &self,
        latents: &CudaTensor,
        timestep: &CudaTensor,
        encoder: &CudaTensor,
        cache: &super::causal::CausalKvCache,
        start_frame: usize,
    ) -> Result<CudaTensor> {
        let [b, _, t, h, w] = latents.shape[..] else {
            return Err(TensorError::Message(format!(
                "forward_kv expects BCTHW latents, got {:?}",
                latents.shape
            )));
        };
        let spec = self.kv_table_spec(t, h, w, cache, start_frame)?;
        let tables = self.kv_tables(&spec, h, w)?;
        let time = self.kv_time(timestep, b)?;
        let text = self.kv_text(encoder)?;
        self.forward_kv_cond(latents, &time, &text, cache, start_frame, &tables)
    }

    /// The RoPE rows a [`Self::forward_kv`] of a `t`-frame block at
    /// `start_frame` reads, given the cache's state now.
    pub fn kv_table_spec(
        &self,
        t: usize,
        h: usize,
        w: usize,
        cache: &super::causal::CausalKvCache,
        start_frame: usize,
    ) -> Result<KvTableSpec> {
        let p = self.cfg.patch_size;
        let frame_tokens = (h / p[1].max(1)) * (w / p[2].max(1));
        let n = t * frame_tokens;
        let (frames, start, rows) = if cache.spec.relativistic {
            // The window's table from position 0: every layer's cache sits
            // at the same pointers, so layer 0's plan sizes it once.
            let window = cache.window_after(start_frame * frame_tokens, n)?;
            (cache.spec.window_frames().max(t), 0, window)
        } else {
            (t, start_frame, n)
        };
        let sink = if cache.spec.rebase_sink {
            let target = cache.spec.sink_target(start_frame + t);
            (target > 0).then(|| (target, cache.spec.sink / frame_tokens.max(1)))
        } else {
            None
        };
        Ok(KvTableSpec {
            frames,
            start_frame: start,
            rows,
            sink,
        })
    }

    /// [`KvTableSpec`]'s tables from the device RoPE cache.
    pub fn kv_tables(&self, spec: &KvTableSpec, h: usize, w: usize) -> Result<KvTables> {
        let (cos, sin) = self.rotary_at(spec.frames, h, w, spec.start_frame)?;
        let rope = if spec.rows == cos.shape[0] {
            (cos, sin)
        } else {
            (cos.narrow(0, 0, spec.rows)?, sin.narrow(0, 0, spec.rows)?)
        };
        let sink = match spec.sink {
            Some((target, sink_f)) => Some((
                target,
                self.rotary_at(sink_f, h, w, 0)?,
                self.rotary_at(sink_f, h, w, target)?,
            )),
            None => None,
        };
        Ok(KvTables { rope, sink })
    }

    /// [`Self::kv_tables`] as host tensors computed afresh (no cache, no
    /// device): the same values, for a caller that copies them into buffers
    /// of its own (the inputs of a CUDA graph, `wan::graph`).
    pub fn kv_tables_host(&self, spec: &KvTableSpec, h: usize, w: usize) -> Result<KvTables> {
        let (cos, sin) = wan_rope_at(&self.cfg, spec.frames, h, w, spec.start_frame)?;
        let rope = if spec.rows == cos.shape[0] {
            (cos, sin)
        } else {
            (cos.narrow(0, 0, spec.rows)?, sin.narrow(0, 0, spec.rows)?)
        };
        let sink = match spec.sink {
            Some((target, sink_f)) => Some((
                target,
                wan_rope_at(&self.cfg, sink_f, h, w, 0)?,
                wan_rope_at(&self.cfg, sink_f, h, w, target)?,
            )),
            None => None,
        };
        Ok(KvTables { rope, sink })
    }

    /// The time conditioning of `timestep` (`b` values), from the time
    /// cache when seen before. Held by the caller, it stays valid whatever
    /// the cache evicts.
    pub fn kv_time(&self, timestep: &CudaTensor, b: usize) -> Result<KvTime> {
        Ok(KvTime(self.prepare_time(timestep, b)?))
    }

    /// The text conditioning of `encoder`: the text embedding and, with the
    /// text cache on ([`Self::begin_text_cache`]), every block's
    /// cross-attention K/V (else each forward computes them from the
    /// embedding).
    pub fn kv_text(&self, encoder: &CudaTensor) -> Result<KvText> {
        let (embedded, text) = self.prepare_text(encoder)?;
        Ok(KvText {
            embedded,
            kv: text.map(|t| t.kv.clone()),
        })
    }

    /// [`Self::forward_kv`] on prepared conditioning and RoPE tables: the
    /// rebased sink (when `tables.sink`), then the block. Past the host
    /// bookkeeping of the cache, only kernel launches and stream-ordered
    /// allocations: what a CUDA graph can capture (`wan::graph`).
    pub fn forward_kv_cond(
        &self,
        latents: &CudaTensor,
        time: &KvTime,
        text: &KvText,
        cache: &super::causal::CausalKvCache,
        start_frame: usize,
        tables: &KvTables,
    ) -> Result<CudaTensor> {
        let [b, _c, t, h, w] = latents.shape[..] else {
            return Err(TensorError::Message(format!(
                "forward_kv expects BCTHW latents, got {:?}",
                latents.shape
            )));
        };
        let p = self.cfg.patch_size;
        let frame_tokens = (h / p[1].max(1)) * (w / p[2].max(1));
        let rope = &tables.rope;
        if let Some((target, orig, new)) = &tables.sink {
            cache.rebase_sink(*target, (&orig.0, &orig.1), (&new.0, &new.1))?;
        }
        let mut hidden = self.patch_embed(latents)?;
        if super::tensor::bf16_residual() {
            hidden = hidden.quantize_bf16()?;
        }
        let time = &time.0;
        let encoder = &text.embedded;
        let dump_blocks = super::dump::blocks();
        for layer in 0..self.blocks.len() {
            let op = dump_blocks && super::dump::op_blocks().contains(&layer);
            super::dump::set_op_block(op.then_some(layer));
            hidden = self.blocks.with(layer, |block| {
                block.forward(
                    &hidden,
                    encoder,
                    BlockCond {
                        e: &time.e[layer],
                        cross_kv: text.kv.as_ref().map(|kv| &kv[layer]),
                    },
                    rope,
                    None,
                    None,
                    None,
                    None,
                    Some(super::causal::KvAt {
                        cache,
                        layer,
                        current_start: start_frame * frame_tokens,
                    }),
                    AttnPlan::default(),
                )
            })?;
            super::dump::set_op_block(None);
            if dump_blocks {
                super::dump::rows_strided(
                    &super::dump::named(&format!("block_{layer}")),
                    &hidden,
                    super::dump::BLOCK_ROW_STRIDE,
                )?;
            }
        }
        let temb_rows = time.temb.unsqueeze(1)?;
        let e = CudaTensor::cat(&[&temb_rows, &temb_rows], 1)?.add(&self.scale_shift_table)?;
        hidden = hidden.ln_adaln_e(&e, 1, 0, self.cfg.eps)?;
        hidden = self.proj_out.forward(&hidden)?;
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic small weights for every key the loader asks for.
    fn generated_map() -> WeightMap {
        WeightMap::generated(|key, shape| {
            let n: usize = shape.iter().product();
            let seed = key.bytes().fold(0x9e37_79b9u32, |h, b| {
                h.rotate_left(5) ^ u32::from(b).wrapping_mul(0x0100_0193)
            });
            (0..n)
                .map(|i| {
                    let x = (i as u32).wrapping_mul(2_654_435_761).wrapping_add(seed);
                    ((x >> 8) as f32 / (1u32 << 24) as f32 - 0.5) * 0.2
                })
                .collect()
        })
    }

    fn ramp(shape: &[usize], k: f32) -> CudaTensor {
        let n: usize = shape.iter().product();
        CudaTensor::from_vec(
            (0..n).map(|i| ((i as f32) * k).sin()).collect(),
            shape.to_vec(),
        )
        .unwrap()
    }

    fn close(a: &CudaTensor, b: &CudaTensor) {
        let (a, b) = (a.host_cow().unwrap(), b.host_cow().unwrap());
        assert_eq!(a.len(), b.len());
        let worst = a
            .iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max);
        assert!(worst < 1e-5, "max abs {worst}");
    }

    fn ts(v: Vec<f32>) -> CudaTensor {
        let n = v.len();
        CudaTensor::from_vec(v, vec![n]).unwrap()
    }

    /// Per-frame timesteps (Wan 2.2 TI2V `expand_timesteps`) with every frame
    /// at the same value are the per-batch timestep: the modulation views
    /// and the output head's must be pure re-indexing, for batch 1 and 2.
    #[test]
    fn per_frame_timesteps_equal_per_batch_when_uniform() {
        let dit = WanTransformer3D::load(WanVideoArchConfig::tiny(), &generated_map()).unwrap();
        let (t, h, w) = (3, 4, 4);
        let lat = ramp(&[1, 4, t, h, w], 0.37);
        let enc = ramp(&[1, 8, 16], 0.11);
        let one = dit.forward(&lat, &ts(vec![500.0]), &enc).unwrap();
        let frames = dit.forward(&lat, &ts(vec![500.0; t]), &enc).unwrap();
        close(&one, &frames);

        let lat2 = CudaTensor::cat(&[&lat, &lat], 0).unwrap();
        let enc2 = CudaTensor::cat(&[&enc, &enc], 0).unwrap();
        let two = dit.forward(&lat2, &ts(vec![500.0, 500.0]), &enc2).unwrap();
        let two_frames = dit.forward(&lat2, &ts(vec![500.0; 2 * t]), &enc2).unwrap();
        close(&two, &two_frames);
        // Frame 0 at timestep 0 changes the output (the modulation reaches it).
        let mut ti2v = vec![500.0; t];
        ti2v[0] = 0.0;
        let pinned = dit.forward(&lat, &ts(ti2v), &enc).unwrap();
        let (a, b) = (one.host_cow().unwrap(), pinned.host_cow().unwrap());
        assert!(a.iter().zip(b.iter()).any(|(x, y)| (x - y).abs() > 1e-4));
        // A timestep count that is neither the batch nor batch x frames is refused.
        assert!(dit.forward(&lat, &ts(vec![1.0, 2.0]), &enc).is_err());
    }

    /// The A14B expert swap and layerwise streaming move weights, not math:
    /// a parked or streamed DiT gives the resident one's bits, before and
    /// after a park (CFG batch of two, the text K/V cache on).
    #[test]
    fn parked_and_streamed_blocks_match_resident() {
        let cfg = WanVideoArchConfig::tiny();
        let map = generated_map();
        let resident = WanTransformer3D::load(cfg.clone(), &map).unwrap();
        let parked =
            WanTransformer3D::load_with_residency(cfg.clone(), &map, WanBlockResidency::Parked)
                .unwrap();
        let streamed =
            WanTransformer3D::load_with_residency(cfg, &map, WanBlockResidency::Streamed).unwrap();
        assert_eq!(resident.block_residency(), WanBlockResidency::Resident);
        assert_eq!(parked.block_residency(), WanBlockResidency::Parked);
        assert_eq!(streamed.block_residency(), WanBlockResidency::Streamed);
        assert_eq!(resident.host_block_bytes(), 0);
        assert!(parked.host_block_bytes() > 0);
        let (t, h, w) = (3, 4, 4);
        let lat = ramp(&[2, 4, t, h, w], 0.29);
        let enc = ramp(&[2, 8, 16], 0.13);
        let bits = |d: &WanTransformer3D| -> Vec<u32> {
            let y = d.forward(&lat, &ts(vec![700.0, 700.0]), &enc).unwrap();
            y.host_cow().unwrap().iter().map(|v| v.to_bits()).collect()
        };
        let want = bits(&resident);
        assert_eq!(want, bits(&parked));
        parked.park();
        assert_eq!(want, bits(&parked));
        assert_eq!(want, bits(&streamed));
        // Parking a resident model does nothing.
        resident.park();
        assert!(resident.is_on_device());
        assert_eq!(want, bits(&resident));
    }
}
