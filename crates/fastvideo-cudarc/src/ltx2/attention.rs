//! `LTX2Attention` and the feed-forward it is paired with — the two layers the
//! connectors and every DiT block are made of.
//!
//! One attention class serves five roles (self, text cross, audio→video,
//! video→audio, connector). They differ only in widths and in which rotary
//! table, if any, rotates q and k, so that is all [`Attention::forward`] takes.
//!
//! Two conventions are easy to get wrong:
//!
//! * `qk_norm = "rms_norm_across_heads"`: q and k are RMS-normalised over the
//!   *whole* inner width (one statistic per token across all heads) with a
//!   learned `[inner]` weight, before the head split. It is not a per-head norm.
//! * the rotary is "split": rotate_half inside each head with a table that
//!   *differs per head*. [`DeviceRope`] folds the head axis into the row axis
//!   of `rope_half`'s `[rows, D]` table — `[B, H, S, D]` viewed as
//!   `[B, 1, H·S, D]` is the same memory — so no kernel is needed for it.

use fastvideo_models::ltx2::memory::FeedForwardChunking;
use fastvideo_models::ltx2::{SplitRope, SplitRopeLut};
use std::borrow::Cow;

use crate::wan::nn::{scaled_dot_product_attention, Linear};
use crate::wan::tensor::{CudaTensor, Result};
use crate::wan::weights::{cuda_tensor_shaped, WeightMap};

use super::keys::Keys;
use super::{msg, pinned};

/// A [`SplitRope`] on the device, in `rope_half`'s layout: `[H·S, D]` cos and
/// sin, head-major, each pair's value duplicated across the two halves.
#[derive(Debug, Clone)]
pub struct DeviceRope {
    cos: CudaTensor,
    sin: CudaTensor,
    heads: usize,
    tokens: usize,
    head_dim: usize,
}

impl DeviceRope {
    pub fn upload(table: &SplitRope) -> Result<Self> {
        let (cos, sin) = table.rotate_half_tables();
        let (rows, d) = (table.heads * table.tokens, table.half * 2);
        Ok(Self {
            cos: pinned(cos, vec![rows, d])?,
            sin: pinned(sin, vec![rows, d])?,
            heads: table.heads,
            tokens: table.tokens,
            head_dim: d,
        })
    }

    /// The table of `lut` ([`SplitRopeLut::expand`]), gathered on the device
    /// when there is one: only the factored form (a few hundred distinct
    /// fractions' cos/sin and a `[tokens, axes]` index) is built on the host
    /// and uploaded, not the `[H·S, D]` tables. Bit-identical to
    /// [`Self::upload`] of the expanded table; without a device it is that.
    pub fn from_lut(lut: &SplitRopeLut) -> Result<Self> {
        #[cfg(feature = "cuda")]
        if crate::wan::device::global_device().is_some() {
            let (rows, d) = (lut.heads * lut.tokens, lut.half * 2);
            let (cos, sin) = crate::wan::ops::ltx_split_rope_device(
                &lut.index, &lut.cos, &lut.sin, lut.heads, lut.tokens, lut.half, lut.axes,
                lut.pad, lut.n,
            )?;
            return Ok(Self {
                cos: CudaTensor::from_device_slice(cos, vec![rows, d])?,
                sin: CudaTensor::from_device_slice(sin, vec![rows, d])?,
                heads: lut.heads,
                tokens: lut.tokens,
                head_dim: d,
            });
        }
        Self::upload(&lut.expand())
    }

    /// The cos / sin tables on the host (`[H·S, D]` each, row-major), for
    /// parity checks against [`SplitRope::rotate_half_tables`].
    pub fn host_tables(&self) -> Result<(Vec<f32>, Vec<f32>)> {
        Ok((self.cos.host_cow()?.into_owned(), self.sin.host_cow()?.into_owned()))
    }

    /// Keep `tokens` (ascending video-token indices) in the head-major table.
    pub fn index_tokens(&self, tokens: &[usize]) -> Result<Self> {
        let mut rows = Vec::with_capacity(self.heads * tokens.len());
        for h in 0..self.heads {
            for &t in tokens {
                if t >= self.tokens {
                    return Err(super::msg(format!(
                        "ltx2 rope: token {t} is past {} video tokens",
                        self.tokens
                    )));
                }
                rows.push(h * self.tokens + t);
            }
        }
        Ok(Self {
            cos: self.cos.index_select_rows(&rows)?,
            sin: self.sin.index_select_rows(&rows)?,
            heads: self.heads,
            tokens: tokens.len(),
            head_dim: self.head_dim,
        })
    }

    /// [`Self::index_tokens`] with the kept tokens already on the device: the
    /// head-major rows are built and gathered there (no host row list).
    /// `None` when the table is not on the device.
    #[cfg(feature = "cuda")]
    pub(crate) fn index_tokens_device(
        &self,
        idx: &cudarc::driver::CudaSlice<u32>,
    ) -> Result<Option<Self>> {
        let (Some(cos), Some(sin)) = (self.cos.dev()?, self.sin.dev()?) else {
            return Ok(None);
        };
        let rows = crate::wan::ops::ltx_rope_rows_device(idx, self.heads, self.tokens)?;
        let shape = vec![self.heads * idx.len(), self.head_dim];
        Ok(Some(Self {
            cos: CudaTensor::from_device_slice(
                crate::wan::ops::index_select_rows_dev_idx(&cos, self.head_dim, &rows)?,
                shape.clone(),
            )?,
            sin: CudaTensor::from_device_slice(
                crate::wan::ops::index_select_rows_dev_idx(&sin, self.head_dim, &rows)?,
                shape,
            )?,
            heads: self.heads,
            tokens: idx.len(),
            head_dim: self.head_dim,
        }))
    }

    /// The f32 `[H·S, r]` cos / sin tables and `r`, when this table rotates
    /// `[1, heads, seq, d]` (what [`Self::apply`] would accept), for the fused
    /// q/k kernel ([`super::fuse::qk_norm_rope`]). Every row holds each pair's
    /// value in both halves ([`Self::upload`]; [`Self::index_tokens`] only
    /// selects rows), which that kernel relies on to read half the table.
    pub(crate) fn fused_tables(
        &self,
        heads: usize,
        seq: usize,
        d: usize,
    ) -> Result<Option<(CudaTensor, CudaTensor, usize)>> {
        if (self.heads, self.tokens, self.head_dim) != (heads, seq, d) {
            return Ok(None);
        }
        let r = match self.cos.shape[..] {
            [rows, r] if rows == heads * seq && self.sin.shape == self.cos.shape && r % 2 == 0 && r > 0 && r <= d => r,
            _ => return Ok(None),
        };
        Ok(Some((self.cos.to_f32_act()?, self.sin.to_f32_act()?, r)))
    }

    /// Rotate `[1, H, S, D]`. Batch 1 only: with the head axis folded into the
    /// rows, a second batch element would need the table repeated.
    pub fn apply(&self, x: &CudaTensor) -> Result<CudaTensor> {
        if x.shape != [1, self.heads, self.tokens, self.head_dim] {
            return Err(msg(format!(
                "split rope built for [1, {}, {}, {}] applied to {:?}",
                self.heads, self.tokens, self.head_dim, x.shape
            )));
        }
        x.reshape(vec![1, 1, self.heads * self.tokens, self.head_dim])?
            .rope_half(&self.cos, &self.sin)?
            .reshape(x.shape.clone())
    }
}

/// Widths of one attention layer. `query_dim` is also the output width.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AttentionDims {
    pub query_dim: usize,
    /// Width of the key/value source; equals `query_dim` for self-attention.
    pub context_dim: usize,
    pub heads: usize,
    pub head_dim: usize,
}

impl AttentionDims {
    pub fn inner(&self) -> usize {
        self.heads * self.head_dim
    }
}

#[derive(Clone)]
pub struct Attention {
    to_q: Linear,
    to_k: Linear,
    to_v: Linear,
    to_out: Linear,
    to_gate_logits: Option<Linear>,
    norm_q: CudaTensor,
    norm_k: CudaTensor,
    dims: AttentionDims,
    eps: f32,
}

impl Attention {
    /// `prefix` is the diffusers module path (`transformer_blocks.3.attn1`);
    /// `keys` spells it for the checkpoint at hand.
    pub fn load(
        map: &WeightMap,
        keys: &Keys,
        prefix: &str,
        dims: AttentionDims,
        eps: f32,
        gated: bool,
    ) -> Result<Self> {
        let inner = dims.inner();
        let lin = |name: &str, i: usize, o: usize| {
            Linear::load(map, &keys.key(&format!("{prefix}.{name}")), i, o, true)
        };
        let norm = |name: &str| -> Result<CudaTensor> {
            let mut w =
                cuda_tensor_shaped(map, &keys.key(&format!("{prefix}.{name}.weight")), &[inner])?;
            w.pin_device()?;
            Ok(w)
        };
        let to_gate_logits = if gated {
            Some(lin("to_gate_logits", dims.query_dim, dims.heads)?)
        } else {
            None
        };
        Ok(Self {
            to_q: lin("to_q", dims.query_dim, inner)?,
            to_k: lin("to_k", dims.context_dim, inner)?,
            to_v: lin("to_v", dims.context_dim, inner)?,
            to_out: lin("to_out.0", inner, dims.query_dim)?,
            to_gate_logits,
            norm_q: norm("norm_q")?,
            norm_k: norm("norm_k")?,
            dims,
            eps,
        })
    }

    /// Per-head `2·σ(logits)` on SDPA output, diffusers `LTX2AudioVideoAttnProcessor`.
    fn apply_head_gates(&self, out: &CudaTensor, gate_logits: &CudaTensor) -> Result<CudaTensor> {
        let heads = self.dims.heads;
        let gates = gate_logits.try_sigmoid()?.try_mul_scalar(2.0)?;
        let gates = gates
            .permute(&[0, 2, 1])?
            .reshape(vec![1, heads, gates.shape[1], 1])?;
        out.mul(&gates)
    }

    /// `x`: `[1, Sq, query_dim]`. `context`: `[1, Sk, context_dim]`, or `None`
    /// for self-attention. `q_rope` rotates q; k is rotated by `k_rope`, or by
    /// `q_rope` when there is none (self-attention) — and not at all when
    /// `q_rope` is `None` (text cross-attention). No mask: every mask LTX-2.0
    /// builds for these layers is all ones.
    pub fn forward(
        &self,
        x: &CudaTensor,
        context: Option<&CudaTensor>,
        q_rope: Option<&DeviceRope>,
        k_rope: Option<&DeviceRope>,
    ) -> Result<CudaTensor> {
        self.forward_kernel(x, context, q_rope, k_rope, VideoAttnKernel::Dense)
    }

    /// Same as [`Self::forward`], but video self-attention can take the Sol or
    /// PISA kernel. Off / dense layers keep the SDPA path byte-identical.
    pub fn forward_kernel(
        &self,
        x: &CudaTensor,
        context: Option<&CudaTensor>,
        q_rope: Option<&DeviceRope>,
        k_rope: Option<&DeviceRope>,
        kernel: VideoAttnKernel,
    ) -> Result<CudaTensor> {
        self.attend(Cow::Borrowed(x), context, q_rope, k_rope, kernel)
    }

    /// [`Self::forward_kernel`] that consumes its input: `x` is freed as
    /// soon as the projections have read it, before the attention runs.
    pub fn forward_kernel_owned(
        &self,
        x: CudaTensor,
        context: Option<&CudaTensor>,
        q_rope: Option<&DeviceRope>,
        k_rope: Option<&DeviceRope>,
        kernel: VideoAttnKernel,
    ) -> Result<CudaTensor> {
        self.attend(Cow::Owned(x), context, q_rope, k_rope, kernel)
    }

    /// [`Self::forward`] that consumes its input.
    pub fn forward_owned(
        &self,
        x: CudaTensor,
        context: Option<&CudaTensor>,
        q_rope: Option<&DeviceRope>,
        k_rope: Option<&DeviceRope>,
    ) -> Result<CudaTensor> {
        self.forward_kernel_owned(x, context, q_rope, k_rope, VideoAttnKernel::Dense)
    }

    fn attend(
        &self,
        x: Cow<'_, CudaTensor>,
        context: Option<&CudaTensor>,
        q_rope: Option<&DeviceRope>,
        k_rope: Option<&DeviceRope>,
        kernel: VideoAttnKernel,
    ) -> Result<CudaTensor> {
        let (heads, d) = (self.dims.heads, self.dims.head_dim);
        let gate_logits = match &self.to_gate_logits {
            Some(l) => Some(l.forward(&x)?),
            None => None,
        };
        // Each step consumes its input (`then`), so at most one stage of
        // q / k / v exists at a time. A method chain would keep every
        // temporary to the end of its statement, and a shadowed binding to
        // the end of the function: at 130k tokens each is 2 GiB.
        // q and k: norm across heads, head split and rotary. Fused into one
        // kernel with bf16 activations (super::fuse, bit-identical), else the
        // three ops in turn.
        let k_table = q_rope.map(|r| k_rope.unwrap_or(r));
        // Each stage consumes the previous one, as above.
        let norm_split_rope = |t: CudaTensor, w: &CudaTensor, rope: Option<&DeviceRope>| {
            if let Some(o) = super::fuse::qk_norm_rope(&t, w, self.eps, heads, d, rope)? {
                return Ok(o);
            }
            let t = then(t, |t| t.rms_norm(w, self.eps))?;
            let t = then(t, |t| t.split_heads_bhsd(0, heads, d))?;
            match rope {
                Some(r) => then(t, |t| r.apply(t)),
                None => Ok(t),
            }
        };
        let q = norm_split_rope(self.to_q.forward(&x)?, &self.norm_q, q_rope)?;
        let ctx = context.unwrap_or(&x);
        let k = norm_split_rope(self.to_k.forward(ctx)?, &self.norm_k, k_table)?;
        let v = then(self.to_v.forward(ctx)?, |t| t.split_heads_bhsd(0, heads, d))?;
        // An owned input is done with: free it before the attention runs.
        drop(x);
        let (k, v) = crate::wan::nvfp4::maybe_kv(k, v)?;
        let scale = Some((d as f32).powf(-0.5));
        let kernel = if context.is_some() {
            VideoAttnKernel::Dense
        } else {
            kernel
        };
        let out = match kernel {
            VideoAttnKernel::Dense => scaled_dot_product_attention(&q, &k, &v, scale)?,
            VideoAttnKernel::Sol { tau } => {
                crate::sol_attn::sol_attn(&q, &k, &v, tau, scale, None, 0)?
            }
            VideoAttnKernel::Pisa { sparsity } => {
                crate::pisa_attn::pisa_attn(&q, &k, &v, sparsity, scale)?
            }
        };
        // q/k/v are dead once the scores are consumed; free them before the
        // gate, head merge and output projection allocate theirs.
        drop((q, k, v));
        let merged = match gate_logits {
            Some(logits) => match super::fuse::gate_merge(&out, &logits)? {
                Some(m) => m,
                None => {
                    let gated = then(out, |o| self.apply_head_gates(o, &logits))?;
                    then(gated, |g| g.merge_heads())?
                }
            },
            None => then(out, |o| o.merge_heads())?,
        };
        then(merged, |m| self.to_out.forward(m))
    }

    pub(crate) fn for_each_linear_mut(
        &mut self,
        f: &mut dyn FnMut(&mut Linear) -> Result<()>,
    ) -> Result<()> {
        f(&mut self.to_q)?;
        f(&mut self.to_k)?;
        f(&mut self.to_v)?;
        f(&mut self.to_out)?;
        if let Some(g) = self.to_gate_logits.as_mut() {
            f(g)?;
        }
        Ok(())
    }
}

/// `f(&t)`, then `t` is freed: the value-consuming step of a pipeline of
/// tensor ops, so an input does not outlive the op that read it.
fn then<R>(t: CudaTensor, f: impl FnOnce(&CudaTensor) -> Result<R>) -> Result<R> {
    f(&t)
}

/// Video self-attention kernel after QKV + RoPE.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum VideoAttnKernel {
    Dense,
    Sol { tau: f64 },
    Pisa { sparsity: f64 },
}

/// diffusers `FeedForward(dim, activation_fn="gelu-approximate")`:
/// `Linear(d, 4d)` → tanh-GELU → `Linear(4d, d)`, keys `net.0.proj` / `net.2`.
#[derive(Clone)]
pub struct FeedForward {
    up: Linear,
    down: Linear,
    /// sol-engine `nvfp4_ffn.py`: both projections on NVFP4 tensor cores
    /// ([`FeedForward::load_scoped`]); `up` / `down` then hold only their
    /// biases.
    #[cfg(feature = "cuda")]
    nvfp4: Option<std::sync::Arc<Nvfp4Ffn>>,
}

/// The FFN's two NVFP4 linears. The GELU runs inside the down projection's
/// activation quantizer, so its bf16 output is never written.
#[cfg(feature = "cuda")]
struct Nvfp4Ffn {
    up: crate::wan::nvfp4_linear::Nvfp4Linear,
    down: crate::wan::nvfp4_linear::Nvfp4Linear,
}

/// The video FFN's NVFP4 W4A4 (`FASTVIDEO_NVFP4`, the `nvfp4` technique):
/// quantize the two loaded bf16 weights with the TE `NVFP4BlockScaling`
/// rule and drop the bf16 copies. `None` (bf16 stays) without NVFP4 tensor
/// cores, for a weight that is not a plain device bf16 matrix, or for a
/// shape the GEMM cannot take: sol-engine's bf16 fallback.
#[cfg(feature = "cuda")]
fn quantize_ffn(
    up: &mut Linear,
    down: &mut Linear,
    rule: fastvideo_models::nvfp4::ScaleRule,
) -> Result<Option<Nvfp4Ffn>> {
    use crate::wan::nvfp4_linear::{tensor_cores, Nvfp4Linear};
    use crate::wan::quant::ptr;
    static SAID: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    let why = if rule == fastvideo_models::nvfp4::ScaleRule::Mse {
        Some("the FourOverSix mse rule has no tensor-core form; use static_6".to_string())
    } else if !tensor_cores() {
        Some("no NVFP4 tensor cores (sm_100 / sm_120 needed)".to_string())
    } else if !up.in_dim().is_multiple_of(64)
        || !down.in_dim().is_multiple_of(64)
        || !up.out_dim().is_multiple_of(16)
        || !down.out_dim().is_multiple_of(16)
    {
        Some(format!(
            "shape {}x{} / {}x{} (k % 64, n % 16)",
            up.out_dim(),
            up.in_dim(),
            down.out_dim(),
            down.in_dim()
        ))
    } else {
        None
    };
    let (wu, wd) = match (why, up.weight_bf16_shared(), down.weight_bf16_shared()) {
        (None, Some(wu), Some(wd)) => (wu, wd),
        (why, _, _) => {
            crate::wan::log::info_once(
                &SAID,
                format_args!(
                    "ltx2 nvfp4: video FFN stays bf16 ({})",
                    why.unwrap_or_else(|| "weights are not device bf16".into())
                ),
            );
            return Ok(None);
        }
    };
    type Bias = Option<std::sync::Arc<cudarc::driver::CudaSlice<half::bf16>>>;
    let bias = |l: &Linear| -> Result<Bias> {
        match &l.bias {
            Some(b) => b.dev_bf16(),
            None => Ok(None),
        }
    };
    let q = Nvfp4Ffn {
        up: Nvfp4Linear::from_bf16(ptr(&wu), up.out_dim(), up.in_dim(), bias(up)?, rule)?,
        down: Nvfp4Linear::from_bf16(ptr(&wd), down.out_dim(), down.in_dim(), bias(down)?, rule)?,
    };
    drop((wu, wd));
    up.release_weight_bf16();
    down.release_weight_bf16();
    crate::wan::log::info_once(
        &SAID,
        format_args!(
            "ltx2 nvfp4: video FFN on NVFP4 tensor cores (cuBLASLt VEC16_UE4M3, TE {} rule, RHT / SR off; {:.1} MiB per block vs {:.1} MiB bf16)",
            rule.as_str(),
            (q.up.held_bytes() + q.down.held_bytes()) as f64 / 1048576.0,
            ((up.out_dim() * up.in_dim() + down.out_dim() * down.in_dim()) * 2) as f64 / 1048576.0
        ),
    );
    Ok(Some(q))
}

impl FeedForward {
    pub fn load(
        map: &WeightMap,
        keys: &Keys,
        prefix: &str,
        dim: usize,
        inner: usize,
        has_bias: bool,
    ) -> Result<Self> {
        Self::load_scoped(map, keys, prefix, dim, inner, has_bias, None)
    }

    /// [`Self::load`], with `nvfp4` the rule for NVFP4 W4A4 on both
    /// projections (the LTX video FFN under `FASTVIDEO_NVFP4`).
    pub fn load_scoped(
        map: &WeightMap,
        keys: &Keys,
        prefix: &str,
        dim: usize,
        inner: usize,
        has_bias: bool,
        nvfp4: Option<fastvideo_models::nvfp4::ScaleRule>,
    ) -> Result<Self> {
        #[allow(unused_mut)]
        let mut up = Linear::load(
            map,
            &keys.key(&format!("{prefix}.net.0.proj")),
            dim,
            inner,
            has_bias,
        )?;
        #[allow(unused_mut)]
        let mut down = Linear::load(
            map,
            &keys.key(&format!("{prefix}.net.2")),
            inner,
            dim,
            has_bias,
        )?;
        #[cfg(feature = "cuda")]
        let nvfp4 = match nvfp4 {
            Some(rule) => quantize_ffn(&mut up, &mut down, rule)?.map(std::sync::Arc::new),
            None => None,
        };
        #[cfg(not(feature = "cuda"))]
        let _ = nvfp4;
        Ok(Self {
            up,
            down,
            #[cfg(feature = "cuda")]
            nvfp4,
        })
    }

    /// Whether both projections run on NVFP4 tensor cores.
    pub fn is_nvfp4(&self) -> bool {
        #[cfg(feature = "cuda")]
        {
            self.nvfp4.is_some()
        }
        #[cfg(not(feature = "cuda"))]
        {
            false
        }
    }

    /// NVFP4 forward of `x` (`[.., tokens, dim]`, device bf16): the up GEMM
    /// (bias in the epilogue), then the down projection quantizes
    /// `gelu_tanh` of its output. `None` when `x` is not a device tensor.
    #[cfg(feature = "cuda")]
    fn forward_nvfp4(&self, q: &Nvfp4Ffn, x: &CudaTensor) -> Result<Option<CudaTensor>> {
        use crate::wan::quant::ptr;
        let Some(&k) = x.shape.last() else {
            return Ok(None);
        };
        let Some(x16) = x.dev_bf16()? else {
            return Ok(None);
        };
        let m = x.numel() / k.max(1);
        crate::wan::evalstats::quant_call("nvfp4_cublaslt");
        let (h, bias_in) = q.up.forward_bf16(ptr(&x16), m, false)?;
        drop(x16);
        // A bias-free FFN (LTX-2.5 `ff_bias = false`) or one whose bias the
        // GEMM epilogue already added uses the GEMM output as is.
        let h = match (&self.up.bias, bias_in) {
            (Some(b), false) => CudaTensor::from_device_slice_bf16(h, vec![m, q.up.n])?
                .add(&b.quantize_bf16()?)?
                .dev_bf16()?
                .ok_or_else(|| msg("nvfp4 ffn: up output not on the device"))?,
            _ => std::sync::Arc::new(h),
        };
        crate::wan::evalstats::quant_call("nvfp4_cublaslt");
        let (y, bias_in) = q.down.forward_bf16(ptr(&h), m, true)?;
        drop(h);
        let mut shape = x.shape.clone();
        *shape.last_mut().unwrap() = q.down.n;
        let mut out = CudaTensor::from_device_slice_bf16(y, shape)?;
        if !bias_in {
            if let Some(b) = &self.down.bias {
                out = out.add(&b.quantize_bf16()?)?;
            }
        }
        Ok(Some(out))
    }

    /// `Linear → tanh-GELU → Linear` on `[.., tokens, dim]`. From
    /// [`FeedForwardChunking::RTX5090`]'s threshold on (65 536 tokens: the
    /// stage-2 video stream at 4K) the token axis runs in 16 384-row pieces,
    /// concatenated back — `memory.py`'s `torch.split` / `torch.cat`. Rows are
    /// independent, so the values are the same; the float32 `[tokens, 4·dim]`
    /// intermediate (8 GiB at 130k tokens) is not.
    pub fn forward(&self, x: &CudaTensor) -> Result<CudaTensor> {
        self.forward_chunked(x, FeedForwardChunking::RTX5090)
    }

    pub fn forward_chunked(
        &self,
        x: &CudaTensor,
        chunking: FeedForwardChunking,
    ) -> Result<CudaTensor> {
        let axis = x.rank().checked_sub(2).ok_or_else(|| {
            msg(format!(
                "ltx2 feed-forward expects [.., tokens, dim], got {:?}",
                x.shape
            ))
        })?;
        let spans = chunking.spans(x.shape[axis]);
        crate::wan::evalstats::ffn(spans.len().max(1));
        if spans.len() <= 1 {
            return self.forward_whole(x);
        }
        let parts = spans
            .iter()
            .map(|&(start, len)| self.forward_whole(&x.narrow(axis, start, len)?))
            .collect::<Result<Vec<_>>>()?;
        CudaTensor::cat(&parts.iter().collect::<Vec<_>>(), axis)
    }

    /// [`Self::forward`] that consumes its input: when the token axis is
    /// chunked, `x` is freed before the pieces are concatenated, so the peak
    /// holds the pieces and their concatenation but not the input as well.
    pub fn forward_owned(&self, x: CudaTensor) -> Result<CudaTensor> {
        let axis = x.rank().checked_sub(2).ok_or_else(|| {
            msg(format!(
                "ltx2 feed-forward expects [.., tokens, dim], got {:?}",
                x.shape
            ))
        })?;
        let spans = FeedForwardChunking::RTX5090.spans(x.shape[axis]);
        crate::wan::evalstats::ffn(spans.len().max(1));
        if spans.len() <= 1 {
            if self.is_nvfp4() {
                return self.forward_whole(&x);
            }
            let up = then(x, |x| self.up.forward_gelu(x))?;
            return self.down.forward(&up);
        }
        let parts = spans
            .iter()
            .map(|&(start, len)| self.forward_whole(&x.narrow(axis, start, len)?))
            .collect::<Result<Vec<_>>>()?;
        drop(x);
        CudaTensor::cat(&parts.iter().collect::<Vec<_>>(), axis)
    }

    fn forward_whole(&self, x: &CudaTensor) -> Result<CudaTensor> {
        #[cfg(feature = "cuda")]
        if let Some(q) = &self.nvfp4 {
            return self.forward_nvfp4(q, x)?.ok_or_else(|| {
                msg(format!(
                    "ltx2 nvfp4 ffn: input {:?} is not a device tensor",
                    x.shape
                ))
            });
        }
        self.down.forward(&self.up.forward_gelu(x)?)
    }

    pub(crate) fn for_each_linear_mut(
        &mut self,
        f: &mut dyn FnMut(&mut Linear) -> Result<()>,
    ) -> Result<()> {
        // NVFP4 projections hold their weights themselves (resident).
        if self.is_nvfp4() {
            return Ok(());
        }
        f(&mut self.up)?;
        f(&mut self.down)
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::ltx2::keys::Layout;

    /// Deterministic weights by key: norm weights near 1, the rest small and
    /// signed so activations stay tame through several layers.
    pub(crate) fn weights() -> WeightMap {
        WeightMap::generated(|key, shape| {
            let seed = key
                .bytes()
                .fold(11u32, |a, b| a.wrapping_mul(31).wrapping_add(u32::from(b)));
            let n: usize = shape.iter().product();
            (0..n)
                .map(|i| {
                    let v = (seed.wrapping_add(i as u32).wrapping_mul(2_654_435_761) >> 8) as f32
                        / (1u32 << 24) as f32;
                    if key.contains("norm") {
                        0.5 + v
                    } else {
                        (v - 0.5) * 0.6
                    }
                })
                .collect()
        })
    }

    pub(crate) fn get(map: &WeightMap, key: &str, shape: &[usize]) -> Vec<f32> {
        cuda_tensor_shaped(map, key, shape)
            .unwrap()
            .host_cow()
            .unwrap()
            .into_owned()
    }

    /// `y = W x + b`, `W` row-major `[o, i]`.
    pub(crate) fn linear(x: &[f32], w: &[f32], b: &[f32]) -> Vec<f32> {
        b.iter()
            .enumerate()
            .map(|(r, b)| {
                b + x
                    .iter()
                    .enumerate()
                    .map(|(c, a)| a * w[r * x.len() + c])
                    .sum::<f32>()
            })
            .collect()
    }

    pub(crate) fn rms(x: &[f32], w: Option<&[f32]>, eps: f32) -> Vec<f32> {
        let ms = x.iter().map(|a| a * a).sum::<f32>() / x.len() as f32;
        x.iter()
            .enumerate()
            .map(|(i, a)| a / (ms + eps).sqrt() * w.map_or(1.0, |w| w[i]))
            .collect()
    }

    /// The reference attention written as loops over tokens, heads and pairs:
    /// full-width QK norm, per-head half-width rotary tables, plain softmax.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn attention_reference(
        map: &WeightMap,
        prefix: &str,
        dims: AttentionDims,
        x: &[Vec<f32>],
        ctx: &[Vec<f32>],
        q_rope: Option<&SplitRope>,
        k_rope: Option<&SplitRope>,
        gated: bool,
    ) -> Vec<Vec<f32>> {
        let (inner, h, d) = (dims.inner(), dims.heads, dims.head_dim);
        let w = |n: &str, o: usize, i: usize| {
            (
                get(map, &format!("{prefix}.{n}.weight"), &[o, i]),
                get(map, &format!("{prefix}.{n}.bias"), &[o]),
            )
        };
        let (wq, wk, wv, wo) = (
            w("to_q", inner, dims.query_dim),
            w("to_k", inner, dims.context_dim),
            w("to_v", inner, dims.context_dim),
            w("to_out.0", dims.query_dim, inner),
        );
        let gate_w = gated.then(|| w("to_gate_logits", dims.heads, dims.query_dim));
        let (nq, nk) = (
            get(map, &format!("{prefix}.norm_q.weight"), &[inner]),
            get(map, &format!("{prefix}.norm_k.weight"), &[inner]),
        );
        let sigmoid = |v: f32| 1.0 / (1.0 + (-v).exp());
        let rotate = |v: &[f32], rope: Option<&SplitRope>, tok: usize| -> Vec<f32> {
            let Some(r) = rope else { return v.to_vec() };
            let mut out = v.to_vec();
            for head in 0..h {
                for j in 0..r.half {
                    let at = (head * r.tokens + tok) * r.half + j;
                    let (c, s) = (r.cos[at], r.sin[at]);
                    let (a, b) = (v[head * d + j], v[head * d + r.half + j]);
                    out[head * d + j] = a * c - b * s;
                    out[head * d + r.half + j] = b * c + a * s;
                }
            }
            out
        };
        let q: Vec<Vec<f32>> = x
            .iter()
            .enumerate()
            .map(|(t, v)| rotate(&rms(&linear(v, &wq.0, &wq.1), Some(&nq), 1e-6), q_rope, t))
            .collect();
        let k: Vec<Vec<f32>> = ctx
            .iter()
            .enumerate()
            .map(|(t, v)| {
                rotate(
                    &rms(&linear(v, &wk.0, &wk.1), Some(&nk), 1e-6),
                    if q_rope.is_some() {
                        k_rope.or(q_rope)
                    } else {
                        None
                    },
                    t,
                )
            })
            .collect();
        let v: Vec<Vec<f32>> = ctx.iter().map(|c| linear(c, &wv.0, &wv.1)).collect();
        q.iter()
            .enumerate()
            .map(|(tok, qi)| {
                let mut merged = vec![0f32; inner];
                let gates: Vec<f32> = gate_w
                    .as_ref()
                    .map(|(gw, gb)| {
                        let logits = linear(&x[tok], gw, gb);
                        logits.iter().map(|l| 2.0 * sigmoid(*l)).collect()
                    })
                    .unwrap_or_else(|| vec![1.0; h]);
                for head in 0..h {
                    let span = head * d..(head + 1) * d;
                    let scores: Vec<f32> = k
                        .iter()
                        .map(|kj| {
                            qi[span.clone()]
                                .iter()
                                .zip(&kj[span.clone()])
                                .map(|(a, b)| a * b)
                                .sum::<f32>()
                                / (d as f32).sqrt()
                        })
                        .collect();
                    let mx = scores.iter().copied().fold(f32::MIN, f32::max);
                    let z: f32 = scores.iter().map(|s| (s - mx).exp()).sum();
                    let g = gates[head];
                    for (j, s) in scores.iter().enumerate() {
                        let p = (s - mx).exp() / z;
                        for c in 0..d {
                            merged[head * d + c] += p * v[j][head * d + c] * g;
                        }
                    }
                }
                linear(&merged, &wo.0, &wo.1)
            })
            .collect()
    }

    pub(crate) fn rows(t: &CudaTensor, width: usize) -> Vec<Vec<f32>> {
        t.host_cow()
            .unwrap()
            .chunks_exact(width)
            .map(<[f32]>::to_vec)
            .collect()
    }

    pub(crate) fn tokens(n: usize, width: usize, k: f32) -> Vec<Vec<f32>> {
        (0..n)
            .map(|t| {
                (0..width)
                    .map(|c| ((t * width + c) as f32 * k).sin())
                    .collect()
            })
            .collect()
    }

    pub(crate) fn tensor(rows: &[Vec<f32>]) -> CudaTensor {
        CudaTensor::from_vec(rows.concat(), vec![1, rows.len(), rows[0].len()]).unwrap()
    }

    pub(crate) fn assert_close(got: &[Vec<f32>], want: &[Vec<f32>], tol: f32, what: &str) {
        assert_eq!(got.len(), want.len(), "{what}: token count");
        for (t, (g, w)) in got.iter().zip(want).enumerate() {
            assert_eq!(g.len(), w.len(), "{what}: width");
            for (c, (a, b)) in g.iter().zip(w).enumerate() {
                assert!((a - b).abs() <= tol, "{what}: token {t} ch {c}: {a} vs {b}");
            }
        }
    }

    #[test]
    fn self_attention_with_a_per_head_table_matches_a_loop_reference() {
        let dims = AttentionDims {
            query_dim: 12,
            context_dim: 12,
            heads: 3,
            head_dim: 4,
        };
        let map = weights();
        let attn = Attention::load(
            &map,
            &Keys::transformer(Layout::Diffusers),
            "blk.attn1",
            dims,
            1e-6,
            false,
        )
        .unwrap();
        let x = tokens(5, 12, 0.37);
        // Three axes over a 12-wide table: 2 freqs per axis, no pad; the three
        // heads get different slices of it.
        let fr: Vec<f32> = (0..15).map(|i| (i as f32 * 0.13).fract()).collect();
        let table = SplitRope::from_fractions(&fr, 3, 12, 3, 10_000.0);
        let rope = DeviceRope::upload(&table).unwrap();
        let got = attn.forward(&tensor(&x), None, Some(&rope), None).unwrap();
        let want = attention_reference(&map, "blk.attn1", dims, &x, &x, Some(&table), None, false);
        assert_close(&rows(&got, 12), &want, 2e-5, "self attention");
        // The table really is per head: rotating every head with head 0's
        // slice must give a different answer.
        let mut same = table.clone();
        for h in 1..3 {
            let (head0, rest) = same.cos.split_at_mut(h * 5 * 2);
            rest[..10].copy_from_slice(&head0[..10]);
            let (head0, rest) = same.sin.split_at_mut(h * 5 * 2);
            rest[..10].copy_from_slice(&head0[..10]);
        }
        let other = attn
            .forward(
                &tensor(&x),
                None,
                Some(&DeviceRope::upload(&same).unwrap()),
                None,
            )
            .unwrap();
        assert!(rows(&other, 12)
            .concat()
            .iter()
            .zip(want.concat())
            .any(|(a, b)| (a - b).abs() > 1e-3));
    }

    #[test]
    fn cross_attention_rotates_each_side_with_its_own_table_and_lengths_may_differ() {
        // Query stream 16 wide, context 8 wide, attention in the context's
        // head layout — the audio→video shape in miniature.
        let dims = AttentionDims {
            query_dim: 16,
            context_dim: 8,
            heads: 2,
            head_dim: 4,
        };
        let map = weights();
        let attn = Attention::load(
            &map,
            &Keys::transformer(Layout::Diffusers),
            "blk.a2v",
            dims,
            1e-6,
            false,
        )
        .unwrap();
        let (x, ctx) = (tokens(6, 16, 0.21), tokens(3, 8, 0.53));
        let qt = SplitRope::from_fractions(&[0.0, 0.1, 0.2, 0.3, 0.4, 0.5], 1, 8, 2, 10_000.0);
        let kt = SplitRope::from_fractions(&[0.05, 0.25, 0.45], 1, 8, 2, 10_000.0);
        let got = attn
            .forward(
                &tensor(&x),
                Some(&tensor(&ctx)),
                Some(&DeviceRope::upload(&qt).unwrap()),
                Some(&DeviceRope::upload(&kt).unwrap()),
            )
            .unwrap();
        assert_eq!(got.shape, vec![1, 6, 16]);
        let want =
            attention_reference(&map, "blk.a2v", dims, &x, &ctx, Some(&qt), Some(&kt), false);
        assert_close(&rows(&got, 16), &want, 2e-5, "a2v attention");
    }

    #[test]
    fn text_cross_attention_is_not_rotated() {
        let dims = AttentionDims {
            query_dim: 8,
            context_dim: 8,
            heads: 2,
            head_dim: 4,
        };
        let map = weights();
        let attn = Attention::load(
            &map,
            &Keys::transformer(Layout::Diffusers),
            "blk.attn2",
            dims,
            1e-6,
            false,
        )
        .unwrap();
        let (x, ctx) = (tokens(4, 8, 0.4), tokens(7, 8, 0.9));
        let got = attn
            .forward(&tensor(&x), Some(&tensor(&ctx)), None, None)
            .unwrap();
        let want = attention_reference(&map, "blk.attn2", dims, &x, &ctx, None, None, false);
        assert_close(&rows(&got, 8), &want, 2e-5, "text cross attention");
    }

    #[test]
    fn a_rope_built_for_another_length_is_refused() {
        let table = SplitRope::from_fractions(&[0.1, 0.2], 1, 8, 2, 10_000.0);
        let rope = DeviceRope::upload(&table).unwrap();
        let err = rope.apply(&CudaTensor::zeros(&[1, 2, 3, 4])).unwrap_err();
        assert!(err.to_string().contains("split rope built for"), "{err}");
    }

    #[test]
    fn gated_attention_matches_two_sigmoid_per_head_reference() {
        let dims = AttentionDims {
            query_dim: 12,
            context_dim: 12,
            heads: 3,
            head_dim: 4,
        };
        let map = weights();
        let attn = Attention::load(
            &map,
            &Keys::transformer(Layout::Diffusers),
            "blk.gattn",
            dims,
            1e-6,
            true,
        )
        .unwrap();
        let x = tokens(4, 12, 0.31);
        let got = attn.forward(&tensor(&x), None, None, None).unwrap();
        let want = attention_reference(&map, "blk.gattn", dims, &x, &x, None, None, true);
        assert_close(&rows(&got, 12), &want, 2e-5, "gated self attention");
    }

    #[test]
    fn feed_forward_loads_without_bias_when_requested() {
        let map = weights();
        let err = FeedForward::load(
            &map,
            &Keys::transformer(Layout::Diffusers),
            "blk.nobias_ff",
            6,
            24,
            false,
        );
        assert!(err.is_ok());
    }

    #[test]
    fn feed_forward_is_linear_tanh_gelu_linear() {
        let map = weights();
        let ff = FeedForward::load(
            &map,
            &Keys::transformer(Layout::Diffusers),
            "blk.ff",
            6,
            24,
            true,
        )
        .unwrap();
        let x = tokens(3, 6, 0.77);
        let got = ff.forward(&tensor(&x)).unwrap();
        let (w0, b0) = (
            get(&map, "blk.ff.net.0.proj.weight", &[24, 6]),
            get(&map, "blk.ff.net.0.proj.bias", &[24]),
        );
        let (w2, b2) = (
            get(&map, "blk.ff.net.2.weight", &[6, 24]),
            get(&map, "blk.ff.net.2.bias", &[6]),
        );
        let gelu = |v: f32| {
            0.5 * v
                * (1.0 + ((2.0 / std::f32::consts::PI).sqrt() * (v + 0.044_715 * v * v * v)).tanh())
        };
        let want: Vec<Vec<f32>> = x
            .iter()
            .map(|v| {
                linear(
                    &linear(v, &w0, &b0)
                        .iter()
                        .map(|a| gelu(*a))
                        .collect::<Vec<_>>(),
                    &w2,
                    &b2,
                )
            })
            .collect();
        assert_close(&rows(&got, 6), &want, 1e-5, "feed forward");
    }

    /// The host memory plan prices dense attention with the same score chunk
    /// the device path allocates.
    #[test]
    fn score_budget_matches_the_memory_plan() {
        assert_eq!(
            crate::wan::attn::DENSE_SCORE_BUDGET,
            fastvideo_models::ltx2::memory::DENSE_SCORE_BUDGET_ELEMS
        );
    }

    /// `memory.py`'s split: pieces of `chunk_tokens` rows from `min_tokens` on,
    /// concatenated back, give the whole forward's values.
    #[test]
    fn chunked_feed_forward_equals_the_whole_forward() {
        let map = weights();
        let ff = FeedForward::load(
            &map,
            &Keys::transformer(Layout::Diffusers),
            "blk.ff",
            6,
            24,
            false,
        )
        .unwrap();
        let x = tensor(&tokens(11, 6, 0.31));
        let whole = ff.forward_chunked(&x, FeedForwardChunking::OFF).unwrap();
        let chunking = FeedForwardChunking {
            chunk_tokens: 4,
            min_tokens: 8,
        };
        assert_eq!(chunking.spans(11), vec![(0, 4), (4, 4), (8, 3)]);
        let split = ff.forward_chunked(&x, chunking).unwrap();
        assert_eq!(split.shape, whole.shape);
        assert_eq!(split.host_cow().unwrap(), whole.host_cow().unwrap());
        // Below the threshold the module's own forward runs.
        let short = tensor(&tokens(7, 6, 0.31));
        assert_eq!(
            ff.forward_chunked(&short, chunking)
                .unwrap()
                .host_cow()
                .unwrap(),
            ff.forward_chunked(&short, FeedForwardChunking::OFF)
                .unwrap()
                .host_cow()
                .unwrap()
        );
    }
}
