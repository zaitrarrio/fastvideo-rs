//! The H3 block's elementwise chain under `FASTVIDEO_BF16_ACT`, fused the way
//! Sol-H3 runs it (`h3_runtime/fusion_install.py` + `fusions.py`; the MXFP8
//! producers of `mxfp8.py` share the same rounding points):
//!
//! 1. `normed = bf16(rms(x) * w * (1 + scale[idx]) + shift[idx])` — one f32
//!    chain, one rounding (`fused_rmsnorm_modulate`);
//! 2. `hidden = bf16(res + gate[idx] * attn)` (one FMA) and the next half's
//!    `normed` from the *unrounded* f32 hidden
//!    (`fused_residual_gate_rmsnorm_modulate`);
//! 3. `ff = bf16(value * (gate * sigmoid(gate)))` (`fused_swiglu`);
//! 4. per-head q/k RMSNorm + partial RoPE with f32 cos/sin, one rounding
//!    (`fused_qknorm_rope`);
//! 5. the last residual stays eager: `bf16(hidden + bf16(gate[idx] * ff))`.
//!
//! Norm statistics accumulate in f32. With an MXFP8 consumer, steps 1-3 write
//! the block-scaled activation directly (the bf16 value is quantized, never
//! stored). The AdaLN table is f32 `[T, 6, H]` holding the bf16 projection
//! outputs (and `1 + scale` formed in f32, as the kernels do); row `r` reads
//! table row `idx[r] + base`, `base = 3 * block`.
//!
//! The rounding points of 1, 2, 4 and 5 are the `H3_*` policies of
//! [`crate::rt::fuse::policy`] (plan B4b, `docs/adr/0022-block-fusion.md`).
//! Every function has a host twin, [`crate::rt::fuse::host`] under the same
//! policy, so a CPU run with the flag on takes the same rounding points.

use crate::rt::fuse::host::{self, HostError, RowOperands};
use crate::rt::fuse::policy::{
    H3_GATE_RESIDUAL, H3_NORM_MOD, H3_QK_NORM_ROPE, H3_RES_GATE_NORM_MOD,
};
use crate::rt::fuse::{self, Switch};
#[cfg(any(feature = "cuda", test))]
use crate::rt::quant;
use crate::rt::tensor::{CudaTensor, Result, TensorDType, TensorError};

fn msg(s: impl Into<String>) -> TensorError {
    TensorError::Message(s.into())
}

fn host_err(e: HostError) -> TensorError {
    msg(e.to_string())
}

/// One step's AdaLN rows for the fused kernels.
pub struct AdaRows {
    /// f32 `[T, 6, hidden]` (ladder rows of every block, then the keyframe
    /// table), device-resident on GPU runs.
    pub tab: CudaTensor,
    pub hidden: usize,
    /// Per sequence row: `src * blocks * 3 + modality`.
    pub idx: std::sync::Arc<Vec<u32>>,
    #[cfg(feature = "cuda")]
    pub idx_dev: Option<std::sync::Arc<cudarc::driver::CudaSlice<u32>>>,
}

impl AdaRows {
    fn row<'a>(&self, tab: &'a [f32], r: usize, base: usize, slot: usize) -> &'a [f32] {
        let t = (self.idx[r] as usize + base) * 6 + slot;
        &tab[t * self.hidden..(t + 1) * self.hidden]
    }
}

/// A normalized activation: bf16, or (device) already MXFP8 for its linear.
pub enum NormOut {
    T(CudaTensor),
    #[cfg(feature = "cuda")]
    Mx(quant::MxAct),
}

#[cfg(feature = "cuda")]
fn device_ok(x: &CudaTensor, rows: &AdaRows) -> bool {
    crate::rt::stats::device_expected() && rows.idx_dev.is_some() && x.is_device_fresh()
}

/// Step 1. `mx` asks for an MXFP8 output (device only; the host returns bf16,
/// which a quantized linear's host emulation then quantizes identically).
#[allow(clippy::too_many_arguments)]
pub fn norm_mod(
    x: &CudaTensor,
    w: &CudaTensor,
    rows: &AdaRows,
    base: usize,
    scale_slot: usize,
    shift_slot: usize,
    eps: f32,
    mx: bool,
) -> Result<NormOut> {
    let dim = rows.hidden;
    let n = x.numel() / dim;
    #[cfg(feature = "cuda")]
    if device_ok(x, rows) {
        return dev::norm_mod(x, w, rows, base, scale_slot, shift_slot, eps, mx, n, dim);
    }
    let _ = mx;
    let (xh, wh, tab) = (x.host_cow()?, w.host_cow()?, rows.tab.host_cow()?);
    let mut out = vec![0.0f32; n * dim];
    for r in 0..n {
        let o = host::res_norm_row(
            &H3_NORM_MOD,
            &RowOperands {
                x: &xh[r * dim..(r + 1) * dim],
                w: Some(&wh),
                scale: Some(rows.row(&tab, r, base, scale_slot)),
                shift: Some(rows.row(&tab, r, base, shift_slot)),
                eps,
                ..RowOperands::default()
            },
        )
        .map_err(host_err)?;
        let o = o
            .normed
            .ok_or_else(|| msg("h3 fused: host twin returned no norm"))?;
        out[r * dim..(r + 1) * dim].copy_from_slice(&o);
    }
    Ok(NormOut::T(CudaTensor::host_only_dtype(
        out,
        x.shape.clone(),
        TensorDType::Bf16,
    )))
}

/// Step 2: `(hidden, normed)`.
#[allow(clippy::too_many_arguments)]
pub fn res_gate_norm_mod(
    res: &CudaTensor,
    branch: &CudaTensor,
    w: &CudaTensor,
    rows: &AdaRows,
    base: usize,
    slots: (usize, usize, usize),
    eps: f32,
    mx: bool,
) -> Result<(CudaTensor, NormOut)> {
    let dim = rows.hidden;
    let n = res.numel() / dim;
    if branch.shape != res.shape {
        return Err(msg(format!(
            "h3 fused residual: {:?} vs {:?}",
            res.shape, branch.shape
        )));
    }
    #[cfg(feature = "cuda")]
    if device_ok(res, rows) {
        return dev::res_gate_norm_mod(res, branch, w, rows, base, slots, eps, mx, n, dim);
    }
    let _ = mx;
    let (gate_slot, scale_slot, shift_slot) = slots;
    let (rh, bh, wh, tab) = (
        res.host_cow()?,
        branch.host_cow()?,
        w.host_cow()?,
        rows.tab.host_cow()?,
    );
    let mut hidden = vec![0.0f32; n * dim];
    let mut normed = vec![0.0f32; n * dim];
    for r in 0..n {
        let span = r * dim..(r + 1) * dim;
        let o = host::res_norm_row(
            &H3_RES_GATE_NORM_MOD,
            &RowOperands {
                x: &rh[span.clone()],
                u: Some(&bh[span.clone()]),
                gate: Some(rows.row(&tab, r, base, gate_slot)),
                w: Some(&wh),
                scale: Some(rows.row(&tab, r, base, scale_slot)),
                shift: Some(rows.row(&tab, r, base, shift_slot)),
                eps,
                ..RowOperands::default()
            },
        )
        .map_err(host_err)?;
        let (Some(h), Some(o)) = (o.hidden, o.normed) else {
            return Err(msg("h3 fused: host twin returned no hidden / norm"));
        };
        hidden[span.clone()].copy_from_slice(&h);
        normed[span].copy_from_slice(&o);
    }
    Ok((
        CudaTensor::host_only_dtype(hidden, res.shape.clone(), TensorDType::Bf16),
        NormOut::T(CudaTensor::host_only_dtype(
            normed,
            res.shape.clone(),
            TensorDType::Bf16,
        )),
    ))
}

/// Step 5: `bf16(res + bf16(gate[idx] * branch))`.
pub fn gate_residual(
    res: &CudaTensor,
    branch: &CudaTensor,
    rows: &AdaRows,
    base: usize,
    gate_slot: usize,
) -> Result<CudaTensor> {
    let dim = rows.hidden;
    #[cfg(feature = "cuda")]
    if device_ok(res, rows) {
        return dev::gate_residual(res, branch, rows, base, gate_slot, dim);
    }
    let (rh, bh, tab) = (res.host_cow()?, branch.host_cow()?, rows.tab.host_cow()?);
    if dim == 0 || !rh.len().is_multiple_of(dim) || bh.len() != rh.len() {
        return Err(msg(format!(
            "h3 fused gate residual: {:?} and {:?} are not rows of {dim}",
            res.shape, branch.shape
        )));
    }
    let n = rh.len() / dim;
    let mut out = vec![0.0f32; rh.len()];
    for r in 0..n {
        let span = r * dim..(r + 1) * dim;
        let o = host::res_norm_row(
            &H3_GATE_RESIDUAL,
            &RowOperands {
                x: &rh[span.clone()],
                u: Some(&bh[span.clone()]),
                gate: Some(rows.row(&tab, r, base, gate_slot)),
                ..RowOperands::default()
            },
        )
        .map_err(host_err)?;
        let h = o
            .hidden
            .ok_or_else(|| msg("h3 fused: host twin returned no hidden"))?;
        out[span].copy_from_slice(&h);
    }
    Ok(CudaTensor::host_only_dtype(
        out,
        res.shape.clone(),
        TensorDType::Bf16,
    ))
}

/// Step 4: columns `[col_off, col_off + heads * d)` of the packed `[1, S, W]`
/// projection, normed per head and rotated over the first `r` channels, as
/// BHSD bf16.
#[allow(clippy::too_many_arguments)]
pub fn qk_norm_rope(
    packed: &CudaTensor,
    w: &CudaTensor,
    rope: Option<(&CudaTensor, &CudaTensor)>,
    heads: usize,
    d: usize,
    col_off: usize,
    eps: f32,
) -> Result<CudaTensor> {
    let [batch, seq, width] = packed.shape[..] else {
        return Err(msg(format!("h3 qk norm: packed {:?}", packed.shape)));
    };
    #[cfg(feature = "cuda")]
    if crate::rt::stats::device_expected() && packed.is_device_fresh() {
        return dev::qk_norm_rope(
            packed, w, rope, batch, seq, heads, d, width, col_off, eps, false,
        );
    }
    let x = packed.host_cow()?;
    let wh = w.host_cow()?;
    let tables = match rope {
        Some((c, s)) => Some((c.host_cow()?, s.host_cow()?)),
        None => None,
    };
    let r = rope.map_or(0, |(c, _)| c.shape[1]);
    let mut out = vec![0.0f32; batch * heads * seq * d];
    for b in 0..batch {
        for s in 0..seq {
            for h in 0..heads {
                let src = &x[(b * seq + s) * width + col_off + h * d..][..d];
                let rot = tables
                    .as_ref()
                    .map(|(c, sn)| (&c[s * r..(s + 1) * r], &sn[s * r..(s + 1) * r]));
                let n = host::qk_norm(&H3_QK_NORM_ROPE, src, &wh, eps).map_err(host_err)?;
                let o = host::qk_rope_head(&H3_QK_NORM_ROPE, &n, rot).map_err(host_err)?;
                out[((b * heads + h) * seq + s) * d..][..d].copy_from_slice(&o);
            }
        }
    }
    Ok(CudaTensor::host_only_dtype(
        out,
        vec![batch, heads, seq, d],
        TensorDType::Bf16,
    ))
}

/// [`qk_norm_rope`] for an f32 consumer (the VSA kernels): the same bf16
/// values, stored widened to f32 by the kernel (no bf16 tensor, no cast).
/// Device only.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub fn qk_norm_rope_f32(
    packed: &CudaTensor,
    w: &CudaTensor,
    rope: Option<(&CudaTensor, &CudaTensor)>,
    heads: usize,
    d: usize,
    col_off: usize,
    eps: f32,
) -> Result<Option<CudaTensor>> {
    let [batch, seq, width] = packed.shape[..] else {
        return Ok(None);
    };
    if !(packed.is_bf16() && packed.is_device_fresh()) {
        return Ok(None);
    }
    dev::qk_norm_rope(
        packed, w, rope, batch, seq, heads, d, width, col_off, eps, true,
    )
    .map(Some)
}

/// `split_heads_bhsd` of a bf16 projection, stored widened to f32 (exact).
#[cfg(feature = "cuda")]
pub fn split_heads_f32(
    packed: &CudaTensor,
    col_off: usize,
    heads: usize,
    d: usize,
) -> Result<Option<CudaTensor>> {
    let [batch, seq, width] = packed.shape[..] else {
        return Ok(None);
    };
    if !packed.is_bf16() || col_off + heads * d > width {
        return Ok(None);
    }
    let Some(src) = packed.device_slice_bf16() else {
        return Ok(None);
    };
    dev::split_heads_f32(src, batch, seq, heads, d, width, col_off).map(Some)
}

/// Step 3 into MXFP8 for the down projection (device only).
#[cfg(feature = "cuda")]
pub fn swiglu_mx(h: &CudaTensor) -> Result<Option<quant::MxAct>> {
    dev::swiglu_mx(h)
}

/// `FASTVIDEO_H3_FUSE` (default on): the Phase 3c fusions on top of the
/// Sol-H3 chain above — the head merge written straight into `to_out`'s
/// MXFP8 activation ([`merge_heads_mx`]) and a block's last residual fused
/// with the next block's norm ([`gate_res_norm_mod`]). Both are
/// bit-identical to the ops they replace; `=0` runs those ops. Resolved by
/// [`crate::rt::fuse::Switch::H3`] (forced, else the DiT's installed plan
/// value, else the process setting).
pub fn fuse_enabled() -> bool {
    fuse::enabled(Switch::H3)
}

/// Run `f` with [`fuse_enabled`] forced on or off (parity checks).
pub fn with_fuse<R>(on: bool, f: impl FnOnce() -> R) -> R {
    fuse::with_forced(Switch::H3, on, f)
}

/// `FASTVIDEO_H3_GLUE` (default on; needs [`fuse_enabled`]): the glue
/// fusions around the block's GEMMs and attention (docs/perf/h3-glue-fusion.md)
/// — VSA fed bf16 q/k/v/gate in place, the q/k/v/gate heads in one kernel,
/// and the NVFP4 activations written by their producers. Each writes the
/// bytes of the ops it replaces; `=0` runs those ops
/// ([`crate::rt::fuse::Switch::H3Glue`]).
pub fn glue_enabled() -> bool {
    fuse::enabled(Switch::H3Glue)
}

/// Run `f` with [`glue_enabled`] forced on or off (parity checks).
pub fn with_glue<R>(on: bool, f: impl FnOnce() -> R) -> R {
    fuse::with_forced(Switch::H3Glue, on, f)
}

/// `merge_heads(out)` as the MXFP8 activation of a linear over `heads * d`
/// columns: the bytes `mxfp8_quantize` writes for the merged bf16 tensor.
/// `out` is `[1, H, S, D]` on the device, bf16 or f32 (rounded to bf16 first,
/// as the quantized linear's bf16 view of it would be); `None` when that does
/// not hold or the fusion is off.
#[cfg(feature = "cuda")]
pub fn merge_heads_mx(out: &CudaTensor) -> Result<Option<quant::MxAct>> {
    if !fuse_enabled() {
        return Ok(None);
    }
    dev::merge_heads_mx(out)
}

/// A block's last residual (step 5, `bf16(res + bf16(gate[idx + base_g] *
/// branch))`) and the next block's step 1 on the stored result (table rows
/// `idx + base_n`) in one kernel: `(hidden, normed)`. Device only; `None`
/// when it does not apply (the caller runs [`gate_residual`] and
/// [`norm_mod`]).
#[allow(clippy::too_many_arguments)]
pub fn gate_res_norm_mod(
    res: &CudaTensor,
    branch: &CudaTensor,
    w: &CudaTensor,
    rows: &AdaRows,
    base_g: usize,
    gate_slot: usize,
    base_n: usize,
    scale_shift: (usize, usize),
    eps: f32,
    mx: bool,
) -> Result<Option<(CudaTensor, NormOut)>> {
    #[cfg(feature = "cuda")]
    if fuse_enabled()
        && device_ok(res, rows)
        && res.is_bf16()
        && res.device_slice_bf16().is_some()
        && branch.shape == res.shape
    {
        return dev::gate_res_norm_mod(
            res,
            branch,
            w,
            rows,
            base_g,
            gate_slot,
            base_n,
            scale_shift,
            eps,
            mx,
        )
        .map(Some);
    }
    let _ = (
        res,
        branch,
        w,
        rows,
        base_g,
        gate_slot,
        base_n,
        scale_shift,
        eps,
        mx,
    );
    Ok(None)
}

/// q, k, v and the VSA gate of one bf16 `[B, S, W]` projection (columns
/// `[q | k | v | gate]`, `heads * d` each) as bf16 BHSD in one launch: q and
/// k exactly as [`qk_norm_rope`], v and the gate as `split_heads_bhsd`.
/// Device only; `None` unless [`glue_enabled`], `d == 128` and the RoPE
/// width is a multiple of 8 (the caller runs the separate ops).
#[cfg(feature = "cuda")]
#[allow(clippy::type_complexity, clippy::too_many_arguments)]
pub fn qkvg_heads(
    packed: &CudaTensor,
    wq: &CudaTensor,
    wk: &CudaTensor,
    rope: Option<(&CudaTensor, &CudaTensor)>,
    heads: usize,
    d: usize,
    gate: bool,
    eps: f32,
) -> Result<Option<(CudaTensor, CudaTensor, CudaTensor, Option<CudaTensor>)>> {
    let [batch, seq, width] = packed.shape[..] else {
        return Ok(None);
    };
    let parts = if gate { 4 } else { 3 };
    let r = rope.map_or(0, |(c, _)| c.shape.get(1).copied().unwrap_or(0));
    if !glue_enabled()
        || d != 128
        || !r.is_multiple_of(8)
        || r > d
        || !width.is_multiple_of(4)
        || parts * heads * d > width
        || !packed.is_bf16()
        || !packed.is_device_fresh()
    {
        return Ok(None);
    }
    let Some(src) = packed.device_slice_bf16() else {
        return Ok(None);
    };
    dev::qkvg_heads(src, wq, wk, rope, batch, seq, heads, parts, r, width, eps).map(Some)
}

/// SwiGLU of the bf16 `[.., 2 * half]` up projection written straight as the
/// NVFP4 operand of a down projection whose activation scale is static
/// (`amax`): the bytes `quantize_bf16_operand_with` writes for the bf16
/// SwiGLU. Device only; `None` unless [`glue_enabled`] and `half % 64 == 0`.
#[cfg(feature = "cuda")]
pub fn swiglu_nvfp4(
    h: &CudaTensor,
    amax: &cudarc::driver::CudaSlice<f32>,
    rule: fastvideo_models::nvfp4::ScaleRule,
) -> Result<Option<crate::rt::nvfp4_linear::Nvfp4Operand>> {
    let last = *h.shape.last().unwrap_or(&0);
    let half = last / 2;
    if !glue_enabled()
        || half == 0
        || !half.is_multiple_of(64)
        || !h.is_bf16()
        || !h.is_device_fresh()
    {
        return Ok(None);
    }
    let Some(src) = h.device_slice_bf16() else {
        return Ok(None);
    };
    dev::swiglu_nvfp4(src, h.numel() / last, half, amax, rule)
}

/// `merge_heads(out)` (rounded to bf16 when `out` is VSA's f32) written
/// straight as `to_out`'s NVFP4 operand: the activation amax measured on the
/// merged bf16 values (or `fixed_amax`, a static scale), then the bytes
/// `quantize_bf16_operand_with` writes. `out` is `[1, H, S, D]` on the
/// device. `None` unless [`glue_enabled`] and `H * D % 64 == 0`.
#[cfg(feature = "cuda")]
pub fn merge_heads_nvfp4(
    out: &CudaTensor,
    fixed_amax: Option<&cudarc::driver::CudaSlice<f32>>,
    rule: fastvideo_models::nvfp4::ScaleRule,
) -> Result<Option<crate::rt::nvfp4_linear::Nvfp4Operand>> {
    let [1, heads, seq, d] = out.shape[..] else {
        return Ok(None);
    };
    if !glue_enabled() || !(heads * d).is_multiple_of(64) || seq == 0 {
        return Ok(None);
    }
    dev::merge_heads_nvfp4(out, heads, seq, d, fixed_amax, rule)
}

// `launch!` names `super::stats` / `super::device` from its call site.
#[cfg(feature = "cuda")]
use crate::rt::{device, stats};

#[cfg(feature = "cuda")]
const _: () = assert!(
    quant::NORM_THREADS == crate::rt::kernels::ROW_BLOCK_THREADS as usize,
    "the host norm twin mirrors the kernel's block size"
);

#[cfg(feature = "cuda")]
mod dev {
    use super::*;
    use crate::rt::act16::{operand, OutBuf};
    use crate::rt::device::global_device;
    use crate::rt::kernels::{cfg_n, cfg_rows, launch};
    use crate::rt::quant::{ptr, ptr_mut, MxAct};

    fn err(e: impl std::fmt::Display) -> TensorError {
        TensorError::Message(e.to_string())
    }

    fn table_ptr(rows: &AdaRows) -> Result<(crate::rt::act16::Operand<'_>, u64)> {
        let t = operand(&rows.tab)?.ok_or_else(|| msg("h3 fused: AdaLN table off device"))?;
        if t.is16 != 0 {
            return Err(msg("h3 fused: the AdaLN table must be f32"));
        }
        let idx = rows
            .idx_dev
            .as_ref()
            .ok_or_else(|| msg("h3 fused: row index off device"))?;
        Ok((t, ptr(idx.as_ref())))
    }

    fn weight_f32(w: &CudaTensor) -> Result<CudaTensor> {
        if w.is_bf16() {
            w.to_f32_act()
        } else {
            Ok(w.clone())
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn norm_mod(
        x: &CudaTensor,
        w: &CudaTensor,
        rows: &AdaRows,
        base: usize,
        scale_slot: usize,
        shift_slot: usize,
        eps: f32,
        mx: bool,
        n: usize,
        dim: usize,
    ) -> Result<NormOut> {
        let dev = global_device().ok_or_else(|| msg("no device"))?;
        let xo = operand(x)?.ok_or_else(|| msg("h3 fused: x off device"))?;
        let w = weight_f32(w)?;
        let wo = operand(&w)?.ok_or_else(|| msg("h3 fused: norm weight"))?;
        let (to, ip) = table_ptr(rows)?;
        let mx = mx && dim.is_multiple_of(32);
        let mut out16 = OutBuf::new(if mx { 1 } else { n * dim }, true)?;
        let mut act = if mx {
            Some(MxAct::alloc(n, dim)?)
        } else {
            None
        };
        let op = out16.ptr();
        let (qp, sp) = match act.as_mut() {
            Some(a) => (ptr_mut(&mut a.q), ptr_mut(&mut a.s)),
            None => (op, op),
        };
        let (b, ss, sh, rows_i, dim_i, mx_i) = (
            base as i64,
            scale_slot as i32,
            shift_slot as i32,
            n as i32,
            dim as i32,
            i32::from(mx),
        );
        launch!(dev.stream, &dev.kernels.h3_norm_mod, cfg_rows(n);
            &xo.ptr, &xo.is16, &wo.ptr, &to.ptr, &ip, &b, &ss, &sh, &op, &qp, &sp,
            &rows_i, &dim_i, &eps, &mx_i)
        .map_err(err)?;
        Ok(match act {
            Some(a) => NormOut::Mx(a),
            None => NormOut::T(out16.into_tensor(x.shape.clone())?),
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn res_gate_norm_mod(
        res: &CudaTensor,
        branch: &CudaTensor,
        w: &CudaTensor,
        rows: &AdaRows,
        base: usize,
        slots: (usize, usize, usize),
        eps: f32,
        mx: bool,
        n: usize,
        dim: usize,
    ) -> Result<(CudaTensor, NormOut)> {
        let dev = global_device().ok_or_else(|| msg("no device"))?;
        let ro = operand(res)?.ok_or_else(|| msg("h3 fused: residual off device"))?;
        let bo = operand(branch)?.ok_or_else(|| msg("h3 fused: branch off device"))?;
        let w = weight_f32(w)?;
        let wo = operand(&w)?.ok_or_else(|| msg("h3 fused: norm weight"))?;
        let (to, ip) = table_ptr(rows)?;
        let mx = mx && dim.is_multiple_of(32);
        let mut hidden = OutBuf::new(n * dim, true)?;
        let mut out16 = OutBuf::new(if mx { 1 } else { n * dim }, true)?;
        let mut act = if mx {
            Some(MxAct::alloc(n, dim)?)
        } else {
            None
        };
        let (hp, op) = (hidden.ptr(), out16.ptr());
        let (qp, sp) = match act.as_mut() {
            Some(a) => (ptr_mut(&mut a.q), ptr_mut(&mut a.s)),
            None => (op, op),
        };
        let (b, gs, ss, sh) = (base as i64, slots.0 as i32, slots.1 as i32, slots.2 as i32);
        let (rows_i, dim_i, mx_i) = (n as i32, dim as i32, i32::from(mx));
        launch!(dev.stream, &dev.kernels.h3_res_gate_norm_mod, cfg_rows(n);
            &ro.ptr, &ro.is16, &bo.ptr, &bo.is16, &wo.ptr, &to.ptr, &ip, &b, &gs, &ss, &sh,
            &hp, &op, &qp, &sp, &rows_i, &dim_i, &eps, &mx_i)
        .map_err(err)?;
        let hidden = hidden.into_tensor(res.shape.clone())?;
        Ok((
            hidden,
            match act {
                Some(a) => NormOut::Mx(a),
                None => NormOut::T(out16.into_tensor(res.shape.clone())?),
            },
        ))
    }

    pub(super) fn gate_residual(
        res: &CudaTensor,
        branch: &CudaTensor,
        rows: &AdaRows,
        base: usize,
        gate_slot: usize,
        dim: usize,
    ) -> Result<CudaTensor> {
        let dev = global_device().ok_or_else(|| msg("no device"))?;
        let ro = operand(res)?.ok_or_else(|| msg("h3 fused: residual off device"))?;
        let bo = operand(branch)?.ok_or_else(|| msg("h3 fused: branch off device"))?;
        let (to, ip) = table_ptr(rows)?;
        let n = res.numel();
        let mut out = OutBuf::new(n, true)?;
        let (op, o16) = (out.ptr(), out.is16());
        let rp = match H3_GATE_RESIDUAL.kernel() {
            Ok(crate::rt::fuse::Kernel::H3GateResidual { round_product }) => round_product,
            _ => {
                return Err(msg(
                    "h3 fused: the gate residual's policy selects no h3_gate_residual",
                ))
            }
        };
        let (b, gs, n_i, dim_i) = (base as i64, gate_slot as i32, n as i64, dim as i64);
        launch!(dev.stream, &dev.kernels.h3_gate_residual, cfg_n(n);
            &ro.ptr, &ro.is16, &bo.ptr, &bo.is16, &to.ptr, &ip, &b, &gs, &op, &o16, &n_i, &dim_i, &rp)
        .map_err(err)?;
        out.into_tensor(res.shape.clone())
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn qk_norm_rope(
        packed: &CudaTensor,
        w: &CudaTensor,
        rope: Option<(&CudaTensor, &CudaTensor)>,
        batch: usize,
        seq: usize,
        heads: usize,
        d: usize,
        width: usize,
        col_off: usize,
        eps: f32,
        widen: bool,
    ) -> Result<CudaTensor> {
        if d == 0 || d > 1024 || col_off + heads * d > width {
            return Err(msg(format!(
                "h3 qk norm: {heads}x{d} at {col_off} of {width}"
            )));
        }
        let dev = global_device().ok_or_else(|| msg("no device"))?;
        let xo = operand(packed)?.ok_or_else(|| msg("h3 qk norm: input off device"))?;
        let w = weight_f32(w)?;
        let wo = operand(&w)?.ok_or_else(|| msg("h3 qk norm: weight"))?;
        let (cos, sin, r, use_rope) = match rope {
            Some((c, s)) => (c.to_f32_act()?, s.to_f32_act()?, c.shape[1], 1i32),
            None => (w.clone(), w.clone(), 0, 0),
        };
        let (co, so) = (
            operand(&cos)?.ok_or_else(|| msg("rope cos"))?,
            operand(&sin)?.ok_or_else(|| msg("rope sin"))?,
        );
        let n = batch * heads * seq * d;
        let mut out = OutBuf::new(n, !widen)?;
        let (op, o16) = (out.ptr(), if widen { 2 } else { out.is16() });
        // `dpad` threads per (batch, seq, head) row, one row per block: four
        // rows per 512-thread block measured slower on RTX PRO 6000 (FastH3
        // 8-step: 0.213 s vs 0.178 s of this kernel per step).
        let dpad = d.next_multiple_of(32);
        let per_block = 1;
        let threads = (dpad * per_block) as u32;
        let rows = (batch * seq * heads).max(1);
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (rows.div_ceil(per_block) as u32, 1, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: (threads / 32) * 4,
        };
        let v = [batch, seq, heads, d, r, width, col_off, dpad].map(|u| u as i32);
        launch!(dev.stream, &dev.kernels.h3_qk_norm_rope, cfg;
            &xo.ptr, &xo.is16, &wo.ptr, &co.ptr, &so.ptr, &use_rope, &op, &o16,
            &v[0], &v[1], &v[2], &v[3], &v[4], &v[5], &v[6], &eps, &v[7])
        .map_err(err)?;
        out.into_tensor(vec![batch, heads, seq, d])
    }

    pub(super) fn split_heads_f32(
        src: &cudarc::driver::CudaSlice<half::bf16>,
        batch: usize,
        seq: usize,
        heads: usize,
        d: usize,
        width: usize,
        col_off: usize,
    ) -> Result<CudaTensor> {
        let dev = global_device().ok_or_else(|| msg("no device"))?;
        let n = batch * heads * seq * d;
        let mut out = OutBuf::new(n, false)?;
        let (op, o16) = (out.ptr(), out.is16());
        let (sp, s16) = (ptr(src), 1i32);
        if crate::rt::act16::split_rows_ok(d, width, col_off) {
            crate::rt::act16::split_rows(
                &dev, sp, s16, op, o16, batch, seq, heads, d, width, col_off,
            )?;
            return out.into_tensor(vec![batch, heads, seq, d]);
        }
        let v = [n, seq, heads, d, width, col_off].map(|u| u as i64);
        launch!(dev.stream, &dev.kernels.mx_split_heads, cfg_n(n);
            &sp, &s16, &op, &o16, &v[0], &v[1], &v[2], &v[3], &v[4], &v[5])
        .map_err(err)?;
        out.into_tensor(vec![batch, heads, seq, d])
    }

    pub(super) fn merge_heads_mx(out: &CudaTensor) -> Result<Option<MxAct>> {
        let [1, heads, seq, d] = out.shape[..] else {
            return Ok(None);
        };
        let k = heads * d;
        if k == 0 || !k.is_multiple_of(32) {
            return Ok(None);
        }
        let (op, o16) = if out.is_bf16() {
            match out.device_slice_bf16() {
                Some(o) => (ptr(o), 1i32),
                None => return Ok(None),
            }
        } else {
            match out.device_slice() {
                Some(o) => (ptr(o), 0i32),
                None => return Ok(None),
            }
        };
        let dev = global_device().ok_or_else(|| msg("no device"))?;
        let mut act = MxAct::alloc(seq, k)?;
        let (qp, sp) = (ptr_mut(&mut act.q), ptr_mut(&mut act.s));
        let (seq_i, heads_i, d_i) = (seq as i32, heads as i32, d as i32);
        launch!(dev.stream, &dev.kernels.fvf_merge_heads_mx, cfg_rows(seq);
            &op, &o16, &qp, &sp, &seq_i, &heads_i, &d_i)
        .map_err(err)?;
        Ok(Some(act))
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn gate_res_norm_mod(
        res: &CudaTensor,
        branch: &CudaTensor,
        w: &CudaTensor,
        rows: &AdaRows,
        base_g: usize,
        gate_slot: usize,
        base_n: usize,
        (scale_slot, shift_slot): (usize, usize),
        eps: f32,
        mx: bool,
    ) -> Result<(CudaTensor, NormOut)> {
        let dim = rows.hidden;
        let n = res.numel() / dim;
        let dev = global_device().ok_or_else(|| msg("no device"))?;
        let rp = ptr(res
            .device_slice_bf16()
            .ok_or_else(|| msg("h3 fused: residual off device"))?);
        let bo = operand(branch)?.ok_or_else(|| msg("h3 fused: branch off device"))?;
        let w = weight_f32(w)?;
        let wo = operand(&w)?.ok_or_else(|| msg("h3 fused: norm weight"))?;
        let (to, ip) = table_ptr(rows)?;
        let mx = mx && dim.is_multiple_of(32);
        let mut hidden = OutBuf::new(n * dim, true)?;
        let mut out16 = OutBuf::new(if mx { 1 } else { n * dim }, true)?;
        let mut act = if mx {
            Some(MxAct::alloc(n, dim)?)
        } else {
            None
        };
        let (hp, op) = (hidden.ptr(), out16.ptr());
        let (qp, sp) = match act.as_mut() {
            Some(a) => (ptr_mut(&mut a.q), ptr_mut(&mut a.s)),
            None => (op, op),
        };
        let (bg, gs, bn, ss, sh) = (
            base_g as i64,
            gate_slot as i32,
            base_n as i64,
            scale_slot as i32,
            shift_slot as i32,
        );
        let (rows_i, dim_i, mx_i) = (n as i32, dim as i32, i32::from(mx));
        launch!(dev.stream, &dev.kernels.fvf_h3_gate_res_norm_mod, cfg_rows(n);
            &rp, &bo.ptr, &bo.is16, &to.ptr, &ip, &bg, &gs, &bn, &ss, &sh, &wo.ptr,
            &hp, &op, &qp, &sp, &rows_i, &dim_i, &eps, &mx_i)
        .map_err(err)?;
        let hidden = hidden.into_tensor(res.shape.clone())?;
        Ok((
            hidden,
            match act {
                Some(a) => NormOut::Mx(a),
                None => NormOut::T(out16.into_tensor(res.shape.clone())?),
            },
        ))
    }

    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    pub(super) fn qkvg_heads(
        src: &cudarc::driver::CudaSlice<half::bf16>,
        wq: &CudaTensor,
        wk: &CudaTensor,
        rope: Option<(&CudaTensor, &CudaTensor)>,
        batch: usize,
        seq: usize,
        heads: usize,
        parts: usize,
        r: usize,
        width: usize,
        eps: f32,
    ) -> Result<(CudaTensor, CudaTensor, CudaTensor, Option<CudaTensor>)> {
        let dev = global_device().ok_or_else(|| msg("no device"))?;
        let d = 128usize;
        let (wq, wk) = (weight_f32(wq)?, weight_f32(wk)?);
        let wqo = operand(&wq)?.ok_or_else(|| msg("h3 qkvg heads: norm_q"))?;
        let wko = operand(&wk)?.ok_or_else(|| msg("h3 qkvg heads: norm_k"))?;
        let (cos, sin, use_rope) = match rope {
            Some((c, s)) => (c.to_f32_act()?, s.to_f32_act()?, 1i32),
            None => (wq.clone(), wq.clone(), 0),
        };
        let (co, so) = (
            operand(&cos)?.ok_or_else(|| msg("rope cos"))?,
            operand(&sin)?.ok_or_else(|| msg("rope sin"))?,
        );
        let n = batch * heads * seq * d;
        let mut outs: Vec<OutBuf> = (0..parts)
            .map(|_| OutBuf::new(n, true))
            .collect::<Result<_>>()?;
        let mut ptrs: Vec<u64> = outs.iter_mut().map(|o| o.ptr()).collect();
        if parts == 3 {
            ptrs.push(ptrs[2]);
        }
        let items = batch * seq * parts * heads;
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (items.div_ceil(8).max(1) as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let sp = ptr(src);
        let v = [batch, seq, heads, parts, r, width, d].map(|u| u as i32);
        launch!(dev.stream, &dev.kernels.fvf_h3_qkvg_heads, cfg;
            &sp, &wqo.ptr, &wko.ptr, &co.ptr, &so.ptr, &use_rope, &ptrs[0], &ptrs[1], &ptrs[2], &ptrs[3],
            &v[0], &v[1], &v[2], &v[3], &v[4], &v[5], &eps, &v[6])
        .map_err(err)?;
        let shape = vec![batch, heads, seq, d];
        let mut it = outs.into_iter();
        let q = it.next().unwrap().into_tensor(shape.clone())?;
        let k = it.next().unwrap().into_tensor(shape.clone())?;
        let v = it.next().unwrap().into_tensor(shape.clone())?;
        let g = it.next().map(|g| g.into_tensor(shape)).transpose()?;
        Ok((q, k, v, g))
    }

    fn rule_code(rule: fastvideo_models::nvfp4::ScaleRule) -> Option<i32> {
        match rule {
            fastvideo_models::nvfp4::ScaleRule::Static6 => Some(0),
            fastvideo_models::nvfp4::ScaleRule::Static4 => Some(1),
            _ => None,
        }
    }

    /// An empty NVFP4 operand for `rows x k` (`quantize_bf16_operand_with`'s
    /// buffers); every byte is written by the producer kernel.
    fn nvfp4_alloc(rows: usize, k: usize) -> Result<crate::rt::nvfp4_linear::Nvfp4Operand> {
        let dev = global_device().ok_or_else(|| msg("no device"))?;
        let rows_pack = rows.next_multiple_of(16);
        let rows_scale = rows.next_multiple_of(128);
        // SAFETY: the producer writes every code of rows_pack rows and every
        // scale of rows_scale rows (fv_nvfp4_pad_row for the padding).
        let packed = unsafe { dev.stream.alloc::<u8>(rows_pack * k / 2) }.map_err(err)?;
        let scales = unsafe { dev.stream.alloc::<u8>(rows_scale * (k / 16)) }.map_err(err)?;
        let amax = dev.stream.alloc_zeros::<f32>(1).map_err(err)?;
        Ok(crate::rt::nvfp4_linear::Nvfp4Operand {
            packed,
            scales,
            amax,
            rows,
            rows_pack,
            k,
        })
    }

    fn nvfp4_cfg(rows: usize) -> cudarc::driver::LaunchConfig {
        cudarc::driver::LaunchConfig {
            grid_dim: (rows.next_multiple_of(128) as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        }
    }

    pub(super) fn swiglu_nvfp4(
        src: &cudarc::driver::CudaSlice<half::bf16>,
        rows: usize,
        half: usize,
        amax: &cudarc::driver::CudaSlice<f32>,
        rule: fastvideo_models::nvfp4::ScaleRule,
    ) -> Result<Option<crate::rt::nvfp4_linear::Nvfp4Operand>> {
        let Some(code) = rule_code(rule) else {
            return Ok(None);
        };
        if amax.len() != 1 || rows == 0 {
            return Ok(None);
        }
        let dev = global_device().ok_or_else(|| msg("no device"))?;
        let mut a = nvfp4_alloc(rows, half)?;
        dev.stream.memcpy_dtod(amax, &mut a.amax).map_err(err)?;
        let (sp, ap, pp, scp) = (
            ptr(src),
            ptr(&a.amax),
            ptr_mut(&mut a.packed),
            ptr_mut(&mut a.scales),
        );
        let (rows_i, pack_i, half_i) = (rows as i32, a.rows_pack as i32, half as i32);
        launch!(dev.stream, &dev.kernels.fvf_h3_swiglu_nvfp4, nvfp4_cfg(rows);
            &sp, &ap, &pp, &scp, &rows_i, &pack_i, &half_i, &code)
        .map_err(err)?;
        Ok(Some(a))
    }

    pub(super) fn merge_heads_nvfp4(
        out: &CudaTensor,
        heads: usize,
        seq: usize,
        d: usize,
        fixed_amax: Option<&cudarc::driver::CudaSlice<f32>>,
        rule: fastvideo_models::nvfp4::ScaleRule,
    ) -> Result<Option<crate::rt::nvfp4_linear::Nvfp4Operand>> {
        let Some(code) = rule_code(rule) else {
            return Ok(None);
        };
        let n = heads * seq * d;
        let (op, o16) = if out.is_bf16() {
            match out.device_slice_bf16() {
                Some(o) => (ptr(o), 1i32),
                None => return Ok(None),
            }
        } else {
            match out.device_slice() {
                Some(o) => (ptr(o), 0i32),
                None => return Ok(None),
            }
        };
        let dev = global_device().ok_or_else(|| msg("no device"))?;
        let k = heads * d;
        let mut a = nvfp4_alloc(seq, k)?;
        let nn = n as i64;
        let amax_cfg = cudarc::driver::LaunchConfig {
            grid_dim: (n.div_ceil(256).clamp(1, 4096) as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        match fixed_amax {
            Some(f) if f.len() == 1 => dev.stream.memcpy_dtod(f, &mut a.amax).map_err(err)?,
            Some(_) => return Err(msg("h3 merge nvfp4: the fixed amax is one f32")),
            // The measured amax of the merged bf16 tensor: the same values in
            // another order, and max does not depend on the order.
            None if o16 == 1 => {
                let g = 0i32;
                launch!(dev.stream, &dev.kernels.nvfp4_amax_bf16, amax_cfg; &op, &g, &mut a.amax, &nn)
                    .map_err(err)?
            }
            None => {
                launch!(dev.stream, &dev.kernels.fvf_amax_f32_r16, amax_cfg; &op, &mut a.amax, &nn)
                    .map_err(err)?
            }
        }
        let (ap, pp, scp) = (ptr(&a.amax), ptr_mut(&mut a.packed), ptr_mut(&mut a.scales));
        let v = [seq, a.rows_pack, heads, d].map(|u| u as i32);
        launch!(dev.stream, &dev.kernels.fvf_h3_merge_heads_nvfp4, nvfp4_cfg(seq);
            &op, &o16, &ap, &pp, &scp, &v[0], &v[1], &v[2], &v[3], &code)
        .map_err(err)?;
        Ok(Some(a))
    }

    pub(super) fn swiglu_mx(h: &CudaTensor) -> Result<Option<MxAct>> {
        let last = *h.shape.last().unwrap_or(&0);
        let half = last / 2;
        if half == 0 || !half.is_multiple_of(32) || !h.is_device_fresh() {
            return Ok(None);
        }
        let dev = global_device().ok_or_else(|| msg("no device"))?;
        let xo = operand(h)?.ok_or_else(|| msg("h3 swiglu: input off device"))?;
        let rows = h.numel() / last;
        let mut act = MxAct::alloc(rows, half)?;
        let (qp, sp) = (ptr_mut(&mut act.q), ptr_mut(&mut act.s));
        let (rows_i, half_i) = (rows as i32, half as i32);
        launch!(dev.stream, &dev.kernels.h3_swiglu_mx, cfg_rows(rows);
            &xo.ptr, &xo.is16, &qp, &sp, &rows_i, &half_i)
        .map_err(err)?;
        Ok(Some(act))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rows(h: usize, t: usize, idx: Vec<u32>) -> AdaRows {
        // Table row `r`, slot `p` holds the constant `r * 10 + p`.
        let tab: Vec<f32> = (0..t * 6 * h)
            .map(|i| ((i / h) / 6 * 10 + (i / h) % 6) as f32)
            .collect();
        AdaRows {
            tab: CudaTensor::from_vec(tab, vec![t, 6, h]).unwrap(),
            hidden: h,
            idx: std::sync::Arc::new(idx),
            #[cfg(feature = "cuda")]
            idx_dev: None,
        }
    }

    #[test]
    fn rows_read_their_index_plus_the_block_base() {
        let h = 4;
        let ada = rows(h, 5, vec![0, 1, 1]);
        let res = CudaTensor::from_vec(vec![0.0; 3 * h], vec![1, 3, h]).unwrap();
        let one = CudaTensor::from_vec(vec![1.0; 3 * h], vec![1, 3, h]).unwrap();
        // base 3: rows read table rows 3, 4, 4; gate slot 5.
        let out = gate_residual(&res, &one, &ada, 3, 5).unwrap();
        let out = out.host_cow().unwrap();
        assert_eq!(out[0], 35.0);
        assert_eq!(out[h], 45.0);
        assert_eq!(out[2 * h], 45.0);
    }

    #[test]
    fn a_residual_that_is_not_whole_rows_is_an_error() {
        let ada = rows(4, 2, vec![0, 0]);
        let res = CudaTensor::from_vec(vec![0.0; 6], vec![1, 6]).unwrap();
        assert!(gate_residual(&res, &res, &ada, 0, 5).is_err());
        let short = CudaTensor::from_vec(vec![0.0; 4], vec![1, 1, 4]).unwrap();
        let two = CudaTensor::from_vec(vec![0.0; 8], vec![1, 2, 4]).unwrap();
        assert!(gate_residual(&two, &short, &ada, 0, 5).is_err());
    }

    #[test]
    fn the_last_residual_keeps_the_eager_product_rounding() {
        let h = 2;
        let mut ada = rows(h, 1, vec![0]);
        ada.tab = CudaTensor::from_vec(vec![0.3337; 6 * h], vec![1, 6, h]).unwrap();
        let (r, b) = (1.0f32, 0.001_234f32);
        let res = CudaTensor::from_vec(vec![r; h], vec![1, 1, h]).unwrap();
        let br = CudaTensor::from_vec(vec![b; h], vec![1, 1, h]).unwrap();
        let out = gate_residual(&res, &br, &ada, 0, 5).unwrap();
        assert!(out.is_bf16());
        let want = quant::bf16_round(r + quant::bf16_round(0.3337 * quant::bf16_round(b)));
        assert_eq!(out.host_cow().unwrap()[0], want);
    }
}
