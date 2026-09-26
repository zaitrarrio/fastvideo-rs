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
//! Every function has a host twin built from [`crate::wan::quant`]'s row
//! references, so a CPU run with the flag on takes the same rounding points.

use crate::wan::quant;
use crate::wan::tensor::{CudaTensor, Result, TensorDType, TensorError};

fn msg(s: impl Into<String>) -> TensorError {
    TensorError::Message(s.into())
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
    crate::wan::stats::device_expected() && rows.idx_dev.is_some() && x.is_device_fresh()
}

fn host_bf16(x: &CudaTensor) -> Result<Vec<f32>> {
    Ok(x.host_cow()?
        .iter()
        .map(|&v| quant::bf16_round(v))
        .collect())
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
    let (xh, wh, tab) = (host_bf16(x)?, w.host_cow()?, rows.tab.host_cow()?);
    let mut out = vec![0.0f32; n * dim];
    for r in 0..n {
        let o = quant::norm_mod_row(
            &xh[r * dim..(r + 1) * dim],
            &wh,
            rows.row(&tab, r, base, scale_slot),
            rows.row(&tab, r, base, shift_slot),
            eps,
        );
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
        host_bf16(res)?,
        host_bf16(branch)?,
        w.host_cow()?,
        rows.tab.host_cow()?,
    );
    let mut hidden = vec![0.0f32; n * dim];
    let mut normed = vec![0.0f32; n * dim];
    for r in 0..n {
        let span = r * dim..(r + 1) * dim;
        let (h, o) = quant::res_gate_norm_mod_row(
            &rh[span.clone()],
            &bh[span.clone()],
            rows.row(&tab, r, base, gate_slot),
            &wh,
            rows.row(&tab, r, base, scale_slot),
            rows.row(&tab, r, base, shift_slot),
            eps,
        );
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
    let (rh, bh, tab) = (host_bf16(res)?, host_bf16(branch)?, rows.tab.host_cow()?);
    let out: Vec<f32> = (0..rh.len())
        .map(|i| {
            let (r, j) = (i / dim, i % dim);
            quant::gate_residual_eager(rh[i], rows.row(&tab, r, base, gate_slot)[j], bh[i])
        })
        .collect();
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
    if crate::wan::stats::device_expected() && packed.is_device_fresh() {
        return dev::qk_norm_rope(packed, w, rope, batch, seq, heads, d, width, col_off, eps);
    }
    let x = host_bf16(packed)?;
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
                let o = quant::qk_norm_rope_row(src, &wh, rot, eps);
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

/// Step 3 into MXFP8 for the down projection (device only).
#[cfg(feature = "cuda")]
pub fn swiglu_mx(h: &CudaTensor) -> Result<Option<quant::MxAct>> {
    dev::swiglu_mx(h)
}

// `launch!` names `super::stats` / `super::device` from its call site.
#[cfg(feature = "cuda")]
use crate::wan::{device, stats};

#[cfg(feature = "cuda")]
const _: () = assert!(
    quant::NORM_THREADS == crate::wan::kernels::ROW_BLOCK_THREADS as usize,
    "the host norm twin mirrors the kernel's block size"
);

#[cfg(feature = "cuda")]
mod dev {
    use super::*;
    use crate::wan::act16::{operand, OutBuf};
    use crate::wan::device::global_device;
    use crate::wan::kernels::{cfg_n, cfg_rows, launch};
    use crate::wan::quant::{ptr, ptr_mut, MxAct};

    fn err(e: impl std::fmt::Display) -> TensorError {
        TensorError::Message(e.to_string())
    }

    fn table_ptr(rows: &AdaRows) -> Result<(crate::wan::act16::Operand<'_>, u64)> {
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
        let (b, gs, n_i, dim_i, rp) = (base as i64, gate_slot as i32, n as i64, dim as i64, 1i32);
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
        let mut out = OutBuf::new(n, true)?;
        let (op, o16) = (out.ptr(), out.is16());
        let threads = d.next_multiple_of(32) as u32;
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: ((batch * seq * heads).max(1) as u32, 1, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: (threads / 32) * 4,
        };
        let v = [batch, seq, heads, d, r, width, col_off].map(|u| u as i32);
        launch!(dev.stream, &dev.kernels.h3_qk_norm_rope, cfg;
            &xo.ptr, &xo.is16, &wo.ptr, &co.ptr, &so.ptr, &use_rope, &op, &o16,
            &v[0], &v[1], &v[2], &v[3], &v[4], &v[5], &v[6], &eps)
        .map_err(err)?;
        out.into_tensor(vec![batch, heads, seq, d])
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
