//! The Wan DiT block's elementwise chain under `FASTVIDEO_BF16_ACT`, with
//! FastVideo's rounding points (`fastvideo/models/wan/transformer.py`
//! `WanTransformerBlock.forward`, `layers/layernorm.py`):
//!
//! 1. `normed = bf16(LN(x) * (1 + scale) + shift)` — [`CudaTensor::ln_adaln_e`];
//! 2. self-attention residual + `norm2` (`ScaleResidualLayerNormScaleShift`,
//!    f32 compute): `hidden = bf16(h + a * gate)` with the product and sum in
//!    f32, and `normed = bf16(LN_affine(h + a * gate))` from the **unrounded**
//!    f32 sum ([`self_residual_norm`]);
//! 3. cross-attention residual + FFN norm: `hidden = bf16(h + a)`, then
//!    `normed = bf16(bf16(LN(hidden)) * (1 + scale) + shift)` — the
//!    `FP32LayerNorm` casts back to bf16 before the f32 modulation
//!    ([`cross_residual_norm_mod`]);
//! 4. `hidden = bf16(h + ff * gate)` in f32 (`ScaleResidual`, [`gate_residual`]);
//! 5. q/k `RMSNorm` across heads: `bf16(bf16(x * rsqrt(mean(x^2) + eps)) * w)`,
//!    then RoPE in f32 with one rounding ([`qk_norm_rope`]).
//!
//! `FASTVIDEO_WAN_FUSE` (default on) runs 2 and 3 as one kernel each; `=0`
//! runs the same math as two kernels (the residual stores its value — f32 for
//! 2 — and a norm kernel reads it back). Both spell every operation with
//! explicit `_rn` intrinsics, so the fused and unfused outputs are
//! byte-identical. Every function returns `None` when the block is not on the
//! bf16 path (f32 activations), and the caller keeps its f32 chain.
//!
//! On CPU runs the host twins take the same rounding points (the LayerNorm
//! reduction order differs from the kernels', so host and device agree to
//! within a bf16 ulp, not bit for bit).

use super::fused::Rope;
use super::quant::bf16_round;
use super::tensor::{CudaTensor, Result, TensorDType, TensorError};

fn msg(s: impl Into<String>) -> TensorError {
    TensorError::Message(s.into())
}

thread_local! {
    static FUSE_OVERRIDE: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
}

/// `FASTVIDEO_WAN_FUSE` (default on).
pub fn enabled() -> bool {
    if let Some(v) = FUSE_OVERRIDE.with(|c| c.get()) {
        return v;
    }
    static FLAG: super::envflag::CachedBool = super::envflag::CachedBool::new();
    FLAG.get_or_init(|| super::envflag::bool_flag("FASTVIDEO_WAN_FUSE", true))
}

/// Run `f` with [`enabled`] forced on or off (parity checks).
pub fn with_fuse<R>(on: bool, f: impl FnOnce() -> R) -> R {
    let prev = FUSE_OVERRIDE.with(|c| c.replace(Some(on)));
    let out = f();
    FUSE_OVERRIDE.with(|c| c.set(prev));
    out
}

/// Whether the block's residual stream `h` is on the bf16 path.
pub fn active(h: &CudaTensor) -> bool {
    super::tensor::bf16_activations() && h.is_bf16()
}

fn dims(h: &CudaTensor, a: &CudaTensor, e: &CudaTensor) -> Result<(usize, usize, usize, usize)> {
    let [batch, seq, dim] = h.shape[..] else {
        return Err(msg(format!("wan fuse: residual {:?}", h.shape)));
    };
    let e_rows = e.shape.get(1).copied().unwrap_or(0);
    if a.shape != h.shape || e.shape != [batch, e_rows, dim] {
        return Err(msg(format!(
            "wan fuse: shapes {:?} {:?} {:?}",
            h.shape, a.shape, e.shape
        )));
    }
    Ok((batch, seq, dim, e_rows))
}

thread_local! {
    static HOST_TWINS: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Run `f` with every function here on its host twin, even with a device
/// live (inputs are downloaded): the reference `fv-gpucheck` compares the
/// kernels against.
pub fn with_host_twins<R>(f: impl FnOnce() -> R) -> R {
    let prev = HOST_TWINS.with(|c| c.replace(true));
    let out = f();
    HOST_TWINS.with(|c| c.set(prev));
    out
}

#[cfg(feature = "cuda")]
fn on_device(ts: &[&CudaTensor]) -> bool {
    !HOST_TWINS.with(|c| c.get())
        && super::stats::device_expected()
        && ts.iter().any(|t| t.act16_device())
}

/// Step 2: `(hidden, normed)` after self-attention (`gate_slot` of `e`,
/// `norm2` weight `w` and bias `b`).
pub fn self_residual_norm(
    h: &CudaTensor,
    attn: &CudaTensor,
    e: &CudaTensor,
    gate_slot: usize,
    w: &CudaTensor,
    b: &CudaTensor,
    eps: f32,
) -> Result<Option<(CudaTensor, CudaTensor)>> {
    if !active(h) {
        return Ok(None);
    }
    let (batch, seq, dim, e_rows) = dims(h, attn, e)?;
    if gate_slot >= e_rows || w.numel() != dim || b.numel() != dim {
        return Err(msg("wan fuse: self residual table/affine"));
    }
    #[cfg(feature = "cuda")]
    if on_device(&[h, attn]) {
        let g = dev::Geo {
            rows: batch * seq,
            seq,
            dim,
            e_rows,
        };
        return dev::res_ln(h, attn, e, gate_slot, Some((w, b)), (0, 0), 0, 0, g, eps).map(Some);
    }
    let (hh, ah, eh) = (h.host_cow()?, attn.host_cow()?, e.host_cow()?);
    let (wh, bh) = (w.host_cow()?, b.host_cow()?);
    let mut hid = vec![0.0f32; h.numel()];
    let mut out = vec![0.0f32; h.numel()];
    for r in 0..batch * seq {
        let g = &eh[((r / seq) * e_rows + gate_slot) * dim..][..dim];
        let span = r * dim..(r + 1) * dim;
        let v: Vec<f32> = hh[span.clone()]
            .iter()
            .zip(&ah[span.clone()])
            .zip(g)
            .map(|((&x, &y), &gv)| x + y * gv)
            .collect();
        let (mean, inv) = ln_stats(&v, eps);
        for j in 0..dim {
            hid[r * dim + j] = bf16_round(v[j]);
            out[r * dim + j] = bf16_round((v[j] - mean) * inv * wh[j] + bh[j]);
        }
    }
    Ok(Some((host16(hid, &h.shape), host16(out, &h.shape))))
}

/// Step 3: `(hidden, normed)` after cross-attention, the FFN's AdaLN from
/// `scale_slot` / `shift_slot` of `e`.
pub fn cross_residual_norm_mod(
    h: &CudaTensor,
    cross: &CudaTensor,
    e: &CudaTensor,
    scale_slot: usize,
    shift_slot: usize,
    eps: f32,
) -> Result<Option<(CudaTensor, CudaTensor)>> {
    if !active(h) {
        return Ok(None);
    }
    let (batch, seq, dim, e_rows) = dims(h, cross, e)?;
    if scale_slot >= e_rows || shift_slot >= e_rows {
        return Err(msg("wan fuse: cross residual table"));
    }
    #[cfg(feature = "cuda")]
    if on_device(&[h, cross]) {
        let g = dev::Geo {
            rows: batch * seq,
            seq,
            dim,
            e_rows,
        };
        return dev::res_ln(h, cross, e, 0, None, (scale_slot, shift_slot), 1, 1, g, eps).map(Some);
    }
    let (hh, ch, eh) = (h.host_cow()?, cross.host_cow()?, e.host_cow()?);
    let mut hid = vec![0.0f32; h.numel()];
    let mut out = vec![0.0f32; h.numel()];
    for r in 0..batch * seq {
        let tb = (r / seq) * e_rows;
        let sc = &eh[(tb + scale_slot) * dim..][..dim];
        let sh = &eh[(tb + shift_slot) * dim..][..dim];
        let span = r * dim..(r + 1) * dim;
        let v: Vec<f32> = hh[span.clone()]
            .iter()
            .zip(&ch[span])
            .map(|(&x, &y)| bf16_round(x + y))
            .collect();
        let (mean, inv) = ln_stats(&v, eps);
        for j in 0..dim {
            hid[r * dim + j] = v[j];
            let n = bf16_round((v[j] - mean) * inv);
            out[r * dim + j] = bf16_round(n * (1.0 + sc[j]) + sh[j]);
        }
    }
    Ok(Some((host16(hid, &h.shape), host16(out, &h.shape))))
}

/// Step 4: the block's last residual, `bf16(h + ff * gate)` in f32.
pub fn gate_residual(
    h: &CudaTensor,
    ff: &CudaTensor,
    e: &CudaTensor,
    gate_slot: usize,
) -> Result<Option<CudaTensor>> {
    if !active(h) {
        return Ok(None);
    }
    let (batch, seq, dim, e_rows) = dims(h, ff, e)?;
    if gate_slot >= e_rows {
        return Err(msg("wan fuse: gate slot"));
    }
    #[cfg(feature = "cuda")]
    if on_device(&[h, ff]) {
        let g = dev::Geo {
            rows: batch * seq,
            seq,
            dim,
            e_rows,
        };
        return dev::res_gate(h, ff, e, gate_slot, 0, false, g).map(|(hid, _)| Some(hid));
    }
    let (hh, fh, eh) = (h.host_cow()?, ff.host_cow()?, e.host_cow()?);
    let out: Vec<f32> = (0..h.numel())
        .map(|i| {
            let (r, j) = (i / dim, i % dim);
            bf16_round(hh[i] + fh[i] * eh[((r / seq) * e_rows + gate_slot) * dim + j])
        })
        .collect();
    let _ = batch;
    Ok(Some(host16(out, &h.shape)))
}

/// Step 5: columns `[col_off, col_off + heads * d)` of the projection
/// `[b, seq, width]`, RMS-normed across all heads by `weight`, rotated by
/// `rope`, as bf16 BHSD. `None` unless the projection is bf16.
pub fn qk_norm_rope(
    proj: &CudaTensor,
    col_off: usize,
    heads: usize,
    weight: &CudaTensor,
    rope: Option<Rope<'_>>,
    eps: f32,
) -> Result<Option<CudaTensor>> {
    if !active(proj) {
        return Ok(None);
    }
    let [batch, seq, width] = proj.shape[..] else {
        return Err(msg(format!("wan qk norm: projection {:?}", proj.shape)));
    };
    let proj_w = weight.numel();
    if heads == 0 || proj_w % heads != 0 || col_off + proj_w > width {
        return Err(msg("wan qk norm: weight / columns"));
    }
    let d = proj_w / heads;
    if let Some(r) = &rope {
        if r.cos.shape != [seq, d] || r.sin.shape != [seq, d] || d % 2 != 0 {
            return Err(msg(format!(
                "wan qk norm: rope {:?} for d={d}",
                r.cos.shape
            )));
        }
    }
    #[cfg(feature = "cuda")]
    if on_device(&[proj]) {
        return dev::qk_norm_rope(proj, col_off, heads, d, weight, rope, eps).map(Some);
    }
    let (x, w) = (proj.host_cow()?, weight.host_cow()?);
    let tables = match &rope {
        Some(r) => Some((r.cos.host_cow()?, r.sin.host_cow()?)),
        None => None,
    };
    let mut out = vec![0.0f32; batch * heads * seq * d];
    for b in 0..batch {
        for s in 0..seq {
            let row = &x[(b * seq + s) * width + col_off..][..proj_w];
            let ss: f32 = row.iter().map(|v| v * v).sum();
            let inv = 1.0 / (ss / proj_w as f32 + eps).sqrt();
            let n: Vec<f32> = row
                .iter()
                .zip(w.iter())
                .map(|(&v, &wv)| bf16_round(bf16_round(v * inv) * wv))
                .collect();
            for h in 0..heads {
                for p in 0..d {
                    let j = h * d + p;
                    let o = ((b * heads + h) * seq + s) * d + p;
                    out[o] = match &tables {
                        Some((c, sn)) => {
                            let even = p - (p & 1);
                            let (x1, x2) = (n[h * d + even], n[h * d + even + 1]);
                            let (cv, sv) = (c[s * d + even], sn[s * d + even + 1]);
                            bf16_round(if p & 1 == 1 {
                                x2 * cv + x1 * sv
                            } else {
                                x1 * cv - x2 * sv
                            })
                        }
                        None => n[j],
                    };
                }
            }
        }
    }
    Ok(Some(host16(out, &[batch, heads, seq, d])))
}

fn ln_stats(v: &[f32], eps: f32) -> (f32, f32) {
    let n = v.len() as f32;
    let mean = v.iter().sum::<f32>() / n;
    let var = v.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / n;
    (mean, 1.0 / (var + eps).sqrt())
}

fn host16(v: Vec<f32>, shape: &[usize]) -> CudaTensor {
    CudaTensor::host_only_dtype(v, shape.to_vec(), TensorDType::Bf16)
}

// `launch!` names `super::stats` / `super::device` from its call site.
#[cfg(feature = "cuda")]
use super::{device, stats};

#[cfg(feature = "cuda")]
pub(crate) mod dev {
    use super::*;
    use crate::wan::act16::{operand, OutBuf};
    use crate::wan::device::global_device;
    use crate::wan::kernels::{cfg_n, cfg_rows, launch};

    fn err(e: impl std::fmt::Display) -> TensorError {
        TensorError::Message(e.to_string())
    }

    #[derive(Clone, Copy)]
    pub(crate) struct Geo {
        pub rows: usize,
        pub seq: usize,
        pub dim: usize,
        pub e_rows: usize,
    }

    fn f32_param(t: &CudaTensor) -> Result<CudaTensor> {
        if t.is_bf16() {
            t.to_f32_act()
        } else {
            Ok(t.clone())
        }
    }

    /// Fused (`FASTVIDEO_WAN_FUSE`) or two-kernel residual + norm.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn res_ln(
        h: &CudaTensor,
        a: &CudaTensor,
        e: &CudaTensor,
        gate_slot: usize,
        affine: Option<(&CudaTensor, &CudaTensor)>,
        (scale_slot, shift_slot): (usize, usize),
        res_mode: i32,
        ln_mode: i32,
        g: Geo,
        eps: f32,
    ) -> Result<(CudaTensor, CudaTensor)> {
        if !super::enabled() {
            // Unfused: the residual kernel stores what the norm reads (the
            // f32 sum for res_mode 0, the bf16 hidden for res_mode 1).
            let (hid, tmp) = res_gate(h, a, e, gate_slot, res_mode, res_mode == 0, g)?;
            let src = tmp.as_ref().unwrap_or(&hid);
            let normed = ln(src, e, affine, (scale_slot, shift_slot), ln_mode, g, eps)?;
            return Ok((hid, normed));
        }
        let dev = global_device().ok_or_else(|| msg("no device"))?;
        let (ho, ao, eo) = (
            operand(h)?.ok_or_else(|| msg("wan fuse: h off device"))?,
            operand(a)?.ok_or_else(|| msg("wan fuse: branch off device"))?,
            operand(e)?.ok_or_else(|| msg("wan fuse: table off device"))?,
        );
        let params = match affine {
            Some((w, b)) => Some((f32_param(w)?, f32_param(b)?)),
            None => None,
        };
        let pops = match &params {
            Some((w, b)) => Some((
                operand(w)?.ok_or_else(|| msg("wan fuse: norm weight"))?,
                operand(b)?.ok_or_else(|| msg("wan fuse: norm bias"))?,
            )),
            None => None,
        };
        let (wp, bp) = pops
            .as_ref()
            .map_or((ho.ptr, ho.ptr), |(w, b)| (w.ptr, b.ptr));
        let n = g.rows * g.dim;
        let mut hid = OutBuf::new(n, true)?;
        let mut out = OutBuf::new(n, true)?;
        let (hp, op) = (hid.ptr(), out.ptr());
        let (gs, ss, sh, p16) = (gate_slot as i32, scale_slot as i32, shift_slot as i32, 0i32);
        let v = [g.rows, g.seq, g.dim, g.e_rows].map(|u| u as i32);
        launch!(dev.stream, &dev.kernels.wan_res_ln, cfg_rows(g.rows);
            &ho.ptr, &ho.is16, &ao.ptr, &ao.is16, &eo.ptr, &eo.is16, &gs,
            &wp, &bp, &p16, &ss, &sh, &res_mode, &ln_mode, &hp, &op,
            &v[0], &v[1], &v[2], &v[3], &eps)
        .map_err(err)?;
        Ok((
            hid.into_tensor(h.shape.clone())?,
            out.into_tensor(h.shape.clone())?,
        ))
    }

    /// Residual alone: bf16 hidden, plus the f32 value when `keep_f32`.
    pub(crate) fn res_gate(
        h: &CudaTensor,
        a: &CudaTensor,
        e: &CudaTensor,
        gate_slot: usize,
        res_mode: i32,
        keep_f32: bool,
        g: Geo,
    ) -> Result<(CudaTensor, Option<CudaTensor>)> {
        let dev = global_device().ok_or_else(|| msg("no device"))?;
        let (ho, ao, eo) = (
            operand(h)?.ok_or_else(|| msg("wan fuse: h off device"))?,
            operand(a)?.ok_or_else(|| msg("wan fuse: branch off device"))?,
            operand(e)?.ok_or_else(|| msg("wan fuse: table off device"))?,
        );
        let n = g.rows * g.dim;
        let mut hid = OutBuf::new(n, true)?;
        let mut tmp = OutBuf::new(if keep_f32 { n } else { 1 }, false)?;
        let (hp, tp, has) = (hid.ptr(), tmp.ptr(), i32::from(keep_f32));
        let gs = gate_slot as i32;
        let n_i = n as i64;
        let v = [g.seq, g.dim, g.e_rows].map(|u| u as i32);
        launch!(dev.stream, &dev.kernels.wan_res_gate, cfg_n(n);
            &ho.ptr, &ho.is16, &ao.ptr, &ao.is16, &eo.ptr, &eo.is16, &gs,
            &res_mode, &hp, &tp, &has, &n_i, &v[0], &v[1], &v[2])
        .map_err(err)?;
        let hid = hid.into_tensor(h.shape.clone())?;
        let tmp = if keep_f32 {
            Some(tmp.into_tensor(h.shape.clone())?)
        } else {
            None
        };
        Ok((hid, tmp))
    }

    /// Norm of a stored row (f32 or bf16) into bf16.
    pub(crate) fn ln(
        x: &CudaTensor,
        e: &CudaTensor,
        affine: Option<(&CudaTensor, &CudaTensor)>,
        (scale_slot, shift_slot): (usize, usize),
        ln_mode: i32,
        g: Geo,
        eps: f32,
    ) -> Result<CudaTensor> {
        let dev = global_device().ok_or_else(|| msg("no device"))?;
        let xo = operand(x)?.ok_or_else(|| msg("wan fuse: norm input off device"))?;
        let eo = operand(e)?.ok_or_else(|| msg("wan fuse: table off device"))?;
        let params = match affine {
            Some((w, b)) => Some((f32_param(w)?, f32_param(b)?)),
            None => None,
        };
        let pops = match &params {
            Some((w, b)) => Some((
                operand(w)?.ok_or_else(|| msg("wan fuse: norm weight"))?,
                operand(b)?.ok_or_else(|| msg("wan fuse: norm bias"))?,
            )),
            None => None,
        };
        let (wp, bp) = pops
            .as_ref()
            .map_or((xo.ptr, xo.ptr), |(w, b)| (w.ptr, b.ptr));
        let mut out = OutBuf::new(g.rows * g.dim, true)?;
        let op = out.ptr();
        let (ss, sh, p16) = (scale_slot as i32, shift_slot as i32, 0i32);
        let v = [g.rows, g.seq, g.dim, g.e_rows].map(|u| u as i32);
        launch!(dev.stream, &dev.kernels.wan_ln, cfg_rows(g.rows);
            &xo.ptr, &xo.is16, &wp, &bp, &p16, &eo.ptr, &eo.is16, &ss, &sh, &ln_mode,
            &op, &v[0], &v[1], &v[2], &v[3], &eps)
        .map_err(err)?;
        out.into_tensor(x.shape.clone())
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn qk_norm_rope(
        proj: &CudaTensor,
        col_off: usize,
        heads: usize,
        d: usize,
        weight: &CudaTensor,
        rope: Option<Rope<'_>>,
        eps: f32,
    ) -> Result<CudaTensor> {
        let [batch, seq, width] = proj.shape[..] else {
            return Err(msg("wan qk norm: shape"));
        };
        let dev = global_device().ok_or_else(|| msg("no device"))?;
        let xo = operand(proj)?.ok_or_else(|| msg("wan qk norm: input off device"))?;
        let wo = operand(weight)?.ok_or_else(|| msg("wan qk norm: weight"))?;
        let (cos, sin, use_rope) = match &rope {
            Some(r) => (r.cos.to_f32_act()?, r.sin.to_f32_act()?, 1i32),
            None => (weight.to_f32_act()?, weight.to_f32_act()?, 0i32),
        };
        let (co, so) = (
            operand(&cos)?.ok_or_else(|| msg("rope cos"))?,
            operand(&sin)?.ok_or_else(|| msg("rope sin"))?,
        );
        if co.is16 != 0 || so.is16 != 0 {
            return Err(msg("wan qk norm: rope tables must be f32"));
        }
        let mut out = OutBuf::new(batch * heads * seq * d, true)?;
        let op = out.ptr();
        let v = [batch, seq, heads, d, width, col_off].map(|u| u as i32);
        launch!(dev.stream, &dev.kernels.wan_qk_norm_rope16, cfg_rows(batch * seq);
            &xo.ptr, &xo.is16, &wo.ptr, &wo.is16, &co.ptr, &so.ptr, &use_rope, &op,
            &v[0], &v[1], &v[2], &v[3], &v[4], &v[5], &eps)
        .map_err(err)?;
        out.into_tensor(vec![batch, heads, seq, d])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wan::tensor::with_bf16_act;

    fn t16(v: Vec<f32>, shape: &[usize]) -> CudaTensor {
        let v = v.into_iter().map(bf16_round).collect();
        CudaTensor::host_only_dtype(v, shape.to_vec(), TensorDType::Bf16)
    }

    fn vals(n: usize, k: f32) -> Vec<f32> {
        (0..n).map(|i| ((i as f32 * k).sin() * 1.3) + 0.1).collect()
    }

    #[test]
    fn inactive_on_f32_activations() {
        let h = CudaTensor::from_vec(vals(12, 0.3), vec![1, 3, 4]).unwrap();
        let e = CudaTensor::from_vec(vals(24, 0.7), vec![1, 6, 4]).unwrap();
        let got = with_bf16_act(false, || gate_residual(&h, &h, &e, 5).unwrap());
        assert!(got.is_none());
    }

    #[test]
    fn host_twins_take_the_reference_rounding_points() {
        let (b, s, d) = (2usize, 3usize, 8usize);
        let h = t16(vals(b * s * d, 0.37), &[b, s, d]);
        let a = t16(vals(b * s * d, 1.1), &[b, s, d]);
        let e = CudaTensor::from_vec(vals(b * 6 * d, 0.29), vec![b, 6, d]).unwrap();
        let w = CudaTensor::from_vec(vals(d, 0.5), vec![d]).unwrap();
        let bias = CudaTensor::from_vec(vals(d, 0.9), vec![d]).unwrap();
        let (hid, normed) = with_bf16_act(true, || {
            self_residual_norm(&h, &a, &e, 2, &w, &bias, 1e-6)
                .unwrap()
                .unwrap()
        });
        assert!(hid.is_bf16() && normed.is_bf16());
        let (hh, ah, eh) = (
            h.host_cow().unwrap(),
            a.host_cow().unwrap(),
            e.host_cow().unwrap(),
        );
        // hidden: one rounding of the f32 sum.
        for i in 0..b * s * d {
            let (r, j) = (i / d, i % d);
            let g = eh[((r / s) * 6 + 2) * d + j];
            assert_eq!(hid.host_cow().unwrap()[i], bf16_round(hh[i] + ah[i] * g));
        }
        // The last residual matches `ScaleResidual` in f32.
        let out = with_bf16_act(true, || gate_residual(&h, &a, &e, 5).unwrap().unwrap());
        let g0 = eh[5 * d];
        assert_eq!(out.host_cow().unwrap()[0], bf16_round(hh[0] + ah[0] * g0));
        // Cross residual: the hidden is the rounded plain sum.
        let (hid2, n2) = with_bf16_act(true, || {
            cross_residual_norm_mod(&h, &a, &e, 4, 3, 1e-6)
                .unwrap()
                .unwrap()
        });
        assert_eq!(hid2.host_cow().unwrap()[1], bf16_round(hh[1] + ah[1]));
        assert!(n2.host_cow().unwrap().iter().all(|v| *v == bf16_round(*v)));
    }

    #[test]
    fn qk_norm_rope_host_matches_the_f32_op_to_a_bf16_ulp() {
        let (b, s, heads, d) = (1usize, 4usize, 2usize, 4usize);
        let width = heads * d;
        let x = t16(vals(b * s * 3 * width, 0.37), &[b, s, 3 * width]);
        let w = CudaTensor::from_vec(
            vals(width, 0.9).iter().map(|v| 1.0 + 0.1 * v).collect(),
            vec![width],
        )
        .unwrap();
        let ang = vals(s * d, 2.1);
        let cos = CudaTensor::from_vec(ang.iter().map(|a| a.cos()).collect(), vec![s, d]).unwrap();
        let sin = CudaTensor::from_vec(ang.iter().map(|a| a.sin()).collect(), vec![s, d]).unwrap();
        let rope = || {
            Some(Rope {
                cos: &cos,
                sin: &sin,
            })
        };
        let got = with_bf16_act(true, || {
            qk_norm_rope(&x, width, heads, &w, rope(), 1e-6)
                .unwrap()
                .unwrap()
        });
        let want = with_bf16_act(false, || {
            x.qk_norm_rope_bhsd(width, heads, &w, rope(), 1e-6).unwrap()
        });
        assert_eq!(got.shape, want.shape);
        for (g, w) in got
            .host_cow()
            .unwrap()
            .iter()
            .zip(want.host_cow().unwrap().iter())
        {
            // Three bf16 roundings of O(1) values; the rotation can cancel.
            assert!((g - w).abs() <= 0.03, "{g} vs {w}");
        }
    }
}
