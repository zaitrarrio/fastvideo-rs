//! The pre-B4b fusion code, verbatim, as the reference the unified module
//! is held to (`equality.rs`). Test-only and frozen: not code to maintain.
//!
//! Copied from main `17e4a1dc`, where the three copies lived. Adaptations,
//! all mechanical: each family's host twins are a nested module of the same
//! name; the `#[cfg(feature = "cuda")]` device branches are left out (the
//! device code is unchanged in place, and the launch arguments it chose are
//! [`launch`], a reduction of every literal mode argument to a table with
//! its source line); in `h3`, `AdaRows::row` (private there) is the free
//! function [`h3::row`] and `quant::*_row` call the copies in [`quant`]; the
//! getters keep their own statics, renamed per family.

#![allow(dead_code, clippy::too_many_arguments)]

/// `rt/quant.rs:203-325` (the bf16 fused-op references).
pub(crate) mod quant {
    use crate::rt::quant::bf16_round;
    pub(crate) const NORM_THREADS: usize = 256;

    fn rms_factor(x: &[f32], eps: f32) -> f32 {
        let ss: f32 = x.iter().map(|v| v * v).sum();
        1.0 / (ss / x.len() as f32 + eps).sqrt()
    }

    /// The fused H3 norm kernels' normalizer, bit for bit: lane `t` accumulates
    /// `x[t], x[t + 256], ...` with FMA, the 256 partials reduce by the kernel's
    /// halving tree, then `rsqrt(sum / n + eps)` correctly rounded
    /// (`__frsqrt_rn`). Sol-H3's Triton kernel uses its own order and an
    /// approximate rsqrt, so no implementation reproduces it bitwise; fixing both
    /// here keeps the device and this twin identical. The rounding points (f32
    /// chain, one bf16 rounding) are the reference's.
    pub(crate) fn row_rsqrt(x: &[f32], eps: f32) -> f32 {
        let mut part = [0.0f32; NORM_THREADS];
        for (t, p) in part.iter_mut().enumerate() {
            let mut j = t;
            while j < x.len() {
                *p = x[j].mul_add(x[j], *p);
                j += NORM_THREADS;
            }
        }
        let mut s = NORM_THREADS / 2;
        while s > 0 {
            for t in 0..s {
                part[t] += part[t + s];
            }
            s >>= 1;
        }
        let v = part[0] / x.len() as f32 + eps;
        // f64 then one rounding: the correctly rounded rsqrt except where the
        // f64 value sits within 2^-53 of an f32 midpoint.
        (1.0 / f64::from(v).sqrt()) as f32
    }

    /// One row of `_rmsnorm_modulate_kernel`: `bf16(rms(x) * w * scale1p + shift)`
    /// with `scale1p = 1 + scale` already formed in f32.
    pub(crate) fn norm_mod_row(
        x: &[f32],
        w: &[f32],
        scale1p: &[f32],
        shift: &[f32],
        eps: f32,
    ) -> Vec<f32> {
        let r = row_rsqrt(x, eps);
        x.iter()
            .enumerate()
            .map(|(j, &v)| bf16_round((v * r * w[j]).mul_add(scale1p[j], shift[j])))
            .collect()
    }

    /// One row of `_residual_gate_rmsnorm_modulate_kernel`: returns
    /// `(bf16(res + gate * branch), bf16(norm_mod(f32 hidden)))` — the norm sees
    /// the unrounded f32 hidden.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn res_gate_norm_mod_row(
        res: &[f32],
        branch: &[f32],
        gate: &[f32],
        w: &[f32],
        scale1p: &[f32],
        shift: &[f32],
        eps: f32,
    ) -> (Vec<f32>, Vec<f32>) {
        let h: Vec<f32> = (0..res.len())
            .map(|j| gate[j].mul_add(branch[j], res[j]))
            .collect();
        let normed = norm_mod_row(&h, w, scale1p, shift, eps);
        (h.into_iter().map(bf16_round).collect(), normed)
    }

    /// Eager `residual + gate * branch` in bf16: the product rounds first.
    pub(crate) fn gate_residual_eager(res: f32, gate: f32, branch: f32) -> f32 {
        bf16_round(res + bf16_round(gate * branch))
    }

    /// `_qknorm_partial_rope_kernel` for one head vector: RMS over `d`, weight,
    /// then rotate_half over the leading `r` channels with f32 `cos`/`sin`.
    pub(crate) fn qk_norm_rope_row(
        x: &[f32],
        w: &[f32],
        rope: Option<(&[f32], &[f32])>,
        eps: f32,
    ) -> Vec<f32> {
        let r_f = rms_factor(x, eps);
        let normed: Vec<f32> = x.iter().zip(w).map(|(&v, &wv)| v * r_f * wv).collect();
        match rope {
            None => normed.into_iter().map(bf16_round).collect(),
            Some((cos, sin)) => {
                let r = cos.len();
                let half = r / 2;
                (0..x.len())
                    .map(|p| {
                        if p >= r {
                            return bf16_round(normed[p]);
                        }
                        let partner = if p < half { p + half } else { p - half };
                        let rot = if p < half {
                            -normed[partner]
                        } else {
                            normed[partner]
                        };
                        bf16_round(normed[p] * cos[p] + rot * sin[p])
                    })
                    .collect()
            }
        }
    }
}

/// `wan/fuse.rs:59-305` (the host halves of the four ops and their helpers).
pub(crate) mod wan {
    use crate::rt::fused::Rope;
    use crate::rt::quant::bf16_round;
    use crate::rt::tensor::{CudaTensor, Result, TensorDType, TensorError};

    fn msg(s: impl Into<String>) -> TensorError {
        TensorError::Message(s.into())
    }

    /// Whether the block's residual stream `h` is on the bf16 path.
    pub fn active(h: &CudaTensor) -> bool {
        crate::rt::tensor::bf16_activations() && h.is_bf16()
    }

    fn dims(
        h: &CudaTensor,
        a: &CudaTensor,
        e: &CudaTensor,
    ) -> Result<(usize, usize, usize, usize)> {
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
        if heads == 0 || !proj_w.is_multiple_of(heads) || col_off + proj_w > width {
            return Err(msg("wan qk norm: weight / columns"));
        }
        let d = proj_w / heads;
        if let Some(r) = &rope {
            if r.cos.shape != [seq, d] || r.sin.shape != [seq, d] || !d.is_multiple_of(2) {
                return Err(msg(format!(
                    "wan qk norm: rope {:?} for d={d}",
                    r.cos.shape
                )));
            }
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
}

/// `h3/fused16.rs:44-240` (the host halves of steps 1, 2, 4 and 5).
pub(crate) mod h3 {
    use super::quant::{
        gate_residual_eager, norm_mod_row, qk_norm_rope_row, res_gate_norm_mod_row,
    };
    use crate::h3::fused16::{AdaRows, NormOut};
    use crate::rt::quant;
    use crate::rt::tensor::{CudaTensor, Result, TensorDType, TensorError};

    fn msg(s: impl Into<String>) -> TensorError {
        TensorError::Message(s.into())
    }

    /// `AdaRows::row` (`h3/fused16.rs:44-47`).
    pub(crate) fn row<'a>(
        rows: &AdaRows,
        tab: &'a [f32],
        r: usize,
        base: usize,
        slot: usize,
    ) -> &'a [f32] {
        let t = (rows.idx[r] as usize + base) * 6 + slot;
        &tab[t * rows.hidden..(t + 1) * rows.hidden]
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
        let _ = mx;
        let (xh, wh, tab) = (host_bf16(x)?, w.host_cow()?, rows.tab.host_cow()?);
        let mut out = vec![0.0f32; n * dim];
        for r in 0..n {
            let o = norm_mod_row(
                &xh[r * dim..(r + 1) * dim],
                &wh,
                row(rows, &tab, r, base, scale_slot),
                row(rows, &tab, r, base, shift_slot),
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
            let (h, o) = res_gate_norm_mod_row(
                &rh[span.clone()],
                &bh[span.clone()],
                row(rows, &tab, r, base, gate_slot),
                &wh,
                row(rows, &tab, r, base, scale_slot),
                row(rows, &tab, r, base, shift_slot),
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
        let (rh, bh, tab) = (host_bf16(res)?, host_bf16(branch)?, rows.tab.host_cow()?);
        let out: Vec<f32> = (0..rh.len())
            .map(|i| {
                let (r, j) = (i / dim, i % dim);
                gate_residual_eager(rh[i], row(rows, &tab, r, base, gate_slot)[j], bh[i])
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
                    let o = qk_norm_rope_row(src, &wh, rot, eps);
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
}

/// The three switches' getters (`ltx2/fuse.rs:29-48`, `wan/fuse.rs:37-56`,
/// `h3/fused16.rs:294-345`), each with its own override and cache.
pub(crate) mod getters {
    use std::cell::Cell;
    use std::sync::OnceLock;

    thread_local! {
        static LTX_OVERRIDE: Cell<Option<bool>> = const { Cell::new(None) };
    }

    /// `FASTVIDEO_LTX_FUSE` (default on).
    pub(crate) fn ltx_enabled() -> bool {
        if let Some(v) = LTX_OVERRIDE.with(|c| c.get()) {
            return v;
        }
        static ENV: OnceLock<bool> = OnceLock::new();
        *ENV.get_or_init(|| crate::rt::envflag::bool_flag("FASTVIDEO_LTX_FUSE", true))
    }

    /// Run `f` with the LTX fusions forced on or off (parity checks).
    pub(crate) fn ltx_with_fuse<R>(on: bool, f: impl FnOnce() -> R) -> R {
        let prev = LTX_OVERRIDE.with(|c| c.replace(Some(on)));
        let out = f();
        LTX_OVERRIDE.with(|c| c.set(prev));
        out
    }

    thread_local! {
        static WAN_FUSE_OVERRIDE: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
    }

    /// `FASTVIDEO_WAN_FUSE` (default on).
    pub(crate) fn wan_enabled() -> bool {
        if let Some(v) = WAN_FUSE_OVERRIDE.with(|c| c.get()) {
            return v;
        }
        static FLAG: crate::rt::envflag::CachedBool = crate::rt::envflag::CachedBool::new();
        FLAG.get_or_init(|| crate::rt::envflag::bool_flag("FASTVIDEO_WAN_FUSE", true))
    }

    /// Run `f` with [`wan_enabled`] forced on or off (parity checks).
    pub(crate) fn wan_with_fuse<R>(on: bool, f: impl FnOnce() -> R) -> R {
        let prev = WAN_FUSE_OVERRIDE.with(|c| c.replace(Some(on)));
        let out = f();
        WAN_FUSE_OVERRIDE.with(|c| c.set(prev));
        out
    }

    thread_local! {
        static FUSE_OVERRIDE: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
    }

    pub(crate) fn fuse_enabled() -> bool {
        if let Some(v) = FUSE_OVERRIDE.with(|c| c.get()) {
            return v;
        }
        static ENV: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ENV.get_or_init(|| crate::rt::envflag::bool_flag("FASTVIDEO_H3_FUSE", true))
    }

    /// Run `f` with [`fuse_enabled`] forced on or off (parity checks).
    pub(crate) fn with_fuse<R>(on: bool, f: impl FnOnce() -> R) -> R {
        let prev = FUSE_OVERRIDE.with(|c| c.replace(Some(on)));
        let out = f();
        FUSE_OVERRIDE.with(|c| c.set(prev));
        out
    }

    thread_local! {
        static GLUE_OVERRIDE: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
    }

    pub(crate) fn glue_enabled() -> bool {
        if !fuse_enabled() {
            return false;
        }
        if let Some(v) = GLUE_OVERRIDE.with(|c| c.get()) {
            return v;
        }
        static ENV: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ENV.get_or_init(|| crate::rt::envflag::bool_flag("FASTVIDEO_H3_GLUE", true))
    }

    /// Run `f` with [`glue_enabled`] forced on or off (parity checks).
    pub(crate) fn with_glue<R>(on: bool, f: impl FnOnce() -> R) -> R {
        let prev = GLUE_OVERRIDE.with(|c| c.replace(Some(on)));
        let out = f();
        GLUE_OVERRIDE.with(|c| c.set(prev));
        out
    }

    /// The plans' reads: `h3/pipeline/load.rs:640-643` (`kernel_fusions`,
    /// a `(fused16, glue)` pair) and `ltx2/pipeline/load.rs:272`.
    pub(crate) fn h3_plan(
        settings: &dyn fastvideo_models::techniques::settings::Source,
    ) -> (bool, bool) {
        let fuse = settings.bool_flag("FASTVIDEO_H3_FUSE", true);
        (fuse, fuse && settings.bool_flag("FASTVIDEO_H3_GLUE", true))
    }

    pub(crate) fn ltx_plan(settings: &dyn fastvideo_models::techniques::settings::Source) -> bool {
        settings.bool_flag("FASTVIDEO_LTX_FUSE", true)
    }
}

/// Every literal kernel choice and mode argument of the residual / norm
/// launches, as the pre-B4b code wrote them (kernel name, then the
/// arguments).
pub(crate) mod launch {
    /// `wan/fuse.rs:124`: `dev::res_ln(.., Some((w, b)), (0, 0), 0, 0, ..)`
    /// (`res_mode` 0, `ln_mode` 0) into `wan_res_ln` (`:397`).
    pub(crate) const WAN_SELF: (&str, i32, i32) = ("wan_res_ln", 0, 0);
    /// `wan/fuse.rs:173`: `dev::res_ln(.., None, (scale, shift), 1, 1, ..)`.
    pub(crate) const WAN_CROSS: (&str, i32, i32) = ("wan_res_ln", 1, 1);
    /// `wan/fuse.rs:220`: `dev::res_gate(.., gate_slot, 0, false, g)` into
    /// `wan_res_gate` (`:431`): `res_mode` 0, no f32 copy.
    pub(crate) const WAN_GATE_RESIDUAL: (&str, i32, bool) = ("wan_res_gate", 0, false);

    /// `wan/fuse.rs:366-368` (`FASTVIDEO_WAN_FUSE=0`): `res_gate(.., res_mode,
    /// res_mode == 0, g)` then `ln(src, .., ln_mode, ..)` (`wan_ln`, `:475`).
    pub(crate) fn wan_unfused(
        res_mode: i32,
        ln_mode: i32,
    ) -> ((&'static str, i32, bool), (&'static str, i32)) {
        (
            ("wan_res_gate", res_mode, res_mode == 0),
            ("wan_ln", ln_mode),
        )
    }

    /// LTX's residual (`ltx2/fuse.rs:193-212`).
    #[derive(Clone, Copy, Debug)]
    pub(crate) enum LtxRes {
        None,
        Plain,
        Gated,
    }

    /// `ltx2/fuse.rs:182-212`: `mode` 0 for `Norm::AdaLn`, 1 for
    /// `Norm::ThenMod`; `res_mode` 0 without a residual, 2 for
    /// `Residual::Plain`, 1 for `Residual::Gated`; into
    /// `fvf_ltx_res_norm_mod` (`:233`). `(name, res_mode, mode)`.
    pub(crate) fn ltx(res: LtxRes, then_mod: bool) -> (&'static str, i32, i32) {
        let mode = if then_mod { 1i32 } else { 0i32 };
        let res_mode = match res {
            LtxRes::None => 0i32,
            LtxRes::Plain => 2,
            LtxRes::Gated => 1,
        };
        ("fvf_ltx_res_norm_mod", res_mode, mode)
    }

    /// `h3/fused16.rs:652-653`: `rp = 1i32` into `h3_gate_residual`.
    pub(crate) const H3_GATE_RESIDUAL: (&str, i32) = ("h3_gate_residual", 1);
    /// `h3/fused16.rs:579`, `:623`, `:813`.
    pub(crate) const H3_NORM_MOD: &str = "h3_norm_mod";
    pub(crate) const H3_RES_GATE_NORM_MOD: &str = "h3_res_gate_norm_mod";
    pub(crate) const H3_GATE_RES_NORM_MOD: &str = "fvf_h3_gate_res_norm_mod";
    /// The q/k kernels: `wan/fuse.rs:512`, `h3/fused16.rs:706`,
    /// `ltx2/fuse.rs:292`.
    pub(crate) const QK: [&str; 3] = [
        "wan_qk_norm_rope16",
        "h3_qk_norm_rope",
        "fvf_ltx_qk_norm_rope",
    ];
}
