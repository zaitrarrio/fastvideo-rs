//! Fused LTX-2 DiT block ops for bf16 activations (the device default).
//!
//! Each function here is one kernel (`kernels.cu`, region "DiT block
//! fusions") standing in for a chain of bf16 ops the block otherwise runs one
//! launch at a time, and it keeps that chain's rounding points exactly, so the
//! fused block is bit-identical to the unfused one:
//!
//! * [`qk_norm_rope`]: `rms_norm` across heads → `split_heads_bhsd` →
//!   [`DeviceRope::apply`] (q and k of every attention);
//! * [`norm_mod`] / [`res_norm_mod`]: `rms_adaln` (`rms_norm(x, 1 + scale) +
//!   shift`) or the text cross-attention's `rms_norm(x, ones)` + `scale_shift`,
//!   optionally behind the residual that produces `x`
//!   (`residual_gate_add_e`, the LTX-2.5 text-cross gate + add, a plain add);
//! * [`gate_merge`]: LTX-2.5's per-head `2·σ(logits)` output gates and the
//!   head merge that feeds `to_out`.
//!
//! Every function returns `Ok(None)` when it does not apply (fusion off, f32
//! activations, an operand not stored as device bf16, a host run) and the
//! caller runs the unfused chain. `FASTVIDEO_LTX_FUSE=0` turns all of them off;
//! `fv-gpucheck kernels` (group `dit_fusion`) checks each against its chain,
//! bit for bit.

use std::cell::Cell;
use std::sync::OnceLock;

use super::attention::DeviceRope;
use crate::wan::tensor::{CudaTensor, Result};

thread_local! {
    static OVERRIDE: Cell<Option<bool>> = const { Cell::new(None) };
}

/// `FASTVIDEO_LTX_FUSE` (default on).
pub fn enabled() -> bool {
    if let Some(v) = OVERRIDE.with(|c| c.get()) {
        return v;
    }
    static ENV: OnceLock<bool> = OnceLock::new();
    *ENV.get_or_init(|| crate::wan::envflag::bool_flag("FASTVIDEO_LTX_FUSE", true))
}

/// Run `f` with the LTX fusions forced on or off (parity checks).
pub fn with_fuse<R>(on: bool, f: impl FnOnce() -> R) -> R {
    let prev = OVERRIDE.with(|c| c.replace(Some(on)));
    let out = f();
    OVERRIDE.with(|c| c.set(prev));
    out
}

/// Rows of a bf16 `[rows, width]` modulation table.
#[derive(Clone, Copy)]
pub struct Mod<'a> {
    pub tab: &'a CudaTensor,
    pub scale: usize,
    pub shift: usize,
}

/// Which modulated norm.
#[derive(Clone, Copy)]
pub enum Norm<'a> {
    /// `rms_norm(x, 1 + scale) + shift` (`rms_adaln`).
    AdaLn(Mod<'a>),
    /// `rms_norm(x, w)`, then `x · (1 + scale) + shift` (`scale_shift`).
    ThenMod(&'a CudaTensor, Mod<'a>),
}

/// The residual in front of the norm: `x + gate[row] · u` (the product
/// rounded first, as `residual_gate_add_e`), or `x + u`.
#[derive(Clone, Copy)]
pub enum Residual<'a> {
    Gated {
        u: &'a CudaTensor,
        gates: &'a CudaTensor,
        row: usize,
    },
    Plain(&'a CudaTensor),
}

#[cfg(feature = "cuda")]
fn active() -> bool {
    enabled() && crate::wan::tensor::bf16_activations() && crate::wan::stats::device_expected()
}

#[cfg(feature = "cuda")]
fn b16(t: &CudaTensor) -> Option<&cudarc::driver::CudaSlice<half::bf16>> {
    if t.is_bf16() {
        t.device_slice_bf16()
    } else {
        None
    }
}

/// `[rows, width]` view of a modulation / gate table (`[rows, width]` or
/// `[1, rows, width]`), when it has `width` columns.
#[cfg(feature = "cuda")]
fn table_rows(t: &CudaTensor, width: usize) -> Option<usize> {
    let w = *t.shape.last()?;
    (w == width && t.numel() % width == 0).then(|| t.numel() / width)
}

/// Fused [`Norm`] of `x` (`[.., width]`, bf16).
pub fn norm_mod(x: &CudaTensor, norm: Norm<'_>, eps: f32) -> Result<Option<CudaTensor>> {
    #[cfg(feature = "cuda")]
    if active() {
        return dev::res_norm_mod(x, None, norm, eps).map(|o| o.map(|(_, n)| n));
    }
    let _ = (x, norm, eps);
    Ok(None)
}

/// `(hidden, normed)`: the residual, then the [`Norm`] of the stored hidden.
pub fn res_norm_mod(
    x: &CudaTensor,
    res: Residual<'_>,
    norm: Norm<'_>,
    eps: f32,
) -> Result<Option<(CudaTensor, CudaTensor)>> {
    #[cfg(feature = "cuda")]
    if active() {
        return dev::res_norm_mod(x, Some(res), norm, eps)
            .map(|o| o.and_then(|(h, n)| h.map(|h| (h, n))));
    }
    let _ = (x, res, norm, eps);
    Ok(None)
}

/// `[1, S, heads·d]` projection → normed (across heads, `[heads·d]` weight),
/// BHSD, rotated by `rope` when given.
pub fn qk_norm_rope(
    x: &CudaTensor,
    w: &CudaTensor,
    eps: f32,
    heads: usize,
    d: usize,
    rope: Option<&DeviceRope>,
) -> Result<Option<CudaTensor>> {
    #[cfg(feature = "cuda")]
    if active() {
        return dev::qk_norm_rope(x, w, eps, heads, d, rope);
    }
    let _ = (x, w, eps, heads, d, rope);
    Ok(None)
}

/// `merge_heads(out · 2σ(logits))`: `out` `[1, H, S, D]`, `logits` `[1, S, H]`.
pub fn gate_merge(out: &CudaTensor, logits: &CudaTensor) -> Result<Option<CudaTensor>> {
    #[cfg(feature = "cuda")]
    if active() {
        return dev::gate_merge(out, logits);
    }
    let _ = (out, logits);
    Ok(None)
}

// `launch!` names `super::stats` / `super::device` from its call site.
#[cfg(feature = "cuda")]
use crate::wan::{device, stats};

#[cfg(feature = "cuda")]
mod dev {
    use super::*;
    use crate::wan::act16::{operand, OutBuf};
    use crate::wan::device::global_device;
    use crate::wan::kernels::{cfg_rows, launch};
    use crate::wan::quant::ptr;
    use crate::wan::tensor::TensorError;

    fn err(e: impl std::fmt::Display) -> TensorError {
        TensorError::Message(e.to_string())
    }

    pub(super) fn res_norm_mod(
        x: &CudaTensor,
        res: Option<Residual<'_>>,
        norm: Norm<'_>,
        eps: f32,
    ) -> Result<Option<(Option<CudaTensor>, CudaTensor)>> {
        let Some(&width) = x.shape.last() else {
            return Ok(None);
        };
        let Some(xs) = b16(x) else { return Ok(None) };
        let (m, w, mode) = match norm {
            Norm::AdaLn(m) => (m, None, 0i32),
            Norm::ThenMod(w, m) => (m, Some(w), 1i32),
        };
        let Some(ts) = b16(m.tab) else {
            return Ok(None);
        };
        match table_rows(m.tab, width) {
            Some(r) if m.scale < r && m.shift < r => {}
            _ => return Ok(None),
        }
        let (us, gs, grow, res_mode) = match res {
            None => (xs, ts, 0usize, 0i32),
            Some(Residual::Plain(u)) => {
                let Some(us) = b16(u).filter(|_| u.shape == x.shape) else {
                    return Ok(None);
                };
                (us, ts, 0, 2)
            }
            Some(Residual::Gated { u, gates, row }) => {
                let (Some(us), Some(gs)) = (b16(u).filter(|_| u.shape == x.shape), b16(gates))
                else {
                    return Ok(None);
                };
                match table_rows(gates, width) {
                    Some(r) if row < r => {}
                    _ => return Ok(None),
                }
                (us, gs, row, 1)
            }
        };
        // The norm weight is read as stored (f32 ones or a bf16 parameter).
        let wo = match w {
            Some(w) if w.numel() == width => Some(operand(w)?.ok_or_else(|| err("norm weight"))?),
            Some(_) => return Ok(None),
            None => None,
        };
        let (wp, w16) = match &wo {
            Some(o) => (o.ptr, o.is16),
            None => (ptr(xs), 1),
        };
        let dev = global_device().ok_or_else(|| err("no device"))?;
        let n = x.numel();
        let rows = n / width.max(1);
        let mut out = OutBuf::new(n, true)?;
        let mut hidden = OutBuf::new(if res_mode == 0 { 1 } else { n }, true)?;
        let (op, hp) = (out.ptr(), hidden.ptr());
        let (xp, up, gp) = (ptr(xs), ptr(us), ptr(gs));
        let tp = ptr(ts);
        let (grow_i, sc, sh) = (grow as i32, m.scale as i32, m.shift as i32);
        let (rows_i, width_i) = (rows as i32, width as i32);
        launch!(dev.stream, &dev.kernels.fvf_ltx_res_norm_mod, cfg_rows(rows);
            &xp, &up, &gp, &grow_i, &res_mode, &wp, &w16, &tp, &sc, &sh, &mode,
            &hp, &op, &rows_i, &width_i, &eps)
        .map_err(err)?;
        let normed = out.into_tensor(x.shape.clone())?;
        let hidden = if res_mode == 0 {
            None
        } else {
            Some(hidden.into_tensor(x.shape.clone())?)
        };
        Ok(Some((hidden, normed)))
    }

    pub(super) fn qk_norm_rope(
        x: &CudaTensor,
        w: &CudaTensor,
        eps: f32,
        heads: usize,
        d: usize,
        rope: Option<&DeviceRope>,
    ) -> Result<Option<CudaTensor>> {
        let [1, seq, inner] = x.shape[..] else {
            return Ok(None);
        };
        if inner != heads * d || w.numel() != inner || d == 0 {
            return Ok(None);
        }
        let Some(xs) = b16(x) else { return Ok(None) };
        let tables = match rope {
            Some(r) => {
                let Some((cos, sin, r_w)) = r.fused_tables(heads, seq, d)? else {
                    return Ok(None);
                };
                Some((cos, sin, r_w))
            }
            None => None,
        };
        let wo = operand(w)?.ok_or_else(|| err("qk norm weight"))?;
        let (co, so, r_w, use_rope) = match &tables {
            Some((c, s, r_w)) => (
                Some(operand(c)?.ok_or_else(|| err("rope cos"))?),
                Some(operand(s)?.ok_or_else(|| err("rope sin"))?),
                *r_w,
                1i32,
            ),
            None => (None, None, 0, 0),
        };
        let (cp, sp) = match (&co, &so) {
            (Some(c), Some(s)) => (c.ptr, s.ptr),
            _ => (wo.ptr, wo.ptr),
        };
        let dev = global_device().ok_or_else(|| err("no device"))?;
        let mut out = OutBuf::new(x.numel(), true)?;
        let op = out.ptr();
        let xp = ptr(xs);
        let x16 = 1i32;
        let v = [seq, heads, d, r_w].map(|u| u as i32);
        // Every DeviceRope table row holds its values twice (see fused_tables).
        let dup = 1i32;
        launch!(dev.stream, &dev.kernels.fvf_ltx_qk_norm_rope, cfg_rows(seq);
            &xp, &x16, &wo.ptr, &wo.is16, &cp, &sp, &use_rope, &op,
            &v[0], &v[1], &v[2], &v[3], &eps, &dup)
        .map_err(err)?;
        out.into_tensor(vec![1, heads, seq, d]).map(Some)
    }

    pub(super) fn gate_merge(out: &CudaTensor, logits: &CudaTensor) -> Result<Option<CudaTensor>> {
        let [1, heads, seq, d] = out.shape[..] else {
            return Ok(None);
        };
        if logits.shape != [1, seq, heads] || d % 2 != 0 || heads == 0 {
            return Ok(None);
        }
        let (Some(os), Some(ls)) = (b16(out), b16(logits)) else {
            return Ok(None);
        };
        let dev = global_device().ok_or_else(|| err("no device"))?;
        let mut merged = OutBuf::new(out.numel(), true)?;
        let mp = merged.ptr();
        let (opp, lp) = (ptr(os), ptr(ls));
        let (seq_i, heads_i, d_i) = (seq as i32, heads as i32, d as i32);
        let mut cfg = cfg_rows(seq);
        cfg.shared_mem_bytes = (heads * std::mem::size_of::<f32>()) as u32;
        launch!(dev.stream, &dev.kernels.fvf_ltx_gate_merge, cfg;
            &opp, &lp, &mp, &seq_i, &heads_i, &d_i)
        .map_err(err)?;
        merged.into_tensor(vec![1, seq, heads * d]).map(Some)
    }
}
