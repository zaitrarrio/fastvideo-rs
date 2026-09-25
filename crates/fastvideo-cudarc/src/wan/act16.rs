//! bf16-native device ops for `FASTVIDEO_BF16_ACT`.
//!
//! Each op reads its operands in whatever dtype they are stored (a bf16
//! activation is never widened into a temporary f32 buffer first), computes in
//! f32 and rounds once when it writes bf16 — torch's per-op bf16 semantics.
//! Norm statistics and softmax stay f32 inside the kernels. The fused H3 block
//! kernels keep Sol-H3's `fusions.py` rounding points instead (see
//! `kernels.cu`, region "bf16 activations + reference FP8 recipes").
//!
//! Which dtype an op produces (see [`super::tensor`] for the callers):
//! * elementwise binary: bf16 iff both operands are bf16 (torch promotion);
//! * everything else (unary, norms, RoPE, SwiGLU, gated residual, head
//!   split/merge, narrow, permute): the dtype of the activation operand —
//!   weights and AdaLN tables are parameters, bf16 in the reference;
//! * `cat`: bf16 if any part is (the reference's explicit `.to(dtype)`).
#![cfg(feature = "cuda")]

use std::sync::Arc;

use cudarc::driver::CudaSlice;

use super::device::{global_device, DeviceContext};
use super::kernels::{cfg_n, cfg_rows, launch};
use super::ops::BcastOp;
use super::quant::{ptr, ptr_mut};
use super::tensor::{CudaTensor, Result, TensorDType, TensorError};

fn msg(s: impl Into<String>) -> TensorError {
    TensorError::Message(s.into())
}

fn err(e: impl std::fmt::Display) -> TensorError {
    TensorError::Message(e.to_string())
}

fn ctx() -> Result<Arc<DeviceContext>> {
    global_device().ok_or_else(|| msg("no global CUDA device context"))
}

/// A tensor's device storage as a raw pointer plus its dtype, keeping any
/// temporary upload alive.
pub(crate) struct Operand<'a> {
    _keep: super::tensor::DevAny<'a>,
    pub ptr: u64,
    pub is16: i32,
}

pub(crate) fn operand(t: &CudaTensor) -> Result<Option<Operand<'_>>> {
    let Some(any) = t.dev_any()? else {
        return Ok(None);
    };
    let (p, is16) = match &any {
        super::tensor::DevAny::F32(s) => (ptr::<f32>(s), 0),
        super::tensor::DevAny::Bf16(s) => (ptr::<half::bf16>(s.as_ref()), 1),
    };
    Ok(Some(Operand {
        _keep: any,
        ptr: p,
        is16,
    }))
}

/// A fresh output buffer of `n` elements in the requested dtype.
pub(crate) enum OutBuf {
    F32(CudaSlice<f32>),
    Bf16(CudaSlice<half::bf16>),
}

impl OutBuf {
    pub(crate) fn new(n: usize, bf16: bool) -> Result<Self> {
        let dev = ctx()?;
        Ok(if bf16 {
            Self::Bf16(unsafe { dev.stream.alloc::<half::bf16>(n.max(1)) }.map_err(err)?)
        } else {
            Self::F32(unsafe { dev.stream.alloc::<f32>(n.max(1)) }.map_err(err)?)
        })
    }

    pub(crate) fn ptr(&mut self) -> u64 {
        match self {
            Self::F32(s) => ptr_mut(s),
            Self::Bf16(s) => ptr_mut(s),
        }
    }

    pub(crate) fn is16(&self) -> i32 {
        i32::from(matches!(self, Self::Bf16(_)))
    }

    pub(crate) fn into_tensor(self, shape: Vec<usize>) -> Result<CudaTensor> {
        let n: usize = shape.iter().product();
        match self {
            Self::F32(s) if n == 0 || s.len() == n => CudaTensor::from_device_slice(s, shape),
            Self::Bf16(s) if n == 0 || s.len() == n => CudaTensor::from_device_slice_bf16(s, shape),
            _ => Err(msg("act16: output size mismatch")),
        }
    }
}

fn is16(t: &CudaTensor) -> bool {
    t.dtype() == TensorDType::Bf16
}

/// `op(big[i], small[(i / inner) % period])` into a fresh tensor of `shape`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn binary(
    big: &CudaTensor,
    small: &CudaTensor,
    op: BcastOp,
    inner: usize,
    period: usize,
    out16: bool,
    shape: Vec<usize>,
) -> Result<Option<CudaTensor>> {
    let (Some(a), Some(b)) = (operand(big)?, operand(small)?) else {
        return Ok(None);
    };
    let dev = ctx()?;
    let n = big.numel();
    let mut out = OutBuf::new(n, out16)?;
    let (op_p, o16) = (out.ptr(), out.is16());
    let (n_i, inner_i, period_i, op_i) = (
        n as i64,
        inner.max(1) as i64,
        period.max(1) as i64,
        op as i32,
    );
    launch!(dev.stream, &dev.kernels.mx_binary, cfg_n(n);
        &a.ptr, &a.is16, &b.ptr, &b.is16, &op_p, &o16, &n_i, &inner_i, &period_i, &op_i)
    .map_err(err)?;
    out.into_tensor(shape).map(Some)
}

/// Unary kinds of `mx_unary`.
#[derive(Clone, Copy, Debug)]
pub(crate) enum Unary {
    Silu = 0,
    GeluTanh = 1,
    GeluErf = 2,
    AddScalar = 3,
    MulScalar = 4,
    Clamp = 5,
    Sigmoid = 7,
}

pub(crate) fn unary(
    x: &CudaTensor,
    kind: Unary,
    p0: f32,
    p1: f32,
    out16: bool,
) -> Result<Option<CudaTensor>> {
    let Some(a) = operand(x)? else {
        return Ok(None);
    };
    let dev = ctx()?;
    let n = x.numel();
    let mut out = OutBuf::new(n, out16)?;
    let (op_p, o16) = (out.ptr(), out.is16());
    let (n_i, k_i) = (n as i64, kind as i32);
    launch!(dev.stream, &dev.kernels.mx_unary, cfg_n(n);
        &a.ptr, &a.is16, &op_p, &o16, &n_i, &k_i, &p0, &p1)
    .map_err(err)?;
    out.into_tensor(x.shape.clone()).map(Some)
}

pub(crate) fn swiglu(x: &CudaTensor, half: usize, shape: Vec<usize>) -> Result<Option<CudaTensor>> {
    let Some(a) = operand(x)? else {
        return Ok(None);
    };
    let dev = ctx()?;
    let n = x.numel() / 2;
    let mut out = OutBuf::new(n, is16(x))?;
    let (op_p, o16) = (out.ptr(), out.is16());
    let (half_i, n_i) = (half as i64, n as i64);
    launch!(dev.stream, &dev.kernels.mx_swiglu, cfg_n(n); &a.ptr, &a.is16, &op_p, &o16, &half_i, &n_i)
        .map_err(err)?;
    out.into_tensor(shape).map(Some)
}

pub(crate) fn rms_norm(
    x: &CudaTensor,
    w: Option<&CudaTensor>,
    eps: f32,
) -> Result<Option<CudaTensor>> {
    let Some(a) = operand(x)? else {
        return Ok(None);
    };
    let width = *x.shape.last().ok_or_else(|| msg("rms_norm on scalar"))?;
    let wo = match w {
        Some(w) => Some(operand(w)?.ok_or_else(|| msg("rms_norm weight upload"))?),
        None => None,
    };
    let dev = ctx()?;
    let rows = x.numel() / width.max(1);
    let mut out = OutBuf::new(x.numel(), is16(x))?;
    let (op_p, o16) = (out.ptr(), out.is16());
    let (wp, w16, has) = match &wo {
        Some(w) => (w.ptr, w.is16, 1i32),
        None => (a.ptr, 0, 0),
    };
    let (rows_i, width_i) = (rows as i32, width as i32);
    launch!(dev.stream, &dev.kernels.mx_rms_norm, cfg_rows(rows);
        &a.ptr, &a.is16, &wp, &w16, &has, &op_p, &o16, &rows_i, &width_i, &eps)
    .map_err(err)?;
    out.into_tensor(x.shape.clone()).map(Some)
}

pub(crate) fn layer_norm(
    x: &CudaTensor,
    affine: Option<(&CudaTensor, &CudaTensor)>,
    eps: f32,
) -> Result<Option<CudaTensor>> {
    let Some(a) = operand(x)? else {
        return Ok(None);
    };
    let width = *x.shape.last().ok_or_else(|| msg("layer_norm on scalar"))?;
    // Both parameters in one dtype: widen/narrow the pair if they differ.
    let pair = match affine {
        Some((w, b)) => {
            let (w, b) = if is16(w) == is16(b) {
                (w.clone(), b.clone())
            } else {
                (w.to_f32_act()?, b.to_f32_act()?)
            };
            Some((w, b))
        }
        None => None,
    };
    let ops = match &pair {
        Some((w, b)) => Some((
            operand(w)?.ok_or_else(|| msg("layer_norm weight"))?,
            operand(b)?.ok_or_else(|| msg("layer_norm bias"))?,
        )),
        None => None,
    };
    let dev = ctx()?;
    let rows = x.numel() / width.max(1);
    let mut out = OutBuf::new(x.numel(), is16(x))?;
    let (op_p, o16) = (out.ptr(), out.is16());
    let (wp, bp, p16, has) = match &ops {
        Some((w, b)) => (w.ptr, b.ptr, w.is16, 1i32),
        None => (a.ptr, a.ptr, 0, 0),
    };
    let (rows_i, width_i) = (rows as i32, width as i32);
    launch!(dev.stream, &dev.kernels.mx_layer_norm, cfg_rows(rows);
        &a.ptr, &a.is16, &wp, &bp, &p16, &has, &op_p, &o16, &rows_i, &width_i, &eps)
    .map_err(err)?;
    out.into_tensor(x.shape.clone()).map(Some)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn ln_adaln_e(
    x: &CudaTensor,
    e: &CudaTensor,
    batch: usize,
    seq: usize,
    dim: usize,
    e_rows: usize,
    scale_slot: usize,
    shift_slot: usize,
    eps: f32,
) -> Result<Option<CudaTensor>> {
    let (Some(a), Some(eo)) = (operand(x)?, operand(e)?) else {
        return Ok(None);
    };
    let dev = ctx()?;
    let mut out = OutBuf::new(x.numel(), is16(x))?;
    let (op_p, o16) = (out.ptr(), out.is16());
    let v = [batch, seq, dim, e_rows, scale_slot, shift_slot].map(|u| u as i32);
    launch!(dev.stream, &dev.kernels.mx_ln_adaln_e, cfg_rows(batch * seq);
        &a.ptr, &a.is16, &eo.ptr, &eo.is16, &op_p, &o16, &v[0], &v[1], &v[2], &v[3], &v[4], &v[5], &eps)
    .map_err(err)?;
    out.into_tensor(x.shape.clone()).map(Some)
}

/// `h + a * e[b, slot]`; a bf16 result rounds the product first (eager).
#[allow(clippy::too_many_arguments)]
pub(crate) fn residual_gate(
    h: &CudaTensor,
    a: &CudaTensor,
    e: &CudaTensor,
    batch_seq_dim: (usize, usize, usize),
    e_rows: usize,
    slot: usize,
) -> Result<Option<CudaTensor>> {
    let (Some(ho), Some(ao), Some(eo)) = (operand(h)?, operand(a)?, operand(e)?) else {
        return Ok(None);
    };
    let (_, seq, dim) = batch_seq_dim;
    let dev = ctx()?;
    let out16 = is16(h);
    let mut out = OutBuf::new(h.numel(), out16)?;
    let (op_p, o16) = (out.ptr(), out.is16());
    let v = [h.numel(), dim, seq, e_rows, slot].map(|u| u as i64);
    let rp = i32::from(out16);
    launch!(dev.stream, &dev.kernels.mx_residual_gate, cfg_n(h.numel());
        &ho.ptr, &ho.is16, &ao.ptr, &ao.is16, &eo.ptr, &eo.is16, &op_p, &o16, &v[0], &v[1], &v[2], &v[3], &v[4], &rp)
    .map_err(err)?;
    out.into_tensor(h.shape.clone()).map(Some)
}

pub(crate) fn split_heads(
    x: &CudaTensor,
    batch: usize,
    seq: usize,
    heads: usize,
    d: usize,
    width: usize,
    col_off: usize,
) -> Result<Option<CudaTensor>> {
    let Some(a) = operand(x)? else {
        return Ok(None);
    };
    let dev = ctx()?;
    let n = batch * heads * seq * d;
    let mut out = OutBuf::new(n, is16(x))?;
    let (op_p, o16) = (out.ptr(), out.is16());
    let v = [n, seq, heads, d, width, col_off].map(|u| u as i64);
    launch!(dev.stream, &dev.kernels.mx_split_heads, cfg_n(n);
        &a.ptr, &a.is16, &op_p, &o16, &v[0], &v[1], &v[2], &v[3], &v[4], &v[5])
    .map_err(err)?;
    out.into_tensor(vec![batch, heads, seq, d]).map(Some)
}

pub(crate) fn merge_heads(
    x: &CudaTensor,
    batch: usize,
    heads: usize,
    seq: usize,
    d: usize,
) -> Result<Option<CudaTensor>> {
    let Some(a) = operand(x)? else {
        return Ok(None);
    };
    let dev = ctx()?;
    let n = x.numel();
    let mut out = OutBuf::new(n, is16(x))?;
    let (op_p, o16) = (out.ptr(), out.is16());
    let v = [n, seq, heads, d].map(|u| u as i64);
    launch!(dev.stream, &dev.kernels.mx_merge_heads, cfg_n(n);
        &a.ptr, &a.is16, &op_p, &o16, &v[0], &v[1], &v[2], &v[3])
    .map_err(err)?;
    out.into_tensor(vec![batch, seq, heads * d]).map(Some)
}

pub(crate) fn permute(
    x: &CudaTensor,
    dims: &[usize],
    out_shape: Vec<usize>,
) -> Result<Option<CudaTensor>> {
    let rank = x.rank();
    if rank > 6 {
        return Ok(None);
    }
    let Some(a) = operand(x)? else {
        return Ok(None);
    };
    let (oshape, strides) = super::ops::permute_strides(&x.shape, dims);
    let mut s = [1i64; 6];
    let mut t = [0i64; 6];
    for k in 0..rank {
        s[k] = oshape[k] as i64;
        t[k] = strides[k] as i64;
    }
    let dev = ctx()?;
    let n = x.numel();
    let mut out = OutBuf::new(n, is16(x))?;
    let (op_p, o16) = (out.ptr(), out.is16());
    let (n_i, rank_i) = (n as i64, rank as i32);
    launch!(dev.stream, &dev.kernels.mx_gather_nd, cfg_n(n);
        &a.ptr, &a.is16, &op_p, &o16, &n_i, &rank_i,
        &s[0], &s[1], &s[2], &s[3], &s[4], &s[5], &t[0], &t[1], &t[2], &t[3], &t[4], &t[5])
    .map_err(err)?;
    out.into_tensor(out_shape).map(Some)
}

/// Strided block copy of `x` into `out` (see `block_copy`).
#[allow(clippy::too_many_arguments)]
pub(crate) fn block_copy_into(
    x: &Operand<'_>,
    out: &mut OutBuf,
    outer: usize,
    len: usize,
    in_stride: usize,
    out_stride: usize,
    in_offset: usize,
    out_offset: usize,
) -> Result<()> {
    let dev = ctx()?;
    let (op_p, o16) = (out.ptr(), out.is16());
    let v = [outer, len, in_stride, out_stride, in_offset, out_offset].map(|u| u as i64);
    launch!(dev.stream, &dev.kernels.mx_block_copy, cfg_n(outer * len);
        &x.ptr, &x.is16, &op_p, &o16, &v[0], &v[1], &v[2], &v[3], &v[4], &v[5])
    .map_err(err)
}

pub(crate) fn rope_half(
    x: &CudaTensor,
    cos: &CudaTensor,
    sin: &CudaTensor,
    s: usize,
    d: usize,
    r: usize,
) -> Result<Option<CudaTensor>> {
    let Some(a) = operand(x)? else {
        return Ok(None);
    };
    // The tables are f32 (the fused reference consumes them unrounded).
    let (cos, sin) = (cos.to_f32_act()?, sin.to_f32_act()?);
    let (Some(c), Some(sn)) = (operand(&cos)?, operand(&sin)?) else {
        return Ok(None);
    };
    let dev = ctx()?;
    let n = x.numel();
    let mut out = OutBuf::new(n, is16(x))?;
    let (op_p, o16) = (out.ptr(), out.is16());
    let (cp, sp) = (c.ptr, sn.ptr);
    let v = [s, d, r, n].map(|u| u as i64);
    launch!(dev.stream, &dev.kernels.mx_rope_half, cfg_n(n);
        &a.ptr, &a.is16, &cp, &sp, &op_p, &o16, &v[0], &v[1], &v[2], &v[3])
    .map_err(err)?;
    out.into_tensor(x.shape.clone()).map(Some)
}
