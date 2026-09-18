//! Kernel wrappers: every NVRTC kernel as a device function, and its plain-Rust
//! twin in [`host`] (the CPU path and the reference `fv-gpucheck` compares
//! against). Device functions return fresh buffers (or update in place where
//! the name says so) and never read back to the host.

#[cfg(feature = "cuda")]
use std::sync::Arc;

#[cfg(feature = "cuda")]
use cudarc::driver::CudaSlice;

#[cfg(feature = "cuda")]
use super::device::{self, DeviceContext};
#[cfg(feature = "cuda")]
use super::kernels::{cfg_n, cfg_rows, launch};
#[cfg(feature = "cuda")]
use cudarc::driver::LaunchConfig;
#[cfg(feature = "cuda")]
use super::tensor::TensorError;
use super::tensor::{CudaTensor, Result};

/// Dense SDPA via [`CudaTensor`].
pub fn scaled_dot_product_attention(
    query: &CudaTensor,
    key: &CudaTensor,
    value: &CudaTensor,
    scale: Option<f32>,
) -> Result<CudaTensor> {
    super::nn::scaled_dot_product_attention(query, key, value, scale)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ElemBinary {
    Add,
    Mul,
    Sub,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ElemUnary {
    Silu,
    GeluTanh,
}

/// `bcast_binary` op codes (see the kernel).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BcastOp {
    Add = 0,
    Sub = 1,
    Mul = 2,
    Div = 3,
    RSub = 4,
    RDiv = 5,
}

impl BcastOp {
    pub fn apply(self, x: f32, y: f32) -> f32 {
        match self {
            BcastOp::Add => x + y,
            BcastOp::Sub => x - y,
            BcastOp::Mul => x * y,
            BcastOp::Div => x / y,
            BcastOp::RSub => y - x,
            BcastOp::RDiv => y / x,
        }
    }
}

#[cfg(feature = "cuda")]
fn err(e: impl std::fmt::Display) -> TensorError {
    TensorError::Message(e.to_string())
}

#[cfg(feature = "cuda")]
fn ctx() -> Result<Arc<DeviceContext>> {
    device::global_device().ok_or_else(|| TensorError::Message("no global CUDA device context".into()))
}

/// Uninitialized device buffer. Every caller fully overwrites it.
#[cfg(feature = "cuda")]
pub fn alloc(n: usize) -> Result<CudaSlice<f32>> {
    let dev = ctx()?;
    unsafe { dev.stream.alloc::<f32>(n) }.map_err(err)
}

#[cfg(feature = "cuda")]
fn check(what: &str, ok: bool) -> Result<()> {
    if ok {
        Ok(())
    } else {
        Err(TensorError::Message(format!("{what}: buffer size mismatch")))
    }
}

#[cfg(feature = "cuda")]
pub fn elem_binary_device(a: &CudaSlice<f32>, b: &CudaSlice<f32>, kind: ElemBinary) -> Result<CudaSlice<f32>> {
    check("elem_binary", a.len() == b.len())?;
    let dev = ctx()?;
    let n = a.len() as i64;
    let mut out = alloc(a.len())?;
    let f = match kind {
        ElemBinary::Add => &dev.kernels.elem_add,
        ElemBinary::Mul => &dev.kernels.elem_mul,
        ElemBinary::Sub => &dev.kernels.elem_sub,
    };
    launch!(dev.stream, f, cfg_n(a.len()); a, b, &mut out, &n).map_err(err)?;
    Ok(out)
}

#[cfg(feature = "cuda")]
pub fn unary_device(a: &CudaSlice<f32>, kind: ElemUnary) -> Result<CudaSlice<f32>> {
    let dev = ctx()?;
    let n = a.len() as i64;
    let mut out = alloc(a.len())?;
    let f = match kind {
        ElemUnary::Silu => &dev.kernels.silu,
        ElemUnary::GeluTanh => &dev.kernels.gelu_tanh,
    };
    launch!(dev.stream, f, cfg_n(a.len()); a, &mut out, &n).map_err(err)?;
    Ok(out)
}

#[cfg(feature = "cuda")]
pub fn mul_scalar_device(a: &CudaSlice<f32>, s: f32) -> Result<CudaSlice<f32>> {
    let dev = ctx()?;
    let n = a.len() as i64;
    let mut out = alloc(a.len())?;
    launch!(dev.stream, &dev.kernels.mul_scalar, cfg_n(a.len()); a, &s, &mut out, &n).map_err(err)?;
    Ok(out)
}

#[cfg(feature = "cuda")]
pub fn add_scalar_device(a: &CudaSlice<f32>, s: f32) -> Result<CudaSlice<f32>> {
    let dev = ctx()?;
    let n = a.len() as i64;
    let mut out = alloc(a.len())?;
    launch!(dev.stream, &dev.kernels.add_scalar, cfg_n(a.len()); a, &s, &mut out, &n).map_err(err)?;
    Ok(out)
}

#[cfg(feature = "cuda")]
pub fn clamp_device(a: &CudaSlice<f32>, lo: f32, hi: f32) -> Result<CudaSlice<f32>> {
    let dev = ctx()?;
    let n = a.len() as i64;
    let mut out = alloc(a.len())?;
    launch!(dev.stream, &dev.kernels.clamp_f, cfg_n(a.len()); a, &lo, &hi, &mut out, &n).map_err(err)?;
    Ok(out)
}

#[cfg(feature = "cuda")]
pub fn fill_device(n: usize, v: f32) -> Result<CudaSlice<f32>> {
    let dev = ctx()?;
    let n_i = n as i64;
    let mut out = alloc(n)?;
    launch!(dev.stream, &dev.kernels.fill_f, cfg_n(n); &mut out, &v, &n_i).map_err(err)?;
    Ok(out)
}

/// `Σ coef·x` over up to three buffers per launch (longer lists chain).
#[cfg(feature = "cuda")]
pub fn lincomb_device(terms: &[(f32, &CudaSlice<f32>)]) -> Result<CudaSlice<f32>> {
    let Some(&(_, first)) = terms.first() else {
        return Err(TensorError::Message("lincomb: no terms".into()));
    };
    let len = first.len();
    check("lincomb", terms.iter().all(|(_, t)| t.len() == len))?;
    let dev = ctx()?;
    let n = len as i64;
    let pick = |i: usize| -> (f32, &CudaSlice<f32>) { terms.get(i).map(|&(c, t)| (c, t)).unwrap_or((0.0, first)) };
    let (a, x) = pick(0);
    let (b, y) = pick(1);
    let (c, z) = pick(2);
    let mut acc = alloc(len)?;
    launch!(dev.stream, &dev.kernels.lincomb3, cfg_n(len); x, y, z, &mut acc, &a, &b, &c, &n).map_err(err)?;
    let mut i = 3;
    while i < terms.len() {
        let (b, y) = pick(i);
        let (c, z) = pick(i + 1);
        let mut next = alloc(len)?;
        let one = 1.0f32;
        launch!(dev.stream, &dev.kernels.lincomb3, cfg_n(len); &acc, y, z, &mut next, &one, &b, &c, &n).map_err(err)?;
        acc = next;
        i += 2;
    }
    Ok(acc)
}

#[cfg(feature = "cuda")]
pub fn bcast_binary_device(
    big: &CudaSlice<f32>,
    small: &CudaSlice<f32>,
    inner: usize,
    period: usize,
    op: BcastOp,
) -> Result<CudaSlice<f32>> {
    check("bcast_binary", small.len() == period && inner > 0 && period > 0 && big.len() % (inner * period) == 0)?;
    let dev = ctx()?;
    let (n, inner_i, period_i, op_i) = (big.len() as i64, inner as i64, period as i64, op as i32);
    let mut out = alloc(big.len())?;
    launch!(dev.stream, &dev.kernels.bcast_binary, cfg_n(big.len()); big, small, &mut out, &n, &inner_i, &period_i, &op_i)
        .map_err(err)?;
    Ok(out)
}

/// `out[i] += bias[(i / inner) % bias.len()]` in place.
#[cfg(feature = "cuda")]
pub fn add_bias_inplace_device(out: &mut CudaSlice<f32>, bias: &CudaSlice<f32>, inner: usize) -> Result<()> {
    check("add_bias", !bias.is_empty() && inner > 0 && out.len() % (inner * bias.len()) == 0)?;
    let dev = ctx()?;
    let (n, inner_i, period) = (out.len() as i64, inner as i64, bias.len() as i64);
    let cfg = cfg_n(out.len());
    launch!(dev.stream, &dev.kernels.add_bias_inplace, cfg; out, bias, &n, &inner_i, &period).map_err(err)
}

/// `x = gelu_tanh(x + bias[i % width])` in place.
#[cfg(feature = "cuda")]
pub fn bias_gelu_inplace_device(x: &mut CudaSlice<f32>, bias: &CudaSlice<f32>) -> Result<()> {
    check("bias_gelu", !bias.is_empty() && x.len() % bias.len() == 0)?;
    let dev = ctx()?;
    let (n, width) = (x.len() as i64, bias.len() as i64);
    let cfg = cfg_n(x.len());
    launch!(dev.stream, &dev.kernels.bias_gelu_inplace, cfg; x, bias, &n, &width).map_err(err)
}

/// f32 → bfloat16 (round to nearest).
#[cfg(feature = "cuda")]
pub fn cast_f32_bf16_device(a: &CudaSlice<f32>) -> Result<CudaSlice<half::bf16>> {
    let dev = ctx()?;
    let n = a.len() as i64;
    let mut out = unsafe { dev.stream.alloc::<half::bf16>(a.len().max(1)) }.map_err(err)?;
    launch!(dev.stream, &dev.kernels.cast_f32_bf16, cfg_n(a.len()); a, &mut out, &n).map_err(err)?;
    Ok(out)
}

/// bfloat16 → f32 with optional `bias[i % bias.len()]` and GELU-tanh.
#[cfg(feature = "cuda")]
pub fn cast_bf16_f32_bias_act_device(a: &CudaSlice<half::bf16>, bias: Option<&CudaSlice<f32>>, gelu: bool) -> Result<CudaSlice<f32>> {
    if let Some(b) = bias {
        check("cast_bf16_f32 bias", !b.is_empty() && a.len() % b.len() == 0)?;
    }
    let dev = ctx()?;
    let mut out = alloc(a.len().max(1))?;
    let (n, width) = (a.len() as i64, bias.map_or(1, |b| b.len()) as i64);
    let (has_bias, act) = (i32::from(bias.is_some()), i32::from(gelu));
    // Without a bias the kernel never reads it, but still needs a pointer to bind.
    let placeholder;
    let bias_arg = match bias {
        Some(b) => b,
        None => {
            placeholder = alloc(1)?;
            &placeholder
        }
    };
    launch!(dev.stream, &dev.kernels.cast_bf16_f32_bias_act, cfg_n(a.len());
        a, bias_arg, &mut out, &n, &width, &has_bias, &act)
    .map_err(err)?;
    Ok(out)
}

#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub fn residual_gate_add_e_device(
    h: &CudaSlice<f32>,
    a: &CudaSlice<f32>,
    e: &CudaSlice<f32>,
    batch: usize,
    seq: usize,
    dim: usize,
    e_rows: usize,
    slot: usize,
) -> Result<CudaSlice<f32>> {
    check("residual_gate_add_e", h.len() == batch * seq * dim && a.len() == h.len() && e.len() == batch * e_rows * dim)?;
    let dev = ctx()?;
    let (n, dim_i, seq_i, rows_i, slot_i) = (h.len() as i64, dim as i64, seq as i64, e_rows as i64, slot as i64);
    let mut out = alloc(h.len())?;
    launch!(dev.stream, &dev.kernels.residual_gate_add_e, cfg_n(h.len()); h, a, e, &mut out, &n, &dim_i, &seq_i, &rows_i, &slot_i)
        .map_err(err)?;
    Ok(out)
}

#[cfg(feature = "cuda")]
pub fn softmax_last_device(a: &CudaSlice<f32>, width: usize) -> Result<CudaSlice<f32>> {
    check("softmax_last", width > 0 && a.len() % width == 0)?;
    let dev = ctx()?;
    let rows = a.len() / width;
    let (rows_i, width_i) = (rows as i32, width as i32);
    let mut out = alloc(a.len())?;
    launch!(dev.stream, &dev.kernels.softmax_last, cfg_rows(rows); a, &mut out, &rows_i, &width_i).map_err(err)?;
    Ok(out)
}

/// [`softmax_last_device`] writing bfloat16 probabilities: half the bytes for
/// the attention `P@V` GEMM, and in fast mode the same math cuBLAS would do
/// internally anyway.
#[cfg(feature = "cuda")]
pub fn softmax_last_bf16_device(a: &CudaSlice<f32>, width: usize) -> Result<CudaSlice<half::bf16>> {
    check("softmax_last_bf16", width > 0 && a.len() % width == 0)?;
    let dev = ctx()?;
    let rows = a.len() / width;
    let (rows_i, width_i) = (rows as i32, width as i32);
    let mut out = unsafe { dev.stream.alloc::<half::bf16>(a.len().max(1)) }.map_err(err)?;
    launch!(dev.stream, &dev.kernels.softmax_last_bf16, cfg_rows(rows); a, &mut out, &rows_i, &width_i)
        .map_err(err)?;
    Ok(out)
}

#[cfg(feature = "cuda")]
pub fn rms_norm_last_device(a: &CudaSlice<f32>, weight: &CudaSlice<f32>, eps: f32) -> Result<CudaSlice<f32>> {
    let width = weight.len();
    check("rms_norm_last", width > 0 && a.len() % width == 0)?;
    let dev = ctx()?;
    let rows = a.len() / width;
    let (rows_i, width_i) = (rows as i32, width as i32);
    let mut out = alloc(a.len())?;
    launch!(dev.stream, &dev.kernels.rms_norm_last, cfg_rows(rows); a, weight, &mut out, &rows_i, &width_i, &eps)
        .map_err(err)?;
    Ok(out)
}

/// LayerNorm over the last dim; `weight`/`bias` both `None` for an unaffine norm.
#[cfg(feature = "cuda")]
pub fn layer_norm_last_device(
    a: &CudaSlice<f32>,
    affine: Option<(&CudaSlice<f32>, &CudaSlice<f32>)>,
    width: usize,
    eps: f32,
) -> Result<CudaSlice<f32>> {
    check("layer_norm_last", width > 0 && a.len() % width == 0)?;
    let dev = ctx()?;
    let rows = a.len() / width;
    let (rows_i, width_i) = (rows as i32, width as i32);
    let mut out = alloc(a.len())?;
    // Unaffine: the kernel never reads w/b, but still needs pointers to bind.
    let (w, b, has) = match affine {
        Some((w, b)) => (w, b, 1i32),
        None => (a, a, 0i32),
    };
    launch!(dev.stream, &dev.kernels.layer_norm_last, cfg_rows(rows); a, w, b, &mut out, &rows_i, &width_i, &eps, &has)
        .map_err(err)?;
    Ok(out)
}

#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub fn ln_adaln_e_device(
    x: &CudaSlice<f32>,
    e: &CudaSlice<f32>,
    batch: usize,
    seq: usize,
    dim: usize,
    e_rows: usize,
    scale_slot: usize,
    shift_slot: usize,
    eps: f32,
) -> Result<CudaSlice<f32>> {
    check("ln_adaln_e", x.len() == batch * seq * dim && e.len() == batch * e_rows * dim)?;
    let dev = ctx()?;
    let args = [batch as i32, seq as i32, dim as i32, e_rows as i32, scale_slot as i32, shift_slot as i32];
    let mut out = alloc(x.len())?;
    launch!(dev.stream, &dev.kernels.ln_adaln_e, cfg_rows(batch * seq);
        x, e, &mut out, &args[0], &args[1], &args[2], &args[3], &args[4], &args[5], &eps)
    .map_err(err)?;
    Ok(out)
}

/// q/k attention prep: RMSNorm over `heads*d`, optional RoPE, out in BHSD.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub fn qk_norm_rope_bhsd_device(
    src: &CudaSlice<f32>,
    weight: &CudaSlice<f32>,
    rope: Option<(&CudaSlice<f32>, &CudaSlice<f32>)>,
    batch: usize,
    seq: usize,
    heads: usize,
    d: usize,
    src_width: usize,
    col_off: usize,
    eps: f32,
) -> Result<CudaSlice<f32>> {
    let width = heads * d;
    check(
        "qk_norm_rope_bhsd",
        src.len() == batch * seq * src_width && col_off + width <= src_width && weight.len() == width,
    )?;
    if let Some((c, s)) = rope {
        check("qk_norm_rope_bhsd rope", c.len() == seq * d && s.len() == seq * d && d % 2 == 0)?;
    }
    let dev = ctx()?;
    let (cos, sin, use_rope) = match rope {
        Some((c, s)) => (c, s, 1i32),
        None => (weight, weight, 0i32),
    };
    let args = [batch as i32, seq as i32, heads as i32, d as i32, src_width as i32, col_off as i32];
    let mut out = alloc(batch * heads * seq * d)?;
    launch!(dev.stream, &dev.kernels.qk_norm_rope_bhsd, cfg_rows(batch * seq);
        src, weight, cos, sin, &mut out, &args[0], &args[1], &args[2], &args[3], &args[4], &args[5], &eps, &use_rope)
    .map_err(err)?;
    Ok(out)
}

#[cfg(feature = "cuda")]
pub fn split_heads_bhsd_device(
    src: &CudaSlice<f32>,
    batch: usize,
    seq: usize,
    heads: usize,
    d: usize,
    src_width: usize,
    col_off: usize,
) -> Result<CudaSlice<f32>> {
    check("split_heads_bhsd", src.len() == batch * seq * src_width && col_off + heads * d <= src_width)?;
    let dev = ctx()?;
    let n = batch * heads * seq * d;
    let a = [n as i64, seq as i64, heads as i64, d as i64, src_width as i64, col_off as i64];
    let mut out = alloc(n)?;
    launch!(dev.stream, &dev.kernels.split_heads_bhsd, cfg_n(n); src, &mut out, &a[0], &a[1], &a[2], &a[3], &a[4], &a[5])
        .map_err(err)?;
    Ok(out)
}

#[cfg(feature = "cuda")]
pub fn merge_heads_device(src: &CudaSlice<f32>, batch: usize, heads: usize, seq: usize, d: usize) -> Result<CudaSlice<f32>> {
    check("merge_heads", src.len() == batch * heads * seq * d)?;
    let dev = ctx()?;
    let n = src.len();
    let a = [n as i64, seq as i64, heads as i64, d as i64];
    let mut out = alloc(n)?;
    launch!(dev.stream, &dev.kernels.merge_heads, cfg_n(n); src, &mut out, &a[0], &a[1], &a[2], &a[3]).map_err(err)?;
    Ok(out)
}

/// Strides of the input axis feeding each output axis of `perm`.
pub fn permute_strides(in_shape: &[usize], perm: &[usize]) -> (Vec<usize>, Vec<usize>) {
    let rank = in_shape.len();
    let mut in_strides = vec![1usize; rank];
    for i in (0..rank.saturating_sub(1)).rev() {
        in_strides[i] = in_strides[i + 1] * in_shape[i + 1];
    }
    let out_shape = perm.iter().map(|&p| in_shape[p]).collect();
    let mapped = perm.iter().map(|&p| in_strides[p]).collect();
    (out_shape, mapped)
}

/// N-D permute (rank ≤ 6) of a contiguous buffer.
#[cfg(feature = "cuda")]
pub fn gather_nd_device(src: &CudaSlice<f32>, in_shape: &[usize], perm: &[usize]) -> Result<CudaSlice<f32>> {
    let rank = in_shape.len();
    check("gather_nd", rank <= 6 && perm.len() == rank && src.len() == in_shape.iter().product::<usize>())?;
    let (out_shape, strides) = permute_strides(in_shape, perm);
    let dev = ctx()?;
    let mut s = [1i64; 6];
    let mut t = [0i64; 6];
    for k in 0..rank {
        s[k] = out_shape[k] as i64;
        t[k] = strides[k] as i64;
    }
    let (n, rank_i) = (src.len() as i64, rank as i32);
    let mut out = alloc(src.len())?;
    launch!(dev.stream, &dev.kernels.gather_nd, cfg_n(src.len());
        src, &mut out, &n, &rank_i, &s[0], &s[1], &s[2], &s[3], &s[4], &s[5], &t[0], &t[1], &t[2], &t[3], &t[4], &t[5])
    .map_err(err)?;
    Ok(out)
}

/// Contiguous block copy: for each outer index copy `len` floats with strides.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub fn block_copy_device(
    input: &CudaSlice<f32>,
    out: &mut CudaSlice<f32>,
    outer: usize,
    len: usize,
    in_stride: usize,
    out_stride: usize,
    in_offset: usize,
    out_offset: usize,
) -> Result<()> {
    if outer == 0 || len == 0 {
        return Ok(());
    }
    check(
        "block_copy",
        (outer - 1) * in_stride + in_offset + len <= input.len() && (outer - 1) * out_stride + out_offset + len <= out.len(),
    )?;
    let dev = ctx()?;
    let a = [outer as i64, len as i64, in_stride as i64, out_stride as i64, in_offset as i64, out_offset as i64];
    launch!(dev.stream, &dev.kernels.block_copy, cfg_n(outer * len); input, out, &a[0], &a[1], &a[2], &a[3], &a[4], &a[5])
        .map_err(err)
}

/// Nearest integer upsample of `[nc, h, w]` planes by `(fy, fx)`.
#[cfg(feature = "cuda")]
pub fn upsample_nearest_device(src: &CudaSlice<f32>, nc: usize, h: usize, w: usize, fy: usize, fx: usize) -> Result<CudaSlice<f32>> {
    check("upsample_nearest", src.len() == nc * h * w)?;
    let dev = ctx()?;
    let n = nc * h * fy * w * fx;
    let a = [n as i64, h as i64, w as i64, fy as i64, fx as i64];
    let mut out = alloc(n)?;
    launch!(dev.stream, &dev.kernels.upsample_nearest, cfg_n(n); src, &mut out, &a[0], &a[1], &a[2], &a[3], &a[4])
        .map_err(err)?;
    Ok(out)
}

/// Channel-axis RMS for `[n, c, spatial]`.
#[cfg(feature = "cuda")]
pub fn rms_norm_channels_device(
    x: &CudaSlice<f32>,
    gamma: &CudaSlice<f32>,
    n: usize,
    c: usize,
    spatial: usize,
    eps: f32,
    silu: bool,
) -> Result<CudaSlice<f32>> {
    check("rms_norm_channels", gamma.len() == c && x.len() == n * c * spatial)?;
    let dev = ctx()?;
    let a = [n as i64, c as i64, spatial as i64];
    let mut out = alloc(x.len())?;
    let act = i32::from(silu);
    launch!(dev.stream, &dev.kernels.rms_norm_channels, cfg_n(n * spatial); x, gamma, &mut out, &a[0], &a[1], &a[2], &eps, &act)
        .map_err(err)?;
    Ok(out)
}

#[cfg(feature = "cuda")]
pub fn index_select_rows_device(table: &CudaSlice<f32>, d: usize, indices: &[u32]) -> Result<CudaSlice<f32>> {
    check("index_select_rows", d > 0 && table.len() % d == 0)?;
    let dev = ctx()?;
    let idx = dev.stream.memcpy_stod(indices).map_err(err)?;
    super::stats::record_h2d(indices.len());
    let n = indices.len() * d;
    let (n_i, d_i) = (n as i64, d as i64);
    let mut out = alloc(n)?;
    launch!(dev.stream, &dev.kernels.index_select_rows, cfg_n(n); table, &idx, &mut out, &n_i, &d_i).map_err(err)?;
    Ok(out)
}

// ---- Video Sparse Attention -------------------------------------------------

/// Device-side tiling geometry, uploaded once per latent grid and reused by
/// every VSA layer and step.
#[cfg(feature = "cuda")]
#[derive(Debug)]
pub struct VsaPlanDev {
    pub slot_src: CudaSlice<i32>,
    pub block_sizes: CudaSlice<i32>,
    pub num_tiles: usize,
    pub tile_elems: usize,
}

#[cfg(feature = "cuda")]
pub fn vsa_plan_upload(slot_src: &[i32], block_sizes: &[u32], tile_elems: usize) -> Result<VsaPlanDev> {
    let dev = ctx()?;
    let sizes: Vec<i32> = block_sizes.iter().map(|&n| n as i32).collect();
    let out = VsaPlanDev {
        slot_src: dev.stream.memcpy_stod(slot_src).map_err(err)?,
        block_sizes: dev.stream.memcpy_stod(&sizes).map_err(err)?,
        num_tiles: block_sizes.len(),
        tile_elems,
    };
    super::stats::record_h2d(slot_src.len() + sizes.len());
    Ok(out)
}

/// Per-tile means of a token-indexed `[bh, seq, dim]` tensor.
#[cfg(feature = "cuda")]
pub fn vsa_tile_mean_device(
    x: &CudaSlice<f32>,
    plan: &VsaPlanDev,
    bh: usize,
    seq: usize,
    dim: usize,
) -> Result<CudaSlice<f32>> {
    check("vsa_tile_mean", x.len() == bh * seq * dim && dim > 0)?;
    let dev = ctx()?;
    let mut out = alloc(bh * plan.num_tiles * dim)?;
    let cfg = LaunchConfig {
        grid_dim: (plan.num_tiles as u32, bh as u32, 1),
        block_dim: (dim.min(256) as u32, 1, 1),
        shared_mem_bytes: 0,
    };
    let (seq_i, dim_i) = (seq as i64, dim as i32);
    let (nt, te) = (plan.num_tiles as i32, plan.tile_elems as i32);
    launch!(dev.stream, &dev.kernels.vsa_tile_mean, cfg;
        x, &plan.slot_src, &plan.block_sizes, &mut out, &seq_i, &dim_i, &nt, &te)
    .map_err(err)?;
    Ok(out)
}

/// Top-k column indices per score row, `[rows, k]`.
#[cfg(feature = "cuda")]
pub fn vsa_topk_device(scores: &CudaSlice<f32>, rows: usize, n: usize, k: usize) -> Result<CudaSlice<u32>> {
    check("vsa_topk", n > 0 && k > 0 && k <= n && scores.len() == rows * n)?;
    let dev = ctx()?;
    let mut out = unsafe { dev.stream.alloc::<u32>((rows * k).max(1)) }.map_err(err)?;
    const THREADS: u32 = 256;
    let cfg = LaunchConfig {
        grid_dim: (rows as u32, 1, 1),
        block_dim: (THREADS, 1, 1),
        shared_mem_bytes: THREADS * std::mem::size_of::<i32>() as u32,
    };
    let (rows_i, n_i, k_i) = (rows as i32, n as i32, k as i32);
    launch!(dev.stream, &dev.kernels.vsa_topk, cfg; scores, &mut out, &rows_i, &n_i, &k_i).map_err(err)?;
    Ok(out)
}

/// Gather the selected tiles' K/V rows into `[bh, group, topk*tile, dim]` bf16.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub fn vsa_gather_kv_device(
    k: &CudaSlice<f32>,
    v: &CudaSlice<f32>,
    selected: &CudaSlice<u32>,
    plan: &VsaPlanDev,
    bh: usize,
    group: usize,
    seq: usize,
    dim: usize,
    topk: usize,
    q_base: usize,
) -> Result<(CudaSlice<half::bf16>, CudaSlice<half::bf16>)> {
    let dev = ctx()?;
    let len = topk * plan.tile_elems;
    let n = bh * group * len * dim;
    let mut kg = unsafe { dev.stream.alloc::<half::bf16>(n.max(1)) }.map_err(err)?;
    let mut vg = unsafe { dev.stream.alloc::<half::bf16>(n.max(1)) }.map_err(err)?;
    // One thread row per gathered slot; x covers dim.
    let rows_per_block = (256 / dim.min(128)).max(1);
    let cfg = LaunchConfig {
        grid_dim: (len.div_ceil(rows_per_block) as u32, group as u32, bh as u32),
        block_dim: (dim.min(128) as u32, rows_per_block as u32, 1),
        shared_mem_bytes: 0,
    };
    let (seq_i, dim_i) = (seq as i64, dim as i32);
    let (tk, te, qb, nt) = (topk as i32, plan.tile_elems as i32, q_base as i32, plan.num_tiles as i32);
    launch!(dev.stream, &dev.kernels.vsa_gather_kv, cfg;
        k, v, selected, &plan.slot_src, &mut kg, &mut vg, &seq_i, &dim_i, &tk, &te, &qb, &nt)
    .map_err(err)?;
    Ok((kg, vg))
}

/// Gather a group of query tiles into `[bh, group, tile, dim]` bf16, in padded
/// slot order. Padding slots are zero; `vsa_combine` drops their outputs.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub fn vsa_gather_q_device(
    q: &CudaSlice<f32>,
    plan: &VsaPlanDev,
    bh: usize,
    group: usize,
    q_base: usize,
    seq: usize,
    dim: usize,
) -> Result<CudaSlice<half::bf16>> {
    let dev = ctx()?;
    let n = bh * group * plan.tile_elems * dim;
    let mut out = unsafe { dev.stream.alloc::<half::bf16>(n.max(1)) }.map_err(err)?;
    let rows_per_block = (256 / dim.min(128)).max(1);
    let cfg = LaunchConfig {
        grid_dim: (plan.tile_elems.div_ceil(rows_per_block) as u32, group as u32, bh as u32),
        block_dim: (dim.min(128) as u32, rows_per_block as u32, 1),
        shared_mem_bytes: 0,
    };
    let (seq_i, dim_i) = (seq as i64, dim as i32);
    let (te, qb) = (plan.tile_elems as i32, q_base as i32);
    launch!(dev.stream, &dev.kernels.vsa_gather_q, cfg;
        q, &plan.slot_src, &mut out, &seq_i, &dim_i, &te, &qb)
    .map_err(err)?;
    Ok(out)
}

/// `-inf` the score columns that fall on tile padding.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub fn vsa_mask_pad_device(
    scores: &mut CudaSlice<f32>,
    selected: &CudaSlice<u32>,
    plan: &VsaPlanDev,
    bh: usize,
    group: usize,
    rows_per_tile: usize,
    topk: usize,
    q_base: usize,
) -> Result<()> {
    let dev = ctx()?;
    let len = topk * plan.tile_elems;
    const THREADS: u32 = 128;
    let cfg = LaunchConfig {
        grid_dim: (len.div_ceil(THREADS as usize) as u32, group as u32, bh as u32),
        block_dim: (THREADS, 1, 1),
        shared_mem_bytes: 0,
    };
    let (rpt, tk, te) = (rows_per_tile as i32, topk as i32, plan.tile_elems as i32);
    let (qb, nt) = (q_base as i32, plan.num_tiles as i32);
    launch!(dev.stream, &dev.kernels.vsa_mask_pad, cfg;
        scores, selected, &plan.block_sizes, &rpt, &tk, &te, &qb, &nt)
    .map_err(err)?;
    Ok(())
}

/// Fused block-sparse attention over every query tile at once. Writes the fine
/// stage's output in padded slot order, `[bh, num_tiles, tile, dim]`, for
/// [`vsa_combine_device`] to scatter.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub fn vsa_fused_attn_device(
    q: &CudaSlice<f32>,
    k: &CudaSlice<f32>,
    v: &CudaSlice<f32>,
    selected: &CudaSlice<u32>,
    plan: &VsaPlanDev,
    bh: usize,
    seq: usize,
    dim: usize,
    topk: usize,
    scale: f32,
) -> Result<CudaSlice<f32>> {
    const THREADS: u32 = 256;
    const HALF: usize = 32;
    check("vsa_fused_attn", dim % 128 == 0 || dim == 64 || dim == 128)?;
    let dev = ctx()?;
    let nb = plan.num_tiles;
    let mut out = alloc(bh * nb * plan.tile_elems * dim)?;
    // Q tile + one K/V half tile in bf16, plus the probability tile in f32.
    let shared = (plan.tile_elems + 2 * HALF) * dim * 2 + plan.tile_elems * HALF * 4;
    let cfg = LaunchConfig {
        grid_dim: (nb as u32, bh as u32, 1),
        block_dim: (THREADS, 1, 1),
        shared_mem_bytes: shared as u32,
    };
    let (seq_i, dim_i) = (seq as i64, dim as i32);
    let (tk, nt) = (topk as i32, nb as i32);
    launch!(dev.stream, &dev.kernels.vsa_fused_attn, cfg;
        q, k, v, selected, &plan.slot_src, &plan.block_sizes, &mut out, &seq_i, &dim_i, &tk, &nt, &scale)
    .map_err(err)?;
    Ok(out)
}

/// Scatter `coarse * gate + sparse` back into token order.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub fn vsa_combine_device(
    sparse: &CudaSlice<f32>,
    coarse: &CudaSlice<f32>,
    gate: Option<&CudaSlice<f32>>,
    plan: &VsaPlanDev,
    out: &mut CudaSlice<f32>,
    bh: usize,
    group: usize,
    q_base: usize,
    seq: usize,
    dim: usize,
) -> Result<()> {
    let dev = ctx()?;
    let rows_per_block = (256 / dim.min(128)).max(1);
    let cfg = LaunchConfig {
        grid_dim: (plan.tile_elems.div_ceil(rows_per_block) as u32, group as u32, bh as u32),
        block_dim: (dim.min(128) as u32, rows_per_block as u32, 1),
        shared_mem_bytes: 0,
    };
    let (seq_i, dim_i) = (seq as i64, dim as i32);
    let (te, qb, nt) = (plan.tile_elems as i32, q_base as i32, plan.num_tiles as i32);
    // The kernel never reads a null gate, but a launch argument still needs a
    // pointer to bind, so pass a one-element placeholder.
    let placeholder;
    let gate_ref = match gate {
        Some(g) => g,
        None => {
            placeholder = alloc(1)?;
            &placeholder
        }
    };
    let has_gate = i32::from(gate.is_some());
    launch!(dev.stream, &dev.kernels.vsa_combine, cfg;
        sparse, coarse, gate_ref, &plan.slot_src, out, &seq_i, &dim_i, &te, &qb, &nt, &has_gate)
    .map_err(err)?;
    Ok(())
}

/// Plain-Rust twins of the device kernels. These are the CPU path, so they
/// are written for clarity of the math first and then parallelized where the
/// result does not depend on evaluation order.
pub mod host {
    use rayon::prelude::*;

    use super::BcastOp;

    const PAR_MIN: usize = 1 << 15;

    pub fn map2(a: &[f32], b: &[f32], f: impl Fn(f32, f32) -> f32 + Sync) -> Vec<f32> {
        let mut out = vec![0.0; a.len()];
        if a.len() >= PAR_MIN {
            out.par_iter_mut().zip(a.par_iter().zip(b.par_iter())).for_each(|(o, (&x, &y))| *o = f(x, y));
        } else {
            out.iter_mut().zip(a.iter().zip(b)).for_each(|(o, (&x, &y))| *o = f(x, y));
        }
        out
    }

    pub fn map1(a: &[f32], f: impl Fn(f32) -> f32 + Sync) -> Vec<f32> {
        if a.len() >= PAR_MIN {
            a.par_iter().map(|&x| f(x)).collect()
        } else {
            a.iter().map(|&x| f(x)).collect()
        }
    }

    pub fn silu(x: f32) -> f32 {
        x / (1.0 + (-x).exp())
    }

    pub fn gelu_tanh(x: f32) -> f32 {
        let c = (2.0 / std::f32::consts::PI).sqrt();
        0.5 * x * (1.0 + (c * (x + 0.044715 * x * x * x)).tanh())
    }

    pub fn lincomb(terms: &[(f32, &[f32])]) -> Vec<f32> {
        let len = terms.first().map(|t| t.1.len()).unwrap_or(0);
        let mut out = vec![0.0f32; len];
        let fill = |i: usize, o: &mut f32| {
            let mut acc = 0.0f32;
            for (c, t) in terms {
                acc += c * t[i];
            }
            *o = acc;
        };
        if len >= PAR_MIN {
            out.par_iter_mut().enumerate().for_each(|(i, o)| fill(i, o));
        } else {
            out.iter_mut().enumerate().for_each(|(i, o)| fill(i, o));
        }
        out
    }

    pub fn bcast_binary(big: &[f32], small: &[f32], inner: usize, period: usize, op: BcastOp) -> Vec<f32> {
        let mut out = vec![0.0; big.len()];
        let f = |i: usize, o: &mut f32| *o = op.apply(big[i], small[(i / inner) % period]);
        if big.len() >= PAR_MIN {
            out.par_iter_mut().enumerate().for_each(|(i, o)| f(i, o));
        } else {
            out.iter_mut().enumerate().for_each(|(i, o)| f(i, o));
        }
        out
    }

    pub fn softmax_last(x: &[f32], width: usize) -> Vec<f32> {
        let mut out = vec![0.0; x.len()];
        out.par_chunks_mut(width).zip(x.par_chunks(width)).for_each(|(o, row)| {
            let m = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            for (oi, &v) in o.iter_mut().zip(row) {
                *oi = (v - m).exp();
                sum += *oi;
            }
            let inv = 1.0 / sum;
            o.iter_mut().for_each(|v| *v *= inv);
        });
        out
    }

    pub fn rms_norm_last(x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
        let width = w.len();
        let mut out = vec![0.0; x.len()];
        out.par_chunks_mut(width).zip(x.par_chunks(width)).for_each(|(o, row)| {
            let ms = row.iter().map(|v| v * v).sum::<f32>() / width as f32;
            let inv = 1.0 / (ms + eps).sqrt();
            for i in 0..width {
                o[i] = row[i] * inv * w[i];
            }
        });
        out
    }

    pub fn layer_norm_last(x: &[f32], width: usize, affine: Option<(&[f32], &[f32])>, eps: f32) -> Vec<f32> {
        let mut out = vec![0.0; x.len()];
        out.par_chunks_mut(width).zip(x.par_chunks(width)).for_each(|(o, row)| {
            let mean = row.iter().sum::<f32>() / width as f32;
            let var = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / width as f32;
            let inv = 1.0 / (var + eps).sqrt();
            for i in 0..width {
                let y = (row[i] - mean) * inv;
                o[i] = match affine {
                    Some((w, b)) => y * w[i] + b[i],
                    None => y,
                };
            }
        });
        out
    }

    #[allow(clippy::too_many_arguments)]
    pub fn ln_adaln_e(x: &[f32], e: &[f32], seq: usize, dim: usize, e_rows: usize, scale_slot: usize, shift_slot: usize, eps: f32) -> Vec<f32> {
        let mut out = vec![0.0; x.len()];
        out.par_chunks_mut(dim).zip(x.par_chunks(dim)).enumerate().for_each(|(row, (o, src))| {
            let b = row / seq;
            let sc = &e[(b * e_rows + scale_slot) * dim..][..dim];
            let sh = &e[(b * e_rows + shift_slot) * dim..][..dim];
            let mean = src.iter().sum::<f32>() / dim as f32;
            let var = src.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / dim as f32;
            let inv = 1.0 / (var + eps).sqrt();
            for j in 0..dim {
                o[j] = (src[j] - mean) * inv * (1.0 + sc[j]) + sh[j];
            }
        });
        out
    }

    #[allow(clippy::too_many_arguments)]
    pub fn residual_gate_add_e(h: &[f32], a: &[f32], e: &[f32], seq: usize, dim: usize, e_rows: usize, slot: usize) -> Vec<f32> {
        let mut out = vec![0.0; h.len()];
        out.par_chunks_mut(dim).enumerate().for_each(|(row, o)| {
            let g = &e[((row / seq) * e_rows + slot) * dim..][..dim];
            for j in 0..dim {
                o[j] = h[row * dim + j] + a[row * dim + j] * g[j];
            }
        });
        out
    }

    #[allow(clippy::too_many_arguments)]
    pub fn qk_norm_rope_bhsd(
        src: &[f32],
        w: &[f32],
        rope: Option<(&[f32], &[f32])>,
        batch: usize,
        seq: usize,
        heads: usize,
        d: usize,
        src_width: usize,
        col_off: usize,
        eps: f32,
    ) -> Vec<f32> {
        let width = heads * d;
        let mut out = vec![0.0f32; batch * heads * seq * d];
        // One output plane per (b, h); rows normalized independently.
        out.par_chunks_mut(seq * d).enumerate().for_each(|(bh, plane)| {
            let (b, h) = (bh / heads, bh % heads);
            for s in 0..seq {
                let x = &src[(b * seq + s) * src_width + col_off..][..width];
                let ms = x.iter().map(|v| v * v).sum::<f32>() / width as f32;
                let inv = 1.0 / (ms + eps).sqrt();
                let o = &mut plane[s * d..(s + 1) * d];
                for p in 0..d {
                    let j = h * d + p;
                    o[p] = match rope {
                        None => x[j] * inv * w[j],
                        Some((cos, sin)) => {
                            let even = p - (p & 1);
                            let j0 = h * d + even;
                            let x1 = x[j0] * inv * w[j0];
                            let x2 = x[j0 + 1] * inv * w[j0 + 1];
                            let c = cos[s * d + even];
                            let sn = sin[s * d + even + 1];
                            if p & 1 == 1 {
                                x1 * sn + x2 * c
                            } else {
                                x1 * c - x2 * sn
                            }
                        }
                    };
                }
            }
        });
        out
    }

    pub fn split_heads_bhsd(src: &[f32], batch: usize, seq: usize, heads: usize, d: usize, src_width: usize, col_off: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; batch * heads * seq * d];
        out.par_chunks_mut(seq * d).enumerate().for_each(|(bh, plane)| {
            let (b, h) = (bh / heads, bh % heads);
            for s in 0..seq {
                let base = (b * seq + s) * src_width + col_off + h * d;
                plane[s * d..(s + 1) * d].copy_from_slice(&src[base..base + d]);
            }
        });
        out
    }

    pub fn merge_heads(src: &[f32], batch: usize, heads: usize, seq: usize, d: usize) -> Vec<f32> {
        let hd = heads * d;
        let mut out = vec![0.0f32; batch * seq * hd];
        out.par_chunks_mut(hd).enumerate().for_each(|(row, o)| {
            let (b, s) = (row / seq, row % seq);
            for h in 0..heads {
                let base = ((b * heads + h) * seq + s) * d;
                o[h * d..(h + 1) * d].copy_from_slice(&src[base..base + d]);
            }
        });
        out
    }

    pub fn permute(src: &[f32], in_shape: &[usize], perm: &[usize]) -> Vec<f32> {
        let (out_shape, strides) = super::permute_strides(in_shape, perm);
        let rank = in_shape.len();
        let last = out_shape.last().copied().unwrap_or(1).max(1);
        let mut out = vec![0.0f32; src.len()];
        out.par_chunks_mut(last).enumerate().for_each(|(row, o)| {
            // Unravel the row index over all but the last output axis once.
            let mut rem = row;
            let mut base = 0usize;
            for k in (0..rank.saturating_sub(1)).rev() {
                base += (rem % out_shape[k]) * strides[k];
                rem /= out_shape[k];
            }
            let step = if rank == 0 { 0 } else { strides[rank - 1] };
            for (j, v) in o.iter_mut().enumerate() {
                *v = src[base + j * step];
            }
        });
        out
    }

    pub fn upsample_nearest(src: &[f32], nc: usize, h: usize, w: usize, fy: usize, fx: usize) -> Vec<f32> {
        let (oh, ow) = (h * fy, w * fx);
        let mut out = vec![0.0f32; nc * oh * ow];
        out.par_chunks_mut(oh * ow).enumerate().for_each(|(c, plane)| {
            for y in 0..oh {
                let row = &src[(c * h + y / fy) * w..][..w];
                for x in 0..ow {
                    plane[y * ow + x] = row[x / fx];
                }
            }
        });
        out
    }

    pub fn rms_norm_channels(x: &[f32], gamma: &[f32], n: usize, c: usize, spatial: usize, eps: f32, silu: bool) -> Vec<f32> {
        let mut out = vec![0.0f32; x.len()];
        for ni in 0..n {
            let (inv_all, ()) = {
                let inv: Vec<f32> = (0..spatial)
                    .into_par_iter()
                    .map(|s| {
                        let acc: f32 = (0..c).map(|ci| x[(ni * c + ci) * spatial + s].powi(2)).sum();
                        1.0 / (acc / c as f32 + eps).sqrt()
                    })
                    .collect();
                (inv, ())
            };
            out[ni * c * spatial..(ni + 1) * c * spatial]
                .par_chunks_mut(spatial)
                .enumerate()
                .for_each(|(ci, plane)| {
                    let src = &x[(ni * c + ci) * spatial..][..spatial];
                    for s in 0..spatial {
                        let v = src[s] * inv_all[s] * gamma[ci];
                        plane[s] = if silu { v / (1.0 + (-v).exp()) } else { v };
                    }
                });
        }
        out
    }

    /// Cross-correlation of `[n, c, h, w]` with `[oc, c, kh, kw]` (symmetric
    /// zero padding), one output plane per (batch, out-channel) in parallel.
    #[allow(clippy::too_many_arguments)]
    pub fn conv2d(x: &[f32], w: &[f32], n: usize, c: usize, h: usize, wd: usize, oc: usize, kh: usize, kw: usize, pad: [usize; 2], stride: [usize; 2]) -> (Vec<f32>, usize, usize) {
        let oh = (h + 2 * pad[0] - kh) / stride[0] + 1;
        let ow = (wd + 2 * pad[1] - kw) / stride[1] + 1;
        let mut out = vec![0.0f32; n * oc * oh * ow];
        out.par_chunks_mut((oh * ow).max(1)).enumerate().for_each(|(p, plane)| {
            let (ni, o) = (p / oc, p % oc);
            for y in 0..oh {
                for xx in 0..ow {
                    let mut acc = 0.0f32;
                    for ci in 0..c {
                        for dy in 0..kh {
                            let iy = y * stride[0] + dy;
                            if iy < pad[0] || iy - pad[0] >= h {
                                continue;
                            }
                            let iy = iy - pad[0];
                            for dx in 0..kw {
                                let ix = xx * stride[1] + dx;
                                if ix < pad[1] || ix - pad[1] >= wd {
                                    continue;
                                }
                                let ix = ix - pad[1];
                                acc += x[((ni * c + ci) * h + iy) * wd + ix] * w[((o * c + ci) * kh + dy) * kw + dx];
                            }
                        }
                    }
                    plane[y * ow + xx] = acc;
                }
            }
        });
        (out, oh, ow)
    }

    /// Temporal unfold `[n, c, t, h, w]` → `[n*ot, c*kt, h, w]`.
    #[allow(clippy::too_many_arguments)]
    pub fn temporal_unfold(x: &[f32], n: usize, c: usize, t: usize, h: usize, w: usize, kt: usize, st: usize) -> (Vec<f32>, usize) {
        let ot = (t - kt) / st + 1;
        let plane = h * w;
        let mut out = vec![0.0f32; n * ot * c * kt * plane];
        out.par_chunks_mut(plane).enumerate().for_each(|(p, o)| {
            let ck = p % (c * kt);
            let bo = p / (c * kt);
            let (cc, k) = (ck / kt, ck % kt);
            let (b, oi) = (bo / ot, bo % ot);
            let src = (((b * c + cc) * t + oi * st + k) * plane)..;
            o.copy_from_slice(&x[src][..plane]);
        });
        (out, ot)
    }
}
