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
    GeluErf,
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
        ElemUnary::GeluErf => &dev.kernels.gelu_erf,
    };
    launch!(dev.stream, f, cfg_n(a.len()); a, &mut out, &n).map_err(err)?;
    Ok(out)
}

/// Last dim `(v, g)` → `v * silu(g)`. `x.len()` is `rows * 2 * half`.
#[cfg(feature = "cuda")]
pub fn swiglu_value_first_device(x: &CudaSlice<f32>, half: usize) -> Result<CudaSlice<f32>> {
    check("swiglu_value_first", half > 0 && x.len() % (2 * half) == 0)?;
    let dev = ctx()?;
    let n = x.len() / 2;
    let half_i = half as i64;
    let n_i = n as i64;
    let mut out = alloc(n)?;
    launch!(dev.stream, &dev.kernels.swiglu_value_first, cfg_n(n); x, &mut out, &half_i, &n_i).map_err(err)?;
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
    // The kernel splits dim across 32 lanes with at most four each.
    check("vsa_fused_attn dim", dim % 32 == 0 && dim / 32 <= 4 && dim > 0)?;
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

/// The reference's `tile()`: one f32 `[bh, seq, dim]` tensor into
/// tile-contiguous bf16 `[bh, padded, dim]`, padding slots zeroed.
#[cfg(feature = "cuda")]
pub fn vsa_tile_qkv_device(
    x: &CudaSlice<f32>,
    plan: &VsaPlanDev,
    bh: usize,
    seq: usize,
    dim: usize,
) -> Result<CudaSlice<half::bf16>> {
    let dev = ctx()?;
    let padded = plan.num_tiles * plan.tile_elems;
    let total = bh * padded * dim;
    let mut out = unsafe { dev.stream.alloc::<half::bf16>(total.max(1)) }.map_err(err)?;
    let mut cfg = cfg_n(padded * dim);
    cfg.grid_dim.2 = bh as u32; // one z-slice per head; grid.x covers padded x dim
    let (seq_i, padded_i, dim_i) = (seq as i64, padded as i64, dim as i32);
    launch!(dev.stream, &dev.kernels.vsa_tile_qkv, cfg; x, &plan.slot_src, &mut out, &seq_i, &padded_i, &dim_i)
        .map_err(err)?;
    Ok(out)
}

/// Fine stage on tensor cores: tile Q/K/V once, then one CUDA block per query
/// tile streams its top-k key tiles through `mma.sync` with an online softmax
/// — the structure every reference block-sparse kernel uses. Output is f32 in
/// tile-slot order, the same contract as [`vsa_fused_attn_device`].
///
/// `q_base` / `q_tiles` restrict the grid to a slice of query tiles. H3 uses
/// this to skip prefix tiles that `vsa_h3_4_prefix_dense` overwrites. The
/// skipped slots stay zero so [`vsa_combine_device`] can still walk every tile.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub fn vsa_mma_attn_device(
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
    vsa_mma_attn_range_device(q, k, v, selected, plan, bh, seq, dim, topk, scale, 0, plan.num_tiles)
}

/// Like [`vsa_mma_attn_device`], but only query tiles `[q_base, q_base + q_tiles)`.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub fn vsa_mma_attn_range_device(
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
    q_base: usize,
    q_tiles: usize,
) -> Result<CudaSlice<f32>> {
    use crate::wan::stats::phase;
    const THREADS: u32 = 128;
    const TILE: usize = 64;
    const DIM: usize = 128;
    // Fragment maps and the swizzle are written for Wan's geometry.
    check("vsa_mma_attn geometry", dim == DIM && plan.tile_elems == TILE)?;
    let dev = ctx()?;
    check("vsa_mma_attn needs sm80+", dev.sm_major >= 8)?;
    let nb = plan.num_tiles;
    let q_tiles = q_tiles.min(nb.saturating_sub(q_base));
    check("vsa_mma_attn q range", q_tiles > 0 && q_base + q_tiles <= nb)?;
    let padded = nb * TILE;

    let (qt, kt, vt) = phase("vsa_mma_tile", || {
        Ok::<_, TensorError>((
            vsa_tile_qkv_device(q, plan, bh, seq, dim)?,
            vsa_tile_qkv_device(k, plan, bh, seq, dim)?,
            vsa_tile_qkv_device(v, plan, bh, seq, dim)?,
        ))
    })?;

    // K[2] + V[2] tiles of 64x128 bf16: 64 KiB, plus two mbarriers on the TMA
    // path. Past the 48 KiB default, so the function has to opt in.
    let shared_mma = (4 * TILE * DIM * 2) as u32;
    let shared_tma = shared_mma + 32;
    opt_in_dynamic_shared(&dev.kernels.vsa_mma_attn, shared_mma)?;

    // Zero-fill so skipped prefix tiles stay a defined 0 for combine.
    let mut out = if q_base == 0 && q_tiles == nb {
        alloc(bh * padded * dim)?
    } else {
        fill_device(bh * padded * dim, 0.0)?
    };
    let (nt, tk, qb) = (nb as i32, topk as i32, q_base as i32);
    let scale_log2 = scale * std::f32::consts::LOG2_E;
    let want_tma = tma_requested(dev.sm_major);
    if want_tma {
        match encode_qkv_panels(&qt, &kt, &vt, bh, padded) {
            Ok(maps) => {
                opt_in_dynamic_shared(&dev.kernels.vsa_mma_attn_tma, shared_tma)?;
                static LOGGED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
                crate::wan::log::info_once(&LOGGED, format_args!("vsa fine kernel: Tma (sm{}, 128B swizzle)", dev.sm_major));
                let cfg = LaunchConfig {
                    grid_dim: (q_tiles as u32, bh as u32, 1),
                    block_dim: (THREADS, 1, 1),
                    shared_mem_bytes: shared_tma,
                };
                launch!(dev.stream, &dev.kernels.vsa_mma_attn_tma, cfg;
                    &maps.tq0, &maps.tq1, &maps.tk0, &maps.tk1, &maps.tv0, &maps.tv1,
                    selected, &plan.block_sizes, &mut out, &nt, &tk, &scale_log2, &qb)
                .map_err(err)?;
                return Ok(out);
            }
            Err(e) => {
                static LOGGED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
                crate::wan::log::info_once(&LOGGED, format_args!("vsa TMA encode failed, using cp.async: {e}"));
            }
        }
    }
    let cfg = LaunchConfig {
        grid_dim: (q_tiles as u32, bh as u32, 1),
        block_dim: (THREADS, 1, 1),
        shared_mem_bytes: shared_mma,
    };
    launch!(dev.stream, &dev.kernels.vsa_mma_attn, cfg;
        &qt, &kt, &vt, selected, &plan.block_sizes, &mut out, &nt, &tk, &scale_log2, &qb)
    .map_err(err)?;
    Ok(out)
}

#[cfg(feature = "cuda")]
fn tma_requested(sm_major: i32) -> bool {
    use crate::wan::envflag::string_flag;
    if string_flag("FASTVIDEO_VSA_TMA", "1") == "0" {
        return false;
    }
    let pick = string_flag("FASTVIDEO_VSA_KERNEL", "auto");
    if pick == "mma" {
        return false; // explicit Ampere path for A/B
    }
    if pick == "tma" {
        return sm_major >= 9;
    }
    sm_major >= 9
}

#[cfg(feature = "cuda")]
fn opt_in_dynamic_shared(func: &cudarc::driver::CudaFunction, shared: u32) -> Result<()> {
    use cudarc::driver::sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES;
    // Once per function per process. Two kernels, two Once locks.
    // set_attribute is idempotent; a failed first call is the one we surface.
    if let Err(e) = func.set_attribute(CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, shared as i32) {
        return Err(TensorError::Message(format!("vsa_mma_attn: dynamic shared opt-in failed: {e}")));
    }
    Ok(())
}

/// 128-byte-aligned tensormap, passed by value (DeviceRepr) as a kernel arg.
#[cfg(feature = "cuda")]
#[repr(C, align(128))]
#[derive(Clone, Copy)]
struct FvTensorMap {
    opaque: [u64; 16],
}

#[cfg(feature = "cuda")]
unsafe impl cudarc::driver::DeviceRepr for FvTensorMap {}

#[cfg(feature = "cuda")]
struct QkvMaps {
    tq0: FvTensorMap,
    tq1: FvTensorMap,
    tk0: FvTensorMap,
    tk1: FvTensorMap,
    tv0: FvTensorMap,
    tv1: FvTensorMap,
}

#[cfg(feature = "cuda")]
fn encode_qkv_panels(
    q: &CudaSlice<half::bf16>,
    k: &CudaSlice<half::bf16>,
    v: &CudaSlice<half::bf16>,
    bh: usize,
    padded: usize,
) -> Result<QkvMaps> {
    use cudarc::driver::DevicePtr;
    let dev = ctx()?;
    let rows = (bh * padded) as u64;
    let (qp, _gq) = q.device_ptr(&dev.stream);
    let (kp, _gk) = k.device_ptr(&dev.stream);
    let (vp, _gv) = v.device_ptr(&dev.stream);
    Ok(QkvMaps {
        tq0: encode_bf16_panel(qp, rows, 0)?,
        tq1: encode_bf16_panel(qp, rows, 64)?,
        tk0: encode_bf16_panel(kp, rows, 0)?,
        tk1: encode_bf16_panel(kp, rows, 64)?,
        tv0: encode_bf16_panel(vp, rows, 0)?,
        tv1: encode_bf16_panel(vp, rows, 64)?,
    })
}

#[cfg(feature = "cuda")]
fn encode_bf16_panel(ptr: cudarc::driver::sys::CUdeviceptr, rows: u64, col0: u64) -> Result<FvTensorMap> {
    use cudarc::driver::sys::{
        self, CUtensorMapDataType, CUtensorMapFloatOOBfill, CUtensorMapInterleave, CUtensorMapL2promotion,
        CUtensorMapSwizzle,
    };
    check("tma panel rows", rows >= 64)?;
    let mut raw = std::mem::MaybeUninit::<sys::CUtensorMap>::zeroed();
    let addr = (ptr as u64 + col0 * 2) as *mut std::ffi::c_void;
    let global_dim = [64u64, rows];
    let global_strides = [256u64];
    let box_dim = [64u32, 64u32];
    let elem_strides = [1u32, 1u32];
    let st = unsafe {
        sys::cuTensorMapEncodeTiled(
            raw.as_mut_ptr(),
            CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
            2,
            addr,
            global_dim.as_ptr(),
            global_strides.as_ptr(),
            box_dim.as_ptr(),
            elem_strides.as_ptr(),
            CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
            CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B,
            CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
            CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE,
        )
    };
    if st != sys::CUresult::CUDA_SUCCESS {
        return Err(TensorError::Message(format!("cuTensorMapEncodeTiled col{col0}: {st:?}")));
    }
    let map = unsafe { raw.assume_init() };
    debug_assert_eq!(std::mem::size_of_val(&map), std::mem::size_of::<FvTensorMap>());
    Ok(unsafe { std::mem::transmute_copy(&map) })
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

    /// Value-first SwiGLU over a packed `[rows, 2*half]` last dim.
    pub fn swiglu_value_first(x: &[f32], half: usize) -> Vec<f32> {
        let width = 2 * half;
        debug_assert!(half > 0 && x.len() % width == 0);
        let n = x.len() / 2;
        let mut out = vec![0.0; n];
        if n >= PAR_MIN {
            out.par_iter_mut().enumerate().for_each(|(i, o)| {
                let row = i / half;
                let col = i % half;
                *o = x[row * width + col] * silu(x[row * width + half + col]);
            });
        } else {
            for (i, o) in out.iter_mut().enumerate() {
                let row = i / half;
                let col = i % half;
                *o = x[row * width + col] * silu(x[row * width + half + col]);
            }
        }
        out
    }

    /// erf to double precision (W. J. Cody's rational approximations), since
    /// std has none and the host path is the reference the kernel is held to.
    pub fn erf(x: f64) -> f64 {
        let ax = x.abs();
        let r = if ax < 0.5 {
            const A: [f64; 5] = [3.16112374387056560e0, 1.13864154151050156e2, 3.77485237685302021e2, 3.20937758913846947e3, 1.85777706184603153e-1];
            const B: [f64; 4] = [2.36012909523441209e1, 2.44024637934444173e2, 1.28261652607737228e3, 2.84423683343917062e3];
            let y = ax * ax;
            let mut num = A[4] * y;
            let mut den = y;
            for i in 0..3 {
                num = (num + A[i]) * y;
                den = (den + B[i]) * y;
            }
            return x * (num + A[3]) / (den + B[3]);
        } else if ax < 4.0 {
            const C: [f64; 9] = [5.64188496988670089e-1, 8.88314979438837594e0, 6.61191906371416295e1, 2.98635138197400131e2, 8.81952221241769090e2, 1.71204761263407058e3, 2.05107837782607147e3, 1.23033935479799725e3, 2.15311535474403846e-8];
            const D: [f64; 8] = [1.57449261107098347e1, 1.17693950891312499e2, 5.37181101862009858e2, 1.62138957456669019e3, 3.29079923573345963e3, 4.36261909014324716e3, 3.43936767414372164e3, 1.23033935480374942e3];
            let mut num = C[8] * ax;
            let mut den = ax;
            for i in 0..7 {
                num = (num + C[i]) * ax;
                den = (den + D[i]) * ax;
            }
            1.0 - (-ax * ax).exp() * (num + C[7]) / (den + D[7])
        } else {
            const P: [f64; 6] = [3.05326634961232344e-1, 3.60344899949804439e-1, 1.25781726111229246e-1, 1.60837851487422766e-2, 6.58749161529837803e-4, 1.63153871373020978e-2];
            const Q: [f64; 5] = [2.56852019228982242e0, 1.87295284992346725e0, 5.27905102951428412e-1, 6.05183413124413191e-2, 2.33520497626869185e-3];
            let y = 1.0 / (ax * ax);
            let mut num = P[5] * y;
            let mut den = y;
            for i in 0..4 {
                num = (num + P[i]) * y;
                den = (den + Q[i]) * y;
            }
            let t = y * (num + P[4]) / (den + Q[4]);
            let t = (1.0 / std::f64::consts::PI.sqrt() - t) / ax;
            1.0 - (-ax * ax).exp() * t
        };
        if x < 0.0 { -r } else { r }
    }

    pub fn gelu_erf(x: f32) -> f32 {
        let x = f64::from(x);
        (0.5 * x * (1.0 + erf(x * std::f64::consts::FRAC_1_SQRT_2))) as f32
    }

    pub fn leaky_relu(x: f32, slope: f32) -> f32 {
        if x >= 0.0 { x } else { x * slope }
    }

    /// `[N, C, L]`: `x + inv_beta[c] * sin^2(alpha[c] x)`.
    pub fn snake_beta(a: &[f32], alpha: &[f32], inv_beta: &[f32], c: usize, l: usize) -> Vec<f32> {
        a.par_iter()
            .enumerate()
            .map(|(i, &x)| {
                let ch = (i / l) % c;
                let sn = (alpha[ch] * x).sin();
                x + inv_beta[ch] * sn * sn
            })
            .collect()
    }

    /// rotate_half RoPE over channels `[0, r)` of `[B, H, S, D]`, `[S, R]` tables.
    pub fn rope_half(x: &[f32], cos: &[f32], sin: &[f32], s: usize, d: usize, r: usize) -> Vec<f32> {
        let half = r / 2;
        x.par_iter()
            .enumerate()
            .map(|(i, &v)| {
                let j = i % d;
                if j >= r {
                    return v;
                }
                let p = (i / d) % s;
                let other = if j < half { -x[i + half] } else { x[i - half] };
                v * cos[p * r + j] + other * sin[p * r + j]
            })
            .collect()
    }

    /// `[rows, cols]` weight → E4M3 codes and one scale per row (`amax / 448`,
    /// 1 for a dead row). Same arithmetic, in the same order, as the kernels.
    pub fn fp8_rows_quantize(w: &[f32], rows: usize, cols: usize) -> (Vec<u8>, Vec<f32>) {
        use fastvideo_ops::fp8;
        let scales: Vec<f32> = (0..rows)
            .into_par_iter()
            .map(|r| {
                let amax = w[r * cols..(r + 1) * cols].iter().fold(0.0f32, |a, v| if v.abs() <= a { a } else { v.abs() });
                // `amax * (1/448)`, as the kernel: see `fp8_row_scales`.
                if amax > 0.0 && amax.is_finite() { amax * (1.0f32 / 448.0) } else { 1.0 }
            })
            .collect();
        let q = w.par_iter().enumerate().map(|(i, &v)| fp8::f32_to_e4m3(v * (1.0 / scales[i / cols]))).collect();
        (q, scales)
    }

    /// Codes and per-row scales back to f32, rounded through bfloat16 as the
    /// device path does (the GEMM consumes a bf16 weight).
    pub fn fp8_rows_dequant(q: &[u8], scales: &[f32], cols: usize) -> Vec<f32> {
        use fastvideo_ops::fp8;
        q.par_iter()
            .enumerate()
            .map(|(i, &b)| half::bf16::from_f32(fp8::e4m3_to_f32(b) * scales[i / cols]).to_f32())
            .collect()
    }

    /// Pad the middle axis of `[outer, len, inner]`.
    pub fn pad_axis(x: &[f32], len: usize, inner: usize, left: usize, right: usize, mode: super::PadMode) -> Vec<f32> {
        let out_len = len + left + right;
        let total = x.len() / len * out_len;
        (0..total)
            .into_par_iter()
            .map(|i| {
                let o = (i / inner) % out_len;
                let src = o as isize - left as isize;
                let src = if src < 0 || src >= len as isize {
                    match mode {
                        super::PadMode::Zeros => return 0.0,
                        super::PadMode::Reflect => if src < 0 { -src } else { 2 * (len as isize - 1) - src },
                        super::PadMode::Replicate => src.clamp(0, len as isize - 1),
                    }
                } else {
                    src
                } as usize;
                x[(i / (inner * out_len)) * len * inner + src * inner + i % inner]
            })
            .collect()
    }

    /// GroupNorm over `[N, C, spatial]`, statistics in f64.
    #[allow(clippy::too_many_arguments)]
    pub fn group_norm(x: &[f32], w: &[f32], b: &[f32], c: usize, spatial: usize, groups: usize, eps: f32, silu: bool) -> Vec<f32> {
        let cg = c / groups;
        let ge = cg * spatial;
        let mut out = vec![0f32; x.len()];
        out.par_chunks_mut(ge).zip(x.par_chunks(ge)).enumerate().for_each(|(gi, (o, g))| {
            let mean = g.iter().map(|v| f64::from(*v)).sum::<f64>() / ge as f64;
            let var = (g.iter().map(|v| f64::from(*v).powi(2)).sum::<f64>() / ge as f64 - mean * mean).max(0.0);
            let inv = 1.0 / (var + f64::from(eps)).sqrt();
            let first = (gi % groups) * cg;
            for (j, (ov, xv)) in o.iter_mut().zip(g).enumerate() {
                let ch = first + j / spatial;
                let y = ((f64::from(*xv) - mean) * inv) as f32 * w[ch] + b[ch];
                *ov = if silu { y / (1.0 + (-y).exp()) } else { y };
            }
        });
        out
    }

    /// 1-D cross-correlation, PyTorch `Conv1d` semantics. `x`: `[n, c, l]`,
    /// `w`: `[oc, c / groups, k]`. Returns the output and its length.
    #[allow(clippy::too_many_arguments)]
    pub fn conv1d(
        x: &[f32],
        (n, c, l): (usize, usize, usize),
        w: &[f32],
        (oc, k): (usize, usize),
        pad: usize,
        stride: usize,
        dilation: usize,
        groups: usize,
    ) -> (Vec<f32>, usize) {
        let (cg, og) = (c / groups, oc / groups);
        let lo = (l + 2 * pad - dilation * (k - 1) - 1) / stride + 1;
        let mut out = vec![0f32; n * oc * lo];
        out.par_chunks_mut(lo).enumerate().for_each(|(row, y)| {
            let (ni, o) = (row / oc, row % oc);
            let g = o / og;
            for (t, yt) in y.iter_mut().enumerate() {
                let mut acc = 0f64;
                for ci in 0..cg {
                    let xrow = &x[(ni * c + g * cg + ci) * l..][..l];
                    let wrow = &w[(o * cg + ci) * k..][..k];
                    for (kk, wv) in wrow.iter().enumerate() {
                        let pos = t * stride + kk * dilation;
                        if pos >= pad && pos - pad < l {
                            acc += f64::from(xrow[pos - pad]) * f64::from(*wv);
                        }
                    }
                }
                *yt = acc as f32;
            }
        });
        (out, lo)
    }

    /// 1-D transposed convolution, PyTorch `ConvTranspose1d` semantics. `x`:
    /// `[n, c, l]`, `w`: `[c, oc / groups, k]`.
    #[allow(clippy::too_many_arguments)]
    pub fn conv_transpose1d(
        x: &[f32],
        (n, c, l): (usize, usize, usize),
        w: &[f32],
        (og, k): (usize, usize),
        pad: usize,
        stride: usize,
        dilation: usize,
        groups: usize,
        out_pad: usize,
    ) -> (Vec<f32>, usize) {
        let (cg, oc) = (c / groups, og * groups);
        let lo = (l - 1) * stride + dilation * (k - 1) + out_pad + 1 - 2 * pad;
        let mut out = vec![0f32; n * oc * lo];
        out.par_chunks_mut(lo).enumerate().for_each(|(row, y)| {
            let (ni, o) = (row / oc, row % oc);
            let (g, oi) = (o / og, o % og);
            for (t, yt) in y.iter_mut().enumerate() {
                let mut acc = 0f64;
                for kk in 0..k {
                    // Output t receives x[i] * w[kk] where t = i * stride - pad + kk * dilation.
                    let Some(num) = (t + pad).checked_sub(kk * dilation) else { continue };
                    if num % stride != 0 || num / stride >= l {
                        continue;
                    }
                    let i = num / stride;
                    for ci in 0..cg {
                        let ch = g * cg + ci;
                        acc += f64::from(x[(ni * c + ch) * l + i]) * f64::from(w[(ch * og + oi) * k + kk]);
                    }
                }
                *yt = acc as f32;
            }
        });
        (out, lo)
    }

    /// `[B, Hkv, S, D]` → `[B, Hkv * rep, S, D]`; `inner = S * D`.
    pub fn repeat_kv(x: &[f32], hkv: usize, rep: usize, inner: usize) -> Vec<f32> {
        let total = x.len() * rep;
        (0..total)
            .into_par_iter()
            .map(|i| {
                let h = (i / inner) % (hkv * rep);
                let b = i / (inner * hkv * rep);
                x[(b * hkv + h / rep) * inner + i % inner]
            })
            .collect()
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

/// `tanh(x / s) * s` — TAEHV's soft limiter.
#[cfg(feature = "cuda")]
pub fn tanh_scaled_device(a: &CudaSlice<f32>, s: f32) -> Result<CudaSlice<f32>> {
    let dev = ctx()?;
    let n = a.len() as i64;
    let sd = dev.stream.memcpy_stod(&[s]).map_err(err)?;
    let mut out = alloc(a.len())?;
    launch!(dev.stream, &dev.kernels.tanh_scaled, cfg_n(a.len()); a, &mut out, &sd, &n).map_err(err)?;
    Ok(out)
}

// ---- decoder-only encoders and audio decoders ----------------------------

#[cfg(feature = "cuda")]
pub fn leaky_relu_device(a: &CudaSlice<f32>, slope: f32) -> Result<CudaSlice<f32>> {
    let dev = ctx()?;
    let n = a.len() as i64;
    let mut out = alloc(a.len())?;
    launch!(dev.stream, &dev.kernels.leaky_relu, cfg_n(a.len()); a, &slope, &mut out, &n).map_err(err)?;
    Ok(out)
}

/// `x + inv_beta[c] * sin^2(alpha[c] * x)` over `[N, C, L]`.
#[cfg(feature = "cuda")]
pub fn snake_beta_device(
    a: &CudaSlice<f32>,
    alpha: &CudaSlice<f32>,
    inv_beta: &CudaSlice<f32>,
    c: usize,
    l: usize,
) -> Result<CudaSlice<f32>> {
    let dev = ctx()?;
    if alpha.len() != c || inv_beta.len() != c || c == 0 || l == 0 || a.len() % (c * l) != 0 {
        return Err(err(format!("snake_beta: {} elements for [N, {c}, {l}] with {} alphas", a.len(), alpha.len())));
    }
    let (n, c, l) = (a.len() as i64, c as i64, l as i64);
    let mut out = alloc(a.len())?;
    launch!(dev.stream, &dev.kernels.snake_beta, cfg_n(a.len()); a, alpha, inv_beta, &mut out, &c, &l, &n).map_err(err)?;
    Ok(out)
}

/// rotate_half RoPE over channels `[0, r)` of `[B, H, S, D]` with `[S, R]` tables.
#[cfg(feature = "cuda")]
pub fn rope_half_device(
    x: &CudaSlice<f32>,
    cos: &CudaSlice<f32>,
    sin: &CudaSlice<f32>,
    s: usize,
    d: usize,
    r: usize,
) -> Result<CudaSlice<f32>> {
    let dev = ctx()?;
    if r == 0 || r % 2 != 0 || r > d || cos.len() != s * r || sin.len() != s * r || x.len() % (s * d) != 0 {
        return Err(err(format!("rope_half: x {} for S={s} D={d} R={r}, tables {}", x.len(), cos.len())));
    }
    let (n, s, d, r) = (x.len() as i64, s as i64, d as i64, r as i64);
    let mut out = alloc(x.len())?;
    launch!(dev.stream, &dev.kernels.rope_half, cfg_n(x.len()); x, cos, sin, &mut out, &s, &d, &r, &n).map_err(err)?;
    Ok(out)
}

/// `[B, Hkv, S, D]` → `[B, Hkv * rep, S, D]` (repeat_interleave on the head axis).
#[cfg(feature = "cuda")]
pub fn repeat_kv_device(x: &CudaSlice<f32>, hkv: usize, rep: usize, inner: usize) -> Result<CudaSlice<f32>> {
    let dev = ctx()?;
    if hkv == 0 || rep == 0 || inner == 0 || x.len() % (hkv * inner) != 0 {
        return Err(err(format!("repeat_kv: {} elements for Hkv={hkv} inner={inner}", x.len())));
    }
    let total = x.len() * rep;
    let (n, hkv, rep, inner) = (total as i64, hkv as i64, rep as i64, inner as i64);
    let mut out = alloc(total)?;
    launch!(dev.stream, &dev.kernels.repeat_kv, cfg_n(total); x, &mut out, &hkv, &rep, &inner, &n).map_err(err)?;
    Ok(out)
}

/// How [`pad_axis_device`] fills positions outside the input.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PadMode {
    Zeros = 0,
    /// Mirror without repeating the edge sample (`torch` `reflect`).
    Reflect = 1,
    /// Repeat the edge sample (`torch` `replicate`).
    Replicate = 2,
}

/// Pad the middle axis of `[outer, len, inner]`.
#[cfg(feature = "cuda")]
pub fn pad_axis_device(
    x: &CudaSlice<f32>,
    len: usize,
    inner: usize,
    left: usize,
    right: usize,
    mode: PadMode,
) -> Result<CudaSlice<f32>> {
    let dev = ctx()?;
    if len == 0 || inner == 0 || x.len() % (len * inner) != 0 {
        return Err(err(format!("pad_axis: {} elements for len={len} inner={inner}", x.len())));
    }
    let out_len = len + left + right;
    let total = x.len() / len * out_len;
    let (n, len, inner, left, out_len, mode) = (total as i64, len as i64, inner as i64, left as i64, out_len as i64, mode as i32);
    let mut out = alloc(total)?;
    launch!(dev.stream, &dev.kernels.pad_axis, cfg_n(total); x, &mut out, &len, &inner, &left, &out_len, &mode, &n).map_err(err)?;
    Ok(out)
}

/// GroupNorm over `[N, C, spatial]` with `groups` groups, affine, optional SiLU.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub fn group_norm_device(
    x: &CudaSlice<f32>,
    w: &CudaSlice<f32>,
    b: &CudaSlice<f32>,
    n: usize,
    c: usize,
    spatial: usize,
    groups: usize,
    eps: f32,
    silu: bool,
) -> Result<CudaSlice<f32>> {
    let dev = ctx()?;
    if groups == 0 || c % groups != 0 || x.len() != n * c * spatial || w.len() != c || b.len() != c || x.is_empty() {
        return Err(err(format!("group_norm: {} elements for [{n}, {c}, {spatial}] in {groups} groups", x.len())));
    }
    let cg = c / groups;
    let group_elems = (cg * spatial) as i64;
    let mut stats = dev.stream.alloc_zeros::<f64>(2 * n * groups).map_err(err)?;
    // A power-of-two block so the halving reduction is exact; small groups
    // (a late, narrow feature map) do not need 512 threads.
    let threads = ((cg * spatial).next_power_of_two()).clamp(1, 512) as u32;
    let cfg = LaunchConfig {
        grid_dim: ((n * groups) as u32, 1, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: threads * 16,
    };
    launch!(dev.stream, &dev.kernels.group_norm_stats, cfg; x, &mut stats, &group_elems).map_err(err)?;
    let mut out = alloc(x.len())?;
    let (total, c, spatial, cg, silu) = (x.len() as i64, c as i64, spatial as i64, cg as i64, i32::from(silu));
    launch!(dev.stream, &dev.kernels.group_norm_apply, cfg_n(x.len()); x, &stats, w, b, &mut out, &c, &spatial, &cg, &eps, &silu, &total)
        .map_err(err)?;
    Ok(out)
}

// ---- weight-only FP8 (per-row scales) -------------------------------------

/// `[rows, cols]` f32 weight → E4M3 codes and one dequantization scale per row.
#[cfg(feature = "cuda")]
pub fn fp8_rows_quantize_device(w: &CudaSlice<f32>, rows: usize, cols: usize) -> Result<(CudaSlice<u8>, CudaSlice<f32>)> {
    let dev = ctx()?;
    if rows == 0 || cols == 0 || w.len() != rows * cols {
        return Err(err(format!("fp8_rows_quantize: {} elements for [{rows}, {cols}]", w.len())));
    }
    let mut scales = alloc(rows)?;
    let threads = cols.next_power_of_two().clamp(1, 256) as u32;
    let cfg = LaunchConfig { grid_dim: (rows as u32, 1, 1), block_dim: (threads, 1, 1), shared_mem_bytes: threads * 4 };
    let c = cols as i64;
    launch!(dev.stream, &dev.kernels.fp8_row_scales, cfg; w, &mut scales, &c).map_err(err)?;
    let mut q = unsafe { dev.stream.alloc::<u8>(rows * cols) }.map_err(err)?;
    let n = (rows * cols) as i64;
    launch!(dev.stream, &dev.kernels.fp8_rows_quantize, cfg_n(rows * cols); w, &scales, &mut q, &c, &n).map_err(err)?;
    Ok((q, scales))
}

/// E4M3 codes with per-row scales → a bfloat16 weight, for the bf16 GEMM.
#[cfg(feature = "cuda")]
pub fn fp8_rows_dequant_bf16_device(q: &CudaSlice<u8>, scales: &CudaSlice<f32>, cols: usize) -> Result<CudaSlice<half::bf16>> {
    let dev = ctx()?;
    if cols == 0 || q.len() != scales.len() * cols {
        return Err(err(format!("fp8_rows_dequant: {} codes for {} rows of {cols}", q.len(), scales.len())));
    }
    let mut out = unsafe { dev.stream.alloc::<half::bf16>(q.len().max(1)) }.map_err(err)?;
    let (c, n) = (cols as i64, q.len() as i64);
    launch!(dev.stream, &dev.kernels.fp8_rows_dequant_bf16, cfg_n(q.len()); q, scales, &mut out, &c, &n).map_err(err)?;
    Ok(out)
}

// ---- frame output --------------------------------------------------------

/// `[frames, 3, h, w]` planar f32 → `[frames, h, w, 3]` interleaved u8 on the
/// device, then one copy down. `byte = trunc(clamp(x * a + b, 0, 255))`.
///
/// The host used to do this per pixel from a 4x larger f32 copy; on the device
/// it is one elementwise launch and the transfer is 3 bytes per pixel.
#[cfg(feature = "cuda")]
pub fn pack_rgb_u8_device(
    x: &CudaSlice<f32>,
    frames: usize,
    h: usize,
    w: usize,
    a: f32,
    b: f32,
) -> Result<Vec<u8>> {
    let dev = ctx()?;
    let pixels = frames * h * w;
    if x.len() != pixels * 3 {
        return Err(err(format!(
            "pack_rgb_u8: {} elements is not [{frames}, 3, {h}, {w}]",
            x.len()
        )));
    }
    let mut out = unsafe { dev.stream.alloc::<u8>((pixels * 3).max(1)) }.map_err(err)?;
    let (fr, hh, ww) = (frames as i32, h as i32, w as i32);
    launch!(dev.stream, &dev.kernels.pack_rgb_u8, cfg_n(pixels); x, &mut out, &fr, &hh, &ww, &a, &b)
        .map_err(err)?;
    let host = dev.stream.memcpy_dtov(&out).map_err(err)?;
    super::stats::record_d2h(pixels * 3 / 4);
    Ok(host)
}

// ---- FP8 E4M3 ------------------------------------------------------------

/// Dynamic per-tensor E4M3 quantization of an activation.
///
/// Returns the E4M3 bytes and a one-float device buffer holding the
/// dequantization `scale`, which is exactly what the GEMM's scale pointer
/// wants. Keeping the whole computation on the device means the amax never has
/// to come back to the host between the reduction and the matmul.
///
/// The scale is recomputed every call rather than calibrated once. That costs
/// one extra pass over the activation, and it is the conservative choice: a
/// stale calibration that under-estimates amax clips the tensor, and clipping
/// in a denoiser compounds across steps.
#[cfg(feature = "cuda")]
pub fn quantize_e4m3_device(a: &CudaSlice<f32>) -> Result<(CudaSlice<u8>, CudaSlice<f32>)> {
    let dev = ctx()?;
    let n = a.len() as i64;
    // amax_abs combines blocks with atomicMax, so the target must start at zero.
    let mut amax = dev.stream.alloc_zeros::<f32>(1).map_err(err)?;
    let blocks = a.len().div_ceil(256).clamp(1, 1024) as u32;
    let cfg = LaunchConfig {
        grid_dim: (blocks, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 256 * 4,
    };
    launch!(dev.stream, &dev.kernels.amax_abs, cfg; a, &mut amax, &n).map_err(err)?;

    let cfg1 = LaunchConfig { grid_dim: (1, 1, 1), block_dim: (1, 1, 1), shared_mem_bytes: 0 };
    let (mut scale, mut inv) = (alloc(1)?, alloc(1)?);
    launch!(dev.stream, &dev.kernels.e4m3_scale_from_amax, cfg1; &amax, &mut scale, &mut inv).map_err(err)?;

    let mut out = unsafe { dev.stream.alloc::<u8>(a.len().max(1)) }.map_err(err)?;
    launch!(dev.stream, &dev.kernels.quantize_e4m3, cfg_n(a.len()); a, &mut out, &inv, &n).map_err(err)?;
    Ok((out, scale))
}

/// E4M3 bytes back to f32, for checking the quantizer against the host
/// reference in the kernels tier.
#[cfg(feature = "cuda")]
pub fn dequantize_e4m3_device(a: &CudaSlice<u8>, scale: &CudaSlice<f32>) -> Result<CudaSlice<f32>> {
    let dev = ctx()?;
    let n = a.len() as i64;
    let mut out = alloc(a.len().max(1))?;
    launch!(dev.stream, &dev.kernels.dequantize_e4m3, cfg_n(a.len()); a, &mut out, scale, &n).map_err(err)?;
    Ok(out)
}

#[cfg(test)]
mod tma_layout {
    /// Mirrors `mma_swz_tma` in kernels.cu: two 64-col 128B-swizzled panels.
    fn mma_swz_tma(row: u32, col: u32) -> u32 {
        let panel = col >> 6;
        let c = col & 63;
        panel * (64 * 128) + row * 128 + (((c >> 3) ^ (row & 7)) << 4) + ((c & 7) << 1)
    }

    #[test]
    fn tma_swizzle_spreads_an_8_row_ldmatrix_across_banks() {
        for col in [0u32, 8, 16, 32, 48, 64, 80, 112] {
            let banks: Vec<u32> = (0..8).map(|r| (mma_swz_tma(r, col) / 4) % 32).collect();
            assert!(banks.iter().any(|&b| b != banks[0]), "col {col} collapsed to one bank: {banks:?}");
        }
    }

    #[test]
    fn tma_swizzle_panels_do_not_overlap() {
        assert!(mma_swz_tma(63, 63) < 64 * 128);
        assert_eq!(mma_swz_tma(0, 64), 64 * 128);
        assert!(mma_swz_tma(63, 127) < 2 * 64 * 128);
    }
}

#[cfg(test)]
mod swiglu {
    #[test]
    fn swiglu_value_first_is_v_times_silu_g() {
        let half = 4;
        let x: Vec<f32> = (0..2 * 2 * half).map(|i| (i as f32 * 0.3 - 1.1).sin()).collect();
        let got = super::host::swiglu_value_first(&x, half);
        for i in 0..got.len() {
            let row = i / half;
            let col = i % half;
            let want = x[row * 2 * half + col] * super::host::silu(x[row * 2 * half + half + col]);
            assert!((got[i] - want).abs() < 1e-6, "i={i}");
        }
    }
}
