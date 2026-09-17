//! CUDA-accelerated ops for Wan (cuBLAS / NVRTC / cuDNN when a device is live).
//!
//! With `--features cuda` and an initialized [`super::device::DeviceContext`]:
//! - GEMM → cuBLAS (host-upload, device-resident, or strided-batched for attention)
//! - same-shape add/mul/sub, scalars, silu/gelu/clamp → NVRTC
//! - softmax / rms_norm on the last axis → NVRTC
//! - 4D permute → NVRTC
//! - NCHW conv2d → cuDNN
//! - SDPA → device-resident strided-batched QKᵀ + softmax + PV ([`super::attn`])
//!
//! When residency is enabled ([`super::resident::residency_enabled`]), device-resident
//! helpers keep results on GPU (`CudaSlice`) without a download. Host-upload APIs
//! remain for fallback when `FASTVIDEO_RESIDENT=0` or no device buffer is available.

#[cfg(feature = "cuda")]
use super::device;
use super::tensor::{CudaTensor, Result, TensorError};

/// `(m,k) @ (k,n)` — cuBLAS when a global CUDA device is live, else host.
pub fn matmul_f32(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Result<Vec<f32>> {
    #[cfg(feature = "cuda")]
    {
        if device::global_device().is_some() {
            return device::matmul_2d_f32(a, b, m, k, n)
                .map_err(|e| TensorError::Message(e.to_string()));
        }
    }
    let _ = (a, b, m, k, n);
    Err(TensorError::Message(
        "matmul_f32: use CudaTensor::matmul (host path) when CUDA device is unset".into(),
    ))
}

/// Upload f32 host buffer to the active CUDA device and download it back (sync check).
#[cfg(feature = "cuda")]
pub fn roundtrip_f32(data: &[f32]) -> device::Result<Vec<f32>> {
    let dev = device::global_device().ok_or_else(|| {
        device::DeviceError::Message("no global CUDA device context".into())
    })?;
    let on_dev = dev.stream.memcpy_stod(data)?;
    Ok(dev.stream.memcpy_dtov(&on_dev)?)
}

/// Convert host f32 weights to little-endian BF16 bytes for GPU upload.
pub fn f32_to_bf16_bytes(values: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(values.len() * 2);
    for &v in values {
        let bits = half::bf16::from_f32(v).to_bits();
        out.extend_from_slice(&bits.to_le_bytes());
    }
    out
}

/// Dense SDPA via [`CudaTensor`] (cuBLAS + NVRTC softmax when CUDA is active).
pub fn scaled_dot_product_attention(
    query: &CudaTensor,
    key: &CudaTensor,
    value: &CudaTensor,
    scale: Option<f32>,
) -> Result<CudaTensor> {
    super::nn::scaled_dot_product_attention(query, key, value, scale)
}

#[cfg(feature = "cuda")]
#[derive(Clone, Copy)]
pub enum ElemBinary {
    Add,
    Mul,
    Sub,
}

#[cfg(feature = "cuda")]
#[derive(Clone, Copy)]
pub enum ElemUnary {
    Silu,
    GeluTanh,
}

/// Device-resident GEMM: writes into `out` (no host roundtrip).
#[cfg(feature = "cuda")]
pub fn matmul_2d_f32_device(
    a: &cudarc::driver::CudaSlice<f32>,
    b: &cudarc::driver::CudaSlice<f32>,
    out: &mut cudarc::driver::CudaSlice<f32>,
    m: usize,
    k: usize,
    n: usize,
) -> Result<()> {
    device::matmul_2d_f32_device(a, b, out, m, k, n).map_err(|e| TensorError::Message(e.to_string()))
}

/// Device-resident same-shape binary op → new `CudaSlice` (no download).
#[cfg(feature = "cuda")]
pub fn elem_binary_device(
    a: &cudarc::driver::CudaSlice<f32>,
    b: &cudarc::driver::CudaSlice<f32>,
    kind: ElemBinary,
) -> Option<cudarc::driver::CudaSlice<f32>> {
    if a.len() != b.len() || a.is_empty() {
        return None;
    }
    let dev = device::global_device()?;
    let n = a.len() as i32;
    let mut out_dev = dev.stream.alloc_zeros::<f32>(a.len()).ok()?;
    let f = match kind {
        ElemBinary::Add => &dev.kernels.elem_add,
        ElemBinary::Mul => &dev.kernels.elem_mul,
        ElemBinary::Sub => &dev.kernels.elem_sub,
    };
    unsafe {
        super::kernels::launch_binary(&dev.stream, f, a, b, &mut out_dev, n).ok()?;
    }
    Some(out_dev)
}

/// Device-resident unary → new `CudaSlice`.
#[cfg(feature = "cuda")]
pub fn unary_device(
    a: &cudarc::driver::CudaSlice<f32>,
    kind: ElemUnary,
) -> Option<cudarc::driver::CudaSlice<f32>> {
    if a.is_empty() {
        return Some(device::global_device()?.stream.alloc_zeros::<f32>(0).ok()?);
    }
    let dev = device::global_device()?;
    let n = a.len() as i32;
    let mut out_dev = dev.stream.alloc_zeros::<f32>(a.len()).ok()?;
    let f = match kind {
        ElemUnary::Silu => &dev.kernels.silu,
        ElemUnary::GeluTanh => &dev.kernels.gelu_tanh,
    };
    unsafe {
        super::kernels::launch_unary(&dev.stream, f, a, &mut out_dev, n).ok()?;
    }
    Some(out_dev)
}

#[cfg(feature = "cuda")]
pub fn mul_scalar_device(
    a: &cudarc::driver::CudaSlice<f32>,
    s: f32,
) -> Option<cudarc::driver::CudaSlice<f32>> {
    if a.is_empty() {
        return Some(device::global_device()?.stream.alloc_zeros::<f32>(0).ok()?);
    }
    let dev = device::global_device()?;
    let n = a.len() as i32;
    let mut out_dev = dev.stream.alloc_zeros::<f32>(a.len()).ok()?;
    unsafe {
        super::kernels::launch_unary_scalar(
            &dev.stream,
            &dev.kernels.mul_scalar,
            a,
            s,
            &mut out_dev,
            n,
        )
        .ok()?;
    }
    Some(out_dev)
}

#[cfg(feature = "cuda")]
pub fn add_scalar_device(
    a: &cudarc::driver::CudaSlice<f32>,
    s: f32,
) -> Option<cudarc::driver::CudaSlice<f32>> {
    if a.is_empty() {
        return Some(device::global_device()?.stream.alloc_zeros::<f32>(0).ok()?);
    }
    let dev = device::global_device()?;
    let n = a.len() as i32;
    let mut out_dev = dev.stream.alloc_zeros::<f32>(a.len()).ok()?;
    unsafe {
        super::kernels::launch_unary_scalar(
            &dev.stream,
            &dev.kernels.add_scalar,
            a,
            s,
            &mut out_dev,
            n,
        )
        .ok()?;
    }
    Some(out_dev)
}

#[cfg(feature = "cuda")]
pub fn clamp_device(
    a: &cudarc::driver::CudaSlice<f32>,
    lo: f32,
    hi: f32,
) -> Option<cudarc::driver::CudaSlice<f32>> {
    if a.is_empty() {
        return Some(device::global_device()?.stream.alloc_zeros::<f32>(0).ok()?);
    }
    let dev = device::global_device()?;
    let n = a.len() as i32;
    let mut out_dev = dev.stream.alloc_zeros::<f32>(a.len()).ok()?;
    unsafe {
        super::kernels::launch_clamp(
            &dev.stream,
            &dev.kernels.clamp_f,
            a,
            lo,
            hi,
            &mut out_dev,
            n,
        )
        .ok()?;
    }
    Some(out_dev)
}

#[cfg(feature = "cuda")]
pub fn softmax_last_device(
    a: &cudarc::driver::CudaSlice<f32>,
    width: usize,
) -> Option<cudarc::driver::CudaSlice<f32>> {
    if a.is_empty() || width == 0 || a.len() % width != 0 {
        return None;
    }
    let rows = a.len() / width;
    let dev = device::global_device()?;
    let mut out_dev = dev.stream.alloc_zeros::<f32>(a.len()).ok()?;
    unsafe {
        super::kernels::launch_softmax_last(
            &dev.stream,
            &dev.kernels.softmax_last,
            a,
            &mut out_dev,
            rows as i32,
            width as i32,
        )
        .ok()?;
    }
    Some(out_dev)
}

#[cfg(feature = "cuda")]
pub fn rms_norm_last_device(
    a: &cudarc::driver::CudaSlice<f32>,
    weight: &cudarc::driver::CudaSlice<f32>,
    eps: f32,
) -> Option<cudarc::driver::CudaSlice<f32>> {
    let width = weight.len();
    if a.is_empty() || width == 0 || a.len() % width != 0 {
        return None;
    }
    let rows = a.len() / width;
    let dev = device::global_device()?;
    let mut out_dev = dev.stream.alloc_zeros::<f32>(a.len()).ok()?;
    unsafe {
        super::kernels::launch_rms_norm_last(
            &dev.stream,
            &dev.kernels.rms_norm_last,
            a,
            weight,
            &mut out_dev,
            rows as i32,
            width as i32,
            eps,
        )
        .ok()?;
    }
    Some(out_dev)
}

/// Device-resident LayerNorm over the last dim. `weight`/`bias` are `None`
/// for an unaffine norm (Wan's AdaLN pre-norms).
#[cfg(feature = "cuda")]
pub fn layer_norm_last_device(
    a: &cudarc::driver::CudaSlice<f32>,
    weight: Option<&cudarc::driver::CudaSlice<f32>>,
    bias: Option<&cudarc::driver::CudaSlice<f32>>,
    width: usize,
    eps: f32,
) -> Option<cudarc::driver::CudaSlice<f32>> {
    if a.is_empty() || width == 0 || a.len() % width != 0 {
        return None;
    }
    let rows = a.len() / width;
    let dev = device::global_device()?;
    let has_affine = weight.is_some() && bias.is_some();
    // The kernel only reads weight/bias when has_affine is set, but it still
    // needs *some* CudaSlice to bind as the arg; a zero-length empty alloc is
    // never dereferenced in that branch.
    let empty;
    let (w, b): (&cudarc::driver::CudaSlice<f32>, &cudarc::driver::CudaSlice<f32>) =
        match (weight, bias) {
            (Some(w), Some(b)) => (w, b),
            _ => {
                empty = dev.stream.alloc_zeros::<f32>(0).ok()?;
                (&empty, &empty)
            }
        };
    let mut out_dev = dev.stream.alloc_zeros::<f32>(a.len()).ok()?;
    unsafe {
        super::kernels::launch_layer_norm_last(
            &dev.stream,
            &dev.kernels.layer_norm_last,
            a,
            w,
            b,
            &mut out_dev,
            rows as i32,
            width as i32,
            eps,
            has_affine,
        )
        .ok()?;
    }
    Some(out_dev)
}

/// Fused `out[b,l,d] = x[b,l,d] * (1 + scale[b,d]) + shift[b,d]`. `x` is
/// `[batch, seq, dim]`; `scale`/`shift` are `[batch, dim]` (broadcast over
/// `seq`) — this is AdaLN's `normed.mul(scale+1).add(shift)` as one launch
/// with no host round trip, replacing the CPU `broadcast_bin` fallback.
#[cfg(feature = "cuda")]
pub fn modulate_scale_shift_device(
    x: &cudarc::driver::CudaSlice<f32>,
    scale: &cudarc::driver::CudaSlice<f32>,
    shift: &cudarc::driver::CudaSlice<f32>,
    batch: usize,
    seq: usize,
    dim: usize,
) -> Option<cudarc::driver::CudaSlice<f32>> {
    if x.len() != batch * seq * dim || scale.len() != batch * dim || shift.len() != batch * dim {
        return None;
    }
    let dev = device::global_device()?;
    let mut out_dev = dev.stream.alloc_zeros::<f32>(x.len().max(1)).ok()?;
    unsafe {
        super::kernels::launch_modulate_scale_shift_last(
            &dev.stream,
            &dev.kernels.modulate_scale_shift_last,
            x,
            scale,
            shift,
            &mut out_dev,
            batch as i32,
            seq as i32,
            dim as i32,
        )
        .ok()?;
    }
    Some(out_dev)
}

/// Fused `out[b,l,d] = x[b,l,d] * gate[b,d]`. Same broadcast shape as
/// [`modulate_scale_shift_device`]; replaces AdaLN's gated-residual
/// `attn.mul(gate)` broadcast.
#[cfg(feature = "cuda")]
pub fn broadcast_mul_last_device(
    x: &cudarc::driver::CudaSlice<f32>,
    gate: &cudarc::driver::CudaSlice<f32>,
    batch: usize,
    seq: usize,
    dim: usize,
) -> Option<cudarc::driver::CudaSlice<f32>> {
    if x.len() != batch * seq * dim || gate.len() != batch * dim {
        return None;
    }
    let dev = device::global_device()?;
    let mut out_dev = dev.stream.alloc_zeros::<f32>(x.len().max(1)).ok()?;
    unsafe {
        super::kernels::launch_broadcast_mul_last(
            &dev.stream,
            &dev.kernels.broadcast_mul_last,
            x,
            gate,
            &mut out_dev,
            batch as i32,
            seq as i32,
            dim as i32,
        )
        .ok()?;
    }
    Some(out_dev)
}

/// In-place add bias along the last dim (`out[i] += bias[i % width]`).
#[cfg(feature = "cuda")]
pub fn add_bias_last_inplace(
    out: &mut cudarc::driver::CudaSlice<f32>,
    bias: &cudarc::driver::CudaSlice<f32>,
) -> Result<()> {
    let width = bias.len();
    if width == 0 || out.len() % width != 0 {
        return Err(TensorError::Message("add_bias_last size mismatch".into()));
    }
    let dev = device::global_device().ok_or_else(|| {
        TensorError::Message("no global CUDA device context".into())
    })?;
    let n = out.len() as i32;
    unsafe {
        super::kernels::launch_add_bias_last(
            &dev.stream,
            &dev.kernels.add_bias_last,
            out,
            bias,
            n,
            width as i32,
        )
        .map_err(|e| TensorError::Message(e.to_string()))?;
    }
    Ok(())
}

/// Try GPU path for same-shape binary ops. Returns `None` to fall back to host.
/// Host-upload API: uploads, runs, downloads.
#[cfg(feature = "cuda")]
pub fn try_elem_binary(a: &[f32], b: &[f32], kind: ElemBinary) -> Option<Vec<f32>> {
    if a.len() != b.len() || a.is_empty() {
        return None;
    }
    let dev = device::global_device()?;
    let a_dev = dev.stream.memcpy_stod(a).ok()?;
    let b_dev = dev.stream.memcpy_stod(b).ok()?;
    let out_dev = elem_binary_device(&a_dev, &b_dev, kind)?;
    dev.stream.memcpy_dtov(&out_dev).ok()
}

#[cfg(feature = "cuda")]
pub fn try_unary(a: &[f32], kind: ElemUnary) -> Option<Vec<f32>> {
    if a.is_empty() {
        return Some(Vec::new());
    }
    let dev = device::global_device()?;
    let a_dev = dev.stream.memcpy_stod(a).ok()?;
    let out_dev = unary_device(&a_dev, kind)?;
    dev.stream.memcpy_dtov(&out_dev).ok()
}

#[cfg(feature = "cuda")]
pub fn try_mul_scalar(a: &[f32], s: f32) -> Option<Vec<f32>> {
    if a.is_empty() {
        return Some(Vec::new());
    }
    let dev = device::global_device()?;
    let a_dev = dev.stream.memcpy_stod(a).ok()?;
    let out_dev = mul_scalar_device(&a_dev, s)?;
    dev.stream.memcpy_dtov(&out_dev).ok()
}

#[cfg(feature = "cuda")]
pub fn try_add_scalar(a: &[f32], s: f32) -> Option<Vec<f32>> {
    if a.is_empty() {
        return Some(Vec::new());
    }
    let dev = device::global_device()?;
    let a_dev = dev.stream.memcpy_stod(a).ok()?;
    let out_dev = add_scalar_device(&a_dev, s)?;
    dev.stream.memcpy_dtov(&out_dev).ok()
}

#[cfg(feature = "cuda")]
pub fn try_clamp(a: &[f32], lo: f32, hi: f32) -> Option<Vec<f32>> {
    if a.is_empty() {
        return Some(Vec::new());
    }
    let dev = device::global_device()?;
    let a_dev = dev.stream.memcpy_stod(a).ok()?;
    let out_dev = clamp_device(&a_dev, lo, hi)?;
    dev.stream.memcpy_dtov(&out_dev).ok()
}

#[cfg(feature = "cuda")]
pub fn try_softmax_last(a: &[f32], width: usize) -> Option<Vec<f32>> {
    if a.is_empty() || width == 0 || a.len() % width != 0 {
        return None;
    }
    let dev = device::global_device()?;
    let a_dev = dev.stream.memcpy_stod(a).ok()?;
    let out_dev = softmax_last_device(&a_dev, width)?;
    dev.stream.memcpy_dtov(&out_dev).ok()
}

#[cfg(feature = "cuda")]
pub fn try_rms_norm_last(a: &[f32], weight: &[f32], eps: f32) -> Option<Vec<f32>> {
    let width = weight.len();
    if a.is_empty() || width == 0 || a.len() % width != 0 {
        return None;
    }
    let dev = device::global_device()?;
    let a_dev = dev.stream.memcpy_stod(a).ok()?;
    let w_dev = dev.stream.memcpy_stod(weight).ok()?;
    let out_dev = rms_norm_last_device(&a_dev, &w_dev, eps)?;
    dev.stream.memcpy_dtov(&out_dev).ok()
}

/// Host-upload fallback for [`layer_norm_last_device`] (used when the input
/// isn't already device-resident but a CUDA device is live).
#[cfg(feature = "cuda")]
pub fn try_layer_norm_last(
    a: &[f32],
    weight: Option<&[f32]>,
    bias: Option<&[f32]>,
    width: usize,
    eps: f32,
) -> Option<Vec<f32>> {
    if a.is_empty() || width == 0 || a.len() % width != 0 {
        return None;
    }
    let dev = device::global_device()?;
    let a_dev = dev.stream.memcpy_stod(a).ok()?;
    let w_dev = weight.map(|w| dev.stream.memcpy_stod(w)).transpose().ok()?;
    let b_dev = bias.map(|b| dev.stream.memcpy_stod(b)).transpose().ok()?;
    let out_dev = layer_norm_last_device(&a_dev, w_dev.as_ref(), b_dev.as_ref(), width, eps)?;
    dev.stream.memcpy_dtov(&out_dev).ok()
}

/// Host-upload fallback for [`modulate_scale_shift_device`].
#[cfg(feature = "cuda")]
pub fn try_modulate_scale_shift(
    x: &[f32],
    scale: &[f32],
    shift: &[f32],
    batch: usize,
    seq: usize,
    dim: usize,
) -> Option<Vec<f32>> {
    let dev = device::global_device()?;
    let x_dev = dev.stream.memcpy_stod(x).ok()?;
    let scale_dev = dev.stream.memcpy_stod(scale).ok()?;
    let shift_dev = dev.stream.memcpy_stod(shift).ok()?;
    let out_dev = modulate_scale_shift_device(&x_dev, &scale_dev, &shift_dev, batch, seq, dim)?;
    dev.stream.memcpy_dtov(&out_dev).ok()
}

/// Host-upload fallback for [`broadcast_mul_last_device`].
#[cfg(feature = "cuda")]
pub fn try_broadcast_mul_last(
    x: &[f32],
    gate: &[f32],
    batch: usize,
    seq: usize,
    dim: usize,
) -> Option<Vec<f32>> {
    let dev = device::global_device()?;
    let x_dev = dev.stream.memcpy_stod(x).ok()?;
    let gate_dev = dev.stream.memcpy_stod(gate).ok()?;
    let out_dev = broadcast_mul_last_device(&x_dev, &gate_dev, batch, seq, dim)?;
    dev.stream.memcpy_dtov(&out_dev).ok()
}

#[cfg(feature = "cuda")]
pub fn try_conv2d(
    input: &[f32],
    weight: &[f32],
    n: usize,
    c_in: usize,
    h: usize,
    w: usize,
    c_out: usize,
    kh: usize,
    kw: usize,
    padding: usize,
    stride: usize,
) -> Option<Vec<f32>> {
    if device::global_device().is_none() {
        return None;
    }
    device::conv2d_f32(
        input, weight, n, c_in, h, w, c_out, kh, kw, padding, stride,
    )
    .ok()
}

/// Device-resident conv2d when both buffers are already on GPU (still downloads
/// via cuDNN path helper that accepts slices by uploading if needed — prefer
/// host try_conv2d until a full cuDNN-device API is wired). Placeholder kept for
/// symmetry; returns `None` so callers fall through.
#[cfg(feature = "cuda")]
pub fn conv2d_device(
    input: &cudarc::driver::CudaSlice<f32>,
    weight: &cudarc::driver::CudaSlice<f32>,
    n: usize,
    c_in: usize,
    h: usize,
    w: usize,
    c_out: usize,
    kh: usize,
    kw: usize,
    padding: usize,
    stride: usize,
) -> Option<cudarc::driver::CudaSlice<f32>> {
    let dev = device::global_device()?;
    let stride = stride.max(1);
    let out_h = (h + 2 * padding - kh) / stride + 1;
    let out_w = (w + 2 * padding - kw) / stride + 1;
    // Download → host conv2d_f32 path would defeat residency; run cuDNN with
    // existing device pointers by mirroring device::conv2d_f32 without H2D.
    use cudarc::cudnn::{sys, ConvForward};

    let pad = [padding as i32, padding as i32];
    let stride_hw = [stride as i32, stride as i32];
    let dilation = [1i32, 1];
    let conv = dev
        .cudnn
        .create_conv2d::<f32>(
            pad,
            stride_hw,
            dilation,
            sys::cudnnConvolutionMode_t::CUDNN_CROSS_CORRELATION,
        )
        .ok()?;
    let x_desc = dev
        .cudnn
        .create_4d_tensor::<f32>(
            sys::cudnnTensorFormat_t::CUDNN_TENSOR_NCHW,
            [n as i32, c_in as i32, h as i32, w as i32],
        )
        .ok()?;
    let w_desc = dev
        .cudnn
        .create_4d_filter::<f32>(
            sys::cudnnTensorFormat_t::CUDNN_TENSOR_NCHW,
            [c_out as i32, c_in as i32, kh as i32, kw as i32],
        )
        .ok()?;
    let y_desc = dev
        .cudnn
        .create_4d_tensor::<f32>(
            sys::cudnnTensorFormat_t::CUDNN_TENSOR_NCHW,
            [n as i32, c_out as i32, out_h as i32, out_w as i32],
        )
        .ok()?;
    let mut y_dev = dev
        .stream
        .alloc_zeros::<f32>(n * c_out * out_h * out_w)
        .ok()?;
    let op = ConvForward {
        conv: &conv,
        x: &x_desc,
        w: &w_desc,
        y: &y_desc,
    };
    let algo = op.pick_algorithm().ok()?;
    let workspace_size = op.get_workspace_size(algo).ok()?;
    let mut workspace = if workspace_size > 0 {
        Some(dev.stream.alloc_zeros::<u8>(workspace_size).ok()?)
    } else {
        None
    };
    unsafe {
        op.launch(
            algo,
            workspace.as_mut(),
            (1.0f32, 0.0f32),
            input,
            weight,
            &mut y_dev,
        )
        .ok()?;
    }
    Some(y_dev)
}

/// Contiguous block gather: for each outer block, copy `len` floats with strides.
#[cfg(feature = "cuda")]
pub fn block_copy_device(
    input: &cudarc::driver::CudaSlice<f32>,
    out: &mut cudarc::driver::CudaSlice<f32>,
    outer: usize,
    len: usize,
    in_stride: usize,
    out_stride: usize,
    in_offset: usize,
    out_offset: usize,
) -> Option<()> {
    let dev = device::global_device()?;
    unsafe {
        super::kernels::launch_block_copy(
            &dev.stream,
            &dev.kernels.block_copy,
            input,
            out,
            outer as i32,
            len as i32,
            in_stride as i32,
            out_stride as i32,
            in_offset as i32,
            out_offset as i32,
        )
        .ok()?;
    }
    Some(())
}

/// Channel-axis RMS for NCHW (or N·spatial with channels as dim1).
#[cfg(feature = "cuda")]
pub fn rms_norm_channels_device(
    x: &cudarc::driver::CudaSlice<f32>,
    gamma: &cudarc::driver::CudaSlice<f32>,
    n: usize,
    c: usize,
    spatial: usize,
    eps: f32,
) -> Option<cudarc::driver::CudaSlice<f32>> {
    if gamma.len() != c || x.len() != n * c * spatial {
        return None;
    }
    let dev = device::global_device()?;
    let mut out = dev.stream.alloc_zeros::<f32>(x.len()).ok()?;
    unsafe {
        super::kernels::launch_rms_norm_channels(
            &dev.stream,
            &dev.kernels.rms_norm_channels,
            x,
            gamma,
            &mut out,
            n as i32,
            c as i32,
            spatial as i32,
            eps,
        )
        .ok()?;
    }
    Some(out)
}

/// Device RoPE for interleaved last-dim pairs (Wan). `x/cos/sin/out` length = rows*dim.
#[cfg(feature = "cuda")]
pub fn rope_interleaved_device(
    x: &cudarc::driver::CudaSlice<f32>,
    cos: &cudarc::driver::CudaSlice<f32>,
    sin: &cudarc::driver::CudaSlice<f32>,
    dim: usize,
) -> Option<cudarc::driver::CudaSlice<f32>> {
    if dim < 2 || dim % 2 != 0 || x.len() != cos.len() || x.len() != sin.len() || x.len() % dim != 0
    {
        return None;
    }
    let rows = x.len() / dim;
    let dev = device::global_device()?;
    let mut out = dev.stream.alloc_zeros::<f32>(x.len()).ok()?;
    unsafe {
        super::kernels::launch_rope_interleaved(
            &dev.stream,
            &dev.kernels.rope_interleaved,
            x,
            cos,
            sin,
            &mut out,
            rows as i32,
            dim as i32,
        )
        .ok()?;
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bf16_roundtrip_bits() {
        let bytes = f32_to_bf16_bytes(&[1.0, -2.0]);
        assert_eq!(bytes.len(), 4);
        // 1.0 bf16 = 0x3f80
        assert_eq!(u16::from_le_bytes([bytes[0], bytes[1]]), 0x3f80);
    }
}
