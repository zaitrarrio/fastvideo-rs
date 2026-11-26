//! Device convolutions: cuDNN N-D forward with cached plans, and 1×1 as GEMM.
//!
//! The Wan VAE decoder runs a handful of distinct conv shapes thousands of
//! times per clip, so descriptors, the heuristic algorithm choice and the
//! workspace size are built once per shape and the workspace buffer is shared
//! and only ever grows. Bias is added in place by one kernel launch.
//!
//! 3-D convs have backends that round differently: cuDNN N-D in f32, a
//! temporal unfold into a cuDNN conv2d (lets cuDNN pick 2-D Winograd for 3×3),
//! and cuDNN on bf16 operands (bf16 math only). Which is faster depends on the
//! GPU (unfold 2× faster on an RTX A5000, cuDNN N-D 1.6× faster on an RTX
//! 5060 Ti at VAE sizes), but a choice made by timing makes the output depend
//! on timing noise: on an H200 the LTX-2.5 upsampler's first conv times
//! cuDNN f32 1.3 ms against cuDNN bf16 1.1-1.4 ms, and two runs of the same
//! request produced different frames depending on which won. So `auto` (the
//! default) is deterministic: cuDNN on bf16 under bf16 math, cuDNN f32 under
//! f32 math. `FASTVIDEO_CONV3D=cudnn|unfold|cudnn-bf16` forces one;
//! `FASTVIDEO_CONV3D_TIMED=1` restores the per-shape timing (fastest, but not
//! reproducible run to run).

#![cfg(feature = "cuda")]

use std::collections::HashMap;

use cudarc::cudnn::{
    sys, ConvBackwardData, ConvDescriptor, ConvForward, FilterDescriptor, TensorDescriptor,
};
use cudarc::driver::CudaSlice;

use super::device::{global_device, DeviceContext, DeviceError, Result};
use super::envflag::CachedString;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct ConvKey {
    x: Vec<usize>,
    w: Vec<usize>,
    pad: Vec<usize>,
    stride: Vec<usize>,
    /// Per spatial axis; all ones for every Wan convolution.
    dilation: Vec<usize>,
    groups: usize,
    /// A transposed convolution (cuDNN backward-data): `x` is its input, and
    /// `out_pad` the extra length PyTorch calls `output_padding`.
    transposed: bool,
    out_pad: Vec<usize>,
    fma: bool,
    bf16: bool,
}

impl ConvKey {
    /// Dense, undilated, forward: every convolution the video VAEs run.
    fn plain(
        x: &[usize],
        w: &[usize],
        pad: &[usize],
        stride: &[usize],
        fma: bool,
        bf16: bool,
    ) -> Self {
        Self {
            x: x.to_vec(),
            w: w.to_vec(),
            pad: pad.to_vec(),
            stride: stride.to_vec(),
            dilation: vec![1; pad.len()],
            groups: 1,
            transposed: false,
            out_pad: vec![0; pad.len()],
            fma,
            bf16,
        }
    }
}

/// The transposed twin of [`ConvPlan`]: cuDNN's backward-data pass *is* the
/// transposed convolution, with the roles of input and output swapped.
struct ConvTransposePlan {
    conv: ConvDescriptor<f32>,
    /// The transposed conv's input (cuDNN's `dy`).
    x: TensorDescriptor<f32>,
    w: FilterDescriptor<f32>,
    /// The transposed conv's output (cuDNN's `dx`).
    y: TensorDescriptor<f32>,
    algo: sys::cudnnConvolutionBwdDataAlgo_t,
    workspace_bytes: usize,
    y_shape: Vec<usize>,
}

struct ConvPlan {
    conv: ConvDescriptor<f32>,
    x: TensorDescriptor<f32>,
    w: FilterDescriptor<f32>,
    y: TensorDescriptor<f32>,
    algo: sys::cudnnConvolutionFwdAlgo_t,
    workspace_bytes: usize,
    y_shape: Vec<usize>,
}

/// The same plan with bfloat16 operands and F32 accumulation. The VAE's
/// convolutions run at this card's TF32 peak, so they are compute-bound: bf16
/// halves the math, and the casts around it cost less than the time saved.
struct ConvPlanBf16 {
    conv: ConvDescriptor<f32>,
    x: TensorDescriptor<half::bf16>,
    w: FilterDescriptor<half::bf16>,
    y: TensorDescriptor<half::bf16>,
    algo: sys::cudnnConvolutionFwdAlgo_t,
    workspace_bytes: usize,
    y_shape: Vec<usize>,
}

/// Which implementation runs a 3-D convolution for a given shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Conv3dPick {
    Cudnn,
    Unfold,
    /// bf16 operands, F32 accumulate. Only a candidate in fast mode: it changes
    /// the numerics, and exact mode has to stay comparable to the CPU path.
    CudnnBf16,
    /// The in-tree implicit GEMM (ADR-0024 step 3, D6): bf16 operands,
    /// channels-last, f32 accumulate, no cuDNN. `FASTVIDEO_CONV3D=igemm`.
    Igemm,
}

#[derive(Default)]
pub struct ConvCache {
    plans: HashMap<ConvKey, ConvPlan>,
    plans_bf16: HashMap<ConvKey, ConvPlanBf16>,
    plans_transpose: HashMap<ConvKey, ConvTransposePlan>,
    workspace: Option<CudaSlice<u8>>,
    /// Chosen 3-D backend per shape.
    conv3d_pick: HashMap<ConvKey, Conv3dPick>,
}

impl ConvCache {
    pub fn plan_count(&self) -> usize {
        self.plans.len()
    }
}

fn contiguous_strides(shape: &[usize]) -> Vec<i32> {
    let mut s = vec![1i32; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        s[i] = s[i + 1] * shape[i + 1] as i32;
    }
    s
}

fn out_shape(key: &ConvKey) -> Result<Vec<usize>> {
    let (x, w, pad, stride, dil) = (&key.x, &key.w, &key.pad, &key.stride, &key.dilation);
    let spatial = x.len().saturating_sub(2);
    let bad = || {
        DeviceError::Message(format!(
            "conv shape mismatch: x={x:?} w={w:?} pad={pad:?} stride={stride:?} dilation={dil:?} groups={} transposed={}",
            key.groups, key.transposed
        ))
    };
    if w.len() != x.len()
        || [pad.len(), stride.len(), dil.len(), key.out_pad.len()]
            .iter()
            .any(|&l| l != spatial)
        || key.groups == 0
        || stride.iter().chain(dil.iter()).any(|&v| v == 0)
        || x[1] % key.groups != 0
    {
        return Err(bad());
    }
    if key.transposed {
        // PyTorch ConvTranspose weight: [C_in, C_out / groups, k...].
        if w[0] != x[1] {
            return Err(bad());
        }
        let mut out = vec![x[0], w[1] * key.groups];
        for i in 0..spatial {
            let full = (x[2 + i] - 1) * stride[i] + dil[i] * (w[2 + i] - 1) + key.out_pad[i] + 1;
            if x[2 + i] == 0 || full <= 2 * pad[i] {
                return Err(bad());
            }
            out.push(full - 2 * pad[i]);
        }
        return Ok(out);
    }
    // Forward weight: [C_out, C_in / groups, k...].
    if w[1] * key.groups != x[1] || w[0] % key.groups != 0 {
        return Err(bad());
    }
    let mut out = vec![x[0], w[0]];
    for i in 0..spatial {
        let padded = x[2 + i] + 2 * pad[i];
        let reach = dil[i] * (w[2 + i] - 1) + 1;
        if padded < reach {
            return Err(DeviceError::Message(format!(
                "conv kernel larger than input: x={x:?} w={w:?} pad={pad:?} dilation={dil:?}"
            )));
        }
        out.push((padded - reach) / stride[i] + 1);
    }
    Ok(out)
}

fn conv_descriptor(dev: &DeviceContext, key: &ConvKey) -> Result<ConvDescriptor<f32>> {
    let i32s = |v: &[usize]| v.iter().map(|&p| p as i32).collect::<Vec<i32>>();
    let mut conv = dev.cudnn.create_convnd::<f32>(
        &i32s(&key.pad),
        &i32s(&key.stride),
        &i32s(&key.dilation),
        sys::cudnnConvolutionMode_t::CUDNN_CROSS_CORRELATION,
    )?;
    if key.groups > 1 {
        conv.set_group_count(key.groups as i32)?;
    }
    Ok(conv)
}

/// TF32 is off for exact runs (`FASTVIDEO_TF32=0`): cuDNN's default math type
/// allows TF32 kernels on Ampere and newer, FMA math forbids them.
fn fma_math(dev: &DeviceContext) -> bool {
    dev.gemm_math == super::device::GemmMath::F32
}

fn build_plan(dev: &DeviceContext, key: &ConvKey) -> Result<ConvPlan> {
    let cudnn = &dev.cudnn;
    let y_shape = out_shape(key)?;
    // cuDNN wants at least 2 spatial dims; 1-D callers pass a unit height.
    let mut conv = conv_descriptor(dev, key)?;
    conv.set_math_type(if key.fma {
        sys::cudnnMathType_t::CUDNN_FMA_MATH
    } else {
        sys::cudnnMathType_t::CUDNN_DEFAULT_MATH
    })?;
    let dims = |s: &[usize]| s.iter().map(|&d| d as i32).collect::<Vec<i32>>();
    let x = cudnn.create_nd_tensor::<f32>(&dims(&key.x), &contiguous_strides(&key.x))?;
    let w = cudnn
        .create_nd_filter::<f32>(sys::cudnnTensorFormat_t::CUDNN_TENSOR_NCHW, &dims(&key.w))?;
    let y = cudnn.create_nd_tensor::<f32>(&dims(&y_shape), &contiguous_strides(&y_shape))?;
    let op = ConvForward {
        conv: &conv,
        x: &x,
        w: &w,
        y: &y,
    };
    let algo = op.pick_algorithm()?;
    let workspace_bytes = op.get_workspace_size(algo)?;
    super::log::debug(format_args!(
        "conv plan x={:?} w={:?} pad={:?} stride={:?} algo={algo:?} workspace={}B",
        key.x, key.w, key.pad, key.stride, workspace_bytes
    ));
    Ok(ConvPlan {
        conv,
        x,
        w,
        y,
        algo,
        workspace_bytes,
        y_shape,
    })
}

fn build_plan_bf16(dev: &DeviceContext, key: &ConvKey) -> Result<ConvPlanBf16> {
    let cudnn = &dev.cudnn;
    let y_shape = out_shape(key)?;
    // F32 accumulation over bf16 operands; tensor-core math is the whole point,
    // so this plan never asks for FMA_MATH.
    let mut conv = conv_descriptor(dev, key)?;
    conv.set_math_type(sys::cudnnMathType_t::CUDNN_TENSOR_OP_MATH)?;
    let dims = |s: &[usize]| s.iter().map(|&d| d as i32).collect::<Vec<i32>>();
    let x = cudnn.create_nd_tensor::<half::bf16>(&dims(&key.x), &contiguous_strides(&key.x))?;
    let w = cudnn.create_nd_filter::<half::bf16>(
        sys::cudnnTensorFormat_t::CUDNN_TENSOR_NCHW,
        &dims(&key.w),
    )?;
    let y = cudnn.create_nd_tensor::<half::bf16>(&dims(&y_shape), &contiguous_strides(&y_shape))?;
    let op = ConvForward {
        conv: &conv,
        x: &x,
        w: &w,
        y: &y,
    };
    let algo = op.pick_algorithm()?;
    let workspace_bytes = op.get_workspace_size(algo)?;
    super::log::debug(format_args!(
        "conv plan (bf16) x={:?} w={:?} algo={algo:?} workspace={}B",
        key.x, key.w, workspace_bytes
    ));
    Ok(ConvPlanBf16 {
        conv,
        x,
        w,
        y,
        algo,
        workspace_bytes,
        y_shape,
    })
}

/// [`cudnn_conv`] with bf16 operands: cast in, convolve, cast back. Worth it
/// only because these convolutions are compute-bound at TF32 peak.
pub fn cudnn_conv_bf16(
    x: &CudaSlice<f32>,
    x_shape: &[usize],
    w: &CudaSlice<f32>,
    w_shape: &[usize],
    pad: &[usize],
    stride: &[usize],
) -> Result<(CudaSlice<f32>, Vec<usize>)> {
    let dev = global_device()
        .ok_or_else(|| DeviceError::Message("no global CUDA device context".into()))?;
    let cast = |e: super::tensor::TensorError| DeviceError::Message(e.to_string());
    let xb = super::ops::cast_f32_bf16_device(x).map_err(cast)?;
    let wb = super::ops::cast_f32_bf16_device(w).map_err(cast)?;
    let key = ConvKey::plain(x_shape, w_shape, pad, stride, false, true);
    let mut cache = dev.conv.lock().expect("conv cache lock");
    if !cache.plans_bf16.contains_key(&key) {
        let plan = build_plan_bf16(&dev, &key)?;
        cache.plans_bf16.insert(key.clone(), plan);
    }
    let need = cache.plans_bf16[&key].workspace_bytes;
    if need > 0 && cache.workspace.as_ref().is_none_or(|ws| ws.len() < need) {
        cache.workspace = None;
        cache.workspace = Some(unsafe { dev.stream.alloc::<u8>(need) }?);
    }
    let ConvCache {
        plans_bf16,
        workspace,
        ..
    } = &mut *cache;
    let plan = &plans_bf16[&key];
    let n_out: usize = plan.y_shape.iter().product();
    let mut yb = unsafe { dev.stream.alloc::<half::bf16>(n_out) }?;
    let op = ConvForward {
        conv: &plan.conv,
        x: &plan.x,
        w: &plan.w,
        y: &plan.y,
    };
    unsafe {
        // cudarc types alpha/beta as the output element type and converts them
        // to cuDNN's F32 scaling parameter internally.
        let (alpha, beta) = (half::bf16::from_f32(1.0), half::bf16::from_f32(0.0));
        op.launch(
            plan.algo,
            if need > 0 { workspace.as_mut() } else { None },
            (alpha, beta),
            &xb,
            &wb,
            &mut yb,
        )?;
    }
    let y_shape = plan.y_shape.clone();
    drop(cache);
    let y = super::ops::cast_bf16_f32_bias_act_device(&yb, None, false).map_err(cast)?;
    Ok((y, y_shape))
}

/// cuDNN cross-correlation of contiguous NC(D)HW `x` with OI(D)HW `w`
/// (symmetric zero padding). Returns the output buffer and its shape.
pub fn cudnn_conv(
    x: &CudaSlice<f32>,
    x_shape: &[usize],
    w: &CudaSlice<f32>,
    w_shape: &[usize],
    pad: &[usize],
    stride: &[usize],
) -> Result<(CudaSlice<f32>, Vec<usize>)> {
    cudnn_conv_ext(x, x_shape, w, w_shape, pad, stride, &vec![1; pad.len()], 1)
}

fn check_buffers(
    x: &CudaSlice<f32>,
    x_shape: &[usize],
    w: &CudaSlice<f32>,
    w_shape: &[usize],
) -> Result<()> {
    if x.len() != x_shape.iter().product::<usize>() || w.len() != w_shape.iter().product::<usize>()
    {
        return Err(DeviceError::Message(format!(
            "conv buffer mismatch: x.len={} shape={x_shape:?} w.len={} shape={w_shape:?}",
            x.len(),
            w.len()
        )));
    }
    Ok(())
}

/// [`cudnn_conv`] with dilation and groups: `w` is `[C_out, C_in / groups, k...]`.
/// What the audio decoders' dilated residual stacks and depthwise anti-alias
/// filters run on.
#[allow(clippy::too_many_arguments)]
pub fn cudnn_conv_ext(
    x: &CudaSlice<f32>,
    x_shape: &[usize],
    w: &CudaSlice<f32>,
    w_shape: &[usize],
    pad: &[usize],
    stride: &[usize],
    dilation: &[usize],
    groups: usize,
) -> Result<(CudaSlice<f32>, Vec<usize>)> {
    cudnn_conv_ext_math(x, x_shape, w, w_shape, pad, stride, dilation, groups, None)
}

/// [`cudnn_conv`] with FMA math whatever `FASTVIDEO_TF32` says: for a
/// network whose output is checked byte for byte against an f32 host
/// reference (the Real-ESRGAN upscaler, `rt::esrgan`): with the TF32
/// default a 480p ×4 frame had 7 % of its 8-bit output one level off the
/// host's (`fv-gpucheck kernels --groups esrgan`, 2026-11-18).
pub fn cudnn_conv_fma(
    x: &CudaSlice<f32>,
    x_shape: &[usize],
    w: &CudaSlice<f32>,
    w_shape: &[usize],
    pad: &[usize],
    stride: &[usize],
) -> Result<(CudaSlice<f32>, Vec<usize>)> {
    cudnn_conv_ext_math(
        x,
        x_shape,
        w,
        w_shape,
        pad,
        stride,
        &vec![1; pad.len()],
        1,
        Some(true),
    )
}

/// [`cudnn_conv_ext`] with the math type given (`None`: the device's,
/// [`fma_math`]).
#[allow(clippy::too_many_arguments)]
fn cudnn_conv_ext_math(
    x: &CudaSlice<f32>,
    x_shape: &[usize],
    w: &CudaSlice<f32>,
    w_shape: &[usize],
    pad: &[usize],
    stride: &[usize],
    dilation: &[usize],
    groups: usize,
    fma: Option<bool>,
) -> Result<(CudaSlice<f32>, Vec<usize>)> {
    let dev = global_device()
        .ok_or_else(|| DeviceError::Message("no global CUDA device context".into()))?;
    check_buffers(x, x_shape, w, w_shape)?;
    let fma = fma.unwrap_or_else(|| fma_math(&dev));
    let mut key = ConvKey::plain(x_shape, w_shape, pad, stride, fma, false);
    key.dilation = dilation.to_vec();
    key.groups = groups;
    let mut cache = dev.conv.lock().expect("conv cache lock");
    if !cache.plans.contains_key(&key) {
        let plan = build_plan(&dev, &key)?;
        cache.plans.insert(key.clone(), plan);
    }
    let need = cache.plans[&key].workspace_bytes;
    if need > 0 && cache.workspace.as_ref().is_none_or(|ws| ws.len() < need) {
        cache.workspace = None;
        cache.workspace = Some(unsafe { dev.stream.alloc::<u8>(need) }?);
    }
    let ConvCache {
        plans, workspace, ..
    } = &mut *cache;
    let plan = &plans[&key];
    let n_out: usize = plan.y_shape.iter().product();
    let mut y = unsafe { dev.stream.alloc::<f32>(n_out) }?;
    let op = ConvForward {
        conv: &plan.conv,
        x: &plan.x,
        w: &plan.w,
        y: &plan.y,
    };
    unsafe {
        op.launch(
            plan.algo,
            if need > 0 { workspace.as_mut() } else { None },
            (1.0f32, 0.0f32),
            x,
            w,
            &mut y,
        )?;
    }
    Ok((y, plan.y_shape.clone()))
}

fn build_transpose_plan(dev: &DeviceContext, key: &ConvKey) -> Result<ConvTransposePlan> {
    let cudnn = &dev.cudnn;
    let y_shape = out_shape(key)?;
    let mut conv = conv_descriptor(dev, key)?;
    conv.set_math_type(if key.fma {
        sys::cudnnMathType_t::CUDNN_FMA_MATH
    } else {
        sys::cudnnMathType_t::CUDNN_DEFAULT_MATH
    })?;
    let dims = |s: &[usize]| s.iter().map(|&d| d as i32).collect::<Vec<i32>>();
    let x = cudnn.create_nd_tensor::<f32>(&dims(&key.x), &contiguous_strides(&key.x))?;
    let w = cudnn
        .create_nd_filter::<f32>(sys::cudnnTensorFormat_t::CUDNN_TENSOR_NCHW, &dims(&key.w))?;
    let y = cudnn.create_nd_tensor::<f32>(&dims(&y_shape), &contiguous_strides(&y_shape))?;
    let op = ConvBackwardData {
        conv: &conv,
        dx: &y,
        w: &w,
        dy: &x,
    };
    let algo =
        deterministic_bwd_data_algo(op.pick_algorithm()?, |a| op.get_workspace_size(a).is_ok());
    let workspace_bytes = op.get_workspace_size(algo)?;
    super::log::debug(format_args!(
        "conv-transpose plan x={:?} w={:?} pad={:?} stride={:?} algo={algo:?} workspace={}B",
        key.x, key.w, key.pad, key.stride, workspace_bytes
    ));
    Ok(ConvTransposePlan {
        conv,
        x,
        w,
        y,
        algo,
        workspace_bytes,
        y_shape,
    })
}

/// The backward-data algorithm a transposed convolution runs: the
/// heuristic's first choice, unless that is `ALGO_0`, the one backward-data
/// algorithm cuDNN documents as nondeterministic (it accumulates with
/// atomics, so the output bits depend on scheduling). Then `ALGO_1` (the
/// deterministic GEMM form) when cuDNN accepts it for the shape.
/// `FASTVIDEO_CONV_BWD_DATA_NONDETERMINISTIC=1` keeps the heuristic's pick.
pub fn deterministic_bwd_data_algo(
    picked: cudarc::cudnn::sys::cudnnConvolutionBwdDataAlgo_t,
    accepts: impl Fn(cudarc::cudnn::sys::cudnnConvolutionBwdDataAlgo_t) -> bool,
) -> cudarc::cudnn::sys::cudnnConvolutionBwdDataAlgo_t {
    use cudarc::cudnn::sys::cudnnConvolutionBwdDataAlgo_t as B;
    if picked != B::CUDNN_CONVOLUTION_BWD_DATA_ALGO_0
        || super::envflag::bool_flag("FASTVIDEO_CONV_BWD_DATA_NONDETERMINISTIC", false)
    {
        return picked;
    }
    if accepts(B::CUDNN_CONVOLUTION_BWD_DATA_ALGO_1) {
        B::CUDNN_CONVOLUTION_BWD_DATA_ALGO_1
    } else {
        picked
    }
}

/// Transposed convolution, PyTorch `ConvTranspose` semantics: `w` is
/// `[C_in, C_out / groups, k...]` and the output length per axis is
/// `(L - 1) * stride - 2 * pad + dilation * (k - 1) + out_pad + 1`. Runs as
/// cuDNN's backward-data pass, which is the same arithmetic.
#[allow(clippy::too_many_arguments)]
pub fn cudnn_conv_transpose(
    x: &CudaSlice<f32>,
    x_shape: &[usize],
    w: &CudaSlice<f32>,
    w_shape: &[usize],
    pad: &[usize],
    stride: &[usize],
    dilation: &[usize],
    groups: usize,
    out_pad: &[usize],
) -> Result<(CudaSlice<f32>, Vec<usize>)> {
    let dev = global_device()
        .ok_or_else(|| DeviceError::Message("no global CUDA device context".into()))?;
    check_buffers(x, x_shape, w, w_shape)?;
    let mut key = ConvKey::plain(x_shape, w_shape, pad, stride, fma_math(&dev), false);
    key.dilation = dilation.to_vec();
    key.groups = groups;
    key.transposed = true;
    key.out_pad = out_pad.to_vec();
    let mut cache = dev.conv.lock().expect("conv cache lock");
    if !cache.plans_transpose.contains_key(&key) {
        let plan = build_transpose_plan(&dev, &key)?;
        cache.plans_transpose.insert(key.clone(), plan);
    }
    let need = cache.plans_transpose[&key].workspace_bytes;
    if need > 0 && cache.workspace.as_ref().is_none_or(|ws| ws.len() < need) {
        cache.workspace = None;
        cache.workspace = Some(unsafe { dev.stream.alloc::<u8>(need) }?);
    }
    let ConvCache {
        plans_transpose,
        workspace,
        ..
    } = &mut *cache;
    let plan = &plans_transpose[&key];
    let n_out: usize = plan.y_shape.iter().product();
    // Backward-data with beta = 0 writes every output element, including the
    // `out_pad` tail no input reaches (it gets zero), so no pre-clear is needed.
    let mut y = unsafe { dev.stream.alloc::<f32>(n_out) }?;
    let op = ConvBackwardData {
        conv: &plan.conv,
        dx: &plan.y,
        w: &plan.w,
        dy: &plan.x,
    };
    unsafe {
        op.launch(
            plan.algo,
            if need > 0 { workspace.as_mut() } else { None },
            (1.0f32, 0.0f32),
            &mut y,
            w,
            x,
        )?;
    }
    Ok((y, plan.y_shape.clone()))
}

static CONV3D_BACKEND: CachedString = CachedString::new();

/// `auto` (default), `cudnn` or `unfold`.
pub fn conv3d_backend() -> String {
    CONV3D_BACKEND.get_or_init(|| {
        fastvideo_models::techniques::kernels::choice(
            fastvideo_models::techniques::kernels::KernelOp::Conv3d,
        )
    })
}

/// `FASTVIDEO_CONV3D_TIMED=1`: `auto` times the backends per shape (the
/// pre-parity behaviour; its output depends on which backend won).
fn conv3d_timed() -> bool {
    static FLAG: super::envflag::CachedBool = super::envflag::CachedBool::new();
    FLAG.get_or_init(|| super::envflag::bool_flag("FASTVIDEO_CONV3D_TIMED", false))
}

/// 3-D conv with the backend chosen per [`conv3d_backend`]. `auto` is cuDNN
/// on bf16 under bf16 math and cuDNN f32 under f32 math, whatever the timing.
/// With [`conv3d_timed`] the first call for a shape runs each backend twice,
/// times the second run of each (synchronized), keeps the faster one's output
/// and remembers the choice.
pub fn conv3d(
    x: &CudaSlice<f32>,
    x_shape: &[usize],
    w: &CudaSlice<f32>,
    w_shape: &[usize],
    pad: [usize; 3],
    stride: [usize; 3],
) -> Result<(CudaSlice<f32>, Vec<usize>)> {
    let run = |pick: Conv3dPick| -> Result<(CudaSlice<f32>, Vec<usize>)> {
        match pick {
            Conv3dPick::Unfold if pad[0] == 0 => {
                conv3d_unfold(x, x_shape, w, w_shape, [pad[1], pad[2]], stride)
            }
            Conv3dPick::CudnnBf16 => cudnn_conv_bf16(x, x_shape, w, w_shape, &pad, &stride),
            Conv3dPick::Igemm => conv3d_igemm(x, x_shape, w, w_shape, &pad, &stride),
            _ => cudnn_conv(x, x_shape, w, w_shape, &pad, &stride),
        }
    };
    match conv3d_backend().as_str() {
        "cudnn" => return run(Conv3dPick::Cudnn),
        "unfold" => return run(Conv3dPick::Unfold),
        "cudnn-bf16" => return run(Conv3dPick::CudnnBf16),
        "igemm" => return run(Conv3dPick::Igemm),
        _ => {}
    }
    let dev = global_device()
        .ok_or_else(|| DeviceError::Message("no global CUDA device context".into()))?;
    if !conv3d_timed() {
        return run(if fma_math(&dev) {
            Conv3dPick::Cudnn
        } else {
            Conv3dPick::CudnnBf16
        });
    }
    let key = ConvKey::plain(x_shape, w_shape, &pad, &stride, fma_math(&dev), false);
    let known = dev
        .conv
        .lock()
        .expect("conv cache lock")
        .conv3d_pick
        .get(&key)
        .copied();
    if let Some(pick) = known {
        return run(pick);
    }
    // Timing drops every output before the next candidate runs: holding the
    // winner-so-far alive while the next one allocates its own doubles the peak,
    // and exact-mode parity already sits at the edge of a 24GB card. The winner
    // is re-run once at the end, which costs one convolution per shape, once.
    let timed = |pick: Conv3dPick| -> Result<f64> {
        drop(run(pick)?);
        dev.synchronize()?;
        let t = std::time::Instant::now();
        drop(run(pick)?);
        dev.synchronize()?;
        Ok(t.elapsed().as_secs_f64())
    };
    // Exact mode must stay comparable to the CPU path, so bf16 only competes
    // when the context is already running reduced-precision math.
    let mut best: Option<(f64, Conv3dPick)> = None;
    let mut report: Vec<String> = Vec::new();
    let candidates: &[Conv3dPick] = if fma_math(&dev) {
        &[Conv3dPick::Cudnn, Conv3dPick::Unfold]
    } else {
        &[Conv3dPick::Cudnn, Conv3dPick::Unfold, Conv3dPick::CudnnBf16]
    };
    for &pick in candidates {
        let secs = timed(pick)?;
        report.push(format!("{pick:?} {:.1}ms", secs * 1e3));
        if best.is_none_or(|(b, _)| secs < b) {
            best = Some((secs, pick));
        }
    }
    let (_, pick) = best.expect("at least one conv3d backend");
    super::log::info(format_args!(
        "conv3d x={x_shape:?} w={w_shape:?}: {} → {pick:?}",
        report.join(", ")
    ));
    dev.conv
        .lock()
        .expect("conv cache lock")
        .conv3d_pick
        .insert(key, pick);
    run(pick)
}

/// 3-D conv through a temporal unfold: gather every `kt`-frame window into
/// channels (`[n*ot, c*kt, h, w]`), run one cuDNN conv2d with the weight
/// reshaped to `[oc, ic*kt, kh, kw]`, and permute back to `[n, oc, ot, oh, ow]`.
pub fn conv3d_unfold(
    x: &CudaSlice<f32>,
    x_shape: &[usize],
    w: &CudaSlice<f32>,
    w_shape: &[usize],
    pad_hw: [usize; 2],
    stride: [usize; 3],
) -> Result<(CudaSlice<f32>, Vec<usize>)> {
    let dev = global_device()
        .ok_or_else(|| DeviceError::Message("no global CUDA device context".into()))?;
    let (n, c, t, h, wd) = (x_shape[0], x_shape[1], x_shape[2], x_shape[3], x_shape[4]);
    let (oc, kt) = (w_shape[0], w_shape[2]);
    let st = stride[0].max(1);
    if t < kt {
        return Err(DeviceError::Message(format!(
            "conv3d unfold: t={t} < kt={kt}"
        )));
    }
    let ot = (t - kt) / st + 1;
    let unfolded_shape = [n * ot, c * kt, h, wd];
    let total: usize = unfolded_shape.iter().product();
    let mut unfolded = unsafe { dev.stream.alloc::<f32>(total) }?;
    let (c_i, t_i, h_i, w_i, kt_i, st_i, ot_i) = (
        c as i64, t as i64, h as i64, wd as i64, kt as i64, st as i64, ot as i64,
    );
    let total_i = total as i64;
    super::kernels::launch!(dev.stream, &dev.kernels.temporal_unfold, super::kernels::cfg_n(total);
        x, &mut unfolded, &total_i, &c_i, &t_i, &h_i, &w_i, &kt_i, &st_i, &ot_i)?;
    let w2 = [oc, c * kt, w_shape[3], w_shape[4]];
    let (y2, y2_shape) = cudnn_conv(
        &unfolded,
        &unfolded_shape,
        w,
        &w2,
        &pad_hw,
        &[stride[1], stride[2]],
    )?;
    drop(unfolded);
    // [n*ot, oc, oh, ow] → [n, oc, ot, oh, ow].
    let (oh, ow) = (y2_shape[2], y2_shape[3]);
    let out_shape = vec![n, oc, ot, oh, ow];
    let perm = super::ops::gather_nd_device(&y2, &[n, ot, oc, oh, ow], &[0, 2, 1, 3, 4])
        .map_err(|e| DeviceError::Message(e.to_string()))?;
    Ok((perm, out_shape))
}

// ---- cuDNN-free routes (ADR-0024 step 3, WO-3a spike) -----------------------
//
// Two routes for the dense N-D convolution that never call cuDNN, timed
// against `cudnn_conv_bf16` by fv-gpucheck's `conv_routes` group
// (docs/plan/2026-11-25-adr24-step3-wo.md). Both take the NC(D)HW f32 tensors
// every caller has today, work on channels-last bf16 with f32 accumulation
// (the cast + permute in and the permute out are part of the route and of its
// timing, as the cuDNN bf16 path's casts are) and return NC(D)HW f32.
// Constraints of the spike: C_in a multiple of 8, groups 1, dilation 1.

/// Dense N-D convolution geometry; 4-D `[n, c, h, w]` is a 5-D with depth 1.
#[derive(Debug, Clone, Copy)]
pub struct ConvGeom {
    pub b: usize,
    pub d: usize,
    pub h: usize,
    pub w: usize,
    pub c: usize,
    pub n: usize,
    pub kd: usize,
    pub kh: usize,
    pub kw: usize,
    pub sd: usize,
    pub sh: usize,
    pub sw: usize,
    pub pd: usize,
    pub ph: usize,
    pub pw: usize,
    pub od: usize,
    pub oh: usize,
    pub ow: usize,
}

impl ConvGeom {
    /// `x_shape` `[n, c, (d,) h, w]`, `w_shape` `[oc, ic, (kd,) kh, kw]`, `pad`
    /// and `stride` per spatial axis (2 or 3 entries, as the shapes).
    pub fn new(
        x_shape: &[usize],
        w_shape: &[usize],
        pad: &[usize],
        stride: &[usize],
    ) -> Result<Self> {
        let rank = x_shape.len();
        let bad = |m: String| DeviceError::Message(format!("conv route: {m}"));
        if !(rank == 4 || rank == 5)
            || w_shape.len() != rank
            || pad.len() != rank - 2
            || stride.len() != rank - 2
        {
            return Err(bad(format!(
                "x {x_shape:?} w {w_shape:?} pad {pad:?} stride {stride:?}: NCHW or NCDHW with one pad/stride per spatial axis"
            )));
        }
        let (b, c) = (x_shape[0], x_shape[1]);
        let (d, h, w) = if rank == 5 {
            (x_shape[2], x_shape[3], x_shape[4])
        } else {
            (1, x_shape[2], x_shape[3])
        };
        let (n, ic) = (w_shape[0], w_shape[1]);
        let (kd, kh, kw) = if rank == 5 {
            (w_shape[2], w_shape[3], w_shape[4])
        } else {
            (1, w_shape[2], w_shape[3])
        };
        let (pd, ph, pw) = if rank == 5 {
            (pad[0], pad[1], pad[2])
        } else {
            (0, pad[0], pad[1])
        };
        let (sd, sh, sw) = if rank == 5 {
            (stride[0], stride[1], stride[2])
        } else {
            (1, stride[0], stride[1])
        };
        if ic != c {
            return Err(bad(format!("x has {c} channels, w expects {ic}")));
        }
        if c % 8 != 0 {
            return Err(bad(format!(
                "C_in = {c} is not a multiple of 8 (the spike's cp.async chunk constraint)"
            )));
        }
        if sd == 0 || sh == 0 || sw == 0 {
            return Err(bad("zero stride".into()));
        }
        let out = |i: usize, k: usize, p: usize, s: usize| -> Result<usize> {
            (i + 2 * p)
                .checked_sub(k)
                .map(|v| v / s + 1)
                .ok_or_else(|| bad(format!("kernel {k} larger than padded input {}", i + 2 * p)))
        };
        Ok(Self {
            b,
            d,
            h,
            w,
            c,
            n,
            kd,
            kh,
            kw,
            sd,
            sh,
            sw,
            pd,
            ph,
            pw,
            od: out(d, kd, pd, sd)?,
            oh: out(h, kh, ph, sh)?,
            ow: out(w, kw, pw, sw)?,
        })
    }

    /// Output pixels (GEMM rows).
    pub fn m(&self) -> usize {
        self.b * self.od * self.oh * self.ow
    }

    /// GEMM depth: taps × C_in.
    pub fn k(&self) -> usize {
        self.kd * self.kh * self.kw * self.c
    }

    fn out_shape(&self, rank: usize) -> Vec<usize> {
        if rank == 5 {
            vec![self.b, self.n, self.od, self.oh, self.ow]
        } else {
            vec![self.b, self.n, self.oh, self.ow]
        }
    }
}

fn tiles32(n: usize) -> u32 {
    n.div_ceil(32).max(1) as u32
}

/// NC(D)HW f32 → N(D)HWC bf16 (one tiled transpose + cast).
fn route_x_nhwc(
    dev: &DeviceContext,
    x: &CudaSlice<f32>,
    g: &ConvGeom,
) -> Result<CudaSlice<half::bf16>> {
    let s = g.d * g.h * g.w;
    let mut out = unsafe { dev.stream.alloc::<half::bf16>((g.b * s * g.c).max(1)) }?;
    let cfg = cudarc::driver::LaunchConfig {
        grid_dim: (tiles32(s), tiles32(g.c), g.b.max(1) as u32),
        block_dim: (32, 8, 1),
        shared_mem_bytes: 0,
    };
    let (c_i, s_i) = (g.c as i64, s as i64);
    super::kernels::launch!(dev.stream, &dev.kernels.conv_nchw_to_nhwc_bf16, cfg; x, &mut out, &c_i, &s_i)?;
    Ok(out)
}

/// `[splits]` × N(D)HWC f32 → NC(D)HW f32, the split-K planes summed in order.
fn route_y_nchw(
    dev: &DeviceContext,
    y: &CudaSlice<f32>,
    g: &ConvGeom,
    splits: usize,
) -> Result<CudaSlice<f32>> {
    let s = g.od * g.oh * g.ow;
    let plane = g.b * s * g.n;
    let mut out = unsafe { dev.stream.alloc::<f32>(plane.max(1)) }?;
    let cfg = cudarc::driver::LaunchConfig {
        grid_dim: (tiles32(g.n), tiles32(s), g.b.max(1) as u32),
        block_dim: (32, 8, 1),
        shared_mem_bytes: 0,
    };
    let (c_i, s_i, z_i, plane_i) = (g.n as i64, s as i64, splits as i64, plane as i64);
    super::kernels::launch!(dev.stream, &dev.kernels.conv_nhwc_to_nchw_f32, cfg; y, &mut out, &c_i, &s_i, &z_i, &plane_i)?;
    Ok(out)
}

/// The implicit GEMM's launch plan for one geometry: the warp tile and the
/// deterministic split-K factor on the 128 × 128 × 32 3-stage CTA tile (the
/// best of every tile tried on 2026-11-25, `docs/perf/2026-11-25-conv-routes.md`).
/// Chosen from shapes and the SM count, never from timing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IgemmPlan {
    /// The kernel variant ([`IGEMM_VARIANTS`]).
    pub variant: IgemmVariant,
    /// Split-K factor (grid.z); the planes are summed in order.
    pub splits: usize,
}

/// One instantiation of the kernel file (`conv_igemm_bf16*`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IgemmVariant {
    pub name: &'static str,
    /// CTA tile rows (output pixels).
    pub bm: usize,
    /// CTA tile columns (output channels).
    pub bn: usize,
    pub threads: u32,
    /// Dynamic shared memory the launch asks for (0: the kernel's static smem).
    pub dyn_smem: u32,
    /// Timing probe (fv-gpucheck only, wrong values).
    pub probe: bool,
    /// The TMA-fed kernel (`conv_igemm_bf16_tma*`, sm_90+, BK = 64): tensor
    /// maps instead of pointers, M tiled per output row.
    pub tma: bool,
}

const fn variant(
    name: &'static str,
    bm: usize,
    bn: usize,
    threads: u32,
    dyn_smem: u32,
    probe: bool,
) -> IgemmVariant {
    IgemmVariant {
        name,
        bm,
        bn,
        threads,
        dyn_smem,
        probe,
        tma: false,
    }
}

const fn tma_variant(name: &'static str, probe: bool) -> IgemmVariant {
    IgemmVariant {
        name,
        bm: 128,
        bn: 128,
        threads: 288,
        dyn_smem: IGEMM_TMA_SMEM,
        probe,
        tma: true,
    }
}

/// The TMA kernel's dynamic shared memory: three 32 KB stages, 1 KB of
/// alignment slack, the mbarriers (within CC 12.0's 99 KB per block).
pub const IGEMM_TMA_SMEM: u32 = 3 * 32 * 1024 + 1024 + 64;

/// The variants the kernel file instantiates, timed side by side by
/// fv-gpucheck `conv_routes`; the first is the default.
pub const IGEMM_VARIANTS: &[IgemmVariant] = &[
    variant("w32", 128, 128, 256, 0, false),
    variant("w64", 128, 128, 128, 0, false),
    variant("ws", 128, 128, 160, 0, false),
    variant("ws2", 128, 128, 192, 0, false),
    tma_variant("tma", false),
    variant("probe_dense", 128, 128, 256, 0, true),
    variant("probe_noload", 128, 128, 256, 0, true),
    variant("ws_probe_noepi", 128, 128, 160, 0, true),
    tma_variant("tma_probe_noepi", true),
];

impl IgemmVariant {
    pub fn by_name(name: &str) -> Option<Self> {
        IGEMM_VARIANTS.iter().copied().find(|v| v.name == name)
    }
}

/// Whether the TMA kernel can run this geometry on this device: sm_90+,
/// C a multiple of 64 (a k-tile never straddles a tap; 128-byte rows), conv
/// strides of 1 or 2 (the box traverses `stride` elements per loaded row:
/// `boxDim = block * stride <= 256`, `elementStrides = stride`), W and the
/// strides within the tensor map's limits.
pub fn igemm_tma_eligible(g: &ConvGeom, sm_major: i32) -> bool {
    sm_major >= 9
        && g.c % 64 == 0
        && (1..=2).contains(&g.sw)
        && (1..=2).contains(&g.sh)
        && (1..=2).contains(&g.sd)
        && g.w <= (1 << 30)
        && (g.b * g.d * g.h * g.w * g.c) < (1u64 << 40) as usize
}

/// The TMA kernel's M tile as a (BD, BH, BW) block of output pixels, each a
/// power of two with product 128, BW >= 8: the block that covers the output
/// volume in the fewest tiles (ties to the widest, then the tallest), so a
/// 24 x 24 x 16 LTX latent is 2 x 8 x 8 blocks exactly and a 1080p row is
/// 1 x 1 x 128.
pub fn igemm_tma_block(g: &ConvGeom) -> (usize, usize, usize) {
    let mut best = (usize::MAX, (1usize, 1usize, 128usize));
    for bwl2 in (3..=7).rev() {
        for bhl2 in (0..=(7 - bwl2)).rev() {
            let (bw, bh) = (1usize << bwl2, 1usize << bhl2);
            let bd = 128 / (bw * bh);
            let tiles = g.ow.div_ceil(bw) * g.oh.div_ceil(bh) * g.od.div_ceil(bd);
            if tiles < best.0 {
                best = (tiles, (bd, bh, bw));
            }
        }
    }
    best.1
}

impl IgemmPlan {
    /// The default plan: the first variant, split-K up to four waves.
    pub fn for_geom(g: &ConvGeom, sms: usize, sm_major: i32) -> Self {
        Self::with(g, sms, sm_major, IGEMM_VARIANTS[0])
    }

    /// The M tiles of a variant: the TMA kernel tiles the output plane in
    /// BH x BW patches ([`igemm_tma_bw`]); the others tile the flattened M.
    pub fn m_tiles(g: &ConvGeom, v: IgemmVariant) -> usize {
        if v.tma {
            let (bd, bh, bw) = igemm_tma_block(g);
            g.b * g.od.div_ceil(bd) * g.oh.div_ceil(bh) * g.ow.div_ceil(bw)
        } else {
            g.m().div_ceil(v.bm)
        }
    }

    /// A plan with the given variant; the split from the SM count: enough
    /// splits for four waves of two CTAs per SM, each split keeping at least
    /// eight k-tiles, at most 16.
    pub fn with(g: &ConvGeom, sms: usize, sm_major: i32, variant: IgemmVariant) -> Self {
        // A TMA variant the geometry or device cannot run falls back to the
        // base kernel (fv-gpucheck reports `igemm_tma_eligible`).
        let variant = if variant.tma && !igemm_tma_eligible(g, sm_major) {
            IGEMM_VARIANTS[0]
        } else {
            variant
        };
        let bk = if variant.tma { 64 } else { 32 };
        let nk = g.k().div_ceil(bk);
        // Four waves: of two CTAs per SM for the base kernels, one for TMA.
        let waves4 = if variant.tma { 4 } else { 8 } * sms.max(1);
        let ctas = Self::m_tiles(g, variant) * g.n.div_ceil(variant.bn);
        let mut splits = 1;
        if ctas < waves4 {
            splits = waves4.div_ceil(ctas).min(nk / 8).clamp(1, 16);
        }
        Self { variant, splits }
    }
}

/// [`IgemmPlan::with`] for these shapes on this device.
pub fn igemm_plan_with(
    x_shape: &[usize],
    w_shape: &[usize],
    pad: &[usize],
    stride: &[usize],
    variant: &str,
) -> Result<IgemmPlan> {
    let dev = global_device()
        .ok_or_else(|| DeviceError::Message("no global CUDA device context".into()))?;
    let g = ConvGeom::new(x_shape, w_shape, pad, stride)?;
    let v = IgemmVariant::by_name(variant)
        .ok_or_else(|| DeviceError::Message(format!("conv igemm: no variant {variant}")))?;
    Ok(IgemmPlan::with(
        &g,
        super::ops::sm_count(&dev),
        dev.sm_major,
        v,
    ))
}

/// Whether the TMA kernel would run these shapes on this device.
pub fn igemm_tma_eligible_for(
    x_shape: &[usize],
    w_shape: &[usize],
    pad: &[usize],
    stride: &[usize],
) -> Result<bool> {
    let dev = global_device()
        .ok_or_else(|| DeviceError::Message("no global CUDA device context".into()))?;
    let g = ConvGeom::new(x_shape, w_shape, pad, stride)?;
    Ok(igemm_tma_eligible(&g, dev.sm_major))
}

/// The plan [`conv3d_igemm`] would use for these shapes on this device.
pub fn igemm_plan(
    x_shape: &[usize],
    w_shape: &[usize],
    pad: &[usize],
    stride: &[usize],
) -> Result<IgemmPlan> {
    let dev = global_device()
        .ok_or_else(|| DeviceError::Message("no global CUDA device context".into()))?;
    let g = ConvGeom::new(x_shape, w_shape, pad, stride)?;
    Ok(IgemmPlan::for_geom(
        &g,
        super::ops::sm_count(&dev),
        dev.sm_major,
    ))
}

/// `[oc, ic, taps]` f32 → bf16 `[oc, K]` (`kn` false) or `[K, oc]` (`kn` true),
/// k = tap × C + ci.
fn route_weight(
    dev: &DeviceContext,
    w: &CudaSlice<f32>,
    g: &ConvGeom,
    kn: bool,
) -> Result<CudaSlice<half::bf16>> {
    let taps = g.kd * g.kh * g.kw;
    let total = g.n * g.c * taps;
    if w.len() != total {
        return Err(DeviceError::Message(format!(
            "conv route: weight has {} elements, geometry says {total}",
            w.len()
        )));
    }
    let mut out = unsafe { dev.stream.alloc::<half::bf16>(total.max(1)) }?;
    let (oc_i, ic_i, taps_i, kn_i) = (g.n as i64, g.c as i64, taps as i64, i32::from(kn));
    super::kernels::launch!(dev.stream, &dev.kernels.conv_weight_bf16, super::kernels::cfg_n(total); w, &mut out, &oc_i, &ic_i, &taps_i, &kn_i)?;
    Ok(out)
}

/// The function and launch config of a plan: the variant's kernel, grid
/// (N tiles, M tiles, splits), threads, and its dynamic shared memory
/// (opted in once per function; the driver call is idempotent).
fn igemm_launch<'a>(
    dev: &'a DeviceContext,
    plan: &IgemmPlan,
    g: &ConvGeom,
    n: usize,
) -> Result<(
    &'a cudarc::driver::CudaFunction,
    cudarc::driver::LaunchConfig,
)> {
    let v = plan.variant;
    let func: &cudarc::driver::CudaFunction = match v.name {
        "w64" => &dev.kernels.conv_igemm_bf16_w64,
        "ws" => &dev.kernels.conv_igemm_bf16_ws,
        "ws2" => &dev.kernels.conv_igemm_bf16_ws2,
        "ws_probe_noepi" => &dev.kernels.conv_igemm_bf16_ws_probe_noepi,
        "tma" => &dev.kernels.conv_igemm_bf16_tma,
        "tma_probe_noepi" => &dev.kernels.conv_igemm_bf16_tma_probe_noepi,
        "probe_dense" => &dev.kernels.conv_igemm_bf16_probe_dense,
        "probe_noload" => &dev.kernels.conv_igemm_bf16_probe_noload,
        _ => &dev.kernels.conv_igemm_bf16,
    };
    if v.dyn_smem > 0 {
        super::ops::opt_in_dynamic_shared(func, v.dyn_smem)
            .map_err(|e| DeviceError::Message(e.to_string()))?;
    }
    let cfg = cudarc::driver::LaunchConfig {
        grid_dim: (
            n.div_ceil(v.bn) as u32,
            IgemmPlan::m_tiles(g, v) as u32,
            plan.splits as u32,
        ),
        block_dim: (v.threads, 1, 1),
        shared_mem_bytes: v.dyn_smem,
    };
    Ok((func, cfg))
}

/// The TMA kernel's tensor maps: the activation as a 5-D bf16 tensor
/// `[b, d, h, w, c]` (box `[1, 1, 1, 128 w, 64 c]`, 128-byte swizzle, zeros
/// outside: the padding) and the weights as 2-D `[n, K]` (box `[128, 64]`).
fn igemm_tma_maps(
    dev: &DeviceContext,
    x: cudarc::driver::sys::CUdeviceptr,
    w: cudarc::driver::sys::CUdeviceptr,
    g: &ConvGeom,
) -> Result<(super::ops::FvTensorMap, super::ops::FvTensorMap)> {
    use cudarc::driver::sys::{
        self, CUtensorMapDataType, CUtensorMapFloatOOBfill, CUtensorMapInterleave,
        CUtensorMapL2promotion, CUtensorMapSwizzle,
    };
    let _ = dev;
    let encode = |ptr: sys::CUdeviceptr,
                  rank: u32,
                  dims: &[u64],
                  strides: &[u64],
                  boxd: &[u32],
                  es: &[u32]|
     -> Result<super::ops::FvTensorMap> {
        let mut raw = std::mem::MaybeUninit::<sys::CUtensorMap>::zeroed();
        let st = unsafe {
            sys::cuTensorMapEncodeTiled(
                raw.as_mut_ptr(),
                CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
                rank,
                ptr as *mut std::ffi::c_void,
                dims.as_ptr(),
                strides.as_ptr(),
                boxd.as_ptr(),
                es.as_ptr(),
                CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
                CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B,
                CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
                CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE,
            )
        };
        if st != sys::CUresult::CUDA_SUCCESS {
            return Err(DeviceError::Message(format!(
                "conv igemm tma: cuTensorMapEncodeTiled: {st:?}"
            )));
        }
        let map = unsafe { raw.assume_init() };
        Ok(unsafe { std::mem::transmute_copy(&map) })
    };
    let (b, d, h, w_, c) = (g.b as u64, g.d as u64, g.h as u64, g.w as u64, g.c as u64);
    let xdims = [c, w_, h, d, b];
    let xstrides = [c * 2, w_ * c * 2, h * w_ * c * 2, d * h * w_ * c * 2];
    let (bd, bh, bw) = igemm_tma_block(g);
    let xbox = [
        64u32,
        (bw * g.sw) as u32,
        (bh * g.sh) as u32,
        (bd * g.sd) as u32,
        1,
    ];
    let xes = [1u32, g.sw as u32, g.sh as u32, g.sd as u32, 1];
    let tmx = encode(x, 5, &xdims, &xstrides, &xbox, &xes)?;
    let k = g.k() as u64;
    let wdims = [k, g.n as u64];
    let wstrides = [k * 2];
    let wbox = [64u32, 128];
    let wes = [1u32, 1];
    let tmw = encode(w, 2, &wdims, &wstrides, &wbox, &wes)?;
    Ok((tmx, tmw))
}

/// Launch the TMA kernel of `plan` on channels-last bf16 buffers (the same
/// argument contract as the pointer kernels, minus the dilation).
#[allow(clippy::too_many_arguments)]
fn igemm_tma_launch(
    dev: &DeviceContext,
    plan: &IgemmPlan,
    g: &ConvGeom,
    x: cudarc::driver::sys::CUdeviceptr,
    w: cudarc::driver::sys::CUdeviceptr,
    yf: &mut CudaSlice<f32>,
    y16: &mut cudarc::driver::CudaViewMut<'_, half::bf16>,
    out16: i32,
) -> Result<()> {
    let (tmx, tmw) = igemm_tma_maps(dev, x, w, g)?;
    let (func, cfg) = igemm_launch(dev, plan, g, g.n)?;
    let i = |v: usize| v as i32;
    let (b, d, h, wd, c, od, oh, ow, nn) = (
        i(g.b),
        i(g.d),
        i(g.h),
        i(g.w),
        i(g.c),
        i(g.od),
        i(g.oh),
        i(g.ow),
        i(g.n),
    );
    let (kd, kh, kw, sd, sh, sw, pd, ph, pw) = (
        i(g.kd),
        i(g.kh),
        i(g.kw),
        i(g.sd),
        i(g.sh),
        i(g.sw),
        i(g.pd),
        i(g.ph),
        i(g.pw),
    );
    let splits_i = plan.splits as i32;
    let (_, bh_, bw_) = igemm_tma_block(g);
    let (bwl2, bhl2) = (bw_.trailing_zeros() as i32, bh_.trailing_zeros() as i32);
    super::kernels::launch!(dev.stream, func, cfg;
        &tmx, &tmw, &mut *yf, &mut *y16, &out16, &b, &d, &h, &wd, &c, &od, &oh, &ow, &nn,
        &kd, &kh, &kw, &sd, &sh, &sw, &pd, &ph, &pw, &splits_i, &bwl2, &bhl2)?;
    Ok(())
}

/// Route R2: the implicit GEMM (`conv_igemm_bf16*`, mma.sync bf16, f32
/// accumulate) with the tile and split-K of [`IgemmPlan`]. Same contract as
/// [`cudnn_conv`].
pub fn conv3d_igemm(
    x: &CudaSlice<f32>,
    x_shape: &[usize],
    w: &CudaSlice<f32>,
    w_shape: &[usize],
    pad: &[usize],
    stride: &[usize],
) -> Result<(CudaSlice<f32>, Vec<usize>)> {
    conv3d_igemm_with(x, x_shape, w, w_shape, pad, stride, None)
}

/// [`conv3d_igemm`] with an explicit plan (`None`: [`IgemmPlan::for_geom`]);
/// fv-gpucheck times the variants through it.
#[allow(clippy::too_many_arguments)]
pub fn conv3d_igemm_with(
    x: &CudaSlice<f32>,
    x_shape: &[usize],
    w: &CudaSlice<f32>,
    w_shape: &[usize],
    pad: &[usize],
    stride: &[usize],
    plan: Option<IgemmPlan>,
) -> Result<(CudaSlice<f32>, Vec<usize>)> {
    let dev = global_device()
        .ok_or_else(|| DeviceError::Message("no global CUDA device context".into()))?;
    let g = ConvGeom::new(x_shape, w_shape, pad, stride)?;
    let x16 = route_x_nhwc(&dev, x, &g)?;
    let w16 = route_weight(&dev, w, &g, false)?;
    let (m, n) = (g.m(), g.n);
    let plan =
        plan.unwrap_or_else(|| IgemmPlan::for_geom(&g, super::ops::sm_count(&dev), dev.sm_major));
    let mut y = unsafe { dev.stream.alloc::<f32>((m * n * plan.splits).max(1)) }?;
    if plan.variant.tma {
        use cudarc::driver::DevicePtr;
        let mut y16 = unsafe { dev.stream.alloc::<half::bf16>(1) }?;
        {
            let (xp, _gx) = x16.device_ptr(&dev.stream);
            let (wp, _gw) = w16.device_ptr(&dev.stream);
            let mut y16v = y16.as_view_mut();
            igemm_tma_launch(&dev, &plan, &g, xp, wp, &mut y, &mut y16v, 0)?;
        }
        drop(y16);
        drop(x16);
        drop(w16);
        let out = route_y_nchw(&dev, &y, &g, plan.splits)?;
        return Ok((out, g.out_shape(x_shape.len())));
    }
    let (func, cfg) = igemm_launch(&dev, &plan, &g, n)?;
    let i = |v: usize| v as i32;
    let (b, d, h, wd, c, od, oh, ow, nn) = (
        i(g.b),
        i(g.d),
        i(g.h),
        i(g.w),
        i(g.c),
        i(g.od),
        i(g.oh),
        i(g.ow),
        i(n),
    );
    let (kd, kh, kw, sd, sh, sw, pd, ph, pw) = (
        i(g.kd),
        i(g.kh),
        i(g.kw),
        i(g.sd),
        i(g.sh),
        i(g.sw),
        i(g.pd),
        i(g.ph),
        i(g.pw),
    );
    let (one, splits_i, zero) = (1i32, plan.splits as i32, 0i32);
    // The f32 route never asks for the bf16 epilogue; the kernel still binds a pointer.
    let mut y16 = unsafe { dev.stream.alloc::<half::bf16>(1) }?;
    super::kernels::launch!(dev.stream, func, cfg;
        &x16, &w16, &mut y, &mut y16, &zero, &b, &d, &h, &wd, &c, &od, &oh, &ow, &nn,
        &kd, &kh, &kw, &sd, &sh, &sw, &pd, &ph, &pw, &one, &one, &one, &splits_i)?;
    drop(x16);
    drop(w16);
    let out = route_y_nchw(&dev, &y, &g, plan.splits)?;
    Ok((out, g.out_shape(x_shape.len())))
}

/// The implicit GEMM on channels-last bf16 buffers (the LTX fast decoder,
/// `ltx2/vae_fast.rs`): `x` is `[b, d, h, w, c]` bf16 (a view into the padded
/// activation), `w16` the `[n, kd, kh, kw, c]` bf16 filter (cuDNN's NHWC
/// layout, which is this kernel's `[N, K]`), `y` gets `[b, od, oh, ow, n]`
/// bf16. `x_shape` and `w_shape` give the geometry in the NC(D)HW / OI(D)HW
/// order every conv entry takes (`[b, c, d, h, w]`), whatever the memory
/// layout. One split writes bf16 from the epilogue; more splits sum the f32
/// planes in order (`conv_sum_planes_bf16`). Same semantics as [`cudnn_conv`].
#[allow(clippy::too_many_arguments)]
pub fn conv3d_igemm_nhwc_bf16(
    dev: &DeviceContext,
    x: &cudarc::driver::CudaView<'_, half::bf16>,
    w16: &CudaSlice<half::bf16>,
    y: &mut cudarc::driver::CudaViewMut<'_, half::bf16>,
    x_shape: &[usize],
    w_shape: &[usize],
    pad: &[usize],
    stride: &[usize],
) -> Result<Vec<usize>> {
    use cudarc::driver::DevicePtr;
    let g = ConvGeom::new(x_shape, w_shape, pad, stride)?;
    let (m, n) = (g.m(), g.n);
    if x.len() < g.b * g.d * g.h * g.w * g.c || w16.len() != n * g.k() || y.len() < m * n {
        return Err(DeviceError::Message(format!(
            "conv igemm nhwc: x {} w {} y {} for x {x_shape:?} w {w_shape:?}",
            x.len(),
            w16.len(),
            y.len()
        )));
    }
    let plan = IgemmPlan::for_geom(&g, super::ops::sm_count(dev), dev.sm_major);
    if plan.variant.tma {
        let (xp, _gx) = x.device_ptr(&dev.stream);
        let (wp, _gw) = w16.device_ptr(&dev.stream);
        if plan.splits == 1 {
            let mut yf = unsafe { dev.stream.alloc::<f32>(1) }?;
            igemm_tma_launch(dev, &plan, &g, xp, wp, &mut yf, y, 1)?;
        } else {
            let plane = m * n;
            let mut yf = unsafe { dev.stream.alloc::<f32>(plane * plan.splits) }?;
            igemm_tma_launch(dev, &plan, &g, xp, wp, &mut yf, y, 0)?;
            let (plane_i, z_i) = (plane as i64, plan.splits as i64);
            super::kernels::launch!(dev.stream, &dev.kernels.conv_sum_planes_bf16, super::kernels::cfg_n(plane);
                &yf, &mut *y, &plane_i, &z_i)?;
        }
        return Ok(g.out_shape(x_shape.len()));
    }
    let (func, cfg) = igemm_launch(dev, &plan, &g, n)?;
    let i = |v: usize| v as i32;
    let (b, d, h, wd, c, od, oh, ow, nn) = (
        i(g.b),
        i(g.d),
        i(g.h),
        i(g.w),
        i(g.c),
        i(g.od),
        i(g.oh),
        i(g.ow),
        i(n),
    );
    let (kd, kh, kw, sd, sh, sw, pd, ph, pw) = (
        i(g.kd),
        i(g.kh),
        i(g.kw),
        i(g.sd),
        i(g.sh),
        i(g.sw),
        i(g.pd),
        i(g.ph),
        i(g.pw),
    );
    let (one, splits_i) = (1i32, plan.splits as i32);
    if plan.splits == 1 {
        let mut yf = unsafe { dev.stream.alloc::<f32>(1) }?;
        let out16 = 1i32;
        super::kernels::launch!(dev.stream, func, cfg;
            x, w16, &mut yf, &mut *y, &out16, &b, &d, &h, &wd, &c, &od, &oh, &ow, &nn,
            &kd, &kh, &kw, &sd, &sh, &sw, &pd, &ph, &pw, &one, &one, &one, &splits_i)?;
    } else {
        let plane = m * n;
        let mut yf = unsafe { dev.stream.alloc::<f32>(plane * plan.splits) }?;
        let out16 = 0i32;
        super::kernels::launch!(dev.stream, func, cfg;
            x, w16, &mut yf, &mut *y, &out16, &b, &d, &h, &wd, &c, &od, &oh, &ow, &nn,
            &kd, &kh, &kw, &sd, &sh, &sw, &pd, &ph, &pw, &one, &one, &one, &splits_i)?;
        let (plane_i, z_i) = (plane as i64, plan.splits as i64);
        super::kernels::launch!(dev.stream, &dev.kernels.conv_sum_planes_bf16, super::kernels::cfg_n(plane);
            &yf, &mut *y, &plane_i, &z_i)?;
    }
    let _ = x.device_ptr(&dev.stream);
    Ok(g.out_shape(x_shape.len()))
}

/// Route R1: explicit im2col (`conv_unfold_nhwc_bf16`) in chunks of output
/// pixels bounded by `budget_bytes` of unfolded bf16 rows, each multiplied by
/// the `[K, N]` weight with cuBLASLt (f32 out, written in place into the
/// N(D)HWC result). Same contract as [`cudnn_conv`].
pub fn conv3d_unfold_lt(
    x: &CudaSlice<f32>,
    x_shape: &[usize],
    w: &CudaSlice<f32>,
    w_shape: &[usize],
    pad: &[usize],
    stride: &[usize],
    budget_bytes: usize,
) -> Result<(CudaSlice<f32>, Vec<usize>)> {
    let dev = global_device()
        .ok_or_else(|| DeviceError::Message("no global CUDA device context".into()))?;
    let g = ConvGeom::new(x_shape, w_shape, pad, stride)?;
    let x16 = route_x_nhwc(&dev, x, &g)?;
    let w16 = route_weight(&dev, w, &g, true)?;
    let (m, n, k) = (g.m(), g.n, g.k());
    let mc = (budget_bytes / (k * 2)).clamp(1, m.max(1));
    let mut y = unsafe { dev.stream.alloc::<f32>((m * n).max(1)) }?;
    let mut a16 = unsafe { dev.stream.alloc::<half::bf16>((mc * k).max(1)) }?;
    let i = |v: usize| v as i32;
    let (d, h, wd, c, od, oh, ow) = (i(g.d), i(g.h), i(g.w), i(g.c), i(g.od), i(g.oh), i(g.ow));
    let (kd, kh, kw, sd, sh, sw, pd, ph, pw) = (
        i(g.kd),
        i(g.kh),
        i(g.kw),
        i(g.sd),
        i(g.sh),
        i(g.sw),
        i(g.pd),
        i(g.ph),
        i(g.pw),
    );
    let one = 1i32;
    let mut m0 = 0usize;
    while m0 < m {
        let mcur = mc.min(m - m0);
        // The GEMM wants `a.len() == mcur * k`: the tail chunk gets its own buffer.
        let mut tail;
        let a = if mcur == mc {
            &mut a16
        } else {
            tail = unsafe { dev.stream.alloc::<half::bf16>(mcur * k) }?;
            &mut tail
        };
        let (m0_i, mc_i) = (m0 as i64, mcur as i64);
        super::kernels::launch!(dev.stream, &dev.kernels.conv_unfold_nhwc_bf16, super::kernels::cfg_n(mcur * k);
            &x16, &mut *a, &m0_i, &mc_i, &d, &h, &wd, &c, &od, &oh, &ow,
            &kd, &kh, &kw, &sd, &sh, &sw, &pd, &ph, &pw, &one, &one, &one)?;
        let mut view = y.slice_mut(m0 * n..(m0 + mcur) * n);
        super::device::matmul_2d_strided_batched_out_view_bf16(
            &*a,
            &w16,
            &mut view,
            mcur * n,
            1,
            mcur,
            k,
            n,
        )?;
        m0 += mcur;
    }
    drop(a16);
    drop(x16);
    drop(w16);
    let out = route_y_nchw(&dev, &y, &g, 1)?;
    Ok((out, g.out_shape(x_shape.len())))
}

/// The unfold route's default chunk budget (bytes of bf16 im2col rows).
pub const CONV_UNFOLD_BUDGET_BYTES: usize = 512 << 20;
