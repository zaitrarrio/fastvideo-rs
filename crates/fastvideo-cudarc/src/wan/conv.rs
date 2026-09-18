//! Device convolutions: cuDNN N-D forward with cached plans, and 1×1 as GEMM.
//!
//! The Wan VAE decoder runs a handful of distinct conv shapes thousands of
//! times per clip, so descriptors, the heuristic algorithm choice and the
//! workspace size are built once per shape and the workspace buffer is shared
//! and only ever grows. Bias is added in place by one kernel launch.
//!
//! 3-D convs have two backends: cuDNN N-D, or a temporal unfold into a cuDNN
//! conv2d (lets cuDNN pick 2-D Winograd for 3×3). Which is faster depends on
//! the GPU (unfold 2× faster on an RTX A5000, cuDNN N-D 1.6× faster on an RTX
//! 5060 Ti at VAE sizes), so by default the first call per shape times both
//! and caches the winner. `FASTVIDEO_CONV3D=cudnn|unfold` forces one.

#![cfg(feature = "cuda")]

use std::collections::HashMap;

use cudarc::cudnn::{sys, ConvDescriptor, ConvForward, FilterDescriptor, TensorDescriptor};
use cudarc::driver::CudaSlice;

use super::device::{global_device, DeviceContext, DeviceError, Result};
use super::envflag::CachedString;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct ConvKey {
    x: Vec<usize>,
    w: Vec<usize>,
    pad: Vec<usize>,
    stride: Vec<usize>,
    fma: bool,
    bf16: bool,
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
}

#[derive(Default)]
pub struct ConvCache {
    plans: HashMap<ConvKey, ConvPlan>,
    plans_bf16: HashMap<ConvKey, ConvPlanBf16>,
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

fn out_shape(x: &[usize], w: &[usize], pad: &[usize], stride: &[usize]) -> Result<Vec<usize>> {
    let spatial = x.len() - 2;
    if w.len() != x.len() || pad.len() != spatial || stride.len() != spatial || w[1] != x[1] {
        return Err(DeviceError::Message(format!(
            "conv shape mismatch: x={x:?} w={w:?} pad={pad:?} stride={stride:?}"
        )));
    }
    let mut out = vec![x[0], w[0]];
    for i in 0..spatial {
        let padded = x[2 + i] + 2 * pad[i];
        if padded < w[2 + i] || stride[i] == 0 {
            return Err(DeviceError::Message(format!(
                "conv kernel larger than input: x={x:?} w={w:?} pad={pad:?}"
            )));
        }
        out.push((padded - w[2 + i]) / stride[i] + 1);
    }
    Ok(out)
}

/// TF32 is off for exact runs (`FASTVIDEO_TF32=0`): cuDNN's default math type
/// allows TF32 kernels on Ampere and newer, FMA math forbids them.
fn fma_math(dev: &DeviceContext) -> bool {
    dev.gemm_math == super::device::GemmMath::F32
}

fn build_plan(dev: &DeviceContext, key: &ConvKey) -> Result<ConvPlan> {
    let cudnn = &dev.cudnn;
    let y_shape = out_shape(&key.x, &key.w, &key.pad, &key.stride)?;
    // cuDNN wants at least 2 spatial dims; conv1d is not used by this crate.
    let pads: Vec<i32> = key.pad.iter().map(|&p| p as i32).collect();
    let strides: Vec<i32> = key.stride.iter().map(|&s| s as i32).collect();
    let dilations = vec![1i32; pads.len()];
    let mut conv = cudnn.create_convnd::<f32>(
        &pads,
        &strides,
        &dilations,
        sys::cudnnConvolutionMode_t::CUDNN_CROSS_CORRELATION,
    )?;
    conv.set_math_type(if key.fma {
        sys::cudnnMathType_t::CUDNN_FMA_MATH
    } else {
        sys::cudnnMathType_t::CUDNN_DEFAULT_MATH
    })?;
    let dims = |s: &[usize]| s.iter().map(|&d| d as i32).collect::<Vec<i32>>();
    let x = cudnn.create_nd_tensor::<f32>(&dims(&key.x), &contiguous_strides(&key.x))?;
    let w = cudnn.create_nd_filter::<f32>(sys::cudnnTensorFormat_t::CUDNN_TENSOR_NCHW, &dims(&key.w))?;
    let y = cudnn.create_nd_tensor::<f32>(&dims(&y_shape), &contiguous_strides(&y_shape))?;
    let op = ConvForward { conv: &conv, x: &x, w: &w, y: &y };
    let algo = op.pick_algorithm()?;
    let workspace_bytes = op.get_workspace_size(algo)?;
    super::log::debug(format_args!(
        "conv plan x={:?} w={:?} pad={:?} stride={:?} algo={algo:?} workspace={}B",
        key.x, key.w, key.pad, key.stride, workspace_bytes
    ));
    Ok(ConvPlan { conv, x, w, y, algo, workspace_bytes, y_shape })
}

fn build_plan_bf16(dev: &DeviceContext, key: &ConvKey) -> Result<ConvPlanBf16> {
    let cudnn = &dev.cudnn;
    let y_shape = out_shape(&key.x, &key.w, &key.pad, &key.stride)?;
    let pads: Vec<i32> = key.pad.iter().map(|&p| p as i32).collect();
    let strides: Vec<i32> = key.stride.iter().map(|&s| s as i32).collect();
    let dilations = vec![1i32; pads.len()];
    // F32 accumulation over bf16 operands; tensor-core math is the whole point,
    // so this plan never asks for FMA_MATH.
    let mut conv = cudnn.create_convnd::<f32>(
        &pads,
        &strides,
        &dilations,
        sys::cudnnConvolutionMode_t::CUDNN_CROSS_CORRELATION,
    )?;
    conv.set_math_type(sys::cudnnMathType_t::CUDNN_TENSOR_OP_MATH)?;
    let dims = |s: &[usize]| s.iter().map(|&d| d as i32).collect::<Vec<i32>>();
    let x = cudnn.create_nd_tensor::<half::bf16>(&dims(&key.x), &contiguous_strides(&key.x))?;
    let w = cudnn
        .create_nd_filter::<half::bf16>(sys::cudnnTensorFormat_t::CUDNN_TENSOR_NCHW, &dims(&key.w))?;
    let y = cudnn.create_nd_tensor::<half::bf16>(&dims(&y_shape), &contiguous_strides(&y_shape))?;
    let op = ConvForward { conv: &conv, x: &x, w: &w, y: &y };
    let algo = op.pick_algorithm()?;
    let workspace_bytes = op.get_workspace_size(algo)?;
    super::log::debug(format_args!(
        "conv plan (bf16) x={:?} w={:?} algo={algo:?} workspace={}B",
        key.x, key.w, workspace_bytes
    ));
    Ok(ConvPlanBf16 { conv, x, w, y, algo, workspace_bytes, y_shape })
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
    let dev = global_device().ok_or_else(|| DeviceError::Message("no global CUDA device context".into()))?;
    let cast = |e: super::tensor::TensorError| DeviceError::Message(e.to_string());
    let xb = super::ops::cast_f32_bf16_device(x).map_err(cast)?;
    let wb = super::ops::cast_f32_bf16_device(w).map_err(cast)?;
    let key = ConvKey {
        x: x_shape.to_vec(),
        w: w_shape.to_vec(),
        pad: pad.to_vec(),
        stride: stride.to_vec(),
        fma: false,
        bf16: true,
    };
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
    let ConvCache { plans_bf16, workspace, .. } = &mut *cache;
    let plan = &plans_bf16[&key];
    let n_out: usize = plan.y_shape.iter().product();
    let mut yb = unsafe { dev.stream.alloc::<half::bf16>(n_out) }?;
    let op = ConvForward { conv: &plan.conv, x: &plan.x, w: &plan.w, y: &plan.y };
    unsafe {
        // cudarc types alpha/beta as the output element type and converts them
        // to cuDNN's F32 scaling parameter internally.
        let (alpha, beta) = (half::bf16::from_f32(1.0), half::bf16::from_f32(0.0));
        op.launch(plan.algo, if need > 0 { workspace.as_mut() } else { None }, (alpha, beta), &xb, &wb, &mut yb)?;
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
    let dev = global_device().ok_or_else(|| DeviceError::Message("no global CUDA device context".into()))?;
    if x.len() != x_shape.iter().product::<usize>() || w.len() != w_shape.iter().product::<usize>() {
        return Err(DeviceError::Message(format!(
            "conv buffer mismatch: x.len={} shape={x_shape:?} w.len={} shape={w_shape:?}",
            x.len(),
            w.len()
        )));
    }
    let key = ConvKey {
        x: x_shape.to_vec(),
        w: w_shape.to_vec(),
        pad: pad.to_vec(),
        stride: stride.to_vec(),
        fma: fma_math(&dev),
        bf16: false,
    };
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
    let ConvCache { plans, workspace, .. } = &mut *cache;
    let plan = &plans[&key];
    let n_out: usize = plan.y_shape.iter().product();
    let mut y = unsafe { dev.stream.alloc::<f32>(n_out) }?;
    let op = ConvForward { conv: &plan.conv, x: &plan.x, w: &plan.w, y: &plan.y };
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

static CONV3D_BACKEND: CachedString = CachedString::new();

/// `auto` (default), `cudnn` or `unfold`.
pub fn conv3d_backend() -> String {
    CONV3D_BACKEND.get_or_init(|| super::envflag::string_flag("FASTVIDEO_CONV3D", "auto"))
}

/// 3-D conv with the backend chosen per [`conv3d_backend`]. In `auto` mode the
/// first call for a shape runs each backend twice, times the second run of
/// each (synchronized), keeps the faster one's output and remembers the choice.
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
            Conv3dPick::Unfold if pad[0] == 0 => conv3d_unfold(x, x_shape, w, w_shape, [pad[1], pad[2]], stride),
            Conv3dPick::CudnnBf16 => cudnn_conv_bf16(x, x_shape, w, w_shape, &pad, &stride),
            _ => cudnn_conv(x, x_shape, w, w_shape, &pad, &stride),
        }
    };
    match conv3d_backend().as_str() {
        "cudnn" => return run(Conv3dPick::Cudnn),
        "unfold" => return run(Conv3dPick::Unfold),
        "cudnn-bf16" => return run(Conv3dPick::CudnnBf16),
        _ => {}
    }
    let dev = global_device().ok_or_else(|| DeviceError::Message("no global CUDA device context".into()))?;
    let key =
        ConvKey { x: x_shape.to_vec(), w: w_shape.to_vec(), pad: pad.to_vec(), stride: stride.to_vec(), fma: fma_math(&dev), bf16: false };
    let known = dev.conv.lock().expect("conv cache lock").conv3d_pick.get(&key).copied();
    if let Some(pick) = known {
        return run(pick);
    }
    let timed = |pick: Conv3dPick| -> Result<(f64, (CudaSlice<f32>, Vec<usize>))> {
        drop(run(pick)?);
        dev.synchronize()?;
        let t = std::time::Instant::now();
        let out = run(pick)?;
        dev.synchronize()?;
        Ok((t.elapsed().as_secs_f64(), out))
    };
    // Exact mode must stay comparable to the CPU path, so bf16 only competes
    // when the context is already running reduced-precision math.
    let mut best: Option<(f64, Conv3dPick, (CudaSlice<f32>, Vec<usize>))> = None;
    let mut report: Vec<String> = Vec::new();
    let candidates: &[Conv3dPick] = if fma_math(&dev) {
        &[Conv3dPick::Cudnn, Conv3dPick::Unfold]
    } else {
        &[Conv3dPick::Cudnn, Conv3dPick::Unfold, Conv3dPick::CudnnBf16]
    };
    for &pick in candidates {
        let (secs, out) = timed(pick)?;
        report.push(format!("{pick:?} {:.1}ms", secs * 1e3));
        if best.as_ref().is_none_or(|(b, _, _)| secs < *b) {
            best = Some((secs, pick, out));
        }
    }
    let (_, pick, out) = best.expect("at least one conv3d backend");
    super::log::info(format_args!("conv3d x={x_shape:?} w={w_shape:?}: {} → {pick:?}", report.join(", ")));
    dev.conv.lock().expect("conv cache lock").conv3d_pick.insert(key, pick);
    Ok(out)
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
    let dev = global_device().ok_or_else(|| DeviceError::Message("no global CUDA device context".into()))?;
    let (n, c, t, h, wd) = (x_shape[0], x_shape[1], x_shape[2], x_shape[3], x_shape[4]);
    let (oc, kt) = (w_shape[0], w_shape[2]);
    let st = stride[0].max(1);
    if t < kt {
        return Err(DeviceError::Message(format!("conv3d unfold: t={t} < kt={kt}")));
    }
    let ot = (t - kt) / st + 1;
    let unfolded_shape = [n * ot, c * kt, h, wd];
    let total: usize = unfolded_shape.iter().product();
    let mut unfolded = unsafe { dev.stream.alloc::<f32>(total) }?;
    let (c_i, t_i, h_i, w_i, kt_i, st_i, ot_i) =
        (c as i64, t as i64, h as i64, wd as i64, kt as i64, st as i64, ot as i64);
    let total_i = total as i64;
    super::kernels::launch!(dev.stream, &dev.kernels.temporal_unfold, super::kernels::cfg_n(total);
        x, &mut unfolded, &total_i, &c_i, &t_i, &h_i, &w_i, &kt_i, &st_i, &ot_i)?;
    let w2 = [oc, c * kt, w_shape[3], w_shape[4]];
    let (y2, y2_shape) = cudnn_conv(&unfolded, &unfolded_shape, w, &w2, &pad_hw, &[stride[1], stride[2]])?;
    drop(unfolded);
    // [n*ot, oc, oh, ow] → [n, oc, ot, oh, ow].
    let (oh, ow) = (y2_shape[2], y2_shape[3]);
    let out_shape = vec![n, oc, ot, oh, ow];
    let perm = super::ops::gather_nd_device(&y2, &[n, ot, oc, oh, ow], &[0, 2, 1, 3, 4])
        .map_err(|e| DeviceError::Message(e.to_string()))?;
    Ok((perm, out_shape))
}
