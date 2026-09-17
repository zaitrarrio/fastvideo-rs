//! Device convolutions: cuDNN N-D forward with cached plans, and 1×1 as GEMM.
//!
//! The Wan VAE decoder runs a handful of distinct conv shapes thousands of
//! times per clip, so descriptors, the heuristic algorithm choice and the
//! workspace size are built once per shape and the workspace buffer is shared
//! and only ever grows. Bias is added in place by one kernel launch.
//!
//! `FASTVIDEO_CONV3D=unfold` routes 3-D convs through a temporal unfold into a
//! cuDNN conv2d instead (lets cuDNN pick 2-D Winograd for 3×3); `fv-gpucheck`
//! benchmarks both.

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

#[derive(Default)]
pub struct ConvCache {
    plans: HashMap<ConvKey, ConvPlan>,
    workspace: Option<CudaSlice<u8>>,
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
    let ConvCache { plans, workspace } = &mut *cache;
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

/// `cudnn` (default) or `unfold`.
pub fn conv3d_backend() -> String {
    CONV3D_BACKEND.get_or_init(|| super::envflag::string_flag("FASTVIDEO_CONV3D", "cudnn"))
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
