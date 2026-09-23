//! FP8 E4M3 GEMM through cuBLASLt.
//!
//! FP8 matmul is not reachable from `cublasGemmEx`; it exists only in cuBLASLt,
//! and cudarc's safe `Matmul<T>` wrapper is homogeneous (one dtype for A, B and
//! C) with no FP8 support at all, so this drives the raw API.
//!
//! Three constraints shape everything below, and all three come from cuBLAS
//! rather than from us:
//!
//! 1. **TN only.** FP8 matmul requires `opA = T`, `opB = N` in cuBLAS's
//!    column-major terms. There is no NN FP8 kernel to fall back to, so a
//!    caller whose data is not already in that layout cannot use this path.
//! 2. **Per-tensor scales.** Through cuBLAS 12.8 the scale pointers are single
//!    device scalars; per-channel (vector) scaling needs 12.9 and the
//!    `*_SCALE_MODE` attributes. Our runtime image ships 12.4, so this is
//!    per-tensor. That is coarse for a 1536x8960 weight — but tolerating
//!    exactly this is what a QAD checkpoint was trained for, which is the whole
//!    reason this path is worth trying on those weights and nowhere else.
//! 3. **Alignment.** Leading dimensions want 16-byte alignment; E4M3 is one
//!    byte per element, so every leading dimension must be a multiple of 16.
//!    Wan's are (1536, 4096, 8960 and a token count that is a multiple of 16),
//!    but `fp8_gemm_supported` checks rather than assumes.

use cudarc::cublaslt::sys as lt;

use super::device::DeviceContext;
use super::tensor::{Result, TensorError};

/// E4M3 needs Ada (sm89) or newer; Ampere has no FP8 tensor cores.
pub const MIN_SM: (i32, i32) = (8, 9);

/// Whether this device and shape can use the FP8 path at all.
///
/// Returns a reason rather than a bare `false`: a silent fallback to bf16 would
/// make an FP8 benchmark quietly measure bf16, which is the exact failure mode
/// that let cuBLAS 12.4 report fake bf16 numbers on Blackwell.
pub fn fp8_gemm_supported(
    dev: &DeviceContext,
    m: usize,
    n: usize,
    k: usize,
) -> std::result::Result<(), String> {
    if (dev.sm_major, dev.sm_minor) < MIN_SM {
        return Err(format!(
            "FP8 needs sm{}{} or newer, this device is sm{}{}",
            MIN_SM.0, MIN_SM.1, dev.sm_major, dev.sm_minor
        ));
    }
    for (name, v) in [("m", m), ("n", n), ("k", k)] {
        if v % 16 != 0 {
            return Err(format!("FP8 needs {name}={v} to be a multiple of 16"));
        }
    }
    Ok(())
}

/// Owns the cuBLASLt handle and its workspace for one device.
pub struct LtContext {
    handle: lt::cublasLtHandle_t,
    workspace: cudarc::driver::CudaSlice<u8>,
    workspace_bytes: usize,
}

// Same reasoning as `DeviceContext`: a raw handle used from one thread at a time.
unsafe impl Send for LtContext {}
unsafe impl Sync for LtContext {}

impl LtContext {
    pub fn new(dev: &DeviceContext) -> Result<Self> {
        let mut handle: lt::cublasLtHandle_t = std::ptr::null_mut();
        let status = unsafe { lt::cublasLtCreate(&mut handle) };
        check(status, "cublasLtCreate")?;
        // 32 MiB is cuBLASLt's own recommendation for Hopper-class split-k.
        let workspace_bytes = 32 * 1024 * 1024;
        let workspace = dev
            .stream
            .alloc_zeros::<u8>(workspace_bytes)
            .map_err(|e| TensorError::Message(format!("cuBLASLt workspace alloc failed: {e}")))?;
        Ok(Self {
            handle,
            workspace,
            workspace_bytes,
        })
    }
}

impl Drop for LtContext {
    fn drop(&mut self) {
        unsafe { lt::cublasLtDestroy(self.handle) };
    }
}

fn check(status: lt::cublasStatus_t, what: &str) -> Result<()> {
    if status == lt::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
        Ok(())
    } else {
        Err(TensorError::Message(format!("{what} failed: {status:?}")))
    }
}

/// RAII for the descriptors, so an early return cannot leak them.
struct Desc(lt::cublasLtMatmulDesc_t);
impl Drop for Desc {
    fn drop(&mut self) {
        unsafe { lt::cublasLtMatmulDescDestroy(self.0) };
    }
}
struct Layout(lt::cublasLtMatrixLayout_t);
impl Drop for Layout {
    fn drop(&mut self) {
        unsafe { lt::cublasLtMatrixLayoutDestroy(self.0) };
    }
}
struct Pref(lt::cublasLtMatmulPreference_t);
impl Drop for Pref {
    fn drop(&mut self) {
        unsafe { lt::cublasLtMatmulPreferenceDestroy(self.0) };
    }
}

unsafe fn set_attr<T>(
    desc: lt::cublasLtMatmulDesc_t,
    attr: lt::cublasLtMatmulDescAttributes_t,
    v: &T,
) -> Result<()> {
    check(
        lt::cublasLtMatmulDescSetAttribute(
            desc,
            attr,
            (v as *const T).cast(),
            std::mem::size_of::<T>(),
        ),
        "cublasLtMatmulDescSetAttribute",
    )
}

/// `D = (a_scale * A^T) * (b_scale * B) * d_scale`, with A and B in E4M3 and D
/// in f32.
///
/// Shapes follow cuBLAS column-major convention: `A` is `k x m` with leading
/// dimension `k`, `B` is `k x n` with leading dimension `k`, `D` is `m x n`
/// with leading dimension `m`. For a row-major `[tokens, in] @ [out, in]^T`
/// linear this means passing the weight as A and the activations as B, which is
/// the layout both already have — the TN constraint costs us nothing here.
///
/// Scale arguments are *device* pointers to f32 scalars, so they can be
/// produced by `e4m3_scale_from_amax` without a host round trip.
///
/// # Safety
/// Pointers must be valid device allocations of the implied sizes, and the
/// element counts must satisfy `fp8_gemm_supported`.
#[allow(clippy::too_many_arguments)]
pub unsafe fn gemm_e4m3(
    dev: &DeviceContext,
    ltc: &LtContext,
    m: usize,
    n: usize,
    k: usize,
    a: cudarc::driver::sys::CUdeviceptr,
    a_scale: cudarc::driver::sys::CUdeviceptr,
    b: cudarc::driver::sys::CUdeviceptr,
    b_scale: cudarc::driver::sys::CUdeviceptr,
    d: cudarc::driver::sys::CUdeviceptr,
) -> Result<()> {
    let e4m3 = lt::cudaDataType_t::CUDA_R_8F_E4M3;
    let f32_ty = lt::cudaDataType_t::CUDA_R_32F;

    let mut desc: lt::cublasLtMatmulDesc_t = std::ptr::null_mut();
    check(
        lt::cublasLtMatmulDescCreate(
            &mut desc,
            lt::cublasComputeType_t::CUBLAS_COMPUTE_32F,
            f32_ty,
        ),
        "cublasLtMatmulDescCreate",
    )?;
    let desc = Desc(desc);

    // TN: the only layout FP8 matmul supports. The operation enum lives in the
    // cublas sys module, not cublaslt's, but the attribute takes the same i32.
    use cudarc::cublas::sys::cublasOperation_t;
    let op_t = cublasOperation_t::CUBLAS_OP_T as i32;
    let op_n = cublasOperation_t::CUBLAS_OP_N as i32;
    set_attr(
        desc.0,
        lt::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSA,
        &op_t,
    )?;
    set_attr(
        desc.0,
        lt::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSB,
        &op_n,
    )?;
    set_attr(
        desc.0,
        lt::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,
        &a_scale,
    )?;
    set_attr(
        desc.0,
        lt::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,
        &b_scale,
    )?;

    let mut la: lt::cublasLtMatrixLayout_t = std::ptr::null_mut();
    let mut lb: lt::cublasLtMatrixLayout_t = std::ptr::null_mut();
    let mut ld: lt::cublasLtMatrixLayout_t = std::ptr::null_mut();
    check(
        lt::cublasLtMatrixLayoutCreate(&mut la, e4m3, k as u64, m as u64, k as i64),
        "layout A",
    )?;
    let la = Layout(la);
    check(
        lt::cublasLtMatrixLayoutCreate(&mut lb, e4m3, k as u64, n as u64, k as i64),
        "layout B",
    )?;
    let lb = Layout(lb);
    check(
        lt::cublasLtMatrixLayoutCreate(&mut ld, f32_ty, m as u64, n as u64, m as i64),
        "layout D",
    )?;
    let ld = Layout(ld);

    let mut pref: lt::cublasLtMatmulPreference_t = std::ptr::null_mut();
    check(
        lt::cublasLtMatmulPreferenceCreate(&mut pref),
        "cublasLtMatmulPreferenceCreate",
    )?;
    let pref = Pref(pref);
    let ws = ltc.workspace_bytes;
    check(
        lt::cublasLtMatmulPreferenceSetAttribute(
            pref.0,
            lt::cublasLtMatmulPreferenceAttributes_t::CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
            (&ws as *const usize).cast(),
            std::mem::size_of::<usize>(),
        ),
        "preference workspace",
    )?;

    let mut heuristic = std::mem::MaybeUninit::<lt::cublasLtMatmulHeuristicResult_t>::uninit();
    let mut found: i32 = 0;
    check(
        lt::cublasLtMatmulAlgoGetHeuristic(
            ltc.handle,
            desc.0,
            la.0,
            lb.0,
            ld.0,
            ld.0,
            pref.0,
            1,
            heuristic.as_mut_ptr(),
            &mut found,
        ),
        "cublasLtMatmulAlgoGetHeuristic",
    )?;
    if found == 0 {
        // No FP8 kernel for this shape. Report it rather than silently running
        // something else: a quiet fallback would make an FP8 benchmark measure
        // whatever cuBLAS chose instead.
        return Err(TensorError::Message(format!(
            "cuBLASLt has no FP8 algorithm for m={m} n={n} k={k} on sm{}{}",
            dev.sm_major, dev.sm_minor
        )));
    }
    let heuristic = heuristic.assume_init();

    let alpha = 1.0f32;
    let beta = 0.0f32;
    let (ws_ptr, _ws_guard) = {
        use cudarc::driver::DevicePtr;
        ltc.workspace.device_ptr(&dev.stream)
    };
    check(
        lt::cublasLtMatmul(
            ltc.handle,
            desc.0,
            (&alpha as *const f32).cast(),
            a as *const _,
            la.0,
            b as *const _,
            lb.0,
            (&beta as *const f32).cast(),
            d as *const _,
            ld.0,
            d as *mut _,
            ld.0,
            &heuristic.algo,
            ws_ptr as *mut _,
            ltc.workspace_bytes,
            dev.stream.cu_stream() as *mut _,
        ),
        "cublasLtMatmul",
    )?;
    Ok(())
}

/// The process-wide cuBLASLt context.
///
/// The handle plus a 32 MiB workspace is far too expensive to build per GEMM,
/// and every linear on a device shares one, exactly as `DeviceContext` holds
/// one cuBLAS handle.
static LT: std::sync::Mutex<Option<std::sync::Arc<LtContext>>> = std::sync::Mutex::new(None);

pub fn lt_context(dev: &DeviceContext) -> Result<std::sync::Arc<LtContext>> {
    let mut guard = LT
        .lock()
        .map_err(|_| TensorError::Message("cuBLASLt context poisoned".into()))?;
    if let Some(c) = guard.as_ref() {
        return Ok(c.clone());
    }
    let ctx = std::sync::Arc::new(LtContext::new(dev)?);
    *guard = Some(ctx.clone());
    Ok(ctx)
}

/// A linear's weight, quantized once at load.
///
/// Quantization happens on the host with `fastvideo_ops::fp8`, the same
/// reference the kernels are checked against: weights are already in host
/// memory at load time, so there is nothing to gain from a device round trip
/// and something to lose — the host path is the exhaustively tested one.
#[derive(Debug)]
pub struct Fp8Weight {
    pub data: cudarc::driver::CudaSlice<u8>,
    /// One-element device buffer; the GEMM takes its address.
    pub scale: cudarc::driver::CudaSlice<f32>,
    pub out_dim: usize,
    pub in_dim: usize,
}

impl Fp8Weight {
    /// Quantize a row-major `[out, in]` f32 weight to per-tensor E4M3.
    pub fn quantize(dev: &DeviceContext, w: &[f32], out_dim: usize, in_dim: usize) -> Result<Self> {
        use fastvideo_ops::fp8;
        if w.len() != out_dim * in_dim {
            return Err(TensorError::Message(format!(
                "fp8 weight {} elements for [{out_dim}, {in_dim}]",
                w.len()
            )));
        }
        let amax = w.iter().fold(0.0f32, |a, &b| a.max(b.abs()));
        if !amax.is_finite() {
            return Err(TensorError::Message(
                "fp8 weight has a non-finite amax".into(),
            ));
        }
        let (scale_v, inv) = fp8::scale_for_amax(amax);
        let bytes: Vec<u8> = w.iter().map(|&v| fp8::f32_to_e4m3(v * inv)).collect();
        let data = dev
            .stream
            .memcpy_stod(&bytes)
            .map_err(|e| TensorError::Message(format!("fp8 weight upload: {e}")))?;
        let scale = dev
            .stream
            .memcpy_stod(&[scale_v])
            .map_err(|e| TensorError::Message(format!("fp8 scale upload: {e}")))?;
        Ok(Self {
            data,
            scale,
            out_dim,
            in_dim,
        })
    }
}
