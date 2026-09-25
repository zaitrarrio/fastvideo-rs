//! The shared cuBLASLt context and the FP8 eligibility check.
//!
//! FP8 matmul exists only in cuBLASLt (not `cublasGemmEx`), TN only, with
//! 16-byte-aligned leading dimensions. The GEMMs themselves — the reference
//! W8A8 tensorwise and MXFP8 block-scaled recipes, bf16 output — live in
//! [`super::quant`]; the old per-tensor E4M3 path with f32 activations and
//! output (17-20 dB against bf16, matching no reference) is retired.

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
    pub(crate) handle: lt::cublasLtHandle_t,
    pub(crate) workspace: cudarc::driver::CudaSlice<u8>,
    pub(crate) workspace_bytes: usize,
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
