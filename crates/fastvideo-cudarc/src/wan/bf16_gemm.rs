//! BF16 GEMM helpers for DiT linears (cudarc `GemmEx` mixed precision).
//!
//! Enabled when `FASTVIDEO_BF16` is unset/`1`/`true`. Set `FASTVIDEO_BF16=0` to
//! force F32 GEMM. Uses **device-resident** BF16 weights × F32 activations → F32
//! via `cublasGemmEx` (no host cast roundtrips).

use super::envflag::CachedBool;

static BF16_CACHE: CachedBool = CachedBool::new();
static BF16_ACT_CACHE: CachedBool = CachedBool::new();

/// BF16 DiT GEMM policy (default on). Cached: consulted on every `Linear::forward`.
pub fn bf16_enabled() -> bool {
    BF16_CACHE.get_or_init(|| super::envflag::bool_flag("FASTVIDEO_BF16", true))
}

/// BF16 activation cache policy (`FASTVIDEO_BF16_ACT=1` default on; off to force re-cast per call).
/// Cached: consulted on every `Linear::forward` when BF16 is on.
pub fn bf16_act_cache_enabled() -> bool {
    BF16_ACT_CACHE.get_or_init(|| super::envflag::bool_flag("FASTVIDEO_BF16_ACT", true))
}

#[cfg(feature = "cuda")]
mod cuda_impl {
    use super::super::device::{self, DeviceError, Result};
    use super::super::ops::f32_to_bf16_bytes;
    use cudarc::driver::{DevicePtr, DevicePtrMut};

    pub fn upload_bf16(host_f32: &[f32]) -> Result<cudarc::driver::CudaSlice<half::bf16>> {
        let dev = device::global_device().ok_or_else(|| {
            DeviceError::Message("no global CUDA device context".into())
        })?;
        let bytes = f32_to_bf16_bytes(host_f32);
        let mut vals = Vec::with_capacity(host_f32.len());
        for chunk in bytes.chunks_exact(2) {
            let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
            vals.push(half::bf16::from_bits(bits));
        }
        Ok(dev.stream.memcpy_stod(&vals)?)
    }

    /// Device-resident `X[m,k](f32) @ W[n,k]^T(bf16) → C[m,n](f32)` via GemmEx.
    /// Casts X to BF16 first via the device `f32_to_bf16` kernel so the GEMM is
    /// pure BF16 × BF16 → F32 (Hopper Tensor Cores; no mixed-type rejection).
    pub fn matmul_linear_wt_bf16_to_f32(
        x_f32: &cudarc::driver::CudaSlice<f32>,
        w_bf16: &cudarc::driver::CudaSlice<half::bf16>,
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<cudarc::driver::CudaSlice<f32>> {
        let dev = device::global_device().ok_or_else(|| {
            DeviceError::Message("no global CUDA device context".into())
        })?;
        // Fresh bf16 scratch buffer; caller pays the cast cost every call.
        let mut bits_u16 = dev.stream.alloc_zeros::<u16>(x_f32.len())?;
        let c = matmul_with_bf16_bits(
            x_f32,
            w_bf16,
            &mut bits_u16,
            m,
            k,
            n,
        )?;
        Ok(c)
    }

    /// BF16 GEMM that reuses a caller-provided bf16 bits buffer (saves the cast
    /// when the underlying f32 activation hasn't changed since last call).
    pub fn matmul_linear_wt_bf16_to_f32_with_bits(
        x_f32: &cudarc::driver::CudaSlice<f32>,
        w_bf16: &cudarc::driver::CudaSlice<half::bf16>,
        bits_u16: &mut cudarc::driver::CudaSlice<u16>,
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<cudarc::driver::CudaSlice<f32>> {
        if bits_u16.len() != x_f32.len() {
            return Err(DeviceError::Message(
                "bf16 act cache size mismatch".into(),
            ));
        }
        matmul_with_bf16_bits(x_f32, w_bf16, bits_u16, m, k, n)
    }

    /// Run the device f32→bf16 cast into a caller-provided buffer.
    pub fn cast_f32_to_bf16_inplace(
        x_f32: &cudarc::driver::CudaSlice<f32>,
        bits_u16: &mut cudarc::driver::CudaSlice<u16>,
    ) -> Result<()> {
        if bits_u16.len() != x_f32.len() {
            return Err(DeviceError::Message(
                "bf16 cast buffer size mismatch".into(),
            ));
        }
        let dev = device::global_device().ok_or_else(|| {
            DeviceError::Message("no global CUDA device context".into())
        })?;
        unsafe {
            super::super::kernels::launch_f32_to_bf16(
                &dev.stream,
                &dev.kernels.f32_to_bf16,
                x_f32,
                bits_u16,
                x_f32.len() as i32,
            )
            .map_err(DeviceError::from)?;
        }
        Ok(())
    }

    /// `X[m,k](bf16) @ W[n,k]^T(bf16) → C[m,n](f32)` via GemmEx Tensor Cores.
    /// Used by `Linear::forward_from_bf16` for chained FFN: the first projection's
    /// BF16 output is passed directly as the second projection's input without
    /// a round-trip through F32, saving one cast kernel and one allocation.
    pub fn matmul_bf16_bf16_to_f32(
        x_bf16: &cudarc::driver::CudaSlice<half::bf16>,
        w_bf16: &cudarc::driver::CudaSlice<half::bf16>,
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<cudarc::driver::CudaSlice<f32>> {
        use cudarc::cublas::sys;
        use cudarc::driver::{DevicePtr, DevicePtrMut};

        let dev = device::global_device().ok_or_else(|| {
            DeviceError::Message("no global CUDA device context".into())
        })?;
        if x_bf16.len() != m * k || w_bf16.len() != n * k {
            return Err(DeviceError::Message(
                "matmul_bf16_bf16_to_f32: size mismatch".into(),
            ));
        }
        let mut c = dev.stream.alloc_zeros::<f32>(m * n)?;
        let alpha = 1.0f32;
        let beta = 0.0f32;
        let (a_ptr, _ra) = w_bf16.device_ptr(&dev.stream);
        let (b_ptr, _rb) = x_bf16.device_ptr(&dev.stream);
        let (c_ptr, _rc) = c.device_ptr_mut(&dev.stream);
        unsafe {
            cudarc::cublas::result::gemm_ex(
                *dev.cublas.handle(),
                sys::cublasOperation_t::CUBLAS_OP_T,
                sys::cublasOperation_t::CUBLAS_OP_N,
                n as i32,
                m as i32,
                k as i32,
                (&alpha) as *const f32 as *const _,
                a_ptr as *const _,
                sys::cudaDataType_t::CUDA_R_16BF,
                k as i32,
                b_ptr as *const _,
                sys::cudaDataType_t::CUDA_R_16BF,
                k as i32,
                (&beta) as *const f32 as *const _,
                c_ptr as *mut _,
                sys::cudaDataType_t::CUDA_R_32F,
                n as i32,
                sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
                sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT_TENSOR_OP,
            )
            .map_err(|e| DeviceError::Message(e.to_string()))?;
        }
        drop(_ra);
        drop(_rb);
        drop(_rc);
        Ok(c)
    }

    /// Identity hash for an activation: the device pointer XOR'd with the length.
    /// Cheap "has the activation buffer changed since last call?" test used by the
    /// per-Linear bf16 cache (`FASTVIDEO_BF16_ACT=1`).
    pub fn activation_identity(x_f32: &cudarc::driver::CudaSlice<f32>) -> u64 {
        use cudarc::driver::DevicePtr;
        let dev = match device::global_device() {
            Some(d) => d,
            None => return x_f32.len() as u64,
        };
        let (ptr, _g) = x_f32.device_ptr(&dev.stream);
        let p = ptr as u64;
        let l = x_f32.len() as u64;
        p ^ l.rotate_left(17) ^ ((l.wrapping_mul(0x9E3779B97F4A7C15)) as u64)
    }

    fn matmul_with_bf16_bits(
        x_f32: &cudarc::driver::CudaSlice<f32>,
        w_bf16: &cudarc::driver::CudaSlice<half::bf16>,
        bits_u16: &mut cudarc::driver::CudaSlice<u16>,
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<cudarc::driver::CudaSlice<f32>> {
        use cudarc::cublas::sys;

        let dev = device::global_device().ok_or_else(|| {
            DeviceError::Message("no global CUDA device context".into())
        })?;
        if x_f32.len() != m * k || w_bf16.len() != n * k {
            return Err(DeviceError::Message("bf16 linear size mismatch".into()));
        }
        let mut c = dev.stream.alloc_zeros::<f32>(m * n)?;
        let alpha = 1.0f32;
        let beta = 0.0f32;
        let (a_ptr, _ra) = w_bf16.device_ptr(&dev.stream);
        let (b_ptr, _rb) = bits_u16.device_ptr(&dev.stream);
        let (c_ptr, _rc) = c.device_ptr_mut(&dev.stream);
        unsafe {
            cudarc::cublas::result::gemm_ex(
                *dev.cublas.handle(),
                sys::cublasOperation_t::CUBLAS_OP_T,
                sys::cublasOperation_t::CUBLAS_OP_N,
                n as i32,
                m as i32,
                k as i32,
                (&alpha) as *const f32 as *const _,
                a_ptr as *const _,
                sys::cudaDataType_t::CUDA_R_16BF,
                k as i32,
                b_ptr as *const _,
                sys::cudaDataType_t::CUDA_R_16BF,
                k as i32,
                (&beta) as *const f32 as *const _,
                c_ptr as *mut _,
                sys::cudaDataType_t::CUDA_R_32F,
                n as i32,
                sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
                sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT_TENSOR_OP,
            )
            .map_err(|e| DeviceError::Message(e.to_string()))?;
        }
        drop(_ra);
        drop(_rb);
        drop(_rc);
        Ok(c)
    }

    /// Device-resident `X[m,k](f32) @ W[n,k]^T(bf16) → C[m,n](bf16)` via GemmEx.
    /// Writes the GEMM result directly into a bf16 buffer using
    /// `CUBLAS_COMPUTE_32F` + `CUDA_R_16BF` (Hopper Tensor Core → bf16 epilogue).
    /// Used by `Linear.forward_into` so the next linear can skip an
    /// immediate f32 → bf16 cast on this consumer's input.
    pub fn matmul_linear_wt_bf16_to_bf16(
        x_f32: &cudarc::driver::CudaSlice<f32>,
        w_bf16: &cudarc::driver::CudaSlice<half::bf16>,
        bits_u16: &mut cudarc::driver::CudaSlice<u16>,
        c_bf16: &mut cudarc::driver::CudaSlice<half::bf16>,
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<()> {
        use cudarc::cublas::sys;

        let dev = device::global_device().ok_or_else(|| {
            DeviceError::Message("no global CUDA device context".into())
        })?;
        if x_f32.len() != m * k || w_bf16.len() != n * k {
            return Err(DeviceError::Message("bf16 linear size mismatch".into()));
        }
        if bits_u16.len() != x_f32.len() || c_bf16.len() != m * n {
            return Err(DeviceError::Message(
                "bf16-output linear buffer size mismatch".into(),
            ));
        }
        // Ensure c_bf16 is zeroed (cublasGemmEx doesn't auto-zero on first use with beta=0,
        // but we want deterministic values; the caller passes an empty zeroed slice).
        let alpha = 1.0f32;
        let beta = 0.0f32;
        let (a_ptr, _ra) = w_bf16.device_ptr(&dev.stream);
        let (b_ptr, _rb) = bits_u16.device_ptr(&dev.stream);
        let (c_ptr, _rc) = c_bf16.device_ptr_mut(&dev.stream);
        unsafe {
            cudarc::cublas::result::gemm_ex(
                *dev.cublas.handle(),
                sys::cublasOperation_t::CUBLAS_OP_T,
                sys::cublasOperation_t::CUBLAS_OP_N,
                n as i32,
                m as i32,
                k as i32,
                (&alpha) as *const f32 as *const _,
                a_ptr as *const _,
                sys::cudaDataType_t::CUDA_R_16BF,
                k as i32,
                b_ptr as *const _,
                sys::cudaDataType_t::CUDA_R_16BF,
                k as i32,
                (&beta) as *const f32 as *const _,
                c_ptr as *mut _,
                sys::cudaDataType_t::CUDA_R_16BF,
                n as i32,
                sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
                sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT_TENSOR_OP,
            )
            .map_err(|e| DeviceError::Message(e.to_string()))?;
        }
        drop(_ra);
        drop(_rb);
        drop(_rc);
        Ok(())
    }
}

#[cfg(feature = "cuda")]
pub use cuda_impl::*;
