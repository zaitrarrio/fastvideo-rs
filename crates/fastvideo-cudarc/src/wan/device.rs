//! CUDA device context (cudarc driver + cuBLAS + cuDNN + NVRTC kernels).

use thiserror::Error;

#[cfg(feature = "cuda")]
use std::sync::{Arc, Mutex, OnceLock};

#[derive(Debug, Error)]
pub enum DeviceError {
    #[error("{0}")]
    Message(String),
    #[cfg(feature = "cuda")]
    #[error(transparent)]
    Cuda(#[from] cudarc::driver::DriverError),
    #[cfg(feature = "cuda")]
    #[error(transparent)]
    Cublas(#[from] cudarc::cublas::result::CublasError),
    #[cfg(feature = "cuda")]
    #[error(transparent)]
    Cudnn(#[from] cudarc::cudnn::CudnnError),
}

pub type Result<T> = std::result::Result<T, DeviceError>;

#[cfg(feature = "cuda")]
pub struct DeviceContext {
    pub ctx: Arc<cudarc::driver::CudaContext>,
    pub stream: Arc<cudarc::driver::CudaStream>,
    pub cublas: cudarc::cublas::CudaBlas,
    pub cudnn: Arc<cudarc::cudnn::Cudnn>,
    pub kernels: super::kernels::KernelFns,
    pub sm_major: i32,
    pub sm_minor: i32,
    pub tf32: bool,
}

// cudarc's `Cudnn` is `!Send`/`!Sync` (raw handle). We only touch it through
// `GLOBAL_DEVICE`'s `Mutex`, so the process-wide context is safe to share.
#[cfg(feature = "cuda")]
unsafe impl Send for DeviceContext {}
#[cfg(feature = "cuda")]
unsafe impl Sync for DeviceContext {}

#[cfg(feature = "cuda")]
impl DeviceContext {
    pub fn new(device_index: usize) -> Result<Self> {
        let ctx = cudarc::driver::CudaContext::new(device_index)?;
        let (sm_major, sm_minor) = ctx.compute_capability().unwrap_or((0, 0));
        let stream = ctx.default_stream();
        let cublas = cudarc::cublas::CudaBlas::new(stream.clone())?;
        let tf32 = super::hopper::tf32_enabled()
            && super::hopper::is_tensor_core_gpu(sm_major);
        if tf32 {
            unsafe {
                cudarc::cublas::sys::cublasSetMathMode(
                    *cublas.handle(),
                    cudarc::cublas::sys::cublasMath_t::CUBLAS_TF32_TENSOR_OP_MATH,
                )
                .result()?;
            }
        }
        let cudnn = cudarc::cudnn::Cudnn::new(stream.clone())?;
        let kernels = super::kernels::KernelFns::compile(&ctx, sm_major, sm_minor)?;
        super::log::info(format_args!(
            "cuda device={device_index} sm_{sm_major}{sm_minor} hopper={} tf32={} resident={} bf16={} \
             sdpa={} sdpa_chunk={} device_sched={}",
            super::hopper::is_hopper(sm_major),
            tf32,
            super::resident::residency_enabled(),
            super::bf16_gemm::bf16_enabled(),
            std::env::var("FASTVIDEO_SDPA").unwrap_or_else(|_| "flash".into()),
            super::hopper::sdpa_query_chunk(sm_major),
            super::hopper::device_sched_enabled(),
        ));
        Ok(Self {
            ctx,
            stream,
            cublas,
            cudnn,
            kernels,
            sm_major,
            sm_minor,
            tf32,
        })
    }
}

#[cfg(feature = "cuda")]
static GLOBAL_DEVICE: OnceLock<Mutex<Option<Arc<DeviceContext>>>> = OnceLock::new();

#[cfg(feature = "cuda")]
fn global_slot() -> &'static Mutex<Option<Arc<DeviceContext>>> {
    GLOBAL_DEVICE.get_or_init(|| Mutex::new(None))
}

#[cfg(feature = "cuda")]
pub fn set_global_device(ctx: DeviceContext) {
    *global_slot().lock().expect("device lock") = Some(Arc::new(ctx));
}

#[cfg(feature = "cuda")]
thread_local! {
    /// Per-thread device override. Every `CudaTensor` op reads its device via
    /// [`global_device`], which checks this before the process-wide default.
    /// Real multi-GPU sequence-parallel dispatch (see [`super::sp`]) spawns
    /// one OS thread per rank and sets this to that rank's own
    /// `DeviceContext` — letting the existing single-device-context call
    /// sites (`ops.rs`, `tensor.rs`, …) fan out across GPUs unchanged, since
    /// each thread transparently sees "its" device as the global one.
    static THREAD_DEVICE: std::cell::RefCell<Option<Arc<DeviceContext>>> =
        const { std::cell::RefCell::new(None) };
}

/// Set (or clear, with `None`) the calling thread's device override. See
/// [`THREAD_DEVICE`]. Callers must clear it (`set_thread_device(None)`)
/// before the thread exits reuse (e.g. a thread pool) to avoid leaking a
/// stale per-rank device onto unrelated work.
#[cfg(feature = "cuda")]
pub fn set_thread_device(ctx: Option<Arc<DeviceContext>>) {
    THREAD_DEVICE.with(|d| *d.borrow_mut() = ctx);
}

#[cfg(feature = "cuda")]
pub fn global_device() -> Option<Arc<DeviceContext>> {
    if let Some(dev) = THREAD_DEVICE.with(|d| d.borrow().clone()) {
        return Some(dev);
    }
    global_slot().lock().expect("device lock").clone()
}

/// True when a live CUDA device is available on this thread (thread-local
/// override or the process-wide default). Defined regardless of the `cuda`
/// feature (always `false` without it) so callers — notably
/// [`super::tensor::strict_device_check`] — don't need their own `cfg`
/// branch just to ask this question.
#[cfg(feature = "cuda")]
pub fn has_live_device() -> bool {
    global_device().is_some()
}

#[cfg(not(feature = "cuda"))]
pub fn has_live_device() -> bool {
    false
}

/// Registry of lazily-created `DeviceContext`s keyed by CUDA device index,
/// for genuine multi-GPU dispatch (as opposed to the single process-wide
/// device set by [`set_global_device`]/[`resolve_device`]). Each physical
/// device's context (CUDA context, cuBLAS/cuDNN handles, compiled NVRTC
/// kernels) is created once and reused across steps.
#[cfg(feature = "cuda")]
static DEVICE_REGISTRY: OnceLock<Mutex<std::collections::HashMap<usize, Arc<DeviceContext>>>> =
    OnceLock::new();

#[cfg(feature = "cuda")]
fn device_registry() -> &'static Mutex<std::collections::HashMap<usize, Arc<DeviceContext>>> {
    DEVICE_REGISTRY.get_or_init(|| Mutex::new(std::collections::HashMap::new()))
}

/// Get-or-create the `DeviceContext` for CUDA device `index`. Used by
/// multi-GPU sequence-parallel dispatch (`sp::device_for_rank` maps a rank to
/// an index; this turns that index into a real, cached context).
#[cfg(feature = "cuda")]
pub fn device_for_index(index: usize) -> Result<Arc<DeviceContext>> {
    let mut reg = device_registry().lock().expect("device registry lock");
    if let Some(dev) = reg.get(&index) {
        return Ok(dev.clone());
    }
    let dev = Arc::new(DeviceContext::new(index)?);
    reg.insert(index, dev.clone());
    Ok(dev)
}

/// Row-major `(m, k) @ (k, n) -> (m, n)` on already-resident device buffers (no H2D/D2H).
#[cfg(feature = "cuda")]
pub fn matmul_2d_f32_device(
    a: &cudarc::driver::CudaSlice<f32>,
    b: &cudarc::driver::CudaSlice<f32>,
    out: &mut cudarc::driver::CudaSlice<f32>,
    m: usize,
    k: usize,
    n: usize,
) -> Result<()> {
    use cudarc::cublas::{safe::Gemm, sys};
    use cudarc::driver::{DevicePtr, DevicePtrMut};

    let dev = global_device().ok_or_else(|| {
        DeviceError::Message("no global CUDA device context".into())
    })?;
    if a.len() != m * k || b.len() != k * n || out.len() != m * n {
        return Err(DeviceError::Message(format!(
            "matmul_device size mismatch: a={} b={} out={} for ({m},{k})@({k},{n})",
            a.len(),
            b.len(),
            out.len()
        )));
    }
    if dev.tf32 {
        let alpha = 1.0f32;
        let beta = 0.0f32;
        let (a_ptr, _ra) = b.device_ptr(&dev.stream);
        let (b_ptr, _rb) = a.device_ptr(&dev.stream);
        let (c_ptr, _rc) = out.device_ptr_mut(&dev.stream);
        unsafe {
            cudarc::cublas::result::gemm_ex(
                *dev.cublas.handle(),
                sys::cublasOperation_t::CUBLAS_OP_N,
                sys::cublasOperation_t::CUBLAS_OP_N,
                n as i32,
                m as i32,
                k as i32,
                (&alpha) as *const f32 as *const _,
                a_ptr as *const _,
                sys::cudaDataType_t::CUDA_R_32F,
                n as i32,
                b_ptr as *const _,
                sys::cudaDataType_t::CUDA_R_32F,
                k as i32,
                (&beta) as *const f32 as *const _,
                c_ptr as *mut _,
                sys::cudaDataType_t::CUDA_R_32F,
                n as i32,
                sys::cublasComputeType_t::CUBLAS_COMPUTE_32F_FAST_TF32,
                sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT_TENSOR_OP,
            )?;
        }
        drop(_ra);
        drop(_rb);
        drop(_rc);
        return Ok(());
    }
    unsafe {
        dev.cublas.gemm(
            cudarc::cublas::GemmConfig {
                transa: sys::cublasOperation_t::CUBLAS_OP_N,
                transb: sys::cublasOperation_t::CUBLAS_OP_N,
                m: n as i32,
                n: m as i32,
                k: k as i32,
                alpha: 1.0,
                lda: n as i32,
                ldb: k as i32,
                beta: 0.0,
                ldc: n as i32,
            },
            b,
            a,
            out,
        )?;
    }
    Ok(())
}

/// `X [m,k] @ W^T` where `W` is row-major `[n, k]` (Linear weight) — no host transpose.
#[cfg(feature = "cuda")]
pub fn matmul_linear_wt_device(
    x: &cudarc::driver::CudaSlice<f32>,
    w: &cudarc::driver::CudaSlice<f32>,
    out: &mut cudarc::driver::CudaSlice<f32>,
    m: usize,
    k: usize,
    n: usize,
) -> Result<()> {
    use cudarc::cublas::{safe::Gemm, sys};
    use cudarc::driver::{DevicePtr, DevicePtrMut};

    let dev = global_device().ok_or_else(|| {
        DeviceError::Message("no global CUDA device context".into())
    })?;
    if x.len() != m * k || w.len() != n * k || out.len() != m * n {
        return Err(DeviceError::Message(format!(
            "matmul_linear_wt size mismatch: x={} w={} out={} for m={m} k={k} n={n}",
            x.len(),
            w.len(),
            out.len()
        )));
    }
    if dev.tf32 {
        let alpha = 1.0f32;
        let beta = 0.0f32;
        let (a_ptr, _ra) = w.device_ptr(&dev.stream);
        let (b_ptr, _rb) = x.device_ptr(&dev.stream);
        let (c_ptr, _rc) = out.device_ptr_mut(&dev.stream);
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
                sys::cudaDataType_t::CUDA_R_32F,
                k as i32,
                b_ptr as *const _,
                sys::cudaDataType_t::CUDA_R_32F,
                k as i32,
                (&beta) as *const f32 as *const _,
                c_ptr as *mut _,
                sys::cudaDataType_t::CUDA_R_32F,
                n as i32,
                sys::cublasComputeType_t::CUBLAS_COMPUTE_32F_FAST_TF32,
                sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT_TENSOR_OP,
            )?;
        }
        drop(_ra);
        drop(_rb);
        drop(_rc);
        return Ok(());
    }
    unsafe {
        dev.cublas.gemm(
            cudarc::cublas::GemmConfig {
                transa: sys::cublasOperation_t::CUBLAS_OP_T,
                transb: sys::cublasOperation_t::CUBLAS_OP_N,
                m: n as i32,
                n: m as i32,
                k: k as i32,
                alpha: 1.0,
                lda: k as i32,
                ldb: k as i32,
                beta: 0.0,
                ldc: n as i32,
            },
            w,
            x,
            out,
        )?;
    }
    Ok(())
}

/// Strided-batched `X [batch,m,k] @ W^T` where each `W` tile is row-major `[n,k]`.
/// Used for attention scores `Q @ K^T` with `batch = B*H`.
#[cfg(feature = "cuda")]
pub fn matmul_linear_wt_strided_batched(
    x: &cudarc::driver::CudaSlice<f32>,
    w: &cudarc::driver::CudaSlice<f32>,
    out: &mut cudarc::driver::CudaSlice<f32>,
    batch: usize,
    m: usize,
    k: usize,
    n: usize,
    scale: f32,
) -> Result<()> {
    use cudarc::cublas::{
        safe::{Gemm, GemmConfig, StridedBatchedConfig},
        sys,
    };
    use cudarc::driver::{DevicePtr, DevicePtrMut};

    let dev = global_device().ok_or_else(|| {
        DeviceError::Message("no global CUDA device context".into())
    })?;
    let stride_x = (m * k) as i64;
    let stride_w = (n * k) as i64;
    let stride_c = (m * n) as i64;
    if x.len() != batch * m * k || w.len() != batch * n * k || out.len() != batch * m * n {
        return Err(DeviceError::Message(format!(
            "strided wt gemm size mismatch: x={} w={} out={} batch={batch} m={m} k={k} n={n}",
            x.len(),
            w.len(),
            out.len()
        )));
    }
    if dev.tf32 {
        let alpha = scale;
        let beta = 0.0f32;
        let (a_ptr, _ra) = w.device_ptr(&dev.stream);
        let (b_ptr, _rb) = x.device_ptr(&dev.stream);
        let (c_ptr, _rc) = out.device_ptr_mut(&dev.stream);
        unsafe {
            cudarc::cublas::result::gemm_strided_batched_ex(
                *dev.cublas.handle(),
                sys::cublasOperation_t::CUBLAS_OP_T,
                sys::cublasOperation_t::CUBLAS_OP_N,
                n as i32,
                m as i32,
                k as i32,
                (&alpha) as *const f32 as *const _,
                a_ptr as *const _,
                sys::cudaDataType_t::CUDA_R_32F,
                k as i32,
                stride_w,
                b_ptr as *const _,
                sys::cudaDataType_t::CUDA_R_32F,
                k as i32,
                stride_x,
                (&beta) as *const f32 as *const _,
                c_ptr as *mut _,
                sys::cudaDataType_t::CUDA_R_32F,
                n as i32,
                stride_c,
                batch as i32,
                sys::cublasComputeType_t::CUBLAS_COMPUTE_32F_FAST_TF32,
                sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT_TENSOR_OP,
            )?;
        }
        drop(_ra);
        drop(_rb);
        drop(_rc);
        return Ok(());
    }
    unsafe {
        dev.cublas.gemm_strided_batched(
            StridedBatchedConfig {
                gemm: GemmConfig {
                    transa: sys::cublasOperation_t::CUBLAS_OP_T,
                    transb: sys::cublasOperation_t::CUBLAS_OP_N,
                    m: n as i32,
                    n: m as i32,
                    k: k as i32,
                    alpha: scale,
                    lda: k as i32,
                    ldb: k as i32,
                    beta: 0.0,
                    ldc: n as i32,
                },
                batch_size: batch as i32,
                stride_a: stride_w,
                stride_b: stride_x,
                stride_c: stride_c,
            },
            w,
            x,
            out,
        )?;
    }
    Ok(())
}

/// Strided-batched row-major `(m,k) @ (k,n)` over `batch` tiles (attention `P @ V`).
#[cfg(feature = "cuda")]
pub fn matmul_2d_strided_batched(
    a: &cudarc::driver::CudaSlice<f32>,
    b: &cudarc::driver::CudaSlice<f32>,
    out: &mut cudarc::driver::CudaSlice<f32>,
    batch: usize,
    m: usize,
    k: usize,
    n: usize,
) -> Result<()> {
    use cudarc::cublas::{
        safe::{Gemm, GemmConfig, StridedBatchedConfig},
        sys,
    };
    use cudarc::driver::{DevicePtr, DevicePtrMut};

    let dev = global_device().ok_or_else(|| {
        DeviceError::Message("no global CUDA device context".into())
    })?;
    let stride_a = (m * k) as i64;
    let stride_b = (k * n) as i64;
    let stride_c = (m * n) as i64;
    if a.len() != batch * m * k || b.len() != batch * k * n || out.len() != batch * m * n {
        return Err(DeviceError::Message(format!(
            "strided gemm size mismatch: a={} b={} out={} batch={batch} ({m},{k})@({k},{n})",
            a.len(),
            b.len(),
            out.len()
        )));
    }
    if dev.tf32 {
        let alpha = 1.0f32;
        let beta = 0.0f32;
        let (a_ptr, _ra) = b.device_ptr(&dev.stream);
        let (b_ptr, _rb) = a.device_ptr(&dev.stream);
        let (c_ptr, _rc) = out.device_ptr_mut(&dev.stream);
        unsafe {
            cudarc::cublas::result::gemm_strided_batched_ex(
                *dev.cublas.handle(),
                sys::cublasOperation_t::CUBLAS_OP_N,
                sys::cublasOperation_t::CUBLAS_OP_N,
                n as i32,
                m as i32,
                k as i32,
                (&alpha) as *const f32 as *const _,
                a_ptr as *const _,
                sys::cudaDataType_t::CUDA_R_32F,
                n as i32,
                stride_b,
                b_ptr as *const _,
                sys::cudaDataType_t::CUDA_R_32F,
                k as i32,
                stride_a,
                (&beta) as *const f32 as *const _,
                c_ptr as *mut _,
                sys::cudaDataType_t::CUDA_R_32F,
                n as i32,
                stride_c,
                batch as i32,
                sys::cublasComputeType_t::CUBLAS_COMPUTE_32F_FAST_TF32,
                sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT_TENSOR_OP,
            )?;
        }
        drop(_ra);
        drop(_rb);
        drop(_rc);
        return Ok(());
    }
    unsafe {
        dev.cublas.gemm_strided_batched(
            StridedBatchedConfig {
                gemm: GemmConfig {
                    transa: sys::cublasOperation_t::CUBLAS_OP_N,
                    transb: sys::cublasOperation_t::CUBLAS_OP_N,
                    m: n as i32,
                    n: m as i32,
                    k: k as i32,
                    alpha: 1.0,
                    lda: n as i32,
                    ldb: k as i32,
                    beta: 0.0,
                    ldc: n as i32,
                },
                batch_size: batch as i32,
                stride_a: stride_b,
                stride_b: stride_a,
                stride_c: stride_c,
            },
            b,
            a,
            out,
        )?;
    }
    Ok(())
}

/// Minimum element count a strided-batched view needs: `batch` tiles of
/// `per_batch_len` elements each, `outer_stride` elements apart, starting
/// from the view's own offset (already baked into the `CudaView` the caller
/// sliced). Shared by [`matmul_linear_wt_strided_batched_x_view`] and
/// [`matmul_2d_strided_batched_out_view`]'s size checks; pulled out as a
/// plain function (no `cfg(feature = "cuda")`) so the arithmetic itself is
/// unit-testable without a live CUDA device — see `tests::` below.
fn strided_view_required_len(batch: usize, outer_stride: usize, per_batch_len: usize) -> usize {
    batch.saturating_sub(1) * outer_stride + per_batch_len
}

/// Chunked-attention variant of [`matmul_linear_wt_strided_batched`]: `x` is
/// a *view* into a larger `[batch, outer_rows, k]` buffer (e.g. one
/// query-chunk of a longer sequence), starting wherever the caller sliced it
/// from and with `outer_stride_x` elements between consecutive batches in
/// that larger buffer — independent of `m` (rows actually used per batch in
/// this call). This lets a chunked SDPA loop run `Q@K^T` directly on a
/// strided sub-range of `Q` with no gather copy, instead of first `memcpy`ing
/// each batch's chunk into a freshly-packed contiguous buffer.
///
/// `w`/`out` are unchanged from the whole-buffer version (`w` — K — is the
/// same for every chunk; `out` — scores — is a fresh per-chunk allocation).
#[cfg(feature = "cuda")]
pub fn matmul_linear_wt_strided_batched_x_view(
    x: &cudarc::driver::CudaView<'_, f32>,
    outer_stride_x: usize,
    w: &cudarc::driver::CudaSlice<f32>,
    out: &mut cudarc::driver::CudaSlice<f32>,
    batch: usize,
    m: usize,
    k: usize,
    n: usize,
    scale: f32,
) -> Result<()> {
    use cudarc::cublas::{
        safe::{Gemm, GemmConfig, StridedBatchedConfig},
        sys,
    };
    use cudarc::driver::{DevicePtr, DevicePtrMut};

    let dev = global_device().ok_or_else(|| {
        DeviceError::Message("no global CUDA device context".into())
    })?;
    let stride_x = outer_stride_x as i64;
    let stride_w = (n * k) as i64;
    let stride_c = (m * n) as i64;
    let x_required = strided_view_required_len(batch, outer_stride_x, m * k);
    if x.len() < x_required || w.len() != batch * n * k || out.len() != batch * m * n {
        return Err(DeviceError::Message(format!(
            "strided wt gemm (x-view) size mismatch: x.len()={} (need >= {x_required}) w={} out={} \
             batch={batch} m={m} k={k} n={n} outer_stride_x={outer_stride_x}",
            x.len(),
            w.len(),
            out.len()
        )));
    }
    if dev.tf32 {
        let alpha = scale;
        let beta = 0.0f32;
        let (a_ptr, _ra) = w.device_ptr(&dev.stream);
        let (b_ptr, _rb) = x.device_ptr(&dev.stream);
        let (c_ptr, _rc) = out.device_ptr_mut(&dev.stream);
        unsafe {
            cudarc::cublas::result::gemm_strided_batched_ex(
                *dev.cublas.handle(),
                sys::cublasOperation_t::CUBLAS_OP_T,
                sys::cublasOperation_t::CUBLAS_OP_N,
                n as i32,
                m as i32,
                k as i32,
                (&alpha) as *const f32 as *const _,
                a_ptr as *const _,
                sys::cudaDataType_t::CUDA_R_32F,
                k as i32,
                stride_w,
                b_ptr as *const _,
                sys::cudaDataType_t::CUDA_R_32F,
                k as i32,
                stride_x,
                (&beta) as *const f32 as *const _,
                c_ptr as *mut _,
                sys::cudaDataType_t::CUDA_R_32F,
                n as i32,
                stride_c,
                batch as i32,
                sys::cublasComputeType_t::CUBLAS_COMPUTE_32F_FAST_TF32,
                sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT_TENSOR_OP,
            )?;
        }
        drop(_ra);
        drop(_rb);
        drop(_rc);
        return Ok(());
    }
    unsafe {
        dev.cublas.gemm_strided_batched(
            StridedBatchedConfig {
                gemm: GemmConfig {
                    transa: sys::cublasOperation_t::CUBLAS_OP_T,
                    transb: sys::cublasOperation_t::CUBLAS_OP_N,
                    m: n as i32,
                    n: m as i32,
                    k: k as i32,
                    alpha: scale,
                    lda: k as i32,
                    ldb: k as i32,
                    beta: 0.0,
                    ldc: n as i32,
                },
                batch_size: batch as i32,
                stride_a: stride_w,
                stride_b: stride_x,
                stride_c,
            },
            w,
            x,
            out,
        )?;
    }
    Ok(())
}

/// Chunked-attention variant of [`matmul_2d_strided_batched`]: `out` is a
/// *view* into a larger `[batch, outer_rows, n]` buffer, with
/// `outer_stride_c` elements between consecutive batches — lets a chunked
/// `P@V` write its result directly into the right slice of the full-sequence
/// output buffer with no scatter copy back afterward.
#[cfg(feature = "cuda")]
pub fn matmul_2d_strided_batched_out_view(
    a: &cudarc::driver::CudaSlice<f32>,
    b: &cudarc::driver::CudaSlice<f32>,
    out: &mut cudarc::driver::CudaViewMut<'_, f32>,
    outer_stride_c: usize,
    batch: usize,
    m: usize,
    k: usize,
    n: usize,
) -> Result<()> {
    use cudarc::cublas::{
        safe::{Gemm, GemmConfig, StridedBatchedConfig},
        sys,
    };
    use cudarc::driver::{DevicePtr, DevicePtrMut};

    let dev = global_device().ok_or_else(|| {
        DeviceError::Message("no global CUDA device context".into())
    })?;
    let stride_a = (m * k) as i64;
    let stride_b = (k * n) as i64;
    let stride_c = outer_stride_c as i64;
    let out_required = strided_view_required_len(batch, outer_stride_c, m * n);
    if a.len() != batch * m * k || b.len() != batch * k * n || out.len() < out_required {
        return Err(DeviceError::Message(format!(
            "strided gemm (out-view) size mismatch: a={} b={} out.len()={} (need >= {out_required}) \
             batch={batch} ({m},{k})@({k},{n}) outer_stride_c={outer_stride_c}",
            a.len(),
            b.len(),
            out.len()
        )));
    }
    if dev.tf32 {
        let alpha = 1.0f32;
        let beta = 0.0f32;
        let (a_ptr, _ra) = b.device_ptr(&dev.stream);
        let (b_ptr, _rb) = a.device_ptr(&dev.stream);
        let (c_ptr, _rc) = out.device_ptr_mut(&dev.stream);
        unsafe {
            cudarc::cublas::result::gemm_strided_batched_ex(
                *dev.cublas.handle(),
                sys::cublasOperation_t::CUBLAS_OP_N,
                sys::cublasOperation_t::CUBLAS_OP_N,
                n as i32,
                m as i32,
                k as i32,
                (&alpha) as *const f32 as *const _,
                a_ptr as *const _,
                sys::cudaDataType_t::CUDA_R_32F,
                n as i32,
                stride_b,
                b_ptr as *const _,
                sys::cudaDataType_t::CUDA_R_32F,
                k as i32,
                stride_a,
                (&beta) as *const f32 as *const _,
                c_ptr as *mut _,
                sys::cudaDataType_t::CUDA_R_32F,
                n as i32,
                stride_c,
                batch as i32,
                sys::cublasComputeType_t::CUBLAS_COMPUTE_32F_FAST_TF32,
                sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT_TENSOR_OP,
            )?;
        }
        drop(_ra);
        drop(_rb);
        drop(_rc);
        return Ok(());
    }
    unsafe {
        dev.cublas.gemm_strided_batched(
            StridedBatchedConfig {
                gemm: GemmConfig {
                    transa: sys::cublasOperation_t::CUBLAS_OP_N,
                    transb: sys::cublasOperation_t::CUBLAS_OP_N,
                    m: n as i32,
                    n: m as i32,
                    k: k as i32,
                    alpha: 1.0,
                    lda: n as i32,
                    ldb: k as i32,
                    beta: 0.0,
                    ldc: n as i32,
                },
                batch_size: batch as i32,
                stride_a: stride_b,
                stride_b: stride_a,
                stride_c,
            },
            b,
            a,
            out,
        )?;
    }
    Ok(())
}

/// GPU 4D permute; `dims[o]` = input axis feeding output axis `o`.
#[cfg(feature = "cuda")]
pub fn permute_4d_device(
    input: &cudarc::driver::CudaSlice<f32>,
    in_shape: [usize; 4],
    dims: [usize; 4],
) -> Result<cudarc::driver::CudaSlice<f32>> {
    let dev = global_device().ok_or_else(|| {
        DeviceError::Message("no global CUDA device context".into())
    })?;
    let n: usize = in_shape.iter().product();
    if input.len() != n {
        return Err(DeviceError::Message("permute_4d len mismatch".into()));
    }
    let mut out = dev.stream.alloc_zeros::<f32>(n)?;
    unsafe {
        super::kernels::launch_permute_4d(
            &dev.stream,
            &dev.kernels.permute_4d,
            input,
            &mut out,
            [
                in_shape[0] as i32,
                in_shape[1] as i32,
                in_shape[2] as i32,
                in_shape[3] as i32,
            ],
            [dims[0] as i32, dims[1] as i32, dims[2] as i32, dims[3] as i32],
        )?;
    }
    Ok(out)
}

/// Row-major `(m, k) @ (k, n) -> (m, n)` via cuBLAS when a global device is set.
/// Host upload API kept for fallback / non-resident paths.
#[cfg(feature = "cuda")]
pub fn matmul_2d_f32(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Result<Vec<f32>> {
    let dev = global_device().ok_or_else(|| {
        DeviceError::Message("no global CUDA device context".into())
    })?;
    if a.len() != m * k || b.len() != k * n {
        return Err(DeviceError::Message(format!(
            "matmul shape mismatch: a={} b={} for ({m},{k})@({k},{n})",
            a.len(),
            b.len()
        )));
    }
    let a_dev = dev.stream.memcpy_stod(a)?;
    let b_dev = dev.stream.memcpy_stod(b)?;
    let mut c_dev = dev.stream.alloc_zeros::<f32>(m * n)?;
    matmul_2d_f32_device(&a_dev, &b_dev, &mut c_dev, m, k, n)?;
    Ok(dev.stream.memcpy_dtov(&c_dev)?)
}

/// NCHW conv2d via cuDNN (groups=1, dilation=1). Bias is applied on host after download.
#[cfg(feature = "cuda")]
pub fn conv2d_f32(
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
) -> Result<Vec<f32>> {
    use cudarc::cudnn::{sys, ConvForward};

    let dev = global_device().ok_or_else(|| {
        DeviceError::Message("no global CUDA device context".into())
    })?;
    let stride = stride.max(1);
    let out_h = (h + 2 * padding - kh) / stride + 1;
    let out_w = (w + 2 * padding - kw) / stride + 1;
    if input.len() != n * c_in * h * w || weight.len() != c_out * c_in * kh * kw {
        return Err(DeviceError::Message("conv2d buffer size mismatch".into()));
    }

    let pad = [padding as i32, padding as i32];
    let stride_hw = [stride as i32, stride as i32];
    let dilation = [1i32, 1];
    let conv = dev.cudnn.create_conv2d::<f32>(
        pad,
        stride_hw,
        dilation,
        sys::cudnnConvolutionMode_t::CUDNN_CROSS_CORRELATION,
    )?;
    let x_desc = dev.cudnn.create_4d_tensor::<f32>(
        sys::cudnnTensorFormat_t::CUDNN_TENSOR_NCHW,
        [n as i32, c_in as i32, h as i32, w as i32],
    )?;
    let w_desc = dev.cudnn.create_4d_filter::<f32>(
        sys::cudnnTensorFormat_t::CUDNN_TENSOR_NCHW,
        [c_out as i32, c_in as i32, kh as i32, kw as i32],
    )?;
    let y_desc = dev.cudnn.create_4d_tensor::<f32>(
        sys::cudnnTensorFormat_t::CUDNN_TENSOR_NCHW,
        [n as i32, c_out as i32, out_h as i32, out_w as i32],
    )?;

    let x_dev = dev.stream.memcpy_stod(input)?;
    let w_dev = dev.stream.memcpy_stod(weight)?;
    let mut y_dev = dev.stream.alloc_zeros::<f32>(n * c_out * out_h * out_w)?;

    let op = ConvForward {
        conv: &conv,
        x: &x_desc,
        w: &w_desc,
        y: &y_desc,
    };
    let algo = op.pick_algorithm()?;
    let workspace_size = op.get_workspace_size(algo)?;
    let mut workspace = if workspace_size > 0 {
        Some(dev.stream.alloc_zeros::<u8>(workspace_size)?)
    } else {
        None
    };
    unsafe {
        op.launch(
            algo,
            workspace.as_mut(),
            (1.0f32, 0.0f32),
            &x_dev,
            &w_dev,
            &mut y_dev,
        )?;
    }
    Ok(dev.stream.memcpy_dtov(&y_dev)?)
}

/// Resolve device from CLI spec (`cpu`, `cuda`, `cuda:0`).
pub fn resolve_device(spec: &str) -> Result<()> {
    let s = spec.trim().to_ascii_lowercase();
    #[cfg(feature = "cuda")]
    {
        if s == "cpu" {
            *global_slot().lock().expect("device lock") = None;
            super::log::info(format_args!("device=cpu (no global CUDA context)"));
            return Ok(());
        }
        if s == "cuda" || s == "cuda:0" {
            set_global_device(DeviceContext::new(0)?);
            return Ok(());
        }
        if let Some(rest) = s.strip_prefix("cuda:") {
            let idx: usize = rest
                .parse()
                .map_err(|_| DeviceError::Message(format!("invalid cuda device `{spec}`")))?;
            set_global_device(DeviceContext::new(idx)?);
            return Ok(());
        }
        return Err(DeviceError::Message(format!(
            "unknown device `{spec}` (expected cpu, cuda, or cuda:N)"
        )));
    }
    #[cfg(not(feature = "cuda"))]
    {
        if s.starts_with("cuda") {
            return Err(DeviceError::Message(
                "CUDA requested but fastvideo-cudarc was built without `--features cuda`".into(),
            ));
        }
        super::log::info(format_args!("device={spec} (cuda feature off)"));
        Ok(())
    }
}

#[cfg(test)]
mod strided_view_tests {
    use super::strided_view_required_len;

    /// Replays `attn::device_dense_sdpa`'s chunked-path loop
    /// (`while start < sq { qlen = chunk.min(sq - start); ...; start += qlen }`)
    /// purely on the index arithmetic, and checks that at every iteration the
    /// no-copy offset-view read (`Q`, `[bh, sq, d]`, offset `start*d`) and
    /// write (`out`, same shape, same offset) both stay within the real
    /// buffer's element count. This is the invariant
    /// `matmul_linear_wt_strided_batched_x_view` /
    /// `matmul_2d_strided_batched_out_view`'s own runtime checks enforce —
    /// this test proves the *caller* (the chunk loop) never violates it,
    /// across chunk sizes that do and don't evenly divide `sq`.
    fn assert_chunk_loop_stays_in_bounds(bh: usize, sq: usize, d: usize, sk: usize, chunk: usize) {
        let buf_len = bh * sq * d; // both Q and the SDPA output share this shape
        let mut start = 0usize;
        let mut iterations = 0usize;
        while start < sq {
            let qlen = chunk.min(sq - start);
            assert!(qlen > 0, "qlen must make progress (start={start}, sq={sq})");

            // The view starts at `start*d` and must cover `x_required` more
            // elements from there — i.e. the *original* (unsliced) buffer
            // must be at least `start*d + x_required` long.
            let x_required = strided_view_required_len(bh, sq * d, qlen * d);
            assert!(
                start * d + x_required <= buf_len,
                "Q read out of bounds: start={start} qlen={qlen} sq={sq} bh={bh} d={d} \
                 (start*d + required = {} > buf_len = {buf_len})",
                start * d + x_required
            );

            let out_required = strided_view_required_len(bh, sq * d, qlen * d);
            assert!(
                start * d + out_required <= buf_len,
                "out write out of bounds: start={start} qlen={qlen} sq={sq} bh={bh} d={d} \
                 (start*d + required = {} > buf_len = {buf_len})",
                start * d + out_required
            );

            // sk only matters for the intermediate `scores`/`probs` buffers,
            // which are freshly allocated per-chunk at exactly `bh*qlen*sk` —
            // no offset math there, so just sanity-check it's nonzero here
            // (it's an input to the real function, unused by this bounds
            // check, but keeps the test signature matching the real call site).
            assert!(sk > 0);

            start += qlen;
            iterations += 1;
            assert!(iterations <= sq + 1, "loop did not terminate");
        }
        assert_eq!(start, sq, "loop must cover the whole sequence exactly once");
    }

    #[test]
    fn chunk_evenly_divides_sequence() {
        assert_chunk_loop_stays_in_bounds(4, 1024, 64, 4096, 256);
    }

    #[test]
    fn chunk_does_not_evenly_divide_sequence() {
        // sq=1000 with chunk=256 -> chunks of 256,256,256,232 (remainder tail).
        assert_chunk_loop_stays_in_bounds(4, 1000, 64, 4096, 256);
    }

    #[test]
    fn chunk_larger_than_sequence_is_one_iteration() {
        assert_chunk_loop_stays_in_bounds(2, 100, 128, 512, 1024);
    }

    #[test]
    fn single_batch_head_odd_remainder() {
        assert_chunk_loop_stays_in_bounds(1, 777, 32, 2048, 100);
    }

    #[test]
    fn many_batch_heads_small_chunk() {
        assert_chunk_loop_stays_in_bounds(40, 4097, 64, 4096, 64);
    }
}
