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

/// Floating-point math cuBLAS may use for F32 GEMMs.
#[cfg(feature = "cuda")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GemmMath {
    /// Plain FP32: the reference (`FASTVIDEO_BF16=0 FASTVIDEO_TF32=0`).
    F32,
    /// TF32 Tensor Core math on F32 buffers (`FASTVIDEO_BF16=0`).
    Tf32,
    /// bfloat16 compute on F32 buffers (default): the same math as upstream's
    /// bf16 autocast, without keeping a second weight copy or casting kernels.
    Bf16,
}

#[cfg(feature = "cuda")]
impl GemmMath {
    pub fn compute_type(self) -> cudarc::cublas::sys::cublasComputeType_t {
        use cudarc::cublas::sys::cublasComputeType_t as C;
        match self {
            GemmMath::F32 => C::CUBLAS_COMPUTE_32F,
            GemmMath::Tf32 => C::CUBLAS_COMPUTE_32F_FAST_TF32,
            GemmMath::Bf16 => C::CUBLAS_COMPUTE_32F_FAST_16BF,
        }
    }
}

#[cfg(feature = "cuda")]
pub struct DeviceContext {
    pub ctx: Arc<cudarc::driver::CudaContext>,
    pub stream: Arc<cudarc::driver::CudaStream>,
    pub cublas: cudarc::cublas::CudaBlas,
    pub cudnn: Arc<cudarc::cudnn::Cudnn>,
    pub kernels: super::kernels::KernelFns,
    pub sm_major: i32,
    pub sm_minor: i32,
    pub gemm_math: GemmMath,
    /// cuDNN convolution plans (descriptors + algorithm) keyed by shape, and
    /// the shared growable workspace.
    pub conv: Mutex<super::conv::ConvCache>,
}

// cudarc's `Cudnn` and descriptors are `!Send`/`!Sync` (raw handles). This
// crate drives one device from one thread at a time (the SP path gives each
// rank its own context), and the conv cache is behind a `Mutex`.
#[cfg(feature = "cuda")]
unsafe impl Send for DeviceContext {}
#[cfg(feature = "cuda")]
unsafe impl Sync for DeviceContext {}

#[cfg(feature = "cuda")]
impl DeviceContext {
    pub fn new(device_index: usize) -> Result<Self> {
        let ctx = cudarc::driver::CudaContext::new(device_index)?;
        // One stream, synchronized explicitly: per-buffer CudaEvents would add
        // two event objects to every allocation and a record to every launch.
        unsafe { ctx.disable_event_tracking() };
        // Sleep instead of spinning while the host waits on the GPU.
        if let Err(e) = ctx.set_blocking_synchronize() {
            super::log::info(format_args!("blocking synchronize unavailable: {e}"));
        }
        keep_memory_pool(&ctx);
        let (sm_major, sm_minor) = ctx.compute_capability().unwrap_or((0, 0));
        let stream = ctx.default_stream();
        let cublas = cudarc::cublas::CudaBlas::new(stream.clone())?;
        let gemm_math = if super::bf16_gemm::bf16_enabled() && super::hopper::is_tensor_core_gpu(sm_major) {
            GemmMath::Bf16
        } else if super::hopper::tf32_enabled() && super::hopper::is_tensor_core_gpu(sm_major) {
            GemmMath::Tf32
        } else {
            GemmMath::F32
        };
        let cudnn = cudarc::cudnn::Cudnn::new(stream.clone())?;
        let (kernels, origin) = super::kernels::KernelFns::load_for(&ctx, sm_major, sm_minor)?;
        super::log::info(format_args!(
            "cuda device={device_index} sm_{sm_major}{sm_minor} kernels={origin:?} gemm={gemm_math:?} resident={} sdpa={} \
             sdpa_chunk={}",
            super::resident::residency_enabled(),
            super::nn::sdpa_backend(),
            super::hopper::sdpa_query_chunk(sm_major),
        ));
        Ok(Self {
            ctx,
            stream,
            cublas,
            cudnn,
            kernels,
            sm_major,
            sm_minor,
            gemm_math,
            conv: Mutex::new(super::conv::ConvCache::default()),
        })
    }

    /// Block until every queued kernel has finished. Timing code must call
    /// this before reading a clock: launches return as soon as they are queued.
    pub fn synchronize(&self) -> Result<()> {
        Ok(self.stream.synchronize()?)
    }
}

/// Keep freed device memory in the driver's default pool instead of returning
/// it to the OS: `malloc_async` then acts as a size-bucketed caching allocator
/// across denoise steps.
#[cfg(feature = "cuda")]
fn keep_memory_pool(ctx: &Arc<cudarc::driver::CudaContext>) {
    use cudarc::driver::sys;
    if ctx.bind_to_thread().is_err() {
        return;
    }
    unsafe {
        let mut pool: sys::CUmemoryPool = std::ptr::null_mut();
        if sys::cuDeviceGetDefaultMemPool(&mut pool, ctx.cu_device()).result().is_err() || pool.is_null() {
            return;
        }
        let mut threshold = u64::MAX;
        let _ = sys::cuMemPoolSetAttribute(
            pool,
            sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_RELEASE_THRESHOLD,
            (&mut threshold as *mut u64).cast(),
        );
    }
}

#[cfg(feature = "cuda")]
static GLOBAL_DEVICE: OnceLock<Mutex<Option<Arc<DeviceContext>>>> = OnceLock::new();

/// Bumped whenever the global device changes, so threads can cache the Arc.
#[cfg(feature = "cuda")]
static GLOBAL_GENERATION: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

#[cfg(feature = "cuda")]
fn global_slot() -> &'static Mutex<Option<Arc<DeviceContext>>> {
    GLOBAL_DEVICE.get_or_init(|| Mutex::new(None))
}

#[cfg(feature = "cuda")]
fn set_global(dev: Option<Arc<DeviceContext>>) {
    *global_slot().lock().expect("device lock") = dev;
    GLOBAL_GENERATION.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
}

#[cfg(feature = "cuda")]
pub fn set_global_device(ctx: DeviceContext) {
    set_global(Some(Arc::new(ctx)));
}

#[cfg(feature = "cuda")]
thread_local! {
    /// Per-thread device override. Every `CudaTensor` op reads its device via
    /// [`global_device`], which checks this before the process-wide default.
    /// Multi-GPU sequence-parallel dispatch (see [`super::sp`]) spawns one OS
    /// thread per rank and sets this to that rank's own `DeviceContext`.
    static THREAD_DEVICE: std::cell::RefCell<Option<Arc<DeviceContext>>> =
        const { std::cell::RefCell::new(None) };

    /// `(generation, device)` snapshot of the global slot, refreshed only when
    /// the generation changes: the per-op lookup is a TLS read, not a mutex.
    static GLOBAL_CACHE: std::cell::RefCell<(u64, Option<Arc<DeviceContext>>)> =
        const { std::cell::RefCell::new((0, None)) };
}

/// Set (or clear, with `None`) the calling thread's device override. Callers
/// must clear it before the thread is reused for unrelated work.
#[cfg(feature = "cuda")]
pub fn set_thread_device(ctx: Option<Arc<DeviceContext>>) {
    THREAD_DEVICE.with(|d| *d.borrow_mut() = ctx);
}

#[cfg(feature = "cuda")]
pub fn global_device() -> Option<Arc<DeviceContext>> {
    if let Some(dev) = THREAD_DEVICE.with(|d| d.borrow().clone()) {
        return Some(dev);
    }
    let generation = GLOBAL_GENERATION.load(std::sync::atomic::Ordering::Relaxed);
    GLOBAL_CACHE.with(|cache| {
        let mut cache = cache.borrow_mut();
        if cache.0 != generation {
            *cache = (generation, global_slot().lock().expect("device lock").clone());
        }
        cache.1.clone()
    })
}

/// True when a live CUDA device is available on this thread. Defined without
/// the `cuda` feature too (always `false`) so callers need no `cfg`.
#[cfg(feature = "cuda")]
pub fn has_live_device() -> bool {
    if THREAD_DEVICE.with(|d| d.borrow().is_some()) {
        return true;
    }
    let generation = GLOBAL_GENERATION.load(std::sync::atomic::Ordering::Relaxed);
    GLOBAL_CACHE.with(|cache| {
        let cache = cache.borrow();
        if cache.0 == generation {
            return cache.1.is_some();
        }
        drop(cache);
        global_device().is_some()
    })
}

#[cfg(not(feature = "cuda"))]
pub fn has_live_device() -> bool {
    false
}

/// Wait for the live device (if any) to finish queued work.
pub fn synchronize() -> Result<()> {
    #[cfg(feature = "cuda")]
    if let Some(dev) = global_device() {
        return dev.synchronize();
    }
    Ok(())
}

/// Registry of lazily-created `DeviceContext`s keyed by CUDA device index,
/// for multi-GPU dispatch (as opposed to the single process-wide device).
#[cfg(feature = "cuda")]
static DEVICE_REGISTRY: OnceLock<Mutex<std::collections::HashMap<usize, Arc<DeviceContext>>>> =
    OnceLock::new();

#[cfg(feature = "cuda")]
fn device_registry() -> &'static Mutex<std::collections::HashMap<usize, Arc<DeviceContext>>> {
    DEVICE_REGISTRY.get_or_init(|| Mutex::new(std::collections::HashMap::new()))
}

/// Get-or-create the `DeviceContext` for CUDA device `index`.
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

#[cfg(feature = "cuda")]
fn no_device() -> DeviceError {
    DeviceError::Message("no global CUDA device context".into())
}

/// Column-major `cublasGemmStridedBatchedEx` over F32 buffers with the
/// context's [`GemmMath`]. `batch == 1` issues a plain `cublasGemmEx`.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
unsafe fn gemm_raw(
    dev: &DeviceContext,
    transa: bool,
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    a: cudarc::driver::sys::CUdeviceptr,
    lda: usize,
    stride_a: usize,
    b: cudarc::driver::sys::CUdeviceptr,
    ldb: usize,
    stride_b: usize,
    c: cudarc::driver::sys::CUdeviceptr,
    ldc: usize,
    stride_c: usize,
    batch: usize,
) -> Result<()> {
    let r32 = cudarc::cublas::sys::cudaDataType_t::CUDA_R_32F;
    gemm_raw_ty(dev, transa, m, n, k, alpha, a, lda, stride_a, b, ldb, stride_b, c, ldc, stride_c, batch, r32, None)
}

/// [`gemm_raw`] with the A/B element type spelled out (C stays F32). bf16
/// inputs cost nothing numerically under `GemmMath::Bf16`, which already
/// rounds F32 operands to bf16 for the tensor-core op.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
unsafe fn gemm_raw_ty(
    dev: &DeviceContext,
    transa: bool,
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    a: cudarc::driver::sys::CUdeviceptr,
    lda: usize,
    stride_a: usize,
    b: cudarc::driver::sys::CUdeviceptr,
    ldb: usize,
    stride_b: usize,
    c: cudarc::driver::sys::CUdeviceptr,
    ldc: usize,
    stride_c: usize,
    batch: usize,
    ab_ty: cudarc::cublas::sys::cudaDataType_t,
    math: Option<GemmMath>,
) -> Result<()> {
    use cudarc::cublas::sys;
    let op_a = if transa {
        sys::cublasOperation_t::CUBLAS_OP_T
    } else {
        sys::cublasOperation_t::CUBLAS_OP_N
    };
    let beta = 0.0f32;
    let r32 = sys::cudaDataType_t::CUDA_R_32F;
    let algo = sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT_TENSOR_OP;
    let compute = math.unwrap_or(dev.gemm_math).compute_type();
    if batch == 1 {
        cudarc::cublas::result::gemm_ex(
            *dev.cublas.handle(),
            op_a,
            sys::cublasOperation_t::CUBLAS_OP_N,
            m as i32,
            n as i32,
            k as i32,
            (&alpha as *const f32).cast(),
            a as *const _,
            ab_ty,
            lda as i32,
            b as *const _,
            ab_ty,
            ldb as i32,
            (&beta as *const f32).cast(),
            c as *mut _,
            r32,
            ldc as i32,
            compute,
            algo,
        )?;
    } else {
        cudarc::cublas::result::gemm_strided_batched_ex(
            *dev.cublas.handle(),
            op_a,
            sys::cublasOperation_t::CUBLAS_OP_N,
            m as i32,
            n as i32,
            k as i32,
            (&alpha as *const f32).cast(),
            a as *const _,
            ab_ty,
            lda as i32,
            stride_a as i64,
            b as *const _,
            ab_ty,
            ldb as i32,
            stride_b as i64,
            (&beta as *const f32).cast(),
            c as *mut _,
            r32,
            ldc as i32,
            stride_c as i64,
            batch as i32,
            compute,
            algo,
        )?;
    }
    Ok(())
}

#[cfg(feature = "cuda")]
fn size_check(what: &str, ok: bool, detail: impl FnOnce() -> String) -> Result<()> {
    if ok {
        Ok(())
    } else {
        Err(DeviceError::Message(format!("{what} size mismatch: {}", detail())))
    }
}

/// Row-major `(m, k) @ (k, n) -> (m, n)` on device buffers.
#[cfg(feature = "cuda")]
pub fn matmul_2d_f32_device(
    a: &cudarc::driver::CudaSlice<f32>,
    b: &cudarc::driver::CudaSlice<f32>,
    out: &mut cudarc::driver::CudaSlice<f32>,
    m: usize,
    k: usize,
    n: usize,
) -> Result<()> {
    matmul_2d_strided_batched(a, b, out, 1, m, k, n)
}

/// `X [m,k] @ W^T` where `W` is row-major `[n, k]` (Linear weight).
#[cfg(feature = "cuda")]
pub fn matmul_linear_wt_device(
    x: &cudarc::driver::CudaSlice<f32>,
    w: &cudarc::driver::CudaSlice<f32>,
    out: &mut cudarc::driver::CudaSlice<f32>,
    m: usize,
    k: usize,
    n: usize,
) -> Result<()> {
    use cudarc::driver::{DevicePtr, DevicePtrMut};
    let dev = global_device().ok_or_else(no_device)?;
    size_check("matmul_linear_wt", x.len() == m * k && w.len() == n * k && out.len() == m * n, || {
        format!("x={} w={} out={} m={m} k={k} n={n}", x.len(), w.len(), out.len())
    })?;
    let (wp, _rw) = w.device_ptr(&dev.stream);
    let (xp, _rx) = x.device_ptr(&dev.stream);
    let (cp, _rc) = out.device_ptr_mut(&dev.stream);
    unsafe { gemm_raw(&dev, true, n, m, k, 1.0, wp, k, 0, xp, k, 0, cp, n, 0, 1) }
}

/// [`matmul_linear_wt_device`] with an explicit [`GemmMath`] instead of the
/// context's (math-mode probes).
#[cfg(feature = "cuda")]
pub fn matmul_linear_wt_math(
    x: &cudarc::driver::CudaSlice<f32>,
    w: &cudarc::driver::CudaSlice<f32>,
    out: &mut cudarc::driver::CudaSlice<f32>,
    m: usize,
    k: usize,
    n: usize,
    math: GemmMath,
) -> Result<()> {
    use cudarc::cublas::sys;
    use cudarc::driver::{DevicePtr, DevicePtrMut};
    let dev = global_device().ok_or_else(no_device)?;
    size_check("matmul_linear_wt_math", x.len() == m * k && w.len() == n * k && out.len() == m * n, || {
        format!("x={} w={} out={} m={m} k={k} n={n}", x.len(), w.len(), out.len())
    })?;
    let (wp, _rw) = w.device_ptr(&dev.stream);
    let (xp, _rx) = x.device_ptr(&dev.stream);
    let (cp, _rc) = out.device_ptr_mut(&dev.stream);
    let (alpha, beta) = (1.0f32, 0.0f32);
    let r32 = sys::cudaDataType_t::CUDA_R_32F;
    unsafe {
        cudarc::cublas::result::gemm_ex(
            *dev.cublas.handle(),
            sys::cublasOperation_t::CUBLAS_OP_T,
            sys::cublasOperation_t::CUBLAS_OP_N,
            n as i32,
            m as i32,
            k as i32,
            (&alpha as *const f32).cast(),
            wp as *const _,
            r32,
            k as i32,
            xp as *const _,
            r32,
            k as i32,
            (&beta as *const f32).cast(),
            cp as *mut _,
            r32,
            n as i32,
            math.compute_type(),
            sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT_TENSOR_OP,
        )?;
    }
    Ok(())
}

/// `X [m,k] @ W^T` with bfloat16 buffers throughout (the PyTorch bf16 linear:
/// 16BF A/B/C, `CUBLAS_COMPUTE_32F` scaling).
#[cfg(feature = "cuda")]
pub fn matmul_linear_wt_bf16(
    x: &cudarc::driver::CudaSlice<half::bf16>,
    w: &cudarc::driver::CudaSlice<half::bf16>,
    out: &mut cudarc::driver::CudaSlice<half::bf16>,
    m: usize,
    k: usize,
    n: usize,
) -> Result<()> {
    use cudarc::cublas::sys;
    use cudarc::driver::{DevicePtr, DevicePtrMut};
    let dev = global_device().ok_or_else(no_device)?;
    size_check("matmul_linear_wt_bf16", x.len() == m * k && w.len() == n * k && out.len() == m * n, || {
        format!("x={} w={} out={} m={m} k={k} n={n}", x.len(), w.len(), out.len())
    })?;
    let (wp, _rw) = w.device_ptr(&dev.stream);
    let (xp, _rx) = x.device_ptr(&dev.stream);
    let (cp, _rc) = out.device_ptr_mut(&dev.stream);
    let (alpha, beta) = (1.0f32, 0.0f32);
    let bf = sys::cudaDataType_t::CUDA_R_16BF;
    unsafe {
        cudarc::cublas::result::gemm_ex(
            *dev.cublas.handle(),
            sys::cublasOperation_t::CUBLAS_OP_T,
            sys::cublasOperation_t::CUBLAS_OP_N,
            n as i32,
            m as i32,
            k as i32,
            (&alpha as *const f32).cast(),
            wp as *const _,
            bf,
            k as i32,
            xp as *const _,
            bf,
            k as i32,
            (&beta as *const f32).cast(),
            cp as *mut _,
            bf,
            n as i32,
            sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
            sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT_TENSOR_OP,
        )?;
    }
    Ok(())
}

/// Strided-batched `X [batch,m,k] @ W^T * scale` where each `W` tile is row-major
/// `[n,k]`: attention scores `Q @ K^T` with `batch = B*H`.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
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
    use cudarc::driver::{DevicePtr, DevicePtrMut};
    let dev = global_device().ok_or_else(no_device)?;
    size_check(
        "strided wt gemm",
        x.len() == batch * m * k && w.len() == batch * n * k && out.len() == batch * m * n,
        || format!("x={} w={} out={} batch={batch} m={m} k={k} n={n}", x.len(), w.len(), out.len()),
    )?;
    let (wp, _rw) = w.device_ptr(&dev.stream);
    let (xp, _rx) = x.device_ptr(&dev.stream);
    let (cp, _rc) = out.device_ptr_mut(&dev.stream);
    unsafe { gemm_raw(&dev, true, n, m, k, scale, wp, k, n * k, xp, k, m * k, cp, n, m * n, batch) }
}

/// [`matmul_linear_wt_strided_batched`] pinned to F32 math regardless of the
/// context's [`GemmMath`]. VSA's coarse scores decide which tiles the fine
/// stage attends to, and that choice is discrete: letting bf16 rounding flip a
/// near-tie changes the output far more than the rounding itself.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub fn matmul_linear_wt_strided_batched_f32(
    x: &cudarc::driver::CudaSlice<f32>,
    w: &cudarc::driver::CudaSlice<f32>,
    out: &mut cudarc::driver::CudaSlice<f32>,
    batch: usize,
    m: usize,
    k: usize,
    n: usize,
    scale: f32,
) -> Result<()> {
    use cudarc::driver::{DevicePtr, DevicePtrMut};
    let dev = global_device().ok_or_else(no_device)?;
    size_check(
        "strided wt gemm (f32-pinned)",
        x.len() == batch * m * k && w.len() == batch * n * k && out.len() == batch * m * n,
        || format!("x={} w={} out={} batch={batch} m={m} k={k} n={n}", x.len(), w.len(), out.len()),
    )?;
    let (wp, _rw) = w.device_ptr(&dev.stream);
    let (xp, _rx) = x.device_ptr(&dev.stream);
    let (cp, _rc) = out.device_ptr_mut(&dev.stream);
    let r32 = cudarc::cublas::sys::cudaDataType_t::CUDA_R_32F;
    unsafe {
        gemm_raw_ty(&dev, true, n, m, k, scale, wp, k, n * k, xp, k, m * k, cp, n, m * n, batch, r32, Some(GemmMath::F32))
    }
}

/// [`matmul_2d_strided_batched`] pinned to F32 math, for VSA's coarse `P @ V`.
#[cfg(feature = "cuda")]
pub fn matmul_2d_strided_batched_f32(
    a: &cudarc::driver::CudaSlice<f32>,
    b: &cudarc::driver::CudaSlice<f32>,
    out: &mut cudarc::driver::CudaSlice<f32>,
    batch: usize,
    m: usize,
    k: usize,
    n: usize,
) -> Result<()> {
    use cudarc::driver::{DevicePtr, DevicePtrMut};
    let dev = global_device().ok_or_else(no_device)?;
    size_check(
        "strided gemm (f32-pinned)",
        a.len() == batch * m * k && b.len() == batch * k * n && out.len() == batch * m * n,
        || format!("a={} b={} out={} batch={batch} ({m},{k})@({k},{n})", a.len(), b.len(), out.len()),
    )?;
    let (bp, _rb) = b.device_ptr(&dev.stream);
    let (ap, _ra) = a.device_ptr(&dev.stream);
    let (cp, _rc) = out.device_ptr_mut(&dev.stream);
    let r32 = cudarc::cublas::sys::cudaDataType_t::CUDA_R_32F;
    unsafe {
        gemm_raw_ty(&dev, false, n, m, k, 1.0, bp, n, k * n, ap, k, m * k, cp, n, m * n, batch, r32, Some(GemmMath::F32))
    }
}

/// [`matmul_linear_wt_strided_batched`] with bf16 operands and an F32 result:
/// VSA's `Q @ K^T` over gathered tiles.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub fn matmul_linear_wt_strided_batched_bf16(
    x: &cudarc::driver::CudaSlice<half::bf16>,
    w: &cudarc::driver::CudaSlice<half::bf16>,
    out: &mut cudarc::driver::CudaSlice<f32>,
    batch: usize,
    m: usize,
    k: usize,
    n: usize,
    scale: f32,
) -> Result<()> {
    use cudarc::driver::{DevicePtr, DevicePtrMut};
    let dev = global_device().ok_or_else(no_device)?;
    size_check(
        "strided wt gemm (bf16)",
        x.len() == batch * m * k && w.len() == batch * n * k && out.len() == batch * m * n,
        || format!("x={} w={} out={} batch={batch} m={m} k={k} n={n}", x.len(), w.len(), out.len()),
    )?;
    let (wp, _rw) = w.device_ptr(&dev.stream);
    let (xp, _rx) = x.device_ptr(&dev.stream);
    let (cp, _rc) = out.device_ptr_mut(&dev.stream);
    let bf = cudarc::cublas::sys::cudaDataType_t::CUDA_R_16BF;
    unsafe { gemm_raw_ty(&dev, true, n, m, k, scale, wp, k, n * k, xp, k, m * k, cp, n, m * n, batch, bf, None) }
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
    use cudarc::driver::{DevicePtr, DevicePtrMut};
    let dev = global_device().ok_or_else(no_device)?;
    size_check(
        "strided gemm",
        a.len() == batch * m * k && b.len() == batch * k * n && out.len() == batch * m * n,
        || format!("a={} b={} out={} batch={batch} ({m},{k})@({k},{n})", a.len(), b.len(), out.len()),
    )?;
    let (bp, _rb) = b.device_ptr(&dev.stream);
    let (ap, _ra) = a.device_ptr(&dev.stream);
    let (cp, _rc) = out.device_ptr_mut(&dev.stream);
    unsafe { gemm_raw(&dev, false, n, m, k, 1.0, bp, n, k * n, ap, k, m * k, cp, n, m * n, batch) }
}

/// `W [oc, ic] @ X [ic, s]` repeated over `batch` inputs with one shared `W`
/// (stride 0): 1×1 convolutions over channel-first activations, no permute.
#[cfg(feature = "cuda")]
pub fn matmul_shared_left(
    w: &cudarc::driver::CudaSlice<f32>,
    x: &cudarc::driver::CudaSlice<f32>,
    out: &mut cudarc::driver::CudaSlice<f32>,
    batch: usize,
    oc: usize,
    ic: usize,
    s: usize,
) -> Result<()> {
    use cudarc::driver::{DevicePtr, DevicePtrMut};
    let dev = global_device().ok_or_else(no_device)?;
    size_check(
        "shared-left gemm",
        w.len() == oc * ic && x.len() == batch * ic * s && out.len() == batch * oc * s,
        || format!("w={} x={} out={} batch={batch} oc={oc} ic={ic} s={s}", w.len(), x.len(), out.len()),
    )?;
    let (xp, _rx) = x.device_ptr(&dev.stream);
    let (wp, _rw) = w.device_ptr(&dev.stream);
    let (cp, _rc) = out.device_ptr_mut(&dev.stream);
    // Column-major: C[s×oc] = X[s×ic] * W[ic×oc]; W is shared (stride 0).
    unsafe { gemm_raw(&dev, false, s, oc, ic, 1.0, xp, s, ic * s, wp, ic, 0, cp, s, oc * s, batch) }
}

/// Minimum element count a strided-batched view needs: `batch` tiles of
/// `per_batch_len` elements each, `outer_stride` elements apart.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
fn strided_view_required_len(batch: usize, outer_stride: usize, per_batch_len: usize) -> usize {
    batch.saturating_sub(1) * outer_stride + per_batch_len
}

/// Chunked-attention variant of [`matmul_linear_wt_strided_batched`]: `x` is
/// a view into a larger `[batch, outer_rows, k]` buffer with `outer_stride_x`
/// elements between consecutive batches, so a query chunk is addressed in
/// place with no gather copy.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
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
    use cudarc::driver::{DevicePtr, DevicePtrMut};
    let dev = global_device().ok_or_else(no_device)?;
    let x_required = strided_view_required_len(batch, outer_stride_x, m * k);
    size_check(
        "strided wt gemm (x-view)",
        x.len() >= x_required && w.len() == batch * n * k && out.len() == batch * m * n,
        || format!("x={} (need >= {x_required}) w={} out={} batch={batch} m={m} k={k} n={n}", x.len(), w.len(), out.len()),
    )?;
    let (wp, _rw) = w.device_ptr(&dev.stream);
    let (xp, _rx) = x.device_ptr(&dev.stream);
    let (cp, _rc) = out.device_ptr_mut(&dev.stream);
    unsafe { gemm_raw(&dev, true, n, m, k, scale, wp, k, n * k, xp, k, outer_stride_x, cp, n, m * n, batch) }
}

/// Chunked-attention variant of [`matmul_2d_strided_batched`]: `out` is a view
/// into a larger `[batch, outer_rows, n]` buffer, so a chunk's `P@V` lands in
/// place with no scatter copy.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
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
    use cudarc::driver::{DevicePtr, DevicePtrMut};
    let dev = global_device().ok_or_else(no_device)?;
    let out_required = strided_view_required_len(batch, outer_stride_c, m * n);
    size_check(
        "strided gemm (out-view)",
        a.len() == batch * m * k && b.len() == batch * k * n && out.len() >= out_required,
        || format!("a={} b={} out={} (need >= {out_required}) batch={batch} ({m},{k})@({k},{n})", a.len(), b.len(), out.len()),
    )?;
    let (bp, _rb) = b.device_ptr(&dev.stream);
    let (ap, _ra) = a.device_ptr(&dev.stream);
    let (cp, _rc) = out.device_ptr_mut(&dev.stream);
    unsafe { gemm_raw(&dev, false, n, m, k, 1.0, bp, n, k * n, ap, k, m * k, cp, n, outer_stride_c, batch) }
}

/// Attention `P [batch,m,k] @ V [batch,k,n]` with bfloat16 operands and an F32
/// result: the `P@V` half of dense SDPA once the probabilities are stored as
/// bf16.
#[cfg(feature = "cuda")]
pub fn matmul_2d_strided_batched_bf16(
    a: &cudarc::driver::CudaSlice<half::bf16>,
    b: &cudarc::driver::CudaSlice<half::bf16>,
    out: &mut cudarc::driver::CudaSlice<f32>,
    batch: usize,
    m: usize,
    k: usize,
    n: usize,
) -> Result<()> {
    use cudarc::driver::{DevicePtr, DevicePtrMut};
    let dev = global_device().ok_or_else(no_device)?;
    size_check(
        "strided gemm (bf16)",
        a.len() == batch * m * k && b.len() == batch * k * n && out.len() == batch * m * n,
        || format!("a={} b={} out={} batch={batch} ({m},{k})@({k},{n})", a.len(), b.len(), out.len()),
    )?;
    let (bp, _rb) = b.device_ptr(&dev.stream);
    let (ap, _ra) = a.device_ptr(&dev.stream);
    let (cp, _rc) = out.device_ptr_mut(&dev.stream);
    let bf = cudarc::cublas::sys::cudaDataType_t::CUDA_R_16BF;
    unsafe { gemm_raw_ty(&dev, false, n, m, k, 1.0, bp, n, k * n, ap, k, m * k, cp, n, m * n, batch, bf, None) }
}

/// [`matmul_2d_strided_batched_bf16`] writing into a strided view of a larger
/// output (the chunked attention path).
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub fn matmul_2d_strided_batched_out_view_bf16(
    a: &cudarc::driver::CudaSlice<half::bf16>,
    b: &cudarc::driver::CudaSlice<half::bf16>,
    out: &mut cudarc::driver::CudaViewMut<'_, f32>,
    outer_stride_c: usize,
    batch: usize,
    m: usize,
    k: usize,
    n: usize,
) -> Result<()> {
    use cudarc::driver::{DevicePtr, DevicePtrMut};
    let dev = global_device().ok_or_else(no_device)?;
    let out_required = strided_view_required_len(batch, outer_stride_c, m * n);
    size_check(
        "strided gemm (bf16, out-view)",
        a.len() == batch * m * k && b.len() == batch * k * n && out.len() >= out_required,
        || format!("a={} b={} out={} (need >= {out_required}) batch={batch} ({m},{k})@({k},{n})", a.len(), b.len(), out.len()),
    )?;
    let (bp, _rb) = b.device_ptr(&dev.stream);
    let (ap, _ra) = a.device_ptr(&dev.stream);
    let (cp, _rc) = out.device_ptr_mut(&dev.stream);
    let bf = cudarc::cublas::sys::cudaDataType_t::CUDA_R_16BF;
    unsafe { gemm_raw_ty(&dev, false, n, m, k, 1.0, bp, n, k * n, ap, k, m * k, cp, n, outer_stride_c, batch, bf, None) }
}

/// Resolve device from CLI spec (`cpu`, `cuda`, `cuda:0`).
pub fn resolve_device(spec: &str) -> Result<()> {
    let s = spec.trim().to_ascii_lowercase();
    #[cfg(feature = "cuda")]
    {
        if s == "cpu" {
            set_global(None);
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
        Err(DeviceError::Message(format!(
            "unknown device `{spec}` (expected cpu, cuda, or cuda:N)"
        )))
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

    /// Replays `attn::device_dense_sdpa`'s chunked loop on the index
    /// arithmetic only: every offset-view read of `Q` and write of `out` must
    /// stay inside the real buffer, for chunks that do and don't divide `sq`.
    fn assert_chunk_loop_stays_in_bounds(bh: usize, sq: usize, d: usize, chunk: usize) {
        let buf_len = bh * sq * d;
        let mut start = 0usize;
        let mut iterations = 0usize;
        while start < sq {
            let qlen = chunk.min(sq - start);
            assert!(qlen > 0);
            let required = strided_view_required_len(bh, sq * d, qlen * d);
            assert!(
                start * d + required <= buf_len,
                "view out of bounds: start={start} qlen={qlen} sq={sq} bh={bh} d={d}"
            );
            start += qlen;
            iterations += 1;
            assert!(iterations <= sq + 1, "loop did not terminate");
        }
        assert_eq!(start, sq);
    }

    #[test]
    fn chunk_evenly_divides_sequence() {
        assert_chunk_loop_stays_in_bounds(4, 1024, 64, 256);
    }

    #[test]
    fn chunk_does_not_evenly_divide_sequence() {
        assert_chunk_loop_stays_in_bounds(4, 1000, 64, 256);
    }

    #[test]
    fn chunk_larger_than_sequence_is_one_iteration() {
        assert_chunk_loop_stays_in_bounds(2, 100, 128, 1024);
    }

    #[test]
    fn many_batch_heads_small_chunk() {
        assert_chunk_loop_stays_in_bounds(40, 4097, 64, 64);
    }
}
