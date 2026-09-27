//! Datacenter tensor-core attention (`attn_dc.cu`): dense SDPA on tcgen05 +
//! tensor memory (sm_100, B200) and on wgmma (sm_90, H100 / H200), both fed
//! by TMA through warp-specialised pipelines.
//!
//! The kernels are arch-specific (`sm_100a` / `sm_90a`), so they live in
//! their own module, built by build.rs only for those targets and loaded
//! here only on a 10.0 or 9.0 device (NVRTC `compute_100a` / `compute_90a`
//! when no cubin was embedded). Every other GPU, and
//! `FASTVIDEO_FLASH_KERNEL=v2` (or `v1`) anywhere, keeps the mma.sync kernels
//! of kernels.cu. `fv-gpucheck kernels` group `attn_dc` holds these kernels to
//! `flash_mma_fwd2` and to an f32 reference.

use std::sync::{Arc, Mutex};

use cudarc::driver::{CudaFunction, CudaSlice, LaunchConfig};
use half::bf16;

use super::tensor::{Result, TensorError};

const SRC: &str = include_str!("attn_dc.cu");

fn msg(s: impl Into<String>) -> TensorError {
    TensorError::Message(s.into())
}

/// Head dim the kernels are built for.
pub const DC_HEAD_DIM: usize = 128;

/// One loaded dense kernel and its launch geometry.
pub struct DcDense {
    pub func: CudaFunction,
    /// Queries per CTA (grid.x = ceil(sq / rows)).
    pub rows: usize,
    pub threads: u32,
    pub smem: u32,
    /// `sm_major * 10 + sm_minor` it was loaded for.
    pub sm: u32,
    /// "cubin" or "nvrtc".
    pub origin: &'static str,
}

// Launch geometry, mirrored from attn_dc.cu (DC100_SMEM / DC90_SMEM, NS = 4).
const TILEB: u32 = 128 * 128 * 2;
const DC100_SMEM: u32 = 1024 + 2 * TILEB + 4 * TILEB + 256;
const DC90_SMEM: u32 = 1024 + TILEB + 4 * TILEB + 256;

type Slot = Option<(usize, Option<Arc<DcDense>>)>;
static LOADED: Mutex<Slot> = Mutex::new(None);

/// The dense kernel for the global device, loaded once per context; `None`
/// on any SM other than 9.0 / 10.0, or when the module fails to load (the
/// reason is logged once and the caller keeps the mma.sync kernel).
pub fn dense() -> Option<Arc<DcDense>> {
    let dev = super::device::global_device()?;
    let key = Arc::as_ptr(&dev.ctx) as usize;
    let mut slot = LOADED.lock().ok()?;
    if let Some((k, d)) = slot.as_ref() {
        if *k == key {
            return d.clone();
        }
    }
    let sm = (dev.sm_major * 10 + dev.sm_minor) as u32;
    let loaded = match load(&dev.ctx, sm) {
        Ok(d) => d.map(Arc::new),
        Err(e) => {
            eprintln!("attn_dc: sm{sm} kernels unavailable ({e}); using the mma.sync kernels");
            None
        }
    };
    *slot = Some((key, loaded.clone()));
    loaded
}

fn load(ctx: &Arc<cudarc::driver::CudaContext>, sm: u32) -> Result<Option<DcDense>> {
    let (entry, rows, smem, arch) = match sm {
        100 => ("fa_dc100_fwd_d128", 256, DC100_SMEM, "compute_100a"),
        90 => ("fa_dc90_fwd_d128", 128, DC90_SMEM, "compute_90a"),
        _ => return Ok(None),
    };
    let err = |e: cudarc::driver::DriverError| msg(e.to_string());
    let (module, origin) = if let Some(cubin) = super::kernels::dc_cubin(sm) {
        let path =
            std::env::temp_dir().join(format!("fv-attn-dc-{}-sm{sm}.cubin", std::process::id()));
        std::fs::write(&path, cubin).map_err(|e| msg(format!("write {}: {e}", path.display())))?;
        let m = ctx.load_module(cudarc::nvrtc::Ptx::from_file(&path));
        let _ = std::fs::remove_file(&path);
        (m.map_err(err)?, "cubin")
    } else {
        let opts = cudarc::nvrtc::CompileOptions {
            arch: Some(arch),
            use_fast_math: Some(false),
            ftz: Some(false),
            prec_div: Some(true),
            prec_sqrt: Some(true),
            fmad: Some(true),
            ..Default::default()
        };
        let ptx = cudarc::nvrtc::compile_ptx_with_opts(SRC, opts)
            .map_err(|e| msg(format!("nvrtc {arch}: {e}")))?;
        (ctx.load_module(ptx).map_err(err)?, "nvrtc")
    };
    let func = module.load_function(entry).map_err(err)?;
    super::ops::opt_in_dynamic_shared(&func, smem)?;
    Ok(Some(DcDense {
        func,
        rows,
        threads: 384,
        smem,
        sm,
        origin,
    }))
}

/// 128-byte tensormap passed by value (`__grid_constant__ DcTensorMap`).
#[repr(C, align(128))]
#[derive(Clone, Copy)]
struct DcTensorMap {
    opaque: [u64; 16],
}
unsafe impl cudarc::driver::DeviceRepr for DcTensorMap {}

/// 3-D map over a `[bh, s, 128]` bf16 tensor: dims (d, s, bh), box
/// (64, 128, 1), 128-byte swizzle; rows past `s` read as zero.
fn encode(ptr: cudarc::driver::sys::CUdeviceptr, s: usize, bh: usize) -> Result<DcTensorMap> {
    use cudarc::driver::sys::{
        self, CUtensorMapDataType, CUtensorMapFloatOOBfill, CUtensorMapInterleave,
        CUtensorMapL2promotion, CUtensorMapSwizzle,
    };
    let mut raw = std::mem::MaybeUninit::<sys::CUtensorMap>::zeroed();
    let dims = [128u64, s as u64, bh as u64];
    let strides = [256u64, s as u64 * 256];
    let boxd = [64u32, 128, 1];
    let es = [1u32, 1, 1];
    let st = unsafe {
        sys::cuTensorMapEncodeTiled(
            raw.as_mut_ptr(),
            CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
            3,
            ptr as *mut std::ffi::c_void,
            dims.as_ptr(),
            strides.as_ptr(),
            boxd.as_ptr(),
            es.as_ptr(),
            CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
            CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B,
            CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
            CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE,
        )
    };
    if st != sys::CUresult::CUDA_SUCCESS {
        return Err(msg(format!("attn_dc: cuTensorMapEncodeTiled: {st:?}")));
    }
    let map = unsafe { raw.assume_init() };
    Ok(unsafe { std::mem::transmute_copy(&map) })
}

/// Output of [`dense_fwd`].
pub enum DcOut<'a> {
    F32(&'a mut CudaSlice<f32>),
    Bf16(&'a mut CudaSlice<bf16>),
}

/// Dense SDPA `[bh, sq, 128] x [bh, sk, 128]` (bf16 in, f32 or bf16 out) on
/// the datacenter kernel. `sl2` = scale * log2(e). `Ok(false)` when no
/// kernel exists for this device or the shape is outside what it runs.
#[allow(clippy::too_many_arguments)]
pub fn dense_fwd(
    q: &CudaSlice<bf16>,
    k: &CudaSlice<bf16>,
    v: &CudaSlice<bf16>,
    out: DcOut<'_>,
    bh: usize,
    sq: usize,
    sk: usize,
    d: usize,
    sl2: f32,
) -> Result<bool> {
    use cudarc::driver::DevicePtr;
    if d != DC_HEAD_DIM
        || sq == 0
        || sk == 0
        || bh == 0
        || bh > 65_535
        || sq > i32::MAX as usize
        || sk > i32::MAX as usize
    {
        return Ok(false);
    }
    let Some(kern) = dense() else {
        return Ok(false);
    };
    let dev = super::device::global_device().ok_or_else(|| msg("attn_dc: no device"))?;
    let (qp, _gq) = q.device_ptr(&dev.stream);
    let (kp, _gk) = k.device_ptr(&dev.stream);
    let (vp, _gv) = v.device_ptr(&dev.stream);
    let (tq, tk, tv) = (
        encode(qp, sq, bh)?,
        encode(kp, sk, bh)?,
        encode(vp, sk, bh)?,
    );
    let cfg = LaunchConfig {
        grid_dim: (sq.div_ceil(kern.rows) as u32, bh as u32, 1),
        block_dim: (kern.threads, 1, 1),
        shared_mem_bytes: kern.smem,
    };
    let (sq_i, sk_i) = (sq as i32, sk as i32);
    let launch_err = |e: super::device::DeviceError| msg(e.to_string());
    let err = |e: cudarc::driver::DriverError| msg(e.to_string());
    // cudarc cannot pass a null pointer: the unused output is a 1-element dummy.
    match out {
        DcOut::F32(o) => {
            let mut dummy = unsafe { dev.stream.alloc::<bf16>(1) }.map_err(err)?;
            let is16 = 0i32;
            super::kernels::launch!(dev.stream, &kern.func, cfg;
                &tq, &tk, &tv, o, &mut dummy, &is16, &sq_i, &sk_i, &sl2)
            .map_err(launch_err)?;
        }
        DcOut::Bf16(o) => {
            let mut dummy = unsafe { dev.stream.alloc::<f32>(1) }.map_err(err)?;
            let is16 = 1i32;
            super::kernels::launch!(dev.stream, &kern.func, cfg;
                &tq, &tk, &tv, &mut dummy, o, &is16, &sq_i, &sk_i, &sl2)
            .map_err(launch_err)?;
        }
    }
    Ok(true)
}
