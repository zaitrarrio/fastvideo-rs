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

/// The nvcc / NVRTC arch of the attn_dc module for SM `sm`, or `None` when
/// this SM has none. 10.3 (B300) has an `sm_103a` cubin, but it was never
/// run on a B300, so it loads only with `FASTVIDEO_DC_SM103=1`.
fn dc_arch(sm: u32) -> Option<&'static str> {
    match sm {
        100 => Some("compute_100a"),
        90 => Some("compute_90a"),
        103 if super::envflag::bool_flag("FASTVIDEO_DC_SM103", false) => Some("compute_103a"),
        _ => None,
    }
}

/// Load attn_dc.cu for SM `sm`: the embedded cubin, else NVRTC.
fn load_module(
    ctx: &Arc<cudarc::driver::CudaContext>,
    sm: u32,
) -> Result<Option<(Arc<cudarc::driver::CudaModule>, &'static str)>> {
    let Some(arch) = dc_arch(sm) else {
        return Ok(None);
    };
    let err = |e: cudarc::driver::DriverError| msg(e.to_string());
    if let Some(cubin) = super::kernels::dc_cubin(sm) {
        let path =
            std::env::temp_dir().join(format!("fv-attn-dc-{}-sm{sm}.cubin", std::process::id()));
        std::fs::write(&path, cubin).map_err(|e| msg(format!("write {}: {e}", path.display())))?;
        let m = ctx.load_module(cudarc::nvrtc::Ptx::from_file(&path));
        let _ = std::fs::remove_file(&path);
        return Ok(Some((m.map_err(err)?, "cubin")));
    }
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
    Ok(Some((ctx.load_module(ptx).map_err(err)?, "nvrtc")))
}

fn load(ctx: &Arc<cudarc::driver::CudaContext>, sm: u32) -> Result<Option<DcDense>> {
    let (entry, rows, smem) = match sm {
        100 | 103 => ("fa_dc100_fwd_d128", 256, DC100_SMEM),
        90 => ("fa_dc90_fwd_d128", 128, DC90_SMEM),
        _ => return Ok(None),
    };
    let err = |e: cudarc::driver::DriverError| msg(e.to_string());
    let Some((module, origin)) = load_module(ctx, sm)? else {
        return Ok(None);
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
    encode_box(ptr, s, bh, 128)
}

/// [`encode`] with a (64, `box_rows`, 1) box.
fn encode_box(
    ptr: cudarc::driver::sys::CUdeviceptr,
    s: usize,
    bh: usize,
    box_rows: u32,
) -> Result<DcTensorMap> {
    use cudarc::driver::sys::{
        self, CUtensorMapDataType, CUtensorMapFloatOOBfill, CUtensorMapInterleave,
        CUtensorMapL2promotion, CUtensorMapSwizzle,
    };
    let mut raw = std::mem::MaybeUninit::<sys::CUtensorMap>::zeroed();
    let dims = [128u64, s as u64, bh as u64];
    let strides = [256u64, s as u64 * 256];
    let boxd = [64u32, box_rows, 1];
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

// ---------------------------------------------------------------- VSA (WP-D)

/// The VSA kernels of attn_dc.cu (KV-tile-list fine stage and fused prep)
/// for one device.
pub struct DcVsa {
    pub fine: CudaFunction,
    pub prep_f32: CudaFunction,
    pub prep_b16: CudaFunction,
    /// `sm_major * 10 + sm_minor` it was loaded for.
    pub sm: u32,
    /// "cubin" or "nvrtc".
    pub origin: &'static str,
}

// Mirrored from attn_dc.cu: DCV100_SMEM_FIXED(8), DCV90_SMEM(6).
const VSA_TILEB: u32 = 64 * 128 * 2;
const DCV100_SMEM_FIXED: u32 = 1024 + 2 * TILEB + 8 * VSA_TILEB + 256;
const DCV90_SMEM: u32 = 1024 + 2 * VSA_TILEB + 2 * 6 * VSA_TILEB + 256;
/// Opt-in dynamic shared memory limit on sm_90 / sm_100 (227 KB).
const MAX_DYN_SMEM: u32 = 232_448;
/// Entry words hold a 20-bit tile index.
const MAX_TILES: usize = 1 << 20;
const DCV_FUSE: i32 = 1;
const DCV_GATE: i32 = 2;
const DCV_GATE16: i32 = 4;

type VsaSlot = Option<(usize, Option<Arc<DcVsa>>)>;
static VSA_LOADED: Mutex<VsaSlot> = Mutex::new(None);

/// The VSA kernels for the global device, loaded once per context; `None`
/// off 9.0 / 10.0 or when the module fails to load (logged once; the caller
/// keeps the mma.sync kernels).
pub fn vsa() -> Option<Arc<DcVsa>> {
    let dev = super::device::global_device()?;
    let key = Arc::as_ptr(&dev.ctx) as usize;
    let mut slot = VSA_LOADED.lock().ok()?;
    if let Some((k, d)) = slot.as_ref() {
        if *k == key {
            return d.clone();
        }
    }
    let sm = (dev.sm_major * 10 + dev.sm_minor) as u32;
    let loaded = match load_vsa(&dev.ctx, sm) {
        Ok(d) => d.map(Arc::new),
        Err(e) => {
            eprintln!("attn_dc: sm{sm} VSA kernels unavailable ({e}); using the mma.sync kernels");
            None
        }
    };
    *slot = Some((key, loaded.clone()));
    loaded
}

fn load_vsa(ctx: &Arc<cudarc::driver::CudaContext>, sm: u32) -> Result<Option<DcVsa>> {
    let entry = match sm {
        100 | 103 => "fa_dc100_vsa",
        90 => "fa_dc90_vsa",
        _ => return Ok(None),
    };
    let err = |e: cudarc::driver::DriverError| msg(e.to_string());
    let Some((module, origin)) = load_module(ctx, sm)? else {
        return Ok(None);
    };
    Ok(Some(DcVsa {
        fine: module.load_function(entry).map_err(err)?,
        prep_f32: module.load_function("dcv_prep_f32").map_err(err)?,
        prep_b16: module.load_function("dcv_prep_b16").map_err(err)?,
        sm,
        origin,
    }))
}

/// Whether `auto` runs VSA on the datacenter kernels: the VSA kernel seam is
/// `auto` or `dc` and the module loaded (9.0 / 10.0). `mma` / `tma` / `tma2`
/// / `gather` keep the mma.sync kernels (the escape hatch); sm_120 and older
/// never load the module, so their path is unchanged by construction.
pub fn vsa_enabled() -> bool {
    use fastvideo_models::techniques::kernels::{choice, KernelOp};
    matches!(choice(KernelOp::VsaAttention).as_str(), "auto" | "" | "dc") && vsa().is_some()
}

/// The gate of H3's compression branch, in the dtype the caller holds.
pub enum VsaGate<'a> {
    None,
    F32(&'a CudaSlice<f32>),
    Bf16(&'a CudaSlice<bf16>),
}

/// Where the fine stage writes.
pub enum VsaEpilogue<'a> {
    /// Tile-slot-ordered f32 `[bh, num_tiles * 64, 128]`, the
    /// `vsa_mma_attn` contract (combined afterwards). Only tiles in the
    /// query range are written.
    Sparse(&'a mut CudaSlice<f32>),
    /// H3's combine, `bf16(bf16(sparse) + bf16(bf16(coarse) * gate))`,
    /// straight into token order `out[bh, seq, 128]` (f32): bit for bit
    /// `vsa_combine(round16)` / `vsa_combine_g16` over the `Sparse` output,
    /// for the query tiles in range (their real slots only).
    CombineRound16 {
        out: &'a mut CudaSlice<f32>,
        coarse: &'a CudaSlice<f32>,
        gate: VsaGate<'a>,
        seq: usize,
    },
}

/// VSA fine stage on the datacenter kernels over tile-ordered bf16 Q/K/V
/// (`[bh, num_tiles * 64, 128]`), query tiles `[q_base, q_base + q_tiles)`,
/// each attending its `topk` selected key tiles (`selected[bh][tile][topk]`).
/// `Ok(false)` when no kernel exists for this device or the shape is
/// outside what it runs (the caller keeps the mma.sync kernel).
#[allow(clippy::too_many_arguments)]
pub fn vsa_fine(
    qt: &CudaSlice<bf16>,
    kt: &CudaSlice<bf16>,
    vt: &CudaSlice<bf16>,
    selected: &CudaSlice<u32>,
    plan: &super::ops::VsaPlanDev,
    bh: usize,
    topk: usize,
    scale: f32,
    q_base: usize,
    q_tiles: usize,
    epi: VsaEpilogue<'_>,
) -> Result<bool> {
    use cudarc::driver::DevicePtr;
    let nt = plan.num_tiles;
    let padded = nt * 64;
    let q_end = q_base + q_tiles;
    if plan.tile_elems != 64
        || nt == 0
        || nt >= MAX_TILES
        || topk == 0
        || topk > nt
        || bh == 0
        || bh > 65_535
        || q_tiles == 0
        || q_end > nt
        || padded > i32::MAX as usize
        || qt.len() < bh * padded * DC_HEAD_DIM
        || kt.len() < bh * padded * DC_HEAD_DIM
        || vt.len() < bh * padded * DC_HEAD_DIM
        || selected.len() < bh * nt * topk
    {
        return Ok(false);
    }
    let Some(kern) = vsa() else {
        return Ok(false);
    };
    let smem = if kern.sm == 90 {
        DCV90_SMEM as usize
    } else {
        let words = nt.div_ceil(32);
        let cap = (2 * topk).min(nt);
        DCV100_SMEM_FIXED as usize + 4 * (2 * words + cap)
    };
    if smem > MAX_DYN_SMEM as usize {
        return Ok(false);
    }
    let smem = smem as u32;
    let dev = super::device::global_device().ok_or_else(|| msg("attn_dc: no device"))?;
    let err = |e: cudarc::driver::DriverError| msg(e.to_string());
    let (tq, tk, tv) = {
        let (qp, _gq) = qt.device_ptr(&dev.stream);
        let (kp, _gk) = kt.device_ptr(&dev.stream);
        let (vp, _gv) = vt.device_ptr(&dev.stream);
        (
            encode_box(qp, padded, bh, 64)?,
            encode_box(kp, padded, bh, 64)?,
            encode_box(vp, padded, bh, 64)?,
        )
    };
    super::ops::opt_in_dynamic_shared(&kern.fine, smem)?;
    let cfg = LaunchConfig {
        grid_dim: (q_tiles.div_ceil(2) as u32, bh as u32, 1),
        block_dim: (if kern.sm == 90 { 384 } else { 192 }, 1, 1),
        shared_mem_bytes: smem,
    };
    let (nt_i, topk_i, qb_i, qe_i) = (nt as i32, topk as i32, q_base as i32, q_end as i32);
    let sl2 = scale * std::f32::consts::LOG2_E;
    // cudarc cannot pass a null pointer: absent operands are 1-element dummies.
    let dummy32 = unsafe { dev.stream.alloc::<f32>(1) }.map_err(err)?;
    let dummy16 = unsafe { dev.stream.alloc::<bf16>(1) }.map_err(err)?;
    let launch_err = |e: super::device::DeviceError| msg(e.to_string());
    match epi {
        VsaEpilogue::Sparse(out) => {
            if out.len() < bh * padded * DC_HEAD_DIM {
                return Err(msg("attn_dc vsa: sparse output too small"));
            }
            let (seq_i, mode) = (0i64, 0i32);
            super::kernels::launch!(dev.stream, &kern.fine, cfg;
                &tq, &tk, &tv, selected, &plan.block_sizes, &plan.slot_src, out,
                &dummy32, &dummy32, &dummy16, &nt_i, &topk_i, &qb_i, &qe_i, &seq_i, &sl2, &mode)
            .map_err(launch_err)?;
        }
        VsaEpilogue::CombineRound16 { out, coarse, gate, seq } => {
            if out.len() < bh * seq * DC_HEAD_DIM || coarse.len() < bh * nt * DC_HEAD_DIM {
                return Err(msg("attn_dc vsa: combine buffers too small"));
            }
            let seq_i = seq as i64;
            match gate {
                VsaGate::None => {
                    let mode = DCV_FUSE;
                    super::kernels::launch!(dev.stream, &kern.fine, cfg;
                        &tq, &tk, &tv, selected, &plan.block_sizes, &plan.slot_src, out,
                        coarse, &dummy32, &dummy16, &nt_i, &topk_i, &qb_i, &qe_i, &seq_i, &sl2, &mode)
                    .map_err(launch_err)?;
                }
                VsaGate::F32(g) => {
                    let mode = DCV_FUSE | DCV_GATE;
                    super::kernels::launch!(dev.stream, &kern.fine, cfg;
                        &tq, &tk, &tv, selected, &plan.block_sizes, &plan.slot_src, out,
                        coarse, g, &dummy16, &nt_i, &topk_i, &qb_i, &qe_i, &seq_i, &sl2, &mode)
                    .map_err(launch_err)?;
                }
                VsaGate::Bf16(g) => {
                    let mode = DCV_FUSE | DCV_GATE | DCV_GATE16;
                    super::kernels::launch!(dev.stream, &kern.fine, cfg;
                        &tq, &tk, &tv, selected, &plan.block_sizes, &plan.slot_src, out,
                        coarse, &dummy32, g, &nt_i, &topk_i, &qb_i, &qe_i, &seq_i, &sl2, &mode)
                    .map_err(launch_err)?;
                }
            }
        }
    }
    Ok(true)
}

/// Output of [`vsa_prep`]: tile-ordered bf16 q/k/v and their tile means.
pub struct VsaPrep {
    pub qt: CudaSlice<bf16>,
    pub kt: CudaSlice<bf16>,
    pub vt: CudaSlice<bf16>,
    pub qc: CudaSlice<f32>,
    pub kc: CudaSlice<f32>,
    /// `None` unless asked for (only the gated compression branch reads it).
    pub vc: Option<CudaSlice<f32>>,
}

/// The inputs of [`vsa_prep`].
pub enum VsaPrepIn<'a> {
    /// f32 `[bh, seq, 128]`; `round16` pools bf16-rounded values (H3).
    F32 {
        q: &'a CudaSlice<f32>,
        k: &'a CudaSlice<f32>,
        v: &'a CudaSlice<f32>,
        round16: bool,
    },
    /// bf16 `[bh, seq, 128]` (pools the values as they are).
    Bf16 {
        q: &'a CudaSlice<bf16>,
        k: &'a CudaSlice<bf16>,
        v: &'a CudaSlice<bf16>,
    },
}

/// VSA prep in one launch: `vsa_tile_qkv` (tile-ordered bf16) and
/// `vsa_tile_mean` of q, k (and v when `want_vc`), reading each input once;
/// bit for bit those kernels' outputs. `Ok(None)` when no kernel exists for
/// this device or the shape is outside what it runs.
pub fn vsa_prep(
    input: VsaPrepIn<'_>,
    plan: &super::ops::VsaPlanDev,
    bh: usize,
    seq: usize,
    want_vc: bool,
) -> Result<Option<VsaPrep>> {
    let nt = plan.num_tiles;
    if plan.tile_elems != 64 || nt == 0 || bh == 0 || bh > 65_535 || nt > u32::MAX as usize {
        return Ok(None);
    }
    let n_in = bh * seq * DC_HEAD_DIM;
    let ok_len = match &input {
        VsaPrepIn::F32 { q, k, v, .. } => q.len() >= n_in && k.len() >= n_in && v.len() >= n_in,
        VsaPrepIn::Bf16 { q, k, v } => q.len() >= n_in && k.len() >= n_in && v.len() >= n_in,
    };
    if !ok_len {
        return Ok(None);
    }
    let Some(kern) = vsa() else {
        return Ok(None);
    };
    let dev = super::device::global_device().ok_or_else(|| msg("attn_dc: no device"))?;
    let err = |e: cudarc::driver::DriverError| msg(e.to_string());
    let tiled = bh * nt * 64 * DC_HEAD_DIM;
    let means = bh * nt * DC_HEAD_DIM;
    let mut qt = unsafe { dev.stream.alloc::<bf16>(tiled) }.map_err(err)?;
    let mut kt = unsafe { dev.stream.alloc::<bf16>(tiled) }.map_err(err)?;
    let mut vt = unsafe { dev.stream.alloc::<bf16>(tiled) }.map_err(err)?;
    let mut qc = unsafe { dev.stream.alloc::<f32>(means) }.map_err(err)?;
    let mut kc = unsafe { dev.stream.alloc::<f32>(means) }.map_err(err)?;
    let mut vc = unsafe { dev.stream.alloc::<f32>(if want_vc { means } else { 1 }) }.map_err(err)?;
    let cfg = LaunchConfig {
        grid_dim: (nt as u32, bh as u32, 3),
        block_dim: (DC_HEAD_DIM as u32, 1, 1),
        shared_mem_bytes: 0,
    };
    let (seq_i, nt_i) = (seq as i64, nt as i32);
    let mask = if want_vc { 7i32 } else { 3 };
    let launch_err = |e: super::device::DeviceError| msg(e.to_string());
    match input {
        VsaPrepIn::F32 { q, k, v, round16 } => {
            let r16 = i32::from(round16);
            super::kernels::launch!(dev.stream, &kern.prep_f32, cfg;
                q, k, v, &plan.slot_src, &plan.block_sizes, &mut qt, &mut kt, &mut vt,
                &mut qc, &mut kc, &mut vc, &seq_i, &nt_i, &mask, &r16)
            .map_err(launch_err)?;
        }
        VsaPrepIn::Bf16 { q, k, v } => {
            super::kernels::launch!(dev.stream, &kern.prep_b16, cfg;
                q, k, v, &plan.slot_src, &plan.block_sizes, &mut qt, &mut kt, &mut vt,
                &mut qc, &mut kc, &mut vc, &seq_i, &nt_i, &mask)
            .map_err(launch_err)?;
        }
    }
    Ok(Some(VsaPrep {
        qt,
        kt,
        vt,
        qc,
        kc,
        vc: want_vc.then_some(vc),
    }))
}
