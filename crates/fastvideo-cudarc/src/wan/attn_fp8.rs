//! SageAttention-style FP8 attention (opt-in, lossy): `Q K^T` on FP8 tensor
//! cores, `P V` in bf16. Kernels: `attn_fp8.cu` (its own NVRTC module,
//! compiled on first use, so nothing here costs anything unless asked for).
//!
//! Recipe (Zhang et al., *SageAttention*, 2024, with FP8 in place of INT8):
//!
//! 1. **Smooth K**: subtract each head's column mean. Exact for softmax
//!    (every score of a query row shifts by `q . mean`), and it removes the
//!    shared channel offsets that would otherwise set K's quantization range.
//! 2. **Quantize** Q per 16 rows (one scale per MMA warp) and K per 64-key
//!    tile to E4M3 (`amax / 448` scales, RNE, saturating).
//! 3. **S = Q8 K8^T** with `mma.sync m16n8k32 e4m3` (sm_89+; f32 accumulate),
//!    dequantized by `q_scale * k_scale`; the online softmax, bf16 `P` and
//!    `P V` are the bf16 flash kernel's.
//!
//! Dispatch: [`set_ops`] (the H3 pipeline sets it per request from its
//! technique set: `fp8_attention` / `FASTVIDEO_ATTN_FP8`). Off is the
//! default and then no call here happens. [`dense_sdpa`] and
//! [`vsa_fine`] return `None` / an error where the device cannot run them
//! (pre-sm_89, head dim other than 128), and the callers fall back to bf16.

use std::sync::atomic::{AtomicU8, Ordering};

static OPS: AtomicU8 = AtomicU8::new(0);
const DENSE: u8 = 1;
const VSA: u8 = 2;

/// Where FP8 attention runs from now on (process-wide; the H3 pipeline sets
/// it at the start of each request).
pub fn set_ops(dense: bool, vsa: bool) {
    OPS.store(
        if dense { DENSE } else { 0 } | if vsa { VSA } else { 0 },
        Ordering::Relaxed,
    );
}

pub fn dense_enabled() -> bool {
    OPS.load(Ordering::Relaxed) & DENSE != 0
}

pub fn vsa_enabled() -> bool {
    OPS.load(Ordering::Relaxed) & VSA != 0
}

/// Head dim the kernels are written for.
pub const HEAD_DIM: usize = 128;

#[cfg(feature = "cuda")]
pub use imp::*;
// `launch!` names `super::stats` and `super::device` from its call site.
#[cfg(feature = "cuda")]
use super::{device, stats};

#[cfg(feature = "cuda")]
mod imp {
    use std::sync::Arc;

    use cudarc::driver::{CudaFunction, CudaSlice, LaunchConfig};

    use super::HEAD_DIM;
    use crate::wan::device::{self, DeviceContext};
    use crate::wan::kernels::launch;
    use crate::wan::ops::VsaPlanDev;
    use crate::wan::tensor::{CudaTensor, Result, TensorError};

    const SRC: &str = include_str!("attn_fp8.cu");
    /// Two (K8, V) stages: 2 x (8 KB + 16 KB).
    const SMEM: u32 = 2 * (64 * 128 + 64 * 128 * 2);

    fn err(e: impl std::fmt::Display) -> TensorError {
        TensorError::Message(format!("attn_fp8: {e}"))
    }

    fn ctx() -> Result<Arc<DeviceContext>> {
        device::global_device().ok_or_else(|| err("no global CUDA device"))
    }

    /// FP8 `mma.sync` (e4m3) needs sm_89 or newer.
    pub fn supported() -> bool {
        device::global_device().is_some_and(|d| (d.sm_major, d.sm_minor) >= (8, 9))
    }

    struct Kernels {
        _module: Arc<cudarc::driver::CudaModule>,
        colsum_bf16: CudaFunction,
        colsum_f32: CudaFunction,
        quant_bf16: CudaFunction,
        quant_tile_f32: CudaFunction,
        fwd_d128: CudaFunction,
        vsa: CudaFunction,
    }

    fn kernels() -> Result<std::rc::Rc<Kernels>> {
        thread_local! {
            static CELL: std::cell::RefCell<Option<std::rc::Rc<Kernels>>> =
                const { std::cell::RefCell::new(None) };
        }
        CELL.with(|c| {
            if let Some(k) = c.borrow().as_ref() {
                return Ok(k.clone());
            }
            let loaded = std::rc::Rc::new(load()?);
            *c.borrow_mut() = Some(loaded.clone());
            Ok(loaded)
        })
    }

    fn load() -> Result<Kernels> {
        use cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions};
        let dev = ctx()?;
        let timer = std::time::Instant::now();
        let mut last = None;
        let mut ptx = None;
        for arch in crate::wan::hopper::nvrtc_arches(dev.sm_major, dev.sm_minor) {
            let opts = CompileOptions {
                arch: Some(arch),
                use_fast_math: Some(true),
                ftz: Some(true),
                options: vec!["-std=c++17".into()],
                ..Default::default()
            };
            match compile_ptx_with_opts(SRC, opts) {
                Ok(p) => {
                    ptx = Some(p);
                    break;
                }
                Err(e) => last = Some(format!("arch={arch}: {e}")),
            }
        }
        let ptx = ptx.ok_or_else(|| {
            err(format!(
                "nvrtc: {}",
                last.unwrap_or_else(|| "no candidate arch".into())
            ))
        })?;
        let module = dev.ctx.load_module(ptx).map_err(err)?;
        let f = |name: &str| module.load_function(name).map_err(err);
        let k = Kernels {
            colsum_bf16: f("attn_fp8_colsum_bf16")?,
            colsum_f32: f("attn_fp8_colsum_f32")?,
            quant_bf16: f("attn_fp8_quant_bf16")?,
            quant_tile_f32: f("attn_fp8_quant_tile_f32")?,
            fwd_d128: f("attn_fp8_fwd_d128")?,
            vsa: f("attn_fp8_vsa")?,
            _module: module,
        };
        crate::wan::ops::opt_in_dynamic_shared(&k.fwd_d128, SMEM)?;
        crate::wan::ops::opt_in_dynamic_shared(&k.vsa, SMEM)?;
        crate::wan::log::info(format_args!(
            "attn_fp8: kernels compiled ({:.1}s, sm{}{})",
            timer.elapsed().as_secs_f64(),
            dev.sm_major,
            dev.sm_minor
        ));
        Ok(k)
    }

    fn zeros_f32(n: usize) -> Result<CudaSlice<f32>> {
        ctx()?.stream.alloc_zeros::<f32>(n.max(1)).map_err(err)
    }

    fn alloc_u8(n: usize) -> Result<CudaSlice<u8>> {
        let dev = ctx()?;
        unsafe { dev.stream.alloc::<u8>(n.max(1)) }.map_err(err)
    }

    fn alloc_f32(n: usize) -> Result<CudaSlice<f32>> {
        let dev = ctx()?;
        unsafe { dev.stream.alloc::<f32>(n.max(1)) }.map_err(err)
    }

    /// Rows per colsum block (atomics per (head, dim) = rows / this).
    const COLSUM_ROWS: usize = 256;

    /// E4M3 rows of a bf16 `[bh, rows, 128]` tensor, padded to `rows_pad`
    /// (a multiple of 64), with scales per `grp` rows; `smooth` subtracts the
    /// per-head column mean first.
    fn quant_bf16(
        k: &Kernels,
        x: &CudaSlice<half::bf16>,
        bh: usize,
        rows: usize,
        rows_pad: usize,
        grp: usize,
        smooth: bool,
    ) -> Result<(CudaSlice<u8>, CudaSlice<f32>)> {
        let dev = ctx()?;
        let mut sum = zeros_f32(bh * HEAD_DIM)?;
        let (rows_i, rpb) = (rows as i32, COLSUM_ROWS as i32);
        if smooth {
            let cfg = LaunchConfig {
                grid_dim: (rows.div_ceil(COLSUM_ROWS) as u32, bh as u32, 1),
                block_dim: (HEAD_DIM as u32, 1, 1),
                shared_mem_bytes: 0,
            };
            launch!(dev.stream, &k.colsum_bf16, cfg; x, &mut sum, &rows_i, &rpb).map_err(err)?;
        }
        let mut out = alloc_u8(bh * rows_pad * HEAD_DIM)?;
        let mut scales = alloc_f32(bh * rows_pad / grp)?;
        let cfg = LaunchConfig {
            grid_dim: ((rows_pad / 64) as u32, bh as u32, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 0,
        };
        let inv = 1.0f32 / rows.max(1) as f32;
        let (sm, pad_i, grp_i) = (i32::from(smooth), rows_pad as i32, grp as i32);
        launch!(dev.stream, &k.quant_bf16, cfg;
            x, &sum, &inv, &sm, &mut out, &mut scales, &rows_i, &pad_i, &grp_i)
        .map_err(err)?;
        Ok((out, scales))
    }

    /// Dense SDPA with FP8 `Q K^T`: `q [b, h, sq, 128]`, `k` / `v`
    /// `[b, h, sk, 128]`. Output f32, or bf16 when `out_bf16`. `None` when
    /// the device or shape is not one the kernel runs.
    pub fn dense_sdpa(
        q: &CudaTensor,
        k: &CudaTensor,
        v: &CudaTensor,
        scale: Option<f32>,
        out_bf16: bool,
    ) -> Result<Option<CudaTensor>> {
        let (Ok([b, h, sq, d]), Ok([kb, kh, sk, kd])) = (
            <[usize; 4]>::try_from(q.shape.as_slice()),
            <[usize; 4]>::try_from(k.shape.as_slice()),
        ) else {
            return Ok(None);
        };
        if !supported()
            || d != HEAD_DIM
            || (kb, kh, kd) != (b, h, d)
            || v.shape != k.shape
            || sq == 0
            || sk == 0
            || sq > i32::MAX as usize / 2
            || sk > i32::MAX as usize / 2
        {
            return Ok(None);
        }
        let (Some(qb), Some(kbf), Some(vb)) = (q.dev_bf16()?, k.dev_bf16()?, v.dev_bf16()?) else {
            return Ok(None);
        };
        let kern = kernels()?;
        let dev = ctx()?;
        let bh = b * h;
        let (sq_pad, sk_pad) = (sq.div_ceil(128) * 128, sk.div_ceil(64) * 64);
        let (q8, qs) = quant_bf16(&kern, &qb, bh, sq, sq_pad, 16, false)?;
        let (k8, ks) = quant_bf16(&kern, &kbf, bh, sk, sk_pad, 64, true)?;
        let scale = scale.unwrap_or(1.0 / (d as f32).sqrt());
        let sl2 = scale * std::f32::consts::LOG2_E;
        static ONCE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        crate::wan::log::info_once(
            &ONCE,
            format_args!("sdpa: FP8 Q K^T (attn_fp8) B={b} H={h} Sq={sq} Sk={sk} D={d}"),
        );
        let cfg = LaunchConfig {
            grid_dim: ((sq_pad / 128) as u32, bh as u32, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: SMEM,
        };
        let n = bh * sq * d;
        let (sq_i, sk_i, sqp_i, skp_i) = (sq as i32, sk as i32, sq_pad as i32, sk_pad as i32);
        if out_bf16 {
            let mut out = unsafe { dev.stream.alloc::<half::bf16>(n) }.map_err(err)?;
            let mut dummy = alloc_f32(1)?;
            let one = 1i32;
            launch!(dev.stream, &kern.fwd_d128, cfg;
                &q8, &qs, &k8, &ks, &*vb, &mut dummy, &mut out, &one, &sq_i, &sk_i, &sqp_i, &skp_i, &sl2)
            .map_err(err)?;
            Ok(Some(CudaTensor::from_device_slice_bf16(
                out,
                vec![b, h, sq, d],
            )?))
        } else {
            let mut out = alloc_f32(n)?;
            let mut dummy = unsafe { dev.stream.alloc::<half::bf16>(1) }.map_err(err)?;
            let zero = 0i32;
            launch!(dev.stream, &kern.fwd_d128, cfg;
                &q8, &qs, &k8, &ks, &*vb, &mut out, &mut dummy, &zero, &sq_i, &sk_i, &sqp_i, &skp_i, &sl2)
            .map_err(err)?;
            Ok(Some(CudaTensor::from_device_slice(out, vec![b, h, sq, d])?))
        }
    }

    /// The VSA fine stage with FP8 `Q K^T`: the contract of
    /// [`crate::wan::ops::vsa_mma_attn_range_device`] (f32 `[bh, seq, 128]`
    /// q/k/v in packed order, `selected` key tiles per query tile, f32 output
    /// `[bh, padded, 128]` in tile-slot order, query tiles
    /// `[q_base, q_base + q_tiles)`; the rest zero).
    #[allow(clippy::too_many_arguments)]
    pub fn vsa_fine(
        q: &CudaSlice<f32>,
        k: &CudaSlice<f32>,
        v: &CudaSlice<f32>,
        selected: &CudaSlice<u32>,
        plan: &VsaPlanDev,
        bh: usize,
        seq: usize,
        dim: usize,
        topk: usize,
        scale: f32,
        q_base: usize,
        q_tiles: usize,
    ) -> Result<CudaSlice<f32>> {
        if !supported() || dim != HEAD_DIM || plan.tile_elems != 64 {
            return Err(err(format!(
                "vsa_fine needs sm_89+, head dim 128 and 64-slot tiles (dim {dim}, tile {})",
                plan.tile_elems
            )));
        }
        let kern = kernels()?;
        let dev = ctx()?;
        let nb = plan.num_tiles;
        let q_tiles = q_tiles.min(nb.saturating_sub(q_base));
        if q_tiles == 0 || topk == 0 {
            return Err(err("vsa_fine: empty query range or top-k"));
        }
        let padded = nb * 64;
        let seq_i = seq as i32;
        let pad_i = padded as i32;
        // K: column sums over the live rows, then tile-slot quantization.
        let mut sum = zeros_f32(bh * HEAD_DIM)?;
        let rpb = COLSUM_ROWS as i32;
        let cfg = LaunchConfig {
            grid_dim: (seq.div_ceil(COLSUM_ROWS) as u32, bh as u32, 1),
            block_dim: (HEAD_DIM as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        launch!(dev.stream, &kern.colsum_f32, cfg; k, &mut sum, &seq_i, &rpb).map_err(err)?;
        let quant = |x: &CudaSlice<f32>,
                     smooth: bool,
                     grp: usize|
         -> Result<(CudaSlice<u8>, CudaSlice<f32>)> {
            let mut out = alloc_u8(bh * padded * HEAD_DIM)?;
            let mut scales = alloc_f32(bh * padded / grp)?;
            let cfg = LaunchConfig {
                grid_dim: (nb as u32, bh as u32, 1),
                block_dim: (128, 1, 1),
                shared_mem_bytes: 0,
            };
            let inv = 1.0f32 / seq.max(1) as f32;
            let (sm, grp_i) = (i32::from(smooth), grp as i32);
            launch!(dev.stream, &kern.quant_tile_f32, cfg;
                x, &plan.slot_src, &sum, &inv, &sm, &mut out, &mut scales, &seq_i, &pad_i, &grp_i)
            .map_err(err)?;
            Ok((out, scales))
        };
        let (q8, qs) = quant(q, false, 16)?;
        let (k8, ks) = quant(k, true, 64)?;
        let vt = crate::wan::ops::vsa_tile_qkv_device(v, plan, bh, seq, dim)?;
        let mut out = if q_base == 0 && q_tiles == nb {
            alloc_f32(bh * padded * dim)?
        } else {
            crate::wan::ops::fill_device(bh * padded * dim, 0.0)?
        };
        static ONCE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        crate::wan::log::info_once(
            &ONCE,
            format_args!("vsa fine kernel: FP8 Q K^T (attn_fp8_vsa), {nb} tiles, top-k {topk}"),
        );
        let cfg = LaunchConfig {
            grid_dim: (q_tiles as u32, bh as u32, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: SMEM,
        };
        let (nt, tk, qb) = (nb as i32, topk as i32, q_base as i32);
        let sl2 = scale * std::f32::consts::LOG2_E;
        launch!(dev.stream, &kern.vsa, cfg;
            &q8, &qs, &k8, &ks, &vt, selected, &plan.block_sizes, &mut out, &nt, &tk, &sl2, &qb)
        .map_err(err)?;
        Ok(out)
    }
}
