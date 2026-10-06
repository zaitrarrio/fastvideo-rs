//! SageAttention2-style dense attention (opt-in, lossy): INT8 `Q K^T` and
//! FP8 `P V` on `mma.sync`, f32 accumulation. Kernels: `attn_sage.cu` (its
//! own module: ahead-of-time cubins for sm_89+, NVRTC otherwise), loaded only
//! when asked for.
//!
//! Recipe (Zhang et al., *SageAttention2*): K smoothed by its per-head column
//! mean (exact under softmax), Q quantized to INT8 per 16 rows and K per
//! 64-key tile (amax / 127); `P` to E4M3 with the fixed scale 448 and V to
//! E4M3 per channel (amax / 448), V stored transposed with keys permuted to
//! the k32 MMA's operand order. docs/perf/sage-attention.md has the upstream
//! comparison and the accuracy / speed numbers.
//!
//! Routing: the dense self-attention of the DiTs
//! (`nn::scaled_dot_product_attention`: H3, LTX-2.5, Wan) goes through
//! [`dense_sdpa`] when it is enabled (below), the head dim is 128, the device
//! is sm_89+ and both sequences are at least `FASTVIDEO_ATTN_SAGE_MIN_SEQ`
//! (default 6144: below that the bf16 kernels are as fast).
//!
//! Enabled, highest precedence first:
//!
//! 1. `FASTVIDEO_ATTN_SAGE=2`: on for every model (any sm_89+ device);
//!    `FASTVIDEO_ATTN_SAGE=0`: off for every model, recipes included. `=3`
//!    (SageAttention3, NVFP4) is refused: it failed the accuracy tolerance
//!    (docs/perf/sage-attention.md).
//! 2. Unset: off, except inside a [`recipe_scope`] a recipe opened with
//!    `true` on a device where the kernel pays ([`arch_pays`]: sm_120, RTX
//!    PRO 6000 / RTX 5090). Only the `ltx-pro` model (`ltx25-distill-dense`,
//!    `Ltx2Recipe::sage_attention`) opens one: the one recipe that passed
//!    the end-to-end gate (docs/perf/sage-attention.md, Phase 3).

/// Head dim the kernels are written for.
pub const HEAD_DIM: usize = 128;

/// The `FASTVIDEO_ATTN_SAGE` override: `Some(2)` (on everywhere), `Some(0)`
/// (off everywhere) or `None` (unset: recipes decide). An unknown value is
/// off with a one-time warning.
pub fn forced() -> Option<u8> {
    static MODE: std::sync::OnceLock<Option<u8>> = std::sync::OnceLock::new();
    *MODE.get_or_init(|| {
        let v = fastvideo_models::techniques::settings::var("FASTVIDEO_ATTN_SAGE")?;
        Some(match v.trim().to_ascii_lowercase().as_str() {
            "" => return None,
            "0" | "off" => 0,
            "2" | "on" => 2,
            other => {
                crate::wan::log::info(format_args!(
                    "FASTVIDEO_ATTN_SAGE={other}: only 2 (INT8 QK + FP8 PV) is implemented; \
                     3 (NVFP4) failed the accuracy tolerance. Sage attention is off."
                ));
                0
            }
        })
    })
}

/// Whether a recipe's Sage enablement applies on `(sm_major, sm_minor)`:
/// sm_120 only (RTX PRO 6000, RTX 5090). On sm_90 / sm_100 the bf16 kernels
/// (cuDNN, `attn_dc`) are close enough that the lossy kernel does not pay
/// (docs/perf/sage-attention.md, H100 section).
pub fn arch_pays(sm_major: i32, sm_minor: i32) -> bool {
    (sm_major, sm_minor) == (12, 0)
}

/// The routing decision, without the device: `forced` is [`forced`], `recipe`
/// whether a [`recipe_scope`] asked for it, `sm` the device's compute
/// capability. The kernel itself also needs sm_89+.
pub fn decide(forced: Option<u8>, recipe: bool, sm: (i32, i32)) -> bool {
    if sm < (8, 9) {
        return false;
    }
    match forced {
        Some(m) => m == 2,
        None => recipe && arch_pays(sm.0, sm.1),
    }
}

thread_local! {
    static RECIPE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Whether the calling thread is inside a [`recipe_scope`] that asked for Sage.
pub fn recipe_wants() -> bool {
    RECIPE.with(|r| r.get())
}

/// Restores the previous recipe setting when dropped.
#[must_use = "the recipe setting lasts as long as the guard"]
pub struct RecipeScope {
    prev: bool,
}

impl Drop for RecipeScope {
    fn drop(&mut self) {
        RECIPE.with(|r| r.set(self.prev));
    }
}

/// A recipe's per-request enablement, for the calling thread (the executor
/// thread that owns the CUDA device and runs the DiT) until the guard drops.
/// `FASTVIDEO_ATTN_SAGE` still overrides it, and it only takes effect on
/// [`arch_pays`] devices.
pub fn recipe_scope(on: bool) -> RecipeScope {
    RecipeScope {
        prev: RECIPE.with(|r| r.replace(on)),
    }
}

/// Smallest `Sq` and `Sk` routed to the kernel.
pub fn min_seq() -> usize {
    static MIN: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *MIN.get_or_init(|| crate::wan::envflag::usize_flag("FASTVIDEO_ATTN_SAGE_MIN_SEQ", 6144))
}

#[cfg(feature = "cuda")]
pub use imp::*;

/// Without CUDA nothing routes here.
#[cfg(not(feature = "cuda"))]
pub fn dense_enabled() -> bool {
    false
}

#[cfg(not(feature = "cuda"))]
pub fn dense_sdpa(
    _q: &super::tensor::CudaTensor,
    _k: &super::tensor::CudaTensor,
    _v: &super::tensor::CudaTensor,
    _scale: Option<f32>,
    _out_bf16: bool,
) -> super::tensor::Result<Option<super::tensor::CudaTensor>> {
    Ok(None)
}
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
    use crate::wan::tensor::{CudaTensor, Result, TensorError};

    const SRC: &str = include_str!("attn_sage.cu");
    /// Two (K8, V8^T) stages: 2 x (8 KB + 8 KB); Q (16 KB) is staged in one.
    const SMEM: u32 = 2 * (64 * 128 + 128 * 64);
    const COLSUM_ROWS: usize = 256;

    fn err(e: impl std::fmt::Display) -> TensorError {
        TensorError::Message(format!("attn_sage: {e}"))
    }

    fn ctx() -> Result<Arc<DeviceContext>> {
        device::global_device().ok_or_else(|| err("no global CUDA device"))
    }

    /// INT8 and FP8 `mma.sync` need sm_89 or newer.
    pub fn supported() -> bool {
        device::global_device().is_some_and(|d| (d.sm_major, d.sm_minor) >= (8, 9))
    }

    /// Whether dense DiT attention goes through [`dense_sdpa`] ([`super::decide`]).
    pub fn dense_enabled() -> bool {
        let forced = super::forced();
        if forced == Some(0) || (forced.is_none() && !super::recipe_wants()) {
            return false;
        }
        device::global_device()
            .is_some_and(|d| super::decide(forced, super::recipe_wants(), (d.sm_major, d.sm_minor)))
    }

    struct Kernels {
        _module: Arc<cudarc::driver::CudaModule>,
        colsum: CudaFunction,
        colsum_reduce: CudaFunction,
        quant: CudaFunction,
        vmax: CudaFunction,
        vquant: CudaFunction,
        fwd_d128: CudaFunction,
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

    mod aot {
        include!(concat!(env!("OUT_DIR"), "/aot_attn_sage.rs"));
    }

    /// The embedded module for this SM: its cubin, else a same-major PTX at
    /// or below it (driver JIT); NVRTC last (see `attn_fp8::load_embedded`).
    fn load_embedded(
        dev: &DeviceContext,
    ) -> Result<Option<(Arc<cudarc::driver::CudaModule>, String)>> {
        let want = (dev.sm_major * 10 + dev.sm_minor) as u32;
        if crate::wan::envflag::string_flag("FASTVIDEO_KERNELS", "auto") == "nvrtc" {
            return Ok(None);
        }
        if let Some((_, cubin, _)) = aot::AOT_ATTN_SAGE.iter().find(|(sm, _, _)| *sm == want) {
            let path = std::env::temp_dir().join(format!(
                "fv-attn-sage-{}-sm{want}.cubin",
                std::process::id()
            ));
            std::fs::write(&path, cubin).map_err(err)?;
            let loaded = dev.ctx.load_module(cudarc::nvrtc::Ptx::from_file(&path));
            let _ = std::fs::remove_file(&path);
            return Ok(Some((loaded.map_err(err)?, format!("cubin sm{want}"))));
        }
        if let Some((sm, _, ptx)) = aot::AOT_ATTN_SAGE
            .iter()
            .filter(|(sm, _, _)| sm / 10 == want / 10 && *sm <= want)
            .max_by_key(|(sm, _, _)| *sm)
        {
            let module = dev
                .ctx
                .load_module(cudarc::nvrtc::Ptx::from_src(*ptx))
                .map_err(err)?;
            return Ok(Some((module, format!("ptx compute{sm}"))));
        }
        Ok(None)
    }

    fn load() -> Result<Kernels> {
        use cudarc::nvrtc::CompileOptions;
        let dev = ctx()?;
        let timer = std::time::Instant::now();
        let (module, origin) = if let Some(m) = load_embedded(&dev)? {
            m
        } else {
            // SASS first (`nvrtc_sass`): NVRTC PTX newer than the driver
            // does not load.
            let opts = CompileOptions {
                use_fast_math: Some(true),
                ftz: Some(true),
                options: vec!["-std=c++17".into()],
                ..Default::default()
            };
            let (module, origin) = crate::wan::nvrtc_sass::load_for_device(
                &dev.ctx,
                dev.sm_major,
                dev.sm_minor,
                SRC,
                &opts,
                "attn-sage",
            )
            .map_err(err)?;
            (module, origin.to_string())
        };
        let f = |name: &str| module.load_function(name).map_err(err);
        let k = Kernels {
            colsum: f("attn_sage_colsum_bf16")?,
            colsum_reduce: f("attn_sage_colsum_reduce")?,
            quant: f("attn_sage_quant_i8")?,
            vmax: f("attn_sage_vmax")?,
            vquant: f("attn_sage_vquant")?,
            fwd_d128: f("attn_sage_fwd_d128")?,
            _module: module,
        };
        crate::wan::ops::opt_in_dynamic_shared(&k.fwd_d128, SMEM)?;
        crate::wan::log::info(format_args!(
            "attn_sage: kernels loaded ({origin}, {:.1}s, sm{}{})",
            timer.elapsed().as_secs_f64(),
            dev.sm_major,
            dev.sm_minor
        ));
        Ok(k)
    }

    fn alloc<T: cudarc::driver::DeviceRepr>(n: usize) -> Result<CudaSlice<T>> {
        let dev = ctx()?;
        unsafe { dev.stream.alloc::<T>(n.max(1)) }.map_err(err)
    }

    fn zeros<T: cudarc::driver::DeviceRepr + cudarc::driver::ValidAsZeroBits>(
        n: usize,
    ) -> Result<CudaSlice<T>> {
        ctx()?.stream.alloc_zeros::<T>(n.max(1)).map_err(err)
    }

    /// bf16 `[bh, rows, 128]` -> int8 `[bh, rows_pad, 128]` + scales
    /// `[bh, rows_pad / grp]`; `smooth` subtracts the column mean first.
    fn quant_i8(
        k: &Kernels,
        x: &CudaSlice<half::bf16>,
        bh: usize,
        rows: usize,
        rows_pad: usize,
        grp: usize,
        smooth: bool,
    ) -> Result<(CudaSlice<u8>, CudaSlice<f32>)> {
        let dev = ctx()?;
        let mut sum = zeros::<f32>(bh * HEAD_DIM)?;
        let (rows_i, rpb) = (rows as i32, COLSUM_ROWS as i32);
        if smooth {
            // One partial per block, then the partials added in block order:
            // the same bits in every run (no float atomics).
            let nblk = rows.div_ceil(COLSUM_ROWS).max(1);
            let n = bh * HEAD_DIM;
            let mut part = alloc::<f32>(nblk * n)?;
            let cfg = LaunchConfig {
                grid_dim: (nblk as u32, bh as u32, 1),
                block_dim: (HEAD_DIM as u32, 1, 1),
                shared_mem_bytes: 0,
            };
            launch!(dev.stream, &k.colsum, cfg; x, &mut part, &rows_i, &rpb).map_err(err)?;
            let (nblk_i, n_i) = (nblk as i32, n as i32);
            let cfg = LaunchConfig {
                grid_dim: (n.div_ceil(256) as u32, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            };
            launch!(dev.stream, &k.colsum_reduce, cfg; &part, &mut sum, &nblk_i, &n_i)
                .map_err(err)?;
        }
        let mut out = alloc::<u8>(bh * rows_pad * HEAD_DIM)?;
        let mut scales = alloc::<f32>(bh * rows_pad / grp)?;
        let cfg = LaunchConfig {
            grid_dim: ((rows_pad / 64) as u32, bh as u32, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 0,
        };
        let inv = 1.0f32 / rows.max(1) as f32;
        let (sm, pad_i, grp_i) = (i32::from(smooth), rows_pad as i32, grp as i32);
        launch!(dev.stream, &k.quant, cfg;
            x, &sum, &inv, &sm, &mut out, &mut scales, &rows_i, &pad_i, &grp_i)
        .map_err(err)?;
        Ok((out, scales))
    }

    /// V bf16 `[bh, rows, 128]` -> e4m3 `[bh, 128, rows_pad]` (transposed,
    /// keys permuted per 16) + per-channel scales `[bh, 128]`.
    fn quant_v(
        k: &Kernels,
        v: &CudaSlice<half::bf16>,
        bh: usize,
        rows: usize,
        rows_pad: usize,
    ) -> Result<(CudaSlice<u8>, CudaSlice<f32>)> {
        let dev = ctx()?;
        let mut vmax = zeros::<u32>(bh * HEAD_DIM)?;
        let (rows_i, rpb, pad_i) = (rows as i32, COLSUM_ROWS as i32, rows_pad as i32);
        let cfg = LaunchConfig {
            grid_dim: (rows.div_ceil(COLSUM_ROWS) as u32, bh as u32, 1),
            block_dim: (HEAD_DIM as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        launch!(dev.stream, &k.vmax, cfg; v, &mut vmax, &rows_i, &rpb).map_err(err)?;
        let mut vt = alloc::<u8>(bh * HEAD_DIM * rows_pad)?;
        let mut vs = alloc::<f32>(bh * HEAD_DIM)?;
        let cfg = LaunchConfig {
            grid_dim: ((rows_pad / 64) as u32, bh as u32, 1),
            block_dim: (HEAD_DIM as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        launch!(dev.stream, &k.vquant, cfg; v, &vmax, &mut vt, &mut vs, &rows_i, &pad_i)
            .map_err(err)?;
        Ok((vt, vs))
    }

    /// Dense SDPA, `[b, h, s, 128]` q/k/v (bf16, or f32 cast once), output
    /// f32 or bf16. `None` where the device or shape is not one the kernel
    /// runs (the caller keeps the bf16 kernels).
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
            || b * h > 65_535
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
        let (q8, qs) = quant_i8(&kern, &qb, bh, sq, sq_pad, 16, false)?;
        let (k8, ks) = quant_i8(&kern, &kbf, bh, sk, sk_pad, 64, true)?;
        let (vt, vs) = quant_v(&kern, &vb, bh, sk, sk_pad)?;
        let scale = scale.unwrap_or(1.0 / (d as f32).sqrt());
        let sl2 = scale * std::f32::consts::LOG2_E;
        static ONCE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        crate::wan::log::info_once(
            &ONCE,
            format_args!("sdpa: SageAttention2 INT8 QK / FP8 PV (attn_sage) B={b} H={h} Sq={sq} Sk={sk} D={d}"),
        );
        let cfg = LaunchConfig {
            grid_dim: ((sq_pad / 128) as u32, bh as u32, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: SMEM,
        };
        let n = bh * sq * d;
        let (sq_i, sk_i, sqp_i, skp_i) = (sq as i32, sk as i32, sq_pad as i32, sk_pad as i32);
        if out_bf16 {
            let mut out = alloc::<half::bf16>(n)?;
            let mut dummy = alloc::<f32>(1)?;
            let one = 1i32;
            launch!(dev.stream, &kern.fwd_d128, cfg;
                &q8, &qs, &k8, &ks, &vt, &vs, &mut dummy, &mut out, &one, &sq_i, &sk_i, &sqp_i, &skp_i, &sl2)
            .map_err(err)?;
            Ok(Some(CudaTensor::from_device_slice_bf16(
                out,
                vec![b, h, sq, d],
            )?))
        } else {
            let mut out = alloc::<f32>(n)?;
            let mut dummy = alloc::<half::bf16>(1)?;
            let zero = 0i32;
            launch!(dev.stream, &kern.fwd_d128, cfg;
                &q8, &qs, &k8, &ks, &vt, &vs, &mut out, &mut dummy, &zero, &sq_i, &sk_i, &sqp_i, &skp_i, &sl2)
            .map_err(err)?;
            Ok(Some(CudaTensor::from_device_slice(out, vec![b, h, sq, d])?))
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn head_dim_is_the_kernels() {
        assert_eq!(super::HEAD_DIM, 128);
    }

    #[test]
    fn recipes_enable_it_on_sm120_only() {
        use super::decide;
        // Unset: only a recipe on sm_120.
        assert!(decide(None, true, (12, 0)));
        for sm in [(8, 9), (9, 0), (10, 0), (10, 3), (12, 1)] {
            assert!(!decide(None, true, sm), "{sm:?}");
        }
        assert!(!decide(None, false, (12, 0)));
        // `=0` wins over a recipe; `=2` turns it on everywhere it can run.
        assert!(!decide(Some(0), true, (12, 0)));
        assert!(decide(Some(2), false, (9, 0)));
        assert!(decide(Some(2), false, (12, 0)));
        // Never below sm_89 (no INT8 / FP8 mma.sync).
        assert!(!decide(Some(2), true, (8, 6)));
    }

    #[test]
    fn recipe_scope_is_per_thread_and_restores() {
        assert!(!super::recipe_wants());
        {
            let _outer = super::recipe_scope(true);
            assert!(super::recipe_wants());
            std::thread::spawn(|| assert!(!super::recipe_wants()))
                .join()
                .unwrap();
            {
                let _inner = super::recipe_scope(false);
                assert!(!super::recipe_wants());
            }
            assert!(super::recipe_wants());
        }
        assert!(!super::recipe_wants());
    }
}
