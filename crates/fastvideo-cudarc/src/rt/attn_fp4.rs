//! Sparse FP4 VSA attention (opt-in, lossy, sm_120 only): the VSA fine stage
//! with `Q K^T` and `P V` both on block-scaled FP4 tensor cores
//! (`mma.sync … kind::mxf4nvf4.block_scale.scale_vec::4X`, E2M1 operands, a
//! UE4M3 scale per 16 elements). Kernels: `attn_fp4.cu`, an sm_120a cubin
//! built ahead of time (NVRTC SASS as the fallback).
//!
//! The recipe is FastVideo's SageAttention3 FP4 path (`attn_qat_infer`):
//! Q and K per 16 head dims, V per 16 keys, P per 16 keys inside the online
//! softmax, each `amax / 6` rounded to E4M3; no smoothing and no global scale.
//! The block mask is VSA-H3's 64-row tile lists, the same (query tile, key
//! tile) set upstream encodes as quadrant bits on 128x128 blocks.
//! Design and measurements: docs/ports/fasth3-sparse-fp4-attn.md.
//!
//! Nothing here allocates anything shaped like `[queries, keys]`: the
//! quantized operands are `rows * 72` bytes per head (codes + scales), and
//! the output is the fine stage's usual f32 `[bh, padded, 128]`.

/// Head dim the kernels are written for.
pub const HEAD_DIM: usize = 128;

/// `FASTVIDEO_H3_VSA_FP4`: the H3 VSA fine stage on this kernel, for ModelOpt
/// NVFP4 checkpoints only. Unset: on for those checkpoints where the sm_120
/// kernel exists (owner decision 2026-10-14); `0` keeps bf16 VSA; `1` insists.
pub const ENV: &str = "FASTVIDEO_H3_VSA_FP4";

/// `FASTVIDEO_H3_VSA_FP4_P=single`: upstream's single-level P quantization
/// (the default here is two-level: P pre-scaled by 448 * 6).
pub const P_ENV: &str = "FASTVIDEO_H3_VSA_FP4_P";

/// The explicit setting: `Some(true)` for `1`, `Some(false)` for `0` / `false`
/// / `off`, `None` when unset (the NVFP4 tier's default: on where available).
pub fn setting() -> Option<bool> {
    fastvideo_models::techniques::settings::var(ENV)
        .map(|_| crate::rt::envflag::bool_flag(ENV, true))
}

/// Single-level P quantization (upstream's `single_level_p_quant=True`).
pub fn p_single() -> bool {
    crate::rt::envflag::string_flag(P_ENV, "two-level") == "single"
}

/// Host E4M3 (OCP FN) value of a byte; scales are never negative.
pub fn e4m3_value(b: u8) -> f32 {
    let (e, m) = (u32::from(b >> 3) & 15, u32::from(b & 7));
    if e == 0 {
        m as f32 * 2f32.powi(-9)
    } else {
        (1.0 + m as f32 / 8.0) * 2f32.powi(e as i32 - 7)
    }
}

/// Host E4M3 encode of `x >= 0` (round to nearest even, saturating at 448),
/// the device's `cvt.rn.satfinite.e4m3x2.f32`.
pub fn e4m3_encode(x: f32) -> u8 {
    if x.is_nan() || x <= 0.0 {
        return 0;
    }
    // Every finite non-negative byte except 0x7f (NaN), in increasing order.
    let mut best = 0u8;
    let mut best_err = f32::INFINITY;
    for b in 0u8..=0x7e {
        let v = e4m3_value(b);
        let err = (v - x).abs();
        // Ties to the even mantissa (lowest bit 0).
        if err < best_err || (err == best_err && b & 1 == 0) {
            best = b;
            best_err = err;
        }
    }
    best
}

/// E2M1 values of the 8 non-negative codes.
pub const E2M1: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];

/// Host E2M1 encode (round to nearest even, saturating at ±6): a nibble.
pub fn e2m1_encode(x: f32) -> u8 {
    let a = x.abs().min(6.0);
    let mut best = 0u8;
    let mut best_err = f32::INFINITY;
    for (i, &v) in E2M1.iter().enumerate() {
        let err = (v - a).abs();
        if err < best_err || (err == best_err && i % 2 == 0) {
            best = i as u8;
            best_err = err;
        }
    }
    if x.is_sign_negative() && best != 0 {
        best | 8
    } else {
        best
    }
}

/// E2M1 nibble -> value.
pub fn e2m1_value(n: u8) -> f32 {
    let v = E2M1[usize::from(n & 7)];
    if n & 8 != 0 {
        -v
    } else {
        v
    }
}

/// 16 values -> (scale byte, 16 nibbles): upstream's per-16 rule.
pub fn quant16(x: &[f32]) -> (u8, [u8; 16]) {
    let amax = x.iter().fold(0f32, |m, v| m.max(v.abs()));
    let sb = e4m3_encode(amax / 6.0);
    let sf = e4m3_value(sb);
    let inv = if sf == 0.0 { 0.0 } else { 1.0 / sf };
    let mut codes = [0u8; 16];
    for (c, v) in codes.iter_mut().zip(x) {
        *c = e2m1_encode(v * inv);
    }
    (sb, codes)
}

/// Stored row of key row `c` within a 64-row K tile (an involution).
pub fn kperm(c: usize) -> usize {
    let l = c & 31;
    (c & 32) | (((l >> 1) & 3) << 3) | ((l >> 3) << 1) | (l & 1)
}

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
    use crate::rt::device::{self, DeviceContext};
    use crate::rt::kernels::launch;
    use crate::rt::ops::VsaPlanDev;
    use crate::rt::tensor::{Result, TensorError};

    const SRC: &str = include_str!("attn_fp4.cu");
    /// Q (4 KB + 512 B) and three (K, V) stages of 9 KB.
    const SMEM: u32 = 4608 + 3 * 9216;

    mod aot {
        include!(concat!(env!("OUT_DIR"), "/aot_attn_fp4.rs"));
    }

    fn err(e: impl std::fmt::Display) -> TensorError {
        TensorError::Message(format!("attn_fp4: {e}"))
    }

    fn ctx() -> Result<Arc<DeviceContext>> {
        device::global_device().ok_or_else(|| err("no global CUDA device"))
    }

    /// Block-scaled FP4 `mma.sync` is sm_120a code: the RTX PRO 6000 /
    /// RTX 50 family (12.0).
    pub fn supported() -> bool {
        device::global_device().is_some_and(|d| (d.sm_major, d.sm_minor) == (12, 0))
    }

    struct Kernels {
        _module: Arc<cudarc::driver::CudaModule>,
        quant: CudaFunction,
        vsa: CudaFunction,
        probe: CudaFunction,
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
        let dev = ctx()?;
        if !supported() {
            return Err(err(format!(
                "needs sm_120 (this device is sm_{}{})",
                dev.sm_major, dev.sm_minor
            )));
        }
        let timer = std::time::Instant::now();
        let (module, origin) =
            if let Some((_, cubin)) = aot::AOT_ATTN_FP4.iter().find(|(sm, _)| *sm == 120) {
                let path = std::env::temp_dir()
                    .join(format!("fv-attn-fp4-{}-sm120a.cubin", std::process::id()));
                std::fs::write(&path, cubin).map_err(err)?;
                let loaded = dev.ctx.load_module(cudarc::nvrtc::Ptx::from_file(&path));
                let _ = std::fs::remove_file(&path);
                (loaded.map_err(err)?, "cubin sm_120a".to_string())
            } else {
                let opts = cudarc::nvrtc::CompileOptions {
                    arch: Some("compute_120a"),
                    use_fast_math: Some(true),
                    ftz: Some(true),
                    options: vec!["-std=c++17".into()],
                    ..Default::default()
                };
                let (m, o) = crate::rt::nvrtc_sass::load(
                    &dev.ctx,
                    SRC,
                    &opts,
                    Some("sm_120a"),
                    &["compute_120a"],
                    "attn-fp4",
                )
                .map_err(err)?;
                (
                    m,
                    if o.sass {
                        "nvrtc sass sm_120a".to_string()
                    } else {
                        "nvrtc ptx".to_string()
                    },
                )
            };
        let f = |name: &str| module.load_function(name).map_err(err);
        let k = Kernels {
            quant: f("attn_fp4_quant_tile")?,
            vsa: f("attn_fp4_vsa")?,
            probe: f("attn_fp4_mma_probe")?,
            _module: module,
        };
        crate::rt::ops::opt_in_dynamic_shared(&k.vsa, SMEM)?;
        crate::rt::log::info(format_args!(
            "attn_fp4: kernels loaded ({origin}, {:.1}s)",
            timer.elapsed().as_secs_f64()
        ));
        Ok(k)
    }

    /// Whether the kernels load on this device (and log why not, once).
    pub fn available() -> bool {
        static ONCE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        match supported().then(kernels) {
            Some(Ok(_)) => true,
            Some(Err(e)) => {
                crate::rt::log::info_once(&ONCE, format_args!("attn_fp4: unavailable: {e}"));
                false
            }
            None => false,
        }
    }

    fn alloc_u8(n: usize) -> Result<CudaSlice<u8>> {
        let dev = ctx()?;
        unsafe { dev.stream.alloc::<u8>(n.max(1)) }.map_err(err)
    }

    /// q / k / v as the fine stage reads them: `[bh, seq, 128]` in packed order.
    pub enum Fp4In<'a> {
        F32 {
            q: &'a CudaSlice<f32>,
            k: &'a CudaSlice<f32>,
            v: &'a CudaSlice<f32>,
        },
        Bf16 {
            q: &'a CudaSlice<half::bf16>,
            k: &'a CudaSlice<half::bf16>,
            v: &'a CudaSlice<half::bf16>,
        },
    }

    /// The FP4 operands in tile-slot order (attn_fp4.cu: `attn_fp4_quant_tile`).
    pub struct Fp4Tiles {
        /// `[2][bh][padded][64]` codes of q (plane 0) and k (plane 1, rows
        /// permuted within each tile by [`super::kperm`]).
        pub qk4: CudaSlice<u8>,
        /// `[2][bh][padded][8]` E4M3 scales, rows as `qk4`.
        pub qksf: CudaSlice<u8>,
        /// `[bh][tiles][128][32]` codes of V^T.
        pub v4: CudaSlice<u8>,
        /// `[bh][tiles][128][4]` scales of V^T (16 keys each).
        pub vsf: CudaSlice<u8>,
        pub bh: usize,
        pub num_tiles: usize,
    }

    /// Quantize q / k / v into tile-slot FP4 ([`Fp4Tiles`]).
    pub fn quantize(
        input: &Fp4In<'_>,
        plan: &VsaPlanDev,
        bh: usize,
        seq: usize,
    ) -> Result<Fp4Tiles> {
        if plan.tile_elems != 64 {
            return Err(err(format!("64-slot tiles only (got {})", plan.tile_elems)));
        }
        let kern = kernels()?;
        let dev = ctx()?;
        let nt = plan.num_tiles;
        let padded = nt * 64;
        let mut t = Fp4Tiles {
            qk4: alloc_u8(2 * bh * padded * 64)?,
            qksf: alloc_u8(2 * bh * padded * 8)?,
            v4: alloc_u8(bh * nt * HEAD_DIM * 32)?,
            vsf: alloc_u8(bh * nt * HEAD_DIM * 4)?,
            bh,
            num_tiles: nt,
        };
        let cfg = LaunchConfig {
            grid_dim: (nt as u32, bh as u32, 3),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 0,
        };
        let (seq_i, nt_i) = (seq as i32, nt as i32);
        match input {
            Fp4In::F32 { q, k, v } => {
                let zero = 0i32;
                launch!(dev.stream, &kern.quant, cfg;
                    *q, *k, *v, &zero, &plan.slot_src, &seq_i, &nt_i, &mut t.qk4, &mut t.qksf, &mut t.v4, &mut t.vsf)
                .map_err(err)?;
            }
            Fp4In::Bf16 { q, k, v } => {
                let one = 1i32;
                launch!(dev.stream, &kern.quant, cfg;
                    *q, *k, *v, &one, &plan.slot_src, &seq_i, &nt_i, &mut t.qk4, &mut t.qksf, &mut t.v4, &mut t.vsf)
                .map_err(err)?;
            }
        }
        Ok(t)
    }

    /// The fine stage on quantized tiles: f32 `[bh, padded, 128]` in
    /// tile-slot order for query tiles `[q_base, q_base + q_tiles)`, zero
    /// elsewhere (the contract of [`crate::rt::ops::vsa_mma_attn_range_device`]).
    #[allow(clippy::too_many_arguments)]
    pub fn fine(
        tiles: &Fp4Tiles,
        selected: &CudaSlice<u32>,
        plan: &VsaPlanDev,
        topk: usize,
        scale: f32,
        q_base: usize,
        q_tiles: usize,
        p_single: bool,
    ) -> Result<CudaSlice<f32>> {
        let kern = kernels()?;
        let dev = ctx()?;
        let (bh, nb) = (tiles.bh, tiles.num_tiles);
        let q_tiles = q_tiles.min(nb.saturating_sub(q_base));
        if q_tiles == 0 || topk == 0 || topk > nb {
            return Err(err(format!(
                "empty query range or top-k {topk} of {nb} tiles"
            )));
        }
        let padded = nb * 64;
        let mut out = if q_base == 0 && q_tiles == nb {
            unsafe { dev.stream.alloc::<f32>(bh * padded * HEAD_DIM) }.map_err(err)?
        } else {
            crate::rt::ops::fill_device(bh * padded * HEAD_DIM, 0.0)?
        };
        let cfg = LaunchConfig {
            grid_dim: (q_tiles as u32, bh as u32, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: SMEM,
        };
        let (nt, tk, qb, ps) = (nb as i32, topk as i32, q_base as i32, i32::from(p_single));
        let sl2 = scale * std::f32::consts::LOG2_E;
        launch!(dev.stream, &kern.vsa, cfg;
            &tiles.qk4, &tiles.qksf, &tiles.v4, &tiles.vsf, selected, &plan.block_sizes, &mut out,
            &nt, &tk, &sl2, &qb, &ps)
        .map_err(err)?;
        Ok(out)
    }

    /// [`quantize`] then [`fine`]: the drop-in for the VSA fine stage.
    #[allow(clippy::too_many_arguments)]
    pub fn vsa_fine(
        input: &Fp4In<'_>,
        selected: &CudaSlice<u32>,
        plan: &VsaPlanDev,
        bh: usize,
        seq: usize,
        topk: usize,
        scale: f32,
        q_base: usize,
        q_tiles: usize,
    ) -> Result<CudaSlice<f32>> {
        static ONCE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        let single = super::p_single();
        crate::rt::log::info_once(
            &ONCE,
            format_args!(
                "vsa fine kernel: sparse FP4 Q K^T + P V (attn_fp4_vsa), {} tiles, top-k {topk}, P {}",
                plan.num_tiles,
                if single { "single-level" } else { "two-level" }
            ),
        );
        let tiles = quantize(input, plan, bh, seq)?;
        fine(&tiles, selected, plan, topk, scale, q_base, q_tiles, single)
    }

    /// One `mma.sync m16n8k64` (mxf4nvf4, 4X) on raw per-lane fragments:
    /// `a [32 * 4]`, `b [32 * 2]`, `sfa [32]`, `sfb [32]` -> `d [32 * 4]`.
    pub fn mma_probe(a: &[u32], b: &[u32], sfa: &[u32], sfb: &[u32]) -> Result<Vec<f32>> {
        if a.len() != 128 || b.len() != 64 || sfa.len() != 32 || sfb.len() != 32 {
            return Err(err("mma_probe: a [128], b [64], sfa [32], sfb [32]"));
        }
        let kern = kernels()?;
        let dev = ctx()?;
        let up = |x: &[u32]| dev.stream.memcpy_stod(x).map_err(err);
        let (a, b, sa, sb) = (up(a)?, up(b)?, up(sfa)?, up(sfb)?);
        let mut d = dev.stream.alloc_zeros::<f32>(128).map_err(err)?;
        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (32, 1, 1),
            shared_mem_bytes: 0,
        };
        launch!(dev.stream, &kern.probe, cfg; &a, &b, &sa, &sb, &mut d).map_err(err)?;
        dev.stream.memcpy_dtov(&d).map_err(err)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn e4m3_round_trips_and_rounds_to_nearest_even() {
        for b in 0u8..=0x7e {
            assert_eq!(e4m3_encode(e4m3_value(b)), b, "byte {b:#x}");
        }
        assert_eq!(e4m3_value(0x7e), 448.0);
        assert_eq!(e4m3_encode(1e9), 0x7e, "saturates");
        // 1.0625 is halfway between 1.0 (0x38) and 1.125 (0x39): even wins.
        assert_eq!(e4m3_encode(1.0625), 0x38);
        assert_eq!(e4m3_value(1), 2f32.powi(-9), "smallest subnormal");
    }

    #[test]
    fn e2m1_rounds_to_nearest_even_and_saturates() {
        assert_eq!(e2m1_encode(0.25), 0, "tie to 0 (even)");
        assert_eq!(e2m1_encode(0.75), 2, "tie to 1.0 (even code)");
        assert_eq!(e2m1_encode(2.5), 4, "tie to 2.0");
        assert_eq!(e2m1_encode(5.0), 6, "tie to 4.0");
        assert_eq!(e2m1_encode(100.0), 7);
        assert_eq!(e2m1_encode(-1.5), 0xb);
        for n in 0u8..16 {
            if n != 8 {
                assert_eq!(e2m1_encode(e2m1_value(n)), n);
            }
        }
    }

    #[test]
    fn the_k_permutation_is_an_involution_that_feeds_p_as_the_a_fragment() {
        for c in 0..64 {
            assert_eq!(kperm(kperm(c)), c);
        }
        // Thread t's accumulator columns 8n + 2t + j (n < 4) are keys 8t + 2n + j:
        // eight consecutive keys, in order, as the A fragment's register a0.
        for t in 0..4 {
            let keys: Vec<usize> = (0..4)
                .flat_map(|n| (0..2).map(move |j| kperm(8 * n + 2 * t + j)))
                .collect();
            assert_eq!(keys, (8 * t..8 * t + 8).collect::<Vec<_>>());
            let hi: Vec<usize> = (4..8)
                .flat_map(|n| (0..2).map(move |j| kperm(8 * n + 2 * t + j)))
                .collect();
            assert_eq!(hi, (32 + 8 * t..32 + 8 * t + 8).collect::<Vec<_>>());
        }
    }

    #[test]
    fn quant16_follows_the_upstream_rule() {
        let x: Vec<f32> = (0..16).map(|i| (i as f32 - 7.5) * 0.37).collect();
        let (sb, codes) = quant16(&x);
        let sf = e4m3_value(sb);
        assert_eq!(sb, e4m3_encode(2.775 / 6.0));
        for (c, v) in codes.iter().zip(&x) {
            let back = e2m1_value(*c) * sf;
            assert!((back - v).abs() <= sf * 1.0 + 1e-6, "{v} -> {back}");
        }
        assert_eq!(quant16(&[0.0; 16]), (0, [0; 16]));
    }
}
