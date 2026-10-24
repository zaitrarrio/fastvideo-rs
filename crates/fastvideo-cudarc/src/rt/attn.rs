//! The dense attention kernels: fused tensor-core SDPA (default on bf16
//! sm80+ contexts; V1 / V2 / V3 / V3s, cuDNN, attn_dc), its block-causal and
//! VDN-window variants, cuBLAS dense (fallback / `FASTVIDEO_SDPA=cublas`) and
//! device flash (opt-in). Which one runs is [`super::sdpa_rule`]'s choice and
//! model code reaches them through [`super::sdpa::sdpa`] (ADR-0015).

#[cfg(feature = "cuda")]
use std::sync::atomic::AtomicBool;

#[cfg(feature = "cuda")]
use super::tensor::TensorError;
use super::tensor::{CudaTensor, Result};

#[cfg(feature = "cuda")]
fn msg(s: impl Into<String>) -> TensorError {
    TensorError::Message(s.into())
}

/// Largest head dim the flash kernel is launched for. Its per-block shared
/// memory is `(2*32*d + d) * 4` bytes; d=384 (the Wan VAE mid-block) is
/// rejected by the driver with CUDA_ERROR_INVALID_VALUE.
pub const FLASH_MAX_HEAD_DIM: usize = 128;

/// Largest attention-score buffer (elements) materialized at once by the dense
/// path; longer queries are processed in chunks that address Q/out in place.
pub const DENSE_SCORE_BUDGET: usize = 256 * 1024 * 1024;

#[cfg(feature = "cuda")]
static PROBS_BF16_CACHE: super::envflag::CachedBool = super::envflag::CachedBool::new();

/// bf16 attention probabilities apply when the context runs bf16 GEMM math,
/// where cuBLAS rounds F32 operands to bf16 for the tensor-core op anyway.
/// `FASTVIDEO_ATTN_PROBS_BF16=0` forces F32 probabilities back (A/B runs, and
/// an escape hatch if a model ever proves sensitive to the rounding).
#[cfg(feature = "cuda")]
fn probs_bf16() -> bool {
    PROBS_BF16_CACHE.get_or_init(|| super::envflag::bool_flag("FASTVIDEO_ATTN_PROBS_BF16", true))
        && super::stats::device_expected()
        && super::device::global_device()
            .is_some_and(|d| d.gemm_math == super::device::GemmMath::Bf16)
}

#[cfg(feature = "cuda")]
fn bhsd(
    q: &CudaTensor,
    k: &CudaTensor,
    v: &CudaTensor,
) -> Option<(usize, usize, usize, usize, usize)> {
    let [b, h, sq, d] = q.shape[..] else {
        return None;
    };
    let sk = k.shape.get(2).copied()?;
    (k.shape == [b, h, sk, d] && v.shape == [b, h, sk, d]).then_some((b, h, sq, sk, d))
}

/// Tiled flash attention on-device (`FASTVIDEO_SDPA=flash`). `None` when the
/// head dim is outside the kernel's limits or no device is expected.
#[cfg(feature = "cuda")]
pub fn device_flash_sdpa(
    q: &CudaTensor,
    k: &CudaTensor,
    v: &CudaTensor,
    scale: Option<f32>,
) -> Result<Option<CudaTensor>> {
    let Some((b, h, sq, sk, d)) = bhsd(q, k, v) else {
        return Ok(None);
    };
    if d == 0 || d % 32 != 0 || d > FLASH_MAX_HEAD_DIM {
        return Ok(None);
    }
    let (Some(qd), Some(kd), Some(vd)) = (q.dev()?, k.dev()?, v.dev()?) else {
        return Ok(None);
    };
    let dev = super::device::global_device().ok_or_else(|| msg("no device"))?;
    let scale = scale.unwrap_or(1.0 / (d as f32).sqrt());
    let bh = b * h;
    static ONCE: AtomicBool = AtomicBool::new(false);
    super::log::info_once(
        &ONCE,
        format_args!("sdpa: GPU flash-tiled B={b} H={h} Sq={sq} Sk={sk} D={d}"),
    );
    let mut out = super::ops::alloc(bh * sq * d)?;
    let (bh_i, sq_i, sk_i, d_i) = (bh as i32, sq as i32, sk as i32, d as i32);
    super::kernels::launch!(dev.stream, &dev.kernels.flash_attn_f32, super::kernels::cfg_flash(bh, sq, d);
        &*qd, &*kd, &*vd, &mut out, &bh_i, &sq_i, &sk_i, &d_i, &scale)
    .map_err(|e| msg(e.to_string()))?;
    Ok(Some(CudaTensor::from_device_slice(out, vec![b, h, sq, d])?))
}

#[cfg(not(feature = "cuda"))]
pub fn device_flash_sdpa(
    _q: &CudaTensor,
    _k: &CudaTensor,
    _v: &CudaTensor,
    _scale: Option<f32>,
) -> Result<Option<CudaTensor>> {
    Ok(None)
}

/// Head dims the fused tensor-core kernel (`flash_mma_fwd_d{64,128}`) exists for.
pub const MMA_HEAD_DIMS: [usize; 2] = [64, 128];

/// Query/key tile of the fused kernel (one CTA per 64 queries), and the
/// kernel choice ([`super::sdpa_rule`] is the one chooser).
pub use super::sdpa_rule::{FlashKernel, MMA_TILE};

/// Whether the fused kernel can run a `[b, h, sq, d] x [b, h, sk, d]` SDPA on
/// an `sm_major` device: bf16 `mma.sync` needs sm80+, the head dim must be
/// one it is built for, both sequences non-empty, and `b*h` / query tiles
/// must fit the launch grid (and the kernel's `int` lengths). Everything else
/// takes the cuBLAS path.
pub fn mma_sdpa_supported(
    b: usize,
    h: usize,
    sq: usize,
    sk: usize,
    d: usize,
    sm_major: i32,
) -> bool {
    let bh = b * h;
    sm_major >= 8
        && MMA_HEAD_DIMS.contains(&d)
        && sq > 0
        && sk > 0
        && (1..=65_535).contains(&bh)
        && sq <= i32::MAX as usize
        && sk <= i32::MAX as usize
}

/// The fused kernel is the default dense SDPA wherever the context already
/// runs bf16 GEMM math (tensor-core GPUs, `FASTVIDEO_BF16` on): there Q/K/V
/// and P are rounded to bf16 by cuBLAS anyway. `FASTVIDEO_SDPA=cublas` (or an
/// exact `FASTVIDEO_BF16=0` context) keeps the materialised cuBLAS path.
#[cfg(feature = "cuda")]
pub fn mma_sdpa_default() -> bool {
    super::stats::device_expected()
        && super::device::global_device()
            .is_some_and(|d| d.gemm_math == super::device::GemmMath::Bf16 && d.sm_major >= 8)
}

#[cfg(not(feature = "cuda"))]
pub fn mma_sdpa_default() -> bool {
    false
}

/// `FASTVIDEO_SDPA_AUTO=timed`: the first call of each `(bh, sq, sk, d)`
/// times cuDNN's plan (after one warm-up execution) against flash_mma_fwd2 on
/// these very inputs and caches the faster; later calls run the winner. Where
/// the two are within noise the winner differs between processes, and so do
/// the output bytes: a diagnostic, not a default. The offline table
/// generator is `fv-gpucheck kernels --groups sdpa_rule`.
#[cfg(feature = "cuda")]
fn timed_cudnn_or_v2(
    q: &CudaTensor,
    k: &CudaTensor,
    v: &CudaTensor,
    scale: Option<f32>,
) -> Result<Option<CudaTensor>> {
    use std::collections::HashMap;
    use std::sync::{Mutex, OnceLock};
    static PICKS: OnceLock<Mutex<HashMap<(usize, usize, usize, usize), bool>>> = OnceLock::new();
    let Some((b, h, sq, sk, d)) = bhsd(q, k, v) else {
        return device_mma_sdpa_with(q, k, v, scale, true, FlashKernel::V2);
    };
    let key = (b * h, sq, sk, d);
    let known = PICKS
        .get_or_init(|| Mutex::new(HashMap::new()))
        .lock()
        .expect("sdpa picks")
        .get(&key)
        .copied();
    if let Some(cudnn) = known {
        let kernel = if cudnn {
            FlashKernel::Cudnn
        } else {
            FlashKernel::V2
        };
        return device_mma_sdpa_with(q, k, v, scale, true, kernel);
    }
    let timed = |kernel| -> Result<(Option<CudaTensor>, f64)> {
        super::device::synchronize().map_err(|e| msg(e.to_string()))?;
        let t = std::time::Instant::now();
        let out = device_mma_sdpa_with(q, k, v, scale, true, kernel)?;
        super::device::synchronize().map_err(|e| msg(e.to_string()))?;
        Ok((out, t.elapsed().as_secs_f64()))
    };
    let has_plan = super::cudnn_sdpa::has_plan(b * h, sq, sk, d);
    let (out, pick) = if has_plan {
        timed(FlashKernel::Cudnn)?; // warm-up (plan build, lazy module load)
        let (oc, tc) = timed(FlashKernel::Cudnn)?;
        let (o2, t2) = timed(FlashKernel::V2)?;
        super::log::info(format_args!(
            "sdpa auto (sm_12x timed): bh={} sq={sq} sk={sk} d={d}: cuDNN {:.2} ms, flash_mma_fwd2 {:.2} ms -> {}",
            b * h,
            tc * 1e3,
            t2 * 1e3,
            if tc < t2 { "cuDNN" } else { "flash_mma_fwd2" }
        ));
        if tc < t2 {
            (oc, true)
        } else {
            (o2, false)
        }
    } else {
        (
            device_mma_sdpa_with(q, k, v, scale, true, FlashKernel::V2)?,
            false,
        )
    };
    PICKS
        .get_or_init(|| Mutex::new(HashMap::new()))
        .lock()
        .expect("sdpa picks")
        .insert(key, pick);
    Ok(out)
}

/// Fused dense SDPA on tensor cores (`flash_mma_fwd_d{64,128}`): bf16 Q/K/V
/// (cast once, RNE, when they are f32), f32 online softmax, bf16 P, f32
/// accumulation; one launch, no score buffer. Output is f32, or bf16 when
/// `out_bf16`. The kernel is [`super::sdpa_rule::mma_kernel`]'s pick. `None`
/// when [`mma_sdpa_supported`] says no or no device is expected, so the
/// caller falls back to [`device_dense_sdpa`].
#[cfg(feature = "cuda")]
pub fn device_mma_sdpa(
    q: &CudaTensor,
    k: &CudaTensor,
    v: &CudaTensor,
    scale: Option<f32>,
    out_bf16: bool,
) -> Result<Option<CudaTensor>> {
    let dims = super::sdpa_rule::Shape::of(&q.shape, &k.shape, &v.shape).dims;
    let kernel = super::sdpa_rule::mma_kernel(dims, out_bf16, &super::sdpa::Live);
    device_mma_sdpa_pick(q, k, v, scale, out_bf16, kernel)
}

/// [`device_mma_sdpa`] on a [`super::sdpa_rule::MmaKernel`]: a fixed kernel,
/// the per-shape rule's (logged once per shape; bf16 out), or the timed
/// pick of `FASTVIDEO_SDPA_AUTO=timed`.
#[cfg(feature = "cuda")]
pub fn device_mma_sdpa_pick(
    q: &CudaTensor,
    k: &CudaTensor,
    v: &CudaTensor,
    scale: Option<f32>,
    out_bf16: bool,
    kernel: super::sdpa_rule::MmaKernel,
) -> Result<Option<CudaTensor>> {
    use super::sdpa_rule::MmaKernel;
    use std::collections::HashSet;
    use std::sync::{Mutex, OnceLock};
    match kernel {
        MmaKernel::Fixed(kernel) => device_mma_sdpa_with(q, k, v, scale, out_bf16, kernel),
        MmaKernel::Timed => timed_cudnn_or_v2(q, k, v, scale),
        MmaKernel::Rule { kernel, sm_major } => {
            if let Some((b, h, sq, sk, d)) = bhsd(q, k, v) {
                static LOGGED: OnceLock<Mutex<HashSet<(usize, usize, usize, usize)>>> =
                    OnceLock::new();
                if LOGGED
                    .get_or_init(|| Mutex::new(HashSet::new()))
                    .lock()
                    .expect("sdpa rule log")
                    .insert((b * h, sq, sk, d))
                {
                    super::log::info(format_args!(
                        "sdpa auto (sm_{sm_major}x rule): bh={} sq={sq} sk={sk} d={d} -> {}",
                        b * h,
                        if kernel == FlashKernel::Cudnn {
                            "cuDNN"
                        } else {
                            "flash_mma_fwd2"
                        }
                    ));
                }
            }
            device_mma_sdpa_with(q, k, v, scale, true, kernel)
        }
    }
}

/// [`device_mma_sdpa`] on an explicit kernel.
#[cfg(feature = "cuda")]
pub fn device_mma_sdpa_with(
    q: &CudaTensor,
    k: &CudaTensor,
    v: &CudaTensor,
    scale: Option<f32>,
    out_bf16: bool,
    kernel: FlashKernel,
) -> Result<Option<CudaTensor>> {
    let Some((b, h, sq, sk, d)) = bhsd(q, k, v) else {
        return Ok(None);
    };
    let Some(dev) = super::device::global_device() else {
        return Ok(None);
    };
    if !mma_sdpa_supported(b, h, sq, sk, d, dev.sm_major) {
        return Ok(None);
    }
    let (Some(qb), Some(kb), Some(vb)) = (q.dev_bf16()?, k.dev_bf16()?, v.dev_bf16()?) else {
        return Ok(None);
    };
    let scale = scale.unwrap_or(1.0 / (d as f32).sqrt());
    let sl2 = scale * std::f32::consts::LOG2_E;
    let bh = b * h;
    static ONCE: AtomicBool = AtomicBool::new(false);
    super::log::info_once(
        &ONCE,
        format_args!("sdpa: fused mma B={b} H={h} Sq={sq} Sk={sk} D={d}"),
    );
    let n = bh * sq * d;
    if kernel == FlashKernel::Cudnn && out_bf16 {
        let mut out =
            unsafe { dev.stream.alloc::<half::bf16>(n) }.map_err(|e| msg(e.to_string()))?;
        if super::cudnn_sdpa::sdpa_bf16(&qb, &kb, &vb, &mut out, bh, sq, sk, d, scale)? {
            return Ok(Some(CudaTensor::from_device_slice_bf16(
                out,
                vec![b, h, sq, d],
            )?));
        }
    }
    if kernel == FlashKernel::Dc {
        if out_bf16 {
            let mut out =
                unsafe { dev.stream.alloc::<half::bf16>(n) }.map_err(|e| msg(e.to_string()))?;
            let dout = super::attn_dc::DcOut::Bf16(&mut out);
            if super::attn_dc::dense_fwd(&qb, &kb, &vb, dout, bh, sq, sk, d, sl2)? {
                return Ok(Some(CudaTensor::from_device_slice_bf16(
                    out,
                    vec![b, h, sq, d],
                )?));
            }
        } else {
            let mut out = super::ops::alloc(n)?;
            let dout = super::attn_dc::DcOut::F32(&mut out);
            if super::attn_dc::dense_fwd(&qb, &kb, &vb, dout, bh, sq, sk, d, sl2)? {
                return Ok(Some(CudaTensor::from_device_slice(out, vec![b, h, sq, d])?));
            }
        }
    }
    let kernel = match kernel {
        FlashKernel::Cudnn | FlashKernel::Dc => FlashKernel::V2,
        FlashKernel::V3 | FlashKernel::V3s if d != 128 => FlashKernel::V2,
        k => k,
    };
    let (func, cfg) = match kernel {
        FlashKernel::V1 => (
            if d == 64 {
                &dev.kernels.flash_mma_fwd_d64
            } else {
                &dev.kernels.flash_mma_fwd_d128
            },
            cudarc::driver::LaunchConfig {
                grid_dim: (sq.div_ceil(MMA_TILE) as u32, bh as u32, 1),
                block_dim: (128, 1, 1),
                // Static shared memory (32 KB at d=128): two CTAs per SM, no opt-in.
                shared_mem_bytes: 0,
            },
        ),
        FlashKernel::V3 | FlashKernel::V3s => {
            let func = if kernel == FlashKernel::V3 {
                &dev.kernels.flash_mma_fwd3_d128
            } else {
                &dev.kernels.flash_mma_fwd3s_d128
            };
            // Three stages of (K, V) 64 x 128 bf16 tiles: 96 KB (opt-in).
            let shared = (6 * MMA_TILE * d * 2) as u32;
            super::ops::opt_in_dynamic_shared(func, shared)?;
            (
                func,
                cudarc::driver::LaunchConfig {
                    grid_dim: (sq.div_ceil(2 * MMA_TILE) as u32, bh as u32, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: shared,
                },
            )
        }
        FlashKernel::V2 | FlashKernel::Cudnn | FlashKernel::Dc => {
            let func = if d == 64 {
                &dev.kernels.flash_mma_fwd2_d64
            } else {
                &dev.kernels.flash_mma_fwd2_d128
            };
            // Two stages of (K, V) 64 x d bf16 tiles: 64 KB at d=128 (opt-in).
            let shared = (4 * MMA_TILE * d * 2) as u32;
            super::ops::opt_in_dynamic_shared(func, shared)?;
            (
                func,
                cudarc::driver::LaunchConfig {
                    grid_dim: (sq.div_ceil(2 * MMA_TILE) as u32, bh as u32, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: shared,
                },
            )
        }
    };
    let (sq_i, sk_i) = (sq as i32, sk as i32);
    let err = |e: cudarc::driver::DriverError| msg(e.to_string());
    let launch_err = |e: super::device::DeviceError| msg(e.to_string());
    // cudarc cannot pass a null pointer: the unused output is a 1-element dummy.
    if out_bf16 {
        let mut out = unsafe { dev.stream.alloc::<half::bf16>(n) }.map_err(err)?;
        let mut dummy = super::ops::alloc(1)?;
        let is_bf16 = 1i32;
        super::kernels::launch!(dev.stream, func, cfg;
            &*qb, &*kb, &*vb, &mut dummy, &mut out, &is_bf16, &sq_i, &sk_i, &sl2)
        .map_err(launch_err)?;
        Ok(Some(CudaTensor::from_device_slice_bf16(
            out,
            vec![b, h, sq, d],
        )?))
    } else {
        let mut out = super::ops::alloc(n)?;
        let mut dummy = unsafe { dev.stream.alloc::<half::bf16>(1) }.map_err(err)?;
        let is_bf16 = 0i32;
        super::kernels::launch!(dev.stream, func, cfg;
            &*qb, &*kb, &*vb, &mut out, &mut dummy, &is_bf16, &sq_i, &sk_i, &sl2)
        .map_err(launch_err)?;
        Ok(Some(CudaTensor::from_device_slice(out, vec![b, h, sq, d])?))
    }
}

#[cfg(not(feature = "cuda"))]
pub fn device_mma_sdpa(
    _q: &CudaTensor,
    _k: &CudaTensor,
    _v: &CudaTensor,
    _scale: Option<f32>,
    _out_bf16: bool,
) -> Result<Option<CudaTensor>> {
    Ok(None)
}

/// Wan's block-causal temporal mask as a kernel parameter rather than an
/// `[S, S]` additive tensor (FastVideo `CausalWanTransformer3DModel.
/// _prepare_blockwise_causal_attn_mask` when the window is whole blocks):
/// query `q` sees key `k` iff `block(k) <= block(q)` (`block(i) = i /
/// block_tokens`, `block_tokens = num_frames_per_block * frame tokens`),
/// among the last `window` blocks counting `q`'s own when `window > 0`, or
/// `block(k) < sink`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BlockCausal {
    pub block_tokens: usize,
    pub window: usize,
    pub sink: usize,
}

impl BlockCausal {
    pub fn allows(&self, q: usize, k: usize) -> bool {
        let bt = self.block_tokens.max(1);
        let (tq, tk) = (q / bt, k / bt);
        (tk <= tq && (self.window == 0 || tq - tk < self.window)) || tk < self.sink
    }

    /// The additive `[sq, sk]` mask (0 visible, `-1e9` masked).
    pub fn dense_mask(&self, sq: usize, sk: usize) -> Vec<f32> {
        let mut m = vec![0.0f32; sq * sk];
        for q in 0..sq {
            for k in 0..sk {
                if !self.allows(q, k) {
                    m[q * sk + k] = -1e9;
                }
            }
        }
        m
    }
}

/// VDN-H3's window softmax mask (docs/ports/vdn-h3.md §1.3) as kernel
/// parameters: rows outside `[video_start, video_end)` are global (dense both
/// ways); frames 0 and `frames - 1` are dense anchors; any other video query
/// frame sees the key frames of its window (`chunk`-aligned, `radius` chunks
/// each side; `chunk == 0` is a centred frame window).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VdnWindow {
    pub video_start: usize,
    pub frame_tokens: usize,
    pub frames: usize,
    pub chunk: usize,
    pub radius: usize,
}

impl VdnWindow {
    pub fn video_end(&self) -> usize {
        self.video_start + self.frames * self.frame_tokens
    }

    fn bound(&self, f: i64, hi: bool) -> i64 {
        let (c, r) = (self.chunk as i64, self.radius as i64);
        match (c > 0, hi) {
            (true, false) => (f / c - r) * c,
            (true, true) => (f / c + r + 1) * c - 1,
            (false, false) => f - r,
            (false, true) => f + r,
        }
    }

    pub fn allows(&self, q: usize, k: usize) -> bool {
        let (vs, ve, t) = (self.video_start, self.video_end(), self.frame_tokens.max(1));
        if !(vs..ve).contains(&q) || !(vs..ve).contains(&k) {
            return true;
        }
        let last = self.frames - 1;
        let (qf, kf) = ((q - vs) / t, (k - vs) / t);
        if qf == 0 || kf == 0 || qf == last || kf == last {
            return true;
        }
        let kf = kf as i64;
        kf >= self.bound(qf as i64, false) && kf <= self.bound(qf as i64, true)
    }

    /// The additive `[sq, sk]` mask (0 visible, `-1e9` masked).
    pub fn dense_mask(&self, sq: usize, sk: usize) -> Vec<f32> {
        let mut m = vec![0.0f32; sq * sk];
        for q in 0..sq {
            for k in 0..sk {
                if !self.allows(q, k) {
                    m[q * sk + k] = -1e9;
                }
            }
        }
        m
    }
}

/// [`device_mma_sdpa`] under a [`VdnWindow`] mask (`flash_mma_fwd2_window_d128`):
/// self-attention (`sq == sk`), head dim 128. `None` when the kernel cannot run.
#[cfg(feature = "cuda")]
pub fn device_mma_sdpa_window(
    q: &CudaTensor,
    k: &CudaTensor,
    v: &CudaTensor,
    scale: Option<f32>,
    out_bf16: bool,
    mask: VdnWindow,
) -> Result<Option<CudaTensor>> {
    let Some((b, h, sq, sk, d)) = bhsd(q, k, v) else {
        return Ok(None);
    };
    let Some(dev) = super::device::global_device() else {
        return Ok(None);
    };
    if d != 128
        || sq != sk
        || !mma_sdpa_supported(b, h, sq, sk, d, dev.sm_major)
        || mask.frames == 0
        || mask.frame_tokens == 0
        || mask.video_end() > sk
        || sk > i32::MAX as usize
    {
        return Ok(None);
    }
    let (Some(qb), Some(kb), Some(vb)) = (q.dev_bf16()?, k.dev_bf16()?, v.dev_bf16()?) else {
        return Ok(None);
    };
    let scale = scale.unwrap_or(1.0 / (d as f32).sqrt());
    let sl2 = scale * std::f32::consts::LOG2_E;
    let bh = b * h;
    static ONCE: AtomicBool = AtomicBool::new(false);
    super::log::info_once(
        &ONCE,
        format_args!(
            "sdpa: vdn window mma B={b} H={h} S={sq} D={d} video={}+{}x{} chunk={} radius={}",
            mask.video_start, mask.frames, mask.frame_tokens, mask.chunk, mask.radius
        ),
    );
    let func = &dev.kernels.flash_mma_fwd2_window_d128;
    let shared = (4 * MMA_TILE * d * 2) as u32;
    super::ops::opt_in_dynamic_shared(func, shared)?;
    let cfg = cudarc::driver::LaunchConfig {
        grid_dim: (sq.div_ceil(2 * MMA_TILE) as u32, bh as u32, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: shared,
    };
    let (sq_i, sk_i) = (sq as i32, sk as i32);
    let p = [
        mask.video_start,
        mask.video_end(),
        mask.frame_tokens,
        mask.frames,
        mask.chunk,
        mask.radius,
    ]
    .map(|x| x as i32);
    let err = |e: cudarc::driver::DriverError| msg(e.to_string());
    let launch_err = |e: super::device::DeviceError| msg(e.to_string());
    let n = bh * sq * d;
    if out_bf16 {
        let mut out = unsafe { dev.stream.alloc::<half::bf16>(n) }.map_err(err)?;
        let mut dummy = super::ops::alloc(1)?;
        let is_bf16 = 1i32;
        super::kernels::launch!(dev.stream, func, cfg;
            &*qb, &*kb, &*vb, &mut dummy, &mut out, &is_bf16, &sq_i, &sk_i, &sl2,
            &p[0], &p[1], &p[2], &p[3], &p[4], &p[5])
        .map_err(launch_err)?;
        Ok(Some(CudaTensor::from_device_slice_bf16(
            out,
            vec![b, h, sq, d],
        )?))
    } else {
        let mut out = super::ops::alloc(n)?;
        let mut dummy = unsafe { dev.stream.alloc::<half::bf16>(1) }.map_err(err)?;
        let is_bf16 = 0i32;
        super::kernels::launch!(dev.stream, func, cfg;
            &*qb, &*kb, &*vb, &mut out, &mut dummy, &is_bf16, &sq_i, &sk_i, &sl2,
            &p[0], &p[1], &p[2], &p[3], &p[4], &p[5])
        .map_err(launch_err)?;
        Ok(Some(CudaTensor::from_device_slice(out, vec![b, h, sq, d])?))
    }
}

#[cfg(not(feature = "cuda"))]
pub fn device_mma_sdpa_window(
    _q: &CudaTensor,
    _k: &CudaTensor,
    _v: &CudaTensor,
    _scale: Option<f32>,
    _out_bf16: bool,
    _mask: VdnWindow,
) -> Result<Option<CudaTensor>> {
    Ok(None)
}

/// The [`BlockCausal`] of a causal Wan config over `frame_tokens` tokens per
/// latent frame, when FastVideo's token-granular window
/// (`local_attn_size * frame_tokens` keys back from the end of the query's
/// block) covers whole blocks; `None` otherwise (the caller builds the dense
/// mask with `causal_temporal_mask`). FastVideo's full-sequence mask has no
/// sink: `sink_size` only keeps frames when the inference KV cache rolls.
pub fn block_causal_for(
    cfg: &fastvideo_models::wan::WanVideoArchConfig,
    frame_tokens: usize,
) -> Option<BlockCausal> {
    let fpb = cfg.num_frames_per_block.max(1);
    let window = match usize::try_from(cfg.local_attn_size) {
        Err(_) => 0,
        Ok(l) if l > 0 && l % fpb == 0 => l / fpb,
        Ok(_) => return None,
    };
    Some(BlockCausal {
        block_tokens: fpb * frame_tokens,
        window,
        sink: 0,
    })
}

/// [`device_mma_sdpa`] under a [`BlockCausal`] mask
/// (`flash_mma_fwd2_causal_d*`): key tiles no query of a CTA can see are
/// skipped, straddling tiles are masked per score, and the online softmax is
/// the dense V2 kernel's. `None` when the kernel cannot run the shape.
#[cfg(feature = "cuda")]
pub fn device_mma_sdpa_causal(
    q: &CudaTensor,
    k: &CudaTensor,
    v: &CudaTensor,
    scale: Option<f32>,
    out_bf16: bool,
    mask: BlockCausal,
) -> Result<Option<CudaTensor>> {
    let Some((b, h, sq, sk, d)) = bhsd(q, k, v) else {
        return Ok(None);
    };
    let Some(dev) = super::device::global_device() else {
        return Ok(None);
    };
    let fits = |x: usize| x <= i32::MAX as usize;
    if !mma_sdpa_supported(b, h, sq, sk, d, dev.sm_major)
        || mask.block_tokens == 0
        || !fits(mask.window)
        || !fits(mask.sink)
        || !fits(sq.max(sk) + mask.block_tokens)
    {
        return Ok(None);
    }
    let (Some(qb), Some(kb), Some(vb)) = (q.dev_bf16()?, k.dev_bf16()?, v.dev_bf16()?) else {
        return Ok(None);
    };
    let scale = scale.unwrap_or(1.0 / (d as f32).sqrt());
    let sl2 = scale * std::f32::consts::LOG2_E;
    let bh = b * h;
    static ONCE: AtomicBool = AtomicBool::new(false);
    super::log::info_once(
        &ONCE,
        format_args!(
            "sdpa: block-causal mma B={b} H={h} Sq={sq} Sk={sk} D={d} block={} window={} sink={}",
            mask.block_tokens, mask.window, mask.sink
        ),
    );
    let func = if d == 64 {
        &dev.kernels.flash_mma_fwd2_causal_d64
    } else {
        &dev.kernels.flash_mma_fwd2_causal_d128
    };
    let shared = (4 * MMA_TILE * d * 2) as u32;
    super::ops::opt_in_dynamic_shared(func, shared)?;
    let cfg = cudarc::driver::LaunchConfig {
        grid_dim: (sq.div_ceil(2 * MMA_TILE) as u32, bh as u32, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: shared,
    };
    let (sq_i, sk_i) = (sq as i32, sk as i32);
    let (ft, win, sink) = (
        mask.block_tokens as i32,
        mask.window as i32,
        mask.sink as i32,
    );
    let err = |e: cudarc::driver::DriverError| msg(e.to_string());
    let launch_err = |e: super::device::DeviceError| msg(e.to_string());
    let n = bh * sq * d;
    if out_bf16 {
        let mut out = unsafe { dev.stream.alloc::<half::bf16>(n) }.map_err(err)?;
        let mut dummy = super::ops::alloc(1)?;
        let is_bf16 = 1i32;
        super::kernels::launch!(dev.stream, func, cfg;
            &*qb, &*kb, &*vb, &mut dummy, &mut out, &is_bf16, &sq_i, &sk_i, &sl2, &ft, &win, &sink)
        .map_err(launch_err)?;
        Ok(Some(CudaTensor::from_device_slice_bf16(
            out,
            vec![b, h, sq, d],
        )?))
    } else {
        let mut out = super::ops::alloc(n)?;
        let mut dummy = unsafe { dev.stream.alloc::<half::bf16>(1) }.map_err(err)?;
        let is_bf16 = 0i32;
        super::kernels::launch!(dev.stream, func, cfg;
            &*qb, &*kb, &*vb, &mut out, &mut dummy, &is_bf16, &sq_i, &sk_i, &sl2, &ft, &win, &sink)
        .map_err(launch_err)?;
        Ok(Some(CudaTensor::from_device_slice(out, vec![b, h, sq, d])?))
    }
}

#[cfg(not(feature = "cuda"))]
pub fn device_mma_sdpa_causal(
    _q: &CudaTensor,
    _k: &CudaTensor,
    _v: &CudaTensor,
    _scale: Option<f32>,
    _out_bf16: bool,
    _mask: BlockCausal,
) -> Result<Option<CudaTensor>> {
    Ok(None)
}

/// Device dense SDPA: strided-batched cuBLAS `Q@Kᵀ` + softmax + `P@V`, with
/// the query axis chunked so the score buffer stays under
/// [`DENSE_SCORE_BUDGET`]. `None` when no device is expected.
#[cfg(feature = "cuda")]
pub fn device_dense_sdpa(
    q: &CudaTensor,
    k: &CudaTensor,
    v: &CudaTensor,
    scale: Option<f32>,
) -> Result<Option<CudaTensor>> {
    device_dense_sdpa_with_budget(q, k, v, scale, DENSE_SCORE_BUDGET)
}

/// [`device_dense_sdpa`] with an explicit score-buffer budget (tests force the
/// chunked path with a small one).
#[cfg(feature = "cuda")]
pub fn device_dense_sdpa_with_budget(
    q: &CudaTensor,
    k: &CudaTensor,
    v: &CudaTensor,
    scale: Option<f32>,
    score_budget: usize,
) -> Result<Option<CudaTensor>> {
    use super::device;
    let Some((b, h, sq, sk, d)) = bhsd(q, k, v) else {
        return Ok(None);
    };
    let (Some(qd), Some(kd), Some(vd)) = (q.dev()?, k.dev()?, v.dev()?) else {
        return Ok(None);
    };
    let scale = scale.unwrap_or(1.0 / (d as f32).sqrt());
    let bh = b * h;
    let chunk = (score_budget / (bh * sk).max(1)).clamp(1, sq.max(1));
    static ONCE: AtomicBool = AtomicBool::new(false);
    super::log::info_once(
        &ONCE,
        format_args!("sdpa: device dense B={b} H={h} Sq={sq} Sk={sk} D={d} query_chunk={chunk}"),
    );
    let err = |e: device::DeviceError| msg(e.to_string());
    let mut out = super::ops::alloc((bh * sq * d).max(1))?;
    // The probability matrix is `bh*sq*sk` — far larger than Q, K, V — so in
    // fast mode it is stored as bf16, halving the dominant traffic. `V` is cast
    // once to match. Exact mode keeps F32 so it stays comparable to the CPU path.
    let v_bf16 = if probs_bf16() {
        Some(super::ops::cast_f32_bf16_device(&vd)?)
    } else {
        None
    };
    if chunk >= sq {
        let mut scores = super::ops::alloc((bh * sq * sk).max(1))?;
        device::matmul_linear_wt_strided_batched(&qd, &kd, &mut scores, bh, sq, d, sk, scale)
            .map_err(err)?;
        match &v_bf16 {
            Some(vb) => {
                let probs = super::ops::softmax_last_bf16_device(&scores, sk)?;
                drop(scores);
                device::matmul_2d_strided_batched_bf16(&probs, vb, &mut out, bh, sq, sk, d)
                    .map_err(err)?;
            }
            None => {
                let probs = super::ops::softmax_last_device(&scores, sk)?;
                drop(scores);
                device::matmul_2d_strided_batched(&probs, &vd, &mut out, bh, sq, sk, d)
                    .map_err(err)?;
            }
        }
    } else {
        let mut start = 0usize;
        while start < sq {
            let qlen = chunk.min(sq - start);
            let q_view = qd.slice(start * d..);
            let mut scores = super::ops::alloc(bh * qlen * sk)?;
            device::matmul_linear_wt_strided_batched_x_view(
                &q_view,
                sq * d,
                &kd,
                &mut scores,
                bh,
                qlen,
                d,
                sk,
                scale,
            )
            .map_err(err)?;
            let mut out_view = out.slice_mut(start * d..);
            match &v_bf16 {
                Some(vb) => {
                    let probs = super::ops::softmax_last_bf16_device(&scores, sk)?;
                    drop(scores);
                    device::matmul_2d_strided_batched_out_view_bf16(
                        &probs,
                        vb,
                        &mut out_view,
                        sq * d,
                        bh,
                        qlen,
                        sk,
                        d,
                    )
                    .map_err(err)?;
                }
                None => {
                    let probs = super::ops::softmax_last_device(&scores, sk)?;
                    drop(scores);
                    device::matmul_2d_strided_batched_out_view(
                        &probs,
                        &vd,
                        &mut out_view,
                        sq * d,
                        bh,
                        qlen,
                        sk,
                        d,
                    )
                    .map_err(err)?;
                }
            }
            start += qlen;
        }
    }
    Ok(Some(CudaTensor::from_device_slice(out, vec![b, h, sq, d])?))
}

#[cfg(not(feature = "cuda"))]
pub fn device_dense_sdpa(
    _q: &CudaTensor,
    _k: &CudaTensor,
    _v: &CudaTensor,
    _scale: Option<f32>,
) -> Result<Option<CudaTensor>> {
    Ok(None)
}

#[cfg(test)]
mod tests {
    use super::super::sdpa::{sdpa, Plan};
    use super::*;

    #[test]
    fn block_causal_is_the_models_temporal_mask() {
        use fastvideo_models::wan::{causal_temporal_mask, WanVideoArchConfig};
        // Windows of whole blocks (and none): the kernel's predicate is
        // FastVideo's blockwise mask.
        for (window, fpb) in [(-1i32, 3usize), (3, 3), (6, 3), (-1, 1), (1, 1), (2, 1)] {
            let mut cfg = WanVideoArchConfig::sf_wan_t2v_1_3b();
            cfg.local_attn_size = window;
            cfg.num_frames_per_block = fpb;
            let (frames, h, w) = (9, 6, 4);
            let want = causal_temporal_mask(&cfg, frames, h, w);
            let spec = block_causal_for(&cfg, (h / 2) * (w / 2)).expect("whole blocks");
            let seq = frames * (h / 2) * (w / 2);
            assert_eq!(spec.dense_mask(seq, seq), want, "window {window} fpb {fpb}");
        }
        let mut cfg = WanVideoArchConfig::sf_wan_t2v_1_3b();
        cfg.local_attn_size = 4;
        assert!(
            block_causal_for(&cfg, 6).is_none(),
            "4 frames is not whole 3-frame blocks"
        );
    }

    #[test]
    fn host_block_causal_matches_the_masked_composed_sdpa() {
        let (b, h, s, d) = (1usize, 2usize, 24usize, 8usize);
        let v = |n: usize, k: f32| {
            (0..n)
                .map(|i| (i as f32 * k).sin() * 0.9)
                .collect::<Vec<_>>()
        };
        let q = CudaTensor::from_vec(v(b * h * s * d, 0.37), vec![b, h, s, d]).unwrap();
        let k = CudaTensor::from_vec(v(b * h * s * d, 0.71), vec![b, h, s, d]).unwrap();
        let vv = CudaTensor::from_vec(v(b * h * s * d, 1.13), vec![b, h, s, d]).unwrap();
        let spec = BlockCausal {
            block_tokens: 5,
            window: 2,
            sink: 1,
        };
        let got = sdpa(&q, &k, &vv, &Plan::block_causal(None, spec)).unwrap();
        let mask = CudaTensor::from_vec(spec.dense_mask(s, s), vec![1, 1, s, s]).unwrap();
        let want = sdpa(&q, &k, &vv, &Plan::masked(None, Some(&mask))).unwrap();
        assert_eq!(got.host_cow().unwrap(), want.host_cow().unwrap());
        // The first query (block 0) sees only block 0: its output is the
        // softmax-weighted mean of the first five value rows alone.
        let masked_first = got.host_cow().unwrap()[..d].to_vec();
        let dense = sdpa(&q, &k, &vv, &Plan::dense(None)).unwrap();
        assert_ne!(masked_first, dense.host_cow().unwrap()[..d].to_vec());
    }

    #[test]
    fn mma_sdpa_supports_video_self_and_cross_attention_shapes() {
        // Self-attention with a tail, cross-attention (sq != sk), d=64 audio.
        assert!(mma_sdpa_supported(1, 40, 38_001, 38_001, 128, 12));
        assert!(mma_sdpa_supported(2, 32, 4_097, 1_024, 128, 8));
        assert!(mma_sdpa_supported(1, 32, 125, 1, 64, 9));
        assert!(mma_sdpa_supported(1, 1, 1, 1, 128, 8));
    }

    #[test]
    fn mma_sdpa_declines_what_the_kernel_cannot_run() {
        assert!(
            !mma_sdpa_supported(1, 8, 1024, 1024, 128, 7),
            "sm75: no bf16 mma"
        );
        for d in [32, 80, 96, 256, 384] {
            assert!(!mma_sdpa_supported(1, 8, 1024, 1024, d, 12), "d={d}");
        }
        assert!(!mma_sdpa_supported(1, 8, 0, 1024, 128, 12), "empty query");
        assert!(!mma_sdpa_supported(1, 8, 1024, 0, 128, 12), "empty keys");
        assert!(
            !mma_sdpa_supported(0, 8, 1024, 1024, 128, 12),
            "empty batch"
        );
        assert!(
            !mma_sdpa_supported(2, 40_000, 64, 64, 128, 12),
            "grid.y > 65535"
        );
    }

    /// Off-device the fused path never claims a call, so CPU runs keep the
    /// host oracle.
    #[test]
    fn mma_sdpa_is_not_default_without_a_device() {
        assert!(!mma_sdpa_default());
        let q = CudaTensor::from_vec(vec![0.5; 2 * 64], vec![1, 1, 2, 64]).unwrap();
        assert!(device_mma_sdpa(&q, &q, &q, None, false).unwrap().is_none());
    }

    fn psnr_db(actual: &[f32], reference: &[f32]) -> f64 {
        let peak = reference.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
        let range = f64::from(2.0 * peak.max(1.0));
        let mut mse = 0.0f64;
        for (&a, &r) in actual.iter().zip(reference) {
            let (a, r) = (f64::from(a), f64::from(r));
            mse += (a - r) * (a - r);
        }
        mse /= actual.len() as f64;
        if mse == 0.0 {
            return f64::INFINITY;
        }
        10.0 * ((range * range) / mse).log10()
    }

    #[test]
    fn dense_bf16_act_psnr_vs_f32() {
        let q = CudaTensor::from_vec(
            (0..24).map(|x| (x as f32) * 0.13 - 0.8).collect(),
            vec![1, 2, 3, 4],
        )
        .unwrap();
        let f32_out = crate::rt::tensor::with_bf16_act(false, || {
            sdpa(&q, &q, &q, &Plan::dense(None)).unwrap()
        });
        let bf16_out = crate::rt::tensor::with_bf16_act(true, || {
            sdpa(&q, &q, &q, &Plan::dense(None)).unwrap()
        });
        assert_eq!(bf16_out.dtype(), crate::rt::tensor::TensorDType::Bf16);
        let p = psnr_db(&bf16_out.host_cow().unwrap(), &f32_out.host_cow().unwrap());
        assert!(p >= 35.0, "sdpa bf16-act PSNR {p:.2} dB < 35");
    }
}
