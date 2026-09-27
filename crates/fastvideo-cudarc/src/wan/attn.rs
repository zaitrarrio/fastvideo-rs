//! Attention kernels for Wan: fused tensor-core dense SDPA (default on bf16
//! sm80+ contexts), cuBLAS dense (fallback / `FASTVIDEO_SDPA=cublas`), device
//! flash (opt-in), and host implementations for CPU runs.

#[cfg(feature = "cuda")]
use std::sync::atomic::AtomicBool;

use super::nn::host_only_op;
use super::tensor::{CudaTensor, Result, TensorError};

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

/// Query/key tile of the fused kernel (one CTA per 64 queries).
pub const MMA_TILE: usize = 64;

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

/// Which fused dense kernel runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FlashKernel {
    /// `flash_mma_fwd_d*`: 64-query CTAs, 4 warps, one K and one V buffer.
    V1,
    /// `flash_mma_fwd2_d*`: 128-query CTAs (8 warps x 16 rows) and
    /// double-buffered K/V; bit-identical to V1 (same per-row arithmetic).
    V2,
    /// `flash_mma_fwd3_d128`: V2's CTA and arithmetic with S_{j+1} issued
    /// before softmax(S_j) and a three-stage K/V ring; bit-identical to V2.
    /// d=128 only (V2 otherwise).
    V3,
    /// `flash_mma_fwd3s_d128`: V3 that skips the O rescale when no row max
    /// rose (the factor V2 applies there is within a few ulp of 1).
    V3s,
    /// cuDNN's fused attention engine (`cudnn_sdpa`), bf16 output only;
    /// V2 whenever cuDNN offers no engine or the caller wants f32 out.
    Cudnn,
}

/// `FASTVIDEO_FLASH_KERNEL=v1|v2|cudnn|auto`. `auto` takes V2 when its
/// 128-query grid fills the GPU for many waves ([`flash_v2_default`] and
/// [`flash_v2_fills`]), V1 otherwise (at LTX 768x512, 6 144 tokens x 32
/// heads, V2's one-CTA-per-SM tail costs 2%).
pub fn flash_kernel_choice() -> FlashKernel {
    flash_kernel_for(usize::MAX, 1, 1)
}

/// [`flash_kernel_choice`] for a `bh` x `sq` grid on `sms` SMs.
pub fn flash_kernel_for(sq: usize, bh: usize, sms: usize) -> FlashKernel {
    // The kernel seam (`[kernels] dense_attention`, else `FASTVIDEO_FLASH_KERNEL`).
    match fastvideo_models::techniques::kernels::choice(fastvideo_models::techniques::kernels::KernelOp::DenseAttention).as_str() {
        "v1" => FlashKernel::V1,
        "v2" => FlashKernel::V2,
        "v3" => FlashKernel::V3,
        "v3s" => FlashKernel::V3s,
        "cudnn" => FlashKernel::Cudnn,
        _ if flash_v2_default() && flash_v2_fills(sq, bh, sms) => FlashKernel::V2,
        _ => FlashKernel::V1,
    }
}

/// Whether V2's grid (one 256-thread CTA per SM) is at least 16 waves.
pub fn flash_v2_fills(sq: usize, bh: usize, sms: usize) -> bool {
    sq.div_ceil(2 * MMA_TILE).saturating_mul(bh) >= 16 * sms.max(1)
}

/// Whether `auto` picks the 128-query double-buffered kernel: yes. It is
/// bit-identical to V1 and 4% faster on RTX PRO 6000 at the H3 / LTX shapes
/// (`attn_bench`: 373 vs 358 TFLOPS). `FASTVIDEO_FLASH_KERNEL=v1` restores V1.
pub fn flash_v2_default() -> bool {
    true
}

/// Whether `auto` hands a bf16-output dense SDPA that V2 would run to
/// cuDNN's unified SDPA node instead (V2 still runs when cuDNN has no plan):
/// on sm_12x only. RTX PRO 6000, cuDNN 9.26, `attn3_bench`: 105.0 vs 108.4 ms
/// at H3 768p, 658.8 vs 682.2 ms at LTX 1080p 20 s, 769.0 vs 759.9 ms at 4K 5 s
/// (engine 11, heuristic mode A config 0). `FASTVIDEO_FLASH_KERNEL=v2`
/// restores flash_mma_fwd2.
pub fn cudnn_default(sm_major: i32) -> bool {
    CUDNN_DEFAULT_ON && sm_major == 12
}

/// Flipped only once a generation run has verified the cuDNN default.
const CUDNN_DEFAULT_ON: bool = false;

/// Fused dense SDPA on tensor cores (`flash_mma_fwd_d{64,128}`): bf16 Q/K/V
/// (cast once, RNE, when they are f32), f32 online softmax, bf16 P, f32
/// accumulation; one launch, no score buffer. Output is f32, or bf16 when
/// `out_bf16`. `None` when [`mma_sdpa_supported`] says no or no device is
/// expected, so the caller falls back to [`device_dense_sdpa`].
#[cfg(feature = "cuda")]
pub fn device_mma_sdpa(
    q: &CudaTensor,
    k: &CudaTensor,
    v: &CudaTensor,
    scale: Option<f32>,
    out_bf16: bool,
) -> Result<Option<CudaTensor>> {
    let kernel = match (bhsd(q, k, v), super::device::global_device()) {
        (Some((b, h, sq, _, _)), Some(dev)) => {
            use cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT;
            let sms = dev
                .ctx
                .attribute(CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
                .map_or(1, |n| n.max(1) as usize);
            match flash_kernel_for(sq, b * h, sms) {
                // `auto` only: an explicit `v2` stays V2.
                FlashKernel::V2
                    if out_bf16
                        && cudnn_default(dev.sm_major)
                        && fastvideo_models::techniques::kernels::choice(
                            fastvideo_models::techniques::kernels::KernelOp::DenseAttention,
                        ) == "auto" =>
                {
                    FlashKernel::Cudnn
                }
                k => k,
            }
        }
        _ => flash_kernel_choice(),
    };
    device_mma_sdpa_with(q, k, v, scale, out_bf16, kernel)
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
        let mut out = unsafe { dev.stream.alloc::<half::bf16>(n) }
            .map_err(|e| msg(e.to_string()))?;
        if super::cudnn_sdpa::sdpa_bf16(&qb, &kb, &vb, &mut out, bh, sq, sk, d, scale)? {
            return Ok(Some(CudaTensor::from_device_slice_bf16(
                out,
                vec![b, h, sq, d],
            )?));
        }
    }
    let kernel = match kernel {
        FlashKernel::Cudnn => FlashKernel::V2,
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
        FlashKernel::V2 | FlashKernel::Cudnn => {
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
/// `[S, S]` additive tensor: query `q` sees key `k` iff
/// `frame(k) <= frame(q)` (`frame(i) = i / frame_tokens`), within `window`
/// frames when `window > 0`, or `frame(k) < sink`. The same predicate as
/// `fastvideo_models::wan::causal_temporal_mask`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BlockCausal {
    pub frame_tokens: usize,
    pub window: usize,
    pub sink: usize,
}

impl BlockCausal {
    pub fn allows(&self, q: usize, k: usize) -> bool {
        let ft = self.frame_tokens.max(1);
        let (tq, tk) = (q / ft, k / ft);
        (tk <= tq && (self.window == 0 || tq - tk <= self.window)) || tk < self.sink
    }

    /// The additive `[sq, sk]` mask (0 visible, `-1e9` masked), as
    /// `causal_temporal_mask` builds it.
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
        || mask.frame_tokens == 0
        || !fits(mask.window)
        || !fits(mask.sink)
        || !fits(sq.max(sk) + mask.frame_tokens)
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
            "sdpa: block-causal mma B={b} H={h} Sq={sq} Sk={sk} D={d} frame={} window={} sink={}",
            mask.frame_tokens, mask.window, mask.sink
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
    let (ft, win, sink) = (mask.frame_tokens as i32, mask.window as i32, mask.sink as i32);
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
        Ok(Some(CudaTensor::from_device_slice_bf16(out, vec![b, h, sq, d])?))
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

/// Online-softmax tiled attention on host (`FASTVIDEO_SDPA=host`, CPU runs).
pub(crate) fn flash_style_sdpa_host(
    q: &CudaTensor,
    k: &CudaTensor,
    v: &CudaTensor,
    scale: Option<f32>,
) -> Result<CudaTensor> {
    let (b, h, sq, sk, d) = bhsd(q, k, v).ok_or_else(|| msg("flash sdpa shape mismatch"))?;
    host_only_op("sdpa_host", format_args!("q={:?} k={:?}", q.shape, k.shape))?;
    let scale = scale.unwrap_or(1.0 / (d as f32).sqrt());
    let tile = 64usize.min(sk.max(1));
    let (qh, kh, vh) = (q.host_cow()?, k.host_cow()?, v.host_cow()?);
    let mut out = vec![0.0f32; b * h * sq * d];
    use rayon::prelude::*;
    out.par_chunks_mut(d).enumerate().for_each(|(row, o)| {
        let (bh, qi) = (row / sq, row % sq);
        let q_off = bh * sq * d + qi * d;
        let mut m_i = f32::NEG_INFINITY;
        let mut l_i = 0.0f32;
        let mut start = 0;
        while start < sk {
            let len = tile.min(sk - start);
            let scores: Vec<f32> = (0..len)
                .map(|tj| {
                    let k_off = bh * sk * d + (start + tj) * d;
                    (0..d).map(|t| qh[q_off + t] * kh[k_off + t]).sum::<f32>() * scale
                })
                .collect();
            let tile_max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let m_new = m_i.max(tile_max);
            let alpha = if m_i.is_finite() {
                (m_i - m_new).exp()
            } else {
                0.0
            };
            o.iter_mut().for_each(|x| *x *= alpha);
            l_i *= alpha;
            for (tj, s) in scores.iter().enumerate() {
                let p = (s - m_new).exp();
                l_i += p;
                let v_off = bh * sk * d + (start + tj) * d;
                for t in 0..d {
                    o[t] += p * vh[v_off + t];
                }
            }
            m_i = m_new;
            start += len;
        }
        let inv = 1.0 / l_i.max(1e-20);
        o.iter_mut().for_each(|x| *x *= inv);
    });
    CudaTensor::from_vec(out, vec![b, h, sq, d])
}

/// Block-sparse local+sink window attention (`FASTVIDEO_VSA=1`). Host only:
/// a GPU run errors instead of computing it on the CPU.
pub fn block_sparse_sdpa(
    q: &CudaTensor,
    k: &CudaTensor,
    v: &CudaTensor,
    scale: Option<f32>,
    window: usize,
) -> Result<CudaTensor> {
    let (b, h, sq, sk, d) = bhsd(q, k, v).ok_or_else(|| msg("sparse sdpa shape mismatch"))?;
    host_only_op("block_sparse_sdpa", format_args!("q={:?}", q.shape))?;
    let scale = scale.unwrap_or(1.0 / (d as f32).sqrt());
    let window = window.max(1);
    let sink = window.min(sk);
    let (qh, kh, vh) = (q.host_cow()?, k.host_cow()?, v.host_cow()?);
    let mut out = vec![0.0f32; b * h * sq * d];
    for bh in 0..b * h {
        for qi in 0..sq {
            let q_off = bh * sq * d + qi * d;
            let (lo, hi) = (qi.saturating_sub(window), (qi + window + 1).min(sk));
            let idx: Vec<usize> = (0..sk)
                .filter(|&j| j < sink || (j >= lo && j < hi))
                .collect();
            let scores: Vec<f32> = idx
                .iter()
                .map(|&j| {
                    (0..d)
                        .map(|t| qh[q_off + t] * kh[bh * sk * d + j * d + t])
                        .sum::<f32>()
                        * scale
                })
                .collect();
            let m = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f32> = scores.iter().map(|s| (s - m).exp()).collect();
            let z: f32 = exps.iter().sum();
            let o = &mut out[q_off..q_off + d];
            for (&j, e) in idx.iter().zip(&exps) {
                for t in 0..d {
                    o[t] += e / z * vh[bh * sk * d + j * d + t];
                }
            }
        }
    }
    CudaTensor::from_vec(out, vec![b, h, sq, d])
}

#[cfg(test)]
mod tests {
    use super::super::nn::scaled_dot_product_attention;
    use super::*;

    #[test]
    fn block_causal_is_the_models_temporal_mask() {
        use fastvideo_models::wan::{causal_temporal_mask, WanVideoArchConfig};
        for (window, sink) in [(-1i32, 0usize), (21, 0), (1, 1), (2, 0), (0, 2)] {
            let mut cfg = WanVideoArchConfig::sf_wan_t2v_1_3b();
            cfg.local_attn_size = window;
            cfg.sink_size = sink;
            let (frames, h, w) = (6, 6, 4);
            let want = causal_temporal_mask(&cfg, frames, h, w);
            let spec = BlockCausal {
                frame_tokens: (h / 2) * (w / 2),
                window: usize::try_from(window).unwrap_or(0),
                sink,
            };
            let seq = frames * spec.frame_tokens;
            assert_eq!(spec.dense_mask(seq, seq), want, "window {window} sink {sink}");
        }
    }

    #[test]
    fn host_block_causal_matches_the_masked_composed_sdpa() {
        let (b, h, s, d) = (1usize, 2usize, 24usize, 8usize);
        let v = |n: usize, k: f32| (0..n).map(|i| ((i as f32 * k).sin() * 0.9)).collect::<Vec<_>>();
        let q = CudaTensor::from_vec(v(b * h * s * d, 0.37), vec![b, h, s, d]).unwrap();
        let k = CudaTensor::from_vec(v(b * h * s * d, 0.71), vec![b, h, s, d]).unwrap();
        let vv = CudaTensor::from_vec(v(b * h * s * d, 1.13), vec![b, h, s, d]).unwrap();
        let spec = BlockCausal {
            frame_tokens: 5,
            window: 2,
            sink: 1,
        };
        let got = super::super::nn::sdpa_block_causal(&q, &k, &vv, None, spec).unwrap();
        let mask = CudaTensor::from_vec(spec.dense_mask(s, s), vec![1, 1, s, s]).unwrap();
        let want =
            super::super::nn::scaled_dot_product_attention_masked(&q, &k, &vv, None, Some(&mask))
                .unwrap();
        assert_eq!(got.host_cow().unwrap(), want.host_cow().unwrap());
        // The first query (frame 0) sees only frame 0: its output is the
        // softmax-weighted mean of the first five value rows alone.
        let masked_first = got.host_cow().unwrap()[..d].to_vec();
        let dense = scaled_dot_product_attention(&q, &k, &vv, None).unwrap();
        assert_ne!(masked_first, dense.host_cow().unwrap()[..d].to_vec());
    }

    #[test]
    fn flash_v2_only_where_its_grid_fills_the_gpu() {
        // RTX PRO 6000: 188 SMs.
        assert!(flash_v2_fills(37_710, 56, 188)); // H3 768p
        assert!(flash_v2_fills(124_440, 32, 188)); // LTX 1080p 20 s stage 2
        assert!(flash_v2_fills(130_560, 32, 188)); // LTX 4K 5 s stage 2
        assert!(!flash_v2_fills(6_144, 32, 188)); // LTX 768x512: V1 wins by 2%
        assert!(!flash_v2_fills(1, 1, 188));
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

    #[test]
    fn host_flash_matches_composed() {
        let q = CudaTensor::from_vec(
            (0..24).map(|x| (x as f32) * 0.01).collect(),
            vec![1, 2, 3, 4],
        )
        .unwrap();
        let dense = scaled_dot_product_attention(&q, &q, &q, None).unwrap();
        let flash = flash_style_sdpa_host(&q, &q, &q, None).unwrap();
        for (a, b) in dense.data.iter().zip(&flash.data) {
            assert!((a - b).abs() < 1e-5, "{a} vs {b}");
        }
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
    fn host_flash_bf16_act_psnr_vs_f32() {
        let q = CudaTensor::from_vec(
            (0..24).map(|x| (x as f32) * 0.13 - 0.8).collect(),
            vec![1, 2, 3, 4],
        )
        .unwrap();
        let f32_out = crate::wan::tensor::with_bf16_act(false, || {
            scaled_dot_product_attention(&q, &q, &q, None).unwrap()
        });
        let bf16_out = crate::wan::tensor::with_bf16_act(true, || {
            scaled_dot_product_attention(&q, &q, &q, None).unwrap()
        });
        assert_eq!(bf16_out.dtype(), crate::wan::tensor::TensorDType::Bf16);
        let p = psnr_db(&bf16_out.host_cow().unwrap(), &f32_out.host_cow().unwrap());
        assert!(p >= 35.0, "sdpa bf16-act PSNR {p:.2} dB < 35");
    }
}
