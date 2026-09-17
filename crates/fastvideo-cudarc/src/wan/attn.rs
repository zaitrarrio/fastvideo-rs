//! Attention kernels for Wan: device dense (default), device flash (opt-in),
//! and host implementations for CPU runs.

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
        && super::device::global_device().is_some_and(|d| d.gemm_math == super::device::GemmMath::Bf16)
}

fn bhsd(q: &CudaTensor, k: &CudaTensor, v: &CudaTensor) -> Option<(usize, usize, usize, usize, usize)> {
    let [b, h, sq, d] = q.shape[..] else { return None };
    let sk = k.shape.get(2).copied()?;
    (k.shape == [b, h, sk, d] && v.shape == [b, h, sk, d]).then_some((b, h, sq, sk, d))
}

/// Tiled flash attention on-device (`FASTVIDEO_SDPA=flash`). `None` when the
/// head dim is outside the kernel's limits or no device is expected.
#[cfg(feature = "cuda")]
pub fn device_flash_sdpa(q: &CudaTensor, k: &CudaTensor, v: &CudaTensor, scale: Option<f32>) -> Result<Option<CudaTensor>> {
    let Some((b, h, sq, sk, d)) = bhsd(q, k, v) else { return Ok(None) };
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
    super::log::info_once(&ONCE, format_args!("sdpa: GPU flash-tiled B={b} H={h} Sq={sq} Sk={sk} D={d}"));
    let mut out = super::ops::alloc(bh * sq * d)?;
    let (bh_i, sq_i, sk_i, d_i) = (bh as i32, sq as i32, sk as i32, d as i32);
    super::kernels::launch!(dev.stream, &dev.kernels.flash_attn_f32, super::kernels::cfg_flash(bh, sq, d);
        &*qd, &*kd, &*vd, &mut out, &bh_i, &sq_i, &sk_i, &d_i, &scale)
    .map_err(|e| msg(e.to_string()))?;
    Ok(Some(CudaTensor::from_device_slice(out, vec![b, h, sq, d])?))
}

#[cfg(not(feature = "cuda"))]
pub fn device_flash_sdpa(_q: &CudaTensor, _k: &CudaTensor, _v: &CudaTensor, _scale: Option<f32>) -> Result<Option<CudaTensor>> {
    Ok(None)
}

/// Device dense SDPA: strided-batched cuBLAS `Q@Kᵀ` + softmax + `P@V`, with
/// the query axis chunked so the score buffer stays under
/// [`DENSE_SCORE_BUDGET`]. `None` when no device is expected.
#[cfg(feature = "cuda")]
pub fn device_dense_sdpa(q: &CudaTensor, k: &CudaTensor, v: &CudaTensor, scale: Option<f32>) -> Result<Option<CudaTensor>> {
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
    let Some((b, h, sq, sk, d)) = bhsd(q, k, v) else { return Ok(None) };
    let (Some(qd), Some(kd), Some(vd)) = (q.dev()?, k.dev()?, v.dev()?) else {
        return Ok(None);
    };
    let scale = scale.unwrap_or(1.0 / (d as f32).sqrt());
    let bh = b * h;
    let chunk = (score_budget / (bh * sk).max(1)).clamp(1, sq.max(1));
    static ONCE: AtomicBool = AtomicBool::new(false);
    super::log::info_once(&ONCE, format_args!("sdpa: device dense B={b} H={h} Sq={sq} Sk={sk} D={d} query_chunk={chunk}"));
    let err = |e: device::DeviceError| msg(e.to_string());
    let mut out = super::ops::alloc((bh * sq * d).max(1))?;
    // The probability matrix is `bh*sq*sk` — far larger than Q, K, V — so in
    // fast mode it is stored as bf16, halving the dominant traffic. `V` is cast
    // once to match. Exact mode keeps F32 so it stays comparable to the CPU path.
    let v_bf16 = if probs_bf16() { Some(super::ops::cast_f32_bf16_device(&vd)?) } else { None };
    if chunk >= sq {
        let mut scores = super::ops::alloc((bh * sq * sk).max(1))?;
        device::matmul_linear_wt_strided_batched(&qd, &kd, &mut scores, bh, sq, d, sk, scale).map_err(err)?;
        match &v_bf16 {
            Some(vb) => {
                let probs = super::ops::softmax_last_bf16_device(&scores, sk)?;
                drop(scores);
                device::matmul_2d_strided_batched_bf16(&probs, vb, &mut out, bh, sq, sk, d).map_err(err)?;
            }
            None => {
                let probs = super::ops::softmax_last_device(&scores, sk)?;
                drop(scores);
                device::matmul_2d_strided_batched(&probs, &vd, &mut out, bh, sq, sk, d).map_err(err)?;
            }
        }
    } else {
        let mut start = 0usize;
        while start < sq {
            let qlen = chunk.min(sq - start);
            let q_view = qd.slice(start * d..);
            let mut scores = super::ops::alloc(bh * qlen * sk)?;
            device::matmul_linear_wt_strided_batched_x_view(&q_view, sq * d, &kd, &mut scores, bh, qlen, d, sk, scale)
                .map_err(err)?;
            let mut out_view = out.slice_mut(start * d..);
            match &v_bf16 {
                Some(vb) => {
                    let probs = super::ops::softmax_last_bf16_device(&scores, sk)?;
                    drop(scores);
                    device::matmul_2d_strided_batched_out_view_bf16(&probs, vb, &mut out_view, sq * d, bh, qlen, sk, d)
                        .map_err(err)?;
                }
                None => {
                    let probs = super::ops::softmax_last_device(&scores, sk)?;
                    drop(scores);
                    device::matmul_2d_strided_batched_out_view(&probs, &vd, &mut out_view, sq * d, bh, qlen, sk, d)
                        .map_err(err)?;
                }
            }
            start += qlen;
        }
    }
    Ok(Some(CudaTensor::from_device_slice(out, vec![b, h, sq, d])?))
}

#[cfg(not(feature = "cuda"))]
pub fn device_dense_sdpa(_q: &CudaTensor, _k: &CudaTensor, _v: &CudaTensor, _scale: Option<f32>) -> Result<Option<CudaTensor>> {
    Ok(None)
}

/// Online-softmax tiled attention on host (`FASTVIDEO_SDPA=host`, CPU runs).
pub(crate) fn flash_style_sdpa_host(q: &CudaTensor, k: &CudaTensor, v: &CudaTensor, scale: Option<f32>) -> Result<CudaTensor> {
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
            let alpha = if m_i.is_finite() { (m_i - m_new).exp() } else { 0.0 };
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
pub fn block_sparse_sdpa(q: &CudaTensor, k: &CudaTensor, v: &CudaTensor, scale: Option<f32>, window: usize) -> Result<CudaTensor> {
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
            let idx: Vec<usize> = (0..sk).filter(|&j| j < sink || (j >= lo && j < hi)).collect();
            let scores: Vec<f32> = idx
                .iter()
                .map(|&j| (0..d).map(|t| qh[q_off + t] * kh[bh * sk * d + j * d + t]).sum::<f32>() * scale)
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
    use super::*;
    use super::super::nn::scaled_dot_product_attention;

    #[test]
    fn host_flash_matches_composed() {
        let q = CudaTensor::from_vec((0..24).map(|x| (x as f32) * 0.01).collect(), vec![1, 2, 3, 4]).unwrap();
        let dense = scaled_dot_product_attention(&q, &q, &q, None).unwrap();
        let flash = flash_style_sdpa_host(&q, &q, &q, None).unwrap();
        for (a, b) in dense.data.iter().zip(&flash.data) {
            assert!((a - b).abs() < 1e-5, "{a} vs {b}");
        }
    }
}
