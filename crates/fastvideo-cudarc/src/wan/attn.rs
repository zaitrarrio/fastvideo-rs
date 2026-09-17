//! Flash-style / sparse / GPU-dense SDPA helpers for Wan attention.

use std::sync::atomic::AtomicBool;

use super::tensor::{record_device_hit, strict_device_check, CudaTensor, Result, TensorError};

/// Fallback query-chunk size when no CUDA device context is available.
pub const GPU_SDPA_QUERY_CHUNK: usize = 256;

/// Tiled flash attention on-device: O(d) peak memory, no full S×S scores buffer.
/// Kernel: `flash_attn_f32` (NVRTC), one block per (bh, q_i), blockDim = head_dim.
/// Falls back to `None` if head_dim is not a multiple of 32 or CUDA is unavailable.
#[cfg(feature = "cuda")]
pub fn device_flash_sdpa(
    q: &CudaTensor,
    k: &CudaTensor,
    v: &CudaTensor,
    scale: Option<f32>,
) -> Result<Option<CudaTensor>> {
    use super::device;
    use super::resident::residency_enabled;

    if !residency_enabled() || device::global_device().is_none() {
        return Ok(None);
    }
    if q.rank() != 4 || k.rank() != 4 || v.rank() != 4 {
        return Ok(None);
    }
    let (b, h, sq, d) = (q.shape[0], q.shape[1], q.shape[2], q.shape[3]);
    let sk = k.shape[2];
    if k.shape[3] != d || v.shape[2] != sk || v.shape[3] != d || k.shape[0] != b || k.shape[1] != h {
        return Ok(None);
    }
    // Flash kernel requires d to be a multiple of the warp size (32).
    if d == 0 || d % 32 != 0 {
        return Ok(None);
    }
    let scale = scale.unwrap_or(1.0 / (d as f32).sqrt());
    let bh = b * h;
    let Some(dev) = device::global_device() else {
        return Ok(None);
    };
    let mut q = q.clone();
    let mut k = k.clone();
    let mut v = v.clone();
    q.ensure_device()?;
    k.ensure_device()?;
    v.ensure_device()?;
    let Some(q_dev) = q.device_slice() else { return Ok(None); };
    let Some(k_dev) = k.device_slice() else { return Ok(None); };
    let Some(v_dev) = v.device_slice() else { return Ok(None); };

    static FLASH_ONCE: AtomicBool = AtomicBool::new(false);
    super::log::info_once(
        &FLASH_ONCE,
        format_args!(
            "sdpa: GPU flash-tiled B={b} H={h} Sq={sq} Sk={sk} D={d} (O(d) mem)"
        ),
    );

    let mut out_dev = dev
        .stream
        .alloc_zeros::<f32>(bh * sq * d)
        .map_err(|e| TensorError::Message(e.to_string()))?;

    unsafe {
        super::kernels::launch_flash_attn_f32(
            &dev.stream,
            &dev.kernels.flash_attn_f32,
            q_dev,
            k_dev,
            v_dev,
            &mut out_dev,
            bh as i32,
            sq as i32,
            sk as i32,
            d as i32,
            scale,
        )
        .map_err(|e| TensorError::Message(e.to_string()))?;
    }

    record_device_hit("flash-attention");
    Ok(Some(CudaTensor::from_device_slice(out_dev, vec![b, h, sq, d])?))
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

/// Device-resident dense SDPA: strided-batched cuBLAS `Q@K^T` + NVRTC softmax + `P@V`.
/// Returns `None` when CUDA/residency is unavailable (caller falls back).
#[cfg(feature = "cuda")]
pub fn device_dense_sdpa(
    q: &CudaTensor,
    k: &CudaTensor,
    v: &CudaTensor,
    scale: Option<f32>,
) -> Result<Option<CudaTensor>> {
    use super::device;
    use super::ops;
    use super::resident::residency_enabled;

    if !residency_enabled() || device::global_device().is_none() {
        static ONCE: AtomicBool = AtomicBool::new(false);
        super::log::debug_once(
            &ONCE,
            format_args!("sdpa: skip device path (resident={} cuda={})",
                residency_enabled(),
                device::global_device().is_some()),
        );
        return Ok(None);
    }
    if q.rank() != 4 || k.rank() != 4 || v.rank() != 4 {
        return Ok(None);
    }
    let (b, h, sq, d) = (q.shape[0], q.shape[1], q.shape[2], q.shape[3]);
    let sk = k.shape[2];
    if k.shape[3] != d || v.shape[2] != sk || v.shape[3] != d || k.shape[0] != b || k.shape[1] != h
    {
        return Ok(None);
    }
    let scale = scale.unwrap_or(1.0 / (d as f32).sqrt());
    let bh = b * h;
    let Some(dev) = device::global_device() else {
        return Ok(None);
    };
    let chunk = super::hopper::sdpa_query_chunk(dev.sm_major);

    let mut q = q.clone();
    let mut k = k.clone();
    let mut v = v.clone();
    q.ensure_device()?;
    k.ensure_device()?;
    v.ensure_device()?;
    let Some(q_dev) = q.device_slice() else {
        return Ok(None);
    };
    let Some(k_dev) = k.device_slice() else {
        return Ok(None);
    };
    let Some(v_dev) = v.device_slice() else {
        return Ok(None);
    };

    static GPU_ONCE: AtomicBool = AtomicBool::new(false);
    super::log::info_once(
        &GPU_ONCE,
        format_args!(
            "sdpa: device dense B={b} H={h} Sq={sq} Sk={sk} D={d} chunk={chunk} tf32={}",
            dev.tf32
        ),
    );
    let mut out_dev = dev
        .stream
        .alloc_zeros::<f32>(bh * sq * d)
        .map_err(|e| TensorError::Message(e.to_string()))?;

    // Full-sequence path when scores fit: one GemmEx + softmax + GemmEx, no gather copies.
    //
    // Chunk dimension: a single cuBLAS strided-batched launch per (Q@K^T, P@V)
    // covers the entire Sq axis. The `chunk` arg here is the upper bound on the
    // query dimension that still fits in scores = B*H*Sq*Sk * f32. For Hopper
    // 80GB, `chunk` is 1024 (see `super::hopper::sdpa_query_chunk`), so for
    // Wan 1.3B at 480p the full path usually fits when sq ≤ 1024.
    let scores_elems = bh.saturating_mul(sq).saturating_mul(sk);
    let full_ok = sq <= chunk || scores_elems <= 64 * 1024 * 1024;

    if full_ok {
        let mut scores = dev
            .stream
            .alloc_zeros::<f32>(bh * sq * sk)
            .map_err(|e| TensorError::Message(e.to_string()))?;
        device::matmul_linear_wt_strided_batched(q_dev, k_dev, &mut scores, bh, sq, d, sk, scale)
            .map_err(|e| TensorError::Message(e.to_string()))?;
        let Some(probs) = ops::softmax_last_device(&scores, sk) else {
            return Ok(None);
        };
        device::matmul_2d_strided_batched(&probs, v_dev, &mut out_dev, bh, sq, sk, d)
            .map_err(|e| TensorError::Message(e.to_string()))?;
        return Ok(Some(CudaTensor::from_device_slice(
            out_dev,
            vec![b, h, sq, d],
        )?));
    }

    // Chunked path: split Sq into `chunk`-sized slices (Q@K^T + softmax + P@V per
    // chunk), writing directly into `q_dev`'s/`out_dev`'s existing buffers via
    // offset+strided views (`matmul_linear_wt_strided_batched_x_view` /
    // `matmul_2d_strided_batched_out_view`) instead of `memcpy_dtod`-gathering
    // each batch-head's chunk into a freshly-allocated contiguous buffer first
    // and scattering the result back afterward. That gather/scatter was
    // `2 * bh` device-to-device copies *per chunk* for no compute benefit —
    // cuBLAS's strided-batched GEMM already supports an inter-batch stride
    // independent of the per-call row count, so the chunk can be addressed
    // in place. Each chunk now issues exactly 3 kernel/GEMM launches (Q@K^T,
    // softmax, P@V) and zero copies; `scores`/`probs`/`chunk_out` remain
    // fresh per-chunk allocations (their natural size already matches the
    // chunk, so there's nothing to view-offset there).
    let mut start = 0usize;
    while start < sq {
        let qlen = chunk.min(sq - start);
        let q_view = q_dev.slice(start * d..);

        let mut scores = dev
            .stream
            .alloc_zeros::<f32>(bh * qlen * sk)
            .map_err(|e| TensorError::Message(e.to_string()))?;
        device::matmul_linear_wt_strided_batched_x_view(
            &q_view, sq * d, k_dev, &mut scores, bh, qlen, d, sk, scale,
        )
        .map_err(|e| TensorError::Message(e.to_string()))?;

        let Some(probs) = ops::softmax_last_device(&scores, sk) else {
            return Ok(None);
        };

        let mut out_view = out_dev.slice_mut(start * d..);
        device::matmul_2d_strided_batched_out_view(
            &probs, v_dev, &mut out_view, sq * d, bh, qlen, sk, d,
        )
        .map_err(|e| TensorError::Message(e.to_string()))?;
        start += qlen;
    }

    Ok(Some(CudaTensor::from_device_slice(
        out_dev,
        vec![b, h, sq, d],
    )?))
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

/// Online-softmax tiled attention. Prefers GPU dense SDPA; CPU only as last resort.
pub fn flash_style_sdpa(
    q: &CudaTensor,
    k: &CudaTensor,
    v: &CudaTensor,
    scale: Option<f32>,
) -> Result<CudaTensor> {
    if let Some(out) = device_dense_sdpa(q, k, v, scale)? {
        record_device_hit("attention");
        return Ok(out);
    }
    strict_device_check(
        "attention",
        format_args!(
            "flash_style_sdpa host fallback, q.shape={:?} k.shape={:?}",
            q.shape, k.shape
        ),
    )?;
    flash_style_sdpa_host(q, k, v, scale)
}

pub(crate) fn flash_style_sdpa_host(
    q: &CudaTensor,
    k: &CudaTensor,
    v: &CudaTensor,
    scale: Option<f32>,
) -> Result<CudaTensor> {
    static HOST_ONCE: AtomicBool = AtomicBool::new(false);
    super::log::info_once(
        &HOST_ONCE,
        format_args!(
            "sdpa: HOST flash-style B={} H={} Sq={} Sk={} D={}",
            q.shape[0],
            q.shape.get(1).copied().unwrap_or(0),
            q.shape.get(2).copied().unwrap_or(0),
            k.shape.get(2).copied().unwrap_or(0),
            q.shape.get(3).copied().unwrap_or(0),
        ),
    );
    if q.rank() != 4 || k.rank() != 4 || v.rank() != 4 {
        return Err(TensorError::Message("flash sdpa expects BHSD".into()));
    }
    let (b, h, sq, d) = (q.shape[0], q.shape[1], q.shape[2], q.shape[3]);
    let sk = k.shape[2];
    if k.shape[3] != d || v.shape[2] != sk || v.shape[3] != d {
        return Err(TensorError::Message("flash sdpa shape mismatch".into()));
    }
    let scale = scale.unwrap_or(1.0 / (d as f32).sqrt());
    let tile = 64usize.min(sk);
    let mut out = vec![0.0f32; b * h * sq * d];
    let qh = q.host_cow()?;
    let kh = k.host_cow()?;
    let vh = v.host_cow()?;

    for bi in 0..b {
        for hi in 0..h {
            for qi in 0..sq {
                let q_off = (bi * h + hi) * sq * d + qi * d;
                let mut m_i = f32::NEG_INFINITY;
                let mut l_i = 0.0f32;
                let mut acc = vec![0.0f32; d];
                let mut start = 0;
                while start < sk {
                    let len = tile.min(sk - start);
                    let mut scores = vec![0.0f32; len];
                    let mut tile_max = f32::NEG_INFINITY;
                    for tj in 0..len {
                        let k_off = (bi * h + hi) * sk * d + (start + tj) * d;
                        let mut dot = 0.0f32;
                        for t in 0..d {
                            dot += qh[q_off + t] * kh[k_off + t];
                        }
                        let s = dot * scale;
                        scores[tj] = s;
                        tile_max = tile_max.max(s);
                    }
                    let m_new = m_i.max(tile_max);
                    let alpha = if m_i.is_finite() {
                        (m_i - m_new).exp()
                    } else {
                        0.0
                    };
                    for t in 0..d {
                        acc[t] *= alpha;
                    }
                    l_i *= alpha;
                    let mut tile_l = 0.0f32;
                    for tj in 0..len {
                        let p = (scores[tj] - m_new).exp();
                        tile_l += p;
                        let v_off = (bi * h + hi) * sk * d + (start + tj) * d;
                        for t in 0..d {
                            acc[t] += p * vh[v_off + t];
                        }
                    }
                    l_i += tile_l;
                    m_i = m_new;
                    start += len;
                }
                let inv = 1.0 / l_i.max(1e-20);
                let o_off = (bi * h + hi) * sq * d + qi * d;
                for t in 0..d {
                    out[o_off + t] = acc[t] * inv;
                }
            }
        }
    }
    let mut t = CudaTensor::from_vec(out, vec![b, h, sq, d])?;
    let _ = t.ensure_device();
    Ok(t)
}

/// Block-sparse local window attention used when `FASTVIDEO_VSA=1`.
pub fn block_sparse_sdpa(
    q: &CudaTensor,
    k: &CudaTensor,
    v: &CudaTensor,
    scale: Option<f32>,
    window: usize,
) -> Result<CudaTensor> {
    static FORCE_SPARSE_CACHE: super::envflag::CachedBool = super::envflag::CachedBool::new();
    let force_sparse =
        FORCE_SPARSE_CACHE.get_or_init(|| super::envflag::bool_flag("FASTVIDEO_VSA_FORCE_SPARSE", false));
    if !force_sparse {
        if let Some(out) = device_dense_sdpa(q, k, v, scale)? {
            record_device_hit("attention");
            return Ok(out);
        }
    }
    if q.rank() != 4 || k.rank() != 4 || v.rank() != 4 {
        return Err(TensorError::Message("sparse sdpa expects BHSD".into()));
    }
    let (b, h, sq, d) = (q.shape[0], q.shape[1], q.shape[2], q.shape[3]);
    let sk = k.shape[2];
    if k.shape[3] != d || v.shape[2] != sk || v.shape[3] != d {
        return Err(TensorError::Message("sparse sdpa shape mismatch".into()));
    }
    let scale = scale.unwrap_or(1.0 / (d as f32).sqrt());
    let window = window.max(1);
    let mut out = vec![0.0f32; b * h * sq * d];
    let qh = q.host_cow()?;
    let kh = k.host_cow()?;
    let vh = v.host_cow()?;
    let sink = window.min(sk);

    for bi in 0..b {
        for hi in 0..h {
            for qi in 0..sq {
                let q_off = (bi * h + hi) * sq * d + qi * d;
                let mut m_i = f32::NEG_INFINITY;
                let mut l_i = 0.0f32;
                let mut acc = vec![0.0f32; d];
                let local_lo = qi.saturating_sub(window);
                let local_hi = (qi + window + 1).min(sk);
                for j in 0..sk {
                    let in_sink = j < sink;
                    let in_local = j >= local_lo && j < local_hi;
                    if !in_sink && !in_local {
                        continue;
                    }
                    let k_off = (bi * h + hi) * sk * d + j * d;
                    let mut dot = 0.0f32;
                    for t in 0..d {
                        dot += qh[q_off + t] * kh[k_off + t];
                    }
                    let s = dot * scale;
                    let m_new = m_i.max(s);
                    let alpha = if m_i.is_finite() {
                        (m_i - m_new).exp()
                    } else {
                        0.0
                    };
                    for t in 0..d {
                        acc[t] *= alpha;
                    }
                    l_i *= alpha;
                    let p = (s - m_new).exp();
                    l_i += p;
                    let v_off = (bi * h + hi) * sk * d + j * d;
                    for t in 0..d {
                        acc[t] += p * vh[v_off + t];
                    }
                    m_i = m_new;
                }
                let inv = 1.0 / l_i.max(1e-20);
                let o_off = (bi * h + hi) * sq * d + qi * d;
                for t in 0..d {
                    out[o_off + t] = acc[t] * inv;
                }
            }
        }
    }
    let mut t = CudaTensor::from_vec(out, vec![b, h, sq, d])?;
    let _ = t.ensure_device();
    Ok(t)
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::nn::scaled_dot_product_attention;

    #[test]
    fn flash_style_matches_dense_small() {
        let prev = std::env::var("FASTVIDEO_SDPA").ok();
        std::env::set_var("FASTVIDEO_SDPA", "dense");
        super::super::nn::reset_sdpa_backend_cache_for_test();
        let q = CudaTensor::from_vec(
            (0..24).map(|x| (x as f32) * 0.01).collect(),
            vec![1, 2, 3, 4],
        )
        .unwrap();
        let k = q.clone();
        let v = q.clone();
        let dense = scaled_dot_product_attention(&q, &k, &v, None).unwrap();
        let flash = flash_style_sdpa_host(&q, &k, &v, None).unwrap();
        let dh = dense.host_cow().unwrap();
        let fh = flash.host_cow().unwrap();
        for (a, b) in dh.iter().zip(fh.iter()) {
            assert!((a - b).abs() < 1e-4, "{a} vs {b}");
        }
        match prev {
            Some(v) => std::env::set_var("FASTVIDEO_SDPA", v),
            None => std::env::remove_var("FASTVIDEO_SDPA"),
        }
        super::super::nn::reset_sdpa_backend_cache_for_test();
    }
}
