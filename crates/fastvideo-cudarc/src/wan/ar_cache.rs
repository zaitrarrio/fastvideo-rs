//! Causal Wan self-attention through [`fastvideo_models::wan::ArKvCache`].
//!
//! One cache per batch item, filled frame by frame inside a single block
//! forward. Packed slots are the resident form when `FASTVIDEO_NVFP4` is on;
//! the span returned to attention is already dequantized.

use fastvideo_models::nvfp4::ScaleRule;
use fastvideo_models::wan::{ArKvCache, ArKvSpec};

use super::nn;
use super::tensor::{CudaTensor, Result, TensorError};

fn msg(s: impl Into<String>) -> TensorError {
    TensorError::Message(s.into())
}

/// Geometry for one causal self-attention pass.
#[derive(Debug, Clone, Copy)]
pub struct ArFrame {
    pub heads: usize,
    pub dim: usize,
    pub capacity: usize,
    pub sink_tokens: usize,
    pub max_attention: usize,
    pub frame_seqlen: usize,
    pub rule: ScaleRule,
}

/// Chunk `q/k/v` (`[B, H, S, D]`) into frames, append each frame to the
/// rolling cache, and attend that frame's queries against the dequantized span.
pub fn attend_cached(
    q: &CudaTensor,
    k: &CudaTensor,
    v: &CudaTensor,
    spec: &ArFrame,
) -> Result<CudaTensor> {
    if spec.frame_seqlen == 0 {
        return Err(msg("ar kv: frame length is 0"));
    }
    let [b, heads, seq, dim] = q.shape[..] else {
        return Err(msg(format!("ar kv: q shape {:?}", q.shape)));
    };
    if k.shape != q.shape || v.shape != q.shape {
        return Err(msg(format!(
            "ar kv: q {:?} k {:?} v {:?}",
            q.shape, k.shape, v.shape
        )));
    }
    if heads != spec.heads || dim != spec.dim {
        return Err(msg(format!(
            "ar kv: BHSD heads {heads} dim {dim} vs spec {} {}",
            spec.heads, spec.dim
        )));
    }
    let kv_spec = ArKvSpec {
        heads,
        dim,
        capacity: spec.capacity,
        sink_tokens: spec.sink_tokens,
        max_attention: spec.max_attention,
    };
    let mut caches = Vec::with_capacity(b);
    for _ in 0..b {
        caches.push(ArKvCache::open(kv_spec, Some(spec.rule)).map_err(msg)?);
    }
    let mut parts = Vec::new();
    let mut start = 0;
    while start < seq {
        let n = spec.frame_seqlen.min(seq - start);
        let qn = q.narrow(2, start, n)?;
        let kn = k.narrow(2, start, n)?;
        let vn = v.narrow(2, start, n)?;
        let mut k_win = Vec::with_capacity(b);
        let mut v_win = Vec::with_capacity(b);
        for bi in 0..b {
            let kb = kn.narrow(0, bi, 1)?;
            let vb = vn.narrow(0, bi, 1)?;
            let k_shd = shd_from_bhsd(&kb.host_cow()?, heads, n, dim);
            let v_shd = shd_from_bhsd(&vb.host_cow()?, heads, n, dim);
            let (ko, vo) = caches[bi].push(&k_shd, &v_shd).map_err(msg)?;
            let view = ko.len() / (heads * dim);
            k_win.push(pinned_bhsd(
                &bhsd_from_shd(&ko, heads, view, dim),
                heads,
                view,
                dim,
            )?);
            v_win.push(pinned_bhsd(
                &bhsd_from_shd(&vo, heads, view, dim),
                heads,
                view,
                dim,
            )?);
        }
        let k_cat = CudaTensor::cat(&k_win.iter().collect::<Vec<_>>(), 0)?;
        let v_cat = CudaTensor::cat(&v_win.iter().collect::<Vec<_>>(), 0)?;
        parts.push(nn::scaled_dot_product_attention(&qn, &k_cat, &v_cat, None)?);
        start += n;
    }
    CudaTensor::cat(&parts.iter().collect::<Vec<_>>(), 2)
}

fn pinned_bhsd(host: &[f32], heads: usize, seq: usize, dim: usize) -> Result<CudaTensor> {
    let mut t = CudaTensor::from_vec(host.to_vec(), vec![1, heads, seq, dim])?;
    t.pin_device()?;
    Ok(t)
}

fn shd_from_bhsd(host: &[f32], heads: usize, seq: usize, dim: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; seq * heads * dim];
    for h in 0..heads {
        for s in 0..seq {
            let src = (h * seq + s) * dim;
            let dst = (s * heads + h) * dim;
            out[dst..dst + dim].copy_from_slice(&host[src..src + dim]);
        }
    }
    out
}

fn bhsd_from_shd(host: &[f32], heads: usize, seq: usize, dim: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; seq * heads * dim];
    for s in 0..seq {
        for h in 0..heads {
            let src = (s * heads + h) * dim;
            let dst = (h * seq + s) * dim;
            out[dst..dst + dim].copy_from_slice(&host[src..src + dim]);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn two_frames_keep_the_first_in_the_second_span() {
        let (heads, dim, frames, spatial) = (1usize, 16usize, 2usize, 1usize);
        let spec = ArFrame {
            heads,
            dim,
            capacity: 4,
            sink_tokens: 0,
            max_attention: 0,
            frame_seqlen: spatial,
            rule: ScaleRule::Mse,
        };
        let n = frames * spatial;
        let q = CudaTensor::from_vec(vec![0.0; heads * n * dim], vec![1, heads, n, dim]).unwrap();
        let mut k = vec![0.0f32; heads * n * dim];
        let mut v = vec![0.0f32; heads * n * dim];
        for s in 0..n {
            for d in 0..dim {
                k[s * dim + d] = (s + 1) as f32 + d as f32 * 0.01;
                v[s * dim + d] = 0.1 * (s + 1) as f32;
            }
        }
        let k = CudaTensor::from_vec(k, vec![1, heads, n, dim]).unwrap();
        let v = CudaTensor::from_vec(v, vec![1, heads, n, dim]).unwrap();
        let out = attend_cached(&q, &k, &v, &spec).unwrap();
        assert_eq!(out.shape, vec![1, heads, n, dim]);
    }
}
