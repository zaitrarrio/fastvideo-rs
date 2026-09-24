//! Resident-encoder attention: GQA without `repeat_kv`, and chunked dense SDPA
//! that applies the causal/padding mask without a full composed score matrix.
//!
//! Device path reuses the same strided-batched GEMMs as [`crate::wan::attn`].
//! Host path falls back to `repeat_kv` + the existing composed SDPA so CPU
//! tests stay bit-identical.

use crate::wan::nn::scaled_dot_product_attention_masked;
use crate::wan::tensor::{CudaTensor, Result, TensorError};

fn msg(s: impl Into<String>) -> TensorError {
    TensorError::Message(s.into())
}

/// GQA (or MHA) masked attention. `q` is `[B, Hq, Sq, D]`; `k`/`v` are
/// `[B, Hkv, Sk, D]` with `Hq` a multiple of `Hkv`. K/V are not repeated.
pub fn scaled_dot_product_attention_gqa(
    q: &CudaTensor,
    k: &CudaTensor,
    v: &CudaTensor,
    scale: Option<f32>,
    mask: Option<&CudaTensor>,
) -> Result<CudaTensor> {
    #[cfg(feature = "cuda")]
    if let Some(out) = device_gqa_masked(q, k, v, scale, mask)? {
        return Ok(out);
    }
    host_gqa_masked(q, k, v, scale, mask)
}

fn gqa_layout(
    q: &CudaTensor,
    k: &CudaTensor,
    v: &CudaTensor,
) -> Result<(usize, usize, usize, usize, usize, usize)> {
    let [b, hq, sq, d] = match q.shape[..] {
        [b, hq, sq, d] => [b, hq, sq, d],
        _ => return Err(msg(format!("gqa sdpa: q {:?} is not BHSD", q.shape))),
    };
    let [bk, hkv, sk, dk] = match k.shape[..] {
        [bk, hkv, sk, dk] => [bk, hkv, sk, dk],
        _ => return Err(msg(format!("gqa sdpa: k {:?} is not BHSD", k.shape))),
    };
    if [bk, sk, dk] != [b, sk, d] || v.shape != [b, hkv, sk, d] {
        return Err(msg(format!(
            "gqa sdpa: q {:?} k {:?} v {:?}",
            q.shape, k.shape, v.shape
        )));
    }
    if hkv == 0 || hq % hkv != 0 {
        return Err(msg(format!(
            "gqa sdpa: {hq} query heads over {hkv} kv heads"
        )));
    }
    Ok((b, hq, hkv, sq, sk, d))
}

fn host_gqa_masked(
    q: &CudaTensor,
    k: &CudaTensor,
    v: &CudaTensor,
    scale: Option<f32>,
    mask: Option<&CudaTensor>,
) -> Result<CudaTensor> {
    let (_b, hq, hkv, _sq, _sk, _d) = gqa_layout(q, k, v)?;
    let g = hq / hkv;
    let (k, v) = if g == 1 {
        (k.clone(), v.clone())
    } else {
        (k.repeat_kv(g)?, v.repeat_kv(g)?)
    };
    scaled_dot_product_attention_masked(q, &k, &v, scale, mask)
}

/// Chunked `Q@Kᵀ` + mask + softmax + `P@V` over KV-head groups so K/V stay at
/// `Hkv`. `None` when no device buffer is available.
#[cfg(feature = "cuda")]
fn device_gqa_masked(
    q: &CudaTensor,
    k: &CudaTensor,
    v: &CudaTensor,
    scale: Option<f32>,
    mask: Option<&CudaTensor>,
) -> Result<Option<CudaTensor>> {
    let (b, hq, hkv, sq, sk, d) = gqa_layout(q, k, v)?;
    let g = hq / hkv;
    // Unmasked MHA: the Wan kernel already chunks the score buffer.
    if mask.is_none() && g == 1 {
        return crate::wan::attn::device_dense_sdpa(q, k, v, scale);
    }
    let (Some(kd), Some(vd)) = (k.dev()?, v.dev()?) else {
        return Ok(None);
    };
    let scale = scale.unwrap_or(1.0 / (d as f32).sqrt());
    let groups = b * hkv;
    let chunk = (crate::wan::attn::DENSE_SCORE_BUDGET / (b * hq * sk).max(1)).clamp(1, sq.max(1));
    let err = |e: crate::wan::device::DeviceError| msg(e.to_string());
    let mut pieces = Vec::new();
    let mut start = 0usize;
    while start < sq {
        let qlen = chunk.min(sq - start);
        let q_owned;
        let q_use = if start == 0 && qlen == sq {
            q
        } else {
            q_owned = q.narrow(2, start, qlen)?;
            &q_owned
        };
        let Some(qd) = q_use.dev()? else {
            return Ok(None);
        };
        let mut scores = crate::wan::ops::alloc((groups * g * qlen * sk).max(1))?;
        crate::wan::device::matmul_linear_wt_strided_batched(
            &qd,
            &kd,
            &mut scores,
            groups,
            g * qlen,
            d,
            sk,
            scale,
        )
        .map_err(err)?;
        let mut scores = CudaTensor::from_device_slice(scores, vec![b, hq, qlen, sk])?;
        if let Some(m) = mask {
            let q_axis = m
                .rank()
                .checked_sub(2)
                .ok_or_else(|| msg(format!("gqa sdpa: mask rank {} is too small", m.rank())))?;
            scores = scores.add(&m.narrow(q_axis, start, qlen)?)?;
        }
        let probs = scores.softmax(-1)?;
        let Some(pd) = probs.dev()? else {
            return Ok(None);
        };
        let mut out = crate::wan::ops::alloc((groups * g * qlen * d).max(1))?;
        crate::wan::device::matmul_2d_strided_batched(&pd, &vd, &mut out, groups, g * qlen, sk, d)
            .map_err(err)?;
        pieces.push(CudaTensor::from_device_slice(out, vec![b, hq, qlen, d])?);
        start += qlen;
    }
    let refs: Vec<&CudaTensor> = pieces.iter().collect();
    Ok(Some(CudaTensor::cat(&refs, 2)?))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn t(data: Vec<f32>, shape: &[usize]) -> CudaTensor {
        CudaTensor::from_vec(data, shape.to_vec()).unwrap()
    }

    /// Grouped scores must match materializing K/V with `repeat_kv`.
    #[test]
    fn gqa_matches_repeated_kv() {
        let (b, hq, hkv, s, d) = (1usize, 4, 2, 3, 4);
        let q = t(
            (0..b * hq * s * d).map(|i| i as f32 * 0.01 - 0.2).collect(),
            &[b, hq, s, d],
        );
        let k = t(
            (0..b * hkv * s * d)
                .map(|i| i as f32 * 0.02 - 0.1)
                .collect(),
            &[b, hkv, s, d],
        );
        let v = t(
            (0..b * hkv * s * d).map(|i| i as f32 * 0.03).collect(),
            &[b, hkv, s, d],
        );
        let mut mask = vec![f32::MIN; s * s];
        for i in 0..s {
            for j in 0..=i {
                mask[i * s + j] = 0.0;
            }
        }
        let mask = t(mask, &[1, 1, s, s]);
        let scale = Some(0.5);
        let got = scaled_dot_product_attention_gqa(&q, &k, &v, scale, Some(&mask)).unwrap();
        let want = scaled_dot_product_attention_masked(
            &q,
            &k.repeat_kv(hq / hkv).unwrap(),
            &v.repeat_kv(hq / hkv).unwrap(),
            scale,
            Some(&mask),
        )
        .unwrap();
        let (a, b) = (got.host_cow().unwrap(), want.host_cow().unwrap());
        for (x, y) in a.iter().zip(b.iter()) {
            assert!((x - y).abs() < 1e-5, "{x} vs {y}");
        }
    }
}
