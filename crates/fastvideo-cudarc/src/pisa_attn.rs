//! PISA piecewise score-route attention on the existing cudarc path.
//!
//! Selection is `techniques/sparse_attention_policies.py` `route_mode=score`
//! (top-k of pooled QK). Unselected blocks use the published zeroth-order
//! remainder. See [`fastvideo_models::pisa_attn`].

use std::sync::atomic::{AtomicBool, Ordering};

use fastvideo_models::pisa_attn::pisa_attn_bhsd;
use fastvideo_models::sol_attn::BLOCK_SIZE;

use crate::wan::log;
use crate::wan::tensor::{CudaTensor, Result, TensorError};

fn msg(s: impl Into<String>) -> TensorError {
    TensorError::Message(s.into())
}

static LOGGED: AtomicBool = AtomicBool::new(false);

fn log_once(sparsity: f64, tokens: usize) {
    if LOGGED.swap(true, Ordering::Relaxed) {
        return;
    }
    log::info(format_args!(
        "pisa kernel: route=score sparsity={sparsity} block={BLOCK_SIZE} tokens={tokens} approx_remainder"
    ));
}

/// PISA on `[B, H, S, D]` q/k/v. `scale` defaults to `1/sqrt(D)`.
pub fn pisa_attn(
    q: &CudaTensor,
    k: &CudaTensor,
    v: &CudaTensor,
    sparsity: f64,
    scale: Option<f32>,
) -> Result<CudaTensor> {
    if q.rank() != 4 || q.shape != k.shape || q.shape != v.shape {
        return Err(msg(format!(
            "pisa-attn expects matching BHSD q/k/v, got {:?} {:?} {:?}",
            q.shape, k.shape, v.shape
        )));
    }
    let (batch, heads, tokens, dim) = (q.shape[0], q.shape[1], q.shape[2], q.shape[3]);
    if tokens == 0 {
        return Ok(q.clone());
    }
    let scale = scale.unwrap_or((dim as f32).sqrt().recip());
    log_once(sparsity, tokens);
    let out = pisa_attn_bhsd(
        q.host_cow()?.as_ref(),
        k.host_cow()?.as_ref(),
        v.host_cow()?.as_ref(),
        batch,
        heads,
        tokens,
        dim,
        sparsity,
        scale,
    )
    .map_err(msg)?;
    pin_like(CudaTensor::from_vec(out, q.shape.clone())?, q)
}

fn pin_like(t: CudaTensor, like: &CudaTensor) -> Result<CudaTensor> {
    #[cfg(feature = "cuda")]
    {
        let mut t = t;
        if like.is_device_fresh() {
            t.pin_device()?;
        }
        return Ok(t);
    }
    let _ = like;
    Ok(t)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wan::nn::scaled_dot_product_attention;

    fn seeded(n: usize, k: f32) -> Vec<f32> {
        (0..n).map(|i| (i as f32 * k + 0.41).sin()).collect()
    }

    #[test]
    fn zero_sparsity_matches_dense_sdpa() {
        let (b, h, t, d) = (1usize, 2usize, 24usize, 8usize);
        let q = CudaTensor::from_vec(seeded(b * h * t * d, 0.11), vec![b, h, t, d]).unwrap();
        let k = CudaTensor::from_vec(seeded(b * h * t * d, 0.17), vec![b, h, t, d]).unwrap();
        let v = CudaTensor::from_vec(seeded(b * h * t * d, 0.23), vec![b, h, t, d]).unwrap();
        let scale = Some((d as f32).sqrt().recip());
        let pisa = pisa_attn(&q, &k, &v, 0.0, scale).unwrap();
        let dense = scaled_dot_product_attention(&q, &k, &v, scale).unwrap();
        let (a, b) = (pisa.host_cow().unwrap(), dense.host_cow().unwrap());
        for (x, y) in a.iter().zip(b.iter()) {
            assert!((x - y).abs() < 3e-5, "{x} vs {y}");
        }
    }
}
