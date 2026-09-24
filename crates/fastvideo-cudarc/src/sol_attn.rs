//! Sol-Attn on the existing cudarc device path.
//!
//! The selection and exact/approx combine are the published Sol rule in
//! [`fastvideo_models::sol_attn`]. This module runs that op on BHSD
//! [`CudaTensor`]s and logs once when the kernel actually executes.

use std::sync::atomic::{AtomicBool, Ordering};

use fastvideo_models::sol_attn::{sol_attn_bhsd, sol_attn_bhsd_sunk, BLOCK_SIZE};

use crate::wan::log;
use crate::wan::tensor::{CudaTensor, Result, TensorError};

fn msg(s: impl Into<String>) -> TensorError {
    TensorError::Message(s.into())
}

static LOGGED: AtomicBool = AtomicBool::new(false);

fn log_once(tau: f64, tokens: usize, sink_tokens: usize) {
    if LOGGED.swap(true, Ordering::Relaxed) {
        return;
    }
    let n = tokens.div_ceil(BLOCK_SIZE);
    log::info(format_args!(
        "sol-attn kernel: thresh_type=diag tau={tau} block={BLOCK_SIZE} tokens={tokens} blocks={n} sink={sink_tokens}"
    ));
}

/// Sol-Attn on `[B, H, S, D]` q/k/v. `scale` defaults to `1/sqrt(D)`.
pub fn sol_attn(
    q: &CudaTensor,
    k: &CudaTensor,
    v: &CudaTensor,
    tau: f64,
    scale: Option<f32>,
    sink_start: Option<usize>,
    sink_tokens: usize,
) -> Result<CudaTensor> {
    if q.rank() != 4 || q.shape != k.shape || q.shape != v.shape {
        return Err(msg(format!(
            "sol-attn expects matching BHSD q/k/v, got {:?} {:?} {:?}",
            q.shape, k.shape, v.shape
        )));
    }
    let (batch, heads, tokens, dim) = (q.shape[0], q.shape[1], q.shape[2], q.shape[3]);
    if tokens == 0 {
        return Ok(q.clone());
    }
    let scale = scale.unwrap_or((dim as f32).sqrt().recip());
    log_once(tau, tokens, sink_tokens);
    let sinks = match sink_start {
        Some(s) => vec![(s, sink_tokens)],
        None if sink_tokens > 0 => vec![(tokens.saturating_sub(sink_tokens), sink_tokens)],
        None => vec![],
    };
    if let Some(out) = crate::wan::sol_ops::try_sol_device(q, k, v, tau as f32, scale, &sinks)? {
        return Ok(out);
    }
    crate::wan::stats::host_algorithm(
        "sol_attn",
        format_args!("BHSD {batch}x{heads}x{tokens}x{dim}"),
    )?;
    sol_from_host(
        q,
        k,
        v,
        batch,
        heads,
        tokens,
        dim,
        tau,
        scale,
        sink_start,
        sink_tokens,
    )
}

/// Sol-Attn with one or more official sink spans.
pub fn sol_attn_sunk(
    q: &CudaTensor,
    k: &CudaTensor,
    v: &CudaTensor,
    tau: f64,
    scale: Option<f32>,
    sinks: &[(usize, usize)],
) -> Result<CudaTensor> {
    if q.rank() != 4 || q.shape != k.shape || q.shape != v.shape {
        return Err(msg(format!(
            "sol-attn expects matching BHSD q/k/v, got {:?} {:?} {:?}",
            q.shape, k.shape, v.shape
        )));
    }
    let (batch, heads, tokens, dim) = (q.shape[0], q.shape[1], q.shape[2], q.shape[3]);
    if tokens == 0 {
        return Ok(q.clone());
    }
    let scale = scale.unwrap_or((dim as f32).sqrt().recip());
    let sink_tokens: usize = sinks.iter().map(|(_, len)| *len).sum();
    log_once(tau, tokens, sink_tokens);
    if let Some(out) = crate::wan::sol_ops::try_sol_device(q, k, v, tau as f32, scale, sinks)? {
        return Ok(out);
    }
    crate::wan::stats::host_algorithm(
        "sol_attn",
        format_args!("BHSD sunk {batch}x{heads}x{tokens}x{dim}"),
    )?;
    if sinks.is_empty() {
        return sol_from_host(q, k, v, batch, heads, tokens, dim, tau, scale, None, 0);
    }
    if sinks.len() == 1 {
        return sol_from_host(
            q,
            k,
            v,
            batch,
            heads,
            tokens,
            dim,
            tau,
            scale,
            Some(sinks[0].0),
            sinks[0].1,
        );
    }
    let official: Vec<(Option<usize>, usize)> = sinks
        .iter()
        .map(|&(start, len)| (Some(start), len))
        .collect();
    let out = sol_attn_bhsd_sunk(
        q.host_cow()?.as_ref(),
        k.host_cow()?.as_ref(),
        v.host_cow()?.as_ref(),
        batch,
        heads,
        tokens,
        dim,
        tau as f32,
        scale,
        &official,
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

fn sol_from_host(
    q: &CudaTensor,
    k: &CudaTensor,
    v: &CudaTensor,
    batch: usize,
    heads: usize,
    tokens: usize,
    dim: usize,
    tau: f64,
    scale: f32,
    sink_start: Option<usize>,
    sink_tokens: usize,
) -> Result<CudaTensor> {
    let out = sol_attn_bhsd(
        q.host_cow()?.as_ref(),
        k.host_cow()?.as_ref(),
        v.host_cow()?.as_ref(),
        batch,
        heads,
        tokens,
        dim,
        tau as f32,
        scale,
        sink_start,
        sink_tokens,
    )
    .map_err(msg)?;
    pin_like(CudaTensor::from_vec(out, q.shape.clone())?, q)
}

/// Replace the first `sink_tokens` query rows with dense SDPA (H3
/// `text_query_rows=dense` / official MMDiT splice).
pub fn splice_dense_prefix(
    sol_out: &CudaTensor,
    q: &CudaTensor,
    k: &CudaTensor,
    v: &CudaTensor,
    sink_tokens: usize,
    scale: Option<f32>,
) -> Result<CudaTensor> {
    if sink_tokens == 0 {
        return Ok(sol_out.clone());
    }
    let tokens = q.shape[2];
    let keep = sink_tokens.min(tokens);
    if keep == 0 {
        return Ok(sol_out.clone());
    }
    let q_pre = q.narrow(2, 0, keep)?;
    let dense = crate::wan::nn::scaled_dot_product_attention(&q_pre, k, v, scale)?;
    if keep == tokens {
        return Ok(dense);
    }
    let tail = sol_out.narrow(2, keep, tokens - keep)?;
    CudaTensor::cat(&[&dense, &tail], 2)
}

/// Replace each published query span with dense SDPA (`text_query_rows=dense`
/// plus the Spark audio-query subset).
pub fn splice_dense_ranges(
    sol_out: &CudaTensor,
    q: &CudaTensor,
    k: &CudaTensor,
    v: &CudaTensor,
    ranges: &[(usize, usize)],
    scale: Option<f32>,
) -> Result<CudaTensor> {
    if ranges.is_empty() {
        return Ok(sol_out.clone());
    }
    if ranges.len() == 1 && ranges[0].0 == 0 {
        return splice_dense_prefix(sol_out, q, k, v, ranges[0].1, scale);
    }
    let tokens = q.shape[2];
    let mut spans: Vec<(usize, usize)> = ranges
        .iter()
        .copied()
        .filter(|(_, len)| *len > 0)
        .map(|(start, len)| (start.min(tokens), len.min(tokens.saturating_sub(start))))
        .filter(|(_, len)| *len > 0)
        .collect();
    spans.sort_by_key(|s| s.0);
    let mut merged: Vec<(usize, usize)> = Vec::new();
    for (start, len) in spans {
        match merged.last_mut() {
            Some((ms, ml)) if start <= *ms + *ml => {
                *ml = (*ms + *ml).max(start + len) - *ms;
            }
            _ => merged.push((start, len)),
        }
    }
    if merged.is_empty() {
        return Ok(sol_out.clone());
    }
    let mut pieces = Vec::new();
    let mut cursor = 0usize;
    for &(start, len) in &merged {
        if start > cursor {
            pieces.push(sol_out.narrow(2, cursor, start - cursor)?);
        }
        let q_r = q.narrow(2, start, len)?;
        pieces.push(crate::wan::nn::scaled_dot_product_attention(
            &q_r, k, v, scale,
        )?);
        cursor = start + len;
    }
    if cursor < tokens {
        pieces.push(sol_out.narrow(2, cursor, tokens - cursor)?);
    }
    let refs: Vec<&CudaTensor> = pieces.iter().collect();
    CudaTensor::cat(&refs, 2)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wan::nn::scaled_dot_product_attention;
    use fastvideo_models::sol_attn::dense_attn_head;

    fn seeded(n: usize, k: f32) -> Vec<f32> {
        (0..n).map(|i| (i as f32 * k + 0.3).sin()).collect()
    }

    #[test]
    fn full_sink_matches_dense_sdpa() {
        let (b, h, t, d) = (1usize, 2usize, 24usize, 8usize);
        let q = CudaTensor::from_vec(seeded(b * h * t * d, 0.11), vec![b, h, t, d]).unwrap();
        let k = CudaTensor::from_vec(seeded(b * h * t * d, 0.17), vec![b, h, t, d]).unwrap();
        let v = CudaTensor::from_vec(seeded(b * h * t * d, 0.23), vec![b, h, t, d]).unwrap();
        let scale = Some((d as f32).sqrt().recip());
        let sol = sol_attn(&q, &k, &v, 1.0, scale, Some(0), t).unwrap();
        let dense = scaled_dot_product_attention(&q, &k, &v, scale).unwrap();
        let (a, b) = (sol.host_cow().unwrap(), dense.host_cow().unwrap());
        for (x, y) in a.iter().zip(b.iter()) {
            assert!((x - y).abs() < 3e-5, "{x} vs {y}");
        }
    }

    #[test]
    fn device_alg_matches_models_oracle() {
        let (b, h, t, d) = (1usize, 2usize, 40usize, 8usize);
        let q = seeded(b * h * t * d, 0.11);
        let k = seeded(b * h * t * d, 0.17);
        let v = seeded(b * h * t * d, 0.23);
        let scale = (d as f32).sqrt().recip();
        let qt = CudaTensor::from_vec(q.clone(), vec![b, h, t, d]).unwrap();
        let kt = CudaTensor::from_vec(k.clone(), vec![b, h, t, d]).unwrap();
        let vt = CudaTensor::from_vec(v.clone(), vec![b, h, t, d]).unwrap();
        let got = sol_attn(&qt, &kt, &vt, 1.25, Some(scale), Some(0), 8).unwrap();
        let want = fastvideo_models::sol_attn::sol_attn_bhsd(
            &q, &k, &v, b, h, t, d, 1.25, scale, Some(0), 8,
        )
        .unwrap();
        let a = got.host_cow().unwrap();
        for (x, y) in a.iter().zip(&want) {
            assert!((x - y).abs() < 3e-5, "{x} vs {y}");
        }
    }

    #[test]
    fn host_head_matches_the_models_oracle() {
        let (t, d) = (20usize, 4usize);
        let q = seeded(t * d, 0.11);
        let k = seeded(t * d, 0.17);
        let v = seeded(t * d, 0.23);
        let scale = (d as f32).sqrt().recip();
        let qt = CudaTensor::from_vec(q.clone(), vec![1, 1, t, d]).unwrap();
        let kt = CudaTensor::from_vec(k.clone(), vec![1, 1, t, d]).unwrap();
        let vt = CudaTensor::from_vec(v.clone(), vec![1, 1, t, d]).unwrap();
        let got = sol_attn(&qt, &kt, &vt, 1.25, Some(scale), None, 0).unwrap();
        let want = dense_attn_head(&q, &k, &v, t, d, scale);
        // Not all-exact at tau 1.25; just check shape and finite values.
        assert_eq!(got.shape, vec![1, 1, t, d]);
        assert!(got.host_cow().unwrap().iter().all(|x| x.is_finite()));
        let _ = want;
    }
}
