//! Device-shaped Sol-Attn / PISA / SLA stages.
//!
//! Sol-Attn runs the fused kernels (`sol_prep_*` + `sol_mma_fwd`, see
//! [`super::ops::sol_fused_device`]) whenever a live resident CUDA device
//! exists; their host reference is
//! [`fastvideo_models::sol_attn::sol_attn_head_faithful`]. The PISA / SLA
//! helpers below are host twins of the multi-launch partial kernels in
//! `// ==== region: sol ====`. Without a device the caller keeps the
//! `host_algorithm`-guarded models-crate oracle.

use fastvideo_models::pisa_attn::{keep_for_sparsity, score_blocks, topk_mask};
use fastvideo_models::sol_attn::{
    block_len, num_blocks, pool_means, pool_sums, SolParams, BLOCK_SIZE,
};

use super::tensor::{CudaTensor, Result, TensorError};

fn msg(s: impl Into<String>) -> TensorError {
    TensorError::Message(s.into())
}

/// Sentinel padding unused exact-list slots (matches the CUDA kernels).
pub const SENTINEL: u32 = 0xFFFF_FFFF;

/// Sequential 64-token VSA-compatible plan: slot `b*64+j` is token `b*64+j`.
pub fn sequential_plan(tokens: usize) -> (Vec<i32>, Vec<u32>) {
    let n = num_blocks(tokens);
    let mut slot_src = vec![-1i32; n * BLOCK_SIZE];
    let mut block_sizes = vec![0u32; n];
    for b in 0..n {
        let len = block_len(tokens, b);
        block_sizes[b] = len as u32;
        for j in 0..len {
            slot_src[b * BLOCK_SIZE + j] = (b * BLOCK_SIZE + j) as i32;
        }
    }
    (slot_src, block_sizes)
}

#[derive(Clone, Debug)]
struct Partials {
    m: Vec<f32>,
    l: Vec<f32>,
    acc: Vec<f32>,
}

impl Partials {
    fn empty(rows: usize, dim: usize) -> Self {
        Self {
            m: vec![f32::NEG_INFINITY; rows],
            l: vec![0.0; rows],
            acc: vec![0.0; rows * dim],
        }
    }
}

#[inline]
fn pow2(x: f32) -> f32 {
    2.0f32.powf(x)
}

/// Online-softmax fold. `log2` uses `2^x` (Sol); otherwise `e^x` (PISA / SLA).
fn fold(
    acc: &mut [f32],
    row_sum: &mut f32,
    row_max: &mut f32,
    score: f32,
    weight: f32,
    val: &[f32],
    log2: bool,
) {
    let new_max = row_max.max(score);
    let (alpha, p) = if log2 {
        (pow2(*row_max - new_max), pow2(score - new_max))
    } else {
        ((*row_max - new_max).exp(), (score - new_max).exp())
    };
    for (d, &v) in acc.iter_mut().zip(val) {
        *d = *d * alpha + p * v;
    }
    *row_sum = *row_sum * alpha + p * weight;
    *row_max = new_max;
}

fn lse_merge(a: &Partials, b: &Partials, dim: usize, log2: bool) -> (Partials, Vec<f32>) {
    let rows = a.m.len();
    let mut out = Partials::empty(rows, dim);
    for i in 0..rows {
        let (m1, l1, m2, l2) = (a.m[i], a.l[i], b.m[i], b.l[i]);
        let m = m1.max(m2);
        let (s1, s2) = if log2 {
            (pow2(m1 - m), pow2(m2 - m))
        } else {
            ((m1 - m).exp(), (m2 - m).exp())
        };
        let s1 = if m1.is_finite() { s1 } else { 0.0 };
        let s2 = if m2.is_finite() { s2 } else { 0.0 };
        out.m[i] = m;
        out.l[i] = l1 * s1 + l2 * s2;
        let (aa, ba, oa) = (
            &a.acc[i * dim..(i + 1) * dim],
            &b.acc[i * dim..(i + 1) * dim],
            &mut out.acc[i * dim..(i + 1) * dim],
        );
        for d in 0..dim {
            oa[d] = aa[d] * s1 + ba[d] * s2;
        }
    }
    let mut y = vec![0.0f32; rows * dim];
    for i in 0..rows {
        if out.l[i] > 0.0 {
            let inv = 1.0 / out.l[i];
            for d in 0..dim {
                y[i * dim + d] = out.acc[i * dim + d] * inv;
            }
        }
    }
    (out, y)
}

fn pool_head_means(x: &[f32], tokens: usize, dim: usize) -> Vec<f32> {
    pool_means(x, tokens, dim)
}

fn pool_head_sums(x: &[f32], tokens: usize, dim: usize) -> Vec<f32> {
    pool_sums(x, tokens, dim)
}

fn is_exact(list: &[u32], n: usize, qblock: usize, kv: u32) -> bool {
    let row = &list[qblock * n..(qblock + 1) * n];
    row.contains(&kv)
}

#[allow(clippy::too_many_arguments)]
fn fine_partials(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    lists: &[u32],
    tokens: usize,
    dim: usize,
    scale: f32,
    log2: bool,
) -> Partials {
    let n = num_blocks(tokens);
    let mut p = Partials::empty(tokens, dim);
    for i in 0..n {
        let q_len = block_len(tokens, i);
        let q_start = i * BLOCK_SIZE;
        for t in 0..q_len {
            let qi = q_start + t;
            let qr = &q[qi * dim..(qi + 1) * dim];
            let acc = &mut p.acc[qi * dim..(qi + 1) * dim];
            for &raw in &lists[i * n..(i + 1) * n] {
                if raw == SENTINEL {
                    break;
                }
                let j = raw as usize;
                let k_start = j * BLOCK_SIZE;
                let k_len = block_len(tokens, j);
                for u in 0..k_len {
                    let mut s = 0.0f32;
                    let ku = &k[(k_start + u) * dim..(k_start + u + 1) * dim];
                    for d in 0..dim {
                        s += qr[d] * ku[d];
                    }
                    s *= scale;
                    fold(
                        acc,
                        &mut p.l[qi],
                        &mut p.m[qi],
                        s,
                        1.0,
                        &v[(k_start + u) * dim..(k_start + u + 1) * dim],
                        log2,
                    );
                }
            }
        }
    }
    p
}

#[allow(clippy::too_many_arguments)]
fn coarse_partials(
    q: &[f32],
    kc: &[f32],
    vc: &[f32],
    lists: &[u32],
    tokens: usize,
    dim: usize,
    scale: f32,
    log2: bool,
) -> Partials {
    let n = num_blocks(tokens);
    let mut p = Partials::empty(tokens, dim);
    for i in 0..n {
        let q_len = block_len(tokens, i);
        let q_start = i * BLOCK_SIZE;
        for t in 0..q_len {
            let qi = q_start + t;
            let qr = &q[qi * dim..(qi + 1) * dim];
            let acc = &mut p.acc[qi * dim..(qi + 1) * dim];
            for j in 0..n {
                if is_exact(lists, n, i, j as u32) {
                    continue;
                }
                let mut s = 0.0f32;
                let kcj = &kc[j * dim..(j + 1) * dim];
                for d in 0..dim {
                    s += qr[d] * kcj[d];
                }
                s *= scale;
                fold(
                    acc,
                    &mut p.l[qi],
                    &mut p.m[qi],
                    s,
                    block_len(tokens, j) as f32,
                    &vc[j * dim..(j + 1) * dim],
                    log2,
                );
            }
        }
    }
    p
}

fn global_h_bar(k: &[f32], v: &[f32], tokens: usize, dim: usize) -> Vec<f32> {
    fastvideo_models::pisa_attn::global_h_bar(k, v, tokens, dim)
}

fn apply_first_order(
    q: &[f32],
    h_bar: &[f32],
    coarse: &Partials,
    merged: &mut Partials,
    tokens: usize,
    dim: usize,
    log2: bool,
) {
    for t in 0..tokens {
        let tail = if log2 {
            let s = if coarse.m[t].is_finite() {
                pow2(coarse.m[t] - merged.m[t])
            } else {
                0.0
            };
            coarse.l[t] * s
        } else {
            let s = if coarse.m[t].is_finite() {
                (coarse.m[t] - merged.m[t]).exp()
            } else {
                0.0
            };
            coarse.l[t] * s
        };
        if tail <= 0.0 {
            continue;
        }
        let qi = &q[t * dim..(t + 1) * dim];
        let acc = &mut merged.acc[t * dim..(t + 1) * dim];
        for d in 0..dim {
            let mut qh = 0.0f32;
            for e in 0..dim {
                qh += qi[e] * h_bar[e * dim + d];
            }
            acc[d] += tail * qh;
        }
    }
}

/// Device-shaped PISA: VSA-style top-k selection, zeroth-order remainder,
/// plus the paper Phase-3 first-order term.
#[allow(clippy::too_many_arguments)]
pub fn pisa_attn_bhsd_device_alg(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    batch: usize,
    heads: usize,
    tokens: usize,
    dim: usize,
    sparsity: f64,
    scale: f32,
) -> Result<Vec<f32>> {
    let want = batch * heads * tokens * dim;
    if q.len() != want || k.len() != want || v.len() != want {
        return Err(msg(format!(
            "pisa device-alg: q/k/v want {want}, got {}/{}/{}",
            q.len(),
            k.len(),
            v.len()
        )));
    }
    let n = num_blocks(tokens);
    let keep = keep_for_sparsity(n, sparsity);
    let stride = tokens * dim;
    let mut out = vec![0.0f32; want];
    for bh in 0..batch * heads {
        let base = bh * stride;
        let qh = &q[base..base + stride];
        let kh = &k[base..base + stride];
        let vh = &v[base..base + stride];
        let q_bar = pool_head_means(qh, tokens, dim);
        let kc = pool_head_means(kh, tokens, dim);
        let vc = pool_head_sums(vh, tokens, dim);
        let scores = score_blocks(&q_bar, &kc, n, dim, scale);
        let mask = topk_mask(&scores, n, keep);
        let mut lists = vec![SENTINEL; n * n];
        for i in 0..n {
            let mut w = 0usize;
            for j in 0..n {
                if mask[i * n + j] {
                    lists[i * n + w] = j as u32;
                    w += 1;
                }
            }
        }
        let coarse = coarse_partials(qh, &kc, &vc, &lists, tokens, dim, scale, false);
        let fine = fine_partials(qh, kh, vh, &lists, tokens, dim, scale, false);
        let (mut merged, _) = lse_merge(&coarse, &fine, dim, false);
        let h_bar = global_h_bar(kh, vh, tokens, dim);
        apply_first_order(qh, &h_bar, &coarse, &mut merged, tokens, dim, false);
        for t in 0..tokens {
            if merged.l[t] > 0.0 {
                let inv = 1.0 / merged.l[t];
                for d in 0..dim {
                    out[base + t * dim + d] = merged.acc[t * dim + d] * inv;
                }
            }
        }
    }
    Ok(out)
}

/// Fine-only exact-block attention (SLA sparse branch). `lists` is
/// `[nq * max_keep]` with [`SENTINEL`] padding; `blk_q` / `blk_k` are the
/// query and key block sizes.
#[allow(clippy::too_many_arguments)]
pub fn sla_sparse_bhsd(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    lists: &[u32],
    batch: usize,
    heads: usize,
    tokens: usize,
    dim: usize,
    nq: usize,
    max_keep: usize,
    blk_q: usize,
    blk_k: usize,
    scale: f32,
) -> Result<Vec<f32>> {
    let want = batch * heads * tokens * dim;
    if q.len() != want || k.len() != want || v.len() != want {
        return Err(msg("sla sparse: buffer length mismatch"));
    }
    if lists.len() != batch * heads * nq * max_keep {
        return Err(msg("sla sparse: list length mismatch"));
    }
    let mut out = vec![0.0f32; want];
    let stride = tokens * dim;
    for bh in 0..batch * heads {
        let base = bh * stride;
        let list_base = bh * nq * max_keep;
        for qi in 0..nq {
            let q0 = qi * blk_q;
            let q1 = (q0 + blk_q).min(tokens);
            for t in q0..q1 {
                let qr = &q[base + t * dim..base + (t + 1) * dim];
                let mut acc = vec![0.0f32; dim];
                let mut row_sum = 0.0f32;
                let mut row_max = f32::NEG_INFINITY;
                for &raw in &lists[list_base + qi * max_keep..list_base + (qi + 1) * max_keep] {
                    if raw == SENTINEL {
                        break;
                    }
                    let k0 = raw as usize * blk_k;
                    let k1 = (k0 + blk_k).min(tokens);
                    for u in k0..k1 {
                        let mut s = 0.0f32;
                        let ku = &k[base + u * dim..base + (u + 1) * dim];
                        for d in 0..dim {
                            s += qr[d] * ku[d];
                        }
                        fold(
                            &mut acc,
                            &mut row_sum,
                            &mut row_max,
                            s * scale,
                            1.0,
                            &v[base + u * dim..base + (u + 1) * dim],
                            false,
                        );
                    }
                }
                if row_sum > 0.0 {
                    let inv = 1.0 / row_sum;
                    for d in 0..dim {
                        out[base + t * dim + d] = acc[d] * inv;
                    }
                }
            }
        }
    }
    Ok(out)
}

/// Run Sol (diag thresholds) on device when a live resident CUDA context
/// exists. `sinks` are `(start, len)` token spans; the fused kernel takes one
/// contiguous sink range, so several spans must merge into one
/// ([`fastvideo_models::sol_attn::merge_sink_spans`]) or this errors.
pub fn try_sol_device(
    q: &CudaTensor,
    k: &CudaTensor,
    v: &CudaTensor,
    tau: f32,
    scale: f32,
    sinks: &[(usize, usize)],
) -> Result<Option<CudaTensor>> {
    #[cfg(feature = "cuda")]
    {
        if super::stats::device_expected() {
            let tokens = q.shape.get(2).copied().unwrap_or(0);
            let (sink_start, sink_tokens) =
                fastvideo_models::sol_attn::merge_sink_spans(tokens, sinks).map_err(msg)?;
            let p = SolParams {
                sink_start,
                sink_tokens,
                ..SolParams::diag(tau, scale)
            };
            return Ok(Some(sol_attn_cuda(q, k, v, &p)?));
        }
    }
    let _ = (q, k, v, tau, scale, sinks);
    Ok(None)
}

/// [`try_sol_device`] with explicit [`SolParams`] (threshold type, single
/// Python-style sink span).
pub fn try_sol_device_params(
    q: &CudaTensor,
    k: &CudaTensor,
    v: &CudaTensor,
    p: &SolParams,
) -> Result<Option<CudaTensor>> {
    #[cfg(feature = "cuda")]
    {
        if super::stats::device_expected() {
            return Ok(Some(sol_attn_cuda(q, k, v, p)?));
        }
    }
    let _ = (q, k, v, p);
    Ok(None)
}

/// Run PISA on device when a live resident CUDA context exists.
pub fn try_pisa_device(
    q: &CudaTensor,
    k: &CudaTensor,
    v: &CudaTensor,
    sparsity: f64,
    scale: f32,
) -> Result<Option<CudaTensor>> {
    #[cfg(feature = "cuda")]
    {
        if super::stats::device_expected() {
            return Ok(Some(pisa_attn_cuda(q, k, v, sparsity, scale)?));
        }
    }
    let _ = (q, k, v, sparsity, scale);
    Ok(None)
}

/// SLA sparse+linear on device when `blk_k == 64` and a device is live.
pub fn try_sla_device(
    q: &CudaTensor,
    k: &CudaTensor,
    v: &CudaTensor,
    proj_l: Option<&super::nn::Linear>,
    topk_ratio: f32,
    blk_q: usize,
    blk_k: usize,
) -> Result<Option<CudaTensor>> {
    #[cfg(feature = "cuda")]
    {
        if super::stats::device_expected() && blk_k == BLOCK_SIZE {
            return Ok(Some(sla_attn_cuda(
                q, k, v, proj_l, topk_ratio, blk_q, blk_k,
            )?));
        }
    }
    let _ = (q, k, v, proj_l, topk_ratio, blk_q, blk_k);
    Ok(None)
}

#[cfg(feature = "cuda")]
fn ensure_dev(t: &CudaTensor) -> Result<super::tensor::DevRef<'_>> {
    t.dev()?
        .ok_or_else(|| msg("sol device path: tensor is host-only"))
}

#[cfg(feature = "cuda")]
fn sol_attn_cuda(
    q: &CudaTensor,
    k: &CudaTensor,
    v: &CudaTensor,
    p: &SolParams,
) -> Result<CudaTensor> {
    let (batch, heads, tokens, dim) = (q.shape[0], q.shape[1], q.shape[2], q.shape[3]);
    let qd = ensure_dev(q)?;
    let kd = ensure_dev(k)?;
    let vd = ensure_dev(v)?;
    let out = super::ops::sol_fused_device(&qd, &kd, &vd, batch * heads, tokens, dim, p)?;
    CudaTensor::from_dev_result(out, q.shape.clone())
}

#[cfg(feature = "cuda")]
fn pisa_attn_cuda(
    q: &CudaTensor,
    k: &CudaTensor,
    v: &CudaTensor,
    sparsity: f64,
    scale: f32,
) -> Result<CudaTensor> {
    let (batch, heads, tokens, dim) = (q.shape[0], q.shape[1], q.shape[2], q.shape[3]);
    let qd = ensure_dev(q)?;
    let kd = ensure_dev(k)?;
    let vd = ensure_dev(v)?;
    let out =
        super::ops::pisa_attn_device(&qd, &kd, &vd, batch, heads, tokens, dim, sparsity, scale)?;
    CudaTensor::from_dev_result(out, q.shape.clone())
}

#[cfg(feature = "cuda")]
fn sla_attn_cuda(
    q: &CudaTensor,
    k: &CudaTensor,
    v: &CudaTensor,
    proj_l: Option<&super::nn::Linear>,
    topk_ratio: f32,
    blk_q: usize,
    blk_k: usize,
) -> Result<CudaTensor> {
    let (batch, heads, tokens, dim) = (q.shape[0], q.shape[1], q.shape[2], q.shape[3]);
    let cfg = super::sla::SlaConfig {
        topk_ratio,
        blk_q,
        blk_k,
    };
    let k_score = sla_smooth_k(k)?;
    let qd = ensure_dev(q)?;
    let ksd = ensure_dev(&k_score)?;
    let kd = ensure_dev(k)?;
    let vd = ensure_dev(v)?;
    let o_s = super::ops::sla_sparse_device(&qd, &ksd, &kd, &vd, batch, heads, tokens, dim, &cfg)?;
    let o_s_t = CudaTensor::from_dev_result(o_s, q.shape.clone())?;
    let o_l = sla_linear_device(q, k, v)?;
    let o_l = if let Some(p) = proj_l {
        let flat = o_l.reshape(vec![batch * heads * tokens, dim])?;
        p.forward(&flat)?.reshape(vec![batch, heads, tokens, dim])?
    } else {
        o_l
    };
    o_s_t.add(&o_l)
}

#[cfg(feature = "cuda")]
fn sla_smooth_k(k: &CudaTensor) -> Result<CudaTensor> {
    let (b, h, s, d) = (k.shape[0], k.shape[1], k.shape[2], k.shape[3]);
    let ones = CudaTensor::from_vec(vec![1.0; b * h * s], vec![b * h, 1, s])?.to_device()?;
    let kr = k.reshape(vec![b * h, s, d])?;
    let mean = ones.matmul(&kr)?.mul_scalar(1.0 / s.max(1) as f32);
    kr.sub(&mean)?.reshape(vec![b, h, s, d])
}

#[cfg(feature = "cuda")]
fn sla_linear_device(q: &CudaTensor, k: &CudaTensor, v: &CudaTensor) -> Result<CudaTensor> {
    let (b, h, s, d) = (q.shape[0], q.shape[1], q.shape[2], q.shape[3]);
    let q_lin = q.softmax(-1)?;
    let k_lin = k.softmax(-1)?;
    let k_r = k_lin.reshape(vec![b * h, s, d])?;
    let k_t = k_r.permute(&[0, 2, 1])?;
    let v_r = v.reshape(vec![b * h, s, d])?;
    let kv = k_t.matmul(&v_r)?;
    let ones = CudaTensor::from_vec(vec![1.0; b * h * s], vec![b * h, 1, s])?.to_device()?;
    let ksum = ones.matmul(&k_r)?;
    let q_r = q_lin.reshape(vec![b * h, s, d])?;
    let num = q_r.matmul(&kv)?;
    let den = q_r.matmul(&ksum.reshape(vec![b * h, d, 1])?)?;
    let den = den.try_add_scalar(1e-5)?;
    num.div(&den)?.reshape(vec![b, h, s, d])
}

#[cfg(test)]
mod tests {
    use super::*;
    use fastvideo_models::pisa_attn::pisa_attn_bhsd;

    fn seeded(n: usize, k: f32) -> Vec<f32> {
        (0..n).map(|i| (i as f32 * k + 0.3).sin()).collect()
    }

    fn max_abs(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max)
    }

    #[test]
    fn sequential_plan_covers_a_partial_tail() {
        let (slots, sizes) = sequential_plan(70);
        assert_eq!(sizes, vec![64, 6]);
        assert_eq!(slots[0], 0);
        assert_eq!(slots[63], 63);
        assert_eq!(slots[64], 64);
        assert_eq!(slots[69], 69);
        assert_eq!(slots[70], -1);
    }

    #[test]
    fn pisa_device_alg_matches_models_oracle() {
        let (b, h, t, d) = (1usize, 2usize, 40usize, 8usize);
        let q = seeded(b * h * t * d, 0.11);
        let k = seeded(b * h * t * d, 0.17);
        let v = seeded(b * h * t * d, 0.23);
        let scale = (d as f32).sqrt().recip();
        let got = pisa_attn_bhsd_device_alg(&q, &k, &v, b, h, t, d, 0.9, scale).unwrap();
        let want = pisa_attn_bhsd(&q, &k, &v, b, h, t, d, 0.9, scale).unwrap();
        let err = max_abs(&got, &want);
        assert!(err < 3e-5, "max abs {err}");
    }

    #[test]
    fn pisa_zero_sparsity_matches_models() {
        let (t, d) = (24usize, 8usize);
        let q = seeded(t * d, 0.11);
        let k = seeded(t * d, 0.17);
        let v = seeded(t * d, 0.23);
        let scale = (d as f32).sqrt().recip();
        let got = pisa_attn_bhsd_device_alg(&q, &k, &v, 1, 1, t, d, 0.0, scale).unwrap();
        let want = pisa_attn_bhsd(&q, &k, &v, 1, 1, t, d, 0.0, scale).unwrap();
        assert!(max_abs(&got, &want) < 2e-5);
    }

    #[test]
    fn try_device_is_none_without_cuda() {
        let t = CudaTensor::from_vec(vec![0.0; 8], vec![1, 1, 2, 4]).unwrap();
        assert!(try_sol_device(&t, &t, &t, 1.0, 1.0, &[]).unwrap().is_none());
        assert!(try_pisa_device(&t, &t, &t, 0.9, 1.0).unwrap().is_none());
    }
}
