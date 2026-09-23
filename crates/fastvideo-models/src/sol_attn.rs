//! Sol-Attn selection and host reference.
//!
//! Official sources:
//! - paper arXiv:2607.24027 eqs. (4), (5), (15) and Algorithm 1
//! - NVlabs/Sana `sol-engine` `techniques/sparse_backends/sol_attn/preprocess.py`
//!   (`_compute_diag_threshold`) and `triton_ref/fwd.py`
//! - `sol_attn_route_is_exact` in `techniques/sparse_backends/sol_attn/common/selector.py`
//!
//! Contract used by LTX-2.5 stage-2 and the H3 Sol routes: `thresh_type=diag`,
//! 64-token blocks, local window `|i-j| <= 1` always exact, optional contiguous
//! KV sink. `tau` is the paper's standardized cutoff `β`.
//!
//! This is not VSA: VSA is trained top-k of pooled tiles plus a compression
//! gate. Sol is training-free threshold routing plus zeroth-order reuse of
//! unselected blocks.

/// Official Sol physical block size (`preprocess.py` `BLOCK_SIZE`).
pub const BLOCK_SIZE: usize = 64;

/// Official diag-threshold floor inside the std (`preprocess.py`).
pub const THRESHOLD_EPS: f32 = 1.0e-6;

/// `log2(e)`: official kernels compare scores in log2 space.
pub const LOG2_E: f32 = 1.4426950408889634;

/// How many 64-token blocks cover `tokens`.
pub fn num_blocks(tokens: usize) -> usize {
    tokens.div_ceil(BLOCK_SIZE)
}

/// Live tokens in block `block` of a length-`tokens` sequence.
pub fn block_len(tokens: usize, block: usize) -> usize {
    let start = block * BLOCK_SIZE;
    tokens.saturating_sub(start).min(BLOCK_SIZE)
}

/// Inclusive-exclusive KV-block range that covers `[sink_start, sink_start +
/// sink_tokens)`, matching `interface.py` `_sink_block_range`.
pub fn sink_block_range(
    tokens: usize,
    sink_start: Option<usize>,
    sink_tokens: usize,
) -> (usize, usize) {
    let blocks = num_blocks(tokens);
    if sink_tokens == 0 || tokens == 0 {
        return (blocks, blocks);
    }
    let start = sink_start.unwrap_or(tokens.saturating_sub(sink_tokens));
    (
        start / BLOCK_SIZE,
        (start + sink_tokens + BLOCK_SIZE - 1) / BLOCK_SIZE,
    )
}

/// OR of official `_sink_block_range` over each published span.
pub fn sink_block_flags(tokens: usize, sinks: &[(Option<usize>, usize)]) -> Vec<bool> {
    let n = num_blocks(tokens);
    let mut flags = vec![false; n];
    for &(start, len) in sinks {
        let (lo, hi) = sink_block_range(tokens, start, len);
        for b in lo..hi.min(n) {
            flags[b] = true;
        }
    }
    flags
}

/// Mean-pool one `[tokens, dim]` matrix into `[n_blocks, dim]`.
///
/// The last block divides by its live length, as `_reduce_kc_kernel` does.
pub fn pool_means(x: &[f32], tokens: usize, dim: usize) -> Vec<f32> {
    let n = num_blocks(tokens);
    let mut out = vec![0.0f32; n * dim];
    for b in 0..n {
        let len = block_len(tokens, b);
        let start = b * BLOCK_SIZE;
        let dst = &mut out[b * dim..(b + 1) * dim];
        for t in 0..len {
            let src = &x[(start + t) * dim..(start + t + 1) * dim];
            for (d, &v) in dst.iter_mut().zip(src) {
                *d += v;
            }
        }
        if len > 0 {
            let inv = 1.0 / len as f32;
            for d in dst {
                *d *= inv;
            }
        }
    }
    out
}

/// Sum-pool one `[tokens, dim]` matrix into `[n_blocks, dim]`.
///
/// Official `vc` is a sum (`_reduce_vc_kernel`), not a mean.
pub fn pool_sums(x: &[f32], tokens: usize, dim: usize) -> Vec<f32> {
    let n = num_blocks(tokens);
    let mut out = vec![0.0f32; n * dim];
    for b in 0..n {
        let len = block_len(tokens, b);
        let start = b * BLOCK_SIZE;
        let dst = &mut out[b * dim..(b + 1) * dim];
        for t in 0..len {
            let src = &x[(start + t) * dim..(start + t + 1) * dim];
            for (d, &v) in dst.iter_mut().zip(src) {
                *d += v;
            }
        }
    }
    out
}

/// Diag threshold per query block: paper eq. (15) + official log2 scaling.
///
/// `q_bar` and `kc` are `[n, dim]`. Returns `[n]` thresholds in log2-score
/// units, ready to compare with `sum(q @ kc.T) / q_len * (scale * log2(e))`.
pub fn diag_threshold(
    q_bar: &[f32],
    kc: &[f32],
    n: usize,
    dim: usize,
    tau: f32,
    scale: f32,
) -> Vec<f32> {
    let log2_scale = scale * LOG2_E;
    let mut mu_k = vec![0.0f32; dim];
    let mut var_k = vec![0.0f32; dim];
    for j in 0..n {
        let row = &kc[j * dim..(j + 1) * dim];
        for d in 0..dim {
            mu_k[d] += row[d];
            var_k[d] += row[d] * row[d];
        }
    }
    let inv = 1.0 / n.max(1) as f32;
    for d in 0..dim {
        mu_k[d] *= inv;
        var_k[d] = (var_k[d] * inv - mu_k[d] * mu_k[d]).max(0.0);
    }
    let mut out = vec![0.0f32; n];
    for i in 0..n {
        let q = &q_bar[i * dim..(i + 1) * dim];
        let mut mean = 0.0f32;
        let mut var = 0.0f32;
        for d in 0..dim {
            mean += q[d] * mu_k[d];
            var += q[d] * q[d] * var_k[d];
        }
        mean *= log2_scale;
        var *= log2_scale * log2_scale;
        let std = (var.max(0.0) + THRESHOLD_EPS).sqrt();
        out[i] = mean + tau * std;
    }
    out
}

/// Official exact test: column mean of token-to-block log2-scores, plus the
/// local window, plus the optional sink, plus a valid-block guard.
pub fn route_is_exact(
    q_block: usize,
    kv_block: usize,
    column_mean: f32,
    threshold: f32,
    sink_start_block: usize,
    sink_end_block: usize,
    valid: bool,
) -> bool {
    if !valid {
        return false;
    }
    let local = q_block.abs_diff(kv_block) <= 1;
    let sink = kv_block >= sink_start_block && kv_block < sink_end_block;
    column_mean > threshold || local || sink
}

/// Build the `[n_q, n_k]` exact mask for one head from token `q` and pooled
/// `kc`. Layout: `q` is `[tokens, dim]`, `kc` is `[n_k, dim]`.
pub fn exact_mask(
    q: &[f32],
    kc: &[f32],
    tokens: usize,
    dim: usize,
    tau: f32,
    scale: f32,
    sink_start: Option<usize>,
    sink_tokens: usize,
) -> Vec<bool> {
    exact_mask_sunk(q, kc, tokens, dim, tau, scale, &[(sink_start, sink_tokens)])
}

/// [`exact_mask`] with several official sink spans (H3 text + audio).
pub fn exact_mask_sunk(
    q: &[f32],
    kc: &[f32],
    tokens: usize,
    dim: usize,
    tau: f32,
    scale: f32,
    sinks: &[(Option<usize>, usize)],
) -> Vec<bool> {
    let n = num_blocks(tokens);
    let q_bar = pool_means(q, tokens, dim);
    let threshold = diag_threshold(&q_bar, kc, n, dim, tau, scale);
    let sink_flags = sink_block_flags(tokens, sinks);
    let log2_scale = scale * LOG2_E;
    let mut mask = vec![false; n * n];
    for i in 0..n {
        let q_len = block_len(tokens, i) as f32;
        let q_start = i * BLOCK_SIZE;
        for j in 0..n {
            let mut sum = 0.0f32;
            let kc_j = &kc[j * dim..(j + 1) * dim];
            for t in 0..block_len(tokens, i) {
                let qt = &q[(q_start + t) * dim..(q_start + t + 1) * dim];
                let mut dot = 0.0f32;
                for d in 0..dim {
                    dot += qt[d] * kc_j[d];
                }
                sum += dot * log2_scale;
            }
            let column_mean = if q_len > 0.0 {
                sum / q_len
            } else {
                f32::NEG_INFINITY
            };
            let (sink_lo, sink_hi) = if sink_flags.get(j).copied().unwrap_or(false) {
                (j, j + 1)
            } else {
                (n, n)
            };
            mask[i * n + j] =
                route_is_exact(i, j, column_mean, threshold[i], sink_lo, sink_hi, true);
        }
    }
    mask
}

/// Dense softmax attention for one `[tokens, dim]` head. Oracle for the
/// all-exact Sol case.
pub fn dense_attn_head(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    tokens: usize,
    dim: usize,
    scale: f32,
) -> Vec<f32> {
    let mut out = vec![0.0f32; tokens * dim];
    for i in 0..tokens {
        let qi = &q[i * dim..(i + 1) * dim];
        let mut scores = vec![0.0f32; tokens];
        let mut m = f32::NEG_INFINITY;
        for j in 0..tokens {
            let mut s = 0.0f32;
            let kj = &k[j * dim..(j + 1) * dim];
            for d in 0..dim {
                s += qi[d] * kj[d];
            }
            s *= scale;
            scores[j] = s;
            if s > m {
                m = s;
            }
        }
        let mut z = 0.0f32;
        for s in &mut scores {
            *s = (*s - m).exp();
            z += *s;
        }
        let dst = &mut out[i * dim..(i + 1) * dim];
        for j in 0..tokens {
            let p = scores[j] / z;
            let vj = &v[j * dim..(j + 1) * dim];
            for d in 0..dim {
                dst[d] += p * vj[d];
            }
        }
    }
    out
}

/// Sol-Attn for one head. `q`/`k`/`v` are `[tokens, dim]`.
///
/// Exact blocks use the full token-to-token scores. Unselected blocks reuse
/// the token-to-block scores against pooled keys and summed values, with
/// multiplicity `block_len` in the softmax denominator (paper eqs. 9–10,
/// official Triton forward).
pub fn sol_attn_head(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    tokens: usize,
    dim: usize,
    tau: f32,
    scale: f32,
    sink_start: Option<usize>,
    sink_tokens: usize,
) -> Vec<f32> {
    sol_attn_head_sunk(
        q,
        k,
        v,
        tokens,
        dim,
        tau,
        scale,
        &[(sink_start, sink_tokens)],
    )
}

/// [`sol_attn_head`] with several official sink spans.
pub fn sol_attn_head_sunk(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    tokens: usize,
    dim: usize,
    tau: f32,
    scale: f32,
    sinks: &[(Option<usize>, usize)],
) -> Vec<f32> {
    let n = num_blocks(tokens);
    let kc = pool_means(k, tokens, dim);
    let vc = pool_sums(v, tokens, dim);
    let mask = exact_mask_sunk(q, &kc, tokens, dim, tau, scale, sinks);
    let log2_scale = scale * LOG2_E;
    let mut out = vec![0.0f32; tokens * dim];
    for i in 0..n {
        let q_len = block_len(tokens, i);
        let q_start = i * BLOCK_SIZE;
        for t in 0..q_len {
            let qi = &q[(q_start + t) * dim..(q_start + t + 1) * dim];
            let mut acc = vec![0.0f32; dim];
            let mut row_sum = 0.0f32;
            let mut row_max = f32::NEG_INFINITY;

            let fold = |acc: &mut [f32],
                        row_sum: &mut f32,
                        row_max: &mut f32,
                        score: f32,
                        weight: f32,
                        val: &[f32]| {
                let new_max = row_max.max(score);
                let alpha = 2.0f32.powf(*row_max - new_max);
                let p = 2.0f32.powf(score - new_max);
                for d in 0..dim {
                    acc[d] = acc[d] * alpha + p * val[d];
                }
                *row_sum = *row_sum * alpha + p * weight;
                *row_max = new_max;
            };

            for j in 0..n {
                if mask[i * n + j] {
                    continue;
                }
                let mut s = 0.0f32;
                let kcj = &kc[j * dim..(j + 1) * dim];
                for d in 0..dim {
                    s += qi[d] * kcj[d];
                }
                s *= log2_scale;
                let len = block_len(tokens, j) as f32;
                fold(
                    &mut acc,
                    &mut row_sum,
                    &mut row_max,
                    s,
                    len,
                    &vc[j * dim..(j + 1) * dim],
                );
            }
            for j in 0..n {
                if !mask[i * n + j] {
                    continue;
                }
                let k_start = j * BLOCK_SIZE;
                let k_len = block_len(tokens, j);
                for u in 0..k_len {
                    let mut s = 0.0f32;
                    let ku = &k[(k_start + u) * dim..(k_start + u + 1) * dim];
                    for d in 0..dim {
                        s += qi[d] * ku[d];
                    }
                    s *= log2_scale;
                    fold(
                        &mut acc,
                        &mut row_sum,
                        &mut row_max,
                        s,
                        1.0,
                        &v[(k_start + u) * dim..(k_start + u + 1) * dim],
                    );
                }
            }
            let dst = &mut out[(q_start + t) * dim..(q_start + t + 1) * dim];
            if row_sum > 0.0 {
                for d in 0..dim {
                    dst[d] = acc[d] / row_sum;
                }
            }
        }
    }
    out
}

/// Sol-Attn over BHSD row-major `q`/`k`/`v`.
#[allow(clippy::too_many_arguments)]
pub fn sol_attn_bhsd(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    batch: usize,
    heads: usize,
    tokens: usize,
    dim: usize,
    tau: f32,
    scale: f32,
    sink_start: Option<usize>,
    sink_tokens: usize,
) -> Result<Vec<f32>, String> {
    let want = batch * heads * tokens * dim;
    if q.len() != want || k.len() != want || v.len() != want {
        return Err(format!(
            "sol-attn: q/k/v want {want} elements, got {}/{}/{}",
            q.len(),
            k.len(),
            v.len()
        ));
    }
    let mut out = vec![0.0f32; want];
    let stride = tokens * dim;
    for bh in 0..batch * heads {
        let base = bh * stride;
        let head = sol_attn_head(
            &q[base..base + stride],
            &k[base..base + stride],
            &v[base..base + stride],
            tokens,
            dim,
            tau,
            scale,
            sink_start,
            sink_tokens,
        );
        out[base..base + stride].copy_from_slice(&head);
    }
    Ok(out)
}

/// [`sol_attn_bhsd`] with several official sink spans.
#[allow(clippy::too_many_arguments)]
pub fn sol_attn_bhsd_sunk(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    batch: usize,
    heads: usize,
    tokens: usize,
    dim: usize,
    tau: f32,
    scale: f32,
    sinks: &[(Option<usize>, usize)],
) -> Result<Vec<f32>, String> {
    let want = batch * heads * tokens * dim;
    if q.len() != want || k.len() != want || v.len() != want {
        return Err(format!(
            "sol-attn: q/k/v want {want} elements, got {}/{}/{}",
            q.len(),
            k.len(),
            v.len()
        ));
    }
    let mut out = vec![0.0f32; want];
    let stride = tokens * dim;
    for bh in 0..batch * heads {
        let base = bh * stride;
        let head = sol_attn_head_sunk(
            &q[base..base + stride],
            &k[base..base + stride],
            &v[base..base + stride],
            tokens,
            dim,
            tau,
            scale,
            sinks,
        );
        out[base..base + stride].copy_from_slice(&head);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn seeded(n: usize, k: f32) -> Vec<f32> {
        (0..n).map(|i| (i as f32 * k + 0.3).sin()).collect()
    }

    #[test]
    fn sink_range_matches_the_python_divmod() {
        assert_eq!(sink_block_range(200, None, 0), (4, 4));
        assert_eq!(sink_block_range(200, Some(0), 64), (0, 1));
        assert_eq!(sink_block_range(200, Some(0), 65), (0, 2));
        assert_eq!(sink_block_range(200, None, 10), (2, 4));
    }

    #[test]
    fn last_block_mean_uses_the_live_length() {
        let x = vec![2.0f32, 4.0, 6.0, 8.0];
        let means = pool_means(&x, 3, 1);
        assert_eq!(num_blocks(3), 1);
        assert!((means[0] - (2.0 + 4.0 + 6.0) / 3.0).abs() < 1e-6);
        let sums = pool_sums(&x, 3, 1);
        assert!((sums[0] - 12.0).abs() < 1e-6);
    }

    #[test]
    fn diag_threshold_is_mean_plus_tau_std_in_log2_space() {
        let dim = 2;
        let q_bar = vec![1.0f32, 0.0, 0.0, 1.0];
        let kc = vec![1.0f32, 0.0, 3.0, 0.0];
        let scale = 1.0f32;
        let tau = 1.0f32;
        let got = diag_threshold(&q_bar, &kc, 2, dim, tau, scale);
        let log2_scale = scale * LOG2_E;
        let mu = [2.0f32, 0.0];
        let var = [1.0f32, 0.0];
        let mean0 = (1.0 * mu[0] + 0.0 * mu[1]) * log2_scale;
        let var0 = (1.0 * var[0] + 0.0 * var[1]) * log2_scale * log2_scale;
        let want0 = mean0 + tau * (var0.max(0.0) + THRESHOLD_EPS).sqrt();
        assert!((got[0] - want0).abs() < 1e-5, "{} vs {want0}", got[0]);
    }

    #[test]
    fn local_window_and_sink_are_always_exact() {
        assert!(route_is_exact(3, 3, -100.0, 0.0, 10, 10, true));
        assert!(route_is_exact(3, 4, -100.0, 0.0, 10, 10, true));
        assert!(route_is_exact(3, 2, -100.0, 0.0, 10, 10, true));
        assert!(!route_is_exact(3, 6, -100.0, 0.0, 10, 10, true));
        assert!(route_is_exact(3, 0, -100.0, 0.0, 0, 2, true));
        assert!(!route_is_exact(3, 6, -100.0, 0.0, 0, 2, false));
        assert!(route_is_exact(3, 6, 1.0, 0.5, 10, 10, true));
    }

    #[test]
    fn larger_tau_keeps_fewer_or_equal_exact_blocks() {
        let (tokens, dim) = (128, 4);
        let q = seeded(tokens * dim, 0.11);
        let k = seeded(tokens * dim, 0.17);
        let kc = pool_means(&k, tokens, dim);
        let scale = (dim as f32).sqrt().recip();
        let lo = exact_mask(&q, &kc, tokens, dim, 0.5, scale, None, 0);
        let hi = exact_mask(&q, &kc, tokens, dim, 3.0, scale, None, 0);
        let (n_lo, n_hi) = (
            lo.iter().filter(|b| **b).count(),
            hi.iter().filter(|b| **b).count(),
        );
        assert!(n_hi <= n_lo, "tau 3.0 kept {n_hi}, tau 0.5 kept {n_lo}");
        let n = num_blocks(tokens);
        for i in 0..n {
            for j in 0..n {
                if i.abs_diff(j) <= 1 {
                    assert!(lo[i * n + j] && hi[i * n + j], "local window {i},{j}");
                }
            }
        }
    }

    #[test]
    fn a_full_sink_matches_dense_attention() {
        let (tokens, dim) = (40, 8);
        let q = seeded(tokens * dim, 0.13);
        let k = seeded(tokens * dim, 0.19);
        let v = seeded(tokens * dim, 0.23);
        let scale = (dim as f32).sqrt().recip();
        let sol = sol_attn_head(&q, &k, &v, tokens, dim, 1.0, scale, Some(0), tokens);
        let dense = dense_attn_head(&q, &k, &v, tokens, dim, scale);
        for (a, b) in sol.iter().zip(&dense) {
            assert!((a - b).abs() < 2e-5, "{a} vs {b}");
        }
    }

    #[test]
    fn bhsd_shape_is_checked() {
        assert!(sol_attn_bhsd(&[0.0], &[0.0], &[0.0], 1, 1, 2, 2, 1.0, 1.0, None, 0).is_err());
    }

    #[test]
    fn two_official_spans_or_into_sink_blocks() {
        let flags = sink_block_flags(200, &[(Some(0), 10), (Some(128), 10)]);
        assert!(flags[0]);
        assert!(!flags[1]);
        assert!(flags[2]);
        assert!(!flags[3]);
    }
}
