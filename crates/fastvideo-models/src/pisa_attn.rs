//! PISA piecewise score-route selection and host reference.
//!
//! Official sources:
//! - paper arXiv:2602.01077 Algorithm 1 (exact selected blocks, block-wise
//!   zeroth-order remainder, Phase-3 global first-order) and the video note
//!   that covariance-aware scoring is image-only
//! - NVlabs/Sana `sol-engine` `techniques/sparse_attention_policies.py`:
//!   `route_mode=score` → `piecewise_score_topk`,
//!   `keep = max(1, min(N, round(N * (1 - sparsity))))`,
//!   `scores = q_bar @ k_bar^T * scale` with `k_var=None`
//! - LTX-2.3 optimized env: sparsity 0.9, block size 64, `approx_remainder=true`
//!
//! This is not random block drop and not Sol-Attn (no tau / diag threshold).

use crate::sol_attn::{block_len, num_blocks, pool_means, pool_sums, BLOCK_SIZE};

/// Official PISA / LTX-2.3 block size.
pub const BLOCK_SIZE_PISA: usize = 64;

/// `keep = max(1, min(n, round(n * density)))` from `_topk_count`.
pub fn topk_count(num_kv_blocks: usize, density: f64) -> usize {
    if num_kv_blocks == 0 {
        return 0;
    }
    let keep = (num_kv_blocks as f64 * density).round() as usize;
    keep.clamp(1, num_kv_blocks)
}

/// Density is `1 - sparsity`.
pub fn keep_for_sparsity(num_kv_blocks: usize, sparsity: f64) -> usize {
    topk_count(num_kv_blocks, (1.0 - sparsity).clamp(0.0, 1.0))
}

/// Block-proxy scores `q_bar @ k_bar^T * scale` for one head.
///
/// `q_bar`/`k_bar` are `[n, dim]`. Returns row-major `[n, n]`.
pub fn score_blocks(q_bar: &[f32], k_bar: &[f32], n: usize, dim: usize, scale: f32) -> Vec<f32> {
    let mut out = vec![0.0f32; n * n];
    for i in 0..n {
        let qi = &q_bar[i * dim..(i + 1) * dim];
        for j in 0..n {
            let mut s = 0.0f32;
            let kj = &k_bar[j * dim..(j + 1) * dim];
            for d in 0..dim {
                s += qi[d] * kj[d];
            }
            out[i * n + j] = s * scale;
        }
    }
    out
}

/// Top-`keep` mask per query block. Ties keep the lower index so the mask is
/// deterministic (matches a stable argsort, not a random drop).
pub fn topk_mask(scores: &[f32], n: usize, keep: usize) -> Vec<bool> {
    let keep = keep.clamp(1, n.max(1));
    let mut mask = vec![false; n * n];
    for i in 0..n {
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&a, &b| {
            scores[i * n + b]
                .partial_cmp(&scores[i * n + a])
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.cmp(&b))
        });
        for &j in order.iter().take(keep) {
            mask[i * n + j] = true;
        }
    }
    mask
}

/// Score-route mask for one `[tokens, dim]` head.
pub fn score_route_mask(
    q: &[f32],
    k: &[f32],
    tokens: usize,
    dim: usize,
    sparsity: f64,
    scale: f32,
) -> Vec<bool> {
    let n = num_blocks(tokens);
    let q_bar = pool_means(q, tokens, dim);
    let k_bar = pool_means(k, tokens, dim);
    let scores = score_blocks(&q_bar, &k_bar, n, dim, scale);
    topk_mask(&scores, n, keep_for_sparsity(n, sparsity))
}

/// Global first-order statistic \(\bar H = \frac{1}{N}\sum_j H_j\).
///
/// Paper: \(H_j:=\sum_n (k_{j,n}-\bar k_j)^\top v_{j,n}\in\mathbb{R}^{d\times d}\).
/// Row-major `[dim, dim]`: row is the key axis, column is the value axis.
pub fn global_h_bar(k: &[f32], v: &[f32], tokens: usize, dim: usize) -> Vec<f32> {
    let n = num_blocks(tokens);
    let kc = pool_means(k, tokens, dim);
    let mut h = vec![0.0f32; dim * dim];
    for j in 0..n {
        let k_start = j * BLOCK_SIZE;
        let k_len = block_len(tokens, j);
        let kj = &kc[j * dim..(j + 1) * dim];
        for u in 0..k_len {
            let ku = &k[(k_start + u) * dim..(k_start + u + 1) * dim];
            let vu = &v[(k_start + u) * dim..(k_start + u + 1) * dim];
            for e in 0..dim {
                let dk = ku[e] - kj[e];
                let row = e * dim;
                for d in 0..dim {
                    h[row + d] += dk * vu[d];
                }
            }
        }
    }
    if n > 0 {
        let inv = 1.0 / n as f32;
        for x in &mut h {
            *x *= inv;
        }
    }
    h
}

/// PISA exact-or-approx for one head: selected blocks exact, remainder
/// zeroth-order plus the paper Phase-3 first-order term
/// \(q_t\bar H\sum_{j\in\mathcal{U}}\alpha_{t,j}\).
pub fn pisa_attn_head(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    tokens: usize,
    dim: usize,
    sparsity: f64,
    scale: f32,
) -> Vec<f32> {
    pisa_attn_head_remainder(q, k, v, tokens, dim, sparsity, scale, true)
}

/// Same selection as [`pisa_attn_head`], but the unselected remainder stops
/// at the block-wise zeroth-order term (paper Phase 2 only).
pub fn pisa_attn_head_zeroth(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    tokens: usize,
    dim: usize,
    sparsity: f64,
    scale: f32,
) -> Vec<f32> {
    pisa_attn_head_remainder(q, k, v, tokens, dim, sparsity, scale, false)
}

fn pisa_attn_head_remainder(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    tokens: usize,
    dim: usize,
    sparsity: f64,
    scale: f32,
    first_order: bool,
) -> Vec<f32> {
    let n = num_blocks(tokens);
    let kc = pool_means(k, tokens, dim);
    let vc = pool_sums(v, tokens, dim);
    let h_bar = if first_order {
        global_h_bar(k, v, tokens, dim)
    } else {
        Vec::new()
    };
    let mask = score_route_mask(q, k, tokens, dim, sparsity, scale);
    let mut out = vec![0.0f32; tokens * dim];
    for i in 0..n {
        let q_len = block_len(tokens, i);
        let q_start = i * BLOCK_SIZE;
        for t in 0..q_len {
            let qi = &q[(q_start + t) * dim..(q_start + t + 1) * dim];
            let mut acc = vec![0.0f32; dim];
            let mut row_sum = 0.0f32;
            let mut row_max = f32::NEG_INFINITY;
            let mut tail = 0.0f32;

            let fold = |acc: &mut [f32],
                        row_sum: &mut f32,
                        row_max: &mut f32,
                        tail: &mut f32,
                        score: f32,
                        weight: f32,
                        val: &[f32],
                        add_tail: bool| {
                let new_max = row_max.max(score);
                let alpha = (*row_max - new_max).exp();
                let p = (score - new_max).exp();
                for d in 0..dim {
                    acc[d] = acc[d] * alpha + p * val[d];
                }
                *row_sum = *row_sum * alpha + p * weight;
                *tail = *tail * alpha + if add_tail { p } else { 0.0 };
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
                s *= scale;
                let len = block_len(tokens, j) as f32;
                fold(
                    &mut acc,
                    &mut row_sum,
                    &mut row_max,
                    &mut tail,
                    s,
                    len,
                    &vc[j * dim..(j + 1) * dim],
                    true,
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
                    s *= scale;
                    fold(
                        &mut acc,
                        &mut row_sum,
                        &mut row_max,
                        &mut tail,
                        s,
                        1.0,
                        &v[(k_start + u) * dim..(k_start + u + 1) * dim],
                        false,
                    );
                }
            }
            if first_order && tail > 0.0 {
                for d in 0..dim {
                    let mut qh = 0.0f32;
                    let col = d;
                    for e in 0..dim {
                        qh += qi[e] * h_bar[e * dim + col];
                    }
                    acc[d] += tail * qh;
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

/// PISA over BHSD row-major `q`/`k`/`v`.
#[allow(clippy::too_many_arguments)]
pub fn pisa_attn_bhsd(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    batch: usize,
    heads: usize,
    tokens: usize,
    dim: usize,
    sparsity: f64,
    scale: f32,
) -> Result<Vec<f32>, String> {
    let want = batch * heads * tokens * dim;
    if q.len() != want || k.len() != want || v.len() != want {
        return Err(format!(
            "pisa-attn: q/k/v want {want} elements, got {}/{}/{}",
            q.len(),
            k.len(),
            v.len()
        ));
    }
    let mut out = vec![0.0f32; want];
    let stride = tokens * dim;
    for bh in 0..batch * heads {
        let base = bh * stride;
        let head = pisa_attn_head(
            &q[base..base + stride],
            &k[base..base + stride],
            &v[base..base + stride],
            tokens,
            dim,
            sparsity,
            scale,
        );
        out[base..base + stride].copy_from_slice(&head);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sol_attn::dense_attn_head;

    fn seeded(n: usize, k: f32) -> Vec<f32> {
        (0..n).map(|i| (i as f32 * k + 0.41).sin()).collect()
    }

    #[test]
    fn topk_count_matches_the_python_round() {
        assert_eq!(topk_count(10, 0.1), 1);
        assert_eq!(keep_for_sparsity(10, 0.9), 1);
        assert_eq!(keep_for_sparsity(64, 0.9), 6);
        assert_eq!(keep_for_sparsity(1, 0.9), 1);
        assert_eq!(keep_for_sparsity(0, 0.9), 0);
    }

    #[test]
    fn score_route_keeps_the_highest_proxy_scores() {
        let n = 3;
        let dim = 1;
        let q_bar = vec![1.0f32, 1.0, 1.0];
        let k_bar = vec![0.0f32, 2.0, 1.0];
        let scores = score_blocks(&q_bar, &k_bar, n, dim, 1.0);
        assert_eq!(scores, vec![0.0, 2.0, 1.0, 0.0, 2.0, 1.0, 0.0, 2.0, 1.0]);
        let mask = topk_mask(&scores, n, 1);
        for i in 0..n {
            assert!(mask[i * n + 1]);
            assert!(!mask[i * n]);
            assert!(!mask[i * n + 2]);
        }
    }

    #[test]
    fn density_one_matches_dense_attention() {
        let (tokens, dim) = (40, 8);
        let q = seeded(tokens * dim, 0.11);
        let k = seeded(tokens * dim, 0.17);
        let v = seeded(tokens * dim, 0.23);
        let scale = (dim as f32).sqrt().recip();
        let pisa = pisa_attn_head(&q, &k, &v, tokens, dim, 0.0, scale);
        let dense = dense_attn_head(&q, &k, &v, tokens, dim, scale);
        for (a, b) in pisa.iter().zip(&dense) {
            assert!((a - b).abs() < 2e-5, "{a} vs {b}");
        }
    }

    #[test]
    fn score_mask_width_is_the_published_keep() {
        let (tokens, dim) = (128, 4);
        let q = seeded(tokens * dim, 0.11);
        let k = seeded(tokens * dim, 0.17);
        let scale = (dim as f32).sqrt().recip();
        let mask = score_route_mask(&q, &k, tokens, dim, 0.9, scale);
        let n = num_blocks(tokens);
        let keep = keep_for_sparsity(n, 0.9);
        for i in 0..n {
            let row = mask[i * n..(i + 1) * n].iter().filter(|b| **b).count();
            assert_eq!(row, keep, "query block {i}");
        }
    }

    #[test]
    fn first_order_matches_the_paper_tail_term() {
        let (tokens, dim) = (8, 2);
        let q = vec![
            1.0f32, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0,
        ];
        let k = vec![
            0.0f32, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 2.0, 0.0, 0.0, 2.0, 2.0, 0.0, 0.0, 2.0,
        ];
        // Constant V: Σ (k − k̄) = 0 ⇒ H_j = 0 for a single block.
        let v = vec![0.5f32; tokens * dim];
        let scale = 1.0;
        let n = num_blocks(tokens);
        let h = global_h_bar(&k, &v, tokens, dim);
        assert_eq!(h.len(), dim * dim);
        assert_eq!(n, 1);
        // One block: H_1 rows sum to 0 because k - k_bar is a mean-zero set.
        for e in 0..dim {
            let mut row = 0.0f32;
            for d in 0..dim {
                row += h[e * dim + d].abs();
            }
            assert!(row < 1e-6, "single-block H_bar should vanish, row {e}");
        }
        let hybrid = pisa_attn_head(&q, &k, &v, tokens, dim, 0.0, scale);
        let zeroth = pisa_attn_head_zeroth(&q, &k, &v, tokens, dim, 0.0, scale);
        for (a, b) in hybrid.iter().zip(&zeroth) {
            assert!((a - b).abs() < 1e-6, "{a} vs {b}");
        }
    }

    #[test]
    fn first_order_moves_sparse_output_off_the_zeroth_term() {
        let (tokens, dim) = (128, 8);
        let q = seeded(tokens * dim, 0.11);
        let k = seeded(tokens * dim, 0.17);
        let v = seeded(tokens * dim, 0.23);
        let scale = (dim as f32).sqrt().recip();
        let hybrid = pisa_attn_head(&q, &k, &v, tokens, dim, 0.9, scale);
        let zeroth = pisa_attn_head_zeroth(&q, &k, &v, tokens, dim, 0.9, scale);
        let drift: f32 = hybrid.iter().zip(&zeroth).map(|(a, b)| (a - b).abs()).sum();
        assert!(
            drift > 1e-4,
            "Phase-3 term should change a sparse remainder"
        );
    }
}
