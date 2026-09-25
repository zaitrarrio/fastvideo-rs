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
pub const LOG2_E: f32 = std::f32::consts::LOG2_E;

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
        (start + sink_tokens).div_ceil(BLOCK_SIZE),
    )
}

/// OR of official `_sink_block_range` over each published span.
pub fn sink_block_flags(tokens: usize, sinks: &[(Option<usize>, usize)]) -> Vec<bool> {
    let n = num_blocks(tokens);
    let mut flags = vec![false; n];
    for &(start, len) in sinks {
        let (lo, hi) = sink_block_range(tokens, start, len);
        for flag in flags.iter_mut().take(hi.min(n)).skip(lo) {
            *flag = true;
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
#[allow(clippy::too_many_arguments)]
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
#[allow(clippy::too_many_arguments)]
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
#[allow(clippy::too_many_arguments)]
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

// ---------------------------------------------------------------------------
// Reference-faithful oracle.
//
// The functions above are the "ideal" f32 Sol-Attn. The ones below reproduce
// the Python reference (`sol_attn` sm120 CuTe path + `preprocess.py`) at its
// rounding points, and are what the fused device kernels are checked
// against:
// - Q/K/V rounded to bf16 (round-to-nearest-even) before anything else;
// - `Kc = bf16(sum / len)`, `Vc = bf16(sum)` over live rows (f32 sums);
// - diag stats from bf16 Kc; exact: bf16 `q_bar`, `M = bf16(bf16(KcᵀKc) / NT)`;
// - route scores `Q·Kc` raw, `col_mean = sum_live_rows(S) * sl2 / len`;
// - per 64-block route group: approximate fold, then exact folds ascending;
// - `P` rounded to bf16 before it multiplies V / Vc, while `l` accumulates
//   the unrounded f32 `p` (weighted by `len(j)` for pooled columns);
// - `out = O / l`, `lse = (m * sl2 + log2 l) * ln 2` (natural log).
// With `bf16_faithful = false` every rounding step is the identity, which is
// the ideal math with the reference's group order, exact thresholds and LSE.

/// Blocks per route group of the sm120 kernel (`M = N = 64`).
pub const ROUTE_GROUP: usize = 64;

/// `thresh_type` of the Python interface.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum SolThresh {
    /// Diagonal covariance (`_compute_diag_threshold`); LTX-2 / H3 / Wan.
    #[default]
    Diag,
    /// Full covariance (`_compute_exact_threshold`).
    Exact,
}

impl SolThresh {
    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "diag" => Some(Self::Diag),
            "exact" => Some(Self::Exact),
            _ => None,
        }
    }
}

/// Numerics of the host oracle.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SolNumerics {
    /// Round at the Python reference's bf16 points (see the section comment).
    pub bf16_faithful: bool,
}

/// One Sol-Attn call: `tau`, softmax `scale`, threshold type and the single
/// contiguous sink span `(sink_start, sink_tokens)` of the Python interface
/// (`sink_start = None` is the suffix sink).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SolParams {
    pub tau: f32,
    pub scale: f32,
    pub thresh: SolThresh,
    pub sink_start: Option<usize>,
    pub sink_tokens: usize,
}

impl SolParams {
    /// `diag`, no sink.
    pub fn diag(tau: f32, scale: f32) -> Self {
        Self {
            tau,
            scale,
            thresh: SolThresh::Diag,
            sink_start: None,
            sink_tokens: 0,
        }
    }

    /// `scale * log2(e)` in f32, as passed to the device kernels.
    pub fn scale_log2(&self) -> f32 {
        self.scale * LOG2_E
    }
}

/// Oracle output for one head, plus routing diagnostics.
#[derive(Clone, Debug)]
pub struct SolHeadOut {
    /// `[tokens, dim]`, f32 (not rounded to bf16).
    pub out: Vec<f32>,
    /// `[tokens]` natural-log LSE of the compensated row sum.
    pub lse: Vec<f32>,
    /// `[n, n]` exact mask, query block major.
    pub mask: Vec<bool>,
    /// `[n, n]` route column means in log2-score units.
    pub col_mean: Vec<f32>,
    /// `[n]` per-query-block thresholds.
    pub threshold: Vec<f32>,
}

/// f32 -> bf16 bits, round-to-nearest-even (`torch.Tensor.to(bfloat16)`);
/// NaN stays NaN. Same bits as the device `sol_bf16_rn`.
pub fn bf16_bits(x: f32) -> u16 {
    let u = x.to_bits();
    if u & 0x7F80_0000 == 0x7F80_0000 {
        let mut h = u >> 16;
        if u & 0x007F_FFFF != 0 {
            h |= 0x0040;
        }
        return h as u16;
    }
    let r = u.wrapping_add(0x7FFF + ((u >> 16) & 1));
    (r >> 16) as u16
}

/// bf16 bits -> f32.
pub fn bf16_to_f32(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}

/// `x` rounded to the nearest bf16 (ties to even).
pub fn bf16_round(x: f32) -> f32 {
    bf16_to_f32(bf16_bits(x))
}

#[inline]
fn rounder(numerics: SolNumerics) -> fn(f32) -> f32 {
    if numerics.bf16_faithful {
        bf16_round
    } else {
        std::convert::identity
    }
}

/// f32 dot product with f64 accumulation (a tensor-core MMA accumulates in
/// f32 in an unspecified order; f64 keeps the oracle order-independent).
#[inline]
fn dot64(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = [0.0f64; 8];
    let chunks = a.len() / 8;
    for c in 0..chunks {
        for l in 0..8 {
            acc[l] += a[c * 8 + l] as f64 * b[c * 8 + l] as f64;
        }
    }
    let mut tail = 0.0f64;
    for i in chunks * 8..a.len() {
        tail += a[i] as f64 * b[i] as f64;
    }
    (((acc[0] + acc[1]) + (acc[2] + acc[3])) + ((acc[4] + acc[5]) + (acc[6] + acc[7])) + tail)
        as f32
}

/// Round every element (no-op when not faithful).
pub fn round_all(x: &[f32], numerics: SolNumerics) -> Vec<f32> {
    let r = rounder(numerics);
    x.iter().map(|&v| r(v)).collect()
}

/// Pooled `Kc` (mean over live rows) and `Vc` (sum), `[n, dim]` each, from
/// already-rounded `k`/`v`. f32 sequential sums then one rounding, exactly as
/// the device `sol_prep_kv` does.
pub fn pool_kv_faithful(
    k: &[f32],
    v: &[f32],
    tokens: usize,
    dim: usize,
    numerics: SolNumerics,
) -> (Vec<f32>, Vec<f32>) {
    let r = rounder(numerics);
    let n = num_blocks(tokens);
    let mut kc = vec![0.0f32; n * dim];
    let mut vc = vec![0.0f32; n * dim];
    for b in 0..n {
        let len = block_len(tokens, b);
        let start = b * BLOCK_SIZE;
        for d in 0..dim {
            let (mut sk, mut sv) = (0.0f32, 0.0f32);
            for t in 0..len {
                sk += k[(start + t) * dim + d];
                sv += v[(start + t) * dim + d];
            }
            kc[b * dim + d] = r(sk / len as f32);
            vc[b * dim + d] = r(sv);
        }
    }
    (kc, vc)
}

/// Diag statistics of `kc` `[n, dim]`: `(mu, var)`, `var = max(E[x²] - mu², 0)`.
/// Same operation order as the device `sol_prep_kstats` (fused `x*x + s2`).
pub fn kc_stats(kc: &[f32], n: usize, dim: usize) -> (Vec<f32>, Vec<f32>) {
    let mut mu = vec![0.0f32; dim];
    let mut var = vec![0.0f32; dim];
    let nf = n.max(1) as f32;
    for d in 0..dim {
        let (mut s, mut s2) = (0.0f32, 0.0f32);
        for j in 0..n {
            let x = kc[j * dim + d];
            s += x;
            s2 = x.mul_add(x, s2);
        }
        let m = s / nf;
        mu[d] = m;
        var[d] = (s2 / nf - m * m).max(0.0);
    }
    (mu, var)
}

/// `M = round(round(KcᵀKc) / n)` `[dim, dim]` for `thresh_type=exact`.
pub fn kc_gram(kc: &[f32], n: usize, dim: usize, numerics: SolNumerics) -> Vec<f32> {
    let r = rounder(numerics);
    let nf = n.max(1) as f32;
    let mut m = vec![0.0f32; dim * dim];
    for d in 0..dim {
        for e in 0..dim {
            let mut acc = 0.0f32;
            for j in 0..n {
                acc = kc[j * dim + d].mul_add(kc[j * dim + e], acc);
            }
            m[d * dim + e] = r(r(acc) / nf);
        }
    }
    m
}

/// Per-query-block thresholds in log2-score units from rounded `q`
/// `[tokens, dim]` and pooled `kc` `[n, dim]`.
pub fn threshold_faithful(
    q: &[f32],
    kc: &[f32],
    tokens: usize,
    dim: usize,
    p: &SolParams,
    numerics: SolNumerics,
) -> Vec<f32> {
    let r = rounder(numerics);
    let n = num_blocks(tokens);
    let sl2 = p.scale_log2();
    let (mu, var) = kc_stats(kc, n, dim);
    let gram = match p.thresh {
        SolThresh::Exact => Some(kc_gram(kc, n, dim, numerics)),
        SolThresh::Diag => None,
    };
    let mut out = vec![0.0f32; n];
    let mut q_bar = vec![0.0f32; dim];
    for (i, th) in out.iter_mut().enumerate() {
        let len = block_len(tokens, i);
        let start = i * BLOCK_SIZE;
        for (d, qb) in q_bar.iter_mut().enumerate() {
            let mut s = 0.0f32;
            for t in 0..len {
                s += q[(start + t) * dim + d];
            }
            *qb = s / len as f32;
        }
        let var_i = match &gram {
            None => {
                let (mut a, mut c) = (0.0f64, 0.0f64);
                for d in 0..dim {
                    a += (q_bar[d] * mu[d]) as f64;
                    c += (q_bar[d] * q_bar[d] * var[d]) as f64;
                }
                let (a, c) = (a as f32, c as f32);
                *th = a * sl2;
                c * (sl2 * sl2)
            }
            Some(m) => {
                for qb in q_bar.iter_mut() {
                    *qb = r(*qb);
                }
                let (mut a, mut c) = (0.0f64, 0.0f64);
                for d in 0..dim {
                    a += (q_bar[d] * mu[d]) as f64;
                    let mut proj = 0.0f32;
                    for e in 0..dim {
                        proj = q_bar[e].mul_add(m[e * dim + d], proj);
                    }
                    c += (proj * q_bar[d]) as f64;
                }
                let (a, c) = (a as f32, c as f32);
                *th = a * sl2;
                (c - a * a).max(0.0) * (sl2 * sl2)
            }
        };
        *th += p.tau * (var_i.max(0.0) + THRESHOLD_EPS).sqrt();
    }
    out
}

/// `[lo, hi)` sink KV blocks of the single Python sink span, clipped to `n`.
pub fn sink_blocks(tokens: usize, sink_start: Option<usize>, sink_tokens: usize) -> (usize, usize) {
    let n = num_blocks(tokens);
    let (lo, hi) = sink_block_range(tokens, sink_start, sink_tokens);
    let (lo, hi) = (lo.min(n), hi.min(n));
    if lo >= hi {
        (n, n)
    } else {
        (lo, hi)
    }
}

/// Express several `(start, len)` sink spans as the ONE contiguous span the
/// Python interface and the fused kernel take. Succeeds only when the union
/// of their `_sink_block_range`s is itself one block range (touching or
/// overlapping), in which case the result selects exactly the same blocks.
/// Empty spans are ignored; no spans gives `(None, 0)`.
pub fn merge_sink_spans(
    tokens: usize,
    spans: &[(usize, usize)],
) -> Result<(Option<usize>, usize), String> {
    let mut live: Vec<(usize, usize, usize, usize)> = spans
        .iter()
        .filter(|(_, len)| *len > 0)
        .map(|&(start, len)| {
            let (lo, hi) = sink_block_range(tokens, Some(start), len);
            (lo, hi, start, start + len)
        })
        .collect();
    if live.is_empty() {
        return Ok((None, 0));
    }
    live.sort_unstable();
    let (mut hi, mut start, mut end) = (live[0].1, live[0].2, live[0].3);
    for &(l, h, s, e) in &live[1..] {
        if l > hi {
            return Err(format!(
                "sol-attn: sink spans {spans:?} cover KV blocks that are not one contiguous range \
                 (gap before block {l}); the reference takes a single (sink_start, sink_tokens)"
            ));
        }
        hi = hi.max(h);
        start = start.min(s);
        end = end.max(e);
    }
    Ok((Some(start), end - start))
}

/// Route one query block: column means (log2 units) and the exact set.
#[allow(clippy::too_many_arguments)]
fn route_block(
    q: &[f32],
    kc: &[f32],
    tokens: usize,
    dim: usize,
    i: usize,
    threshold: f32,
    sl2: f32,
    sinks: (usize, usize),
    scores: &mut [f32],
) -> (Vec<bool>, Vec<f32>) {
    let n = num_blocks(tokens);
    let len = block_len(tokens, i);
    let start = i * BLOCK_SIZE;
    for t in 0..len {
        let qt = &q[(start + t) * dim..(start + t + 1) * dim];
        for j in 0..n {
            scores[t * n + j] = dot64(qt, &kc[j * dim..(j + 1) * dim]);
        }
    }
    let mut mask = vec![false; n];
    let mut cm = vec![0.0f32; n];
    for j in 0..n {
        let mut sum = 0.0f32;
        for t in 0..len {
            sum += scores[t * n + j];
        }
        cm[j] = sum * sl2 / len as f32;
        let local = i.abs_diff(j) <= 1;
        let sink = j >= sinks.0 && j < sinks.1;
        mask[j] = cm[j] > threshold || local || sink;
    }
    (mask, cm)
}

/// Online-softmax state of one query row (raw-score max, compensated sum).
struct RowState {
    m: f32,
    l: f32,
    o: Vec<f32>,
}

impl RowState {
    /// Fold columns `(raw score, weight, value row)`; `P` is rounded with `r`
    /// before it multiplies the value, `l` takes the unrounded `p`.
    fn fold<'a>(
        &mut self,
        cols: impl Iterator<Item = (f32, f32, &'a [f32])> + Clone,
        sl2: f32,
        r: fn(f32) -> f32,
    ) {
        let mx = cols
            .clone()
            .fold(f32::NEG_INFINITY, |a, (s, _, _)| a.max(s));
        let mn = self.m.max(mx);
        let safe = if mn == f32::NEG_INFINITY { 0.0 } else { mn };
        let alpha = ((self.m - safe) * sl2).exp2();
        let ms = safe * sl2;
        let mut add = 0.0f32;
        for x in self.o.iter_mut() {
            *x *= alpha;
        }
        let mut pv = vec![0.0f64; self.o.len()];
        for (s, w, val) in cols {
            let p = (s * sl2 - ms).exp2();
            add += p * w;
            let pr = r(p) as f64;
            if pr != 0.0 {
                for (acc, &v) in pv.iter_mut().zip(val) {
                    *acc += pr * v as f64;
                }
            }
        }
        for (x, a) in self.o.iter_mut().zip(&pv) {
            *x += *a as f32;
        }
        self.l = self.l * alpha + add;
        self.m = mn;
    }
}

/// Reference-faithful Sol-Attn for one `[tokens, dim]` head.
pub fn sol_attn_head_faithful(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    tokens: usize,
    dim: usize,
    p: &SolParams,
    numerics: SolNumerics,
) -> SolHeadOut {
    use rayon::prelude::*;
    let r = rounder(numerics);
    let (q, k, v) = (
        round_all(q, numerics),
        round_all(k, numerics),
        round_all(v, numerics),
    );
    let n = num_blocks(tokens);
    let sl2 = p.scale_log2();
    let (kc, vc) = pool_kv_faithful(&k, &v, tokens, dim, numerics);
    let threshold = threshold_faithful(&q, &kc, tokens, dim, p, numerics);
    let sinks = sink_blocks(tokens, p.sink_start, p.sink_tokens);
    let groups = n.div_ceil(ROUTE_GROUP);

    struct Block {
        mask: Vec<bool>,
        cm: Vec<f32>,
        out: Vec<f32>,
        lse: Vec<f32>,
    }
    let blocks: Vec<Block> = (0..n)
        .into_par_iter()
        .map(|i| {
            let len = block_len(tokens, i);
            let start = i * BLOCK_SIZE;
            let mut scores = vec![0.0f32; len * n];
            let (mask, cm) = route_block(
                &q,
                &kc,
                tokens,
                dim,
                i,
                threshold[i],
                sl2,
                sinks,
                &mut scores,
            );
            let mut out = vec![0.0f32; len * dim];
            let mut lse = vec![0.0f32; len];
            let mut kscore = vec![0.0f32; BLOCK_SIZE];
            for t in 0..len {
                let qt = &q[(start + t) * dim..(start + t + 1) * dim];
                let mut st = RowState {
                    m: f32::NEG_INFINITY,
                    l: 0.0,
                    o: vec![0.0; dim],
                };
                for g in 0..groups {
                    let (g0, g1) = (g * ROUTE_GROUP, ((g + 1) * ROUTE_GROUP).min(n));
                    if (g0..g1).any(|j| !mask[j]) {
                        let cols = (g0..g1).filter(|&j| !mask[j]).map(|j| {
                            (
                                scores[t * n + j],
                                block_len(tokens, j) as f32,
                                &vc[j * dim..(j + 1) * dim],
                            )
                        });
                        st.fold(cols, sl2, r);
                    }
                    for j in (g0..g1).filter(|&j| mask[j]) {
                        let (k0, kl) = (j * BLOCK_SIZE, block_len(tokens, j));
                        for u in 0..kl {
                            kscore[u] = dot64(qt, &k[(k0 + u) * dim..(k0 + u + 1) * dim]);
                        }
                        let cols = (0..kl)
                            .map(|u| (kscore[u], 1.0f32, &v[(k0 + u) * dim..(k0 + u + 1) * dim]));
                        st.fold(cols, sl2, r);
                    }
                }
                let bad = st.l == 0.0 || st.l.is_nan();
                let inv = if bad { 1.0 } else { 1.0 / st.l };
                for (dst, x) in out[t * dim..(t + 1) * dim].iter_mut().zip(&st.o) {
                    *dst = x * inv;
                }
                lse[t] = if bad {
                    f32::NEG_INFINITY
                } else {
                    (st.m * sl2 + st.l.log2()) * std::f32::consts::LN_2
                };
            }
            Block { mask, cm, out, lse }
        })
        .collect();

    let mut res = SolHeadOut {
        out: Vec::with_capacity(tokens * dim),
        lse: Vec::with_capacity(tokens),
        mask: Vec::with_capacity(n * n),
        col_mean: Vec::with_capacity(n * n),
        threshold,
    };
    for b in blocks {
        res.out.extend(b.out);
        res.lse.extend(b.lse);
        res.mask.extend(b.mask);
        res.col_mean.extend(b.cm);
    }
    res
}

/// [`sol_attn_head_faithful`] over BHSD `q`/`k`/`v`: `(out, lse)`, with
/// `out` `[B, H, T, D]` and `lse` `[B, H, T]`.
#[allow(clippy::too_many_arguments)]
pub fn sol_attn_bhsd_faithful(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    batch: usize,
    heads: usize,
    tokens: usize,
    dim: usize,
    p: &SolParams,
    numerics: SolNumerics,
) -> Result<(Vec<f32>, Vec<f32>), String> {
    let want = batch * heads * tokens * dim;
    if q.len() != want || k.len() != want || v.len() != want {
        return Err(format!(
            "sol-attn: q/k/v want {want} elements, got {}/{}/{}",
            q.len(),
            k.len(),
            v.len()
        ));
    }
    let stride = tokens * dim;
    let mut out = Vec::with_capacity(want);
    let mut lse = Vec::with_capacity(batch * heads * tokens);
    for bh in 0..batch * heads {
        let s = bh * stride..(bh + 1) * stride;
        let head = sol_attn_head_faithful(
            &q[s.clone()],
            &k[s.clone()],
            &v[s],
            tokens,
            dim,
            p,
            numerics,
        );
        out.extend(head.out);
        lse.extend(head.lse);
    }
    Ok((out, lse))
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

    // ---- reference-faithful oracle ----------------------------------------

    /// Seeded N(0, std) (splitmix64 + Box-Muller): independent of `rand`.
    fn normal(seed: u64, n: usize, std: f32) -> Vec<f32> {
        let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
        let mut next = || {
            s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = s;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            ((z >> 11) as f64 + 0.5) / (1u64 << 53) as f64
        };
        (0..n)
            .map(|_| {
                let (u1, u2) = (next(), next());
                ((-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()) as f32 * std
            })
            .collect()
    }

    /// Token rows share a per-block base (`base ~ N(0, 1.5)`) plus 0.3 noise,
    /// so pooled keys are informative and routing is non-trivial.
    fn structured(seed: u64, tokens: usize, dim: usize, shared: usize) -> Vec<f32> {
        let n = num_blocks(tokens);
        let base = normal(seed, n * dim, 1.5);
        let common = normal(seed ^ 0xABCD, n * dim, 1.5);
        let noise = normal(seed + 7, tokens * dim, 0.3);
        (0..tokens * dim)
            .map(|i| {
                let (t, d) = (i / dim, i % dim);
                let b = t / BLOCK_SIZE;
                let src = if b % 4 < shared { &common } else { &base };
                src[(b % n) * dim + d] + noise[i]
            })
            .collect()
    }

    fn rel_l2(a: &[f32], b: &[f32]) -> f64 {
        let (mut num, mut den) = (0.0f64, 0.0f64);
        for (x, y) in a.iter().zip(b) {
            num += (*x as f64 - *y as f64).powi(2);
            den += (*y as f64).powi(2);
        }
        (num / den.max(1e-30)).sqrt()
    }

    const D: usize = 128;

    fn scale() -> f32 {
        (D as f32).sqrt().recip()
    }

    const FAITHFUL: SolNumerics = SolNumerics {
        bf16_faithful: true,
    };
    const IDEAL: SolNumerics = SolNumerics {
        bf16_faithful: false,
    };

    #[test]
    fn bf16_rounding_is_nearest_even() {
        // 1 + 2^-8 is a tie between 1 and 1 + 2^-7: even mantissa (1) wins.
        assert_eq!(bf16_round(1.0 + 1.0 / 256.0), 1.0);
        // 1 + 3 * 2^-8 ties between 1 + 2^-7 (odd) and 1 + 2^-6 (even).
        assert_eq!(bf16_round(1.0 + 3.0 / 256.0), 1.0 + 1.0 / 64.0);
        assert_eq!(bf16_round(-1.0 - 3.0 / 256.0), -1.0 - 1.0 / 64.0);
        assert_eq!(bf16_round(1.0 + 1.5 / 256.0), 1.0 + 1.0 / 128.0);
        assert!(bf16_round(f32::NAN).is_nan());
        assert_eq!(bf16_round(f32::INFINITY), f32::INFINITY);
        assert_eq!(bf16_bits(f32::MAX), 0x7F80); // rounds up to +inf, as torch does
    }

    #[test]
    fn sink_spans_merge_only_when_contiguous() {
        // Nothing to merge.
        assert_eq!(merge_sink_spans(4096, &[]).unwrap(), (None, 0));
        assert_eq!(merge_sink_spans(4096, &[(5, 0)]).unwrap(), (None, 0));
        // One span passes through.
        assert_eq!(
            merge_sink_spans(4096, &[(1000, 300)]).unwrap(),
            (Some(1000), 300)
        );
        // Adjacent block ranges ([0,2) and [2,3)) merge into one span with
        // the same block range.
        let merged = merge_sink_spans(4096, &[(130, 20), (0, 77)]).unwrap();
        assert_eq!(merged, (Some(0), 150));
        assert_eq!(sink_block_range(4096, merged.0, merged.1), (0, 3));
        // Same block, different tokens.
        assert_eq!(
            merge_sink_spans(4096, &[(0, 10), (60, 3)]).unwrap(),
            (Some(0), 63)
        );
        // A gap (blocks [0,2) and [32,33)) is not expressible as one span.
        assert!(merge_sink_spans(4096, &[(0, 77), (2048, 64)]).is_err());
    }

    #[test]
    fn sink_blocks_match_the_python_range_and_clip() {
        assert_eq!(sink_blocks(4096, None, 100), (62, 64));
        assert_eq!(sink_blocks(4096, Some(1000), 300), (15, 21));
        assert_eq!(sink_blocks(4096, None, 0), (64, 64));
        assert_eq!(sink_blocks(4000, Some(3990), 100), (62, 63));
    }

    #[test]
    fn faithful_pools_bf16_and_divides_by_the_live_length() {
        let tokens = 4033; // last block holds one token
        let k = round_all(&normal(3, tokens * D, 1.0), FAITHFUL);
        let v = round_all(&normal(4, tokens * D, 1.0), FAITHFUL);
        let (kc, vc) = pool_kv_faithful(&k, &v, tokens, D, FAITHFUL);
        let last = num_blocks(tokens) - 1;
        for d in 0..D {
            // A single live row: mean and sum are that row, exactly.
            assert_eq!(kc[last * D + d], k[(tokens - 1) * D + d]);
            assert_eq!(vc[last * D + d], v[(tokens - 1) * D + d]);
            assert_eq!(bf16_round(kc[d]), kc[d]);
        }
        let ideal = pool_means(&k, tokens, D);
        let ulp = kc
            .iter()
            .zip(&ideal)
            .map(|(a, b)| (a - b).abs() / b.abs().max(1e-3))
            .fold(0.0f32, f32::max);
        assert!(ulp <= 1.0 / 128.0, "kc off by {ulp} relative");
    }

    #[test]
    fn faithful_all_exact_equals_dense_attention() {
        // A full sink and a hugely negative tau are both "everything exact".
        let tokens = 1100; // partial tail block (len 12)
        let (q, k, v) = (
            normal(10, tokens * D, 1.0),
            normal(11, tokens * D, 1.0),
            normal(12, tokens * D, 1.0),
        );
        let dense = dense_attn_head(&q, &k, &v, tokens, D, scale());
        for p in [
            SolParams {
                sink_start: Some(0),
                sink_tokens: tokens,
                ..SolParams::diag(1.0, scale())
            },
            SolParams::diag(-1.0e4, scale()),
        ] {
            let got = sol_attn_head_faithful(&q, &k, &v, tokens, D, &p, FAITHFUL);
            assert!(got.mask.iter().all(|&m| m));
            let err = rel_l2(&got.out, &dense);
            assert!(err <= 5e-3, "faithful vs dense rel-L2 {err}");
            let ideal = sol_attn_head_faithful(&q, &k, &v, tokens, D, &p, IDEAL);
            let err = rel_l2(&ideal.out, &dense);
            assert!(err <= 1e-5, "ideal vs dense rel-L2 {err}");
            // LSE is the natural-log partition function of the ideal scores.
            for t in [0usize, 517, tokens - 1] {
                let s: Vec<f64> = (0..tokens)
                    .map(|u| {
                        let mut acc = 0.0f64;
                        for d in 0..D {
                            acc += q[t * D + d] as f64 * k[u * D + d] as f64;
                        }
                        acc * scale() as f64
                    })
                    .collect();
                let m = s.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let want = m + s.iter().map(|x| (x - m).exp()).sum::<f64>().ln();
                assert!(
                    (ideal.lse[t] as f64 - want).abs() < 1e-4,
                    "lse[{t}] {} vs {want}",
                    ideal.lse[t]
                );
                assert!((got.lse[t] as f64 - want).abs() < 2e-2);
            }
        }
    }

    #[test]
    fn huge_tau_keeps_only_the_local_window() {
        // 4096 tokens: 64 blocks, one route group; everything outside the
        // 3-block band goes through pooled Kc / summed Vc with len(j) mass.
        let tokens = 4096;
        let (q, k, v) = (
            structured(20, tokens, D, 1),
            structured(21, tokens, D, 1),
            normal(22, tokens * D, 1.0),
        );
        let p = SolParams::diag(1.0e4, scale());
        let got = sol_attn_head_faithful(&q, &k, &v, tokens, D, &p, FAITHFUL);
        let n = num_blocks(tokens);
        for i in 0..n {
            for j in 0..n {
                assert_eq!(got.mask[i * n + j], i.abs_diff(j) <= 1, "block {i},{j}");
            }
        }
        // The ideal f32 oracle folds approximate-then-exact per row instead
        // of per group; the math is the same.
        let old = sol_attn_head(&q, &k, &v, tokens, D, 1.0e4, scale(), None, 0);
        let ideal = sol_attn_head_faithful(&q, &k, &v, tokens, D, &p, IDEAL);
        assert!(
            rel_l2(&ideal.out, &old) < 1e-5,
            "{}",
            rel_l2(&ideal.out, &old)
        );
        let err = rel_l2(&got.out, &old);
        assert!(err <= 1e-2, "faithful vs ideal rel-L2 {err}");
    }

    #[test]
    fn headline_routing_is_nontrivial_and_faithful_tracks_ideal() {
        // Spec case 1 (one head): T=4096, diag, tau 1.0 on structured data.
        let tokens = 4096;
        let (q, k, v) = (
            structured(30, tokens, D, 2),
            structured(31, tokens, D, 2),
            normal(32, tokens * D, 1.0),
        );
        let n = num_blocks(tokens);
        for tau in [1.0f32, 1.5] {
            let p = SolParams::diag(tau, scale());
            let got = sol_attn_head_faithful(&q, &k, &v, tokens, D, &p, FAITHFUL);
            let exact = got.mask.iter().filter(|&&m| m).count();
            let local = 3 * n - 2;
            assert!(exact > local, "tau {tau}: only the local window routed");
            assert!(exact < n * n, "tau {tau}: everything routed");
            let ideal = sol_attn_head_faithful(&q, &k, &v, tokens, D, &p, IDEAL);
            let flips = got
                .mask
                .iter()
                .zip(&ideal.mask)
                .filter(|(a, b)| a != b)
                .count();
            assert!(flips * 1000 <= n * n, "tau {tau}: {flips} route flips");
            let err = rel_l2(&got.out, &ideal.out);
            assert!(err <= 1e-2, "tau {tau}: faithful vs ideal rel-L2 {err}");
            // The old f32 oracle (per-row order, f32 threshold) agrees with
            // the ideal new one wherever the routing agrees.
            let old_mask = exact_mask(
                &q,
                &pool_means(&k, tokens, D),
                tokens,
                D,
                tau,
                scale(),
                None,
                0,
            );
            if old_mask == ideal.mask {
                let old = sol_attn_head(&q, &k, &v, tokens, D, tau, scale(), None, 0);
                assert!(rel_l2(&ideal.out, &old) < 1e-5);
            }
        }
    }

    #[test]
    fn several_route_groups_with_a_one_block_tail_group() {
        // 8256 tokens: 129 blocks, G = 3 route groups, last group = 1 block.
        let tokens = 8256;
        let (q, k, v) = (
            structured(40, tokens, D, 1),
            structured(41, tokens, D, 1),
            normal(42, tokens * D, 1.0),
        );
        let n = num_blocks(tokens);
        assert_eq!((n, n.div_ceil(ROUTE_GROUP)), (129, 3));
        let p = SolParams {
            sink_start: Some(1000),
            sink_tokens: 300,
            ..SolParams::diag(2.0, scale())
        };
        let got = sol_attn_head_faithful(&q, &k, &v, tokens, D, &p, FAITHFUL);
        let ideal = sol_attn_head_faithful(&q, &k, &v, tokens, D, &p, IDEAL);
        for i in 0..n {
            for j in 15..21 {
                assert!(got.mask[i * n + j], "sink block {j} for query block {i}");
            }
            assert!(got.mask[i * n + i]);
        }
        let err = rel_l2(&got.out, &ideal.out);
        assert!(err <= 1e-2, "faithful vs ideal rel-L2 {err}");
        assert!(got.out.iter().all(|x| x.is_finite()));
        assert!(got.lse.iter().all(|x| x.is_finite()));
        // Same routing, per-row order: the old multi-span oracle agrees.
        if got.mask == ideal.mask {
            let old = sol_attn_head_sunk(&q, &k, &v, tokens, D, 2.0, scale(), &[(Some(1000), 300)]);
            let old_mask = exact_mask_sunk(
                &q,
                &pool_means(&k, tokens, D),
                tokens,
                D,
                2.0,
                scale(),
                &[(Some(1000), 300)],
            );
            if old_mask == ideal.mask {
                assert!(rel_l2(&ideal.out, &old) < 1e-5);
            }
        }
    }

    #[test]
    fn suffix_sink_forces_its_blocks_exact() {
        let tokens = 4000; // partial tail (len 32)
        let (q, k, v) = (
            structured(50, tokens, D, 1),
            structured(51, tokens, D, 1),
            normal(52, tokens * D, 1.0),
        );
        let p = SolParams {
            sink_start: None,
            sink_tokens: 100,
            ..SolParams::diag(1.0e4, scale())
        };
        let got = sol_attn_head_faithful(&q, &k, &v, tokens, D, &p, FAITHFUL);
        let n = num_blocks(tokens);
        let (lo, hi) = sink_blocks(tokens, None, 100);
        assert_eq!((lo, hi), (60, 63));
        for i in 0..n {
            for j in 0..n {
                let want = i.abs_diff(j) <= 1 || (lo..hi).contains(&j);
                assert_eq!(got.mask[i * n + j], want, "block {i},{j}");
            }
        }
    }

    #[test]
    fn exact_threshold_matches_the_full_covariance_definition() {
        let tokens = 2048;
        let (q, k) = (structured(60, tokens, D, 2), structured(61, tokens, D, 2));
        let n = num_blocks(tokens);
        let tau = 1.25f32;
        let p = SolParams {
            thresh: SolThresh::Exact,
            ..SolParams::diag(tau, scale())
        };
        let kc = pool_means(&k, tokens, D);
        let got = threshold_faithful(&q, &kc, tokens, D, &p, IDEAL);
        // Independent f64: mean = qbar.mu, var = qbar^T E[kc kc^T] qbar - mean^2.
        let q_bar = pool_means(&q, tokens, D);
        let sl2 = (scale() * LOG2_E) as f64;
        for i in [0usize, 7, n - 1] {
            let qb = &q_bar[i * D..(i + 1) * D];
            let mut mean = 0.0f64;
            let mut second = 0.0f64;
            for d in 0..D {
                let mu: f64 = (0..n).map(|j| kc[j * D + d] as f64).sum::<f64>() / n as f64;
                mean += qb[d] as f64 * mu;
            }
            for j in 0..n {
                let dot: f64 = (0..D).map(|d| qb[d] as f64 * kc[j * D + d] as f64).sum();
                second += dot * dot / n as f64;
            }
            let var = (second - mean * mean).max(0.0) * sl2 * sl2;
            let want = mean * sl2 + tau as f64 * (var + 1e-6).sqrt();
            let tol = 1e-3 * want.abs().max(1.0);
            assert!(
                (got[i] as f64 - want).abs() < tol,
                "block {i}: {} vs {want}",
                got[i]
            );
        }
        // Faithful exact mode differs from diag, and still routes sensibly.
        let fe = threshold_faithful(&q, &kc, tokens, D, &p, FAITHFUL);
        let fd = threshold_faithful(&q, &kc, tokens, D, &SolParams::diag(tau, scale()), FAITHFUL);
        assert!(fe.iter().zip(&fd).any(|(a, b)| (a - b).abs() > 1e-3));
        for (a, b) in fe.iter().zip(&got) {
            assert!((a - b).abs() <= 2e-2 * b.abs().max(1.0), "{a} vs {b}");
        }
        let v = normal(62, tokens * D, 1.0);
        let out = sol_attn_head_faithful(&q, &k, &v, tokens, D, &p, FAITHFUL);
        let ideal = sol_attn_head_faithful(&q, &k, &v, tokens, D, &p, IDEAL);
        assert!(rel_l2(&out.out, &ideal.out) <= 1e-2);
    }

    #[test]
    fn tiny_sequences_are_dense() {
        // NT <= 2: every block is inside the local window. (At NT = 3,
        // blocks 0 and 2 are two apart; only the middle query block is
        // guaranteed dense, checked below.)
        for tokens in [1usize, 63, 64, 65, 128, 130] {
            let (q, k, v) = (
                normal(70 + tokens as u64, tokens * D, 1.0),
                normal(80 + tokens as u64, tokens * D, 1.0),
                normal(90 + tokens as u64, tokens * D, 1.0),
            );
            let p = SolParams::diag(1.0, scale());
            let got = sol_attn_head_faithful(&q, &k, &v, tokens, D, &p, FAITHFUL);
            let dense = dense_attn_head(&q, &k, &v, tokens, D, scale());
            let rows = if num_blocks(tokens) <= 2 {
                0..tokens
            } else {
                BLOCK_SIZE..2 * BLOCK_SIZE
            };
            let r = rows.start * D..rows.end * D;
            let err = rel_l2(&got.out[r.clone()], &dense[r]);
            assert!(err <= 5e-3, "T={tokens}: rel-L2 {err}");
        }
    }

    #[test]
    fn bhsd_faithful_indexes_every_head() {
        let (b, h, tokens) = (2usize, 3usize, 1000usize);
        let n = b * h * tokens * D;
        let (q, k, v) = (normal(1, n, 1.0), normal(2, n, 1.0), normal(3, n, 1.0));
        let p = SolParams::diag(1.25, scale());
        let (out, lse) = sol_attn_bhsd_faithful(&q, &k, &v, b, h, tokens, D, &p, FAITHFUL).unwrap();
        assert_eq!((out.len(), lse.len()), (n, b * h * tokens));
        let stride = tokens * D;
        let head = 4;
        let one = sol_attn_head_faithful(
            &q[head * stride..(head + 1) * stride],
            &k[head * stride..(head + 1) * stride],
            &v[head * stride..(head + 1) * stride],
            tokens,
            D,
            &p,
            FAITHFUL,
        );
        assert_eq!(&out[head * stride..(head + 1) * stride], &one.out[..]);
        assert!(sol_attn_bhsd_faithful(&q[1..], &k, &v, b, h, tokens, D, &p, FAITHFUL).is_err());
    }
}
