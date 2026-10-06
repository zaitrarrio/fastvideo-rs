//! LingBot MoE router (host reference).
//!
//! `LingBotVideoRouter.forward` of `transformer_lingbot_video.py`:
//!
//! 1. `logits = x @ W_r^T` in f32; `scores = sigmoid(logits)` (or softmax);
//! 2. selection uses `scores + e_score_correction_bias`; with `n_group > 1`
//!    the experts split into `n_group` contiguous groups, each scored by the
//!    sum of its two best biased scores, and only the `topk_group` best groups
//!    stay eligible (DeepSeek-V3 group-limited routing);
//! 3. the gate weights gather the **bias-free** scores of the chosen experts,
//!    are L1-normalized (`+ 1e-20`) when `norm_topk_prob`, then multiplied by
//!    `routed_scaling_factor` and rounded to the activation dtype (bf16).
//!
//! The device kernel (`moe_group_topk` in `kernels.cu`) implements the same
//! rule; [`route`] is its oracle.

use rayon::prelude::*;

use super::config::{LingBotTransformerConfig, ScoreFunc};

/// Routing parameters of one MoE layer.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RouterSpec {
    pub num_experts: usize,
    pub top_k: usize,
    pub score_func: ScoreFunc,
    pub norm_topk_prob: bool,
    /// `None` or `Some(1)`: plain top-k over all experts.
    pub n_group: Option<usize>,
    pub topk_group: usize,
    pub route_scale: f32,
    /// Round the gate weights to bf16 (`top_scores.to(tokens.dtype)`).
    pub round_bf16: bool,
}

impl RouterSpec {
    pub fn from_config(cfg: &LingBotTransformerConfig) -> Self {
        let n_group = cfg.n_group.filter(|&g| g > 1);
        Self {
            num_experts: cfg.num_experts,
            top_k: cfg.num_experts_per_tok,
            score_func: cfg.score_func,
            norm_topk_prob: cfg.norm_topk_prob,
            n_group,
            topk_group: cfg.topk_group.or(n_group).unwrap_or(1),
            route_scale: cfg.routed_scaling_factor,
            round_bf16: true,
        }
    }
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn bf16_round(x: f32) -> f32 {
    // Round-to-nearest-even on the top 16 bits (torch's float → bfloat16).
    if !x.is_finite() {
        return x;
    }
    let bits = x.to_bits();
    let lsb = (bits >> 16) & 1;
    let rounded = bits.wrapping_add(0x7fff + lsb) & 0xffff_0000;
    f32::from_bits(rounded)
}

/// Indices of the `k` largest values (descending; ties keep the lower index).
fn topk_desc(values: &[f32], k: usize) -> Vec<usize> {
    let mut order: Vec<usize> = (0..values.len()).collect();
    order.sort_by(|&a, &b| {
        values[b]
            .partial_cmp(&values[a])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(&b))
    });
    order.truncate(k);
    order
}

/// Route one token from its router logits. Returns `(experts, weights)`,
/// experts in descending selection-score order.
pub fn route_token(logits: &[f32], bias: &[f32], spec: &RouterSpec) -> (Vec<u32>, Vec<f32>) {
    let e = spec.num_experts;
    debug_assert_eq!(logits.len(), e);
    let scores: Vec<f32> = match spec.score_func {
        ScoreFunc::Sigmoid => logits.iter().map(|&x| sigmoid(x)).collect(),
        ScoreFunc::Softmax => {
            let m = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let ex: Vec<f32> = logits.iter().map(|&x| (x - m).exp()).collect();
            let z: f32 = ex.iter().sum();
            ex.into_iter().map(|v| v / z).collect()
        }
    };
    let mut choice: Vec<f32> = scores
        .iter()
        .zip(bias.iter().chain(std::iter::repeat(&0.0)))
        .map(|(s, b)| s + b)
        .collect();
    if let Some(g) = spec.n_group {
        let per = e / g;
        let group_scores: Vec<f32> = (0..g)
            .map(|gi| {
                let top2 = topk_desc(&choice[gi * per..(gi + 1) * per], 2.min(per));
                top2.iter().map(|&j| choice[gi * per + j]).sum()
            })
            .collect();
        let keep = topk_desc(&group_scores, spec.topk_group);
        for gi in 0..g {
            if !keep.contains(&gi) {
                for v in &mut choice[gi * per..(gi + 1) * per] {
                    *v = f32::NEG_INFINITY;
                }
            }
        }
    }
    let idx = topk_desc(&choice, spec.top_k);
    let mut w: Vec<f32> = idx.iter().map(|&i| scores[i]).collect();
    if spec.top_k > 1 && spec.norm_topk_prob {
        let z: f32 = w.iter().sum::<f32>() + 1e-20;
        for v in &mut w {
            *v /= z;
        }
    }
    for v in &mut w {
        *v *= spec.route_scale;
        if spec.round_bf16 {
            *v = bf16_round(*v);
        }
    }
    (idx.into_iter().map(|i| i as u32).collect(), w)
}

/// Route `n = logits.len() / num_experts` tokens. Flat `[n, top_k]` outputs.
pub fn route(logits: &[f32], bias: &[f32], spec: &RouterSpec) -> (Vec<u32>, Vec<f32>) {
    let e = spec.num_experts;
    assert!(e > 0 && logits.len() % e == 0, "router logits / experts");
    let k = spec.top_k;
    let rows: Vec<(Vec<u32>, Vec<f32>)> = logits
        .par_chunks(e)
        .map(|row| route_token(row, bias, spec))
        .collect();
    let mut idx = Vec::with_capacity(rows.len() * k);
    let mut val = Vec::with_capacity(rows.len() * k);
    for (i, w) in rows {
        idx.extend(i);
        val.extend(w);
    }
    (idx, val)
}

/// Assignments grouped by expert: `order[j]` is the flat `(token * k + slot)`
/// of the j-th row in expert-sorted order, `counts[e]` rows per expert, and
/// `pos[token * k + slot]` that row's index in the sorted order (what the
/// combine step gathers from).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Dispatch {
    pub order: Vec<usize>,
    pub counts: Vec<usize>,
    pub pos: Vec<u32>,
}

/// Stable counting sort of `[n, k]` expert ids.
pub fn dispatch(idx: &[u32], num_experts: usize) -> Dispatch {
    let mut counts = vec![0usize; num_experts];
    for &e in idx {
        counts[e as usize] += 1;
    }
    let mut start = vec![0usize; num_experts];
    for e in 1..num_experts {
        start[e] = start[e - 1] + counts[e - 1];
    }
    let mut order = vec![0usize; idx.len()];
    let mut pos = vec![0u32; idx.len()];
    let mut next = start;
    for (flat, &e) in idx.iter().enumerate() {
        let slot = next[e as usize];
        order[slot] = flat;
        pos[flat] = slot as u32;
        next[e as usize] += 1;
    }
    Dispatch { order, counts, pos }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec(groups: Option<usize>) -> RouterSpec {
        RouterSpec {
            num_experts: 8,
            top_k: 2,
            score_func: ScoreFunc::Sigmoid,
            norm_topk_prob: true,
            n_group: groups,
            topk_group: 2,
            route_scale: 2.5,
            round_bf16: false,
        }
    }

    fn logit(p: f32) -> f32 {
        (p / (1.0 - p)).ln()
    }

    #[test]
    fn plain_topk_normalizes_and_scales() {
        let probs = [0.1, 0.9, 0.2, 0.3, 0.8, 0.05, 0.4, 0.15];
        let logits: Vec<f32> = probs.iter().map(|&p| logit(p)).collect();
        let (idx, w) = route_token(&logits, &[0.0; 8], &spec(None));
        assert_eq!(idx, vec![1, 4]);
        let z = 0.9 + 0.8;
        assert!((w[0] - 2.5 * 0.9 / z).abs() < 1e-5);
        assert!((w[1] - 2.5 * 0.8 / z).abs() < 1e-5);
    }

    #[test]
    fn group_limit_excludes_weak_groups() {
        // Groups of 2: g0 {0,1}, g1 {2,3}, g2 {4,5}, g3 {6,7}.
        // Expert 0 is the single best, but its group's top-2 sum is weak, so
        // the two strongest groups (g1, g3) win and expert 0 is never chosen.
        let probs = [0.95, 0.01, 0.6, 0.6, 0.3, 0.3, 0.55, 0.55];
        let logits: Vec<f32> = probs.iter().map(|&p| logit(p)).collect();
        let (idx, _) = route_token(&logits, &[0.0; 8], &spec(Some(4)));
        assert!(!idx.contains(&0), "{idx:?}");
        assert!(idx.iter().all(|&i| [2, 3, 6, 7].contains(&i)), "{idx:?}");
        // Without groups expert 0 wins.
        let (plain, _) = route_token(&logits, &[0.0; 8], &spec(None));
        assert_eq!(plain[0], 0);
    }

    #[test]
    fn bias_selects_but_weights_stay_bias_free() {
        let probs = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5];
        let logits: Vec<f32> = probs.iter().map(|&p| logit(p)).collect();
        let mut bias = [0.0f32; 8];
        bias[5] = 1.0;
        bias[6] = 0.5;
        let (idx, w) = route_token(&logits, &bias, &spec(None));
        assert_eq!(idx, vec![5, 6]);
        // Bias-free scores are equal → equal weights 2.5 * 0.5.
        assert!((w[0] - 1.25).abs() < 1e-5 && (w[1] - 1.25).abs() < 1e-5);
    }

    #[test]
    fn bf16_rounding_matches_torch_rne() {
        assert_eq!(bf16_round(1.0), 1.0);
        // 1 + 2^-8 sits exactly between bf16 neighbours 1 and 1+2^-7: ties to even (1.0).
        assert_eq!(bf16_round(1.0 + 1.0 / 256.0), 1.0);
        assert_eq!(bf16_round(1.0 + 3.0 / 256.0), 1.0 + 4.0 / 256.0);
    }

    #[test]
    fn dispatch_is_a_stable_permutation() {
        let idx = vec![3u32, 1, 1, 0, 3, 3];
        let d = dispatch(&idx, 4);
        assert_eq!(d.counts, vec![1, 2, 0, 3]);
        assert_eq!(d.order, vec![3, 1, 2, 0, 4, 5]);
        for (flat, &p) in d.pos.iter().enumerate() {
            assert_eq!(d.order[p as usize], flat);
        }
    }

    #[test]
    fn batched_route_matches_per_token() {
        let s = spec(Some(4));
        let logits: Vec<f32> = (0..5 * 8).map(|i| ((i as f32) * 0.37).sin() * 3.0).collect();
        let bias: Vec<f32> = (0..8).map(|i| (i as f32) * 0.01).collect();
        let (idx, w) = route(&logits, &bias, &s);
        for t in 0..5 {
            let (i1, w1) = route_token(&logits[t * 8..(t + 1) * 8], &bias, &s);
            assert_eq!(&idx[t * 2..t * 2 + 2], &i1[..]);
            assert_eq!(&w[t * 2..t * 2 + 2], &w1[..]);
        }
    }
}
