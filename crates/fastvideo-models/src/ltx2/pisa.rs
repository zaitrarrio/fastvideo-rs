//! LTX-2.3 stage-2 PISA contract (`models/ltx23/optimized/env.sh` on
//! NVlabs/Sana `sol-engine`).
//!
//! Video self-attention only. Layers 0 and 1 stay dense. Later layers use
//! the PISA score-route kernel at sparsity 0.9 and block size 64. Stage-2
//! steps 1 and 2 are the midpoint token-prune steps (keep half, by feature
//! norm). The stage-1 SCSP preset `8of15_last_29calls` skips steps 16-28
//! (`techniques/presets.py` `_SCSP_SKIP_STEPS`). LoRA strengths are fused
//! by `ltx2::lora` when the distilled file is present.

/// Video blocks that stay dense on every stage-2 forward.
pub const DENSE_LAYERS: [usize; 2] = [0, 1];

pub const LAYERS_PER_FORWARD: usize = 48;

pub const FORWARDS: usize = 3;

pub const SPARSITY: f64 = 0.9;

pub const BLOCK_SIZE: usize = 64;

pub const STAGE1_LORA_STRENGTH: f64 = 0.25;

pub const STAGE2_LORA_STRENGTH: f64 = 0.5;

/// Keep this fraction of video tokens. `1.0 - ratio` is dropped.
pub const PRUNE_RATIO: f64 = 0.5;

/// Refine steps that prune. Step 0 does not.
pub const PRUNE_STEPS: [usize; 2] = [1, 2];

/// Named stage-1 cache preset. Skip mask is `_SCSP_SKIP_STEPS = "16-28"`.
pub const STAGE1_CACHE_PRESET: &str = "8of15_last_29calls";

/// Inclusive skip range from `techniques/presets.py` for this preset.
pub const STAGE1_CACHE_SKIP_START: usize = 16;
pub const STAGE1_CACHE_SKIP_END: usize = 28;

/// `FASTVIDEO_LTX2_STAGE1_CACHE=1` (or the preset name) applies [`stage1_skips_step`].
pub fn stage1_cache_requested(value: Option<&str>) -> bool {
    match value.map(str::trim) {
        Some("1") => true,
        Some(v) => v.eq_ignore_ascii_case(STAGE1_CACHE_PRESET) || v.eq_ignore_ascii_case("scsp"),
        None => false,
    }
}

pub const STAGE1_CACHE_APPLIED: &str =
    "ltx2 pisa: stage-1 SCSP preset 8of15_last_29calls skips steps 16-28 \
(techniques/presets.py _SCSP_SKIP_STEPS; whole-step velocity reuse, delta_scale 0)";

pub fn stage1_skips_step(step: usize) -> bool {
    (STAGE1_CACHE_SKIP_START..=STAGE1_CACHE_SKIP_END).contains(&step)
}

/// `FASTVIDEO_LTX2_MIDPOINT_PRUNE=1` (or `feat_norm`) prunes stage-2 video tokens.
pub fn midpoint_prune_requested(value: Option<&str>) -> bool {
    match value.map(str::trim) {
        Some("1") => true,
        Some(v) => v.eq_ignore_ascii_case("feat_norm") || v.eq_ignore_ascii_case("prune"),
        None => false,
    }
}

pub const PRUNE_APPLIED: &str = "ltx2 pisa: midpoint feat_norm prune ratio 0.5 steps 1,2 \
(video tokens only, all 48 blocks, prev-hidden write-back)";

/// Compensation for dropped tokens: previous step's full video hidden.
pub const PRUNE_COMPENSATION: &str = "prev";

/// Ascending kept-token indices. `tokens` is row-major `[seq, dim]`.
///
/// Score is batch-mean L2² (`hidden.pow(2).sum(-1).mean(0)`). Keep
/// `round(seq * keep_ratio)` top rows, then sort the indices.
pub fn feat_norm_keep_indices(
    tokens: &[f32],
    seq: usize,
    dim: usize,
    keep_ratio: f64,
) -> Vec<usize> {
    if seq == 0 || dim == 0 || tokens.len() < seq * dim {
        return Vec::new();
    }
    let keep = ((seq as f64 * keep_ratio).round() as usize).clamp(1, seq);
    if keep >= seq {
        return (0..seq).collect();
    }
    let mut scored: Vec<(f32, usize)> = (0..seq)
        .map(|i| {
            let row = &tokens[i * dim..(i + 1) * dim];
            let score: f32 = row.iter().map(|x| x * x).sum();
            (score, i)
        })
        .collect();
    scored.sort_by(|a, b| match b.0.partial_cmp(&a.0) {
        Some(std::cmp::Ordering::Equal) | None => a.1.cmp(&b.1),
        Some(order) => order,
    });
    let mut idx: Vec<usize> = scored[..keep].iter().map(|(_, i)| *i).collect();
    idx.sort_unstable();
    idx
}

/// Write `kept` rows into a clone of `prev` at `idx`. All slices are `[seq, dim]`.
pub fn scatter_prev(prev: &[f32], seq: usize, dim: usize, idx: &[usize], kept: &[f32]) -> Vec<f32> {
    let mut full = prev[..seq * dim].to_vec();
    for (j, &i) in idx.iter().enumerate() {
        let src = j * dim;
        let dst = i * dim;
        full[dst..dst + dim].copy_from_slice(&kept[src..src + dim]);
    }
    full
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Ltx23PisaRoute {
    Dense,
    /// Piecewise sparse video self-attention (score-route top-k + approx remainder).
    Pisa {
        sparsity: f64,
        block_size: usize,
    },
}

pub fn route(forward: usize, layer: usize) -> Result<Ltx23PisaRoute, String> {
    if forward >= FORWARDS {
        return Err(format!(
            "ltx2 pisa: stage-2 forward {forward} is past {FORWARDS} forwards"
        ));
    }
    if layer >= LAYERS_PER_FORWARD {
        return Err(format!(
            "ltx2 pisa: layer {layer} is past {LAYERS_PER_FORWARD} video blocks"
        ));
    }
    if DENSE_LAYERS.contains(&layer) {
        Ok(Ltx23PisaRoute::Dense)
    } else {
        Ok(Ltx23PisaRoute::Pisa {
            sparsity: SPARSITY,
            block_size: BLOCK_SIZE,
        })
    }
}

pub fn prunes_step(step: usize) -> bool {
    PRUNE_STEPS.contains(&step)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layers_zero_and_one_stay_dense() {
        assert_eq!(route(0, 0).unwrap(), Ltx23PisaRoute::Dense);
        assert_eq!(route(1, 1).unwrap(), Ltx23PisaRoute::Dense);
        assert_eq!(route(2, 0).unwrap(), Ltx23PisaRoute::Dense);
        assert_eq!(
            route(0, 2).unwrap(),
            Ltx23PisaRoute::Pisa {
                sparsity: 0.9,
                block_size: 64
            }
        );
        assert_eq!(
            route(2, 47).unwrap(),
            Ltx23PisaRoute::Pisa {
                sparsity: SPARSITY,
                block_size: BLOCK_SIZE
            }
        );
        assert!(route(3, 0).is_err());
        assert!(route(0, 48).is_err());
    }

    #[test]
    fn prune_is_the_middle_refine_steps() {
        assert!(!prunes_step(0));
        assert!(prunes_step(1));
        assert!(prunes_step(2));
        assert!(!prunes_step(3));
        assert_eq!(STAGE1_CACHE_PRESET, "8of15_last_29calls");
        assert_eq!(STAGE1_LORA_STRENGTH, 0.25);
        assert_eq!(STAGE2_LORA_STRENGTH, 0.5);
    }

    #[test]
    fn stage1_cache_env_is_off_until_the_preset() {
        assert!(!stage1_cache_requested(None));
        assert!(!stage1_cache_requested(Some("off")));
        assert!(stage1_cache_requested(Some("1")));
        assert!(stage1_cache_requested(Some(STAGE1_CACHE_PRESET)));
        assert!(stage1_cache_requested(Some("scsp")));
        assert!(STAGE1_CACHE_APPLIED.contains("16-28"));
        assert!(!stage1_skips_step(0));
        assert!(!stage1_skips_step(15));
        assert!(stage1_skips_step(16));
        assert!(stage1_skips_step(28));
        assert!(!stage1_skips_step(29));
    }

    #[test]
    fn midpoint_prune_env_is_off_until_feat_norm() {
        assert!(!midpoint_prune_requested(None));
        assert!(!midpoint_prune_requested(Some("off")));
        assert!(midpoint_prune_requested(Some("1")));
        assert!(midpoint_prune_requested(Some("feat_norm")));
        assert!(PRUNE_APPLIED.contains("prev-hidden"));
        assert_eq!(PRUNE_COMPENSATION, "prev");
    }

    #[test]
    fn feat_norm_keeps_the_largest_l2_rows() {
        // 4 tokens × 2 dim. Scores: 1, 25, 4, 0. Keep half → 2 tokens (1 and 2).
        let tokens = [1.0f32, 0.0, 3.0, 4.0, 2.0, 0.0, 0.0, 0.0];
        assert_eq!(feat_norm_keep_indices(&tokens, 4, 2, 0.5), vec![1, 2]);
        let prev = [10.0f32, 10.0, 20.0, 20.0, 30.0, 30.0, 40.0, 40.0];
        let kept = [3.0f32, 4.0, 2.0, 0.0];
        assert_eq!(
            scatter_prev(&prev, 4, 2, &[1, 2], &kept),
            vec![10.0, 10.0, 3.0, 4.0, 2.0, 0.0, 40.0, 40.0]
        );
    }
}
