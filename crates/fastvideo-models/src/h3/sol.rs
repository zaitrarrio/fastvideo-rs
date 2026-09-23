//! MiniMax-H3 Sol-Attn and cache contracts from NVlabs/Sana `sol-engine`
//! (`models/minimax_h3/Sol-H3` and `models/minimax_h3.toml`).
//!
//! The one-GPU Sol-H3 recipe stays dense. This module records the optional
//! Stage-1 Sol-Attn route (`--ref-stage1-attn sol`) and the RTX 4090/5090
//! TeaCache knobs. Video self-attention still runs dense SDPA until a Sol
//! kernel is linked. The RTX TeaCache signal is not published here, so those
//! knobs stay a record.

/// Body layers in MiniMax-H3.
pub const LAYERS_PER_FORWARD: usize = 50;

/// Sol-H3 / Spark Stage-1 transformer updates.
pub const STAGE1_FORWARDS: usize = 4;

/// Taus on the three Sol updates after the dense first update.
pub const STAGE1_TAUS: [f64; 3] = [1.0, 1.25, 1.5];

/// RTX 4090/5090 cell (`models/minimax_h3.toml` `[rtx4090.policy]`).
pub const RTX_TEACACHE_THRESHOLD: f64 = 0.10;
pub const RTX_TEACACHE_RETAIN_STEPS: usize = 5;
pub const RTX_TEACACHE_COOLDOWN_STEPS: usize = 1;
pub const RTX_FIRST_DENSE_STEPS: usize = 10;
pub const RTX_FIRST_DENSE_LAYERS: usize = 2;

/// `FASTVIDEO_H3_SOL_ATTN=1` (or `sol`) records the Stage-1 Sol route.
pub fn sol_attn_requested(value: Option<&str>) -> bool {
    match value.map(str::trim) {
        Some("1") => true,
        Some(v) => v.eq_ignore_ascii_case("sol"),
        None => false,
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum H3SolRoute {
    /// First update, and body layer 0 of later updates.
    Dense,
    /// Body layers 1..=49 of updates 1..=3.
    Sol { tau: f64 },
}

/// Stage-1 Spark route: update 0 is dense; later updates keep layer 0 dense
/// and send the rest to Sol with tau 1.0, then 1.25, then 1.5.
pub fn stage1_route(forward: usize, layer: usize) -> Result<H3SolRoute, String> {
    if forward >= STAGE1_FORWARDS {
        return Err(format!(
            "h3 sol: stage-1 forward {forward} is past {STAGE1_FORWARDS} updates"
        ));
    }
    if layer >= LAYERS_PER_FORWARD {
        return Err(format!(
            "h3 sol: layer {layer} is past {LAYERS_PER_FORWARD} body layers"
        ));
    }
    if forward == 0 || layer == 0 {
        return Ok(H3SolRoute::Dense);
    }
    Ok(H3SolRoute::Sol {
        tau: STAGE1_TAUS[forward - 1],
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn first_update_is_dense() {
        assert_eq!(stage1_route(0, 0).unwrap(), H3SolRoute::Dense);
        assert_eq!(stage1_route(0, 49).unwrap(), H3SolRoute::Dense);
    }

    #[test]
    fn later_updates_keep_layer_zero_dense() {
        assert_eq!(stage1_route(1, 0).unwrap(), H3SolRoute::Dense);
        assert_eq!(stage1_route(1, 1).unwrap(), H3SolRoute::Sol { tau: 1.0 });
        assert_eq!(stage1_route(2, 49).unwrap(), H3SolRoute::Sol { tau: 1.25 });
        assert_eq!(stage1_route(3, 1).unwrap(), H3SolRoute::Sol { tau: 1.5 });
        assert!(stage1_route(4, 0).is_err());
        assert!(stage1_route(1, 50).is_err());
    }

    #[test]
    fn env_is_off_until_sol_or_one() {
        assert!(!sol_attn_requested(None));
        assert!(!sol_attn_requested(Some("off")));
        assert!(sol_attn_requested(Some("1")));
        assert!(sol_attn_requested(Some("sol")));
    }

    #[test]
    fn rtx_teacache_knobs_match_the_toml() {
        assert_eq!(RTX_TEACACHE_THRESHOLD, 0.10);
        assert_eq!(RTX_TEACACHE_RETAIN_STEPS, 5);
        assert_eq!(RTX_TEACACHE_COOLDOWN_STEPS, 1);
        assert_eq!(RTX_FIRST_DENSE_STEPS, 10);
        assert_eq!(RTX_FIRST_DENSE_LAYERS, 2);
    }
}
