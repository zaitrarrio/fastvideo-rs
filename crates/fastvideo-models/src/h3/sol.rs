//! MiniMax-H3 Sol-Attn and cache contracts from NVlabs/Sana `sol-engine`
//! (`models/minimax_h3/Sol-H3` and `models/minimax_h3.toml`).
//!
//! The one-GPU Sol-H3 recipe stays dense. This module records the Spark
//! Stage-1 Sol-Attn route (`--ref-stage1-attn sol`), the RTX 4090/5090
//! 50-step Sol-Attn route, and the RTX TeaCache knobs. Body layers on the
//! Sol route use the shared Sol-Attn kernel (`thresh_type=diag`). The RTX
//! TeaCache signal and reuse payload are unpublished, so those knobs stay
//! a record.

use super::packing::H3PackedLayout;

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
pub const RTX_TAU: f64 = 1.0;

/// `FASTVIDEO_H3_SOL_CACHE=teacache` (or `1`) logs [`TEACACHE_GAP`] and stays dense.
pub const TEACACHE_GAP: &str = "h3 sol: RTX TeaCache stays unported \
(threshold 0.10 retain 5 cooldown 1 are recorded; the similarity signal and \
reuse payload are unpublished in the Spark README, sol-engine snapshots, and \
this crate). H3 is guidance-distilled: one forward per step, no CFG pair";

/// Which Stage-1 Sol-Attn host route an env value selects.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum H3SolAttnPolicy {
    Off,
    /// Spark / Sol-H3: 4 updates, taus 1 / 1.25 / 1.5.
    Spark,
    /// RTX 4090/5090: first 10 steps dense, first 2 layers dense, tau 1.0.
    Rtx,
}

/// `FASTVIDEO_H3_SOL_ATTN=1` / `sol` / `spark` is the 4-update Spark route.
/// `rtx` is the 50-step RTX route. Sol layers call the Sol-Attn kernel.
pub fn sol_attn_policy(value: Option<&str>) -> H3SolAttnPolicy {
    match value.map(str::trim) {
        Some("1") => H3SolAttnPolicy::Spark,
        Some(v) if v.eq_ignore_ascii_case("sol") || v.eq_ignore_ascii_case("spark") => {
            H3SolAttnPolicy::Spark
        }
        Some(v) if v.eq_ignore_ascii_case("rtx") => H3SolAttnPolicy::Rtx,
        _ => H3SolAttnPolicy::Off,
    }
}

/// `FASTVIDEO_H3_SOL_ATTN=1` (or `sol` / `spark` / `rtx`) records a Sol route.
pub fn sol_attn_requested(value: Option<&str>) -> bool {
    !matches!(sol_attn_policy(value), H3SolAttnPolicy::Off)
}

/// `FASTVIDEO_H3_SOL_CACHE=teacache` (or `1`) asks for the unpublished RTX skips.
pub fn teacache_requested(value: Option<&str>) -> bool {
    match value.map(str::trim) {
        Some("1") => true,
        Some(v) => v.eq_ignore_ascii_case("teacache"),
        None => false,
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum H3SolRoute {
    /// A dense video self-attention call.
    Dense,
    /// A Sol-Attn call at this tau (`thresh_type=diag`).
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

/// RTX 4090/5090 route: the first 10 steps stay dense. Later steps keep
/// layers 0 and 1 dense and send the rest to Sol at tau 1.0.
pub fn rtx_route(step: usize, layer: usize) -> Result<H3SolRoute, String> {
    if layer >= LAYERS_PER_FORWARD {
        return Err(format!(
            "h3 sol: layer {layer} is past {LAYERS_PER_FORWARD} body layers"
        ));
    }
    if step < RTX_FIRST_DENSE_STEPS || layer < RTX_FIRST_DENSE_LAYERS {
        return Ok(H3SolRoute::Dense);
    }
    Ok(H3SolRoute::Sol { tau: RTX_TAU })
}

/// Route for an env policy. Spark steps past the 4 published updates stay dense.
pub fn policy_route(
    policy: H3SolAttnPolicy,
    step: usize,
    layer: usize,
) -> Result<H3SolRoute, String> {
    match policy {
        H3SolAttnPolicy::Off => Ok(H3SolRoute::Dense),
        H3SolAttnPolicy::Spark if step >= STAGE1_FORWARDS => Ok(H3SolRoute::Dense),
        H3SolAttnPolicy::Spark => stage1_route(step, layer),
        H3SolAttnPolicy::Rtx => rtx_route(step, layer),
    }
}

/// Official `sol_attn` sink spans and the query rows recomputed with dense FA.
///
/// Spark README: only text and audio are forced sinks; cond/ref video is not.
/// Each span is one `interface.py` `_sink_block_range`. `text_query_rows=dense`
/// plus the Spark audio-query subset use the same spans.
pub fn attn_spans(layout: &H3PackedLayout) -> (Vec<(usize, usize)>, Vec<(usize, usize)>) {
    let mut spans = Vec::new();
    if layout.text.len > 0 {
        spans.push((layout.text.start, layout.text.len));
    }
    if layout.audio.len > 0 {
        spans.push((layout.audio.start, layout.audio.len));
    }
    (spans.clone(), spans)
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
    fn rtx_keeps_the_first_ten_steps_and_two_layers_dense() {
        assert_eq!(rtx_route(0, 49).unwrap(), H3SolRoute::Dense);
        assert_eq!(rtx_route(9, 49).unwrap(), H3SolRoute::Dense);
        assert_eq!(rtx_route(10, 0).unwrap(), H3SolRoute::Dense);
        assert_eq!(rtx_route(10, 1).unwrap(), H3SolRoute::Dense);
        assert_eq!(rtx_route(10, 2).unwrap(), H3SolRoute::Sol { tau: 1.0 });
        assert_eq!(rtx_route(49, 49).unwrap(), H3SolRoute::Sol { tau: 1.0 });
        assert!(rtx_route(10, 50).is_err());
    }

    #[test]
    fn env_is_off_until_sol_or_one() {
        assert_eq!(sol_attn_policy(None), H3SolAttnPolicy::Off);
        assert_eq!(sol_attn_policy(Some("off")), H3SolAttnPolicy::Off);
        assert_eq!(sol_attn_policy(Some("1")), H3SolAttnPolicy::Spark);
        assert_eq!(sol_attn_policy(Some("sol")), H3SolAttnPolicy::Spark);
        assert_eq!(sol_attn_policy(Some("spark")), H3SolAttnPolicy::Spark);
        assert_eq!(sol_attn_policy(Some("rtx")), H3SolAttnPolicy::Rtx);
        assert!(!sol_attn_requested(None));
        assert!(sol_attn_requested(Some("1")));
        assert!(sol_attn_requested(Some("rtx")));
    }

    #[test]
    fn teacache_env_is_off_until_teacache_or_one() {
        assert!(!teacache_requested(None));
        assert!(!teacache_requested(Some("")));
        assert!(!teacache_requested(Some("off")));
        assert!(teacache_requested(Some("1")));
        assert!(teacache_requested(Some("teacache")));
        assert!(TEACACHE_GAP.contains("unpublished"));
        assert!(TEACACHE_GAP.contains("no CFG pair"));
    }

    #[test]
    fn rtx_teacache_knobs_match_the_toml() {
        assert_eq!(RTX_TEACACHE_THRESHOLD, 0.10);
        assert_eq!(RTX_TEACACHE_RETAIN_STEPS, 5);
        assert_eq!(RTX_TEACACHE_COOLDOWN_STEPS, 1);
        assert_eq!(RTX_FIRST_DENSE_STEPS, 10);
        assert_eq!(RTX_FIRST_DENSE_LAYERS, 2);
        assert_eq!(RTX_TAU, 1.0);
    }

    #[test]
    fn spark_steps_past_four_stay_dense() {
        assert_eq!(
            policy_route(H3SolAttnPolicy::Spark, 4, 10).unwrap(),
            H3SolRoute::Dense
        );
        assert_eq!(
            policy_route(H3SolAttnPolicy::Spark, 1, 2).unwrap(),
            H3SolRoute::Sol { tau: 1.0 }
        );
        assert_eq!(
            policy_route(H3SolAttnPolicy::Off, 1, 2).unwrap(),
            H3SolRoute::Dense
        );
    }

    #[test]
    fn t2va_spans_are_the_text_audio_prefix() {
        let l = super::super::packing::H3PackedLayout::new(3, (2, 4, 4), 2, [1, 2, 2]).unwrap();
        let (sinks, queries) = attn_spans(&l);
        assert_eq!(sinks, vec![(0, 3), (3, 4)]);
        assert_eq!(queries, sinks);
    }

    #[test]
    fn fl2va_spans_skip_the_cond_rows() {
        let l = super::super::packing::H3PackedLayout::with_keyframes(
            3,
            (2, 4, 4),
            2,
            [1, 2, 2],
            &[super::super::packing::KeyframeAnchor::First],
        )
        .unwrap();
        let (sinks, _) = attn_spans(&l);
        assert_eq!(sinks, vec![(0, 3), (7, 4)]);
        assert_eq!(l.cond, super::super::packing::RowRange { start: 3, len: 4 });
    }
}
