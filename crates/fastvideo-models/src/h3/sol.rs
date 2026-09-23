//! MiniMax-H3 Sol-Attn and cache contracts from NVlabs/Sana `sol-engine`
//! (`models/minimax_h3/Sol-H3`, `models/minimax_h3.toml`, and
//! `models/minimax_h3/RTX4090/teacache.py`).
//!
//! The one-GPU Sol-H3 recipe stays dense. This module records the Spark
//! Stage-1 Sol-Attn route (`--ref-stage1-attn sol`), the RTX 4090/5090
//! 50-step Sol-Attn route, and the RTX TeaCache controller. Body layers on
//! the Sol route use the shared Sol-Attn kernel (`thresh_type=diag`).
//! `FASTVIDEO_H3_SOL_CACHE=teacache` skips the block stack when the
//! published residual controller says so.

use super::packing::H3PackedLayout;

/// Body layers in MiniMax-H3.
pub const LAYERS_PER_FORWARD: usize = 50;

/// Sol-H3 / Spark Stage-1 transformer updates.
pub const STAGE1_FORWARDS: usize = 4;

/// Sol-H3-Spark draft canvas (`models/minimax_h3/Sol-H3-Spark` README).
/// The H3×2 upscaler and H3-to-LTX adapter run when their checkpoints are
/// configured. `FASTVIDEO_LTX2_WEIGHTS` then runs the joint 3-step LTX refiner
/// on the bridged latent and the H3 PCM.
pub const SPARK_DRAFT_WIDTH: usize = 672;
pub const SPARK_DRAFT_HEIGHT: usize = 384;
pub const SPARK_DRAFT_FRAMES: usize = 124;
pub const SPARK_OUTPUT_WIDTH: usize = 1344;
pub const SPARK_OUTPUT_HEIGHT: usize = 768;
pub const SPARK_OUTPUT_FRAMES: usize = 121;

/// Taus on the three Sol updates after the dense first update.
pub const STAGE1_TAUS: [f64; 3] = [1.0, 1.25, 1.5];

/// RTX 4090/5090 cell (`models/minimax_h3.toml` `[rtx4090.policy]`).
pub const RTX_TEACACHE_THRESHOLD: f64 = 0.10;
pub const RTX_TEACACHE_RETAIN_STEPS: usize = 5;
pub const RTX_TEACACHE_COOLDOWN_STEPS: usize = 1;
pub const RTX_FIRST_DENSE_STEPS: usize = 10;
pub const RTX_FIRST_DENSE_LAYERS: usize = 2;
pub const RTX_TAU: f64 = 1.0;

/// Identity Horner polynomial from `H3_TEACACHE_COEFFICIENTS` default `1.0,0.0`.
pub const RTX_TEACACHE_COEFFICIENTS: [f64; 2] = [1.0, 0.0];

/// Official MiniMax-H3 eval count: 50 sigma points drive 49 forwards.
pub const RTX_TEACACHE_NUM_FORWARDS: usize = 49;

/// `FASTVIDEO_H3_SOL_CACHE=teacache` (or `1`) runs the RTX residual TeaCache.
pub const TEACACHE_APPLIED: &str = "h3 sol teacache: threshold 0.10 retain 5 \
cooldown 1 (block-0 AdaLN-modulated RMS-norm probe, whole-stack residual, \
one forward per step, no CFG pair)";

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

/// `FASTVIDEO_H3_SOL_CACHE=teacache` (or `1`) turns the RTX residual skips on.
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

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct H3TeaDecision {
    pub compute: bool,
    pub reason: &'static str,
    pub relative_l1: Option<f64>,
    pub indicator: Option<f64>,
    pub accumulator: f64,
}

/// RTX TeaCache from `models/minimax_h3/RTX4090/teacache.py`.
///
/// The signal is block 0's AdaLN-modulated RMS-norm hidden. The reuse
/// payload is the residual across the 50-block stack (`hidden - input`).
/// H3 is guidance-distilled: one accumulator, no CFG pair, no Wan
/// timestep-projection.
#[derive(Debug, Clone)]
pub struct H3TeaCache {
    pub threshold: f64,
    pub retain_steps: usize,
    pub cooldown_steps: usize,
    pub num_forwards: usize,
    pub coefficients: Vec<f64>,
    has_signal: bool,
    has_residual: bool,
    acc: f64,
}

impl H3TeaCache {
    pub fn official(num_forwards: usize) -> Result<Self, String> {
        Self::new(
            RTX_TEACACHE_THRESHOLD,
            RTX_TEACACHE_RETAIN_STEPS,
            RTX_TEACACHE_COOLDOWN_STEPS,
            num_forwards,
            RTX_TEACACHE_COEFFICIENTS.to_vec(),
        )
    }

    pub fn new(
        threshold: f64,
        retain_steps: usize,
        cooldown_steps: usize,
        num_forwards: usize,
        coefficients: Vec<f64>,
    ) -> Result<Self, String> {
        if threshold <= 0.0 {
            return Err("h3 teacache requires a positive threshold".into());
        }
        if coefficients.is_empty() {
            return Err("h3 teacache requires at least one coefficient".into());
        }
        Ok(Self {
            threshold,
            retain_steps,
            cooldown_steps,
            num_forwards,
            coefficients,
            has_signal: false,
            has_residual: false,
            acc: 0.0,
        })
    }

    pub fn needs_signal(&self, step: usize) -> bool {
        self.force_reason(step).is_none()
    }

    /// `relative_l1` is sum |signal − previous| / sum |previous|.
    /// Ignored on a forced step.
    pub fn decide(&mut self, step: usize, relative_l1: f64) -> H3TeaDecision {
        let (compute, reason, relative_l1, indicator) =
            if let Some(reason) = self.force_reason(step) {
                if reason == "warmup" || reason == "cooldown" {
                    self.acc = 0.0;
                }
                (true, reason, None, None)
            } else {
                let indicator = poly(&self.coefficients, relative_l1);
                self.acc += indicator;
                let compute = self.acc >= self.threshold;
                if compute {
                    self.acc = 0.0;
                }
                (
                    compute,
                    if compute {
                        "threshold"
                    } else {
                        "below_threshold"
                    },
                    Some(relative_l1),
                    Some(indicator),
                )
            };
        self.has_signal = true;
        H3TeaDecision {
            compute,
            reason,
            relative_l1,
            indicator,
            accumulator: self.acc,
        }
    }

    pub fn note_computed(&mut self) {
        self.acc = 0.0;
        self.has_residual = true;
    }

    fn force_reason(&self, step: usize) -> Option<&'static str> {
        if step < self.retain_steps {
            Some("warmup")
        } else if step >= self.num_forwards.saturating_sub(self.cooldown_steps) {
            Some("cooldown")
        } else if !self.has_signal || !self.has_residual {
            Some("initialize")
        } else {
            None
        }
    }
}

/// Sum |current − previous| / sum |previous|. Matches RTX `teacache.py`.
pub fn relative_l1(current: &[f32], previous: &[f32]) -> f64 {
    let mut num = 0.0;
    let mut den = 0.0;
    for (a, b) in current.iter().zip(previous.iter()) {
        num += (f64::from(*a) - f64::from(*b)).abs();
        den += f64::from(*b).abs();
    }
    num / den.max(1.0e-8)
}

fn poly(coefficients: &[f64], x: f64) -> f64 {
    let mut value = 0.0;
    for coefficient in coefficients {
        value = value * x + coefficient;
    }
    value
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
        assert!(TEACACHE_APPLIED.contains("block-0 AdaLN"));
        assert!(TEACACHE_APPLIED.contains("no CFG pair"));
    }

    #[test]
    fn teacache_warms_then_skips_then_cools() {
        let mut cache = H3TeaCache::official(10).unwrap();
        for step in 0..5 {
            let d = cache.decide(step, 9.0);
            assert!(d.compute, "{step}");
            assert_eq!(d.reason, "warmup");
            cache.note_computed();
        }
        let skip = cache.decide(5, 0.01);
        assert!(!skip.compute);
        assert_eq!(skip.reason, "below_threshold");
        let hit = cache.decide(6, 0.10);
        assert!(hit.compute);
        assert_eq!(hit.reason, "threshold");
        cache.note_computed();
        let tail = cache.decide(9, 0.0);
        assert!(tail.compute);
        assert_eq!(tail.reason, "cooldown");
    }

    #[test]
    fn teacache_relative_l1_is_sum_over_sum() {
        let prev = [2.0f32, 0.0];
        let cur = [4.0f32, 2.0];
        // sum |d| = 4, sum |prev| = 2
        assert!((relative_l1(&cur, &prev) - 2.0).abs() < 1e-12);
    }

    #[test]
    fn spark_draft_is_half_the_official_h3_canvas() {
        assert_eq!(
            (SPARK_DRAFT_WIDTH, SPARK_DRAFT_HEIGHT, SPARK_DRAFT_FRAMES),
            (672, 384, 124)
        );
        assert_eq!(
            (SPARK_OUTPUT_WIDTH, SPARK_OUTPUT_HEIGHT, SPARK_OUTPUT_FRAMES),
            (1344, 768, 121)
        );
        assert_eq!(SPARK_DRAFT_WIDTH * 2, SPARK_OUTPUT_WIDTH);
        assert_eq!(SPARK_DRAFT_HEIGHT * 2, SPARK_OUTPUT_HEIGHT);
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
