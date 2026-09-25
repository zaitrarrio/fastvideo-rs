//! MiniMax-H3 Sol-Attn and cache contracts from NVlabs/Sana `sol-engine`
//! (`models/minimax_h3/Sol-H3/h3_runtime/{engine,sparse_attention}.py`,
//! `models/minimax_h3/RTX5090/{adapter,teacache}.py` and
//! `models/minimax_h3/Sol-H3-Spark/runtime/stage1_ops/sol.py`).
//!
//! Attention routes, one per [`H3SolAttnPolicy`]:
//!
//! * 4-step `sol-h3` T2V/I2V/Ref2VA runs **dense** attention by default. The
//!   Sol-H3 engine refuses sparse attention on one GPU (`engine.py`: "SOL
//!   attention requires 2, 4, or 8 GPU processes").
//!   `FASTVIDEO_H3_SOL_ATTN=1|sol|engine` opts into the engine's multi-GPU
//!   policy ([`H3SolAttnPolicy::Engine`]): forward 0 dense, blocks 0 and 1
//!   dense, tau 1.0 elsewhere, and a `prefix` sink over every row before the
//!   target video (`[0, video.start)` in packed order).
//! * `sol-h3-rtx` is the RTX 5090 cell ([`H3SolAttnPolicy::Rtx`]): forwards
//!   0-9 dense, blocks 0 and 1 dense, tau 1.0, and an exact **text-only** KV
//!   sink whose query rows are recomputed densely (`adapter.py`).
//! * `sol-h3-spark` runs VSA 0.9 with Sol-Attn off. `FASTVIDEO_H3_SOL_ATTN=spark`
//!   opts into the Spark Ref2VA draft ladder ([`H3SolAttnPolicy::Spark`]:
//!   forward 0 dense, then block 0 dense and tau 1 / 1.25 / 1.5). That route
//!   permutes Q/K/V to `[visual | text+audio]` and sinks the suffix, exactly as
//!   `stage1_ops/sol.py` does; it is rejected on the other recipes.
//!
//! Every sink is ONE contiguous `(start, len)` range in the coordinates of the
//! tensors handed to the kernel (packed order `[text | cond | audio | video]`
//! for Engine/Rtx, permuted order for Spark). The same rows are the dense
//! query rows. `FASTVIDEO_H3_SOL_CACHE=teacache` adds the RTX residual TeaCache.

use super::config::{TAG_AUDIO, TAG_TEXT, TAG_VIDEO};
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

/// Spark Ref2VA draft taus on the three Sol updates after the dense first
/// update (`stage1_ops/sol.py` `TAUS`). Only [`H3SolAttnPolicy::Spark`] reads it.
pub const STAGE1_TAUS: [f64; 3] = [1.0, 1.25, 1.5];

/// Sol-H3 engine multi-GPU policy (`engine.py` `sparse_attention.install`
/// for T2V/I2V): `tau=1.0, dense_steps=1, dense_layers=2, sink_mode="prefix"`.
pub const ENGINE_TAU: f64 = 1.0;
pub const ENGINE_DENSE_STEPS: usize = 1;
pub const ENGINE_DENSE_LAYERS: usize = 2;

/// RTX 5090 cell (`RTX5090/run_minimax_h3_gpu.sh`, `rtx5090_sol.toml`).
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

/// Which Sol-Attn route a forward takes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum H3SolAttnPolicy {
    /// Dense attention everywhere.
    Off,
    /// Sol-H3 engine: forward 0 dense, blocks 0-1 dense, tau 1.0, prefix sink.
    Engine,
    /// Spark Ref2VA draft: forward 0 dense, block 0 dense, taus 1 / 1.25 / 1.5,
    /// permuted `[visual | text+audio]` suffix sink.
    Spark,
    /// RTX 5090: forwards 0-9 dense, blocks 0-1 dense, tau 1.0, text sink.
    Rtx,
}

/// `FASTVIDEO_H3_SOL_ATTN`: `1` / `sol` / `engine` is the Sol-H3 engine
/// policy, `spark` the Spark draft ladder, `rtx` the RTX cell. Anything else
/// (including `off` and unset) is dense.
pub fn sol_attn_policy(value: Option<&str>) -> H3SolAttnPolicy {
    match value.map(str::trim) {
        Some("1") => H3SolAttnPolicy::Engine,
        Some(v) if v.eq_ignore_ascii_case("sol") || v.eq_ignore_ascii_case("engine") => {
            H3SolAttnPolicy::Engine
        }
        Some(v) if v.eq_ignore_ascii_case("spark") => H3SolAttnPolicy::Spark,
        Some(v) if v.eq_ignore_ascii_case("rtx") => H3SolAttnPolicy::Rtx,
        _ => H3SolAttnPolicy::Off,
    }
}

/// Env wins. Unset `FASTVIDEO_H3_SOL_ATTN`: `sol-h3-rtx` is the RTX cell and
/// every other recipe (including 4-step `sol-h3` and `sol-h3-spark`) is
/// dense / VSA. `ref2va` is whether the request runs the Ref2VA transformer.
///
/// Errors: the Spark ladder outside `sol-h3-spark`, and the engine policy on
/// Ref2VA (the engine only accepts `sol_bsa` with a text/audio block mask
/// there, which this route does not implement).
pub fn recipe_sol_attn_policy(
    recipe: Option<&str>,
    env: Option<&str>,
    ref2va: bool,
) -> Result<H3SolAttnPolicy, String> {
    let policy = match env {
        Some(_) => sol_attn_policy(env),
        None => match recipe {
            Some(name) if super::lora::is_sol_h3_rtx_recipe(name) => H3SolAttnPolicy::Rtx,
            _ => H3SolAttnPolicy::Off,
        },
    };
    let spark_recipe = recipe.is_some_and(super::lora::is_sol_h3_spark_recipe);
    match policy {
        H3SolAttnPolicy::Spark if !spark_recipe => Err(format!(
            "h3 sol: FASTVIDEO_H3_SOL_ATTN=spark is the Spark draft ladder; recipe {} is not sol-h3-spark (use 1|sol|engine or rtx)",
            recipe.unwrap_or("auto")
        )),
        H3SolAttnPolicy::Engine if ref2va => Err(
            "h3 sol: the Sol-H3 engine policy is T2V/I2V only (Ref2VA needs sol_bsa with a text/audio block mask); unset FASTVIDEO_H3_SOL_ATTN for dense Ref2VA".into(),
        ),
        other => Ok(other),
    }
}

/// `FASTVIDEO_H3_SOL_ATTN=1` (or `sol` / `engine` / `spark` / `rtx`) records a Sol route.
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

fn check_layer(layer: usize) -> Result<(), String> {
    if layer >= LAYERS_PER_FORWARD {
        return Err(format!(
            "h3 sol: layer {layer} is past {LAYERS_PER_FORWARD} body layers"
        ));
    }
    Ok(())
}

/// Spark Ref2VA draft route (`stage1_ops/sol.py` `route`): update 0 is dense;
/// later updates keep layer 0 dense and send the rest to Sol with tau 1.0,
/// then 1.25, then 1.5.
pub fn stage1_route(forward: usize, layer: usize) -> Result<H3SolRoute, String> {
    if forward >= STAGE1_FORWARDS {
        return Err(format!(
            "h3 sol: stage-1 forward {forward} is past {STAGE1_FORWARDS} updates"
        ));
    }
    check_layer(layer)?;
    if forward == 0 || layer == 0 {
        return Ok(H3SolRoute::Dense);
    }
    Ok(H3SolRoute::Sol {
        tau: STAGE1_TAUS[forward - 1],
    })
}

/// Sol-H3 engine route (`sparse_attention.py` `_declined_contract`): `step`
/// is the per-request forward counter (0 on the first forward, +1 per
/// transformer forward), `layer` the 0-based block index. Declines as
/// `warmup_step` while `step < dense_steps`, then as `dense_layer` while
/// `layer < dense_layers`.
pub fn engine_route(step: usize, layer: usize) -> Result<H3SolRoute, String> {
    check_layer(layer)?;
    if step < ENGINE_DENSE_STEPS || layer < ENGINE_DENSE_LAYERS {
        return Ok(H3SolRoute::Dense);
    }
    Ok(H3SolRoute::Sol { tau: ENGINE_TAU })
}

/// RTX route (`adapter.py` `_dense_policy`: `step_index < 10 or layer_index
/// < 2`). `step_index` counts model forwards from 0 per request.
pub fn rtx_route(step: usize, layer: usize) -> Result<H3SolRoute, String> {
    check_layer(layer)?;
    if step < RTX_FIRST_DENSE_STEPS || layer < RTX_FIRST_DENSE_LAYERS {
        return Ok(H3SolRoute::Dense);
    }
    Ok(H3SolRoute::Sol { tau: RTX_TAU })
}

/// Route for a policy. `step` is the 0-based transformer forward of the
/// request. Spark steps past the 4 published updates stay dense.
pub fn policy_route(
    policy: H3SolAttnPolicy,
    step: usize,
    layer: usize,
) -> Result<H3SolRoute, String> {
    match policy {
        H3SolAttnPolicy::Off => Ok(H3SolRoute::Dense),
        H3SolAttnPolicy::Engine => engine_route(step, layer),
        H3SolAttnPolicy::Spark if step >= STAGE1_FORWARDS => Ok(H3SolRoute::Dense),
        H3SolAttnPolicy::Spark => stage1_route(step, layer),
        H3SolAttnPolicy::Rtx => rtx_route(step, layer),
    }
}

/// Permutation that groups visual rows, then a single text+audio suffix.
///
/// Matches `models/minimax_h3/Sol-H3-Spark/runtime/stage1_ops/sol.py`
/// `sink_plan`: cond/ref video stays visual (not a sink); native relative
/// order inside each part is preserved.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct H3SolSinkPlan {
    pub permutation: Vec<usize>,
    pub inverse: Vec<usize>,
    pub sink_start: usize,
    pub sink_tokens: usize,
    pub text_sink_tokens: usize,
    pub audio_sink_tokens: usize,
}

/// Where a Sol layer's exact KV sink sits, and whether Q/K/V are permuted
/// before the kernel. `sink` is one contiguous `(start, len)` range in the
/// coordinates of the tensors the kernel sees; the same rows are recomputed
/// with dense attention as queries.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct H3SolSinkSpec {
    pub sink: Option<(usize, usize)>,
    /// `Some` only for [`H3SolAttnPolicy::Spark`]: gather rows by
    /// `permutation` before attention, by `inverse` after.
    pub plan: Option<H3SolSinkPlan>,
}

/// `sink_mode="prefix"` (`sparse_attention.py` `_sink_range`): every row
/// before the target-video tail, i.e. text, condition video and audio.
pub fn prefix_sink(layout: &H3PackedLayout) -> Option<(usize, usize)> {
    (layout.video.start > 0).then_some((0, layout.video.start))
}

/// RTX text sink (`adapter.py` `_sol_varlen`): the contiguous text rows.
pub fn text_sink(layout: &H3PackedLayout) -> Option<(usize, usize)> {
    (layout.text.len > 0).then_some((layout.text.start, layout.text.len))
}

/// The sink (and permutation) a policy uses on `layout`.
pub fn sink_spec(
    policy: H3SolAttnPolicy,
    layout: &H3PackedLayout,
) -> Result<H3SolSinkSpec, String> {
    match policy {
        H3SolAttnPolicy::Off => Ok(H3SolSinkSpec {
            sink: None,
            plan: None,
        }),
        H3SolAttnPolicy::Engine => Ok(H3SolSinkSpec {
            sink: prefix_sink(layout),
            plan: None,
        }),
        H3SolAttnPolicy::Rtx => Ok(H3SolSinkSpec {
            sink: text_sink(layout),
            plan: None,
        }),
        H3SolAttnPolicy::Spark => {
            let plan = sink_plan(layout)?;
            Ok(H3SolSinkSpec {
                sink: Some((plan.sink_start, plan.sink_tokens)),
                plan: Some(plan),
            })
        }
    }
}

/// One-line description of `spec` for the pipeline log.
pub fn describe_sink(policy: H3SolAttnPolicy, spec: &H3SolSinkSpec) -> String {
    let range = spec
        .sink
        .map_or("none".to_string(), |(s, l)| format!("[{s}, {})", s + l));
    match policy {
        H3SolAttnPolicy::Off => "h3 sol sink: none (dense)".into(),
        H3SolAttnPolicy::Engine => {
            format!("h3 sol sink: prefix {range} in packed order, dense prefix query rows")
        }
        H3SolAttnPolicy::Rtx => {
            format!("h3 sol sink: text {range} in packed order, dense text query rows")
        }
        H3SolAttnPolicy::Spark => format!(
            "h3 sol sink: device gather to [visual | text+audio], suffix {range}, dense suffix query rows, inverse gather after"
        ),
    }
}

pub fn sink_plan(layout: &H3PackedLayout) -> Result<H3SolSinkPlan, String> {
    let tags = &layout.token_tags;
    if tags.is_empty() || tags.iter().any(|&tag| tag > TAG_AUDIO) {
        return Err("h3 sol: native joint token tags must be visual0/text1/audio2".into());
    }
    let visual: Vec<usize> = tags
        .iter()
        .enumerate()
        .filter(|(_, tag)| **tag == TAG_VIDEO)
        .map(|(i, _)| i)
        .collect();
    let sinks: Vec<usize> = tags
        .iter()
        .enumerate()
        .filter(|(_, tag)| **tag == TAG_TEXT || **tag == TAG_AUDIO)
        .map(|(i, _)| i)
        .collect();
    if visual.is_empty() || sinks.is_empty() {
        return Err("h3 sol: suffix sink requires both visual and text/audio rows".into());
    }
    let start = visual.len();
    // `spill = visual[start // 64 * 64:]` must be generated video, i.e. inside
    // `video_indices[condition_video_rows:]`: the packed target-video tail.
    let spill = &visual[start / 64 * 64..];
    let target = layout.video.start..layout.video.end();
    if !spill.iter().all(|index| target.contains(index)) {
        return Err("h3 sol: sink KV boundary spill must contain generated video only".into());
    }
    let mut permutation = visual;
    permutation.extend(&sinks);
    let mut inverse = vec![0; tags.len()];
    for (destination, source) in permutation.iter().enumerate() {
        inverse[*source] = destination;
    }
    Ok(H3SolSinkPlan {
        permutation,
        inverse,
        sink_start: start,
        sink_tokens: sinks.len(),
        text_sink_tokens: tags.iter().filter(|&&tag| tag == TAG_TEXT).count(),
        audio_sink_tokens: tags.iter().filter(|&&tag| tag == TAG_AUDIO).count(),
    })
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

    /// Forget the signal, the residual and the accumulator: the next
    /// request starts as the first one did (its buffers were freed).
    pub fn reset(&mut self) {
        self.acc = 0.0;
        self.has_signal = false;
        self.has_residual = false;
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
    relative_l1_from_sums(num, den)
}

/// `sum|current - previous| / clamp_min(sum|previous|, 1e-8)` from the two
/// sums (the device reduction returns exactly these).
pub fn relative_l1_from_sums(abs_diff_sum: f64, abs_prev_sum: f64) -> f64 {
    abs_diff_sum / abs_prev_sum.max(1.0e-8)
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
    use super::super::packing::{H3PackedLayout, KeyframeAnchor, RowRange};
    use super::*;

    /// text 3 | audio 4 | video 8.
    fn t2va() -> H3PackedLayout {
        H3PackedLayout::new(3, (2, 4, 4), 2, [1, 2, 2]).unwrap()
    }

    /// text 3 | cond 4 | audio 4 | video 8.
    fn fl2va() -> H3PackedLayout {
        H3PackedLayout::with_keyframes(3, (2, 4, 4), 2, [1, 2, 2], &[KeyframeAnchor::First])
            .unwrap()
    }

    #[test]
    fn first_update_is_dense() {
        assert_eq!(stage1_route(0, 0).unwrap(), H3SolRoute::Dense);
        assert_eq!(stage1_route(0, 49).unwrap(), H3SolRoute::Dense);
    }

    #[test]
    fn spark_ladder_keeps_layer_zero_dense() {
        assert_eq!(stage1_route(1, 0).unwrap(), H3SolRoute::Dense);
        assert_eq!(stage1_route(1, 1).unwrap(), H3SolRoute::Sol { tau: 1.0 });
        assert_eq!(stage1_route(2, 49).unwrap(), H3SolRoute::Sol { tau: 1.25 });
        assert_eq!(stage1_route(3, 1).unwrap(), H3SolRoute::Sol { tau: 1.5 });
        assert!(stage1_route(4, 0).is_err());
        assert!(stage1_route(1, 50).is_err());
    }

    #[test]
    fn engine_clock_is_forward_zero_and_blocks_zero_one_dense() {
        // sparse_attention.py: step < dense_steps(1) -> warmup_step,
        // layer < dense_layers(2) -> dense_layer, else tau 1.0.
        for layer in 0..LAYERS_PER_FORWARD {
            assert_eq!(engine_route(0, layer).unwrap(), H3SolRoute::Dense);
        }
        for step in 1..STAGE1_FORWARDS {
            assert_eq!(engine_route(step, 0).unwrap(), H3SolRoute::Dense);
            assert_eq!(engine_route(step, 1).unwrap(), H3SolRoute::Dense);
            for layer in 2..LAYERS_PER_FORWARD {
                assert_eq!(
                    engine_route(step, layer).unwrap(),
                    H3SolRoute::Sol { tau: 1.0 }
                );
            }
        }
        assert!(engine_route(1, 50).is_err());
        // 3 sparse forwards x 48 layers per 4-step request.
        let sparse = (0..STAGE1_FORWARDS)
            .flat_map(|s| (0..LAYERS_PER_FORWARD).map(move |l| (s, l)))
            .filter(|&(s, l)| engine_route(s, l).unwrap() != H3SolRoute::Dense)
            .count();
        assert_eq!(sparse, 3 * 48);
        assert_eq!(
            (ENGINE_TAU, ENGINE_DENSE_STEPS, ENGINE_DENSE_LAYERS),
            (1.0, 1, 2)
        );
    }

    #[test]
    fn rtx_keeps_the_first_ten_steps_and_two_layers_dense() {
        assert_eq!(rtx_route(0, 49).unwrap(), H3SolRoute::Dense);
        assert_eq!(rtx_route(9, 49).unwrap(), H3SolRoute::Dense);
        assert_eq!(rtx_route(10, 0).unwrap(), H3SolRoute::Dense);
        assert_eq!(rtx_route(10, 1).unwrap(), H3SolRoute::Dense);
        assert_eq!(rtx_route(10, 2).unwrap(), H3SolRoute::Sol { tau: 1.0 });
        assert_eq!(rtx_route(48, 49).unwrap(), H3SolRoute::Sol { tau: 1.0 });
        assert!(rtx_route(10, 50).is_err());
        // 49 forwards: 39 sparse forwards x 48 sparse layers.
        let sparse = (0..RTX_TEACACHE_NUM_FORWARDS)
            .flat_map(|s| (0..LAYERS_PER_FORWARD).map(move |l| (s, l)))
            .filter(|&(s, l)| rtx_route(s, l).unwrap() != H3SolRoute::Dense)
            .count();
        assert_eq!(sparse, 39 * 48);
    }

    #[test]
    fn env_values_pick_the_policy() {
        assert_eq!(sol_attn_policy(None), H3SolAttnPolicy::Off);
        assert_eq!(sol_attn_policy(Some("off")), H3SolAttnPolicy::Off);
        assert_eq!(sol_attn_policy(Some("1")), H3SolAttnPolicy::Engine);
        assert_eq!(sol_attn_policy(Some("sol")), H3SolAttnPolicy::Engine);
        assert_eq!(sol_attn_policy(Some("engine")), H3SolAttnPolicy::Engine);
        assert_eq!(sol_attn_policy(Some("spark")), H3SolAttnPolicy::Spark);
        assert_eq!(sol_attn_policy(Some("rtx")), H3SolAttnPolicy::Rtx);
        assert!(!sol_attn_requested(None));
        assert!(sol_attn_requested(Some("1")));
        assert!(sol_attn_requested(Some("rtx")));
    }

    #[test]
    fn one_gpu_sol_h3_is_dense_by_default() {
        for recipe in ["sol-h3", "sol-h3-t2v", "sol-h3-i2v", "sol-h3-ref2va", "sol-h3-spark"] {
            assert_eq!(
                recipe_sol_attn_policy(Some(recipe), None, false).unwrap(),
                H3SolAttnPolicy::Off,
                "{recipe}"
            );
        }
        assert_eq!(
            recipe_sol_attn_policy(Some("sol-h3-ref2va"), None, true).unwrap(),
            H3SolAttnPolicy::Off
        );
        assert_eq!(
            recipe_sol_attn_policy(Some("sol-h3-rtx"), None, false).unwrap(),
            H3SolAttnPolicy::Rtx
        );
        assert_eq!(
            recipe_sol_attn_policy(Some("4step-vsa"), None, false).unwrap(),
            H3SolAttnPolicy::Off
        );
        assert_eq!(recipe_sol_attn_policy(None, None, false).unwrap(), H3SolAttnPolicy::Off);
    }

    #[test]
    fn explicit_opt_ins_and_rejections() {
        assert_eq!(
            recipe_sol_attn_policy(Some("sol-h3"), Some("1"), false).unwrap(),
            H3SolAttnPolicy::Engine
        );
        assert_eq!(
            recipe_sol_attn_policy(Some("sol-h3-i2v"), Some("sol"), false).unwrap(),
            H3SolAttnPolicy::Engine
        );
        assert_eq!(
            recipe_sol_attn_policy(Some("sol-h3"), Some("rtx"), false).unwrap(),
            H3SolAttnPolicy::Rtx
        );
        assert_eq!(
            recipe_sol_attn_policy(Some("sol-h3-rtx"), Some("off"), false).unwrap(),
            H3SolAttnPolicy::Off
        );
        assert_eq!(
            recipe_sol_attn_policy(Some("sol-h3-spark"), Some("spark"), false).unwrap(),
            H3SolAttnPolicy::Spark
        );
        // The Spark Ref2VA ladder never reaches the T2V/I2V routes.
        for recipe in ["sol-h3", "sol-h3-t2v", "sol-h3-i2v", "sol-h3-rtx"] {
            assert!(recipe_sol_attn_policy(Some(recipe), Some("spark"), false).is_err());
        }
        assert!(recipe_sol_attn_policy(None, Some("spark"), false).is_err());
        // The engine refuses its `sol` backend on Ref2VA.
        assert!(recipe_sol_attn_policy(Some("sol-h3-ref2va"), Some("1"), true).is_err());
    }

    #[test]
    fn policy_route_dispatches_each_clock() {
        use H3SolAttnPolicy::*;
        assert_eq!(policy_route(Off, 20, 20).unwrap(), H3SolRoute::Dense);
        assert_eq!(policy_route(Engine, 0, 20).unwrap(), H3SolRoute::Dense);
        assert_eq!(policy_route(Engine, 1, 1).unwrap(), H3SolRoute::Dense);
        assert_eq!(
            policy_route(Engine, 1, 2).unwrap(),
            H3SolRoute::Sol { tau: 1.0 }
        );
        assert_eq!(
            policy_route(Engine, 3, 49).unwrap(),
            H3SolRoute::Sol { tau: 1.0 }
        );
        assert_eq!(
            policy_route(Spark, 2, 1).unwrap(),
            H3SolRoute::Sol { tau: 1.25 }
        );
        assert_eq!(policy_route(Spark, 4, 10).unwrap(), H3SolRoute::Dense);
        assert_eq!(policy_route(Rtx, 3, 49).unwrap(), H3SolRoute::Dense);
        assert_eq!(
            policy_route(Rtx, 10, 2).unwrap(),
            H3SolRoute::Sol { tau: 1.0 }
        );
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
            assert!(!cache.needs_signal(step));
            let d = cache.decide(step, 9.0);
            assert!(d.compute, "{step}");
            assert_eq!(d.reason, "warmup");
            cache.note_computed();
        }
        assert!(cache.needs_signal(5));
        let skip = cache.decide(5, 0.01);
        assert!(!skip.compute);
        assert_eq!(skip.reason, "below_threshold");
        let hit = cache.decide(6, 0.10);
        assert!(hit.compute, "0.01 + 0.10 reaches 0.10");
        assert_eq!(hit.reason, "threshold");
        assert_eq!(hit.accumulator, 0.0);
        cache.note_computed();
        assert!(!cache.needs_signal(9));
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
        assert!((relative_l1_from_sums(4.0, 2.0) - 2.0).abs() < 1e-12);
        assert_eq!(relative_l1_from_sums(1.0, 0.0), 1.0e8);
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
    fn rtx_knobs_match_the_run_script() {
        assert_eq!(RTX_TEACACHE_THRESHOLD, 0.10);
        assert_eq!(RTX_TEACACHE_RETAIN_STEPS, 5);
        assert_eq!(RTX_TEACACHE_COOLDOWN_STEPS, 1);
        assert_eq!(RTX_TEACACHE_NUM_FORWARDS, 49);
        assert_eq!(RTX_TEACACHE_COEFFICIENTS, [1.0, 0.0]);
        assert_eq!(RTX_FIRST_DENSE_STEPS, 10);
        assert_eq!(RTX_FIRST_DENSE_LAYERS, 2);
        assert_eq!(RTX_TAU, 1.0);
    }

    #[test]
    fn engine_sink_is_the_packed_prefix_before_target_video() {
        let l = t2va();
        assert_eq!(l.text, RowRange { start: 0, len: 3 });
        assert_eq!(l.audio, RowRange { start: 3, len: 4 });
        assert_eq!(l.video.start, 7);
        let spec = sink_spec(H3SolAttnPolicy::Engine, &l).unwrap();
        assert_eq!(spec.sink, Some((0, 7)), "text + audio, never video rows");
        assert!(spec.plan.is_none());
        // FL2VA: text | cond | audio all sit in the prefix.
        let l = fl2va();
        assert_eq!(l.video.start, 11);
        let spec = sink_spec(H3SolAttnPolicy::Engine, &l).unwrap();
        assert_eq!(spec.sink, Some((0, 11)));
        let (s, n) = spec.sink.unwrap();
        assert!((s..s + n).all(|i| i < l.video.start));
    }

    #[test]
    fn rtx_sink_is_text_rows_only() {
        for l in [t2va(), fl2va()] {
            let spec = sink_spec(H3SolAttnPolicy::Rtx, &l).unwrap();
            assert_eq!(spec.sink, Some((l.text.start, l.text.len)));
            assert!(spec.plan.is_none());
            let (s, n) = spec.sink.unwrap();
            assert!((s..s + n).all(|i| l.token_tags[i] == TAG_TEXT));
        }
    }

    #[test]
    fn off_has_no_sink() {
        let spec = sink_spec(H3SolAttnPolicy::Off, &t2va()).unwrap();
        assert_eq!(spec.sink, None);
        assert!(spec.plan.is_none());
    }

    #[test]
    fn spark_sink_is_one_suffix_range_in_permuted_order() {
        let l = t2va();
        let spec = sink_spec(H3SolAttnPolicy::Spark, &l).unwrap();
        let plan = spec.plan.clone().unwrap();
        assert_eq!(spec.sink, Some((l.video.len, 3 + 4)));
        assert_eq!(
            plan.permutation[..plan.sink_start],
            (l.video.start..l.video.end()).collect::<Vec<_>>()
        );
        assert_eq!(&plan.permutation[plan.sink_start..], &[0, 1, 2, 3, 4, 5, 6]);
        // Permuted rows inside the sink are exactly the text + audio rows.
        let (s, n) = spec.sink.unwrap();
        assert!(plan.permutation[s..s + n]
            .iter()
            .all(|&i| l.token_tags[i] != TAG_VIDEO));
        for (dst, &src) in plan.permutation.iter().enumerate() {
            assert_eq!(plan.inverse[src], dst);
        }
    }

    #[test]
    fn spark_spill_rule_matches_upstream() {
        // Upstream: spill = visual[start // 64 * 64:] must be target video.
        // Tiny FL2VA: start < 64, so the cond rows are in the spill.
        assert!(sink_plan(&fl2va()).is_err());
        // One 64-row cond frame + two target frames: no partial block.
        let l = H3PackedLayout::with_keyframes(
            3,
            (2, 16, 16),
            2,
            [1, 2, 2],
            &[KeyframeAnchor::First],
        )
        .unwrap();
        assert_eq!((l.cond.len, l.video.len), (64, 128));
        let plan = sink_plan(&l).unwrap();
        assert_eq!(plan.sink_start, 192);
        assert_eq!(plan.sink_tokens, 3 + 4);
        assert!(plan.permutation[..plan.sink_start]
            .iter()
            .all(|&i| l.token_tags[i] == TAG_VIDEO));
    }

    #[test]
    fn every_sink_is_a_single_contiguous_range() {
        for l in [t2va(), fl2va()] {
            for policy in [
                H3SolAttnPolicy::Off,
                H3SolAttnPolicy::Engine,
                H3SolAttnPolicy::Rtx,
            ] {
                let spec = sink_spec(policy, &l).unwrap();
                if let Some((s, n)) = spec.sink {
                    assert!(n > 0 && s + n <= l.sequence_length());
                }
            }
        }
        assert!(describe_sink(
            H3SolAttnPolicy::Engine,
            &sink_spec(H3SolAttnPolicy::Engine, &t2va()).unwrap()
        )
        .contains("[0, 7)"));
    }
}
