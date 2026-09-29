//! The multimodal guider of `ltx_core` (`components/guiders.py`
//! `MultiModalGuider`, Lightricks/LTX-2 `fd4ded7`) and the defaults the dev
//! pipelines run it with (`ltx_pipelines/utils/constants.py`).
//!
//! One guided step runs up to four passes of the same DiT on the same
//! latents (`utils/denoisers.py` `_guided_denoise`), each giving an `x0`
//! (`X0Model`'s denoised output):
//!
//! | pass | video context | audio context | perturbation |
//! |---|---|---|---|
//! | `cond` | positive | positive | none |
//! | `uncond` (CFG) | negative | the audio guider's negative, else positive | none |
//! | `ptb` (STG) | positive | positive | video self-attention of `stg_blocks` replaced by its value projection |
//! | `mod` (modality) | positive | positive | every audio↔video cross-attention skipped |
//!
//! and combines them per stream in float32:
//!
//! ```text
//! pred = cond + (cfg − 1)(cond − uncond) + stg (cond − ptb) + (mod − 1)(cond − mod)
//! pred *= r · std(cond) / std(pred) + (1 − r)        (when r = rescale_scale ≠ 0)
//! ```
//!
//! `std` is over the whole tensor (unbiased), then the result is stored in
//! the latent dtype (bf16). A pass is run only when some stream's guider
//! needs it; a stream whose scale leaves a term at its neutral value simply
//! drops that term.

/// `MultiModalGuiderParams` (`guiders.py`).
#[derive(Debug, Clone, PartialEq)]
pub struct GuiderParams {
    /// CFG scale; 1 is off.
    pub cfg_scale: f32,
    /// STG scale; 0 is off.
    pub stg_scale: f32,
    /// Blocks whose self-attention the STG pass perturbs.
    pub stg_blocks: Vec<usize>,
    /// Rescale toward the conditional prediction's std; 0 is off.
    pub rescale_scale: f32,
    /// Modality (isolated audio / video) scale; 1 is off.
    pub modality_scale: f32,
    /// Reuse the previous step's guided `x0` except every `skip_step + 1`-th
    /// step; 0 never skips.
    pub skip_step: usize,
}

impl Default for GuiderParams {
    /// The dataclass defaults: a guider that returns `cond` unchanged.
    fn default() -> Self {
        Self {
            cfg_scale: 1.0,
            stg_scale: 0.0,
            stg_blocks: Vec::new(),
            rescale_scale: 0.0,
            modality_scale: 1.0,
            skip_step: 0,
        }
    }
}

impl GuiderParams {
    /// `do_unconditional_generation`.
    pub fn needs_uncond(&self) -> bool {
        !close(self.cfg_scale, 1.0)
    }

    /// `do_perturbed_generation`.
    pub fn needs_ptb(&self) -> bool {
        !close(self.stg_scale, 0.0)
    }

    /// `do_isolated_modality_generation`.
    pub fn needs_mod(&self) -> bool {
        !close(self.modality_scale, 1.0)
    }

    /// `should_skip_step`.
    pub fn skips(&self, step: usize) -> bool {
        self.skip_step != 0 && !step.is_multiple_of(self.skip_step + 1)
    }

    /// The weights of `(cond, uncond, ptb, mod)` in `pred` before the
    /// rescale: `cond + (c−1)(cond−u) + s(cond−p) + (m−1)(cond−m)`. They sum
    /// to 1. A pass that does not run gets weight 0 (its term is the
    /// reference's `0.0` placeholder times a zero scale).
    pub fn weights(&self) -> [f32; 4] {
        let (c, s, m) = (self.cfg_scale - 1.0, self.stg_scale, self.modality_scale - 1.0);
        [1.0 + c + s + m, -c, -s, -m]
    }

    /// The rescale factor from the two standard deviations (`calculate`):
    /// `r · std(cond) / std(pred) + (1 − r)`; `None` when `r` is 0.
    pub fn rescale_factor(&self, std_cond: f64, std_pred: f64) -> Option<f32> {
        if self.rescale_scale == 0.0 {
            return None;
        }
        let r = self.rescale_scale as f64;
        Some((r * (std_cond / std_pred) + (1.0 - r)) as f32)
    }
}

/// `math.isclose` at its default tolerance (rel 1e-9): exact for our scales.
fn close(a: f32, b: f32) -> bool {
    (a as f64 - b as f64).abs() <= 1e-9 * (a.abs().max(b.abs()) as f64)
}

/// `LTX_2_4_PARAMS.num_inference_steps` (`constants.py`): the dev stage-1
/// step count of every checkpoint at or above model version 2.4 — the LTX-2.5
/// transformers declare `model_version = "2.5.0"` (`detect_params`).
pub const LTX25_DEV_STEPS: usize = 30;

/// `LTX_2_4_PARAMS.video_guider_params`: CFG 3, STG 1 on block 28, rescale
/// 0.7, modality 3, no skipped steps. `a2vid_two_stage.py` guides the video
/// with these (`--video-*` / `--a2v-guidance-scale` defaults) and the frozen
/// audio with [`GuiderParams::default`].
pub fn ltx25_video_guider() -> GuiderParams {
    GuiderParams {
        cfg_scale: 3.0,
        stg_scale: 1.0,
        stg_blocks: vec![28],
        rescale_scale: 0.7,
        modality_scale: 3.0,
        skip_step: 0,
    }
}

/// `DEFAULT_NEGATIVE_PROMPT` (`constants.py`), the CLI's `--negative-prompt`
/// default.
pub const DEFAULT_NEGATIVE_PROMPT: &str = "has_subtitles, has_blurbox, transition from black, transition to black, speech_ending_short, \
blurry, out of focus, overexposed, underexposed, low contrast, washed out colors, excessive noise, \
grainy texture, poor lighting, flickering, motion blur, distorted proportions, unnatural skin tones, \
deformed facial features, asymmetrical face, missing facial features, extra limbs, disfigured hands, \
wrong hand count, artifacts around text, inconsistent perspective, camera shake, incorrect depth of \
field, background too sharp, background clutter, distracting reflections, harsh shadows, inconsistent \
lighting direction, color banding, cartoonish rendering, 3D CGI look, unrealistic materials, uncanny \
valley effect, incorrect ethnicity, wrong gender, exaggerated expressions, wrong gaze direction, \
mismatched lip sync, silent or muted audio, distorted voice, robotic voice, echo, background noise, \
off-sync audio, incorrect dialogue, added dialogue, repetitive speech, jittery movement, awkward \
pauses, incorrect timing, unnatural transitions, inconsistent framing, tilted camera, flat lighting, \
inconsistent tone, cinematic oversaturation, stylized filters, or AI artifacts.";

/// Unbiased standard deviation over every element (`torch.Tensor.std()`),
/// accumulated in f64 (two passes).
pub fn std_unbiased(x: &[f32]) -> f64 {
    let n = x.len();
    if n < 2 {
        return f64::NAN;
    }
    let mean = x.iter().map(|&v| v as f64).sum::<f64>() / n as f64;
    let ss: f64 = x.iter().map(|&v| (v as f64 - mean).powi(2)).sum();
    (ss / (n - 1) as f64).sqrt()
}

/// Host reference of one guided combine (`MultiModalGuider.calculate`), for
/// the tests and the oracle: float32 terms as the reference evaluates them,
/// then the rescale.
pub fn combine_host(p: &GuiderParams, cond: &[f32], uncond: &[f32], ptb: &[f32], md: &[f32]) -> Vec<f32> {
    let (c, s, m) = (p.cfg_scale - 1.0, p.stg_scale, p.modality_scale - 1.0);
    let mut pred: Vec<f32> = (0..cond.len())
        .map(|i| {
            let x = cond[i];
            let mut y = x + c * (x - uncond[i]);
            y += s * (x - ptb[i]);
            y + m * (x - md[i])
        })
        .collect();
    if let Some(f) = p.rescale_factor(std_unbiased(cond), std_unbiased(&pred)) {
        for v in &mut pred {
            *v *= f;
        }
    }
    pred
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_match_the_2_4_params() {
        let v = ltx25_video_guider();
        assert_eq!(
            (v.cfg_scale, v.stg_scale, v.rescale_scale, v.modality_scale, v.skip_step),
            (3.0, 1.0, 0.7, 3.0, 0)
        );
        assert_eq!(v.stg_blocks, vec![28]);
        assert!(v.needs_uncond() && v.needs_ptb() && v.needs_mod());
        let a = GuiderParams::default();
        assert!(!a.needs_uncond() && !a.needs_ptb() && !a.needs_mod());
        assert_eq!(a.weights(), [1.0, 0.0, 0.0, 0.0]);
        assert_eq!(a.rescale_factor(1.0, 2.0), None);
        assert_eq!(LTX25_DEV_STEPS, 30);
        assert!(DEFAULT_NEGATIVE_PROMPT.starts_with("has_subtitles, has_blurbox,"));
        assert!(DEFAULT_NEGATIVE_PROMPT.ends_with("stylized filters, or AI artifacts."));
        assert!(DEFAULT_NEGATIVE_PROMPT.contains("mismatched lip sync, silent or muted audio"));
    }

    #[test]
    fn weights_sum_to_one_and_reproduce_the_formula() {
        let p = ltx25_video_guider();
        let w = p.weights();
        assert!((w.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        assert_eq!(w, [6.0, -2.0, -1.0, -2.0]);
        let (cond, u, pt, md) = ([1.0f32, -2.0, 0.5], [0.5f32, -1.0, 0.0], [0.9f32, -2.5, 1.0], [2.0f32, 0.0, 0.25]);
        let host = combine_host(&GuiderParams { rescale_scale: 0.0, ..p.clone() }, &cond, &u, &pt, &md);
        for i in 0..3 {
            let lin = w[0] * cond[i] + w[1] * u[i] + w[2] * pt[i] + w[3] * md[i];
            assert!((lin - host[i]).abs() < 1e-5, "{i}: {lin} vs {}", host[i]);
        }
    }

    #[test]
    fn rescale_pulls_the_std_toward_cond() {
        let p = ltx25_video_guider();
        let cond = [1.0f32, -1.0, 2.0, -2.0];
        let pred_raw: Vec<f32> = cond.iter().map(|v| v * 3.0).collect();
        let f = p.rescale_factor(std_unbiased(&cond), std_unbiased(&pred_raw)).unwrap();
        // r / 3 + (1 − r) with r = 0.7.
        assert!((f - (0.7 / 3.0 + 0.3) as f32).abs() < 1e-6);
        assert!((std_unbiased(&[1.0, 2.0, 3.0, 4.0]) - (5.0f64 / 3.0).sqrt()).abs() < 1e-12);
    }

    #[test]
    fn skip_step_keeps_every_nth() {
        let p = GuiderParams { skip_step: 2, ..GuiderParams::default() };
        let run: Vec<bool> = (0..7).map(|i| !p.skips(i)).collect();
        assert_eq!(run, [true, false, false, true, false, false, true]);
        assert!(!GuiderParams::default().skips(5));
    }
}
