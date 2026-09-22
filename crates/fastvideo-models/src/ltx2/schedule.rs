//! LTX-2 flow-match sigma schedules, host side.
//!
//! The distilled checkpoint is not driven by a step count: it was distilled
//! against one fixed sigma list, and the scheduler is configured
//! (`use_dynamic_shifting=false`, `shift_terminal=null`) to pass that list
//! through untouched. Anything else — a linspace, the dev model's
//! resolution-dependent shift — "quietly costs quality" (diffusers ltx2.md).
//!
//! The dev schedule is here too because the two-stage recipe and the oracle
//! both need it. References: diffusers `pipelines/ltx2/utils.py:27,36`,
//! `pipeline_ltx2.py:1335-1360`, `scheduling_flow_match_euler_discrete.py:343-377`;
//! Lightricks `ltx_pipelines/utils/constants.py:17,20`,
//! `ltx_core/components/schedulers.py:21-57`.

use super::config::Ltx2SchedulerConfig;

/// Ancestral Euler knobs for LTX-2.5 stage 1 (`eta=1`, `s_noise=1`, noise RNG
/// seeded `pipeline_seed + 10000` in the reference).
#[derive(Debug, Clone, Copy)]
pub struct AncestralOpts {
    pub eta: f64,
    pub s_noise: f64,
    pub noise_seed: u64,
}

/// Stage 1 (or single-stage) distilled schedule: 8 model evaluations.
pub const DISTILLED_SIGMA_VALUES: [f64; 8] = [1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875];

/// Distilled list including the terminal 0 — FastVideo's
/// `_distilled_subset_sigmas` indexes this 9-long table.
const DISTILLED_SIGMA_WITH_TERMINAL: [f64; 9] =
    [1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0];

/// Stage 2 refinement after the ×2 latent upsampler: the tail of the list
/// above, 3 evaluations. The stage-1 result is re-noised to the first entry.
pub const STAGE_2_DISTILLED_SIGMA_VALUES: [f64; 3] = [0.909375, 0.725, 0.421875];

/// A resolved schedule: `sigmas` has one more entry than `timesteps` (the
/// terminal 0), exactly like the diffusers scheduler after `set_timesteps`.
#[derive(Debug, Clone, PartialEq)]
pub struct Ltx2Schedule {
    pub sigmas: Vec<f64>,
    /// What the transformer receives: `sigma * num_train_timesteps`.
    pub timesteps: Vec<f64>,
    pub num_train_timesteps: usize,
}

impl Ltx2Schedule {
    /// Wrap an explicit sigma list (no terminal zero) the way
    /// `FlowMatchEulerDiscreteScheduler.set_timesteps(sigmas=…)` does when both
    /// dynamic shifting and the terminal stretch are disabled.
    pub fn from_sigmas(sigmas: &[f64], num_train_timesteps: usize) -> Self {
        let n = num_train_timesteps as f64;
        let timesteps = sigmas.iter().map(|s| s * n).collect();
        let mut sigmas = sigmas.to_vec();
        sigmas.push(0.0);
        Self { sigmas, timesteps, num_train_timesteps }
    }

    /// The 8-step distilled schedule.
    pub fn distilled() -> Self {
        Self::from_sigmas(&DISTILLED_SIGMA_VALUES, 1000)
    }

    /// Distilled subset for `steps` ∈ [1, 8]: same algorithm as FastVideo
    /// `_distilled_subset_sigmas` — preserve endpoints, pick interior indices
    /// that minimize `(max_gap, last_gap, sum_gap²)`.
    ///
    /// Used by LTX-2.3 for stage-1 when `num_inference_steps` is 5 (the `5+2`
    /// recipe) rather than the full 8.
    pub fn distilled_subset(steps: usize) -> Result<Self, String> {
        let max_steps = DISTILLED_SIGMA_WITH_TERMINAL.len() - 1;
        if steps < 1 || steps > max_steps {
            return Err(format!("ltx2 distilled subset supports steps in [1, {max_steps}], got {steps}"));
        }
        if steps == max_steps {
            return Ok(Self::distilled());
        }
        let max_index = DISTILLED_SIGMA_WITH_TERMINAL.len() - 1;
        let interior_count = steps - 1;
        let interiors: Vec<usize> = (1..max_index).collect();
        let mut best_key: Option<(f64, f64, f64)> = None;
        let mut best: Option<Vec<usize>> = None;
        for combo in combinations(&interiors, interior_count) {
            let mut candidate = Vec::with_capacity(steps + 1);
            candidate.push(0);
            candidate.extend_from_slice(&combo);
            candidate.push(max_index);
            let gaps: Vec<f64> = candidate
                .windows(2)
                .map(|w| DISTILLED_SIGMA_WITH_TERMINAL[w[0]] - DISTILLED_SIGMA_WITH_TERMINAL[w[1]])
                .collect();
            let key = (
                gaps.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
                *gaps.last().unwrap_or(&0.0),
                gaps.iter().map(|g| g * g).sum::<f64>(),
            );
            if best_key.map(|k| key < k).unwrap_or(true) {
                best_key = Some(key);
                best = Some(candidate);
            }
        }
        let indices = best.ok_or_else(|| "ltx2: failed to build distilled subset".to_string())?;
        // Drop the terminal 0; `from_sigmas` appends it.
        let sigmas: Vec<f64> = indices[..indices.len() - 1]
            .iter()
            .map(|&i| DISTILLED_SIGMA_WITH_TERMINAL[i])
            .collect();
        Ok(Self::from_sigmas(&sigmas, 1000))
    }

    /// The 3-step stage-2 schedule.
    pub fn distilled_stage_2() -> Self {
        Self::from_sigmas(&STAGE_2_DISTILLED_SIGMA_VALUES, 1000)
    }

    /// Stage-2 refine: only 2 or 3 steps are supported (FastVideo LTX-2 refine).
    /// The 2-step recipe keeps `[0.909375, 0.421875]` (omits `0.725`).
    pub fn distilled_stage_2_steps(steps: usize) -> Result<Self, String> {
        match steps {
            3 => Ok(Self::distilled_stage_2()),
            2 => Ok(Self::from_sigmas(&[STAGE_2_DISTILLED_SIGMA_VALUES[0], STAGE_2_DISTILLED_SIGMA_VALUES[2]], 1000)),
            n => Err(format!("ltx2 refine supports only 2 or 3 steps, got {n}")),
        }
    }

    /// The dev model's schedule for `num_inference_steps` steps at a given
    /// video token count: `linspace(1, 1/N, N)`, exponential time shift by
    /// `mu(seq_len)`, then a stretch so the last sigma equals `shift_terminal`.
    pub fn dev(cfg: &Ltx2SchedulerConfig, num_inference_steps: usize, video_seq_len: usize) -> Self {
        let n = num_inference_steps.max(1);
        let mut sigmas: Vec<f64> = (0..n)
            .map(|i| if n == 1 { 1.0 } else { 1.0 + (1.0 / n as f64 - 1.0) * i as f64 / (n - 1) as f64 })
            .collect();
        if cfg.use_dynamic_shifting {
            let mu = dynamic_shift_mu(cfg, video_seq_len);
            // exponential: exp(mu) / (exp(mu) + (1/t - 1)); linear: mu / (mu + (1/t - 1)).
            let k = if cfg.exponential_time_shift { mu.exp() } else { mu };
            for s in &mut sigmas {
                *s = k / (k + (1.0 / *s - 1.0));
            }
        } else {
            for s in &mut sigmas {
                *s = cfg.shift * *s / (1.0 + (cfg.shift - 1.0) * *s);
            }
        }
        if let Some(terminal) = cfg.shift_terminal {
            let scale = (1.0 - sigmas[n - 1]) / (1.0 - terminal);
            for s in &mut sigmas {
                *s = 1.0 - (1.0 - *s) / scale;
            }
        }
        Self::from_sigmas(&sigmas, cfg.num_train_timesteps)
    }

    pub fn num_steps(&self) -> usize {
        self.timesteps.len()
    }

    /// The model input at step `i`, bit-for-bit what diffusers hands the
    /// transformer: sigmas are cast to float32 *before* the ×1000.
    pub fn timestep_f32(&self, i: usize) -> f32 {
        self.sigmas[i] as f32 * self.num_train_timesteps as f32
    }

    /// Euler step size at step `i`: `sigma[i+1] - sigma[i]` (negative).
    pub fn dt(&self, i: usize) -> f64 {
        self.sigmas[i + 1] - self.sigmas[i]
    }

    /// `x ← x + dt · v`, both streams use the same rule and the same sigmas.
    pub fn euler_step(&self, i: usize, sample: &mut [f32], velocity: &[f32]) {
        assert_eq!(sample.len(), velocity.len(), "euler_step length mismatch");
        let dt = self.dt(i) as f32;
        for (x, v) in sample.iter_mut().zip(velocity) {
            *x += dt * v;
        }
    }

    /// Rectified-flow `x0` from a velocity prediction at noise level `sigma`
    /// (`x += dt·v` Euler convention used by the DiT).
    pub fn denoised_from_velocity(sample: f32, velocity: f32, sigma: f64) -> f32 {
        sample - (sigma as f32) * velocity
    }

    /// One `EulerAncestralDiffusionStep` (rectified flow). When `sigma_next` is
    /// zero, `sample` is replaced with `denoised`. If `eta > 0`, `noise` must
    /// be the same length as `sample` (drawn with `s_noise` scaling).
    pub fn ancestral_step(
        sample: &mut [f32],
        denoised: &[f32],
        sigma: f64,
        sigma_next: f64,
        eta: f64,
        s_noise: f64,
        noise: Option<&[f32]>,
    ) {
        assert_eq!(sample.len(), denoised.len());
        if sigma_next == 0.0 {
            sample.copy_from_slice(denoised);
            return;
        }
        let downstep_ratio = 1.0 + (sigma_next / sigma - 1.0) * eta;
        let sigma_down = sigma_next * downstep_ratio;
        let scale = sigma_down / sigma;
        let blend = 1.0 - scale;
        for (x, d) in sample.iter_mut().zip(denoised) {
            *x = (scale * f64::from(*x) + blend * f64::from(*d)) as f32;
        }
        if eta > 0.0 {
            let noise = noise.expect("ancestral renoise needs a noise vector when eta > 0");
            assert_eq!(noise.len(), sample.len());
            let alpha_next = 1.0 - sigma_next;
            let alpha_down = 1.0 - sigma_down;
            let renoise_coeff = (sigma_next * sigma_next - sigma_down * sigma_down * alpha_next * alpha_next / (alpha_down * alpha_down))
                .max(0.0)
                .sqrt();
            let factor = alpha_next / alpha_down;
            for (x, n) in sample.iter_mut().zip(noise) {
                *x = (factor * f64::from(*x) + f64::from(*n) * s_noise * renoise_coeff) as f32;
            }
        }
    }
}

/// `calculate_shift`: a line through `(base_seq_len, base_shift)` and
/// `(max_seq_len, max_shift)`, **not clamped** — 24576 tokens gives mu ≈ 9.38.
pub fn dynamic_shift_mu(cfg: &Ltx2SchedulerConfig, video_seq_len: usize) -> f64 {
    let m = (cfg.max_shift - cfg.base_shift) / (cfg.max_image_seq_len as f64 - cfg.base_image_seq_len as f64);
    let b = cfg.base_shift - m * cfg.base_image_seq_len as f64;
    video_seq_len as f64 * m + b
}

/// Stage-2 entry: blend the upsampled stage-1 latent with fresh noise at
/// `noise_scale = sigmas[0]` (`pipeline_ltx2.py:716-722`).
pub fn renoise(latent: &mut [f32], noise: &[f32], noise_scale: f32) {
    assert_eq!(latent.len(), noise.len(), "renoise length mismatch");
    for (x, n) in latent.iter_mut().zip(noise) {
        *x = noise_scale * n + (1.0 - noise_scale) * *x;
    }
}

/// Lexicographic combinations of `items` taken `k` at a time (no itertools).
fn combinations(items: &[usize], k: usize) -> Vec<Vec<usize>> {
    if k == 0 {
        return vec![vec![]];
    }
    if k > items.len() {
        return vec![];
    }
    let mut out = Vec::new();
    let mut stack: Vec<(usize, Vec<usize>)> = vec![(0, Vec::new())];
    while let Some((start, prefix)) = stack.pop() {
        if prefix.len() == k {
            out.push(prefix);
            continue;
        }
        let need = k - prefix.len();
        let remaining = items.len().saturating_sub(start);
        if remaining < need {
            continue;
        }
        // Push in reverse so the forward order stays lexicographic when popping.
        for i in (start..=items.len() - need).rev() {
            let mut next = prefix.clone();
            next.push(items[i]);
            stack.push((i + 1, next));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol
    }

    #[test]
    fn distilled_is_the_published_list() {
        let s = Ltx2Schedule::distilled();
        assert_eq!(s.num_steps(), 8);
        assert_eq!(s.sigmas.len(), 9);
        assert_eq!(s.sigmas[0], 1.0);
        assert_eq!(s.sigmas[8], 0.0);
        // Four small steps of 1/160 near pure noise, then three large ones.
        for i in 0..4 {
            assert!(close(s.dt(i), -0.00625, 1e-12), "dt[{i}] = {}", s.dt(i));
        }
        assert!(close(s.dt(4), -0.065_625, 1e-12));
        assert!(close(s.dt(5), -0.184_375, 1e-12));
        assert!(close(s.dt(6), -0.303_125, 1e-12));
        assert!(close(s.dt(7), -0.421_875, 1e-12));
        // The steps sum to exactly -1: noise all the way to data.
        let total: f64 = (0..8).map(|i| s.dt(i)).sum();
        assert!(close(total, -1.0, 1e-12));
        assert_eq!(s.timesteps, vec![1000.0, 993.75, 987.5, 981.25, 975.0, 909.375, 725.0, 421.875]);
    }

    #[test]
    fn stage_2_is_the_tail_of_stage_1() {
        let (a, b) = (Ltx2Schedule::distilled(), Ltx2Schedule::distilled_stage_2());
        assert_eq!(b.num_steps(), 3);
        assert_eq!(&a.sigmas[5..], &b.sigmas[..]);
        assert_eq!(STAGE_2_DISTILLED_SIGMA_VALUES[0], 0.909_375);
    }

    #[test]
    fn distilled_subset_matches_fastvideo_5_and_8() {
        let s8 = Ltx2Schedule::distilled_subset(8).unwrap();
        assert_eq!(s8, Ltx2Schedule::distilled());
        // FastVideo 5-step: indices [0,4,5,6,7,8] → drop terminal for from_sigmas.
        let s5 = Ltx2Schedule::distilled_subset(5).unwrap();
        assert_eq!(s5.num_steps(), 5);
        assert_eq!(s5.sigmas, vec![1.0, 0.975, 0.909_375, 0.725, 0.421_875, 0.0]);
        assert!(Ltx2Schedule::distilled_subset(0).is_err());
        assert!(Ltx2Schedule::distilled_subset(9).is_err());
    }

    #[test]
    fn stage_2_two_step_omits_mid_sigma() {
        let s = Ltx2Schedule::distilled_stage_2_steps(2).unwrap();
        assert_eq!(s.sigmas, vec![0.909_375, 0.421_875, 0.0]);
        assert_eq!(Ltx2Schedule::distilled_stage_2_steps(3).unwrap(), Ltx2Schedule::distilled_stage_2());
        assert!(Ltx2Schedule::distilled_stage_2_steps(1).is_err());
        assert!(Ltx2Schedule::distilled_stage_2_steps(4).is_err());
    }

    #[test]
    fn model_timestep_goes_through_f32() {
        let s = Ltx2Schedule::distilled();
        assert_eq!(s.timestep_f32(0), 1000.0);
        assert_eq!(s.timestep_f32(7), 421.875); // 27/64 is exact in binary
        assert_eq!(s.timestep_f32(1), 0.99375_f32 * 1000.0_f32);
        assert!((s.timestep_f32(1) - 993.75).abs() < 1e-3);
    }

    #[test]
    fn mu_is_linear_and_unclamped() {
        let cfg = Ltx2SchedulerConfig::ltx2_19b();
        assert!(close(dynamic_shift_mu(&cfg, 1024), 0.95, 1e-12));
        assert!(close(dynamic_shift_mu(&cfg, 4096), 2.05, 1e-12));
        // 768x512x121 → 6144 tokens: 0.95 + 5120 · 1.1/3072.
        assert!(close(dynamic_shift_mu(&cfg, 6144), 2.783_333_333_333_333, 1e-12));
        // 1536x1024x121 → 24576 tokens, far past max_shift.
        assert!(close(dynamic_shift_mu(&cfg, 24576), 9.383_333_333_333_333, 1e-12));
    }

    #[test]
    fn dev_schedule_matches_hand_computation() {
        let cfg = Ltx2SchedulerConfig::ltx2_19b();
        // N=4 at 4096 tokens: mu = 2.05, e^mu = 7.767901…
        //   t = [1, .75, .5, .25] → e/(e + 1/t - 1) = [1, .958856, .885947, .721396]
        //   stretch by (1 - .721396)/0.9 → [1, .867083, .631569, .1]
        let s = Ltx2Schedule::dev(&cfg, 4, 4096);
        let want = [1.0, 0.867_083_205_818_382_4, 0.631_568_571_232_125_7, 0.1, 0.0];
        for (got, want) in s.sigmas.iter().zip(want) {
            assert!(close(*got, want, 1e-12), "{got} vs {want}");
        }

        // N=8 at the default 6144 tokens.
        let s = Ltx2Schedule::dev(&cfg, 8, 6144);
        let want = [
            1.0,
            0.973_913_245_449_646_9,
            0.939_833_316_743_069_5,
            0.893_421_801_79,
            0.826_507_140_760_079_6,
            0.721_651_017_517_138,
            0.533_814_741_329_259_1,
            0.1,
            0.0,
        ];
        for (got, want) in s.sigmas.iter().zip(want) {
            assert!(close(*got, want, 1e-12), "{got} vs {want}");
        }
        assert!(s.sigmas.windows(2).all(|w| w[0] > w[1]));
    }

    #[test]
    fn distilled_scheduler_config_leaves_sigmas_alone() {
        // With the distilled config the generic path degenerates to a linspace:
        // shift = 1 is the identity and there is no terminal stretch.
        let cfg = Ltx2SchedulerConfig::ltx2_19b_distilled();
        let s = Ltx2Schedule::dev(&cfg, 4, 6144);
        assert_eq!(s.sigmas, vec![1.0, 0.75, 0.5, 0.25, 0.0]);
    }

    #[test]
    fn euler_walks_a_constant_velocity_to_data() {
        // With v = noise - data constant, x_T = noise lands exactly on data.
        let s = Ltx2Schedule::distilled();
        let (noise, data) = (3.0_f32, -1.0_f32);
        let mut x = [noise];
        for i in 0..s.num_steps() {
            s.euler_step(i, &mut x, &[noise - data]);
        }
        assert!((x[0] - data).abs() < 1e-5, "{}", x[0]);
    }

    #[test]
    fn renoise_blend() {
        let mut x = [2.0_f32, -2.0];
        renoise(&mut x, &[1.0, 1.0], 0.75);
        assert_eq!(x, [1.25, 0.25]);
    }

    #[test]
    fn ancestral_step_matches_hand_formula() {
        let (sigma, sigma_next) = (0.975_f64, 0.909_375);
        let (x0, v0) = (1.2_f32, -0.4_f32);
        let denoised = Ltx2Schedule::denoised_from_velocity(x0, v0, sigma);
        assert!((denoised - (x0 - sigma as f32 * v0)).abs() < 1e-6);
        let eta = 1.0;
        let downstep_ratio = 1.0 + (sigma_next / sigma - 1.0) * eta;
        let sigma_down = sigma_next * downstep_ratio;
        let scale = sigma_down / sigma;
        let blend = 1.0 - scale;
        let mut x = [x0];
        let mut want = scale as f32 * x0 + blend as f32 * denoised;
        let _alpha = 1.0 - sigma;
        let alpha_next = 1.0 - sigma_next;
        let alpha_down = 1.0 - sigma_down;
        let renoise_coeff = (sigma_next * sigma_next - sigma_down * sigma_down * alpha_next * alpha_next / (alpha_down * alpha_down))
            .max(0.0)
            .sqrt();
        let noise = 0.5_f32;
        want = (alpha_next / alpha_down) as f32 * want + noise * renoise_coeff as f32;
        Ltx2Schedule::ancestral_step(&mut x, &[denoised], sigma, sigma_next, eta, 1.0, Some(&[noise]));
        assert!((x[0] - want).abs() < 1e-5, "{} vs {want}", x[0]);
        let mut terminal = [x0];
        Ltx2Schedule::ancestral_step(&mut terminal, &[denoised], sigma, 0.0, eta, 1.0, None);
        assert!((terminal[0] - denoised).abs() < 1e-6);
    }
}
