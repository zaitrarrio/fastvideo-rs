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

/// Stage 1 (or single-stage) distilled schedule: 8 model evaluations.
pub const DISTILLED_SIGMA_VALUES: [f64; 8] = [1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875];

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

    /// The 3-step stage-2 schedule.
    pub fn distilled_stage_2() -> Self {
        Self::from_sigmas(&STAGE_2_DISTILLED_SIGMA_VALUES, 1000)
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
}
