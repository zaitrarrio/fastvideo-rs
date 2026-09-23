//! rCM (recurrent Consistency Model) scheduler for TurboDiffusion / TurboWan.
//!
//! Port of FastVideo `scheduling_rcm.py`: TrigFlow → RectifiedFlow timesteps,
//! initial noise scaled by `sigmas[0]`, and the SDE update
//! `x ← (1 − t_next)·(x − t_cur·v) + t_next·noise`. Optimized for 1–4 steps.
//!
//! Reference: TurboDiffusion (arXiv:2512.16093).

/// Default intermediate TrigFlow timesteps (visual-quality tuned).
pub const RCM_MID_TIMESTEPS: [f64; 3] = [1.5, 1.4, 1.0];

/// T2V TurboWan default `sigma_max`.
pub const RCM_SIGMA_MAX_T2V: f64 = 80.0;

/// I2V TurboWan default `sigma_max`.
pub const RCM_SIGMA_MAX_I2V: f64 = 200.0;

/// Scalars for one rCM step: `x0 = x − t_cur·v`, then
/// `out = (1 − t_next)·x0 + t_next·noise` (always re-noise except when
/// `t_next == 0`, which collapses to `x0`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RcmStepCoeffs {
    pub t_cur: f64,
    pub t_next: f64,
    /// Model input: `t_cur * num_train_timesteps` (1000).
    pub model_timestep: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RcmSchedule {
    /// Raw RectifiedFlow sigmas including the terminal 0 (`len = steps + 1`).
    pub sigmas: Vec<f64>,
    /// `sigmas[..steps] * num_train_timesteps` — what the DiT receives.
    pub timesteps: Vec<f64>,
    pub num_train_timesteps: usize,
    pub sigma_max: f64,
}

impl RcmSchedule {
    /// Build the 1–4 step schedule. `num_inference_steps` outside that range is
    /// allowed (FastVideo only warns) but quality is untested.
    pub fn new(num_inference_steps: usize, sigma_max: f64) -> Self {
        let n = num_inference_steps.max(1);
        let mid_take = n.saturating_sub(1).min(RCM_MID_TIMESTEPS.len());
        let mut trig: Vec<f64> = Vec::with_capacity(n + 1);
        trig.push(sigma_max.atan());
        trig.extend_from_slice(&RCM_MID_TIMESTEPS[..mid_take]);
        trig.push(0.0);
        // TrigFlow → RectifiedFlow: t = sin(t) / (cos(t) + sin(t)).
        let sigmas: Vec<f64> = trig
            .iter()
            .map(|&t| {
                let (s, c) = t.sin_cos();
                s / (c + s)
            })
            .collect();
        let num_train_timesteps = 1000usize;
        let timesteps: Vec<f64> = sigmas[..sigmas.len() - 1]
            .iter()
            .map(|s| s * num_train_timesteps as f64)
            .collect();
        Self {
            sigmas,
            timesteps,
            num_train_timesteps,
            sigma_max,
        }
    }

    pub fn t2v(num_inference_steps: usize) -> Self {
        Self::new(num_inference_steps, RCM_SIGMA_MAX_T2V)
    }

    pub fn i2v(num_inference_steps: usize) -> Self {
        Self::new(num_inference_steps, RCM_SIGMA_MAX_I2V)
    }

    pub fn num_steps(&self) -> usize {
        self.timesteps.len()
    }

    /// Scale for `x₀ = noise · init_noise_scale` (raw `sigmas[0]`).
    pub fn init_noise_scale(&self) -> f64 {
        self.sigmas[0]
    }

    pub fn step_coeffs(&self, i: usize) -> RcmStepCoeffs {
        assert!(i < self.num_steps(), "rcm step {i} out of range");
        let t_cur = self.sigmas[i];
        let t_next = self.sigmas[i + 1];
        RcmStepCoeffs {
            t_cur,
            t_next,
            model_timestep: (t_cur * self.num_train_timesteps as f64) as f32,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol
    }

    #[test]
    fn four_step_t2v_matches_trigflow_hand() {
        let s = RcmSchedule::t2v(4);
        assert_eq!(s.num_steps(), 4);
        assert_eq!(s.sigmas.len(), 5);
        assert!(close(s.sigmas[4], 0.0, 0.0));
        // atan(80) → RF
        let t0 = 80.0_f64.atan();
        let want0 = t0.sin() / (t0.cos() + t0.sin());
        assert!(close(s.sigmas[0], want0, 1e-12));
        // mid TrigFlow values 1.5, 1.4, 1.0
        for (i, mid) in [1.5_f64, 1.4, 1.0].into_iter().enumerate() {
            let want = mid.sin() / (mid.cos() + mid.sin());
            assert!(close(s.sigmas[i + 1], want, 1e-12), "mid[{i}]");
        }
        assert!(close(s.init_noise_scale(), s.sigmas[0], 0.0));
        let c0 = s.step_coeffs(0);
        assert!(close(c0.t_cur as f64, s.sigmas[0], 0.0));
        assert!(close(c0.model_timestep as f64, s.timesteps[0], 1e-4));
    }

    #[test]
    fn one_step_is_sigma_max_then_zero() {
        let s = RcmSchedule::t2v(1);
        assert_eq!(s.num_steps(), 1);
        assert_eq!(s.sigmas.len(), 2);
        assert_eq!(s.sigmas[1], 0.0);
        let c = s.step_coeffs(0);
        assert_eq!(c.t_next, 0.0);
    }

    #[test]
    fn i2v_uses_larger_sigma_max() {
        let (t2v, i2v) = (RcmSchedule::t2v(4), RcmSchedule::i2v(4));
        assert!(i2v.sigmas[0] > t2v.sigmas[0]);
        assert_eq!(i2v.sigma_max, RCM_SIGMA_MAX_I2V);
    }
}
