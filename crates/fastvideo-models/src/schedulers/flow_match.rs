//! Flow-matching Euler discrete schedule.
//!
//! Ported from FastVideo `scheduling_flow_match_euler_discrete.py`
//! (modified from Hugging Face Diffusers, Apache-2.0). Uses f64 for the
//! default `np.linspace` path so rounded timesteps match Python.

/// `shift * sigma / (1 + (shift - 1) * sigma)`
pub fn apply_shift(sigma: f64, shift: f64) -> f64 {
    shift * sigma / (1.0 + (shift - 1.0) * sigma)
}

#[derive(Debug, Clone)]
pub struct FlowMatchEulerDiscreteScheduler {
    pub num_train_timesteps: i32,
    pub shift: f64,
    sigma_min: f64,
    sigma_max: f64,
    pub sigmas: Vec<f64>,
    pub timesteps: Vec<f64>,
    step_index: Option<usize>,
}

impl FlowMatchEulerDiscreteScheduler {
    pub fn new(num_train_timesteps: i32, shift: f64) -> Self {
        let mut train_sigmas = Vec::with_capacity(num_train_timesteps as usize);
        // np.linspace(1, N, N, dtype=float32)[::-1] / N, then shift.
        for i in 0..num_train_timesteps {
            let t = (num_train_timesteps - i) as f64;
            let sigma = t / f64::from(num_train_timesteps);
            train_sigmas.push(apply_shift(sigma, shift));
        }
        let sigma_max = train_sigmas[0];
        let sigma_min = *train_sigmas.last().expect("train sigmas");
        Self {
            num_train_timesteps,
            shift,
            sigma_min,
            sigma_max,
            sigmas: train_sigmas,
            timesteps: Vec::new(),
            step_index: None,
        }
    }

    pub fn sigma_max(&self) -> f64 {
        self.sigma_max
    }

    pub fn sigma_min(&self) -> f64 {
        self.sigma_min
    }

    /// Default inference schedule: linspace(t_max, t_min, steps) in f64,
    /// divide by num_train_timesteps, apply shift, append terminal 0.
    pub fn set_timesteps(&mut self, num_inference_steps: usize) {
        let t_max = self.sigma_max() * f64::from(self.num_train_timesteps);
        let t_min = self.sigma_min() * f64::from(self.num_train_timesteps);
        let n = num_inference_steps.max(1);
        let mut sigmas = Vec::with_capacity(n + 1);
        if n == 1 {
            sigmas.push(t_max / f64::from(self.num_train_timesteps));
        } else {
            for i in 0..n {
                let t = t_max + (t_min - t_max) * (i as f64) / ((n - 1) as f64);
                sigmas.push(t / f64::from(self.num_train_timesteps));
            }
        }
        for s in &mut sigmas {
            *s = apply_shift(*s, self.shift);
        }
        let timesteps: Vec<f64> = sigmas
            .iter()
            .map(|s| s * f64::from(self.num_train_timesteps))
            .collect();
        sigmas.push(0.0);
        self.timesteps = timesteps;
        self.sigmas = sigmas;
        self.step_index = None;
    }

    pub fn inference_timesteps(&self) -> &[f64] {
        &self.timesteps
    }

    pub fn inference_sigmas(&self) -> &[f64] {
        &self.sigmas
    }

    /// `prev = sample + (sigma_next - sigma) * model_output`
    pub fn step_euler(&mut self, sample: &[f32], model_output: &[f32]) -> Result<Vec<f32>, String> {
        if sample.len() != model_output.len() {
            return Err("sample / model_output length mismatch".into());
        }
        let idx = self.step_index.unwrap_or(0);
        if idx + 1 >= self.sigmas.len() {
            return Err("step past end of schedule".into());
        }
        let dt = (self.sigmas[idx + 1] - self.sigmas[idx]) as f32;
        let prev: Vec<f32> = sample
            .iter()
            .zip(model_output)
            .map(|(x, v)| x + dt * v)
            .collect();
        self.step_index = Some(idx + 1);
        Ok(prev)
    }
}

/// UniPC on Wan uses this same shifted flow-matching sigma table.
/// The predictor-corrector update itself is Phase 1.
pub fn unipc_sigmas(num_inference_steps: usize, shift: f64) -> Vec<f64> {
    let mut sched = FlowMatchEulerDiscreteScheduler::new(1000, shift);
    sched.set_timesteps(num_inference_steps);
    sched.sigmas
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shift_identity_when_one() {
        assert!((apply_shift(0.5, 1.0) - 0.5).abs() < 1e-12);
    }

    #[test]
    fn wan_1_3b_50_step_schedule() {
        let mut sched = FlowMatchEulerDiscreteScheduler::new(1000, 3.0);
        sched.set_timesteps(50);
        assert_eq!(sched.timesteps.len(), 50);
        assert_eq!(sched.sigmas.len(), 51);
        assert!((sched.sigmas[50] - 0.0).abs() < 1e-12);
        // After shift, sigma_max is apply_shift(1.0, 3.0) = 3/3 = 1.0
        assert!((sched.sigmas[0] - 1.0).abs() < 1e-9);
        assert!(sched.sigmas[1] < sched.sigmas[0]);
    }

    #[test]
    fn euler_step_zero_velocity() {
        let mut sched = FlowMatchEulerDiscreteScheduler::new(1000, 3.0);
        sched.set_timesteps(4);
        let sample = vec![1.0f32, 2.0, 3.0];
        let vel = vec![0.0f32, 0.0, 0.0];
        let out = sched.step_euler(&sample, &vel).unwrap();
        assert_eq!(out, sample);
    }
}
