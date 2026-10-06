//! SANA-Video sampler: Diffusers `DPMSolverMultistepScheduler` with the
//! flow settings of `SANA-Video_2B_480p_diffusers/scheduler/scheduler_config.json`
//! (`algorithm_type=dpmsolver++`, `solver_order=2`, `solver_type=midpoint`,
//! `prediction_type=flow_prediction`, `use_flow_sigmas`, `flow_shift=8`,
//! `final_sigmas_type=zero`, `lower_order_final`).
//!
//! Each step is a linear combination of sample-shaped tensors
//! ([`DpmStepPlan`]): the device applies it with `CudaTensor::lincomb`, the
//! host reference [`SanaDpmSolver::step_host`] with plain loops.
//!
//! Precision follows the reference: the sigma table is float64 math cast to
//! float32 (`astype(np.float32)`), the timesteps are the float64 `sigma * 1000`
//! truncated to int64, and the per-step scalars are float32 tensor math.

/// Published default (`scheduler_config.json` `flow_shift`).
pub const SANA_FLOW_SHIFT: f64 = 8.0;
pub const SANA_TRAIN_TIMESTEPS: usize = 1000;

/// One sampler update as `x_next = Σ coef_i · term_i`.
///
/// `x0` is the data prediction of this step, `sample − sigma · model_output`
/// (`convert_model_output`, flow prediction); `prev_x0` is the previous
/// step's. `x_next = sample_coef · sample + x0_coef · x0 + prev_x0_coef · prev_x0`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DpmStepPlan {
    pub sigma: f32,
    pub sample_coef: f32,
    pub x0_coef: f32,
    pub prev_x0_coef: f32,
    pub order: usize,
}

#[derive(Debug, Clone)]
pub struct SanaDpmSolver {
    /// `num_inference_steps + 1` float32 sigmas, the last 0.
    sigmas: Vec<f32>,
    /// Integer timesteps handed to the transformer.
    timesteps: Vec<i64>,
    step_index: usize,
    lower_order_nums: usize,
}

impl SanaDpmSolver {
    pub fn new(num_inference_steps: usize, flow_shift: f64) -> Result<Self, String> {
        if num_inference_steps == 0 {
            return Err("SANA-Video: need at least one step".into());
        }
        let n = num_inference_steps;
        let train = SANA_TRAIN_TIMESTEPS as f64;
        // alphas = np.linspace(1, 1 / N_train, n + 1); sigmas = 1 - alphas.
        let mut shifted: Vec<f64> = (0..=n)
            .map(|i| {
                let alpha = 1.0 + (1.0 / train - 1.0) * (i as f64) / (n as f64);
                let s = 1.0 - alpha;
                flow_shift * s / (1.0 + (flow_shift - 1.0) * s)
            })
            .collect();
        // np.flip(...)[:-1]: descending, drop the trailing 0.
        shifted.reverse();
        shifted.truncate(n);
        let timesteps = shifted.iter().map(|s| (s * train) as i64).collect();
        let mut sigmas: Vec<f32> = shifted.iter().map(|&s| s as f32).collect();
        sigmas.push(0.0);
        Ok(Self {
            sigmas,
            timesteps,
            step_index: 0,
            lower_order_nums: 0,
        })
    }

    pub fn sigmas(&self) -> &[f32] {
        &self.sigmas
    }

    pub fn timesteps(&self) -> &[i64] {
        &self.timesteps
    }

    pub fn num_steps(&self) -> usize {
        self.timesteps.len()
    }

    pub fn step_index(&self) -> usize {
        self.step_index
    }

    /// The noise scale of the initial latent (`init_noise_sigma` is 1.0 for
    /// this scheduler: the latents are plain N(0, 1)).
    pub fn init_noise_sigma(&self) -> f32 {
        1.0
    }

    /// Plan the current step and advance. `prev_x0_coef` is 0 on a
    /// first-order step (the first step and the final, `final_sigmas_type =
    /// zero`, step).
    pub fn plan_step(&mut self) -> Result<DpmStepPlan, String> {
        let i = self.step_index;
        let n = self.num_steps();
        if i >= n {
            return Err("SANA-Video sampler: step past the end of the schedule".into());
        }
        // lower_order_final: final step with final_sigmas_type == "zero".
        let lower_order_final = i == n - 1;
        // lower_order_second only applies for fewer than 15 steps.
        let lower_order_second = i + 2 == n && n < 15;
        let first_order = self.lower_order_nums < 1 || lower_order_final;
        let ab = |sigma: f32| (1.0f32 - sigma, sigma);
        let lambda = |sigma: f32| {
            let (a, s) = ab(sigma);
            a.ln() - s.ln()
        };
        let sigma_s0 = self.sigmas[i];
        let sigma_t = self.sigmas[i + 1];
        let (alpha_t, _) = ab(sigma_t);
        let lambda_t = lambda(sigma_t);
        let lambda_s0 = lambda(sigma_s0);
        let h = lambda_t - lambda_s0;
        // alpha_t * (exp(-h) - 1)
        let c = alpha_t * ((-h).exp() - 1.0);
        let plan = if first_order {
            // x_t = (sigma_t / sigma_s) * sample - alpha_t * (exp(-h) - 1) * x0
            DpmStepPlan {
                sigma: sigma_s0,
                sample_coef: sigma_t / sigma_s0,
                x0_coef: -c,
                prev_x0_coef: 0.0,
                order: 1,
            }
        } else {
            let _ = lower_order_second; // order 2 is the maximum here
            let sigma_s1 = self.sigmas[i - 1];
            let lambda_s1 = lambda(sigma_s1);
            let h0 = lambda_s0 - lambda_s1;
            let r0 = h0 / h;
            // D0 = m0, D1 = (1 / r0) * (m0 - m1)
            // x_t = (sigma_t / sigma_s0) * sample - c * D0 - 0.5 * c * D1
            let d1 = 0.5 * c * (1.0 / r0);
            DpmStepPlan {
                sigma: sigma_s0,
                sample_coef: sigma_t / sigma_s0,
                x0_coef: -c - d1,
                prev_x0_coef: d1,
                order: 2,
            }
        };
        if self.lower_order_nums < 2 {
            self.lower_order_nums += 1;
        }
        self.step_index += 1;
        Ok(plan)
    }

    /// Host reference: one step on flat buffers. `prev_x0` is the previous
    /// step's data prediction (ignored on a first-order step). Returns
    /// `(x_next, x0)`; keep `x0` for the next call.
    pub fn step_host(
        &mut self,
        model_output: &[f32],
        sample: &[f32],
        prev_x0: Option<&[f32]>,
    ) -> Result<(Vec<f32>, Vec<f32>), String> {
        if model_output.len() != sample.len() {
            return Err("SANA-Video sampler: model_output / sample length".into());
        }
        let plan = self.plan_step()?;
        let x0: Vec<f32> = sample
            .iter()
            .zip(model_output)
            .map(|(x, v)| x - plan.sigma * v)
            .collect();
        let next = match (plan.order, prev_x0) {
            (1, _) => sample
                .iter()
                .zip(&x0)
                .map(|(s, m)| plan.sample_coef * s + plan.x0_coef * m)
                .collect(),
            (_, Some(p)) if p.len() == sample.len() => sample
                .iter()
                .zip(&x0)
                .zip(p)
                .map(|((s, m), q)| plan.sample_coef * s + plan.x0_coef * m + plan.prev_x0_coef * q)
                .collect(),
            _ => return Err("SANA-Video sampler: second-order step needs the previous x0".into()),
        };
        Ok((next, x0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Diffusers `DPMSolverMultistepScheduler.set_timesteps(50)` with the
    /// published config (`use_flow_sigmas` branch), endpoints recomputed from
    /// the reference formula. The full table is checked against the Python
    /// reference in the GPU parity plan (docs/ports/sana-video.md).
    #[test]
    fn schedule_matches_the_reference_endpoints() {
        let s = SanaDpmSolver::new(50, SANA_FLOW_SHIFT).unwrap();
        assert_eq!(s.sigmas().len(), 51);
        assert_eq!(s.timesteps().len(), 50);
        // sigma_0 = shift(0.999) = 8*0.999 / (1 + 7*0.999)
        let s0 = 8.0 * 0.999 / (1.0 + 7.0 * 0.999);
        assert!((f64::from(s.sigmas()[0]) - s0).abs() < 1e-7);
        assert_eq!(s.timesteps()[0], (s0 * 1000.0) as i64);
        assert_eq!(s.timesteps()[0], 999);
        assert_eq!(*s.sigmas().last().unwrap(), 0.0);
        // Last real sigma: 1 - alpha_1 with alpha_1 = 1 + (0.001 - 1) / 50.
        let a1 = 1.0 + (0.001 - 1.0) / 50.0;
        let last = 8.0 * (1.0 - a1) / (1.0 + 7.0 * (1.0 - a1));
        assert!((f64::from(s.sigmas()[49]) - last).abs() < 1e-7);
        assert_eq!(s.timesteps()[49], (last * 1000.0) as i64);
        // Strictly decreasing.
        assert!(s.sigmas().windows(2).all(|w| w[0] > w[1]));
    }

    #[test]
    fn orders_follow_lower_order_final() {
        let mut s = SanaDpmSolver::new(50, SANA_FLOW_SHIFT).unwrap();
        let orders: Vec<usize> = (0..50).map(|_| s.plan_step().unwrap().order).collect();
        assert_eq!(orders[0], 1);
        assert!(orders[1..49].iter().all(|&o| o == 2));
        assert_eq!(orders[49], 1);
        assert!(s.plan_step().is_err());
    }

    #[test]
    fn final_step_returns_the_data_prediction() {
        let mut s = SanaDpmSolver::new(3, SANA_FLOW_SHIFT).unwrap();
        s.plan_step().unwrap();
        s.plan_step().unwrap();
        let last = s.plan_step().unwrap();
        assert_eq!(last.sample_coef, 0.0);
        assert!((last.x0_coef - 1.0).abs() < 1e-7);
        assert_eq!(last.prev_x0_coef, 0.0);
    }

    /// A constant velocity field `v = noise - x_data` is integrated exactly
    /// by the flow ODE; the solver must land on the data point.
    #[test]
    fn integrates_a_straight_flow_to_the_data() {
        let data = [0.25f32, -1.5, 3.0];
        let noise = [1.0f32, 0.5, -0.75];
        let v: Vec<f32> = noise.iter().zip(&data).map(|(n, d)| n - d).collect();
        let mut s = SanaDpmSolver::new(20, SANA_FLOW_SHIFT).unwrap();
        let sigma0 = s.sigmas()[0];
        let mut x: Vec<f32> = noise
            .iter()
            .zip(&data)
            .map(|(n, d)| sigma0 * n + (1.0 - sigma0) * d)
            .collect();
        let mut prev: Option<Vec<f32>> = None;
        for _ in 0..20 {
            let (next, x0) = s.step_host(&v, &x, prev.as_deref()).unwrap();
            x = next;
            prev = Some(x0);
        }
        for (a, b) in x.iter().zip(&data) {
            assert!((a - b).abs() < 1e-4, "{a} vs {b}");
        }
    }
}
