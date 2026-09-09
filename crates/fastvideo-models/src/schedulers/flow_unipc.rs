//! Flow-matching UniPC (predictor-corrector).
//!
//! Ported from FastVideo `scheduling_flow_unipc_multistep.py` (Diffusers
//! UniPC converted for flow matching, Apache-2.0). Default Wan settings:
//! `solver_order=2`, `solver_type=bh2`, `predict_x0=true`, `final_sigmas=zero`.

use super::flow_match::apply_shift;

const EPS: f64 = 1e-12;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UniPcSolverType {
    Bh1,
    Bh2,
}

#[derive(Debug, Clone)]
pub struct FlowUniPCMultistepScheduler {
    pub num_train_timesteps: i32,
    pub shift: f64,
    pub solver_order: usize,
    pub predict_x0: bool,
    pub solver_type: UniPcSolverType,
    pub lower_order_final: bool,
    sigma_min: f64,
    sigma_max: f64,
    sigmas: Vec<f64>,
    timesteps: Vec<f64>,
    timesteps_i64: Vec<i64>,
    model_outputs: Vec<Option<Vec<f32>>>,
    last_sample: Option<Vec<f32>>,
    step_index: Option<usize>,
    lower_order_nums: usize,
    this_order: usize,
}

impl FlowUniPCMultistepScheduler {
    pub fn new(num_train_timesteps: i32, shift: f64) -> Self {
        let n = num_train_timesteps as usize;
        let mut alphas = vec![0f32; n];
        // np.linspace(1, 1/N, N, dtype=float32)[::-1]
        if n == 1 {
            alphas[0] = 1.0;
        } else {
            let start = 1.0f32;
            let end = 1.0 / n as f32;
            for i in 0..n {
                alphas[n - 1 - i] = start + (end - start) * (i as f32) / ((n - 1) as f32);
            }
        }
        let mut sigma_max = 0.0f64;
        let mut sigma_min = 0.0f64;
        for (i, alpha) in alphas.iter().enumerate() {
            let sigma = apply_shift(f64::from(1.0 - *alpha), shift);
            if i == 0 {
                sigma_max = sigma;
            }
            if i + 1 == n {
                sigma_min = sigma;
            }
        }
        Self {
            num_train_timesteps,
            shift,
            solver_order: 2,
            predict_x0: true,
            solver_type: UniPcSolverType::Bh2,
            lower_order_final: true,
            sigma_min,
            sigma_max,
            sigmas: Vec::new(),
            timesteps: Vec::new(),
            timesteps_i64: Vec::new(),
            model_outputs: vec![None; 2],
            last_sample: None,
            step_index: None,
            lower_order_nums: 0,
            this_order: 0,
        }
    }

    pub fn sigma_max(&self) -> f64 {
        self.sigma_max
    }

    pub fn sigma_min(&self) -> f64 {
        self.sigma_min
    }

    pub fn set_timesteps(&mut self, num_inference_steps: usize) {
        let n = num_inference_steps.max(1);
        let mut sigmas = Vec::with_capacity(n + 1);
        if n == 1 {
            sigmas.push(self.sigma_max);
        } else {
            for i in 0..n {
                // np.linspace(sigma_max, sigma_min, n+1)[:-1]
                let t = self.sigma_max
                    + (self.sigma_min - self.sigma_max) * (i as f64) / (n as f64);
                sigmas.push(t);
            }
        }
        for s in &mut sigmas {
            *s = apply_shift(*s, self.shift);
        }
        let timesteps: Vec<f64> = sigmas
            .iter()
            .map(|s| s * f64::from(self.num_train_timesteps))
            .collect();
        let timesteps_i64: Vec<i64> = timesteps.iter().map(|t| *t as i64).collect();
        sigmas.push(0.0);
        self.timesteps = timesteps;
        self.timesteps_i64 = timesteps_i64;
        self.sigmas = sigmas;
        self.model_outputs = vec![None; self.solver_order];
        self.last_sample = None;
        self.step_index = None;
        self.lower_order_nums = 0;
        self.this_order = 0;
    }

    pub fn inference_timesteps(&self) -> &[f64] {
        &self.timesteps
    }

    pub fn inference_timesteps_i64(&self) -> &[i64] {
        &self.timesteps_i64
    }

    pub fn inference_sigmas(&self) -> &[f64] {
        &self.sigmas
    }

    fn sigma_to_alpha_sigma(sigma: f64) -> (f64, f64) {
        (1.0 - sigma, sigma)
    }

    fn convert_model_output(&self, model_output: &[f32], sample: &[f32]) -> Vec<f32> {
        let idx = self.step_index.expect("step_index");
        let sigma_t = self.sigmas[idx] as f32;
        sample
            .iter()
            .zip(model_output)
            .map(|(x, m)| x - sigma_t * m)
            .collect()
    }

    fn bh(&self, hh: f64) -> f64 {
        match self.solver_type {
            UniPcSolverType::Bh1 => hh,
            UniPcSolverType::Bh2 => hh.exp_m1(),
        }
    }

    fn predictor_update(&self, sample: &[f32], order: usize) -> Result<Vec<f32>, String> {
        let idx = self.step_index.expect("step_index");
        let m0 = self.model_outputs.last().and_then(|m| m.as_ref()).ok_or("missing m0")?;
        let (alpha_t, sigma_t) = Self::sigma_to_alpha_sigma(self.sigmas[idx + 1]);
        let (alpha_s0, sigma_s0) = Self::sigma_to_alpha_sigma(self.sigmas[idx]);
        let lambda_t = (alpha_t.max(EPS)).ln() - (sigma_t.max(EPS)).ln();
        let lambda_s0 = (alpha_s0.max(EPS)).ln() - (sigma_s0.max(EPS)).ln();
        let h = lambda_t - lambda_s0;
        let mut rks = Vec::new();
        let mut d1s: Vec<Vec<f32>> = Vec::new();
        for i in 1..order {
            let si = idx.checked_sub(i).ok_or("predictor history underflow")?;
            let mi = self
                .model_outputs
                .get(self.model_outputs.len().wrapping_sub(i + 1))
                .and_then(|m| m.as_ref())
                .ok_or("missing mi")?;
            let (alpha_si, sigma_si) = Self::sigma_to_alpha_sigma(self.sigmas[si]);
            let lambda_si = (alpha_si.max(EPS)).ln() - (sigma_si.max(EPS)).ln();
            let rk = (lambda_si - lambda_s0) / h;
            rks.push(rk);
            d1s.push(
                mi.iter()
                    .zip(m0)
                    .map(|(a, b)| (a - b) / rk as f32)
                    .collect(),
            );
        }
        rks.push(1.0);
        let hh = if self.predict_x0 { -h } else { h };
        let h_phi_1 = hh.exp_m1();
        let mut h_phi_k = h_phi_1 / hh - 1.0;
        let mut factorial_i = 1.0;
        let b_h = self.bh(hh);
        let mut b = Vec::with_capacity(order);
        let mut r_rows = Vec::with_capacity(order);
        for i in 1..=order {
            r_rows.push(rks.iter().map(|rk| rk.powi((i - 1) as i32)).collect::<Vec<_>>());
            b.push(h_phi_k * factorial_i / b_h);
            factorial_i *= (i + 1) as f64;
            h_phi_k = h_phi_k / hh - 1.0 / factorial_i;
        }
        let rhos_p = if d1s.is_empty() {
            Vec::new()
        } else if order == 2 {
            vec![0.5]
        } else {
            let r_cut: Vec<Vec<f64>> = r_rows[..order - 1]
                .iter()
                .map(|row| row[..order - 1].to_vec())
                .collect();
            solve_linear(&r_cut, &b[..order - 1])?
        };
        let mut x_t: Vec<f32> = sample
            .iter()
            .zip(m0)
            .map(|(x, m)| {
                let base = if self.predict_x0 {
                    (sigma_t / sigma_s0) * f64::from(*x) - alpha_t * h_phi_1 * f64::from(*m)
                } else {
                    (alpha_t / alpha_s0) * f64::from(*x) - sigma_t * h_phi_1 * f64::from(*m)
                };
                base as f32
            })
            .collect();
        if !d1s.is_empty() {
            let pred_res = mix_history(&d1s, &rhos_p);
            let scale = if self.predict_x0 {
                alpha_t * b_h
            } else {
                sigma_t * b_h
            } as f32;
            for (x, r) in x_t.iter_mut().zip(pred_res) {
                *x -= scale * r;
            }
        }
        Ok(x_t)
    }

    fn corrector_update(
        &self,
        this_model_output: &[f32],
        last_sample: &[f32],
        this_sample: &[f32],
        order: usize,
    ) -> Result<Vec<f32>, String> {
        let idx = self.step_index.expect("step_index");
        let m0 = self.model_outputs.last().and_then(|m| m.as_ref()).ok_or("missing m0")?;
        let (alpha_t, sigma_t) = Self::sigma_to_alpha_sigma(self.sigmas[idx]);
        let (alpha_s0, sigma_s0) = Self::sigma_to_alpha_sigma(self.sigmas[idx - 1]);
        let lambda_t = (alpha_t.max(EPS)).ln() - (sigma_t.max(EPS)).ln();
        let lambda_s0 = (alpha_s0.max(EPS)).ln() - (sigma_s0.max(EPS)).ln();
        let h = lambda_t - lambda_s0;
        let mut rks = Vec::new();
        let mut d1s: Vec<Vec<f32>> = Vec::new();
        for i in 1..order {
            let si = idx
                .checked_sub(i + 1)
                .ok_or("corrector history underflow")?;
            let mi = self
                .model_outputs
                .get(self.model_outputs.len().wrapping_sub(i + 1))
                .and_then(|m| m.as_ref())
                .ok_or("missing mi")?;
            let (alpha_si, sigma_si) = Self::sigma_to_alpha_sigma(self.sigmas[si]);
            let lambda_si = (alpha_si.max(EPS)).ln() - (sigma_si.max(EPS)).ln();
            let rk = (lambda_si - lambda_s0) / h;
            rks.push(rk);
            d1s.push(
                mi.iter()
                    .zip(m0)
                    .map(|(a, b)| (a - b) / rk as f32)
                    .collect(),
            );
        }
        rks.push(1.0);
        let hh = if self.predict_x0 { -h } else { h };
        let h_phi_1 = hh.exp_m1();
        let mut h_phi_k = h_phi_1 / hh - 1.0;
        let mut factorial_i = 1.0;
        let b_h = self.bh(hh);
        let mut b = Vec::with_capacity(order);
        let mut r_rows = Vec::with_capacity(order);
        for i in 1..=order {
            r_rows.push(rks.iter().map(|rk| rk.powi((i - 1) as i32)).collect::<Vec<_>>());
            b.push(h_phi_k * factorial_i / b_h);
            factorial_i *= (i + 1) as f64;
            h_phi_k = h_phi_k / hh - 1.0 / factorial_i;
        }
        let rhos_c = if order == 1 {
            vec![0.5]
        } else {
            solve_linear(&r_rows, &b)?
        };
        let mut x_t: Vec<f32> = last_sample
            .iter()
            .zip(m0)
            .map(|(x, m)| {
                let base = if self.predict_x0 {
                    (sigma_t / sigma_s0) * f64::from(*x) - alpha_t * h_phi_1 * f64::from(*m)
                } else {
                    (alpha_t / alpha_s0) * f64::from(*x) - sigma_t * h_phi_1 * f64::from(*m)
                };
                base as f32
            })
            .collect();
        let corr_res = if d1s.is_empty() {
            vec![0.0f32; this_sample.len()]
        } else {
            mix_history(&d1s, &rhos_c[..rhos_c.len() - 1])
        };
        let rho_last = *rhos_c.last().unwrap_or(&0.5) as f32;
        let scale = if self.predict_x0 {
            alpha_t * b_h
        } else {
            sigma_t * b_h
        } as f32;
        for i in 0..x_t.len() {
            let d1_t = this_model_output[i] - m0[i];
            x_t[i] -= scale * (corr_res[i] + rho_last * d1_t);
        }
        Ok(x_t)
    }

    /// One UniPC predictor-corrector step. `model_output` is flow velocity.
    pub fn step(&mut self, model_output: &[f32], sample: &[f32]) -> Result<Vec<f32>, String> {
        if model_output.len() != sample.len() {
            return Err("sample / model_output length mismatch".into());
        }
        if self.sigmas.len() < 2 {
            return Err("call set_timesteps before step".into());
        }
        if self.step_index.is_none() {
            self.step_index = Some(0);
        }
        let idx = self.step_index.unwrap();
        if idx + 1 >= self.sigmas.len() {
            return Err("step past end of schedule".into());
        }
        let use_corrector = idx > 0 && self.last_sample.is_some();
        let converted = self.convert_model_output(model_output, sample);
        let mut sample = sample.to_vec();
        if use_corrector {
            sample = self.corrector_update(
                &converted,
                self.last_sample.as_ref().unwrap(),
                &sample,
                self.this_order,
            )?;
        }
        for i in 0..self.solver_order.saturating_sub(1) {
            self.model_outputs[i] = self.model_outputs[i + 1].clone();
        }
        if let Some(last) = self.model_outputs.last_mut() {
            *last = Some(converted);
        }
        let this_order = if self.lower_order_final {
            self.solver_order.min(self.timesteps.len() - idx)
        } else {
            self.solver_order
        };
        self.this_order = this_order.min(self.lower_order_nums + 1).max(1);
        self.last_sample = Some(sample.clone());
        let prev = self.predictor_update(&sample, self.this_order)?;
        if self.lower_order_nums < self.solver_order {
            self.lower_order_nums += 1;
        }
        self.step_index = Some(idx + 1);
        Ok(prev)
    }

    /// Run the full UniPC loop with a velocity callback `f(sample, sigma_t, t)`.
    pub fn denoise(
        &mut self,
        mut sample: Vec<f32>,
        mut velocity: impl FnMut(&[f32], f64, f64) -> Vec<f32>,
    ) -> Result<Vec<f32>, String> {
        let ts: Vec<f64> = self.timesteps.clone();
        let sigmas: Vec<f64> = self.sigmas.clone();
        for (i, &t) in ts.iter().enumerate() {
            let v = velocity(&sample, sigmas[i], t);
            sample = self.step(&v, &sample)?;
        }
        Ok(sample)
    }
}

fn mix_history(d1s: &[Vec<f32>], rhos: &[f64]) -> Vec<f32> {
    let n = d1s.first().map(Vec::len).unwrap_or(0);
    let mut out = vec![0.0f32; n];
    for (k, d1) in d1s.iter().enumerate() {
        let w = rhos.get(k).copied().unwrap_or(0.0) as f32;
        for (o, v) in out.iter_mut().zip(d1) {
            *o += w * *v;
        }
    }
    out
}

fn solve_linear(a: &[Vec<f64>], b: &[f64]) -> Result<Vec<f64>, String> {
    let n = b.len();
    if a.len() != n || a.iter().any(|row| row.len() != n) {
        return Err("singular or ragged linear system".into());
    }
    let mut m = a.to_vec();
    let mut x = b.to_vec();
    for k in 0..n {
        let mut pivot = k;
        for i in k + 1..n {
            if m[i][k].abs() > m[pivot][k].abs() {
                pivot = i;
            }
        }
        if m[pivot][k].abs() < 1e-18 {
            return Err("singular UniPC linear system".into());
        }
        m.swap(k, pivot);
        x.swap(k, pivot);
        let diag = m[k][k];
        for j in k..n {
            m[k][j] /= diag;
        }
        x[k] /= diag;
        for i in 0..n {
            if i == k {
                continue;
            }
            let f = m[i][k];
            for j in k..n {
                m[i][j] -= f * m[k][j];
            }
            x[i] -= f * x[k];
        }
    }
    Ok(x)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wan_1_3b_50_step_matches_fastvideo_float32_schedule() {
        let mut sched = FlowUniPCMultistepScheduler::new(1000, 3.0);
        sched.set_timesteps(50);
        assert_eq!(sched.timesteps.len(), 50);
        assert_eq!(sched.sigmas.len(), 51);
        assert!((sched.sigmas[50] - 0.0).abs() < 1e-12);
        let s0 = sched.sigmas[0];
        let s1 = sched.sigmas[1];
        let s49 = sched.sigmas[49];
        assert!(
            (s0 - 0.9998887777328491).abs() < 1e-7,
            "sigma[0]={s0} sigma_max={} sigma_min={}",
            sched.sigma_max(),
            sched.sigma_min()
        );
        assert!((s1 - 0.9931312799453735).abs() < 1e-6, "sigma[1]={s1}");
        assert!((s49 - 0.057673800736665726).abs() < 1e-6, "sigma[49]={s49}");
        assert_eq!(sched.timesteps_i64[0], 999);
        assert_eq!(sched.timesteps_i64.len(), 50);
    }

    #[test]
    fn predictor_corrector_constant_flow_golden() {
        let mut sched = FlowUniPCMultistepScheduler::new(1000, 3.0);
        sched.set_timesteps(4);
        let vel = vec![0.1f32, -0.2, 0.05, 0.3, -0.15];
        let mut x = vec![0.2f32, -0.4, 0.8, 1.5, -1.1];
        let mut traj = Vec::new();
        for _ in 0..4 {
            x = sched.step(&vel, &x).unwrap();
            traj.push(x.clone());
        }
        let expected = [
            [0.18999911, -0.37999822, 0.79499956, 1.4699973, -1.0849987],
            [0.17499861, -0.34999722, 0.7874993, 1.4249958, -1.0624979],
            [0.15000000, -0.30000000, 0.7750000, 1.3500000, -1.0250000],
            [0.10001112, -0.20002225, 0.75000554, 1.2000334, -0.9500167],
        ];
        for (got, exp) in traj.iter().zip(expected) {
            for (g, e) in got.iter().zip(exp) {
                assert!(
                    (g - e).abs() < 2e-5,
                    "uniPC traj mismatch {g} vs {e} (full {traj:?})"
                );
            }
        }
    }

    #[test]
    fn denoise_flow_to_zero_shrinks_sample() {
        let mut sched = FlowUniPCMultistepScheduler::new(1000, 3.0);
        sched.set_timesteps(8);
        let x0 = vec![1.0f32, -2.0, 0.5];
        let out = sched
            .denoise(x0.clone(), |sample, sigma, _t| {
                sample.iter().map(|v| v / sigma.max(1e-6) as f32).collect()
            })
            .unwrap();
        let start: f32 = x0.iter().map(|v| v.abs()).sum();
        let end: f32 = out.iter().map(|v| v.abs()).sum();
        assert!(end < start * 0.25, "expected flow-to-zero, got {out:?}");
    }
}
