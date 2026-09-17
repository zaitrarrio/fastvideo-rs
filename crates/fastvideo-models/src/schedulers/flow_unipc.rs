//! Flow-matching UniPC (predictor-corrector).
//!
//! Ported from FastVideo `scheduling_flow_unipc_multistep.py` (Diffusers
//! UniPC converted for flow matching, Apache-2.0). Default Wan settings:
//! `solver_order=2`, `solver_type=bh2`, `predict_x0=true`, `final_sigmas=zero`.
//!
//! All scalar math lives in [`FlowUniPCMultistepScheduler::plan_step`], which
//! expresses one step as linear combinations of sample-shaped tensors
//! ([`UniPcStepPlan`]). [`FlowUniPCMultistepScheduler::step`] is the host
//! reference that applies those plans to `&[f32]` buffers; device backends
//! apply the same plans with their own axpy kernels.

use super::flow_match::apply_shift;

const EPS: f64 = 1e-12;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UniPcSolverType {
    Bh1,
    Bh2,
}

/// A sample-shaped tensor a [`UniPcStepPlan`] combination refers to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UniPcTerm {
    /// This step's sample. In the predictor: the corrected sample (or the raw
    /// input sample when the plan has no corrector).
    Sample,
    /// The sample the previous step's predictor consumed (its corrected sample).
    LastSample,
    /// This step's converted model output (`sample - convert_scale * model_output`).
    Converted,
    /// Converted outputs of earlier steps, indexed against the history *before*
    /// this step's `Converted` is pushed: `History(0)` = previous step,
    /// `History(1)` = two steps back, ...
    History(usize),
}

/// One UniPC step as scalar coefficients over [`UniPcTerm`] tensors.
///
/// Caller protocol:
/// 1. `converted = sample - convert_scale * model_output`
/// 2. `corrected = Σ coef * term` over `corrector` if present, else `sample`
/// 3. `prev_sample = Σ coef * term` over `predictor` (`Sample` = `corrected`)
/// 4. push `converted` to the front of the history, keep `history_len` entries;
///    remember `corrected` as the next step's `LastSample`.
#[derive(Debug, Clone, PartialEq)]
pub struct UniPcStepPlan {
    /// Current sigma (`sigmas[step_index]`).
    pub sigma: f64,
    /// `converted = sample - convert_scale * model_output`. Equals `sigma` for
    /// `predict_x0` (x0 prediction) and `1 - sigma` otherwise (epsilon).
    pub convert_scale: f64,
    /// `corrected_sample = Σ coef * term`, terms from `{LastSample, Converted, History(k)}`.
    pub corrector: Option<Vec<(UniPcTerm, f64)>>,
    /// `prev_sample = Σ coef * term`, terms from `{Sample, Converted, History(k)}`.
    pub predictor: Vec<(UniPcTerm, f64)>,
    /// Converted outputs to retain after pushing this step's `Converted`
    /// (== `solver_order`).
    pub history_len: usize,
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
    /// Host `step` history of converted outputs, newest first.
    history: Vec<Vec<f32>>,
    /// Host `step` sample consumed by the previous predictor.
    last_sample: Option<Vec<f32>>,
    /// Scalar mirror of upstream `last_sample is not None`.
    has_last_sample: bool,
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
            history: Vec::new(),
            last_sample: None,
            has_last_sample: false,
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
                let t =
                    self.sigma_max + (self.sigma_min - self.sigma_max) * (i as f64) / (n as f64);
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
        self.history.clear();
        self.last_sample = None;
        self.has_last_sample = false;
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

    fn lambda(sigma: f64) -> f64 {
        let (alpha, sigma) = Self::sigma_to_alpha_sigma(sigma);
        alpha.max(EPS).ln() - sigma.max(EPS).ln()
    }

    fn bh(&self, hh: f64) -> f64 {
        match self.solver_type {
            UniPcSolverType::Bh1 => hh,
            UniPcSolverType::Bh2 => hh.exp_m1(),
        }
    }

    /// Shared UniP/UniC scalars. Returns `(h_phi_1, B_h, R rows, b)`.
    fn bh_system(&self, h: f64, rks: &[f64], order: usize) -> (f64, f64, Vec<Vec<f64>>, Vec<f64>) {
        let hh = if self.predict_x0 { -h } else { h };
        let h_phi_1 = hh.exp_m1();
        let mut h_phi_k = h_phi_1 / hh - 1.0;
        let mut factorial_i = 1.0;
        let b_h = self.bh(hh);
        let mut b = Vec::with_capacity(order);
        let mut r_rows = Vec::with_capacity(order);
        for i in 1..=order {
            r_rows.push(
                rks.iter()
                    .map(|rk| rk.powi((i - 1) as i32))
                    .collect::<Vec<_>>(),
            );
            b.push(h_phi_k * factorial_i / b_h);
            factorial_i *= (i + 1) as f64;
            h_phi_k = h_phi_k / hh - 1.0 / factorial_i;
        }
        (h_phi_1, b_h, r_rows, b)
    }

    /// `(x coef, m0 coef, residual scale)` of `x_t_ = cx*x - cm*m0; x_t = x_t_ - scale*res`.
    fn base_coeffs(&self, sigma_t: f64, sigma_s0: f64, h_phi_1: f64, b_h: f64) -> (f64, f64, f64) {
        let (alpha_t, sigma_t) = Self::sigma_to_alpha_sigma(sigma_t);
        let (alpha_s0, sigma_s0) = Self::sigma_to_alpha_sigma(sigma_s0);
        if self.predict_x0 {
            (sigma_t / sigma_s0, alpha_t * h_phi_1, alpha_t * b_h)
        } else {
            (alpha_t / alpha_s0, sigma_t * h_phi_1, sigma_t * b_h)
        }
    }

    /// `multistep_uni_p_bh_update` as coefficients. `m0` is `Converted`,
    /// `m_i` (i >= 1) is `History(i - 1)`.
    fn predictor_terms(&self, idx: usize, order: usize) -> Result<Vec<(UniPcTerm, f64)>, String> {
        let sigma_t = self.sigmas[idx + 1];
        let sigma_s0 = self.sigmas[idx];
        let lambda_s0 = Self::lambda(sigma_s0);
        let h = Self::lambda(sigma_t) - lambda_s0;
        let mut rks = Vec::with_capacity(order);
        for i in 1..order {
            let si = idx.checked_sub(i).ok_or("predictor history underflow")?;
            rks.push((Self::lambda(self.sigmas[si]) - lambda_s0) / h);
        }
        let n_hist = rks.len();
        rks.push(1.0);
        let (h_phi_1, b_h, r_rows, b) = self.bh_system(h, &rks, order);
        let rhos_p = if n_hist == 0 {
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
        let (cx, cm, scale) = self.base_coeffs(sigma_t, sigma_s0, h_phi_1, b_h);
        // x_t = cx*x - cm*m0 - scale * Σ_k rho_k (m_{k+1} - m0) / rk_k
        let mut m0 = -cm;
        let mut hist = Vec::with_capacity(n_hist);
        for k in 0..n_hist {
            let w = rhos_p[k] / rks[k];
            m0 += scale * w;
            hist.push((UniPcTerm::History(k), -scale * w));
        }
        let mut terms = vec![(UniPcTerm::Sample, cx), (UniPcTerm::Converted, m0)];
        terms.extend(hist);
        Ok(terms)
    }

    /// `multistep_uni_c_bh_update` as coefficients. `m0` is `History(0)`,
    /// `m_i` (i >= 1) is `History(i)`, `model_t` is `Converted`.
    fn corrector_terms(&self, idx: usize, order: usize) -> Result<Vec<(UniPcTerm, f64)>, String> {
        let sigma_t = self.sigmas[idx];
        let sigma_s0 = self.sigmas[idx - 1];
        let lambda_s0 = Self::lambda(sigma_s0);
        let h = Self::lambda(sigma_t) - lambda_s0;
        let mut rks = Vec::with_capacity(order);
        for i in 1..order {
            let si = idx
                .checked_sub(i + 1)
                .ok_or("corrector history underflow")?;
            rks.push((Self::lambda(self.sigmas[si]) - lambda_s0) / h);
        }
        let n_hist = rks.len();
        rks.push(1.0);
        let (h_phi_1, b_h, r_rows, b) = self.bh_system(h, &rks, order);
        let rhos_c = if order == 1 {
            vec![0.5]
        } else {
            solve_linear(&r_rows, &b)?
        };
        let (cx, cm, scale) = self.base_coeffs(sigma_t, sigma_s0, h_phi_1, b_h);
        // x_t = cx*x - cm*m0 - scale * (Σ_k rho_k (m_{k+1} - m0) / rk_k + rho_last (model_t - m0))
        let rho_last = *rhos_c.last().ok_or("empty UniC rhos")?;
        let mut m0 = -cm + scale * rho_last;
        let mut hist = Vec::with_capacity(n_hist);
        for k in 0..n_hist {
            let w = rhos_c[k] / rks[k];
            m0 += scale * w;
            hist.push((UniPcTerm::History(k + 1), -scale * w));
        }
        let mut terms = vec![
            (UniPcTerm::LastSample, cx),
            (UniPcTerm::Converted, -scale * rho_last),
            (UniPcTerm::History(0), m0),
        ];
        terms.extend(hist);
        Ok(terms)
    }

    /// Plan one UniPC predictor-corrector step and advance the scalar state
    /// (`step_index`, `this_order`, `lower_order_nums`, last-sample flag)
    /// exactly as [`Self::step`]. See [`UniPcStepPlan`] for how to apply it.
    pub fn plan_step(&mut self) -> Result<UniPcStepPlan, String> {
        if self.sigmas.len() < 2 {
            return Err("call set_timesteps before step".into());
        }
        if self.solver_order == 0 {
            return Err("solver_order must be >= 1".into());
        }
        let idx = self.step_index.unwrap_or(0);
        if idx + 1 >= self.sigmas.len() {
            return Err("step past end of schedule".into());
        }
        // Converted outputs available before this step's push.
        let available = self.lower_order_nums.min(self.solver_order);
        let use_corrector = idx > 0 && self.has_last_sample;
        let corrector = if use_corrector {
            if self.this_order > available {
                return Err("missing UniC history".into());
            }
            Some(self.corrector_terms(idx, self.this_order)?)
        } else {
            None
        };
        let this_order = if self.lower_order_final {
            self.solver_order.min(self.timesteps.len() - idx)
        } else {
            self.solver_order
        };
        let this_order = this_order.min(self.lower_order_nums + 1).max(1);
        if this_order - 1 > available {
            return Err("missing UniP history".into());
        }
        let predictor = self.predictor_terms(idx, this_order)?;
        let sigma = self.sigmas[idx];

        self.this_order = this_order;
        self.has_last_sample = true;
        if self.lower_order_nums < self.solver_order {
            self.lower_order_nums += 1;
        }
        self.step_index = Some(idx + 1);
        Ok(UniPcStepPlan {
            sigma,
            convert_scale: if self.predict_x0 { sigma } else { 1.0 - sigma },
            corrector,
            predictor,
            history_len: self.solver_order,
        })
    }

    /// One UniPC predictor-corrector step. `model_output` is flow velocity.
    pub fn step(&mut self, model_output: &[f32], sample: &[f32]) -> Result<Vec<f32>, String> {
        if model_output.len() != sample.len() {
            return Err("sample / model_output length mismatch".into());
        }
        let plan = self.plan_step()?;
        let converted: Vec<f32> = sample
            .iter()
            .zip(model_output)
            .map(|(&x, &m)| (f64::from(x) - plan.convert_scale * f64::from(m)) as f32)
            .collect();
        let corrected = match &plan.corrector {
            Some(terms) => combine(terms, sample.len(), |term| match term {
                UniPcTerm::LastSample => self.last_sample.as_deref().ok_or("missing last_sample"),
                UniPcTerm::Converted => Ok(&converted),
                UniPcTerm::History(k) => self
                    .history
                    .get(k)
                    .map(Vec::as_slice)
                    .ok_or("missing UniC history"),
                UniPcTerm::Sample => Err("Sample term in corrector"),
            })?,
            None => sample.to_vec(),
        };
        let prev = combine(&plan.predictor, sample.len(), |term| match term {
            UniPcTerm::Sample => Ok(&corrected),
            UniPcTerm::Converted => Ok(&converted),
            UniPcTerm::History(k) => self
                .history
                .get(k)
                .map(Vec::as_slice)
                .ok_or("missing UniP history"),
            UniPcTerm::LastSample => Err("LastSample term in predictor"),
        })?;
        self.history.insert(0, converted);
        self.history.truncate(plan.history_len);
        self.last_sample = Some(corrected);
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

/// `Σ coef * term`, accumulated per element in f64.
fn combine<'a>(
    terms: &[(UniPcTerm, f64)],
    n: usize,
    resolve: impl Fn(UniPcTerm) -> Result<&'a [f32], &'static str>,
) -> Result<Vec<f32>, String> {
    let mut acc = vec![0.0f64; n];
    for &(term, coef) in terms {
        let src = resolve(term)?;
        if src.len() != n {
            return Err("UniPC term length mismatch".into());
        }
        for (a, &v) in acc.iter_mut().zip(src) {
            *a += coef * f64::from(v);
        }
    }
    Ok(acc.into_iter().map(|v| v as f32).collect())
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
    fn plan_corrector_starts_on_step_one() {
        let mut sched = FlowUniPCMultistepScheduler::new(1000, 3.0);
        sched.set_timesteps(6);
        let p0 = sched.plan_step().unwrap();
        assert!(p0.corrector.is_none());
        assert_eq!(p0.history_len, 2);
        assert_eq!(p0.predictor.len(), 2); // order 1: Sample + Converted
        let p1 = sched.plan_step().unwrap();
        let corr = p1.corrector.expect("corrector on step 1");
        assert!(corr.iter().any(|(t, _)| *t == UniPcTerm::LastSample));
        assert!(corr.iter().any(|(t, _)| *t == UniPcTerm::History(0)));
        assert!(p1
            .predictor
            .iter()
            .any(|(t, _)| *t == UniPcTerm::History(0)));
        sched.set_timesteps(6);
        assert!(sched.plan_step().unwrap().corrector.is_none());
    }

    /// Device-style application of plans: f32 only, `out = out + c * x` chains.
    fn device_run(
        sched: &mut FlowUniPCMultistepScheduler,
        mut x: Vec<f32>,
        steps: usize,
        velocity: impl Fn(&[f32], usize) -> Vec<f32>,
    ) -> Vec<f32> {
        fn axpy_chain(
            terms: &[(UniPcTerm, f64)],
            sample: &[f32],
            last: Option<&[f32]>,
            converted: &[f32],
            history: &[Vec<f32>],
        ) -> Vec<f32> {
            let mut out = vec![0.0f32; sample.len()];
            for &(term, coef) in terms {
                let src: &[f32] = match term {
                    UniPcTerm::Sample => sample,
                    UniPcTerm::LastSample => last.unwrap(),
                    UniPcTerm::Converted => converted,
                    UniPcTerm::History(k) => &history[k],
                };
                let c = coef as f32;
                for (o, s) in out.iter_mut().zip(src) {
                    *o = 1.0f32 * *o + c * *s;
                }
            }
            out
        }
        let mut history: Vec<Vec<f32>> = Vec::new();
        let mut last: Option<Vec<f32>> = None;
        for i in 0..steps {
            let v = velocity(&x, i);
            let plan = sched.plan_step().unwrap();
            let neg = -(plan.convert_scale as f32);
            let converted: Vec<f32> = x.iter().zip(&v).map(|(a, b)| a + neg * b).collect();
            let corrected = match &plan.corrector {
                Some(t) => axpy_chain(t, &x, last.as_deref(), &converted, &history),
                None => x.clone(),
            };
            x = axpy_chain(&plan.predictor, &corrected, None, &converted, &history);
            history.insert(0, converted);
            history.truncate(plan.history_len);
            last = Some(corrected);
        }
        x
    }

    #[test]
    fn device_plans_match_host_step_nonconstant_flow() {
        let n = 64;
        let mut seed = 0x2545_f491_4f6c_dd1du64;
        let x0: Vec<f32> = (0..n)
            .map(|_| {
                seed = seed
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((seed >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
            })
            .collect();
        let velocity = |x: &[f32], step: usize| -> Vec<f32> {
            x.iter()
                .enumerate()
                .map(|(j, &v)| {
                    let phase = 0.37 * j as f32 + 0.61 * step as f32;
                    (1.3 * v + phase).sin() - 0.4 * v + 0.2 * (0.5 * phase).cos()
                })
                .collect()
        };
        let steps = 12;
        for order in 1..=3 {
            for solver_type in [UniPcSolverType::Bh1, UniPcSolverType::Bh2] {
                let make = || {
                    let mut s = FlowUniPCMultistepScheduler::new(1000, 3.0);
                    s.solver_order = order;
                    s.solver_type = solver_type;
                    s.set_timesteps(steps);
                    s
                };
                let mut host = make();
                let mut x = x0.clone();
                for i in 0..steps {
                    let v = velocity(&x, i);
                    x = host.step(&v, &x).unwrap();
                }
                let dev = device_run(&mut make(), x0.clone(), steps, velocity);
                let max_abs = x
                    .iter()
                    .zip(&dev)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0f32, f32::max);
                let diff2: f64 = x
                    .iter()
                    .zip(&dev)
                    .map(|(a, b)| f64::from(a - b).powi(2))
                    .sum();
                let norm2: f64 = x.iter().map(|a| f64::from(*a).powi(2)).sum();
                let rel = (diff2 / norm2).sqrt();
                assert!(
                    max_abs < 1e-4 && rel < 1e-5,
                    "order={order} {solver_type:?}: max_abs={max_abs} rel_l2={rel}"
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
