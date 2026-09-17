//! Distribution Matching Distillation (DMD) discrete timesteps.
//!
//! FastVideo `FastWan2_1_T2V_480P_Config` uses
//! `dmd_denoising_steps = [1000, 757, 522]`. The DMD stage owns a
//! `FlowMatchEulerDiscreteScheduler(shift=DMD_TRAINING_NOISE_SHIFT)` without
//! `set_timesteps`, i.e. the complete float32 training-noise table, and maps
//! every DMD timestep onto its nearest table entry. This is independent of the
//! configurable inference `flow_shift` used by UniPC.
//!
//! Per step (`stages/dmd.py`):
//! `x0 = x - sigma_t * v` (`pred_noise_to_pred_video`, float64), then unless
//! it is the last step `x = (1 - sigma_next) * x0 + sigma_next * noise`
//! (`add_noise`, float32) with fresh standard-normal noise.

pub const FAST_WAN_1_3B_DMD_STEPS: [i32; 3] = [1000, 757, 522];
/// Shift of the training-noise table FastWan DMD indexes (not the inference
/// `flow_shift`); equals [`DMD_TRAINING_NOISE_SHIFT`].
pub const FAST_WAN_1_3B_DMD_SHIFT: f64 = DMD_TRAINING_NOISE_SHIFT;
/// FastVideo `DMD_TRAINING_NOISE_SHIFT`.
pub const DMD_TRAINING_NOISE_SHIFT: f64 = 8.0;

#[derive(Debug, Clone)]
pub struct DmdSchedule {
    pub train_timesteps: Vec<i32>,
    /// Table sigma for each DMD step, followed by a trailing `0.0`.
    pub sigmas: Vec<f64>,
    /// Shift of the training-noise table (always [`DMD_TRAINING_NOISE_SHIFT`]).
    pub shift: f64,
    table_sigmas: Vec<f32>,
    table_timesteps: Vec<f32>,
}

/// Scalars for one DMD step, for callers that combine tensors themselves:
/// `x0 = x - sigma_t * v`; then `out = (1 - sigma_next) * x0 + sigma_next * noise`
/// when `sigma_next` is `Some`, else `out = x0`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DmdStepCoeffs {
    pub sigma_t: f64,
    pub sigma_next: Option<f64>,
}

/// `FlowMatchEulerDiscreteScheduler.__init__` sigma / timestep tables (float32).
fn training_table(num_train_timesteps: i32, shift: f64) -> (Vec<f32>, Vec<f32>) {
    let n = num_train_timesteps.max(1) as usize;
    let n_f32 = num_train_timesteps as f32;
    let shift_f32 = shift as f32;
    let shift_m1 = (shift - 1.0) as f32;
    let mut sigmas = Vec::with_capacity(n);
    let mut timesteps = Vec::with_capacity(n);
    for i in 0..n {
        // np.linspace(1, N, N, dtype=float32)[::-1] is exactly N, N-1, ..., 1.
        let t_raw = (n - i) as f32;
        let s = t_raw / n_f32;
        let s = (shift_f32 * s) / (1.0 + shift_m1 * s);
        sigmas.push(s);
        timesteps.push(s * n_f32);
    }
    (sigmas, timesteps)
}

impl DmdSchedule {
    pub fn new(steps: &[i32], num_train_timesteps: i32) -> Self {
        let (table_sigmas, table_timesteps) =
            training_table(num_train_timesteps, DMD_TRAINING_NOISE_SHIFT);
        let mut sched = Self {
            train_timesteps: steps.to_vec(),
            sigmas: Vec::new(),
            shift: DMD_TRAINING_NOISE_SHIFT,
            table_sigmas,
            table_timesteps,
        };
        sched.sigmas = steps
            .iter()
            .map(|&t| sched.sigma_for_timestep(f64::from(t)))
            .chain(std::iter::once(0.0))
            .collect();
        sched
    }

    pub fn fast_wan_1_3b() -> Self {
        Self::new(&FAST_WAN_1_3B_DMD_STEPS, 1000)
    }

    /// Training-table sigmas (float32, descending), index 0 = timestep N.
    pub fn table_sigmas(&self) -> &[f32] {
        &self.table_sigmas
    }

    /// Training-table timesteps (`sigmas * N`, float32).
    pub fn table_timesteps(&self) -> &[f32] {
        &self.table_timesteps
    }

    /// `argmin |timesteps - t|` (float64, first index on ties) → table sigma.
    pub fn sigma_for_timestep(&self, t: f64) -> f64 {
        let mut best = 0usize;
        let mut best_d = f64::INFINITY;
        for (i, &ts) in self.table_timesteps.iter().enumerate() {
            let d = (f64::from(ts) - t).abs();
            if d < best_d {
                best_d = d;
                best = i;
            }
        }
        f64::from(self.table_sigmas[best])
    }

    pub fn num_steps(&self) -> usize {
        self.train_timesteps.len()
    }

    pub fn is_last(&self, i: usize) -> bool {
        i + 1 >= self.train_timesteps.len()
    }

    pub fn step_coeffs(&self, i: usize) -> DmdStepCoeffs {
        DmdStepCoeffs {
            sigma_t: self.sigmas[i],
            sigma_next: if self.is_last(i) {
                None
            } else {
                Some(self.sigmas[i + 1])
            },
        }
    }

    /// Host reference for DMD step `i`. `noise` is required iff `!is_last(i)`.
    pub fn step(
        &self,
        i: usize,
        sample: &[f32],
        velocity: &[f32],
        noise: Option<&[f32]>,
    ) -> Result<Vec<f32>, String> {
        if i >= self.train_timesteps.len() {
            return Err(format!("DMD step {i} past end of schedule"));
        }
        if sample.len() != velocity.len() {
            return Err("sample / velocity length mismatch".into());
        }
        let c = self.step_coeffs(i);
        // pred_noise_to_pred_video: float64 then cast back.
        let x0 = sample
            .iter()
            .zip(velocity)
            .map(|(&x, &v)| (f64::from(x) - c.sigma_t * f64::from(v)) as f32);
        match (c.sigma_next, noise) {
            (None, None) => Ok(x0.collect()),
            (None, Some(_)) => Err("noise must not be supplied on the last DMD step".into()),
            (Some(_), None) => Err("noise required on non-final DMD step".into()),
            (Some(sigma_next), Some(noise)) => {
                if noise.len() != sample.len() {
                    return Err("noise length mismatch".into());
                }
                // add_noise: float32 table sigma, float32 tensor arithmetic.
                let s = sigma_next as f32;
                Ok(x0
                    .zip(noise)
                    .map(|(x0, &n)| (1.0 - s) * x0 + s * n)
                    .collect())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fastwan_steps_are_exact() {
        let sched = DmdSchedule::fast_wan_1_3b();
        assert_eq!(sched.train_timesteps, vec![1000, 757, 522]);
        assert_eq!(sched.shift, DMD_TRAINING_NOISE_SHIFT);
        assert_eq!(sched.sigmas.len(), 4);
        assert_eq!(sched.sigmas[0], 1.0);
        assert_eq!(sched.sigmas[3], 0.0);
    }

    #[test]
    fn training_table_matches_float32_formula() {
        let sched = DmdSchedule::new(&[1000], 1000);
        assert_eq!(sched.table_sigmas().len(), 1000);
        assert_eq!(sched.table_timesteps().len(), 1000);
        assert_eq!(sched.sigma_for_timestep(1000.0), 1.0);
        // Independent recomputation of a few entries (index i ↔ raw t = 1000 - i).
        for &(idx, raw) in &[(0usize, 1000.0f32), (243, 757.0), (478, 522.0), (999, 1.0)] {
            let s = raw / 1000.0f32;
            let s = 8.0f32 * s / (1.0f32 + 7.0f32 * s);
            assert_eq!(sched.table_sigmas()[idx], s, "idx {idx}");
            assert_eq!(sched.table_timesteps()[idx], s * 1000.0f32, "idx {idx}");
        }
    }

    #[test]
    fn fastwan_sigmas_are_table_quantized() {
        let sched = DmdSchedule::fast_wan_1_3b();
        let table = sched.table_sigmas();
        for (&t, &want) in [757, 522].iter().zip(&[0.757, 0.522]) {
            let s = sched.sigma_for_timestep(f64::from(t));
            assert!((s - want).abs() < 1e-3, "t={t} sigma={s}");
            assert!(
                table.iter().any(|&e| f64::from(e) == s),
                "t={t} not a table entry"
            );
        }
        assert_eq!(sched.sigmas[1], sched.sigma_for_timestep(757.0));
        assert_eq!(sched.sigmas[2], sched.sigma_for_timestep(522.0));
    }

    #[test]
    fn lookup_ties_pick_first_index() {
        let sched = DmdSchedule::new(&[1000], 1000);
        let ts = sched.table_timesteps();
        // Exact midpoint between two neighbouring entries → the earlier one.
        let (a, b) = (f64::from(ts[500]), f64::from(ts[501]));
        let mid = 0.5 * (a + b);
        assert_eq!((a - mid).abs(), (b - mid).abs());
        assert_eq!(
            sched.sigma_for_timestep(mid),
            f64::from(sched.table_sigmas()[500])
        );
        // Far outside the table clamps to the ends.
        assert_eq!(sched.sigma_for_timestep(5000.0), 1.0);
        assert_eq!(
            sched.sigma_for_timestep(-5.0),
            f64::from(*sched.table_sigmas().last().unwrap())
        );
    }

    #[test]
    fn float32_add_noise_lookup_agrees_for_integer_timesteps() {
        // `add_noise` does the argmin in float32 (int timestep promoted to f32),
        // `pred_noise_to_pred_video` in float64; they must agree for int steps.
        let sched = DmdSchedule::new(&[1000], 1000);
        for t in 0..=1000i32 {
            let tf = t as f32;
            let mut best = 0usize;
            let mut best_d = f32::INFINITY;
            for (i, &ts) in sched.table_timesteps().iter().enumerate() {
                let d = (ts - tf).abs();
                if d < best_d {
                    best_d = d;
                    best = i;
                }
            }
            assert_eq!(
                f64::from(sched.table_sigmas()[best]),
                sched.sigma_for_timestep(f64::from(t)),
                "t={t}"
            );
        }
    }

    #[test]
    fn step_golden_non_last_and_last() {
        let sched = DmdSchedule::fast_wan_1_3b();
        let x = [0.5f32, -1.25, 2.0];
        let v = [0.1f32, 0.3, -0.7];
        let noise = [1.0f32, -0.5, 0.25];

        let c = sched.step_coeffs(1);
        let (st, sn) = (c.sigma_t, c.sigma_next.unwrap());
        assert_eq!(st, sched.sigmas[1]);
        assert_eq!(sn, sched.sigmas[2]);
        let got = sched.step(1, &x, &v, Some(&noise)).unwrap();
        for k in 0..3 {
            let x0 = (f64::from(x[k]) - st * f64::from(v[k])) as f32;
            let s = sn as f32;
            let want = (1.0 - s) * x0 + s * noise[k];
            assert_eq!(got[k], want);
            assert!(
                (f64::from(got[k])
                    - ((1.0 - sn) * (f64::from(x[k]) - st * f64::from(v[k]))
                        + sn * f64::from(noise[k])))
                .abs()
                    < 1e-5
            );
        }

        assert!(sched.is_last(2));
        assert_eq!(sched.step_coeffs(2).sigma_next, None);
        let got = sched.step(2, &x, &v, None).unwrap();
        let st = sched.sigmas[2];
        for k in 0..3 {
            assert_eq!(got[k], (f64::from(x[k]) - st * f64::from(v[k])) as f32);
        }

        assert!(sched.step(1, &x, &v, None).is_err());
        assert!(sched.step(2, &x, &v, Some(&noise)).is_err());
    }
}
