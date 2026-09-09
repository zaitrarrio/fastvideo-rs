//! Distribution Matching Distillation (DMD) discrete timesteps.
//!
//! FastVideo `FastWan2_1_T2V_480P_Config` uses
//! `dmd_denoising_steps = [1000, 757, 522]` and `flow_shift = 8.0`.
//! Dense DMD indexes the complete training-noise table; this is independent
//! of the configurable inference `flow_shift` used by UniPC.

use super::flow_match::apply_shift;

pub const FAST_WAN_1_3B_DMD_STEPS: [i32; 3] = [1000, 757, 522];
pub const FAST_WAN_1_3B_DMD_SHIFT: f64 = 8.0;
/// FastVideo `DMD_TRAINING_NOISE_SHIFT`.
pub const DMD_TRAINING_NOISE_SHIFT: f64 = 8.0;

#[derive(Debug, Clone)]
pub struct DmdSchedule {
    pub train_timesteps: Vec<i32>,
    pub sigmas: Vec<f64>,
    pub shift: f64,
}

impl DmdSchedule {
    pub fn new(steps: &[i32], shift: f64, num_train_timesteps: i32) -> Self {
        let sigmas = steps
            .iter()
            .map(|&t| {
                let sigma = f64::from(t) / f64::from(num_train_timesteps);
                apply_shift(sigma, shift)
            })
            .chain(std::iter::once(0.0))
            .collect();
        Self {
            train_timesteps: steps.to_vec(),
            sigmas,
            shift,
        }
    }

    pub fn fast_wan_1_3b() -> Self {
        Self::new(&FAST_WAN_1_3B_DMD_STEPS, FAST_WAN_1_3B_DMD_SHIFT, 1000)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fastwan_steps_are_exact() {
        let sched = DmdSchedule::fast_wan_1_3b();
        assert_eq!(sched.train_timesteps, vec![1000, 757, 522]);
        assert_eq!(sched.sigmas.len(), 4);
        assert!((sched.sigmas[0] - 1.0).abs() < 1e-12); // shift(1.0, 8) = 1
        assert!((sched.sigmas[3] - 0.0).abs() < 1e-12);
        let s757 = apply_shift(0.757, 8.0);
        assert!((sched.sigmas[1] - s757).abs() < 1e-12);
    }
}
