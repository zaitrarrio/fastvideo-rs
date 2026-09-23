//! LongCat FlowMatch schedule.

use crate::schedulers::FlowMatchEulerDiscreteScheduler;

use super::config::LongCatPreset;

#[derive(Debug, Clone)]
pub struct LongCatSchedule {
    pub inner: FlowMatchEulerDiscreteScheduler,
    pub flow_shift: f64,
}

impl LongCatSchedule {
    pub fn new(num_steps: usize, preset: LongCatPreset) -> Self {
        let flow_shift = preset.flow_shift();
        let mut inner = FlowMatchEulerDiscreteScheduler::new(1_000, flow_shift);
        inner.set_timesteps(num_steps.max(1));
        Self { inner, flow_shift }
    }

    pub fn timesteps(&self) -> &[f64] {
        self.inner.inference_timesteps()
    }

    pub fn sigmas(&self) -> &[f64] {
        self.inner.inference_sigmas()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn steps_len() {
        let s = LongCatSchedule::new(12, LongCatPreset::T2v480p);
        assert_eq!(s.timesteps().len(), 12);
        assert!((s.flow_shift - 1.0).abs() < 1e-12);
    }
}
