//! LingBot FlowMatch schedule (`flow_shift=3`).

use crate::schedulers::FlowMatchEulerDiscreteScheduler;

use super::config::LingBotPreset;

#[derive(Debug, Clone)]
pub struct LingBotSchedule {
    pub inner: FlowMatchEulerDiscreteScheduler,
    pub flow_shift: f64,
}

impl LingBotSchedule {
    pub fn new(num_steps: usize, preset: LingBotPreset) -> Self {
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
    fn shift_is_three() {
        let s = LingBotSchedule::new(8, LingBotPreset::Dense13b);
        assert!((s.flow_shift - 3.0).abs() < 1e-12);
        assert_eq!(s.timesteps().len(), 8);
    }
}
