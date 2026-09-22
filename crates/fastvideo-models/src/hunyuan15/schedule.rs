//! Flow-match Euler schedule for HunyuanVideo 1.5.

use crate::schedulers::FlowMatchEulerDiscreteScheduler;

use super::config::Hunyuan15Preset;

/// Thin wrapper: FlowMatch Euler with the preset's `flow_shift`.
#[derive(Debug, Clone)]
pub struct Hunyuan15Schedule {
    pub inner: FlowMatchEulerDiscreteScheduler,
    pub flow_shift: f64,
    pub num_inference_steps: usize,
}

impl Hunyuan15Schedule {
    pub fn new(num_steps: usize, preset: Hunyuan15Preset) -> Self {
        Self::with_shift(num_steps, preset.flow_shift())
    }

    pub fn with_shift(num_steps: usize, flow_shift: f64) -> Self {
        let mut inner = FlowMatchEulerDiscreteScheduler::new(1_000, flow_shift);
        inner.set_timesteps(num_steps.max(1));
        Self {
            inner,
            flow_shift,
            num_inference_steps: num_steps.max(1),
        }
    }

    pub fn num_steps(&self) -> usize {
        self.num_inference_steps
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
    fn shift_follows_preset() {
        let s = Hunyuan15Schedule::new(20, Hunyuan15Preset::T2v480p);
        assert_eq!(s.flow_shift, 5.0);
        assert_eq!(s.num_steps(), 20);
        assert_eq!(s.timesteps().len(), 20);
    }
}
