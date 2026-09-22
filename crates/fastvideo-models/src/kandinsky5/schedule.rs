//! Kandinsky 5 FlowMatch schedule (flow_shift 5).

use crate::schedulers::FlowMatchEulerDiscreteScheduler;

use super::config::Kandinsky5Preset;

#[derive(Debug, Clone)]
pub struct Kandinsky5Schedule {
    pub inner: FlowMatchEulerDiscreteScheduler,
    pub flow_shift: f64,
    pub num_inference_steps: usize,
}

impl Kandinsky5Schedule {
    pub fn new(num_steps: usize, preset: Kandinsky5Preset) -> Self {
        let flow_shift = preset.flow_shift();
        let mut inner = FlowMatchEulerDiscreteScheduler::new(1_000, flow_shift);
        inner.set_timesteps(num_steps.max(1));
        Self {
            inner,
            flow_shift,
            num_inference_steps: num_steps.max(1),
        }
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
    fn shift_is_five() {
        let s = Kandinsky5Schedule::new(10, Kandinsky5Preset::LiteT2v5s);
        assert_eq!(s.flow_shift, 5.0);
        assert_eq!(s.timesteps().len(), 10);
    }
}
