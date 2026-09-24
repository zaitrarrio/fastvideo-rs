//! Cosmos3-Super FlowMatch Euler schedule.
//!
//! Predict2 Video2World is EDM and records flow-shift 10 without applying it.
//! This Super path is FlowMatch, so the published shift is applied.

use crate::cosmos::sol::OFFICIAL_STEPS;
use crate::schedulers::FlowMatchEulerDiscreteScheduler;

use super::config::Cosmos3Preset;

/// Thin wrapper: FlowMatch Euler with the Super `flow_shift`.
#[derive(Debug, Clone)]
pub struct Cosmos3Schedule {
    pub inner: FlowMatchEulerDiscreteScheduler,
    pub flow_shift: f64,
    pub num_inference_steps: usize,
}

impl Cosmos3Schedule {
    pub fn official() -> Self {
        Self::new(OFFICIAL_STEPS, Cosmos3Preset::Super64bT2v)
    }

    pub fn new(num_steps: usize, preset: Cosmos3Preset) -> Self {
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
    use crate::cosmos::sol::{
        fp4_linear, official_requested, teacache_requested, TEACACHE_MAX_CONSECUTIVE,
        TEACACHE_START_STEP, TEACACHE_THRESHOLD,
    };
    use crate::schedulers::flow_match::apply_shift;

    #[test]
    fn official_applies_recorded_flow_shift() {
        let s = Cosmos3Schedule::official();
        assert_eq!(s.flow_shift, 10.0);
        assert_eq!(s.num_steps(), 35);
        assert_eq!(s.timesteps().len(), 35);
        assert!((s.sigmas()[0] - apply_shift(1.0, 10.0)).abs() < 1e-9);
        assert!((s.sigmas().last().copied().unwrap() - 0.0).abs() < 1e-12);
    }

    #[test]
    fn teacache_knobs_are_predict2_sol() {
        assert_eq!(TEACACHE_THRESHOLD, 1.15);
        assert_eq!(TEACACHE_START_STEP, 10);
        assert_eq!(TEACACHE_MAX_CONSECUTIVE, 3);
        assert!(teacache_requested(Some("teacache")));
        assert!(official_requested(Some("1")));
        assert!(fp4_linear(10, 35));
        assert!(!fp4_linear(0, 35));
        assert!(!fp4_linear(32, 35));
    }
}
