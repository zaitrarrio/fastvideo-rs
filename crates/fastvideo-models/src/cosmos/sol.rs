//! Cosmos3-Super optimized step policy from `models/cosmos3/optimized/env.sh`
//! on NVlabs/Sana `sol-engine`.
//!
//! TeaCache is threshold 1.15, first eligible step 10, at most 3 consecutive
//! reuses. NVFP4 would cover the middle linear steps and leave the first 3 and
//! last 3 dense. This module decides those steps. It does not quantize, and it
//! does not see the time-embedding signal the runtime stashes inside the DiT.

pub const TEACACHE_THRESHOLD: f64 = 1.15;
pub const TEACACHE_START_STEP: usize = 10;
pub const TEACACHE_MAX_CONSECUTIVE: usize = 3;
pub const FP4_SKIP_FIRST: usize = 3;
pub const FP4_SKIP_LAST: usize = 3;

/// Middle steps are the ones the optimized arm would run as NVFP4 linears.
pub fn fp4_linear(step: usize, num_steps: usize) -> bool {
    step >= FP4_SKIP_FIRST && step + FP4_SKIP_LAST < num_steps
}

#[derive(Debug, Clone)]
pub struct TeaCacheWindow {
    pub threshold: f64,
    pub start_step: usize,
    pub max_consecutive: usize,
    hits: usize,
    acc: f64,
}

impl TeaCacheWindow {
    pub fn official() -> Self {
        Self {
            threshold: TEACACHE_THRESHOLD,
            start_step: TEACACHE_START_STEP,
            max_consecutive: TEACACHE_MAX_CONSECUTIVE,
            hits: 0,
            acc: 0.0,
        }
    }

    /// `indicator` is the rescaled distance of this step's TeaCache signal.
    /// A reuse accumulates it. Crossing the threshold, the warmup, or the
    /// consecutive cap forces a compute and clears the accumulator.
    pub fn decide(&mut self, step: usize, indicator: f64) -> bool {
        let compute = step < self.start_step || self.hits >= self.max_consecutive || {
            self.acc += indicator;
            self.acc >= self.threshold
        };
        if compute {
            self.hits = 0;
            self.acc = 0.0;
            true
        } else {
            self.hits += 1;
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fp4_leaves_the_ends_dense() {
        assert!(!fp4_linear(0, 35));
        assert!(!fp4_linear(2, 35));
        assert!(fp4_linear(3, 35));
        assert!(fp4_linear(31, 35));
        assert!(!fp4_linear(32, 35));
        assert!(!fp4_linear(34, 35));
    }

    #[test]
    fn teacache_waits_then_caps_consecutive_reuses() {
        let mut cache = TeaCacheWindow::official();
        for step in 0..10 {
            assert!(cache.decide(step, 0.0), "{step}");
        }
        assert!(!cache.decide(10, 0.1));
        assert!(!cache.decide(11, 0.1));
        assert!(!cache.decide(12, 0.1));
        assert!(cache.decide(13, 0.1));
        assert!(!cache.decide(14, 0.1));
        assert!(cache.decide(15, 1.15));
    }
}
