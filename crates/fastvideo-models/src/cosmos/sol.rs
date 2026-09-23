//! Cosmos3-Super optimized step policy from `models/cosmos3/optimized/env.sh`
//! on NVlabs/Sana `sol-engine`.
//!
//! TeaCache is threshold 1.15, first eligible step 10, at most 3 consecutive
//! reuses. The published signal is the DiT time embed (`time_embed` /
//! `temb`). The comparison is mean-absolute relative L1 of that embed against
//! the previous step. NVFP4 would cover the middle linear steps and leave the
//! first 3 and last 3 dense. This module decides those steps and compares the
//! time-embed; it does not quantize.

pub const TEACACHE_THRESHOLD: f64 = 1.15;
pub const TEACACHE_START_STEP: usize = 10;
pub const TEACACHE_MAX_CONSECUTIVE: usize = 3;
pub const FP4_SKIP_FIRST: usize = 3;
pub const FP4_SKIP_LAST: usize = 3;

/// `FASTVIDEO_COSMOS_SOL=teacache` (or `1`) turns the step cache on.
pub fn teacache_requested(value: Option<&str>) -> bool {
    match value.map(str::trim) {
        Some("1") => true,
        Some(v) => v.eq_ignore_ascii_case("teacache"),
        None => false,
    }
}

/// Middle steps are the ones the optimized arm would run as NVFP4 linears.
pub fn fp4_linear(step: usize, num_steps: usize) -> bool {
    step >= FP4_SKIP_FIRST && step + FP4_SKIP_LAST < num_steps
}

/// Mean |current − previous| / mean |previous|. Matches Wan / generic TeaCache.
pub fn relative_l1(current: &[f32], previous: &[f32]) -> f64 {
    let n = current.len().max(1) as f64;
    let mut num = 0.0;
    let mut den = 0.0;
    for (a, b) in current.iter().zip(previous.iter()) {
        num += (f64::from(*a) - f64::from(*b)).abs();
        den += f64::from(*b).abs();
    }
    (num / n) / (den / n).max(1e-8)
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

/// Time-embed TeaCache. The first forward of a step decides; CFG uncond follows.
#[derive(Debug, Clone)]
pub struct SolCosmosTea {
    window: TeaCacheWindow,
    previous: Option<Vec<f32>>,
    last_compute: bool,
}

impl SolCosmosTea {
    pub fn official() -> Self {
        Self {
            window: TeaCacheWindow::official(),
            previous: None,
            last_compute: true,
        }
    }

    /// Cond (or the only CFG branch) decision from this step's time embed.
    pub fn decide(&mut self, step: usize, signal: &[f32]) -> bool {
        let indicator = match &self.previous {
            Some(prev) if step >= self.window.start_step => relative_l1(signal, prev),
            _ => 0.0,
        };
        let compute = self.window.decide(step, indicator);
        self.previous = Some(signal.to_vec());
        self.last_compute = compute;
        compute
    }

    /// Uncond follows the cond pair of the same step.
    pub fn follow(&self) -> bool {
        self.last_compute
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

    #[test]
    fn env_is_off_until_teacache_or_one() {
        assert!(!teacache_requested(None));
        assert!(!teacache_requested(Some("")));
        assert!(!teacache_requested(Some("off")));
        assert!(teacache_requested(Some("1")));
        assert!(teacache_requested(Some("teacache")));
        assert!(teacache_requested(Some("TeaCache")));
    }

    #[test]
    fn identical_time_embeds_reuse_after_warmup() {
        let mut cache = SolCosmosTea::official();
        let signal = [0.25f32, -0.5, 1.0];
        for step in 0..10 {
            assert!(cache.decide(step, &signal), "{step}");
            assert!(cache.follow());
        }
        assert!(!cache.decide(10, &signal));
        assert!(!cache.follow());
        assert!(!cache.decide(11, &signal));
        assert!(!cache.decide(12, &signal));
        assert!(cache.decide(13, &signal));
    }

    #[test]
    fn relative_l1_is_mean_abs_over_mean_abs() {
        let prev = [2.0f32, 0.0];
        let cur = [4.0f32, 2.0];
        // mean |d| = 2, mean |prev| = 1
        assert!((relative_l1(&cur, &prev) - 2.0).abs() < 1e-12);
    }
}
