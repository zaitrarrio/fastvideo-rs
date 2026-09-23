//! Wan EasyCache from `models/wan22_ti2v_5b` cache runtime on NVlabs/Sana
//! `sol-engine` (`WAN22_CACHE_FAMILY=easycache`).
//!
//! The cond branch decides. The uncond branch follows that decision. A reuse
//! step adds the stored `output - input` residual instead of running the
//! transformer. Defaults match the published env: threshold 0.05, retain the
//! first 7 steps, recompute the last step.

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EasyCacheDecision {
    pub compute: bool,
    pub reason: &'static str,
    pub estimate: Option<f64>,
    pub accumulator: f64,
    pub k: Option<f64>,
}

#[derive(Debug, Clone)]
pub struct EasyCache {
    pub threshold: f64,
    pub retain_steps: usize,
    pub cooldown_steps: usize,
    pub num_steps: usize,
    has_previous_step: bool,
    has_residual: bool,
    k: Option<f64>,
    acc: f64,
    has_last_full: bool,
    last_cond_compute: bool,
}

impl EasyCache {
    pub fn official(num_steps: usize) -> Result<Self, String> {
        Self::new(num_steps, 0.05, 7, 1)
    }

    pub fn new(
        num_steps: usize,
        threshold: f64,
        retain_steps: usize,
        cooldown_steps: usize,
    ) -> Result<Self, String> {
        if threshold <= 0.0 {
            return Err("wan easycache requires a positive threshold".into());
        }
        if num_steps == 0 {
            return Err("wan easycache requires at least one step".into());
        }
        Ok(Self {
            threshold,
            retain_steps,
            cooldown_steps,
            num_steps,
            has_previous_step: false,
            has_residual: false,
            k: None,
            acc: 0.0,
            has_last_full: false,
            last_cond_compute: true,
        })
    }

    /// True when `decide_cond` will read the input drift and the last full output.
    pub fn needs_cond_signal(&self, step: usize) -> bool {
        !self.forced(step) && self.has_previous_step && self.has_residual && self.k.is_some()
    }

    /// Cond decision. `input_change` is mean |x − previous step input|.
    /// `output_norm` is mean |last full output|. Both are ignored on a forced step.
    pub fn decide_cond(
        &mut self,
        step: usize,
        input_change: f64,
        output_norm: f64,
    ) -> EasyCacheDecision {
        let (compute, reason, estimate) = if step < self.retain_steps {
            self.acc = 0.0;
            (true, "warmup", None)
        } else if step >= self.num_steps.saturating_sub(self.cooldown_steps) {
            self.acc = 0.0;
            (true, "cooldown", None)
        } else if !self.has_previous_step || !self.has_residual || self.k.is_none() {
            (true, "initialize", None)
        } else {
            let output_norm = output_norm.max(1e-8);
            let estimate = self.k.unwrap_or(0.0) * input_change / output_norm;
            self.acc += estimate;
            if self.acc >= self.threshold {
                self.acc = 0.0;
                (true, "threshold", Some(estimate))
            } else {
                (false, "below_threshold", Some(estimate))
            }
        };
        self.has_previous_step = true;
        self.last_cond_compute = compute;
        EasyCacheDecision {
            compute,
            reason,
            estimate,
            accumulator: self.acc,
            k: self.k,
        }
    }

    /// After a cond transformer call. Returns the refreshed `k` when a previous
    /// full pair existed. `full_input_change` is mean |x − last full input|.
    pub fn note_cond_computed(
        &mut self,
        full_input_change: f64,
        output_change: f64,
    ) -> Option<f64> {
        let refreshed = if self.has_last_full {
            let k = output_change / full_input_change.max(1e-8);
            self.k = Some(k);
            Some(k)
        } else {
            None
        };
        self.has_last_full = true;
        self.has_residual = true;
        refreshed
    }

    /// Uncond follows the cond pair. It still computes when it has no residual yet.
    pub fn uncond_compute(&self, has_uncond_residual: bool) -> bool {
        self.last_cond_compute || !has_uncond_residual
    }

    fn forced(&self, step: usize) -> bool {
        step < self.retain_steps || step >= self.num_steps.saturating_sub(self.cooldown_steps)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn warmup_and_cooldown_always_compute() {
        let mut cache = EasyCache::official(10).unwrap();
        for step in 0..7 {
            let d = cache.decide_cond(step, 99.0, 1.0);
            assert!(d.compute, "{step}");
            assert_eq!(d.reason, "warmup");
            cache.note_cond_computed(1.0, 1.0);
        }
        let tail = cache.decide_cond(9, 0.0, 1.0);
        assert!(tail.compute);
        assert_eq!(tail.reason, "cooldown");
    }

    #[test]
    fn drift_accumulates_until_the_threshold() {
        let mut cache = EasyCache::new(20, 0.05, 1, 1).unwrap();
        assert!(cache.decide_cond(0, 0.0, 1.0).compute);
        cache.note_cond_computed(1.0, 0.0);
        // Second compute installs k. input change 1, output change 0.02 → k = 0.02.
        assert_eq!(cache.decide_cond(1, 0.0, 1.0).reason, "initialize");
        assert_eq!(cache.note_cond_computed(1.0, 0.02), Some(0.02));

        // estimate = 0.02 * 1.0 / 1.0 = 0.02 < 0.05
        let skip = cache.decide_cond(2, 1.0, 1.0);
        assert!(!skip.compute);
        assert_eq!(skip.reason, "below_threshold");
        assert!((skip.accumulator - 0.02).abs() < 1e-12);

        // 0.02 + 0.04 >= 0.05, then the accumulator resets.
        let hit = cache.decide_cond(3, 2.0, 1.0);
        assert!(hit.compute);
        assert_eq!(hit.reason, "threshold");
        assert_eq!(hit.accumulator, 0.0);
        assert!(cache.uncond_compute(true));
        cache.note_cond_computed(1.0, 0.02);
        let skip = cache.decide_cond(4, 1.0, 1.0);
        assert!(!skip.compute);
        assert!(cache.uncond_compute(false));
        assert!(!cache.uncond_compute(true));
    }

    #[test]
    fn rejects_a_non_positive_threshold() {
        assert!(EasyCache::new(4, 0.0, 7, 1).is_err());
    }
}
