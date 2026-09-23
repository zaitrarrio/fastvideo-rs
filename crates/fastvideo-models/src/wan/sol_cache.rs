//! Wan step-cache controllers from `models/wan22_ti2v_5b` on NVlabs/Sana
//! `sol-engine` (`WAN22_CACHE_FAMILY`).
//!
//! EasyCache decides from the raw latent and reuses `output - input`.
//! TeaCache decides from the timestep projection and reuses the residual
//! across the transformer blocks. Defaults match the published env.

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

/// Which CFG branch a TeaCache residual belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TeaBranch {
    Cond,
    Uncond,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TeaCacheDecision {
    pub compute: bool,
    pub reason: &'static str,
    pub relative_l1: Option<f64>,
    pub indicator: Option<f64>,
    pub accumulator: f64,
}

#[derive(Debug, Clone)]
struct TeaBranchState {
    has_signal: bool,
    has_residual: bool,
    acc: f64,
    hits: usize,
    since: usize,
}

impl Default for TeaBranchState {
    fn default() -> Self {
        Self {
            has_signal: false,
            has_residual: false,
            acc: 0.0,
            hits: 0,
            since: 0,
        }
    }
}

/// Block-residual TeaCache. The signal is the Wan timestep projection.
/// `coefficients` rescale relative L1 with Horner's method; `[1, 0]` leaves it unchanged.
#[derive(Debug, Clone)]
pub struct SolTeaCache {
    pub threshold: f64,
    pub start_step: usize,
    pub end_step: usize,
    pub max_hits: usize,
    pub periodic: usize,
    pub coefficients: Vec<f64>,
    branches: [TeaBranchState; 2],
}

impl SolTeaCache {
    /// Threshold 0.12, warmup the first two steps, recompute the last two,
    /// identity rescale.
    pub fn official(num_steps: usize) -> Result<Self, String> {
        Self::new(0.12, 2, num_steps.saturating_sub(2), 0, 0, vec![1.0, 0.0])
    }

    pub fn new(
        threshold: f64,
        start_step: usize,
        end_step: usize,
        max_hits: usize,
        periodic: usize,
        coefficients: Vec<f64>,
    ) -> Result<Self, String> {
        if threshold <= 0.0 {
            return Err("wan teacache requires a positive threshold".into());
        }
        if coefficients.is_empty() {
            return Err("wan teacache requires at least one coefficient".into());
        }
        Ok(Self {
            threshold,
            start_step,
            end_step,
            max_hits,
            periodic,
            coefficients,
            branches: [TeaBranchState::default(), TeaBranchState::default()],
        })
    }

    pub fn needs_signal(&self, branch: TeaBranch, step: usize) -> bool {
        self.force_reason(branch, step).is_none()
    }

    /// `relative_l1` is mean |signal − previous| / mean |previous|. Ignored on a forced step.
    pub fn decide(&mut self, branch: TeaBranch, step: usize, relative_l1: f64) -> TeaCacheDecision {
        let (compute, reason, relative_l1, indicator) =
            if let Some(reason) = self.force_reason(branch, step) {
                (true, reason, None, None)
            } else {
                let indicator = poly(&self.coefficients, relative_l1);
                let state = self.branch_mut(branch);
                state.acc += indicator;
                let compute = state.acc >= self.threshold;
                (
                    compute,
                    if compute {
                        "threshold"
                    } else {
                        "below_threshold"
                    },
                    Some(relative_l1),
                    Some(indicator),
                )
            };
        self.branch_mut(branch).has_signal = true;
        let accumulator = self.branch(branch).acc;
        TeaCacheDecision {
            compute,
            reason,
            relative_l1,
            indicator,
            accumulator,
        }
    }

    pub fn note_computed(&mut self, branch: TeaBranch) {
        let state = self.branch_mut(branch);
        state.acc = 0.0;
        state.hits = 0;
        state.since = 0;
        state.has_residual = true;
    }

    pub fn note_reused(&mut self, branch: TeaBranch) {
        let state = self.branch_mut(branch);
        state.hits += 1;
        state.since += 1;
    }

    fn force_reason(&self, branch: TeaBranch, step: usize) -> Option<&'static str> {
        let state = self.branch(branch);
        if step < self.start_step {
            Some("warmup")
        } else if step >= self.end_step {
            Some("cooldown")
        } else if !state.has_signal || !state.has_residual {
            Some("initialize")
        } else if self.periodic > 0 && state.since >= self.periodic {
            Some("periodic")
        } else if self.max_hits > 0 && state.hits >= self.max_hits {
            Some("hit_cap")
        } else {
            None
        }
    }

    fn branch(&self, branch: TeaBranch) -> &TeaBranchState {
        &self.branches[branch as usize]
    }

    fn branch_mut(&mut self, branch: TeaBranch) -> &mut TeaBranchState {
        &mut self.branches[branch as usize]
    }
}

fn poly(coefficients: &[f64], x: f64) -> f64 {
    let mut value = 0.0;
    for coefficient in coefficients {
        value = value * x + coefficient;
    }
    value
}

#[cfg(test)]
mod tea_tests {
    use super::*;

    #[test]
    fn warmup_then_cooldown_bookend_the_schedule() {
        let mut cache = SolTeaCache::official(10).unwrap();
        for step in 0..2 {
            let d = cache.decide(TeaBranch::Cond, step, 9.0);
            assert!(d.compute);
            assert_eq!(d.reason, "warmup");
            cache.note_computed(TeaBranch::Cond);
        }
        for step in 8..10 {
            let d = cache.decide(TeaBranch::Cond, step, 0.0);
            assert!(d.compute, "{step}");
            assert_eq!(d.reason, "cooldown");
        }
    }

    #[test]
    fn identity_rescale_accumulates_relative_l1() {
        let mut cache = SolTeaCache::new(0.12, 0, 100, 0, 0, vec![1.0, 0.0]).unwrap();
        assert_eq!(cache.decide(TeaBranch::Cond, 0, 0.0).reason, "initialize");
        cache.note_computed(TeaBranch::Cond);
        let skip = cache.decide(TeaBranch::Cond, 1, 0.05);
        assert!(!skip.compute);
        assert!((skip.indicator.unwrap() - 0.05).abs() < 1e-12);
        assert!((skip.accumulator - 0.05).abs() < 1e-12);
        cache.note_reused(TeaBranch::Cond);
        let hit = cache.decide(TeaBranch::Cond, 2, 0.08);
        assert!(hit.compute);
        assert_eq!(hit.reason, "threshold");
        cache.note_computed(TeaBranch::Cond);
        assert_eq!(cache.decide(TeaBranch::Cond, 3, 0.01).accumulator, 0.01);
    }

    #[test]
    fn branches_do_not_share_an_accumulator() {
        let mut cache = SolTeaCache::new(0.12, 0, 100, 0, 0, vec![2.0, 0.0]).unwrap();
        cache.decide(TeaBranch::Cond, 0, 0.0);
        cache.note_computed(TeaBranch::Cond);
        cache.decide(TeaBranch::Uncond, 0, 0.0);
        cache.note_computed(TeaBranch::Uncond);
        let cond = cache.decide(TeaBranch::Cond, 1, 0.04);
        assert!(!cond.compute);
        assert!((cond.indicator.unwrap() - 0.08).abs() < 1e-12);
        let uncond = cache.decide(TeaBranch::Uncond, 1, 0.01);
        assert!((uncond.accumulator - 0.02).abs() < 1e-12);
    }

    #[test]
    fn hit_cap_forces_a_recompute() {
        let mut cache = SolTeaCache::new(10.0, 0, 100, 1, 0, vec![1.0, 0.0]).unwrap();
        cache.decide(TeaBranch::Cond, 0, 0.0);
        cache.note_computed(TeaBranch::Cond);
        let skip = cache.decide(TeaBranch::Cond, 1, 0.01);
        assert!(!skip.compute);
        cache.note_reused(TeaBranch::Cond);
        let forced = cache.decide(TeaBranch::Cond, 2, 0.01);
        assert!(forced.compute);
        assert_eq!(forced.reason, "hit_cap");
    }
}
