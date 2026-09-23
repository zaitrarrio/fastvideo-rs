//! Wan step-cache controllers from `models/wan22_ti2v_5b` on NVlabs/Sana
//! `sol-engine` (`WAN22_CACHE_FAMILY`).
//!
//! EasyCache decides from the raw latent and reuses `output - input`.
//! TeaCache decides from the timestep projection and reuses the residual
//! across the transformer blocks. Defaults match the published env.
//!
//! TaylorSeer lite forecasts `proj_out` and skips the block stack on the
//! forecast steps. The schedule is `wan_cache.py` `TaylorSeerRuntime`
//! (`WAN22_TAYLOR_*`). The factors are diffusers `TaylorSeerState`: divided
//! differences on compute steps, then `offset^k / k!` on a forecast.

/// Published Wan2.2 TaylorSeer lite defaults (`cache_runtime.py`).
pub const TAYLOR_INTERVAL: usize = 3;
pub const TAYLOR_WARMUP: usize = 3;
pub const TAYLOR_ORDER: usize = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TaylorDecision {
    pub compute: bool,
    pub reason: &'static str,
    pub step: usize,
}

/// Diffusers `TaylorSeerCacheHook._measure_should_compute`.
///
/// `current_step` starts at -1. Each forward increments it, then warmup,
/// `(step - warmup - 1) % interval == 0`, or cooldown forces a full proj_out.
#[derive(Debug, Clone)]
pub struct TaylorSchedule {
    pub interval: usize,
    pub warmup: usize,
    pub cooldown_start: usize,
    pub max_order: usize,
    current_step: isize,
}

impl TaylorSchedule {
    pub fn official(num_steps: usize) -> Result<Self, String> {
        Self::new(
            TAYLOR_INTERVAL,
            TAYLOR_WARMUP,
            num_steps.saturating_sub(2),
            TAYLOR_ORDER,
        )
    }

    pub fn new(
        interval: usize,
        warmup: usize,
        cooldown_start: usize,
        max_order: usize,
    ) -> Result<Self, String> {
        if interval == 0 {
            return Err("wan taylorseer requires a positive interval".into());
        }
        Ok(Self {
            interval,
            warmup,
            cooldown_start,
            max_order,
            current_step: -1,
        })
    }

    pub fn current_step(&self) -> isize {
        self.current_step
    }

    pub fn begin_forward(&mut self) -> TaylorDecision {
        self.current_step += 1;
        let step = self.current_step as usize;
        let refresh =
            (self.current_step - self.warmup as isize - 1).rem_euclid(self.interval as isize) == 0;
        let (compute, reason) = if step < self.warmup {
            (true, "warmup")
        } else if step >= self.cooldown_start {
            (true, "cooldown")
        } else if refresh {
            (true, "interval_refresh")
        } else {
            (false, "taylor_forecast")
        };
        TaylorDecision {
            compute,
            reason,
            step,
        }
    }
}

/// Order-0 value plus divided differences. `delta_step` is the forward gap
/// since the previous compute (`TaylorSeerState.update`).
pub fn taylor_update(
    previous: &[Vec<f32>],
    features: &[f32],
    delta_step: isize,
    max_order: usize,
) -> Result<Vec<Vec<f32>>, String> {
    if delta_step == 0 && !previous.is_empty() {
        return Err("wan taylorseer delta step cannot be zero".into());
    }
    let mut factors = vec![features.to_vec()];
    if previous.is_empty() {
        return Ok(factors);
    }
    let inv = 1.0 / delta_step as f32;
    for j in 0..max_order {
        let Some(prev) = previous.get(j) else {
            break;
        };
        if prev.len() != factors[j].len() {
            return Err("wan taylorseer factor length changed".into());
        }
        factors.push(
            factors[j]
                .iter()
                .zip(prev)
                .map(|(next, old)| (next - old) * inv)
                .collect(),
        );
    }
    Ok(factors)
}

/// `sum_k factor_k * offset^k / k!` (`TaylorSeerState.predict`).
pub fn taylor_predict(factors: &[Vec<f32>], step_offset: isize) -> Result<Vec<f32>, String> {
    let base = factors.first().ok_or("wan taylorseer has no factors")?;
    let mut output = vec![0.0f32; base.len()];
    let mut pow = 1.0f64;
    let mut fact = 1.0f64;
    for (order, factor) in factors.iter().enumerate() {
        if factor.len() != base.len() {
            return Err("wan taylorseer factor length changed".into());
        }
        if order > 0 {
            pow *= step_offset as f64;
            fact *= order as f64;
        }
        let coeff = (pow / fact) as f32;
        for (dst, value) in output.iter_mut().zip(factor) {
            *dst += value * coeff;
        }
    }
    Ok(output)
}

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

    #[test]
    fn taylor_lite_matches_the_diffusers_step_counter() {
        let mut schedule = TaylorSchedule::official(10).unwrap();
        let mut saw = Vec::new();
        for _ in 0..10 {
            saw.push(schedule.begin_forward().compute);
        }
        // warmup 0..2, then (step - 3 - 1) % 3 == 0 refreshes, cooldown at 8.
        assert_eq!(
            saw,
            vec![true, true, true, false, true, false, false, true, true, true]
        );
        assert_eq!(schedule.begin_forward().reason, "cooldown");
    }

    #[test]
    fn order_one_forecast_is_value_plus_difference() {
        let first = taylor_update(&[], &[2.0, 4.0], 1, 1).unwrap();
        assert_eq!(first, vec![vec![2.0, 4.0]]);
        let second = taylor_update(&first, &[6.0, 8.0], 2, 1).unwrap();
        assert_eq!(second[0], vec![6.0, 8.0]);
        assert_eq!(second[1], vec![2.0, 2.0]);
        assert_eq!(taylor_predict(&second, 1).unwrap(), vec![8.0, 10.0]);
        assert!(taylor_update(&second, &[1.0, 1.0], 0, 1).is_err());
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
