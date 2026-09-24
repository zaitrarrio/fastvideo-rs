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

/// Delivered EasyCache knobs. `official()` stays on [`EasyCacheProfile::CodeDefault`]
/// so existing tests do not silently pick up a manifest.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EasyCacheProfile {
    /// Historical in-tree default: threshold 0.05, retain 7, cooldown 1.
    CodeDefault,
    /// Wan 2.2 TI2V-5B and Wan 2.1 14B fullstack manifests: 0.036 / 7 / 1.
    Fullstack,
    /// Wan 2.1 14B `fullstack_tuned`: 0.10 retain 5, cooldown 1.
    Tuned14b,
}

impl EasyCacheProfile {
    pub const fn params(self) -> (f64, usize, usize) {
        match self {
            Self::CodeDefault => (0.05, 7, 1),
            Self::Fullstack => (0.036, 7, 1),
            Self::Tuned14b => (0.10, 5, 1),
        }
    }

    /// `FASTVIDEO_WAN_EASYCACHE_PROFILE`, then `WAN22_EASYCACHE_PROFILE`.
    /// Unset is [`Self::CodeDefault`].
    pub fn from_env() -> Result<Self, String> {
        let raw = std::env::var("FASTVIDEO_WAN_EASYCACHE_PROFILE")
            .or_else(|_| std::env::var("WAN22_EASYCACHE_PROFILE"))
            .unwrap_or_default();
        Self::parse(&raw)
    }

    /// `code-default` / `default` / `official`; `fullstack` / `5b` / `14b`;
    /// `14b-tuned` / `tuned`. Empty is [`Self::CodeDefault`].
    pub fn parse(name: &str) -> Result<Self, String> {
        match name.trim().to_ascii_lowercase().as_str() {
            "" | "code-default" | "code_default" | "default" | "official" => Ok(Self::CodeDefault),
            "fullstack" | "5b" | "14b" | "14b-fullstack" | "14b_fullstack" => Ok(Self::Fullstack),
            "14b-tuned" | "14b_tuned" | "tuned" => Ok(Self::Tuned14b),
            other => Err(format!(
                "wan easycache profile {other:?} is not code-default, fullstack, or 14b-tuned"
            )),
        }
    }
}

impl EasyCache {
    pub fn official(num_steps: usize) -> Result<Self, String> {
        Self::from_profile(num_steps, EasyCacheProfile::CodeDefault)
    }

    pub fn from_profile(num_steps: usize, profile: EasyCacheProfile) -> Result<Self, String> {
        let (threshold, retain, cooldown) = profile.params();
        Self::new(num_steps, threshold, retain, cooldown)
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
    fn official_stays_on_the_code_default_profile() {
        let cache = EasyCache::official(50).unwrap();
        assert_eq!(
            (cache.threshold, cache.retain_steps, cache.cooldown_steps),
            EasyCacheProfile::CodeDefault.params()
        );
        assert_eq!(EasyCacheProfile::CodeDefault.params(), (0.05, 7, 1));
        assert_eq!(EasyCacheProfile::Fullstack.params(), (0.036, 7, 1));
        assert_eq!(EasyCacheProfile::Tuned14b.params(), (0.10, 5, 1));
        assert_eq!(
            EasyCacheProfile::parse("fullstack").unwrap(),
            EasyCacheProfile::Fullstack
        );
        assert_eq!(
            EasyCacheProfile::parse("5b").unwrap(),
            EasyCacheProfile::Fullstack
        );
        assert_eq!(
            EasyCacheProfile::parse("14b-tuned").unwrap(),
            EasyCacheProfile::Tuned14b
        );
        assert_eq!(
            EasyCacheProfile::parse("code-default").unwrap(),
            EasyCacheProfile::CodeDefault
        );
        let tuned = EasyCache::from_profile(40, EasyCacheProfile::Tuned14b).unwrap();
        assert_eq!(tuned.retain_steps, 5);
        assert!((tuned.threshold - 0.10).abs() < 1e-12);
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

/// Wan 2.2 A14B EasyCache placement (`cache_controller.py` `CachedTailStack`).
/// Block 0 always runs; the reuse payload is the residual of blocks 1..=39.
pub const A14B_FRESH_PREFIX_BLOCKS: usize = 1;
pub const A14B_CACHED_TAIL_BLOCKS: usize = 39;
pub const A14B_TOTAL_BLOCKS: usize = 40;
pub const A14B_EASY_THRESHOLD: f64 = 0.30;
pub const A14B_START_STEP: usize = 5;
pub const A14B_TAIL_STEPS: usize = 3;
pub const A14B_MAX_REUSE: usize = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum A14bExpert {
    HighNoise,
    LowNoise,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct A14bDecision {
    pub compute: bool,
    pub reason: &'static str,
    pub estimate: Option<f64>,
    pub accumulator: f64,
}

#[derive(Debug, Clone)]
struct A14bExpertState {
    accumulated: f64,
    consecutive_reuse: usize,
    has_payload: bool,
    has_previous_input: bool,
    k: Option<f64>,
    last_output_norm: Option<f64>,
}

impl Default for A14bExpertState {
    fn default() -> Self {
        Self {
            accumulated: 0.0,
            consecutive_reuse: 0,
            has_payload: false,
            has_previous_input: false,
            k: None,
            last_output_norm: None,
        }
    }
}

/// Per-expert A14B EasyCache. Not the 5B whole-stack controller.
///
/// `models/wan22_t2v_a14b/optimized/cache_controller.py`: threshold 0.30,
/// start 5, tail 3, max_reuse 1, one accumulator per MoE expert.
#[derive(Debug, Clone)]
pub struct A14bCacheController {
    pub threshold: f64,
    pub start_step: usize,
    pub tail_steps: usize,
    pub max_reuse: usize,
    pub num_steps: usize,
    experts: [A14bExpertState; 2],
}

impl A14bCacheController {
    pub fn official(num_steps: usize) -> Result<Self, String> {
        Self::new(
            num_steps,
            A14B_EASY_THRESHOLD,
            A14B_START_STEP,
            A14B_TAIL_STEPS,
            A14B_MAX_REUSE,
        )
    }

    pub fn new(
        num_steps: usize,
        threshold: f64,
        start_step: usize,
        tail_steps: usize,
        max_reuse: usize,
    ) -> Result<Self, String> {
        if threshold <= 0.0 {
            return Err("wan a14b cache requires a positive threshold".into());
        }
        if num_steps == 0 {
            return Err("wan a14b cache requires at least one step".into());
        }
        Ok(Self {
            threshold,
            start_step,
            tail_steps,
            max_reuse,
            num_steps,
            experts: [A14bExpertState::default(), A14bExpertState::default()],
        })
    }

    pub fn placement() -> (usize, usize, usize) {
        (
            A14B_FRESH_PREFIX_BLOCKS,
            A14B_CACHED_TAIL_BLOCKS,
            A14B_TOTAL_BLOCKS,
        )
    }

    /// True when `decide` will read the previous input probe and `k`.
    pub fn needs_input_signal(&self, expert: A14bExpert, step: usize) -> bool {
        self.force_reason(expert, step).is_none()
            && self.expert(expert).k.is_some()
            && self.expert(expert).last_output_norm.is_some()
            && self.expert(expert).has_previous_input
    }

    /// Cond decision for one expert. `input_change` is mean |probe − previous|.
    pub fn decide(&mut self, expert: A14bExpert, step: usize, input_change: f64) -> A14bDecision {
        let (compute, reason, estimate) = if let Some(reason) = self.force_reason(expert, step) {
            self.expert_mut(expert).accumulated = 0.0;
            (true, reason, None)
        } else {
            let state = self.expert(expert);
            let estimate = state.k.unwrap_or(0.0) * input_change
                / state.last_output_norm.unwrap_or(1.0).max(1e-8);
            let threshold = self.threshold;
            let state = self.expert_mut(expert);
            state.accumulated += estimate;
            if state.accumulated < threshold {
                (false, "online_error_below_threshold", Some(estimate))
            } else {
                state.accumulated = 0.0;
                (true, "online_error_refresh", Some(estimate))
            }
        };
        self.expert_mut(expert).has_previous_input = true;
        A14bDecision {
            compute,
            reason,
            estimate,
            accumulator: self.expert(expert).accumulated,
        }
    }

    /// After a computed CFG pair: refresh `k` from the cond (branch-0) tail.
    pub fn note_computed(
        &mut self,
        expert: A14bExpert,
        full_input_change: f64,
        output_change: f64,
        output_norm: f64,
    ) {
        let state = self.expert_mut(expert);
        if state.has_payload {
            state.k = Some(output_change / full_input_change.max(1e-8));
        }
        state.last_output_norm = Some(output_norm);
        state.has_payload = true;
        state.consecutive_reuse = 0;
        state.accumulated = 0.0;
    }

    pub fn note_reused(&mut self, expert: A14bExpert) {
        let state = self.expert_mut(expert);
        state.consecutive_reuse += 1;
    }

    fn force_reason(&self, expert: A14bExpert, step: usize) -> Option<&'static str> {
        let state = self.expert(expert);
        if step < self.start_step {
            Some("warmup_guard")
        } else if step >= self.num_steps.saturating_sub(self.tail_steps) {
            Some("tail_guard")
        } else if !state.has_payload {
            Some("payload_seed")
        } else if self.max_reuse > 0 && state.consecutive_reuse >= self.max_reuse {
            Some("max_reuse_guard")
        } else if !state.has_previous_input || state.k.is_none() || state.last_output_norm.is_none()
        {
            Some("online_error_seed")
        } else {
            None
        }
    }

    fn expert(&self, expert: A14bExpert) -> &A14bExpertState {
        &self.experts[expert as usize]
    }

    fn expert_mut(&mut self, expert: A14bExpert) -> &mut A14bExpertState {
        &mut self.experts[expert as usize]
    }
}

#[cfg(test)]
mod a14b_tests {
    use super::*;

    #[test]
    fn placement_is_block0_plus_tail_39() {
        assert_eq!(A14bCacheController::placement(), (1, 39, 40));
        let cache = A14bCacheController::official(40).unwrap();
        assert!((cache.threshold - 0.30).abs() < 1e-12);
        assert_eq!(cache.start_step, 5);
        assert_eq!(cache.tail_steps, 3);
        assert_eq!(cache.max_reuse, 1);
        assert_ne!(
            (cache.threshold, cache.start_step, cache.tail_steps),
            EasyCacheProfile::CodeDefault.params()
        );
        assert_ne!(cache.threshold, EasyCacheProfile::Fullstack.params().0);
    }

    #[test]
    fn warmup_tail_and_max_reuse_force_compute() {
        let mut cache = A14bCacheController::official(12).unwrap();
        for step in 0..5 {
            let d = cache.decide(A14bExpert::HighNoise, step, 99.0);
            assert!(d.compute, "{step}");
            assert_eq!(d.reason, "warmup_guard");
            cache.note_computed(A14bExpert::HighNoise, 1.0, 0.02, 1.0);
        }
        // Warmup already installed k. estimate = 0.02 * 1 / 1 = 0.02 < 0.30.
        let skip = cache.decide(A14bExpert::HighNoise, 5, 1.0);
        assert!(!skip.compute);
        assert_eq!(skip.reason, "online_error_below_threshold");
        cache.note_reused(A14bExpert::HighNoise);

        // max_reuse 1 forces the next step even if drift is tiny.
        let capped = cache.decide(A14bExpert::HighNoise, 6, 0.0);
        assert!(capped.compute);
        assert_eq!(capped.reason, "max_reuse_guard");
        cache.note_computed(A14bExpert::HighNoise, 1.0, 0.02, 1.0);

        let tail = cache.decide(A14bExpert::HighNoise, 9, 0.0);
        assert!(tail.compute);
        assert_eq!(tail.reason, "tail_guard");
    }

    #[test]
    fn experts_do_not_share_an_accumulator() {
        let mut cache = A14bCacheController::official(20).unwrap();
        for expert in [A14bExpert::HighNoise, A14bExpert::LowNoise] {
            cache.decide(expert, 0, 0.0);
            cache.note_computed(expert, 1.0, 0.0, 1.0);
            cache.decide(expert, 5, 0.0);
            cache.note_computed(expert, 1.0, 0.20, 1.0);
        }
        let high = cache.decide(A14bExpert::HighNoise, 6, 1.0);
        assert!(!high.compute);
        assert!((high.estimate.unwrap() - 0.20).abs() < 1e-12);
        let low = cache.decide(A14bExpert::LowNoise, 6, 0.1);
        assert!(!low.compute);
        assert!((low.estimate.unwrap() - 0.02).abs() < 1e-12);
        assert!((low.accumulator - 0.02).abs() < 1e-12);
        assert!((high.accumulator - 0.20).abs() < 1e-12);
    }
}
