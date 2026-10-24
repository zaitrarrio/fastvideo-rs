//! The pre-B4a step-cache controllers (main `2e3c71b1`), kept as regression
//! fixtures for the equality tests in `equality.rs`. Each module names the
//! file and lines it comes from. Every module is copied verbatim (only the
//! constants it reads are restated) except `wan_teacache4wan21`: that
//! controller lived inside a cudarc runtime with its tensors, so it is a
//! hand reduction of `maybe_reuse` / `store` to the decision, with every
//! arithmetic line (the f32 accumulator, the written-out quartic, the
//! `step <= 1` warm-up, `accumulated < thresh`) kept as written. Test-only:
//! nothing outside `equality.rs` uses them.
#![allow(dead_code, clippy::all)]

/// `crates/fastvideo-models/src/wan/sol_cache.rs:13-146`.
pub mod wan_taylor {
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
            let refresh = (self.current_step - self.warmup as isize - 1)
                .rem_euclid(self.interval as isize)
                == 0;
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
}

/// `crates/fastvideo-models/src/wan/sol_cache.rs:148-321`.
pub mod wan_easy {
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
}

/// `crates/fastvideo-models/src/wan/sol_cache.rs:432-596`.
pub mod wan_tea {
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
        pub fn decide(
            &mut self,
            branch: TeaBranch,
            step: usize,
            relative_l1: f64,
        ) -> TeaCacheDecision {
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
}

/// `crates/fastvideo-models/src/wan/sol_cache.rs:663-850`.
pub mod wan_a14b {
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
        pub fn decide(
            &mut self,
            expert: A14bExpert,
            step: usize,
            input_change: f64,
        ) -> A14bDecision {
            let (compute, reason, estimate) = if let Some(reason) = self.force_reason(expert, step)
            {
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
            } else if !state.has_previous_input
                || state.k.is_none()
                || state.last_output_norm.is_none()
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
}

/// `crates/fastvideo-models/src/h3/sol.rs:405-558`.
pub mod h3_tea {
    pub const RTX_TEACACHE_THRESHOLD: f64 = 0.10;
    pub const RTX_TEACACHE_RETAIN_STEPS: usize = 5;
    pub const RTX_TEACACHE_COOLDOWN_STEPS: usize = 1;
    pub const RTX_TEACACHE_COEFFICIENTS: [f64; 2] = [1.0, 0.0];
    #[derive(Debug, Clone, Copy, PartialEq)]
    pub struct H3TeaDecision {
        pub compute: bool,
        pub reason: &'static str,
        pub relative_l1: Option<f64>,
        pub indicator: Option<f64>,
        pub accumulator: f64,
    }

    /// RTX TeaCache from `models/minimax_h3/RTX4090/teacache.py`.
    ///
    /// The signal is block 0's AdaLN-modulated RMS-norm hidden. The reuse
    /// payload is the residual across the 50-block stack (`hidden - input`).
    /// H3 is guidance-distilled: one accumulator, no CFG pair, no Wan
    /// timestep-projection.
    #[derive(Debug, Clone)]
    pub struct H3TeaCache {
        pub threshold: f64,
        pub retain_steps: usize,
        pub cooldown_steps: usize,
        pub num_forwards: usize,
        pub coefficients: Vec<f64>,
        has_signal: bool,
        has_residual: bool,
        acc: f64,
    }

    impl H3TeaCache {
        pub fn official(num_forwards: usize) -> Result<Self, String> {
            Self::new(
                RTX_TEACACHE_THRESHOLD,
                RTX_TEACACHE_RETAIN_STEPS,
                RTX_TEACACHE_COOLDOWN_STEPS,
                num_forwards,
                RTX_TEACACHE_COEFFICIENTS.to_vec(),
            )
        }

        pub fn new(
            threshold: f64,
            retain_steps: usize,
            cooldown_steps: usize,
            num_forwards: usize,
            coefficients: Vec<f64>,
        ) -> Result<Self, String> {
            if threshold <= 0.0 {
                return Err("h3 teacache requires a positive threshold".into());
            }
            if coefficients.is_empty() {
                return Err("h3 teacache requires at least one coefficient".into());
            }
            Ok(Self {
                threshold,
                retain_steps,
                cooldown_steps,
                num_forwards,
                coefficients,
                has_signal: false,
                has_residual: false,
                acc: 0.0,
            })
        }

        pub fn needs_signal(&self, step: usize) -> bool {
            self.force_reason(step).is_none()
        }

        /// `relative_l1` is sum |signal − previous| / sum |previous|.
        /// Ignored on a forced step.
        pub fn decide(&mut self, step: usize, relative_l1: f64) -> H3TeaDecision {
            let (compute, reason, relative_l1, indicator) =
                if let Some(reason) = self.force_reason(step) {
                    if reason == "warmup" || reason == "cooldown" {
                        self.acc = 0.0;
                    }
                    (true, reason, None, None)
                } else {
                    let indicator = poly(&self.coefficients, relative_l1);
                    self.acc += indicator;
                    let compute = self.acc >= self.threshold;
                    if compute {
                        self.acc = 0.0;
                    }
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
            self.has_signal = true;
            H3TeaDecision {
                compute,
                reason,
                relative_l1,
                indicator,
                accumulator: self.acc,
            }
        }

        pub fn note_computed(&mut self) {
            self.acc = 0.0;
            self.has_residual = true;
        }

        /// Forget the signal, the residual and the accumulator: the next
        /// request starts as the first one did (its buffers were freed).
        pub fn reset(&mut self) {
            self.acc = 0.0;
            self.has_signal = false;
            self.has_residual = false;
        }

        fn force_reason(&self, step: usize) -> Option<&'static str> {
            if step < self.retain_steps {
                Some("warmup")
            } else if step >= self.num_forwards.saturating_sub(self.cooldown_steps) {
                Some("cooldown")
            } else if !self.has_signal || !self.has_residual {
                Some("initialize")
            } else {
                None
            }
        }
    }

    /// Sum |current − previous| / sum |previous|. Matches RTX `teacache.py`.
    pub fn relative_l1(current: &[f32], previous: &[f32]) -> f64 {
        let mut num = 0.0;
        let mut den = 0.0;
        for (a, b) in current.iter().zip(previous.iter()) {
            num += (f64::from(*a) - f64::from(*b)).abs();
            den += f64::from(*b).abs();
        }
        relative_l1_from_sums(num, den)
    }

    /// `sum|current - previous| / clamp_min(sum|previous|, 1e-8)` from the two
    /// sums (the device reduction returns exactly these).
    pub fn relative_l1_from_sums(abs_diff_sum: f64, abs_prev_sum: f64) -> f64 {
        abs_diff_sum / abs_prev_sum.max(1.0e-8)
    }

    fn poly(coefficients: &[f64], x: f64) -> f64 {
        let mut value = 0.0;
        for coefficient in coefficients {
            value = value * x + coefficient;
        }
        value
    }
}

/// `crates/fastvideo-models/src/cosmos/sol.rs:53-137`.
pub mod cosmos_tea {
    pub const TEACACHE_THRESHOLD: f64 = 1.15;
    pub const TEACACHE_START_STEP: usize = 10;
    pub const TEACACHE_MAX_CONSECUTIVE: usize = 3;
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
}

/// `crates/fastvideo-models/src/ltx2/fbcache.rs:55-190`.
pub mod ltx_fb {
    pub const THRESHOLD: f64 = 0.08;
    pub const WARMUP: usize = 1;
    pub const MAX_CONSECUTIVE: usize = 10;
    #[derive(Debug, Clone, Copy, PartialEq)]
    pub struct FbDecision {
        pub skip: bool,
        pub reason: &'static str,
        pub distance: Option<f64>,
        pub accumulator: f64,
    }

    #[derive(Debug, Clone)]
    struct PassState {
        has_signal: bool,
        has_residual: bool,
        acc: f64,
        consecutive: usize,
    }

    impl Default for PassState {
        fn default() -> Self {
            Self {
                has_signal: false,
                has_residual: false,
                acc: 0.0,
                consecutive: 0,
            }
        }
    }

    /// Stage-1 First-Block Cache. State is keyed by CFG pass ordinal.
    #[derive(Debug, Clone)]
    pub struct FbCache {
        pub threshold: f64,
        pub warmup: usize,
        pub max_consecutive: usize,
        step_index: usize,
        pass: usize,
        states: Vec<PassState>,
    }

    impl FbCache {
        pub fn official() -> Self {
            Self {
                threshold: THRESHOLD,
                warmup: WARMUP,
                max_consecutive: MAX_CONSECUTIVE,
                step_index: 0,
                pass: 0,
                states: Vec::new(),
            }
        }

        pub fn begin_step(&mut self, step_index: usize) {
            self.step_index = step_index;
            self.pass = 0;
        }

        pub fn needs_signal(&self) -> bool {
            let state = self.states.get(self.pass);
            match state {
                Some(state) => {
                    self.step_index >= self.warmup
                        && state.has_signal
                        && state.has_residual
                        && state.consecutive < self.max_consecutive
                }
                None => false,
            }
        }

        /// `distance` is mean |signal − previous| / mean |previous|.
        /// Ignored when the pass must compute.
        pub fn decide(&mut self, distance: f64) -> FbDecision {
            while self.states.len() <= self.pass {
                self.states.push(PassState::default());
            }
            let p = self.pass;
            self.pass += 1;
            let state = &mut self.states[p];
            let (skip, reason, distance) = if self.step_index < self.warmup {
                (false, "warmup", None)
            } else if !state.has_signal || !state.has_residual {
                (false, "initialize", None)
            } else if state.consecutive >= self.max_consecutive {
                (false, "max_consecutive", None)
            } else {
                state.acc += distance;
                if state.acc < self.threshold {
                    (true, "below_threshold", Some(distance))
                } else {
                    (false, "threshold", Some(distance))
                }
            };
            state.has_signal = true;
            FbDecision {
                skip,
                reason,
                distance,
                accumulator: state.acc,
            }
        }

        pub fn note_computed(&mut self, pass: usize) {
            let state = &mut self.states[pass];
            state.acc = 0.0;
            state.consecutive = 0;
            state.has_residual = true;
        }

        pub fn note_reused(&mut self, pass: usize) {
            self.states[pass].consecutive += 1;
        }

        pub fn current_pass(&self) -> usize {
            self.pass
        }

        pub fn last_pass(&self) -> usize {
            self.pass.saturating_sub(1)
        }
    }

    /// Mean |current − previous| / mean |previous|. `inf` when previous is zero.
    pub fn relative_l1(current: &[f32], previous: &[f32]) -> f64 {
        let n = current.len().max(1) as f64;
        let mut num = 0.0;
        let mut den = 0.0;
        for (a, b) in current.iter().zip(previous.iter()) {
            num += (f64::from(*a) - f64::from(*b)).abs();
            den += f64::from(*b).abs();
        }
        let den = den / n;
        if den == 0.0 {
            f64::INFINITY
        } else {
            (num / n) / den
        }
    }
}

/// `crates/fastvideo-models/src/lingbot/sol.rs:110-208`.
pub mod lingbot_easy {
    /// EasyCache knobs of one stage.
    #[derive(Debug, Clone, Copy, PartialEq)]
    pub struct EasyCacheConfig {
        pub threshold: f64,
        pub head_steps: usize,
        pub tail_steps: usize,
        pub max_reuse: usize,
    }

    impl EasyCacheConfig {
        /// `LINGBOT_EASYCACHE_*` of the published winner (base, ≥ 20 steps).
        pub const BASE: Self = Self {
            threshold: 0.08,
            head_steps: 4,
            tail_steps: 2,
            max_reuse: 2,
        };
        /// `LINGBOT_EASYCACHE_REFINER_*`.
        pub const REFINER: Self = Self {
            threshold: 0.25,
            head_steps: 2,
            tail_steps: 1,
            max_reuse: 2,
        };
        /// `LINGBOT_EASYCACHE_MIN_STEPS`: schedules at least this long use [`Self::BASE`].
        pub const MIN_BASE_STEPS: usize = 20;

        pub fn for_schedule(num_steps: usize) -> Self {
            if num_steps >= Self::MIN_BASE_STEPS {
                Self::BASE
            } else {
                Self::REFINER
            }
        }
    }

    /// Step-reuse controller. The caller asks [`Self::reuse`] before a step's
    /// transformer pass; on `false` it computes and calls [`Self::computed`].
    #[derive(Debug, Clone)]
    pub struct EasyCache {
        pub cfg: EasyCacheConfig,
        num_steps: usize,
        /// Latent at the last computed step (`_ec["ref"]`).
        reference: Option<Vec<f32>>,
        run: usize,
        pub computed_steps: usize,
        pub reused_steps: usize,
    }

    impl EasyCache {
        pub fn new(cfg: EasyCacheConfig, num_steps: usize) -> Self {
            Self {
                cfg,
                num_steps,
                reference: None,
                run: 0,
                computed_steps: 0,
                reused_steps: 0,
            }
        }

        /// `mean|x - ref| / (mean|ref| + 1e-8)`.
        pub fn relative_l1(latent: &[f32], reference: &[f32]) -> f64 {
            let n = latent.len().max(1) as f64;
            let diff: f64 = latent
                .iter()
                .zip(reference)
                .map(|(&a, &b)| f64::from((a - b).abs()))
                .sum::<f64>()
                / n;
            let base: f64 = reference.iter().map(|&b| f64::from(b.abs())).sum::<f64>() / n;
            diff / (base + 1e-8)
        }

        /// Whether step `i` may reuse the cached outputs for `latent`.
        pub fn reuse(&mut self, i: usize, latent: &[f32]) -> bool {
            let Some(reference) = self.reference.as_deref() else {
                return false;
            };
            let window =
                self.cfg.head_steps <= i && i < self.num_steps.saturating_sub(self.cfg.tail_steps);
            if !window || self.run >= self.cfg.max_reuse {
                return false;
            }
            let ok = Self::relative_l1(latent, reference) < self.cfg.threshold;
            if ok {
                self.run += 1;
                self.reused_steps += 1;
            }
            ok
        }

        /// Record a computed step: its input latent becomes the reference.
        pub fn computed(&mut self, latent: &[f32]) {
            self.reference = Some(latent.to_vec());
            self.run = 0;
            self.computed_steps += 1;
        }
    }
}

/// `crates/fastvideo-cudarc/src/wan/pipeline.rs:1571-1671` (`TeaCache`,
/// `FASTVIDEO_TEACACHE`). Not verbatim: a hand reduction to the decision,
/// its tensors replaced by a flag (the arithmetic lines are as written):
/// `maybe_reuse(rel)` returns `true` for a reuse, `rel` being the
/// `num / (den + 1e-6)` the runtime reduces; `store` records a computed step.
pub mod wan_teacache4wan21 {
    pub const COEFFICIENTS: [f64; 5] = [
        2.39676752e3,
        -1.31110545e3,
        2.01331979e2,
        -8.29855975,
        1.37887774e-1,
    ];
    pub const RET_COEFFICIENTS: [f64; 5] = [
        -5.21862437e4,
        9.23041404e3,
        -5.28275948e2,
        1.36987616e1,
        -4.99875664e-2,
    ];

    pub struct TeaCache {
        pub thresh: f32,
        pub accumulated: f32,
        pub step: usize,
        pub coefficients: [f64; 5],
        pub has_prev: bool,
    }

    impl TeaCache {
        pub fn new(thresh: f32, coefficients: [f64; 5]) -> Self {
            Self {
                thresh,
                accumulated: 0.0,
                step: 0,
                coefficients,
                has_prev: false,
            }
        }

        fn poly_rescale(&self, x: f64) -> f64 {
            let c = &self.coefficients;
            ((((c[0] * x + c[1]) * x + c[2]) * x + c[3]) * x) + c[4]
        }

        pub fn maybe_reuse(&mut self, rel: f64) -> bool {
            if !self.has_prev {
                return false;
            }
            self.accumulated += self.poly_rescale(rel) as f32;
            self.step += 1;
            // Always compute first / last-ish steps; skip when accumulated below thresh.
            if self.step <= 1 {
                return false;
            }
            if self.accumulated < self.thresh {
                return true;
            }
            self.accumulated = 0.0;
            false
        }

        pub fn store(&mut self) {
            self.has_prev = true;
        }
    }
}
