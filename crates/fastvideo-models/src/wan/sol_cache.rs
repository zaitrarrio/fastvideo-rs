//! Wan step-cache settings from `models/wan22_ti2v_5b` on NVlabs/Sana
//! `sol-engine` (`WAN22_CACHE_FAMILY`), mapped onto the one step-cache
//! implementation per algorithm (`techniques::step_cache`, ADR-0020).
//!
//! EasyCache decides from the raw latent and reuses `output - input`.
//! TeaCache decides from the timestep projection and reuses the residual
//! across the transformer blocks. Defaults match the published env.
//!
//! TaylorSeer lite forecasts `proj_out` and skips the block stack on the
//! forecast steps. The schedule is `wan_cache.py` `TaylorSeerRuntime`
//! (`WAN22_TAYLOR_*`). The factors are diffusers `TaylorSeerState`: divided
//! differences on compute steps, then `offset^k / k!` on a forecast.

use crate::techniques::step_cache::{
    Accumulate, Cap, Coefficients, Comparison, Cooldown, EasyCache, EasyCacheParams, EasyLabels,
    Estimator, Lane, Rescale, ResetOn, Signal, StepCache, StepCacheError, Taylor, TaylorParams,
    TeaCache, TeaCacheParams, Window,
};

/// Published Wan2.2 TaylorSeer lite defaults (`cache_runtime.py`).
pub const TAYLOR_INTERVAL: usize = 3;
pub const TAYLOR_WARMUP: usize = 3;
pub const TAYLOR_ORDER: usize = 1;

const TAYLOR_NAME: &str = "wan taylorseer";

/// TaylorSeer lite (`TaylorSeerCacheHook._measure_should_compute`).
pub fn taylor(
    interval: usize,
    warmup: usize,
    cooldown_start: usize,
    max_order: usize,
) -> Result<Taylor, StepCacheError> {
    Taylor::new(TaylorParams {
        name: TAYLOR_NAME,
        interval,
        warmup,
        cooldown_start,
        max_order,
    })
}

/// The published schedule: interval 3, warmup 3, cool-down the last 2, order 1.
pub fn taylor_official(num_steps: usize) -> Result<Taylor, StepCacheError> {
    taylor(
        TAYLOR_INTERVAL,
        TAYLOR_WARMUP,
        num_steps.saturating_sub(2),
        TAYLOR_ORDER,
    )
}

/// Delivered EasyCache knobs. [`easycache_official`] stays on
/// [`EasyCacheProfile::CodeDefault`] so existing tests do not silently pick
/// up a manifest.
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
        let raw = crate::settings::env_var("FASTVIDEO_WAN_EASYCACHE_PROFILE")
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

const EASY_NAME: &str = "wan easycache";

/// Whole-stack EasyCache (Wan 5B / 14B, SANA-Video): cond decides on
/// `k · mean|Δx| / max(mean|out_last|, 1e-8)` accumulated, `compute = acc >=
/// threshold`; steps `< retain` and the last `cooldown` compute.
pub fn easycache(
    num_steps: usize,
    threshold: f64,
    retain_steps: usize,
    cooldown_steps: usize,
) -> Result<EasyCache, StepCacheError> {
    if threshold <= 0.0 {
        return Err(StepCacheError::NonPositiveThreshold { cache: EASY_NAME });
    }
    if num_steps == 0 {
        return Err(StepCacheError::NoSteps { cache: EASY_NAME });
    }
    Ok(EasyCache::new(EasyCacheParams {
        name: EASY_NAME,
        threshold,
        window: Window {
            warmup: retain_steps,
            cooldown: Cooldown::Last {
                steps: cooldown_steps,
                of: num_steps,
            },
        },
        estimator: Estimator::KScaled,
        comparison: Comparison::ComputeAtOrAbove,
        reuse_cap: None,
        labels: EasyLabels::WAN,
    }))
}

pub fn easycache_official(num_steps: usize) -> Result<EasyCache, StepCacheError> {
    easycache_from_profile(num_steps, EasyCacheProfile::CodeDefault)
}

pub fn easycache_from_profile(
    num_steps: usize,
    profile: EasyCacheProfile,
) -> Result<EasyCache, StepCacheError> {
    let (threshold, retain, cooldown) = profile.params();
    easycache(num_steps, threshold, retain, cooldown)
}

const TEA_NAME: &str = "wan teacache";

/// Block-residual Sol TeaCache. The signal is the Wan timestep projection,
/// one lane per CFG branch ([`Lane::COND`], [`Lane::UNCOND`]).
/// `coefficients` rescale relative L1 with Horner's method; `[1, 0]` leaves
/// it unchanged. Steps `< start_step` and `>= end_step` compute; `max_hits`
/// and `periodic` of `0` are off. The accumulator clears when the branch
/// computes (`note_computed`), not on the decision.
pub fn sol_teacache(
    threshold: f64,
    start_step: usize,
    end_step: usize,
    max_hits: usize,
    periodic: usize,
    coefficients: Vec<f64>,
) -> Result<TeaCache, StepCacheError> {
    if threshold <= 0.0 {
        return Err(StepCacheError::NonPositiveThreshold { cache: TEA_NAME });
    }
    Ok(TeaCache::new(TeaCacheParams {
        name: TEA_NAME,
        threshold,
        window: Window {
            warmup: start_step,
            cooldown: Cooldown::From(end_step),
        },
        rescale: Rescale::Horner(Coefficients::new(TEA_NAME, coefficients)?),
        comparison: Comparison::ComputeAtOrAbove,
        accumulate: Accumulate::F64,
        reset: ResetOn::Computed,
        require_signal: true,
        periodic: (periodic > 0).then_some(periodic),
        hit_cap: Cap::nonzero(max_hits, "hit_cap"),
    }))
}

/// Threshold 0.12, warmup the first two steps, recompute the last two,
/// identity rescale.
pub fn sol_teacache_official(num_steps: usize) -> Result<TeaCache, StepCacheError> {
    sol_teacache(0.12, 2, num_steps.saturating_sub(2), 0, 0, vec![1.0, 0.0])
}

/// Wan2.1 T2V 1.3B TeaCache4Wan2.1 coefficients (`use_ret_steps=False`).
pub const TEACACHE4WAN21_COEFFICIENTS: [f64; 5] = [
    2.39676752e3,
    -1.31110545e3,
    2.01331979e2,
    -8.29855975,
    1.37887774e-1,
];

/// The `use_ret_steps=True` coefficients for 1.3B.
pub const TEACACHE4WAN21_RET_COEFFICIENTS: [f64; 5] = [
    -5.21862437e4,
    9.23041404e3,
    -5.28275948e2,
    1.36987616e1,
    -4.99875664e-2,
];

const TEA4WAN_NAME: &str = "wan teacache4wan21";

/// `FASTVIDEO_TEACACHE` (upstream TeaCache4Wan2.1): an f32 accumulator of
/// the written-out quartic of `rel` that integrates every decision, the
/// first one included, and clears only when it reaches `threshold`;
/// `reuse = acc < threshold`. The step is the runtime's count of decisions
/// with a signal (the first is warm-up). No threshold check (the env value
/// is used as given).
pub fn teacache4wan21(threshold: f32, ret_steps: bool) -> Result<TeaCache, StepCacheError> {
    let coefficients = if ret_steps {
        TEACACHE4WAN21_RET_COEFFICIENTS
    } else {
        TEACACHE4WAN21_COEFFICIENTS
    };
    Ok(TeaCache::new(TeaCacheParams {
        name: TEA4WAN_NAME,
        threshold: f64::from(threshold),
        window: Window {
            warmup: 1,
            cooldown: Cooldown::None,
        },
        rescale: Rescale::HornerLeading(Coefficients::new(TEA4WAN_NAME, coefficients.to_vec())?),
        comparison: Comparison::ReuseBelow,
        accumulate: Accumulate::F32,
        reset: ResetOn::Threshold,
        require_signal: false,
        periodic: None,
        hit_cap: None,
    }))
}

/// The `FASTVIDEO_TEACACHE` decision clock: the step is the number of
/// earlier decisions that had a signal (the runtime decides only once it
/// holds a previous input and output), so the first one is the warm-up.
#[derive(Debug, Clone)]
pub struct TeaCache4Wan21 {
    state: TeaCache,
    decisions: usize,
}

impl TeaCache4Wan21 {
    pub fn new(threshold: f32, ret_steps: bool) -> Result<Self, StepCacheError> {
        Ok(Self {
            state: teacache4wan21(threshold, ret_steps)?,
            decisions: 0,
        })
    }

    pub fn state(&self) -> &TeaCache {
        &self.state
    }

    /// `rel = Σ|x − x_prev| / (Σ|x_prev| + 1e-6)`; `true` reuses.
    pub fn reuse(&mut self, rel: f64) -> Result<bool, StepCacheError> {
        let step = self.decisions;
        self.decisions += 1;
        let decision = self.state.decide(Lane::ONLY, step, Signal::Distance(rel))?;
        Ok(decision.reuse())
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

/// The MoE expert a decision belongs to (one lane each).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum A14bExpert {
    HighNoise,
    LowNoise,
}

impl A14bExpert {
    pub fn lane(self) -> Lane {
        Lane(self as usize)
    }
}

const A14B_NAME: &str = "wan a14b cache";

/// Per-expert A14B EasyCache. Not the 5B whole-stack controller.
///
/// `models/wan22_t2v_a14b/optimized/cache_controller.py`: the estimate uses
/// the output norm stored at the last compute, `reuse = acc < threshold`,
/// every guard clears the accumulator, `max_reuse` of `0` is off.
pub fn a14b_easycache(
    num_steps: usize,
    threshold: f64,
    start_step: usize,
    tail_steps: usize,
    max_reuse: usize,
) -> Result<EasyCache, StepCacheError> {
    if threshold <= 0.0 {
        return Err(StepCacheError::NonPositiveThreshold { cache: A14B_NAME });
    }
    if num_steps == 0 {
        return Err(StepCacheError::NoSteps { cache: A14B_NAME });
    }
    Ok(EasyCache::new(EasyCacheParams {
        name: A14B_NAME,
        threshold,
        window: Window {
            warmup: start_step,
            cooldown: Cooldown::Last {
                steps: tail_steps,
                of: num_steps,
            },
        },
        estimator: Estimator::KScaledStoredNorm,
        comparison: Comparison::ReuseBelow,
        reuse_cap: Cap::nonzero(max_reuse, "max_reuse_guard"),
        labels: EasyLabels::A14B,
    }))
}

/// Threshold 0.30, start 5, tail 3, max_reuse 1.
pub fn a14b_easycache_official(num_steps: usize) -> Result<EasyCache, StepCacheError> {
    a14b_easycache(
        num_steps,
        A14B_EASY_THRESHOLD,
        A14B_START_STEP,
        A14B_TAIL_STEPS,
        A14B_MAX_REUSE,
    )
}

/// `(fresh prefix, cached tail, total)` blocks.
pub fn a14b_placement() -> (usize, usize, usize) {
    (
        A14B_FRESH_PREFIX_BLOCKS,
        A14B_CACHED_TAIL_BLOCKS,
        A14B_TOTAL_BLOCKS,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::techniques::step_cache::Refresh;

    fn drift(input_change: f64, output_norm: f64) -> Signal {
        Signal::Drift {
            input_change,
            output_norm,
        }
    }

    fn refresh(full_input_change: f64, output_change: f64) -> Refresh {
        Refresh::Drift {
            full_input_change,
            output_change,
            output_norm: None,
        }
    }

    #[test]
    fn warmup_and_cooldown_always_compute() {
        let mut cache = easycache_official(10).unwrap();
        for step in 0..7 {
            let d = cache.decide(Lane::COND, step, drift(99.0, 1.0)).unwrap();
            assert!(d.compute(), "{step}");
            assert_eq!(d.reason, "warmup");
            cache.note_computed(Lane::COND, refresh(1.0, 1.0)).unwrap();
        }
        let tail = cache.decide(Lane::COND, 9, drift(0.0, 1.0)).unwrap();
        assert!(tail.compute());
        assert_eq!(tail.reason, "cooldown");
    }

    #[test]
    fn rejects_a_non_positive_threshold() {
        assert_eq!(
            easycache(4, 0.0, 7, 1).unwrap_err().to_string(),
            "wan easycache requires a positive threshold"
        );
        assert!(sol_teacache(0.0, 2, 8, 0, 0, vec![1.0]).is_err());
        assert!(sol_teacache(0.1, 2, 8, 0, 0, vec![]).is_err());
        assert!(a14b_easycache(0, 0.3, 5, 3, 1).is_err());
        assert!(taylor(0, 3, 8, 1).is_err());
    }

    #[test]
    fn official_stays_on_the_code_default_profile() {
        let cache = easycache_official(50).unwrap();
        let p = cache.params();
        assert_eq!(
            (p.threshold, p.window.warmup, p.window.cooldown.steps()),
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
        let tuned = easycache_from_profile(40, EasyCacheProfile::Tuned14b).unwrap();
        assert_eq!(tuned.params().window.warmup, 5);
        assert!((tuned.params().threshold - 0.10).abs() < 1e-12);
    }

    #[test]
    fn sol_teacache_defaults() {
        let t = sol_teacache_official(10).unwrap();
        let p = t.params();
        assert_eq!(
            (p.threshold, p.window.warmup, p.window.cooldown.start()),
            (0.12, 2, 8)
        );
        assert_eq!(p.rescale.coefficients(), &[1.0, 0.0]);
        assert_eq!((p.periodic, p.hit_cap), (None, None));
        let t = teacache4wan21(0.08, false).unwrap();
        assert_eq!(t.params().threshold, f64::from(0.08f32));
        assert_eq!(
            t.params().rescale.coefficients(),
            &TEACACHE4WAN21_COEFFICIENTS
        );
    }

    #[test]
    fn a14b_placement_is_block0_plus_tail_39() {
        assert_eq!(a14b_placement(), (1, 39, 40));
        let cache = a14b_easycache_official(40).unwrap();
        let p = cache.params();
        assert!((p.threshold - 0.30).abs() < 1e-12);
        assert_eq!(p.window.warmup, 5);
        assert_eq!(p.window.cooldown.steps(), 3);
        assert_eq!(p.reuse_cap.map(|c| c.limit), Some(1));
        assert_ne!(p.threshold, EasyCacheProfile::Fullstack.params().0);
        assert_eq!(A14bExpert::LowNoise.lane(), Lane(1));
    }
}
