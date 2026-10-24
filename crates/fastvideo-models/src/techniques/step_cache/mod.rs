//! One step-cache interface and one implementation per algorithm
//! (ADR-0020, plan B4a).
//!
//! A step cache decides, per denoise step (or per forward) and per *lane*
//! (a CFG branch, a CFG pass ordinal), whether the expensive
//! part of the transformer runs (`Compute`), is replaced by a stored residual
//! (`Reuse`), or is forecast from a derivative history (`Extrapolate`).
//!
//! The controllers here hold only scalars: flags, accumulators, counters and
//! the per-lane `k` of EasyCache. The tensors a family keeps between forwards
//! (the previous signal, the residual, Taylor's factors) stay in the family's
//! runtime, which measures the signal ([`Signal`]) and hands it in. The four
//! algorithms are:
//!
//! | type | algorithm | families |
//! |---|---|---|
//! | [`EasyCache`] | drift-estimate accumulate (Wan / Sana, Wan A14B) or drift-since-compute (LingBot) | Wan 2.x, SANA-Video, LingBot |
//! | [`TeaCache`] | accumulate a (rescaled) relative L1 of a step signal | H3, Wan (Sol and TeaCache4Wan2.1), Cosmos, Cosmos3 |
//! | [`FbCache`] | TeaCache's rule on the block-0 residual, one lane per CFG pass | LTX-2.5 stage 1 |
//! | [`Taylor`] | TaylorSeer lite: forecast on a fixed schedule | Wan |
//!
//! Where two families' copies differed in arithmetic before B4a, both
//! behaviours are explicit parameters ([`Comparison`], [`teacache::ResetOn`],
//! [`teacache::Accumulate`], [`Rescale`], [`easycache::Estimator`],
//! [`Cooldown`], [`Cap`]); see ADR-0020 for the contract per family.

use std::cmp::Ordering;

pub mod easycache;
pub mod fbcache;
pub mod taylor;
pub mod teacache;

#[cfg(test)]
mod equality;
#[cfg(test)]
mod legacy;

pub use easycache::{EasyCache, EasyCacheParams, EasyLabels, Estimator};
pub use fbcache::FbCache;
pub use taylor::{Taylor, TaylorParams};
pub use teacache::{Accumulate, ResetOn, TeaCache, TeaCacheParams};

/// A typed step-cache error. `cache` is the family's name for the cache
/// (`"h3 teacache"`, `"wan easycache"`, ...), so the messages are the ones
/// the per-family controllers printed before B4a.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum StepCacheError {
    #[error("{cache} requires a positive threshold")]
    NonPositiveThreshold { cache: &'static str },
    #[error("{cache} requires at least one step")]
    NoSteps { cache: &'static str },
    #[error("{cache} requires at least one coefficient")]
    NoCoefficients { cache: &'static str },
    #[error("{cache} requires a positive interval")]
    ZeroInterval { cache: &'static str },
    #[error("{cache} delta step cannot be zero")]
    ZeroDelta { cache: &'static str },
    #[error("{cache} factor length changed")]
    FactorLength { cache: &'static str },
    #[error("{cache} has no factors")]
    NoFactors { cache: &'static str },
    #[error("{cache} forecast before the first proj_out")]
    ForecastBeforeUpdate { cache: &'static str },
    #[error("{cache}: a computed step needs a {expected} refresh")]
    WrongRefresh {
        cache: &'static str,
        expected: &'static str,
    },
    #[error("{cache}: step {step} needs a {expected} signal")]
    WrongSignal {
        cache: &'static str,
        step: usize,
        expected: &'static str,
    },
}

/// The four algorithms.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Algorithm {
    EasyCache,
    TeaCache,
    FbCache,
    Taylor,
}

/// Which independent state a decision belongs to: a CFG branch (Wan Sol
/// TeaCache), a CFG pass ordinal (LTX FBCache), or the only lane of a
/// single-branch family. A controller can also key lanes by MoE expert
/// (`wan::sol_cache::A14bExpert::lane`); the Wan A14B runtime instead keeps
/// one controller per expert's transformer and uses the high-noise lane in
/// both. Lanes are created on first use.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Lane(pub usize);

impl Lane {
    /// The lane of a family with one decision per step (H3, Cosmos, the
    /// cond branch that decides for both).
    pub const ONLY: Lane = Lane(0);
    /// The conditional CFG branch.
    pub const COND: Lane = Lane(0);
    /// The unconditional CFG branch.
    pub const UNCOND: Lane = Lane(1);
}

/// What the step does with the expensive part of the forward.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Action {
    /// Run it (and refresh the stored residual / factors).
    Compute,
    /// Skip it and add the stored residual.
    Reuse,
    /// Skip it and forecast its output from the derivative history.
    Extrapolate,
}

/// The measurement a family hands to [`StepCache::decide`]. Ignored on a
/// step the cache forces to compute ([`StepCache::needs_signal`] false).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Signal {
    /// Nothing measured (a forced step, or Taylor, which needs none).
    None,
    /// One scalar: the relative L1 of the step signal against the previous
    /// one (TeaCache, FbCache), of the latent against the last computed one
    /// (LingBot EasyCache), or the mean input change (Wan A14B EasyCache).
    Distance(f64),
    /// Wan / Sana EasyCache: mean |x − previous step input| and mean |last
    /// full output|, both measured at the decision.
    Drift { input_change: f64, output_norm: f64 },
}

impl Signal {
    fn distance(self, cache: &'static str, step: usize) -> Result<f64, StepCacheError> {
        match self {
            Signal::Distance(d) => Ok(d),
            _ => Err(StepCacheError::WrongSignal {
                cache,
                step,
                expected: "distance",
            }),
        }
    }
}

/// What a computed step hands back to [`StepCache::note_computed`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Refresh {
    /// Nothing to refresh (TeaCache, FbCache, Taylor, LingBot EasyCache).
    None,
    /// EasyCache's `k` refresh: mean |x − last full input| and mean |output −
    /// last full output|; A14B also stores the new output's mean |·|.
    Drift {
        full_input_change: f64,
        output_change: f64,
        output_norm: Option<f64>,
    },
}

/// One decision. `distance` is the signal the cache read (`None` on a forced
/// step), `indicator` what it added to the accumulator (the rescaled
/// distance, EasyCache's estimate), `accumulator` the accumulator after the
/// decision, `k` EasyCache's slope when it has one.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Decision {
    pub action: Action,
    pub reason: &'static str,
    pub step: usize,
    pub distance: Option<f64>,
    pub indicator: Option<f64>,
    pub accumulator: f64,
    pub k: Option<f64>,
}

impl Decision {
    pub fn compute(&self) -> bool {
        self.action == Action::Compute
    }

    pub fn reuse(&self) -> bool {
        self.action == Action::Reuse
    }
}

/// The one step-cache interface. A family's runtime calls, per forward:
/// [`Self::needs_signal`] (measure only when true), [`Self::decide`], then
/// [`Self::note_computed`] after a computed forward or [`Self::note_reused`]
/// after a reused one; [`Self::reset`] at the end of a denoise when the
/// buffers are freed.
pub trait StepCache {
    /// The family's name for this cache (log and error prefix).
    fn name(&self) -> &'static str;

    fn algorithm(&self) -> Algorithm;

    /// True when [`Self::decide`] at `step` will read the signal: the step is
    /// not forced, so the family must measure it.
    fn needs_signal(&self, lane: Lane, step: usize) -> bool;

    /// Decide `step` for `lane`. `step` is the family's step clock; a cache
    /// with its own clock ([`Taylor`]: one tick per forward) ignores it and
    /// reports its own in [`Decision::step`].
    fn decide(
        &mut self,
        lane: Lane,
        step: usize,
        signal: Signal,
    ) -> Result<Decision, StepCacheError>;

    /// The forward ran: the family stored a fresh residual (or factors).
    fn note_computed(&mut self, lane: Lane, refresh: Refresh) -> Result<(), StepCacheError>;

    /// The forward reused the stored residual.
    fn note_reused(&mut self, lane: Lane);

    /// Forget every lane: the next denoise starts as the first one did.
    fn reset(&mut self);
}

/// The dense head and tail of a schedule: steps in the head (`step <
/// warmup`) and the tail always compute.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Window {
    pub warmup: usize,
    pub cooldown: Cooldown,
}

/// Where the dense tail starts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Cooldown {
    /// No dense tail.
    None,
    /// The last `steps` of `of`: `step >= of.saturating_sub(steps)` (H3,
    /// Wan / Sana and A14B EasyCache, LingBot).
    Last { steps: usize, of: usize },
    /// `step >= start` (Wan Sol TeaCache's `end_step`).
    From(usize),
}

impl Cooldown {
    /// The configured tail length (`Last`), 0 otherwise.
    pub fn steps(&self) -> usize {
        match self {
            Cooldown::Last { steps, .. } => *steps,
            Cooldown::None | Cooldown::From(_) => 0,
        }
    }

    /// The first step of the tail, `usize::MAX` without one.
    pub fn start(&self) -> usize {
        match self {
            Cooldown::None => usize::MAX,
            Cooldown::Last { steps, of } => of.saturating_sub(*steps),
            Cooldown::From(start) => *start,
        }
    }
}

/// Which end of the window forced the step.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Edge {
    Head,
    Tail,
}

impl Window {
    fn edge(&self, step: usize) -> Option<Edge> {
        if step < self.warmup {
            Some(Edge::Head)
        } else if step >= self.cooldown.start() {
            Some(Edge::Tail)
        } else {
            None
        }
    }
}

/// A cap on consecutive reuses: at `limit` consecutive reuses the next step
/// computes. `Some(Cap { limit: 0, .. })` computes every step (Cosmos'
/// `hits >= max_consecutive`, LTX's `consecutive >= max_consecutive` and
/// LingBot's `run >= max_reuse` compare without a zero guard); families
/// whose knob reads `0` as "no cap" (Wan Sol `max_hits`, A14B `max_reuse`)
/// map `0` to `None`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Cap {
    pub limit: usize,
    pub reason: &'static str,
}

impl Cap {
    /// `None` for `0`, else a cap of `limit` (the `knob > 0 && count >= knob`
    /// form).
    pub fn nonzero(limit: usize, reason: &'static str) -> Option<Cap> {
        (limit > 0).then_some(Cap { limit, reason })
    }
}

/// How the accumulator is compared with the threshold. The two forms differ
/// only on NaN, which is why both stay: `acc >= threshold` reuses a NaN,
/// `!(acc < threshold)` computes it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Comparison {
    /// `compute = acc >= threshold` (H3, Wan Sol TeaCache, Cosmos, Wan /
    /// Sana EasyCache).
    ComputeAtOrAbove,
    /// `reuse = acc < threshold` (LTX FBCache, A14B EasyCache, LingBot,
    /// TeaCache4Wan2.1).
    ReuseBelow,
}

impl Comparison {
    fn compute_f64(self, acc: f64, threshold: f64) -> bool {
        match self {
            Comparison::ComputeAtOrAbove => acc >= threshold,
            // `!(acc < threshold)`: NaN computes.
            Comparison::ReuseBelow => acc.partial_cmp(&threshold) != Some(Ordering::Less),
        }
    }

    fn compute_f32(self, acc: f32, threshold: f32) -> bool {
        match self {
            Comparison::ComputeAtOrAbove => acc >= threshold,
            Comparison::ReuseBelow => acc.partial_cmp(&threshold) != Some(Ordering::Less),
        }
    }
}

/// Non-empty polynomial coefficients, highest degree first.
#[derive(Debug, Clone, PartialEq)]
pub struct Coefficients(Vec<f64>);

impl Coefficients {
    pub fn new(cache: &'static str, coefficients: Vec<f64>) -> Result<Self, StepCacheError> {
        if coefficients.is_empty() {
            return Err(StepCacheError::NoCoefficients { cache });
        }
        Ok(Self(coefficients))
    }

    pub fn as_slice(&self) -> &[f64] {
        &self.0
    }
}

/// How a TeaCache distance becomes the accumulator's indicator.
#[derive(Debug, Clone, PartialEq)]
pub enum Rescale {
    /// The distance itself (Cosmos, LTX FBCache).
    Identity,
    /// Horner from zero, `v = 0; for c: v = v * x + c` (H3 `teacache.py`,
    /// Wan Sol): `[1, 0]` gives `x` for finite `x`.
    Horner(Coefficients),
    /// Horner from the leading coefficient, `v = c0; for c in rest: v = v *
    /// x + c` (TeaCache4Wan2.1's written-out quartic). Differs from
    /// [`Rescale::Horner`] only where `0 * x` is NaN (`x` infinite).
    HornerLeading(Coefficients),
}

impl Rescale {
    pub fn apply(&self, x: f64) -> f64 {
        match self {
            Rescale::Identity => x,
            Rescale::Horner(c) => {
                let mut value = 0.0;
                for coefficient in c.as_slice() {
                    value = value * x + coefficient;
                }
                value
            }
            Rescale::HornerLeading(c) => match c.as_slice().split_first() {
                Some((first, rest)) => {
                    let mut value = *first;
                    for coefficient in rest {
                        value = value * x + coefficient;
                    }
                    value
                }
                // `Coefficients` is never empty.
                None => x,
            },
        }
    }

    /// The polynomial's coefficients (empty for [`Rescale::Identity`]).
    pub fn coefficients(&self) -> &[f64] {
        match self {
            Rescale::Identity => &[],
            Rescale::Horner(c) | Rescale::HornerLeading(c) => c.as_slice(),
        }
    }
}

/// Host relative-L1 reductions the families' CPU paths use. The device
/// paths reduce the same two sums on the GPU (`rt::ops::abs_diff_sums_device`,
/// `ltx_abs_diff_sums_device`) and finish with the same scalar formula.
pub mod distance {
    /// `(Σ|a − b|, Σ|b|)` in f64, the difference taken in f64.
    pub fn abs_sums(current: &[f32], previous: &[f32]) -> (f64, f64) {
        let mut num = 0.0;
        let mut den = 0.0;
        for (a, b) in current.iter().zip(previous.iter()) {
            num += (f64::from(*a) - f64::from(*b)).abs();
            den += f64::from(*b).abs();
        }
        (num, den)
    }

    /// H3 `teacache.py`: `Σ|cur − prev| / max(Σ|prev|, 1e-8)`.
    pub fn sum_over_sum_clamped(abs_diff_sum: f64, abs_prev_sum: f64) -> f64 {
        abs_diff_sum / abs_prev_sum.max(1.0e-8)
    }

    /// Cosmos / Wan Sol: `(Σ|d| / n) / max(Σ|prev| / n, 1e-8)`.
    pub fn mean_over_mean_clamped(abs_diff_sum: f64, abs_prev_sum: f64, n: usize) -> f64 {
        let n = n.max(1) as f64;
        (abs_diff_sum / n) / (abs_prev_sum / n).max(1e-8)
    }

    /// LTX FBCache host path: `(Σ|d| / n) / (Σ|prev| / n)`, `inf` when the
    /// previous mean is zero.
    pub fn mean_over_mean_or_inf(abs_diff_sum: f64, abs_prev_sum: f64, n: usize) -> f64 {
        let n = n.max(1) as f64;
        let den = abs_prev_sum / n;
        if den == 0.0 {
            f64::INFINITY
        } else {
            (abs_diff_sum / n) / den
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn comparisons_differ_only_on_nan() {
        for (acc, thr) in [(0.0, 0.1), (0.1, 0.1), (0.2, 0.1), (f64::INFINITY, 0.1)] {
            assert_eq!(
                Comparison::ComputeAtOrAbove.compute_f64(acc, thr),
                Comparison::ReuseBelow.compute_f64(acc, thr)
            );
        }
        assert!(!Comparison::ComputeAtOrAbove.compute_f64(f64::NAN, 0.1));
        assert!(Comparison::ReuseBelow.compute_f64(f64::NAN, 0.1));
        // A tie computes in both forms and both precisions.
        for c in [Comparison::ComputeAtOrAbove, Comparison::ReuseBelow] {
            assert!(c.compute_f64(0.125, 0.125));
            assert!(c.compute_f32(0.125, 0.125));
            assert!(!c.compute_f32(0.0625, 0.125));
            assert!(c.compute_f32(0.25, 0.125));
        }
        assert!(!Comparison::ComputeAtOrAbove.compute_f32(f32::NAN, 0.1));
        assert!(Comparison::ReuseBelow.compute_f32(f32::NAN, 0.1));
    }

    #[test]
    fn horner_forms_agree_on_finite_inputs_and_differ_on_infinity() {
        let c = || Coefficients::new("t", vec![2.0, -1.0, 0.5]).unwrap_or_else(|e| panic!("{e}"));
        let a = Rescale::Horner(c());
        let b = Rescale::HornerLeading(c());
        for x in [0.0, 0.25, 3.0, -1.5] {
            assert_eq!(a.apply(x).to_bits(), b.apply(x).to_bits(), "{x}");
        }
        assert!(a.apply(f64::INFINITY).is_nan());
        assert!(b.apply(f64::INFINITY).is_infinite());
        assert_eq!(Rescale::Identity.apply(0.3), 0.3);
        assert!(Coefficients::new("t", vec![]).is_err());
    }

    #[test]
    fn cooldown_forms() {
        assert_eq!(Cooldown::None.start(), usize::MAX);
        assert_eq!(Cooldown::Last { steps: 2, of: 10 }.start(), 8);
        assert_eq!(Cooldown::Last { steps: 20, of: 10 }.start(), 0);
        assert_eq!(Cooldown::From(7).start(), 7);
        assert_eq!(Cooldown::Last { steps: 2, of: 10 }.steps(), 2);
        assert_eq!(Cap::nonzero(0, "x"), None);
        assert_eq!(Cap::nonzero(3, "x").map(|c| c.limit), Some(3));
    }

    #[test]
    fn distances() {
        let (n, d) = distance::abs_sums(&[4.0, 2.0], &[2.0, 0.0]);
        assert_eq!((n, d), (4.0, 2.0));
        assert_eq!(distance::sum_over_sum_clamped(4.0, 2.0), 2.0);
        assert_eq!(distance::sum_over_sum_clamped(1.0, 0.0), 1.0e8);
        assert_eq!(distance::mean_over_mean_clamped(4.0, 2.0, 2), 2.0);
        assert!(distance::mean_over_mean_or_inf(1.0, 0.0, 1).is_infinite());
    }
}
