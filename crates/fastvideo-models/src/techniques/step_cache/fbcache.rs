//! First-Block Cache: TeaCache's accumulate-and-threshold rule on the
//! residual of block 0 (`hidden − input`), with one lane per CFG pass of a
//! step (LTX-2.5 GB200 `step_cache.py`).
//!
//! The family runs block 0 every forward, measures the relative L1 of its
//! residual against the previous one of the same pass, and on a skip adds
//! the stored whole-stack residuals. The pass ordinal is counted here:
//! [`FbCache::begin_step`] resets it, every [`FbCache::decide_next`]
//! advances it.

use super::{Accumulate, ResetOn};
use super::{
    Algorithm, Cap, Comparison, Cooldown, Decision, Lane, Refresh, Rescale, Signal, StepCache,
    StepCacheError, TeaCache, TeaCacheParams, Window,
};

/// FBCache knobs (`fullopt.toml`, `LTX25_CACHE_*`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FbCacheParams {
    pub name: &'static str,
    pub threshold: f64,
    /// `step_index < warmup` computes.
    pub warmup: usize,
    /// Consecutive skips per pass before a forced compute.
    pub max_consecutive: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FbCache {
    rule: TeaCache,
    step_index: usize,
    pass: usize,
}

impl FbCache {
    /// The rule is TeaCache's with: no rescale, `reuse = acc < threshold`,
    /// the accumulator cleared by `note_computed` only, no cool-down, and
    /// `consecutive >= max_consecutive` forcing a compute (no zero guard).
    pub fn new(p: FbCacheParams) -> Self {
        Self {
            rule: TeaCache::new(TeaCacheParams {
                name: p.name,
                threshold: p.threshold,
                window: Window {
                    warmup: p.warmup,
                    cooldown: Cooldown::None,
                },
                rescale: Rescale::Identity,
                comparison: Comparison::ReuseBelow,
                accumulate: Accumulate::F64,
                reset: ResetOn::Computed,
                require_signal: true,
                periodic: None,
                hit_cap: Some(Cap {
                    limit: p.max_consecutive,
                    reason: "max_consecutive",
                }),
            }),
            step_index: 0,
            pass: 0,
        }
    }

    pub fn params(&self) -> &TeaCacheParams {
        self.rule.params()
    }

    /// Start step `step_index`: the next decision is pass 0.
    pub fn begin_step(&mut self, step_index: usize) {
        self.step_index = step_index;
        self.pass = 0;
    }

    /// Whether the next pass's decision reads the signal.
    pub fn needs_signal_next(&self) -> bool {
        self.rule.needs_signal(Lane(self.pass), self.step_index)
    }

    /// Decide the next pass of the current step and advance the ordinal.
    pub fn decide_next(&mut self, signal: Signal) -> Result<Decision, StepCacheError> {
        let lane = Lane(self.pass);
        self.pass += 1;
        self.rule.decide(lane, self.step_index, signal)
    }

    /// The pass the next decision belongs to.
    pub fn current_pass(&self) -> usize {
        self.pass
    }

    /// The pass of the last decision.
    pub fn last_pass(&self) -> usize {
        self.pass.saturating_sub(1)
    }
}

impl StepCache for FbCache {
    fn name(&self) -> &'static str {
        self.rule.name()
    }

    fn algorithm(&self) -> Algorithm {
        Algorithm::FbCache
    }

    fn needs_signal(&self, lane: Lane, step: usize) -> bool {
        self.rule.needs_signal(lane, step)
    }

    fn decide(
        &mut self,
        lane: Lane,
        step: usize,
        signal: Signal,
    ) -> Result<Decision, StepCacheError> {
        self.step_index = step;
        self.pass = lane.0 + 1;
        self.rule.decide(lane, step, signal)
    }

    fn note_computed(&mut self, lane: Lane, refresh: Refresh) -> Result<(), StepCacheError> {
        self.rule.note_computed(lane, refresh)
    }

    fn note_reused(&mut self, lane: Lane) {
        self.rule.note_reused(lane);
    }

    fn reset(&mut self) {
        self.rule.reset();
        self.step_index = 0;
        self.pass = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn official() -> FbCache {
        FbCache::new(FbCacheParams {
            name: "ltx2 fbcache",
            threshold: 0.08,
            warmup: 1,
            max_consecutive: 10,
        })
    }

    /// The decision sequence of two CFG passes on a recorded signal.
    #[test]
    fn decision_sequence_on_a_recorded_signal() {
        let mut c = official();
        let signal = [
            [0.0, 0.0],
            [0.03125, 0.09375],
            [0.0625, 0.015625],
            [0.015625, 0.03125],
        ];
        let mut seen = Vec::new();
        for (step, row) in signal.iter().enumerate() {
            c.begin_step(step);
            for &x in row {
                let pass = c.current_pass();
                let s = if c.needs_signal_next() {
                    Signal::Distance(x)
                } else {
                    Signal::None
                };
                let d = c.decide_next(s).unwrap();
                assert_eq!(c.last_pass(), pass);
                if d.reuse() {
                    c.note_reused(Lane(pass));
                } else {
                    c.note_computed(Lane(pass), Refresh::None).unwrap();
                }
                seen.push((d.reason, d.accumulator));
            }
        }
        assert_eq!(
            seen,
            [
                ("warmup", 0.0),
                ("warmup", 0.0),
                ("below_threshold", 0.03125),
                ("threshold", 0.09375),
                ("threshold", 0.09375),
                ("below_threshold", 0.015625),
                ("below_threshold", 0.015625),
                ("below_threshold", 0.046875),
            ]
        );
    }

    #[test]
    fn max_consecutive_forces_a_recompute() {
        let mut c = FbCache::new(FbCacheParams {
            max_consecutive: 1,
            ..FbCacheParams {
                name: "ltx2 fbcache",
                threshold: 0.08,
                warmup: 1,
                max_consecutive: 10,
            }
        });
        c.begin_step(0);
        c.decide_next(Signal::None).unwrap();
        c.note_computed(Lane(0), Refresh::None).unwrap();
        c.begin_step(1);
        assert!(c.decide_next(Signal::Distance(0.01)).unwrap().reuse());
        c.note_reused(Lane(0));
        c.begin_step(2);
        assert_eq!(
            c.decide_next(Signal::None).unwrap().reason,
            "max_consecutive"
        );
    }
}
