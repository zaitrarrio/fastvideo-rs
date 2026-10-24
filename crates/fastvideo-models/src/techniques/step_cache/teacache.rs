//! TeaCache: accumulate a (rescaled) relative L1 of a step signal and reuse
//! the stored residual while the accumulator stays under the threshold.
//!
//! One implementation for H3 (`teacache.py`, block-0 modulated input), Wan
//! Sol (timestep projection, per CFG branch), Cosmos / Cosmos3 (time embed,
//! cond decides) and TeaCache4Wan2.1 (`FASTVIDEO_TEACACHE`, the latents).
//! Their arithmetic differs in five places, each a parameter here:
//! [`ResetOn`], [`Accumulate`], [`Rescale`], [`Comparison`] and the forced
//! steps (window, `require_signal`, `periodic`, `hit_cap`). ADR-0020 lists
//! every family's values with the pre-B4a file:line they reproduce.

use super::{
    Action, Algorithm, Cap, Comparison, Decision, Edge, Lane, Refresh, Rescale, Signal, StepCache,
    StepCacheError, Window,
};

/// When the accumulator (and the reuse counters) clear.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResetOn {
    /// Only [`StepCache::note_computed`] clears them; a decision reports the
    /// accumulator as it stands (Wan Sol TeaCache, LTX FBCache).
    Computed,
    /// Every compute decision clears them, forced or not, so a threshold
    /// compute reports `0` (H3, Cosmos); `note_computed` clears them too.
    Decision,
    /// The accumulator integrates every signalled decision, forced ones
    /// included, and only a threshold compute clears it; `note_computed`
    /// leaves it (TeaCache4Wan2.1, `wan/pipeline.rs` `maybe_reuse`).
    Threshold,
}

/// The accumulator's precision.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Accumulate {
    F64,
    /// `acc: f32 += indicator as f32`, compared with an f32 threshold
    /// (TeaCache4Wan2.1).
    F32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TeaCacheParams {
    /// Log and error prefix.
    pub name: &'static str,
    pub threshold: f64,
    pub window: Window,
    pub rescale: Rescale,
    pub comparison: Comparison,
    pub accumulate: Accumulate,
    pub reset: ResetOn,
    /// Compute until the lane has a signal and a residual (`"initialize"`).
    pub require_signal: bool,
    /// Compute after this many reuses since the last compute (`"periodic"`).
    pub periodic: Option<usize>,
    /// Compute after this many consecutive reuses.
    pub hit_cap: Option<Cap>,
}

/// Why a step must compute (its reason string is what the family logs).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Forced {
    Warmup,
    Cooldown,
    Initialize,
    Periodic,
    HitCap(&'static str),
}

impl Forced {
    fn reason(self) -> &'static str {
        match self {
            Forced::Warmup => "warmup",
            Forced::Cooldown => "cooldown",
            Forced::Initialize => "initialize",
            Forced::Periodic => "periodic",
            Forced::HitCap(reason) => reason,
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq)]
struct LaneState {
    has_signal: bool,
    has_residual: bool,
    acc: f64,
    hits: usize,
    since: usize,
}

impl LaneState {
    fn clear_counters(&mut self) {
        self.acc = 0.0;
        self.hits = 0;
        self.since = 0;
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct TeaCache {
    params: TeaCacheParams,
    lanes: Vec<LaneState>,
}

impl TeaCache {
    pub fn new(params: TeaCacheParams) -> Self {
        Self {
            params,
            lanes: Vec::new(),
        }
    }

    pub fn params(&self) -> &TeaCacheParams {
        &self.params
    }

    /// The accumulator of `lane` (0 for a lane not seen yet).
    pub fn accumulator(&self, lane: Lane) -> f64 {
        self.lanes.get(lane.0).map_or(0.0, |s| s.acc)
    }

    fn lane(&self, lane: Lane) -> LaneState {
        self.lanes.get(lane.0).cloned().unwrap_or_default()
    }

    fn lane_mut(&mut self, lane: Lane) -> &mut LaneState {
        if self.lanes.len() <= lane.0 {
            self.lanes.resize(lane.0 + 1, LaneState::default());
        }
        &mut self.lanes[lane.0]
    }

    /// Why `step` must compute, if it must. Order: warm-up, cool-down,
    /// initialize, periodic, hit cap (Wan `sol_cache.rs:564`).
    fn forced(&self, lane: Lane, step: usize) -> Option<Forced> {
        let state = self.lane(lane);
        match self.params.window.edge(step) {
            Some(Edge::Head) => return Some(Forced::Warmup),
            Some(Edge::Tail) => return Some(Forced::Cooldown),
            None => {}
        }
        if self.params.require_signal && (!state.has_signal || !state.has_residual) {
            Some(Forced::Initialize)
        } else if self.params.periodic.is_some_and(|p| state.since >= p) {
            Some(Forced::Periodic)
        } else {
            match self.params.hit_cap {
                Some(cap) if state.hits >= cap.limit => Some(Forced::HitCap(cap.reason)),
                _ => None,
            }
        }
    }

    fn add(&self, acc: f64, indicator: f64) -> f64 {
        match self.params.accumulate {
            Accumulate::F64 => acc + indicator,
            Accumulate::F32 => f64::from(acc as f32 + indicator as f32),
        }
    }

    fn crosses(&self, acc: f64) -> bool {
        match self.params.accumulate {
            Accumulate::F64 => self
                .params
                .comparison
                .compute_f64(acc, self.params.threshold),
            Accumulate::F32 => self
                .params
                .comparison
                .compute_f32(acc as f32, self.params.threshold as f32),
        }
    }
}

impl StepCache for TeaCache {
    fn name(&self) -> &'static str {
        self.params.name
    }

    fn algorithm(&self) -> Algorithm {
        Algorithm::TeaCache
    }

    fn needs_signal(&self, lane: Lane, step: usize) -> bool {
        match self.params.reset {
            // Every decision but an uninitialized one integrates the signal.
            ResetOn::Threshold => self.forced(lane, step) != Some(Forced::Initialize),
            ResetOn::Computed | ResetOn::Decision => self.forced(lane, step).is_none(),
        }
    }

    fn decide(
        &mut self,
        lane: Lane,
        step: usize,
        signal: Signal,
    ) -> Result<Decision, StepCacheError> {
        let forced = self.forced(lane, step);
        let reset = self.params.reset;
        let (action, reason, distance, indicator) = match (reset, forced) {
            (ResetOn::Threshold, Some(why @ Forced::Initialize))
            | (ResetOn::Computed, Some(why)) => (Action::Compute, why.reason(), None, None),
            (ResetOn::Decision, Some(why)) => {
                self.lane_mut(lane).clear_counters();
                (Action::Compute, why.reason(), None, None)
            }
            (ResetOn::Threshold, Some(why)) => {
                let reason = why.reason();
                let x = signal.distance(self.params.name, step)?;
                let indicator = self.params.rescale.apply(x);
                let acc = self.add(self.lane(lane).acc, indicator);
                self.lane_mut(lane).acc = acc;
                (Action::Compute, reason, Some(x), Some(indicator))
            }
            (_, None) => {
                let x = signal.distance(self.params.name, step)?;
                let indicator = self.params.rescale.apply(x);
                let acc = self.add(self.lane(lane).acc, indicator);
                let compute = self.crosses(acc);
                let state = self.lane_mut(lane);
                state.acc = acc;
                if compute && reset != ResetOn::Computed {
                    // H3 / Cosmos clear the counters on the decision;
                    // TeaCache4Wan2.1 clears its accumulator only here.
                    state.clear_counters();
                }
                let (action, reason) = if compute {
                    (Action::Compute, "threshold")
                } else {
                    (Action::Reuse, "below_threshold")
                };
                (action, reason, Some(x), Some(indicator))
            }
        };
        let state = self.lane_mut(lane);
        state.has_signal = true;
        Ok(Decision {
            action,
            reason,
            step,
            distance,
            indicator,
            accumulator: state.acc,
            k: None,
        })
    }

    fn note_computed(&mut self, lane: Lane, _refresh: Refresh) -> Result<(), StepCacheError> {
        let reset = self.params.reset;
        let state = self.lane_mut(lane);
        if reset != ResetOn::Threshold {
            state.clear_counters();
        }
        state.has_residual = true;
        Ok(())
    }

    fn note_reused(&mut self, lane: Lane) {
        let state = self.lane_mut(lane);
        state.hits += 1;
        state.since += 1;
    }

    fn reset(&mut self) {
        self.lanes.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::super::{Coefficients, Cooldown};
    use super::*;

    fn h3_like(threshold: f64) -> TeaCache {
        TeaCache::new(TeaCacheParams {
            name: "t",
            threshold,
            window: Window {
                warmup: 2,
                cooldown: Cooldown::Last { steps: 1, of: 8 },
            },
            rescale: Rescale::Horner(Coefficients::new("t", vec![1.0, 0.0]).unwrap()),
            comparison: Comparison::ComputeAtOrAbove,
            accumulate: Accumulate::F64,
            reset: ResetOn::Decision,
            require_signal: true,
            periodic: None,
            hit_cap: None,
        })
    }

    /// The decision sequence on a recorded signal: warm-up, initialize is
    /// skipped by the warm-up's residual, reuse until 0.10 accumulates,
    /// cool-down at the last step.
    #[test]
    fn decision_sequence_on_a_recorded_signal() {
        let mut c = h3_like(0.10);
        let signal = [9.0, 9.0, 0.04, 0.05, 0.02, 0.2, 0.01, 0.01];
        let mut seen = Vec::new();
        for (step, &x) in signal.iter().enumerate() {
            let s = if c.needs_signal(Lane::ONLY, step) {
                Signal::Distance(x)
            } else {
                Signal::None
            };
            let d = c.decide(Lane::ONLY, step, s).unwrap();
            if d.compute() {
                c.note_computed(Lane::ONLY, Refresh::None).unwrap();
            } else {
                c.note_reused(Lane::ONLY);
            }
            seen.push(d.reason);
        }
        assert_eq!(
            seen,
            [
                "warmup",
                "warmup",
                "below_threshold",
                "below_threshold",
                "threshold",
                "threshold",
                "below_threshold",
                "cooldown"
            ]
        );
    }

    #[test]
    fn a_needed_signal_must_be_a_distance() {
        let mut c = h3_like(0.10);
        c.decide(Lane::ONLY, 0, Signal::None).unwrap();
        c.note_computed(Lane::ONLY, Refresh::None).unwrap();
        c.decide(Lane::ONLY, 1, Signal::None).unwrap();
        c.note_computed(Lane::ONLY, Refresh::None).unwrap();
        assert!(c.needs_signal(Lane::ONLY, 2));
        assert!(matches!(
            c.decide(Lane::ONLY, 2, Signal::None),
            Err(StepCacheError::WrongSignal { step: 2, .. })
        ));
    }

    #[test]
    fn reset_forgets_every_lane() {
        let mut c = h3_like(0.10);
        c.decide(Lane::UNCOND, 3, Signal::None).unwrap();
        c.note_computed(Lane::UNCOND, Refresh::None).unwrap();
        assert!(c.needs_signal(Lane::UNCOND, 3));
        c.reset();
        assert!(!c.needs_signal(Lane::UNCOND, 3));
    }
}
