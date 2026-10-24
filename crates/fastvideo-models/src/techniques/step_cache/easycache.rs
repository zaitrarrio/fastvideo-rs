//! EasyCache: reuse `output − input` while an estimate of the output's drift
//! stays under a threshold.
//!
//! One implementation for three pre-B4a copies, which differ in how the
//! estimate is formed ([`Estimator`]), in the comparison ([`Comparison`]),
//! in a consecutive-reuse cap ([`Cap`]) and in their reason names
//! ([`EasyLabels`]):
//!
//! * Wan 2.x 5B / 14B and SANA-Video (`wan/sol_cache.rs` `EasyCache`):
//!   `estimate = k · mean|x − x_prev| / max(mean|out_last|, 1e-8)`,
//!   accumulated; `k = mean|Δout| / max(mean|Δin|, 1e-8)` between the last
//!   two computed steps. Cond decides, uncond follows.
//! * Wan 2.2 A14B (the pre-B4a `A14bCacheController`): the same estimate,
//!   one controller per MoE expert's transformer, the output norm stored at
//!   the compute, `reuse = acc < threshold`, at most `max_reuse` reuses in
//!   a row.
//! * LingBot (`lingbot/sol.rs` `EasyCache`): the relative L1 of the latent
//!   against the last computed step's latent, compared directly (no
//!   accumulation), at most `max_reuse` in a row.

use super::{
    Action, Algorithm, Cap, Comparison, Decision, Edge, Lane, Refresh, Signal, StepCache,
    StepCacheError, Window,
};

/// How the per-step estimate is formed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Estimator {
    /// `k · input_change / max(output_norm, 1e-8)` from a
    /// [`Signal::Drift`] measured at the decision; accumulated. A compute
    /// decision clears the accumulator; `note_computed` refreshes `k` only
    /// (Wan / Sana).
    KScaled,
    /// `k · input_change / max(last_output_norm, 1e-8)` from a
    /// [`Signal::Distance`] (the input change) and the norm stored by the
    /// last `note_computed`; accumulated. Every forced step clears the
    /// accumulator; `note_computed` clears it and the reuse run (A14B).
    KScaledStoredNorm,
    /// The [`Signal::Distance`] itself, not accumulated (LingBot).
    Raw,
}

/// The reason strings a family logs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EasyLabels {
    pub warmup: &'static str,
    pub cooldown: &'static str,
    /// No residual yet.
    pub seed: &'static str,
    /// No previous input, `k` or stored norm yet.
    pub online_seed: &'static str,
    pub refresh: &'static str,
    pub below: &'static str,
}

impl EasyLabels {
    /// Wan / Sana / LingBot (`wan/sol_cache.rs:264-281`).
    pub const WAN: EasyLabels = EasyLabels {
        warmup: "warmup",
        cooldown: "cooldown",
        seed: "initialize",
        online_seed: "initialize",
        refresh: "threshold",
        below: "below_threshold",
    };
    /// A14B (`cache_controller.py`, `wan/sol_cache.rs:825-841`).
    pub const A14B: EasyLabels = EasyLabels {
        warmup: "warmup_guard",
        cooldown: "tail_guard",
        seed: "payload_seed",
        online_seed: "online_error_seed",
        refresh: "online_error_refresh",
        below: "online_error_below_threshold",
    };
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EasyCacheParams {
    pub name: &'static str,
    pub threshold: f64,
    pub window: Window,
    pub estimator: Estimator,
    pub comparison: Comparison,
    pub reuse_cap: Option<Cap>,
    pub labels: EasyLabels,
}

#[derive(Debug, Clone, PartialEq)]
struct LaneState {
    acc: f64,
    k: Option<f64>,
    has_previous_input: bool,
    has_payload: bool,
    last_output_norm: Option<f64>,
    consecutive: usize,
    last_compute: bool,
}

impl Default for LaneState {
    fn default() -> Self {
        Self {
            acc: 0.0,
            k: None,
            has_previous_input: false,
            has_payload: false,
            last_output_norm: None,
            consecutive: 0,
            last_compute: true,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct EasyCache {
    params: EasyCacheParams,
    lanes: Vec<LaneState>,
}

impl EasyCache {
    pub fn new(params: EasyCacheParams) -> Self {
        Self {
            params,
            lanes: Vec::new(),
        }
    }

    pub fn params(&self) -> &EasyCacheParams {
        &self.params
    }

    /// The slope `k` of `lane`, once two computed steps installed it.
    pub fn k(&self, lane: Lane) -> Option<f64> {
        self.lanes.get(lane.0).and_then(|s| s.k)
    }

    /// The uncond branch follows `lane`'s last decision; it still computes
    /// while it has no residual of its own (`uncond_compute`).
    pub fn uncond_compute(&self, lane: Lane, has_uncond_residual: bool) -> bool {
        self.lane(lane).last_compute || !has_uncond_residual
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

    /// Order: warm-up, cool-down, seed, reuse cap, online seed (A14B
    /// `force_reason`; Wan's `initialize` is seed or online seed, LingBot's
    /// guards return before measuring).
    fn force_reason(&self, lane: Lane, step: usize) -> Option<&'static str> {
        let labels = self.params.labels;
        match self.params.window.edge(step) {
            Some(Edge::Head) => return Some(labels.warmup),
            Some(Edge::Tail) => return Some(labels.cooldown),
            None => {}
        }
        let state = self.lane(lane);
        if !state.has_payload {
            return Some(labels.seed);
        }
        if let Some(cap) = self.params.reuse_cap {
            if state.consecutive >= cap.limit {
                return Some(cap.reason);
            }
        }
        let online_ready = match self.params.estimator {
            Estimator::KScaled => state.has_previous_input && state.k.is_some(),
            Estimator::KScaledStoredNorm => {
                state.has_previous_input && state.k.is_some() && state.last_output_norm.is_some()
            }
            Estimator::Raw => true,
        };
        (!online_ready).then_some(labels.online_seed)
    }

    fn estimate(&self, lane: Lane, step: usize, signal: Signal) -> Result<f64, StepCacheError> {
        let state = self.lane(lane);
        match self.params.estimator {
            Estimator::KScaled => match signal {
                Signal::Drift {
                    input_change,
                    output_norm,
                } => {
                    let output_norm = output_norm.max(1e-8);
                    Ok(state.k.unwrap_or(0.0) * input_change / output_norm)
                }
                _ => Err(StepCacheError::WrongSignal {
                    cache: self.params.name,
                    step,
                    expected: "drift",
                }),
            },
            Estimator::KScaledStoredNorm => {
                let input_change = signal.distance(self.params.name, step)?;
                Ok(state.k.unwrap_or(0.0) * input_change
                    / state.last_output_norm.unwrap_or(1.0).max(1e-8))
            }
            Estimator::Raw => signal.distance(self.params.name, step),
        }
    }
}

impl StepCache for EasyCache {
    fn name(&self) -> &'static str {
        self.params.name
    }

    fn algorithm(&self) -> Algorithm {
        Algorithm::EasyCache
    }

    fn needs_signal(&self, lane: Lane, step: usize) -> bool {
        self.force_reason(lane, step).is_none()
    }

    fn decide(
        &mut self,
        lane: Lane,
        step: usize,
        signal: Signal,
    ) -> Result<Decision, StepCacheError> {
        let estimator = self.params.estimator;
        let (compute, reason, distance, estimate) = match self.force_reason(lane, step) {
            Some(reason) => {
                // Wan clears on warm-up and cool-down; its `initialize` only
                // happens before the first accumulation, so clearing there
                // too changes nothing. A14B clears on every guard.
                if estimator != Estimator::Raw {
                    self.lane_mut(lane).acc = 0.0;
                }
                (true, reason, None, None)
            }
            None => {
                let estimate = self.estimate(lane, step, signal)?;
                let distance = match signal {
                    Signal::Distance(d) => Some(d),
                    Signal::Drift { input_change, .. } => Some(input_change),
                    Signal::None => None,
                };
                let threshold = self.params.threshold;
                let comparison = self.params.comparison;
                let labels = self.params.labels;
                let state = self.lane_mut(lane);
                let value = if estimator == Estimator::Raw {
                    estimate
                } else {
                    state.acc += estimate;
                    state.acc
                };
                if comparison.compute_f64(value, threshold) {
                    if estimator != Estimator::Raw {
                        state.acc = 0.0;
                    }
                    (true, labels.refresh, distance, Some(estimate))
                } else {
                    (false, labels.below, distance, Some(estimate))
                }
            }
        };
        let state = self.lane_mut(lane);
        state.has_previous_input = true;
        state.last_compute = compute;
        Ok(Decision {
            action: if compute {
                Action::Compute
            } else {
                Action::Reuse
            },
            reason,
            step,
            distance,
            indicator: estimate,
            accumulator: state.acc,
            k: state.k,
        })
    }

    fn note_computed(&mut self, lane: Lane, refresh: Refresh) -> Result<(), StepCacheError> {
        let estimator = self.params.estimator;
        let name = self.params.name;
        let state = self.lane_mut(lane);
        match (estimator, refresh) {
            (
                Estimator::KScaled | Estimator::KScaledStoredNorm,
                Refresh::Drift {
                    full_input_change,
                    output_change,
                    output_norm,
                },
            ) => {
                if estimator == Estimator::KScaledStoredNorm && output_norm.is_none() {
                    return Err(StepCacheError::WrongRefresh {
                        cache: name,
                        expected: "drift with output norm",
                    });
                }
                if state.has_payload {
                    state.k = Some(output_change / full_input_change.max(1e-8));
                }
                if estimator == Estimator::KScaledStoredNorm {
                    state.last_output_norm = output_norm;
                    state.consecutive = 0;
                    state.acc = 0.0;
                }
            }
            (Estimator::KScaled | Estimator::KScaledStoredNorm, Refresh::None) => {
                return Err(StepCacheError::WrongRefresh {
                    cache: name,
                    expected: "drift",
                });
            }
            (Estimator::Raw, _) => {
                state.consecutive = 0;
            }
        }
        state.has_payload = true;
        Ok(())
    }

    fn note_reused(&mut self, lane: Lane) {
        self.lane_mut(lane).consecutive += 1;
    }

    fn reset(&mut self) {
        self.lanes.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::super::Cooldown;
    use super::*;

    fn wan(num_steps: usize) -> EasyCache {
        EasyCache::new(EasyCacheParams {
            name: "wan easycache",
            threshold: 0.05,
            window: Window {
                warmup: 1,
                cooldown: Cooldown::Last {
                    steps: 1,
                    of: num_steps,
                },
            },
            estimator: Estimator::KScaled,
            comparison: Comparison::ComputeAtOrAbove,
            reuse_cap: None,
            labels: EasyLabels::WAN,
        })
    }

    fn drift(input_change: f64) -> Signal {
        Signal::Drift {
            input_change,
            output_norm: 1.0,
        }
    }

    fn refresh(full_input_change: f64, output_change: f64) -> Refresh {
        Refresh::Drift {
            full_input_change,
            output_change,
            output_norm: None,
        }
    }

    /// The decision sequence on a recorded signal: warm-up, two computes
    /// install `k = 0.02`, then the estimate accumulates to the threshold.
    #[test]
    fn decision_sequence_on_a_recorded_signal() {
        let mut c = wan(8);
        let mut seen = Vec::new();
        for (step, x) in [0.0, 0.0, 1.0, 2.0, 1.0, 1.0, 1.0, 0.0]
            .into_iter()
            .enumerate()
        {
            let s = if c.needs_signal(Lane::COND, step) {
                drift(x)
            } else {
                Signal::None
            };
            let d = c.decide(Lane::COND, step, s).unwrap();
            if d.compute() {
                c.note_computed(Lane::COND, refresh(1.0, 0.02)).unwrap();
            }
            seen.push(d.reason);
        }
        assert_eq!(c.k(Lane::COND), Some(0.02));
        assert_eq!(
            seen,
            [
                "warmup",
                "initialize",
                "below_threshold",
                "threshold",
                "below_threshold",
                "below_threshold",
                "threshold",
                "cooldown"
            ]
        );
    }

    #[test]
    fn uncond_follows_cond_until_it_has_a_residual() {
        let mut c = wan(8);
        c.decide(Lane::COND, 0, Signal::None).unwrap();
        assert!(c.uncond_compute(Lane::COND, true));
        c.note_computed(Lane::COND, refresh(1.0, 0.0)).unwrap();
        c.decide(Lane::COND, 1, Signal::None).unwrap();
        c.note_computed(Lane::COND, refresh(1.0, 0.02)).unwrap();
        assert!(!c.decide(Lane::COND, 2, drift(1.0)).unwrap().compute());
        assert!(!c.uncond_compute(Lane::COND, true));
        assert!(c.uncond_compute(Lane::COND, false));
    }

    #[test]
    fn raw_estimator_compares_without_accumulating() {
        let mut c = EasyCache::new(EasyCacheParams {
            name: "lingbot easycache",
            threshold: 0.08,
            window: Window {
                warmup: 1,
                cooldown: Cooldown::Last { steps: 1, of: 10 },
            },
            estimator: Estimator::Raw,
            comparison: Comparison::ReuseBelow,
            reuse_cap: Some(Cap {
                limit: 2,
                reason: "max_reuse",
            }),
            labels: EasyLabels::WAN,
        });
        assert_eq!(
            c.decide(Lane::ONLY, 1, Signal::None).unwrap().reason,
            "initialize"
        );
        c.note_computed(Lane::ONLY, Refresh::None).unwrap();
        for step in 2..4 {
            let d = c.decide(Lane::ONLY, step, Signal::Distance(0.05)).unwrap();
            assert!(d.reuse());
            assert_eq!(d.accumulator, 0.0);
            c.note_reused(Lane::ONLY);
        }
        assert_eq!(
            c.decide(Lane::ONLY, 4, Signal::None).unwrap().reason,
            "max_reuse"
        );
    }
}
