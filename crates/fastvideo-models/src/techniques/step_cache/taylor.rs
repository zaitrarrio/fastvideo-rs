//! TaylorSeer lite: forecast `proj_out` from divided differences on a fixed
//! schedule and skip the block stack on the forecast forwards (diffusers
//! `TaylorSeerCacheHook`, sol-engine `wan_cache.py` `TaylorSeerRuntime`).
//!
//! The clock is the forward counter: it starts at −1 and every
//! [`StepCache::decide`] advances it. Warm-up, `(step − warmup − 1) %
//! interval == 0` and the cool-down compute; every other forward
//! extrapolates. The factor history lives with the family (device tensors
//! on the cudarc path, [`host_update`] / [`host_predict`] on the host);
//! this type owns the step bookkeeping and the coefficients both use:
//! the divided-difference scale `1 / delta` (f32) and the forecast
//! coefficients `offset^k / k!` (f64, cast to f32).

use super::{Action, Algorithm, Decision, Lane, Refresh, Signal, StepCache, StepCacheError};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TaylorParams {
    pub name: &'static str,
    pub interval: usize,
    pub warmup: usize,
    pub cooldown_start: usize,
    pub max_order: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Taylor {
    params: TaylorParams,
    current_step: isize,
    last_update: Option<isize>,
}

impl Taylor {
    pub fn new(params: TaylorParams) -> Result<Self, StepCacheError> {
        if params.interval == 0 {
            return Err(StepCacheError::ZeroInterval { cache: params.name });
        }
        Ok(Self {
            params,
            current_step: -1,
            last_update: None,
        })
    }

    pub fn params(&self) -> &TaylorParams {
        &self.params
    }

    /// The forward counter (−1 before the first forward).
    pub fn current_step(&self) -> isize {
        self.current_step
    }

    /// The forward of the last factor update.
    pub fn last_update(&self) -> Option<isize> {
        self.last_update
    }

    /// The gap to the last update for a factor refresh at this forward:
    /// 1 for the first, an error for a second refresh at the same forward.
    pub fn update_delta(&self) -> Result<isize, StepCacheError> {
        let delta = self
            .last_update
            .map(|last| self.current_step - last)
            .unwrap_or(1);
        if self.last_update.is_some() && delta == 0 {
            return Err(StepCacheError::ZeroDelta {
                cache: self.params.name,
            });
        }
        Ok(delta)
    }

    /// The forecast offset `current − last update` of an extrapolated forward.
    pub fn forecast_offset(&self) -> Result<isize, StepCacheError> {
        self.last_update.map(|last| self.current_step - last).ok_or(
            StepCacheError::ForecastBeforeUpdate {
                cache: self.params.name,
            },
        )
    }
}

/// The divided-difference scale of a refresh `delta` forwards after the
/// last: `1.0 / delta as f32`.
pub fn difference_scale(delta: isize) -> f32 {
    1.0 / delta as f32
}

/// The forecast coefficients `offset^k / k!` for orders `0..orders`, the
/// power and factorial accumulated in f64 and each quotient cast to f32
/// (`TaylorSeerState.predict`). Order 0 is `1.0`.
pub fn forecast_coefficients(orders: usize, step_offset: isize) -> Vec<f32> {
    let mut out = Vec::with_capacity(orders);
    let mut pow = 1.0f64;
    let mut fact = 1.0f64;
    for order in 0..orders {
        if order > 0 {
            pow *= step_offset as f64;
            fact *= order as f64;
        }
        out.push((pow / fact) as f32);
    }
    out
}

/// Host factor refresh: the new value, then divided differences against the
/// previous factors (`TaylorSeerState.update`).
pub fn host_update(
    cache: &'static str,
    previous: &[Vec<f32>],
    features: &[f32],
    delta_step: isize,
    max_order: usize,
) -> Result<Vec<Vec<f32>>, StepCacheError> {
    if delta_step == 0 && !previous.is_empty() {
        return Err(StepCacheError::ZeroDelta { cache });
    }
    let mut factors = vec![features.to_vec()];
    if previous.is_empty() {
        return Ok(factors);
    }
    let inv = difference_scale(delta_step);
    for j in 0..max_order {
        let Some(prev) = previous.get(j) else {
            break;
        };
        if prev.len() != factors[j].len() {
            return Err(StepCacheError::FactorLength { cache });
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

/// Host forecast `Σ_k factor_k · coefficient_k`, accumulated from zero.
pub fn host_predict(
    cache: &'static str,
    factors: &[Vec<f32>],
    step_offset: isize,
) -> Result<Vec<f32>, StepCacheError> {
    let base = factors.first().ok_or(StepCacheError::NoFactors { cache })?;
    let coefficients = forecast_coefficients(factors.len(), step_offset);
    let mut output = vec![0.0f32; base.len()];
    for (factor, coeff) in factors.iter().zip(coefficients) {
        if factor.len() != base.len() {
            return Err(StepCacheError::FactorLength { cache });
        }
        for (dst, value) in output.iter_mut().zip(factor) {
            *dst += value * coeff;
        }
    }
    Ok(output)
}

impl StepCache for Taylor {
    fn name(&self) -> &'static str {
        self.params.name
    }

    fn algorithm(&self) -> Algorithm {
        Algorithm::Taylor
    }

    fn needs_signal(&self, _lane: Lane, _step: usize) -> bool {
        false
    }

    /// Advance the forward counter and decide it (`_measure_should_compute`).
    fn decide(
        &mut self,
        _lane: Lane,
        _step: usize,
        _signal: Signal,
    ) -> Result<Decision, StepCacheError> {
        self.current_step += 1;
        let p = self.params;
        let step = self.current_step as usize;
        let refresh =
            (self.current_step - p.warmup as isize - 1).rem_euclid(p.interval as isize) == 0;
        let (action, reason) = if step < p.warmup {
            (Action::Compute, "warmup")
        } else if step >= p.cooldown_start {
            (Action::Compute, "cooldown")
        } else if refresh {
            (Action::Compute, "interval_refresh")
        } else {
            (Action::Extrapolate, "taylor_forecast")
        };
        Ok(Decision {
            action,
            reason,
            step,
            distance: None,
            indicator: None,
            accumulator: 0.0,
            k: None,
        })
    }

    /// The factors were refreshed at this forward.
    fn note_computed(&mut self, _lane: Lane, _refresh: Refresh) -> Result<(), StepCacheError> {
        self.last_update = Some(self.current_step);
        Ok(())
    }

    fn note_reused(&mut self, _lane: Lane) {}

    fn reset(&mut self) {
        self.current_step = -1;
        self.last_update = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn official(num_steps: usize) -> Taylor {
        Taylor::new(TaylorParams {
            name: "wan taylorseer",
            interval: 3,
            warmup: 3,
            cooldown_start: num_steps.saturating_sub(2),
            max_order: 1,
        })
        .unwrap()
    }

    /// The decision sequence of ten forwards: warm-up 0-2, then
    /// `(step − 4) % 3 == 0` refreshes, cool-down from 8.
    #[test]
    fn decision_sequence() {
        let mut t = official(10);
        let mut seen = Vec::new();
        for _ in 0..10 {
            let d = t.decide(Lane::ONLY, 0, Signal::None).unwrap();
            if d.compute() {
                t.note_computed(Lane::ONLY, Refresh::None).unwrap();
            } else {
                assert!(t.forecast_offset().unwrap() > 0);
            }
            seen.push(d.reason);
        }
        assert_eq!(
            seen,
            [
                "warmup",
                "warmup",
                "warmup",
                "taylor_forecast",
                "interval_refresh",
                "taylor_forecast",
                "taylor_forecast",
                "interval_refresh",
                "cooldown",
                "cooldown"
            ]
        );
    }

    #[test]
    fn delta_and_forecast_bookkeeping() {
        let mut t = official(10);
        assert!(t.forecast_offset().is_err());
        t.decide(Lane::ONLY, 0, Signal::None).unwrap();
        assert_eq!(t.update_delta().unwrap(), 1);
        t.note_computed(Lane::ONLY, Refresh::None).unwrap();
        assert!(t.update_delta().is_err());
        t.decide(Lane::ONLY, 0, Signal::None).unwrap();
        t.decide(Lane::ONLY, 0, Signal::None).unwrap();
        assert_eq!(t.update_delta().unwrap(), 2);
        assert_eq!(t.forecast_offset().unwrap(), 2);
        assert!(Taylor::new(TaylorParams {
            interval: 0,
            ..*t.params()
        })
        .is_err());
    }

    #[test]
    fn order_one_forecast_is_value_plus_difference() {
        let first = host_update("t", &[], &[2.0, 4.0], 1, 1).unwrap();
        let second = host_update("t", &first, &[6.0, 8.0], 2, 1).unwrap();
        assert_eq!(second[1], vec![2.0, 2.0]);
        assert_eq!(host_predict("t", &second, 1).unwrap(), vec![8.0, 10.0]);
        assert_eq!(forecast_coefficients(3, 2), vec![1.0, 2.0, 2.0]);
        assert_eq!(difference_scale(4), 0.25);
    }
}
