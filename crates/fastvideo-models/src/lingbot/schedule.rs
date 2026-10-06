//! LingBot sampling schedules: FlowUniPC (`scheduling_flow_unipc.py`) for the
//! base stage and the refiner's truncated low-noise tail (`utils.py`
//! `compute_refiner_sigmas`).
//!
//! The Hub scheduler config is `shift=1`; the pipeline passes `shift=3` to
//! `set_timesteps`, so `sigma_max = 0.999`, `sigma_min = 0` come from the
//! unshifted training table and the shift applies once, at set time.

use crate::schedulers::{FlowMatchEulerDiscreteScheduler, FlowUniPCMultistepScheduler};

use super::config::LingBotPreset;

/// `LOW_NOISE_TAIL_V1_DEFAULT_STEPS`.
pub const REFINER_DEFAULT_TAIL_STEPS: usize = 2;

/// Base-stage FlowUniPC: `FlowUniPCMultistepScheduler(shift=1)` then
/// `set_timesteps(steps, shift=shift)`.
pub fn base_unipc(steps: usize, shift: f64) -> FlowUniPCMultistepScheduler {
    let mut s = FlowUniPCMultistepScheduler::new(1_000, 1.0);
    s.shift = shift;
    s.set_timesteps(steps.max(1));
    s
}

/// `compute_refiner_sigmas`: the shifted base grid cut at `t_thresh` (which
/// is inserted as the first sigma when absent), plus `tail_steps` extra sigmas
/// spaced linearly from the last one toward `sigma_min`. Values are rounded
/// to f32 as the reference returns them.
pub fn refiner_sigmas(
    sigma_max: f64,
    sigma_min: f64,
    num_inference_steps: usize,
    shift: f64,
    t_thresh: f64,
    tail_steps: usize,
) -> Result<Vec<f64>, String> {
    if !(t_thresh > 0.0 && t_thresh <= 1.0) {
        return Err(format!("refiner t_thresh must lie in (0, 1], got {t_thresh}"));
    }
    if num_inference_steps == 0 {
        return Err("refiner steps must be >= 1".into());
    }
    let n = num_inference_steps;
    let eps = 1e-6;
    let mut sigmas: Vec<f64> = (0..n)
        .map(|i| sigma_max + (sigma_min - sigma_max) * i as f64 / n as f64)
        .map(|b| shift * b / (1.0 + (shift - 1.0) * b))
        .filter(|&s| s <= t_thresh + eps)
        .collect();
    if sigmas.first().is_none_or(|&s| (s - t_thresh).abs() > eps) {
        sigmas.insert(0, t_thresh);
    }
    if tail_steps > 0 {
        let start = *sigmas.last().expect("non-empty");
        let stop = sigma_min.min(start);
        // np.linspace(start, stop, tail + 2)[1:-1]
        let m = tail_steps + 1;
        for i in 1..=tail_steps {
            sigmas.push(start + (stop - start) * i as f64 / m as f64);
        }
    }
    if sigmas.windows(2).any(|w| w[1] >= w[0]) {
        return Err(format!("refiner sigmas not strictly descending: {sigmas:?}"));
    }
    Ok(sigmas.into_iter().map(|s| f64::from(s as f32)).collect())
}

/// Refiner FlowUniPC: `set_timesteps(len, sigmas=refiner_sigmas, shift=1)`.
pub fn refiner_unipc(
    steps: usize,
    shift: f64,
    t_thresh: f64,
    tail_steps: usize,
) -> Result<FlowUniPCMultistepScheduler, String> {
    let mut s = FlowUniPCMultistepScheduler::new(1_000, 1.0);
    let sigmas = refiner_sigmas(s.sigma_max(), s.sigma_min(), steps, shift, t_thresh, tail_steps)?;
    s.set_sigmas(&sigmas);
    Ok(s)
}

/// `_transformer_timestep`: the int64 scheduler timestep → sigma rounded to
/// the transformer dtype (bf16) → `* 1000` in f32.
pub fn transformer_timestep(t: i64, bf16: bool) -> f32 {
    let sigma = t as f32 / 1000.0;
    let sigma = if bf16 {
        half_bf16_round(sigma)
    } else {
        sigma
    };
    sigma * 1000.0
}

fn half_bf16_round(x: f32) -> f32 {
    let bits = x.to_bits();
    let lsb = (bits >> 16) & 1;
    f32::from_bits(bits.wrapping_add(0x7fff + lsb) & 0xffff_0000)
}

/// Legacy FlowMatch Euler wrapper (dense 1.3B smoke path).
#[derive(Debug, Clone)]
pub struct LingBotSchedule {
    pub inner: FlowMatchEulerDiscreteScheduler,
    pub flow_shift: f64,
}

impl LingBotSchedule {
    pub fn new(num_steps: usize, preset: LingBotPreset) -> Self {
        let flow_shift = preset.flow_shift();
        let mut inner = FlowMatchEulerDiscreteScheduler::new(1_000, flow_shift);
        inner.set_timesteps(num_steps.max(1));
        Self { inner, flow_shift }
    }

    pub fn timesteps(&self) -> &[f64] {
        self.inner.inference_timesteps()
    }

    pub fn sigmas(&self) -> &[f64] {
        self.inner.inference_sigmas()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base_schedule_is_shift_three_from_0_999() {
        let s = base_unipc(40, 3.0);
        assert!((s.sigma_max() - 0.999).abs() < 1e-6);
        assert_eq!(s.sigma_min(), 0.0);
        let sig = s.inference_sigmas();
        assert_eq!(sig.len(), 41);
        let want0 = 3.0 * 0.999 / (1.0 + 2.0 * 0.999);
        // sigma_max is the f32 table entry (0.999 ± 1e-7), as upstream.
        assert!((sig[0] - want0).abs() < 1e-6);
        assert_eq!(*sig.last().unwrap(), 0.0);
        // np.linspace(0.999, 0, 41)[39] shifted.
        let b = 0.999 - 0.999 * 39.0 / 40.0;
        assert!((sig[39] - 3.0 * b / (1.0 + 2.0 * b)).abs() < 1e-6);
        assert_eq!(s.inference_timesteps_i64()[0], 999);
    }

    #[test]
    fn refiner_sigmas_match_reference_values() {
        // python: compute_refiner_sigmas(sigma_max=0.999, sigma_min=0.0,
        //   num_inference_steps=8, shift=3, t_thresh=0.85, tail_steps=2)
        let s = refiner_sigmas(0.999, 0.0, 8, 3.0, 0.85, 2).unwrap();
        let base: Vec<f64> = (0..8)
            .map(|i| 0.999 - 0.999 * i as f64 / 8.0)
            .map(|b| 3.0 * b / (1.0 + 2.0 * b))
            .collect();
        let kept: Vec<f64> = base.iter().copied().filter(|&v| v <= 0.85 + 1e-6).collect();
        assert_eq!(s[0], f64::from(0.85f32));
        assert_eq!(s.len(), 1 + kept.len() + 2);
        for (a, b) in s[1..1 + kept.len()].iter().zip(&kept) {
            assert_eq!(*a, f64::from(*b as f32));
        }
        let last = *kept.last().unwrap();
        assert!((s[s.len() - 2] - last * 2.0 / 3.0).abs() < 1e-6);
        assert!((s[s.len() - 1] - last / 3.0).abs() < 1e-6);
        assert!(s.windows(2).all(|w| w[1] < w[0]));
    }

    #[test]
    fn refiner_scheduler_runs_the_tail() {
        let s = refiner_unipc(8, 3.0, 0.85, 2).unwrap();
        let n = s.inference_timesteps().len();
        assert_eq!(n, s.inference_sigmas().len() - 1);
        assert_eq!(s.inference_timesteps_i64()[0], 850);
        assert_eq!(*s.inference_sigmas().last().unwrap(), 0.0);
    }

    #[test]
    fn transformer_timestep_rounds_sigma_to_bf16() {
        // 0.85 is not a bf16 value: 0.8515625 is the nearest.
        assert_eq!(transformer_timestep(850, true), 0.8515625 * 1000.0);
        assert_eq!(transformer_timestep(850, false), 0.85f32 * 1000.0);
    }

    #[test]
    fn legacy_shift_is_three() {
        let s = LingBotSchedule::new(8, LingBotPreset::Dense13b);
        assert!((s.flow_shift - 3.0).abs() < 1e-12);
        assert_eq!(s.timesteps().len(), 8);
    }
}
