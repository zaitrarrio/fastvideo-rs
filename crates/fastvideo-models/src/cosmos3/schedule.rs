//! Cosmos3-Super sampling schedule.
//!
//! The Hub `scheduler/scheduler_config.json` is Diffusers'
//! `UniPCMultistepScheduler` with `use_karras_sigmas`, `use_flow_sigmas`,
//! `sigma_min = 0.147`, `sigma_max = 200`, `solver_order = 2`, `bh2`,
//! `predict_x0`, `final_sigmas_type = "zero"`. In `set_timesteps` the Karras
//! branch runs first:
//!
//! ```text
//! ramp = linspace(0, 1, n);  s = (max^(1/7) + ramp (min^(1/7) - max^(1/7)))^7
//! sigma = s / (s + 1);       timestep = int64(sigma * 1000)
//! ```
//!
//! and never reaches the `flow_shift` branch, so the `flow_shift=10.0` the
//! HF example and `models/cosmos3.toml` pass is inert under this config.

use crate::schedulers::FlowUniPCMultistepScheduler;

/// Hub scheduler constants.
pub const KARRAS_SIGMA_MIN: f64 = 0.147;
pub const KARRAS_SIGMA_MAX: f64 = 200.0;
pub const KARRAS_RHO: f64 = 7.0;

/// Karras flow sigmas (no terminal zero) in f64, as `set_timesteps` derives the
/// int64 timesteps from them before storing the sigmas as f32.
pub fn karras_flow_sigmas(steps: usize, sigma_min: f64, sigma_max: f64) -> Vec<f64> {
    let n = steps.max(1);
    let (lo, hi) = (sigma_min.powf(1.0 / KARRAS_RHO), sigma_max.powf(1.0 / KARRAS_RHO));
    (0..n)
        .map(|i| {
            let ramp = if n == 1 { 0.0 } else { i as f64 / (n - 1) as f64 };
            let s = (hi + ramp * (lo - hi)).powf(KARRAS_RHO);
            s / (s + 1.0)
        })
        .collect()
}

/// The configured UniPC (`solver_order 2`, `bh2`, `predict_x0`, lower-order final).
pub fn unipc(steps: usize) -> FlowUniPCMultistepScheduler {
    let mut s = FlowUniPCMultistepScheduler::new(1_000, 1.0);
    s.set_sigmas(&karras_flow_sigmas(steps, KARRAS_SIGMA_MIN, KARRAS_SIGMA_MAX));
    s
}

/// The transformer's timestep input: `int64` scheduler timestep × `timestep_scale`.
pub fn transformer_timestep(t: i64, timestep_scale: f32) -> f32 {
    t as f32 * timestep_scale
}

#[cfg(test)]
mod tests {
    use super::*;

    const HUB_SCHED: &str = include_str!("hub/super_scheduler_config.json");

    #[test]
    fn hub_scheduler_is_karras_flow_unipc() {
        let v: serde_json::Value = serde_json::from_str(HUB_SCHED).unwrap();
        assert_eq!(v["_class_name"], "UniPCMultistepScheduler");
        assert_eq!(v["use_karras_sigmas"], true);
        assert_eq!(v["use_flow_sigmas"], true);
        assert_eq!(v["solver_order"], 2);
        assert_eq!(v["solver_type"], "bh2");
        assert_eq!(v["predict_x0"], true);
        assert_eq!(v["final_sigmas_type"], "zero");
        assert_eq!(v["sigma_min"].as_f64(), Some(KARRAS_SIGMA_MIN));
        assert_eq!(v["sigma_max"].as_f64(), Some(KARRAS_SIGMA_MAX));
    }

    #[test]
    fn official_35_step_schedule() {
        let s = unipc(35);
        let sig = s.inference_sigmas();
        assert_eq!(sig.len(), 36);
        // First: 200 / 201; last before zero: 0.147 / 1.147.
        assert!((sig[0] - 200.0 / 201.0).abs() < 1e-6);
        assert!((sig[34] - 0.147 / 1.147).abs() < 1e-6);
        assert_eq!(sig[35], 0.0);
        assert!(sig.windows(2).all(|w| w[1] < w[0]));
        let ts = s.inference_timesteps_i64();
        assert_eq!(ts[0], 995);
        assert_eq!(ts[34], 128);
        assert!((transformer_timestep(ts[0], 0.001) - 0.995).abs() < 1e-6);
    }
}
