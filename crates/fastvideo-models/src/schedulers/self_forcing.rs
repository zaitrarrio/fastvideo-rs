//! FastVideo `SelfForcingFlowMatchScheduler` as the causal DMD stage uses it
//! (`pipelines/basic/wan/stages/causal_denoising.py CausalDMDDenosingStage`).
//!
//! The scheduler is the checkpoint's own (`scheduler/scheduler_config.json` of
//! `wlsaidhi/SFWan2.1-T2V-1.3B-Diffusers`: `num_inference_steps` 1000, `shift`
//! 5, `sigma_min` 0, `extra_one_step`), never re-`set_timesteps`: a float32
//! table `sigmas = shift*s / (1 + (shift-1)*s)` over
//! `s = linspace(1, 0, 1001)[:-1]`, `timesteps = sigmas * 1000`.
//!
//! With `warp_denoising_step` (on for every Self-Forcing config) the DMD
//! steps `[1000, 750, 500, 250]` become `cat(timesteps, [0])[1000 - step]`:
//! 1000, 937.5, 833.3, 625 (fractional timesteps reach the DiT as float).
//! Each step: `x0 = x - sigma(t) * v` (`pred_noise_to_pred_video`, float64,
//! cast back to the DiT dtype), then unless last
//! `x = (1 - sigma(t_next)) * x0 + sigma(t_next) * noise` (`add_noise`).
//! `sigma(t)` is the table sigma at `argmin |timesteps - t|`.

/// SF-Wan 1.3B `dmd_denoising_steps` (`SelfForcingWanT2V480PConfig`).
pub const SF_WAN_1_3B_DMD_STEPS: [i32; 4] = [1000, 750, 500, 250];

#[derive(Debug, Clone)]
pub struct SelfForcingSchedule {
    table_sigmas: Vec<f32>,
    table_timesteps: Vec<f32>,
    /// The (warped) timesteps each block is denoised at.
    pub timesteps: Vec<f32>,
}

/// `torch.linspace(start, end, steps)` in float32 (CPU kernel: the first half
/// from `start`, the second half from `end`).
fn linspace_f32(start: f32, end: f32, steps: usize) -> Vec<f32> {
    if steps == 1 {
        return vec![start];
    }
    let step = (end - start) / (steps - 1) as f32;
    let half = steps / 2;
    (0..steps)
        .map(|i| {
            if i < half {
                start + step * i as f32
            } else {
                end - step * (steps - i - 1) as f32
            }
        })
        .collect()
}

impl SelfForcingSchedule {
    /// The table of a `SelfForcingFlowMatchScheduler(num_inference_steps=n,
    /// shift, sigma_min=0, extra_one_step=True)` and `steps` warped
    /// (`warp = true`, FastVideo's default) or taken as they are.
    pub fn new(steps: &[i32], shift: f64, num_train_timesteps: usize, warp: bool) -> Self {
        let n = num_train_timesteps.max(1);
        let (sigma_max, sigma_min) = (1.0f32, 0.0f32);
        let mut lin = linspace_f32(sigma_max, sigma_min, n + 1);
        lin.pop();
        let shift = shift as f32;
        let table_sigmas: Vec<f32> = lin
            .iter()
            .map(|&s| (shift * s) / (1.0 + (shift - 1.0) * s))
            .collect();
        let table_timesteps: Vec<f32> = table_sigmas.iter().map(|&s| s * n as f32).collect();
        let timesteps = steps
            .iter()
            .map(|&t| {
                if warp {
                    let idx = (n as i64 - i64::from(t)).clamp(0, n as i64) as usize;
                    table_timesteps.get(idx).copied().unwrap_or(0.0)
                } else {
                    t as f32
                }
            })
            .collect();
        Self {
            table_sigmas,
            table_timesteps,
            timesteps,
        }
    }

    /// SF-Wan 1.3B: steps [1000, 750, 500, 250], shift 5, warped.
    pub fn sf_wan_1_3b() -> Self {
        Self::new(&SF_WAN_1_3B_DMD_STEPS, 5.0, 1000, true)
    }

    /// Table sigma at `argmin |timesteps - t|` (first index on ties).
    pub fn sigma(&self, t: f32) -> f64 {
        let mut best = 0usize;
        let mut best_d = f64::INFINITY;
        for (i, &ts) in self.table_timesteps.iter().enumerate() {
            let d = (f64::from(ts) - f64::from(t)).abs();
            if d < best_d {
                best_d = d;
                best = i;
            }
        }
        f64::from(self.table_sigmas[best])
    }

    pub fn num_steps(&self) -> usize {
        self.timesteps.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn warped_sf_wan_steps() {
        let s = SelfForcingSchedule::sf_wan_1_3b();
        let want = [1000.0f32, 937.5, 833.333_3, 625.0];
        for (g, w) in s.timesteps.iter().zip(want) {
            assert!((g - w).abs() < 1e-2, "{g} vs {w}");
        }
        // Each warped step maps onto its own table sigma, t / 1000.
        for &t in &s.timesteps {
            assert!((s.sigma(t) - f64::from(t) / 1000.0).abs() < 1e-5);
        }
        assert_eq!(s.sigma(0.0), f64::from(s.table_sigmas[999]));
    }

    #[test]
    fn unwarped_steps_pass_through() {
        let s = SelfForcingSchedule::new(&[1000, 750], 5.0, 1000, false);
        assert_eq!(s.timesteps, vec![1000.0, 750.0]);
    }
}
