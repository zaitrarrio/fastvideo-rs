//! Cosmos Predict2 EDM-style FlowMatch sigmas (`sigma_max`/`sigma_min`).

use super::config::CosmosPreset;

#[derive(Debug, Clone)]
pub struct CosmosSchedule {
    pub sigmas: Vec<f64>,
    pub timesteps: Vec<f64>,
    pub sigma_max: f64,
    pub sigma_min: f64,
    pub sigma_data: f64,
}

impl CosmosSchedule {
    /// Karras-ish log-σ schedule used by Diffusers Cosmos2 (ρ=7).
    pub fn new(num_steps: usize, preset: CosmosPreset) -> Self {
        let sigma_max = preset.sigma_max();
        let sigma_min = preset.sigma_min();
        let sigma_data = preset.sigma_data();
        let n = num_steps.max(1);
        let rho = 7.0f64;
        let max_inv = sigma_max.powf(1.0 / rho);
        let min_inv = sigma_min.powf(1.0 / rho);
        let mut sigmas = Vec::with_capacity(n + 1);
        for i in 0..n {
            let t = if n == 1 {
                0.0
            } else {
                i as f64 / (n - 1) as f64
            };
            let s = (max_inv + t * (min_inv - max_inv)).powf(rho);
            sigmas.push(s);
        }
        // final_sigmas_type == "sigma_min": keep last as sigma_min (not 0).
        if let Some(last) = sigmas.last_mut() {
            *last = sigma_min;
        }
        // Append terminal for Euler (duplicate min → zero-ish step).
        sigmas.push(sigma_min);
        let timesteps: Vec<f64> = sigmas[..n].iter().map(|s| s / (s + 1.0) * 1000.0).collect();
        Self {
            sigmas,
            timesteps,
            sigma_max,
            sigma_min,
            sigma_data,
        }
    }

    pub fn inference_sigmas(&self) -> &[f64] {
        // Exclude terminal duplicate for step indexing.
        &self.sigmas[..self.sigmas.len().saturating_sub(1)]
    }

    pub fn timesteps(&self) -> &[f64] {
        &self.timesteps
    }

    /// EDM packing coeffs for current σ: `(c_in, c_skip, c_out)`.
    pub fn edm_coeffs(sigma: f64) -> (f64, f64, f64) {
        let t = sigma / (sigma + 1.0);
        (1.0 - t, 1.0 - t, -t)
    }

    /// Euler: `x_{i+1} = x_i + (σ_{i+1} - σ_i) * d`.
    pub fn step_euler(
        &self,
        sample: &[f32],
        derivative: &[f32],
        step: usize,
    ) -> Result<Vec<f32>, String> {
        if sample.len() != derivative.len() {
            return Err(format!(
                "cosmos step: sample {} vs deriv {}",
                sample.len(),
                derivative.len()
            ));
        }
        if step + 1 >= self.sigmas.len() {
            return Err(format!(
                "cosmos step {step} past sigmas {}",
                self.sigmas.len()
            ));
        }
        let ds = (self.sigmas[step + 1] - self.sigmas[step]) as f32;
        Ok(sample
            .iter()
            .zip(derivative.iter())
            .map(|(&x, &d)| x + ds * d)
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sigma_bounds() {
        let s = CosmosSchedule::new(8, CosmosPreset::V2w2b);
        assert!((s.sigmas[0] - 80.0).abs() < 1e-6);
        assert!((s.inference_sigmas().last().copied().unwrap() - 0.002).abs() < 1e-9);
        assert_eq!(s.timesteps().len(), 8);
    }

    #[test]
    fn edm_at_zeroish() {
        let (c_in, c_skip, c_out) = CosmosSchedule::edm_coeffs(0.0);
        assert!((c_in - 1.0).abs() < 1e-12);
        assert!((c_skip - 1.0).abs() < 1e-12);
        assert!(c_out.abs() < 1e-12);
    }
}
