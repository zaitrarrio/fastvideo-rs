//! GEN3C EDM schedule (`sigma_data=0.5`). Spec: docs/ports/gen3c.md.

use super::config::Gen3CPreset;

#[derive(Debug, Clone)]
pub struct Gen3CSchedule {
    pub sigmas: Vec<f64>,
    pub timesteps: Vec<f64>,
    pub sigma_max: f64,
    pub sigma_min: f64,
    pub sigma_data: f64,
}

impl Gen3CSchedule {
    /// Same Karras-ish log-σ path as Cosmos Predict2 (ρ=7), GEN3C σ bounds.
    pub fn new(num_steps: usize, preset: Gen3CPreset) -> Self {
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
        if let Some(last) = sigmas.last_mut() {
            *last = sigma_min;
        }
        sigmas.push(sigma_min);
        let timesteps: Vec<f64> = sigmas[..n]
            .iter()
            .map(|s| s / (s + 1.0) * 1000.0)
            .collect();
        Self {
            sigmas,
            timesteps,
            sigma_max,
            sigma_min,
            sigma_data,
        }
    }

    pub fn inference_sigmas(&self) -> &[f64] {
        &self.sigmas[..self.sigmas.len().saturating_sub(1)]
    }

    /// EDM packing coeffs for current σ: `(c_in, c_skip, c_out)`.
    pub fn edm_coeffs(sigma: f64) -> (f64, f64, f64) {
        let t = sigma / (sigma + 1.0);
        (1.0 - t, 1.0 - t, -t)
    }

    pub fn step_euler(
        &self,
        sample: &[f32],
        derivative: &[f32],
        step: usize,
    ) -> Result<Vec<f32>, String> {
        if sample.len() != derivative.len() {
            return Err(format!(
                "gen3c step: sample {} vs deriv {}",
                sample.len(),
                derivative.len()
            ));
        }
        if step + 1 >= self.sigmas.len() {
            return Err(format!(
                "gen3c step {step} past sigmas {}",
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
    fn sigma_data_is_half() {
        let s = Gen3CSchedule::new(8, Gen3CPreset::Cosmos7b);
        assert!((s.sigma_data - 0.5).abs() < 1e-12);
        assert!((s.sigmas[0] - 80.0).abs() < 1e-6);
    }
}
