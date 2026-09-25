//! Numerical checks that fit in CI without a Python / Diffusers snapshot.

use super::family::{compute_empirical_mu, packed_hw};

/// Mean squared error; `None` if lengths differ or empty.
pub fn mse(a: &[f32], b: &[f32]) -> Option<f32> {
    if a.len() != b.len() || a.is_empty() {
        return None;
    }
    let s: f32 = a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum();
    Some(s / a.len() as f32)
}

/// PSNR in dB. Identical buffers → `+inf`.
pub fn psnr(a: &[f32], b: &[f32], max: f32) -> Option<f32> {
    let err = mse(a, b)?;
    if err == 0.0 {
        return Some(f32::INFINITY);
    }
    Some(10.0 * (max * max / err).log10())
}

/// Klein 1024² / 4-step μ used by the Vast compare harness.
pub fn klein_1024_mu() -> f64 {
    let (h, w) = packed_hw(1024, 1024, 8);
    compute_empirical_mu(h * w, 4)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::flux2::{flux2_time_shift, pack_latents_2x2, unpatchify_2x2};
    use crate::schedulers::FlowMatchEulerDiscreteScheduler;

    #[test]
    fn klein_1024_timestep_table() {
        let mu = klein_1024_mu();
        let expected = {
            let seq = 4096.0;
            let m_200 = 0.00016927 * seq + 0.45666666;
            let m_10 = 8.73809524e-05 * seq + 1.89833333;
            let a = (m_200 - m_10) / 190.0;
            let b = m_200 - 200.0 * a;
            a * 4.0 + b
        };
        assert!((mu - expected).abs() < 1e-12, "mu={mu} expected={expected}");
        let mut sched = FlowMatchEulerDiscreteScheduler::new(1000, 1.0);
        sched.set_timesteps_flux2(4, Some(mu));
        assert_eq!(sched.inference_timesteps().len(), 4);
        assert!((sched.inference_sigmas()[0] - flux2_time_shift(1.0, mu)).abs() < 1e-12);
        assert!((sched.inference_sigmas()[4] - 0.0).abs() < 1e-12);
        assert!(sched.inference_sigmas()[1] < sched.inference_sigmas()[0]);
    }

    #[test]
    fn pack_roundtrip_and_1024_seq() {
        let input: Vec<f32> = (0..32 * 8 * 8).map(|i| i as f32 * 0.01).collect();
        let packed = pack_latents_2x2(&input, 32, 8, 8).unwrap();
        assert_eq!(packed.len(), 128 * 4 * 4);
        let back = unpatchify_2x2(&packed, 32, 4, 4).unwrap();
        assert_eq!(back, input);
        assert_eq!(packed_hw(1024, 1024, 8), (64, 64));
    }

    #[test]
    fn psnr_identical_is_inf() {
        let a = [0.1f32, -0.2, 0.3];
        assert!(psnr(&a, &a, 1.0).unwrap().is_infinite());
        assert!(mse(&[1.0], &[1.0, 2.0]).is_none());
    }
}
