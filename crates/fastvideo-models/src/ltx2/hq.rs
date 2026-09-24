//! LTX-2.3 official HQ sampling from `models/ltx23.toml` `[official_config]`
//! on NVlabs/Sana `sol-engine`.
//!
//! Stage-1 is 15 dev-schedule **res2s** steps (29 model calls: 2 per step
//! except the last). Stage-2 is the published 3-sigma Euler refine (the
//! 3-forward Sol/PISA contract). Guidance is 3.0. Distilled LoRA is fused
//! only on the **dev** BF16 DiT (`use_dynamic_shifting`); a distilled
//! checkpoint already has the adapter baked in. Default-off: unset env /
//! false flags leave the existing 2.3 base and distilled paths alone.

/// Stage-1 HQ step count (`official_config.steps`).
pub const STAGE1_STEPS: usize = 15;

/// Stage-2 sigma endpoints, including the terminal 0.
pub const STAGE2_SIGMAS: [f64; 4] = [0.909375, 0.725, 0.421875, 0.0];

pub const GUIDANCE_SCALE: f32 = 3.0;

pub const WIDTH: usize = 1920;

pub const HEIGHT: usize = 1088;

pub const FRAMES: usize = 241;

pub const FPS: f64 = 24.0;

/// 15-step res2s: 2 evaluations per step except the last (`2·15 − 1`).
pub const STAGE1_RES2S_CALLS: usize = 29;

/// ODE res2s midpoint position. Official `c2 = 0.5`.
pub const RES2S_C2: f64 = 0.5;

/// Model evaluations for an `n`-step res2s run (last step is x0 only).
pub fn res2s_num_calls(steps: usize) -> usize {
    steps.saturating_mul(2).saturating_sub(1)
}

fn factorial(n: usize) -> f64 {
    (1..=n).map(|i| i as f64).product()
}

/// φⱼ(z), `z = -h`. Taylor at 0 is `1/j!`.
pub fn res2s_phi(j: usize, neg_h: f64) -> f64 {
    if neg_h.abs() < 1e-10 {
        return 1.0 / factorial(j);
    }
    let remainder: f64 = (0..j).map(|k| neg_h.powi(k as i32) / factorial(k)).sum();
    (neg_h.exp() - remainder) / neg_h.powi(j as i32)
}

/// `(a21, b1, b2)` for one ODE res2s step of log-size `h = -ln(σ_{i+1}/σ_i)`.
pub fn res2s_coefficients(h: f64, c2: f64) -> (f64, f64, f64) {
    let a21 = c2 * res2s_phi(1, -h * c2);
    let b2 = res2s_phi(2, -h) / c2;
    let b1 = res2s_phi(1, -h) - b2;
    (a21, b1, b2)
}

/// `FASTVIDEO_LTX2_HQ=1` (or `hq`) selects the official 15+3 HQ contract.
pub fn requested(value: Option<&str>) -> bool {
    match value.map(str::trim) {
        Some("1") => true,
        Some(v) => v.eq_ignore_ascii_case("hq") || v.eq_ignore_ascii_case("official"),
        None => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn env_is_off_until_hq() {
        assert!(!requested(None));
        assert!(!requested(Some("")));
        assert!(!requested(Some("off")));
        assert!(requested(Some("1")));
        assert!(requested(Some("hq")));
        assert!(requested(Some("official")));
    }

    #[test]
    fn numbers_match_the_ltx23_toml() {
        use super::super::schedule::STAGE_2_DISTILLED_SIGMA_VALUES;
        assert_eq!(STAGE1_STEPS, 15);
        assert_eq!(STAGE2_SIGMAS[..3], STAGE_2_DISTILLED_SIGMA_VALUES[..]);
        assert_eq!(*STAGE2_SIGMAS.last().unwrap(), 0.0);
        assert_eq!(GUIDANCE_SCALE, 3.0);
        assert_eq!((WIDTH, HEIGHT, FRAMES), (1920, 1088, 241));
        assert_eq!(FPS, 24.0);
        assert_eq!(res2s_num_calls(STAGE1_STEPS), STAGE1_RES2S_CALLS);
        assert_eq!(STAGE1_RES2S_CALLS, 29);
    }

    #[test]
    fn res2s_phi_matches_closed_form() {
        let z = -0.5_f64;
        let phi1 = (z.exp() - 1.0) / z;
        let phi2 = (z.exp() - 1.0 - z) / (z * z);
        assert!((res2s_phi(1, z) - phi1).abs() < 1e-12);
        assert!((res2s_phi(2, z) - phi2).abs() < 1e-12);
        assert!((res2s_phi(1, 0.0) - 1.0).abs() < 1e-12);
        assert!((res2s_phi(2, 0.0) - 0.5).abs() < 1e-12);
        let (a21, b1, b2) = res2s_coefficients(-(0.5_f64).ln(), RES2S_C2);
        assert!(a21.is_finite() && b1.is_finite() && b2.is_finite());
        assert_eq!(res2s_num_calls(8), 15);
        assert_eq!(res2s_num_calls(1), 1);
        assert_eq!(res2s_num_calls(0), 0);
    }
}
