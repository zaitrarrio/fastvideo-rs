//! LTX-2.3 official HQ sampling from `models/ltx23.toml` `[official_config]`
//! on NVlabs/Sana `sol-engine`.
//!
//! Stage-1 is 15 dev-schedule steps. Stage-2 is the published 3-sigma refine.
//! Guidance is 3.0. LoRA strengths stay in [`super::pisa`] and are fused by
//! `ltx2::lora` when the distilled file is present. Default-off: unset env /
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
    }
}
