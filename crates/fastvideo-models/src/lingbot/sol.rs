//! LingBot-Video Sol-engine sampling and gap record.
//!
//! `models/lingbot_video.toml` publishes the official two-stage T2V contract
//! (base 832×480 / 121f / 40 steps, refiner 1920×1088 / 8 steps, t_thresh
//! 0.85, sigma tail 2). Cache / PISA / topology stay gap-logged: the profile
//! names those techniques and leaves the algorithms unspecified.

/// Official base canvas (`[official_config]`).
pub const OFFICIAL_WIDTH: usize = 832;
pub const OFFICIAL_HEIGHT: usize = 480;
pub const OFFICIAL_FRAMES: usize = 121;
pub const OFFICIAL_STEPS: usize = 40;
pub const OFFICIAL_GUIDANCE: f32 = 3.0;
pub const OFFICIAL_SHIFT: f64 = 3.0;
pub const OFFICIAL_FPS: u32 = 24;

/// Official 1080p refiner. Recorded; this crate has no LingBot upsampler.
pub const REFINER_WIDTH: usize = 1920;
pub const REFINER_HEIGHT: usize = 1088;
pub const REFINER_FRAMES: usize = 121;
pub const REFINER_STEPS: usize = 8;
pub const REFINER_GUIDANCE: f32 = 3.0;
pub const REFINER_SHIFT: f64 = 3.0;
pub const REFINER_T_THRESH: f64 = 0.85;
pub const REFINER_SIGMA_TAIL_STEPS: usize = 2;

/// `FASTVIDEO_LINGBOT_OFFICIAL=1` (or `official`) selects the published base
/// sampling contract. Unset leaves the existing 81-frame / no-CFG path alone.
pub fn official_requested(value: Option<&str>) -> bool {
    match value.map(str::trim) {
        Some("1") => true,
        Some(v) => v.eq_ignore_ascii_case("official"),
        None => false,
    }
}

/// `FASTVIDEO_LINGBOT_SOL=1` (or `cache` / `pisa`) logs the gap and stays dense.
pub fn requested(value: Option<&str>) -> bool {
    let Some(v) = value.map(str::trim) else {
        return false;
    };
    if v.is_empty() || v.eq_ignore_ascii_case("off") || v == "0" || v.eq_ignore_ascii_case("false")
    {
        return false;
    }
    v == "1"
        || v.eq_ignore_ascii_case("cache")
        || v.eq_ignore_ascii_case("pisa")
        || v.eq_ignore_ascii_case("sol")
}

pub const GAP: &str = "lingbot sol: cache/pisa/topology stay unported \
(sol-engine LingBot profile names the techniques and leaves the algorithms unspecified)";

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn env_is_off_by_default() {
        assert!(!requested(None));
        assert!(!requested(Some("")));
        assert!(!requested(Some("off")));
        assert!(!requested(Some("0")));
        assert!(requested(Some("1")));
        assert!(requested(Some("cache")));
        assert!(requested(Some("pisa")));
    }

    #[test]
    fn gap_names_the_missing_algorithms() {
        assert!(GAP.contains("unspecified"));
    }

    #[test]
    fn official_env_is_off_until_named() {
        assert!(!official_requested(None));
        assert!(!official_requested(Some("off")));
        assert!(official_requested(Some("1")));
        assert!(official_requested(Some("official")));
        assert!(!official_requested(Some("cache")));
    }

    #[test]
    fn official_numbers_match_the_lingbot_toml() {
        assert_eq!(
            (OFFICIAL_WIDTH, OFFICIAL_HEIGHT, OFFICIAL_FRAMES),
            (832, 480, 121)
        );
        assert_eq!(
            (OFFICIAL_STEPS, OFFICIAL_GUIDANCE, OFFICIAL_SHIFT),
            (40, 3.0, 3.0)
        );
        assert_eq!(
            (REFINER_WIDTH, REFINER_HEIGHT, REFINER_STEPS),
            (1920, 1088, 8)
        );
        assert_eq!((REFINER_T_THRESH, REFINER_SIGMA_TAIL_STEPS), (0.85, 2));
    }
}
