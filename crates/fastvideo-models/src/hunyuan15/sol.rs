//! Hunyuan Sol-engine sampling and gap record.
//!
//! The sol-engine profile (`models/hunyuan_video.toml`) is HunyuanVideo-13B
//! diffusers. Official sampling is 1280×720 / 129f / 50 steps / guidance 6.
//! This crate's family is HunyuanVideo 1.5: a different MMDiT whose time path
//! is `time_in` only. The 720p preset can carry the canvas and step count.
//! Guidance 6 is recorded; hunyuan15 has no CFG pair. The 13B TeaCache
//! controller and `time_text_embed` coefficients are not applied.

/// Official HunyuanVideo sampling numbers from `models/hunyuan_video.toml`.
pub const OFFICIAL_WIDTH: usize = 1280;
pub const OFFICIAL_HEIGHT: usize = 720;
pub const OFFICIAL_FRAMES: usize = 129;
pub const OFFICIAL_STEPS: usize = 50;
/// Recorded from the 13B profile. hunyuan15 generate has no CFG.
pub const OFFICIAL_GUIDANCE: f32 = 6.0;

/// `FASTVIDEO_HUNYUAN15_OFFICIAL=1` (or `official`) applies the published
/// 1280×720 / 129f / 50-step canvas. Unset leaves existing request defaults.
pub fn official_requested(value: Option<&str>) -> bool {
    match value.map(str::trim) {
        Some("1") => true,
        Some(v) => v.eq_ignore_ascii_case("official"),
        None => false,
    }
}

/// `FASTVIDEO_HUNYUAN15_SOL=teacache` (or `1`) logs the gap and stays dense.
pub fn teacache_requested(value: Option<&str>) -> bool {
    match value.map(str::trim) {
        Some("1") => true,
        Some(v) => v.eq_ignore_ascii_case("teacache"),
        None => false,
    }
}

pub const GAP: &str = "hunyuan15 sol: HunyuanVideo-13B TeaCache stays unported \
(time_text_embed controller and coefficients belong to a different model family)";

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn env_is_off_by_default() {
        assert!(!teacache_requested(None));
        assert!(!teacache_requested(Some("")));
        assert!(!teacache_requested(Some("off")));
        assert!(teacache_requested(Some("1")));
        assert!(teacache_requested(Some("teacache")));
    }

    #[test]
    fn gap_names_the_family_mismatch() {
        assert!(GAP.contains("HunyuanVideo-13B"));
        assert!(GAP.contains("different model family"));
    }

    #[test]
    fn official_env_is_off_until_named() {
        assert!(!official_requested(None));
        assert!(!official_requested(Some("teacache")));
        assert!(official_requested(Some("1")));
        assert!(official_requested(Some("official")));
        assert_eq!(
            (OFFICIAL_WIDTH, OFFICIAL_HEIGHT, OFFICIAL_FRAMES),
            (1280, 720, 129)
        );
        assert_eq!((OFFICIAL_STEPS, OFFICIAL_GUIDANCE), (50, 6.0));
    }
}
