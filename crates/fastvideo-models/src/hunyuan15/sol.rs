//! Hunyuan Sol-engine gap record.
//!
//! The sol-engine profile (`models/hunyuan_video.toml`) is HunyuanVideo-13B
//! diffusers. Its TeaCache seam feeds `time_text_embed` (`temb`) into the
//! generic controller behind `SGLANG_HQ_TEACACHE_*`. This crate's family is
//! HunyuanVideo 1.5: a different MMDiT whose time path is `time_in` only.
//! The 13B coefficients and the 13B controller are not applied here.

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
}
