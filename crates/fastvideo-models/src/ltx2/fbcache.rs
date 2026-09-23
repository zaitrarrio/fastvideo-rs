//! LTX-2.5 GB200 multi-step stage-1 First Block Cache from
//! `models/ltx25/README.md` on NVlabs/Sana `sol-engine`.
//!
//! The published cell names FBCache 0.08 on the 30-step CFG first stage.
//! The compared tensor, the reused payload, and the warmup/cooldown window
//! are not in the sol-engine snapshots, so this module records the
//! threshold and leaves stage-1 dense.

/// Residual-diff threshold from the GB200 multi-step delivery row.
pub const THRESHOLD: f64 = 0.08;

/// `FASTVIDEO_LTX2_FBCACHE=1` (or `fbcache`) logs [`GAP`] and stays dense.
pub fn requested(value: Option<&str>) -> bool {
    match value.map(str::trim) {
        Some("1") => true,
        Some(v) => v.eq_ignore_ascii_case("fbcache"),
        None => false,
    }
}

pub const GAP: &str = "ltx2 sol: GB200 FBCache 0.08 stays unported \
(the compared tensor, reused payload, and warmup/cooldown are unpublished \
in the sol-engine snapshots)";

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn env_is_off_by_default() {
        assert!(!requested(None));
        assert!(!requested(Some("")));
        assert!(!requested(Some("off")));
        assert!(requested(Some("1")));
        assert!(requested(Some("fbcache")));
        assert!(requested(Some("FBCache")));
    }

    #[test]
    fn threshold_matches_the_gb200_row() {
        assert_eq!(THRESHOLD, 0.08);
        assert!(GAP.contains("unpublished"));
    }
}
