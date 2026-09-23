//! LingBot-Video Sol-engine gap record.
//!
//! `models/lingbot_video.toml` lists technique names (`kernel`, `cache`,
//! `pisa`, `topology`) on a CP4+FSDP+FA2 baseline. No cache signal, PISA
//! sparsity, or skip mask is published in the sol-engine snapshots, so this
//! family stays dense.

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
}
