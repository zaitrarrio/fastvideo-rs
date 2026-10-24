//! LTX-2.5 GB200 stage-1 First Block Cache from
//! `models/ltx25/GB200/ltx_src/ltx_core/opt/step_cache.py` and
//! `models/ltx25/GB200/fullopt.toml` on NVlabs/Sana `sol-engine`.
//!
//! Block 0 always runs. Its residual (`hidden − input`) is the signal.
//! A skip reuses the previous whole-stack video and audio residuals.
//! Stage 2 is out of scope. Default-off:
//! `FASTVIDEO_LTX2_FBCACHE` unset leaves stage-1 dense.

use crate::techniques::step_cache::distance;
pub use crate::techniques::step_cache::fbcache::FbCacheParams;
pub use crate::techniques::step_cache::FbCache;

/// Residual-diff threshold from the GB200 `fullopt.toml` delivery row.
pub const THRESHOLD: f64 = 0.08;

/// `step_index < warmup` always computes. Published warmup is 1.
pub const WARMUP: usize = 1;

/// Cap consecutive skips. `LTX25_CACHE_MAX_CONSECUTIVE=10` on the GB200 cell.
pub const MAX_CONSECUTIVE: usize = 10;

/// `FASTVIDEO_LTX2_FBCACHE=1` (or `fbcache`) turns the stage-1 skips on.
pub fn requested(value: Option<&str>) -> bool {
    match value.map(str::trim) {
        Some("1") => true,
        Some(v) => v.eq_ignore_ascii_case("fbcache"),
        None => false,
    }
}

/// Where the cache may run. The only reference profile with FBCache is the
/// GB200 dev two-stage (`models/ltx25/GB200/fullopt.toml`): stage 1 of
/// `TI2VidTwoStages` on the dev checkpoint under the multimodal guider. The
/// RTX5090 distilled profile (`RTX5090/run_ltx25_gpu.sh`) and both refiners run
/// without it, and no other version has a cached profile. Anything else is
/// refused: a skip changes the output.
pub fn scope(
    version: super::config::Ltx2ModelVersion,
    distilled: bool,
    guided: bool,
    stage: usize,
) -> Result<(), String> {
    if version == super::config::Ltx2ModelVersion::V25 && !distilled && guided && stage == 1 {
        Ok(())
    } else {
        Err(format!(
            "ltx2 fbcache: only the GB200 LTX-2.5 dev guided stage 1 uses FBCache \
(fullopt.toml); refused for {version:?} {} {} stage {stage}",
            if distilled { "distilled" } else { "dev" },
            if guided { "guided" } else { "unguided" },
        ))
    }
}

pub const APPLIED: &str = "ltx2 sol: GB200 FBCache 0.08 warmup 1 max_consecutive 10 \
(block-0 residual signal, whole-stack residual reuse, stage-1 only)";

/// The GB200 stage-1 FBCache on the one step-cache implementation
/// (ADR-0020): one lane per CFG pass, `step_index < warmup` computes,
/// `reuse = acc < threshold` on the accumulated distance, the accumulator
/// cleared when the pass computes, `consecutive >= max_consecutive` forces
/// a compute.
pub fn official() -> FbCache {
    FbCache::new(FbCacheParams {
        name: "ltx2 fbcache",
        threshold: THRESHOLD,
        warmup: WARMUP,
        max_consecutive: MAX_CONSECUTIVE,
    })
}

/// Mean |current − previous| / mean |previous|. `inf` when previous is zero.
pub fn relative_l1(current: &[f32], previous: &[f32]) -> f64 {
    let (num, den) = distance::abs_sums(current, previous);
    distance::mean_over_mean_or_inf(num, den, current.len())
}

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
    fn knobs_match_the_gb200_fullopt() {
        assert_eq!(THRESHOLD, 0.08);
        assert_eq!(WARMUP, 1);
        assert_eq!(MAX_CONSECUTIVE, 10);
        assert!(APPLIED.contains("block-0 residual"));
    }

    use crate::techniques::step_cache::{Lane, Refresh, Signal, StepCache};

    fn distance(cache: &FbCache, d: f64) -> Signal {
        if cache.needs_signal_next() {
            Signal::Distance(d)
        } else {
            Signal::None
        }
    }

    #[test]
    fn warmup_then_skip_until_threshold() {
        let mut cache = official();
        cache.begin_step(0);
        let first = cache.decide_next(distance(&cache, 0.0)).unwrap();
        assert!(!first.reuse());
        assert_eq!(first.reason, "warmup");
        cache.note_computed(Lane(0), Refresh::None).unwrap();

        cache.begin_step(1);
        let skip = cache.decide_next(distance(&cache, 0.03)).unwrap();
        assert!(skip.reuse());
        assert_eq!(skip.reason, "below_threshold");
        cache.note_reused(Lane(0));

        cache.begin_step(2);
        let hit = cache.decide_next(distance(&cache, 0.06)).unwrap();
        assert!(!hit.reuse());
        assert_eq!(hit.reason, "threshold");
        assert_eq!(hit.accumulator, 0.03 + 0.06);
        cache.note_computed(Lane(0), Refresh::None).unwrap();
        cache.begin_step(3);
        let fresh = cache.decide_next(distance(&cache, 0.01)).unwrap();
        assert_eq!(fresh.accumulator, 0.01, "the compute cleared the pass");
    }

    #[test]
    fn cfg_passes_do_not_share_state() {
        let mut cache = official();
        cache.begin_step(0);
        assert!(!cache.decide_next(distance(&cache, 0.0)).unwrap().reuse());
        cache.note_computed(Lane(0), Refresh::None).unwrap();
        assert!(!cache.decide_next(distance(&cache, 0.0)).unwrap().reuse());
        cache.note_computed(Lane(1), Refresh::None).unwrap();

        cache.begin_step(1);
        assert!(cache.decide_next(distance(&cache, 0.02)).unwrap().reuse());
        cache.note_reused(Lane(0));
        let uncond = cache.decide_next(distance(&cache, 0.09)).unwrap();
        assert!(!uncond.reuse());
        assert_eq!(uncond.reason, "threshold");
    }

    #[test]
    fn max_consecutive_forces_a_recompute() {
        let mut cache = FbCache::new(FbCacheParams {
            name: "ltx2 fbcache",
            threshold: THRESHOLD,
            warmup: WARMUP,
            max_consecutive: 1,
        });
        cache.begin_step(0);
        cache.decide_next(distance(&cache, 0.0)).unwrap();
        cache.note_computed(Lane(0), Refresh::None).unwrap();
        cache.begin_step(1);
        assert!(cache.decide_next(distance(&cache, 0.01)).unwrap().reuse());
        cache.note_reused(Lane(0));
        cache.begin_step(2);
        let forced = cache.decide_next(distance(&cache, 0.01)).unwrap();
        assert!(!forced.reuse());
        assert_eq!(forced.reason, "max_consecutive");
    }

    #[test]
    fn relative_l1_is_mean_abs_over_mean_abs() {
        let prev = [2.0f32, 0.0];
        let cur = [4.0f32, 2.0];
        assert!((relative_l1(&cur, &prev) - 2.0).abs() < 1e-12);
        assert!(relative_l1(&[1.0], &[0.0]).is_infinite());
    }

    #[test]
    fn fbcache_is_only_the_gb200_dev_guided_stage_one() {
        use crate::ltx2::config::Ltx2ModelVersion as V;
        assert!(scope(V::V25, false, true, 1).is_ok());
        // RTX5090 distilled two-stage: none on either stage.
        assert!(scope(V::V25, true, false, 1).is_err());
        assert!(scope(V::V25, true, false, 2).is_err());
        // Stage 2 and the refiners: none.
        assert!(scope(V::V25, false, false, 2).is_err());
        assert!(scope(V::V25, false, true, 2).is_err());
        assert!(scope(V::V23, false, true, 1).is_err());
        assert!(scope(V::V20, false, true, 1).is_err());
    }
}
