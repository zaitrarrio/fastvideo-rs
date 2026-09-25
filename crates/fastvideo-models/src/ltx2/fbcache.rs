//! LTX-2.5 GB200 stage-1 First Block Cache from
//! `models/ltx25/GB200/ltx_src/ltx_core/opt/step_cache.py` and
//! `models/ltx25/GB200/fullopt.toml` on NVlabs/Sana `sol-engine`.
//!
//! Block 0 always runs. Its residual (`hidden − input`) is the signal.
//! A skip reuses the previous whole-stack video and audio residuals.
//! Stage 2 is out of scope. Default-off:
//! `FASTVIDEO_LTX2_FBCACHE` unset leaves stage-1 dense.

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

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FbDecision {
    pub skip: bool,
    pub reason: &'static str,
    pub distance: Option<f64>,
    pub accumulator: f64,
}

#[derive(Debug, Clone)]
struct PassState {
    has_signal: bool,
    has_residual: bool,
    acc: f64,
    consecutive: usize,
}

impl Default for PassState {
    fn default() -> Self {
        Self {
            has_signal: false,
            has_residual: false,
            acc: 0.0,
            consecutive: 0,
        }
    }
}

/// Stage-1 First-Block Cache. State is keyed by CFG pass ordinal.
#[derive(Debug, Clone)]
pub struct FbCache {
    pub threshold: f64,
    pub warmup: usize,
    pub max_consecutive: usize,
    step_index: usize,
    pass: usize,
    states: Vec<PassState>,
}

impl FbCache {
    pub fn official() -> Self {
        Self {
            threshold: THRESHOLD,
            warmup: WARMUP,
            max_consecutive: MAX_CONSECUTIVE,
            step_index: 0,
            pass: 0,
            states: Vec::new(),
        }
    }

    pub fn begin_step(&mut self, step_index: usize) {
        self.step_index = step_index;
        self.pass = 0;
    }

    pub fn needs_signal(&self) -> bool {
        let state = self.states.get(self.pass);
        match state {
            Some(state) => {
                self.step_index >= self.warmup
                    && state.has_signal
                    && state.has_residual
                    && state.consecutive < self.max_consecutive
            }
            None => false,
        }
    }

    /// `distance` is mean |signal − previous| / mean |previous|.
    /// Ignored when the pass must compute.
    pub fn decide(&mut self, distance: f64) -> FbDecision {
        while self.states.len() <= self.pass {
            self.states.push(PassState::default());
        }
        let p = self.pass;
        self.pass += 1;
        let state = &mut self.states[p];
        let (skip, reason, distance) = if self.step_index < self.warmup {
            (false, "warmup", None)
        } else if !state.has_signal || !state.has_residual {
            (false, "initialize", None)
        } else if state.consecutive >= self.max_consecutive {
            (false, "max_consecutive", None)
        } else {
            state.acc += distance;
            if state.acc < self.threshold {
                (true, "below_threshold", Some(distance))
            } else {
                (false, "threshold", Some(distance))
            }
        };
        state.has_signal = true;
        FbDecision {
            skip,
            reason,
            distance,
            accumulator: state.acc,
        }
    }

    pub fn note_computed(&mut self, pass: usize) {
        let state = &mut self.states[pass];
        state.acc = 0.0;
        state.consecutive = 0;
        state.has_residual = true;
    }

    pub fn note_reused(&mut self, pass: usize) {
        self.states[pass].consecutive += 1;
    }

    pub fn current_pass(&self) -> usize {
        self.pass
    }

    pub fn last_pass(&self) -> usize {
        self.pass.saturating_sub(1)
    }
}

/// Mean |current − previous| / mean |previous|. `inf` when previous is zero.
pub fn relative_l1(current: &[f32], previous: &[f32]) -> f64 {
    let n = current.len().max(1) as f64;
    let mut num = 0.0;
    let mut den = 0.0;
    for (a, b) in current.iter().zip(previous.iter()) {
        num += (f64::from(*a) - f64::from(*b)).abs();
        den += f64::from(*b).abs();
    }
    let den = den / n;
    if den == 0.0 {
        f64::INFINITY
    } else {
        (num / n) / den
    }
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

    #[test]
    fn warmup_then_skip_until_threshold() {
        let mut cache = FbCache::official();
        cache.begin_step(0);
        let first = cache.decide(0.0);
        assert!(!first.skip);
        assert_eq!(first.reason, "warmup");
        cache.note_computed(0);

        cache.begin_step(1);
        let skip = cache.decide(0.03);
        assert!(skip.skip);
        assert_eq!(skip.reason, "below_threshold");
        cache.note_reused(0);

        cache.begin_step(2);
        let hit = cache.decide(0.06);
        assert!(!hit.skip);
        assert_eq!(hit.reason, "threshold");
        assert_eq!(hit.accumulator, 0.09);
        cache.note_computed(0);
        assert_eq!(cache.states[0].acc, 0.0);
        assert_eq!(cache.states[0].consecutive, 0);
    }

    #[test]
    fn cfg_passes_do_not_share_state() {
        let mut cache = FbCache::official();
        cache.begin_step(0);
        assert!(!cache.decide(0.0).skip);
        cache.note_computed(0);
        assert!(!cache.decide(0.0).skip);
        cache.note_computed(1);

        cache.begin_step(1);
        assert!(cache.decide(0.02).skip);
        cache.note_reused(0);
        let uncond = cache.decide(0.09);
        assert!(!uncond.skip);
        assert_eq!(uncond.reason, "threshold");
    }

    #[test]
    fn max_consecutive_forces_a_recompute() {
        let mut cache = FbCache {
            max_consecutive: 1,
            ..FbCache::official()
        };
        cache.begin_step(0);
        cache.decide(0.0);
        cache.note_computed(0);
        cache.begin_step(1);
        assert!(cache.decide(0.01).skip);
        cache.note_reused(0);
        cache.begin_step(2);
        let forced = cache.decide(0.01);
        assert!(!forced.skip);
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
