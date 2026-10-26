//! Script settings fv-lab does not read, refused instead of ignored
//! (docs/adr/0019-fv-lab.md §9.4, §10, §12).
//!
//! `runpod-matrix.sh` families read knobs from the environment that change
//! their arms (`FV_PRECISION_ARM`, `FV_VAE_*`, …). No tier reads them: a
//! tier's arms are its cells. A knob set for an fv-lab run would be ignored
//! and the run would silently differ from what was asked, so `fv-lab run`
//! (its environment) and `fv-lab pod up --extra-env` refuse them, as
//! `runpod-http.sh` does for its `FV_EXTRA_ENV` (`FV_LAB_NOT_READ` there;
//! `tests/knobs.rs` holds the two lists equal). Two of the dropped modes are
//! tiers of their own since §12 ([`RESTORED_MODES`], `FV_LAB_RESTORED` of
//! `scripts/gpu/fv-lab-retired.sh`): their setting is answered with the tier.

/// A mode of a retired `runpod-matrix.sh` family that is an fv-lab tier now
/// (ADR-0019 §12).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RestoredMode {
    /// The tier that is the mode.
    pub tier: &'static str,
    /// The script family whose mode it was.
    pub family: &'static str,
    /// The setting that selected the mode (`<key>=<value>`).
    pub key: &'static str,
    pub value: &'static str,
}

/// `FV_LAB_RESTORED` of `scripts/gpu/fv-lab-retired.sh`, in its order.
pub const RESTORED_MODES: [RestoredMode; 2] = [
    RestoredMode {
        tier: "precision-nvfp4",
        family: "precision",
        key: "FV_PRECISION_ARM",
        value: "ltx-nvfp4",
    },
    RestoredMode {
        tier: "ltxvae-gen",
        family: "ltxvae",
        key: "FV_VAE_GEN",
        value: "1",
    },
];

/// The retired families' knobs that changed their arms (ADR-0019 §9.4) or
/// picked their pairs or groups (`FV_TECH_CELLS`, `FV_B6_CELLS`,
/// `FV_B4A_CELLS`, `FV_DET_CELLS`): no tier reads them. `FV_B4A_CELLS` joined
/// when `b4a` retired (§11.6); with it ignored, every `b4a` cell would run, at
/// three times the cost of the one asked for. The five `FV_DET_*` joined when
/// `det-short` and `determinism` retired (§15.10): their tiers are the
/// script's defaults. `FV_ATTN3_ARMS` and `FV_ATTN3_VSA` (attn3's arm
/// pickers) joined when `attn3` retired (§17.6): the tier is their defaults
/// (`v2 cudnn`, `0 1`), selected by arm.
pub const MODE_KNOBS: [&str; 18] = [
    "FV_PRECISION_ARM",
    "FV_NVFP4_WORKLOADS",
    "FV_VAE_GEN",
    "FV_VAE_TRACE",
    "FV_VAE_UNTRACED",
    "FV_VAE_WORKLOADS",
    "FV_LTXFPS",
    "FV_LTXFPS_FRAMES",
    "FV_TECH_CELLS",
    "FV_B6_CELLS",
    "FV_B4A_CELLS",
    "FV_DET_CELLS",
    "FV_DET_REPEATS",
    "FV_DET_PROMPT",
    "FV_DET_T2V_PROMPT",
    "FV_DET_I2V_PROMPT",
    "FV_ATTN3_ARMS",
    "FV_ATTN3_VSA",
];

/// The knobs of the tiers whose `runpod-matrix.sh` branch is still there:
/// the script reads them, no tier does. fv-lab refuses them; `runpod-http.sh`'s
/// `FV_LAB_NOT_READ` gains them in the PR that retires the branch (as the
/// `FV_DET_*` knobs did with `det-short` and `determinism`, §15.10, and
/// `FV_ATTN3_*` with `attn3`, §17.6).
///
/// `oracle` (ADR-0019 §15), whose branch stays whole (§15.10: its
/// clip-metric targets, `sfwan13` and `scripts/gpu/oracle.sh`'s runtime side
/// still run there): `FV_ORACLE_*`, `FV_REF2VA_PROMPT`, and the prompts of
/// the clip-metric targets (§20: the tier's extra cells are their defaults):
/// `FV_LTX_REF_PROMPT`, `FV_LTX_A2V_PROMPT`, `FV_LTX_A2V_I2V_PROMPT`,
/// `FV_LTX_RETAKE_PROMPT`, `FV_LTX_EXTEND_PROMPT`.
pub const SCRIPT_KNOBS: [&str; 10] = [
    "FV_ORACLE_TARGETS",
    "FV_ORACLE_F32",
    "FV_ORACLE_OWN_TEXT",
    "FV_ORACLE_PREFETCH",
    "FV_REF2VA_PROMPT",
    "FV_LTX_REF_PROMPT",
    "FV_LTX_A2V_PROMPT",
    "FV_LTX_A2V_I2V_PROMPT",
    "FV_LTX_RETAKE_PROMPT",
    "FV_LTX_EXTEND_PROMPT",
];

/// The knobs of a step dropped from a live family's branch, which neither
/// the branch nor any tier reads any more: `hd`'s post-script hook
/// (`FV_HD_POST_URL`, a script fetched and run on the pod, and its cap
/// `FV_HD_POST_TIMEOUT_S`), dropped with the family's derived step when `hd`
/// became a tier (ADR-0019 §22, S2). Refused everywhere, as runpod-http.sh's
/// `FV_LAB_NOT_READ` (and its script path) refuses them.
pub const DROPPED_KNOBS: [&str; 2] = ["FV_HD_POST_URL", "FV_HD_POST_TIMEOUT_S"];

/// The blocks the oracle cells dump (`FASTVIDEO_DUMP_OPS` in every arm's
/// `env`): the script's default. Another value for the `oracle` tier would be
/// overridden by the cells, so it is refused.
pub const ORACLE_DUMP_OPS: &str = "0,1,24,47";

/// The knobs that picked workloads or arm groups of a restored mode: one cell
/// per workload in the tier, so `--cells` / `FV_CELLS` by arm replaces them.
const BY_ARM: [&str; 3] = ["FV_NVFP4_WORKLOADS", "FV_VAE_WORKLOADS", "FV_VAE_UNTRACED"];

/// What `--extra-env` may not carry besides [`MODE_KNOBS`]: the prompts
/// (a cell's workload, ADR-0019 §7) and what the pod's start command sets
/// after `FV_EXTRA_ENV` (so the entry would never reach the run).
pub const START_SET: [&str; 8] = [
    "FV_PROMPTS",
    "FV_PROMPT",
    "FV_SEED",
    "FV_WORK",
    "FV_SCRATCH",
    "FV_RUN_TAG",
    "FV_CELLS",
    "FV_GEN_TIMEOUT_S",
];

/// Every key `fv-lab pod up --extra-env` refuses: `runpod-http.sh`'s
/// `FV_LAB_NOT_READ`, in its order.
pub fn not_read() -> impl Iterator<Item = &'static str> {
    MODE_KNOBS
        .into_iter()
        .chain(DROPPED_KNOBS)
        .chain(START_SET)
}

/// The restored mode `key=value` selects, if any.
pub fn restored_mode(key: &str, value: &str) -> Option<&'static RestoredMode> {
    RESTORED_MODES
        .iter()
        .find(|m| m.key == key && m.value == value)
}

/// Why `key=value` is refused for a run of `tier`, or `None`. `start_set`:
/// also refuse [`START_SET`] (`--extra-env`; `fv-lab run` reads those as its
/// own inputs, and only warns about the prompts, §7).
fn why(tier: Option<&str>, key: &str, value: &str, start_set: bool) -> Option<String> {
    if let Some(m) = restored_mode(key, value) {
        return Some(if tier == Some(m.tier) {
            format!(
                "{key}={value}: the tier {} is the {} family's {key}={value} mode already; drop the setting (ADR-0019 §12)",
                m.tier, m.family
            )
        } else {
            format!(
                "{key}={value} was a mode of runpod-matrix.sh {}; it is the fv-lab tier {} now: run {} (fv-lab run {} / pod up --family {}), without {key}={value} (ADR-0019 §12)",
                m.family, m.tier, m.tier, m.tier, m.tier
            )
        });
    }
    if key == "FV_B4A_CELLS" {
        return Some(
            "FV_B4A_CELLS: fv-lab selects by arm: pass --cells <cell>-base <cell>-pr (FV_CELLS), not FV_B4A_CELLS"
                .to_owned(),
        );
    }
    if key == "FV_DET_CELLS" || key == "FV_ORACLE_TARGETS" {
        return Some(format!(
            "{key}: fv-lab selects by arm: pass --cells (FV_CELLS) with the arms of those groups or targets, one cell each (ADR-0019 §15)"
        ));
    }
    if key == "FV_ORACLE_URL" && tier == Some("determinism") {
        return Some(
            "FV_ORACLE_URL: determinism no longer runs the oracle first; run the oracle on its own (scripts/gpu/oracle.sh; ADR-0019 §15.5, §15.10)"
                .to_owned(),
        );
    }
    if key == "FV_ATTN3_ARMS" || key == "FV_ATTN3_VSA" {
        return Some(format!(
            "{key}: the attn3 tier runs the script's default arms (FV_ATTN3_ARMS \"v2 cudnn\", FV_ATTN3_VSA \"0 1\"); narrow it by arm with --cells (FV_CELLS) (ADR-0019 §17.2, §17.6)"
        ));
    }
    if DROPPED_KNOBS.contains(&key) {
        return Some(format!(
            "{key}: the hd family's post-script hook was dropped with its derived step (the Lanczos compares, the keyframe prune); nothing runs a fetched script (ADR-0019 §22, S2)"
        ));
    }
    if key == "FV_ORACLE_PREFETCH" {
        return Some(
            "FV_ORACLE_PREFETCH: fv-lab always fetches the references in the background (ADR-0019 §15)"
                .to_owned(),
        );
    }
    if SCRIPT_KNOBS.contains(&key) || (key.starts_with("FV_DET_") && MODE_KNOBS.contains(&key)) {
        return Some(format!(
            "{key}: no tier reads it; the det-short, determinism and oracle tiers are the script's defaults (ADR-0019 §15)"
        ));
    }
    if key == "FASTVIDEO_DUMP_OPS" && tier == Some("oracle") && value != ORACLE_DUMP_OPS {
        return Some(format!(
            "FASTVIDEO_DUMP_OPS={value}: the oracle cells dump blocks {ORACLE_DUMP_OPS} (each arm sets it; ADR-0019 §15)"
        ));
    }
    if BY_ARM.contains(&key) {
        return Some(format!(
            "{key}: the restored tiers (precision-nvfp4, ltxvae-gen) have one cell per workload; narrow them by arm with --cells / FV_CELLS (ADR-0019 §12)"
        ));
    }
    if MODE_KNOBS.contains(&key) {
        return Some(format!(
            "{key}: no tier reads it; it changed a retired family's arms, and a tier's arms are its cells (ADR-0019 §9.4, §10)"
        ));
    }
    if start_set && START_SET.contains(&key) {
        return Some(format!(
            "{key}: the pod's start command sets it after --extra-env (or a cell's workload does); use the flag (ADR-0019 §7, §10)"
        ));
    }
    None
}

/// One message for every refused entry, the restored modes' first (so the
/// answer that names a tier is never hidden behind another refusal).
fn message(mut found: Vec<(bool, String)>) -> Result<(), String> {
    if found.is_empty() {
        return Ok(());
    }
    found.sort_by_key(|(mode, _)| !*mode);
    Err(format!(
        "refused, fv-lab would not read {}: {}",
        if found.len() == 1 { "it" } else { "them" },
        found
            .into_iter()
            .map(|(_, w)| w)
            .collect::<Vec<_>>()
            .join("; ")
    ))
}

/// `fv-lab pod up --extra-env`: every `K=V` entry (an entry may hold several,
/// space-separated, as `FV_EXTRA_ENV` does) whose key no tier reads.
pub fn check_extra_env(tier: &str, entries: &[String]) -> Result<(), String> {
    let found = entries
        .iter()
        .flat_map(|e| e.split_whitespace())
        .filter_map(|e| {
            let (k, v) = e.split_once('=').unwrap_or((e, ""));
            why(Some(tier), k, v, true)
                .map(|w| (restored_mode(k, v).is_some(), format!("--extra-env {w}")))
        })
        .collect();
    message(found)
}

/// `fv-lab run`: the [`MODE_KNOBS`], [`SCRIPT_KNOBS`] and [`DROPPED_KNOBS`] set (non-empty) in
/// its environment, the oracle's `FASTVIDEO_DUMP_OPS`, and `FV_ORACLE_URL`
/// for `determinism` (which the script's branch read to run the oracle
/// first; the tier does not, ADR-0019 §15.10). `get` reads one variable (the
/// process environment in the binary).
pub fn check_run_env(tier: &str, get: impl Fn(&str) -> Option<String>) -> Result<(), String> {
    let found = MODE_KNOBS
        .iter()
        .chain(&SCRIPT_KNOBS)
        .chain(&DROPPED_KNOBS)
        .chain(&["FASTVIDEO_DUMP_OPS", "FV_ORACLE_URL"])
        .filter_map(|k| get(k).filter(|v| !v.is_empty()).map(|v| (*k, v)))
        .filter_map(|(k, v)| {
            why(Some(tier), k, &v, false).map(|w| (restored_mode(k, &v).is_some(), w))
        })
        .collect();
    message(found)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_restored_mode_names_its_tier_first() {
        let e = check_extra_env(
            "precision",
            &[
                "FV_VAE_TRACE=0 FASTVIDEO_TEST_X=1".into(),
                "FV_VAE_GEN=1".into(),
            ],
        )
        .unwrap_err();
        let tier = e.find("it is the fv-lab tier ltxvae-gen now: run ltxvae-gen");
        let other = e.find("FV_VAE_TRACE:");
        assert!(tier.is_some() && tier < other, "{e}");
        assert!(
            e.contains("--extra-env FV_VAE_TRACE: no tier reads it"),
            "{e}"
        );
        assert!(!e.contains("FASTVIDEO_TEST_X"), "{e}");
    }

    #[test]
    fn the_tier_itself_drops_the_setting() {
        let e =
            check_extra_env("precision-nvfp4", &["FV_PRECISION_ARM=ltx-nvfp4".into()]).unwrap_err();
        assert!(e.contains("is the precision family's FV_PRECISION_ARM=ltx-nvfp4 mode already"));
    }

    #[test]
    fn other_values_and_keys() {
        assert!(check_extra_env(
            "ltxvae",
            &[
                "FASTVIDEO_TEST_X=1".into(),
                "FV_BASELINE_SHA=abcdef1".into()
            ]
        )
        .is_ok());
        let e = check_extra_env("ltxvae", &["FV_VAE_GEN=0".into()]).unwrap_err();
        assert!(e.contains("FV_VAE_GEN: no tier reads it"), "{e}");
        let e = check_extra_env("ltxvae-gen", &["FV_VAE_WORKLOADS=4k5s".into()]).unwrap_err();
        assert!(e.contains("narrow them by arm"), "{e}");
        for k in START_SET {
            let e = check_extra_env("rtx6000", &[format!("{k}=x")]).unwrap_err();
            assert!(e.contains("start command sets it"), "{k}: {e}");
        }
    }

    #[test]
    fn run_refuses_only_the_mode_knobs() {
        let env = |pairs: &'static [(&'static str, &'static str)]| {
            move |k: &str| {
                pairs
                    .iter()
                    .find(|(x, _)| *x == k)
                    .map(|(_, v)| (*v).to_owned())
            }
        };
        assert!(check_run_env("precision", env(&[("FV_WORK", "/w"), ("FV_PROMPTS", "5")])).is_ok());
        assert!(check_run_env("precision", env(&[("FV_PRECISION_ARM", "")])).is_ok());
        let e = check_run_env("precision", env(&[("FV_PRECISION_ARM", "ltx-nvfp4")])).unwrap_err();
        assert!(
            e.contains("it is the fv-lab tier precision-nvfp4 now"),
            "{e}"
        );
        let e =
            check_run_env("ltxvae", env(&[("FV_B6_CELLS", "x"), ("FV_VAE_GEN", "1")])).unwrap_err();
        assert!(
            e.starts_with("refused, fv-lab would not read them: FV_VAE_GEN=1 was a mode"),
            "{e}"
        );
    }

    #[test]
    fn b4a_cells_is_refused_by_arm() {
        let e = check_extra_env("b4a", &["FV_B4A_CELLS=b4a-h3-4step-tea".into()]).unwrap_err();
        assert!(e.contains("not FV_B4A_CELLS"), "{e}");
        let env = |k: &str| (k == "FV_B4A_CELLS").then(|| "x".to_owned());
        let e = check_run_env("b4a", env).unwrap_err();
        assert!(e.contains("not FV_B4A_CELLS"), "{e}");
        // Retired with the b4a branch (ADR-0019 §11.6): runpod-http.sh's
        // FV_LAB_NOT_READ refuses it too (tests/knobs.rs holds them equal).
        assert!(not_read().any(|k| k == "FV_B4A_CELLS"));
    }

    #[test]
    fn the_retired_determinism_families_knobs_are_not_read() {
        let e = check_extra_env("det-short", &["FV_DET_CELLS=h3".into()]).unwrap_err();
        assert!(e.contains("selects by arm"), "{e}");
        let e = check_extra_env("determinism", &["FV_DET_T2V_PROMPT=x".into()]).unwrap_err();
        assert!(e.contains("FV_DET_T2V_PROMPT: no tier reads it"), "{e}");
        let env = |k: &str| (k == "FV_DET_REPEATS").then(|| "5".to_owned());
        let e = check_run_env("det-short", env).unwrap_err();
        assert!(e.contains("FV_DET_REPEATS: no tier reads it"), "{e}");
        let e = check_extra_env(
            "determinism",
            &["FV_ORACLE_URL=https://x/upstream/t".into()],
        )
        .unwrap_err();
        assert!(e.contains("no longer runs the oracle first"), "{e}");
        assert!(check_extra_env("oracle", &["FV_ORACLE_URL=https://x/upstream/t".into()]).is_ok());
        // `fv-lab run` with FV_ORACLE_URL in its environment: refused for
        // determinism (it would be dropped), read by oracle.
        let url = |k: &str| (k == "FV_ORACLE_URL").then(|| "https://x/upstream/t".to_owned());
        let e = check_run_env("determinism", url).unwrap_err();
        assert!(e.contains("no longer runs the oracle first"), "{e}");
        assert!(check_run_env("oracle", url).is_ok());
        assert!(check_run_env("det-short", url).is_ok());
        // Retired with their branches (ADR-0019 §15.10): runpod-http.sh's
        // FV_LAB_NOT_READ refuses them too (tests/knobs.rs holds them equal).
        for k in [
            "FV_DET_CELLS",
            "FV_DET_REPEATS",
            "FV_DET_PROMPT",
            "FV_DET_T2V_PROMPT",
            "FV_DET_I2V_PROMPT",
        ] {
            assert!(not_read().any(|n| n == k), "{k}");
        }
    }

    #[test]
    fn the_oracle_knobs_are_refused_by_fv_lab_only() {
        let e = check_extra_env("oracle", &["FV_ORACLE_F32=1".into()]).unwrap_err();
        assert!(e.contains("no tier reads it"), "{e}");
        assert!(check_extra_env("oracle", &["FV_ORACLE_URL=https://x/upstream/t".into()]).is_ok());
        assert!(check_extra_env("oracle", &["FASTVIDEO_DUMP_OPS=0,1,24,47".into()]).is_ok());
        let e = check_extra_env("oracle", &["FASTVIDEO_DUMP_OPS=0,1".into()]).unwrap_err();
        assert!(e.contains("dump blocks 0,1,24,47"), "{e}");
        assert!(check_extra_env("rtx6000", &["FASTVIDEO_DUMP_OPS=0,1".into()]).is_ok());
        let env = |k: &str| (k == "FV_ORACLE_PREFETCH").then(|| "0".to_owned());
        let e = check_run_env("oracle", env).unwrap_err();
        assert!(e.contains("always fetches"), "{e}");
        // The script's oracle branch still reads them (ADR-0019 §15.10):
        // runpod-http.sh's list gains them when it goes (tests/knobs.rs
        // holds that list to not_read()).
        assert!(SCRIPT_KNOBS.iter().all(|k| !not_read().any(|n| n == *k)));
    }

    #[test]
    fn the_retired_attn3_knobs_are_not_read() {
        // attn3's arm pickers: refused by fv-lab, by arm (ADR-0019 §17.2),
        // for every tier, in --extra-env and in fv-lab run's environment.
        let e = check_extra_env("attn3", &["FV_ATTN3_ARMS=v2".into()]).unwrap_err();
        assert!(e.contains("narrow it by arm"), "{e}");
        let env = |k: &str| (k == "FV_ATTN3_VSA").then(|| "1".to_owned());
        let e = check_run_env("attn3", env).unwrap_err();
        assert!(e.contains("FV_ATTN3_VSA: the attn3 tier runs"), "{e}");
        let e = check_extra_env("fuse", &["FV_ATTN3_VSA=0".into()]).unwrap_err();
        assert!(e.contains("FV_ATTN3_VSA: the attn3 tier runs"), "{e}");
        // Retired with the branch (§17.6): out of SCRIPT_KNOBS, into
        // MODE_KNOBS, so runpod-http.sh's FV_LAB_NOT_READ refuses them too
        // (tests/knobs.rs holds them equal).
        for k in ["FV_ATTN3_ARMS", "FV_ATTN3_VSA"] {
            assert!(!SCRIPT_KNOBS.contains(&k), "{k}");
            assert!(MODE_KNOBS.contains(&k), "{k}");
            assert!(not_read().any(|n| n == k), "{k}");
        }
    }

    #[test]
    fn the_dropped_hd_post_script_knobs_are_not_read() {
        // hd's post-script hook left the script with its derived step
        // (ADR-0019 §22, S2): no tier and no branch reads them, so fv-lab
        // refuses them for every tier, in --extra-env and in fv-lab run's
        // environment, and runpod-http.sh's FV_LAB_NOT_READ does too
        // (tests/knobs.rs holds them equal).
        for tier in ["hd", "hd5", "rtx6000"] {
            let e = check_extra_env(
                tier,
                &["FV_HD_POST_URL=file:///fvscratch/files/hd-upscaler.sh".into()],
            )
            .unwrap_err();
            assert!(
                e.contains("--extra-env FV_HD_POST_URL: the hd family's post-script hook was dropped"),
                "{e}"
            );
        }
        let env = |k: &str| (k == "FV_HD_POST_TIMEOUT_S").then(|| "5400".to_owned());
        let e = check_run_env("hd", env).unwrap_err();
        assert!(e.contains("FV_HD_POST_TIMEOUT_S: the hd family's"), "{e}");
        assert!(e.contains("ADR-0019 §22"), "{e}");
        for k in DROPPED_KNOBS {
            assert!(not_read().any(|n| n == k), "{k}");
            assert!(!MODE_KNOBS.contains(&k) && !SCRIPT_KNOBS.contains(&k), "{k}");
        }
    }

    #[test]
    fn not_read_is_both_lists() {
        assert_eq!(
            not_read().count(),
            MODE_KNOBS.len() + DROPPED_KNOBS.len() + START_SET.len()
        );
        for m in RESTORED_MODES {
            assert!(MODE_KNOBS.contains(&m.key), "{}", m.key);
        }
    }
}
