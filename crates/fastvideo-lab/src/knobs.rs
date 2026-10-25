//! Script settings fv-lab does not read, refused instead of ignored
//! (docs/adr/0019-fv-lab.md §9.4, §10, §11).
//!
//! `runpod-matrix.sh` families read knobs from the environment that change
//! their arms (`FV_PRECISION_ARM`, `FV_VAE_*`, …). No tier reads them: a
//! tier's arms are its cells. A knob set for an fv-lab run would be ignored
//! and the run would silently differ from what was asked, so `fv-lab run`
//! (its environment) and `fv-lab pod up --extra-env` refuse them, as
//! `runpod-http.sh` does for its `FV_EXTRA_ENV` (`FV_LAB_NOT_READ` there;
//! `tests/knobs.rs` holds the two lists equal). Two of the dropped modes are
//! tiers of their own since §11 ([`RESTORED_MODES`], `FV_LAB_RESTORED` of
//! `scripts/gpu/fv-lab-retired.sh`): their setting is answered with the tier.

/// A mode of a retired `runpod-matrix.sh` family that is an fv-lab tier now
/// (ADR-0019 §11).
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

/// The retired families' knobs that changed their arms (ADR-0019 §9.4): no
/// tier reads them.
pub const MODE_KNOBS: [&str; 10] = [
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
];

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
    MODE_KNOBS.into_iter().chain(START_SET)
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
                "{key}={value}: the tier {} is the {} family's {key}={value} mode already; drop the setting (ADR-0019 §11)",
                m.tier, m.family
            )
        } else {
            format!(
                "{key}={value} was a mode of runpod-matrix.sh {}; it is the fv-lab tier {} now: run {} (fv-lab run {} / pod up --family {}), without {key}={value} (ADR-0019 §11)",
                m.family, m.tier, m.tier, m.tier, m.tier
            )
        });
    }
    if BY_ARM.contains(&key) {
        return Some(format!(
            "{key}: the restored tiers (precision-nvfp4, ltxvae-gen) have one cell per workload; narrow them by arm with --cells / FV_CELLS (ADR-0019 §11)"
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

/// `fv-lab run`: the [`MODE_KNOBS`] set (non-empty) in its environment.
/// `get` reads one variable (the process environment in the binary).
pub fn check_run_env(tier: &str, get: impl Fn(&str) -> Option<String>) -> Result<(), String> {
    let found = MODE_KNOBS
        .iter()
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
            &["FV_VAE_TRACE=0 FASTVIDEO_TEST_X=1".into(), "FV_VAE_GEN=1".into()],
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
            &["FASTVIDEO_TEST_X=1".into(), "FV_BASELINE_SHA=abcdef1".into()]
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
    fn not_read_is_both_lists() {
        assert_eq!(not_read().count(), MODE_KNOBS.len() + START_SET.len());
        for m in RESTORED_MODES {
            assert!(MODE_KNOBS.contains(&m.key), "{}", m.key);
        }
    }
}
