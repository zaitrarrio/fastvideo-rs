//! The restored script modes (docs/adr/0019-fv-lab.md §11): `FV_LAB_RESTORED`
//! of `scripts/gpu/fv-lab-retired.sh`, `<tier>:<family>:<K=V>`. Their
//! fixtures are what the script wrote in that mode at 5987e6a4
//! (`scripts/gpu/tests/fv-lab-restored-goldens.sh`), so they carry the
//! family's name where fv-lab writes the tier's, and the mode's setting
//! where fv-lab has none. These maps are the only differences allowed.
#![allow(dead_code)]

use std::path::PathBuf;

/// One `<tier>:<family>:<K=V>` entry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Restored {
    pub tier: String,
    pub family: String,
    pub mode: String,
}

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// `FV_LAB_RESTORED` of `scripts/gpu/fv-lab-retired.sh`.
pub fn restored() -> Vec<Restored> {
    let p = repo_root().join("scripts/gpu/fv-lab-retired.sh");
    let t = std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("{}: {e}", p.display()));
    let line = t
        .lines()
        .find_map(|l| l.strip_prefix("FV_LAB_RESTORED=\""))
        .and_then(|l| l.strip_suffix('"'))
        .expect("FV_LAB_RESTORED=\"…\"");
    line.split_whitespace()
        .map(|e| {
            let mut it = e.splitn(3, ':');
            let (tier, family, mode) = (it.next(), it.next(), it.next());
            match (tier, family, mode) {
                (Some(t), Some(f), Some(m))
                    if !t.is_empty() && !f.is_empty() && m.contains('=') =>
                {
                    Restored {
                        tier: t.into(),
                        family: f.into(),
                        mode: m.into(),
                    }
                }
                _ => panic!("FV_LAB_RESTORED entry `{e}` is not <tier>:<family>:<K=V>"),
            }
        })
        .collect()
}

/// The restored mode a tier is, if it is one.
pub fn restored_tier(tier: &str) -> Option<Restored> {
    restored().into_iter().find(|r| r.tier == tier)
}

impl Restored {
    /// A dry-run line of the script in this mode, as fv-lab's tier writes
    /// it: the `[<family>] ` prefix and the run directory
    /// `/runs/<family>/` carry the tier's name.
    pub fn dry_run_line(&self, line: &str) -> String {
        let prefix = format!("[{}] ", self.family);
        let rest = line
            .strip_prefix(&prefix)
            .unwrap_or_else(|| panic!("{}: `{line}` lacks {prefix:?}", self.tier));
        format!("[{}] {}", self.tier, self.runs(rest))
    }

    /// `/runs/<family>/` as `/runs/<tier>/`.
    pub fn runs(&self, s: &str) -> String {
        s.replace(
            &format!("/runs/{}/", self.family),
            &format!("/runs/{}/", self.tier),
        )
    }

    /// The start command of `runpod-http.sh plan` in this mode, as fv-lab's
    /// pod for the tier would carry it under the script's runner line: the
    /// run directory and the runner's family named by the tier, and the
    /// mode's `FV_EXTRA_ENV` gone from both `env` lines (fv-lab passes none).
    pub fn start_command(&self, cmd: &str) -> String {
        let with_mode = format!("env {} ", self.mode);
        assert_eq!(
            cmd.matches(&with_mode).count(),
            2,
            "{}: the mode reaches the kernels and the run line",
            self.tier
        );
        let script = |f: &str| format!("scripts/gpu/runpod-matrix.sh {f} >");
        assert_eq!(
            cmd.matches(&script(&self.family)).count(),
            1,
            "{}",
            self.tier
        );
        let out = self
            .runs(&cmd.replace(&with_mode, "env  "))
            .replace(&script(&self.family), &script(&self.tier));
        assert!(!out.contains(&self.mode), "{}: the mode is gone", self.tier);
        out
    }

    /// The pod name: `fv-lab-<family>-<tag>` as `fv-lab-<tier>-<tag>`.
    pub fn pod_name(&self, name: &str) -> String {
        let from = format!("fv-lab-{}-", self.family);
        let rest = name
            .strip_prefix(&from)
            .unwrap_or_else(|| panic!("{}: pod name `{name}`", self.tier));
        format!("fv-lab-{}-{rest}", self.tier)
    }
}
