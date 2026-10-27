//! The restored script modes (docs/adr/0019-fv-lab.md §12): `FV_LAB_RESTORED`
//! of `scripts/gpu/fv-lab-retired.sh`, `<tier>:<family>:<K=V>`. Their
//! fixtures are what the script wrote in that mode at 5987e6a4
//! (`scripts/gpu/tests/fv-lab-restored-goldens.sh`), so they carry the
//! family's name where fv-lab writes the tier's, and the mode's setting
//! where fv-lab has none. These maps are the only differences allowed.
#![allow(dead_code)]

use std::path::{Path, PathBuf};
use std::process::Command;

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

/// `FV_LAB_EXPERIMENTS` of `scripts/gpu/fv-lab-retired.sh`: the tiers made
/// from a port one-off (docs/adr/0019-fv-lab.md §14), never script families.
pub fn experiments() -> Vec<String> {
    let p = repo_root().join("scripts/gpu/fv-lab-retired.sh");
    let t = std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("{}: {e}", p.display()));
    let line = t
        .lines()
        .find_map(|l| l.strip_prefix("FV_LAB_EXPERIMENTS=\""))
        .and_then(|l| l.strip_suffix('"'))
        .expect("FV_LAB_EXPERIMENTS=\"…\"");
    line.split_whitespace().map(str::to_owned).collect()
}

/// One `<tier>:<family>` entry of `FV_LAB_SIBLINGS` (ADR-0019 §21): a tier
/// cut from a live script family's targets that its own tier cannot run
/// (`oracle-r2`: the `oracle` family's `sfwan13`, its tree from R2). Its
/// fixtures are the family's dry run under the tier's settings
/// (`scripts/gpu/tests/fv-lab-goldens.sh` `fam_env`), so they carry the
/// family's name where fv-lab writes the tier's, and, the tier being
/// `volume+r2`, the volume's weight root where fv-lab reads its merged root.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Sibling {
    pub tier: String,
    pub family: String,
}

/// `FV_LAB_SIBLINGS` of `scripts/gpu/fv-lab-retired.sh`.
pub fn siblings() -> Vec<Sibling> {
    let p = repo_root().join("scripts/gpu/fv-lab-retired.sh");
    let t = std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("{}: {e}", p.display()));
    let line = t
        .lines()
        .find_map(|l| l.strip_prefix("FV_LAB_SIBLINGS=\""))
        .and_then(|l| l.strip_suffix('"'))
        .expect("FV_LAB_SIBLINGS=\"…\"");
    line.split_whitespace()
        .map(|e| match e.split_once(':') {
            Some((t, f)) if !t.is_empty() && !f.is_empty() && !f.contains(':') => Sibling {
                tier: t.into(),
                family: f.into(),
            },
            _ => panic!("FV_LAB_SIBLINGS entry `{e}` is not <tier>:<family>"),
        })
        .collect()
}

/// The sibling a tier is, if it is one.
pub fn sibling_tier(tier: &str) -> Option<Sibling> {
    siblings().into_iter().find(|s| s.tier == tier)
}

impl Sibling {
    /// A dry-run line of the family's script as fv-lab's tier writes it:
    /// the `[<family>] ` prefix and the run directory `/runs/<family>/` carry
    /// the tier's name. (The weight root is the volume+r2 rule,
    /// [`merged_root`].)
    pub fn dry_run_line(&self, line: &str) -> String {
        let prefix = format!("[{}] ", self.family);
        let rest = line
            .strip_prefix(&prefix)
            .unwrap_or_else(|| panic!("{}: `{line}` lacks {prefix:?}", self.tier));
        format!(
            "[{}] {}",
            self.tier,
            rest.replace(
                &format!("/runs/{}/", self.family),
                &format!("/runs/{}/", self.tier)
            )
        )
    }

    /// The script's log of the prompt set its `FV_PROMPTS` names
    /// (`[<family>] prompt set: <file>`, before `matrix start`): a sibling
    /// tier that is a prompt-set mode of its family (`hd5`, the `hd` family
    /// with `FV_PROMPTS=scripts/gpu/prompts-hd5.json`, ADR-0019 §22, S2) has
    /// no counterpart of it, since its prompt set is its cells'
    /// `workload.prompt_set`, rendered in each arm's `--prompts`. The one
    /// line of the fixture fv-lab does not print
    /// (`tests/dry_run.rs::the_hd_tiers_are_the_hd_family_without_its_derived_step`
    /// holds it to the cell's prompt set).
    pub fn prompt_set_line(&self, line: &str) -> bool {
        line.starts_with(&format!("[{}] prompt set: ", self.family))
    }
}

/// A script's line as a `volume+r2` tier reads its weights: `{weights}` is
/// the merged root on the container disk (`/fvscratch/weights`, ADR-0019
/// §16.3) where the script reads the volume (`/workspace/weights`): a path
/// under it, or the root itself as a whole word (`serve-engine`'s
/// `--weights-root /workspace/weights `, ADR-0019 §22, S3). A longer name
/// that merely starts with it (`/workspace/weights2`) is not the root.
pub fn merged_root(line: &str) -> String {
    const VOLUME: &str = "/workspace/weights";
    const MERGED: &str = "/fvscratch/weights";
    let mut out = String::with_capacity(line.len());
    let mut rest = line;
    while let Some(i) = rest.find(VOLUME) {
        let after = &rest[i + VOLUME.len()..];
        out.push_str(&rest[..i]);
        if after.is_empty() || after.starts_with('/') || after.starts_with(' ') {
            out.push_str(MERGED);
        } else {
            out.push_str(VOLUME);
        }
        rest = after;
    }
    out.push_str(rest);
    out
}

/// Writes the executable `path` (a fake tool a test then runs) so that this
/// process never holds it open for writing. A fork by another test thread
/// between `open` and `close` of an in-process write would carry that fd into
/// a child until the child execs, and an `exec` of the file meanwhile fails
/// with ETXTBSY ("Text file busy", errno 26): the flake CI hit in
/// `oracle_live.rs`. So the content goes to `<path>.src`, which is never run,
/// and `install -m 0755` (a child process, whose descriptors are its own)
/// creates `path`; once it has exited, nothing holds the file open.
pub fn write_exe(path: &Path, content: &str) {
    if let Some(d) = path.parent() {
        std::fs::create_dir_all(d).unwrap_or_else(|e| panic!("{}: {e}", d.display()));
    }
    let src = PathBuf::from(format!("{}.src", path.display()));
    std::fs::write(&src, content).unwrap_or_else(|e| panic!("{}: {e}", src.display()));
    let st = Command::new("install")
        .arg("-m")
        .arg("0755")
        .arg(&src)
        .arg(path)
        .status()
        .unwrap_or_else(|e| panic!("install {}: {e}", path.display()));
    assert!(st.success(), "install {}: {st}", path.display());
    std::fs::remove_file(&src).unwrap_or_else(|e| panic!("{}: {e}", src.display()));
}
