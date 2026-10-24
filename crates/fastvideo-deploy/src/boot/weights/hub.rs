//! `weights_source hub` (owner-approved downloads; control providers.ts
//! `HUB_FETCH_TREE_SH` / `WEIGHTS_LOOP_SH`, serverless/weights.ts `SLS_HUB_SH`):
//! the `tree` rows of `FV_WEIGHTS_TREES_B64` from the Hugging Face Hub at
//! their pinned revisions onto the container disk, through
//! `scripts/gpu/fetch-hub-tree.py` at the image's commit (`FV_SCRIPTS_URL`),
//! which checks every file against the Hub listing and renames the tree into
//! place only then. A tree complete at its revision is kept (a restarted
//! container or a parked Brev VM: warm), one at another revision is moved
//! aside and removed (the pod's own disk, never a shared volume). With
//! `FV_HUB_SHA256_B64` each tree's recorded files must also match their
//! SHA-256 and size (scripts/gpu/weights-sha256.tsv), else the tree is
//! removed and the boot fails before fv-serve starts.

use std::path::{Path, PathBuf};
use std::time::Duration;

use super::super::error::BootError;
use super::super::plan::{self, HubSum, Row, TreeRow};
use super::super::report::step_name;
use super::super::sys::{Io, SysError};
use super::super::Ctx;
use super::{apt_install, aux_files, tree_present, Prepared, SourceKind, WeightsSource};

/// `weights_source hub`.
pub struct Hub;

fn mkdir(p: &Path) -> Result<(), BootError> {
    std::fs::create_dir_all(p).map_err(|e| BootError::Sys(SysError::io(p, e)))
}

/// `sha256.txt` (`<hex>  <path>` lines) as a lookup.
pub fn sha_list(text: &str) -> Vec<(String, String)> {
    text.lines()
        .filter_map(|l| {
            let (h, p) = l.split_once("  ")?;
            Some((h.trim().to_ascii_lowercase(), p.to_owned()))
        })
        .collect()
}

/// The recorded files of `dest` that the fetcher's hashes or the sizes on
/// disk contradict (`path` or `path(size)`), and how many were recorded.
pub fn verify(
    root: &Path,
    dest: &str,
    sums: &[HubSum],
    fetched: &[(String, String)],
) -> (usize, Vec<String>) {
    let mut n = 0;
    let mut bad = Vec::new();
    for s in sums.iter().filter(|s| s.dest == dest) {
        n += 1;
        let got = fetched
            .iter()
            .find(|(_, p)| *p == s.path)
            .map(|(h, _)| h.as_str())
            .unwrap_or("");
        if !s.sha256.iter().any(|h| h == got) {
            bad.push(s.path.clone());
        }
        if let Some(b) = s.bytes {
            if std::fs::metadata(root.join(dest).join(&s.path))
                .map(|m| m.len())
                .ok()
                != Some(b)
            {
                bad.push(format!("{}(size)", s.path));
            }
        }
    }
    (n, bad)
}

fn remove_aside(root: &Path, dest: &str, now: f64) -> Result<(), BootError> {
    let old = root.join(format!(".stale-{}-{}", dest.replace('/', "_"), now as u64));
    std::fs::rename(root.join(dest), &old)
        .and_then(|()| std::fs::remove_dir_all(&old))
        .map_err(|_| BootError::Weights(format!("tree {dest}: cannot move the old copy aside")))
}

struct Fetcher<'c, 'a> {
    c: &'c Ctx<'a>,
    root: PathBuf,
    state: PathBuf,
    ready: bool,
}

impl Fetcher<'_, '_> {
    fn venv(&self) -> PathBuf {
        self.c.paths.tools.join("venv")
    }
    fn script(&self) -> PathBuf {
        self.c.paths.tools.join("scripts").join("fetch-hub-tree.py")
    }

    fn step(&self, name: &str, since: f64) {
        // Pods log `step hub <tree>`; serverless reports `hub_<tree>` (the shell's tr ' /' '__').
        let secs = (self.c.now() - since).max(0.0);
        self.c.log.step(name, secs, "");
        self.c.rep.step(
            self.c.deps.http,
            self.c.deps.clock,
            &step_name(name),
            secs,
            "",
        );
    }

    /// apt python3-venv / pip, a venv with huggingface_hub and the fetcher, on first use.
    fn tools(&mut self) -> Result<(), BootError> {
        if self.ready {
            return Ok(());
        }
        let c = self.c;
        let t = c.now();
        if !apt_install(
            c,
            &["python3-venv", "python3-pip"],
            &self.state.join("apt.log"),
        ) {
            return Err(BootError::Weights("apt python3-venv".into()));
        }
        let venv = self.venv();
        let pip_log = self.state.join("pip.log");
        let mk = c
            .cmd("python3")
            .args(["-m", "venv", &venv.display().to_string()])
            .stdout(Io::Append(pip_log.clone()))
            .stderr(Io::Append(pip_log.clone()));
        let pip = c
            .cmd(venv.join("bin/pip").display().to_string())
            .args(["install", "-q", "huggingface_hub>=0.34"])
            .stdout(Io::Append(pip_log.clone()))
            .stderr(Io::Append(pip_log));
        if c.deps.spawner.status(&mk)? != Some(0) || c.deps.spawner.status(&pip)? != Some(0) {
            return Err(BootError::Weights("pip huggingface_hub".into()));
        }
        let base = c
            .get("FV_SCRIPTS_URL")
            .ok_or_else(|| BootError::Env("weights_source hub: no FV_SCRIPTS_URL".into()))?;
        let script = self.script();
        if let Some(p) = script.parent() {
            mkdir(p)?;
        }
        c.deps
            .http
            .download(
                &format!("{}/fetch-hub-tree.py", base.trim_end_matches('/')),
                &script,
                Duration::from_secs(60),
            )
            .map_err(|_| BootError::Weights("fetch-hub-tree.py".into()))?;
        self.ready = true;
        self.step("hub_tools", t);
        Ok(())
    }

    fn fetch(&mut self, row: &TreeRow, n: usize, sums: &[HubSum]) -> Result<(), BootError> {
        self.tools()?;
        let c = self.c;
        let srv = self.state.join(format!("fetch-{n}"));
        mkdir(&srv)?;
        let t = c.now();
        let out = self.state.join(format!("fetch-{n}.out"));
        let cmd = c
            .cmd(self.venv().join("bin/python").display().to_string())
            .arg(self.script().display().to_string())
            .env("FETCH_REPO", row.repo.clone())
            .env("FETCH_REVISION", row.revision.clone())
            .env("FETCH_DEST", row.dest.clone())
            .env("FETCH_GLOBS", row.globs.clone())
            .env("FETCH_WEIGHTS", self.root.display().to_string())
            .env("FETCH_SRV", srv.display().to_string())
            .env("FETCH_MIN_FREE_GB", "5")
            .stdout(Io::Truncate(out.clone()))
            .stderr(Io::Append(out));
        let tail = |n: usize, skip_indented: bool| -> String {
            let t = std::fs::read_to_string(srv.join("log.txt")).unwrap_or_default();
            let lines: Vec<&str> = t
                .lines()
                .filter(|l| !(skip_indented && l.starts_with(' ')))
                .collect();
            lines[lines.len().saturating_sub(n)..]
                .join(" ")
                .replace('"', " ")
        };
        if c.deps.spawner.status(&cmd)? != Some(0) {
            return Err(BootError::Weights(format!(
                "tree {}: {}",
                row.dest,
                tail(2, false)
            )));
        }
        // fetch-hub-tree.py records its outcome in DONE (0: renamed into place after every check).
        let done = std::fs::read_to_string(srv.join("DONE")).unwrap_or_default();
        if done.trim() != "0" {
            return Err(BootError::Weights(format!(
                "tree {}: {}",
                row.dest,
                tail(2, true)
            )));
        }
        let fetched =
            sha_list(&std::fs::read_to_string(srv.join("sha256.txt")).unwrap_or_default());
        let (k, bad) = verify(&self.root, &row.dest, sums, &fetched);
        if !bad.is_empty() {
            let _ = remove_aside(&self.root, &row.dest, c.now());
            return Err(BootError::Weights(format!(
                "tree {}: not the recorded SHA-256 (weights-sha256.tsv): {}",
                row.dest,
                bad.join(" ")
            )));
        }
        if k == 0 {
            c.say(&format!(
                "tree {}: no weights-sha256.tsv rows: checked against the Hub listing only",
                row.dest
            ));
        } else {
            c.say(&format!(
                "tree {}: {k} files match weights-sha256.tsv",
                row.dest
            ));
        }
        self.step(&format!("hub {}", row.dest), t);
        Ok(())
    }
}

fn du(p: &Path) -> u64 {
    let Ok(m) = std::fs::symlink_metadata(p) else {
        return 0;
    };
    if m.is_dir() {
        std::fs::read_dir(p)
            .map(|rd| rd.filter_map(Result::ok).map(|e| du(&e.path())).sum())
            .unwrap_or(0)
    } else {
        m.len()
    }
}

impl WeightsSource for Hub {
    fn kind(&self) -> SourceKind {
        SourceKind::Hub
    }

    fn prepare(&self, c: &mut Ctx<'_>) -> Result<Prepared, BootError> {
        let root = c.weights_dir();
        let state = c.state_dir();
        mkdir(&root)?;
        c.say("weights: Hub trees at pinned revisions (owner-approved)");
        let boot_note = format!("container started; {}", c.driver);
        c.sls_report("boot", &boot_note);
        let rows = plan::rows(c.get("FV_WEIGHTS_TREES_B64"))?;
        let sums = plan::hub_sums(c.get("FV_HUB_SHA256_B64"))?;
        let miss: Vec<&str> = rows
            .iter()
            .filter_map(|r| match r {
                Row::Tree(t) if !tree_present(&root, &t.dest, &t.revision) => Some(t.dest.as_str()),
                _ => None,
            })
            .collect();
        if miss.is_empty() {
            c.say(&format!(
                "weights: cache hit (every tree at its revision under {}): no download",
                root.display()
            ));
            c.sls_report("cache", "hit");
        } else {
            c.say(&format!(
                "weights: cache miss: {}: downloading from the Hub at the pinned revisions",
                miss.join(" ")
            ));
            c.sls_report("cache", &format!("miss: {}", miss.join(" ")));
        }
        let missed: Vec<String> = miss.iter().map(|s| (*s).to_owned()).collect();
        super::clean_leftovers(&root);
        c.pod_report("weights", "", "checking");
        let t0 = c.now();
        let mut f = Fetcher {
            c: &*c,
            root: root.clone(),
            state: state.clone(),
            ready: false,
        };
        let (mut kept, mut got) = (0, 0);
        let mut aux = Vec::new();
        for (i, r) in rows.iter().enumerate() {
            match r {
                Row::Tree(t) => {
                    if tree_present(&root, &t.dest, &t.revision) {
                        f.c.say(&format!("tree {}: present at {}", t.dest, t.revision));
                        kept += 1;
                        continue;
                    }
                    if root.join(&t.dest).exists() {
                        f.c.say(&format!(
                            "tree {}: not complete at {}: replaced",
                            t.dest, t.revision
                        ));
                        remove_aside(&root, &t.dest, f.c.now())?;
                    }
                    f.c.say(&format!("tree {} <- {}@{}", t.dest, t.repo, t.revision));
                    f.c.pod_report("weights", "", &format!("downloading {}", t.dest));
                    f.fetch(t, i + 1, &sums)?;
                    std::fs::write(root.join(&t.dest).join(".fv-revision"), &t.revision).map_err(
                        |_| BootError::Weights(format!("tree {}: revision mark", t.dest)),
                    )?;
                    got += 1;
                }
                Row::Aux(a) => aux.push(a.clone()),
            }
        }
        aux_files(c, &root, &aux, false)?;
        c.say(&format!("weights ready: {kept} present, {got} fetched"));
        if !missed.is_empty() {
            let bytes: u64 = missed.iter().map(|d| du(&root.join(d))).sum();
            let secs = (c.now() - t0).max(0.0);
            let mbps = if secs > 0.0 {
                (bytes as f64 / 1e6 / secs) as u64
            } else {
                0
            };
            c.sls_report(
                "weights",
                &format!(
                    "weights ready: hub {:.2} GB fetched in {secs:.1}s ({mbps} MB/s)",
                    bytes as f64 / 1e9
                ),
            );
        }
        c.step("weights", t0, "Hub");
        Ok(Prepared {
            root: Some(root),
            streaming: None,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sha_lists_and_verify() {
        let d = std::env::temp_dir().join(format!("fvb-hub-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(d.join("t/sub")).unwrap();
        std::fs::write(d.join("t/sub/a.bin"), b"12345").unwrap();
        let fetched = sha_list("AA  sub/a.bin\nbb  other\n");
        assert_eq!(fetched[0], ("aa".into(), "sub/a.bin".into()));
        let sums = vec![
            HubSum {
                dest: "t".into(),
                path: "sub/a.bin".into(),
                bytes: Some(5),
                sha256: vec!["zz".into(), "aa".into()],
            },
            HubSum {
                dest: "u".into(),
                path: "x".into(),
                bytes: None,
                sha256: vec!["x".into()],
            },
        ];
        assert_eq!(verify(&d, "t", &sums, &fetched), (1, vec![]));
        let wrong = vec![HubSum {
            dest: "t".into(),
            path: "sub/a.bin".into(),
            bytes: Some(6),
            sha256: vec!["cc".into()],
        }];
        assert_eq!(
            verify(&d, "t", &wrong, &fetched),
            (1, vec!["sub/a.bin".into(), "sub/a.bin(size)".into()])
        );
        std::fs::remove_dir_all(&d).unwrap();
    }
}
