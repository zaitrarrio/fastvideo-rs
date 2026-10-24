//! `fv-lab gold check`: the CI bridge of plan D4 (docs/adr/0023-lab-report-and-ci-bridge.md).
//!
//! Three checks over `lab/gold/*.toml` and `families/*.toml`, and one table:
//!
//! 1. every gold file parses and validates ([`crate::gold::load_dir`]), and
//!    every family variant has one;
//! 2. each file resolves against the family manifests: its variant exists in
//!    the family it names, with the same tier and profile, and its
//!    `registry_name` (and `@rev`) is the one the family file publishes it as;
//! 3. a `[last_validated].commit`, when present, names a commit of this
//!    repository (`git cat-file -e`), when the tree is a git checkout.
//!
//! The table is the validation age of every variant: days since
//! `last_validated.date`, and commits since `last_validated.commit` when git
//! can say. Missing validation is a warning, never a failure (no gold file is
//! validated yet); `--max-age-days` turns an over-age validated variant into
//! a failure.

use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use fastvideo_manifest::FamilySet;

use crate::gold::{check_date, Gold};

/// Days since 1970-01-01 of a proleptic Gregorian date (H. Hinnant's
/// `days_from_civil`).
pub fn days_from_civil(y: i64, m: u32, d: u32) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = y - era * 400;
    let m = i64::from(m);
    let doy = (153 * (if m > 2 { m - 3 } else { m + 9 }) + 2) / 5 + i64::from(d) - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

/// The day number of a `YYYY-MM-DD` date (validated by [`check_date`]).
pub fn day_number(date: &str) -> Result<i64, String> {
    check_date(date)?;
    let part = |r: std::ops::Range<usize>| -> Result<u32, String> {
        date.get(r)
            .and_then(|p| p.parse::<u32>().ok())
            .ok_or_else(|| format!("{date:?}: expected YYYY-MM-DD"))
    };
    Ok(days_from_civil(
        i64::from(part(0..4)?),
        part(5..7)?,
        part(8..10)?,
    ))
}

/// What the check asks of git; [`GitCli`] in the binary, a fake in tests.
pub trait Git {
    /// Whether `commit` names a commit object of the repository.
    fn commit_exists(&self, commit: &str) -> Result<bool, String>;
    /// Commits on `HEAD` since `commit` (`git rev-list --count <c>..HEAD`),
    /// when git can say (a shallow clone may not).
    fn commits_since(&self, commit: &str) -> Option<u64>;
}

/// `git` as a process, in one work tree.
#[derive(Debug, Clone)]
pub struct GitCli {
    root: PathBuf,
}

impl GitCli {
    /// The work tree at `root`, or why there is none (not a checkout, no
    /// `git` binary).
    pub fn detect(root: &Path) -> Result<GitCli, String> {
        let out = Command::new("git")
            .arg("-C")
            .arg(root)
            .args(["rev-parse", "--is-inside-work-tree"])
            .stdin(Stdio::null())
            .stderr(Stdio::null())
            .output()
            .map_err(|e| format!("git not runnable: {e}"))?;
        if out.status.success() && String::from_utf8_lossy(&out.stdout).trim() == "true" {
            Ok(GitCli {
                root: root.to_path_buf(),
            })
        } else {
            Err(format!("{} is not a git work tree", root.display()))
        }
    }

    fn git(&self) -> Command {
        let mut c = Command::new("git");
        c.arg("-C").arg(&self.root).stdin(Stdio::null());
        c
    }
}

impl Git for GitCli {
    fn commit_exists(&self, commit: &str) -> Result<bool, String> {
        let st = self
            .git()
            .args(["cat-file", "-e", &format!("{commit}^{{commit}}")])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .map_err(|e| format!("git cat-file: {e}"))?;
        Ok(st.success())
    }

    fn commits_since(&self, commit: &str) -> Option<u64> {
        let out = self
            .git()
            .args(["rev-list", "--count", &format!("{commit}..HEAD")])
            .stderr(Stdio::null())
            .output()
            .ok()?;
        if !out.status.success() {
            return None;
        }
        String::from_utf8_lossy(&out.stdout).trim().parse().ok()
    }
}

/// Whether a validated commit was found.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CommitState {
    Found,
    Missing,
    /// Not checked, and why (no git checkout).
    NotChecked(String),
}

/// One variant's `[last_validated]`, as the check sees it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Validation {
    pub date: String,
    pub commit: String,
    /// Days from `date` to the check's day.
    pub age_days: i64,
    pub commit_state: CommitState,
    pub commits_since: Option<u64>,
}

/// One row of the age table.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AgeRow {
    pub variant: String,
    pub family: String,
    /// `draft` | `turbo` | `max`, or `-` for an untiered variant.
    pub tier: String,
    /// `name@rev`, `name` (no rev yet) or `unpublished`.
    pub registry: String,
    pub validation: Option<Validation>,
    /// `ok`, `warn: …` or `FAIL: …`.
    pub status: String,
}

/// The whole check: the table, the failures and the warnings.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct CheckReport {
    /// `YYYY-MM-DD` the ages are counted to.
    pub today: String,
    pub rows: Vec<AgeRow>,
    pub errors: Vec<String>,
    pub warnings: Vec<String>,
    /// Why commits were not checked, when they were not.
    pub git_note: Option<String>,
    pub max_age_days: Option<i64>,
}

impl CheckReport {
    pub fn ok(&self) -> bool {
        self.errors.is_empty()
    }

    /// The report as Markdown: what CI writes to `$GITHUB_STEP_SUMMARY`.
    pub fn markdown(&self) -> String {
        let mut s = String::new();
        s.push_str("## Gold files: validation age\n\n");
        let validated = self.rows.iter().filter(|r| r.validation.is_some()).count();
        s.push_str(&format!(
            "{} gold files (`lab/gold/*.toml`) resolved against `families/*.toml`; {validated} validated by `fv-lab run gold`. Ages are counted to {}.",
            self.rows.len(),
            self.today
        ));
        match self.max_age_days {
            Some(n) => s.push_str(&format!(" A validated variant older than {n} days fails.")),
            None => s.push_str(" Warn-only: no age limit (`--max-age-days`)."),
        }
        if let Some(n) = &self.git_note {
            s.push_str(&format!(" Commits not checked: {n}."));
        }
        s.push_str("\n\n| variant | family | tier | registry | last validated | age (days) | commit | commits since | status |\n");
        s.push_str("|---|---|---|---|---|---:|---|---:|---|\n");
        for r in &self.rows {
            let (date, age, commit, since) = match &r.validation {
                Some(v) => (
                    v.date.clone(),
                    v.age_days.to_string(),
                    match &v.commit_state {
                        CommitState::Found => format!("`{}`", v.commit),
                        CommitState::Missing => format!("`{}` (missing)", v.commit),
                        CommitState::NotChecked(_) => format!("`{}` (not checked)", v.commit),
                    },
                    v.commits_since
                        .map_or_else(|| "-".to_owned(), |n| n.to_string()),
                ),
                None => ("never".into(), "-".into(), "-".into(), "-".into()),
            };
            s.push_str(&format!(
                "| `{}` | {} | {} | {} | {date} | {age} | {commit} | {since} | {} |\n",
                r.variant, r.family, r.tier, r.registry, r.status
            ));
        }
        if !self.errors.is_empty() {
            s.push_str("\n### Failures\n\n");
            for e in &self.errors {
                s.push_str(&format!("- {e}\n"));
            }
        }
        if !self.warnings.is_empty() {
            s.push_str("\n### Warnings\n\n");
            for w in &self.warnings {
                s.push_str(&format!("- {w}\n"));
            }
        }
        s
    }
}

/// How one gold file resolves against the family set: `Ok(registry)` (the
/// registry column) or every disagreement.
pub fn resolve(g: &Gold, families: &FamilySet) -> Result<String, Vec<String>> {
    let v = &g.model.variant;
    let Some((fam, var)) = families.find(v) else {
        return Err(vec![format!("`{v}`: no family file has this variant")]);
    };
    let mut errs = Vec::new();
    if var.id != *v {
        errs.push(format!(
            "`{v}`: a served name of `{}`, not a variant id",
            var.id
        ));
    }
    if fam.id() != &g.model.family {
        errs.push(format!(
            "`{v}`: family `{}`, but families/{}.toml has it",
            g.model.family,
            fam.id()
        ));
    }
    if g.tier.name != var.tier {
        errs.push(format!(
            "`{v}`: tier {:?}, the family file says {:?}",
            g.tier.name.map(|t| t.as_str()),
            var.tier.map(|t| t.as_str())
        ));
    }
    let profile = var.engine.get("profile").and_then(|p| p.as_str());
    if g.model.profile.as_deref() != profile {
        errs.push(format!(
            "`{v}`: profile {:?}, the family file says {:?}",
            g.model.profile, profile
        ));
    }
    let registry = match (&g.model.registry_name, &var.registry_name) {
        (None, None) => {
            if g.model.rev.is_some() {
                errs.push(format!("`{v}`: a rev but no registry_name"));
            }
            "unpublished".to_owned()
        }
        (Some(ours), None) => {
            errs.push(format!(
                "`{v}`: registry_name `{ours}`, but the family file publishes no registry_name"
            ));
            String::new()
        }
        (None, Some(theirs)) => {
            // Published in the family file, not yet certified by the gold
            // file: the rev arrives with the first validation.
            format!("`{theirs}` (not certified)")
        }
        (Some(ours), Some(theirs)) => {
            if ours.without_rev() != theirs.without_rev() {
                errs.push(format!(
                    "`{v}`: registry_name `{ours}`, the family file says `{theirs}`"
                ));
            }
            match &g.model.rev {
                Some(r) => format!("`{ours}@{r}`"),
                None => format!("`{ours}` (no rev)"),
            }
        }
    };
    if errs.is_empty() {
        Ok(registry)
    } else {
        Err(errs)
    }
}

/// Runs the check. `today` is `YYYY-MM-DD`; `git` is `Err(why)` outside a
/// checkout (commits are then not checked, and the report says why).
pub fn check(
    golds: &[Gold],
    families: &FamilySet,
    git: Result<&dyn Git, String>,
    today: &str,
    max_age_days: Option<i64>,
) -> Result<CheckReport, String> {
    let today_n = day_number(today)?;
    let mut rep = CheckReport {
        today: today.to_owned(),
        max_age_days,
        git_note: git.as_ref().err().cloned(),
        ..CheckReport::default()
    };
    for (_, var) in families.variants() {
        if !golds.iter().any(|g| g.model.variant == var.id) {
            rep.errors.push(format!(
                "`{}`: a family variant without lab/gold/{}.toml",
                var.id, var.id
            ));
        }
    }
    let mut never = 0usize;
    for g in golds {
        let mut status: Vec<String> = Vec::new();
        let mut failed = false;
        let registry = match resolve(g, families) {
            Ok(r) => r,
            Err(es) => {
                failed = true;
                status.push("does not resolve".into());
                rep.errors.extend(es);
                "-".into()
            }
        };
        let lv = &g.last_validated;
        let validation = match (&lv.commit, &lv.date) {
            (Some(commit), Some(date)) => {
                let age_days = today_n - day_number(date)?;
                let (commit_state, commits_since) = match &git {
                    Ok(git) => match git.commit_exists(commit) {
                        Ok(true) => (CommitState::Found, git.commits_since(commit)),
                        Ok(false) => (CommitState::Missing, None),
                        Err(e) => (CommitState::NotChecked(e), None),
                    },
                    Err(why) => (CommitState::NotChecked(why.clone()), None),
                };
                if commit_state == CommitState::Missing {
                    failed = true;
                    status.push("commit missing".into());
                    rep.errors.push(format!(
                        "`{}`: last_validated.commit `{commit}` is not a commit of this repository",
                        g.model.variant
                    ));
                }
                if age_days < 0 {
                    failed = true;
                    status.push("dated in the future".into());
                    rep.errors.push(format!(
                        "`{}`: last_validated.date {date} is after {today}",
                        g.model.variant
                    ));
                }
                if let Some(max) = max_age_days {
                    if age_days > max {
                        failed = true;
                        status.push(format!("older than {max} days"));
                        rep.errors.push(format!(
                            "`{}`: validated {age_days} days ago ({date}), over --max-age-days {max}",
                            g.model.variant
                        ));
                    }
                }
                Some(Validation {
                    date: date.clone(),
                    commit: commit.clone(),
                    age_days,
                    commit_state,
                    commits_since,
                })
            }
            _ => {
                never += 1;
                None
            }
        };
        let status = match (failed, &validation) {
            (true, _) => format!("FAIL: {}", status.join(", ")),
            (false, None) => "warn: never validated".to_owned(),
            (false, Some(_)) => "ok".to_owned(),
        };
        rep.rows.push(AgeRow {
            variant: g.model.variant.clone(),
            family: g.model.family.to_string(),
            tier: g.tier.name.map_or("-", |t| t.as_str()).to_owned(),
            registry,
            validation,
            status,
        });
    }
    if never > 0 {
        rep.warnings.push(format!(
            "{never} of {} variants were never validated by `fv-lab run gold` (no `[last_validated]`): their numbers are the cited sources' (ADR-0009)",
            golds.len()
        ));
    }
    if let Some(n) = &rep.git_note {
        rep.warnings
            .push(format!("last_validated commits not checked: {n}"));
    }
    Ok(rep)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gold::Gold;

    const FAMILY: &str = r#"
schema = 1

[family]
id = "demo"
display = "Demo"
licence = "MIT"
port_doc = "docs/ports/demo.md"

[[variants]]
id = "demo-turbo"
tier = "turbo"
recipe_name = "demo-turbo"
trees = ["demo-base"]

[variants.gpu]
min_vram_gib = 24
min_sm = 89

[variants.caps.frames]
step = 4
offset = 1
min = 5
max = 121

[variants.caps.canvas]
multiple = 32
max_area = 1032192
aspect = [0.25, 4.0]
short_edges = [480]

[variants.caps.fps]
allowed = [24]
default = 24

[variants.caps.knobs]
seed = true
negative = false
steps = false
guidance = false
guidance_2 = false
flow_shift = false

[variants.serving]
tasks = ["t2v"]
frames_default = 121

[variants.engine]
family = "demo"
profile = "demo/demo_turbo"
"#;

    fn gold(extra: &str) -> Gold {
        let text = format!(
            r#"
schema = 1

[model]
variant = "demo-turbo"
family = "demo"
profile = "demo/demo_turbo"

[tier]
name = "turbo"
gpu = "RTX PRO 6000 Blackwell"
sm = 120
vram_gib = 96

[workload]
width = 832
height = 480
frames = 121
prompt_set = "one prompt"

{extra}

[[source]]
fields = ["workload"]
path = "docs/perf/demo.md"
heading = "Demo"
date = "2026-10-01"

[[gap]]
field = "latency.p50_s"
why = "x"
[[gap]]
field = "latency.min_s"
why = "x"
[[gap]]
field = "latency.max_s"
why = "x"
[[gap]]
field = "latency.denoise_s"
why = "x"
[[gap]]
field = "latency.decode_s"
why = "x"
[[gap]]
field = "latency.load_s"
why = "x"
[[gap]]
field = "latency.peak_gib"
why = "x"
[[gap]]
field = "latency.fps"
why = "x"
[[gap]]
field = "latency.ttff_s"
why = "x"
[[gap]]
field = "quality.psnr_vs_upstream_min_db"
why = "x"
[[gap]]
field = "quality.lpips_vs_upstream_max"
why = "x"
[[gap]]
field = "quality.sharpness_ratio"
why = "x"
[[gap]]
field = "quality.jitter_ratio"
why = "x"
[[gap]]
field = "cost.usd_per_hr"
why = "x"
[[gap]]
field = "cost.usd_per_clip"
why = "x"
"#
        );
        Gold::parse(&text).expect("test gold file")
    }

    fn fams(text: &str) -> FamilySet {
        FamilySet::parse_all([("demo.toml", text)]).expect("test family")
    }

    struct FakeGit(&'static [&'static str]);
    impl Git for FakeGit {
        fn commit_exists(&self, c: &str) -> Result<bool, String> {
            Ok(self.0.contains(&c))
        }
        fn commits_since(&self, _: &str) -> Option<u64> {
            Some(3)
        }
    }

    #[test]
    fn day_numbers() {
        assert_eq!(days_from_civil(1970, 1, 1), 0);
        assert_eq!(day_number("2000-03-01").unwrap(), 11_017);
        assert_eq!(
            day_number("2026-10-24").unwrap() - day_number("2026-09-24").unwrap(),
            30
        );
        assert_eq!(
            day_number("2025-01-01").unwrap() - day_number("2024-01-01").unwrap(),
            366
        );
        assert!(day_number("2026-13-01").is_err());
    }

    #[test]
    fn never_validated_is_a_warning_not_a_failure() {
        let g = gold("");
        let git = FakeGit(&[]);
        let r = check(&[g], &fams(FAMILY), Ok(&git), "2026-10-24", Some(30)).unwrap();
        assert!(r.ok(), "{:?}", r.errors);
        assert_eq!(r.rows[0].status, "warn: never validated");
        assert_eq!(r.rows[0].registry, "unpublished");
        assert_eq!(r.warnings.len(), 1);
        let md = r.markdown();
        assert!(md.contains("| `demo-turbo` | demo | turbo | unpublished | never | - | - | - | warn: never validated |"), "{md}");
    }

    #[test]
    fn a_validated_commit_must_exist_and_its_age_is_counted() {
        let lv = "[last_validated]\ncommit = \"abcdef1\"\ndate = \"2026-10-14\"\nrun = \"artifacts/runpod/gold/x\"";
        let found = FakeGit(&["abcdef1"]);
        let r = check(&[gold(lv)], &fams(FAMILY), Ok(&found), "2026-10-24", None).unwrap();
        assert!(r.ok(), "{:?}", r.errors);
        let v = r.rows[0].validation.as_ref().unwrap();
        assert_eq!((v.age_days, v.commits_since), (10, Some(3)));
        assert_eq!(r.rows[0].status, "ok");

        let missing = FakeGit(&[]);
        let r = check(&[gold(lv)], &fams(FAMILY), Ok(&missing), "2026-10-24", None).unwrap();
        assert!(!r.ok());
        assert!(r.errors[0].contains("not a commit"), "{:?}", r.errors);

        // Over the age limit fails only with --max-age-days.
        let r = check(
            &[gold(lv)],
            &fams(FAMILY),
            Ok(&found),
            "2026-10-24",
            Some(7),
        )
        .unwrap();
        assert!(!r.ok());
        assert!(r.rows[0].status.contains("older than 7 days"));

        // Not a checkout: noted, not failed.
        let r = check(
            &[gold(lv)],
            &fams(FAMILY),
            Err("not a git work tree".into()),
            "2026-10-24",
            None,
        )
        .unwrap();
        assert!(r.ok());
        assert_eq!(
            r.rows[0].validation.as_ref().unwrap().commit_state,
            CommitState::NotChecked("not a git work tree".into())
        );
        assert!(r
            .markdown()
            .contains("Commits not checked: not a git work tree."));
    }

    #[test]
    fn resolution_names_every_disagreement() {
        let other = FAMILY
            .replace("tier = \"turbo\"", "tier = \"max\"")
            .replace("profile = \"demo/demo_turbo\"", "profile = \"demo/other\"");
        let git = FakeGit(&[]);
        let r = check(&[gold("")], &fams(&other), Ok(&git), "2026-10-24", None).unwrap();
        assert!(!r.ok());
        assert_eq!(r.errors.len(), 2, "{:?}", r.errors);
        assert!(r.rows[0].status.starts_with("FAIL: does not resolve"));

        // A variant without a gold file, and a gold file without a variant.
        let renamed = FAMILY.replace("id = \"demo-turbo\"", "id = \"demo-max\"");
        let r = check(&[gold("")], &fams(&renamed), Ok(&git), "2026-10-24", None).unwrap();
        assert!(r
            .errors
            .iter()
            .any(|e| e.contains("without lab/gold/demo-max.toml")));
        assert!(r.errors.iter().any(|e| e.contains("no family file has")));
    }

    #[test]
    fn registry_name_and_rev_resolve_against_the_family_file() {
        let published = FAMILY.replace(
            "recipe_name = \"demo-turbo\"",
            "registry_name = \"demo/demo/dmd:4:bf16\"\nrecipe_name = \"demo-turbo\"",
        );
        let git = FakeGit(&[]);
        let named = gold("").to_toml_string().unwrap().replace(
            "[model]\n",
            "[model]\nregistry_name = \"demo/demo/dmd:4:bf16\"\nrev = \"0123456789ab\"\n",
        );
        let g = Gold::parse(&named).unwrap();
        let r = check(
            std::slice::from_ref(&g),
            &fams(&published),
            Ok(&git),
            "2026-10-24",
            None,
        )
        .unwrap();
        assert!(r.ok(), "{:?}", r.errors);
        assert_eq!(r.rows[0].registry, "`demo/demo/dmd:4:bf16@0123456789ab`");

        // The gold file names a registry name the family file does not publish.
        let r = check(&[g], &fams(FAMILY), Ok(&git), "2026-10-24", None).unwrap();
        assert!(!r.ok());

        // Published in the family file, not yet certified by the gold file.
        let r = check(&[gold("")], &fams(&published), Ok(&git), "2026-10-24", None).unwrap();
        assert!(r.ok());
        assert_eq!(r.rows[0].registry, "`demo/demo/dmd:4:bf16` (not certified)");
    }
}
