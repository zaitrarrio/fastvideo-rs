//! The dry-run equality (docs/adr/0019-fv-lab.md §5, D1b): for every tier of
//! `lab/tiers.toml`, `fv-lab run <tier> --dry-run` prints exactly what
//! `FV_DRY_RUN=1 runpod-matrix.sh <tier>` logs (`fixtures/<tier>-dry-run.txt`,
//! written from the script by `scripts/gpu/tests/fv-lab-goldens.sh`), and
//! every arm gates on the script's verify-weights.sh cell
//! (`fixtures/<tier>-gates.tsv`) and lists that cell's trees.
//!
//! Frozen since the GPU gate (ADR-0019 §10): the tiers of
//! `scripts/gpu/fv-lab-retired.sh` have no script branch left, so their
//! fixtures are the ones written before the branch went, and fv-lab's own
//! goldens since. A change to such a tier's commands updates its fixture in
//! the same PR; these tests hold fv-lab to it all the same.
//!
//! The restored modes (ADR-0019 §12, `FV_LAB_RESTORED`) started frozen: their
//! fixtures are the script's dry run in that mode at 5987e6a4, and fv-lab's
//! equal them with the family's name read as the tier's (`common`).

mod common;

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use fastvideo_lab::goldcell::GOLD_TIER;
use fastvideo_lab::plan::{Lab, Step};
use fastvideo_lab::run::{dry_run_lines, RunConfig};

// The inputs of fv-lab-goldens.sh.
const TAG: &str = "1234567-10241200";
const BASELINE_SHA: &str = "7654321";

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// A tier's dry-run fixture as fv-lab must print it: the script's lines,
/// with a restored mode's family named by its tier (`common::Restored`).
fn dry_run_fixture(tier: &str) -> Vec<String> {
    let text = fixture(&format!("{tier}-dry-run.txt"));
    match common::restored_tier(tier) {
        Some(r) => text.lines().map(|l| r.dry_run_line(l)).collect(),
        None => text.lines().map(str::to_owned).collect(),
    }
}

fn fixture(name: &str) -> String {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("fixtures")
        .join(name);
    std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("{}: {e}", p.display()))
}

fn config(family: &str) -> RunConfig {
    let mut c = RunConfig::new(family, Path::new("/opt/fastvideo-rs"));
    c.work = "/workspace".into();
    c.scratch = "/fvscratch".into();
    c.tag = Some(TAG.into());
    c.baseline_sha = Some(BASELINE_SHA.into());
    // The script's $TAE as the goldens' volume has it: its auxiliary/tae copy
    // is complete (as on bz0ofkfx45), so the volume's directory
    // (RunConfig::tae resolves the same; `run::tests::pins_and_paths`).
    c.tae_dir = Some("/workspace/weights/auxiliary/tae".into());
    c
}

fn steps(lab: &Lab, cfg: &RunConfig) -> Vec<Step> {
    let (_, cells) = lab.resolve(&cfg.family, None).expect("tier");
    Lab::steps(&cells, &cfg.bin, &cfg.paths()).expect("render")
}

/// The tiers that are (or were, before fv-lab-retired.sh) runpod-matrix.sh
/// families: every tier but `gold`, whose cells come from the gold files
/// (tests/gold_tier.rs holds its dry run).
fn families(lab: &Lab) -> Vec<String> {
    lab.tiers
        .tiers
        .iter()
        .filter(|t| t.name != GOLD_TIER)
        .map(|t| t.name.clone())
        .collect()
}

#[test]
fn dry_run_equals_runpod_matrix_sh_for_every_tier() {
    let lab = Lab::load(&repo_root()).expect("lab/ parses");
    let fams = families(&lab);
    assert!(fams.len() >= 10, "rtx6000 and the D1b families: {fams:?}");
    for family in fams {
        let cfg = config(&family);
        let ours = dry_run_lines(&cfg, &steps(&lab, &cfg));
        let theirs = dry_run_fixture(&family);
        for (i, (a, b)) in ours.iter().zip(&theirs).enumerate() {
            assert_eq!(a, b, "{family}: line {i} differs");
        }
        assert_eq!(ours.len(), theirs.len(), "{family}: line count");
        let (_, cells) = lab.resolve(&family, None).expect("tier");
        let names: BTreeSet<&str> = cells
            .iter()
            .flat_map(|c| &c.arms)
            .map(|a| a.name.as_str())
            .collect();
        // `dry-run <arm>: <command>`; a verdict arm's `rm -rf` line aside.
        let arms = ours
            .iter()
            .filter_map(|l| l.split_once("] dry-run ").map(|(_, r)| r))
            .filter_map(|r| r.split_once(": "))
            .filter(|(arm, cmd)| names.contains(arm) && !cmd.starts_with("rm -rf "))
            .count();
        assert_eq!(arms, names.len(), "{family}: one dry-run line per arm");
    }
    // D1a's family, as pinned then.
    let cfg = config("rtx6000");
    assert_eq!(
        dry_run_lines(&cfg, &steps(&lab, &cfg))
            .iter()
            .filter(|l| l.contains("] dry-run "))
            .count(),
        7
    );
}

#[test]
fn baseline_arms_run_the_fetched_binary() {
    let lab = Lab::load(&repo_root()).expect("lab/ parses");
    for family in ["techniques", "b6", "b4a"] {
        let (_, cells) = lab.resolve(family, None).expect("tier");
        assert!(Lab::needs_baseline(&cells), "{family}");
        let cfg = config(family);
        let base = format!("/fvscratch/baseline-{BASELINE_SHA}/fv-gpucheck");
        let mut pairs = 0;
        let steps = steps(&lab, &cfg);
        for s in Step::flatten(&steps) {
            if let Step::Arm {
                name,
                command,
                baseline,
                ..
            } = s
            {
                assert_eq!(*baseline, name.ends_with("-base"), "{family}: {name}");
                assert_eq!(command.contains(&base), *baseline, "{family}: {name}");
                pairs += usize::from(*baseline);
            }
        }
        assert!(pairs >= 3, "{family}: {pairs} baseline arms");
        // Without the baseline commit the tier does not render.
        let mut no = config(family);
        no.baseline_sha = None;
        let e = Lab::steps(&cells, &no.bin, &no.paths()).unwrap_err();
        assert!(e.to_string().contains("FV_BASELINE_SHA"), "{e}");
    }
    let (_, cells) = lab.resolve("rtx6000", None).expect("tier");
    assert!(!Lab::needs_baseline(&cells));
}

/// One row of `<tier>-gates.tsv`, what `FV_DRY_RUN_GATES` recorded.
fn gate_row(s: &Step) -> Option<Vec<String>> {
    match s {
        Step::Arm {
            name, gate, tae, ..
        } => Some(vec![
            "arm".into(),
            name.clone(),
            gate.clone(),
            tae.as_ref()
                .map_or_else(|| "-".to_owned(), |t| t.display().to_string()),
        ]),
        Step::Gate {
            baseline,
            candidate,
            kind,
            off,
        } => Some(vec![
            "gate".into(),
            baseline.clone(),
            candidate.clone(),
            kind.as_str().to_owned(),
            off.clone().unwrap_or_else(|| "-".into()),
        ]),
        Step::Compare { .. } | Step::Verdict { .. } | Step::FamilyVerdict { .. } => None,
    }
}

#[test]
fn arms_and_gates_are_the_scripts_and_arms_name_their_trees() {
    // Every gated cell (its verify-weights.sh cell and its TAE file, or `-`)
    // and every promotion gate (kind, OFF arm), in run order, as the script's
    // dry run recorded them; then each arm's trees against its cell's needs.
    let lab = Lab::load(&repo_root()).expect("lab/ parses");
    let mut tae_arms = 0;
    let mut gates_seen = 0;
    for family in families(&lab) {
        let cfg = config(&family);
        let rendered = steps(&lab, &cfg);
        let steps = Step::flatten(&rendered);
        let fixture = fixture(&format!("{family}-gates.tsv"));
        let rows: Vec<Vec<&str>> = fixture.lines().map(|l| l.split('\t').collect()).collect();
        let ours: Vec<Vec<String>> = steps.iter().copied().filter_map(gate_row).collect();
        assert_eq!(ours.len(), rows.len(), "{family}: rows");
        for (o, row) in ours.iter().zip(&rows) {
            let width = o.len().min(row.len());
            assert_eq!(
                o.iter().map(String::as_str).collect::<Vec<_>>(),
                row[..width].to_vec(),
                "{family}"
            );
            assert!(row.len() >= o.len(), "{family}: short row {row:?}");
            tae_arms += usize::from(o[0] == "arm" && o[3] != "-");
            gates_seen += usize::from(o[0] == "gate");
        }
        let arms = steps.iter().filter_map(|s| match s {
            Step::Arm {
                name, gate, trees, ..
            } => Some((name, gate, trees)),
            _ => None,
        });
        let needs = rows.iter().filter(|r| r[0] == "arm").map(|r| r[4]);
        for ((name, gate, trees), needs) in arms.zip(needs) {
            // The roots of the cell's needs (`root:component`; `:path` reads a
            // file directly under the weight root, whose first part is the tree).
            let roots: BTreeSet<String> = needs
                .split_whitespace()
                .map(|n| match n.split_once(':') {
                    Some(("", path)) => path.split('/').next().unwrap_or("").to_owned(),
                    Some((root, _)) => root.to_owned(),
                    None => n.to_owned(),
                })
                .collect();
            let listed: BTreeSet<String> = trees.iter().cloned().collect();
            assert_eq!(
                listed, roots,
                "{family} arm {name}: trees vs verify-weights.sh {gate}"
            );
        }
    }
    // rtx5090 (5), fastvideo (7), h3arms (4) read a tiny autoencoder;
    // fastvideo (7), h3arms (10), precision (6), precision-nvfp4 (2) gate.
    assert_eq!((tae_arms, gates_seen), (16, 25));
}

/// `FV_LAB_RETIRED` of `scripts/gpu/fv-lab-retired.sh`.
fn retired() -> Vec<String> {
    let p = repo_root().join("scripts/gpu/fv-lab-retired.sh");
    let t = std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("{}: {e}", p.display()));
    let line = t
        .lines()
        .find_map(|l| l.strip_prefix("FV_LAB_RETIRED=\""))
        .and_then(|l| l.strip_suffix('"'))
        .expect("FV_LAB_RETIRED=\"…\"");
    line.split_whitespace().map(str::to_owned).collect()
}

/// A label of runpod-matrix.sh's top-level `case "$FAMILY" in` naming `f`:
/// two-space indent, `f)`, `f )`, `a|f)` and a trailing comment all count;
/// nested cases' labels (deeper indent) do not (fv-lab-goldens.sh `has_branch`).
fn has_branch(matrix: &str, f: &str) -> bool {
    matrix
        .lines()
        .skip_while(|l| *l != "case \"$FAMILY\" in")
        .take_while(|l| *l != "esac")
        .filter_map(|l| l.strip_prefix("  "))
        .filter(|l| !l.starts_with(' '))
        .filter_map(|l| l.split_once(')').map(|(head, _)| head))
        .any(|head| {
            let alts: Vec<&str> = head.split('|').map(str::trim).collect();
            alts.iter().all(|a| {
                !a.is_empty()
                    && a.bytes()
                        .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-')
            }) && alts.contains(&f)
        })
}

#[test]
fn branch_labels_are_read_as_bash_reads_them() {
    let m = |label: &str| format!("x\ncase \"$FAMILY\" in\n{label}\n    ;;\nesac\n");
    for yes in ["  trace)", "  trace )", "  x|trace) # c", "  trace | y)"] {
        assert!(has_branch(&m(yes), "trace"), "{yes}");
    }
    for no in ["    trace)", "  tracex)", "  x$(trace)"] {
        assert!(!has_branch(&m(no), "trace"), "{no}");
    }
    assert!(!has_branch("  trace)\n", "trace"), "outside the case");
}

#[test]
fn retired_families_are_frozen_tiers_without_a_branch() {
    // The families whose GPU gate passed (ADR-0019 §10): each a tier with its
    // three fixtures, no `case` branch left in runpod-matrix.sh; every other
    // script tier still has its branch (fv-lab-goldens.sh checks the same).
    let lab = Lab::load(&repo_root()).expect("lab/ parses");
    let matrix = std::fs::read_to_string(repo_root().join("scripts/gpu/runpod-matrix.sh"))
        .expect("runpod-matrix.sh");
    let branch = |f: &str| has_branch(&matrix, f);
    let retired = retired();
    for f in [
        "rtx6000",
        "techniques",
        "b6",
        "rtx5090",
        "fastvideo",
        "h3arms",
        "trace",
        "ltxfps",
        "ltxvae",
        "precision",
        "precision-nvfp4",
        "ltxvae-gen",
    ] {
        assert!(retired.iter().any(|r| r == f), "{f} retired");
    }
    // A restored mode (§12) is a mode of a retired family, frozen from the
    // start: its family's branch is gone too, and it never had one of its own.
    let restored = common::restored();
    assert_eq!(
        restored
            .iter()
            .map(|r| (r.tier.as_str(), r.family.as_str(), r.mode.as_str()))
            .collect::<Vec<_>>(),
        [
            ("precision-nvfp4", "precision", "FV_PRECISION_ARM=ltx-nvfp4"),
            ("ltxvae-gen", "ltxvae", "FV_VAE_GEN=1"),
        ]
    );
    for r in &restored {
        assert!(retired.contains(&r.tier), "{}: in FV_LAB_RETIRED", r.tier);
        assert!(retired.contains(&r.family), "{}: family retired", r.family);
        assert!(!branch(&r.family), "{}: no branch", r.family);
    }
    let fams = families(&lab);
    for f in &retired {
        assert!(fams.contains(f), "{f}: a script tier of lab/tiers.toml");
        assert!(!branch(f), "{f}: its runpod-matrix.sh branch is gone");
        for fx in [
            format!("{f}-dry-run.txt"),
            format!("{f}-gates.tsv"),
            format!("runpod-http-{f}.json"),
        ] {
            assert!(!fixture(&fx).trim().is_empty(), "{fx}: frozen, not empty");
        }
    }
    for f in fams.iter().filter(|f| !retired.contains(f)) {
        assert!(branch(f), "{f}: no branch and not in fv-lab-retired.sh");
    }
}
