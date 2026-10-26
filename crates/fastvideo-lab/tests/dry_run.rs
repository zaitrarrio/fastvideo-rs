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
//!
//! A sibling tier (ADR-0019 §21, `FV_LAB_SIBLINGS`: `oracle-r2`, the `oracle`
//! family's `sfwan13` target) is its family's live branch under the tier's
//! settings, read with the tier's name (`common::Sibling`); being
//! `volume+r2`, fv-lab reads `{weights}` from the merged root where the script
//! reads the volume (`common::merged_root`), and its R2 plan lines (`dry-run
//! weights …`, `dry-run r2 <arm>: …`, ADR-0019 §16.4) are its own, checked on
//! their own (`the_oracle_r2_tier_dry_runs_as_the_oracle_family_with_sfwan13`).

mod common;

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use fastvideo_lab::goldcell::GOLD_TIER;
use fastvideo_lab::plan::{Lab, Step, TierWeights};
use fastvideo_lab::run::{dry_run_lines, RunConfig};
use fastvideo_lab::weights::{R2Setup, TreeSources};

// The inputs of fv-lab-goldens.sh.
const TAG: &str = "1234567-10241200";
const BASELINE_SHA: &str = "7654321";

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// A tier's dry-run fixture as fv-lab must print it: the script's lines,
/// with a restored mode's or a sibling tier's family named by its tier
/// (`common::Restored`, `common::Sibling`), and a `volume+r2` tier's weight
/// root its merged root (`common::merged_root`).
fn dry_run_fixture(lab: &Lab, tier: &str) -> Vec<String> {
    let text = fixture(&format!("{tier}-dry-run.txt"));
    let r2 = lab
        .tier(tier)
        .is_some_and(|t| t.weights == TierWeights::VolumeR2);
    let root = |l: String| if r2 { common::merged_root(&l) } else { l };
    match (common::restored_tier(tier), common::sibling_tier(tier)) {
        (Some(r), _) => text.lines().map(|l| root(r.dry_run_line(l))).collect(),
        (None, Some(s)) => text.lines().map(|l| root(s.dry_run_line(l))).collect(),
        (None, None) => text.lines().map(|l| root(l.to_owned())).collect(),
    }
}

/// fv-lab's own lines of a `volume+r2` run's dry run, which the script has
/// no counterpart of: its weights line and each arm's R2 step (ADR-0019 §16.4).
fn r2_line(l: &str) -> bool {
    l.contains("] dry-run weights ") || l.contains("] dry-run r2 ")
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
    // A volume+r2 tier's runner (ADR-0019 §16): the merged root, no key (a
    // dry run fetches nothing).
    let lab = Lab::load(&repo_root()).expect("lab/ parses");
    if lab
        .tier(family)
        .is_some_and(|t| t.weights == TierWeights::VolumeR2)
    {
        let sources = TreeSources::load(&repo_root()).expect("volumes.sh, weights-r2.tsv");
        c.r2 = Some(R2Setup::new(sources, &c.scratch, None));
    }
    c
}

fn steps(lab: &Lab, cfg: &RunConfig) -> Vec<Step> {
    let (_, cells) = lab.resolve(&cfg.family, None).expect("tier");
    Lab::steps(&cells, &cfg.bin, &cfg.paths()).expect("render")
}

/// The tiers that are (or were, before fv-lab-retired.sh) runpod-matrix.sh
/// families: every tier but `gold`, whose cells come from the gold files
/// (tests/gold_tier.rs holds its dry run), and the tiers made from a port
/// one-off (`FV_LAB_EXPERIMENTS`, tests/experiment_tiers.rs holds theirs).
fn families(lab: &Lab) -> Vec<String> {
    let experiments = common::experiments();
    lab.tiers
        .tiers
        .iter()
        .filter(|t| t.name != GOLD_TIER && !experiments.contains(&t.name))
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
        let ours: Vec<String> = dry_run_lines(&cfg, &steps(&lab, &cfg))
            .into_iter()
            .filter(|l| !r2_line(l))
            .collect();
        let theirs = dry_run_fixture(&lab, &family);
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

/// One row of `<tier>-gates.tsv`, what `FV_DRY_RUN_GATES` recorded: a
/// gated arm (an ungated one, the script's bare `run_cell`, records none).
fn gate_row(s: &Step) -> Option<Vec<String>> {
    match s {
        Step::Arm {
            name,
            gate: Some(gate),
            tae,
            ..
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
        _ => None,
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
                name,
                gate: Some(gate),
                trees,
                ..
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
    // fastvideo (7), h3arms (10), precision (6), precision-nvfp4 (2) and
    // fuse (3, exact, D1f) gate.
    assert_eq!((tae_arms, gates_seen), (16, 28));
}

/// A tier's extra cells (ADR-0019 §20.3: `extra_cells`, run on request):
/// `fv-lab run oracle --dry-run` selecting every arm of the ten clip-metric
/// targets prints `FV_DRY_RUN=1 runpod-matrix.sh oracle` with
/// `FV_ORACLE_TARGETS` naming them (`fixtures/oracle-extra-dry-run.txt`),
/// line for line, the derived-clip metric commands (`dry-run metric <tag>:
/// <ffmpeg / ffprobe …>`) included, after a `skip <cell> (no arm in
/// FV_CELLS)` per default cell (the script, given its targets, names no
/// other); its gates are `oracle-extra-gates.tsv`'s. Without a selection
/// the extra cells do not run (the default run is `oracle-dry-run.txt`,
/// above).
#[test]
fn the_extra_cells_dry_run_as_the_script_with_their_targets() {
    let lab = Lab::load(&repo_root()).expect("lab/ parses");
    let tier = lab.tier("oracle").expect("oracle");
    assert_eq!(tier.extra_cells.len(), 10);
    let extra_arms: Vec<String> = tier
        .extra_cells
        .iter()
        .flat_map(|c| &lab.cells[c].arms)
        .map(|a| a.name.clone())
        .collect();
    let mut cfg = config("oracle");
    cfg.only = Some(extra_arms.clone());
    let (_, cells) = lab
        .resolve_selected("oracle", None, cfg.only.as_deref())
        .expect("tier");
    assert_eq!(cells.len(), 19);
    let steps = Lab::steps(&cells, &cfg.bin, &cfg.paths()).expect("render");
    let ours = dry_run_lines(&cfg, &steps);
    let skips: Vec<String> = tier
        .cells
        .iter()
        .map(|c| format!("[oracle] skip {c} (no arm in FV_CELLS)"))
        .collect();
    assert_eq!(
        ours[1..=skips.len()],
        skips[..],
        "the default cells, skipped"
    );
    let mut rest = vec![ours[0].clone()];
    rest.extend(ours[skips.len() + 1..].iter().cloned());
    let theirs = dry_run_fixture(&lab, "oracle-extra");
    for (i, (a, b)) in rest.iter().zip(&theirs).enumerate() {
        assert_eq!(a, b, "oracle-extra: line {i} differs");
    }
    assert_eq!(rest.len(), theirs.len(), "oracle-extra: line count");
    let metrics = rest
        .iter()
        .filter(|l| l.contains("] dry-run metric "))
        .count();
    // i2v 6, kf 12, ref2v 4, a2v 5, a2v-i2v 11, retake* and extend 4 x 11, guided 5.
    assert_eq!(metrics, 6 + 12 + 4 + 5 + 11 + 4 * 11 + 5);
    // Each kind: ffprobe, a whole clip, kept frames, a pinned frame.
    for k in [
        "ffprobe -v error",
        r"format=yuv420p\[b\]\;\[a\]\[b\]ssim",
        r"select=\'lt\(n\\\,33\)",
        r"select=eq\(n\\\,120\)",
    ] {
        assert!(rest.iter().any(|l| l.contains(k)), "{k}");
    }
    // The gates: every gated arm of the ten targets, as the script recorded.
    let rows: Vec<Vec<String>> = fixture("oracle-extra-gates.tsv")
        .lines()
        .map(|l| l.split('\t').take(3).map(str::to_owned).collect())
        .collect();
    let gated: Vec<Vec<String>> = Step::flatten(&steps)
        .into_iter()
        .filter_map(|s| match s {
            Step::Arm {
                name,
                gate: Some(g),
                ..
            } if extra_arms.contains(name) => Some(vec!["arm".to_owned(), name.clone(), g.clone()]),
            _ => None,
        })
        .collect();
    assert_eq!(gated, rows);
    // Without a selection the tier is its nine cells, as before.
    let (_, default) = lab.resolve("oracle", None).expect("tier");
    assert_eq!(default.len(), 9);
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

/// `FV_SCRIPT_RETIRED` of `scripts/gpu/fv-lab-retired.sh`: the families
/// retired with no tier (ADR-0019 §18).
fn script_retired() -> Vec<String> {
    let p = repo_root().join("scripts/gpu/fv-lab-retired.sh");
    let t = std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("{}: {e}", p.display()));
    let line = t
        .lines()
        .find_map(|l| l.strip_prefix("FV_SCRIPT_RETIRED=\""))
        .and_then(|l| l.strip_suffix('"'))
        .expect("FV_SCRIPT_RETIRED=\"…\"");
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
fn families_retired_with_no_tier_have_no_branch_and_no_tier() {
    // D1g (ADR-0019 §18): h3 (§17.4), the B4 families (§17.3 option (b)) and
    // §14.7's rows left runpod-matrix.sh with no tier; then §22's S1 nine
    // (FVID-2026-10-26-d1-remaining): no branch, no family-specific code left
    // in the script (eval's prompt and LPIPS defaults), no tier of
    // lab/tiers.toml, no fixtures, and in neither FV_LAB_RETIRED nor
    // FV_LAB_EXPERIMENTS (runpod-http.sh refuses them instead of routing).
    let lab = Lab::load(&repo_root()).expect("lab/ parses");
    let matrix = std::fs::read_to_string(repo_root().join("scripts/gpu/runpod-matrix.sh"))
        .expect("runpod-matrix.sh");
    let gone = script_retired();
    assert_eq!(
        gone,
        [
            "h3",
            "hunyuan",
            "sana-video",
            "b200",
            "writer",
            "precision-debug",
            "identity",
            "b4b",
            "b4c",
            "b4c2",
            "h3attn",
            "sfquality",
            "headline",
            "speechtest",
            "cold",
            "eval",
            "ltxoffload",
            "solbench",
            "sol-lingbot",
            "sol-cosmos3",
        ]
    );
    let tiers: Vec<&str> = lab.tiers.tiers.iter().map(|t| t.name.as_str()).collect();
    let retired = retired();
    let experiments = common::experiments();
    for f in &gone {
        assert!(
            !has_branch(&matrix, f),
            "{f}: its runpod-matrix.sh branch is gone"
        );
        assert!(
            !matrix.contains(&format!("\"$FAMILY\" == {f} "))
                && !matrix.contains(&format!("\"$FAMILY\" == {f}\n")),
            "{f}: no family-specific code left in runpod-matrix.sh"
        );
        assert!(
            !tiers.contains(&f.as_str()),
            "{f}: no tier of lab/tiers.toml"
        );
        assert!(!retired.contains(f), "{f}: not in FV_LAB_RETIRED");
        assert!(!experiments.contains(f), "{f}: not in FV_LAB_EXPERIMENTS");
        for fx in [
            format!("{f}-dry-run.txt"),
            format!("{f}-gates.tsv"),
            format!("runpod-http-{f}.json"),
        ] {
            let p = repo_root().join("crates/fastvideo-lab/fixtures").join(&fx);
            assert!(!p.exists(), "{fx}: no fixture for a family with no tier");
        }
    }
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
        // D1d's verdict family, after its GPU gate (ADR-0019 §11.6).
        "b4a",
        // D1d's determinism families, after their GPU checks (§15.10).
        "det-short",
        "determinism",
        // D1f's families, after their GPU checks (§17.6).
        "fuse",
        "attn3",
        "ltx",
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
        // A sibling tier (FV_LAB_SIBLINGS) is its family's branch, and has
        // none of its own.
        match common::sibling_tier(f) {
            Some(s) => {
                assert!(
                    branch(&s.family),
                    "{f}: its family {} has a branch",
                    s.family
                );
                assert!(!branch(f), "{f}: a sibling tier has no branch of its own");
            }
            None => assert!(branch(f), "{f}: no branch and not in fv-lab-retired.sh"),
        }
    }
}

/// The `oracle-r2` tier (ADR-0019 §20.5, §21): `fv-lab run oracle-r2
/// --dry-run` is `FV_DRY_RUN=1 FV_ORACLE_TARGETS=sfwan13 runpod-matrix.sh
/// oracle` (`fixtures/oracle-r2-dry-run.txt`) line for line, read with the
/// tier's name and its merged weight root: the reference, the tensor-diff
/// arms, the two decodes (`dry-run metric sfwan13-frames: ffmpeg …`), the
/// directory compares (`dry-run compare <tag> `) in the script's order, the
/// pre-fix and composed arms; plus fv-lab's R2 plan, which fetches
/// sfwan21-1.3b (28.93 GB) once, before its first arm.
#[test]
fn the_oracle_r2_tier_dry_runs_as_the_oracle_family_with_sfwan13() {
    let lab = Lab::load(&repo_root()).expect("lab/ parses");
    let tier = lab.tier("oracle-r2").expect("oracle-r2");
    assert_eq!(tier.weights, TierWeights::VolumeR2);
    assert_eq!(tier.cells, ["oracle-sfwan13"]);
    assert!(tier.extra_cells.is_empty());
    assert_eq!(
        common::sibling_tier("oracle-r2").map(|s| s.family),
        Some("oracle".to_owned())
    );
    let cfg = config("oracle-r2");
    let all = dry_run_lines(&cfg, &steps(&lab, &cfg));
    let r2: Vec<&String> = all.iter().filter(|l| r2_line(l)).collect();
    assert_eq!(r2.len(), 2, "{r2:?}");
    assert!(
        r2[0].starts_with("[oracle-r2] dry-run weights /fvscratch/weights: ")
            && r2[0].contains("; 1 R2 trees (28.93 GB) fetched before the arms that read them, at most 28.93 GB on the container disk at once"),
        "{}",
        r2[0]
    );
    assert_eq!(
        r2[1],
        "[oracle-r2] dry-run r2 oracle-sfwan13: fetch sfwan21-1.3b@4b44356635ae5e927ca552a220f768022be76004 (28.93 GB) into /fvscratch/weights"
    );
    // The R2 line comes right before the first arm that reads the tree.
    let i = all.iter().position(|l| l == r2[1]).expect("the r2 line");
    assert!(all[i + 1].starts_with("[oracle-r2] dry-run oracle-sfwan13: env "));
    let ours: Vec<String> = all.into_iter().filter(|l| !r2_line(l)).collect();
    let theirs = dry_run_fixture(&lab, "oracle-r2");
    assert_eq!(ours, theirs);
    // The script's order: the arms, the decodes and compares where it runs them.
    let kinds: Vec<&str> = ours
        .iter()
        .map(|l| {
            let r = l.split_once("] ").map_or("", |(_, r)| r);
            if r.starts_with("dry-run metric sfwan13-frames: ffmpeg -loglevel error -y -i ") {
                "decode"
            } else if r.starts_with("dry-run compare ") {
                "compare"
            } else if r.starts_with("dry-run reference ") {
                "reference"
            } else if r.starts_with("dry-run oracle-") {
                "arm"
            } else {
                "frame"
            }
        })
        .collect();
    assert_eq!(
        kinds,
        [
            "frame",
            "reference",
            "arm",
            "arm",
            "arm",
            "arm",
            "arm",
            "decode",
            "decode",
            "compare",
            "compare",
            "arm",
            "arm",
            "compare",
            "arm",
            "arm",
            "arm",
            "frame"
        ]
    );
    for want in [
        "[oracle-r2] dry-run metric sfwan13-frames: ffmpeg -loglevel error -y -i /fvscratch/oracle-ref/sfwan13/dump/ref.mp4 -start_number 0 /fvscratch/oracle-ref/sfwan13/frames/frame-%03d.png ",
        "[oracle-r2] dry-run compare oracle-sfwan13-frames-vs-ref ",
        "[oracle-r2] dry-run compare oracle-sfwan13-codec-floor ",
        "[oracle-r2] dry-run compare oracle-sfwan13-legacy-frames-vs-ref ",
    ] {
        assert!(ours.iter().any(|l| l == want), "{want}");
    }
    // Every gated arm reads the tree from the merged root, never the volume.
    for l in ours.iter().filter(|l| l.contains(" --weights ")) {
        assert!(
            l.contains(" --weights /fvscratch/weights/sfwan21-1.3b "),
            "{l}"
        );
    }
}
