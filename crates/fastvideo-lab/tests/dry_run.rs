//! The dry-run equality (docs/adr/0019-fv-lab.md §5, D1b): for every tier of
//! `lab/tiers.toml`, `fv-lab run <tier> --dry-run` prints exactly what
//! `FV_DRY_RUN=1 runpod-matrix.sh <tier>` logs (`fixtures/<tier>-dry-run.txt`,
//! written from the script by `scripts/gpu/tests/fv-lab-goldens.sh`), and
//! every arm gates on the script's verify-weights.sh cell
//! (`fixtures/<tier>-gates.tsv`) and lists that cell's trees.

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

/// The tiers that are runpod-matrix.sh families: every tier but `gold`,
/// whose cells come from the gold files (tests/gold_tier.rs holds its dry run).
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
        let theirs: Vec<String> = fixture(&format!("{family}-dry-run.txt"))
            .lines()
            .map(str::to_owned)
            .collect();
        for (i, (a, b)) in ours.iter().zip(&theirs).enumerate() {
            assert_eq!(a, b, "{family}: line {i} differs");
        }
        assert_eq!(ours.len(), theirs.len(), "{family}: line count");
        let arms = ours
            .iter()
            .filter(|l| l.contains("] dry-run ") && !l.contains("] dry-run compare "))
            .count();
        let (_, cells) = lab.resolve(&family, None).expect("tier");
        assert_eq!(
            arms,
            cells.iter().map(|c| c.arms.len()).sum::<usize>(),
            "{family}: one dry-run line per arm"
        );
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
    for family in ["techniques", "b6"] {
        let (_, cells) = lab.resolve(family, None).expect("tier");
        assert!(Lab::needs_baseline(&cells), "{family}");
        let cfg = config(family);
        let base = format!("/fvscratch/baseline-{BASELINE_SHA}/fv-gpucheck");
        let mut pairs = 0;
        for s in steps(&lab, &cfg) {
            if let Step::Arm {
                name,
                command,
                baseline,
                ..
            } = s
            {
                assert_eq!(baseline, name.ends_with("-base"), "{family}: {name}");
                assert_eq!(command.contains(&base), baseline, "{family}: {name}");
                pairs += usize::from(baseline);
            }
        }
        assert!(pairs >= 4, "{family}: {pairs} baseline arms");
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
        Step::Compare { .. } => None,
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
        let steps = steps(&lab, &cfg);
        let fixture = fixture(&format!("{family}-gates.tsv"));
        let rows: Vec<Vec<&str>> = fixture.lines().map(|l| l.split('\t').collect()).collect();
        let ours: Vec<Vec<String>> = steps.iter().filter_map(gate_row).collect();
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
    // fastvideo (7), h3arms (10), precision (6) gate.
    assert_eq!((tae_arms, gates_seen), (16, 23));
}
