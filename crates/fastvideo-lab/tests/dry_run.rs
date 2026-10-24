//! The dry-run equality (docs/adr/0019-fv-lab.md §5): `fv-lab run rtx6000
//! --dry-run` prints exactly what `FV_DRY_RUN=1 runpod-matrix.sh rtx6000`
//! logs (`fixtures/rtx6000-dry-run.txt`, written from the script by
//! `scripts/gpu/tests/fv-lab-goldens.sh`), and every arm gates on the
//! script's verify-weights.sh cell (`fixtures/rtx6000-gates.tsv`) and lists
//! that cell's trees.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use fastvideo_lab::plan::{Lab, Step};
use fastvideo_lab::run::{dry_run_lines, RunConfig};

// The inputs of fv-lab-goldens.sh.
const FAMILY: &str = "rtx6000";
const TAG: &str = "1234567-10241200";

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn fixture(name: &str) -> String {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("fixtures")
        .join(name);
    std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("{}: {e}", p.display()))
}

fn config() -> RunConfig {
    let mut c = RunConfig::new(FAMILY, Path::new("/opt/fastvideo-rs"));
    c.work = "/workspace".into();
    c.scratch = "/fvscratch".into();
    c.tag = Some(TAG.into());
    c
}

fn steps(lab: &Lab, cfg: &RunConfig) -> Vec<Step> {
    let (_, cells) = lab.resolve(FAMILY, None).expect("tier");
    Lab::steps(&cells, &cfg.bin, &cfg.paths()).expect("render")
}

#[test]
fn dry_run_equals_runpod_matrix_sh() {
    let lab = Lab::load(&repo_root()).expect("lab/ parses");
    let cfg = config();
    let ours = dry_run_lines(&cfg, &steps(&lab, &cfg));
    let theirs: Vec<String> = fixture("rtx6000-dry-run.txt")
        .lines()
        .map(str::to_owned)
        .collect();
    for (i, (a, b)) in ours.iter().zip(&theirs).enumerate() {
        assert_eq!(a, b, "line {i} differs");
    }
    assert_eq!(ours.len(), theirs.len(), "line count");
    assert!(ours.iter().filter(|l| l.contains("] dry-run ")).count() == 7);
}

#[test]
fn arms_gate_on_the_scripts_cells_and_name_their_trees() {
    let lab = Lab::load(&repo_root()).expect("lab/ parses");
    let cfg = config();
    let arms: Vec<(String, String, Vec<String>)> = steps(&lab, &cfg)
        .into_iter()
        .filter_map(|s| match s {
            Step::Arm {
                name, gate, trees, ..
            } => Some((name, gate, trees)),
            _ => None,
        })
        .collect();
    let gates = fixture("rtx6000-gates.tsv");
    let rows: Vec<Vec<&str>> = gates.lines().map(|l| l.split('\t').collect()).collect();
    assert_eq!(arms.len(), rows.len());
    for ((name, gate, trees), row) in arms.iter().zip(&rows) {
        assert_eq!((name.as_str(), gate.as_str()), (row[0], row[1]));
        // The roots of the cell's needs (`root:component`; `:path` reads a
        // file directly under the weight root, whose first part is the tree).
        let roots: BTreeSet<String> = row[2]
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
            "arm {name}: trees vs verify-weights.sh {gate}"
        );
    }
}
