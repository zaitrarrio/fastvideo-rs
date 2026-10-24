//! The checked-in run plan (`lab/tiers.toml`, `lab/cells/*.toml`) against the
//! repository: every tree an arm names is a weights-manifest.tsv tree, every
//! gate a verify-weights.sh cell, and the arms the EU volume cannot serve are
//! exactly the ones the script skips there too (docs/adr/0019-fv-lab.md §4).

use std::collections::BTreeSet;
use std::path::PathBuf;

use fastvideo_deploy::launch::parse_eu_trees;
use fastvideo_lab::plan::Lab;

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn read(rel: &str) -> String {
    let p = repo_root().join(rel);
    std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("{}: {e}", p.display()))
}

#[test]
fn trees_and_gates_exist() {
    let lab = Lab::load(&repo_root()).expect("lab/ parses");
    let manifest: BTreeSet<String> = read("scripts/gpu/weights-manifest.tsv")
        .lines()
        .filter(|l| !l.starts_with('#'))
        .filter_map(|l| l.split('\t').next())
        .map(str::to_owned)
        .collect();
    let verify = read("scripts/gpu/verify-weights.sh");
    let cells_line = verify
        .split("CELLS=(")
        .nth(1)
        .and_then(|r| r.split(')').next())
        .expect("verify-weights.sh CELLS=(…)");
    let gates: BTreeSet<&str> = cells_line.split_whitespace().collect();
    for t in &lab.tiers.tiers {
        for p in &t.preflight {
            assert!(gates.contains(p.as_str()), "tier {}: preflight {p}", t.name);
        }
    }
    for c in lab.cells.values() {
        for a in &c.arms {
            assert!(
                gates.contains(a.gate.as_str()),
                "{}: gate {}",
                a.name,
                a.gate
            );
            for tree in &a.trees {
                assert!(
                    manifest.contains(tree),
                    "{}: tree {tree} not in weights-manifest.tsv",
                    a.name
                );
            }
        }
    }
}

#[test]
fn only_fasth3_8step_needs_a_tree_the_eu_volume_lacks() {
    let lab = Lab::load(&repo_root()).expect("lab/ parses");
    let eu: BTreeSet<String> = parse_eu_trees(&read("scripts/gpu/volumes.sh"))
        .into_iter()
        .collect();
    let tier = lab.tier("rtx6000").expect("rtx6000");
    let missing: Vec<String> = tier
        .cells
        .iter()
        .filter_map(|c| lab.cells.get(c))
        .flat_map(|c| &c.arms)
        .filter(|a| a.trees.iter().any(|t| !eu.contains(t)))
        .map(|a| a.name.clone())
        .collect();
    // h3-8step is R2-only since 2026-10-13: runpod-matrix.sh records the arm
    // as skipped ("weights incomplete") on bz0ofkfx45, and so does fv-lab.
    assert_eq!(missing, ["fasth3-8step"]);
}

#[test]
fn every_cell_file_belongs_to_a_tier() {
    let lab = Lab::load(&repo_root()).expect("lab/ parses");
    let used: BTreeSet<&str> = lab
        .tiers
        .tiers
        .iter()
        .flat_map(|t| t.cells.iter().map(String::as_str))
        .collect();
    for name in lab.cells.keys() {
        assert!(
            used.contains(name.as_str()),
            "lab/cells/{name}.toml is in no tier"
        );
    }
}
