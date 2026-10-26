//! The checked-in run plan (`lab/tiers.toml`, `lab/cells/*.toml`) against the
//! repository: every tree an arm names is a weights-manifest.tsv tree, every
//! gate a verify-weights.sh cell, and the arms the EU volume cannot serve are
//! exactly the ones the script skips there too (docs/adr/0019-fv-lab.md §4).

use std::collections::BTreeSet;
use std::path::PathBuf;

use fastvideo_deploy::launch::parse_eu_trees;
use fastvideo_lab::plan::{CellFile, Lab, TierWeights};
use fastvideo_lab::weights::{Source, TreeSources, WeightsPlan, DISK_RESERVE_GB};

/// The cells of the runpod-matrix.sh families on the volume: every cell but
/// those of a `volume+r2` tier (ADR-0019 §16): the `gold` tier's, which gate
/// on `variant:<id>` cells and read the trees of their family variant
/// (tests/gold_tier.rs), and `oracle-r2`'s, whose R2 trees are fetched
/// (`a_volume_r2_script_tier_reads_trees_on_the_volume_or_in_r2`).
fn script_cells(lab: &Lab) -> Vec<&CellFile> {
    let r2: BTreeSet<&str> = lab
        .tiers
        .tiers
        .iter()
        .filter(|t| t.weights == TierWeights::VolumeR2)
        .flat_map(|t| t.all_cells().map(String::as_str))
        .collect();
    lab.cells
        .values()
        .filter(|c| !r2.contains(c.name.as_str()))
        .collect()
}

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
    // A tree is a weights-manifest.tsv row, or a tree of the EU volume
    // (volumes.sh FV_EU_VOLUME_TREES) that is no hub row there (h3-ref2va:
    // two repos of plain files, recorded by hash in weights-sha256.tsv).
    let mut manifest: BTreeSet<String> = read("scripts/gpu/weights-manifest.tsv")
        .lines()
        .filter(|l| !l.starts_with('#'))
        .filter_map(|l| l.split('\t').next())
        .map(str::to_owned)
        .collect();
    manifest.extend(parse_eu_trees(&read("scripts/gpu/volumes.sh")));
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
    for c in script_cells(&lab) {
        for a in &c.arms {
            // An ungated arm (D1d §15: compare-dumps) reads no tree.
            let Some(gate) = &a.gate else {
                assert!(a.trees.is_empty(), "{}: ungated with trees", a.name);
                continue;
            };
            assert!(gates.contains(gate.as_str()), "{}: gate {gate}", a.name);
            for tree in &a.trees {
                assert!(
                    manifest.contains(tree),
                    "{}: tree {tree} not in weights-manifest.tsv or the EU volume's trees",
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
fn the_trees_tiers_lack_on_the_eu_volume_are_h3_8step_fastwan22_and_the_older_ltx2s() {
    // Every other tree an arm reads is on bz0ofkfx45 (volumes.sh
    // FV_EU_VOLUME_TREES): the GPU runs exercise every arm but these, which
    // both runners record as skipped there: h3-8step (R2-only since
    // 2026-10-13; rtx6000, the D1b families, the oracle's fasth3-8step
    // cells, fuse's 8-step arms), fastwan22-ti2v-5b (det-short's wan group,
    // ADR-0019 §15), and ltx23 and ltx2 (the ltx tier's LTX-2.3 and LTX-2.0
    // arms, D1f §17, gated in the script from then until its branch went,
    // §17.6).
    let lab = Lab::load(&repo_root()).expect("lab/ parses");
    let eu: BTreeSet<String> = parse_eu_trees(&read("scripts/gpu/volumes.sh"))
        .into_iter()
        .collect();
    let lacking: BTreeSet<&str> = script_cells(&lab)
        .into_iter()
        .flat_map(|c| &c.arms)
        .flat_map(|a| &a.trees)
        .filter(|t| !eu.contains(t.as_str()))
        .map(String::as_str)
        .collect();
    assert_eq!(
        lacking,
        BTreeSet::from(["h3-8step", "fastwan22-ti2v-5b", "ltx23", "ltx2"])
    );
    let older_ltx: Vec<&str> = script_cells(&lab)
        .into_iter()
        .flat_map(|c| &c.arms)
        .filter(|a| a.trees.iter().any(|t| t == "ltx23" || t == "ltx2"))
        .map(|a| a.name.as_str())
        .collect();
    assert_eq!(older_ltx, ["ltx23-two-stage", "ltx20-distilled-8step"]);
    let wan: Vec<&str> = script_cells(&lab)
        .into_iter()
        .flat_map(|c| &c.arms)
        .filter(|a| a.trees.iter().any(|t| t == "fastwan22-ti2v-5b"))
        .map(|a| a.name.as_str())
        .collect();
    assert_eq!(wan, ["wan-r1", "wan-r2", "wan-r3", "wan-timed"]);
}

/// `oracle-r2` (ADR-0019 §21), the one `volume+r2` tier cut from a script
/// family: its arms gate on verify-weights.sh cells and name manifest trees
/// like every script cell, each tree is on the EU volume or published in R2
/// (sfwan21-1.3b, R2-only since 2026-10-13), and its container disk holds its
/// largest arm's R2 trees plus the volume tiers' 120 GB.
#[test]
fn a_volume_r2_script_tier_reads_trees_on_the_volume_or_in_r2() {
    let lab = Lab::load(&repo_root()).expect("lab/ parses");
    let tier = lab.tier("oracle-r2").expect("oracle-r2");
    assert_eq!(tier.weights, TierWeights::VolumeR2);
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
    let sources = TreeSources::load(&repo_root()).expect("volumes.sh, weights-r2.tsv");
    let mut arms = Vec::new();
    for c in tier.all_cells().filter_map(|c| lab.cells.get(c)) {
        for a in &c.arms {
            let Some(gate) = &a.gate else {
                assert!(a.trees.is_empty(), "{}: ungated with trees", a.name);
                continue;
            };
            assert!(gates.contains(gate.as_str()), "{}: gate {gate}", a.name);
            for t in &a.trees {
                assert!(
                    manifest.contains(t),
                    "{}: {t} not in weights-manifest.tsv",
                    a.name
                );
                assert!(
                    matches!(sources.source(t), Source::R2(_) | Source::Volume(_)),
                    "{}: {t} neither on the EU volume nor in R2",
                    a.name
                );
            }
            arms.push((a.name.as_str(), a.trees.as_slice()));
        }
    }
    assert_eq!(arms.len(), 4, "the four sfwan13 generations");
    let plan = WeightsPlan::new(&sources, &arms).expect("its R2 trees");
    // sfwan21-1.3b, fetched once before the first arm and kept.
    assert_eq!(plan.fetch_bytes, 28_928_814_212);
    assert_eq!(plan.peak_bytes, 28_928_814_212);
    let peak_gb = plan.peak_bytes.div_ceil(1_000_000_000);
    assert!(u64::from(tier.container_disk_gb) >= peak_gb + DISK_RESERVE_GB);
    assert_eq!(tier.container_disk_gb, 150);
}

#[test]
fn every_tae_file_is_one_fetch_tae_sh_fetches() {
    // The pod's start command runs fetch-tae.sh (the tier's `tae = true`), and
    // the runner runs it again when an arm's file is missing.
    let lab = Lab::load(&repo_root()).expect("lab/ parses");
    let fetch = read("scripts/gpu/fetch-tae.sh");
    let mut n = 0;
    for c in lab.cells.values() {
        for a in &c.arms {
            if let Some(f) = &a.tae {
                assert!(fetch.contains(&format!("\nfetch {f} ")), "{}: {f}", a.name);
                n += 1;
            }
        }
    }
    assert!(n > 0, "some arm reads a tiny autoencoder");
    let tiers_with_tae_arms = lab.tiers.tiers.iter().filter(|t| {
        t.cells
            .iter()
            .filter_map(|c| lab.cells.get(c))
            .any(|c| c.arms.iter().any(|a| a.tae.is_some()))
    });
    for t in tiers_with_tae_arms {
        assert!(t.tae, "tier {} has TAE arms: tae = true", t.name);
    }
}

#[test]
fn every_cell_file_belongs_to_exactly_one_tier() {
    // But two: `ltxvae-gen` (ADR-0019 §12) is FV_VAE_GEN=1 runpod-matrix.sh
    // ltxvae, the ltxvae tier's cells with a gen cell after each workload's,
    // in the script's order. It shares exactly ltxvae's cells. And `hd5`
    // (§22, S2) is FV_PROMPTS=scripts/gpu/prompts-hd5.json runpod-matrix.sh
    // hd, whose 10 s arms that setting does not reach: it shares exactly hd's
    // `hd-10s`. No other tier shares any.
    let lab = Lab::load(&repo_root()).expect("lab/ parses");
    let cells = |t: &str| -> Vec<&str> {
        lab.tier(t)
            .unwrap_or_else(|| panic!("{t}"))
            .all_cells()
            .map(String::as_str)
            .collect()
    };
    let (hd, hd5) = (cells("hd"), cells("hd5"));
    assert_eq!(
        hd5.iter()
            .filter(|c| hd.contains(c))
            .copied()
            .collect::<Vec<_>>(),
        ["hd-10s"],
        "hd5 shares hd's 10 s cell only"
    );
    let ltxvae = &lab.tier("ltxvae").expect("ltxvae").cells;
    let gen = &lab.tier("ltxvae-gen").expect("ltxvae-gen").cells;
    let shared: Vec<&String> = gen.iter().filter(|c| ltxvae.contains(c)).collect();
    assert_eq!(
        shared,
        ltxvae.iter().collect::<Vec<_>>(),
        "ltxvae's cells, in order"
    );
    assert_eq!(
        gen.iter()
            .filter(|c| !ltxvae.contains(c))
            .collect::<Vec<_>>(),
        ["ltxvae-gen-4k5s", "ltxvae-gen-1080p20s"]
    );
    let mut seen = BTreeSet::new();
    for t in lab
        .tiers
        .tiers
        .iter()
        .filter(|t| t.name != "ltxvae-gen" && t.name != "hd5")
    {
        for c in t.all_cells() {
            assert!(
                seen.insert(c.as_str()),
                "lab/cells/{c}.toml is in two tiers"
            );
        }
    }
    for c in gen
        .iter()
        .filter(|c| !ltxvae.contains(c))
        .map(String::as_str)
        .chain(hd5.iter().copied().filter(|c| !hd.contains(c)))
    {
        assert!(seen.insert(c), "lab/cells/{c}.toml is in two tiers");
    }
}

#[test]
fn every_cell_file_belongs_to_a_tier() {
    let lab = Lab::load(&repo_root()).expect("lab/ parses");
    let used: BTreeSet<&str> = lab
        .tiers
        .tiers
        .iter()
        .flat_map(|t| t.all_cells().map(String::as_str))
        .collect();
    for name in lab.cells.keys() {
        assert!(
            used.contains(name.as_str()),
            "lab/cells/{name}.toml is in no tier"
        );
    }
}

#[test]
fn the_runtime_image_stays_python_free_so_the_runner_brings_python3() {
    // The runtime image is Python-free by design (its CI job fails on a
    // python3); a `volume+r2` run (ADR-0019 §16) installs python3 with apt at
    // its first fetch instead, and the pod's idle limit counts that install.
    let docker = read("docker/gpucheck.Dockerfile");
    let stage = docker
        .split("FROM base-cuda AS runtime")
        .nth(1)
        .and_then(|r| r.split("\nFROM ").next())
        .expect("the runtime stage");
    assert!(
        docker.contains("No Python, no PyTorch"),
        "the runtime stage's rule"
    );
    assert!(!stage.contains("python3"), "python3 in the runtime stage");
    assert_eq!(fastvideo_lab::run::APT_CAP_S, 600);
    assert_eq!(fastvideo_lab::weights::r2_idle_min(3600), 85);
}
