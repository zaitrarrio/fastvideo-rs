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

/// `wan` (ADR-0019 §22, S6): every arm gated on a verify-weights.sh cell
/// whose trees are all R2-only (none on the EU volume), but four ungated
/// ones (kernels-wan and the three compare-dumps); in the script's arm order
/// fv-lab's R2 plan (drop what the arm does not read, then fetch, §16.4)
/// makes eleven fetches, 428.10 GB, with at most wan21-t2v-14b's 80.41 GB on
/// the container disk at once, which the tier's 210 GB holds with the 120 GB
/// reserve. Derived from weights-r2.tsv's bytes, not typed in: each switch
/// of tree is one fetch of the new tree.
#[test]
fn the_wan_tier_fetches_its_trees_in_turn_within_its_disk() {
    let lab = Lab::load(&repo_root()).expect("lab/ parses");
    let tier = lab.tier("wan").expect("wan");
    assert_eq!(tier.weights, TierWeights::VolumeR2);
    assert_eq!(tier.min_ram_gb, Some(256), "the 14B arms' host load, as gold's");
    let verify = read("scripts/gpu/verify-weights.sh");
    let cells_line = verify
        .split("CELLS=(")
        .nth(1)
        .and_then(|r| r.split(')').next())
        .expect("verify-weights.sh CELLS=(…)");
    let gates: BTreeSet<&str> = cells_line.split_whitespace().collect();
    let sources = TreeSources::load(&repo_root()).expect("volumes.sh, weights-r2.tsv");
    let mut arms = Vec::new();
    let mut ungated = Vec::new();
    for c in tier.all_cells().filter_map(|c| lab.cells.get(c)) {
        for a in &c.arms {
            let Some(gate) = &a.gate else {
                assert!(a.trees.is_empty(), "{}: ungated with trees", a.name);
                ungated.push(a.name.as_str());
                continue;
            };
            assert!(gates.contains(gate.as_str()), "{}: gate {gate}", a.name);
            assert_eq!(a.trees, [gate.clone()], "{}: its gate's one tree", a.name);
            for t in &a.trees {
                assert!(
                    matches!(sources.source(t), Source::R2(_)),
                    "{}: {t} in R2 only",
                    a.name
                );
            }
            arms.push((a.name.as_str(), a.trees.as_slice()));
        }
    }
    assert_eq!(arms.len(), 45, "the script's 45 gated arms");
    assert_eq!(
        ungated,
        [
            "kernels-wan",
            "wan5b-oracle-diff",
            "wan5b-oracle-exact-diff",
            "fw22-oracle-diff"
        ]
    );
    // The expected sequence of fetches: a new tree whenever the arm's tree
    // differs from the one before (each arm reads one tree).
    let mut fetches: Vec<&str> = Vec::new();
    for (_, trees) in &arms {
        if fetches.last() != Some(&trees[0].as_str()) {
            fetches.push(trees[0].as_str());
        }
    }
    assert_eq!(
        fetches,
        [
            "fastwan21-1.3b",
            "sfwan21-1.3b",
            "wan22-ti2v-5b",
            "wan21-t2v-14b",
            "fastwan21-1.3b",
            "wan21-t2v-14b",
            "wan22-ti2v-5b",
            "fastwan22-ti2v-5b",
            "wan22-ti2v-5b",
            "fastwan22-ti2v-5b",
            "sfwan21-1.3b"
        ]
    );
    let bytes = |t: &str| sources.r2.get(t).map(|r| r.bytes).expect("an R2 tree");
    let plan = WeightsPlan::new(&sources, &arms).expect("its R2 trees");
    let fetched: Vec<&str> = plan
        .arms
        .iter()
        .flat_map(|(_, w)| w.fetch.iter().map(|t| t.dest.as_str()))
        .collect();
    assert_eq!(fetched, fetches, "one fetch per switch, nothing else");
    // Each fetch drops the tree before it, so one tree at a time.
    for (_, w) in &plan.arms {
        assert!(w.fetch.len() <= 1 && w.drop.len() <= 1, "{w:?}");
    }
    assert_eq!(
        plan.fetch_bytes,
        fetches.iter().map(|t| bytes(t)).sum::<u64>()
    );
    assert_eq!(plan.fetch_bytes, 428_103_503_172, "428.10 GB");
    assert_eq!(plan.peak_bytes, bytes("wan21-t2v-14b"));
    assert_eq!(plan.peak_bytes, 80_406_918_983, "80.41 GB");
    let peak_gb = plan.peak_bytes.div_ceil(1_000_000_000);
    assert!(u64::from(tier.container_disk_gb) >= peak_gb + DISK_RESERVE_GB);
    assert_eq!(tier.container_disk_gb, 210);
    // The GPU check's arms (§22.18): two trees, one after the other, 53.41 GB;
    // the script side stages both at once (174 GB with the reserve).
    let check: Vec<(&str, &[String])> = arms
        .iter()
        .copied()
        .filter(|(n, _)| ["wan13-dmd", "wan13-dmd-dense", "cli-fw22", "engine-fw22"].contains(n))
        .collect();
    let p = WeightsPlan::new(&sources, &check).expect("the check's trees");
    assert_eq!(p.fetch_bytes, bytes("fastwan21-1.3b") + bytes("fastwan22-ti2v-5b"));
    assert_eq!(p.fetch_bytes, 53_413_890_791);
    assert_eq!(p.peak_bytes, bytes("fastwan21-1.3b"));
    assert_eq!(p.fetch_bytes.div_ceil(1_000_000_000) + DISK_RESERVE_GB, 174);
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
    // Addition 16 (ADR-0019 §22.4): the arms that read `{tae}` without a
    // `tae` file are sfstream's two (`FASTVIDEO_TAE_DIR={tae}`), in a `tae =
    // true` tier (`Lab::load` refuses any other), and the file they need,
    // taew2_1, is one fetch-tae.sh fetches.
    let readers: Vec<(&str, &str)> = lab
        .tiers
        .tiers
        .iter()
        .flat_map(|t| {
            t.all_cells()
                .filter_map(|c| lab.cells.get(c))
                .flat_map(|c| c.tae_dir_arms())
                .map(move |a| (t.name.as_str(), a))
        })
        .collect();
    let (wan, others): (Vec<(&str, &str)>, Vec<(&str, &str)>) =
        readers.iter().partition(|(t, _)| *t == "wan");
    assert_eq!(
        others,
        [
            ("sfstream", "sfstream-main"),
            ("sfstream", "sfstream-10min")
        ]
    );
    assert!(lab.tier("sfstream").is_some_and(|t| t.tae));
    // sfstream's arms read the directory; `wan stream` reads taew2_1 in it.
    assert!(fetch.contains("\nfetch taew2_1.safetensors "));
    // wan (ADR-0019 §22, S6): each of its 45 gated arms reads the directory
    // (FASTVIDEO_TAE_DIR={tae}), the oracle arms taew2_2 in it too; the `auto`
    // decoder of the distilled presets looks for taew2_1 there, the Wan 2.2
    // latent space (wan5b-taehv, fw22-taehv, the oracle arms) for taew2_2,
    // and fetch-tae.sh fetches both.
    let wan_tier = lab.tier("wan").expect("wan");
    assert!(wan_tier.tae);
    let gated: Vec<&str> = wan_tier
        .all_cells()
        .filter_map(|c| lab.cells.get(c))
        .flat_map(|c| &c.arms)
        .filter(|a| a.gate.is_some())
        .map(|a| a.name.as_str())
        .collect();
    assert_eq!(gated.len(), 45);
    assert_eq!(
        wan.iter().map(|(_, a)| *a).collect::<Vec<_>>(),
        gated,
        "every gated wan arm, none of the ungated four"
    );
    assert!(fetch.contains("\nfetch taew2_2.safetensors "));
}

#[test]
fn every_tae_file_a_tae_dir_arm_names_is_one_fetch_tae_sh_fetches() {
    // Addition 16 (ADR-0019 §22.4): an arm that reads `{tae}` without a `tae`
    // file is not TAE-gated, so nothing but the pod start's fetch-tae.sh (its
    // tier's `tae = true`) tries to provide what it reads. Every
    // `{tae}/<file>` such an arm names, in any tier (cells and extra cells),
    // in `env` or the spliced argv, must then be a file fetch-tae.sh fetches.
    // (`wan`'s oracle arms read `--taehv {tae}/taew2_2.safetensors`, which
    // fetch-tae.sh fetches since S6.)
    fn tae_files(s: &str) -> Vec<&str> {
        let mut out = Vec::new();
        let mut rest = s;
        while let Some(i) = rest.find("{tae}/") {
            let after = &rest[i + "{tae}/".len()..];
            let end = after.find(['/', ' ', '"', ',']).unwrap_or(after.len());
            out.push(&after[..end]);
            rest = &after[end..];
        }
        out
    }
    assert_eq!(
        tae_files("--taehv {tae}/taew2_2.safetensors x={tae}/a,{tae}"),
        ["taew2_2.safetensors", "a"]
    );
    let lab = Lab::load(&repo_root()).expect("lab/ parses");
    let fetch = read("scripts/gpu/fetch-tae.sh");
    let mut readers_seen = 0;
    for t in &lab.tiers.tiers {
        for c in t.all_cells().filter_map(|c| lab.cells.get(c)) {
            let readers = c.tae_dir_arms();
            for a in c.arms.iter().filter(|a| readers.contains(&a.name.as_str())) {
                readers_seen += 1;
                assert!(t.tae, "{}: tier {} without tae = true", a.name, t.name);
                let argv = c.spliced(a).expect("lists splice");
                for s in a.env_entries().iter().chain(&argv) {
                    for file in tae_files(s) {
                        assert!(
                            fetch.contains(&format!("\nfetch {file} ")),
                            "{} (tier {}) reads {{tae}}/{file}, which fetch-tae.sh does not fetch",
                            a.name,
                            t.name
                        );
                    }
                }
            }
        }
    }
    assert!(
        readers_seen >= 2 + 45,
        "sfstream's two arms and wan's 45 at least"
    );
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
