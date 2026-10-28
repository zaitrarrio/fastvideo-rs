//! The tiers made from a port one-off of `scripts/gpu/experiments/`
//! (docs/adr/0019-fv-lab.md §14, D1e; `FV_LAB_EXPERIMENTS` of
//! `scripts/gpu/fv-lab-retired.sh`). None was a runpod-matrix.sh family, so
//! there is no script dry run to equal. Instead:
//!
//! - every command an arm runs is the experiment's own pod script's, cut from
//!   it and run with its binary stood in by
//!   `scripts/gpu/tests/fv-lab-experiment-goldens.sh`
//!   (`fixtures/experiments/<tier>-script.txt`), word for word after the
//!   path mapping of ADR-0019 §14.3 (and nothing else);
//! - fv-lab's own dry run is the tier's golden (`fixtures/<tier>-dry-run.txt`),
//!   frozen from the start like the restored modes' (§12): a change to the
//!   tier's commands changes it in the same PR.
//!
//! The `sol-bench-<set>` tiers (ADR-0019 §23) are made the same way from
//! `scripts/gpu/sol-bench-pod.sh`, one tier per `FV_SOL_SET`
//! (`fixtures/experiments/sol-bench-script.txt`): their arms are the set's
//! cells word for word, with no mapping but the weight root.

mod common;

use std::path::{Path, PathBuf};

use std::collections::BTreeSet;

use fastvideo_deploy::launch::parse_eu_trees;
use fastvideo_lab::plan::{Lab, Step, TierWeights};
use fastvideo_lab::run::{dry_run_lines, RunConfig};
use fastvideo_lab::weights::{R2Setup, TreeSources, WeightsPlan, DISK_RESERVE_GB};

// The inputs of fv-lab-goldens.sh, as tests/dry_run.rs uses them.
const TAG: &str = "1234567-10241200";
const RUNS: &str = "/fvscratch/runs/h3-graphs/1234567-10241200";

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn fixture(name: &str) -> String {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("fixtures")
        .join(name);
    std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("{}: {e}", p.display()))
}

fn config(tier: &str) -> RunConfig {
    let mut c = RunConfig::new(tier, Path::new("/opt/fastvideo-rs"));
    c.work = "/workspace".into();
    c.scratch = "/fvscratch".into();
    c.tag = Some(TAG.into());
    c.tae_dir = Some("/workspace/weights/auxiliary/tae".into());
    c
}

fn steps(lab: &Lab, tier: &str) -> Vec<Step> {
    let cfg = config(tier);
    let (_, cells) = lab.resolve(tier, None).expect("tier");
    Lab::steps(&cells, &cfg.bin, &cfg.paths()).expect("render")
}

/// The arms of a tier, in run order: (name, gate, command); an ungated arm
/// (ADR-0019 §15) has the gate `""`.
fn arms(steps: &[Step]) -> Vec<(String, String, Vec<String>)> {
    Step::flatten(steps)
        .into_iter()
        .filter_map(|s| match s {
            Step::Arm {
                name,
                gate,
                command,
                ..
            } => Some((
                name.clone(),
                gate.clone().unwrap_or_default(),
                command.clone(),
            )),
            _ => None,
        })
        .collect()
}

#[test]
fn experiment_tiers_are_tiers_without_a_script_family() {
    let lab = Lab::load(&repo_root()).expect("lab/ parses");
    let experiments = common::experiments();
    let mut want = vec!["h3-graphs".to_owned()];
    want.extend(sol_sets().iter().map(|(s, _)| format!("sol-bench-{s}")));
    assert_eq!(
        experiments, want,
        "h3-graphs, then sol-bench's sets in the script's order"
    );
    let retired_line = std::fs::read_to_string(repo_root().join("scripts/gpu/fv-lab-retired.sh"))
        .expect("fv-lab-retired.sh");
    let retired = retired_line
        .lines()
        .find_map(|l| l.strip_prefix("FV_LAB_RETIRED=\""))
        .expect("FV_LAB_RETIRED");
    let matrix = std::fs::read_to_string(repo_root().join("scripts/gpu/runpod-matrix.sh"))
        .expect("runpod-matrix.sh");
    for t in &experiments {
        assert!(lab.tier(t).is_some(), "{t}: a tier of lab/tiers.toml");
        assert!(
            !retired
                .split_whitespace()
                .any(|r| r.trim_end_matches('"') == t),
            "{t}: not a retired family"
        );
        assert!(
            !matrix
                .lines()
                .any(|l| l.trim_start().starts_with(&format!("{t})"))),
            "{t}: no runpod-matrix.sh branch"
        );
        assert!(
            !fixture(&format!("runpod-http-{t}.json")).trim().is_empty(),
            "{t}: the pod body golden (fv-lab-goldens.sh)"
        );
    }
}

#[test]
fn h3_graphs_dry_run_is_its_golden() {
    let lab = Lab::load(&repo_root()).expect("lab/ parses");
    let ours = dry_run_lines(&config("h3-graphs"), &steps(&lab, "h3-graphs"));
    let golden = fixture("h3-graphs-dry-run.txt");
    assert_eq!(
        ours,
        golden.lines().collect::<Vec<_>>(),
        "fixtures/h3-graphs-dry-run.txt is fv-lab's own dry run of the tier (ADR-0019 §14.3); \
         this run prints:\n{}",
        ours.join("\n")
    );
}

/// The pod script's paths, mapped to the arm's (ADR-0019 §14.3): its
/// directory under /srv is the arm's run directory, its `clip/` the arm's
/// `frames/`, its /fvw scratch files the tier's.
fn map_word(word: &str, script_dir: &str, arm: &str) -> String {
    let arm_dir = format!("{RUNS}/{arm}");
    let pairs = [
        (format!("{script_dir}/clip"), format!("{arm_dir}/frames")),
        (format!("{script_dir}/report"), format!("{arm_dir}/report")),
        (
            "/fvw/adaln-turbo.cache".to_owned(),
            format!("{RUNS}/h3-graphs-turbo-adaln.cache"),
        ),
        (
            "/fvw/text-cache".to_owned(),
            "/fvscratch/h3-graphs-text-cache".to_owned(),
        ),
        (
            "/fvw/prompts-768.json".to_owned(),
            "/opt/fastvideo-rs/scripts/gpu/prompts-h3-graphs-768.json".to_owned(),
        ),
        (
            "/fvw/prompts-trace-768p.json".to_owned(),
            "/opt/fastvideo-rs/scripts/gpu/prompts-h3-graphs-trace-768p.json".to_owned(),
        ),
    ];
    for (from, to) in pairs {
        if word == from {
            return to;
        }
    }
    assert!(
        !word.starts_with("/srv") && !word.starts_with("/fvw"),
        "{arm}: the script's path {word} has no mapping"
    );
    word.to_owned()
}

#[test]
fn h3_graphs_runs_the_pod_scripts_commands() {
    // The script's h3-turbo processes, in its order (`trace`, then `cells`
    // with FV_ARMS's default "sync lazy graphs"), and the arms they are.
    let script = fixture("experiments/h3-graphs-script.txt");
    let wanted = [
        ("/srv/trace/turbo-768p", "h3-graphs-turbo-trace-768p"),
        ("/srv/cells/turbo-sync", "h3-graphs-turbo-sync"),
        ("/srv/cells/turbo-lazy", "h3-graphs-turbo-lazy"),
        ("/srv/cells/turbo-graphs", "h3-graphs-turbo-graphs"),
    ];
    let turbo: Vec<(&str, Vec<&str>)> = script
        .lines()
        .map(|l| {
            let mut w = l.split(' ');
            (w.next().expect("a directory"), w.collect())
        })
        .filter(|(dir, _)| dir.contains("/turbo-"))
        .collect();
    assert_eq!(
        turbo.iter().map(|(d, _)| *d).collect::<Vec<_>>(),
        wanted.iter().map(|(d, _)| *d).collect::<Vec<_>>(),
        "the script's h3-turbo processes"
    );
    let lab = Lab::load(&repo_root()).expect("lab/ parses");
    let ours = arms(&steps(&lab, "h3-graphs"));
    assert_eq!(
        ours.iter().map(|(n, _, _)| n.as_str()).collect::<Vec<_>>(),
        wanted.iter().map(|(_, a)| *a).collect::<Vec<_>>(),
        "the tier's arms, in the script's order"
    );
    for ((dir, words), (arm, gate, command)) in turbo.iter().zip(&ours) {
        // `env [K=V…] timeout 2400 <bin> <args…>`: the runner's per-arm cap
        // replaces the script's `timeout 2400` (§14.3), the rest is mapped.
        let t = words
            .iter()
            .position(|w| *w == "timeout")
            .expect("the script's timeout");
        assert_eq!(words[t + 1], "2400", "{arm}");
        let expected: Vec<String> = words[..t]
            .iter()
            .chain(&words[t + 2..])
            .map(|w| map_word(w, dir, arm))
            .collect();
        assert_eq!(command, &expected, "{arm}: the pod script's command");
        // The script ran h3-turbo only when its trees were there (TURBO_OK):
        // h3-base with the 4step-vsa MXFP8 DiT and the LoRA, the
        // fasth3-4step-vsa weight cell's trees.
        assert_eq!(gate, "fasth3-4step-vsa", "{arm}");
    }
}

#[test]
fn h3_graphs_compares_every_arm_with_the_default_for_byte_identity() {
    let lab = Lab::load(&repo_root()).expect("lab/ parses");
    let compares: Vec<(String, String, Vec<String>)> = Step::flatten(&steps(&lab, "h3-graphs"))
        .into_iter()
        .filter_map(|s| match s {
            Step::Compare {
                baseline,
                candidate,
                args,
            } => Some((baseline.clone(), candidate.clone(), args.clone())),
            _ => None,
        })
        .collect();
    let off = vec!["--off-identity".to_owned()];
    assert_eq!(
        compares,
        [
            (
                "h3-graphs-turbo-lazy".to_owned(),
                "h3-graphs-turbo-sync".to_owned(),
                off.clone()
            ),
            (
                "h3-graphs-turbo-lazy".to_owned(),
                "h3-graphs-turbo-graphs".to_owned(),
                off
            ),
        ]
    );
}

// ------------------------------------------------------------------ sol-bench

/// One cell the pod script hands its `cell`: name, `verify-weights.sh` cell,
/// the script's estimate (s, replaced by the runner's deadline) and command.
struct SolCell {
    name: String,
    gate: String,
    estimate_s: u64,
    command: Vec<String>,
}

/// `fixtures/experiments/sol-bench-script.txt`, per set in the script's order.
fn sol_sets() -> Vec<(String, Vec<SolCell>)> {
    let text = fixture("experiments/sol-bench-script.txt");
    let mut sets: Vec<(String, Vec<SolCell>)> = Vec::new();
    for line in text.lines() {
        if let Some(head) = line.strip_prefix("== ") {
            let w: Vec<&str> = head.split(' ').collect();
            assert_eq!(
                w.len(),
                4,
                "`== <set> <cell> <weight cell> <estimate>`: {line}"
            );
            let cell = SolCell {
                name: w[1].to_owned(),
                gate: w[2].to_owned(),
                estimate_s: w[3].parse().expect("an estimate in seconds"),
                command: Vec::new(),
            };
            match sets.last_mut() {
                Some((s, cells)) if s == w[0] => cells.push(cell),
                _ => sets.push((w[0].to_owned(), vec![cell])),
            }
        } else {
            let (_, cells) = sets.last_mut().expect("a `==` line first");
            cells
                .last_mut()
                .expect("a cell")
                .command
                .push(line.to_owned());
        }
    }
    sets
}

/// A sol-bench tier's runner, as on its pod: a `volume+r2` tier's `{weights}`
/// is the merged root (no key: nothing is fetched here).
fn sol_config(lab: &Lab, tier: &str) -> RunConfig {
    let mut c = config(tier);
    if lab.tier(tier).expect("tier").weights == TierWeights::VolumeR2 {
        let sources = TreeSources::load(&repo_root()).expect("volumes.sh, weights-r2.tsv");
        c.r2 = Some(R2Setup::new(sources, &c.scratch, None));
    }
    c
}

fn sol_steps(lab: &Lab, tier: &str) -> Vec<Step> {
    let cfg = sol_config(lab, tier);
    let (_, cells) = lab.resolve(tier, None).expect("tier");
    Lab::steps(&cells, &cfg.bin, &cfg.paths()).expect("render")
}

#[test]
fn sol_bench_tiers_run_the_pod_scripts_commands() {
    let lab = Lab::load(&repo_root()).expect("lab/ parses");
    let sets = sol_sets();
    // The script's 22 sets and 57 cells (sol-bench-pod.sh at the time of ADR-0019 §23).
    assert_eq!(sets.len(), 22);
    assert_eq!(sets.iter().map(|(_, c)| c.len()).sum::<usize>(), 57);
    let mut unset_arms = Vec::new();
    for (set, cells) in &sets {
        let tier = format!("sol-bench-{set}");
        let t = lab.tier(&tier).unwrap_or_else(|| panic!("{tier}: a tier"));
        assert_eq!(
            t.cells,
            std::slice::from_ref(&tier),
            "{tier}: one cell, the set's"
        );
        // The script's $W is the volume's /workspace/weights; on a volume+r2
        // tier it is the merged root, as the script's FV_R2_TREES run makes it
        // (FV_WORK=/fvscratch, ADR-0019 §21.8). Nothing else is mapped.
        let root = match t.weights {
            TierWeights::Volume => "/workspace/weights",
            TierWeights::VolumeR2 => "/fvscratch/weights",
        };
        let ours = arms(&sol_steps(&lab, &tier));
        assert_eq!(
            ours.iter().map(|(n, _, _)| n.as_str()).collect::<Vec<_>>(),
            cells.iter().map(|c| c.name.as_str()).collect::<Vec<_>>(),
            "{tier}: the set's cells, named as its run directories, in its order"
        );
        for ((arm, gate, command), c) in ours.iter().zip(cells) {
            assert_eq!(gate, &c.gate, "{tier} {arm}: the script's weight cell");
            let want: Vec<String> = c
                .command
                .iter()
                .map(|w| w.replace("/workspace/weights", root))
                .collect();
            assert_eq!(command, &want, "{tier} {arm}: the pod script's command");
            assert!(c.estimate_s > 0, "{tier} {arm}");
            if command.get(1).map(String::as_str) == Some("-u") {
                unset_arms.push(format!("{set}/{arm}"));
            }
        }
    }
    // Addition 18's users: a2's two `env -u FASTVIDEO_WAN_SOL_CACHE` arms.
    assert_eq!(unset_arms, ["a2/wan5b-base", "a2/wan14-720p-base-s15"]);
}

#[test]
fn sol_bench_dry_runs_are_their_goldens() {
    let lab = Lab::load(&repo_root()).expect("lab/ parses");
    for (set, _) in sol_sets() {
        let tier = format!("sol-bench-{set}");
        let ours = dry_run_lines(&sol_config(&lab, &tier), &sol_steps(&lab, &tier));
        let name = format!("{tier}-dry-run.txt");
        // `FV_LAB_WRITE=<dir>`: write the golden there instead of comparing
        // (as tests/gold_tier.rs); copy it over the committed one after a review.
        if let Some(dir) = std::env::var_os("FV_LAB_WRITE") {
            let p = PathBuf::from(dir).join("fixtures").join(&name);
            std::fs::create_dir_all(p.parent().expect("a parent")).expect("mkdir");
            std::fs::write(&p, ours.join("\n") + "\n").expect("write");
            continue;
        }
        assert_eq!(
            ours,
            fixture(&name).lines().collect::<Vec<_>>(),
            "fixtures/{name} is fv-lab's own dry run of the tier (ADR-0019 §23); this run prints:\n{}",
            ours.join("\n")
        );
    }
}

/// The owner's rules as tier settings (ADR-0019 §23, docs/perf/sol-bench.md):
/// the RTX PRO 6000 only, a 65-min cap for both backstops, the 3300 s cell
/// cap, no TAE fetch; the volume for a1 / a3, whose trees are all there, and
/// `volume+r2` for every other set, with its disk and host-RAM floor.
#[test]
fn sol_bench_tiers_keep_the_owners_rules() {
    let lab = Lab::load(&repo_root()).expect("lab/ parses");
    let sources = TreeSources::load(&repo_root()).expect("volumes.sh, weights-r2.tsv");
    let eu: BTreeSet<String> = parse_eu_trees(
        &std::fs::read_to_string(repo_root().join("scripts/gpu/volumes.sh")).expect("volumes.sh"),
    )
    .into_iter()
    .collect();
    let mut disks = Vec::new();
    for (set, _) in sol_sets() {
        let tier = format!("sol-bench-{set}");
        let t = lab.tier(&tier).expect("tier");
        assert_eq!(
            t.gpu, "NVIDIA RTX PRO 6000 Blackwell Server Edition",
            "{tier}"
        );
        assert_eq!(
            t.wall_cap_s, 3900,
            "{tier}: 65 min, both backstops (rule 17)"
        );
        assert_eq!(t.cell_timeout_s, 3300, "{tier}: the script's budget");
        assert!(!t.tae, "{tier}: no TAE fetch (rule 22)");
        assert!(t.extra_cells.is_empty(), "{tier}");
        let (_, cells) = lab.resolve(&tier, None).expect("tier");
        let arms: Vec<(&str, &[String])> = cells
            .iter()
            .flat_map(|c| &c.arms)
            .map(|a| (a.name.as_str(), a.trees.as_slice()))
            .collect();
        let on_volume = arms
            .iter()
            .all(|(_, trees)| trees.iter().all(|x| eu.contains(x)));
        if on_volume {
            assert_eq!(
                t.weights,
                TierWeights::Volume,
                "{tier}: every tree on the volume"
            );
            assert_eq!(t.container_disk_gb, 120, "{tier}");
            assert_eq!(t.min_ram_gb, None, "{tier}");
        } else {
            assert_eq!(
                t.weights,
                TierWeights::VolumeR2,
                "{tier}: R2 trees (rule 22, 2026-10-28)"
            );
            let plan = WeightsPlan::new(&sources, &arms).expect("its R2 trees");
            let peak_gb = plan.peak_bytes.div_ceil(1_000_000_000);
            assert!(
                u64::from(t.container_disk_gb) >= peak_gb + DISK_RESERVE_GB,
                "{tier}: {} GB < {peak_gb} + {DISK_RESERVE_GB}",
                t.container_disk_gb
            );
            // The next 10 GB up, no more.
            assert!(
                u64::from(t.container_disk_gb) < peak_gb + DISK_RESERVE_GB + 10,
                "{tier}: {} GB",
                t.container_disk_gb
            );
            assert_eq!(
                t.min_ram_gb,
                Some(188),
                "{tier}: the 2026-10-06/07 pods' RAM"
            );
        }
        disks.push(format!("{set}:{}", t.container_disk_gb));
    }
    assert_eq!(
        disks.join(" "),
        "a1:120 a3:120 a2:210 b1:150 b2:250 b2r:250 b3:250 b4:250 b5:250 b6:250 c1:250 c2:250 \
         c3:250 c4:250 c6:250 d1:250 d2:250 e1:250 f1:250 f2:250 g1:250 g2:250"
    );
}

/// The run plan (ADR-0019 §23): `lab/tiers.toml`'s `# Set <s>: one pod.` or
/// `# Set <s>: pods: --cells "<arms>" | --cells "<arms>".` above each tier.
/// Each set's groups name its arms, each arm once, in the tier's order.
#[test]
fn sol_bench_run_plan_covers_every_arm_once() {
    let lab = Lab::load(&repo_root()).expect("lab/ parses");
    let toml = std::fs::read_to_string(repo_root().join("lab/tiers.toml")).expect("tiers.toml");
    let mut pods = 0;
    for (set, cells) in sol_sets() {
        let head = format!("# Set {set}: ");
        let line = toml
            .lines()
            .find(|l| l.starts_with(&head))
            .unwrap_or_else(|| panic!("{set}: no `{head}` line in lab/tiers.toml"));
        let rest = &line[head.len()..];
        let arms: Vec<String> = cells.iter().map(|c| c.name.clone()).collect();
        let groups: Vec<Vec<String>> = if rest.starts_with("one pod.") {
            vec![arms.clone()]
        } else {
            let list = rest
                .strip_prefix("pods: ")
                .and_then(|r| r.split_once('.'))
                .map(|(g, _)| g)
                .unwrap_or_else(|| panic!("{set}: `{line}`"));
            list.split(" | ")
                .map(|g| {
                    g.strip_prefix("--cells \"")
                        .and_then(|g| g.strip_suffix('"'))
                        .unwrap_or_else(|| panic!("{set}: `{g}`"))
                        .split(' ')
                        .map(str::to_owned)
                        .collect()
                })
                .collect()
        };
        let mut seen: Vec<String> = Vec::new();
        for g in &groups {
            assert!(!g.is_empty(), "{set}: an empty group");
            // In the tier's order (--cells keeps it).
            let idx: Vec<usize> = g
                .iter()
                .map(|a| {
                    arms.iter()
                        .position(|x| x == a)
                        .unwrap_or_else(|| panic!("{set}: `{a}` is no arm of sol-bench-{set}"))
                })
                .collect();
            assert!(
                idx.windows(2).all(|w| w[0] < w[1]),
                "{set}: {g:?} out of order"
            );
            seen.extend(g.iter().cloned());
        }
        let mut sorted = seen.clone();
        sorted.sort();
        sorted.dedup();
        assert_eq!(sorted.len(), seen.len(), "{set}: an arm in two pods");
        assert_eq!(seen.len(), arms.len(), "{set}: every arm in one pod");
        assert!(lab.tier(&format!("sol-bench-{set}")).is_some());
        pods += groups.len();
    }
    assert_eq!(pods, 35, "the whole benchmark through fv-lab: 35 pods");
}
