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

mod common;

use std::path::{Path, PathBuf};

use fastvideo_lab::plan::{Lab, Step};
use fastvideo_lab::run::{dry_run_lines, RunConfig};

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
    assert_eq!(experiments, ["h3-graphs"]);
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
