//! fv-lab's own tiers (`FV_LAB_OWN` of `scripts/gpu/fv-lab-retired.sh`,
//! docs/dev/fv-lab.md "fv-lab's own tiers"): tiers written for fv-lab after
//! `runpod-matrix.sh` went (ADR-0019 §22.20), like `gold` but with
//! hand-written cells. No script ever ran them, so there is no script dry run
//! to equal and no experiment pod script to be cut from. Instead:
//!
//! - each is a tier of `lab/tiers.toml`, in no other list of
//!   `fv-lab-retired.sh`, with no `runpod-matrix.sh` branch (the script is
//!   gone);
//! - its pod body golden (`fixtures/runpod-http-<tier>.json`) is
//!   `runpod-http.sh plan`'s, written by `fv-lab-goldens.sh` as for `gold`,
//!   and tests/payload_golden.rs holds fv-lab's pod to it;
//! - its dry run is fv-lab's own golden (`fixtures/<tier>-dry-run.txt`),
//!   frozen from the start like the experiment tiers' (§12, §14): a change to
//!   the tier's cells changes it in the same PR (`FV_LAB_WRITE=<dir>` writes
//!   it, as tests/gold_tier.rs does for `gold`).

mod common;

use std::path::{Path, PathBuf};

use fastvideo_lab::plan::{Lab, Step};
use fastvideo_lab::run::{dry_run_lines, RunConfig};
use fastvideo_lab::weights::{R2Setup, TreeSources};

const TAG: &str = "1234567-10241200";

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn fixture_path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("fixtures")
        .join(name)
}

fn config(tier: &str) -> RunConfig {
    let mut c = RunConfig::new(tier, Path::new("/opt/fastvideo-rs"));
    c.work = "/workspace".into();
    c.scratch = "/fvscratch".into();
    c.tag = Some(TAG.into());
    c.tae_dir = Some("/workspace/weights/auxiliary/tae".into());
    let sources = TreeSources::load(&repo_root()).expect("volumes.sh, weights-r2.tsv");
    c.r2 = Some(R2Setup::new(sources, &c.scratch, None));
    c
}

#[test]
fn own_tiers_are_tiers_in_no_other_list_with_a_pod_body_golden() {
    let lab = Lab::load(&repo_root()).expect("lab/ parses");
    let experiments = common::experiments();
    let retired = std::fs::read_to_string(repo_root().join("scripts/gpu/fv-lab-retired.sh"))
        .expect("fv-lab-retired.sh");
    let retired: Vec<&str> = retired
        .lines()
        .find_map(|l| l.strip_prefix("FV_LAB_RETIRED=\""))
        .and_then(|l| l.strip_suffix('"'))
        .expect("FV_LAB_RETIRED")
        .split_whitespace()
        .collect();
    assert!(
        !repo_root().join("scripts/gpu/runpod-matrix.sh").exists(),
        "runpod-matrix.sh is gone (ADR-0019 §22.20)"
    );
    for t in common::own_tiers() {
        assert!(lab.tier(&t).is_some(), "{t}: a tier of lab/tiers.toml");
        assert_ne!(t, "gold", "gold has its own tests (tests/gold_tier.rs)");
        assert!(!experiments.contains(&t), "{t}: not an experiment tier");
        assert!(!retired.contains(&t.as_str()), "{t}: not a retired family");
        let body = fixture_path(&format!("runpod-http-{t}.json"));
        let body = std::fs::read_to_string(&body)
            .unwrap_or_else(|e| panic!("{}: {e} (fv-lab-goldens.sh)", body.display()));
        assert!(!body.trim().is_empty(), "{t}: the pod body golden");
    }
}

/// `fv-lab run <tier> --dry-run` is the tier's own golden.
#[test]
fn own_tier_dry_runs_equal_their_goldens() {
    let lab = Lab::load(&repo_root()).expect("lab/ parses");
    for t in common::own_tiers() {
        let cfg = config(&t);
        let (_, cells) = lab.resolve(&t, None).expect("tier");
        let steps = Lab::steps(&cells, &cfg.bin, &cfg.paths()).expect("render");
        assert!(steps.iter().all(|s| matches!(s, Step::Arm { .. })), "{t}");
        let ours = dry_run_lines(&cfg, &steps);
        let name = format!("{t}-dry-run.txt");
        // `FV_LAB_WRITE=<dir>`: write the golden there instead of comparing;
        // copy it over the committed one after a review.
        if let Some(dir) = std::env::var_os("FV_LAB_WRITE") {
            let p = PathBuf::from(dir).join("fixtures").join(&name);
            std::fs::create_dir_all(p.parent().expect("a parent")).expect("mkdir");
            std::fs::write(&p, ours.join("\n") + "\n").expect("write");
            continue;
        }
        let p = fixture_path(&name);
        let fixture = std::fs::read_to_string(&p).unwrap_or_else(|e| {
            panic!(
                "{}: {e} (FV_LAB_WRITE=<dir> cargo test -p fastvideo-lab --test own_tiers)",
                p.display()
            )
        });
        assert_eq!(
            ours,
            fixture.lines().collect::<Vec<_>>(),
            "{t}: regenerate with `FV_LAB_WRITE=crates/fastvideo-lab cargo test -p fastvideo-lab --test own_tiers`"
        );
    }
}
