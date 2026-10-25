//! fv-lab and the scripts refuse the same settings (ADR-0019 §11): the keys
//! `fv-lab pod up --extra-env` refuses are `runpod-http.sh`'s
//! `FV_LAB_NOT_READ`, and the restored modes are `fv-lab-retired.sh`'s
//! `FV_LAB_RESTORED`, each a tier of `lab/tiers.toml`.

mod common;

use std::path::PathBuf;

use fastvideo_lab::knobs::{not_read, RESTORED_MODES};
use fastvideo_lab::plan::Lab;

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

#[test]
fn extra_env_refusals_are_runpod_http_shs() {
    let p = repo_root().join("scripts/gpu/runpod-http.sh");
    let t = std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("{}: {e}", p.display()));
    let start =
        t.find("FV_LAB_NOT_READ=\"").expect("FV_LAB_NOT_READ=\"…\"") + "FV_LAB_NOT_READ=\"".len();
    let end = start + t[start..].find('"').expect("closing quote");
    let script: Vec<&str> = t[start..end].split_whitespace().collect();
    assert_eq!(not_read().collect::<Vec<_>>(), script);
}

#[test]
fn restored_modes_are_fv_lab_retired_shs_and_tiers() {
    let lab = Lab::load(&repo_root()).expect("lab/ parses");
    let script = common::restored();
    assert_eq!(script.len(), RESTORED_MODES.len());
    for (m, r) in RESTORED_MODES.iter().zip(&script) {
        assert_eq!(
            (m.tier, m.family, format!("{}={}", m.key, m.value)),
            (r.tier.as_str(), r.family.as_str(), r.mode.clone())
        );
        assert!(lab.tier(m.tier).is_some(), "{}: a tier", m.tier);
        assert!(lab.tier(m.family).is_some(), "{}: a tier", m.family);
    }
}
