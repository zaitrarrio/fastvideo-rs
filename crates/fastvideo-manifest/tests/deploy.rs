//! The committed deployment data (ADR-0011): `deploy/deploy.toml` parses and
//! validates against the committed family manifests (`families/*.toml`), and
//! every worker config it names exists under `configs/serve`. The library
//! does no I/O; this test reads the files.

mod common;

use std::path::PathBuf;

use fastvideo_manifest::{DeployManifest, FamilySet, EU_VOLUME_ID};

fn repo() -> PathBuf {
    common::crate_root().join("../..")
}

fn families() -> FamilySet {
    let dir = repo().join("families");
    let mut paths: Vec<PathBuf> = std::fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("{}: {e}", dir.display()))
        .map(|e| e.expect("dir entry").path())
        .filter(|p| p.extension().is_some_and(|x| x == "toml"))
        .collect();
    paths.sort();
    let texts: Vec<(String, String)> = paths
        .iter()
        .map(|p| (p.display().to_string(), common::read(p)))
        .collect();
    FamilySet::parse_all(texts.iter().map(|(l, t)| (l.as_str(), t.as_str())))
        .unwrap_or_else(|e| panic!("families: {e}"))
}

fn committed() -> (DeployManifest, FamilySet) {
    let set = families();
    let text = common::read(&repo().join("deploy/deploy.toml"));
    let d = DeployManifest::parse_with(&text, &set)
        .unwrap_or_else(|e| panic!("deploy/deploy.toml: {e}"));
    (d, set)
}

#[test]
fn committed_deploy_file_validates_against_the_families() {
    let (d, set) = committed();
    assert_eq!(d.presets.len(), 24);
    assert_eq!(d.family_presets.len(), 4);
    assert_eq!(
        d.built_images().map(|i| i.id.as_str()).collect::<Vec<_>>(),
        ["h3-turbo", "h3-max", "ltx", "wan", "wan5b", "sfwan", "cpu"]
    );
    // Every family preset's card class meets its members' floor (checked by
    // validate_with); the NVFP4 members put the H3 families at 96 GB.
    for fp in &d.family_presets {
        let floor = d
            .family_preset_variants(&set, fp)
            .iter()
            .map(|(_, v)| v.gpu.min_vram_gib)
            .max()
            .unwrap_or(0);
        assert!(fp.min_vram_gb >= floor, "{}", fp.id);
    }
    // The only volume named is the EU one.
    let text = common::read(&repo().join("deploy/deploy.toml"));
    for line in text.lines().filter(|l| l.contains("volume_id")) {
        assert!(line.contains(EU_VOLUME_ID), "{line}");
    }
}

#[test]
fn every_named_worker_config_exists() {
    let (d, _) = committed();
    let serve = repo().join("configs/serve");
    let mut names: Vec<&str> = Vec::new();
    for i in &d.images {
        names.push(&i.config);
        names.extend(i.also.iter().map(String::as_str));
    }
    names.extend(d.presets.iter().map(|p| p.config.as_str()));
    names.extend(d.family_presets.iter().map(|p| p.config.as_str()));
    for r in &d.gcp {
        names.push(&r.config);
        names.extend(r.runpod_twin.as_deref());
    }
    for n in names {
        assert!(serve.join(n).is_file(), "configs/serve/{n} does not exist");
    }
}
