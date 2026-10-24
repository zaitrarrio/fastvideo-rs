//! The committed gold files (`lab/gold/*.toml`, ADR-0009) against the family
//! files (`families/*.toml`): every file parses and validates, the set of
//! gold files equals the set of variant ids (both directions), each file
//! names its variant's family, tier and profile as the family file does,
//! and every path a `[[source]]` or `model.profile` names exists in the tree.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use fastvideo_lab::{load_dir, Gold, NUMERIC_FIELDS};
use fastvideo_manifest::{FamilyManifest, FamilySet};

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .expect("repository root")
}

fn gold_files() -> Vec<Gold> {
    let dir = repo_root().join("lab/gold");
    load_dir(&dir).unwrap_or_else(|e| panic!("{}: {e}", dir.display()))
}

fn families() -> FamilySet {
    let dir = repo_root().join("families");
    let mut files = Vec::new();
    for entry in std::fs::read_dir(&dir).expect("families dir") {
        let p = entry.expect("entry").path();
        if p.extension().is_some_and(|x| x == "toml") {
            let text = std::fs::read_to_string(&p).expect("read family file");
            files.push((p.display().to_string(), text));
        }
    }
    files.sort();
    FamilySet::parse_all(files.iter().map(|(l, t)| (l.as_str(), t.as_str())))
        .expect("family files valid")
}

#[test]
fn every_gold_file_parses_and_validates() {
    let golds = gold_files();
    assert!(!golds.is_empty(), "no gold files under lab/gold");
    for g in &golds {
        // A second parse of the serialised form is the same file.
        let text = g.to_toml_string().expect("serialise");
        assert_eq!(
            &Gold::parse(&text).expect("re-parse"),
            g,
            "{}",
            g.model.variant
        );
    }
}

#[test]
fn gold_files_and_family_variants_are_the_same_set() {
    let golds = gold_files();
    let fams = families();
    let gold_ids: BTreeSet<&str> = golds.iter().map(|g| g.model.variant.as_str()).collect();
    let variant_ids: BTreeSet<&str> = fams.variants().map(|(_, v)| v.id.as_str()).collect();
    let missing: Vec<&&str> = variant_ids.difference(&gold_ids).collect();
    let extra: Vec<&&str> = gold_ids.difference(&variant_ids).collect();
    assert!(
        missing.is_empty() && extra.is_empty(),
        "variants without a gold file: {missing:?}; gold files without a variant: {extra:?}"
    );
    assert_eq!(
        gold_ids.len(),
        golds.len(),
        "duplicate variant among the gold files"
    );
}

#[test]
fn each_gold_file_agrees_with_its_family_file() {
    let fams = families();
    for g in gold_files() {
        let (fam, var) = fams
            .find(&g.model.variant)
            .unwrap_or_else(|| panic!("{}: no such variant", g.model.variant));
        assert_eq!(g.model.family, *fam.id(), "{}: family", g.model.variant);
        assert_eq!(g.tier.name, var.tier, "{}: tier", g.model.variant);
        let profile = var
            .engine
            .get("profile")
            .and_then(|v| v.as_str())
            .map(str::to_owned);
        assert_eq!(g.model.profile, profile, "{}: profile", g.model.variant);
        if let Some(name) = &g.model.registry_name {
            assert_eq!(
                Some(name),
                var.registry_name.as_ref(),
                "{}: registry_name",
                g.model.variant
            );
        }
        // The tier's GPU class is at least the variant's floor.
        assert!(
            g.tier.vram_gib >= var.gpu.min_vram_gib && g.tier.sm >= var.gpu.min_sm,
            "{}: tier GPU below the variant's gpu needs",
            g.model.variant
        );
    }
}

#[test]
fn every_cited_path_exists() {
    let root = repo_root();
    for g in gold_files() {
        if let Some(p) = &g.model.profile {
            let f = root.join("profiles").join(format!("{p}.toml"));
            assert!(
                f.is_file(),
                "{}: profile {} missing",
                g.model.variant,
                f.display()
            );
        }
        for s in &g.source {
            assert!(
                root.join(&s.path).exists(),
                "{}: source path {} does not exist",
                g.model.variant,
                s.path
            );
        }
        if let Some(r) = &g.last_validated.run {
            assert!(
                root.join(r).is_file(),
                "{}: run record {r} missing",
                g.model.variant
            );
        }
    }
}

#[test]
fn family_files_parse_on_their_own() {
    // The gold tests read the family files through the pure parser; this
    // keeps the dependency direction visible when a family file breaks.
    let dir = repo_root().join("families");
    for entry in std::fs::read_dir(&dir).expect("families dir") {
        let p = entry.expect("entry").path();
        if p.extension().is_some_and(|x| x == "toml") {
            let text = std::fs::read_to_string(&p).expect("read");
            FamilyManifest::parse(&text).unwrap_or_else(|e| panic!("{}: {e}", p.display()));
        }
    }
}

/// Printed by `cargo test -p fastvideo-lab -- --nocapture coverage_table`:
/// which variants carry which numbers (the D2 hand-back table).
#[test]
fn coverage_table() {
    let golds = gold_files();
    let mut by_field: BTreeMap<&str, usize> = BTreeMap::new();
    for g in &golds {
        let measured = g.measured_fields();
        for f in &measured {
            *by_field.entry(f).or_default() += 1;
        }
        println!(
            "{:<36} tier={:<6} measured={:<2} gaps={:<2} validated={}",
            g.model.variant,
            g.tier.name.map(|t| t.as_str()).unwrap_or("-"),
            measured.len(),
            g.gap.len(),
            g.last_validated.is_validated()
        );
    }
    for f in NUMERIC_FIELDS {
        println!(
            "{f:<36} {}/{} files",
            by_field.get(f).copied().unwrap_or(0),
            golds.len()
        );
    }
}
