//! The family manifests (`families/*.toml`, layer 2, plan A2) against the
//! hand-written CUDA catalog: the proof that the files carry every row.
//! CPU only, no GPU, no weights on disk.
//!
//! Regenerate the files from the catalog (never hand-type them):
//! `FV_FAMILIES_WRITE=<dir> cargo test -p fastvideo-engine-service --test
//! families regenerate_family_files -- --ignored` and copy
//! `<dir>/families/*.toml` to `families/`.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use fastvideo_engine_service::cuda::caps::{catalog, catalog_from_manifests, WeightLayout};
use fastvideo_engine_service::cuda::families::{catalog_family_manifests, family_id};
use fastvideo_engine_service::tier_alias;
use fastvideo_manifest::family::{load_dir, FamilyManifest, FamilySet, Tier as ManifestTier};
use fastvideo_protocol::{Family, Tier};

fn families_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../families")
}

fn committed() -> FamilySet {
    load_dir(&families_dir()).unwrap_or_else(|e| panic!("families/: {e}"))
}

/// A layout unlike the one the files were generated under: the proof that
/// nothing in them depends on where the weights are.
fn synthetic_layout() -> WeightLayout {
    WeightLayout::new("/synthetic/weights").with_tae_dir("/synthetic/tae")
}

#[test]
fn every_family_file_validates() {
    let set = committed();
    let ids: Vec<&str> = set.families().iter().map(|f| f.id().as_str()).collect();
    assert_eq!(ids, ["h3", "ltx2", "wan"]);
    for f in set.families() {
        f.validate().unwrap_or_else(|e| panic!("{}: {e}", f.id()));
        assert!(!f.variants.is_empty(), "{}", f.id());
    }
}

/// `catalog_from_manifests(layout, families/) == catalog(layout)` row for
/// row, under a synthetic layout and the default one.
#[test]
fn manifest_projection_equals_the_catalog_row_for_row() {
    let set = committed();
    for layout in [synthetic_layout(), WeightLayout::default()] {
        let want = catalog(&layout);
        let got = catalog_from_manifests(&layout, set.families()).unwrap_or_else(|e| panic!("{e}"));
        let by_id = |v: &[fastvideo_engine_service::cuda::caps::CudaModel]| {
            let mut v = v.to_vec();
            v.sort_by(|a, b| a.id.as_str().cmp(b.id.as_str()));
            v
        };
        let (want, got) = (by_id(&want), by_id(&got));
        assert_eq!(
            want.iter().map(|m| m.id.as_str()).collect::<Vec<_>>(),
            got.iter().map(|m| m.id.as_str()).collect::<Vec<_>>(),
            "variant ids differ from the catalog ids"
        );
        for (w, g) in want.iter().zip(&got) {
            assert_eq!(g, w, "{} under {}", w.id, layout.root.display());
        }
        assert_eq!(got, want);
    }
}

/// The committed files are exactly what the generator writes from the
/// catalog (parsed equality: caps, trees, GPU needs and the engine table).
#[test]
fn committed_files_equal_the_generated_ones() {
    let set = committed();
    let generated = catalog_family_manifests(&WeightLayout::default()).unwrap();
    let gen_set = FamilySet::new(generated).unwrap();
    for (c, g) in set.families().iter().zip(gen_set.families()) {
        assert_eq!(c.family, g.family);
        assert_eq!(
            c.variants.iter().map(|v| v.id.as_str()).collect::<Vec<_>>(),
            g.variants.iter().map(|v| v.id.as_str()).collect::<Vec<_>>(),
            "{}: variant order",
            c.id()
        );
        for (cv, gv) in c.variants.iter().zip(&g.variants) {
            assert_eq!(
                cv,
                gv,
                "{}/{}: regenerate families/ (see tests/families.rs)",
                c.id(),
                cv.id
            );
        }
    }
    assert_eq!(set.families(), gen_set.families());
    // A layout with a separate TAE directory generates the same files.
    let other = catalog_family_manifests(&synthetic_layout()).unwrap();
    assert_eq!(FamilySet::new(other).unwrap().families(), set.families());
}

/// Every variant id in the files has a catalog row and vice versa, and the
/// served names agree.
#[test]
fn variant_ids_and_catalog_ids_are_the_same_set() {
    let set = committed();
    let cat = catalog(&WeightLayout::default());
    let in_files: BTreeSet<&str> = set.variants().map(|(_, v)| v.id.as_str()).collect();
    let in_catalog: BTreeSet<&str> = cat.iter().map(|m| m.id.as_str()).collect();
    assert_eq!(in_files, in_catalog);
    for m in &cat {
        let (f, v) = set.find(m.id.as_str()).unwrap();
        assert_eq!(Some(f.id().as_str()), family_id(m.family()), "{}", m.id);
        let mut served: Vec<&str> = m.served_names.iter().map(String::as_str).collect();
        served.retain(|n| *n != m.id.as_str());
        assert_eq!(
            v.served_names
                .iter()
                .map(String::as_str)
                .collect::<Vec<_>>(),
            served,
            "{}",
            m.id
        );
        for n in &m.served_names {
            assert_eq!(
                set.find(n).map(|(_, v)| v.id.as_str()),
                Some(m.id.as_str()),
                "{n}"
            );
        }
    }
}

/// `families/` covers every `(Family, Tier)` that has a tier alias.
#[test]
fn families_cover_every_tier_alias() {
    let set = committed();
    let tiers = [Tier::Max, Tier::Turbo, Tier::Draft];
    for family in [
        Family::H3,
        Family::Ltx2,
        Family::Wan,
        Family::MmAudio,
        Family::Loopback,
    ] {
        for tier in tiers {
            let Some(alias) = tier_alias(family, tier) else {
                assert!(
                    family_id(family).is_none(),
                    "{family:?} has no alias but a family id"
                );
                continue;
            };
            let fid = family_id(family).unwrap();
            let want = match tier {
                Tier::Draft => ManifestTier::Draft,
                Tier::Turbo => ManifestTier::Turbo,
                Tier::Max => ManifestTier::Max,
            };
            let f = set
                .get(fid)
                .unwrap_or_else(|| panic!("{alias}: no families/{fid}.toml"));
            assert!(
                f.variants.iter().any(|v| v.tier == Some(want)),
                "{alias}: no variant of families/{fid}.toml carries tier {want}"
            );
        }
    }
}

/// Each family's variant count, the summary the plan's execution log records.
#[test]
fn variant_counts() {
    let set = committed();
    let counts: Vec<(String, usize)> = set
        .families()
        .iter()
        .map(|f| (f.id().to_string(), f.variants.len()))
        .collect();
    eprintln!("variants per family: {counts:?}");
    let total: usize = counts.iter().map(|(_, n)| n).sum();
    assert_eq!(total, catalog(&WeightLayout::default()).len());
}

/// Writes `families/*.toml` from the catalog under `$FV_FAMILIES_WRITE`.
#[test]
#[ignore = "generator: set FV_FAMILIES_WRITE=<dir>"]
fn regenerate_family_files() {
    let dir = std::env::var_os("FV_FAMILIES_WRITE")
        .map(PathBuf::from)
        .expect("FV_FAMILIES_WRITE=<dir>");
    let out = dir.join("families");
    std::fs::create_dir_all(&out).expect("mkdir");
    for fm in catalog_family_manifests(&WeightLayout::default()).unwrap() {
        let text = render(&fm);
        let p = out.join(format!("{}.toml", fm.id()));
        std::fs::write(&p, &text).unwrap_or_else(|e| panic!("write {}: {e}", p.display()));
        // What was written parses back to what was generated.
        assert_eq!(FamilyManifest::parse(&text).unwrap(), fm);
        eprintln!("wrote {} ({} variants)", p.display(), fm.variants.len());
    }
}

fn render(fm: &FamilyManifest) -> String {
    let mut text = format!(
        "# Family manifest (layer 2, docs/adr/0008-family-manifest-layer-2.md): {}.\n\
         # GENERATED from crates/fastvideo-engine-service/src/cuda/caps.rs `catalog()` by\n\
         # `tests/families.rs` (`regenerate_family_files`); do not hand-edit until plan A3\n\
         # makes this file the source. Paths: `${{weights}}/<tree>/...` and `${{tae}}/<file>`\n\
         # are substituted with the engine's weight layout at load (families/README.md).\n\n",
        fm.family.display
    );
    text.push_str(&fm.to_toml_string().expect("TOML"));
    text
}
