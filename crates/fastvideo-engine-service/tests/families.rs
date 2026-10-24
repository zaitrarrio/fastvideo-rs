//! The family manifests (`families/*.toml`, layer 2) as the source of the
//! engine catalog (plan A3): the committed files are what the engine
//! embeds, they project onto models under any weight layout, and they carry
//! what the recipes read. CPU only, no GPU, no weights on disk. The byte
//! identity of the served API over the files is
//! `crates/fastvideo-serve/tests/capabilities_golden.rs`.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use fastvideo_engine_service::cuda::caps::{catalog, catalog_from_manifests, WeightLayout};
use fastvideo_engine_service::cuda::families::{family_id, EMBEDDED_FAMILIES};
use fastvideo_engine_service::cuda::CudaRecipe;
use fastvideo_engine_service::tier_alias;
use fastvideo_manifest::family::{load_dir, FamilySet, Tier as ManifestTier};
use fastvideo_protocol::{Family, Task, Tier};

fn families_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../families")
}

fn committed() -> FamilySet {
    load_dir(&families_dir()).unwrap_or_else(|e| panic!("families/: {e}"))
}

/// A layout unlike the default: the proof that nothing in the files
/// depends on where the weights are.
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

/// The engine embeds exactly the committed files (a stale build would
/// serve another catalog than the tree says).
#[test]
fn the_embedded_files_are_the_committed_ones() {
    let dir = families_dir();
    for (name, text) in EMBEDDED_FAMILIES {
        let p = dir.join(Path::new(name).file_name().unwrap());
        let on_disk =
            std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("{}: {e}", p.display()));
        assert_eq!(on_disk, text, "{name} differs from the embedded copy");
    }
    assert_eq!(
        FamilySet::parse_all(EMBEDDED_FAMILIES).unwrap().families(),
        committed().families()
    );
}

/// The files project onto a catalog under the default and a synthetic
/// layout; every path of every recipe is under that layout (none leaks
/// from the machine the files were authored on), and the two catalogs
/// differ only in those paths.
#[test]
fn the_files_project_under_any_layout() {
    let set = committed();
    let default = catalog_from_manifests(&WeightLayout::default(), set.families())
        .unwrap_or_else(|e| panic!("{e}"));
    let synthetic = catalog_from_manifests(&synthetic_layout(), set.families())
        .unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(default.len(), synthetic.len());
    assert_eq!(default.len(), set.variants().count());
    for (d, s) in default.iter().zip(&synthetic) {
        assert_eq!(d.id, s.id);
        let text = toml::to_string(&s.recipe).unwrap();
        for line in text.lines().filter(|l| l.contains('/')) {
            assert!(
                line.contains("/synthetic/") || !line.contains("= \"/"),
                "{}: a path outside the layout: {line}",
                s.id
            );
        }
        // The same model but for the paths.
        let (mut dj, mut sj) = (
            serde_json::to_value(d).unwrap(),
            serde_json::to_value(s).unwrap(),
        );
        strip_paths(&mut dj);
        strip_paths(&mut sj);
        assert_eq!(dj, sj, "{}", d.id);
    }
    assert_eq!(catalog(&WeightLayout::default()).unwrap(), default);
}

fn strip_paths(v: &mut serde_json::Value) {
    match v {
        serde_json::Value::String(s) if s.starts_with('/') => *s = String::new(),
        serde_json::Value::Array(a) => a.iter_mut().for_each(strip_paths),
        serde_json::Value::Object(o) => o.values_mut().for_each(strip_paths),
        _ => {}
    }
}

/// Every variant id and served name resolves to its variant, every family
/// id names an engine family, and the served names of the catalog rows are
/// the files'.
#[test]
fn names_resolve_to_their_variants() {
    let set = committed();
    let cat = catalog(&WeightLayout::default()).unwrap();
    let in_files: BTreeSet<&str> = set.variants().map(|(_, v)| v.id.as_str()).collect();
    let in_catalog: BTreeSet<&str> = cat.iter().map(|m| m.id.as_str()).collect();
    assert_eq!(in_files, in_catalog);
    for m in &cat {
        let (f, v) = set.find(m.id.as_str()).unwrap();
        assert_eq!(Some(f.id().as_str()), family_id(m.family()), "{}", m.id);
        assert_eq!(m.variant, v.id);
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
        assert_eq!(m.public_names, v.public_names, "{}", m.id);
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

/// `trees` names what the recipe reads, including what it reads without
/// naming a path (the FastH3 Preview adapters beside `h3-base`, the Trim
/// NVFP4 file beside the Trim tree, the Turbo LoRAs, the LongLive-Plug
/// adapters, the TAEHV decoder of a `tae_dir`, the LongLive checkpoint):
/// the rule the generator applied before A3, now checked on the files.
#[test]
fn trees_name_what_the_recipes_read() {
    let set = committed();
    let trees = |id: &str| -> Vec<String> { set.find(id).unwrap().1.trees.clone() };
    let want: BTreeMap<&str, &[&str]> = [
        ("sol-h3", &["FastH3-4-step-Preview-v1-LoRA", "h3-base"][..]),
        (
            "fasth3-4step-vsa-480p-taeh3",
            &[
                "FastH3-4-step-Preview-v1-LoRA",
                "auxiliary/tae/taeh3.safetensors",
                "h3-base",
            ][..],
        ),
        (
            "fasth3-trim-nvfp4",
            &["fasth3-trim-8step", "fasth3-trim-8step-nvfp4", "h3-base"][..],
        ),
        ("h3-larry-turbo", &["h3-base", "h3-turbo-loras/larry"][..]),
        (
            "h3-lightx2v-turbo-v01",
            &["h3-base", "h3-turbo-loras/lightx2v-v0.1"][..],
        ),
        (
            "vdn-h3-8step",
            &["h3-base", "vdn-h3-dmd-250-turbo", "vdn-h3-stage-b-2000"][..],
        ),
        ("h3-ref2v-turbo", &["h3-base", "h3-ref2va"][..]),
        ("ltx25-a2v-guided", &["ltx25", "ltx25-dev"][..]),
        ("ltx25-ref2v", &["ltx25", "ltx25-ic-lora-ingredients"][..]),
        (
            "wan14b-plug-4step",
            &[
                "longlive-plug/wan21-t2v-14b-cfg",
                "longlive-plug/wan21-t2v-14b-few-step",
                "wan21-t2v-14b",
            ][..],
        ),
        (
            "fastwan22-ti2v-5b-taehv",
            &["auxiliary/tae/taew2_2.safetensors", "fastwan22-ti2v-5b"][..],
        ),
        ("fastwan22-ti2v-5b", &["fastwan22-ti2v-5b"][..]),
        (
            "sfwan21-1.3b",
            &["auxiliary/tae/taew2_1.safetensors", "sfwan21-1.3b"][..],
        ),
        (
            "longlive-1.3b",
            &[
                "auxiliary/tae/taew2_1.safetensors",
                "longlive-1.3b-safetensors",
                "sfwan21-1.3b",
            ][..],
        ),
    ]
    .into_iter()
    .collect();
    for (id, t) in want {
        assert_eq!(trees(id), t, "{id}");
    }
    // Every Turbo LoRA variant names its adapter tree, every TAEHV Wan
    // decoder its file, every LongLive-Plug recipe its adapters.
    let cat = catalog(&WeightLayout::default()).unwrap();
    for m in &cat {
        let t = trees(m.id.as_str());
        match &m.recipe {
            CudaRecipe::H3(r) => {
                if let Some(l) = fastvideo_engine_service::cuda::caps::turbo_lora_of(&r.recipe) {
                    let want = format!(
                        "{}/{}",
                        fastvideo_models::h3::turbo_lora::TURBO_LORA_TREE,
                        l.subdir()
                    );
                    assert!(t.contains(&want), "{}: {t:?} lacks {want}", m.id);
                }
                if fastvideo_models::h3::config::is_trim_nvfp4_recipe(&r.recipe) {
                    assert!(
                        t.contains(&"fasth3-trim-8step-nvfp4".to_owned()),
                        "{}",
                        m.id
                    );
                }
            }
            CudaRecipe::Wan(w) => wan_trees(m.id.as_str(), w, &t),
            CudaRecipe::SfWan(s) => {
                wan_trees(m.id.as_str(), &s.wan, &t);
                if s.longlive.is_some() {
                    assert!(
                        t.contains(&"longlive-1.3b-safetensors".to_owned()),
                        "{}",
                        m.id
                    );
                }
            }
            CudaRecipe::Ltx2(_) => {}
        }
    }
}

fn wan_trees(id: &str, w: &fastvideo_engine_service::cuda::WanRecipe, trees: &[String]) {
    if let Some(p) = &w.plug {
        let plug = fastvideo_models::plug::PlugRecipe::named(p).unwrap();
        for a in &plug.adapters {
            let want = format!("longlive-plug/{}", a.dir);
            assert!(trees.contains(&want), "{id}: {trees:?} lacks {want}");
        }
    }
    if w.decoder == fastvideo_engine_service::cuda::WanDecoder::Taehv && w.tae_dir.is_some() {
        let file = if w.preset.contains("2_2") {
            "taew2_2"
        } else {
            "taew2_1"
        };
        let want = format!("auxiliary/tae/{file}.safetensors");
        assert!(trees.contains(&want), "{id}: {trees:?} lacks {want}");
    }
}

/// The caps the files carry are what the rows serve: the task list, the
/// audio, the default length, the H3 1080P tier on the two tiers that
/// offer it, the container-only fps of the Wan clip models.
#[test]
fn serving_caps_project_onto_model_caps() {
    let cat = catalog(&WeightLayout::default()).unwrap();
    let get = |id: &str| cat.iter().find(|m| m.id.as_str() == id).unwrap().caps();
    let turbo = get("fasth3-4step-vsa");
    assert_eq!(
        turbo.tasks,
        [Task::T2V, Task::I2V, Task::Keyframes]
            .into_iter()
            .collect()
    );
    assert_eq!(turbo.frames.default, 124);
    assert_eq!(turbo.audio.as_ref().map(|a| a.native_rate), Some(32_000));
    assert_eq!(
        turbo.canvas.hd,
        Some(fastvideo_protocol::HdTier::h3_1080p())
    );
    assert_eq!(turbo.canvas.short_edges, [768, 480, 1080]);
    for id in ["sol-h3", "fasth3-4step-vsa"] {
        assert!(get(id).canvas.hd.is_some(), "{id}");
    }
    for m in &cat {
        let c = m.caps();
        if !matches!(m.id.as_str(), "sol-h3" | "fasth3-4step-vsa") {
            assert!(c.canvas.hd.is_none(), "{}", m.id);
        }
        assert!(c.frames.contains(c.frames.default), "{}", m.id);
        assert_eq!(
            c.fps.container_only,
            m.family() == Family::Wan
                && c.stream
                    .as_ref()
                    .is_some_and(|s| matches!(s, fastvideo_protocol::StreamCaps::Clip { .. })),
            "{}",
            m.id
        );
        assert_eq!(c.id, m.id);
        assert_eq!(c.recipe.as_deref(), Some(m.recipe_name.as_str()));
    }
    let ref2v = get("ltx25-ref2v");
    assert!(ref2v.knobs.reference_strength && ref2v.tasks == [Task::Ref2V].into_iter().collect());
    assert!(get("ltx25-distill-sol").canvas.pad_and_crop);
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
    assert_eq!(total, 32);
    assert_eq!(total, catalog(&WeightLayout::default()).unwrap().len());
}
