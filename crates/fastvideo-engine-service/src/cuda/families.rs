//! The engine's projection of the family manifests (layer 2, ADR-0008; plan
//! A3): `families/<family>.toml` -> [`CudaModel`] rows. The files are the
//! source; there is no hand-written catalog.
//!
//! **Where the files come from.** The three family files are embedded at
//! build time (`include_str!`), so `fv-serve`, `fv-gpucheck` and every test
//! carry the manifests they were built with and the serve image needs no
//! extra file. [`FAMILIES_DIR_ENV`] (`FV_FAMILIES_DIR=<dir>`) overrides them
//! at run time with a directory of `*.toml` (a lab run with an unreleased
//! variant, a config experiment); the override is read on every
//! [`family_set`] call, never cached, so a process sees what its environment
//! says.
//!
//! **The substitution rule.** A variant's `engine` table is the serde form
//! of its [`CudaRecipe`] with every path under the weights layout written
//! relative to it: a path under `layout.tae_dir` is `${tae}/<rest>` (`${tae}`
//! alone for the directory), any other path under `layout.root` is
//! `${weights}/<rest>`. [`recipe_from_engine`] substitutes the running
//! process's layout, so one file serves every machine.
//!
//! **Caps.** A variant's `caps` (layer 1) and `serving` (layer 2) tables
//! become the model's [`ModelCaps`] ([`model_caps_of`]); only two things
//! stay derived from the recipe at [`CudaModel::caps`] time, because they
//! change with the deployment: whether the H3 1080P tier is offered
//! (`hd_1080p`, turned off by `h3_1080p = false` or the memory gate) and the
//! streaming shape (`StreamCaps`: causal for an SF-Wan recipe, with its
//! KV window and the LongLive re-cache, else a clip whose bounds are the
//! frame grid over the default fps).

use std::collections::BTreeSet;
use std::path::Path;

use fastvideo_manifest::family::{
    load_dir, FamilyError, FamilyManifest, FamilySet, Tier as ManifestTier, VariantManifest,
    TAE_VAR, WEIGHTS_VAR,
};
use fastvideo_protocol::{
    AudioCaps, CanvasCaps, Family, FpsCaps, FrameGrid, HdTier, KnobCaps, ModelCaps, ModelId,
    RefLimits, Task, Tier,
};

use super::caps::{CudaModel, CudaRecipe, WeightLayout};

/// `FV_FAMILIES_DIR=<dir>`: read the family files from this directory
/// instead of the embedded ones.
pub const FAMILIES_DIR_ENV: &str = "FV_FAMILIES_DIR";

/// The family files this binary was built with (`families/*.toml`).
pub const EMBEDDED_FAMILIES: [(&str, &str); 3] = [
    (
        "families/h3.toml",
        include_str!("../../../../families/h3.toml"),
    ),
    (
        "families/ltx2.toml",
        include_str!("../../../../families/ltx2.toml"),
    ),
    (
        "families/wan.toml",
        include_str!("../../../../families/wan.toml"),
    ),
];

/// Why the catalog could not be built from the family files.
#[derive(thiserror::Error, Debug)]
pub enum CatalogError {
    /// A family file (or the set) does not validate.
    #[error("family manifests: {0}")]
    Manifest(#[from] FamilyError),
    /// One variant cannot be projected onto an engine model.
    #[error("{family}/{variant}: {msg}")]
    Variant {
        family: String,
        variant: String,
        msg: String,
    },
}

impl CatalogError {
    fn variant(f: &FamilyManifest, v: &VariantManifest, msg: impl Into<String>) -> Self {
        CatalogError::Variant {
            family: f.family.id.to_string(),
            variant: v.id.clone(),
            msg: msg.into(),
        }
    }
}

/// The family id of a catalog family (the `[family] id` and layer 1's
/// `runtime.pipeline`).
pub fn family_id(f: Family) -> Option<&'static str> {
    match f {
        Family::H3 => Some("h3"),
        Family::Ltx2 => Some("ltx2"),
        Family::Wan => Some("wan"),
        Family::MmAudio | Family::Loopback => None,
    }
}

/// The catalog family of a family id.
pub fn family_of_id(id: &str) -> Option<Family> {
    match id {
        "h3" => Some(Family::H3),
        "ltx2" => Some(Family::Ltx2),
        "wan" => Some(Family::Wan),
        _ => None,
    }
}

fn tier_from_manifest(t: ManifestTier) -> Tier {
    match t {
        ManifestTier::Draft => Tier::Draft,
        ManifestTier::Turbo => Tier::Turbo,
        ManifestTier::Max => Tier::Max,
    }
}

/// The family files: `FV_FAMILIES_DIR` when set, else the embedded ones.
pub fn family_set() -> Result<FamilySet, CatalogError> {
    match std::env::var_os(FAMILIES_DIR_ENV).filter(|d| !d.is_empty()) {
        Some(dir) => Ok(load_dir(Path::new(&dir))?),
        None => Ok(FamilySet::parse_all(EMBEDDED_FAMILIES)?),
    }
}

/// Every servable model: the variants of every family file, in file order
/// (families by id). The one catalog the engine has.
pub fn catalog(layout: &WeightLayout) -> Result<Vec<CudaModel>, CatalogError> {
    catalog_from_manifests(layout, family_set()?.families())
}

/// Applies `f` to every string of a TOML value tree.
fn map_strings(v: toml::Value, f: &dyn Fn(&str) -> String) -> toml::Value {
    match v {
        toml::Value::String(s) => toml::Value::String(f(&s)),
        toml::Value::Array(items) => {
            toml::Value::Array(items.into_iter().map(|i| map_strings(i, f)).collect())
        }
        toml::Value::Table(t) => {
            toml::Value::Table(t.into_iter().map(|(k, v)| (k, map_strings(v, f))).collect())
        }
        other => other,
    }
}

/// `${tae}/…` and `${weights}/…` under the running process's layout; any
/// other string unchanged.
fn resolve(s: &str, layout: &WeightLayout) -> String {
    for (var, base) in [(TAE_VAR, &layout.tae_dir), (WEIGHTS_VAR, &layout.root)] {
        if s == var {
            return base.to_string_lossy().into_owned();
        }
        if let Some(rest) = s.strip_prefix(var).and_then(|r| r.strip_prefix('/')) {
            return base.join(rest).to_string_lossy().into_owned();
        }
    }
    s.to_owned()
}

/// A variant's `engine` table as a [`CudaRecipe`] under `layout`.
pub fn recipe_from_engine(
    engine: &toml::Table,
    layout: &WeightLayout,
) -> Result<CudaRecipe, String> {
    map_strings(toml::Value::Table(engine.clone()), &|s| resolve(s, layout))
        .try_into()
        .map_err(|e| format!("engine table: {e}"))
}

/// A variant's `caps` and `serving` tables as the engine's `ModelCaps`,
/// with the variant id as the model id (the row's id, served names, tier,
/// residency and recipe name are filled by [`CudaModel::caps`]). Every
/// layer-1 table is required of a served variant.
pub fn model_caps_of(family: Family, v: &VariantManifest) -> Result<ModelCaps, String> {
    let frames = v.caps.frames.as_ref().ok_or("caps.frames missing")?;
    let canvas = v.caps.canvas.as_ref().ok_or("caps.canvas missing")?;
    let fps = v.caps.fps.as_ref().ok_or("caps.fps missing")?;
    let refs = v.caps.refs.as_ref().ok_or("caps.refs missing")?;
    let knobs = v.caps.knobs.as_ref().ok_or("caps.knobs missing")?;
    let s = &v.serving;
    let mut tasks = BTreeSet::new();
    for t in &s.tasks {
        let task: Task = serde_json::from_value(serde_json::Value::String(t.clone()))
            .map_err(|_| format!("serving.tasks: unknown task `{t}`"))?;
        tasks.insert(task);
    }
    Ok(ModelCaps {
        id: ModelId::new(&v.id),
        family,
        served_names: vec![v.id.clone()],
        tasks,
        audio: s.audio.as_ref().map(|a| AudioCaps {
            native_rate: a.native_rate,
            channels: a.channels,
            via_sidecar: a.via_sidecar,
        }),
        fps: FpsCaps {
            allowed: fps.allowed.clone(),
            default: fps.default,
            container_only: s.fps_container_only,
        },
        frames: FrameGrid::new(
            frames.step,
            frames.offset,
            frames.min,
            frames.max,
            s.frames_default,
        ),
        canvas: CanvasCaps {
            multiple: canvas.multiple,
            max_area: canvas.max_area,
            aspect: (canvas.aspect[0] as f32, canvas.aspect[1] as f32),
            short_edges: canvas.short_edges.clone(),
            pad_and_crop: s.pad_and_crop,
            hd: s.hd.as_ref().map(|t| HdTier {
                short_edge: t.short_edge,
                max_area: t.max_area,
                max_frames: t.max_frames,
                experimental_max_frames: t.experimental_max_frames,
            }),
        },
        refs: RefLimits {
            images: refs.images,
            videos: refs.videos,
            audio: refs.audios,
            total: refs.total,
        },
        // Set per model by `CudaModel::caps` (the recipe decides).
        stream: None,
        knobs: KnobCaps {
            seed: knobs.seed,
            negative: knobs.negative,
            steps: knobs.steps,
            guidance: knobs.guidance,
            guidance_2: knobs.guidance_2,
            flow_shift: knobs.flow_shift,
            reference_strength: s.reference_strength,
        },
        resident: v.resident_default,
        tier: None,
        recipe: None,
    })
}

/// The engine catalog of a set of family manifests: one [`CudaModel`] per
/// variant, in file order. Fails when a variant's `engine` table is not a
/// recipe, names another family than its file, or lacks a caps table.
pub fn catalog_from_manifests(
    layout: &WeightLayout,
    families: &[FamilyManifest],
) -> Result<Vec<CudaModel>, CatalogError> {
    let mut out = Vec::new();
    for f in families {
        let family = family_of_id(f.family.id.as_str()).ok_or_else(|| CatalogError::Variant {
            family: f.family.id.to_string(),
            variant: String::new(),
            msg: "not a family this engine serves (h3, ltx2, wan)".to_owned(),
        })?;
        for v in &f.variants {
            let recipe = recipe_from_engine(&v.engine, layout)
                .map_err(|e| CatalogError::variant(f, v, e))?;
            if recipe.family() != family {
                return Err(CatalogError::variant(
                    f,
                    v,
                    format!(
                        "the engine table is a {} recipe",
                        family_id(recipe.family()).unwrap_or("?")
                    ),
                ));
            }
            let base_caps = model_caps_of(family, v).map_err(|e| CatalogError::variant(f, v, e))?;
            let mut served_names = vec![v.id.clone()];
            for n in &v.served_names {
                if !served_names.contains(n) {
                    served_names.push(n.clone());
                }
            }
            out.push(CudaModel {
                id: ModelId::new(&v.id),
                variant: v.id.clone(),
                served_names,
                public_names: v.public_names.clone(),
                tier: v.tier.map(tier_from_manifest),
                resident: v.resident_default,
                recipe,
                recipe_name: v.recipe_name.clone(),
                gpu: v.gpu.clone(),
                base_caps,
            });
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    /// The inverse of [`resolve`]: a path under the layout as a `${tae}/…`
    /// or `${weights}/…` string (how the files were first generated).
    fn relativize(s: &str, layout: &WeightLayout) -> String {
        let p = Path::new(s);
        for (var, base) in [(TAE_VAR, &layout.tae_dir), (WEIGHTS_VAR, &layout.root)] {
            if let Ok(rest) = p.strip_prefix(base) {
                let rest = rest.to_string_lossy();
                return if rest.is_empty() {
                    var.to_owned()
                } else {
                    format!("{var}/{rest}")
                };
            }
        }
        s.to_owned()
    }

    #[test]
    fn substitution_is_a_bijection_on_layout_paths() {
        let l = WeightLayout::new("/w").with_tae_dir("/t");
        for (abs, rel) in [
            ("/w/h3-base", "${weights}/h3-base"),
            (
                "/w/ltx25-dev/transformer_full",
                "${weights}/ltx25-dev/transformer_full",
            ),
            ("/t/taeh3.safetensors", "${tae}/taeh3.safetensors"),
            ("/t", "${tae}"),
            ("/w", "${weights}"),
            ("auto", "auto"),
            ("/elsewhere/x", "/elsewhere/x"),
            ("/workspace/weights/x", "/workspace/weights/x"),
        ] {
            assert_eq!(relativize(abs, &l), rel, "{abs}");
            assert_eq!(resolve(rel, &l), abs, "{rel}");
        }
        // The default layout nests the TAE directory under the root: TAE first.
        let d = WeightLayout::default();
        assert_eq!(
            relativize("/workspace/weights/auxiliary/tae/taew2_1.safetensors", &d),
            "${tae}/taew2_1.safetensors"
        );
        assert_eq!(
            PathBuf::from(resolve("${tae}/taew2_1.safetensors", &d)),
            d.tae_dir.join("taew2_1.safetensors")
        );
    }

    /// Every embedded variant's engine table deserialises under two layouts,
    /// and serialising the recipe back gives the table (no path escapes the
    /// substitution).
    #[test]
    fn every_embedded_recipe_round_trips_through_its_engine_table() {
        let set = FamilySet::parse_all(EMBEDDED_FAMILIES).unwrap();
        for layout in [
            WeightLayout::new("/run/weights"),
            WeightLayout::new("/gen/w").with_tae_dir("/gen/t"),
        ] {
            for (f, v) in set.variants() {
                let recipe = recipe_from_engine(&v.engine, &layout)
                    .unwrap_or_else(|e| panic!("{}/{}: {e}", f.id(), v.id));
                assert_eq!(
                    family_id(recipe.family()),
                    Some(f.id().as_str()),
                    "{}",
                    v.id
                );
                let back = toml::Value::try_from(&recipe).unwrap();
                let back = map_strings(back, &|s| relativize(s, &layout));
                assert_eq!(back, toml::Value::Table(v.engine.clone()), "{}", v.id);
            }
        }
    }

    #[test]
    fn the_embedded_files_are_the_catalog() {
        let cat = catalog(&WeightLayout::default()).unwrap();
        assert_eq!(cat.len(), 32);
        let ids: Vec<&str> = cat.iter().map(|m| m.id.as_str()).collect();
        assert!(ids.contains(&"fasth3-4step-vsa"));
        assert!(ids.contains(&"longlive-1.3b"));
        assert!(ids.contains(&"h3-lightx2v-turbo-v01"));
        let m = cat
            .iter()
            .find(|m| m.id.as_str() == "fasth3-4step-vsa")
            .unwrap();
        assert_eq!(m.public_names, ["MiniMax-H3", "MiniMax-H3-Turbo"]);
        assert_eq!(m.variant, "fasth3-4step-vsa");
        assert_eq!(m.gpu.needs, ["fp8"]);
        assert!(m.requirements().fp8 && !m.requirements().nvfp4);
    }

    #[test]
    fn families_dir_override_is_read_per_call() {
        // Not a directory: the override fails loudly instead of falling back.
        let bad = std::env::temp_dir().join("fv-families-missing-dir");
        let _ = std::fs::remove_dir_all(&bad);
        let set = FamilySet::parse_all(EMBEDDED_FAMILIES).unwrap();
        let dir = std::env::temp_dir().join("fv-families-override-test");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let text = set.get("wan").unwrap().to_toml_string().unwrap();
        std::fs::write(dir.join("wan.toml"), text).unwrap();
        let got = load_dir(&dir).unwrap();
        assert_eq!(got.len(), 1);
        assert_eq!(
            catalog_from_manifests(&WeightLayout::default(), got.families())
                .unwrap()
                .len(),
            set.get("wan").unwrap().variants.len()
        );
        assert!(load_dir(&bad).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
