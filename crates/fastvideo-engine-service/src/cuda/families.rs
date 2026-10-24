//! The engine's two projections of the family manifest (layer 2, ADR-0008;
//! plan A2): `families/<family>.toml` -> [`CudaModel`] rows
//! ([`catalog_from_manifests`]) and, until A3 deletes the hand-written
//! [`catalog`], [`catalog`] rows -> family files ([`family_manifests`], the
//! generator the committed files were written with).
//!
//! **The substitution rule.** A variant's `engine` table is the serde form
//! of its [`CudaRecipe`] with every path under the weights layout written
//! relative to it: a path under `layout.tae_dir` becomes
//! `${tae}/<rest>` (`${tae}` alone for the directory), any other path under
//! `layout.root` becomes `${weights}/<rest>`. [`recipe_from_engine`]
//! applies the inverse with the running process's layout, so one file serves
//! every machine. `tae_dir` is checked first because it lives under the root
//! by default (`<root>/auxiliary/tae`).
//!
//! **Trees.** `trees` lists what the variant reads, as
//! `scripts/gpu/weights-manifest.tsv` names it: the first component of every
//! `${weights}/…` reference, `auxiliary/tae/<file>` for every `${tae}/<file>`
//! reference, plus the trees a recipe reads without naming a path (the
//! FastH3 Preview adapters beside `h3-base`, the Trim NVFP4 file beside the
//! Trim tree, the Turbo LoRAs, the LongLive-Plug adapters, the TAEHV decoder
//! of a `tae_dir`), derived from the recipe in [`implicit_trees`].

use std::collections::BTreeSet;
use std::path::Path;

use fastvideo_manifest::family::{
    AudioCaps as ManifestAudioCaps, FamilyHeader, FamilyManifest, GpuNeeds, HdTier as ManifestHdTier,
    ServingCaps, Tier as ManifestTier, VariantManifest, FAMILY_SCHEMA, TAE_VAR, WEIGHTS_VAR,
};
use fastvideo_manifest::{CanvasCaps, Caps, FpsCaps, FrameCaps, KnobCaps, RefCaps};
use fastvideo_models::h3::config as h3cfg;
use fastvideo_models::h3::turbo_lora::TURBO_LORA_TREE;
use fastvideo_protocol::{Family, ModelCaps, ModelId, Tier};

use super::caps::{catalog, turbo_lora_of, CudaModel, CudaRecipe, WanDecoder, WeightLayout};

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

/// The `[family]` header of each served family. `licence` names the weights
/// licence of the family's vendor trees (`weights-manifest.tsv`); per-tree
/// detail is each tree's layer-1 manifest.
fn family_header(f: Family) -> Option<FamilyHeader> {
    let (id, display, licence, port_doc) = match f {
        Family::H3 => (
            "h3",
            "MiniMax-H3 / FastH3",
            "MiniMax-H3 Community License",
            "docs/ports/h3.md",
        ),
        Family::Ltx2 => (
            "ltx2",
            "LTX-2 / LTX-2.3 / LTX-2.5",
            "LTX-2 Community License",
            "docs/ports/ltx2.md",
        ),
        Family::Wan => (
            "wan",
            "Wan 2.1 / Wan 2.2 / FastWan",
            "Apache-2.0",
            "docs/ports/wan.md",
        ),
        Family::MmAudio | Family::Loopback => return None,
    };
    Some(FamilyHeader {
        id: id.parse().ok()?,
        display: display.to_owned(),
        licence: licence.to_owned(),
        port_doc: port_doc.to_owned(),
    })
}

fn tier_to_manifest(t: Tier) -> ManifestTier {
    match t {
        Tier::Draft => ManifestTier::Draft,
        Tier::Turbo => ManifestTier::Turbo,
        Tier::Max => ManifestTier::Max,
    }
}

fn tier_from_manifest(t: ManifestTier) -> Tier {
    match t {
        ManifestTier::Draft => Tier::Draft,
        ManifestTier::Turbo => Tier::Turbo,
        ManifestTier::Max => Tier::Max,
    }
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

/// A path under the layout as a `${tae}/…` or `${weights}/…` string; any
/// other string unchanged.
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

/// The inverse of [`relativize`] for the running process's layout.
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

/// A recipe as a variant's `engine` table: its serde form, paths relative to
/// `layout` (the substitution rule in the module doc).
pub fn engine_table(recipe: &CudaRecipe, layout: &WeightLayout) -> Result<toml::Table, String> {
    let v = toml::Value::try_from(recipe).map_err(|e| format!("recipe to TOML: {e}"))?;
    match map_strings(v, &|s| relativize(s, layout)) {
        toml::Value::Table(t) => Ok(t),
        other => Err(format!(
            "recipe serialised to a {}, not a table",
            other.type_str()
        )),
    }
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

/// Rebuilds the engine catalog from family manifests: one [`CudaModel`] per
/// variant, in file order. Fails when an `engine` table is not a recipe or
/// names another family than its file.
pub fn catalog_from_manifests(
    layout: &WeightLayout,
    families: &[FamilyManifest],
) -> Result<Vec<CudaModel>, String> {
    let mut out = Vec::new();
    for f in families {
        for v in &f.variants {
            let recipe = recipe_from_engine(&v.engine, layout)
                .map_err(|e| format!("{}/{}: {e}", f.family.id, v.id))?;
            let fid = family_id(recipe.family());
            if fid != Some(f.family.id.as_str()) {
                return Err(format!(
                    "{}/{}: the engine table is a {} recipe",
                    f.family.id,
                    v.id,
                    fid.unwrap_or("?")
                ));
            }
            let mut served_names = vec![v.id.clone()];
            for n in &v.served_names {
                if !served_names.contains(n) {
                    served_names.push(n.clone());
                }
            }
            out.push(CudaModel {
                id: ModelId::new(&v.id),
                served_names,
                tier: v.tier.map(tier_from_manifest),
                resident: v.resident_default,
                recipe,
                recipe_name: v.recipe_name.clone(),
            });
        }
    }
    Ok(out)
}

/// The trees a recipe reads without naming a path (module doc "Trees").
pub fn implicit_trees(recipe: &CudaRecipe) -> Vec<String> {
    let mut out = Vec::new();
    match recipe {
        CudaRecipe::H3(r) => {
            // The FastH3 Preview v1 adapters, searched beside `h3-base`
            // (`fastvideo_models::h3::lora::SolH3AdapterSpec`).
            if matches!(r.recipe.as_str(), "4step-vsa" | "4step-dense" | "sol-h3") {
                out.push("FastH3-4-step-Preview-v1-LoRA".to_owned());
            }
            // The Trim's NVFP4 file, read beside the Trim tree
            // (`fastvideo_cudarc::h3::nvfp4::TREE`).
            if h3cfg::is_trim_nvfp4_recipe(&r.recipe) {
                out.push("fasth3-trim-8step-nvfp4".to_owned());
            }
            if let Some(t) = turbo_lora_of(&r.recipe) {
                out.push(format!("{TURBO_LORA_TREE}/{}", t.subdir()));
            }
        }
        CudaRecipe::Ltx2(_) => {}
        CudaRecipe::Wan(w) => wan_implicit_trees(w, &mut out),
        CudaRecipe::SfWan(s) => wan_implicit_trees(&s.wan, &mut out),
    }
    out
}

fn wan_implicit_trees(w: &super::caps::WanRecipe, out: &mut Vec<String>) {
    if let Some(p) = &w.plug {
        if let Some(plug) = fastvideo_models::plug::PlugRecipe::named(p) {
            for a in &plug.adapters {
                out.push(format!("longlive-plug/{}", a.dir));
            }
        }
    }
    // TAEHV decodes the Wan 2.1 latent space with `taew2_1`, the 2.2 one
    // with `taew2_2` (`fastvideo_cudarc::wan::pipeline`).
    if w.decoder == WanDecoder::Taehv && w.tae_dir.is_some() {
        let file = if w.preset.contains("2_2") {
            "taew2_2"
        } else {
            "taew2_1"
        };
        out.push(format!("auxiliary/tae/{file}.safetensors"));
    }
}

/// `trees` of a variant: the module doc's rule over its engine table and
/// its recipe.
pub fn trees_of(engine: &toml::Table, recipe: &CudaRecipe) -> Vec<String> {
    let mut trees: BTreeSet<String> = implicit_trees(recipe).into_iter().collect();
    collect_tree_refs(&toml::Value::Table(engine.clone()), &mut trees);
    trees.into_iter().collect()
}

fn collect_tree_refs(v: &toml::Value, out: &mut BTreeSet<String>) {
    match v {
        toml::Value::String(s) => {
            if let Some(rel) = s
                .strip_prefix(WEIGHTS_VAR)
                .and_then(|r| r.strip_prefix('/'))
            {
                if let Some(first) = rel.split('/').next().filter(|c| !c.is_empty()) {
                    out.insert(first.to_owned());
                }
            } else if let Some(file) = s.strip_prefix(TAE_VAR).and_then(|r| r.strip_prefix('/')) {
                if !file.is_empty() {
                    out.insert(format!("auxiliary/tae/{file}"));
                }
            }
        }
        toml::Value::Array(items) => items.iter().for_each(|i| collect_tree_refs(i, out)),
        toml::Value::Table(t) => t.values().for_each(|i| collect_tree_refs(i, out)),
        _ => {}
    }
}

/// The smallest card class each variant is served on, in GiB
/// (`control/src/presets.ts` `min_vram_gb`: the H3 and LTX presets name
/// 80 GB cards, VDN-H3 96 GB, the Wan 5B tiers 32 GB, the 1.3B models 24 GB;
/// the 14B tier peaks at 51 GB, docs/serve/research-longlive.md §12).
fn min_vram_gib(m: &CudaModel) -> u32 {
    match &m.recipe {
        CudaRecipe::H3(r) if r.vdn_weights.is_some() => 96,
        CudaRecipe::H3(_) | CudaRecipe::Ltx2(_) => 80,
        CudaRecipe::Wan(w) if w.plug.is_some() => 80,
        CudaRecipe::Wan(w) if w.preset.contains("2_2") => 32,
        CudaRecipe::Wan(_) | CudaRecipe::SfWan(_) => 24,
    }
}

fn gpu_needs(m: &CudaModel) -> GpuNeeds {
    let req = m.requirements();
    let ((major, minor), _) = req.min_sm();
    let mut needs = Vec::new();
    if req.fp8 {
        needs.push("fp8".to_owned());
    }
    if req.nvfp4 {
        needs.push("nvfp4".to_owned());
    }
    GpuNeeds {
        min_vram_gib: min_vram_gib(m),
        min_sm: major * 10 + minor,
        needs,
    }
}

/// The layer-1 `[caps]` of a served model: the subset of `ModelCaps` the
/// registry carries (the H3 1080P tier's own budget and the container-only
/// fps flag stay with the engine).
pub fn caps_of(c: &ModelCaps) -> Caps {
    Caps {
        frames: Some(FrameCaps {
            step: c.frames.step,
            offset: c.frames.offset,
            min: c.frames.min,
            max: c.frames.max,
        }),
        canvas: Some(CanvasCaps {
            multiple: c.canvas.multiple,
            max_area: c.canvas.max_area,
            aspect: [f64::from(c.canvas.aspect.0), f64::from(c.canvas.aspect.1)],
            short_edges: c.canvas.short_edges.clone(),
        }),
        fps: Some(FpsCaps {
            allowed: c.fps.allowed.clone(),
            default: c.fps.default,
        }),
        refs: Some(RefCaps {
            images: c.refs.images,
            videos: c.refs.videos,
            audios: c.refs.audio,
            total: c.refs.total,
        }),
        knobs: Some(KnobCaps {
            seed: c.knobs.seed,
            negative: c.knobs.negative,
            steps: c.knobs.steps,
            guidance: c.knobs.guidance,
            guidance_2: c.knobs.guidance_2,
            flow_shift: c.knobs.flow_shift,
        }),
    }
}

/// The layer-2 `[serving]` table of a served model: what `ModelCaps`
/// advertises beyond layer 1's `Caps`.
pub fn serving_of(c: &ModelCaps) -> ServingCaps {
    ServingCaps {
        tasks: c
            .tasks
            .iter()
            .filter_map(|t| serde_json::to_value(t).ok())
            .filter_map(|v| v.as_str().map(str::to_owned))
            .collect(),
        frames_default: c.frames.default,
        audio: c.audio.as_ref().map(|a| ManifestAudioCaps {
            native_rate: a.native_rate,
            channels: a.channels,
            via_sidecar: a.via_sidecar,
        }),
        fps_container_only: c.fps.container_only,
        pad_and_crop: c.canvas.pad_and_crop,
        hd: c.canvas.hd.map(|t| ManifestHdTier {
            short_edge: t.short_edge,
            max_area: t.max_area,
            max_frames: t.max_frames,
            experimental_max_frames: t.experimental_max_frames,
        }),
        reference_strength: c.knobs.reference_strength,
    }
}

/// The public API names of each catalog row: what the serve configs'
/// `[aliases]` tables mapped onto the model by hand (the MiniMax `model`
/// ids). A tier's name is also on its task companion, so a server serving
/// only the companion (runpod-h3-ref2v.toml) still answers to it.
fn public_names_of(id: &str) -> Vec<String> {
    let names: &[&str] = match id {
        "sol-h3" | "h3-ref2v-max" => &["MiniMax-H3-Max"],
        "fasth3-4step-vsa" => &["MiniMax-H3", "MiniMax-H3-Turbo"],
        "h3-ref2v-turbo" => &["MiniMax-H3-Turbo"],
        "fasth3-4step-vsa-taeh3" => &["MiniMax-H3-Turbo-Preview"],
        "fasth3-v2-nvfp4-taeh3" => &["MiniMax-H3-V2-NVFP4-Preview"],
        "fasth3-4step-vsa-nvfp4-taeh3" => &["MiniMax-H3-Turbo-NVFP4-Preview"],
        "fasth3-trim-8step" => &["MiniMax-H3-Trim"],
        "fasth3-trim-nvfp4" => &["MiniMax-H3-Trim-NVFP4"],
        "fasth3-trim-nvfp4-fast" => &["MiniMax-H3-Trim-NVFP4-Fast"],
        "fasth3-v2-nvfp4" => &["MiniMax-H3-V2-NVFP4"],
        "fasth3-4step-vsa-nvfp4" => &["MiniMax-H3-Turbo-NVFP4"],
        "h3-larry-turbo" => &["MiniMax-H3-Turbo-Lora"],
        "h3-lightx2v-turbo" => &["MiniMax-H3-Turbo-Lightx2v"],
        "vdn-h3" => &["VDN-H3"],
        "vdn-h3-8step" => &["VDN-H3-8step"],
        _ => &[],
    };
    names.iter().map(|n| (*n).to_owned()).collect()
}

/// One catalog row as a `[[variants]]` table.
pub fn variant_manifest(m: &CudaModel, layout: &WeightLayout) -> Result<VariantManifest, String> {
    let engine = engine_table(&m.recipe, layout)?;
    let trees = trees_of(&engine, &m.recipe);
    let caps = m.caps();
    Ok(VariantManifest {
        id: m.id.as_str().to_owned(),
        registry_name: None,
        tier: m.tier.map(tier_to_manifest),
        served_names: m
            .served_names
            .iter()
            .filter(|n| n.as_str() != m.id.as_str())
            .cloned()
            .collect(),
        public_names: public_names_of(m.id.as_str()),
        resident_default: m.resident,
        recipe_name: m.recipe_name.clone(),
        trees,
        gpu: gpu_needs(m),
        caps: caps_of(&caps),
        serving: serving_of(&caps),
        engine,
    })
}

/// The family files of a catalog, one per served family in the order h3,
/// ltx2, wan; the variants in catalog order. The generator of
/// `families/*.toml` (`tests/families.rs`, `regenerate_family_files`).
pub fn family_manifests(
    models: &[CudaModel],
    layout: &WeightLayout,
) -> Result<Vec<FamilyManifest>, String> {
    let mut out = Vec::new();
    for family in [Family::H3, Family::Ltx2, Family::Wan] {
        let Some(header) = family_header(family) else {
            continue;
        };
        let variants = models
            .iter()
            .filter(|m| m.family() == family)
            .map(|m| variant_manifest(m, layout))
            .collect::<Result<Vec<_>, _>>()?;
        if variants.is_empty() {
            continue;
        }
        let fm = FamilyManifest {
            schema: FAMILY_SCHEMA,
            family: header,
            variants,
        };
        fm.validate()
            .map_err(|e| format!("{}: generated file does not validate: {e}", fm.family.id))?;
        out.push(fm);
    }
    Ok(out)
}

/// The family files of the hand-written [`catalog`] under `layout`.
pub fn catalog_family_manifests(layout: &WeightLayout) -> Result<Vec<FamilyManifest>, String> {
    family_manifests(&catalog(layout), layout)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

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

    #[test]
    fn every_catalog_recipe_round_trips_through_its_engine_table() {
        let from = WeightLayout::new("/gen/w").with_tae_dir("/gen/t");
        let to = WeightLayout::new("/run/weights");
        let want = catalog(&to);
        for (m, w) in catalog(&from).iter().zip(&want) {
            let t = engine_table(&m.recipe, &from).unwrap();
            // The tag is the recipe kind (`h3`, `ltx2`, `wan`, `sf_wan`), not
            // the family id; the family id is `CudaRecipe::family`'s.
            assert!(t.contains_key("family"), "{}", m.id);
            let back = recipe_from_engine(&t, &to).unwrap();
            assert_eq!(back, w.recipe, "{}", m.id);
            assert_eq!(family_id(back.family()), family_id(m.family()));
            // No absolute path survives in the table.
            let text = toml::to_string(&t).unwrap();
            assert!(!text.contains("/gen/"), "{}: {text}", m.id);
        }
    }

    #[test]
    fn trees_name_what_the_recipe_reads() {
        let l = WeightLayout::default();
        let cat = catalog(&l);
        let trees = |id: &str| {
            let m = cat.iter().find(|m| m.id.as_str() == id).unwrap();
            trees_of(&engine_table(&m.recipe, &l).unwrap(), &m.recipe)
        };
        assert_eq!(
            trees("sol-h3"),
            ["FastH3-4-step-Preview-v1-LoRA", "h3-base"]
        );
        assert_eq!(
            trees("fasth3-4step-vsa-480p-taeh3"),
            [
                "FastH3-4-step-Preview-v1-LoRA",
                "auxiliary/tae/taeh3.safetensors",
                "h3-base"
            ]
        );
        assert_eq!(
            trees("fasth3-trim-nvfp4"),
            ["fasth3-trim-8step", "fasth3-trim-8step-nvfp4", "h3-base"]
        );
        assert_eq!(trees("h3-larry-turbo"), ["h3-base", "h3-turbo-loras/larry"]);
        assert_eq!(
            trees("vdn-h3-8step"),
            ["h3-base", "vdn-h3-dmd-250-turbo", "vdn-h3-stage-b-2000"]
        );
        assert_eq!(trees("ltx25-a2v-guided"), ["ltx25", "ltx25-dev"]);
        assert_eq!(trees("ltx25-ref2v"), ["ltx25", "ltx25-ic-lora-ingredients"]);
        assert_eq!(
            trees("wan14b-plug-4step"),
            [
                "longlive-plug/wan21-t2v-14b-cfg",
                "longlive-plug/wan21-t2v-14b-few-step",
                "wan21-t2v-14b"
            ]
        );
        assert_eq!(
            trees("fastwan22-ti2v-5b-taehv"),
            ["auxiliary/tae/taew2_2.safetensors", "fastwan22-ti2v-5b"]
        );
        assert_eq!(trees("fastwan22-ti2v-5b"), ["fastwan22-ti2v-5b"]);
        assert_eq!(
            trees("sfwan21-1.3b"),
            ["auxiliary/tae/taew2_1.safetensors", "sfwan21-1.3b"]
        );
    }
}
