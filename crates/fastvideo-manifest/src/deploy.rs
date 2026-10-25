//! The deployment data (plan C1, ADR-0011): `deploy/deploy.toml`, how the
//! variants of the family manifests (layer 2, [`crate::family`]) are
//! deployed. Like layer 2 it is this repository's own format, versioned by
//! `schema`, not a registry contract.
//!
//! The deployment unit is a **preset**: one pool, a set of variants served
//! together by one fv-serve process, with one worker config
//! (`configs/serve/<config>`, sent at boot), queue and timeouts. A **family
//! preset** is one pool serving the models of several model presets of one
//! family (owner decision 2026-10-18). Beside them: the serve images CI
//! builds (one CUDA image and the CPU image: no image bakes a config, every
//! launch sends one; owner decision 2026-10-25),
//! the GCP VM rows, the edge dispatch family of each model family, and the
//! autoscaler's pools (an opaque table each: `fastvideo-autoscale` owns that
//! format, the way the engine owns a variant's `engine` table).
//!
//! [`DeployManifest::parse`] checks the file on its own (shape, ids, names,
//! numbers, the EU-only volume rule); [`DeployManifest::validate_with`]
//! checks it against a [`FamilySet`]: every variant a preset names exists,
//! every recipe name resolves to that variant, a family preset's card class
//! meets its variants' `gpu.min_vram_gib` floor and serves one family,
//! pre-quantized trees are nested in the preset's weight trees, the dispatch
//! map names exactly the families. What needs the rest of the tree (the
//! serve configs exist, the Dockerfile stages, the CI build list, the
//! autoscale prices) is checked by `control/gen-configs.mjs`, which also
//! generates every deployment table from this file.
//!
//! Pure: no I/O.

use std::collections::{BTreeMap, BTreeSet};

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::family::{tree_covers, FamilyManifest, FamilySet, VariantManifest};
use crate::path::check_relative_path;

/// The only deployment-file schema this crate reads and writes.
pub const DEPLOY_SCHEMA: u32 = 1;
/// The one region with a weights volume (CLAUDE.md "Model weights and network
/// volumes": EU only, owner decisions 2026-10-06 and 2026-10-13).
pub const EU_REGION: &str = "eu";
/// The EU weights volume's data centre.
pub const EU_DATA_CENTER: &str = "EU-RO-1";
/// The EU weights volume (`fv-weights-h3-ltx-500`): the only volume a
/// deployment may mount until the owner rebuilds another one.
pub const EU_VOLUME_ID: &str = "bz0ofkfx45";
/// The fake-engine config every serve image carries under `/etc/fv`: CI's
/// no-GPU smoke check boots it with `FV_WORKER_CONFIG` (the one config a
/// launch may name in the image instead of sending it).
pub const SMOKE_CONFIG: &str = "runpod-fake.toml";
/// Longest preset, image or GCP row id (fv-control's name rule: a DNS label
/// of at most 31 characters).
pub const DEPLOY_ID_MAX_LEN: usize = 31;
/// Weight trees under this prefix are the auxiliary files every boot fetches
/// (tiny autoencoders, LPIPS), never a preset's weights.
pub const AUX_TREE_PREFIX: &str = "auxiliary/";

/// Why a deployment file is not valid.
#[derive(thiserror::Error, Debug, Clone, PartialEq, Eq)]
pub enum DeployError {
    #[error("TOML syntax: {0}")]
    Syntax(String),
    #[error("unsupported schema {0}; this crate reads schema {DEPLOY_SCHEMA}")]
    UnsupportedSchema(i64),
    #[error("`{path}`: {msg}")]
    Field { path: String, msg: String },
}

impl DeployError {
    fn field(path: impl Into<String>, msg: impl Into<String>) -> Self {
        DeployError::Field {
            path: path.into(),
            msg: msg.into(),
        }
    }

    /// The dotted field path, when the error names one.
    pub fn path(&self) -> Option<&str> {
        match self {
            DeployError::Field { path, .. } => Some(path),
            DeployError::UnsupportedSchema(_) => Some("schema"),
            DeployError::Syntax(_) => None,
        }
    }
}

/// `deploy/deploy.toml`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct DeployManifest {
    /// Always 1.
    pub schema: u32,
    /// Model family id -> its edge / dispatch family (`ltx2` -> `ltx`).
    pub dispatch: BTreeMap<String, String>,
    /// The serve images: one per compute (the CUDA image, the CPU image).
    pub images: Vec<Image>,
    /// The model presets, in the dashboard's order.
    pub presets: Vec<Preset>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub family_presets: Vec<FamilyPreset>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub gcp: Vec<GcpRow>,
    #[serde(default)]
    pub autoscale: AutoscaleSection,
}

/// What an image's fv-serve is built for.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "lowercase")]
pub enum Compute {
    /// The CUDA fv-serve (`serve-cuda`).
    #[default]
    Gpu,
    /// fv-serve without CUDA, the fake engine.
    Cpu,
}

fn is_default<T: Default + PartialEq>(x: &T) -> bool {
    *x == T::default()
}

/// One `[[images]]` table: a serve image (`docker/gpucheck.Dockerfile`
/// stage `serve-<id>`, GHCR tags `<id>-<channel>`). It bakes no worker
/// config: every launch sends one (`FV_WORKER_TOML_B64`, or
/// `FV_WORKER_CONFIG` naming a file it carries), and an image started
/// without one refuses to boot (fv-boot, ADR-0014).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Image {
    pub id: String,
    /// The worker configs the image carries under `/etc/fv` (`configs/serve`
    /// file names): [`SMOKE_CONFIG`] for CI's smoke check.
    pub carries: Vec<String>,
    #[serde(default, skip_serializing_if = "is_default")]
    pub compute: Compute,
    /// A short note for the documentation tables.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub note: Option<String>,
}

impl Image {
    /// Whether the image carries this worker config under `/etc/fv`.
    pub fn carries(&self, config: &str) -> bool {
        self.carries.iter().any(|c| c == config)
    }
}

/// One model of a preset's pool.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct PoolModel {
    /// The variant id (`families/*.toml`), never a served name.
    pub variant: String,
    /// The model id the pool serves (default: the variant id).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    /// The name the pool uses for the variant when it is not the id: a
    /// served name or the family's tier alias (default: the variant id).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub recipe: Option<String>,
}

impl PoolModel {
    pub fn model_id(&self) -> &str {
        self.id.as_deref().unwrap_or(&self.variant)
    }

    pub fn recipe_name(&self) -> &str {
        self.recipe.as_deref().unwrap_or(&self.variant)
    }
}

/// One `[[presets]]` table: a model preset (one pool).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Preset {
    pub id: String,
    pub title: String,
    pub description: String,
    /// One of the default cluster's pools.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub standard: bool,
    /// The serve image ([`Image::id`]): the CUDA image.
    pub image: String,
    /// The worker config (a `configs/serve` file name), sent at boot.
    pub config: String,
    /// Derived pre-quantized trees nested in a weight tree (a faster load).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub prequant: Vec<String>,
    /// Set when the weights' licence restricts use.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub licence: Option<String>,
    /// Container disk per worker (GB) when not sized from the weights.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub container_disk_gb: Option<u32>,
    pub models: Vec<PoolModel>,
    pub max_queued: u32,
    pub job_timeout_s: u32,
    pub stale_after_s: u32,
}

/// Something a family preset does not serve, and why.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct LeftOut {
    pub what: String,
    pub why: String,
}

/// One `[[family_presets]]` table: one pool serving the models of several
/// model presets of one family (one process, `[engine] swap`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct FamilyPreset {
    pub id: String,
    /// `{members}` stands for the members, comma separated.
    pub title: String,
    pub description: String,
    pub image: String,
    pub config: String,
    /// The card class (GB of GPU memory), hand-chosen: at least the members'
    /// variants' `gpu.min_vram_gib`.
    pub min_vram_gb: u32,
    /// Model preset ids, in the presets' order.
    pub members: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub prequant: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub left_out: Vec<LeftOut>,
}

/// One `[[gcp]]` table: a row of `scripts/gcp/families.tsv`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct GcpRow {
    pub name: String,
    /// The VM's worker config (a `configs/serve` file name).
    pub config: String,
    /// The Runpod config it mirrors; its `[[models]]` give the row's weight
    /// trees and served ids.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub runpod_twin: Option<String>,
    /// A value of [`DeployManifest::dispatch`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dispatch_family: Option<String>,
    /// `scripts/gpu/verify-weights.sh` cells.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub verify_cells: Vec<String>,
    pub machine: String,
    pub fal_app: String,
    pub minimax_model: String,
    /// The served model ids of a row without a twin.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub served: Vec<String>,
    pub reactor_mode: String,
}

/// `[autoscale]`: only its pools; the controller's own settings and prices
/// stay in `configs/autoscale.toml`.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct AutoscaleSection {
    /// `[[autoscale.pools]]`, opaque to this crate (`fastvideo-autoscale`
    /// owns the format) but for `name`, `family` and the volume rule.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    #[schemars(with = "Vec<BTreeMap<String, serde_json::Value>>")]
    pub pools: Vec<toml::Table>,
}

/// A preset's GPU floor: what its variants need.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct GpuFloor {
    pub min_vram_gib: u32,
    pub min_sm: u32,
    pub needs: BTreeSet<String>,
}

impl GpuFloor {
    fn of<'a>(variants: impl IntoIterator<Item = &'a VariantManifest>) -> GpuFloor {
        let mut f = GpuFloor::default();
        for v in variants {
            f.min_vram_gib = f.min_vram_gib.max(v.gpu.min_vram_gib);
            f.min_sm = f.min_sm.max(v.gpu.min_sm);
            f.needs.extend(v.gpu.needs.iter().cloned());
        }
        f
    }
}

fn check_id(id: &str, path: &str) -> Result<(), DeployError> {
    let ok = !id.is_empty()
        && id.len() <= DEPLOY_ID_MAX_LEN
        && id.starts_with(|c: char| c.is_ascii_lowercase())
        && !id.ends_with('-')
        && id
            .chars()
            .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-');
    if ok {
        Ok(())
    } else {
        Err(DeployError::field(
            path,
            format!(
                "`{id}`: 1-{DEPLOY_ID_MAX_LEN} lower-case letters, digits and '-', starting with a letter, not ending with '-'"
            ),
        ))
    }
}

fn check_config(c: &str, path: &str) -> Result<(), DeployError> {
    let ok = c.len() > ".toml".len()
        && c.ends_with(".toml")
        && c.chars()
            .all(|x| x.is_ascii_alphanumeric() || matches!(x, '.' | '_' | '-'));
    if ok {
        Ok(())
    } else {
        Err(DeployError::field(
            path,
            format!("`{c}`: a configs/serve file name (`<name>.toml`)"),
        ))
    }
}

fn check_text(s: &str, path: &str) -> Result<(), DeployError> {
    if s.trim().is_empty() {
        Err(DeployError::field(path, "empty"))
    } else {
        Ok(())
    }
}

fn check_positive(n: u32, path: &str) -> Result<(), DeployError> {
    if n == 0 {
        Err(DeployError::field(path, "must be positive"))
    } else {
        Ok(())
    }
}

/// Every table under `v` that names a `volume_id` must be the EU volume in
/// its data centre.
fn check_volumes(v: &toml::Value, path: &str) -> Result<(), DeployError> {
    match v {
        toml::Value::Table(t) => {
            if let Some(id) = t.get("volume_id") {
                let dc = t.get("data_center").and_then(toml::Value::as_str);
                if id.as_str() != Some(EU_VOLUME_ID) || dc != Some(EU_DATA_CENTER) {
                    return Err(DeployError::field(
                        path,
                        format!(
                            "volume {id} in {}: the only weights volume is {EU_VOLUME_ID} in {EU_DATA_CENTER} (region {EU_REGION}; CLAUDE.md, EU only)",
                            dc.unwrap_or("no data_center")
                        ),
                    ));
                }
            }
            for (k, x) in t {
                check_volumes(x, &format!("{path}.{k}"))?;
            }
            Ok(())
        }
        toml::Value::Array(items) => items
            .iter()
            .enumerate()
            .try_for_each(|(i, x)| check_volumes(x, &format!("{path}[{i}]"))),
        _ => Ok(()),
    }
}

/// `prequant` entries are relative paths strictly inside one of `trees`.
fn check_prequant(prequant: &[String], trees: &[&str], path: &str) -> Result<(), DeployError> {
    for (i, q) in prequant.iter().enumerate() {
        check_relative_path(q)
            .map_err(|e| DeployError::field(path, format!("entry {i} {q:?}: {e}")))?;
        if !trees.iter().any(|t| *t != q && tree_covers(t, q)) {
            return Err(DeployError::field(
                path,
                format!(
                    "`{q}` is not nested in one of the weight trees ({})",
                    trees.join(", ")
                ),
            ));
        }
    }
    Ok(())
}

impl DeployManifest {
    /// Parse and check one deployment file on its own.
    pub fn parse(text: &str) -> Result<DeployManifest, DeployError> {
        let table: toml::Table = text
            .parse()
            .map_err(|e: toml::de::Error| DeployError::Syntax(e.to_string()))?;
        match table.get("schema") {
            Some(toml::Value::Integer(1)) => {}
            Some(toml::Value::Integer(other)) => {
                return Err(DeployError::UnsupportedSchema(*other))
            }
            Some(_) => return Err(DeployError::field("schema", "expected integer")),
            None => return Err(DeployError::field("schema", "missing")),
        }
        let d: DeployManifest =
            toml::from_str(text).map_err(|e| DeployError::Syntax(e.to_string()))?;
        d.validate()?;
        Ok(d)
    }

    /// Parse, then check against the family manifests.
    pub fn parse_with(text: &str, families: &FamilySet) -> Result<DeployManifest, DeployError> {
        let d = Self::parse(text)?;
        d.validate_with(families)?;
        Ok(d)
    }

    pub fn image(&self, id: &str) -> Option<&Image> {
        self.images.iter().find(|i| i.id == id)
    }

    pub fn preset(&self, id: &str) -> Option<&Preset> {
        self.presets.iter().find(|p| p.id == id)
    }

    pub fn family_preset(&self, id: &str) -> Option<&FamilyPreset> {
        self.family_presets.iter().find(|p| p.id == id)
    }

    /// The CUDA serve image (validation guarantees exactly one).
    pub fn gpu_image(&self) -> Option<&Image> {
        self.images.iter().find(|i| i.compute == Compute::Gpu)
    }

    /// The rules that need nothing but the file.
    pub fn validate(&self) -> Result<(), DeployError> {
        if self.schema != DEPLOY_SCHEMA {
            return Err(DeployError::UnsupportedSchema(i64::from(self.schema)));
        }
        for (k, v) in &self.dispatch {
            check_text(v, &format!("dispatch.{k}"))?;
        }
        let mut image_ids = BTreeSet::new();
        for (i, im) in self.images.iter().enumerate() {
            let p = |f: &str| format!("images[{i}].{f}");
            check_id(&im.id, &p("id"))?;
            if !image_ids.insert(im.id.as_str()) {
                return Err(DeployError::field(p("id"), format!("`{}` twice", im.id)));
            }
            for (j, a) in im.carries.iter().enumerate() {
                check_config(a, &format!("{}[{j}]", p("carries")))?;
                if im.carries[..j].contains(a) {
                    return Err(DeployError::field(p("carries"), format!("`{a}` repeats")));
                }
            }
            if !im.carries(SMOKE_CONFIG) {
                return Err(DeployError::field(
                    p("carries"),
                    format!("every serve image carries {SMOKE_CONFIG} (CI's no-GPU smoke check)"),
                ));
            }
            if let Some(n) = &im.note {
                check_text(n, &p("note"))?;
            }
        }
        // One image per compute: one CUDA image (every preset), at most one CPU
        // image (owner decision 2026-10-25: no per-variant images).
        for (compute, name, want_one) in [(Compute::Gpu, "GPU", true), (Compute::Cpu, "CPU", false)]
        {
            let n = self.images.iter().filter(|i| i.compute == compute).count();
            if n > 1 || (want_one && n == 0) {
                return Err(DeployError::field(
                    "images",
                    format!("{n} {name} images: one CUDA image (and at most one CPU image), configs are sent at boot"),
                ));
            }
        }
        let mut preset_ids = BTreeSet::new();
        let mut claim = |id: &str, path: String| -> Result<(), DeployError> {
            check_id(id, &path)?;
            if preset_ids.insert(id.to_owned()) {
                Ok(())
            } else {
                Err(DeployError::field(path, format!("preset `{id}` twice")))
            }
        };
        for (i, ps) in self.presets.iter().enumerate() {
            let p = |f: &str| format!("presets[{i}].{f}");
            claim(&ps.id, p("id"))?;
            check_text(&ps.title, &p("title"))?;
            check_text(&ps.description, &p("description"))?;
            self.serving_image(&ps.image, &p("image"))?;
            check_config(&ps.config, &p("config"))?;
            if let Some(l) = &ps.licence {
                check_text(l, &p("licence"))?;
            }
            if let Some(d) = ps.container_disk_gb {
                check_positive(d, &p("container_disk_gb"))?;
            }
            check_positive(ps.max_queued, &p("max_queued"))?;
            check_positive(ps.job_timeout_s, &p("job_timeout_s"))?;
            check_positive(ps.stale_after_s, &p("stale_after_s"))?;
            if ps.models.is_empty() {
                return Err(DeployError::field(p("models"), "at least one model"));
            }
            let mut model_ids = BTreeSet::new();
            for (j, m) in ps.models.iter().enumerate() {
                let mp = format!("{}[{j}]", p("models"));
                check_text(&m.variant, &format!("{mp}.variant"))?;
                if let Some(r) = &m.recipe {
                    check_text(r, &format!("{mp}.recipe"))?;
                }
                if !model_ids.insert(m.model_id()) {
                    return Err(DeployError::field(
                        format!("{mp}.id"),
                        format!("model `{}` twice in the pool", m.model_id()),
                    ));
                }
            }
        }
        for (i, fp) in self.family_presets.iter().enumerate() {
            let p = |f: &str| format!("family_presets[{i}].{f}");
            claim(&fp.id, p("id"))?;
            check_text(&fp.title, &p("title"))?;
            check_text(&fp.description, &p("description"))?;
            self.serving_image(&fp.image, &p("image"))?;
            check_config(&fp.config, &p("config"))?;
            check_positive(fp.min_vram_gb, &p("min_vram_gb"))?;
            if fp.members.is_empty() {
                return Err(DeployError::field(p("members"), "at least one member"));
            }
            for (j, m) in fp.members.iter().enumerate() {
                if self.preset(m).is_none() {
                    return Err(DeployError::field(
                        p("members"),
                        format!("`{m}` is not a model preset"),
                    ));
                }
                if fp.members[..j].contains(m) {
                    return Err(DeployError::field(p("members"), format!("`{m}` repeats")));
                }
            }
            for (j, l) in fp.left_out.iter().enumerate() {
                check_text(&l.what, &format!("{}[{j}].what", p("left_out")))?;
                check_text(&l.why, &format!("{}[{j}].why", p("left_out")))?;
            }
        }
        let dispatch: BTreeSet<&str> = self.dispatch.values().map(String::as_str).collect();
        let mut gcp_names = BTreeSet::new();
        for (i, r) in self.gcp.iter().enumerate() {
            let p = |f: &str| format!("gcp[{i}].{f}");
            check_id(&r.name, &p("name"))?;
            if !gcp_names.insert(r.name.as_str()) {
                return Err(DeployError::field(p("name"), format!("`{}` twice", r.name)));
            }
            check_config(&r.config, &p("config"))?;
            if let Some(t) = &r.runpod_twin {
                check_config(t, &p("runpod_twin"))?;
            }
            if let Some(d) = &r.dispatch_family {
                if !dispatch.contains(d.as_str()) {
                    return Err(DeployError::field(
                        p("dispatch_family"),
                        format!("`{d}` is not a dispatch family"),
                    ));
                }
            }
            for c in &r.verify_cells {
                if c.is_empty() || c.chars().any(char::is_whitespace) {
                    return Err(DeployError::field(
                        p("verify_cells"),
                        format!("`{c}`: one cell name"),
                    ));
                }
            }
            if r.runpod_twin.is_some() == !r.served.is_empty() {
                return Err(DeployError::field(
                    p("served"),
                    "only (and then required) without a runpod_twin, which supplies them",
                ));
            }
            for (k, v) in [
                ("machine", &r.machine),
                ("fal_app", &r.fal_app),
                ("minimax_model", &r.minimax_model),
                ("reactor_mode", &r.reactor_mode),
            ] {
                check_text(v, &p(k))?;
            }
        }
        let mut pool_names = BTreeSet::new();
        for (i, pool) in self.autoscale.pools.iter().enumerate() {
            let p = format!("autoscale.pools[{i}]");
            let name = pool.get("name").and_then(toml::Value::as_str).unwrap_or("");
            if name.is_empty() || !pool_names.insert(name.to_owned()) {
                return Err(DeployError::field(
                    format!("{p}.name"),
                    "a unique, non-empty pool name",
                ));
            }
            let family = pool
                .get("family")
                .and_then(toml::Value::as_str)
                .unwrap_or("");
            if !dispatch.contains(family) {
                return Err(DeployError::field(
                    format!("{p}.family"),
                    format!("`{family}` is not a dispatch family"),
                ));
            }
            check_volumes(&toml::Value::Table(pool.clone()), &p)?;
        }
        Ok(())
    }

    fn serving_image(&self, id: &str, path: &str) -> Result<&Image, DeployError> {
        match self.image(id) {
            Some(im) if im.compute == Compute::Gpu => Ok(im),
            _ => Err(DeployError::field(
                path,
                format!("`{id}` is not the GPU image"),
            )),
        }
    }

    /// The rules against the family manifests.
    pub fn validate_with(&self, families: &FamilySet) -> Result<(), DeployError> {
        let fam_ids: BTreeSet<&str> = families
            .families()
            .iter()
            .map(|f| f.id().as_str())
            .collect();
        let named: BTreeSet<&str> = self.dispatch.keys().map(String::as_str).collect();
        if fam_ids != named {
            return Err(DeployError::field(
                "dispatch",
                format!(
                    "names {:?}, the families are {:?}",
                    named.iter().collect::<Vec<_>>(),
                    fam_ids.iter().collect::<Vec<_>>()
                ),
            ));
        }
        for (i, ps) in self.presets.iter().enumerate() {
            let mut variants = Vec::with_capacity(ps.models.len());
            for (j, m) in ps.models.iter().enumerate() {
                let path = format!("presets[{i}].models[{j}]");
                let (f, v) = variant_by_id(families, &m.variant).ok_or_else(|| {
                    DeployError::field(
                        format!("{path}.variant"),
                        format!(
                            "`{}` is not a variant id of the family manifests",
                            m.variant
                        ),
                    )
                })?;
                check_recipe(families, f, v, m.recipe_name(), &format!("{path}.recipe"))?;
                variants.push(v);
            }
            let trees = weight_trees(variants.iter().copied());
            check_prequant(&ps.prequant, &trees, &format!("presets[{i}].prequant"))?;
        }
        for (i, fp) in self.family_presets.iter().enumerate() {
            let p = |f: &str| format!("family_presets[{i}].{f}");
            let variants = self.family_preset_variants(families, fp);
            let fams: BTreeSet<&str> = variants.iter().map(|(f, _)| f.id().as_str()).collect();
            if fams.len() != 1 {
                return Err(DeployError::field(
                    p("members"),
                    format!("one model family per process, not {fams:?}"),
                ));
            }
            let floor = GpuFloor::of(variants.iter().map(|(_, v)| *v));
            if fp.min_vram_gb < floor.min_vram_gib {
                return Err(DeployError::field(
                    p("min_vram_gb"),
                    format!(
                        "{} is below its variants' gpu.min_vram_gib {}",
                        fp.min_vram_gb, floor.min_vram_gib
                    ),
                ));
            }
            let trees = weight_trees(variants.iter().map(|(_, v)| *v));
            check_prequant(&fp.prequant, &trees, &p("prequant"))?;
        }
        Ok(())
    }

    /// The variants a model preset serves, with their families, in its
    /// models' order (names that do not resolve are skipped; validate first).
    pub fn preset_variants<'a>(
        &self,
        families: &'a FamilySet,
        preset: &Preset,
    ) -> Vec<(&'a FamilyManifest, &'a VariantManifest)> {
        preset
            .models
            .iter()
            .filter_map(|m| variant_by_id(families, &m.variant))
            .collect()
    }

    /// The variants of a family preset's members, in order.
    pub fn family_preset_variants<'a>(
        &self,
        families: &'a FamilySet,
        fp: &FamilyPreset,
    ) -> Vec<(&'a FamilyManifest, &'a VariantManifest)> {
        fp.members
            .iter()
            .filter_map(|m| self.preset(m))
            .flat_map(|ps| self.preset_variants(families, ps))
            .collect()
    }

    /// What a preset's variants need from the GPU (the card class a model
    /// preset is served on: fv-control derives its memory from this).
    pub fn preset_floor(&self, families: &FamilySet, preset: &Preset) -> GpuFloor {
        GpuFloor::of(
            self.preset_variants(families, preset)
                .into_iter()
                .map(|(_, v)| v),
        )
    }
}

fn variant_by_id<'a>(
    families: &'a FamilySet,
    id: &str,
) -> Option<(&'a FamilyManifest, &'a VariantManifest)> {
    families.variants().find(|(_, v)| v.id == id)
}

/// A pool's name for a variant: the id, a served name of it, or (a name the
/// manifests do not know) the family's tier alias, which resolves to the
/// first variant of its tier. The exact alias spelling is the engine's
/// (`tier_alias`); `control/gen-configs.mjs` checks it.
fn check_recipe(
    families: &FamilySet,
    family: &FamilyManifest,
    v: &VariantManifest,
    recipe: &str,
    path: &str,
) -> Result<(), DeployError> {
    if let Some((_, other)) = families.find(recipe) {
        return if other.id == v.id {
            Ok(())
        } else {
            Err(DeployError::field(
                path,
                format!("`{recipe}` names variant `{}`, not `{}`", other.id, v.id),
            ))
        };
    }
    let first_of_tier = v
        .tier
        .and_then(|t| family.variants.iter().find(|x| x.tier == Some(t)));
    match first_of_tier {
        Some(first) if first.id == v.id => Ok(()),
        _ => Err(DeployError::field(
            path,
            format!(
                "`{recipe}` is not a name of `{}` (its id, a served name, or its family's tier alias when it is the first of its tier)",
                v.id
            ),
        )),
    }
}

/// The non-auxiliary trees of the variants, unique, in order.
fn weight_trees<'a>(variants: impl IntoIterator<Item = &'a VariantManifest>) -> Vec<&'a str> {
    let mut out: Vec<&str> = Vec::new();
    for v in variants {
        for t in &v.trees {
            if !t.starts_with(AUX_TREE_PREFIX) && !out.contains(&t.as_str()) {
                out.push(t);
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    const FAMILY: &str = r#"
schema = 1

[family]
id = "demo"
display = "Demo family"
licence = "Apache-2.0"
port_doc = "docs/ports/demo.md"

[[variants]]
id = "demo-turbo"
tier = "turbo"
served_names = ["demo-fast"]
recipe_name = "demo-4step"
trees = ["auxiliary/tae/taedemo.safetensors", "demo-base", "demo-lora"]

[variants.gpu]
min_vram_gib = 48
min_sm = 89

[variants.serving]
tasks = ["t2v"]
frames_default = 81

[variants.caps.frames]
step = 4
offset = 1
min = 9
max = 81

[variants.caps.fps]
allowed = [16, 24]
default = 16

[variants.engine]
weights = "${weights}/demo-base"
adapter = "${weights}/demo-lora/adapter.safetensors"

[[variants]]
id = "demo-big"
recipe_name = "demo-big"
trees = ["demo-big"]

[variants.gpu]
min_vram_gib = 96
min_sm = 100
needs = ["nvfp4"]

[variants.serving]
tasks = ["t2v"]
frames_default = 81

[variants.caps.frames]
step = 4
offset = 1
min = 9
max = 81

[variants.caps.fps]
allowed = [16, 24]
default = 16

[variants.engine]
weights = "${weights}/demo-big"
"#;

    const FILE: &str = r#"
schema = 1

[dispatch]
demo = "demo"

[[images]]
id = "cuda"
carries = ["runpod-fake.toml"]

[[images]]
id = "cpu"
carries = ["runpod-fake.toml"]
compute = "cpu"

[[presets]]
id = "demo-turbo"
title = "Demo turbo"
description = "The turbo tier."
standard = true
image = "cuda"
config = "runpod-demo.toml"
prequant = ["demo-base/text_encoder_fp8"]
models = [{ variant = "demo-turbo", id = "demo", recipe = "demo-turbo-alias" }]
max_queued = 32
job_timeout_s = 1800
stale_after_s = 120

[[presets]]
id = "demo-big"
title = "Demo big"
description = "The big one."
image = "cuda"
config = "runpod-demo-big.toml"
models = [{ variant = "demo-big" }]
max_queued = 8
job_timeout_s = 3600
stale_after_s = 180

[[family_presets]]
id = "demo-family"
title = "Demo: {members}"
description = "Every demo variant ({members})."
image = "cuda"
config = "runpod-family-demo.toml"
min_vram_gb = 96
members = ["demo-turbo", "demo-big"]

[[family_presets.left_out]]
what = "nothing"
why = "a test"

[[gcp]]
name = "demo"
config = "gcp-demo.toml"
runpod_twin = "runpod-demo.toml"
dispatch_family = "demo"
verify_cells = ["demo", "aux"]
machine = "g4-standard-48"
fal_app = "x/demo"
minimax_model = "Demo"
reactor_mode = "av"

[[autoscale.pools]]
name = "demo-live"
family = "demo"
kind = "pod"
max_workers = 2

[autoscale.pools.pod]
placements = [{ data_center = "EU-RO-1", volume_id = "bz0ofkfx45" }]
"#;

    fn families() -> FamilySet {
        FamilySet::parse_all([("demo.toml", FAMILY)]).unwrap()
    }

    fn with(from: &str, to: &str) -> Result<DeployManifest, DeployError> {
        assert!(FILE.contains(from), "{from}");
        DeployManifest::parse_with(&FILE.replacen(from, to, 1), &families())
    }

    #[test]
    fn parses_round_trips_and_resolves() {
        let set = families();
        let d = DeployManifest::parse_with(FILE, &set).unwrap();
        assert_eq!(d.images.len(), 2);
        assert_eq!(d.gpu_image().map(|i| i.id.as_str()), Some("cuda"));
        let turbo = d.preset("demo-turbo").unwrap();
        assert_eq!(turbo.models[0].model_id(), "demo");
        assert_eq!(turbo.models[0].recipe_name(), "demo-turbo-alias");
        assert!(d.image("cuda").unwrap().carries(SMOKE_CONFIG));
        assert!(!d.image("cuda").unwrap().carries("runpod-demo-big.toml"));
        let big = d.preset("demo-big").unwrap();
        assert_eq!(big.models[0].model_id(), "demo-big");
        let floor = d.preset_floor(&set, big);
        assert_eq!((floor.min_vram_gib, floor.min_sm), (96, 100));
        assert!(floor.needs.contains("nvfp4"));
        let fp = d.family_preset("demo-family").unwrap();
        assert_eq!(d.family_preset_variants(&set, fp).len(), 2);
        let text = toml::to_string(&d).unwrap();
        assert_eq!(DeployManifest::parse_with(&text, &set).unwrap(), d);
    }

    #[test]
    fn rejects_bad_files_with_paths() {
        let path =
            |r: Result<DeployManifest, DeployError>| r.unwrap_err().path().map(str::to_owned);
        assert_eq!(
            with("schema = 1", "schema = 2").unwrap_err(),
            DeployError::UnsupportedSchema(2)
        );
        assert!(matches!(
            with("title = \"Demo big\"", "titel = \"Demo big\"").unwrap_err(),
            DeployError::Syntax(_)
        ));
        // Images: the preset's must exist and serve on a GPU.
        assert_eq!(
            path(with(
                "image = \"cuda\"\nconfig = \"runpod-demo-big.toml\"",
                "image = \"cpu\"\nconfig = \"runpod-demo-big.toml\""
            )),
            Some("presets[1].image".into())
        );
        assert_eq!(
            path(with(
                "image = \"cuda\"\nconfig = \"runpod-demo-big.toml\"",
                "image = \"nope\"\nconfig = \"runpod-demo-big.toml\""
            )),
            Some("presets[1].image".into())
        );
        // One CUDA image, at most one CPU image, each carrying the smoke config; none bakes a config.
        let images = |r: Result<DeployManifest, DeployError>| r.unwrap_err().to_string();
        assert!(images(with(
            "[[images]]\nid = \"cpu\"",
            "[[images]]\nid = \"h3-max\"\ncarries = [\"runpod-fake.toml\"]\n\n[[images]]\nid = \"cpu\""
        ))
        .contains("2 GPU images"));
        assert!(images(with(
            "id = \"cuda\"\ncarries",
            "id = \"cuda\"\ncompute = \"cpu\"\ncarries"
        ))
        .contains("0 GPU images"));
        assert_eq!(
            path(with(
                "id = \"cuda\"\ncarries = [\"runpod-fake.toml\"]",
                "id = \"cuda\"\ncarries = [\"runpod-demo.toml\"]"
            )),
            Some("images[0].carries".into())
        );
        assert!(matches!(
            with(
                "id = \"cuda\"\ncarries",
                "id = \"cuda\"\nconfig = \"runpod-demo.toml\"\ncarries"
            )
            .unwrap_err(),
            DeployError::Syntax(_)
        ));
        assert_eq!(
            path(with(
                "config = \"runpod-demo-big.toml\"\nmodels",
                "config = \"x/runpod.toml\"\nmodels"
            )),
            Some("presets[1].config".into())
        );
        // Ids: a DNS label, unique across model and family presets.
        assert_eq!(
            path(with("id = \"demo-family\"", "id = \"demo-big\"")),
            Some("family_presets[0].id".into())
        );
        assert_eq!(
            path(with("id = \"demo-big\"\ntitle", "id = \"Demo_Big\"\ntitle")),
            Some("presets[1].id".into())
        );
        assert_eq!(
            path(with("max_queued = 8", "max_queued = 0")),
            Some("presets[1].max_queued".into())
        );
        // Variants: an id the manifests know (a served name is not an id).
        assert_eq!(
            path(with(
                "{ variant = \"demo-big\" }",
                "{ variant = \"demo-huge\" }"
            )),
            Some("presets[1].models[0].variant".into())
        );
        assert_eq!(
            path(with(
                "{ variant = \"demo-big\" }",
                "{ variant = \"demo-fast\" }"
            )),
            Some("presets[1].models[0].variant".into())
        );
        // A recipe naming another variant, or an alias of a variant that is not first of its tier.
        assert_eq!(
            path(with(
                "recipe = \"demo-turbo-alias\"",
                "recipe = \"demo-big\""
            )),
            Some("presets[0].models[0].recipe".into())
        );
        assert_eq!(
            path(with(
                "{ variant = \"demo-big\" }",
                "{ variant = \"demo-big\", recipe = \"demo-max\" }"
            )),
            Some("presets[1].models[0].recipe".into())
        );
        assert!(with("recipe = \"demo-turbo-alias\"", "recipe = \"demo-fast\"").is_ok());
        // Pre-quantized trees nested in a weight tree, never a tree itself.
        assert_eq!(
            path(with(
                "prequant = [\"demo-base/text_encoder_fp8\"]",
                "prequant = [\"demo-big/x\"]"
            )),
            Some("presets[0].prequant".into())
        );
        assert_eq!(
            path(with(
                "prequant = [\"demo-base/text_encoder_fp8\"]",
                "prequant = [\"demo-base\"]"
            )),
            Some("presets[0].prequant".into())
        );
        // A family preset's card class meets its variants' floor; members are model presets.
        let e = with("min_vram_gb = 96", "min_vram_gb = 80").unwrap_err();
        assert_eq!(e.path(), Some("family_presets[0].min_vram_gb"));
        assert!(e.to_string().contains("96"), "{e}");
        assert_eq!(
            path(with(
                "members = [\"demo-turbo\", \"demo-big\"]",
                "members = [\"demo-turbo\", \"demo-family\"]"
            )),
            Some("family_presets[0].members".into())
        );
        assert_eq!(
            path(with(
                "members = [\"demo-turbo\", \"demo-big\"]",
                "members = [\"demo-turbo\", \"demo-turbo\"]"
            )),
            Some("family_presets[0].members".into())
        );
        // Dispatch: exactly the families; GCP and autoscale families are dispatch families.
        assert_eq!(
            path(with("demo = \"demo\"", "other = \"demo\"")),
            Some("dispatch".into())
        );
        assert_eq!(
            path(with(
                "dispatch_family = \"demo\"",
                "dispatch_family = \"ltx\""
            )),
            Some("gcp[0].dispatch_family".into())
        );
        assert_eq!(
            path(with("family = \"demo\"\nkind", "family = \"ltx\"\nkind")),
            Some("autoscale.pools[0].family".into())
        );
        assert_eq!(
            path(with("runpod_twin = \"runpod-demo.toml\"\n", "")),
            Some("gcp[0].served".into())
        );
        // EU only: any other volume, or the EU volume elsewhere, is refused.
        let e = with("volume_id = \"bz0ofkfx45\"", "volume_id = \"s2k01690bi\"").unwrap_err();
        assert_eq!(e.path(), Some("autoscale.pools[0].pod.placements[0]"));
        assert!(e.to_string().contains(EU_VOLUME_ID), "{e}");
        assert!(with("\"EU-RO-1\"", "\"US-CA-2\"").is_err());
        // Without a volume a placement may be anywhere.
        assert!(with(", volume_id = \"bz0ofkfx45\"", "").is_ok());
    }

    #[test]
    fn a_family_preset_serves_one_family() {
        let other = FAMILY
            .replace("id = \"demo\"", "id = \"other\"")
            .replace("demo-turbo", "other-turbo")
            .replace("demo-fast", "other-fast")
            .replace("demo-big", "other-big");
        let set =
            FamilySet::parse_all([("demo.toml", FAMILY), ("other.toml", other.as_str())]).unwrap();
        let text = FILE
            .replace("demo = \"demo\"", "demo = \"demo\"\nother = \"other\"")
            .replace("{ variant = \"demo-big\" }", "{ variant = \"other-big\" }");
        let e = DeployManifest::parse_with(&text, &set).unwrap_err();
        assert_eq!(e.path(), Some("family_presets[0].members"));
    }
}
