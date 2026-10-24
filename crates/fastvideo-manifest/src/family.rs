//! Layer 2 of the model family manifest (ADR-0008): `families/<family>.toml`,
//! the runtime's superset that the engine catalog, the serve configs and
//! the deployment tooling are generated from (review §7.2).
//!
//! A family file is a `[family]` header and one `[[variants]]` table per
//! servable configuration. Everything a variant says is pure data this
//! crate validates (ids, served names, public names, tier, weights trees,
//! GPU needs, the layer-1 [`Caps`] and the runtime's [`ServingCaps`] beside
//! them), except `engine`: an opaque TOML table the runtime interprets
//! (today: the serde form of the engine's `CudaRecipe`). Paths in that table
//! are written relative to the weights root as `${weights}/<tree>/…`
//! ([`WEIGHTS_VAR`]) and tiny-autoencoder files as `${tae}/<file>`
//! ([`TAE_VAR`], `${tae}` alone for the directory); the runtime substitutes
//! its own layout when it loads the table, so a family file names trees,
//! never machines. Every such reference must be covered by `trees`: a
//! `${weights}/<rel>` by an entry equal to it or a component prefix of it, a
//! `${tae}/<file>` by `auxiliary/tae/<file>`. A bare `${tae}` names the
//! tiny-autoencoder directory as a search path (which file a recipe reads
//! there, if any, is the runtime's choice, so `trees` lists that file when
//! the recipe does read one); a bare `${weights}` (the whole root) is
//! refused.
//!
//! Pure: no I/O by default. The optional `fs` feature adds [`load_dir`], the
//! one directory reader every runtime consumer shares.

use std::collections::BTreeSet;
use std::fmt;
use std::str::FromStr;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::model_toml::{Caps, ModelTomlError};
use crate::name::ModelName;
use crate::path::check_relative_path;

/// The only family-file schema this crate reads and writes.
pub const FAMILY_SCHEMA: u32 = 1;
/// Placeholder for the weights root in `engine` paths: `${weights}/<tree>/…`.
pub const WEIGHTS_VAR: &str = "${weights}";
/// Placeholder for the tiny-autoencoder directory in `engine` paths:
/// `${tae}/<file>` (or `${tae}` alone for the directory).
pub const TAE_VAR: &str = "${tae}";
/// Longest family id.
pub const FAMILY_ID_MAX_LEN: usize = 32;
/// Where the tiny autoencoders live under the weights root: the `trees`
/// entry a `${tae}/<file>` reference needs is `auxiliary/tae/<file>`.
pub const TAE_TREE_PREFIX: &str = "auxiliary/tae";

/// Why a family file, or a set of them, is not valid.
#[derive(thiserror::Error, Debug, Clone, PartialEq, Eq)]
pub enum FamilyError {
    #[error("TOML syntax: {0}")]
    Syntax(String),
    #[error("unsupported schema {0}; this crate reads schema {FAMILY_SCHEMA}")]
    UnsupportedSchema(i64),
    #[error("`{path}`: {msg}")]
    Field { path: String, msg: String },
    /// A name (family id, variant id or served name) used twice across a set.
    #[error("{what} `{name}` is declared twice (families {first} and {second})")]
    Duplicate {
        what: &'static str,
        name: String,
        first: String,
        second: String,
    },
    /// An error inside one file of a set.
    #[error("{file}: {error}")]
    InFile {
        file: String,
        error: Box<FamilyError>,
    },
    /// Reading a directory or file (`fs` feature).
    #[error("{path}: {msg}")]
    Io { path: String, msg: String },
}

impl FamilyError {
    fn field(path: impl Into<String>, msg: impl Into<String>) -> Self {
        FamilyError::Field {
            path: path.into(),
            msg: msg.into(),
        }
    }

    /// The dotted field path, when the error names one.
    pub fn path(&self) -> Option<&str> {
        match self {
            FamilyError::Field { path, .. } => Some(path),
            FamilyError::UnsupportedSchema(_) => Some("schema"),
            FamilyError::InFile { error, .. } => error.path(),
            _ => None,
        }
    }
}

impl From<ModelTomlError> for FamilyError {
    fn from(e: ModelTomlError) -> Self {
        match e {
            ModelTomlError::Field { path, msg } => FamilyError::Field { path, msg },
            other => FamilyError::Field {
                path: other.path().unwrap_or("").to_owned(),
                msg: other.to_string(),
            },
        }
    }
}

/// A family id (`h3`, `ltx2`, `wan`): the `runtime.pipeline` of layer 1.
/// Lowercase ASCII letters, digits and `-` (not at the ends), 1..=32 chars.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct FamilyId(String);

impl FamilyId {
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl FromStr for FamilyId {
    type Err = FamilyError;
    fn from_str(s: &str) -> Result<Self, FamilyError> {
        let bad = |msg: &str| FamilyError::field("family.id", format!("{s:?}: {msg}"));
        if s.is_empty() {
            return Err(bad("empty"));
        }
        if s.len() > FAMILY_ID_MAX_LEN {
            return Err(bad("longer than 32 characters"));
        }
        if !s
            .chars()
            .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-')
        {
            return Err(bad("lowercase letters, digits and `-` only"));
        }
        if s.starts_with('-') || s.ends_with('-') {
            return Err(bad("`-` may not start or end the id"));
        }
        Ok(FamilyId(s.to_owned()))
    }
}

impl fmt::Display for FamilyId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl Serialize for FamilyId {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.collect_str(self)
    }
}

impl<'de> Deserialize<'de> for FamilyId {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let s = String::deserialize(d)?;
        s.parse().map_err(serde::de::Error::custom)
    }
}

/// A served variant's quality tier (`fastvideo-protocol`'s `Tier`, as text).
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize, JsonSchema,
)]
#[serde(rename_all = "lowercase")]
pub enum Tier {
    /// Faster configurations that do not pass the quality gate.
    Draft,
    /// The fastest configuration that passes the gate.
    Turbo,
    /// The highest-quality configuration.
    Max,
}

impl Tier {
    pub const ALL: [Tier; 3] = [Tier::Draft, Tier::Turbo, Tier::Max];

    pub fn as_str(&self) -> &'static str {
        match self {
            Tier::Draft => "draft",
            Tier::Turbo => "turbo",
            Tier::Max => "max",
        }
    }
}

impl FromStr for Tier {
    type Err = FamilyError;
    fn from_str(s: &str) -> Result<Self, FamilyError> {
        Self::ALL
            .into_iter()
            .find(|t| t.as_str() == s)
            .ok_or_else(|| FamilyError::field("tier", format!("unknown tier {s:?}")))
    }
}

impl fmt::Display for Tier {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// One `families/<family>.toml`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct FamilyManifest {
    /// Always 1.
    pub schema: u32,
    pub family: FamilyHeader,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub variants: Vec<VariantManifest>,
}

/// The `[family]` table.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct FamilyHeader {
    #[schemars(with = "String")]
    pub id: FamilyId,
    /// Human name (`MiniMax-H3 / FastH3`).
    pub display: String,
    /// The licence the family's weights are under (SPDX id or the vendor's
    /// licence name); the per-tree detail is in each tree's layer-1 manifest.
    pub licence: String,
    /// The port document, relative to the repository root (`docs/ports/h3.md`).
    pub port_doc: String,
}

/// One `[[variants]]` table: a servable configuration of the family.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct VariantManifest {
    /// The one id everything else derives from (the engine catalog id).
    pub id: String,
    /// The layer-1 name this variant publishes as, once it is published.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(with = "Option<String>")]
    pub registry_name: Option<ModelName>,
    /// `draft` | `turbo` | `max`; absent for an untiered variant.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tier: Option<Tier>,
    /// Names the variant answers to besides its id.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub served_names: Vec<String>,
    /// The public API names (the MiniMax `model` ids, `MiniMax-H3-Turbo`) a
    /// server maps onto this variant when it serves it: what the serve
    /// configs' `[aliases]` tables used to list by hand. Several variants may
    /// claim one name (a tier and its task companion): the server gives it
    /// to the served model that generates from text, else the first listed.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub public_names: Vec<String>,
    /// Loaded before readiness when a deployment does not say otherwise.
    #[serde(default = "default_true")]
    pub resident_default: bool,
    /// The short recipe name the capabilities report.
    pub recipe_name: String,
    /// The weights trees the variant reads, named as
    /// `scripts/gpu/weights-manifest.tsv` names them (relative to the weights
    /// root, `/` separators). Sorted, unique.
    pub trees: Vec<String>,
    pub gpu: GpuNeeds,
    /// The ranges the engine negotiates against (layer 1's `[caps]`).
    #[serde(default)]
    pub caps: Caps,
    /// What the runtime advertises beside `caps` and layer 1 does not carry.
    pub serving: ServingCaps,
    /// Opaque to this crate: the runtime's recipe in its own serde form, with
    /// paths under the weights root written `${weights}/…` and tiny
    /// autoencoders `${tae}/…`.
    #[schemars(with = "std::collections::BTreeMap<String, serde_json::Value>")]
    pub engine: toml::Table,
}

fn default_true() -> bool {
    true
}

/// What a variant needs from the GPU, machine-readable.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct GpuNeeds {
    /// The smallest card class the variant is served on, in GiB.
    pub min_vram_gib: u32,
    /// Lowest compute capability as `major * 10 + minor` (`89` = sm_89).
    pub min_sm: u32,
    /// Hardware features (`fp8`, `nvfp4`). Unique, non-empty entries.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub needs: Vec<String>,
}

/// The runtime's capabilities beside layer 1's `Caps`: what `ModelCaps`
/// advertises and `model.toml` does not carry (ADR-0008 amendment, plan
/// A3). Layer 2 only; widening layer 1 is a pinned-format change.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ServingCaps {
    /// The tasks the variant serves, as the protocol names them (`t2v`,
    /// `i2v`, `keyframes`, `ref2v`, `a2v`, `retake`, `extend`). Non-empty,
    /// unique.
    pub tasks: Vec<String>,
    /// The frame count of a request without a length; on `caps.frames`'s grid.
    pub frames_default: u32,
    /// Native audio output; absent for a video-only model.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub audio: Option<AudioCaps>,
    /// The fps only sets the container rate (Wan): the frames are generated
    /// the same regardless.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub fps_container_only: bool,
    /// Canvases off the side multiple are generated padded and cropped back
    /// (LTX 1080 -> 1088).
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub pad_and_crop: bool,
    /// An opt-in canvas tier above the trained pixel budget (the H3 1080P
    /// tier), also listed in `caps.canvas.short_edges`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hd: Option<HdTier>,
    /// The reference-strength knobs (LTX reference-to-video) are honoured.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub reference_strength: bool,
}

/// Native audio output of a variant.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct AudioCaps {
    /// Hz.
    pub native_rate: u32,
    pub channels: u8,
    /// The audio comes from a sidecar model; the variant itself is video-only.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub via_sidecar: bool,
}

/// An opt-in canvas tier with its own pixel budget ([`ServingCaps::hd`]).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct HdTier {
    /// The delivered short edge.
    pub short_edge: u32,
    /// Largest generation canvas area at this tier.
    pub max_area: u64,
    /// Longest clip at this tier, in frames (absent: the frame grid's).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_frames: Option<u32>,
    /// The longer cap an experimental feature flag allows while it is off.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub experimental_max_frames: Option<u32>,
}

/// `${weights}/<rel>` references in an engine table, as `<rel>`, sorted
/// and unique.
pub fn weights_refs(table: &toml::Table) -> Vec<String> {
    let mut out = BTreeSet::new();
    collect_weights_refs(&toml::Value::Table(table.clone()), &mut out);
    out.into_iter().collect()
}

fn collect_weights_refs(v: &toml::Value, out: &mut BTreeSet<String>) {
    match v {
        toml::Value::String(s) => {
            if let Some(rest) = s.strip_prefix(WEIGHTS_VAR) {
                if let Some(rel) = rest.strip_prefix('/') {
                    out.insert(rel.to_owned());
                }
            }
        }
        toml::Value::Array(items) => items.iter().for_each(|i| collect_weights_refs(i, out)),
        toml::Value::Table(t) => t.values().for_each(|i| collect_weights_refs(i, out)),
        _ => {}
    }
}

/// `${tae}` references in an engine table: `Some(<file>)` for
/// `${tae}/<file>`, `None` for the bare directory. Sorted and unique.
pub fn tae_refs(table: &toml::Table) -> Vec<Option<String>> {
    let mut out = BTreeSet::new();
    collect_tae_refs(&toml::Value::Table(table.clone()), &mut out);
    out.into_iter().collect()
}

fn collect_tae_refs(v: &toml::Value, out: &mut BTreeSet<Option<String>>) {
    match v {
        toml::Value::String(s) => {
            if let Some(rest) = s.strip_prefix(TAE_VAR) {
                if rest.is_empty() {
                    out.insert(None);
                } else if let Some(file) = rest.strip_prefix('/') {
                    out.insert(Some(file.to_owned()));
                }
            }
        }
        toml::Value::Array(items) => items.iter().for_each(|i| collect_tae_refs(i, out)),
        toml::Value::Table(t) => t.values().for_each(|i| collect_tae_refs(i, out)),
        _ => {}
    }
}

/// Whether any string of an engine table is the bare `${weights}` (the whole
/// weights root), which no recipe may reference.
fn has_bare_weights(v: &toml::Value) -> bool {
    match v {
        toml::Value::String(s) => s == WEIGHTS_VAR,
        toml::Value::Array(items) => items.iter().any(has_bare_weights),
        toml::Value::Table(t) => t.values().any(has_bare_weights),
        _ => false,
    }
}

/// Whether `tree` is `rel` or a component prefix of it (`a/b` covers
/// `a/b` and `a/b/c`, not `a/bc`).
pub fn tree_covers(tree: &str, rel: &str) -> bool {
    rel == tree
        || rel
            .strip_prefix(tree)
            .is_some_and(|rest| rest.starts_with('/'))
}

impl FamilyManifest {
    /// Parse and validate one family file.
    pub fn parse(text: &str) -> Result<FamilyManifest, FamilyError> {
        let table: toml::Table = text
            .parse()
            .map_err(|e: toml::de::Error| FamilyError::Syntax(e.to_string()))?;
        match table.get("schema") {
            Some(toml::Value::Integer(1)) => {}
            Some(toml::Value::Integer(other)) => {
                return Err(FamilyError::UnsupportedSchema(*other))
            }
            Some(_) => return Err(FamilyError::field("schema", "expected integer")),
            None => return Err(FamilyError::field("schema", "missing")),
        }
        let fm: FamilyManifest =
            toml::from_str(text).map_err(|e| FamilyError::Syntax(e.to_string()))?;
        fm.validate()?;
        Ok(fm)
    }

    /// Serialise to TOML text (semantically equal on re-parse; arrays
    /// inline, one `[[variants]]` table per variant).
    pub fn to_toml_string(&self) -> Result<String, toml::ser::Error> {
        toml::to_string(self)
    }

    pub fn id(&self) -> &FamilyId {
        &self.family.id
    }

    /// The variant with this id or served name.
    pub fn find(&self, name: &str) -> Option<&VariantManifest> {
        self.variants
            .iter()
            .find(|v| v.id == name || v.served_names.iter().any(|s| s == name))
    }

    /// Semantic rules on top of the shape.
    pub fn validate(&self) -> Result<(), FamilyError> {
        if self.schema != FAMILY_SCHEMA {
            return Err(FamilyError::UnsupportedSchema(i64::from(self.schema)));
        }
        let h = &self.family;
        if h.display.trim().is_empty() {
            return Err(FamilyError::field("family.display", "empty"));
        }
        if h.licence.trim().is_empty() {
            return Err(FamilyError::field("family.licence", "empty"));
        }
        check_relative_path(&h.port_doc)
            .map_err(|e| FamilyError::field("family.port_doc", e.to_string()))?;
        if self.variants.is_empty() {
            return Err(FamilyError::field("variants", "at least one variant"));
        }
        let mut names: BTreeSet<&str> = BTreeSet::new();
        for (i, v) in self.variants.iter().enumerate() {
            let p = |field: &str| format!("variants[{i}].{field}");
            v.validate(&p(""))?;
            for (what, n) in std::iter::once(("id", v.id.as_str()))
                .chain(v.served_names.iter().map(|s| ("served_names", s.as_str())))
            {
                if !names.insert(n) {
                    return Err(FamilyError::field(
                        p(what),
                        format!("`{n}` is already a variant id or served name in this file"),
                    ));
                }
            }
        }
        Ok(())
    }
}

impl VariantManifest {
    fn validate(&self, prefix: &str) -> Result<(), FamilyError> {
        let p = |field: &str| format!("{prefix}{field}");
        if self.id.trim().is_empty() {
            return Err(FamilyError::field(p("id"), "empty"));
        }
        if self.id.chars().any(char::is_whitespace) || self.id.contains('/') {
            return Err(FamilyError::field(
                p("id"),
                "no whitespace or `/` in a variant id",
            ));
        }
        for (i, s) in self.served_names.iter().enumerate() {
            if s.trim().is_empty() {
                return Err(FamilyError::field(
                    p("served_names"),
                    format!("entry {i} is empty"),
                ));
            }
            if *s == self.id {
                return Err(FamilyError::field(
                    p("served_names"),
                    format!("`{s}` repeats the variant id (the id is always served)"),
                ));
            }
            if self.served_names[..i].contains(s) {
                return Err(FamilyError::field(
                    p("served_names"),
                    format!("`{s}` repeats"),
                ));
            }
        }
        for (i, n) in self.public_names.iter().enumerate() {
            if n.trim().is_empty() {
                return Err(FamilyError::field(
                    p("public_names"),
                    format!("entry {i} is empty"),
                ));
            }
            if self.public_names[..i].contains(n) {
                return Err(FamilyError::field(
                    p("public_names"),
                    format!("`{n}` repeats"),
                ));
            }
        }
        if let Some(n) = &self.registry_name {
            if n.rev().is_some() {
                return Err(FamilyError::field(
                    p("registry_name"),
                    "must not carry @rev (the rev is the published manifest's)",
                ));
            }
        }
        if self.recipe_name.trim().is_empty() {
            return Err(FamilyError::field(p("recipe_name"), "empty"));
        }
        if self.trees.is_empty() {
            return Err(FamilyError::field(p("trees"), "at least one tree"));
        }
        for (i, t) in self.trees.iter().enumerate() {
            check_relative_path(t)
                .map_err(|e| FamilyError::field(p("trees"), format!("entry {i} {t:?}: {e}")))?;
            if i > 0 && self.trees[i - 1] >= *t {
                return Err(FamilyError::field(
                    p("trees"),
                    format!("not sorted and unique at entry {i} ({t:?})"),
                ));
            }
        }
        if self.gpu.min_vram_gib == 0 {
            return Err(FamilyError::field(
                p("gpu.min_vram_gib"),
                "must be positive",
            ));
        }
        if self.gpu.min_sm < 10 || self.gpu.min_sm > 999 {
            return Err(FamilyError::field(
                p("gpu.min_sm"),
                "major * 10 + minor (80 = sm_80)",
            ));
        }
        for (i, n) in self.gpu.needs.iter().enumerate() {
            if n.trim().is_empty() {
                return Err(FamilyError::field(
                    p("gpu.needs"),
                    format!("entry {i} is empty"),
                ));
            }
            if self.gpu.needs[..i].contains(n) {
                return Err(FamilyError::field(p("gpu.needs"), format!("`{n}` repeats")));
            }
        }
        self.caps
            .validate()
            .map_err(|e| match FamilyError::from(e) {
                FamilyError::Field { path, msg } => FamilyError::Field {
                    path: p(&path),
                    msg,
                },
                other => other,
            })?;
        self.serving.validate(&self.caps, &p("serving."))?;
        if self.engine.is_empty() {
            return Err(FamilyError::field(p("engine"), "empty table"));
        }
        if has_bare_weights(&toml::Value::Table(self.engine.clone())) {
            return Err(FamilyError::field(
                p("engine"),
                format!("references `{WEIGHTS_VAR}` alone (the whole weights root)"),
            ));
        }
        for r in tae_refs(&self.engine) {
            match r {
                Some(file) => {
                    if file.is_empty() {
                        return Err(FamilyError::field(
                            p("engine"),
                            format!("`{TAE_VAR}/` with nothing after it"),
                        ));
                    }
                    check_relative_path(&file)
                        .map_err(|e| FamilyError::field(p("engine"), format!("{file:?}: {e}")))?;
                    let want = format!("{TAE_TREE_PREFIX}/{file}");
                    if !self.trees.iter().any(|t| *t == want) {
                        return Err(FamilyError::field(
                            p("engine"),
                            format!("references `{TAE_VAR}/{file}`; `trees` lacks `{want}`"),
                        ));
                    }
                }
                // The bare directory is a search path (module doc).
                None => {}
            }
        }
        Ok(())
    }

    /// The variant's tier as text, `None` when untiered.
    pub fn tier_str(&self) -> Option<&'static str> {
        self.tier.map(|t| t.as_str())
    }
}

impl ServingCaps {
    /// `prefix` is the dotted path of this table (`variants[0].serving.`).
    fn validate(&self, caps: &Caps, prefix: &str) -> Result<(), FamilyError> {
        let p = |field: &str| format!("{prefix}{field}");
        if self.tasks.is_empty() {
            return Err(FamilyError::field(p("tasks"), "at least one task"));
        }
        for (i, t) in self.tasks.iter().enumerate() {
            if t.is_empty()
                || !t
                    .chars()
                    .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit())
            {
                return Err(FamilyError::field(
                    p("tasks"),
                    format!("entry {i} {t:?}: lowercase letters and digits only"),
                ));
            }
            if self.tasks[..i].contains(t) {
                return Err(FamilyError::field(p("tasks"), format!("`{t}` repeats")));
            }
        }
        if let Some(f) = &caps.frames {
            let on_grid = self.frames_default >= f.offset
                && (self.frames_default - f.offset).is_multiple_of(f.step.max(1));
            if !on_grid || self.frames_default < f.min || self.frames_default > f.max {
                return Err(FamilyError::field(
                    p("frames_default"),
                    format!(
                        "{} is not on the frame grid {}k+{} within {}..={}",
                        self.frames_default, f.step, f.offset, f.min, f.max
                    ),
                ));
            }
        }
        if let Some(a) = &self.audio {
            if a.native_rate == 0 {
                return Err(FamilyError::field(
                    p("audio.native_rate"),
                    "must be positive",
                ));
            }
            if a.channels == 0 {
                return Err(FamilyError::field(p("audio.channels"), "must be positive"));
            }
        }
        if let Some(hd) = &self.hd {
            if hd.max_area == 0 {
                return Err(FamilyError::field(p("hd.max_area"), "must be positive"));
            }
            match &caps.canvas {
                Some(c) if c.short_edges.contains(&hd.short_edge) => {}
                _ => {
                    return Err(FamilyError::field(
                        p("hd.short_edge"),
                        format!(
                            "{} is not listed in caps.canvas.short_edges",
                            hd.short_edge
                        ),
                    ))
                }
            }
            if let (Some(m), Some(x)) = (hd.max_frames, hd.experimental_max_frames) {
                if x < m {
                    return Err(FamilyError::field(
                        p("hd.experimental_max_frames"),
                        "below max_frames",
                    ));
                }
            }
        }
        Ok(())
    }
}

/// Every family file of a tree, checked together: family ids, variant ids
/// and served names are unique across the set.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct FamilySet {
    families: Vec<FamilyManifest>,
}

impl FamilySet {
    /// Build a set from validated manifests, checking cross-file uniqueness.
    pub fn new(mut families: Vec<FamilyManifest>) -> Result<FamilySet, FamilyError> {
        families.sort_by(|a, b| a.family.id.cmp(&b.family.id));
        for w in families.windows(2) {
            if w[0].family.id == w[1].family.id {
                return Err(FamilyError::Duplicate {
                    what: "family id",
                    name: w[0].family.id.to_string(),
                    first: w[0].family.id.to_string(),
                    second: w[1].family.id.to_string(),
                });
            }
        }
        // Variant ids and served names share one namespace across the set.
        let mut seen: std::collections::BTreeMap<&str, &str> = std::collections::BTreeMap::new();
        for f in &families {
            let fid = f.family.id.as_str();
            for v in &f.variants {
                let names = std::iter::once(("variant id", v.id.as_str()))
                    .chain(v.served_names.iter().map(|s| ("served name", s.as_str())));
                for (what, n) in names {
                    if let Some(first) = seen.insert(n, fid) {
                        return Err(FamilyError::Duplicate {
                            what,
                            name: n.to_owned(),
                            first: first.to_owned(),
                            second: fid.to_owned(),
                        });
                    }
                }
            }
        }
        Ok(FamilySet { families })
    }

    /// Parse `(label, text)` pairs (the label names the file in errors).
    pub fn parse_all<'a>(
        files: impl IntoIterator<Item = (&'a str, &'a str)>,
    ) -> Result<FamilySet, FamilyError> {
        let mut out = Vec::new();
        for (label, text) in files {
            let fm = FamilyManifest::parse(text).map_err(|e| FamilyError::InFile {
                file: label.to_owned(),
                error: Box::new(e),
            })?;
            out.push(fm);
        }
        Self::new(out)
    }

    /// The families, sorted by id.
    pub fn families(&self) -> &[FamilyManifest] {
        &self.families
    }

    pub fn into_families(self) -> Vec<FamilyManifest> {
        self.families
    }

    pub fn get(&self, id: &str) -> Option<&FamilyManifest> {
        self.families.iter().find(|f| f.family.id.as_str() == id)
    }

    /// Every variant with its family, in file order.
    pub fn variants(&self) -> impl Iterator<Item = (&FamilyManifest, &VariantManifest)> {
        self.families
            .iter()
            .flat_map(|f| f.variants.iter().map(move |v| (f, v)))
    }

    /// The variant with this id or served name, and its family.
    pub fn find(&self, name: &str) -> Option<(&FamilyManifest, &VariantManifest)> {
        self.families
            .iter()
            .find_map(|f| f.find(name).map(|v| (f, v)))
    }

    pub fn len(&self) -> usize {
        self.families.len()
    }

    pub fn is_empty(&self) -> bool {
        self.families.is_empty()
    }
}

/// Read every `*.toml` directly under `dir` (sorted by file name) into a
/// [`FamilySet`]. The one I/O entry point of this crate, behind the `fs`
/// feature; the registry never enables it.
#[cfg(feature = "fs")]
pub fn load_dir(dir: &std::path::Path) -> Result<FamilySet, FamilyError> {
    let io = |p: &std::path::Path, e: std::io::Error| FamilyError::Io {
        path: p.display().to_string(),
        msg: e.to_string(),
    };
    let mut paths: Vec<std::path::PathBuf> = std::fs::read_dir(dir)
        .map_err(|e| io(dir, e))?
        .map(|e| e.map(|e| e.path()).map_err(|e| io(dir, e)))
        .collect::<Result<_, _>>()?;
    paths.retain(|p| p.extension().is_some_and(|x| x == "toml"));
    paths.sort();
    let mut labels = Vec::with_capacity(paths.len());
    let mut texts = Vec::with_capacity(paths.len());
    for p in &paths {
        labels.push(p.display().to_string());
        texts.push(std::fs::read_to_string(p).map_err(|e| io(p, e))?);
    }
    FamilySet::parse_all(
        labels
            .iter()
            .map(String::as_str)
            .zip(texts.iter().map(String::as_str)),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    const FILE: &str = r#"
schema = 1

[family]
id = "demo"
display = "Demo family"
licence = "Apache-2.0"
port_doc = "docs/ports/demo.md"

[[variants]]
id = "demo-turbo"
tier = "turbo"
served_names = ["demo", "demo-fast"]
public_names = ["Demo-Turbo"]
recipe_name = "demo-4step"
trees = ["auxiliary/tae/taedemo.safetensors", "demo-base", "demo-lora"]

[variants.gpu]
min_vram_gib = 80
min_sm = 89
needs = ["fp8"]

[variants.caps.frames]
step = 4
offset = 1
min = 9
max = 81

[variants.caps.fps]
allowed = [16, 24]
default = 16

[variants.serving]
tasks = ["t2v", "i2v"]
frames_default = 81

[variants.serving.audio]
native_rate = 16000
channels = 1

[variants.engine]
family = "demo"
weights = "${weights}/demo-base"
adapter = "${weights}/demo-lora/adapter.safetensors"
tae = "${tae}/taedemo.safetensors"
steps = 4
"#;

    #[test]
    fn parses_and_round_trips() {
        let fm = FamilyManifest::parse(FILE).unwrap();
        assert_eq!(fm.id().as_str(), "demo");
        let v = fm.find("demo-fast").unwrap();
        assert_eq!((v.id.as_str(), v.tier), ("demo-turbo", Some(Tier::Turbo)));
        assert!(v.resident_default);
        assert_eq!(v.public_names, ["Demo-Turbo"]);
        assert_eq!(v.serving.tasks, ["t2v", "i2v"]);
        assert_eq!(tae_refs(&v.engine), [Some("taedemo.safetensors".to_owned())]);
        assert_eq!(
            weights_refs(&v.engine),
            vec![
                "demo-base".to_owned(),
                "demo-lora/adapter.safetensors".to_owned()
            ]
        );
        let text = fm.to_toml_string().unwrap();
        assert_eq!(FamilyManifest::parse(&text).unwrap(), fm);
    }

    fn with(from: &str, to: &str) -> Result<FamilyManifest, FamilyError> {
        assert!(FILE.contains(from), "{from}");
        FamilyManifest::parse(&FILE.replacen(from, to, 1))
    }

    #[test]
    fn rejects_bad_files_with_paths() {
        assert_eq!(
            with("schema = 1", "schema = 2").unwrap_err(),
            FamilyError::UnsupportedSchema(2)
        );
        assert!(matches!(
            with("steps = 4", "steps = 4\n\n[[variants]]\nid = \"x\"").unwrap_err(),
            FamilyError::Syntax(_)
        ));
        assert_eq!(
            with(
                "served_names = [\"demo\", \"demo-fast\"]",
                "served_names = [\"demo-turbo\"]"
            )
            .unwrap_err()
            .path(),
            Some("variants[0].served_names")
        );
        assert_eq!(
            with(
                "\"demo-base\", \"demo-lora\"",
                "\"demo-lora\", \"demo-base\""
            )
            .unwrap_err()
            .path(),
            Some("variants[0].trees")
        );
        assert_eq!(
            with("\"demo-lora\"]", "\"/demo-lora\"]")
                .unwrap_err()
                .path(),
            Some("variants[0].trees")
        );
        // A `${weights}` reference must be covered by a tree.
        let e = with("${weights}/demo-lora/adapter", "${weights}/other/adapter").unwrap_err();
        assert_eq!(e.path(), Some("variants[0].engine"));
        assert!(e.to_string().contains("other/adapter"), "{e}");
        assert_eq!(
            with("max = 81", "max = 80").unwrap_err().path(),
            Some("variants[0].caps.frames.max")
        );
        assert_eq!(
            with("min_sm = 89", "min_sm = 8").unwrap_err().path(),
            Some("variants[0].gpu.min_sm")
        );
        assert!(matches!(
            with("tier = \"turbo\"", "tier = \"pro\"").unwrap_err(),
            FamilyError::Syntax(_)
        ));
        // A `${tae}/<file>` reference needs `auxiliary/tae/<file>` in trees.
        let e = with("${tae}/taedemo.safetensors", "${tae}/other.safetensors").unwrap_err();
        assert_eq!(e.path(), Some("variants[0].engine"));
        assert!(e.to_string().contains("auxiliary/tae/other.safetensors"), "{e}");
        // The bare directory is a search path (no tree needed); the bare root is refused.
        assert!(with("${tae}/taedemo.safetensors", "${tae}").is_ok());
        let e = with("weights = \"${weights}/demo-base\"", "weights = \"${weights}\"").unwrap_err();
        assert!(e.to_string().contains("alone"), "{e}");
        // Serving caps: tasks, the default frame count on the grid, the HD tier.
        assert_eq!(
            with("tasks = [\"t2v\", \"i2v\"]", "tasks = []").unwrap_err().path(),
            Some("variants[0].serving.tasks")
        );
        assert_eq!(
            with("tasks = [\"t2v\", \"i2v\"]", "tasks = [\"t2v\", \"T2V\"]")
                .unwrap_err()
                .path(),
            Some("variants[0].serving.tasks")
        );
        assert_eq!(
            with("frames_default = 81", "frames_default = 80").unwrap_err().path(),
            Some("variants[0].serving.frames_default")
        );
        assert_eq!(
            with("frames_default = 81", "frames_default = 85").unwrap_err().path(),
            Some("variants[0].serving.frames_default")
        );
        assert_eq!(
            with("channels = 1", "channels = 0").unwrap_err().path(),
            Some("variants[0].serving.audio.channels")
        );
        assert_eq!(
            with(
                "frames_default = 81",
                "frames_default = 81\n[variants.serving.hd]\nshort_edge = 1080\nmax_area = 100"
            )
            .unwrap_err()
            .path(),
            Some("variants[0].serving.hd.short_edge")
        );
        assert_eq!(
            with("public_names = [\"Demo-Turbo\"]", "public_names = [\"Demo-Turbo\", \"Demo-Turbo\"]")
                .unwrap_err()
                .path(),
            Some("variants[0].public_names")
        );
        assert!(matches!(
            with("display = ", "displai = ").unwrap_err(),
            FamilyError::Syntax(_)
        ));
    }

    #[test]
    fn tree_prefixes_are_by_component() {
        assert!(tree_covers("a/b", "a/b"));
        assert!(tree_covers("a/b", "a/b/c.safetensors"));
        assert!(!tree_covers("a/b", "a/bc"));
        assert!(!tree_covers("a/b/c", "a/b"));
    }

    #[test]
    fn set_rejects_cross_file_duplicates() {
        let a = FamilyManifest::parse(FILE).unwrap();
        let mut b = a.clone();
        b.family.id = "other".parse().unwrap();
        assert_eq!(
            FamilySet::new(vec![a.clone(), b.clone()]).unwrap_err(),
            FamilyError::Duplicate {
                what: "variant id",
                name: "demo-turbo".into(),
                first: "demo".into(),
                second: "other".into(),
            }
        );
        b.variants[0].id = "other-turbo".into();
        b.variants[0].served_names = vec!["demo".into()];
        assert!(matches!(
            FamilySet::new(vec![a.clone(), b.clone()]).unwrap_err(),
            FamilyError::Duplicate {
                what: "served name",
                ..
            }
        ));
        b.variants[0].served_names.clear();
        let set = FamilySet::new(vec![b, a]).unwrap();
        assert_eq!(
            set.families()
                .iter()
                .map(|f| f.id().as_str())
                .collect::<Vec<_>>(),
            ["demo", "other"]
        );
        assert_eq!(set.find("other-turbo").unwrap().0.id().as_str(), "other");
        assert_eq!(set.variants().count(), 2);
        let err = FamilySet::parse_all([("a.toml", FILE), ("b.toml", "schema = 1\n")]).unwrap_err();
        assert!(
            matches!(err, FamilyError::InFile { ref file, .. } if file == "b.toml"),
            "{err}"
        );
    }

    #[test]
    fn ids_and_tiers() {
        assert!("h3".parse::<FamilyId>().is_ok());
        assert!("ltx2".parse::<FamilyId>().is_ok());
        for bad in ["", "H3", "-h3", "h3-", "h 3", "h3/x"] {
            assert!(bad.parse::<FamilyId>().is_err(), "{bad:?}");
        }
        for t in Tier::ALL {
            assert_eq!(t.as_str().parse::<Tier>().unwrap(), t);
        }
        assert!("pro".parse::<Tier>().is_err());
    }
}
