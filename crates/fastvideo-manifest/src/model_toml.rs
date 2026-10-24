//! `model.toml` v1 (ADR-0006): the one config file a clone carries.
//!
//! The pinned WO-001 shape: `schema`, `name`, `parent` (forks only, with
//! `@rev`), `license`, `[runtime]`, `[defaults]`, `[components]`, `[change]`
//! (forks only); plus the two widenings decided 2026-10-23 before the first
//! publish: `[variant]`, `[runtime].recipe` / `.profile`, and `[caps]`
//! (mirroring the shapes of `fastvideo-protocol`'s `ModelCaps` without
//! depending on it). Unknown keys are rejected with their dotted path.
//! Floats are allowed here (`guidance`, `aspect`); never in manifests.

use std::collections::BTreeMap;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::name::ModelName;
use crate::path::check_relative_path;

/// The only `model.toml` schema this crate reads and writes.
pub const MODEL_TOML_SCHEMA: u32 = 1;

/// Why a `model.toml` is not valid. [`ModelTomlError::path`] is the dotted
/// field path (`runtime.engine`, `components.vae.path`, `caps.frames`) where
/// one applies.
#[derive(thiserror::Error, Debug, Clone, PartialEq, Eq)]
pub enum ModelTomlError {
    #[error("TOML syntax: {0}")]
    Syntax(String),
    #[error("unsupported schema {0}; this crate reads schema {MODEL_TOML_SCHEMA}")]
    UnsupportedSchema(i64),
    #[error("unknown key `{path}`")]
    UnknownKey { path: String },
    #[error("`{path}`: {msg}")]
    Field { path: String, msg: String },
}

impl ModelTomlError {
    /// The variant name, as the invalid fixtures' `.expect` files spell it.
    pub fn kind(&self) -> &'static str {
        match self {
            ModelTomlError::Syntax(_) => "Syntax",
            ModelTomlError::UnsupportedSchema(_) => "UnsupportedSchema",
            ModelTomlError::UnknownKey { .. } => "UnknownKey",
            ModelTomlError::Field { .. } => "Field",
        }
    }

    /// The dotted field path, when the error names one.
    pub fn path(&self) -> Option<&str> {
        match self {
            ModelTomlError::UnknownKey { path } | ModelTomlError::Field { path, .. } => Some(path),
            ModelTomlError::Syntax(_) => None,
            ModelTomlError::UnsupportedSchema(_) => Some("schema"),
        }
    }

    fn field(path: impl Into<String>, msg: impl Into<String>) -> Self {
        ModelTomlError::Field {
            path: path.into(),
            msg: msg.into(),
        }
    }
}

/// A clone's `model.toml`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ModelToml {
    /// Always 1.
    pub schema: u32,
    /// The published name, without `@rev`.
    #[schemars(with = "String")]
    pub name: ModelName,
    /// Required for forks (with `@rev`); absent for vendor refs.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(with = "Option<String>")]
    pub parent: Option<ModelName>,
    pub license: License,
    pub runtime: Runtime,
    #[serde(default, skip_serializing_if = "Defaults::is_empty")]
    pub defaults: Defaults,
    /// Logical component -> file under the tree. At least one.
    pub components: BTreeMap<String, Component>,
    /// Required for forks.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub change: Option<Change>,
    /// Widening (2026-10-23): what the name grammar does not carry.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub variant: Option<VariantTable>,
    /// Widening (2026-10-23): the ranges the engine negotiates against.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub caps: Option<Caps>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct License {
    /// SPDX identifier, or the vendor's own licence name.
    pub spdx: String,
    /// Whether the registry may mirror these weights.
    pub redistribution: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Runtime {
    /// `fastvideo-rs`.
    pub engine: String,
    /// Semver of the oldest engine that runs this model.
    #[schemars(with = "String")]
    pub min_version: semver::Version,
    /// Pipeline / family id (`wan`, `ltx`, `h3`); open set, validated by the runtime.
    pub pipeline: String,
    /// Widening: the runtime recipe id (e.g. `4step-vsa`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub recipe: Option<String>,
    /// Widening: the technique profile (e.g. `h3/fasth3_4step_vsa`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub profile: Option<String>,
}

/// Generation defaults. Every key optional.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Defaults {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub steps: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub guidance: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub width: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub height: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub frames: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fps: Option<u32>,
}

impl Defaults {
    pub fn is_empty(&self) -> bool {
        *self == Defaults::default()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Component {
    /// Relative path under the tree, `/` separators.
    pub path: String,
    /// Storage dtype (`bf16`, `fp8`, ...).
    pub dtype: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Change {
    pub summary: String,
    pub reason: String,
}

/// Widening: the dimensions of a served variant the name does not carry.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct VariantTable {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub attention_route: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dit_quant: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub decoder: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub adapters: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stages: Option<u32>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tiers: Vec<String>,
}

/// Widening: the ranges the engine negotiates against.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Caps {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub frames: Option<FrameCaps>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub canvas: Option<CanvasCaps>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fps: Option<FpsCaps>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub refs: Option<RefCaps>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub knobs: Option<KnobCaps>,
}

/// Admissible frame counts: `offset + step * k` within `min..=max`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct FrameCaps {
    pub step: u32,
    pub offset: u32,
    pub min: u32,
    pub max: u32,
}

/// Output canvas constraints.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct CanvasCaps {
    /// Side multiple.
    pub multiple: u32,
    /// Maximum pixels.
    pub max_area: u64,
    /// `[min, max]` width / height.
    pub aspect: [f64; 2],
    /// Short-edge tiers; the first is the default.
    pub short_edges: Vec<u32>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct FpsCaps {
    pub allowed: Vec<u32>,
    pub default: u32,
}

/// Reference input limits.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct RefCaps {
    pub images: u32,
    pub videos: u32,
    pub audios: u32,
    pub total: u32,
}

/// Which sampling knobs a request may set.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct KnobCaps {
    #[serde(default)]
    pub seed: bool,
    #[serde(default)]
    pub negative: bool,
    #[serde(default)]
    pub steps: bool,
    #[serde(default)]
    pub guidance: bool,
    #[serde(default)]
    pub guidance_2: bool,
    #[serde(default)]
    pub flow_shift: bool,
}

// ---- key tree: unknown keys and types, with paths -------------------------

#[derive(Clone, Copy)]
enum Kind {
    Int,
    Number,
    Bool,
    Str,
    StrArray,
    IntArray,
    NumberPair,
    Table(&'static [Key]),
    /// A map of tables that all share one shape (`[components]`).
    TableOf(&'static [Key]),
}

#[derive(Clone, Copy)]
struct Key {
    name: &'static str,
    kind: Kind,
    required: bool,
}

const fn req(name: &'static str, kind: Kind) -> Key {
    Key {
        name,
        kind,
        required: true,
    }
}

const fn opt(name: &'static str, kind: Kind) -> Key {
    Key {
        name,
        kind,
        required: false,
    }
}

const LICENSE: &[Key] = &[
    req("spdx", Kind::Str),
    req("redistribution", Kind::Bool),
    opt("url", Kind::Str),
];
const RUNTIME: &[Key] = &[
    req("engine", Kind::Str),
    req("min_version", Kind::Str),
    req("pipeline", Kind::Str),
    opt("recipe", Kind::Str),
    opt("profile", Kind::Str),
];
const DEFAULTS: &[Key] = &[
    opt("steps", Kind::Int),
    opt("guidance", Kind::Number),
    opt("width", Kind::Int),
    opt("height", Kind::Int),
    opt("frames", Kind::Int),
    opt("fps", Kind::Int),
];
const COMPONENT: &[Key] = &[req("path", Kind::Str), req("dtype", Kind::Str)];
const CHANGE: &[Key] = &[req("summary", Kind::Str), req("reason", Kind::Str)];
const VARIANT: &[Key] = &[
    opt("attention_route", Kind::Str),
    opt("dit_quant", Kind::Str),
    opt("decoder", Kind::Str),
    opt("adapters", Kind::StrArray),
    opt("stages", Kind::Int),
    opt("tiers", Kind::StrArray),
];
const FRAMES: &[Key] = &[
    req("step", Kind::Int),
    req("offset", Kind::Int),
    req("min", Kind::Int),
    req("max", Kind::Int),
];
const CANVAS: &[Key] = &[
    req("multiple", Kind::Int),
    req("max_area", Kind::Int),
    req("aspect", Kind::NumberPair),
    req("short_edges", Kind::IntArray),
];
const FPS: &[Key] = &[req("allowed", Kind::IntArray), req("default", Kind::Int)];
const REFS: &[Key] = &[
    req("images", Kind::Int),
    req("videos", Kind::Int),
    req("audios", Kind::Int),
    req("total", Kind::Int),
];
const KNOBS: &[Key] = &[
    opt("seed", Kind::Bool),
    opt("negative", Kind::Bool),
    opt("steps", Kind::Bool),
    opt("guidance", Kind::Bool),
    opt("guidance_2", Kind::Bool),
    opt("flow_shift", Kind::Bool),
];
const CAPS: &[Key] = &[
    opt("frames", Kind::Table(FRAMES)),
    opt("canvas", Kind::Table(CANVAS)),
    opt("fps", Kind::Table(FPS)),
    opt("refs", Kind::Table(REFS)),
    opt("knobs", Kind::Table(KNOBS)),
];
const ROOT: &[Key] = &[
    req("schema", Kind::Int),
    req("name", Kind::Str),
    opt("parent", Kind::Str),
    req("license", Kind::Table(LICENSE)),
    req("runtime", Kind::Table(RUNTIME)),
    opt("defaults", Kind::Table(DEFAULTS)),
    req("components", Kind::TableOf(COMPONENT)),
    opt("change", Kind::Table(CHANGE)),
    opt("variant", Kind::Table(VARIANT)),
    opt("caps", Kind::Table(CAPS)),
];

fn join(prefix: &str, key: &str) -> String {
    if prefix.is_empty() {
        key.to_string()
    } else {
        format!("{prefix}.{key}")
    }
}

fn type_name(v: &toml::Value) -> &'static str {
    match v {
        toml::Value::String(_) => "string",
        toml::Value::Integer(_) => "integer",
        toml::Value::Float(_) => "float",
        toml::Value::Boolean(_) => "boolean",
        toml::Value::Datetime(_) => "datetime",
        toml::Value::Array(_) => "array",
        toml::Value::Table(_) => "table",
    }
}

fn check_kind(path: &str, value: &toml::Value, kind: Kind) -> Result<(), ModelTomlError> {
    let expect = |what: &str| {
        ModelTomlError::field(path, format!("expected {what}, found {}", type_name(value)))
    };
    match kind {
        Kind::Int => match value {
            toml::Value::Integer(i) if *i >= 0 => Ok(()),
            toml::Value::Integer(_) => Err(ModelTomlError::field(
                path,
                "expected a non-negative integer",
            )),
            _ => Err(expect("integer")),
        },
        Kind::Number => match value {
            toml::Value::Integer(_) | toml::Value::Float(_) => Ok(()),
            _ => Err(expect("number")),
        },
        Kind::Bool => match value {
            toml::Value::Boolean(_) => Ok(()),
            _ => Err(expect("boolean")),
        },
        Kind::Str => match value {
            toml::Value::String(_) => Ok(()),
            _ => Err(expect("string")),
        },
        Kind::StrArray => match value {
            toml::Value::Array(items)
                if items.iter().all(|i| matches!(i, toml::Value::String(_))) =>
            {
                Ok(())
            }
            _ => Err(expect("array of strings")),
        },
        Kind::IntArray => match value {
            toml::Value::Array(items)
                if items
                    .iter()
                    .all(|i| matches!(i, toml::Value::Integer(n) if *n >= 0)) =>
            {
                Ok(())
            }
            _ => Err(expect("array of non-negative integers")),
        },
        Kind::NumberPair => match value {
            toml::Value::Array(items)
                if items.len() == 2
                    && items
                        .iter()
                        .all(|i| matches!(i, toml::Value::Integer(_) | toml::Value::Float(_))) =>
            {
                Ok(())
            }
            _ => Err(expect("[min, max] pair of numbers")),
        },
        Kind::Table(keys) => match value {
            toml::Value::Table(t) => check_table(path, t, keys),
            _ => Err(expect("table")),
        },
        Kind::TableOf(keys) => match value {
            toml::Value::Table(t) => {
                for (k, v) in t {
                    let p = join(path, k);
                    match v {
                        toml::Value::Table(inner) => check_table(&p, inner, keys)?,
                        other => {
                            return Err(ModelTomlError::field(
                                p,
                                format!("expected table, found {}", type_name(other)),
                            ))
                        }
                    }
                }
                Ok(())
            }
            _ => Err(expect("table")),
        },
    }
}

fn check_table(prefix: &str, table: &toml::Table, keys: &[Key]) -> Result<(), ModelTomlError> {
    for (k, v) in table {
        let path = join(prefix, k);
        let Some(key) = keys.iter().find(|key| key.name == k) else {
            return Err(ModelTomlError::UnknownKey { path });
        };
        check_kind(&path, v, key.kind)?;
    }
    for key in keys.iter().filter(|k| k.required) {
        if !table.contains_key(key.name) {
            return Err(ModelTomlError::field(join(prefix, key.name), "missing"));
        }
    }
    Ok(())
}

impl ModelToml {
    /// Parse and validate `model.toml` text.
    pub fn parse(text: &str) -> Result<ModelToml, ModelTomlError> {
        let table: toml::Table = text
            .parse()
            .map_err(|e: toml::de::Error| ModelTomlError::Syntax(e.to_string()))?;
        match table.get("schema") {
            Some(toml::Value::Integer(1)) => {}
            Some(toml::Value::Integer(other)) => {
                return Err(ModelTomlError::UnsupportedSchema(*other))
            }
            Some(other) => {
                return Err(ModelTomlError::field(
                    "schema",
                    format!("expected integer, found {}", type_name(other)),
                ))
            }
            None => return Err(ModelTomlError::field("schema", "missing")),
        }
        check_table("", &table, ROOT)?;
        // Value-level parses the key tree cannot express, checked here so
        // that every error names its field.
        if let Some(toml::Value::String(s)) = table.get("name") {
            s.parse::<ModelName>()
                .map_err(|e| ModelTomlError::field("name", e.to_string()))?;
        }
        if let Some(toml::Value::String(s)) = table.get("parent") {
            s.parse::<ModelName>()
                .map_err(|e| ModelTomlError::field("parent", e.to_string()))?;
        }
        if let Some(toml::Value::Table(rt)) = table.get("runtime") {
            if let Some(toml::Value::String(v)) = rt.get("min_version") {
                semver::Version::parse(v)
                    .map_err(|e| ModelTomlError::field("runtime.min_version", e.to_string()))?;
            }
        }
        let mt: ModelToml =
            toml::from_str(text).map_err(|e| ModelTomlError::field("", e.to_string()))?;
        mt.validate()?;
        Ok(mt)
    }

    /// Serialise to TOML text (semantically equal on re-parse; layout may differ).
    pub fn to_toml_string(&self) -> Result<String, toml::ser::Error> {
        toml::to_string(self)
    }

    pub fn is_fork(&self) -> bool {
        self.name.is_fork()
    }

    /// Semantic rules on top of the shape.
    pub fn validate(&self) -> Result<(), ModelTomlError> {
        if self.schema != MODEL_TOML_SCHEMA {
            return Err(ModelTomlError::UnsupportedSchema(i64::from(self.schema)));
        }
        if self.name.rev().is_some() {
            return Err(ModelTomlError::field("name", "must not carry @rev"));
        }
        if self.name.is_fork() {
            let Some(parent) = &self.parent else {
                return Err(ModelTomlError::field("parent", "required for forks"));
            };
            if parent.rev().is_none() {
                return Err(ModelTomlError::field("parent", "must carry @rev"));
            }
            if parent.model() != self.name.model() {
                return Err(ModelTomlError::field(
                    "parent",
                    "a fork keeps its parent's model id",
                ));
            }
            if self.change.is_none() {
                return Err(ModelTomlError::field("change", "required for forks"));
            }
        } else if self.parent.is_some() {
            return Err(ModelTomlError::field(
                "parent",
                "a vendor ref has no parent",
            ));
        }
        if self.license.spdx.trim().is_empty() {
            return Err(ModelTomlError::field("license.spdx", "empty"));
        }
        if self.runtime.engine.trim().is_empty() {
            return Err(ModelTomlError::field("runtime.engine", "empty"));
        }
        if self.runtime.pipeline.trim().is_empty() {
            return Err(ModelTomlError::field("runtime.pipeline", "empty"));
        }
        if self.components.is_empty() {
            return Err(ModelTomlError::field(
                "components",
                "at least one component",
            ));
        }
        for (k, c) in &self.components {
            check_relative_path(&c.path).map_err(|e| {
                ModelTomlError::field(format!("components.{k}.path"), e.to_string())
            })?;
            if c.dtype.trim().is_empty() {
                return Err(ModelTomlError::field(
                    format!("components.{k}.dtype"),
                    "empty",
                ));
            }
        }
        if let Some(change) = &self.change {
            if change.summary.trim().is_empty() {
                return Err(ModelTomlError::field("change.summary", "empty"));
            }
            if change.reason.trim().is_empty() {
                return Err(ModelTomlError::field("change.reason", "empty"));
            }
        }
        let d = &self.defaults;
        for (name, v) in [
            ("steps", d.steps),
            ("width", d.width),
            ("height", d.height),
            ("frames", d.frames),
            ("fps", d.fps),
        ] {
            if v == Some(0) {
                return Err(ModelTomlError::field(
                    format!("defaults.{name}"),
                    "must be positive",
                ));
            }
        }
        if let Some(g) = d.guidance {
            if !g.is_finite() {
                return Err(ModelTomlError::field("defaults.guidance", "must be finite"));
            }
        }
        if let Some(v) = &self.variant {
            if v.stages == Some(0) {
                return Err(ModelTomlError::field("variant.stages", "must be positive"));
            }
            check_unique_nonempty("variant.adapters", &v.adapters)?;
            check_unique_nonempty("variant.tiers", &v.tiers)?;
        }
        if let Some(caps) = &self.caps {
            caps.validate()?;
        }
        Ok(())
    }
}

fn check_unique_nonempty(path: &str, items: &[String]) -> Result<(), ModelTomlError> {
    for (i, item) in items.iter().enumerate() {
        if item.trim().is_empty() {
            return Err(ModelTomlError::field(path, format!("entry {i} is empty")));
        }
        if items[..i].contains(item) {
            return Err(ModelTomlError::field(
                path,
                format!("entry {item:?} repeats"),
            ));
        }
    }
    Ok(())
}

impl Caps {
    pub fn validate(&self) -> Result<(), ModelTomlError> {
        if let Some(f) = &self.frames {
            if f.step == 0 {
                return Err(ModelTomlError::field(
                    "caps.frames.step",
                    "must be positive",
                ));
            }
            if f.min > f.max {
                return Err(ModelTomlError::field(
                    "caps.frames",
                    format!("min {} > max {}", f.min, f.max),
                ));
            }
            if f.min < f.offset {
                return Err(ModelTomlError::field(
                    "caps.frames",
                    format!("min {} < offset {}", f.min, f.offset),
                ));
            }
            for (name, v) in [("min", f.min), ("max", f.max)] {
                if (v - f.offset) % f.step != 0 {
                    return Err(ModelTomlError::field(
                        format!("caps.frames.{name}"),
                        format!(
                            "{v} is not on the grid offset {} + step {} * k",
                            f.offset, f.step
                        ),
                    ));
                }
            }
        }
        if let Some(c) = &self.canvas {
            if c.multiple == 0 {
                return Err(ModelTomlError::field(
                    "caps.canvas.multiple",
                    "must be positive",
                ));
            }
            if c.max_area == 0 {
                return Err(ModelTomlError::field(
                    "caps.canvas.max_area",
                    "must be positive",
                ));
            }
            let [lo, hi] = c.aspect;
            if !(lo.is_finite() && hi.is_finite()) || lo <= 0.0 || lo > hi {
                return Err(ModelTomlError::field(
                    "caps.canvas.aspect",
                    format!("need 0 < min <= max, got [{lo}, {hi}]"),
                ));
            }
            if c.short_edges.is_empty() {
                return Err(ModelTomlError::field(
                    "caps.canvas.short_edges",
                    "at least one tier",
                ));
            }
            if c.short_edges.contains(&0) {
                return Err(ModelTomlError::field(
                    "caps.canvas.short_edges",
                    "must be positive",
                ));
            }
        }
        if let Some(f) = &self.fps {
            if f.allowed.is_empty() {
                return Err(ModelTomlError::field(
                    "caps.fps.allowed",
                    "at least one rate",
                ));
            }
            if f.allowed.contains(&0) {
                return Err(ModelTomlError::field(
                    "caps.fps.allowed",
                    "must be positive",
                ));
            }
            if !f.allowed.contains(&f.default) {
                return Err(ModelTomlError::field(
                    "caps.fps.default",
                    format!("{} is not in allowed {:?}", f.default, f.allowed),
                ));
            }
        }
        if let Some(r) = &self.refs {
            if r.total < r.images.max(r.videos).max(r.audios) {
                return Err(ModelTomlError::field(
                    "caps.refs.total",
                    "smaller than one of images / videos / audios",
                ));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const VENDOR: &str = r#"
schema = 1
name = "wan-ai/wan2.2-t2v"
license = { spdx = "Apache-2.0", redistribution = true }

[runtime]
engine = "fastvideo-rs"
min_version = "0.1.0"
pipeline = "wan"

[components]
transformer = { path = "weights/transformer.safetensors", dtype = "bf16" }
"#;

    #[test]
    fn parses_vendor_and_round_trips() {
        let mt = ModelToml::parse(VENDOR).unwrap();
        assert!(!mt.is_fork());
        let text = mt.to_toml_string().unwrap();
        assert_eq!(ModelToml::parse(&text).unwrap(), mt);
    }

    #[test]
    fn unknown_key_has_a_path() {
        let text = VENDOR.replace("pipeline = \"wan\"", "pipeline = \"wan\"\npipelin = \"x\"");
        assert_eq!(
            ModelToml::parse(&text).unwrap_err(),
            ModelTomlError::UnknownKey {
                path: "runtime.pipelin".to_string()
            }
        );
    }

    #[test]
    fn type_error_has_a_path() {
        let text = format!("{VENDOR}\n[defaults]\nsteps = \"four\"\n");
        let err = ModelToml::parse(&text).unwrap_err();
        assert_eq!(err.path(), Some("defaults.steps"));
    }
}
