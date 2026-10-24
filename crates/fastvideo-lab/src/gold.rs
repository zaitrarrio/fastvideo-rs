//! `lab/gold/<variant>.toml`: the gold standard of one served variant
//! (review `docs/gaps/2026-10-22-architecture-review.md` §7.5, ADR-0009).
//!
//! A gold file names the variant (the family-file id, and the registry name
//! and `@rev` it certifies once published), the tier and the GPU class the
//! tier is defined on, the reference cell its numbers refer to, the measured
//! latency, quality and cost, when `fv-lab` last validated it, and where
//! every number came from.
//!
//! Two rules the types enforce, because a gold standard that invents is
//! worse than none:
//!
//! - **Every number is cited.** Each numeric field that is present must be
//!   named by a `[[source]]` entry (a repository path plus heading); a
//!   source may not cite a field that is absent.
//! - **Every missing number is declared.** Each numeric field of
//!   [`NUMERIC_FIELDS`] that is absent must be named by a `[[gap]]` entry
//!   that says why there is no measurement, so a reader can tell "not yet
//!   measured" from "forgotten".
//!
//! Every numeric field is optional. The numbers are the measured values the
//! sources report (medians where the source gives medians), never bounds
//! with an invented margin; `fv-lab` (plan D1/D3) compares a new run to them
//! with the gate's tolerance (`scripts/gpu/gate-policy.toml`).

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::path::{Path, PathBuf};
use std::str::FromStr;

use fastvideo_manifest::{check_relative_path, FamilyId, ModelName, Tier};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// The only gold-file schema this crate reads and writes.
pub const GOLD_SCHEMA: u32 = 1;

/// Length of a registry `@rev` (the first 12 hex digits of a manifest digest).
pub const REV_LEN: usize = 12;

/// The numeric fields a gold file may carry, as dotted paths. Each one is
/// either present and cited by a `[[source]]`, or absent and declared by a
/// `[[gap]]`.
pub const NUMERIC_FIELDS: [&str; 15] = [
    "latency.p50_s",
    "latency.min_s",
    "latency.max_s",
    "latency.denoise_s",
    "latency.decode_s",
    "latency.load_s",
    "latency.peak_gib",
    "latency.fps",
    "latency.ttff_s",
    "quality.psnr_vs_upstream_min_db",
    "quality.lpips_vs_upstream_max",
    "quality.sharpness_ratio",
    "quality.jitter_ratio",
    "cost.usd_per_hr",
    "cost.usd_per_clip",
];

/// Non-numeric fields a `[[source]]` may cite besides [`NUMERIC_FIELDS`].
pub const CITABLE_FIELDS: [&str; 9] = [
    "tier",
    "tier.gpu",
    "workload",
    "workload.upstream",
    "quality.oracle",
    "quality.byte_identical",
    "quality.gate",
    "quality.cell",
    "latency.measures",
];

/// Relative tolerance of `cost.usd_per_clip` against
/// `usd_per_hr * p50_s / 3600`.
pub const USD_PER_CLIP_TOLERANCE: f64 = 0.01;

/// Why a gold file, or a set of them, is not valid.
#[derive(thiserror::Error, Debug, Clone, PartialEq)]
pub enum GoldError {
    #[error("TOML syntax: {0}")]
    Syntax(String),
    #[error("unsupported schema {0}; this crate reads schema {GOLD_SCHEMA}")]
    UnsupportedSchema(i64),
    #[error("`{path}`: {msg}")]
    Field { path: String, msg: String },
    /// An error inside one file of a set.
    #[error("{file}: {error}")]
    InFile { file: String, error: Box<GoldError> },
    /// Reading a directory or file.
    #[error("{path}: {msg}")]
    Io { path: String, msg: String },
}

impl GoldError {
    fn field(path: impl Into<String>, msg: impl Into<String>) -> Self {
        GoldError::Field {
            path: path.into(),
            msg: msg.into(),
        }
    }

    /// The dotted field path, when the error names one.
    pub fn path(&self) -> Option<&str> {
        match self {
            GoldError::Field { path, .. } => Some(path),
            GoldError::UnsupportedSchema(_) => Some("schema"),
            GoldError::InFile { error, .. } => error.path(),
            _ => None,
        }
    }
}

/// One `lab/gold/<variant>.toml`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Gold {
    /// Always 1.
    pub schema: u32,
    pub model: Model,
    pub tier: TierSpec,
    pub workload: Workload,
    #[serde(default)]
    pub latency: Latency,
    #[serde(default)]
    pub quality: Quality,
    #[serde(default)]
    pub cost: Cost,
    #[serde(default)]
    pub last_validated: LastValidated,
    /// Where every number came from (file path plus heading). At least one.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub source: Vec<Source>,
    /// The numeric fields that have no measurement yet, and why.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub gap: Vec<Gap>,
}

/// The `[model]` table: which variant this file certifies.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Model {
    /// The variant id of `families/<family>.toml`; also the file's stem.
    pub variant: String,
    /// The family id (`h3`, `ltx2`, `wan`).
    #[schemars(with = "String")]
    pub family: FamilyId,
    /// The layer-1 name the variant publishes as, once published; without
    /// `@rev` (that is `rev`). Absent until a model is published.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(with = "Option<String>")]
    pub registry_name: Option<ModelName>,
    /// The published manifest's `@rev` (12 hex digits) this file certifies.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rev: Option<String>,
    /// The technique profile the variant runs (`profiles/<profile>.toml`),
    /// when the recipe has one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub profile: Option<String>,
}

/// The `[tier]` table: the public tier and the GPU class it is defined on.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct TierSpec {
    /// `draft` | `turbo` | `max`; absent for an untiered variant.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<Tier>,
    /// The GPU class (`RTX PRO 6000 Blackwell`).
    pub gpu: String,
    /// Compute capability as `major * 10 + minor` (`120` = sm_120).
    pub sm: u32,
    /// Device memory of the class, GiB.
    pub vram_gib: u32,
    /// The provider offer the class is rented as, when one is pinned.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub offer: Option<String>,
}

/// The `[workload]` table: the reference cell the numbers refer to.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Workload {
    pub width: u32,
    pub height: u32,
    /// Frames of the clip (absent for a stream, which gives `seconds`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub frames: Option<u32>,
    /// Length in seconds of video.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seconds: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fps: Option<u32>,
    /// Denoising forwards.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub steps: Option<u32>,
    /// The prompt set by name or path (`scripts/gpu/prompts-eval.json`).
    pub prompt_set: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub seeds: Vec<u64>,
    /// Anything the cell needs said (warm process, text cache, no MP4). A
    /// number measured under a non-default setting stays in the file only
    /// when this note names the exact setting that reproduces it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub note: Option<String>,
    /// The `FASTVIDEO_*` environment the cell must run under to reproduce
    /// the numbers, when it differs from the served default (for example
    /// `FASTVIDEO_H3_VAE_MXFP8 = "0"`, the bf16 video decode). `note` names
    /// every key too.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub env: BTreeMap<String, String>,
    /// The upstream reference the quality numbers compare against.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub upstream: Option<Upstream>,
}

/// The upstream (Python) reference of a cell.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Upstream {
    /// The oracle target (`fasth3-4step-vsa`) or the upstream cell name.
    pub target: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub image: Option<String>,
    /// The upstream commit.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rev: Option<String>,
}

/// What a wall-clock number measures.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum Measures {
    /// The denoise stage alone.
    Denoise,
    /// Denoise + decode + host glue; text encoding and the MP4 excluded
    /// (`total_s - text_s`, `docs/perf/raw-inference.md`).
    Gen,
    /// The whole generation in one process, text excluded, MP4 included.
    EndToEnd,
    /// An fv-serve job as the API reports it (encode and job overhead included).
    ServeJob,
    /// A causal stream: `p50_s` is seconds of compute per `workload.seconds`
    /// of video, `fps` the steady frame rate.
    Stream,
}

impl Measures {
    pub const ALL: [Measures; 5] = [
        Measures::Denoise,
        Measures::Gen,
        Measures::EndToEnd,
        Measures::ServeJob,
        Measures::Stream,
    ];

    pub fn as_str(&self) -> &'static str {
        match self {
            Measures::Denoise => "denoise",
            Measures::Gen => "gen",
            Measures::EndToEnd => "end_to_end",
            Measures::ServeJob => "serve_job",
            Measures::Stream => "stream",
        }
    }
}

impl FromStr for Measures {
    type Err = GoldError;
    fn from_str(s: &str) -> Result<Self, GoldError> {
        Self::ALL
            .into_iter()
            .find(|m| m.as_str() == s)
            .ok_or_else(|| GoldError::field("latency.measures", format!("unknown value {s:?}")))
    }
}

impl fmt::Display for Measures {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// The `[latency]` table. Seconds, medians where the source gives medians.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Latency {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub measures: Option<Measures>,
    /// The median of the timed runs.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub p50_s: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub min_s: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_s: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub denoise_s: Option<f64>,
    /// The video decode (audio decode included where the source includes it).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub decode_s: Option<f64>,
    /// Model load in the process that measured, outside the timed region.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub load_s: Option<f64>,
    /// Peak device memory of the process, GiB.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub peak_gib: Option<f64>,
    /// Steady frame rate of a stream.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fps: Option<f64>,
    /// Time to first frame of a stream.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ttff_s: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub note: Option<String>,
}

/// The `[quality]` table: the oracle result versus upstream
/// (`docs/oracle.md`) and the gate verdict.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Quality {
    /// The cell the quality numbers refer to, when it is not `[workload]`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cell: Option<String>,
    /// The oracle verdict (`docs/oracle.md`): how ours relates to the
    /// reference at the tensor level.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub oracle: Option<String>,
    /// PSNR of our clip against the upstream reference's, dB.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub psnr_vs_upstream_min_db: Option<f64>,
    /// LPIPS of our clip against the upstream reference's.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lpips_vs_upstream_max: Option<f64>,
    /// Sharpness ratio of the gate's quality section against its baseline.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sharpness_ratio: Option<f64>,
    /// Temporal-jitter ratio of the gate's quality section against its baseline.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub jitter_ratio: Option<f64>,
    /// Whether the same request gives the same bytes across processes
    /// (`docs/perf/determinism.md`), when measured for this variant.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub byte_identical: Option<bool>,
    /// The `fv-gpucheck gate` verdict and its baseline, in words.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gate: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub note: Option<String>,
}

/// The `[cost]` table.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Cost {
    /// The rented price of the tier's GPU class.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usd_per_hr: Option<f64>,
    /// `usd_per_hr * latency.p50_s / 3600`, derived.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usd_per_clip: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub note: Option<String>,
}

impl Cost {
    /// The derived `usd_per_clip` of a price and a clip time, rounded to
    /// five decimals (the precision the files carry).
    pub fn per_clip(usd_per_hr: f64, p50_s: f64) -> f64 {
        (usd_per_hr * p50_s / 3600.0 * 1e5).round() / 1e5
    }
}

/// The `[last_validated]` table: the last `fv-lab run gold` of this file.
/// Empty until `fv-lab` (plan D1) has run.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct LastValidated {
    /// The commit the run validated (7 to 40 hex digits).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub commit: Option<String>,
    /// The run record, relative to the repository root.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub run: Option<String>,
    /// `YYYY-MM-DD`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub date: Option<String>,
}

impl LastValidated {
    pub fn is_validated(&self) -> bool {
        self.commit.is_some()
    }
}

/// One `[[source]]`: where the named fields' values come from.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Source {
    /// Dotted field paths this source backs (`latency.p50_s`); empty for a
    /// note that cites a related measurement without backing a field.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub fields: Vec<String>,
    /// Repository-relative path of the document or artifact.
    pub path: String,
    /// The heading the numbers are under, exactly as the Markdown file writes
    /// it (without the `#`s), or the top-level key of a JSON file. Required
    /// (`validate`); `tests/gold_files.rs` checks that it exists in the file.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub heading: Option<String>,
    /// `YYYY-MM-DD`: the measurement date, or for a source that backs no
    /// field the date of the document state cited. Required (`validate`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub date: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub note: Option<String>,
}

/// One `[[gap]]`: a numeric field with no measurement, and why.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Gap {
    /// A dotted path of [`NUMERIC_FIELDS`].
    pub field: String,
    pub why: String,
}

fn is_hex_lower(s: &str) -> bool {
    !s.is_empty()
        && s.chars()
            .all(|c| c.is_ascii_digit() || ('a'..='f').contains(&c))
}

/// `YYYY-MM-DD` with a plausible month and day.
pub fn check_date(s: &str) -> Result<(), String> {
    let b = s.as_bytes();
    if b.len() != 10 || b[4] != b'-' || b[7] != b'-' {
        return Err(format!("{s:?}: expected YYYY-MM-DD"));
    }
    let num = |r: std::ops::Range<usize>| -> Result<u32, String> {
        s.get(r.clone())
            .filter(|p| p.bytes().all(|c| c.is_ascii_digit()))
            .and_then(|p| p.parse::<u32>().ok())
            .ok_or_else(|| format!("{s:?}: expected YYYY-MM-DD"))
    };
    let (y, m, d) = (num(0..4)?, num(5..7)?, num(8..10)?);
    if y < 2000 || !(1..=12).contains(&m) || !(1..=31).contains(&d) {
        return Err(format!("{s:?}: not a calendar date"));
    }
    Ok(())
}

fn positive(path: &str, v: Option<f64>) -> Result<(), GoldError> {
    if let Some(x) = v {
        if !x.is_finite() || x <= 0.0 {
            return Err(GoldError::field(
                path,
                format!("{x}: must be a positive number"),
            ));
        }
    }
    Ok(())
}

impl Gold {
    /// Parse and validate one gold file.
    pub fn parse(text: &str) -> Result<Gold, GoldError> {
        let table: toml::Table = text
            .parse()
            .map_err(|e: toml::de::Error| GoldError::Syntax(e.to_string()))?;
        match table.get("schema") {
            Some(toml::Value::Integer(1)) => {}
            Some(toml::Value::Integer(other)) => return Err(GoldError::UnsupportedSchema(*other)),
            Some(_) => return Err(GoldError::field("schema", "expected integer")),
            None => return Err(GoldError::field("schema", "missing")),
        }
        let g: Gold = toml::from_str(text).map_err(|e| GoldError::Syntax(e.to_string()))?;
        g.validate()?;
        Ok(g)
    }

    /// Serialise to TOML text (semantically equal on re-parse).
    pub fn to_toml_string(&self) -> Result<String, toml::ser::Error> {
        toml::to_string(self)
    }

    /// The registry key this file certifies, `name@rev`, once both are set.
    pub fn registry_key(&self) -> Option<String> {
        match (&self.model.registry_name, &self.model.rev) {
            (Some(n), Some(r)) => Some(format!("{n}@{r}")),
            _ => None,
        }
    }

    /// The value of a numeric field by its dotted path, when present.
    pub fn numeric(&self, field: &str) -> Option<f64> {
        let l = &self.latency;
        let q = &self.quality;
        let c = &self.cost;
        match field {
            "latency.p50_s" => l.p50_s,
            "latency.min_s" => l.min_s,
            "latency.max_s" => l.max_s,
            "latency.denoise_s" => l.denoise_s,
            "latency.decode_s" => l.decode_s,
            "latency.load_s" => l.load_s,
            "latency.peak_gib" => l.peak_gib,
            "latency.fps" => l.fps,
            "latency.ttff_s" => l.ttff_s,
            "quality.psnr_vs_upstream_min_db" => q.psnr_vs_upstream_min_db,
            "quality.lpips_vs_upstream_max" => q.lpips_vs_upstream_max,
            "quality.sharpness_ratio" => q.sharpness_ratio,
            "quality.jitter_ratio" => q.jitter_ratio,
            "cost.usd_per_hr" => c.usd_per_hr,
            "cost.usd_per_clip" => c.usd_per_clip,
            _ => None,
        }
    }

    /// The numeric fields this file carries.
    pub fn measured_fields(&self) -> Vec<&'static str> {
        NUMERIC_FIELDS
            .into_iter()
            .filter(|f| self.numeric(f).is_some())
            .collect()
    }

    /// The numeric fields this file declares as not yet measured.
    pub fn gap_fields(&self) -> Vec<&str> {
        self.gap.iter().map(|g| g.field.as_str()).collect()
    }

    /// Semantic rules on top of the shape; every error names a dotted path.
    pub fn validate(&self) -> Result<(), GoldError> {
        if self.schema != GOLD_SCHEMA {
            return Err(GoldError::UnsupportedSchema(i64::from(self.schema)));
        }
        self.validate_model()?;
        self.validate_tier()?;
        self.validate_workload()?;
        self.validate_latency()?;
        self.validate_quality()?;
        self.validate_cost()?;
        self.validate_last_validated()?;
        self.validate_sources_and_gaps()
    }

    fn validate_model(&self) -> Result<(), GoldError> {
        let m = &self.model;
        if m.variant.trim().is_empty() {
            return Err(GoldError::field("model.variant", "empty"));
        }
        if m.variant.chars().any(char::is_whitespace) || m.variant.contains('/') {
            return Err(GoldError::field(
                "model.variant",
                "no whitespace or `/` in a variant id",
            ));
        }
        if let Some(n) = &m.registry_name {
            if n.rev().is_some() {
                return Err(GoldError::field(
                    "model.registry_name",
                    "must not carry @rev (that is `model.rev`)",
                ));
            }
        }
        if let Some(r) = &m.rev {
            if r.len() != REV_LEN || !is_hex_lower(r) {
                return Err(GoldError::field(
                    "model.rev",
                    format!("{r:?}: expected {REV_LEN} lowercase hex digits"),
                ));
            }
            if m.registry_name.is_none() {
                return Err(GoldError::field(
                    "model.rev",
                    "a rev without a registry_name certifies nothing",
                ));
            }
        }
        if let Some(p) = &m.profile {
            check_relative_path(p).map_err(|e| GoldError::field("model.profile", e.to_string()))?;
            if p.ends_with(".toml") {
                return Err(GoldError::field(
                    "model.profile",
                    "name the profile without `.toml` (`h3/fasth3_4step_vsa`)",
                ));
            }
        }
        Ok(())
    }

    fn validate_tier(&self) -> Result<(), GoldError> {
        let t = &self.tier;
        if t.gpu.trim().is_empty() {
            return Err(GoldError::field("tier.gpu", "empty"));
        }
        if t.sm < 10 || t.sm > 999 {
            return Err(GoldError::field(
                "tier.sm",
                "major * 10 + minor (120 = sm_120)",
            ));
        }
        if t.vram_gib == 0 {
            return Err(GoldError::field("tier.vram_gib", "must be positive"));
        }
        Ok(())
    }

    fn validate_workload(&self) -> Result<(), GoldError> {
        let w = &self.workload;
        if w.width == 0 || w.height == 0 {
            return Err(GoldError::field(
                "workload",
                "width and height must be positive",
            ));
        }
        if w.frames.is_none() && w.seconds.is_none() {
            return Err(GoldError::field(
                "workload",
                "a cell has `frames` (a clip) or `seconds` (a stream)",
            ));
        }
        if w.frames == Some(0) {
            return Err(GoldError::field("workload.frames", "must be positive"));
        }
        positive("workload.seconds", w.seconds)?;
        if w.fps == Some(0) {
            return Err(GoldError::field("workload.fps", "must be positive"));
        }
        if w.steps == Some(0) {
            return Err(GoldError::field("workload.steps", "must be positive"));
        }
        if w.prompt_set.trim().is_empty() {
            return Err(GoldError::field("workload.prompt_set", "empty"));
        }
        if let Some(u) = &w.upstream {
            if u.target.trim().is_empty() {
                return Err(GoldError::field("workload.upstream.target", "empty"));
            }
        }
        for (k, v) in &w.env {
            let name_ok = k.strip_prefix("FASTVIDEO_").is_some_and(|rest| {
                !rest.is_empty()
                    && rest
                        .chars()
                        .all(|c| c.is_ascii_uppercase() || c.is_ascii_digit() || c == '_')
            });
            if !name_ok {
                return Err(GoldError::field(
                    "workload.env",
                    format!("`{k}`: a FASTVIDEO_* variable (uppercase, digits, `_`)"),
                ));
            }
            if v.trim().is_empty() {
                return Err(GoldError::field("workload.env", format!("`{k}` is empty")));
            }
            let named = w.note.as_deref().is_some_and(|n| n.contains(k.as_str()));
            if !named {
                return Err(GoldError::field(
                    "workload.note",
                    format!("must name `{k}`, the setting the numbers were measured under"),
                ));
            }
        }
        Ok(())
    }

    fn validate_latency(&self) -> Result<(), GoldError> {
        let l = &self.latency;
        for (name, v) in [
            ("latency.p50_s", l.p50_s),
            ("latency.min_s", l.min_s),
            ("latency.max_s", l.max_s),
            ("latency.denoise_s", l.denoise_s),
            ("latency.decode_s", l.decode_s),
            ("latency.load_s", l.load_s),
            ("latency.peak_gib", l.peak_gib),
            ("latency.fps", l.fps),
            ("latency.ttff_s", l.ttff_s),
        ] {
            positive(name, v)?;
        }
        let has_time = l.p50_s.is_some()
            || l.min_s.is_some()
            || l.max_s.is_some()
            || l.denoise_s.is_some()
            || l.decode_s.is_some()
            || l.fps.is_some()
            || l.ttff_s.is_some();
        if has_time && l.measures.is_none() {
            return Err(GoldError::field(
                "latency.measures",
                "say what the times measure (denoise | gen | end_to_end | serve_job | stream)",
            ));
        }
        if let (Some(lo), Some(p)) = (l.min_s, l.p50_s) {
            if lo > p {
                return Err(GoldError::field("latency.min_s", "above p50_s"));
            }
        }
        if let (Some(p), Some(hi)) = (l.p50_s, l.max_s) {
            if hi < p {
                return Err(GoldError::field("latency.max_s", "below p50_s"));
            }
        }
        if let (Some(lo), Some(hi)) = (l.min_s, l.max_s) {
            if lo > hi {
                return Err(GoldError::field("latency.min_s", "above max_s"));
            }
        }
        if l.fps.is_some() && l.measures != Some(Measures::Stream) {
            return Err(GoldError::field(
                "latency.fps",
                "a frame rate is a stream measurement (`measures = \"stream\"`)",
            ));
        }
        Ok(())
    }

    fn validate_quality(&self) -> Result<(), GoldError> {
        let q = &self.quality;
        positive("quality.psnr_vs_upstream_min_db", q.psnr_vs_upstream_min_db)?;
        if let Some(x) = q.lpips_vs_upstream_max {
            if !x.is_finite() || !(0.0..=1.0).contains(&x) {
                return Err(GoldError::field(
                    "quality.lpips_vs_upstream_max",
                    format!("{x}: LPIPS is in 0..=1"),
                ));
            }
        }
        positive("quality.sharpness_ratio", q.sharpness_ratio)?;
        positive("quality.jitter_ratio", q.jitter_ratio)?;
        Ok(())
    }

    fn validate_cost(&self) -> Result<(), GoldError> {
        let c = &self.cost;
        positive("cost.usd_per_hr", c.usd_per_hr)?;
        positive("cost.usd_per_clip", c.usd_per_clip)?;
        if let Some(clip) = c.usd_per_clip {
            let (Some(hr), Some(p50)) = (c.usd_per_hr, self.latency.p50_s) else {
                return Err(GoldError::field(
                    "cost.usd_per_clip",
                    "derived from cost.usd_per_hr and latency.p50_s, which must both be present",
                ));
            };
            let expect = Cost::per_clip(hr, p50);
            if ((clip - expect) / expect).abs() > USD_PER_CLIP_TOLERANCE {
                return Err(GoldError::field(
                    "cost.usd_per_clip",
                    format!("{clip} is not usd_per_hr * p50_s / 3600 = {expect}"),
                ));
            }
        }
        Ok(())
    }

    fn validate_last_validated(&self) -> Result<(), GoldError> {
        let v = &self.last_validated;
        if let Some(c) = &v.commit {
            if c.len() < 7 || c.len() > 40 || !is_hex_lower(c) {
                return Err(GoldError::field(
                    "last_validated.commit",
                    format!("{c:?}: expected 7 to 40 lowercase hex digits"),
                ));
            }
            if v.date.is_none() {
                return Err(GoldError::field(
                    "last_validated.date",
                    "a validated commit has a date",
                ));
            }
        }
        if let Some(r) = &v.run {
            check_relative_path(r)
                .map_err(|e| GoldError::field("last_validated.run", e.to_string()))?;
            if v.commit.is_none() {
                return Err(GoldError::field(
                    "last_validated.commit",
                    "a run record without the commit it validated",
                ));
            }
        }
        if let Some(d) = &v.date {
            check_date(d).map_err(|e| GoldError::field("last_validated.date", e))?;
        }
        Ok(())
    }

    fn validate_sources_and_gaps(&self) -> Result<(), GoldError> {
        if self.source.is_empty() {
            return Err(GoldError::field(
                "source",
                "at least one [[source]] (where the numbers, or their absence, come from)",
            ));
        }
        let known: BTreeSet<&str> = NUMERIC_FIELDS
            .iter()
            .chain(CITABLE_FIELDS.iter())
            .copied()
            .collect();
        let mut cited: BTreeSet<&str> = BTreeSet::new();
        for (i, s) in self.source.iter().enumerate() {
            let p = |f: &str| format!("source[{i}].{f}");
            check_relative_path(&s.path).map_err(|e| GoldError::field(p("path"), e.to_string()))?;
            match &s.heading {
                Some(h) if !h.trim().is_empty() => {}
                _ => {
                    return Err(GoldError::field(
                        p("heading"),
                        "required: the heading (or JSON key) the numbers are under",
                    ))
                }
            }
            match &s.date {
                Some(d) => check_date(d).map_err(|e| GoldError::field(p("date"), e))?,
                None => {
                    return Err(GoldError::field(
                        p("date"),
                        "required: the measurement date (or, for a source that backs no field, the date of the document cited)",
                    ))
                }
            }
            for f in &s.fields {
                if !known.contains(f.as_str()) {
                    return Err(GoldError::field(
                        p("fields"),
                        format!(
                            "`{f}` is not a gold field (see NUMERIC_FIELDS and CITABLE_FIELDS)"
                        ),
                    ));
                }
                if NUMERIC_FIELDS.contains(&f.as_str()) && self.numeric(f).is_none() {
                    return Err(GoldError::field(
                        p("fields"),
                        format!("cites `{f}`, which the file does not carry"),
                    ));
                }
                cited.insert(f.as_str());
            }
        }
        let mut gaps: BTreeSet<&str> = BTreeSet::new();
        for (i, g) in self.gap.iter().enumerate() {
            let p = |f: &str| format!("gap[{i}].{f}");
            if !NUMERIC_FIELDS.contains(&g.field.as_str()) {
                return Err(GoldError::field(
                    p("field"),
                    format!("`{}` is not a numeric gold field", g.field),
                ));
            }
            if self.numeric(&g.field).is_some() {
                return Err(GoldError::field(
                    p("field"),
                    format!("`{}` is declared a gap but the file carries it", g.field),
                ));
            }
            if g.why.trim().is_empty() {
                return Err(GoldError::field(p("why"), "empty"));
            }
            if !gaps.insert(g.field.as_str()) {
                return Err(GoldError::field(
                    p("field"),
                    format!("`{}` repeats", g.field),
                ));
            }
        }
        for f in NUMERIC_FIELDS {
            match (self.numeric(f).is_some(), cited.contains(f), gaps.contains(f)) {
                (true, true, _) | (false, _, true) => {}
                (true, false, _) => {
                    return Err(GoldError::field(
                        f,
                        "a number with no [[source]] citing it (never invent: name the file and heading)",
                    ))
                }
                (false, _, false) => {
                    return Err(GoldError::field(
                        f,
                        "absent and not declared in [[gap]] (say why there is no measurement)",
                    ))
                }
            }
        }
        Ok(())
    }
}

/// Read every `*.toml` directly under `dir` (sorted by file name), parse and
/// validate each, and check that its file stem is its `model.variant`.
pub fn load_dir(dir: &Path) -> Result<Vec<Gold>, GoldError> {
    let io = |p: &Path, e: std::io::Error| GoldError::Io {
        path: p.display().to_string(),
        msg: e.to_string(),
    };
    let mut paths: Vec<PathBuf> = std::fs::read_dir(dir)
        .map_err(|e| io(dir, e))?
        .map(|e| e.map(|e| e.path()).map_err(|e| io(dir, e)))
        .collect::<Result<_, _>>()?;
    paths.retain(|p| p.extension().is_some_and(|x| x == "toml"));
    paths.sort();
    let mut out = Vec::with_capacity(paths.len());
    for p in &paths {
        let label = p.display().to_string();
        let in_file = |e: GoldError| GoldError::InFile {
            file: label.clone(),
            error: Box::new(e),
        };
        let text = std::fs::read_to_string(p).map_err(|e| io(p, e))?;
        let g = Gold::parse(&text).map_err(in_file)?;
        let stem = p.file_stem().and_then(|s| s.to_str()).unwrap_or("");
        if stem != g.model.variant {
            return Err(in_file(GoldError::field(
                "model.variant",
                format!(
                    "`{}` does not match the file name `{stem}.toml`",
                    g.model.variant
                ),
            )));
        }
        out.push(g);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    const FILE: &str = r#"
schema = 1

[model]
variant = "demo-turbo"
family = "demo"
profile = "demo/demo_turbo"

[tier]
name = "turbo"
gpu = "RTX PRO 6000 Blackwell"
sm = 120
vram_gib = 96

[workload]
width = 832
height = 480
frames = 124
fps = 24
steps = 4
prompt_set = "scripts/gpu/prompts-eval.json"
seeds = [7]

[latency]
measures = "gen"
p50_s = 10.0
min_s = 9.5
max_s = 10.5
peak_gib = 60.3

[quality]
psnr_vs_upstream_min_db = 19.7
lpips_vs_upstream_max = 0.236

[cost]
usd_per_hr = 2.49
usd_per_clip = 0.00692

[last_validated]

[[source]]
fields = ["latency.p50_s", "latency.min_s", "latency.max_s", "latency.peak_gib"]
path = "docs/perf/raw-inference.md"
heading = "RTX PRO 6000 (sm_120)"
date = "2026-09-30"

[[source]]
fields = ["quality.psnr_vs_upstream_min_db", "quality.lpips_vs_upstream_max"]
path = "docs/oracle.md"
heading = "Results: H3 turbo on H100"
date = "2026-09-29"

[[source]]
fields = ["cost.usd_per_hr", "cost.usd_per_clip"]
path = "docs/perf/h3-nvfp4-tier-options.md"
heading = "Option 5"
date = "2026-10-14"

[[gap]]
field = "latency.denoise_s"
why = "not in the source"

[[gap]]
field = "latency.decode_s"
why = "not in the source"

[[gap]]
field = "latency.load_s"
why = "not in the source"

[[gap]]
field = "latency.fps"
why = "a clip, not a stream"

[[gap]]
field = "latency.ttff_s"
why = "a clip, not a stream"

[[gap]]
field = "quality.sharpness_ratio"
why = "no gate run"

[[gap]]
field = "quality.jitter_ratio"
why = "no gate run"
"#;

    fn with(from: &str, to: &str) -> Result<Gold, GoldError> {
        assert!(FILE.contains(from), "{from}");
        Gold::parse(&FILE.replacen(from, to, 1))
    }

    #[test]
    fn parses_and_round_trips() {
        let g = Gold::parse(FILE).unwrap();
        assert_eq!(g.model.variant, "demo-turbo");
        assert_eq!(g.tier.name, Some(Tier::Turbo));
        assert_eq!(g.latency.measures, Some(Measures::Gen));
        assert_eq!(g.registry_key(), None);
        assert!(!g.last_validated.is_validated());
        assert_eq!(g.measured_fields().len(), 8);
        assert_eq!(g.gap_fields().len(), 7);
        let text = g.to_toml_string().unwrap();
        assert_eq!(Gold::parse(&text).unwrap(), g);
    }

    #[test]
    fn every_number_is_cited_and_every_gap_declared() {
        // A number nobody cites.
        let e = with(", \"latency.peak_gib\"]", "]").unwrap_err();
        assert_eq!(e.path(), Some("latency.peak_gib"));
        assert!(e.to_string().contains("no [[source]]"), "{e}");
        // A gap that is not declared.
        let e = with(
            "[[gap]]\nfield = \"latency.load_s\"\nwhy = \"not in the source\"\n",
            "",
        )
        .unwrap_err();
        assert_eq!(e.path(), Some("latency.load_s"));
        // A source citing a number the file lacks.
        let e = with(
            "\"latency.peak_gib\"]",
            "\"latency.peak_gib\", \"latency.ttff_s\"]",
        )
        .unwrap_err();
        assert_eq!(e.path(), Some("source[0].fields"));
        // A gap for a number the file carries.
        let e = with("field = \"latency.denoise_s\"", "field = \"latency.p50_s\"").unwrap_err();
        assert_eq!(e.path(), Some("gap[0].field"));
        // An unknown field name.
        let e = with("\"latency.peak_gib\"]", "\"latency.speed\"]").unwrap_err();
        assert_eq!(e.path(), Some("source[0].fields"));
    }

    #[test]
    fn rejects_bad_files_with_paths() {
        assert_eq!(
            with("schema = 1", "schema = 2").unwrap_err(),
            GoldError::UnsupportedSchema(2)
        );
        assert!(matches!(
            with("vram_gib = 96", "vram_gib = 96\nvolts = 3").unwrap_err(),
            GoldError::Syntax(_)
        ));
        assert_eq!(
            with("min_s = 9.5", "min_s = 10.5").unwrap_err().path(),
            Some("latency.min_s")
        );
        assert_eq!(
            with("usd_per_clip = 0.00692", "usd_per_clip = 0.1")
                .unwrap_err()
                .path(),
            Some("cost.usd_per_clip")
        );
        assert_eq!(
            with("measures = \"gen\"\n", "").unwrap_err().path(),
            Some("latency.measures")
        );
        assert_eq!(
            with(
                "lpips_vs_upstream_max = 0.236",
                "lpips_vs_upstream_max = 1.5"
            )
            .unwrap_err()
            .path(),
            Some("quality.lpips_vs_upstream_max")
        );
        assert_eq!(
            with("date = \"2026-09-30\"", "date = \"2026-13-30\"")
                .unwrap_err()
                .path(),
            Some("source[0].date")
        );
        assert_eq!(
            with("[last_validated]", "[last_validated]\ncommit = \"abc1234\"")
                .unwrap_err()
                .path(),
            Some("last_validated.date")
        );
        assert_eq!(
            with(
                "[last_validated]",
                "[last_validated]\nrun = \"artifacts/gold/x.json\""
            )
            .unwrap_err()
            .path(),
            Some("last_validated.commit")
        );
        assert_eq!(
            with(
                "profile = \"demo/demo_turbo\"",
                "profile = \"demo/demo_turbo.toml\""
            )
            .unwrap_err()
            .path(),
            Some("model.profile")
        );
        assert_eq!(
            with(
                "family = \"demo\"",
                "family = \"demo\"\nrev = \"0123456789ab\""
            )
            .unwrap_err()
            .path(),
            Some("model.rev")
        );
        assert_eq!(
            with(
                "family = \"demo\"",
                "family = \"demo\"\nregistry_name = \"minimax/h3@0123456789ab\""
            )
            .unwrap_err()
            .path(),
            Some("model.registry_name")
        );
        assert_eq!(
            with("fps = 24\n", "fps = 24\nseconds = 0.0\n")
                .unwrap_err()
                .path(),
            Some("workload.seconds")
        );
        assert_eq!(
            with("\nfps = 24\nsteps = 4", "\nfps = 24\nsteps = 4\nfoo = 1")
                .map(|_| ())
                .unwrap_err()
                .path(),
            None
        );
    }

    #[test]
    fn registry_key_and_validated_record() {
        let g = with(
            "family = \"demo\"",
            "family = \"demo\"\nregistry_name = \"minimax/h3/dmd:4:bf16\"\nrev = \"0123456789ab\"",
        )
        .unwrap();
        assert_eq!(
            g.registry_key().as_deref(),
            Some("minimax/h3/dmd:4:bf16@0123456789ab")
        );
        let g = with(
            "[last_validated]",
            "[last_validated]\ncommit = \"fb1bce3\"\nrun = \"artifacts/gold/demo/fb1bce3.json\"\ndate = \"2026-10-14\"",
        )
        .unwrap();
        assert!(g.last_validated.is_validated());
    }

    #[test]
    fn streams_need_the_stream_kind() {
        let e = with("measures = \"gen\"", "measures = \"gen\"\nfps = 15.2").unwrap_err();
        assert_eq!(e.path(), Some("latency.fps"));
        let stream = FILE
            .replacen("measures = \"gen\"", "measures = \"stream\"\nfps = 15.2", 1)
            .replacen(
                "[[gap]]\nfield = \"latency.fps\"\nwhy = \"a clip, not a stream\"\n",
                "",
                1,
            );
        // fps is now present but still uncited.
        assert_eq!(
            Gold::parse(&stream).unwrap_err().path(),
            Some("latency.fps")
        );
        let cited = stream.replacen(
            "\"latency.peak_gib\"]",
            "\"latency.peak_gib\", \"latency.fps\"]",
            1,
        );
        let g = Gold::parse(&cited).unwrap();
        assert_eq!(
            (g.latency.measures, g.latency.fps),
            (Some(Measures::Stream), Some(15.2))
        );
    }

    #[test]
    fn sources_need_a_heading_and_a_date() {
        let e = with("heading = \"Option 5\"\n", "").unwrap_err();
        assert_eq!(e.path(), Some("source[2].heading"));
        let e = with("heading = \"Option 5\"\n", "heading = \"  \"\n").unwrap_err();
        assert_eq!(e.path(), Some("source[2].heading"));
        let e = with("date = \"2026-10-14\"\n", "").unwrap_err();
        assert_eq!(e.path(), Some("source[2].date"));
    }

    #[test]
    fn workload_env_is_named_in_the_note() {
        let g = with(
            "seeds = [7]",
            "seeds = [7]\nnote = \"bf16 decode: FASTVIDEO_H3_VAE_MXFP8=0\"\nenv = { FASTVIDEO_H3_VAE_MXFP8 = \"0\" }",
        )
        .unwrap();
        assert_eq!(
            g.workload
                .env
                .get("FASTVIDEO_H3_VAE_MXFP8")
                .map(String::as_str),
            Some("0")
        );
        let e = with(
            "seeds = [7]",
            "seeds = [7]\nnote = \"bf16 decode\"\nenv = { FASTVIDEO_H3_VAE_MXFP8 = \"0\" }",
        )
        .unwrap_err();
        assert_eq!(e.path(), Some("workload.note"));
        let e = with(
            "seeds = [7]",
            "seeds = [7]\nnote = \"PATH=/x\"\nenv = { PATH = \"/x\" }",
        )
        .unwrap_err();
        assert_eq!(e.path(), Some("workload.env"));
        let e = with(
            "seeds = [7]",
            "seeds = [7]\nnote = \"FASTVIDEO_TEST_X\"\nenv = { FASTVIDEO_TEST_X = \"\" }",
        )
        .unwrap_err();
        assert_eq!(e.path(), Some("workload.env"));
    }

    #[test]
    fn per_clip_rounds_to_five_decimals() {
        assert_eq!(Cost::per_clip(2.49, 10.0), 0.00692);
        assert_eq!(Cost::per_clip(2.09, 9.84), 0.00571);
        for m in Measures::ALL {
            assert_eq!(m.as_str().parse::<Measures>().unwrap(), m);
        }
        assert!("fast".parse::<Measures>().is_err());
        assert!(check_date("2026-10-24").is_ok());
        assert!(check_date("2026-1-24").is_err());
        assert!(check_date("1999-10-24").is_err());
    }
}
