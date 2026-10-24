//! The run plan `fv-lab` reads (docs/adr/0019-fv-lab.md §3):
//!
//! - `lab/tiers.toml` ([`Tiers`]): one `[[tier]]` per run plan on one GPU
//!   class, what `scripts/gpu/runpod-matrix.sh` calls a family (`rtx6000`):
//!   the GPU, the money and time caps, the disk, the image flavour, where the
//!   weights come from, the start-of-pod weight report and the cells, in
//!   order.
//! - `lab/cells/<cell>.toml` ([`CellFile`]): one workload (prompt, seed,
//!   length, canvas, steps) and the arms that run it (each arm one run
//!   directory, one fv-gpucheck command line, gated on its weight trees),
//!   plus the paired-clip compares and promotion gates between arms.
//!
//! Every type is serde with `deny_unknown_fields` and schemars; the JSON
//! Schemas are `schemas/{tiers,cell}.v1.json` (drift-checked by
//! `tests/schema.rs`). [`Lab::load`] parses and validates both, and
//! [`Lab::steps`] renders a tier into the exact command lines the runner
//! executes (and `--dry-run` prints).

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use fastvideo_deploy::provider::{CloudType, WeightsSource};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// The only `lab/tiers.toml` schema this crate reads.
pub const TIERS_SCHEMA: u32 = 1;
/// The only `lab/cells/*.toml` schema this crate reads.
pub const CELL_SCHEMA: u32 = 1;
/// Where the run plan lives, relative to the repository (or image) root.
pub const TIERS_PATH: &str = "lab/tiers.toml";
pub const CELLS_DIR: &str = "lab/cells";
/// The runtime image repository (`image = "runtime"`).
pub const RUNTIME_IMAGE_REPO: &str = "ghcr.io/zaitrarrio/fastvideo-rs-runtime";
/// The longest wall-clock cap a tier may ask for: fv-deploy's upper bound
/// without `--force-guards` (`fastvideo_deploy::guard::MAX_CAP_S`, 4 h).
pub const MAX_WALL_CAP_S: u64 = fastvideo_deploy::guard::MAX_CAP_S;

/// Why the run plan is not valid, or could not be read.
#[derive(thiserror::Error, Debug, Clone, PartialEq)]
pub enum LabError {
    #[error("{file}: TOML: {msg}")]
    Syntax { file: String, msg: String },
    #[error("{file}: unsupported schema {got}; this crate reads schema {want}")]
    Schema { file: String, got: u32, want: u32 },
    #[error("{file}: `{field}`: {msg}")]
    Field {
        file: String,
        field: String,
        msg: String,
    },
    #[error("{path}: {msg}")]
    Io { path: String, msg: String },
    #[error("no tier or cell named `{0}` (lab/tiers.toml, lab/cells/)")]
    Unknown(String),
}

fn field(file: &str, field: impl Into<String>, msg: impl Into<String>) -> LabError {
    LabError::Field {
        file: file.to_owned(),
        field: field.into(),
        msg: msg.into(),
    }
}

/// A name usable as a run directory and a pod name part:
/// `[a-z0-9][a-z0-9._-]*`, at most 64 characters.
pub fn is_name(s: &str) -> bool {
    !s.is_empty()
        && s.len() <= 64
        && s.bytes()
            .next()
            .is_some_and(|b| b.is_ascii_lowercase() || b.is_ascii_digit())
        && s.bytes().all(|b| {
            b.is_ascii_lowercase() || b.is_ascii_digit() || matches!(b, b'.' | b'_' | b'-')
        })
}

/// A `verify-weights.sh` cell or a tree name: letters, digits, `._-/:`.
fn is_token(s: &str) -> bool {
    !s.is_empty()
        && s.bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'.' | b'_' | b'-' | b'/' | b':'))
}

// ------------------------------------------------------------------ tiers

/// `lab/tiers.toml`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Tiers {
    /// Always 1.
    pub schema: u32,
    /// The run plans, one per GPU class and purpose.
    #[serde(rename = "tier")]
    pub tiers: Vec<Tier>,
}

/// Where a tier's pods come from. Runpod pods only today (plan C3b adds
/// Vast and GCP to the `Provider` trait).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "kebab-case")]
pub enum TierProvider {
    RunpodPod,
}

/// The image a tier's pod boots.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "kebab-case")]
pub enum ImageFlavour {
    /// `ghcr.io/zaitrarrio/fastvideo-rs-runtime:sha-<7>`: fv-gpucheck,
    /// fv-lab, scripts/gpu and the CUDA runtime (docker/gpucheck.Dockerfile
    /// `runtime`).
    Runtime,
}

impl ImageFlavour {
    /// The image reference for a commit (7 hex digits).
    pub fn image(self, sha7: &str) -> String {
        match self {
            ImageFlavour::Runtime => format!("{RUNTIME_IMAGE_REPO}:sha-{sha7}"),
        }
    }
}

/// One `[[tier]]`: a run plan on one GPU class (a `runpod-matrix.sh` family).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Tier {
    /// The family name (`fv-lab run <name>`; the run directory
    /// `runs/<name>/<tag>` and `artifacts/runpod/<name>/<tag>`).
    pub name: String,
    pub description: String,
    pub provider: TierProvider,
    /// The provider's GPU type id.
    pub gpu: String,
    /// GPUs per pod (1: the runner drives one GPU).
    pub gpu_count: u32,
    #[serde(default)]
    pub cloud: CloudType,
    /// fv-control's region id (`eu`: the only one with a weights volume).
    pub region: String,
    pub image: ImageFlavour,
    /// Price cap ($/hr): a quote over it is never created; a pod billed over
    /// it is deleted at once (fv-deploy's guard).
    pub max_usd_per_hr: f64,
    /// Wall-clock cap (s): the detached backstop deletes the pod then.
    pub wall_cap_s: u64,
    /// Per-arm cap (s), `timeout --kill-after=15` around each command; 0 = none.
    pub cell_timeout_s: u64,
    pub container_disk_gb: u32,
    /// Where the arms' weight trees come from: `volume` (the EU volume,
    /// mounted at `/workspace`, trees under `/workspace/weights`) today;
    /// `registry` (the registry's `hydrate`) later (ADR-0019 §4).
    pub weights: WeightsSource,
    /// `verify-weights.sh` cells the pod checks at start into `weights.log`
    /// (a report only: each arm gates on its own cell).
    #[serde(default)]
    pub preflight: Vec<String>,
    /// Fetch the tiny autoencoders (`fetch-tae.sh`) onto the container disk
    /// at start, as `runpod-http.sh` does unless `FV_SKIP_TAE=1`.
    #[serde(default)]
    pub tae: bool,
    /// The cell files (`lab/cells/<name>.toml`), in run order.
    pub cells: Vec<String>,
}

impl Tiers {
    pub fn parse(text: &str, file: &str) -> Result<Tiers, LabError> {
        let t: Tiers = toml::from_str(text).map_err(|e| LabError::Syntax {
            file: file.to_owned(),
            msg: e.to_string(),
        })?;
        t.validate(file)?;
        Ok(t)
    }

    pub fn validate(&self, file: &str) -> Result<(), LabError> {
        if self.schema != TIERS_SCHEMA {
            return Err(LabError::Schema {
                file: file.to_owned(),
                got: self.schema,
                want: TIERS_SCHEMA,
            });
        }
        let mut seen = BTreeSet::new();
        for (i, t) in self.tiers.iter().enumerate() {
            let at = |f: &str| format!("tier[{i}].{f}");
            if !is_name(&t.name) {
                return Err(field(
                    file,
                    at("name"),
                    format!("`{}` is not [a-z0-9][a-z0-9._-]*", t.name),
                ));
            }
            if !seen.insert(t.name.as_str()) {
                return Err(field(file, at("name"), format!("`{}` twice", t.name)));
            }
            if t.gpu.trim().is_empty() {
                return Err(field(file, at("gpu"), "empty"));
            }
            if t.gpu_count != 1 {
                return Err(field(
                    file,
                    at("gpu_count"),
                    "the runner drives one GPU (1)",
                ));
            }
            if t.region != fastvideo_manifest::deploy::EU_REGION {
                return Err(field(
                    file,
                    at("region"),
                    "only `eu` (the EU weights volume, CLAUDE.md)",
                ));
            }
            if !(t.max_usd_per_hr.is_finite() && t.max_usd_per_hr > 0.0) {
                return Err(field(file, at("max_usd_per_hr"), "a positive number"));
            }
            if t.wall_cap_s == 0 || t.wall_cap_s > MAX_WALL_CAP_S {
                return Err(field(
                    file,
                    at("wall_cap_s"),
                    format!("1..={MAX_WALL_CAP_S} (fv-deploy's 4 h bound)"),
                ));
            }
            if t.container_disk_gb == 0 {
                return Err(field(file, at("container_disk_gb"), "positive"));
            }
            if t.weights != WeightsSource::Volume {
                return Err(field(
                    file,
                    at("weights"),
                    "`volume` only until the registry's hydrate lands (ADR-0019 §4)",
                ));
            }
            if let Some(p) = t.preflight.iter().find(|p| !is_token(p)) {
                return Err(field(
                    file,
                    at("preflight"),
                    format!("`{p}` is not a verify-weights.sh cell"),
                ));
            }
            if t.cells.is_empty() {
                return Err(field(file, at("cells"), "empty"));
            }
            if let Some(c) = t.cells.iter().find(|c| !is_name(c)) {
                return Err(field(
                    file,
                    at("cells"),
                    format!("`{c}` is not a cell name"),
                ));
            }
        }
        Ok(())
    }
}

// ------------------------------------------------------------------ cells

/// `lab/cells/<name>.toml`: one workload and its arms.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct CellFile {
    /// Always 1.
    pub schema: u32,
    /// The file stem.
    pub name: String,
    pub description: String,
    pub workload: Workload,
    /// Named argument lists an arm splices with an element `"{@name}"`.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub lists: BTreeMap<String, Vec<String>>,
    /// The arms, in run order.
    #[serde(rename = "arm")]
    pub arms: Vec<Arm>,
    /// Paired-clip compares (`fv-gpucheck compare-clips`), after the arms.
    #[serde(default, rename = "compare", skip_serializing_if = "Vec::is_empty")]
    pub compares: Vec<Compare>,
    /// Promotion gates (`fv-gpucheck gate`), after the compares.
    #[serde(default, rename = "gate", skip_serializing_if = "Vec::is_empty")]
    pub gates: Vec<GatePair>,
}

/// What every arm of a cell generates. Each value is referenced from the
/// arms' command lines as a placeholder (`{prompt}`, `{seed}`, …), so the
/// declared workload is the one that runs; a value no arm uses is an error.
/// An absent value is the pipeline's (or the recipe's) default.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Workload {
    /// The fv-gpucheck pipeline (`h3`, `ltx2`, `wan`, …).
    pub model: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt: Option<String>,
    /// A prompt-set JSON, relative to the root (`scripts/gpu/prompts-eval.json`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt_set: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seconds: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub frames: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub width: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub height: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub steps: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub note: Option<String>,
}

/// One arm: one run directory `runs/<family>/<tag>/<name>/`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Arm {
    /// The run directory (unique within a tier).
    pub name: String,
    /// The `verify-weights.sh` cell that must pass before the arm runs (an
    /// incomplete tree is recorded as skipped, not run).
    pub gate: String,
    /// The weight trees the arm reads (weights-manifest.tsv `dest` names):
    /// the roots of `gate` and every `{weights}/<tree>` of the command.
    pub trees: Vec<String>,
    /// Denoising steps when the recipe fixes them (descriptive: the recipe
    /// flag sets them, no `--steps` is passed).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub steps: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub note: Option<String>,
    /// `K=V` settings, in order, for `env K=V … fv-gpucheck …`.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub env: Vec<String>,
    /// fv-gpucheck's arguments (the binary is the runner's).
    pub argv: Vec<String>,
}

/// `compare_cells <baseline> <candidate> [args]`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Compare {
    pub baseline: String,
    pub candidate: String,
    /// Extra `compare-clips` arguments (`--off-identity`).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub args: Vec<String>,
}

/// The gate's kind (`fv-gpucheck gate --kind`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "lowercase")]
pub enum GateKind {
    #[default]
    Lossy,
    Exact,
}

impl GateKind {
    pub fn as_str(self) -> &'static str {
        match self {
            GateKind::Lossy => "lossy",
            GateKind::Exact => "exact",
        }
    }
}

/// `gate_cells <baseline> <candidate> [kind] [off]`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct GatePair {
    pub baseline: String,
    pub candidate: String,
    #[serde(default)]
    pub kind: GateKind,
    /// An OFF-arm whose compare with the baseline must be byte-identical.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub off: Option<String>,
}

/// The run-time paths placeholders stand for.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Paths {
    /// `{weights}`: the weight root (`/workspace/weights` on the volume).
    pub weights: PathBuf,
    /// `{scratch}`: the container disk (`/fvscratch`).
    pub scratch: PathBuf,
    /// `{runs}`: the run directory (`/fvscratch/runs/<family>/<tag>`).
    pub runs: PathBuf,
    /// `{root}`: the repository / image root (`/opt/fastvideo-rs`).
    pub root: PathBuf,
}

const PATH_KEYS: [&str; 5] = ["weights", "scratch", "runs", "root", "arm"];
const WORKLOAD_KEYS: [&str; 8] = [
    "prompt",
    "prompt_set",
    "seed",
    "seconds",
    "frames",
    "width",
    "height",
    "steps",
];

impl Workload {
    fn value(&self, key: &str) -> Option<String> {
        match key {
            "prompt" => self.prompt.clone(),
            "prompt_set" => self.prompt_set.clone(),
            "seed" => self.seed.map(|v| v.to_string()),
            "seconds" => self.seconds.map(|v| v.to_string()),
            "frames" => self.frames.map(|v| v.to_string()),
            "width" => self.width.map(|v| v.to_string()),
            "height" => self.height.map(|v| v.to_string()),
            "steps" => self.steps.map(|v| v.to_string()),
            _ => None,
        }
    }
}

/// The `{name}` placeholders of one string, in order.
fn placeholders(s: &str) -> Result<Vec<&str>, String> {
    let mut out = Vec::new();
    let mut rest = s;
    while let Some(i) = rest.find('{') {
        let after = &rest[i + 1..];
        let j = after
            .find('}')
            .ok_or_else(|| format!("`{s}`: unclosed `{{`"))?;
        out.push(&after[..j]);
        rest = &after[j + 1..];
    }
    if rest.contains('}') {
        return Err(format!("`{s}`: unmatched `}}`"));
    }
    Ok(out)
}

impl CellFile {
    pub fn parse(text: &str, file: &str) -> Result<CellFile, LabError> {
        let c: CellFile = toml::from_str(text).map_err(|e| LabError::Syntax {
            file: file.to_owned(),
            msg: e.to_string(),
        })?;
        c.validate(file)?;
        Ok(c)
    }

    /// An arm's argv with its `{@list}` elements spliced (placeholders kept).
    pub fn spliced(&self, arm: &Arm) -> Result<Vec<String>, String> {
        let mut out = Vec::with_capacity(arm.argv.len());
        for a in &arm.argv {
            match a.strip_prefix("{@").and_then(|r| r.strip_suffix('}')) {
                Some(list) => out.extend(
                    self.lists
                        .get(list)
                        .ok_or_else(|| format!("no list `{list}`"))?
                        .iter()
                        .cloned(),
                ),
                None => out.push(a.clone()),
            }
        }
        Ok(out)
    }

    pub fn validate(&self, file: &str) -> Result<(), LabError> {
        if self.schema != CELL_SCHEMA {
            return Err(LabError::Schema {
                file: file.to_owned(),
                got: self.schema,
                want: CELL_SCHEMA,
            });
        }
        if !is_name(&self.name) {
            return Err(field(
                file,
                "name",
                format!("`{}` is not [a-z0-9][a-z0-9._-]*", self.name),
            ));
        }
        if self.workload.model.trim().is_empty() {
            return Err(field(file, "workload.model", "empty"));
        }
        if self.arms.is_empty() {
            return Err(field(file, "arm", "no arm"));
        }
        for (name, list) in &self.lists {
            if list.iter().any(|a| a.starts_with("{@")) {
                return Err(field(file, format!("lists.{name}"), "lists do not nest"));
            }
        }
        let mut names = BTreeSet::new();
        let mut used_lists = BTreeSet::new();
        let mut used_workload = BTreeSet::new();
        for (i, arm) in self.arms.iter().enumerate() {
            let at = |f: &str| format!("arm[{i}].{f}");
            if !is_name(&arm.name) {
                return Err(field(
                    file,
                    at("name"),
                    format!("`{}` is not a run directory name", arm.name),
                ));
            }
            if !names.insert(arm.name.as_str()) {
                return Err(field(file, at("name"), format!("`{}` twice", arm.name)));
            }
            if !is_token(&arm.gate) {
                return Err(field(
                    file,
                    at("gate"),
                    format!("`{}` is not a verify-weights.sh cell", arm.gate),
                ));
            }
            if arm.trees.is_empty() {
                return Err(field(
                    file,
                    at("trees"),
                    "name the weight trees the arm reads",
                ));
            }
            for t in &arm.trees {
                if !is_token(t)
                    || t.starts_with('/')
                    || t.split('/').any(|p| p == ".." || p.is_empty())
                {
                    return Err(field(
                        file,
                        at("trees"),
                        format!("`{t}` is not a tree name"),
                    ));
                }
            }
            if arm.argv.is_empty() {
                return Err(field(file, at("argv"), "empty"));
            }
            for e in &arm.env {
                let key = e.split_once('=').map(|(k, _)| k).unwrap_or("");
                let ok = !key.is_empty()
                    && key
                        .bytes()
                        .next()
                        .is_some_and(|b| b.is_ascii_uppercase() || b == b'_')
                    && key
                        .bytes()
                        .all(|b| b.is_ascii_uppercase() || b.is_ascii_digit() || b == b'_');
                if !ok {
                    return Err(field(file, at("env"), format!("`{e}` is not K=V")));
                }
            }
            for a in &arm.argv {
                if let Some(l) = a.strip_prefix("{@").and_then(|r| r.strip_suffix('}')) {
                    if !self.lists.contains_key(l) {
                        return Err(field(file, at("argv"), format!("no list `{l}`")));
                    }
                    used_lists.insert(l.to_owned());
                }
            }
            let argv = self.spliced(arm).map_err(|m| field(file, at("argv"), m))?;
            let mut referenced_trees = BTreeSet::new();
            for s in arm.env.iter().chain(&argv) {
                let ph = placeholders(s).map_err(|m| field(file, at("argv"), m))?;
                for p in ph {
                    if p.starts_with('@') {
                        return Err(field(
                            file,
                            at("argv"),
                            format!("`{s}`: a list splice is a whole element"),
                        ));
                    }
                    if WORKLOAD_KEYS.contains(&p) {
                        if self.workload.value(p).is_none() {
                            return Err(field(
                                file,
                                at("argv"),
                                format!("`{{{p}}}` but workload.{p} is not set"),
                            ));
                        }
                        used_workload.insert(p.to_owned());
                    } else if !PATH_KEYS.contains(&p) {
                        return Err(field(
                            file,
                            at("argv"),
                            format!("unknown placeholder `{{{p}}}`"),
                        ));
                    }
                }
                let mut rest = s.as_str();
                while let Some(i) = rest.find("{weights}/") {
                    let after = &rest[i + "{weights}/".len()..];
                    let end = after.find(['/', ' ', '"']).unwrap_or(after.len());
                    referenced_trees.insert(after[..end].to_owned());
                    rest = &after[end..];
                }
            }
            for t in &referenced_trees {
                let covered = arm
                    .trees
                    .iter()
                    .any(|tree| tree == t || tree.split('/').next() == Some(t.as_str()));
                if !covered {
                    return Err(field(
                        file,
                        at("trees"),
                        format!("the command reads `{{weights}}/{t}`; list `{t}` in trees"),
                    ));
                }
            }
        }
        for key in WORKLOAD_KEYS {
            if self.workload.value(key).is_some() && !used_workload.contains(key) {
                return Err(field(
                    file,
                    format!("workload.{key}"),
                    "declared but no arm uses it (`{…}` placeholder)",
                ));
            }
        }
        if let Some(l) = self.lists.keys().find(|l| !used_lists.contains(*l)) {
            return Err(field(file, format!("lists.{l}"), "no arm splices it"));
        }
        let arm = |n: &str| names.contains(n);
        for (i, c) in self.compares.iter().enumerate() {
            for n in [&c.baseline, &c.candidate] {
                if !arm(n) {
                    return Err(field(
                        file,
                        format!("compare[{i}]"),
                        format!("`{n}` is not an arm of this cell"),
                    ));
                }
            }
        }
        for (i, g) in self.gates.iter().enumerate() {
            for n in [Some(&g.baseline), Some(&g.candidate), g.off.as_ref()]
                .into_iter()
                .flatten()
            {
                if !arm(n) {
                    return Err(field(
                        file,
                        format!("gate[{i}]"),
                        format!("`{n}` is not an arm of this cell"),
                    ));
                }
            }
        }
        Ok(())
    }

    /// One string with its placeholders replaced.
    fn render(&self, s: &str, arm: &str, p: &Paths) -> Result<String, String> {
        let mut out = String::with_capacity(s.len());
        let mut rest = s;
        while let Some(i) = rest.find('{') {
            out.push_str(&rest[..i]);
            let after = &rest[i + 1..];
            let j = after
                .find('}')
                .ok_or_else(|| format!("`{s}`: unclosed `{{`"))?;
            let key = &after[..j];
            let v = match key {
                "weights" => p.weights.display().to_string(),
                "scratch" => p.scratch.display().to_string(),
                "runs" => p.runs.display().to_string(),
                "root" => p.root.display().to_string(),
                "arm" => arm.to_owned(),
                "prompt_set" => match &self.workload.prompt_set {
                    Some(ps) => p.root.join(ps).display().to_string(),
                    None => return Err(format!("`{s}`: workload.prompt_set is not set")),
                },
                k => self
                    .workload
                    .value(k)
                    .ok_or_else(|| format!("`{s}`: no value for `{{{k}}}`"))?,
            };
            out.push_str(&v);
            rest = &after[j + 1..];
        }
        out.push_str(rest);
        Ok(out)
    }

    /// An arm's full command line, as `runpod-matrix.sh` passes it to
    /// `run_cell`: `[env K=V …] <bin> <args…>`.
    pub fn command(&self, arm: &Arm, bin: &Path, p: &Paths) -> Result<Vec<String>, String> {
        let mut cmd = Vec::new();
        if !arm.env.is_empty() {
            cmd.push("env".to_owned());
            for e in &arm.env {
                cmd.push(self.render(e, &arm.name, p)?);
            }
        }
        cmd.push(bin.display().to_string());
        for a in self.spliced(arm)? {
            cmd.push(self.render(&a, &arm.name, p)?);
        }
        Ok(cmd)
    }
}

// ------------------------------------------------------------------ lab

/// One step of a run, rendered.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Step {
    /// `gated_cell <name> <gate> <command…>`.
    Arm {
        name: String,
        gate: String,
        trees: Vec<String>,
        command: Vec<String>,
    },
    /// `compare_cells <baseline> <candidate> <args…>`.
    Compare {
        baseline: String,
        candidate: String,
        args: Vec<String>,
    },
    /// `gate_cells <baseline> <candidate> <kind> [off]`.
    Gate {
        baseline: String,
        candidate: String,
        kind: GateKind,
        off: Option<String>,
    },
}

/// The checked-in run plan.
#[derive(Debug, Clone, PartialEq)]
pub struct Lab {
    /// The root `lab/` is under (the repository, or `/opt/fastvideo-rs`).
    pub root: PathBuf,
    pub tiers: Tiers,
    pub cells: BTreeMap<String, CellFile>,
}

impl Lab {
    /// The nearest directory at or above `start` holding `lab/tiers.toml`.
    pub fn find_root(start: &Path) -> Option<PathBuf> {
        start
            .ancestors()
            .find(|d| d.join(TIERS_PATH).is_file())
            .map(Path::to_path_buf)
    }

    /// Reads and validates `<root>/lab/tiers.toml` and every
    /// `<root>/lab/cells/*.toml`.
    pub fn load(root: &Path) -> Result<Lab, LabError> {
        let io = |p: &Path, e: std::io::Error| LabError::Io {
            path: p.display().to_string(),
            msg: e.to_string(),
        };
        let tp = root.join(TIERS_PATH);
        let tiers = Tiers::parse(
            &std::fs::read_to_string(&tp).map_err(|e| io(&tp, e))?,
            TIERS_PATH,
        )?;
        let dir = root.join(CELLS_DIR);
        let mut paths: Vec<PathBuf> = std::fs::read_dir(&dir)
            .map_err(|e| io(&dir, e))?
            .map(|e| e.map(|e| e.path()).map_err(|e| io(&dir, e)))
            .collect::<Result<_, _>>()?;
        paths.retain(|p| p.extension().is_some_and(|x| x == "toml"));
        paths.sort();
        let mut cells = BTreeMap::new();
        for p in &paths {
            let label = format!(
                "{CELLS_DIR}/{}",
                p.file_name().and_then(|s| s.to_str()).unwrap_or("?")
            );
            let c = CellFile::parse(&std::fs::read_to_string(p).map_err(|e| io(p, e))?, &label)?;
            let stem = p.file_stem().and_then(|s| s.to_str()).unwrap_or("");
            if stem != c.name {
                return Err(field(
                    &label,
                    "name",
                    format!("`{}` is not the file stem `{stem}`", c.name),
                ));
            }
            cells.insert(c.name.clone(), c);
        }
        let lab = Lab {
            root: root.to_path_buf(),
            tiers,
            cells,
        };
        lab.validate()?;
        Ok(lab)
    }

    fn validate(&self) -> Result<(), LabError> {
        for t in &self.tiers.tiers {
            let mut arms = BTreeSet::new();
            for c in &t.cells {
                let cell = self.cells.get(c).ok_or_else(|| {
                    field(
                        TIERS_PATH,
                        format!("tier `{}`.cells", t.name),
                        format!("no {CELLS_DIR}/{c}.toml"),
                    )
                })?;
                for a in &cell.arms {
                    if !arms.insert(a.name.as_str()) {
                        return Err(field(
                            TIERS_PATH,
                            format!("tier `{}`.cells", t.name),
                            format!(
                                "arm `{}` in two cells (run directories must be unique)",
                                a.name
                            ),
                        ));
                    }
                }
            }
        }
        Ok(())
    }

    pub fn tier(&self, name: &str) -> Option<&Tier> {
        self.tiers.tiers.iter().find(|t| t.name == name)
    }

    /// `fv-lab run <target>`: a tier (all its cells), or one cell of a tier
    /// (`family`: the tier whose run directory it writes into; default the
    /// first tier listing it). The tier and the cells, in run order.
    pub fn resolve(
        &self,
        target: &str,
        family: Option<&str>,
    ) -> Result<(&Tier, Vec<&CellFile>), LabError> {
        if let Some(t) = self.tier(target) {
            let cells = t.cells.iter().filter_map(|c| self.cells.get(c)).collect();
            return Ok((t, cells));
        }
        let cell = self
            .cells
            .get(target)
            .ok_or_else(|| LabError::Unknown(target.to_owned()))?;
        let tier = match family {
            Some(f) => self
                .tier(f)
                .ok_or_else(|| LabError::Unknown(f.to_owned()))?,
            None => self
                .tiers
                .tiers
                .iter()
                .find(|t| t.cells.iter().any(|c| c == target))
                .ok_or_else(|| {
                    LabError::Unknown(format!("{target} (in no tier; pass --family)"))
                })?,
        };
        Ok((tier, vec![cell]))
    }

    /// The steps of `cells`, rendered with `bin` and `paths`: per cell its
    /// arms, then its compares, then its gates.
    pub fn steps(cells: &[&CellFile], bin: &Path, paths: &Paths) -> Result<Vec<Step>, LabError> {
        let mut out = Vec::new();
        for c in cells {
            let label = format!("{CELLS_DIR}/{}.toml", c.name);
            for a in &c.arms {
                let command = c
                    .command(a, bin, paths)
                    .map_err(|m| field(&label, format!("arm `{}`", a.name), m))?;
                out.push(Step::Arm {
                    name: a.name.clone(),
                    gate: a.gate.clone(),
                    trees: a.trees.clone(),
                    command,
                });
            }
            for cmp in &c.compares {
                out.push(Step::Compare {
                    baseline: cmp.baseline.clone(),
                    candidate: cmp.candidate.clone(),
                    args: cmp.args.clone(),
                });
            }
            for g in &c.gates {
                out.push(Step::Gate {
                    baseline: g.baseline.clone(),
                    candidate: g.candidate.clone(),
                    kind: g.kind,
                    off: g.off.clone(),
                });
            }
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const CELL: &str = r#"
schema = 1
name = "c1"
description = "test"

[workload]
model = "h3"
prompt = "a b"
seed = 7

[lists]
common = ["--prompt", "{prompt}", "--seed", "{seed}"]

[[arm]]
name = "a1"
gate = "h3-base"
trees = ["h3-base"]
env = ["FASTVIDEO_X={weights}/h3-base/x"]
argv = ["h3", "gen", "--weights", "{weights}/h3-base", "--clip-dir", "{runs}/{arm}/frames", "{@common}"]

[[arm]]
name = "a2"
gate = "h3-base"
trees = ["h3-base"]
argv = ["h3", "gen", "{@common}"]

[[compare]]
baseline = "a1"
candidate = "a2"
"#;

    fn paths() -> Paths {
        Paths {
            weights: "/workspace/weights".into(),
            scratch: "/fvscratch".into(),
            runs: "/fvscratch/runs/f/t".into(),
            root: "/opt/fastvideo-rs".into(),
        }
    }

    #[test]
    fn renders_env_prefix_lists_and_placeholders() {
        let c = CellFile::parse(CELL, "c1.toml").unwrap();
        let cmd = c
            .command(&c.arms[0], Path::new("/bin/fv"), &paths())
            .unwrap();
        assert_eq!(
            cmd,
            [
                "env",
                "FASTVIDEO_X=/workspace/weights/h3-base/x",
                "/bin/fv",
                "h3",
                "gen",
                "--weights",
                "/workspace/weights/h3-base",
                "--clip-dir",
                "/fvscratch/runs/f/t/a1/frames",
                "--prompt",
                "a b",
                "--seed",
                "7"
            ]
        );
        let plain = c
            .command(&c.arms[1], Path::new("/bin/fv"), &paths())
            .unwrap();
        assert_eq!(plain[0], "/bin/fv");
    }

    fn bad(edit: impl Fn(&str) -> String, want: &str) {
        let text = edit(CELL);
        let e = CellFile::parse(&text, "c1.toml").unwrap_err().to_string();
        assert!(e.contains(want), "{e} lacks {want}");
    }

    #[test]
    fn rejects_what_would_not_run_as_declared() {
        bad(
            |s| s.replace("seed = 7\n", "seed = 7\nseconds = 5\n"),
            "workload.seconds",
        );
        bad(
            |s| s.replace("\"{prompt}\"", "\"{frames}\""),
            "workload.frames is not set",
        );
        bad(
            |s| s.replace("{runs}/{arm}", "{run}/{arm}"),
            "unknown placeholder `{run}`",
        );
        bad(
            |s| s.replacen("trees = [\"h3-base\"]", "trees = [\"ltx25\"]", 1),
            "list `h3-base` in trees",
        );
        bad(
            |s| s.replace("candidate = \"a2\"", "candidate = \"zz\""),
            "not an arm",
        );
        bad(|s| s.replace("name = \"a2\"", "name = \"a1\""), "twice");
        bad(|s| s.replace("FASTVIDEO_X=", "lower="), "not K=V");
        bad(
            |s| s.replace("[lists]\n", "[lists]\nextra = [\"x\"]\n"),
            "no arm splices",
        );
        bad(
            |s| s.replace("schema = 1", "schema = 2"),
            "unsupported schema 2",
        );
        bad(|s| s.replace("description", "descr"), "TOML");
    }

    const TIERS: &str = r#"
schema = 1
[[tier]]
name = "t1"
description = "d"
provider = "runpod-pod"
gpu = "G"
gpu_count = 1
region = "eu"
image = "runtime"
max_usd_per_hr = 2.5
wall_cap_s = 3600
cell_timeout_s = 600
container_disk_gb = 120
weights = "volume"
cells = ["c1"]
"#;

    #[test]
    fn tiers_hold_the_guards_bounds() {
        let t = Tiers::parse(TIERS, "tiers.toml").unwrap();
        assert_eq!(t.tiers[0].cloud, CloudType::Secure);
        assert_eq!(
            t.tiers[0].image.image("abcdef1"),
            format!("{RUNTIME_IMAGE_REPO}:sha-abcdef1")
        );
        for (from, to, want) in [
            ("region = \"eu\"", "region = \"us\"", "only `eu`"),
            ("wall_cap_s = 3600", "wall_cap_s = 20000", "wall_cap_s"),
            ("gpu_count = 1", "gpu_count = 2", "one GPU"),
            ("weights = \"volume\"", "weights = \"hub\"", "`volume` only"),
            ("max_usd_per_hr = 2.5", "max_usd_per_hr = 0.0", "positive"),
            ("name = \"t1\"", "name = \"T1\"", "not [a-z0-9]"),
        ] {
            let e = Tiers::parse(&TIERS.replace(from, to), "tiers.toml")
                .unwrap_err()
                .to_string();
            assert!(e.contains(want), "{e} lacks {want}");
        }
    }

    #[test]
    fn placeholder_scanner() {
        assert_eq!(placeholders("a{b}c{d}").unwrap(), ["b", "d"]);
        assert!(placeholders("a{b").is_err());
        assert!(placeholders("a}b").is_err());
        assert!(is_name("rtx6000") && !is_name("-x") && !is_name(""));
    }
}
