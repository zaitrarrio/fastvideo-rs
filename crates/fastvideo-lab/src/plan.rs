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

/// Where a tier's arms read their weight trees (docs/adr/0019-fv-lab.md §4, §16).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub enum TierWeights {
    /// The EU volume, mounted at `/workspace`; `{weights}` is
    /// `/workspace/weights`, and a tree it lacks skips its arm.
    #[default]
    #[serde(rename = "volume")]
    Volume,
    /// The EU volume for the trees it holds, the R2 weights store for the
    /// rest: fetched onto the container disk before the arm that reads them;
    /// `{weights}` is `<scratch>/weights`, the volume's trees linked in.
    #[serde(rename = "volume+r2")]
    VolumeR2,
}

impl TierWeights {
    pub fn as_str(self) -> &'static str {
        match self {
            TierWeights::Volume => "volume",
            TierWeights::VolumeR2 => "volume+r2",
        }
    }

    /// The pod's own weights source: the volume is mounted either way (the
    /// R2 trees are the runner's, not fv-boot's).
    pub fn launch_source(self) -> WeightsSource {
        WeightsSource::Volume
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
    /// mounted at `/workspace`, trees under `/workspace/weights`) or
    /// `volume+r2` (the volume's trees, and the rest fetched from the R2
    /// weights store per arm, ADR-0019 §16); the registry's `hydrate` later
    /// (§4).
    pub weights: TierWeights,
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
    /// The cell's verdict (D1d): evidence read after each of its arms and a
    /// pass / fail file after its compares ([`Verdict`]).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub verdict: Option<Verdict>,
    /// The upstream reference dump the cell's arms read as `{ref}` (D1d §15,
    /// the `oracle` family): fetched before the cell's first arm
    /// ([`Reference`]).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reference: Option<Reference>,
    /// Prompt-set files written into the run directory before the cell's
    /// first arm (D1d §15, the `determinism` family's `det_set`).
    #[serde(default, rename = "prompt_file", skip_serializing_if = "Vec::is_empty")]
    pub prompt_files: Vec<PromptFile>,
    /// Frame-hash equality across the cell's repeated arms (D1d §15, the
    /// `det-short` / `determinism` families' hashes and `determinism.txt`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub repeats: Option<Repeats>,
}

/// The reference of an oracle cell (docs/adr/0019-fv-lab.md §15): what an
/// upstream pod made for `oracle-<target>` (`scripts/gpu/upstream/oracle.sh`),
/// served under `FV_ORACLE_URL` as `oracle-<target>/oracle-dump.tar` once
/// `oracle-<target>/ORACLE_DONE` exists. The runner fetches it (in the
/// background from the start of the run, as the script's prefetch) into
/// `<scratch>/oracle-ref/<target>/`, the arms read `{ref}`
/// (`<scratch>/oracle-ref/<target>/dump`). Unavailable (no URL, not finished
/// within `FV_ORACLE_WAIT_S`, a failed download): the cell's first arm is
/// recorded as skipped (`reference dump unavailable`) and nothing else of the
/// cell runs. After the cell, the reference and every arm's `fresh` path are
/// removed (the dumps are hundreds of MB).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Reference {
    /// The upstream target (`fasth3-4step-vsa`): `oracle-<target>` there.
    pub target: String,
}

/// A prompt set of one prompt repeated under several names, written to
/// `{runs}/<file>` before the cell's first arm, byte for byte as the
/// script's `det_set` writes it:
/// `{"prompts": [{"name": "<n>", "seed": <seed>, "prompt": "<prompt>"},…]}`,
/// the prompt and the seed the cell's workload's.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct PromptFile {
    /// The file name (`t2v-ab.json`), in the run directory.
    pub file: String,
    /// The prompt names, in order (`run1`, `run2`): each a run directory
    /// `frames/<name>/` of the arm that reads the set.
    pub names: Vec<String>,
}

/// How [`Repeats`] hashes an arm's frames.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "kebab-case")]
pub enum RepeatHash {
    /// One hash per arm (`det-short`'s `sh_hash`): every `frame-*.png` under
    /// the arm's directory but `*/warmup/*`, in path order, concatenated,
    /// and every `audio.wav` the same way, into `<arm>/hashes.json` and a
    /// line of `<runs>/hashes.txt`.
    Arm,
    /// One hash per run directory `frames/<run>/` but `warmup`
    /// (`determinism`'s `det_hash`, a prompt set's runs): its `frame-*.png`
    /// in name order and its `audio.wav`, into `<arm>/hashes.txt`; the
    /// stage digests of `stderr.log` into `<arm>/digests.txt`.
    Run,
}

/// Frame-hash equality across repeats (docs/adr/0019-fv-lab.md §15): after
/// each arm of the cell its frames are hashed ([`RepeatHash`]) and all but
/// `keep` deleted; after every cell, `<runs>/determinism.txt` reports each
/// hash against the reference's (`IDENTICAL` / `DIFFERENT`), as the script
/// writes it. The verdict (fv-lab's; the script only reports): the cell
/// passes when the reference has frames and every hash of the `judged` arms
/// equals it; `controls` are reported, not judged (the timed-pick control,
/// which may differ by design); a reference arm that did not run (skipped)
/// leaves the cell undecided.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Repeats {
    pub hash: RepeatHash,
    /// The hash every other is compared with: an arm (`hash = "arm"`), or
    /// `<arm>/<run>` (`hash = "run"`: `ltx-t2v-ab/run1`).
    pub reference: String,
    /// The arms whose every hash must equal the reference's; the reference
    /// arm is one of them.
    pub judged: Vec<String>,
    /// The arms reported but not judged.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub controls: Vec<String>,
    /// The frame files kept after hashing (`frame-000.png`); every other
    /// `frame-*.png` is deleted, as the script does (the hashes and
    /// `frames.sha256.json` are written before).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub keep: Vec<String>,
}

impl Repeats {
    /// The reference arm.
    pub fn reference_arm(&self) -> &str {
        self.reference
            .split_once('/')
            .map_or(self.reference.as_str(), |(a, _)| a)
    }

    /// The reference run (`hash = "run"`).
    pub fn reference_run(&self) -> Option<&str> {
        self.reference.split_once('/').map(|(_, r)| r)
    }
}

/// A cell's verdict (docs/adr/0019-fv-lab.md §11, D1d): what the script's
/// byte-identity families (`b4a`) do besides arms and compares, as data.
///
/// The verdict's arms start from an empty directory (`<runs>/<arm>` is
/// removed before each, so no earlier run's output stands in for this one's).
/// After each of them, in order: its `summary.json` must record exit 0 (else
/// `<label>:exit`); its `stderr.log` must exist (else `<label>:no-run`, and
/// nothing more is read for it); each [`Evidence`] file is written; each
/// [`LineCount`] is logged and checked. After the cell's compares: each
/// [`Identical`] pair, then each evidence `diff`. The verdict is
/// `<runs>/<family>-<cell>.verdict.json`,
/// `{"cell":"<cell>","verdict":"pass|fail","why":"<reasons>"}`; and every run
/// with verdicts ends with `<runs>/<family>.verdict.json` over the cells it
/// ran (`runpod-matrix.sh`'s `b4_family_verdict`). A verdict passes only on
/// positive evidence: a missing file fails it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Verdict {
    /// The verdict files' prefix and the log's (`b4a`: `b4a-<cell>.verdict.json`,
    /// `b4a.verdict.json`); the name of the tier listing the cell.
    pub family: String,
    /// The arms the verdict reads, in order, each named `<cell>-<label>`; the
    /// label names the arm in the reasons (`base:exit`).
    pub arms: Vec<String>,
    /// The dry run's line for the verdict: `dry-run <cell>: <describe>`.
    pub describe: String,
    /// Files of matching `stderr.log` lines, written into each arm's directory.
    #[serde(default, rename = "evidence", skip_serializing_if = "Vec::is_empty")]
    pub evidence: Vec<Evidence>,
    /// Matching `stderr.log` lines counted per arm, with a minimum.
    #[serde(default, rename = "count", skip_serializing_if = "Vec::is_empty")]
    pub counts: Vec<LineCount>,
    /// `--off-identity` compares of this cell that must have passed.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub identical: Vec<Identical>,
}

/// How an evidence line is cut before it is written.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "kebab-case")]
pub enum LineStrip {
    /// As logged.
    #[default]
    None,
    /// Without the leading `[…]` groups (and the spaces after each) a log line
    /// starts with: its timestamp, level and target (`sed -E
    /// 's/^(\[[^]]*\] *)+//'`), so two runs' lines compare.
    BracketPrefixes,
}

/// One evidence file per arm: the `stderr.log` lines that match any pattern.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Evidence {
    /// The file, in each arm's directory (`cache-evidence.log`); the name ends
    /// in `-evidence.log`, which `fv-lab pod fetch` takes.
    pub file: String,
    /// Line patterns: a line matches when it contains the pattern, `*` standing
    /// for any run of characters (`h3 teacache step *REUSE`).
    pub lines: Vec<String>,
    /// How each kept line is cut (default `none`).
    #[serde(default, skip_serializing_if = "is_default_strip")]
    pub strip: LineStrip,
    /// The two arms' files must be equal: `diff <first> <second>` is written to
    /// `<runs>/<cell>.<diff>` (`evidence.diff`), and a difference fails the
    /// verdict (`evidence-differs`). Needs exactly two arms.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub diff: Option<String>,
}

fn is_default_strip(s: &LineStrip) -> bool {
    *s == LineStrip::None
}

/// A count of matching `stderr.log` lines per arm (`grep -c`), logged as
/// `[<arm>] <what>: <n>`; under `min` fails the verdict (`<label>:<why>`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct LineCount {
    /// What is counted, as the log line names it.
    pub what: String,
    /// Line patterns, as [`Evidence::lines`].
    pub lines: Vec<String>,
    /// The fewest lines that pass.
    pub min: u64,
    /// The reason a smaller count adds, after the arm's label (`no-reuse`).
    pub why: String,
}

/// A compare of the cell (`[[compare]]` with `--off-identity`) whose report
/// must exist, not be a skip, and pass both identity checks
/// (`off_identity`, `off_identity_frame_count`); else `why`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Identical {
    /// The compare's baseline arm (one of the verdict's arms).
    pub baseline: String,
    /// The compare's candidate arm.
    pub candidate: String,
    /// The reason a failed or missing compare adds (default `not-identical`).
    #[serde(default = "not_identical")]
    pub why: String,
}

fn not_identical() -> String {
    "not-identical".into()
}

/// Whether `line` contains `pattern`, `*` matching any run of bytes (the
/// pieces between the stars in order, without overlap): grep's `a.*b` on one
/// line.
pub fn line_matches(line: &[u8], pattern: &str) -> bool {
    let mut rest = line;
    for piece in pattern.split('*').filter(|p| !p.is_empty()) {
        let p = piece.as_bytes();
        match rest.windows(p.len()).position(|w| w == p) {
            Some(i) => rest = &rest[i + p.len()..],
            None => return false,
        }
    }
    true
}

/// `sed -E 's/^(\[[^]]*\] *)+//'`: the line without its leading `[…]` groups.
pub fn strip_bracket_prefixes(mut line: &[u8]) -> &[u8] {
    while line.first() == Some(&b'[') {
        let Some(close) = line.iter().position(|&b| b == b']') else {
            break;
        };
        line = &line[close + 1..];
        while line.first() == Some(&b' ') {
            line = &line[1..];
        }
    }
    line
}

impl Evidence {
    /// The evidence of one `stderr.log`: `grep -h -e … | sed …`, every kept
    /// line ending in a newline.
    pub fn extract(&self, stderr: &[u8]) -> Vec<u8> {
        let mut out = Vec::new();
        for line in log_lines(stderr) {
            if self.lines.iter().any(|p| line_matches(line, p)) {
                let kept = match self.strip {
                    LineStrip::None => line,
                    LineStrip::BracketPrefixes => strip_bracket_prefixes(line),
                };
                out.extend_from_slice(kept);
                out.push(b'\n');
            }
        }
        out
    }
}

impl LineCount {
    /// `grep -c`: the matching lines of one `stderr.log`.
    pub fn count(&self, stderr: &[u8]) -> u64 {
        log_lines(stderr)
            .filter(|l| self.lines.iter().any(|p| line_matches(l, p)))
            .count() as u64
    }
}

/// The lines of a log as grep reads them: split on `\n`, no empty last line.
pub fn log_lines(text: &[u8]) -> impl Iterator<Item = &[u8]> {
    let body = (!text.is_empty()).then(|| text.strip_suffix(b"\n").unwrap_or(text));
    body.into_iter().flat_map(|b| b.split(|&c| c == b'\n'))
}

impl Verdict {
    /// The label of a verdict arm of `cell` (`b4a-x-base` → `base`).
    pub fn label<'a>(cell: &str, arm: &'a str) -> &'a str {
        arm.strip_prefix(cell)
            .and_then(|r| r.strip_prefix('-'))
            .unwrap_or(arm)
    }
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

/// Which fv-gpucheck an arm runs (docs/adr/0019-fv-lab.md, D1b).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "kebab-case")]
pub enum ArmBin {
    /// The image's own (`FV_GPUCHECK`, `/opt/fastvideo-rs/target/release/fv-gpucheck`).
    #[default]
    Gpucheck,
    /// The baseline commit's: `FV_BASELINE_SHA`'s runtime image binary,
    /// fetched by `scripts/gpu/fetch-baseline.sh` into
    /// `<scratch>/baseline-<sha>/fv-gpucheck` before the first arm, as the
    /// script's `techniques` / `b6` / `b4a` families do. The byte-identity
    /// families pair a `baseline` arm with a `gpucheck` arm and compare them
    /// with `--off-identity`.
    Baseline,
}

impl ArmBin {
    fn is_default(&self) -> bool {
        *self == ArmBin::Gpucheck
    }
}

/// One arm: one run directory `runs/<family>/<tag>/<name>/`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Arm {
    /// The run directory (unique within a tier).
    pub name: String,
    /// The `verify-weights.sh` cell that must pass before the arm runs (an
    /// incomplete tree is recorded as skipped, not run). Absent: an ungated
    /// arm (the script's bare `run_cell`, D1d §15: `compare-dumps`), which
    /// reads no weight tree.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gate: Option<String>,
    /// The weight trees the arm reads (weights-manifest.tsv `dest` names):
    /// the roots of `gate` and every `{weights}/<tree>` of the command
    /// (none for an ungated arm).
    #[serde(default)]
    pub trees: Vec<String>,
    /// Denoising steps when the recipe fixes them (descriptive: the recipe
    /// flag sets them, no `--steps` is passed).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub steps: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub note: Option<String>,
    /// The binary: `gpucheck` (default) or `baseline` ([`ArmBin`]).
    #[serde(default, skip_serializing_if = "ArmBin::is_default")]
    pub bin: ArmBin,
    /// A tiny-autoencoder file the arm reads as `{tae}/<file>` (the script's
    /// `tae_gated_cell`): fetched once with `fetch-tae.sh` when missing, and
    /// the arm recorded as skipped (`tae weights missing`) when still absent.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tae: Option<String>,
    /// `K=V` settings, in order, for `env K=V … fv-gpucheck …`. Absent: no
    /// `env`; `env = []`: a bare `env` before the binary, as the script's
    /// `env ${e[@]} "$bin"` renders an empty list (the command line equal
    /// word for word).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub env: Option<Vec<String>>,
    /// fv-gpucheck's arguments (the binary is the runner's).
    pub argv: Vec<String>,
    /// Paths under `{runs}/` removed before the arm (selected or not), as the
    /// script's `oracle_run` removes its dump directory; in a cell with a
    /// [`Reference`] also removed after the cell.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub fresh: Vec<String>,
    /// Directories that must all exist for the arm to run; otherwise it is
    /// passed over without a trace (the script's `oracle_diff`: no dumps,
    /// nothing to compare). A dry run lists the arm all the same.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub requires: Vec<String>,
    /// Line patterns (as [`Evidence::lines`]) of the arm's `stderr.log`
    /// copied into the log after it, each line as `[<arm>] <line>`
    /// (`oracle_diff`'s `compare-dumps` lines).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub echo: Vec<String>,
}

impl Arm {
    /// The arm's `K=V` settings (none when `env` is absent).
    pub fn env_entries(&self) -> &[String] {
        self.env.as_deref().unwrap_or(&[])
    }
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
    /// The promotion gate of the same pair, run right after this compare
    /// (the script's `compare_cells b c; gate_cells b c …` in one loop
    /// iteration). A `[[gate]]` runs after every compare of the cell instead.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gate: Option<PairedGate>,
}

/// The gate of a compare's own pair (`Compare::gate`).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct PairedGate {
    #[serde(default)]
    pub kind: GateKind,
    /// An OFF-arm whose compare with the baseline must be byte-identical.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub off: Option<String>,
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

/// `<scratch>/oracle-ref/<target>`: where a [`Reference`] is fetched (its
/// `dump/` is `{ref}`), as the script's `oracle` family.
pub fn reference_dir(p: &Paths, target: &str) -> PathBuf {
    p.scratch.join("oracle-ref").join(target)
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
    /// `{tae}`: the tiny-autoencoder directory (the script's `$TAE`: the
    /// volume's `auxiliary/tae` when complete, else `<scratch>/tae`).
    pub tae: PathBuf,
    /// The baseline binary `bin = "baseline"` arms run
    /// (`<scratch>/baseline-<sha>/fv-gpucheck`); `None` without a baseline
    /// commit, and such an arm does not render.
    pub baseline: Option<PathBuf>,
}

const PATH_KEYS: [&str; 7] = ["weights", "scratch", "runs", "root", "arm", "tae", "ref"];
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
            match &arm.gate {
                Some(g) if !is_token(g) => {
                    return Err(field(
                        file,
                        at("gate"),
                        format!("`{g}` is not a verify-weights.sh cell"),
                    ));
                }
                Some(_) if arm.trees.is_empty() => {
                    return Err(field(
                        file,
                        at("trees"),
                        "name the weight trees the arm reads",
                    ));
                }
                Some(_) => {}
                // An ungated arm reads no weights (no gate checks them).
                None if !arm.trees.is_empty() || arm.tae.is_some() => {
                    return Err(field(
                        file,
                        at("gate"),
                        "an arm that reads weight trees or a tiny autoencoder gates on them",
                    ));
                }
                None => {}
            }
            self.validate_arm_paths(arm, file, &at)?;
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
            for e in arm.env_entries() {
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
            if let Some(t) = &arm.tae {
                if !is_token(t) || t.contains('/') {
                    return Err(field(file, at("tae"), format!("`{t}` is not a file name")));
                }
                let want = format!("{{tae}}/{t}");
                if !arm
                    .env_entries()
                    .iter()
                    .chain(&argv)
                    .any(|s| s.contains(&want))
                {
                    return Err(field(
                        file,
                        at("tae"),
                        format!("the command does not read `{want}`"),
                    ));
                }
            }
            for s in arm.env_entries().iter().chain(&argv) {
                let ph = placeholders(s).map_err(|m| field(file, at("argv"), m))?;
                for p in ph {
                    if p.starts_with('@') {
                        return Err(field(
                            file,
                            at("argv"),
                            format!("`{s}`: a list splice is a whole element"),
                        ));
                    }
                    if p == "ref" && self.reference.is_none() {
                        return Err(field(
                            file,
                            at("argv"),
                            "`{ref}` in a cell without a [reference]",
                        ));
                    }
                    if p == "tae" && arm.tae.is_none() {
                        return Err(field(
                            file,
                            at("argv"),
                            "`{tae}` in an arm without `tae` (the file it gates on)",
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
            if arm.gate.is_none() && !referenced_trees.is_empty() {
                return Err(field(
                    file,
                    at("gate"),
                    "the command reads `{weights}/…`: gate the arm on its trees",
                ));
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
        if !self.prompt_files.is_empty() {
            // det_set writes the workload's prompt and seed into the file.
            for key in ["prompt", "seed"] {
                if self.workload.value(key).is_none() {
                    return Err(field(
                        file,
                        "prompt_file",
                        format!("a prompt file needs workload.{key}"),
                    ));
                }
                used_workload.insert(key.to_owned());
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
            let off = c.gate.as_ref().and_then(|g| g.off.as_ref());
            for n in [Some(&c.baseline), Some(&c.candidate), off]
                .into_iter()
                .flatten()
            {
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
        if let Some(v) = &self.verdict {
            self.validate_verdict(v, file, &names)?;
        }
        self.validate_d1d(file)?;
        Ok(())
    }

    /// An arm's `fresh`, `requires` and `echo` (D1d §15).
    fn validate_arm_paths(
        &self,
        arm: &Arm,
        file: &str,
        at: &dyn Fn(&str) -> String,
    ) -> Result<(), LabError> {
        let path_ok = |s: &str, roots: &[&str]| {
            roots
                .iter()
                .any(|r| s == *r || s.starts_with(&format!("{r}/")))
                && !s.split('/').any(|p| p == "..")
                && placeholders(s).is_ok_and(|ph| {
                    ph.iter()
                        .all(|p| matches!(*p, "runs" | "scratch" | "ref" | "arm"))
                })
        };
        for f in &arm.fresh {
            // Removed with rm -rf: only below the run directory, never the
            // run directory itself (`{runs}/`, `{runs}/.`, `{runs}//x/..`).
            let below = f.strip_prefix("{runs}/").is_some_and(|rest| {
                !rest.is_empty()
                    && rest
                        .split('/')
                        .all(|c| !c.is_empty() && c != "." && c != "..")
            });
            if !below || !path_ok(f, &["{runs}"]) {
                return Err(field(
                    file,
                    at("fresh"),
                    format!("`{f}` is not a path below `{{runs}}/`"),
                ));
            }
        }
        for r in &arm.requires {
            if !path_ok(r, &["{runs}", "{scratch}", "{ref}"]) {
                return Err(field(
                    file,
                    at("requires"),
                    format!("`{r}` is not a path under `{{runs}}`, `{{scratch}}` or `{{ref}}`"),
                ));
            }
        }
        for s in arm.fresh.iter().chain(&arm.requires) {
            if s.contains("{ref}") && self.reference.is_none() {
                return Err(field(
                    file,
                    at("requires"),
                    "`{ref}` in a cell without a [reference]",
                ));
            }
        }
        if let Some(p) = arm.echo.iter().find(|p| {
            p.trim().is_empty() || p.contains(['\n', '\r']) || p.trim_matches('*').is_empty()
        }) {
            return Err(field(
                file,
                at("echo"),
                format!("`{p}` is not a one-line pattern with text besides `*`"),
            ));
        }
        Ok(())
    }

    /// `[reference]`, `[[prompt_file]]` and `[repeats]` (D1d §15).
    fn validate_d1d(&self, file: &str) -> Result<(), LabError> {
        // A reference cell runs as its own step: a verdict on it would never
        // be evaluated.
        if self.reference.is_some() && self.verdict.is_some() {
            return Err(field(
                file,
                "reference",
                "a cell with a [reference] takes no [verdict]",
            ));
        }
        if let Some(r) = &self.reference {
            if !is_name(&r.target) {
                return Err(field(
                    file,
                    "reference.target",
                    format!("`{}` is not a target name", r.target),
                ));
            }
        }
        let mut files = BTreeSet::new();
        for (i, pf) in self.prompt_files.iter().enumerate() {
            let at = |f: &str| format!("prompt_file[{i}].{f}");
            if !is_name(&pf.file) || !pf.file.ends_with(".json") {
                return Err(field(
                    file,
                    at("file"),
                    format!("`{}` is not a file name ending in .json", pf.file),
                ));
            }
            if !files.insert(pf.file.as_str()) {
                return Err(field(file, at("file"), format!("`{}` twice", pf.file)));
            }
            if pf.names.is_empty() {
                return Err(field(file, at("names"), "no name"));
            }
            if let Some(n) = pf.names.iter().find(|n| !is_name(n)) {
                return Err(field(file, at("names"), format!("`{n}` is not a run name")));
            }
            let want = format!("{{runs}}/{}", pf.file);
            let read = self
                .arms
                .iter()
                .filter_map(|a| self.spliced(a).ok())
                .any(|argv| argv.contains(&want));
            if !read {
                return Err(field(file, at("file"), format!("no arm reads `{want}`")));
            }
        }
        if let Some(r) = &self.repeats {
            let arms: Vec<&str> = self.arms.iter().map(|a| a.name.as_str()).collect();
            let ref_arm = r.reference_arm();
            if !arms.contains(&ref_arm) {
                return Err(field(
                    file,
                    "repeats.reference",
                    format!("`{ref_arm}` is not an arm of this cell"),
                ));
            }
            match (r.hash, r.reference_run()) {
                (RepeatHash::Arm, Some(_)) => {
                    return Err(field(
                        file,
                        "repeats.reference",
                        "hash = \"arm\": an arm, not `<arm>/<run>`",
                    ));
                }
                (RepeatHash::Run, None) => {
                    return Err(field(
                        file,
                        "repeats.reference",
                        "hash = \"run\": `<arm>/<run>`",
                    ));
                }
                (RepeatHash::Run, Some(run)) if !is_name(run) || run == "warmup" => {
                    return Err(field(
                        file,
                        "repeats.reference",
                        format!("`{run}` is not a run name"),
                    ));
                }
                _ => {}
            }
            let mut seen = BTreeSet::new();
            for a in r.judged.iter().chain(&r.controls) {
                if !arms.contains(&a.as_str()) {
                    return Err(field(
                        file,
                        "repeats",
                        format!("`{a}` is not an arm of this cell"),
                    ));
                }
                if !seen.insert(a.as_str()) {
                    return Err(field(file, "repeats", format!("`{a}` twice")));
                }
            }
            if seen.len() != arms.len() {
                return Err(field(file, "repeats", "every arm is judged or a control"));
            }
            if !r.judged.iter().any(|a| a == ref_arm) {
                return Err(field(
                    file,
                    "repeats.judged",
                    format!("the reference arm `{ref_arm}` is judged"),
                ));
            }
            if let Some(k) = r
                .keep
                .iter()
                .find(|k| !(k.starts_with("frame-") && k.ends_with(".png") && is_name(k)))
            {
                return Err(field(
                    file,
                    "repeats.keep",
                    format!("`{k}` is not a frame-*.png name"),
                ));
            }
        }
        Ok(())
    }

    fn validate_verdict(
        &self,
        v: &Verdict,
        file: &str,
        names: &BTreeSet<&str>,
    ) -> Result<(), LabError> {
        let one_line = |s: &str| !s.trim().is_empty() && !s.contains(['\n', '\r']);
        // Written verbatim into a JSON string and a log line.
        let token = |s: &str| is_name(s) && !s.contains(['"', '\\']);
        let patterns = |at: &str, lines: &[String]| {
            if lines.is_empty() {
                return Err(field(file, at, "no pattern"));
            }
            match lines
                .iter()
                .find(|p| !one_line(p) || p.trim_matches('*').is_empty())
            {
                Some(p) => Err(field(
                    file,
                    at,
                    format!("`{p}` is not a one-line pattern with text besides `*`"),
                )),
                None => Ok(()),
            }
        };
        if !is_name(&v.family) {
            return Err(field(
                file,
                "verdict.family",
                format!("`{}` is not a tier name", v.family),
            ));
        }
        if !one_line(&v.describe) {
            return Err(field(file, "verdict.describe", "one line, not empty"));
        }
        if v.arms.is_empty() {
            return Err(field(file, "verdict.arms", "no arm"));
        }
        let mut seen = BTreeSet::new();
        for a in &v.arms {
            if !names.contains(a.as_str()) {
                return Err(field(
                    file,
                    "verdict.arms",
                    format!("`{a}` is not an arm of this cell"),
                ));
            }
            let label = Verdict::label(&self.name, a);
            if label == a || !token(label) {
                return Err(field(
                    file,
                    "verdict.arms",
                    format!("`{a}` is not `{}-<label>`", self.name),
                ));
            }
            if !seen.insert(a.as_str()) {
                return Err(field(file, "verdict.arms", format!("`{a}` twice")));
            }
        }
        let mut files = BTreeSet::new();
        let mut diffs = BTreeSet::new();
        for (i, e) in v.evidence.iter().enumerate() {
            let at = |f: &str| format!("verdict.evidence[{i}].{f}");
            if !token(&e.file) || !e.file.ends_with("-evidence.log") {
                return Err(field(
                    file,
                    at("file"),
                    format!("`{}` is not a file name ending in -evidence.log", e.file),
                ));
            }
            if !files.insert(e.file.as_str()) {
                return Err(field(file, at("file"), format!("`{}` twice", e.file)));
            }
            patterns(&at("lines"), &e.lines)?;
            if let Some(d) = &e.diff {
                if !token(d) || !d.ends_with(".diff") {
                    return Err(field(
                        file,
                        at("diff"),
                        format!("`{d}` is not a file name ending in .diff"),
                    ));
                }
                if v.arms.len() != 2 {
                    return Err(field(file, at("diff"), "a diff needs exactly two arms"));
                }
                // Two evidence files diffed into one file: the second
                // overwrites the first's, and a difference would go unseen.
                if !diffs.insert(d.as_str()) {
                    return Err(field(file, at("diff"), format!("`{d}` twice")));
                }
            }
        }
        for (i, c) in v.counts.iter().enumerate() {
            let at = |f: &str| format!("verdict.count[{i}].{f}");
            if !one_line(&c.what) {
                return Err(field(file, at("what"), "one line, not empty"));
            }
            patterns(&at("lines"), &c.lines)?;
            if !token(&c.why) {
                return Err(field(
                    file,
                    at("why"),
                    format!("`{}` is not a reason token", c.why),
                ));
            }
        }
        for (i, p) in v.identical.iter().enumerate() {
            let at = format!("verdict.identical[{i}]");
            for n in [&p.baseline, &p.candidate] {
                if !v.arms.contains(n) {
                    return Err(field(
                        file,
                        &at,
                        format!("`{n}` is not an arm of the verdict"),
                    ));
                }
            }
            if !token(&p.why) {
                return Err(field(
                    file,
                    &at,
                    format!("`{}` is not a reason token", p.why),
                ));
            }
            let compared = self.compares.iter().any(|c| {
                c.baseline == p.baseline
                    && c.candidate == p.candidate
                    && c.args.iter().any(|a| a == "--off-identity")
            });
            if !compared {
                return Err(field(
                    file,
                    &at,
                    format!(
                        "no [[compare]] of `{}` and `{}` with --off-identity",
                        p.baseline, p.candidate
                    ),
                ));
            }
        }
        Ok(())
    }

    /// A [`PromptFile`]'s bytes, as the script's `det_set` prints them: the
    /// workload's prompt with `\` and `"` escaped (its `sed`), its seed.
    pub fn prompt_file_text(&self, pf: &PromptFile) -> String {
        let prompt = self.workload.prompt.as_deref().unwrap_or("");
        let esc = prompt.replace('\\', "\\\\").replace('"', "\\\"");
        let seed = self
            .workload
            .seed
            .map(|s| s.to_string())
            .unwrap_or_default();
        let entries: Vec<String> = pf
            .names
            .iter()
            .map(|n| format!("{{\"name\": \"{n}\", \"seed\": {seed}, \"prompt\": \"{esc}\"}}"))
            .collect();
        format!("{{\"prompts\": [{}]}}\n", entries.join(","))
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
                "tae" => p.tae.display().to_string(),
                "ref" => match &self.reference {
                    Some(r) => reference_dir(p, &r.target)
                        .join("dump")
                        .display()
                        .to_string(),
                    None => return Err(format!("`{s}`: the cell has no [reference]")),
                },
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
    /// `run_cell`: `[env K=V …] <bin> <args…>`; `bin` is the image's
    /// fv-gpucheck, a `baseline` arm's is `p.baseline`.
    pub fn command(&self, arm: &Arm, bin: &Path, p: &Paths) -> Result<Vec<String>, String> {
        let mut cmd = Vec::new();
        if let Some(env) = &arm.env {
            cmd.push("env".to_owned());
            for e in env {
                cmd.push(self.render(e, &arm.name, p)?);
            }
        }
        let bin = match arm.bin {
            ArmBin::Gpucheck => bin,
            ArmBin::Baseline => p.baseline.as_deref().ok_or_else(|| {
                "a `bin = \"baseline\"` arm needs the baseline commit (FV_BASELINE_SHA, --baseline-sha)"
                    .to_owned()
            })?,
        };
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
    /// `gated_cell <name> <gate> <command…>`, or with `tae`
    /// `tae_gated_cell <name> <tae file> <gate> <command…>`.
    Arm {
        name: String,
        /// `None`: an ungated arm (`run_cell` without `gated_cell`).
        gate: Option<String>,
        trees: Vec<String>,
        command: Vec<String>,
        /// The tiny-autoencoder file the arm needs (`{tae}/<Arm::tae>`).
        tae: Option<PathBuf>,
        /// The arm runs the baseline binary (fetched before the first arm).
        baseline: bool,
        /// Removed before the arm ([`Arm::fresh`]).
        fresh: Vec<PathBuf>,
        /// Directories the arm needs, else it is passed over ([`Arm::requires`]).
        requires: Vec<PathBuf>,
        /// `stderr.log` line patterns echoed into the log after it ([`Arm::echo`]).
        echo: Vec<String>,
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
    /// A cell with a verdict (D1d): its own steps (arms, compares, gates), the
    /// verdict's evidence after each of its arms, then the verdict file. Run
    /// only when one of the verdict's arms is selected (`FV_CELLS`), as the
    /// script ran only the cells `FV_B4A_CELLS` named (its `b4a` branch,
    /// retired, ADR-0019 §11.6).
    Verdict {
        cell: String,
        verdict: Verdict,
        steps: Vec<Step>,
    },
    /// `<runs>/<family>.verdict.json` over the verdict cells of `family` that
    /// ran (each `(cell, its verdict arms)`), after every other step.
    FamilyVerdict {
        family: String,
        cells: Vec<VerdictCellArms>,
    },
    /// A cell with a [`Reference`] (D1d §15): the reference fetched into
    /// `dir` (`dir/dump` is `{ref}`), then the cell's steps, then `dir` and
    /// `remove` deleted. Run only when one of `arms` is selected.
    Reference {
        cell: String,
        target: String,
        dir: PathBuf,
        arms: Vec<String>,
        steps: Vec<Step>,
        remove: Vec<PathBuf>,
    },
    /// A [`PromptFile`], rendered: its bytes at `path`.
    PromptFile { path: PathBuf, text: String },
    /// After an arm of a [`Repeats`] cell: its frames hashed, all but `keep`
    /// deleted. Done only when one of `cell_arms` is selected (the script
    /// hashes every arm of a group it runs, an unselected one too).
    Hash {
        arm: String,
        hash: RepeatHash,
        keep: Vec<String>,
        cell_arms: Vec<String>,
    },
    /// After every cell: `<runs>/determinism.txt` over the [`Repeats`] cells
    /// that ran, in order, and their verdicts.
    RepeatsReport {
        hash: RepeatHash,
        cells: Vec<RepeatsCell>,
    },
}

/// One [`Repeats`] cell of a [`Step::RepeatsReport`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RepeatsCell {
    pub cell: String,
    /// The cell's arms, in order (the report's order).
    pub arms: Vec<String>,
    pub repeats: Repeats,
}

/// A verdict cell and its verdict arms (what decides whether it ran).
pub type VerdictCellArms = (String, Vec<String>);

impl Step {
    /// The steps with every [`Step::Verdict`]'s own steps in its place, the
    /// verdict steps themselves kept (run order).
    pub fn flatten(steps: &[Step]) -> Vec<&Step> {
        let mut out = Vec::new();
        for s in steps {
            out.push(s);
            if let Step::Verdict { steps, .. } | Step::Reference { steps, .. } = s {
                out.extend(Step::flatten(steps));
            }
        }
        out
    }
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
            let mut prompt_files = BTreeSet::new();
            let mut hashes = BTreeSet::new();
            let mut targets = BTreeSet::new();
            for c in t.cells.iter().filter_map(|c| self.cells.get(c)) {
                // Two cells of one target would fetch into one directory at
                // once, and the first to finish would remove it.
                if let Some(r) = c
                    .reference
                    .as_ref()
                    .filter(|r| !targets.insert(r.target.as_str()))
                {
                    return Err(field(
                        TIERS_PATH,
                        format!("tier `{}`.cells", t.name),
                        format!(
                            "reference target `{}` in two cells (one oracle-ref directory)",
                            r.target
                        ),
                    ));
                }
                for pf in &c.prompt_files {
                    if !prompt_files.insert(pf.file.as_str()) {
                        return Err(field(
                            TIERS_PATH,
                            format!("tier `{}`.cells", t.name),
                            format!("prompt file `{}` in two cells (one run directory)", pf.file),
                        ));
                    }
                }
                if let Some(r) = &c.repeats {
                    hashes.insert(format!("{:?}", r.hash));
                }
            }
            if hashes.len() > 1 {
                return Err(field(
                    TIERS_PATH,
                    format!("tier `{}`.cells", t.name),
                    "one determinism.txt per run: every [repeats] cell of a tier hashes the same way",
                ));
            }
            for c in &t.cells {
                let cell = self.cells.get(c).ok_or_else(|| {
                    field(
                        TIERS_PATH,
                        format!("tier `{}`.cells", t.name),
                        format!("no {CELLS_DIR}/{c}.toml"),
                    )
                })?;
                if let Some(v) = cell.verdict.as_ref().filter(|v| v.family != t.name) {
                    return Err(field(
                        &format!("{CELLS_DIR}/{c}.toml"),
                        "verdict.family",
                        format!("`{}`, but the cell is in tier `{}`", v.family, t.name),
                    ));
                }
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

    /// Whether any arm of `cells` runs the baseline binary (the run then
    /// needs `FV_BASELINE_SHA` and fetches it first).
    pub fn needs_baseline(cells: &[&CellFile]) -> bool {
        cells
            .iter()
            .flat_map(|c| &c.arms)
            .any(|a| a.bin == ArmBin::Baseline)
    }

    /// The steps of `cells`, rendered with `bin` and `paths`: per cell its
    /// arms, then its compares (each followed by its paired gate), then its
    /// gates; a cell with a verdict as one [`Step::Verdict`] holding those,
    /// and after all cells one [`Step::FamilyVerdict`] per verdict family
    /// (in the order first seen).
    pub fn steps(cells: &[&CellFile], bin: &Path, paths: &Paths) -> Result<Vec<Step>, LabError> {
        let mut out = Vec::new();
        let mut families: Vec<(String, Vec<VerdictCellArms>)> = Vec::new();
        let mut repeats: Vec<RepeatsCell> = Vec::new();
        for c in cells {
            let steps = Self::cell_steps(c, bin, paths)?;
            if let Some(r) = &c.repeats {
                repeats.push(RepeatsCell {
                    cell: c.name.clone(),
                    arms: c.arms.iter().map(|a| a.name.clone()).collect(),
                    repeats: r.clone(),
                });
            }
            if let Some(r) = &c.reference {
                let label = format!("{CELLS_DIR}/{}.toml", c.name);
                let mut remove = Vec::new();
                for a in &c.arms {
                    for f in &a.fresh {
                        remove.push(PathBuf::from(
                            c.render(f, &a.name, paths)
                                .map_err(|m| field(&label, format!("arm `{}`", a.name), m))?,
                        ));
                    }
                }
                out.push(Step::Reference {
                    cell: c.name.clone(),
                    target: r.target.clone(),
                    dir: reference_dir(paths, &r.target),
                    arms: c.arms.iter().map(|a| a.name.clone()).collect(),
                    steps,
                    remove,
                });
                continue;
            }
            match &c.verdict {
                None => out.extend(steps),
                Some(v) => {
                    let entry = (c.name.clone(), v.arms.clone());
                    match families.iter_mut().find(|(f, _)| *f == v.family) {
                        Some((_, cells)) => cells.push(entry),
                        None => families.push((v.family.clone(), vec![entry])),
                    }
                    out.push(Step::Verdict {
                        cell: c.name.clone(),
                        verdict: v.clone(),
                        steps,
                    });
                }
            }
        }
        if let Some(first) = repeats.first() {
            out.push(Step::RepeatsReport {
                hash: first.repeats.hash,
                cells: repeats,
            });
        }
        out.extend(
            families
                .into_iter()
                .map(|(family, cells)| Step::FamilyVerdict { family, cells }),
        );
        Ok(out)
    }

    fn cell_steps(c: &CellFile, bin: &Path, paths: &Paths) -> Result<Vec<Step>, LabError> {
        let mut out = Vec::new();
        let label = format!("{CELLS_DIR}/{}.toml", c.name);
        for pf in &c.prompt_files {
            out.push(Step::PromptFile {
                path: paths.runs.join(&pf.file),
                text: c.prompt_file_text(pf),
            });
        }
        for a in &c.arms {
            let err = |m: String| field(&label, format!("arm `{}`", a.name), m);
            let command = c.command(a, bin, paths).map_err(err)?;
            let render_all = |list: &[String]| -> Result<Vec<PathBuf>, LabError> {
                list.iter()
                    .map(|x| c.render(x, &a.name, paths).map(PathBuf::from).map_err(err))
                    .collect()
            };
            out.push(Step::Arm {
                name: a.name.clone(),
                gate: a.gate.clone(),
                trees: a.trees.clone(),
                command,
                tae: a.tae.as_ref().map(|t| paths.tae.join(t)),
                baseline: a.bin == ArmBin::Baseline,
                fresh: render_all(&a.fresh)?,
                requires: render_all(&a.requires)?,
                echo: a.echo.clone(),
            });
            if let Some(r) = &c.repeats {
                out.push(Step::Hash {
                    arm: a.name.clone(),
                    hash: r.hash,
                    keep: r.keep.clone(),
                    cell_arms: c.arms.iter().map(|a| a.name.clone()).collect(),
                });
            }
        }
        for cmp in &c.compares {
            out.push(Step::Compare {
                baseline: cmp.baseline.clone(),
                candidate: cmp.candidate.clone(),
                args: cmp.args.clone(),
            });
            if let Some(g) = &cmp.gate {
                out.push(Step::Gate {
                    baseline: cmp.baseline.clone(),
                    candidate: cmp.candidate.clone(),
                    kind: g.kind,
                    off: g.off.clone(),
                });
            }
        }
        for g in &c.gates {
            out.push(Step::Gate {
                baseline: g.baseline.clone(),
                candidate: g.candidate.clone(),
                kind: g.kind,
                off: g.off.clone(),
            });
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
env = ["FASTVIDEO_TEST_X={weights}/h3-base/x"]
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
            tae: "/fvscratch/tae".into(),
            baseline: None,
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
                "FASTVIDEO_TEST_X=/workspace/weights/h3-base/x",
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
        bad(|s| s.replace("FASTVIDEO_TEST_X=", "lower="), "not K=V");
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
        assert_eq!(t.tiers[0].weights, TierWeights::Volume);
        let r2 = Tiers::parse(
            &TIERS.replace("weights = \"volume\"", "weights = \"volume+r2\""),
            "tiers.toml",
        )
        .unwrap();
        assert_eq!(r2.tiers[0].weights, TierWeights::VolumeR2);
        assert_eq!(r2.tiers[0].weights.as_str(), "volume+r2");
        assert_eq!(r2.tiers[0].weights.launch_source(), WeightsSource::Volume);
        assert_eq!(t.tiers[0].cloud, CloudType::Secure);
        assert_eq!(
            t.tiers[0].image.image("abcdef1"),
            format!("{RUNTIME_IMAGE_REPO}:sha-abcdef1")
        );
        for (from, to, want) in [
            ("region = \"eu\"", "region = \"us\"", "only `eu`"),
            ("wall_cap_s = 3600", "wall_cap_s = 20000", "wall_cap_s"),
            ("gpu_count = 1", "gpu_count = 2", "one GPU"),
            (
                "weights = \"volume\"",
                "weights = \"hub\"",
                "unknown variant `hub`",
            ),
            (
                "weights = \"volume\"",
                "weights = \"r2\"",
                "unknown variant `r2`",
            ),
            ("max_usd_per_hr = 2.5", "max_usd_per_hr = 0.0", "positive"),
            ("name = \"t1\"", "name = \"T1\"", "not [a-z0-9]"),
        ] {
            let e = Tiers::parse(&TIERS.replace(from, to), "tiers.toml")
                .unwrap_err()
                .to_string();
            assert!(e.contains(want), "{e} lacks {want}");
        }
    }

    const PAIRS: &str = r#"
schema = 1
name = "p1"
description = "baseline pairs"

[workload]
model = "h3"
seed = 7

[[arm]]
name = "x-base"
gate = "h3-base"
trees = ["h3-base"]
bin = "baseline"
env = []
argv = ["h3", "gen", "--seed", "{seed}"]

[[arm]]
name = "x-pr"
gate = "h3-base"
trees = ["h3-base"]
env = []
argv = ["h3", "gen", "--seed", "{seed}"]

[[arm]]
name = "x-tae"
gate = "h3-base"
trees = ["h3-base"]
tae = "taeh3.safetensors"
argv = ["h3", "gen", "--taeh3-weights", "{tae}/taeh3.safetensors"]

[[compare]]
baseline = "x-base"
candidate = "x-pr"
args = ["--off-identity"]
gate = { kind = "exact", off = "x-pr" }

[[compare]]
baseline = "x-pr"
candidate = "x-tae"

[[gate]]
baseline = "x-pr"
candidate = "x-tae"
"#;

    #[test]
    fn baseline_arms_bare_env_tae_and_paired_gates() {
        let c = CellFile::parse(PAIRS, "p1.toml").unwrap();
        let mut p = paths();
        // Without the baseline commit a baseline arm does not render.
        let e = c.command(&c.arms[0], Path::new("/bin/fv"), &p).unwrap_err();
        assert!(e.contains("FV_BASELINE_SHA"), "{e}");
        p.baseline = Some("/fvscratch/baseline-7654321/fv-gpucheck".into());
        assert_eq!(
            c.command(&c.arms[0], Path::new("/bin/fv"), &p).unwrap(),
            [
                "env",
                "/fvscratch/baseline-7654321/fv-gpucheck",
                "h3",
                "gen",
                "--seed",
                "7"
            ]
        );
        assert_eq!(
            c.command(&c.arms[1], Path::new("/bin/fv"), &p).unwrap()[..2],
            ["env", "/bin/fv"]
        );
        assert_eq!(
            c.command(&c.arms[2], Path::new("/bin/fv"), &p).unwrap()[4],
            "/fvscratch/tae/taeh3.safetensors"
        );
        assert!(Lab::needs_baseline(&[&c]));
        let steps = Lab::steps(&[&c], Path::new("/bin/fv"), &p).unwrap();
        let shape: Vec<String> = steps
            .iter()
            .map(|s| match s {
                Step::Arm {
                    name,
                    tae,
                    baseline,
                    ..
                } => format!("arm {name} {baseline} {tae:?}"),
                Step::Compare {
                    baseline,
                    candidate,
                    ..
                } => format!("compare {baseline}--{candidate}"),
                Step::Gate {
                    baseline,
                    candidate,
                    kind,
                    off,
                } => format!("gate {baseline}--{candidate} {} {off:?}", kind.as_str()),
                Step::Verdict { cell, .. } => format!("verdict {cell}"),
                Step::FamilyVerdict { family, .. } => format!("family {family}"),
                other => format!("{other:?}"),
            })
            .collect();
        assert_eq!(
            shape,
            [
                "arm x-base true None",
                "arm x-pr false None",
                "arm x-tae false Some(\"/fvscratch/tae/taeh3.safetensors\")",
                "compare x-base--x-pr",
                "gate x-base--x-pr exact Some(\"x-pr\")",
                "compare x-pr--x-tae",
                "gate x-pr--x-tae lossy None",
            ]
        );
        let bad = |from: &str, to: &str, want: &str| {
            let e = CellFile::parse(&PAIRS.replacen(from, to, 1), "p1.toml")
                .unwrap_err()
                .to_string();
            assert!(e.contains(want), "{e} lacks {want}");
        };
        bad(
            "tae = \"taeh3.safetensors\"",
            "tae = \"taeh3x.safetensors\"",
            "does not read `{tae}/taeh3x.safetensors`",
        );
        bad("tae = \"taeh3.safetensors\"\n", "", "without `tae`");
        bad(
            "tae = \"taeh3.safetensors\"",
            "tae = \"a/b\"",
            "not a file name",
        );
        bad("off = \"x-pr\"", "off = \"zz\"", "not an arm");
        bad("bin = \"baseline\"", "bin = \"other\"", "TOML");
    }

    const VERDICT: &str = r#"
schema = 1
name = "v1"
description = "a verdict cell"

[workload]
model = "h3"
seed = 7

[[arm]]
name = "v1-base"
gate = "h3-base"
trees = ["h3-base"]
bin = "baseline"
argv = ["h3", "gen", "--seed", "{seed}"]

[[arm]]
name = "v1-pr"
gate = "h3-base"
trees = ["h3-base"]
argv = ["h3", "gen", "--seed", "{seed}"]

[[compare]]
baseline = "v1-base"
candidate = "v1-pr"
args = ["--off-identity"]

[verdict]
family = "fam"
arms = ["v1-base", "v1-pr"]
describe = "verdict also needs both arms exit 0"

[[verdict.evidence]]
file = "cache-evidence.log"
lines = ["h3 teacache step", "fuse step cache"]
strip = "bracket-prefixes"
diff = "evidence.diff"

[[verdict.count]]
what = "teacache reused forwards"
lines = ["h3 teacache step *REUSE"]
min = 1
why = "no-reuse"

[[verdict.identical]]
baseline = "v1-base"
candidate = "v1-pr"
"#;

    #[test]
    fn a_verdict_cell_renders_as_one_step_and_a_family_verdict_follows() {
        let c = CellFile::parse(VERDICT, "v1.toml").unwrap();
        let v = c.verdict.as_ref().unwrap();
        assert_eq!(v.identical[0].why, "not-identical");
        assert_eq!(v.evidence[0].strip, LineStrip::BracketPrefixes);
        let plain = CellFile::parse(CELL, "c1.toml").unwrap();
        let mut p = paths();
        p.baseline = Some("/b/fv-gpucheck".into());
        let steps = Lab::steps(&[&plain, &c], Path::new("/bin/fv"), &p).unwrap();
        // c1's three steps, the verdict cell, the family verdict.
        assert_eq!(steps.len(), 5);
        let Step::Verdict {
            cell, steps: inner, ..
        } = &steps[3]
        else {
            panic!("{:?}", steps[3]);
        };
        assert_eq!(cell, "v1");
        assert_eq!(inner.len(), 3, "two arms and the compare");
        assert_eq!(
            steps[4],
            Step::FamilyVerdict {
                family: "fam".into(),
                cells: vec![("v1".into(), vec!["v1-base".into(), "v1-pr".into()])],
            }
        );
        assert_eq!(Step::flatten(&steps).len(), 8);
        assert_eq!(Verdict::label("v1", "v1-base"), "base");
    }

    #[test]
    fn verdicts_name_their_arms_files_and_compares() {
        let bad = |from: &str, to: &str, want: &str| {
            let e = CellFile::parse(&VERDICT.replacen(from, to, 1), "v1.toml")
                .unwrap_err()
                .to_string();
            assert!(e.contains(want), "{from} -> {to}: {e} lacks {want}");
        };
        bad(
            "arms = [\"v1-base\", \"v1-pr\"]",
            "arms = [\"v1-base\", \"zz\"]",
            "not an arm of this cell",
        );
        let e = CellFile::parse(&VERDICT.replace("v1-pr", "w1-pr"), "v1.toml")
            .unwrap_err()
            .to_string();
        assert!(e.contains("is not `v1-<label>`"), "{e}");
        bad(
            "arms = [\"v1-base\", \"v1-pr\"]",
            "arms = [\"v1-base\"]",
            "exactly two arms",
        );
        bad(
            "file = \"cache-evidence.log\"",
            "file = \"cache.txt\"",
            "-evidence.log",
        );
        bad("diff = \"evidence.diff\"", "diff = \"evidence\"", ".diff");
        bad(
            "[[verdict.count]]",
            "[[verdict.evidence]]\nfile = \"other-evidence.log\"\nlines = [\"x\"]\ndiff = \"evidence.diff\"\n\n[[verdict.count]]",
            "`evidence.diff` twice",
        );
        bad(
            "lines = [\"h3 teacache step *REUSE\"]",
            "lines = [\"**\"]",
            "text besides",
        );
        bad("why = \"no-reuse\"", "why = \"no reuse\"", "reason token");
        bad(
            "args = [\"--off-identity\"]",
            "args = []",
            "with --off-identity",
        );
        bad(
            "describe = \"verdict also needs both arms exit 0\"",
            "describe = \"\"",
            "one line",
        );
        bad("family = \"fam\"", "family = \"Fam\"", "tier name");
        bad(
            "strip = \"bracket-prefixes\"",
            "strip = \"brackets\"",
            "TOML",
        );
        bad("min = 1", "min = 1\nmax = 2", "TOML");
    }

    #[test]
    fn evidence_lines_are_grep_and_sed() {
        let e = Evidence {
            file: "cache-evidence.log".into(),
            lines: vec!["h3 teacache step".into(), "fuse step cache".into()],
            strip: LineStrip::BracketPrefixes,
            diff: None,
        };
        let log = b"[INFO] [fastvideo] h3 teacache step 1: REUSE (x)\nother\n[a[b] ]  fuse step cache on\n[unclosed h3 teacache step 2: compute\nh3 teacache step 3: REUSE";
        assert_eq!(
            e.extract(log),
            b"h3 teacache step 1: REUSE (x)\n]  fuse step cache on\n[unclosed h3 teacache step 2: compute\nh3 teacache step 3: REUSE\n"
        );
        // As sed: `[a[b]` is one group, the `]` after it starts the text.
        assert_eq!(e.extract(b""), b"");
        let c = LineCount {
            what: "reused".into(),
            lines: vec!["h3 teacache step *REUSE".into()],
            min: 1,
            why: "no-reuse".into(),
        };
        assert_eq!(c.count(log), 2);
        assert_eq!(c.count(b"REUSE h3 teacache step 1\n"), 0, "in order");
        assert!(line_matches(b"abcabc", "c*a*c") && !line_matches(b"abc", "c*a"));
        assert_eq!(strip_bracket_prefixes(b"[x]   [y]z [w]"), b"z [w]");
    }

    const ORACLE: &str = r#"
schema = 1
name = "o1"
description = "an oracle cell"

[workload]
model = "h3"
prompt = "a \"b\" c\\d"
seed = 7

[reference]
target = "t1"

[[prompt_file]]
file = "p-ab.json"
names = ["run1", "run2"]

[[arm]]
name = "o1-main"
gate = "h3-base"
trees = ["h3-base"]
env = ["FASTVIDEO_INJECT_DIR={ref}", "FASTVIDEO_DUMP_DIR={runs}/o1-dump"]
argv = ["h3", "gen", "--prompt", "{prompt}", "--seed", "{seed}", "--prompts", "{runs}/p-ab.json"]
fresh = ["{runs}/o1-dump"]

[[arm]]
name = "o1-diff"
argv = ["compare-dumps", "--baseline", "{ref}", "--candidate", "{runs}/o1-dump"]
requires = ["{ref}", "{runs}/o1-dump"]
echo = ["compare-dumps"]

[repeats]
hash = "arm"
reference = "o1-main"
judged = ["o1-main"]
controls = ["o1-diff"]
keep = ["frame-000.png"]
"#;

    #[test]
    fn references_ungated_arms_prompt_files_and_repeats() {
        let c = CellFile::parse(ORACLE, "o1.toml").unwrap();
        let p = paths();
        let steps = Lab::steps(&[&c], Path::new("/bin/fv"), &p).unwrap();
        // The reference cell, then the repeats report.
        assert_eq!(steps.len(), 2, "{steps:?}");
        let Step::Reference {
            target,
            dir,
            arms,
            steps: inner,
            remove,
            ..
        } = &steps[0]
        else {
            panic!("{:?}", steps[0]);
        };
        assert_eq!(
            (target.as_str(), dir.as_path()),
            ("t1", Path::new("/fvscratch/oracle-ref/t1"))
        );
        assert_eq!(arms, &["o1-main", "o1-diff"]);
        assert_eq!(remove, &[PathBuf::from("/fvscratch/runs/f/t/o1-dump")]);
        let Step::PromptFile { path, text } = &inner[0] else {
            panic!("{:?}", inner[0]);
        };
        assert_eq!(path, Path::new("/fvscratch/runs/f/t/p-ab.json"));
        // det_set's bytes: `\` and `"` escaped, the workload's seed.
        assert_eq!(
            text,
            "{\"prompts\": [{\"name\": \"run1\", \"seed\": 7, \"prompt\": \"a \\\"b\\\" c\\\\d\"},{\"name\": \"run2\", \"seed\": 7, \"prompt\": \"a \\\"b\\\" c\\\\d\"}]}\n"
        );
        let Step::Arm { command, gate, .. } = &inner[3] else {
            panic!("{:?}", inner[3]);
        };
        assert_eq!(gate, &None);
        assert_eq!(
            command,
            &[
                "/bin/fv",
                "compare-dumps",
                "--baseline",
                "/fvscratch/oracle-ref/t1/dump",
                "--candidate",
                "/fvscratch/runs/f/t/o1-dump"
            ]
        );
        assert!(matches!(&inner[2], Step::Hash { arm, .. } if arm == "o1-main"));
        assert!(
            matches!(&steps[1], Step::RepeatsReport { hash: RepeatHash::Arm, cells } if cells.len() == 1)
        );
        let bad = |from: &str, to: &str, want: &str| {
            let e = CellFile::parse(&ORACLE.replacen(from, to, 1), "o1.toml")
                .unwrap_err()
                .to_string();
            assert!(e.contains(want), "{from} -> {to}: {e} lacks {want}");
        };
        bad(
            "[reference]\ntarget = \"t1\"\n",
            "",
            "without a [reference]",
        );
        bad(
            "fresh = [\"{runs}/o1-dump\"]",
            "fresh = [\"{scratch}/x\"]",
            "below `{runs}/`",
        );
        bad(
            "fresh = [\"{runs}/o1-dump\"]",
            "fresh = [\"{runs}/../x\"]",
            "below `{runs}/`",
        );
        for whole in [
            "{runs}/",
            "{runs}/.",
            "{runs}//x",
            "{runs}/x/./y",
            "{runs}/x/",
        ] {
            bad(
                "fresh = [\"{runs}/o1-dump\"]",
                &format!("fresh = [\"{whole}\"]"),
                "below `{runs}/`",
            );
        }
        bad(
            "[reference]\n",
            "[verdict]\nfamily = \"f\"\narms = [\"o1-main\"]\ndescribe = \"d\"\n\n[reference]\n",
            "takes no [verdict]",
        );
        bad(
            "requires = [\"{ref}\"",
            "requires = [\"/etc\"",
            "not a path under",
        );
        bad(
            "echo = [\"compare-dumps\"]",
            "echo = [\"*\"]",
            "one-line pattern",
        );
        bad(
            "argv = [\"compare-dumps\"",
            "trees = [\"h3-base\"]\nargv = [\"compare-dumps\"",
            "gates on them",
        );
        bad(
            "\"compare-dumps\", \"--baseline\"",
            "\"--weights\", \"{weights}/h3-base\", \"--baseline\"",
            "gate the arm on its trees",
        );
        bad("names = [\"run1\", \"run2\"]", "names = []", "no name");
        bad("file = \"p-ab.json\"", "file = \"p-ab.txt\"", ".json");
        bad("file = \"p-ab.json\"", "file = \"q.json\"", "no arm reads");
        bad("judged = [\"o1-main\"]", "judged = []", "is judged");
        bad(
            "controls = [\"o1-diff\"]\n",
            "",
            "every arm is judged or a control",
        );
        bad(
            "reference = \"o1-main\"",
            "reference = \"o1-main/run1\"",
            "an arm, not",
        );
        bad("hash = \"arm\"", "hash = \"run\"", "`<arm>/<run>`");
        bad(
            "keep = [\"frame-000.png\"]",
            "keep = [\"x.png\"]",
            "frame-*.png",
        );
        let e = CellFile::parse(
            &ORACLE
                .replacen("seed = 7\n", "", 1)
                .replacen("\"--seed\", \"{seed}\", ", "", 1),
            "o1.toml",
        )
        .unwrap_err()
        .to_string();
        assert!(e.contains("needs workload.seed"), "{e}");
    }

    #[test]
    fn one_reference_target_per_tier() {
        let cell = |name: &str, target: &str| {
            CellFile::parse(
                &ORACLE
                    .replace("o1", name)
                    .replace("p-ab.json", &format!("{name}.json"))
                    .replace("target = \"t1\"", &format!("target = \"{target}\"")),
                "x.toml",
            )
            .unwrap()
        };
        let lab = |t2: &str| Lab {
            root: PathBuf::new(),
            tiers: Tiers::parse(
                &TIERS.replace("cells = [\"c1\"]", "cells = [\"o1\", \"o2\"]"),
                "tiers.toml",
            )
            .unwrap(),
            cells: [
                ("o1".into(), cell("o1", "t1")),
                ("o2".into(), cell("o2", t2)),
            ]
            .into(),
        };
        let e = lab("t1").validate().unwrap_err().to_string();
        assert!(e.contains("reference target `t1` in two cells"), "{e}");
        assert!(lab("t2").validate().is_ok());
    }

    #[test]
    fn placeholder_scanner() {
        assert_eq!(placeholders("a{b}c{d}").unwrap(), ["b", "d"]);
        assert!(placeholders("a{b").is_err());
        assert!(placeholders("a}b").is_err());
        assert!(is_name("rtx6000") && !is_name("-x") && !is_name(""));
    }
}
