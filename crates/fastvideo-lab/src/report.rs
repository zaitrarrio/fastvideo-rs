//! `fv-lab report tiers`: `docs/perf/tiers.md`, the gold variants as one run
//! set measured them (plan D3, docs/adr/0023-lab-report-and-ci-bridge.md §1).
//!
//! The input is one run set: a directory `artifacts/runpod/<tier>/<tag>/` as
//! `fv-lab pod fetch` (or `runpod-http.sh`) writes it, one arm directory per
//! run directory (`summary.json`, and `benchmark.json` for a CLI generation or
//! `gpucheck-out/engine.json` for a serve-engine job), plus `pod.json`; or
//! several such directories of one tier and one commit, rendered as one run
//! set ([`Runs`]: a `gold` run split in two with `--cells`, ADR-0019 §16.7). A
//! gold variant is measured by the arm of the same name, and only when that
//! arm is its gold cell: the run set is the `gold` tier's (which names its arms
//! by variant id, `crate::goldcell`), or the arm's engine report names the
//! variant as the model it ran (`context.model.id`). A CLI arm of another tier
//! that happens to share the name is not the variant. Everything the table
//! shows is read from those files or the gold files: a variant without an arm,
//! with a skipped or failed arm, or with no timing record is "not measured",
//! and no number is derived but `$/clip` (the run's own `usd_per_hr` times its
//! own wall time).
//!
//! The rendering is deterministic: the same gold files and run set give the
//! same bytes, and `--check` compares them with the committed file, whose
//! header names the run set it was rendered from.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use serde_json::Value;

use crate::gold::Gold;
use crate::goldcell::GOLD_TIER;
use crate::goldcheck::day_number;

/// The committed table.
pub const TIERS_MD: &str = "docs/perf/tiers.md";

/// The header line that names the run set (`--check` reads it back).
const RUN_SET_PREFIX: &str = "Run set: ";
const NO_RUN_SET: &str = "none";
/// Between the run directories of a run set of several runs.
const RUN_SET_SEP: &str = ", ";

/// What an arm's timing record says, as read.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Metrics {
    /// The file it came from (`benchmark.json`, `gpucheck-out/engine.json`).
    pub source: String,
    /// What the wall time is: `gen` (one warm generate(): text, denoise,
    /// decode, MP4) or `job` (a serve-engine job, submit to MP4).
    pub measures: String,
    pub gpu: Option<String>,
    pub width: Option<u64>,
    pub height: Option<u64>,
    pub frames: Option<u64>,
    pub steps: Option<u64>,
    pub wall_s: Option<f64>,
    pub text_s: Option<f64>,
    pub denoise_s: Option<f64>,
    pub decode_s: Option<f64>,
    pub load_s: Option<f64>,
    pub peak_gib: Option<f64>,
    /// The catalog model an engine report ran (`context.model.id`).
    pub model_id: Option<String>,
}

/// One arm directory of a run set.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct ArmRecord {
    pub exit: Option<i64>,
    pub skipped: Option<String>,
    /// `ended` of `summary.json` (`YYYY-MM-DDTHH:MM:SSZ`).
    pub ended: Option<String>,
    pub metrics: Option<Metrics>,
}

/// One run set.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct RunSet {
    /// The directory as given (relative to the repository root).
    pub label: String,
    pub tier: String,
    pub tag: String,
    /// `pod.json`: the GPU the provider rented, its price and the commit.
    pub pod_gpu: Option<String>,
    pub usd_per_hr: Option<f64>,
    pub sha: Option<String>,
    pub arms: BTreeMap<String, ArmRecord>,
}

fn read_json(p: &Path) -> Result<Option<Value>, String> {
    match std::fs::read_to_string(p) {
        Ok(t) => serde_json::from_str(&t)
            .map(Some)
            .map_err(|e| format!("{}: {e}", p.display())),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(format!("{}: {e}", p.display())),
    }
}

fn f(v: &Value, ptr: &str) -> Option<f64> {
    v.pointer(ptr).and_then(Value::as_f64)
}

fn u(v: &Value, ptr: &str) -> Option<u64> {
    v.pointer(ptr).and_then(Value::as_u64)
}

/// A `benchmark.json` (`fastvideo-rs.benchmark.v1`).
fn benchmark_metrics(b: &Value) -> Metrics {
    let peak_mib = f(b, "/max_device_memory_used_mib").or_else(|| f(b, "/peak_memory_mb"));
    Metrics {
        source: "benchmark.json".into(),
        measures: "gen".into(),
        gpu: b.get("gpu").and_then(Value::as_str).map(str::to_owned),
        width: u(b, "/workload/width"),
        height: u(b, "/workload/height"),
        frames: u(b, "/workload/num_frames"),
        steps: u(b, "/workload/measured_steps"),
        wall_s: f(b, "/total_s"),
        text_s: f(b, "/text_s"),
        denoise_s: f(b, "/denoise_s"),
        decode_s: f(b, "/decode_s"),
        load_s: f(b, "/load_s"),
        peak_gib: peak_mib.map(|m| m / 1024.0),
        model_id: None,
    }
}

/// `gpucheck-out/engine.json` (`fv-gpucheck engine`'s report).
fn engine_metrics(r: &Value) -> Metrics {
    let check = |name: &str| -> Option<&Value> {
        r.get("checks")?
            .as_array()?
            .iter()
            .find(|c| c.get("name").and_then(Value::as_str) == Some(name))
    };
    let timings = check("timings").and_then(|c| c.get("values"));
    let t = |ptr: &str| timings.and_then(|v| f(v, ptr));
    let mp4 = check("mp4").and_then(|c| c.get("values"));
    let job = r.pointer("/context/job");
    Metrics {
        source: "gpucheck-out/engine.json".into(),
        measures: "job".into(),
        gpu: None,
        width: mp4
            .and_then(|v| u(v, "/width"))
            .or_else(|| job.and_then(|j| u(j, "/width"))),
        height: mp4
            .and_then(|v| u(v, "/height"))
            .or_else(|| job.and_then(|j| u(j, "/height"))),
        frames: job.and_then(|j| u(j, "/num_frames")),
        steps: check("progress").and_then(|c| u(c, "/limits/steps")),
        wall_s: t("/submit_to_mp4_s"),
        text_s: t("/metrics/stage_durations/text"),
        denoise_s: t("/metrics/stage_durations/denoise"),
        // The decode through the finished MP4, as benchmark.json's `decode_s`
        // (the encode included): the engine reports its encode on its own.
        decode_s: t("/metrics/stage_durations/video_decode")
            .map(|d| d + t("/metrics/stage_durations/encode").unwrap_or(0.0)),
        load_s: t("/load_s"),
        peak_gib: t("/metrics/peak_memory_mb").map(|m| m / 1024.0),
        model_id: r
            .pointer("/context/model/id")
            .and_then(Value::as_str)
            .map(str::to_owned),
    }
}

impl RunSet {
    /// Reads `<root>/<label>`: every arm directory with a `summary.json`.
    pub fn load(root: &Path, label: &str) -> Result<RunSet, String> {
        let dir = root.join(label);
        let label = label.trim_end_matches('/').to_owned();
        let mut parts = label.rsplit('/');
        let tag = parts.next().unwrap_or_default().to_owned();
        let tier = parts.next().unwrap_or_default().to_owned();
        if tag.is_empty() || tier.is_empty() {
            return Err(format!("{label}: expected <dir>/<tier>/<tag>"));
        }
        let mut rs = RunSet {
            label,
            tier,
            tag,
            ..RunSet::default()
        };
        if let Some(p) = read_json(&dir.join("pod.json"))? {
            rs.pod_gpu = p.get("gpu").and_then(Value::as_str).map(str::to_owned);
            rs.usd_per_hr = p.get("usd_per_hr").and_then(Value::as_f64);
            rs.sha = p.get("sha").and_then(Value::as_str).map(str::to_owned);
        }
        let entries = std::fs::read_dir(&dir).map_err(|e| format!("{}: {e}", dir.display()))?;
        for e in entries {
            let e = e.map_err(|e| format!("{}: {e}", dir.display()))?;
            let arm_dir = e.path();
            if !arm_dir.is_dir() {
                continue;
            }
            let Some(summary) = read_json(&arm_dir.join("summary.json"))? else {
                continue;
            };
            let name = e.file_name().to_string_lossy().into_owned();
            let metrics = match read_json(&arm_dir.join("benchmark.json"))? {
                Some(b) => Some(benchmark_metrics(&b)),
                None => read_json(&arm_dir.join("gpucheck-out").join("engine.json"))?
                    .map(|r| engine_metrics(&r)),
            };
            rs.arms.insert(
                name,
                ArmRecord {
                    exit: summary.get("exit").and_then(Value::as_i64),
                    skipped: summary
                        .get("skipped")
                        .and_then(Value::as_str)
                        .map(str::to_owned),
                    ended: summary
                        .get("ended")
                        .and_then(Value::as_str)
                        .map(str::to_owned),
                    metrics,
                },
            );
        }
        Ok(rs)
    }

    /// The commit the run set ran: `pod.json`'s, else the tag's first field
    /// (fv-lab and runpod-http.sh tags start with the 7-digit commit).
    pub fn commit(&self) -> Option<String> {
        self.sha.clone().or_else(|| {
            let head = self.tag.split('-').next()?;
            (head.len() == 7 && head.bytes().all(|b| b.is_ascii_hexdigit()))
                .then(|| head.to_owned())
        })
    }

    /// The last day an arm ended (`YYYY-MM-DD`), the day validation ages are
    /// counted to.
    pub fn date(&self) -> Option<String> {
        self.arms
            .values()
            .filter_map(|a| a.ended.as_deref())
            .filter_map(|e| e.get(..10))
            .max()
            .map(str::to_owned)
    }
}

/// One or more run sets of one tier, rendered as one (`--runs a,b`):
/// a `gold` run split in two with `--cells` (ADR-0019 §16.7). Built only by
/// [`Runs::new`], which refuses what would not be one run set:
///
/// - runs of different tiers;
/// - with several runs, a run whose commit is not known, or runs of
///   different commits ([`known_commit`]: the `pod.json` sha, else the
///   tag's head, each only when it is at least 7 hex digits; compared
///   lowercased, a short sha and a longer one agree when one begins the
///   other);
/// - the same directory twice (paths compared by their components, so
///   `./a` and `a/` are `a`);
/// - an arm that passed (`exit` 0, not skipped) in more than one run.
///
/// An arm in several runs is shown from one copy ([`Runs::arm`]): the
/// copy that passed; else the last copy that ran and failed (no `skipped`,
/// `exit` not 0), in `--runs` order; else the last skipped copy. So a failed
/// arm rerun in a later run shows the rerun, and an arm skipped as `wall cap
/// near` and run in the second run shows the second run's.
#[derive(Debug, Clone, PartialEq)]
pub struct Runs {
    sets: Vec<RunSet>,
    /// Per arm name, the index in `sets` of the copy shown.
    shown: BTreeMap<String, usize>,
}

impl From<RunSet> for Runs {
    fn from(set: RunSet) -> Runs {
        let shown = set.arms.keys().map(|k| (k.clone(), 0)).collect();
        Runs {
            sets: vec![set],
            shown,
        }
    }
}

/// A copy's rank in choosing the one shown: passed, then ran and failed,
/// then skipped.
fn copy_rank(a: &ArmRecord) -> u8 {
    match (&a.skipped, a.exit) {
        (Some(_), _) => 0,
        (None, Some(0)) => 2,
        (None, _) => 1,
    }
}

/// What a copy that is not shown did, in words.
fn copy_words(a: &ArmRecord) -> String {
    match (&a.skipped, a.exit) {
        (Some(why), _) => format!("skipped here: {why}"),
        (None, Some(e)) => format!("failed here: exit {e}"),
        (None, None) => "no exit recorded here".to_owned(),
    }
}

/// The commit a run names, when it names one: its `pod.json` sha, else its
/// tag's first field ([`RunSet::commit`]), lowercased, and only when it is at
/// least 7 hex digits.
pub fn known_commit(r: &RunSet) -> Option<String> {
    let ok = |c: &String| c.len() >= 7 && c.bytes().all(|b| b.is_ascii_hexdigit());
    r.sha
        .as_deref()
        .map(str::to_ascii_lowercase)
        .filter(ok)
        .or_else(|| {
            let head = r.tag.split('-').next()?.to_ascii_lowercase();
            (head.len() == 7).then_some(head).filter(ok)
        })
}

/// Two known commits agree when one begins the other (a 7-digit tag field
/// and a 40-digit `pod.json` sha).
fn same_commit(a: &str, b: &str) -> bool {
    a.starts_with(b) || b.starts_with(a)
}

/// A run directory as given, compared by its components (`./a`, `a/` and
/// `a//` are `a`).
fn same_dir(a: &str, b: &str) -> bool {
    let norm = |p: &str| -> std::path::PathBuf {
        Path::new(p)
            .components()
            .filter(|c| !matches!(c, std::path::Component::CurDir))
            .collect()
    };
    norm(a) == norm(b)
}

impl Runs {
    /// The run sets as one, in the order given; see [`Runs`] for what is refused.
    pub fn new(sets: Vec<RunSet>) -> Result<Runs, String> {
        let Some(first) = sets.first() else {
            return Err("no run set given".to_owned());
        };
        for (i, r) in sets.iter().enumerate() {
            if r.tier != first.tier {
                return Err(format!(
                    "the runs are of different tiers: `{}` is `{}`, `{}` is `{}`; one table renders one tier's runs",
                    first.label, first.tier, r.label, r.tier
                ));
            }
            if let Some(o) = sets[..i].iter().find(|o| same_dir(&o.label, &r.label)) {
                return Err(format!(
                    "`{}` and `{}` are the same run directory",
                    o.label, r.label
                ));
            }
        }
        if sets.len() > 1 {
            let mut commits: Vec<(&str, String)> = Vec::new();
            for r in &sets {
                let c = known_commit(r).ok_or_else(|| {
                    format!(
                        "`{}`: its commit is not known (no `pod.json` sha and no commit heading its tag, of at least 7 hex digits); several runs are rendered as one only when each names its commit",
                        r.label
                    )
                })?;
                if let Some((l, o)) = commits.iter().find(|(_, o)| !same_commit(o, &c)) {
                    return Err(format!(
                        "the runs are of different commits: `{l}` ran `{o}`, `{}` ran `{c}`; one table renders one commit's runs",
                        r.label
                    ));
                }
                commits.push((r.label.as_str(), c));
            }
        }
        let mut passed: BTreeMap<&str, Vec<&str>> = BTreeMap::new();
        for r in &sets {
            for (name, a) in &r.arms {
                if copy_rank(a) == 2 {
                    passed
                        .entry(name.as_str())
                        .or_default()
                        .push(r.label.as_str());
                }
            }
        }
        if let Some((name, labels)) = passed.iter().find(|(_, l)| l.len() > 1) {
            return Err(format!(
                "arm `{name}` passed in more than one run ({}); one table shows one measurement per arm: leave one of those runs out",
                labels
                    .iter()
                    .map(|l| format!("`{l}`"))
                    .collect::<Vec<_>>()
                    .join(", ")
            ));
        }
        let mut shown: BTreeMap<String, usize> = BTreeMap::new();
        for (i, r) in sets.iter().enumerate() {
            for (name, a) in &r.arms {
                let keep = shown
                    .get(name)
                    .and_then(|&j| sets[j].arms.get(name))
                    .is_some_and(|b| copy_rank(b) > copy_rank(a));
                if !keep {
                    shown.insert(name.clone(), i);
                }
            }
        }
        Ok(Runs { sets, shown })
    }

    /// The run sets, in the order given.
    pub fn sets(&self) -> &[RunSet] {
        &self.sets
    }

    /// The tier every run is of.
    pub fn tier(&self) -> &str {
        self.sets.first().map_or("", |r| r.tier.as_str())
    }

    /// The arm of this name and the run it is from: the copy shown (see
    /// [`Runs`]).
    pub fn arm(&self, name: &str) -> Option<(&RunSet, &ArmRecord)> {
        let r = self.sets.get(*self.shown.get(name)?)?;
        Some((r, r.arms.get(name)?))
    }

    /// Whether some arm is in more than one run.
    pub fn has_duplicates(&self) -> bool {
        self.sets.iter().map(|r| r.arms.len()).sum::<usize>() > self.shown.len()
    }

    /// The arms of run `i` shown from it.
    pub fn shown_from(&self, i: usize) -> usize {
        self.shown.values().filter(|&&j| j == i).count()
    }

    /// The arms of run `i` not shown from it: each with the run it is shown
    /// from.
    pub fn not_shown_from(&self, i: usize) -> Vec<(&str, &ArmRecord, &RunSet)> {
        let Some(r) = self.sets.get(i) else {
            return Vec::new();
        };
        r.arms
            .iter()
            .filter_map(|(name, a)| {
                let j = *self.shown.get(name)?;
                (j != i).then(|| (name.as_str(), a, &self.sets[j]))
            })
            .collect()
    }

    /// Every arm name of every run.
    pub fn arm_names(&self) -> BTreeSet<&str> {
        self.shown.keys().map(String::as_str).collect()
    }

    /// The commit the runs ran: the longest one known ([`known_commit`]).
    pub fn commit(&self) -> Option<String> {
        self.sets
            .iter()
            .filter_map(known_commit)
            .max_by_key(String::len)
    }

    /// The last day an arm of any run ended.
    pub fn date(&self) -> Option<String> {
        self.sets.iter().filter_map(RunSet::date).max()
    }
}

/// Text for a Markdown table cell: `|` escaped, line breaks flattened.
fn cell_text(s: &str) -> String {
    s.replace('|', "\\|").replace(['\n', '\r'], " ")
}

fn opt(v: Option<f64>, digits: usize) -> String {
    v.map_or_else(|| "-".to_owned(), |x| format!("{x:.digits$}"))
}

fn cell_of(m: &Metrics) -> String {
    let n = |v: Option<u64>| v.map_or_else(|| "?".to_owned(), |x| x.to_string());
    let mut s = format!("{}x{}x{}", n(m.width), n(m.height), n(m.frames));
    if let Some(st) = m.steps {
        s.push_str(&format!(", {st} steps"));
    }
    s
}

/// The gold cell in the same words, and whether the run measured it.
fn vs_gold(m: &Metrics, g: &Gold) -> String {
    let w = &g.workload;
    let gold = format!(
        "{}x{}x{}",
        w.width,
        w.height,
        w.frames
            .map_or_else(|| "stream".to_owned(), |f| f.to_string())
    );
    let same = m.width == Some(u64::from(w.width))
        && m.height == Some(u64::from(w.height))
        && (w.frames.is_none() || m.frames == w.frames.map(u64::from))
        && (w.steps.is_none() || m.steps.is_none() || m.steps == w.steps.map(u64::from));
    if same {
        "same".to_owned()
    } else {
        format!("differs (gold {gold})")
    }
}

fn last_validated(g: &Gold, run_day: Option<i64>) -> String {
    let lv = &g.last_validated;
    match (&lv.date, &lv.commit) {
        (Some(d), Some(c)) => {
            let age = run_day
                .zip(day_number(d).ok())
                .map(|(r, v)| format!(", {} d before the run set", r - v))
                .unwrap_or_default();
            format!("{d} `{c}`{age}")
        }
        _ => "never".to_owned(),
    }
}

/// One run's pod: the GPU the provider rented and its price, in words.
fn pod_words(r: &RunSet) -> (String, String) {
    let price = r
        .usd_per_hr
        .map_or_else(|| "no price recorded".to_owned(), |p| format!("${p:.2}/hr"));
    let gpu = r
        .pod_gpu
        .clone()
        .unwrap_or_else(|| "not recorded".to_owned());
    (gpu, price)
}

/// Renders the table from the gold files (in order) and the run set. One run
/// renders as it always has; several ([`Runs`]) name every run in the
/// `Run set:` line and list each run's tag, end, GPU and price, and each row
/// reads its arm's own run (its `pod.json` GPU and price).
pub fn render(golds: &[Gold], runs: Option<&Runs>) -> String {
    let mut s = String::new();
    s.push_str("<!-- GENERATED by `fv-lab report tiers` (crates/fastvideo-lab/src/report.rs) from lab/gold/*.toml and the run set below. Do not edit: rerun it; `fv-lab report tiers --check` fails when this file differs. -->\n");
    s.push_str("# Tier table: the gold variants, measured by one run set\n\n");
    match runs {
        Some(r) => s.push_str(&format!(
            "{RUN_SET_PREFIX}{}\n\n",
            r.sets()
                .iter()
                .map(|x| format!("`{}`", x.label))
                .collect::<Vec<_>>()
                .join(RUN_SET_SEP)
        )),
        None => s.push_str(&format!("{RUN_SET_PREFIX}{NO_RUN_SET}\n\n")),
    }
    s.push_str(
        "One row per gold file (`lab/gold/<variant>.toml`, ADR-0009). A variant is measured by the arm of the same name in the run set (`fv-lab run gold --all` names its arms by variant id); every value is read from that arm's files (`summary.json`, `benchmark.json` or `gpucheck-out/engine.json`) and the run's `pod.json`, and `$/clip` is the run's own `usd_per_hr` times its own wall time. A variant without an arm, with a skipped or failed arm, or without a timing record is not measured: nothing is filled in from elsewhere. A variant counts as measured only by its gold cell: an arm of a `gold` run set, or an arm whose engine report ran that variant; a same-named arm of another tier is not. Wall time `gen` is one warm `generate()` (text, denoise, decode, MP4; load excluded), `job` a serve-engine job from submit to MP4; `decode s` is the decode through the finished MP4 in both (for `job`, the engine's `video_decode` plus its `encode`). `vs gold` compares the measured canvas, frames and steps with the gold file's `[workload]`. Design: docs/adr/0023-lab-report-and-ci-bridge.md.\n\n",
    );
    match runs {
        Some(Runs { sets, .. }) if sets.len() == 1 => {
            let r = &sets[0];
            let commit = r.commit().unwrap_or_else(|| "unknown".to_owned());
            let (gpu, price) = pod_words(r);
            let date = r.date().unwrap_or_else(|| "unknown".to_owned());
            s.push_str(&format!(
                "Tier `{}`, tag `{}`, commit `{commit}`, ended {date}; pod GPU {gpu}, {price}.\n\n",
                r.tier, r.tag
            ));
        }
        Some(all) => {
            let sets = all.sets();
            let commit = all.commit().unwrap_or_else(|| "unknown".to_owned());
            let date = all.date().unwrap_or_else(|| "unknown".to_owned());
            s.push_str(&format!(
                "Tier `{}`, commit `{commit}`, ended {date}: {} runs rendered as one run set (ADR-0019 §16.7).",
                all.tier(),
                sets.len()
            ));
            if all.has_duplicates() {
                s.push_str(" An arm in more than one run is shown from the copy that passed, else the last that failed, else the last skipped (in this order of runs); its other copies are listed under their run.");
            }
            s.push_str("\n\n");
            for (i, r) in sets.iter().enumerate() {
                let (gpu, price) = pod_words(r);
                let date = r.date().unwrap_or_else(|| "unknown".to_owned());
                let n = all.shown_from(i);
                s.push_str(&format!(
                    "- `{}`: tag `{}`, {n} {} shown from it, ended {date}; pod GPU {}, {price}.",
                    r.label,
                    r.tag,
                    if n == 1 { "arm" } else { "arms" },
                    cell_text(&gpu)
                ));
                let other = all.not_shown_from(i);
                if !other.is_empty() {
                    s.push_str(&format!(
                        " Also here, shown from another run: {}.",
                        other
                            .iter()
                            .map(|(name, a, from)| format!(
                                "`{name}` ({}; shown from `{}`)",
                                cell_text(&copy_words(a)),
                                from.label
                            ))
                            .collect::<Vec<_>>()
                            .join(", ")
                    ));
                }
                s.push('\n');
            }
            s.push('\n');
        }
        None => s.push_str("No run set yet: every variant is not measured. The gold runs (`fv-lab run gold --all` on the `gold` tier, the `gold-run` workflow) fill this table; the numbers the repository already holds are in each gold file and the documents it cites.\n\n"),
    }
    // Validation ages count to the day the row's arm's run ended; a row
    // without an arm, to the last day of the run set.
    let day = |d: Option<String>| d.and_then(|d| day_number(&d).ok());
    let set_day = day(runs.and_then(Runs::date));
    let mut families: BTreeMap<String, Vec<&Gold>> = BTreeMap::new();
    for g in golds {
        families
            .entry(g.model.family.to_string())
            .or_default()
            .push(g);
    }
    for (fam, gs) in &families {
        s.push_str(&format!("## {fam}\n\n"));
        s.push_str("| variant | tier | status | GPU | cell (WxHxframes) | vs gold | wall s | text s | denoise s | decode s | load s | peak GiB | $/clip | last validated |\n");
        s.push_str("|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|\n");
        for g in gs {
            let v = &g.model.variant;
            let tier = g.tier.name.map_or("-", |t| t.as_str());
            let arm = runs.and_then(|r| r.arm(v.as_str()));
            let run_day = match arm {
                Some((r, _)) => day(r.date()),
                None => set_day,
            };
            let lv = last_validated(g, run_day);
            let not = |why: String| {
                let why = cell_text(&why);
                format!("| `{v}` | {tier} | not measured ({why}) | - | - | - | - | - | - | - | - | - | - | {lv} |\n")
            };
            let row = match (runs, arm) {
                (None, _) => not("no run set".into()),
                (Some(_), None) => not("no arm in the run set".into()),
                (Some(_), Some((_, a))) if a.skipped.is_some() => {
                    not(format!("skipped: {}", a.skipped.as_deref().unwrap_or("")))
                }
                (Some(_), Some((_, a))) if a.exit != Some(0) => not(match a.exit {
                    Some(e) => format!("failed: exit {e}"),
                    None => "no exit recorded".into(),
                }),
                (Some(_), Some((r, a))) => match &a.metrics {
                    Some(m) if r.tier != GOLD_TIER && m.model_id.as_deref() != Some(v.as_str()) => {
                        not(format!(
                            "arm `{v}` of tier `{}` is not its gold cell",
                            r.tier
                        ))
                    }
                    None if r.tier != GOLD_TIER => not(format!(
                        "arm `{v}` of tier `{}` is not its gold cell",
                        r.tier
                    )),
                    Some(m) if m.wall_s.is_some() => {
                        let gpu = cell_text(
                            &m.gpu
                                .clone()
                                .or_else(|| r.pod_gpu.clone())
                                .unwrap_or_else(|| "-".into()),
                        );
                        let clip = r.usd_per_hr.zip(m.wall_s).map(|(p, w)| p * w / 3600.0);
                        format!(
                            "| `{v}` | {tier} | measured | {gpu} | {} | {} | {} {} | {} | {} | {} | {} | {} | {} | {lv} |\n",
                            cell_of(m),
                            vs_gold(m, g),
                            opt(m.wall_s, 2),
                            m.measures,
                            opt(m.text_s, 2),
                            opt(m.denoise_s, 2),
                            opt(m.decode_s, 2),
                            opt(m.load_s, 1),
                            opt(m.peak_gib, 1),
                            clip.map_or_else(|| "-".to_owned(), |c| format!("{c:.5}")),
                        )
                    }
                    Some(m) => not(format!("{} has no wall time", m.source)),
                    None => not("no timing record".into()),
                },
            };
            s.push_str(&row);
        }
        s.push('\n');
    }
    if let Some(r) = runs {
        let extra: Vec<&str> = r
            .arm_names()
            .into_iter()
            .filter(|a| !golds.iter().any(|g| g.model.variant.as_str() == *a))
            .collect();
        if !extra.is_empty() {
            s.push_str(&format!(
                "Arms of the run set that are no gold variant (not shown): {}.\n",
                extra
                    .iter()
                    .map(|a| format!("`{a}`"))
                    .collect::<Vec<_>>()
                    .join(", ")
            ));
        }
    }
    s
}

/// The run directories a committed table names, in order (empty: rendered
/// without a run set).
pub fn run_set_of(committed: &str) -> Result<Vec<String>, String> {
    let line = committed
        .lines()
        .find_map(|l| l.strip_prefix(RUN_SET_PREFIX))
        .ok_or_else(|| format!("no `{RUN_SET_PREFIX}` line"))?;
    if line == NO_RUN_SET {
        return Ok(Vec::new());
    }
    line.strip_prefix('`')
        .and_then(|l| l.strip_suffix('`'))
        .filter(|l| !l.is_empty())
        .map(|l| l.split("`, `").map(str::to_owned).collect::<Vec<_>>())
        .filter(|ls| ls.iter().all(|l| !l.is_empty() && !l.contains('`')))
        .ok_or_else(|| {
            format!(
                "`{RUN_SET_PREFIX}{line}`: expected paths in backticks, separated by `{RUN_SET_SEP}`, or `{NO_RUN_SET}`"
            )
        })
}

/// The first line where two renderings differ, for `--check`'s message.
pub fn first_difference(a: &str, b: &str) -> Option<(usize, String, String)> {
    let (al, bl): (Vec<&str>, Vec<&str>) = (a.lines().collect(), b.lines().collect());
    (0..al.len().max(bl.len())).find_map(|i| {
        let (x, y) = (al.get(i).copied(), bl.get(i).copied());
        (x != y).then(|| {
            (
                i + 1,
                x.unwrap_or("<end>").to_owned(),
                y.unwrap_or("<end>").to_owned(),
            )
        })
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn run_set_line_round_trips() {
        assert_eq!(
            run_set_of("x\nRun set: none\n").unwrap(),
            Vec::<String>::new()
        );
        assert_eq!(
            run_set_of("Run set: `artifacts/runpod/gold/abc`\n").unwrap(),
            vec!["artifacts/runpod/gold/abc".to_owned()]
        );
        assert_eq!(
            run_set_of("Run set: `a/gold/x`, `a/gold/y`\n").unwrap(),
            vec!["a/gold/x".to_owned(), "a/gold/y".to_owned()]
        );
        assert!(run_set_of("nothing").is_err());
        assert!(run_set_of("Run set: artifacts").is_err());
        assert!(run_set_of("Run set: ``").is_err());
        assert!(run_set_of("Run set: `a/gold/x`, b").is_err());
        assert!(run_set_of("Run set: `a/gold/x`, ``").is_err());
    }

    /// A run of `tier`/`tag` with `pod.json` sha `sha` and these arms:
    /// `(name, skipped, exit, ended)`.
    fn run(
        tier: &str,
        tag: &str,
        sha: Option<&str>,
        arms: &[(&str, Option<&str>, Option<i64>, &str)],
    ) -> RunSet {
        RunSet {
            label: format!("artifacts/runpod/{tier}/{tag}"),
            tier: tier.into(),
            tag: tag.into(),
            pod_gpu: Some(format!("GPU of {tag}")),
            usd_per_hr: Some(2.0),
            sha: sha.map(str::to_owned),
            arms: arms
                .iter()
                .map(|(n, skipped, exit, ended)| {
                    (
                        (*n).to_owned(),
                        ArmRecord {
                            exit: *exit,
                            skipped: skipped.map(str::to_owned),
                            ended: Some((*ended).to_owned()),
                            metrics: None,
                        },
                    )
                })
                .collect(),
        }
    }

    const T1: &str = "2026-10-26T10:00:00Z";
    const T2: &str = "2026-10-27T01:00:00Z";

    #[test]
    fn runs_of_one_tier_and_commit_merge_their_arms() {
        let a = run(
            "gold",
            "1234567-1026080000-aaaa",
            Some("1234567"),
            &[
                ("r2-arm", None, Some(0), T1),
                ("late-arm", Some("wall cap near"), None, T1),
                ("never-ran", Some("weights incomplete"), None, T1),
                ("rerun", None, Some(101), T1),
                ("fails", None, Some(101), T1),
            ],
        );
        let b = run(
            "gold",
            "1234567-1026200000-bbbb",
            // A full sha, in capitals, agrees with the other run's short one.
            Some("1234567890ABCDEF1234567890ABCDEF12345678"),
            &[
                ("volume-arm", None, Some(101), T2),
                ("late-arm", None, Some(0), T2),
                ("never-ran", Some("wall cap near"), None, T2),
                ("fails", None, None, T2),
            ],
        );
        let c = run(
            "gold",
            "1234567-1026230000-cccc",
            None,
            &[
                ("rerun", None, Some(0), T2),
                ("fails", Some("wall cap near"), None, T2),
            ],
        );
        let runs = Runs::new(vec![a.clone(), b.clone(), c.clone()]).expect("one run set");
        assert_eq!(runs.tier(), "gold");
        assert_eq!(
            runs.arm_names().into_iter().collect::<Vec<_>>(),
            [
                "fails",
                "late-arm",
                "never-ran",
                "r2-arm",
                "rerun",
                "volume-arm"
            ]
        );
        let from = |name: &str| {
            let (r, arm) = runs.arm(name).expect(name);
            (r.tag.clone(), arm.exit, arm.skipped.clone())
        };
        // Each arm comes with its own run.
        assert_eq!(from("r2-arm"), (a.tag.clone(), Some(0), None));
        assert_eq!(from("volume-arm"), (b.tag.clone(), Some(101), None));
        // Skipped in the first run, passed in the second: the pass.
        assert_eq!(from("late-arm"), (b.tag.clone(), Some(0), None));
        // Failed in the first run, passed in a rerun: the pass.
        assert_eq!(from("rerun"), (c.tag.clone(), Some(0), None));
        // Failed twice, then skipped: the last failed copy.
        assert_eq!(from("fails"), (b.tag.clone(), None, None));
        // Skipped in both: the last skipped copy.
        assert_eq!(
            from("never-ran"),
            (b.tag.clone(), None, Some("wall cap near".to_owned()))
        );
        assert!(runs.arm("absent").is_none());
        assert!(runs.has_duplicates());
        assert_eq!(
            (runs.shown_from(0), runs.shown_from(1), runs.shown_from(2)),
            (1, 4, 1)
        );
        let not_a: Vec<(&str, &str)> = runs
            .not_shown_from(0)
            .into_iter()
            .map(|(n, _, r)| (n, r.tag.as_str()))
            .collect();
        assert_eq!(
            not_a,
            [
                ("fails", b.tag.as_str()),
                ("late-arm", b.tag.as_str()),
                ("never-ran", b.tag.as_str()),
                ("rerun", c.tag.as_str()),
            ]
        );
        assert_eq!(runs.date().as_deref(), Some("2026-10-27"));
        assert_eq!(
            runs.commit().as_deref(),
            Some("1234567890abcdef1234567890abcdef12345678")
        );
        // One run alone: every arm shown from it, none duplicated.
        let one = Runs::from(a.clone());
        assert!(!one.has_duplicates());
        assert_eq!(one.shown_from(0), a.arms.len());
        assert!(Runs::new(vec![a]).is_ok());
    }

    #[test]
    fn commits_are_known_only_as_seven_or_more_hex_digits() {
        let r = |sha: Option<&str>, tag: &str| run("gold", tag, sha, &[]);
        assert_eq!(
            known_commit(&r(Some("ABCDEF0"), "x")).as_deref(),
            Some("abcdef0")
        );
        assert_eq!(
            known_commit(&r(None, "ABCDEF0-1-a")).as_deref(),
            Some("abcdef0")
        );
        // Too short, empty or not hex: the tag's head, else not known.
        assert_eq!(
            known_commit(&r(Some(""), "1234567-1-a")).as_deref(),
            Some("1234567")
        );
        assert_eq!(known_commit(&r(Some("123456"), "x-1")), None);
        assert_eq!(known_commit(&r(Some("123456g"), "nocommit")), None);
        assert_eq!(known_commit(&r(None, "12345678-1-a")), None);
        assert_eq!(known_commit(&r(None, "")), None);
        assert!(!same_commit("1234567", "7654321"));
        assert!(same_commit("1234567", "1234567890"));
        assert!(same_dir("./a/gold/x/", "a//gold/x"));
        assert!(!same_dir("a/gold/x", "a/gold/y"));
    }

    #[test]
    fn runs_that_are_not_one_run_set_are_refused() {
        let a = run("gold", "1234567-1-a", None, &[("x", None, Some(0), T1)]);
        let err = |sets: Vec<RunSet>| Runs::new(sets).expect_err("refused");
        assert_eq!(err(Vec::new()), "no run set given");
        let other_tier = run("rtx6000", "1234567-1-b", None, &[("y", None, Some(0), T1)]);
        let e = err(vec![a.clone(), other_tier]);
        assert!(
            e.contains("different tiers") && e.contains("`rtx6000`"),
            "{e}"
        );
        let other_commit = run("gold", "7654321-1-b", None, &[("y", None, Some(0), T1)]);
        let e = err(vec![a.clone(), other_commit]);
        assert!(
            e.contains("different commits") && e.contains("`1234567`") && e.contains("`7654321`"),
            "{e}"
        );
        // pod.json's sha decides over the tag where both are there.
        let mut sha = run("gold", "1234567-1-c", Some("abcdef0"), &[]);
        assert!(err(vec![a.clone(), sha.clone()]).contains("different commits"));
        sha.sha = None;
        assert!(Runs::new(vec![a.clone(), sha]).is_ok());
        // With several runs, every run names its commit: an empty or short
        // sha, or a tag without one, is refused, not joined to any set.
        for (sha, tag) in [
            (Some(""), "nocommit"),
            (Some("12345"), "x-1"),
            (None, "nocommit"),
        ] {
            let unknown = run("gold", tag, sha, &[("z", None, Some(0), T1)]);
            let e = err(vec![a.clone(), unknown.clone()]);
            assert!(e.contains("its commit is not known"), "{e}");
            // Alone it renders, as one run always has.
            assert!(Runs::new(vec![unknown]).is_ok());
        }
        // The same directory, however written.
        let mut again = a.clone();
        again.label = format!("./{}/", a.label);
        let e = err(vec![a.clone(), again]);
        assert!(e.contains("are the same run directory"), "{e}");
        // An arm that passed in two runs.
        let twice = run("gold", "1234567-1-d", None, &[("x", None, Some(0), T2)]);
        let e = err(vec![a.clone(), twice]);
        assert!(
            e.contains("arm `x` passed in more than one run")
                && e.contains(
                    "`artifacts/runpod/gold/1234567-1-a`, `artifacts/runpod/gold/1234567-1-d`"
                ),
            "{e}"
        );
        // A pass beside failed or skipped copies is one run set.
        for (skipped, exit) in [(None, Some(1)), (None, None), (Some("wall cap near"), None)] {
            let other = run("gold", "1234567-1-d", None, &[("x", skipped, exit, T2)]);
            assert!(Runs::new(vec![a.clone(), other]).is_ok());
        }
    }

    #[test]
    fn one_run_renders_as_before_and_several_name_every_run() {
        let a = run(
            "gold",
            "1234567-1-a",
            Some("1234567"),
            &[("x", None, Some(0), T1), ("y", None, Some(101), T1)],
        );
        let mut b = run(
            "gold",
            "1234567-1-b",
            None,
            &[
                ("y", None, Some(0), T2),
                ("z", Some("wall cap near"), None, T2),
            ],
        );
        b.usd_per_hr = None;
        let one = render(&[], Some(&Runs::from(a.clone())));
        assert!(
            one.contains("Run set: `artifacts/runpod/gold/1234567-1-a`\n"),
            "{one}"
        );
        assert!(one.contains(
            "Tier `gold`, tag `1234567-1-a`, commit `1234567`, ended 2026-10-26; pod GPU GPU of 1234567-1-a, $2.00/hr.\n"
        ));
        assert_eq!(run_set_of(&one).unwrap(), vec![a.label.clone()]);
        let both = Runs::new(vec![a.clone(), b.clone()]).expect("one run set");
        let two = render(&[], Some(&both));
        assert!(
            two.contains("Run set: `artifacts/runpod/gold/1234567-1-a`, `artifacts/runpod/gold/1234567-1-b`\n"),
            "{two}"
        );
        assert!(two.contains(
            "Tier `gold`, commit `1234567`, ended 2026-10-27: 2 runs rendered as one run set (ADR-0019 §16.7). An arm in more than one run is shown from the copy that passed"
        ), "{two}");
        assert!(two.contains(
            "- `artifacts/runpod/gold/1234567-1-a`: tag `1234567-1-a`, 1 arm shown from it, ended 2026-10-26; pod GPU GPU of 1234567-1-a, $2.00/hr. Also here, shown from another run: `y` (failed here: exit 101; shown from `artifacts/runpod/gold/1234567-1-b`).\n"
        ), "{two}");
        assert!(two.contains(
            "- `artifacts/runpod/gold/1234567-1-b`: tag `1234567-1-b`, 2 arms shown from it, ended 2026-10-27; pod GPU GPU of 1234567-1-b, no price recorded.\n"
        ), "{two}");
        assert!(!two.contains("Tier `gold`, tag"), "{two}");
        assert!(two.contains("not shown): `x`, `y`, `z`."), "{two}");
        assert_eq!(run_set_of(&two).unwrap(), vec![a.label.clone(), b.label]);
        // No arm in two runs: no sentence about copies.
        let c = run("gold", "1234567-1-c", None, &[("w", None, Some(0), T2)]);
        let two = render(&[], Some(&Runs::new(vec![a, c]).expect("one run set")));
        assert!(
            two.contains("2 runs rendered as one run set (ADR-0019 §16.7).\n\n"),
            "{two}"
        );
        assert!(!two.contains("Also here"), "{two}");
    }

    #[test]
    fn commit_and_date_of_a_run_set() {
        let mut r = RunSet {
            tag: "492e28a-1024114432-9d08".into(),
            ..RunSet::default()
        };
        assert_eq!(r.commit().as_deref(), Some("492e28a"));
        r.sha = Some("1234567".into());
        assert_eq!(r.commit().as_deref(), Some("1234567"));
        r.arms.insert(
            "a".into(),
            ArmRecord {
                ended: Some("2026-10-24T11:49:23Z".into()),
                ..ArmRecord::default()
            },
        );
        r.arms.insert(
            "b".into(),
            ArmRecord {
                ended: Some("2026-10-23T23:00:00Z".into()),
                ..ArmRecord::default()
            },
        );
        assert_eq!(r.date().as_deref(), Some("2026-10-24"));
    }

    #[test]
    fn table_cells_escape_pipes() {
        assert_eq!(cell_text("a|b\nc"), "a\\|b c");
    }

    #[test]
    fn first_difference_names_the_line() {
        assert_eq!(first_difference("a\nb\n", "a\nb\n"), None);
        assert_eq!(
            first_difference("a\nb\n", "a\nc\n"),
            Some((2, "b".into(), "c".into()))
        );
        assert_eq!(
            first_difference("a\n", "a\nc\n"),
            Some((2, "<end>".into(), "c".into()))
        );
    }
}
