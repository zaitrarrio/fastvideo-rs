//! `fv-lab report tiers`: `docs/perf/tiers.md`, the gold variants as one run
//! set measured them (plan D3, docs/adr/0023-lab-report-and-ci-bridge.md §1).
//!
//! The input is one run set: a directory `artifacts/runpod/<tier>/<tag>/` as
//! `fv-lab pod fetch` (or `runpod-http.sh`) writes it, one arm directory per
//! run directory (`summary.json`, and `benchmark.json` for a CLI generation or
//! `gpucheck-out/engine.json` for a serve-engine job), plus `pod.json`. A
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

use std::collections::BTreeMap;
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

/// Renders the table from the gold files (in order) and the run set.
pub fn render(golds: &[Gold], runs: Option<&RunSet>) -> String {
    let mut s = String::new();
    s.push_str("<!-- GENERATED by `fv-lab report tiers` (crates/fastvideo-lab/src/report.rs) from lab/gold/*.toml and the run set below. Do not edit: rerun it; `fv-lab report tiers --check` fails when this file differs. -->\n");
    s.push_str("# Tier table: the gold variants, measured by one run set\n\n");
    match runs {
        Some(r) => s.push_str(&format!("{RUN_SET_PREFIX}`{}`\n\n", r.label)),
        None => s.push_str(&format!("{RUN_SET_PREFIX}{NO_RUN_SET}\n\n")),
    }
    s.push_str(
        "One row per gold file (`lab/gold/<variant>.toml`, ADR-0009). A variant is measured by the arm of the same name in the run set (`fv-lab run gold --all` names its arms by variant id); every value is read from that arm's files (`summary.json`, `benchmark.json` or `gpucheck-out/engine.json`) and the run's `pod.json`, and `$/clip` is the run's own `usd_per_hr` times its own wall time. A variant without an arm, with a skipped or failed arm, or without a timing record is not measured: nothing is filled in from elsewhere. A variant counts as measured only by its gold cell: an arm of a `gold` run set, or an arm whose engine report ran that variant; a same-named arm of another tier is not. Wall time `gen` is one warm `generate()` (text, denoise, decode, MP4; load excluded), `job` a serve-engine job from submit to MP4; `decode s` is the decode through the finished MP4 in both (for `job`, the engine's `video_decode` plus its `encode`). `vs gold` compares the measured canvas, frames and steps with the gold file's `[workload]`. Design: docs/adr/0023-lab-report-and-ci-bridge.md.\n\n",
    );
    match runs {
        Some(r) => {
            let commit = r.commit().unwrap_or_else(|| "unknown".to_owned());
            let price = r
                .usd_per_hr
                .map_or_else(|| "no price recorded".to_owned(), |p| format!("${p:.2}/hr"));
            let gpu = r.pod_gpu.clone().unwrap_or_else(|| "not recorded".to_owned());
            let date = r.date().unwrap_or_else(|| "unknown".to_owned());
            s.push_str(&format!(
                "Tier `{}`, tag `{}`, commit `{commit}`, ended {date}; pod GPU {gpu}, {price}.\n\n",
                r.tier, r.tag
            ));
        }
        None => s.push_str("No run set yet: every variant is not measured. The gold runs (`fv-lab run gold --all` on the `gold` tier, the `gold-run` workflow) fill this table; the numbers the repository already holds are in each gold file and the documents it cites.\n\n"),
    }
    let run_day = runs
        .and_then(RunSet::date)
        .and_then(|d| day_number(&d).ok());
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
            let lv = last_validated(g, run_day);
            let arm = runs.and_then(|r| r.arms.get(v.as_str()));
            let not = |why: String| {
                let why = cell_text(&why);
                format!("| `{v}` | {tier} | not measured ({why}) | - | - | - | - | - | - | - | - | - | - | {lv} |\n")
            };
            let row = match (runs, arm) {
                (None, _) => not("no run set".into()),
                (Some(_), None) => not("no arm in the run set".into()),
                (Some(_), Some(a)) if a.skipped.is_some() => {
                    not(format!("skipped: {}", a.skipped.as_deref().unwrap_or("")))
                }
                (Some(_), Some(a)) if a.exit != Some(0) => not(match a.exit {
                    Some(e) => format!("failed: exit {e}"),
                    None => "no exit recorded".into(),
                }),
                (Some(r), Some(a)) => match &a.metrics {
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
            .arms
            .keys()
            .filter(|a| !golds.iter().any(|g| &g.model.variant == *a))
            .map(String::as_str)
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

/// The run set a committed table names (`None`: rendered without one).
pub fn run_set_of(committed: &str) -> Result<Option<String>, String> {
    let line = committed
        .lines()
        .find_map(|l| l.strip_prefix(RUN_SET_PREFIX))
        .ok_or_else(|| format!("no `{RUN_SET_PREFIX}` line"))?;
    if line == NO_RUN_SET {
        return Ok(None);
    }
    line.strip_prefix('`')
        .and_then(|l| l.strip_suffix('`'))
        .map(|l| Some(l.to_owned()))
        .ok_or_else(|| {
            format!("`{RUN_SET_PREFIX}{line}`: expected a path in backticks or `{NO_RUN_SET}`")
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
        assert_eq!(run_set_of("x\nRun set: none\n").unwrap(), None);
        assert_eq!(
            run_set_of("Run set: `artifacts/runpod/gold/abc`\n").unwrap(),
            Some("artifacts/runpod/gold/abc".to_owned())
        );
        assert!(run_set_of("nothing").is_err());
        assert!(run_set_of("Run set: artifacts").is_err());
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
