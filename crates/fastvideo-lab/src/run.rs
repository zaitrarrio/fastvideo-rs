//! `fv-lab run`: the on-pod cell runner, a port of `scripts/gpu/runpod-matrix.sh`'s
//! frame (the library links, `run_cell`, `gated_cell`, `compare_cells`,
//! `gate_cells`, the log) with the family's cells read from `lab/` instead of
//! a `case` branch (docs/adr/0019-fv-lab.md §5).
//!
//! It writes what the script writes, where the script writes it, so
//! `runpod-http.sh`'s fetch, the existing summaries and `compare` keep
//! working on an fv-lab run directory:
//!
//! ```text
//! <scratch>/runs/<family>/<tag>/
//!   live.log                    every log line ([HH:MM:SS] [<family>] …)
//!   done.json                   {"family":…,"ended":…}
//!   <arm>.weights.log           verify-weights.sh <gate>
//!   <arm>/summary.json          {"cell","family","exit","seconds","ended"} or {…,"skipped":…}
//!   <arm>/stdout.log, stderr.log, nvidia-smi-{start,end}.txt
//!   <arm>/frames/…, benchmark.json, gpucheck-out/   (fv-gpucheck's own)
//!   compare/compare-clips-<b>--<c>[-<prompt>].json, <tag>.{stdout,stderr}.log
//!   gate/gate-<b>--<c>.json, <tag>.{stdout,stderr}.log
//!   lpips-fetch.log             (FV_LPIPS=1)
//!   <family>-<cell>.verdict.json, <family>.verdict.json, <cell>.<diff>,
//!   <arm>/<evidence file>       (a cell's [verdict], D1d: b4a's files)
//!   <arm>/r2-fetch.log, r2-fetch.json, r2-apt.log   (a `volume+r2` tier, ADR-0019 §14)
//! ```
//!
//! [`Runner::regate`] is `fv-lab gate`: the same gates and verdicts evaluated
//! again from a finished run directory, with nothing run and nothing removed.
//!
//! The weight gate is `verify-weights.sh` and the clip metrics and the
//! promotion gate are `fv-gpucheck compare-clips` / `gate`: called, never
//! reimplemented. `--dry-run` prints each arm's command line as the script's
//! `FV_DRY_RUN=1` does (`[<family>] dry-run <arm>: <printf %q …>`), and
//! nothing is written.

use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File, OpenOptions};
use std::io::Write as _;
use std::os::unix::fs::PermissionsExt as _;
use std::os::unix::process::ExitStatusExt as _;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use fastvideo_deploy::boot::sys::utc_hms;
use fastvideo_deploy::launch::utc_iso;

use crate::plan::{GateKind, Paths, Step, Verdict};
use crate::quote::printf_q_line;
use crate::weights::{self, gb, R2Setup, R2Tree, WeightsPlan, R2_KEY_ENV, R2_STATE_DIR};

/// `runpod-matrix.sh`'s default binary.
pub const DEFAULT_GPUCHECK: &str = "/opt/fastvideo-rs/target/release/fv-gpucheck";

/// Why a run stopped before its cells (the script's `exit 2` cases).
#[derive(thiserror::Error, Debug)]
pub enum RunError {
    #[error("FATAL: {0}")]
    Fatal(String),
    #[error("{path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
}

fn io(path: &Path) -> impl FnOnce(std::io::Error) -> RunError + '_ {
    move |source| RunError::Io {
        path: path.to_path_buf(),
        source,
    }
}

/// What one run needs (the script's environment, as flags).
#[derive(Debug, Clone)]
pub struct RunConfig {
    /// The tier (`runs/<family>/…`, the log prefix, `summary.json`).
    pub family: String,
    /// `FV_WORK`: the volume mount (`/workspace`); weights under `weights/`.
    pub work: PathBuf,
    /// `FV_SCRATCH`: everything written (`/fvscratch` on a pod).
    pub scratch: PathBuf,
    /// `FV_RUN_TAG`: `runs/<family>/<tag>`.
    pub tag: Option<String>,
    /// `FV_RUNS_DIR`: the run directory itself, instead of
    /// `<scratch>/runs/<family>[/<tag>]` (the script's switch for one family
    /// running another's cells into its own run directory: `headline`).
    pub runs_dir: Option<PathBuf>,
    /// `FV_GPUCHECK`.
    pub bin: PathBuf,
    /// The repository / image root (`scripts/gpu/…`, `{root}`).
    pub root: PathBuf,
    /// `FV_GEN_TIMEOUT_S` (0: none).
    pub cell_timeout_s: u64,
    /// `FV_CELLS`: only these arms (and the others logged as skipped).
    pub only: Option<Vec<String>>,
    /// `FV_LPIPS=1`: fetch-lpips.sh, then compare-clips with `--lpips`.
    pub lpips: bool,
    /// `FV_GATE_POLICY` (default `<root>/scripts/gpu/gate-policy.toml`).
    pub gate_policy: Option<PathBuf>,
    /// `nvidia-smi` (a test points it at a fake).
    pub nvidia_smi: String,
    /// `timeout` (coreutils).
    pub timeout: String,
    /// The pause between GPU-memory polls before a cell (2 s).
    pub gpu_poll: Duration,
    /// `FV_BASELINE_SHA`: the commit whose runtime image's fv-gpucheck the
    /// `bin = "baseline"` arms run (fetched into `<scratch>/baseline-<sha>/`).
    pub baseline_sha: Option<String>,
    /// `FV_AUX_DIR` (default `<weights>/auxiliary`).
    pub aux_dir: Option<PathBuf>,
    /// `FV_TAE_DIR` (default: `<aux>/tae` when complete, else `<scratch>/tae`).
    pub tae_dir: Option<PathBuf>,
    /// A `weights = "volume+r2"` tier (ADR-0019 §14): `{weights}` is the
    /// merged root on the scratch disk, and the trees the volume lacks are
    /// fetched from R2 before the arms that read them. `None`: the volume.
    pub r2: Option<R2Setup>,
    /// `FV_RUN_DEADLINE` (unix seconds): the run must have ended by then (the
    /// pod's wall cap less what the driver needs to fetch it, set by `pod
    /// up`). An arm that could not finish in the time left is skipped
    /// (`wall cap near`); the others run under the time left.
    pub deadline: Option<f64>,
}

/// The least time an arm gets before the deadline (s): less, and it is skipped.
pub const MIN_ARM_S: f64 = 600.0;
/// The fetch rate the deadline assumes (bytes/s): well under the 0.61-0.79
/// GB/s two R2-mode pods measured (docs/serve/e2e/h3-test-round-2026-10-14.md).
pub const R2_RATE_FLOOR: f64 = 250e6;
/// The fallback `apt-get` (python3 when the image lacks it) runs under this cap (s).
pub const APT_CAP_S: u64 = 600;

impl RunConfig {
    pub fn new(family: &str, root: &Path) -> Self {
        Self {
            family: family.to_owned(),
            work: PathBuf::from("/workspace"),
            scratch: PathBuf::from("/workspace"),
            tag: None,
            runs_dir: None,
            bin: PathBuf::from(DEFAULT_GPUCHECK),
            root: root.to_path_buf(),
            cell_timeout_s: 300,
            only: None,
            lpips: false,
            gate_policy: None,
            nvidia_smi: "nvidia-smi".into(),
            timeout: "timeout".into(),
            gpu_poll: Duration::from_secs(2),
            baseline_sha: None,
            aux_dir: None,
            tae_dir: None,
            r2: None,
            deadline: None,
        }
    }

    /// `$AUX`: `FV_AUX_DIR`, else `<weights>/auxiliary`.
    pub fn aux(&self) -> PathBuf {
        self.aux_dir
            .clone()
            .unwrap_or_else(|| self.weights().join("auxiliary"))
    }

    /// `$TAE`: `FV_TAE_DIR`; else the volume's `<aux>/tae` when its
    /// `.complete` marker exists; else `<scratch>/tae` (where the pod's start
    /// command fetches the files).
    /// A `volume+r2` run reads the marker on the volume itself: the merged
    /// root's link to it is made when the run starts, after the command
    /// lines are rendered, and the volume arms read the volume's
    /// `auxiliary/tae` through it (ADR-0019 §14.3).
    pub fn tae(&self) -> PathBuf {
        let marker = match (&self.r2, &self.aux_dir) {
            (Some(_), None) => self.volume_weights().join("auxiliary/tae/.complete"),
            _ => self.aux().join("tae/.complete"),
        };
        match &self.tae_dir {
            Some(d) => d.clone(),
            None if marker.is_file() => self.aux().join("tae"),
            None => self.scratch.join("tae"),
        }
    }

    /// The time left before [`RunConfig::deadline`] (s), if there is one.
    pub fn time_left(&self) -> Option<f64> {
        self.deadline.map(|d| d - now_unix())
    }

    /// An arm's cap: the tier's `cell_timeout_s` (0: none), never past the deadline.
    pub fn arm_cap_s(&self) -> u64 {
        match self.time_left() {
            None => self.cell_timeout_s,
            Some(left) => {
                let left = left.max(1.0) as u64;
                if self.cell_timeout_s == 0 {
                    left
                } else {
                    self.cell_timeout_s.min(left)
                }
            }
        }
    }

    /// `$BASE`'s directory: `<scratch>/baseline-<sha>`.
    pub fn baseline_dir(&self) -> Option<PathBuf> {
        self.baseline_sha
            .as_ref()
            .map(|s| self.scratch.join(format!("baseline-{s}")))
    }

    /// `$W`: the volume's `<work>/weights`, or a `volume+r2` run's merged
    /// root (`<scratch>/weights`).
    pub fn weights(&self) -> PathBuf {
        match &self.r2 {
            Some(r) => r.root.clone(),
            None => self.volume_weights(),
        }
    }

    /// The volume's weights directory (`<work>/weights`).
    pub fn volume_weights(&self) -> PathBuf {
        self.work.join("weights")
    }

    /// The selected arms and their trees, in run order (what a `volume+r2`
    /// run fetches for).
    pub fn selected_arms<'a>(&self, steps: &'a [Step]) -> Vec<(&'a str, &'a [String])> {
        Step::flatten(steps)
            .into_iter()
            .filter_map(|s| match s {
                Step::Arm { name, trees, .. } if self.selected(name) => {
                    Some((name.as_str(), trees.as_slice()))
                }
                _ => None,
            })
            .collect()
    }

    /// A `volume+r2` run's plan of the selected arms (every fetch assumed to
    /// succeed); `None` on a volume tier.
    pub fn weights_plan(&self, steps: &[Step]) -> Option<Result<WeightsPlan, String>> {
        self.r2
            .as_ref()
            .map(|r| WeightsPlan::new(&r.sources, &self.selected_arms(steps)))
    }

    /// `$RUNS`: `FV_RUNS_DIR`, else `<scratch>/runs/<family>[/<tag>]`.
    pub fn runs(&self) -> PathBuf {
        if let Some(d) = &self.runs_dir {
            return d.clone();
        }
        let r = self.scratch.join("runs").join(&self.family);
        match &self.tag {
            Some(t) if !t.is_empty() => r.join(t),
            _ => r,
        }
    }

    /// The placeholders' values.
    pub fn paths(&self) -> Paths {
        Paths {
            weights: self.weights(),
            scratch: self.scratch.clone(),
            runs: self.runs(),
            root: self.root.clone(),
            tae: self.tae(),
            baseline: self.baseline_dir().map(|d| d.join("fv-gpucheck")),
        }
    }

    fn scripts(&self) -> PathBuf {
        self.root.join("scripts/gpu")
    }

    fn selected(&self, name: &str) -> bool {
        self.only
            .as_ref()
            .is_none_or(|o| o.iter().any(|c| c == name))
    }
}

/// The lines `--dry-run` prints (without the `[HH:MM:SS] ` the log adds):
/// what `FV_DRY_RUN=1 runpod-matrix.sh <family>` logs for the same cells.
pub fn dry_run_lines(cfg: &RunConfig, steps: &[Step]) -> Vec<String> {
    let f = &cfg.family;
    let build = build_id(&cfg.bin);
    let mut out = vec![format!("[{f}] matrix start image={build}")];
    let plan = match cfg.weights_plan(steps) {
        None => None,
        Some(Ok(p)) => {
            if let Some(r) = &cfg.r2 {
                out.push(format!("[{f}] dry-run {}", weights_line(cfg, r, &p)));
            }
            Some(p)
        }
        Some(Err(e)) => {
            out.push(format!("[{f}] FATAL: {e}"));
            return out;
        }
    };
    dry_run_steps(cfg, plan.as_ref(), steps, &mut out);
    out.push(format!("[{f}] matrix done"));
    out
}

/// `weights <root>: …`: a `volume+r2` run's weights, logged at its start.
fn weights_line(cfg: &RunConfig, r: &R2Setup, p: &WeightsPlan) -> String {
    format!(
        "weights {}: {} volume trees linked from {}; {} R2 trees ({}) fetched before the arms that read them, at most {} on the container disk at once",
        r.root.display(),
        r.sources.link_roots().len(),
        cfg.volume_weights().display(),
        p.fetched_trees().len(),
        gb(p.fetch_bytes),
        gb(p.peak_bytes),
    )
}

/// The dry run's line of one arm's R2 step (`drop …; fetch …`), when it has one.
fn r2_step_line(root: &Path, w: &weights::ArmWeights) -> Option<String> {
    let mut parts = Vec::new();
    if !w.drop.is_empty() {
        let d: Vec<String> = w
            .drop
            .iter()
            .map(|t| format!("{} ({})", t.dest, gb(t.bytes)))
            .collect();
        parts.push(format!("drop {}", d.join(", ")));
    }
    if !w.fetch.is_empty() {
        let d: Vec<String> = w
            .fetch
            .iter()
            .map(|t| format!("{} ({})", t.spec(), gb(t.bytes)))
            .collect();
        parts.push(format!("fetch {} into {}", d.join(", "), root.display()));
    }
    (!parts.is_empty()).then(|| parts.join("; "))
}

fn dry_run_steps(
    cfg: &RunConfig,
    plan: Option<&WeightsPlan>,
    steps: &[Step],
    out: &mut Vec<String>,
) {
    let f = &cfg.family;
    for s in steps {
        match s {
            Step::Arm { name, command, .. } => {
                if !cfg.selected(name) {
                    out.push(format!("[{f}] skip {name} (not in FV_CELLS)"));
                } else {
                    let r2 = plan.zip(cfg.r2.as_ref()).and_then(|(p, r)| {
                        p.arm(name).and_then(|w| r2_step_line(&r.root, w))
                    });
                    if let Some(l) = r2 {
                        out.push(format!("[{f}] dry-run r2 {name}: {l}"));
                    }
                    out.push(format!("[{f}] dry-run {name}: {}", printf_q_line(command)));
                }
            }
            Step::Compare {
                baseline,
                candidate,
                args,
            } => out.push(format!(
                "[{f}] dry-run compare {baseline}--{candidate} {}",
                args.join(" ")
            )),
            Step::Gate {
                baseline,
                candidate,
                ..
            } => out.push(format!(
                "[{f}] skip gate {baseline}--{candidate} (benchmark.json missing)"
            )),
            // The script's b4a_pair: each verdict arm's directory removed
            // before it, the cell's steps, the verdict's description.
            Step::Verdict {
                cell,
                verdict,
                steps,
            } => {
                if !verdict_selected(cfg, &verdict.arms) {
                    out.push(format!("[{f}] skip {cell} (no arm in FV_CELLS)"));
                    continue;
                }
                for s in steps {
                    if let Step::Arm { name, .. } = s {
                        if verdict.arms.contains(name) {
                            out.push(format!(
                                "[{f}] dry-run {name}: rm -rf {}",
                                cfg.runs().join(name).display()
                            ));
                        }
                    }
                    dry_run_steps(cfg, plan, std::slice::from_ref(s), out);
                }
                out.push(format!("[{f}] dry-run {cell}: {}", verdict.describe));
            }
            // The script's line names its function, b4-verdict.sh
            // `b4_family_verdict`, whose rule `Runner::family_verdict` is.
            Step::FamilyVerdict { family, cells } => out.push(format!(
                "[{f}] dry-run {family}: family verdict b4_family_verdict {family} {} -> {} (a named cell without a passing verdict fails)",
                selected_cells(cfg, cells).join(" "),
                cfg.runs().join(format!("{family}.verdict.json")).display()
            )),
        }
    }
}

/// A verdict cell runs when any of its verdict arms is selected.
fn verdict_selected(cfg: &RunConfig, arms: &[String]) -> bool {
    arms.iter().any(|a| cfg.selected(a))
}

/// The cells of a family verdict that ran (one of their arms selected).
fn selected_cells<'a>(cfg: &RunConfig, cells: &'a [(String, Vec<String>)]) -> Vec<&'a str> {
    cells
        .iter()
        .filter(|(_, arms)| verdict_selected(cfg, arms))
        .map(|(c, _)| c.as_str())
        .collect()
}

/// `b4-verdict.sh b4_flat`: the bytes without spaces, tabs, CR or LF.
fn flat(bytes: &[u8]) -> Vec<u8> {
    bytes
        .iter()
        .copied()
        .filter(|b| !matches!(b, b' ' | b'\t' | b'\n' | b'\r'))
        .collect()
}

fn contains(hay: &[u8], needle: &[u8]) -> bool {
    hay.windows(needle.len()).any(|w| w == needle)
}

/// `b4-verdict.sh b4_exit_ok`: `summary.json` records `"exit":0` (followed
/// by `,` or `}`), not a skip, a failure or nothing.
pub fn summary_exit_ok(summary: &Path) -> bool {
    if !summary.is_file() {
        return false;
    }
    let f = flat(&fs::read(summary).unwrap_or_default());
    contains(&f, b"\"exit\":0,") || contains(&f, b"\"exit\":0}")
}

/// `b4-verdict.sh b4_identical`: the `--off-identity` compare report exists,
/// is not a skip, and both identity checks passed.
pub fn compare_identical(report: &Path) -> bool {
    if !report.is_file() {
        return false;
    }
    let f = flat(&fs::read(report).unwrap_or_default());
    !contains(&f, b"\"status\":\"skipped\"")
        && contains(&f, b"\"name\":\"off_identity\",\"ok\":true")
        && contains(&f, b"\"name\":\"off_identity_frame_count\",\"ok\":true")
}

/// What one gate or verdict decided (`fv-lab gate`'s summary).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Outcome {
    /// `gate <b>--<c>`, `verdict <cell>`, `family <family>`.
    pub what: String,
    pub decision: Decision,
}

/// A gate's or verdict's decision.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Decision {
    Pass,
    Fail,
    /// Nothing to decide: a gate whose arms did not both run (skipped or
    /// unselected), as the script's `skip gate … (benchmark.json missing)`.
    Skipped,
}

impl Decision {
    fn of(pass: bool) -> Self {
        if pass {
            Decision::Pass
        } else {
            Decision::Fail
        }
    }
}

/// Every file the gates and verdicts of `steps` may write, relative to the
/// output directory: gate reports, verdict files, evidence files and diffs.
pub fn outputs(steps: &[Step]) -> Vec<String> {
    let mut out = Vec::new();
    for s in Step::flatten(steps) {
        match s {
            Step::Gate {
                baseline,
                candidate,
                ..
            } => out.push(format!("gate/gate-{baseline}--{candidate}.json")),
            Step::Verdict { cell, verdict, .. } => {
                out.push(format!("{}-{cell}.verdict.json", verdict.family));
                for e in &verdict.evidence {
                    for a in &verdict.arms {
                        out.push(format!("{a}/{}", e.file));
                    }
                    if let Some(d) = &e.diff {
                        out.push(format!("{cell}.{d}"));
                    }
                }
            }
            Step::FamilyVerdict { family, .. } => out.push(format!("{family}.verdict.json")),
            Step::Arm { .. } | Step::Compare { .. } => {}
        }
    }
    out
}

/// The run directory's record of the arms a run selected (`FV_CELLS`), which
/// `fv-lab run` writes before its first arm and `pod fetch` takes, so
/// `fv-lab gate` knows the run's cells even when some never wrote anything.
pub fn selection_file(family: &str) -> String {
    format!("{family}.arms.json")
}

/// The arms `fv-lab gate` evaluates a finished run with (ADR-0019 §11.3):
/// `cells` when given; else the run's selection record; else, for a run
/// without one (a script run, or fv-lab before the record), every arm of the
/// tier when the run holds them all, and an error otherwise: a run holding
/// fewer arms may have been cut short, and guessing its selection from what
/// it holds would pass the cells it never finished.
pub fn gate_selection(
    family: &str,
    tier_arms: &[String],
    run_dir: &Path,
    cells: Option<Vec<String>>,
) -> Result<Vec<String>, String> {
    let known = |list: Vec<String>, from: &str| match list.iter().find(|a| !tier_arms.contains(a)) {
        Some(a) => Err(format!("{from}: `{a}` is not an arm of tier `{family}`")),
        None => Ok(list),
    };
    if let Some(c) = cells {
        return known(c, "--cells");
    }
    let record = run_dir.join(selection_file(family));
    if record.is_file() {
        let text = fs::read_to_string(&record).map_err(|e| format!("{}: {e}", record.display()))?;
        let v: serde_json::Value =
            serde_json::from_str(&text).map_err(|e| format!("{}: {e}", record.display()))?;
        let arms = v
            .get("arms")
            .and_then(|a| a.as_array())
            .ok_or_else(|| format!("{}: no `arms` list", record.display()))?
            .iter()
            .map(|a| a.as_str().map(str::to_owned))
            .collect::<Option<Vec<_>>>()
            .ok_or_else(|| format!("{}: `arms` holds a non-string", record.display()))?;
        return known(arms, &record.display().to_string());
    }
    let present: Vec<&String> = tier_arms
        .iter()
        .filter(|a| run_dir.join(a).join("summary.json").is_file())
        .collect();
    if present.len() == tier_arms.len() {
        return Ok(tier_arms.to_vec());
    }
    Err(format!(
        "{} holds {} of the {} arms of `{family}` and no {} (a script run, or an fv-lab run before the record): pass --cells with the arms the run selected",
        run_dir.display(),
        present.len(),
        tier_arms.len(),
        selection_file(family)
    ))
}

/// `cat <bin>.build-id`, or `unknown`.
fn build_id(bin: &Path) -> String {
    let mut p = bin.as_os_str().to_owned();
    p.push(".build-id");
    fs::read_to_string(PathBuf::from(p))
        .ok()
        .map(|s| s.trim().to_owned())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "unknown".into())
}

/// The CUDA library names of `scripts/gpu/cuda-13.pins` (`KEY=value` lines).
pub fn read_pins(text: &str) -> BTreeMap<String, String> {
    text.lines()
        .map(str::trim)
        .filter(|l| !l.starts_with('#'))
        .filter_map(|l| l.split_once('='))
        .map(|(k, v)| (k.trim().to_owned(), v.trim().to_owned()))
        .collect()
}

/// `exit` of a finished process as bash reports it (128 + signal).
fn exit_code(st: std::process::ExitStatus) -> i32 {
    st.code()
        .or_else(|| st.signal().map(|s| 128 + s))
        .unwrap_or(1)
}

/// The runner.
pub struct Runner {
    pub cfg: RunConfig,
    /// What the steps read: the run directory.
    runs: PathBuf,
    /// Where the gates and verdicts write: the run directory on a pod;
    /// `fv-lab gate --out` offline.
    out: PathBuf,
    /// `live.log`; none offline (`fv-lab gate` logs to stdout only).
    log_path: Option<PathBuf>,
    /// Set on every child (`export PATH`, `LD_LIBRARY_PATH`, `FV_AUX_DIR`).
    env: Vec<(String, String)>,
    lpips_args: Vec<String>,
    /// `tae_fetched`: `fetch-tae.sh` runs at most once per run.
    tae_fetched: bool,
    /// `fv-lab gate`: re-evaluate a finished run's gates and verdicts; run no
    /// arm and no compare, remove nothing.
    offline: bool,
    /// Every gate and verdict decided, in order.
    outcomes: Vec<Outcome>,
    /// A `volume+r2` run: the trees fetched so far and still on the disk.
    held: BTreeSet<R2Tree>,
    /// `python3` ran (checked, and installed, at a fetch; a failure is not kept).
    python_ok: bool,
}

impl Runner {
    pub fn new(cfg: RunConfig) -> Self {
        let runs = cfg.runs();
        Self {
            cfg,
            out: runs.clone(),
            log_path: Some(runs.join("live.log")),
            runs,
            env: Vec::new(),
            lpips_args: Vec::new(),
            tae_fetched: false,
            offline: false,
            outcomes: Vec::new(),
            held: BTreeSet::new(),
            python_ok: false,
        }
    }

    /// The runner of `fv-lab gate`: reads the finished run directory
    /// (`cfg.runs()`, e.g. `artifacts/runpod/<tier>/<tag>`) and writes the
    /// re-evaluated gate reports (`<out>/gate/`) and verdicts (`<out>/…`)
    /// into `out` (the run directory itself, or elsewhere). No GPU.
    pub fn offline(cfg: RunConfig, out: &Path) -> Self {
        let mut r = Self::new(cfg);
        r.out = out.to_path_buf();
        r.log_path = None;
        r.offline = true;
        r
    }

    pub fn runs(&self) -> &Path {
        &self.runs
    }

    /// Every gate and verdict decided so far, in order.
    pub fn outcomes(&self) -> &[Outcome] {
        &self.outcomes
    }

    /// `log`: `[HH:MM:SS] [<family>] msg` to stdout and `live.log` (a write
    /// error there never stops the run, as in the script).
    pub fn log(&self, msg: &str) {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs_f64())
            .unwrap_or(0.0);
        self.raw(&format!("[{}] [{}] {msg}", utc_hms(now), self.cfg.family));
    }

    /// One line to stdout and `live.log` as is (`tee -a "$LOG"`).
    fn raw(&self, line: &str) {
        // Never a panic on a closed stdout (`println!` would): the line still
        // reaches live.log.
        let _ = writeln!(std::io::stdout().lock(), "{line}");
        let Some(log) = &self.log_path else { return };
        if let Ok(mut f) = OpenOptions::new().create(true).append(true).open(log) {
            let _ = writeln!(f, "{line}");
        }
    }

    fn cmd(&self, program: impl AsRef<std::ffi::OsStr>) -> Command {
        let mut c = Command::new(program);
        c.env_remove("FASTVIDEO_LTX2_WEIGHTS");
        // The R2 key reaches the plan step alone (`stage_r2`), never an arm.
        for (k, _) in R2_KEY_ENV {
            c.env_remove(k);
        }
        for (k, v) in &self.env {
            c.env(k, v);
        }
        c.stdin(Stdio::null());
        c
    }

    fn write_json(&self, dest: &Path, text: &str) {
        let _ = fs::write(dest, format!("{text}\n"));
    }

    fn now_iso() -> String {
        utc_iso(
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs_f64())
                .unwrap_or(0.0),
        )
    }

    /// The script's preamble: directories, the CUDA library links in
    /// `<scratch>/fv-libs`, the children's environment, LPIPS, the binary and
    /// GPU checks.
    pub fn prepare(&mut self) -> Result<(), RunError> {
        let libs = self.cfg.scratch.join("fv-libs");
        for d in [
            self.runs.clone(),
            self.cfg.scratch.join("gpucheck-out/logs"),
            libs.clone(),
        ] {
            fs::create_dir_all(&d).map_err(io(&d))?;
        }
        // Before anything reads `<weights>/auxiliary` (FV_AUX_DIR, LPIPS).
        if self.cfg.r2.is_some() {
            self.link_volume()?;
        }
        let pins_path = self.cfg.scripts().join("cuda-13.pins");
        let pins = read_pins(&fs::read_to_string(&pins_path).unwrap_or_default());
        let pin = |k: &str, d: &str| pins.get(k).cloned().unwrap_or_else(|| d.to_owned());
        if !libs.join("libcudnn.so").exists() {
            let cuda_lib = "/usr/local/cuda/targets/x86_64-linux/lib";
            for (name, src) in [
                (
                    "cudnn",
                    format!(
                        "/lib/x86_64-linux-gnu/{}",
                        pin("CUDA_CUDNN_SONAME", "libcudnn.so.9")
                    ),
                ),
                (
                    "cublas",
                    format!(
                        "{cuda_lib}/{}",
                        pin("CUDA_CUBLAS_SONAME", "libcublas.so.13")
                    ),
                ),
                (
                    "cublasLt",
                    format!(
                        "{cuda_lib}/{}",
                        pin("CUDA_CUBLASLT_SONAME", "libcublasLt.so.13")
                    ),
                ),
                (
                    "nvrtc",
                    format!("{cuda_lib}/{}", pin("CUDA_NVRTC_SONAME", "libnvrtc.so.13")),
                ),
            ] {
                link(&src, &libs.join(format!("lib{name}.so")));
            }
        }
        if !libs.join("libcupti.so").exists() {
            let soname = pin("CUDA_CUPTI_SONAME", "libcupti.so.13");
            let from_ldconfig = Command::new("ldconfig")
                .arg("-p")
                .stderr(Stdio::null())
                .output()
                .ok()
                .and_then(|o| {
                    String::from_utf8_lossy(&o.stdout).lines().find_map(|l| {
                        let mut w = l.split_whitespace();
                        (w.next() == Some(soname.as_str()))
                            .then(|| l.split_whitespace().last().map(str::to_owned))
                            .flatten()
                    })
                });
            let src = from_ldconfig
                .unwrap_or_else(|| format!("/usr/local/cuda/targets/x86_64-linux/lib/{soname}"));
            link(&src, &libs.join("libcupti.so"));
        }
        let path = std::env::var("PATH").unwrap_or_default();
        let ld = std::env::var("LD_LIBRARY_PATH").unwrap_or_default();
        let aux = self.cfg.aux().display().to_string();
        self.env = vec![
            (
                "PATH".into(),
                format!("/opt/fastvideo-rs/target/release:/usr/local/bin:/usr/local/cuda/bin:{path}"),
            ),
            (
                "LD_LIBRARY_PATH".into(),
                format!(
                    "{}:/lib/x86_64-linux-gnu:/usr/local/cuda-13.0/lib64:/usr/local/cuda/lib64:/usr/local/cuda/targets/x86_64-linux/lib:{ld}",
                    libs.display()
                ),
            ),
            ("FV_AUX_DIR".into(), aux.clone()),
        ];
        if !libs.join("libcudnn.so").exists() {
            self.raw("FATAL: libcudnn.so not linked");
            return Err(RunError::Fatal("libcudnn.so not linked".into()));
        }
        if self.cfg.lpips {
            let complete = Path::new(&aux).join("lpips/.complete").is_file();
            let dir = match std::env::var("FV_LPIPS_DIR").ok().filter(|s| !s.is_empty()) {
                Some(d) => PathBuf::from(d),
                None if complete => Path::new(&aux).join("lpips"),
                None => self.cfg.scratch.join("lpips"),
            };
            let log = self.runs.join("lpips-fetch.log");
            let ok = self
                .cmd("bash")
                .arg(self.cfg.scripts().join("fetch-lpips.sh"))
                .arg(&dir)
                .stdout(append(&log)?)
                .stderr(append(&log)?)
                .status()
                .is_ok_and(|s| s.success());
            if ok {
                self.lpips_args = vec!["--lpips".into(), dir.display().to_string()];
                self.log(&format!("lpips weights: {}", dir.display()));
            } else {
                self.log("WARN: LPIPS weights unavailable (lpips-fetch.log); compare-clips runs without LPIPS");
            }
        }
        self.log(&format!("matrix start image={}", build_id(&self.cfg.bin)));
        if !fastvideo_deploy::boot::sys::is_executable(&self.cfg.bin) {
            self.log(&format!(
                "FATAL: fv-gpucheck missing at {}",
                self.cfg.bin.display()
            ));
            return Err(RunError::Fatal(format!(
                "fv-gpucheck missing at {}",
                self.cfg.bin.display()
            )));
        }
        for args in [
            vec!["-L"],
            vec![
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader",
            ],
        ] {
            match self
                .cmd(&self.cfg.nvidia_smi)
                .args(&args)
                .stderr(Stdio::inherit())
                .output()
            {
                Ok(o) => {
                    for l in String::from_utf8_lossy(&o.stdout).lines() {
                        self.raw(l);
                    }
                    // The script runs these under `set -e -o pipefail`: a
                    // failing nvidia-smi ends the run with its status.
                    if !o.status.success() {
                        let why = format!(
                            "nvidia-smi {} failed (exit {})",
                            args.join(" "),
                            exit_code(o.status)
                        );
                        self.log(&format!("FATAL: {why}"));
                        return Err(RunError::Fatal(why));
                    }
                }
                Err(_) => {
                    self.log("FATAL: nvidia-smi missing");
                    return Err(RunError::Fatal("nvidia-smi missing".into()));
                }
            }
        }
        Ok(())
    }

    /// The merged root is off the volume ([`weights::check_off_volume`]).
    fn r2_guard(&self) -> Result<(), String> {
        match &self.cfg.r2 {
            Some(r) => weights::check_off_volume(&self.cfg.scratch, &r.root, &self.cfg.work),
            None => Ok(()),
        }
    }

    /// A `volume+r2` run's merged root (`<scratch>/weights`): one symlink per
    /// volume tree into `<work>/weights` (the volume is only read), made
    /// once; the fetched trees land beside them.
    fn link_volume(&self) -> Result<(), RunError> {
        let Some(r) = &self.cfg.r2 else {
            return Ok(());
        };
        if let Err(e) = self.r2_guard() {
            self.log(&format!("FATAL: {e}"));
            return Err(RunError::Fatal(e));
        }
        // A plan an interrupted run left (presigned URLs) goes first.
        let _ = fs::remove_file(self.cfg.scratch.join(R2_STATE_DIR).join("plan.json"));
        fs::create_dir_all(&r.root).map_err(io(&r.root))?;
        for t in r.sources.link_roots() {
            let link = r.root.join(t);
            if fs::symlink_metadata(&link).is_ok() {
                continue;
            }
            if let Some(p) = link.parent() {
                fs::create_dir_all(p).map_err(io(p))?;
            }
            std::os::unix::fs::symlink(self.cfg.volume_weights().join(t), &link)
                .map_err(io(&link))?;
        }
        if r.key.is_none() {
            self.log(&format!(
                "WARN: no R2 key in the environment ({} / {}, from the Runpod secrets {} / {}): arms that read R2 trees are skipped (r2 fetch failed)",
                R2_KEY_ENV[0].0, R2_KEY_ENV[1].0, R2_KEY_ENV[0].1, R2_KEY_ENV[1].1
            ));
        }
        Ok(())
    }

    /// Every step, then `matrix done` and `done.json`.
    pub fn run(&mut self, steps: &[Step]) -> Result<(), RunError> {
        self.prepare()?;
        match self.cfg.weights_plan(steps) {
            None => {}
            Some(Ok(p)) => {
                if let Some(r) = &self.cfg.r2 {
                    self.log(&weights_line(&self.cfg, r, &p));
                }
            }
            Some(Err(e)) => {
                self.log(&format!("FATAL: {e}"));
                return Err(RunError::Fatal(e));
            }
        }
        if Step::flatten(steps)
            .iter()
            .any(|s| matches!(s, Step::Arm { baseline: true, .. }))
        {
            self.fetch_baseline()?;
        }
        self.clear_verdicts(steps);
        self.write_selection(steps);
        for s in steps {
            self.step(s);
        }
        self.log("matrix done");
        self.write_json(
            &self.runs.join("done.json"),
            &format!(
                "{{\"family\":\"{}\",\"ended\":\"{}\"}}",
                self.cfg.family,
                Self::now_iso()
            ),
        );
        Ok(())
    }

    /// One step. Offline (`fv-lab gate`) only the gates and verdicts are
    /// evaluated: no arm or compare runs.
    fn step(&mut self, s: &Step) {
        match s {
            Step::Arm { .. } | Step::Compare { .. } if self.offline => {}
            Step::Arm {
                name,
                gate,
                trees,
                command,
                tae: Some(file),
                ..
            } => self.tae_gated_cell(name, file, gate, trees, command),
            Step::Arm {
                name,
                gate,
                trees,
                command,
                tae: None,
                ..
            } => self.gated_cell(name, gate, trees, command),
            Step::Compare {
                baseline,
                candidate,
                args,
            } => self.compare_cells(baseline, candidate, args),
            Step::Gate {
                baseline,
                candidate,
                kind,
                off,
            } => {
                let what = format!("gate {baseline}--{candidate}");
                let decision = match self.gate_cells(baseline, candidate, *kind, off.as_deref()) {
                    Some(rc) => Decision::of(rc == 0),
                    // Both arms ran (an exit recorded) and still no
                    // benchmark.json: a crashed arm, not a pair with
                    // nothing to gate.
                    None if self.arm_ran(baseline) && self.arm_ran(candidate) => {
                        if self.offline {
                            self.log(&format!(
                                "FAIL {what}: both arms ran and a benchmark.json is missing"
                            ));
                        }
                        Decision::Fail
                    }
                    None => Decision::Skipped,
                };
                self.outcomes.push(Outcome { what, decision });
            }
            Step::Verdict {
                cell,
                verdict,
                steps,
            } => self.verdict_cell(cell, verdict, steps),
            Step::FamilyVerdict { family, cells } => self.family_verdict(family, cells),
        }
    }

    /// `<runs>/<family>.arms.json` ([`selection_file`]): the arms this run
    /// selected, in run order, before the first of them runs.
    fn write_selection(&self, steps: &[Step]) {
        let arms: Vec<&str> = Step::flatten(steps)
            .into_iter()
            .filter_map(|s| match s {
                Step::Arm { name, .. } if self.cfg.selected(name) => Some(name.as_str()),
                _ => None,
            })
            .collect();
        self.write_json(
            &self.runs.join(selection_file(&self.cfg.family)),
            &serde_json::json!({ "family": self.cfg.family, "arms": arms }).to_string(),
        );
    }

    /// Whether an arm ran: its `summary.json` records an exit (not a skip).
    fn arm_ran(&self, arm: &str) -> bool {
        fs::read_to_string(self.runs.join(arm).join("summary.json"))
            .ok()
            .and_then(|t| serde_json::from_str::<serde_json::Value>(&t).ok())
            .is_some_and(|v| v.get("exit").is_some_and(serde_json::Value::is_number))
    }

    /// `fv-lab gate`: the gates (`fv-gpucheck gate`, CPU only) and verdicts
    /// of a finished run, evaluated again from its files into `out`, which is
    /// never the run directory or inside it (the run's own files stay what
    /// the pod wrote); the outcomes. What an earlier evaluation left in `out`
    /// is removed first.
    pub fn regate(&mut self, steps: &[Step]) -> Result<&[Outcome], RunError> {
        self.offline = true;
        // Checked before anything is created: not even an empty directory
        // goes into the run.
        if resolved(&self.out).starts_with(resolved(&self.runs)) {
            return Err(RunError::Fatal(format!(
                "{} is the run directory or inside it: fv-lab gate writes elsewhere (--out)",
                self.out.display()
            )));
        }
        fs::create_dir_all(&self.out).map_err(io(&self.out))?;
        for rel in outputs(steps) {
            let p = self.out.join(&rel);
            let _ = fs::remove_file(&p);
        }
        for s in steps {
            self.step(s);
        }
        Ok(&self.outcomes)
    }

    /// `fv-lab gate --check`: each file the evaluation writes against the
    /// run's own, one line per difference (none: the run wrote what fv-lab
    /// derives). Verdict files, evidence files and diffs byte for byte; a
    /// gate report by its `status` (it also holds timings).
    pub fn differences(&self, steps: &[Step]) -> Vec<String> {
        let mut out = Vec::new();
        for rel in outputs(steps) {
            let (ours, theirs) = (self.out.join(&rel), self.runs.join(&rel));
            match (ours.is_file(), theirs.is_file()) {
                (false, false) => {}
                (true, false) => out.push(format!("{rel}: not in the run")),
                (false, true) => out.push(format!("{rel}: in the run, not derived from it")),
                (true, true) if rel.starts_with("gate/") => {
                    let status = |p: &Path| {
                        fs::read_to_string(p)
                            .ok()
                            .and_then(|t| serde_json::from_str::<serde_json::Value>(&t).ok())
                            .and_then(|v| v.get("status").cloned())
                    };
                    let (a, b) = (status(&ours), status(&theirs));
                    if a.is_none() || a != b {
                        out.push(format!("{rel}: status {a:?} derived, {b:?} in the run"));
                    }
                }
                (true, true) => {
                    if fs::read(&ours).ok() != fs::read(&theirs).ok() {
                        out.push(format!("{rel}: differs (diff the two files)"));
                    }
                }
            }
        }
        out
    }

    /// The script's `rm -f "$RUNS"/b4a-*.verdict.json "$RUNS/b4a.verdict.json"`
    /// before the first arm: no verdict of an earlier run in this directory
    /// may stand in for this one.
    fn clear_verdicts(&self, steps: &[Step]) {
        for s in steps {
            let Step::FamilyVerdict { family, .. } = s else {
                continue;
            };
            let prefix = format!("{family}-");
            for e in fs::read_dir(&self.out).into_iter().flatten().flatten() {
                let n = e.file_name().to_string_lossy().into_owned();
                if n.starts_with(&prefix) && n.ends_with(".verdict.json") {
                    let _ = fs::remove_file(e.path());
                }
            }
            let _ = fs::remove_file(self.out.join(format!("{family}.verdict.json")));
        }
    }

    /// A verdict cell (the script's `b4a_pair`): each verdict arm from an
    /// empty directory and its evidence right after it, the cell's compares
    /// (their identity reports removed first), then
    /// `<family>-<cell>.verdict.json`. Skipped whole when none of its arms is
    /// selected.
    fn verdict_cell(&mut self, cell: &str, v: &Verdict, steps: &[Step]) {
        if !verdict_selected(&self.cfg, &v.arms) {
            self.log(&format!("skip {cell} (no arm in FV_CELLS)"));
            return;
        }
        let mut why = String::new();
        if !self.offline {
            for p in &v.identical {
                let _ = fs::remove_file(self.compare_report(&p.baseline, &p.candidate));
            }
        }
        for s in steps {
            match s {
                Step::Arm { name, .. } if v.arms.contains(name) => {
                    if !self.offline {
                        let _ = fs::remove_dir_all(self.runs.join(name));
                    }
                    self.step(s);
                    self.arm_evidence(cell, v, name, &mut why);
                }
                other => self.step(other),
            }
        }
        for p in &v.identical {
            if !compare_identical(&self.compare_report(&p.baseline, &p.candidate)) {
                why.push_str(&format!(" {}", p.why));
            }
        }
        for e in &v.evidence {
            let (Some(d), [a, b]) = (&e.diff, v.arms.as_slice()) else {
                continue;
            };
            let (fa, fb) = (
                self.out.join(a).join(&e.file),
                self.out.join(b).join(&e.file),
            );
            if !(fa.is_file() && fb.is_file()) {
                continue;
            }
            let same = File::create(self.out.join(format!("{cell}.{d}"))).is_ok_and(|f| {
                // As the script: on bytes, and as text even with a NUL, so
                // the file never names a path (`Binary files … differ`).
                self.cmd("diff")
                    .env("LC_ALL", "C")
                    .arg("-a")
                    .arg(&fa)
                    .arg(&fb)
                    .stdout(f)
                    .stderr(Stdio::null())
                    .status()
                    .is_ok_and(|s| s.success())
            });
            if !same {
                why.push_str(" evidence-differs");
            }
        }
        let verdict = if why.is_empty() { "pass" } else { "fail" };
        let why = why.strip_prefix(' ').unwrap_or(&why);
        self.write_json(
            &self.out.join(format!("{}-{cell}.verdict.json", v.family)),
            &format!("{{\"cell\":\"{cell}\",\"verdict\":\"{verdict}\",\"why\":\"{why}\"}}"),
        );
        let shown = if why.is_empty() {
            String::new()
        } else {
            format!(" ({why})")
        };
        self.log(&format!("{} {cell}: {verdict}{shown}", v.family));
        self.outcomes.push(Outcome {
            what: format!("verdict {cell}"),
            decision: Decision::of(verdict == "pass"),
        });
    }

    fn compare_report(&self, base: &str, cand: &str) -> PathBuf {
        self.runs
            .join("compare")
            .join(format!("compare-clips-{base}--{cand}.json"))
    }

    /// After a verdict arm: its exit, then (when it wrote a `stderr.log`) its
    /// evidence files and line counts; reasons appended to `why`.
    fn arm_evidence(&self, cell: &str, v: &Verdict, arm: &str, why: &mut String) {
        let label = Verdict::label(cell, arm);
        if !summary_exit_ok(&self.runs.join(arm).join("summary.json")) {
            why.push_str(&format!(" {label}:exit"));
        }
        let err = self.runs.join(arm).join("stderr.log");
        if !err.is_file() {
            why.push_str(&format!(" {label}:no-run"));
            return;
        }
        let text = fs::read(&err).unwrap_or_default();
        for e in &v.evidence {
            let d = self.out.join(arm);
            let _ = fs::create_dir_all(&d);
            let _ = fs::write(d.join(&e.file), e.extract(&text));
        }
        for c in &v.counts {
            let n = c.count(&text);
            self.log(&format!("[{arm}] {}: {n}", c.what));
            if n < c.min {
                why.push_str(&format!(" {label}:{}", c.why));
            }
        }
    }

    /// `b4-verdict.sh b4_family_verdict`: `<family>.verdict.json`, a pass only
    /// when every cell that ran wrote a passing verdict.
    fn family_verdict(&mut self, family: &str, cells: &[(String, Vec<String>)]) {
        let ran = selected_cells(&self.cfg, cells);
        // Offline a run directory without any of the cells holds no verdict
        // to re-evaluate (on a pod, as in the script, that is a failure).
        if self.offline && ran.is_empty() {
            self.log(&format!(
                "skip {family} family verdict (no cell of it in the run)"
            ));
            return;
        }
        let mut why = String::new();
        if ran.is_empty() {
            why.push_str(" no-cells");
        }
        for c in &ran {
            let f = self.out.join(format!("{family}-{c}.verdict.json"));
            if !f.is_file() {
                why.push_str(&format!(" {c}:no-verdict"));
            } else if !contains(
                &flat(&fs::read(&f).unwrap_or_default()),
                b"\"verdict\":\"pass\"",
            ) {
                why.push_str(&format!(" {c}:fail"));
            }
        }
        let verdict = if why.is_empty() { "pass" } else { "fail" };
        let why = why.strip_prefix(' ').unwrap_or(&why);
        self.write_json(
            &self.out.join(format!("{family}.verdict.json")),
            &format!(
                "{{\"family\":\"{family}\",\"verdict\":\"{verdict}\",\"cells\":\"{}\",\"why\":\"{why}\"}}",
                ran.join(" ")
            ),
        );
        let shown = if why.is_empty() {
            String::new()
        } else {
            format!(" ({why})")
        };
        self.log(&format!("{family} family: {verdict}{shown}"));
        self.outcomes.push(Outcome {
            what: format!("family {family}"),
            decision: Decision::of(verdict == "pass"),
        });
    }

    /// The baseline binary (the script's `techniques` / `b6` preamble):
    /// `fetch-baseline.sh <sha> <scratch>/baseline-<sha>` into
    /// `baseline-fetch.log`, then `baseline <its last line>`; a failure is the
    /// script's FATAL (exit 2, before any arm).
    pub fn fetch_baseline(&self) -> Result<(), RunError> {
        let (Some(sha), Some(dir)) = (self.cfg.baseline_sha.clone(), self.cfg.baseline_dir())
        else {
            self.log("FATAL: baseline arms need FV_BASELINE_SHA (the baseline commit)");
            return Err(RunError::Fatal(
                "baseline arms need FV_BASELINE_SHA (the baseline commit)".into(),
            ));
        };
        let log = self.runs.join("baseline-fetch.log");
        let ok = self
            .cmd("bash")
            .arg(self.cfg.scripts().join("fetch-baseline.sh"))
            .arg(&sha)
            .arg(&dir)
            .stdout(append(&log)?)
            .stderr(append(&log)?)
            .status()
            .is_ok_and(|s| s.success());
        let text = fs::read_to_string(&log).unwrap_or_default();
        if !ok {
            let why = format!("baseline binary sha-{sha} unavailable (baseline-fetch.log)");
            self.log(&format!("FATAL: {why}"));
            for l in text.lines() {
                self.raw(l);
            }
            return Err(RunError::Fatal(why));
        }
        self.log(&format!("baseline {}", text.lines().last().unwrap_or("")));
        // A baseline that is this build makes every byte-identity compare
        // pass whatever the build does: refuse it (fv-lab only; `pod up`
        // also refuses a baseline commit equal to the run's).
        let base_id = build_id(&dir.join("fv-gpucheck"));
        let image_id = build_id(&self.cfg.bin);
        self.log(&format!(
            "baseline build-id {base_id}, image build-id {image_id}"
        ));
        if base_id != "unknown" && base_id == image_id {
            let why = format!("the baseline binary sha-{sha} is this build ({base_id})");
            self.log(&format!("FATAL: {why}"));
            return Err(RunError::Fatal(why));
        }
        Ok(())
    }

    /// `tae_gated_cell`: `FV_CELLS`, then the tiny-autoencoder file (one
    /// `fetch-tae.sh` per run when it is missing; still missing: recorded as
    /// skipped), then `gated_cell`.
    pub fn tae_gated_cell(
        &mut self,
        name: &str,
        file: &Path,
        gate: &str,
        trees: &[String],
        command: &[String],
    ) {
        if !self.cfg.selected(name) {
            self.log(&format!("skip {name} (not in FV_CELLS)"));
            return;
        }
        if !file.is_file() && !self.tae_fetched {
            self.tae_fetched = true;
            let log = self.runs.join("tae-fetch.log");
            if let (Ok(out), Ok(err)) = (append(&log), append(&log)) {
                let _ = self
                    .cmd("bash")
                    .arg(self.cfg.scripts().join("fetch-tae.sh"))
                    .arg(self.cfg.tae())
                    .stdout(out)
                    .stderr(err)
                    .status();
            }
        }
        if !file.is_file() {
            let _ = fs::create_dir_all(self.runs.join(name));
            self.log(&format!(
                "SKIP {name}: {} missing (see tae-fetch.log)",
                file.display()
            ));
            self.skipped(name, "tae weights missing");
            return;
        }
        self.gated_cell(name, gate, trees, command);
    }

    fn skipped(&self, name: &str, why: &str) {
        let d = self.runs.join(name);
        let _ = fs::create_dir_all(&d);
        self.write_json(
            &d.join("summary.json"),
            &format!(
                "{{\"cell\":\"{name}\",\"family\":\"{}\",\"exit\":null,\"skipped\":\"{why}\"}}",
                self.cfg.family
            ),
        );
    }

    /// `gated_cell`: `FV_CELLS`, then (a `volume+r2` run) the arm's R2
    /// trees, then `verify-weights.sh <gate>`, then the arm.
    pub fn gated_cell(&mut self, name: &str, gate: &str, trees: &[String], command: &[String]) {
        if !self.cfg.selected(name) {
            self.log(&format!("skip {name} (not in FV_CELLS)"));
            return;
        }
        if let Some(why) = self.deadline_skip(trees) {
            let _ = fs::create_dir_all(self.runs.join(name));
            self.log(&format!("SKIP {name}: wall cap near ({why})"));
            self.skipped(name, "wall cap near");
            return;
        }
        if self.cfg.r2.is_some() {
            if let Err(why) = self.stage_r2(name, trees) {
                let _ = fs::create_dir_all(self.runs.join(name));
                self.log(&format!("SKIP {name}: r2 fetch failed: {why}"));
                self.skipped(name, "r2 fetch failed");
                return;
            }
        }
        let wlog = self.runs.join(format!("{name}.weights.log"));
        let verified = match File::create(&wlog) {
            Ok(out) => out.try_clone().is_ok_and(|err| {
                self.cmd("bash")
                    .arg(self.cfg.scripts().join("verify-weights.sh"))
                    .arg(gate)
                    .env("FV_WEIGHTS", self.cfg.weights())
                    .stdout(out)
                    .stderr(err)
                    .status()
                    .is_ok_and(|s| s.success())
            }),
            Err(_) => false,
        };
        if !verified {
            let _ = fs::create_dir_all(self.runs.join(name));
            self.log(&format!("SKIP {name}: weights incomplete"));
            for l in fs::read_to_string(&wlog).unwrap_or_default().lines() {
                self.raw(l);
            }
            self.skipped(name, "weights incomplete");
            return;
        }
        self.run_cell(name, command);
    }

    /// A `volume+r2` arm's R2 trees (ADR-0019 §14), before its gate: when
    /// one is not on the disk yet, every fetched tree the arm does not read
    /// is removed, then `r2-weights.py plan` (with the R2 key, the only child
    /// that gets it) presigns the missing trees into `<scratch>/r2/plan.json`
    /// and `r2-fetch.py` fetches them into the merged root under the arm's
    /// cap; the plan is removed after. The fetch's log and status go to
    /// `<arm>/r2-fetch.log` and `<arm>/r2-fetch.json`.
    fn stage_r2(&mut self, name: &str, trees: &[String]) -> Result<(), String> {
        let Some(r) = self.cfg.r2.clone() else {
            return Ok(());
        };
        self.r2_guard()?;
        let need = r.sources.arm_r2(name, trees)?;
        let st = weights::step(&self.held, &need);
        for t in &st.drop {
            remove_fetched(&self.cfg, &r.root, &t.dest)?;
            self.held.remove(t);
            self.log(&format!("r2 {name}: drop {} ({})", t.dest, gb(t.bytes)));
        }
        let mut todo = Vec::new();
        for t in st.fetch {
            if fastvideo_deploy::boot::weights::tree_present(&r.root, &t.dest, &t.rev) {
                self.log(&format!("r2 {name}: {} already on the disk", t.spec()));
                self.held.insert(t);
            } else {
                todo.push(t);
            }
        }
        if todo.is_empty() {
            return Ok(());
        }
        let key = r.key.clone().ok_or_else(|| {
            format!(
                "no R2 key in the environment ({} / {})",
                R2_KEY_ENV[0].0, R2_KEY_ENV[1].0
            )
        })?;
        let dir = self.runs.join(name);
        fs::create_dir_all(&dir).map_err(|e| format!("{}: {e}", dir.display()))?;
        let state = self.cfg.scratch.join(R2_STATE_DIR);
        fs::create_dir_all(&state).map_err(|e| format!("{}: {e}", state.display()))?;
        let _ = fs::set_permissions(&state, fs::Permissions::from_mode(0o700));
        self.ensure_python(&r, &state)?;
        let plan = state.join("plan.json");
        let _ = fs::remove_file(&plan);
        let log = dir.join("r2-fetch.log");
        let specs: Vec<String> = todo.iter().map(R2Tree::spec).collect();
        let bytes: u64 = todo.iter().map(|t| t.bytes).sum();
        self.log(&format!(
            "r2 {name}: fetch {} ({}) into {}",
            specs.join(" "),
            gb(bytes),
            r.root.display()
        ));
        let mut cap = weights::fetch_cap_s(self.cfg.cell_timeout_s);
        if let Some(left) = self.cfg.time_left() {
            cap = cap.min((left - MIN_ARM_S).max(1.0) as u64);
        }
        let out = |l: &Path| append(l).map_err(|e| e.to_string());
        // `--pin`: each manifest's SHA-256 is the one weights-r2.tsv records
        // (the table the trees were chosen from), as fv-control checks it.
        let planned = self
            .cmd(&r.python)
            .arg(self.cfg.scripts().join("r2-weights.py"))
            .arg("plan")
            .args(&specs)
            .arg("--pin")
            .arg(self.cfg.scripts().join(weights::R2_TABLE_NAME))
            .arg("--ttl")
            .arg((cap + 600).to_string())
            .arg("--out")
            .arg(&plan)
            .envs(key.env())
            .stdout(out(&log)?)
            .stderr(out(&log)?)
            .status()
            .is_ok_and(|s| s.success());
        if !planned {
            let _ = fs::remove_file(&plan);
            return Err(format!("r2-weights.py plan: {}", last_line(&log)));
        }
        self.r2_guard()?;
        // Every tree of the plan: no FV_R2_SKIP_TREES from the pod's env.
        let fetched = self
            .cmd(&self.cfg.timeout)
            .env_remove("FV_R2_SKIP_TREES")
            .args(["--signal=TERM", "--kill-after=15", &cap.to_string()])
            .arg(&r.python)
            .arg(self.cfg.scripts().join("r2-fetch.py"))
            .arg("--root")
            .arg(&r.root)
            .arg("--plan")
            .arg(&plan)
            .arg("--skip-trees")
            .arg("")
            .arg("--status")
            .arg(dir.join("r2-fetch.json"))
            .current_dir(&state)
            .stdout(out(&log)?)
            .stderr(out(&log)?)
            .status()
            .is_ok_and(|s| s.success());
        let _ = fs::remove_file(&plan);
        if !fetched {
            // A tree the fetch completed stays (held); one it did not is
            // removed, and so is the staging, so the disk holds only held
            // trees. Never on the volume (checked again).
            self.r2_guard()?;
            for t in todo {
                if fastvideo_deploy::boot::weights::tree_present(&r.root, &t.dest, &t.rev) {
                    self.held.insert(t);
                } else if let Err(e) = remove_fetched(&self.cfg, &r.root, &t.dest) {
                    self.log(&format!("r2 {name}: {e}"));
                }
            }
            let _ = fs::remove_dir_all(r.root.join(".r2-partial"));
            fastvideo_deploy::boot::weights::clean_leftovers(&r.root);
            let why = fastvideo_deploy::boot::weights::r2::failed_line(&log);
            return Err(format!(
                "r2-fetch.py: {}",
                if why.is_empty() { last_line(&log) } else { why }
            ));
        }
        self.held.extend(todo);
        let ready = fastvideo_deploy::boot::weights::r2::ready_line(&log);
        self.log(&format!("r2 {name}: {ready}"));
        Ok(())
    }

    /// `python3` for the R2 tools. The runtime image carries it
    /// (`docker/gpucheck.Dockerfile` `runtime`); an older image gets it with
    /// `apt-get install python3` (as fv-boot does), under `timeout 600` and
    /// a 120 s dpkg lock wait, into `<runs>/r2-apt.log`. Only a success is
    /// remembered: after a failure the next fetching arm tries again.
    fn ensure_python(&mut self, r: &R2Setup, state: &Path) -> Result<(), String> {
        if self.python_ok {
            return Ok(());
        }
        let check = |me: &Self| {
            me.cmd(&r.python)
                .args(["-c", weights::PYTHON_CHECK])
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .status()
                .is_ok_and(|s| s.success())
        };
        let mut ok = check(self);
        if !ok {
            let log = self.runs.join("r2-apt.log");
            self.log("r2: no python3 in the image: apt-get install python3 (r2-apt.log)");
            let apt = |args: &[&str]| {
                let (Ok(o), Ok(e)) = (append(&log), append(&log)) else {
                    return false;
                };
                self.cmd(&self.cfg.timeout)
                    .args(["--signal=TERM", "--kill-after=15", &APT_CAP_S.to_string()])
                    .arg(&r.apt_get)
                    .env("DEBIAN_FRONTEND", "noninteractive")
                    .args([
                        "-o",
                        "Acquire::http::Timeout=15",
                        "-o",
                        "Acquire::Retries=3",
                        "-o",
                        "DPkg::Lock::Timeout=120",
                    ])
                    .args(args)
                    .current_dir(state)
                    .stdout(o)
                    .stderr(e)
                    .status()
                    .is_ok_and(|s| s.success())
            };
            ok = apt(&["update", "-qq"])
                && apt(&["install", "-y", "-qq", "--no-install-recommends", "python3"])
                && check(self);
        }
        self.python_ok = ok;
        if ok {
            Ok(())
        } else {
            Err("python3 unavailable (r2-apt.log)".into())
        }
    }

    /// Why an arm is skipped for the deadline, if it is: less than
    /// [`MIN_ARM_S`] would be left after its R2 fetch (at [`R2_RATE_FLOOR`]).
    fn deadline_skip(&self, trees: &[String]) -> Option<String> {
        let left = self.cfg.time_left()?;
        let fetch = match &self.cfg.r2 {
            Some(r) => r.sources.arm_r2("", trees).map_or(0.0, |need| {
                weights::step(&self.held, &need)
                    .fetch
                    .iter()
                    .map(|t| t.bytes as f64)
                    .sum::<f64>()
                    / R2_RATE_FLOOR
            }),
            None => 0.0,
        };
        (left < fetch + MIN_ARM_S).then(|| {
            format!(
                "{:.0} s to the deadline, {:.0} s of fetch and {MIN_ARM_S:.0} s of arm needed",
                left.max(0.0),
                fetch
            )
        })
    }

    /// `gpu_clean`: wait for the previous process's memory to drain, kill a
    /// leftover compute process, refuse otherwise.
    fn gpu_clean(&self) -> bool {
        let mut used = String::new();
        for i in 1..=30 {
            used = self
                .cmd(&self.cfg.nvidia_smi)
                .args(["--query-gpu=memory.used", "--format=csv,noheader,nounits"])
                .stderr(Stdio::null())
                .output()
                .ok()
                .map(|o| {
                    String::from_utf8_lossy(&o.stdout)
                        .lines()
                        .next()
                        .unwrap_or("")
                        .replace(' ', "")
                })
                .unwrap_or_default();
            if used.parse::<u64>().is_ok_and(|u| u < 1024) {
                return true;
            }
            if i == 10 {
                let pids = self
                    .cmd(&self.cfg.nvidia_smi)
                    .args(["--query-compute-apps=pid", "--format=csv,noheader"])
                    .stderr(Stdio::null())
                    .output()
                    .map(|o| String::from_utf8_lossy(&o.stdout).into_owned())
                    .unwrap_or_default();
                for pid in pids.split_whitespace() {
                    self.log(&format!(
                        "killing leftover GPU process {pid} ({used} MiB in use)"
                    ));
                    let _ = self
                        .cmd("sh")
                        .args(["-c", "kill -9 \"$0\"", pid])
                        .stdout(Stdio::null())
                        .stderr(Stdio::null())
                        .status();
                }
            }
            std::thread::sleep(self.cfg.gpu_poll);
        }
        let shown = if used.is_empty() { "?" } else { used.as_str() };
        self.log(&format!(
            "GPU not clean before next cell: {shown} MiB in use"
        ));
        false
    }

    fn smi_snap(&self, dest: &Path) {
        if let Ok(f) = File::create(dest) {
            let _ = self
                .cmd(&self.cfg.nvidia_smi)
                .args([
                    "--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu",
                    "--format=csv,noheader",
                ])
                .stdout(f)
                .stderr(Stdio::null())
                .status();
        }
    }

    /// `run_cell`: an empty GPU, the command in the arm's directory under
    /// `timeout --signal=TERM --kill-after=15 <cap>`, its logs, `summary.json`.
    pub fn run_cell(&self, name: &str, command: &[String]) {
        if !self.cfg.selected(name) {
            self.log(&format!("skip {name} (not in FV_CELLS)"));
            return;
        }
        if !self.gpu_clean() {
            self.skipped(name, "gpu not clean");
            return;
        }
        let cell = self.runs.join(name);
        let _ = fs::create_dir_all(&cell);
        let t0 = Instant::now();
        let cap = self.cfg.arm_cap_s();
        self.log(&format!(
            "▶ {name}  {}  (wall cap {cap}s; 0=none)",
            command.join(" ")
        ));
        self.smi_snap(&cell.join("nvidia-smi-start.txt"));
        let rc = self.exec(&cell, command, cap);
        let secs = t0.elapsed().as_secs();
        self.smi_snap(&cell.join("nvidia-smi-end.txt"));
        if rc != 0 {
            self.log(&format!("FAIL {name}  {secs}s  exit={rc}"));
            let err = fs::read_to_string(cell.join("stderr.log")).unwrap_or_default();
            for l in fail_tail(&err) {
                self.raw(l);
            }
        } else {
            let peak = peak_line(&cell);
            self.log(&format!("ok {name}  {secs}s  {peak}"));
        }
        self.write_json(
            &cell.join("summary.json"),
            &format!(
                "{{\"cell\":\"{name}\",\"family\":\"{}\",\"exit\":{rc},\"seconds\":{secs},\"ended\":\"{}\"}}",
                self.cfg.family,
                Self::now_iso()
            ),
        );
        if let Some(d) = frames_digest(&cell, name) {
            let _ = fs::write(cell.join(FRAMES_DIGEST), d);
        }
    }

    /// Runs `command` in `cell` with stdout / stderr to its logs; bash's exit
    /// status (127 when it cannot start).
    fn exec(&self, cell: &Path, command: &[String], cap: u64) -> i32 {
        let (out, err) = match (
            File::create(cell.join("stdout.log")),
            File::create(cell.join("stderr.log")),
        ) {
            (Ok(o), Ok(e)) => (o, e),
            _ => return 1,
        };
        let mut c = if cap == 0 {
            let mut c = self.cmd(&command[0]);
            c.args(&command[1..]);
            c
        } else {
            let mut c = self.cmd(&self.cfg.timeout);
            c.args(["--signal=TERM", "--kill-after=15", &cap.to_string()])
                .args(command);
            c
        };
        let mut err_note = match err.try_clone() {
            Ok(e) => e,
            Err(_) => return 1,
        };
        match c.current_dir(cell).stdout(out).stderr(err).status() {
            Ok(st) => exit_code(st),
            Err(e) => {
                let _ = writeln!(err_note, "fv-lab: {}: {e}", command[0]);
                127
            }
        }
    }

    fn has_png(dir: &Path) -> bool {
        fs::read_dir(dir).is_ok_and(|rd| {
            rd.flatten()
                .any(|e| e.path().extension().is_some_and(|x| x == "png"))
        })
    }

    /// The prompt directories of a frames directory (sorted; `cold` and
    /// `warmup` are the untimed passes).
    fn prompt_dirs(frames: &Path) -> Vec<String> {
        let mut v: Vec<String> = fs::read_dir(frames)
            .map(|rd| {
                rd.flatten()
                    .filter(|e| e.path().is_dir())
                    .filter_map(|e| e.file_name().into_string().ok())
                    .filter(|n| n != "cold" && n != "warmup")
                    .collect()
            })
            .unwrap_or_default();
        v.sort();
        v
    }

    /// `compare_cells`: one compare, or one per prompt directory.
    pub fn compare_cells(&self, base: &str, cand: &str, args: &[String]) {
        let bf = self.runs.join(base).join("frames");
        let cf = self.runs.join(cand).join("frames");
        if Self::has_png(&bf) || !bf.is_dir() {
            self.compare_one(&format!("{base}--{cand}"), &bf, &cf, args);
            return;
        }
        let mut found = false;
        for p in Self::prompt_dirs(&bf) {
            if !Self::has_png(&bf.join(&p)) {
                continue;
            }
            found = true;
            self.compare_one(
                &format!("{base}--{cand}-{p}"),
                &bf.join(&p),
                &cf.join(&p),
                args,
            );
        }
        if !found {
            self.compare_one(&format!("{base}--{cand}"), &bf, &cf, args);
        }
    }

    fn compare_one(&self, tag: &str, bf: &Path, cf: &Path, args: &[String]) {
        let dir = self.runs.join("compare");
        let _ = fs::create_dir_all(&dir);
        if !Self::has_png(bf) || !Self::has_png(cf) {
            self.log(&format!("skip compare {tag} (frames missing)"));
            self.write_json(
                &dir.join(format!("compare-clips-{tag}.json")),
                &format!(
                    "{{\"stage\":\"compare-clips-{tag}\",\"status\":\"skipped\",\"reason\":\"frames missing\"}}"
                ),
            );
            return;
        }
        let rc = self.gpucheck(
            &dir,
            tag,
            &[
                vec![
                    "compare-clips".into(),
                    "--baseline".into(),
                    bf.display().to_string(),
                    "--candidate".into(),
                    cf.display().to_string(),
                ],
                self.lpips_args.clone(),
                args.to_vec(),
            ]
            .concat(),
        );
        let err = fs::read_to_string(dir.join(format!("{tag}.stderr.log"))).unwrap_or_default();
        self.log(&format!(
            "compare {tag} exit={rc} {} {}",
            last_psnr(&err),
            lpips_summary(&err)
        ));
    }

    /// `fv-gpucheck --out <dir> --tag <tag> <args>` with its logs in `dir`.
    fn gpucheck(&self, dir: &Path, tag: &str, args: &[String]) -> i32 {
        let (out, err) = match (
            File::create(dir.join(format!("{tag}.stdout.log"))),
            File::create(dir.join(format!("{tag}.stderr.log"))),
        ) {
            (Ok(o), Ok(e)) => (o, e),
            _ => return 1,
        };
        match self
            .cmd(&self.cfg.bin)
            .arg("--out")
            .arg(dir)
            .args(["--tag", tag])
            .args(args)
            .stdout(out)
            .stderr(err)
            .status()
        {
            Ok(st) => exit_code(st),
            Err(_) => 127,
        }
    }

    /// The compare reports of `<t>` for candidate `c` (one, or one per prompt).
    fn compare_reports(&self, t: &str, c: &str) -> Vec<PathBuf> {
        let mut v = Vec::new();
        let one = self.runs.join(format!("compare/compare-clips-{t}.json"));
        if one.is_file() {
            v.push(one);
        }
        for p in Self::prompt_dirs(&self.runs.join(c).join("frames")) {
            let f = self
                .runs
                .join(format!("compare/compare-clips-{t}-{p}.json"));
            if f.is_file() {
                v.push(f);
            }
        }
        v
    }

    /// `gate_cells`: `fv-gpucheck gate` with the gate policy; its exit status,
    /// or `None` when the pair has nothing to gate (a `benchmark.json` missing).
    pub fn gate_cells(
        &self,
        base: &str,
        cand: &str,
        kind: GateKind,
        off: Option<&str>,
    ) -> Option<i32> {
        let tag = format!("{base}--{cand}");
        let dir = self.out.join("gate");
        let _ = fs::create_dir_all(&dir);
        let mut args: Vec<String> = Vec::new();
        for f in self.compare_reports(&tag, cand) {
            args.push("--compare".into());
            args.push(f.display().to_string());
        }
        if let Some(off) = off {
            for f in self.compare_reports(&format!("{base}--{off}"), off) {
                args.push("--off-compare".into());
                args.push(f.display().to_string());
            }
        }
        if !self.runs.join(base).join("benchmark.json").is_file()
            || !self.runs.join(cand).join("benchmark.json").is_file()
        {
            self.log(&format!("skip gate {tag} (benchmark.json missing)"));
            return None;
        }
        let policy = self
            .cfg
            .gate_policy
            .clone()
            .unwrap_or_else(|| self.cfg.scripts().join("gate-policy.toml"));
        let rc = self.gpucheck(
            &dir,
            &tag,
            &[
                vec![
                    "gate".into(),
                    "--baseline".into(),
                    self.runs.join(base).display().to_string(),
                    "--candidate".into(),
                    self.runs.join(cand).display().to_string(),
                    "--policy".into(),
                    policy.display().to_string(),
                    "--kind".into(),
                    kind.as_str().into(),
                ],
                args,
            ]
            .concat(),
        );
        let err = fs::read_to_string(dir.join(format!("{tag}.stderr.log"))).unwrap_or_default();
        let verdict = err
            .lines()
            .filter_map(|l| l.find("gate verdict: ").map(|i| &l[i..]))
            .next_back()
            .unwrap_or("");
        self.log(&format!("gate {tag} exit={rc} {verdict}"));
        Some(rc)
    }
}

/// `p` with its longest existing ancestor canonicalized (symlinks, `..`)
/// and the rest appended: where `p` is or would be created.
pub(crate) fn resolved(p: &Path) -> PathBuf {
    let abs = std::path::absolute(p).unwrap_or_else(|_| p.to_path_buf());
    for base in abs.ancestors() {
        if let Ok(c) = fs::canonicalize(base) {
            let rest = abs.strip_prefix(base).unwrap_or(Path::new(""));
            return c.join(rest);
        }
    }
    abs
}

/// Removes a tree this run fetched from the merged root: a plain directory
/// only (a volume tree is a symlink there and is never removed), and never
/// with the root on the volume ([`weights::check_off_volume`]).
fn remove_fetched(cfg: &RunConfig, root: &Path, dest: &str) -> Result<(), String> {
    weights::check_off_volume(&cfg.scratch, root, &cfg.work)?;
    let p = root.join(dest);
    match fs::symlink_metadata(&p) {
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(e) => Err(format!("{}: {e}", p.display())),
        Ok(m) if m.file_type().is_dir() => {
            fs::remove_dir_all(&p).map_err(|e| format!("{}: {e}", p.display()))
        }
        Ok(_) => Err(format!(
            "{} is not a fetched tree (not a directory): left as it is",
            p.display()
        )),
    }
}

/// Now, in unix seconds.
fn now_unix() -> f64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0)
}

/// The last non-empty line of a log (at most 240 characters).
fn last_line(p: &Path) -> String {
    fs::read_to_string(p)
        .unwrap_or_default()
        .lines()
        .rev()
        .find(|l| !l.trim().is_empty())
        .unwrap_or("")
        .chars()
        .take(240)
        .collect()
}

fn append(p: &Path) -> Result<File, RunError> {
    OpenOptions::new()
        .create(true)
        .append(true)
        .open(p)
        .map_err(io(p))
}

/// The per-arm frames record (docs/adr/0019-fv-lab.md §8): what both runners
/// write after an arm and both fetches take, so two runs of one commit are
/// compared byte for byte without fetching frames.
pub const FRAMES_DIGEST: &str = "frames.sha256.json";

/// `<arm>/frames.sha256.json`: every `*.png` file under `<arm>/frames`
/// (recursive, symlinks not followed), in byte order of its path relative to
/// `frames/`, with its SHA-256, and the count. Byte-identical to
/// `runpod-matrix.sh`'s `frames_digest` (`tests/frames_digest.rs` against the
/// script's output). `None` when the arm wrote no `frames/` directory.
pub fn frames_digest(cell: &Path, arm: &str) -> Option<String> {
    use sha2::{Digest as _, Sha256};
    let frames = cell.join("frames");
    if !frames.is_dir() {
        return None;
    }
    let mut files = Vec::new();
    collect_png(&frames, "", &mut files);
    files.sort();
    let mut out = format!("{{\"arm\":\"{arm}\",\"frames\":[");
    let mut sep = "";
    for rel in &files {
        let hash = fs::read(frames.join(rel))
            .map(|b| {
                Sha256::digest(&b)
                    .iter()
                    .map(|x| format!("{x:02x}"))
                    .collect::<String>()
            })
            .unwrap_or_default();
        out.push_str(&format!(
            "{sep}\n{{\"path\":\"{rel}\",\"sha256\":\"{hash}\"}}"
        ));
        sep = ",";
    }
    out.push_str(&format!("\n],\"count\":{}}}\n", files.len()));
    Some(out)
}

/// `find . -type f -name '*.png' -printf '%P\n'`.
fn collect_png(dir: &Path, prefix: &str, out: &mut Vec<String>) {
    let Ok(rd) = fs::read_dir(dir) else { return };
    for e in rd.flatten() {
        let Ok(name) = e.file_name().into_string() else {
            continue;
        };
        let Ok(ft) = e.file_type() else { continue };
        let rel = if prefix.is_empty() {
            name.clone()
        } else {
            format!("{prefix}/{name}")
        };
        if ft.is_dir() {
            collect_png(&e.path(), &rel, out);
        } else if ft.is_file() && name.ends_with(".png") {
            out.push(rel);
        }
    }
}

/// `ln -sf "$(readlink -f "$src")" "$dst"` when `src` exists.
fn link(src: &str, dst: &Path) {
    let src = Path::new(src);
    if !src.exists() {
        return;
    }
    let target = fs::canonicalize(src).unwrap_or_else(|_| src.to_path_buf());
    let _ = fs::remove_file(dst);
    let _ = std::os::unix::fs::symlink(target, dst);
}

/// The script's failure excerpt: the last 20 stderr lines without progress
/// spam (`MiB/`, `%]`, `block N/`), then the last 12 of those.
pub fn fail_tail(stderr: &str) -> Vec<&str> {
    let lines: Vec<&str> = stderr.lines().collect();
    let last20 = &lines[lines.len().saturating_sub(20)..];
    let kept: Vec<&str> = last20
        .iter()
        .copied()
        .filter(|l| !(l.contains("MiB/") || l.contains("%]") || has_block_progress(l)))
        .collect();
    kept[kept.len().saturating_sub(12)..].to_vec()
}

/// `block [0-9]+/`.
fn has_block_progress(l: &str) -> bool {
    let mut rest = l;
    while let Some(i) = rest.find("block ") {
        let after = &rest[i + 6..];
        let digits = after.bytes().take_while(u8::is_ascii_digit).count();
        if digits > 0 && after.as_bytes().get(digits) == Some(&b'/') {
            return true;
        }
        rest = after;
    }
    false
}

/// `grep -E -i 'peak.*(mib|vram)|peak_mib' stdout.log stderr.log | tail -1`
/// (grep prefixes each line with its file).
fn peak_line(cell: &Path) -> String {
    let mut last = String::new();
    for f in ["stdout.log", "stderr.log"] {
        let p = cell.join(f);
        for l in fs::read_to_string(&p).unwrap_or_default().lines() {
            let low = l.to_lowercase();
            let hit = low
                .find("peak")
                .is_some_and(|i| low[i..].contains("mib") || low[i..].contains("vram"));
            if hit {
                last = format!("{}:{l}", p.display());
            }
        }
    }
    last
}

/// `grep -o '"psnr_mean":[^,]*' | tail -1`.
fn last_psnr(err: &str) -> String {
    let mut last = String::new();
    for l in err.lines() {
        let mut rest = l;
        while let Some(i) = rest.find("\"psnr_mean\":") {
            let after = &rest[i..];
            let end = after.find(',').unwrap_or(after.len());
            last = after[..end].to_owned();
            rest = &after[end..];
        }
    }
    last
}

/// The `"max":… "mean":…` of the `lpips {"backend…` lines, space-joined.
fn lpips_summary(err: &str) -> String {
    let mut out = String::new();
    for l in err.lines() {
        let Some(i) = l.find("lpips {\"backend") else {
            continue;
        };
        let seg = &l[i..];
        let seg = &seg[..seg.find('}').unwrap_or(seg.len())];
        for key in ["\"max\":", "\"mean\":"] {
            if let Some(j) = seg.find(key) {
                let v = &seg[j..];
                let end = v.find(',').unwrap_or(v.len());
                out.push_str(&v[..end]);
                out.push(' ');
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fail_tail_drops_progress_and_keeps_twelve() {
        let mut s = String::new();
        for i in 0..30 {
            s.push_str(&format!("line {i}\n"));
        }
        s.push_str("loading 12 MiB/s\n[####  ] 40%]\nblock 3/48\nreal error\n");
        let t = fail_tail(&s);
        assert_eq!(t.len(), 12);
        assert_eq!(t.last(), Some(&"real error"));
        assert!(!t
            .iter()
            .any(|l| l.contains("MiB/") || l.contains("block 3/")));
        assert!(has_block_progress("x block 12/40 y") && !has_block_progress("block x/"));
    }

    #[test]
    fn metric_excerpts() {
        let e = "a \"psnr_mean\":31.2,\"x\":1\nb \"psnr_mean\":30.0,\nlpips {\"backend\":\"alex\",\"max\":0.2,\"mean\":0.1,\"n\":3}\n";
        assert_eq!(last_psnr(e), "\"psnr_mean\":30.0");
        assert_eq!(lpips_summary(e), "\"max\":0.2 \"mean\":0.1 ");
    }

    #[test]
    fn pins_and_paths() {
        let p = read_pins("# c\nCUDA_CUDNN_SONAME=libcudnn.so.9\nX=1\n");
        assert_eq!(p["CUDA_CUDNN_SONAME"], "libcudnn.so.9");
        let mut c = RunConfig::new("rtx6000", Path::new("/opt/fastvideo-rs"));
        c.scratch = "/fvscratch".into();
        c.tag = Some("abc-1".into());
        assert_eq!(c.runs(), Path::new("/fvscratch/runs/rtx6000/abc-1"));
        // FV_RUNS_DIR (headline's sub-runs) replaces the whole path.
        let mut h = c.clone();
        h.runs_dir = Some("/fvscratch/runs/headline/abc-1".into());
        assert_eq!(h.runs(), Path::new("/fvscratch/runs/headline/abc-1"));
        assert_eq!(h.paths().runs, Path::new("/fvscratch/runs/headline/abc-1"));
        assert_eq!(c.weights(), Path::new("/workspace/weights"));
        assert_eq!(c.aux(), Path::new("/workspace/weights/auxiliary"));
        if !Path::new("/workspace/weights/auxiliary/tae/.complete").exists() {
            assert_eq!(c.tae(), Path::new("/fvscratch/tae"));
        }
        c.tae_dir = Some("/t".into());
        assert_eq!(c.paths().tae, Path::new("/t"));
        assert_eq!(c.paths().baseline, None);
        c.baseline_sha = Some("7654321".into());
        assert_eq!(
            c.paths().baseline.as_deref(),
            Some(Path::new("/fvscratch/baseline-7654321/fv-gpucheck"))
        );
    }

    /// `b4-verdict.sh`'s `b4_exit_ok` and `b4_identical` on the stub shapes
    /// of `scripts/gpu/tests/b4-verdict.test.sh` (pretty-printed reports).
    #[test]
    fn exit_and_identity_read_as_b4_verdict_sh() {
        let d = std::env::temp_dir().join(format!("fv-lab-b4-pieces-{}", std::process::id()));
        let _ = fs::create_dir_all(&d);
        let f = d.join("x.json");
        for (body, ok) in [
            ("{\"cell\":\"a\",\"exit\":0,\"seconds\":12}\n", true),
            ("{\n  \"exit\": 0\n}\n", true),
            ("{\"exit\":10,\"seconds\":1}", false),
            ("{\"exit\":01}", false),
            (
                "{\"cell\":\"a\",\"exit\":null,\"skipped\":\"weights incomplete\"}",
                false,
            ),
        ] {
            fs::write(&f, body).unwrap();
            assert_eq!(summary_exit_ok(&f), ok, "{body}");
        }
        assert!(!summary_exit_ok(&d.join("missing.json")));
        let report = |ok: &str, frames: &str, status: &str| {
            format!(
                "{{\n  \"checks\": [\n    {{\n      \"name\": \"off_identity\",\n      \"ok\": {ok},\n      \"values\": {{}}\n    }},\n    {{\n      \"name\": \"off_identity_frame_count\",\n      \"ok\": {frames}\n    }}\n  ],\n  \"status\": \"{status}\"\n}}\n"
            )
        };
        for (body, ok) in [
            (report("true", "true", "pass"), true),
            (report("false", "true", "fail"), false),
            (report("true", "false", "fail"), false),
            (
                "{\"stage\":\"compare-clips-a--b\",\"status\":\"skipped\",\"reason\":\"frames missing\"}".to_owned(),
                false,
            ),
        ] {
            fs::write(&f, &body).unwrap();
            assert_eq!(compare_identical(&f), ok, "{body}");
        }
        assert!(!compare_identical(&d.join("missing.json")));
        let _ = fs::remove_dir_all(&d);
    }

    #[test]
    fn dry_run_lines_follow_the_script() {
        let mut c = RunConfig::new("f", Path::new("/r"));
        c.bin = "/nonexistent/fv".into();
        c.only = Some(vec!["a".into()]);
        let steps = vec![
            Step::Arm {
                name: "a".into(),
                gate: "g".into(),
                trees: vec![],
                command: vec!["/nonexistent/fv".into(), "x y".into()],
                tae: None,
                baseline: false,
            },
            Step::Arm {
                name: "b".into(),
                gate: "g".into(),
                trees: vec![],
                command: vec!["/nonexistent/fv".into()],
                tae: Some("/nonexistent/tae/t.safetensors".into()),
                baseline: true,
            },
            Step::Compare {
                baseline: "a".into(),
                candidate: "b".into(),
                args: vec![],
            },
            Step::Gate {
                baseline: "a".into(),
                candidate: "b".into(),
                kind: GateKind::Lossy,
                off: None,
            },
        ];
        assert_eq!(
            dry_run_lines(&c, &steps),
            [
                "[f] matrix start image=unknown",
                "[f] dry-run a: /nonexistent/fv x\\ y ",
                "[f] skip b (not in FV_CELLS)",
                "[f] dry-run compare a--b ",
                "[f] skip gate a--b (benchmark.json missing)",
                "[f] matrix done",
            ]
        );
    }
}
