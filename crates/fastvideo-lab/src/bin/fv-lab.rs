//! `fv-lab`: the validation driver (docs/adr/0019-fv-lab.md, docs/dev/fv-lab.md).
//!
//! ```text
//! fv-lab run <tier|cell> [--dry-run] [--cells a,b] [--work D] [--scratch D] [--tag T]
//!                 [--baseline-sha <7>] [--runs-dir D]
//!                 the on-pod runner (runpod-matrix.sh's frame, the cells of lab/);
//!                 --dry-run prints each arm's command line and writes nothing
//! fv-lab check    parse and validate lab/tiers.toml and lab/cells/*.toml
//! fv-lab schema tiers|cell
//! fv-lab pod up --sha <7> --family <tier> [--yes] [--wait] [--cells …] [--extra-env K=V …]
//!                 [--baseline-sha <7>]   (required by a tier with `bin = "baseline"` arms)
//!                 [--wall-cap-s S]       (lower the tier's wall cap)
//!                 one pod of the tier on fv-deploy's guards: price cap, $8 floor, EU
//!                 volume, backstop at the tier's wall cap, ready within 600 s (3 cycles);
//!                 without --yes it prints the plan; --wait also waits, fetches and deletes
//! fv-lab pod status <id> | fetch <id> [--wait] [--down] | down <id> | reap [--yes] [--account]
//!                 reap: the pods recorded here past their cap; --account: every
//!                 fv-lab-<tier>-<sha7>-<MMDDHHMMSS>-<hex4> pod past its tier's cap
//! fv-lab gold check [--max-age-days N] [--today YYYY-MM-DD]
//!                 every gold file parses and resolves against families/, a validated
//!                 commit exists (git), and the per-variant validation age (Markdown)
//! fv-lab gold cells [--write | --check]
//!                 the `gold` tier's cells, generated from lab/gold/ and families/
//! fv-lab report tiers [--runs artifacts/runpod/<tier>/<tag>] [--check] [--out F]
//!                 docs/perf/tiers.md from the gold files and one run set
//! fv-lab backstop --kind K --id ID --deadline T --armed-file F   (started by `pod up`)
//! ```
//!
//! The Runpod key comes from `RUNPOD_API_KEY` and is never printed.

use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use clap::{Args, Parser, Subcommand};
use serde_json::json;

use fastvideo_deploy::boot::env::BootEnv;
use fastvideo_deploy::boot::http::ReqwestHttp;
use fastvideo_deploy::boot::sys::{
    Clock, Http, LogSink, RealClock, Request, Response, StderrLog, SysError,
};
use fastvideo_deploy::guard::{self, run_backstop, DetachedBackstop};
use fastvideo_deploy::launch::{ghcr_resolve, utc_stamp};
use fastvideo_deploy::lifecycle::{Deployer, Ledger, UpOptions};
use fastvideo_deploy::provider::runpod::{Pod, RunpodApi, RunpodUrls, API_KEY_VAR};
use fastvideo_deploy::provider::{
    Instance, InstanceId, InstanceState, Provider, ProviderKind, ReachOptions,
};

use fastvideo_lab::plan::{Lab, Paths, Tier};
use fastvideo_lab::pod::{
    baseline_for_pod, check_cells, check_sha7_named, fetch_run, judge_run, launch, name_taken,
    random_suffix, reap_account, reap_recorded, run_dir, run_tag, undeleted_in_ledger, wait_done,
    Done, LabPod, PodRecord, StartOptions,
};
use fastvideo_lab::run::{dry_run_lines, RunConfig, RunError, Runner};
use fastvideo_lab::schema;
use fastvideo_lab::{goldcell, goldcheck, report};

/// `-V` / `--version`: the tools release this binary was built as
/// (`FV_RELEASE_VERSION` at build time, docs/dev/tools-releases.md; the
/// release gate checks it), else the workspace version.
const VERSION: &str = match option_env!("FV_RELEASE_VERSION") {
    Some(v) => v,
    None => env!("CARGO_PKG_VERSION"),
};

#[derive(Parser)]
#[command(
    name = "fv-lab",
    version = VERSION,
    about = "The lab's validation driver: run a tier's cells on a pod, and the pod itself (docs/dev/fv-lab.md)"
)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Run a tier (or one cell) on this GPU box.
    Run(RunArgs),
    /// Parse and validate lab/tiers.toml and lab/cells/*.toml.
    Check {
        #[arg(long)]
        root: Option<PathBuf>,
    },
    /// Print a JSON Schema (tiers, cell).
    Schema { name: String },
    /// A Runpod pod for a tier.
    #[command(subcommand)]
    Pod(PodCmd),
    /// The gold files: the CI check and the `gold` tier's cells.
    #[command(subcommand)]
    Gold(GoldCmd),
    /// Reports generated from run sets.
    #[command(subcommand)]
    Report(ReportCmd),
    /// The detached backstop `pod up` starts (not for direct use).
    #[command(hide = true)]
    Backstop {
        #[arg(long)]
        kind: String,
        #[arg(long)]
        id: String,
        #[arg(long)]
        deadline: f64,
        #[arg(long)]
        armed_file: Option<PathBuf>,
    },
}

#[derive(Subcommand)]
enum GoldCmd {
    /// Every gold file parses and resolves against families/*.toml, every
    /// `last_validated.commit` is a commit here, and the validation age per
    /// variant (Markdown on stdout, for $GITHUB_STEP_SUMMARY).
    Check {
        /// Fail when a validated variant is older than this (default: warn only).
        #[arg(long)]
        max_age_days: Option<i64>,
        /// The day ages are counted to (default today, UTC).
        #[arg(long)]
        today: Option<String>,
        #[arg(long)]
        root: Option<PathBuf>,
    },
    /// The `gold` tier's cells (lab/cells/gold-<variant>.toml), generated from
    /// the gold and family files: print them, write them, or check them.
    Cells {
        #[arg(long, conflicts_with = "check")]
        write: bool,
        #[arg(long)]
        check: bool,
        #[arg(long)]
        root: Option<PathBuf>,
    },
}

#[derive(Subcommand)]
enum ReportCmd {
    /// docs/perf/tiers.md: the gold variants as one run set measured them.
    Tiers {
        /// The run set, relative to the root (artifacts/runpod/<tier>/<tag>);
        /// with --check, default the one the committed file names.
        #[arg(long)]
        runs: Option<String>,
        /// Fail when the committed file differs from the rendering.
        #[arg(long)]
        check: bool,
        /// The table (default docs/perf/tiers.md under the root).
        #[arg(long)]
        out: Option<PathBuf>,
        /// Print instead of writing.
        #[arg(long, conflicts_with = "check")]
        stdout: bool,
        /// The gold files (default lab/gold under the root).
        #[arg(long)]
        gold: Option<PathBuf>,
        #[arg(long)]
        root: Option<PathBuf>,
    },
}

#[derive(Args)]
struct RunArgs {
    /// A tier of lab/tiers.toml (`rtx6000`, `gold`) or a cell of lab/cells/.
    target: String,
    /// Every arm of the tier (`fv-lab run gold --all`): the default for a
    /// tier, said explicitly; not with --cells.
    #[arg(long, conflicts_with = "cells")]
    all: bool,
    /// Print the command lines (as runpod-matrix.sh FV_DRY_RUN=1); run and write nothing.
    #[arg(long)]
    dry_run: bool,
    /// The tier whose run directory a single cell writes into.
    #[arg(long)]
    family: Option<String>,
    /// Only these arms (FV_CELLS; comma or space separated).
    #[arg(long)]
    cells: Option<String>,
    /// The volume mount, weights under <work>/weights (FV_WORK, /workspace).
    #[arg(long)]
    work: Option<PathBuf>,
    /// Where everything is written (FV_SCRATCH, default <work>).
    #[arg(long)]
    scratch: Option<PathBuf>,
    /// The run tag: runs/<tier>/<tag> (FV_RUN_TAG).
    #[arg(long)]
    tag: Option<String>,
    /// The run directory itself, instead of <scratch>/runs/<tier>/<tag>
    /// (FV_RUNS_DIR; runpod-matrix.sh `headline` runs a tier's cells into its own).
    #[arg(long)]
    runs_dir: Option<PathBuf>,
    /// fv-gpucheck (FV_GPUCHECK).
    #[arg(long)]
    gpucheck: Option<PathBuf>,
    /// Per-arm wall cap in s, 0 = none (FV_GEN_TIMEOUT_S; default the tier's).
    #[arg(long)]
    cell_timeout_s: Option<u64>,
    /// The commit whose runtime image's fv-gpucheck the `bin = "baseline"`
    /// arms run (FV_BASELINE_SHA; fetched by scripts/gpu/fetch-baseline.sh).
    #[arg(long)]
    baseline_sha: Option<String>,
    /// The root holding lab/ and scripts/gpu (FV_LAB_ROOT; default: found above
    /// the current directory, then above this binary).
    #[arg(long)]
    root: Option<PathBuf>,
}

#[derive(Subcommand)]
enum PodCmd {
    /// Create a pod for the tier and wait until it serves its run directory.
    Up(UpArgs),
    /// The pod's state, its server and the run's last log lines.
    Status(IdArgs),
    /// Fetch the run into artifacts/runpod/<tier>/<tag>/.
    Fetch {
        #[command(flatten)]
        id: IdArgs,
        /// Wait for the run to finish first.
        #[arg(long)]
        wait: bool,
        /// Delete the pod after the fetch.
        #[arg(long)]
        down: bool,
    },
    /// Delete a pod (recorded here or not), or with --from-ledger every pod the
    /// ledger shows created and not deleted.
    Down(DownArgs),
    /// Delete the pods `pod up` recorded on this machine that are older than
    /// their wall cap (+10 min).
    Reap {
        /// Delete (without it, only list).
        #[arg(long)]
        yes: bool,
        /// Account-wide instead: every pod named fv-lab-<known tier>-<sha7>-<MMDDHHMMSS>-<4 hex>
        /// past its tier's cap, whoever created it.
        #[arg(long)]
        account: bool,
        #[arg(long)]
        root: Option<PathBuf>,
        #[arg(long)]
        state_dir: Option<PathBuf>,
    },
}

#[derive(Args)]
struct UpArgs {
    /// The commit whose runtime image the pod boots (7 hex digits).
    #[arg(long)]
    sha: String,
    /// The tier (lab/tiers.toml).
    #[arg(long)]
    family: String,
    /// Create (without it: print the plan).
    #[arg(long)]
    yes: bool,
    /// Then wait for the run, fetch it and delete the pod.
    #[arg(long)]
    wait: bool,
    /// Only these arms (FV_CELLS on the pod).
    #[arg(long, num_args = 1..)]
    cells: Vec<String>,
    /// K=V settings for the run (FV_EXTRA_ENV).
    #[arg(long, num_args = 1..)]
    extra_env: Vec<String>,
    /// LPIPS in the compares (FV_LPIPS=1).
    #[arg(long)]
    lpips: bool,
    /// The baseline commit of a tier with `bin = "baseline"` arms
    /// (FV_BASELINE_SHA on the pod; required for such a tier).
    #[arg(long)]
    baseline_sha: Option<String>,
    /// The run tag (default <sha>-<MMDDHHMMSS>-<4 random hex>).
    #[arg(long)]
    tag: Option<String>,
    /// A wall cap in seconds below the tier's `wall_cap_s` (runpod-http.sh's
    /// FV_POD_CAP_S): the backstop's deadline and the pod's own cap. Never 0
    /// and never above the tier's.
    #[arg(long)]
    wall_cap_s: Option<u64>,
    #[arg(long)]
    root: Option<PathBuf>,
    /// Ledger, pod records, backstop logs (default <root>/artifacts/fv-lab).
    #[arg(long)]
    state_dir: Option<PathBuf>,
}

#[derive(Args)]
struct DownArgs {
    /// The pod id (no record needed).
    #[arg(
        required_unless_present = "from_ledger",
        conflicts_with = "from_ledger"
    )]
    id: Option<String>,
    /// Every pod `<state>/ledger.tsv` records as created (`runpod-pod-created`)
    /// and not deleted: the cleanup after an interrupted `pod up`, whose record
    /// is only written once the pod is ready (docs/adr/0023-lab-report-and-ci-bridge.md §4).
    #[arg(long)]
    from_ledger: bool,
    #[arg(long)]
    root: Option<PathBuf>,
    #[arg(long)]
    state_dir: Option<PathBuf>,
}

#[derive(Args)]
struct IdArgs {
    /// The pod id `pod up` printed.
    id: String,
    #[arg(long)]
    root: Option<PathBuf>,
    #[arg(long)]
    state_dir: Option<PathBuf>,
}

fn env_nonempty(k: &str) -> Option<String> {
    std::env::var(k).ok().filter(|v| !v.is_empty())
}

fn find_root(arg: Option<&PathBuf>) -> Result<PathBuf, String> {
    if let Some(r) = arg
        .cloned()
        .or_else(|| env_nonempty("FV_LAB_ROOT").map(PathBuf::from))
    {
        return Ok(r);
    }
    let cwd = std::env::current_dir().map_err(|e| e.to_string())?;
    if let Some(r) = Lab::find_root(&cwd) {
        return Ok(r);
    }
    let exe = std::env::current_exe().map_err(|e| e.to_string())?;
    exe.parent().and_then(Lab::find_root).ok_or_else(|| {
        "no lab/tiers.toml above the current directory or this binary (--root)".into()
    })
}

fn print_json(v: &impl serde::Serialize) {
    match serde_json::to_string_pretty(v) {
        Ok(s) => println!("{s}"),
        Err(e) => eprintln!("[fv-lab] json: {e}"),
    }
}

fn split_list(s: &str) -> Vec<String> {
    s.split([',', ' '])
        .filter(|x| !x.is_empty())
        .map(str::to_owned)
        .collect()
}

// ------------------------------------------------------------------ run

fn cmd_run(a: &RunArgs) -> Result<ExitCode, String> {
    let root = find_root(a.root.as_ref())?;
    let lab = Lab::load(&root).map_err(|e| e.to_string())?;
    let (tier, cells) = lab
        .resolve(&a.target, a.family.as_deref())
        .map_err(|e| e.to_string())?;
    // A retired family's mode knob would be ignored: refused (ADR-0019 §11).
    fastvideo_lab::knobs::check_run_env(&tier.name, env_nonempty)?;
    if a.all && tier.name != a.target {
        return Err(format!(
            "--all runs a whole tier; `{}` is a cell (of `{}`)",
            a.target, tier.name
        ));
    }
    let mut cfg = RunConfig::new(&tier.name, &root);
    cfg.work = a
        .work
        .clone()
        .or_else(|| env_nonempty("FV_WORK").map(PathBuf::from))
        .unwrap_or_else(|| PathBuf::from("/workspace"));
    cfg.scratch = a
        .scratch
        .clone()
        .or_else(|| env_nonempty("FV_SCRATCH").map(PathBuf::from))
        .unwrap_or_else(|| cfg.work.clone());
    cfg.tag = a.tag.clone().or_else(|| env_nonempty("FV_RUN_TAG"));
    cfg.runs_dir = a
        .runs_dir
        .clone()
        .or_else(|| env_nonempty("FV_RUNS_DIR").map(PathBuf::from));
    if let Some(b) = a
        .gpucheck
        .clone()
        .or_else(|| env_nonempty("FV_GPUCHECK").map(PathBuf::from))
    {
        cfg.bin = b;
    }
    cfg.cell_timeout_s = match (a.cell_timeout_s, env_nonempty("FV_GEN_TIMEOUT_S")) {
        (Some(t), _) => t,
        (None, Some(t)) => t
            .parse()
            .map_err(|_| format!("FV_GEN_TIMEOUT_S `{t}` is not a number of seconds"))?,
        (None, None) => tier.cell_timeout_s,
    };
    cfg.only = a
        .cells
        .clone()
        .or_else(|| env_nonempty("FV_CELLS"))
        .map(|s| split_list(&s));
    cfg.lpips = env_nonempty("FV_LPIPS").as_deref() == Some("1");
    cfg.gate_policy = env_nonempty("FV_GATE_POLICY").map(PathBuf::from);
    cfg.aux_dir = env_nonempty("FV_AUX_DIR").map(PathBuf::from);
    cfg.tae_dir = env_nonempty("FV_TAE_DIR").map(PathBuf::from);
    cfg.baseline_sha = a
        .baseline_sha
        .clone()
        .or_else(|| env_nonempty("FV_BASELINE_SHA"));
    if let Some(sha) = &cfg.baseline_sha {
        check_sha7_named("--baseline-sha / FV_BASELINE_SHA", sha).map_err(|e| e.to_string())?;
    }
    if Lab::needs_baseline(&cells) && cfg.baseline_sha.is_none() {
        return Err(format!(
            "{} runs baseline arms: pass --baseline-sha <7> (FV_BASELINE_SHA), the baseline commit",
            a.target
        ));
    }
    let paths: Paths = cfg.paths();
    let steps = Lab::steps(&cells, &cfg.bin, &paths).map_err(|e| e.to_string())?;
    if a.dry_run {
        use std::io::Write as _;
        let now = RealClock.now();
        let mut out = std::io::stdout().lock();
        for l in dry_run_lines(&cfg, &steps) {
            if writeln!(out, "[{}] {l}", fastvideo_deploy::boot::sys::utc_hms(now)).is_err() {
                break;
            }
        }
        return Ok(ExitCode::SUCCESS);
    }
    for k in ["FV_PROMPTS", "FV_PROMPT", "FV_SEED"] {
        if let Some(v) = env_nonempty(k) {
            eprintln!(
                "[fv-lab] {k}={v} is ignored: a cell's prompts and seeds are its workload (lab/cells)"
            );
        }
    }
    let mut runner = Runner::new(cfg);
    match runner.run(&steps) {
        Ok(()) => Ok(ExitCode::SUCCESS),
        Err(e @ RunError::Fatal(_)) => {
            eprintln!("[fv-lab] {e}");
            Ok(ExitCode::from(2))
        }
        Err(e) => Err(e.to_string()),
    }
}

// ------------------------------------------------------------------ pod

/// The plan's HTTP: never called.
struct NoHttp;

impl Http for NoHttp {
    fn send(&self, req: &Request) -> Result<Response, SysError> {
        Err(SysError::Http {
            url: req.url.clone(),
            why: "plan makes no request".into(),
        })
    }
    fn download(&self, url: &str, _dest: &Path, _t: Duration) -> Result<u64, SysError> {
        Err(SysError::Http {
            url: url.to_owned(),
            why: "plan makes no request".into(),
        })
    }
}

/// The first SIGINT / SIGTERM sets the flag (the lifecycle deletes what it
/// created); a second one quits at once (the backstop still deletes).
fn interrupt_flag() -> Arc<AtomicBool> {
    let flag = Arc::new(AtomicBool::new(false));
    let f = Arc::clone(&flag);
    std::thread::spawn(move || {
        let Ok(rt) = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
        else {
            return;
        };
        rt.block_on(async move {
            use tokio::signal::unix::{signal, SignalKind};
            let Ok(mut term) = signal(SignalKind::terminate()) else {
                return;
            };
            tokio::select! {
                _ = tokio::signal::ctrl_c() => {}
                _ = term.recv() => {}
            }
            eprintln!("[fv-lab] interrupted: deleting what this run created (again to quit now)");
            f.store(true, Ordering::SeqCst);
            tokio::select! {
                _ = tokio::signal::ctrl_c() => {}
                _ = term.recv() => {}
            }
            eprintln!("[fv-lab] second signal: quitting; the backstop deletes at the cap");
            std::process::exit(130);
        });
    });
    flag
}

fn state_dir(arg: Option<&PathBuf>, root: &Path) -> PathBuf {
    arg.cloned()
        .unwrap_or_else(|| root.join("artifacts/fv-lab"))
}

fn write_record(path: &Path, rec: &PodRecord) -> Result<(), String> {
    if let Some(d) = path.parent() {
        std::fs::create_dir_all(d).map_err(|e| format!("{}: {e}", d.display()))?;
    }
    let text = serde_json::to_string_pretty(rec).map_err(|e| e.to_string())?;
    std::fs::write(path, text + "\n").map_err(|e| format!("{}: {e}", path.display()))
}

fn read_record(state: &Path, id: &str) -> Result<PodRecord, String> {
    let p = state.join("pods").join(format!("{id}.json"));
    let t = std::fs::read_to_string(&p).map_err(|e| {
        format!(
            "{}: {e} (pods `fv-lab pod up` created on this machine)",
            p.display()
        )
    })?;
    serde_json::from_str(&t).map_err(|e| format!("{}: {e}", p.display()))
}

fn tier<'a>(lab: &'a Lab, name: &str) -> Result<&'a Tier, String> {
    lab.tier(name)
        .ok_or_else(|| format!("no tier `{name}` in lab/tiers.toml"))
}

fn cmd_up(u: &UpArgs) -> Result<bool, String> {
    let root = find_root(u.root.as_ref())?;
    let lab = Lab::load(&root).map_err(|e| e.to_string())?;
    let t = tier(&lab, &u.family)?;
    let capped;
    let t = match u.wall_cap_s {
        None => t,
        Some(c) if c == 0 || c > t.wall_cap_s => {
            return Err(format!(
                "--wall-cap-s {c}: a tier's cap can only be lowered (1..={} s for `{}`)",
                t.wall_cap_s, t.name
            ));
        }
        Some(c) => {
            capped = Tier {
                wall_cap_s: c,
                ..t.clone()
            };
            &capped
        }
    };
    let clock = RealClock;
    let tag = match &u.tag {
        Some(t) => t.clone(),
        None => run_tag(
            &u.sha,
            clock.now(),
            random_suffix().map_err(|e| e.to_string())?,
        ),
    };
    let mut so = StartOptions::for_tier(t, &tag);
    so.cells = u.cells.iter().flat_map(|c| split_list(c)).collect();
    so.extra_env = u.extra_env.clone();
    // What no tier reads would be ignored on the pod: refused (ADR-0019 §11).
    fastvideo_lab::knobs::check_extra_env(&t.name, &so.extra_env)?;
    let cells: Vec<_> = t.cells.iter().filter_map(|c| lab.cells.get(c)).collect();
    check_cells(
        &t.name,
        cells
            .iter()
            .flat_map(|c| c.arms.iter().map(|a| a.name.as_str())),
        &so.cells,
    )
    .map_err(|e| e.to_string())?;
    baseline_for_pod(
        &u.sha,
        u.baseline_sha.as_deref(),
        &mut so.extra_env,
        Lab::needs_baseline(&cells),
    )
    .map_err(|e| format!("tier `{}`: {e}", t.name))?;
    so.lpips = u.lpips;
    let mut l = launch(t, &u.sha, &so).map_err(|e| e.to_string())?;
    guard::check_guards(&l.guards).map_err(|e| e.to_string())?;
    guard::check_region(&l).map_err(|e| e.to_string())?;
    if !u.yes {
        eprintln!("[fv-lab] dry run (add --yes to create): the plan follows");
        let api = RunpodApi::new(
            &NoHttp,
            &clock,
            "",
            RunpodUrls::from_env(&BootEnv::from_process()),
        );
        let p = LabPod {
            inner: Pod::new(api),
            family: t.name.clone(),
        };
        let calls = p.payload(&l).map_err(|e| e.to_string())?;
        print_json(&json!({
            "launch": l,
            "requests": calls,
            "note": "`--yes` pins the image to its digest; the pod runs `fv-lab run` and serves its run directory on 8000",
        }));
        return Ok(true);
    }
    let env = BootEnv::from_process();
    let http = ReqwestHttp::new().map_err(|e| e.to_string())?;
    let log = StderrLog;
    let api = RunpodApi::from_env(&http, &clock, &env).map_err(|e| e.to_string())?;
    // Never two live pods with one name: fv-deploy's exit sweep deletes by
    // exact name, so a second run would delete the first run's pod.
    let live = Pod::new(RunpodApi::from_env(&http, &clock, &env).map_err(|e| e.to_string())?)
        .list()
        .map_err(|e| e.to_string())?;
    if name_taken(&live, &l.name) {
        return Err(format!(
            "a live pod is already named {}: not creating another (pick another --tag)",
            l.name
        ));
    }
    let state = state_dir(u.state_dir.as_ref(), &root);
    std::fs::create_dir_all(&state).map_err(|e| format!("{}: {e}", state.display()))?;
    let tag_image = l.image.clone();
    l.image = ghcr_resolve(&http, &tag_image).map_err(|e| e.to_string())?;
    log.line(&format!(
        "[fv-lab] {} on {}: image {} ({tag_image})",
        l.name, t.gpu, l.image
    ));
    let api_key = env.get(API_KEY_VAR).unwrap_or("").to_owned();
    let leaks = |s: &str| !api_key.is_empty() && s.contains(api_key.as_str());
    let provider = LabPod {
        inner: Pod::new(api),
        family: t.name.clone(),
    };
    let stamp = utc_stamp(clock.now());
    let exe = std::env::current_exe().map_err(|e| e.to_string())?;
    let backstop = DetachedBackstop::new(
        exe,
        Some(state.join(format!("backstop-{stamp}.log"))),
        state.join("armed"),
    );
    let ledger = Ledger {
        path: state.join("ledger.tsv"),
    };
    let flag = interrupt_flag();
    let interrupted = || flag.load(Ordering::SeqCst);
    let d = Deployer {
        provider: &provider,
        clock: &clock,
        log: &log,
        backstop: &backstop,
        ledger: Some(&ledger),
        interrupted: &interrupted,
        leaks: &leaks,
        poll: Duration::from_secs(5),
    };
    let report = d
        .up(
            &l,
            &UpOptions {
                run_key: None,
                smoke: None,
                keep: true,
            },
        )
        .map_err(|e| e.to_string())?;
    print_json(&report);
    let Some(inst) = report.instance else {
        return Err("up returned no pod".into());
    };
    let rec = PodRecord {
        id: inst.id.to_string(),
        name: l.name.clone(),
        family: t.name.clone(),
        tag: tag.clone(),
        sha: u.sha.clone(),
        image: l.image.clone(),
        url: inst
            .url
            .clone()
            .unwrap_or_else(|| provider.inner.api.urls.pod_url(inst.id.as_str())),
        gpu: inst.gpu.clone(),
        usd_per_hr: inst.usd_per_hr,
        created_at: inst.created_at,
        cap_s: t.wall_cap_s,
    };
    write_record(&state.join("pods").join(format!("{}.json", rec.id)), &rec)?;
    let out = run_dir(&root, &t.name, &tag);
    write_record(&out.join("pod.json"), &rec)?;
    log.line(&format!(
        "[fv-lab] pod {} runs `fv-lab run {}` (tag {tag}); results: fv-lab pod fetch {} --wait --down",
        rec.id, t.name, rec.id
    ));
    if !u.wait {
        return Ok(true);
    }
    let result = finish(
        &provider,
        &http,
        &rec,
        &root,
        true,
        &clock,
        &log,
        &interrupted,
    );
    // The pod goes whatever the wait or the fetch did.
    let deleted = d.destroy_retry(&inst.id, "done").map_err(|e| {
        format!(
            "could not delete {}: {e} (the backstop deletes it at the cap)",
            inst.id
        )
    });
    after_delete(result, deleted)
}

/// A run's result once its pod's delete was attempted: the delete's error
/// wins only when the run itself had none (both are reported).
fn after_delete(result: Result<bool, String>, deleted: Result<(), String>) -> Result<bool, String> {
    match (result, deleted) {
        (r, Ok(())) => r,
        (Ok(_), Err(d)) => Err(d),
        (Err(r), Err(d)) => Err(format!("{r}; and {d}")),
    }
}

/// Wait (optionally), fetch, and judge the run as `runpod-http.sh run` does.
#[allow(clippy::too_many_arguments)]
fn finish(
    provider: &dyn Provider,
    http: &dyn Http,
    rec: &PodRecord,
    root: &Path,
    wait: bool,
    clock: &dyn Clock,
    log: &dyn LogSink,
    interrupted: &dyn Fn() -> bool,
) -> Result<bool, String> {
    let mut ok = true;
    let id = InstanceId::new(rec.id.clone());
    if wait {
        let elapsed = rec.created_at.map_or(0.0, |c| (clock.now() - c).max(0.0));
        let limit = (rec.cap_s as f64 - elapsed).max(0.0) as u64;
        match wait_done(
            http,
            provider,
            &id,
            &rec.url,
            &rec.family,
            &rec.tag,
            limit,
            clock,
            log,
            interrupted,
        ) {
            Done::Finished => {}
            other => {
                log.line(&format!("[fv-lab] the run did not finish: {other:?}"));
                ok = false;
            }
        }
    }
    let dest = run_dir(root, &rec.family, &rec.tag);
    fetch_run(http, &rec.url, &rec.family, &rec.tag, &dest, clock, log)
        .map_err(|e| e.to_string())?;
    // A success needs the fetched DONE and `matrix_exit=0` (pod::judge_run).
    for problem in judge_run(&dest, &rec.tag) {
        log.line(&format!("[fv-lab] {problem}"));
        ok = false;
    }
    Ok(ok)
}

fn with_pod<T>(
    id: &IdArgs,
    f: impl FnOnce(&LabPod<'_>, &ReqwestHttp, PodRecord, &Path, &Path) -> Result<T, String>,
) -> Result<T, String> {
    let root = find_root(id.root.as_ref())?;
    let state = state_dir(id.state_dir.as_ref(), &root);
    let rec = read_record(&state, &id.id)?;
    let env = BootEnv::from_process();
    let http = ReqwestHttp::new().map_err(|e| e.to_string())?;
    let http2 = ReqwestHttp::new().map_err(|e| e.to_string())?;
    let api = RunpodApi::from_env(&http, &RealClock, &env).map_err(|e| e.to_string())?;
    let p = LabPod {
        inner: Pod::new(api),
        family: rec.family.clone(),
    };
    f(&p, &http2, rec, &root, &state)
}

fn deployer_down(p: &dyn Provider, state: &Path, id: &InstanceId) -> Result<(), String> {
    let ledger = Ledger {
        path: state.join("ledger.tsv"),
    };
    let never = || false;
    let leaks = |_: &str| false;
    let none = DetachedBackstop::new(PathBuf::new(), None, state.join("armed"));
    let d = Deployer {
        provider: p,
        clock: &RealClock,
        log: &StderrLog,
        backstop: &none,
        ledger: Some(&ledger),
        interrupted: &never,
        leaks: &leaks,
        poll: Duration::from_secs(5),
    };
    d.destroy_retry(id, "down").map_err(|e| e.to_string())
}

/// `pod down <id>` or `pod down --from-ledger`: deletes by id through
/// fv-deploy's retries and ledger; no pod record is needed (a pod is deleted
/// by id, and a 404 counts as deleted).
fn cmd_down(a: &DownArgs) -> Result<bool, String> {
    let root = find_root(a.root.as_ref())?;
    let state = state_dir(a.state_dir.as_ref(), &root);
    let ids = match (&a.id, a.from_ledger) {
        (Some(id), false) => vec![id.clone()],
        (None, true) => {
            let p = state.join("ledger.tsv");
            let text = match std::fs::read_to_string(&p) {
                Ok(t) => t,
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => String::new(),
                Err(e) => return Err(format!("{}: {e}", p.display())),
            };
            undeleted_in_ledger(&text)
        }
        _ => return Err("give a pod id or --from-ledger".into()),
    };
    if ids.is_empty() {
        println!("no pod to delete (the ledger shows none created and not deleted)");
        return Ok(true);
    }
    let env = BootEnv::from_process();
    let http = ReqwestHttp::new().map_err(|e| e.to_string())?;
    let api = RunpodApi::from_env(&http, &RealClock, &env).map_err(|e| e.to_string())?;
    // The family only matters to readiness (`reach`), which a delete never asks.
    let p = LabPod {
        inner: Pod::new(api),
        family: String::new(),
    };
    let mut ok = true;
    for id in ids {
        match deployer_down(&p, &state, &InstanceId::new(id.clone())) {
            Ok(()) => println!("deleted {id}"),
            Err(e) => {
                eprintln!("[fv-lab] pod {id}: {e}");
                ok = false;
            }
        }
    }
    Ok(ok)
}

fn cmd_pod(c: &PodCmd) -> Result<bool, String> {
    match c {
        PodCmd::Up(u) => cmd_up(u),
        PodCmd::Status(id) => with_pod(id, |p, http, rec, _, _| {
            let iid = InstanceId::new(rec.id.clone());
            let st = p.status(&iid).map_err(|e| e.to_string())?;
            let inst = Instance {
                id: iid,
                provider: ProviderKind::RunpodPod,
                name: rec.name.clone(),
                state: InstanceState::Unknown,
                url: Some(rec.url.clone()),
                gpu: rec.gpu.clone(),
                data_center: None,
                usd_per_hr: rec.usd_per_hr,
                created_at: rec.created_at,
            };
            let ping = p
                .reach(&inst, &ReachOptions::default())
                .map(|mut r| format!("{:?}", r.ping()))
                .unwrap_or_else(|e| e.to_string());
            let tail: Vec<String> = http
                .send(&Request::new(
                    fastvideo_deploy::boot::sys::Method::Get,
                    format!("{}/{}/{}/live.log", rec.url, rec.family, rec.tag),
                    Duration::from_secs(30),
                ))
                .ok()
                .filter(|r| r.ok())
                .map(|r| {
                    let t = String::from_utf8_lossy(&r.body).into_owned();
                    let v: Vec<&str> = t.lines().collect();
                    v[v.len().saturating_sub(5)..]
                        .iter()
                        .map(|s| (*s).to_owned())
                        .collect()
                })
                .unwrap_or_default();
            print_json(
                &json!({"id": rec.id, "family": rec.family, "tag": rec.tag, "state": st, "server": ping, "live_log": tail}),
            );
            Ok(true)
        }),
        PodCmd::Fetch { id, wait, down } => with_pod(id, |p, http, rec, root, state| {
            let flag = interrupt_flag();
            let interrupted = || flag.load(Ordering::SeqCst);
            let result = finish(
                p,
                http,
                &rec,
                root,
                *wait,
                &RealClock,
                &StderrLog,
                &interrupted,
            );
            if !*down {
                return result;
            }
            // --down deletes whatever the wait or the fetch did.
            after_delete(
                result,
                deployer_down(p, state, &InstanceId::new(rec.id.clone())),
            )
        }),
        PodCmd::Down(a) => cmd_down(a),
        PodCmd::Reap {
            yes,
            account,
            root,
            state_dir: sd,
        } => {
            let root = find_root(root.as_ref())?;
            let lab = Lab::load(&root).map_err(|e| e.to_string())?;
            let state = state_dir(sd.as_ref(), &root);
            let env = BootEnv::from_process();
            let http = ReqwestHttp::new().map_err(|e| e.to_string())?;
            let api = RunpodApi::from_env(&http, &RealClock, &env).map_err(|e| e.to_string())?;
            let pod = Pod::new(api);
            let pods = pod.list().map_err(|e| e.to_string())?;
            let now = RealClock.now();
            let old = if *account {
                let caps: Vec<(&str, u64)> = lab
                    .tiers
                    .tiers
                    .iter()
                    .map(|t| (t.name.as_str(), t.wall_cap_s))
                    .collect();
                reap_account(&pods, now, &caps)
            } else {
                let records = read_records(&state)?;
                reap_recorded(&pods, now, &records)
            };
            let mut ok = true;
            for p in &old {
                let age = p.created_at.map_or(0.0, |t| now - t);
                if *yes {
                    match deployer_down(&pod, &state, &p.id) {
                        Ok(()) => println!("reaped {} ({}, {age:.0} s old)", p.id, p.name),
                        Err(e) => {
                            eprintln!("[fv-lab] reap of {} failed: {e}", p.id);
                            ok = false;
                        }
                    }
                } else {
                    println!(
                        "would reap {} ({}, {age:.0} s old; --yes deletes)",
                        p.id, p.name
                    );
                }
            }
            if old.is_empty() {
                println!(
                    "nothing to reap ({})",
                    if *account {
                        "no fv-lab-<tier>-<sha7>-<stamp>-<suffix> pod past its tier's cap"
                    } else {
                        "no pod recorded here is past its cap; --account for the account-wide sweep"
                    }
                );
            }
            Ok(ok)
        }
    }
}

/// Every pod record of this machine (`<state>/pods/*.json`).
fn read_records(state: &Path) -> Result<Vec<PodRecord>, String> {
    let dir = state.join("pods");
    let Ok(rd) = std::fs::read_dir(&dir) else {
        return Ok(Vec::new());
    };
    let mut out = Vec::new();
    for e in rd.flatten() {
        let p = e.path();
        if p.extension().is_some_and(|x| x == "json") {
            let t = std::fs::read_to_string(&p).map_err(|e| format!("{}: {e}", p.display()))?;
            out.push(serde_json::from_str(&t).map_err(|e| format!("{}: {e}", p.display()))?);
        }
    }
    Ok(out)
}

fn cmd_backstop(
    kind: &str,
    id: &str,
    deadline: f64,
    armed: Option<&PathBuf>,
) -> Result<bool, String> {
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(|e| e.to_string())?;
    let (hup, int) = rt.block_on(async {
        use tokio::signal::unix::{signal, SignalKind};
        (
            signal(SignalKind::hangup()),
            signal(SignalKind::interrupt()),
        )
    });
    let _guards = (
        hup.map_err(|e| format!("SIGHUP handler: {e}"))?,
        int.map_err(|e| format!("SIGINT handler: {e}"))?,
    );
    if ProviderKind::parse(kind) != Some(ProviderKind::RunpodPod) {
        return Err(format!("fv-lab backstops Runpod pods only, not {kind}"));
    }
    let env = BootEnv::from_process();
    let http = ReqwestHttp::new().map_err(|e| e.to_string())?;
    let clock = RealClock;
    let api = RunpodApi::from_env(&http, &clock, &env).map_err(|e| e.to_string())?;
    let p = Pod::new(api);
    if let Some(f) = armed {
        guard::report_armed(f).map_err(|e| format!("{}: {e}", f.display()))?;
    }
    eprintln!("[backstop] armed (pid {})", std::process::id());
    run_backstop(&p, &InstanceId::new(id), deadline, &clock, &StderrLog)
        .map(|_| true)
        .map_err(|e| e.to_string())
}

fn run() -> Result<ExitCode, String> {
    let cli = Cli::parse();
    let b = |ok: bool| {
        if ok {
            ExitCode::SUCCESS
        } else {
            ExitCode::FAILURE
        }
    };
    match cli.cmd {
        Cmd::Run(a) => cmd_run(&a),
        Cmd::Check { root } => {
            let root = find_root(root.as_ref())?;
            let lab = Lab::load(&root).map_err(|e| e.to_string())?;
            for t in &lab.tiers.tiers {
                let arms: usize = t
                    .cells
                    .iter()
                    .filter_map(|c| lab.cells.get(c))
                    .map(|c| c.arms.len())
                    .sum();
                println!(
                    "{}: {} cell(s), {arms} arm(s), {} at <= ${}/hr, cap {} s",
                    t.name,
                    t.cells.len(),
                    t.gpu,
                    t.max_usd_per_hr,
                    t.wall_cap_s
                );
            }
            Ok(ExitCode::SUCCESS)
        }
        Cmd::Schema { name } => {
            let s =
                schema::by_name(&name).ok_or_else(|| format!("no schema {name} (tiers, cell)"))?;
            print!("{}", schema::schema_text(&s));
            Ok(ExitCode::SUCCESS)
        }
        Cmd::Pod(c) => cmd_pod(&c).map(b),
        Cmd::Gold(c) => cmd_gold(&c),
        Cmd::Report(c) => cmd_report(&c),
        Cmd::Backstop {
            kind,
            id,
            deadline,
            armed_file,
        } => cmd_backstop(&kind, &id, deadline, armed_file.as_ref()).map(b),
    }
}

// ------------------------------------------------------------------ gold, report

fn load_golds(root: &Path) -> Result<Vec<fastvideo_lab::Gold>, String> {
    fastvideo_lab::load_dir(&root.join("lab/gold")).map_err(|e| e.to_string())
}

fn load_families(root: &Path) -> Result<fastvideo_manifest::FamilySet, String> {
    fastvideo_manifest::family::load_dir(&root.join("families")).map_err(|e| e.to_string())
}

fn cmd_gold(c: &GoldCmd) -> Result<ExitCode, String> {
    match c {
        GoldCmd::Check {
            max_age_days,
            today,
            root,
        } => {
            let root = find_root(root.as_ref())?;
            let golds = load_golds(&root)?;
            let families = load_families(&root)?;
            let today = match today {
                Some(t) => t.clone(),
                None => fastvideo_deploy::launch::utc_iso(RealClock.now())
                    .get(..10)
                    .unwrap_or_default()
                    .to_owned(),
            };
            let git = goldcheck::GitCli::detect(&root);
            let git_ref: Result<&dyn goldcheck::Git, String> = match &git {
                Ok(g) => Ok(g),
                Err(e) => Err(e.clone()),
            };
            let rep = goldcheck::check(&golds, &families, git_ref, &today, *max_age_days)?;
            print!("{}", rep.markdown());
            for w in &rep.warnings {
                eprintln!("[fv-lab] warning: {w}");
            }
            for e in &rep.errors {
                eprintln!("[fv-lab] error: {e}");
            }
            Ok(if rep.ok() {
                ExitCode::SUCCESS
            } else {
                ExitCode::FAILURE
            })
        }
        GoldCmd::Cells { write, check, root } => {
            let root = find_root(root.as_ref())?;
            let cells = goldcell::gold_cells(&load_golds(&root)?, &load_families(&root)?)?;
            let dir = root.join("lab/cells");
            let committed = committed_gold_cells(&dir)?;
            let tiers_path = root.join(fastvideo_lab::plan::TIERS_PATH);
            let tiers_text = std::fs::read_to_string(&tiers_path)
                .map_err(|e| format!("{}: {e}", tiers_path.display()))?;
            let tiers = fastvideo_lab::Tiers::parse(&tiers_text, fastvideo_lab::plan::TIERS_PATH)
                .map_err(|e| e.to_string())?;
            let tier_cells = tiers
                .tiers
                .iter()
                .find(|t| t.name == goldcell::GOLD_TIER)
                .map(|t| t.cells.as_slice());
            if *write {
                for (name, text) in &cells {
                    let p = dir.join(format!("{name}.toml"));
                    std::fs::write(&p, text).map_err(|e| format!("{}: {e}", p.display()))?;
                    println!("wrote lab/cells/{name}.toml");
                }
                for name in committed.keys() {
                    if !cells.iter().any(|(n, _)| n == name) {
                        let p = dir.join(format!("{name}.toml"));
                        std::fs::remove_file(&p).map_err(|e| format!("{}: {e}", p.display()))?;
                        println!("removed lab/cells/{name}.toml (no gold file generates it)");
                    }
                }
                let now = committed_gold_cells(&dir)?;
                let rest = goldcell::drift(&now, tier_cells, &cells);
                for d in &rest {
                    eprintln!("[fv-lab] {d}");
                }
                return Ok(if rest.is_empty() {
                    ExitCode::SUCCESS
                } else {
                    ExitCode::FAILURE
                });
            }
            if *check {
                let d = goldcell::drift(&committed, tier_cells, &cells);
                for x in &d {
                    eprintln!("[fv-lab] {x} (fv-lab gold cells --write)");
                }
                if d.is_empty() {
                    println!("{} gold cells current", cells.len());
                }
                return Ok(if d.is_empty() {
                    ExitCode::SUCCESS
                } else {
                    ExitCode::FAILURE
                });
            }
            for (name, text) in &cells {
                println!("# ---- lab/cells/{name}.toml\n{text}");
            }
            Ok(ExitCode::SUCCESS)
        }
    }
}

/// Every committed `lab/cells/gold-*.toml`, by stem.
fn committed_gold_cells(dir: &Path) -> Result<std::collections::BTreeMap<String, String>, String> {
    let mut out = std::collections::BTreeMap::new();
    let rd = std::fs::read_dir(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    for e in rd {
        let p = e.map_err(|e| format!("{}: {e}", dir.display()))?.path();
        let Some(stem) = p.file_stem().and_then(|s| s.to_str()) else {
            continue;
        };
        if stem.starts_with("gold-") && p.extension().is_some_and(|x| x == "toml") {
            let text = std::fs::read_to_string(&p).map_err(|e| format!("{}: {e}", p.display()))?;
            out.insert(stem.to_owned(), text);
        }
    }
    Ok(out)
}

fn cmd_report(c: &ReportCmd) -> Result<ExitCode, String> {
    match c {
        ReportCmd::Tiers {
            runs,
            check,
            out,
            stdout,
            gold,
            root,
        } => {
            let root = find_root(root.as_ref())?;
            let golds = match gold {
                Some(d) => fastvideo_lab::load_dir(d).map_err(|e| e.to_string())?,
                None => load_golds(&root)?,
            };
            let out = out.clone().unwrap_or_else(|| root.join(report::TIERS_MD));
            let committed = if *check {
                Some(std::fs::read_to_string(&out).map_err(|e| format!("{}: {e}", out.display()))?)
            } else {
                None
            };
            let label = match (runs, &committed) {
                (Some(r), _) => Some(r.clone()),
                (None, Some(text)) => {
                    report::run_set_of(text).map_err(|e| format!("{}: {e}", out.display()))?
                }
                (None, None) => None,
            };
            let set = label
                .as_deref()
                .map(|l| report::RunSet::load(&root, l))
                .transpose()?;
            let text = report::render(&golds, set.as_ref());
            if let Some(committed) = committed {
                return Ok(match report::first_difference(&committed, &text) {
                    None => {
                        println!("{} is current ({} gold files)", out.display(), golds.len());
                        ExitCode::SUCCESS
                    }
                    Some((line, ours, theirs)) => {
                        eprintln!(
                            "[fv-lab] {} differs from `fv-lab report tiers` at line {line}:\n  committed: {ours}\n  rendered:  {theirs}\nrerun `fv-lab report tiers{}` and commit it",
                            out.display(),
                            label.map(|l| format!(" --runs {l}")).unwrap_or_default()
                        );
                        ExitCode::FAILURE
                    }
                });
            }
            if *stdout {
                print!("{text}");
            } else {
                std::fs::write(&out, &text).map_err(|e| format!("{}: {e}", out.display()))?;
                println!("wrote {}", out.display());
            }
            Ok(ExitCode::SUCCESS)
        }
    }
}

fn main() -> ExitCode {
    match run() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("[fv-lab] FAILED: {e}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::after_delete;

    #[test]
    fn the_delete_is_reported_with_the_runs_own_result() {
        assert_eq!(after_delete(Ok(true), Ok(())), Ok(true));
        assert_eq!(after_delete(Ok(false), Ok(())), Ok(false));
        assert_eq!(
            after_delete(Err("fetch".into()), Ok(())),
            Err("fetch".into())
        );
        assert_eq!(after_delete(Ok(true), Err("del".into())), Err("del".into()));
        assert_eq!(
            after_delete(Err("fetch".into()), Err("del".into())),
            Err("fetch; and del".into())
        );
    }
}
