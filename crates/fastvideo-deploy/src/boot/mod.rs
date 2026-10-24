//! `fv-boot`: the one boot sequence of a serve container (docs/adr/0014-fv-boot.md;
//! review §7.4 C2). It replaces the shell boots that differed per provider
//! (`deploy/runpod/fv-entry.sh`, fv-control's `WORKER_BOOT`,
//! `EDGE_WORKER_BOOT`, `WATCHDOG_WORKER_BOOT`, `PROVIDER_BOOT`, `SLS_BOOT`,
//! `SLS_R2_BOOT`, `SLS_HUB_BOOT` and their snippets) with typed steps:
//!
//! 1. **start** marker, then the **driver check** ([`driver`]);
//! 2. the **volume** (the serverless link, `FV_WEIGHTS`, the timeline marker)
//!    and the state directory;
//! 3. **config materialization** ([`config`]) and `fv-serve --print-config`;
//! 4. the prompt upsampler **sidecar** start ([`sidecar`]);
//! 5. the **weights plan** ([`weights`]: `volume`, `r2` through the existing
//!    `r2-fetch.py`, `hub`, `none`; `registry` is typed and not available until
//!    the registry's `hydrate` crate, model-registry-rs WO-002);
//! 6. the **tunnel** of a GMI / Brev pod ([`tunnel`]) and its **report**;
//! 7. the deadline / balance **watchdog** ([`watchdog`], a process of its own);
//! 8. the sidecar wait, then **exec** fv-serve ([`ExecPlan`]).
//!
//! Every step runs on the seams in [`sys`], so the tests drive the whole
//! sequence with a fake provider (fake processes, HTTP and clock).

pub mod config;
pub mod driver;
pub mod env;
pub mod error;
pub mod plan;
pub mod report;
pub mod sidecar;
pub mod sys;
pub mod tunnel;
pub mod watchdog;
pub mod weights;

#[cfg(feature = "boot")]
pub mod http;

#[cfg(test)]
mod tests;

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::Duration;

pub use env::{BootEnv, Kind, Paths, Provider};
pub use error::BootError;
pub use report::{Log, Reporter};
pub use sys::{Clock, Cmd, Http, LogSink, Spawner};

/// fv-boot's version: the tools release being built (`FV_RELEASE_VERSION`,
/// which scripts/dev/release-artifacts-pod.sh exports and the tools-release.sh
/// gates check: `fv-boot version` = `fv-boot <release version>`), else the
/// package version, as fastvideo-serve's build.rs does. rustc records the
/// `option_env!` read, so cargo rebuilds when it changes.
pub const VERSION: &str = match option_env!("FV_RELEASE_VERSION") {
    Some(v) => v,
    None => env!("CARGO_PKG_VERSION"),
};

/// The marker a fv-serve built with the streaming loader carries
/// (docs/ops/r2-weights.md §11): the boot only streams into a binary that has it.
pub const STREAMING_FETCH_MARKER: &str = "fv-streaming-fetch/1";

/// The machine the boot runs on.
#[derive(Clone, Copy)]
pub struct Deps<'a> {
    pub spawner: &'a dyn Spawner,
    pub http: &'a dyn Http,
    pub clock: &'a dyn Clock,
    pub log: &'a dyn LogSink,
}

/// What the boot ends in: `exec` of this program with these arguments and
/// exactly this environment.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExecPlan {
    pub program: PathBuf,
    pub args: Vec<String>,
    pub env: BTreeMap<String, String>,
}

/// The state every step shares.
pub struct Ctx<'a> {
    /// The environment the boot started with.
    pub env: BootEnv,
    /// What the boot exports to the processes it starts and to fv-serve.
    pub exports: BTreeMap<String, String>,
    /// Variables fv-serve must not see (`FV_TUNNEL_TOKEN`).
    pub unset: Vec<String>,
    pub kind: Kind,
    pub deps: Deps<'a>,
    pub paths: Paths,
    pub log: Log<'a>,
    pub rep: Reporter,
    /// The driver check's result line (`driver 595.91.07 ok`, or why it did not run).
    pub driver: String,
    /// The pid fv-serve will have (`exec` keeps it): the streaming fetch's
    /// guard stops it on a failure.
    pub pid: u32,
}

impl<'a> Ctx<'a> {
    pub fn new(env: BootEnv, deps: Deps<'a>, paths: Paths, pid: u32) -> Self {
        let t0 = deps.clock.now();
        Self::with_t0(env, deps, paths, pid, t0)
    }

    /// A context whose clock started at `t0` (the fetch guard: the boot's own
    /// start, so its elapsed times and its serverless boot id match the boot's).
    pub fn with_t0(env: BootEnv, deps: Deps<'a>, paths: Paths, pid: u32, t0: f64) -> Self {
        let kind = env.kind();
        let rep = Reporter::from_env(&env, t0);
        Self {
            env,
            exports: BTreeMap::new(),
            unset: watchdog::BACKSTOP_SECRETS
                .iter()
                .map(|k| (*k).to_owned())
                .collect(),
            kind,
            deps,
            paths,
            log: Log::new(deps.log, deps.clock, t0),
            rep,
            driver: "driver not checked".into(),
            pid,
        }
    }

    /// A variable as the boot sees it now: its own export first.
    pub fn get(&self, k: &str) -> Option<&str> {
        match self.exports.get(k) {
            Some(v) if !v.is_empty() => Some(v.as_str()),
            Some(_) => None,
            None => self.env.get(k),
        }
    }

    pub fn get_or<'b>(&'b self, k: &str, d: &'b str) -> &'b str {
        self.get(k).unwrap_or(d)
    }

    pub fn export(&mut self, k: &str, v: impl Into<String>) {
        self.exports.insert(k.to_owned(), v.into());
    }

    /// A program with the boot's exports in its environment.
    pub fn cmd(&self, program: impl Into<String>) -> Cmd {
        let mut c = Cmd::new(program);
        for (k, v) in &self.exports {
            c = c.env(k.clone(), v.clone());
        }
        for k in &self.unset {
            c = c.env_remove(k.clone());
        }
        c
    }

    pub fn now(&self) -> f64 {
        self.deps.clock.now()
    }

    /// The state directory (`FV_STATE_DIR`, default `/fvstate`).
    pub fn state_dir(&self) -> PathBuf {
        self.get("FV_STATE_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|| self.paths.state.clone())
    }

    /// The weights root (`FV_WEIGHTS`, default the pods' mount).
    pub fn weights_dir(&self) -> PathBuf {
        self.get("FV_WEIGHTS")
            .map(PathBuf::from)
            .unwrap_or_else(|| self.paths.workspace_weights.clone())
    }

    /// The serve binary (`FV_SERVE_BIN`, default the image's).
    pub fn serve_bin(&self) -> PathBuf {
        self.get("FV_SERVE_BIN")
            .map(PathBuf::from)
            .unwrap_or_else(|| self.paths.serve_bin.clone())
    }

    /// A timed step: logged, and reported to a serverless sink.
    pub fn step(&self, name: &str, since: f64, note: &str) {
        let secs = (self.now() - since).max(0.0);
        self.log.step(name, secs, note);
        self.rep
            .step(self.deps.http, self.deps.clock, name, secs, note);
    }

    pub fn say(&self, m: &str) {
        self.log.say(m);
    }

    /// A pod-protocol report.
    pub fn pod_report(&self, phase: &str, url: &str, detail: &str) {
        self.rep
            .pod_phase(self.deps.http, self.deps.clock, phase, url, detail);
    }

    /// A serverless-protocol report.
    pub fn sls_report(&self, phase: &str, detail: &str) {
        self.rep
            .sls_phase(self.deps.http, self.deps.clock, phase, detail);
    }

    /// The pause before a failed boot exits: `FV_FAIL_SLEEP_S`, else 600 s on
    /// a GMI / Brev pod (PROVIDER_BOOT: the provider keeps the log readable)
    /// and 30 s elsewhere; the driver check's `FV_DRIVER_FAIL_SLEEP_S` (20 s).
    pub fn fail_sleep(&self, e: &BootError) -> Duration {
        let s = if e.is_driver() {
            self.env.secs("FV_DRIVER_FAIL_SLEEP_S", 20.0)
        } else {
            let d = if matches!(self.kind, Kind::Provider(_)) {
                600.0
            } else {
                30.0
            };
            self.env.secs("FV_FAIL_SLEEP_S", d)
        };
        Duration::from_secs_f64(s)
    }

    /// A failed boot: the `FAILED:` line, the `failed` report, the pause.
    /// Returns the exit code.
    pub fn fail(&self, e: &BootError) -> i32 {
        let msg = e.to_string();
        self.say(&format!("FAILED: {msg}"));
        self.rep.failed(self.deps.http, self.deps.clock, &msg);
        self.deps.clock.sleep(self.fail_sleep(e));
        1
    }
}

fn mkdir(p: &Path) -> Result<(), BootError> {
    std::fs::create_dir_all(p).map_err(|e| BootError::Sys(sys::SysError::io(p, e)))
}

/// The driver check (`FV_SKIP_DRIVER_CHECK=1` skips it).
fn driver_step(c: &mut Ctx<'_>) -> Result<(), BootError> {
    if c.env.flag("FV_SKIP_DRIVER_CHECK") {
        c.say("driver check skipped (FV_SKIP_DRIVER_CHECK=1)");
        return Ok(());
    }
    let fb: Vec<&Path> = driver::LIBCUDA_FALLBACKS.iter().map(Path::new).collect();
    let facts = driver::facts(&c.env, c.deps.spawner, &fb);
    match driver::verdict(&facts) {
        Ok(line) => {
            c.say(&line);
            c.driver = line;
            Ok(())
        }
        Err(why) => Err(BootError::DriverTooOld(why)),
    }
}

/// The volume: a serverless worker's mount linked where the trees expect it,
/// `FV_WEIGHTS` defaulted to the root that exists, the timeline marker.
fn volume_step(c: &mut Ctx<'_>) {
    let rv = c.paths.runpod_volume_weights.clone();
    let ws = c.paths.workspace_weights.clone();
    if rv.is_dir() && std::fs::symlink_metadata(&ws).is_err() {
        if let Some(parent) = ws.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        match std::os::unix::fs::symlink(&rv, &ws) {
            Ok(()) => c.say(&format!(
                "volume: linked {} -> {}",
                ws.display(),
                rv.display()
            )),
            Err(e) => c.say(&format!(
                "volume: cannot link {} -> {}: {e}",
                ws.display(),
                rv.display()
            )),
        }
    }
    if c.get("FV_WEIGHTS").is_none() {
        if rv.is_dir() {
            c.export("FV_WEIGHTS", rv.display().to_string());
        } else if ws.is_dir() {
            c.export("FV_WEIGHTS", ws.display().to_string());
        }
    }
    let d = c.weights_dir();
    match std::fs::read_dir(&d) {
        Ok(rd) => {
            let n = rd
                .filter_map(Result::ok)
                .filter(|e| !e.file_name().to_string_lossy().starts_with('.'))
                .count();
            c.log
                .marker(&format!("volume: {} ({n} trees)", d.display()));
        }
        Err(_) => c.log.marker(&format!("volume: {} missing", d.display())),
    }
}

/// `fv-serve … --print-config`: a config fv-serve refuses fails the boot
/// before any download (PROVIDER_BOOT's preflight, now everywhere).
fn config_check(c: &Ctx<'_>, m: &config::Materialized) -> Result<(), BootError> {
    let bin = c.serve_bin();
    if !sys::is_executable(&bin) {
        return Err(BootError::Config(format!(
            "no {} in the image",
            bin.display()
        )));
    }
    let o = c.deps.spawner.output(
        &c.cmd(bin.display().to_string())
            .args(m.serve_args.iter().cloned())
            .arg("--print-config"),
    )?;
    if !o.ok() {
        let err = o.stderr_text();
        // fv-serve's own lines, verbatim (`fv-serve: config: …`): fv-control's
        // boot diagnosis (podlogs.ts CONTAINER_FATAL) keys on them.
        for l in err.lines().filter(|l| !l.trim().is_empty()) {
            c.log.raw(l);
        }
        let tail: Vec<&str> = err
            .lines()
            .rev()
            .take(3)
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect();
        let tail = tail.join(" ").replace('"', " ");
        return Err(BootError::Config(format!("fv-serve config: {tail}")));
    }
    Ok(())
}

/// The trees an old-form config names that are missing (fv-entry.sh's
/// preflight: a warning; fv-serve reports the error itself).
fn tree_preflight(c: &Ctx<'_>, m: &config::Materialized) {
    let Some(p) = &m.path else { return };
    let Ok(text) = std::fs::read_to_string(p) else {
        return;
    };
    let root = c.weights_dir();
    for t in config::weight_trees(&text) {
        if !root.join(&t).exists() {
            c.say(&format!(
                "variant {}: weights tree {}/{t} is missing",
                c.get_or("FV_VARIANT", "?"),
                root.display()
            ));
        }
    }
}

/// `FV_BOOT_LEGACY=1`: the rollback switch on every path (ADR-0014 §6). fv-control's shell
/// boots skip fv-boot with it; the image entrypoint (serverless without a boot override,
/// v2 load balancers, GCP / CloudRift `docker run`, Vast) gets the old `fv-entry.sh`
/// behaviour here: the serverless volume link, the `FV_WEIGHTS` default, the upsampler
/// sidecar, then exec fv-serve with the arguments as given. No driver check, config
/// step, weights plan, tunnel, watchdog or report.
pub fn legacy_requested(env: &BootEnv) -> bool {
    env.flag("FV_BOOT_LEGACY")
}

/// The `FV_BOOT_LEGACY=1` entrypoint (see [`legacy_requested`]).
pub fn legacy(c: &mut Ctx<'_>, args: &[String]) -> ExecPlan {
    c.say("FV_BOOT_LEGACY=1: fv-entry's steps only (volume link, FV_WEIGHTS, upsampler), then fv-serve");
    let rv = c.paths.runpod_volume_weights.clone();
    let ws = c.paths.workspace_weights.clone();
    if rv.is_dir() && std::fs::symlink_metadata(&ws).is_err() {
        if let Some(parent) = ws.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let _ = std::os::unix::fs::symlink(&rv, &ws);
    }
    if c.get("FV_WEIGHTS").is_none() {
        if rv.is_dir() {
            c.export("FV_WEIGHTS", rv.display().to_string());
        } else if ws.is_dir() {
            c.export("FV_WEIGHTS", ws.display().to_string());
        }
    }
    sidecar::start(c);
    sidecar::wait(c);
    let mut env = c.env.vars().clone();
    for (k, v) in &c.exports {
        env.insert(k.clone(), v.clone());
    }
    ExecPlan {
        program: c.serve_bin(),
        args: args.to_vec(),
        env,
    }
}

/// fv-serve's own informational flags: `fv-boot --version` (CI's image
/// smoke, `docker run <image> --version`), `--help`, `--print-config` run
/// fv-serve directly, without a boot.
pub fn passthrough(args: &[String]) -> bool {
    args.iter().any(|a| {
        matches!(
            a.as_str(),
            "--version" | "-V" | "--help" | "-h" | "--print-config"
        )
    })
}

/// The whole sequence up to the `exec`.
pub fn run(c: &mut Ctx<'_>, args: &[String]) -> Result<ExecPlan, BootError> {
    c.log
        .marker(&format!("start ({}; fv-boot {VERSION})", c.kind.label()));
    driver_step(c)?;
    volume_step(c);
    mkdir(&c.state_dir())?;

    let m = config::materialize(&c.env, &c.kind, args, &c.paths.worker_toml)?;
    config_check(c, &m)?;
    c.say(&format!(
        "config ok: {} ({})",
        m.path
            .as_ref()
            .map(|p| p.display().to_string())
            .unwrap_or_else(|| "default".into()),
        m.source.describe()
    ));
    match &c.kind {
        Kind::ClusterWorker => {
            if let Some(id) = c.env.get("RUNPOD_POD_ID").map(str::to_owned) {
                c.export("FV_WORKER_ID", id);
            }
        }
        Kind::Provider(_) => {
            if let Some(id) = c.env.get("FV_POD_ID").map(str::to_owned) {
                c.export("FV_WORKER_ID", id);
            }
        }
        _ => {}
    }

    sidecar::start(c);

    let source = weights::SourceKind::parse(c.get("FV_WEIGHTS_SOURCE"))?;
    let prepared = weights::source(source).prepare(c)?;
    if source == weights::SourceKind::Volume {
        tree_preflight(c, &m);
    }

    if matches!(c.kind, Kind::Provider(_)) {
        tunnel::run(c)?;
    } else if c.get("FV_DISPATCH_FRONT") == Some("1") {
        // An edge front on Runpod: where the edge forwards (EDGE_WORKER_BOOT).
        if let Some(id) = c.env.get("RUNPOD_POD_ID").map(str::to_owned) {
            c.export(
                "FV_DISPATCH_ENDPOINT",
                format!("https://{id}-8000.proxy.runpod.net"),
            );
        }
    }

    watchdog::start(c);
    sidecar::wait(c);

    let bin = c.serve_bin();
    let streaming = prepared
        .streaming
        .as_ref()
        .is_some_and(|s| weights::r2::before_serve(c, s));
    if let Some(root) = &prepared.root {
        let el = c.log.elapsed();
        let note = if streaming {
            format!("weights streaming under {} after {el:.1}s: starting fv-serve (it loads while the fetch runs)", root.display())
        } else {
            format!(
                "weights ready under {} after {el:.1}s: starting fv-serve",
                root.display()
            )
        };
        c.sls_report("serve", &note);
        c.say(if streaming {
            "starting fv-serve while the weights arrive"
        } else {
            "starting fv-serve"
        });
    }
    let version = c
        .deps
        .spawner
        .output(&c.cmd(bin.display().to_string()).arg("--version"))
        .ok()
        .and_then(|o| o.stdout_text().lines().next().map(str::to_owned))
        .unwrap_or_default();
    let shown: Vec<String> = std::iter::once(bin.display().to_string())
        .chain(m.serve_args.iter().cloned())
        .collect();
    c.say(&format!(
        "exec {} ({}); {version}",
        shown.join(" "),
        m.source.describe()
    ));
    let mut env = c.env.vars().clone();
    for (k, v) in &c.exports {
        env.insert(k.clone(), v.clone());
    }
    for k in &c.unset {
        env.remove(k);
    }
    Ok(ExecPlan {
        program: bin,
        args: m.serve_args,
        env,
    })
}
