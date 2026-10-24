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
//! ```
//!
//! The weight gate is `verify-weights.sh` and the clip metrics and the
//! promotion gate are `fv-gpucheck compare-clips` / `gate`: called, never
//! reimplemented. `--dry-run` prints each arm's command line as the script's
//! `FV_DRY_RUN=1` does (`[<family>] dry-run <arm>: <printf %q …>`), and
//! nothing is written.

use std::collections::BTreeMap;
use std::fs::{self, File, OpenOptions};
use std::io::Write as _;
use std::os::unix::process::ExitStatusExt as _;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use fastvideo_deploy::boot::sys::utc_hms;
use fastvideo_deploy::launch::utc_iso;

use crate::plan::{GateKind, Paths, Step};
use crate::quote::printf_q_line;

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
}

impl RunConfig {
    pub fn new(family: &str, root: &Path) -> Self {
        Self {
            family: family.to_owned(),
            work: PathBuf::from("/workspace"),
            scratch: PathBuf::from("/workspace"),
            tag: None,
            bin: PathBuf::from(DEFAULT_GPUCHECK),
            root: root.to_path_buf(),
            cell_timeout_s: 300,
            only: None,
            lpips: false,
            gate_policy: None,
            nvidia_smi: "nvidia-smi".into(),
            timeout: "timeout".into(),
            gpu_poll: Duration::from_secs(2),
        }
    }

    /// `$W`.
    pub fn weights(&self) -> PathBuf {
        self.work.join("weights")
    }

    /// `$RUNS`.
    pub fn runs(&self) -> PathBuf {
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
    for s in steps {
        match s {
            Step::Arm { name, command, .. } => {
                if !cfg.selected(name) {
                    out.push(format!("[{f}] skip {name} (not in FV_CELLS)"));
                } else {
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
        }
    }
    out.push(format!("[{f}] matrix done"));
    out
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
    runs: PathBuf,
    log_path: PathBuf,
    /// Set on every child (`export PATH`, `LD_LIBRARY_PATH`, `FV_AUX_DIR`).
    env: Vec<(String, String)>,
    lpips_args: Vec<String>,
}

impl Runner {
    pub fn new(cfg: RunConfig) -> Self {
        let runs = cfg.runs();
        let log_path = runs.join("live.log");
        Self {
            cfg,
            runs,
            log_path,
            env: Vec::new(),
            lpips_args: Vec::new(),
        }
    }

    pub fn runs(&self) -> &Path {
        &self.runs
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
        if let Ok(mut f) = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.log_path)
        {
            let _ = writeln!(f, "{line}");
        }
    }

    fn cmd(&self, program: impl AsRef<std::ffi::OsStr>) -> Command {
        let mut c = Command::new(program);
        c.env_remove("FASTVIDEO_LTX2_WEIGHTS");
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
        let aux = std::env::var("FV_AUX_DIR")
            .ok()
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| self.cfg.weights().join("auxiliary").display().to_string());
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
                }
                Err(_) => {
                    self.log("FATAL: nvidia-smi missing");
                    return Err(RunError::Fatal("nvidia-smi missing".into()));
                }
            }
        }
        Ok(())
    }

    /// Every step, then `matrix done` and `done.json`.
    pub fn run(&mut self, steps: &[Step]) -> Result<(), RunError> {
        self.prepare()?;
        for s in steps {
            match s {
                Step::Arm {
                    name,
                    gate,
                    command,
                    ..
                } => self.gated_cell(name, gate, command),
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
                } => self.gate_cells(baseline, candidate, *kind, off.as_deref()),
            }
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

    /// `gated_cell`: `FV_CELLS`, then `verify-weights.sh <gate>`, then the arm.
    pub fn gated_cell(&self, name: &str, gate: &str, command: &[String]) {
        if !self.cfg.selected(name) {
            self.log(&format!("skip {name} (not in FV_CELLS)"));
            return;
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
        let cap = self.cfg.cell_timeout_s;
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

    /// `gate_cells`: `fv-gpucheck gate` with the gate policy.
    pub fn gate_cells(&self, base: &str, cand: &str, kind: GateKind, off: Option<&str>) {
        let tag = format!("{base}--{cand}");
        let dir = self.runs.join("gate");
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
            return;
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
    }
}

fn append(p: &Path) -> Result<File, RunError> {
    OpenOptions::new()
        .create(true)
        .append(true)
        .open(p)
        .map_err(io(p))
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
        assert_eq!(c.weights(), Path::new("/workspace/weights"));
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
            },
            Step::Arm {
                name: "b".into(),
                gate: "g".into(),
                trees: vec![],
                command: vec!["/nonexistent/fv".into()],
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
