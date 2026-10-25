//! The boot sequence against a fake provider: fake processes, HTTP, clock and
//! log (no GPU, no network). Each test builds a temporary image root, an
//! environment as fv-control (or the image) would give it, runs the boot and
//! checks the files, the reports, the processes and the `exec`.

use std::collections::BTreeMap;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::Duration;

use base64::Engine as _;
use serde_json::Value;

use super::sys::{Clock, Cmd, Http, LogSink, Output, Request, Response, Spawner, SysError};
use super::weights::hex;
use super::*;

// ------------------------------------------------------------------ fakes

/// What a fake process did.
#[derive(Default)]
struct Did {
    code: i32,
    stdout: String,
    stderr: String,
}

type Handler = Box<dyn Fn(&Cmd) -> Did + Send + Sync>;

/// Every program run through it is recorded and answered by `handler`.
struct FakeSpawner {
    calls: Mutex<Vec<(String, Cmd)>>,
    handler: Handler,
}

impl FakeSpawner {
    fn new(h: impl Fn(&Cmd) -> Did + Send + Sync + 'static) -> Self {
        Self {
            calls: Mutex::new(Vec::new()),
            handler: Box::new(h),
        }
    }
    fn record(&self, how: &str, c: &Cmd) {
        self.calls.lock().unwrap().push((how.into(), c.clone()));
    }
    fn calls(&self) -> Vec<(String, Cmd)> {
        self.calls.lock().unwrap().clone()
    }
    fn ran(&self, needle: &str) -> Vec<Cmd> {
        self.calls()
            .into_iter()
            .filter(|(_, c)| c.display().contains(needle))
            .map(|(_, c)| c)
            .collect()
    }
}

impl Spawner for FakeSpawner {
    fn output(&self, cmd: &Cmd) -> Result<Output, SysError> {
        self.record("output", cmd);
        let d = (self.handler)(cmd);
        Ok(Output {
            code: Some(d.code),
            stdout: d.stdout.into_bytes(),
            stderr: d.stderr.into_bytes(),
        })
    }
    fn status(&self, cmd: &Cmd) -> Result<Option<i32>, SysError> {
        self.record("status", cmd);
        Ok(Some((self.handler)(cmd).code))
    }
    fn tee(&self, cmd: &Cmd, log: &Path) -> Result<Option<i32>, SysError> {
        self.record("tee", cmd);
        let d = (self.handler)(cmd);
        std::fs::write(log, format!("{}{}", d.stdout, d.stderr)).unwrap();
        Ok(Some(d.code))
    }
    fn spawn(&self, cmd: &Cmd) -> Result<u32, SysError> {
        self.record("spawn", cmd);
        (self.handler)(cmd);
        Ok(4242)
    }
}

type HttpHandler = Box<dyn Fn(&Request) -> Response + Send + Sync>;

/// Records every request; `downloads` serves bodies by URL.
struct FakeHttp {
    sent: Mutex<Vec<Request>>,
    downloads: BTreeMap<String, Vec<u8>>,
    handler: HttpHandler,
}

impl FakeHttp {
    fn new() -> Self {
        Self {
            sent: Mutex::new(Vec::new()),
            downloads: BTreeMap::new(),
            handler: Box::new(|_| Response {
                status: 200,
                body: b"{}".to_vec(),
            }),
        }
    }
    fn serve(mut self, url: &str, body: &[u8]) -> Self {
        self.downloads.insert(url.into(), body.to_vec());
        self
    }
    fn answer(mut self, h: impl Fn(&Request) -> Response + Send + Sync + 'static) -> Self {
        self.handler = Box::new(h);
        self
    }
    fn sent(&self) -> Vec<Request> {
        self.sent.lock().unwrap().clone()
    }
    /// The JSON bodies posted to `url`.
    fn bodies(&self, url: &str) -> Vec<Value> {
        self.sent()
            .into_iter()
            .filter(|r| r.url == url)
            .filter_map(|r| serde_json::from_slice(r.body.as_deref()?).ok())
            .collect()
    }
}

impl Http for FakeHttp {
    fn send(&self, req: &Request) -> Result<Response, SysError> {
        self.sent.lock().unwrap().push(req.clone());
        Ok((self.handler)(req))
    }
    fn download(&self, url: &str, dest: &Path, _t: Duration) -> Result<u64, SysError> {
        self.sent
            .lock()
            .unwrap()
            .push(Request::new(sys::Method::Get, url, Duration::ZERO));
        match self.downloads.get(url) {
            Some(b) => {
                std::fs::write(dest, b).map_err(|e| SysError::io(dest, e))?;
                Ok(b.len() as u64)
            }
            None => Err(SysError::Status {
                url: url.into(),
                status: 404,
            }),
        }
    }
}

/// Time that moves only when the boot sleeps (and a little per read).
struct FakeClock {
    now: Mutex<f64>,
    slept: Mutex<Vec<Duration>>,
}

impl FakeClock {
    fn new() -> Self {
        Self {
            now: Mutex::new(1_761_300_000.0),
            slept: Mutex::new(Vec::new()),
        }
    }
    fn slept(&self) -> Vec<Duration> {
        self.slept.lock().unwrap().clone()
    }
}

impl Clock for FakeClock {
    fn now(&self) -> f64 {
        let mut n = self.now.lock().unwrap();
        *n += 0.05;
        *n
    }
    fn sleep(&self, d: Duration) {
        *self.now.lock().unwrap() += d.as_secs_f64();
        self.slept.lock().unwrap().push(d);
    }
}

#[derive(Default)]
struct CaptureLog(Mutex<Vec<String>>);

impl LogSink for CaptureLog {
    fn line(&self, s: &str) {
        self.0.lock().unwrap().push(s.to_owned());
    }
}

impl CaptureLog {
    fn lines(&self) -> Vec<String> {
        self.0.lock().unwrap().clone()
    }
    fn has(&self, needle: &str) -> bool {
        self.lines().iter().any(|l| l.contains(needle))
    }
}

/// A temporary image: its paths, a fake fv-serve, a PATH.
struct Image {
    root: PathBuf,
    paths: Paths,
}

fn exe(p: &Path, body: &[u8]) {
    if let Some(d) = p.parent() {
        std::fs::create_dir_all(d).unwrap();
    }
    std::fs::write(p, body).unwrap();
    std::fs::set_permissions(p, std::fs::Permissions::from_mode(0o755)).unwrap();
}

impl Image {
    fn new(name: &str) -> Self {
        let root = std::env::temp_dir().join(format!("fvb-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();
        let r = |p: &str| root.join(p);
        let paths = Paths {
            worker_toml: r("fv-worker.toml"),
            runpod_volume_weights: r("runpod-volume/weights"),
            workspace_weights: r("workspace/weights"),
            serve_bin: r("opt/bin/fv-serve"),
            upsampler: r("opt/bin/fv-upsampler"),
            r2_fetch_image: r("opt/bin/r2-fetch.py"),
            tools: r("opt/fv"),
            state: r("fvstate"),
            cloudflared: r("usr/local/bin/cloudflared"),
            cloudflared_url: "https://example.invalid/cloudflared".into(),
            cloudflared_sha256: hex(&<sha2::Sha256 as sha2::Digest>::digest(b"cloudflared-bin")),
            self_exe: r("opt/bin/fv-boot"),
        };
        exe(&paths.serve_bin, b"\x7fELF fv-serve");
        exe(&r("bin/python3"), b"#!/bin/sh\n");
        Self { root, paths }
    }
    fn path(&self) -> String {
        self.root.join("bin").display().to_string()
    }
    fn weights(&self) -> PathBuf {
        self.paths.workspace_weights.clone()
    }
}

impl Drop for Image {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.root);
    }
}

fn b64(s: &str) -> String {
    base64::engine::general_purpose::STANDARD.encode(s)
}

fn ok() -> Did {
    Did::default()
}

/// The default fake processes: everything succeeds, nvidia-smi is absent.
fn default_handler(c: &Cmd) -> Did {
    match c.program.as_str() {
        "nvidia-smi" | "ldconfig" => Did {
            code: 127,
            ..Did::default()
        },
        _ if c.args.iter().any(|a| a == "--version") => Did {
            stdout: "fv-serve 0.1.1 (test)\n".into(),
            ..Did::default()
        },
        _ => ok(),
    }
}

struct Run {
    result: Result<ExecPlan, BootError>,
    log: CaptureLog,
}

fn boot(
    img: &Image,
    env: &[(&str, String)],
    args: &[&str],
    sp: &FakeSpawner,
    http: &FakeHttp,
    clock: &FakeClock,
) -> Run {
    let log = CaptureLog::default();
    let mut e: Vec<(String, String)> = env
        .iter()
        .map(|(k, v)| ((*k).to_owned(), v.clone()))
        .collect();
    e.push(("PATH".into(), img.path()));
    let deps = Deps {
        spawner: sp,
        http,
        clock,
        log: &log,
    };
    let args: Vec<String> = args.iter().map(|s| (*s).to_owned()).collect();
    let result = {
        let mut c = Ctx::new(BootEnv::from_pairs(e), deps, img.paths.clone(), 777);
        let r = run(&mut c, &args);
        if let Err(err) = &r {
            c.fail(err);
        }
        r
    };
    Run { result, log }
}

fn s(v: &str) -> String {
    v.to_owned()
}

// ------------------------------------------------------------------ the sequence

#[test]
fn cluster_worker_on_the_volume_with_an_inline_config() {
    let img = Image::new("cluster");
    std::fs::create_dir_all(img.weights().join("h3-base")).unwrap();
    std::fs::create_dir_all(img.weights().join("ltx25")).unwrap();
    std::fs::create_dir_all(img.weights().join(".hidden")).unwrap();
    let sp = FakeSpawner::new(default_handler);
    let http = FakeHttp::new();
    let clock = FakeClock::new();
    let toml = "[server]\nbind = \"0.0.0.0:8000\"\n";
    let r = boot(
        &img,
        &[
            ("FV_SKIP_DRIVER_CHECK", s("1")),
            ("FV_SERVE_ROLE", s("worker")),
            ("RUNPOD_POD_ID", s("abc123")),
            ("FV_WORKER_TOML_B64", b64(toml)),
            ("FV_WORKER_CONFIG", s("/fv-worker.toml")),
            ("FV_WEIGHTS", img.weights().display().to_string()),
            ("FV_STATE_DIR", img.paths.state.display().to_string()),
            ("FV_INTERNAL_TOKEN", s("secret")),
        ],
        &[],
        &sp,
        &http,
        &clock,
    );
    let plan = r.result.unwrap();
    let lines = r.log.lines();
    assert!(
        lines[0].starts_with("[fv-boot] start (cluster worker; fv-boot "),
        "{lines:?}"
    );
    assert!(
        lines.contains(&format!(
            "[fv-boot] volume: {} (2 trees)",
            img.weights().display()
        )),
        "{lines:?}"
    );
    assert_eq!(
        std::fs::read_to_string(&img.paths.worker_toml).unwrap(),
        format!("{toml}\n[gateway]\nregister = false\n")
    );
    assert_eq!(plan.program, img.paths.serve_bin);
    assert_eq!(
        plan.args,
        vec![s("--config"), img.paths.worker_toml.display().to_string()]
    );
    assert_eq!(plan.env["FV_WORKER_ID"], "abc123");
    assert_eq!(plan.env["FV_INTERNAL_TOKEN"], "secret");
    // fv-serve checked the config before anything else ran on it.
    let checks = sp.ran("--print-config");
    assert_eq!(checks.len(), 1);
    assert_eq!(
        checks[0].args,
        vec![
            s("--config"),
            img.paths.worker_toml.display().to_string(),
            s("--print-config")
        ]
    );
    // No report sink, no watchdog, no tunnel.
    assert!(http.sent().is_empty());
    assert!(sp.ran("watchdog").is_empty());
    assert!(img.paths.state.is_dir());
}

#[test]
fn the_config_forms_and_explicit_arguments() {
    let img = Image::new("forms");
    let file = img.root.join("etc/fv/runpod-fake.toml");
    std::fs::create_dir_all(file.parent().unwrap()).unwrap();
    std::fs::write(&file, "[server]\n[gateway]\npool = \"ltx\"\n").unwrap();
    let base = |extra: &[(&'static str, String)]| -> Vec<(&'static str, String)> {
        let mut v = vec![
            ("FV_SKIP_DRIVER_CHECK", s("1")),
            ("FV_STATE_DIR", img.paths.state.display().to_string()),
        ];
        v.extend(extra.iter().cloned());
        v
    };
    let go = |env: Vec<(&str, String)>, args: &[&str]| {
        let sp = FakeSpawner::new(default_handler);
        boot(&img, &env, args, &sp, &FakeHttp::new(), &FakeClock::new())
            .result
            .unwrap()
    };
    let wt = img.paths.worker_toml.display().to_string();

    // 1. inline (serverless: no gateway edit); an old baked FV_CONFIG is not read
    let p = go(
        base(&[
            ("RUNPOD_ENDPOINT_ID", s("ep1")),
            ("FV_WORKER_TOML_B64", b64("[server]\n")),
            ("FV_CONFIG", file.display().to_string()),
        ]),
        &[],
    );
    assert_eq!(p.args, vec![s("--config"), wt.clone()]);
    assert_eq!(
        std::fs::read_to_string(&img.paths.worker_toml).unwrap(),
        "[server]\n"
    );

    // 2. named: a controller worker's FV_WORKER_CONFIG, copied with the gateway off
    let p = go(
        base(&[
            ("FV_SERVE_ROLE", s("worker")),
            ("RUNPOD_POD_ID", s("p1")),
            ("FV_WORKER_CONFIG", file.display().to_string()),
        ]),
        &[],
    );
    assert_eq!(p.args, vec![s("--config"), wt.clone()]);
    assert_eq!(
        std::fs::read_to_string(&img.paths.worker_toml).unwrap(),
        "[server]\n[gateway]\nregister = false\npool = \"ltx\"\n"
    );

    //    ... and the entrypoint's own `--config` (GCP, CloudRift, Vast): as given
    let p = go(
        base(&[("FV_SERVE_ROLE", s("worker")), ("RUNPOD_POD_ID", s("p1"))]),
        &["--config", "/etc/fv-gcp/serve.toml"],
    );
    assert_eq!(p.args, vec![s("--config"), s("/etc/fv-gcp/serve.toml")]);

    //    ... the image on its own (CI's smoke check, the scripts): FV_WORKER_CONFIG as it is, no edit
    let p = go(
        base(&[("FV_WORKER_CONFIG", file.display().to_string())]),
        &[],
    );
    assert_eq!(p.args, vec![s("--config"), file.display().to_string()]);

    // An inline config wins over the arguments' --config; other arguments stay.
    let p = go(
        base(&[("FV_WORKER_TOML_B64", b64("[server]\n"))]),
        &["--config", "/a.toml", "--x"],
    );
    assert_eq!(p.args, vec![s("--config"), wt, s("--x")]);
}

/// The one image bakes no config (ADR-0011, owner decision 2026-10-25): a
/// boot without one fails at once, before the driver check and any weights,
/// with the `FAILED:` line, the serverless `failed` report and a non-zero exit.
#[test]
fn an_image_started_without_a_config_refuses_to_boot() {
    let img = Image::new("noconfig");
    for env in [
        vec![
            ("RUNPOD_ENDPOINT_ID", s("ep1")),
            ("FV_WEIGHTS_SOURCE", s("r2")),
        ],
        vec![("FV_CONFIG", s("/etc/fv/runpod.toml"))],
        vec![("FV_SERVE_ROLE", s("worker")), ("RUNPOD_POD_ID", s("p1"))],
        vec![],
    ] {
        let sp = FakeSpawner::new(default_handler);
        let clock = FakeClock::new();
        let r = boot(&img, &env, &[], &sp, &FakeHttp::new(), &clock);
        match &r.result {
            Err(BootError::Config(m)) => {
                assert!(
                    m.starts_with("no worker config: the image serves no default"),
                    "{m}"
                )
            }
            other => panic!("{other:?}"),
        }
        assert!(r.log.has("FAILED: no worker config"));
        // Nothing ran: not the driver check, not fv-serve, not a fetch.
        assert!(sp.calls().is_empty(), "{:?}", sp.calls());
    }
}

#[test]
fn config_failures_stop_the_boot_before_the_weights() {
    let img = Image::new("cfgfail");
    let sp = FakeSpawner::new(|c| {
        if c.args.iter().any(|a| a == "--print-config") {
            return Did {
                code: 2,
                stderr: "fv-serve: config: unknown field `\"bogus\"`\n".into(),
                ..Did::default()
            };
        }
        default_handler(c)
    });
    let clock = FakeClock::new();
    let r = boot(
        &img,
        &[
            ("FV_SKIP_DRIVER_CHECK", s("1")),
            ("FV_WORKER_CONFIG", s("/etc/fv/x.toml")),
            ("FV_WEIGHTS_SOURCE", s("r2")),
        ],
        &[],
        &sp,
        &FakeHttp::new(),
        &clock,
    );
    match r.result {
        Err(BootError::Config(m)) => assert_eq!(
            m,
            "fv-serve config: fv-serve: config: unknown field ` bogus `"
        ),
        other => panic!("{other:?}"),
    }
    assert!(r.log.has("FAILED: fv-serve config: "));
    assert_eq!(clock.slept(), vec![Duration::from_secs(30)]);

    let r = boot(
        &img,
        &[
            ("FV_SKIP_DRIVER_CHECK", s("1")),
            ("FV_WORKER_TOML_B64", s("%%%")),
        ],
        &[],
        &FakeSpawner::new(default_handler),
        &FakeHttp::new(),
        &FakeClock::new(),
    );
    assert!(
        matches!(r.result, Err(BootError::Config(m)) if m == "FV_WORKER_TOML_B64 does not decode")
    );

    let r = boot(
        &img,
        &[
            ("FV_SKIP_DRIVER_CHECK", s("1")),
            ("FV_SERVE_ROLE", s("worker")),
            ("RUNPOD_POD_ID", s("p1")),
            ("FV_WORKER_CONFIG", s("/nope.toml")),
        ],
        &[],
        &FakeSpawner::new(default_handler),
        &FakeHttp::new(),
        &FakeClock::new(),
    );
    assert!(
        matches!(r.result, Err(BootError::Config(m)) if m == "no worker config /nope.toml in the image")
    );

    std::fs::remove_file(&img.paths.serve_bin).unwrap();
    let r = boot(
        &img,
        &[
            ("FV_SKIP_DRIVER_CHECK", s("1")),
            ("FV_WORKER_CONFIG", s("/etc/fv/runpod-fake.toml")),
        ],
        &[],
        &FakeSpawner::new(default_handler),
        &FakeHttp::new(),
        &FakeClock::new(),
    );
    assert!(
        matches!(r.result, Err(BootError::Config(m)) if m.starts_with("no ") && m.ends_with("fv-serve in the image"))
    );
}

#[test]
fn an_old_driver_fails_fast_and_is_reported() {
    let img = Image::new("driver");
    let libcuda = img.root.join("libcuda.so.1");
    std::fs::write(&libcuda, b"\x7fELF cuCtxGetDevice cuInit").unwrap();
    let sp = FakeSpawner::new(|c| match c.program.as_str() {
        "nvidia-smi" => Did {
            stdout: "550.54.15\n".into(),
            ..Did::default()
        },
        _ => default_handler(c),
    });
    let http = FakeHttp::new();
    let clock = FakeClock::new();
    let r = boot(
        &img,
        &[
            ("FV_PROVIDER", s("gmi")),
            ("FV_POD_ID", s("gmi:fv-pod-a-1")),
            (
                "FV_ENDPOINT_REPORT_URL",
                s("https://control/ingest/v1/endpoint"),
            ),
            ("FV_ENDPOINT_REPORT_TOKEN", s("ingest_tok")),
            ("FV_LIBCUDA", libcuda.display().to_string()),
            ("FV_WORKER_TOML_B64", b64("[server]\n")),
        ],
        &[],
        &sp,
        &http,
        &clock,
    );
    let msg = "driver_too_old: host driver 550.54.15 < 580 (the image needs CUDA 13)";
    assert!(matches!(&r.result, Err(BootError::DriverTooOld(m)) if m == msg));
    assert!(r.log.has(&format!("FAILED: {msg}")));
    let sent = http.sent();
    assert_eq!(sent.len(), 1);
    assert_eq!(
        sent[0].headers[0],
        (s("authorization"), s("Bearer ingest_tok"))
    );
    assert_eq!(
        String::from_utf8(sent[0].body.clone().unwrap()).unwrap(),
        format!(r#"{{"pod":"gmi:fv-pod-a-1","phase":"failed","url":"","detail":"{msg}"}}"#)
    );
    // FV_DRIVER_FAIL_SLEEP_S's default, and nothing else ran (no config check, no fetch).
    assert_eq!(clock.slept(), vec![Duration::from_secs(20)]);
    assert!(sp.ran("--print-config").is_empty());

    // The symbol check: a new-looking version on an old libcuda.
    let sp = FakeSpawner::new(|c| match c.program.as_str() {
        "nvidia-smi" => Did {
            stdout: "580.1\n".into(),
            ..Did::default()
        },
        _ => default_handler(c),
    });
    let r = boot(
        &img,
        &[
            ("FV_LIBCUDA", libcuda.display().to_string()),
            ("FV_WORKER_CONFIG", s("/etc/fv/runpod-fake.toml")),
        ],
        &[],
        &sp,
        &FakeHttp::new(),
        &FakeClock::new(),
    );
    assert!(
        matches!(&r.result, Err(BootError::DriverTooOld(m)) if m.contains("has no cuCtxGetDevice_v2"))
    );

    // A good host.
    std::fs::write(&libcuda, b"\x7fELF cuCtxGetDevice_v2").unwrap();
    let r = boot(
        &img,
        &[
            ("FV_LIBCUDA", libcuda.display().to_string()),
            ("FV_WORKER_CONFIG", s("/etc/fv/runpod-fake.toml")),
        ],
        &[],
        &sp,
        &FakeHttp::new(),
        &FakeClock::new(),
    );
    assert!(r.result.is_ok());
    assert!(r.log.has("driver 580.1 ok"));
}

#[test]
fn serverless_volume_link_and_weights_default() {
    let img = Image::new("slsvol");
    std::fs::create_dir_all(img.paths.runpod_volume_weights.join("h3-base")).unwrap();
    let cfg = img.root.join("runpod.toml");
    std::fs::write(&cfg, "[[models]]\nweights = \"${FV_WEIGHTS}/h3-base\"\n[[models]]\nweights = \"${FV_WEIGHTS}/missing-tree\"\n").unwrap();
    let sp = FakeSpawner::new(default_handler);
    let r = boot(
        &img,
        &[
            ("FV_SKIP_DRIVER_CHECK", s("1")),
            ("RUNPOD_ENDPOINT_ID", s("ep")),
            ("FV_WORKER_CONFIG", cfg.display().to_string()),
        ],
        &[],
        &sp,
        &FakeHttp::new(),
        &FakeClock::new(),
    );
    let plan = r.result.unwrap();
    let ws = &img.paths.workspace_weights;
    assert_eq!(
        std::fs::read_link(ws).unwrap(),
        img.paths.runpod_volume_weights
    );
    assert_eq!(
        plan.env["FV_WEIGHTS"],
        img.paths.runpod_volume_weights.display().to_string()
    );
    assert!(r.log.has(&format!(
        "[fv-boot] volume: {} (1 trees)",
        img.paths.runpod_volume_weights.display()
    )));
    assert!(r
        .log
        .has(&format!("config {}: weights tree", cfg.display())));
    assert!(r.log.has("/missing-tree is missing"));
    assert!(!r.log.has("/h3-base is missing"));
}

// ------------------------------------------------------------------ r2

const PLAN_URL: &str = "https://control/ingest/v1/weights-plan";
const SLS_URL: &str = "https://control/ingest/v1/sls-report";

fn r2_fetcher(img: &Image) -> String {
    let body = b"#!/usr/bin/env python3\n# r2-fetch.py (test copy)\n";
    std::fs::write(&img.paths.r2_fetch_image, body).unwrap();
    hex(&<sha2::Sha256 as sha2::Digest>::digest(body))
}

/// The fake fetcher: writes the tree as r2-fetch.py would and prints its summary.
fn fake_fetch(c: &Cmd, stream: bool) -> Did {
    let root = c
        .args
        .iter()
        .skip_while(|a| *a != "--root")
        .nth(1)
        .cloned()
        .unwrap_or_default();
    let d = Path::new(&root).join("demo");
    std::fs::create_dir_all(&d).unwrap();
    std::fs::write(d.join("config.json"), b"{}").unwrap();
    std::fs::write(d.join(".fv-revision"), b"rev1").unwrap();
    std::fs::write(d.join(".complete"), b"").unwrap();
    if stream {
        let ready = c
            .args
            .iter()
            .skip_while(|a| *a != "--stream-ready")
            .nth(1)
            .cloned()
            .unwrap_or_default();
        std::fs::write(ready, b"").unwrap();
    }
    Did {
        stderr: "r2 1.00/2.00 GB, 600 MB/s\nweights ready: 1 tree(s), 2.00 GB fetched in 3.3s (606 MB/s); retries 0, bad parts 0, resumed parts 0, slow restarts 0\n".into(),
        ..Did::default()
    }
}

fn sls_r2_env(img: &Image, sha: &str) -> Vec<(&'static str, String)> {
    let aux_body = b"tae-weights";
    let aux_sha = hex(&<sha2::Sha256 as sha2::Digest>::digest(aux_body));
    vec![
        ("FV_SKIP_DRIVER_CHECK", s("1")),
        ("RUNPOD_ENDPOINT_ID", s("ep")),
        ("FV_CONTROL_ENDPOINT", s("h3")),
        ("RUNPOD_POD_ID", s("wkr-abc123")),
        ("FV_WEIGHTS_SOURCE", s("r2")),
        ("FV_WEIGHTS", img.weights().display().to_string()),
        ("FV_STATE_DIR", img.paths.state.display().to_string()),
        ("FV_WEIGHTS_PLAN_URL", s(PLAN_URL)),
        (
            "FV_R2_FETCH_URL",
            s("https://control/ingest/v1/r2-fetch.py"),
        ),
        ("FV_R2_FETCH_SHA256", s(sha)),
        ("FV_WEIGHTS_TOKEN", s("fvw_tok")),
        ("FV_SLS_REPORT_URL", s(SLS_URL)),
        ("FV_R2_TREE_REVS", s("demo@rev1 h3_mxfp8@rev9")),
        ("FV_R2_SKIP_TREES", s("*_mxfp8")),
        (
            "FV_WEIGHTS_TREES_B64",
            b64(&format!(
                "aux\tauxiliary/tae/taeh3.pth\thttps://aux/taeh3.pth\t{aux_sha}\t{}\n",
                aux_body.len()
            )),
        ),
        ("FV_WORKER_CONFIG", s("/etc/fv/runpod.toml")),
    ]
}

#[test]
fn serverless_r2_cold_then_warm() {
    let img = Image::new("slsr2");
    let sha = r2_fetcher(&img);
    let env = sls_r2_env(&img, &sha);
    let sp = FakeSpawner::new(|c| {
        if c.program == "python3" {
            fake_fetch(c, false)
        } else {
            default_handler(c)
        }
    });
    let http = FakeHttp::new().serve("https://aux/taeh3.pth", b"tae-weights");
    let clock = FakeClock::new();
    let r = boot(&img, &env, &[], &sp, &http, &clock);
    let plan = r.result.unwrap();
    // The fetcher: the image's copy (its SHA-256 is the pinned one), the plan with the weights token, the boot id.
    let f = sp.ran("--plan-url");
    assert_eq!(f.len(), 1);
    let a = &f[0].args;
    assert_eq!(a[0], img.paths.r2_fetch_image.display().to_string());
    let arg = |k: &str| {
        a.iter()
            .skip_while(|x| *x != k)
            .nth(1)
            .cloned()
            .unwrap_or_default()
    };
    assert_eq!(arg("--root"), img.weights().display().to_string());
    assert_eq!(arg("--token-env"), "FV_WEIGHTS_TOKEN");
    assert_eq!(arg("--pod"), "wkr-abc123@1761300000");
    assert_eq!(arg("--report-url"), SLS_URL);
    assert!(
        !a.contains(&s("--stream")),
        "a fv-serve without the marker is not streamed into"
    );
    assert_eq!(
        std::fs::read(img.weights().join("auxiliary/tae/taeh3.pth")).unwrap(),
        b"tae-weights"
    );
    assert_eq!(plan.env["FV_R2_SKIP_TREES"], "*_mxfp8");
    // The serverless reports, in order, as fv-control's applyReport reads them.
    let b = http.bodies(SLS_URL);
    let phases: Vec<&str> = b.iter().map(|x| x["phase"].as_str().unwrap()).collect();
    assert_eq!(
        phases,
        ["boot", "cache", "step", "step", "step", "step", "serve"]
    );
    assert!(b
        .iter()
        .all(|x| x["pod"] == "wkr-abc123@1761300000" && x["url"] == ""));
    assert_eq!(b[0]["detail"], "container started; driver not checked");
    assert_eq!(b[1]["detail"], "miss: demo");
    let steps: Vec<&str> = b[2..6]
        .iter()
        .map(|x| x["detail"].as_str().unwrap())
        .collect();
    assert!(
        steps[0].starts_with("tools ") && steps[0].ends_with("(python3 in the image)"),
        "{steps:?}"
    );
    assert!(steps[1].starts_with("fetcher ") && steps[1].ends_with("(from the image)"));
    assert!(
        steps[2].starts_with("fetch ")
            && steps[2].contains("(weights ready: 1 tree(s), 2.00 GB fetched in 3.3s (606 MB/s)")
    );
    assert!(
        steps[3].starts_with("aux ")
            && steps[3].ends_with("(1 files checked, 1 from their pinned URLs)")
    );
    assert!(b[6]["detail"].as_str().unwrap().starts_with(&format!(
        "weights ready under {} after ",
        img.weights().display()
    )));
    assert!(http
        .sent()
        .iter()
        .filter(|r| r.url == SLS_URL)
        .all(|r| r.headers[0].1 == "Bearer fvw_tok"));

    // A second boot on the same disk: cache hit, no fetcher, aux already right.
    let sp2 = FakeSpawner::new(default_handler);
    let http2 = FakeHttp::new();
    boot(&img, &env, &[], &sp2, &http2, &FakeClock::new())
        .result
        .unwrap();
    assert!(sp2.ran("--plan-url").is_empty());
    let b = http2.bodies(SLS_URL);
    assert_eq!(b[1]["detail"], "hit");
    assert!(b
        .iter()
        .any(|x| x["detail"].as_str().unwrap().starts_with("aux ")
            && x["detail"]
                .as_str()
                .unwrap()
                .ends_with("(1 files checked, 0 from their pinned URLs)")));
}

#[test]
fn r2_failures_and_a_fetcher_from_control() {
    let img = Image::new("r2fail");
    let body = b"#!/usr/bin/env python3\n# fv-control's copy\n";
    let sha = hex(&<sha2::Sha256 as sha2::Digest>::digest(body));
    std::fs::write(&img.paths.r2_fetch_image, b"an older copy").unwrap();
    let mut env = sls_r2_env(&img, &sha);
    env.retain(|(k, _)| {
        *k != "FV_SLS_REPORT_URL"
            && *k != "FV_WEIGHTS_TOKEN"
            && *k != "RUNPOD_ENDPOINT_ID"
            && *k != "FV_CONTROL_ENDPOINT"
    });
    env.push(("FV_SERVE_ROLE", s("worker")));
    env.push(("FV_WORKER_TOML_B64", b64("[server]\n")));
    env.push(("FV_ENDPOINT_REPORT_TOKEN", s("ingest_tok")));
    let sp = FakeSpawner::new(|c| {
        if c.program == "python3" {
            return Did {
                code: 1,
                stderr: "r2-fetch: FAILED: demo/x part 3: sha256 \"mismatch\"\n".into(),
                ..Did::default()
            };
        }
        default_handler(c)
    });
    let http = FakeHttp::new().serve("https://control/ingest/v1/r2-fetch.py", body);
    let clock = FakeClock::new();
    let r = boot(&img, &env, &[], &sp, &http, &clock);
    match &r.result {
        Err(BootError::Weights(m)) => assert_eq!(
            m,
            "r2 weights: r2-fetch: FAILED: demo/x part 3: sha256 'mismatch'"
        ),
        o => panic!("{o:?}"),
    }
    assert!(r.log.has("step fetcher: ") && r.log.has("(downloaded from fv-control)"));
    // A Runpod pod's fetch: the ingest token, its own pod id (exported for fv-serve too).
    let f = sp.ran("--plan-url");
    let arg = |k: &str| {
        f[0].args
            .iter()
            .skip_while(|x| *x != k)
            .nth(1)
            .cloned()
            .unwrap_or_default()
    };
    assert_eq!(arg("--token-env"), "FV_ENDPOINT_REPORT_TOKEN");
    assert_eq!(arg("--pod"), "wkr-abc123");
    assert_eq!(
        f[0].env
            .iter()
            .find(|(k, _)| k == "FV_POD_ID")
            .map(|(_, v)| v.as_str()),
        Some("wkr-abc123")
    );

    // A fetcher whose SHA-256 is not the pinned one is refused.
    let http = FakeHttp::new().serve("https://control/ingest/v1/r2-fetch.py", b"tampered");
    let r = boot(
        &img,
        &env,
        &[],
        &FakeSpawner::new(default_handler),
        &http,
        &FakeClock::new(),
    );
    assert!(matches!(&r.result, Err(BootError::Weights(m)) if m == "r2-fetch.py sha256"));

    // Missing plan URL.
    env.retain(|(k, _)| *k != "FV_WEIGHTS_PLAN_URL");
    let r = boot(
        &img,
        &env,
        &[],
        &FakeSpawner::new(default_handler),
        &FakeHttp::new(),
        &FakeClock::new(),
    );
    assert!(matches!(&r.result, Err(BootError::Env(m)) if m.contains("FV_WEIGHTS_PLAN_URL")));
}

#[test]
fn r2_streaming_hands_the_fetch_to_the_guard() {
    let img = Image::new("stream");
    let sha = r2_fetcher(&img);
    exe(
        &img.paths.serve_bin,
        format!("\x7fELF ... {STREAMING_FETCH_MARKER} ...").as_bytes(),
    );
    let mut env = sls_r2_env(&img, &sha);
    env.retain(|(k, _)| *k != "FV_WEIGHTS_TREES_B64");
    let sp = FakeSpawner::new(|c| {
        if c.args.first().map(String::as_str) == Some("fetch-guard") {
            let rest: Vec<&String> = c.args.iter().skip_while(|a| *a != "--").skip(2).collect();
            let cmd = Cmd::new("python3").args(rest.into_iter().cloned());
            return fake_fetch(&cmd, true);
        }
        default_handler(c)
    });
    let http = FakeHttp::new();
    let r = boot(&img, &env, &[], &sp, &http, &FakeClock::new());
    let plan = r.result.unwrap();
    let g = sp.ran("fetch-guard");
    assert_eq!(g.len(), 1);
    assert_eq!(g[0].program, img.paths.self_exe.display().to_string());
    let a = &g[0].args;
    assert_eq!(&a[..3], &[s("fetch-guard"), s("--boot-pid"), s("777")]);
    assert!(a.contains(&s("--stream")) && a.contains(&s("--stream-ready")));
    assert!(
        !a.iter().any(|x| x.contains("fv-serve")),
        "pkill -f fv-serve must never match the guard"
    );
    assert_eq!(plan.env["FASTVIDEO_STREAMING_FETCH"], "1");
    assert_eq!(
        plan.env["FASTVIDEO_STREAMING_FETCH_STATUS"],
        img.paths.state.join("r2-fetch.json").display().to_string()
    );
    assert!(img.paths.state.join("serve-start").is_file());
    assert_eq!(sp.ran("renice").len(), 1);
    let b = http.bodies(SLS_URL);
    assert!(b
        .iter()
        .any(|x| x["detail"].as_str().unwrap().starts_with("stream_ready ")));
    let serve = b.iter().find(|x| x["phase"] == "serve").unwrap();
    assert!(serve["detail"]
        .as_str()
        .unwrap()
        .contains("weights streaming under"));
}

#[test]
fn the_guard_times_the_fetch_or_stops_fv_serve() {
    let img = Image::new("guard");
    let st = img.paths.state.clone();
    std::fs::create_dir_all(&st).unwrap();
    let env = BootEnv::from_pairs([
        ("FV_SLS_REPORT_URL", SLS_URL),
        ("FV_WEIGHTS_TOKEN", "fvw_tok"),
        ("RUNPOD_POD_ID", "w1"),
    ]);
    let log = CaptureLog::default();
    let clock = FakeClock::new();
    let http = FakeHttp::new();
    let ok_sp = FakeSpawner::new(|c| fake_fetch(c, false));
    let deps = Deps {
        spawner: &ok_sp,
        http: &http,
        clock: &clock,
        log: &log,
    };
    let c = Ctx::with_t0(env.clone(), deps, img.paths.clone(), 1, 1_761_300_000.0);
    std::fs::write(st.join("serve-start"), format!("{:.6}", 1_761_300_001.0)).unwrap();
    let args = vec![s("--root"), img.weights().display().to_string()];
    assert_eq!(weights::r2::guard(&c, 999, &st, "python3", &args), 0);
    assert_eq!(
        std::fs::read_to_string(st.join("r2-fetch.rc")).unwrap(),
        "0"
    );
    let d: Vec<String> = http
        .bodies(SLS_URL)
        .iter()
        .map(|x| x["detail"].as_str().unwrap().to_owned())
        .collect();
    assert!(
        d[0].starts_with("fetch ") && d[1].starts_with("load_overlap "),
        "{d:?}"
    );
    assert!(http
        .bodies(SLS_URL)
        .iter()
        .all(|x| x["pod"] == "w1@1761300000"));

    // A failure after fv-serve started: TERM, the grace, KILL, a failed report.
    std::fs::write(st.join("r2-stream-ready"), b"").unwrap();
    let bad = FakeSpawner::new(|c| {
        if c.program == "python3" {
            Did {
                code: 1,
                stderr: "r2-fetch: FAILED: part 9\n".into(),
                ..Did::default()
            }
        } else {
            ok()
        }
    });
    let http = FakeHttp::new();
    let clock = FakeClock::new();
    let deps = Deps {
        spawner: &bad,
        http: &http,
        clock: &clock,
        log: &log,
    };
    let c = Ctx::with_t0(env, deps, img.paths.clone(), 1, 1_761_300_000.0);
    assert_eq!(weights::r2::guard(&c, 999, &st, "python3", &args), 1);
    let kills: Vec<Vec<String>> = bad.ran("kill -s").into_iter().map(|c| c.args).collect();
    let k = |sig: &str| vec![s("-c"), s("kill -s \"$0\" \"$1\""), s(sig), s("999")];
    assert_eq!(kills, vec![k("TERM"), k("KILL")]);
    assert_eq!(clock.slept(), vec![Duration::from_secs(20)]);
    assert_eq!(
        http.bodies(SLS_URL)[0]["detail"],
        "r2 weights: r2-fetch: FAILED: part 9"
    );
    assert_eq!(
        std::fs::read_to_string(st.join("r2-fetch.rc")).unwrap(),
        "1"
    );
}

// ------------------------------------------------------------------ hub, tunnel, provider pods

#[test]
fn gmi_pod_hub_weights_tunnel_watchdog() {
    let img = Image::new("gmi");
    let shard = b"safetensors-bytes";
    let shard_sha = hex(&<sha2::Sha256 as sha2::Digest>::digest(shard));
    let rows = "tree\th3-base\tFastVideo/H3-Base\tabc123\t*.json transformer/*\n";
    let sums = format!(
        "h3-base\ttransformer/model.safetensors\t{}\t{shard_sha}\n",
        shard.len()
    );
    let sp = FakeSpawner::new(move |c| {
        if c.args.iter().any(|a| a.ends_with("fetch-hub-tree.py")) {
            let get = |k: &str| {
                c.env
                    .iter()
                    .find(|(x, _)| x == k)
                    .map(|(_, v)| v.clone())
                    .unwrap_or_default()
            };
            let d = Path::new(&get("FETCH_WEIGHTS")).join(get("FETCH_DEST"));
            std::fs::create_dir_all(d.join("transformer")).unwrap();
            std::fs::write(
                d.join("transformer/model.safetensors"),
                b"safetensors-bytes",
            )
            .unwrap();
            std::fs::write(d.join(".complete"), b"").unwrap();
            let srv = Path::new(&get("FETCH_SRV")).to_path_buf();
            std::fs::write(srv.join("DONE"), b"0\n").unwrap();
            let h = hex(&<sha2::Sha256 as sha2::Digest>::digest(
                b"safetensors-bytes",
            ));
            std::fs::write(
                srv.join("sha256.txt"),
                format!("{h}  transformer/model.safetensors\n"),
            )
            .unwrap();
            return ok();
        }
        if c.program.ends_with("cloudflared") {
            let log = match &c.stdout {
                sys::Io::Truncate(p) => p.clone(),
                _ => panic!("cloudflared's log"),
            };
            std::fs::write(log, "INF |  https://brave-fox-12.trycloudflare.com  |\n").unwrap();
            return ok();
        }
        default_handler(c)
    });
    let report = "https://control/ingest/v1/endpoint";
    let http = FakeHttp::new()
        .serve(
            "https://raw.githubusercontent.com/x/abc/scripts/gpu/fetch-hub-tree.py",
            b"#!/usr/bin/env python3\n",
        )
        .serve("https://example.invalid/cloudflared", b"cloudflared-bin");
    let clock = FakeClock::new();
    let r = boot(
        &img,
        &[
            ("FV_SKIP_DRIVER_CHECK", s("1")),
            ("FV_PROVIDER", s("gmi")),
            ("FV_SERVE_ROLE", s("worker")),
            ("FV_POD_ID", s("gmi:fv-pod-a-1")),
            ("FV_POD_NAME", s("fv-pod-a-1")),
            ("FV_WORKER_TOML_B64", b64("[server]\n")),
            ("FV_WEIGHTS", img.weights().display().to_string()),
            ("FV_STATE_DIR", img.paths.state.display().to_string()),
            ("FV_WEIGHTS_SOURCE", s("hub")),
            ("FV_WEIGHTS_TREES_B64", b64(rows)),
            ("FV_HUB_SHA256_B64", b64(&sums)),
            (
                "FV_SCRIPTS_URL",
                s("https://raw.githubusercontent.com/x/abc/scripts/gpu"),
            ),
            ("FV_ENDPOINT_REPORT_URL", s(report)),
            ("FV_ENDPOINT_REPORT_TOKEN", s("ingest_tok")),
            ("FV_CLUSTER_DEADLINE", s("1761400000")),
            ("FV_BACKSTOP_API_KEY", s("gmi-key")),
            ("FV_DISPATCH_FRONT", s("1")),
        ],
        &[],
        &sp,
        &http,
        &clock,
    );
    let plan = r.result.unwrap();
    assert_eq!(
        std::fs::read_to_string(img.weights().join("h3-base/.fv-revision")).unwrap(),
        "abc123"
    );
    assert!(r.log.has("tree h3-base: 1 files match weights-sha256.tsv"));
    assert!(r.log.lines().contains(&s(
        "[fv-boot] tunnel https://brave-fox-12.trycloudflare.com"
    )));
    assert!(r.log.has("step hub h3-base: "));
    let b = http.bodies(report);
    let got: Vec<(String, String, String)> = b
        .iter()
        .map(|x| {
            (
                x["phase"].as_str().unwrap().into(),
                x["url"].as_str().unwrap().into(),
                x["detail"].as_str().unwrap().into(),
            )
        })
        .collect();
    assert_eq!(
        got,
        vec![
            (s("weights"), s(""), s("checking")),
            (s("weights"), s(""), s("downloading h3-base")),
            (
                s("tunnel"),
                s("https://brave-fox-12.trycloudflare.com"),
                s("")
            ),
        ]
    );
    assert!(b.iter().all(|x| x["pod"] == "gmi:fv-pod-a-1"));
    assert_eq!(plan.env["FV_WORKER_ID"], "gmi:fv-pod-a-1");
    assert_eq!(
        plan.env["FV_DISPATCH_ENDPOINT"],
        "https://brave-fox-12.trycloudflare.com"
    );
    assert_eq!(
        std::fs::read_to_string(&img.paths.worker_toml).unwrap(),
        "[server]\n\n[gateway]\nregister = false\n"
    );
    let w = sp.ran("watchdog");
    assert_eq!(w.len(), 1);
    assert_eq!(w[0].args, vec![s("watchdog"), s("--serve-pid"), s("777")]);
    assert!(w[0].env.contains(&(s("FV_BACKSTOP_API_KEY"), s("gmi-key"))));
    // The tools were set up once: apt, venv, pip.
    assert_eq!(sp.ran("huggingface_hub").len(), 1);

    // A warm restart (a parked Brev VM): the tree is kept, nothing downloaded.
    let sp2 = FakeSpawner::new(|c| {
        if c.program.ends_with("cloudflared") {
            if let sys::Io::Truncate(p) = &c.stdout {
                std::fs::write(p, "https://warm-1.trycloudflare.com\n").unwrap();
            }
        }
        default_handler(c)
    });
    let http2 = FakeHttp::new().serve("https://example.invalid/cloudflared", b"cloudflared-bin");
    let r = boot(
        &img,
        &[
            ("FV_SKIP_DRIVER_CHECK", s("1")),
            ("FV_PROVIDER", s("brev")),
            ("FV_POD_ID", s("brev:fv-pod-b-1")),
            ("FV_WORKER_TOML_B64", b64("[server]\n")),
            ("FV_WEIGHTS", img.weights().display().to_string()),
            ("FV_STATE_DIR", img.paths.state.display().to_string()),
            ("FV_WEIGHTS_SOURCE", s("hub")),
            ("FV_WEIGHTS_TREES_B64", b64(rows)),
            ("FV_TUNNEL_TOKEN", s("eyJhIjoisecret")),
            ("FV_TUNNEL_URL", s("https://fv-pod-b-1.tunnels.example")),
        ],
        &[],
        &sp2,
        &http2,
        &FakeClock::new(),
    );
    // A named tunnel that never registers: the boot fails with the (redacted) log tail.
    match &r.result {
        Err(BootError::Tunnel(m)) => assert!(m.starts_with("named tunnel did not connect"), "{m}"),
        o => panic!("{o:?}"),
    }
    assert!(r.log.has("tree h3-base: present at abc123"));
    assert!(sp2.ran("fetch-hub-tree").is_empty());
    let cf = sp2.ran("cloudflared");
    let named: Vec<&Cmd> = cf.iter().filter(|c| c.args.contains(&s("run"))).collect();
    assert_eq!(named.len(), 1);
    assert!(named[0]
        .env
        .contains(&(s("TUNNEL_TOKEN"), s("eyJhIjoisecret"))));
    assert!(named[0].env_remove.contains(&s("FV_TUNNEL_TOKEN")));
    assert!(!named[0].args.iter().any(|a| a.contains("eyJ")));
}

#[test]
fn named_tunnel_registers_and_its_token_never_reaches_fv_serve() {
    let img = Image::new("named");
    let sp = FakeSpawner::new(|c| {
        if c.program.ends_with("cloudflared") {
            if let sys::Io::Truncate(p) = &c.stdout {
                std::fs::write(p, "INF Registered tunnel connection connIndex=0\n").unwrap();
            }
        }
        default_handler(c)
    });
    let http = FakeHttp::new().serve("https://example.invalid/cloudflared", b"cloudflared-bin");
    let clock = FakeClock::new();
    let r = boot(
        &img,
        &[
            ("FV_SKIP_DRIVER_CHECK", s("1")),
            ("FV_PROVIDER", s("brev")),
            ("FV_POD_ID", s("brev:fv-pod-b-1")),
            ("FV_WORKER_TOML_B64", b64("[server]\n")),
            ("FV_STATE_DIR", img.paths.state.display().to_string()),
            ("FV_TUNNEL_TOKEN", s("eyJhIjoisecret")),
            ("FV_TUNNEL_URL", s("https://fv-pod-b-1.tunnels.example")),
            ("FV_ENDPOINT_REPORT_URL", s("https://control/e")),
        ],
        &[],
        &sp,
        &http,
        &clock,
    );
    let plan = r.result.unwrap();
    assert!(!plan.env.contains_key("FV_TUNNEL_TOKEN"));
    assert!(r
        .log
        .lines()
        .contains(&s("[fv-boot] tunnel https://fv-pod-b-1.tunnels.example")));
    assert_eq!(
        http.bodies("https://control/e")[0]["url"],
        "https://fv-pod-b-1.tunnels.example"
    );
    // The fv-serve --version probe already runs without it.
    assert!(sp.ran("--version")[0]
        .env_remove
        .contains(&s("FV_TUNNEL_TOKEN")));
    // A tampered cloudflared download is refused.
    let http = FakeHttp::new().serve("https://example.invalid/cloudflared", b"tampered");
    std::fs::remove_file(&img.paths.cloudflared).unwrap();
    let r = boot(
        &img,
        &[
            ("FV_SKIP_DRIVER_CHECK", s("1")),
            ("FV_PROVIDER", s("brev")),
            ("FV_WORKER_TOML_B64", b64("[server]\n")),
        ],
        &[],
        &FakeSpawner::new(default_handler),
        &http,
        &FakeClock::new(),
    );
    assert!(matches!(&r.result, Err(BootError::Tunnel(m)) if m == "cloudflared sha256"));
    // A provider pod's fail sleep (PROVIDER_BOOT: 600 s).
    let clock = FakeClock::new();
    let _ = boot(
        &img,
        &[
            ("FV_SKIP_DRIVER_CHECK", s("1")),
            ("FV_PROVIDER", s("brev")),
            ("FV_WORKER_TOML_B64", b64("[server]\n")),
        ],
        &[],
        &FakeSpawner::new(default_handler),
        &FakeHttp::new(),
        &clock,
    );
    assert_eq!(clock.slept(), vec![Duration::from_secs(600)]);
}

#[test]
fn edge_front_on_runpod_and_its_watchdog() {
    let img = Image::new("edge");
    let sp = FakeSpawner::new(default_handler);
    let r = boot(
        &img,
        &[
            ("FV_SKIP_DRIVER_CHECK", s("1")),
            ("FV_SERVE_ROLE", s("worker")),
            ("RUNPOD_POD_ID", s("pod77")),
            ("FV_WORKER_TOML_B64", b64("[server]\n")),
            ("FV_DISPATCH_FRONT", s("1")),
            ("FV_CLUSTER_DEADLINE", s("1761400000")),
            ("FV_MIN_BALANCE", s("8")),
            ("FV_BACKSTOP_API_KEY", s("rp-key")),
            ("FV_STATE_DIR", img.paths.state.display().to_string()),
        ],
        &[],
        &sp,
        &FakeHttp::new(),
        &FakeClock::new(),
    );
    let plan = r.result.unwrap();
    assert_eq!(
        plan.env["FV_DISPATCH_ENDPOINT"],
        "https://pod77-8000.proxy.runpod.net"
    );
    assert_eq!(sp.ran("watchdog").len(), 1);
    assert!(sp.ran("cloudflared").is_empty());
    // The backstop key: the watchdog's alone; fv-serve and the other children never see it.
    let w = &sp.ran("watchdog")[0];
    assert!(w.env.contains(&(s("FV_BACKSTOP_API_KEY"), s("rp-key"))));
    assert!(!plan.env.contains_key("FV_BACKSTOP_API_KEY"));
    assert!(sp.ran("--print-config")[0]
        .env_remove
        .contains(&s("FV_BACKSTOP_API_KEY")));
    assert_eq!(plan.env["FV_CLUSTER_DEADLINE"], "1761400000");
}

#[test]
fn watchdog_loops() {
    let img = Image::new("wd");
    let log = CaptureLog::default();
    let sp = FakeSpawner::new(default_handler);
    // Runpod: the balance is checked every other round; below the floor the pod deletes itself.
    let http = FakeHttp::new().answer(|r| {
        if r.url == watchdog::RUNPOD_GRAPHQL {
            return Response {
                status: 200,
                body: br#"{"data":{"myself":{"clientBalance":7.5}}}"#.to_vec(),
            };
        }
        Response {
            status: 200,
            body: Vec::new(),
        }
    });
    let clock = FakeClock::new();
    let deps = Deps {
        spawner: &sp,
        http: &http,
        clock: &clock,
        log: &log,
    };
    let c = Ctx::new(BootEnv::default(), deps, img.paths.clone(), 1);
    let m = watchdog::Mode::Runpod {
        pod: "p1".into(),
        key: "k".into(),
        deadline: 1_761_400_000.0,
        min_balance: 8.0,
    };
    watchdog::run(&c, &m, Some(4321), Some(3));
    let sent = http.sent();
    let gq = sent
        .iter()
        .filter(|r| r.url == watchdog::RUNPOD_GRAPHQL)
        .count();
    assert_eq!(gq, 2);
    let del: Vec<&Request> = sent
        .iter()
        .filter(|r| r.method == sys::Method::Delete)
        .collect();
    assert_eq!(del.len(), 2);
    assert_eq!(del[0].url, "https://rest.runpod.io/v1/pods/p1");
    assert_eq!(del[0].headers[0].1, "Bearer k");
    assert!(log.has("[watchdog] balance 7.5 below 8: deleting p1"));

    // The deadline.
    let http = FakeHttp::new();
    let clock = FakeClock::new();
    let deps = Deps {
        spawner: &sp,
        http: &http,
        clock: &clock,
        log: &log,
    };
    let c = Ctx::new(BootEnv::default(), deps, img.paths.clone(), 1);
    let m = watchdog::Mode::Runpod {
        pod: "p1".into(),
        key: "k".into(),
        deadline: 0.0,
        min_balance: 0.0,
    };
    watchdog::run(&c, &m, Some(4321), Some(1));
    assert_eq!(
        http.sent()
            .iter()
            .filter(|r| r.method == sys::Method::Delete)
            .count(),
        1
    );

    // GMI: the container by name, then fv-serve.
    let http = FakeHttp::new().answer(|r| {
        if r.method == sys::Method::Get {
            return Response {
                status: 200,
                body: br#"[{"id":"c-1","name":"fv-pod-a-1"}]"#.to_vec(),
            };
        }
        Response {
            status: 200,
            body: Vec::new(),
        }
    });
    let clock = FakeClock::new();
    let sp = FakeSpawner::new(default_handler);
    let deps = Deps {
        spawner: &sp,
        http: &http,
        clock: &clock,
        log: &log,
    };
    let c = Ctx::new(BootEnv::default(), deps, img.paths.clone(), 1);
    let m = watchdog::Mode::Gmi {
        api: "https://gmi/api".into(),
        key: "g".into(),
        name: "fv-pod-a-1".into(),
        deadline: 0.0,
    };
    watchdog::run(&c, &m, Some(4321), Some(1));
    let urls: Vec<String> = http
        .sent()
        .iter()
        .map(|r| format!("{:?} {}", r.method, r.url))
        .collect();
    assert_eq!(
        urls,
        vec![
            s("Get https://gmi/api/v1/containers"),
            s("Delete https://gmi/api/v1/containers/c-1")
        ]
    );
    // fv-serve (the boot's pid) through the shell's kill builtin: no procps in the images.
    let kills: Vec<Vec<String>> = sp.ran("kill -s").into_iter().map(|c| c.args).collect();
    assert_eq!(
        kills,
        vec![vec![
            s("-c"),
            s("kill -s \"$0\" \"$1\""),
            s("TERM"),
            s("4321")
        ]]
    );
}

/// The Runpod delete counts as done only on a 2xx or a 404 (the pod is gone);
/// a 403 (a key REST refuses, ADR-0023 §4), a 5xx or another status is retried
/// (3 attempts, 5 s apart), and the deadline round repeats it a minute later.
#[test]
fn watchdog_retries_a_refused_runpod_delete() {
    let img = Image::new("wdr");
    let sp = FakeSpawner::new(default_handler);
    for (status, attempts) in [
        (200, 1),
        (204, 1),
        (404, 1),
        (403, 3),
        (401, 3),
        (500, 3),
        (302, 3),
    ] {
        let log = CaptureLog::default();
        let http = FakeHttp::new().answer(move |_| Response {
            status,
            body: br#"{"error":"x"}"#.to_vec(),
        });
        let clock = FakeClock::new();
        let deps = Deps {
            spawner: &sp,
            http: &http,
            clock: &clock,
            log: &log,
        };
        let c = Ctx::new(BootEnv::default(), deps, img.paths.clone(), 1);
        let m = watchdog::Mode::Runpod {
            pod: "p9".into(),
            key: "rpa_acct".into(),
            deadline: 0.0,
            min_balance: 0.0,
        };
        watchdog::run(&c, &m, Some(4321), Some(1));
        let del: Vec<Request> = http
            .sent()
            .into_iter()
            .filter(|r| r.method == sys::Method::Delete)
            .collect();
        assert_eq!(del.len(), attempts, "HTTP {status}");
        assert!(del
            .iter()
            .all(|r| r.url == "https://rest.runpod.io/v1/pods/p9"));
        // 5 s between attempts, then the deadline round's minute.
        let mut want = vec![Duration::from_secs(5); attempts - 1];
        want.push(Duration::from_secs(60));
        assert_eq!(clock.slept(), want, "HTTP {status}");
        assert_eq!(
            log.has(&format!("[watchdog] delete p9: HTTP {status}")),
            attempts > 1,
            "HTTP {status}"
        );
        // The key is never logged.
        assert!(log.lines().iter().all(|l| !l.contains("rpa_acct")));
    }
    for (status, done) in [
        (200, true),
        (299, true),
        (404, true),
        (403, false),
        (400, false),
        (500, false),
    ] {
        let r = Response {
            status,
            body: Vec::new(),
        };
        assert_eq!(watchdog::runpod_deleted(&r), done, "HTTP {status}");
    }
}

#[test]
fn upsampler_sidecar_start_wait_and_report() {
    let img = Image::new("ups");
    exe(&img.paths.upsampler, b"#!/bin/sh\n");
    let sp = FakeSpawner::new(|c| {
        if c.program.ends_with("fv-upsampler") && c.args == ["wait"] {
            return Did {
                stdout: "export FV_UPSAMPLER_BASE_URL='http://127.0.0.1:18080/v1'\nexport FV_UPSAMPLER_MODEL='qwen3-vl-4b-instruct'\nexport FV_UPSAMPLER_LOCAL='1'\n".into(),
                stderr: "[fv-boot] 10:00:04 step upsampler: 41.2s (llama-server 127.0.0.1:18080, qwen3-vl-4b-instruct)\n".into(),
                ..Did::default()
            };
        }
        default_handler(c)
    });
    let http = FakeHttp::new();
    let r = boot(
        &img,
        &[
            ("FV_SKIP_DRIVER_CHECK", s("1")),
            ("FV_UPSAMPLER_LOCAL", s("1")),
            ("FV_STATE_DIR", img.paths.state.display().to_string()),
            ("FV_SLS_REPORT_URL", s(SLS_URL)),
            ("RUNPOD_POD_ID", s("wkr-abc123")),
            ("FV_WORKER_CONFIG", s("/etc/fv/runpod-fake.toml")),
        ],
        &[],
        &sp,
        &http,
        &FakeClock::new(),
    );
    let plan = r.result.unwrap();
    assert_eq!(
        plan.env["FV_UPSAMPLER_BASE_URL"],
        "http://127.0.0.1:18080/v1"
    );
    assert_eq!(plan.env["FV_UPSAMPLER_MODEL"], "qwen3-vl-4b-instruct");
    let calls: Vec<Vec<String>> = sp.ran("fv-upsampler").into_iter().map(|c| c.args).collect();
    assert_eq!(calls, vec![vec![s("start")], vec![s("wait")]]);
    assert_eq!(
        http.bodies(SLS_URL)[0]["detail"],
        "upsampler 41.2s (llama-server 127.0.0.1:18080, qwen3-vl-4b-instruct)"
    );
    assert!(r.log.has("[fv-boot] 10:00:04 step upsampler: 41.2s"));
    assert!(img.paths.state.join("upsampler-wait.log").is_file());

    // An image without the sidecar: said, and the boot goes on.
    std::fs::remove_file(&img.paths.upsampler).unwrap();
    let r = boot(
        &img,
        &[
            ("FV_SKIP_DRIVER_CHECK", s("1")),
            ("FV_UPSAMPLER_LOCAL", s("1")),
            ("FV_WORKER_CONFIG", s("/etc/fv/runpod-fake.toml")),
        ],
        &[],
        &FakeSpawner::new(default_handler),
        &FakeHttp::new(),
        &FakeClock::new(),
    );
    assert!(r.result.is_ok());
    assert!(r
        .log
        .has("upsampler: FAILED: this image has no llama-server sidecar"));
}

#[test]
fn registry_and_unknown_sources_fail_loudly() {
    let img = Image::new("reg");
    let r = boot(
        &img,
        &[
            ("FV_SKIP_DRIVER_CHECK", s("1")),
            ("FV_WEIGHTS_SOURCE", s("registry")),
            ("FV_WORKER_CONFIG", s("/etc/fv/runpod-fake.toml")),
        ],
        &[],
        &FakeSpawner::new(default_handler),
        &FakeHttp::new(),
        &FakeClock::new(),
    );
    match &r.result {
        Err(
            e @ BootError::SourceNotAvailable {
                name: "registry", ..
            },
        ) => assert!(e.to_string().contains("model-registry-rs WO-002")),
        o => panic!("{o:?}"),
    }
    let r = boot(
        &img,
        &[
            ("FV_SKIP_DRIVER_CHECK", s("1")),
            ("FV_WEIGHTS_SOURCE", s("s3")),
            ("FV_WORKER_CONFIG", s("/etc/fv/runpod-fake.toml")),
        ],
        &[],
        &FakeSpawner::new(default_handler),
        &FakeHttp::new(),
        &FakeClock::new(),
    );
    assert!(matches!(&r.result, Err(BootError::UnknownSource(v)) if v == "s3"));
    let r = boot(
        &img,
        &[
            ("FV_SKIP_DRIVER_CHECK", s("1")),
            ("FV_WEIGHTS_SOURCE", s("none")),
            ("FV_WORKER_CONFIG", s("/etc/fv/runpod-fake.toml")),
        ],
        &[],
        &FakeSpawner::new(default_handler),
        &FakeHttp::new(),
        &FakeClock::new(),
    );
    assert!(r.result.is_ok());
}

#[test]
fn hub_refuses_a_tree_that_is_not_the_recorded_one() {
    let img = Image::new("hubbad");
    let rows = "tree\th3-base\tFastVideo/H3-Base\tabc123\t*\n";
    let sums = "h3-base\ttransformer/model.safetensors\t-\tdeadbeef\n";
    let sp = FakeSpawner::new(|c| {
        if c.args.iter().any(|a| a.ends_with("fetch-hub-tree.py")) {
            let get = |k: &str| {
                c.env
                    .iter()
                    .find(|(x, _)| x == k)
                    .map(|(_, v)| v.clone())
                    .unwrap_or_default()
            };
            let d = Path::new(&get("FETCH_WEIGHTS")).join(get("FETCH_DEST"));
            std::fs::create_dir_all(&d).unwrap();
            let srv = Path::new(&get("FETCH_SRV")).to_path_buf();
            std::fs::write(srv.join("DONE"), b"0").unwrap();
            std::fs::write(
                srv.join("sha256.txt"),
                "cafe  transformer/model.safetensors\n",
            )
            .unwrap();
            return ok();
        }
        default_handler(c)
    });
    let http = FakeHttp::new().serve("https://s/fetch-hub-tree.py", b"x");
    let r = boot(
        &img,
        &[
            ("FV_SKIP_DRIVER_CHECK", s("1")),
            ("FV_WEIGHTS", img.weights().display().to_string()),
            ("FV_STATE_DIR", img.paths.state.display().to_string()),
            ("FV_WEIGHTS_SOURCE", s("hub")),
            ("FV_WEIGHTS_TREES_B64", b64(rows)),
            ("FV_HUB_SHA256_B64", b64(sums)),
            ("FV_SCRIPTS_URL", s("https://s")),
            ("FV_WORKER_CONFIG", s("/etc/fv/runpod-fake.toml")),
        ],
        &[],
        &sp,
        &http,
        &FakeClock::new(),
    );
    assert!(
        matches!(&r.result, Err(BootError::Weights(m)) if m == "tree h3-base: not the recorded SHA-256 (weights-sha256.tsv): transformer/model.safetensors")
    );
    assert!(
        !img.weights().join("h3-base").exists(),
        "the refused tree is removed"
    );
}

#[test]
fn informational_flags_skip_the_boot() {
    let a = |v: &[&str]| v.iter().map(|x| (*x).to_owned()).collect::<Vec<_>>();
    assert!(passthrough(&a(&["--version"])));
    assert!(passthrough(&a(&[
        "--config",
        "/etc/fv/x.toml",
        "--print-config"
    ])));
    assert!(!passthrough(&a(&["--config", "/etc/fv/x.toml"])));
    assert!(!passthrough(&[]));
}

#[test]
fn a_controller_worker_without_a_config_fails_like_the_shell() {
    let img = Image::new("noconf");
    let r = boot(
        &img,
        &[
            ("FV_SKIP_DRIVER_CHECK", s("1")),
            ("FV_SERVE_ROLE", s("worker")),
            ("RUNPOD_POD_ID", s("p1")),
            ("FV_CONFIG", s("/etc/fv/runpod.toml")),
        ],
        &[],
        &FakeSpawner::new(default_handler),
        &FakeHttp::new(),
        &FakeClock::new(),
    );
    assert!(
        matches!(&r.result, Err(BootError::Config(m)) if m.starts_with("no worker config: the image serves no default"))
    );
    let r = boot(
        &img,
        &[
            ("FV_SKIP_DRIVER_CHECK", s("1")),
            ("FV_PROVIDER", s("gmi")),
            ("FV_CONFIG", s("/etc/fv/runpod.toml")),
        ],
        &[],
        &FakeSpawner::new(default_handler),
        &FakeHttp::new(),
        &FakeClock::new(),
    );
    assert!(matches!(&r.result, Err(BootError::Config(_))));
}

#[test]
fn a_refused_config_logs_fv_serves_own_lines() {
    let img = Image::new("cfglines");
    let line = fixture()["log_markers"]
        .as_array()
        .unwrap()
        .iter()
        .find(|m| m["fv_serve_fatal"] == true)
        .unwrap()["line"]
        .as_str()
        .unwrap()
        .to_owned();
    let l2 = line.clone();
    let sp = FakeSpawner::new(move |c| {
        if c.args.iter().any(|a| a == "--print-config") {
            return Did {
                code: 2,
                stderr: format!("{l2}\n"),
                ..Did::default()
            };
        }
        default_handler(c)
    });
    let r = boot(
        &img,
        &[
            ("FV_SKIP_DRIVER_CHECK", s("1")),
            ("FV_WORKER_CONFIG", s("/etc/fv/x.toml")),
        ],
        &[],
        &sp,
        &FakeHttp::new(),
        &FakeClock::new(),
    );
    assert!(r.result.is_err());
    // Verbatim, at the start of a line: podlogs.ts CONTAINER_FATAL `^fv-serve: `.
    assert!(r.log.lines().contains(&line), "{:?}", r.log.lines());
}

#[test]
fn reports_stop_on_any_http_answer() {
    let http = FakeHttp::new().answer(|_| Response {
        status: 401,
        body: Vec::new(),
    });
    let clock = FakeClock::new();
    let rep = Reporter::from_env(
        &BootEnv::from_pairs([
            ("FV_ENDPOINT_REPORT_URL", "https://c/e"),
            ("FV_POD_ID", "gmi:fv-a-1"),
        ]),
        1.0,
    );
    rep.failed(&http, &clock, "x");
    assert_eq!(http.sent().len(), 1);
    assert!(clock.slept().is_empty());
    // No answer at all (a connection error): retried, 5 x with 5 s pauses.
    struct Down(Mutex<usize>);
    impl Http for Down {
        fn send(&self, _r: &Request) -> Result<Response, SysError> {
            *self.0.lock().unwrap() += 1;
            Err(SysError::Http {
                url: "u".into(),
                why: "connection refused".into(),
            })
        }
        fn download(&self, url: &str, _d: &Path, _t: Duration) -> Result<u64, SysError> {
            Err(SysError::Status {
                url: url.into(),
                status: 404,
            })
        }
    }
    let down = Down(Mutex::new(0));
    let clock = FakeClock::new();
    rep.failed(&down, &clock, "x");
    assert_eq!(*down.0.lock().unwrap(), 5);
    assert_eq!(clock.slept().len(), 4);
}

#[test]
fn fv_boot_legacy_runs_only_fv_entrys_steps() {
    let img = Image::new("legacy");
    std::fs::create_dir_all(&img.paths.runpod_volume_weights).unwrap();
    let sp = FakeSpawner::new(default_handler);
    let log = CaptureLog::default();
    let clock = FakeClock::new();
    let http = FakeHttp::new();
    let deps = Deps {
        spawner: &sp,
        http: &http,
        clock: &clock,
        log: &log,
    };
    let env = BootEnv::from_pairs([
        ("FV_BOOT_LEGACY", "1"),
        ("FV_CONFIG", "/etc/fv/runpod.toml"),
        ("FV_WEIGHTS_SOURCE", "registry"),
    ]);
    assert!(legacy_requested(&env));
    assert!(!legacy_requested(&BootEnv::from_pairs([(
        "FV_BOOT_LEGACY",
        "0"
    )])));
    let mut c = Ctx::new(env, deps, img.paths.clone(), 1);
    let plan = legacy(&mut c, &[s("--config"), s("/etc/fv/x.toml")]).unwrap();
    assert_eq!(plan.program, img.paths.serve_bin);
    assert_eq!(plan.args, vec![s("--config"), s("/etc/fv/x.toml")]);
    assert_eq!(
        plan.env["FV_WEIGHTS"],
        img.paths.runpod_volume_weights.display().to_string()
    );
    assert_eq!(
        std::fs::read_link(&img.paths.workspace_weights).unwrap(),
        img.paths.runpod_volume_weights
    );
    // Nothing else: no driver check, no config check, no weights plan (registry would fail), no report.
    assert!(sp.calls().is_empty(), "{:?}", sp.calls());
    assert!(http.sent().is_empty());
}

#[test]
fn provider_r2_reports_checking_and_cleans_leftovers() {
    let img = Image::new("provr2");
    let sha = r2_fetcher(&img);
    std::fs::create_dir_all(img.weights().join(".demo.partial-123")).unwrap();
    std::fs::create_dir_all(img.weights().join(".stale-x-1")).unwrap();
    let report = "https://control/ingest/v1/endpoint";
    let sp = FakeSpawner::new(|c| {
        if c.program == "python3" {
            fake_fetch(c, false)
        } else {
            default_handler(c)
        }
    });
    let http = FakeHttp::new();
    let r = boot(
        &img,
        &[
            ("FV_SKIP_DRIVER_CHECK", s("1")),
            ("FV_PROVIDER", s("brev")),
            ("FV_POD_ID", s("brev:fv-pod-b-1")),
            ("FV_WORKER_TOML_B64", b64("[server]\n")),
            ("FV_WEIGHTS", img.weights().display().to_string()),
            ("FV_STATE_DIR", img.paths.state.display().to_string()),
            ("FV_WEIGHTS_SOURCE", s("r2")),
            ("FV_WEIGHTS_PLAN_URL", s(PLAN_URL)),
            ("FV_R2_FETCH_SHA256", sha),
            ("FV_ENDPOINT_REPORT_URL", s(report)),
            ("FV_ENDPOINT_REPORT_TOKEN", s("ingest_tok")),
            ("FV_TUNNEL_URL", s("")),
        ],
        &[],
        &sp,
        &http,
        &FakeClock::new(),
    );
    // The tunnel fails here (no cloudflared download in this fake), after the weights.
    assert!(
        matches!(&r.result, Err(BootError::Tunnel(_))),
        "{:?}",
        r.result
    );
    let b = http.bodies(report);
    assert_eq!(b[0]["phase"], "weights");
    assert_eq!(b[0]["detail"], "checking");
    assert!(!img.weights().join(".demo.partial-123").exists());
    assert!(!img.weights().join(".stale-x-1").exists());
    // FV_R2_SKIP_TREES was unset and nvidia-smi says nothing: it stays unset for the fetcher.
    let f = &sp.ran("--plan-url")[0];
    assert!(
        !f.env.iter().any(|(k, _)| k == "FV_R2_SKIP_TREES"),
        "{:?}",
        f.env
    );
}

#[test]
fn signals_go_through_the_shell_builtin_to_a_real_process() {
    // A real child, a real `sh -c 'kill -s TERM "$1"'` (RealSpawner): no kill / pkill binary needed.
    let mut child = std::process::Command::new("sleep")
        .arg("30")
        .spawn()
        .unwrap();
    let pid = child.id();
    let o = sys::RealSpawner
        .output(&sys::kill_cmd("TERM", pid))
        .unwrap();
    assert!(o.ok(), "{}", o.stderr_text());
    let st = child.wait().unwrap();
    assert!(!st.success());

    // The guard stopping a "fv-serve" (a real process) after a failed streaming fetch.
    let img = Image::new("realkill");
    let st_dir = img.paths.state.clone();
    std::fs::create_dir_all(&st_dir).unwrap();
    std::fs::write(st_dir.join("r2-stream-ready"), b"").unwrap();
    let mut serve = std::process::Command::new("sleep")
        .arg("30")
        .spawn()
        .unwrap();
    let log = CaptureLog::default();
    let clock = FakeClock::new();
    let http = FakeHttp::new();
    let sp = sys::RealSpawner;
    let deps = Deps {
        spawner: &sp,
        http: &http,
        clock: &clock,
        log: &log,
    };
    let c = Ctx::with_t0(BootEnv::default(), deps, img.paths.clone(), 1, 1.0);
    let code = weights::r2::guard(
        &c,
        serve.id(),
        &st_dir,
        "sh",
        &[s("-c"), s("echo 'r2-fetch: FAILED: part 9'; exit 3")],
    );
    assert_eq!(code, 3);
    let st = serve.wait().unwrap();
    assert!(!st.success(), "fv-serve was stopped");
    assert!(log.has("FAILED: r2 weights while fv-serve was loading: r2-fetch: FAILED: part 9: stopping fv-serve"));
}

// ------------------------------------------------------------------ the protocol fixture

fn fixture() -> Value {
    serde_json::from_str(include_str!("../../fixtures/boot-protocol.json")).unwrap()
}

#[test]
fn fixture_reports_are_the_bytes_fv_boot_sends() {
    let f = fixture();
    for r in f["pod_reports"].as_array().unwrap() {
        let env = BootEnv::from_pairs([
            ("FV_ENDPOINT_REPORT_URL", "https://c/e"),
            ("FV_POD_ID", r["pod"].as_str().unwrap()),
        ]);
        let rep = Reporter::from_env(&env, 1.0);
        assert_eq!(
            rep.body(
                r["phase"].as_str().unwrap(),
                r["url"].as_str().unwrap(),
                r["detail"].as_str().unwrap()
            )
            .unwrap(),
            r["body"].as_str().unwrap()
        );
    }
    for r in f["sls_reports"].as_array().unwrap() {
        let pod = r["pod"].as_str().unwrap();
        let (w, t) = pod.split_once('@').unwrap();
        let env = BootEnv::from_pairs([("FV_SLS_REPORT_URL", "https://c/s"), ("RUNPOD_POD_ID", w)]);
        let rep = Reporter::from_env(&env, t.parse::<f64>().unwrap() + 0.9);
        assert_eq!(
            rep.body(
                r["phase"].as_str().unwrap(),
                "ignored",
                r["detail"].as_str().unwrap()
            )
            .unwrap(),
            r["body"].as_str().unwrap()
        );
    }
}

#[test]
fn fixture_constants_and_markers_match_the_code() {
    let f = fixture();
    let k = &f["constants"];
    assert_eq!(k["min_driver_major"], driver::MIN_DRIVER_MAJOR);
    assert_eq!(k["cuda13_symbol"], driver::CUDA13_SYMBOL);
    assert_eq!(k["driver_too_old"], driver::DRIVER_TOO_OLD);
    assert_eq!(k["cloudflared_url"], tunnel::CLOUDFLARED_URL);
    assert_eq!(k["cloudflared_sha256"], tunnel::CLOUDFLARED_SHA256);
    assert_eq!(k["mxfp8_tree_glob"], weights::r2::MXFP8_TREE_GLOB);
    assert_eq!(k["streaming_fetch_marker"], STREAMING_FETCH_MARKER);
    let p = Paths::image();
    assert_eq!(k["serve_bin"], p.serve_bin.display().to_string());
    assert_eq!(k["upsampler_bin"], p.upsampler.display().to_string());
    assert_eq!(k["r2_fetch_image"], p.r2_fetch_image.display().to_string());
    assert_eq!(k["worker_toml"], p.worker_toml.display().to_string());
    assert!(k["fv_boot_bin"].as_str().unwrap().ends_with("/fv-boot"));

    // The markers are what Log prints.
    let log = CaptureLog::default();
    let clock = FakeClock::new();
    let l = Log::new(&log, &clock, 1_761_300_000.0);
    for m in f["log_markers"].as_array().unwrap() {
        let line = m["line"].as_str().unwrap();
        if let Some(v) = m["volume"].as_str() {
            l.marker(&format!("volume: {v}"));
            assert_eq!(log.lines().last().unwrap(), line);
        } else if m["brev_tunnel"] == true {
            l.marker(line.trim_start_matches("[fv-boot] "));
            assert_eq!(log.lines().last().unwrap(), line);
        } else if m["driver_too_old"] == true {
            let msg = line.split_once("FAILED: ").unwrap().1;
            assert_eq!(
                driver::verdict(&driver::Facts {
                    driver: Some("550.54.15".into()),
                    ..Default::default()
                })
                .unwrap_err(),
                msg
            );
            l.say(&format!("FAILED: {msg}"));
            let got = log.lines().last().unwrap().clone();
            assert!(
                got.starts_with("[fv-boot] ") && got.ends_with(&format!("FAILED: {msg}")),
                "{got}"
            );
        } else if m["fv_serve_fatal"] == true {
            // fv-serve's own line, logged verbatim by the config check (a_refused_config_logs_fv_serves_own_lines).
            assert!(line.starts_with("fv-serve: "));
        } else {
            assert!(line.starts_with("[fv-boot] start ("));
        }
    }
}
