//! `weights_source r2` end to end with the real `scripts/gpu/r2-fetch.py`
//! against a fake chunk source: an in-process HTTP server that answers
//! fv-control's plan request (presigned-style per-file URLs, part hashes) and
//! ranged GETs, and records every report. The boot runs real processes (the
//! fetcher, a stub fv-serve, the `fv-boot fetch-guard` binary for streaming);
//! only fv-boot's own report HTTP is a recorder. Skipped without python3.

#![cfg(feature = "boot")]

use std::collections::BTreeMap;
use std::net::SocketAddr;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use axum::body::Body;
use axum::extract::{Path as AxPath, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::Response;
use axum::routing::{get, post};
use axum::Router;
use fastvideo_deploy::boot::sys::{
    which, Http, LogSink, Method, RealClock, RealSpawner, Request, Response as HttpResponse,
    SysError,
};
use fastvideo_deploy::boot::weights::hex;
use fastvideo_deploy::boot::{run, BootEnv, Ctx, Deps, Paths};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

const PART: usize = 65_536;

#[derive(Default)]
struct Fake {
    files: BTreeMap<String, Vec<u8>>,
    reports: Mutex<Vec<Value>>,
    plan_auth: Mutex<Vec<String>>,
    gets: Mutex<usize>,
}

fn sha(b: &[u8]) -> String {
    hex(&Sha256::digest(b))
}

async fn plan(
    State((f, base)): State<(Arc<Fake>, String)>,
    h: HeaderMap,
    body: String,
) -> Response {
    let auth = h
        .get("authorization")
        .and_then(|v| v.to_str().ok())
        .unwrap_or_default()
        .to_owned();
    f.plan_auth.lock().unwrap().push(format!("{auth} {body}"));
    if auth != "Bearer fvw_tok" {
        return Response::builder().status(401).body(Body::empty()).unwrap();
    }
    let files: Vec<Value> = f
        .files
        .iter()
        .map(|(p, b)| {
            json!({"path": p, "size": b.len(), "sha256": sha(b), "parts": b.chunks(PART).map(sha).collect::<Vec<_>>(), "url": format!("{base}/obj/demo/rev1/{p}")})
        })
        .collect();
    let total: usize = f.files.values().map(Vec::len).sum();
    let p = json!({"format": "fv-weights-plan/1", "expires_at": 4_000_000_000u64, "trees": [{"dest": "demo", "rev": "rev1", "part_size": PART, "total_bytes": total, "files": files}]});
    Response::builder()
        .status(200)
        .header("content-type", "application/json")
        .body(Body::from(p.to_string()))
        .unwrap()
}

async fn obj(
    State((f, _)): State<(Arc<Fake>, String)>,
    AxPath(path): AxPath<String>,
    h: HeaderMap,
) -> Response {
    *f.gets.lock().unwrap() += 1;
    let key = path.trim_start_matches("demo/rev1/");
    let Some(b) = f.files.get(key) else {
        return Response::builder().status(404).body(Body::empty()).unwrap();
    };
    let range = h
        .get("range")
        .and_then(|v| v.to_str().ok())
        .and_then(|r| r.strip_prefix("bytes="))
        .and_then(|r| r.split_once('-'));
    match range {
        Some((a, z)) => {
            let a: usize = a.parse().unwrap_or(0);
            let z: usize = z
                .parse()
                .map(|z: usize| z.min(b.len() - 1))
                .unwrap_or(b.len() - 1);
            Response::builder()
                .status(206)
                .header("content-range", format!("bytes {a}-{z}/{}", b.len()))
                .header("content-length", (z + 1 - a).to_string())
                .body(Body::from(b[a..=z].to_vec()))
                .unwrap()
        }
        None => Response::builder()
            .status(200)
            .body(Body::from(b.clone()))
            .unwrap(),
    }
}

async fn report(State((f, _)): State<(Arc<Fake>, String)>, body: String) -> StatusCode {
    if let Ok(v) = serde_json::from_str(&body) {
        f.reports.lock().unwrap().push(v);
    }
    StatusCode::OK
}

fn serve(f: Arc<Fake>) -> String {
    let (tx, rx) = std::sync::mpsc::channel::<SocketAddr>();
    std::thread::spawn(move || {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        rt.block_on(async move {
            let l = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
            let addr = l.local_addr().unwrap();
            let base = format!("http://{addr}");
            let app = Router::new()
                .route("/plan", post(plan))
                .route("/obj/{*path}", get(obj))
                .route("/report", post(report))
                .with_state((f, base));
            tx.send(addr).unwrap();
            axum::serve(l, app).await.unwrap();
        });
    });
    format!("http://{}", rx.recv().unwrap())
}

/// fv-boot's own reports, recorded (the fetcher posts to the server itself).
#[derive(Default)]
struct Recorder(Mutex<Vec<Value>>);
impl Http for Recorder {
    fn send(&self, req: &Request) -> Result<HttpResponse, SysError> {
        if req.method == Method::Post {
            if let Some(v) = req
                .body
                .as_deref()
                .and_then(|b| serde_json::from_slice(b).ok())
            {
                self.0.lock().unwrap().push(v);
            }
        }
        Ok(HttpResponse {
            status: 200,
            body: Vec::new(),
        })
    }
    fn download(&self, url: &str, _d: &Path, _t: Duration) -> Result<u64, SysError> {
        Err(SysError::Status {
            url: url.into(),
            status: 404,
        })
    }
}

#[derive(Default)]
struct Lines(Mutex<Vec<String>>);
impl LogSink for Lines {
    fn line(&self, s: &str) {
        eprintln!("{s}");
        self.0.lock().unwrap().push(s.to_owned());
    }
}

fn exe(p: &Path, body: &str) {
    std::fs::create_dir_all(p.parent().unwrap()).unwrap();
    std::fs::write(p, body).unwrap();
    std::fs::set_permissions(p, std::fs::Permissions::from_mode(0o755)).unwrap();
}

struct Tree {
    root: PathBuf,
    paths: Paths,
}

fn image(name: &str, stream: bool) -> Tree {
    let root = std::env::temp_dir().join(format!("fvb-it-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    let r = |p: &str| root.join(p);
    let mut paths = Paths::image();
    paths.worker_toml = r("fv-worker.toml");
    paths.runpod_volume_weights = r("runpod-volume/weights");
    paths.workspace_weights = r("workspace/weights");
    paths.serve_bin = r("bin/fv-serve");
    paths.upsampler = r("bin/fv-upsampler");
    paths.r2_fetch_image = r("bin/r2-fetch.py");
    paths.tools = r("opt/fv");
    paths.state = r("fvstate");
    paths.self_exe = PathBuf::from(env!("CARGO_BIN_EXE_fv-boot"));
    let marker = if stream { "# fv-streaming-fetch/1" } else { "" };
    exe(&paths.serve_bin, &format!("#!/bin/sh\n{marker}\ncase \"$*\" in *--print-config*) echo '{{}}' ;; *--version*) echo 'fv-serve stub' ;; esac\nexit 0\n"));
    let fetcher = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../scripts/gpu/r2-fetch.py");
    std::fs::copy(fetcher, &paths.r2_fetch_image).unwrap();
    Tree { root, paths }
}

fn files() -> BTreeMap<String, Vec<u8>> {
    let mut m = BTreeMap::new();
    m.insert("config.json".into(), br#"{"x": 1}"#.to_vec());
    m.insert(
        "transformer/model.safetensors".into(),
        (0..300_000u32).map(|i| (i * 7 % 251) as u8).collect(),
    );
    m.insert(
        "text_encoder/model.safetensors".into(),
        (0..150_000u32).map(|i| (i * 13 % 241) as u8).collect(),
    );
    m
}

fn boot(
    t: &Tree,
    base: &str,
) -> (
    Result<fastvideo_deploy::boot::ExecPlan, fastvideo_deploy::boot::BootError>,
    Vec<Value>,
    Vec<String>,
) {
    let sha = fastvideo_deploy::boot::weights::sha256_file(&t.paths.r2_fetch_image)
        .unwrap()
        .0;
    let vars: Vec<(&str, String)> = vec![
        ("FV_SKIP_DRIVER_CHECK", "1".into()),
        // The image bakes no config (ADR-0011): the boot is sent one.
        ("FV_WORKER_CONFIG", "/etc/fv/runpod-fake.toml".into()),
        ("RUNPOD_ENDPOINT_ID", "ep".into()),
        ("RUNPOD_POD_ID", "wkr-it".into()),
        ("FV_WEIGHTS_SOURCE", "r2".into()),
        (
            "FV_WEIGHTS",
            t.paths.workspace_weights.display().to_string(),
        ),
        ("FV_STATE_DIR", t.paths.state.display().to_string()),
        ("FV_WEIGHTS_PLAN_URL", format!("{base}/plan")),
        ("FV_R2_FETCH_SHA256", sha),
        ("FV_WEIGHTS_TOKEN", "fvw_tok".into()),
        ("FV_SLS_REPORT_URL", format!("{base}/report")),
        ("FV_R2_TREE_REVS", "demo@rev1".into()),
        ("FV_R2_SKIP_TREES", String::new()),
        ("FV_R2_MIN_FREE_GB", "0".into()),
        ("FV_R2_CONCURRENCY", "4".into()),
        ("FV_STREAM_KILL_GRACE_S", "1".into()),
        ("NO_PROXY", "127.0.0.1".into()),
        ("no_proxy", "127.0.0.1".into()),
        ("PATH", std::env::var("PATH").unwrap_or_default()),
    ];
    let (sp, clock, log, http) = (
        RealSpawner,
        RealClock,
        Lines::default(),
        Recorder::default(),
    );
    let deps = Deps {
        spawner: &sp,
        http: &http,
        clock: &clock,
        log: &log,
    };
    let mut c = Ctx::new(
        BootEnv::from_pairs(vars.iter().map(|(k, v)| (*k, v.clone()))),
        deps,
        t.paths.clone(),
        std::process::id(),
    );
    // The child processes (the fetcher, the guard) get the boot's environment
    // through its exports here; in an image it is the process environment.
    for (k, v) in &vars {
        c.export(k, v.clone());
    }
    let r = run(&mut c, &[]);
    let reps = http.0.lock().unwrap().clone();
    let lines = log.0.lock().unwrap().clone();
    (r, reps, lines)
}

fn have_python() -> bool {
    let ok = which("python3", std::env::var("PATH").ok().as_deref()).is_some();
    if !ok {
        eprintln!("skipped: no python3 on PATH");
    }
    ok
}

#[test]
fn r2_with_the_real_fetcher_cold_then_warm() {
    if !have_python() {
        return;
    }
    let fake = Arc::new(Fake {
        files: files(),
        ..Fake::default()
    });
    let base = serve(Arc::clone(&fake));
    let t = image("cold", false);
    let (r, reps, lines) = boot(&t, &base);
    let plan = r.unwrap();
    for (p, b) in &fake.files {
        assert_eq!(
            &std::fs::read(t.paths.workspace_weights.join("demo").join(p)).unwrap(),
            b,
            "{p}"
        );
    }
    assert_eq!(
        std::fs::read_to_string(t.paths.workspace_weights.join("demo/.fv-revision")).unwrap(),
        "rev1"
    );
    assert!(fake.plan_auth.lock().unwrap()[0].starts_with("Bearer fvw_tok {\"pod\": \"wkr-it@"));
    // One ranged GET per two parts (the fetcher's default --get-parts).
    let parts: usize = fake
        .files
        .values()
        .map(|b| b.len().div_ceil(PART).div_ceil(2))
        .sum();
    assert!(*fake.gets.lock().unwrap() >= parts);
    let d: Vec<String> = reps
        .iter()
        .map(|x| {
            format!(
                "{} {}",
                x["phase"].as_str().unwrap(),
                x["detail"].as_str().unwrap()
            )
        })
        .collect();
    assert!(d.iter().any(|x| x.starts_with("cache miss: demo")), "{d:?}");
    assert!(
        d.iter()
            .any(|x| x.starts_with("step fetch ") && x.contains("weights ready: 1 tree(s)")),
        "{d:?}"
    );
    assert!(
        d.last().unwrap().starts_with("serve weights ready under"),
        "{d:?}"
    );
    // The fetcher reported its own progress / summary to the same endpoint.
    assert!(fake
        .reports
        .lock()
        .unwrap()
        .iter()
        .any(|x| x["phase"] == "weights"
            && x["detail"]
                .as_str()
                .unwrap()
                .starts_with("r2 weights ready")));
    assert!(!plan.env.contains_key("FASTVIDEO_STREAMING_FETCH"));
    assert!(lines
        .iter()
        .any(|l| l.contains("step fetcher: ") && l.contains("(from the image)")));

    // Warm: the cache check asks fv-control for nothing.
    let n = fake.plan_auth.lock().unwrap().len();
    let (r, reps, _) = boot(&t, &base);
    r.unwrap();
    assert_eq!(fake.plan_auth.lock().unwrap().len(), n);
    assert!(reps
        .iter()
        .any(|x| x["phase"] == "cache" && x["detail"] == "hit"));
    let _ = std::fs::remove_dir_all(&t.root);
}

#[test]
fn r2_streaming_through_the_fetch_guard_binary() {
    if !have_python() {
        return;
    }
    let fake = Arc::new(Fake {
        files: files(),
        ..Fake::default()
    });
    let base = serve(Arc::clone(&fake));
    let t = image("stream", true);
    let (r, _reps, _) = boot(&t, &base);
    let plan = r.unwrap();
    assert_eq!(plan.env["FASTVIDEO_STREAMING_FETCH"], "1");
    // The guard (a real `fv-boot fetch-guard`) runs the fetch to its end and records its exit code.
    let rc = t.paths.state.join("r2-fetch.rc");
    for _ in 0..600 {
        if rc.exists() {
            break;
        }
        std::thread::sleep(Duration::from_millis(100));
    }
    assert_eq!(std::fs::read_to_string(&rc).unwrap(), "0");
    for (p, b) in &fake.files {
        assert_eq!(
            &std::fs::read(t.paths.workspace_weights.join("demo").join(p)).unwrap(),
            b,
            "{p}"
        );
    }
    assert!(t.paths.workspace_weights.join("demo/.complete").is_file());
    // The guard's own reports went to the endpoint (its HTTP is real): the fetch step.
    let steps: Vec<String> = fake
        .reports
        .lock()
        .unwrap()
        .iter()
        .filter(|x| x["phase"] == "step")
        .map(|x| x["detail"].as_str().unwrap().to_owned())
        .collect();
    assert!(steps.iter().any(|s| s.starts_with("fetch ")), "{steps:?}");
    assert!(fake
        .reports
        .lock()
        .unwrap()
        .iter()
        .all(|x| x["pod"].as_str().unwrap().starts_with("wkr-it@")));
    let _ = std::fs::remove_dir_all(&t.root);
}
