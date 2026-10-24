//! Golden `/fv/v1/capabilities` (and `/v1/models`) for every shipped serve
//! config (`configs/serve/*.toml`, `tests/compat/serve.toml`), rendered on
//! the fake engine serving the config's models with their real caps (plan
//! A3: the API must not change while the engine's catalog becomes a
//! projection of `families/*.toml`).
//!
//! The golden is `tests/goldens/capabilities.json`, one object per config.
//! A difference fails the test and names the config and the first differing
//! path; a deliberate change is regenerated with `FV_GOLDEN_WRITE=<file>
//! cargo test -p fastvideo-serve --test capabilities_golden` (on the build
//! pod: a file under its target directory, fetched with `build-pod.sh
//! fetch`) and listed in the PR (never silently).

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::Duration;

use axum::body::Body;
use axum::http::Request;
use fastvideo_engine_service::{
    EngineConfig, EngineService, FakeBackend, FakeConfig, FakeTiming, Mp4Mode,
};
use fastvideo_protocol::ModelId;
use fastvideo_serve::config::{Config, EngineBackendKind};
use fastvideo_serve::{App, Overrides};
use fastvideo_serve_kit::KeyRing;
use serde_json::{json, Value};
use tower::ServiceExt;

const KEY: &str = "sk-golden";

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn golden_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/goldens/capabilities.json")
}

/// Every shipped config: `configs/serve/*.toml` sorted by name, then the
/// compat suite's.
fn configs() -> Vec<(String, PathBuf)> {
    let dir = root().join("configs/serve");
    let mut out: Vec<(String, PathBuf)> = std::fs::read_dir(&dir)
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|x| x == "toml"))
        .map(|p| (p.file_stem().unwrap().to_string_lossy().into_owned(), p))
        .collect();
    out.sort();
    out.push((
        "compat-serve".to_owned(),
        root().join("tests/compat/serve.toml"),
    ));
    out
}

async fn get(app: &axum::Router, uri: &str) -> (u16, Value) {
    let req = Request::builder()
        .method("GET")
        .uri(uri)
        .header("authorization", format!("Bearer {KEY}"))
        .body(Body::empty())
        .unwrap();
    let resp = app.clone().oneshot(req).await.unwrap();
    let status = resp.status().as_u16();
    let bytes = axum::body::to_bytes(resp.into_body(), 1 << 24)
        .await
        .unwrap();
    let v = serde_json::from_slice(&bytes)
        .unwrap_or_else(|_| Value::String(String::from_utf8_lossy(&bytes).into_owned()));
    (status, v)
}

/// The config as the server loads it, with the environment a test needs.
fn load(name: &str, path: &Path, state_dir: &Path) -> Config {
    let text = std::fs::read_to_string(path).unwrap();
    let mut c = Config::from_toml(&text, name).unwrap_or_else(|e| panic!("{name}: {e}"));
    let mut env = BTreeMap::new();
    env.insert("FV_API_KEYS".to_owned(), KeyRing::hash_hex(KEY));
    env.insert("FV_URL_SIGNING_KEY".to_owned(), "golden-key".to_owned());
    env.insert("FV_STATE_DIR".to_owned(), state_dir.display().to_string());
    env.insert("FV_PUBLIC_BASE_URL".to_owned(), "http://fv.test".to_owned());
    env.insert("FV_WEIGHTS".to_owned(), "/workspace/weights".to_owned());
    c.apply_env(&env).unwrap();
    // The capabilities are rendered on the fake engine: local stores.
    c.jobs.backend = fastvideo_serve::config::JobBackend::Memory;
    c.artifacts.backend = fastvideo_serve::config::ArtifactBackend::Local;
    c.validate().unwrap_or_else(|e| panic!("{name}: {e}"));
    c
}

/// The engine of a CUDA config: the fake backend serving the config's
/// models (real caps, real recipes, the pipelines' job checks), as an 80 GB
/// class GPU would (the H3 1080P tier on).
fn cuda_engine(c: &Config, state_dir: &Path) -> EngineService {
    let models = fastvideo_serve::app::cuda_models(c).unwrap();
    let fc = FakeConfig {
        timing: FakeTiming {
            load: Duration::ZERO,
            ..FakeTiming::default()
        },
        mp4: Mp4Mode::Off,
        ..FakeConfig::cuda_catalog(&models)
    };
    let cfg = EngineConfig {
        queue_max: c.limits.queue_max,
        swap: c.engine.swap,
        tier_overrides: c
            .engine
            .tier_overrides
            .iter()
            .map(|(k, v)| (k.clone(), ModelId::new(v)))
            .collect(),
        output_dir: state_dir.join("engine-out"),
        job_deadline: c.engine.job_deadline(),
        ..EngineConfig::default()
    };
    EngineService::start(cfg, vec![Box::new(FakeBackend::new(fc))]).unwrap()
}

/// What this config serves: the capabilities body and, where the OpenAI
/// videos API is mounted, `/v1/models` without its `created` stamps.
async fn render(name: &str, path: &Path) -> Value {
    let dir = tempfile::Builder::new()
        .prefix(&format!("fv-golden-{name}-"))
        .tempdir()
        .unwrap();
    let c = load(name, path, dir.path());
    let openai = c.protocols.openai_videos;
    let engine = (c.engine.backend == EngineBackendKind::Cuda).then(|| cuda_engine(&c, dir.path()));
    let app = App::build(
        c,
        Overrides {
            engine,
            ..Default::default()
        },
    )
    .await
    .unwrap_or_else(|e| panic!("{name}: {e}"));
    app.gate.engine().wait_ready().await;
    let (s, caps) = get(&app.router, "/fv/v1/capabilities").await;
    assert_eq!(s, 200, "{name}: {caps}");
    let models = if openai {
        let (s, mut v) = get(&app.router, "/v1/models").await;
        assert_eq!(s, 200, "{name}: {v}");
        if let Some(list) = v.get_mut("data").and_then(Value::as_array_mut) {
            for m in list {
                if let Some(o) = m.as_object_mut() {
                    o.remove("created");
                }
            }
        }
        v
    } else {
        Value::Null
    };
    drop(app);
    json!({ "capabilities": caps, "models": models })
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn capabilities_are_golden_for_every_shipped_config() {
    let write = std::env::var_os("FV_GOLDEN_WRITE").map(PathBuf::from);
    let mut got = serde_json::Map::new();
    for (name, path) in configs() {
        got.insert(name.clone(), render(&name, &path).await);
    }
    let got = Value::Object(got);
    let text = format!("{}\n", serde_json::to_string_pretty(&got).unwrap());
    if let Some(out) = write {
        if let Some(d) = out.parent() {
            std::fs::create_dir_all(d).unwrap();
        }
        std::fs::write(&out, &text).unwrap();
        eprintln!("wrote {}", out.display());
        return;
    }
    let golden = golden_path();
    let want_text = std::fs::read_to_string(&golden).unwrap_or_else(|e| {
        panic!(
            "{}: {e} (FV_GOLDEN_WRITE=<file> writes it)",
            golden.display()
        )
    });
    if want_text == text {
        return;
    }
    let want: Value = serde_json::from_str(&want_text).unwrap();
    let mut diffs = Vec::new();
    let (w, g) = (want.as_object().unwrap(), got.as_object().unwrap());
    for k in w.keys().chain(g.keys().filter(|k| !w.contains_key(*k))) {
        match (w.get(k), g.get(k)) {
            (Some(x), Some(y)) if x != y => diffs.push(format!("{k}: {}", first_diff(x, y, "$"))),
            (Some(_), None) => diffs.push(format!("{k}: a golden without a config")),
            (None, Some(_)) => diffs.push(format!("{k}: a config without a golden")),
            _ => {}
        }
    }
    panic!(
        "capabilities drifted from {} (FV_GOLDEN_WRITE=<file> regenerates; list the change in the PR):\n{}",
        golden.display(),
        diffs.join("\n")
    );
}

/// The first path where two JSON values differ, with both sides.
fn first_diff(want: &Value, got: &Value, path: &str) -> String {
    match (want, got) {
        (Value::Object(a), Value::Object(b)) => {
            for k in a.keys().chain(b.keys().filter(|k| !a.contains_key(*k))) {
                match (a.get(k), b.get(k)) {
                    (Some(x), Some(y)) if x != y => {
                        return first_diff(x, y, &format!("{path}.{k}"));
                    }
                    (Some(x), None) => return format!("{path}.{k}: golden {x}, now absent"),
                    (None, Some(y)) => return format!("{path}.{k}: golden absent, now {y}"),
                    _ => {}
                }
            }
            format!("{path}: objects differ")
        }
        (Value::Array(a), Value::Array(b)) => {
            for (i, (x, y)) in a.iter().zip(b).enumerate() {
                if x != y {
                    return first_diff(x, y, &format!("{path}[{i}]"));
                }
            }
            format!("{path}: lengths {} vs {}", a.len(), b.len())
        }
        _ => format!("{path}: golden {want}, now {got}"),
    }
}
