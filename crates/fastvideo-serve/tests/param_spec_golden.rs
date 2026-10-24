//! Golden `GET /fv/v1/params` (plan E2, ADR-0012) for every shipped serve
//! config (`configs/serve/*.toml`, `tests/compat/serve.toml`), rendered on
//! the fake engine serving the config's models with their real caps, as
//! `capabilities_golden` does: the ParamSpec of every mounted endpoint for
//! every served model, grouped by surface.
//!
//! The golden is `tests/goldens/param_specs.json`. A difference fails the
//! test and names the config, the surface and the first differing path; a
//! deliberate change is regenerated with `FV_GOLDEN_WRITE=<file> cargo test
//! -p fastvideo-serve --test param_spec_golden` and listed in the PR.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use axum::body::Body;
use axum::http::Request;
use fastvideo_serve::config::{Config, EngineBackendKind};
use fastvideo_serve::{App, Overrides};
use fastvideo_serve_kit::KeyRing;
use serde_json::{Map, Value};
use tower::ServiceExt;

const KEY: &str = "sk-golden";

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn golden_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/goldens/param_specs.json")
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

/// The config's specs, grouped by surface (in the order served).
async fn render(name: &str, path: &Path) -> Value {
    let dir = tempfile::Builder::new()
        .prefix(&format!("fv-params-{name}-"))
        .tempdir()
        .unwrap();
    let mut c = load(name, path, dir.path());
    if c.engine.backend == EngineBackendKind::Cuda {
        c.engine.backend = EngineBackendKind::Fake;
        c.engine.fake = Default::default();
        c.engine.fake.mp4 = false;
    }
    let app = App::build(c, Overrides::default())
        .await
        .unwrap_or_else(|e| panic!("{name}: {e}"));
    app.gate.engine().wait_ready().await;
    let (s, body) = get(&app.router, "/fv/v1/params").await;
    assert_eq!(s, 200, "{name}: {body}");
    assert_eq!(body["object"], "fv.param_specs", "{name}");
    drop(app);
    let mut by_surface = Map::new();
    for spec in body["specs"].as_array().unwrap() {
        let surface = spec["surface"].as_str().unwrap().to_owned();
        let list = by_surface
            .entry(surface)
            .or_insert_with(|| Value::Array(Vec::new()));
        list.as_array_mut().unwrap().push(spec.clone());
    }
    Value::Object(by_surface)
}

/// Every spec is well formed: unique parameter names, a domain that holds
/// its own valid example and none of its invalid ones, and a default (when
/// the parameter has one and the model serves it) inside the domain.
fn check_well_formed(name: &str, specs: &Value) {
    use fastvideo_protocol::ParamSpec;
    for list in specs.as_object().unwrap().values() {
        for s in list.as_array().unwrap() {
            let spec: ParamSpec =
                serde_json::from_value(s.clone()).unwrap_or_else(|e| panic!("{name}: {e}: {s}"));
            let mut seen = std::collections::BTreeSet::new();
            for p in &spec.params {
                let at = format!("{name}: {} {} `{}`", spec.endpoint, spec.model, p.name);
                assert!(seen.insert(p.name.clone()), "{at}: duplicate");
                if let Some(v) = p.domain.valid_example() {
                    assert!(p.domain.contains(&v), "{at}: {v}");
                }
                for v in p.domain.invalid_examples() {
                    assert!(!p.domain.contains(&v), "{at}: {v}");
                }
            }
        }
    }
}

/// The golden layout: `configs` maps each config to its surfaces' spec
/// keys; `specs` holds each distinct spec once, under `surface endpoint
/// model @ engine_model` (`#n` when two configs serve different specs
/// under one key), one parameter per line.
fn layout(got: &Map<String, Value>) -> (Value, BTreeMap<String, Value>) {
    let mut table: BTreeMap<String, Value> = BTreeMap::new();
    let mut configs = Map::new();
    for (name, surfaces) in got {
        let mut by = Map::new();
        for (surface, list) in surfaces.as_object().unwrap() {
            let mut keys = Vec::new();
            for spec in list.as_array().unwrap() {
                let s = |k: &str| spec[k].as_str().unwrap_or_default().to_owned();
                let base = format!(
                    "{} {} {} @ {}",
                    s("surface"),
                    s("endpoint"),
                    s("model"),
                    s("engine_model")
                );
                let mut key = base.clone();
                let mut n = 1;
                while table.get(&key).is_some_and(|v| v != spec) {
                    n += 1;
                    key = format!("{base} #{n}");
                }
                table.insert(key.clone(), spec.clone());
                keys.push(Value::String(key));
            }
            by.insert(surface.clone(), Value::Array(keys));
        }
        configs.insert(name.clone(), Value::Object(by));
    }
    (Value::Object(configs), table)
}

fn render_text(configs: &Value, table: &BTreeMap<String, Value>) -> String {
    let mut out = String::from("{\n\"configs\": ");
    out += &serde_json::to_string_pretty(configs).unwrap();
    out += ",\n\"specs\": {\n";
    let n = table.len();
    for (i, (key, spec)) in table.iter().enumerate() {
        let mut head = spec.as_object().unwrap().clone();
        let params = head.remove("params").unwrap_or(Value::Array(Vec::new()));
        out += &format!("{}: ", serde_json::to_string(key).unwrap());
        let head = serde_json::to_string(&Value::Object(head)).unwrap();
        // `{...,"params":[` then one parameter per line.
        out += &head[..head.len() - 1];
        out += ",\"params\":[\n";
        let ps = params.as_array().unwrap();
        for (j, p) in ps.iter().enumerate() {
            out += &serde_json::to_string(p).unwrap();
            out += if j + 1 < ps.len() { ",\n" } else { "\n" };
        }
        out += if i + 1 < n { "]},\n" } else { "]}\n" };
    }
    out += "}\n}\n";
    out
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn param_specs_are_golden_for_every_shipped_config() {
    let write = std::env::var_os("FV_GOLDEN_WRITE").map(PathBuf::from);
    let mut got = Map::new();
    for (name, path) in configs() {
        let specs = render(&name, &path).await;
        check_well_formed(&name, &specs);
        got.insert(name.clone(), specs);
    }
    let (configs, table) = layout(&got);
    let text = render_text(&configs, &table);
    let parsed: Value = serde_json::from_str(&text).expect("the golden layout is JSON");
    assert_eq!(parsed["configs"], configs);
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
    for section in ["configs", "specs"] {
        let (w, g) = (
            want[section].as_object().unwrap(),
            parsed[section].as_object().unwrap(),
        );
        for k in w.keys().chain(g.keys().filter(|k| !w.contains_key(*k))) {
            match (w.get(k), g.get(k)) {
                (Some(x), Some(y)) if x != y => {
                    diffs.push(format!("{section}.{k}: {}", first_diff(x, y, "$")))
                }
                (Some(_), None) => diffs.push(format!("{section}.{k}: in the golden only")),
                (None, Some(_)) => diffs.push(format!("{section}.{k}: new")),
                _ => {}
            }
        }
    }
    panic!(
        "param specs drifted from {} (FV_GOLDEN_WRITE=<file> regenerates; list the change in the PR):\n{}",
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
