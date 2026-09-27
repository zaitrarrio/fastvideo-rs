//! End-to-end: the assembled fv-serve router over the real `EngineService`
//! with the fake backend (design §7.3), through serve-kit's generic handlers,
//! the engine gate, the job stores (file, and D1 over the SQLite mock of the
//! D1 HTTP API), artifacts, health, metrics and drain.
//!
//! The batch path here is the native `/fv/v1/jobs` API, which uses the same
//! serve-kit `submit`/`status`/`result` handlers the adapters use; the LTX
//! adapter (the first on main) has its own suite in `tests/ltx.rs`.

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Duration;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use axum::Router;
use fastvideo_protocol::JobStore;
use fastvideo_serve::config::{Config, JobBackend};
use fastvideo_serve::router::{route_table, Owner};
use fastvideo_serve::{App, Overrides};
use fastvideo_serve_kit::d1::mock::MockD1;
use fastvideo_serve_kit::{D1Client, D1JobStore, D1Options, KeyRing};
use serde_json::{json, Value};
use tower::ServiceExt;

const KEY: &str = "sk-test-fv";

fn config(tag: &str) -> Config {
    let dir = std::env::temp_dir().join(format!("fv-serve-e2e-{tag}-{}", uuid_like()));
    let mut env = BTreeMap::new();
    env.insert("FV_API_KEYS".to_owned(), KeyRing::hash_hex(KEY));
    env.insert("FV_URL_SIGNING_KEY".to_owned(), "test-signing-key".to_owned());
    env.insert("FV_STATE_DIR".to_owned(), dir.display().to_string());
    env.insert("FV_PUBLIC_BASE_URL".to_owned(), "http://fv.test".to_owned());
    let mut c = Config::default();
    c.apply_env(&env).unwrap();
    c.jobs.backend = JobBackend::File;
    c.engine.fake.step_ms = 1;
    c.validate().unwrap();
    c
}

fn uuid_like() -> String {
    format!("{:x}", std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos())
}

async fn call(app: &Router, method: &str, uri: &str, body: Option<Value>, auth: bool) -> (StatusCode, Value, axum::http::HeaderMap) {
    let mut b = Request::builder().method(method).uri(uri);
    if auth {
        b = b.header("authorization", format!("Bearer {KEY}"));
    }
    let req = match body {
        Some(v) => b.header("content-type", "application/json").body(Body::from(v.to_string())).unwrap(),
        None => b.body(Body::empty()).unwrap(),
    };
    let resp = app.clone().oneshot(req).await.unwrap();
    let status = resp.status();
    let headers = resp.headers().clone();
    let bytes = axum::body::to_bytes(resp.into_body(), 1 << 24).await.unwrap();
    let v = serde_json::from_slice(&bytes).unwrap_or_else(|_| Value::String(String::from_utf8_lossy(&bytes).into_owned()));
    (status, v, headers)
}

async fn wait_ready(app: &App) {
    app.gate.engine().wait_ready().await;
}

async fn wait_status(app: &Router, id: &str, want: &[&str]) -> Value {
    for _ in 0..4000 {
        let (s, v, _) = call(app, "GET", &format!("/fv/v1/jobs/{id}"), None, true).await;
        assert_eq!(s, 200, "{v}");
        if want.contains(&v["status"].as_str().unwrap_or_default()) {
            return v;
        }
        tokio::time::sleep(Duration::from_millis(25)).await;
    }
    panic!("job {id} never reached {want:?}");
}

fn submit_body(prompt: &str) -> Value {
    json!({"model": "h3-turbo", "prompt": prompt, "aspect_ratio": "16:9", "short_edge": 768})
}

#[tokio::test]
async fn batch_job_end_to_end_with_file_store() {
    let app = App::build(config("file"), Overrides::default()).await.unwrap();
    wait_ready(&app).await;
    let r = app.router.clone();

    // Health family.
    let (s, v, _) = call(&r, "GET", "/ping", None, false).await;
    assert_eq!((s, v), (StatusCode::OK, json!({"status": "healthy"})));
    let (s, v, _) = call(&r, "GET", "/health", None, false).await;
    assert_eq!(s, 200);
    assert_eq!(v, json!({"status": "ok", "model_loaded": true, "state": "AVAILABLE"}));
    let (s, v, _) = call(&r, "GET", "/healthz", None, false).await;
    assert_eq!(s, 200);
    assert_eq!(v["state"], "ready");
    assert_eq!(v["stores"], json!({"jobs": "file", "artifacts": "local"}));
    let (_, v, _) = call(&r, "GET", "/", None, false).await;
    assert!(v["model"].is_string(), "{v}");

    // Auth and validation errors.
    let (s, _, _) = call(&r, "POST", "/fv/v1/jobs", Some(submit_body("a cat")), false).await;
    assert_eq!(s, 401);
    let (s, v, _) = call(&r, "POST", "/fv/v1/jobs", Some(json!({"model": "nope", "prompt": "x"})), true).await;
    assert_eq!(s, 400, "{v}");
    assert_eq!(v["error"]["param"], "model");
    let (s, _, _) = call(&r, "POST", "/fv/v1/jobs", Some(json!({"model": "h3-turbo", "prompt": "x", "bogus": 1})), true).await;
    assert_eq!(s, 400, "unknown fields are refused");

    // Capabilities show the tier aliases.
    let (s, v, _) = call(&r, "GET", "/fv/v1/capabilities", None, true).await;
    assert_eq!(s, 200);
    assert_eq!(v["aliases"]["h3-turbo"], "fake-h3-turbo");

    // Submit -> succeeded with a downloadable artifact.
    let (s, v, _) = call(&r, "POST", "/fv/v1/jobs", Some(submit_body("a cat on a skateboard")), true).await;
    assert_eq!(s, 202, "{v}");
    let id = v["id"].as_str().unwrap().to_owned();
    assert!(id.starts_with("fvjob_"));
    assert_eq!((v["width"].as_u64(), v["height"].as_u64()), (Some(1344), Some(768)));
    let done = wait_status(&r, &id, &["succeeded", "failed"]).await;
    assert_eq!(done["status"], "succeeded", "{done}");
    assert_eq!(done["progress"], 1.0);
    assert_eq!(done["tier"], "turbo");
    let url = done["output"]["url"].as_str().unwrap();
    let path = url.strip_prefix("http://fv.test").unwrap();
    let (s, body, _) = call(&r, "GET", path, None, false).await;
    assert_eq!(s, 200);
    // The fake's MP4 through ffmpeg when present, else the placeholder.
    assert!(body.as_str().is_some_and(|b| !b.is_empty()), "{body}");
    let (s, _, h) = call(&r, "GET", &format!("/fv/v1/jobs/{id}/content"), None, true).await;
    assert_eq!(s, 302);
    assert!(h["location"].to_str().unwrap().starts_with("http://fv.test/files/"));

    // An injected engine failure fails the job, not the server.
    let (_, v, _) = call(&r, "POST", "/fv/v1/jobs", Some(submit_body("boom [fake:fail]")), true).await;
    let failed = wait_status(&r, v["id"].as_str().unwrap(), &["failed", "succeeded"]).await;
    assert_eq!(failed["status"], "failed");
    assert_eq!(failed["error"]["kind"], "engine_failed");

    // Listing (owner-scoped, newest first) and filters.
    let (s, v, _) = call(&r, "GET", "/fv/v1/jobs?limit=10", None, true).await;
    assert_eq!(s, 200);
    assert_eq!(v["total"], 2);
    assert_eq!(v["data"][1]["id"], id.as_str());
    let (_, v, _) = call(&r, "GET", "/fv/v1/jobs?status=succeeded", None, true).await;
    assert_eq!(v["data"].as_array().unwrap().len(), 1);

    // Delete a finished job.
    let (s, v, _) = call(&r, "DELETE", &format!("/fv/v1/jobs/{id}"), None, true).await;
    assert_eq!((s, v["deleted"].as_bool()), (StatusCode::OK, Some(true)));
    let (s, _, _) = call(&r, "GET", &format!("/fv/v1/jobs/{id}"), None, true).await;
    assert_eq!(s, 404);

    // Metrics saw the requests and the jobs.
    let (s, m, _) = call(&r, "GET", "/metrics", None, false).await;
    assert_eq!(s, 200);
    let m = m.as_str().unwrap().to_owned();
    assert!(m.contains("fv_http_requests_total"), "{m}");
    assert!(m.contains("route=\"/fv/v1/jobs/{id}\""), "{m}");
    assert!(m.contains("fv_jobs_finished_total"), "{m}");
    assert!(m.contains("fv_ready 1"), "{m}");
}

/// Every route the §9 table gives to serve, serve-kit and native exists on
/// the assembled router (not the router's own 404/405).
#[tokio::test]
async fn assembled_router_serves_its_route_table() {
    let app = App::build(config("routes"), Overrides::default()).await.unwrap();
    wait_ready(&app).await;
    for spec in route_table(&[]) {
        if !matches!(spec.owner, Owner::Serve | Owner::ServeKit | Owner::Native) {
            continue;
        }
        let uri = spec.path.replace("{artifact}", "a").replace("{name}", "b.mp4").replace("{token}", "t").replace("{id}", "fvjob_x");
        let body = (spec.method == "POST" || spec.method == "PUT").then(|| json!({}));
        let (s, v, _) = call(&app.router, spec.method, &uri, body, true).await;
        assert_ne!(s, StatusCode::METHOD_NOT_ALLOWED, "{} {uri}", spec.method);
        if s == StatusCode::NOT_FOUND {
            // Handler 404s (unknown job/file) carry a body or come from the
            // signed-file route; the router's own 404 is empty.
            assert!(spec.owner == Owner::ServeKit || v != Value::String(String::new()), "{} {uri} unrouted", spec.method);
        }
    }
}

#[tokio::test]
async fn ping_is_204_while_loading() {
    let mut c = config("loading");
    c.engine.fake.load_ms = 1500;
    let app = App::build(c, Overrides::default()).await.unwrap();
    let (s, _, _) = call(&app.router, "GET", "/ping", None, false).await;
    assert_eq!(s, StatusCode::NO_CONTENT);
    let (s, v, _) = call(&app.router, "GET", "/health", None, false).await;
    assert_eq!(s, 503);
    assert_eq!(v["model_loaded"], false);
    let (s, v, h) = call(&app.router, "POST", "/fv/v1/jobs", Some(submit_body("x")), true).await;
    assert_eq!(s, 503, "{v}");
    assert_eq!(h["retry-after"], "1");
    wait_ready(&app).await;
    let (s, _, _) = call(&app.router, "GET", "/ping", None, false).await;
    assert_eq!(s, 200);
}

#[tokio::test]
async fn d1_job_store_end_to_end_over_the_mock() {
    let mock = MockD1::new();
    let db = || D1Client::new(Arc::new(mock.clone()));
    let store = D1JobStore::new(db(), D1Options::new("worker-a")).open(time::OffsetDateTime::now_utc()).await.unwrap();
    let app = App::build(config("d1"), Overrides { jobs: Some(store.clone()), ..Default::default() }).await.unwrap();
    wait_ready(&app).await;
    let r = app.router.clone();
    let (s, v, _) = call(&r, "POST", "/fv/v1/jobs", Some(submit_body("a d1 job")), true).await;
    assert_eq!(s, 202, "{v}");
    let id = v["id"].as_str().unwrap().to_owned();
    wait_status(&r, &id, &["succeeded"]).await;
    let rows = mock.sql("SELECT status, protocol, worker, owner FROM jobs WHERE external_id = ?", &[json!(id)]).unwrap();
    assert_eq!(rows[0]["status"], "succeeded");
    assert_eq!(rows[0]["protocol"], "native");
    assert_eq!(rows[0]["worker"], "worker-a");
    assert!(rows[0]["owner"].as_str().unwrap().starts_with("key_"));
    // Another worker (after a restart or scale-out) answers from D1.
    let other = D1JobStore::new(db(), D1Options::new("worker-b")).open(time::OffsetDateTime::now_utc()).await.unwrap();
    let j = other.by_external(fastvideo_protocol::ProtocolId::Native, &id).await.unwrap();
    assert_eq!(j.status(), fastvideo_protocol::JobStatus::Succeeded);
    assert_eq!(j.artifacts.len(), 1);
    // The list endpoint runs in D1.
    let (_, v, _) = call(&r, "GET", "/fv/v1/jobs", None, true).await;
    assert_eq!(v["total"], 1);
    assert!(mock.statements().iter().any(|s| s.starts_with("SELECT COUNT(*)")));
}

#[tokio::test]
async fn drain_cancels_queued_and_refuses_new_work() {
    let mut c = config("drain");
    c.engine.fake.step_ms = 300;
    let app = App::build(c, Overrides::default()).await.unwrap();
    wait_ready(&app).await;
    let r = app.router.clone();
    let (_, a, _) = call(&r, "POST", "/fv/v1/jobs", Some(submit_body("long one")), true).await;
    let (_, b, _) = call(&r, "POST", "/fv/v1/jobs", Some(submit_body("queued one")), true).await;
    let (a, b) = (a["id"].as_str().unwrap().to_owned(), b["id"].as_str().unwrap().to_owned());
    wait_status(&r, &a, &["running"]).await;
    fastvideo_serve::app::drain(&app.gate, Duration::from_millis(100), None).await;
    let (s, _, _) = call(&r, "GET", "/ping", None, false).await;
    assert_eq!(s, 503, "draining");
    let (s, _, _) = call(&r, "POST", "/fv/v1/jobs", Some(submit_body("late")), true).await;
    assert_eq!(s, 503);
    let vb = wait_status(&r, &b, &["cancelled"]).await;
    assert_eq!(vb["status"], "cancelled");
    let va = wait_status(&r, &a, &["cancelled", "succeeded"]).await;
    assert_eq!(va["status"], "cancelled", "the running job was cut after the grace period");
}

#[test]
fn shipped_configs_parse() {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../configs/serve");
    let mut n = 0;
    for e in std::fs::read_dir(&dir).unwrap() {
        let p = e.unwrap().path();
        if p.extension().and_then(|x| x.to_str()) != Some("toml") {
            continue;
        }
        let text = std::fs::read_to_string(&p).unwrap();
        let c = Config::from_toml(&text, &p.display().to_string()).unwrap();
        c.validate().unwrap_or_else(|e| panic!("{}: {e}", p.display()));
        n += 1;
    }
    assert!(n >= 2, "configs/serve has {n} configs");
}
