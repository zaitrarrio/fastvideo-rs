//! Versions and releases through the gateway (docs/serve/releases.md): a
//! gateway in front of one pod pool with a real fake-engine worker and a
//! stub worker that reports another build, the D1 mock, and a mock GitHub
//! API for the workflow dispatch.
//!
//! Covered: the worker's `build` in its internal status; the public
//! `/fv/v1/status` per-pool version summary (short sha and channel only)
//! and the mixed-version flag; the full builds on the admin pools route;
//! `/fv/v1/admin/releases`, `/fv/v1/admin/deployments` (drift) and
//! promote / rollback (admin token, dry runs, dispatch, rollback choice).

#![cfg(all(feature = "http-client", feature = "openai-videos", feature = "minimax", feature = "fal", feature = "ltxapi"))]

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use axum::routing::{get, post};
use axum::{Json, Router};
use fastvideo_serve::build_info::BuildInfo;
use fastvideo_serve::config::{Config, EngineBackendKind, JobBackend, KeyStoreBackend, PoolCfg, PoolKind, Role};
use fastvideo_serve::{App, Overrides};
use fastvideo_serve_kit::d1::mock::MockD1;
use fastvideo_serve_kit::{D1Client, KeyRing};
use serde_json::{json, Value};

const KEY: &str = "sk-releases-user";
const TOKEN: &str = "rel-internal-token";
const ADMIN: &str = "fvadm_releases_test";
const GH_TOKEN: &str = "ghp_release_test_token";
const OTHER_SHA: &str = "0123456789abcdef0123456789abcdef01234567";
const OTHER_DIGEST: &str = "sha256:1111111111111111111111111111111111111111111111111111111111111111";

fn tmp(tag: &str) -> PathBuf {
    std::env::temp_dir().join(format!(
        "fv-rel-{tag}-{:x}",
        std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()
    ))
}

fn base_config(tag: &str, arts: &std::path::Path) -> Config {
    let mut env = BTreeMap::new();
    env.insert("FV_API_KEYS".to_owned(), KeyRing::hash_hex(KEY));
    env.insert("FV_URL_SIGNING_KEY".to_owned(), "shared-signing-key".to_owned());
    env.insert("FV_STATE_DIR".to_owned(), tmp(tag).display().to_string());
    env.insert("FV_INTERNAL_TOKEN".to_owned(), TOKEN.to_owned());
    env.insert("FV_ARTIFACTS_DIR".to_owned(), arts.display().to_string());
    env.insert("FV_ADMIN_TOKEN".to_owned(), ADMIN.to_owned());
    env.insert("FV_CF_ACCOUNT_ID".to_owned(), "acct".to_owned());
    env.insert("FV_CF_API_TOKEN".to_owned(), "tok".to_owned());
    env.insert("FV_D1_DATABASE_ID".to_owned(), "db".to_owned());
    let mut c = Config::default();
    c.apply_env(&env).unwrap();
    c.jobs.backend = JobBackend::D1;
    c.jobs.heartbeat_s = 1;
    c.auth.key_store = KeyStoreBackend::Memory;
    c.engine.fake.step_ms = 5;
    c.protocols.fastwan = false;
    c
}

async fn listen() -> (tokio::net::TcpListener, String) {
    let l = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let base = format!("http://{}", l.local_addr().unwrap());
    (l, base)
}

async fn serve(c: Config, mock: &MockD1, l: tokio::net::TcpListener) -> App {
    let d1 = D1Client::new(Arc::new(mock.clone())).with_retry(fastvideo_serve_kit::d1::RetryPolicy::immediate(1));
    let app = App::build(c, Overrides { d1: Some(d1), ..Default::default() }).await.unwrap();
    app.gate.engine().wait_ready().await;
    let router = app.router.clone();
    tokio::spawn(async move {
        let _ = axum::serve(l, router).await;
    });
    app
}

/// A worker that answers only its status probe, reporting another build.
async fn stub_worker() -> String {
    let (l, base) = listen().await;
    let r = Router::new().route(
        "/fv/v1/internal/status",
        get(|| async {
            Json(json!({
                "object": "fv.worker", "worker_id": "stub-1", "pool": "h3", "readiness": "ready", "draining": false,
                "stats": {"queued_batch": 0, "queued_stream": 0, "running": 0, "sessions": 0}, "models": [],
                "version": "0.1.0",
                "build": {"version": "0.1.0", "git_sha": OTHER_SHA, "build_time": "2026-09-28T10:00:00Z", "variant": "h3-turbo",
                          "channel": "stable", "image": {"ref": format!("ghcr.io/x/serve@{OTHER_DIGEST}"), "tag": "h3-turbo-sha-0123456", "digest": OTHER_DIGEST}}
            }))
        }),
    );
    tokio::spawn(async move {
        let _ = axum::serve(l, r).await;
    });
    base
}

/// The GitHub API: records workflow dispatches (with the token they carried).
async fn github() -> (String, Arc<Mutex<Vec<(String, Value)>>>) {
    let seen: Arc<Mutex<Vec<(String, Value)>>> = Arc::default();
    let s2 = seen.clone();
    let (l, base) = listen().await;
    let r = Router::new().route(
        "/repos/{owner}/{repo}/actions/workflows/{wf}/dispatches",
        post(move |headers: axum::http::HeaderMap, Json(b): Json<Value>| {
            let s = s2.clone();
            async move {
                let auth = headers.get("authorization").and_then(|v| v.to_str().ok()).unwrap_or("").to_owned();
                s.lock().unwrap().push((auth, b));
                axum::http::StatusCode::NO_CONTENT
            }
        }),
    );
    tokio::spawn(async move {
        let _ = axum::serve(l, r).await;
    });
    (base, seen)
}

async fn call(method: &str, url: &str, body: Option<Value>, admin: bool) -> (u16, Value) {
    let c = reqwest::Client::builder().no_proxy().build().unwrap();
    let mut r = c.request(reqwest::Method::from_bytes(method.as_bytes()).unwrap(), url);
    if admin {
        r = r.header("authorization", format!("Bearer {ADMIN}"));
    }
    if let Some(b) = body {
        r = r.json(&b);
    }
    let resp = r.send().await.unwrap();
    let s = resp.status().as_u16();
    let bytes = resp.bytes().await.unwrap();
    (s, serde_json::from_slice(&bytes).unwrap_or(Value::String(String::from_utf8_lossy(&bytes).into_owned())))
}

async fn poll(url: &str, done: impl Fn(&Value) -> bool) -> Value {
    for _ in 0..400 {
        let (s, v) = call("GET", url, None, true).await;
        if s == 200 && done(&v) {
            return v;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    panic!("{url} never got there");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 6)]
async fn versions_releases_and_promotion_through_the_gateway() {
    let (gh, dispatches) = github().await;
    // The release routes read their settings from the process env (one
    // test in this binary, set before the gateway is built).
    std::env::set_var("FV_GITHUB_API", &gh);
    std::env::set_var("FV_GITHUB_TOKEN", GH_TOKEN);
    std::env::set_var("FV_GITHUB_REPO", "owner/repo");

    let mock = MockD1::new();
    let arts = tmp("arts");
    std::fs::create_dir_all(&arts).unwrap();
    // A real worker (this build) and a stub (another build).
    let (wl, wbase) = listen().await;
    let mut wc = base_config("w1", &arts);
    wc.server.role = Role::Worker;
    wc.server.public_base_url = Some(wbase.clone());
    wc.server.worker_id = Some("worker-real".into());
    wc.engine.fake.models = vec!["fake-h3-turbo".into()];
    wc.validate().unwrap();
    let _worker = serve(wc, &mock, wl).await;
    let stub = stub_worker().await;

    // The worker's internal status carries its build.
    let c = reqwest::Client::builder().no_proxy().build().unwrap();
    let st: Value = c.get(format!("{wbase}/fv/v1/internal/status")).header("x-fv-internal-token", TOKEN).send().await.unwrap().json().await.unwrap();
    assert_eq!(st["build"], BuildInfo::current().json(), "{st}");

    let (gl, g) = listen().await;
    let mut gc = base_config("gw", &arts);
    gc.engine.backend = EngineBackendKind::Remote;
    gc.server.public_base_url = Some(g.clone());
    gc.pools = vec![PoolCfg {
        id: "h3".into(),
        kind: PoolKind::Pod,
        urls: vec![wbase.clone(), stub.clone()],
        fake_models: vec!["fake-h3-turbo".into()],
        stale_after_s: 3,
        ..PoolCfg::default()
    }];
    gc.gateway.tick_s = 1;
    gc.validate().unwrap();
    let _gw = serve(gc, &mock, gl).await;

    // Public status: short sha and channel per pool, the mix flagged.
    let own = BuildInfo::current().git_sha_short.clone();
    let st = poll(&format!("{g}/fv/v1/status"), |v| v["pools"][0]["versions"].as_array().is_some_and(|a| a.iter().map(|x| x["workers"].as_u64().unwrap()).sum::<u64>() == 2)).await;
    let pool = &st["pools"][0];
    assert_eq!(pool["mixed_versions"], true, "{st}");
    assert_eq!(st["mixed_versions"], true);
    let shas: Vec<&str> = pool["versions"].as_array().unwrap().iter().map(|v| v["sha"].as_str().unwrap()).collect();
    assert!(shas.contains(&"0123456") && shas.contains(&own.as_str()), "{st}");
    assert_eq!(st["version"]["sha"], own.as_str());
    for secret in [OTHER_DIGEST, OTHER_SHA, &stub, &wbase, "h3-turbo-sha-0123456", "stub-1"] {
        assert!(!st.to_string().contains(secret), "public status leaks {secret}: {st}");
    }
    for v in pool["versions"].as_array().unwrap() {
        let keys: Vec<&String> = v.as_object().unwrap().keys().collect();
        assert_eq!(keys, ["channel", "sha", "workers"], "{v}");
    }

    // The admin pools route: every worker's full build.
    let (s, pools) = call("GET", &format!("{g}/fv/v1/gateway/pools"), None, true).await;
    assert_eq!(s, 200, "{pools}");
    let workers = pools["state"][0]["workers"].as_array().unwrap();
    let stub_w = workers.iter().find(|w| w["url"] == stub.as_str()).unwrap();
    assert_eq!(stub_w["build"]["image_digest"], OTHER_DIGEST, "{pools}");
    assert_eq!(stub_w["build"]["git_sha"], OTHER_SHA);
    assert_eq!(pools["state"][0]["mixed_versions"], true);
    assert_eq!(pools["gateway_build"], BuildInfo::current().json());

    // Releases: admin token only; empty history at first (tables created).
    for (m, path) in [("GET", "/fv/v1/admin/releases"), ("GET", "/fv/v1/admin/deployments"), ("POST", "/fv/v1/admin/releases/promote"), ("POST", "/fv/v1/admin/releases/rollback")] {
        let (s, _) = call(m, &format!("{g}{path}"), Some(json!({})), false).await;
        assert_eq!(s, 401, "{m} {path}");
    }
    let (s, r) = call("GET", &format!("{g}/fv/v1/admin/releases"), None, true).await;
    assert_eq!(s, 200, "{r}");
    assert_eq!((r["heads"].clone(), r["history"].clone(), r["template_channel"].clone()), (json!([]), json!([]), json!("stable")));
    assert_eq!(r["dispatch"]["configured"], true);
    assert!(!r.to_string().contains(GH_TOKEN));

    // History: A, B (stable, B the stub's digest), latest C.
    let digests = |d: &str| json!({"h3-turbo": format!("ghcr.io/x/serve@{d}"), "debug": "ghcr.io/x/serve@sha256:dd"}).to_string();
    let ins = "INSERT INTO releases (channel, git_sha, digests, action, promoted_at, promoted_by, templates_updated) VALUES (?, ?, ?, ?, ?, 'test', 1)";
    mock.sql(ins, &[json!("stable"), json!("a".repeat(40)), json!(digests("sha256:aa")), json!("promote"), json!(1)]).unwrap();
    mock.sql(ins, &[json!("stable"), json!(OTHER_SHA), json!(digests(OTHER_DIGEST)), json!("promote"), json!(2)]).unwrap();
    mock.sql(ins, &[json!("latest"), json!("c".repeat(40)), json!(digests("sha256:cc")), json!("build"), json!(3)]).unwrap();
    mock.sql(
        "INSERT INTO deployments (id, kind, runpod_id, variant, digest, git_sha, channel, created_at, updated_at, created_by, status) VALUES \
         ('pod:p1', 'pod', 'p1', 'h3-turbo', ?, ?, 'stable', 1, 1, 'test', 'ready'), \
         ('pod:p2', 'pod', 'p2', 'h3-turbo', 'sha256:aa', ?, 'stable', 1, 1, 'test', 'ready'), \
         ('pod:p3', 'pod', 'p3', 'h3-turbo', 'sha256:aa', ?, 'stable', 1, 5, 'test', 'deleted')",
        &[json!(OTHER_DIGEST), json!(OTHER_SHA), json!("a".repeat(40)), json!("a".repeat(40))],
    )
    .unwrap();
    mock.sql("UPDATE deployments SET deleted_at = 5 WHERE id = 'pod:p3'", &[]).unwrap();

    let (_, r) = call("GET", &format!("{g}/fv/v1/admin/releases?channel=stable"), None, true).await;
    assert_eq!(r["heads"].as_array().unwrap().len(), 2, "{r}");
    assert_eq!(r["history"].as_array().unwrap().iter().map(|h| h["id"].as_i64().unwrap()).collect::<Vec<_>>(), [2, 1]);
    assert!(r["history"][0]["digests"]["h3-turbo"].is_string(), "digests are parsed: {r}");

    let (s, d) = call("GET", &format!("{g}/fv/v1/admin/deployments"), None, true).await;
    assert_eq!(s, 200, "{d}");
    let deps = d["deployments"].as_array().unwrap();
    assert_eq!(deps.len(), 2, "deleted rows are not live: {d}");
    let drift_of = |id: &str| deps.iter().find(|x| x["runpod_id"] == id).unwrap()["drift"].clone();
    assert_eq!(drift_of("p1"), json!(false));
    assert_eq!(drift_of("p2"), json!("0123456"), "p2 is behind stable");
    let stub_live = d["pools"][0]["workers"].as_array().unwrap().iter().find(|w| w["url"] == stub.as_str()).unwrap().clone();
    assert_eq!(stub_live["build"]["drift"], json!(false), "{stub_live}");
    assert_eq!(d["pools"][0]["mixed_versions"], true);
    assert_eq!(d["gateway"]["sha"], own.as_str());

    // Promote: validation, dry run (no dispatch), then the dispatch.
    let promote = format!("{g}/fv/v1/admin/releases/promote");
    for bad in [json!({}), json!({"target": "x;rm -rf"}), json!({"target": "2cd1ba0", "channel": "wan"}), json!({"target": "2cd1ba0", "channel": "Stable"})] {
        let (s, v) = call("POST", &promote, Some(bad.clone()), true).await;
        assert_eq!(s, 400, "{bad} -> {v}");
    }
    let (s, plan) = call("POST", &promote, Some(json!({"target": "2cd1ba0", "channel": "stable", "notes": "n", "dry_run": true})), true).await;
    assert_eq!(s, 200, "{plan}");
    assert_eq!(plan["dry_run"], true);
    assert_eq!(plan["inputs"]["action"], "promote");
    assert_eq!(plan["current"]["id"], 2);
    assert_eq!(plan["templates"], true);
    assert!(dispatches.lock().unwrap().is_empty(), "a dry run dispatches nothing");
    let (s, done) = call("POST", &promote, Some(json!({"target": "2cd1ba0", "channel": "latest"})), true).await;
    assert_eq!(s, 202, "{done}");
    assert_eq!(done["templates"], false);
    {
        let seen = dispatches.lock().unwrap();
        assert_eq!(seen.len(), 1);
        assert_eq!(seen[0].0, format!("Bearer {GH_TOKEN}"));
        assert_eq!(seen[0].1["ref"], "main");
        assert_eq!(seen[0].1["inputs"]["target"], "2cd1ba0");
        assert_eq!(seen[0].1["inputs"]["channel"], "latest");
    }
    assert!(!done.to_string().contains(GH_TOKEN));

    // Rollback: the previous stable release (A = #1), dispatched as `to`.
    let rollback = format!("{g}/fv/v1/admin/releases/rollback");
    let (s, plan) = call("POST", &rollback, Some(json!({"channel": "stable", "dry_run": true})), true).await;
    assert_eq!(s, 200, "{plan}");
    assert_eq!((plan["current"]["id"].clone(), plan["target"]["id"].clone(), plan["inputs"]["to"].clone()), (json!(2), json!(1), json!("1")));
    let (s, _) = call("POST", &rollback, Some(json!({"channel": "stable"})), true).await;
    assert_eq!(s, 202);
    assert_eq!(dispatches.lock().unwrap()[1].1["inputs"]["action"], "rollback");
    let (s, v) = call("POST", &rollback, Some(json!({"channel": "latest", "dry_run": true})), true).await;
    assert_eq!(s, 409, "latest has one release: {v}");
    let (s, _) = call("POST", &rollback, Some(json!({"channel": "canary", "dry_run": true})), true).await;
    assert_eq!(s, 409);
    let (s, v) = call("POST", &rollback, Some(json!({"channel": "stable", "to": 3, "dry_run": true})), true).await;
    assert_eq!(s, 409, "release 3 is not stable's: {v}");

    // Without a token the gateway refuses to dispatch; dry runs still work.
    std::env::remove_var("FV_GITHUB_TOKEN");
    let (gl2, g2) = listen().await;
    let mut gc2 = base_config("gw2", &arts);
    gc2.engine.backend = EngineBackendKind::Remote;
    gc2.server.public_base_url = Some(g2.clone());
    gc2.pools = vec![PoolCfg { id: "h3".into(), kind: PoolKind::Pod, urls: vec![stub.clone()], fake_models: vec!["fake-h3-turbo".into()], ..PoolCfg::default() }];
    gc2.validate().unwrap();
    let _gw2 = serve(gc2, &mock, gl2).await;
    let body = json!({"target": "2cd1ba0", "channel": "stable"});
    let (s, v) = call("POST", &format!("{g2}/fv/v1/admin/releases/promote"), Some(body.clone()), true).await;
    assert_eq!(s, 503, "{v}");
    assert_eq!(v["error"]["kind"], "not_configured");
    let mut dry = body;
    dry["dry_run"] = json!(true);
    let (s, v) = call("POST", &format!("{g2}/fv/v1/admin/releases/promote"), Some(dry), true).await;
    assert_eq!((s, v["dispatch"]["configured"].clone()), (200, json!(false)), "{v}");
    assert_eq!(dispatches.lock().unwrap().len(), 2);
}
