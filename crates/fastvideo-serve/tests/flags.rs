//! Experimental feature flags (crate::flags, docs/serve/console.md §7) and
//! the startup capability check (docs/serve/h3-1080p-and-upscaler.md):
//!
//! - `h3_1080p_long`: toggled through the admin API; off (the default) an
//!   H3 1080P request longer than 5 s is a clear 4xx on every API that
//!   offers 1080P (fal, native, `/v1/videos`), naming the 5 s limit and the
//!   experimental feature; on, up to 10 s is accepted. The console's fal
//!   form schema and `/fv/v1/capabilities` follow the flag. The flag
//!   persists across a restart on the same state dir.
//! - a fake engine on a simulated A100 (sm80) fails its FP8 models with the
//!   reason in `/fv/v1/status` instead of reporting ready.

#![cfg(all(feature = "fal", feature = "openai-videos"))]

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use axum::body::Body;
use axum::http::{Request, StatusCode};
use axum::Router;
use fastvideo_serve::config::{Config, JobBackend, KeyStoreBackend};
use fastvideo_serve::{App, Overrides};
use fastvideo_serve_kit::KeyRing;
use serde_json::{json, Value};
use tower::ServiceExt;

const ADMIN: &str = "fvadm_flags-test-admin";
const ADMIN_AUTH: &str = "Bearer fvadm_flags-test-admin";
const KEY: &str = "sk-flags-test";

fn state_dir(tag: &str) -> PathBuf {
    tempfile::Builder::new().prefix(&format!("fv-serve-flags-{tag}-")).tempdir().unwrap().keep()
}

fn config(dir: &Path, extra: &[(&str, &str)]) -> Config {
    let mut env = BTreeMap::new();
    env.insert("FV_URL_SIGNING_KEY".to_owned(), "k".to_owned());
    env.insert("FV_STATE_DIR".to_owned(), dir.display().to_string());
    env.insert("FV_PUBLIC_BASE_URL".to_owned(), "http://fv.test".to_owned());
    env.insert("FV_ADMIN_TOKEN".to_owned(), ADMIN.to_owned());
    env.insert("FV_API_KEYS".to_owned(), KeyRing::hash_hex(KEY));
    for (k, v) in extra {
        env.insert((*k).to_owned(), (*v).to_owned());
    }
    let mut c = Config::default();
    c.apply_env(&env).unwrap();
    c.jobs.backend = JobBackend::Memory;
    // Jobs stay queued/running: the test only looks at admission.
    c.engine.fake.step_ms = 60_000;
    c.validate().unwrap();
    c
}

async fn call(r: &Router, method: &str, uri: &str, auth: Option<&str>, body: Option<Value>) -> (StatusCode, Value) {
    let mut b = Request::builder().method(method).uri(uri);
    if let Some(a) = auth {
        b = b.header("authorization", a);
    }
    let req = match body {
        Some(v) => b.header("content-type", "application/json").body(Body::from(v.to_string())).unwrap(),
        None => b.body(Body::empty()).unwrap(),
    };
    let resp = r.clone().oneshot(req).await.unwrap();
    let status = resp.status();
    let bytes = axum::body::to_bytes(resp.into_body(), 1 << 24).await.unwrap();
    (status, serde_json::from_slice(&bytes).unwrap_or_else(|_| Value::String(String::from_utf8_lossy(&bytes).into_owned())))
}

/// One 1080P request of `secs` seconds on each API that offers 1080P.
async fn submit_1080p(r: &Router, secs: u32) -> Vec<(&'static str, StatusCode, Value)> {
    let fal = call(
        r,
        "POST",
        "/minimax/h3-turbo/text-to-video",
        Some(&format!("Key {KEY}")),
        Some(json!({"prompt": "a fox runs", "resolution": "1080P", "duration": secs})),
    )
    .await;
    let native = call(
        r,
        "POST",
        "/fv/v1/jobs",
        Some(&format!("Bearer {KEY}")),
        Some(json!({"model": "fake-h3-turbo", "prompt": "a fox runs", "short_edge": 1080, "aspect_ratio": "16:9", "seconds": secs})),
    )
    .await;
    let openai = call(
        r,
        "POST",
        "/v1/videos",
        Some(&format!("Bearer {KEY}")),
        Some(json!({"model": "fake-h3-turbo", "prompt": "a fox runs", "size": "1920x1080", "seconds": secs.to_string()})),
    )
    .await;
    vec![("fal", fal.0, fal.1), ("native", native.0, native.1), ("/v1/videos", openai.0, openai.1)]
}

fn duration_schema(v: &Value) -> Value {
    v["properties"]["duration"].clone()
}

#[tokio::test]
async fn h3_1080p_long_flag_gates_1080p_length_on_every_api() {
    let dir = state_dir("gate");
    let app = App::build(config(&dir, &[("FV_FAKE_H3_1080P", "1")]), Overrides::default()).await.unwrap();
    app.gate.engine().wait_ready().await;
    let r = &app.router;

    // Admin API: the token is required; the flag is listed, off.
    let (st, _) = call(r, "GET", "/fv/v1/admin/flags", None, None).await;
    assert_eq!(st, StatusCode::UNAUTHORIZED);
    let (st, _) = call(r, "PUT", "/fv/v1/admin/flags/h3_1080p_long", Some("Bearer fvadm_wrong"), Some(json!({"enabled": true}))).await;
    assert_eq!(st, StatusCode::UNAUTHORIZED);
    let (st, v) = call(r, "GET", "/fv/v1/admin/flags", Some(ADMIN_AUTH), None).await;
    assert_eq!(st, 200, "{v}");
    assert_eq!(v["backend"], "file");
    let flag = v["flags"].as_array().unwrap().iter().find(|f| f["name"] == "h3_1080p_long").cloned().unwrap();
    assert_eq!((flag["enabled"].clone(), flag["default"].clone()), (json!(false), json!(false)), "{flag}");
    let (st, _) = call(r, "PUT", "/fv/v1/admin/flags/nope", Some(ADMIN_AUTH), Some(json!({"enabled": true}))).await;
    assert_eq!(st, StatusCode::NOT_FOUND);
    let (st, _) = call(r, "PUT", "/fv/v1/admin/flags/h3_1080p_long", Some(ADMIN_AUTH), Some(json!({"on": 1}))).await;
    assert_eq!(st, StatusCode::BAD_REQUEST);

    // Off: 5 s at 1080P is accepted, 10 s refused with a clear 4xx on every API.
    for (api, st, v) in submit_1080p(r, 5).await {
        assert!(st.is_success(), "{api} 5 s at 1080P: {st} {v}");
    }
    for (api, st, v) in submit_1080p(r, 10).await {
        assert!(st.is_client_error(), "{api} 10 s at 1080P must be refused: {st} {v}");
        let msg = v.to_string();
        assert!(msg.contains("5 s") && msg.contains("experimental") && msg.contains("h3_1080p_long"), "{api}: {msg}");
    }
    // 768P keeps the whole grid.
    let (st, v) = call(r, "POST", "/minimax/h3-turbo/text-to-video", Some(&format!("Key {KEY}")), Some(json!({"prompt": "x", "resolution": "768P", "duration": 10}))).await;
    assert!(st.is_success(), "{st} {v}");

    // The console's form: the 1080P duration cap.
    let (st, schema) = call(r, "GET", "/fal/schema/minimax/h3-turbo/text-to-video", None, None).await;
    assert_eq!(st, 200);
    let d = duration_schema(&schema);
    assert_eq!(d["x-fv-max-by-resolution"], json!({"1080P": 5}), "{d}");
    assert!(d["description"].as_str().unwrap().contains("experimental"), "{d}");
    let (_, caps) = call(r, "GET", "/fv/v1/capabilities", Some(&format!("Bearer {KEY}")), None).await;
    let hd = |caps: &Value| {
        caps["models"].as_array().unwrap().iter().find(|m| m["caps"]["id"] == "fake-h3-turbo").unwrap()["caps"]["canvas"]["hd"].clone()
    };
    assert_eq!(hd(&caps)["max_frames"], 124, "{caps}");

    // On: up to 10 s everywhere; 11 s still refused (the 10 s cap).
    let (st, v) = call(r, "PUT", "/fv/v1/admin/flags/h3_1080p_long", Some(ADMIN_AUTH), Some(json!({"enabled": true}))).await;
    assert_eq!(st, 200, "{v}");
    assert_eq!(v["flags"][0]["enabled"], true);
    for (api, st, v) in submit_1080p(r, 10).await {
        assert!(st.is_success(), "{api} 10 s at 1080P with the flag on: {st} {v}");
    }
    let (st, v) = call(r, "POST", "/minimax/h3-turbo/text-to-video", Some(&format!("Key {KEY}")), Some(json!({"prompt": "x", "resolution": "1080P", "duration": 11}))).await;
    assert!(st.is_client_error() && v.to_string().contains("10 s"), "{st} {v}");
    let (_, schema) = call(r, "GET", "/fal/schema/minimax/h3-turbo/text-to-video", None, None).await;
    assert_eq!(duration_schema(&schema)["x-fv-max-by-resolution"], json!({"1080P": 10}));
    let (_, caps) = call(r, "GET", "/fv/v1/capabilities", Some(&format!("Bearer {KEY}")), None).await;
    assert_eq!(hd(&caps)["max_frames"], 243);
    assert!(hd(&caps).get("experimental_max_frames").is_none());
    drop(app);

    // Durable: a restart on the same state dir keeps it on.
    let app = App::build(config(&dir, &[("FV_FAKE_H3_1080P", "1")]), Overrides::default()).await.unwrap();
    app.gate.engine().wait_ready().await;
    assert!(app.flags.enabled("h3_1080p_long"));
    for (api, st, v) in submit_1080p(&app.router, 10).await {
        assert!(st.is_success(), "{api} after a restart: {st} {v}");
    }
    // Off again: refused again.
    let (st, _) = call(&app.router, "PUT", "/fv/v1/admin/flags/h3_1080p_long", Some(ADMIN_AUTH), Some(json!({"enabled": false}))).await;
    assert_eq!(st, 200);
    for (api, st, v) in submit_1080p(&app.router, 10).await {
        assert!(st.is_client_error(), "{api} with the flag off again: {st} {v}");
    }
    drop(app);
    std::fs::remove_dir_all(&dir).ok();
}

/// Flags in D1 are shared: set on one server, another on the same database
/// picks it up (here on the reload the admin list does; the 30 s refresh in
/// production).
#[tokio::test]
async fn flags_live_in_d1_when_the_jobs_do() {
    let mock = fastvideo_serve_kit::d1::mock::MockD1::new();
    let client = || fastvideo_serve_kit::D1Client::new(std::sync::Arc::new(mock.clone()));
    let mk = |tag: &str| {
        let mut c = config(
            &state_dir(tag),
            &[("FV_FAKE_H3_1080P", "1"), ("FV_CF_ACCOUNT_ID", "acct"), ("FV_CF_API_TOKEN", "tok"), ("FV_D1_DATABASE_ID", "db")],
        );
        c.jobs.backend = JobBackend::D1;
        c.auth.key_store = KeyStoreBackend::Memory;
        c
    };
    let a = App::build(mk("d1a"), Overrides { d1: Some(client()), ..Default::default() }).await.unwrap();
    let b = App::build(mk("d1b"), Overrides { d1: Some(client()), ..Default::default() }).await.unwrap();
    assert_eq!(a.flags.backend_kind(), "d1");
    let (st, v) = call(&a.router, "PUT", "/fv/v1/admin/flags/h3_1080p_long", Some(ADMIN_AUTH), Some(json!({"enabled": true}))).await;
    assert_eq!(st, 200, "{v}");
    assert!(mock.statements().iter().any(|s| s.contains("feature_flags")));
    let (_, v) = call(&b.router, "GET", "/fv/v1/admin/flags", Some(ADMIN_AUTH), None).await;
    assert_eq!(v["flags"][0]["enabled"], true, "{v}");
    assert!(b.flags.enabled("h3_1080p_long"));
}

/// A GPU that cannot run a model: the fake engine on a simulated A100
/// (sm80, no FP8) fails the FP8 H3 models with the reason, so the server
/// is not ready and `/fv/v1/status` says why; the other models load.
#[tokio::test]
async fn an_fp8_model_on_an_a100_is_failed_not_ready() {
    let dir = state_dir("sm80");
    let app = App::build(config(&dir, &[("FV_FAKE_DEVICE", "a100")]), Overrides::default()).await.unwrap();
    let readiness = app.gate.engine().wait_ready().await;
    let msg = match &readiness {
        fastvideo_engine_service::Readiness::Failed(m) => m.clone(),
        other => panic!("expected a failed readiness, got {other:?}"),
    };
    assert!(msg.contains("sm80") && msg.contains("FP8"), "{msg}");
    let (st, v) = call(&app.router, "GET", "/fv/v1/status", None, None).await;
    assert_eq!(st, 200);
    assert_eq!(v["state"], "failed", "{v}");
    let h3 = &v["models"]["fake-h3-turbo"];
    assert_eq!(h3["state"], "failed", "{v}");
    let why = h3["reason"].as_str().unwrap();
    assert!(why.contains("A100") && why.contains("sm80") && why.contains("sm89"), "{why}");
    assert!(v["models"]["fake-wan"].get("reason").is_none(), "{v}");
    let (st, _) = call(&app.router, "GET", "/health", None, None).await;
    assert_eq!(st, StatusCode::SERVICE_UNAVAILABLE);
    drop(app);

    // The same set on an H100 (sm90) is ready.
    let app = App::build(config(&state_dir("sm90"), &[("FV_FAKE_DEVICE", "h100")]), Overrides::default()).await.unwrap();
    assert!(app.gate.engine().wait_ready().await.is_ready());
    let (_, v) = call(&app.router, "GET", "/fv/v1/status", None, None).await;
    assert_ne!(v["models"]["fake-h3-turbo"]["state"], "failed", "{v}");
    std::fs::remove_dir_all(&dir).ok();
}
