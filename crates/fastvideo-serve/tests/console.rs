//! The console's server side (WP-20): the admin token, minted API keys
//! (mint / list / revoke through `/fv/v1/admin/keys`, accepted by every
//! adapter, persisted in the file store), the fal schema catalog and the
//! `/console` pages, through the assembled fv-serve router with the fake
//! engine.

#![cfg(feature = "fal")]

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::Duration;

use axum::body::Body;
use axum::http::{header, Request, StatusCode};
use axum::Router;
use fastvideo_serve::config::{Config, JobBackend, KeyStoreBackend};
use fastvideo_serve::{App, Overrides};
use serde_json::{json, Value};
use tower::ServiceExt;

const ADMIN: &str = "fvadm_test-admin-token";

fn state_dir(tag: &str) -> PathBuf {
    std::env::temp_dir().join(format!(
        "fv-serve-console-{tag}-{:x}",
        std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()
    ))
}

fn config(dir: &Path, admin: Option<&str>) -> Config {
    let mut env = BTreeMap::new();
    env.insert("FV_URL_SIGNING_KEY".to_owned(), "k".to_owned());
    env.insert("FV_STATE_DIR".to_owned(), dir.display().to_string());
    env.insert("FV_PUBLIC_BASE_URL".to_owned(), "http://fv.test".to_owned());
    if let Some(a) = admin {
        env.insert("FV_ADMIN_TOKEN".to_owned(), a.to_owned());
    }
    let mut c = Config::default();
    c.apply_env(&env).unwrap();
    c.jobs.backend = JobBackend::Memory;
    c.engine.fake.step_ms = 1;
    c.validate().unwrap();
    c
}

async fn app(c: Config) -> App {
    let a = App::build(c, Overrides::default()).await.unwrap();
    a.gate.engine().wait_ready().await;
    a
}

struct Resp {
    status: StatusCode,
    headers: axum::http::HeaderMap,
    body: Vec<u8>,
}

impl Resp {
    fn json(&self) -> Value {
        serde_json::from_slice(&self.body).unwrap_or(Value::Null)
    }
    fn text(&self) -> String {
        String::from_utf8_lossy(&self.body).into_owned()
    }
}

async fn call(r: &Router, method: &str, uri: &str, auth: Option<String>, body: Option<Value>) -> Resp {
    let mut b = Request::builder().method(method).uri(uri);
    if let Some(a) = auth {
        b = b.header("authorization", a);
    }
    let req = match body {
        Some(v) => b.header("content-type", "application/json").body(Body::from(v.to_string())).unwrap(),
        None => b.body(Body::empty()).unwrap(),
    };
    let resp = r.clone().oneshot(req).await.unwrap();
    let (status, headers) = (resp.status(), resp.headers().clone());
    let body = axum::body::to_bytes(resp.into_body(), 1 << 24).await.unwrap().to_vec();
    Resp { status, headers, body }
}

fn admin() -> Option<String> {
    Some(format!("Bearer {ADMIN}"))
}

async fn mint(r: &Router, name: &str) -> (String, String) {
    let m = call(r, "POST", "/fv/v1/admin/keys", admin(), Some(json!({"name": name}))).await;
    assert_eq!(m.status, 201, "{}", m.text());
    let v = m.json();
    (v["api_key"].as_str().unwrap().to_owned(), v["key"]["id"].as_str().unwrap().to_owned())
}

#[tokio::test]
async fn minted_keys_drive_every_api_and_revoke_immediately() {
    let dir = state_dir("keys");
    let a = app(config(&dir, Some(ADMIN))).await;
    assert!(!a.admin_token_generated);
    assert_eq!(a.keys.backend_kind(), "file");
    let r = a.router.clone();

    // The admin API needs the admin token, not an API key.
    assert_eq!(call(&r, "GET", "/fv/v1/admin/keys", None, None).await.status, 401);
    assert_eq!(call(&r, "GET", "/fv/v1/admin/keys", Some("Bearer fvadm_nope".into()), None).await.status, 401);
    let (key, id) = mint(&r, "laptop").await;
    assert!(key.starts_with("fv_"));
    assert_eq!(call(&r, "GET", "/fv/v1/admin/keys", Some(format!("Bearer {key}")), None).await.status, 401);

    // fal (`Key`), native (`Bearer`) and MiniMax (`Bearer`) accept it.
    let sub = call(&r, "POST", "/minimax/h3-turbo/text-to-video", Some(format!("Key {key}")), Some(json!({"prompt": "a red fox"}))).await;
    assert_eq!(sub.status, 200, "{}", sub.text());
    let rid = sub.json()["request_id"].as_str().unwrap().to_owned();
    let caps = call(&r, "GET", "/fv/v1/capabilities", Some(format!("Bearer {key}")), None).await;
    assert_eq!(caps.status, 200);
    // The console learns the auth mode here; no key or token is echoed.
    assert_eq!(caps.json()["auth"], json!({"mode": "keys"}), "{}", caps.text());
    assert!(!caps.text().contains(&key) && !caps.text().contains(ADMIN));
    // Keys mode: the unauthenticated probe is refused, so the console keeps asking for a key.
    assert_eq!(call(&r, "GET", "/fv/v1/capabilities", None, None).await.status, 401);
    let mm = call(&r, "GET", "/v2/query/video_generation?task_id=1", Some(format!("Bearer {key}")), None).await;
    assert_ne!(mm.status, 401, "{}", mm.text());
    assert_ne!(mm.json()["base_resp"]["status_code"], 1004, "{}", mm.text());

    // The job finishes and belongs to the key.
    let mut done = false;
    for _ in 0..2000 {
        let s = call(&r, "GET", &format!("/minimax/h3-turbo/requests/{rid}/status"), Some(format!("Key {key}")), None).await;
        if s.json()["status"] == "COMPLETED" {
            done = true;
            break;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    assert!(done);
    let out = call(&r, "GET", &format!("/minimax/h3-turbo/requests/{rid}"), Some(format!("Key {key}")), None).await;
    assert_eq!(out.status, 200, "{}", out.text());
    assert!(out.json()["video"]["url"].as_str().unwrap().contains("/files/"));
    let job = a.ctx.jobs().by_external(fastvideo_protocol::ProtocolId::Fal, &rid).await.unwrap();
    assert_eq!(job.owner.unwrap().0, id);

    // Listed (without the key); last_used_at recorded.
    let l = call(&r, "GET", "/fv/v1/admin/keys", admin(), None).await.json();
    assert_eq!(l["backend"], "file");
    assert_eq!(l["keys"][0]["id"], id.as_str());
    assert!(l["keys"][0]["last_used_at"].is_string(), "{l}");
    assert!(!l.to_string().contains(&key));

    // Revoked: refused everywhere at once.
    let d = call(&r, "DELETE", &format!("/fv/v1/admin/keys/{id}"), admin(), None).await;
    assert_eq!(d.status, 200);
    assert_eq!(call(&r, "POST", "/minimax/h3-turbo/text-to-video", Some(format!("Key {key}")), Some(json!({"prompt": "x"}))).await.status, 401);
    assert_eq!(call(&r, "GET", "/fv/v1/capabilities", Some(format!("Bearer {key}")), None).await.status, 401);
    assert_eq!(call(&r, "DELETE", "/fv/v1/admin/keys/key_000000000000", admin(), None).await.status, 404);

    // Only digests on disk.
    let (key2, _) = mint(&r, "second").await;
    let text = std::fs::read_to_string(dir.join("api_keys.json")).unwrap();
    assert!(!text.contains(&key) && !text.contains(&key2) && !text.contains(ADMIN));
    drop(a);

    // A restart keeps the keys and the revocation.
    let a = app(config(&dir, Some(ADMIN))).await;
    let r = a.router.clone();
    assert_eq!(call(&r, "GET", "/fv/v1/capabilities", Some(format!("Bearer {key2}")), None).await.status, 200);
    assert_eq!(call(&r, "GET", "/fv/v1/capabilities", Some(format!("Bearer {key}")), None).await.status, 401);
    assert_eq!(call(&r, "GET", "/fv/v1/admin/keys", admin(), None).await.json()["keys"].as_array().unwrap().len(), 2);
    drop(a);
    std::fs::remove_dir_all(&dir).ok();
}

/// `FV_AUTH_MODE=none`: capabilities report `auth.mode = "none"` without
/// credentials, and every call the console makes (capabilities, fal submit,
/// status, result, upload) works with no `Authorization`, as the console
/// sends none in that mode. The admin routes still need the admin token.
#[tokio::test]
async fn auth_none_is_reported_and_the_console_needs_no_key() {
    let dir = state_dir("none");
    let mut env = BTreeMap::new();
    env.insert("FV_AUTH_MODE".to_owned(), "none".to_owned());
    let mut c = config(&dir, Some(ADMIN));
    c.apply_env(&env).unwrap();
    c.validate().unwrap();
    let a = app(c).await;
    let r = a.router.clone();

    let caps = call(&r, "GET", "/fv/v1/capabilities", None, None).await;
    assert_eq!(caps.status, 200, "{}", caps.text());
    assert_eq!(caps.json()["auth"], json!({"mode": "none"}), "{}", caps.text());
    assert!(!caps.text().contains(ADMIN));
    assert!(caps.json()["models"].as_array().is_some_and(|m| !m.is_empty()));

    // The console's model page: upload, submit, status, result, all without a key.
    let up = call(
        &r,
        "POST",
        "/storage/upload/initiate?storage_type=fal-cdn-v3",
        None,
        Some(json!({"content_type": "image/png", "file_name": "a.png"})),
    )
    .await;
    assert_eq!(up.status, 200, "{}", up.text());
    let sub = call(&r, "POST", "/minimax/h3-turbo/text-to-video", None, Some(json!({"prompt": "a red fox"}))).await;
    assert_eq!(sub.status, 200, "{}", sub.text());
    let rid = sub.json()["request_id"].as_str().unwrap().to_owned();
    let mut done = false;
    for _ in 0..2000 {
        let s = call(&r, "GET", &format!("/minimax/h3-turbo/requests/{rid}/status?logs=1"), None, None).await;
        assert_eq!(s.status, 200, "{}", s.text());
        if s.json()["status"] == "COMPLETED" {
            done = true;
            break;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    assert!(done);
    let out = call(&r, "GET", &format!("/minimax/h3-turbo/requests/{rid}"), None, None).await;
    assert_eq!(out.status, 200, "{}", out.text());
    assert!(out.json()["video"]["url"].as_str().unwrap().contains("/files/"));

    // The public status view: one local pool, ready once the job is done.
    let st = call(&r, "GET", "/fv/v1/status", None, None).await;
    assert_eq!(st.status, 200, "{}", st.text());
    assert_eq!(st.json()["pools"][0]["id"], "local");

    // Admin routes keep their token.
    assert_eq!(call(&r, "GET", "/fv/v1/admin/keys", None, None).await.status, 401);
    assert_eq!(call(&r, "GET", "/fv/v1/admin/keys", admin(), None).await.status, 200);

    // The shipped console reads the mode and drops the key prompts.
    let common = call(&r, "GET", "/console/assets/common.js", None, None).await.text();
    assert!(common.contains("c.auth.mode") && common.contains("export const needsKey"));
    for page in ["model.js", "director.js"] {
        let js = call(&r, "GET", &format!("/console/assets/{page}"), None, None).await.text();
        assert!(js.contains("needsKey()") && !js.contains("!apiKey()"), "{page} still gates on apiKey()");
    }
    drop(a);
    std::fs::remove_dir_all(&dir).ok();
}

/// `GET /fv/v1/status` on a single server: public even with keys, the
/// engine as one `local` pool (state, queue depth, running jobs), the name
/// map the console resolves fal apps with, and nothing secret.
#[tokio::test]
async fn single_server_status_is_public_and_tracks_the_engine() {
    let dir = state_dir("status");
    let mut c = config(&dir, Some(ADMIN));
    c.engine.fake.step_ms = 40;
    let a = app(c).await;
    let r = a.router.clone();
    let (key, _) = mint(&r, "status").await;

    let st = call(&r, "GET", "/fv/v1/status", None, None).await;
    assert_eq!(st.status, 200, "{}", st.text());
    let v = st.json();
    assert_eq!(v["object"], "fv.status");
    assert_eq!(v["gateway"], false);
    assert_eq!(v["state"], "ready", "{v}");
    let p = &v["pools"][0];
    assert_eq!((p["id"].as_str(), p["kind"].as_str(), p["state"].as_str()), (Some("local"), Some("local"), Some("ready")), "{v}");
    assert_eq!(p["available"], true);
    assert_eq!(p["workers"].as_array().unwrap().len(), 1);
    assert_eq!(p["workers"][0]["label"], "local");
    assert_eq!((p["queued"].as_u64(), p["running"].as_u64()), (Some(0), Some(0)));
    let models = p["models"].as_array().unwrap();
    assert!(!models.is_empty());
    for m in models {
        assert_eq!(v["models"][m.as_str().unwrap()]["state"], "ready");
        assert_eq!(v["models"][m.as_str().unwrap()]["pools"], json!(["local"]));
    }
    // fal apps name models by served name or alias; `names` resolves them.
    let target = v["names"]["h3-turbo"].as_str().expect("names resolves h3-turbo");
    assert!(models.iter().any(|m| m == target), "{v}");
    assert!(!st.text().contains(ADMIN) && !st.text().contains(&key) && !st.text().contains("fv.test"));

    // A running job shows as busy with the job counted.
    let sub = call(&r, "POST", "/minimax/h3-turbo/text-to-video", Some(format!("Key {key}")), Some(json!({"prompt": "a red fox"}))).await;
    assert_eq!(sub.status, 200, "{}", sub.text());
    let mut busy = false;
    for _ in 0..500 {
        let v = call(&r, "GET", "/fv/v1/status", None, None).await.json();
        if v["pools"][0]["state"] == "busy" {
            assert!(v["pools"][0]["running"].as_u64().unwrap() + v["pools"][0]["queued"].as_u64().unwrap() >= 1, "{v}");
            assert_eq!(v["state"], "busy");
            busy = true;
            break;
        }
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    assert!(busy, "never saw the engine busy");

    // Draining (shutdown step 1) shows as draining.
    a.gate.stop_admission();
    let v = call(&r, "GET", "/fv/v1/status", None, None).await.json();
    assert_eq!(v["pools"][0]["state"], "draining", "{v}");
    assert_eq!(v["pools"][0]["available"], false);
    drop(a);
    std::fs::remove_dir_all(&dir).ok();
}

#[tokio::test]
async fn admin_token_is_generated_when_unset() {
    let dir = state_dir("gen");
    let mut c = config(&dir, None);
    c.auth.key_store = KeyStoreBackend::Memory;
    let a = app(c).await;
    assert!(a.admin_token_generated);
    assert_eq!(a.keys.backend_kind(), "memory");
    // Nothing guessable gets in.
    for t in ["", "Bearer ", "Bearer fvadm_", "Bearer admin"] {
        let auth = (!t.is_empty()).then(|| t.to_owned());
        assert_eq!(call(&a.router, "GET", "/fv/v1/admin/keys", auth, None).await.status, 401);
    }
    assert!(!dir.join("api_keys.json").exists());
    drop(a);
    std::fs::remove_dir_all(&dir).ok();
}

#[tokio::test]
async fn console_pages_and_schema_serve() {
    let dir = state_dir("pages");
    let a = app(config(&dir, Some(ADMIN))).await;
    let r = a.router.clone();
    for uri in ["/console", "/console/admin", "/console/models/minimax/h3-max/reference-to-video", "/console/models/minimax/h3-max/director"] {
        let p = call(&r, "GET", uri, None, None).await;
        assert_eq!(p.status, 200, "{uri}");
        assert_eq!(p.headers[header::CONTENT_TYPE], "text/html; charset=utf-8");
        assert!(p.headers[header::CONTENT_SECURITY_POLICY].to_str().unwrap().contains("script-src 'self'"));
    }
    let js = call(&r, "GET", "/console/assets/model.js", None, None).await;
    assert_eq!(js.headers[header::CONTENT_TYPE], "text/javascript; charset=utf-8");
    let css = call(&r, "GET", "/console/assets/console.css", None, None).await;
    assert_eq!(css.headers[header::CONTENT_TYPE], "text/css; charset=utf-8");

    let cat = call(&r, "GET", "/fal/schema", None, None).await.json();
    let ids: Vec<&str> = cat["apps"].as_array().unwrap().iter().map(|a| a["id"].as_str().unwrap()).collect();
    assert_eq!(ids, fastvideo_fal::DEFAULT_APPS);
    // A multi-segment console page (fal's LTX / Wan sub-paths) serves the same page.
    let p = call(&r, "GET", "/console/models/fal-ai/wan/v2.2-5b/text-to-video/fast-wan", None, None).await;
    assert_eq!(p.status, 200);
    let s = call(&r, "GET", "/fal/schema/minimax/h3-max/image-to-video", None, None).await;
    assert_eq!(s.status, 200);
    assert_eq!(s.json()["properties"]["image_url"]["x-fv-media"], "image");
    assert_eq!(call(&r, "GET", "/fal/schema/minimax/h9/text-to-video", None, None).await.status, 404);
    // The director form: the resolutions the app's model serves.
    let d = call(&r, "GET", "/fal/schema/minimax/h3-max/director", None, None).await;
    assert_eq!(d.status, 200);
    assert!(d.json()["properties"]["resolution"]["enum"].as_array().is_some_and(|l| !l.is_empty()));

    // `server.console = false` unmounts the pages only.
    drop(a);
    let mut c = config(&dir, Some(ADMIN));
    c.server.console = false;
    let a = app(c).await;
    assert_eq!(call(&a.router, "GET", "/console", None, None).await.status, 404);
    assert_eq!(call(&a.router, "GET", "/fal/schema", None, None).await.status, 200);
    drop(a);
    std::fs::remove_dir_all(&dir).ok();
}
