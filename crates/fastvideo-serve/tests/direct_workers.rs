//! Direct workers (fv-control `control_plane: "direct"`,
//! docs/control/gateway-less-auth.md): fv-serve workers that clients call
//! at their own URL, with `gateway.direct` (`FV_WORKER_DIRECT=1`). They
//! check API keys themselves; minted keys live in the shared D1 `api_keys`
//! table (the SQLite mock here). Moved from the retired `tests/gateway.rs`.

#![cfg(all(feature = "http-client", feature = "minimax"))]

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::time::Duration;

use fastvideo_serve::config::{Config, JobBackend, KeyStoreBackend, Role};
use fastvideo_serve::{App, Overrides};
use fastvideo_serve_kit::d1::mock::MockD1;
use fastvideo_serve_kit::{D1Client, KeyRing};
use serde_json::{json, Value};

const KEY: &str = "sk-direct-user";
const TOKEN: &str = "direct-internal-token";
const ADMIN: &str = "fvadm_direct_test";

fn tmp(tag: &str) -> PathBuf {
    tempfile::Builder::new().prefix(&format!("fv-direct-{tag}-")).tempdir().unwrap().keep()
}

struct Shared {
    mock: MockD1,
    arts: PathBuf,
}

impl Shared {
    fn new() -> Self {
        let arts = tmp("arts");
        std::fs::create_dir_all(&arts).unwrap();
        Self { mock: MockD1::new(), arts }
    }
    fn d1(&self) -> D1Client {
        D1Client::new(std::sync::Arc::new(self.mock.clone())).with_retry(fastvideo_serve_kit::d1::RetryPolicy::immediate(1))
    }
}

struct Running {
    app: App,
    base: String,
    _task: tokio::task::JoinHandle<()>,
}

/// A direct worker serving `fake-h3-turbo`.
async fn direct_worker(sh: &Shared, tag: &str) -> Running {
    let l = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let base = format!("http://{}", l.local_addr().unwrap());
    let mut env = BTreeMap::new();
    env.insert("FV_API_KEYS".to_owned(), KeyRing::hash_hex(KEY));
    env.insert("FV_URL_SIGNING_KEY".to_owned(), "shared-signing-key".to_owned());
    env.insert("FV_STATE_DIR".to_owned(), tmp(tag).display().to_string());
    env.insert("FV_INTERNAL_TOKEN".to_owned(), TOKEN.to_owned());
    env.insert("FV_ARTIFACTS_DIR".to_owned(), sh.arts.display().to_string());
    env.insert("FV_ADMIN_TOKEN".to_owned(), ADMIN.to_owned());
    env.insert("FV_CF_ACCOUNT_ID".to_owned(), "acct".to_owned());
    env.insert("FV_CF_API_TOKEN".to_owned(), "tok".to_owned());
    env.insert("FV_D1_DATABASE_ID".to_owned(), "db".to_owned());
    let mut c = Config::default();
    c.apply_env(&env).unwrap();
    c.jobs.backend = JobBackend::D1;
    c.jobs.progress_interval_ms = 100;
    c.jobs.heartbeat_s = 1;
    c.engine.fake.step_ms = 5;
    c.protocols.fastwan = false;
    c.server.role = Role::Worker;
    c.gateway.direct = true;
    c.gateway.register = false;
    c.server.public_base_url = Some(base.clone());
    c.server.worker_id = Some(format!("direct-{tag}"));
    c.auth.key_store = KeyStoreBackend::D1;
    c.engine.fake.models = vec!["fake-h3-turbo".into()];
    c.validate().unwrap();
    let app = App::build(c, Overrides { d1: Some(sh.d1()), ..Default::default() }).await.unwrap();
    app.gate.engine().wait_ready().await;
    let router = app.router.clone();
    let task = tokio::spawn(async move {
        let _ = axum::serve(l, router).await;
    });
    Running { app, base, _task: task }
}

struct Http(reqwest::Client);

impl Http {
    fn new() -> Self {
        Self(reqwest::Client::builder().no_proxy().redirect(reqwest::redirect::Policy::none()).build().unwrap())
    }
    async fn call(&self, method: &str, url: &str, body: Option<Value>, auth: Option<&str>) -> (u16, Value, reqwest::header::HeaderMap) {
        let mut r = self.0.request(reqwest::Method::from_bytes(method.as_bytes()).unwrap(), url);
        if let Some(a) = auth {
            r = r.header("authorization", a);
        }
        if let Some(b) = body {
            r = r.json(&b);
        }
        let resp = r.send().await.unwrap();
        let s = resp.status().as_u16();
        let h = resp.headers().clone();
        let bytes = resp.bytes().await.unwrap();
        (s, serde_json::from_slice(&bytes).unwrap_or(Value::String(String::from_utf8_lossy(&bytes).into_owned())), h)
    }
    async fn poll(&self, url: &str, auth: Option<&str>, done: impl Fn(&Value) -> bool) -> Value {
        for _ in 0..1200 {
            let (s, v, _) = self.call("GET", url, None, auth).await;
            if s == 200 && done(&v) {
                return v;
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
        panic!("{url} never finished");
    }
}

fn bearer() -> Option<&'static str> {
    Some("Bearer sk-direct-user")
}

/// Direct workers authenticate clients themselves: no key → 401, the
/// admin token mints and revokes, a minted key works on every worker (also
/// on one started later: a restart or a scale-up), a revoked one on none,
/// and the internal token still guards `/fv/v1/internal/*` only.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn direct_workers_take_minted_keys_and_the_admin_token() {
    let sh = Shared::new();
    let w1 = direct_worker(&sh, "direct-1").await;
    let w2 = direct_worker(&sh, "direct-2").await;
    let http = Http::new();
    let admin = format!("Bearer {ADMIN}");
    let caps = |b: &str| format!("{b}/fv/v1/capabilities");
    let keys = |b: &str| format!("{b}/fv/v1/admin/keys");

    // Open: health. Closed without a key: the APIs. The static FV_API_KEYS key works.
    assert_eq!(http.call("GET", &format!("{}/health", w1.base), None, None).await.0, 200);
    let (s, v, _) = http.call("GET", &caps(&w1.base), None, None).await;
    assert_eq!(s, 401, "{v}");
    assert!(!v.to_string().contains("internal token"), "a direct worker answers like a server, not like a cluster worker: {v}");
    assert_eq!(http.call("POST", &format!("{}/minimax/h3-turbo/text-to-video", w1.base), Some(json!({"prompt": "x"})), None).await.0, 401);
    let (s, v, _) = http.call("GET", &caps(&w1.base), None, bearer()).await;
    assert_eq!(s, 200, "{v}");
    assert_eq!(v["auth"], json!({"mode": "keys"}), "{v}");
    assert!(!v.to_string().contains(TOKEN) && !v.to_string().contains(ADMIN));
    // The internal routes keep the internal token (not the admin token).
    let status = format!("{}/fv/v1/internal/status", w1.base);
    assert_eq!(http.call("GET", &status, None, Some(&admin)).await.0, 401);
    let r = http.0.get(&status).header("x-fv-internal-token", TOKEN).send().await.unwrap();
    assert_eq!(r.status().as_u16(), 200);

    // Admin routes: the admin token only.
    assert_eq!(http.call("GET", &keys(&w1.base), None, bearer()).await.0, 401);
    let (s, v, _) = http.call("GET", &keys(&w1.base), None, Some(&admin)).await;
    assert_eq!(s, 200, "{v}");
    assert_eq!(v["backend"], "d1");

    // Mint on one worker: it works there at once and runs a job.
    let (s, m, _) = http.call("POST", &keys(&w1.base), Some(json!({"name": "laptop"})), Some(&admin)).await;
    assert_eq!(s, 201, "{m}");
    let plain = m["api_key"].as_str().unwrap().to_owned();
    let kid = m["key"]["id"].as_str().unwrap().to_owned();
    let (bk, kk) = (format!("Bearer {plain}"), format!("Key {plain}"));
    assert_eq!(http.call("GET", &caps(&w1.base), None, Some(&bk)).await.0, 200);
    let (s, sub, _) = http.call("POST", &format!("{}/minimax/h3-turbo/text-to-video", w1.base), Some(json!({"prompt": "a red fox"})), Some(&kk)).await;
    assert_eq!(s, 200, "{sub}");
    let rid = sub["request_id"].as_str().unwrap();
    http.poll(&format!("{}/minimax/h3-turbo/requests/{rid}/status", w1.base), Some(&kk), |v| v["status"] == "COMPLETED").await;
    // The other worker after its D1 refresh (every 30 s; forced here).
    w2.app.keys.refresh().await.unwrap();
    assert_eq!(http.call("GET", &caps(&w2.base), None, Some(&bk)).await.0, 200);
    // A worker started later (restart, scale-up) loads it at start.
    let w3 = direct_worker(&sh, "direct-3").await;
    assert_eq!(http.call("GET", &caps(&w3.base), None, Some(&bk)).await.0, 200);

    // Revoke on every worker (what fv-control does): refused everywhere at once.
    for w in [&w1, &w2, &w3] {
        let (s, v, _) = http.call("DELETE", &format!("{}/{kid}", keys(&w.base)), None, Some(&admin)).await;
        assert_eq!(s, 200, "{v}");
        assert_eq!(v["key"]["revoked"], true, "{v}");
    }
    for w in [&w1, &w2, &w3] {
        assert_eq!(http.call("GET", &caps(&w.base), None, Some(&bk)).await.0, 401);
    }
    // A worker that never saw the DELETE learns it from D1 (start or refresh).
    let w4 = direct_worker(&sh, "direct-4").await;
    assert_eq!(http.call("GET", &caps(&w4.base), None, Some(&bk)).await.0, 401);
    // The admin token is not a client key.
    assert_eq!(http.call("GET", &caps(&w1.base), None, Some(&admin)).await.0, 401);
}
