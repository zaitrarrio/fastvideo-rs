//! The gateway (docs/serve/gateway.md) end to end on the fake engine: one
//! gateway (`engine.backend = "remote"`) in front of pod pools (fv-serve
//! workers on loopback TCP, `server.role = "worker"`) and a serverless pool
//! (a Runpod queue worker behind the local simulator), all sharing one D1
//! (the SQLite mock) and one local artifacts directory.
//!
//! Covered: every API routed to the right pool with status/result through
//! the gateway, caps aggregation, cancel, worker loss (re-dispatch on the
//! other worker), 503 for a pool that cannot take work, pool metrics and
//! the scaler hook, and director / Reactor signalling through the gateway.

#![cfg(all(feature = "http-client", feature = "openai-videos", feature = "minimax", feature = "fal", feature = "ltxapi"))]

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use fastvideo_deploy::runpod::sim::Sim;
use fastvideo_deploy::runpod::{RouterTransport, RunpodEnv, Worker, WorkerOptions};
use fastvideo_serve::config::{Config, EngineBackendKind, JobBackend, KeyStoreBackend, Mode, PoolCfg, PoolKind, Role};
use fastvideo_serve::deploy::{handler, Boot};
use fastvideo_serve::gateway::scale::{PoolMetrics, PoolScaler};
use fastvideo_serve::{App, Overrides};
use fastvideo_serve_kit::d1::client::{D1Error, D1Transport, RawReply};
use fastvideo_serve_kit::d1::mock::MockD1;
use fastvideo_serve_kit::{D1Client, KeyRing};
use serde_json::{json, Value};

const KEY: &str = "sk-gateway-user";
const TOKEN: &str = "gw-internal-token";
const ADMIN: &str = "fvadm_gateway_test";

fn tmp(tag: &str) -> PathBuf {
    std::env::temp_dir().join(format!(
        "fv-gw-{tag}-{:x}",
        std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()
    ))
}

/// A D1 transport that can be cut (a lost worker stops writing), with an
/// optional round-trip latency (the real D1 API: ~0.28 s).
#[derive(Clone)]
struct Cuttable {
    mock: MockD1,
    dead: Arc<AtomicBool>,
    delay: Duration,
}

#[async_trait::async_trait]
impl D1Transport for Cuttable {
    async fn post(&self, body: &Value) -> Result<RawReply, D1Error> {
        if self.dead.load(Ordering::SeqCst) {
            return Err(D1Error::Transport("worker is gone".into()));
        }
        if !self.delay.is_zero() {
            tokio::time::sleep(self.delay).await;
        }
        self.mock.post(body).await
    }
}

struct Shared {
    mock: MockD1,
    arts: PathBuf,
    d1_delay: Duration,
}

impl Shared {
    fn new() -> Self {
        let arts = tmp("arts");
        std::fs::create_dir_all(&arts).unwrap();
        Self { mock: MockD1::new(), arts, d1_delay: Duration::ZERO }
    }
    fn d1(&self, dead: &Arc<AtomicBool>) -> D1Client {
        D1Client::new(Arc::new(Cuttable { mock: self.mock.clone(), dead: dead.clone(), delay: self.d1_delay }))
            .with_retry(fastvideo_serve_kit::d1::RetryPolicy::immediate(1))
    }
    fn rows(&self, sql: &str) -> Vec<serde_json::Map<String, Value>> {
        self.mock.sql(sql, &[]).unwrap()
    }
}

fn base_config(tag: &str, sh: &Shared) -> Config {
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
    env.insert("FV_CALLBACKS_ALLOW_PRIVATE".to_owned(), "1".to_owned());
    let mut c = Config::default();
    c.apply_env(&env).unwrap();
    c.jobs.backend = JobBackend::D1;
    c.jobs.progress_interval_ms = 100;
    c.jobs.heartbeat_s = 1;
    c.auth.key_store = KeyStoreBackend::Memory;
    c.engine.fake.step_ms = 5;
    c.protocols.fastwan = false;
    c.webrtc.public_ip = "127.0.0.1".into();
    c.webrtc.ice_servers = vec![toml::toml! { urls = ["stun:127.0.0.1:9"] }.into()];
    c.reactor.short_edge = Some(96);
    c.reactor.h264 = "off".into();
    c
}

struct Running {
    app: App,
    base: String,
    dead: Arc<AtomicBool>,
    task: tokio::task::JoinHandle<()>,
}

impl Running {
    /// Stops answering HTTP and writing D1 (a lost worker).
    fn kill(&self) {
        self.dead.store(true, Ordering::SeqCst);
        self.task.abort();
    }
}

async fn serve(c: Config, sh: &Shared, listener: tokio::net::TcpListener, base: String, ov: Overrides) -> Running {
    let dead = Arc::new(AtomicBool::new(false));
    let ov = Overrides { d1: Some(sh.d1(&dead)), ..ov };
    let app = App::build(c, ov).await.unwrap();
    app.gate.engine().wait_ready().await;
    let router = app.router.clone();
    let task = tokio::spawn(async move {
        let _ = axum::serve(listener, router).await;
    });
    Running { app, base, dead, task }
}

async fn listen() -> (tokio::net::TcpListener, String) {
    let l = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let base = format!("http://{}", l.local_addr().unwrap());
    (l, base)
}

/// A pod worker serving `models`.
async fn pod_worker(sh: &Shared, tag: &str, models: &[&str], step_ms: u64) -> Running {
    let (l, base) = listen().await;
    let mut c = base_config(tag, sh);
    c.server.role = Role::Worker;
    c.server.public_base_url = Some(base.clone());
    c.server.worker_id = Some(format!("worker-{tag}"));
    c.engine.fake.models = models.iter().map(|m| m.to_string()).collect();
    c.engine.fake.step_ms = step_ms;
    c.validate().unwrap();
    serve(c, sh, l, base, Overrides::default()).await
}

/// A Runpod queue worker (serverless pool) taking jobs from `sim`.
async fn queue_worker(sh: &Shared, sim: &Sim, models: &[&str]) -> (App, tokio::task::JoinHandle<()>) {
    let mut c = base_config("queue", sh);
    c.server.role = Role::Worker;
    c.server.mode = Mode::RunpodQueue;
    c.server.worker_id = Some("queue-worker".into());
    c.engine.fake.models = models.iter().map(|m| m.to_string()).collect();
    c.validate().unwrap();
    let dead = Arc::new(AtomicBool::new(false));
    let app = App::build(c, Overrides { d1: Some(sh.d1(&dead)), ..Default::default() }).await.unwrap();
    app.gate.engine().wait_ready().await;
    let vars: BTreeMap<String, String> = sim.env("http://sim.local", "qw1", 200).into_iter().collect();
    let env = RunpodEnv::from_lookup(|k| vars.get(k).cloned()).unwrap();
    let h = handler(&app, Boot::now(), Arc::new(std::sync::OnceLock::new()));
    let opts = WorkerOptions { idle_backoff: Duration::from_millis(5), ..WorkerOptions::default() };
    let w = Worker::new(env, RouterTransport(sim.router()), h, opts);
    let t = tokio::spawn(async move {
        let _ = w.run(std::future::pending::<()>()).await;
    });
    (app, t)
}

fn pod_pool(id: &str, urls: &[&str], models: &[&str]) -> PoolCfg {
    PoolCfg {
        id: id.into(),
        kind: PoolKind::Pod,
        urls: urls.iter().map(|u| u.to_string()).collect(),
        fake_models: models.iter().map(|m| m.to_string()).collect(),
        stale_after_s: 3,
        retries: 1,
        ..PoolCfg::default()
    }
}

fn serverless_pool(id: &str, models: &[&str]) -> PoolCfg {
    PoolCfg {
        id: id.into(),
        kind: PoolKind::RunpodServerless,
        endpoint_id: Some("sim-ep".into()),
        fake_models: models.iter().map(|m| m.to_string()).collect(),
        stale_after_s: 3,
        ..PoolCfg::default()
    }
}

async fn gateway(sh: &Shared, pools: Vec<PoolCfg>, runpod_base: &str, ov: Overrides) -> Running {
    gateway_with(sh, pools, runpod_base, ov, |_| {}).await
}

/// [`gateway`] with `edit` applied to its config before validation.
async fn gateway_with(sh: &Shared, pools: Vec<PoolCfg>, runpod_base: &str, ov: Overrides, edit: impl FnOnce(&mut Config)) -> Running {
    let (l, base) = listen().await;
    let mut c = base_config("gw", sh);
    c.engine.backend = EngineBackendKind::Remote;
    c.server.public_base_url = Some(base.clone());
    c.pools = pools;
    c.gateway.tick_s = 1;
    c.gateway.watch_poll_ms = 100;
    c.gateway.reactor_model = Some("fake-sfwan".into());
    c.gateway.runpod_api_base = format!("{runpod_base}/v2");
    c.gateway.runpod_api_key = fastvideo_serve::config::Secret("rp-key".into());
    edit(&mut c);
    c.validate().unwrap();
    serve(c, sh, l, base, ov).await
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
    Some("Bearer sk-gateway-user")
}

fn fal_key() -> Option<&'static str> {
    Some("Key sk-gateway-user")
}

/// The pool each finished dispatch went to, by job external id.
fn pool_of(sh: &Shared, external_id: &str) -> String {
    let rows = sh
        .mock
        .sql(
            "SELECT d.pool AS pool FROM gw_dispatch d JOIN jobs j ON j.id = d.job_id WHERE j.external_id = ?",
            &[json!(external_id)],
        )
        .unwrap();
    rows.first().and_then(|r| r.get("pool")).and_then(Value::as_str).unwrap_or_default().to_owned()
}

#[derive(Default)]
struct Recorder(Mutex<Vec<Vec<PoolMetrics>>>);

#[async_trait::async_trait]
impl PoolScaler for Recorder {
    async fn observe(&self, pools: &[PoolMetrics]) {
        self.0.lock().unwrap().push(pools.to_vec());
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 6)]
async fn every_api_routes_to_its_pool_through_the_gateway() {
    let sh = Shared::new();
    let h3 = pod_worker(&sh, "h3", &["fake-h3-turbo", "fake-h3-max"], 5).await;
    let ltx = pod_worker(&sh, "ltx", &["fake-ltx-pro", "fake-ltx-turbo"], 5).await;
    let sim = Sim::new("rp-key").with_long_poll(Duration::from_millis(50));
    let (sim_base, _sim_task) = sim.serve("127.0.0.1:0").await.unwrap();
    let (_qapp, _qw) = queue_worker(&sh, &sim, &["fake-wan", "fake-sfwan"]).await;
    let rec = Arc::new(Recorder::default());
    let gw = gateway(
        &sh,
        vec![
            pod_pool("h3", &[&h3.base], &["fake-h3-turbo", "fake-h3-max"]),
            pod_pool("ltx", &[&ltx.base], &["fake-ltx-pro", "fake-ltx-turbo"]),
            serverless_pool("wan", &["fake-wan", "fake-sfwan"]),
        ],
        &sim_base,
        Overrides { scalers: vec![rec.clone()], ..Default::default() },
    )
    .await;
    let g = &gw.base;
    let http = Http::new();

    // Caps: every model of every pool, each naming its pool.
    let (s, caps, _) = http.call("GET", &format!("{g}/fv/v1/capabilities"), None, bearer()).await;
    assert_eq!(s, 200, "{caps}");
    let models: BTreeMap<String, Value> = caps["models"]
        .as_array()
        .unwrap()
        .iter()
        .map(|m| (m["caps"]["id"].as_str().unwrap().to_owned(), m["pools"].clone()))
        .collect();
    for (m, p) in [("fake-h3-turbo", "h3"), ("fake-h3-max", "h3"), ("fake-ltx-pro", "ltx"), ("fake-wan", "wan"), ("fake-sfwan", "wan")] {
        assert_eq!(models.get(m), Some(&json!([p])), "{m}: {caps}");
    }
    assert_eq!(caps["aliases"]["h3-turbo"], "fake-h3-turbo");
    assert_eq!(caps["pools"].as_array().unwrap().len(), 3);
    // The gateway's own auth mode (the console reads it), no secrets.
    assert_eq!(caps["auth"], json!({"mode": "keys"}), "{caps}");
    assert!(!caps.to_string().contains(KEY) && !caps.to_string().contains(TOKEN) && !caps.to_string().contains(ADMIN));

    // Public status view (no key even in keys mode): per-pool and per-worker
    // state from the tick's probes, with no URLs, endpoint or worker ids.
    let stv = http
        .poll(&format!("{g}/fv/v1/status"), None, |v| {
            // Probed at least once (before that, configured workers are assumed up with no last-seen).
            v["pools"].as_array().is_some_and(|p| p.iter().any(|p| p["id"] == "h3" && p["workers"][0]["last_seen_s"].is_number()))
        })
        .await;
    assert_eq!(stv["object"], "fv.status");
    assert_eq!(stv["gateway"], true);
    let text = stv.to_string();
    for secret in [h3.base.as_str(), ltx.base.as_str(), "127.0.0.1", KEY, TOKEN, ADMIN, "rp-key", "sim-ep", "worker-h3", "worker-ltx"] {
        assert!(!text.contains(secret), "status leaks {secret}: {stv}");
    }
    let by_id: BTreeMap<String, Value> = stv["pools"].as_array().unwrap().iter().map(|p| (p["id"].as_str().unwrap().to_owned(), p.clone())).collect();
    let h3s = &by_id["h3"];
    assert_eq!(h3s["kind"], "pod");
    assert!(matches!(h3s["state"].as_str(), Some("ready" | "busy")), "{h3s}");
    assert_eq!(h3s["available"], true);
    assert_eq!(h3s["workers"][0]["label"], "w1");
    assert!(matches!(h3s["workers"][0]["state"].as_str(), Some("ready" | "busy")), "{h3s}");
    assert!(h3s["workers"][0]["last_seen_s"].as_f64().is_some_and(|s| s < 30.0), "{h3s}");
    assert!(h3s["queued"].is_u64() && h3s["running"].is_u64());
    let wan = &by_id["wan"];
    assert_eq!(wan["kind"], "runpod-serverless");
    assert!(wan["worker_counts"].is_object(), "{wan}");
    assert_ne!(wan["state"], "down", "{wan}");
    assert_eq!(stv["models"]["fake-h3-turbo"]["pools"], json!(["h3"]));
    assert_eq!(stv["names"]["h3-turbo"], "fake-h3-turbo");
    // The worker itself refuses anything without the internal token.
    let (s, _, _) = http.call("GET", &format!("{}/fv/v1/capabilities", h3.base), None, bearer()).await;
    assert_eq!(s, 401);
    let (s, _, _) = http.call("GET", &format!("{}/health", h3.base), None, None).await;
    assert_eq!(s, 200);

    // Native → serverless pool (Runpod queue); result through the gateway.
    let (s, v, _) = http
        .call("POST", &format!("{g}/fv/v1/jobs"), Some(json!({"model": "fake-wan", "prompt": "a red fox", "seed": 3})), bearer())
        .await;
    assert_eq!(s, 202, "{v}");
    let nid = v["id"].as_str().unwrap().to_owned();
    let done = http.poll(&format!("{g}/fv/v1/jobs/{nid}"), bearer(), |v| matches!(v["status"].as_str(), Some("succeeded" | "failed"))).await;
    assert_eq!(done["status"], "succeeded", "{done}");
    let (s, _, hs) = http.call("GET", &format!("{g}/fv/v1/jobs/{nid}/content"), None, bearer()).await;
    assert_eq!(s, 302);
    let loc = hs["location"].to_str().unwrap().to_owned();
    assert!(loc.starts_with(g.as_str()), "the gateway signs output URLs: {loc}");
    let file = http.0.get(&loc).send().await.unwrap();
    assert_eq!(file.status(), 200);
    assert!(!file.bytes().await.unwrap().is_empty());
    assert_eq!(pool_of(&sh, &nid), "wan");

    // fal queue → pod pool h3.
    let (s, v, _) = http.call("POST", &format!("{g}/minimax/h3-turbo/text-to-video"), Some(json!({"prompt": "a kitten", "seed": 7})), fal_key()).await;
    assert_eq!(s, 200, "{v}");
    let rid = v["request_id"].as_str().unwrap().to_owned();
    let st = http.poll(&format!("{g}/minimax/h3-turbo/requests/{rid}/status?logs=1"), fal_key(), |v| v["status"] == "COMPLETED").await;
    assert!(st.get("error").is_none(), "{st}");
    let (s, out, _) = http.call("GET", &format!("{g}/minimax/h3-turbo/requests/{rid}"), None, fal_key()).await;
    assert_eq!(s, 200, "{out}");
    assert!(out["video"]["url"].as_str().unwrap().starts_with(g.as_str()));
    assert_eq!(pool_of(&sh, &rid), "h3");
    let worker = sh.mock.sql("SELECT worker FROM jobs WHERE external_id = ?", &[json!(rid)]).unwrap();
    assert_eq!(worker[0]["worker"], "worker-h3", "the pod worker adopted the row");

    // fal status/stream (SSE polled from D1 on the gateway).
    let (s, v, _) = http.call("POST", &format!("{g}/minimax/h3-max/text-to-video"), Some(json!({"prompt": "a lynx"})), fal_key()).await;
    assert_eq!(s, 200, "{v}");
    let rid2 = v["request_id"].as_str().unwrap().to_owned();
    let sse = http
        .0
        .get(format!("{g}/minimax/h3-max/requests/{rid2}/status/stream"))
        .header("authorization", fal_key().unwrap())
        .send()
        .await
        .unwrap();
    assert_eq!(sse.status(), 200);
    let text = tokio::time::timeout(Duration::from_secs(60), sse.text()).await.unwrap().unwrap();
    assert!(text.contains("COMPLETED"), "{text}");

    // MiniMax → pod pool h3.
    let body = json!({"model": "MiniMax-H3-Turbo", "content": [{"type": "text", "text": "a red fox"}], "resolution": "768P", "duration": 5, "ratio": "16:9"});
    let (s, v, _) = http.call("POST", &format!("{g}/v2/video_generation"), Some(body), bearer()).await;
    assert_eq!(s, 200, "{v}");
    let tid = v["task_id"].as_str().unwrap().to_owned();
    let t = http
        .poll(&format!("{g}/v2/query/video_generation/{tid}"), bearer(), |v| matches!(v["task"]["status"].as_str(), Some("succeeded" | "failed")))
        .await;
    assert_eq!(t["task"]["status"], "succeeded", "{t}");
    assert_eq!(pool_of(&sh, &tid), "h3");

    // LTX v2 → pod pool ltx.
    let (s, v, _) = http
        .call("POST", &format!("{g}/v2/text-to-video"), Some(json!({"prompt": "a red fox", "model": "ltx-2-5-fast", "duration": 6, "resolution": "1920x1080"})), bearer())
        .await;
    assert_eq!(s, 202, "{v}");
    let lid = v["id"].as_str().unwrap().to_owned();
    let lv = http.poll(&format!("{g}/v2/text-to-video/{lid}"), bearer(), |v| matches!(v["status"].as_str(), Some("completed" | "failed"))).await;
    assert_eq!(lv["status"], "completed", "{lv}");
    assert_eq!(pool_of(&sh, &lid), "ltx");

    // FastVideo /v1/videos (+ sync) → h3.
    let (s, v, _) = http.call("POST", &format!("{g}/v1/videos"), Some(json!({"model": "h3-turbo", "prompt": "a cat", "seconds": "5"})), None).await;
    assert_eq!(s, 200, "{v}");
    let vid = v["id"].as_str().unwrap().to_owned();
    let vv = http.poll(&format!("{g}/v1/videos/{vid}"), None, |v| matches!(v["status"].as_str(), Some("completed" | "failed"))).await;
    assert_eq!(vv["status"], "completed", "{vv}");
    assert_eq!(pool_of(&sh, &vid), "h3");
    let (s, _, hs) = http.call("POST", &format!("{g}/v1/videos/sync"), Some(json!({"model": "fake-wan", "prompt": "waves"})), None).await;
    assert!(s == 200 || s == 302, "sync answered {s}");
    let _ = hs;

    // Metrics: the tick closed the rows with durations; the scaler saw them.
    tokio::time::sleep(Duration::from_millis(2500)).await;
    let (s, pools, _) = http.call("GET", &format!("{g}/fv/v1/gateway/pools"), None, Some(&format!("Bearer {ADMIN}"))).await;
    assert_eq!(s, 200, "{pools}");
    let p: Vec<PoolMetrics> = serde_json::from_value(pools["pools"].clone()).unwrap();
    let h3m = p.iter().find(|m| m.pool == "h3").unwrap();
    assert!(h3m.run_time.count >= 3, "{h3m:?}");
    assert_eq!((h3m.queued, h3m.running), (0, 0));
    assert!(h3m.available && h3m.workers.total == 1 && h3m.workers.ready == 1, "{h3m:?}");
    assert!(p.iter().find(|m| m.pool == "wan").unwrap().run_time.count >= 1);
    let (s, _, _) = http.call("GET", &format!("{g}/fv/v1/gateway/pools"), None, bearer()).await;
    assert_eq!(s, 401, "pool metrics need the admin token");
    assert!(!rec.0.lock().unwrap().is_empty(), "the scaler hook ran");
    let (_, m, _) = http.call("GET", &format!("{g}/metrics"), None, None).await;
    assert!(m.as_str().unwrap_or_default().contains("fv_pool_queued"), "{m}");
    assert!(h3m.submitted_total >= 4, "{h3m:?}");
    let (s, hz, _) = http.call("GET", &format!("{g}/healthz"), None, None).await;
    assert_eq!((s, hz["state"].as_str()), (200, Some("ready")), "{hz}");

    // Drain (the autoscaler's request): the worker takes nothing new and
    // the gateway stops dispatching to it; undrain restores it.
    let tok = |r: reqwest::RequestBuilder| r.header("x-fv-internal-token", TOKEN);
    let d: Value = tok(http.0.post(format!("{}/fv/v1/internal/drain", h3.base))).send().await.unwrap().json().await.unwrap();
    assert_eq!(d["draining"], true, "{d}");
    tokio::time::sleep(Duration::from_millis(2500)).await;
    let (s, v, h) = http.call("POST", &format!("{g}/fv/v1/jobs"), Some(json!({"model": "fake-h3-turbo", "prompt": "while drained"})), bearer()).await;
    assert_eq!(s, 503, "{v}");
    assert!(h.get("retry-after").is_some());
    let d: Value = tok(http.0.post(format!("{}/fv/v1/internal/undrain", h3.base))).send().await.unwrap().json().await.unwrap();
    assert_eq!(d["draining"], false, "{d}");
    tokio::time::sleep(Duration::from_millis(2500)).await;
    let (s, v, _) = http.call("POST", &format!("{g}/fv/v1/jobs"), Some(json!({"model": "fake-h3-turbo", "prompt": "after undrain"})), bearer()).await;
    assert_eq!(s, 202, "{v}");
    drop(gw);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 6)]
async fn cancel_worker_loss_and_unavailable_pool() {
    let sh = Shared::new();
    // Slow jobs (4 steps x 1.5 s) on two h3 workers.
    let a = pod_worker(&sh, "a", &["fake-h3-turbo"], 1500).await;
    let b = pod_worker(&sh, "b", &["fake-h3-turbo"], 1500).await;
    // A pool whose only worker does not answer.
    let (dead_l, dead_base) = listen().await;
    drop(dead_l);
    let sim = Sim::new("rp-key");
    let (sim_base, _t) = sim.serve("127.0.0.1:0").await.unwrap();
    let gw = gateway(
        &sh,
        vec![pod_pool("h3", &[&a.base, &b.base], &["fake-h3-turbo"]), pod_pool("ltx", &[&dead_base], &["fake-ltx-turbo"])],
        &sim_base,
        Overrides::default(),
    )
    .await;
    let g = &gw.base;
    let http = Http::new();

    // 503 + Retry-After: the model is known (static caps) but its pool is down.
    let (s, v, h) = http.call("POST", &format!("{g}/fv/v1/jobs"), Some(json!({"model": "fake-ltx-turbo", "prompt": "x"})), bearer()).await;
    assert_eq!(s, 503, "{v}");
    assert!(h.get("retry-after").is_some());
    let (_, caps, _) = http.call("GET", &format!("{g}/fv/v1/capabilities"), None, bearer()).await;
    assert!(caps["models"].as_array().unwrap().iter().any(|m| m["caps"]["id"] == "fake-ltx-turbo"), "static caps still advertise it");
    // An unknown model is the API's own 4xx, not 503.
    let (s, _, _) = http.call("POST", &format!("{g}/fv/v1/jobs"), Some(json!({"model": "nope", "prompt": "x"})), bearer()).await;
    assert!((400..500).contains(&s));

    // Cancel through the gateway reaches the worker.
    let (s, v, _) = http.call("POST", &format!("{g}/fv/v1/jobs"), Some(json!({"model": "fake-h3-turbo", "prompt": "to cancel"})), bearer()).await;
    assert_eq!(s, 202, "{v}");
    let cid = v["id"].as_str().unwrap().to_owned();
    http.poll(&format!("{g}/fv/v1/jobs/{cid}"), bearer(), |v| v["status"] == "running").await;
    let (s, v, _) = http.call("DELETE", &format!("{g}/fv/v1/jobs/{cid}"), None, bearer()).await;
    assert_eq!(s, 200, "{v}");
    let c = http.poll(&format!("{g}/fv/v1/jobs/{cid}"), bearer(), |v| matches!(v["status"].as_str(), Some("cancelled" | "succeeded" | "failed"))).await;
    assert_eq!(c["status"], "cancelled", "{c}");

    // Worker loss: the worker running the job disappears; the reaper
    // dispatches it again on the other worker, where it completes.
    let (s, v, _) = http.call("POST", &format!("{g}/fv/v1/jobs"), Some(json!({"model": "fake-h3-turbo", "prompt": "survives a lost worker"})), bearer()).await;
    assert_eq!(s, 202, "{v}");
    let lid = v["id"].as_str().unwrap().to_owned();
    http.poll(&format!("{g}/fv/v1/jobs/{lid}"), bearer(), |v| v["status"] == "running").await;
    let holder = sh.mock.sql("SELECT worker FROM jobs WHERE external_id = ?", &[json!(lid)]).unwrap()[0]["worker"].as_str().unwrap().to_owned();
    let (victim, survivor) = if holder == "worker-a" { (&a, "worker-b") } else { (&b, "worker-a") };
    victim.kill();
    let done = http.poll(&format!("{g}/fv/v1/jobs/{lid}"), bearer(), |v| matches!(v["status"].as_str(), Some("succeeded" | "failed"))).await;
    assert_eq!(done["status"], "succeeded", "{done}");
    let row = sh.mock.sql("SELECT worker FROM jobs WHERE external_id = ?", &[json!(lid)]).unwrap();
    assert_eq!(row[0]["worker"], survivor);
    let d = sh.rows("SELECT attempt, state FROM gw_dispatch");
    assert!(d.iter().any(|r| r["attempt"] == json!(2.0) || r["attempt"] == json!(2)), "{d:?}");
    let _ = a.app.config.server.role;
    drop((gw, a, b));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 6)]
async fn serverless_cancel_and_lost_runpod_job() {
    let sh = Shared::new();
    let sim = Sim::new("rp-key").with_long_poll(Duration::from_millis(50));
    let (sim_base, _t) = sim.serve("127.0.0.1:0").await.unwrap();
    // No queue worker yet: jobs wait IN_QUEUE.
    let gw = gateway(&sh, vec![serverless_pool("wan", &["fake-wan"])], &sim_base, Overrides::default()).await;
    let g = &gw.base;
    let http = Http::new();
    let (s, v, _) = http.call("POST", &format!("{g}/fv/v1/jobs"), Some(json!({"model": "fake-wan", "prompt": "queued then cancelled"})), bearer()).await;
    assert_eq!(s, 202, "{v}");
    let id = v["id"].as_str().unwrap().to_owned();
    let (s, v, _) = http.call("DELETE", &format!("{g}/fv/v1/jobs/{id}"), None, bearer()).await;
    assert_eq!(s, 200, "{v}");
    let st = http.poll(&format!("{g}/fv/v1/jobs/{id}"), bearer(), |v| v["status"] != "queued").await;
    assert_eq!(st["status"], "cancelled", "{st}");
    let rp = sh.rows("SELECT ref FROM gw_dispatch");
    let rid = rp[0]["ref"].as_str().unwrap().to_owned();
    assert_eq!(sim.job(&rid).unwrap().status, fastvideo_deploy::runpod::sim::SimStatus::Cancelled);

    // A Runpod job that dies before the worker writes anything is lost:
    // with one retry it is queued again as a new Runpod job.
    let (s, v, _) = http.call("POST", &format!("{g}/fv/v1/jobs"), Some(json!({"model": "fake-wan", "prompt": "lost in the queue"})), bearer()).await;
    assert_eq!(s, 202, "{v}");
    let id2 = v["id"].as_str().unwrap().to_owned();
    let first = sh.mock.sql("SELECT d.ref AS ref FROM gw_dispatch d JOIN jobs j ON j.id = d.job_id WHERE j.external_id = ?", &[json!(id2)]).unwrap()[0]["ref"]
        .as_str()
        .unwrap()
        .to_owned();
    sim.cancel(&first); // the platform drops it (as a failed worker would)
    // Now a worker comes up and takes the re-dispatched job.
    let (_qapp, _qw) = queue_worker(&sh, &sim, &["fake-wan"]).await;
    let done = http.poll(&format!("{g}/fv/v1/jobs/{id2}"), bearer(), |v| matches!(v["status"].as_str(), Some("succeeded" | "failed"))).await;
    assert_eq!(done["status"], "succeeded", "{done}");
    let second = sh.mock.sql("SELECT d.ref AS ref, d.attempt AS attempt FROM gw_dispatch d JOIN jobs j ON j.id = d.job_id WHERE j.external_id = ?", &[json!(id2)]).unwrap();
    assert_ne!(second[0]["ref"].as_str().unwrap(), first);
}

/// A gateway with `FV_AUTH_MODE=none` (its workers stay trust-gateway
/// behind the internal token): capabilities say so without credentials, and
/// a fal job runs through the pool with no key, as the console sends it.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn gateway_with_auth_none_reports_it_and_takes_keyless_jobs() {
    let sh = Shared::new();
    let h3 = pod_worker(&sh, "h3-open", &["fake-h3-turbo"], 5).await;
    let sim = Sim::new("rp-key");
    let (sim_base, _t) = sim.serve("127.0.0.1:0").await.unwrap();
    let gw = gateway_with(&sh, vec![pod_pool("h3", &[&h3.base], &["fake-h3-turbo"])], &sim_base, Overrides::default(), |c| {
        c.auth.mode = fastvideo_serve_kit::AuthMode::None;
    })
    .await;
    let g = &gw.base;
    let http = Http::new();

    let (s, caps, _) = http.call("GET", &format!("{g}/fv/v1/capabilities"), None, None).await;
    assert_eq!(s, 200, "{caps}");
    assert_eq!(caps["auth"], json!({"mode": "none"}), "{caps}");
    assert_eq!(caps["gateway"], true);
    assert!(!caps.to_string().contains(TOKEN) && !caps.to_string().contains(ADMIN));

    let (s, sub, _) = http.call("POST", &format!("{g}/minimax/h3-turbo/text-to-video"), Some(json!({"prompt": "a red fox"})), None).await;
    assert_eq!(s, 200, "{sub}");
    let rid = sub["request_id"].as_str().unwrap();
    http.poll(&format!("{g}/minimax/h3-turbo/requests/{rid}/status"), None, |v| v["status"] == "COMPLETED").await;
    let (s, out, _) = http.call("GET", &format!("{g}/minimax/h3-turbo/requests/{rid}"), None, None).await;
    assert_eq!(s, 200, "{out}");
    assert!(out["video"]["url"].is_string(), "{out}");

    // Admin routes keep the admin token.
    let (s, _, _) = http.call("GET", &format!("{g}/fv/v1/admin/keys"), None, None).await;
    assert_eq!(s, 401);

    // The status view follows the worker: ready, then down once it is gone.
    let st = http
        .poll(&format!("{g}/fv/v1/status"), None, |v| v["pools"][0]["workers"][0]["state"] == "ready" && v["pools"][0]["workers"][0]["last_seen_s"].is_number())
        .await;
    assert_eq!(st["pools"][0]["state"], "ready", "{st}");
    assert_eq!(st["state"], "ready");
    assert_eq!(st["models"]["fake-h3-turbo"]["state"], "ready");
    h3.kill();
    let st = http.poll(&format!("{g}/fv/v1/status"), None, |v| v["pools"][0]["state"] == "down").await;
    assert_eq!(st["pools"][0]["workers"][0]["state"], "down", "{st}");
    assert_eq!(st["pools"][0]["available"], false);
    assert!(st["pools"][0]["workers"][0]["last_seen_s"].is_number(), "{st}");
    assert_eq!(st["models"]["fake-h3-turbo"]["state"], "down");
    assert!(!st.to_string().contains(&h3.base));
}

#[cfg(feature = "reactor")]
#[tokio::test(flavor = "multi_thread", worker_threads = 6)]
async fn director_and_reactor_sessions_through_the_gateway() {
    use fastvideo_webrtc::host::{AudioLayout, HostConfig, OfferOptions, PeerEvent, RtcHost};
    use fastvideo_webrtc::sdp::Direction;
    let sh = Shared::new();
    let d = pod_worker(&sh, "dir", &["fake-h3-turbo", "fake-h3-max"], 5).await;
    let r = pod_worker(&sh, "rt", &["fake-sfwan"], 5).await;
    let sim = Sim::new("rp-key");
    let (sim_base, _t) = sim.serve("127.0.0.1:0").await.unwrap();
    let gw = gateway(
        &sh,
        vec![
            pod_pool("h3", &[&d.base], &["fake-h3-turbo", "fake-h3-max"]),
            pod_pool("sfwan-live", &[&r.base], &["fake-sfwan"]),
        ],
        &sim_base,
        Overrides::default(),
    )
    .await;
    let g = &gw.base;
    let http = Http::new();

    // Director: ICE servers and a session answered by the h3 worker; media
    // flows peer ↔ worker directly.
    let (s, ice, _) = http.call("POST", &format!("{g}/wma/ice"), Some(json!({"app_id": "minimax/h3-max/director"})), fal_key()).await;
    assert_eq!(s, 200, "{ice}");
    assert!(ice["ice_servers"].is_array());
    let (s, _, _) = http.call("POST", &format!("{g}/wma/ice"), Some(json!({"app_id": "minimax/h3-max/director"})), None).await;
    assert_eq!(s, 401, "the gateway authenticates");
    let host = RtcHost::bind(HostConfig::loopback(true, false)).await.unwrap();
    let (pending, offer) = host
        .offer(OfferOptions {
            video: Some(Direction::RecvOnly),
            audio: Some((Direction::RecvOnly, AudioLayout::Stereo)),
            channels: vec!["control".into()],
            ..Default::default()
        })
        .await
        .unwrap();
    let (s, v, _) = http.call("POST", &format!("{g}/wma/session"), Some(json!({"app_id": "minimax/h3-max/director", "sdp": offer, "type": "offer"})), fal_key()).await;
    assert_eq!(s, 200, "{v}");
    let mut peer = pending.accept_answer(v["sdp"].as_str().unwrap()).await.unwrap();
    let info = tokio::time::timeout(Duration::from_secs(30), async {
        loop {
            match peer.next_event().await {
                Some(PeerEvent::Message(m)) => break serde_json::from_str::<Value>(m.as_text().unwrap()).unwrap(),
                Some(PeerEvent::Closed(r)) => panic!("closed: {r:?}"),
                Some(_) => {}
                None => panic!("peer gone"),
            }
        }
    })
    .await
    .unwrap();
    assert_eq!(info["type"], "session_info");
    let (s, hb, _) = http.call("POST", &format!("{g}/wma/session/heartbeat"), Some(json!({"session_id": v["session_id"]})), fal_key()).await;
    assert_eq!((s, hb), (200, json!({"alive": true})));
    let lease = sh.rows("SELECT kind, target, state FROM gw_sessions WHERE kind = 'director'");
    assert_eq!(lease[0]["target"], d.base.as_str());

    // Reactor: the session and signalling follow the caller's lease.
    let (s, v, _) = http.call("GET", &format!("{g}/session"), None, None).await;
    assert_eq!((s, v["state"].as_str()), (200, Some("ready")), "{v}");
    let (s, v, _) = http.call("POST", &format!("{g}/start_session"), Some(json!({})), None).await;
    assert_eq!(s, 200, "{v}");
    assert_eq!(v["state"], "waiting");
    let sid = v["session_id"].as_str().unwrap().to_owned();
    let (s, v, _) = http.call("GET", &format!("{g}/sessions/{sid}/transport/webrtc/ice_servers"), None, None).await;
    assert_eq!(s, 200, "{v}");
    let (s, v, _) = http.call("POST", &format!("{g}/sessions/{sid}/transport/webrtc/connections"), None, None).await;
    assert_eq!(s, 201, "{v}");
    let (s, _, _) = http.call("POST", &format!("{g}/stop_session"), Some(json!({"reason": "done"})), None).await;
    assert_eq!(s, 200);
    let l = sh.rows("SELECT state, target FROM gw_sessions WHERE kind = 'reactor'");
    assert_eq!((l[0]["state"].as_str(), l[0]["target"].as_str()), (Some("ended"), Some(r.base.as_str())));
    drop((gw, d, r));
}

/// `configs/serve/gateway.toml`: every pool's static caps resolve against
/// the CUDA catalog (CPU only) and the aliases name served models.
#[test]
fn shipped_gateway_config_resolves_every_pool() {
    let p = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../configs/serve/gateway.toml");
    let mut c = Config::from_toml(&std::fs::read_to_string(&p).unwrap(), "gateway.toml").unwrap();
    let mut env: BTreeMap<String, String> = [("FV_INTERNAL_TOKEN", "t"), ("FV_CF_ACCOUNT_ID", "a"), ("FV_CF_API_TOKEN", "t"), ("FV_D1_DATABASE_ID", "d")]
        .into_iter()
        .map(|(k, v)| (k.to_owned(), v.to_owned()))
        .collect();
    for pool in &c.pools {
        env.insert(format!("{}ENDPOINT", pool.env_prefix()), "ep".into());
    }
    env.insert("FV_POOL_SFWAN_LIVE_URLS".into(), "https://a.example, https://b.example".into());
    c.apply_env(&env).unwrap();
    c.validate().unwrap();
    assert_eq!(c.pools.iter().find(|p| p.id == "h3-turbo").unwrap().endpoint_id.as_deref(), Some("ep"));
    assert_eq!(c.pools.iter().find(|p| p.id == "sfwan-live").unwrap().urls.len(), 2);
    let mut ids = Vec::new();
    for pool in &c.pools {
        let caps = fastvideo_serve::gateway::static_caps(pool).unwrap_or_else(|e| panic!("{}: {e}", pool.id));
        assert!(!caps.is_empty(), "{}", pool.id);
        ids.extend(caps.into_iter().map(|(m, _)| m.id.0));
    }
    for (alias, model) in &c.aliases {
        assert!(ids.contains(model), "alias {alias} → {model} is not served by a pool ({ids:?})");
    }
    // The worker configs name their pool.
    for (f, pool) in [("runpod.toml", "h3-turbo"), ("runpod-h3-max.toml", "h3-max"), ("runpod-ltx.toml", "ltx"), ("runpod-wan.toml", "wan"), ("runpod-sfwan.toml", "sfwan-live")] {
        let p = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../configs/serve").join(f);
        let w = Config::from_toml(&std::fs::read_to_string(&p).unwrap(), f).unwrap();
        assert_eq!(w.gateway.pool.as_deref(), Some(pool), "{f}");
        assert!(c.pools.iter().any(|p| p.id == pool), "{f}");
        let models: Vec<String> = w.models.iter().map(|m| m.id.clone()).collect();
        let static_ids: Vec<String> = c.pools.iter().find(|p| p.id == pool).unwrap().models.iter().map(|m| m.id.clone()).collect();
        assert_eq!(models, static_ids, "{f}: the pool's static caps repeat the worker's [[models]]");
    }
}

/// Deterministic noise (PNG does not shrink it): a photo-sized input.
fn noise_png(w: u32, h: u32) -> String {
    use base64::Engine as _;
    let mut x: u32 = 0x9e37_79b9;
    let img = image::RgbImage::from_fn(w, h, |_, _| {
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        image::Rgb([x as u8, (x >> 8) as u8, (x >> 16) as u8])
    });
    let mut out = std::io::Cursor::new(Vec::new());
    img.write_to(&mut out, image::ImageFormat::Png).unwrap();
    format!("data:image/png;base64,{}", base64::engine::general_purpose::STANDARD.encode(out.into_inner()))
}

/// The shared artifacts directory with R2-like latency (measured on the
/// cluster, 2026-09-28: PUT 1-3 s, GET ~1 s).
struct SlowStore {
    inner: fastvideo_serve_kit::LocalArtifactStore,
    put: Duration,
    get: Duration,
}

impl SlowStore {
    fn arc(sh: &Shared, base: &str, put: Duration, get: Duration) -> Arc<dyn fastvideo_serve_kit::ArtifactStore> {
        let urls = fastvideo_serve_kit::artifacts::LocalUrls { public_base: base.parse().unwrap(), key: fastvideo_serve_kit::UrlKey::new("shared-signing-key") };
        Arc::new(Self { inner: fastvideo_serve_kit::LocalArtifactStore::new(sh.arts.clone(), urls), put, get })
    }
}

impl fastvideo_protocol::UrlSigner for SlowStore {
    fn url_for(&self, a: &fastvideo_protocol::Artifact, ttl: Duration) -> url::Url {
        fastvideo_protocol::UrlSigner::url_for(&self.inner, a, ttl)
    }
}

#[async_trait::async_trait]
impl fastvideo_serve_kit::ArtifactStore for SlowStore {
    async fn put(&self, src: &std::path::Path, meta: fastvideo_serve_kit::ArtifactMeta) -> Result<fastvideo_protocol::Artifact, fastvideo_protocol::ApiError> {
        tokio::time::sleep(self.put).await;
        self.inner.put(src, meta).await
    }
    async fn delete(&self, a: &fastvideo_protocol::Artifact) {
        self.inner.delete(a).await
    }
    fn signer(&self) -> &dyn fastvideo_protocol::UrlSigner {
        self
    }
    async fn open(&self, a: &fastvideo_protocol::Artifact) -> Result<fastvideo_serve_kit::ArtifactBody, fastvideo_protocol::ApiError> {
        tokio::time::sleep(self.get).await;
        self.inner.open(a).await
    }
}

/// One fal image-to-video job's phases (seconds).
#[derive(Clone, Copy, Debug, Default)]
struct Phases {
    /// Client: the submit call's latency.
    submit: f64,
    /// `created_at` → `dispatched_at` (the worker holds the job and its input).
    dispatch: f64,
    /// `dispatched_at` → `started_at`.
    wait: f64,
    /// `created_at` → `started_at` (fal `timings.queue`).
    queue: f64,
}

/// Submits `n` fal image-to-video jobs one after another through a gateway
/// whose inputs go inline up to `inline_max` bytes, over R2-like store and
/// D1 latency; returns each job's phases.
async fn i2v_phases(inline_max: u64, n: usize, tag: &str) -> Vec<Phases> {
    let mut sh = Shared::new();
    sh.d1_delay = Duration::from_millis(250);
    let (put, get) = (Duration::from_millis(1500), Duration::from_millis(1000));
    let (wl, wbase) = listen().await;
    let mut wc = base_config(&format!("{tag}-w"), &sh);
    wc.server.role = Role::Worker;
    wc.server.public_base_url = Some(wbase.clone());
    wc.server.worker_id = Some(format!("worker-{tag}"));
    wc.engine.fake.models = vec!["fake-h3-turbo".into()];
    wc.validate().unwrap();
    let w = serve(wc, &sh, wl, wbase.clone(), Overrides { artifacts: Some(SlowStore::arc(&sh, &wbase, put, get)), ..Default::default() }).await;
    let sim = Sim::new("rp-key");
    let (sim_base, _t) = sim.serve("127.0.0.1:0").await.unwrap();
    let (gl, gbase) = listen().await;
    let mut c = base_config(&format!("{tag}-gw"), &sh);
    c.engine.backend = EngineBackendKind::Remote;
    c.server.public_base_url = Some(gbase.clone());
    c.pools = vec![pod_pool("h3", &[&w.base], &["fake-h3-turbo"])];
    c.gateway.tick_s = 1;
    c.gateway.watch_poll_ms = 100;
    c.gateway.runpod_api_base = format!("{sim_base}/v2");
    c.gateway.inline_inputs_max_bytes = inline_max;
    c.validate().unwrap();
    let gw = serve(c, &sh, gl, gbase.clone(), Overrides { artifacts: Some(SlowStore::arc(&sh, &gbase, put, get)), ..Default::default() }).await;
    let g = &gw.base;
    let http = Http::new();
    let image = noise_png(1024, 576);
    let mut out = Vec::new();
    for i in 0..n {
        let t0 = std::time::Instant::now();
        let (s, v, _) = http
            .call("POST", &format!("{g}/minimax/h3-turbo/image-to-video"), Some(json!({"prompt": format!("a fox {i}"), "image_url": image})), fal_key())
            .await;
        let submit = t0.elapsed().as_secs_f64();
        assert_eq!(s, 200, "{v}");
        let rid = v["request_id"].as_str().unwrap().to_owned();
        http.poll(&format!("{g}/minimax/h3-turbo/requests/{rid}/status"), fal_key(), |v| v["status"] == "COMPLETED").await;
        let (s, res, _) = http.call("GET", &format!("{g}/minimax/h3-turbo/requests/{rid}"), None, fal_key()).await;
        assert_eq!(s, 200, "{res}");
        let row = sh.mock.sql("SELECT job FROM jobs WHERE external_id = ?", &[json!(rid)]).unwrap();
        let job: fastvideo_protocol::Job = serde_json::from_str(row[0]["job"].as_str().unwrap()).unwrap();
        let secs = |a: time::OffsetDateTime, b: time::OffsetDateTime| (b - a).as_seconds_f64();
        let d = job.dispatched_at.expect("the worker records dispatched_at");
        let st = job.started_at.unwrap();
        let p = Phases { submit, dispatch: secs(job.created_at, d), wait: secs(d, st), queue: secs(job.created_at, st) };
        // fal `timings`: dispatch + wait == queue.
        let t = &res["timings"];
        assert!(t["dispatch"].is_number() && t["wait"].is_number(), "{res}");
        let sum = t["dispatch"].as_f64().unwrap() + t["wait"].as_f64().unwrap();
        assert!((sum - t["queue"].as_f64().unwrap()).abs() < 1e-6, "{t}");
        out.push(p);
    }
    drop((gw, w));
    out
}

fn mean(v: &[Phases], f: impl Fn(&Phases) -> f64) -> f64 {
    v.iter().map(f).sum::<f64>() / v.len().max(1) as f64
}

/// Before/after of the input path (docs/serve/gateway.md §3.2): the same
/// image-to-video jobs through the store (R2 PUT on the gateway, GET on the
/// worker: `inline_inputs_max_bytes = 0`, the old path) and inline in the
/// dispatch request (the default), with R2-like store latency and D1 round
/// trips of 250 ms. Prints the phases.
#[tokio::test(flavor = "multi_thread", worker_threads = 6)]
async fn inline_inputs_take_the_store_off_the_dispatch_path() {
    let before = i2v_phases(0, 3, "before").await;
    let after = i2v_phases(8 * 1024 * 1024, 3, "after").await;
    println!("fake-engine gateway, fal image-to-video (1024x576 noise PNG, {} KB), R2 PUT 1.5 s / GET 1.0 s, D1 0.25 s per call", noise_png(1024, 576).len() * 3 / 4 / 1024);
    println!("{:<34} {:>9} {:>9} {:>9} {:>9}", "input path (mean of 3 jobs)", "submit_s", "dispatch", "wait", "queue");
    for (name, v) in [("before: R2 staging (inline 0)", &before), ("after: inline (default 8 MiB)", &after)] {
        println!(
            "{:<34} {:>9.2} {:>9.2} {:>9.2} {:>9.2}",
            name,
            mean(v, |p| p.submit),
            mean(v, |p| p.dispatch),
            mean(v, |p| p.wait),
            mean(v, |p| p.queue)
        );
    }
    for (i, (b, a)) in before.iter().zip(&after).enumerate() {
        println!("job {i}: before {b:?}\n       after  {a:?}");
    }
    let (b, a) = (mean(&before, |p| p.dispatch), mean(&after, |p| p.dispatch));
    assert!(b - a > 2.0, "the store round trips (1.5 s + 1.0 s) left the dispatch path: before {b:.2} s, after {a:.2} s");
    assert!(mean(&after, |p| p.queue) < mean(&before, |p| p.queue));
}

/// Inline inputs end to end: the job runs, the dispatch row never holds
/// the bytes, the background copy lets a re-dispatch after a worker loss
/// find the input, and finished jobs leave no input copy in the store.
/// Also: fal status polls are served from the gateway's memory, and a
/// worker refuses a passed-through URL the SSRF guard does not allow (424).
#[tokio::test(flavor = "multi_thread", worker_threads = 6)]
async fn inline_inputs_survive_a_worker_loss_and_are_cleaned_up() {
    let sh = Shared::new();
    let a = pod_worker(&sh, "ia", &["fake-h3-turbo"], 1500).await;
    let b = pod_worker(&sh, "ib", &["fake-h3-turbo"], 1500).await;
    let sim = Sim::new("rp-key");
    let (sim_base, _t) = sim.serve("127.0.0.1:0").await.unwrap();
    let gw = gateway(&sh, vec![pod_pool("h3", &[&a.base, &b.base], &["fake-h3-turbo"])], &sim_base, Overrides::default()).await;
    let g = &gw.base;
    let http = Http::new();
    let image = noise_png(320, 180);

    let (s, v, _) = http.call("POST", &format!("{g}/minimax/h3-turbo/image-to-video"), Some(json!({"prompt": "a fox", "image_url": image})), fal_key()).await;
    assert_eq!(s, 200, "{v}");
    let rid = v["request_id"].as_str().unwrap().to_owned();
    // Rapid status polls: most are answered from the gateway's view.
    for _ in 0..10 {
        let (s, _, _) = http.call("GET", &format!("{g}/minimax/h3-turbo/requests/{rid}/status"), None, fal_key()).await;
        assert_eq!(s, 200);
    }
    let (_, m, _) = http.call("GET", &format!("{g}/metrics"), None, None).await;
    assert!(m.as_str().unwrap_or_default().contains("fv_gateway_job_reads_total{source=\"memory\"}"), "{m}");
    http.poll(&format!("{g}/minimax/h3-turbo/requests/{rid}/status"), fal_key(), |v| v["status"] == "IN_PROGRESS").await;
    let row = || sh.mock.sql("SELECT d.inputs AS inputs FROM gw_dispatch d JOIN jobs j ON j.id = d.job_id WHERE j.external_id = ?", &[json!(rid)]).unwrap();
    let inputs: Value = serde_json::from_str(row()[0]["inputs"].as_str().unwrap()).unwrap();
    assert!(inputs[0].get("inline").is_none(), "the dispatch row never holds the bytes: {inputs}");
    // The background copy for a re-dispatch lands in the row.
    for _ in 0..100 {
        let i: Value = serde_json::from_str(row()[0]["inputs"].as_str().unwrap()).unwrap();
        if i[0]["artifact"].is_object() {
            break;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    let i: Value = serde_json::from_str(row()[0]["inputs"].as_str().unwrap()).unwrap();
    assert!(i[0]["artifact"].is_object() && i[0]["url"].is_string(), "{i}");

    // The worker holding it disappears: the job runs again on the other one.
    let holder = sh.mock.sql("SELECT worker FROM jobs WHERE external_id = ?", &[json!(rid)]).unwrap()[0]["worker"].as_str().unwrap().to_owned();
    let (victim, survivor) = if holder == "worker-ia" { (&a, "worker-ib") } else { (&b, "worker-ia") };
    victim.kill();
    let done = http.poll(&format!("{g}/minimax/h3-turbo/requests/{rid}/status"), fal_key(), |v| v["status"] == "COMPLETED").await;
    assert!(done.get("error").is_none(), "{done}");
    let (s, res, _) = http.call("GET", &format!("{g}/minimax/h3-turbo/requests/{rid}"), None, fal_key()).await;
    assert_eq!(s, 200, "{res}");
    assert!(res["video"]["url"].is_string(), "{res}");
    let w = sh.mock.sql("SELECT worker FROM jobs WHERE external_id = ?", &[json!(rid)]).unwrap();
    assert_eq!(w[0]["worker"], survivor);

    // Once the row is closed, no input copy is left in the store.
    let left = || {
        std::fs::read_dir(&sh.arts)
            .unwrap()
            .flatten()
            .flat_map(|d| std::fs::read_dir(d.path()).into_iter().flatten().flatten())
            .filter(|f| f.path().extension().is_some_and(|e| e == "png"))
            .count()
    };
    for _ in 0..100 {
        if left() == 0 {
            break;
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    assert_eq!(left(), 0, "input copies left in {}", sh.arts.display());

    // A passed-through URL the SSRF guard refuses: 424 (the gateway then
    // sends that input through the store).
    let survivor_run = if survivor == "worker-ia" { &a } else { &b };
    let mut job: Value = serde_json::from_str(sh.mock.sql("SELECT job FROM jobs WHERE external_id = ?", &[json!(rid)]).unwrap()[0]["job"].as_str().unwrap()).unwrap();
    job["id"] = json!(uuid::Uuid::new_v4().to_string());
    job["external_id"] = json!(uuid::Uuid::new_v4().to_string());
    job["state"] = json!({"status": "queued"});
    let env = json!({"job": job, "inputs": [{"path": "/gw/in/a.mp4", "source": "http://127.0.0.1:9/a.mp4", "kind": "video", "bytes": 10}], "attempt": 1});
    let r = http.0.post(format!("{}/fv/v1/internal/jobs", survivor_run.base)).header("x-fv-internal-token", TOKEN).json(&env).send().await.unwrap();
    let st = r.status().as_u16();
    let body = r.text().await.unwrap();
    assert_eq!(st, 424, "{body}");
    drop((gw, a, b));
}
