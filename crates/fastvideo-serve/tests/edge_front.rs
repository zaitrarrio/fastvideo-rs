//! The edge as the only entry point (docs/serve/edge-control-plane.md): the
//! native edge host (`fastvideo_serve::edge_host`: family objects, registry,
//! public front) in front of fake-engine workers that are API fronts
//! (`dispatch.front`, `auth.mode = trust-edge`), all on one D1 mock.
//!
//! - `every_api_through_the_edge`: native, fal queue (status, SSE, result by
//!   id twice through two fronts, the same URL), MiniMax, LTX and
//!   OpenAI-style jobs each run in their model's family; the merged
//!   capabilities, models and status views; auth refusals in each API's shape.
//! - `keys_quotas_and_admission`: a key minted at the edge works at once and
//!   is refused at once after its revoke; the per-key rate limit and the
//!   per-model queue limit are refused in the API's own shape; a cancel
//!   through the edge stops a job running on another front.
//! - `uploads_follow_their_front`: fal storage and LTX uploads land on the
//!   front that issued the ticket; a job on another front ingests them
//!   through the edge.
//! - `a_reactor_session_is_admitted_by_the_edge`: the family object admits
//!   the session, the edge proxies the signalling, a batch job waits for the
//!   GPU, a second session is refused, stop frees the GPU.

#![cfg(all(feature = "http-client", feature = "openai-videos", feature = "minimax", feature = "fal", feature = "ltxapi"))]

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use fastvideo_dispatch_proto::front::Quotas;
use fastvideo_dispatch_proto::sched::Cfg;
use fastvideo_serve::config::{Config, JobBackend, KeyStoreBackend, Role};
use fastvideo_serve::edge_host::{EdgeHost, EdgeHostCfg};
use fastvideo_serve::{App, Overrides};
use fastvideo_serve_kit::d1::mock::MockD1;
use fastvideo_serve_kit::keys::{D1KeyBackend, KeyStore};
use fastvideo_serve_kit::{D1Client, KeyRing};
use serde_json::{json, Value};

const KEY: &str = "sk-edge-user";
const ADMIN: &str = "fvadm_edge_test";
const TOKEN: &str = "edge-internal-token";

fn init_log() {
    if std::env::var("RUST_LOG").is_ok() {
        let _ = tracing_subscriber::fmt().with_env_filter(tracing_subscriber::EnvFilter::from_default_env()).with_test_writer().try_init();
    }
}

fn tmp(tag: &str) -> PathBuf {
    std::env::temp_dir().join(format!("fv-edgefront-{tag}-{:x}", std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()))
}

struct Env {
    mock: MockD1,
    arts: PathBuf,
    edge: Arc<EdgeHost>,
    base: String,
    http: Http,
}

struct Running {
    #[allow(dead_code)]
    app: App,
    base: String,
    task: tokio::task::JoinHandle<()>,
}

impl Drop for Running {
    fn drop(&mut self) {
        self.task.abort();
    }
}

impl Env {
    async fn new(edit: impl FnOnce(&mut EdgeHostCfg)) -> Self {
        let mock = MockD1::new();
        let arts = tmp("arts");
        std::fs::create_dir_all(&arts).unwrap();
        let keys = KeyStore::open(Arc::new(D1KeyBackend::open(D1Client::new(Arc::new(mock.clone()))).await.unwrap())).await.unwrap();
        let mut cfg = EdgeHostCfg::new(TOKEN, keys);
        cfg.admin_token = Some(ADMIN.into());
        cfg.static_keys = KeyRing::from_plain([KEY]);
        cfg.sched = Cfg { reconnect_grace_ms: 3_000, stale_after_ms: 8_000, ack_timeout_ms: 2_000, ..Cfg::default() };
        cfg.reactor_model = Some("fake-sfwan".into());
        edit(&mut cfg);
        let l = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let (edge, base) = EdgeHost::start(cfg, l).await;
        Self { mock, arts, edge, base, http: Http::new() }
    }

    fn config(&self, tag: &str) -> Config {
        let mut env = BTreeMap::new();
        env.insert("FV_URL_SIGNING_KEY".to_owned(), "shared-signing-key".to_owned());
        env.insert("FV_STATE_DIR".to_owned(), tmp(tag).display().to_string());
        env.insert("FV_INTERNAL_TOKEN".to_owned(), TOKEN.to_owned());
        env.insert("FV_ARTIFACTS_DIR".to_owned(), self.arts.display().to_string());
        env.insert("FV_ADMIN_TOKEN".to_owned(), ADMIN.to_owned());
        env.insert("FV_CF_ACCOUNT_ID".to_owned(), "acct".to_owned());
        env.insert("FV_CF_API_TOKEN".to_owned(), "tok".to_owned());
        env.insert("FV_D1_DATABASE_ID".to_owned(), "db".to_owned());
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
        c.gateway.watch_poll_ms = 200;
        c
    }

    /// A fake-engine worker that is a front for `families` (`models` with
    /// their family).
    async fn front(&self, tag: &str, models: &[(&str, &str)], step_ms: u64, edit: impl FnOnce(&mut Config)) -> Running {
        let l = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let own = format!("http://{}", l.local_addr().unwrap());
        let mut c = self.config(tag);
        c.server.role = Role::Worker;
        c.server.worker_id = Some(format!("worker-{tag}"));
        c.server.public_base_url = Some(self.base.clone());
        c.engine.fake.models = models.iter().map(|(m, _)| m.to_string()).collect();
        c.engine.fake.step_ms = step_ms;
        c.dispatch.do_url = Some(self.base.clone());
        c.dispatch.front = true;
        c.dispatch.endpoint = Some(own.clone());
        let mut fams: Vec<String> = models.iter().map(|(_, f)| f.to_string()).collect();
        fams.sort();
        fams.dedup();
        c.dispatch.families = fams;
        c.dispatch.model_families = models.iter().map(|(m, f)| (m.to_string(), f.to_string())).collect();
        c.dispatch.capacity = 1;
        c.dispatch.sessions = 1;
        c.dispatch.status_s = 1;
        edit(&mut c);
        c.validate().unwrap();
        let app = App::build(c, Overrides { d1: Some(D1Client::new(Arc::new(self.mock.clone()))), ..Overrides::default() }).await.unwrap();
        app.gate.engine().wait_ready().await;
        let router = app.router.clone();
        let task = tokio::spawn(async move {
            let _ = axum::serve(l, router).await;
        });
        Running { app, base: own, task }
    }

    /// Waits until `n` fronts are ready at the edge.
    async fn fronts(&self, n: usize) {
        let t0 = Instant::now();
        loop {
            let r = self.edge.registry();
            let mut ids: Vec<String> = r.fronts().map(|x| x.worker.worker_id.clone()).collect();
            ids.sort();
            ids.dedup();
            if ids.len() >= n {
                return;
            }
            assert!(t0.elapsed() < Duration::from_secs(30), "fronts never got ready: {r:?}");
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
    }

    fn worker_of(&self, external_id: &str) -> String {
        let r = self.mock.sql("SELECT worker FROM jobs WHERE external_id = ?", &[json!(external_id)]).unwrap();
        r.first().and_then(|r| r["worker"].as_str()).unwrap_or_default().to_owned()
    }
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
    Some("Bearer sk-edge-user")
}

fn fal_key() -> Option<&'static str> {
    Some("Key sk-edge-user")
}

fn native_done(v: &Value) -> bool {
    matches!(v["status"].as_str(), Some("succeeded" | "failed" | "cancelled"))
}

#[tokio::test(flavor = "multi_thread", worker_threads = 6)]
async fn every_api_through_the_edge() {
    init_log();
    let e = Env::new(|_| {}).await;
    let a = e.front("a", &[("fake-h3-turbo", "h3"), ("fake-h3-max", "h3")], 5, |_| {}).await;
    let a2 = e.front("a2", &[("fake-h3-turbo", "h3"), ("fake-h3-max", "h3")], 5, |_| {}).await;
    let b = e.front("b", &[("fake-ltx-turbo", "ltx"), ("fake-wan", "wan")], 5, |_| {}).await;
    e.fronts(3).await;
    let g = e.base.clone();
    let http = &e.http;

    // Auth refusals, in each API's shape.
    let (s, _, _) = http.call("POST", &format!("{g}/fv/v1/jobs"), Some(json!({"model": "fake-wan", "prompt": "x"})), None).await;
    assert_eq!(s, 401);
    let (s, _, _) = http.call("POST", &format!("{g}/fv/v1/jobs"), Some(json!({"model": "fake-wan", "prompt": "x"})), Some("Bearer nope")).await;
    assert_eq!(s, 401);
    let (s, v, _) = http.call("POST", &format!("{g}/minimax/h3-turbo/text-to-video"), Some(json!({"prompt": "x"})), bearer()).await;
    assert_eq!(s, 401, "fal wants `Key`: {v}");
    // The workers' own URLs refuse callers without the internal token.
    let (s, _, _) = http.call("POST", &format!("{}/fv/v1/jobs", a.base), Some(json!({"model": "fake-h3-turbo", "prompt": "x"})), bearer()).await;
    assert_eq!(s, 401);

    // Native → family wan (worker b).
    let (s, v, _) = http.call("POST", &format!("{g}/fv/v1/jobs"), Some(json!({"model": "fake-wan", "prompt": "a red fox", "seed": 3})), bearer()).await;
    assert_eq!(s, 202, "{v}");
    let nid = v["id"].as_str().unwrap().to_owned();
    let done = http.poll(&format!("{g}/fv/v1/jobs/{nid}"), bearer(), native_done).await;
    assert_eq!(done["status"], "succeeded", "{done}");
    assert_eq!(e.worker_of(&nid), "worker-b");
    let (s, _, hs) = http.call("GET", &format!("{g}/fv/v1/jobs/{nid}/content"), None, bearer()).await;
    assert_eq!(s, 302);
    let loc = hs["location"].to_str().unwrap().to_owned();
    assert!(loc.starts_with(g.as_str()), "output URLs point at the edge: {loc}");
    let file = http.0.get(&loc).send().await.unwrap();
    assert_eq!(file.status(), 200, "the edge serves the file through a front");
    assert!(!file.bytes().await.unwrap().is_empty());

    // fal queue → family h3.
    let (s, v, _) = http.call("POST", &format!("{g}/minimax/h3-turbo/text-to-video"), Some(json!({"prompt": "a kitten", "seed": 7})), fal_key()).await;
    assert_eq!(s, 200, "{v}");
    let rid = v["request_id"].as_str().unwrap().to_owned();
    assert!(v["status_url"].as_str().unwrap().starts_with(g.as_str()), "fal URLs point at the edge: {v}");
    http.poll(&format!("{g}/minimax/h3-turbo/requests/{rid}/status"), fal_key(), |v| v["status"] == "COMPLETED").await;
    let (s, out, _) = http.call("GET", &format!("{g}/minimax/h3-turbo/requests/{rid}"), None, fal_key()).await;
    assert_eq!(s, 200, "{out}");
    assert!(e.worker_of(&rid).starts_with("worker-a"), "{}", e.worker_of(&rid));
    let url1 = out["video"]["url"].as_str().unwrap().to_owned();
    // The same result URL a second later, whichever front answers (#12).
    tokio::time::sleep(Duration::from_millis(1_100)).await;
    let owner = format!("key_{}", &KeyRing::hash_hex(KEY)[..12]);
    let verdict = json!({"v": 1, "key": owner, "presented": true, "valid": true, "scheme": "key"}).to_string();
    for (who, base) in [("a", &a.base), ("a2", &a2.base)] {
        let r = http
            .0
            .get(format!("{base}/minimax/h3-turbo/requests/{rid}"))
            .header("x-fv-internal-token", TOKEN)
            .header("x-fv-edge-auth", &verdict)
            .send()
            .await
            .unwrap();
        let v: Value = r.json().await.unwrap();
        assert_eq!(v["video"]["url"].as_str(), Some(url1.as_str()), "front {who} signs the same URL: {v}");
    }
    // SSE status stream through the edge.
    let (s, v, _) = http.call("POST", &format!("{g}/minimax/h3-max/text-to-video"), Some(json!({"prompt": "a lynx"})), fal_key()).await;
    assert_eq!(s, 200, "{v}");
    let rid2 = v["request_id"].as_str().unwrap().to_owned();
    let sse = http.0.get(format!("{g}/minimax/h3-max/requests/{rid2}/status/stream")).header("authorization", fal_key().unwrap()).send().await.unwrap();
    assert_eq!(sse.status(), 200);
    let text = tokio::time::timeout(Duration::from_secs(60), sse.text()).await.unwrap().unwrap();
    assert!(text.contains("COMPLETED"), "{text}");

    // MiniMax → h3 (an API model name).
    let body = json!({"model": "MiniMax-H3-Turbo", "content": [{"type": "text", "text": "a red fox"}], "resolution": "768P", "duration": 5, "ratio": "16:9"});
    let (s, v, _) = http.call("POST", &format!("{g}/v2/video_generation"), Some(body), bearer()).await;
    assert_eq!(s, 200, "{v}");
    let tid = v["task_id"].as_str().unwrap().to_owned();
    let t = http.poll(&format!("{g}/v2/query/video_generation/{tid}"), bearer(), |v| matches!(v["task"]["status"].as_str(), Some("succeeded" | "failed"))).await;
    assert_eq!(t["task"]["status"], "succeeded", "{t}");
    assert!(e.worker_of(&tid).starts_with("worker-a"));

    // LTX v2 → ltx (worker b).
    let (s, v, _) = http.call("POST", &format!("{g}/v2/text-to-video"), Some(json!({"prompt": "a red fox", "model": "ltx-2-5-fast", "duration": 6, "resolution": "1920x1080"})), bearer()).await;
    assert_eq!(s, 202, "{v}");
    let lid = v["id"].as_str().unwrap().to_owned();
    let lv = http.poll(&format!("{g}/v2/text-to-video/{lid}"), bearer(), |v| matches!(v["status"].as_str(), Some("completed" | "failed"))).await;
    assert_eq!(lv["status"], "completed", "{lv}");
    assert_eq!(e.worker_of(&lid), "worker-b");

    // OpenAI-style → h3 (a tier alias), and its list.
    let (s, v, _) = http.call("POST", &format!("{g}/v1/videos"), Some(json!({"model": "h3-turbo", "prompt": "a cat", "seconds": "5"})), None).await;
    assert_eq!(s, 200, "{v}");
    let vid = v["id"].as_str().unwrap().to_owned();
    let vv = http.poll(&format!("{g}/v1/videos/{vid}"), None, |v| matches!(v["status"].as_str(), Some("completed" | "failed"))).await;
    assert_eq!(vv["status"], "completed", "{vv}");

    // Merged views.
    let (s, _, _) = http.call("GET", &format!("{g}/fv/v1/capabilities"), None, None).await;
    assert_eq!(s, 401, "the native API wants a key");
    let (s, caps, _) = http.call("GET", &format!("{g}/fv/v1/capabilities"), None, bearer()).await;
    assert_eq!(s, 200);
    assert_eq!(caps["auth"]["mode"], "keys", "{}", caps["auth"]);
    let ids: Vec<&str> = caps["models"].as_array().unwrap().iter().filter_map(|m| m["caps"]["id"].as_str()).collect();
    for m in ["fake-h3-turbo", "fake-h3-max", "fake-ltx-turbo", "fake-wan"] {
        assert!(ids.contains(&m), "{m} in {ids:?}");
    }
    assert_eq!(caps["readiness"], "ready");
    let (_, models, _) = http.call("GET", &format!("{g}/v1/models"), None, None).await;
    assert!(models["data"].as_array().unwrap().len() >= 4, "{models}");
    let (s, st, _) = http.call("GET", &format!("{g}/fv/v1/status"), None, None).await;
    assert_eq!(s, 200);
    let pools: Vec<&str> = st["pools"].as_array().unwrap().iter().filter_map(|p| p["id"].as_str()).collect();
    assert_eq!(pools, ["h3", "ltx", "wan"], "{st}");
    assert!(st.to_string().find(&a.base).is_none(), "no worker URLs in the public view");
    let (s, _, _) = http.call("GET", &format!("{g}/fv/v1/edge/families"), None, bearer()).await;
    assert_eq!(s, 401, "the families view needs the admin token");
    let (s, fam, _) = http.call("GET", &format!("{g}/fv/v1/edge/families"), None, Some(&format!("Bearer {ADMIN}"))).await;
    assert_eq!(s, 200);
    assert_eq!(fam["metrics"]["h3"]["workers"], 2, "{fam}");
    drop((a, a2, b));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 6)]
async fn keys_quotas_and_admission() {
    init_log();
    let e = Env::new(|c| c.quotas = Quotas { key_rpm: 40, key_in_flight: 0, invalid_key_rpm: 1000 }).await;
    // A slow front: jobs take a while, one slot, at most one queued per model.
    let a = e.front("slow", &[("fake-h3-turbo", "h3")], 200, |c| c.dispatch.max_queued = 1).await;
    let b = e.front("fast", &[("fake-wan", "wan")], 5, |_| {}).await;
    e.fronts(2).await;
    let g = e.base.clone();
    let http = &e.http;
    let admin = format!("Bearer {ADMIN}");

    // Mint at the edge: works at once on a front that never saw it.
    let (s, v, _) = http.call("POST", &format!("{g}/fv/v1/admin/keys"), Some(json!({"name": "ci"})), Some(&admin)).await;
    assert_eq!(s, 201, "{v}");
    let k = v["api_key"].as_str().unwrap().to_owned();
    let kid = v["key"]["id"].as_str().unwrap().to_owned();
    let kb = format!("Bearer {k}");
    let (s, v, _) = http.call("POST", &format!("{g}/fv/v1/jobs"), Some(json!({"model": "fake-wan", "prompt": "minted"})), Some(&kb)).await;
    assert_eq!(s, 202, "{v}");
    let (s, _, _) = http.call("DELETE", &format!("{g}/fv/v1/admin/keys/{kid}"), None, Some(&admin)).await;
    assert_eq!(s, 200);
    let (s, _, _) = http.call("POST", &format!("{g}/fv/v1/jobs"), Some(json!({"model": "fake-wan", "prompt": "revoked"})), Some(&kb)).await;
    assert_eq!(s, 401, "a revoked key is refused at once");
    // The admin token is no client key.
    let (s, _, _) = http.call("POST", &format!("{g}/fv/v1/jobs"), Some(json!({"model": "fake-wan", "prompt": "x"})), Some(&admin)).await;
    assert_eq!(s, 401);

    // Queue admission: one running, one queued, the third refused (429).
    let mut codes = Vec::new();
    let mut ids = Vec::new();
    for i in 0..3 {
        let (s, v, _) = http.call("POST", &format!("{g}/fv/v1/jobs"), Some(json!({"model": "fake-h3-turbo", "prompt": format!("slow {i}")})), bearer()).await;
        codes.push(s);
        if s == 202 {
            ids.push(v["id"].as_str().unwrap().to_owned());
        }
        tokio::time::sleep(Duration::from_millis(150)).await;
    }
    assert_eq!(codes, [202, 202, 429], "{codes:?}");

    // Cancel through the edge, from a front that does not run the job.
    let running = ids[0].clone();
    http.poll(&format!("{g}/fv/v1/jobs/{running}"), bearer(), |v| v["status"] == "running").await;
    let (s, v, _) = http.call("DELETE", &format!("{g}/fv/v1/jobs/{running}"), None, bearer()).await;
    assert!(s == 200 || s == 202, "{s} {v}");
    let c = http.poll(&format!("{g}/fv/v1/jobs/{running}"), bearer(), native_done).await;
    assert_eq!(c["status"], "cancelled", "{c}");
    let q = http.poll(&format!("{g}/fv/v1/jobs/{}", ids[1]), bearer(), native_done).await;
    assert_eq!(q["status"], "succeeded", "the queued job ran after the cancel: {q}");

    // The per-key rate limit, in each API's shape.
    let mut limited = None;
    for _ in 0..60 {
        let (s, v, _) = http.call("GET", &format!("{g}/fv/v1/jobs"), None, bearer()).await;
        if s == 429 {
            limited = Some(v);
            break;
        }
    }
    let v = limited.expect("the key's rate limit applies");
    assert_eq!(v["error"]["kind"], "rate_limited", "{v}");
    drop((a, b));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 6)]
async fn uploads_follow_their_front() {
    init_log();
    let e = Env::new(|_| {}).await;
    let a1 = e.front("u1", &[("fake-h3-turbo", "h3")], 5, |_| {}).await;
    let a2 = e.front("u2", &[("fake-h3-turbo", "h3")], 5, |_| {}).await;
    let l1 = e.front("l1", &[("fake-ltx-turbo", "ltx")], 5, |_| {}).await;
    let l2 = e.front("l2", &[("fake-ltx-turbo", "ltx")], 5, |_| {}).await;
    e.fronts(4).await;
    let g = e.base.clone();
    let http = &e.http;
    let png = {
        let mut img = image::RgbImage::new(64, 48);
        for (x, y, p) in img.enumerate_pixels_mut() {
            *p = image::Rgb([(x * 4) as u8, (y * 5) as u8, 128]);
        }
        let mut b = std::io::Cursor::new(Vec::new());
        img.write_to(&mut b, image::ImageFormat::Png).unwrap();
        b.into_inner()
    };
    // fal: initiate on some front, PUT through the edge, then four jobs
    // (whichever front takes each) read the file through the edge.
    for i in 0..4 {
        let (s, v, _) = http.call("POST", &format!("{g}/storage/upload/initiate"), Some(json!({"file_name": "a.png", "content_type": "image/png"})), fal_key()).await;
        assert_eq!(s, 200, "{v}");
        let up = v["upload_url"].as_str().unwrap().to_owned();
        let file_url = v["file_url"].as_str().unwrap().to_owned();
        assert!(up.starts_with(g.as_str()) && file_url.starts_with(g.as_str()), "{v}");
        let r = http.0.put(&up).header("content-type", "image/png").body(png.clone()).send().await.unwrap();
        assert_eq!(r.status(), 200, "the PUT reaches the issuing front");
        let (s, v, _) = http.call("POST", &format!("{g}/minimax/h3-turbo/image-to-video"), Some(json!({"prompt": format!("fox {i}"), "image_url": file_url})), fal_key()).await;
        assert_eq!(s, 200, "{v}");
        let rid = v["request_id"].as_str().unwrap().to_owned();
        let st = http.poll(&format!("{g}/minimax/h3-turbo/requests/{rid}/status"), fal_key(), |v| matches!(v["status"].as_str(), Some("COMPLETED"))).await;
        assert!(st.get("error").is_none(), "{st}");
    }
    // LTX: `ltx://uploads/<token>` resolved on another front through the edge.
    for i in 0..4 {
        let (s, v, _) = http.call("POST", &format!("{g}/v1/upload"), Some(json!({})), bearer()).await;
        assert_eq!(s, 200, "{v}");
        let up = v["upload_url"].as_str().unwrap().to_owned();
        let uri = v["storage_uri"].as_str().unwrap().to_owned();
        let r = http.0.put(&up).header("content-type", "image/png").body(png.clone()).send().await.unwrap();
        assert!(r.status().is_success(), "{}", r.status());
        let (s, v, _) = http
            .call("POST", &format!("{g}/v2/image-to-video"), Some(json!({"prompt": format!("fox {i}"), "model": "ltx-2-5-fast", "image_uri": uri, "duration": 6, "resolution": "1920x1080"})), bearer())
            .await;
        assert_eq!(s, 202, "{v}");
        let lid = v["id"].as_str().unwrap().to_owned();
        let lv = http.poll(&format!("{g}/v2/image-to-video/{lid}"), bearer(), |v| matches!(v["status"].as_str(), Some("completed" | "failed"))).await;
        assert_eq!(lv["status"], "completed", "{lv}");
    }
    drop((a1, a2, l1, l2));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 6)]
async fn a_reactor_session_is_admitted_by_the_edge() {
    init_log();
    let e = Env::new(|_| {}).await;
    // One GPU: Reactor sessions (sfwan) and batch jobs (wan) share its slot.
    let w = e.front("rt", &[("fake-sfwan", "sfwan"), ("fake-wan", "wan")], 5, |_| {}).await;
    e.fronts(1).await;
    let g = e.base.clone();
    let http = &e.http;
    let (s, v, _) = http.call("POST", &format!("{g}/start_session"), Some(json!({})), bearer()).await;
    assert_eq!(s, 200, "{v}");
    let fam = e.edge.registry();
    assert_eq!(fam.counts["sfwan"].sessions, 1, "{fam:?}");
    // While the session holds the GPU, a batch job waits.
    let (s, j, _) = http.call("POST", &format!("{g}/fv/v1/jobs"), Some(json!({"model": "fake-wan", "prompt": "after the session"})), bearer()).await;
    assert_eq!(s, 202, "{j}");
    let jid = j["id"].as_str().unwrap().to_owned();
    tokio::time::sleep(Duration::from_millis(1_500)).await;
    let (_, jv, _) = http.call("GET", &format!("{g}/fv/v1/jobs/{jid}"), None, bearer()).await;
    assert_eq!(jv["status"], "queued", "the GPU is the session's: {jv}");
    // The caller's session answers its follow-ups.
    let (s, _, _) = http.call("GET", &format!("{g}/session"), None, bearer()).await;
    assert_eq!(s, 200);
    // Another caller finds no GPU.
    let (s, v, _) = http.call("POST", &format!("{g}/start_session"), Some(json!({})), None).await;
    assert!(s == 503 || s == 429, "{s} {v}");
    // Stop: the lease ends and the job runs.
    let (s, _, _) = http.call("POST", &format!("{g}/stop_session"), Some(json!({"reason": "done"})), bearer()).await;
    assert_eq!(s, 200);
    let jv = http.poll(&format!("{g}/fv/v1/jobs/{jid}"), bearer(), native_done).await;
    assert_eq!(jv["status"], "succeeded", "{jv}");
    assert_eq!(e.edge.registry().counts["sfwan"].sessions, 0);
    drop(w);
}
