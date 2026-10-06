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
use fastvideo_engine_service::ManualClock;
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
    assert_eq!(
        (&v["status"], &v["model_loaded"], &v["state"]),
        (&json!("ok"), &json!(true), &json!("AVAILABLE"))
    );
    // Build identity (docs/serve/releases.md): the compiled-in git sha and
    // the package version, the same object on /healthz.
    assert_eq!(v["version"], env!("CARGO_PKG_VERSION"));
    assert!(v["build"]["git_sha"].as_str().is_some_and(|s| !s.is_empty()), "{v}");
    assert_eq!(v["build"], fastvideo_serve::build_info::BuildInfo::current().json());
    let (s, v, _) = call(&r, "GET", "/healthz", None, false).await;
    assert_eq!(s, 200);
    assert_eq!(v["state"], "ready");
    assert_eq!(v["build"], fastvideo_serve::build_info::BuildInfo::current().json());
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

/// A fake engine whose load and denoise steps run on a manual clock: the
/// test decides when loading and each step end.
fn manual_engine(c: &Config) -> (fastvideo_engine_service::EngineService, Arc<ManualClock>) {
    let clock = Arc::new(ManualClock::new());
    let engine = fastvideo_serve::app::build_engine_with_clock(c, Some(clock.clone())).unwrap();
    (engine, clock)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn ping_is_204_while_loading() {
    let mut c = config("loading");
    c.engine.fake.load_ms = 1500;
    // Loading cannot finish before the clock moves, however slow the host.
    let (engine, clock) = manual_engine(&c);
    let app = App::build(c, Overrides { engine: Some(engine), ..Default::default() }).await.unwrap();
    let (s, _, _) = call(&app.router, "GET", "/ping", None, false).await;
    assert_eq!(s, StatusCode::NO_CONTENT);
    let (s, v, _) = call(&app.router, "GET", "/health", None, false).await;
    assert_eq!(s, 503);
    assert_eq!(v["model_loaded"], false);
    let (s, v, h) = call(&app.router, "POST", "/fv/v1/jobs", Some(submit_body("x")), true).await;
    assert_eq!(s, 503, "{v}");
    assert_eq!(h["retry-after"], "1");
    // Let the load run: each of its slices sleeps from the time it starts.
    let c2 = clock.clone();
    let engine = app.gate.engine().clone();
    tokio::task::spawn_blocking(move || {
        while !matches!(engine.readiness(), fastvideo_engine_service::Readiness::Ready) {
            if c2.wait_for_sleepers(1, Duration::from_millis(20)) {
                c2.advance(Duration::from_millis(375));
            }
        }
    })
    .await
    .unwrap();
    wait_ready(&app).await;
    let (s, _, _) = call(&app.router, "GET", "/ping", None, false).await;
    assert_eq!(s, 200);
}

/// The binary binds before `App::build` finishes: `/ping` answers 204 at
/// once while the build is held back (a load balancer probing a booting
/// worker never hangs), other routes 503; after the build, the app serves
/// every route and drains on `stop`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn ping_answers_while_the_app_builds() {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let base = format!("http://{}", listener.local_addr().unwrap());
    let (go_tx, go_rx) = tokio::sync::oneshot::channel::<()>();
    let (stop_tx, stop_rx) = tokio::sync::oneshot::channel::<()>();
    let c = config("booting");
    let build = async move {
        let _ = go_rx.await;
        App::build(c, Overrides::default()).await
    };
    let server = tokio::spawn(fastvideo_serve::app::serve_while_building(listener, build, async {
        let _ = stop_rx.await;
    }));
    let http = reqwest::Client::new();
    let get = |p: &str| {
        let req = http.get(format!("{base}{p}")).timeout(Duration::from_secs(5));
        async move { req.send().await.unwrap() }
    };
    assert_eq!(get("/ping").await.status(), 204);
    let r = get("/health").await;
    assert_eq!(r.status(), 503);
    assert_eq!(r.json::<Value>().await.unwrap()["model_loaded"], false);
    let r = get("/fv/v1/capabilities").await;
    assert_eq!(r.status(), 503);
    assert!(r.headers().get("retry-after").is_some());

    go_tx.send(()).unwrap();
    let mut ready = false;
    for _ in 0..400 {
        if get("/ping").await.status() == 200 {
            ready = true;
            break;
        }
        tokio::time::sleep(Duration::from_millis(25)).await;
    }
    assert!(ready, "/ping never reached 200 after the build");
    let r = http.get(format!("{base}/fv/v1/capabilities")).bearer_auth(KEY).send().await.unwrap();
    assert_eq!(r.status(), 200);
    stop_tx.send(()).unwrap();
    tokio::time::timeout(Duration::from_secs(30), server).await.unwrap().unwrap().unwrap();
}

/// A failed build stops the early listener and returns the error.
#[tokio::test]
async fn a_failed_build_stops_the_early_listener() {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let build = async { Err::<App, _>(anyhow::anyhow!("no engine")) };
    let r = fastvideo_serve::app::serve_while_building(listener, build, std::future::pending()).await;
    assert!(r.is_err());
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

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn drain_cancels_queued_and_refuses_new_work() {
    let mut c = config("drain");
    c.engine.fake.step_ms = 300;
    // `a` stays in its first denoise step until the clock moves, so the
    // drain always finds it running and `b` queued.
    let (engine, clock) = manual_engine(&c);
    let app = App::build(c, Overrides { engine: Some(engine), ..Default::default() }).await.unwrap();
    wait_ready(&app).await;
    let r = app.router.clone();
    let (_, a, _) = call(&r, "POST", "/fv/v1/jobs", Some(submit_body("long one")), true).await;
    let (_, b, _) = call(&r, "POST", "/fv/v1/jobs", Some(submit_body("queued one")), true).await;
    let (a, b) = (a["id"].as_str().unwrap().to_owned(), b["id"].as_str().unwrap().to_owned());
    wait_status(&r, &a, &["running"]).await;
    let c2 = clock.clone();
    assert!(tokio::task::spawn_blocking(move || c2.wait_for_sleepers(1, Duration::from_secs(60))).await.unwrap());
    // `a` is parked in its first step and virtual time stands still for
    // the first second of the drain (ten times its 100 ms grace), so the
    // drain always cancels a running job; then time moves so the step ends
    // and the executor sees the cancellation.
    let tick = {
        let clock = clock.clone();
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_secs(1)).await;
            loop {
                clock.advance(Duration::from_millis(50));
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
    };
    fastvideo_serve::app::drain(&app.gate, Duration::from_millis(100), None).await;
    let (s, _, _) = call(&r, "GET", "/ping", None, false).await;
    assert_eq!(s, 503, "draining");
    let (s, _, _) = call(&r, "POST", "/fv/v1/jobs", Some(submit_body("late")), true).await;
    assert_eq!(s, 503);
    let vb = wait_status(&r, &b, &["cancelled"]).await;
    assert_eq!(vb["status"], "cancelled");
    let va = wait_status(&r, &a, &["cancelled", "succeeded"]).await;
    assert_eq!(va["status"], "cancelled", "the running job was cut after the grace period");
    tick.abort();
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
        let mut c = Config::from_toml(&text, &p.display().to_string()).unwrap();
        // The gateway's secrets and pool endpoints come from the environment.
        if c.engine.backend == fastvideo_serve::config::EngineBackendKind::Remote {
            let mut env: BTreeMap<String, String> = [
                ("FV_INTERNAL_TOKEN", "t"),
                ("FV_CF_ACCOUNT_ID", "a"),
                ("FV_CF_API_TOKEN", "t"),
                ("FV_D1_DATABASE_ID", "d"),
            ]
            .into_iter()
            .map(|(k, v)| (k.to_owned(), v.to_owned()))
            .collect();
            for pool in &c.pools {
                env.insert(format!("{}ENDPOINT", pool.env_prefix()), "ep".into());
            }
            c.apply_env(&env).unwrap();
        }
        c.validate().unwrap_or_else(|e| panic!("{}: {e}", p.display()));
        // CUDA configs: every `[[models]]` entry resolves against the
        // catalog and the set shares one process.
        if c.engine.backend == fastvideo_serve::config::EngineBackendKind::Cuda {
            let models = fastvideo_serve::app::cuda_models(&c).unwrap_or_else(|e| panic!("{}: {e}", p.display()));
            assert!(!models.is_empty(), "{}", p.display());
        }
        n += 1;
    }
    assert!(n >= 2, "configs/serve has {n} configs");
}

/// The Google Cloud configs (`configs/serve/gcp-*.toml`, scripts/gcp/families.tsv,
/// docs/serve/deploy-gcp.md): each parses under today's schema, serves the same
/// `[[models]]`, aliases and `[protocols]` as its Runpod twin, and validates
/// standalone, as a family Durable Object worker (the env scripts/gcp/vm.sh
/// `worker` sets) and as a gateway-less direct worker.
#[test]
fn gcp_configs_mirror_runpod_and_join_family_dos() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let dir = root.join("configs/serve");
    let table = std::fs::read_to_string(root.join("scripts/gcp/families.tsv")).unwrap();
    let load = |f: &str| Config::from_toml(&std::fs::read_to_string(dir.join(f)).unwrap(), f).unwrap_or_else(|e| panic!("{f}: {e}"));
    let env_of = |kv: &[(&str, &str)]| -> BTreeMap<String, String> { kv.iter().map(|(k, v)| ((*k).to_owned(), (*v).to_owned())).collect() };
    // Dispatch family of a model family (docs/serve/dispatch-do-family.md §4).
    let do_family = |model_family: &str| match model_family {
        "ltx2" => "ltx".to_owned(),
        f => f.to_owned(),
    };
    let mut listed = Vec::new();
    for line in table.lines().filter(|l| !l.starts_with('#') && !l.trim().is_empty()) {
        let cols: Vec<&str> = line.split('\t').collect();
        assert_eq!(cols.len(), 11, "families.tsv: {line}");
        let (family, cfg, twin, dfam) = (cols[0], cols[1], cols[2], cols[3]);
        if !cfg.starts_with("gcp-") {
            assert!(dir.join(cfg).exists(), "{family}: {cfg}");
            continue;
        }
        listed.push(cfg.to_owned());
        let c = load(cfg);
        c.validate().unwrap_or_else(|e| panic!("{cfg} (standalone): {e}"));
        assert_eq!(c.engine.backend, fastvideo_serve::config::EngineBackendKind::Cuda, "{cfg}");
        assert!(!fastvideo_serve::app::cuda_models(&c).unwrap_or_else(|e| panic!("{cfg}: {e}")).is_empty(), "{cfg}");
        assert_eq!(c.server.state_dir, std::path::PathBuf::from("/fvstate"), "{cfg}: state on the boot disk, never the read-only weight disk");
        assert_eq!((c.webrtc.udp_port, c.webrtc.tcp_port), (40010, 40000), "{cfg}: real ports (the VM's firewall rule opens them)");
        assert_eq!(c.director.chunk_seconds, 5.0, "{cfg}: director chunk");
        assert!(c.gateway.pool.is_none(), "{cfg}: a GCP worker joins family objects, not a pool socket");
        // The Runpod twin serves the same thing.
        let r = load(twin);
        assert_eq!(serde_json::to_value(&c.models).unwrap(), serde_json::to_value(&r.models).unwrap(), "{cfg} vs {twin}: [[models]]");
        assert_eq!(c.aliases, r.aliases, "{cfg} vs {twin}: [aliases]");
        assert_eq!(serde_json::to_value(&c.protocols).unwrap(), serde_json::to_value(&r.protocols).unwrap(), "{cfg} vs {twin}: [protocols]");
        // The family object the table names is the models' family.
        for m in &c.models {
            assert_eq!(do_family(&m.family), dfam, "{cfg}: model {} (family {}) vs dispatch family {dfam}", m.id, m.family);
        }
        // A family Durable Object worker (vm.sh worker).
        let worker = [
            ("FV_SERVE_ROLE", "worker"),
            ("FV_INTERNAL_TOKEN", "t"),
            ("FV_DISPATCH_DO_URL", "https://fv-edge.example.workers.dev"),
            ("FV_DISPATCH_FAMILIES", dfam),
            ("FV_DISPATCH_DIRECT_UPLOAD", "1"),
            ("FV_DISPATCH_CAPACITY", "2"),
            ("FV_DISPATCH_SESSIONS", "1"),
            ("FV_JOBS_HEARTBEAT_S", "10"),
            ("FV_WEIGHTS", "/workspace/weights"),
            ("FV_PUBLIC_BASE_URL", "https://203-0-113-9.sslip.io"),
        ];
        let mut w = load(cfg);
        w.apply_env(&env_of(&worker)).unwrap();
        w.validate().unwrap_or_else(|e| panic!("{cfg} (family DO worker): {e}"));
        assert_eq!(w.server.role, fastvideo_serve::config::Role::Worker);
        assert_eq!(w.dispatch.families, vec![dfam.to_owned()], "{cfg}");
        assert!(w.dispatch.direct_upload && w.dispatch.do_url.is_some(), "{cfg}");
        // Gateway-less: the workers check client keys themselves (D1 key store).
        let mut direct = worker.to_vec();
        direct.extend([
            ("FV_WORKER_DIRECT", "1"),
            ("FV_AUTH_MODE", "keys"),
            ("FV_KEY_STORE", "d1"),
            ("FV_ADMIN_TOKEN", "fvadm_test"),
            ("FV_CF_ACCOUNT_ID", "a"),
            ("FV_CF_API_TOKEN", "t"),
            ("FV_D1_DATABASE_ID", "d"),
        ]);
        let mut d = load(cfg);
        d.apply_env(&env_of(&direct)).unwrap();
        d.validate().unwrap_or_else(|e| panic!("{cfg} (direct worker): {e}"));
        assert!(d.gateway.direct && d.auth.key_store == fastvideo_serve::config::KeyStoreBackend::D1, "{cfg}");
    }
    // Every gcp-*.toml is in the table (vm.sh can deploy it).
    for e in std::fs::read_dir(&dir).unwrap() {
        let n = e.unwrap().file_name().to_string_lossy().into_owned();
        if n.starts_with("gcp-") && n.ends_with(".toml") {
            assert!(listed.contains(&n), "{n} is not in scripts/gcp/families.tsv");
        }
    }
    assert_eq!(listed.len(), 4, "{listed:?}");
}

/// Runpod load balancer with `workers.max > 1` and no shared state (file
/// jobs, local artifacts): only routes any worker can answer are served.
#[tokio::test]
async fn multi_worker_without_shared_state_serves_only_local_routes() {
    let mut c = config("lb");
    c.server.workers_max = 2;
    c.validate().unwrap();
    let app = App::build(c, Overrides::default()).await.unwrap();
    wait_ready(&app).await;
    let r = app.router.clone();
    for (m, uri) in [("GET", "/health"), ("GET", "/ping"), ("GET", "/healthz"), ("GET", "/fv/v1/capabilities"), ("GET", "/v1/models")] {
        let (s, v, _) = call(&r, m, uri, None, true).await;
        assert_eq!(s, 200, "{m} {uri}: {v}");
    }
    for (m, uri, body) in [
        ("POST", "/fv/v1/jobs", Some(submit_body("x"))),
        ("GET", "/fv/v1/jobs", None),
        ("POST", "/v1/videos/sync", Some(json!({"prompt": "x"}))),
        ("POST", "/v1/videos", Some(json!({"prompt": "x"}))),
        ("DELETE", "/fv/v1/jobs/fvjob_x", None),
        ("GET", "/files/a/b.mp4", None),
        ("PUT", "/uploads/t", Some(json!({}))),
        ("POST", "/fv/v1/streams", Some(json!({}))),
        ("POST", "/start_session", Some(json!({}))),
        ("GET", "/fv/v1/admin/keys", None),
        ("GET", "/not-a-route", None),
    ] {
        let (s, v, h) = call(&r, m, uri, body, true).await;
        assert_eq!(s, StatusCode::NOT_FOUND, "{m} {uri}: {v}");
        assert_eq!(h["x-fv-multi-worker"], "not-served", "{m} {uri}");
        assert_eq!(v["error"]["type"], "not_served_by_multi_worker_deployment", "{m} {uri}");
    }
    // CORS preflight passes the filter.
    let resp = r
        .clone()
        .oneshot(Request::builder().method("OPTIONS").uri("/fv/v1/jobs").body(Body::empty()).unwrap())
        .await
        .unwrap();
    assert!(resp.headers().get("x-fv-multi-worker").is_none());
}

/// Two workers sharing D1 (and, in production, R2): a job submitted on one
/// is read, listed and polled to completion on the other; cancel/delete is
/// refused because the running job is authoritative on its own worker.
#[tokio::test]
async fn multi_worker_with_shared_jobs_reads_across_workers() {
    use fastvideo_serve::multiworker::{layer, Policy};
    let mock = MockD1::new();
    let db = || D1Client::new(Arc::new(mock.clone()));
    let policy = Policy { workers_max: 2, jobs: true, artifacts: true, keys: false, gateway: false };
    let mut workers = Vec::new();
    for (tag, id) in [("lb-a", "worker-a"), ("lb-b", "worker-b")] {
        let store = D1JobStore::new(db(), D1Options::new(id)).open(time::OffsetDateTime::now_utc()).await.unwrap();
        let c = config(tag);
        let apps = c.protocols.fal_apps.clone();
        let app = App::build(c, Overrides { jobs: Some(store), ..Default::default() }).await.unwrap();
        wait_ready(&app).await;
        let router = layer(app.router.clone(), policy, &apps);
        workers.push((app, router));
    }
    let (a, b) = (&workers[0].1, &workers[1].1);
    let (s, v, _) = call(a, "POST", "/fv/v1/jobs", Some(submit_body("lb job")), true).await;
    assert_eq!(s, 202, "{v}");
    let id = v["id"].as_str().unwrap().to_owned();
    // Worker B answers status and list from D1.
    let done = wait_status(b, &id, &["succeeded"]).await;
    assert_eq!(done["id"], id.as_str());
    let (s, v, _) = call(b, "GET", "/fv/v1/jobs", None, true).await;
    assert_eq!((s, v["total"].clone()), (StatusCode::OK, json!(1)), "{v}");
    let (s, _, h) = call(b, "DELETE", &format!("/fv/v1/jobs/{id}"), None, true).await;
    assert_eq!((s, &h["x-fv-multi-worker"]), (StatusCode::NOT_FOUND, &"not-served".parse::<axum::http::HeaderValue>().unwrap()));
    let (s, _, _) = call(b, "GET", "/fv/v1/admin/keys", None, true).await;
    assert_eq!(s, StatusCode::NOT_FOUND, "minted keys are not shared here");
}
