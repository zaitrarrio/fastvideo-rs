//! fv-serve in `runpod-queue` mode (WP-16): the Rust Runpod worker taking
//! jobs from the local queue simulator and dispatching them into the real
//! fv-serve router over the fake engine; plus the Vast forwarder route.

use std::collections::BTreeMap;
use std::time::Duration;

use axum::body::Body;
use axum::http::Request;
use fastvideo_deploy::runpod::sim::{Sim, SimStatus};
use fastvideo_deploy::runpod::{RouterTransport, RunpodEnv, Worker, WorkerOptions};
use fastvideo_serve::config::{Config, JobBackend};
use fastvideo_serve::deploy::{handler, Boot};
use fastvideo_serve::{App, Overrides};
use fastvideo_serve_kit::KeyRing;
use serde_json::{json, Value};
use tower::ServiceExt;

const KEY: &str = "sk-test-queue";

fn config(forward: bool) -> Config {
    let dir = std::env::temp_dir().join(format!(
        "fv-serve-queue-{}",
        std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()
    ));
    let mut env = BTreeMap::new();
    env.insert("FV_API_KEYS".to_owned(), KeyRing::hash_hex(KEY));
    env.insert("FV_STATE_DIR".to_owned(), dir.display().to_string());
    env.insert("FV_PUBLIC_BASE_URL".to_owned(), "http://fv.test".to_owned());
    env.insert("FV_SERVE_MODE".to_owned(), "runpod-queue".to_owned());
    if forward {
        env.insert("FV_SERVE_FORWARD".to_owned(), "1".to_owned());
    }
    let mut c = Config::default();
    c.apply_env(&env).unwrap();
    c.jobs.backend = JobBackend::Memory;
    c.engine.fake.step_ms = 1;
    c.validate().unwrap();
    c
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn queue_worker_runs_native_jobs_on_the_fake_engine() {
    let app = App::build(config(false), Overrides::default()).await.unwrap();
    app.gate.engine().wait_ready().await;
    let sim = Sim::new("rk").with_long_poll(Duration::from_millis(20));
    let vars: BTreeMap<String, String> = sim.env("http://sim.local", "w1", 100).into_iter().collect();
    let env = RunpodEnv::from_lookup(|k| vars.get(k).cloned()).unwrap();
    let ready = std::sync::Arc::new(std::sync::OnceLock::new());
    let h = handler(&app, Boot::now(), ready);
    let opts = WorkerOptions { idle_backoff: Duration::from_millis(5), ..WorkerOptions::default() };
    let (tx, rx) = tokio::sync::oneshot::channel::<()>();
    let w = tokio::spawn(Worker::new(env, RouterTransport(sim.router()), h, opts).run(async move {
        let _ = rx.await;
    }));
    let auth = json!({"authorization": format!("Bearer {KEY}")});

    let caps = sim.submit(json!({"kind": "http", "method": "GET", "path": "/fv/v1/capabilities", "headers": auth}));
    let job = sim.submit(json!({"kind": "http", "method": "POST", "path": "/fv/v1/jobs", "headers": auth,
        "body": {"model": "fake-wan", "prompt": "a red fox in fresh snow", "seed": 1}, "wait": true}));
    let unauth = sim.submit(json!({"kind": "http", "method": "GET", "path": "/fv/v1/capabilities"}));
    let info = sim.submit(json!({"kind": "info"}));
    let stream = sim.submit(json!({"kind": "stream", "model": "fake-sfwan", "prompt": "p", "whip_url": "https://whip.example/x"}));

    let t = Duration::from_secs(30);
    let c = sim.wait_terminal(&caps, t).await.unwrap();
    assert_eq!(c.status, SimStatus::Completed, "{c:?}");
    let out = c.output.unwrap();
    assert_eq!(out["status"], 200);
    assert!(out["body"]["models"].as_array().unwrap().len() > 1);

    let j = sim.wait_terminal(&job, t).await.unwrap();
    assert_eq!(j.status, SimStatus::Completed, "{j:?}");
    let out = j.output.unwrap();
    assert_eq!(out["status"], 200, "{out}");
    assert_eq!(out["body"]["status"], "succeeded", "{out}");
    assert!(out["poll_path"].as_str().unwrap().starts_with("/fv/v1/jobs/"));
    assert!(!j.progress.is_empty());

    let u = sim.wait_terminal(&unauth, t).await.unwrap();
    assert_eq!(u.output.unwrap()["status"], 401);

    let i = sim.wait_terminal(&info, t).await.unwrap();
    let out = i.output.unwrap();
    assert_eq!(out["server"], "fv-serve");
    assert_eq!(out["readiness"], "ready");
    assert!(out["models"].as_array().unwrap().iter().any(|m| m == "fake-wan"));

    // Native WHIP streams are not in this build: the job fails with the reason.
    let s = sim.wait_terminal(&stream, t).await.unwrap();
    assert_eq!(s.status, SimStatus::Failed);
    assert_eq!(s.error_report().unwrap()["error_type"], "StreamRefused");

    let _ = tx.send(());
    let r = w.await.unwrap();
    assert_eq!(r.taken, 5);
    assert_eq!(r.undelivered, 0);
}

#[tokio::test]
async fn forward_route_dispatches_envelopes() {
    let app = App::build(config(true), Overrides::default()).await.unwrap();
    app.gate.engine().wait_ready().await;
    let call = |v: Value| {
        let r = app.router.clone();
        async move {
            let req = Request::builder()
                .method("POST")
                .uri("/fv/v1/forward")
                .header("content-type", "application/json")
                .body(Body::from(v.to_string()))
                .unwrap();
            let resp = r.oneshot(req).await.unwrap();
            let s = resp.status().as_u16();
            let b = axum::body::to_bytes(resp.into_body(), 1 << 22).await.unwrap();
            (s, serde_json::from_slice::<Value>(&b).unwrap())
        }
    };
    let (s, v) = call(json!({"kind": "http", "method": "GET", "path": "/fv/v1/capabilities",
        "headers": {"authorization": format!("Bearer {KEY}")}}))
    .await;
    assert_eq!(s, 200);
    assert_eq!(v["status"], 200);
    let (s, _) = call(json!({"kind": "stream", "model": "m", "whip_url": "https://x"})).await;
    assert_eq!(s, 400);
    let (s, _) = call(json!({"kind": "http", "path": "/fv/v1/forward"})).await;
    assert_eq!(s, 400);
    let (s, v) = call(json!({"kind": "info"})).await;
    assert_eq!(s, 200);
    assert_eq!(v["server"], "fv-serve");
}
