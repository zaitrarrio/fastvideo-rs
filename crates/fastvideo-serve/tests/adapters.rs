//! End-to-end through the adapters on main (FastVideo `/v1/videos` +
//! FastWan, MiniMax V2, fal queue) mounted in the assembled fv-serve router,
//! over the real `EngineService` with the fake backend and the shared engine
//! gate. LTX has its own suite (`tests/ltx.rs`).

#![cfg(all(feature = "openai-videos", feature = "minimax", feature = "fal"))]

use std::collections::BTreeMap;
use std::time::Duration;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use axum::Router;
use fastvideo_protocol::ProtocolId;
use fastvideo_serve::config::{Config, JobBackend};
use fastvideo_serve::router::{route_table, Owner};
use fastvideo_serve::{App, Overrides};
use fastvideo_serve_kit::KeyRing;
use serde_json::{json, Value};
use tower::ServiceExt;

const KEY: &str = "sk-adapters";

async fn app(tag: &str) -> App {
    let dir = std::env::temp_dir().join(format!(
        "fv-serve-adapters-{tag}-{:x}",
        std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()
    ));
    let mut env = BTreeMap::new();
    env.insert("FV_API_KEYS".to_owned(), KeyRing::hash_hex(KEY));
    env.insert("FV_URL_SIGNING_KEY".to_owned(), "k".to_owned());
    env.insert("FV_STATE_DIR".to_owned(), dir.display().to_string());
    env.insert("FV_PUBLIC_BASE_URL".to_owned(), "http://fv.test".to_owned());
    let mut c = Config::default();
    c.apply_env(&env).unwrap();
    c.jobs.backend = JobBackend::Memory;
    c.engine.fake.step_ms = 1;
    c.protocols.fastwan = true;
    c.validate().unwrap();
    let a = App::build(c, Overrides::default()).await.unwrap();
    a.gate.engine().wait_ready().await;
    a
}

struct Resp {
    status: StatusCode,
    headers: axum::http::HeaderMap,
    bytes: Vec<u8>,
}

impl Resp {
    fn json(&self) -> Value {
        serde_json::from_slice(&self.bytes).unwrap_or(Value::Null)
    }
}

async fn call(app: &Router, method: &str, uri: &str, body: Option<Value>, auth: Option<&str>) -> Resp {
    let mut b = Request::builder().method(method).uri(uri);
    if let Some(a) = auth {
        b = b.header("authorization", a);
    }
    let req = match body {
        Some(v) => b.header("content-type", "application/json").body(Body::from(v.to_string())).unwrap(),
        None => b.body(Body::empty()).unwrap(),
    };
    let r = app.clone().oneshot(req).await.unwrap();
    let (status, headers) = (r.status(), r.headers().clone());
    let bytes = axum::body::to_bytes(r.into_body(), 1 << 26).await.unwrap().to_vec();
    Resp { status, headers, bytes }
}

fn bearer() -> Option<&'static str> {
    Some("Bearer sk-adapters")
}

async fn poll(app: &Router, uri: &str, auth: Option<&str>, done: impl Fn(&Value) -> bool) -> Value {
    for _ in 0..4000 {
        let r = call(app, "GET", uri, None, auth).await;
        let v = r.json();
        if r.status == 200 && done(&v) {
            return v;
        }
        tokio::time::sleep(Duration::from_millis(25)).await;
    }
    panic!("{uri} never finished");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn fastvideo_videos_and_fastwan() {
    let a = app("ov").await;
    let r = a.router.clone();
    let s = call(&r, "POST", "/v1/videos", Some(json!({"model": "h3-turbo", "prompt": "a cat", "seconds": "5"})), None).await;
    assert_eq!(s.status, 200, "{}", String::from_utf8_lossy(&s.bytes));
    let id = s.json()["id"].as_str().unwrap().to_owned();
    let v = poll(&r, &format!("/v1/videos/{id}"), None, |v| matches!(v["status"].as_str(), Some("completed" | "failed"))).await;
    assert_eq!(v["status"], "completed", "{v}");
    let c = call(&r, "GET", &format!("/v1/videos/{id}/content"), None, None).await;
    assert_eq!(c.status, 200);
    assert_eq!(c.headers["content-type"], "video/mp4");
    let m = call(&r, "GET", "/v1/models", None, None).await;
    assert_eq!(m.json()["object"], "list");

    // FastWan: /generate -> /status -> /video; `/` names its model and
    // `/health` carries model_loaded.
    let root = call(&r, "GET", "/", None, None).await.json();
    assert_eq!(root["model"], "fake-wan");
    let h = call(&r, "GET", "/health", None, None).await;
    assert_eq!((h.status, h.json()["model_loaded"].as_bool()), (StatusCode::OK, Some(true)));
    let g = call(&r, "POST", "/generate", Some(json!({"prompt": "waves", "width": 832, "height": 480, "num_frames": 49})), None).await;
    assert_eq!(g.status, 200, "{}", String::from_utf8_lossy(&g.bytes));
    let pid = g.json()["prompt_id"].as_str().unwrap().to_owned();
    let st = poll(&r, &format!("/status/{pid}"), None, |v| matches!(v["status"].as_str(), Some("completed" | "failed"))).await;
    assert_eq!(st["status"], "completed", "{st}");
    let vid = call(&r, "GET", &format!("/video/{pid}"), None, None).await;
    assert_eq!(vid.status, 200);
    assert!(!vid.bytes.is_empty());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn minimax_create_query_list() {
    let a = app("mm").await;
    let r = a.router.clone();
    let body = json!({"model": "MiniMax-H3-Turbo", "content": [{"type": "text", "text": "a red fox"}],
        "resolution": "768P", "duration": 5, "ratio": "16:9"});
    let s = call(&r, "POST", "/v2/video_generation", Some(body), bearer()).await;
    assert_eq!(s.status, 200, "{}", String::from_utf8_lossy(&s.bytes));
    let tid = s.json()["task_id"].as_str().unwrap().to_owned();
    assert_eq!(tid.len(), 18);
    let t = poll(&r, &format!("/v2/query/video_generation/{tid}"), bearer(), |v| {
        matches!(v["task"]["status"].as_str(), Some("succeeded" | "failed"))
    })
    .await;
    assert_eq!(t["task"]["status"], "succeeded", "{t}");
    let url = t["task"]["content"]["url"].as_str().unwrap();
    let f = call(&r, "GET", url.strip_prefix("http://fv.test").unwrap(), None, None).await;
    assert_eq!(f.status, 200);
    let l = call(&r, "GET", "/v2/query/video_generation?page_num=1&page_size=10", None, bearer()).await;
    assert_eq!(l.status, 200);
    assert_eq!(l.json()["total"], 1);
    let s = call(&r, "POST", "/v2/video_generation", Some(json!({})), None).await;
    assert_eq!(s.status, 401);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn fal_queue_submit_status_result() {
    let a = app("fal").await;
    let r = a.router.clone();
    let key = Some("Key sk-adapters");
    let s = call(&r, "POST", "/minimax/h3-turbo/text-to-video", Some(json!({"prompt": "a kitten", "seed": 7})), key).await;
    assert_eq!(s.status, 200, "{}", String::from_utf8_lossy(&s.bytes));
    let v = s.json();
    let rid = v["request_id"].as_str().unwrap().to_owned();
    let status_path = format!("/minimax/h3-turbo/requests/{rid}/status?logs=1");
    let done = poll(&r, &status_path, key, |v| v["status"] == "COMPLETED").await;
    assert!(done.get("error").is_none(), "{done}");
    let out = call(&r, "GET", &format!("/minimax/h3-turbo/requests/{rid}"), None, key).await;
    assert_eq!(out.status, 200);
    let o = out.json();
    let url = o["video"]["url"].as_str().unwrap();
    // Named like hosted fal output (`<nanoid21>_minimax-h3.mp4`).
    assert!(url.split('?').next().unwrap().ends_with("_minimax-h3.mp4"), "{url}");
    let job = a.ctx.jobs().by_external(ProtocolId::Fal, &rid).await.unwrap();
    assert_eq!(job.artifacts[0].file_name, fastvideo_fal::output_file_name(&job));
    // JWKS for our webhook signatures.
    let j = call(&r, "GET", "/.well-known/jwks.json", None, None).await;
    assert_eq!(j.status, 200);
    assert!(j.json()["keys"].as_array().is_some_and(|k| !k.is_empty()));
    // Unknown apps are not routed.
    let s = call(&r, "POST", "/minimax/h9/text-to-video", Some(json!({"prompt": "x"})), key).await;
    assert_eq!(s.status, 404);
}

/// Every route of the §9 table for a mounted owner exists on the assembled
/// router (handler answers, not the router's empty 404 or a 405).
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn every_mounted_route_of_the_table_answers() {
    let a = app("table").await;
    let apps = a.config.protocols.fal_apps.clone();
    for spec in route_table(&apps) {
        if matches!(spec.owner, Owner::Reactor | Owner::FalDirector) {
            continue;
        }
        let mut uri = String::new();
        let mut in_param = false;
        for ch in spec.path.chars() {
            match ch {
                '{' => in_param = true,
                '}' => {
                    in_param = false;
                    uri.push('x');
                }
                c if !in_param => uri.push(c),
                _ => {}
            }
        }
        let body = matches!(spec.method, "POST" | "PUT").then(|| json!({}));
        let s = call(&a.router, spec.method, &uri, body, bearer()).await;
        assert_ne!(s.status, StatusCode::METHOD_NOT_ALLOWED, "{} {uri} ({:?})", spec.method, spec.owner);
        assert!(
            !(s.status == 404 && s.bytes.is_empty()),
            "{} {uri} ({:?}) is not routed",
            spec.method,
            spec.owner
        );
    }
}
