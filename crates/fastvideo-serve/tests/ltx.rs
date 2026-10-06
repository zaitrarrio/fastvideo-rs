//! End-to-end through the LTX adapter (`fastvideo-ltxapi`, on main) mounted
//! in the assembled fv-serve router, over the real `EngineService` with the
//! fake backend: v2 async jobs, v1 sync bytes, uploads, the 403 stubs, the
//! pad-and-crop geometry and `generate_audio: false` on the stored artifact,
//! and (with `http-client`) v1 sync answered from an S3/R2 artifact store.

#![cfg(feature = "ltxapi")]

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

const KEY: &str = "sk-ltx";

fn config(tag: &str) -> Config {
    let dir = tempfile::Builder::new().prefix(&format!("fv-serve-ltx-{tag}-")).tempdir().unwrap().keep();
    let mut env = BTreeMap::new();
    env.insert("FV_API_KEYS".to_owned(), KeyRing::hash_hex(KEY));
    env.insert("FV_URL_SIGNING_KEY".to_owned(), "k".to_owned());
    env.insert("FV_STATE_DIR".to_owned(), dir.display().to_string());
    env.insert("FV_PUBLIC_BASE_URL".to_owned(), "http://fv.test".to_owned());
    let mut c = Config::default();
    c.apply_env(&env).unwrap();
    c.jobs.backend = JobBackend::Memory;
    c.engine.fake.step_ms = 1;
    c
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

async fn call(app: &Router, method: &str, uri: &str, body: Option<Value>, auth: bool) -> Resp {
    let mut b = Request::builder().method(method).uri(uri);
    if auth {
        b = b.header("authorization", format!("Bearer {KEY}"));
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

fn t2v(model: &str, res: &str) -> Value {
    json!({"prompt": "a red fox in the snow", "model": model, "duration": 6, "resolution": res})
}

async fn poll(app: &Router, id: &str) -> Value {
    for _ in 0..4000 {
        let r = call(app, "GET", &format!("/v2/text-to-video/{id}"), None, true).await;
        assert_eq!(r.status, 200);
        let v = r.json();
        if matches!(v["status"].as_str(), Some("completed" | "failed")) {
            return v;
        }
        tokio::time::sleep(Duration::from_millis(25)).await;
    }
    panic!("LTX job {id} did not finish");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn ltx_v2_v1_upload_and_stubs() {
    let app = App::build(config("main"), Overrides::default()).await.unwrap();
    app.gate.engine().wait_ready().await;
    let r = app.router.clone();

    // v2 async: 202 {id, created_at}, pad-and-crop, completed with a URL.
    let s = call(&r, "POST", "/v2/text-to-video", Some(t2v("ltx-2-5-fast", "1280x720")), true).await;
    assert_eq!(s.status, 202, "{}", String::from_utf8_lossy(&s.bytes));
    let id = s.json()["id"].as_str().unwrap().to_owned();
    assert_eq!(s.headers["x-request-id"].len(), 32);
    let job = app.ctx.jobs().by_external(ProtocolId::LtxV2, &id).await.unwrap();
    assert_eq!((job.resolved.width, job.resolved.height), (1280, 768));
    assert_eq!(job.resolved.post.crop, Some((1280, 720)));
    let v = poll(&r, &id).await;
    assert_eq!(v["status"], "completed", "{v}");
    let url = v["result"]["video_url"].as_str().unwrap();
    let file = call(&r, "GET", url.strip_prefix("http://fv.test").unwrap(), None, false).await;
    assert_eq!(file.status, 200);
    // The stored artifact reports the cropped canvas.
    let job = app.ctx.jobs().by_external(ProtocolId::LtxV2, &id).await.unwrap();
    let a = &job.artifacts[0];
    assert_eq!((a.width, a.height), (1280, 720));
    assert!(a.audio.is_some());

    // generate_audio: false -> the artifact carries no audio.
    let mut b = t2v("ltx-2-5-fast", "1280x720");
    b["generate_audio"] = json!(false);
    let s = call(&r, "POST", "/v2/text-to-video", Some(b), true).await;
    assert_eq!(s.status, 202, "{}", String::from_utf8_lossy(&s.bytes));
    let silent = s.json()["id"].as_str().unwrap().to_owned();
    assert_eq!(poll(&r, &silent).await["status"], "completed");
    let job = app.ctx.jobs().by_external(ProtocolId::LtxV2, &silent).await.unwrap();
    assert!(job.resolved.post.drop_audio);
    assert_eq!(job.artifacts[0].audio, None);

    // v1 sync: the MP4 bytes on the same request.
    let s = call(&r, "POST", "/v1/text-to-video", Some(t2v("ltx-turbo", "1280x720")), true).await;
    assert_eq!(s.status, 200, "{}", String::from_utf8_lossy(&s.bytes));
    assert_eq!(s.headers["content-type"], "video/mp4");
    assert!(!s.bytes.is_empty());

    // Upload ticket then PUT through serve-kit's shared route.
    let s = call(&r, "POST", "/v1/upload", None, true).await;
    assert_eq!(s.status, 200);
    let up = s.json();
    assert!(up["storage_uri"].as_str().unwrap().starts_with("ltx://uploads/"));
    let put_path = up["upload_url"].as_str().unwrap().strip_prefix("http://fv.test").unwrap().to_owned();
    let req = Request::put(&put_path).header("content-type", "image/png").body(Body::from(vec![1u8, 2, 3])).unwrap();
    assert_eq!(r.clone().oneshot(req).await.unwrap().status(), 200);

    // 403 stubs, 400 for an edit without its video, 401 without a key, 404
    // for another endpoint's job.
    let s = call(&r, "POST", "/v2/video-to-video-hdr", Some(json!({})), true).await;
    assert_eq!(s.status, 403);
    assert_eq!(s.json()["error"]["type"], "permission_error");
    let s = call(&r, "POST", "/v2/retake", Some(json!({})), true).await;
    assert_eq!(s.status, 400);
    assert_eq!(s.json()["error"]["type"], "invalid_request_error");
    let s = call(&r, "POST", "/v2/text-to-video", Some(t2v("ltx-turbo", "1280x720")), false).await;
    assert_eq!(s.status, 401);
    let s = call(&r, "GET", &format!("/v2/image-to-video/{id}"), None, true).await;
    assert_eq!(s.status, 404);

    // Every LTX route of the §9 table is mounted.
    for spec in route_table(&[]).into_iter().filter(|s| s.owner == Owner::Ltx) {
        let uri = spec.path.replace("{id}", "00000000-0000-0000-0000-000000000000");
        let body = (spec.method == "POST").then(|| json!({}));
        let s = call(&r, spec.method, &uri, body, true).await;
        assert_ne!(s.status, StatusCode::METHOD_NOT_ALLOWED, "{} {uri}", spec.method);
        assert!(!(s.status == 404 && s.bytes.is_empty()), "{} {uri} unrouted", spec.method);
    }
}

/// LTX v1 sync with outputs in an S3-compatible bucket (the R2 deployment):
/// the bytes are read back through `ArtifactStore::open`.
#[cfg(feature = "http-client")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn ltx_v1_sync_reads_back_from_s3() {
    use axum::extract::Path;
    use axum::response::IntoResponse;
    use axum::routing::put;
    use std::collections::HashMap;
    use std::sync::{Arc, Mutex};
    let objects: Arc<Mutex<HashMap<String, Vec<u8>>>> = Arc::default();
    let (o1, o2) = (objects.clone(), objects.clone());
    let s3: Router = Router::new().route(
        "/{*key}",
        put(move |Path(k): Path<String>, body: axum::body::Bytes| async move {
            o1.lock().unwrap().insert(k, body.to_vec());
            StatusCode::OK
        })
        .get(move |Path(k): Path<String>| async move {
            match o2.lock().unwrap().get(&k) {
                Some(b) => (StatusCode::OK, b.clone()).into_response(),
                None => StatusCode::NOT_FOUND.into_response(),
            }
        }),
    );
    let l = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = l.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(l, s3).await.unwrap() });

    let mut c = config("s3");
    let mut env = BTreeMap::new();
    env.insert("FV_R2_ENDPOINT".to_owned(), format!("http://{addr}"));
    env.insert("FV_R2_BUCKET".to_owned(), "fv-media".to_owned());
    env.insert("FV_R2_ACCESS_KEY_ID".to_owned(), "AK".to_owned());
    env.insert("FV_R2_SECRET_ACCESS_KEY".to_owned(), "SK".to_owned());
    c.apply_env(&env).unwrap();
    c.validate().unwrap();
    let app = App::build(c, Overrides::default()).await.unwrap();
    app.gate.engine().wait_ready().await;
    let s = call(&app.router, "POST", "/v1/text-to-video", Some(t2v("ltx-turbo", "1280x720")), true).await;
    assert_eq!(s.status, 200, "{}", String::from_utf8_lossy(&s.bytes));
    assert_eq!(s.headers["content-type"], "video/mp4");
    let stored = objects.lock().unwrap().values().next().cloned().expect("uploaded to the bucket");
    assert_eq!(s.bytes, stored);
}
