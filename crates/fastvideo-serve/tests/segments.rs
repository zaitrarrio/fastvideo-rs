//! `/v1/segments` over the fake engine (docs/serve/segments.md): one causal
//! session per stream, segments cut from one continuous rollout, MP4
//! artifacts behind signed URLs, idempotent resubmits, executor contention
//! and explicit close.

use std::collections::BTreeMap;
use std::time::Duration;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use axum::Router;
use fastvideo_serve::config::{Config, JobBackend};
use fastvideo_serve::{App, Overrides};
use fastvideo_serve_kit::KeyRing;
use serde_json::{json, Value};
use tower::ServiceExt;

const KEY: &str = "sk-test-fv";

fn config(tag: &str) -> Config {
    let dir = tempfile::Builder::new()
        .prefix(&format!("fv-serve-segments-{tag}-"))
        .tempdir()
        .unwrap()
        .keep();
    let mut env = BTreeMap::new();
    env.insert("FV_API_KEYS".to_owned(), KeyRing::hash_hex(KEY));
    env.insert(
        "FV_URL_SIGNING_KEY".to_owned(),
        "test-signing-key".to_owned(),
    );
    env.insert("FV_STATE_DIR".to_owned(), dir.display().to_string());
    env.insert("FV_PUBLIC_BASE_URL".to_owned(), "http://fv.test".to_owned());
    let mut c = Config::default();
    c.apply_env(&env).unwrap();
    c.jobs.backend = JobBackend::File;
    c.engine.fake.step_ms = 1;
    c.segments.encoder = "libx264".into();
    c.segments.idle_ttl_s = 2;
    c.validate().unwrap();
    c
}

async fn call(app: &Router, method: &str, uri: &str, body: Option<Value>) -> (StatusCode, Value) {
    let mut b = Request::builder()
        .method(method)
        .uri(uri)
        .header("authorization", format!("Bearer {KEY}"));
    let req = match body {
        Some(v) => {
            b = b.header("content-type", "application/json");
            b.body(Body::from(v.to_string())).unwrap()
        }
        None => b.body(Body::empty()).unwrap(),
    };
    let resp = app.clone().oneshot(req).await.unwrap();
    let status = resp.status();
    let bytes = axum::body::to_bytes(resp.into_body(), 1 << 26)
        .await
        .unwrap();
    let v = serde_json::from_slice(&bytes)
        .unwrap_or_else(|_| Value::String(String::from_utf8_lossy(&bytes).into_owned()));
    (status, v)
}

async fn wait_done(app: &Router, id: &str) -> Value {
    for _ in 0..2000 {
        let (s, v) = call(app, "GET", &format!("/v1/segments/{id}"), None).await;
        assert_eq!(s, 200, "{v}");
        if matches!(v["status"].as_str(), Some("succeeded" | "failed")) {
            return v;
        }
        tokio::time::sleep(Duration::from_millis(25)).await;
    }
    panic!("segment {id} never finished");
}

fn segment(stream: &str, index: u64, prompt: &str, seconds: f64) -> Value {
    json!({"stream_id": stream, "segment_index": index, "prompt": prompt, "duration_sec": seconds})
}

#[tokio::test]
async fn segments_come_from_one_continuous_causal_session() {
    let app = App::build(config("flow"), Overrides::default())
        .await
        .unwrap();
    app.gate.engine().wait_ready().await;
    let r = app.router.clone();

    // Unauthenticated and invalid requests.
    let resp = r
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/segments")
                .header("content-type", "application/json")
                .body(Body::from(segment("s", 0, "p", 3.0).to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), 401);
    let (s, v) = call(&r, "POST", "/v1/segments", Some(json!({"stream_id": "str_a", "segment_index": 0, "prompt": "p", "duration_sec": 3.0, "context": {"image_url": "https://x/0.png"}}))).await;
    assert_eq!(s, 400, "{v}");
    assert_eq!(v["error"]["param"], "context.image_url");
    let (s, v) = call(&r, "POST", "/v1/segments", Some(json!({"stream_id": "str_a", "segment_index": 0, "prompt": "p", "duration_sec": 3.0, "model": "h3-turbo"}))).await;
    assert_eq!(s, 400, "{v}");
    assert_eq!(v["error"]["param"], "model");

    // Segment 0: opens the session. 3 s at 16 fps = 48 frames = 4 blocks.
    let (s, v) = call(
        &r,
        "POST",
        "/v1/segments",
        Some(segment("str_a", 0, "a harbor at dusk", 3.0)),
    )
    .await;
    assert_eq!(s, 202, "{v}");
    assert_eq!(v["status"], "queued");
    assert_eq!(v["trim_head_sec"], 0.0);
    let id0 = v["id"].as_str().unwrap().to_owned();
    assert!(id0.starts_with("fvseg_"));
    // Resubmitting the same (stream, index) is idempotent.
    let (s, again) = call(
        &r,
        "POST",
        "/v1/segments",
        Some(segment("str_a", 0, "a harbor at dusk", 3.0)),
    )
    .await;
    assert_eq!(s, 202);
    assert_eq!(again["id"], id0);

    // Segment 1 queues behind it; 2.5 s = 40 frames, which is not a whole
    // number of 12-frame blocks: the rest carries over.
    let (s, v) = call(
        &r,
        "POST",
        "/v1/segments",
        Some(segment("str_a", 1, "a storm breaks over the harbor", 2.5)),
    )
    .await;
    assert_eq!(s, 202, "{v}");
    let id1 = v["id"].as_str().unwrap().to_owned();

    let d0 = wait_done(&r, &id0).await;
    assert_eq!(d0["status"], "succeeded", "{d0}");
    assert_eq!(d0["frames"], 48);
    assert_eq!(d0["fps"], 16);
    assert_eq!(d0["height"], 480);
    assert_eq!(d0["blocks"], json!([0, 3]));
    assert_eq!(d0["session"]["restarted"], false);
    let d1 = wait_done(&r, &id1).await;
    assert_eq!(d1["status"], "succeeded", "{d1}");
    assert_eq!(d1["frames"], 40);
    // Blocks 4..7 (48 frames) cover 40 frames; 8 carry into the next segment.
    assert_eq!(d1["blocks"], json!([4, 7]));

    // The artifact is an MP4 behind a signed URL on this server.
    let url = d0["video_url"].as_str().unwrap();
    let path = url.strip_prefix("http://fv.test").unwrap();
    let resp = r
        .clone()
        .oneshot(Request::builder().uri(path).body(Body::empty()).unwrap())
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    assert_eq!(
        resp.headers()
            .get("content-type")
            .and_then(|v| v.to_str().ok()),
        Some("video/mp4")
    );
    let bytes = axum::body::to_bytes(resp.into_body(), 1 << 26)
        .await
        .unwrap();
    assert!(bytes.len() > 1000, "mp4 of {} bytes", bytes.len());
    assert_eq!(&bytes[4..8], b"ftyp");

    // Segment 2 uses the 8 carried frames plus 2 new blocks (16 + 24 = 40 >= 32).
    let (s, v) = call(
        &r,
        "POST",
        "/v1/segments",
        Some(segment("str_a", 2, "a storm breaks over the harbor", 2.0)),
    )
    .await;
    assert_eq!(s, 202, "{v}");
    let d2 = wait_done(&r, v["id"].as_str().unwrap()).await;
    assert_eq!(d2["frames"], 32);
    assert_eq!(d2["blocks"], json!([8, 9]));

    // The stream's session is visible, and one executor serves one stream:
    // a second stream's segment fails with the executor busy.
    let (s, v) = call(&r, "GET", "/v1/segments/streams/str_a", None).await;
    assert_eq!(s, 200);
    assert_eq!(v["model"], "fake-sfwan");
    assert_eq!(v["sessions"], 1);
    assert!(v["blocks"].as_u64().unwrap() >= 10);
    let (s, v) = call(
        &r,
        "POST",
        "/v1/segments",
        Some(segment("str_b", 0, "elsewhere", 1.0)),
    )
    .await;
    assert_eq!(s, 202, "{v}");
    let db = wait_done(&r, v["id"].as_str().unwrap()).await;
    assert_eq!(db["status"], "failed", "{db}");

    // Closing the stream frees the executor; the other stream now runs.
    let (s, v) = call(&r, "DELETE", "/v1/segments/streams/str_a", None).await;
    assert_eq!(s, 200);
    assert_eq!(v["closed"], true);
    tokio::time::sleep(Duration::from_millis(100)).await;
    let (s, v) = call(
        &r,
        "POST",
        "/v1/segments",
        Some(segment("str_b", 1, "elsewhere", 1.0)),
    )
    .await;
    assert_eq!(s, 202, "{v}");
    let db = wait_done(&r, v["id"].as_str().unwrap()).await;
    assert_eq!(db["status"], "succeeded", "{db}");
    assert_eq!(db["frames"], 16);
    // The `/fv/v1` alias serves the same objects.
    let (s, v) = call(
        &r,
        "GET",
        &format!("/fv/v1/segments/{}", db["id"].as_str().unwrap()),
        None,
    )
    .await;
    assert_eq!(s, 200);
    assert_eq!(v["status"], "succeeded");
    let (s, _) = call(&r, "GET", "/v1/segments/fvseg_nope", None).await;
    assert_eq!(s, 404);
}

#[tokio::test]
async fn idle_streams_close_and_a_reset_cuts() {
    let app = App::build(config("idle"), Overrides::default())
        .await
        .unwrap();
    app.gate.engine().wait_ready().await;
    let r = app.router.clone();
    let (s, v) = call(
        &r,
        "POST",
        "/v1/segments",
        Some(segment("str_i", 0, "quiet", 1.0)),
    )
    .await;
    assert_eq!(s, 202, "{v}");
    let d = wait_done(&r, v["id"].as_str().unwrap()).await;
    assert_eq!(d["status"], "succeeded", "{d}");
    // A scene cut: the KV cache is reset, blocks restart at 0.
    let (s, v) = call(&r, "POST", "/v1/segments", Some(json!({"stream_id": "str_i", "segment_index": 1, "prompt": "a new place", "duration_sec": 1.0, "reset": true}))).await;
    assert_eq!(s, 202, "{v}");
    let d = wait_done(&r, v["id"].as_str().unwrap()).await;
    assert_eq!(d["status"], "succeeded", "{d}");
    assert_eq!(d["blocks"], json!([0, 1]));
    assert_eq!(d["session"]["restarted"], false);
    // Idle for longer than the TTL: the session closes and the stream is gone.
    tokio::time::sleep(Duration::from_millis(2600)).await;
    let (s, _) = call(&r, "GET", "/v1/segments/streams/str_i", None).await;
    assert_eq!(s, 404);
    // ...and the next segment opens a fresh session on the same stream id.
    let (s, v) = call(
        &r,
        "POST",
        "/v1/segments",
        Some(segment("str_i", 2, "back again", 1.0)),
    )
    .await;
    assert_eq!(s, 202, "{v}");
    let d = wait_done(&r, v["id"].as_str().unwrap()).await;
    assert_eq!(d["status"], "succeeded", "{d}");
    assert_eq!(d["blocks"], json!([0, 1]));
}
