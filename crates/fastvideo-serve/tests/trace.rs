//! Request tracing through every protocol adapter (docs/serve/tracing.md):
//! a request with `x-fv-trace: 1` and a `traceparent` is recorded under that
//! trace id from the HTTP layer through the adapter, the job store, the
//! engine queue, every engine stage and denoise step (host and fake-device
//! clocks) to the terminal write; the answer carries `traceparent` and the
//! clock sample `x-fv-trace-t`. Untraced requests record nothing.

#![cfg(all(feature = "openai-videos", feature = "minimax", feature = "fal"))]

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

const KEY: &str = "sk-trace";

async fn app(tag: &str) -> App {
    let dir = tempfile::Builder::new()
        .prefix(&format!("fv-serve-trace-{tag}-"))
        .tempdir()
        .unwrap()
        .keep();
    let mut env = BTreeMap::new();
    env.insert("FV_API_KEYS".to_owned(), KeyRing::hash_hex(KEY));
    env.insert("FV_URL_SIGNING_KEY".to_owned(), "k".to_owned());
    env.insert("FV_STATE_DIR".to_owned(), dir.display().to_string());
    env.insert("FV_PUBLIC_BASE_URL".to_owned(), "http://fv.test".to_owned());
    let mut c = Config::default();
    c.apply_env(&env).unwrap();
    c.jobs.backend = JobBackend::Memory;
    c.engine.fake.step_ms = 1;
    c.validate().unwrap();
    let a = App::build(c, Overrides::default()).await.unwrap();
    a.gate.engine().wait_ready().await;
    a
}

struct Resp {
    status: StatusCode,
    headers: axum::http::HeaderMap,
    json: Value,
}

async fn call(
    app: &Router,
    method: &str,
    uri: &str,
    body: Option<Value>,
    auth: Option<&str>,
    trace: Option<&str>,
) -> Resp {
    let mut b = Request::builder().method(method).uri(uri);
    if let Some(a) = auth {
        b = b.header("authorization", a);
    }
    if let Some(tp) = trace {
        b = b.header("traceparent", tp).header("x-fv-trace", "1");
    }
    let req = match body {
        Some(v) => b
            .header("content-type", "application/json")
            .body(Body::from(v.to_string()))
            .unwrap(),
        None => b.body(Body::empty()).unwrap(),
    };
    let r = app.clone().oneshot(req).await.unwrap();
    let (status, headers) = (r.status(), r.headers().clone());
    let bytes = axum::body::to_bytes(r.into_body(), 1 << 26).await.unwrap();
    Resp {
        status,
        headers,
        json: serde_json::from_slice(&bytes).unwrap_or(Value::Null),
    }
}

async fn poll(
    app: &Router,
    uri: &str,
    auth: Option<&str>,
    trace: Option<&str>,
    done: impl Fn(&Value) -> bool,
) {
    for _ in 0..4000 {
        let r = call(app, "GET", uri, None, auth, trace).await;
        if r.status == 200 && done(&r.json) {
            return;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    panic!("{uri} never finished");
}

fn tp(id: &str) -> String {
    format!("00-{id}-00f067aa0ba902b7-01")
}

/// The trace's events once the terminal write is in (the pump records it
/// right after the status flips).
async fn events(app: &Router, id: &str) -> Vec<(String, String)> {
    for _ in 0..200 {
        let r = call(app, "GET", &format!("/fv/v1/traces/{id}"), None, None, None).await;
        if r.status == 200 {
            let ev: Vec<(String, String)> = r.json["events"]
                .as_array()
                .unwrap()
                .iter()
                .map(|e| {
                    (
                        e["comp"].as_str().unwrap().to_owned(),
                        e["name"].as_str().unwrap().to_owned(),
                    )
                })
                .collect();
            if ev.iter().any(|(c, n)| c == "store" && n == "terminal") {
                assert_eq!(r.json["stats"]["dropped"], 0, "no event dropped");
                return ev;
            }
        }
        tokio::time::sleep(Duration::from_millis(25)).await;
    }
    panic!("trace {id} never got its terminal write");
}

fn assert_path(ev: &[(String, String)], what: &str) {
    for (c, n) in [
        ("http", "post"),
        ("adapter", "validate"),
        ("adapter", "ingest_negotiate"),
        ("store", "insert"),
        ("queue", "submit"),
        ("queue", "wait"),
        ("engine", "run"),
        ("engine", "denoise.step"),
        ("gpu", "denoise.step"),
        ("engine", "finished"),
        ("post", "finalize"),
        ("upload", "artifact_put"),
        ("store", "terminal_write"),
        ("store", "terminal"),
        ("http", "get"),
        ("store", "lookup"),
    ] {
        // MiniMax has its own submit path (no shared validate/ingest spans).
        if what == "minimax" && c == "adapter" {
            continue;
        }
        assert!(
            ev.iter().any(|(a, b)| a == c && b == n),
            "{what}: no {c}.{n} in {ev:?}"
        );
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn every_adapter_records_the_whole_path_under_the_callers_trace() {
    let a = app("adapters").await;
    let r = a.router.clone();
    let bearer = Some("Bearer sk-trace");
    let key = Some("Key sk-trace");

    // Native.
    let id = "11111111111111111111111111111111";
    let s = call(
        &r,
        "POST",
        "/fv/v1/jobs",
        Some(json!({"model": "h3-turbo", "prompt": "a fox", "short_edge": 768})),
        bearer,
        Some(&tp(id)),
    )
    .await;
    assert_eq!(s.status, 202, "{}", s.json);
    let echoed = s.headers["traceparent"].to_str().unwrap().to_owned();
    assert!(
        echoed.starts_with(&format!("00-{id}-")) && echoed.ends_with("-01"),
        "{echoed}"
    );
    let t = s.headers["x-fv-trace-t"].to_str().unwrap();
    let (t1, t2) = t.split_once(';').unwrap();
    assert!(t1.parse::<i64>().unwrap() <= t2.parse::<i64>().unwrap());
    assert!(s.headers.get("server-timing").is_some());
    let nid = s.json["id"].as_str().unwrap().to_owned();
    poll(
        &r,
        &format!("/fv/v1/jobs/{nid}"),
        bearer,
        Some(&tp(id)),
        |v| v["status"] == "succeeded",
    )
    .await;
    assert_path(&events(&r, id).await, "native");

    // OpenAI-style videos.
    let id = "22222222222222222222222222222222";
    let s = call(
        &r,
        "POST",
        "/v1/videos",
        Some(json!({"model": "h3-turbo", "prompt": "a cat", "seconds": "5"})),
        None,
        Some(&tp(id)),
    )
    .await;
    assert_eq!(s.status, 200, "{}", s.json);
    let vid = s.json["id"].as_str().unwrap().to_owned();
    poll(&r, &format!("/v1/videos/{vid}"), None, Some(&tp(id)), |v| {
        v["status"] == "completed"
    })
    .await;
    assert_path(&events(&r, id).await, "openai");

    // MiniMax.
    let id = "33333333333333333333333333333333";
    let body = json!({"model": "MiniMax-H3-Turbo", "content": [{"type": "text", "text": "a red fox"}], "resolution": "768P", "duration": 5});
    let s = call(
        &r,
        "POST",
        "/v2/video_generation",
        Some(body),
        bearer,
        Some(&tp(id)),
    )
    .await;
    assert_eq!(s.status, 200, "{}", s.json);
    let tid = s.json["task_id"].as_str().unwrap().to_owned();
    poll(
        &r,
        &format!("/v2/query/video_generation/{tid}"),
        bearer,
        Some(&tp(id)),
        |v| v["task"]["status"] == "succeeded",
    )
    .await;
    assert_path(&events(&r, id).await, "minimax");

    // fal queue (what the console uses).
    let id = "44444444444444444444444444444444";
    let s = call(
        &r,
        "POST",
        "/minimax/h3-turbo/text-to-video",
        Some(json!({"prompt": "a kitten"})),
        key,
        Some(&tp(id)),
    )
    .await;
    assert_eq!(s.status, 200, "{}", s.json);
    let rid = s.json["request_id"].as_str().unwrap().to_owned();
    poll(
        &r,
        &format!("/minimax/h3-turbo/requests/{rid}/status"),
        key,
        Some(&tp(id)),
        |v| v["status"] == "COMPLETED",
    )
    .await;
    assert_path(&events(&r, id).await, "fal");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn untraced_requests_record_nothing_and_events_can_be_posted() {
    let a = app("untraced").await;
    let r = a.router.clone();
    let id = "55555555555555555555555555555555";
    // A traceparent alone does not opt in.
    let mut b = Request::builder()
        .method("POST")
        .uri("/v1/videos")
        .header("traceparent", tp(id))
        .header("content-type", "application/json");
    b = b.header("x-request-id", "plain");
    let resp = r
        .clone()
        .oneshot(
            b.body(Body::from(
                json!({"model": "h3-turbo", "prompt": "a dog", "seconds": "5"}).to_string(),
            ))
            .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    assert!(resp.headers().get("x-fv-trace-t").is_none());
    let g = call(&r, "GET", &format!("/fv/v1/traces/{id}"), None, None, None).await;
    assert_eq!(g.status, StatusCode::NOT_FOUND);
    assert_eq!(
        call(&r, "GET", "/fv/v1/traces/nothex", None, None, None)
            .await
            .status,
        StatusCode::BAD_REQUEST
    );

    // The browser's beacon / the edge's shipment.
    let id = "66666666666666666666666666666666";
    let ev = json!({"events": [{"trace": "ignored", "comp": "client", "name": "click", "t_wall_ns": 1_800_000_000_000_000_000i64}]});
    let p = call(
        &r,
        "POST",
        &format!("/fv/v1/traces/{id}/events"),
        Some(ev),
        None,
        None,
    )
    .await;
    assert_eq!(p.status, StatusCode::ACCEPTED, "{}", p.json);
    let g = call(&r, "GET", &format!("/fv/v1/traces/{id}"), None, None, None).await;
    assert_eq!(g.status, 200);
    assert_eq!(g.json["events"][0]["host"], "client");
    assert_eq!(g.json["events"][0]["trace"], id);
}
