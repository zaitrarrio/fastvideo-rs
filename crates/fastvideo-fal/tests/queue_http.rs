//! The fal queue API over HTTP against the fake engine: both path forms,
//! status invariants, cancel semantics, SSE, sync, `sync_mode`, proxy mode,
//! storage initiate, webhooks, queue limits and refusals.

mod queue_common;

use std::time::Duration;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use base64::Engine as _;
use queue_common::*;
use ed25519_dalek::{Signature, Verifier, VerifyingKey};
use fastvideo_protocol::{ApiError, JobStatus, ProtocolId};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

const T2V: &str = "/minimax/h3-max/text-to-video";

fn slow() -> Opts {
    Opts { step: Duration::from_millis(150), ..Opts::default() }
}

async fn submit(app: &axum::Router, path: &str, body: Value) -> Value {
    let r = call(app, "POST", path, Some(body)).await;
    assert_eq!(r.status, StatusCode::OK, "{:?}", r.json());
    let v = r.json();
    assert_eq!(r.header("x-fal-request-id"), v["request_id"].as_str());
    v
}

#[tokio::test]
async fn submit_poll_result_both_path_forms() {
    let f = fixture(Opts { engine: Engine::Fake, ..Opts::default() }).await;
    let s = submit(&f.app, T2V, json!({"prompt": "a kitten", "seed": 7})).await;
    let rid = s["request_id"].as_str().unwrap().to_owned();
    assert!(uuid::Uuid::parse_str(&rid).is_ok());
    let base = format!("https://fal.fv.test/minimax/h3-max/requests/{rid}");
    assert_eq!(s["response_url"], base, "app-only, no sub-path, no /response");
    assert_eq!(s["status_url"], format!("{base}/status"));
    assert_eq!(s["cancel_url"], format!("{base}/cancel"));
    assert!(s["queue_position"].is_u64());

    let done = wait_completed(&f.app, &format!("{}?logs=1", path_of(s["status_url"].as_str().unwrap()))).await;
    assert!(done.get("error").is_none(), "{done}");
    let logs = done["logs"].as_array().unwrap();
    assert!(!logs.is_empty(), "logs=1 returns the job's logs");
    for l in logs {
        assert_eq!(l["source"], "USER");
        assert!(["DEBUG", "INFO", "WARN", "ERROR"].contains(&l["level"].as_str().unwrap()));
        assert!(l["timestamp"].as_str().unwrap().ends_with('Z'));
    }
    assert!(done["metrics"]["inference_time"].is_number());

    // The full endpoint form and ?logs=0 / false.
    for q in ["logs=0", "logs=false", "logs=true", ""] {
        let r = call(&f.app, "GET", &format!("{T2V}/requests/{rid}/status?{q}"), None).await;
        assert_eq!(r.status, StatusCode::OK);
        let v = r.json();
        assert_eq!(v["status"], "COMPLETED");
        assert_eq!(v["logs"].as_array().unwrap().is_empty(), q != "logs=true", "{q}: {v}");
    }

    // Result under every alias, identical.
    let mut bodies = Vec::new();
    for p in [
        format!("/minimax/h3-max/requests/{rid}"),
        format!("/minimax/h3-max/requests/{rid}/response"),
        format!("{T2V}/requests/{rid}"),
        format!("{T2V}/requests/{rid}/response"),
    ] {
        let r = call(&f.app, "GET", &p, None).await;
        assert_eq!(r.status, StatusCode::OK, "{p}");
        assert_eq!(r.header("x-fal-request-id"), Some(rid.as_str()));
        assert_eq!(r.header("x-fv-tier"), Some("max"));
        assert_eq!(r.header("x-fv-recipe"), Some("full-dense"));
        bodies.push(r.json());
    }
    assert!(bodies.windows(2).all(|w| w[0]["video"]["file_size"] == w[1]["video"]["file_size"]));
    let out = &bodies[0];
    assert_eq!(out["expanded_prompt"], Value::Null);
    assert_eq!(out["seed"], 7, "the effective seed is returned on t2v too");
    assert!(out["timings"]["inference"].is_number());
    let v = &out["video"];
    assert_eq!(v["content_type"], "video/mp4");
    let name = v["file_name"].as_str().unwrap();
    assert!(name.ends_with("_minimax-h3-max.mp4") && name.len() == 40, "{name}");
    let url = v["url"].as_str().unwrap();
    assert!(url.starts_with("https://fal.fv.test/files/") && url.contains(name));

    // The output downloads without a key (FL: "must not be sent one").
    let r = send(&f.app, Request::get(path_of(url)).body(Body::empty()).unwrap()).await;
    assert_eq!(r.status, StatusCode::OK);
    assert_eq!(r.header("access-control-allow-origin"), Some("*"));
    assert_eq!(r.body.len() as u64, v["file_size"].as_u64().unwrap());

    // The job: fal protocol, t2v 1344x768 124 frames, seed kept, echo model.
    let j = f.ctx.jobs().by_external(ProtocolId::Fal, &rid).await.unwrap();
    assert_eq!((j.resolved.width, j.resolved.height, j.resolved.num_frames, j.resolved.fps), (1344, 768, 124, 24));
    assert_eq!(j.resolved.seed, 7);
    assert_eq!(j.resolved.model.0, "fake-h3-max");
    assert_eq!(j.requested_model(), "minimax/h3-max/text-to-video");
    assert_eq!(j.expires_at - j.created_at, time::Duration::hours(24));
}

#[tokio::test]
async fn unknown_foreign_and_unauthorized() {
    let f = fixture(Opts::default()).await;
    let nope = uuid::Uuid::new_v4();
    for (m, p) in [
        ("GET", format!("/minimax/h3-max/requests/{nope}/status")),
        ("GET", format!("/minimax/h3-max/requests/{nope}")),
        ("GET", format!("/minimax/h3-max/requests/{nope}/response")),
        ("GET", format!("/minimax/h3-max/requests/{nope}/status/stream")),
        ("PUT", format!("/minimax/h3-max/requests/{nope}/cancel")),
        ("GET", format!("{T2V}/requests/{nope}/status")),
    ] {
        let r = call(&f.app, m, &p, None).await;
        assert_eq!((r.status, r.json()), (StatusCode::NOT_FOUND, json!({"status": "NOT_FOUND"})), "{m} {p}");
        // FL's key probe: a bad key is 401 on the same route.
        let r = send(&f.app, req(m, &p, Some("wrong"), None)).await;
        assert_eq!((r.status, r.json()), (StatusCode::UNAUTHORIZED, json!({"detail": "invalid key credentials"})));
        assert_eq!(r.header("x-fal-error-type"), Some("unauthorized"));
    }
    let r = send(&f.app, req("POST", T2V, None, Some(json!({"prompt": "p"})))).await;
    assert_eq!(r.status, StatusCode::UNAUTHORIZED);
    let r = send(&f.app, Request::post(T2V).header("authorization", format!("Bearer {KEY}")).body(Body::from("{}")).unwrap()).await;
    assert_eq!(r.status, StatusCode::UNAUTHORIZED, "fal takes `Key`, not `Bearer`");

    let s = submit(&f.app, T2V, json!({"prompt": "p"})).await;
    let rid = s["request_id"].as_str().unwrap();
    // Another key's request is not visible.
    let r = send(&f.app, req("GET", &format!("/minimax/h3-max/requests/{rid}/status"), Some(OTHER_KEY), None)).await;
    assert_eq!(r.status, StatusCode::NOT_FOUND);
    // Another app's prefix does not find it.
    let r = call(&f.app, "GET", &format!("/minimax/h3-turbo/requests/{rid}/status"), None).await;
    assert_eq!(r.status, StatusCode::NOT_FOUND);
    // A sub-path of the right app does.
    let r = call(&f.app, "GET", &format!("/minimax/h3-max/image-to-video/requests/{rid}/status"), None).await;
    assert_eq!(r.status, StatusCode::OK);
}

#[tokio::test]
async fn cancel_codes_and_semantics() {
    let f = fixture(Opts { engine: Engine::Fake, ..slow() }).await;
    let a = submit(&f.app, T2V, json!({"prompt": "running one"})).await;
    let b = submit(&f.app, T2V, json!({"prompt": "queued one"})).await;
    let (ra, rb) = (a["request_id"].as_str().unwrap(), b["request_id"].as_str().unwrap());

    // B waits behind A: IN_QUEUE with a position.
    let st = call(&f.app, "GET", &path_of(b["status_url"].as_str().unwrap()), None).await.json();
    assert_eq!(st["status"], "IN_QUEUE", "{st}");
    assert!(st["queue_position"].as_u64().is_some());

    // Queued: cancelled at once.
    let r = call(&f.app, "PUT", &path_of(b["cancel_url"].as_str().unwrap()), None).await;
    assert_eq!((r.status, r.json()), (StatusCode::ACCEPTED, json!({"status": "CANCELLATION_REQUESTED"})));
    assert_eq!(r.header("x-fal-request-id"), Some(rb));
    let st = call(&f.app, "GET", &format!("/minimax/h3-max/requests/{rb}/status"), None).await.json();
    assert_eq!((st["status"].as_str(), st["error_type"].as_str()), (Some("COMPLETED"), Some("client_cancelled")), "{st}");
    assert!(st["logs"].is_array());
    let r = call(&f.app, "PUT", &format!("{T2V}/requests/{rb}/cancel"), None).await;
    assert_eq!((r.status, r.json()), (StatusCode::BAD_REQUEST, json!({"status": "ALREADY_COMPLETED"})));
    let r = call(&f.app, "GET", &format!("/minimax/h3-max/requests/{rb}"), None).await;
    assert_eq!(r.status.as_u16(), 499);
    assert_eq!(r.json()["error_type"], "client_cancelled");

    // Running: the engine stops at its next step.
    let deadline = tokio::time::Instant::now() + T;
    loop {
        let st = call(&f.app, "GET", &format!("/minimax/h3-max/requests/{ra}/status"), None).await.json();
        if st["status"] == "IN_PROGRESS" {
            break;
        }
        assert!(tokio::time::Instant::now() < deadline);
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    let r = call(&f.app, "PUT", &format!("/minimax/h3-max/requests/{ra}/cancel"), None).await;
    assert_eq!(r.status, StatusCode::ACCEPTED);
    let done = wait_completed(&f.app, &format!("/minimax/h3-max/requests/{ra}/status")).await;
    assert_eq!(done["error_type"], "client_cancelled", "{done}");
    let j = f.ctx.jobs().by_external(ProtocolId::Fal, ra).await.unwrap();
    assert_eq!(j.status(), JobStatus::Cancelled);

    // Completed: 400.
    let f2 = fixture(Opts::default()).await;
    let c = submit(&f2.app, T2V, json!({"prompt": "p"})).await;
    wait_completed(&f2.app, &path_of(c["status_url"].as_str().unwrap())).await;
    let r = call(&f2.app, "PUT", &path_of(c["cancel_url"].as_str().unwrap()), None).await;
    assert_eq!((r.status, r.json()), (StatusCode::BAD_REQUEST, json!({"status": "ALREADY_COMPLETED"})));
}

#[tokio::test]
async fn result_before_completion_and_failures() {
    let f = fixture(slow()).await;
    let s = submit(&f.app, T2V, json!({"prompt": "p"})).await;
    let r = call(&f.app, "GET", &path_of(s["response_url"].as_str().unwrap()), None).await;
    assert_eq!((r.status, r.json()), (StatusCode::BAD_REQUEST, json!({"detail": "Request is still in progress"})));

    let f = fixture(Opts::default()).await;
    let s = submit(&f.app, T2V, json!({"prompt": "boom [fake:fail]"})).await;
    let done = wait_completed(&f.app, &path_of(s["status_url"].as_str().unwrap())).await;
    assert_eq!(done["error_type"], "internal_server_error", "{done}");
    assert!(done["error"].as_str().unwrap().contains("injected failure"));
    let r = call(&f.app, "GET", &path_of(s["response_url"].as_str().unwrap()), None).await;
    assert_eq!(r.status, StatusCode::INTERNAL_SERVER_ERROR);
    assert!(r.json()["detail"][0]["msg"].as_str().unwrap().contains("injected failure"));
    assert!(r.json().get("video").is_none(), "FL must not find a video");
}

#[tokio::test]
async fn sse_status_stream() {
    let f = fixture(Opts { step: Duration::from_millis(40), ..Opts::default() }).await;
    let s = submit(&f.app, T2V, json!({"prompt": "p"})).await;
    let rid = s["request_id"].as_str().unwrap();
    let r = tokio::time::timeout(T, call(&f.app, "GET", &format!("/minimax/h3-max/requests/{rid}/status/stream?logs=1"), None))
        .await
        .expect("the stream closes after COMPLETED");
    assert_eq!(r.status, StatusCode::OK);
    assert_eq!(r.header("content-type"), Some("text/event-stream"));
    let text = String::from_utf8(r.body.to_vec()).unwrap();
    let events: Vec<Value> = text.lines().filter_map(|l| l.strip_prefix("data: ")).map(|d| serde_json::from_str(d).unwrap()).collect();
    assert!(events.len() >= 2, "{text}");
    assert_eq!(events.last().unwrap()["status"], "COMPLETED");
    assert!(events.iter().all(|e| e["request_id"] == rid && e.get("response_url").is_some()));
    assert!(events.iter().any(|e| e["status"] == "IN_PROGRESS" || e["status"] == "IN_QUEUE"));
    assert!(!events.last().unwrap()["logs"].as_array().unwrap().is_empty(), "logs=1 holds for followed events");
    // A finished job: one event, then close.
    let r = call(&f.app, "GET", &format!("{T2V}/requests/{rid}/status/stream"), None).await;
    let text = String::from_utf8(r.body.to_vec()).unwrap();
    assert_eq!(text.lines().filter(|l| l.starts_with("data: ")).count(), 1);
}

#[tokio::test]
async fn sync_run_and_sync_mode() {
    let f = fixture(Opts::default()).await;
    let r = call(&f.app, "POST", "/run/minimax/h3-max/text-to-video", Some(json!({"prompt": "p"}))).await;
    assert_eq!(r.status, StatusCode::OK, "{:?}", r.json());
    let rid = r.header("x-fal-request-id").unwrap().to_owned();
    let out = r.json();
    assert!(out["video"]["url"].as_str().unwrap().starts_with("https://fal.fv.test/files/"));
    // No seed in the request: the output reports the one the server drew.
    let j = f.ctx.jobs().by_external(ProtocolId::Fal, &rid).await.unwrap();
    assert_eq!(out["seed"].as_u64(), Some(j.resolved.seed), "{out}");
    // The sync job is visible through the queue routes too.
    let st = call(&f.app, "GET", &format!("/minimax/h3-max/requests/{rid}/status"), None).await.json();
    assert_eq!(st["status"], "COMPLETED");

    // sync_mode: the video comes back inline, on the queue result and on /run.
    for path in ["/run/minimax/h3-max/text-to-video", T2V] {
        let r = call(&f.app, "POST", path, Some(json!({"prompt": "p", "sync_mode": true}))).await;
        assert_eq!(r.status, StatusCode::OK);
        let out = if path == T2V {
            let s = r.json();
            wait_completed(&f.app, &path_of(s["status_url"].as_str().unwrap())).await;
            call(&f.app, "GET", &path_of(s["response_url"].as_str().unwrap()), None).await.json()
        } else {
            r.json()
        };
        let url = out["video"]["url"].as_str().unwrap();
        let b64 = url.strip_prefix("data:video/mp4;base64,").expect("data URI");
        let bytes = base64::engine::general_purpose::STANDARD.decode(b64).unwrap();
        assert_eq!(bytes.len() as u64, out["video"]["file_size"].as_u64().unwrap());
    }

    // Sync errors are rendered like queue submit errors.
    let r = call(&f.app, "POST", "/run/minimax/h3-max/text-to-video", Some(json!({"prompt": "p", "duration": 3}))).await;
    assert_eq!(r.status, StatusCode::UNPROCESSABLE_ENTITY);
    assert!(r.header("x-fal-request-id").is_some());
    // A failed sync job answers the error.
    let r = call(&f.app, "POST", "/run/minimax/h3-max/text-to-video", Some(json!({"prompt": "[fake:fail]"}))).await;
    assert_eq!(r.status, StatusCode::INTERNAL_SERVER_ERROR);
}

fn proxied(target: &str, method: &str, body: Option<Value>) -> Request<Body> {
    let mut r = req(method, "/fal/proxy", Some(KEY), body);
    r.headers_mut().insert("x-fal-target-url", target.parse().unwrap());
    r
}

#[tokio::test]
async fn proxy_mode_routes_by_target_url() {
    let f = fixture(Opts::default()).await;
    let r = send(&f.app, proxied("https://queue.fal.run/minimax/h3-max/text-to-video", "POST", Some(json!({"prompt": "p"})))).await;
    assert_eq!(r.status, StatusCode::OK, "{:?}", r.json());
    let rid = r.json()["request_id"].as_str().unwrap().to_owned();
    let deadline = tokio::time::Instant::now() + T;
    loop {
        let r = send(&f.app, proxied(&format!("https://queue.fal.run/minimax/h3-max/requests/{rid}/status?logs=1"), "GET", None)).await;
        assert_eq!(r.status, StatusCode::OK);
        if r.json()["status"] == "COMPLETED" {
            break;
        }
        assert!(tokio::time::Instant::now() < deadline);
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    let r = send(&f.app, proxied(&format!("https://queue.fal.run/minimax/h3-max/requests/{rid}"), "GET", None)).await;
    assert_eq!(r.status, StatusCode::OK);
    assert!(r.json()["video"]["url"].is_string());
    let r = send(&f.app, proxied(&format!("https://queue.fal.run/minimax/h3-max/requests/{rid}/cancel"), "PUT", None)).await;
    assert_eq!(r.status, StatusCode::BAD_REQUEST);
    // Sync through fal.run.
    let r = send(&f.app, proxied("https://fal.run/minimax/h3-max/text-to-video", "POST", Some(json!({"prompt": "p"})))).await;
    assert_eq!(r.status, StatusCode::OK);
    assert!(r.header("x-fal-request-id").is_some());
    // Storage initiate through rest.fal.ai.
    let r = send(
        &f.app,
        proxied("https://rest.fal.ai/storage/upload/initiate?storage_type=fal-cdn-v3", "POST", Some(json!({"content_type": "image/png", "file_name": "a.png"}))),
    )
    .await;
    assert_eq!(r.status, StatusCode::OK);
    assert!(r.json()["upload_url"].as_str().unwrap().contains("/uploads/"));
    // Auth still applies.
    let mut r0 = proxied("https://queue.fal.run/minimax/h3-max/text-to-video", "POST", Some(json!({"prompt": "p"})));
    r0.headers_mut().remove("authorization");
    assert_eq!(send(&f.app, r0).await.status, StatusCode::UNAUTHORIZED);
    // Bad targets.
    let r = send(&f.app, req("POST", "/fal/proxy", Some(KEY), Some(json!({})))).await;
    assert_eq!(r.status, StatusCode::BAD_REQUEST);
    assert!(r.json()["detail"].as_str().unwrap().contains("x-fal-target-url"));
    let r = send(&f.app, proxied("https://evil.example.com/minimax/h3-max/text-to-video", "POST", Some(json!({})))).await;
    assert_eq!(r.status, StatusCode::BAD_REQUEST);
    let r = send(&f.app, proxied("https://queue.fal.run/other/app/text-to-video", "POST", Some(json!({})))).await;
    assert_eq!(r.status, StatusCode::NOT_FOUND);
}

fn png(w: u32, h: u32) -> Vec<u8> {
    let img = image::RgbImage::from_pixel(w, h, image::Rgb([10, 200, 30]));
    let mut out = std::io::Cursor::new(Vec::new());
    img.write_to(&mut out, image::ImageFormat::Png).unwrap();
    out.into_inner()
}

#[tokio::test]
async fn storage_upload_feeds_image_to_video() {
    let f = fixture(Opts::default()).await;
    let r = call(&f.app, "POST", "/storage/upload/initiate?storage_type=fal-cdn-v3", Some(json!({"content_type": "image/png", "file_name": "cat.png"}))).await;
    assert_eq!(r.status, StatusCode::OK, "{:?}", r.json());
    let t = r.json();
    let (upload_url, file_url) = (t["upload_url"].as_str().unwrap(), t["file_url"].as_str().unwrap());
    assert!(upload_url.starts_with("https://fal.fv.test/uploads/"));
    assert!(file_url.starts_with("https://fal.fv.test/files/") && file_url.contains("cat.png"));
    // Plain PUT, no auth (JS uses bare fetch).
    let put = Request::put(path_of(upload_url)).header("content-type", "image/png").body(Body::from(png(96, 128))).unwrap();
    assert_eq!(send(&f.app, put).await.status, StatusCode::OK);
    let s = submit(&f.app, "/minimax/h3-max/image-to-video", json!({"prompt": "p", "image_url": file_url})).await;
    let j = f.ctx.jobs().by_external(ProtocolId::Fal, s["request_id"].as_str().unwrap()).await.unwrap();
    assert_eq!(j.resolved.task, fastvideo_protocol::Task::I2V);
    assert!(j.resolved.height > j.resolved.width, "the canvas follows the 3:4 image: {}x{}", j.resolved.width, j.resolved.height);
    wait_completed(&f.app, &path_of(s["status_url"].as_str().unwrap())).await;
    // Unauthenticated initiate is refused.
    let r = send(&f.app, req("POST", "/storage/upload/initiate", None, Some(json!({})))).await;
    assert_eq!(r.status, StatusCode::UNAUTHORIZED);
}

#[tokio::test]
async fn inputs_by_data_uri_and_reference_to_video() {
    let f = fixture(Opts::default()).await;
    let uri = format!("data:image/png;base64,{}", base64::engine::general_purpose::STANDARD.encode(png(128, 128)));
    // First + last frame on i2v (fl2va keyframes).
    let s = submit(&f.app, "/minimax/h3-max/image-to-video", json!({"prompt": "p", "image_url": uri, "end_image_url": uri})).await;
    let j = f.ctx.jobs().by_external(ProtocolId::Fal, s["request_id"].as_str().unwrap()).await.unwrap();
    assert_eq!(j.resolved.task, fastvideo_protocol::Task::Keyframes);
    assert_eq!(j.resolved.width, j.resolved.height, "square image, square canvas");
    // r2v on the max tier (ref2va resident): the output has `seed`.
    let s = submit(&f.app, "/minimax/h3-max/reference-to-video", json!({"prompt": "Image 1 dances", "reference_image_urls": [uri], "seed": 99})).await;
    wait_completed(&f.app, &path_of(s["status_url"].as_str().unwrap())).await;
    let out = call(&f.app, "GET", &path_of(s["response_url"].as_str().unwrap()), None).await.json();
    assert_eq!(out["seed"], 99);
    // r2v on turbo (no ref2va) is a gap.
    let r = call(&f.app, "POST", "/minimax/h3-turbo/reference-to-video", Some(json!({"prompt": "p", "reference_image_urls": [uri]}))).await;
    assert_eq!(r.status, StatusCode::UNPROCESSABLE_ENTITY, "{:?}", r.json());
    // Undecodable image: 422 naming the field.
    let r = call(&f.app, "POST", "/minimax/h3-max/image-to-video", Some(json!({"prompt": "p", "end_image_url": "data:image/png;base64,AAAA"}))).await;
    assert_eq!(r.status, StatusCode::UNPROCESSABLE_ENTITY);
    assert_eq!(r.json()["detail"][0]["loc"], json!(["body", "end_image_url"]), "{:?}", r.json());
    assert!(f.ctx.jobs().list(Default::default()).await.items.len() == 2, "refusals create no job");
}

#[tokio::test]
async fn tiers_gaps_and_admission() {
    let f = fixture(Opts::default()).await;
    // Turbo resolves to the turbo model and says so.
    let s = submit(&f.app, "/minimax/h3-turbo/text-to-video", json!({"prompt": "p"})).await;
    assert!(s["response_url"].as_str().unwrap().contains("/minimax/h3-turbo/requests/"));
    let j = f.ctx.jobs().by_external(ProtocolId::Fal, s["request_id"].as_str().unwrap()).await.unwrap();
    assert_eq!(j.resolved.model.0, "fake-h3-turbo");
    wait_completed(&f.app, &path_of(s["status_url"].as_str().unwrap())).await;
    let r = call(&f.app, "GET", &path_of(s["response_url"].as_str().unwrap()), None).await;
    assert_eq!(r.header("x-fv-tier"), Some("turbo"));
    // No draft model in this fixture: the app is not found.
    let r = call(&f.app, "POST", "/minimax/h3-draft/text-to-video", Some(json!({"prompt": "p"}))).await;
    assert_eq!(r.status, StatusCode::NOT_FOUND);
    assert_eq!(r.json()["detail"], "Application \"minimax/h3-draft\" not found");

    // Gaps are 422 with the fal field in `loc`.
    for (body, loc) in [
        (json!({"prompt": "p", "resolution": "1080P"}), "resolution"),
        (json!({"prompt": "p", "target_audio_url": "https://a.test/a.mp3"}), "target_audio_url"),
        (json!({"prompt": "p", "resolution": "480P"}), "resolution"),
    ] {
        let r = call(&f.app, "POST", T2V, Some(body.clone())).await;
        assert_eq!(r.status, StatusCode::UNPROCESSABLE_ENTITY, "{body}");
        assert_eq!(r.json()["detail"][0]["loc"], json!(["body", loc]), "{:?}", r.json());
        assert_eq!(r.header("x-fal-error-type"), Some("value_error"));
    }
    let r = send(&f.app, Request::post(T2V).header("authorization", format!("Key {KEY}")).body(Body::from("{not json")).unwrap()).await;
    assert_eq!(r.status, StatusCode::UNPROCESSABLE_ENTITY);

    // Loading: 503 runner_scheduling_failure + Retry-After.
    *f.gate.refuse.lock().unwrap() = Some(ApiError::loading("models are loading"));
    let r = call(&f.app, "POST", T2V, Some(json!({"prompt": "p"}))).await;
    assert_eq!(r.status, StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(r.json()["error_type"], "runner_scheduling_failure");
    assert_eq!(r.header("retry-after"), Some("1"));
    *f.gate.refuse.lock().unwrap() = None;
}

#[tokio::test]
async fn fal_max_queue_length() {
    let f = fixture(slow()).await;
    submit(&f.app, T2V, json!({"prompt": "a"})).await;
    submit(&f.app, T2V, json!({"prompt": "b"})).await;
    // At least one is waiting; a limit of 0 refuses.
    let r = call(&f.app, "POST", &format!("{T2V}?fal_max_queue_length=0"), Some(json!({"prompt": "c"}))).await;
    assert_eq!(r.status, StatusCode::TOO_MANY_REQUESTS, "{:?}", r.json());
    assert_eq!(r.json()["error_type"], "concurrent_requests_limit");
    assert_eq!(r.header("x-fal-needs-retry"), Some("1"));
    let r = call(&f.app, "POST", &format!("{T2V}?fal_max_queue_length=5"), Some(json!({"prompt": "c"}))).await;
    assert_eq!(r.status, StatusCode::OK);
    let r = call(&f.app, "POST", &format!("{T2V}?fal_max_queue_length=x"), Some(json!({"prompt": "c"}))).await;
    assert_eq!(r.status, StatusCode::UNPROCESSABLE_ENTITY);
    assert_eq!(r.json()["detail"][0]["loc"], json!(["query", "fal_max_queue_length"]));
}

fn header<'a>(h: &'a [(String, String)], k: &str) -> &'a str {
    h.iter().find(|(n, _)| n == k).map(|(_, v)| v.as_str()).unwrap_or_else(|| panic!("{k} missing"))
}

#[tokio::test]
async fn webhooks_are_signed_with_our_key() {
    let f = fixture(Opts::default()).await;
    let jwks = call(&f.app, "GET", "/.well-known/jwks.json", None).await;
    assert_eq!(jwks.status, StatusCode::OK);
    let k = &jwks.json()["keys"][0];
    assert_eq!((k["kty"].as_str(), k["crv"].as_str()), (Some("OKP"), Some("Ed25519")));
    let x = base64::engine::general_purpose::URL_SAFE_NO_PAD.decode(k["x"].as_str().unwrap()).unwrap();
    let key = VerifyingKey::from_bytes(&x.try_into().unwrap()).unwrap();

    let hook = "https://hooks.example.com/fal?user=1";
    let ok = submit(&f.app, &format!("{T2V}?fal_webhook={}", urlencode(hook)), json!({"prompt": "p"})).await;
    let bad = submit(&f.app, &format!("{T2V}?fal_webhook={}", urlencode(hook)), json!({"prompt": "[fake:fail]"})).await;
    let plain = submit(&f.app, T2V, json!({"prompt": "no hook"})).await;
    for s in [&ok, &bad, &plain] {
        wait_completed(&f.app, &path_of(s["status_url"].as_str().unwrap())).await;
    }
    let deadline = tokio::time::Instant::now() + T;
    while f.hooks.0.lock().unwrap().len() < 2 {
        assert!(tokio::time::Instant::now() < deadline, "webhooks not delivered");
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    tokio::time::sleep(Duration::from_millis(50)).await;
    let posts = f.hooks.0.lock().unwrap().clone();
    assert_eq!(posts.len(), 2, "one POST per webhook job, on completion only");
    for (url, headers, body) in &posts {
        assert_eq!(url.as_str(), hook);
        let v: Value = serde_json::from_slice(body).unwrap();
        let rid = v["request_id"].as_str().unwrap();
        assert_eq!(v["gateway_request_id"], rid);
        assert_eq!(header(headers, "x-fal-webhook-request-id"), rid);
        assert_eq!(header(headers, "x-fal-webhook-user-id"), "fv-test-user");
        let ts = header(headers, "x-fal-webhook-timestamp");
        let now = time::OffsetDateTime::now_utc().unix_timestamp();
        assert!((ts.parse::<i64>().unwrap() - now).abs() < 300);
        let msg = format!("{rid}\nfv-test-user\n{ts}\n{}", hex(&Sha256::digest(body)));
        let sig = unhex(header(headers, "x-fal-webhook-signature"));
        key.verify(msg.as_bytes(), &Signature::from_slice(&sig).unwrap()).expect("signature verifies against our JWKS");
        if rid == ok["request_id"] {
            assert_eq!(v["status"], "OK");
            assert!(v["payload"]["video"]["url"].is_string());
        } else {
            assert_eq!(rid, bad["request_id"]);
            assert_eq!(v["status"], "ERROR");
            assert_eq!(v["error"], "Invalid status code: 500");
            assert!(v["payload"]["detail"].is_array());
        }
    }
    // An invalid webhook URL is refused up front.
    let r = call(&f.app, "POST", &format!("{T2V}?fal_webhook=nope"), Some(json!({"prompt": "p"}))).await;
    assert_eq!(r.status, StatusCode::UNPROCESSABLE_ENTITY);
}

#[tokio::test]
async fn mp4_output_matches_hosted_h3() {
    if ffmpeg().is_none() {
        eprintln!("skipped: no ffmpeg (set FV_FFMPEG) for the fake engine's MP4 writer");
        return;
    }
    let f = fixture(Opts { engine: Engine::Fake, mp4: true, ..Opts::default() }).await;
    let s = submit(&f.app, T2V, json!({"prompt": "p"})).await;
    wait_completed(&f.app, &path_of(s["status_url"].as_str().unwrap())).await;
    let out = call(&f.app, "GET", &path_of(s["response_url"].as_str().unwrap()), None).await.json();
    let r = send(&f.app, Request::get(path_of(out["video"]["url"].as_str().unwrap())).body(Body::empty()).unwrap()).await;
    let p = f.dir.join("check.mp4");
    std::fs::write(&p, &r.body).unwrap();
    let info = fastvideo_media::mp4::inspect(&p).unwrap();
    assert_eq!(info.fal_h3_problems(), Vec::<String>::new(), "{info:?}");
    let v = info.video().unwrap();
    assert_eq!((v.width, v.height, v.samples), (Some(1344), Some(768), 124));
}

fn urlencode(s: &str) -> String {
    s.bytes()
        .map(|b| match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => (b as char).to_string(),
            _ => format!("%{b:02X}"),
        })
        .collect()
}

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

fn unhex(s: &str) -> Vec<u8> {
    (0..s.len()).step_by(2).map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap()).collect()
}

/// fal's own app ids with multi-segment subs (docs/serve/fal-parity.md P0):
/// `lightricks/ltx-2.5` and `fal-ai/wan` pick their tier per endpoint, the
/// status and result URLs are app-only, the output is named after the app,
/// and `minimax/h3` / `minimax/h3-max-turbo` alias the H3 tiers.
#[tokio::test]
async fn family_apps_and_multi_segment_subs() {
    use fastvideo_engine_service::cuda::caps::{catalog, WeightLayout};
    use fastvideo_engine_service::FakeModel;
    use fastvideo_protocol::{Family, Tier};
    let models: Vec<FakeModel> = catalog(&WeightLayout::default())
        .into_iter()
        .filter(|m| matches!(m.tier, Some(Tier::Max | Tier::Turbo)))
        .filter(|m| m.family() != Family::H3 || m.tier == Some(Tier::Turbo))
        .map(|m| FakeModel { caps: m.caps(), recipe: m.describe() })
        .collect();
    let f = fixture(Opts {
        models,
        fal_apps: vec!["lightricks/ltx-2.5", "fal-ai/wan", "minimax/h3-max-turbo", "minimax/h3"],
        ..Opts::default()
    })
    .await;
    for (path, app, body, model, dims, slug) in [
        (
            "/lightricks/ltx-2.5/text-to-video/fast",
            "lightricks/ltx-2.5",
            json!({"prompt": "p", "resolution": "720p", "fps": 24, "duration": 8}),
            "ltx25-distill-sol",
            (1280, 768, 193, 24),
            "ltx-2.5",
        ),
        (
            "/lightricks/ltx-2.5/text-to-video/pro",
            "lightricks/ltx-2.5",
            json!({"prompt": "p", "aspect_ratio": "9:16"}),
            "ltx25-distill-dense",
            (1088, 1920, 153, 25),
            "ltx-2.5",
        ),
        (
            "/fal-ai/wan/v2.2-5b/text-to-video",
            "fal-ai/wan",
            json!({"prompt": "p", "resolution": "580p", "num_frames": 161, "frames_per_second": 30}),
            "wan22-ti2v-5b",
            (1024, 576, 161, 30),
            "wan",
        ),
        (
            "/fal-ai/wan/v2.2-5b/text-to-video/fast-wan",
            "fal-ai/wan",
            json!({"prompt": "p", "resolution": "480p", "aspect_ratio": "1:1"}),
            "fastwan22-ti2v-5b",
            (480, 480, 81, 24),
            "wan",
        ),
        (
            "/minimax/h3-max-turbo/text-to-video",
            "minimax/h3-max-turbo",
            json!({"prompt": "p"}),
            "fasth3-4step-vsa",
            (1344, 768, 124, 24),
            "minimax-h3-max-turbo",
        ),
    ] {
        let s = submit(&f.app, path, body.clone()).await;
        let rid = s["request_id"].as_str().unwrap().to_owned();
        assert_eq!(s["status_url"], format!("https://fal.fv.test/{app}/requests/{rid}/status"), "{path}");
        let j = f.ctx.jobs().by_external(ProtocolId::Fal, &rid).await.unwrap();
        assert_eq!(j.resolved.model.0, model, "{path}");
        let r = &j.resolved;
        assert_eq!((r.width, r.height, r.num_frames, r.fps), dims, "{path}");
        assert_eq!(j.requested_model(), &path[1..]);
        wait_completed(&f.app, &path_of(s["status_url"].as_str().unwrap())).await;
        // App-only and full-endpoint forms.
        for p in [format!("/{app}/requests/{rid}"), format!("{path}/requests/{rid}/status")] {
            assert_eq!(call(&f.app, "GET", &p, None).await.status, StatusCode::OK, "{p}");
        }
        let out = call(&f.app, "GET", &format!("/{app}/requests/{rid}"), None).await.json();
        let name = out["video"]["file_name"].as_str().unwrap();
        // `<nanoid21>_<app slug>[-<tier>].mp4`: named by app and tier.
        let named = name.strip_suffix(".mp4").and_then(|n| n.get(22..)).unwrap_or_default();
        assert!(named.starts_with(slug), "{name}");
        if let Some(t) = j.resolved.tier {
            assert!(named.split(['-', '_', '.']).any(|w| w == t.as_str()), "{name} names no tier");
        }
        // Another app's prefix does not find it.
        let other = if app == "fal-ai/wan" { "lightricks/ltx-2.5" } else { "fal-ai/wan" };
        assert_eq!(call(&f.app, "GET", &format!("/{other}/requests/{rid}/status"), None).await.status, StatusCode::NOT_FOUND);
    }
    // The sync route with a multi-segment sub.
    let r = call(&f.app, "POST", "/run/fal-ai/wan/v2.2-5b/text-to-video/fast-wan", Some(json!({"prompt": "p", "num_frames": 17}))).await;
    assert_eq!(r.status, StatusCode::OK, "{:?}", r.json());
    let name = r.json()["video"]["file_name"].as_str().unwrap().to_owned();
    assert!(name.contains("_wan") && name.ends_with(".mp4"), "{name}");
    // Validation names fal's fields; the base H3 app refuses 2K / 4K cleanly.
    for (path, body, loc) in [
        ("/lightricks/ltx-2.5/text-to-video/fast", json!({"prompt": "p", "duration": 20}), "duration"),
        ("/lightricks/ltx-2.5/text-to-video/pro", json!({"prompt": "p", "resolution": "2160p"}), "resolution"),
        ("/lightricks/ltx-2.5/image-to-video/fast", json!({"prompt": "p"}), "image_url"),
        ("/fal-ai/wan/v2.2-5b/text-to-video", json!({"prompt": "p", "frames_per_second": 61}), "frames_per_second"),
        ("/fal-ai/wan/v2.2-5b/text-to-video", json!({"prompt": "p", "num_frames": 162}), "num_frames"),
        ("/minimax/h3-max-turbo/text-to-video", json!({"prompt": "p", "resolution": "2K"}), "resolution"),
    ] {
        let r = call(&f.app, "POST", path, Some(body.clone())).await;
        assert_eq!(r.status, StatusCode::UNPROCESSABLE_ENTITY, "{path} {body}");
        assert_eq!(r.json()["detail"][0]["loc"], json!(["body", loc]), "{path} {body}: {:?}", r.json());
    }
    // `minimax/h3` runs on the Max tier, absent from this fixture: a clean 404.
    let r = call(&f.app, "POST", "/minimax/h3/text-to-video", Some(json!({"prompt": "p"}))).await;
    assert_eq!(r.status, StatusCode::NOT_FOUND);
    // The family apps have only their own subs.
    for p in ["/lightricks/ltx-2.5/text-to-video", "/fal-ai/wan/text-to-video", "/fal-ai/wan/v2.2-5b/reference-to-video"] {
        let r = call(&f.app, "POST", p, Some(json!({"prompt": "p"}))).await;
        assert!(matches!(r.status.as_u16(), 404 | 405), "{p}: {}", r.status);
    }
    // Schemas for the console, by multi-segment sub.
    let r = call(&f.app, "GET", "/fal/schema/fal-ai/wan/v2.2-5b/text-to-video/fast-wan", None).await;
    assert_eq!(r.status, StatusCode::OK);
    assert!(r.json()["properties"].get("num_inference_steps").is_none());
    let r = call(&f.app, "GET", "/fal/schema/lightricks/ltx-2.5/image-to-video/pro", None).await;
    assert_eq!(r.json()["properties"]["resolution"]["enum"], json!(["720p", "1080p"]));
    assert_eq!(call(&f.app, "GET", "/fal/schema/lightricks/ltx-2.5/reference-to-video", None).await.status, StatusCode::NOT_FOUND);
    let c = call(&f.app, "GET", "/fal/schema", None).await.json();
    assert_eq!(c["apps"][1]["endpoints"][2]["endpoint_id"], "fal-ai/wan/v2.2-5b/text-to-video/fast-wan");
}

/// `minimax/h3` (base) on the Max tier: 768P runs, 2K and 4K are the
/// `H3Resolution2K` gap (422 on `resolution`).
#[tokio::test]
async fn base_h3_app() {
    let f = fixture(Opts { fal_apps: vec!["minimax/h3", "minimax/h3-max-turbo"], ..Opts::default() }).await;
    let s = submit(&f.app, "/minimax/h3/text-to-video", json!({"prompt": "p", "resolution": "768P"})).await;
    let j = f.ctx.jobs().by_external(ProtocolId::Fal, s["request_id"].as_str().unwrap()).await.unwrap();
    assert_eq!(j.resolved.model.0, "fake-h3-max");
    let s = submit(&f.app, "/minimax/h3-max-turbo/image-to-video", json!({"prompt": "p"})).await;
    let j = f.ctx.jobs().by_external(ProtocolId::Fal, s["request_id"].as_str().unwrap()).await.unwrap();
    assert_eq!(j.resolved.model.0, "fake-h3-turbo");
    for res in ["2K", "4K"] {
        let r = call(&f.app, "POST", "/minimax/h3/text-to-video", Some(json!({"prompt": "p", "resolution": res}))).await;
        assert_eq!(r.status, StatusCode::UNPROCESSABLE_ENTITY, "{res}");
        let d = &r.json()["detail"][0];
        assert_eq!(d["loc"], json!(["body", "resolution"]), "{res}");
        assert_eq!(d["msg"], "2K resolution is not supported by this server", "{res}");
    }
}
