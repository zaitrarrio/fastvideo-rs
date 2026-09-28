//! End to end: the real `EngineService` with the fake backend, serve-kit's
//! `ServeCtx` (job store, artifacts, ingestion, generic handlers) and this
//! crate's routes, driven over HTTP the way clients call them.

mod common;

use std::sync::Arc;
use std::time::Duration;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use common::*;
use fastvideo_engine_service::ManualClock;
use fastvideo_protocol::{ProtocolId, Task};
use serde_json::json;

fn done(v: &serde_json::Value) -> bool {
    matches!(v["status"].as_str(), Some("completed" | "failed"))
}

// ---------------------------------------------------------------- FastVideo

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn fastvideo_submit_poll_download_delete() {
    let clock = Arc::new(ManualClock::new());
    let f = fixture(Opts {
        clock: Some(clock.clone()),
        ..Opts::default()
    })
    .await;
    // A job behind a running one is `queued` (a lone job may already be
    // `in_progress` when the create response is read).
    hold_executor(&f, &clock).await;
    // The tier id resolves to the turbo model; defaults: 1344x768, 5 s.
    let r = call(
        &f.app,
        "POST",
        "/v1/videos",
        Some(json!({"model": "h3-turbo", "prompt": "a red fox in snow", "seed": 7})),
    )
    .await;
    assert_eq!(r.status, StatusCode::OK, "{}", r.json());
    let v = r.json();
    assert_eq!(
        (
            v["object"].as_str(),
            v["status"].as_str(),
            v["model"].as_str()
        ),
        (Some("video"), Some("queued"), Some("h3-turbo"))
    );
    assert_eq!(
        (v["size"].as_str(), v["seconds"].as_str()),
        (Some("1344x768"), Some("5"))
    );
    assert_eq!(
        v["metadata"],
        json!({"tier": "turbo", "quality_gate": true, "recipe": "4step-vsa"})
    );
    let id = v["id"].as_str().unwrap().to_owned();
    assert!(id.starts_with("video_gen_") && id.len() == 42);
    let job = f
        .ctx
        .jobs()
        .by_external(ProtocolId::OpenAiVideos, &id)
        .await
        .unwrap();
    assert_eq!(
        (
            job.resolved.model.0.as_str(),
            job.resolved.num_frames,
            job.resolved.seed
        ),
        ("fake-h3-turbo", 124, 7)
    );

    let _run = ClockDriver::start(clock, Duration::from_millis(5));
    let v = poll(&f.app, &format!("/v1/videos/{id}"), done).await;
    assert_eq!(
        (v["status"].as_str(), v["progress"].as_u64()),
        (Some("completed"), Some(100)),
        "{v}"
    );
    assert!(v["completed_at"].is_i64());
    assert!(v["stage_durations"]["denoise"].is_f64());
    assert_eq!(v["file_name"], "video.mp4");

    let c = call(&f.app, "GET", &format!("/v1/videos/{id}/content"), None).await;
    assert_eq!(c.status, StatusCode::OK);
    assert_eq!(c.headers["content-type"], "video/mp4");
    assert_eq!(&c.bytes[4..8], b"ftyp");
    let c = call(
        &f.app,
        "GET",
        &format!("/v1/videos/{id}/content?variant=video"),
        None,
    )
    .await;
    assert_eq!(c.status, StatusCode::OK);
    let c = call(
        &f.app,
        "GET",
        &format!("/v1/videos/{id}/content?variant=thumbnail"),
        None,
    )
    .await;
    assert_eq!(
        (c.status, c.json()["error"]["param"].as_str()),
        (StatusCode::BAD_REQUEST, Some("variant"))
    );

    let l = call(&f.app, "GET", "/v1/videos", None).await.json();
    assert_eq!(
        (
            l["object"].as_str(),
            l["first_id"].as_str(),
            l["has_more"].as_bool()
        ),
        (Some("list"), Some(id.as_str()), Some(false))
    );

    let d = call(&f.app, "DELETE", &format!("/v1/videos/{id}"), None).await;
    assert_eq!(d.status, StatusCode::OK);
    assert_eq!(
        d.json(),
        json!({"id": id, "deleted": true, "object": "video.deleted"})
    );
    let g = call(&f.app, "GET", &format!("/v1/videos/{id}"), None).await;
    assert_eq!(
        (g.status, g.json()["error"]["code"].as_u64()),
        (StatusCode::NOT_FOUND, Some(404))
    );
    assert_eq!(
        call(&f.app, "DELETE", &format!("/v1/videos/{id}"), None)
            .await
            .status,
        StatusCode::NOT_FOUND
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn fastvideo_sync_returns_mp4_with_metrics_headers() {
    let f = fixture(Opts::default()).await;
    let r = call(&f.app, "POST", "/v1/videos/sync", Some(json!({"model": "fake-wan", "prompt": "waves", "size": "832x480", "num_frames": 49, "fps": 16}))).await;
    assert_eq!(
        r.status,
        StatusCode::OK,
        "{}",
        String::from_utf8_lossy(&r.bytes)
    );
    assert_eq!(r.headers["content-type"], "video/mp4");
    assert_eq!(&r.bytes[4..8], b"ftyp");
    assert!(String::from_utf8_lossy(&r.bytes).ends_with("832x480x49@16"));
    let rid = r.headers["x-request-id"].to_str().unwrap();
    assert!(rid.starts_with("video_gen_"));
    assert_eq!(r.headers["x-model"], "fake-wan");
    assert!(r.headers["x-inference-time-s"]
        .to_str()
        .unwrap()
        .parse::<f64>()
        .is_ok());
    let stages: serde_json::Value =
        serde_json::from_str(r.headers["x-stage-durations"].to_str().unwrap()).unwrap();
    assert!(stages["denoise"].is_f64());
    assert_eq!(r.headers["x-fv-recipe"], "dmd-3step");
    // The temporary result is removed after sending.
    assert!(f
        .ctx
        .jobs()
        .by_external(ProtocolId::OpenAiVideos, rid)
        .await
        .is_none());

    // A failing generation answers a 500 server_error envelope.
    let r = call(&f.app, "POST", "/v1/videos/sync", Some(json!({"model": "fake-wan", "prompt": "[fake:fail]", "size": "832x480", "num_frames": 49, "fps": 16}))).await;
    assert_eq!(r.status, StatusCode::INTERNAL_SERVER_ERROR);
    assert_eq!(r.json()["error"]["type"], "server_error");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn fastvideo_failed_job_is_200_failed_and_content_422() {
    let f = fixture(Opts::default()).await;
    let r = call(&f.app, "POST", "/v1/videos/generations", Some(json!({"model": "fake-wan", "prompt": "[fake:fail] boom", "size": "832x480", "num_frames": 49, "fps": 16}))).await;
    assert_eq!(r.status, StatusCode::OK, "{}", r.json());
    let id = r.json()["id"].as_str().unwrap().to_owned();
    let v = poll(&f.app, &format!("/v1/videos/{id}"), done).await;
    assert_eq!(v["status"], "failed");
    assert_eq!(v["error"]["code"], "generation_failed");
    assert!(v["error"]["message"]
        .as_str()
        .unwrap()
        .contains("injected failure"));
    let c = call(&f.app, "GET", &format!("/v1/videos/{id}/content"), None).await;
    assert_eq!(
        (c.status, c.json()["error"]["code"].as_u64()),
        (StatusCode::UNPROCESSABLE_ENTITY, Some(422))
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn fastvideo_multipart_upload_in_progress_and_cancel() {
    // Slow steps keep the job running while we look at it.
    let f = fixture(Opts {
        step: Duration::from_millis(400),
        ..Opts::default()
    })
    .await;
    let boundary = "fvboundary";
    let png = png(64, 36);
    let mut body = Vec::new();
    for (k, v) in [
        ("model", "h3-turbo"),
        ("prompt", "the photo comes alive"),
        ("seconds", "5"),
    ] {
        body.extend_from_slice(
            format!("--{boundary}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n{v}\r\n")
                .as_bytes(),
        );
    }
    body.extend_from_slice(
        format!("--{boundary}\r\nContent-Disposition: form-data; name=\"input_reference\"; filename=\"a.png\"\r\nContent-Type: image/png\r\n\r\n").as_bytes(),
    );
    body.extend_from_slice(&png);
    body.extend_from_slice(format!("\r\n--{boundary}--\r\n").as_bytes());
    let req = Request::post("/v1/videos")
        .header(
            "content-type",
            format!("multipart/form-data; boundary={boundary}"),
        )
        .body(Body::from(body))
        .unwrap();
    let r = send(&f.app, req).await;
    assert_eq!(r.status, StatusCode::OK, "{}", r.json());
    let id = r.json()["id"].as_str().unwrap().to_owned();
    let job = f
        .ctx
        .jobs()
        .by_external(ProtocolId::OpenAiVideos, &id)
        .await
        .unwrap();
    assert_eq!(job.resolved.task, Task::I2V);
    assert!(
        job.resolved.keyframes[0]
            .1
            .starts_with(f.ctx.inputs_dir(job.id)),
        "the upload was staged"
    );

    let v = poll(&f.app, &format!("/v1/videos/{id}"), |v| {
        v["status"] == "in_progress"
    })
    .await;
    assert_eq!(v["status"], "in_progress");
    let c = call(&f.app, "GET", &format!("/v1/videos/{id}/content"), None).await;
    assert_eq!(
        (c.status, c.json()["error"]["message"].as_str()),
        (
            StatusCode::NOT_FOUND,
            Some("Generation is still in-progress")
        )
    );

    // DELETE cancels the running generation and removes the job.
    let d = call(&f.app, "DELETE", &format!("/v1/videos/{id}"), None).await;
    assert_eq!(d.status, StatusCode::OK);
    assert!(f.ctx.jobs().get(job.id).await.is_none());
    assert_eq!(
        call(&f.app, "GET", &format!("/v1/videos/{id}"), None)
            .await
            .status,
        StatusCode::NOT_FOUND
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn fastvideo_form_body() {
    let f = fixture(Opts::default()).await;
    let form = "model=fake-wan&prompt=a+quiet+lake&width=832&height=480&num_frames=49&fps=16&seed=3&extra_body=%7B%22negative_prompt%22%3A%22blur%22%7D";
    let req = Request::post("/v1/videos")
        .header("content-type", "application/x-www-form-urlencoded")
        .body(Body::from(form))
        .unwrap();
    let r = send(&f.app, req).await;
    assert_eq!(r.status, StatusCode::OK, "{}", r.json());
    let job = f
        .ctx
        .jobs()
        .by_external(ProtocolId::OpenAiVideos, r.json()["id"].as_str().unwrap())
        .await
        .unwrap();
    assert_eq!(
        (
            job.resolved.width,
            job.resolved.num_frames,
            job.resolved.seed
        ),
        (832, 49, 3)
    );
    assert_eq!(job.resolved.negative_prompt, "blur");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn fastvideo_error_mapping_over_http() {
    let f = fixture(Opts::default()).await;
    let cases: Vec<(serde_json::Value, u16, Option<&str>)> = vec![
        (
            json!({"model": "sora-2", "prompt": "p"}),
            400,
            Some("model"),
        ),
        (
            json!({"model": "h3-turbo", "prompt": "p", "bogus": true}),
            400,
            Some("bogus"),
        ),
        (
            json!({"model": "h3-turbo", "prompt": "p", "fps": 30}),
            400,
            Some("fps"),
        ),
        (
            json!({"model": "h3-turbo", "prompt": "p", "num_frames": 125}),
            400,
            Some("num_frames"),
        ),
        (
            json!({"model": "h3-turbo", "prompt": "p", "seconds": 4}),
            400,
            Some("seconds"),
        ),
        (
            json!({"model": "h3-turbo", "prompt": "p", "seconds": 16}),
            400,
            Some("seconds"),
        ),
        (
            json!({"model": "h3-turbo", "prompt": "p", "size": "1344x770"}),
            400,
            Some("size"),
        ),
        (
            json!({"model": "h3-turbo", "prompt": "p", "num_inference_steps": 8}),
            400,
            Some("num_inference_steps"),
        ),
        (
            json!({"model": "h3-turbo", "prompt": "p", "guidance_scale": 3.0}),
            400,
            Some("guidance_scale"),
        ),
        (
            json!({"model": "h3-turbo", "prompt": "p", "image_reference": {"file_id": "file-1"}}),
            400,
            None,
        ),
        (
            json!({"model": "h3-turbo", "prompt": "p", "task": "ref2va", "image_reference": {"image_url": "https://x.test/a.png"}}),
            400,
            None,
        ),
        (
            json!({"model": "fake-wan", "prompt": "p", "task": "t2va"}),
            400,
            Some("task"),
        ),
        (
            json!({"model": "h3-turbo", "prompt": "p", "n": 2}),
            400,
            Some("n"),
        ),
        (
            json!({"model": "h3-turbo", "prompt": "p", "image_reference": {"image_url": "data:image/png;base64,AAAA"}}),
            415,
            None,
        ),
    ];
    for (body, status, param) in cases {
        let r = call(&f.app, "POST", "/v1/videos", Some(body.clone())).await;
        let v = r.json();
        assert_eq!(r.status.as_u16(), status, "{body}: {v}");
        assert_eq!(
            v["error"]["code"].as_u64(),
            Some(status as u64),
            "{body}: {v}"
        );
        assert_eq!(v["error"]["type"], "invalid_request_error", "{body}");
        if let Some(p) = param {
            assert_eq!(v["error"]["param"].as_str(), Some(p), "{body}: {v}");
        }
    }
    // FastH3's fixed guidance 1 is accepted.
    let r = call(&f.app, "POST", "/v1/videos", Some(json!({"model": "h3-turbo", "prompt": "p", "guidance_scale": 1.0, "negative_prompt": "x"}))).await;
    assert_eq!(r.status, StatusCode::OK, "{}", r.json());
    // ref2va runs on the max tier, whose model has it resident.
    let r = call(&f.app, "POST", "/v1/videos", Some(json!({"model": "h3-max", "prompt": "p", "task": "ref2va", "image_reference": {"image_url": format!("data:image/png;base64,{}", b64(&png(64, 36)))}}))).await;
    assert_eq!(r.status, StatusCode::OK, "{}", r.json());
    let r = call(&f.app, "GET", "/v1/videos?limit=0", None).await;
    assert_eq!(
        (r.status, r.json()["error"]["param"].as_str()),
        (StatusCode::BAD_REQUEST, Some("limit"))
    );
    let r = call(&f.app, "GET", "/v1/videos/video_gen_nope", None).await;
    assert_eq!(r.status, StatusCode::NOT_FOUND);
    let r = send(
        &f.app,
        Request::post("/v1/videos")
            .header("content-type", "application/json")
            .body(Body::from("{nope"))
            .unwrap(),
    )
    .await;
    assert_eq!(r.status, StatusCode::BAD_REQUEST);

    let f = fixture(Opts {
        loading: true,
        ..Opts::default()
    })
    .await;
    let r = call(&f.app, "POST", "/v1/videos", Some(json!({"prompt": "p"}))).await;
    assert_eq!(
        (r.status, r.json()["error"]["type"].as_str()),
        (StatusCode::SERVICE_UNAVAILABLE, Some("server_error"))
    );
    assert_eq!(r.headers["retry-after"], "1");
}

fn b64(b: &[u8]) -> String {
    use base64::Engine as _;
    base64::engine::general_purpose::STANDARD.encode(b)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn fastvideo_list_paging_and_owners() {
    let f = fixture(Opts {
        keys: true,
        step: Duration::from_millis(200),
        ..Opts::default()
    })
    .await;
    let mut ids = Vec::new();
    for i in 0..3 {
        let r = call_key(&f.app, "POST", "/v1/videos", Some("sk-a"), Some(json!({"model": "fake-wan", "prompt": format!("p{i}"), "size": "832x480", "num_frames": 49, "fps": 16}))).await;
        assert_eq!(r.status, StatusCode::OK, "{}", r.json());
        ids.push(r.json()["id"].as_str().unwrap().to_owned());
    }
    let other = call_key(&f.app, "POST", "/v1/videos", Some("sk-b"), Some(json!({"model": "fake-wan", "prompt": "b", "size": "832x480", "num_frames": 49, "fps": 16}))).await;
    let other = other.json()["id"].as_str().unwrap().to_owned();

    let l = call_key(&f.app, "GET", "/v1/videos?limit=2", Some("sk-a"), None)
        .await
        .json();
    let got: Vec<&str> = l["data"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v["id"].as_str().unwrap())
        .collect();
    assert_eq!(got, [ids[2].as_str(), ids[1].as_str()], "newest first");
    assert_eq!(
        (l["has_more"].as_bool(), l["last_id"].as_str()),
        (Some(true), Some(ids[1].as_str()))
    );
    let l = call_key(
        &f.app,
        "GET",
        &format!("/v1/videos?limit=2&after={}", ids[1]),
        Some("sk-a"),
        None,
    )
    .await
    .json();
    let got: Vec<&str> = l["data"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v["id"].as_str().unwrap())
        .collect();
    assert_eq!(
        (got, l["has_more"].as_bool()),
        (vec![ids[0].as_str()], Some(false))
    );
    let l = call_key(&f.app, "GET", "/v1/videos?order=asc", Some("sk-a"), None)
        .await
        .json();
    assert_eq!(l["first_id"].as_str(), Some(ids[0].as_str()));
    assert_eq!(
        l["data"].as_array().unwrap().len(),
        3,
        "other owners' jobs are not listed"
    );
    // Other owners' jobs are 404.
    assert_eq!(
        call_key(
            &f.app,
            "GET",
            &format!("/v1/videos/{other}"),
            Some("sk-a"),
            None
        )
        .await
        .status,
        StatusCode::NOT_FOUND
    );
    assert_eq!(
        call_key(
            &f.app,
            "DELETE",
            &format!("/v1/videos/{other}"),
            Some("sk-a"),
            None
        )
        .await
        .status,
        StatusCode::NOT_FOUND
    );
    // Anonymous callers see only anonymous jobs.
    let l = call(&f.app, "GET", "/v1/videos", None).await.json();
    assert_eq!(l["data"].as_array().unwrap().len(), 0);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn models_routes() {
    let f = fixture(Opts::default()).await;
    let v = call(&f.app, "GET", "/v1/models", None).await.json();
    let ids: Vec<&str> = v["data"]
        .as_array()
        .unwrap()
        .iter()
        .map(|c| c["id"].as_str().unwrap())
        .collect();
    for want in [
        "fake-h3-turbo",
        "fake-h3-max",
        "fake-wan",
        "fastwan-5b",
        "h3-turbo",
        "h3-max",
        "wan-turbo",
    ] {
        assert!(ids.contains(&want), "{want} in {ids:?}");
    }
    let c = call(&f.app, "GET", "/v1/models/h3-max", None).await;
    assert_eq!(
        c.json(),
        json!({"id": "h3-max", "object": "model", "created": 1_700_000_000, "owned_by": "fastvideo", "root": "fake-h3-max"})
    );
    assert_eq!(
        call(&f.app, "GET", "/v1/models/sora-2", None).await.status,
        StatusCode::NOT_FOUND
    );
    let i = call(&f.app, "GET", "/v1/model_info", None).await.json();
    assert!(i["served_model_name"].is_string() && i["lora"].is_null());
}

// ---------------------------------------------------------------- FastWan

/// `fastwan_link.py` end to end: `_wait_reachable` (`/health`, `/`), then
/// `_generate` (`POST /generate`, poll `/status/{id}` while `queued` /
/// `processing`, `GET /video/{id}`), then `_delete_job`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn fastwan_link_flow() {
    let clock = Arc::new(ManualClock::new());
    let f = fixture(Opts {
        clock: Some(clock.clone()),
        ..Opts::default()
    })
    .await;
    let h = call(&f.app, "GET", "/health", None).await;
    assert_eq!(h.status, StatusCode::OK);
    assert_eq!(h.json()["model_loaded"], true);
    let root = call(&f.app, "GET", "/", None).await.json();
    assert_eq!(root["model"], "fastwan-5b");

    // Another client's clip holds the executor, so this one is first seen
    // `queued`, deterministically.
    hold_executor(&f, &clock).await;
    let body = json!({"prompt": "a neon street in the rain", "width": 640, "height": 352, "num_frames": 49, "fps": 24, "seed": 1000});
    let r = call(&f.app, "POST", "/generate", Some(body)).await;
    assert_eq!(r.status, StatusCode::OK, "{}", r.json());
    let mut job = r.json();
    let pid = job["prompt_id"].as_str().expect("prompt_id").to_owned();
    assert!(uuid::Uuid::parse_str(&pid).is_ok());
    assert_eq!(job["status"], "queued", "{job}");
    let _run = ClockDriver::start(clock, Duration::from_millis(5));
    // The client's loop, verbatim in spirit.
    let deadline = tokio::time::Instant::now() + T;
    let mut seen = Vec::new();
    while job["status"] != "completed" {
        let s = job["status"].as_str().unwrap().to_owned();
        assert_ne!(s, "failed", "{job}");
        assert!(
            ["queued", "processing"].contains(&s.as_str()),
            "unknown status {s}"
        );
        seen.push(s);
        assert!(tokio::time::Instant::now() < deadline, "no result after {T:?}: {job}");
        tokio::time::sleep(Duration::from_millis(10)).await;
        let r = call(&f.app, "GET", &format!("/status/{pid}"), None).await;
        assert_eq!(r.status, StatusCode::OK);
        job = r.json();
    }
    assert_eq!(seen.first().map(String::as_str), Some("queued"));
    let v = call(&f.app, "GET", &format!("/video/{pid}"), None).await;
    assert_eq!(
        (v.status, v.headers["content-type"].to_str().unwrap()),
        (StatusCode::OK, "video/mp4")
    );
    assert!(String::from_utf8_lossy(&v.bytes).ends_with("640x352x49@24"));
    let stored = f
        .ctx
        .jobs()
        .by_external(ProtocolId::FastWan, &pid)
        .await
        .unwrap();
    assert_eq!(
        (stored.resolved.model.0.as_str(), stored.resolved.seed),
        ("fake-fastwan", 1000)
    );

    let d = call(&f.app, "DELETE", &format!("/video/{pid}"), None).await;
    assert_eq!((d.status, d.json()), (StatusCode::OK, json!({})));
    let s = call(&f.app, "GET", &format!("/status/{pid}"), None).await;
    assert_eq!(s.status, StatusCode::NOT_FOUND);
    assert!(s.json()["detail"].is_string(), "FastAPI detail string");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn fastwan_rejections_failures_and_loading() {
    let f = fixture(Opts::default()).await;
    // Rejections the client treats as a failed clip (400/413/415/422).
    for body in [
        json!({"prompt": "p", "width": 640, "height": 352, "num_frames": 50, "fps": 24, "seed": 1}),
        json!({"prompt": "p", "width": 641, "height": 352, "num_frames": 49, "fps": 24, "seed": 1}),
        json!({"prompt": "p", "width": 640, "height": 352, "num_frames": 49, "fps": 30, "seed": 1}),
        json!({"prompt": "", "num_frames": 49}),
        json!({"width": 640}),
    ] {
        let r = call(&f.app, "POST", "/generate", Some(body.clone())).await;
        assert_eq!(r.status, StatusCode::BAD_REQUEST, "{body}: {}", r.json());
        assert!(r.json()["detail"].is_string(), "{body}");
    }
    // A failing job: status `failed` with a string error; the video is 422.
    let r = call(&f.app, "POST", "/generate", Some(json!({"prompt": "[fake:fail]", "width": 640, "height": 352, "num_frames": 49, "fps": 24, "seed": 1}))).await;
    let pid = r.json()["prompt_id"].as_str().unwrap().to_owned();
    let v = poll(&f.app, &format!("/status/{pid}"), done).await;
    assert_eq!(v["status"], "failed");
    assert!(v["error"].as_str().unwrap().contains("injected failure"));
    assert_eq!(
        call(&f.app, "GET", &format!("/video/{pid}"), None)
            .await
            .status,
        StatusCode::UNPROCESSABLE_ENTITY
    );
    assert_eq!(
        call(
            &f.app,
            "GET",
            "/video/0e7a5f7c-4d7b-4f43-9d3a-6f0d0c1c2b3a",
            None
        )
        .await
        .status,
        StatusCode::NOT_FOUND
    );

    // Loading: /health says not loaded, /generate is 503 (the client retries).
    let f = fixture(Opts {
        loading: true,
        ..Opts::default()
    })
    .await;
    let h = call(&f.app, "GET", "/health", None).await;
    assert_eq!(
        (h.status, h.json()["model_loaded"].as_bool()),
        (StatusCode::SERVICE_UNAVAILABLE, Some(false))
    );
    let r = call(
        &f.app,
        "POST",
        "/generate",
        Some(json!({"prompt": "p", "num_frames": 49})),
    )
    .await;
    assert_eq!(r.status, StatusCode::SERVICE_UNAVAILABLE);
    assert!(r.json()["detail"].is_string());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn fastwan_delete_cancels_running_job() {
    let f = fixture(Opts {
        step: Duration::from_millis(400),
        ..Opts::default()
    })
    .await;
    let r = call(&f.app, "POST", "/generate", Some(json!({"prompt": "slow", "width": 640, "height": 352, "num_frames": 49, "fps": 24, "seed": 1}))).await;
    let pid = r.json()["prompt_id"].as_str().unwrap().to_owned();
    poll(&f.app, &format!("/status/{pid}"), |v| {
        v["status"] == "processing"
    })
    .await;
    assert_eq!(
        call(&f.app, "GET", &format!("/video/{pid}"), None)
            .await
            .status,
        StatusCode::NOT_FOUND
    );
    let id = f
        .ctx
        .jobs()
        .by_external(ProtocolId::FastWan, &pid)
        .await
        .unwrap()
        .id;
    let d = call(&f.app, "DELETE", &format!("/video/{pid}"), None).await;
    assert_eq!(d.status, StatusCode::OK);
    assert!(f.ctx.jobs().get(id).await.is_none());
    // The engine stops the running job at its next step.
    let idle = async {
        while f.bridge.engine.stats().running > 0 {
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    };
    tokio::time::timeout(Duration::from_secs(5), idle)
        .await
        .expect("the engine stopped the job");
}
