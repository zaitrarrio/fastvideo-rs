//! End to end: the MiniMax router over serve-kit, the engine service with
//! the fake backend, and a real local HTTP callback receiver.

mod common;

use std::sync::atomic::Ordering;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use common::*;
use fastvideo_minimax::MiniMaxConfig;
use fastvideo_protocol::{JobStatus, ProtocolId, Task};
use fastvideo_serve_kit::events::{apply_event, JobEvent};
use serde_json::json;
use tower::ServiceExt;

async fn job_of(f: &Fixture, id: &str) -> fastvideo_protocol::Job {
    f.ctx.jobs().by_external(ProtocolId::MiniMaxV2, id).await.expect("job stored")
}

#[tokio::test(flavor = "multi_thread")]
async fn t2v_create_query_download_with_callbacks() {
    let f = fixture().await;
    let mut body = t2v("MiniMax-H3", 5);
    body["callback_url"] = json!(f.rx.url("/cb"));
    let id = create_ok(&f, body).await;
    assert!(id.len() == 18 && id.bytes().all(|b| b.is_ascii_digit()));

    let t = wait_done(&f, &id).await;
    assert_eq!(t["status"], "succeeded", "{t}");
    assert_eq!((t["id"].as_str(), t["model"].as_str()), (Some(id.as_str()), Some("MiniMax-H3")));
    assert_eq!((t["resolution"].as_str(), t["duration"].as_u64(), t["ratio"].as_str()), (Some("768P"), Some(5), Some("16:9")));
    assert_eq!(t["usage"], json!({"total_seconds": 5, "input_seconds": 0, "output_seconds": 5, "input_image_count": 0}));
    assert!(t.get("error").is_none());
    assert_eq!(t["metadata"]["tier"], "max");

    let j = job_of(&f, &id).await;
    assert_eq!(j.resolved.model.0, "fake-h3-max", "MiniMax-H3 -> the max tier by default");
    assert_eq!((j.resolved.width, j.resolved.height, j.resolved.num_frames, j.resolved.fps), (1344, 768, 124, 24));
    assert_eq!(j.expires_at - j.created_at, time::Duration::days(7));

    // content.url downloads without a key and is re-signed per query.
    let u = url::Url::parse(t["content"]["url"].as_str().unwrap()).unwrap();
    let r = f.app.clone().oneshot(Request::get(format!("{}?{}", u.path(), u.query().unwrap())).body(Body::empty()).unwrap()).await.unwrap();
    assert_eq!(r.status(), StatusCode::OK);

    // Callbacks: challenge first, then one POST per status change.
    let seen = f.rx.wait_terminal("/cb").await;
    assert_eq!(seen[0].as_object().unwrap().keys().collect::<Vec<_>>(), ["challenge"]);
    let statuses: Vec<&str> = seen[1..].iter().map(|v| v["task"]["status"].as_str().unwrap()).collect();
    assert_eq!(statuses, ["queued", "running", "succeeded"]);
    assert!(seen.last().unwrap()["task"]["content"]["url"].as_str().is_some());
}

#[tokio::test(flavor = "multi_thread")]
async fn callback_without_echo_gets_nothing_more() {
    let f = fixture().await;
    let mut body = t2v("MiniMax-H3-Turbo", 5);
    body["callback_url"] = json!(f.rx.url("/bad"));
    let id = create_ok(&f, body).await;
    assert_eq!(wait_done(&f, &id).await["status"], "succeeded");
    tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    let seen = f.rx.bodies("/bad");
    assert!(!seen.is_empty());
    // Only challenges, never a task body. (serve-kit re-challenges on a
    // later status change after a failed echo; see the WP-07 report.)
    assert!(seen.iter().all(|v| v.get("challenge").is_some() && v.get("task").is_none()), "{seen:?}");
}

#[tokio::test(flavor = "multi_thread")]
async fn callback_target_guard() {
    // The default SSRF policy refuses loopback targets at create time.
    let f = fixture().await;
    let mut sender = fastvideo_serve_kit::CallbackSender::new(None, None);
    sender.target.allow_private = false;
    let blocked = fastvideo_serve_kit::net::check_url(&url::Url::parse("http://127.0.0.1:9/cb").unwrap(), &sender.target);
    assert!(blocked.is_err());
    // This fixture allows private targets; a non-http scheme is still a 400.
    let mut body = t2v("MiniMax-H3", 5);
    body["callback_url"] = json!("file:///etc/passwd");
    let (s, _, v) = call(&f.app, "POST", "/v2/video_generation", Some(KEY), Some(body)).await;
    assert_eq!(s, StatusCode::BAD_REQUEST, "{v}");
    assert!(v["error"]["message"].as_str().unwrap().contains("callback_url"));
}

#[tokio::test(flavor = "multi_thread")]
async fn four_seconds_for_h3_not_for_h3_max() {
    let f = fixture().await;
    let id = create_ok(&f, t2v("MiniMax-H3", 4)).await;
    let t = wait_done(&f, &id).await;
    assert_eq!((t["status"].as_str(), t["duration"].as_u64()), (Some("succeeded"), Some(4)));
    assert_eq!(t["usage"]["output_seconds"], 4);
    assert_eq!(job_of(&f, &id).await.resolved.num_frames, 107, "4 s on the 17n+5 grid");

    let (s, _, v) = call(&f.app, "POST", "/v2/video_generation", Some(KEY), Some(t2v("MiniMax-H3-Max", 4))).await;
    assert_eq!(s, StatusCode::BAD_REQUEST);
    assert_eq!(v["error"]["type"], "bad_request_error");
    assert!(v["error"]["message"].as_str().unwrap().ends_with("(2013)"));
    // Turbo (our id) runs 4 s too.
    let id = create_ok(&f, t2v("MiniMax-H3-Turbo", 4)).await;
    assert_eq!(wait_done(&f, &id).await["status"], "succeeded");
    assert_eq!(job_of(&f, &id).await.resolved.num_frames, 107);
    // 15 s is the top of the grid.
    let id = create_ok(&f, t2v("MiniMax-H3", 15)).await;
    assert_eq!(job_of(&f, &id).await.resolved.num_frames, 362);
}

#[tokio::test(flavor = "multi_thread")]
async fn resolutions_and_ratios_resolve() {
    let f = fixture().await;
    let mk = |model: &str, res: &str, ratio: &str| {
        json!({"model": model, "content": [{"type": "text", "text": "x"}], "resolution": res, "duration": 5, "ratio": ratio})
    };
    for (ratio, want) in [("16:9", (1344, 768)), ("9:16", (768, 1344)), ("1:1", (768, 768)), ("4:3", (1024, 768))] {
        let id = create_ok(&f, mk("MiniMax-H3", "768P", ratio)).await;
        let j = job_of(&f, &id).await;
        assert_eq!((j.resolved.width, j.resolved.height), want, "{ratio}");
    }
    // 480P on H3-Max: 832x480 at 16:9.
    let id = create_ok(&f, mk("MiniMax-H3-Max", "480P", "16:9")).await;
    let j = job_of(&f, &id).await;
    assert_eq!((j.resolved.width, j.resolved.height), (832, 480));
    assert_eq!(wait_done(&f, &id).await["resolution"], "480P");
    // 2K on H3: the permanent engine gap, 400 (2013).
    let (s, _, v) = call(&f.app, "POST", "/v2/video_generation", Some(KEY), Some(mk("MiniMax-H3", "2K", "16:9"))).await;
    assert_eq!(s, StatusCode::BAD_REQUEST);
    assert_eq!(v["error"]["message"], "invalid params, 2K resolution is not supported by this server; model `fake-h3-max` serves 480p, 768p (2013)");
    // t2va with adaptive: 400.
    let (s, _, v) = call(&f.app, "POST", "/v2/video_generation", Some(KEY), Some(mk("MiniMax-H3", "768P", "adaptive"))).await;
    assert_eq!(s, StatusCode::BAD_REQUEST, "{v}");
}

#[tokio::test(flavor = "multi_thread")]
async fn i2v_and_keyframes_follow_the_image() {
    let f = fixture().await;
    let body = json!({"model": "MiniMax-H3", "resolution": "768P", "duration": 5, "ratio": "16:9",
        "content": [{"type": "text", "text": "x"}, {"type": "image_url", "image_url": {"url": png_uri(90, 160)}}]});
    let id = create_ok(&f, body).await;
    let j = job_of(&f, &id).await;
    assert_eq!(j.resolved.task, Task::I2V);
    assert_eq!((j.resolved.width, j.resolved.height), (768, 1344), "the 9:16 image wins over ratio 16:9");
    let t = wait_done(&f, &id).await;
    assert_eq!((t["ratio"].as_str(), t["usage"]["input_image_count"].as_u64()), (Some("9:16"), Some(1)));

    let body = json!({"model": "MiniMax-H3", "resolution": "768P", "duration": 5,
        "content": [{"type": "text", "text": "x"},
            {"type": "image_url", "image_url": {"url": png_uri(160, 90)}, "role": "first_frame"},
            {"type": "image_url", "image_url": {"url": png_uri(160, 90)}, "role": "last_frame"}]});
    let id = create_ok(&f, body).await;
    let j = job_of(&f, &id).await;
    assert_eq!(j.resolved.task, Task::Keyframes);
    assert_eq!(j.resolved.keyframes.len(), 2);
    assert_eq!(wait_done(&f, &id).await["usage"]["input_image_count"], 2);
}

#[tokio::test(flavor = "multi_thread")]
async fn references_on_max_and_the_ref2va_gap_on_turbo() {
    let f = fixture().await;
    let content = json!([{"type": "text", "text": "voice follows reference audio 1"},
        {"type": "video_url", "video_url": {"url": mp4_uri()}, "role": "reference_video"},
        {"type": "image_url", "image_url": {"url": png_uri(160, 90)}, "role": "reference_image"},
        {"type": "audio_url", "audio_url": {"url": wav_uri()}, "role": "reference_audio"}]);
    let id = create_ok(&f, json!({"model": "MiniMax-H3-Max", "content": content, "resolution": "768P", "duration": 5})).await;
    let j = job_of(&f, &id).await;
    assert_eq!(j.resolved.task, Task::Ref2V);
    let kinds: Vec<_> = j.resolved.references.iter().map(|r| r.0).collect();
    use fastvideo_protocol::MediaKind::*;
    assert_eq!(kinds, [Video, Image, Audio], "content order kept");
    let t = wait_done(&f, &id).await;
    assert_eq!(t["usage"], json!({"total_seconds": 9, "input_seconds": 4, "output_seconds": 5,
        "input_image_count": 1, "input_audio_seconds": 6}));
    assert_eq!(t["ratio"], "16:9");

    // Turbo has no ref2va DiT resident.
    let (s, _, v) = call(&f.app, "POST", "/v2/video_generation", Some(KEY),
        Some(json!({"model": "MiniMax-H3-Turbo", "content": content, "resolution": "768P", "duration": 5}))).await;
    assert_eq!(s, StatusCode::BAD_REQUEST);
    assert_eq!(v["error"]["message"], "invalid params, reference-to-video is not enabled on this server (2013)");
    // mm_file:// references: provider files are not supported.
    let (s, _, v) = call(&f.app, "POST", "/v2/video_generation", Some(KEY),
        Some(json!({"model": "MiniMax-H3", "resolution": "768P", "duration": 5,
            "content": [{"type": "text", "text": "x"}, {"type": "image_url", "image_url": {"url": "mm_file://1"}}]}))).await;
    assert_eq!(s, StatusCode::BAD_REQUEST);
    assert!(v["error"]["message"].as_str().unwrap().contains("provider file"), "{v}");
    // A data URI whose bytes are not an image: 400.
    let (s, _, _) = call(&f.app, "POST", "/v2/video_generation", Some(KEY),
        Some(json!({"model": "MiniMax-H3", "resolution": "768P", "duration": 5,
            "content": [{"type": "text", "text": "x"}, {"type": "image_url", "image_url": {"url": mp4_uri().replace("video/mp4", "image/png")}}]}))).await;
    assert_eq!(s, StatusCode::BAD_REQUEST);
}

#[tokio::test(flavor = "multi_thread")]
async fn draft_tier_is_marked_and_failures_render() {
    let f = fixture().await;
    let id = create_ok(&f, t2v("MiniMax-H3-Draft", 5)).await;
    let t = wait_done(&f, &id).await;
    assert_eq!(t["model"], "MiniMax-H3-Draft");
    assert_eq!(t["metadata"], json!({"tier": "draft", "recipe": "4step-vsa-480p-tiny-vae", "quality": "draft"}));
    assert_eq!(job_of(&f, &id).await.resolved.model.0, "fake-h3-draft");

    let mut body = t2v("MiniMax-H3", 5);
    body["content"] = json!([{"type": "text", "text": "fail please [fake:fail]"}]);
    body["callback_url"] = json!(f.rx.url("/cb"));
    let id = create_ok(&f, body).await;
    let t = wait_done(&f, &id).await;
    assert_eq!(t["status"], "failed");
    assert_eq!(t["error"]["code"], "1000");
    assert!(t["error"]["message"].as_str().unwrap().contains("injected failure"));
    assert!(t.get("usage").is_none());
    assert_eq!(t["content"], json!({}));
    let seen = f.rx.wait_terminal("/cb").await;
    assert_eq!(seen.last().unwrap()["task"]["status"], "failed");
}

#[tokio::test(flavor = "multi_thread")]
async fn auth_and_error_envelopes() {
    let f = fixture().await;
    let (s, _, v) = call(&f.app, "POST", "/v2/video_generation", None, Some(t2v("MiniMax-H3", 5))).await;
    assert_eq!(s, StatusCode::UNAUTHORIZED);
    assert_eq!(v["error"]["type"], "authorized_error");
    assert_eq!(v["error"]["http_code"], "401");
    assert!(v["error"]["message"].as_str().unwrap().starts_with("login fail: Please carry the API secret key"));
    assert_eq!(v["request_id"].as_str().unwrap().len(), 32);
    let (s, _, _) = call(&f.app, "POST", "/v2/video_generation", Some("nope"), Some(t2v("MiniMax-H3", 5))).await;
    assert_eq!(s, StatusCode::UNAUTHORIZED);
    let (s, _, _) = call(&f.app, "GET", "/v2/query/video_generation/123456789012345678", None, None).await;
    assert_eq!(s, StatusCode::UNAUTHORIZED);

    // Bad JSON, non-object, empty body.
    for raw in ["{nope", "[1,2]", ""] {
        let r = f.app.clone().oneshot(Request::post("/v2/video_generation").header("authorization", format!("Bearer {KEY}"))
            .body(Body::from(raw)).unwrap()).await.unwrap();
        assert_eq!(r.status(), StatusCode::BAD_REQUEST, "{raw}");
    }
    // No text item: MiniMax's own example message.
    let (s, _, v) = call(&f.app, "POST", "/v2/video_generation", Some(KEY),
        Some(json!({"model": "MiniMax-H3", "content": [], "resolution": "768P", "duration": 5, "ratio": "16:9"}))).await;
    assert_eq!(s, StatusCode::BAD_REQUEST);
    assert_eq!(v["error"]["message"], "invalid params, content must include a non-empty text item (prompt is required) (2013)");

    // Unknown task, and another key's task: 404 invalid task_id.
    let (s, _, v) = call(&f.app, "GET", "/v2/query/video_generation/999999999999999999", Some(KEY), None).await;
    assert_eq!(s, StatusCode::NOT_FOUND);
    assert_eq!((v["error"]["type"].as_str(), v["error"]["message"].as_str()), (Some("bad_request_error"), Some("invalid task_id (2013)")));
    let id = create_ok(&f, t2v("MiniMax-H3", 5)).await;
    let (s, _, _) = call(&f.app, "GET", &format!("/v2/query/video_generation/{id}"), Some(KEY_B), None).await;
    assert_eq!(s, StatusCode::NOT_FOUND);
    let (s, _, _) = call(&f.app, "DELETE", &format!("/v2/video_generation/{id}"), Some(KEY_B), None).await;
    assert_eq!(s, StatusCode::NOT_FOUND);

    // Context-IR and regeneration are not served.
    for route in ["/v2/h3_context_ir", "/v2/video_regeneration"] {
        let (s, _, v) = call(&f.app, "POST", route, Some(KEY), Some(json!({"model": "MiniMax-H3"}))).await;
        assert_eq!(s, StatusCode::BAD_REQUEST);
        assert_eq!(v["error"]["message"], format!("invalid params, {route} is not supported by this server (2013)"));
        let (s, _, _) = call(&f.app, "POST", route, None, None).await;
        assert_eq!(s, StatusCode::UNAUTHORIZED);
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn body_limit_is_a_400() {
    let f = fixture_with(MiniMaxConfig { body_max: 1024, ..MiniMaxConfig::default() }).await;
    let mut body = t2v("MiniMax-H3", 5);
    body["content"][0]["text"] = json!("x".repeat(2000));
    let (s, _, v) = call(&f.app, "POST", "/v2/video_generation", Some(KEY), Some(body)).await;
    assert_eq!(s, StatusCode::BAD_REQUEST);
    assert_eq!(v["error"]["message"], "invalid params, request body exceeds 1024 bytes (2013)");
}

#[tokio::test(flavor = "multi_thread")]
async fn list_filters_and_paging() {
    let f = fixture().await;
    let a = create_ok(&f, t2v("MiniMax-H3", 5)).await;
    wait_done(&f, &a).await;
    let b = create_ok(&f, t2v("MiniMax-H3-Turbo", 6)).await;
    wait_done(&f, &b).await;
    let mut fail = t2v("MiniMax-H3", 5);
    fail["content"][0]["text"] = json!("[fake:fail]");
    let c = create_ok(&f, fail).await;
    wait_done(&f, &c).await;
    // Another key's task never shows.
    let (s, _, _) = call(&f.app, "POST", "/v2/video_generation", Some(KEY_B), Some(t2v("MiniMax-H3", 5))).await;
    assert_eq!(s, StatusCode::OK);

    let list = |q: &str| {
        let app = f.app.clone();
        let q = q.to_owned();
        async move {
            let (s, _, v) = call(&app, "GET", &format!("/v2/query/video_generation{q}"), Some(KEY), None).await;
            assert_eq!(s, StatusCode::OK, "{v}");
            let ids: Vec<String> = v["items"].as_array().unwrap().iter().map(|t| t["id"].as_str().unwrap().to_owned()).collect();
            (ids, v["total"].as_u64().unwrap())
        }
    };
    assert_eq!(list("").await, (vec![c.clone(), b.clone(), a.clone()], 3), "newest first, own tasks");
    assert_eq!(list("?page_num=1&page_size=2").await, (vec![c.clone(), b.clone()], 3));
    assert_eq!(list("?page_num=2&page_size=2").await, (vec![a.clone()], 3));
    assert_eq!(list("?filter.status=failed").await, (vec![c.clone()], 1));
    assert_eq!(list("?filter.model=MiniMax-H3-Turbo").await, (vec![b.clone()], 1));
    assert_eq!(list(&format!("?filter.task_ids={a}&filter.task_ids={c}")).await, (vec![c.clone(), a.clone()], 2));
    assert_eq!(list("?filter.task_type=generation").await.1, 3);
    assert_eq!(list("?filter.task_type=h3_context_ir").await, (vec![], 0));
    let (s, _, v) = call(&f.app, "GET", "/v2/query/video_generation?page_num=0", Some(KEY), None).await;
    assert_eq!(s, StatusCode::BAD_REQUEST, "{v}");
    // Items are full VideoTasks.
    let (_, _, v) = call(&f.app, "GET", "/v2/query/video_generation?page_size=1", Some(KEY), None).await;
    assert_eq!(v["items"][0]["task_type"], "generation");
}

#[tokio::test(flavor = "multi_thread")]
async fn delete_semantics() {
    let f = fixture().await;
    // queued -> cancelled (and the callback reports it).
    f.gate.hold.store(true, Ordering::SeqCst);
    let mut body = t2v("MiniMax-H3", 5);
    body["callback_url"] = json!(f.rx.url("/cb"));
    let q = create_ok(&f, body).await;
    let (s, _, v) = call(&f.app, "DELETE", &format!("/v2/video_generation/{q}"), Some(KEY), None).await;
    assert_eq!(s, StatusCode::OK);
    assert_eq!(v, json!({"task_id": q, "action": "cancelled", "status": "cancelled"}));
    assert_eq!(query(&f, &q).await["status"], "cancelled");
    let seen = f.rx.wait_terminal("/cb").await;
    assert_eq!(seen.last().unwrap()["task"]["status"], "cancelled");
    // cancelled -> 400.
    let (s, _, v) = call(&f.app, "DELETE", &format!("/v2/video_generation/{q}"), Some(KEY), None).await;
    assert_eq!(s, StatusCode::BAD_REQUEST, "{v}");

    // running -> 400, and it keeps running.
    let r = create_ok(&f, t2v("MiniMax-H3", 5)).await;
    let rid = job_of(&f, &r).await.id;
    apply_event(&f.ctx, rid, JobEvent::Started).await.unwrap();
    let (s, _, v) = call(&f.app, "DELETE", &format!("/v2/video_generation/{r}"), Some(KEY), None).await;
    assert_eq!(s, StatusCode::BAD_REQUEST);
    assert!(v["error"]["message"].as_str().unwrap().contains("running"));
    assert_eq!(job_of(&f, &r).await.status(), JobStatus::Running);
    assert!(!job_of(&f, &r).await.cancel_requested);

    // succeeded / failed -> deleted (then 404).
    f.gate.hold.store(false, Ordering::SeqCst);
    let ok = create_ok(&f, t2v("MiniMax-H3", 5)).await;
    wait_done(&f, &ok).await;
    let mut failing = t2v("MiniMax-H3", 5);
    failing["content"][0]["text"] = json!("[fake:fail]");
    let bad = create_ok(&f, failing).await;
    assert_eq!(wait_done(&f, &bad).await["status"], "failed");
    for id in [&ok, &bad] {
        let (s, _, v) = call(&f.app, "DELETE", &format!("/v2/video_generation/{id}"), Some(KEY), None).await;
        assert_eq!(s, StatusCode::OK);
        assert_eq!(v, json!({"task_id": id, "action": "deleted", "status": "deleted"}));
        let (s, _, _) = call(&f.app, "GET", &format!("/v2/query/video_generation/{id}"), Some(KEY), None).await;
        assert_eq!(s, StatusCode::NOT_FOUND);
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn rate_limits() {
    // 2 creates per minute.
    let f = fixture_with(MiniMaxConfig { rpm: 2, ..MiniMaxConfig::default() }).await;
    create_ok(&f, t2v("MiniMax-H3", 5)).await;
    create_ok(&f, t2v("MiniMax-H3", 5)).await;
    let (s, h, v) = call(&f.app, "POST", "/v2/video_generation", Some(KEY), Some(t2v("MiniMax-H3", 5))).await;
    assert_eq!(s, StatusCode::TOO_MANY_REQUESTS);
    assert_eq!(v["error"]["type"], "rate_limit_error");
    assert_eq!(v["error"]["message"], "rate limit, please retry later (1002)");
    assert!(h.get("retry-after").is_some());
    // Another key has its own budget.
    let (s, _, _) = call(&f.app, "POST", "/v2/video_generation", Some(KEY_B), Some(t2v("MiniMax-H3", 5))).await;
    assert_eq!(s, StatusCode::OK);

    // 1 task in flight.
    let f = fixture_with(MiniMaxConfig { max_in_flight: 1, ..MiniMaxConfig::default() }).await;
    f.gate.hold.store(true, Ordering::SeqCst);
    create_ok(&f, t2v("MiniMax-H3", 5)).await;
    let (s, _, v) = call(&f.app, "POST", "/v2/video_generation", Some(KEY), Some(t2v("MiniMax-H3", 5))).await;
    assert_eq!(s, StatusCode::TOO_MANY_REQUESTS, "{v}");
}

#[tokio::test(flavor = "multi_thread")]
async fn unknown_model_and_engine_alias_override() {
    let f = fixture().await;
    let mut body = t2v("MiniMax-H3", 5);
    body["model"] = json!("MiniMax-Hailuo-02");
    let (s, _, v) = call(&f.app, "POST", "/v2/video_generation", Some(KEY), Some(body)).await;
    assert_eq!(s, StatusCode::BAD_REQUEST);
    assert!(v["error"]["message"].as_str().unwrap().contains("MiniMax-Hailuo-02"));

    // minimax.models maps a name onto an engine alias / id.
    let mut cfg = MiniMaxConfig::default();
    cfg.models.insert("MiniMax-H3".into(), "h3-turbo".into());
    let f = fixture_with(cfg).await;
    let id = create_ok(&f, t2v("MiniMax-H3", 5)).await;
    assert_eq!(job_of(&f, &id).await.resolved.model.0, "fake-h3-turbo");
}
