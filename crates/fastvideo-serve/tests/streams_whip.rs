//! WP-15 acceptance, end to end on the fake engine: `POST /fv/v1/streams`
//! → session → pacer → H.264 (OpenH264) + Opus → WHIP publish to an
//! in-process WHIP endpoint (axum + a str0m host, recvonly) that decodes
//! what it receives and reads back the frame index the fake burned into
//! every frame. `DELETE` sends the WHIP `DELETE`.
//!
//! With `FV_TEST_MEDIAMTX=http://127.0.0.1:8889` the causal test also
//! publishes to that MediaMTX and checks a WHEP viewer receives decodable
//! H.264 (`scripts/serve/whip-e2e.sh` starts one).

#![cfg(all(feature = "webrtc", feature = "http-client", feature = "encoders"))]

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use axum::body::Body;
use axum::extract::{Path, State};
use axum::http::{HeaderMap, Request, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{delete, post};
use axum::Router;
use fastvideo_engine_service::fake::decode_frame_index;
use fastvideo_media::video::openh264_backend::decode_rgb;
use fastvideo_serve::config::{Config, JobBackend};
use fastvideo_serve::{App, Overrides};
use fastvideo_serve_kit::KeyRing;
use fastvideo_webrtc::channel::ChannelPolicy;
use fastvideo_webrtc::host::{AnswerOptions, AudioLayout, HostConfig, OfferOptions, Peer, PeerEvent, RtcHost};
use fastvideo_webrtc::sdp::{Direction, MediaKind, Sdp};
use fastvideo_webrtc::writer::TrackKind;
use serde_json::{json, Value};
use tower::ServiceExt;

const KEY: &str = "sk-streams";

async fn app() -> App {
    app_with(&[]).await
}

async fn app_with(extra: &[(&str, &str)]) -> App {
    let dir = std::env::temp_dir().join(format!(
        "fv-serve-streams-{:x}",
        std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()
    ));
    let mut env = BTreeMap::new();
    env.insert("FV_API_KEYS".to_owned(), KeyRing::hash_hex(KEY));
    env.insert("FV_URL_SIGNING_KEY".to_owned(), "k".to_owned());
    env.insert("FV_STATE_DIR".to_owned(), dir.display().to_string());
    env.insert("FV_PUBLIC_BASE_URL".to_owned(), "http://fv.test".to_owned());
    for (k, v) in extra {
        env.insert((*k).to_owned(), (*v).to_owned());
    }
    let mut c = Config::default();
    c.apply_env(&env).unwrap();
    c.jobs.backend = JobBackend::Memory;
    c.engine.fake.step_ms = 1;
    c.validate().unwrap();
    // Loopback only: no STUN probe.
    std::env::set_var("FV_STREAM_STUN", "none");
    let a = App::build(c, Overrides::default()).await.unwrap();
    a.gate.engine().wait_ready().await;
    a
}

async fn call(app: &Router, method: &str, uri: &str, body: Option<Value>) -> (StatusCode, HeaderMap, Value) {
    let mut b = Request::builder().method(method).uri(uri).header("authorization", format!("Bearer {KEY}"));
    let req = match body {
        Some(v) => {
            b = b.header("content-type", "application/json");
            b.body(Body::from(v.to_string())).unwrap()
        }
        None => b.body(Body::empty()).unwrap(),
    };
    let r = app.clone().oneshot(req).await.unwrap();
    let (s, h) = (r.status(), r.headers().clone());
    let bytes = axum::body::to_bytes(r.into_body(), 1 << 24).await.unwrap();
    (s, h, serde_json::from_slice(&bytes).unwrap_or(Value::Null))
}

/// What the WHIP endpoint received.
#[derive(Default)]
struct Rx {
    posts: usize,
    deletes: usize,
    offers: Vec<String>,
    video: Vec<(u64, bool, bytes::Bytes)>,
    audio: Vec<(u64, bytes::Bytes)>,
}

#[derive(Clone)]
struct Endpoint {
    host: RtcHost,
    rx: Arc<Mutex<Rx>>,
}

async fn whip_post(State(m): State<Endpoint>, body: String) -> Response {
    m.rx.lock().unwrap().posts += 1;
    m.rx.lock().unwrap().offers.push(body.clone());
    let audio = Sdp::parse(&body)
        .ok()
        .and_then(|s| s.media.iter().any(|x| x.kind() == MediaKind::Audio).then_some(AudioLayout::Stereo));
    match m
        .host
        .answer(&body, AnswerOptions { audio, channels: ChannelPolicy::none(), ..Default::default() })
        .await
    {
        Ok((peer, answer)) => {
            let id = peer.id();
            let rx = m.rx.clone();
            tokio::spawn(collect(peer, rx));
            (StatusCode::CREATED, [("content-type", "application/sdp".to_string()), ("location", format!("whip/{id}"))], answer)
                .into_response()
        }
        Err(e) => (StatusCode::BAD_REQUEST, e.to_string()).into_response(),
    }
}

async fn collect(mut peer: Peer, rx: Arc<Mutex<Rx>>) {
    while let Some(ev) = peer.next_event().await {
        match ev {
            PeerEvent::Media { kind: TrackKind::Video, rtp_time, keyframe, data, .. } => {
                rx.lock().unwrap().video.push((rtp_time, keyframe, data))
            }
            PeerEvent::Media { kind: TrackKind::Audio, rtp_time, data, .. } => rx.lock().unwrap().audio.push((rtp_time, data)),
            PeerEvent::Closed(_) => break,
            _ => {}
        }
    }
}

async fn whip_delete(State(m): State<Endpoint>, Path(_id): Path<String>) -> StatusCode {
    m.rx.lock().unwrap().deletes += 1;
    StatusCode::OK
}

async fn endpoint() -> (String, Arc<Mutex<Rx>>) {
    let host = RtcHost::bind(HostConfig::loopback(true, false)).await.unwrap();
    let rx = Arc::new(Mutex::new(Rx::default()));
    let app = Router::new()
        .route("/live/whip", post(whip_post))
        .route("/live/whip/{id}", delete(whip_delete))
        .with_state(Endpoint { host, rx: rx.clone() });
    let l = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = l.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(l, app).await.unwrap() });
    (format!("http://{addr}/live/whip"), rx)
}

async fn wait<F: Fn() -> bool>(what: &str, secs: u64, f: F) {
    let t0 = std::time::Instant::now();
    while !f() {
        assert!(t0.elapsed() < Duration::from_secs(secs), "timed out waiting for {what}");
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
}

/// Decodes the received access units and returns the burned-in indices.
fn indices(video: &[(u64, bool, bytes::Bytes)]) -> Vec<u32> {
    let start = video.iter().position(|v| v.1).expect("an IDR");
    let aus: Vec<&[u8]> = video[start..].iter().map(|v| v.2.as_ref()).collect();
    decode_rgb(&aus).unwrap().iter().filter_map(decode_frame_index).collect()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn causal_stream_publishes_decodable_h264_over_whip() {
    let a = app().await;
    let (url, rx) = endpoint().await;
    // Busy / unknown / bad-body answers first.
    let (s, _, _) = call(&a.router, "POST", "/fv/v1/streams", Some(json!({"model": "fake-sfwan", "whip_url": url}))).await;
    assert_eq!(s, StatusCode::BAD_REQUEST, "a causal stream needs a prompt");
    let body = json!({
        "model": "fake-sfwan", "whip_url": url, "prompt": "a lighthouse at dusk",
        "width": 640, "height": 352, "whip_target": "mediamtx",
    });
    let (s, _, v) = call(&a.router, "POST", "/fv/v1/streams", Some(body.clone())).await;
    assert_eq!(s, StatusCode::CREATED, "{v}");
    let id = v["id"].as_str().unwrap().to_owned();
    assert_eq!(v["mode"], "causal");
    // One stream session per executor: the second is 429 with Retry-After.
    let (s, h, _) = call(&a.router, "POST", "/fv/v1/streams", Some(body)).await;
    assert_eq!(s, StatusCode::TOO_MANY_REQUESTS);
    assert!(h.contains_key("retry-after"));

    wait("48 video frames", 60, || rx.lock().unwrap().video.len() >= 48).await;
    let (s, _, v) = call(&a.router, "GET", &format!("/fv/v1/streams/{id}"), None).await;
    assert_eq!(s, StatusCode::OK);
    assert_eq!(v["status"]["state"], "streaming", "{v}");
    assert_eq!(v["status"]["video_codec"], "H264");
    let ttff = &v["ttff"];
    for k in ["load_ms", "first_block_ms", "transport_ms", "total_ms"] {
        assert!(ttff[k].as_f64().is_some(), "ttff.{k}: {v}");
    }
    assert!(v["pacer"]["unique_fps"].as_f64().unwrap() > 0.0);
    // Causal commands through the native API.
    let (s, _, r) = call(
        &a.router,
        "POST",
        &format!("/fv/v1/streams/{id}/commands"),
        Some(json!({"type": "set_prompt", "data": {"prompt": "a storm"}})),
    )
    .await;
    assert_eq!(s, StatusCode::OK);
    assert_eq!(r["reply"]["type"], "state_update");
    assert_eq!(r["reply"]["data"]["prompt"], "a storm");

    // Video-only model: no audio m-line was offered (design §5.3).
    {
        let rx = rx.lock().unwrap();
        let offer = Sdp::parse(&rx.offers[0]).unwrap();
        assert!(offer.media.iter().all(|m| m.kind() != MediaKind::Audio));
        assert!(rx.audio.is_empty());
        let rtps: Vec<u64> = rx.video.iter().map(|v| v.0).collect();
        assert!(rtps.windows(2).all(|w| w[1] > w[0]), "monotonic RTP");
    }
    let idx = indices(&rx.lock().unwrap().video);
    assert!(idx.len() >= 24, "decoded {} frames with a readable index", idx.len());
    assert!(idx.windows(2).all(|w| w[1] >= w[0]), "frames in order: {idx:?}");

    // Optional: MediaMTX + WHEP viewer.
    if let Ok(mtx) = std::env::var("FV_TEST_MEDIAMTX") {
        let (s, _, _) = call(&a.router, "DELETE", &format!("/fv/v1/streams/{id}"), None).await;
        assert_eq!(s, StatusCode::OK);
        mediamtx_roundtrip(&a.router, mtx.trim_end_matches('/')).await;
        return;
    }

    let (s, _, v) = call(&a.router, "DELETE", &format!("/fv/v1/streams/{id}"), None).await;
    assert_eq!(s, StatusCode::OK);
    assert_eq!(v["status"]["state"], "closed", "{v}");
    assert_eq!(v["status"]["end_reason"], "stopped");
    wait("WHIP DELETE", 10, || rx.lock().unwrap().deletes == 1).await;
    // The executor is free again.
    let (s, _, l) = call(&a.router, "GET", "/fv/v1/streams", None).await;
    assert_eq!(s, StatusCode::OK);
    assert_eq!(l["data"].as_array().unwrap().len(), 1);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn clip_stream_publishes_av_and_takes_clip_commands() {
    let a = app().await;
    let (url, rx) = endpoint().await;
    let body = json!({
        "model": "fake-h3-turbo", "whip_url": url, "width": 640, "height": 352,
        "clips": [{"prompt": "a"}], "max_seconds": 3,
    });
    let (s, _, v) = call(&a.router, "POST", "/fv/v1/streams", Some(body)).await;
    assert_eq!(s, StatusCode::CREATED, "{v}");
    assert_eq!(v["mode"], "clip");
    let id = v["id"].as_str().unwrap().to_owned();
    wait("audio and video", 60, || {
        let r = rx.lock().unwrap();
        r.video.len() >= 24 && r.audio.len() >= 50
    })
    .await;
    {
        let r = rx.lock().unwrap();
        let offer = Sdp::parse(&r.offers[0]).unwrap();
        let audio = offer.media.iter().find(|m| m.kind() == MediaKind::Audio).expect("audio m-line");
        assert_eq!(audio.direction(), Direction::SendOnly);
        // Opus RTP time advances by 960 per 20 ms packet.
        let d: Vec<u64> = r.audio.windows(2).map(|w| w[1].0 - w[0].0).collect();
        assert!(d.iter().filter(|x| **x == 960).count() * 10 >= d.len() * 9, "{:?}", &d[..d.len().min(20)]);
    }
    let (s, _, r) = call(
        &a.router,
        "POST",
        &format!("/fv/v1/streams/{id}/commands"),
        Some(json!({"type": "get_state"})),
    )
    .await;
    assert_eq!(s, StatusCode::OK);
    assert_eq!(r["reply"]["type"], "state_update");
    assert_eq!(r["reply"]["data"]["autoplay"], true);
    // The session's fast-h3 events are recorded.
    let (_, _, v) = call(&a.router, "GET", &format!("/fv/v1/streams/{id}"), None).await;
    let names: Vec<&str> = v["events"].as_array().unwrap().iter().filter_map(|e| e["type"].as_str()).collect();
    assert!(names.contains(&"clip_generated") && names.contains(&"clip_started"), "{names:?}");
    // max_seconds (3 s of video) ends the stream by itself.
    let t0 = std::time::Instant::now();
    loop {
        let (_, _, v) = call(&a.router, "GET", &format!("/fv/v1/streams/{id}"), None).await;
        if v["status"]["state"] == "closed" {
            assert_eq!(v["status"]["end_reason"], "session_limit", "{v}");
            break;
        }
        assert!(t0.elapsed() < Duration::from_secs(30), "{v}");
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    wait("WHIP DELETE", 10, || rx.lock().unwrap().deletes == 1).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn causal_streams_end_at_the_session_limit() {
    // [streams] limits from the environment, small enough for real time.
    let a = app_with(&[("FV_CAUSAL_DEFAULT_MAX_S", "2"), ("FV_CAUSAL_HARD_MAX_S", "3")]).await;
    let (_, _, caps) = call(&a.router, "GET", "/fv/v1/capabilities", None).await;
    let sf = caps["models"].as_array().unwrap().iter().find(|m| m["caps"]["id"] == "fake-sfwan").unwrap();
    assert_eq!(sf["stream_limits"]["default_max_s"], 2, "{sf}");
    assert_eq!(sf["stream_limits"]["hard_max_s"], 3);
    assert_eq!(sf["stream_limits"]["reset_restarts_clock"], true);
    let (url, rx) = endpoint().await;
    let body = json!({"model": "fake-sfwan", "whip_url": url, "prompt": "a road", "width": 64, "height": 32});
    let (s, _, v) = call(&a.router, "POST", "/fv/v1/streams", Some(body.clone())).await;
    assert_eq!(s, StatusCode::CREATED, "{v}");
    // No max_seconds: the default.
    assert_eq!(v["max_seconds"], 2);
    let id = v["id"].as_str().unwrap().to_owned();
    let t0 = std::time::Instant::now();
    let v = loop {
        let (_, _, v) = call(&a.router, "GET", &format!("/fv/v1/streams/{id}"), None).await;
        if v["status"]["state"] == "closed" {
            break v;
        }
        assert!(t0.elapsed() < Duration::from_secs(30), "{v}");
        tokio::time::sleep(Duration::from_millis(100)).await;
    };
    assert_eq!(v["status"]["end_reason"], "session_limit", "{v}");
    let vs = v["pacer"]["video_seconds"].as_f64().unwrap();
    assert!((2.0..2.5).contains(&vs), "{vs}");
    wait("WHIP DELETE", 10, || rx.lock().unwrap().deletes == 1).await;
    // A longer request is clamped to the hard ceiling; 0 is refused.
    let mut zero = body.clone();
    zero["max_seconds"] = json!(0);
    let (s, _, v) = call(&a.router, "POST", "/fv/v1/streams", Some(zero)).await;
    assert_eq!(s, StatusCode::BAD_REQUEST, "{v}");
    let mut long = body;
    long["max_seconds"] = json!(3600);
    let mut created = Value::Null;
    // The executor may still be releasing the previous session.
    for _ in 0..50 {
        let (s, _, v) = call(&a.router, "POST", "/fv/v1/streams", Some(long.clone())).await;
        if s == StatusCode::CREATED {
            created = v;
            break;
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    assert_eq!(created["max_seconds"], 3, "{created}");
    let id = created["id"].as_str().unwrap().to_owned();
    call(&a.router, "DELETE", &format!("/fv/v1/streams/{id}"), None).await;
}

/// Publishes to MediaMTX and plays it back through WHEP.
async fn mediamtx_roundtrip(router: &Router, base: &str) {
    let whip = format!("{base}/fvtest/whip");
    let body = json!({
        "model": "fake-sfwan", "whip_url": whip, "prompt": "a lighthouse", "width": 640, "height": 352,
    });
    // The executor may still be releasing the previous session.
    let mut id = String::new();
    for _ in 0..50 {
        let (s, _, v) = call(router, "POST", "/fv/v1/streams", Some(body.clone())).await;
        if s == StatusCode::CREATED {
            id = v["id"].as_str().unwrap().to_owned();
            break;
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    assert!(!id.is_empty());
    let t0 = std::time::Instant::now();
    loop {
        let (_, _, v) = call(router, "GET", &format!("/fv/v1/streams/{id}"), None).await;
        if v["status"]["state"] == "streaming" && v["status"]["frames_sent"].as_u64().unwrap_or(0) > 10 {
            break;
        }
        assert!(t0.elapsed() < Duration::from_secs(30), "{v}");
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    let got = whep_view(&format!("{base}/fvtest/whep"), 48).await;
    let idx = indices(&got);
    eprintln!("whep viewer: {} access units, {} decoded with a frame index", got.len(), idx.len());
    assert!(idx.len() >= 24);
    // MediaMTX asks its WebRTC publishers for a keyframe (PLI) every 2 s.
    // Those must be answered by the periodic IDRs, not by forced ones (a
    // forced NVENC IDR restarts ffmpeg).
    while t0.elapsed() < Duration::from_secs(9) {
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    let (_, _, v) = call(router, "GET", &format!("/fv/v1/streams/{id}"), None).await;
    eprintln!("mediamtx publish status after ~9 s: {}", v["status"]);
    let st = &v["status"];
    let n = |k: &str| st[k].as_u64().unwrap_or(0);
    assert!(n("keyframe_requests") >= 3, "MediaMTX sent no periodic PLIs? {st}");
    // At most the reader joining (MediaMTX asks for a keyframe then too)
    // may force one; before, every periodic PLI did.
    assert!(n("forced_idrs") <= 2, "periodic PLIs forced IDRs: {st}");
    assert!(n("keyframes_sent") >= 4, "the 2 s GOP should give ~4 IDRs in 9 s: {st}");
    let (s, _, _) = call(router, "DELETE", &format!("/fv/v1/streams/{id}"), None).await;
    assert_eq!(s, StatusCode::OK);
}

/// A WHEP viewer: recvonly offer, POST, collect `n` video access units.
pub async fn whep_view(url: &str, n: usize) -> Vec<(u64, bool, bytes::Bytes)> {
    let host = RtcHost::bind(HostConfig::loopback(true, false)).await.unwrap();
    let (pending, offer) = host
        .offer(OfferOptions {
            video: Some(Direction::RecvOnly),
            audio: None,
            ..OfferOptions::default()
        })
        .await
        .unwrap();
    let resp = reqwest::Client::new()
        .post(url)
        .header("content-type", "application/sdp")
        .body(offer)
        .send()
        .await
        .unwrap();
    assert!(resp.status().is_success(), "WHEP POST: {}", resp.status());
    let answer = resp.text().await.unwrap();
    let mut peer = pending.accept_answer(&answer).await.unwrap();
    let mut out = Vec::new();
    let deadline = tokio::time::Instant::now() + Duration::from_secs(30);
    while out.len() < n {
        match tokio::time::timeout_at(deadline, peer.next_event()).await.expect("WHEP media") {
            Some(PeerEvent::Media { kind: TrackKind::Video, rtp_time, keyframe, data, .. }) => out.push((rtp_time, keyframe, data)),
            Some(PeerEvent::Closed(r)) => panic!("whep closed: {r:?}"),
            Some(_) => {}
            None => panic!("whep ended"),
        }
    }
    out
}
