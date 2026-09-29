//! Native WHIP ingest (`/fv/v1/streams/ingest`, design §5.11) in the
//! assembled fv-serve: a str0m client POSTs a send-receive VP8 + Opus
//! offer to the loopback echo, publishes a blue camera and receives the
//! echo on the same peer; decoded here, the picture carries the camera's
//! colour inside the magenta overlay border. Also: the key is required,
//! offers must be `application/sdp`, non-duplex models are refused, a
//! second stream is busy (429), commands, the status object, `DELETE`
//! releases the model, capabilities advertise the ingest URL. Skips
//! without ffmpeg's libvpx.

#![cfg(feature = "webrtc")]

use std::collections::BTreeMap;
use std::time::{Duration, Instant};

use axum::body::Body;
use axum::http::{HeaderMap, Request, StatusCode};
use axum::Router;
use fastvideo_engine_service::ECHO_BORDER;
use fastvideo_media::decode::{VideoDecoder, VideoDecoderConfig};
use fastvideo_media::vp8::{libvpx_available, Vp8Config, Vp8Encoder};
use fastvideo_protocol::{InputVideoCodec, RgbFrame};
use fastvideo_serve::config::{Config, JobBackend};
use fastvideo_serve::{App, Overrides};
use fastvideo_serve_kit::KeyRing;
use fastvideo_webrtc::host::{AudioLayout, HostConfig, OfferOptions, Peer, PeerEvent, RtcHost};
use fastvideo_webrtc::sdp::Direction;
use fastvideo_webrtc::writer::{TrackKind, VideoCodec, VideoFrame};
use serde_json::{json, Value};
use tower::ServiceExt;

const KEY: &str = "sk-ingest";

async fn app() -> App {
    let dir = tempfile::Builder::new().prefix("fv-serve-ingest-").tempdir().unwrap().keep();
    let mut env = BTreeMap::new();
    env.insert("FV_API_KEYS".to_owned(), KeyRing::hash_hex(KEY));
    env.insert("FV_URL_SIGNING_KEY".to_owned(), "k".to_owned());
    env.insert("FV_STATE_DIR".to_owned(), dir.display().to_string());
    env.insert("FV_PUBLIC_BASE_URL".to_owned(), "http://fv.test".to_owned());
    env.insert("FV_ECHO_MODEL".to_owned(), "1".to_owned());
    let mut c = Config::default();
    c.apply_env(&env).unwrap();
    c.jobs.backend = JobBackend::Memory;
    c.engine.fake.step_ms = 1;
    c.webrtc.public_ip = "127.0.0.1".into();
    c.validate().unwrap();
    let a = App::build(c, Overrides::default()).await.unwrap();
    a.gate.engine().wait_ready().await;
    a
}

async fn req(app: &Router, method: &str, uri: &str, ct: Option<&str>, body: String, auth: bool) -> (StatusCode, HeaderMap, String) {
    let mut b = Request::builder().method(method).uri(uri);
    if auth {
        b = b.header("authorization", format!("Bearer {KEY}"));
    }
    if let Some(ct) = ct {
        b = b.header("content-type", ct);
    }
    let r = app.clone().oneshot(b.body(Body::from(body)).unwrap()).await.unwrap();
    let (s, h) = (r.status(), r.headers().clone());
    let bytes = axum::body::to_bytes(r.into_body(), 1 << 24).await.unwrap();
    (s, h, String::from_utf8_lossy(&bytes).into_owned())
}

async fn json_req(app: &Router, method: &str, uri: &str, body: Option<Value>) -> (StatusCode, Value) {
    let (s, _, b) = req(app, method, uri, body.as_ref().map(|_| "application/json"), body.map(|v| v.to_string()).unwrap_or_default(), true).await;
    (s, serde_json::from_str(&b).unwrap_or(Value::Null))
}

async fn offer(host: &RtcHost) -> (fastvideo_webrtc::host::PendingOffer, String) {
    host.offer(OfferOptions {
        video: Some(Direction::SendRecv),
        audio: Some((Direction::SendRecv, AudioLayout::Stereo)),
        video_codecs: vec![VideoCodec::Vp8],
        ..OfferOptions::default()
    })
    .await
    .unwrap()
}

async fn pump(peer: &mut Peer, video: &mut Vec<Vec<u8>>, audio: &mut usize, d: Duration) {
    let end = Instant::now() + d;
    loop {
        let left = end.saturating_duration_since(Instant::now());
        if left.is_zero() {
            return;
        }
        match tokio::time::timeout(left, peer.next_event()).await {
            Ok(Some(PeerEvent::Media { kind: TrackKind::Video, data, .. })) => video.push(data.to_vec()),
            Ok(Some(PeerEvent::Media { kind: TrackKind::Audio, .. })) => *audio += 1,
            Ok(Some(PeerEvent::Closed(r))) => panic!("closed: {r:?}"),
            Ok(Some(_)) => {}
            Ok(None) | Err(_) => return,
        }
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn whip_ingest_round_trips_the_camera_through_the_echo() {
    if !libvpx_available() {
        eprintln!("skipping: ffmpeg has no libvpx");
        return;
    }
    let a = app().await;
    let r = &a.router;
    // Capabilities advertise the ingest URL of the duplex model.
    let (s, caps) = json_req(r, "GET", "/fv/v1/capabilities", None).await;
    assert_eq!(s, StatusCode::OK);
    let echo = caps["models"].as_array().unwrap().iter().find(|m| m["caps"]["id"] == "fv-echo").expect("fv-echo is served").clone();
    assert_eq!(echo["ingest"]["whip"], "/fv/v1/streams/ingest?model=fv-echo");
    assert_eq!(echo["caps"]["stream"]["duplex"]["input"]["max_bitrate_kbps"], 4000);
    assert_eq!(echo["stream_limits"]["hard_max_s"], 300);

    let host = RtcHost::bind(HostConfig::loopback(true, false)).await.unwrap();
    let (pending, sdp) = offer(&host).await;
    let url = "/fv/v1/streams/ingest?model=fv-echo&scene=a%20desk&max_seconds=60";
    // Refusals: no key, not SDP, not a duplex model.
    assert_eq!(req(r, "POST", url, Some("application/sdp"), sdp.clone(), false).await.0, StatusCode::UNAUTHORIZED);
    assert_eq!(req(r, "POST", url, Some("application/json"), sdp.clone(), true).await.0, StatusCode::UNSUPPORTED_MEDIA_TYPE);
    let (s, _, b) = req(r, "POST", "/fv/v1/streams/ingest?model=fake-h3-max", Some("application/sdp"), sdp.clone(), true).await;
    assert_eq!(s, StatusCode::BAD_REQUEST, "{b}");
    assert!(b.contains("not a duplex model"), "{b}");

    let (s, h, answer) = req(r, "POST", url, Some("application/sdp"), sdp, true).await;
    assert_eq!(s, StatusCode::CREATED, "{answer}");
    assert_eq!(h["content-type"], "application/sdp");
    let loc = h["location"].to_str().unwrap().to_owned();
    assert!(loc.starts_with("/fv/v1/streams/ingest/fvingest_"), "{loc}");
    assert!(answer.contains("b=AS:4000"), "the answer caps the camera: {answer}");
    // One echo at a time.
    let (_, sdp2) = offer(&host).await;
    let (s, h2, _) = req(r, "POST", url, Some("application/sdp"), sdp2, true).await;
    assert_eq!(s, StatusCode::TOO_MANY_REQUESTS);
    assert!(h2.get("retry-after").is_some());

    let mut peer = pending.accept_answer(&answer).await.unwrap();
    assert_eq!(peer.video_codec(), Some(VideoCodec::Vp8));
    let (mut video, mut audio) = (Vec::new(), 0usize);
    let t0 = Instant::now();
    loop {
        match tokio::time::timeout(Duration::from_secs(10), peer.next_event()).await.expect("connect") {
            Some(PeerEvent::Connected) => break,
            Some(PeerEvent::Closed(r)) => panic!("{r:?}"),
            _ => assert!(t0.elapsed() < Duration::from_secs(10)),
        }
    }
    // The camera: blue VP8, pre-encoded, sent in real time until the echo
    // has shown 30 input frames.
    let mut enc = Vp8Encoder::new(Vp8Config::new(320, 240, 30)).unwrap();
    let mut frames = Vec::new();
    for i in 0..60u64 {
        frames.extend(enc.encode(&RgbFrame::solid(320, 240, [20, 40, 230], i)).unwrap());
    }
    frames.extend(enc.finish().unwrap());
    let mut k = 0u64;
    let start = Instant::now();
    let mut status = Value::Null;
    for _ in 0..8 {
        for f in &frames {
            peer.send_video(VideoFrame::new(f.data.clone(), k * 3000)).unwrap();
            k += 1;
            let next = start + Duration::from_millis(k * 33);
            pump(&mut peer, &mut video, &mut audio, next.saturating_duration_since(Instant::now())).await;
        }
        status = json_req(r, "GET", &loc, None).await.1;
        // Enough input shown, and enough output back (on a loaded host the
        // encoders' ffmpeg starts late).
        if status["session"]["input_frames"].as_u64().unwrap_or(0) >= 30
            && status["output"]["frames_sent"].as_u64().unwrap_or(0) > 30
            && audio > 30
        {
            break;
        }
    }
    pump(&mut peer, &mut video, &mut audio, Duration::from_millis(500)).await;
    eprintln!("sent {k}; status {status}");
    assert_eq!(status["state"], "streaming", "{status}");
    assert_eq!(status["model"], "fv-echo");
    assert_eq!(status["max_seconds"], 60);
    assert_eq!(status["session"]["context"]["scene"], "a desk");
    assert!(status["session"]["input_frames"].as_u64().unwrap() >= 30, "{status}");
    assert_eq!(status["ingest"]["codec"], "vp8");
    assert_eq!(status["ingest"]["source_size"], json!([320, 240]));
    assert_eq!(status["output"]["video_codec"], "vp8");
    assert!(status["output"]["frames_sent"].as_u64().unwrap() > 30, "{status}");
    assert!(status["output"]["audio_packets_sent"].as_u64().unwrap() > 30, "{status}");
    assert!(audio > 30, "echo audio packets received: {audio}");

    // The echo, decoded: blue inside the magenta border.
    let (w, h) = (640, 360);
    let mut dec = VideoDecoder::new(VideoDecoderConfig { codec: InputVideoCodec::Vp8, width: w, height: h }).unwrap();
    let start = video.iter().position(|f| f.first().is_some_and(|b| b & 1 == 0)).expect("a keyframe");
    let mut pics = Vec::new();
    for (i, f) in video[start..].iter().enumerate() {
        dec.push(f, i as u64).unwrap();
        pics.extend(dec.poll().unwrap());
    }
    pics.extend(dec.finish().unwrap());
    let near = |p: Option<[u8; 3]>, want: [u8; 3]| p.is_some_and(|p| p.iter().zip(want).all(|(a, b)| (i32::from(*a) - i32::from(b)).abs() < 48));
    let echoed = pics.iter().filter(|p| near(p.frame.pixel(w / 2, h * 2 / 3), [20, 40, 230]) && near(p.frame.pixel(3, h / 2), ECHO_BORDER)).count();
    assert!(echoed >= 10, "{echoed} of {} decoded echo frames show the camera with the overlay", pics.len());

    // Commands.
    let (s, v) = json_req(r, "POST", &format!("{loc}/commands"), Some(json!({"type": "set_paused", "data": {"paused": true}}))).await;
    assert_eq!(s, StatusCode::OK, "{v}");
    assert_eq!(v["reply"]["data"]["paused"], true);
    let (s, _) = json_req(r, "POST", &format!("{loc}/commands"), Some(json!({"type": "reset"}))).await;
    assert_eq!(s, StatusCode::BAD_REQUEST);
    // No trickle.
    assert_eq!(req(r, "PATCH", &loc, Some("application/trickle-ice-sdpfrag"), String::new(), true).await.0, StatusCode::METHOD_NOT_ALLOWED);
    // Listed; DELETE ends it and frees the echo.
    let (_, list) = json_req(r, "GET", "/fv/v1/streams/ingest", None).await;
    assert_eq!(list["data"][0]["id"], loc.rsplit('/').next().unwrap());
    let (s, v) = json_req(r, "DELETE", &loc, None).await;
    assert_eq!(s, StatusCode::OK);
    assert_eq!(v["state"], "closed", "{v}");
    assert_eq!(v["end_reason"], "stopped");
    let t0 = Instant::now();
    loop {
        let (_, sdp3) = offer(&host).await;
        let (s, h3, _) = req(r, "POST", url, Some("application/sdp"), sdp3, true).await;
        if s == StatusCode::CREATED {
            let loc3 = h3["location"].to_str().unwrap().to_owned();
            assert_eq!(json_req(r, "DELETE", &loc3, None).await.0, StatusCode::OK);
            break;
        }
        assert_eq!(s, StatusCode::TOO_MANY_REQUESTS);
        assert!(t0.elapsed() < Duration::from_secs(10), "the echo was not released");
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
}
