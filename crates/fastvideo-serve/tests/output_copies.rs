//! Director output copies (docs/serve/stream-outputs.md "Director and
//! Reactor"): a fal director session on the fake engine with
//! `configure.x_fv_output` publishes a second encode over WHIP to an
//! in-process endpoint while the client peer plays; `output` is refused
//! unless `[director] allow_output`. The Reactor copy is tested in
//! `fastvideo-reactor/tests/runtime.rs` (`output_copy_gets_the_paced_ticks`).
//!
//! Also run without `encoders` (check.sh), the serve images' feature set:
//! the copy's Opus was once gated on `encoders`, so image builds offered an
//! audio m-line and never sent a packet (gpu-check-2026-10-13, bug 3).

#![cfg(all(feature = "fal", feature = "webrtc", feature = "http-client"))]

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use axum::body::Body;
use axum::extract::{Path, State};
use axum::http::{Request, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{delete, post};
use axum::Router;
use fastvideo_serve::config::{Config, JobBackend};
use fastvideo_serve::{App, Overrides};
use fastvideo_serve_kit::KeyRing;
use fastvideo_webrtc::channel::{ChannelMessage, ChannelPolicy};
use fastvideo_webrtc::host::{AnswerOptions, AudioLayout, HostConfig, OfferOptions, Peer, PeerEvent, RtcHost};
use fastvideo_webrtc::sdp::{Direction, MediaKind, Sdp};
use fastvideo_webrtc::writer::TrackKind;
use serde_json::{json, Value};
use tower::ServiceExt;

const KEY: &str = "fal-output-key";
const T: Duration = Duration::from_secs(90);

#[derive(Default)]
struct Rx {
    video: Vec<(bool, bytes::Bytes)>,
    audio: usize,
    deletes: usize,
}

#[derive(Clone)]
struct Endpoint {
    host: RtcHost,
    rx: Arc<Mutex<Rx>>,
}

async fn whip_post(State(m): State<Endpoint>, body: String) -> Response {
    let audio = Sdp::parse(&body)
        .ok()
        .and_then(|s| s.media.iter().any(|x| x.kind() == MediaKind::Audio).then_some(AudioLayout::Stereo));
    match m.host.answer(&body, AnswerOptions { audio, channels: ChannelPolicy::none(), ..Default::default() }).await {
        Ok((mut peer, answer)) => {
            let id = peer.id();
            let rx = m.rx.clone();
            tokio::spawn(async move {
                while let Some(ev) = peer.next_event().await {
                    match ev {
                        PeerEvent::Media { kind: TrackKind::Video, keyframe, data, .. } => rx.lock().unwrap().video.push((keyframe, data)),
                        PeerEvent::Media { kind: TrackKind::Audio, .. } => rx.lock().unwrap().audio += 1,
                        PeerEvent::Closed(_) => break,
                        _ => {}
                    }
                }
            });
            (StatusCode::CREATED, [("content-type", "application/sdp".to_string()), ("location", format!("whip/{id}"))], answer).into_response()
        }
        Err(e) => (StatusCode::BAD_REQUEST, e.to_string()).into_response(),
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

async fn app() -> App {
    let dir = tempfile::Builder::new().prefix("fv-serve-copies-").tempdir().unwrap().keep();
    let mut env = BTreeMap::new();
    env.insert("FV_API_KEYS".to_owned(), KeyRing::hash_hex(KEY));
    env.insert("FV_URL_SIGNING_KEY".to_owned(), "k".to_owned());
    env.insert("FV_STATE_DIR".to_owned(), dir.display().to_string());
    env.insert("FV_PUBLIC_BASE_URL".to_owned(), "http://fv.test".to_owned());
    env.insert("FV_OUTPUTS_ALLOW_PRIVATE".to_owned(), "1".to_owned());
    env.insert("FV_STREAM_ENCODER".to_owned(), "x264-test".to_owned());
    let mut c = Config::default();
    c.apply_env(&env).unwrap();
    c.jobs.backend = JobBackend::Memory;
    c.engine.fake.step_ms = 1;
    c.webrtc.public_ip = "127.0.0.1".into();
    c.validate().unwrap();
    std::env::set_var("FV_STREAM_STUN", "none");
    let a = App::build(c, Overrides::default()).await.unwrap();
    a.gate.engine().wait_ready().await;
    a
}

async fn post_json(app: &Router, uri: &str, body: Value) -> (u16, Value) {
    let req = Request::builder()
        .method("POST")
        .uri(uri)
        .header("authorization", format!("Key {KEY}"))
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .unwrap();
    let r = app.clone().oneshot(req).await.unwrap();
    let status = r.status().as_u16();
    let bytes = axum::body::to_bytes(r.into_body(), 1 << 24).await.unwrap();
    (status, serde_json::from_slice(&bytes).unwrap_or(Value::Null))
}

/// The next control message of `type` (others are skipped).
async fn next_msg(peer: &mut Peer, ty: &str) -> Value {
    tokio::time::timeout(T, async {
        loop {
            match peer.next_event().await {
                Some(PeerEvent::Message(m)) => {
                    let v: Value = serde_json::from_str(m.as_text().unwrap()).unwrap();
                    if v["type"] == ty {
                        return v;
                    }
                }
                Some(PeerEvent::Closed(r)) => panic!("closed waiting for {ty}: {r:?}"),
                Some(_) => {}
                None => panic!("peer gone waiting for {ty}"),
            }
        }
    })
    .await
    .unwrap_or_else(|_| panic!("no {ty}"))
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn director_session_copies_to_whip() {
    let a = app().await;
    let (url, rx) = endpoint().await;
    let host = RtcHost::bind(HostConfig::loopback(true, false)).await.unwrap();
    let (pending, offer) = host
        .offer(OfferOptions {
            video: Some(Direction::RecvOnly),
            audio: Some((Direction::RecvOnly, AudioLayout::Stereo)),
            channels: vec!["control".into()],
            // VP8 too: a build without OpenH264 answers the client with VP8.
            video_codecs: vec![fastvideo_webrtc::writer::VideoCodec::H264, fastvideo_webrtc::writer::VideoCodec::Vp8],
            ..Default::default()
        })
        .await
        .unwrap();
    let (s, v) = post_json(&a.router, "/wma/session", json!({"app_id": "minimax/h3-max/director", "sdp": offer, "type": "offer"})).await;
    assert_eq!(s, 200, "{v}");
    let mut peer = pending.accept_answer(v["sdp"].as_str().unwrap()).await.unwrap();
    let info = next_msg(&mut peer, "session_info").await;
    assert_eq!(info["x_fv_output"]["available"], true, "{info}");
    assert_eq!(info["x_fv_output"]["fields"], json!(["x_fv_output"]));
    assert!(info["x_fv_output"]["server_message_types"].as_array().unwrap().iter().any(|t| t == "output_ready"));
    let ctl = |m: Value| ChannelMessage::text("control", m.to_string());
    // `output` is not in fal's schema: refused without [director] allow_output.
    let out = json!({"kind": "whip", "url": url, "token": "copy-secret-token"});
    peer.send_message(ctl(json!({"type": "configure", "prompt_version": 1, "prompt": "a lighthouse", "output": out}))).await.unwrap();
    let e = next_msg(&mut peer, "error").await;
    assert_eq!(e["code"], "invalid_message", "{e}");
    assert!(!e.to_string().contains("copy-secret-token"));
    // A bad x_fv_output refuses the configure, naming the field.
    peer.send_message(ctl(json!({"type": "configure", "prompt_version": 1, "prompt": "a lighthouse", "x_fv_output": {"kind": "rtmps", "url": url}}))).await.unwrap();
    let e = next_msg(&mut peer, "error").await;
    assert_eq!(e["code"], "invalid_input", "{e}");
    assert!(e["error"].as_str().unwrap_or_default().starts_with("x_fv_output: output.url") || e.to_string().contains("x_fv_output"), "{e}");
    // `x_fv_output`: configured echoes the redacted output, then output_ready.
    peer.send_message(ctl(json!({"type": "configure", "prompt_version": 1, "prompt": "a lighthouse", "x_fv_output": out}))).await.unwrap();
    let c = next_msg(&mut peer, "configured").await;
    assert_eq!(c["x_fv_output"]["kind"], "whip", "{c}");
    assert_eq!(c["x_fv_output"]["credentials"], "set");
    assert!(!c.to_string().contains("copy-secret-token"));
    let ready = next_msg(&mut peer, "output_ready").await;
    assert_eq!(ready["output"]["kind"], "whip", "{ready}");
    let t0 = std::time::Instant::now();
    while rx.lock().unwrap().video.len() < 24 {
        assert!(t0.elapsed() < T, "the copy delivered {} frames", rx.lock().unwrap().video.len());
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    {
        let rx = rx.lock().unwrap();
        let idr = rx.video.iter().find(|v| v.0).expect("an IDR");
        let sps = fastvideo_media::h264::find_sps(&idr.1).expect("SPS");
        assert_eq!((sps.width, sps.height), (1344, 768), "the session's native canvas");
        assert!(rx.audio > 0, "the session's audio is copied");
    }
    // Stop: the copy ends with the session (WHIP DELETE).
    peer.send_message(ctl(json!({"type": "stop"}))).await.unwrap();
    let t0 = std::time::Instant::now();
    while rx.lock().unwrap().deletes == 0 {
        assert!(t0.elapsed() < Duration::from_secs(20), "no WHIP DELETE after stop");
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
}
