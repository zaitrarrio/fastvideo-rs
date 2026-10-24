//! Stream outputs end to end on the fake engine (docs/serve/stream-outputs.md):
//! the Cloudflare default against a fake Stream API whose live inputs point
//! their WebRTC ingest at an in-process WHIP endpoint (a str0m host that
//! records what arrives), VP8 on request, request validation and secret
//! redaction, and the Cloudflare error path.
//!
//! The real Cloudflare and MediaMTX runs are in `outputs_live.rs` (gated).

#![cfg(all(feature = "webrtc", feature = "http-client", feature = "encoders"))]

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use axum::body::Body;
use axum::extract::{Path, State};
use axum::http::{HeaderMap, Request, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{delete, post};
use axum::{Json, Router};
use fastvideo_serve::config::{Config, JobBackend};
use fastvideo_serve::{App, Overrides};
use fastvideo_serve_kit::KeyRing;
use fastvideo_webrtc::channel::ChannelPolicy;
use fastvideo_webrtc::host::{AnswerOptions, AudioLayout, HostConfig, Peer, PeerEvent, RtcHost};
use fastvideo_webrtc::sdp::{MediaKind, Sdp};
use fastvideo_webrtc::writer::{TrackKind, VideoCodec};
use serde_json::{json, Value};
use tower::ServiceExt;

const KEY: &str = "sk-outputs";
const CF_TOKEN: &str = "cf-stream-token-do-not-print";
/// The WHIP publish key the fake API puts in the WebRTC URL path.
const PUBLISH_KEY: &str = "b236bde30eb07b9d01318940e5fc3edake34a3efb3896e18f2dc277ce6cc993ad";
const RTMPS_KEY: &str = "MTQ0MTcjM3MjI1NDE3ODIyNTI1MjYyMjE4NTI2ODI1NDcxMzUyMzcf";
const SRT_PASS: &str = "766d7a3dfb1bf6cbb7c0d1b5d3a0a4e3";

// ---- fake Cloudflare Stream API ------------------------------------------------

#[derive(Default)]
struct CfSeen {
    creates: Vec<(Option<String>, Value)>,
    deletes: Vec<(String, Option<String>)>,
    disables: Vec<(String, Value)>,
    /// Answer every create with this status and Cloudflare error.
    fail: Option<(u16, u32, String)>,
}

#[derive(Clone)]
struct Cf {
    seen: Arc<Mutex<CfSeen>>,
    whip_base: String,
}

fn hdr(h: &HeaderMap, k: &str) -> Option<String> {
    h.get(k).and_then(|v| v.to_str().ok()).map(str::to_owned)
}

async fn cf_create(
    State(cf): State<Cf>,
    Path(acct): Path<String>,
    h: HeaderMap,
    Json(body): Json<Value>,
) -> Response {
    assert_eq!(acct, "acct123");
    let mut seen = cf.seen.lock().unwrap();
    seen.creates.push((hdr(&h, "authorization"), body.clone()));
    if let Some((status, code, msg)) = seen.fail.clone() {
        return (
            StatusCode::from_u16(status).unwrap(),
            Json(json!({"success": false, "errors": [{"code": code, "message": msg}], "messages": [], "result": null})),
        )
            .into_response();
    }
    let uid = format!("{:032x}", seen.creates.len());
    let rec = body["recording"]["mode"].clone();
    Json(json!({
        "success": true, "errors": [], "messages": [],
        "result": {
            "uid": uid,
            "meta": body["meta"],
            "recording": {"mode": rec},
            // Unreachable ingests on this host (port 9: nothing listens).
            "rtmps": {"url": "rtmps://localhost:9/live/", "streamKey": RTMPS_KEY},
            "srt": {"url": "srt://localhost:9", "streamId": "f256e6ea9341d51eea64c9454659e576", "passphrase": SRT_PASS},
            "webRTC": {"url": format!("{}/{PUBLISH_KEY}/webRTC/publish", cf.whip_base)},
            "webRTCPlayback": {"url": format!("https://customer-abc123.cloudflarestream.com/{uid}/webRTC/play")},
        }
    }))
    .into_response()
}

async fn cf_put(
    State(cf): State<Cf>,
    Path((_acct, uid)): Path<(String, String)>,
    Json(body): Json<Value>,
) -> Json<Value> {
    cf.seen.lock().unwrap().disables.push((uid, body));
    Json(json!({"success": true, "errors": [], "messages": [], "result": null}))
}

async fn cf_delete(
    State(cf): State<Cf>,
    Path((_acct, uid)): Path<(String, String)>,
    h: HeaderMap,
) -> Json<Value> {
    cf.seen
        .lock()
        .unwrap()
        .deletes
        .push((uid, hdr(&h, "authorization")));
    Json(json!({"success": true, "errors": [], "messages": [], "result": null}))
}

// ---- WHIP endpoint (the "Cloudflare WebRTC ingest") -----------------------------

#[derive(Default)]
struct Rx {
    posts: usize,
    deletes: usize,
    offers: Vec<String>,
    video: Vec<(u64, bool, bytes::Bytes)>,
    audio: usize,
}

#[derive(Clone)]
struct Endpoint {
    host: RtcHost,
    rx: Arc<Mutex<Rx>>,
}

async fn whip_post(State(m): State<Endpoint>, Path(key): Path<String>, body: String) -> Response {
    assert_eq!(key, PUBLISH_KEY, "the publish key is the credential");
    m.rx.lock().unwrap().posts += 1;
    m.rx.lock().unwrap().offers.push(body.clone());
    let audio = Sdp::parse(&body).ok().and_then(|s| {
        s.media
            .iter()
            .any(|x| x.kind() == MediaKind::Audio)
            .then_some(AudioLayout::Stereo)
    });
    let opts = AnswerOptions {
        audio,
        channels: ChannelPolicy::none(),
        // Cloudflare's WHIP codecs.
        video_codecs: vec![VideoCodec::H264, VideoCodec::Vp8, VideoCodec::Vp9],
        ..Default::default()
    };
    match m.host.answer(&body, opts).await {
        Ok((peer, answer)) => {
            let id = peer.id();
            tokio::spawn(collect(peer, m.rx.clone()));
            (
                StatusCode::CREATED,
                [
                    ("content-type", "application/sdp".to_string()),
                    ("location", format!("publish/{id}")),
                ],
                answer,
            )
                .into_response()
        }
        Err(e) => (StatusCode::BAD_REQUEST, e.to_string()).into_response(),
    }
}

async fn collect(mut peer: Peer, rx: Arc<Mutex<Rx>>) {
    while let Some(ev) = peer.next_event().await {
        match ev {
            PeerEvent::Media {
                kind: TrackKind::Video,
                rtp_time,
                keyframe,
                data,
                ..
            } => rx.lock().unwrap().video.push((rtp_time, keyframe, data)),
            PeerEvent::Media {
                kind: TrackKind::Audio,
                ..
            } => rx.lock().unwrap().audio += 1,
            PeerEvent::Closed(_) => break,
            _ => {}
        }
    }
}

async fn whip_delete(
    State(m): State<Endpoint>,
    Path((_key, _id)): Path<(String, String)>,
) -> StatusCode {
    m.rx.lock().unwrap().deletes += 1;
    StatusCode::OK
}

/// The fake Stream API and its WHIP ingest on one listener.
async fn fake_cloudflare() -> (String, Arc<Mutex<CfSeen>>, Arc<Mutex<Rx>>) {
    let host = RtcHost::bind(HostConfig::loopback(true, false))
        .await
        .unwrap();
    let rx = Arc::new(Mutex::new(Rx::default()));
    let seen = Arc::new(Mutex::new(CfSeen::default()));
    let l = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let base = format!("http://{}", l.local_addr().unwrap());
    let whip = Router::new()
        .route("/{key}/webRTC/publish", post(whip_post))
        .route("/{key}/webRTC/publish/{id}", delete(whip_delete))
        .with_state(Endpoint {
            host,
            rx: rx.clone(),
        });
    let api = Router::new()
        .route(
            "/client/v4/accounts/{acct}/stream/live_inputs",
            post(cf_create),
        )
        .route(
            "/client/v4/accounts/{acct}/stream/live_inputs/{uid}",
            delete(cf_delete).put(cf_put),
        )
        .with_state(Cf {
            seen: seen.clone(),
            whip_base: format!("{base}/ingest"),
        });
    let app = Router::new().nest("/ingest", whip).merge(api);
    tokio::spawn(async move { axum::serve(l, app).await.unwrap() });
    (base, seen, rx)
}

// ---- the server ------------------------------------------------------------------

async fn app_with(extra: &[(&str, &str)]) -> App {
    let dir = tempfile::Builder::new()
        .prefix("fv-serve-outputs-")
        .tempdir()
        .unwrap()
        .keep();
    let mut env = BTreeMap::new();
    env.insert("FV_API_KEYS".to_owned(), KeyRing::hash_hex(KEY));
    env.insert("FV_URL_SIGNING_KEY".to_owned(), "k".to_owned());
    env.insert("FV_STATE_DIR".to_owned(), dir.display().to_string());
    env.insert("FV_PUBLIC_BASE_URL".to_owned(), "http://fv.test".to_owned());
    // ffmpeg libx264 (the CPU test encoder) keeps a native 1344x768 stream in real time.
    env.insert("FV_STREAM_ENCODER".to_owned(), "x264-test".to_owned());
    for (k, v) in extra {
        env.insert((*k).to_owned(), (*v).to_owned());
    }
    let mut c = Config::default();
    c.apply_env(&env).unwrap();
    c.jobs.backend = JobBackend::Memory;
    c.engine.fake.step_ms = 1;
    c.validate().unwrap();
    std::env::set_var("FV_STREAM_STUN", "none");
    let a = App::build(c, Overrides::default()).await.unwrap();
    a.gate.engine().wait_ready().await;
    a
}

fn cf_env(base: &str) -> Vec<(&'static str, String)> {
    vec![
        ("FV_CF_ACCOUNT_ID", "acct123".into()),
        ("FV_CF_STREAM_API_TOKEN", CF_TOKEN.into()),
        ("FV_CF_STREAM_API_BASE", format!("{base}/client/v4")),
        ("FV_CF_STREAM_NAME_PREFIX", "fv-test-".into()),
    ]
}

async fn cf_app(base: &str, extra: &[(&str, &str)]) -> App {
    let env = cf_env(base);
    let mut all: Vec<(&str, &str)> = env.iter().map(|(k, v)| (*k, v.as_str())).collect();
    all.extend_from_slice(extra);
    app_with(&all).await
}

async fn call(app: &Router, method: &str, uri: &str, body: Option<Value>) -> (StatusCode, String) {
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
    let r = app.clone().oneshot(req).await.unwrap();
    let s = r.status();
    let bytes = axum::body::to_bytes(r.into_body(), 1 << 24).await.unwrap();
    (s, String::from_utf8_lossy(&bytes).into_owned())
}

fn json_of(s: &str) -> Value {
    serde_json::from_str(s).unwrap_or(Value::Null)
}

/// No response may carry a credential.
fn assert_clean(text: &str) {
    for secret in [
        CF_TOKEN,
        PUBLISH_KEY,
        RTMPS_KEY,
        SRT_PASS,
        "user-secret-key",
        "s3cret-passphrase",
    ] {
        assert!(!text.contains(secret), "secret {secret} leaked: {text}");
    }
}

async fn wait<F: Fn() -> bool>(what: &str, secs: u64, f: F) {
    let t0 = std::time::Instant::now();
    while !f() {
        assert!(
            t0.elapsed() < Duration::from_secs(secs),
            "timed out waiting for {what}"
        );
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
}

// ---- tests -----------------------------------------------------------------------

/// No `output`: a Cloudflare live input `fv-test-<id>` (recording off), WHIP
/// at the native 1344x768 with `42e01f` offered, WHEP playback in the
/// stream object, nothing secret in any response, the input deleted on stop.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn cloudflare_default_publishes_native_h264_and_cleans_up() {
    let (base, seen, rx) = fake_cloudflare().await;
    let a = cf_app(&base, &[]).await;
    let (s, info) = call(&a.router, "GET", "/fv/v1/outputs", None).await;
    assert_eq!(s, StatusCode::OK, "{info}");
    let info = json_of(&info);
    assert_eq!(info["default"], "cloudflare");
    assert_eq!(info["kinds"].as_array().unwrap().len(), 6);
    assert_clean(&info.to_string());

    // fake-h3-turbo: a clip model at H3's native 1344x768 with audio.
    let body = json!({"model": "fake-h3-turbo", "prompt": "a lighthouse at dusk"});
    let (s, text) = call(&a.router, "POST", "/fv/v1/streams", Some(body)).await;
    assert_eq!(s, StatusCode::CREATED, "{text}");
    assert_clean(&text);
    let v = json_of(&text);
    let id = v["id"].as_str().unwrap().to_owned();
    assert_eq!(v["output"]["kind"], "cloudflare");
    assert_eq!(v["output"]["codec"], "h264");
    assert_eq!(v["playback"]["provider"], "cloudflare");
    assert_eq!(v["playback"]["format"], "whep");
    let uid = v["playback"]["live_input_id"].as_str().unwrap().to_owned();
    assert_eq!(
        v["playback"]["whep_url"],
        format!("https://customer-abc123.cloudflarestream.com/{uid}/webRTC/play")
    );
    assert_eq!(
        v["playback"]["iframe_url"],
        format!("https://customer-abc123.cloudflarestream.com/{uid}/iframe")
    );
    assert!(
        v["playback"].get("hls_url").is_none(),
        "WebRTC inputs have no HLS"
    );
    {
        let seen = seen.lock().unwrap();
        assert_eq!(seen.creates.len(), 1);
        let (auth, body) = &seen.creates[0];
        assert_eq!(auth.as_deref(), Some(format!("Bearer {CF_TOKEN}").as_str()));
        assert_eq!(body["meta"]["name"], format!("fv-test-{id}"));
        assert_eq!(body["recording"]["mode"], "off");
    }

    wait("48 video frames at the ingest", 90, || {
        rx.lock().unwrap().video.len() >= 48
    })
    .await;
    let (s, text) = call(&a.router, "GET", &format!("/fv/v1/streams/{id}"), None).await;
    assert_eq!(s, StatusCode::OK);
    assert_clean(&text);
    let v = json_of(&text);
    assert_eq!(v["status"]["state"], "streaming", "{v}");
    assert_eq!(v["status"]["output_state"], "publishing");
    assert_eq!(v["status"]["video_codec"], "H264");
    assert_eq!(
        (
            v["status"]["width"].as_u64(),
            v["status"]["height"].as_u64()
        ),
        (Some(1344), Some(768)),
        "native canvas"
    );
    let resource = v["status"]["whip_resource"].as_str().unwrap();
    assert!(
        resource.contains("/<key>/webRTC/publish/"),
        "redacted resource: {resource}"
    );
    {
        let rx = rx.lock().unwrap();
        let offer = &rx.offers[0];
        assert!(
            offer.contains("profile-level-id=42e01f"),
            "Cloudflare's documented H.264 offer"
        );
        assert!(!offer.contains("VP8"), "one codec offered");
        // The bitstream: native 1344x768 at the level it needs (3.2).
        let idr = rx.video.iter().find(|x| x.1).expect("an IDR");
        let sps = fastvideo_media::h264::find_sps(&idr.2).expect("SPS with the IDR");
        assert_eq!((sps.width, sps.height, sps.level_idc), (1344, 768, 32));
        assert!(sps.is_constrained_baseline());
        assert!(rx.audio > 0, "Opus flows for an A/V model");
    }
    let (s, text) = call(&a.router, "DELETE", &format!("/fv/v1/streams/{id}"), None).await;
    assert_eq!(s, StatusCode::OK);
    assert_clean(&text);
    wait("WHIP DELETE", 10, || rx.lock().unwrap().deletes == 1).await;
    wait("live input DELETE", 10, || {
        seen.lock().unwrap().deletes.len() == 1
    })
    .await;
    let seen = seen.lock().unwrap();
    assert_eq!(seen.deletes[0].0, uid);
    assert_eq!(
        seen.deletes[0].1.as_deref(),
        Some(format!("Bearer {CF_TOKEN}").as_str())
    );
}

/// `codec: vp8`: only VP8 is offered and Cloudflare's ingest gets VP8
/// keyframes of the native canvas (ffmpeg libvpx).
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn cloudflare_vp8_on_request() {
    if !fastvideo_media::vp8::libvpx_available() {
        eprintln!("skipping: ffmpeg has no libvpx");
        return;
    }
    let (base, _seen, rx) = fake_cloudflare().await;
    let a = cf_app(&base, &[]).await;
    let body = json!({"model": "fake-sfwan", "prompt": "a road", "width": 832, "height": 480, "output": {"kind": "cloudflare", "codec": "vp8"}});
    let (s, text) = call(&a.router, "POST", "/fv/v1/streams", Some(body)).await;
    assert_eq!(s, StatusCode::CREATED, "{text}");
    let id = json_of(&text)["id"].as_str().unwrap().to_owned();
    wait("30 VP8 frames", 90, || rx.lock().unwrap().video.len() >= 30).await;
    let (_, text) = call(&a.router, "GET", &format!("/fv/v1/streams/{id}"), None).await;
    let v = json_of(&text);
    assert_eq!(v["status"]["video_codec"], "VP8", "{v}");
    assert_eq!(v["output"]["codec"], "vp8");
    {
        let rx = rx.lock().unwrap();
        let offer = Sdp::parse(&rx.offers[0]).unwrap();
        let video = offer
            .media
            .iter()
            .find(|m| m.kind() == MediaKind::Video)
            .unwrap();
        assert!(
            video.codecs().iter().all(|c| c.is("VP8") || c.is("rtx")),
            "{}",
            video.m_line
        );
        let key = rx
            .video
            .iter()
            .find(|x| fastvideo_media::vp8::is_keyframe(&x.2))
            .expect("a VP8 keyframe");
        assert_eq!(
            fastvideo_media::decode::vp8_keyframe_size(&key.2),
            Some((832, 480))
        );
    }
    call(&a.router, "DELETE", &format!("/fv/v1/streams/{id}"), None).await;
}

/// Bad requests are refused before a session opens, without echoing
/// credentials; private destinations need `[outputs] allow_private`; a
/// Cloudflare refusal (no Stream permission) is a clear error and leaves the
/// executor free.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn refusals_and_cloudflare_errors() {
    // No Cloudflare account: an output is required.
    let a = app_with(&[]).await;
    let (s, text) = call(
        &a.router,
        "POST",
        "/fv/v1/streams",
        Some(json!({"model": "fake-sfwan", "prompt": "p"})),
    )
    .await;
    assert_eq!(s, StatusCode::BAD_REQUEST, "{text}");
    assert_eq!(json_of(&text)["error"]["param"], "output");
    let (_, info) = call(&a.router, "GET", "/fv/v1/outputs", None).await;
    assert_eq!(json_of(&info)["default"], Value::Null);
    // Loopback refused (the old `whip_url` and `output` alike).
    for body in [
        json!({"model": "fake-sfwan", "prompt": "p", "whip_url": "http://127.0.0.1:8889/live/whip"}),
        json!({"model": "fake-sfwan", "prompt": "p", "output": {"kind": "srt", "url": "srt://10.1.2.3:9000", "passphrase": "s3cret-passphrase"}}),
        json!({"model": "fake-sfwan", "prompt": "p", "output": {"kind": "rtmps", "url": "rtmp://localhost/live", "stream_key": "user-secret-key"}}),
    ] {
        let (s, text) = call(&a.router, "POST", "/fv/v1/streams", Some(body)).await;
        assert_eq!(s, StatusCode::BAD_REQUEST, "{text}");
        assert!(text.contains("allow_private"), "{text}");
        assert_clean(&text);
    }
    // Shape errors name the field, never the value.
    for (body, param) in [
        (
            json!({"kind": "rtmps", "url": "https://x.example/live", "stream_key": "user-secret-key"}),
            "output.url",
        ),
        (
            json!({"kind": "whip", "url": "https://x.example/w", "codec": "av1", "token": "user-secret-key"}),
            "output.codec",
        ),
        (
            json!({"kind": "cloudflare-srt", "passphrase": "s3cret-passphrase"}),
            "output.passphrase",
        ),
        (
            json!({"kind": "srt", "url": "srt://x.example:9", "nope": "user-secret-key"}),
            "output",
        ),
    ] {
        let (s, text) = call(
            &a.router,
            "POST",
            "/fv/v1/streams",
            Some(json!({"model": "fake-sfwan", "prompt": "p", "output": body})),
        )
        .await;
        assert_eq!(s, StatusCode::BAD_REQUEST, "{text}");
        assert_clean(&text);
        let e = json_of(&text);
        assert!(
            e["error"]["param"].as_str().unwrap().starts_with(param),
            "{text}"
        );
    }

    // Cloudflare refuses the token (no Stream permission): a clear 500 that
    // names the permission, the session released.
    let (base, seen, _rx) = fake_cloudflare().await;
    seen.lock().unwrap().fail = Some((403, 10000, "Authentication error".into()));
    let a = cf_app(&base, &[]).await;
    let (s, text) = call(
        &a.router,
        "POST",
        "/fv/v1/streams",
        Some(json!({"model": "fake-sfwan", "prompt": "p"})),
    )
    .await;
    assert_eq!(s, StatusCode::INTERNAL_SERVER_ERROR, "{text}");
    assert!(
        text.contains("10000 Authentication error") && text.contains("Stream > Edit"),
        "{text}"
    );
    assert_clean(&text);
    seen.lock().unwrap().fail = None;
    // The executor is free again: the next stream starts.
    let (s, text) = call(
        &a.router,
        "POST",
        "/fv/v1/streams",
        Some(json!({"model": "fake-sfwan", "prompt": "p"})),
    )
    .await;
    assert_eq!(s, StatusCode::CREATED, "{text}");
    let id = json_of(&text)["id"].as_str().unwrap().to_owned();
    call(&a.router, "DELETE", &format!("/fv/v1/streams/{id}"), None).await;
    wait("live input DELETE", 10, || {
        seen.lock().unwrap().deletes.len() == 1
    })
    .await;
}

/// RTMPS / SRT through Cloudflare: the inputs record (HLS needs it), the
/// stream object carries HLS / DASH, and on stop the input is disabled, not
/// deleted (`on_stop = auto` keeps the recording reachable). The ingest is
/// unreachable here (no RTMP server): the output reports the error and keeps
/// the stream alive.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn cloudflare_rtmps_and_srt_get_hls_playback() {
    let (base, seen, _rx) = fake_cloudflare().await;
    let a = cf_app(&base, &[]).await;
    for kind in ["cloudflare-rtmps", "cloudflare-srt"] {
        let body = json!({"model": "fake-sfwan", "prompt": "p", "width": 64, "height": 32, "output": {"kind": kind}});
        let (s, text) = call(&a.router, "POST", "/fv/v1/streams", Some(body)).await;
        assert_eq!(s, StatusCode::CREATED, "{text}");
        assert_clean(&text);
        let v = json_of(&text);
        let id = v["id"].as_str().unwrap().to_owned();
        let uid = v["playback"]["live_input_id"].as_str().unwrap().to_owned();
        assert_eq!(v["playback"]["format"], "hls");
        assert_eq!(
            v["playback"]["hls_url"],
            format!("https://customer-abc123.cloudflarestream.com/{uid}/manifest/video.m3u8")
        );
        assert_eq!(
            v["playback"]["dash_url"],
            format!("https://customer-abc123.cloudflarestream.com/{uid}/manifest/video.mpd")
        );
        assert_eq!(
            seen.lock().unwrap().creates.last().unwrap().1["recording"]["mode"],
            "automatic"
        );
        // ffmpeg starts and fails to reach the (fake) ingest: reported, scrubbed.
        let t0 = std::time::Instant::now();
        let mut st = Value::Null;
        while t0.elapsed() < Duration::from_secs(30) {
            let (_, text) = call(&a.router, "GET", &format!("/fv/v1/streams/{id}"), None).await;
            assert_clean(&text);
            st = json_of(&text)["status"].clone();
            if st["output_error"].is_string() {
                break;
            }
            tokio::time::sleep(Duration::from_millis(200)).await;
        }
        assert!(
            st["output_error"].is_string(),
            "the unreachable ingest is reported: {st}"
        );
        assert_eq!(
            st["state"], "streaming",
            "a sink failure does not end the stream: {st}"
        );
        let (_, text) = call(&a.router, "DELETE", &format!("/fv/v1/streams/{id}"), None).await;
        assert_clean(&text);
    }
    // Recording inputs are disabled (PUT enabled=false), not deleted.
    wait("two disables", 10, || {
        seen.lock().unwrap().disables.len() == 2
    })
    .await;
    let seen = seen.lock().unwrap();
    assert!(seen.deletes.is_empty());
    assert!(seen
        .disables
        .iter()
        .all(|d| d.1 == json!({"enabled": false})));
}
