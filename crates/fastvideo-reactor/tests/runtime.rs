//! End-to-end tests of the Reactor runtime on the fake engine: HTTP routes
//! (tower oneshot), a loopback str0m client standing in for an SDK peer
//! (offering VP8 like the Python reactor_sdk), v1 and v0 wire traffic,
//! the pause gate, the black start frame, the watchdog and the orphan
//! timeout.
//!
//! Waits are event-driven: [`pump`] returns as soon as its condition holds,
//! and its limit ([`T`]) only bounds a hang, so a loaded host (the shared
//! build pod) passes. Frame counts that hold only when the host keeps real
//! time (a clip delivered frame for frame, a take's frame floor) are checked
//! by the `realtime_` twins, ignored by default and run serialized by
//! `scripts/serve/check.sh --realtime` (docs/dev/testing.md).

use std::sync::Arc;
use std::time::{Duration, Instant};

use axum::body::Body;
use axum::http::{Request, StatusCode};
use axum::Router;
use fastvideo_engine_service::{EngineConfig, EngineService, FakeBackend, FakeConfig, FakeModel, FakeTiming, Mp4Mode};
use fastvideo_reactor::pb;
use fastvideo_reactor::wire::json_to_struct;
use fastvideo_protocol::CausalLimits;
use fastvideo_reactor::{router, session_limit_reason, H264Backend, Reactor, ReactorConfig, RtState, SESSION_ID};
use fastvideo_webrtc::channel::ChannelMessage;
use fastvideo_webrtc::host::{AudioLayout, HostConfig, OfferOptions, Peer, PeerEvent, RtcHost};
use fastvideo_webrtc::sdp::{Direction, MediaKind, Sdp};
use fastvideo_webrtc::writer::{TrackKind, VideoCodec};
use prost::Message as _;
use serde_json::{json, Value};
use tower::ServiceExt;

const W: &str = "/sessions/00000000-0000-0000-0000-000000000000/transport/webrtc";

/// How long a wait for something that must happen may take before the test
/// fails. Generous: video waits for an ffmpeg encoder process, which starts
/// in seconds on a loaded host.
const T: Duration = Duration::from_secs(60);

fn engine(model: FakeModel, load: Duration) -> EngineService {
    let fc = FakeConfig {
        models: vec![model],
        timing: FakeTiming { load, rtf: Some(0.1), step: Duration::from_millis(2), ..FakeTiming::default() },
        mp4: Mp4Mode::Off,
        ..FakeConfig::default()
    };
    EngineService::start(
        EngineConfig { output_dir: std::env::temp_dir().join("fv-reactor-tests"), ..EngineConfig::default() },
        vec![Box::new(FakeBackend::new(fc))],
    )
    .unwrap()
}

async fn runtime(e: EngineService, tweak: impl FnOnce(&mut ReactorConfig)) -> (Reactor, Router) {
    let host = RtcHost::bind(HostConfig::loopback(true, false)).await.unwrap();
    let mut cfg = ReactorConfig {
        short_edge: Some(96),
        h264: H264Backend::Off,
        seed: Some(3),
        latch_grace: Duration::from_millis(500),
        ..ReactorConfig::default()
    };
    tweak(&mut cfg);
    let rt = Reactor::new(cfg, Arc::new(e), host);
    let app = router(rt.clone());
    (rt, app)
}

async fn call(app: &Router, method: &str, uri: &str, body: Option<Value>) -> (StatusCode, axum::http::HeaderMap, Value) {
    let mut b = Request::builder().method(method).uri(uri);
    b = b.header("reactor-webrtc-version", "1.0");
    let req = match body {
        Some(v) => b.header("content-type", "application/json").body(Body::from(v.to_string())).unwrap(),
        None => b.body(Body::empty()).unwrap(),
    };
    let resp = app.clone().oneshot(req).await.unwrap();
    let (parts, body) = resp.into_parts();
    let bytes = axum::body::to_bytes(body, 1 << 20).await.unwrap();
    let v = if bytes.is_empty() { Value::Null } else { serde_json::from_slice(&bytes).unwrap_or(Value::Null) };
    (parts.status, parts.headers, v)
}

struct Client {
    peer: Peer,
    #[allow(dead_code)]
    host: RtcHost,
    cid: u64,
}

/// Registers a connection, offers (VP8 + optional audio, both channels),
/// polls the answer and waits until both channels are open.
async fn connect(app: &Router, audio: bool) -> Client {
    let (s, _, v) = call(app, "POST", &format!("{W}/connections"), Some(json!({}))).await;
    assert_eq!(s, StatusCode::CREATED, "{v}");
    let cid = v["connection_id"].as_u64().unwrap();
    assert!((1002..=9999).contains(&cid), "{cid}");
    let host = RtcHost::bind(HostConfig::loopback(true, false)).await.unwrap();
    let (pending, offer) = host
        .offer(OfferOptions {
            video: Some(Direction::RecvOnly),
            audio: Some((Direction::RecvOnly, AudioLayout::Mono)),
            channels: vec!["data".into(), "control".into()],
            video_codecs: vec![VideoCodec::Vp8],
            ..OfferOptions::default()
        })
        .await
        .unwrap();
    let sdp = Sdp::parse(&offer).unwrap();
    let mid = |k: MediaKind| sdp.media.iter().find(|m| m.kind() == k).and_then(|m| m.mid()).unwrap().to_owned();
    let mut mapping = vec![json!({"mid": mid(MediaKind::Video), "name": "main_video", "kind": "video", "direction": "recvonly"})];
    if audio {
        mapping.push(json!({"mid": mid(MediaKind::Audio), "name": "main_audio", "kind": "audio", "direction": "recvonly"}));
    }
    // Candidates may arrive before the offer: buffered.
    let (s, _, _) = call(
        app,
        "POST",
        &format!("{W}/connections/{cid}/ice_candidates"),
        Some(json!({"candidates": [{"candidate": "", "sdp_mid": "0", "sdp_mline_index": 0}], "is_final": false})),
    )
    .await;
    assert_eq!(s, StatusCode::ACCEPTED);
    let (s, _, v) = call(
        app,
        "POST",
        &format!("{W}/connections/{cid}/sdp_params"),
        Some(json!({"sdp_offer": offer, "track_mapping": mapping, "client_info": {"sdk_version": "test"}})),
    )
    .await;
    assert_eq!(s, StatusCode::ACCEPTED, "{v}");
    assert_eq!(v["connection_id"].as_u64(), Some(cid));
    let t0 = Instant::now();
    let answer = loop {
        let (s, _, v) = call(app, "GET", &format!("{W}/connections/{cid}/sdp_params"), None).await;
        if s == StatusCode::OK {
            assert_eq!(v["connection_id"].as_u64(), Some(cid));
            break v["sdp_answer"].as_str().unwrap().to_owned();
        }
        assert_eq!(s, StatusCode::ACCEPTED, "{v}");
        assert!(t0.elapsed() < T, "no answer");
        tokio::time::sleep(Duration::from_millis(20)).await;
    };
    // Taken: a second GET is 202 again.
    let (s, _, _) = call(app, "GET", &format!("{W}/connections/{cid}/sdp_params"), None).await;
    assert_eq!(s, StatusCode::ACCEPTED);
    let a = Sdp::parse(&answer).unwrap();
    assert!(a.has_end_of_candidates(), "{answer}");
    assert!(!answer.contains("x-reactor-frame-metadata"));
    assert!(answer.contains("VP8/90000"), "{answer}");
    let mut peer = pending.accept_answer(&answer).await.unwrap();
    let mut open = std::collections::HashSet::new();
    let mut connected = false;
    let t0 = Instant::now();
    while !(connected && open.len() == 2) {
        let ev = tokio::time::timeout(T, peer.next_event()).await.expect("connect timeout");
        match ev {
            Some(PeerEvent::Connected) => connected = true,
            Some(PeerEvent::ChannelOpen { label }) => {
                open.insert(label);
            }
            Some(PeerEvent::Closed(r)) => panic!("closed while connecting: {r:?}"),
            Some(_) => {}
            None => panic!("peer gone"),
        }
        assert!(t0.elapsed() < T, "connect timeout");
    }
    Client { peer, host, cid }
}

fn control(p: pb::control_client_message::Payload, rid: &str, kind: pb::MessageKind) -> ChannelMessage {
    let m = pb::ControlClientMessage { request_id: rid.into(), kind: kind as i32, payload: Some(p) };
    ChannelMessage::binary("control", m.encode_to_vec())
}

fn command(name: &str, data: Value, rid: &str) -> ChannelMessage {
    let m = pb::DataClientMessage {
        request_id: rid.into(),
        kind: pb::MessageKind::Request as i32,
        payload: Some(pb::data_client_message::Payload::Command(pb::Command {
            r#type: name.into(),
            data: Some(json_to_struct(&data)),
            uploads: Default::default(),
        })),
    };
    ChannelMessage::binary("data", m.encode_to_vec())
}

#[derive(Debug, Default)]
struct Seen {
    control: Vec<pb::ControlServerMessage>,
    data: Vec<pb::DataServerMessage>,
    text: Vec<(String, Value)>,
    video: Vec<(bool, usize)>,
    audio: usize,
    closed: bool,
}

impl Seen {
    fn model(&self, t: &str) -> Vec<(String, Value)> {
        self.data
            .iter()
            .filter_map(|m| match &m.payload {
                Some(pb::data_server_message::Payload::Message(mm)) if mm.r#type == t => Some((
                    m.request_id.clone(),
                    Value::Object(fastvideo_reactor::wire::struct_to_map(mm.data.clone().unwrap_or_default())),
                )),
                _ => None,
            })
            .collect()
    }
    fn reply(&self, rid: &str) -> Option<&pb::DataServerMessage> {
        self.data.iter().find(|m| m.request_id == rid)
    }
}

/// Drains events for `d`, or until `until` holds.
async fn pump(peer: &mut Peer, seen: &mut Seen, d: Duration, until: impl Fn(&Seen) -> bool) {
    let end = Instant::now() + d;
    while Instant::now() < end && !until(seen) {
        let left = end.saturating_duration_since(Instant::now());
        match tokio::time::timeout(left, peer.next_event()).await {
            Ok(Some(PeerEvent::Message(m))) => {
                if !m.binary {
                    let v: Value = serde_json::from_slice(&m.data).unwrap();
                    seen.text.push((m.label.clone(), v));
                } else if m.label == "control" {
                    seen.control.push(pb::ControlServerMessage::decode(&m.data[..]).unwrap());
                } else {
                    seen.data.push(pb::DataServerMessage::decode(&m.data[..]).unwrap());
                }
            }
            Ok(Some(PeerEvent::Media { kind: TrackKind::Video, keyframe, data, .. })) => {
                // VP8 key frames have bit 0 of the first byte clear.
                seen.video.push((keyframe || data.first().is_some_and(|b| b & 1 == 0), data.len()))
            }
            Ok(Some(PeerEvent::Media { kind: TrackKind::Audio, .. })) => seen.audio += 1,
            Ok(Some(PeerEvent::Closed(_))) | Ok(None) => {
                seen.closed = true;
                return;
            }
            Ok(Some(_)) => {}
            Err(_) => return,
        }
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn lifecycle_routes_and_codes() {
    let e = engine(FakeModel::h3_turbo(), Duration::from_millis(400));
    let (rt, app) = runtime(e.clone(), |_| {}).await;

    // CREATED while loading: 503 + Retry-After: 1.
    let (_, _, d) = call(&app, "GET", "/session", None).await;
    assert_eq!(d["state"], "created");
    assert_eq!(d["session_id"], SESSION_ID);
    assert!(d.get("capabilities").is_none());
    assert_eq!(call(&app, "GET", "/schema", None).await.2, json!({}));
    let (s, h, v) = call(&app, "POST", "/start_session", Some(json!({}))).await;
    assert_eq!(s, StatusCode::SERVICE_UNAVAILABLE, "{v}");
    assert_eq!(h.get("retry-after").unwrap(), "1");
    assert!(v["detail"].as_str().unwrap().starts_with("cannot start session while created"));
    e.wait_ready().await;

    // READY: descriptor with capabilities (client perspective).
    let (_, _, d) = call(&app, "GET", "/session", None).await;
    assert_eq!(d["state"], "ready");
    assert_eq!(d["cluster"], "local");
    assert_eq!(d["selected_transport"], json!({"protocol": "webrtc", "version": "1.0"}));
    assert_eq!(
        d["capabilities"]["tracks"],
        json!([{"name":"main_video","kind":"video","direction":"recvonly"},{"name":"main_audio","kind":"audio","direction":"recvonly"}])
    );
    let (_, _, schema) = call(&app, "GET", "/schema", None).await;
    assert_eq!(
        schema["x-reactor"]["tracks"],
        json!([{"name":"main_video","kind":"video","direction":"out"},{"name":"main_audio","kind":"audio","direction":"out"}])
    );
    assert!(schema["paths"]["/events/enqueue"].is_object());

    // Signalling before a session: 400.
    let (s, _, v) = call(&app, "GET", &format!("{W}/ice_servers"), None).await;
    assert_eq!((s, v), (StatusCode::BAD_REQUEST, json!({"detail": "No session running"})));
    let (s, _, _) = call(&app, "POST", "/stop_session", None).await;
    assert_eq!(s, StatusCode::CONFLICT);

    // WAITING.
    let (s, _, d) = call(&app, "POST", "/start_session", Some(json!({"extra_args": {}}))).await;
    assert_eq!(s, StatusCode::OK, "{d}");
    assert_eq!(d["state"], "waiting");
    let (s, _, v) = call(&app, "POST", "/start_session", None).await;
    assert_eq!(s, StatusCode::CONFLICT, "{v}");
    assert_eq!(v["detail"], "cannot start session while waiting");
    let (s, _, v) = call(&app, "GET", "/sessions/other/transport/webrtc/ice_servers", None).await;
    assert_eq!((s, v), (StatusCode::NOT_FOUND, json!({"detail": "Unknown session"})));
    let (s, _, v) = call(&app, "GET", &format!("{W}/ice_servers"), None).await;
    assert_eq!(s, StatusCode::OK);
    assert!(v["ice_servers"].is_array());
    let (s, _, v) = call(&app, "POST", &format!("{W}/connections"), None).await;
    assert_eq!(s, StatusCode::CREATED);
    assert_eq!(
        v["track_map"],
        json!({"main_video": {"kind":"video","direction":"out","rate":0.0}, "main_audio": {"kind":"audio","direction":"out","rate":48000.0}})
    );
    let cid = v["connection_id"].as_u64().unwrap();
    let (s, _, _) = call(&app, "GET", &format!("{W}/connections/{cid}/sdp_params"), None).await;
    assert_eq!(s, StatusCode::ACCEPTED);
    let (s, _, _) = call(&app, "GET", &format!("{W}/connections/1/sdp_params"), None).await;
    assert_eq!(s, StatusCode::NOT_FOUND);
    let (s, _, _) = call(&app, "POST", &format!("{W}/connections/{cid}/sdp_params"), Some(json!({"track_mapping": []}))).await;
    assert_eq!(s, StatusCode::UNPROCESSABLE_ENTITY);
    let (s, _, _) = call(
        &app,
        "POST",
        &format!("{W}/connections/{cid}/sdp_params"),
        Some(json!({"sdp_offer": "v=0", "ice_credentials": {"ufrag": "a", "pwd": "b"}})),
    )
    .await;
    assert_eq!(s, StatusCode::UNPROCESSABLE_ENTITY);
    // A bad offer: 202, then the poll reports the failure.
    let (s, _, _) = call(&app, "POST", &format!("{W}/connections/{cid}/sdp_params"), Some(json!({"sdp_offer": "v=0\r\n"}))).await;
    assert_eq!(s, StatusCode::ACCEPTED);
    // 202 while the offer is processed, then the failure.
    let t0 = Instant::now();
    let (s, v) = loop {
        let (s, _, v) = call(&app, "GET", &format!("{W}/connections/{cid}/sdp_params"), None).await;
        if s != StatusCode::ACCEPTED || t0.elapsed() > T {
            break (s, v);
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    };
    assert_eq!(s, StatusCode::BAD_REQUEST, "{v}");

    // CLOSING → READY.
    let (s, _, _) = call(&app, "POST", "/stop_session", Some(json!({"reason": "x".repeat(65)}))).await;
    assert_eq!(s, StatusCode::UNPROCESSABLE_ENTITY);
    let (s, _, _) = call(&app, "POST", "/stop_session", None).await;
    assert_eq!(s, StatusCode::OK);
    assert_eq!(rt.state(), RtState::Ready);
    let (replay, _) = rt.journal().subscribe(0);
    let evs: Vec<&str> = replay.iter().map(|e| e.data["event"].as_str().unwrap()).collect();
    assert_eq!(evs, ["start_session", "stop_session", "cleanup_complete"]);
    assert_eq!(replay[0].data["to"], "waiting");
    // The engine slot is free again: a new session starts.
    let (s, _, _) = call(&app, "POST", "/start_session", None).await;
    assert_eq!(s, StatusCode::OK);
    rt.drain().await;
    assert_eq!(rt.state(), RtState::Ready);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn v1_client_av_session() {
    v1_av_session(false).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "real-time delivery: scripts/serve/check.sh --realtime"]
async fn realtime_v1_client_av_session() {
    v1_av_session(true).await;
}

/// `realtime`: also check that the clip arrives frame for frame and mostly
/// as inter frames (a loaded host sheds frames and forces keyframes).
async fn v1_av_session(realtime: bool) {
    let e = engine(FakeModel::h3_turbo(), Duration::ZERO);
    e.wait_ready().await;
    let (rt, app) = runtime(e, |_| {}).await;
    let (s, _, _) = call(&app, "POST", "/start_session", None).await;
    assert_eq!(s, StatusCode::OK);
    let Client { mut peer, cid, .. } = connect(&app, true).await;
    use pb::control_client_message::Payload as C;
    use pb::MessageKind as K;
    let h = peer.handle().clone();
    // First frame latches v1.
    h.send_message(control(C::Ping(pb::Ping {}), "", K::Notification)).await.unwrap();
    let mut seen = Seen::default();
    // The greeting (state_update + queue_update) arrives as v1.
    pump(&mut peer, &mut seen, T, |s| !s.model("queue_update").is_empty()).await;
    assert_eq!(seen.model("state_update").len(), 1, "{seen:?}");
    assert!(seen.text.is_empty(), "v0 text on a v1 connection: {:?}", seen.text);
    assert_eq!(rt.state(), RtState::Streaming);
    assert_eq!(rt.descriptor()["state"], "streaming");
    // Pause gate: nothing on the wire before ResumeTrack.
    pump(&mut peer, &mut seen, Duration::from_millis(400), |_| false).await;
    assert!(seen.video.is_empty() && seen.audio == 0, "media before ResumeTrack: {} video, {} audio", seen.video.len(), seen.audio);

    h.send_message(control(C::RequestSchema(pb::RequestSchema {}), "ctrl_1", K::Request)).await.unwrap();
    h.send_message(control(C::PublishTrack(pb::PublishTrack { name: "cam".into() }), "ctrl_2", K::Request)).await.unwrap();
    h.send_message(control(C::RequestClip(pb::RequestClip { duration_seconds: 2.0 }), "ctrl_3", K::Request)).await.unwrap();
    h.send_message(control(C::ResumeTrack(pb::ResumeTrack { name: "main_video".into() }), "", K::Notification)).await.unwrap();
    h.send_message(control(C::ResumeTrack(pb::ResumeTrack { name: "main_audio".into() }), "", K::Notification)).await.unwrap();
    // Video waits for the encoder to start: an ffmpeg process, seconds on a
    // loaded host (the shared build pod); the pump returns as soon as it
    // has everything.
    pump(&mut peer, &mut seen, T, |s| s.control.len() >= 3 && !s.video.is_empty() && s.audio > 5).await;
    use pb::control_server_message::Payload as CS;
    let by = |rid: &str| seen.control.iter().find(|m| m.request_id == rid).and_then(|m| m.payload.clone());
    let Some(CS::ModelSchema(ms)) = by("ctrl_1") else { panic!("{:?}", seen.control) };
    let doc = Value::Object(fastvideo_reactor::wire::struct_to_map(ms.openapi.unwrap()));
    assert_eq!(doc["x-reactor"]["tracks"][1]["name"], "main_audio");
    assert!(matches!(by("ctrl_2"), Some(CS::Error(e)) if e.code == "publish_refused"));
    assert!(matches!(by("ctrl_3"), Some(CS::ClipFailed(c)) if c.reason == "recording disabled"));
    // The start-of-connection black frame, as a key frame.
    assert!(seen.video[0].0, "first video frame is not a key frame");
    assert!(seen.audio > 0);

    // Commands: correlated replies, bodyless acks, contract errors.
    h.send_message(command("get_state", json!({}), "data_1")).await.unwrap();
    h.send_message(command("set_autoplay", json!({"enabled": true}), "data_2")).await.unwrap();
    h.send_message(command("set_seed", json!({"seed": -1}), "data_3")).await.unwrap();
    h.send_message(command("nope", json!({}), "data_4")).await.unwrap();
    h.send_message(command("play", json!({}), "data_5")).await.unwrap();
    pump(&mut peer, &mut seen, T, |s| (1..=5).all(|i| s.reply(&format!("data_{i}")).is_some())).await;
    let state = seen.model("state_update").into_iter().find(|(r, _)| r == "data_1").expect("get_state reply").1;
    assert_eq!(state["autoplay"], false);
    assert!(state["valid_commands"].as_array().unwrap().iter().any(|c| c == "enqueue"));
    assert_eq!(seen.reply("data_1").unwrap().kind, pb::MessageKind::Response as i32);
    assert_eq!(seen.model("autoplay_accepted")[0], ("data_2".to_owned(), json!({"enabled": true})));
    use pb::data_server_message::Payload as DS;
    assert!(matches!(&seen.reply("data_3").unwrap().payload, Some(DS::Error(e)) if e.code == "invalid_command"));
    assert!(matches!(&seen.reply("data_4").unwrap().payload, Some(DS::Error(e)) if e.code == "invalid_command"));
    // `play` with nothing queued: refused by broadcast, bodyless ack.
    assert_eq!(seen.reply("data_5").unwrap().payload, None);
    pump(&mut peer, &mut seen, T, |s| !s.model("command_error").is_empty()).await;
    assert_eq!(seen.model("command_error")[0].1["command"], "play");

    // A clip end to end: queued, generated, started, finished, then black.
    let v0 = seen.video.len();
    h.send_message(command("enqueue", json!({"prompt": "a red kite", "metadata": "k1"}), "data_6")).await.unwrap();
    pump(&mut peer, &mut seen, 2 * T, |s| !s.model("clip_finished").is_empty()).await;
    let (rid, q) = seen.model("clip_queued").remove(0);
    assert_eq!(rid, "data_6");
    assert_eq!(q["clip"]["metadata"], "k1");
    for t in ["clip_generated", "clip_started", "clip_finished"] {
        assert_eq!(seen.model(t).len(), 1, "{t}");
    }
    let frames = q["clip"]["frames"].as_u64().unwrap() as usize;
    // The peer starts on a key frame.
    assert!(seen.video[0].0, "first frame is not a key frame");
    if realtime {
        pump(&mut peer, &mut seen, Duration::from_millis(500), |_| false).await;
        let sent = seen.video.len() - v0;
        assert!(sent > frames, "{sent} video frames for a {frames}-frame clip (+ black)");
        // With ffmpeg libvpx the rest are mostly inter frames (a host that
        // sheds frames forces keyframes); the libwebp fallback sends only
        // key frames.
        let keys = seen.video.iter().filter(|(k, _)| *k).count();
        if fastvideo_media::vp8::libvpx_available() {
            assert!(keys * 4 < seen.video.len(), "{keys} key frames of {}", seen.video.len());
        } else {
            assert_eq!(keys, seen.video.len());
        }
    } else {
        // The clip's frames reach the peer (a loaded host may shed some).
        pump(&mut peer, &mut seen, T, |s| s.video.len() > v0).await;
        assert!(seen.video.len() > v0, "no video frames for a {frames}-frame clip");
        // The libwebp fallback sends only key frames, loaded or not.
        if !fastvideo_media::vp8::libvpx_available() {
            let keys = seen.video.iter().filter(|(k, _)| *k).count();
            assert_eq!(keys, seen.video.len());
        }
    }

    // Stop with a reason: session_ended, then the wire closes; READY.
    let (s, _, _) = call(&app, "POST", "/stop_session", Some(json!({"reason": "bye"}))).await;
    assert_eq!(s, StatusCode::OK);
    let ended = |s: &Seen| s.control.iter().any(|m| matches!(&m.payload, Some(CS::SessionEnded(e)) if e.reason == "bye"));
    pump(&mut peer, &mut seen, T, ended).await;
    assert!(ended(&seen), "{:?}", seen.control);
    assert!(seen.control.iter().any(|m| m.kind == pb::MessageKind::Notification as i32));
    assert_eq!(rt.state(), RtState::Ready);
    let _ = cid;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn v0_client_video_only() {
    let e = engine(FakeModel::wan(), Duration::ZERO);
    e.wait_ready().await;
    let (rt, app) = runtime(e, |_| {}).await;
    let (_, _, d) = call(&app, "POST", "/start_session", None).await;
    assert_eq!(d["capabilities"]["tracks"], json!([{"name":"main_video","kind":"video","direction":"recvonly"}]));
    let (_, _, v) = call(&app, "POST", &format!("{W}/connections"), None).await;
    assert_eq!(v["track_map"], json!({"main_video": {"kind":"video","direction":"out","rate":0.0}}));
    // The client offers audio anyway: the answer leaves it inactive.
    let Client { mut peer, .. } = connect(&app, false).await;
    assert!(!peer.handle().sends(TrackKind::Audio));
    let h = peer.handle().clone();
    h.send_message(ChannelMessage::text("data", r#"{"scope":"runtime","data":{"type":"ping","data":{}}}"#)).await.unwrap();
    h.send_message(ChannelMessage::text("data", r#"{"scope":"runtime","data":{"type":"requestSchema","data":{}}}"#)).await.unwrap();
    h.send_message(ChannelMessage::text("data", r#"{"scope":"application","data":{"type":"get_state","data":{}}}"#)).await.unwrap();
    h.send_message(ChannelMessage::text("data", r#"{"scope":"application","data":{"type":"set_seed","data":{"seed":"x"}}}"#)).await.unwrap();
    h.send_message(ChannelMessage::text("control", r#"{"type":"request","method":"publish_track","request_id":"r1","data":{"name":"cam"}}"#)).await.unwrap();
    h.send_message(ChannelMessage::text("control", r#"{"type":"notification","event":"resume_track","data":{"name":"main_video"}}"#)).await.unwrap();
    let mut seen = Seen::default();
    // As in `v1_client_av_session`: video waits for the encoder to start.
    pump(&mut peer, &mut seen, T, |s| {
        s.text.iter().any(|(_, v)| v["data"]["type"] == "modelSchema")
            && s.text.iter().filter(|(_, v)| v["data"]["type"] == "state_update").count() >= 2
            && s.text.iter().any(|(l, _)| l == "control")
            && !s.video.is_empty()
    })
    .await;
    assert!(seen.data.is_empty() && seen.control.is_empty(), "binary frames on a v0 connection");
    let schema = seen.text.iter().find(|(_, v)| v["data"]["type"] == "modelSchema").unwrap();
    assert_eq!(schema.0, "data");
    assert_eq!(schema.1["scope"], "runtime");
    assert_eq!(schema.1["data"]["data"]["x-reactor"]["tracks"], json!([{"name":"main_video","kind":"video","direction":"out"}]));
    let su = seen.text.iter().find(|(_, v)| v["data"]["type"] == "state_update").unwrap();
    assert_eq!(su.1["scope"], "application");
    let pubr = seen.text.iter().find(|(l, _)| l == "control").unwrap();
    assert_eq!(pubr.1["type"], "response");
    assert_eq!(pubr.1["error"]["code"], "publish_refused");
    assert!(!seen.video.is_empty());
    assert_eq!(seen.audio, 0);
    rt.drain().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn watchdog_and_orphan_timeout() {
    let e = engine(FakeModel::wan(), Duration::ZERO);
    e.wait_ready().await;
    let (rt, app) = runtime(e, |c| {
        c.ping_timeout = Duration::from_millis(900);
        c.watchdog_interval = Duration::from_millis(100);
        c.orphan_timeout = Duration::from_millis(1200);
    })
    .await;
    call(&app, "POST", "/start_session", None).await;
    let Client { mut peer, .. } = connect(&app, false).await;
    let t0 = Instant::now();
    while rt.state() != RtState::Streaming && t0.elapsed() < Duration::from_secs(3) {
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    assert_eq!(rt.state(), RtState::Streaming);
    // Pings keep it alive.
    let h = peer.handle().clone();
    let mut seen = Seen::default();
    for _ in 0..6 {
        h.send_message(control(pb::control_client_message::Payload::Ping(pb::Ping {}), "", pb::MessageKind::Notification))
            .await
            .unwrap();
        pump(&mut peer, &mut seen, Duration::from_millis(300), |_| false).await;
    }
    assert!(!seen.closed);
    assert_eq!(rt.state(), RtState::Streaming);
    // Silence: the watchdog drops the connection, the session is ORPHANED.
    let t0 = Instant::now();
    while rt.state() == RtState::Streaming && t0.elapsed() < Duration::from_secs(3) {
        pump(&mut peer, &mut seen, Duration::from_millis(50), |_| false).await;
    }
    assert_eq!(rt.state(), RtState::Orphaned, "watchdog did not fire");
    assert!(t0.elapsed() >= Duration::from_millis(500), "{:?}", t0.elapsed());
    // ...and the orphan timeout closes it.
    tokio::time::sleep(Duration::from_millis(1600)).await;
    assert_eq!(rt.state(), RtState::Ready);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn causal_session_setters() {
    let e = engine(FakeModel::sf_wan(), Duration::ZERO);
    e.wait_ready().await;
    let (rt, app) = runtime(e, |_| {}).await;
    let (s, _, d) = call(&app, "POST", "/start_session", None).await;
    assert_eq!(s, StatusCode::OK, "{d}");
    let (_, _, schema) = call(&app, "GET", "/schema", None).await;
    assert!(schema["paths"]["/events/set_prompt"].is_object());
    assert!(schema["paths"].get("/events/enqueue").is_none());
    let Client { mut peer, .. } = connect(&app, false).await;
    let h = peer.handle().clone();
    h.send_message(control(
        pb::control_client_message::Payload::ResumeTrack(pb::ResumeTrack { name: "main_video".into() }),
        "",
        pb::MessageKind::Notification,
    ))
    .await
    .unwrap();
    h.send_message(command("set_prompt", json!({"prompt": "a road"}), "data_1")).await.unwrap();
    h.send_message(command("get_state", json!({}), "data_2")).await.unwrap();
    let mut seen = Seen::default();
    pump(&mut peer, &mut seen, T, |s| s.video.len() > 20 && s.reply("data_2").is_some()).await;
    assert_eq!(seen.reply("data_1").unwrap().payload, None, "setter → bodyless ack");
    let st = seen.model("state_update").into_iter().find(|(r, _)| r == "data_2").unwrap().1;
    assert_eq!(st["prompt"], "a road");
    assert!(seen.video.len() > 20, "{} frames", seen.video.len());
    rt.drain().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn causal_session_ends_at_its_length_limit() {
    use pb::control_server_message::Payload as CS;
    let e = engine(FakeModel::sf_wan(), Duration::ZERO);
    e.wait_ready().await;
    // Small limits for real time: 3 s by default, 4 s at most. Longer than
    // a video encoder takes to start (an ffmpeg process: 1-2 s on a loaded
    // host such as the shared build pod), so video reaches the peer before
    // the session ends and its connections close.
    let (rt, app) = runtime(e, |c| c.causal_limits = CausalLimits { default_max_s: 3, hard_max_s: 4 }).await;
    let (_, _, schema) = call(&app, "GET", "/schema", None).await;
    assert_eq!(schema["x-reactor"]["session_limits"]["default_max_s"], 3, "{schema}");
    assert_eq!(schema["x-reactor"]["session_limits"]["hard_max_s"], 4);
    let (s, _, _) = call(&app, "POST", "/start_session", Some(json!({"max_seconds": 0}))).await;
    assert_eq!(s, StatusCode::BAD_REQUEST);
    // The default, then a longer request clamped to the ceiling.
    for (params, limit) in [(json!({}), 3u32), (json!({"max_seconds": 600}), 4)] {
        let (s, _, d) = call(&app, "POST", "/start_session", Some(params)).await;
        assert_eq!(s, StatusCode::OK, "{d}");
        let Client { mut peer, .. } = connect(&app, false).await;
        let h = peer.handle().clone();
        h.send_message(control(
            pb::control_client_message::Payload::ResumeTrack(pb::ResumeTrack { name: "main_video".into() }),
            "",
            pb::MessageKind::Notification,
        ))
        .await
        .unwrap();
        h.send_message(command("set_prompt", json!({"prompt": "a road"}), "data_1")).await.unwrap();
        let want = session_limit_reason(limit);
        let ended =
            |s: &Seen| s.control.iter().any(|m| matches!(&m.payload, Some(CS::SessionEnded(e)) if e.reason == want));
        let t0 = Instant::now();
        let mut seen = Seen::default();
        pump(&mut peer, &mut seen, T, |s| ended(s) && !s.video.is_empty()).await;
        assert!(ended(&seen), "{:?}", seen.control);
        assert!(!seen.video.is_empty(), "no video received");
        // Not before the limit: the clock starts at the first frame.
        assert!(t0.elapsed() >= Duration::from_secs(u64::from(limit)), "{:?}", t0.elapsed());
        let t1 = Instant::now();
        while rt.state() != RtState::Ready && t1.elapsed() < T {
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        assert_eq!(rt.state(), RtState::Ready);
    }
}

/// `PUT` of raw bytes.
async fn put_bytes(app: &Router, uri: &str, bytes: Vec<u8>) -> StatusCode {
    let req = Request::builder().method("PUT").uri(uri).header("content-type", "application/octet-stream").body(Body::from(bytes)).unwrap();
    app.clone().oneshot(req).await.unwrap().status()
}

/// A command whose file parameter rides `Command.uploads` (the SDK's way).
fn command_with_upload(name: &str, param: &str, upload_id: &str, rid: &str) -> ChannelMessage {
    let m = pb::DataClientMessage {
        request_id: rid.into(),
        kind: pb::MessageKind::Request as i32,
        payload: Some(pb::data_client_message::Payload::Command(pb::Command {
            r#type: name.into(),
            data: Some(json_to_struct(&json!({}))),
            uploads: [(param.to_owned(), pb::UploadReference { upload_id: upload_id.into(), name: "face.png".into(), mime_type: "image/png".into(), size: 0 })].into(),
        })),
    };
    ChannelMessage::binary("data", m.encode_to_vec())
}

fn png(w: u32, h: u32) -> Vec<u8> {
    let img = image::RgbImage::from_fn(w, h, |x, y| image::Rgb([(x * 2) as u8, (y * 2) as u8, 128]));
    let mut out = std::io::Cursor::new(Vec::new());
    img.write_to(&mut out, image::ImageFormat::Png).unwrap();
    out.into_inner()
}

/// Avatar mode (Reactor `ltx`) on the fake LTX engine: upload the photo,
/// set the script, start; the take streams window by window (A/V), pauses
/// and resumes, completes, and reset clears the conditions.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn avatar_script_take_streams_in_windows() {
    avatar_take(false).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "real-time delivery: scripts/serve/check.sh --realtime"]
async fn realtime_avatar_script_take_streams_in_windows() {
    avatar_take(true).await;
}

/// `realtime`: also check the take's video frame floor.
async fn avatar_take(realtime: bool) {
    use fastvideo_reactor::engine::Mode;
    use fastvideo_reactor::AvatarSettings;
    use pb::control_client_message::Payload as C;
    use pb::MessageKind as K;
    let e = engine(FakeModel::ltx_turbo(), Duration::ZERO);
    e.wait_ready().await;
    let (rt, app) = runtime(e, |c| {
        c.mode = Some(Mode::Avatar);
        c.avatar = AvatarSettings { size: (128, 64), speed: 3.0, session_max_s: 120, ..AvatarSettings::default() };
    })
    .await;
    let (_, _, schema) = call(&app, "GET", "/schema", None).await;
    assert_eq!(schema["x-reactor"]["mode"], "avatar", "{schema}");
    assert_eq!(schema["x-reactor"]["session_limits"]["take_max_s"], 300.0);
    assert!(schema["paths"]["/events/set_avatar_image"].is_object());
    assert_eq!(
        schema["paths"]["/events/set_avatar_image"]["post"]["requestBody"]["content"]["application/json"]["schema"]["properties"]["avatar_image"]["$ref"],
        "#/components/schemas/ReactorUploadReference"
    );
    let (s, _, d) = call(&app, "POST", "/start_session", Some(json!({"max_seconds": 121}))).await;
    assert_eq!(s, StatusCode::BAD_REQUEST, "{d}");
    let (s, _, d) = call(&app, "POST", "/start_session", None).await;
    assert_eq!(s, StatusCode::OK, "{d}");
    assert_eq!(d["capabilities"]["tracks"][1]["name"], "main_audio");

    // The upload: register, then PUT the bytes to the presigned URL.
    let bytes = png(96, 96);
    let (s, _, up) = call(
        &app,
        "POST",
        "/sessions/00000000-0000-0000-0000-000000000000/uploads",
        Some(json!({"name": "face.png", "size": bytes.len(), "mime_type": "image/png"})),
    )
    .await;
    assert_eq!(s, StatusCode::CREATED, "{up}");
    let id = up["presigned_id"].as_str().unwrap().to_owned();
    assert_eq!(put_bytes(&app, up["path"].as_str().unwrap(), bytes[..10].to_vec()).await, StatusCode::BAD_REQUEST);
    assert_eq!(put_bytes(&app, up["path"].as_str().unwrap(), bytes.clone()).await, StatusCode::OK);

    let Client { mut peer, .. } = connect(&app, true).await;
    let h = peer.handle().clone();
    h.send_message(control(C::Ping(pb::Ping {}), "", K::Notification)).await.unwrap();
    for t in ["main_video", "main_audio"] {
        h.send_message(control(C::ResumeTrack(pb::ResumeTrack { name: t.into() }), "", K::Notification)).await.unwrap();
    }
    let mut seen = Seen::default();
    // Greeting, then `start` without an image: refused.
    h.send_message(command("start", json!({}), "d1")).await.unwrap();
    pump(&mut peer, &mut seen, T, |s| !s.model("command_error").is_empty()).await;
    assert!(seen.model("state_update")[0].1["has_avatar_image"] == false);
    assert_eq!(seen.model("command_error")[0].1["reason"], "set_avatar_image first");

    h.send_message(command_with_upload("set_avatar_image", "avatar_image", &id, "d2")).await.unwrap();
    h.send_message(command_with_upload("set_avatar_image", "avatar_image", "never-uploaded-x", "d2b")).await.unwrap();
    let script = "Hello and welcome. This is a short script for the avatar test, and it is long enough to need two windows \
                  of speech at the default rate. Thanks for watching, see you soon.";
    h.send_message(command("set_script", json!({"script": script}), "d3")).await.unwrap();
    h.send_message(command("set_wpm", json!({"wpm": 300}), "d4")).await.unwrap();
    h.send_message(command("set_seed", json!({"seed": 11}), "d5")).await.unwrap();
    pump(&mut peer, &mut seen, T, |s| s.reply("d5").is_some() && s.reply("d2b").is_some()).await;
    let acc = seen.model("avatar_image_accepted");
    assert_eq!(acc[0].1, json!({"name": "face.png", "width": 96, "height": 96}), "{acc:?}");
    use pb::data_server_message::Payload as DS;
    assert!(matches!(&seen.reply("d2b").unwrap().payload, Some(DS::Error(e)) if e.code == "unresolved_upload"));
    let sa = &seen.model("script_accepted")[0].1;
    let words = sa["words"].as_u64().unwrap();
    assert_eq!(words, script.split_whitespace().count() as u64);
    assert!(matches!(&seen.reply("d4").unwrap().payload, Some(DS::Error(e)) if e.code == "invalid_command"));
    assert_eq!(seen.model("seed_accepted")[0].1["seed"], 11);

    let v0 = seen.video.len();
    let a0 = seen.audio;
    h.send_message(command("start", json!({}), "d6")).await.unwrap();
    pump(&mut peer, &mut seen, T, |s| !s.model("window_started").is_empty()).await;
    let started = seen.model("generation_started");
    assert_eq!(started.len(), 1, "{:?}", seen.model("command_error"));
    let total = started[0].1["total_windows"].as_u64().unwrap();
    assert!(total >= 2, "{started:?}");
    assert_eq!((started[0].1["width"].as_u64(), started[0].1["height"].as_u64()), (Some(128), Some(64)));
    // Pause and resume mid-take.
    h.send_message(command("pause", json!({}), "d7")).await.unwrap();
    pump(&mut peer, &mut seen, T, |s| !s.model("generation_paused").is_empty()).await;
    h.send_message(command("pause", json!({}), "d7b")).await.unwrap();
    h.send_message(command("resume", json!({}), "d8")).await.unwrap();
    pump(&mut peer, &mut seen, 2 * T, |s| !s.model("generation_complete").is_empty()).await;
    assert_eq!(seen.model("generation_resumed").len(), 1);
    assert!(seen.model("command_error").iter().any(|(_, e)| e["command"] == "pause"), "second pause refused");
    let progress = seen.model("window_progress");
    assert_eq!(progress.len() as u64, total, "{progress:?}");
    for (i, (_, p)) in progress.iter().enumerate() {
        assert_eq!(p["window_index"].as_u64(), Some(i as u64));
    }
    let built = seen.model("window_built");
    assert_eq!(built.len() as u64, total);
    assert!(built.iter().all(|(_, b)| b["rtf"].as_f64().is_some_and(|r| r > 0.0)));
    let done = &seen.model("generation_complete")[0].1;
    let secs = done["seconds_sent"].as_f64().unwrap();
    let effective = sa["effective_seconds"].as_f64().unwrap();
    assert!((secs - effective).abs() < 0.1, "sent {secs} of {effective}");
    if realtime {
        pump(&mut peer, &mut seen, Duration::from_millis(500), |_| false).await;
        let frames = seen.video.len() - v0;
        // Played at 3x real time: the encoder's drop-oldest input queue may
        // shed frames, so only a floor is checked.
        assert!(frames as f64 >= effective * 24.0 * 0.4, "{frames} frames for {effective} s");
    } else {
        // The take's video reaches the peer (a loaded host sheds frames).
        pump(&mut peer, &mut seen, T, |s| s.video.len() > v0).await;
        assert!(seen.video.len() > v0, "no video frames for the take");
    }
    assert!(seen.audio - a0 > 50, "{} audio packets", seen.audio - a0);
    let st = seen.model("state_update").last().unwrap().1.clone();
    assert_eq!(st["finished"], true);
    assert!(st["valid_commands"].as_array().unwrap().iter().any(|c| c == "start"), "{st}");

    // A queued change while generating is listed; reset clears everything.
    h.send_message(command("reset", json!({}), "d9")).await.unwrap();
    pump(&mut peer, &mut seen, T, |s| !s.model("generation_reset").is_empty()).await;
    h.send_message(command("get_state", json!({}), "d10")).await.unwrap();
    pump(&mut peer, &mut seen, T, |s| s.reply("d10").is_some()).await;
    let st = seen.model("state_update").into_iter().find(|(r, _)| r == "d10").unwrap().1;
    assert_eq!(st["has_avatar_image"], false);
    assert_eq!(st["script"], "");
    rt.drain().await;
}
