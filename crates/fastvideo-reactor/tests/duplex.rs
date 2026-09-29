//! Reactor duplex mode on the loopback echo (design §5.11): input tracks in
//! the descriptor, `track_map` and `/schema`; `PublishTrack` /
//! `UnpublishTrack` slots (first come, first served, unknown names
//! refused); the ingest auth; a loopback str0m client publishing a VP8
//! camera and an Opus microphone and receiving the echo — decoded here, the
//! picture carries the camera's colour inside the magenta overlay border —
//! and the microphone back on `main_audio`; `get_state` with the ingest
//! counters. Skips (passes with a note) without ffmpeg's libvpx.

use std::sync::Arc;
use std::time::{Duration, Instant};

use axum::body::Body;
use axum::http::{Request, StatusCode};
use axum::Router;
use fastvideo_engine_service::{EchoBackend, EngineConfig, EngineService, ECHO_BORDER};
use fastvideo_media::decode::{VideoDecoder, VideoDecoderConfig};
use fastvideo_media::vp8::{libvpx_available, Vp8Config, Vp8Encoder};
use fastvideo_protocol::{InputVideoCodec, RgbFrame};
use fastvideo_reactor::wire::{json_to_struct, struct_to_map};
use fastvideo_reactor::{pb, router, H264Backend, IngestAuth, Reactor, ReactorConfig};
use fastvideo_webrtc::channel::ChannelMessage;
use fastvideo_webrtc::host::{AudioLayout, HostConfig, OfferOptions, Peer, PeerEvent, RtcHost};
use fastvideo_webrtc::sdp::{Direction, MediaKind, Sdp};
use fastvideo_webrtc::writer::{AudioPacket, TrackKind, VideoCodec, VideoFrame};
use prost::Message as _;
use serde_json::{json, Value};
use tower::ServiceExt;

const W: &str = "/sessions/00000000-0000-0000-0000-000000000000/transport/webrtc";
const KEY: &str = "Bearer duplex-test-key";

async fn runtime() -> (Reactor, Router) {
    let e = EngineService::start(
        EngineConfig { output_dir: std::env::temp_dir().join("fv-reactor-duplex"), ..EngineConfig::default() },
        vec![Box::new(EchoBackend)],
    )
    .unwrap();
    e.wait_ready().await;
    let host = RtcHost::bind(HostConfig::loopback(true, false)).await.unwrap();
    let auth = IngestAuth(Arc::new(|h: &axum::http::HeaderMap| {
        match h.get("authorization").and_then(|v| v.to_str().ok()) {
            Some(KEY) => Ok(()),
            _ => Err("an API key is required for duplex sessions".into()),
        }
    }));
    let cfg = ReactorConfig {
        model: Some("fv-echo".into()),
        short_edge: Some(180),
        h264: H264Backend::Off,
        latch_grace: Duration::from_millis(300),
        ingest_auth: Some(auth),
        ..ReactorConfig::default()
    };
    let rt = Reactor::new(cfg, Arc::new(e), host);
    let app = router(rt.clone());
    (rt, app)
}

async fn call(app: &Router, method: &str, uri: &str, body: Option<Value>, auth: bool) -> (StatusCode, Value) {
    let mut b = Request::builder().method(method).uri(uri).header("reactor-webrtc-version", "1.0");
    if auth {
        b = b.header("authorization", KEY);
    }
    let req = match body {
        Some(v) => b.header("content-type", "application/json").body(Body::from(v.to_string())).unwrap(),
        None => b.body(Body::empty()).unwrap(),
    };
    let resp = app.clone().oneshot(req).await.unwrap();
    let status = resp.status();
    let bytes = axum::body::to_bytes(resp.into_body(), 1 << 20).await.unwrap();
    (status, serde_json::from_slice(&bytes).unwrap_or(Value::Null))
}

struct Client {
    peer: Peer,
    #[allow(dead_code)]
    host: RtcHost,
    cid: u64,
}

/// A client with one send-receive VP8 video m-line (camera out, echo in) and
/// one send-receive mono Opus m-line, both channels open.
async fn connect(app: &Router) -> Client {
    let (s, v) = call(app, "POST", &format!("{W}/connections"), None, false).await;
    assert_eq!(s, StatusCode::UNAUTHORIZED, "duplex signalling needs the key: {v}");
    let (s, v) = call(app, "POST", &format!("{W}/connections"), None, true).await;
    assert_eq!(s, StatusCode::CREATED, "{v}");
    assert_eq!(v["track_map"]["input_video"], json!({"kind": "video", "direction": "in", "rate": 30.0}));
    assert_eq!(v["track_map"]["input_audio"], json!({"kind": "audio", "direction": "in", "rate": 48000.0}));
    assert_eq!(v["track_map"]["main_video"]["direction"], "out");
    let cid = v["connection_id"].as_u64().unwrap();
    let host = RtcHost::bind(HostConfig::loopback(true, false)).await.unwrap();
    let (pending, offer) = host
        .offer(OfferOptions {
            video: Some(Direction::SendRecv),
            audio: Some((Direction::SendRecv, AudioLayout::Mono)),
            channels: vec!["data".into(), "control".into()],
            video_codecs: vec![VideoCodec::Vp8],
            ..OfferOptions::default()
        })
        .await
        .unwrap();
    let sdp = Sdp::parse(&offer).unwrap();
    let mid = |k: MediaKind| sdp.media.iter().find(|m| m.kind() == k).and_then(|m| m.mid()).unwrap().to_owned();
    let (vm, am) = (mid(MediaKind::Video), mid(MediaKind::Audio));
    let mapping = json!([
        {"mid": vm, "name": "main_video", "kind": "video", "direction": "recvonly"},
        {"mid": vm, "name": "input_video", "kind": "video", "direction": "sendonly"},
        {"mid": am, "name": "main_audio", "kind": "audio", "direction": "recvonly"},
        {"mid": am, "name": "input_audio", "kind": "audio", "direction": "sendonly"},
    ]);
    let (s, v) =
        call(app, "POST", &format!("{W}/connections/{cid}/sdp_params"), Some(json!({"sdp_offer": offer, "track_mapping": mapping})), true).await;
    assert_eq!(s, StatusCode::ACCEPTED, "{v}");
    let t0 = Instant::now();
    let answer = loop {
        let (s, v) = call(app, "GET", &format!("{W}/connections/{cid}/sdp_params"), None, true).await;
        if s == StatusCode::OK {
            break v["sdp_answer"].as_str().unwrap().to_owned();
        }
        assert_eq!(s, StatusCode::ACCEPTED, "{v}");
        assert!(t0.elapsed() < Duration::from_secs(10));
        tokio::time::sleep(Duration::from_millis(20)).await;
    };
    // We receive on both m-lines, capped at the model's bitrate.
    assert!(answer.contains("b=AS:4000"), "{answer}");
    assert!(answer.contains("VP8/90000"), "{answer}");
    let mut peer = pending.accept_answer(&answer).await.unwrap();
    assert_eq!(peer.video_codec(), Some(VideoCodec::Vp8));
    let mut open = std::collections::HashSet::new();
    let mut connected = false;
    while !(connected && open.len() == 2) {
        match tokio::time::timeout(Duration::from_secs(10), peer.next_event()).await.expect("connect timeout") {
            Some(PeerEvent::Connected) => connected = true,
            Some(PeerEvent::ChannelOpen { label }) => {
                open.insert(label);
            }
            Some(PeerEvent::Closed(r)) => panic!("closed while connecting: {r:?}"),
            Some(_) => {}
            None => panic!("peer gone"),
        }
    }
    Client { peer, host, cid }
}

fn control(p: pb::control_client_message::Payload, rid: &str) -> ChannelMessage {
    let kind = if rid.is_empty() { pb::MessageKind::Notification } else { pb::MessageKind::Request };
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

#[derive(Default)]
struct Seen {
    control: Vec<pb::ControlServerMessage>,
    data: Vec<pb::DataServerMessage>,
    video: Vec<Vec<u8>>,
    audio: usize,
}

impl Seen {
    fn publish_reply(&self, rid: &str) -> Option<Result<(), String>> {
        self.control.iter().find(|m| m.request_id == rid).map(|m| match &m.payload {
            Some(pb::control_server_message::Payload::PublishTrack(_)) => Ok(()),
            Some(pb::control_server_message::Payload::Error(e)) => Err(format!("{}: {}", e.code, e.message)),
            other => Err(format!("{other:?}")),
        })
    }
    fn reply(&self, rid: &str) -> Option<(String, Value)> {
        self.data.iter().find(|m| m.request_id == rid).and_then(|m| match &m.payload {
            Some(pb::data_server_message::Payload::Message(mm)) => {
                Some((mm.r#type.clone(), Value::Object(struct_to_map(mm.data.clone().unwrap_or_default()))))
            }
            _ => None,
        })
    }
}

async fn pump(peer: &mut Peer, seen: &mut Seen, d: Duration, until: impl Fn(&Seen) -> bool) {
    let end = Instant::now() + d;
    while Instant::now() < end && !until(seen) {
        let left = end.saturating_duration_since(Instant::now());
        match tokio::time::timeout(left, peer.next_event()).await {
            Ok(Some(PeerEvent::Message(m))) if m.binary && m.label == "control" => {
                seen.control.push(pb::ControlServerMessage::decode(&m.data[..]).unwrap())
            }
            Ok(Some(PeerEvent::Message(m))) if m.binary => seen.data.push(pb::DataServerMessage::decode(&m.data[..]).unwrap()),
            Ok(Some(PeerEvent::Media { kind: TrackKind::Video, data, .. })) => seen.video.push(data.to_vec()),
            Ok(Some(PeerEvent::Media { kind: TrackKind::Audio, .. })) => seen.audio += 1,
            Ok(Some(PeerEvent::Closed(r))) => panic!("closed: {r:?}"),
            Ok(Some(_)) => {}
            Ok(None) | Err(_) => return,
        }
    }
}

fn publish(name: &str, rid: &str) -> ChannelMessage {
    control(pb::control_client_message::Payload::PublishTrack(pb::PublishTrack { name: name.into() }), rid)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn tracks_publish_slots_and_the_echo_round_trip() {
    if !libvpx_available() {
        eprintln!("skipping: ffmpeg has no libvpx");
        return;
    }
    let (rt, app) = runtime().await;
    // READY: the input tracks are declared in all three places.
    let (_, d) = call(&app, "GET", "/session", None, false).await;
    assert_eq!(
        d["capabilities"]["tracks"],
        json!([
            {"name":"main_video","kind":"video","direction":"recvonly"},
            {"name":"main_audio","kind":"audio","direction":"recvonly"},
            {"name":"input_video","kind":"video","direction":"sendonly"},
            {"name":"input_audio","kind":"audio","direction":"sendonly"}
        ])
    );
    let (_, schema) = call(&app, "GET", "/schema", None, false).await;
    assert_eq!(schema["x-reactor"]["mode"], "duplex");
    assert_eq!(schema["x-reactor"]["tracks"][2], json!({"name":"input_video","kind":"video","direction":"in"}));
    assert_eq!(schema["x-reactor"]["input"]["video"]["max_width"], 1280);
    assert_eq!(schema["x-reactor"]["session_limits"]["hard_max_s"], 300);
    assert!(schema["paths"]["/events/set_paused"].is_object());
    assert!(schema["webhooks"]["input_rejected"].is_object());

    // Duplex sessions need the key; the context is checked.
    let (s, _) = call(&app, "POST", "/start_session", Some(json!({})), false).await;
    assert_eq!(s, StatusCode::UNAUTHORIZED);
    let (s, v) = call(&app, "POST", "/start_session", Some(json!({"context": {"mood": "x"}})), true).await;
    assert_eq!(s, StatusCode::BAD_REQUEST, "{v}");
    let (s, d) = call(&app, "POST", "/start_session", Some(json!({"context": {"scene": "a studio"}, "max_seconds": 60})), true).await;
    assert_eq!(s, StatusCode::OK, "{d}");

    let mut a = connect(&app).await;
    let mut b = connect(&app).await;
    // Watch the echo on main_video / main_audio.
    for n in ["main_video", "main_audio"] {
        a.peer.send_message(control(pb::control_client_message::Payload::ResumeTrack(pb::ResumeTrack { name: n.into() }), "")).await.unwrap();
    }
    // Publish: A gets the camera slot, B is refused, unknown names are refused.
    a.peer.send_message(publish("input_video", "p1")).await.unwrap();
    a.peer.send_message(publish("input_audio", "p2")).await.unwrap();
    a.peer.send_message(publish("webcam", "p3")).await.unwrap();
    let mut sa = Seen::default();
    let mut sb = Seen::default();
    pump(&mut a.peer, &mut sa, Duration::from_secs(5), |s| s.publish_reply("p3").is_some()).await;
    b.peer.send_message(publish("input_video", "p4")).await.unwrap();
    pump(&mut b.peer, &mut sb, Duration::from_secs(5), |s| s.publish_reply("p4").is_some()).await;
    assert_eq!(sa.publish_reply("p1"), Some(Ok(())));
    assert_eq!(sa.publish_reply("p2"), Some(Ok(())));
    assert!(sa.publish_reply("p3").unwrap().unwrap_err().contains("publish_refused: the model declares no input track `webcam`"));
    assert_eq!(sb.publish_reply("p4"), Some(Err("publish_refused: track already published".into())));

    // A's camera: 3 s of blue VP8 at 30 fps, and its microphone (encoded
    // up front: ffmpeg's start must not eat into the real-time loop).
    let mut enc = Vp8Encoder::new(Vp8Config::new(320, 240, 30)).unwrap();
    let mut frames = Vec::new();
    for i in 0..90u64 {
        frames.extend(enc.encode(&RgbFrame::solid(320, 240, [20, 40, 230], i)).unwrap());
    }
    frames.extend(enc.finish().unwrap());
    let mut opus = fastvideo_media::opus::OpusEncoder::new(
        fastvideo_media::opus::OpusConfig { channels: 1, ..fastvideo_media::opus::OpusConfig::whip() },
        0,
    )
    .unwrap();
    let tone: Vec<f32> = (0..90 * 1600).map(|i| ((i as f32) * 0.06).sin() * 0.3).collect();
    let packets = opus.push(&tone).unwrap();
    // Real time, looped (the stream restarts at the keyframe) until the
    // echo has shown enough input: on a loaded host the first frames can
    // wait for the (pre-started) decoder.
    let t0 = Instant::now();
    let mut k = 0u64;
    let mut state = Value::Null;
    for cycle in 0..8 {
        for (i, f) in frames.iter().enumerate() {
            a.peer.send_video(VideoFrame::new(f.data.clone(), k * 3000)).unwrap();
            for p in &packets[(i * 5 / 3).min(packets.len())..((i + 1) * 5 / 3).min(packets.len())] {
                let rtp = u64::from(p.rtp_ts) + cycle * 90 * 1600;
                a.peer.send_audio(AudioPacket::new(p.data.clone(), rtp)).unwrap();
            }
            k += 1;
            let next = t0 + Duration::from_millis(k * 33);
            pump(&mut a.peer, &mut sa, next.saturating_duration_since(Instant::now()), |_| false).await;
        }
        let rid = format!("g{cycle}");
        a.peer.send_message(command("get_state", json!({}), &rid)).await.unwrap();
        pump(&mut a.peer, &mut sa, Duration::from_secs(5), |s| s.reply(&rid).is_some()).await;
        state = sa.reply(&rid).unwrap().1;
        if state["input_frames"].as_f64().unwrap_or(0.0) >= 30.0 {
            break;
        }
    }
    pump(&mut a.peer, &mut sa, Duration::from_millis(500), |_| false).await;
    eprintln!("sent {k} frames; echo state {state}");
    assert!(state["input_frames"].as_f64().unwrap() >= 30.0, "{state}");
    // The camera's colour reached the model.
    let c = &state["last_input_centre"];
    assert!((c[2].as_f64().unwrap() - 230.0).abs() < 40.0 && c[0].as_f64().unwrap() < 60.0, "{state}");
    assert!(sa.video.len() > 30, "echo frames received: {}", sa.video.len());
    assert!(sa.audio > 20, "echo audio packets received: {}", sa.audio);

    // Decode what came back: the camera's blue inside the magenta border.
    let mut dec = VideoDecoder::new(VideoDecoderConfig { codec: InputVideoCodec::Vp8, width: 320, height: 180 }).unwrap();
    let start = sa.video.iter().position(|f| f.first().is_some_and(|b| b & 1 == 0)).expect("a keyframe");
    let mut pics = Vec::new();
    for (i, f) in sa.video[start..].iter().enumerate() {
        dec.push(f, i as u64).unwrap();
        pics.extend(dec.poll().unwrap());
    }
    pics.extend(dec.finish().unwrap());
    let near = |p: Option<[u8; 3]>, want: [u8; 3]| {
        p.is_some_and(|p| p.iter().zip(want).all(|(a, b)| (i32::from(*a) - i32::from(b)).abs() < 48))
    };
    let mut runs: Vec<(Option<[u8; 3]>, usize)> = Vec::new();
    for p in &pics {
        let c = p.frame.pixel(160, 120).map(|c| c.map(|x| x / 32 * 32));
        match runs.last_mut() {
            Some((k, n)) if *k == c => *n += 1,
            _ => runs.push((c, 1)),
        }
    }
    let keys: Vec<usize> = sa.video.iter().enumerate().filter(|(_, f)| f.first().is_some_and(|b| b & 1 == 0)).map(|(i, _)| i).collect();
    eprintln!("received {} frames, keyframes at {keys:?}; centre runs {runs:?}", sa.video.len());
    let echoed = pics.iter().filter(|p| near(p.frame.pixel(160, 120), [20, 40, 230]) && near(p.frame.pixel(2, 90), ECHO_BORDER)).count();
    assert!(echoed > 10, "{echoed} of {} decoded echo frames show the camera with the overlay", pics.len());

    // get_state: the model's counters and A's ingest.
    a.peer.send_message(command("get_state", json!({}), "g1")).await.unwrap();
    pump(&mut a.peer, &mut sa, Duration::from_secs(5), |s| s.reply("g1").is_some()).await;
    let (t, st) = sa.reply("g1").unwrap();
    assert_eq!(t, "state_update");
    assert!(st["input_frames"].as_f64().unwrap() > 10.0, "{st}");
    assert_eq!(st["context"]["scene"], "a studio");
    assert_eq!(st["publishers"]["input_video"].as_f64(), Some(a.cid as f64));
    assert_eq!(st["ingest"]["codec"], "vp8");
    assert!(st["ingest"]["video_frames_decoded"].as_f64().unwrap() > 10.0, "{st}");
    let size = &st["ingest"]["source_size"];
    assert_eq!((size[0].as_f64(), size[1].as_f64()), (Some(320.0), Some(240.0)), "{st}");

    // Unpublish frees the slot for B.
    a.peer.send_message(control(pb::control_client_message::Payload::UnpublishTrack(pb::UnpublishTrack { name: "input_video".into() }), "")).await.unwrap();
    a.peer.send_message(command("get_state", json!({}), "g2")).await.unwrap();
    pump(&mut a.peer, &mut sa, Duration::from_secs(5), |s| s.reply("g2").is_some()).await;
    let (_, st) = sa.reply("g2").unwrap();
    let pubs = st["publishers"].as_object().unwrap();
    assert_eq!(pubs.len(), 1, "{st}");
    assert_eq!(pubs["input_audio"].as_f64(), Some(a.cid as f64), "{st}");
    b.peer.send_message(publish("input_video", "p5")).await.unwrap();
    // B has not been pumped while A published: its echo backlog comes
    // first, which takes a while on a loaded host.
    pump(&mut b.peer, &mut sb, Duration::from_secs(30), |s| s.publish_reply("p5").is_some()).await;
    assert_eq!(sb.publish_reply("p5"), Some(Ok(())));

    let (s, _) = call(&app, "POST", "/stop_session", None, false).await;
    assert_eq!(s, StatusCode::UNAUTHORIZED);
    let (s, _) = call(&app, "POST", "/stop_session", None, true).await;
    assert_eq!(s, StatusCode::OK);
    rt.drain().await;
}
