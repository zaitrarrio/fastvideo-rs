//! The WMA director end to end (design §5.6): signalling routes, a real
//! WebRTC client (a second str0m host) on loopback, the control channel,
//! and A/V received at the model rate from the fake engine.
//!
//! Media tests need an H.264 encoder: ffmpeg with libx264 (`FV_FFMPEG` or
//! `ffmpeg` on PATH), else OpenH264 (`--features openh264`); they skip
//! otherwise. The browser test (`fal.realtime.open` in Chromium) lives in
//! `director_browser.rs`.
#![cfg(feature = "director")]

mod director_common;

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use director_common::*;
use fastvideo_protocol::{EndReason, SessionState};
use fastvideo_webrtc::channel::ChannelMessage;
use fastvideo_webrtc::host::{AudioLayout, HostConfig, OfferOptions, PeerEvent, PeerHandle, RtcHost};
use fastvideo_webrtc::sdp::{Direction, MediaKind, Sdp};
use fastvideo_webrtc::writer::TrackKind;
use serde_json::{json, Value};
use tokio::sync::mpsc;

/// What the client received.
#[derive(Default, Debug)]
struct MediaLog {
    /// (arrival, rtp_time) per video frame.
    video: Vec<(Instant, u64)>,
    /// (arrival, rtp_time) per Opus packet.
    audio: Vec<(Instant, u64)>,
    keyframes: u64,
}

struct Client {
    _host: RtcHost,
    beats: Option<tokio::task::JoinHandle<()>>,
    peer: PeerHandle,
    msgs: mpsc::UnboundedReceiver<Value>,
    media: Arc<Mutex<MediaLog>>,
    session_id: String,
    answer: String,
}

impl Drop for Client {
    fn drop(&mut self) {
        if let Some(b) = self.beats.take() {
            b.abort();
        }
    }
}

impl Client {
    /// Stop heartbeating (as a client that went away does).
    fn stop_beats(&mut self) {
        if let Some(b) = self.beats.take() {
            b.abort();
        }
    }

    async fn send(&self, v: Value) {
        self.peer.send_message(ChannelMessage::text("control", v.to_string())).await.unwrap();
    }

    /// The next control message of type `ty` (others are skipped, but
    /// `error` messages fail the test unless asked for).
    async fn expect(&mut self, ty: &str) -> Value {
        let deadline = tokio::time::Instant::now() + T;
        loop {
            let m = tokio::time::timeout_at(deadline, self.msgs.recv())
                .await
                .unwrap_or_else(|_| panic!("timed out waiting for `{ty}`"))
                .unwrap_or_else(|| panic!("channel closed waiting for `{ty}`"));
            if m["type"] == ty {
                return m;
            }
            if m["type"] == "error" && ty != "error" {
                panic!("unexpected error while waiting for `{ty}`: {m}");
            }
        }
    }

    /// Video frames / audio samples per second over `window`, measured
    /// from arrival times and RTP clocks.
    async fn rates(&self, window: Duration) -> (f64, f64, f64, f64) {
        let (v0, a0) = {
            let l = self.media.lock().unwrap();
            (l.video.len(), l.audio.len())
        };
        let t0 = Instant::now();
        tokio::time::sleep(window).await;
        let dt = t0.elapsed().as_secs_f64();
        let l = self.media.lock().unwrap();
        let v = &l.video[v0..];
        let a = &l.audio[a0..];
        let fps = v.len() as f64 / dt;
        let rtp_fps = if v.len() > 1 { (v.len() - 1) as f64 * 90_000.0 / (v.last().unwrap().1 - v[0].1) as f64 } else { 0.0 };
        let audio_rate = if a.len() > 1 { (a.last().unwrap().1 - a[0].1) as f64 / dt } else { 0.0 };
        let audio_rtp_step = if a.len() > 1 { (a.last().unwrap().1 - a[0].1) as f64 / (a.len() - 1) as f64 } else { 0.0 };
        (fps, rtp_fps, audio_rate, audio_rtp_step)
    }
}

async fn open(f: &Fixture, app_id: &str) -> Client {
    let host = RtcHost::bind(HostConfig::loopback(true, false)).await.unwrap();
    let (pending, offer) = host
        .offer(OfferOptions {
            video: Some(Direction::RecvOnly),
            audio: Some((Direction::RecvOnly, AudioLayout::Stereo)),
            channels: vec!["control".into()],
            ..Default::default()
        })
        .await
        .unwrap();
    let r = post(&f.app, "/wma/session", json!({"app_id": app_id, "sdp": offer, "type": "offer"})).await;
    assert_eq!(r.status, 200, "{}", r.text());
    let v = r.json();
    assert_eq!(v["type"], "answer");
    let answer = v["sdp"].as_str().unwrap().to_owned();
    let session_id = v["session_id"].as_str().unwrap().to_owned();
    assert_eq!(session_id.len(), 36, "uuid session id");
    // Non-trickle: every candidate is in the answer.
    assert!(Sdp::parse(&answer).unwrap().has_end_of_candidates(), "{answer}");
    let peer = pending.accept_answer(&answer).await.unwrap();
    let (handle, mut events) = peer.split();
    let (tx, msgs) = mpsc::unbounded_channel();
    let media = Arc::new(Mutex::new(MediaLog::default()));
    let log = media.clone();
    tokio::spawn(async move {
        while let Some(e) = events.recv().await {
            match e {
                PeerEvent::Message(m) => {
                    let v: Value = serde_json::from_str(m.as_text().unwrap()).unwrap();
                    let _ = tx.send(v);
                }
                PeerEvent::Media { kind, rtp_time, keyframe, .. } => {
                    let mut l = log.lock().unwrap();
                    match kind {
                        TrackKind::Video => {
                            l.video.push((Instant::now(), rtp_time));
                            l.keyframes += u64::from(keyframe);
                        }
                        TrackKind::Audio => l.audio.push((Instant::now(), rtp_time)),
                    }
                }
                PeerEvent::Closed(_) => break,
                _ => {}
            }
        }
    });
    // Heartbeats every 5 s, as the JS client sends them.
    let app = f.app.clone();
    let sid = session_id.clone();
    let beats = tokio::spawn(async move {
        let mut t = tokio::time::interval(Duration::from_secs(5));
        loop {
            t.tick().await;
            let _ = post(&app, "/wma/session/heartbeat", json!({"session_id": sid})).await;
        }
    });
    Client { _host: host, beats: Some(beats), peer: handle, msgs, media, session_id, answer }
}

fn h264_or_skip() -> Option<fastvideo_media::video::EncoderBackend> {
    let b = h264_backend();
    if b.is_none() {
        eprintln!("skipped: no H.264 encoder (set FV_FFMPEG to an ffmpeg with libx264, or build with --features openh264)");
    }
    b
}

async fn wait_closed(f: &Fixture, id: &str, within: Duration) -> Option<SessionState> {
    let h = f.svc.session(id)?;
    let mut rx = h.closed();
    tokio::time::timeout(within, rx.wait_for(|c| *c)).await.ok()?.ok()?;
    Some(h.state())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn av_session_end_to_end() {
    let Some(h264) = h264_or_skip() else { return };
    let f = fixture(Opts { h264, ..Opts::default() }).await;
    let mut c = open(&f, "minimax/h3-max/director").await;

    // session_info first, with our constants.
    let info = c.expect("session_info").await;
    assert_eq!(info["fps"], 24);
    assert_eq!(info["app"], "minimax-h3-max-director");
    assert_eq!(info["resolutions"], json!(["480p", "768p"]));
    assert_eq!(info["audio_sample_rate"], 48_000);
    assert_eq!(info["continuation_context_frames"], 1);

    // Strict schemas and ordering rules.
    c.send(json!({"type": "configure", "prompt_version": 1, "prompt": "x", "bogus": 1})).await;
    let e = c.expect("error").await;
    assert_eq!((e["code"].as_str(), e["prompt_version"].as_u64()), (Some("invalid_message"), Some(1)));
    c.send(json!({"type": "prompt", "prompt_version": 2, "prompt": "early"})).await;
    assert_eq!(c.expect("error").await["code"], "not_configured");
    c.send(json!({"type": "ping", "ts": 1234.5})).await;
    assert_eq!(c.expect("pong").await["client_ts"], 1234.5);

    c.send(json!({"type": "configure", "prompt_version": 1, "prompt": "A lighthouse at dusk", "resolution": "480p", "aspect_ratio": "16:9", "memory": 3, "protocol_version": 1})).await;
    let cfgd = c.expect("configured").await;
    assert_eq!(cfgd["prompt_version"], 1);
    assert_eq!(cfgd["resolution"], "480p");
    assert_eq!(cfgd["has_initial_audio"], false);
    assert_eq!(cfgd["enable_safety_checker"], false);
    c.send(json!({"type": "configure", "prompt_version": 1, "prompt": "again"})).await;
    assert_eq!(c.expect("error").await["code"], "immutable_settings");

    let ch0 = c.expect("chunk").await;
    let t_ch0 = Instant::now();
    assert_eq!(ch0["chunk_index"], 0);
    assert_eq!(ch0["generated_frame_count"], 124, "5 s on 17n+5: {ch0}");
    assert_eq!(ch0["trimmed_context_frames"], 0);
    assert_eq!(ch0["route"], "unknown");
    assert!(ch0["dispatch"]["wall_ms"].as_f64().unwrap() > 0.0);
    let m0 = c.expect("chunk_metrics").await;
    assert_eq!((m0["chunk_index"].as_u64(), m0["units"].as_str()), (Some(0), Some("ms")));

    // A direction change: pending now, applied when its chunk dispatches.
    c.send(json!({"type": "prompt", "prompt_version": 2, "prompt": "The storm arrives"})).await;
    assert_eq!(c.expect("prompt_pending").await["prompt_version"], 2);
    c.send(json!({"type": "prompt", "prompt_version": 2, "prompt": "reused"})).await;
    let rej = c.expect("prompt_rejected").await;
    assert_eq!((rej["prompt_version"].as_u64(), rej["reason"].as_str()), (Some(2), Some("stale_prompt_version")));
    assert_eq!(c.expect("prompt_applied").await["prompt_version"], 2);

    // Continuation chunks start from the previous last frame (trimmed).
    let ch1 = c.expect("chunk").await;
    assert_eq!(ch1["chunk_index"], 1);
    assert_eq!((ch1["trimmed_context_frames"].as_u64(), ch1["presented_frame_count"].as_u64()), (Some(1), Some(123)));
    assert_eq!(ch1["prompt_version"], 2);

    {
        // Audio (silence) flows from the start; video from the first chunk.
        let l = c.media.lock().unwrap();
        assert!(l.audio.first().is_some_and(|a| a.0 <= t_ch0), "audio before the first chunk");
        assert!(!l.video.is_empty(), "video after the first chunk");
    }
    // A/V at 24 fps / 48 kHz.
    let (fps, rtp_fps, audio_rate, audio_step) = c.rates(Duration::from_secs(4)).await;
    eprintln!("received: {fps:.2} fps (rtp {rtp_fps:.2}), audio {audio_rate:.0} samples/s, {audio_step} per packet");
    assert!((fps - 24.0).abs() < 2.4, "video arrives at {fps} fps");
    assert!((rtp_fps - 24.0).abs() < 0.01, "video RTP advances at {rtp_fps} fps");
    assert!((audio_rate - 48_000.0).abs() < 4_800.0, "audio arrives at {audio_rate} samples/s");
    assert_eq!(audio_step, 960.0, "20 ms Opus packets");
    assert!(c.answer.contains("stereo=1"), "stereo Opus: {}", c.answer);
    assert!(c.media.lock().unwrap().keyframes > 0);

    // Heartbeats keep the lease.
    let hb = post(&f.app, "/wma/session/heartbeat", json!({"session_id": c.session_id})).await;
    assert_eq!(hb.json(), json!({"alive": true}));

    c.send(json!({"type": "stop"})).await;
    let ex = c.expect("stream_exhausted").await;
    assert_eq!(ex["reason"], "stopped");
    assert!(ex["chunks"].as_u64().unwrap() >= 2);
    let fm = c.expect("session_metrics").await;
    assert_eq!(fm["final"], true);
    let st = wait_closed(&f, &c.session_id, Duration::from_secs(10)).await;
    assert!(matches!(st, None | Some(SessionState::Closed(EndReason::Stopped))), "{st:?}");
    tokio::time::sleep(Duration::from_millis(200)).await;
    let hb = post(&f.app, "/wma/session/heartbeat", json!({"session_id": c.session_id})).await;
    assert_eq!(hb.json(), json!({"alive": false}));
    // The executor is free again: a new session is admitted.
    let _c2 = open(&f, "minimax/h3-max/director").await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn video_only_session() {
    let Some(h264) = h264_or_skip() else { return };
    let f = fixture(Opts { h264, ..Opts::default() }).await;
    let mut c = open(&f, "fv/h3-silent/director").await;
    // The offered audio m-line is answered inactive (design §5.3).
    let a = Sdp::parse(&c.answer).unwrap();
    let audio = a.media.iter().find(|m| m.kind() == MediaKind::Audio).unwrap();
    assert_eq!(audio.direction(), Direction::Inactive, "{}", c.answer);
    c.expect("session_info").await;
    c.send(json!({"type": "configure", "prompt_version": 1, "prompt": "silent film", "resolution": "480p"})).await;
    c.expect("configured").await;
    c.expect("chunk").await;
    let (fps, rtp_fps, _, _) = c.rates(Duration::from_secs(3)).await;
    eprintln!("video-only: {fps:.2} fps (rtp {rtp_fps:.2})");
    assert!((fps - 24.0).abs() < 2.4, "{fps}");
    assert!((rtp_fps - 24.0).abs() < 0.01, "{rtp_fps}");
    assert!(c.media.lock().unwrap().audio.is_empty(), "no audio on a video-only session");
    c.send(json!({"type": "stop"})).await;
    c.expect("stream_exhausted").await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn signalling_rules() {
    let f = fixture(Opts::default()).await;
    // /ice: config servers, strict body, known apps, auth.
    let r = post(&f.app, "/wma/ice", json!({"app_id": "minimax/h3-max/director"})).await;
    assert_eq!((r.status.as_u16(), r.json()), (200, json!({"ice_servers": []})));
    let r = post(&f.app, "/wma/ice", json!({})).await;
    assert_eq!((r.status.as_u16(), r.text()), (422, "missing field `app_id`".to_string()));
    let r = post(&f.app, "/wma/ice", json!({"app_id": "someone/else/director"})).await;
    assert_eq!(r.status, 404);
    assert!(r.json()["error"].is_string());
    let r = send(&f.app, req("POST", "/wma/ice", None, Some(json!({"app_id": "minimax/h3-max/director"}).to_string()))).await;
    assert_eq!(r.status, 401);
    for path in ["/minimax/h3-max/director/ice", "/run/minimax/h3-max/director/ice"] {
        let r = post(&f.app, path, json!({})).await;
        assert_eq!((r.status.as_u16(), r.json()), (200, json!({"ice_servers": []})), "{path}");
    }
    // Through the fal proxy, as `proxyUrl` clients call it.
    for (target, body) in [
        ("https://wma.fal.run/ice", json!({"app_id": "minimax/h3-max/director"})),
        ("https://fal.run/minimax/h3-max/director/ice", json!({})),
    ] {
        let mut rq = req("POST", "/fal/proxy", Some(KEY), Some(body.to_string()));
        rq.headers_mut().insert("x-fal-target-url", target.parse().unwrap());
        let r = send(&f.app, rq).await;
        assert_eq!((r.status.as_u16(), r.json()), (200, json!({"ice_servers": []})), "{target}");
    }
    // /session: strict, typed, known apps.
    let r = post(&f.app, "/wma/session", json!({"app_id": "minimax/h3-max/director", "sdp": "v=0", "type": "answer"})).await;
    assert_eq!(r.status, 422);
    let r = post(&f.app, "/wma/session", json!({"app_id": "minimax/h3-max/director", "type": "offer"})).await;
    assert_eq!((r.status.as_u16(), r.text()), (422, "missing field `sdp`".to_string()));
    let r = post(&f.app, "/wma/session", json!({"app_id": "minimax/h3-max/director", "sdp": "v=0\r\n", "type": "offer"})).await;
    assert_eq!(r.status, 422, "{}", r.text());
    assert!(r.json()["error"].as_str().unwrap().contains("offer"));
    // A bad offer released the engine session: one real session is admitted,
    // a second is busy (429).
    let c = open(&f, "minimax/h3-max/director").await;
    let host = RtcHost::bind(HostConfig::loopback(true, false)).await.unwrap();
    let (_p, offer) = host.offer(OfferOptions { channels: vec!["control".into()], ..Default::default() }).await.unwrap();
    let r = post(&f.app, "/wma/session", json!({"app_id": "minimax/h3-max/director", "sdp": offer, "type": "offer"})).await;
    assert_eq!(r.status, 429, "{}", r.text());
    assert!(r.json()["error"].is_string());
    // Heartbeats.
    let hb = post(&f.app, "/wma/session/heartbeat", json!({"session_id": c.session_id})).await;
    assert_eq!(hb.json(), json!({"alive": true}));
    let hb = post(&f.app, "/wma/session/heartbeat", json!({"session_id": "nope"})).await;
    assert_eq!(hb.json(), json!({"alive": false}));
    let hb = post(&f.app, "/wma/session/heartbeat", json!({"session": "x"})).await;
    assert_eq!(hb.status, 422);
    // Runner-side /info.
    for m in ["GET", "POST"] {
        let r = send(&f.app, req(m, "/info", Some(KEY), None)).await;
        assert_eq!(r.status, 200, "{m} {}", r.text());
        let v = r.json();
        assert_eq!(v["app"], "minimax-h3-max-director");
        assert_eq!(v["fps"], 24);
        assert!(v.get("type").is_none());
    }
    let r = send(&f.app, req("POST", "/info", None, None)).await;
    assert_eq!(r.status, 401);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn heartbeat_expiry_closes_the_session() {
    let f = fixture(Opts { heartbeat_timeout: Duration::from_millis(1500), ..Opts::default() }).await;
    // Beating keeps it alive past the timeout.
    let mut c = open(&f, "minimax/h3-max/director").await;
    c.stop_beats();
    c.expect("session_info").await;
    for _ in 0..4 {
        tokio::time::sleep(Duration::from_millis(600)).await;
        let hb = post(&f.app, "/wma/session/heartbeat", json!({"session_id": c.session_id})).await;
        assert_eq!(hb.json(), json!({"alive": true}));
    }
    // Then silence: Closing(ClientGone) within the timeout (plus a tick).
    let t0 = Instant::now();
    let st = wait_closed(&f, &c.session_id, Duration::from_secs(5)).await;
    assert!(matches!(st, None | Some(SessionState::Closed(EndReason::ClientGone))), "{st:?}");
    assert!(t0.elapsed() >= Duration::from_millis(1200), "closed too early: {:?}", t0.elapsed());
    tokio::time::sleep(Duration::from_millis(100)).await;
    let hb = post(&f.app, "/wma/session/heartbeat", json!({"session_id": c.session_id})).await;
    assert_eq!(hb.json(), json!({"alive": false}));
    // The engine session was released.
    let _again = open(&f, "minimax/h3-max/director").await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn configure_failures_end_the_session() {
    let f = fixture(Opts::default()).await;
    let mut c = open(&f, "minimax/h3-max/director").await;
    c.expect("session_info").await;
    c.send(json!({"type": "configure", "prompt_version": 1, "prompt": "x", "resolution": "1080p"})).await;
    let e = c.expect("error").await;
    assert_eq!(e["code"], "invalid_input");
    let st = wait_closed(&f, &c.session_id, Duration::from_secs(5)).await;
    assert!(matches!(st, None | Some(SessionState::Closed(EndReason::Error(_)))), "{st:?}");

    let mut c = open(&f, "minimax/h3-max/director").await;
    c.expect("session_info").await;
    c.send(json!({"type": "configure", "prompt_version": 1, "prompt": "x", "audio_url": "https://example.com/a.wav"})).await;
    assert_eq!(c.expect("error").await["code"], "invalid_initial_audio");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn session_limit_and_runner_sse() {
    let Some(h264) = h264_or_skip() else { return };
    let f = fixture(Opts { h264, max_session_seconds: Some(2), ..Opts::default() }).await;
    let mut c = open(&f, "minimax/h3-max/director").await;
    let info = c.expect("session_info").await;
    assert_eq!(info["max_session_seconds"], 2);
    c.send(json!({"type": "configure", "prompt_version": 1, "prompt": "short", "resolution": "480p"})).await;
    c.expect("configured").await;
    let ex = c.expect("stream_exhausted").await;
    assert_eq!(ex["reason"], "session_limit");
    let st = wait_closed(&f, &c.session_id, Duration::from_secs(10)).await;
    assert!(matches!(st, None | Some(SessionState::Closed(EndReason::SessionLimit))), "{st:?}");
    tokio::time::sleep(Duration::from_millis(200)).await;

    // Runner side: SSE whose first event is the answer; dropping the
    // response ends the session.
    let host = RtcHost::bind(HostConfig::loopback(true, false)).await.unwrap();
    let (_pending, offer) = host
        .offer(OfferOptions { video: Some(Direction::RecvOnly), channels: vec!["control".into()], ..Default::default() })
        .await
        .unwrap();
    let r = f
        .app
        .clone()
        .oneshot_req(req("POST", "/start-session", Some(KEY), Some(json!({"sdp": offer, "type": "offer", "session_id": "runner-1"}).to_string())))
        .await;
    assert_eq!(r.status(), 200);
    assert_eq!(r.headers()["content-type"], "text/event-stream");
    use futures::StreamExt;
    let mut body = r.into_body().into_data_stream();
    let first = tokio::time::timeout(T, body.next()).await.unwrap().unwrap().unwrap();
    let text = String::from_utf8_lossy(&first).into_owned();
    assert!(text.starts_with("data: "), "{text}");
    let ev: Value = serde_json::from_str(text.trim_start_matches("data: ").trim()).unwrap();
    assert_eq!((ev["type"].as_str(), ev["session_id"].as_str()), (Some("answer"), Some("runner-1")));
    assert!(ev["sdp"].as_str().unwrap().contains("a=end-of-candidates"));
    // Runner sessions are not heartbeat sessions.
    let hb = post(&f.app, "/wma/session/heartbeat", json!({"session_id": "runner-1"})).await;
    assert_eq!(hb.json(), json!({"alive": false}));
    let h = f.svc.session("runner-1").expect("registered");
    let mut closed = h.closed();
    drop(body);
    tokio::time::timeout(Duration::from_secs(5), closed.wait_for(|c| *c)).await.unwrap().unwrap();
    assert_eq!(h.state(), SessionState::Closed(EndReason::ClientGone));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn late_chunks_report_deadline_missed() {
    let Some(h264) = h264_or_skip() else { return };
    // Builds slower than real time: each 5 s chunk takes ~7 s.
    let f = fixture(Opts { h264, step: Duration::from_millis(870), ..Opts::default() }).await;
    let mut c = open(&f, "minimax/h3-max/director").await;
    c.expect("session_info").await;
    c.send(json!({"type": "configure", "prompt_version": 1, "prompt": "slow", "resolution": "480p"})).await;
    c.expect("configured").await;
    let d = c.expect("deadline_missed").await;
    assert_eq!(d["chunk_index"], 1);
    assert_eq!(d["behavior"], "freeze_video_and_silence_audio_until_ready");
    assert!(d["late_by_seconds"].as_f64().unwrap() > 0.5, "{d}");
    // Audio kept flowing (silence) through the underrun.
    let (_, _, audio_rate, _) = c.rates(Duration::from_secs(1)).await;
    assert!(audio_rate > 40_000.0, "{audio_rate}");
    c.send(json!({"type": "stop"})).await;
    c.expect("stream_exhausted").await;
}

/// `oneshot` returning the raw response (streaming body).
trait OneshotReq {
    async fn oneshot_req(self, r: axum::http::Request<axum::body::Body>) -> axum::response::Response;
}

impl OneshotReq for axum::Router {
    async fn oneshot_req(self, r: axum::http::Request<axum::body::Body>) -> axum::response::Response {
        use tower::ServiceExt;
        self.oneshot(r).await.unwrap()
    }
}
