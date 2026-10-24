//! The WMA director end to end (design §5.6): signalling routes, a real
//! WebRTC client (a second str0m host) on loopback, the control channel,
//! and A/V received at the model rate from the fake engine.
//!
//! Media tests need an H.264 encoder: ffmpeg with libx264 (`FV_FFMPEG` or
//! `ffmpeg` on PATH), else OpenH264 (`--features openh264`); they skip
//! otherwise. The browser test (`fal.realtime.open` in Chromium) lives in
//! `director_browser.rs`.
//!
//! Each media scenario runs twice. The plain test checks what holds on a
//! loaded host (protocol, order, the RTP grids, frame and packet counts);
//! its `realtime_` twin, ignored by default, also measures the delivery
//! rates (24 fps, 48 kHz) and runs in `scripts/serve/check.sh --realtime`,
//! serialized after the build (docs/dev/testing.md).
#![cfg(feature = "director")]

mod director_common;

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use director_common::*;
use fastvideo_media::clock::ManualClock;
use fastvideo_protocol::{EndReason, SessionState};
use fastvideo_webrtc::channel::ChannelMessage;
use fastvideo_webrtc::host::{
    AudioLayout, HostConfig, OfferOptions, PeerEvent, PeerHandle, RtcHost,
};
use fastvideo_webrtc::sdp::{Direction, MediaKind, Sdp};
use fastvideo_webrtc::writer::{TrackKind, VideoCodec};
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
    /// The first video frame's payload (to tell H.264 from VP8).
    first_video: Option<bytes::Bytes>,
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
        self.peer
            .send_message(ChannelMessage::text("control", v.to_string()))
            .await
            .unwrap();
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

    /// Waits for the first video frame, then for the encoder's input queue
    /// to drain. Video follows the first chunk once the encoder has started:
    /// an ffmpeg encoder can take a second or more to start on a loaded
    /// host (measured 1-2 s on the build pod), longer than chunk 1 takes to
    /// build, and the (up to 10) frames queued meanwhile go out in a burst
    /// that a rate window must not count.
    async fn first_video(&self) {
        let deadline = Instant::now() + T;
        while self.media.lock().unwrap().video.is_empty() {
            assert!(Instant::now() < deadline, "no video after the first chunk");
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        tokio::time::sleep(Duration::from_secs(1)).await;
    }

    /// Waits until `video` more frames and `audio` more Opus packets have
    /// arrived (counts, not rates: a loaded host delivers them late).
    async fn more_media(&self, video: usize, audio: usize) {
        let (v0, a0) = {
            let l = self.media.lock().unwrap();
            (l.video.len(), l.audio.len())
        };
        let deadline = Instant::now() + T;
        loop {
            let (v, a) = {
                let l = self.media.lock().unwrap();
                (l.video.len() - v0, l.audio.len() - a0)
            };
            if v >= video && a >= audio {
                return;
            }
            assert!(
                Instant::now() < deadline,
                "{v} of {video} video frames, {a} of {audio} audio packets"
            );
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    }

    /// What was received is on the model's clocks, in order: video RTP
    /// times on the 24 fps grid (3750 ticks of 90 kHz), audio on the 20 ms
    /// Opus grid (960 samples). A frame or packet lost to load leaves a
    /// multiple of the step, never an off-grid time.
    fn assert_rtp_grids(&self) {
        self.assert_rtp_grids_at(24);
    }

    /// [`Self::assert_rtp_grids`] for a model at `fps` (90 kHz / `fps`
    /// ticks per frame).
    fn assert_rtp_grids_at(&self, fps: u32) {
        let step = u64::from(90_000 / fps);
        let l = self.media.lock().unwrap();
        let bad = l
            .video
            .windows(2)
            .find(|w| w[1].1 <= w[0].1 || (w[1].1 - w[0].1) % step != 0);
        assert!(bad.is_none(), "video RTP off the {fps} fps grid: {bad:?}");
        let bad = l
            .audio
            .windows(2)
            .find(|w| w[1].1 <= w[0].1 || (w[1].1 - w[0].1) % 960 != 0);
        assert!(bad.is_none(), "audio RTP off the 20 ms grid: {bad:?}");
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
        let rtp_fps = if v.len() > 1 {
            (v.len() - 1) as f64 * 90_000.0 / (v.last().unwrap().1 - v[0].1) as f64
        } else {
            0.0
        };
        let audio_rate = if a.len() > 1 {
            (a.last().unwrap().1 - a[0].1) as f64 / dt
        } else {
            0.0
        };
        let audio_rtp_step = if a.len() > 1 {
            (a.last().unwrap().1 - a[0].1) as f64 / (a.len() - 1) as f64
        } else {
            0.0
        };
        (fps, rtp_fps, audio_rate, audio_rtp_step)
    }
}

async fn open(f: &Fixture, app_id: &str) -> Client {
    open_with(f, app_id, vec![VideoCodec::H264]).await
}

/// [`open`] with the video codecs the client offers (a browser offers
/// H.264 and VP8).
async fn open_with(f: &Fixture, app_id: &str, video_codecs: Vec<VideoCodec>) -> Client {
    let host = RtcHost::bind(HostConfig::loopback(true, false))
        .await
        .unwrap();
    let (pending, offer) = host
        .offer(OfferOptions {
            video: Some(Direction::RecvOnly),
            audio: Some((Direction::RecvOnly, AudioLayout::Stereo)),
            channels: vec!["control".into()],
            video_codecs,
            ..Default::default()
        })
        .await
        .unwrap();
    let r = post(
        &f.app,
        "/wma/session",
        json!({"app_id": app_id, "sdp": offer, "type": "offer"}),
    )
    .await;
    assert_eq!(r.status, 200, "{}", r.text());
    let v = r.json();
    assert_eq!(v["type"], "answer");
    let answer = v["sdp"].as_str().unwrap().to_owned();
    let session_id = v["session_id"].as_str().unwrap().to_owned();
    assert_eq!(session_id.len(), 36, "uuid session id");
    // Non-trickle: every candidate is in the answer.
    assert!(
        Sdp::parse(&answer).unwrap().has_end_of_candidates(),
        "{answer}"
    );
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
                PeerEvent::Media {
                    kind,
                    rtp_time,
                    keyframe,
                    data,
                    ..
                } => {
                    let mut l = log.lock().unwrap();
                    match kind {
                        TrackKind::Video => {
                            l.first_video.get_or_insert(data);
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
    Client {
        _host: host,
        beats: Some(beats),
        peer: handle,
        msgs,
        media,
        session_id,
        answer,
    }
}

fn h264_or_skip() -> Option<fastvideo_media::video::EncoderBackend> {
    let b = h264_backend();
    if b.is_none() {
        eprintln!("skipped: no H.264 encoder (set FV_FFMPEG to an ffmpeg with libx264, or build with --features openh264)");
    }
    b
}

/// The heartbeat route reports a closed session gone (it is dropped from
/// the registry just after it closes).
async fn assert_not_alive(f: &Fixture, id: &str) {
    let deadline = Instant::now() + T;
    loop {
        let hb = post(&f.app, "/wma/session/heartbeat", json!({"session_id": id}))
            .await
            .json();
        if hb == json!({"alive": false}) {
            return;
        }
        assert_eq!(hb, json!({"alive": true}));
        assert!(
            Instant::now() < deadline,
            "session {id} still alive after it closed"
        );
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
}

async fn wait_closed(f: &Fixture, id: &str, within: Duration) -> Option<SessionState> {
    let h = f.svc.session(id)?;
    let mut rx = h.closed();
    tokio::time::timeout(within, rx.wait_for(|c| *c))
        .await
        .ok()?
        .ok()?;
    Some(h.state())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn av_session_end_to_end() {
    av_session(false).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "real-time rates: scripts/serve/check.sh --realtime"]
async fn realtime_av_session_end_to_end() {
    av_session(true).await;
}

/// `realtime`: also measure the A/V delivery rates.
async fn av_session(realtime: bool) {
    let Some(h264) = h264_or_skip() else { return };
    let f = fixture(Opts {
        h264,
        hold_builds: true,
        ..Opts::default()
    })
    .await;
    let mut c = open(&f, "minimax/h3-max/director").await;

    // session_info first, with our constants.
    let info = c.expect("session_info").await;
    assert_eq!(info["fps"], 24);
    assert_eq!(info["app"], "minimax-h3-max-director");
    assert_eq!(info["resolutions"], json!(["480p", "768p"]));
    assert_eq!(info["audio_sample_rate"], 48_000);
    assert_eq!(info["continuation_context_frames"], 1);
    assert_eq!(
        (
            info["default_chunk_duration"].as_u64(),
            info["chunk_seconds"].as_u64()
        ),
        (Some(5), Some(5))
    );
    assert_eq!(info["chunk_duration_options"], json!([5, 10]));
    assert_eq!(info["chunk_duration_frames"], json!({"5": 124, "10": 243}));

    // Strict schemas and ordering rules.
    c.send(json!({"type": "configure", "prompt_version": 1, "prompt": "x", "chunk_duration": 7}))
        .await;
    let e = c.expect("error").await;
    assert_eq!(e["code"], "invalid_message");
    assert!(
        e["error"].as_str().unwrap().contains("chunk_duration"),
        "{e}"
    );
    c.send(json!({"type": "configure", "prompt_version": 1, "prompt": "x", "bogus": 1}))
        .await;
    let e = c.expect("error").await;
    assert_eq!(
        (e["code"].as_str(), e["prompt_version"].as_u64()),
        (Some("invalid_message"), Some(1))
    );
    c.send(json!({"type": "prompt", "prompt_version": 2, "prompt": "early"}))
        .await;
    assert_eq!(c.expect("error").await["code"], "not_configured");
    c.send(json!({"type": "ping", "ts": 1234.5})).await;
    assert_eq!(c.expect("pong").await["client_ts"], 1234.5);

    c.send(json!({"type": "configure", "prompt_version": 1, "prompt": "A lighthouse at dusk", "resolution": "480p", "aspect_ratio": "16:9", "memory": 3, "protocol_version": 1})).await;
    let cfgd = c.expect("configured").await;
    assert_eq!(cfgd["prompt_version"], 1);
    assert_eq!(cfgd["resolution"], "480p");
    assert_eq!(cfgd["has_initial_audio"], false);
    assert_eq!(cfgd["enable_safety_checker"], false);
    assert_eq!(cfgd["chunk_duration"], 5, "the default chunk");
    c.send(json!({"type": "configure", "prompt_version": 1, "prompt": "again"}))
        .await;
    assert_eq!(c.expect("error").await["code"], "immutable_settings");

    // A direction change: pending now, applied when its chunk dispatches.
    // Chunk 0 was dispatched at `configure` (prompt_version 1) and its
    // build is held, so the prompt targets chunk 1 on any machine: chunk 1
    // is dispatched once chunk 0 is built and playing. (Sent after chunk 0
    // arrived, the prompt raced that dispatch, which follows within one
    // playout tick: it landed on chunk 1 or chunk 2, and `prompt_applied`
    // could precede the reply to the second prompt.)
    c.send(json!({"type": "prompt", "prompt_version": 2, "prompt": "The storm arrives"}))
        .await;
    assert_eq!(c.expect("prompt_pending").await["prompt_version"], 2);
    c.send(json!({"type": "prompt", "prompt_version": 2, "prompt": "reused"}))
        .await;
    let rej = c.expect("prompt_rejected").await;
    assert_eq!(
        (rej["prompt_version"].as_u64(), rej["reason"].as_str()),
        (Some(2), Some("stale_prompt_version"))
    );
    f.release_builds();

    let ch0 = c.expect("chunk").await;
    let t_ch0 = Instant::now();
    assert_eq!(ch0["chunk_index"], 0);
    assert_eq!(ch0["prompt_version"], 1);
    assert_eq!(ch0["generated_frame_count"], 124, "5 s on 17n+5: {ch0}");
    assert_eq!(ch0["trimmed_context_frames"], 0);
    assert_eq!(ch0["route"], "unknown");
    assert!(ch0["dispatch"]["wall_ms"].as_f64().unwrap() > 0.0);
    let m0 = c.expect("chunk_metrics").await;
    assert_eq!(
        (m0["chunk_index"].as_u64(), m0["units"].as_str()),
        (Some(0), Some("ms"))
    );

    assert_eq!(c.expect("prompt_applied").await["prompt_version"], 2);

    // Continuation chunks start from the previous last frame (trimmed).
    let ch1 = c.expect("chunk").await;
    assert_eq!(ch1["chunk_index"], 1);
    assert_eq!(
        (
            ch1["trimmed_context_frames"].as_u64(),
            ch1["presented_frame_count"].as_u64()
        ),
        (Some(1), Some(123))
    );
    assert_eq!(ch1["prompt_version"], 2);

    // Audio (silence) flows from the start; video from the first chunk.
    assert!(
        c.media
            .lock()
            .unwrap()
            .audio
            .first()
            .is_some_and(|a| a.0 <= t_ch0),
        "audio before the first chunk"
    );
    c.first_video().await;
    if realtime {
        // A/V at 24 fps / 48 kHz.
        let (fps, rtp_fps, audio_rate, audio_step) = c.rates(Duration::from_secs(4)).await;
        eprintln!("received: {fps:.2} fps (rtp {rtp_fps:.2}), audio {audio_rate:.0} samples/s, {audio_step} per packet");
        assert!((fps - 24.0).abs() < 2.4, "video arrives at {fps} fps");
        assert!(
            (rtp_fps - 24.0).abs() < 0.01,
            "video RTP advances at {rtp_fps} fps"
        );
        assert!(
            (audio_rate - 48_000.0).abs() < 4_800.0,
            "audio arrives at {audio_rate} samples/s"
        );
        assert_eq!(audio_step, 960.0, "20 ms Opus packets");
    } else {
        // Two seconds of media arrive: 48 frames and 100 packets.
        c.more_media(48, 100).await;
    }
    c.assert_rtp_grids();
    assert!(c.answer.contains("stereo=1"), "stereo Opus: {}", c.answer);
    assert!(c.media.lock().unwrap().keyframes > 0);

    // Heartbeats keep the lease.
    let hb = post(
        &f.app,
        "/wma/session/heartbeat",
        json!({"session_id": c.session_id}),
    )
    .await;
    assert_eq!(hb.json(), json!({"alive": true}));

    c.send(json!({"type": "stop"})).await;
    let ex = c.expect("stream_exhausted").await;
    assert_eq!(ex["reason"], "stopped");
    assert!(ex["chunks"].as_u64().unwrap() >= 2);
    let fm = c.expect("session_metrics").await;
    assert_eq!(fm["final"], true);
    let st = wait_closed(&f, &c.session_id, Duration::from_secs(10)).await;
    assert!(
        matches!(st, None | Some(SessionState::Closed(EndReason::Stopped))),
        "{st:?}"
    );
    assert_not_alive(&f, &c.session_id).await;
    // The executor is free again: a new session is admitted.
    let _c2 = open(&f, "minimax/h3-max/director").await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn video_only_session() {
    video_only(false).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "real-time rates: scripts/serve/check.sh --realtime"]
async fn realtime_video_only_session() {
    video_only(true).await;
}

async fn video_only(realtime: bool) {
    let Some(h264) = h264_or_skip() else { return };
    let f = fixture(Opts {
        h264,
        ..Opts::default()
    })
    .await;
    let mut c = open(&f, "fv/h3-silent/director").await;
    // The offered audio m-line is answered inactive (design §5.3).
    let a = Sdp::parse(&c.answer).unwrap();
    let audio = a
        .media
        .iter()
        .find(|m| m.kind() == MediaKind::Audio)
        .unwrap();
    assert_eq!(audio.direction(), Direction::Inactive, "{}", c.answer);
    c.expect("session_info").await;
    // A 10 s session: 243 frames on H3's 17n+5 grid.
    c.send(json!({"type": "configure", "prompt_version": 1, "prompt": "silent film", "resolution": "480p", "chunk_duration": 10})).await;
    assert_eq!(c.expect("configured").await["chunk_duration"], 10);
    let ch0 = c.expect("chunk").await;
    assert_eq!(
        (
            ch0["generated_frame_count"].as_u64(),
            ch0["requested_duration_seconds"].as_f64()
        ),
        (Some(243), Some(10.0)),
        "{ch0}"
    );
    c.first_video().await;
    if realtime {
        let (fps, rtp_fps, _, _) = c.rates(Duration::from_secs(3)).await;
        eprintln!("video-only: {fps:.2} fps (rtp {rtp_fps:.2})");
        assert!((fps - 24.0).abs() < 2.4, "{fps}");
        assert!((rtp_fps - 24.0).abs() < 0.01, "{rtp_fps}");
    } else {
        c.more_media(48, 0).await;
    }
    c.assert_rtp_grids();
    assert!(
        c.media.lock().unwrap().audio.is_empty(),
        "no audio on a video-only session"
    );
    c.send(json!({"type": "stop"})).await;
    c.expect("stream_exhausted").await;
}

/// A director whose H.264 backend is not in this build (an `auto` encoder
/// on a GPU without NVENC resolves to OpenH264, which images built without
/// the `openh264` feature lack) answers a browser's H.264 + VP8 offer with
/// VP8 and streams. Before, it answered H.264 and the session failed at its
/// first frame ("encoding failed: ... built without the `openh264` feature";
/// seen on an H100 worker).
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn browser_offer_without_a_usable_h264_encoder_gets_vp8() {
    browser_offer_vp8(false).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "real-time rates: scripts/serve/check.sh --realtime"]
async fn realtime_browser_offer_without_a_usable_h264_encoder_gets_vp8() {
    browser_offer_vp8(true).await;
}

async fn browser_offer_vp8(realtime: bool) {
    use fastvideo_media::video::EncoderBackend;
    let f = fixture(Opts {
        h264: EncoderBackend::OpenH264,
        ..Opts::default()
    })
    .await;
    let mut c = open_with(
        &f,
        "fv/h3-silent/director",
        vec![VideoCodec::H264, VideoCodec::Vp8],
    )
    .await;
    c.expect("session_info").await;
    c.send(json!({"type": "configure", "prompt_version": 1, "prompt": "silent film", "resolution": "480p"})).await;
    c.expect("configured").await;
    c.expect("chunk").await;
    c.first_video().await;
    // What was sent: H.264 arrives as Annex-B (start code first), a VP8
    // keyframe carries its start code 9d 01 2a at bytes 3..6.
    let head = c.media.lock().unwrap().first_video.clone().unwrap();
    let h264 = head.starts_with(&[0, 0, 0, 1]) || head.starts_with(&[0, 0, 1]);
    let vp8 = head.len() > 6 && head[3..6] == [0x9d, 0x01, 0x2a];
    let codec = if h264 {
        "H.264"
    } else if vp8 {
        "VP8"
    } else {
        "unknown"
    };
    let want = if EncoderBackend::OpenH264.compiled() {
        "H.264"
    } else {
        "VP8"
    };
    assert_eq!(
        codec,
        want,
        "first video frame {:02x?}",
        &head[..head.len().min(8)]
    );
    if realtime {
        let (fps, _, _, _) = c.rates(Duration::from_secs(3)).await;
        assert!((fps - 24.0).abs() < 2.4, "{codec}: {fps} fps");
    } else {
        c.more_media(48, 0).await;
    }
    c.assert_rtp_grids();
    c.send(json!({"type": "stop"})).await;
    c.expect("stream_exhausted").await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn signalling_rules() {
    let f = fixture(Opts::default()).await;
    // /ice: config servers, strict body, known apps, auth.
    let r = post(
        &f.app,
        "/wma/ice",
        json!({"app_id": "minimax/h3-max/director"}),
    )
    .await;
    assert_eq!(
        (r.status.as_u16(), r.json()),
        (200, json!({"ice_servers": []}))
    );
    let r = post(&f.app, "/wma/ice", json!({})).await;
    assert_eq!(
        (r.status.as_u16(), r.text()),
        (422, "missing field `app_id`".to_string())
    );
    let r = post(
        &f.app,
        "/wma/ice",
        json!({"app_id": "someone/else/director"}),
    )
    .await;
    assert_eq!(r.status, 404);
    assert!(r.json()["error"].is_string());
    let r = send(
        &f.app,
        req(
            "POST",
            "/wma/ice",
            None,
            Some(json!({"app_id": "minimax/h3-max/director"}).to_string()),
        ),
    )
    .await;
    assert_eq!(r.status, 401);
    for path in [
        "/minimax/h3-max/director/ice",
        "/run/minimax/h3-max/director/ice",
    ] {
        let r = post(&f.app, path, json!({})).await;
        assert_eq!(
            (r.status.as_u16(), r.json()),
            (200, json!({"ice_servers": []})),
            "{path}"
        );
    }
    // Through the fal proxy, as `proxyUrl` clients call it.
    for (target, body) in [
        (
            "https://wma.fal.run/ice",
            json!({"app_id": "minimax/h3-max/director"}),
        ),
        ("https://fal.run/minimax/h3-max/director/ice", json!({})),
    ] {
        let mut rq = req("POST", "/fal/proxy", Some(KEY), Some(body.to_string()));
        rq.headers_mut()
            .insert("x-fal-target-url", target.parse().unwrap());
        let r = send(&f.app, rq).await;
        assert_eq!(
            (r.status.as_u16(), r.json()),
            (200, json!({"ice_servers": []})),
            "{target}"
        );
    }
    // /session: strict, typed, known apps.
    let r = post(
        &f.app,
        "/wma/session",
        json!({"app_id": "minimax/h3-max/director", "sdp": "v=0", "type": "answer"}),
    )
    .await;
    assert_eq!(r.status, 422);
    let r = post(
        &f.app,
        "/wma/session",
        json!({"app_id": "minimax/h3-max/director", "type": "offer"}),
    )
    .await;
    assert_eq!(
        (r.status.as_u16(), r.text()),
        (422, "missing field `sdp`".to_string())
    );
    let r = post(
        &f.app,
        "/wma/session",
        json!({"app_id": "minimax/h3-max/director", "sdp": "v=0\r\n", "type": "offer"}),
    )
    .await;
    assert_eq!(r.status, 422, "{}", r.text());
    assert!(r.json()["error"].as_str().unwrap().contains("offer"));
    // A bad offer released the engine session: one real session is admitted,
    // a second is busy (429).
    let c = open(&f, "minimax/h3-max/director").await;
    let host = RtcHost::bind(HostConfig::loopback(true, false))
        .await
        .unwrap();
    let (_p, offer) = host
        .offer(OfferOptions {
            channels: vec!["control".into()],
            ..Default::default()
        })
        .await
        .unwrap();
    let r = post(
        &f.app,
        "/wma/session",
        json!({"app_id": "minimax/h3-max/director", "sdp": offer, "type": "offer"}),
    )
    .await;
    assert_eq!(r.status, 429, "{}", r.text());
    assert!(r.json()["error"].is_string());
    // Heartbeats.
    let hb = post(
        &f.app,
        "/wma/session/heartbeat",
        json!({"session_id": c.session_id}),
    )
    .await;
    assert_eq!(hb.json(), json!({"alive": true}));
    let hb = post(
        &f.app,
        "/wma/session/heartbeat",
        json!({"session_id": "nope"}),
    )
    .await;
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
    let f = fixture(Opts {
        heartbeat_timeout: Duration::from_millis(1500),
        ..Opts::default()
    })
    .await;
    // Beating keeps it alive past the timeout.
    let mut c = open(&f, "minimax/h3-max/director").await;
    c.stop_beats();
    c.expect("session_info").await;
    for _ in 0..4 {
        tokio::time::sleep(Duration::from_millis(600)).await;
        let hb = post(
            &f.app,
            "/wma/session/heartbeat",
            json!({"session_id": c.session_id}),
        )
        .await;
        assert_eq!(hb.json(), json!({"alive": true}));
    }
    // Then silence: Closing(ClientGone) within the timeout (plus a tick).
    let t0 = Instant::now();
    let st = wait_closed(&f, &c.session_id, Duration::from_secs(5)).await;
    assert!(
        matches!(st, None | Some(SessionState::Closed(EndReason::ClientGone))),
        "{st:?}"
    );
    assert!(
        t0.elapsed() >= Duration::from_millis(1200),
        "closed too early: {:?}",
        t0.elapsed()
    );
    assert_not_alive(&f, &c.session_id).await;
    // The engine session was released.
    let _again = open(&f, "minimax/h3-max/director").await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn configure_failures_end_the_session() {
    let f = fixture(Opts::default()).await;
    let mut c = open(&f, "minimax/h3-max/director").await;
    c.expect("session_info").await;
    c.send(json!({"type": "configure", "prompt_version": 1, "prompt": "x", "resolution": "1080p"}))
        .await;
    let e = c.expect("error").await;
    assert_eq!(e["code"], "invalid_input");
    let st = wait_closed(&f, &c.session_id, Duration::from_secs(5)).await;
    assert!(
        matches!(st, None | Some(SessionState::Closed(EndReason::Error(_)))),
        "{st:?}"
    );

    let mut c = open(&f, "minimax/h3-max/director").await;
    c.expect("session_info").await;
    c.send(json!({"type": "configure", "prompt_version": 1, "prompt": "x", "audio_url": "https://example.com/a.wav"})).await;
    assert_eq!(c.expect("error").await["code"], "invalid_initial_audio");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn session_limit_and_runner_sse() {
    let Some(h264) = h264_or_skip() else { return };
    let f = fixture(Opts {
        h264,
        max_session_seconds: Some(2),
        ..Opts::default()
    })
    .await;
    let mut c = open(&f, "minimax/h3-max/director").await;
    let info = c.expect("session_info").await;
    assert_eq!(info["max_session_seconds"], 2);
    c.send(
        json!({"type": "configure", "prompt_version": 1, "prompt": "short", "resolution": "480p"}),
    )
    .await;
    c.expect("configured").await;
    let ex = c.expect("stream_exhausted").await;
    assert_eq!(ex["reason"], "session_limit");
    let st = wait_closed(&f, &c.session_id, Duration::from_secs(10)).await;
    assert!(
        matches!(
            st,
            None | Some(SessionState::Closed(EndReason::SessionLimit))
        ),
        "{st:?}"
    );
    // Gone from the registry: its engine session is released.
    assert_not_alive(&f, &c.session_id).await;

    // Runner side: SSE whose first event is the answer; dropping the
    // response ends the session.
    let host = RtcHost::bind(HostConfig::loopback(true, false))
        .await
        .unwrap();
    let (_pending, offer) = host
        .offer(OfferOptions {
            video: Some(Direction::RecvOnly),
            channels: vec!["control".into()],
            ..Default::default()
        })
        .await
        .unwrap();
    let r = f
        .app
        .clone()
        .oneshot_req(req(
            "POST",
            "/start-session",
            Some(KEY),
            Some(json!({"sdp": offer, "type": "offer", "session_id": "runner-1"}).to_string()),
        ))
        .await;
    assert_eq!(r.status(), 200);
    assert_eq!(r.headers()["content-type"], "text/event-stream");
    use futures::StreamExt;
    let mut body = r.into_body().into_data_stream();
    let first = tokio::time::timeout(T, body.next())
        .await
        .unwrap()
        .unwrap()
        .unwrap();
    let text = String::from_utf8_lossy(&first).into_owned();
    assert!(text.starts_with("data: "), "{text}");
    let ev: Value = serde_json::from_str(text.trim_start_matches("data: ").trim()).unwrap();
    assert_eq!(
        (ev["type"].as_str(), ev["session_id"].as_str()),
        (Some("answer"), Some("runner-1"))
    );
    assert!(ev["sdp"].as_str().unwrap().contains("a=end-of-candidates"));
    // Runner sessions are not heartbeat sessions.
    let hb = post(
        &f.app,
        "/wma/session/heartbeat",
        json!({"session_id": "runner-1"}),
    )
    .await;
    assert_eq!(hb.json(), json!({"alive": false}));
    let h = f.svc.session("runner-1").expect("registered");
    let mut closed = h.closed();
    drop(body);
    tokio::time::timeout(Duration::from_secs(5), closed.wait_for(|c| *c))
        .await
        .unwrap()
        .unwrap();
    assert_eq!(h.state(), SessionState::Closed(EndReason::ClientGone));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn late_chunks_report_deadline_missed() {
    late_chunks(false).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "real-time rates: scripts/serve/check.sh --realtime"]
async fn realtime_late_chunks_report_deadline_missed() {
    late_chunks(true).await;
}

async fn late_chunks(realtime: bool) {
    let Some(h264) = h264_or_skip() else { return };
    // Builds slower than real time: each 5 s chunk takes ~7 s.
    let f = fixture(Opts {
        h264,
        step: Duration::from_millis(870),
        ..Opts::default()
    })
    .await;
    let mut c = open(&f, "minimax/h3-max/director").await;
    c.expect("session_info").await;
    c.send(
        json!({"type": "configure", "prompt_version": 1, "prompt": "slow", "resolution": "480p"}),
    )
    .await;
    c.expect("configured").await;
    let d = c.expect("deadline_missed").await;
    assert_eq!(d["chunk_index"], 1);
    assert_eq!(d["behavior"], "freeze_video_and_silence_audio_until_ready");
    assert!(d["late_by_seconds"].as_f64().unwrap() > 0.5, "{d}");
    // Audio kept flowing (silence) through the underrun.
    if realtime {
        let (_, _, audio_rate, _) = c.rates(Duration::from_secs(1)).await;
        assert!(audio_rate > 40_000.0, "{audio_rate}");
    } else {
        c.more_media(0, 50).await;
    }
    c.assert_rtp_grids();
    c.send(json!({"type": "stop"})).await;
    c.expect("stream_exhausted").await;
}

/// The `chunk` messages until one satisfies `done` (other messages are
/// collected too, by type).
async fn chunks_until(
    c: &mut Client,
    mut done: impl FnMut(&Value) -> bool,
) -> (Vec<Value>, Vec<Value>) {
    let deadline = tokio::time::Instant::now() + T;
    let (mut chunks, mut others) = (Vec::new(), Vec::new());
    loop {
        let m = tokio::time::timeout_at(deadline, c.msgs.recv())
            .await
            .expect("timed out")
            .expect("closed");
        if m["type"] == "error" {
            panic!("unexpected error: {m}");
        }
        if m["type"] == "chunk" {
            let stop = done(&m);
            chunks.push(m);
            if stop {
                return (chunks, others);
            }
        } else {
            others.push(m);
        }
    }
}

/// A causal model (fake LongLive) streams one continuous rollout: chunks of
/// 4 blocks (48 frames) at 16 fps, video only; a prompt update reaches the
/// engine once and is applied at the next block with one KV re-cache;
/// end images are refused. The `realtime_` twin also checks the delivery
/// and generation rates and that nothing underran.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn causal_session_streams_and_recaches_once_per_switch() {
    causal_session(false).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "real-time rates: scripts/serve/check.sh --realtime"]
async fn realtime_causal_session_streams_and_recaches_once_per_switch() {
    causal_session(true).await;
}

async fn causal_session(realtime: bool) {
    let Some(h264) = h264_or_skip() else { return };
    let f = fixture(Opts {
        h264,
        ..Opts::default()
    })
    .await;
    let mut c = open(&f, "fastvideo/longlive/director").await;
    let a = Sdp::parse(&c.answer).unwrap();
    let audio = a
        .media
        .iter()
        .find(|m| m.kind() == MediaKind::Audio)
        .unwrap();
    assert_eq!(
        audio.direction(),
        Direction::Inactive,
        "video only: {}",
        c.answer
    );

    let info = c.expect("session_info").await;
    assert_eq!(info["app"], "fastvideo-longlive-director");
    assert_eq!(info["fps"], 16);
    assert_eq!(info["resolutions"], json!(["480p"]));
    assert_eq!(info["aspect_ratios"], json!(["16:9"]));
    assert_eq!(info["chunk_seconds"], 3);
    assert_eq!(
        info["continuation_context_frames"], 48,
        "the KV window (12 latent frames)"
    );
    assert_eq!(info["causal"]["prompt_switch"], "recache");
    assert_eq!(info["causal"]["block_frames"], 12);
    assert!(
        info.get("chunk_duration_options").is_none(),
        "no chunk length to choose: {info}"
    );

    // `chunk_duration` is ignored: the director chunk stays 4 blocks (3 s).
    c.send(json!({"type": "configure", "prompt_version": 1, "prompt": "A lighthouse at dusk", "seed": 7, "chunk_duration": 10})).await;
    let cfgd = c.expect("configured").await;
    assert_eq!(cfgd["chunk_duration"], 3);
    assert_eq!(
        (cfgd["resolution"].as_str(), cfgd["aspect_ratio"].as_str()),
        (Some("480p"), Some("16:9"))
    );
    let (chunks, _) = chunks_until(&mut c, |m| m["chunk_index"] == 1).await;
    for (i, ch) in chunks.iter().enumerate() {
        assert_eq!(ch["chunk_index"], i as u64);
        assert_eq!(ch["prompt_version"], 1);
        assert_eq!(
            (
                ch["generated_frame_count"].as_u64(),
                ch["presented_frame_count"].as_u64()
            ),
            (Some(48), Some(48)),
            "{ch}"
        );
        assert_eq!(ch["trimmed_context_frames"], 0);
        assert_eq!(ch["playback_seconds"], 3.0);
        assert_eq!(ch["causal"]["blocks"], 4);
        assert_eq!(ch["causal"]["recaches"], 0);
        if realtime {
            assert!(
                ch["causal"]["generation_fps"].as_f64().unwrap() > 16.0,
                "faster than real time: {ch}"
            );
        }
    }
    c.first_video().await;
    if realtime {
        let (fps, rtp_fps, _, _) = c.rates(Duration::from_secs(3)).await;
        eprintln!("causal: {fps:.2} fps (rtp {rtp_fps:.2})");
        assert!((fps - 16.0).abs() < 1.6, "video arrives at {fps} fps");
        assert!((rtp_fps - 16.0).abs() < 0.01, "{rtp_fps}");
    } else {
        // Two seconds of video arrive, on the 16 fps RTP grid.
        c.more_media(32, 0).await;
    }
    c.assert_rtp_grids_at(16);
    assert!(c.media.lock().unwrap().audio.is_empty());
    // Paced: the rollout stays about `causal_lead_seconds` (2 s) ahead.
    let last = chunks.last().unwrap();
    assert!(
        last["buffer_depth_seconds"].as_f64().unwrap() < 4.5,
        "{last}"
    );

    // A direction change: one prompt to the engine, applied at the next
    // block, with one re-cache.
    let before = f.engine_prompts.lock().unwrap().len();
    assert_eq!(f.engine_prompts.lock().unwrap()[0], "A lighthouse at dusk");
    c.send(json!({"type": "prompt", "prompt_version": 2, "prompt": "The storm arrives"}))
        .await;
    assert_eq!(c.expect("prompt_pending").await["prompt_version"], 2);
    let (chunks, others) = chunks_until(&mut c, |m| m["prompt_version"] == 2).await;
    assert!(
        others
            .iter()
            .any(|m| m["type"] == "prompt_applied" && m["prompt_version"] == 2),
        "{others:?}"
    );
    let (more, _) = chunks_until(&mut c, |_| true).await;
    let recaches: u64 = chunks
        .iter()
        .chain(&more)
        .map(|m| m["causal"]["recaches"].as_u64().unwrap())
        .sum();
    assert_eq!(recaches, 1, "one re-cache per switch: {chunks:?} {more:?}");
    {
        let p = f.engine_prompts.lock().unwrap();
        assert_eq!(p.len(), before + 1, "{p:?}");
        assert_eq!(p.last().unwrap(), "A lighthouse at dusk The storm arrives");
    }
    // Text only: end images are refused, the session goes on.
    c.send(json!({"type": "prompt", "prompt_version": 3, "end_image_url": "https://example.com/e.png"})).await;
    let rej = c.expect("prompt_rejected").await;
    assert_eq!(
        (rej["prompt_version"].as_u64(), rej["reason"].as_str()),
        (Some(3), Some("invalid_image"))
    );

    c.send(json!({"type": "stop"})).await;
    c.expect("stream_exhausted").await;
    let fm = c.expect("session_metrics").await;
    assert_eq!(fm["final"], true);
    if realtime {
        // A loaded host generates slower than real time and underruns.
        assert_eq!(fm["gauges"]["underruns"], 0, "{fm}");
    }
    assert_eq!(fm["causal"]["recaches"], 1, "{fm}");
    let st = wait_closed(&f, &c.session_id, Duration::from_secs(10)).await;
    assert!(
        matches!(st, None | Some(SessionState::Closed(EndReason::Stopped))),
        "{st:?}"
    );
    // The causal lease is released (the session left the registry): a
    // clip session is admitted next.
    assert_not_alive(&f, &c.session_id).await;
    let mut c2 = open(&f, "minimax/h3-max/director").await;
    c2.expect("session_info").await;
}

/// Causal refusals end the session with clear codes (image conditioning);
/// a causal model without re-cache (fake SF-Wan) keeps its cache.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn causal_refusals_and_keep_policy() {
    let Some(h264) = h264_or_skip() else { return };
    let f = fixture(Opts {
        h264,
        ..Opts::default()
    })
    .await;
    let mut c = open(&f, "fastvideo/longlive/director").await;
    c.expect("session_info").await;
    c.send(json!({"type": "configure", "prompt_version": 1, "prompt": "x", "image_url": "https://example.com/a.png"})).await;
    let e = c.expect("error").await;
    assert_eq!(e["code"], "invalid_initial_image");
    assert!(
        e["error"].as_str().unwrap().contains("text-to-video only"),
        "{e}"
    );
    let st = wait_closed(&f, &c.session_id, Duration::from_secs(5)).await;
    assert!(
        matches!(st, None | Some(SessionState::Closed(EndReason::Error(_)))),
        "{st:?}"
    );
    // Gone from the registry: its engine session is released.
    assert_not_alive(&f, &c.session_id).await;

    let mut c = open(&f, "fastvideo/fake-sfwan/director").await;
    let info = c.expect("session_info").await;
    assert_eq!(info["causal"]["prompt_switch"], "unknown");
    assert_eq!(info["continuation_context_frames"], 12);
    c.send(json!({"type": "configure", "prompt_version": 1, "prompt": "a river"}))
        .await;
    c.expect("configured").await;
    c.expect("chunk").await;
    c.send(json!({"type": "prompt", "prompt_version": 2, "prompt": "at night"}))
        .await;
    let (chunks, _) = chunks_until(&mut c, |m| m["prompt_version"] == 2).await;
    assert!(
        chunks.iter().all(|m| m["causal"]["recaches"] == 0),
        "{chunks:?}"
    );
    c.send(json!({"type": "stop"})).await;
    c.expect("stream_exhausted").await;
}

/// The catalog marks causal directors and the LongLive licence; the form
/// offers 480p and 16:9 only.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn causal_catalog_and_form() {
    let f = fixture(Opts::default()).await;
    let r = send(&f.app, req("GET", "/fal/schema", None, None)).await;
    let cat = r.json();
    let app = cat["apps"]
        .as_array()
        .unwrap()
        .iter()
        .find(|a| a["id"] == "fastvideo/longlive")
        .unwrap()
        .clone();
    assert_eq!(
        (app["director"].as_bool(), app["director_mode"].as_str()),
        (Some(true), Some("causal")),
        "{app}"
    );
    assert!(
        app["licence"].as_str().unwrap().contains("non-commercial"),
        "{app}"
    );
    let h3 = cat["apps"]
        .as_array()
        .unwrap()
        .iter()
        .find(|a| a["id"] == "minimax/h3-max")
        .unwrap()
        .clone();
    assert_eq!(h3["director"], true);
    assert!(h3.get("director_mode").is_none());
    let form = send(
        &f.app,
        req("GET", "/fal/schema/fastvideo/longlive/director", None, None),
    )
    .await
    .json();
    assert_eq!(form["properties"]["resolution"]["enum"], json!(["480p"]));
    assert_eq!(form["properties"]["resolution"]["default"], "480p");
    assert_eq!(
        form["properties"]["aspect_ratio"]["enum"],
        json!(["auto", "16:9"])
    );
    assert!(
        form["properties"].get("chunk_duration").is_none(),
        "a causal model has no chunk length: {form}"
    );
    assert!(
        form["x-fv-licence"]
            .as_str()
            .unwrap()
            .contains("non-commercial"),
        "{form}"
    );
}

/// `oneshot` returning the raw response (streaming body).
trait OneshotReq {
    async fn oneshot_req(
        self,
        r: axum::http::Request<axum::body::Body>,
    ) -> axum::response::Response;
}

impl OneshotReq for axum::Router {
    async fn oneshot_req(
        self,
        r: axum::http::Request<axum::body::Body>,
    ) -> axum::response::Response {
        use tower::ServiceExt;
        self.oneshot(r).await.unwrap()
    }
}

// ---- prompt upsampling (docs/serve/prompt-upsampling.md) -------------------------

/// A fake showrunner LLM: call `n` answers beat `n` (Mara on her tower),
/// after 50 ms; a direction containing "fail" gets a 503.
struct ShowLlm {
    n: std::sync::atomic::AtomicUsize,
    users: Mutex<Vec<String>>,
    /// The images of each request (vision).
    images: Mutex<Vec<Vec<String>>>,
}

#[async_trait::async_trait]
impl fastvideo_upsampler::ChatBackend for ShowLlm {
    async fn complete(
        &self,
        req: &fastvideo_upsampler::ChatRequest,
    ) -> Result<String, fastvideo_upsampler::UpsampleError> {
        self.users.lock().unwrap().push(req.user.clone());
        self.images.lock().unwrap().push(req.images.clone());
        tokio::time::sleep(Duration::from_millis(50)).await;
        if req
            .user
            .rsplit("NEW DIRECTION:")
            .next()
            .is_some_and(|d| d.contains("fail"))
        {
            return Err(fastvideo_upsampler::UpsampleError::Status {
                status: 503,
                body: "down".into(),
            });
        }
        let n = self.n.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        Ok(json!({
            "beat_summary": format!("beat {n}"), "entity_ids": ["tower", "mara"], "new_entities": [],
            "action": format!("Mara step {n}."), "camera": "c", "lighting": "l", "audio": "a", "dialogue": [],
        })
        .to_string())
    }
}

fn show_llm() -> (Arc<ShowLlm>, Arc<fastvideo_upsampler::Service>) {
    show_llm_with(fastvideo_upsampler::Settings::default())
}

fn show_llm_with(
    s: fastvideo_upsampler::Settings,
) -> (Arc<ShowLlm>, Arc<fastvideo_upsampler::Service>) {
    let llm = Arc::new(ShowLlm {
        n: Default::default(),
        users: Mutex::new(Vec::new()),
        images: Mutex::new(Vec::new()),
    });
    let settings = fastvideo_upsampler::Settings {
        model: "fake".into(),
        hedge: false,
        timeout: Duration::from_secs(5),
        ..s
    };
    (
        llm.clone(),
        Arc::new(fastvideo_upsampler::Service::new(llm, settings)),
    )
}

/// The H3 segment prompt the fake LLM's beat `n` assembles to.
fn h3_segment(n: usize) -> String {
    format!(
        "integrated_multimodal_description: a white lighthouse on black rocks. Mara: a wiry woman in a yellow oilskin. Mara step {n}. c. l. 35mm film.\n\
         overall_soundscape: a No spoken words.\nnon_diegetic_music: None."
    )
}

fn world_configure(prompt: &str) -> Value {
    json!({
        "type": "configure", "prompt_version": 1, "prompt": prompt, "upsample": true, "style": "35mm film.",
        "entities": [
            {"id": "mara", "kind": "character", "name": "Mara", "descriptor": "a wiry woman in a yellow oilskin"},
            {"id": "tower", "kind": "location", "name": "Lighthouse", "descriptor": "a white lighthouse on black rocks"},
        ],
    })
}

fn segments(msgs: &[Value]) -> Vec<Value> {
    msgs.iter()
        .filter(|m| m["type"] == "segment_expanded")
        .cloned()
        .collect()
}

/// A clip session with `upsample` on H3: every chunk's prompt is the
/// showrunner's segment in H3's Context-IR layout (location, cast
/// descriptors, action, camera, lighting, style; soundscape; music),
/// reported in `segment_expanded`; a direction becomes the LLM's NEW
/// DIRECTION once; a failed call falls back to the direction's text (never
/// the premise) and the stream goes on.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn upsampled_clip_session() {
    let Some(h264) = h264_or_skip() else { return };
    let (llm, svc) = show_llm();
    let f = fixture(Opts {
        h264,
        upsampler: Some(svc),
        ..Opts::default()
    })
    .await;
    let mut c = open(&f, "minimax/h3-max/director").await;
    let info = c.expect("session_info").await;
    assert_eq!(info["prompt_expander"], "fast");
    assert_eq!(
        info["upsample"],
        json!({"available": true, "default": false, "max_prompt_chars": 3200, "max_prompt_tokens": 1024, "prompt_limit_unit": "chars",
               "tokenizer": null, "layout": "h3", "context_tokens": 28000, "deck_size": 6, "vision": false,
               "server_message_types": ["segment_expanded"]})
    );
    c.send(world_configure("A lighthouse keeper on a stormy island."))
        .await;
    let cfgd = c.expect("configured").await;
    assert_eq!(cfgd["upsample"], true);
    assert_eq!(
        (
            cfgd["upsample_premise"]["chars"].as_u64(),
            cfgd["upsample_premise"]["digest"].is_null()
        ),
        (Some(39), true),
        "{cfgd}"
    );
    let (_, others) = chunks_until(&mut c, |m| m["chunk_index"] == 1).await;
    let s = segments(&others);
    assert!(s.len() >= 2, "{others:?}");
    let want0 = h3_segment(0);
    assert_eq!(
        (
            s[0]["chunk_index"].as_u64(),
            s[0]["segment_index"].as_u64(),
            s[0]["prompt"].as_str()
        ),
        (Some(0), Some(0), Some(want0.as_str())),
        "{}",
        s[0]
    );
    assert_eq!(
        (s[0]["prompt_unit"].as_str(), s[0]["prompt_limit"].as_u64()),
        (Some("chars"), Some(3200))
    );
    assert!(s[0]["planned"].as_u64().is_some(), "{}", s[0]);
    assert_eq!(
        (
            s[0]["fallback"].as_bool(),
            s[0]["direction_id"].is_null(),
            s[0]["summary"].as_str()
        ),
        (Some(false), true, Some("beat 0"))
    );
    assert_eq!(
        (s[1]["chunk_index"].as_u64(), s[1]["segment_index"].as_u64()),
        (Some(1), Some(1))
    );
    {
        let u = llm.users.lock().unwrap();
        assert!(u[0].starts_with("WORLD BIBLE:\nA lighthouse keeper on a stormy island.\n\nESTABLISHED ENTITIES:\n- mara (character, \"Mara\")\n- tower (location, \"Lighthouse\")"), "{}", u[0]);
        assert!(u[0].ends_with("NEW DIRECTION:\n(none, advance the story one beat)"));
        assert!(u[1].contains("[0] beat 0\n") && u[1].contains(&want0));
    }

    // A direction: expanded for the next undispatched chunk.
    c.send(json!({"type": "prompt", "prompt_version": 2, "prompt": "The storm arrives."}))
        .await;
    let (_, others) = chunks_until(&mut c, |m| m["prompt_version"] == 2).await;
    let s = segments(&others);
    let d = s
        .iter()
        .find(|m| m["prompt_version"] == 2)
        .unwrap_or_else(|| panic!("{others:?}"));
    assert_eq!(
        (d["direction_id"].as_str(), d["fallback"].as_bool()),
        (Some("2"), Some(false)),
        "{d}"
    );
    assert!(llm
        .users
        .lock()
        .unwrap()
        .iter()
        .any(|u| u.ends_with("NEW DIRECTION:\nThe storm arrives.")));

    // A direction whose call fails: the direction's text, reported.
    c.send(json!({"type": "prompt", "prompt_version": 3, "prompt": "Everything should fail."}))
        .await;
    let (_, others) = chunks_until(&mut c, |m| m["prompt_version"] == 3).await;
    let fb = segments(&others)
        .into_iter()
        .find(|m| m["prompt_version"] == 3)
        .unwrap_or_else(|| panic!("{others:?}"));
    assert_eq!(fb["fallback"], true, "{fb}");
    assert_eq!(fb["prompt"], "Everything should fail.");
    assert_eq!(fb["error"], "upsampler LLM 503: down");
    assert!(fb["segment_index"].is_null());
    // ...and the next chunk is upsampled again.
    let (_, others) = chunks_until(&mut c, |_| true).await;
    let (_, more) = chunks_until(&mut c, |_| true).await;
    assert!(
        segments(&others)
            .iter()
            .chain(&segments(&more))
            .any(|m| m["fallback"] == false),
        "{others:?} {more:?}"
    );

    c.send(json!({"type": "stop"})).await;
    c.expect("stream_exhausted").await;
}

/// Without `[upsampler]`, `upsample: true` is refused softly: `configured`
/// says so and the session runs on the raw prompts.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn upsample_without_an_upsampler() {
    let Some(h264) = h264_or_skip() else { return };
    let f = fixture(Opts {
        h264,
        ..Opts::default()
    })
    .await;
    let mut c = open(&f, "minimax/h3-max/director").await;
    let info = c.expect("session_info").await;
    assert_eq!(info["prompt_expander"], "none");
    assert!(info.get("upsample").is_none());
    c.send(world_configure("A lighthouse keeper.")).await;
    let cfgd = c.expect("configured").await;
    assert_eq!(cfgd["upsample"], false);
    assert!(cfgd["upsample_error"]
        .as_str()
        .unwrap()
        .contains("[upsampler]"));
    let (_, others) = chunks_until(&mut c, |m| m["chunk_index"] == 0).await;
    assert!(segments(&others).is_empty());
    // A strict schema still: unknown entity fields and kinds are refused.
    let e = fastvideo_fal::director::messages::parse(
        &json!({"type": "configure", "prompt_version": 1, "prompt": "p", "entities": [{"id": "a", "kind": "animal", "name": "A", "descriptor": "d"}]}).to_string(),
    )
    .unwrap_err();
    assert!(e.error.contains("animal"), "{e:?}");
    let e = fastvideo_fal::director::messages::parse(
        &json!({"type": "configure", "prompt_version": 1, "prompt": "p", "entities": [{"id": "a", "kind": "prop", "name": "A", "descriptor": "d"}, {"id": "a", "kind": "prop", "name": "B", "descriptor": "d"}]}).to_string(),
    )
    .unwrap_err();
    assert!(e.error.contains("not unique"), "{e:?}");
    c.send(json!({"type": "stop"})).await;
    c.expect("stream_exhausted").await;
}

/// A causal session (fake LongLive) with `upsample`: the rollout starts on
/// the first segment's prompt, a new segment goes to the engine every
/// director chunk, and a direction reaches it at the next block.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn upsampled_causal_session() {
    let Some(h264) = h264_or_skip() else { return };
    let (llm, svc) = show_llm();
    let f = fixture(Opts {
        h264,
        upsampler: Some(svc),
        ..Opts::default()
    })
    .await;
    let mut c = open(&f, "fastvideo/longlive/director").await;
    let info = c.expect("session_info").await;
    assert_eq!(info["upsample"]["max_prompt_chars"], 1800, "umT5: {info}");
    c.send(json!({"type": "configure", "prompt_version": 1, "prompt": "A lighthouse at dusk", "upsample": true, "memory": 2})).await;
    assert_eq!(c.expect("configured").await["upsample"], true);
    let (_, others) = chunks_until(&mut c, |m| m["chunk_index"] == 2).await;
    let s = segments(&others);
    assert!(
        s.len() >= 2,
        "a new segment every director chunk: {others:?}"
    );
    assert_eq!(
        (s[0]["first_block"].as_u64(), s[0]["chunk_index"].is_null()),
        (Some(0), true)
    );
    // No cast registered: the ids are dropped, the prompt is the beat.
    assert_eq!(
        s[0]["prompt"],
        "Mara step 0. Camera: c Lighting: l Sound: a No spoken words."
    );
    {
        let p = f.engine_prompts.lock().unwrap();
        assert_eq!(
            p[0],
            "Mara step 0. Camera: c Lighting: l Sound: a No spoken words."
        );
        assert!(p.len() >= 2, "{p:?}");
    }
    c.send(json!({"type": "prompt", "prompt_version": 2, "prompt": "The storm arrives."}))
        .await;
    let (_, others) = chunks_until(&mut c, |m| m["prompt_version"] == 2).await;
    assert!(
        others
            .iter()
            .any(|m| m["type"] == "prompt_applied" && m["prompt_version"] == 2),
        "{others:?}"
    );
    let d = segments(&others)
        .into_iter()
        .find(|m| m["prompt_version"] == 2)
        .unwrap_or_else(|| panic!("{others:?}"));
    assert_eq!(d["direction_id"], "2");
    {
        // memory 2: at most the last two segments as context.
        let u = llm.users.lock().unwrap();
        let last = u
            .iter()
            .rfind(|u| u.ends_with("NEW DIRECTION:\nThe storm arrives."))
            .unwrap();
        assert_eq!(last.matches("\n[").count(), 2, "{last}");
    }
    c.send(json!({"type": "stop"})).await;
    c.expect("stream_exhausted").await;
}

/// Collects control messages (of every type) until one satisfies `done`.
async fn msgs_until(c: &mut Client, mut done: impl FnMut(&Value) -> bool) -> Vec<Value> {
    let deadline = tokio::time::Instant::now() + T;
    let mut out = Vec::new();
    loop {
        let m = tokio::time::timeout_at(deadline, c.msgs.recv())
            .await
            .expect("timed out")
            .expect("closed");
        assert_ne!(m["type"], "error", "{m}");
        let stop = done(&m);
        out.push(m);
        if stop {
            return out;
        }
    }
}

fn pos(msgs: &[Value], f: impl Fn(&Value) -> bool) -> usize {
    msgs.iter()
        .position(f)
        .unwrap_or_else(|| panic!("{msgs:?}"))
}

/// fal's prompt deck: segments are planned ahead (`planned` in
/// `segment_expanded`); `replan: false` appends after the planned
/// segments; `replan: true` busts them and applies at the next chunk;
/// `prompt_pending` → `prompt_applied` → the chunk, in that order.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn upsampled_prompt_deck_plan_bust_append() {
    let Some(h264) = h264_or_skip() else { return };
    let (llm, svc) = show_llm_with(fastvideo_upsampler::Settings {
        deck_size: 3,
        ..Default::default()
    });
    let f = fixture(Opts {
        h264,
        upsampler: Some(svc),
        ..Opts::default()
    })
    .await;
    let mut c = open(&f, "minimax/h3-max/director").await;
    assert_eq!(c.expect("session_info").await["upsample"]["deck_size"], 3);
    c.send(world_configure("A lighthouse keeper on a stormy island."))
        .await;
    c.expect("configured").await;
    let first = msgs_until(&mut c, |m| m["type"] == "chunk" && m["chunk_index"] == 1).await;
    let s = segments(&first);
    // Planned ahead: by the second chunk the deck holds the next ones.
    assert!(
        s.iter().any(|m| m["planned"].as_u64().unwrap_or(0) >= 1),
        "{s:?}"
    );
    // Wait until the deck is full (3 planned ahead of the chunk building).
    let deadline = Instant::now() + Duration::from_secs(20);
    while llm.users.lock().unwrap().len() < 5 && Instant::now() < deadline {
        tokio::time::sleep(Duration::from_millis(20)).await;
    }

    // Append: lands after the planned segments, not at the next chunk.
    c.send(
        json!({"type": "prompt", "prompt_version": 2, "prompt": "A gull lands.", "replan": false}),
    )
    .await;
    let m = msgs_until(&mut c, |m| m["type"] == "chunk" && m["prompt_version"] == 2).await;
    let pending = pos(&m, |x| {
        x["type"] == "prompt_pending" && x["prompt_version"] == 2
    });
    let applied = pos(&m, |x| {
        x["type"] == "prompt_applied" && x["prompt_version"] == 2
    });
    let seg = pos(&m, |x| {
        x["type"] == "segment_expanded" && x["prompt_version"] == 2
    });
    let chunk = m.len() - 1;
    assert!(pending < applied && applied < chunk && seg < chunk, "{m:?}");
    let seg = &m[seg];
    assert_eq!(seg["direction_id"], "2");
    let next_after_append = first.iter().filter(|x| x["type"] == "chunk").count() as u64;
    assert!(
        seg["chunk_index"].as_u64().unwrap() > next_after_append,
        "behind the planned ones: {seg}"
    );
    let between: Vec<&Value> = m.iter().filter(|x| x["type"] == "chunk").collect();
    assert!(
        between.len() >= 2,
        "at least one planned chunk played before the appended direction: {m:?}"
    );
    {
        let u = llm.users.lock().unwrap();
        let gull = u
            .iter()
            .find(|u| u.ends_with("NEW DIRECTION:\nA gull lands."))
            .unwrap();
        // Written after the planned beats it follows.
        assert!(gull.matches("\n[").count() >= 3, "{gull}");
    }

    // Replan: busts the deck; the next chunk carries it.
    let calls = llm.users.lock().unwrap().len();
    c.send(json!({"type": "prompt", "prompt_version": 3, "prompt": "The storm breaks."}))
        .await;
    let m = msgs_until(&mut c, |m| m["type"] == "chunk" && m["prompt_version"] == 3).await;
    let pending = pos(&m, |x| {
        x["type"] == "prompt_pending" && x["prompt_version"] == 3
    });
    let applied = pos(&m, |x| {
        x["type"] == "prompt_applied" && x["prompt_version"] == 3
    });
    assert!(pending < applied, "{m:?}");
    let seg = m
        .iter()
        .find(|x| x["type"] == "segment_expanded" && x["prompt_version"] == 3)
        .unwrap();
    assert_eq!(
        (seg["direction_id"].as_str(), seg["fallback"].as_bool()),
        (Some("3"), Some(false))
    );
    assert!(
        m.iter().filter(|x| x["type"] == "chunk").count() <= 2,
        "at most the chunk already building came first: {m:?}"
    );
    {
        let u = llm.users.lock().unwrap();
        let storm = u[calls..]
            .iter()
            .find(|u| u.ends_with("NEW DIRECTION:\nThe storm breaks."))
            .unwrap();
        // Written on the executed segments only: no discarded beat in it.
        let executed = seg["segment_index"].as_u64().unwrap();
        assert!(!storm.contains(&format!("\n[{executed}] ")), "{storm}");
    }
    c.send(json!({"type": "stop"})).await;
    c.expect("stream_exhausted").await;
}

/// A tiny PNG as a data URI.
fn png_data_uri() -> String {
    use base64::Engine as _;
    let mut b = std::io::Cursor::new(Vec::new());
    image::RgbImage::from_pixel(64, 36, image::Rgb([10, 120, 200]))
        .write_to(&mut b, image::ImageFormat::Png)
        .unwrap();
    format!(
        "data:image/png;base64,{}",
        base64::engine::general_purpose::STANDARD.encode(b.into_inner())
    )
}

/// Vision (`[upsampler] vision`): the opening image goes to the LLM as an
/// image (a JPEG data URL) with the first expansion only; the others carry
/// none. Without vision, one line says a first frame exists.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn upsampled_opening_is_grounded_in_the_image() {
    let Some(h264) = h264_or_skip() else { return };
    for vision in [true, false] {
        let (llm, svc) = show_llm_with(fastvideo_upsampler::Settings {
            vision,
            ..Default::default()
        });
        let f = fixture(Opts {
            h264,
            upsampler: Some(svc),
            ..Opts::default()
        })
        .await;
        let mut c = open(&f, "minimax/h3-max/director").await;
        assert_eq!(c.expect("session_info").await["upsample"]["vision"], vision);
        let mut cfg = world_configure("A lighthouse keeper.");
        cfg["image_url"] = json!(png_data_uri());
        c.send(cfg).await;
        c.expect("configured").await;
        chunks_until(&mut c, |m| m["chunk_index"] == 1).await;
        let (u, im) = (
            llm.users.lock().unwrap().clone(),
            llm.images.lock().unwrap().clone(),
        );
        assert!(u.len() >= 2);
        if vision {
            assert_eq!(im[0].len(), 1, "{im:?}");
            assert!(im[0][0].starts_with("data:image/jpeg;base64,"));
            assert!(
                u[0].contains("FIRST FRAME: this segment starts from a given image (attached)"),
                "{}",
                u[0]
            );
        } else {
            assert!(im[0].is_empty());
            assert!(
                u[0].contains(
                    "FIRST FRAME: this segment starts from a given image (not shown to you)"
                ),
                "{}",
                u[0]
            );
        }
        assert!(
            im[1..].iter().all(Vec::is_empty) && !u[1].contains("FRAME:"),
            "only the opening: {im:?}"
        );
        c.send(json!({"type": "stop"})).await;
        c.expect("stream_exhausted").await;
    }
}

/// A 50 000-character premise (fal's limit) is accepted, never sent to the
/// engine: every chunk gets the assembled segment.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn upsampled_long_premise_never_reaches_the_engine() {
    let Some(h264) = h264_or_skip() else { return };
    let (llm, svc) = show_llm();
    let f = fixture(Opts {
        h264,
        upsampler: Some(svc),
        ..Opts::default()
    })
    .await;
    let mut c = open(&f, "minimax/h3-max/director").await;
    c.expect("session_info").await;
    let unit = "Mara keeps the light on a stormy island and logs every ship that passes. ";
    let mut premise = unit.repeat(50_000 / unit.len() + 1);
    premise.truncate(50_000);
    c.send(world_configure(&premise)).await;
    let cfgd = c.expect("configured").await;
    assert_eq!(
        (
            cfgd["upsample_premise"]["chars"].as_u64(),
            cfgd["upsample_premise"]["digest"].is_null()
        ),
        (Some(50_000), true),
        "{cfgd}"
    );
    let (_, others) = chunks_until(&mut c, |m| m["chunk_index"] == 1).await;
    let s = segments(&others);
    assert_eq!(s[0]["prompt"].as_str(), Some(h3_segment(0).as_str()));
    assert!(llm.users.lock().unwrap()[0].starts_with(&format!("WORLD BIBLE:\n{premise}\n\n")));
    // Each chunk is built with its segment prompt (`segment_expanded.prompt`).
    assert!(
        s.iter()
            .all(|m| m["prompt"].as_str().unwrap().chars().count() < 1000),
        "the premise never reaches the engine: {s:?}"
    );
    c.send(json!({"type": "stop"})).await;
    c.expect("stream_exhausted").await;
}

/// The server's default (`buffer_chunks = 0`): generation never waits for
/// playout. Chunk N+1 is dispatched as soon as chunk N is built, so while
/// playout stands still the video queued for it keeps growing. With a hold
/// of one chunk (the earlier default) it could never exceed the playing
/// chunk plus one queued plus the new one (15 s of 5 s chunks).
///
/// Playout runs on a manual clock, held at its start until the queue is
/// that deep, then let go in real time. On wall time the queue grows only by
/// what building beats real time by, which is the machine's speed and not
/// the property: an instrumented (coverage) build on a shared runner builds
/// a 5 s chunk in 2-6 s, and its queue peaked at 15.7 s by chunk 11. The
/// playout clock is untouched (the RTP grid).
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn generation_never_waits_for_playout() {
    let Some(h264) = h264_or_skip() else { return };
    let clock = Arc::new(ManualClock::new(0.0));
    let f = fixture(Opts {
        h264,
        buffer_chunks: 0,
        buffer_max_bytes: 4 << 30,
        playout_clock: Some(clock.clone()),
        ..Opts::default()
    })
    .await;
    let mut c = open(&f, "minimax/h3-max/director").await;
    c.expect("session_info").await;
    c.send(json!({"type": "configure", "prompt_version": 1, "prompt": "A lighthouse at dusk", "resolution": "480p"})).await;
    c.expect("configured").await;
    let t0 = Instant::now();
    let depth_of = |m: &Value| m["buffer_depth_seconds"].as_f64().unwrap();
    let deadline = tokio::time::Instant::now() + T;
    let mut chunks: Vec<Value> = Vec::new();
    while !chunks
        .last()
        .is_some_and(|m| m["chunk_index"] == 11 || depth_of(m) > 16.0)
    {
        match tokio::time::timeout_at(deadline, c.msgs.recv()).await {
            Ok(Some(m)) if m["type"] == "chunk" => chunks.push(m),
            Ok(Some(m)) => assert_ne!(m["type"], "error", "{m}"),
            Ok(None) => panic!("the control channel closed"),
            Err(_) => panic!(
                "with playout held, generation stopped after {} chunks: it waits for playout",
                chunks.len()
            ),
        }
    }
    let depth = chunks.iter().map(depth_of).fold(0.0, f64::max);
    eprintln!(
        "{} chunks built in {:?} with playout held; deepest playout queue {depth:.1} s",
        chunks.len(),
        t0.elapsed()
    );
    assert!(
        depth > 16.0,
        "chunks queue ahead of playout: {:?}",
        chunks
            .iter()
            .map(|m| (
                &m["chunk_index"],
                &m["buffer_depth_seconds"],
                &m["generation_seconds"]
            ))
            .collect::<Vec<_>>()
    );
    // Playout goes on in real time.
    let playout = {
        let clock = clock.clone();
        tokio::spawn(async move {
            let start = Instant::now();
            loop {
                tokio::time::sleep(Duration::from_millis(5)).await;
                clock.set(start.elapsed().as_secs_f64());
            }
        })
    };
    c.more_media(24, 0).await;
    c.assert_rtp_grids();
    c.send(json!({"type": "stop"})).await;
    c.expect("stream_exhausted").await;
    playout.abort();
}

/// The memory guard (`buffer_max_bytes`) is the one thing that holds
/// generation: below one chunk's bytes, chunk 1 is dispatched only once
/// chunk 0 has played out (~5 s), and the session goes on.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn the_memory_guard_holds_generation() {
    let Some(h264) = h264_or_skip() else { return };
    let f = fixture(Opts {
        h264,
        buffer_chunks: 0,
        buffer_max_bytes: 1,
        ..Opts::default()
    })
    .await;
    let mut c = open(&f, "minimax/h3-max/director").await;
    c.expect("session_info").await;
    c.send(json!({"type": "configure", "prompt_version": 1, "prompt": "A lighthouse at dusk", "resolution": "480p"})).await;
    c.expect("configured").await;
    let (_, _) = chunks_until(&mut c, |m| m["chunk_index"] == 0).await;
    let t0 = Instant::now();
    let (_, _) = chunks_until(&mut c, |m| m["chunk_index"] == 1).await;
    assert!(
        t0.elapsed() > Duration::from_millis(4_500),
        "chunk 1 waited for playout: {:?}",
        t0.elapsed()
    );
    c.send(json!({"type": "stop"})).await;
    c.expect("stream_exhausted").await;
}

/// The server's default (`upsample_wait = false`): a chunk whose segment is
/// not expanded yet is built at once with the unexpanded prompt (reported
/// as a fallback), and later chunks get their expansions.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn upsampled_chunks_do_not_wait_for_their_expansion() {
    let Some(h264) = h264_or_skip() else { return };
    let (_llm, svc) = show_llm();
    let f = fixture(Opts {
        h264,
        upsampler: Some(svc),
        upsample_wait: false,
        ..Opts::default()
    })
    .await;
    let mut c = open(&f, "minimax/h3-max/director").await;
    c.expect("session_info").await;
    c.send(world_configure("A lighthouse keeper on a stormy island."))
        .await;
    c.expect("configured").await;
    // Chunk 0 is dispatched with `configure`, before any call can answer.
    let (_, others) = chunks_until(&mut c, |m| m["chunk_index"] == 0).await;
    let s0 = segments(&others)
        .into_iter()
        .find(|m| m["chunk_index"] == 0)
        .unwrap_or_else(|| panic!("{others:?}"));
    assert_eq!(s0["fallback"], true, "{s0}");
    assert_eq!(s0["prompt"], "A lighthouse keeper on a stormy island.");
    assert!(s0["error"].as_str().unwrap().contains("not ready"), "{s0}");
    let mut expanded = false;
    for _ in 0..4 {
        let (_, others) = chunks_until(&mut c, |_| true).await;
        expanded |= segments(&others).iter().any(|m| m["fallback"] == false);
    }
    assert!(expanded, "later chunks are upsampled");
    c.send(json!({"type": "stop"})).await;
    c.expect("stream_exhausted").await;
}
