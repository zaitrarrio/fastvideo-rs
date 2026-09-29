//! Receiving client media (design §5.11): a loopback client publishes VP8
//! camera frames (libvpx via ffmpeg) and Opus-shaped packets to a host that
//! answers with `AnswerOptions::receive`; the answer caps the bitrate
//! (`b=AS`); frames arrive on the bounded inbound queue with codec and
//! keyframe flags; PLIs reach the client; the bitrate cap and a full queue
//! drop frames and mark the next one non-contiguous; the ingest decodes
//! into the input rings at the model's size and refuses oversized input.
#![cfg(feature = "str0m")]

use std::sync::Arc;
use std::time::{Duration, Instant};

use fastvideo_media::ring::{InputBufferConfig, InputBuffers};
use fastvideo_media::vp8::{libvpx_available, Vp8Config, Vp8Encoder};
use fastvideo_media::RgbFrame;
use fastvideo_protocol::{InputVideoCodec, VideoInputCaps};
use fastvideo_webrtc::host::{
    AnswerOptions, AudioLayout, HostConfig, InboundMedia, OfferOptions, Peer, PeerEvent, ReceiveOptions, RtcHost,
};
use fastvideo_webrtc::ingest::{Ingest, IngestConfig};
use fastvideo_webrtc::sdp::{Direction, MediaKind, Sdp};
use fastvideo_webrtc::writer::{AudioPacket, TrackKind, VideoCodec, VideoFrame};
use tokio::sync::mpsc;

async fn wait_for(peer: &mut Peer, what: &str, mut f: impl FnMut(&PeerEvent) -> bool) -> PeerEvent {
    let deadline = tokio::time::Instant::now() + Duration::from_secs(10);
    loop {
        match tokio::time::timeout_at(deadline, peer.next_event()).await {
            Ok(Some(e)) if f(&e) => return e,
            Ok(Some(PeerEvent::Closed(r))) => panic!("peer closed ({r:?}) while waiting for {what}"),
            Ok(Some(_)) => {}
            Ok(None) => panic!("event stream ended while waiting for {what}"),
            Err(_) => panic!("timed out waiting for {what}"),
        }
    }
}

/// VP8 frames of `n` solid-colour pictures (keyframe first, then inter).
fn vp8_frames(w: u32, h: u32, rgb: [u8; 3], n: u64) -> Option<Vec<Vec<u8>>> {
    vp8_encode(w, h, n, |i| RgbFrame::solid(w, h, rgb, i))
}

/// VP8 frames of noise (large frames, for the bitrate cap).
fn vp8_noise(w: u32, h: u32, n: u64) -> Option<Vec<Vec<u8>>> {
    vp8_encode(w, h, n, |i| {
        let mut x = 0x9e37_79b9u32.wrapping_add(i as u32);
        let data: Vec<u8> = (0..w * h * 3)
            .map(|_| {
                x ^= x << 13;
                x ^= x >> 17;
                x ^= x << 5;
                x as u8
            })
            .collect();
        RgbFrame::new(w, h, data.into(), i).unwrap()
    })
}

fn vp8_encode(w: u32, h: u32, n: u64, pic: impl Fn(u64) -> RgbFrame) -> Option<Vec<Vec<u8>>> {
    if !libvpx_available() {
        eprintln!("skipping: ffmpeg has no libvpx");
        return None;
    }
    let mut c = Vp8Config::new(w, h, 30);
    c.gop_seconds = 100.0;
    let mut e = Vp8Encoder::new(c).unwrap();
    let mut out = Vec::new();
    for i in 0..n {
        out.extend(e.encode(&pic(i)).unwrap());
    }
    out.extend(e.finish().unwrap());
    Some(out.into_iter().map(|f| f.data.to_vec()).collect())
}

/// One 20 ms mono Opus packet (a real one with the `opus` feature).
fn opus_packet(n: u64) -> Vec<u8> {
    #[cfg(feature = "opus")]
    {
        use fastvideo_media::opus::{OpusConfig, OpusEncoder};
        let mut e = OpusEncoder::new(OpusConfig { channels: 1, ..OpusConfig::whip() }, 0).unwrap();
        let tone: Vec<f32> = (0..960).map(|i| ((i as f32 + n as f32) * 0.05).sin() * 0.2).collect();
        e.encode_frame(&tone).unwrap().to_vec()
    }
    #[cfg(not(feature = "opus"))]
    {
        vec![0xf8, n as u8]
    }
}

struct Pair {
    #[allow(dead_code)]
    hosts: (RtcHost, RtcHost),
    server: Peer,
    client: Peer,
    inbound: mpsc::Receiver<InboundMedia>,
    answer: String,
}

/// The client offers a send-only VP8 camera and an Opus microphone.
async fn connect(recv: ReceiveOptions) -> Pair {
    let server = RtcHost::bind(HostConfig::loopback(true, false)).await.unwrap();
    let client = RtcHost::bind(HostConfig::loopback(true, false)).await.unwrap();
    let (pending, offer) = client
        .offer(OfferOptions {
            video: Some(Direction::SendOnly),
            audio: Some((Direction::SendOnly, AudioLayout::Mono)),
            video_codecs: vec![VideoCodec::Vp8],
            ..OfferOptions::default()
        })
        .await
        .unwrap();
    let (mut sp, answer) = server
        .answer(
            &offer,
            AnswerOptions { video: false, audio: None, receive: Some(recv), ..AnswerOptions::default() },
        )
        .await
        .unwrap();
    let inbound = sp.take_inbound().expect("an inbound queue");
    assert!(sp.take_inbound().is_none(), "taken once");
    let mut cp = pending.accept_answer(&answer).await.unwrap();
    assert_eq!(cp.video_codec(), Some(VideoCodec::Vp8), "the client sends VP8");
    wait_for(&mut sp, "server connected", |e| *e == PeerEvent::Connected).await;
    wait_for(&mut cp, "client connected", |e| *e == PeerEvent::Connected).await;
    Pair { hosts: (server, client), server: sp, client: cp, inbound, answer }
}

async fn recv_n(rx: &mut mpsc::Receiver<InboundMedia>, kind: TrackKind, n: usize) -> Vec<InboundMedia> {
    let mut out = Vec::new();
    let deadline = tokio::time::Instant::now() + Duration::from_secs(10);
    while out.len() < n {
        match tokio::time::timeout_at(deadline, rx.recv()).await {
            Ok(Some(m)) if m.kind == kind => out.push(m),
            Ok(Some(_)) => {}
            _ => panic!("got {} of {n} {kind:?} frames", out.len()),
        }
    }
    out
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn receives_camera_and_microphone_and_sends_plis() {
    let Some(frames) = vp8_frames(160, 120, [200, 40, 40], 10) else { return };
    let mut p = connect(ReceiveOptions { max_bitrate_kbps: 1500, ..ReceiveOptions::default() }).await;
    // The answer receives on both m-lines and caps the client's bitrate.
    let a = Sdp::parse(&p.answer).unwrap();
    for m in &a.media {
        if m.kind() == MediaKind::Video {
            assert_eq!(m.direction(), Direction::RecvOnly, "{}", p.answer);
            assert!(m.lines.iter().any(|l| l == "b=AS:1500"), "{}", p.answer);
        }
        if m.kind() == MediaKind::Audio {
            assert_eq!(m.direction(), Direction::RecvOnly, "{}", p.answer);
            assert!(m.lines.iter().any(|l| l == "b=AS:128"), "{}", p.answer);
        }
    }
    for (i, f) in frames.iter().enumerate() {
        p.client.send_video(VideoFrame::new(f.clone(), i as u64 * 3000)).unwrap();
        // 20 ms Opus packets, in real time (the host forwards bytes;
        // decoding is the ingest's).
        for k in 0..2u64 {
            let n = i as u64 * 2 + k;
            p.client.send_audio(AudioPacket::new(opus_packet(n), n * 960)).unwrap();
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    }
    // Both tracks share the queue: collect until each has arrived.
    let (mut v, mut audio) = (Vec::new(), 0usize);
    let deadline = tokio::time::Instant::now() + Duration::from_secs(10);
    while v.len() < frames.len() || audio < 5 {
        match tokio::time::timeout_at(deadline, p.inbound.recv()).await {
            Ok(Some(m)) if m.kind == TrackKind::Video => v.push(m),
            Ok(Some(m)) => {
                assert_eq!((m.codec, m.keyframe), (None, false));
                audio += 1;
            }
            _ => panic!("got {} video frames and {audio} audio packets", v.len()),
        }
    }
    assert!(v[0].keyframe && v[0].contiguous);
    assert!(v.iter().skip(1).all(|m| !m.keyframe));
    assert!(v.iter().all(|m| m.codec == Some(VideoCodec::Vp8)));
    assert_eq!(v.iter().map(|m| m.data.to_vec()).collect::<Vec<_>>(), frames, "frames arrive intact");
    assert_eq!(v[1].rtp_time - v[0].rtp_time, 3000);
    let mid = v[0].mid.clone();
    // A PLI from the server reaches the client as a keyframe request.
    p.server.request_keyframe(&mid).unwrap();
    wait_for(&mut p.client, "the PLI", |e| matches!(e, PeerEvent::KeyframeRequest { .. })).await;
    tokio::time::sleep(Duration::from_millis(300)).await;
    let st = p.server.stats().inbound;
    assert_eq!(st.video_frames, frames.len() as u64);
    assert!(st.audio_packets >= 5);
    assert_eq!(st.keyframe_requests_sent, 1);
    assert_eq!((st.dropped_bitrate, st.dropped_queue), (0, 0));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn bitrate_cap_drops_and_marks_the_gap() {
    let Some(frames) = vp8_noise(160, 120, 40) else { return };
    let total: usize = frames.iter().map(Vec::len).sum();
    // A cap that takes about half of what is sent in the first 2 s window.
    let kbps = ((total * 8 / 2 / 1000) as u32 * 4 / 5 / 2).max(1);
    let mut p = connect(ReceiveOptions { max_bitrate_kbps: kbps, ..ReceiveOptions::default() }).await;
    for (i, f) in frames.iter().enumerate() {
        p.client.send_video(VideoFrame::new(f.clone(), i as u64 * 3000)).unwrap();
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    tokio::time::sleep(Duration::from_millis(300)).await;
    let st = p.server.stats().inbound;
    assert!(st.dropped_bitrate > 0, "cap {kbps} kbit/s, sent {total} bytes: {st:?}");
    assert!(st.video_frames > 0, "{st:?}");
    let mut got = Vec::new();
    while let Ok(m) = p.inbound.try_recv() {
        got.push(m);
    }
    assert!(got[0].keyframe);
    // A new window: the next frame goes through, flagged after the loss.
    tokio::time::sleep(Duration::from_millis(2100)).await;
    let more = vp8_noise(160, 120, 1).unwrap();
    p.client.send_video(VideoFrame::new(more[0].clone(), 40 * 3000)).unwrap();
    let m = recv_n(&mut p.inbound, TrackKind::Video, 1).await;
    assert!(!m[0].contiguous, "the first frame after a drop is not contiguous");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_full_queue_drops_new_frames() {
    let Some(frames) = vp8_frames(160, 120, [40, 200, 40], 12) else { return };
    let mut p = connect(ReceiveOptions { queue: 3, ..ReceiveOptions::default() }).await;
    for (i, f) in frames.iter().enumerate() {
        p.client.send_video(VideoFrame::new(f.clone(), i as u64 * 3000)).unwrap();
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    tokio::time::sleep(Duration::from_millis(300)).await;
    let st = p.server.stats().inbound;
    assert_eq!(st.video_frames, 3, "{st:?}");
    assert_eq!(st.dropped_queue, 9, "{st:?}");
    let got: Vec<_> = std::iter::from_fn(|| p.inbound.try_recv().ok()).collect();
    assert_eq!(got.len(), 3);
    assert!(got[0].keyframe && got.iter().all(|m| m.contiguous));
}

fn caps(max: (u32, u32)) -> VideoInputCaps {
    VideoInputCaps {
        width: 64,
        height: 48,
        max_width: max.0,
        max_height: max.1,
        max_fps: 60,
        codecs: vec![InputVideoCodec::Vp8, InputVideoCodec::H264],
    }
}

fn buffers() -> Arc<InputBuffers> {
    InputBuffers::new(InputBufferConfig { video_frames: 64, audio_chunks: 64, span: Duration::from_secs(10) })
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn ingest_decodes_into_the_rings_at_the_model_size() {
    let Some(frames) = vp8_frames(160, 120, [30, 60, 220], 12) else { return };
    let mut p = connect(ReceiveOptions::default()).await;
    let bufs = buffers();
    let mut ingest = Ingest::start(IngestConfig::new(Some(caps((1280, 720))), None), bufs.clone(), p.server.handle().clone()).unwrap();
    // A P-frame before any keyframe is dropped (and asks for one).
    for (i, f) in frames.iter().enumerate() {
        p.client.send_video(VideoFrame::new(f.clone(), i as u64 * 3000)).unwrap();
        tokio::time::sleep(Duration::from_millis(33)).await;
    }
    let v = recv_n(&mut p.inbound, TrackKind::Video, frames.len()).await;
    // Start mid-stream: frame 1 (inter) first, then the keyframe and the rest.
    ingest.push(v[1].clone());
    for m in v.iter().cloned() {
        ingest.push(m);
    }
    let t0 = Instant::now();
    while bufs.video.len() < frames.len() - 1 && t0.elapsed() < Duration::from_secs(10) {
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    ingest.close();
    let st = ingest.stats();
    assert_eq!(st.codec.as_deref(), Some("vp8"));
    assert_eq!(st.source_size, Some((160, 120)));
    assert_eq!(st.dropped_waiting_keyframe, 1, "{st:?}");
    assert!(st.keyframe_requests >= 1, "{st:?}");
    let got = bufs.video.drain();
    assert_eq!(got.len(), frames.len(), "every frame decoded, {st:?}");
    for (i, t) in got.iter().enumerate() {
        let f = &t.item.frame;
        assert_eq!((f.width, f.height), (64, 48));
        assert_eq!(t.item.source, (160, 120));
        assert_eq!(t.pts_us, i as u64 * 3000 * 1_000_000 / 90_000, "pts from the RTP clock");
        let c = f.pixel(32, 24).unwrap();
        assert!(c.iter().zip([30u8, 60, 220]).all(|(a, b)| (i32::from(*a) - i32::from(b)).abs() < 40), "{c:?}");
    }
    assert_eq!(st.ended.as_deref(), Some("closed"));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn ingest_refuses_oversized_video_and_unpublished_tracks() {
    let Some(frames) = vp8_frames(320, 240, [90, 90, 90], 4) else { return };
    let mut p = connect(ReceiveOptions::default()).await;
    for (i, f) in frames.iter().enumerate() {
        p.client.send_video(VideoFrame::new(f.clone(), i as u64 * 3000)).unwrap();
    }
    let v = recv_n(&mut p.inbound, TrackKind::Video, frames.len()).await;
    // Above the size limit: refused, nothing buffered.
    let bufs = buffers();
    let mut ingest = Ingest::start(IngestConfig::new(Some(caps((160, 160))), None), bufs.clone(), p.server.handle().clone()).unwrap();
    for m in v.iter().cloned() {
        ingest.push(m);
    }
    tokio::time::sleep(Duration::from_millis(300)).await;
    ingest.close();
    let st = ingest.stats();
    assert!(st.rejected.as_deref().is_some_and(|r| r.contains("320x240 exceeds the 160x160")), "{st:?}");
    assert_eq!(st.dropped_rejected, 1);
    assert_eq!(st.dropped_waiting_keyframe, 3);
    assert!(bufs.video.is_empty());
    // A disabled (unpublished) track drops everything before decoding.
    let mut ingest = Ingest::start(IngestConfig::new(Some(caps((1280, 720))), None), bufs.clone(), p.server.handle().clone()).unwrap();
    ingest.set_enabled(TrackKind::Video, false);
    for m in v.iter().cloned() {
        ingest.push(m);
    }
    // Media on another mid is ignored.
    ingest.set_enabled(TrackKind::Video, true);
    ingest.set_mid(TrackKind::Video, Some("nope".into()));
    for m in v.iter().cloned() {
        ingest.push(m);
    }
    tokio::time::sleep(Duration::from_millis(200)).await;
    ingest.close();
    let st = ingest.stats();
    assert_eq!(st.dropped_unpublished, 4);
    assert_eq!(st.video_frames_in, 0);
    assert!(bufs.video.is_empty());
}
