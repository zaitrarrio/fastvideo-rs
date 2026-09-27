//! Loopback tests for the str0m host (WP-04): two real hosts on 127.0.0.1
//! exchange data-channel messages and pre-encoded H.264/Opus; a raw str0m
//! client connects over ICE-TCP; a Chrome-shaped offer is answered.
#![cfg(feature = "str0m")]

use std::net::SocketAddr;
use std::time::{Duration, Instant};

use bytes::Bytes;
use fastvideo_webrtc::channel::{ChannelMessage, ChannelPolicy};
use fastvideo_webrtc::host::{AnswerOptions, AudioLayout, CloseReason, HostConfig, OfferOptions, Peer, PeerEvent, RtcHost};
use fastvideo_webrtc::sdp::{Direction, MediaKind, Sdp};
use fastvideo_webrtc::writer::{video_rtp_time, AudioPacket, TrackKind, VideoFrame, OPUS_FRAME_SAMPLES};

const CHROME_OFFER: &str = include_str!("fixtures/chrome_offer_recvonly.sdp");

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

/// An IDR access unit: SPS, PPS and a large IDR slice (forces FU-A).
fn idr(i: u8) -> Vec<u8> {
    let mut v = vec![0, 0, 0, 1, 0x67, 0x42, 0xe0, 0x1f, 0xda, 0x01, 0x40];
    v.extend_from_slice(&[0, 0, 0, 1, 0x68, 0xce, 0x38, 0x80]);
    v.extend_from_slice(&[0, 0, 0, 1, 0x65, 0x88, i]);
    v.extend((0..3000u32).map(|k| (k as u8).wrapping_mul(7).wrapping_add(i) | 1));
    v
}

fn p_frame(i: u8) -> Vec<u8> {
    let mut v = vec![0, 0, 0, 1, 0x41, 0x9a, i];
    v.extend((0..400u32).map(|k| (k as u8).wrapping_add(i) | 1));
    v
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn two_hosts_exchange_data_and_av() {
    let server = RtcHost::bind(HostConfig::loopback(true, true)).await.unwrap();
    let client = RtcHost::bind(HostConfig::loopback(true, false)).await.unwrap();

    let (pending, offer) = client
        .offer(OfferOptions {
            video: Some(Direction::RecvOnly),
            audio: Some((Direction::RecvOnly, AudioLayout::Stereo)),
            channels: vec!["control".into()],
            srflx: vec![],
        })
        .await
        .unwrap();
    // Our offers are complete (non-trickle) too.
    let o = Sdp::parse(&offer).unwrap();
    assert!(o.has_end_of_candidates());
    assert!(!offer.contains("a=ice-options:trickle"));

    let (mut sp, answer) = server
        .answer(&offer, AnswerOptions { channels: ChannelPolicy::fal(), ..AnswerOptions::default() })
        .await
        .unwrap();
    let a = Sdp::parse(&answer).unwrap();
    assert!(a.has_end_of_candidates(), "{answer}");
    assert!(!answer.contains("a=ice-options:trickle"));
    // Both server transports are advertised: UDP host and ICE-TCP passive.
    let udp_port = server.udp_addr().unwrap().port();
    let tcp_port = server.tcp_addr().unwrap().port();
    assert!(a.candidates().iter().any(|c| c.contains(&format!(" udp ")) && c.contains(&format!(" 127.0.0.1 {udp_port} typ host"))), "{answer}");
    assert!(a.candidates().iter().any(|c| c.contains(" tcp ") && c.contains(&format!(" 127.0.0.1 {tcp_port} typ host")) && c.contains("tcptype passive")), "{answer}");
    assert!(answer.contains("stereo=1"), "{answer}");
    assert_eq!(a.media.iter().find(|m| m.kind() == MediaKind::Video).unwrap().direction(), Direction::SendOnly);
    assert!(sp.sends(TrackKind::Video) && sp.sends(TrackKind::Audio));

    let mut cp = pending.accept_answer(&answer).await.unwrap();
    wait_for(&mut sp, "server connected", |e| *e == PeerEvent::Connected).await;
    wait_for(&mut cp, "client connected", |e| *e == PeerEvent::Connected).await;
    wait_for(&mut sp, "server channel open", |e| matches!(e, PeerEvent::ChannelOpen { label } if label == "control")).await;

    // Data channel, both directions (JSON text as fal/Reactor send it).
    cp.send_message(ChannelMessage::text("control", r#"{"type":"ping","ts":1}"#)).await.unwrap();
    let got = wait_for(&mut sp, "client message", |e| matches!(e, PeerEvent::Message(_))).await;
    let PeerEvent::Message(m) = got else { unreachable!() };
    assert_eq!(m.label, "control");
    assert_eq!(m.as_text(), Some(r#"{"type":"ping","ts":1}"#));
    sp.send_message(ChannelMessage::text("control", r#"{"type":"pong","client_ts":1}"#)).await.unwrap();
    sp.send_message(ChannelMessage::binary("control", vec![1u8, 2, 3])).await.unwrap();
    let got = wait_for(&mut cp, "server text", |e| matches!(e, PeerEvent::Message(_))).await;
    let PeerEvent::Message(m) = got else { unreachable!() };
    assert_eq!(m.as_text(), Some(r#"{"type":"pong","client_ts":1}"#));
    let got = wait_for(&mut cp, "server binary", |e| matches!(e, PeerEvent::Message(_))).await;
    let PeerEvent::Message(m) = got else { unreachable!() };
    assert!(m.binary);
    assert_eq!(&m.data[..], &[1, 2, 3]);

    // A/V: 24 fps video and 20 ms Opus, both on their own RTP clocks.
    let frames: Vec<VideoFrame> = (0..36u64)
        .map(|i| {
            let data = if i % 12 == 0 { idr(i as u8) } else { p_frame(i as u8) };
            VideoFrame::new(data, 1000 + video_rtp_time(i, 24))
        })
        .collect();
    let packets: Vec<AudioPacket> =
        (0..75u64).map(|i| AudioPacket::new(vec![0xfc, i as u8, 0x11, 0x22], 5000 + i * OPUS_FRAME_SAMPLES as u64)).collect();
    let sender = sp.handle().clone();
    let (vf, ap) = (frames.clone(), packets.clone());
    let started = Instant::now();
    tokio::spawn(async move {
        let mut a = ap.into_iter();
        for f in vf {
            sender.send_video(f).unwrap();
            for p in a.by_ref().take(2) {
                sender.send_audio(p).unwrap();
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        for p in a {
            sender.send_audio(p).unwrap();
        }
    });

    let mut video = Vec::new();
    let mut audio = Vec::new();
    let deadline = tokio::time::Instant::now() + Duration::from_secs(10);
    while video.len() < frames.len() - 2 || audio.len() < packets.len() - 2 {
        match tokio::time::timeout_at(deadline, cp.next_event()).await {
            Ok(Some(PeerEvent::Media { kind: TrackKind::Video, rtp_time, data, keyframe, .. })) => video.push((rtp_time, data, keyframe)),
            Ok(Some(PeerEvent::Media { kind: TrackKind::Audio, rtp_time, data, .. })) => audio.push((rtp_time, data)),
            Ok(Some(PeerEvent::Closed(r))) => panic!("closed: {r:?}"),
            Ok(Some(_)) => {}
            Ok(None) | Err(_) => break,
        }
    }
    let elapsed = started.elapsed();
    eprintln!("loopback: {} / {} video frames, {} / {} opus packets in {elapsed:?}", video.len(), frames.len(), audio.len(), packets.len());
    assert!(video.len() >= frames.len() - 2, "video frames received: {}", video.len());
    assert!(audio.len() >= packets.len() - 2, "audio packets received: {}", audio.len());
    // Every received frame is byte-identical to the one sent with its timestamp.
    for (t, data, keyframe) in &video {
        let sent = frames.iter().find(|f| f.rtp_time == *t).unwrap_or_else(|| panic!("unexpected video ts {t}"));
        assert_eq!(&sent.data[..], &data[..], "video frame at {t}");
        assert_eq!(sent.is_keyframe(), *keyframe);
    }
    assert!(video.windows(2).all(|w| w[0].0 < w[1].0), "video in order");
    for (t, data) in &audio {
        let sent = packets.iter().find(|p| p.rtp_time == *t).unwrap_or_else(|| panic!("unexpected audio ts {t}"));
        assert_eq!(sent.data, *data);
    }

    // Stats are refreshed from str0m.
    tokio::time::sleep(Duration::from_millis(1200)).await;
    let st = sp.stats();
    assert!(st.connected);
    assert_eq!(st.video.written, frames.len() as u64);
    assert_eq!(st.audio.written, packets.len() as u64);
    assert!(st.bytes_tx > 0);
    assert_eq!(st.transport.as_deref(), Some("udp"));
    assert!(server.stats().udp_rx_packets > 0);

    // Close: the remote side notices, the local side reports Local.
    sp.close();
    wait_for(&mut sp, "local close", |e| *e == PeerEvent::Closed(CloseReason::Local)).await;
    client.shutdown().await;
    server.shutdown().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn pause_gate_and_video_only_answers() {
    let server = RtcHost::bind(HostConfig::loopback(true, false)).await.unwrap();
    let client = RtcHost::bind(HostConfig::loopback(true, false)).await.unwrap();
    let (pending, offer) = client
        .offer(OfferOptions {
            video: Some(Direction::RecvOnly),
            audio: Some((Direction::RecvOnly, AudioLayout::Mono)),
            channels: vec!["data".into(), "control".into()],
            srflx: vec![],
        })
        .await
        .unwrap();
    // Video-only session (e.g. SF-Wan): audio m-line answered inactive,
    // tracks start paused (Reactor pause gate).
    let (mut sp, answer) = server
        .answer(&offer, AnswerOptions { audio: None, start_paused: true, channels: ChannelPolicy::reactor(), ..Default::default() })
        .await
        .unwrap();
    let a = Sdp::parse(&answer).unwrap();
    assert_eq!(a.media.iter().find(|m| m.kind() == MediaKind::Audio).unwrap().direction(), Direction::Inactive, "{answer}");
    assert!(!sp.sends(TrackKind::Audio));
    let mut cp = pending.accept_answer(&answer).await.unwrap();
    wait_for(&mut sp, "connected", |e| *e == PeerEvent::Connected).await;
    wait_for(&mut cp, "client connected", |e| *e == PeerEvent::Connected).await;

    // Paused: nothing is sent.
    for i in 0..5u64 {
        sp.send_video(VideoFrame::new(idr(i as u8), video_rtp_time(i, 24))).unwrap();
        sp.send_audio(AudioPacket::new(vec![0xfc, 1], i * 960)).unwrap();
    }
    tokio::time::sleep(Duration::from_millis(300)).await;
    while let Some(e) = cp.try_next_event() {
        assert!(!matches!(e, PeerEvent::Media { .. }), "media leaked through the pause gate: {e:?}");
    }
    // Resume video: a keyframe request is raised for the encoder, frames flow.
    let video_mid = sp.media().iter().find(|m| m.kind == TrackKind::Video).unwrap().mid.clone();
    sp.set_paused(&video_mid, false).unwrap();
    wait_for(&mut sp, "keyframe request on resume", |e| matches!(e, PeerEvent::KeyframeRequest { mid } if *mid == video_mid)).await;
    for i in 5..10u64 {
        sp.send_video(VideoFrame::new(idr(i as u8), video_rtp_time(i, 24))).unwrap();
        sp.send_audio(AudioPacket::new(vec![0xfc, 1], i * 960)).unwrap();
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    wait_for(&mut cp, "video after resume", |e| matches!(e, PeerEvent::Media { kind: TrackKind::Video, .. })).await;
    tokio::time::sleep(Duration::from_millis(300)).await;
    let st = sp.stats();
    assert_eq!(st.video.dropped_paused, 5);
    assert_eq!(st.audio.dropped_paused, 10, "audio inactive: every packet dropped");
    assert_eq!(st.audio.written, 0);
    server.shutdown().await;
    client.shutdown().await;
}

#[tokio::test]
async fn answers_a_chrome_offer() {
    let server = RtcHost::bind(HostConfig::loopback(true, true)).await.unwrap();
    let (peer, answer) = server.answer(CHROME_OFFER, AnswerOptions::default()).await.unwrap();
    let a = Sdp::parse(&answer).unwrap();
    assert_eq!(a.media.len(), 3);
    assert_eq!(a.media.iter().map(|m| m.mid().unwrap().to_string()).collect::<Vec<_>>(), vec!["0", "1", "2"]);
    let v = &a.media[0];
    assert_eq!(v.direction(), Direction::SendOnly);
    // Only H.264 packetization-mode=1 CB/B survives; the answer keeps the
    // offerer's payload types.
    let codecs: Vec<_> = v.codecs().into_iter().filter(|c| !c.is("rtx")).collect();
    assert!(!codecs.is_empty(), "{answer}");
    assert!(codecs.iter().all(|c| c.is_sendable_h264()), "{codecs:?}");
    assert!(codecs.iter().any(|c| c.pt == 106 || c.pt == 102));
    assert_eq!(a.media[1].direction(), Direction::SendOnly);
    assert!(a.media[1].codecs().iter().any(|c| c.is_opus() && c.pt == 111));
    assert!(answer.contains("stereo=1;sprop-stereo=1"));
    assert_eq!(a.media[2].kind(), MediaKind::Application);
    assert!(a.has_end_of_candidates());
    assert!(!answer.contains("a=ice-options:trickle"));
    assert!(a.session_attr("group").is_some_and(|g| g.starts_with("BUNDLE")));
    assert_eq!(peer.media().len(), 2);

    // Reactor client trickle: mDNS and end-of-candidates are ignored, IPs added.
    assert!(!peer.add_remote_candidate("").await.unwrap());
    assert!(!peer.add_remote_candidate("candidate:1 1 udp 2113937151 abc.local 5000 typ host").await.unwrap());
    assert!(peer.add_remote_candidate("a=candidate:2 1 udp 2113937151 192.0.2.10 5000 typ host generation 0").await.unwrap());
    assert!(peer.add_remote_candidate("candidate:3 1 udp 1677729535 198.51.100.9 6000 typ srflx raddr 0.0.0.0 rport 0").await.unwrap());

    // Video-only session answering the same offer: audio inactive.
    let (_p2, answer2) = server.answer(CHROME_OFFER, AnswerOptions { audio: None, ..Default::default() }).await.unwrap();
    let a2 = Sdp::parse(&answer2).unwrap();
    assert_eq!(a2.media[1].direction(), Direction::Inactive);
    assert_eq!(a2.media[0].direction(), Direction::SendOnly);
    server.shutdown().await;
}

#[tokio::test]
async fn rejects_unusable_offers_and_enforces_limits() {
    let mut cfg = HostConfig::loopback(true, false);
    cfg.max_peers = 1;
    cfg.negotiation_timeout = Duration::from_millis(300);
    let server = RtcHost::bind(cfg).await.unwrap();
    // No H.264: refused before str0m sees it.
    let vp8_only = CHROME_OFFER.replace("H264/90000", "H263/90000");
    let err = server.answer(&vp8_only, AnswerOptions::default()).await.unwrap_err();
    assert!(err.to_string().contains("H.264"), "{err}");
    assert!(server.answer("not sdp", AnswerOptions::default()).await.is_err());

    let (mut p, _) = server.answer(CHROME_OFFER, AnswerOptions::default()).await.unwrap();
    let err = server.answer(CHROME_OFFER, AnswerOptions::default()).await.unwrap_err();
    assert!(err.to_string().contains("peer limit"), "{err}");
    // Nobody connects: negotiation timeout closes the peer and frees the slot.
    wait_for(&mut p, "negotiation timeout", |e| *e == PeerEvent::Closed(CloseReason::NegotiationTimeout)).await;
    tokio::time::sleep(Duration::from_millis(50)).await;
    server.answer(CHROME_OFFER, AnswerOptions::default()).await.unwrap();
    server.shutdown().await;
}

#[tokio::test]
async fn public_address_candidates_are_advertised() {
    let mut cfg = HostConfig::loopback(true, true);
    cfg.public.tcp = Some("203.0.113.7:40123".parse().unwrap());
    cfg.public.udp = Some("203.0.113.7:41234".parse().unwrap());
    let server = RtcHost::bind(cfg).await.unwrap();
    let (_p, answer) = server.answer(CHROME_OFFER, AnswerOptions::default()).await.unwrap();
    let a = Sdp::parse(&answer).unwrap();
    let c = a.candidates();
    assert!(c.iter().any(|c| c.contains(" tcp ") && c.contains("203.0.113.7 40123 typ host tcptype passive")), "{c:?}");
    assert!(c.iter().any(|c| c.contains(" udp ") && c.contains("203.0.113.7 41234 typ host")), "{c:?}");
    // Local addresses are still advertised for same-host clients.
    assert!(c.iter().any(|c| c.contains("127.0.0.1")), "{c:?}");
    server.shutdown().await;
}

/// ICE-TCP: a raw str0m client with only an active TCP candidate connects to
/// a TCP-only host (the Runpod pod case: no UDP at all).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn ice_tcp_only_host() {
    use fastvideo_webrtc::framing;
    use fastvideo_webrtc::str0m::change::SdpAnswer;
    use fastvideo_webrtc::str0m::media::{Direction as RDir, MediaKind as RKind};
    use fastvideo_webrtc::str0m::net::{Protocol, Receive, TcpType};
    use fastvideo_webrtc::str0m::{Candidate, Event, Input, Output, Rtc};
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    let server = RtcHost::bind(HostConfig::loopback(false, true)).await.unwrap();
    assert!(server.udp_addr().is_none());
    let server_addr: SocketAddr = server.tcp_addr().unwrap();
    let stream = tokio::net::TcpStream::connect(server_addr).await.unwrap();
    let local = stream.local_addr().unwrap();
    let (mut rd, mut wr) = stream.into_split();

    let mut rtc = Rtc::builder().build(Instant::now());
    rtc.add_local_candidate(Candidate::builder().tcp().host(local).tcptype(TcpType::Active).build().unwrap());
    let mut api = rtc.sdp_api();
    api.add_media(RKind::Video, RDir::RecvOnly, None, None, None);
    api.add_channel("control".into());
    let (offer, pending) = api.apply().unwrap();

    let (mut sp, answer) = server
        .answer(&offer.to_sdp_string(), AnswerOptions { audio: None, channels: ChannelPolicy::fal(), ..Default::default() })
        .await
        .unwrap();
    let a = Sdp::parse(&answer).unwrap();
    let cands = a.candidates();
    assert!(cands.iter().all(|c| c.contains(" tcp ")), "tcp-only host advertises tcp only: {cands:?}");
    assert!(cands.iter().any(|c| c.contains(&format!("{} {} typ host tcptype passive", server_addr.ip(), server_addr.port()))), "{cands:?}");
    rtc.sdp_api().accept_answer(pending, SdpAnswer::from_sdp_string(&answer).unwrap()).unwrap();

    // Server side: greet on channel open.
    let greeter = tokio::spawn(async move {
        wait_for(&mut sp, "tcp connected", |e| *e == PeerEvent::Connected).await;
        wait_for(&mut sp, "tcp channel", |e| matches!(e, PeerEvent::ChannelOpen { .. })).await;
        sp.send_message(ChannelMessage::text("control", "hello over ice-tcp")).await.unwrap();
        tokio::time::sleep(Duration::from_millis(500)).await;
        sp.stats()
    });

    // Drive the client by hand: RFC 4571 frames over the TCP stream.
    let mut dec = framing::Decoder::new();
    let mut buf = vec![0u8; 4096];
    let deadline = Instant::now() + Duration::from_secs(10);
    let mut got = None;
    'outer: while Instant::now() < deadline {
        let timeout = loop {
            match rtc.poll_output().unwrap() {
                Output::Timeout(t) => break t,
                Output::Transmit(t) => {
                    assert_eq!(t.proto, Protocol::Tcp);
                    assert_eq!(t.destination, server_addr);
                    wr.write_all(&framing::frame(&t.contents).unwrap()).await.unwrap();
                }
                Output::Event(Event::ChannelData(d)) => {
                    got = Some(String::from_utf8(d.data).unwrap());
                    break 'outer;
                }
                Output::Event(_) => {}
            }
        };
        let wait = timeout.saturating_duration_since(Instant::now()).max(Duration::from_millis(1));
        match tokio::time::timeout(wait, rd.read(&mut buf)).await {
            Ok(Ok(0)) => panic!("server closed the ice-tcp stream"),
            Ok(Ok(n)) => {
                for f in dec.push(&buf[..n]).unwrap() {
                    let r = Receive::new(Protocol::Tcp, server_addr, local, &f).unwrap();
                    rtc.handle_input(Input::Receive(Instant::now(), r)).unwrap();
                }
            }
            Ok(Err(e)) => panic!("{e}"),
            Err(_) => rtc.handle_input(Input::Timeout(Instant::now())).unwrap(),
        }
    }
    assert_eq!(got.as_deref(), Some("hello over ice-tcp"));
    // Keep answering the server's checks while it reports stats.
    let stats = tokio::time::timeout(Duration::from_secs(5), async {
        loop {
            while let Ok(o) = rtc.poll_output() {
                match o {
                    Output::Timeout(_) => break,
                    Output::Transmit(t) => {
                        let _ = wr.write_all(&framing::frame(&t.contents).unwrap()).await;
                    }
                    Output::Event(_) => {}
                }
            }
            if greeter.is_finished() {
                return greeter.await.unwrap();
            }
            match tokio::time::timeout(Duration::from_millis(20), rd.read(&mut buf)).await {
                Ok(Ok(n)) if n > 0 => {
                    for f in dec.push(&buf[..n]).unwrap() {
                        let r = Receive::new(Protocol::Tcp, server_addr, local, &f).unwrap();
                        let _ = rtc.handle_input(Input::Receive(Instant::now(), r));
                    }
                }
                _ => {
                    let _ = rtc.handle_input(Input::Timeout(Instant::now()));
                }
            }
        }
    })
    .await
    .unwrap();
    assert!(stats.connected);
    assert!(server.stats().tcp_rx_packets > 0);
    assert_eq!(server.stats().udp_rx_packets, 0);
    let _ = Bytes::new();
    server.shutdown().await;
}

#[tokio::test]
async fn stun_probe_learns_the_mapped_address() {
    let fake = tokio::net::UdpSocket::bind("127.0.0.1:0").await.unwrap();
    let fake_addr = fake.local_addr().unwrap();
    tokio::spawn(async move {
        let mut buf = [0u8; 1500];
        loop {
            let Ok((n, src)) = fake.recv_from(&mut buf).await else { return };
            if let Some(tx) = fastvideo_webrtc::stun::parse_binding_request(&buf[..n]) {
                let _ = fake.send_to(&fastvideo_webrtc::stun::binding_success(&tx, "203.0.113.5:4444".parse().unwrap()), src).await;
                let _ = src;
            }
        }
    });
    let host = RtcHost::bind(HostConfig::loopback(true, false)).await.unwrap();
    let mapped = host.stun_probe(fake_addr, Duration::from_secs(2)).await.unwrap();
    assert_eq!(mapped, "203.0.113.5:4444".parse::<SocketAddr>().unwrap());
    let servers = vec![fastvideo_webrtc::ice::IceServer::stun(format!("stun:127.0.0.1:{}", fake_addr.port()))];
    assert_eq!(host.gather_srflx(&servers, Duration::from_secs(2)).await, vec![mapped]);
    // The srflx candidate is advertised in our offer.
    let (_pending, offer) = host.offer(OfferOptions { srflx: vec![mapped], ..Default::default() }).await.unwrap();
    assert!(offer.contains("203.0.113.5 4444 typ srflx"), "{offer}");
    // A silent server times out.
    let dead = tokio::net::UdpSocket::bind("127.0.0.1:0").await.unwrap();
    assert!(host.stun_probe(dead.local_addr().unwrap(), Duration::from_millis(200)).await.is_err());
    host.shutdown().await;
}

#[test]
fn bytes_is_cheap_to_share() {
    // Fan-out clones the encoded bitstream per peer; Bytes clones are O(1).
    let b = Bytes::from(vec![0u8; 1 << 20]);
    let c = b.clone();
    assert_eq!(b.as_ptr(), c.as_ptr());
}
