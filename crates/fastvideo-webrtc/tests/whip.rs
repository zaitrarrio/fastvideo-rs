//! WHIP publisher against an in-process mock WHIP endpoint (axum) whose
//! media side is a second str0m host (WP-04).
#![cfg(all(feature = "whip", feature = "str0m"))]

use std::sync::{Arc, Mutex};
use std::time::Duration;

use axum::extract::{Path, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{delete, post};
use axum::Router;
use fastvideo_webrtc::channel::ChannelPolicy;
use fastvideo_webrtc::host::{AnswerOptions, AudioLayout, HostConfig, Peer, PeerEvent, RtcHost};
use fastvideo_webrtc::profile::EncodeProfile;
use fastvideo_webrtc::sdp::{MediaKind, Sdp};
use fastvideo_webrtc::whip::{
    WhipAuth, WhipClient, WhipConfig, WhipPublishOptions, WhipPublisher, WhipTarget,
};
use fastvideo_webrtc::writer::{video_rtp_time, AudioPacket, TrackKind, VideoFrame};
use fastvideo_webrtc::WebrtcError;
use url::Url;

#[derive(Default)]
struct Seen {
    posts: Vec<(Option<String>, Option<String>, String)>, // content-type, authorization, body
    deletes: Vec<(String, Option<String>)>,               // id, authorization
}

#[derive(Clone)]
struct Mock {
    host: RtcHost,
    seen: Arc<Mutex<Seen>>,
    peers: Arc<Mutex<Vec<Peer>>>,
    token: &'static str,
}

fn header(h: &HeaderMap, k: &str) -> Option<String> {
    h.get(k).and_then(|v| v.to_str().ok()).map(str::to_string)
}

async fn whip_post(State(m): State<Mock>, headers: HeaderMap, body: String) -> Response {
    let auth = header(&headers, "authorization");
    m.seen.lock().unwrap().posts.push((
        header(&headers, "content-type"),
        auth.clone(),
        body.clone(),
    ));
    if auth.as_deref() != Some(&format!("Bearer {}", m.token))
        && auth.as_deref() != Some("Basic c3Ryb2JlOnB3")
    {
        return (StatusCode::UNAUTHORIZED, "bad token").into_response();
    }
    if header(&headers, "content-type").as_deref() != Some("application/sdp") {
        return StatusCode::UNSUPPORTED_MEDIA_TYPE.into_response();
    }
    // An SFU receives: answer every offered m-line recvonly (str0m inverts).
    let audio = Sdp::parse(&body).ok().and_then(|s| {
        s.media
            .iter()
            .any(|x| x.kind() == MediaKind::Audio)
            .then_some(AudioLayout::Stereo)
    });
    match m
        .host
        .answer(
            &body,
            AnswerOptions {
                audio,
                channels: ChannelPolicy::none(),
                ..Default::default()
            },
        )
        .await
    {
        Ok((peer, answer)) => {
            let id = peer.id();
            m.peers.lock().unwrap().push(peer);
            // Relative Location, resolved by the client per RFC 3986.
            (
                StatusCode::CREATED,
                [
                    ("content-type", "application/sdp".to_string()),
                    ("location", format!("whip/{id}")),
                ],
                answer,
            )
                .into_response()
        }
        Err(e) => (StatusCode::BAD_REQUEST, e.to_string()).into_response(),
    }
}

async fn whip_delete(
    State(m): State<Mock>,
    Path(id): Path<String>,
    headers: HeaderMap,
) -> StatusCode {
    m.seen
        .lock()
        .unwrap()
        .deletes
        .push((id, header(&headers, "authorization")));
    StatusCode::OK
}

async fn start_mock(token: &'static str) -> (Url, Mock) {
    let host = RtcHost::bind(HostConfig::loopback(true, false))
        .await
        .unwrap();
    let mock = Mock {
        host,
        seen: Default::default(),
        peers: Default::default(),
        token,
    };
    let app = Router::new()
        .route("/live/whip", post(whip_post))
        .route("/live/whip/{id}", delete(whip_delete))
        .with_state(mock.clone());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    (
        Url::parse(&format!("http://{addr}/live/whip")).unwrap(),
        mock,
    )
}

async fn next_media(peer: &mut Peer, kind: TrackKind) -> (u64, bytes::Bytes) {
    let deadline = tokio::time::Instant::now() + Duration::from_secs(10);
    loop {
        match tokio::time::timeout_at(deadline, peer.next_event())
            .await
            .expect("timed out waiting for media")
        {
            Some(PeerEvent::Media {
                kind: k,
                rtp_time,
                data,
                ..
            }) if k == kind => return (rtp_time, data),
            Some(PeerEvent::Closed(r)) => panic!("closed: {r:?}"),
            Some(_) => {}
            None => panic!("events ended"),
        }
    }
}

fn idr(i: u8) -> Vec<u8> {
    let mut v = vec![
        0, 0, 0, 1, 0x67, 0x42, 0xe0, 0x1f, 0, 0, 0, 1, 0x68, 0xce, 0, 0, 0, 1, 0x65, i,
    ];
    v.extend(std::iter::repeat_n(0x5a, 2500));
    v
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn publishes_av_and_deletes_on_teardown() {
    let (url, mock) = start_mock("tok").await;
    let publisher_host = RtcHost::bind(HostConfig::loopback(true, false))
        .await
        .unwrap();
    let mut publisher = WhipPublisher::publish(
        &publisher_host,
        WhipConfig::new(url.clone(), WhipAuth::Bearer("tok".into())),
        WhipPublishOptions {
            audio: Some(AudioLayout::Stereo),
            stun: vec![],
            ..Default::default()
        },
    )
    .await
    .unwrap();

    // What the endpoint saw.
    {
        let seen = mock.seen.lock().unwrap();
        let (ct, auth, offer) = &seen.posts[0];
        assert_eq!(ct.as_deref(), Some("application/sdp"));
        assert_eq!(auth.as_deref(), Some("Bearer tok"));
        let o = Sdp::parse(offer).unwrap();
        assert!(o.has_end_of_candidates(), "complete offer");
        assert!(!o.candidates().is_empty());
        assert!(!offer.contains("a=ice-options:trickle"));
        let video = o
            .media
            .iter()
            .find(|m| m.kind() == MediaKind::Video)
            .unwrap();
        let first = video.codecs().into_iter().next().unwrap();
        assert!(first.is("H264"), "H.264 offered first: {}", video.m_line);
        assert!(
            video.codecs().iter().all(|c| c.is("H264") || c.is("rtx")),
            "no VP8: {}",
            video.m_line
        );
        assert_eq!(
            video.direction(),
            fastvideo_webrtc::sdp::Direction::SendOnly
        );
        assert!(offer.contains("stereo=1"));
        // A MediaMTX-style URL is guessed as a native target: level 4.0.
        assert!(offer.contains("profile-level-id=42e028"), "{offer}");
        assert!(!offer.contains("profile-level-id=42e01f"), "{offer}");
    }
    assert_eq!(publisher.profile(), EncodeProfile::NATIVE);
    assert_eq!(publisher.negotiated_video_codec(), Some("H264"));
    let resource = publisher.resource_url().unwrap().clone();
    assert!(
        resource.as_str().starts_with(url.as_str()),
        "{resource} resolved against {url}"
    );

    let mut server_peer = mock.peers.lock().unwrap().pop().unwrap();
    let deadline = tokio::time::Instant::now() + Duration::from_secs(10);
    loop {
        match tokio::time::timeout_at(deadline, publisher.peer().next_event())
            .await
            .unwrap()
        {
            Some(PeerEvent::Connected) => break,
            Some(PeerEvent::Closed(r)) => panic!("{r:?}"),
            _ => {}
        }
    }
    let handle = publisher.peer().handle().clone();
    let sender = tokio::spawn(async move {
        for i in 0..20u64 {
            handle
                .send_video(VideoFrame::new(idr(i as u8), video_rtp_time(i, 24)))
                .unwrap();
            handle
                .send_audio(AudioPacket::new(vec![0xfc, i as u8], i * 960))
                .unwrap();
            tokio::time::sleep(Duration::from_millis(15)).await;
        }
    });
    let (t, data) = next_media(&mut server_peer, TrackKind::Video).await;
    assert_eq!(data[..], idr((t / 3750) as u8)[..]);
    let (t, data) = next_media(&mut server_peer, TrackKind::Audio).await;
    assert_eq!(&data[..], &[0xfc, (t / 960) as u8]);
    sender.await.unwrap();

    // Teardown DELETEs the resolved resource with the same credentials.
    let status = publisher.teardown().await.unwrap();
    assert_eq!(status, Some(200));
    let seen = mock.seen.lock().unwrap();
    assert_eq!(seen.deletes.len(), 1);
    assert_eq!(seen.deletes[0].1.as_deref(), Some("Bearer tok"));
    assert_eq!(
        resource.path_segments().unwrap().next_back().unwrap(),
        seen.deletes[0].0
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn video_only_offer_has_no_audio_mline_and_basic_auth() {
    let (url, mock) = start_mock("unused").await;
    let host = RtcHost::bind(HostConfig::loopback(true, false))
        .await
        .unwrap();
    let publisher = WhipPublisher::publish(
        &host,
        WhipConfig::new(url, WhipAuth::from_user_token("strobe", "pw"))
            .with_target(WhipTarget::Cloudflare),
        WhipPublishOptions {
            audio: None,
            stun: vec![],
            ..Default::default()
        },
    )
    .await
    .unwrap();
    {
        let seen = mock.seen.lock().unwrap();
        let (_, auth, offer) = &seen.posts[0];
        assert_eq!(auth.as_deref(), Some("Basic c3Ryb2JlOnB3"));
        let o = Sdp::parse(offer).unwrap();
        assert_eq!(o.media.len(), 1, "video m-line only: {offer}");
        assert_eq!(o.media[0].kind(), MediaKind::Video);
        // Cloudflare target: level 3.1 offered, 720p cap for the encoder.
        assert!(offer.contains("profile-level-id=42e01f"), "{offer}");
        assert!(!offer.contains("profile-level-id=42e028"), "{offer}");
    }
    assert_eq!(publisher.profile(), EncodeProfile::CLOUDFLARE);
    // Dropping without teardown still DELETEs (best effort, spawned).
    drop(publisher);
    for _ in 0..50 {
        if !mock.seen.lock().unwrap().deletes.is_empty() {
            break;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    assert_eq!(mock.seen.lock().unwrap().deletes.len(), 1);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn rejected_post_is_an_error_without_delete() {
    let (url, mock) = start_mock("right").await;
    let host = RtcHost::bind(HostConfig::loopback(true, false))
        .await
        .unwrap();
    let err = WhipPublisher::publish(
        &host,
        WhipConfig::new(url.clone(), WhipAuth::Bearer("wrong".into())),
        WhipPublishOptions {
            stun: vec![],
            ..Default::default()
        },
    )
    .await
    .unwrap_err();
    match err {
        WebrtcError::WhipStatus { status, body } => {
            assert_eq!(status, 401);
            assert_eq!(body, "bad token");
        }
        e => panic!("{e:?}"),
    }
    assert!(mock.seen.lock().unwrap().deletes.is_empty());

    // Raw client: a non-SDP 2xx is refused, and timeouts are reported.
    let client = WhipClient::new(WhipConfig {
        timeout: Duration::from_millis(500),
        ..WhipConfig::new(
            Url::parse("http://127.0.0.1:9/whip").unwrap(),
            WhipAuth::None,
        )
    })
    .unwrap();
    assert!(matches!(
        client.post_offer("v=0\r\n").await,
        Err(WebrtcError::Whip(_))
    ));
}
