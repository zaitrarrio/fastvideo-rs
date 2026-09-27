//! The fal WMA director mounted in the assembled fv-serve router (WP-14):
//! signalling routes answer directly and through `/fal/proxy`, a real
//! str0m client gets an answer and `session_info`, and a second session is
//! busy. The full A/V and browser suites live in `fastvideo-fal`
//! (`tests/director_e2e.rs`, `tests/director_browser.rs`).

#![cfg(all(feature = "fal", feature = "webrtc"))]

use std::collections::BTreeMap;
use std::time::Duration;

use axum::body::Body;
use axum::http::Request;
use axum::Router;
use fastvideo_serve::config::{Config, JobBackend};
use fastvideo_serve::{App, Overrides};
use fastvideo_serve_kit::KeyRing;
use fastvideo_webrtc::channel::ChannelMessage;
use fastvideo_webrtc::host::{AudioLayout, HostConfig, OfferOptions, PeerEvent, RtcHost};
use fastvideo_webrtc::sdp::Direction;
use serde_json::{json, Value};
use tower::ServiceExt;

const KEY: &str = "fal-director-key";

async fn app() -> App {
    let dir = std::env::temp_dir().join(format!(
        "fv-serve-director-{:x}",
        std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()
    ));
    let mut env = BTreeMap::new();
    env.insert("FV_API_KEYS".to_owned(), KeyRing::hash_hex(KEY));
    env.insert("FV_URL_SIGNING_KEY".to_owned(), "k".to_owned());
    env.insert("FV_STATE_DIR".to_owned(), dir.display().to_string());
    env.insert("FV_PUBLIC_BASE_URL".to_owned(), "http://fv.test".to_owned());
    let mut c = Config::default();
    c.apply_env(&env).unwrap();
    c.jobs.backend = JobBackend::Memory;
    c.engine.fake.step_ms = 1;
    c.webrtc.public_ip = "127.0.0.1".into();
    c.webrtc.ice_servers = vec![toml::toml! { urls = ["stun:stun.example.test:3478"] }.into()];
    c.validate().unwrap();
    let a = App::build(c, Overrides::default()).await.unwrap();
    a.gate.engine().wait_ready().await;
    a
}

async fn post(app: &Router, uri: &str, body: Value, target: Option<&str>) -> (u16, Value) {
    let mut b = Request::builder().method("POST").uri(uri).header("authorization", format!("Key {KEY}"));
    if let Some(t) = target {
        b = b.header("x-fal-target-url", t);
    }
    let req = b.header("content-type", "application/json").body(Body::from(body.to_string())).unwrap();
    let r = app.clone().oneshot(req).await.unwrap();
    let status = r.status().as_u16();
    let bytes = axum::body::to_bytes(r.into_body(), 1 << 24).await.unwrap();
    (status, serde_json::from_slice(&bytes).unwrap_or(Value::Null))
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn director_is_mounted() {
    let a = app().await;
    let r = &a.router;
    let ice = json!({"ice_servers": [{"urls": ["stun:stun.example.test:3478"]}]});
    assert_eq!(post(r, "/wma/ice", json!({"app_id": "minimax/h3-max/director"}), None).await, (200, ice.clone()));
    assert_eq!(post(r, "/minimax/h3-turbo/director/ice", json!({}), None).await, (200, ice.clone()));
    assert_eq!(post(r, "/fal/proxy", json!({"app_id": "minimax/h3-draft/director"}), Some("https://wma.fal.run/ice")).await, (200, ice.clone()));
    assert_eq!(post(r, "/fal/proxy", json!({}), Some("https://fal.run/minimax/h3-max/director/ice")).await, (200, ice));
    let (s, info) = post(r, "/info", json!({}), None).await;
    assert_eq!((s, info["app"].as_str()), (200, Some("minimax-h3-max-director")));

    // A real client through the proxy: answer, then `session_info` on the
    // client-created control channel.
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
    let (s, v) = post(r, "/fal/proxy", json!({"app_id": "minimax/h3-max/director", "sdp": offer, "type": "offer"}), Some("https://wma.fal.run/session")).await;
    assert_eq!(s, 200, "{v}");
    let mut peer = pending.accept_answer(v["sdp"].as_str().unwrap()).await.unwrap();
    let info = tokio::time::timeout(Duration::from_secs(30), async {
        loop {
            match peer.next_event().await {
                Some(PeerEvent::Message(m)) => break serde_json::from_str::<Value>(m.as_text().unwrap()).unwrap(),
                Some(PeerEvent::Closed(r)) => panic!("closed: {r:?}"),
                Some(_) => {}
                None => panic!("peer gone"),
            }
        }
    })
    .await
    .unwrap();
    assert_eq!((info["type"].as_str(), info["fps"].as_u64()), (Some("session_info"), Some(24)));
    peer.send_message(ChannelMessage::text("control", r#"{"type":"ping","ts":5}"#)).await.unwrap();
    let (s, hb) = post(r, "/wma/session/heartbeat", json!({"session_id": v["session_id"]}), None).await;
    assert_eq!((s, hb), (200, json!({"alive": true})));

    // One session per machine.
    let host2 = RtcHost::bind(HostConfig::loopback(true, false)).await.unwrap();
    let (_p2, offer2) = host2.offer(OfferOptions { channels: vec!["control".into()], ..Default::default() }).await.unwrap();
    let (s, v2) = post(r, "/wma/session", json!({"app_id": "minimax/h3-max/director", "sdp": offer2, "type": "offer"}), None).await;
    assert_eq!(s, 429, "{v2}");
}
