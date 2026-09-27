//! A standalone Reactor local runtime over the fake engine, for client
//! compatibility runs (`tests/compat/reactor_sdk_compat.py`):
//!
//! ```text
//! cargo run -p fastvideo-reactor --example fake_runtime -- \
//!     --port 8080 --model av|video|causal [--short-edge 160] [--rtf 0.2]
//! ```
//!
//! `av` is a FastH3-style clip model (video + 48 kHz audio, 24 fps), `video`
//! a video-only Wan clip model (16 fps), `causal` the SF-Wan block model.
//! Prints `REACTOR READY <port>` once listening.

use std::sync::Arc;
use std::time::Duration;

use fastvideo_engine_service::{EngineConfig, EngineService, FakeBackend, FakeConfig, FakeModel, FakeTiming, Mp4Mode};
use fastvideo_reactor::{router, H264Backend, Reactor, ReactorConfig};
use fastvideo_webrtc::host::{HostConfig, RtcHost};

fn arg(args: &[String], name: &str) -> Option<String> {
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1).cloned())
}

#[tokio::main]
async fn main() {
    let args: Vec<String> = std::env::args().collect();
    let port: u16 = arg(&args, "--port").and_then(|p| p.parse().ok()).unwrap_or(8080);
    let model = arg(&args, "--model").unwrap_or_else(|| "av".into());
    let short_edge: u32 = arg(&args, "--short-edge").and_then(|p| p.parse().ok()).unwrap_or(160);
    let rtf: f64 = arg(&args, "--rtf").and_then(|p| p.parse().ok()).unwrap_or(0.2);
    let ping_timeout: u64 = arg(&args, "--ping-timeout").and_then(|p| p.parse().ok()).unwrap_or(20);
    let m = match model.as_str() {
        "av" => FakeModel::h3_turbo(),
        "video" => FakeModel::wan(),
        "causal" => FakeModel::sf_wan(),
        other => panic!("unknown --model {other} (av | video | causal)"),
    };
    let fc = FakeConfig {
        models: vec![m],
        timing: FakeTiming { rtf: Some(rtf), step: Duration::from_millis(5), ..FakeTiming::default() },
        mp4: Mp4Mode::Off,
        ..FakeConfig::default()
    };
    let engine = EngineService::start(
        EngineConfig { output_dir: std::env::temp_dir().join("fv-reactor-fake"), ..EngineConfig::default() },
        vec![Box::new(FakeBackend::new(fc))],
    )
    .expect("engine");
    let host = RtcHost::bind(HostConfig { ice_servers: Vec::new(), ..HostConfig::default() })
        .await
        .expect("webrtc host");
    let cfg = ReactorConfig {
        short_edge: Some(short_edge),
        ping_timeout: Duration::from_secs(ping_timeout),
        h264: if cfg!(feature = "openh264") { H264Backend::OpenH264 } else { H264Backend::Off },
        seed: Some(7),
        ..ReactorConfig::default()
    };
    let rt = Reactor::new(cfg, Arc::new(engine), host);
    let app = router(rt);
    let listener = tokio::net::TcpListener::bind(("127.0.0.1", port)).await.expect("bind");
    let port = listener.local_addr().expect("addr").port();
    println!("REACTOR READY {port}");
    axum::serve(listener, app).await.expect("serve");
}
