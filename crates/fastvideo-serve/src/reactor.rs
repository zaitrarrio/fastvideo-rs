//! The Reactor local runtime mount (design §5.7, WP-13): the str0m host from
//! `[webrtc]` (symbolic Runpod/Vast ports resolved from the environment,
//! design §5.8) and `fastvideo_reactor::Reactor` over the shared engine.

use std::sync::Arc;
use std::time::Duration;

use anyhow::anyhow;
use fastvideo_engine_service::EngineService;
use fastvideo_reactor::{H264Backend, IngestAuth, Reactor, ReactorConfig};
use fastvideo_webrtc::host::RtcHost;

use crate::config::{Config, ReactorCfg, StreamsCfg};

/// `[reactor] h264` → the encoder behind H.264 peers (`auto`, normally
/// resolved at startup by [`crate::encoders::resolve`], probes here).
pub fn h264_backend(s: &str) -> anyhow::Result<H264Backend> {
    match s {
        "nvenc" => Ok(H264Backend::Nvenc),
        "openh264" => Ok(H264Backend::OpenH264),
        "off" => Ok(H264Backend::Off),
        "auto" => h264_backend(&crate::encoders::auto_selection().1.reactor),
        other => Err(anyhow!("[reactor] h264 = {other:?} (auto | nvenc | openh264 | off)")),
    }
}

/// `ReactorConfig` from `[reactor]` and the `[streams]` causal limits.
pub fn reactor_config(c: &ReactorCfg, streams: &StreamsCfg) -> anyhow::Result<ReactorConfig> {
    Ok(ReactorConfig {
        model: c.model.clone(),
        short_edge: c.short_edge,
        aspect: c.aspect.clone(),
        seed: c.seed,
        orphan_timeout: Duration::from_secs(c.orphan_timeout_s),
        ping_timeout: Duration::from_secs(c.ping_timeout_s),
        max_connections: c.max_connections,
        h264: h264_backend(&c.h264)?,
        h264_bitrate_bps: c.h264_bitrate_bps,
        causal_limits: streams.causal_limits(),
        mode: match c.mode.as_str() {
            "avatar" => Some(fastvideo_reactor::engine::Mode::Avatar),
            _ => None,
        },
        avatar: fastvideo_reactor::AvatarSettings {
            window_s: c.avatar_window_s,
            size: crate::config::parse_size(&c.avatar_size)
                .ok_or_else(|| anyhow!("[reactor] avatar_size = {:?} (WxH)", c.avatar_size))?,
            session_max_s: c.avatar_session_max_s,
            ..fastvideo_reactor::AvatarSettings::default()
        },
        ..ReactorConfig::default()
    })
}

// The host config is shared with the fal director (one WebRTC host).
pub use crate::rtc::{host_config, ice_servers};

/// Binds the host and builds the runtime.
pub async fn build(c: &Config, engine: &EngineService) -> anyhow::Result<Reactor> {
    let host = crate::rtc::bind(c).await?;
    build_on(c, engine, host, None)
}

/// Duplex sessions (client camera/microphone in, design §5.11) need an API
/// key like the native API (`Authorization: Bearer`); clip and causal
/// Reactor sessions stay open, as RT. `auth.mode = none` lets everyone in.
pub fn ingest_auth(ctx: fastvideo_serve_kit::ServeCtx) -> IngestAuth {
    IngestAuth(Arc::new(move |h: &axum::http::HeaderMap| {
        ctx.auth()
            .authenticate(fastvideo_protocol::ProtocolId::Native, h)
            .map(|_| ())
            .map_err(|e| e.message)
    }))
}

/// Builds the runtime on an already bound host (shared with the fal
/// director in `App::build`); `auth` guards duplex sessions.
pub fn build_on(c: &Config, engine: &EngineService, host: RtcHost, auth: Option<IngestAuth>) -> anyhow::Result<Reactor> {
    let mut cfg = reactor_config(&c.reactor, &c.streams)?;
    cfg.ingest_auth = auth;
    tracing::info!(udp = ?host.udp_addr(), tcp = ?host.tcp_addr(), "reactor runtime ready to answer offers");
    Ok(Reactor::new(cfg, Arc::new(engine.clone()), host))
}

#[cfg(test)]
mod tests {
    use super::*;
    use fastvideo_webrtc::ice::IceServer;

    #[test]
    fn config_maps() {
        let c = Config::default();
        let r = reactor_config(&c.reactor, &c.streams).unwrap();
        assert_eq!(r.causal_limits, fastvideo_protocol::CausalLimits { default_max_s: 120, hard_max_s: 300 });
        assert_eq!(r.orphan_timeout, Duration::from_secs(60));
        assert_eq!(r.ping_timeout, Duration::from_secs(20));
        assert_eq!(r.max_connections, 64);
        assert_eq!(r.mode, None);
        assert_eq!(r.avatar.size, (640, 352));
        let mut rc = c.reactor.clone();
        rc.mode = "avatar".into();
        assert_eq!(reactor_config(&rc, &c.streams).unwrap().mode, Some(fastvideo_reactor::engine::Mode::Avatar));
        assert_eq!(h264_backend("nvenc").unwrap(), H264Backend::Nvenc);
        assert!(h264_backend("x264").is_err());
        // `auto` follows the per-process NVENC probe.
        let want = h264_backend(&crate::encoders::auto_selection().1.reactor).unwrap();
        assert_eq!(r.h264, want);
    }

    #[test]
    fn ports_and_ice() {
        let c = Config::default();
        // Symbolic keys and no platform env: an ephemeral UDP port.
        let h = host_config(&c, |_| None);
        assert_eq!(h.udp_bind.map(|a| a.port()), Some(0));
        assert!(h.tcp_bind.is_none());
        assert_eq!(h.ice_servers, vec![IceServer::default_stun()]);
        // Runpod pod: ICE-TCP only on the mapped symmetrical port.
        let env = |k: &str| match k {
            "RUNPOD_POD_ID" => Some("p".to_owned()),
            "RUNPOD_PUBLIC_IP" => Some("203.0.113.9".to_owned()),
            "RUNPOD_TCP_PORT_70000" => Some("41234".to_owned()),
            "STUN_SERVERS" => Some("stun:a.example:3478".to_owned()),
            _ => None,
        };
        let h = host_config(&c, env);
        assert!(h.udp_bind.is_none());
        assert_eq!(h.tcp_bind.map(|a| a.port()), Some(41234));
        assert_eq!(h.public.tcp.map(|a| a.to_string()).as_deref(), Some("203.0.113.9:41234"));
        assert_eq!(h.ice_servers, vec![IceServer::stun("stun:a.example:3478")]);
    }
}
