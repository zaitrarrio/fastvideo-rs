//! The Reactor local runtime mount (design §5.7, WP-13): the str0m host from
//! `[webrtc]` (symbolic Runpod/Vast ports resolved from the environment,
//! design §5.8) and `fastvideo_reactor::Reactor` over the shared engine.

use std::sync::Arc;
use std::time::Duration;

use anyhow::{anyhow, Context};
use fastvideo_engine_service::EngineService;
use fastvideo_reactor::{H264Backend, Reactor, ReactorConfig};
use fastvideo_webrtc::host::{HostConfig, RtcHost};
use fastvideo_webrtc::ice::{ice_servers_from_rt_env, resolve_ports, IceServer};

use crate::config::{Config, ReactorCfg};

/// `[reactor] h264` → the encoder behind H.264 peers.
pub fn h264_backend(s: &str) -> anyhow::Result<H264Backend> {
    match s {
        "nvenc" => Ok(H264Backend::Nvenc),
        "openh264" => Ok(H264Backend::OpenH264),
        "off" => Ok(H264Backend::Off),
        other => Err(anyhow!("[reactor] h264 = {other:?} (nvenc | openh264 | off)")),
    }
}

/// `ReactorConfig` from `[reactor]`.
pub fn reactor_config(c: &ReactorCfg) -> anyhow::Result<ReactorConfig> {
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
        ..ReactorConfig::default()
    })
}

/// ICE servers for clients: `[webrtc] ice_servers`, else RT's
/// `STUN_SERVERS`/`TURN_SERVERS`, else Google STUN.
pub fn ice_servers(c: &Config, env: impl Fn(&str) -> Option<String>) -> Vec<IceServer> {
    let configured: Vec<IceServer> = c
        .webrtc
        .ice_servers
        .iter()
        .filter_map(|v| serde_json::to_value(v).ok())
        .filter_map(|v| IceServer::from_reactor_json(&v))
        .collect();
    if !configured.is_empty() {
        return configured;
    }
    ice_servers_from_rt_env(env("STUN_SERVERS").as_deref(), env("TURN_SERVERS").as_deref())
}

/// The host config for this machine: `[webrtc]` ports resolved against the
/// platform environment; an ephemeral UDP port when nothing resolves (a
/// plain dev box without the symbolic port variables).
pub fn host_config(c: &Config, env: impl Fn(&str) -> Option<String>) -> HostConfig {
    let public_ip = c.webrtc.public_ip.trim().to_owned();
    let get = |k: &str| {
        if k == "FV_PUBLIC_IP" && !public_ip.is_empty() && public_ip != "auto" {
            return Some(public_ip.clone());
        }
        env(k)
    };
    let r = resolve_ports(get, c.webrtc.udp_port, c.webrtc.tcp_port);
    let any = |p: u16| std::net::SocketAddr::from(([0, 0, 0, 0], p));
    let (udp, tcp) = match (r.udp_bind, r.tcp_bind) {
        (None, None) => (Some(any(0)), None),
        (u, t) => (u.map(any), t.map(any)),
    };
    HostConfig {
        udp_bind: udp,
        tcp_bind: tcp,
        public: r.public,
        ice_servers: ice_servers(c, &env),
        max_peers: c.reactor.max_connections.max(1),
        ..HostConfig::default()
    }
}

/// Binds the host and builds the runtime.
pub async fn build(c: &Config, engine: &EngineService) -> anyhow::Result<Reactor> {
    let cfg = reactor_config(&c.reactor)?;
    let host = RtcHost::bind(host_config(c, |k| std::env::var(k).ok()))
        .await
        .context("binding the WebRTC host for Reactor")?;
    tracing::info!(udp = ?host.udp_addr(), tcp = ?host.tcp_addr(), "reactor runtime ready to answer offers");
    Ok(Reactor::new(cfg, Arc::new(engine.clone()), host))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_maps() {
        let c = Config::default();
        let r = reactor_config(&c.reactor).unwrap();
        assert_eq!(r.orphan_timeout, Duration::from_secs(60));
        assert_eq!(r.ping_timeout, Duration::from_secs(20));
        assert_eq!(r.max_connections, 64);
        assert_eq!(r.h264, H264Backend::Nvenc);
        assert!(h264_backend("x264").is_err());
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
