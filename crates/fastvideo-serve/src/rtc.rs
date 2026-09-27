//! The one WebRTC host of this process (design §5.8): the Reactor runtime
//! (WP-13) and the fal director (WP-14) answer offers on it, so they share
//! the `[webrtc]` UDP/ICE-TCP ports (symbolic Runpod/Vast keys resolved
//! from the environment).

use anyhow::Context;
use fastvideo_webrtc::host::{HostConfig, RtcHost};
use fastvideo_webrtc::ice::{ice_servers_from_rt_env, resolve_ports, IceServer};

use crate::config::Config;

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

/// Binds the shared host from `[webrtc]` and the process environment.
pub async fn bind(c: &Config) -> anyhow::Result<RtcHost> {
    let host = RtcHost::bind(host_config(c, |k| std::env::var(k).ok())).await.context("binding the WebRTC host")?;
    tracing::info!(udp = ?host.udp_addr(), tcp = ?host.tcp_addr(), "webrtc host bound");
    Ok(host)
}
