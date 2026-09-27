//! ICE configuration: servers handed to clients, and the addresses the host
//! advertises as candidates (design §5.8, deploy §2-§3).
//!
//! The server itself never uses TURN (str0m has no TURN client, risk R2).
//! [`IceServer`]s are configuration **for clients**: the fal `/wma/ice`
//! reply, the Reactor `GET …/ice_servers` reply, and STUN for the WHIP
//! publisher's server-reflexive candidate.
//!
//! Candidate addresses: the host binds one UDP socket and one TCP listener
//! and advertises *host* candidates at public addresses when the platform
//! maps ports (Runpod `RUNPOD_PUBLIC_IP:$RUNPOD_TCP_PORT_70000`, Vast
//! `PUBLIC_IPADDR:$VAST_UDP_PORT_70010`). Inbound packets are attributed to
//! the advertised address (1:1 NAT), see [`CandidatePlan::destination_for`].

use std::net::{IpAddr, SocketAddr};

use serde::{Deserialize, Serialize};

/// One STUN/TURN server, in the W3C `RTCIceServer` shape (fal `/ice`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IceServer {
    pub urls: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub username: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub credential: Option<String>,
}

impl IceServer {
    pub fn stun(url: impl Into<String>) -> Self {
        IceServer {
            urls: vec![url.into()],
            username: None,
            credential: None,
        }
    }

    pub fn turn(
        url: impl Into<String>,
        username: impl Into<String>,
        credential: impl Into<String>,
    ) -> Self {
        IceServer {
            urls: vec![url.into()],
            username: Some(username.into()),
            credential: Some(credential.into()),
        }
    }

    /// RT's default when neither `STUN_SERVERS` nor `TURN_SERVERS` is set
    /// (reactor §3.3), also our config default (§6.1).
    pub fn default_stun() -> Self {
        IceServer::stun("stun:stun.l.google.com:19302")
    }

    /// `RTCIceServer` JSON: `{"urls":[…],"username"?,"credential"?}` (fal).
    pub fn to_w3c_json(&self) -> serde_json::Value {
        serde_json::to_value(self).unwrap_or(serde_json::Value::Null)
    }

    /// Reactor shape: `{"uris":[…],"credentials":{"username","password"}}`
    /// (reactor §3.3 route 1).
    pub fn to_reactor_json(&self) -> serde_json::Value {
        let mut v = serde_json::json!({ "uris": self.urls });
        if let (Some(u), Some(p)) = (&self.username, &self.credential) {
            v["credentials"] = serde_json::json!({ "username": u, "password": p });
        }
        v
    }

    /// Parse the Reactor shape back (client-supplied `ice_servers` in
    /// `SdpParamsRequest`).
    pub fn from_reactor_json(v: &serde_json::Value) -> Option<Self> {
        let urls: Vec<String> = v
            .get("uris")
            .or_else(|| v.get("urls"))?
            .as_array()?
            .iter()
            .filter_map(|u| u.as_str().map(str::to_string))
            .collect();
        if urls.is_empty() {
            return None;
        }
        let creds = v.get("credentials");
        let s = |c: Option<&serde_json::Value>, k: &str| {
            c.and_then(|c| c.get(k))
                .and_then(|x| x.as_str())
                .map(str::to_string)
        };
        Some(IceServer {
            urls,
            username: s(creds, "username").or_else(|| s(Some(v), "username")),
            credential: s(creds, "password").or_else(|| s(Some(v), "credential")),
        })
    }

    /// `stun:` URLs as `(host, port)` (default port 3478). `stuns:`, `turn:`
    /// and `turns:` are skipped: the server only probes plain STUN.
    pub fn stun_targets(&self) -> Vec<(String, u16)> {
        self.urls
            .iter()
            .filter_map(|u| u.strip_prefix("stun:"))
            .filter_map(|hp| {
                let hp = hp.split('?').next()?;
                if let Some(rest) = hp.strip_prefix('[') {
                    // [v6]:port
                    let (h, p) = rest.split_once(']')?;
                    let port = p
                        .strip_prefix(':')
                        .map(|p| p.parse().ok())
                        .unwrap_or(Some(3478))?;
                    return Some((h.to_string(), port));
                }
                match hp.rsplit_once(':') {
                    Some((h, p)) => Some((h.to_string(), p.parse().ok()?)),
                    None => Some((hp.to_string(), 3478)),
                }
            })
            .collect()
    }
}

/// Parse RT-style environment values (reactor §3.3): `STUN_SERVERS` is
/// comma-separated URLs, `TURN_SERVERS` is comma-separated `user;cred;url`.
/// With neither set, the default is Google STUN.
pub fn ice_servers_from_rt_env(stun: Option<&str>, turn: Option<&str>) -> Vec<IceServer> {
    let mut out = Vec::new();
    for u in stun
        .unwrap_or_default()
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
    {
        out.push(IceServer::stun(u));
    }
    for t in turn
        .unwrap_or_default()
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
    {
        let mut parts = t.splitn(3, ';');
        if let (Some(user), Some(cred), Some(url)) = (parts.next(), parts.next(), parts.next()) {
            out.push(IceServer::turn(url, user, cred));
        }
    }
    if stun.is_none() && turn.is_none() {
        out.push(IceServer::default_stun());
    }
    out
}

/// Public addresses discovered from the platform environment
/// (deploy §2, §3.2; design §5.8).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PublicAddrs {
    pub udp: Option<SocketAddr>,
    pub tcp: Option<SocketAddr>,
}

/// Ports to bind plus the public addresses to advertise for them.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ResolvedPorts {
    /// Local UDP port to bind (`None`: no UDP, e.g. a Runpod pod).
    pub udp_bind: Option<u16>,
    /// Local TCP port to bind for ICE-TCP.
    pub tcp_bind: Option<u16>,
    pub public: PublicAddrs,
}

/// Resolve the configured `[webrtc] udp_port`/`tcp_port` against the
/// platform environment. `get` is the env lookup
/// (`|k| std::env::var(k).ok()`), injectable for tests.
///
/// **Port keys above 65535 are symbolic.** Runpod and Vast hand out
/// *symmetrical* mappings when a port "above 70000" is requested: the real
/// port (internal == external) is in `RUNPOD_TCP_PORT_<key>` /
/// `VAST_TCP_PORT_<key>` / `VAST_UDP_PORT_<key>` (deploy §2, §3.2). That is
/// what design §5.8's `70000/tcp` and `70010/udp` mean: we bind the value of
/// the variable and advertise `PUBLIC_IP:value`. Without the variable a
/// symbolic key cannot be bound and that transport is off.
///
/// A key ≤ 65535 is a real port: bind it, and advertise the mapped external
/// port when a mapping variable exists (Vast random mapping), else the same
/// port (identity mapping on a plain VM).
///
/// - Runpod pods: `RUNPOD_PUBLIC_IP` + `RUNPOD_TCP_PORT_<key>`. Pods have no
///   inbound UDP (deploy §2), so UDP is off.
/// - Vast: `PUBLIC_IPADDR` + `VAST_UDP_PORT_<key>` / `VAST_TCP_PORT_<key>`.
/// - `FV_PUBLIC_IP` overrides the public IP anywhere.
pub fn resolve_ports(
    get: impl Fn(&str) -> Option<String>,
    udp_key: u32,
    tcp_key: u32,
) -> ResolvedPorts {
    let ip = |k: &str| get(k).and_then(|v| v.trim().parse::<IpAddr>().ok());
    let port = |k: String| get(&k).and_then(|v| v.trim().parse::<u16>().ok());
    let real = |key: u32| u16::try_from(key).ok();
    let override_ip = ip("FV_PUBLIC_IP");

    if get("RUNPOD_POD_ID").is_some() || get("RUNPOD_PUBLIC_IP").is_some() {
        let mapped = port(format!("RUNPOD_TCP_PORT_{tcp_key}"));
        let tcp_bind = real(tcp_key).or(mapped);
        let public_ip = override_ip.or_else(|| ip("RUNPOD_PUBLIC_IP"));
        let tcp = public_ip.zip(mapped).map(|(i, p)| SocketAddr::new(i, p));
        return ResolvedPorts {
            udp_bind: None,
            tcp_bind,
            public: PublicAddrs { udp: None, tcp },
        };
    }

    let vast = get("VAST_CONTAINERLABEL").is_some() || get("PUBLIC_IPADDR").is_some();
    let public_ip = override_ip.or_else(|| ip("PUBLIC_IPADDR"));
    let one = |proto: &str, key: u32| -> (Option<u16>, Option<SocketAddr>) {
        let mapped = port(format!("VAST_{proto}_PORT_{key}"));
        let bind = real(key).or(mapped);
        let external = mapped.or(if vast { None } else { bind });
        (
            bind,
            public_ip.zip(external).map(|(i, p)| SocketAddr::new(i, p)),
        )
    };
    let (udp_bind, udp) = one("UDP", udp_key);
    let (tcp_bind, tcp) = one("TCP", tcp_key);
    ResolvedPorts {
        udp_bind,
        tcp_bind,
        public: PublicAddrs { udp, tcp },
    }
}

/// Transport of a candidate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Transport {
    Udp,
    /// ICE-TCP, passive (RFC 6544): the remote connects to us.
    TcpPassive,
}

/// The host candidates one host advertises, per transport.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CandidatePlan {
    pub udp: Vec<SocketAddr>,
    pub tcp: Vec<SocketAddr>,
}

impl CandidatePlan {
    /// Build the plan from the bound local sockets plus explicit/public
    /// addresses. A socket bound to a specific IP advertises that address;
    /// one bound to `0.0.0.0`/`::` advertises `default_ip` (the primary
    /// interface) with the bound port. Public addresses are added first so
    /// they are preferred for non-local peers. Loopback is advertised when
    /// bound to loopback (tests, same-host clients).
    pub fn build(
        udp_local: Option<SocketAddr>,
        tcp_local: Option<SocketAddr>,
        public: &PublicAddrs,
        extra_udp: &[SocketAddr],
        extra_tcp: &[SocketAddr],
        default_ip: Option<IpAddr>,
    ) -> Self {
        let expand =
            |local: Option<SocketAddr>, public: Option<SocketAddr>, extra: &[SocketAddr]| {
                let mut v: Vec<SocketAddr> = Vec::new();
                let mut push = |a: SocketAddr| {
                    if !v.contains(&a) && !a.ip().is_unspecified() {
                        v.push(a);
                    }
                };
                if let Some(p) = public {
                    push(p);
                }
                for e in extra {
                    push(*e);
                }
                if let Some(l) = local {
                    if l.ip().is_unspecified() {
                        if let Some(ip) = default_ip {
                            push(SocketAddr::new(ip, l.port()));
                        }
                    } else {
                        push(l);
                    }
                }
                v
            };
        CandidatePlan {
            udp: if udp_local.is_some() {
                expand(udp_local, public.udp, extra_udp)
            } else {
                Vec::new()
            },
            tcp: if tcp_local.is_some() {
                expand(tcp_local, public.tcp, extra_tcp)
            } else {
                Vec::new()
            },
        }
    }

    pub fn addrs(&self, t: Transport) -> &[SocketAddr] {
        match t {
            Transport::Udp => &self.udp,
            Transport::TcpPassive => &self.tcp,
        }
    }

    /// Which advertised address an inbound packet from `source` was aimed
    /// at. The socket can't tell us (NAT rewrites it), so: loopback sources
    /// map to a loopback candidate, other sources to the first non-loopback
    /// candidate of the same IP family (public addresses come first).
    pub fn destination_for(&self, t: Transport, source: SocketAddr) -> Option<SocketAddr> {
        let addrs = self.addrs(t);
        let same_family = |a: &&SocketAddr| a.is_ipv4() == source.is_ipv4();
        if source.ip().is_loopback() {
            if let Some(a) = addrs
                .iter()
                .filter(same_family)
                .find(|a| a.ip().is_loopback())
            {
                return Some(*a);
            }
        }
        addrs
            .iter()
            .filter(same_family)
            .find(|a| !a.ip().is_loopback())
            .or_else(|| addrs.iter().find(same_family))
            .or_else(|| addrs.first())
            .copied()
    }

    /// The `a=candidate` values this plan produces, before str0m assigns
    /// foundations (priority as RFC 8445 with libwebrtc's TCP preference).
    /// Used for logging and the pure tests; the host adds the same
    /// candidates through str0m.
    pub fn candidate_lines(&self) -> Vec<String> {
        let mut out = Vec::new();
        let n = |i: usize| 65_535u32.saturating_sub(i as u32);
        for (i, a) in self.udp.iter().enumerate() {
            let prio = (126u32 << 24) | (n(i) << 8) | 255;
            out.push(format!(
                "candidate:{} 1 udp {prio} {} {} typ host",
                i + 1,
                a.ip(),
                a.port()
            ));
        }
        for (i, a) in self.tcp.iter().enumerate() {
            let prio = (90u32 << 24) | (n(i) << 8) | 255;
            out.push(format!(
                "candidate:{} 1 tcp {prio} {} {} typ host tcptype passive",
                100 + i + 1,
                a.ip(),
                a.port()
            ));
        }
        out
    }
}

/// The primary interface IP, found by "connecting" a UDP socket to a
/// public address (no packet is sent). `None` without a default route.
pub fn default_interface_ip(v6: bool) -> Option<IpAddr> {
    let (bind, target) = if v6 {
        ("[::]:0", "[2001:4860:4860::8888]:80")
    } else {
        ("0.0.0.0:0", "8.8.8.8:80")
    };
    let s = std::net::UdpSocket::bind(bind).ok()?;
    s.connect(target).ok()?;
    let ip = s.local_addr().ok()?.ip();
    (!ip.is_unspecified()).then_some(ip)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn env(pairs: &[(&str, &str)]) -> impl Fn(&str) -> Option<String> {
        let m: HashMap<String, String> = pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect();
        move |k| m.get(k).cloned()
    }

    #[test]
    fn ice_server_json_shapes() {
        let s = IceServer::turn("turn:turn.example:3478?transport=tcp", "u", "p");
        assert_eq!(
            s.to_w3c_json(),
            serde_json::json!({"urls":["turn:turn.example:3478?transport=tcp"],"username":"u","credential":"p"})
        );
        assert_eq!(
            s.to_reactor_json(),
            serde_json::json!({"uris":["turn:turn.example:3478?transport=tcp"],"credentials":{"username":"u","password":"p"}})
        );
        assert_eq!(
            IceServer::from_reactor_json(&s.to_reactor_json()),
            Some(s.clone())
        );
        assert_eq!(IceServer::from_reactor_json(&s.to_w3c_json()), Some(s));
        assert_eq!(
            IceServer::default_stun().to_reactor_json(),
            serde_json::json!({"uris":["stun:stun.l.google.com:19302"]})
        );
    }

    #[test]
    fn rt_env_parsing() {
        assert_eq!(
            ice_servers_from_rt_env(None, None),
            vec![IceServer::default_stun()]
        );
        let v = ice_servers_from_rt_env(Some("stun:a:1, stun:b:2"), Some("user;cred;turn:t:3478"));
        assert_eq!(
            v,
            vec![
                IceServer::stun("stun:a:1"),
                IceServer::stun("stun:b:2"),
                IceServer::turn("turn:t:3478", "user", "cred")
            ]
        );
        assert!(ice_servers_from_rt_env(Some(""), None).is_empty());
    }

    #[test]
    fn stun_targets() {
        let s = IceServer {
            urls: vec![
                "stun:stun.l.google.com:19302".into(),
                "stun:example.org".into(),
                "stun:[2001:db8::1]:3479".into(),
                "turn:t:3478".into(),
            ],
            username: None,
            credential: None,
        };
        assert_eq!(
            s.stun_targets(),
            vec![
                ("stun.l.google.com".into(), 19302),
                ("example.org".into(), 3478),
                ("2001:db8::1".into(), 3479)
            ]
        );
    }

    #[test]
    fn runpod_symmetrical_port_is_bound_and_advertised() {
        let r = resolve_ports(
            env(&[
                ("RUNPOD_POD_ID", "abc"),
                ("RUNPOD_PUBLIC_IP", "203.0.113.7"),
                ("RUNPOD_TCP_PORT_70000", "40123"),
            ]),
            70010,
            70000,
        );
        assert_eq!(
            r,
            ResolvedPorts {
                udp_bind: None,
                tcp_bind: Some(40123),
                public: PublicAddrs {
                    udp: None,
                    tcp: Some("203.0.113.7:40123".parse().unwrap())
                },
            }
        );
        // No mapping variable: a symbolic key can't be bound.
        let r = resolve_ports(
            env(&[
                ("RUNPOD_POD_ID", "abc"),
                ("RUNPOD_PUBLIC_IP", "203.0.113.7"),
            ]),
            70010,
            70000,
        );
        assert_eq!(r.tcp_bind, None);
        assert_eq!(r.public.tcp, None);
    }

    #[test]
    fn vast_env_maps_ports() {
        // Symbolic keys: identity mapping read from the variables.
        let r = resolve_ports(
            env(&[
                ("PUBLIC_IPADDR", "198.51.100.4"),
                ("VAST_UDP_PORT_70010", "41234"),
                ("VAST_TCP_PORT_70000", "41235"),
            ]),
            70010,
            70000,
        );
        assert_eq!((r.udp_bind, r.tcp_bind), (Some(41234), Some(41235)));
        assert_eq!(r.public.udp, Some("198.51.100.4:41234".parse().unwrap()));
        assert_eq!(r.public.tcp, Some("198.51.100.4:41235".parse().unwrap()));
        // Real internal port with a random external mapping.
        let r = resolve_ports(
            env(&[
                ("PUBLIC_IPADDR", "198.51.100.4"),
                ("VAST_UDP_PORT_8189", "41000"),
            ]),
            8189,
            8190,
        );
        assert_eq!(r.udp_bind, Some(8189));
        assert_eq!(r.public.udp, Some("198.51.100.4:41000".parse().unwrap()));
        // Unmapped port on Vast is not reachable from outside: no public candidate.
        assert_eq!(r.tcp_bind, Some(8190));
        assert_eq!(r.public.tcp, None);
        // Plain VM with an explicit public IP: identity mapping.
        let r = resolve_ports(env(&[("FV_PUBLIC_IP", "192.0.2.9")]), 40010, 40000);
        assert_eq!(r.public.udp, Some("192.0.2.9:40010".parse().unwrap()));
        assert_eq!(r.public.tcp, Some("192.0.2.9:40000".parse().unwrap()));
        // Nothing known: bind real ports, no public candidates.
        let r = resolve_ports(env(&[]), 40010, 70000);
        assert_eq!(
            r,
            ResolvedPorts {
                udp_bind: Some(40010),
                tcp_bind: None,
                public: PublicAddrs::default()
            }
        );
    }

    #[test]
    fn plan_and_destination_mapping() {
        let public = PublicAddrs {
            udp: Some("203.0.113.7:41234".parse().unwrap()),
            tcp: Some("203.0.113.7:40000".parse().unwrap()),
        };
        let plan = CandidatePlan::build(
            Some("0.0.0.0:40010".parse().unwrap()),
            Some("0.0.0.0:40000".parse().unwrap()),
            &public,
            &["127.0.0.1:40010".parse().unwrap()],
            &[],
            Some("10.0.0.5".parse().unwrap()),
        );
        assert_eq!(
            plan.udp,
            vec![
                "203.0.113.7:41234".parse().unwrap(),
                "127.0.0.1:40010".parse().unwrap(),
                "10.0.0.5:40010".parse::<SocketAddr>().unwrap()
            ]
        );
        assert_eq!(
            plan.tcp,
            vec![
                "203.0.113.7:40000".parse().unwrap(),
                "10.0.0.5:40000".parse::<SocketAddr>().unwrap()
            ]
        );
        assert_eq!(
            plan.destination_for(Transport::Udp, "127.0.0.1:5555".parse().unwrap()),
            Some("127.0.0.1:40010".parse().unwrap())
        );
        assert_eq!(
            plan.destination_for(Transport::Udp, "8.8.8.8:5555".parse().unwrap()),
            Some("203.0.113.7:41234".parse().unwrap())
        );
        assert_eq!(
            plan.destination_for(Transport::TcpPassive, "8.8.8.8:5555".parse().unwrap()),
            Some("203.0.113.7:40000".parse().unwrap())
        );
        // Loopback source without a loopback candidate falls back to the first.
        assert_eq!(
            plan.destination_for(Transport::TcpPassive, "127.0.0.1:1".parse().unwrap()),
            Some("203.0.113.7:40000".parse().unwrap())
        );
    }

    #[test]
    fn ice_tcp_candidate_lines() {
        let plan = CandidatePlan::build(
            None,
            Some("0.0.0.0:40000".parse().unwrap()),
            &PublicAddrs {
                udp: None,
                tcp: Some("203.0.113.7:40000".parse().unwrap()),
            },
            &[],
            &[],
            None,
        );
        assert!(plan.udp.is_empty());
        let lines = plan.candidate_lines();
        assert_eq!(lines.len(), 1);
        assert!(
            lines[0].ends_with(" tcp 1526726655 203.0.113.7 40000 typ host tcptype passive"),
            "{}",
            lines[0]
        );
        // Priorities: UDP host outranks TCP host.
        let both = CandidatePlan {
            udp: vec!["1.2.3.4:1".parse().unwrap()],
            tcp: vec!["1.2.3.4:2".parse().unwrap()],
        };
        let prio = |l: &str| l.split_whitespace().nth(3).unwrap().parse::<u32>().unwrap();
        let l = both.candidate_lines();
        assert!(prio(&l[0]) > prio(&l[1]));
    }
}
