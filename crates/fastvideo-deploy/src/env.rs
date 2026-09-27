//! Where am I running, and how do clients reach me? (design §5.8, §6.2;
//! research-deploy §1.6, §2, §3.2).
//!
//! [`Discovery`] reads the variables each platform injects and answers:
//!
//! - the platform ([`Platform`]): Runpod serverless (`RUNPOD_WEBHOOK_GET_JOB`
//!   set), Runpod load-balancer worker (`RUNPOD_ENDPOINT_ID` + `PORT`, no
//!   job-take URL), Runpod pod (`RUNPOD_POD_ID`), Vast (`CONTAINER_ID` /
//!   `VAST_CONTAINERLABEL` / any `VAST_TCP_PORT_*`), else local;
//! - the public IP (`RUNPOD_PUBLIC_IP`, `PUBLIC_IPADDR`);
//! - external ports (`RUNPOD_TCP_PORT_<n>`, `VAST_TCP_PORT_<n>`,
//!   `VAST_UDP_PORT_<n>`);
//! - the ICE candidates the WebRTC host must advertise
//!   ([`Discovery::ice_candidates`]): Vast gets a UDP host candidate (and
//!   ICE-TCP when mapped); Runpod never UDP ("Pods do not support UDP"), only
//!   ICE-TCP on a mapped TCP port;
//! - the public HTTP base URL ([`Discovery::public_base_url`]).
//!
//! Ports above 65535 (`70000`, `70010`) are the platforms' *symmetric*
//! request form: the container must listen on the port the variable names,
//! which is also the external port. Ports ≤ 65535 map internal → external.

use std::collections::BTreeMap;
use std::net::{IpAddr, SocketAddr};

use serde::Serialize;

/// The deploy target this process runs on.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum Platform {
    /// Runpod serverless queue worker (job-take URL injected).
    RunpodServerless,
    /// Runpod serverless load-balancer worker (HTTP on `$PORT`).
    RunpodLoadBalancer,
    /// Runpod pod.
    RunpodPod,
    /// Vast instance (or a Vast serverless worker).
    Vast,
    /// Anything else (laptop, CI, plain VM).
    Local,
}

/// ICE transport of a candidate.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum IceProto {
    Udp,
    Tcp,
}

/// One address the WebRTC host must advertise: bind `local_port` inside the
/// container, publish `public` in the SDP (a host candidate with NAT 1:1).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct IceCandidate {
    pub proto: IceProto,
    pub public: SocketAddr,
    pub local_port: u16,
}

/// Environment snapshot (tests pass a map; production [`Discovery::from_process`]).
#[derive(Clone, Debug, Default)]
pub struct Discovery {
    vars: BTreeMap<String, String>,
}

/// The variables discovery reads (for [`Discovery::from_process`] and for
/// redacted diagnostics). Port variables are matched by prefix.
const NAMES: &[&str] = &[
    "RUNPOD_WEBHOOK_GET_JOB",
    "RUNPOD_POD_ID",
    "RUNPOD_ENDPOINT_ID",
    "RUNPOD_PUBLIC_IP",
    "RUNPOD_DC_ID",
    "RUNPOD_GPU_COUNT",
    "PORT",
    "PUBLIC_IPADDR",
    "CONTAINER_ID",
    "VAST_CONTAINERLABEL",
    "GPU_COUNT",
];
const PREFIXES: &[&str] = &["RUNPOD_TCP_PORT_", "VAST_TCP_PORT_", "VAST_UDP_PORT_"];

fn wanted(name: &str) -> bool {
    NAMES.contains(&name) || PREFIXES.iter().any(|p| name.starts_with(p))
}

impl Discovery {
    /// From explicit pairs (tests, or a captured environment).
    pub fn from_pairs<K: Into<String>, V: Into<String>>(pairs: impl IntoIterator<Item = (K, V)>) -> Self {
        let vars = pairs
            .into_iter()
            .map(|(k, v)| (k.into(), v.into()))
            .filter(|(k, v)| wanted(k) && !v.trim().is_empty())
            .collect();
        Self { vars }
    }

    /// From the process environment.
    pub fn from_process() -> Self {
        Self::from_pairs(std::env::vars())
    }

    fn var(&self, name: &str) -> Option<&str> {
        self.vars.get(name).map(|s| s.trim())
    }

    pub fn platform(&self) -> Platform {
        if self.var("RUNPOD_WEBHOOK_GET_JOB").is_some() {
            Platform::RunpodServerless
        } else if self.var("RUNPOD_ENDPOINT_ID").is_some() && self.var("PORT").is_some() {
            Platform::RunpodLoadBalancer
        } else if self.var("RUNPOD_POD_ID").is_some() {
            Platform::RunpodPod
        } else if self.var("CONTAINER_ID").is_some()
            || self.var("VAST_CONTAINERLABEL").is_some()
            || self.vars.keys().any(|k| k.starts_with("VAST_"))
        {
            Platform::Vast
        } else {
            Platform::Local
        }
    }

    /// The public IP the platform reports.
    pub fn public_ip(&self) -> Option<IpAddr> {
        self.var("RUNPOD_PUBLIC_IP").or_else(|| self.var("PUBLIC_IPADDR")).and_then(|s| s.parse().ok())
    }

    fn port(&self, var: &str) -> Option<u16> {
        self.var(var).and_then(|s| s.parse().ok()).filter(|p| *p != 0)
    }

    /// External TCP port for internal port `internal` (Runpod or Vast).
    pub fn tcp_port(&self, internal: u32) -> Option<u16> {
        self.port(&format!("RUNPOD_TCP_PORT_{internal}")).or_else(|| self.port(&format!("VAST_TCP_PORT_{internal}")))
    }

    /// External UDP port for internal port `internal` (Vast only; Runpod has
    /// no inbound UDP).
    pub fn udp_port(&self, internal: u32) -> Option<u16> {
        self.port(&format!("VAST_UDP_PORT_{internal}"))
    }

    /// The candidates to advertise for a WebRTC host configured with
    /// `udp_port` / `tcp_port` (`[webrtc]`, design §6.1). `public_ip`
    /// overrides the discovered IP (`"auto"` or empty = discover).
    ///
    /// Empty when the platform maps nothing (e.g. Runpod serverless without
    /// "Expose TCP ports"): streaming then has to go out over WHIP.
    pub fn ice_candidates(&self, udp_port: u32, tcp_port: u32, public_ip: &str) -> Vec<IceCandidate> {
        let ip = match public_ip.trim() {
            "" | "auto" => self.public_ip(),
            s => s.parse().ok(),
        };
        let Some(ip) = ip else { return Vec::new() };
        let local = |internal: u32, external: u16| -> u16 {
            // Symmetric (> 65535) requests listen on the assigned port.
            u16::try_from(internal).unwrap_or(external)
        };
        let mut out = Vec::new();
        let runpod = matches!(
            self.platform(),
            Platform::RunpodPod | Platform::RunpodServerless | Platform::RunpodLoadBalancer
        );
        if !runpod {
            if let Some(ext) = self.udp_port(udp_port) {
                out.push(IceCandidate {
                    proto: IceProto::Udp,
                    public: SocketAddr::new(ip, ext),
                    local_port: local(udp_port, ext),
                });
            }
        }
        if let Some(ext) = self.tcp_port(tcp_port) {
            out.push(IceCandidate { proto: IceProto::Tcp, public: SocketAddr::new(ip, ext), local_port: local(tcp_port, ext) });
        }
        out
    }

    /// The URL clients use to reach HTTP port `port` of this process:
    /// Runpod pod → `https://<pod>-<port>.proxy.runpod.net`; Runpod LB →
    /// `https://<endpoint>.api.runpod.ai`; Vast (or a Runpod public TCP port)
    /// → `http://<public ip>:<external port>`. `None` on a queue worker
    /// (nothing is reachable) and locally.
    pub fn public_base_url(&self, port: u16) -> Option<String> {
        match self.platform() {
            Platform::RunpodPod => {
                let pod = self.var("RUNPOD_POD_ID")?;
                Some(format!("https://{pod}-{port}.proxy.runpod.net"))
            }
            Platform::RunpodLoadBalancer => {
                let ep = self.var("RUNPOD_ENDPOINT_ID")?;
                Some(format!("https://{ep}.api.runpod.ai"))
            }
            Platform::Vast => {
                let ip = self.public_ip()?;
                let ext = self.tcp_port(u32::from(port))?;
                Some(match ip {
                    IpAddr::V4(v4) => format!("http://{v4}:{ext}"),
                    IpAddr::V6(v6) => format!("http://[{v6}]:{ext}"),
                })
            }
            Platform::RunpodServerless | Platform::Local => None,
        }
    }

    /// A non-secret summary for logs and the Runpod `info` job.
    pub fn summary(&self, http_port: u16, udp_port: u32, tcp_port: u32) -> serde_json::Value {
        serde_json::json!({
            "platform": self.platform(),
            "public_ip": self.public_ip(),
            "public_base_url": self.public_base_url(http_port),
            "ice_candidates": self.ice_candidates(udp_port, tcp_port, "auto"),
            "pod_id": self.var("RUNPOD_POD_ID").or_else(|| self.var("CONTAINER_ID")),
            "endpoint_id": self.var("RUNPOD_ENDPOINT_ID"),
            "dc": self.var("RUNPOD_DC_ID"),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vast_udp_and_tcp_candidates() {
        let d = Discovery::from_pairs([
            ("CONTAINER_ID", "123"),
            ("PUBLIC_IPADDR", "203.0.113.7"),
            ("VAST_UDP_PORT_70010", "41234"),
            ("VAST_TCP_PORT_70000", "41235"),
            ("VAST_TCP_PORT_8000", "40001"),
            ("HOME", "/root"),
        ]);
        assert_eq!(d.platform(), Platform::Vast);
        let c = d.ice_candidates(70010, 70000, "auto");
        assert_eq!(c.len(), 2);
        assert_eq!(c[0].proto, IceProto::Udp);
        assert_eq!(c[0].public, "203.0.113.7:41234".parse().unwrap());
        // Symmetric request: listen on the assigned port.
        assert_eq!(c[0].local_port, 41234);
        assert_eq!(c[1].proto, IceProto::Tcp);
        assert_eq!(d.public_base_url(8000).as_deref(), Some("http://203.0.113.7:40001"));
    }

    #[test]
    fn vast_mapped_port_listens_on_internal() {
        let d = Discovery::from_pairs([("PUBLIC_IPADDR", "203.0.113.7"), ("VAST_UDP_PORT_5000", "41234")]);
        let c = d.ice_candidates(5000, 5001, "");
        assert_eq!(c, vec![IceCandidate { proto: IceProto::Udp, public: "203.0.113.7:41234".parse().unwrap(), local_port: 5000 }]);
    }

    #[test]
    fn runpod_pod_is_tcp_only() {
        let d = Discovery::from_pairs([
            ("RUNPOD_POD_ID", "abc123"),
            ("RUNPOD_PUBLIC_IP", "198.51.100.2"),
            ("RUNPOD_TCP_PORT_70000", "70123"),
            // Runpod never maps UDP; a stray variable must not produce a candidate.
            ("VAST_UDP_PORT_70010", "1"),
        ]);
        assert_eq!(d.platform(), Platform::RunpodPod);
        let c = d.ice_candidates(70010, 70000, "auto");
        // 70123 is not a valid port: nothing.
        assert!(c.is_empty(), "{c:?}");
        let d = Discovery::from_pairs([
            ("RUNPOD_POD_ID", "abc123"),
            ("RUNPOD_PUBLIC_IP", "198.51.100.2"),
            ("RUNPOD_TCP_PORT_70000", "40123"),
        ]);
        let c = d.ice_candidates(70010, 70000, "auto");
        assert_eq!(c, vec![IceCandidate { proto: IceProto::Tcp, public: "198.51.100.2:40123".parse().unwrap(), local_port: 40123 }]);
        assert_eq!(d.public_base_url(8000).as_deref(), Some("https://abc123-8000.proxy.runpod.net"));
    }

    #[test]
    fn serverless_lb_and_local() {
        let q = Discovery::from_pairs([("RUNPOD_WEBHOOK_GET_JOB", "https://x/job-take/$ID?gpu=a"), ("RUNPOD_POD_ID", "w1")]);
        assert_eq!(q.platform(), Platform::RunpodServerless);
        assert_eq!(q.public_base_url(8000), None);
        assert!(q.ice_candidates(70010, 70000, "auto").is_empty());
        let lb = Discovery::from_pairs([("RUNPOD_ENDPOINT_ID", "ep1"), ("PORT", "8000"), ("RUNPOD_POD_ID", "w1")]);
        assert_eq!(lb.platform(), Platform::RunpodLoadBalancer);
        assert_eq!(lb.public_base_url(8000).as_deref(), Some("https://ep1.api.runpod.ai"));
        let l = Discovery::from_pairs(Vec::<(String, String)>::new());
        assert_eq!(l.platform(), Platform::Local);
        // An explicit IP still yields nothing without mapped ports.
        assert!(l.ice_candidates(70010, 70000, "192.0.2.1").is_empty());
        let s = lb.summary(8000, 70010, 70000);
        assert_eq!(s["platform"], "runpod-load-balancer");
    }
}
