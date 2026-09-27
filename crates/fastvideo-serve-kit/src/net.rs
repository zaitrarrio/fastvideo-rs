//! Outbound target checks shared by media fetch and callbacks: the SSRF guard.
//!
//! A target is resolved once, every resolved address is checked, and the
//! request is then pinned to a checked address (no second DNS lookup, so no
//! rebinding between check and connect).

use std::net::{IpAddr, Ipv4Addr, SocketAddr};

use url::{Host, Url};

/// Which outbound targets are allowed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TargetPolicy {
    /// Only `https://` (LTX: "HTTPS only").
    pub https_only: bool,
    /// Literal IP hosts allowed (LTX: "domain names only").
    pub allow_ip_literals: bool,
    /// Loopback/private/link-local targets allowed. Off in production; the
    /// SSRF guard. Tests turn it on to reach a local server.
    pub allow_private: bool,
}

impl Default for TargetPolicy {
    fn default() -> Self {
        Self {
            https_only: false,
            allow_ip_literals: true,
            allow_private: false,
        }
    }
}

/// Why a target was refused.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum TargetError {
    #[error("only https URLs are accepted")]
    NotHttps,
    #[error("unsupported URL scheme `{0}`")]
    Scheme(String),
    #[error("IP address hosts are not accepted")]
    IpLiteral,
    #[error("URL has no host")]
    NoHost,
    #[error("host `{0}` resolves to a non-public address")]
    Private(String),
    #[error("host `{0}` could not be resolved")]
    Resolve(String),
}

/// Whether `ip` is not a public unicast address.
pub fn is_forbidden_ip(ip: IpAddr) -> bool {
    match ip {
        IpAddr::V4(v4) => forbidden_v4(v4),
        IpAddr::V6(v6) => {
            if let Some(v4) = v6.to_ipv4_mapped() {
                return forbidden_v4(v4);
            }
            let s = v6.segments();
            v6.is_loopback()
                || v6.is_unspecified()
                || v6.is_multicast()
                || (s[0] & 0xfe00) == 0xfc00 // unique local fc00::/7
                || (s[0] & 0xffc0) == 0xfe80 // link local fe80::/10
                || (s[0] == 0x2001 && s[1] == 0x0db8) // documentation
                || (s[0] == 0x0064 && s[1] == 0xff9b) // NAT64 (maps v4)
                || (s[0] == 0 && s[1] == 0 && s[2] == 0 && s[3] == 0 && s[4] == 0 && s[5] == 0) // v4-compatible
        }
    }
}

fn forbidden_v4(ip: Ipv4Addr) -> bool {
    let o = ip.octets();
    ip.is_private()
        || ip.is_loopback()
        || ip.is_link_local()
        || ip.is_unspecified()
        || ip.is_broadcast()
        || ip.is_multicast()
        || ip.is_documentation()
        || o[0] == 0
        || (o[0] == 100 && (o[1] & 0xc0) == 64) // CGNAT 100.64/10
        || (o[0] == 192 && o[1] == 0 && o[2] == 0) // IETF 192.0.0/24
        || (o[0] == 198 && (o[1] & 0xfe) == 18) // benchmarking 198.18/15
        || o[0] >= 240 // reserved
}

/// Checks scheme and host form without DNS.
pub fn check_url(url: &Url, p: &TargetPolicy) -> Result<(), TargetError> {
    match url.scheme() {
        "https" => {}
        "http" if p.https_only => return Err(TargetError::NotHttps),
        "http" => {}
        s => return Err(TargetError::Scheme(s.to_owned())),
    }
    match url.host() {
        None => Err(TargetError::NoHost),
        Some(Host::Domain(_)) => Ok(()),
        Some(Host::Ipv4(ip)) => check_ip(IpAddr::V4(ip), p, url),
        Some(Host::Ipv6(ip)) => check_ip(IpAddr::V6(ip), p, url),
    }
}

fn check_ip(ip: IpAddr, p: &TargetPolicy, url: &Url) -> Result<(), TargetError> {
    if !p.allow_ip_literals {
        return Err(TargetError::IpLiteral);
    }
    if !p.allow_private && is_forbidden_ip(ip) {
        return Err(TargetError::Private(url.host_str().unwrap_or_default().to_owned()));
    }
    Ok(())
}

/// Checks `url` and resolves it to the allowed socket addresses to pin the
/// connection to. Every resolved address must be allowed (a name with any
/// private address is refused).
pub async fn resolve_target(url: &Url, p: &TargetPolicy) -> Result<Vec<SocketAddr>, TargetError> {
    check_url(url, p)?;
    let port = url.port_or_known_default().unwrap_or(443);
    let host = url.host().ok_or(TargetError::NoHost)?;
    let addrs: Vec<SocketAddr> = match host {
        Host::Ipv4(ip) => vec![SocketAddr::new(IpAddr::V4(ip), port)],
        Host::Ipv6(ip) => vec![SocketAddr::new(IpAddr::V6(ip), port)],
        Host::Domain(d) => tokio::net::lookup_host((d, port))
            .await
            .map_err(|_| TargetError::Resolve(d.to_owned()))?
            .collect(),
    };
    let name = url.host_str().unwrap_or_default().to_owned();
    if addrs.is_empty() {
        return Err(TargetError::Resolve(name));
    }
    if !p.allow_private && addrs.iter().any(|a| is_forbidden_ip(a.ip())) {
        return Err(TargetError::Private(name));
    }
    Ok(addrs)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn forbidden_ranges() {
        for s in [
            "127.0.0.1", "10.1.2.3", "172.16.0.1", "192.168.1.1", "169.254.169.254", "0.0.0.0",
            "100.64.0.1", "224.0.0.1", "255.255.255.255", "::1", "fc00::1", "fe80::1",
            "::ffff:127.0.0.1", "::ffff:10.0.0.1", "64:ff9b::a00:1",
        ] {
            assert!(is_forbidden_ip(s.parse().unwrap()), "{s}");
        }
        for s in ["8.8.8.8", "1.1.1.1", "2606:4700:4700::1111"] {
            assert!(!is_forbidden_ip(s.parse().unwrap()), "{s}");
        }
    }

    #[test]
    fn url_rules() {
        let ltx = TargetPolicy { https_only: true, allow_ip_literals: false, allow_private: false };
        let u = |s: &str| Url::parse(s).unwrap();
        assert_eq!(check_url(&u("http://example.com/a.png"), &ltx), Err(TargetError::NotHttps));
        assert_eq!(check_url(&u("https://8.8.8.8/a.png"), &ltx), Err(TargetError::IpLiteral));
        assert!(check_url(&u("https://example.com/a.png"), &ltx).is_ok());
        assert!(matches!(check_url(&u("ftp://example.com/a"), &ltx), Err(TargetError::Scheme(_))));
        let d = TargetPolicy::default();
        assert!(matches!(check_url(&u("http://127.0.0.1/"), &d), Err(TargetError::Private(_))));
        assert!(matches!(check_url(&u("http://[::1]/"), &d), Err(TargetError::Private(_))));
        assert!(check_url(&u("http://8.8.8.8/"), &d).is_ok());
    }

    #[tokio::test]
    async fn localhost_name_is_private() {
        let u = Url::parse("http://localhost:9/x").unwrap();
        let r = resolve_target(&u, &TargetPolicy::default()).await;
        assert!(matches!(r, Err(TargetError::Private(_)) | Err(TargetError::Resolve(_))));
        let open = TargetPolicy { allow_private: true, ..TargetPolicy::default() };
        if let Ok(a) = resolve_target(&u, &open).await {
            assert!(a.iter().all(|a| a.ip().is_loopback()));
        }
    }
}
