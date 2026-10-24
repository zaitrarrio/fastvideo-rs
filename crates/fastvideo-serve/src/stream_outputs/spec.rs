//! The `output` request object (docs/serve/stream-outputs.md "API"): parsing,
//! validation and redaction. Pure (no I/O); DNS checks happen when the
//! output starts.
//!
//! ```json
//! "output": {"kind": "cloudflare" | "cloudflare-rtmps" | "cloudflare-srt" | "whip" | "rtmps" | "srt",
//!            "url": "…", "token": "…", "user": "…", "stream_key": "…",
//!            "stream_id": "…", "passphrase": "…", "codec": "h264" | "vp8" | "vp9",
//!            "target": "cloudflare" | "mediamtx", "insecure_skip_verify": false}
//! ```
//!
//! Credentials (`token`, `stream_key`, `passphrase`, a URL's userinfo, query
//! or RTMP key segment) live only in [`OutputSpec`], whose `Debug` redacts
//! them; [`OutputSpec::view`] is what responses, logs and messages show:
//! the URL without credentials and `"credentials": "set"`.

use std::net::IpAddr;

use fastvideo_protocol::ApiError;
use fastvideo_serve_kit::net::is_forbidden_ip;
use fastvideo_webrtc::whip::WhipTarget;
use fastvideo_webrtc::writer::VideoCodec;
use serde::Deserialize;
use serde_json::{json, Value};
use url::{Host, Url};

/// Where a session's output goes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum OutputKind {
    /// The default: a Cloudflare Stream live input per stream, published
    /// over WHIP (WebRTC playback over WHEP).
    Cloudflare,
    /// A Cloudflare live input published over RTMPS (HLS/DASH playback).
    CloudflareRtmps,
    /// A Cloudflare live input published over SRT (HLS/DASH playback).
    CloudflareSrt,
    /// Any WHIP endpoint (MediaMTX, Cloudflare WHIP URL, an SFU).
    Whip,
    /// Any RTMP or RTMPS ingest (`rtmp://`, `rtmps://`).
    Rtmps,
    /// Any SRT listener (`srt://host:port`).
    Srt,
}

impl OutputKind {
    pub const ALL: [OutputKind; 6] = [
        OutputKind::Cloudflare,
        OutputKind::CloudflareRtmps,
        OutputKind::CloudflareSrt,
        OutputKind::Whip,
        OutputKind::Rtmps,
        OutputKind::Srt,
    ];

    pub fn as_str(self) -> &'static str {
        match self {
            OutputKind::Cloudflare => "cloudflare",
            OutputKind::CloudflareRtmps => "cloudflare-rtmps",
            OutputKind::CloudflareSrt => "cloudflare-srt",
            OutputKind::Whip => "whip",
            OutputKind::Rtmps => "rtmps",
            OutputKind::Srt => "srt",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        Some(match s.trim().to_ascii_lowercase().as_str() {
            "cloudflare" | "cloudflare-whip" | "cloudflare-webrtc" => OutputKind::Cloudflare,
            "cloudflare-rtmps" | "cloudflare-rtmp" => OutputKind::CloudflareRtmps,
            "cloudflare-srt" => OutputKind::CloudflareSrt,
            "whip" => OutputKind::Whip,
            "rtmps" | "rtmp" => OutputKind::Rtmps,
            "srt" => OutputKind::Srt,
            _ => return None,
        })
    }

    /// A live input on the server's Cloudflare account.
    pub fn is_cloudflare(self) -> bool {
        matches!(
            self,
            OutputKind::Cloudflare | OutputKind::CloudflareRtmps | OutputKind::CloudflareSrt
        )
    }

    /// Published over WHIP (else through the ffmpeg sink).
    pub fn is_whip(self) -> bool {
        matches!(self, OutputKind::Cloudflare | OutputKind::Whip)
    }

    /// What a viewer plays: `whep` for WebRTC kinds, `hls` for RTMPS/SRT.
    pub fn playback_format(self) -> &'static str {
        if self.is_whip() {
            "whep"
        } else {
            "hls"
        }
    }
}

/// The request object, as sent (unknown fields refused).
#[derive(Clone, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OutputRequest {
    #[serde(default)]
    pub kind: Option<String>,
    #[serde(default)]
    pub url: Option<String>,
    /// WHIP bearer token (or the Basic password with `user`).
    #[serde(default)]
    pub token: Option<String>,
    /// WHIP Basic auth user (MediaMTX).
    #[serde(default)]
    pub user: Option<String>,
    /// RTMP(S): appended to `url` as its last path segment.
    #[serde(default)]
    pub stream_key: Option<String>,
    /// SRT `streamid`.
    #[serde(default)]
    pub stream_id: Option<String>,
    /// SRT passphrase (10-79 characters).
    #[serde(default)]
    pub passphrase: Option<String>,
    /// `h264` (default) | `vp8` | `vp9`: WHIP kinds only.
    #[serde(default)]
    pub codec: Option<String>,
    /// WHIP endpoint kind: `cloudflare` | `mediamtx` (default: from the URL).
    #[serde(default)]
    pub target: Option<String>,
    /// `rtmps://` only, tests only: skip the server certificate check.
    /// Refused unless `[outputs] allow_private` (`FV_OUTPUTS_ALLOW_PRIVATE=1`).
    #[serde(default)]
    pub insecure_skip_verify: Option<bool>,
}

/// A validated output. `Debug` never prints credentials.
#[derive(Clone, PartialEq)]
pub struct OutputSpec {
    pub kind: OutputKind,
    /// The destination with its credentials (`None` for Cloudflare kinds
    /// until the live input exists).
    pub url: Option<Url>,
    pub token: Option<String>,
    pub user: Option<String>,
    pub stream_key: Option<String>,
    pub stream_id: Option<String>,
    pub passphrase: Option<String>,
    pub codec: VideoCodec,
    pub whip_target: Option<WhipTarget>,
    /// RTMPS without certificate verification (tests against a self-signed
    /// MediaMTX; only with `allow_private`).
    pub insecure_skip_verify: bool,
}

impl std::fmt::Debug for OutputSpec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "OutputSpec({})", self.view())
    }
}

/// Where a request may send media.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct UrlPolicy {
    /// Loopback / private / link-local destinations allowed
    /// (`[outputs] allow_private`, for MediaMTX tests).
    pub allow_private: bool,
}

fn bad(param: &str, msg: impl Into<String>) -> ApiError {
    ApiError::invalid_param(param, msg.into())
}

fn nonempty(v: &Option<String>) -> Option<String> {
    v.as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_owned)
}

impl OutputSpec {
    /// Parses and validates the `output` object.
    pub fn parse(v: &Value, policy: UrlPolicy) -> Result<Self, ApiError> {
        if !v.is_object() {
            return Err(bad("output", "`output` is an object: {\"kind\": \"cloudflare\" | \"whip\" | \"rtmps\" | \"srt\", …}"));
        }
        // serde's message names the unknown field, never a value.
        let r: OutputRequest =
            serde_json::from_value(v.clone()).map_err(|e| bad("output", e.to_string()))?;
        Self::from_request(r, policy)
    }

    pub fn from_request(r: OutputRequest, policy: UrlPolicy) -> Result<Self, ApiError> {
        let kind = match nonempty(&r.kind) {
            None => OutputKind::Cloudflare,
            Some(k) => OutputKind::parse(&k).ok_or_else(|| {
                bad("output.kind", "unknown output kind (cloudflare | cloudflare-rtmps | cloudflare-srt | whip | rtmps | srt)")
            })?,
        };
        let codec = match nonempty(&r.codec) {
            None => VideoCodec::H264,
            Some(c) => c
                .parse::<VideoCodec>()
                .map_err(|e| bad("output.codec", e))?,
        };
        if codec != VideoCodec::H264 && !kind.is_whip() {
            return Err(bad(
                "output.codec",
                format!(
                    "{} carries H.264 only (Cloudflare Stream Live: H.264 + AAC)",
                    kind.as_str()
                ),
            ));
        }
        let (token, user, stream_key, stream_id, passphrase) = (
            nonempty(&r.token),
            nonempty(&r.user),
            nonempty(&r.stream_key),
            nonempty(&r.stream_id),
            nonempty(&r.passphrase),
        );
        let whip_target = match nonempty(&r.target) {
            None => None,
            Some(t) if kind == OutputKind::Whip => Some(
                t.parse::<WhipTarget>()
                    .map_err(|e| bad("output.target", e.to_string()))?,
            ),
            Some(_) => return Err(bad("output.target", "`target` applies to kind whip only")),
        };
        let insecure_skip_verify = r.insecure_skip_verify.unwrap_or(false);
        let mut spec = OutputSpec {
            kind,
            url: None,
            token,
            user,
            stream_key,
            stream_id,
            passphrase,
            codec,
            whip_target,
            insecure_skip_verify,
        };
        let allowed = |what: &str, ok: bool| {
            if ok {
                Ok(())
            } else {
                Err(bad(
                    &format!("output.{what}"),
                    format!("`{what}` does not apply to kind {}", kind.as_str()),
                ))
            }
        };
        if insecure_skip_verify && kind != OutputKind::Rtmps {
            return Err(bad(
                "output.insecure_skip_verify",
                format!(
                    "`insecure_skip_verify` does not apply to kind {} (rtmps:// URLs only)",
                    kind.as_str()
                ),
            ));
        }
        match kind {
            OutputKind::Cloudflare | OutputKind::CloudflareRtmps | OutputKind::CloudflareSrt => {
                if nonempty(&r.url).is_some() {
                    return Err(bad(
                        "output.url",
                        "a Cloudflare output creates its own live input on the server's account; for your own destination use kind whip, rtmps or srt",
                    ));
                }
                for (name, set) in [
                    ("token", spec.token.is_some()),
                    ("user", spec.user.is_some()),
                    ("stream_key", spec.stream_key.is_some()),
                    ("stream_id", spec.stream_id.is_some()),
                    ("passphrase", spec.passphrase.is_some()),
                ] {
                    allowed(name, !set)?;
                }
            }
            OutputKind::Whip => {
                let u = parse_url(&r.url, &["http", "https"], policy)?;
                allowed("stream_key", spec.stream_key.is_none())?;
                allowed("stream_id", spec.stream_id.is_none())?;
                allowed("passphrase", spec.passphrase.is_none())?;
                spec.url = Some(u);
            }
            OutputKind::Rtmps => {
                let u = parse_url(&r.url, &["rtmp", "rtmps"], policy)?;
                allowed("token", spec.token.is_none())?;
                allowed("user", spec.user.is_none())?;
                allowed("stream_id", spec.stream_id.is_none())?;
                allowed("passphrase", spec.passphrase.is_none())?;
                if let Some(k) = &spec.stream_key {
                    if k.contains(['/', '?', '#', ' ']) {
                        return Err(bad("output.stream_key", "a stream key is one path segment"));
                    }
                } else if u
                    .path()
                    .trim_matches('/')
                    .split('/')
                    .filter(|s| !s.is_empty())
                    .count()
                    < 2
                {
                    return Err(bad(
                        "output.stream_key",
                        "give `stream_key`, or a URL ending in /<app>/<key>",
                    ));
                }
                if insecure_skip_verify {
                    if u.scheme() != "rtmps" {
                        return Err(bad(
                            "output.insecure_skip_verify",
                            "`insecure_skip_verify` applies to rtmps:// URLs only",
                        ));
                    }
                    if !policy.allow_private {
                        return Err(bad(
                            "output.insecure_skip_verify",
                            "certificate verification can only be skipped on a test server ([outputs] allow_private / FV_OUTPUTS_ALLOW_PRIVATE=1)",
                        ));
                    }
                }
                spec.url = Some(u);
            }
            OutputKind::Srt => {
                let u = parse_url(&r.url, &["srt"], policy)?;
                if u.port().is_none() {
                    return Err(bad(
                        "output.url",
                        "an SRT URL needs a port (srt://host:port)",
                    ));
                }
                allowed("token", spec.token.is_none())?;
                allowed("user", spec.user.is_none())?;
                allowed("stream_key", spec.stream_key.is_none())?;
                if let Some(p) = &spec.passphrase {
                    if !(10..=79).contains(&p.chars().count()) {
                        return Err(bad(
                            "output.passphrase",
                            "an SRT passphrase is 10 to 79 characters",
                        ));
                    }
                }
                // ffmpeg takes them verbatim from the URL query.
                for (name, v) in [
                    ("passphrase", &spec.passphrase),
                    ("stream_id", &spec.stream_id),
                ] {
                    if v.as_deref()
                        .is_some_and(|v| !fastvideo_media::sink::srt_value_ok(v))
                    {
                        return Err(bad(
                            &format!("output.{name}"),
                            "SRT values cannot contain spaces, &, #, ? or %",
                        ));
                    }
                }
                spec.url = Some(u);
            }
        }
        Ok(spec)
    }

    /// The legacy native-stream fields (`whip_url`, `whip_token`,
    /// `whip_user`, `whip_target`) as an `output` of kind `whip`.
    pub fn from_legacy(
        whip_url: &str,
        token: Option<String>,
        user: Option<String>,
        target: Option<String>,
        policy: UrlPolicy,
    ) -> Result<Self, ApiError> {
        let r = OutputRequest {
            kind: Some("whip".into()),
            url: Some(whip_url.to_owned()),
            token,
            user,
            target,
            ..OutputRequest::default()
        };
        Self::from_request(r, policy).map_err(|mut e| {
            // Name the field the caller used.
            if let Some(p) = &e.param {
                e.param = Some(match p.as_str() {
                    "output.url" => "whip_url".into(),
                    "output.target" => "whip_target".into(),
                    other => other.to_owned(),
                });
            }
            e
        })
    }

    /// Whether any credential is set (a token, key, passphrase, or URL
    /// userinfo / query).
    pub fn has_credentials(&self) -> bool {
        self.token.is_some()
            || self.stream_key.is_some()
            || self.passphrase.is_some()
            || self.stream_id.is_some()
            || self.url.as_ref().is_some_and(|u| {
                !u.username().is_empty() || u.password().is_some() || u.query().is_some()
            })
            || (self.kind == OutputKind::Rtmps && self.stream_key.is_none())
    }

    /// The response / log shape: no credentials.
    pub fn view(&self) -> Value {
        let mut v = json!({
            "kind": self.kind.as_str(),
            "url": self.url.as_ref().map(|u| redact_url(u, self.kind)),
            "credentials": if self.has_credentials() { "set" } else { "none" },
            "playback_format": self.kind.playback_format(),
        });
        if self.kind.is_whip() {
            v["codec"] = json!(codec_name(self.codec));
        }
        if self.insecure_skip_verify {
            v["tls_verify"] = json!(false);
        }
        if let Some(t) = self.whip_target {
            v["target"] = json!(match t {
                WhipTarget::Cloudflare => "cloudflare",
                WhipTarget::Mediamtx => "mediamtx",
            });
        }
        v
    }

    /// The RTMP URL to publish to: `url` plus `/stream_key`.
    pub fn rtmp_publish_url(&self) -> Option<String> {
        let u = self.url.as_ref()?;
        Some(match &self.stream_key {
            Some(k) => format!("{}/{}", u.as_str().trim_end_matches('/'), k),
            None => u.as_str().to_owned(),
        })
    }

    /// The host to check against private ranges when the output starts.
    pub fn host(&self) -> Option<(String, u16)> {
        let u = self.url.as_ref()?;
        let port = u.port_or_known_default().or(match u.scheme() {
            "rtmp" => Some(1935),
            "rtmps" => Some(443),
            _ => None,
        })?;
        Some((u.host_str()?.trim_matches(['[', ']']).to_owned(), port))
    }
}

/// `h264` | `vp8` | `vp9`.
pub fn codec_name(c: VideoCodec) -> &'static str {
    match c {
        VideoCodec::H264 => "h264",
        VideoCodec::Vp8 => "vp8",
        VideoCodec::Vp9 => "vp9",
    }
}

fn parse_url(raw: &Option<String>, schemes: &[&str], policy: UrlPolicy) -> Result<Url, ApiError> {
    let raw = nonempty(raw).ok_or_else(|| bad("output.url", "this kind needs `url`"))?;
    // url::ParseError messages never echo the input.
    let u = Url::parse(&raw).map_err(|e| bad("output.url", format!("not a URL: {e}")))?;
    if !schemes.contains(&u.scheme()) {
        return Err(bad(
            "output.url",
            format!("the URL scheme must be one of {}", schemes.join(", ")),
        ));
    }
    match u.host() {
        None => return Err(bad("output.url", "the URL has no host")),
        Some(Host::Ipv4(ip)) => check_ip(IpAddr::V4(ip), policy)?,
        Some(Host::Ipv6(ip)) => check_ip(IpAddr::V6(ip), policy)?,
        Some(Host::Domain(d)) => {
            // `srt://` and `rtmp://` are not special schemes: their hosts
            // are opaque, so an IP literal arrives here as a name.
            if let Ok(ip) = d.trim_matches(['[', ']']).parse::<IpAddr>() {
                check_ip(ip, policy)?;
            }
            let d = d.to_ascii_lowercase();
            if !policy.allow_private
                && (d == "localhost"
                    || d.ends_with(".localhost")
                    || d.ends_with(".local")
                    || d.ends_with(".internal"))
            {
                return Err(bad(
                    "output.url",
                    "private and loopback destinations are refused ([outputs] allow_private)",
                ));
            }
        }
    }
    Ok(u)
}

/// Refuses a non-public address unless the policy allows it.
pub fn check_ip(ip: IpAddr, policy: UrlPolicy) -> Result<(), ApiError> {
    if !policy.allow_private && is_forbidden_ip(ip) {
        return Err(bad(
            "output.url",
            "private and loopback destinations are refused ([outputs] allow_private)",
        ));
    }
    Ok(())
}

/// A URL without credentials: no userinfo, no query or fragment; an RTMP
/// URL's last path segment (the stream key) and any long opaque path
/// segment (Cloudflare's WHIP key) become `<key>`.
pub fn redact_url(u: &Url, kind: OutputKind) -> String {
    let mut r = format!("{}://{}", u.scheme(), u.host_str().unwrap_or_default());
    if let Some(p) = u.port() {
        r.push_str(&format!(":{p}"));
    }
    let segs: Vec<&str> = u
        .path_segments()
        .map(|s| s.filter(|x| !x.is_empty()).collect())
        .unwrap_or_default();
    let n = segs.len();
    for (i, s) in segs.iter().enumerate() {
        let key = (kind == OutputKind::Rtmps && i + 1 == n && n >= 2) || looks_like_key(s);
        r.push('/');
        r.push_str(if key { "<key>" } else { s });
    }
    r
}

/// A path segment that looks like a credential: 20+ characters of letters
/// and digits with both present (Cloudflare's `…/<key>/webRTC/publish`).
pub fn looks_like_key(s: &str) -> bool {
    s.len() >= 20
        && s.chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
        && s.chars().any(|c| c.is_ascii_digit())
        && s.chars().any(|c| c.is_ascii_alphabetic())
}

/// Removes every occurrence of `secrets` from `text` (for errors that might
/// quote a URL).
pub fn scrub(text: &str, secrets: &[&str]) -> String {
    let mut out = text.to_owned();
    for s in secrets.iter().filter(|s| s.len() >= 4) {
        out = out.replace(*s, "<redacted>");
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(v: Value) -> Result<OutputSpec, ApiError> {
        OutputSpec::parse(&v, UrlPolicy::default())
    }

    #[test]
    fn kinds_and_defaults() {
        let s = parse(json!({})).unwrap();
        assert_eq!(
            (s.kind, s.codec),
            (OutputKind::Cloudflare, VideoCodec::H264)
        );
        assert_eq!(
            s.view(),
            json!({"kind": "cloudflare", "url": null, "credentials": "none", "playback_format": "whep", "codec": "h264"})
        );
        let s = parse(json!({"kind": "cloudflare", "codec": "vp8"})).unwrap();
        assert_eq!(s.codec, VideoCodec::Vp8);
        assert_eq!(
            parse(json!({"kind": "cloudflare-srt"}))
                .unwrap()
                .kind
                .playback_format(),
            "hls"
        );
        assert_eq!(
            parse(json!({"kind": "RTMP", "url": "rtmp://a.example/live/k"}))
                .unwrap()
                .kind,
            OutputKind::Rtmps
        );
        for k in OutputKind::ALL {
            assert_eq!(OutputKind::parse(k.as_str()), Some(k));
        }
        let e = parse(json!({"kind": "hls"})).unwrap_err();
        assert_eq!(e.param.as_deref(), Some("output.kind"));
        let e =
            parse(json!({"kind": "whip", "url": "https://a.example/w", "bogus": 1})).unwrap_err();
        assert!(e.message.contains("bogus"), "{}", e.message);
        assert!(parse(json!("cloudflare")).is_err());
    }

    #[test]
    fn scheme_and_field_rules_per_kind() {
        let p = |v| parse(v).unwrap_err().param.unwrap_or_default();
        assert_eq!(p(json!({"kind": "whip"})), "output.url");
        assert_eq!(
            p(json!({"kind": "whip", "url": "rtmp://a.example/live/k"})),
            "output.url"
        );
        assert_eq!(
            p(json!({"kind": "rtmps", "url": "https://a.example/live/k"})),
            "output.url"
        );
        assert_eq!(
            p(json!({"kind": "srt", "url": "srt://a.example"})),
            "output.url"
        );
        assert_eq!(
            p(json!({"kind": "srt", "url": "srt://a.example:778", "passphrase": "short"})),
            "output.passphrase"
        );
        assert_eq!(
            p(
                json!({"kind": "srt", "url": "srt://a.example:778", "passphrase": "with space 1234"})
            ),
            "output.passphrase"
        );
        assert_eq!(
            p(json!({"kind": "srt", "url": "srt://a.example:778", "stream_id": "a&b"})),
            "output.stream_id"
        );
        assert_eq!(
            p(json!({"kind": "rtmps", "url": "rtmps://a.example/live"})),
            "output.stream_key"
        );
        assert_eq!(
            p(json!({"kind": "rtmps", "url": "rtmps://a.example/live", "stream_key": "a/b"})),
            "output.stream_key"
        );
        assert_eq!(
            p(json!({"kind": "rtmps", "url": "rtmps://a.example/live/k", "codec": "vp8"})),
            "output.codec"
        );
        assert_eq!(
            p(json!({"kind": "cloudflare", "url": "https://a.example/w"})),
            "output.url"
        );
        assert_eq!(
            p(json!({"kind": "cloudflare-srt", "passphrase": "0123456789ab"})),
            "output.passphrase"
        );
        assert_eq!(
            p(json!({"kind": "srt", "url": "srt://a.example:9", "token": "t"})),
            "output.token"
        );
        assert_eq!(
            p(json!({"kind": "rtmps", "url": "rtmps://a.example/live/k", "target": "mediamtx"})),
            "output.target"
        );
        assert_eq!(
            p(json!({"kind": "whip", "url": "https://a.example/w", "codec": "av1"})),
            "output.codec"
        );
        let s = parse(json!({"kind": "srt", "url": "srt://a.example:778", "stream_id": "id1", "passphrase": "0123456789ab"})).unwrap();
        assert_eq!(s.host(), Some(("a.example".into(), 778)));
        let s = parse(json!({"kind": "rtmps", "url": "rtmps://live.example.com:443/live/", "stream_key": "k3y"})).unwrap();
        assert_eq!(
            s.rtmp_publish_url().unwrap(),
            "rtmps://live.example.com:443/live/k3y"
        );
        assert_eq!(s.host(), Some(("live.example.com".into(), 443)));
        let s = parse(json!({"kind": "whip", "url": "https://a.example/w", "codec": "VP9", "target": "cloudflare"})).unwrap();
        assert_eq!(
            (s.codec, s.whip_target),
            (VideoCodec::Vp9, Some(WhipTarget::Cloudflare))
        );
    }

    #[test]
    fn private_and_loopback_targets_need_the_config() {
        for u in [
            "http://127.0.0.1:8889/live/whip",
            "http://10.0.0.5/whip",
            "http://[::1]:8889/w",
            "http://localhost:8889/w",
            "http://169.254.169.254/latest",
            "http://media.local/w",
        ] {
            let e = parse(json!({"kind": "whip", "url": u})).unwrap_err();
            assert_eq!(e.param.as_deref(), Some("output.url"), "{u}");
            assert!(e.message.contains("allow_private"), "{u}");
            OutputSpec::parse(
                &json!({"kind": "whip", "url": u}),
                UrlPolicy {
                    allow_private: true,
                },
            )
            .unwrap();
        }
        let e = parse(json!({"kind": "srt", "url": "srt://192.168.1.2:9000"})).unwrap_err();
        assert_eq!(e.param.as_deref(), Some("output.url"));
        parse(json!({"kind": "srt", "url": "srt://203.0.113.9:9000"})).unwrap_err(); // documentation range
        parse(json!({"kind": "srt", "url": "srt://8.8.8.8:9000"})).unwrap();
    }

    #[test]
    fn insecure_skip_verify_is_for_test_servers_only() {
        let req = json!({"kind": "rtmps", "url": "rtmps://127.0.0.1:1937/live/k", "insecure_skip_verify": true});
        let private = UrlPolicy {
            allow_private: true,
        };
        // Off by default; on only with allow_private, and shown.
        let s = OutputSpec::parse(
            &json!({"kind": "rtmps", "url": "rtmps://live.example.com/live/k"}),
            private,
        )
        .unwrap();
        assert!(!s.insecure_skip_verify);
        assert!(s.view().get("tls_verify").is_none());
        let s = OutputSpec::parse(&req, private).unwrap();
        assert!(s.insecure_skip_verify);
        assert_eq!(s.view()["tls_verify"], false);
        let pub_req = json!({"kind": "rtmps", "url": "rtmps://live.example.com/live/k", "insecure_skip_verify": true});
        let e = parse(pub_req).unwrap_err();
        assert_eq!(e.param.as_deref(), Some("output.insecure_skip_verify"));
        assert!(e.message.contains("allow_private"), "{}", e.message);
        // rtmps:// only: nothing to skip on rtmp://, SRT, WHIP or Cloudflare.
        for v in [
            json!({"kind": "rtmps", "url": "rtmp://127.0.0.1/live/k", "insecure_skip_verify": true}),
            json!({"kind": "whip", "url": "https://127.0.0.1/w", "insecure_skip_verify": true}),
            json!({"kind": "srt", "url": "srt://127.0.0.1:9000", "insecure_skip_verify": true}),
            json!({"kind": "cloudflare-rtmps", "insecure_skip_verify": true}),
        ] {
            let e = OutputSpec::parse(&v, private).unwrap_err();
            assert_eq!(
                e.param.as_deref(),
                Some("output.insecure_skip_verify"),
                "{v}"
            );
        }
        // `false` is accepted anywhere.
        parse(json!({"kind": "cloudflare-rtmps", "insecure_skip_verify": false})).unwrap();
    }

    #[test]
    fn credentials_are_never_shown() {
        let secret_key = "sk_live_0123456789abcdefghij";
        let s = parse(json!({"kind": "rtmps", "url": "rtmps://live.example.com:443/live/", "stream_key": secret_key})).unwrap();
        let shown = format!("{} {:?}", s.view(), s);
        assert!(!shown.contains(secret_key), "{shown}");
        assert_eq!(s.view()["url"], "rtmps://live.example.com:443/live");
        assert_eq!(s.view()["credentials"], "set");
        // The key inside the URL itself.
        let s = parse(
            json!({"kind": "rtmps", "url": format!("rtmp://live.example.com/app/{secret_key}")}),
        )
        .unwrap();
        assert_eq!(s.view()["url"], "rtmp://live.example.com/app/<key>");
        assert_eq!(s.view()["credentials"], "set");
        // WHIP: userinfo, query and a key-like path segment.
        let s = parse(json!({"kind": "whip", "url": "https://u:pw@customer-ab12.example.com/d41d8cd98f00b204e9800998ecf8427e/webRTC/publish?tok=1", "token": "bearer-secret"})).unwrap();
        let shown = format!("{} {:?}", s.view(), s);
        for x in [
            "pw",
            "bearer-secret",
            "d41d8cd98f00b204e9800998ecf8427e",
            "tok=1",
        ] {
            assert!(!shown.contains(x), "{x} in {shown}");
        }
        assert_eq!(
            s.view()["url"],
            "https://customer-ab12.example.com/<key>/webRTC/publish"
        );
        // SRT passphrase and stream id.
        let s = parse(json!({"kind": "srt", "url": "srt://live.example.com:778", "stream_id": "abcdef0123456789", "passphrase": "pass-phrase-123"})).unwrap();
        let shown = format!("{} {:?}", s.view(), s);
        assert!(
            !shown.contains("pass-phrase-123") && !shown.contains("abcdef0123456789"),
            "{shown}"
        );
        assert_eq!(s.view()["url"], "srt://live.example.com:778");
        // A plain MediaMTX URL has none.
        let s = OutputSpec::parse(
            &json!({"kind": "whip", "url": "http://127.0.0.1:8889/cam/whip"}),
            UrlPolicy {
                allow_private: true,
            },
        )
        .unwrap();
        assert_eq!(s.view()["credentials"], "none");
        assert_eq!(
            scrub("POST https://h/sk_live_x failed", &["sk_live_x"]),
            "POST https://h/<redacted> failed"
        );
    }

    #[test]
    fn legacy_fields_map_to_whip() {
        let s = OutputSpec::from_legacy(
            "https://relay.example/live/whip",
            Some("t".into()),
            None,
            Some("mediamtx".into()),
            UrlPolicy::default(),
        )
        .unwrap();
        assert_eq!(
            (s.kind, s.token.as_deref(), s.whip_target),
            (OutputKind::Whip, Some("t"), Some(WhipTarget::Mediamtx))
        );
        let e = OutputSpec::from_legacy("ftp://x.example/", None, None, None, UrlPolicy::default())
            .unwrap_err();
        assert_eq!(e.param.as_deref(), Some("whip_url"));
        let e = OutputSpec::from_legacy(
            "https://x.example/w",
            None,
            None,
            Some("rtmp".into()),
            UrlPolicy::default(),
        )
        .unwrap_err();
        assert_eq!(e.param.as_deref(), Some("whip_target"));
    }
}
