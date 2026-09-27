//! WHIP publisher (design §5.8; streaming-refs §1.9; deploy §5.1).
//!
//! We are the WebRTC **client**: POST a complete offer, get the answer, keep
//! the resource URL, DELETE it on teardown. No inbound port is needed, which
//! is what makes Runpod serverless streaming work. Behaviour follows strobe's
//! proven publisher (`strobe/src/strobe/whip.py`):
//!
//! - `Content-Type: application/sdp`; the body is the **full** offer
//!   (non-trickle): host candidates plus a STUN srflx candidate gathered
//!   before the POST, and `a=end-of-candidates`.
//! - H.264 is offered first and VP8 not at all (Cloudflare plays VP8 black).
//! - 200 or 201 carries the answer; anything else is an error with the body.
//! - `Location` may be relative and is resolved against the (final) POST URL
//!   per RFC 3986; it is kept for `DELETE`.
//! - Auth: `Basic user:token` (MediaMTX internal auth) or `Bearer token`
//!   (Cloudflare).
//! - 30 s timeout; teardown is best effort.
//! - A video-only session offers only a video m-line (design §5.3).
//! - The config names the [`WhipTarget`], which picks the [`EncodeProfile`]
//!   (design §0 decision 2): Cloudflare caps at 1280x720 and H.264 level
//!   3.1; MediaMTX and other relays take native resolution at level 4.0. The
//!   offer advertises that level and the media layer reads
//!   [`WhipConfig::profile`] to scale and encode.
//!
//! There is no PATCH (trickle) and no ICE restart, as in strobe.

use std::time::Duration;

use serde::{Deserialize, Serialize};
use url::Url;

use crate::profile::EncodeProfile;
use crate::WebrtcError;

/// Default POST/DELETE timeout (strobe `whip_timeout_s=30`).
pub const DEFAULT_TIMEOUT: Duration = Duration::from_secs(30);

/// WHIP endpoint credentials.
#[derive(Clone, Default, PartialEq, Eq)]
pub enum WhipAuth {
    #[default]
    None,
    /// `Authorization: Bearer <token>` (Cloudflare Stream / Realtime).
    Bearer(String),
    /// `Authorization: Basic base64(user:password)` (MediaMTX).
    Basic { user: String, password: String },
}

impl std::fmt::Debug for WhipAuth {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Never log secrets.
        match self {
            WhipAuth::None => f.write_str("None"),
            WhipAuth::Bearer(_) => f.write_str("Bearer(<redacted>)"),
            WhipAuth::Basic { user, .. } => write!(f, "Basic({user}:<redacted>)"),
        }
    }
}

impl WhipAuth {
    /// strobe's rule: a user means Basic with the token as the password,
    /// otherwise a non-empty token means Bearer.
    pub fn from_user_token(user: &str, token: &str) -> Self {
        if !user.is_empty() {
            WhipAuth::Basic {
                user: user.to_string(),
                password: token.to_string(),
            }
        } else if !token.is_empty() {
            WhipAuth::Bearer(token.to_string())
        } else {
            WhipAuth::None
        }
    }

    /// The `Authorization` header value, if any.
    pub fn header_value(&self) -> Option<String> {
        match self {
            WhipAuth::None => None,
            WhipAuth::Bearer(t) => Some(format!("Bearer {t}")),
            WhipAuth::Basic { user, password } => Some(format!(
                "Basic {}",
                base64(format!("{user}:{password}").as_bytes())
            )),
        }
    }
}

fn base64(input: &[u8]) -> String {
    const T: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(input.len().div_ceil(3) * 4);
    for c in input.chunks(3) {
        let b = [c[0], *c.get(1).unwrap_or(&0), *c.get(2).unwrap_or(&0)];
        let n = (b[0] as u32) << 16 | (b[1] as u32) << 8 | b[2] as u32;
        out.push(T[(n >> 18) as usize & 63] as char);
        out.push(T[(n >> 12) as usize & 63] as char);
        out.push(if c.len() > 1 {
            T[(n >> 6) as usize & 63] as char
        } else {
            '='
        });
        out.push(if c.len() > 2 {
            T[n as usize & 63] as char
        } else {
            '='
        });
    }
    out
}

/// The kind of WHIP endpoint, which fixes the encode profile.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum WhipTarget {
    /// Cloudflare Stream / Realtime: 720p cap, H.264 level 3.1.
    Cloudflare,
    /// MediaMTX (self-hosted relay): native resolution, level 4.0.
    #[serde(alias = "peer")]
    Mediamtx,
}

impl WhipTarget {
    pub fn profile(self) -> EncodeProfile {
        match self {
            WhipTarget::Cloudflare => EncodeProfile::CLOUDFLARE,
            WhipTarget::Mediamtx => EncodeProfile::NATIVE,
        }
    }

    /// Best guess from the endpoint URL when the job does not say:
    /// Cloudflare hosts (`*.cloudflare.com`, `*.cloudflarestream.com`)
    /// get the Cloudflare profile, anything else native.
    pub fn guess(url: &Url) -> Self {
        let host = url.host_str().unwrap_or_default().to_ascii_lowercase();
        let cf = ["cloudflare.com", "cloudflarestream.com"]
            .iter()
            .any(|d| host == *d || host.ends_with(&format!(".{d}")));
        if cf {
            WhipTarget::Cloudflare
        } else {
            WhipTarget::Mediamtx
        }
    }
}

impl std::str::FromStr for WhipTarget {
    type Err = WebrtcError;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_ascii_lowercase().as_str() {
            "cloudflare" => Ok(WhipTarget::Cloudflare),
            "mediamtx" | "peer" => Ok(WhipTarget::Mediamtx),
            other => Err(WebrtcError::Config(format!(
                "unknown WHIP target {other:?} (cloudflare | mediamtx)"
            ))),
        }
    }
}

/// Where to publish.
#[derive(Debug, Clone)]
pub struct WhipConfig {
    pub url: Url,
    pub auth: WhipAuth,
    pub timeout: Duration,
    /// Selects the encode profile (design §0 decision 2).
    pub target: WhipTarget,
}

impl WhipConfig {
    /// The target is guessed from the URL; set it with [`Self::with_target`]
    /// when the job names it.
    pub fn new(url: Url, auth: WhipAuth) -> Self {
        let target = WhipTarget::guess(&url);
        WhipConfig {
            url,
            auth,
            timeout: DEFAULT_TIMEOUT,
            target,
        }
    }

    pub fn with_target(mut self, target: WhipTarget) -> Self {
        self.target = target;
        self
    }

    /// Resolution cap and H.264 level the media layer must encode to.
    pub fn profile(&self) -> EncodeProfile {
        self.target.profile()
    }
}

/// Resolve a `Location` header against the request URL (RFC 3986 §5).
pub fn resolve_location(base: &Url, location: &str) -> Result<Url, WebrtcError> {
    base.join(location.trim())
        .map_err(|e| WebrtcError::Whip(format!("bad Location {location:?}: {e}")))
}

/// The answer to a WHIP POST.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WhipAnswer {
    pub status: u16,
    pub sdp: String,
    /// The session resource (absolute), for `DELETE`.
    pub resource: Option<Url>,
}

/// Classify a WHIP POST response (pure, for tests and the client).
pub fn check_post_response(
    status: u16,
    request_url: &Url,
    location: Option<&str>,
    body: String,
) -> Result<WhipAnswer, WebrtcError> {
    if status != 200 && status != 201 {
        let body: String = body.chars().take(500).collect();
        return Err(WebrtcError::WhipStatus { status, body });
    }
    if !body.trim_start().starts_with("v=0") {
        return Err(WebrtcError::Whip(format!(
            "HTTP {status} without an SDP answer"
        )));
    }
    let resource = location
        .map(|l| resolve_location(request_url, l))
        .transpose()?;
    Ok(WhipAnswer {
        status,
        sdp: body,
        resource,
    })
}

#[cfg(feature = "whip")]
pub use client::WhipClient;

#[cfg(feature = "whip")]
mod client {
    use super::*;

    /// The HTTP half of WHIP.
    #[derive(Debug, Clone)]
    pub struct WhipClient {
        http: reqwest::Client,
        cfg: WhipConfig,
    }

    impl WhipClient {
        pub fn new(cfg: WhipConfig) -> Result<Self, WebrtcError> {
            let http = reqwest::Client::builder()
                .timeout(cfg.timeout)
                .build()
                .map_err(|e| WebrtcError::Whip(format!("http client: {e}")))?;
            Ok(WhipClient { http, cfg })
        }

        pub fn config(&self) -> &WhipConfig {
            &self.cfg
        }

        fn auth(&self, rb: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
            match self.cfg.auth.header_value() {
                Some(v) => rb.header(reqwest::header::AUTHORIZATION, v),
                None => rb,
            }
        }

        /// POST the complete offer; 200/201 with an SDP body is success.
        pub async fn post_offer(&self, offer_sdp: &str) -> Result<WhipAnswer, WebrtcError> {
            let rb = self
                .http
                .post(self.cfg.url.clone())
                .header(reqwest::header::CONTENT_TYPE, "application/sdp")
                .header(reqwest::header::ACCEPT, "application/sdp")
                .body(offer_sdp.to_string());
            let resp = self.auth(rb).send().await.map_err(|e| {
                if e.is_timeout() {
                    WebrtcError::Whip(format!(
                        "POST timed out after {:?} to {}",
                        self.cfg.timeout,
                        redact(&self.cfg.url)
                    ))
                } else {
                    WebrtcError::Whip(format!("POST {}: {e}", redact(&self.cfg.url)))
                }
            })?;
            let status = resp.status().as_u16();
            let final_url = resp.url().clone();
            let location = resp
                .headers()
                .get(reqwest::header::LOCATION)
                .and_then(|v| v.to_str().ok())
                .map(str::to_string);
            let body = resp
                .text()
                .await
                .map_err(|e| WebrtcError::Whip(format!("reading answer: {e}")))?;
            check_post_response(status, &final_url, location.as_deref(), body)
        }

        /// DELETE the session resource. Returns the HTTP status.
        pub async fn delete(&self, resource: &Url) -> Result<u16, WebrtcError> {
            let resp = self
                .auth(self.http.delete(resource.clone()))
                .send()
                .await
                .map_err(|e| WebrtcError::Whip(format!("DELETE {}: {e}", redact(resource))))?;
            Ok(resp.status().as_u16())
        }
    }

    /// URLs can carry stream keys in the path (Cloudflare); log only the origin.
    fn redact(u: &Url) -> String {
        u.origin().ascii_serialization()
    }
}

#[cfg(all(feature = "whip", feature = "str0m"))]
pub use publisher::{WhipPublishOptions, WhipPublisher};

#[cfg(all(feature = "whip", feature = "str0m"))]
mod publisher {
    use super::*;
    use crate::host::{AudioLayout, OfferOptions, Peer, RtcHost};
    use crate::ice::IceServer;
    use crate::sdp::Direction;

    /// What to publish.
    #[derive(Debug, Clone)]
    pub struct WhipPublishOptions {
        /// `None`: video-only, no audio m-line at all (design §5.3).
        pub audio: Option<AudioLayout>,
        /// STUN servers probed for the srflx candidate (empty: skip).
        pub stun: Vec<IceServer>,
        pub stun_timeout: Duration,
    }

    impl Default for WhipPublishOptions {
        fn default() -> Self {
            WhipPublishOptions {
                audio: Some(AudioLayout::Stereo),
                stun: vec![IceServer::default_stun()],
                stun_timeout: Duration::from_secs(2),
            }
        }
    }

    /// A live WHIP session: the peer plus the resource to DELETE.
    #[derive(Debug)]
    pub struct WhipPublisher {
        client: WhipClient,
        resource: Option<Url>,
        peer: Option<Peer>,
        negotiated_video: Option<String>,
    }

    impl WhipPublisher {
        /// Offer, POST, apply the answer. Media flows once the returned
        /// peer reports [`crate::host::PeerEvent::Connected`]; write with
        /// [`crate::host::PeerHandle::send_video`] / `send_audio`.
        ///
        /// Design §5.2: call this **after** the first frame exists, so the
        /// endpoint never sees an empty track.
        pub async fn publish(
            host: &RtcHost,
            cfg: WhipConfig,
            opts: WhipPublishOptions,
        ) -> Result<Self, WebrtcError> {
            let profile = cfg.profile();
            let client = WhipClient::new(cfg)?;
            let srflx = if opts.stun.is_empty() {
                Vec::new()
            } else {
                host.gather_srflx(&opts.stun, opts.stun_timeout).await
            };
            let (pending, offer) = host
                .offer(OfferOptions {
                    video: Some(Direction::SendOnly),
                    audio: opts.audio.map(|l| (Direction::SendOnly, l)),
                    channels: Vec::new(),
                    srflx,
                    h264_level: profile.h264_level,
                    video_codecs: vec![crate::writer::VideoCodec::H264],
                })
                .await?;
            let answer = client.post_offer(&offer).await?;
            let negotiated_video = crate::sdp::Sdp::parse(&answer.sdp)
                .ok()
                .and_then(|s| s.negotiated_video_codec());
            tracing::info!(codec = ?negotiated_video, status = answer.status, ?profile, "whip negotiated");
            let mut me = WhipPublisher {
                client,
                resource: answer.resource.clone(),
                peer: None,
                negotiated_video,
            };
            match pending.accept_answer(&answer.sdp).await {
                Ok(peer) => {
                    me.peer = Some(peer);
                    Ok(me)
                }
                Err(e) => {
                    // The endpoint created a resource we can't use: release it.
                    let _ = me.teardown().await;
                    Err(e)
                }
            }
        }

        pub fn peer(&mut self) -> &mut Peer {
            self.peer.as_mut().expect("peer present until teardown")
        }

        pub fn resource_url(&self) -> Option<&Url> {
            self.resource.as_ref()
        }

        /// The resolution cap and H.264 level the encoder must use.
        pub fn profile(&self) -> EncodeProfile {
            self.client.config().profile()
        }

        /// The first video codec of the answer (should be `H264`).
        pub fn negotiated_video_codec(&self) -> Option<&str> {
            self.negotiated_video.as_deref()
        }

        /// DELETE the resource (best effort) and close the peer. Returns the
        /// DELETE status when there was a resource.
        pub async fn teardown(mut self) -> Result<Option<u16>, WebrtcError> {
            if let Some(p) = self.peer.take() {
                p.close();
            }
            match self.resource.take() {
                Some(r) => self.client.delete(&r).await.map(Some),
                None => Ok(None),
            }
        }
    }

    impl Drop for WhipPublisher {
        fn drop(&mut self) {
            if let Some(p) = self.peer.take() {
                p.close();
            }
            // Best effort DELETE when dropped without teardown().
            if let (Some(r), Ok(rt)) = (self.resource.take(), tokio::runtime::Handle::try_current())
            {
                let client = self.client.clone();
                rt.spawn(async move {
                    let _ = client.delete(&r).await;
                });
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn auth_headers() {
        assert_eq!(WhipAuth::from_user_token("", "").header_value(), None);
        assert_eq!(
            WhipAuth::from_user_token("", "tok")
                .header_value()
                .as_deref(),
            Some("Bearer tok")
        );
        // RFC 7617 example.
        assert_eq!(
            WhipAuth::Basic {
                user: "Aladdin".into(),
                password: "open sesame".into()
            }
            .header_value()
            .as_deref(),
            Some("Basic QWxhZGRpbjpvcGVuIHNlc2FtZQ==")
        );
        assert_eq!(
            WhipAuth::from_user_token("strobe", "pw")
                .header_value()
                .as_deref(),
            Some("Basic c3Ryb2JlOnB3")
        );
        assert_eq!(base64(b"a"), "YQ==");
        assert_eq!(base64(b"ab"), "YWI=");
        assert_eq!(base64(b""), "");
        assert!(!format!("{:?}", WhipAuth::Bearer("secret".into())).contains("secret"));
    }

    #[test]
    fn target_selects_profile() {
        let cf =
            Url::parse("https://customer-abc.cloudflarestream.com/xyz/webRTC/publish").unwrap();
        let mtx = Url::parse("http://10.0.0.5:8889/strobe/whip").unwrap();
        assert_eq!(
            WhipConfig::new(cf.clone(), WhipAuth::None).target,
            WhipTarget::Cloudflare
        );
        assert_eq!(
            WhipConfig::new(mtx.clone(), WhipAuth::None).target,
            WhipTarget::Mediamtx
        );
        assert_eq!(
            WhipTarget::guess(&Url::parse("https://notcloudflare.com/w").unwrap()),
            WhipTarget::Mediamtx
        );
        let c = WhipConfig::new(mtx, WhipAuth::None).with_target(WhipTarget::Cloudflare);
        assert_eq!(c.profile(), EncodeProfile::CLOUDFLARE);
        assert_eq!(c.profile().output_size(1344, 768), (1260, 720));
        assert_eq!(WhipTarget::Mediamtx.profile(), EncodeProfile::NATIVE);
        assert_eq!("peer".parse::<WhipTarget>().unwrap(), WhipTarget::Mediamtx);
        assert_eq!(
            " Cloudflare ".parse::<WhipTarget>().unwrap(),
            WhipTarget::Cloudflare
        );
        assert!("rtmp".parse::<WhipTarget>().is_err());
        let t: WhipTarget = serde_json::from_str("\"cloudflare\"").unwrap();
        assert_eq!(t, WhipTarget::Cloudflare);
        let t: WhipTarget = serde_json::from_str("\"peer\"").unwrap();
        assert_eq!(t, WhipTarget::Mediamtx);
    }

    #[test]
    fn location_resolution() {
        let base = Url::parse("https://sfu.example/live/whip?x=1").unwrap();
        assert_eq!(
            resolve_location(&base, "/resource/abc").unwrap().as_str(),
            "https://sfu.example/resource/abc"
        );
        assert_eq!(
            resolve_location(&base, "abc").unwrap().as_str(),
            "https://sfu.example/live/abc"
        );
        assert_eq!(
            resolve_location(&base, "../r/1").unwrap().as_str(),
            "https://sfu.example/r/1"
        );
        assert_eq!(
            resolve_location(&base, "https://other.example/r")
                .unwrap()
                .as_str(),
            "https://other.example/r"
        );
    }

    #[test]
    fn post_response_status_handling() {
        let base = Url::parse("http://h:8889/strobe/whip").unwrap();
        let ok = check_post_response(201, &base, Some("/strobe/whip/1"), "v=0\r\n".into()).unwrap();
        assert_eq!(ok.resource.unwrap().as_str(), "http://h:8889/strobe/whip/1");
        let ok = check_post_response(200, &base, None, "v=0\r\n".into()).unwrap();
        assert_eq!(ok.resource, None);
        match check_post_response(401, &base, None, "unauthorized".into()) {
            Err(WebrtcError::WhipStatus { status: 401, body }) => assert_eq!(body, "unauthorized"),
            other => panic!("{other:?}"),
        }
        assert!(matches!(
            check_post_response(202, &base, None, "v=0".into()),
            Err(WebrtcError::WhipStatus { status: 202, .. })
        ));
        assert!(matches!(
            check_post_response(201, &base, None, "".into()),
            Err(WebrtcError::Whip(_))
        ));
        let long = "x".repeat(2000);
        match check_post_response(500, &base, None, long) {
            Err(WebrtcError::WhipStatus { body, .. }) => assert_eq!(body.len(), 500),
            other => panic!("{other:?}"),
        }
    }
}
