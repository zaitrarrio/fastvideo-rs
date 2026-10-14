//! Stream outputs (docs/serve/stream-outputs.md): where a live session's
//! video goes, for native streams (`POST /fv/v1/streams`) and as an
//! optional copy of director and Reactor sessions.
//!
//! | kind | destination | transport | playback |
//! |---|---|---|---|
//! | `cloudflare` (**default**) | a Cloudflare Stream live input created per stream (`fv-<id>`) | WHIP, H.264 (`codec: vp8 \| vp9` optional) | WHEP (`webRTCPlayback`) |
//! | `cloudflare-rtmps` | the same, RTMPS ingest | ffmpeg FLV, H.264 + AAC | HLS / DASH (the input records) |
//! | `cloudflare-srt` | the same, SRT ingest | ffmpeg MPEG-TS, H.264 + AAC | HLS / DASH |
//! | `whip` | `url` (+ `token` / `user`) | WHIP | none known (the client gives its WHEP URL) |
//! | `rtmps` | `url` (`rtmp://` or `rtmps://`) + `stream_key` | ffmpeg FLV | none known (the client gives its HLS URL) |
//! | `srt` | `url` (`srt://host:port`) + `stream_id` / `passphrase` | ffmpeg MPEG-TS | none known |
//!
//! Configuration (`[outputs]`, env): `allow_private`
//! (`FV_OUTPUTS_ALLOW_PRIVATE`), `srt_latency_ms`, and `[outputs.cloudflare]`
//! `account_id` (`FV_CF_ACCOUNT_ID`), `api_token` (secret,
//! `FV_CF_STREAM_API_TOKEN`), `recording`, `on_stop`. Without a Cloudflare
//! account a request must name its own destination.
//!
//! Secrets (`token`, `stream_key`, `passphrase`, URL credentials, the
//! Cloudflare API token and ingest keys) are never echoed, logged, stored
//! unredacted or put in metrics: see [`spec`].

pub mod spec;

#[cfg(feature = "http-client")]
pub mod cloudflare;

#[cfg(all(feature = "webrtc", feature = "http-client"))]
pub mod publish;

#[cfg(feature = "http-client")]
pub mod rtmps;

use std::sync::{Arc, Mutex};
use std::time::Duration;

use fastvideo_protocol::ApiError;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

pub use spec::{OutputKind, OutputSpec, UrlPolicy};

/// `[outputs.cloudflare] recording`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum Recording {
    /// Off for WebRTC inputs (WHEP playback needs none), automatic for
    /// RTMPS/SRT (Cloudflare serves HLS/DASH of a live input only when it
    /// records).
    #[default]
    Auto,
    Off,
    Automatic,
}

/// `[outputs.cloudflare] on_stop`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum OnStop {
    /// Delete the input, unless it recorded (then disable it, so the
    /// recording stays reachable by the input id).
    #[default]
    Auto,
    Delete,
    Disable,
    Keep,
}

/// The server's Cloudflare Stream account. `Debug` redacts the token.
#[derive(Clone, PartialEq, Eq)]
pub struct CloudflareAccount {
    pub account_id: String,
    pub api_token: String,
    pub api_base: String,
    pub recording: Recording,
    pub on_stop: OnStop,
    pub delete_recording_after_days: Option<u32>,
    pub customer_domain: Option<String>,
    /// Live inputs are named `<name_prefix><label>`.
    pub name_prefix: String,
}

impl std::fmt::Debug for CloudflareAccount {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CloudflareAccount")
            .field("account_id", &self.account_id)
            .field("api_token", &"<redacted>")
            .field("api_base", &self.api_base)
            .field("recording", &self.recording)
            .field("on_stop", &self.on_stop)
            .finish()
    }
}

/// Resolved output settings (`[outputs]` + the WHIP bits of `[webrtc]`).
#[derive(Clone)]
pub struct OutputsConfig {
    pub policy: UrlPolicy,
    pub srt_latency_ms: u32,
    /// Extra CA certificates (PEM) RTMPS trusts besides the system roots.
    pub rtmps_ca_file: Option<std::path::PathBuf>,
    /// `None`: no Cloudflare default (account id or token missing).
    pub cloudflare: Option<CloudflareAccount>,
    /// The server's default WHIP bearer token (`FV_WHIP_TOKEN`), for kind
    /// `whip` without a `token`.
    pub whip_token: Option<String>,
    /// Default WHIP endpoint kind (`[webrtc] whip_target`).
    pub whip_target: Option<String>,
    /// `[webrtc] encoder` (`auto` | `nvenc` | `openh264` | `x264-test`).
    pub encoder: String,
    /// STUN for the WHIP publisher's srflx candidate (`FV_STREAM_STUN`).
    pub stun: Vec<String>,
    pub udp_bind: std::net::SocketAddr,
    pub connect_timeout: Duration,
}

impl std::fmt::Debug for OutputsConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OutputsConfig")
            .field("policy", &self.policy)
            .field("srt_latency_ms", &self.srt_latency_ms)
            .field("cloudflare", &self.cloudflare)
            .field("whip_token", &self.whip_token.as_ref().map(|_| "<redacted>"))
            .field("encoder", &self.encoder)
            .finish_non_exhaustive()
    }
}

impl Default for OutputsConfig {
    fn default() -> Self {
        Self {
            policy: UrlPolicy::default(),
            srt_latency_ms: 1000,
            rtmps_ca_file: None,
            cloudflare: None,
            whip_token: None,
            whip_target: None,
            encoder: "auto".into(),
            stun: vec!["stun:stun.l.google.com:19302".into()],
            udp_bind: "0.0.0.0:0".parse().expect("static addr"),
            connect_timeout: Duration::from_secs(30),
        }
    }
}

impl OutputsConfig {
    /// From the serve config plus `FV_STREAM_STUN` (comma-separated;
    /// `none` disables the probe).
    pub fn from_config(c: &crate::config::Config) -> Self {
        let o = &c.outputs;
        let cf = &o.cloudflare;
        let cloudflare = (!cf.account_id.trim().is_empty() && !cf.api_token.is_empty()).then(|| CloudflareAccount {
            account_id: cf.account_id.trim().to_owned(),
            api_token: cf.api_token.expose().trim().to_owned(),
            api_base: cf.api_base.clone(),
            recording: cf.recording,
            on_stop: cf.on_stop,
            delete_recording_after_days: cf.delete_recording_after_days,
            customer_domain: cf.customer_domain.clone().filter(|d| !d.trim().is_empty()),
            name_prefix: cf.name_prefix.clone(),
        });
        let mut s = Self {
            policy: UrlPolicy { allow_private: o.allow_private },
            srt_latency_ms: o.srt_latency_ms.clamp(20, 8000),
            rtmps_ca_file: o.rtmps_ca_file.clone(),
            cloudflare,
            whip_token: (!c.webrtc.whip_token.is_empty()).then(|| c.webrtc.whip_token.expose().to_owned()),
            whip_target: c.webrtc.whip_target.clone(),
            encoder: c.webrtc.encoder.clone(),
            ..Self::default()
        };
        if let Ok(v) = std::env::var("FV_STREAM_STUN") {
            s.stun = if v.trim() == "none" {
                Vec::new()
            } else {
                v.split(',').map(|x| x.trim().to_owned()).filter(|x| !x.is_empty()).collect()
            };
        }
        s
    }
}

/// Publisher counters (native streams flatten them into `status`).
#[derive(Clone, Debug, Default, Serialize)]
pub struct OutputStatus {
    /// `provisioning` | `waiting` (first frame) | `connecting` | `publishing` | `ended` | `failed`.
    pub output_state: String,
    /// The last destination error (scrubbed of credentials).
    pub output_error: Option<String>,
    /// The WHIP session resource, redacted.
    pub whip_resource: Option<String>,
    /// The negotiated codec (`H264`, `VP8`, `VP9`; sinks: `H264`).
    pub video_codec: Option<String>,
    pub encoder: Option<String>,
    /// The encoded (= native) canvas.
    pub width: Option<u32>,
    pub height: Option<u32>,
    pub frames_sent: u64,
    pub keyframes_sent: u64,
    pub audio_packets_sent: u64,
    /// IDRs forced on request (a forced NVENC IDR restarts ffmpeg).
    pub forced_idrs: u64,
    /// PLI/FIR received from the WHIP endpoint.
    pub keyframe_requests: u64,
    /// Requests answered by the periodic IDR or a keyframe that had just
    /// gone out, without forcing one.
    pub keyframe_requests_covered: u64,
    pub video_send_errors: u64,
    /// Ticks the publisher fell behind on (drop-oldest).
    pub ticks_dropped: u64,
    /// ffmpeg sink restarts (RTMP(S)/SRT).
    pub sink_restarts: u64,
}

/// One output's live state: counters, the redacted request, playback URLs.
#[derive(Debug)]
pub struct OutputState {
    pub status: Mutex<OutputStatus>,
    pub view: Value,
    pub playback: Mutex<Value>,
}

impl OutputState {
    pub fn new(view: Value) -> Arc<Self> {
        Arc::new(Self {
            status: Mutex::new(OutputStatus { output_state: "provisioning".into(), ..OutputStatus::default() }),
            view,
            playback: Mutex::new(json!({})),
        })
    }

    pub fn set(&self, f: impl FnOnce(&mut OutputStatus)) {
        if let Ok(mut s) = self.status.lock() {
            f(&mut s);
        }
    }

    pub fn status(&self) -> OutputStatus {
        self.status.lock().map(|s| s.clone()).unwrap_or_default()
    }

    pub fn playback(&self) -> Value {
        self.playback.lock().map(|p| p.clone()).unwrap_or(Value::Null)
    }
}

/// Output copies of director and Reactor sessions (`x_fv_output`): the
/// opener over this config, when the build can publish.
pub fn hook(c: &crate::config::Config) -> Option<fastvideo_media::output::OutputHook> {
    #[cfg(all(feature = "webrtc", feature = "http-client"))]
    {
        Some(fastvideo_media::output::OutputHook(Arc::new(publish::Opener(Outputs::from_config(c)))))
    }
    #[cfg(not(all(feature = "webrtc", feature = "http-client")))]
    {
        let _ = c;
        None
    }
}

/// The legacy native-stream WHIP fields.
#[derive(Clone, Debug, Default)]
pub struct LegacyWhip {
    pub url: String,
    pub token: Option<String>,
    pub user: Option<String>,
    pub target: Option<String>,
}

/// The output service: request resolution, Cloudflare provisioning and the
/// publisher ([`publish`]).
pub struct Outputs {
    pub(crate) cfg: OutputsConfig,
    #[cfg(feature = "http-client")]
    cf: Option<cloudflare::CfStream>,
}

impl std::fmt::Debug for Outputs {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Outputs").field("cfg", &self.cfg).finish()
    }
}

impl Outputs {
    pub fn new(cfg: OutputsConfig) -> Arc<Self> {
        #[cfg(feature = "http-client")]
        let cf = cfg.cloudflare.as_ref().and_then(|a| {
            cloudflare::CfStream::new(cloudflare::CfSettings {
                account_id: a.account_id.clone(),
                api_token: a.api_token.clone(),
                api_base: a.api_base.clone(),
                recording: a.recording,
                on_stop: a.on_stop,
                delete_recording_after_days: a.delete_recording_after_days,
                customer_domain: a.customer_domain.clone(),
                timeout: Duration::from_secs(20),
            })
            .map_err(|e| tracing::warn!(error = %e.message, "cloudflare stream client"))
            .ok()
        });
        Arc::new(Self {
            cfg,
            #[cfg(feature = "http-client")]
            cf,
        })
    }

    pub fn from_config(c: &crate::config::Config) -> Arc<Self> {
        Self::new(OutputsConfig::from_config(c))
    }

    pub fn config(&self) -> &OutputsConfig {
        &self.cfg
    }

    pub fn policy(&self) -> UrlPolicy {
        self.cfg.policy
    }

    #[cfg(feature = "http-client")]
    pub fn cloudflare(&self) -> Option<&cloudflare::CfStream> {
        self.cf.as_ref()
    }

    /// Whether the Cloudflare default can serve (account configured and a
    /// build that can call it).
    pub fn cloudflare_ready(&self) -> bool {
        #[cfg(feature = "http-client")]
        {
            self.cf.is_some()
        }
        #[cfg(not(feature = "http-client"))]
        {
            false
        }
    }

    /// The output of a request: its `output` object, else the legacy WHIP
    /// fields, else the default (Cloudflare when configured). Both at once
    /// is an error.
    pub fn resolve(&self, output: Option<&Value>, legacy: Option<LegacyWhip>) -> Result<OutputSpec, ApiError> {
        let output = output.filter(|v| !v.is_null());
        let spec = match (output, legacy) {
            (Some(_), Some(_)) => {
                return Err(ApiError::invalid_param("output", "give `output` or the legacy `whip_url` fields, not both"))
            }
            (Some(v), None) => OutputSpec::parse(v, self.cfg.policy)?,
            (None, Some(l)) => OutputSpec::from_legacy(&l.url, l.token, l.user, l.target, self.cfg.policy)?,
            (None, None) => {
                if !self.cloudflare_ready() {
                    return Err(ApiError::invalid_param(
                        "output",
                        "no output: give `output` ({\"kind\": \"whip\" | \"rtmps\" | \"srt\", \"url\": …}); this server has no Cloudflare Stream default ([outputs.cloudflare])",
                    ));
                }
                OutputSpec::parse(&json!({}), self.cfg.policy)?
            }
        };
        if spec.kind.is_cloudflare() && !self.cloudflare_ready() {
            return Err(ApiError::invalid_param(
                "output.kind",
                "this server has no Cloudflare Stream account ([outputs.cloudflare] account_id and FV_CF_STREAM_API_TOKEN)",
            ));
        }
        Ok(spec)
    }

    /// `GET /fv/v1/outputs`: what this server can publish to.
    pub fn info(&self) -> Value {
        let cf = self.cloudflare_ready();
        let kinds: Vec<&str> = OutputKind::ALL.iter().filter(|k| cf || !k.is_cloudflare()).map(|k| k.as_str()).collect();
        json!({
            "object": "fv.outputs",
            "default": if cf { Value::from("cloudflare") } else { Value::Null },
            "kinds": kinds,
            "codecs": {"whip": ["h264", "vp8", "vp9"], "rtmps": ["h264"], "srt": ["h264"]},
            "cloudflare": cf.then(|| json!({
                "recording": self.cfg.cloudflare.as_ref().map(|a| a.recording),
                "on_stop": self.cfg.cloudflare.as_ref().map(|a| a.on_stop),
            })),
            "allow_private": self.cfg.policy.allow_private,
            "publisher": cfg!(all(feature = "webrtc", feature = "http-client")),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn outs(cf: bool) -> Arc<Outputs> {
        let mut c = OutputsConfig::default();
        if cf {
            c.cloudflare = Some(CloudflareAccount {
                account_id: "acct".into(),
                api_token: "tok-secret".into(),
                api_base: "http://127.0.0.1:9".into(),
                recording: Recording::Auto,
                on_stop: OnStop::Auto,
                delete_recording_after_days: None,
                customer_domain: None,
                name_prefix: "fv-".into(),
            });
        }
        Outputs::new(c)
    }

    #[test]
    fn default_is_cloudflare_when_configured() {
        let o = outs(cfg!(feature = "http-client"));
        if o.cloudflare_ready() {
            assert_eq!(o.resolve(None, None).unwrap().kind, OutputKind::Cloudflare);
            assert_eq!(o.info()["default"], "cloudflare");
            assert!(!format!("{:?}", o).contains("tok-secret"));
        }
        let o = outs(false);
        let e = o.resolve(None, None).unwrap_err();
        assert_eq!(e.param.as_deref(), Some("output"));
        let e = o.resolve(Some(&json!({"kind": "cloudflare-srt"})), None).unwrap_err();
        assert_eq!(e.param.as_deref(), Some("output.kind"));
        assert_eq!(o.info()["default"], Value::Null);
        assert_eq!(o.info()["kinds"], json!(["whip", "rtmps", "srt"]));
        // Legacy fields and `output` are exclusive.
        let l = LegacyWhip { url: "https://relay.example/w".into(), ..LegacyWhip::default() };
        assert_eq!(o.resolve(None, Some(l.clone())).unwrap().kind, OutputKind::Whip);
        assert!(o.resolve(Some(&json!({"kind": "whip", "url": "https://a.example/w"})), Some(l)).is_err());
        assert_eq!(o.resolve(Some(&Value::Null), None).unwrap_err().param.as_deref(), Some("output"));
    }
}
