//! Cloudflare Stream Live Inputs (developers.cloudflare.com/api/resources/stream/subresources/live_inputs/):
//! the default output of a stream (docs/serve/stream-outputs.md).
//!
//! One live input per stream, named `fv-<label>`:
//!
//! | call | API |
//! |---|---|
//! | create | `POST /accounts/{account}/stream/live_inputs` `{"meta":{"name":…},"recording":{"mode":"off"\|"automatic"}}` |
//! | delete (on stop, default) | `DELETE …/live_inputs/{uid}` |
//! | disable (on stop, `on_stop = "disable"`) | `PUT …/live_inputs/{uid}` `{"enabled": false}` |
//!
//! The response carries every ingest (`webRTC.url`, `rtmps.url` +
//! `streamKey`, `srt.url` + `streamId` + `passphrase`) and the WHEP playback
//! URL (`webRTCPlayback.url`). HLS/DASH and the Stream player are templated
//! on the customer domain (the host of the WebRTC URLs):
//! `https://customer-<code>.cloudflarestream.com/<uid>/manifest/video.m3u8`,
//! `…/video.mpd`, `…/iframe`. HLS/DASH of a live input only exist with
//! `recording.mode = "automatic"` (Cloudflare: "By default, recording mode
//! is set to off, and the input will not be recorded or available for
//! playback"), so RTMPS/SRT inputs record by default and WHIP inputs (whose
//! playback is WHEP, never HLS) do not.
//!
//! The API token is sent only as `Authorization: Bearer`; `Debug` redacts
//! it, and errors carry Cloudflare's error codes and messages, never the
//! ingest credentials.

use std::time::Duration;

use fastvideo_protocol::{ApiError, ErrorKind};
use serde_json::{json, Value};

pub use super::{OnStop, Recording};

/// What the client needs.
#[derive(Clone, Debug)]
pub struct CfSettings {
    pub account_id: String,
    pub api_token: String,
    /// `https://api.cloudflare.com/client/v4` (tests: a fake server).
    pub api_base: String,
    pub recording: Recording,
    pub on_stop: OnStop,
    /// `deleteRecordingAfterDays` for recording inputs (`None`: keep).
    pub delete_recording_after_days: Option<u32>,
    /// `customer-<code>.cloudflarestream.com` when the API response does
    /// not name it (it always has in practice: the WebRTC URLs).
    pub customer_domain: Option<String>,
    pub timeout: Duration,
}

/// One live input's ingest and playback endpoints. `Debug` redacts.
#[derive(Clone, Default, PartialEq, Eq)]
pub struct LiveInput {
    pub uid: String,
    pub webrtc_url: Option<String>,
    pub webrtc_playback_url: Option<String>,
    pub rtmps_url: Option<String>,
    pub rtmps_key: Option<String>,
    pub srt_url: Option<String>,
    pub srt_stream_id: Option<String>,
    pub srt_passphrase: Option<String>,
    /// Whether the input records (HLS/DASH available).
    pub recording: bool,
}

impl std::fmt::Debug for LiveInput {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LiveInput").field("uid", &self.uid).field("recording", &self.recording).finish_non_exhaustive()
    }
}

impl LiveInput {
    /// Parses a live input object (the API's `result`).
    pub fn from_json(r: &Value) -> Result<Self, ApiError> {
        let s = |p: &str| r.pointer(p).and_then(Value::as_str).filter(|x| !x.is_empty()).map(str::to_owned);
        let uid = s("/uid").ok_or_else(|| ApiError::internal("Cloudflare returned a live input without a uid"))?;
        Ok(LiveInput {
            uid,
            webrtc_url: s("/webRTC/url"),
            webrtc_playback_url: s("/webRTCPlayback/url"),
            rtmps_url: s("/rtmps/url"),
            rtmps_key: s("/rtmps/streamKey"),
            srt_url: s("/srt/url"),
            srt_stream_id: s("/srt/streamId"),
            srt_passphrase: s("/srt/passphrase"),
            recording: r.pointer("/recording/mode").and_then(Value::as_str) == Some("automatic"),
        })
    }

    /// `customer-<code>.cloudflarestream.com`, from the WebRTC URLs.
    pub fn customer_domain(&self) -> Option<String> {
        [&self.webrtc_playback_url, &self.webrtc_url]
            .into_iter()
            .flatten()
            .filter_map(|u| url::Url::parse(u).ok())
            .filter_map(|u| u.host_str().map(str::to_owned))
            .find(|h| h.starts_with("customer-"))
    }

    /// The playback block of a stream (public URLs: the WHEP URL names the
    /// input's playback id, not its publish key).
    pub fn playback(&self, customer_domain: Option<&str>, whip: bool) -> Value {
        let domain = self.customer_domain().or_else(|| customer_domain.map(str::to_owned));
        let mut p = json!({"provider": "cloudflare", "live_input_id": self.uid, "format": if whip { "whep" } else { "hls" }});
        if whip {
            if let Some(u) = &self.webrtc_playback_url {
                p["whep_url"] = json!(u);
            }
        }
        if let Some(d) = domain {
            let base = format!("https://{d}/{}", self.uid);
            p["iframe_url"] = json!(format!("{base}/iframe"));
            if !whip || self.recording {
                p["hls_url"] = json!(format!("{base}/manifest/video.m3u8"));
                p["dash_url"] = json!(format!("{base}/manifest/video.mpd"));
            }
        }
        p
    }
}

/// The Stream API client.
#[derive(Clone)]
pub struct CfStream {
    http: reqwest::Client,
    s: CfSettings,
}

impl std::fmt::Debug for CfStream {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CfStream")
            .field("account_id", &self.s.account_id)
            .field("api_token", &"<redacted>")
            .field("api_base", &self.s.api_base)
            .finish()
    }
}

/// Cloudflare's `errors: [{code, message}]`, joined.
fn cf_errors(v: &Value) -> String {
    v.get("errors")
        .and_then(Value::as_array)
        .map(|a| {
            a.iter()
                .map(|e| {
                    format!(
                        "{} {}",
                        e.get("code").map(|c| c.to_string()).unwrap_or_default(),
                        e.get("message").and_then(Value::as_str).unwrap_or_default()
                    )
                })
                .collect::<Vec<_>>()
                .join("; ")
        })
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "no error detail".into())
}

impl CfStream {
    pub fn new(s: CfSettings) -> Result<Self, ApiError> {
        let http = reqwest::Client::builder()
            .timeout(s.timeout)
            .build()
            .map_err(|e| ApiError::internal(format!("cloudflare http client: {e}")))?;
        Ok(Self { http, s })
    }

    pub fn settings(&self) -> &CfSettings {
        &self.s
    }

    fn url(&self, tail: &str) -> String {
        format!("{}/accounts/{}/stream/live_inputs{tail}", self.s.api_base.trim_end_matches('/'), self.s.account_id)
    }

    /// Whether a new input for a WHIP (`true`) or RTMPS/SRT stream records.
    pub fn records(&self, whip: bool) -> bool {
        match self.s.recording {
            Recording::Auto => !whip,
            Recording::Off => false,
            Recording::Automatic => true,
        }
    }

    async fn call(&self, rb: reqwest::RequestBuilder, what: &str) -> Result<Value, ApiError> {
        let resp = rb.bearer_auth(&self.s.api_token).send().await.map_err(|e| {
            let kind = if e.is_timeout() { ErrorKind::Timeout } else { ErrorKind::Internal };
            // reqwest's error names the URL (account id, input uid): no secret.
            ApiError::new(kind, format!("Cloudflare Stream {what}: {}", e.without_url()))
        })?;
        let status = resp.status().as_u16();
        let body: Value = resp.json().await.unwrap_or(Value::Null);
        if (200..300).contains(&status) && body.get("success").and_then(Value::as_bool) != Some(false) {
            return Ok(body.get("result").cloned().unwrap_or(Value::Null));
        }
        let detail = cf_errors(&body);
        let hint = if status == 401 || status == 403 || detail.contains("10000") {
            ": the API token needs the Account > Stream > Edit permission and Stream must be enabled on the account (docs/serve/stream-outputs.md)"
        } else {
            ""
        };
        Err(ApiError::internal(format!("Cloudflare Stream {what}: HTTP {status}: {detail}{hint}")))
    }

    /// Creates the live input `name`.
    pub async fn create(&self, name: &str, record: bool) -> Result<LiveInput, ApiError> {
        let mut body = json!({
            "meta": {"name": name},
            "recording": {"mode": if record { "automatic" } else { "off" }},
        });
        if let (true, Some(d)) = (record, self.s.delete_recording_after_days) {
            body["deleteRecordingAfterDays"] = json!(d);
        }
        let r = self.call(self.http.post(self.url("")).json(&body), "create live input").await?;
        LiveInput::from_json(&r)
    }

    /// One live input (its current state, `status`).
    pub async fn get(&self, uid: &str) -> Result<Value, ApiError> {
        self.call(self.http.get(self.url(&format!("/{uid}"))), "get live input").await
    }

    pub async fn delete(&self, uid: &str) -> Result<(), ApiError> {
        self.call(self.http.delete(self.url(&format!("/{uid}"))), "delete live input").await.map(|_| ())
    }

    /// Stops the input from taking new broadcasts (its recordings stay).
    pub async fn disable(&self, uid: &str) -> Result<(), ApiError> {
        self.call(self.http.put(self.url(&format!("/{uid}"))).json(&json!({"enabled": false})), "disable live input")
            .await
            .map(|_| ())
    }

    /// What `on_stop` does with an input that did (not) record: `delete`,
    /// `disable` or `keep`.
    pub fn stop_action(&self, recorded: bool) -> OnStop {
        match self.s.on_stop {
            OnStop::Auto if recorded => OnStop::Disable,
            OnStop::Auto => OnStop::Delete,
            other => other,
        }
    }

    /// Releases an input per `on_stop`. Returns the action taken.
    pub async fn release(&self, input: &LiveInput) -> Result<OnStop, ApiError> {
        let a = self.stop_action(input.recording);
        match a {
            OnStop::Delete | OnStop::Auto => self.delete(&input.uid).await?,
            OnStop::Disable => self.disable(&input.uid).await?,
            OnStop::Keep => {}
        }
        Ok(a)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> Value {
        json!({
            "uid": "66be4bf738797e01e1fca35a7bdecdcd",
            "rtmps": {"url": "rtmps://live.cloudflare.com:443/live/", "streamKey": "MTQ0MTcjM3MjI1NDE3ODIyNTI1MjYyMjE4NTI2ODI1NDcxMzUyMzcf"},
            "srt": {"url": "srt://live.cloudflare.com:778", "streamId": "f256e6ea9341d51eea64c9454659e576", "passphrase": "766d7a3dfb1bf6cbb7c0d1b5d3a0a4e3"},
            "webRTC": {"url": "https://customer-m033z5x00ks6nunl.cloudflarestream.com/b236bde30eb07b9d01318940e5fc3edake34a3efb3896e18f2dc277ce6cc993ad/webRTC/publish"},
            "webRTCPlayback": {"url": "https://customer-m033z5x00ks6nunl.cloudflarestream.com/197a6b5a3e1d8bf74d9b53cbd2f2bcd4/webRTC/play"},
            "recording": {"mode": "off"},
            "meta": {"name": "fv-fvstream_x"},
        })
    }

    #[test]
    fn parses_ingests_and_playback() {
        let li = LiveInput::from_json(&sample()).unwrap();
        assert_eq!(li.uid, "66be4bf738797e01e1fca35a7bdecdcd");
        assert_eq!(li.srt_stream_id.as_deref(), Some("f256e6ea9341d51eea64c9454659e576"));
        assert!(!li.recording);
        assert_eq!(li.customer_domain().as_deref(), Some("customer-m033z5x00ks6nunl.cloudflarestream.com"));
        let p = li.playback(None, true);
        assert_eq!(p["whep_url"], "https://customer-m033z5x00ks6nunl.cloudflarestream.com/197a6b5a3e1d8bf74d9b53cbd2f2bcd4/webRTC/play");
        assert_eq!(p["iframe_url"], "https://customer-m033z5x00ks6nunl.cloudflarestream.com/66be4bf738797e01e1fca35a7bdecdcd/iframe");
        assert!(p.get("hls_url").is_none(), "no HLS without recording");
        let mut rec = li.clone();
        rec.recording = true;
        let p = rec.playback(None, false);
        assert_eq!(p["format"], "hls");
        assert_eq!(p["hls_url"], "https://customer-m033z5x00ks6nunl.cloudflarestream.com/66be4bf738797e01e1fca35a7bdecdcd/manifest/video.m3u8");
        assert_eq!(p["dash_url"], "https://customer-m033z5x00ks6nunl.cloudflarestream.com/66be4bf738797e01e1fca35a7bdecdcd/manifest/video.mpd");
        assert!(p.get("whep_url").is_none());
        // Debug never prints the ingest credentials.
        let d = format!("{li:?}");
        assert!(!d.contains("MTQ0") && !d.contains("766d7a") && !d.contains("b236bde"), "{d}");
        assert!(LiveInput::from_json(&json!({})).is_err());
    }

    #[test]
    fn errors_name_the_permission() {
        let e = cf_errors(&json!({"errors": [{"code": 10000, "message": "Authentication error"}]}));
        assert_eq!(e, "10000 Authentication error");
        assert_eq!(cf_errors(&json!({})), "no error detail");
    }

    #[test]
    fn recording_and_stop_policy() {
        let mk = |recording, on_stop| {
            CfStream::new(CfSettings {
                account_id: "a".into(),
                api_token: "secret-token".into(),
                api_base: "http://127.0.0.1:9".into(),
                recording,
                on_stop,
                delete_recording_after_days: None,
                customer_domain: None,
                timeout: Duration::from_secs(1),
            })
            .unwrap()
        };
        let c = mk(Recording::Auto, OnStop::Auto);
        assert!(!c.records(true) && c.records(false));
        assert_eq!(c.stop_action(false), OnStop::Delete);
        assert_eq!(c.stop_action(true), OnStop::Disable);
        assert!(!format!("{c:?}").contains("secret-token"));
        let c = mk(Recording::Off, OnStop::Delete);
        assert!(!c.records(false));
        assert_eq!(c.stop_action(true), OnStop::Delete);
        assert!(mk(Recording::Automatic, OnStop::Keep).records(true));
    }
}
