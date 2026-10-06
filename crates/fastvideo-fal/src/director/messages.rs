//! Control-channel messages of the WMA director (fal §8.4-8.5, ASYNC-DIR).
//!
//! **Client → model** messages are parsed strictly: every published schema
//! says "Additional properties: not allowed", so an unknown field, a wrong
//! type, an out-of-range value or an unknown `type` is refused with
//! `error{code:"invalid_message"}` (a diagnostic: the message is dropped and
//! the session continues). Values that are schema-valid but that we cannot
//! serve (`resolution:"1080p"` without the H3 1080P tier, `audio_url`) are refused later with their
//! own codes, by the control state machine.
//!
//! **Model → client** messages are built by the constructors at the bottom,
//! each emitting exactly the fields of its published schema.

use serde::Deserialize;
use serde_json::{json, Map, Value};

/// JS `Number.MAX_SAFE_INTEGER`: versions must stay within it (fal §8.3).
pub const MAX_SAFE_INTEGER: u64 = (1 << 53) - 1;
/// `prompt` / beat prompt length bounds.
pub const PROMPT_MAX_CHARS: usize = 50_000;
/// `script` beat count bounds.
pub const SCRIPT_MAX_BEATS: usize = 64;
/// `configure.chunk_duration` values (whole seconds): an extension of
/// fal's schema, which echoes the session's chunk length as
/// `configured.chunk_duration` but takes none. Clip models only; a causal
/// model ignores it.
pub const CHUNK_DURATIONS: [u32; 2] = [5, 10];
/// `configure.chunk_duration` when absent (the director's default).
pub const DEFAULT_CHUNK_DURATION: u32 = 5;
/// Reserved WMA network-info vocabulary (JS `wma.ts`).
pub const NETWORK_INFO_REQUEST: &str = "wma.network-info.request";
pub const NETWORK_INFO_RESPONSE: &str = "wma.network-info.response";

/// Client message types we accept (`client_message_types`).
pub const CLIENT_MESSAGE_TYPES: [&str; 4] = ["configure", "ping", "prompt", "stop"];
/// Server message types (`server_message_types`, the 16 of fal §8.5).
pub const SERVER_MESSAGE_TYPES: [&str; 16] = [
    "session_info",
    "configured",
    "prompt_pending",
    "prompt_applied",
    "prompt_rejected",
    "audio_pending",
    "audio_applied",
    "audio_rejected",
    "audio_exhausted",
    "chunk",
    "chunk_metrics",
    "deadline_missed",
    "error",
    "pong",
    "session_metrics",
    "stream_exhausted",
];

/// `configure.resolution` (lowercase `p`, unlike the HTTP endpoints).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize)]
pub enum Resolution {
    #[serde(rename = "480p")]
    R480,
    /// Not in fal's published schema (an extension for models with a 720
    /// tier, e.g. LTX); clients that do not know it never send it.
    #[serde(rename = "720p")]
    R720,
    #[serde(rename = "768p")]
    R768,
    #[serde(rename = "1080p")]
    R1080,
}

impl Resolution {
    /// Every value, lowest first (the order forms list them in).
    pub const ALL: [Resolution; 4] = [Resolution::R480, Resolution::R720, Resolution::R768, Resolution::R1080];

    pub fn as_str(self) -> &'static str {
        match self {
            Resolution::R480 => "480p",
            Resolution::R720 => "720p",
            Resolution::R768 => "768p",
            Resolution::R1080 => "1080p",
        }
    }
    pub fn short_edge(self) -> u32 {
        match self {
            Resolution::R480 => 480,
            Resolution::R720 => 720,
            Resolution::R768 => 768,
            Resolution::R1080 => 1080,
        }
    }
}

/// `configure.aspect_ratio`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize)]
pub enum Aspect {
    #[serde(rename = "16:9")]
    Landscape,
    #[serde(rename = "9:16")]
    Portrait,
    #[serde(rename = "1:1")]
    Square,
}

impl Aspect {
    pub fn as_str(self) -> &'static str {
        match self {
            Aspect::Landscape => "16:9",
            Aspect::Portrait => "9:16",
            Aspect::Square => "1:1",
        }
    }
    /// The served aspect closest to `ratio` (`width / height`) on a log
    /// scale: 16:9 for a landscape image, 9:16 for a portrait one, 1:1 for
    /// a near-square one (within 4:3 and 3:4).
    pub fn nearest(ratio: f64) -> Self {
        let l = ratio.max(1e-6).ln();
        [Aspect::Landscape, Aspect::Portrait, Aspect::Square]
            .into_iter()
            .min_by(|a, b| (a.ratio().ln() - l).abs().total_cmp(&(b.ratio().ln() - l).abs()))
            .unwrap_or(Aspect::Landscape)
    }
    /// The protocol's aspect ratio (`CanvasSpec::Aspect`).
    pub fn as_ratio(self) -> fastvideo_protocol::Ratio {
        match self {
            Aspect::Landscape => fastvideo_protocol::Ratio::R16_9,
            Aspect::Portrait => fastvideo_protocol::Ratio::R9_16,
            Aspect::Square => fastvideo_protocol::Ratio::R1_1,
        }
    }
    /// `width / height`.
    pub fn ratio(self) -> f64 {
        match self {
            Aspect::Landscape => 16.0 / 9.0,
            Aspect::Portrait => 9.0 / 16.0,
            Aspect::Square => 1.0,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum AudioBehavior {
    Replace,
    Queue,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ScriptMode {
    Replace,
    Append,
}

impl ScriptMode {
    pub fn as_str(self) -> &'static str {
        match self {
            ScriptMode::Replace => "replace",
            ScriptMode::Append => "append",
        }
    }
}

/// `ScriptBeat`: a direction on the associated video's clock.
#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ScriptBeat {
    /// Whole seconds from the start of the first video of this script.
    pub offset: u64,
    #[serde(default)]
    pub audio_url: Option<String>,
    #[serde(default)]
    pub end_image_url: Option<String>,
    #[serde(default)]
    pub prompt: Option<String>,
}

/// `configure` (correlation `/prompt_version`).
#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Configure {
    #[serde(rename = "type")]
    _type: String,
    pub prompt_version: u64,
    pub prompt: String,
    #[serde(default)]
    pub resolution: Option<Resolution>,
    #[serde(default)]
    pub aspect_ratio: Option<Aspect>,
    #[serde(default)]
    pub image_url: Option<String>,
    #[serde(default)]
    pub end_image_url: Option<String>,
    #[serde(default)]
    pub audio_url: Option<String>,
    #[serde(default)]
    pub memory: Option<u32>,
    #[serde(default)]
    pub audio_bitrate: Option<u32>,
    #[serde(default)]
    pub seed: Option<i64>,
    #[serde(default)]
    pub script: Option<Vec<ScriptBeat>>,
    #[serde(default)]
    pub protocol_version: Option<u64>,
    /// Seconds per chunk, one of [`CHUNK_DURATIONS`] (ours, not in fal's
    /// schema; named after `configured.chunk_duration`). `None`: the
    /// server's default (`default_chunk_duration`).
    #[serde(default)]
    pub chunk_duration: Option<u32>,
}

/// `prompt` (correlation `/prompt_version`).
#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Prompt {
    #[serde(rename = "type")]
    _type: String,
    pub prompt_version: u64,
    #[serde(default)]
    pub prompt: Option<String>,
    #[serde(default)]
    pub end_image_url: Option<String>,
    #[serde(default)]
    pub audio_url: Option<String>,
    #[serde(default)]
    pub audio_behavior: Option<AudioBehavior>,
    #[serde(default)]
    pub replan: Option<bool>,
    #[serde(default)]
    pub script: Option<Vec<ScriptBeat>>,
    #[serde(default)]
    pub script_mode: Option<ScriptMode>,
}

#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
struct Ping {
    #[serde(rename = "type")]
    _type: String,
    ts: serde_json::Number,
}

#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
struct Stop {
    #[serde(rename = "type")]
    _type: String,
}

#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
struct NetworkInfoRequest {
    #[serde(rename = "type")]
    _type: String,
    request_id: String,
}

/// One parsed client message.
#[derive(Clone, Debug, PartialEq)]
pub enum ClientMessage {
    Configure(Box<Configure>),
    Prompt(Box<Prompt>),
    /// `ts` echoed verbatim as `pong.client_ts`.
    Ping(serde_json::Number),
    Stop,
    /// The JS client's reserved session-affine network query.
    NetworkInfo { request_id: String },
}

/// Why a message was refused as `invalid_message`.
#[derive(Clone, Debug, PartialEq)]
pub struct Invalid {
    pub error: String,
    /// The message's `prompt_version` when it had a readable one.
    pub prompt_version: Option<u64>,
}

fn invalid(error: impl Into<String>, v: &Value) -> Invalid {
    Invalid { error: error.into(), prompt_version: v.get("prompt_version").and_then(Value::as_u64) }
}

fn check_text(field: &str, s: &str, v: &Value) -> Result<(), Invalid> {
    let n = s.chars().count();
    if n == 0 || n > PROMPT_MAX_CHARS {
        return Err(invalid(format!("`{field}` must be 1..{PROMPT_MAX_CHARS} characters"), v));
    }
    Ok(())
}

fn check_url(field: &str, s: &Option<String>, v: &Value) -> Result<(), Invalid> {
    if matches!(s, Some(u) if u.is_empty()) {
        return Err(invalid(format!("`{field}` must not be empty"), v));
    }
    Ok(())
}

fn check_version(ver: u64, v: &Value) -> Result<(), Invalid> {
    if !(1..=MAX_SAFE_INTEGER).contains(&ver) {
        return Err(invalid("`prompt_version` must be an integer in 1..2^53-1", v));
    }
    Ok(())
}

fn check_script(script: &Option<Vec<ScriptBeat>>, v: &Value) -> Result<(), Invalid> {
    let Some(beats) = script else { return Ok(()) };
    if beats.is_empty() || beats.len() > SCRIPT_MAX_BEATS {
        return Err(invalid(format!("`script` must have 1..{SCRIPT_MAX_BEATS} beats"), v));
    }
    for (i, b) in beats.iter().enumerate() {
        if let Some(p) = &b.prompt {
            check_text(&format!("script[{i}].prompt"), p, v)?;
        }
        check_url(&format!("script[{i}].end_image_url"), &b.end_image_url, v)?;
        check_url(&format!("script[{i}].audio_url"), &b.audio_url, v)?;
    }
    Ok(())
}

/// Parses one control-channel text frame strictly.
pub fn parse(text: &str) -> Result<ClientMessage, Invalid> {
    let v: Value = serde_json::from_str(text)
        .map_err(|e| Invalid { error: format!("control messages are JSON objects: {e}"), prompt_version: None })?;
    let Some(obj) = v.as_object() else {
        return Err(invalid("control messages are JSON objects", &v));
    };
    let Some(ty) = obj.get("type").and_then(Value::as_str) else {
        return Err(invalid("missing string field `type`", &v));
    };
    let de = |e: serde_json::Error| invalid(format!("invalid `{ty}` message: {e}"), &v);
    match ty {
        "configure" => {
            // Checked on the raw value so a wrong type names the field too.
            match obj.get("chunk_duration") {
                None | Some(Value::Null) => {}
                Some(d) if d.as_u64().is_some_and(|d| CHUNK_DURATIONS.iter().any(|&c| u64::from(c) == d)) => {}
                Some(d) => return Err(invalid(format!("`chunk_duration` must be 5 or 10 (whole seconds), got {d}"), &v)),
            }
            let m: Configure = serde_json::from_value(v.clone()).map_err(de)?;
            check_version(m.prompt_version, &v)?;
            check_text("prompt", &m.prompt, &v)?;
            check_url("image_url", &m.image_url, &v)?;
            if let Some(mem) = m.memory {
                if !(1..=50).contains(&mem) {
                    return Err(invalid("`memory` must be in 1..50", &v));
                }
            }
            if let Some(b) = m.audio_bitrate {
                if !matches!(b, 96_000 | 128_000 | 192_000) {
                    return Err(invalid("`audio_bitrate` must be 96000, 128000 or 192000", &v));
                }
            }
            if m.protocol_version.is_some_and(|p| p != 1) {
                return Err(invalid("`protocol_version` must be 1", &v));
            }
            check_script(&m.script, &v)?;
            Ok(ClientMessage::Configure(Box::new(m)))
        }
        "prompt" => {
            let m: Prompt = serde_json::from_value(v.clone()).map_err(de)?;
            check_version(m.prompt_version, &v)?;
            if let Some(p) = &m.prompt {
                check_text("prompt", p, &v)?;
            }
            check_script(&m.script, &v)?;
            Ok(ClientMessage::Prompt(Box::new(m)))
        }
        "ping" => {
            let m: Ping = serde_json::from_value(v.clone()).map_err(de)?;
            Ok(ClientMessage::Ping(m.ts))
        }
        "stop" => {
            let _: Stop = serde_json::from_value(v.clone()).map_err(de)?;
            Ok(ClientMessage::Stop)
        }
        NETWORK_INFO_REQUEST => {
            let m: NetworkInfoRequest = serde_json::from_value(v.clone()).map_err(de)?;
            Ok(ClientMessage::NetworkInfo { request_id: m.request_id })
        }
        other => Err(invalid(format!("unknown message type `{other}`"), &v)),
    }
}

// ---------------------------------------------------------------------------
// Model → client
// ---------------------------------------------------------------------------

/// `error` codes (fal §8.3 taxonomy).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ErrorCode {
    // sessionFailure
    ConfigurationTimeout,
    InitializationTimeout,
    InvalidInitialImage,
    InvalidInitialAudio,
    InvalidInitialScript,
    InvalidInput,
    GenerationTimeout,
    GenerationFailed,
    // diagnostic
    InvalidMessage,
    NotConfigured,
    ImmutableSettings,
}

impl ErrorCode {
    pub fn as_str(self) -> &'static str {
        match self {
            ErrorCode::ConfigurationTimeout => "configuration_timeout",
            ErrorCode::InitializationTimeout => "initialization_timeout",
            ErrorCode::InvalidInitialImage => "invalid_initial_image",
            ErrorCode::InvalidInitialAudio => "invalid_initial_audio",
            ErrorCode::InvalidInitialScript => "invalid_initial_script",
            ErrorCode::InvalidInput => "invalid_input",
            ErrorCode::GenerationTimeout => "generation_timeout",
            ErrorCode::GenerationFailed => "generation_failed",
            ErrorCode::InvalidMessage => "invalid_message",
            ErrorCode::NotConfigured => "not_configured",
            ErrorCode::ImmutableSettings => "immutable_settings",
        }
    }

    /// Session failures end the session; diagnostics do not.
    pub fn is_session_failure(self) -> bool {
        !matches!(self, ErrorCode::InvalidMessage | ErrorCode::NotConfigured | ErrorCode::ImmutableSettings)
    }
}

/// `prompt_rejected.reason`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RejectReason {
    PreparationFailed,
    StalePromptVersion,
    InvalidScript,
    InfeasibleTiming,
    InvalidAudio,
    InvalidImage,
    QueueFull,
}

impl RejectReason {
    pub fn as_str(self) -> &'static str {
        match self {
            RejectReason::PreparationFailed => "preparation_failed",
            RejectReason::StalePromptVersion => "stale_prompt_version",
            RejectReason::InvalidScript => "invalid_script",
            RejectReason::InfeasibleTiming => "infeasible_timing",
            RejectReason::InvalidAudio => "invalid_audio",
            RejectReason::InvalidImage => "invalid_image",
            RejectReason::QueueFull => "queue_full",
        }
    }
}

pub fn error(code: ErrorCode, message: impl Into<String>, prompt_version: Option<u64>) -> Value {
    json!({"type": "error", "code": code.as_str(), "error": message.into(), "prompt_version": prompt_version, "detail": null})
}

pub fn pong(client_ts: serde_json::Number) -> Value {
    json!({"type": "pong", "client_ts": client_ts})
}

pub fn prompt_pending(v: u64) -> Value {
    json!({"type": "prompt_pending", "prompt_version": v})
}

/// `prompt_applied`; `script` = (origin chunk, queued scripts, beats, mode).
pub fn prompt_applied(v: u64, script: Option<(u32, u32, u32, ScriptMode)>) -> Value {
    match script {
        None => json!({"type": "prompt_applied", "prompt_version": v}),
        Some((origin, queued, beats, mode)) => json!({
            "type": "prompt_applied",
            "prompt_version": v,
            "script_origin_chunk_index": origin,
            "script_queued": queued,
            "script_beats": beats,
            "script_mode": mode.as_str(),
        }),
    }
}

pub fn prompt_rejected(v: u64, reason: RejectReason, message: impl Into<String>) -> Value {
    json!({"type": "prompt_rejected", "prompt_version": v, "reason": reason.as_str(), "error": message.into()})
}

pub fn stream_exhausted(chunks: u32, reason: &str) -> Value {
    json!({"type": "stream_exhausted", "chunks": chunks, "reason": reason})
}

pub fn deadline_missed(chunk_index: u32, late_by_seconds: f64) -> Value {
    json!({
        "type": "deadline_missed",
        "chunk_index": chunk_index,
        "late_by_seconds": late_by_seconds,
        "behavior": "freeze_video_and_silence_audio_until_ready",
    })
}

/// `wma.network-info.response` with the runner's view of the path, in the
/// snake_case shape the JS client's `normalizeRunnerPath` reads.
pub fn network_info(request_id: &str, path: Value) -> Value {
    json!({"type": NETWORK_INFO_RESPONSE, "request_id": request_id, "path": path})
}

/// Keeps only `Some` values of a field list (for optional schema fields).
pub fn object(fields: Vec<(&str, Value)>) -> Value {
    let mut m = Map::new();
    for (k, v) in fields {
        m.insert(k.to_owned(), v);
    }
    Value::Object(m)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nearest_aspect_follows_the_image() {
        for (w, h, want) in [
            (1920.0, 1080.0, Aspect::Landscape),
            (1080.0, 1920.0, Aspect::Portrait),
            (1024.0, 1024.0, Aspect::Square),
            (1200.0, 1000.0, Aspect::Square),
            (1500.0, 1000.0, Aspect::Landscape),
            (1000.0, 1250.0, Aspect::Square),
            (1000.0, 1400.0, Aspect::Portrait),
            (1000.0, 1500.0, Aspect::Portrait),
            (8000.0, 500.0, Aspect::Landscape),
            (500.0, 8000.0, Aspect::Portrait),
        ] {
            assert_eq!(Aspect::nearest(w / h), want, "{w}x{h}");
        }
    }

    fn err(s: &str) -> Invalid {
        parse(s).unwrap_err()
    }

    #[test]
    fn configure_parses_the_documented_example() {
        let m = parse(
            r#"{"aspect_ratio":"16:9","protocol_version":1,"memory":3,"prompt_version":1,"prompt":"A sitcom","type":"configure","resolution":"768p"}"#,
        )
        .unwrap();
        let ClientMessage::Configure(c) = m else { panic!() };
        assert_eq!(c.prompt_version, 1);
        assert_eq!(c.resolution, Some(Resolution::R768));
        assert_eq!(c.aspect_ratio, Some(Aspect::Landscape));
        assert_eq!(c.memory, Some(3));
        // Nulls are accepted for every nullable field.
        assert!(parse(r#"{"type":"configure","prompt_version":1,"prompt":"x","image_url":null,"end_image_url":null,"audio_url":null,"audio_bitrate":null,"seed":null,"script":null,"chunk_duration":null}"#).is_ok());
        assert_eq!(c.chunk_duration, None);
        for d in CHUNK_DURATIONS {
            let ClientMessage::Configure(c) = parse(&format!(r#"{{"type":"configure","prompt_version":1,"prompt":"x","chunk_duration":{d}}}"#)).unwrap() else {
                panic!()
            };
            assert_eq!(c.chunk_duration, Some(d));
        }
    }

    #[test]
    fn every_resolution_round_trips() {
        for r in Resolution::ALL {
            let m = parse(&format!(r#"{{"type":"configure","prompt_version":1,"prompt":"x","resolution":"{}"}}"#, r.as_str())).unwrap();
            let ClientMessage::Configure(c) = m else { panic!() };
            assert_eq!(c.resolution, Some(r));
            assert_eq!(r.as_str().trim_end_matches('p').parse::<u32>().unwrap(), r.short_edge());
        }
        assert!(parse(r#"{"type":"configure","prompt_version":1,"prompt":"x","resolution":"720P"}"#).is_err());
    }

    #[test]
    fn strictness() {
        // Extra properties.
        let e = err(r#"{"type":"configure","prompt_version":1,"prompt":"x","colour":"red"}"#);
        assert!(e.error.contains("colour"), "{e:?}");
        assert_eq!(e.prompt_version, Some(1));
        assert!(parse(r#"{"type":"ping","ts":1,"extra":true}"#).is_err());
        assert!(parse(r#"{"type":"stop","now":true}"#).is_err());
        assert!(parse(r#"{"type":"prompt","prompt_version":2,"prompt":"x","script":[{"offset":0,"prompt":"a","speed":1}]}"#).is_err());
        // Types, enums, ranges.
        assert!(parse(r#"{"type":"configure","prompt_version":"1","prompt":"x"}"#).is_err());
        // 720p is an extension (LTX's tier); tiers nobody serves stay refused.
        assert!(parse(r#"{"type":"configure","prompt_version":1,"prompt":"x","resolution":"360p"}"#).is_err());
        assert!(parse(r#"{"type":"configure","prompt_version":1,"prompt":"x","resolution":"768P"}"#).is_err());
        assert!(parse(r#"{"type":"configure","prompt_version":1,"prompt":"x","aspect_ratio":"4:3"}"#).is_err());
        assert!(parse(r#"{"type":"configure","prompt_version":1,"prompt":"x","memory":0}"#).is_err());
        assert!(parse(r#"{"type":"configure","prompt_version":1,"prompt":"x","memory":51}"#).is_err());
        assert!(parse(r#"{"type":"configure","prompt_version":1,"prompt":"x","audio_bitrate":64000}"#).is_err());
        assert!(parse(r#"{"type":"configure","prompt_version":1,"prompt":"x","protocol_version":2}"#).is_err());
        // chunk_duration: whole seconds, 5 or 10.
        for bad in ["0", "3", "7", "15", "-5", "5.5", "\"5\""] {
            let e = err(&format!(r#"{{"type":"configure","prompt_version":1,"prompt":"x","chunk_duration":{bad}}}"#));
            assert!(e.error.contains("chunk_duration"), "{bad}: {e:?}");
            assert_eq!(e.prompt_version, Some(1));
        }
        assert!(parse(r#"{"type":"configure","prompt_version":0,"prompt":"x"}"#).is_err());
        assert!(parse(r#"{"type":"configure","prompt_version":9007199254740992,"prompt":"x"}"#).is_err());
        assert!(parse(r#"{"type":"configure","prompt_version":1,"prompt":""}"#).is_err());
        assert!(parse(r#"{"type":"configure","prompt_version":1,"prompt":"x","image_url":""}"#).is_err());
        assert!(parse(r#"{"type":"configure","prompt_version":1.5,"prompt":"x"}"#).is_err());
        assert!(parse(r#"{"type":"prompt","prompt_version":2,"script":[]}"#).is_err());
        assert!(parse(r#"{"type":"prompt","prompt_version":2,"audio_behavior":"mix"}"#).is_err());
        assert!(parse(r#"{"type":"prompt","prompt_version":2,"script_mode":"merge"}"#).is_err());
        // Required fields.
        assert!(parse(r#"{"type":"configure","prompt":"x"}"#).is_err());
        assert!(parse(r#"{"type":"ping"}"#).is_err());
        // Not objects / unknown types.
        assert!(parse("[]").is_err());
        assert!(parse("not json").is_err());
        assert_eq!(err(r#"{"type":"dance"}"#).error, "unknown message type `dance`");
        assert!(parse(r#"{"kind":"ping"}"#).is_err());
    }

    #[test]
    fn prompt_ping_stop_network_info() {
        let ClientMessage::Prompt(p) =
            parse(r#"{"type":"prompt","prompt":"They follow a narrow path down to the harbor.","prompt_version":2}"#).unwrap()
        else {
            panic!()
        };
        assert_eq!(p.prompt_version, 2);
        assert_eq!(p.replan, None);
        let ClientMessage::Ping(ts) = parse(r#"{"type":"ping","ts":1712.5}"#).unwrap() else { panic!() };
        assert_eq!(pong(ts)["client_ts"], json!(1712.5));
        assert_eq!(parse(r#"{"type":"stop"}"#).unwrap(), ClientMessage::Stop);
        assert_eq!(
            parse(r#"{"type":"wma.network-info.request","request_id":"r1"}"#).unwrap(),
            ClientMessage::NetworkInfo { request_id: "r1".into() }
        );
    }

    #[test]
    fn server_messages() {
        assert_eq!(
            error(ErrorCode::InvalidInput, "no", Some(1)),
            json!({"type":"error","code":"invalid_input","error":"no","prompt_version":1,"detail":null})
        );
        assert_eq!(prompt_rejected(3, RejectReason::StalePromptVersion, "old")["reason"], "stale_prompt_version");
        assert_eq!(stream_exhausted(4, "stopped"), json!({"type":"stream_exhausted","chunks":4,"reason":"stopped"}));
        assert_eq!(deadline_missed(2, 0.5)["behavior"], "freeze_video_and_silence_audio_until_ready");
        assert_eq!(prompt_applied(5, Some((3, 0, 2, ScriptMode::Append)))["script_mode"], "append");
        assert!(!ErrorCode::InvalidMessage.is_session_failure());
        assert!(ErrorCode::InvalidInitialAudio.is_session_failure());
        assert_eq!(SERVER_MESSAGE_TYPES.len(), 16);
    }
}
