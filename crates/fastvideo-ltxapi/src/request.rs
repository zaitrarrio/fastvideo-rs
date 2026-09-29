//! `TextToVideoRequest` / `ImageToVideoRequest` -> `GenerationRequest`
//! (design §4.5, ltx §2.3-§2.4).
//!
//! The body is read as JSON and validated field by field so every refusal is
//! an LTX `invalid_request_error` with a readable message (e.g. `duration is
//! required`, ltx §1.5). v1 and v2 share the request schema (ltx §1.3).
//!
//! | LTX field | Normalized |
//! |---|---|
//! | `prompt` (required, ≤ 5000 chars) | `prompt` |
//! | `model` (required) | the target's engine name ([`crate::models`]) |
//! | `duration` (key required; integer per matrix) | `Seconds{Nearest}` → 8k+1 frames, at most the engine's 481 (20 s at 25 fps runs 481 frames, noted in the job log); `null` → `Length::Auto` (400 `Unsupported(LtxAutoDuration)` at negotiation) |
//! | `resolution` (required, `WxH`) | `Exact` (the engine pads to its multiple and crops back) |
//! | `fps` (24 / 25 / 48 / 50, default 24) | `fps` (the engine's caps decide; 24 / 25 / 48 / 50 since E4, else 400 `Unsupported(LtxFps)`) |
//! | `generate_audio` (default `true`) | `false` → `AudioOut::Silent` |
//! | `camera_motion` | 400 `Unsupported(LtxCameraMotion)` |
//! | `image_uri` (i2v, required) | `Keyframe{First}`: `I2V` |
//! | `last_frame_uri` (i2v) | `Keyframe{Last}`: `Keyframes` (E9; 400 `Unsupported(LtxKeyframes)` on an engine without it) |
//!
//! Media URIs must be `https://…`, `data:…;base64,…` or `ltx://uploads/<token>`
//! (ltx §2.0). `image_uri` / `last_frame_uri` on text-to-video are refused
//! rather than ignored. Other unknown fields are ignored (the OAS does not
//! forbid additional properties).

use std::sync::Arc;

use fastvideo_protocol::{
    Anchor, ApiError, AudioOut, CanvasSpec, GapId, GenerationRequest, HttpReply, Job, Keyframe,
    Length, MediaRef, NormalizeCtx, Snap, SubmitEndpoint, Task, TimingSpec, ViewCtx,
};
use serde_json::{Map, Value};

use crate::error::Api;
use crate::models::{self, LtxModels, Resolution};

/// Longest prompt (OAS `maxLength`).
pub const PROMPT_MAX_CHARS: usize = 5000;

/// `camera_motion` values (OAS enum, ltx §2.0).
pub const CAMERA_MOTIONS: [&str; 8] = [
    "dolly_in",
    "dolly_out",
    "dolly_left",
    "dolly_right",
    "jib_up",
    "jib_down",
    "static",
    "focus_shift",
];

/// The generation endpoints we serve.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Endpoint {
    TextToVideo,
    ImageToVideo,
}

impl Endpoint {
    pub const ALL: [Endpoint; 2] = [Endpoint::TextToVideo, Endpoint::ImageToVideo];

    /// The path segment (`text-to-video`).
    pub fn segment(&self) -> &'static str {
        match self {
            Endpoint::TextToVideo => "text-to-video",
            Endpoint::ImageToVideo => "image-to-video",
        }
    }
    pub fn from_segment(s: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|e| e.segment() == s)
    }
    /// The endpoint a job was submitted on (from its task).
    pub fn of_task(t: Task) -> Option<Self> {
        match t {
            Task::T2V => Some(Endpoint::TextToVideo),
            Task::I2V | Task::Keyframes => Some(Endpoint::ImageToVideo),
            _ => None,
        }
    }
}

/// One submit endpoint on one surface. `Body` is raw JSON so validation
/// messages follow the LTX wording.
#[derive(Clone, Debug)]
pub struct Submit {
    pub endpoint: Endpoint,
    pub api: Api,
    pub models: Arc<LtxModels>,
}

impl Submit {
    pub fn new(endpoint: Endpoint, api: Api, models: Arc<LtxModels>) -> Self {
        Self {
            endpoint,
            api,
            models,
        }
    }
}

impl SubmitEndpoint for Submit {
    type Body = Value;

    fn normalize(&self, body: Value, _cx: &NormalizeCtx) -> Result<GenerationRequest, ApiError> {
        normalize(self.endpoint, self.api, &self.models, &body)
    }

    /// `202 {id, created_at}` (v2). The v1 routes answer with the video
    /// bytes instead ([`crate::v1`]).
    fn submit_reply(&self, job: &Job, _cx: &ViewCtx) -> HttpReply {
        crate::v2::created_reply(job)
    }
}

fn invalid(param: &str, msg: impl Into<String>) -> ApiError {
    ApiError::invalid_param(param, msg)
}

/// A present, non-null field.
fn field<'a>(o: &'a Map<String, Value>, k: &str) -> Option<&'a Value> {
    o.get(k).filter(|v| !v.is_null())
}

fn req_str<'a>(o: &'a Map<String, Value>, k: &str) -> Result<&'a str, ApiError> {
    match field(o, k) {
        None => Err(invalid(k, format!("{k} is required"))),
        Some(Value::String(s)) => Ok(s),
        Some(_) => Err(invalid(k, format!("{k} must be a string"))),
    }
}

fn opt_str<'a>(o: &'a Map<String, Value>, k: &str) -> Result<Option<&'a str>, ApiError> {
    match field(o, k) {
        None => Ok(None),
        Some(Value::String(s)) => Ok(Some(s)),
        Some(_) => Err(invalid(k, format!("{k} must be a string"))),
    }
}

/// An integer (integral floats such as `8.0` accepted).
fn as_int(v: &Value) -> Option<u64> {
    v.as_u64().or_else(|| {
        v.as_f64()
            .filter(|f| f.is_finite() && *f >= 0.0 && f.fract() == 0.0 && *f <= u32::MAX as f64)
            .map(|f| f as u64)
    })
}

fn int_field(o: &Map<String, Value>, k: &str) -> Result<Option<u32>, ApiError> {
    match field(o, k) {
        None => Ok(None),
        Some(v) => as_int(v)
            .and_then(|n| u32::try_from(n).ok())
            .map(Some)
            .ok_or_else(|| invalid(k, format!("{k} must be an integer"))),
    }
}

/// Classifies a media URI per ltx §2.0.
pub fn media_uri(s: &str, param: &str) -> Result<MediaRef, ApiError> {
    let t = s.trim();
    let bad = || {
        invalid(
            param,
            format!(
                "{param} must be an https URL, a base64 data URI or an ltx://uploads/ URI from /v1/upload"
            ),
        )
    };
    if t.starts_with("data:") || t.starts_with("ltx://uploads/") || t.starts_with("https://") {
        let r = MediaRef::parse(t, param)?;
        return match r {
            MediaRef::Http(ref u) if u.scheme() != "https" => Err(bad()),
            MediaRef::ProviderFile(_) => Err(bad()),
            r => Ok(r),
        };
    }
    Err(bad())
}

/// Validates a request body for `endpoint`.
pub fn normalize(
    endpoint: Endpoint,
    api: Api,
    models: &LtxModels,
    body: &Value,
) -> Result<GenerationRequest, ApiError> {
    let o = body
        .as_object()
        .ok_or_else(|| ApiError::invalid("request body must be a JSON object"))?;

    let prompt = req_str(o, "prompt")?;
    if prompt.chars().count() > PROMPT_MAX_CHARS {
        return Err(invalid(
            "prompt",
            format!("prompt must be at most {PROMPT_MAX_CHARS} characters"),
        ));
    }
    let model_id = req_str(o, "model")?;
    let model = models.lookup(model_id)?;
    let res = Resolution::parse(req_str(o, "resolution")?)?;

    // `duration` must be present; `null` asks for automatic duration.
    let duration = match o.get("duration") {
        None => return Err(invalid("duration", "duration is required")),
        Some(Value::Null) => None,
        Some(v) => Some(
            as_int(v)
                .and_then(|n| u32::try_from(n).ok())
                .ok_or_else(|| invalid("duration", "duration must be an integer number of seconds"))?,
        ),
    };
    let fps = int_field(o, "fps")?.unwrap_or(models::DEFAULT_FPS);
    models::check_fps(fps)?;
    if let Some(d) = duration {
        models::check_duration(model_id, model, res, fps, d)?;
    } else if !model.auto_duration {
        return Err(invalid(
            "duration",
            format!("duration must be an integer for {model_id}; automatic duration (null) is not available for this model"),
        ));
    }

    let generate_audio = match field(o, "generate_audio") {
        None => true,
        Some(Value::Bool(b)) => *b,
        Some(_) => return Err(invalid("generate_audio", "generate_audio must be a boolean")),
    };
    if let Some(cm) = opt_str(o, "camera_motion")? {
        if !CAMERA_MOTIONS.contains(&cm) {
            return Err(invalid(
                "camera_motion",
                format!("camera_motion must be one of: {}", CAMERA_MOTIONS.join(", ")),
            ));
        }
        return Err(ApiError::unsupported(GapId::LtxCameraMotion).with_param("camera_motion"));
    }

    let mut keyframes = Vec::new();
    let task = match endpoint {
        Endpoint::TextToVideo => {
            for k in ["image_uri", "last_frame_uri"] {
                if field(o, k).is_some() {
                    return Err(invalid(
                        k,
                        format!("{k} is not accepted by text-to-video; use image-to-video"),
                    ));
                }
            }
            Task::T2V
        }
        Endpoint::ImageToVideo => {
            let first = media_uri(req_str(o, "image_uri")?, "image_uri")?;
            keyframes.push(Keyframe {
                at: Anchor::First,
                image: first,
            });
            match opt_str(o, "last_frame_uri")? {
                None => Task::I2V,
                Some(last) => {
                    if duration.is_none() {
                        return Err(invalid(
                            "duration",
                            "automatic duration (null) cannot be combined with last_frame_uri",
                        ));
                    }
                    keyframes.push(Keyframe {
                        at: Anchor::Last,
                        image: media_uri(last, "last_frame_uri")?,
                    });
                    Task::Keyframes
                }
            }
        }
    };

    let mut req = GenerationRequest::text(api.protocol(), model.target.engine_name(), prompt);
    req.task = task;
    req.canvas = CanvasSpec::Exact {
        width: res.width,
        height: res.height,
    };
    req.timing = TimingSpec {
        length: match duration {
            // The matrix's longest clips (20 s at 25 fps, 10 s at 50 fps)
            // are past the engine's 481-frame grid: they run at its longest.
            Some(d) => Length::Seconds {
                value: d as f64,
                snap: Snap::Nearest,
            },
            None => Length::Auto,
        },
        fps: Some(fps),
    };
    req.keyframes = keyframes;
    if !generate_audio {
        req.audio_out = AudioOut::Silent;
    }
    Ok(req)
}

#[cfg(test)]
mod tests {
    use super::*;
    use fastvideo_protocol::ErrorKind;
    use serde_json::json;

    fn t2v(body: Value) -> Result<GenerationRequest, ApiError> {
        normalize(Endpoint::TextToVideo, Api::V2, &LtxModels::default(), &body)
    }
    fn i2v(body: Value) -> Result<GenerationRequest, ApiError> {
        normalize(Endpoint::ImageToVideo, Api::V2, &LtxModels::default(), &body)
    }
    fn base() -> Value {
        json!({"prompt": "a cat", "model": "ltx-2-5-fast", "duration": 8, "resolution": "1920x1080"})
    }
    fn with(k: &str, v: Value) -> Value {
        let mut b = base();
        b[k] = v;
        b
    }
    fn without(k: &str) -> Value {
        let mut b = base();
        b.as_object_mut().unwrap().remove(k);
        b
    }
    fn msg(r: Result<GenerationRequest, ApiError>) -> (ErrorKind, String) {
        let e = r.unwrap_err();
        (e.kind, e.message)
    }

    #[test]
    fn t2v_basic() {
        let r = t2v(base()).unwrap();
        assert_eq!(r.model, "ltx-turbo");
        assert_eq!(r.task, Task::T2V);
        assert_eq!(r.canvas, CanvasSpec::Exact { width: 1920, height: 1080 });
        assert_eq!(r.timing.fps, Some(24));
        assert_eq!(r.audio_out, AudioOut::ModelDefault);
        let r = t2v(with("generate_audio", json!(false))).unwrap();
        assert_eq!(r.audio_out, AudioOut::Silent);
        assert_eq!(t2v(with("duration", json!(8.0))).unwrap().timing, r.timing);
    }

    #[test]
    fn required_fields() {
        for k in ["prompt", "model", "resolution", "duration"] {
            let (kind, m) = msg(t2v(without(k)));
            assert_eq!(kind, ErrorKind::InvalidRequest);
            assert_eq!(m, format!("{k} is required"));
        }
        assert!(t2v(json!([1])).is_err());
    }

    #[test]
    fn matrix_enforced() {
        // fast 1080p @24: 20 s ok; pro: only 6/8/10.
        assert!(t2v(with("duration", json!(20))).is_ok());
        let mut b = with("duration", json!(20));
        b["model"] = json!("ltx-2-5-pro");
        assert!(msg(t2v(b)).1.contains("allowed: 6, 8, 10"));
        // fast 4K: 6/8/10.
        let mut b = with("resolution", json!("3840x2160"));
        b["duration"] = json!(12);
        assert!(t2v(b).is_err());
        // fast 1080p @48: 6/8/10.
        let mut b = with("fps", json!(48));
        b["duration"] = json!(12);
        assert!(t2v(b).is_err());
        assert!(t2v(with("fps", json!(48))).is_ok());
        for bad in [json!(7), json!(4), json!(22), json!(6.5), json!("8")] {
            assert!(t2v(with("duration", bad.clone())).is_err(), "{bad}");
        }
        assert!(t2v(with("fps", json!(30))).is_err());
        assert!(t2v(with("resolution", json!("1920x1088"))).is_err());
    }

    #[test]
    fn gaps() {
        // duration null: 2.5 -> Length::Auto (the gap is raised at negotiation).
        let r = t2v(with("duration", Value::Null)).unwrap();
        assert_eq!(r.timing.length, Length::Auto);
        let mut b = with("duration", Value::Null);
        b["model"] = json!("ltx-2-3-fast");
        assert_eq!(msg(t2v(b)).0, ErrorKind::InvalidRequest);
        let (k, _) = msg(t2v(with("camera_motion", json!("dolly_in"))));
        assert_eq!(k, ErrorKind::Unsupported(GapId::LtxCameraMotion));
        let (k, _) = msg(t2v(with("camera_motion", json!("zoom"))));
        assert_eq!(k, ErrorKind::InvalidRequest);
        assert!(t2v(with("camera_motion", Value::Null)).is_ok());
        for id in ["ltx-2-fast", "ltx-2-pro", "nope"] {
            assert_eq!(msg(t2v(with("model", json!(id)))).0, ErrorKind::InvalidRequest);
        }
    }

    #[test]
    fn i2v_shapes() {
        let b = with("image_uri", json!("ltx://uploads/abc"));
        let r = i2v(b.clone()).unwrap();
        assert_eq!(r.task, Task::I2V);
        assert_eq!(r.keyframes[0].image, MediaRef::Upload(fastvideo_protocol::UploadId("abc".into())));
        let mut k = b.clone();
        k["last_frame_uri"] = json!("https://example.com/l.png");
        let r = i2v(k.clone()).unwrap();
        assert_eq!(r.task, Task::Keyframes);
        assert_eq!(r.keyframes[1].at, Anchor::Last);
        k["duration"] = Value::Null;
        assert!(msg(i2v(k)).1.contains("last_frame_uri"));
        assert_eq!(msg(i2v(base())).1, "image_uri is required");
        for bad in ["http://example.com/a.png", "ftp://x/y", "mm_file://1", "ltx://other/1", "ltx://uploads/"] {
            assert!(i2v(with("image_uri", json!(bad))).is_err(), "{bad}");
        }
        assert!(i2v(with("image_uri", json!("data:image/png;base64,AAAA"))).is_ok());
        // T2V refuses image fields rather than dropping them.
        assert!(t2v(with("image_uri", json!("https://example.com/a.png"))).is_err());
    }

    #[test]
    fn prompt_limit() {
        assert!(t2v(with("prompt", json!("é".repeat(5000)))).is_ok());
        assert!(t2v(with("prompt", json!("a".repeat(5001)))).is_err());
    }
}
