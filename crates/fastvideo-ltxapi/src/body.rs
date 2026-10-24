//! The typed LTX request bodies, field for field from
//! `docs/serve/specs/ltx/openapi.json` (`TextToVideoRequest`,
//! `ImageToVideoRequest`, `AudioToVideoRequest`, `EditVideoRequest`,
//! `ExtendVideoRequest`), plus our `enhance_prompt` (plan E2).
//!
//! [`crate::request`] reads a JSON body into these types, validating field
//! by field in the order the LTX API reports errors, and builds the
//! `GenerationRequest` from the typed value. `Serialize` gives the wire
//! shape back (every property, `null` when absent), which the tests compare
//! with the OpenAPI document.

use fastvideo_protocol::ApiError;
use serde::Serialize;
use serde_json::{Map, Value};

/// `POST /v{1,2}/text-to-video`.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct TextToVideoRequest {
    pub prompt: String,
    pub model: String,
    /// Seconds per the support matrix; `None` is `null`: automatic duration
    /// (the key itself is required).
    pub duration: Option<u32>,
    /// `WIDTHxHEIGHT`.
    pub resolution: String,
    pub fps: Option<u32>,
    pub generate_audio: Option<bool>,
    pub camera_motion: Option<String>,
    /// Ours (named after the LTX CLI's `--enhance-prompt`): read leniently,
    /// a non-boolean is ignored.
    pub enhance_prompt: Option<bool>,
}

/// `POST /v{1,2}/image-to-video`.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ImageToVideoRequest {
    pub image_uri: String,
    pub last_frame_uri: Option<String>,
    #[serde(flatten)]
    pub base: TextToVideoRequest,
}

/// `POST /v{1,2}/audio-to-video`.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct AudioToVideoRequest {
    pub audio_uri: String,
    pub image_uri: Option<String>,
    pub last_frame_uri: Option<String>,
    /// Required without `image_uri`; may be empty with one.
    pub prompt: Option<String>,
    pub model: Option<String>,
    pub resolution: Option<String>,
    pub fps: Option<u32>,
    pub camera_motion: Option<String>,
}

/// `POST /v{1,2}/retake`.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct EditVideoRequest {
    pub video_uri: String,
    pub prompt: Option<String>,
    pub model: Option<String>,
    pub start_time: f64,
    pub duration: f64,
    pub mode: Option<String>,
    pub resolution: Option<String>,
}

/// `POST /v{1,2}/extend`.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ExtendVideoRequest {
    pub video_uri: String,
    pub prompt: Option<String>,
    pub model: Option<String>,
    pub duration: f64,
    pub mode: Option<String>,
    pub context: Option<f64>,
}

impl TextToVideoRequest {
    /// The OpenAPI `required` list.
    pub const REQUIRED: [&'static str; 4] = ["prompt", "model", "duration", "resolution"];
}
impl ImageToVideoRequest {
    pub const REQUIRED: [&'static str; 5] =
        ["image_uri", "prompt", "model", "duration", "resolution"];
}
impl AudioToVideoRequest {
    pub const REQUIRED: [&'static str; 1] = ["audio_uri"];
}
impl EditVideoRequest {
    pub const REQUIRED: [&'static str; 3] = ["video_uri", "start_time", "duration"];
}
impl ExtendVideoRequest {
    pub const REQUIRED: [&'static str; 2] = ["video_uri", "duration"];
}

// ---- field readers (LTX wording: `<field> is required`, `<field> must be a string`)

fn invalid(param: &str, msg: impl Into<String>) -> ApiError {
    ApiError::invalid_param(param, msg)
}

/// A JSON object body, read one typed field at a time.
pub(crate) struct Fields<'a>(pub(crate) &'a Map<String, Value>);

impl<'a> Fields<'a> {
    /// A present, non-null field.
    pub(crate) fn get(&self, k: &str) -> Option<&'a Value> {
        self.0.get(k).filter(|v| !v.is_null())
    }
    /// Whether the key is present at all (`null` included).
    pub(crate) fn raw(&self, k: &str) -> Option<&'a Value> {
        self.0.get(k)
    }
    pub(crate) fn req_str(&self, k: &str) -> Result<&'a str, ApiError> {
        match self.get(k) {
            None => Err(invalid(k, format!("{k} is required"))),
            Some(Value::String(s)) => Ok(s),
            Some(_) => Err(invalid(k, format!("{k} must be a string"))),
        }
    }
    pub(crate) fn opt_str(&self, k: &str) -> Result<Option<&'a str>, ApiError> {
        match self.get(k) {
            None => Ok(None),
            Some(Value::String(s)) => Ok(Some(s)),
            Some(_) => Err(invalid(k, format!("{k} must be a string"))),
        }
    }
    pub(crate) fn int(&self, k: &str) -> Result<Option<u32>, ApiError> {
        match self.get(k) {
            None => Ok(None),
            Some(v) => as_int(v)
                .and_then(|n| u32::try_from(n).ok())
                .map(Some)
                .ok_or_else(|| invalid(k, format!("{k} must be an integer"))),
        }
    }
    /// A number (`start_time`, `duration`, `context`).
    pub(crate) fn num(&self, k: &str) -> Result<Option<f64>, ApiError> {
        match self.get(k) {
            None => Ok(None),
            Some(v) => v
                .as_f64()
                .filter(|f| f.is_finite())
                .map(Some)
                .ok_or_else(|| invalid(k, format!("{k} must be a number"))),
        }
    }
    /// A boolean, refused when of another type.
    pub(crate) fn bool(&self, k: &str) -> Result<Option<bool>, ApiError> {
        match self.get(k) {
            None => Ok(None),
            Some(Value::Bool(b)) => Ok(Some(*b)),
            Some(_) => Err(invalid(k, format!("{k} must be a boolean"))),
        }
    }
    /// A boolean read leniently: `None` when absent or of another type.
    pub(crate) fn lenient_bool(&self, k: &str) -> Option<bool> {
        self.get(k).and_then(Value::as_bool)
    }
}

/// An integer (integral floats such as `8.0` accepted).
pub(crate) fn as_int(v: &Value) -> Option<u64> {
    v.as_u64().or_else(|| {
        v.as_f64()
            .filter(|f| f.is_finite() && *f >= 0.0 && f.fract() == 0.0 && *f <= u32::MAX as f64)
            .map(|f| f as u64)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    fn spec_schema(name: &str) -> Value {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../docs/serve/specs/ltx/openapi.json"
        );
        let doc: Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
        doc["components"]["schemas"][name].clone()
    }

    fn keys(v: &Value) -> BTreeSet<String> {
        v.as_object().unwrap().keys().cloned().collect()
    }

    fn strings(v: &Value) -> BTreeSet<String> {
        v.as_array()
            .unwrap()
            .iter()
            .map(|s| s.as_str().unwrap().to_owned())
            .collect()
    }

    fn t2v() -> TextToVideoRequest {
        TextToVideoRequest {
            prompt: "p".into(),
            model: "ltx-2-5-fast".into(),
            duration: Some(8),
            resolution: "1920x1080".into(),
            fps: Some(24),
            generate_audio: Some(true),
            camera_motion: Some("static".into()),
            enhance_prompt: Some(true),
        }
    }

    /// Every body type has exactly the OpenAPI schema's properties (plus our
    /// `enhance_prompt` on the generation bodies) and its `required` list.
    #[test]
    fn bodies_match_the_openapi_document() {
        let i2v = ImageToVideoRequest {
            image_uri: "https://a/i.png".into(),
            last_frame_uri: Some("https://a/l.png".into()),
            base: t2v(),
        };
        let a2v = AudioToVideoRequest {
            audio_uri: "https://a/a.mp3".into(),
            image_uri: None,
            last_frame_uri: None,
            prompt: None,
            model: None,
            resolution: None,
            fps: None,
            camera_motion: None,
        };
        let retake = EditVideoRequest {
            video_uri: "https://a/v.mp4".into(),
            prompt: None,
            model: None,
            start_time: 0.0,
            duration: 2.0,
            mode: None,
            resolution: None,
        };
        let extend = ExtendVideoRequest {
            video_uri: "https://a/v.mp4".into(),
            prompt: None,
            model: None,
            duration: 2.0,
            mode: None,
            context: None,
        };
        let cases: [(&str, Value, &[&str], &[&str]); 5] = [
            (
                "TextToVideoRequest",
                serde_json::to_value(t2v()).unwrap(),
                &TextToVideoRequest::REQUIRED,
                &["enhance_prompt"],
            ),
            (
                "ImageToVideoRequest",
                serde_json::to_value(i2v).unwrap(),
                &ImageToVideoRequest::REQUIRED,
                &["enhance_prompt"],
            ),
            (
                "AudioToVideoRequest",
                serde_json::to_value(a2v).unwrap(),
                &AudioToVideoRequest::REQUIRED,
                &[],
            ),
            (
                "EditVideoRequest",
                serde_json::to_value(retake).unwrap(),
                &EditVideoRequest::REQUIRED,
                &[],
            ),
            (
                "ExtendVideoRequest",
                serde_json::to_value(extend).unwrap(),
                &ExtendVideoRequest::REQUIRED,
                &[],
            ),
        ];
        for (name, body, required, ours) in cases {
            let s = spec_schema(name);
            let mut want = keys(&s["properties"]);
            want.extend(ours.iter().map(|k| (*k).to_owned()));
            assert_eq!(keys(&body), want, "{name}: properties");
            let req: BTreeSet<String> = required.iter().map(|k| (*k).to_owned()).collect();
            assert_eq!(req, strings(&s["required"]), "{name}: required");
        }
    }
}
