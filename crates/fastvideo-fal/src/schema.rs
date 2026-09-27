//! Request and output schemas of the `minimax/h3-{max,turbo,draft}` apps
//! (design §4.4; fal §3-§5).
//!
//! Inputs are validated field by field from the JSON body, so every
//! refusal names the fal field (`ApiError::param`, rendered as the
//! pydantic-style `loc`). Unknown fields are ignored: the queue OpenAPI does
//! not set `additionalProperties: false` (fal §3.3).
//!
//! | field | default | constraints |
//! |---|---|---|
//! | `prompt` | required | 1..=50000 characters |
//! | `duration` | `5` | integer 5..=15 |
//! | `resolution` | `"768P"` | `480P`, `768P`, `1080P` |
//! | `seed` | `null` | integer or null (drawn when null) |
//! | `enable_safety_checker` | `true` | boolean (accepted, no checker: risk R8) |
//! | `sync_mode` | `false` | boolean (`video.url` becomes a `data:` URI) |
//! | `prompt_expansion_mode` | `"balanced"` | any string (accepted, no expansion) |
//! | `aspect_ratio` (t2v) | `"16:9"` | `21:9 16:9 4:3 1:1 3:4 9:16` |
//! | `aspect_ratio` (r2v) | `"adaptive"` | `adaptive` plus the t2v values |
//! | `target_audio_url` (t2v, i2v) | `null` | non-blank string |
//! | `image_url`, `end_image_url` (i2v) | `null` | strings |
//! | `reference_{image,video,audio}_urls` (r2v) | `[]` | at most 9 / 3 / 3, 12 in total |

use base64::Engine as _;
use fastvideo_protocol::{
    Anchor, ApiError, AudioInput, AudioRole, CallbackSpec, CanvasSpec, GenerationRequest, Job,
    Keyframe, Length, MediaKind, MediaRef, NormalizeCtx, ProtocolId, Ratio, Reference, Snap,
    Task, TimingSpec,
};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

/// `prompt` maxLength (fal §3.3).
pub const PROMPT_MAX_CHARS: usize = 50_000;
/// `duration` bounds in seconds (fal §3.3).
pub const DURATION_MIN: i64 = 5;
pub const DURATION_MAX: i64 = 15;
/// Reference list limits (fal §5.2).
pub const MAX_REFERENCE_IMAGES: usize = 9;
pub const MAX_REFERENCE_VIDEOS: usize = 3;
pub const MAX_REFERENCE_AUDIO: usize = 3;
pub const MAX_REFERENCES: usize = 12;
/// The hosted file name suffix (`<nanoid21>_minimax-h3.mp4`, fal §3.1).
pub const OUTPUT_SLUG: &str = "minimax-h3";

/// One of the three HTTP endpoints under an app.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Endpoint {
    TextToVideo,
    ImageToVideo,
    ReferenceToVideo,
}

impl Endpoint {
    pub const ALL: [Endpoint; 3] = [
        Endpoint::TextToVideo,
        Endpoint::ImageToVideo,
        Endpoint::ReferenceToVideo,
    ];

    /// The path segment after the app id.
    pub fn sub(&self) -> &'static str {
        match self {
            Endpoint::TextToVideo => "text-to-video",
            Endpoint::ImageToVideo => "image-to-video",
            Endpoint::ReferenceToVideo => "reference-to-video",
        }
    }

    pub fn from_sub(s: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|e| e.sub() == s)
    }
}

/// `resolution` (fal §3.3).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Resolution {
    #[serde(rename = "480P")]
    P480,
    #[serde(rename = "768P")]
    P768,
    #[serde(rename = "1080P")]
    P1080,
}

impl Resolution {
    pub const ALL: [Resolution; 3] = [Resolution::P480, Resolution::P768, Resolution::P1080];

    pub fn as_str(&self) -> &'static str {
        match self {
            Resolution::P480 => "480P",
            Resolution::P768 => "768P",
            Resolution::P1080 => "1080P",
        }
    }
    /// Short edge the canvas is generated at. 1080P is the hosted latent
    /// refinement from a 768P source; `negotiate` refuses it as
    /// `Unsupported(H3Refine1080P)`.
    pub fn short_edge(&self) -> u32 {
        match self {
            Resolution::P480 => 480,
            Resolution::P768 => 768,
            Resolution::P1080 => 1080,
        }
    }
}

/// `aspect_ratio` (t2v: all but `adaptive`; r2v: all).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AspectRatio {
    #[serde(rename = "adaptive")]
    Adaptive,
    #[serde(rename = "21:9")]
    R21x9,
    #[serde(rename = "16:9")]
    R16x9,
    #[serde(rename = "4:3")]
    R4x3,
    #[serde(rename = "1:1")]
    R1x1,
    #[serde(rename = "3:4")]
    R3x4,
    #[serde(rename = "9:16")]
    R9x16,
}

impl AspectRatio {
    /// The t2v enum, in schema order.
    pub const T2V: [AspectRatio; 6] = [
        AspectRatio::R21x9,
        AspectRatio::R16x9,
        AspectRatio::R4x3,
        AspectRatio::R1x1,
        AspectRatio::R3x4,
        AspectRatio::R9x16,
    ];
    /// The r2v enum, in schema order.
    pub const R2V: [AspectRatio; 7] = [
        AspectRatio::Adaptive,
        AspectRatio::R21x9,
        AspectRatio::R16x9,
        AspectRatio::R4x3,
        AspectRatio::R1x1,
        AspectRatio::R3x4,
        AspectRatio::R9x16,
    ];

    pub fn as_str(&self) -> &'static str {
        match self {
            AspectRatio::Adaptive => "adaptive",
            AspectRatio::R21x9 => "21:9",
            AspectRatio::R16x9 => "16:9",
            AspectRatio::R4x3 => "4:3",
            AspectRatio::R1x1 => "1:1",
            AspectRatio::R3x4 => "3:4",
            AspectRatio::R9x16 => "9:16",
        }
    }
    /// `None` for `adaptive`.
    pub fn ratio(&self) -> Option<Ratio> {
        Some(match self {
            AspectRatio::Adaptive => return None,
            AspectRatio::R21x9 => Ratio::new(21, 9),
            AspectRatio::R16x9 => Ratio::R16_9,
            AspectRatio::R4x3 => Ratio::R4_3,
            AspectRatio::R1x1 => Ratio::R1_1,
            AspectRatio::R3x4 => Ratio::R3_4,
            AspectRatio::R9x16 => Ratio::R9_16,
        })
    }
}

/// The §3.3 fields every endpoint shares, with defaults applied.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CommonInput {
    pub prompt: String,
    pub duration: u32,
    pub resolution: Resolution,
    pub seed: Option<u64>,
    pub enable_safety_checker: bool,
    pub sync_mode: bool,
    pub prompt_expansion_mode: String,
}

/// A validated input for one endpoint.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "endpoint", rename_all = "kebab-case")]
pub enum FalInput {
    TextToVideo {
        #[serde(flatten)]
        common: CommonInput,
        aspect_ratio: AspectRatio,
        target_audio_url: Option<String>,
    },
    ImageToVideo {
        #[serde(flatten)]
        common: CommonInput,
        target_audio_url: Option<String>,
        image_url: Option<String>,
        end_image_url: Option<String>,
    },
    ReferenceToVideo {
        #[serde(flatten)]
        common: CommonInput,
        aspect_ratio: AspectRatio,
        reference_image_urls: Vec<String>,
        reference_video_urls: Vec<String>,
        reference_audio_urls: Vec<String>,
    },
}

// ---------------------------------------------------------------- parsing

fn type_name(v: &Value) -> &'static str {
    match v {
        Value::Null => "null",
        Value::Bool(_) => "a boolean",
        Value::Number(_) => "a number",
        Value::String(_) => "a string",
        Value::Array(_) => "an array",
        Value::Object(_) => "an object",
    }
}

fn bad(field: impl Into<String>, msg: impl Into<String>) -> ApiError {
    ApiError::invalid_param(field, msg)
}

struct Fields<'a>(&'a Map<String, Value>);

impl<'a> Fields<'a> {
    /// The field, treating JSON `null` as absent.
    fn get(&self, k: &str) -> Option<&'a Value> {
        self.0.get(k).filter(|v| !v.is_null())
    }

    fn string(&self, k: &str) -> Result<Option<&'a str>, ApiError> {
        match self.get(k) {
            None => Ok(None),
            Some(Value::String(s)) => Ok(Some(s)),
            Some(v) => Err(bad(k, format!("Input should be a valid string, got {}", type_name(v)))),
        }
    }

    fn bool(&self, k: &str, default: bool) -> Result<bool, ApiError> {
        match self.get(k) {
            None => Ok(default),
            Some(Value::Bool(b)) => Ok(*b),
            Some(v) => Err(bad(k, format!("Input should be a valid boolean, got {}", type_name(v)))),
        }
    }

    /// An integer; `1.0` is accepted as `1` (pydantic lax mode), `1.5` is not.
    fn int(&self, k: &str) -> Result<Option<i64>, ApiError> {
        match self.get(k) {
            None => Ok(None),
            Some(Value::Number(n)) => {
                if let Some(i) = n.as_i64() {
                    return Ok(Some(i));
                }
                if n.as_u64().is_some() {
                    return Err(bad(k, "Input should be a valid integer, got a number out of range"));
                }
                match n.as_f64() {
                    Some(f) if f.fract() == 0.0 && f.abs() < 9.0e15 => Ok(Some(f as i64)),
                    _ => Err(bad(k, "Input should be a valid integer, got a number with a fractional part")),
                }
            }
            Some(v) => Err(bad(k, format!("Input should be a valid integer, got {}", type_name(v)))),
        }
    }

    fn url_list(&self, k: &str, max: usize) -> Result<Vec<String>, ApiError> {
        let Some(v) = self.get(k) else { return Ok(Vec::new()) };
        let Value::Array(items) = v else {
            return Err(bad(k, format!("Input should be a valid list, got {}", type_name(v))));
        };
        if items.len() > max {
            return Err(bad(
                k,
                format!("List should have at most {max} items after validation, not {}", items.len()),
            ));
        }
        items
            .iter()
            .enumerate()
            .map(|(i, it)| match it {
                Value::String(s) if !s.trim().is_empty() => Ok(s.clone()),
                Value::String(_) => Err(bad(format!("{k}[{i}]"), "String should match pattern '\\S'")),
                other => Err(bad(
                    format!("{k}[{i}]"),
                    format!("Input should be a valid string, got {}", type_name(other)),
                )),
            })
            .collect()
    }
}

fn parse_enum<T: Copy>(f: &Fields, k: &str, allowed: &[T], name: fn(&T) -> &'static str, default: T) -> Result<T, ApiError> {
    let Some(s) = f.string(k)? else { return Ok(default) };
    allowed.iter().copied().find(|a| name(a) == s).ok_or_else(|| {
        let list: Vec<String> = allowed.iter().map(|a| format!("'{}'", name(a))).collect();
        bad(k, format!("Input should be {}", list.join(", ")))
    })
}

fn parse_common(f: &Fields) -> Result<CommonInput, ApiError> {
    let prompt = match f.get("prompt") {
        None => return Err(bad("prompt", "Field required")),
        Some(Value::String(s)) => s.clone(),
        Some(v) => return Err(bad("prompt", format!("Input should be a valid string, got {}", type_name(v)))),
    };
    let n = prompt.chars().count();
    if n < 1 {
        return Err(bad("prompt", "String should have at least 1 character"));
    }
    if n > PROMPT_MAX_CHARS {
        return Err(bad("prompt", format!("String should have at most {PROMPT_MAX_CHARS} characters")));
    }
    let duration = f.int("duration")?.unwrap_or(DURATION_MIN);
    if duration < DURATION_MIN {
        return Err(bad("duration", format!("Input should be greater than or equal to {DURATION_MIN}")));
    }
    if duration > DURATION_MAX {
        return Err(bad("duration", format!("Input should be less than or equal to {DURATION_MAX}")));
    }
    let resolution = parse_enum(f, "resolution", &Resolution::ALL, Resolution::as_str, Resolution::P768)?;
    let seed = match f.int("seed")? {
        None => None,
        Some(s) if s < 0 => return Err(bad("seed", "Input should be greater than or equal to 0")),
        Some(s) => Some(s as u64),
    };
    Ok(CommonInput {
        prompt,
        duration: duration as u32,
        resolution,
        seed,
        enable_safety_checker: f.bool("enable_safety_checker", true)?,
        sync_mode: f.bool("sync_mode", false)?,
        prompt_expansion_mode: f.string("prompt_expansion_mode")?.unwrap_or("balanced").to_owned(),
    })
}

fn target_audio(f: &Fields) -> Result<Option<String>, ApiError> {
    match f.string("target_audio_url")? {
        None => Ok(None),
        Some(s) if s.trim().is_empty() => Err(bad("target_audio_url", "String should match pattern '\\S'")),
        Some(s) => Ok(Some(s.to_owned())),
    }
}

fn opt_url(f: &Fields, k: &str) -> Result<Option<String>, ApiError> {
    match f.string(k)? {
        None => Ok(None),
        Some(s) if s.trim().is_empty() => Err(bad(k, "String should match pattern '\\S'")),
        Some(s) => Ok(Some(s.to_owned())),
    }
}

impl FalInput {
    /// Validates a request body for `endpoint`, applying defaults.
    pub fn parse(endpoint: Endpoint, body: &Value) -> Result<Self, ApiError> {
        let Value::Object(map) = body else {
            return Err(ApiError::invalid(format!(
                "Input should be a valid dictionary, got {}",
                type_name(body)
            )));
        };
        let f = Fields(map);
        let common = parse_common(&f)?;
        Ok(match endpoint {
            Endpoint::TextToVideo => FalInput::TextToVideo {
                aspect_ratio: parse_enum(&f, "aspect_ratio", &AspectRatio::T2V, AspectRatio::as_str, AspectRatio::R16x9)?,
                target_audio_url: target_audio(&f)?,
                common,
            },
            Endpoint::ImageToVideo => FalInput::ImageToVideo {
                target_audio_url: target_audio(&f)?,
                image_url: opt_url(&f, "image_url")?,
                end_image_url: opt_url(&f, "end_image_url")?,
                common,
            },
            Endpoint::ReferenceToVideo => {
                let images = f.url_list("reference_image_urls", MAX_REFERENCE_IMAGES)?;
                let videos = f.url_list("reference_video_urls", MAX_REFERENCE_VIDEOS)?;
                let audio = f.url_list("reference_audio_urls", MAX_REFERENCE_AUDIO)?;
                let total = images.len() + videos.len() + audio.len();
                if total > MAX_REFERENCES {
                    return Err(ApiError::invalid(format!(
                        "Reference images, videos, and audio clips must add up to at most {MAX_REFERENCES} files, got {total}"
                    )));
                }
                if total == 0 {
                    return Err(ApiError::invalid(
                        "Provide at least one of reference_image_urls, reference_video_urls or reference_audio_urls",
                    ));
                }
                FalInput::ReferenceToVideo {
                    aspect_ratio: parse_enum(&f, "aspect_ratio", &AspectRatio::R2V, AspectRatio::as_str, AspectRatio::Adaptive)?,
                    reference_image_urls: images,
                    reference_video_urls: videos,
                    reference_audio_urls: audio,
                    common,
                }
            }
        })
    }

    pub fn endpoint(&self) -> Endpoint {
        match self {
            FalInput::TextToVideo { .. } => Endpoint::TextToVideo,
            FalInput::ImageToVideo { .. } => Endpoint::ImageToVideo,
            FalInput::ReferenceToVideo { .. } => Endpoint::ReferenceToVideo,
        }
    }

    pub fn common(&self) -> &CommonInput {
        match self {
            FalInput::TextToVideo { common, .. }
            | FalInput::ImageToVideo { common, .. }
            | FalInput::ReferenceToVideo { common, .. } => common,
        }
    }

    /// The normalized request (design §4.4 mapping table). `model` is the
    /// name the app resolves through (a tier alias such as `h3-max`).
    pub fn normalize(&self, model: &str, cx: &NormalizeCtx) -> Result<GenerationRequest, ApiError> {
        let c = self.common();
        let mut r = GenerationRequest::text(ProtocolId::Fal, model, c.prompt.clone());
        r.seed = c.seed;
        r.timing = TimingSpec {
            length: Length::Seconds { value: c.duration as f64, snap: Snap::AlignUp },
            fps: None,
        };
        r.output.inline_data_uri = c.sync_mode;
        r.note_noop("enable_safety_checker");
        r.note_noop("prompt_expansion_mode");
        let short_edge = c.resolution.short_edge();
        let media = |s: &str, param: &str| MediaRef::parse(s, param);
        match self {
            FalInput::TextToVideo { aspect_ratio, target_audio_url, .. } => {
                r.canvas = aspect_canvas(*aspect_ratio, short_edge);
                if let Some(a) = target_audio_url {
                    r.audio_in = Some(AudioInput { media: media(a, "target_audio_url")?, role: AudioRole::TargetSoundtrack });
                }
            }
            FalInput::ImageToVideo { target_audio_url, image_url, end_image_url, .. } => {
                if let Some(u) = image_url {
                    r.keyframes.push(Keyframe { at: Anchor::First, image: media(u, "image_url")? });
                }
                if let Some(u) = end_image_url {
                    r.keyframes.push(Keyframe { at: Anchor::Last, image: media(u, "end_image_url")? });
                }
                (r.task, r.canvas) = match (image_url.is_some(), end_image_url.is_some()) {
                    // "If both images are omitted, the request is handled as
                    // text-to-video (16:9 by default)" (fal §5.1).
                    (false, false) => (Task::T2V, aspect_canvas(AspectRatio::R16x9, short_edge)),
                    (true, false) => (Task::I2V, CanvasSpec::FollowImage { short_edge }),
                    _ => (Task::Keyframes, CanvasSpec::FollowImage { short_edge }),
                };
                if let Some(a) = target_audio_url {
                    r.audio_in = Some(AudioInput { media: media(a, "target_audio_url")?, role: AudioRole::TargetSoundtrack });
                }
            }
            FalInput::ReferenceToVideo {
                aspect_ratio,
                reference_image_urls,
                reference_video_urls,
                reference_audio_urls,
                ..
            } => {
                r.task = Task::Ref2V;
                // "Image 1… Video 1… Audio 1…": images, then videos, then audio.
                for (kind, list, field) in [
                    (MediaKind::Image, reference_image_urls, "reference_image_urls"),
                    (MediaKind::Video, reference_video_urls, "reference_video_urls"),
                    (MediaKind::Audio, reference_audio_urls, "reference_audio_urls"),
                ] {
                    for (i, u) in list.iter().enumerate() {
                        r.references.push(Reference { kind, media: media(u, &format!("{field}[{i}]"))? });
                    }
                }
                r.canvas = match aspect_ratio {
                    // INFERRED: `adaptive` follows the first reference image;
                    // with no reference image it falls back to 16:9.
                    AspectRatio::Adaptive if !reference_image_urls.is_empty() => CanvasSpec::FollowImage { short_edge },
                    AspectRatio::Adaptive => aspect_canvas(AspectRatio::R16x9, short_edge),
                    a => aspect_canvas(*a, short_edge),
                };
            }
        }
        if let Some(hook) = cx.query_param("fal_webhook") {
            let url = url::Url::parse(hook)
                .ok()
                .filter(|u| matches!(u.scheme(), "http" | "https"))
                .ok_or_else(|| bad("fal_webhook", "fal_webhook must be an http(s) URL"))?;
            r.callback = Some(CallbackSpec::FalWebhook { url });
        }
        Ok(r)
    }
}

fn aspect_canvas(a: AspectRatio, short_edge: u32) -> CanvasSpec {
    match a.ratio() {
        Some(ratio) => CanvasSpec::Aspect { ratio, short_edge },
        None => CanvasSpec::FollowImage { short_edge },
    }
}

/// Maps a normalized/ingestion param name (`keyframes[0]`, `references[3]`,
/// `audio`) back to the fal field of `req`.
pub fn fal_param(param: &str, req: &GenerationRequest) -> String {
    let idx = |p: &str, name: &str| -> Option<usize> {
        p.strip_prefix(name)?.strip_prefix('[')?.strip_suffix(']')?.parse().ok()
    };
    if let Some(i) = idx(param, "keyframes") {
        return match req.keyframes.get(i).map(|k| k.at) {
            Some(Anchor::Last) => "end_image_url".into(),
            _ => "image_url".into(),
        };
    }
    if let Some(i) = idx(param, "references") {
        if let Some(r) = req.references.get(i) {
            let k = req.references[..i].iter().filter(|x| x.kind == r.kind).count();
            let field = match r.kind {
                MediaKind::Image => "reference_image_urls",
                MediaKind::Video => "reference_video_urls",
                MediaKind::Audio => "reference_audio_urls",
            };
            return format!("{field}[{k}]");
        }
    }
    match param {
        "audio" | "audio_url" => "target_audio_url".into(),
        "references" => match req.references.first().map(|r| r.kind) {
            Some(MediaKind::Video) => "reference_video_urls".into(),
            Some(MediaKind::Audio) => "reference_audio_urls".into(),
            _ => "reference_image_urls".into(),
        },
        other => other.into(),
    }
}

/// `loc` for a param: `["body"]`, `["body", "seed"]`,
/// `["body", "reference_image_urls", 3]`.
pub fn loc(param: Option<&str>) -> Value {
    let root = match param {
        Some("fal_webhook" | "fal_max_queue_length") => "query",
        _ => "body",
    };
    let mut v = vec![Value::from(root)];
    if let Some(p) = param.filter(|p| !p.is_empty() && *p != "task") {
        match p.split_once('[') {
            Some((name, rest)) => {
                v.push(name.into());
                match rest.trim_end_matches(']').parse::<u64>() {
                    Ok(i) => v.push(i.into()),
                    Err(_) => v.push(rest.trim_end_matches(']').into()),
                }
            }
            None => v.push(p.into()),
        }
    }
    Value::Array(v)
}

// ---------------------------------------------------------------- outputs

/// `File` (fal §3.1).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct File {
    pub url: String,
    pub content_type: Option<String>,
    pub file_name: Option<String>,
    pub file_size: Option<u64>,
}

/// t2v / i2v / r2v output (fal §3.4, §5.2). `seed` is set on r2v only.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct VideoOutput {
    pub video: File,
    pub expanded_prompt: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub seed: Option<u64>,
    pub timings: Option<serde_json::Map<String, Value>>,
}

/// The output artifact's file name: `<nanoid21>_minimax-h3.mp4`, derived
/// from the job id so it is stable (fal §3.1). The binary names fal
/// artifacts with it (`ArtifactMeta::file_name`).
pub fn output_file_name(job: &Job) -> String {
    let b64 = base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(job.id.0.as_bytes());
    format!("{}_{OUTPUT_SLUG}.mp4", &b64[..21])
}

/// Whether the job asked for `sync_mode` (inline data URI).
pub fn wants_inline(job: &Job) -> bool {
    job.request_echo.get("sync_mode").and_then(Value::as_bool).unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use time::macros::datetime;

    fn cx() -> NormalizeCtx {
        NormalizeCtx::new(datetime!(2026-09-27 12:00 UTC))
    }

    #[test]
    fn loc_forms() {
        assert_eq!(loc(None), json!(["body"]));
        assert_eq!(loc(Some("task")), json!(["body"]));
        assert_eq!(loc(Some("seed")), json!(["body", "seed"]));
        assert_eq!(loc(Some("reference_image_urls[3]")), json!(["body", "reference_image_urls", 3]));
        assert_eq!(loc(Some("fal_webhook")), json!(["query", "fal_webhook"]));
    }

    #[test]
    fn param_mapping() {
        let body = json!({"prompt": "p", "end_image_url": "https://a.test/x.png"});
        let r = FalInput::parse(Endpoint::ImageToVideo, &body).unwrap().normalize("h3-max", &cx()).unwrap();
        assert_eq!(fal_param("keyframes[0]", &r), "end_image_url");
        let body = json!({"prompt": "p", "reference_image_urls": ["https://a.test/1.png"], "reference_audio_urls": ["https://a.test/a.wav", "https://a.test/b.wav"]});
        let r = FalInput::parse(Endpoint::ReferenceToVideo, &body).unwrap().normalize("h3-max", &cx()).unwrap();
        assert_eq!(fal_param("references[2]", &r), "reference_audio_urls[1]");
        assert_eq!(fal_param("audio", &r), "target_audio_url");
        assert_eq!(fal_param("duration", &r), "duration");
    }
}
