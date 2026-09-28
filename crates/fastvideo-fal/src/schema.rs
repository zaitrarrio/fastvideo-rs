//! Request and output schemas of the fal apps (design §4.4; fal §3-§5;
//! docs/serve/fal-parity.md).
//!
//! Each app follows one family's schema ([`AppKind`], from its id):
//!
//! | App | Kind | Endpoints (sub-paths) |
//! |---|---|---|
//! | `minimax/h3-max`, `minimax/h3-max-turbo`, `minimax/h3-{turbo,draft}`, any other `owner/alias` | [`AppKind::H3`] | `text-to-video`, `image-to-video`, `reference-to-video` |
//! | `minimax/h3` (base) | [`AppKind::H3Base`] | the same; `resolution` `480P 768P 2K 4K` |
//! | `lightricks/ltx-2.5` | [`AppKind::Ltx25`] | `{text,image}-to-video/{fast,pro}` ([`ltx`]) |
//! | `fal-ai/ltx-2.3-quality` | [`AppKind::LtxQuality`] | `ingredient` ([`ingredient`]: reference-sheet video, `ltx-pro` Ref2V) |
//! | `fal-ai/wan` | [`AppKind::Wan`] | `v2.2-5b/text-to-video`, `v2.2-5b/image-to-video`, `v2.2-5b/text-to-video/fast-wan` ([`wan`]) |
//!
//! The H3 fields:
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
//!
//! `minimax/h3` (base) lists `480P`, `768P`, `2K` and `4K` (fal's default is
//! `2K`, an upscale from 768P). 2K and 4K normalize to the 1440 and 2160
//! short edges, which the H3 caps refuse as `Unsupported(H3Resolution2K)`,
//! so an omitted `resolution` means `768P` here.

pub mod ingredient;
pub mod ltx;
pub mod wan;

use base64::Engine as _;
use fastvideo_protocol::{
    Anchor, ApiError, AudioInput, AudioRole, CallbackSpec, CanvasSpec, Family, GenerationRequest,
    Job, Keyframe, Length, MediaKind, MediaRef, NormalizeCtx, ProtocolId, Ratio, Reference, Snap,
    Task, Tier, TimingSpec,
};

pub use ingredient::IngredientInput;
pub use ltx::{LtxClass, LtxInput};
pub use wan::{WanInput, WanVariant};
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

/// One HTTP endpoint under an app. The sub-path may have several segments
/// (`text-to-video/fast`, `v2.2-5b/text-to-video/fast-wan`); every sub is
/// unique across families, so the sub alone names the endpoint.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Endpoint {
    TextToVideo,
    ImageToVideo,
    ReferenceToVideo,
    /// `lightricks/ltx-2.5/text-to-video/fast`.
    LtxTextToVideoFast,
    /// `lightricks/ltx-2.5/text-to-video/pro`.
    LtxTextToVideoPro,
    /// `lightricks/ltx-2.5/image-to-video/fast`.
    LtxImageToVideoFast,
    /// `lightricks/ltx-2.5/image-to-video/pro`.
    LtxImageToVideoPro,
    /// `fal-ai/wan/v2.2-5b/text-to-video`.
    WanTextToVideo,
    /// `fal-ai/wan/v2.2-5b/image-to-video`.
    WanImageToVideo,
    /// `fal-ai/wan/v2.2-5b/text-to-video/fast-wan`.
    WanFastWan,
    /// `fal-ai/ltx-2.3-quality/ingredient` (reference sheet, IC-LoRA).
    LtxIngredient,
}

impl Endpoint {
    /// The H3 endpoints (every H3-schema app has these three).
    pub const ALL: [Endpoint; 3] = [
        Endpoint::TextToVideo,
        Endpoint::ImageToVideo,
        Endpoint::ReferenceToVideo,
    ];
    /// `lightricks/ltx-2.5`.
    pub const LTX: [Endpoint; 4] = [
        Endpoint::LtxTextToVideoFast,
        Endpoint::LtxTextToVideoPro,
        Endpoint::LtxImageToVideoFast,
        Endpoint::LtxImageToVideoPro,
    ];
    /// `fal-ai/wan`.
    pub const WAN: [Endpoint; 3] = [Endpoint::WanTextToVideo, Endpoint::WanImageToVideo, Endpoint::WanFastWan];
    /// `fal-ai/ltx-2.3-quality`.
    pub const LTX_QUALITY: [Endpoint; 1] = [Endpoint::LtxIngredient];
    /// Every endpoint of every family.
    pub const EVERY: [Endpoint; 11] = [
        Endpoint::TextToVideo,
        Endpoint::ImageToVideo,
        Endpoint::ReferenceToVideo,
        Endpoint::LtxTextToVideoFast,
        Endpoint::LtxTextToVideoPro,
        Endpoint::LtxImageToVideoFast,
        Endpoint::LtxImageToVideoPro,
        Endpoint::WanTextToVideo,
        Endpoint::WanImageToVideo,
        Endpoint::WanFastWan,
        Endpoint::LtxIngredient,
    ];

    /// The path after the app id (one or more segments).
    pub fn sub(&self) -> &'static str {
        match self {
            Endpoint::TextToVideo => "text-to-video",
            Endpoint::ImageToVideo => "image-to-video",
            Endpoint::ReferenceToVideo => "reference-to-video",
            Endpoint::LtxTextToVideoFast => "text-to-video/fast",
            Endpoint::LtxTextToVideoPro => "text-to-video/pro",
            Endpoint::LtxImageToVideoFast => "image-to-video/fast",
            Endpoint::LtxImageToVideoPro => "image-to-video/pro",
            Endpoint::WanTextToVideo => "v2.2-5b/text-to-video",
            Endpoint::WanImageToVideo => "v2.2-5b/image-to-video",
            Endpoint::WanFastWan => "v2.2-5b/text-to-video/fast-wan",
            Endpoint::LtxIngredient => "ingredient",
        }
    }

    pub fn from_sub(s: &str) -> Option<Self> {
        Self::EVERY.into_iter().find(|e| e.sub() == s)
    }

    /// The console / catalog title.
    pub fn title(&self) -> &'static str {
        match self {
            Endpoint::TextToVideo => "Text to Video",
            Endpoint::ImageToVideo => "Image to Video",
            Endpoint::ReferenceToVideo => "Reference to Video",
            Endpoint::LtxTextToVideoFast => "Text to Video · Fast",
            Endpoint::LtxTextToVideoPro => "Text to Video · Pro",
            Endpoint::LtxImageToVideoFast => "Image to Video · Fast",
            Endpoint::LtxImageToVideoPro => "Image to Video · Pro",
            Endpoint::WanTextToVideo => "Text to Video · 5B",
            Endpoint::WanImageToVideo => "Image to Video · 5B",
            Endpoint::WanFastWan => "Text to Video · FastWan",
            Endpoint::LtxIngredient => "Reference Sheet to Video · Ingredients",
        }
    }

    /// The family tier an LTX or Wan endpoint runs on (`None` for the H3
    /// endpoints: they run the app's own model).
    pub fn target(&self) -> Option<(Family, Tier)> {
        Some(match self {
            Endpoint::TextToVideo | Endpoint::ImageToVideo | Endpoint::ReferenceToVideo => return None,
            Endpoint::LtxTextToVideoFast | Endpoint::LtxImageToVideoFast => (Family::Ltx2, Tier::Turbo),
            // `ltx-pro`: its Ref2V requests route to the IC-LoRA companion.
            Endpoint::LtxTextToVideoPro | Endpoint::LtxImageToVideoPro | Endpoint::LtxIngredient => {
                (Family::Ltx2, Tier::Max)
            }
            Endpoint::WanTextToVideo | Endpoint::WanImageToVideo => (Family::Wan, Tier::Max),
            Endpoint::WanFastWan => (Family::Wan, Tier::Turbo),
        })
    }
}

/// Which family schema an app follows (from its id).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AppKind {
    /// `minimax/h3-max[-turbo]`, `minimax/h3-{turbo,draft}` and any other
    /// `owner/alias`: the H3 Max schema.
    H3,
    /// `minimax/h3`: the base H3 schema (`2K`/`4K` listed, refused).
    H3Base,
    /// `lightricks/ltx-2.5`.
    Ltx25,
    /// `fal-ai/wan`.
    Wan,
    /// `fal-ai/ltx-2.3-quality` (the `ingredient` endpoint).
    LtxQuality,
}

impl AppKind {
    pub fn of(app_id: &str) -> Self {
        match app_id {
            "minimax/h3" => AppKind::H3Base,
            "lightricks/ltx-2.5" => AppKind::Ltx25,
            "fal-ai/wan" => AppKind::Wan,
            "fal-ai/ltx-2.3-quality" => AppKind::LtxQuality,
            _ => AppKind::H3,
        }
    }
    pub fn endpoints(self) -> &'static [Endpoint] {
        match self {
            AppKind::H3 | AppKind::H3Base => &Endpoint::ALL,
            AppKind::Ltx25 => &Endpoint::LTX,
            AppKind::Wan => &Endpoint::WAN,
            AppKind::LtxQuality => &Endpoint::LTX_QUALITY,
        }
    }
    /// Whether the app has the H3 WMA director (`{app}/director`).
    pub fn director(self) -> bool {
        matches!(self, AppKind::H3 | AppKind::H3Base)
    }
    /// The H3 schemas' `resolution` enum.
    pub fn h3_resolutions(self) -> &'static [Resolution] {
        match self {
            AppKind::H3Base => &Resolution::BASE,
            _ => &Resolution::ALL,
        }
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
    /// `minimax/h3` only (an upscale from 768P on fal).
    #[serde(rename = "2K")]
    K2,
    /// `minimax/h3` only.
    #[serde(rename = "4K")]
    K4,
}

impl Resolution {
    /// The H3 Max enum (`minimax/h3-max[-turbo]`).
    pub const ALL: [Resolution; 3] = [Resolution::P480, Resolution::P768, Resolution::P1080];
    /// The base H3 enum (`minimax/h3`).
    pub const BASE: [Resolution; 4] = [Resolution::P480, Resolution::P768, Resolution::K2, Resolution::K4];

    pub fn as_str(&self) -> &'static str {
        match self {
            Resolution::P480 => "480P",
            Resolution::P768 => "768P",
            Resolution::P1080 => "1080P",
            Resolution::K2 => "2K",
            Resolution::K4 => "4K",
        }
    }
    /// Short edge the canvas is generated at. 1080P is the hosted latent
    /// refinement from a 768P source; `negotiate` refuses it as
    /// `Unsupported(H3Refine1080P)`, and 2K / 4K (1440 / 2160) as
    /// `Unsupported(H3Resolution2K)`.
    pub fn short_edge(&self) -> u32 {
        match self {
            Resolution::P480 => 480,
            Resolution::P768 => 768,
            Resolution::P1080 => 1080,
            Resolution::K2 => 1440,
            Resolution::K4 => 2160,
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
    /// `lightricks/ltx-2.5/*`.
    Ltx(LtxInput),
    /// `fal-ai/wan/v2.2-5b/*`.
    Wan(WanInput),
    /// `fal-ai/ltx-2.3-quality/ingredient`.
    Ingredient(IngredientInput),
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

fn parse_prompt(f: &Fields, max: usize) -> Result<String, ApiError> {
    let prompt = match f.get("prompt") {
        None => return Err(bad("prompt", "Field required")),
        Some(Value::String(s)) => s.clone(),
        Some(v) => return Err(bad("prompt", format!("Input should be a valid string, got {}", type_name(v)))),
    };
    let n = prompt.chars().count();
    if n < 1 {
        return Err(bad("prompt", "String should have at least 1 character"));
    }
    if n > max {
        return Err(bad("prompt", format!("String should have at most {max} characters")));
    }
    Ok(prompt)
}

fn parse_seed(f: &Fields) -> Result<Option<u64>, ApiError> {
    match f.int("seed")? {
        None => Ok(None),
        Some(s) if s < 0 => Err(bad("seed", "Input should be greater than or equal to 0")),
        Some(s) => Ok(Some(s as u64)),
    }
}

fn parse_common(f: &Fields, resolutions: &[Resolution]) -> Result<CommonInput, ApiError> {
    let prompt = parse_prompt(f, PROMPT_MAX_CHARS)?;
    let duration = f.int("duration")?.unwrap_or(DURATION_MIN);
    if duration < DURATION_MIN {
        return Err(bad("duration", format!("Input should be greater than or equal to {DURATION_MIN}")));
    }
    if duration > DURATION_MAX {
        return Err(bad("duration", format!("Input should be less than or equal to {DURATION_MAX}")));
    }
    let resolution = parse_enum(f, "resolution", resolutions, Resolution::as_str, Resolution::P768)?;
    let seed = parse_seed(f)?;
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
    /// Validates a request body for `endpoint`, applying defaults. The H3
    /// endpoints follow the H3 Max schema; see [`FalInput::parse_for`].
    pub fn parse(endpoint: Endpoint, body: &Value) -> Result<Self, ApiError> {
        Self::parse_for(AppKind::H3, endpoint, body)
    }

    /// Validates a request body for `endpoint` of an app of `kind` (the
    /// kind only matters for the H3 endpoints: `minimax/h3`'s resolutions).
    pub fn parse_for(kind: AppKind, endpoint: Endpoint, body: &Value) -> Result<Self, ApiError> {
        let Value::Object(map) = body else {
            return Err(ApiError::invalid(format!(
                "Input should be a valid dictionary, got {}",
                type_name(body)
            )));
        };
        let f = Fields(map);
        match endpoint {
            Endpoint::LtxTextToVideoFast => return Ok(FalInput::Ltx(ltx::parse(&f, LtxClass::Fast, false)?)),
            Endpoint::LtxTextToVideoPro => return Ok(FalInput::Ltx(ltx::parse(&f, LtxClass::Pro, false)?)),
            Endpoint::LtxImageToVideoFast => return Ok(FalInput::Ltx(ltx::parse(&f, LtxClass::Fast, true)?)),
            Endpoint::LtxImageToVideoPro => return Ok(FalInput::Ltx(ltx::parse(&f, LtxClass::Pro, true)?)),
            Endpoint::WanTextToVideo => return Ok(FalInput::Wan(wan::parse(&f, WanVariant::TextToVideo)?)),
            Endpoint::WanImageToVideo => return Ok(FalInput::Wan(wan::parse(&f, WanVariant::ImageToVideo)?)),
            Endpoint::WanFastWan => return Ok(FalInput::Wan(wan::parse(&f, WanVariant::FastWan)?)),
            Endpoint::LtxIngredient => return Ok(FalInput::Ingredient(ingredient::parse(&f)?)),
            Endpoint::TextToVideo | Endpoint::ImageToVideo | Endpoint::ReferenceToVideo => {}
        }
        let common = parse_common(&f, kind.h3_resolutions())?;
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
            _ => return Err(ApiError::internal("not an H3 endpoint")),
        })
    }

    pub fn endpoint(&self) -> Endpoint {
        match self {
            FalInput::TextToVideo { .. } => Endpoint::TextToVideo,
            FalInput::ImageToVideo { .. } => Endpoint::ImageToVideo,
            FalInput::ReferenceToVideo { .. } => Endpoint::ReferenceToVideo,
            FalInput::Ltx(i) => match (i.class, i.image_to_video) {
                (LtxClass::Fast, false) => Endpoint::LtxTextToVideoFast,
                (LtxClass::Pro, false) => Endpoint::LtxTextToVideoPro,
                (LtxClass::Fast, true) => Endpoint::LtxImageToVideoFast,
                (LtxClass::Pro, true) => Endpoint::LtxImageToVideoPro,
            },
            FalInput::Wan(i) => match i.variant {
                WanVariant::TextToVideo => Endpoint::WanTextToVideo,
                WanVariant::ImageToVideo => Endpoint::WanImageToVideo,
                WanVariant::FastWan => Endpoint::WanFastWan,
            },
            FalInput::Ingredient(_) => Endpoint::LtxIngredient,
        }
    }

    /// The H3 fields (`None` for the LTX and Wan inputs).
    pub fn common(&self) -> Option<&CommonInput> {
        match self {
            FalInput::TextToVideo { common, .. }
            | FalInput::ImageToVideo { common, .. }
            | FalInput::ReferenceToVideo { common, .. } => Some(common),
            FalInput::Ltx(_) | FalInput::Wan(_) | FalInput::Ingredient(_) => None,
        }
    }

    fn normalize_h3(&self, model: &str) -> Result<GenerationRequest, ApiError> {
        let Some(c) = self.common() else { return Err(ApiError::internal("not an H3 input")) };
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
            FalInput::Ltx(_) | FalInput::Wan(_) | FalInput::Ingredient(_) => {}
        }
        Ok(r)
    }

    /// The normalized request (design §4.4 mapping table). `model` is the
    /// name the endpoint resolves through (a tier alias such as `h3-max`).
    pub fn normalize(&self, model: &str, cx: &NormalizeCtx) -> Result<GenerationRequest, ApiError> {
        let mut r = match self {
            FalInput::Ltx(i) => i.normalize(model)?,
            FalInput::Wan(i) => i.normalize(model)?,
            FalInput::Ingredient(i) => i.normalize(model)?,
            _ => self.normalize_h3(model)?,
        };
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

/// An omitted `resolution` defaults to `768P` (the MiniMax H3 contract). A
/// model without a 768 tier (an LTX app, `h3-draft`) gets its own first
/// tier instead, so a body without `resolution` runs on every app.
/// `short_edges` is the resolved model's `CanvasCaps::short_edges`.
pub fn default_resolution_for(req: &mut GenerationRequest, short_edges: &[u32]) {
    let default = Resolution::P768.short_edge();
    let Some(&first) = short_edges.first() else { return };
    if short_edges.contains(&default) {
        return;
    }
    match &mut req.canvas {
        CanvasSpec::Aspect { short_edge, .. } | CanvasSpec::FollowImage { short_edge } if *short_edge == default => {
            *short_edge = first;
        }
        _ => {}
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
    // `ingredient` (the only fal input that sets the reference LoRA strength)
    // names its sheet `image_url` and the LoRA strength `ingredient_strength`.
    let ingredient = req.sampling.reference_lora_strength.is_some();
    if ingredient && (param == "references" || idx(param, "references").is_some()) {
        return "image_url".into();
    }
    if ingredient && param == "reference_lora_strength" {
        return "ingredient_strength".into();
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
    // The Wan schema sends `num_frames` / `frames_per_second` (Length::Frames);
    // H3 and LTX send `duration` / `fps`.
    let wan = matches!(req.timing.length, Length::Frames { .. });
    match param {
        "fps" if wan => "frames_per_second".into(),
        "flow_shift" => "shift".into(),
        "image_uri" => "image_url".into(),
        "last_frame_uri" => "end_image_url".into(),
        "size" => "resolution".into(),
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

/// t2v / i2v / r2v output (fal §3.4, §5.2). `seed` is the effective seed
/// (requested or drawn) on every task: required by fal's r2v schema, an
/// extra key on t2v/i2v.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct VideoOutput {
    pub video: File,
    pub expanded_prompt: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub seed: Option<u64>,
    pub timings: Option<serde_json::Map<String, Value>>,
}

/// The file name slug of an app's outputs: `minimax-<alias>` for the
/// `minimax/*` apps (hosted fal writes `minimax-h3` for all of them; the
/// alias keeps the tier: `minimax-h3-max`, `minimax-h3-turbo`), else the app
/// alias (`ltx-2.5`, `wan`, `fastwan21-1.3b`), limited to `[A-Za-z0-9._-]`.
pub fn output_slug(app_id: &str) -> String {
    let (owner, alias) = app_id.split_once('/').unwrap_or(("", app_id));
    let clean = |s: &str| -> String {
        s.chars().map(|c| if c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.') { c } else { '-' }).collect()
    };
    match (owner, alias) {
        (_, "") => "video".to_owned(),
        ("minimax", a) => format!("minimax-{}", clean(a)),
        (_, a) => clean(a),
    }
}

/// The output artifact's file name: `<nanoid21>_<slug>.mp4` in hosted fal's
/// form (fal §3.1), with the 21 characters derived from the job id so it is
/// stable. `slug` is [`output_slug`] of the job's app plus `-<tier>` when
/// the resolved tier is not already a word of it (`wan` at turbo →
/// `wan-turbo`, `ltx-2.5` at max → `ltx-2.5-max`; `minimax-h3-max` stays).
/// The binary names fal artifacts with it (`ArtifactMeta::file_name`).
pub fn output_file_name(job: &Job) -> String {
    let b64 = base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(job.id.0.as_bytes());
    let mut slug = output_slug(app_id(job.requested_model()));
    if let Some(t) = job.resolved.tier.map(|t| t.as_str()) {
        if !slug.split(['-', '_', '.']).any(|w| w == t) {
            slug.push('-');
            slug.push_str(t);
        }
    }
    format!("{}_{slug}.mp4", &b64[..21])
}

/// The app id of an endpoint id (`fal-ai/wan/v2.2-5b/text-to-video/fast-wan`
/// → `fal-ai/wan`): the longest known sub-path suffix is dropped. An id
/// with no known sub is returned as is.
pub fn app_id(endpoint_id: &str) -> &str {
    Endpoint::EVERY
        .iter()
        .filter_map(|e| endpoint_id.strip_suffix(e.sub())?.strip_suffix('/'))
        .filter(|app| app.split('/').count() == 2)
        .min_by_key(|app| app.len())
        .unwrap_or(endpoint_id)
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

    #[test]
    fn ingredient_is_a_reference_sheet_request_on_ltx_pro() {
        let kind = AppKind::of("fal-ai/ltx-2.3-quality");
        assert_eq!(kind, AppKind::LtxQuality);
        assert_eq!(kind.endpoints(), &[Endpoint::LtxIngredient]);
        assert_eq!(Endpoint::from_sub("ingredient"), Some(Endpoint::LtxIngredient));
        assert_eq!(Endpoint::LtxIngredient.target(), Some((Family::Ltx2, Tier::Max)));
        assert_eq!(app_id("fal-ai/ltx-2.3-quality/ingredient"), "fal-ai/ltx-2.3-quality");
        let body = json!({
            "prompt": "Reference sheet: a crab. Generated video: the crab walks.",
            "image_url": "https://a.test/sheet.png",
            "ingredient_strength": 1.5,
            "num_frames": 100,
            "generate_audio": false,
            "negative_prompt": "blurry",
        });
        let i = FalInput::parse_for(kind, Endpoint::LtxIngredient, &body).unwrap();
        assert_eq!(i.endpoint(), Endpoint::LtxIngredient);
        let r = i.normalize("ltx-pro", &cx()).unwrap();
        assert_eq!(r.task, Task::Ref2V);
        assert_eq!(r.model, "ltx-pro");
        assert_eq!(r.references.len(), 1);
        assert_eq!(r.references[0].kind, MediaKind::Image);
        assert_eq!(r.canvas, CanvasSpec::Exact { width: 1536, height: 896 });
        assert_eq!(r.timing.length, Length::Frames { value: 100, snap: Snap::AlignUp });
        assert_eq!(r.timing.fps, Some(24));
        assert_eq!(r.sampling.reference_lora_strength, Some(1.5));
        assert_eq!(r.sampling.reference_strength, Some(1.0));
        assert_eq!(r.audio_out, fastvideo_protocol::AudioOut::Silent);
        // Engine errors name the fal fields.
        assert_eq!(fal_param("references[0]", &r), "image_url");
        assert_eq!(fal_param("reference_lora_strength", &r), "ingredient_strength");
        assert_eq!(fal_param("reference_strength", &r), "reference_strength");
        assert_eq!(fal_param("fps", &r), "frames_per_second");
        // The sheet is required; strengths stay within fal's 0..=2.
        let no_sheet = json!({"prompt": "p"});
        assert!(FalInput::parse_for(kind, Endpoint::LtxIngredient, &no_sheet).is_err());
        let hot = json!({"prompt": "p", "image_url": "https://a.test/s.png", "reference_strength": 2.5});
        let e = FalInput::parse_for(kind, Endpoint::LtxIngredient, &hot).unwrap_err();
        assert_eq!(e.param.as_deref(), Some("reference_strength"));
    }

    #[test]
    fn omitted_resolution_follows_the_model_tiers() {
        let t2v = |b: Value| FalInput::parse(Endpoint::TextToVideo, &b).unwrap().normalize("m", &cx()).unwrap();
        let edge = |r: &GenerationRequest| match r.canvas {
            CanvasSpec::Aspect { short_edge, .. } | CanvasSpec::FollowImage { short_edge } => short_edge,
            _ => 0,
        };
        // H3 (768 served): unchanged.
        let mut r = t2v(json!({"prompt": "p"}));
        default_resolution_for(&mut r, &[768, 480]);
        assert_eq!(edge(&r), 768);
        // LTX (no 768 tier): the model's first tier; h3-draft: 480.
        default_resolution_for(&mut r, &[1080, 720, 1440, 2160]);
        assert_eq!(edge(&r), 1080);
        let mut r = t2v(json!({"prompt": "p", "aspect_ratio": "9:16"}));
        default_resolution_for(&mut r, &[480]);
        assert_eq!(edge(&r), 480);
        assert!(matches!(r.canvas, CanvasSpec::Aspect { ratio, .. } if ratio.w == 9 && ratio.h == 16));
    }
}
