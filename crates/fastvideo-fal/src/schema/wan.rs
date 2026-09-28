//! `fal-ai/wan/v2.2-5b/{text-to-video, image-to-video, text-to-video/fast-wan}`
//! inputs (docs/serve/fal-parity.md §3.1). The 5B endpoints run `wan-max`
//! (Wan2.2 TI2V-5B, UniPC), `fast-wan` runs `wan-turbo` (FastWan2.2
//! TI2V-5B, DMD 3-step).
//!
//! | field | default | constraints |
//! |---|---|---|
//! | `prompt` | required | 1..=50000 characters |
//! | `negative_prompt` | `""` | string (`""`: the model's default); a no-op on `fast-wan` (DMD is unguided) |
//! | `num_frames` | `81` | integer 17..=161 (up to the Wan 4k+1 grid) |
//! | `frames_per_second` | `24` | integer 4..=60 (the MP4 rate; the frames do not depend on it) |
//! | `resolution` | `"720p"` | `580p 720p` (`fast-wan` also `480p`) |
//! | `aspect_ratio` | t2v `"16:9"`, i2v `"auto"` | `16:9 9:16 1:1` (+ `auto` on i2v) |
//! | `num_inference_steps` (5B) | `40` | integer 2..=50 |
//! | `guidance_scale` | `3.5` | number 1..=10 (a no-op on `fast-wan`) |
//! | `shift` (5B) | `5` | number 1..=10 |
//! | `interpolator_model` | `"none"` | `none film rife` |
//! | `num_interpolated_frames` | `0` | integer 0..=4; > 0 with `film`/`rife` is refused (no interpolator on this server) |
//! | `enable_prompt_expansion` | `false` | boolean (accepted, no expansion) |
//! | `image_url` (i2v) | required | string |
//! | `seed` | `null` | integer or null |
//! | `enable_safety_checker` | `true` | boolean (accepted, no checker) |
//! | `sync_mode` | `false` | boolean |
//!
//! Resolutions map to the Wan 5B sizes (multiples of 32): 16:9 is `480p`
//! 832x480, `580p` 1024x576, `720p` 1280x704 (the 5B's trained size); 9:16
//! the transpose; 1:1 and `auto` (the image's aspect) keep the short edge
//! 480, 576 or 704.

use fastvideo_protocol::{
    Anchor, ApiError, CanvasSpec, GenerationRequest, Keyframe, Length, MediaRef, ProtocolId, Ratio,
    Snap, Task, TimingSpec,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::{bad, parse_enum, type_name, Fields, PROMPT_MAX_CHARS};

pub const WAN_FRAMES_MIN: i64 = 17;
pub const WAN_FRAMES_MAX: i64 = 161;
pub const WAN_FRAMES_DEFAULT: i64 = 81;
pub const WAN_FPS_MIN: i64 = 4;
pub const WAN_FPS_MAX: i64 = 60;
pub const WAN_FPS_DEFAULT: i64 = 24;
pub const WAN_STEPS_MIN: i64 = 2;
pub const WAN_STEPS_MAX: i64 = 50;
pub const WAN_STEPS_DEFAULT: i64 = 40;
pub const WAN_GUIDANCE: (f64, f64, f64) = (1.0, 10.0, 3.5);
pub const WAN_SHIFT: (f64, f64, f64) = (1.0, 10.0, 5.0);
pub const WAN_INTERPOLATED_MAX: i64 = 4;
pub const INTERPOLATORS: [&str; 3] = ["none", "film", "rife"];

/// Which 5B endpoint.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum WanVariant {
    /// `v2.2-5b/text-to-video` (`wan-max`).
    TextToVideo,
    /// `v2.2-5b/image-to-video` (`wan-max`).
    ImageToVideo,
    /// `v2.2-5b/text-to-video/fast-wan` (`wan-turbo`).
    FastWan,
}

impl WanVariant {
    pub fn resolutions(self) -> &'static [WanResolution] {
        match self {
            WanVariant::FastWan => &WanResolution::ALL,
            _ => &WanResolution::ALL[1..],
        }
    }
    pub fn aspects(self) -> &'static [WanAspect] {
        match self {
            WanVariant::ImageToVideo => &WanAspect::I2V,
            _ => &WanAspect::T2V,
        }
    }
    pub fn default_aspect(self) -> WanAspect {
        match self {
            WanVariant::ImageToVideo => WanAspect::Auto,
            _ => WanAspect::R16x9,
        }
    }
    /// The DMD tier takes no step count, and its guidance is a no-op.
    pub fn distilled(self) -> bool {
        self == WanVariant::FastWan
    }
}

/// `resolution`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum WanResolution {
    #[serde(rename = "480p")]
    P480,
    #[serde(rename = "580p")]
    P580,
    #[serde(rename = "720p")]
    P720,
}

impl WanResolution {
    pub const ALL: [WanResolution; 3] = [WanResolution::P480, WanResolution::P580, WanResolution::P720];
    pub fn as_str(&self) -> &'static str {
        match self {
            WanResolution::P480 => "480p",
            WanResolution::P580 => "580p",
            WanResolution::P720 => "720p",
        }
    }
    /// The Wan 5B short edge (multiple of 32).
    pub fn short_edge(&self) -> u32 {
        match self {
            WanResolution::P480 => 480,
            WanResolution::P580 => 576,
            WanResolution::P720 => 704,
        }
    }
    /// The landscape size of `16:9` (the 5B's trained 1280x704; 1024x576;
    /// FastVideo's 832x480), multiples of 32.
    pub fn landscape(&self) -> (u32, u32) {
        match self {
            WanResolution::P480 => (832, 480),
            WanResolution::P580 => (1024, 576),
            WanResolution::P720 => (1280, 704),
        }
    }
}

/// `aspect_ratio`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum WanAspect {
    #[serde(rename = "auto")]
    Auto,
    #[serde(rename = "16:9")]
    R16x9,
    #[serde(rename = "9:16")]
    R9x16,
    #[serde(rename = "1:1")]
    R1x1,
}

impl WanAspect {
    pub const T2V: [WanAspect; 3] = [WanAspect::R16x9, WanAspect::R9x16, WanAspect::R1x1];
    pub const I2V: [WanAspect; 4] = [WanAspect::Auto, WanAspect::R16x9, WanAspect::R9x16, WanAspect::R1x1];
    pub fn as_str(&self) -> &'static str {
        match self {
            WanAspect::Auto => "auto",
            WanAspect::R16x9 => "16:9",
            WanAspect::R9x16 => "9:16",
            WanAspect::R1x1 => "1:1",
        }
    }
}

/// A validated Wan 5B input.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct WanInput {
    pub variant: WanVariant,
    pub prompt: String,
    pub negative_prompt: Option<String>,
    pub num_frames: u32,
    pub frames_per_second: u32,
    pub resolution: WanResolution,
    pub aspect_ratio: WanAspect,
    /// `None` on `fast-wan`.
    pub num_inference_steps: Option<u32>,
    pub guidance_scale: f64,
    /// `None` on `fast-wan`.
    pub shift: Option<f64>,
    pub interpolator_model: String,
    pub num_interpolated_frames: u32,
    pub enable_prompt_expansion: bool,
    pub image_url: Option<String>,
    pub seed: Option<u64>,
    pub enable_safety_checker: bool,
    pub sync_mode: bool,
}

fn int_in(f: &Fields, k: &str, lo: i64, hi: i64, default: i64) -> Result<u32, ApiError> {
    let v = f.int(k)?.unwrap_or(default);
    if v < lo {
        return Err(bad(k, format!("Input should be greater than or equal to {lo}")));
    }
    if v > hi {
        return Err(bad(k, format!("Input should be less than or equal to {hi}")));
    }
    Ok(v as u32)
}

fn num_in(f: &Fields, k: &str, (lo, hi, default): (f64, f64, f64)) -> Result<f64, ApiError> {
    let v = match f.get(k) {
        None => default,
        Some(Value::Number(n)) => n.as_f64().unwrap_or(f64::NAN),
        Some(v) => return Err(bad(k, format!("Input should be a valid number, got {}", type_name(v)))),
    };
    if !(v >= lo) {
        return Err(bad(k, format!("Input should be greater than or equal to {lo}")));
    }
    if v > hi {
        return Err(bad(k, format!("Input should be less than or equal to {hi}")));
    }
    Ok(v)
}

pub(super) fn parse(f: &Fields, variant: WanVariant) -> Result<WanInput, ApiError> {
    let prompt = super::parse_prompt(f, PROMPT_MAX_CHARS)?;
    let negative_prompt = f.string("negative_prompt")?.filter(|s| !s.trim().is_empty()).map(str::to_owned);
    let num_frames = int_in(f, "num_frames", WAN_FRAMES_MIN, WAN_FRAMES_MAX, WAN_FRAMES_DEFAULT)?;
    let frames_per_second = int_in(f, "frames_per_second", WAN_FPS_MIN, WAN_FPS_MAX, WAN_FPS_DEFAULT)?;
    let resolution = parse_enum(f, "resolution", variant.resolutions(), WanResolution::as_str, WanResolution::P720)?;
    let aspect_ratio = parse_enum(f, "aspect_ratio", variant.aspects(), WanAspect::as_str, variant.default_aspect())?;
    let (num_inference_steps, shift) = if variant.distilled() {
        (None, None)
    } else {
        (
            Some(int_in(f, "num_inference_steps", WAN_STEPS_MIN, WAN_STEPS_MAX, WAN_STEPS_DEFAULT)?),
            Some(num_in(f, "shift", WAN_SHIFT)?),
        )
    };
    let guidance_scale = num_in(f, "guidance_scale", WAN_GUIDANCE)?;
    let interpolator_model = match f.string("interpolator_model")? {
        None => "none".to_owned(),
        Some(s) if INTERPOLATORS.contains(&s) => s.to_owned(),
        Some(_) => {
            let list: Vec<String> = INTERPOLATORS.iter().map(|c| format!("'{c}'")).collect();
            return Err(bad("interpolator_model", format!("Input should be {}", list.join(", "))));
        }
    };
    let num_interpolated_frames = int_in(f, "num_interpolated_frames", 0, WAN_INTERPOLATED_MAX, 0)?;
    if interpolator_model != "none" && num_interpolated_frames > 0 {
        return Err(bad(
            "interpolator_model",
            "frame interpolation (film, rife) is not available on this server; use 'none' or num_interpolated_frames 0",
        ));
    }
    let image_url = if variant == WanVariant::ImageToVideo {
        Some(super::opt_url(f, "image_url")?.ok_or_else(|| bad("image_url", "Field required"))?)
    } else {
        None
    };
    Ok(WanInput {
        variant,
        prompt,
        negative_prompt,
        num_frames,
        frames_per_second,
        resolution,
        aspect_ratio,
        num_inference_steps,
        guidance_scale,
        shift,
        interpolator_model,
        num_interpolated_frames,
        enable_prompt_expansion: f.bool("enable_prompt_expansion", false)?,
        image_url,
        seed: super::parse_seed(f)?,
        enable_safety_checker: f.bool("enable_safety_checker", true)?,
        sync_mode: f.bool("sync_mode", false)?,
    })
}

impl WanInput {
    pub(super) fn normalize(&self, model: &str) -> Result<GenerationRequest, ApiError> {
        let mut r = GenerationRequest::text(ProtocolId::Fal, model, self.prompt.clone());
        r.seed = self.seed;
        r.output.inline_data_uri = self.sync_mode;
        r.note_noop("enable_safety_checker");
        r.note_noop("enable_prompt_expansion");
        r.timing = TimingSpec {
            length: Length::Frames { value: self.num_frames, snap: Snap::AlignUp },
            fps: Some(self.frames_per_second),
        };
        if self.variant.distilled() {
            // DMD runs one conditional pass: no negative prompt, no CFG.
            r.note_noop("guidance_scale");
            if self.negative_prompt.is_some() {
                r.note_noop("negative_prompt");
            }
        } else {
            r.negative_prompt = self.negative_prompt.clone();
            r.sampling.steps = self.num_inference_steps;
            r.sampling.guidance = Some(self.guidance_scale as f32);
            r.sampling.flow_shift = self.shift;
        }
        let short_edge = self.resolution.short_edge();
        let (w, h) = self.resolution.landscape();
        r.canvas = match self.aspect_ratio {
            WanAspect::Auto => CanvasSpec::FollowImage { short_edge },
            WanAspect::R16x9 => CanvasSpec::Exact { width: w, height: h },
            WanAspect::R9x16 => CanvasSpec::Exact { width: h, height: w },
            WanAspect::R1x1 => CanvasSpec::Aspect { ratio: Ratio::R1_1, short_edge },
        };
        if let Some(u) = &self.image_url {
            r.keyframes.push(Keyframe { at: Anchor::First, image: MediaRef::parse(u, "image_url")? });
            r.task = Task::I2V;
        }
        Ok(r)
    }
}
