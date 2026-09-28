//! `lightricks/ltx-2.5/{text,image}-to-video/{fast,pro}` inputs
//! (docs/serve/fal-parity.md §2.1, §2.3). fal's LTX-2.5 partner endpoints
//! have the api.ltx.io contract; the field names are fal's.
//!
//! | field | default | constraints |
//! |---|---|---|
//! | `prompt` | required | 1..=5000 characters |
//! | `duration` | `6` | fast: 6, 8, …, 20 or `"auto"`; pro: 6, 8, 10 or `"auto"` |
//! | `resolution` | `"1080p"` | fast: `720p 1080p 1440p 2160p`; pro: `720p 1080p` |
//! | `aspect_ratio` | t2v `"16:9"`, i2v `"auto"` | `16:9 9:16` (+ `auto` on i2v) |
//! | `fps` | `25` | fast: 24, 25, 48, 50; pro: 24, 25, 50 |
//! | `generate_audio` | `true` | boolean |
//! | `camera_motion` | `null` | the 8 LTX motions; `static` is a no-op, the others `Unsupported(LtxCameraMotion)` |
//! | `image_url` (i2v) | required | string |
//! | `end_image_url` (i2v) | `null` | string: a first-to-last transition (`Task::Keyframes`) |
//! | `seed` | `null` | integer or null |
//! | `sync_mode` | `false` | boolean |
//!
//! Deviations from fal, each a server limit rather than a schema choice:
//!
//! - fal's `duration` default is `"auto"` (the LTX-2.5 duration head). It is
//!   accepted and answers `Unsupported(LtxAutoDuration)` until that head is
//!   loaded, so an omitted `duration` means 6 s here.
//! - The matrix (fast: > 10 s only at 720p/1080p and 24/25 fps) is fal's;
//!   on top of it, the frame count (`duration × fps + 1` on the 8k+1 grid)
//!   must fit the engine's LTX grid, [`LTX_FRAMES_MAX`] (481): 20 s at
//!   24 fps fits, 20 s at 25 fps and 10 s at 50 fps do not.
//! - Image-to-video normalizes to `Task::I2V` / `Task::Keyframes`; the engine
//!   refuses them (`Ltx25I2V`, `LtxKeyframes`) until the LTX I2V port lands.

use fastvideo_protocol::{
    Anchor, ApiError, AudioOut, CanvasSpec, GapId, GenerationRequest, Keyframe, Length, MediaRef,
    ProtocolId, Snap, Task, TimingSpec,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::{bad, parse_enum, type_name, Fields};

/// `prompt` maxLength (the LTX API contract).
pub const LTX_PROMPT_MAX_CHARS: usize = 5000;
/// The omitted `duration` (fal: `"auto"`, see the module docs).
pub const LTX_DEFAULT_DURATION: u32 = 6;
pub const LTX_DEFAULT_FPS: u32 = 25;
pub const LTX_FPS_FAST: [u32; 4] = [24, 25, 48, 50];
pub const LTX_FPS_PRO: [u32; 3] = [24, 25, 50];
/// The engine's LTX frame grid ceiling (`ltx2_caps`: 8k+1, 9..=481).
pub const LTX_FRAMES_MAX: u32 = 481;
/// `camera_motion` values (LTX API).
pub const CAMERA_MOTIONS: [&str; 8] =
    ["dolly_in", "dolly_out", "dolly_left", "dolly_right", "jib_up", "jib_down", "static", "focus_shift"];

/// fast (`ltx-turbo`) or pro (`ltx-pro`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LtxClass {
    Fast,
    Pro,
}

impl LtxClass {
    pub fn resolutions(self) -> &'static [LtxResolution] {
        match self {
            LtxClass::Fast => &LtxResolution::ALL,
            LtxClass::Pro => &LtxResolution::ALL[..2],
        }
    }
    pub fn fps(self) -> &'static [u32] {
        match self {
            LtxClass::Fast => &LTX_FPS_FAST,
            LtxClass::Pro => &LTX_FPS_PRO,
        }
    }
    /// Every duration the schema lists (the matrix narrows it per fps and
    /// resolution, see [`max_duration`]).
    pub fn durations(self) -> Vec<u32> {
        let max = match self {
            LtxClass::Fast => 20,
            LtxClass::Pro => 10,
        };
        (6..=max).step_by(2).collect()
    }
}

/// `resolution`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum LtxResolution {
    #[serde(rename = "720p")]
    P720,
    #[serde(rename = "1080p")]
    P1080,
    #[serde(rename = "1440p")]
    P1440,
    #[serde(rename = "2160p")]
    P2160,
}

impl LtxResolution {
    pub const ALL: [LtxResolution; 4] =
        [LtxResolution::P720, LtxResolution::P1080, LtxResolution::P1440, LtxResolution::P2160];
    pub fn as_str(&self) -> &'static str {
        match self {
            LtxResolution::P720 => "720p",
            LtxResolution::P1080 => "1080p",
            LtxResolution::P1440 => "1440p",
            LtxResolution::P2160 => "2160p",
        }
    }
    /// Landscape `(width, height)` (the LTX API sizes).
    pub fn landscape(&self) -> (u32, u32) {
        match self {
            LtxResolution::P720 => (1280, 720),
            LtxResolution::P1080 => (1920, 1080),
            LtxResolution::P1440 => (2560, 1440),
            LtxResolution::P2160 => (3840, 2160),
        }
    }
}

/// `aspect_ratio` (`auto` on image-to-video only: the image's aspect).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum LtxAspect {
    #[serde(rename = "auto")]
    Auto,
    #[serde(rename = "16:9")]
    R16x9,
    #[serde(rename = "9:16")]
    R9x16,
}

impl LtxAspect {
    pub const T2V: [LtxAspect; 2] = [LtxAspect::R16x9, LtxAspect::R9x16];
    pub const I2V: [LtxAspect; 3] = [LtxAspect::Auto, LtxAspect::R16x9, LtxAspect::R9x16];
    pub fn as_str(&self) -> &'static str {
        match self {
            LtxAspect::Auto => "auto",
            LtxAspect::R16x9 => "16:9",
            LtxAspect::R9x16 => "9:16",
        }
    }
}

/// fal's matrix: fast allows up to 20 s at 720p/1080p and 24/25 fps, 10 s
/// otherwise; pro 10 s.
pub fn max_duration(class: LtxClass, res: LtxResolution, fps: u32) -> u32 {
    let long = class == LtxClass::Fast
        && matches!(res, LtxResolution::P720 | LtxResolution::P1080)
        && matches!(fps, 24 | 25);
    if long {
        20
    } else {
        10
    }
}

/// Frames the engine generates for `duration` s at `fps` (8k+1, rounded up).
pub fn frames_for(duration: u32, fps: u32) -> u32 {
    let raw = duration * fps;
    raw.div_ceil(8) * 8 + 1
}

/// A validated LTX input.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LtxInput {
    pub class: LtxClass,
    pub image_to_video: bool,
    pub prompt: String,
    /// `None` = `"auto"`.
    pub duration: Option<u32>,
    pub resolution: LtxResolution,
    pub aspect_ratio: LtxAspect,
    pub fps: u32,
    pub generate_audio: bool,
    pub camera_motion: Option<String>,
    pub image_url: Option<String>,
    pub end_image_url: Option<String>,
    pub seed: Option<u64>,
    pub sync_mode: bool,
}

fn parse_duration(f: &Fields, class: LtxClass) -> Result<Option<u32>, ApiError> {
    let allowed = class.durations();
    let list = || {
        let mut v: Vec<String> = allowed.iter().map(u32::to_string).collect();
        v.push("'auto'".into());
        v.join(", ")
    };
    let d = match f.get("duration") {
        None => return Ok(Some(LTX_DEFAULT_DURATION)),
        Some(Value::String(s)) if s == "auto" => return Ok(None),
        Some(Value::String(s)) => s.trim().parse::<i64>().ok(),
        Some(Value::Number(_)) => f.int("duration")?,
        Some(v) => return Err(bad("duration", format!("Input should be an integer or 'auto', got {}", type_name(v)))),
    };
    match d {
        Some(d) if d > 0 && allowed.contains(&(d as u32)) => Ok(Some(d as u32)),
        _ => Err(bad("duration", format!("Input should be {}", list()))),
    }
}

fn parse_fps(f: &Fields, class: LtxClass) -> Result<u32, ApiError> {
    let allowed = class.fps();
    let fps = match f.get("fps") {
        None => return Ok(LTX_DEFAULT_FPS),
        Some(Value::String(s)) => s.trim().parse::<i64>().ok(),
        Some(_) => f.int("fps")?,
    };
    match fps {
        Some(v) if v > 0 && allowed.contains(&(v as u32)) => Ok(v as u32),
        _ => {
            let list: Vec<String> = allowed.iter().map(u32::to_string).collect();
            Err(bad("fps", format!("Input should be {}", list.join(", "))))
        }
    }
}

pub(super) fn parse(f: &Fields, class: LtxClass, image_to_video: bool) -> Result<LtxInput, ApiError> {
    let prompt = super::parse_prompt(f, LTX_PROMPT_MAX_CHARS)?;
    let duration = parse_duration(f, class)?;
    let resolution = parse_enum(f, "resolution", class.resolutions(), LtxResolution::as_str, LtxResolution::P1080)?;
    let fps = parse_fps(f, class)?;
    if let Some(d) = duration {
        let max = max_duration(class, resolution, fps);
        if d > max {
            return Err(bad(
                "duration",
                format!("duration {d} is not available at {} and {fps} fps; the maximum is {max}", resolution.as_str()),
            ));
        }
        let frames = frames_for(d, fps);
        if frames > LTX_FRAMES_MAX {
            let longest = (6..=d).step_by(2).filter(|&x| frames_for(x, fps) <= LTX_FRAMES_MAX).last().unwrap_or(6);
            return Err(bad(
                "duration",
                format!(
                    "duration {d} at {fps} fps is {frames} frames, more than this server's LTX limit of {LTX_FRAMES_MAX}; the longest at {fps} fps is {longest}"
                ),
            ));
        }
    }
    let (aspects, default_aspect): (&[LtxAspect], _) =
        if image_to_video { (&LtxAspect::I2V, LtxAspect::Auto) } else { (&LtxAspect::T2V, LtxAspect::R16x9) };
    let aspect_ratio = parse_enum(f, "aspect_ratio", aspects, LtxAspect::as_str, default_aspect)?;
    let camera_motion = match f.string("camera_motion")? {
        None => None,
        Some(c) if CAMERA_MOTIONS.contains(&c) => Some(c.to_owned()),
        Some(_) => {
            let list: Vec<String> = CAMERA_MOTIONS.iter().map(|c| format!("'{c}'")).collect();
            return Err(bad("camera_motion", format!("Input should be {}", list.join(", "))));
        }
    };
    let (image_url, end_image_url) = if image_to_video {
        let first = super::opt_url(f, "image_url")?.ok_or_else(|| bad("image_url", "Field required"))?;
        (Some(first), super::opt_url(f, "end_image_url")?)
    } else {
        (None, None)
    };
    Ok(LtxInput {
        class,
        image_to_video,
        prompt,
        duration,
        resolution,
        aspect_ratio,
        fps,
        generate_audio: f.bool("generate_audio", true)?,
        camera_motion,
        image_url,
        end_image_url,
        seed: super::parse_seed(f)?,
        sync_mode: f.bool("sync_mode", false)?,
    })
}

impl LtxInput {
    pub(super) fn normalize(&self, model: &str) -> Result<GenerationRequest, ApiError> {
        let mut r = GenerationRequest::text(ProtocolId::Fal, model, self.prompt.clone());
        r.seed = self.seed;
        r.output.inline_data_uri = self.sync_mode;
        r.timing = TimingSpec {
            length: match self.duration {
                Some(d) => Length::Seconds { value: d as f64, snap: Snap::AlignUp },
                None => Length::Auto,
            },
            fps: Some(self.fps),
        };
        if !self.generate_audio {
            r.audio_out = AudioOut::Silent;
        }
        match self.camera_motion.as_deref() {
            None => {}
            Some("static") => r.note_noop("camera_motion"),
            Some(_) => return Err(ApiError::unsupported(GapId::LtxCameraMotion).with_param("camera_motion")),
        }
        let (w, h) = self.resolution.landscape();
        r.canvas = match self.aspect_ratio {
            LtxAspect::R16x9 => CanvasSpec::Exact { width: w, height: h },
            LtxAspect::R9x16 => CanvasSpec::Exact { width: h, height: w },
            LtxAspect::Auto => CanvasSpec::FollowImage { short_edge: h },
        };
        if let Some(u) = &self.image_url {
            r.keyframes.push(Keyframe { at: Anchor::First, image: MediaRef::parse(u, "image_url")? });
            r.task = Task::I2V;
        }
        if let Some(u) = &self.end_image_url {
            r.keyframes.push(Keyframe { at: Anchor::Last, image: MediaRef::parse(u, "end_image_url")? });
            r.task = Task::Keyframes;
        }
        Ok(r)
    }
}
