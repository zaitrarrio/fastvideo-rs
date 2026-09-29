//! `lightricks/ltx-2.5/audio-to-video/{fast,pro}` inputs (fal OpenAPI
//! `Ltx25AudioToVideo{Fast,Pro}Input`, docs/serve/fal-parity.md §2.1): an
//! audio track drives the video; an optional image is its first frame.
//!
//! | field | default | constraints |
//! |---|---|---|
//! | `audio_url` | required | string: 2 to 20 s (pro: at most 10 s); the video follows its length |
//! | `image_url` | `null` | string or null: the first frame |
//! | `prompt` | `null` | 1..=5000 characters or null; required without `image_url` |
//! | `aspect_ratio` | `"auto"` | `auto 16:9 9:16`; `auto`: 9:16 for a portrait image, else 16:9 |
//! | `guidance_scale` | `null` | number 1..=50 or null (fal: 5 for text, 9 with an image) |
//!
//! This server adds fal's usual `seed` and `sync_mode` (the fal schema has
//! neither). The output is 1080p (1920x1080 or 1080x1920; generated at 1088
//! and cropped), at 24 fps, with the input audio as its soundtrack.
//!
//! Deviations, each documented in fal-parity:
//!
//! - `guidance_scale`: on `pro` it is the video CFG scale of the guided
//!   pipeline (`A2VidPipelineTwoStage` on the LTX-2.5 dev transformer;
//!   unset: the reference default 3, not fal's 5 / 9). On `fast` (the
//!   distilled model, one unguided pass per step) it is validated and then
//!   a no-op.
//! - The frame count is the longest 8k+1 clip whose length fits in the audio
//!   (`negotiate`); fal's exact rule is not published.

use fastvideo_protocol::{
    Anchor, ApiError, AudioInput, AudioRole, CanvasSpec, GenerationRequest, Keyframe, MediaRef,
    ProtocolId, Task,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::ltx::{LtxAspect, LtxClass, LTX_PROMPT_MAX_CHARS};
use super::{bad, parse_enum, type_name, Fields};

/// `guidance_scale` bounds.
pub const A2V_GUIDANCE: (f64, f64) = (1.0, 50.0);
/// The pro endpoints' audio ceiling ("pro models support a maximum of 10
/// seconds"); fast takes the protocol's 20 s.
pub const A2V_PRO_MAX_S: u32 = 10;
/// Short edge of the output (fal: 1080p).
pub const A2V_SHORT_EDGE: u32 = 1080;
/// Output frame rate (the LTX-2.5 default).
pub const A2V_FPS: u32 = 24;
/// `aspect_ratio` values, in schema order.
pub const A2V_ASPECTS: [LtxAspect; 3] = [LtxAspect::Auto, LtxAspect::R16x9, LtxAspect::R9x16];

/// A validated audio-to-video input.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LtxA2vInput {
    pub class: LtxClass,
    pub audio_url: String,
    pub image_url: Option<String>,
    pub prompt: Option<String>,
    pub aspect_ratio: LtxAspect,
    pub guidance_scale: Option<f64>,
    pub seed: Option<u64>,
    pub sync_mode: bool,
}

pub(super) fn parse(f: &Fields, class: LtxClass) -> Result<LtxA2vInput, ApiError> {
    let audio_url = super::opt_url(f, "audio_url")?.ok_or_else(|| bad("audio_url", "Field required"))?;
    let image_url = super::opt_url(f, "image_url")?;
    let prompt = match f.get("prompt") {
        None => None,
        Some(_) => Some(super::parse_prompt(f, LTX_PROMPT_MAX_CHARS)?),
    };
    if prompt.is_none() && image_url.is_none() {
        return Err(bad("prompt", "prompt is required when image_url is not provided"));
    }
    let guidance_scale = match f.get("guidance_scale") {
        None => None,
        Some(Value::Number(n)) => {
            let g = n.as_f64().unwrap_or(f64::NAN);
            let (lo, hi) = A2V_GUIDANCE;
            if g.is_nan() || g < lo {
                return Err(bad("guidance_scale", format!("Input should be greater than or equal to {lo}")));
            }
            if g > hi {
                return Err(bad("guidance_scale", format!("Input should be less than or equal to {hi}")));
            }
            Some(g)
        }
        Some(v) => return Err(bad("guidance_scale", format!("Input should be a valid number, got {}", type_name(v)))),
    };
    Ok(LtxA2vInput {
        class,
        audio_url,
        image_url,
        prompt,
        aspect_ratio: parse_enum(f, "aspect_ratio", &A2V_ASPECTS, LtxAspect::as_str, LtxAspect::Auto)?,
        guidance_scale,
        seed: super::parse_seed(f)?,
        sync_mode: f.bool("sync_mode", false)?,
    })
}

impl LtxA2vInput {
    pub(super) fn normalize(&self, model: &str) -> Result<GenerationRequest, ApiError> {
        let mut r = GenerationRequest::text(ProtocolId::Fal, model, self.prompt.clone().unwrap_or_default());
        r.task = Task::A2V;
        r.seed = self.seed;
        r.output.inline_data_uri = self.sync_mode;
        // The length follows the audio (`Length::ModelDefault` on A2V).
        r.timing.fps = Some(A2V_FPS);
        r.audio_in = Some(AudioInput {
            media: MediaRef::parse(&self.audio_url, "audio_url")?,
            role: AudioRole::Drive,
            max_s: (self.class == LtxClass::Pro).then_some(A2V_PRO_MAX_S),
        });
        if let Some(u) = &self.image_url {
            r.keyframes.push(Keyframe { at: Anchor::First, image: MediaRef::parse(u, "image_url")? });
        }
        let (w, h) = (A2V_SHORT_EDGE * 16 / 9, A2V_SHORT_EDGE);
        r.canvas = match self.aspect_ratio {
            // "determined automatically based on the input image, or defaults
            // to 16:9": 9:16 for a portrait image (the LTX API rule).
            LtxAspect::Auto => CanvasSpec::Oriented { width: w, height: h },
            LtxAspect::R16x9 => CanvasSpec::Exact { width: w, height: h },
            LtxAspect::R9x16 => CanvasSpec::Exact { width: h, height: w },
        };
        match (self.class, self.guidance_scale) {
            // `pro`: the guided dev pipeline's video CFG scale.
            (LtxClass::Pro, g) => r.sampling.guidance = g.map(|g| g as f32),
            // `fast`: the distilled two-stage is unguided, nothing reads it.
            (LtxClass::Fast, Some(_)) => r.note_noop("guidance_scale"),
            (LtxClass::Fast, None) => {}
        }
        Ok(r)
    }
}
