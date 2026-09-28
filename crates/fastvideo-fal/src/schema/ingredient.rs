//! `fal-ai/ltx-2.3-quality/ingredient` inputs (docs/serve/fal-parity.md §2.1,
//! §2.3): reference-sheet video with the LTX Ingredients IC-LoRA. fal runs it
//! on LTX-2.3; this server runs the LTX-2.5 build of the same LoRA
//! (`Lightricks/LTX-2.5-22b-IC-LoRA-Ingredients`, docs/ports/ltx-ref2v.md)
//! on the `ltx-pro` tier, whose reference-to-video companion
//! (`ltx25-ref2v`) takes it through `route_task`.
//!
//! | field | default | constraints |
//! |---|---|---|
//! | `prompt` | required | 1..=5000 characters; "Reference sheet: … Generated video: …" |
//! | `image_url` | required | string: the reference sheet (character, prop and location panels) |
//! | `ingredient_strength` | `1` | number 0..=2: the IC-LoRA's strength at stage 1 |
//! | `reference_strength` | `1` | number 0..=2; this server takes 0..=1 (the reference tokens' denoise mask is `1 − s`), above 1 is refused by `negotiate` |
//! | `num_frames` | `121` | integer 9..=481 (8k+1, rounded up); this server generates at most 241 |
//! | `frames_per_second` | `24` | integer 1..=60; this server takes the LTX rates 24, 25, 48, 50 |
//! | `generate_audio` | `true` | boolean |
//! | `negative_prompt` | `""` | string; a no-op (the distilled model runs one unguided pass) |
//! | `seed` | `null` | integer or null |
//! | `sync_mode` | `false` | boolean |
//!
//! The canvas is fal's default, 1536x896: a 768x448 first stage (the LoRA's
//! trained bucket) and the 2x refine. The sheet is looped into a static clip
//! of the output's length (the model card's reference input) by the engine.

use fastvideo_protocol::{
    ApiError, AudioOut, CanvasSpec, GenerationRequest, Length, MediaKind, MediaRef, ProtocolId,
    Reference, Snap, Task, TimingSpec,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::ltx::LTX_PROMPT_MAX_CHARS;
use super::{bad, type_name, Fields};

pub const INGREDIENT_FRAMES_MIN: i64 = 9;
pub const INGREDIENT_FRAMES_MAX: i64 = 481;
pub const INGREDIENT_FRAMES_DEFAULT: i64 = 121;
pub const INGREDIENT_FPS_MIN: i64 = 1;
pub const INGREDIENT_FPS_MAX: i64 = 60;
pub const INGREDIENT_FPS_DEFAULT: i64 = 24;
/// `(min, max, default)` of `ingredient_strength` and `reference_strength`.
pub const INGREDIENT_STRENGTH: (f64, f64, f64) = (0.0, 2.0, 1.0);
/// fal's default output size (768x448 first stage + 2x refine).
pub const INGREDIENT_CANVAS: (u32, u32) = (1536, 896);

/// A validated `ingredient` input.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct IngredientInput {
    pub prompt: String,
    pub image_url: String,
    pub ingredient_strength: f64,
    pub reference_strength: f64,
    pub num_frames: u32,
    pub frames_per_second: u32,
    pub generate_audio: bool,
    pub negative_prompt: Option<String>,
    pub seed: Option<u64>,
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
    if v.is_nan() || v < lo {
        return Err(bad(k, format!("Input should be greater than or equal to {lo}")));
    }
    if v > hi {
        return Err(bad(k, format!("Input should be less than or equal to {hi}")));
    }
    Ok(v)
}

pub(super) fn parse(f: &Fields) -> Result<IngredientInput, ApiError> {
    let prompt = super::parse_prompt(f, LTX_PROMPT_MAX_CHARS)?;
    let image_url = super::opt_url(f, "image_url")?.ok_or_else(|| bad("image_url", "Field required"))?;
    Ok(IngredientInput {
        prompt,
        image_url,
        ingredient_strength: num_in(f, "ingredient_strength", INGREDIENT_STRENGTH)?,
        reference_strength: num_in(f, "reference_strength", INGREDIENT_STRENGTH)?,
        num_frames: int_in(f, "num_frames", INGREDIENT_FRAMES_MIN, INGREDIENT_FRAMES_MAX, INGREDIENT_FRAMES_DEFAULT)?,
        frames_per_second: int_in(f, "frames_per_second", INGREDIENT_FPS_MIN, INGREDIENT_FPS_MAX, INGREDIENT_FPS_DEFAULT)?,
        generate_audio: f.bool("generate_audio", true)?,
        negative_prompt: f.string("negative_prompt")?.filter(|s| !s.trim().is_empty()).map(str::to_owned),
        seed: super::parse_seed(f)?,
        sync_mode: f.bool("sync_mode", false)?,
    })
}

impl IngredientInput {
    pub(super) fn normalize(&self, model: &str) -> Result<GenerationRequest, ApiError> {
        let mut r = GenerationRequest::text(ProtocolId::Fal, model, self.prompt.clone());
        r.task = Task::Ref2V;
        r.seed = self.seed;
        r.output.inline_data_uri = self.sync_mode;
        r.timing = TimingSpec {
            length: Length::Frames { value: self.num_frames, snap: Snap::AlignUp },
            fps: Some(self.frames_per_second),
        };
        if !self.generate_audio {
            r.audio_out = AudioOut::Silent;
        }
        if self.negative_prompt.is_some() {
            // The distilled two-stage is unguided: nothing reads it.
            r.note_noop("negative_prompt");
        }
        let (width, height) = INGREDIENT_CANVAS;
        r.canvas = CanvasSpec::Exact { width, height };
        r.references.push(Reference {
            kind: MediaKind::Image,
            media: MediaRef::parse(&self.image_url, "image_url")?,
        });
        r.sampling.reference_strength = Some(self.reference_strength as f32);
        r.sampling.reference_lora_strength = Some(self.ingredient_strength as f32);
        Ok(r)
    }
}
