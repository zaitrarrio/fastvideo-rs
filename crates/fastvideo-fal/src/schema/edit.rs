//! `fal-ai/ltx-2.3/retake-video` and `fal-ai/ltx-2.3/extend-video` inputs
//! (fal OpenAPI `Ltx23RetakeVideoInput` / `Ltx23ExtendVideoInput`, read
//! 2026-09-29; docs/serve/fal-parity.md §2.1). fal runs them on LTX-2.3;
//! this server runs the same edits on its LTX-2.5 distilled weights, on the
//! `ltx-pro` tier (docs/oracle.md "LTX-2.5 retake and extend").
//!
//! Retake:
//!
//! | field | default | constraints |
//! |---|---|---|
//! | `video_url` | required | string |
//! | `prompt` | required | 1..=5000 characters |
//! | `start_time` | `0` | number 0..=20 |
//! | `duration` | `5` | number 2..=20 (clamped to the video) |
//! | `retake_mode` | `"replace_audio_and_video"` | `replace_audio replace_video replace_audio_and_video` |
//!
//! Extend:
//!
//! | field | default | constraints |
//! |---|---|---|
//! | `video_url` | required | string |
//! | `prompt` | `null` | ≤ 5000 characters or null |
//! | `duration` | `5` | number 2..=20 |
//! | `mode` | `"end"` | `start end` |
//! | `context` | `null` | number 1..=20 or null: "defaults to maximize available context within the 505 frame limit" |
//!
//! This server adds fal's usual `seed` and `sync_mode`. The output keeps
//! the source's frame rate and size (snapped down to multiples of 32, at most
//! 1920x1088 worth of pixels), and an extension keeps the whole source.

use fastvideo_protocol::{
    ApiError, EditOp, ExtendAt, GenerationRequest, MediaRef, ProtocolId, RetakeMode, Task, VideoEdit,
    EDIT_DURATION_MAX_S, EDIT_DURATION_MIN_S, EXTEND_CONTEXT_MAX_S, EXTEND_CONTEXT_MIN_S,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::ltx::LTX_PROMPT_MAX_CHARS;
use super::{bad, type_name, Fields};

/// `start_time` bounds.
pub const RETAKE_START: (f64, f64) = (0.0, 20.0);
/// `duration` default (both endpoints).
pub const EDIT_DURATION_DEFAULT: f64 = 5.0;
/// `retake_mode` values, in schema order.
pub const RETAKE_MODES: [RetakeMode; 3] = RetakeMode::ALL;
/// `mode` values, in schema order.
pub const EXTEND_MODES: [ExtendAt; 2] = [ExtendAt::Start, ExtendAt::End];

/// A validated retake or extend input.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LtxEditInput {
    pub video_url: String,
    pub prompt: Option<String>,
    pub op: EditOp,
    pub seed: Option<u64>,
    pub sync_mode: bool,
}

/// A number within `[lo, hi]` (`None` when absent).
fn number(f: &Fields, k: &str, lo: f64, hi: f64) -> Result<Option<f64>, ApiError> {
    let v = match f.get(k) {
        None => return Ok(None),
        Some(Value::Number(n)) => n.as_f64().unwrap_or(f64::NAN),
        Some(v) => return Err(bad(k, format!("Input should be a valid number, got {}", type_name(v)))),
    };
    if v.is_nan() || v < lo {
        return Err(bad(k, format!("Input should be greater than or equal to {lo}")));
    }
    if v > hi {
        return Err(bad(k, format!("Input should be less than or equal to {hi}")));
    }
    Ok(Some(v))
}

fn video_url(f: &Fields) -> Result<String, ApiError> {
    super::opt_url(f, "video_url")?.ok_or_else(|| bad("video_url", "Field required"))
}

fn prompt(f: &Fields, required: bool) -> Result<Option<String>, ApiError> {
    if f.get("prompt").is_none() && !required {
        return Ok(None);
    }
    super::parse_prompt(f, LTX_PROMPT_MAX_CHARS).map(Some)
}

pub(super) fn parse_retake(f: &Fields) -> Result<LtxEditInput, ApiError> {
    let video_url = video_url(f)?;
    let prompt = prompt(f, true)?;
    let start_s = number(f, "start_time", RETAKE_START.0, RETAKE_START.1)?.unwrap_or(0.0);
    let duration_s = number(f, "duration", EDIT_DURATION_MIN_S, EDIT_DURATION_MAX_S)?.unwrap_or(EDIT_DURATION_DEFAULT);
    let mode = super::parse_enum(f, "retake_mode", &RETAKE_MODES, RetakeMode::as_str, RetakeMode::default())?;
    Ok(LtxEditInput {
        video_url,
        prompt,
        op: EditOp::Retake { start_s, duration_s, mode },
        seed: super::parse_seed(f)?,
        sync_mode: f.bool("sync_mode", false)?,
    })
}

pub(super) fn parse_extend(f: &Fields) -> Result<LtxEditInput, ApiError> {
    let video_url = video_url(f)?;
    let prompt = prompt(f, false)?;
    let duration_s = number(f, "duration", EDIT_DURATION_MIN_S, EDIT_DURATION_MAX_S)?.unwrap_or(EDIT_DURATION_DEFAULT);
    let at = super::parse_enum(f, "mode", &EXTEND_MODES, ExtendAt::as_str, ExtendAt::End)?;
    let context_s = number(f, "context", EXTEND_CONTEXT_MIN_S, EXTEND_CONTEXT_MAX_S)?;
    Ok(LtxEditInput {
        video_url,
        prompt,
        op: EditOp::Extend { duration_s, at, context_s },
        seed: super::parse_seed(f)?,
        sync_mode: f.bool("sync_mode", false)?,
    })
}

impl LtxEditInput {
    pub fn is_retake(&self) -> bool {
        matches!(self.op, EditOp::Retake { .. })
    }

    pub(super) fn normalize(&self, model: &str) -> Result<GenerationRequest, ApiError> {
        let mut r = GenerationRequest::text(ProtocolId::Fal, model, self.prompt.clone().unwrap_or_default());
        r.task = if self.is_retake() { Task::Retake } else { Task::Extend };
        r.seed = self.seed;
        r.output.inline_data_uri = self.sync_mode;
        r.edit = Some(VideoEdit { video: MediaRef::parse(&self.video_url, "video_url")?, op: self.op });
        Ok(r)
    }
}
