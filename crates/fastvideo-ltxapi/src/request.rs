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
//! | `enhance_prompt` (ours, boolean) | `prompt_expansion`: the server's upsampler rewrites the prompt when it has one (absent: only with `default_on`; docs/serve/prompt-upsampling.md) |
//!
//! `AudioToVideoRequest` (ltx §2.5, [`normalize_a2v`]): `audio_uri`
//! (required) drives an `A2V` job whose length follows the audio;
//! `image_uri` / `last_frame_uri` pin the first / last frame; `prompt` is
//! required only without an image (may be empty with one); `model` defaults
//! to `ltx-2-3-pro` (served by our max tier; `ltx-2-3-fast` has no A2V);
//! `resolution` defaults to 1920x1080, or 1080x1920 for a portrait image; the
//! audio may be at most 10 s on `ltx-2-5-pro` and at 1440p/4K, else 20 s.
//!
//! `EditVideoRequest` (`/retake`, ltx §2.6, [`normalize_retake`]) and
//! `ExtendVideoRequest` (`/extend`, ltx §2.7, [`normalize_extend`]):
//! `video_uri` (required) is edited on LTX-2.5; `model` defaults to
//! `ltx-2-3-pro` (served by our max tier); the output keeps the source's
//! size (snapped down to multiples of 32; a `resolution` can only ask for
//! less, never an upscale) and frame rate. Retake: `start_time` (≥ 0),
//! `duration` (≥ 2, clamped to the video), `mode`
//! (`replace_audio_and_video` default, `replace_video`, `replace_audio`).
//! Extend: `duration` (2 to 20), `mode` (`end` default, `start`), `context`
//! (1 to 20 s, default as much as fits); context plus extension are at most
//! 505 frames (the LTX API's limit).
//!
//! The bodies are typed ([`crate::body`]): each endpoint reads its JSON
//! into its OpenAPI type, field by field in the order the errors are
//! reported, then builds the request from the typed value. Every limit is
//! a surface rule of `fastvideo_protocol::params::rules::ltx`.
//!
//! Media URIs must be `https://…`, `data:…;base64,…` or `ltx://uploads/<token>`
//! (ltx §2.0). `image_uri` / `last_frame_uri` on text-to-video are refused
//! rather than ignored. Other unknown fields are ignored (the OAS does not
//! forbid additional properties).

use std::sync::Arc;

use fastvideo_protocol::params::rules::ltx as rules;
use fastvideo_protocol::{
    Anchor, ApiError, AudioInput, AudioOut, AudioRole, CanvasSpec, EditOp, ExpansionMode, ExtendAt,
    GapId, GenerationRequest, HttpReply, Job, Keyframe, Length, MediaRef, NormalizeCtx,
    PromptExpansion, RetakeMode, Snap, SubmitEndpoint, Task, TimingSpec, VideoEdit, ViewCtx,
};
use serde_json::{Map, Value};

use crate::body::{
    AudioToVideoRequest, EditVideoRequest, ExtendVideoRequest, Fields, ImageToVideoRequest,
    TextToVideoRequest,
};
use crate::error::Api;
use crate::models::{self, LtxModel, LtxModels, ResTier, Resolution};

/// `camera_motion` values (OAS enum, ltx §2.0; `rules::ltx::CAMERA_MOTIONS`).
pub use fastvideo_protocol::params::rules::ltx::CAMERA_MOTIONS;
/// Longest prompt (OAS `maxLength`; `rules::ltx::PROMPT_MAX_CHARS`).
pub use fastvideo_protocol::params::rules::ltx::PROMPT_MAX_CHARS;
/// Retake `resolution` values (OAS enum).
pub use fastvideo_protocol::params::rules::ltx::RETAKE_RESOLUTIONS;
/// Input audio ceilings (OAS `audio_uri`): 20 s at 720p/1080p, 10 s at
/// 1440p/4K, and 10 s at every size on `ltx-2-5-pro`.
pub use fastvideo_protocol::params::rules::ltx::{A2V_MAX_S_LONG, A2V_MAX_S_SHORT};

/// The generation endpoints we serve.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Endpoint {
    TextToVideo,
    ImageToVideo,
    AudioToVideo,
    Retake,
    Extend,
}

impl Endpoint {
    pub const ALL: [Endpoint; 5] = [
        Endpoint::TextToVideo,
        Endpoint::ImageToVideo,
        Endpoint::AudioToVideo,
        Endpoint::Retake,
        Endpoint::Extend,
    ];

    /// The path segment (`text-to-video`).
    pub fn segment(&self) -> &'static str {
        match self {
            Endpoint::TextToVideo => "text-to-video",
            Endpoint::ImageToVideo => "image-to-video",
            Endpoint::AudioToVideo => "audio-to-video",
            Endpoint::Retake => "retake",
            Endpoint::Extend => "extend",
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
            Task::A2V => Some(Endpoint::AudioToVideo),
            Task::Retake => Some(Endpoint::Retake),
            Task::Extend => Some(Endpoint::Extend),
            _ => None,
        }
    }
}

/// One submit endpoint on one surface. `Body` is raw JSON so validation
/// messages follow the LTX wording; [`normalize`] reads it into the
/// endpoint's typed body ([`crate::body`]).
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

fn check_prompt(prompt: &str) -> Result<(), ApiError> {
    if prompt.chars().count() > PROMPT_MAX_CHARS {
        return Err(invalid(
            "prompt",
            format!("prompt must be at most {PROMPT_MAX_CHARS} characters"),
        ));
    }
    Ok(())
}

/// `camera_motion`: a listed value is an engine gap; another is refused.
fn camera_motion(f: &Fields) -> Result<Option<String>, ApiError> {
    if let Some(cm) = f.opt_str("camera_motion")? {
        if !CAMERA_MOTIONS.contains(&cm) {
            return Err(invalid(
                "camera_motion",
                format!(
                    "camera_motion must be one of: {}",
                    CAMERA_MOTIONS.join(", ")
                ),
            ));
        }
        return Err(ApiError::unsupported(GapId::LtxCameraMotion).with_param("camera_motion"));
    }
    Ok(None)
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
    match endpoint {
        Endpoint::AudioToVideo => normalize_a2v(api, models, o),
        Endpoint::Retake => normalize_retake(api, models, o),
        Endpoint::Extend => normalize_extend(api, models, o),
        Endpoint::TextToVideo => {
            let (b, m) = read_t2v(o, models, true)?;
            t2v_request(api, &b, m, Task::T2V, Vec::new())
        }
        Endpoint::ImageToVideo => {
            let (b, m) = read_i2v(o, models)?;
            i2v_request(api, &b, m)
        }
    }
}

/// The fields text- and image-to-video share, in the LTX API's order:
/// `prompt`, `model`, `resolution`, `duration`, `fps` (and the matrix),
/// `generate_audio`, `camera_motion`. `t2v`: the image fields are refused.
fn read_t2v<'m>(
    o: &Map<String, Value>,
    models: &'m LtxModels,
    t2v: bool,
) -> Result<(TextToVideoRequest, &'m LtxModel), ApiError> {
    let f = Fields(o);
    let prompt = f.req_str("prompt")?;
    check_prompt(prompt)?;
    let model_id = f.req_str("model")?;
    let model = models.lookup(model_id)?;
    let resolution = f.req_str("resolution")?;
    let res = Resolution::parse(resolution)?;

    // `duration` must be present; `null` asks for automatic duration.
    let duration = match f.raw("duration") {
        None => return Err(invalid("duration", "duration is required")),
        Some(Value::Null) => None,
        Some(v) => Some(
            crate::body::as_int(v)
                .and_then(|n| u32::try_from(n).ok())
                .ok_or_else(|| {
                    invalid("duration", "duration must be an integer number of seconds")
                })?,
        ),
    };
    let fps = f.int("fps")?;
    models::check_fps(fps.unwrap_or(rules::API_DEFAULT_FPS))?;
    if let Some(d) = duration {
        models::check_duration(
            model_id,
            model,
            res,
            fps.unwrap_or(rules::API_DEFAULT_FPS),
            d,
        )?;
    } else if !model.auto_duration {
        return Err(invalid(
            "duration",
            format!("duration must be an integer for {model_id}; automatic duration (null) is not available for this model"),
        ));
    }
    let generate_audio = f.bool("generate_audio")?;
    let camera_motion = camera_motion(&f)?;
    if t2v {
        for k in ["image_uri", "last_frame_uri"] {
            if f.get(k).is_some() {
                return Err(invalid(
                    k,
                    format!("{k} is not accepted by text-to-video; use image-to-video"),
                ));
            }
        }
    }
    Ok((
        TextToVideoRequest {
            prompt: prompt.to_owned(),
            model: model_id.to_owned(),
            duration,
            resolution: resolution.to_owned(),
            fps,
            generate_audio,
            camera_motion,
            enhance_prompt: f.lenient_bool("enhance_prompt"),
        },
        model,
    ))
}

/// `ImageToVideoRequest`: the shared fields, then `image_uri` and
/// `last_frame_uri`.
fn read_i2v<'m>(
    o: &Map<String, Value>,
    models: &'m LtxModels,
) -> Result<(ImageToVideoRequest, &'m LtxModel), ApiError> {
    let (base, model) = read_t2v(o, models, false)?;
    let f = Fields(o);
    let image_uri = f.req_str("image_uri")?;
    media_uri(image_uri, "image_uri")?;
    let last_frame_uri = f.opt_str("last_frame_uri")?;
    if let Some(last) = last_frame_uri {
        if base.duration.is_none() {
            return Err(invalid(
                "duration",
                "automatic duration (null) cannot be combined with last_frame_uri",
            ));
        }
        media_uri(last, "last_frame_uri")?;
    }
    Ok((
        ImageToVideoRequest {
            image_uri: image_uri.to_owned(),
            last_frame_uri: last_frame_uri.map(str::to_owned),
            base,
        },
        model,
    ))
}

fn t2v_request(
    api: Api,
    b: &TextToVideoRequest,
    model: &LtxModel,
    task: Task,
    keyframes: Vec<Keyframe>,
) -> Result<GenerationRequest, ApiError> {
    let res = Resolution::parse(&b.resolution)?;
    let mut req =
        GenerationRequest::text(api.protocol(), model.target.engine_name(), b.prompt.clone());
    req.task = task;
    req.canvas = CanvasSpec::Exact {
        width: res.width,
        height: res.height,
    };
    req.timing = TimingSpec {
        length: match b.duration {
            // The matrix's longest clips (20 s at 25 fps, 10 s at 50 fps)
            // are past the engine's 481-frame grid: they run at its longest.
            Some(d) => Length::Seconds {
                value: d as f64,
                snap: Snap::Nearest,
            },
            None => Length::Auto,
        },
        fps: Some(b.fps.unwrap_or(rules::API_DEFAULT_FPS)),
    };
    req.keyframes = keyframes;
    if b.generate_audio == Some(false) {
        req.audio_out = AudioOut::Silent;
    }
    req.prompt_expansion = Some(enhance_prompt(b.enhance_prompt));
    Ok(req)
}

fn i2v_request(
    api: Api,
    b: &ImageToVideoRequest,
    model: &LtxModel,
) -> Result<GenerationRequest, ApiError> {
    let mut keyframes = vec![Keyframe {
        at: Anchor::First,
        image: media_uri(&b.image_uri, "image_uri")?,
    }];
    let task = match &b.last_frame_uri {
        None => Task::I2V,
        Some(last) => {
            keyframes.push(Keyframe {
                at: Anchor::Last,
                image: media_uri(last, "last_frame_uri")?,
            });
            Task::Keyframes
        }
    };
    t2v_request(api, &b.base, model, task, keyframes)
}

/// `enhance_prompt` (ours, named after the LTX CLI's `--enhance-prompt`;
/// the OAS has no switch and the hosted API enhances server-side): a
/// boolean opts in or out of the server's upsampler; absent, the hosted
/// default (on) applies only with `[upsampler] default_on`
/// (docs/serve/prompt-upsampling.md). A non-boolean is ignored, like any
/// unknown field.
fn enhance_prompt(v: Option<bool>) -> PromptExpansion {
    match v {
        Some(on) => PromptExpansion::explicit(
            if on {
                ExpansionMode::Balanced
            } else {
                ExpansionMode::Disabled
            },
            "enhance_prompt",
        ),
        None => PromptExpansion::default_for(ExpansionMode::Balanced, "enhance_prompt"),
    }
}

/// The A2V default model (OAS: `AudioToVideoRequest.model` defaults to it).
pub const A2V_DEFAULT_MODEL: &str = "ltx-2-3-pro";

/// `AudioToVideoRequest` (ltx §2.5) → an `A2V` request (see the module docs).
pub fn normalize_a2v(
    api: Api,
    models: &LtxModels,
    o: &Map<String, Value>,
) -> Result<GenerationRequest, ApiError> {
    let (b, model) = read_a2v(models, o)?;
    a2v_request(api, &b, model)
}

fn read_a2v<'m>(
    models: &'m LtxModels,
    o: &Map<String, Value>,
) -> Result<(AudioToVideoRequest, &'m LtxModel), ApiError> {
    let f = Fields(o);
    let audio_uri = f.req_str("audio_uri")?;
    media_uri(audio_uri, "audio_uri")?;
    let image_uri = f.opt_str("image_uri")?;
    if let Some(s) = image_uri {
        media_uri(s, "image_uri")?;
    }
    let last_frame_uri = f.opt_str("last_frame_uri")?;
    if let Some(s) = last_frame_uri {
        media_uri(s, "last_frame_uri")?;
    }
    if last_frame_uri.is_some() && image_uri.is_none() {
        return Err(invalid(
            "last_frame_uri",
            "last_frame_uri requires image_uri",
        ));
    }
    let prompt = f.opt_str("prompt")?;
    let p = prompt.unwrap_or_default();
    if p.trim().is_empty() && image_uri.is_none() {
        return Err(invalid(
            "prompt",
            "prompt is required if image_uri is not provided",
        ));
    }
    check_prompt(p)?;
    let model_id = f.opt_str("model")?;
    let id = model_id.unwrap_or(A2V_DEFAULT_MODEL);
    if id == "ltx-2-3-fast" {
        return Err(invalid("model", "ltx-2-3-fast does not support audio-to-video; use ltx-2-3-pro, ltx-2-5-fast or ltx-2-5-pro"));
    }
    let model = models.lookup(id)?;
    let resolution = f.opt_str("resolution")?;
    if let Some(r) = resolution {
        Resolution::parse(r)?;
    }
    let fps = f.int("fps")?;
    models::check_fps(fps.unwrap_or(rules::API_DEFAULT_FPS))?;
    let camera_motion = camera_motion(&f)?;
    Ok((
        AudioToVideoRequest {
            audio_uri: audio_uri.to_owned(),
            image_uri: image_uri.map(str::to_owned),
            last_frame_uri: last_frame_uri.map(str::to_owned),
            prompt: prompt.map(str::to_owned),
            model: model_id.map(str::to_owned),
            resolution: resolution.map(str::to_owned),
            fps,
            camera_motion,
        },
        model,
    ))
}

fn a2v_request(
    api: Api,
    b: &AudioToVideoRequest,
    model: &LtxModel,
) -> Result<GenerationRequest, ApiError> {
    let res = b.resolution.as_deref().map(Resolution::parse).transpose()?;
    let id = b.model.as_deref().unwrap_or(A2V_DEFAULT_MODEL);
    let short =
        matches!(res.map(|r| r.tier), Some(ResTier::P1440 | ResTier::K4)) || id == "ltx-2-5-pro";
    let max_s = if short {
        A2V_MAX_S_SHORT
    } else {
        A2V_MAX_S_LONG
    };
    let mut req = GenerationRequest::text(
        api.protocol(),
        model.target.engine_name(),
        b.prompt.clone().unwrap_or_default(),
    );
    req.task = Task::A2V;
    req.canvas = match res {
        Some(r) => CanvasSpec::Exact {
            width: r.width,
            height: r.height,
        },
        // "Portrait image → 1080x1920, landscape → 1920x1080; no image → 1920x1080".
        None => {
            let (width, height) = ResTier::P1080.landscape();
            CanvasSpec::Oriented { width, height }
        }
    };
    // The audio sets the length (`negotiate`).
    req.timing = TimingSpec {
        length: Length::ModelDefault,
        fps: Some(b.fps.unwrap_or(rules::API_DEFAULT_FPS)),
    };
    if let Some(first) = &b.image_uri {
        req.keyframes.push(Keyframe {
            at: Anchor::First,
            image: media_uri(first, "image_uri")?,
        });
    }
    if let Some(l) = &b.last_frame_uri {
        req.keyframes.push(Keyframe {
            at: Anchor::Last,
            image: media_uri(l, "last_frame_uri")?,
        });
    }
    req.audio_in = Some(AudioInput {
        media: media_uri(&b.audio_uri, "audio_uri")?,
        role: AudioRole::Drive,
        max_s: Some(max_s),
    });
    Ok(req)
}

/// The retake / extend default model (OAS: `ltx-2-3-pro`, the only value
/// listed; this server takes its other ids too).
pub const EDIT_DEFAULT_MODEL: &str = "ltx-2-3-pro";

/// The fields both edit bodies start with.
struct EditCommon<'a, 'm> {
    video_uri: &'a str,
    prompt: Option<&'a str>,
    model_id: Option<&'a str>,
    model: &'m LtxModel,
}

/// `prompt` (optional on the edit endpoints), `model` and `video_uri`.
fn read_edit_common<'a, 'm>(
    models: &'m LtxModels,
    f: &Fields<'a>,
) -> Result<EditCommon<'a, 'm>, ApiError> {
    let video_uri = f.req_str("video_uri")?;
    media_uri(video_uri, "video_uri")?;
    let prompt = f.opt_str("prompt")?;
    check_prompt(prompt.unwrap_or_default())?;
    let model_id = f.opt_str("model")?;
    let model = models.lookup(model_id.unwrap_or(EDIT_DEFAULT_MODEL))?;
    Ok(EditCommon {
        video_uri,
        prompt,
        model_id,
        model,
    })
}

fn edit_request(
    api: Api,
    model: &LtxModel,
    prompt: Option<&str>,
    video_uri: &str,
) -> Result<(GenerationRequest, MediaRef), ApiError> {
    Ok((
        GenerationRequest::text(
            api.protocol(),
            model.target.engine_name(),
            prompt.unwrap_or_default(),
        ),
        media_uri(video_uri, "video_uri")?,
    ))
}

/// `EditVideoRequest` (ltx §2.6) → a `Retake` request.
pub fn normalize_retake(
    api: Api,
    models: &LtxModels,
    o: &Map<String, Value>,
) -> Result<GenerationRequest, ApiError> {
    let (b, model) = read_retake(models, o)?;
    let (mut req, video) = edit_request(api, model, b.prompt.as_deref(), &b.video_uri)?;
    // "Clamped to video length": the window is clamped at negotiation; this
    // server regenerates at most 20 s at once.
    let duration_s = b.duration.min(fastvideo_protocol::EDIT_DURATION_MAX_S);
    let mode = match b.mode.as_deref() {
        None => RetakeMode::default(),
        Some(m) => RetakeMode::parse(m).ok_or_else(|| invalid("mode", "mode is invalid"))?,
    };
    if let Some(r) = &b.resolution {
        let res = Resolution::parse(r)?;
        req.canvas = CanvasSpec::Exact {
            width: res.width,
            height: res.height,
        };
    }
    req.task = Task::Retake;
    req.edit = Some(VideoEdit {
        video,
        op: EditOp::Retake {
            start_s: b.start_time,
            duration_s,
            mode,
        },
    });
    Ok(req)
}

fn read_retake<'m>(
    models: &'m LtxModels,
    o: &Map<String, Value>,
) -> Result<(EditVideoRequest, &'m LtxModel), ApiError> {
    let f = Fields(o);
    let EditCommon {
        video_uri,
        prompt,
        model_id,
        model,
    } = read_edit_common(models, &f)?;
    let start_time = f
        .num("start_time")?
        .ok_or_else(|| invalid("start_time", "start_time is required"))?;
    if start_time < 0.0 {
        return Err(invalid("start_time", "start_time must be at least 0"));
    }
    let duration = f
        .num("duration")?
        .ok_or_else(|| invalid("duration", "duration is required"))?;
    if duration < fastvideo_protocol::EDIT_DURATION_MIN_S {
        return Err(invalid(
            "duration",
            format!(
                "duration must be at least {}",
                fastvideo_protocol::EDIT_DURATION_MIN_S
            ),
        ));
    }
    let mode = f.opt_str("mode")?;
    if let Some(m) = mode {
        if RetakeMode::parse(m).is_none() {
            return Err(invalid(
                "mode",
                "mode must be one of: replace_audio, replace_video, replace_audio_and_video",
            ));
        }
    }
    let resolution = f.opt_str("resolution")?;
    if let Some(r) = resolution {
        if !RETAKE_RESOLUTIONS.contains(&r) {
            return Err(invalid(
                "resolution",
                format!(
                    "resolution must be one of: {}",
                    RETAKE_RESOLUTIONS.join(", ")
                ),
            ));
        }
        Resolution::parse(r)?;
    }
    Ok((
        EditVideoRequest {
            video_uri: video_uri.to_owned(),
            prompt: prompt.map(str::to_owned),
            model: model_id.map(str::to_owned),
            start_time,
            duration,
            mode: mode.map(str::to_owned),
            resolution: resolution.map(str::to_owned),
        },
        model,
    ))
}

/// `ExtendVideoRequest` (ltx §2.7) → an `Extend` request.
pub fn normalize_extend(
    api: Api,
    models: &LtxModels,
    o: &Map<String, Value>,
) -> Result<GenerationRequest, ApiError> {
    let (b, model) = read_extend(models, o)?;
    let (mut req, video) = edit_request(api, model, b.prompt.as_deref(), &b.video_uri)?;
    let at = match b.mode.as_deref() {
        None => ExtendAt::End,
        Some(m) => ExtendAt::parse(m).ok_or_else(|| invalid("mode", "mode is invalid"))?,
    };
    req.task = Task::Extend;
    req.edit = Some(VideoEdit {
        video,
        op: EditOp::Extend {
            duration_s: b.duration,
            at,
            context_s: b.context,
        },
    });
    Ok(req)
}

fn read_extend<'m>(
    models: &'m LtxModels,
    o: &Map<String, Value>,
) -> Result<(ExtendVideoRequest, &'m LtxModel), ApiError> {
    let f = Fields(o);
    let EditCommon {
        video_uri,
        prompt,
        model_id,
        model,
    } = read_edit_common(models, &f)?;
    let (lo, hi) = (
        fastvideo_protocol::EDIT_DURATION_MIN_S,
        fastvideo_protocol::EDIT_DURATION_MAX_S,
    );
    let duration = f
        .num("duration")?
        .ok_or_else(|| invalid("duration", "duration is required"))?;
    if !(lo..=hi).contains(&duration) {
        return Err(invalid(
            "duration",
            format!("duration must be between {lo} and {hi} seconds"),
        ));
    }
    let mode = f.opt_str("mode")?;
    if let Some(m) = mode {
        if ExtendAt::parse(m).is_none() {
            return Err(invalid("mode", "mode must be one of: start, end"));
        }
    }
    let context = f.num("context")?;
    if let Some(c) = context {
        let (clo, chi) = (
            fastvideo_protocol::EXTEND_CONTEXT_MIN_S,
            fastvideo_protocol::EXTEND_CONTEXT_MAX_S,
        );
        if !(clo..=chi).contains(&c) {
            return Err(invalid(
                "context",
                format!("context must be between {clo} and {chi} seconds"),
            ));
        }
    }
    Ok((
        ExtendVideoRequest {
            video_uri: video_uri.to_owned(),
            prompt: prompt.map(str::to_owned),
            model: model_id.map(str::to_owned),
            duration,
            mode: mode.map(str::to_owned),
            context,
        },
        model,
    ))
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
        normalize(
            Endpoint::ImageToVideo,
            Api::V2,
            &LtxModels::default(),
            &body,
        )
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
        assert_eq!(
            r.canvas,
            CanvasSpec::Exact {
                width: 1920,
                height: 1080
            }
        );
        assert_eq!(r.timing.fps, Some(24));
        assert_eq!(r.audio_out, AudioOut::ModelDefault);
        let r = t2v(with("generate_audio", json!(false))).unwrap();
        assert_eq!(r.audio_out, AudioOut::Silent);
        assert_eq!(t2v(with("duration", json!(8.0))).unwrap().timing, r.timing);
    }

    #[test]
    fn enhance_prompt_maps_to_the_upsampler() {
        use fastvideo_protocol::{ExpansionMode as M, PromptExpansion as P};
        const F: &str = "enhance_prompt";
        // Absent: the hosted default (on), applied only with `default_on`.
        assert_eq!(
            t2v(base()).unwrap().prompt_expansion,
            Some(P::default_for(M::Balanced, F))
        );
        assert_eq!(
            t2v(with(F, json!(true))).unwrap().prompt_expansion,
            Some(P::explicit(M::Balanced, F))
        );
        assert_eq!(
            t2v(with(F, json!(false))).unwrap().prompt_expansion,
            Some(P::explicit(M::Disabled, F))
        );
        // Not a boolean: ignored like any unknown field.
        assert_eq!(
            t2v(with(F, json!("yes"))).unwrap().prompt_expansion,
            Some(P::default_for(M::Balanced, F))
        );
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
            assert_eq!(
                msg(t2v(with("model", json!(id)))).0,
                ErrorKind::InvalidRequest
            );
        }
    }

    #[test]
    fn i2v_shapes() {
        let b = with("image_uri", json!("ltx://uploads/abc"));
        let r = i2v(b.clone()).unwrap();
        assert_eq!(r.task, Task::I2V);
        assert_eq!(
            r.keyframes[0].image,
            MediaRef::Upload(fastvideo_protocol::UploadId("abc".into()))
        );
        let mut k = b.clone();
        k["last_frame_uri"] = json!("https://example.com/l.png");
        let r = i2v(k.clone()).unwrap();
        assert_eq!(r.task, Task::Keyframes);
        assert_eq!(r.keyframes[1].at, Anchor::Last);
        k["duration"] = Value::Null;
        assert!(msg(i2v(k)).1.contains("last_frame_uri"));
        assert_eq!(msg(i2v(base())).1, "image_uri is required");
        for bad in [
            "http://example.com/a.png",
            "ftp://x/y",
            "mm_file://1",
            "ltx://other/1",
            "ltx://uploads/",
        ] {
            assert!(i2v(with("image_uri", json!(bad))).is_err(), "{bad}");
        }
        assert!(i2v(with("image_uri", json!("data:image/png;base64,AAAA"))).is_ok());
        // T2V refuses image fields rather than dropping them.
        assert!(t2v(with("image_uri", json!("https://example.com/a.png"))).is_err());
    }

    fn a2v(body: Value) -> Result<GenerationRequest, ApiError> {
        normalize(
            Endpoint::AudioToVideo,
            Api::V2,
            &LtxModels::default(),
            &body,
        )
    }

    #[test]
    fn a2v_shapes() {
        // Defaults: ltx-2-3-pro (our max tier), 1920x1080 or portrait by the
        // image, the length from the audio, 20 s of audio at 1080p.
        let r = a2v(json!({"audio_uri": "https://example.com/a.mp3", "prompt": "a man talks"}))
            .unwrap();
        assert_eq!((r.task, r.model.as_str()), (Task::A2V, "ltx-pro"));
        assert_eq!(
            r.canvas,
            CanvasSpec::Oriented {
                width: 1920,
                height: 1080
            }
        );
        assert_eq!(
            r.timing,
            TimingSpec {
                length: Length::ModelDefault,
                fps: Some(24)
            }
        );
        let a = r.audio_in.unwrap();
        assert_eq!((a.role, a.max_s), (AudioRole::Drive, Some(20)));
        assert!(r.keyframes.is_empty());
        // An image: the prompt may be empty; a last frame with it.
        let r = a2v(json!({"audio_uri": "ltx://uploads/a", "image_uri": "ltx://uploads/i", "last_frame_uri": "https://example.com/l.png", "prompt": "", "model": "ltx-2-5-fast", "resolution": "1080x1920", "fps": 25})).unwrap();
        assert_eq!(r.keyframes.len(), 2);
        assert_eq!(
            r.canvas,
            CanvasSpec::Exact {
                width: 1080,
                height: 1920
            }
        );
        assert_eq!((r.model.as_str(), r.timing.fps), ("ltx-turbo", Some(25)));
        // 10 s: pro, and 1440p / 4K.
        let pro = a2v(json!({"audio_uri": "https://example.com/a.mp3", "prompt": "p", "model": "ltx-2-5-pro"})).unwrap();
        assert_eq!(pro.audio_in.unwrap().max_s, Some(10));
        let k4 = a2v(json!({"audio_uri": "https://example.com/a.mp3", "prompt": "p", "model": "ltx-2-5-fast", "resolution": "3840x2160"})).unwrap();
        assert_eq!(k4.audio_in.unwrap().max_s, Some(10));
        // Refusals.
        for (b, param) in [
            (json!({"prompt": "p"}), "audio_uri"),
            (json!({"audio_uri": "https://example.com/a.mp3"}), "prompt"),
            (
                json!({"audio_uri": "https://example.com/a.mp3", "prompt": "p", "last_frame_uri": "https://example.com/l.png"}),
                "last_frame_uri",
            ),
            (
                json!({"audio_uri": "http://example.com/a.mp3", "prompt": "p"}),
                "audio_uri",
            ),
            (
                json!({"audio_uri": "https://example.com/a.mp3", "prompt": "p", "model": "ltx-2-3-fast"}),
                "model",
            ),
            (
                json!({"audio_uri": "https://example.com/a.mp3", "prompt": "p", "fps": 30}),
                "fps",
            ),
        ] {
            let e = a2v(b.clone()).unwrap_err();
            assert_eq!(e.param.as_deref(), Some(param), "{b}");
        }
        let (k, _) = msg(a2v(
            json!({"audio_uri": "https://example.com/a.mp3", "prompt": "p", "camera_motion": "jib_up"}),
        ));
        assert_eq!(k, ErrorKind::Unsupported(GapId::LtxCameraMotion));
        assert_eq!(Endpoint::of_task(Task::A2V), Some(Endpoint::AudioToVideo));
    }

    #[test]
    fn prompt_limit() {
        assert!(t2v(with("prompt", json!("é".repeat(5000)))).is_ok());
        assert!(t2v(with("prompt", json!("a".repeat(5001)))).is_err());
    }

    #[test]
    fn retake_and_extend_shapes() {
        let ret = |b: Value| normalize(Endpoint::Retake, Api::V2, &LtxModels::default(), &b);
        let ext = |b: Value| normalize(Endpoint::Extend, Api::V2, &LtxModels::default(), &b);
        let r = ret(
            json!({"video_uri": "https://example.com/v.mp4", "start_time": 1.5, "duration": 3}),
        )
        .unwrap();
        assert_eq!(
            (r.task, r.model.as_str(), r.prompt.as_str()),
            (Task::Retake, "ltx-pro", "")
        );
        assert_eq!(
            r.edit.as_ref().unwrap().op,
            EditOp::Retake {
                start_s: 1.5,
                duration_s: 3.0,
                mode: RetakeMode::ReplaceAudioAndVideo
            }
        );
        assert_eq!(r.canvas, CanvasSpec::ModelDefault);
        let r = ret(json!({"video_uri": "ltx://uploads/v", "start_time": 0, "duration": 30, "mode": "replace_audio", "resolution": "1080x1920", "prompt": "rain"})).unwrap();
        assert_eq!(
            r.edit.unwrap().op,
            EditOp::Retake {
                start_s: 0.0,
                duration_s: 20.0,
                mode: RetakeMode::ReplaceAudio
            }
        );
        assert_eq!(
            r.canvas,
            CanvasSpec::Exact {
                width: 1080,
                height: 1920
            }
        );
        for (b, param) in [
            (json!({"start_time": 0, "duration": 3}), "video_uri"),
            (
                json!({"video_uri": "https://example.com/v.mp4", "duration": 3}),
                "start_time",
            ),
            (
                json!({"video_uri": "https://example.com/v.mp4", "start_time": -1, "duration": 3}),
                "start_time",
            ),
            (
                json!({"video_uri": "https://example.com/v.mp4", "start_time": 0, "duration": 1.5}),
                "duration",
            ),
            (
                json!({"video_uri": "https://example.com/v.mp4", "start_time": 0, "duration": 3, "mode": "both"}),
                "mode",
            ),
            (
                json!({"video_uri": "https://example.com/v.mp4", "start_time": 0, "duration": 3, "resolution": "1280x720"}),
                "resolution",
            ),
            (
                json!({"video_uri": "http://example.com/v.mp4", "start_time": 0, "duration": 3}),
                "video_uri",
            ),
        ] {
            assert_eq!(
                ret(b.clone()).unwrap_err().param.as_deref(),
                Some(param),
                "{b}"
            );
        }
        let x = ext(json!({"video_uri": "https://example.com/v.mp4", "duration": 5, "prompt": "the car drives on"})).unwrap();
        assert_eq!((x.task, x.model.as_str()), (Task::Extend, "ltx-pro"));
        assert_eq!(
            x.edit.unwrap().op,
            EditOp::Extend {
                duration_s: 5.0,
                at: ExtendAt::End,
                context_s: None
            }
        );
        let x = ext(json!({"video_uri": "https://example.com/v.mp4", "duration": 2, "mode": "start", "context": 4, "model": "ltx-2-5-fast"})).unwrap();
        assert_eq!(x.model, "ltx-turbo");
        assert_eq!(
            x.edit.unwrap().op,
            EditOp::Extend {
                duration_s: 2.0,
                at: ExtendAt::Start,
                context_s: Some(4.0)
            }
        );
        for (b, param) in [
            (
                json!({"video_uri": "https://example.com/v.mp4"}),
                "duration",
            ),
            (
                json!({"video_uri": "https://example.com/v.mp4", "duration": 21}),
                "duration",
            ),
            (
                json!({"video_uri": "https://example.com/v.mp4", "duration": 5, "mode": "middle"}),
                "mode",
            ),
            (
                json!({"video_uri": "https://example.com/v.mp4", "duration": 5, "context": 0.5}),
                "context",
            ),
        ] {
            assert_eq!(
                ext(b.clone()).unwrap_err().param.as_deref(),
                Some(param),
                "{b}"
            );
        }
        assert_eq!(Endpoint::of_task(Task::Retake), Some(Endpoint::Retake));
        assert_eq!(Endpoint::from_segment("extend"), Some(Endpoint::Extend));
    }
}
