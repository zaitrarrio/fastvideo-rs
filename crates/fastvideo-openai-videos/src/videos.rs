//! FastVideo `/v1/videos*` routes (design §4.1; minimax-fastvideo §2.1, §3.3).
//!
//! | Route | Handler |
//! |---|---|
//! | `POST /v1/videos`, `POST /v1/videos/generations` | [`create`](routes): JSON, multipart or form; `VideoResponse` `queued` |
//! | `POST /v1/videos/sync` | blocks; `video/mp4` with the `X-*` metric headers |
//! | `GET /v1/videos?after&limit&order` | `{object:"list",data,first_id,last_id,has_more}` |
//! | `GET /v1/videos/{id}` | `VideoResponse` (a failed job is 200 `failed`) |
//! | `GET /v1/videos/{id}/content?variant=video` | the MP4; 400 other variant, 422 failed, 404 in progress |
//! | `DELETE /v1/videos/{id}` | cancels a running job, removes it and its file |
//!
//! [`VideosCreate::normalize`] is pure; the parts that need the model's caps
//! (default short edge, the H3-only `task`, FastVideo's H3 rules) run in
//! [`apply_model`].

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use axum::body::Bytes;
use axum::extract::{DefaultBodyLimit, FromRequest, Multipart, Path, Query, Request, State};
use axum::http::{header, HeaderMap};
use axum::response::Response;
use axum::routing::{get, post};
use axum::Router;
use base64::Engine as _;
use fastvideo_protocol::{
    Anchor, ApiError, ArtifactLocation, BatchProtocol, CanvasSpec, ErrorCtx, Family,
    GenerationRequest, HttpReply, Job, JobId, JobStatus, JobView, KeyId, Keyframe, Length,
    ListQuery, MediaKind, MediaRef, ModelCaps, NormalizeCtx, ProtocolId, Ratio, Reference,
    SamplingOverrides, Snap, SortOrder, SubmitEndpoint, Task, TimingSpec, ViewCtx,
};
use fastvideo_serve_kit::events::{cancel_job, wait_terminal};
use fastvideo_serve_kit::handlers::{find_job, submit_request};
use fastvideo_serve_kit::{handlers, into_response, ArtifactBody, ServeCtx};
use serde::Deserialize;
use serde_json::{json, Map, Value};

use crate::error::{body_error, openai_error, openai_error_status};
use crate::models::{default_model, resolve_public};
use crate::VideosConfig;

/// `Aspect { short_edge }` placeholder for "the model's default short edge";
/// [`apply_model`] replaces it.
pub const DEFAULT_SHORT_EDGE: u32 = 0;

/// FastVideo's H3 minimum clip length (5-15 s, `packing.py:21-29`).
pub const H3_MIN_SECONDS: u32 = 5;

/// How long signed URLs handed out by redirects stay valid.
const URL_TTL: Duration = Duration::from_secs(3600);

// ---------------------------------------------------------------- protocol

/// The FastVideo `/v1/videos` API (`ProtocolId::OpenAiVideos`).
#[derive(Clone, Copy, Debug, Default)]
pub struct OpenAiVideos;

impl BatchProtocol for OpenAiVideos {
    fn id(&self) -> ProtocolId {
        ProtocolId::OpenAiVideos
    }
    /// `video_gen_<32hex>`.
    fn new_external_id(&self, job: JobId) -> String {
        format!("video_gen_{}", job.0.simple())
    }
    fn render_error(&self, err: &ApiError, _cx: &ErrorCtx) -> HttpReply {
        openai_error(err)
    }
}

// ---------------------------------------------------------------- request

/// `seconds`: an integer >= 1 or a string matching `^[1-9]\d*$`.
#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(untagged)]
pub enum Seconds {
    Int(u64),
    Str(String),
}

impl Seconds {
    fn value(&self) -> Result<u32, ApiError> {
        let bad = || ApiError::invalid_param("seconds", "seconds must be a positive integer");
        match self {
            Seconds::Int(0) => Err(bad()),
            Seconds::Int(n) => u32::try_from(*n).map_err(|_| bad()),
            Seconds::Str(s) => {
                let ok = s.bytes().next().is_some_and(|b| (b'1'..=b'9').contains(&b))
                    && s.bytes().all(|b| b.is_ascii_digit());
                if !ok {
                    return Err(bad());
                }
                s.parse().map_err(|_| bad())
            }
        }
    }
    /// The echoed string form.
    pub fn as_string(&self) -> String {
        match self {
            Seconds::Int(n) => n.to_string(),
            Seconds::Str(s) => s.clone(),
        }
    }
}

/// One object or a list of them.
#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(untagged)]
pub enum OneOrMany<T> {
    One(T),
    Many(Vec<T>),
}

impl<T> OneOrMany<T> {
    fn into_vec(self) -> Vec<T> {
        match self {
            OneOrMany::One(t) => vec![t],
            OneOrMany::Many(v) => v,
        }
    }
}

/// `{image_url}` | `{file_id}`.
#[derive(Clone, Debug, Default, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ImageRef {
    pub image_url: Option<String>,
    pub file_id: Option<String>,
}

/// `{video_url}` | `{file_id}`.
#[derive(Clone, Debug, Default, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct VideoRef {
    pub video_url: Option<String>,
    pub file_id: Option<String>,
}

/// `{audio_url}`.
#[derive(Clone, Debug, Default, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AudioRef {
    pub audio_url: Option<String>,
}

/// vLLM-Omni `video_params`.
#[derive(Clone, Debug, Default, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct VideoParams {
    pub width: Option<u32>,
    pub height: Option<u32>,
    pub num_frames: Option<u32>,
    pub fps: Option<u32>,
}

/// `VideoGenerationRequest` (`openai/protocol.py:112-201`), `extra="forbid"`.
#[derive(Clone, Debug, Default, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct VideoGenerationRequest {
    pub prompt: String,
    pub model: Option<String>,
    pub seconds: Option<Seconds>,
    pub size: Option<String>,
    pub width: Option<u32>,
    pub height: Option<u32>,
    pub fps: Option<u32>,
    pub num_frames: Option<u32>,
    pub video_params: Option<VideoParams>,
    pub aspect_ratio: Option<String>,
    pub short_edge: Option<u32>,
    pub image_reference: Option<OneOrMany<ImageRef>>,
    pub video_reference: Option<OneOrMany<VideoRef>>,
    pub audio_reference: Option<OneOrMany<AudioRef>>,
    pub input_reference: Option<String>,
    pub reference_url: Option<String>,
    pub video_path: Option<String>,
    pub video_url: Option<String>,
    pub task: Option<String>,
    pub n: Option<u32>,
    pub num_outputs_per_prompt: Option<u32>,
    pub quality: Option<String>,
    pub negative_prompt: Option<String>,
    pub num_inference_steps: Option<u32>,
    pub guidance_scale: Option<f32>,
    pub guidance_scale_2: Option<f32>,
    pub boundary_ratio: Option<f32>,
    pub flow_shift: Option<f64>,
    pub true_cfg_scale: Option<f64>,
    pub seed: Option<i64>,
    pub max_sequence_length: Option<u32>,
    pub enable_teacache: Option<bool>,
    pub generate_sound: Option<bool>,
    pub sound_duration: Option<f64>,
    pub start_time_seconds: Option<f64>,
    pub enable_frame_interpolation: Option<bool>,
    pub frame_interpolation_exp: Option<Value>,
    pub frame_interpolation_scale: Option<Value>,
    pub frame_interpolation_model_path: Option<Value>,
    pub lora: Option<Value>,
    pub extra_params: Option<Map<String, Value>>,
    pub user: Option<String>,
}

/// `extra_params` keys FastVideo accepts (`fastvideo/api/compat.py:47-56`).
/// All are recipe-level here, so any of them is refused per request.
pub const EXTRA_PARAMS: [&str; 8] = [
    "ltx2_audio_latents",
    "ltx2_audio_clean_latent",
    "ltx2_audio_denoise_mask",
    "audio_num_frames",
    "video_position_offset_sec",
    "vsa_mode",
    "vsa_dense_first_n_steps",
    "vsa_dense_layers",
];

const QUALITIES: [&str; 4] = ["auto", "default", "standard", "hd"];

/// Form fields that carry JSON strings (minimax-fastvideo §2.1).
const FORM_JSON: [&str; 8] = [
    "image_reference",
    "video_reference",
    "audio_reference",
    "video_params",
    "lora",
    "extra_params",
    "extra_body",
    "extra_json",
];
const FORM_INT: [&str; 11] = [
    "width",
    "height",
    "fps",
    "num_frames",
    "short_edge",
    "n",
    "num_outputs_per_prompt",
    "num_inference_steps",
    "seed",
    "max_sequence_length",
    "frame_interpolation_exp",
];
const FORM_FLOAT: [&str; 8] = [
    "guidance_scale",
    "guidance_scale_2",
    "boundary_ratio",
    "flow_shift",
    "true_cfg_scale",
    "sound_duration",
    "start_time_seconds",
    "frame_interpolation_scale",
];
const FORM_BOOL: [&str; 3] = [
    "enable_teacache",
    "generate_sound",
    "enable_frame_interpolation",
];

/// Converts form string values to the JSON types the request expects:
/// JSON-string fields are parsed, numeric and boolean fields converted.
/// Unconvertible values stay strings (and fail typed parsing with 400).
pub fn coerce_form(map: &mut Map<String, Value>) -> Result<(), ApiError> {
    for (k, v) in map.iter_mut() {
        let Value::String(s) = v else { continue };
        let k = k.as_str();
        if FORM_JSON.contains(&k) {
            *v = serde_json::from_str(s).map_err(|_| {
                ApiError::invalid_param(k, format!("`{k}` must be a JSON string in form bodies"))
            })?;
        } else if FORM_INT.contains(&k) {
            if let Ok(n) = s.trim().parse::<i64>() {
                *v = Value::from(n);
            }
        } else if FORM_FLOAT.contains(&k) {
            if let Some(n) = s
                .trim()
                .parse::<f64>()
                .ok()
                .and_then(serde_json::Number::from_f64)
            {
                *v = Value::Number(n);
            }
        } else if FORM_BOOL.contains(&k) {
            match s.trim().to_ascii_lowercase().as_str() {
                "true" | "1" | "yes" | "on" => *v = Value::Bool(true),
                "false" | "0" | "no" | "off" => *v = Value::Bool(false),
                _ => {}
            }
        }
    }
    Ok(())
}

/// Merges `extra_body` and `extra_json` objects into the top level
/// (`video_api.py:_parse_video_request`); their fields win.
pub fn merge_extra(map: &mut Map<String, Value>) -> Result<(), ApiError> {
    for key in ["extra_body", "extra_json"] {
        match map.remove(key) {
            None | Some(Value::Null) => {}
            Some(Value::Object(o)) => map.extend(o),
            Some(_) => {
                return Err(ApiError::invalid_param(
                    key,
                    format!("`{key}` must be an object"),
                ))
            }
        }
    }
    Ok(())
}

/// Parses a merged body object into the typed request (`extra="forbid"`).
pub fn parse_request(map: Map<String, Value>) -> Result<VideoGenerationRequest, ApiError> {
    serde_json::from_value(Value::Object(map)).map_err(|e| body_error(&e))
}

fn refused(field: &str) -> ApiError {
    ApiError::invalid_param(field, format!("`{field}` is not supported by this server"))
}

fn positive(field: &str, v: Option<u32>) -> Result<Option<u32>, ApiError> {
    match v {
        Some(0) => Err(ApiError::invalid_param(
            field,
            format!("`{field}` must be at least 1"),
        )),
        v => Ok(v),
    }
}

fn parse_size(s: &str) -> Result<(u32, u32), ApiError> {
    let bad = || ApiError::invalid_param("size", format!("size `{s}` must be WIDTHxHEIGHT"));
    let (w, h) = s.split_once('x').ok_or_else(bad)?;
    let digits = |t: &str| !t.is_empty() && t.bytes().all(|b| b.is_ascii_digit());
    if !digits(w) || !digits(h) {
        return Err(bad());
    }
    let (w, h): (u32, u32) = (w.parse().map_err(|_| bad())?, h.parse().map_err(|_| bad())?);
    if w == 0 || h == 0 {
        return Err(bad());
    }
    Ok((w, h))
}

fn media(
    url: Option<String>,
    file_id: Option<String>,
    field: &str,
    key: &str,
) -> Result<MediaRef, ApiError> {
    match (url, file_id) {
        (Some(u), None) => MediaRef::parse(&u, field),
        (None, Some(f)) => Ok(MediaRef::ProviderFile(format!("file_id:{f}"))),
        _ => Err(ApiError::invalid_param(
            field,
            format!("each `{field}` item needs exactly one of `{key}` or `file_id`"),
        )),
    }
}

/// `POST /v1/videos` normalization.
#[derive(Clone, Copy, Debug, Default)]
pub struct VideosCreate;

impl SubmitEndpoint for VideosCreate {
    type Body = VideoGenerationRequest;

    fn normalize(
        &self,
        b: VideoGenerationRequest,
        _cx: &NormalizeCtx,
    ) -> Result<GenerationRequest, ApiError> {
        if b.prompt.trim().is_empty() {
            return Err(ApiError::invalid_param(
                "prompt",
                "prompt must not be empty",
            ));
        }
        for (field, v) in [
            ("n", b.n),
            ("num_outputs_per_prompt", b.num_outputs_per_prompt),
        ] {
            match v {
                None | Some(1) => {}
                Some(2..=10) => {
                    return Err(ApiError::invalid_param(
                        field,
                        "exactly one video output per request is supported",
                    ))
                }
                Some(_) => {
                    return Err(ApiError::invalid_param(
                        field,
                        format!("`{field}` must be within 1..=10"),
                    ))
                }
            }
        }
        // Fields no engine path takes: refuse, never drop (design §3.2 rule 7).
        let present = [
            ("true_cfg_scale", b.true_cfg_scale.is_some()),
            ("max_sequence_length", b.max_sequence_length.is_some()),
            ("sound_duration", b.sound_duration.is_some()),
            ("start_time_seconds", b.start_time_seconds.is_some()),
            ("enable_teacache", b.enable_teacache == Some(true)),
            (
                "enable_frame_interpolation",
                b.enable_frame_interpolation == Some(true),
            ),
            (
                "frame_interpolation_exp",
                b.frame_interpolation_exp
                    .as_ref()
                    .is_some_and(|v| !v.is_null()),
            ),
            (
                "frame_interpolation_scale",
                b.frame_interpolation_scale
                    .as_ref()
                    .is_some_and(|v| !v.is_null()),
            ),
            (
                "frame_interpolation_model_path",
                b.frame_interpolation_model_path
                    .as_ref()
                    .is_some_and(|v| !v.is_null()),
            ),
        ];
        if let Some((f, _)) = present.iter().find(|(_, p)| *p) {
            return Err(refused(f));
        }
        if b.lora.as_ref().is_some_and(|v| !v.is_null()) {
            return Err(ApiError::unsupported(fastvideo_protocol::GapId::Lora).with_param("lora"));
        }
        if let Some(k) = b.extra_params.as_ref().and_then(|m| m.keys().next()) {
            return Err(if EXTRA_PARAMS.contains(&k.as_str()) {
                ApiError::invalid_param(
                    "extra_params",
                    format!("`extra_params.{k}` is fixed by the model recipe and cannot be set per request"),
                )
            } else {
                ApiError::invalid_param("extra_params", format!("unknown `extra_params` key `{k}`"))
            });
        }
        if b.video_path.is_some() || b.video_url.is_some() {
            let f = if b.video_path.is_some() {
                "video_path"
            } else {
                "video_url"
            };
            return Err(ApiError::invalid_param(
                f,
                "direct video input is not supported; send `video_reference` with task `ref2va`",
            ));
        }
        if let Some(q) = &b.quality {
            if !QUALITIES.contains(&q.as_str()) {
                return Err(ApiError::invalid_param(
                    "quality",
                    "quality must be one of auto, default, standard, hd",
                ));
            }
        }
        if let Some(s) = b.num_inference_steps {
            if !(1..=200).contains(&s) {
                return Err(ApiError::invalid_param(
                    "num_inference_steps",
                    "num_inference_steps must be within 1..=200",
                ));
            }
        }
        if let Some(g) = b.guidance_scale {
            if !(0.0..=20.0).contains(&g) {
                return Err(ApiError::invalid_param(
                    "guidance_scale",
                    "guidance_scale must be within 0..=20",
                ));
            }
        }
        if let Some(r) = b.boundary_ratio {
            if !(0.0..=1.0).contains(&r) {
                return Err(ApiError::invalid_param(
                    "boundary_ratio",
                    "boundary_ratio must be within 0..=1",
                ));
            }
        }
        let seed = match b.seed {
            Some(s) if s < 0 => {
                return Err(ApiError::invalid_param("seed", "seed must be non-negative"))
            }
            s => s.map(|s| s as u64),
        };

        let mut req = GenerationRequest::text(
            ProtocolId::OpenAiVideos,
            b.model.clone().unwrap_or_default(),
            b.prompt.clone(),
        );
        req.seed = seed;
        req.negative_prompt = b.negative_prompt.clone().filter(|n| !n.trim().is_empty());

        // Canvas: size > width/height > video_params > aspect_ratio.
        let vp = b.video_params.clone().unwrap_or_default();
        let (w, h) = (positive("width", b.width)?, positive("height", b.height)?);
        let (vw, vh) = (
            positive("video_params.width", vp.width)?,
            positive("video_params.height", vp.height)?,
        );
        req.canvas = if let Some(s) = &b.size {
            let (w, h) = parse_size(s)?;
            CanvasSpec::Exact {
                width: w,
                height: h,
            }
        } else if w.is_some() || h.is_some() {
            match (w, h) {
                (Some(width), Some(height)) => CanvasSpec::Exact { width, height },
                _ => {
                    return Err(ApiError::invalid_param(
                        "width",
                        "width and height must both be set",
                    ))
                }
            }
        } else if vw.is_some() || vh.is_some() {
            match (vw, vh) {
                (Some(width), Some(height)) => CanvasSpec::Exact { width, height },
                _ => {
                    return Err(ApiError::invalid_param(
                        "video_params",
                        "video_params.width and height must both be set",
                    ))
                }
            }
        } else if let Some(a) = &b.aspect_ratio {
            let ratio: Ratio = a.parse()?;
            let short_edge = positive("short_edge", b.short_edge)?.unwrap_or(DEFAULT_SHORT_EDGE);
            CanvasSpec::Aspect { ratio, short_edge }
        } else {
            CanvasSpec::ModelDefault
        };
        if b.short_edge.is_some() && b.aspect_ratio.is_none() {
            return Err(ApiError::invalid_param(
                "short_edge",
                "short_edge requires aspect_ratio",
            ));
        }

        // Timing: num_frames (explicit, must be on the grid) > seconds (aligned up).
        let fps = positive("fps", b.fps)?.or(positive("video_params.fps", vp.fps)?);
        let frames = positive("num_frames", b.num_frames)?
            .or(positive("video_params.num_frames", vp.num_frames)?);
        let length = match (frames, &b.seconds) {
            (Some(n), _) => Length::Frames {
                value: n,
                snap: Snap::Exact,
            },
            (None, Some(s)) => Length::Seconds {
                value: s.value()? as f64,
                snap: Snap::AlignUp,
            },
            (None, None) => Length::ModelDefault,
        };
        req.timing = TimingSpec { length, fps };

        // Media: images (image_reference or one legacy field), videos, audio.
        let legacy = match (&b.input_reference, &b.reference_url) {
            (Some(_), Some(_)) => {
                return Err(ApiError::invalid_param(
                    "reference_url",
                    "send at most one of input_reference and reference_url",
                ))
            }
            (Some(u), None) => Some(("input_reference", u.clone())),
            (None, Some(u)) => Some(("reference_url", u.clone())),
            (None, None) => None,
        };
        let mut images = Vec::new();
        match (b.image_reference.clone(), legacy) {
            (Some(_), Some((f, _))) => {
                return Err(ApiError::invalid_param(
                    f,
                    format!("`{f}` cannot be combined with image_reference"),
                ))
            }
            (Some(list), None) => {
                for r in list.into_vec() {
                    images.push(media(
                        r.image_url,
                        r.file_id,
                        "image_reference",
                        "image_url",
                    )?);
                }
            }
            (None, Some((f, u))) => images.push(MediaRef::parse(&u, f)?),
            (None, None) => {}
        }
        let mut videos = Vec::new();
        for r in b
            .video_reference
            .clone()
            .map(OneOrMany::into_vec)
            .unwrap_or_default()
        {
            videos.push(media(
                r.video_url,
                r.file_id,
                "video_reference",
                "video_url",
            )?);
        }
        let mut audio = Vec::new();
        for r in b
            .audio_reference
            .clone()
            .map(OneOrMany::into_vec)
            .unwrap_or_default()
        {
            let u = r.audio_url.ok_or_else(|| {
                ApiError::invalid_param(
                    "audio_reference",
                    "each `audio_reference` item needs `audio_url`",
                )
            })?;
            audio.push(MediaRef::parse(&u, "audio_reference")?);
        }

        let refs_task = |images: Vec<MediaRef>, videos: Vec<MediaRef>, audio: Vec<MediaRef>| {
            if images.is_empty() && videos.is_empty() {
                return Err(ApiError::invalid_param(
                    "audio_reference",
                    "audio-only reference sets are not supported",
                ));
            }
            // FastVideo order: images, then videos, then audio.
            let refs = images
                .into_iter()
                .map(|m| Reference {
                    kind: MediaKind::Image,
                    media: m,
                })
                .chain(videos.into_iter().map(|m| Reference {
                    kind: MediaKind::Video,
                    media: m,
                }))
                .chain(audio.into_iter().map(|m| Reference {
                    kind: MediaKind::Audio,
                    media: m,
                }))
                .collect::<Vec<_>>();
            Ok(refs)
        };
        let frames_task = |images: Vec<MediaRef>| -> (Task, Vec<Keyframe>) {
            let mut it = images.into_iter();
            let first = it.next().map(|m| Keyframe {
                at: Anchor::First,
                image: m,
            });
            let last = it.next().map(|m| Keyframe {
                at: Anchor::Last,
                image: m,
            });
            match last {
                None => (Task::I2V, first.into_iter().collect()),
                Some(l) => (Task::Keyframes, first.into_iter().chain([l]).collect()),
            }
        };
        let no_av = videos.is_empty() && audio.is_empty();
        match b.task.as_deref() {
            Some("t2va") => {
                if !(images.is_empty() && no_av) {
                    return Err(ApiError::invalid_param(
                        "task",
                        "t2va takes no image, video or audio references",
                    ));
                }
            }
            Some("fl2va") => {
                if !no_av || !(1..=2).contains(&images.len()) {
                    return Err(ApiError::invalid_param(
                        "task",
                        "fl2va takes 1 or 2 images and no video or audio references",
                    ));
                }
                (req.task, req.keyframes) = frames_task(images);
            }
            Some("ref2va") => {
                req.task = Task::Ref2V;
                req.references = refs_task(images, videos, audio)?;
            }
            Some(t) => {
                return Err(ApiError::invalid_param(
                    "task",
                    format!("unknown task `{t}`; expected t2va, fl2va or ref2va"),
                ))
            }
            None if images.is_empty() && no_av => {}
            None if no_av && images.len() <= 2 => (req.task, req.keyframes) = frames_task(images),
            None => {
                req.task = Task::Ref2V;
                req.references = refs_task(images, videos, audio)?;
            }
        }

        req.sampling = SamplingOverrides {
            steps: b.num_inference_steps,
            guidance: b.guidance_scale,
            guidance_2: b.guidance_scale_2,
            flow_shift: b.flow_shift,
            boundary_ratio: b.boundary_ratio,
        };
        if b.generate_sound.is_some() {
            req.note_noop("generate_sound");
        }
        if b.quality.is_some() {
            req.note_noop("quality");
        }
        if b.user.is_some() {
            req.note_noop("user");
        }
        Ok(req)
    }

    fn submit_reply(&self, job: &Job, cx: &ViewCtx) -> HttpReply {
        HttpReply::json(200, video_response(job, cx))
    }
}

/// The caps-dependent part of normalization: resolves the model id, fills
/// the default short edge, and applies FastVideo's model rules.
///
/// - `task` is only defined for H3 (FastVideo: "only defined for MiniMax-H3").
/// - H3: `guidance_scale: 1` (the value FastH3 requires) and a negative
///   prompt are accepted no-ops, as in FastVideo; clips are 5-15 s.
pub fn apply_model(
    req: &mut GenerationRequest,
    caps: &ModelCaps,
    explicit_task: bool,
) -> Result<(), ApiError> {
    req.model = caps.id.0.clone();
    if let CanvasSpec::Aspect { short_edge, .. } = &mut req.canvas {
        if *short_edge == DEFAULT_SHORT_EDGE {
            *short_edge = *caps.canvas.short_edges.first().ok_or_else(|| {
                ApiError::internal(format!("model `{}` declares no canvas tier", caps.id))
            })?;
        }
    }
    if caps.family != Family::H3 {
        if explicit_task {
            return Err(ApiError::invalid_param(
                "task",
                "`task` is only defined for H3 models",
            ));
        }
        return Ok(());
    }
    if req.sampling.guidance == Some(1.0) {
        req.sampling.guidance = None;
        req.note_noop("guidance_scale");
    }
    if req.negative_prompt.take().is_some() {
        req.note_noop("negative_prompt");
    }
    let fps = req.timing.fps.unwrap_or(caps.fps.default);
    let min = caps.frames.next_on_grid(H3_MIN_SECONDS * fps).unwrap_or(0);
    let (short, param) = match req.timing.length {
        Length::Seconds { value, .. } => (value < H3_MIN_SECONDS as f64, "seconds"),
        Length::Frames { value, .. } => (value < min, "num_frames"),
        _ => (false, ""),
    };
    if short {
        return Err(ApiError::invalid_param(
            param,
            format!("H3 clips are 5 to 15 seconds long ({min} frames or more at {fps} fps)"),
        ));
    }
    Ok(())
}

// ---------------------------------------------------------------- views

/// FastVideo status names: `Queued` -> `queued`, `Running` -> `in_progress`,
/// `Succeeded` -> `completed`, `Failed`/`Cancelled` -> `failed`.
pub fn status_name(s: JobStatus) -> &'static str {
    match s {
        JobStatus::Queued => "queued",
        JobStatus::Running => "in_progress",
        JobStatus::Succeeded => "completed",
        JobStatus::Failed | JobStatus::Cancelled => "failed",
    }
}

/// The resolved tier and recipe, when the model has them (design §0.3, §0.6):
/// `{tier, recipe, quality_gate}`. `quality_gate: false` marks draft results.
pub fn metadata(job: &Job) -> Option<Value> {
    let r = &job.resolved;
    if r.tier.is_none() && r.recipe.is_none() {
        return None;
    }
    let mut m = Map::new();
    if let Some(t) = r.tier {
        m.insert("tier".into(), t.as_str().into());
        m.insert("quality_gate".into(), t.passes_quality_gate().into());
    }
    if let Some(rc) = &r.recipe {
        m.insert("recipe".into(), rc.clone().into());
    }
    Some(Value::Object(m))
}

fn echo_str(job: &Job, key: &str) -> Option<String> {
    match job.request_echo.get(key)? {
        Value::String(s) => Some(s.clone()),
        Value::Number(n) => Some(n.to_string()),
        _ => None,
    }
}

fn opt_f64(v: Option<f64>) -> Value {
    v.and_then(serde_json::Number::from_f64)
        .map_or(Value::Null, Value::Number)
}

/// `VideoResponse` (`openai/protocol.py:213-234`).
pub fn video_response(job: &Job, _cx: &ViewCtx) -> Value {
    let r = &job.resolved;
    let (w, h) = r.output_size();
    let status = job.status();
    let error = match &job.state {
        fastvideo_protocol::JobState::Failed(e) => {
            json!({ "code": "generation_failed", "message": e.message })
        }
        fastvideo_protocol::JobState::Cancelled => {
            json!({ "code": "generation_failed", "message": "the generation was cancelled" })
        }
        _ => Value::Null,
    };
    let seconds =
        echo_str(job, "seconds").unwrap_or_else(|| format!("{}", r.duration_s().round() as u64));
    let art = job.artifacts.first();
    let mut v = json!({
        "id": job.external_id,
        "object": "video",
        "model": job.requested_model(),
        "prompt": echo_str(job, "prompt").unwrap_or_else(|| r.prompt.clone()),
        "status": status_name(status),
        "progress": (job.progress.clamp(0.0, 1.0) * 100.0).round() as u32,
        "created_at": job.created_at.unix_timestamp(),
        "size": format!("{w}x{h}"),
        "seconds": seconds,
        "quality": echo_str(job, "quality").unwrap_or_else(|| "standard".into()),
        "url": null,
        "remixed_from_video_id": null,
        "expires_at": job.expires_at.unix_timestamp(),
        "file_path": null,
        "file_name": art.map(|a| a.file_name.clone()),
        "media_type": art.map_or_else(|| "video/mp4".to_owned(), |a| a.mime.clone()),
        "completed_at": job.completed_at.map(|t| t.unix_timestamp()),
        "error": error,
        "peak_memory_mb": opt_f64(job.metrics.peak_memory_mb),
        "inference_time_s": opt_f64(job.metrics.inference_s),
        "stage_durations": job.metrics.stage_durations,
    });
    if let (Some(m), Value::Object(o)) = (metadata(job), &mut v) {
        o.insert("metadata".into(), m);
    }
    v
}

/// The finished MP4 of `job`: a local file, or a redirect to a presigned URL.
pub fn artifact_reply(job: &Job, cx: &ViewCtx) -> Option<HttpReply> {
    let a = job.artifacts.first()?;
    Some(match &a.location {
        ArtifactLocation::Local(p) => HttpReply::file(200, p.clone(), a.mime.clone()),
        ArtifactLocation::Object { .. } => {
            HttpReply::empty(302).with_header("location", cx.urls.url_for(a, URL_TTL).to_string())
        }
    })
}

/// `GET /v1/videos/{id}` and `.../content` rendering.
#[derive(Clone, Copy, Debug, Default)]
pub struct VideosView;

impl JobView for VideosView {
    fn status_reply(&self, job: &Job, cx: &ViewCtx) -> HttpReply {
        HttpReply::json(200, video_response(job, cx))
    }
    /// The content of `variant=video`: 422 when failed, 404 while in progress.
    fn result_reply(&self, job: &Job, cx: &ViewCtx) -> HttpReply {
        match &job.state {
            fastvideo_protocol::JobState::Failed(e) => openai_error_status(
                422,
                &ApiError::engine_failed(format!("Generation failed: {}", e.message)),
            ),
            fastvideo_protocol::JobState::Cancelled => openai_error_status(
                422,
                &ApiError::cancelled("Generation failed: the generation was cancelled"),
            ),
            fastvideo_protocol::JobState::Succeeded => {
                artifact_reply(job, cx).unwrap_or_else(|| {
                    openai_error_status(404, &ApiError::not_found("Generated video file not found"))
                })
            }
            _ => openai_error_status(404, &ApiError::not_found("Generation is still in-progress")),
        }
    }
}

// ---------------------------------------------------------------- handlers

/// Reads the body by content type into one JSON object: JSON, multipart
/// (`input_reference` upload -> a data URI: `video_reference` when it is a
/// video, `input_reference` otherwise) or form-urlencoded. Forms are coerced
/// ([`coerce_form`]); `extra_body` / `extra_json` are merged.
pub async fn read_body(req: Request, limit: usize) -> Result<Map<String, Value>, ApiError> {
    let ct = req
        .headers()
        .get(header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("")
        .to_ascii_lowercase();
    let (parts, body) = req.into_parts();
    let body = axum::body::to_bytes(body, limit)
        .await
        .map_err(|e| ApiError::payload_too_large(format!("request body: {e}")))?;
    let mut map = if ct.starts_with("multipart/form-data") {
        let req = Request::from_parts(parts, axum::body::Body::from(body));
        let mp = Multipart::from_request(req, &())
            .await
            .map_err(|e| ApiError::invalid(format!("invalid multipart body: {e}")))?;
        let mut m = read_multipart(mp).await?;
        coerce_form(&mut m)?;
        m
    } else if ct.starts_with("application/x-www-form-urlencoded") {
        let mut m: Map<String, Value> = url::form_urlencoded::parse(&body)
            .map(|(k, v)| (k.into_owned(), Value::String(v.into_owned())))
            .collect();
        coerce_form(&mut m)?;
        m
    } else {
        json_object(&body)?
    };
    merge_extra(&mut map)?;
    Ok(map)
}

fn json_object(body: &Bytes) -> Result<Map<String, Value>, ApiError> {
    if body.iter().all(u8::is_ascii_whitespace) {
        return Ok(Map::new());
    }
    match serde_json::from_slice(body)
        .map_err(|e| ApiError::invalid(format!("invalid JSON body: {e}")))?
    {
        Value::Object(m) => Ok(m),
        _ => Err(ApiError::invalid("the request body must be a JSON object")),
    }
}

async fn read_multipart(mut mp: Multipart) -> Result<Map<String, Value>, ApiError> {
    let bad = |e: axum::extract::multipart::MultipartError| {
        ApiError::invalid(format!("invalid multipart body: {e}"))
    };
    let mut m = Map::new();
    while let Some(field) = mp.next_field().await.map_err(bad)? {
        let name = field.name().unwrap_or("").to_owned();
        let is_file = field.file_name().is_some();
        let declared = field.content_type().map(str::to_owned);
        if !is_file {
            let text = field.text().await.map_err(bad)?;
            m.insert(name, Value::String(text));
            continue;
        }
        if name != "input_reference" {
            return Err(ApiError::invalid_param(
                name.clone(),
                format!("`{name}` cannot be a file upload"),
            ));
        }
        let data = field.bytes().await.map_err(bad)?;
        let mime = declared
            .filter(|t| !t.is_empty() && t != "application/octet-stream")
            .or_else(|| fastvideo_serve_kit::ingest::sniff_mime(&data).map(str::to_owned))
            .unwrap_or_else(|| "application/octet-stream".into());
        let uri = format!(
            "data:{mime};base64,{}",
            base64::engine::general_purpose::STANDARD.encode(&data)
        );
        if mime.starts_with("video/") {
            let item = json!({ "video_url": uri });
            match m.get_mut("video_reference") {
                Some(Value::Array(a)) => a.push(item),
                _ => {
                    m.insert("video_reference".into(), Value::Array(vec![item]));
                }
            }
        } else {
            m.insert("input_reference".into(), Value::String(uri));
        }
    }
    Ok(m)
}

fn request_echo(b: &VideoGenerationRequest) -> Value {
    let mut e = Map::new();
    e.insert("model".into(), b.model.clone().into());
    e.insert("prompt".into(), b.prompt.clone().into());
    if let Some(s) = &b.seconds {
        e.insert("seconds".into(), s.as_string().into());
    }
    if let Some(s) = &b.size {
        e.insert("size".into(), s.clone().into());
    }
    if let Some(q) = &b.quality {
        e.insert("quality".into(), q.clone().into());
    }
    Value::Object(e)
}

/// Create (and sync): body -> typed request -> model -> normalize ->
/// `apply_model` -> serve-kit `submit_request`.
async fn create(
    ctx: &ServeCtx,
    cfg: &VideosConfig,
    req: Request,
    owner: Option<KeyId>,
) -> Result<Job, ApiError> {
    ctx.engine().admit()?;
    let map = read_body(req, ctx.config().body_max_bytes).await?;
    let mut body = parse_request(map)?;
    let models = ctx.engine().models();
    let name = match body.model.clone().filter(|m| !m.is_empty()) {
        Some(m) => m,
        None => default_model(&models, cfg.default_model.as_deref())
            .ok_or_else(|| ApiError::invalid_param("model", "no model is served here"))?,
    };
    body.model = Some(name.clone());
    let caps = resolve_public(ctx.engine().as_ref(), &name)?;
    let explicit_task = body.task.is_some();
    let echo = request_echo(&body);
    let mut ncx = NormalizeCtx::new(ctx.now());
    ncx.owner = owner.clone();
    let mut gen = VideosCreate.normalize(body, &ncx)?;
    apply_model(&mut gen, &caps, explicit_task)?;
    submit_request(ctx, &OpenAiVideos, gen, owner, echo, &cfg.ingest).await
}

/// `POST /v1/videos/sync`: waits, then answers the MP4 bytes with
/// `X-Request-Id`, `X-Model`, `X-Inference-Time-S`, `X-Stage-Durations`,
/// `X-Peak-Memory-MB` (and `X-FV-Tier` / `X-FV-Recipe` when known). The job
/// and its file are removed after reading, as FastVideo removes its
/// temporary MP4. Artifacts in an object store (S3/R2) are read back through
/// `ArtifactStore::open` and streamed, not redirected.
async fn sync_reply(ctx: &ServeCtx, job: Job) -> Result<HttpReply, ApiError> {
    let job = wait_terminal(ctx, job.id, ctx.config().sync_timeout)
        .await
        .unwrap_or(job);
    if !job.is_terminal() {
        let _ = cancel_job(ctx, job.id).await;
        ctx.jobs().remove(job.id).await;
        return Err(ApiError::timeout("the generation did not finish in time"));
    }
    let cx = ctx.view_ctx(false);
    let reply = match &job.state {
        fastvideo_protocol::JobState::Succeeded => {
            match job.artifacts.first().map(|a| (&a.location, a)) {
                Some((ArtifactLocation::Local(p), a)) => {
                    let data = tokio::fs::read(p)
                        .await
                        .map_err(|e| ApiError::internal(format!("reading the output: {e}")))?;
                    HttpReply::bytes(200, a.mime.clone(), data)
                }
                // S3/R2 artifacts are read back and streamed, so the reply
                // is `video/mp4` with the `X-*` headers (design §4.1), as
                // the LTX v1 sync path does; a store that cannot read
                // objects back falls back to the redirect.
                Some((_, a)) => match ctx.artifacts().open(a).await {
                    Ok(ArtifactBody::Bytes(b)) => HttpReply::bytes(200, a.mime.clone(), b),
                    Ok(ArtifactBody::File(p)) => {
                        let data = tokio::fs::read(&p)
                            .await
                            .map_err(|e| ApiError::internal(format!("reading the output: {e}")))?;
                        HttpReply::bytes(200, a.mime.clone(), data)
                    }
                    Err(e) => {
                        tracing::warn!(error = %e.message, "/v1/videos/sync: reading the artifact back failed; answering a redirect");
                        artifact_reply(&job, &cx).unwrap_or_else(|| HttpReply::empty(500))
                    }
                },
                None => return Err(ApiError::internal("the generation produced no file")),
            }
        }
        fastvideo_protocol::JobState::Failed(e) => {
            let e = ApiError::engine_failed(format!("Generation failed: {}", e.message));
            ctx.jobs().remove(job.id).await;
            return Err(e);
        }
        _ => {
            ctx.jobs().remove(job.id).await;
            return Err(ApiError::engine_failed(
                "Generation failed: the generation was cancelled",
            ));
        }
    };
    let mut reply = reply
        .with_header("x-request-id", job.external_id.clone())
        .with_header("x-model", job.requested_model().to_owned());
    let m = &job.metrics;
    if let Some(s) = m.inference_s {
        reply.push_header("x-inference-time-s", format!("{s:.3}"));
    }
    reply.push_header(
        "x-stage-durations",
        serde_json::to_string(&m.stage_durations).unwrap_or_else(|_| "{}".into()),
    );
    if let Some(p) = m.peak_memory_mb {
        reply.push_header("x-peak-memory-mb", format!("{p:.1}"));
    }
    if let Some(t) = job.resolved.tier {
        reply.push_header("x-fv-tier", t.as_str());
    }
    if let Some(r) = &job.resolved.recipe {
        reply.push_header("x-fv-recipe", r.clone());
    }
    if reply.status == 200 {
        ctx.jobs().remove(job.id).await;
    }
    Ok(reply)
}

/// `GET /v1/videos` query.
#[derive(Clone, Debug, Default, Deserialize)]
pub struct ListParams {
    pub after: Option<String>,
    pub limit: Option<String>,
    pub order: Option<String>,
}

/// Parses the list query: `limit` 1..=100 (default 20), `order` asc|desc
/// (default desc).
pub fn list_query(p: &ListParams) -> Result<ListQuery, ApiError> {
    let limit = match &p.limit {
        None => 20,
        Some(s) => match s.parse::<usize>() {
            Ok(n) if (1..=100).contains(&n) => n,
            _ => {
                return Err(ApiError::invalid_param(
                    "limit",
                    "limit must be within 1..=100",
                ))
            }
        },
    };
    let order = match p.order.as_deref() {
        None | Some("desc") => SortOrder::Desc,
        Some("asc") => SortOrder::Asc,
        Some(_) => {
            return Err(ApiError::invalid_param(
                "order",
                "order must be asc or desc",
            ))
        }
    };
    Ok(ListQuery {
        protocol: Some(ProtocolId::OpenAiVideos),
        order,
        after: p.after.clone().filter(|a| !a.is_empty()),
        limit,
        ..ListQuery::default()
    })
}

/// `{object:"list", data, first_id, last_id, has_more}`.
pub fn list_body(jobs: &[Job], has_more: bool, cx: &ViewCtx) -> Value {
    let data: Vec<Value> = jobs.iter().map(|j| video_response(j, cx)).collect();
    json!({
        "object": "list",
        "data": data,
        "first_id": jobs.first().map(|j| j.external_id.clone()),
        "last_id": jobs.last().map(|j| j.external_id.clone()),
        "has_more": has_more,
    })
}

/// Lists the caller's jobs: exactly the jobs whose owner equals the caller
/// (anonymous callers see anonymous jobs).
async fn list(ctx: &ServeCtx, owner: Option<KeyId>, p: &ListParams) -> Result<HttpReply, ApiError> {
    let q = list_query(p)?;
    let all = ctx
        .jobs()
        .list(ListQuery {
            protocol: Some(ProtocolId::OpenAiVideos),
            owner: owner.clone(),
            limit: usize::MAX,
            ..ListQuery::default()
        })
        .await;
    let mine: Vec<&Job> = all.items.iter().filter(|j| j.owner == owner).collect();
    let page = q.apply(mine);
    Ok(HttpReply::json(
        200,
        list_body(&page.items, page.has_more, &ctx.view_ctx(false)),
    ))
}

/// `DELETE`: cancels a job still queued or running (our observer can stop a
/// running generation, FastVideo cannot), then removes it and its file.
pub async fn delete_job(ctx: &ServeCtx, job: &Job) {
    if !job.is_terminal() {
        let _ = cancel_job(ctx, job.id).await;
    }
    ctx.jobs().remove(job.id).await;
}

fn ecx(route: &str, ext: Option<&str>) -> ErrorCtx {
    ErrorCtx {
        request_id: Some(fastvideo_serve_kit::random_token()),
        route: Some(route.to_owned()),
        external_id: ext.map(str::to_owned),
    }
}

async fn respond(ctx: &ServeCtx, r: Result<HttpReply, ApiError>, cx: ErrorCtx) -> Response {
    let reply = r.unwrap_or_else(|e| OpenAiVideos.render_error(&e, &cx));
    into_response(reply, ctx, None).await
}

fn auth(ctx: &ServeCtx, headers: &HeaderMap) -> Result<Option<KeyId>, ApiError> {
    ctx.auth().authenticate(ProtocolId::OpenAiVideos, headers)
}

/// The `/v1/videos*` routes.
pub fn routes(cfg: Arc<VideosConfig>) -> Router<ServeCtx> {
    let proto = Arc::new(OpenAiVideos);
    let view = Arc::new(VideosView);
    let submit = |wait: bool| {
        let cfg = cfg.clone();
        post(move |State(ctx): State<ServeCtx>, req: Request| {
            let cfg = cfg.clone();
            async move {
                let route = if wait {
                    "/v1/videos/sync"
                } else {
                    "/v1/videos"
                };
                let r = async {
                    let owner = auth(&ctx, req.headers())?;
                    let job = create(&ctx, &cfg, req, owner).await?;
                    if wait {
                        sync_reply(&ctx, job).await
                    } else {
                        Ok(VideosCreate.submit_reply(&job, &ctx.view_ctx(false)))
                    }
                }
                .await;
                respond(&ctx, r, ecx(route, None)).await
            }
        })
        .layer(DefaultBodyLimit::disable())
    };
    Router::new()
        .route("/v1/videos", submit(false).get(
            |State(ctx): State<ServeCtx>, Query(p): Query<ListParams>, headers: HeaderMap| async move {
                let r = async { list(&ctx, auth(&ctx, &headers)?, &p).await }.await;
                respond(&ctx, r, ecx("/v1/videos", None)).await
            },
        ))
        .route("/v1/videos/generations", submit(false))
        .route("/v1/videos/sync", submit(true))
        .route(
            "/v1/videos/{id}",
            handlers::status(proto.clone(), view.clone(), "id").delete(
                |State(ctx): State<ServeCtx>, Path(id): Path<String>, headers: HeaderMap| async move {
                    let r = async {
                        let owner = auth(&ctx, &headers)?;
                        let job = find_job(&ctx, &OpenAiVideos, &id, owner.as_ref()).await?;
                        delete_job(&ctx, &job).await;
                        Ok(HttpReply::json(200, json!({ "id": id, "deleted": true, "object": "video.deleted" })))
                    }
                    .await;
                    respond(&ctx, r, ecx("/v1/videos/{id}", Some(&id))).await
                },
            ),
        )
        .route(
            "/v1/videos/{id}/content",
            get(
                |State(ctx): State<ServeCtx>,
                 Path(id): Path<String>,
                 Query(q): Query<HashMap<String, String>>,
                 headers: HeaderMap| async move {
                    let r = async {
                        let owner = auth(&ctx, &headers)?;
                        if let Some(v) = q.get("variant").filter(|v| v.as_str() != "video") {
                            return Err(ApiError::invalid_param(
                                "variant",
                                format!("variant `{v}` is not supported; only `video` is available"),
                            ));
                        }
                        let job = find_job(&ctx, &OpenAiVideos, &id, owner.as_ref()).await?;
                        Ok(VideosView.result_reply(&job, &ctx.view_ctx(false)))
                    }
                    .await;
                    respond(&ctx, r, ecx("/v1/videos/{id}/content", Some(&id))).await
                },
            ),
        )
}
