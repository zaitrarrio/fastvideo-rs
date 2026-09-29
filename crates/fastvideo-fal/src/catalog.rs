//! Input JSON Schemas of the app endpoints and the catalog of mounted apps
//! (console, WP-20). Not a fal route: fal publishes its schemas as OpenAPI
//! on its own site; these are ours, served for the console forms.
//!
//! | Route | Behaviour |
//! |---|---|
//! | `GET /fal/schema` | `{apps:[{id, model, tier, kind, director, endpoints:[{sub, endpoint_id, title}]}]}` for the configured apps; an endpoint whose task the app's model does not serve here is left out |
//! | `GET /fal/schema/{owner}/{alias}/{*sub}` | The endpoint's input JSON Schema (draft 2020-12) as served here; `sub` may have several segments (`text-to-video/fast`) |
//! | `GET /fal/schema/{owner}/{alias}/director` | The director form: the `configure` resolutions and aspect ratios the app's model serves |
//!
//! **Served schemas** ([`served_schema`]). fal's schema lists every value
//! fal's own hosts take; the one served here (the console's form) is
//! narrowed to what the endpoint's model serves, from its caps: the
//! `resolution` tiers (h3-draft: 480P only; `minimax/h3` without 2K / 4K; a
//! GPU below the 1080P tier's memory plan without 1080P), the LTX rates and
//! durations (no `auto`: the duration head is not loaded), and
//! frame ranges within the model's grid, with defaults moved onto a served
//! value. The parsers still take fal's full lists: a value outside the
//! served schema is refused at submit with the values that are served.
//!
//! The schema is built from the same constants [`FalInput::parse_for`]
//! enforces (the H3 `PROMPT_MAX_CHARS`, `DURATION_MIN/MAX`, `Resolution` and
//! `AspectRatio` enums and reference limits; the [`ltx`] and [`wan`]
//! constants), and the tests check both agree, so the console form cannot
//! drift from validation. Property order follows fal's
//! `x-fal-order-properties`. Vendor keys for the console: `x-fv-media`
//! (`image` / `video` / `audio`: an uploadable URL field), `x-fv-advanced`
//! (shown under "Additional settings") and `x-fv-multiline`.
//!
//! [`FalInput::parse_for`]: crate::schema::FalInput::parse_for

use std::sync::Arc;

use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::get;
use axum::{Json, Router};
use fastvideo_protocol::{route_task, ModelCaps, Task};
use fastvideo_serve_kit::ServeCtx;
use serde_json::{json, Map, Value};

use crate::schema::{a2v, ingredient};
use crate::schema::ltx::{self, LtxAspect, LtxClass, CAMERA_MOTIONS, LTX_PROMPT_MAX_CHARS};
use crate::schema::wan::{self, WanAspect, WanResolution, WanVariant, INTERPOLATORS};
use crate::schema::{
    AppKind, AspectRatio, Endpoint, Resolution, DURATION_MAX, DURATION_MIN, MAX_REFERENCES, MAX_REFERENCE_AUDIO,
    MAX_REFERENCE_IMAGES, MAX_REFERENCE_VIDEOS, PROMPT_MAX_CHARS,
};
use crate::{FalApp, FalConfig};

fn media_url(kind: &str, description: &str) -> Value {
    json!({
        "anyOf": [{"type": "string", "minLength": 1, "pattern": "\\S"}, {"type": "null"}],
        "default": null,
        "description": description,
        "x-fv-media": kind,
    })
}

fn required_media_url(kind: &str, description: &str) -> Value {
    json!({"type": "string", "minLength": 1, "pattern": "\\S", "description": description, "x-fv-media": kind})
}

fn media_list(kind: &str, max: usize, description: &str) -> Value {
    json!({
        "type": "array",
        "items": {"type": "string", "minLength": 1, "pattern": "\\S"},
        "maxItems": max,
        "default": [],
        "description": description,
        "x-fv-media": kind,
    })
}

fn prompt(max: usize, description: &str) -> Value {
    json!({"type": "string", "minLength": 1, "maxLength": max, "description": description, "x-fv-multiline": true})
}

fn seed() -> Value {
    json!({"anyOf": [{"type": "integer", "minimum": 0}, {"type": "null"}], "default": null, "description": "Random seed. A random seed is selected when omitted.", "x-fv-advanced": true})
}

fn sync_mode() -> Value {
    json!({"type": "boolean", "default": false, "description": "Return the generated video as base64 instead of a CDN URL.", "x-fv-advanced": true})
}

fn safety_checker() -> Value {
    json!({"type": "boolean", "default": true, "description": "If set to true, the safety checker will be enabled. Accepted; this server has no checker.", "x-fv-advanced": true})
}

fn strings<T>(list: &[T], name: fn(&T) -> &'static str) -> Vec<&'static str> {
    list.iter().map(name).collect()
}

const TARGET_AUDIO: &str = "Optional URL of an audio clip at least 2 seconds long (maximum 15 MB) to pin to the generated soundtrack. Accepts an HTTP(S) URL or a base64 data URI.";

fn object(title: &str, props: Map<String, Value>, order: Vec<&str>, required: &[&str]) -> Value {
    debug_assert_eq!(order.len(), props.len());
    json!({
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": title,
        "type": "object",
        "properties": Value::Object(props),
        "required": required,
        "x-fal-order-properties": order,
    })
}

/// The input JSON Schema of `endpoint` on an H3 Max app (for every
/// endpoint of the LTX and Wan apps, their own schema).
pub fn input_schema(endpoint: Endpoint) -> Value {
    input_schema_for(AppKind::H3, endpoint)
}

/// The input JSON Schema of `endpoint` on an app of `kind`.
pub fn input_schema_for(kind: AppKind, endpoint: Endpoint) -> Value {
    match endpoint {
        Endpoint::LtxTextToVideoFast => ltx_schema(endpoint, LtxClass::Fast, false),
        Endpoint::LtxTextToVideoPro => ltx_schema(endpoint, LtxClass::Pro, false),
        Endpoint::LtxImageToVideoFast => ltx_schema(endpoint, LtxClass::Fast, true),
        Endpoint::LtxImageToVideoPro => ltx_schema(endpoint, LtxClass::Pro, true),
        Endpoint::WanTextToVideo => wan_schema(endpoint, WanVariant::TextToVideo),
        Endpoint::WanImageToVideo => wan_schema(endpoint, WanVariant::ImageToVideo),
        Endpoint::WanFastWan => wan_schema(endpoint, WanVariant::FastWan),
        Endpoint::LtxIngredient => ingredient_schema(endpoint),
        Endpoint::LtxAudioToVideoFast => a2v_schema(endpoint, LtxClass::Fast),
        Endpoint::LtxAudioToVideoPro => a2v_schema(endpoint, LtxClass::Pro),
        Endpoint::TextToVideo | Endpoint::ImageToVideo | Endpoint::ReferenceToVideo => h3_schema(kind, endpoint),
    }
}

fn h3_schema(kind: AppKind, endpoint: Endpoint) -> Value {
    let mut props = Map::new();
    let prompt_desc = match endpoint {
        Endpoint::ReferenceToVideo => "Text prompt for video generation. Refer to reference assets by their modality and order in the reference lists: Image 1, Image 2, Video 1, Audio 1, and so on.",
        _ => "Text prompt for video generation",
    };
    props.insert("prompt".into(), prompt(PROMPT_MAX_CHARS, prompt_desc));
    props.insert(
        "duration".into(),
        json!({"type": "integer", "minimum": DURATION_MIN, "maximum": DURATION_MAX, "default": DURATION_MIN, "description": "The duration of the video in seconds."}),
    );
    let res_desc = match kind {
        AppKind::H3Base => "The generation resolution. 480P and 768P are native; 2K and 4K (fal: upscaled from 768P) are not available on this server.",
        _ => "The generation resolution. 480P and 768P are native; 1080P is generated natively at 1920x1088 and cropped to 1080 (about 2.5x the time of 768P; served on 80 GB-class GPUs).",
    };
    props.insert(
        "resolution".into(),
        json!({
            "type": "string",
            "enum": strings(kind.h3_resolutions(), Resolution::as_str),
            "default": Resolution::P768.as_str(),
            "description": res_desc,
        }),
    );
    props.insert("seed".into(), seed());
    props.insert("enable_safety_checker".into(), safety_checker());
    props.insert("sync_mode".into(), sync_mode());
    let expansion = match kind {
        AppKind::H3Base => json!(["disabled", "fast", "balanced", "quality"]),
        _ => json!(["disabled", "balanced", "quality"]),
    };
    props.insert(
        "prompt_expansion_mode".into(),
        json!({"type": "string", "default": "balanced", "examples": expansion, "description": "How much effort to spend rewriting the prompt before generation. Accepted; this server does not expand prompts.", "x-fv-advanced": true}),
    );
    let mut extra = Vec::<&str>::new();
    match endpoint {
        Endpoint::TextToVideo => {
            props.insert("target_audio_url".into(), {
                let mut v = media_url("audio", TARGET_AUDIO);
                v["x-fv-advanced"] = true.into();
                v
            });
            props.insert(
                "aspect_ratio".into(),
                json!({"type": "string", "enum": strings(&AspectRatio::T2V, AspectRatio::as_str), "default": AspectRatio::R16x9.as_str(), "description": "The aspect ratio of the generated video."}),
            );
            extra.extend(["target_audio_url", "aspect_ratio"]);
        }
        Endpoint::ImageToVideo => {
            props.insert("target_audio_url".into(), {
                let mut v = media_url("audio", TARGET_AUDIO);
                v["x-fv-advanced"] = true.into();
                v
            });
            props.insert(
                "image_url".into(),
                media_url("image", "Optional URL of the image to use as the first frame. When provided, the output canvas follows this image. If both images are omitted, the request is handled as text-to-video (16:9 by default)."),
            );
            props.insert(
                "end_image_url".into(),
                media_url("image", "Optional URL of the image to use as the last frame. It may be provided alone for end-only keyframe generation; in that case the output canvas follows this image."),
            );
            extra.extend(["target_audio_url", "image_url", "end_image_url"]);
        }
        _ => {
            props.insert(
                "aspect_ratio".into(),
                json!({"type": "string", "enum": strings(&AspectRatio::R2V, AspectRatio::as_str), "default": AspectRatio::Adaptive.as_str(), "description": "The aspect ratio of the generated video."}),
            );
            props.insert(
                "reference_image_urls".into(),
                media_list("image", MAX_REFERENCE_IMAGES, "URLs of subject/style reference images, referenced in the prompt as Image 1, Image 2, and so on."),
            );
            props.insert(
                "reference_video_urls".into(),
                media_list("video", MAX_REFERENCE_VIDEOS, "URLs of motion/reference video clips (2-15 seconds each), referenced in the prompt as Video 1, Video 2, and so on."),
            );
            props.insert(
                "reference_audio_urls".into(),
                media_list("audio", MAX_REFERENCE_AUDIO, "URLs of reference audio clips (2-15 seconds each), referenced in the prompt as Audio 1, Audio 2, and so on."),
            );
            extra.extend(["aspect_ratio", "reference_image_urls", "reference_video_urls", "reference_audio_urls"]);
        }
    }
    let mut order: Vec<&str> =
        vec!["prompt", "duration", "resolution", "seed", "enable_safety_checker", "sync_mode", "prompt_expansion_mode"];
    order.extend(extra);
    let mut schema = object(endpoint.title(), props, order, &["prompt"]);
    if endpoint == Endpoint::ReferenceToVideo {
        schema["x-fv-max-references"] = MAX_REFERENCES.into();
        schema["x-fv-min-references"] = 1.into();
    }
    schema
}

fn ltx_schema(endpoint: Endpoint, class: LtxClass, i2v: bool) -> Value {
    let mut props = Map::new();
    let mut order = vec!["prompt"];
    props.insert("prompt".into(), prompt(LTX_PROMPT_MAX_CHARS, "The prompt to generate the video from."));
    if i2v {
        props.insert(
            "image_url".into(),
            required_media_url("image", "URL of the image to use as the first frame. With aspect_ratio `auto` the output follows its aspect."),
        );
        props.insert(
            "end_image_url".into(),
            media_url("image", "Optional URL of the image to use as the last frame (a first-to-last transition)."),
        );
        order.extend(["image_url", "end_image_url"]);
    }
    let mut durations: Vec<Value> = class.durations().into_iter().map(Value::from).collect();
    durations.push("auto".into());
    let max_note = match class {
        LtxClass::Fast => "Over 10 s only at 720p/1080p and 24 or 25 fps.",
        LtxClass::Pro => "",
    };
    props.insert(
        "duration".into(),
        json!({
            "enum": durations,
            "default": ltx::LTX_DEFAULT_DURATION,
            "description": format!(
                "The duration of the generated video in seconds. {max_note} This server generates at most {} frames (so 20 s needs 24 fps). `auto` needs the LTX-2.5 duration head, which this server does not load yet.",
                ltx::LTX_FRAMES_MAX
            ).replace("  ", " "),
        }),
    );
    props.insert(
        "resolution".into(),
        json!({"type": "string", "enum": strings(class.resolutions(), ltx::LtxResolution::as_str), "default": "1080p", "description": "The resolution of the generated video."}),
    );
    let (aspects, default_aspect): (&[LtxAspect], _) =
        if i2v { (&LtxAspect::I2V, LtxAspect::Auto) } else { (&LtxAspect::T2V, LtxAspect::R16x9) };
    props.insert(
        "aspect_ratio".into(),
        json!({"type": "string", "enum": strings(aspects, LtxAspect::as_str), "default": default_aspect.as_str(), "description": if i2v { "The aspect ratio of the generated video; `auto` follows the input image." } else { "The aspect ratio of the generated video." }}),
    );
    props.insert(
        "fps".into(),
        json!({"type": "integer", "enum": class.fps(), "default": ltx::LTX_DEFAULT_FPS, "description": "The frame rate of the generated video."}),
    );
    props.insert(
        "generate_audio".into(),
        json!({"type": "boolean", "default": true, "description": "Whether to generate audio for the video."}),
    );
    props.insert(
        "camera_motion".into(),
        json!({"anyOf": [{"type": "string", "enum": CAMERA_MOTIONS}, {"type": "null"}], "default": null, "description": "Camera motion. `static` is accepted (a no-op); the other motions need LTX camera LoRAs this server does not have.", "x-fv-advanced": true}),
    );
    props.insert("seed".into(), seed());
    props.insert("sync_mode".into(), sync_mode());
    order.extend(["duration", "resolution", "aspect_ratio", "fps", "generate_audio", "camera_motion", "seed", "sync_mode"]);
    let required: &[&str] = if i2v { &["prompt", "image_url"] } else { &["prompt"] };
    object(endpoint.title(), props, order, required)
}

fn wan_schema(endpoint: Endpoint, variant: WanVariant) -> Value {
    let mut props = Map::new();
    let mut order = vec!["prompt"];
    props.insert("prompt".into(), prompt(PROMPT_MAX_CHARS, "The text prompt to guide video generation."));
    if variant == WanVariant::ImageToVideo {
        props.insert(
            "image_url".into(),
            required_media_url("image", "URL of the image to use as the first frame. With aspect_ratio `auto` the output follows its aspect."),
        );
        order.push("image_url");
    }
    props.insert(
        "num_frames".into(),
        json!({"type": "integer", "minimum": wan::WAN_FRAMES_MIN, "maximum": wan::WAN_FRAMES_MAX, "default": wan::WAN_FRAMES_DEFAULT, "description": "Number of frames to generate (rounded up to 4k+1)."}),
    );
    props.insert(
        "frames_per_second".into(),
        json!({"type": "integer", "minimum": wan::WAN_FPS_MIN, "maximum": wan::WAN_FPS_MAX, "default": wan::WAN_FPS_DEFAULT, "description": "Frame rate of the output video (the frames are the same at every rate)."}),
    );
    props.insert(
        "resolution".into(),
        json!({"type": "string", "enum": strings(variant.resolutions(), WanResolution::as_str), "default": WanResolution::P720.as_str(), "description": "Resolution of the generated video (580p: 576 short edge; 720p: 704, the model's trained size)."}),
    );
    props.insert(
        "aspect_ratio".into(),
        json!({"type": "string", "enum": strings(variant.aspects(), WanAspect::as_str), "default": variant.default_aspect().as_str(), "description": "Aspect ratio of the generated video."}),
    );
    order.extend(["num_frames", "frames_per_second", "resolution", "aspect_ratio"]);
    let distilled = variant.distilled();
    props.insert(
        "negative_prompt".into(),
        json!({"type": "string", "default": "", "description": if distilled { "Negative prompt. Accepted; the distilled model runs one unguided pass." } else { "Negative prompt for video generation (empty: the model's default)." }, "x-fv-advanced": true, "x-fv-multiline": true}),
    );
    order.push("negative_prompt");
    if !distilled {
        props.insert(
            "num_inference_steps".into(),
            json!({"type": "integer", "minimum": wan::WAN_STEPS_MIN, "maximum": wan::WAN_STEPS_MAX, "default": wan::WAN_STEPS_DEFAULT, "description": "Number of inference steps.", "x-fv-advanced": true}),
        );
        order.push("num_inference_steps");
    }
    let (glo, ghi, gdef) = wan::WAN_GUIDANCE;
    props.insert(
        "guidance_scale".into(),
        json!({"type": "number", "minimum": glo, "maximum": ghi, "default": gdef, "description": if distilled { "Classifier-free guidance scale. Accepted; the distilled model is unguided." } else { "Classifier-free guidance scale." }, "x-fv-advanced": true}),
    );
    order.push("guidance_scale");
    if !distilled {
        let (slo, shi, sdef) = wan::WAN_SHIFT;
        props.insert(
            "shift".into(),
            json!({"type": "number", "minimum": slo, "maximum": shi, "default": sdef, "description": "Flow-matching shift.", "x-fv-advanced": true}),
        );
        order.push("shift");
    }
    props.insert(
        "interpolator_model".into(),
        json!({"type": "string", "enum": INTERPOLATORS, "default": "none", "description": "Frame interpolator. This server has none: `film` and `rife` need num_interpolated_frames 0.", "x-fv-advanced": true}),
    );
    props.insert(
        "num_interpolated_frames".into(),
        json!({"type": "integer", "minimum": 0, "maximum": wan::WAN_INTERPOLATED_MAX, "default": 0, "description": "Frames interpolated between each generated pair (0 only on this server).", "x-fv-advanced": true}),
    );
    props.insert(
        "enable_prompt_expansion".into(),
        json!({"type": "boolean", "default": false, "description": "Expand the prompt with an LLM. Accepted; this server does not expand prompts.", "x-fv-advanced": true}),
    );
    props.insert("seed".into(), seed());
    props.insert("enable_safety_checker".into(), safety_checker());
    props.insert("sync_mode".into(), sync_mode());
    order.extend([
        "interpolator_model",
        "num_interpolated_frames",
        "enable_prompt_expansion",
        "seed",
        "enable_safety_checker",
        "sync_mode",
    ]);
    let required: &[&str] = if variant == WanVariant::ImageToVideo { &["prompt", "image_url"] } else { &["prompt"] };
    object(endpoint.title(), props, order, required)
}

fn ingredient_schema(endpoint: Endpoint) -> Value {
    let mut props = Map::new();
    props.insert(
        "prompt".into(),
        prompt(
            LTX_PROMPT_MAX_CHARS,
            "The prompt in two parts: \"Reference sheet: <the panels: characters, props, location> Generated video: <the shot and action>\".",
        ),
    );
    props.insert(
        "image_url".into(),
        required_media_url(
            "image",
            "URL of the reference sheet: one composite image with a clean panel per character (face close-up and turnaround), prop and location, on a black background.",
        ),
    );
    let (lo, hi, def) = ingredient::INGREDIENT_STRENGTH;
    props.insert(
        "ingredient_strength".into(),
        json!({"type": "number", "minimum": lo, "maximum": hi, "default": def, "description": "Strength of the Ingredients IC-LoRA (1: the trained strength)."}),
    );
    props.insert(
        "reference_strength".into(),
        json!({"type": "number", "minimum": lo, "maximum": hi, "default": def, "description": "How strongly the reference sheet is held (1: kept clean). This server accepts 0 to 1."}),
    );
    props.insert(
        "num_frames".into(),
        json!({"type": "integer", "minimum": ingredient::INGREDIENT_FRAMES_MIN, "maximum": ingredient::INGREDIENT_FRAMES_MAX, "default": ingredient::INGREDIENT_FRAMES_DEFAULT, "description": "Number of frames to generate (rounded up to 8k+1). This server generates at most 241 frames with a reference."}),
    );
    props.insert(
        "frames_per_second".into(),
        json!({"type": "integer", "minimum": ingredient::INGREDIENT_FPS_MIN, "maximum": ingredient::INGREDIENT_FPS_MAX, "default": ingredient::INGREDIENT_FPS_DEFAULT, "description": "Frame rate of the generated video. This server generates at 24, 25, 48 or 50."}),
    );
    props.insert(
        "generate_audio".into(),
        json!({"type": "boolean", "default": true, "description": "Whether to generate audio for the video."}),
    );
    props.insert(
        "negative_prompt".into(),
        json!({"type": "string", "default": "", "description": "Negative prompt. Accepted; the distilled model runs one unguided pass.", "x-fv-advanced": true, "x-fv-multiline": true}),
    );
    props.insert("seed".into(), seed());
    props.insert("sync_mode".into(), sync_mode());
    let order = vec![
        "prompt",
        "image_url",
        "ingredient_strength",
        "reference_strength",
        "num_frames",
        "frames_per_second",
        "generate_audio",
        "negative_prompt",
        "seed",
        "sync_mode",
    ];
    object(endpoint.title(), props, order, &["prompt", "image_url"])
}

/// `lightricks/ltx-2.5/audio-to-video/{fast,pro}` (fal's
/// `Ltx25AudioToVideo*Input`, plus this server's `seed` and `sync_mode`).
fn a2v_schema(endpoint: Endpoint, class: LtxClass) -> Value {
    let mut props = Map::new();
    let max_s = match class {
        LtxClass::Fast => fastvideo_protocol::A2V_AUDIO_MAX_S,
        LtxClass::Pro => f64::from(a2v::A2V_PRO_MAX_S),
    };
    props.insert(
        "audio_url".into(),
        required_media_url(
            "audio",
            &format!(
                "URL of the audio file to generate a video from. Duration must be between {} and {max_s} seconds; the video follows its length and carries it as the soundtrack. An HTTP(S) URL or a base64 data URI.",
                fastvideo_protocol::A2V_AUDIO_MIN_S
            ),
        ),
    );
    props.insert(
        "image_url".into(),
        media_url("image", "URL of an image to use as the first frame of the video. If not provided, prompt is required."),
    );
    props.insert(
        "prompt".into(),
        json!({"type": "string", "minLength": 1, "maxLength": LTX_PROMPT_MAX_CHARS, "description": "Text description of how the video should be generated. Required if image_url is not provided. When image_url is provided, this describes how the image should be animated.", "x-fv-multiline": true}),
    );
    props.insert(
        "aspect_ratio".into(),
        json!({"type": "string", "enum": strings(&a2v::A2V_ASPECTS, LtxAspect::as_str), "default": LtxAspect::Auto.as_str(), "description": "The aspect ratio of the generated video. If 'auto', the aspect ratio will be determined automatically based on the input image, or defaults to 16:9 if no image is provided."}),
    );
    let (glo, ghi) = a2v::A2V_GUIDANCE;
    props.insert(
        "guidance_scale".into(),
        json!({"anyOf": [{"type": "number", "minimum": glo, "maximum": ghi}, {"type": "null"}], "default": null, "description": "Guidance scale for video generation. Accepted; the distilled model this server runs is unguided.", "x-fv-advanced": true}),
    );
    props.insert("seed".into(), seed());
    props.insert("sync_mode".into(), sync_mode());
    let order = vec!["audio_url", "image_url", "prompt", "aspect_ratio", "guidance_scale", "seed", "sync_mode"];
    object(endpoint.title(), props, order, &["audio_url"])
}

/// One catalog entry; `served` keeps the endpoints the app's models serve
/// (every endpoint when `None`).
fn app_entry(a: &FalApp, served: Option<&dyn Fn(Endpoint) -> bool>) -> Value {
    let kind = a.kind();
    json!({
        "id": a.id,
        "model": a.model,
        // The LTX and Wan apps pick a tier per endpoint.
        "tier": a.tier.filter(|_| kind.director()).map(|(_, t)| t.as_str()),
        "kind": kind,
        "director": kind.director(),
        "endpoints": a.endpoints().iter().filter(|e| served.is_none_or(|f| f(**e))).map(|e| {
            let (model, tier) = a.target(*e);
            json!({
                "sub": e.sub(),
                "endpoint_id": format!("{}/{}", a.id, e.sub()),
                "title": e.title(),
                "model": model,
                "tier": tier.map(|(_, t)| t.as_str()),
            })
        }).collect::<Vec<_>>(),
    })
}

/// `GET /fal/schema` body with every endpoint of every configured app.
pub fn catalog(cfg: &FalConfig) -> Value {
    let apps: Vec<Value> = cfg.apps.iter().filter(|a| a.is_valid()).map(|a| app_entry(a, None)).collect();
    json!({"apps": apps})
}

/// `GET /fal/schema` body: the configured apps with the endpoints this
/// server's models serve.
pub fn catalog_served(cfg: &FalConfig, models: &[ModelCaps], alias: &dyn Fn(&str) -> Option<String>) -> Value {
    let apps: Vec<Value> = cfg
        .apps
        .iter()
        .filter(|a| a.is_valid())
        .map(|a| app_entry(a, Some(&|e| endpoint_caps(a, e, models, alias).is_some())))
        .collect();
    json!({"apps": apps})
}

/// The task an endpoint's form submits (the H3 image-to-video endpoint is
/// image-to-video once an image is set).
pub fn endpoint_task(e: Endpoint) -> Task {
    match e {
        Endpoint::TextToVideo | Endpoint::LtxTextToVideoFast | Endpoint::LtxTextToVideoPro | Endpoint::WanTextToVideo | Endpoint::WanFastWan => Task::T2V,
        Endpoint::ImageToVideo | Endpoint::LtxImageToVideoFast | Endpoint::LtxImageToVideoPro | Endpoint::WanImageToVideo => Task::I2V,
        Endpoint::ReferenceToVideo | Endpoint::LtxIngredient => Task::Ref2V,
        Endpoint::LtxAudioToVideoFast | Endpoint::LtxAudioToVideoPro => Task::A2V,
    }
}

/// The caps of the model an endpoint of `app` runs on (its name through the
/// aliases, else its tier, then the task companion), when served here and
/// serving the endpoint's task.
pub fn endpoint_caps(app: &FalApp, e: Endpoint, models: &[ModelCaps], alias: &dyn Fn(&str) -> Option<String>) -> Option<ModelCaps> {
    let (name, tier) = app.target(e);
    let caps = fastvideo_protocol::resolve_model(&name, alias, models)
        .ok()
        .or_else(|| tier.and_then(|(family, tier)| fastvideo_protocol::resolve_tier(family, tier, models).ok()))?;
    let task = endpoint_task(e);
    let caps = route_task(caps, task, models);
    caps.supports(task).then(|| caps.clone())
}

/// The caps of the model an app's director runs (the app's own model).
pub fn director_caps(app: &FalApp, models: &[ModelCaps], alias: &dyn Fn(&str) -> Option<String>) -> Option<ModelCaps> {
    app.kind().director().then_some(())?;
    endpoint_caps(app, Endpoint::TextToVideo, models, alias)
}

fn narrow_enum(schema: &mut Value, field: &str, keep: impl Fn(&Value) -> bool, preferred_default: Option<Value>) {
    let Some(p) = schema["properties"].get_mut(field) else { return };
    let path = if p.get("enum").is_some() { "/enum" } else { "/anyOf/0/enum" };
    let Some(Value::Array(list)) = p.pointer(path) else { return };
    let kept: Vec<Value> = list.iter().filter(|v| keep(v)).cloned().collect();
    if kept.is_empty() || kept.len() == list.len() {
        return;
    }
    let default = p.get("default").cloned().unwrap_or(Value::Null);
    if !default.is_null() && !kept.contains(&default) {
        p["default"] = preferred_default.filter(|d| kept.contains(d)).unwrap_or_else(|| kept[0].clone());
    }
    if let Some(slot) = p.pointer_mut(path) {
        *slot = Value::Array(kept);
    }
}

fn narrow_range(schema: &mut Value, field: &str, lo: i64, hi: i64) {
    let Some(p) = schema["properties"].get_mut(field) else { return };
    let (Some(a), Some(b)) = (p["minimum"].as_i64(), p["maximum"].as_i64()) else { return };
    let (a, b) = (a.max(lo), b.min(hi));
    if a > b {
        return;
    }
    p["minimum"] = a.into();
    p["maximum"] = b.into();
    if let Some(d) = p["default"].as_i64() {
        p["default"] = d.clamp(a, b).into();
    }
}

/// The schema this server serves for `endpoint` on an app of `kind` whose
/// model has `caps`: fal's [`input_schema_for`], narrowed to the values the
/// model serves (see the module docs).
pub fn served_schema(kind: AppKind, endpoint: Endpoint, caps: &ModelCaps) -> Value {
    let mut s = input_schema_for(kind, endpoint);
    let tiers = &caps.canvas.short_edges;
    match endpoint {
        Endpoint::TextToVideo | Endpoint::ImageToVideo | Endpoint::ReferenceToVideo => {
            let served = |v: &Value| {
                Resolution::ALL.iter().chain(&Resolution::BASE).any(|r| v == r.as_str() && tiers.contains(&r.short_edge()))
            };
            // An omitted resolution is 768P, or the model's first tier.
            let default = if tiers.contains(&Resolution::P768.short_edge()) {
                Resolution::P768.as_str()
            } else {
                match tiers.first() {
                    Some(480) => Resolution::P480.as_str(),
                    Some(1080) => Resolution::P1080.as_str(),
                    _ => Resolution::P768.as_str(),
                }
            };
            narrow_enum(&mut s, "resolution", served, Some(default.into()));
            // Durations within the model's grid (whole seconds).
            let fps = caps.fps.default.max(1);
            let lo = i64::from(caps.frames.min.div_ceil(fps));
            let hi = i64::from(caps.frames.max / fps);
            narrow_range(&mut s, "duration", lo, hi);
            // The opt-in H3 1080P tier's own clip cap (5 s; 10 s with the
            // `h3_1080p_long` experimental flag): the console form narrows
            // the duration when 1080P is picked.
            if let Some(t) = caps.canvas.hd.filter(|t| tiers.contains(&t.short_edge)) {
                if let (Some(max), Some(p)) = (t.max_frames, s["properties"].get_mut("duration")) {
                    let cap = i64::from(max / fps).min(hi);
                    let res = Resolution::ALL.iter().find(|r| r.short_edge() == t.short_edge).map(|r| r.as_str());
                    if let Some(res) = res {
                        let mut by = Map::new();
                        by.insert(res.to_owned(), cap.into());
                        p["x-fv-max-by-resolution"] = Value::Object(by);
                        let note = match t.experimental_max_frames {
                            Some(long) => format!(
                                " At {res} the longest clip is {cap} s; up to {} s at {res} is an experimental feature (`{}`) an admin can enable.",
                                long / fps,
                                fastvideo_protocol::FLAG_H3_1080P_LONG
                            ),
                            None => format!(" At {res} the longest clip is {cap} s."),
                        };
                        let d = p["description"].as_str().unwrap_or("").to_owned();
                        p["description"] = Value::String(format!("{d}{note}").trim_start().to_owned());
                    }
                }
            }
        }
        Endpoint::LtxTextToVideoFast
        | Endpoint::LtxTextToVideoPro
        | Endpoint::LtxImageToVideoFast
        | Endpoint::LtxImageToVideoPro
        | Endpoint::LtxAudioToVideoFast
        | Endpoint::LtxAudioToVideoPro => {
            narrow_enum(&mut s, "resolution", |v| ltx::LtxResolution::ALL.iter().any(|r| v == r.as_str() && tiers.contains(&r.landscape().1)), None);
            narrow_enum(&mut s, "fps", |v| v.as_u64().is_some_and(|f| caps.fps.allows(f as u32)), None);
            // `auto` needs the LTX-2.5 duration head, which is not loaded.
            narrow_enum(&mut s, "duration", Value::is_number, None);
        }
        Endpoint::WanTextToVideo | Endpoint::WanImageToVideo | Endpoint::WanFastWan => {
            narrow_enum(&mut s, "resolution", |v| WanResolution::ALL.iter().any(|r| v == r.as_str() && tiers.contains(&r.short_edge())), None);
            narrow_range(&mut s, "num_frames", i64::from(caps.frames.min), i64::from(caps.frames.max));
        }
        Endpoint::LtxIngredient => {
            narrow_range(&mut s, "num_frames", i64::from(caps.frames.min), i64::from(caps.frames.max));
            // The rates the model generates at, as a list.
            if let Some(p) = s["properties"].get_mut("frames_per_second") {
                let (lo, hi) = (p["minimum"].as_u64().unwrap_or(0), p["maximum"].as_u64().unwrap_or(u64::MAX));
                let rates: Vec<Value> = caps.fps.allowed.iter().filter(|&&f| (lo..=hi).contains(&u64::from(f))).map(|&f| f.into()).collect();
                if !rates.is_empty() {
                    let def = p["default"].clone();
                    let obj = p.as_object_mut().expect("property object");
                    obj.remove("minimum");
                    obj.remove("maximum");
                    obj.insert("default".into(), if rates.contains(&def) { def } else { rates[0].clone() });
                    obj.insert("enum".into(), Value::Array(rates));
                }
            }
        }
    }
    s
}

/// The director form of an app whose model has `caps`: the `configure`
/// fields the console sets, with the values the model serves.
pub fn director_schema(caps: &ModelCaps) -> Value {
    use crate::director::messages::Resolution as R;
    let res: Vec<&str> = [R::R480, R::R768, R::R1080]
        .into_iter()
        .filter(|r| caps.canvas.short_edges.contains(&r.short_edge()))
        .map(|r| r.as_str())
        .collect();
    // The session's own default: 768p when served, else the last tier.
    let default = if res.contains(&"768p") { "768p" } else { res.last().copied().unwrap_or("768p") };
    let mut props = Map::new();
    props.insert(
        "resolution".into(),
        json!({"type": "string", "enum": res, "default": default, "description": "The resolution of every chunk (1080p takes about 2.5x as long per chunk as 768p)."}),
    );
    props.insert(
        "aspect_ratio".into(),
        json!({"type": "string", "enum": ["auto", "16:9", "9:16", "1:1"], "default": "auto", "description": "`auto` sends no aspect_ratio: the session follows the opening image (16:9 without one)."}),
    );
    object("Director", props, vec!["resolution", "aspect_ratio"], &[])
}

pub(crate) fn routes(router: Router<ServeCtx>, cfg: &Arc<FalConfig>) -> Router<ServeCtx> {
    let c = cfg.clone();
    let c2 = cfg.clone();
    router
        .route(
            "/fal/schema",
            get(move |State(ctx): State<ServeCtx>| {
                let models = ctx.engine().models();
                let engine = ctx.engine().clone();
                std::future::ready(Json(catalog_served(&c, &models, &|n| engine.alias(n))))
            }),
        )
        .route(
            "/fal/schema/{owner}/{alias}/{*sub}",
            get(move |State(ctx): State<ServeCtx>, Path((owner, alias, sub)): Path<(String, String, String)>| {
                let id = format!("{owner}/{alias}");
                let sub = sub.trim_matches('/').to_owned();
                let app = c2.apps.iter().find(|a| a.is_valid() && a.id == id);
                let models = ctx.engine().models();
                let engine = ctx.engine().clone();
                let alias = |n: &str| engine.alias(n);
                std::future::ready(match (app, sub.as_str()) {
                    (Some(a), "director") => match director_caps(a, &models, &alias) {
                        Some(caps) => Json(director_schema(&caps)).into_response(),
                        None => not_found(&id, &sub),
                    },
                    (Some(a), _) => match Endpoint::from_sub(&sub).filter(|e| a.endpoints().contains(e)) {
                        Some(e) => match endpoint_caps(a, e, &models, &alias) {
                            Some(caps) => Json(served_schema(a.kind(), e, &caps)).into_response(),
                            None => not_found(&id, &sub),
                        },
                        None => not_found(&id, &sub),
                    },
                    _ => not_found(&id, &sub),
                })
            }),
        )
}

fn not_found(app: &str, sub: &str) -> Response {
    (StatusCode::NOT_FOUND, Json(json!({"detail": format!("no endpoint `{app}/{sub}` on this server")}))).into_response()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::FalInput;

    const KINDS: [AppKind; 5] = [AppKind::H3, AppKind::H3Base, AppKind::Ltx25, AppKind::Wan, AppKind::LtxQuality];

    fn defaults(s: &Value) -> Map<String, Value> {
        let mut m = Map::new();
        for (k, p) in s["properties"].as_object().unwrap() {
            if let Some(d) = p.get("default") {
                m.insert(k.clone(), d.clone());
            }
        }
        m
    }

    /// The fields a body needs besides the defaults: the prompt and the
    /// required media.
    fn minimal(e: Endpoint) -> Map<String, Value> {
        let mut m = Map::new();
        m.insert("prompt".into(), "a cat".into());
        match e {
            Endpoint::ReferenceToVideo => {
                m.insert("reference_image_urls".into(), json!(["https://a.test/1.png"]));
            }
            Endpoint::LtxImageToVideoFast
            | Endpoint::LtxImageToVideoPro
            | Endpoint::WanImageToVideo
            | Endpoint::LtxIngredient => {
                m.insert("image_url".into(), "https://a.test/1.png".into());
            }
            Endpoint::LtxAudioToVideoFast | Endpoint::LtxAudioToVideoPro => {
                m.insert("audio_url".into(), "https://a.test/speech.mp3".into());
            }
            _ => {}
        }
        m
    }

    /// Every property the schema lists is one the parser reads, and the
    /// schema's defaults, bounds and enums are exactly what it accepts.
    #[test]
    fn schema_matches_validation() {
        for kind in KINDS {
            for &e in kind.endpoints() {
                let s = input_schema_for(kind, e);
                let parse = |b: &Map<String, Value>| FalInput::parse_for(kind, e, &Value::Object(b.clone()));
                let props = s["properties"].as_object().unwrap();
                let order: Vec<&str> = s["x-fal-order-properties"].as_array().unwrap().iter().map(|v| v.as_str().unwrap()).collect();
                assert_eq!(order.len(), props.len(), "{e:?}");
                assert!(order.iter().all(|k| props.contains_key(*k)));
                for r in s["required"].as_array().unwrap() {
                    let mut b = minimal(e);
                    b.remove(r.as_str().unwrap());
                    assert!(parse(&b).is_err(), "{e:?} without {r}");
                }
                let mut body = defaults(&s);
                body.extend(minimal(e));
                let parsed = parse(&body).unwrap_or_else(|err| panic!("{kind:?} {e:?} defaults: {err:?}"));
                // Defaults in the schema equal the parser's defaults.
                assert_eq!(parse(&minimal(e)).unwrap(), parsed, "{e:?} defaults");
                // LTX: 20 s is only within the engine's frame grid at 24 fps.
                if kind == AppKind::Ltx25 {
                    body.insert("fps".into(), 24.into());
                }
                for (k, p) in props {
                    let p_enum = p.get("enum").or_else(|| p.pointer("/anyOf/0/enum"));
                    if let Some(list) = p_enum.and_then(Value::as_array) {
                        for v in list {
                            let mut b = body.clone();
                            b.insert(k.clone(), v.clone());
                            parse(&b).unwrap_or_else(|err| panic!("{e:?} {k}={v}: {err:?}"));
                        }
                        let mut b = body.clone();
                        b.insert(k.clone(), "bogus".into());
                        assert!(parse(&b).is_err(), "{e:?} {k}");
                    }
                    let (lo, hi) = (p.get("minimum").and_then(Value::as_f64), p.get("maximum").and_then(Value::as_f64));
                    if let (Some(lo), Some(hi)) = (lo, hi) {
                        let step = if p["type"] == "integer" { 1.0 } else { 0.01 };
                        for (v, ok) in [(lo, true), (hi, true), (lo - step, false), (hi + step, false)] {
                            let mut b = body.clone();
                            let v = if p["type"] == "integer" { json!(v as i64) } else { json!(v) };
                            b.insert(k.clone(), v.clone());
                            assert_eq!(parse(&b).is_ok(), ok, "{e:?} {k}={v}");
                        }
                    }
                    if let Some(max) = p.get("maxItems").and_then(Value::as_u64) {
                        let mut b = body.clone();
                        b.insert(k.clone(), Value::Array(vec!["https://a.test/x".into(); max as usize + 1]));
                        assert!(parse(&b).is_err(), "{e:?} {k} maxItems");
                    }
                }
                let max = s["properties"]["prompt"]["maxLength"].as_u64().unwrap() as usize;
                let mut b = body.clone();
                b.insert("prompt".into(), "x".repeat(max + 1).into());
                assert!(parse(&b).is_err());
            }
        }
    }

    #[test]
    fn audio_to_video_form_uploads_audio() {
        let fam = FalConfig { apps: vec![FalApp::from_id("lightricks/ltx-2.5")], ..FalConfig::default() };
        let c = catalog(&fam);
        let eps = c["apps"][0]["endpoints"].as_array().unwrap();
        let a2v: Vec<(&str, &str)> = eps
            .iter()
            .filter(|e| e["sub"].as_str().unwrap().starts_with("audio-to-video"))
            .map(|e| (e["endpoint_id"].as_str().unwrap(), e["model"].as_str().unwrap()))
            .collect();
        assert_eq!(
            a2v,
            [("lightricks/ltx-2.5/audio-to-video/fast", "ltx-turbo"), ("lightricks/ltx-2.5/audio-to-video/pro", "ltx-pro")]
        );
        for (e, max) in [(Endpoint::LtxAudioToVideoFast, "20"), (Endpoint::LtxAudioToVideoPro, "10")] {
            let s = input_schema_for(AppKind::Ltx25, e);
            assert_eq!(s["required"], json!(["audio_url"]));
            assert_eq!(s["properties"]["audio_url"]["x-fv-media"], "audio");
            assert_eq!(s["properties"]["image_url"]["x-fv-media"], "image");
            assert!(s["properties"]["audio_url"]["description"].as_str().unwrap().contains(&format!("2 and {max} seconds")));
            assert_eq!(s["x-fal-order-properties"][0], "audio_url");
        }
    }

    #[test]
    fn catalog_lists_configured_apps() {
        let c = catalog(&FalConfig::default());
        let ids: Vec<&str> = c["apps"].as_array().unwrap().iter().map(|a| a["id"].as_str().unwrap()).collect();
        assert_eq!(ids, ["minimax/h3-max", "minimax/h3-turbo", "minimax/h3-draft", "minimax/h3-max-turbo", "minimax/h3"]);
        assert_eq!(c["apps"][0]["endpoints"][2]["endpoint_id"], "minimax/h3-max/reference-to-video");
        assert_eq!(c["apps"][1]["tier"], "turbo");
        assert_eq!(c["apps"][3]["tier"], "turbo");
        assert_eq!(c["apps"][4]["kind"], "h3_base");
        let fam = FalConfig { apps: vec![FalApp::from_id("lightricks/ltx-2.5"), FalApp::from_id("fal-ai/wan")], ..FalConfig::default() };
        let c = catalog(&fam);
        assert_eq!(c["apps"][0]["endpoints"][1]["endpoint_id"], "lightricks/ltx-2.5/text-to-video/pro");
        assert_eq!(c["apps"][0]["endpoints"][1]["model"], "ltx-pro");
        assert_eq!(c["apps"][0]["director"], false);
        assert_eq!(c["apps"][1]["endpoints"][2]["endpoint_id"], "fal-ai/wan/v2.2-5b/text-to-video/fast-wan");
        assert_eq!(c["apps"][1]["endpoints"][2]["model"], "wan-turbo");
        assert_eq!(c["apps"][1]["endpoints"][0]["model"], "wan-max");
        let q = FalConfig { apps: vec![FalApp::from_id("fal-ai/ltx-2.3-quality")], ..FalConfig::default() };
        let c = catalog(&q);
        assert_eq!(c["apps"][0]["kind"], "ltx_quality");
        assert_eq!(c["apps"][0]["endpoints"][0]["endpoint_id"], "fal-ai/ltx-2.3-quality/ingredient");
        assert_eq!(c["apps"][0]["endpoints"][0]["model"], "ltx-pro");
    }
}
