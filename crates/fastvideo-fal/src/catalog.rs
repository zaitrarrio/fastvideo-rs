//! Input JSON Schemas of the app endpoints and the catalog of mounted apps
//! (console, WP-20). Not a fal route: fal publishes its schemas as OpenAPI
//! on its own site; these are ours, served for the console forms.
//!
//! | Route | Behaviour |
//! |---|---|
//! | `GET /fal/schema` | `{apps:[{id, model, tier, kind, director, endpoints:[{sub, endpoint_id, title}]}]}` for the configured apps |
//! | `GET /fal/schema/{owner}/{alias}/{*sub}` | The endpoint's input JSON Schema (draft 2020-12); `sub` may have several segments (`text-to-video/fast`) |
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

use axum::extract::Path;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::get;
use axum::{Json, Router};
use fastvideo_serve_kit::ServeCtx;
use serde_json::{json, Map, Value};

use crate::schema::ingredient;
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

/// One catalog entry.
fn app_entry(a: &FalApp) -> Value {
    let kind = a.kind();
    json!({
        "id": a.id,
        "model": a.model,
        // The LTX and Wan apps pick a tier per endpoint.
        "tier": a.tier.filter(|_| kind.director()).map(|(_, t)| t.as_str()),
        "kind": kind,
        "director": kind.director(),
        "endpoints": a.endpoints().iter().map(|e| {
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

/// `GET /fal/schema` body.
pub fn catalog(cfg: &FalConfig) -> Value {
    let apps: Vec<Value> = cfg.apps.iter().filter(|a| a.is_valid()).map(app_entry).collect();
    json!({"apps": apps})
}

pub(crate) fn routes(router: Router<ServeCtx>, cfg: &Arc<FalConfig>) -> Router<ServeCtx> {
    let c = cfg.clone();
    let c2 = cfg.clone();
    router
        .route("/fal/schema", get(move || std::future::ready(Json(catalog(&c)))))
        .route(
            "/fal/schema/{owner}/{alias}/{*sub}",
            get(move |Path((owner, alias, sub)): Path<(String, String, String)>| {
                let id = format!("{owner}/{alias}");
                let sub = sub.trim_matches('/').to_owned();
                let app = c2.apps.iter().find(|a| a.is_valid() && a.id == id);
                let ep = Endpoint::from_sub(&sub);
                std::future::ready(match (app, ep) {
                    (Some(a), Some(e)) if a.endpoints().contains(&e) => Json(input_schema_for(a.kind(), e)).into_response(),
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
