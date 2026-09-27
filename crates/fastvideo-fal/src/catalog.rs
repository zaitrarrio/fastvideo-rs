//! Input JSON Schemas of the app endpoints and the catalog of mounted apps
//! (console, WP-20). Not a fal route: fal publishes its schemas as OpenAPI
//! on its own site; these are ours, served for the console forms.
//!
//! | Route | Behaviour |
//! |---|---|
//! | `GET /fal/schema` | `{apps:[{id, model, tier, endpoints:[{sub, endpoint_id, title}]}]}` for the configured apps |
//! | `GET /fal/schema/{owner}/{alias}/{sub}` | The endpoint's input JSON Schema (draft 2020-12) |
//!
//! The schema is built from the same constants [`FalInput::parse`] enforces
//! (`PROMPT_MAX_CHARS`, `DURATION_MIN/MAX`, the `Resolution` and
//! `AspectRatio` enums, the reference limits), and the tests check both
//! agree, so the console form cannot drift from validation. Property order
//! follows fal's `x-fal-order-properties`. Vendor keys for the console:
//! `x-fv-media` (`image` / `video` / `audio`: an uploadable URL field) and
//! `x-fv-advanced` (shown under "Additional settings").
//!
//! [`FalInput::parse`]: crate::schema::FalInput::parse

use std::sync::Arc;

use axum::extract::Path;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::get;
use axum::{Json, Router};
use fastvideo_serve_kit::ServeCtx;
use serde_json::{json, Map, Value};

use crate::schema::{
    AspectRatio, Endpoint, Resolution, DURATION_MAX, DURATION_MIN, MAX_REFERENCES, MAX_REFERENCE_AUDIO,
    MAX_REFERENCE_IMAGES, MAX_REFERENCE_VIDEOS, PROMPT_MAX_CHARS,
};
use crate::FalConfig;

fn title(e: Endpoint) -> &'static str {
    match e {
        Endpoint::TextToVideo => "Text to Video",
        Endpoint::ImageToVideo => "Image to Video",
        Endpoint::ReferenceToVideo => "Reference to Video",
    }
}

fn media_url(kind: &str, description: &str) -> Value {
    json!({
        "anyOf": [{"type": "string", "minLength": 1, "pattern": "\\S"}, {"type": "null"}],
        "default": null,
        "description": description,
        "x-fv-media": kind,
    })
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

const TARGET_AUDIO: &str = "Optional URL of an audio clip at least 2 seconds long (maximum 15 MB) to pin to the generated soundtrack. Accepts an HTTP(S) URL or a base64 data URI.";

/// The input JSON Schema of `endpoint`.
pub fn input_schema(endpoint: Endpoint) -> Value {
    let mut props = Map::new();
    let prompt_desc = match endpoint {
        Endpoint::ReferenceToVideo => "Text prompt for video generation. Refer to reference assets by their modality and order in the reference lists: Image 1, Image 2, Video 1, Audio 1, and so on.",
        _ => "Text prompt for video generation",
    };
    props.insert(
        "prompt".into(),
        json!({"type": "string", "minLength": 1, "maxLength": PROMPT_MAX_CHARS, "description": prompt_desc, "x-fv-multiline": true}),
    );
    props.insert(
        "duration".into(),
        json!({"type": "integer", "minimum": DURATION_MIN, "maximum": DURATION_MAX, "default": DURATION_MIN, "description": "The duration of the video in seconds."}),
    );
    props.insert(
        "resolution".into(),
        json!({
            "type": "string",
            "enum": Resolution::ALL.iter().map(|r| r.as_str()).collect::<Vec<_>>(),
            "default": Resolution::P768.as_str(),
            "description": "The native generation resolution, or 1080P latent refinement from a native 768P source.",
        }),
    );
    props.insert(
        "seed".into(),
        json!({"anyOf": [{"type": "integer", "minimum": 0}, {"type": "null"}], "default": null, "description": "Random seed. A random seed is selected when omitted.", "x-fv-advanced": true}),
    );
    props.insert(
        "enable_safety_checker".into(),
        json!({"type": "boolean", "default": true, "description": "If set to true, the safety checker will be enabled.", "x-fv-advanced": true}),
    );
    props.insert(
        "sync_mode".into(),
        json!({"type": "boolean", "default": false, "description": "Return the generated video as base64 instead of a CDN URL.", "x-fv-advanced": true}),
    );
    props.insert(
        "prompt_expansion_mode".into(),
        json!({"type": "string", "default": "balanced", "examples": ["disabled", "balanced", "quality"], "description": "How much effort to spend rewriting the prompt before generation. Accepted; this server does not expand prompts.", "x-fv-advanced": true}),
    );
    let ratios = |list: &[AspectRatio]| list.iter().map(|a| a.as_str()).collect::<Vec<_>>();
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
                json!({"type": "string", "enum": ratios(&AspectRatio::T2V), "default": AspectRatio::R16x9.as_str(), "description": "The aspect ratio of the generated video."}),
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
        Endpoint::ReferenceToVideo => {
            props.insert(
                "aspect_ratio".into(),
                json!({"type": "string", "enum": ratios(&AspectRatio::R2V), "default": AspectRatio::Adaptive.as_str(), "description": "The aspect ratio of the generated video."}),
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
    let mut schema = json!({
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": title(endpoint),
        "type": "object",
        "properties": Value::Object(props),
        "required": ["prompt"],
        "x-fal-order-properties": order,
    });
    if endpoint == Endpoint::ReferenceToVideo {
        schema["x-fv-max-references"] = MAX_REFERENCES.into();
        schema["x-fv-min-references"] = 1.into();
    }
    schema
}

/// `GET /fal/schema` body.
pub fn catalog(cfg: &FalConfig) -> Value {
    let apps: Vec<Value> = cfg
        .apps
        .iter()
        .filter(|a| a.is_valid())
        .map(|a| {
            json!({
                "id": a.id,
                "model": a.model,
                "tier": a.tier.map(|(_, t)| t.as_str()),
                "endpoints": Endpoint::ALL.iter().map(|e| json!({
                    "sub": e.sub(),
                    "endpoint_id": format!("{}/{}", a.id, e.sub()),
                    "title": title(*e),
                })).collect::<Vec<_>>(),
            })
        })
        .collect();
    json!({"apps": apps})
}

pub(crate) fn routes(router: Router<ServeCtx>, cfg: &Arc<FalConfig>) -> Router<ServeCtx> {
    let c = cfg.clone();
    let c2 = cfg.clone();
    router
        .route("/fal/schema", get(move || std::future::ready(Json(catalog(&c)))))
        .route(
            "/fal/schema/{owner}/{alias}/{sub}",
            get(move |Path((owner, alias, sub)): Path<(String, String, String)>| {
                let id = format!("{owner}/{alias}");
                let known = c2.apps.iter().any(|a| a.is_valid() && a.id == id);
                std::future::ready(match (known, Endpoint::from_sub(&sub)) {
                    (true, Some(e)) => Json(input_schema(e)).into_response(),
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

    fn defaults(e: Endpoint) -> Map<String, Value> {
        let s = input_schema(e);
        let mut m = Map::new();
        for (k, p) in s["properties"].as_object().unwrap() {
            if let Some(d) = p.get("default") {
                m.insert(k.clone(), d.clone());
            }
        }
        m
    }

    /// Every property the schema lists is one the parser reads, and the
    /// schema's defaults, bounds and enums are exactly what it accepts.
    #[test]
    fn schema_matches_validation() {
        for e in Endpoint::ALL {
            let s = input_schema(e);
            let props = s["properties"].as_object().unwrap();
            let order: Vec<&str> = s["x-fal-order-properties"].as_array().unwrap().iter().map(|v| v.as_str().unwrap()).collect();
            assert_eq!(order.len(), props.len(), "{e:?}");
            assert!(order.iter().all(|k| props.contains_key(*k)));
            let mut body = defaults(e);
            body.insert("prompt".into(), "a cat".into());
            if e == Endpoint::ReferenceToVideo {
                body.insert("reference_image_urls".into(), json!(["https://a.test/1.png"]));
            }
            let parsed = FalInput::parse(e, &Value::Object(body.clone())).unwrap();
            // Defaults in the schema equal the parser's defaults.
            let mut min = Map::new();
            min.insert("prompt".into(), "a cat".into());
            if e == Endpoint::ReferenceToVideo {
                min.insert("reference_image_urls".into(), json!(["https://a.test/1.png"]));
            }
            assert_eq!(FalInput::parse(e, &Value::Object(min)).unwrap(), parsed, "{e:?} defaults");
            for (k, p) in props {
                if let Some(list) = p.get("enum").and_then(Value::as_array) {
                    for v in list {
                        let mut b = body.clone();
                        b.insert(k.clone(), v.clone());
                        FalInput::parse(e, &Value::Object(b)).unwrap_or_else(|err| panic!("{e:?} {k}={v}: {err:?}"));
                    }
                    let mut b = body.clone();
                    b.insert(k.clone(), "bogus".into());
                    assert!(FalInput::parse(e, &Value::Object(b)).is_err(), "{e:?} {k}");
                }
                if let (Some(lo), Some(hi)) = (p.get("minimum").and_then(Value::as_i64), p.get("maximum").and_then(Value::as_i64)) {
                    for (v, ok) in [(lo, true), (hi, true), (lo - 1, false), (hi + 1, false)] {
                        let mut b = body.clone();
                        b.insert(k.clone(), v.into());
                        assert_eq!(FalInput::parse(e, &Value::Object(b)).is_ok(), ok, "{e:?} {k}={v}");
                    }
                }
                if let Some(max) = p.get("maxItems").and_then(Value::as_u64) {
                    let mut b = body.clone();
                    b.insert(k.clone(), Value::Array(vec!["https://a.test/x".into(); max as usize + 1]));
                    assert!(FalInput::parse(e, &Value::Object(b)).is_err(), "{e:?} {k} maxItems");
                }
            }
            let max = s["properties"]["prompt"]["maxLength"].as_u64().unwrap() as usize;
            let mut b = body.clone();
            b.insert("prompt".into(), "x".repeat(max + 1).into());
            assert!(FalInput::parse(e, &Value::Object(b)).is_err());
        }
    }

    #[test]
    fn catalog_lists_configured_apps() {
        let c = catalog(&FalConfig::default());
        let ids: Vec<&str> = c["apps"].as_array().unwrap().iter().map(|a| a["id"].as_str().unwrap()).collect();
        assert_eq!(ids, ["minimax/h3-max", "minimax/h3-turbo", "minimax/h3-draft"]);
        assert_eq!(c["apps"][0]["endpoints"][2]["endpoint_id"], "minimax/h3-max/reference-to-video");
        assert_eq!(c["apps"][1]["tier"], "turbo");
    }
}
