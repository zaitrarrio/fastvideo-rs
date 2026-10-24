//! The [`ParamSpec`]s of the LTX generation endpoints (plan E2, ADR-0012):
//! one per (endpoint, `model` id) whose target is served here, from the
//! typed bodies ([`crate::body`]), the LTX product rules
//! (`fastvideo_protocol::params::rules::ltx`, the matrix the parser
//! enforces) and the served model's caps. `/v1/*` and `/v2/*` share the
//! request schema; the spec names the `/v2` path.

use fastvideo_protocol::params::rules::ltx as rules;
use fastvideo_protocol::params::{
    knob, Domain, MediaField, Param, ParamSpec, ParamType, Support, UnknownFields,
};
use fastvideo_protocol::{
    resolve_model, resolve_tier, route_task, ExtendAt, Family, GapId, ModelCaps, NumRange,
    RetakeMode, Task, EDIT_DURATION_MAX_S, EDIT_DURATION_MIN_S, EXTEND_CONTEXT_MAX_S,
    EXTEND_CONTEXT_MIN_S,
};

use crate::models::{
    parse_tier_alias, resolution_strings, LtxModel, LtxModels, ModelClass, ResTier,
};
use crate::request::{Endpoint, A2V_DEFAULT_MODEL, EDIT_DEFAULT_MODEL};

const UPSTREAM: &str = "docs/serve/specs/ltx/openapi.json#/components/schemas/";

/// The served LTX model an engine name binds to (`routes::bind_model`): a
/// model, served name or alias of the LTX family, else the model tagged
/// with the canonical tier alias's tier.
pub fn bind<'a>(
    engine_name: &str,
    alias: &dyn Fn(&str) -> Option<String>,
    models: &'a [ModelCaps],
) -> Option<&'a ModelCaps> {
    if let Ok(c) = resolve_model(engine_name, alias, models) {
        if c.family == Family::Ltx2 {
            return Some(c);
        }
    }
    parse_tier_alias(engine_name).and_then(|t| resolve_tier(Family::Ltx2, t, models).ok())
}

fn schema_of(e: Endpoint) -> &'static str {
    match e {
        Endpoint::TextToVideo => "TextToVideoRequest",
        Endpoint::ImageToVideo => "ImageToVideoRequest",
        Endpoint::AudioToVideo => "AudioToVideoRequest",
        Endpoint::Retake => "EditVideoRequest",
        Endpoint::Extend => "ExtendVideoRequest",
    }
}

fn task_of(e: Endpoint) -> Task {
    match e {
        Endpoint::TextToVideo => Task::T2V,
        Endpoint::ImageToVideo => Task::I2V,
        Endpoint::AudioToVideo => Task::A2V,
        Endpoint::Retake => Task::Retake,
        Endpoint::Extend => Task::Extend,
    }
}

/// The spec of `endpoint` with `model: <id>` (`m`) served by `caps`.
pub fn endpoint_spec(
    endpoint: Endpoint,
    id: &str,
    m: &LtxModel,
    ids: &[&str],
    caps: &ModelCaps,
) -> ParamSpec {
    use ParamType as T;
    let schema = schema_of(endpoint);
    let p = |name: &str, ty: ParamType, maps_to: &str| {
        Param::new(name, ty, maps_to).upstream(format!("{UPSTREAM}{schema}"))
    };
    let mut s = ParamSpec::new(
        "ltx",
        format!("POST /v2/{}", endpoint.segment()),
        id,
        caps,
        UnknownFields::Ignore,
    );
    let fast = m.class == ModelClass::Fast;
    let all_res = resolution_strings();
    let all: Vec<&str> = all_res.iter().map(String::as_str).collect();
    let served_res: Vec<&str> = all
        .iter()
        .copied()
        .filter(|r| {
            ResTier::ALL.iter().any(|t| {
                let (w, h) = t.landscape();
                (*r == format!("{w}x{h}") || *r == format!("{h}x{w}"))
                    && caps.canvas.short_edges.contains(&t.short_edge())
            })
        })
        .collect();
    let resolution = p(
        "resolution",
        T::String,
        "canvas (exact; padded to the multiple and cropped)",
    )
    .domain(Domain::strings(&served_res))
    .wire(Domain::strings(&all));
    let served_fps: Vec<u32> = rules::FPS
        .iter()
        .copied()
        .filter(|f| caps.fps.allows(*f))
        .collect();
    let fps = p("fps", T::Integer, "timing.fps")
        .with_default(rules::API_DEFAULT_FPS)
        .domain(Domain::one_of(served_fps))
        .wire(Domain::one_of(rules::FPS))
        .note(format!(
            "unserved listed rates: Unsupported({})",
            GapId::LtxFps.code()
        ));
    let prompt = |required: bool| {
        let q = p("prompt", T::String, "prompt").domain(Domain::Text {
            min_chars: 0,
            max_chars: Some(rules::PROMPT_MAX_CHARS),
            non_blank: false,
        });
        if required {
            q.required()
        } else {
            q
        }
    };
    let camera = p("camera_motion", T::String, "-")
        .domain(Domain::strings(&rules::CAMERA_MOTIONS))
        .support(Support::Gap {
            gap: GapId::LtxCameraMotion,
        });
    let enhance = Param::new("enhance_prompt", T::Boolean, "prompt_expansion")
        .note("ours: opts in or out of the server's upsampler; a non-boolean is ignored");
    let model = |default: Option<&str>| {
        let q = p("model", T::String, "model").domain(Domain::strings(ids));
        match default {
            Some(d) => q.with_default(d),
            None => q.required(),
        }
    };
    match endpoint {
        Endpoint::TextToVideo | Endpoint::ImageToVideo => {
            let max = rules::max_duration(fast, rules::LONG_MAX_SHORT_EDGE, rules::API_DEFAULT_FPS);
            let mut duration = p("duration", T::Integer, "timing.length (seconds, nearest on the grid)")
                .required()
                .domain(Domain::one_of(rules::durations(max)))
                .note(format!(
                    "the support matrix narrows it per resolution and fps (at most {} s off 720p/1080p at 24/25 fps)",
                    rules::DURATION_MAX_SHORT
                ));
            if m.auto_duration {
                duration = duration.nullable().note(format!(
                    "null (automatic): Unsupported({}); the matrix narrows numbers per resolution and fps",
                    GapId::LtxAutoDuration.code()
                ));
            }
            if endpoint == Endpoint::ImageToVideo {
                s.push(
                    p("image_uri", T::String, "keyframes (first)")
                        .required()
                        .domain(Domain::media(MediaField::Image)),
                );
                s.push(
                    p("last_frame_uri", T::String, "keyframes (last)")
                        .domain(Domain::media(MediaField::Image))
                        .support(knob(
                            caps.supports(Task::Keyframes),
                            Support::Gap {
                                gap: GapId::LtxKeyframes,
                            },
                        )),
                );
            }
            s.push(prompt(true))
                .push(model(None))
                .push(duration)
                .push(resolution.required())
                .push(fps)
                .push(
                    p("generate_audio", T::Boolean, "audio_out")
                        .with_default(true)
                        .support(knob(caps.audio.is_some(), Support::NoOp)),
                )
                .push(camera)
                .push(enhance);
        }
        Endpoint::AudioToVideo => {
            s.push(
                p("audio_uri", T::String, "audio_in (drive; sets the length)")
                    .required()
                    .domain(Domain::media(MediaField::Audio)),
            )
            .push(
                p("image_uri", T::String, "keyframes (first)")
                    .domain(Domain::media(MediaField::Image)),
            )
            .push(
                p("last_frame_uri", T::String, "keyframes (last)")
                    .domain(Domain::media(MediaField::Image))
                    .note("requires image_uri"),
            )
            .push(prompt(false).note("required without image_uri"))
            .push(model(Some(A2V_DEFAULT_MODEL)))
            .push(resolution.note("omitted: 1920x1080, or 1080x1920 for a portrait image"))
            .push(fps)
            .push(camera)
            .push(
                Param::new("audio_uri(duration)", T::Number, "audio_in.max_s")
                    .domain(Domain::num(NumRange::new(
                        fastvideo_protocol::A2V_AUDIO_MIN_S,
                        f64::from(if id == "ltx-2-5-pro" {
                            rules::A2V_MAX_S_SHORT
                        } else {
                            rules::A2V_MAX_S_LONG
                        }),
                    )))
                    .note(format!(
                        "seconds of audio; {} s at 1440p / 4K",
                        rules::A2V_MAX_S_SHORT
                    )),
            );
        }
        Endpoint::Retake => {
            let modes: Vec<&str> = RetakeMode::ALL.iter().map(RetakeMode::as_str).collect();
            s.push(
                p("video_uri", T::String, "edit.video")
                    .required()
                    .domain(Domain::media(MediaField::Video)),
            )
            .push(prompt(false))
            .push(model(Some(EDIT_DEFAULT_MODEL)))
            .push(
                p("start_time", T::Number, "edit.start_s")
                    .required()
                    .domain(Domain::num_min(0.0)),
            )
            .push(
                p("duration", T::Number, "edit.duration_s")
                    .required()
                    .domain(Domain::num_min(EDIT_DURATION_MIN_S))
                    .note(format!(
                        "clamped to the video and to {EDIT_DURATION_MAX_S} s"
                    )),
            )
            .push(
                p("mode", T::String, "edit.mode")
                    .with_default(RetakeMode::default().as_str())
                    .domain(Domain::strings(&modes)),
            )
            .push(
                p("resolution", T::String, "canvas (at most the source)")
                    .domain(Domain::strings(&rules::RETAKE_RESOLUTIONS)),
            );
        }
        Endpoint::Extend => {
            let modes: Vec<&str> = [ExtendAt::Start, ExtendAt::End]
                .iter()
                .map(ExtendAt::as_str)
                .collect();
            s.push(
                p("video_uri", T::String, "edit.video")
                    .required()
                    .domain(Domain::media(MediaField::Video)),
            )
            .push(prompt(false))
            .push(model(Some(EDIT_DEFAULT_MODEL)))
            .push(
                p("duration", T::Number, "edit.duration_s")
                    .required()
                    .domain(Domain::num(NumRange::new(
                        EDIT_DURATION_MIN_S,
                        EDIT_DURATION_MAX_S,
                    ))),
            )
            .push(
                p("mode", T::String, "edit.at")
                    .with_default(ExtendAt::End.as_str())
                    .domain(Domain::strings(&modes)),
            )
            .push(
                p("context", T::Number, "edit.context_s").domain(Domain::num(NumRange::new(
                    EXTEND_CONTEXT_MIN_S,
                    EXTEND_CONTEXT_MAX_S,
                ))),
            );
        }
    }
    s
}

/// Every (endpoint, `model` id) whose target is served here and whose
/// task the target (or its task companion) serves.
pub fn param_specs(
    table: &LtxModels,
    models: &[ModelCaps],
    alias: &dyn Fn(&str) -> Option<String>,
) -> Vec<ParamSpec> {
    let ids: Vec<&str> = table.ids().collect();
    let mut out = Vec::new();
    for e in Endpoint::ALL {
        for id in &ids {
            if e == Endpoint::AudioToVideo && *id == "ltx-2-3-fast" {
                continue;
            }
            let Some(m) = table.get(id) else { continue };
            let Some(caps) = bind(&m.target.engine_name(), alias, models) else {
                continue;
            };
            let caps = route_task(caps, task_of(e), models);
            if caps.supports(task_of(e)) {
                out.push(endpoint_spec(e, id, m, &ids, caps));
            }
        }
    }
    out
}
