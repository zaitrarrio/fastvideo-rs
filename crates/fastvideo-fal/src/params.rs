//! The [`ParamSpec`]s of the fal queue endpoints (plan E2, ADR-0012): one
//! per (app, endpoint) whose model is served here, read from the schema
//! the endpoint is served with ([`served_schema`]: fal's schema narrowed
//! to the model's caps) and fal's full schema ([`input_schema_for`], what
//! the parser takes, built from the `fastvideo_protocol::params::rules`
//! constants it validates with; the catalog tests check the parser and the
//! schema agree).

use fastvideo_protocol::params::{params_from_json_schema, ParamSpec, Support, UnknownFields};
use fastvideo_protocol::{GapId, ModelCaps};

use crate::catalog::{endpoint_caps, input_schema_for, served_schema};
use crate::schema::{AppKind, Endpoint};
use crate::FalConfig;

/// The internal field a fal field fills, and its support on `endpoint`.
pub fn field_map(endpoint: Endpoint, field: &str) -> (String, Support) {
    use Endpoint as E;
    use Support::{Gap, Honoured, NoOp, Refused};
    let fast_wan = endpoint == E::WanFastWan;
    let (to, support) = match field {
        "prompt" => ("prompt", Honoured),
        "duration" => match endpoint {
            E::LtxRetake | E::LtxExtend => ("edit.duration_s", Honoured),
            _ => ("timing.length (seconds)", Honoured),
        },
        "resolution" => ("canvas.short_edge", Honoured),
        "aspect_ratio" => ("canvas.ratio", Honoured),
        "seed" => ("seed", Honoured),
        "sync_mode" => ("output.inline_data_uri", Honoured),
        "enable_safety_checker" => ("-", NoOp),
        "prompt_expansion_mode" | "enable_prompt_expansion" => ("prompt_expansion", Honoured),
        "target_audio_url" => (
            "audio_in (target soundtrack)",
            Gap {
                gap: GapId::H3TargetAudio,
            },
        ),
        "image_url" => match endpoint {
            E::LtxIngredient => ("references (the reference sheet)", Honoured),
            _ => ("keyframes (first)", Honoured),
        },
        "end_image_url" => ("keyframes (last)", Honoured),
        "reference_image_urls" | "reference_video_urls" | "reference_audio_urls" => {
            ("references", Honoured)
        }
        "generate_audio" => ("audio_out", Honoured),
        "camera_motion" => (
            "-",
            Gap {
                gap: GapId::LtxCameraMotion,
            },
        ),
        "fps" | "frames_per_second" => ("timing.fps", Honoured),
        "num_frames" => ("timing.length (frames, aligned up)", Honoured),
        "negative_prompt" if fast_wan || endpoint == E::LtxIngredient => ("-", NoOp),
        "negative_prompt" => ("negative_prompt", Honoured),
        "num_inference_steps" => ("sampling.steps", Honoured),
        "guidance_scale" if fast_wan || endpoint == E::LtxAudioToVideoFast => ("-", NoOp),
        "guidance_scale" => ("sampling.guidance", Honoured),
        "shift" => ("sampling.flow_shift", Honoured),
        "interpolator_model" | "num_interpolated_frames" => ("-", Refused),
        "ingredient_strength" => ("sampling.reference_lora_strength", Honoured),
        "reference_strength" => ("sampling.reference_strength", Honoured),
        "audio_url" => ("audio_in (drive; sets the length)", Honoured),
        "video_url" => ("edit.video", Honoured),
        "start_time" => ("edit.start_s", Honoured),
        "retake_mode" => ("edit.mode", Honoured),
        "mode" => ("edit.at", Honoured),
        "context" => ("edit.context_s", Honoured),
        _ => ("-", Honoured),
    };
    (to.to_owned(), support)
}

/// The vendored upstream document an endpoint follows: H3 apps follow the
/// `minimax/h3-max` schema (`minimax/h3` its own), the family apps their
/// own endpoint ids.
fn upstream(kind: AppKind, app: &str, sub: &str) -> String {
    let app = match kind {
        AppKind::H3 => "minimax/h3-max",
        AppKind::H3Base => "minimax/h3",
        _ => app,
    };
    format!("docs/serve/specs/fal/{app}/{sub}.openapi.json")
}

/// Every (app, endpoint) of `cfg` whose model is served here.
pub fn param_specs(
    cfg: &FalConfig,
    models: &[ModelCaps],
    alias: &dyn Fn(&str) -> Option<String>,
) -> Vec<ParamSpec> {
    let mut out = Vec::new();
    for app in cfg.apps.iter().filter(|a| a.is_valid()) {
        for &e in app.endpoints() {
            let Some(caps) = endpoint_caps(app, e, models, alias) else {
                continue;
            };
            let served = served_schema(app.kind(), e, &caps);
            let wire = input_schema_for(app.kind(), e);
            let id = format!("{}/{}", app.id, e.sub());
            let mut s = ParamSpec::new(
                "fal",
                id.clone(),
                app.id.clone(),
                &caps,
                UnknownFields::Ignore,
            );
            s.params = params_from_json_schema(&served, Some(&wire), &|n| field_map(e, n))
                .into_iter()
                .map(|p| p.upstream(upstream(app.kind(), &app.id, e.sub())))
                .collect();
            out.push(s);
        }
    }
    out
}

/// The director control channel's client messages `configure` and `prompt`
/// (plan E3, ADR-0016) as [`ParamSpec`]s, surface `fal_director`, for any
/// director app (`model` `*`): what [`crate::director::messages::parse`]
/// accepts. Values the schema allows but a model cannot serve
/// (`resolution: "1080p"` without the tier, `audio_url` on a model without
/// audio-to-video) are refused later by the control state machine with
/// their own codes; the spec describes the parse.
pub fn director_specs() -> Vec<ParamSpec> {
    use crate::director::messages::{Resolution, CHUNK_DURATIONS};
    use fastvideo_protocol::params::rules::fal_director as rules;
    use fastvideo_protocol::params::{Domain, IntRange, MediaField, Param, ParamType as T};
    const UPSTREAM: &str = "docs/serve/specs/fal/fal-ai/minimax-h3-max-director/asyncapi.json";
    let spec = |endpoint: &str| ParamSpec {
        surface: "fal_director".into(),
        endpoint: endpoint.into(),
        model: "*".into(),
        engine_model: "*".into(),
        unknown_fields: UnknownFields::Reject,
        params: Vec::new(),
    };
    let p = |name: &str, ty: T, maps_to: &str| Param::new(name, ty, maps_to).upstream(UPSTREAM);
    let version = |what: &str| {
        p("prompt_version", T::Integer, what)
            .required()
            .domain(Domain::int(IntRange::new(
                1,
                i64::try_from(rules::PROMPT_VERSION_MAX).unwrap_or(i64::MAX),
            )))
    };
    let text = Domain::chars(rules::PROMPT_MAX_CHARS);
    let script = || {
        p("script", T::Array, "script beats")
            .nullable()
            .domain(Domain::items(
                u32::try_from(rules::SCRIPT_MAX_BEATS).unwrap_or(u32::MAX),
            ))
            .note("1 or more beats `{offset, prompt?, end_image_url?, audio_url?}`")
    };
    let resolutions: Vec<&str> = Resolution::ALL.iter().map(|r| r.as_str()).collect();
    let mut configure = spec("message configure");
    configure
        .push(version("correlation"))
        .push(
            p("prompt", T::String, "prompt")
                .required()
                .domain(text.clone()),
        )
        .push(
            p("resolution", T::String, "canvas.short_edge")
                .nullable()
                .domain(Domain::strings(&resolutions))
                .note("a tier the model lacks is refused by the session, not the parser"),
        )
        .push(
            p("aspect_ratio", T::String, "canvas.ratio")
                .nullable()
                .domain(Domain::strings(&["16:9", "9:16", "1:1"])),
        )
        .push(
            p("image_url", T::String, "keyframes (first)")
                .nullable()
                .domain(Domain::media(MediaField::Image)),
        )
        .push(
            p("end_image_url", T::String, "keyframes (last)")
                .nullable()
                .domain(Domain::media(MediaField::Image)),
        )
        .push(
            p("audio_url", T::String, "audio_in")
                .nullable()
                .domain(Domain::media(MediaField::Audio)),
        )
        .push(
            p("memory", T::Integer, "session memory")
                .nullable()
                .domain(Domain::int(rules::MEMORY)),
        )
        .push(
            p("audio_bitrate", T::Integer, "audio encoder bitrate")
                .nullable()
                .domain(Domain::one_of(rules::AUDIO_BITRATES)),
        )
        .push(
            p("seed", T::Integer, "seed")
                .nullable()
                .domain(Domain::Int {
                    min: None,
                    max: None,
                }),
        )
        .push(script())
        .push(
            p("protocol_version", T::Integer, "-")
                .nullable()
                .domain(Domain::one_of([rules::PROTOCOL_VERSION])),
        )
        .push(
            Param::new("chunk_duration", T::Integer, "chunk length (s)")
                .nullable()
                .with_default(rules::DEFAULT_CHUNK_DURATION)
                .domain(Domain::one_of(CHUNK_DURATIONS))
                .note("ours: clip models only"),
        )
        .push(
            Param::new("upsample", T::Boolean, "prompt_expansion")
                .nullable()
                .note("ours"),
        )
        .push(
            Param::new("style", T::String, "world bible style")
                .nullable()
                .domain(Domain::Text {
                    min_chars: 0,
                    max_chars: Some(rules::STYLE_MAX_CHARS),
                    non_blank: false,
                })
                .note("ours"),
        )
        .push(
            Param::new("entities", T::Array, "world bible cast")
                .nullable()
                .domain(Domain::items(
                    u32::try_from(rules::ENTITIES_MAX).unwrap_or(u32::MAX),
                ))
                .note("ours: `{id, kind, name, descriptor}`, unique ids"),
        )
        .push(
            Param::new("output", T::Object, "output copy")
                .nullable()
                .note("ours: taken only with `[director] allow_output`"),
        )
        .push(
            Param::new("x_fv_output", T::Object, "output copy")
                .nullable()
                .note("ours"),
        );
    let mut prompt = spec("message prompt");
    prompt
        .push(version("correlation"))
        .push(p("prompt", T::String, "prompt").nullable().domain(text))
        .push(
            p("end_image_url", T::String, "keyframes (last)")
                .nullable()
                .domain(Domain::media(MediaField::Image)),
        )
        .push(
            p("audio_url", T::String, "audio_in")
                .nullable()
                .domain(Domain::media(MediaField::Audio)),
        )
        .push(
            p("audio_behavior", T::String, "audio queueing")
                .nullable()
                .domain(Domain::strings(&["replace", "queue"])),
        )
        .push(p("replan", T::Boolean, "-").nullable())
        .push(script())
        .push(
            p("script_mode", T::String, "script queueing")
                .nullable()
                .domain(Domain::strings(&["replace", "append"])),
        );
    vec![configure, prompt]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::AspectRatio;
    use fastvideo_protocol::params::rules::h3;

    /// fal's typed H3 aspect enums are the shared H3 set (with `adaptive`
    /// first on reference-to-video).
    #[test]
    fn h3_aspects_are_the_shared_set() {
        let t2v: Vec<&str> = AspectRatio::T2V.iter().map(AspectRatio::as_str).collect();
        assert_eq!(t2v, h3::ASPECT_RATIOS);
        let r2v: Vec<&str> = AspectRatio::R2V.iter().map(AspectRatio::as_str).collect();
        let mut want = vec![h3::ADAPTIVE];
        want.extend(h3::ASPECT_RATIOS);
        assert_eq!(r2v, want);
    }

    #[test]
    fn every_field_of_every_schema_has_a_mapping() {
        use crate::schema::AppKind;
        for kind in [
            AppKind::H3,
            AppKind::H3Base,
            AppKind::Ltx25,
            AppKind::Wan,
            AppKind::LtxQuality,
            AppKind::Ltx23,
        ] {
            for &e in kind.endpoints() {
                let s = input_schema_for(kind, e);
                for name in s["properties"].as_object().unwrap().keys() {
                    let (to, _) = field_map(e, name);
                    assert!(
                        to != "-"
                            || matches!(
                                name.as_str(),
                                "enable_safety_checker"
                                    | "camera_motion"
                                    | "interpolator_model"
                                    | "num_interpolated_frames"
                                    | "negative_prompt"
                                    | "guidance_scale"
                            ),
                        "{kind:?} {e:?} {name}: unmapped"
                    );
                }
            }
        }
    }
}
