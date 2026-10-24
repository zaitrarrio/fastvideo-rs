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
