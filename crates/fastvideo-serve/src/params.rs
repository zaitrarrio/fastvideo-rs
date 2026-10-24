//! `GET /fv/v1/params`: the [`ParamSpec`] of every mounted submit endpoint
//! for every served model (plan E2, ADR-0012), and the native API's own
//! spec ([`native_spec`]).
//!
//! | Surface | Specs |
//! |---|---|
//! | `native` | `POST /fv/v1/jobs`, one per served model |
//! | `openai_videos` | `POST /v1/videos`, one per served model |
//! | `fastwan` | `POST /generate` for the FastWan model |
//! | `minimax` | `POST /v2/video_generation`, one per MiniMax model name that resolves here |
//! | `ltx` | `POST /v2/<endpoint>` per (endpoint, `model` id) whose target serves the task |
//! | `fal` | one per configured (app, endpoint) whose model is served |
//!
//! The body is `{"object": "fv.param_specs", "specs": [...]}`; the
//! `param_spec_golden` test pins it for every shipped config. The director
//! and Reactor control channels are not in it yet (plan E3).

use axum::extract::State;
use axum::http::HeaderMap;
use axum::response::{IntoResponse, Response};
use axum::routing::get;
use axum::{Json, Router};
use fastvideo_protocol::params::{
    fps_domain, knob, short_edge_domain, Domain, MediaField, Param, ParamSpec, ParamType, Support,
    UnknownFields,
};
use fastvideo_protocol::{
    ApiError, ErrorCtx, GapId, ModelCaps, NumRange, ProtocolId, RetakeMode, Task,
    EDIT_DURATION_MAX_S, EDIT_DURATION_MIN_S, EXTEND_CONTEXT_MAX_S, EXTEND_CONTEXT_MIN_S,
};
use fastvideo_serve_kit::handlers::error_reply;
use fastvideo_serve_kit::ServeCtx;
use serde_json::{json, Value};

use crate::adapters::MountCfg;
use crate::native::Native;

/// `POST /fv/v1/jobs` for the model `caps` ([`crate::native::NativeBody`]).
pub fn native_spec(caps: &ModelCaps) -> ParamSpec {
    use ParamType as T;
    let k = &caps.knobs;
    let refused = Support::Refused;
    let mut s = ParamSpec::new(
        "native",
        "POST /fv/v1/jobs",
        caps.id.0.clone(),
        caps,
        UnknownFields::Reject,
    );
    let canvas = Domain::canvas(&caps.canvas);
    let fps = caps.fps.default.max(1);
    let max_s = f64::from(caps.frames.max) / f64::from(fps);
    let edits = caps.supports(Task::Retake) || caps.supports(Task::Extend);
    let modes: Vec<&str> = RetakeMode::ALL.iter().map(RetakeMode::as_str).collect();
    let p = Param::new;
    s.push(p("model", T::String, "model").required().note("an id, served name or alias"))
        .push(
            p("prompt", T::String, "prompt")
                .domain(Domain::non_blank(None))
                .note("required, except for audio-to-video with image_url and for retake / extend"),
        )
        .push(p("negative_prompt", T::String, "negative_prompt").nullable().support(knob(k.negative, refused)))
        .push(p("seed", T::Integer, "seed").nullable().domain(Domain::int_min(0)).support(knob(k.seed, refused)))
        .push(p("size", T::String, "canvas (exact)").nullable().domain(canvas.clone()))
        .push(p("width", T::Integer, "canvas.width").nullable().domain(canvas.clone()))
        .push(p("height", T::Integer, "canvas.height").nullable().domain(canvas))
        .push(p("aspect_ratio", T::String, "canvas.ratio").nullable().domain(Domain::aspect(&caps.canvas)))
        .push(p("short_edge", T::Integer, "canvas.short_edge").nullable().domain(short_edge_domain(caps)))
        .push(
            p("seconds", T::Number, "timing.length (seconds, aligned up)")
                .nullable()
                .domain(Domain::Number {
                    min: Some(0.0),
                    max: Some(max_s),
                    exclusive_min: true,
                }),
        )
        .push(
            p("num_frames", T::Integer, "timing.length (frames, exact)")
                .nullable()
                .domain(Domain::frames(&caps.frames))
                .with_default(caps.frames.default),
        )
        .push(p("fps", T::Integer, "timing.fps").nullable().domain(fps_domain(caps)).with_default(caps.fps.default))
        .push(
            p("steps", T::Integer, "sampling.steps")
                .nullable()
                .domain(Domain::int_min(1))
                .support(knob(
                    k.steps,
                    if caps.family == fastvideo_protocol::Family::H3 {
                        Support::Gap { gap: GapId::PerRequestSteps }
                    } else {
                        refused
                    },
                )),
        )
        .push(p("guidance", T::Number, "sampling.guidance").nullable().domain(Domain::num_min(0.0)).support(knob(k.guidance, refused)))
        .push(
            p("image_url", T::String, "keyframes (first)")
                .nullable()
                .domain(Domain::media(MediaField::Image))
                .support(knob(caps.supports(Task::I2V) || caps.supports(Task::A2V), refused)),
        )
        .push(
            p("last_image_url", T::String, "keyframes (last)")
                .nullable()
                .domain(Domain::media(MediaField::Image))
                .support(knob(caps.supports(Task::Keyframes), refused)),
        )
        .push(
            p("reference_urls", T::Array, "references (images)")
                .domain(Domain::items(caps.refs.images))
                .support(knob(caps.supports(Task::Ref2V), refused)),
        )
        .push(
            p("reference_strength", T::Number, "sampling.reference_strength")
                .nullable()
                .domain(Domain::num(NumRange::new(
                    0.0,
                    f64::from(fastvideo_protocol::negotiate::REFERENCE_STRENGTH_MAX),
                )))
                .support(knob(k.reference_strength, refused)),
        )
        .push(
            p("reference_lora_strength", T::Number, "sampling.reference_lora_strength")
                .nullable()
                .domain(Domain::num(NumRange::new(
                    0.0,
                    f64::from(fastvideo_protocol::negotiate::REFERENCE_LORA_STRENGTH_MAX),
                )))
                .support(knob(k.reference_strength, refused)),
        )
        .push(
            p("audio_url", T::String, "audio_in (drive; or a retake's window audio)")
                .nullable()
                .domain(Domain::media(MediaField::Audio))
                .support(knob(caps.supports(Task::A2V) || caps.supports(Task::Retake), refused)),
        )
        .push(
            p("video_url", T::String, "edit.video")
                .nullable()
                .domain(Domain::media(MediaField::Video))
                .support(knob(edits, refused)),
        )
        .push(p("start_s", T::Number, "edit.start_s").nullable().domain(Domain::num_min(0.0)).support(knob(edits, refused)))
        .push(
            p("end_s", T::Number, "edit.start_s + duration_s")
                .nullable()
                .note(format!(
                    "greater than start_s; the window is {EDIT_DURATION_MIN_S} to {EDIT_DURATION_MAX_S} s"
                ))
                .support(knob(edits, refused)),
        )
        .push(
            p("retake_mode", T::String, "edit.mode")
                .nullable()
                .domain(Domain::strings(&modes))
                .support(knob(edits, refused)),
        )
        .push(
            p("extend_s", T::Number, "edit.duration_s")
                .nullable()
                .domain(Domain::num(NumRange::new(EDIT_DURATION_MIN_S, EDIT_DURATION_MAX_S)))
                .support(knob(edits, refused)),
        )
        .push(p("extend_at", T::String, "edit.at").nullable().domain(Domain::strings(&["start", "end"])).support(knob(edits, refused)))
        .push(
            p("context_s", T::Number, "edit.context_s")
                .nullable()
                .domain(Domain::num(NumRange::new(EXTEND_CONTEXT_MIN_S, EXTEND_CONTEXT_MAX_S)))
                .support(knob(edits, refused)),
        )
        .push(
            p("prompt_expansion", T::String, "prompt_expansion")
                .nullable()
                .domain(Domain::strings(&["disabled", "balanced", "quality"])),
        );
    // The native parser checks shapes and these sets only; every other
    // bound is checked by negotiation against the caps (or, for an edit,
    // the source video).
    const PARSED: [&str; 4] = ["prompt", "prompt_expansion", "retake_mode", "extend_at"];
    for prm in &mut s.params {
        if prm.domain != Domain::Any && !PARSED.contains(&prm.name.as_str()) {
            prm.wire = Some(Domain::Any);
        }
    }
    s
}

/// Every spec this server's mounted APIs offer for `models`.
pub fn all_specs(
    mcfg: &MountCfg,
    native: bool,
    models: &[ModelCaps],
    alias: &dyn Fn(&str) -> Option<String>,
) -> Vec<ParamSpec> {
    let p = &mcfg.protocols;
    let mut out: Vec<ParamSpec> = Vec::new();
    if native {
        out.extend(models.iter().map(native_spec));
    }
    #[cfg(feature = "openai-videos")]
    {
        use fastvideo_openai_videos as ov;
        let fastwan = p
            .fastwan
            .then(|| ov::fastwan::fastwan_model(models, None))
            .flatten();
        out.extend(ov::params::param_specs(
            models,
            p.openai_videos,
            fastwan.as_deref(),
        ));
    }
    #[cfg(feature = "minimax")]
    if p.minimax {
        let mm = fastvideo_minimax::MiniMax::new(crate::adapters::minimax_config(mcfg));
        out.extend(fastvideo_minimax::params::param_specs(&mm, models, alias));
    }
    #[cfg(feature = "ltxapi")]
    if p.ltx {
        let cfg = crate::adapters::ltx_config(mcfg);
        out.extend(fastvideo_ltxapi::params::param_specs(
            &cfg.models,
            models,
            alias,
        ));
    }
    #[cfg(feature = "fal")]
    if p.fal {
        let cfg = crate::adapters::fal_config(mcfg);
        out.extend(fastvideo_fal::params::param_specs(&cfg, models, alias));
    }
    let _ = (p, alias);
    out
}

/// `GET /fv/v1/params` (auth as the native API).
pub fn routes(mcfg: MountCfg, native: bool) -> Router<ServeCtx> {
    Router::new().route(
        "/fv/v1/params",
        get(move |State(ctx): State<ServeCtx>, headers: HeaderMap| {
            let mcfg = mcfg.clone();
            async move {
                if let Err(e) = ctx.auth().authenticate(ProtocolId::Native, &headers) {
                    return reply_err(e);
                }
                let models = ctx.engine().models();
                let engine = ctx.engine().clone();
                let specs = all_specs(&mcfg, native, &models, &|n| engine.alias(n));
                Json(json!({"object": "fv.param_specs", "specs": specs})).into_response()
            }
        }),
    )
}

fn reply_err(e: ApiError) -> Response {
    let r = error_reply(&Native, &e, &ErrorCtx::default());
    let body = r.json_body().cloned().unwrap_or(Value::Null);
    (
        axum::http::StatusCode::from_u16(r.status)
            .unwrap_or(axum::http::StatusCode::INTERNAL_SERVER_ERROR),
        Json(body),
    )
        .into_response()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::native::NativeSubmit;
    use fastvideo_protocol::params::{model_agreement, parse_agreement};
    use fastvideo_protocol::{NormalizeCtx, SubmitEndpoint};

    /// The native parser enforces exactly the spec's parse-time domains.
    #[test]
    fn native_parser_enforces_the_spec() {
        let caps = ModelCaps::h3("h3", true);
        let Value::Object(base) = json!({"model": "h3", "prompt": "a cat"}) else {
            unreachable!()
        };
        let parse = |v: Value| {
            let b = serde_json::from_value(v).map_err(|e| ApiError::invalid(e.to_string()))?;
            NativeSubmit
                .normalize(b, &NormalizeCtx::new(time::OffsetDateTime::UNIX_EPOCH))
                .map(|_| ())
        };
        // The edit fields need `video_url` (checked together).
        let skip = ["retake_mode", "extend_at"];
        let bad = parse_agreement(&native_spec(&caps), &base, &parse, &skip);
        assert!(bad.is_empty(), "{bad:#?}");
    }

    /// The caps-derived domains are negotiation's own (`precheck`), on a
    /// model honouring every knob. Negotiation names fields in the
    /// normalized vocabulary (`duration`, `num_inference_steps`,
    /// `guidance_scale`), which the native API reports as is: mapped here
    /// to the native field names (an E3 finding, see docs/serve/params.md).
    #[test]
    fn native_domains_are_negotiations() {
        use fastvideo_protocol::{FpsCaps, FrameGrid, KnobCaps};
        let mut caps = ModelCaps::h3("wan", false);
        caps.family = fastvideo_protocol::Family::Wan;
        caps.tasks = [Task::T2V].into_iter().collect();
        caps.audio = None;
        caps.fps = FpsCaps {
            allowed: vec![16, 24],
            default: 24,
            container_only: true,
        };
        caps.frames = FrameGrid::new(4, 1, 17, 161, 81);
        caps.knobs = KnobCaps::all();
        caps.stream = None;
        let Value::Object(base) = json!({"model": "wan", "prompt": "a cat"}) else {
            unreachable!()
        };
        let check = |v: Value| {
            let b = serde_json::from_value(v).map_err(|e| ApiError::invalid(e.to_string()))?;
            let req =
                NativeSubmit.normalize(b, &NormalizeCtx::new(time::OffsetDateTime::UNIX_EPOCH))?;
            fastvideo_protocol::precheck(&req, &caps).map_err(|mut e| {
                e.param = e.param.map(|p| {
                    match p.as_str() {
                        "duration" => "seconds",
                        "num_inference_steps" => "steps",
                        "guidance_scale" => "guidance",
                        other => other,
                    }
                    .to_owned()
                });
                e
            })
        };
        // Checked with other fields: the canvas pair and tier, the edit
        // fields (they need `video_url`). `seed` is a u64: a negative one
        // is a body (type) error.
        let skip = [
            "seed",
            "width",
            "height",
            "short_edge",
            "retake_mode",
            "extend_at",
            "start_s",
            "extend_s",
            "context_s",
        ];
        let bad = model_agreement(&native_spec(&caps), &base, &check, &skip);
        assert!(bad.is_empty(), "{bad:#?}");
    }
}
