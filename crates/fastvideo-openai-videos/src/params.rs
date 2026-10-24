//! The [`ParamSpec`]s of `POST /v1/videos` and FastWan `POST /generate`
//! (plan E2, ADR-0012): every field of [`crate::VideoGenerationRequest`]
//! and [`crate::FastWanGenerate`], from the FastVideo contract
//! (`docs/serve/specs/fastvideo/videos.schema.json`, the
//! [`rules`] constants the parser validates with) and the model's caps.

use fastvideo_protocol::params::rules::openai_videos as rules;
use fastvideo_protocol::params::{
    fps_domain, knob, seconds_domain, short_edge_domain, Domain, MediaField, Param, ParamSpec,
    ParamType, Support, UnknownFields,
};
use fastvideo_protocol::{Family, GapId, ModelCaps, Task};

use crate::videos::EXTRA_PARAMS;

const UPSTREAM: &str = "docs/serve/specs/fastvideo/videos.schema.json";
const FASTWAN_UPSTREAM: &str = "docs/serve/research-minimax-fastvideo.md §2.2 (fastwan_link.py)";

fn p(name: &str, ty: ParamType, maps_to: &str) -> Param {
    Param::new(name, ty, maps_to).upstream(format!(
        "{UPSTREAM}#/properties/{}",
        name.split('.').next().unwrap_or(name)
    ))
}

/// The public name a client sends for `caps` (its first served name).
pub fn public_name(caps: &ModelCaps) -> String {
    caps.served_names
        .first()
        .cloned()
        .unwrap_or_else(|| caps.id.0.clone())
}

/// `POST /v1/videos` for the model `caps`.
pub fn videos_spec(caps: &ModelCaps) -> ParamSpec {
    use ParamType as T;
    let h3 = caps.family == Family::H3;
    let k = &caps.knobs;
    let mut s = ParamSpec::new(
        "openai_videos",
        "POST /v1/videos",
        public_name(caps),
        caps,
        UnknownFields::Reject,
    );
    let refused = Support::Refused;
    let frames = Domain::frames(&caps.frames);
    // H3 clips start at FastVideo's 5 s (`apply_model`).
    let h3_frames = || {
        let fps = caps.fps.default;
        let lo = caps
            .frames
            .next_on_grid(rules::H3_SECONDS.min_u32().saturating_mul(fps))
            .unwrap_or(caps.frames.min)
            .max(caps.frames.min);
        Domain::Frames {
            step: caps.frames.step,
            offset: caps.frames.offset,
            min: lo,
            max: caps.frames.max,
        }
    };
    let num_frames = if h3 { h3_frames() } else { frames.clone() };
    let seconds = seconds_domain(h3.then_some(rules::H3_SECONDS), caps);
    let canvas = Domain::canvas(&caps.canvas);
    let size_side = Domain::int_min(1);
    let refs = &caps.refs;
    s.push(
        p("prompt", T::String, "prompt")
            .required()
            .domain(Domain::non_blank(None)),
    )
    .push(
        p("model", T::String, "model")
            .nullable()
            .note("a served name, an alias or a tier id; default: the configured default model"),
    )
    .push(
        p(
            "seconds",
            T::IntegerOrString,
            "timing.length (seconds, aligned up to the grid)",
        )
        .nullable()
        .domain(seconds)
        .wire(Domain::int_min(1)),
    )
    .push(
        p("size", T::String, "canvas (exact)")
            .nullable()
            .domain(canvas.clone())
            .note("WIDTHxHEIGHT; wins over width/height, video_params and aspect_ratio"),
    )
    .push(
        p("width", T::Integer, "canvas.width")
            .nullable()
            .domain(canvas.clone())
            .wire(size_side.clone()),
    )
    .push(
        p("height", T::Integer, "canvas.height")
            .nullable()
            .domain(canvas.clone())
            .wire(size_side.clone()),
    )
    .push(
        p("fps", T::Integer, "timing.fps")
            .nullable()
            .domain(fps_domain(caps))
            .wire(Domain::int_min(1))
            .with_default(caps.fps.default),
    )
    .push(
        p("num_frames", T::Integer, "timing.length (frames, exact)")
            .nullable()
            .domain(num_frames.clone())
            .wire(Domain::int_min(1))
            .with_default(caps.frames.default),
    )
    .push(p("video_params", T::Object, "canvas, timing (vLLM-Omni)").nullable())
    .push(
        p("video_params.width", T::Integer, "canvas.width")
            .nullable()
            .domain(canvas.clone())
            .wire(size_side.clone()),
    )
    .push(
        p("video_params.height", T::Integer, "canvas.height")
            .nullable()
            .domain(canvas.clone())
            .wire(size_side.clone()),
    )
    .push(
        p(
            "video_params.num_frames",
            T::Integer,
            "timing.length (frames, exact)",
        )
        .nullable()
        .domain(num_frames)
        .wire(Domain::int_min(1)),
    )
    .push(
        p("video_params.fps", T::Integer, "timing.fps")
            .nullable()
            .domain(fps_domain(caps))
            .wire(Domain::int_min(1)),
    )
    .push(
        p("aspect_ratio", T::String, "canvas.ratio")
            .nullable()
            .domain(Domain::aspect(&caps.canvas)),
    )
    .push(
        p("short_edge", T::Integer, "canvas.short_edge")
            .nullable()
            .domain(short_edge_domain(caps))
            .wire(Domain::int_min(1))
            .note(
                "with aspect_ratio; alone, the tier an image-conditioned job follows its image at",
            ),
    );
    let ref2v = caps.supports(Task::Ref2V);
    let keyframes = rules::KEYFRAME_IMAGES.max_u32();
    let images_max = if ref2v {
        refs.images.max(keyframes)
    } else {
        keyframes
    };
    s.push(
        p(
            "image_reference",
            T::Array,
            "keyframes (1-2 images) or references",
        )
        .nullable()
        .domain(Domain::items(images_max))
        .note("one object or a list of {image_url | file_id}; file_id is a provider-file gap"),
    )
    .push(
        p("video_reference", T::Array, "references (video)")
            .nullable()
            .domain(Domain::items(refs.videos))
            .support(knob(ref2v && refs.videos > 0, refused)),
    )
    .push(
        p("audio_reference", T::Array, "references (audio)")
            .nullable()
            .domain(Domain::items(refs.audio))
            .support(knob(ref2v && refs.audio > 0, refused)),
    )
    .push(
        p("input_reference", T::String, "keyframes (first frame)")
            .nullable()
            .domain(Domain::media(MediaField::Image)),
    )
    .push(
        p("reference_url", T::String, "keyframes (first frame)")
            .nullable()
            .domain(Domain::media(MediaField::Image)),
    )
    .push(p("video_path", T::String, "-").nullable().support(refused))
    .push(p("video_url", T::String, "-").nullable().support(refused))
    .push(
        p("task", T::String, "task")
            .nullable()
            .domain(Domain::strings(&rules::TASKS))
            .support(knob(h3, refused))
            .note("only defined for H3"),
    );
    let outputs = Domain::int(fastvideo_protocol::IntRange::new(
        rules::OUTPUTS_SERVED,
        rules::OUTPUTS_SERVED,
    ));
    s.push(
        p("n", T::Integer, "-")
            .nullable()
            .domain(outputs.clone())
            .wire(Domain::int(rules::OUTPUTS)),
    )
    .push(
        p("num_outputs_per_prompt", T::Integer, "-")
            .domain(outputs)
            .wire(Domain::int(rules::OUTPUTS))
            .with_default(rules::OUTPUTS_SERVED),
    )
    .push(
        p("quality", T::String, "-")
            .nullable()
            .domain(Domain::strings(&rules::QUALITIES))
            .support(Support::NoOp),
    )
    .push(
        p("negative_prompt", T::String, "negative_prompt")
            .nullable()
            .support(if h3 {
                Support::NoOp
            } else {
                knob(k.negative, refused)
            }),
    )
    .push(
        p("num_inference_steps", T::Integer, "sampling.steps")
            .nullable()
            .domain(Domain::int(rules::NUM_INFERENCE_STEPS))
            .support(knob(
                k.steps,
                if h3 {
                    Support::Gap {
                        gap: GapId::PerRequestSteps,
                    }
                } else {
                    refused
                },
            )),
    )
    .push(
        p("guidance_scale", T::Number, "sampling.guidance")
            .nullable()
            .domain(Domain::num(rules::GUIDANCE_SCALE))
            .support(knob(k.guidance, refused))
            .note(if h3 {
                "1 is accepted as a no-op (the value FastH3 requires)"
            } else {
                ""
            }),
    )
    .push(
        p("guidance_scale_2", T::Number, "sampling.guidance_2")
            .nullable()
            .domain(Domain::num(rules::GUIDANCE_SCALE))
            .support(knob(k.guidance_2, refused)),
    )
    .push(
        p("boundary_ratio", T::Number, "sampling.boundary_ratio")
            .nullable()
            .domain(Domain::num(rules::BOUNDARY_RATIO))
            .support(knob(k.guidance_2, refused)),
    )
    .push(
        p("flow_shift", T::Number, "sampling.flow_shift")
            .nullable()
            .domain(Domain::num_above(0.0))
            .wire(Domain::Any)
            .support(knob(k.flow_shift, refused)),
    )
    .push(
        p("true_cfg_scale", T::Number, "-")
            .nullable()
            .support(refused),
    )
    .push(
        p("seed", T::Integer, "seed")
            .nullable()
            .domain(Domain::int_min(0))
            .support(knob(k.seed, refused))
            .note("upstream takes any int64; negative seeds are refused here"),
    )
    .push(
        p("max_sequence_length", T::Integer, "-")
            .nullable()
            .support(refused),
    )
    .push(
        p("enable_teacache", T::Boolean, "-")
            .with_default(false)
            .domain(Domain::one_of([false]))
            .wire(Domain::Any),
    )
    .push(
        p("generate_sound", T::Boolean, "-")
            .with_default(false)
            .support(Support::NoOp),
    )
    .push(
        p("sound_duration", T::Number, "-")
            .nullable()
            .support(refused),
    )
    .push(
        p("start_time_seconds", T::Number, "-")
            .nullable()
            .support(refused),
    )
    .push(
        p("enable_frame_interpolation", T::Boolean, "-")
            .with_default(false)
            .domain(Domain::one_of([false]))
            .wire(Domain::Any),
    )
    .push(
        p("frame_interpolation_exp", T::Any, "-")
            .nullable()
            .support(refused),
    )
    .push(
        p("frame_interpolation_scale", T::Any, "-")
            .nullable()
            .support(refused),
    )
    .push(
        p("frame_interpolation_model_path", T::Any, "-")
            .nullable()
            .support(refused),
    )
    .push(
        p("lora", T::Object, "-")
            .nullable()
            .support(Support::Gap { gap: GapId::Lora }),
    )
    .push(
        p("extra_params", T::Object, "-")
            .nullable()
            .support(refused)
            .note(format!(
                "known keys are fixed by the recipe: {}",
                EXTRA_PARAMS.join(", ")
            )),
    )
    .push(p("user", T::String, "-").nullable().support(Support::NoOp));
    for prm in &mut s.params {
        if prm.note.as_deref() == Some("") {
            prm.note = None;
        }
    }
    s
}

/// FastWan `POST /generate` for the model `caps` (the API has no model
/// field: every job runs the configured FastWan model).
pub fn fastwan_spec(caps: &ModelCaps) -> ParamSpec {
    use ParamType as T;
    let mut s = ParamSpec::new(
        "fastwan",
        "POST /generate",
        public_name(caps),
        caps,
        UnknownFields::Ignore,
    );
    let canvas = Domain::canvas(&caps.canvas);
    let q = |name: &str, ty: ParamType, maps_to: &str| {
        Param::new(name, ty, maps_to).upstream(FASTWAN_UPSTREAM)
    };
    s.push(
        q("prompt", T::String, "prompt")
            .required()
            .domain(Domain::non_blank(None)),
    )
    .push(
        q("width", T::Integer, "canvas.width")
            .nullable()
            .domain(canvas.clone())
            .wire(Domain::int_min(1)),
    )
    .push(
        q("height", T::Integer, "canvas.height")
            .nullable()
            .domain(canvas)
            .wire(Domain::int_min(1)),
    )
    .push(
        q("num_frames", T::Integer, "timing.length (frames, exact)")
            .nullable()
            .domain(Domain::frames(&caps.frames))
            .with_default(caps.frames.default),
    )
    .push(
        q("fps", T::Integer, "timing.fps")
            .nullable()
            .domain(fps_domain(caps))
            .wire(Domain::int_min(1))
            .with_default(caps.fps.default),
    )
    .push(
        q("seed", T::Integer, "seed")
            .nullable()
            .domain(Domain::int_min(0))
            .support(knob(caps.knobs.seed, Support::Refused)),
    );
    s
}

/// The specs of the mounted APIs for every served model: `/v1/videos` for
/// each model when `videos`, `/generate` for the FastWan model when
/// `fastwan_model` names one.
pub fn param_specs(
    models: &[ModelCaps],
    videos: bool,
    fastwan_model: Option<&str>,
) -> Vec<ParamSpec> {
    let mut out = Vec::new();
    if videos {
        out.extend(models.iter().map(videos_spec));
    }
    if let Some(name) = fastwan_model {
        if let Some(c) = models.iter().find(|m| m.answers_to(name)) {
            out.push(fastwan_spec(c));
        }
    }
    out
}
