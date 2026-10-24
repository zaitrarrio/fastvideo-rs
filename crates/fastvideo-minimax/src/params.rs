//! The [`ParamSpec`]s of `POST /v2/video_generation` (plan E2, ADR-0012):
//! one per MiniMax model name that resolves to a served model, from the
//! MiniMax contract (`docs/serve/specs/minimax/v2-video-generation.json`,
//! the `fastvideo_protocol::params::rules::minimax` constants the parser
//! validates with), the name's own rules ([`MiniMaxModel`]) and the caps.

use fastvideo_protocol::params::rules::minimax as rules;
use fastvideo_protocol::params::{
    knob, seconds_domain, Domain, MediaField, Param, ParamSpec, ParamType, Support, UnknownFields,
};
use fastvideo_protocol::{GapId, ModelCaps, Task};

use crate::create::{CONTENT_TYPES, ROLES};
use crate::{MiniMax, MiniMaxModel};

const UPSTREAM: &str =
    "docs/serve/specs/minimax/v2-video-generation.json#/components/schemas/VideoGenerationV2Req";

fn p(name: &str, ty: ParamType, maps_to: &str) -> Param {
    Param::new(name, ty, maps_to).upstream(UPSTREAM)
}

/// `POST /v2/video_generation` with `model: <model>` served by `caps`.
pub fn create_spec(model: MiniMaxModel, caps: &ModelCaps) -> ParamSpec {
    use ParamType as T;
    let mut s = ParamSpec::new(
        "minimax",
        "POST /v2/video_generation",
        model.as_str(),
        caps,
        UnknownFields::Ignore,
    );
    let names: Vec<&str> = MiniMaxModel::ALL.iter().map(|m| m.as_str()).collect();
    let listed: Vec<&str> = model.resolutions().iter().map(|r| r.as_str()).collect();
    let served: Vec<&str> = model
        .resolutions()
        .iter()
        .filter(|r| caps.canvas.short_edges.contains(&r.short_edge()))
        .map(|r| r.as_str())
        .collect();
    let refs = rules::REFERENCES;
    let ref2v = caps.supports(Task::Ref2V);
    let mut resolution = p("resolution", T::String, "canvas.short_edge")
        .domain(Domain::strings(&served))
        .wire(Domain::strings(&listed));
    if model.resolution_optional() {
        resolution = resolution
            .with_default("768P")
            .note("omitted: 768P, or the model's own default tier when it has no 768 tier");
    } else {
        resolution = resolution.required();
    }
    if listed.contains(&"2K") {
        resolution = resolution.note(format!("2K: Unsupported({})", GapId::H3Resolution2K.code()));
    }
    let durations = model.durations();
    s.push(p("model", T::String, "model").required().domain(Domain::strings(&names)))
        .push(
            p("content", T::Array, "prompt, keyframes, references")
                .required()
                .note("exactly one text item; first_frame / last_frame images or reference items, not both"),
        )
        .push(p("content[].type", T::String, "-").required().domain(Domain::strings(&CONTENT_TYPES)))
        .push(p("content[].role", T::String, "keyframes / references").domain(Domain::strings(&ROLES)))
        .push(
            p("content[].text", T::String, "prompt")
                .domain(Domain::non_blank(Some(rules::PROMPT_MAX_CHARS))),
        )
        .push(p("content[].image_url.url", T::String, "keyframes / references (image)").domain(Domain::media(MediaField::Image)))
        .push(
            p("content[].video_url.url", T::String, "references (video)")
                .domain(Domain::media(MediaField::Video))
                .support(knob(ref2v, Support::Refused)),
        )
        .push(
            p("content[].audio_url.url", T::String, "references (audio)")
                .domain(Domain::media(MediaField::Audio))
                .support(knob(ref2v, Support::Refused)),
        )
        .push(p("content(reference_image)", T::Array, "references (image)").domain(Domain::items(refs.images)).support(knob(ref2v, Support::Refused)))
        .push(p("content(reference_video)", T::Array, "references (video)").domain(Domain::items(refs.videos)).support(knob(ref2v, Support::Refused)))
        .push(p("content(reference_audio)", T::Array, "references (audio)").domain(Domain::items(refs.audio)).support(knob(ref2v, Support::Refused)))
        .push(p("content(references)", T::Array, "references").domain(Domain::items(refs.total)).support(knob(ref2v, Support::Refused)))
        .push(resolution)
        .push(
            p("duration", T::Integer, "timing.length (seconds, aligned up)")
                .required()
                .domain(seconds_domain(Some(durations), caps))
                .wire(Domain::int(durations)),
        )
        .push(
            p("ratio", T::String, "canvas.ratio")
                .with_default(fastvideo_protocol::params::rules::h3::ADAPTIVE)
                .domain(Domain::strings(&rules::RATIOS))
                .note("text-to-video needs a fixed ratio; image-to-video is always adaptive"),
        )
        .push(
            p("extra", T::Object, "prompt_expansion")
                .nullable()
                .support(knob(model.accepts_extra(), Support::Refused)),
        );
    if model.accepts_extra() {
        s.push(
            p("extra.prompt_expansion_mode", T::String, "prompt_expansion")
                .with_default("balanced")
                .domain(Domain::strings(&rules::PROMPT_EXPANSION_MODES))
                .note("applied by the server's upsampler when it has one, else a no-op"),
        );
    }
    s.push(p("callback_url", T::String, "callback").note("an http(s) URL the server may reach; blank is none"))
        .push(
            Param::new("prompt_optimizer", T::Boolean, "prompt_expansion")
                .note("MiniMax V1's switch (not in the V2 spec): opts in or out of the server's upsampler; a non-boolean is ignored"),
        );
    s
}

/// The spec of every MiniMax model name that resolves to a served model.
pub fn param_specs(
    mm: &MiniMax,
    models: &[ModelCaps],
    alias: &dyn Fn(&str) -> Option<String>,
) -> Vec<ParamSpec> {
    MiniMaxModel::ALL
        .iter()
        .filter_map(|m| {
            mm.resolve_caps(m.as_str(), alias, models)
                .ok()
                .map(|(c, _)| create_spec(*m, c))
        })
        .collect()
}
