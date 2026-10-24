//! Plan E2: the `/v1/videos` and FastWan parsers enforce exactly the
//! parse-time domains of their `ParamSpec`s (`fastvideo_protocol::params`).

use std::collections::BTreeSet;

use fastvideo_openai_videos::error::fastvideo_param;
use fastvideo_openai_videos::params::{fastwan_spec, videos_spec};
use fastvideo_openai_videos::videos::apply_model;
use fastvideo_openai_videos::videos::{merge_extra, parse_request};
use fastvideo_openai_videos::{FastWanSubmit, VideosCreate};
use fastvideo_protocol::params::{model_agreement, parse_agreement};
use fastvideo_protocol::{
    CanvasCaps, Family, FpsCaps, FrameGrid, KnobCaps, ModelCaps, ModelId, NormalizeCtx, RefLimits,
    SubmitEndpoint, Task,
};
use serde_json::{json, Map, Value};
use time::OffsetDateTime;

fn wan() -> ModelCaps {
    ModelCaps {
        id: ModelId("wan-max".into()),
        family: Family::Wan,
        served_names: vec!["wan-max".into()],
        tasks: [Task::T2V, Task::I2V].into_iter().collect::<BTreeSet<_>>(),
        audio: None,
        fps: FpsCaps {
            allowed: vec![16, 24],
            default: 24,
            container_only: true,
        },
        frames: FrameGrid::new(4, 1, 17, 161, 81),
        canvas: CanvasCaps {
            multiple: 32,
            max_area: 1280 * 704,
            aspect: (0.25, 4.0),
            short_edges: vec![704, 480],
            pad_and_crop: false,
            hd: None,
        },
        refs: RefLimits::none(),
        stream: None,
        knobs: KnobCaps::all(),
        resident: true,
        tier: None,
        recipe: None,
    }
}

fn base() -> Map<String, Value> {
    let Value::Object(m) = json!({"prompt": "a cat"}) else {
        unreachable!()
    };
    m
}

#[test]
fn videos_parser_enforces_the_spec() {
    let parse = |v: Value| {
        let Value::Object(mut m) = v else {
            unreachable!()
        };
        merge_extra(&mut m)?;
        let ncx = NormalizeCtx::new(OffsetDateTime::UNIX_EPOCH);
        VideosCreate.normalize(parse_request(m)?, &ncx).map(|_| ())
    };
    for caps in [ModelCaps::h3("h3", true), wan()] {
        let spec = videos_spec(&caps);
        // Checked with other fields (E3's mutual-exclusion cases): `task`
        // against the inputs (`fl2va` needs an image), `width` / `height`
        // together, `short_edge` with `aspect_ratio` or an image.
        let skip = ["task", "width", "height", "short_edge"];
        let bad = parse_agreement(&spec, &base(), &parse, &skip);
        assert!(bad.is_empty(), "{}: {bad:#?}", caps.id);
    }
}

#[test]
fn fastwan_parser_enforces_the_spec() {
    let submit = FastWanSubmit {
        model: Some("wan-max".into()),
    };
    let parse = |v: Value| {
        let body = serde_json::from_value(v)
            .map_err(|e| fastvideo_protocol::ApiError::invalid(e.to_string()))?;
        let ncx = NormalizeCtx::new(OffsetDateTime::UNIX_EPOCH);
        submit.normalize(body, &ncx).map(|_| ())
    };
    // `width` and `height` are checked together (one error on `width`).
    let bad = parse_agreement(&fastwan_spec(&wan()), &base(), &parse, &["width", "height"]);
    assert!(bad.is_empty(), "{bad:#?}");
}

/// The caps-derived domains are negotiation's own: parsing, the FastVideo
/// model rules (`apply_model`) and `negotiate::precheck` refuse exactly the
/// values outside them, naming the field as this API reports it.
#[test]
fn videos_domains_are_negotiations() {
    for caps in [ModelCaps::h3("h3", true), wan()] {
        let check = |v: Value| {
            let Value::Object(mut m) = v else {
                unreachable!()
            };
            merge_extra(&mut m)?;
            let ncx = NormalizeCtx::new(OffsetDateTime::UNIX_EPOCH);
            let mut req = VideosCreate.normalize(parse_request(m)?, &ncx)?;
            apply_model(&mut req, &caps, false)?;
            fastvideo_protocol::precheck(&req, &caps).map_err(|mut e| {
                e.param = e.param.as_deref().map(fastvideo_param);
                e
            })
        };
        let skip = ["task", "width", "height", "short_edge"];
        let bad = model_agreement(&videos_spec(&caps), &base(), &check, &skip);
        assert!(bad.is_empty(), "{}: {bad:#?}", caps.id);
    }
}

/// Plan E2's one behaviour change: `guidance_scale_2` is bounded to
/// FastVideo's `0..=20` at parse time (the `guidance_scale` wording), on
/// every model, ahead of `boundary_ratio`, the task checks and negotiation.
#[test]
fn guidance_scale_2_is_bounded_like_upstream() {
    let normalize = |v: Value| {
        let Value::Object(m) = v else { unreachable!() };
        let ncx = NormalizeCtx::new(OffsetDateTime::UNIX_EPOCH);
        VideosCreate.normalize(parse_request(m)?, &ncx)
    };
    for bad in [json!(20.5), json!(-1), json!(25)] {
        let e = normalize(json!({"prompt": "a cat", "guidance_scale_2": bad, "boundary_ratio": 2}))
            .unwrap_err();
        assert_eq!(
            e.kind,
            fastvideo_protocol::ErrorKind::InvalidRequest,
            "{bad}"
        );
        assert_eq!(e.param.as_deref(), Some("guidance_scale_2"));
        assert_eq!(e.message, "guidance_scale_2 must be within 0..=20");
        let r = fastvideo_openai_videos::error::openai_error(&e);
        assert_eq!(r.status, 400);
        assert_eq!(
            r.json_body().unwrap(),
            &json!({"error": {
                "message": "guidance_scale_2 must be within 0..=20",
                "type": "invalid_request_error",
                "param": "guidance_scale_2",
                "code": 400,
            }})
        );
    }
    for ok in [json!(0), json!(20), json!(7.5)] {
        assert!(
            normalize(json!({"prompt": "a cat", "guidance_scale_2": ok})).is_ok(),
            "{ok}"
        );
    }
}
