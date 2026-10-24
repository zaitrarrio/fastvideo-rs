//! `Request::validate` agrees with the serve path's `negotiate::precheck`
//! (plan B2 review): over a grid of requests, against every served model's
//! real caps (the `/fv/v1/capabilities` golden of every shipped serve config,
//! `crates/fastvideo-serve/tests/goldens/capabilities.json`, which A3 pins to
//! the projection of `families/*.toml`), each request is stated twice: as a
//! `Request` and, independently, as the `GenerationRequest` an API adapter
//! would build. `validate` must accept exactly what `precheck` accepts and
//! refuse with the same error.

use std::collections::BTreeMap;
use std::path::PathBuf;

use fastvideo_core::{Canvas, Conditioning, Request, RequestError, Sampler, Timing};
use fastvideo_protocol::caps::ModelCaps;
use fastvideo_protocol::negotiate::precheck;
use fastvideo_protocol::request::{
    Anchor, AudioInput, AudioRole, CanvasSpec, GenerationRequest, Keyframe, Length, MediaKind,
    MediaRef, ProtocolId, Reference, Snap, Task,
};

/// Every distinct served model in the capabilities golden.
fn served_caps() -> Vec<ModelCaps> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../fastvideo-serve/tests/goldens/capabilities.json");
    let text =
        std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    let golden: serde_json::Value =
        serde_json::from_str(&text).unwrap_or_else(|e| panic!("parse golden: {e}"));
    let mut by_id: BTreeMap<String, ModelCaps> = BTreeMap::new();
    for config in golden.as_object().into_iter().flat_map(|o| o.values()) {
        let models = config["capabilities"]["models"]
            .as_array()
            .cloned()
            .unwrap_or_default();
        for m in models {
            let caps: ModelCaps = serde_json::from_value(m["caps"].clone())
                .unwrap_or_else(|e| panic!("caps of {}: {e}", m["caps"]["id"]));
            by_id.entry(caps.id.0.clone()).or_insert(caps);
        }
    }
    by_id.into_values().collect()
}

#[derive(Clone, Copy, Debug)]
enum Cond {
    Text,
    Image,
    FirstLast,
    LastOnly,
    RefImage,
    RefVideos(u32),
    Audio,
}

#[derive(Clone, Copy, Debug)]
enum Knob {
    None,
    Seed,
    Negative,
    Steps,
    Guidance,
    Guidance2,
}

fn media(p: &str) -> MediaRef {
    MediaRef::ProviderFile(p.to_string())
}

/// The request as an API adapter states it.
fn adapter_request(
    model: &str,
    canvas: Option<(u32, u32)>,
    frames: Option<u32>,
    seconds: Option<u32>,
    fps: Option<u32>,
    cond: Cond,
    knob: Knob,
) -> GenerationRequest {
    let mut g = GenerationRequest::text(ProtocolId::Native, model, "a raccoon");
    g.canvas = match canvas {
        Some((width, height)) => CanvasSpec::Exact { width, height },
        None => CanvasSpec::ModelDefault,
    };
    g.timing.length = match (frames, seconds) {
        (Some(value), _) => Length::Frames {
            value,
            snap: Snap::Exact,
        },
        (None, Some(s)) => Length::Seconds {
            value: f64::from(s),
            snap: Snap::AlignUp,
        },
        (None, None) => Length::ModelDefault,
    };
    g.timing.fps = fps;
    let first = Keyframe {
        at: Anchor::First,
        image: media("first.png"),
    };
    let last = Keyframe {
        at: Anchor::Last,
        image: media("last.png"),
    };
    match cond {
        Cond::Text => {}
        Cond::Image => {
            g.task = Task::I2V;
            g.keyframes = vec![first];
        }
        Cond::FirstLast => {
            g.task = Task::Keyframes;
            g.keyframes = vec![first, last];
        }
        Cond::LastOnly => {
            g.task = Task::Keyframes;
            g.keyframes = vec![last];
        }
        Cond::RefImage => {
            g.task = Task::Ref2V;
            g.references = vec![Reference {
                kind: MediaKind::Image,
                media: media("ref.png"),
            }];
        }
        Cond::RefVideos(n) => {
            g.task = Task::Ref2V;
            g.references = (0..n)
                .map(|i| Reference {
                    kind: MediaKind::Video,
                    media: media(&format!("ref{i}.mp4")),
                })
                .collect();
        }
        Cond::Audio => {
            g.task = Task::A2V;
            g.audio_in = Some(AudioInput {
                media: media("drive.wav"),
                role: AudioRole::Drive,
                max_s: None,
            });
        }
    }
    match knob {
        Knob::None => {}
        Knob::Seed => g.seed = Some(7),
        Knob::Negative => g.negative_prompt = Some("blurry".into()),
        Knob::Steps => g.sampling.steps = Some(4),
        Knob::Guidance => g.sampling.guidance = Some(3.0),
        Knob::Guidance2 => g.sampling.guidance_2 = Some(2.0),
    }
    g
}

/// The same request as a `Request`.
fn request(
    canvas: Option<(u32, u32)>,
    frames: Option<u32>,
    seconds: Option<u32>,
    fps: Option<u32>,
    cond: Cond,
    knob: Knob,
) -> Request {
    let conditioning = match cond {
        Cond::Text => vec![],
        Cond::Image => vec![Conditioning::Image("first.png".into())],
        Cond::FirstLast => vec![Conditioning::Keyframes {
            first: Some("first.png".into()),
            last: "last.png".into(),
        }],
        Cond::LastOnly => vec![Conditioning::Keyframes {
            first: None,
            last: "last.png".into(),
        }],
        Cond::RefImage => vec![Conditioning::Refs(vec!["ref.png".into()])],
        Cond::RefVideos(n) => vec![Conditioning::Refs(
            (0..n)
                .map(|i| PathBuf::from(format!("ref{i}.mp4")))
                .collect(),
        )],
        Cond::Audio => vec![Conditioning::Audio("drive.wav".into())],
    };
    let mut r = Request {
        prompt: "a raccoon".into(),
        canvas: Canvas {
            width: canvas.map(|c| c.0),
            height: canvas.map(|c| c.1),
        },
        timing: Timing {
            frames,
            seconds,
            fps,
        },
        sampler: Sampler::default(),
        conditioning,
        ..Request::default()
    };
    match knob {
        Knob::None => {}
        Knob::Seed => r.seed = Some(7),
        Knob::Negative => r.negative = Some("blurry".into()),
        Knob::Steps => r.sampler.steps = Some(4),
        Knob::Guidance => r.sampler.guidance = Some(3.0),
        Knob::Guidance2 => r.sampler.guidance_2 = Some(2.0),
    }
    r
}

const CANVASES: &[Option<(u32, u32)>] = &[
    None,
    Some((1344, 768)),
    Some((768, 1344)),
    // A short edge between tiers: H3's check_canvas accepts it.
    Some((512, 1344)),
    Some((1920, 1080)),
    Some((1920, 1088)),
    Some((1088, 1920)),
    Some((832, 480)),
    Some((1280, 704)),
    Some((1280, 720)),
    Some((704, 1280)),
    Some((830, 480)),
    Some((768, 512)),
    Some((2560, 480)),
    Some((3840, 2160)),
];
const LENGTHS: &[(Option<u32>, Option<u32>)] = &[
    (None, None),
    (Some(81), None),
    (Some(121), None),
    (Some(124), None),
    (Some(241), None),
    (Some(80), None),
    (Some(5), None),
    (None, Some(5)),
    (None, Some(10)),
    (None, Some(15)),
    (None, Some(20)),
];
const FPS: &[Option<u32>] = &[None, Some(24), Some(16), Some(30)];
const CONDS: &[Cond] = &[
    Cond::Text,
    Cond::Image,
    Cond::FirstLast,
    Cond::LastOnly,
    Cond::RefImage,
    Cond::RefVideos(4),
    Cond::Audio,
];
const KNOBS: &[Knob] = &[
    Knob::None,
    Knob::Seed,
    Knob::Negative,
    Knob::Steps,
    Knob::Guidance,
    Knob::Guidance2,
];

#[test]
fn validate_agrees_with_precheck_on_every_served_model() {
    let models = served_caps();
    assert!(models.len() >= 30, "only {} served models", models.len());
    let (mut accepted, mut refused) = (0usize, 0usize);
    for caps in &models {
        let model = caps.id.0.as_str();
        for &canvas in CANVASES {
            for &(frames, seconds) in LENGTHS {
                for &fps in FPS {
                    for &cond in CONDS {
                        for &knob in KNOBS {
                            let want = precheck(
                                &adapter_request(model, canvas, frames, seconds, fps, cond, knob),
                                caps,
                            );
                            let got =
                                request(canvas, frames, seconds, fps, cond, knob).validate(caps);
                            let case = || {
                                format!(
                                    "{model}: canvas {canvas:?} frames {frames:?} seconds \
                                     {seconds:?} fps {fps:?} {cond:?} {knob:?}"
                                )
                            };
                            match (&want, &got) {
                                (Ok(()), Ok(())) => accepted += 1,
                                (Err(w), Err(RequestError::Negotiate(g))) if w == g => refused += 1,
                                _ => panic!("{}: precheck {want:?}, validate {got:?}", case()),
                            }
                        }
                    }
                }
            }
        }
    }
    eprintln!(
        "validate_parity: {} served models, {accepted} accepted, {refused} refused",
        models.len()
    );
    assert!(accepted > 1000 && refused > 1000, "{accepted} / {refused}");
}

/// The cases the review named, on the real caps.
#[test]
fn named_cases_on_real_caps() {
    let models = served_caps();
    let caps = |id: &str| {
        models
            .iter()
            .find(|c| c.id.0 == id)
            .unwrap_or_else(|| panic!("no served model {id}"))
    };
    let at = |w: u32, h: u32| request(Some((w, h)), None, None, None, Cond::Text, Knob::None);
    // H3: a 512 short edge between tiers (h3::check_canvas accepts it).
    let h3 = caps("fasth3");
    assert_eq!(at(1344, 512).validate(h3), Ok(()));
    assert_eq!(at(512, 1344).validate(h3), Ok(()));
    // The 1080P tier, served by `fasth3` and `sol-h3`.
    assert!(h3.canvas.hd.is_some());
    assert_eq!(at(1920, 1080).validate(h3), Ok(()));
    assert_eq!(at(1088, 1920).validate(h3), Ok(()));
    // ... and its length cap (5 s by default).
    let long = request(
        Some((1920, 1080)),
        None,
        Some(15),
        None,
        Cond::Text,
        Knob::None,
    );
    assert!(matches!(long.validate(h3), Err(RequestError::Negotiate(_))));
    // Every case agrees with precheck.
    for (w, h) in [(1344, 512), (512, 1344), (1920, 1080), (1088, 1920)] {
        let g = adapter_request(
            "fasth3",
            Some((w, h)),
            None,
            None,
            None,
            Cond::Text,
            Knob::None,
        );
        assert_eq!(precheck(&g, h3), Ok(()), "{w}x{h}");
    }
}
