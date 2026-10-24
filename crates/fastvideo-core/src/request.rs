//! The common generation request (plan B2, review §7.3 item 2).
//!
//! One [`Request`] describes what to generate for every family: the canvas,
//! the timing, the sampler knobs and the conditioning inputs. A field left
//! `None` means "the variant's default"; the family fills it in. Where the
//! model's capabilities are known (the served variants: `families/*.toml`
//! projected to [`ModelCaps`]), [`Request::validate`] checks the request
//! against them once, before any weight is touched. It does not re-derive
//! the rules: it states the request as the protocol's
//! [`GenerationRequest`] ([`Request::to_generation_request`]) and runs the
//! serve path's own [`negotiate::precheck`] (canvas tiers and pixel budgets,
//! pad-and-crop, the HD tier and its length cap, H3 geometry, the frame
//! grid, fps, references, knobs, audio).
//!
//! The CLI builds a `Request` from its flags and does not call
//! [`Request::validate`]: its presets have no caps (only the served variants
//! do), and the CLI has always accepted what each family's own checks accept
//! (the tiny CPU goldens run at 64x64x5, below every served frame grid).

use std::path::{Path, PathBuf};

use fastvideo_models::h3::reference::{infer_reference_kind, ReferenceKind};
use fastvideo_protocol::caps::ModelCaps;
use fastvideo_protocol::error::ApiError;
use fastvideo_protocol::negotiate;
use fastvideo_protocol::request::{
    Anchor, AudioInput, AudioRole, CanvasSpec, GenerationRequest, Keyframe, Length, MediaKind,
    MediaRef, ProtocolId, Reference, SamplingOverrides, Snap, Task, TimingSpec,
};
use thiserror::Error;

/// Output size in pixels.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Canvas {
    pub width: Option<u32>,
    pub height: Option<u32>,
}

/// Clip length and rate. `frames` and `seconds` are alternatives; a family
/// that knows both (H3) applies `seconds` first and `frames` over it.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Timing {
    pub frames: Option<u32>,
    /// Whole seconds (H3 `--seconds`, 5..=15).
    pub seconds: Option<u32>,
    pub fps: Option<u32>,
}

/// Sampler knobs.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Sampler {
    pub steps: Option<u32>,
    pub guidance: Option<f32>,
    /// The second guidance scale of a two-expert model (Wan 2.2 MoE, the
    /// low-noise expert): the protocol's `SamplingOverrides::guidance_2`.
    pub guidance_2: Option<f32>,
    /// The audio branch's guidance (LTX-2). No served model takes it per
    /// request (the protocol has no such knob; the LTX engine pins it), so
    /// [`Request::validate`] refuses it; the CLI's `--guidance-2` sets it for
    /// the families whose second guidance is the audio one
    /// ([`crate::family::VideoFamily::second_guidance_is_audio`]).
    pub audio_guidance: Option<f32>,
}

/// One conditioning input. A request carries zero (text only) or more; the
/// combinations a model takes are what [`Request::validate`] checks.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Conditioning {
    /// A first-frame image (I2V). MMAudio reads it as the source video (V2A).
    Image(PathBuf),
    /// A last frame, with an optional first frame (H3 FL2VA).
    Keyframes {
        first: Option<PathBuf>,
        last: PathBuf,
    },
    /// Ordered references: image / video / audio by extension (H3 Ref2VA).
    Refs(Vec<PathBuf>),
    /// A control or reference frame (Wan Fun Control, Lucy edit).
    Control(PathBuf),
    /// A driving audio track (LTX audio-to-video).
    Audio(PathBuf),
}

impl Conditioning {
    /// The task this input asks of the model.
    pub fn task(&self) -> Task {
        match self {
            Self::Image(_) => Task::I2V,
            Self::Keyframes { .. } => Task::Keyframes,
            Self::Refs(_) => Task::Ref2V,
            Self::Control(_) => Task::V2V,
            Self::Audio(_) => Task::A2V,
        }
    }

    fn kind(&self) -> &'static str {
        match self {
            Self::Image(_) => "image",
            Self::Keyframes { .. } => "keyframes",
            Self::Refs(_) => "refs",
            Self::Control(_) => "control",
            Self::Audio(_) => "audio",
        }
    }
}

/// What to generate.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Request {
    pub prompt: String,
    pub negative: Option<String>,
    pub seed: Option<u64>,
    pub canvas: Canvas,
    pub timing: Timing,
    pub sampler: Sampler,
    pub conditioning: Vec<Conditioning>,
}

/// Why a request does not fit a model.
#[derive(Debug, Clone, PartialEq, Error)]
pub enum RequestError {
    #[error("more than one {0} input")]
    Duplicate(&'static str),
    #[error("pass either an image/keyframes or references, not both")]
    ImageAndRefs,
    #[error("{0} and {1} inputs do not combine in one request")]
    Conflict(&'static str, &'static str),
    #[error("set width and height together (got {width:?} x {height:?})")]
    HalfCanvas {
        width: Option<u32>,
        height: Option<u32>,
    },
    #[error("{model} does not take a per-request `{knob}`")]
    Knob { model: String, knob: &'static str },
    /// What the serve path's negotiation refuses.
    #[error(transparent)]
    Negotiate(#[from] ApiError),
}

impl Request {
    /// A text-only request with every other field at the variant default.
    pub fn text(prompt: impl Into<String>) -> Self {
        Self {
            prompt: prompt.into(),
            ..Self::default()
        }
    }

    /// The conditioning from the CLI's inputs: `--image` alone is I2V,
    /// `--last-image` (with or without `--image`) keyframes, `--ref` the
    /// ordered references, `--control` a control frame. Inputs are kept as
    /// given; the family (or [`Self::validate`]) decides which it takes.
    pub fn conditioning_from_inputs(
        image: Option<PathBuf>,
        last_image: Option<PathBuf>,
        references: Vec<PathBuf>,
        control: Option<PathBuf>,
    ) -> Vec<Conditioning> {
        let mut out = Vec::new();
        match (image, last_image) {
            (first, Some(last)) => out.push(Conditioning::Keyframes { first, last }),
            (Some(first), None) => out.push(Conditioning::Image(first)),
            (None, None) => {}
        }
        if !references.is_empty() {
            out.push(Conditioning::Refs(references));
        }
        if let Some(c) = control {
            out.push(Conditioning::Control(c));
        }
        out
    }

    /// The first frame: an I2V image or the keyframes' first frame.
    pub fn first_image(&self) -> Option<&Path> {
        self.conditioning.iter().find_map(|c| match c {
            Conditioning::Image(p) => Some(p.as_path()),
            Conditioning::Keyframes { first, .. } => first.as_deref(),
            _ => None,
        })
    }

    /// The keyframes' last frame.
    pub fn last_image(&self) -> Option<&Path> {
        self.conditioning.iter().find_map(|c| match c {
            Conditioning::Keyframes { last, .. } => Some(last.as_path()),
            _ => None,
        })
    }

    /// The ordered references (empty without any).
    pub fn references(&self) -> &[PathBuf] {
        self.conditioning
            .iter()
            .find_map(|c| match c {
                Conditioning::Refs(r) => Some(r.as_slice()),
                _ => None,
            })
            .unwrap_or(&[])
    }

    /// The control frame.
    pub fn control(&self) -> Option<&Path> {
        self.conditioning.iter().find_map(|c| match c {
            Conditioning::Control(p) => Some(p.as_path()),
            _ => None,
        })
    }

    /// The driving audio track.
    pub fn audio(&self) -> Option<&Path> {
        self.conditioning.iter().find_map(|c| match c {
            Conditioning::Audio(p) => Some(p.as_path()),
            _ => None,
        })
    }

    /// The protocol task this request asks for: text only is `T2V`; a
    /// driving audio track `A2V` (with an optional first frame or
    /// keyframes); references `Ref2V`; a control frame `V2V` (with an
    /// optional image, as Wan Fun Control reads it); keyframes `Keyframes`;
    /// an image `I2V`. Any other combination is refused.
    pub fn task(&self) -> Result<Task, RequestError> {
        let mut kinds: Vec<&'static str> = Vec::new();
        for c in &self.conditioning {
            if kinds.contains(&c.kind()) {
                return Err(RequestError::Duplicate(c.kind()));
            }
            kinds.push(c.kind());
        }
        let has = |k: &str| kinds.contains(&k);
        let framed = has("image") || has("keyframes");
        if has("refs") && framed {
            return Err(RequestError::ImageAndRefs);
        }
        if has("image") && has("keyframes") {
            return Err(RequestError::Conflict("image", "keyframes"));
        }
        let main: Vec<&'static str> = ["audio", "refs", "control"]
            .into_iter()
            .filter(|k| has(k))
            .collect();
        if let [a, b, ..] = main[..] {
            return Err(RequestError::Conflict(a, b));
        }
        if has("control") && has("keyframes") {
            return Err(RequestError::Conflict("control", "keyframes"));
        }
        Ok(match main.first().copied() {
            Some("audio") => Task::A2V,
            Some("refs") => Task::Ref2V,
            Some("control") => Task::V2V,
            _ if has("keyframes") => Task::Keyframes,
            _ if has("image") => Task::I2V,
            _ => Task::T2V,
        })
    }

    /// The request as the protocol states it, for model `caps`: what an API
    /// adapter would hand to `negotiate`. Local paths become
    /// [`MediaRef::ProviderFile`] (the precheck reads only kinds and counts);
    /// `frames` is an exact count, `seconds` aligns up to the frame grid (as
    /// H3's seconds always did), `frames` wins when both are set.
    pub fn to_generation_request(
        &self,
        caps: &ModelCaps,
    ) -> Result<GenerationRequest, RequestError> {
        let task = self.task()?;
        let media = |p: &Path| MediaRef::ProviderFile(p.display().to_string());
        let mut req = GenerationRequest::text(ProtocolId::Native, caps.id.0.clone(), &self.prompt);
        req.task = task;
        req.negative_prompt = self.negative.clone();
        req.seed = self.seed;
        req.canvas = match (self.canvas.width, self.canvas.height) {
            (Some(width), Some(height)) => CanvasSpec::Exact { width, height },
            (None, None) => CanvasSpec::ModelDefault,
            (width, height) => return Err(RequestError::HalfCanvas { width, height }),
        };
        req.timing = TimingSpec {
            length: match (self.timing.frames, self.timing.seconds) {
                (Some(value), _) => Length::Frames {
                    value,
                    snap: Snap::Exact,
                },
                (None, Some(s)) => Length::Seconds {
                    value: f64::from(s),
                    snap: Snap::AlignUp,
                },
                (None, None) => Length::ModelDefault,
            },
            fps: self.timing.fps,
        };
        for c in &self.conditioning {
            match c {
                Conditioning::Image(p) => req.keyframes.push(Keyframe {
                    at: Anchor::First,
                    image: media(p),
                }),
                Conditioning::Keyframes { first, last } => {
                    if let Some(f) = first {
                        req.keyframes.push(Keyframe {
                            at: Anchor::First,
                            image: media(f),
                        });
                    }
                    req.keyframes.push(Keyframe {
                        at: Anchor::Last,
                        image: media(last),
                    });
                }
                Conditioning::Refs(refs) => {
                    req.references = refs
                        .iter()
                        .map(|r| Reference {
                            kind: media_kind(r),
                            media: media(r),
                        })
                        .collect();
                }
                // The control frame rides as the edit source's first frame;
                // V2V is refused by the protocol today either way.
                Conditioning::Control(p) => {
                    if req.keyframes.is_empty() {
                        req.keyframes.push(Keyframe {
                            at: Anchor::First,
                            image: media(p),
                        });
                    }
                }
                Conditioning::Audio(p) => {
                    req.audio_in = Some(AudioInput {
                        media: media(p),
                        role: AudioRole::Drive,
                        max_s: None,
                    });
                }
            }
        }
        req.sampling = SamplingOverrides {
            steps: self.sampler.steps,
            guidance: self.sampler.guidance,
            guidance_2: self.sampler.guidance_2,
            ..SamplingOverrides::default()
        };
        Ok(req)
    }

    /// Check the request against a model's capabilities with the serve
    /// path's own rules ([`negotiate::precheck`]), after the checks the
    /// protocol cannot state: conflicting inputs, a half-set canvas, and the
    /// audio guidance no served model takes.
    pub fn validate(&self, caps: &ModelCaps) -> Result<(), RequestError> {
        let req = self.to_generation_request(caps)?;
        if self.sampler.audio_guidance.is_some() {
            return Err(RequestError::Knob {
                model: caps.id.0.clone(),
                knob: "audio_guidance",
            });
        }
        negotiate::precheck(&req, caps)?;
        Ok(())
    }
}

fn media_kind(path: &Path) -> MediaKind {
    match infer_reference_kind(path) {
        ReferenceKind::Image => MediaKind::Image,
        ReferenceKind::Video => MediaKind::Video,
        ReferenceKind::Audio => MediaKind::Audio,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn h3() -> ModelCaps {
        ModelCaps::h3("h3-test", true)
    }

    #[test]
    fn text_request_at_defaults_fits() {
        assert_eq!(Request::text("a cat").validate(&h3()), Ok(()));
    }

    #[test]
    fn conditioning_from_cli_inputs_keeps_every_input() {
        let c = Request::conditioning_from_inputs(
            Some("a.png".into()),
            Some("b.png".into()),
            vec!["r.png".into()],
            Some("c.png".into()),
        );
        assert_eq!(
            c,
            vec![
                Conditioning::Keyframes {
                    first: Some("a.png".into()),
                    last: "b.png".into()
                },
                Conditioning::Refs(vec!["r.png".into()]),
                Conditioning::Control("c.png".into()),
            ]
        );
        let r = Request {
            conditioning: c,
            ..Request::default()
        };
        assert_eq!(r.first_image(), Some(Path::new("a.png")));
        assert_eq!(r.last_image(), Some(Path::new("b.png")));
        assert_eq!(r.references(), &[PathBuf::from("r.png")]);
        assert_eq!(r.control(), Some(Path::new("c.png")));
        assert_eq!(
            Request::conditioning_from_inputs(Some("a.png".into()), None, vec![], None),
            vec![Conditioning::Image("a.png".into())]
        );
        assert!(Request::conditioning_from_inputs(None, None, vec![], None).is_empty());
    }

    #[test]
    fn task_follows_the_conditioning() {
        let with = |c: Vec<Conditioning>| Request {
            conditioning: c,
            ..Request::text("x")
        };
        assert_eq!(with(vec![]).task(), Ok(Task::T2V));
        assert_eq!(
            with(vec![Conditioning::Image("a.png".into())]).task(),
            Ok(Task::I2V)
        );
        assert_eq!(
            with(vec![Conditioning::Keyframes {
                first: None,
                last: "b.png".into()
            }])
            .task(),
            Ok(Task::Keyframes)
        );
        assert_eq!(
            with(vec![Conditioning::Refs(vec!["r.png".into()])]).task(),
            Ok(Task::Ref2V)
        );
        assert_eq!(
            with(vec![
                Conditioning::Audio("a.wav".into()),
                Conditioning::Image("a.png".into())
            ])
            .task(),
            Ok(Task::A2V)
        );
        assert_eq!(
            with(vec![
                Conditioning::Image("a.png".into()),
                Conditioning::Refs(vec!["r.png".into()])
            ])
            .task(),
            Err(RequestError::ImageAndRefs)
        );
        assert_eq!(
            with(vec![
                Conditioning::Image("a.png".into()),
                Conditioning::Image("b.png".into())
            ])
            .task(),
            Err(RequestError::Duplicate("image"))
        );
        assert_eq!(
            with(vec![
                Conditioning::Audio("a.wav".into()),
                Conditioning::Refs(vec!["r.png".into()])
            ])
            .task(),
            Err(RequestError::Conflict("audio", "refs"))
        );
    }

    #[test]
    fn a_half_set_canvas_is_refused() {
        let mut r = Request::text("x");
        r.canvas.width = Some(832);
        assert_eq!(
            r.validate(&h3()),
            Err(RequestError::HalfCanvas {
                width: Some(832),
                height: None
            })
        );
    }

    #[test]
    fn audio_guidance_is_refused_and_guidance_2_is_the_moe_knob() {
        let mut r = Request::text("x");
        r.sampler.audio_guidance = Some(7.0);
        assert!(matches!(
            r.validate(&h3()),
            Err(RequestError::Knob {
                knob: "audio_guidance",
                ..
            })
        ));
        let mut r = Request::text("x");
        r.sampler.guidance_2 = Some(3.0);
        let req = r
            .to_generation_request(&h3())
            .unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(req.sampling.guidance_2, Some(3.0));
        assert!(matches!(r.validate(&h3()), Err(RequestError::Negotiate(_))));
    }

    #[test]
    fn the_protocol_statement_carries_every_field() {
        let r = Request {
            prompt: "p".into(),
            negative: Some("n".into()),
            seed: Some(7),
            canvas: Canvas {
                width: Some(1344),
                height: Some(768),
            },
            timing: Timing {
                frames: None,
                seconds: Some(5),
                fps: Some(24),
            },
            sampler: Sampler {
                steps: Some(4),
                guidance: Some(1.5),
                guidance_2: None,
                audio_guidance: None,
            },
            conditioning: vec![Conditioning::Keyframes {
                first: Some("a.png".into()),
                last: "b.png".into(),
            }],
        };
        let g = r
            .to_generation_request(&h3())
            .unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(g.model, "h3-test");
        assert_eq!(g.task, Task::Keyframes);
        assert_eq!(g.negative_prompt.as_deref(), Some("n"));
        assert_eq!(g.seed, Some(7));
        assert_eq!(
            g.canvas,
            CanvasSpec::Exact {
                width: 1344,
                height: 768
            }
        );
        assert_eq!(
            g.timing.length,
            Length::Seconds {
                value: 5.0,
                snap: Snap::AlignUp
            }
        );
        assert_eq!(g.timing.fps, Some(24));
        assert_eq!(
            g.keyframes.iter().map(|k| k.at).collect::<Vec<_>>(),
            vec![Anchor::First, Anchor::Last]
        );
        assert_eq!(g.sampling.steps, Some(4));
        assert_eq!(g.sampling.guidance, Some(1.5));
        let refs = Request {
            conditioning: vec![Conditioning::Refs(vec![
                "a.png".into(),
                "b.mp4".into(),
                "c.wav".into(),
            ])],
            ..Request::text("x")
        };
        let g = refs
            .to_generation_request(&h3())
            .unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(
            g.references.iter().map(|r| r.kind).collect::<Vec<_>>(),
            vec![MediaKind::Image, MediaKind::Video, MediaKind::Audio]
        );
    }
}
