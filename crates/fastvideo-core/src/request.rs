//! The common generation request (plan B2, review §7.3 item 2).
//!
//! One [`Request`] describes what to generate for every family: the canvas,
//! the timing, the sampler knobs and the conditioning inputs. A field left
//! `None` means "the variant's default"; the family fills it in. Where the
//! model's capabilities are known (the served variants: `families/*.toml`
//! projected to [`ModelCaps`]), [`Request::validate`] checks the request
//! against them once, before any weight is touched.
//!
//! The CLI builds a `Request` from its flags and does not call
//! [`Request::validate`]: its presets have no caps (only the served variants
//! do), and the CLI has always accepted what each family's own checks accept
//! (the tiny CPU goldens run at 64x64x5, below every served frame grid).

use std::path::{Path, PathBuf};

use fastvideo_models::h3::reference::{infer_reference_kind, ReferenceKind};
use fastvideo_protocol::caps::ModelCaps;
use fastvideo_protocol::request::Task;
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
    /// The second guidance scale: the Wan 2.2 low-noise expert, the LTX audio
    /// branch.
    pub guidance_2: Option<f32>,
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
    #[error("{model} does not take {task:?} requests")]
    Task { model: String, task: Task },
    #[error("more than one {0} input")]
    Duplicate(&'static str),
    #[error("pass either an image/keyframes or references, not both")]
    ImageAndRefs,
    #[error("{model}: {width}x{height} {why}")]
    Canvas {
        model: String,
        width: u32,
        height: u32,
        why: String,
    },
    #[error("{model}: {frames} frames is not on the grid {offset}+{step}k within {min}..={max}")]
    Frames {
        model: String,
        frames: u32,
        offset: u32,
        step: u32,
        min: u32,
        max: u32,
    },
    #[error("{model}: {seconds} s at {fps} fps is outside the frame range {min}..={max}")]
    Seconds {
        model: String,
        seconds: u32,
        fps: u32,
        min: u32,
        max: u32,
    },
    #[error("{model}: {fps} fps is not one of {allowed:?}")]
    Fps {
        model: String,
        fps: u32,
        allowed: Vec<u32>,
    },
    #[error("{model}: {count} {kind} references, at most {max}")]
    Refs {
        model: String,
        kind: &'static str,
        count: u32,
        max: u32,
    },
    #[error("{model} does not honour `{knob}`")]
    Knob { model: String, knob: &'static str },
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

    /// The tasks this request asks of the model: `T2V` for text only, else
    /// one per conditioning input.
    pub fn tasks(&self) -> Vec<Task> {
        if self.conditioning.is_empty() {
            return vec![Task::T2V];
        }
        self.conditioning.iter().map(Conditioning::task).collect()
    }

    /// Check the request against a model's capabilities: the tasks, the
    /// canvas (multiple, aspect, pixel budget), the frame grid, the duration,
    /// the fps, the reference counts and the knobs it sets. Fields left at
    /// the default are not checked (the variant's default fits by
    /// construction).
    pub fn validate(&self, caps: &ModelCaps) -> Result<(), RequestError> {
        let model = || caps.id.0.clone();
        let mut kinds: Vec<&'static str> = Vec::new();
        for c in &self.conditioning {
            if kinds.contains(&c.kind()) {
                return Err(RequestError::Duplicate(c.kind()));
            }
            kinds.push(c.kind());
        }
        if kinds.contains(&"refs") && (kinds.contains(&"image") || kinds.contains(&"keyframes")) {
            return Err(RequestError::ImageAndRefs);
        }
        for task in self.tasks() {
            if !caps.supports(task) {
                return Err(RequestError::Task {
                    model: model(),
                    task,
                });
            }
        }
        self.validate_canvas(caps)?;
        let grid = &caps.frames;
        if let Some(frames) = self.timing.frames {
            if !grid.contains(frames) {
                return Err(RequestError::Frames {
                    model: model(),
                    frames,
                    offset: grid.offset,
                    step: grid.step,
                    min: grid.min,
                    max: grid.max,
                });
            }
        }
        if let Some(seconds) = self.timing.seconds {
            let fps = self.timing.fps.unwrap_or(caps.fps.default);
            if grid.align_up(seconds.saturating_mul(fps)).is_none() {
                return Err(RequestError::Seconds {
                    model: model(),
                    seconds,
                    fps,
                    min: grid.min,
                    max: grid.max,
                });
            }
        }
        if let Some(fps) = self.timing.fps {
            if !caps.fps.allows(fps) {
                return Err(RequestError::Fps {
                    model: model(),
                    fps,
                    allowed: caps.fps.allowed.clone(),
                });
            }
        }
        self.validate_refs(caps)?;
        let knobs = [
            ("seed", self.seed.is_some(), caps.knobs.seed),
            ("negative", self.negative.is_some(), caps.knobs.negative),
            ("steps", self.sampler.steps.is_some(), caps.knobs.steps),
            (
                "guidance",
                self.sampler.guidance.is_some(),
                caps.knobs.guidance,
            ),
            (
                "guidance_2",
                self.sampler.guidance_2.is_some(),
                caps.knobs.guidance_2,
            ),
        ];
        for (knob, set, honoured) in knobs {
            if set && !honoured {
                return Err(RequestError::Knob {
                    model: model(),
                    knob,
                });
            }
        }
        Ok(())
    }

    fn validate_canvas(&self, caps: &ModelCaps) -> Result<(), RequestError> {
        let c = &caps.canvas;
        let err = |width: u32, height: u32, why: String| RequestError::Canvas {
            model: caps.id.0.clone(),
            width,
            height,
            why,
        };
        let (w, h) = (self.canvas.width, self.canvas.height);
        if !c.pad_and_crop && c.multiple > 0 {
            for side in [w, h].into_iter().flatten() {
                if !side.is_multiple_of(c.multiple) {
                    return Err(err(
                        w.unwrap_or(0),
                        h.unwrap_or(0),
                        format!("is not a multiple of {}", c.multiple),
                    ));
                }
            }
        }
        let (Some(w), Some(h)) = (w, h) else {
            return Ok(());
        };
        if !c.aspect_ok(w, h) {
            return Err(err(
                w,
                h,
                format!("is outside the aspect range {:?}", c.aspect),
            ));
        }
        let budget = if c.short_edges.is_empty() {
            c.max_area
        } else {
            c.area_at(w.min(h))
        };
        let area = u64::from(w) * u64::from(h);
        if budget > 0 && area > budget {
            return Err(err(w, h, format!("exceeds the pixel budget {budget}")));
        }
        Ok(())
    }

    fn validate_refs(&self, caps: &ModelCaps) -> Result<(), RequestError> {
        let refs = self.references();
        if refs.is_empty() {
            return Ok(());
        }
        let (mut images, mut videos, mut audio) = (0u32, 0u32, 0u32);
        for r in refs {
            match infer_reference_kind(r) {
                ReferenceKind::Image => images += 1,
                ReferenceKind::Video => videos += 1,
                ReferenceKind::Audio => audio += 1,
            }
        }
        let l = &caps.refs;
        for (kind, count, max) in [
            ("image", images, l.images),
            ("video", videos, l.videos),
            ("audio", audio, l.audio),
            ("total", images + videos + audio, l.total),
        ] {
            if count > max {
                return Err(RequestError::Refs {
                    model: caps.id.0.clone(),
                    kind,
                    count,
                    max,
                });
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fastvideo_protocol::caps::{FpsCaps, FrameGrid, KnobCaps, RefLimits};

    fn h3() -> ModelCaps {
        ModelCaps::h3("h3-test", true)
    }

    /// A Wan-like model: t2v only, the `4k+1` grid, 32-pixel sides, 16 fps,
    /// every knob.
    fn wan() -> ModelCaps {
        let mut caps = ModelCaps::h3("wan-test", false);
        caps.tasks = [Task::T2V].into_iter().collect();
        caps.frames = FrameGrid::new(4, 1, 9, 161, 81);
        caps.fps = FpsCaps::fixed(16);
        caps.refs = RefLimits::none();
        caps.knobs = KnobCaps::all();
        caps.canvas.multiple = 32;
        caps.canvas.aspect = (0.25, 4.0);
        caps.canvas.short_edges = vec![480];
        caps.canvas.max_area = 832 * 480;
        caps.canvas.hd = None;
        caps
    }

    #[test]
    fn text_request_at_defaults_fits() {
        assert_eq!(Request::text("a cat").validate(&h3()), Ok(()));
        assert_eq!(Request::text("a cat").validate(&wan()), Ok(()));
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
    fn tasks_must_be_supported() {
        let mut r = Request::text("x");
        r.conditioning = vec![Conditioning::Image("a.png".into())];
        assert_eq!(r.validate(&h3()), Ok(()));
        assert!(matches!(
            r.validate(&wan()),
            Err(RequestError::Task {
                task: Task::I2V,
                ..
            })
        ));
        r.conditioning = vec![Conditioning::Refs(vec!["a.png".into()])];
        assert_eq!(r.validate(&h3()), Ok(()));
        assert!(matches!(
            r.validate(&ModelCaps::h3("no-ref", false)),
            Err(RequestError::Task {
                task: Task::Ref2V,
                ..
            })
        ));
    }

    #[test]
    fn image_and_references_conflict() {
        let mut r = Request::text("x");
        r.conditioning = Request::conditioning_from_inputs(
            Some("a.png".into()),
            None,
            vec!["r.png".into()],
            None,
        );
        assert_eq!(r.validate(&h3()), Err(RequestError::ImageAndRefs));
        r.conditioning = vec![
            Conditioning::Image("a.png".into()),
            Conditioning::Image("b.png".into()),
        ];
        assert_eq!(r.validate(&h3()), Err(RequestError::Duplicate("image")));
    }

    #[test]
    fn canvas_checks_multiple_aspect_and_budget() {
        let mut r = Request::text("x");
        r.canvas = Canvas {
            width: Some(832),
            height: Some(480),
        };
        assert_eq!(r.validate(&wan()), Ok(()));
        r.canvas.width = Some(830);
        assert!(matches!(
            r.validate(&wan()),
            Err(RequestError::Canvas { .. })
        ));
        // Aspect fine (2.13), area 491520 over the 832x480 budget.
        r.canvas = Canvas {
            width: Some(1024),
            height: Some(480),
        };
        assert!(matches!(
            r.validate(&wan()),
            Err(RequestError::Canvas { .. })
        ));
        r.canvas = Canvas {
            width: Some(2560),
            height: Some(480),
        };
        assert!(matches!(
            r.validate(&wan()),
            Err(RequestError::Canvas { .. })
        ));
    }

    #[test]
    fn frames_seconds_and_fps_follow_the_caps() {
        let mut r = Request::text("x");
        r.timing.frames = Some(81);
        assert_eq!(r.validate(&wan()), Ok(()));
        r.timing.frames = Some(80);
        assert!(matches!(
            r.validate(&wan()),
            Err(RequestError::Frames { frames: 80, .. })
        ));
        r.timing.frames = Some(5);
        assert!(matches!(
            r.validate(&wan()),
            Err(RequestError::Frames { .. })
        ));
        let mut r = Request::text("x");
        r.timing.seconds = Some(5);
        assert_eq!(r.validate(&h3()), Ok(()));
        r.timing.seconds = Some(60);
        assert!(matches!(
            r.validate(&h3()),
            Err(RequestError::Seconds { .. })
        ));
        let mut r = Request::text("x");
        r.timing.fps = Some(16);
        assert_eq!(r.validate(&wan()), Ok(()));
        r.timing.fps = Some(30);
        assert!(matches!(
            r.validate(&wan()),
            Err(RequestError::Fps { fps: 30, .. })
        ));
    }

    #[test]
    fn reference_counts_follow_the_limits() {
        let mut r = Request::text("x");
        r.conditioning = vec![Conditioning::Refs(
            (0..4).map(|i| PathBuf::from(format!("v{i}.mp4"))).collect(),
        )];
        assert!(matches!(
            r.validate(&h3()),
            Err(RequestError::Refs {
                kind: "video",
                count: 4,
                max: 3,
                ..
            })
        ));
        r.conditioning = vec![Conditioning::Refs(vec![
            "a.png".into(),
            "b.mp4".into(),
            "c.wav".into(),
        ])];
        assert_eq!(r.validate(&h3()), Ok(()));
    }

    #[test]
    fn knobs_the_model_ignores_are_refused() {
        let mut r = Request::text("x");
        r.seed = Some(7);
        assert_eq!(r.validate(&h3()), Ok(()));
        r.sampler.steps = Some(4);
        assert!(matches!(
            r.validate(&h3()),
            Err(RequestError::Knob { knob: "steps", .. })
        ));
        assert_eq!(r.validate(&wan()), Ok(()));
    }
}
