//! `negotiate()`: request + caps + staged inputs -> `ResolvedJob` (design §3.2).
//!
//! Rules, applied in this order (the first failure wins):
//!
//! 1. The model resolves through aliases ([`resolve_model`], before
//!    `negotiate`, which receives the chosen caps).
//! 2. `task ∈ caps.tasks`, and the inputs fit the task (I2V: one first frame;
//!    Keyframes: last, or first+last; Ref2V: references only; T2V: none).
//!    Provider-file refs are refused here too.
//! 3. The canvas resolves (H3: `fastvideo_models::h3::config::resolve_canvas_size`
//!    and `check_canvas`; others: its short-edge generalization).
//! 4. The frame count resolves on the grid.
//! 5. The fps is allowed.
//! 6. Reference limits hold.
//! 7. Knobs: any knob the caps do not honour -> `InvalidRequest` naming the
//!    field ("refuse, do not drop").
//! 8. Audio: `Sidecar` requires `via_sidecar`; input audio needs a model path.
//!
//! Engine gaps (design §4.7) answer `Unsupported(GapId)` at the rule that
//! detects them. [`precheck`] runs every rule that does not need staged media,
//! so handlers can refuse before downloading anything.

use std::path::PathBuf;

use fastvideo_models::h3::config as h3;
use serde::{Deserialize, Serialize};

use crate::caps::{CanvasCaps, ModelCaps, Tier};
use crate::error::{ApiError, GapId};
use crate::request::{
    Anchor, AudioOut, AudioRole, CanvasSpec, Family, GenerationRequest, Length, MediaKind,
    MediaRef, ModelId, Ratio, SamplingOverrides, Snap, Task,
};

/// Media staged by serve-kit ingestion, in request order.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct StagedInputs {
    pub keyframes: Vec<(Anchor, StagedMedia)>,
    pub references: Vec<(MediaKind, StagedMedia)>,
    pub audio_in: Option<StagedMedia>,
}

/// One staged input file.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StagedMedia {
    pub path: PathBuf,
    pub mime: String,
    pub bytes: u64,
    pub probe: MediaProbe,
}

/// What probing found (image decode / ffprobe). Missing facts are `None`.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct MediaProbe {
    pub width: Option<u32>,
    pub height: Option<u32>,
    pub duration_s: Option<f64>,
    pub fps: Option<f64>,
    pub audio_rate: Option<u32>,
}

impl MediaProbe {
    /// `(width, height)` when both are known and non-zero.
    pub fn dims(&self) -> Option<(u32, u32)> {
        match (self.width, self.height) {
            (Some(w), Some(h)) if w > 0 && h > 0 => Some((w, h)),
            _ => None,
        }
    }
}

/// What the output's audio will be.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "plan", rename_all = "snake_case")]
pub enum AudioPlan {
    /// The model's own audio at its native format.
    Native { rate: u32, channels: u8 },
    /// The model makes audio but the output drops it (`AudioOut::Silent`).
    Drop,
    /// MMAudio V2A sidecar.
    Sidecar,
    /// Video-only output.
    None,
}

impl AudioPlan {
    /// Whether the output file carries an audio stream.
    pub fn has_audio(&self) -> bool {
        matches!(self, AudioPlan::Native { .. } | AudioPlan::Sidecar)
    }
}

/// Post-processing applied by `fastvideo-media::mp4::finalize`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct PostProcess {
    /// Crop to `(width, height)` (pad-and-crop canvases).
    pub crop: Option<(u32, u32)>,
    /// `-an`.
    pub drop_audio: bool,
}

/// Everything the engine needs; no protocol detail left.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ResolvedJob {
    pub model: ModelId,
    pub task: Task,
    pub prompt: String,
    /// Empty = the model's default negative prompt.
    pub negative_prompt: String,
    pub seed: u64,
    /// Generation canvas (before any `post.crop`).
    pub width: u32,
    pub height: u32,
    pub num_frames: u32,
    pub fps: u32,
    pub keyframes: Vec<(Anchor, PathBuf)>,
    /// Order preserved exactly as sent.
    pub references: Vec<(MediaKind, PathBuf)>,
    pub audio_in: Option<(AudioRole, PathBuf)>,
    pub audio: AudioPlan,
    pub post: PostProcess,
    /// Only knobs the caps honour; others were already rejected.
    pub sampling: SamplingOverrides,
    /// The model's tier, from [`ModelCaps::tier`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tier: Option<Tier>,
    /// The resolved internal recipe, from [`ModelCaps::recipe`]; adapters echo
    /// it in response metadata where the wire format allows (design §0.3).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub recipe: Option<String>,
}

impl ResolvedJob {
    /// The delivered canvas: the crop when there is one, else the generation canvas.
    pub fn output_size(&self) -> (u32, u32) {
        self.post.crop.unwrap_or((self.width, self.height))
    }
    /// `num_frames / fps`.
    pub fn duration_s(&self) -> f64 {
        if self.fps == 0 {
            0.0
        } else {
            self.num_frames as f64 / self.fps as f64
        }
    }
}

/// A fresh seed in `0..2^32` (JSON-safe for every client), from OS randomness.
pub fn draw_seed() -> u64 {
    let b = uuid::Uuid::new_v4().into_bytes();
    u64::from(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
}

/// Rule 1: resolve the name a client sent to one model's caps. `alias` maps a
/// name to an engine model id (`ServeConfig.aliases`); the result is the model
/// whose id or served name matches the alias target, or the name itself.
/// Unknown -> `InvalidRequest` naming `model`.
pub fn resolve_model<'a>(
    requested: &str,
    alias: impl Fn(&str) -> Option<String>,
    models: impl IntoIterator<Item = &'a ModelCaps>,
) -> Result<&'a ModelCaps, ApiError> {
    let target = alias(requested).unwrap_or_else(|| requested.to_owned());
    let models: Vec<&ModelCaps> = models.into_iter().collect();
    models
        .iter()
        .find(|m| m.id.0 == target)
        .or_else(|| models.iter().find(|m| m.answers_to(&target)))
        .copied()
        .ok_or_else(|| {
            ApiError::invalid_param("model", format!("model `{requested}` is not served here"))
        })
}

/// The served model of `family` at `tier` (design §0.3), for adapters that
/// address models by tier rather than by a configured alias. When several
/// match (e.g. LTX-2.3 and LTX-2.5 both `Max`), the first wins, so aliases
/// should be used to pick a specific version. None -> `InvalidRequest` on
/// `model` naming the tier.
pub fn resolve_tier<'a>(
    family: Family,
    tier: Tier,
    models: impl IntoIterator<Item = &'a ModelCaps>,
) -> Result<&'a ModelCaps, ApiError> {
    models
        .into_iter()
        .find(|m| m.family == family && m.tier == Some(tier))
        .ok_or_else(|| {
            ApiError::invalid_param(
                "model",
                format!("no {tier} tier model of family `{family:?}` is served here"),
            )
        })
}

/// Pure and deterministic (except that a missing seed is drawn with
/// [`draw_seed`]). Called after media ingestion has staged inputs.
pub fn negotiate(
    req: &GenerationRequest,
    caps: &ModelCaps,
    staged: &StagedInputs,
) -> Result<ResolvedJob, ApiError> {
    check_task(req, caps)?;
    check_staged(req, staged)?;
    let fps = requested_fps(req, caps);
    let (width, height, crop) = resolve_canvas(&req.canvas, caps, follow_dims(req, staged)?)?;
    let num_frames = resolve_frames(&req.timing.length, fps, caps)?;
    check_fps(fps, caps)?;
    check_h3_geometry(caps, width, height, num_frames)?;
    check_refs(req, caps)?;
    check_knobs(req, caps)?;
    let audio = plan_audio(req, caps)?;

    let keyframes = staged
        .keyframes
        .iter()
        .map(|(a, m)| (*a, m.path.clone()))
        .collect();
    let references = staged
        .references
        .iter()
        .map(|(k, m)| (*k, m.path.clone()))
        .collect();
    let audio_in = match (&req.audio_in, &staged.audio_in) {
        (Some(a), Some(m)) => Some((a.role, m.path.clone())),
        _ => None,
    };
    Ok(ResolvedJob {
        model: caps.id.clone(),
        task: req.task,
        prompt: req.prompt.clone(),
        negative_prompt: req.negative_prompt.clone().unwrap_or_default(),
        seed: req.seed.unwrap_or_else(draw_seed),
        width,
        height,
        num_frames,
        fps,
        keyframes,
        references,
        audio_in,
        audio,
        post: PostProcess {
            crop,
            drop_audio: audio == AudioPlan::Drop,
        },
        sampling: req.sampling.clone(),
        tier: caps.tier,
        recipe: caps.recipe.clone(),
    })
}

/// Every rule that does not need staged media (for `FollowImage` only the
/// short-edge tier is checked). Call before ingestion to fail fast.
pub fn precheck(req: &GenerationRequest, caps: &ModelCaps) -> Result<(), ApiError> {
    check_task(req, caps)?;
    let fps = requested_fps(req, caps);
    let canvas = match &req.canvas {
        CanvasSpec::FollowImage { short_edge } => {
            check_tier(*short_edge, caps)?;
            None
        }
        other => Some(resolve_canvas(other, caps, None)?),
    };
    let num_frames = resolve_frames(&req.timing.length, fps, caps)?;
    check_fps(fps, caps)?;
    if let Some((w, h, _)) = canvas {
        check_h3_geometry(caps, w, h, num_frames)?;
    }
    check_refs(req, caps)?;
    check_knobs(req, caps)?;
    plan_audio(req, caps).map(|_| ())
}

/// H3 only: the final shape check through `fastvideo_models::h3::config::H3Geometry`.
fn check_h3_geometry(
    caps: &ModelCaps,
    width: u32,
    height: u32,
    num_frames: u32,
) -> Result<(), ApiError> {
    if caps.family == Family::H3 {
        h3::H3Geometry::new(height as usize, width as usize, num_frames as usize)
            .map_err(|e| ApiError::invalid(format!("H3 geometry: {e}")))?;
    }
    Ok(())
}

// ---- rule 2: task ------------------------------------------------------------

fn check_task(req: &GenerationRequest, caps: &ModelCaps) -> Result<(), ApiError> {
    let task = req.task;
    if !caps.supports(task) {
        let gap = match (task, caps.family) {
            (t, _) if t.is_edit_endpoint() => Some(GapId::LtxEndpoint),
            (Task::Ref2V, Family::H3) => Some(GapId::H3Ref2vaNotLoaded),
            (Task::Keyframes, Family::Ltx2) => Some(GapId::LtxKeyframes),
            (Task::I2V, Family::Ltx2) => Some(GapId::Ltx25I2V),
            _ => None,
        };
        return Err(match gap {
            Some(g) => ApiError::unsupported(g),
            None => ApiError::invalid_param(
                "task",
                format!(
                    "model `{}` does not support task {}",
                    caps.id,
                    task_name(task)
                ),
            ),
        });
    }

    let first = req
        .keyframes
        .iter()
        .filter(|k| k.at == Anchor::First)
        .count();
    let last = req
        .keyframes
        .iter()
        .filter(|k| k.at == Anchor::Last)
        .count();
    let refs = req.references.len();
    let shape_ok = match task {
        Task::T2V => req.keyframes.is_empty() && refs == 0,
        Task::I2V => first == 1 && last == 0 && refs == 0,
        Task::Keyframes => last == 1 && first <= 1 && refs == 0,
        Task::Ref2V => refs > 0 && req.keyframes.is_empty(),
        Task::A2V | Task::Extend | Task::Retake | Task::V2V => true,
    };
    if !shape_ok {
        let msg = match task {
            Task::T2V => "text-to-video takes no images or references",
            Task::I2V => "image-to-video takes exactly one first-frame image and no references",
            Task::Keyframes => {
                "keyframes take a last frame and at most one first frame, and no references"
            }
            Task::Ref2V => {
                "reference-to-video takes one or more references and no first/last frames"
            }
            _ => "inputs do not match the task",
        };
        return Err(ApiError::invalid_param("task", msg));
    }
    if req
        .media_refs()
        .any(|m| matches!(m, MediaRef::ProviderFile(_)))
    {
        return Err(ApiError::unsupported(GapId::ProviderFiles));
    }
    if req.prompt.trim().is_empty() && task == Task::T2V {
        return Err(ApiError::invalid_param(
            "prompt",
            "prompt must not be empty",
        ));
    }
    Ok(())
}

fn task_name(t: Task) -> &'static str {
    match t {
        Task::T2V => "t2v",
        Task::I2V => "i2v",
        Task::Keyframes => "keyframes",
        Task::Ref2V => "ref2v",
        Task::A2V => "a2v",
        Task::Extend => "extend",
        Task::Retake => "retake",
        Task::V2V => "v2v",
    }
}

fn check_staged(req: &GenerationRequest, staged: &StagedInputs) -> Result<(), ApiError> {
    let mismatch = || ApiError::internal("staged inputs do not match the request");
    if staged.keyframes.len() != req.keyframes.len()
        || staged.references.len() != req.references.len()
        || staged.audio_in.is_some() != req.audio_in.is_some()
    {
        return Err(mismatch());
    }
    if req
        .keyframes
        .iter()
        .zip(&staged.keyframes)
        .any(|(k, (a, _))| k.at != *a)
        || req
            .references
            .iter()
            .zip(&staged.references)
            .any(|(r, (k, _))| r.kind != *k)
    {
        return Err(mismatch());
    }
    let kinds = staged
        .keyframes
        .iter()
        .map(|(_, m)| (MediaKind::Image, m, "image_url"))
        .chain(staged.references.iter().map(|(k, m)| (*k, m, "references")))
        .chain(
            staged
                .audio_in
                .iter()
                .map(|m| (MediaKind::Audio, m, "audio_url")),
        );
    for (kind, m, param) in kinds {
        if !mime_fits(kind, &m.mime) {
            return Err(ApiError::unsupported_media(format!(
                "expected {} input, got `{}`",
                kind_name(kind),
                m.mime
            ))
            .with_param(param));
        }
    }
    Ok(())
}

fn kind_name(k: MediaKind) -> &'static str {
    match k {
        MediaKind::Image => "image",
        MediaKind::Video => "video",
        MediaKind::Audio => "audio",
    }
}

/// A mime conflicts only when its top-level type is another media kind.
fn mime_fits(kind: MediaKind, mime: &str) -> bool {
    let top = mime
        .split('/')
        .next()
        .unwrap_or("")
        .trim()
        .to_ascii_lowercase();
    match top.as_str() {
        "image" => kind == MediaKind::Image,
        "video" => kind == MediaKind::Video,
        "audio" => kind == MediaKind::Audio,
        _ => true,
    }
}

// ---- rule 3: canvas ------------------------------------------------------------

/// The image `FollowImage` follows: the first keyframe (first-frame anchor
/// preferred), else the first image reference.
fn follow_dims(
    req: &GenerationRequest,
    staged: &StagedInputs,
) -> Result<Option<(u32, u32)>, ApiError> {
    if !matches!(req.canvas, CanvasSpec::FollowImage { .. }) {
        return Ok(None);
    }
    let img = staged
        .keyframes
        .iter()
        .find(|(a, _)| *a == Anchor::First)
        .or_else(|| staged.keyframes.first())
        .map(|(_, m)| m)
        .or_else(|| {
            staged
                .references
                .iter()
                .find(|(k, _)| *k == MediaKind::Image)
                .map(|(_, m)| m)
        });
    let Some(img) = img else {
        return Err(ApiError::invalid_param(
            "aspect_ratio",
            "an adaptive aspect ratio needs an input image",
        ));
    };
    img.probe.dims().map(Some).ok_or_else(|| {
        ApiError::unsupported_media("could not read the input image size").with_param("image_url")
    })
}

fn check_tier(short_edge: u32, caps: &ModelCaps) -> Result<(), ApiError> {
    if caps.canvas.short_edges.contains(&short_edge) {
        return Ok(());
    }
    if caps.family == Family::H3 {
        if short_edge == 1080 {
            return Err(ApiError::unsupported(GapId::H3Refine1080P));
        }
        if short_edge > 1080 {
            return Err(ApiError::unsupported(GapId::H3Resolution2K));
        }
    }
    Err(ApiError::invalid_param(
        "resolution",
        format!(
            "short edge {short_edge} is not supported by model `{}`; supported: {:?}",
            caps.id, caps.canvas.short_edges
        ),
    ))
}

/// A resolved canvas: `(width, height, crop)` where `crop` is the delivered
/// size when the generation canvas is padded (`CanvasCaps::pad_and_crop`).
pub type ResolvedCanvas = (u32, u32, Option<(u32, u32)>);

/// Rule 3. `follow` is the input image size for `FollowImage`.
pub fn resolve_canvas(
    spec: &CanvasSpec,
    caps: &ModelCaps,
    follow: Option<(u32, u32)>,
) -> Result<ResolvedCanvas, ApiError> {
    let c = &caps.canvas;
    match spec {
        CanvasSpec::Exact { width, height } => exact_canvas(*width, *height, caps),
        CanvasSpec::Aspect { ratio, short_edge } => {
            check_tier(*short_edge, caps)?;
            if !c.aspect_ok(ratio.w, ratio.h) {
                return Err(ApiError::invalid_param(
                    "aspect_ratio",
                    format!(
                        "aspect ratio {ratio} is outside {}..{}",
                        c.aspect.0, c.aspect.1
                    ),
                ));
            }
            aspect_canvas(ratio.w as f64, ratio.h as f64, *short_edge, caps)
        }
        CanvasSpec::FollowImage { short_edge } => {
            check_tier(*short_edge, caps)?;
            let (w, h) = follow.ok_or_else(|| {
                ApiError::invalid_param(
                    "aspect_ratio",
                    "an adaptive aspect ratio needs an input image",
                )
            })?;
            if !c.aspect_ok(w, h) {
                return Err(ApiError::invalid_param(
                    "image_url",
                    format!(
                        "input image aspect {w}x{h} is outside {}..{}",
                        c.aspect.0, c.aspect.1
                    ),
                ));
            }
            aspect_canvas(w as f64, h as f64, *short_edge, caps)
        }
        CanvasSpec::ModelDefault => {
            let short = *c.short_edges.first().ok_or_else(|| {
                ApiError::internal(format!("model `{}` declares no canvas tier", caps.id))
            })?;
            aspect_canvas(Ratio::R16_9.w as f64, Ratio::R16_9.h as f64, short, caps)
        }
    }
}

fn exact_canvas(w: u32, h: u32, caps: &ModelCaps) -> Result<ResolvedCanvas, ApiError> {
    let c = &caps.canvas;
    if w == 0 || h == 0 {
        return Err(ApiError::invalid_param(
            "size",
            "width and height must be positive",
        ));
    }
    if caps.family == Family::H3 {
        h3::check_canvas(h as usize, w as usize).map_err(|e| ApiError::invalid_param("size", e))?;
        return Ok((w, h, None));
    }
    if !c.aspect_ok(w, h) {
        return Err(ApiError::invalid_param(
            "size",
            format!(
                "{w}x{h} is outside the aspect range {}..{}",
                c.aspect.0, c.aspect.1
            ),
        ));
    }
    let m = c.multiple.max(1);
    let (gw, gh) = if c.pad_and_crop {
        (w.div_ceil(m) * m, h.div_ceil(m) * m)
    } else {
        (w, h)
    };
    if gw % m != 0 || gh % m != 0 {
        return Err(ApiError::invalid_param(
            "size",
            format!("width and height must be multiples of {m}, got {w}x{h}"),
        ));
    }
    if gw as u64 * gh as u64 > c.max_area {
        return Err(ApiError::invalid_param(
            "size",
            format!("{w}x{h} exceeds the model's {} pixel budget", c.max_area),
        ));
    }
    let crop = ((gw, gh) != (w, h)).then_some((w, h));
    Ok((gw, gh, crop))
}

fn aspect_canvas(
    aw: f64,
    ah: f64,
    short_edge: u32,
    caps: &ModelCaps,
) -> Result<ResolvedCanvas, ApiError> {
    if caps.family == Family::H3 && short_edge as usize == h3::H3_SHORT_EDGE {
        let (h, w) = h3::resolve_canvas_size(aw, ah)
            .map_err(|e| ApiError::invalid_param("aspect_ratio", e))?;
        return Ok((w as u32, h as u32, None));
    }
    let c = &caps.canvas;
    if c.pad_and_crop {
        // Exact target (even sides), generated padded up to the multiple.
        let r = aw / ah;
        let s = short_edge as f64;
        let even = |v: f64| ((v / 2.0).round() as u32).max(1) * 2;
        let (w, h) = if r >= 1.0 {
            (even(s * r), short_edge)
        } else {
            (short_edge, even(s / r))
        };
        return exact_canvas(w, h, caps);
    }
    let (w, h) = canvas_for_aspect(c, aw / ah, short_edge);
    Ok((w, h, None))
}

/// The short-edge generalization of `resolve_canvas_size` (`packing.py`):
/// `short_edge` on the short side, capped at `caps.area_at(short_edge)`
/// pixels, each side snapped to the nearest multiple (ties to even).
/// Returns `(width, height)`. Equals `resolve_canvas_size` for the H3 768 tier.
pub fn canvas_for_aspect(c: &CanvasCaps, ratio: f64, short_edge: u32) -> (u32, u32) {
    let s = short_edge as f64;
    let (mut w, mut h) = if ratio >= 1.0 {
        (s * ratio, s)
    } else {
        (s, s / ratio)
    };
    let cap = c.area_at(short_edge) as f64;
    if w * h > cap {
        let k = (cap / (w * h)).sqrt();
        w *= k;
        h *= k;
    }
    let m = c.multiple.max(1) as f64;
    let snap = |v: f64| (((v / m).round_ties_even() * m) as u32).max(m as u32);
    (snap(w), snap(h))
}

// ---- rules 4-5: frames and fps ------------------------------------------------

fn requested_fps(req: &GenerationRequest, caps: &ModelCaps) -> u32 {
    req.timing.fps.unwrap_or(caps.fps.default)
}

fn check_fps(fps: u32, caps: &ModelCaps) -> Result<(), ApiError> {
    if caps.fps.allows(fps) {
        return Ok(());
    }
    if caps.family == Family::Ltx2 {
        return Err(ApiError::unsupported(GapId::LtxFps).with_param("fps"));
    }
    Err(ApiError::invalid_param(
        "fps",
        format!(
            "fps {fps} is not supported by model `{}`; allowed: {:?}",
            caps.id, caps.fps.allowed
        ),
    ))
}

/// Resolves a length on `caps.frames` at `fps`.
pub fn resolve_frames(length: &Length, fps: u32, caps: &ModelCaps) -> Result<u32, ApiError> {
    let g = &caps.frames;
    let (raw, snap, param) = match *length {
        Length::ModelDefault => return Ok(g.default),
        Length::Auto => {
            return Err(if caps.family == Family::Ltx2 {
                ApiError::unsupported(GapId::LtxAutoDuration)
            } else {
                ApiError::invalid_param("duration", "an explicit duration is required")
            })
        }
        Length::Seconds { value, snap } => {
            if !value.is_finite() || value <= 0.0 || fps == 0 {
                return Err(ApiError::invalid_param(
                    "duration",
                    "duration must be a positive number of seconds",
                ));
            }
            let raw = (value * fps as f64 - 1e-6).ceil();
            if raw > u32::MAX as f64 {
                return Err(ApiError::invalid_param("duration", "duration is too long"));
            }
            (raw as u32, snap, "duration")
        }
        Length::Frames { value, snap } => (value, snap, "num_frames"),
    };
    let next = g.next_on_grid(raw);
    // H3 4 s (107 frames) is on the grid since E3; APIs that keep
    // FastVideo's 5 s floor (openai-videos) enforce it themselves.
    let range = || {
        let f = fps.max(1) as f64;
        format!(
            "{}..={} frames ({:.3}..={:.3} s at {fps} fps) on the {}k+{} grid",
            g.min,
            g.max,
            g.min as f64 / f,
            g.max as f64 / f,
            g.step,
            g.offset
        )
    };
    match snap {
        Snap::AlignUp => g.align_up(raw).ok_or_else(|| {
            ApiError::invalid_param(param, format!("length must be within {}", range()))
        }),
        Snap::Exact if g.contains(raw) => Ok(raw),
        Snap::Exact => {
            let hint = g
                .align_up(raw)
                .map(|n| format!("; nearest valid: {n}"))
                .unwrap_or_default();
            Err(ApiError::invalid_param(
                param,
                format!("{param} {raw} must be one of {}{hint}", range()),
            ))
        }
    }
}

// ---- rules 6-8 -------------------------------------------------------------------

fn check_refs(req: &GenerationRequest, caps: &ModelCaps) -> Result<(), ApiError> {
    let l = &caps.refs;
    let count = |k: MediaKind| req.references.iter().filter(|r| r.kind == k).count() as u32;
    let checks = [
        (count(MediaKind::Image), l.images, "reference images"),
        (count(MediaKind::Video), l.videos, "reference videos"),
        (count(MediaKind::Audio), l.audio, "reference audio clips"),
        (req.references.len() as u32, l.total, "references in total"),
    ];
    for (n, max, what) in checks {
        if n > max {
            return Err(ApiError::invalid_param(
                "references",
                format!("at most {max} {what} are allowed, got {n}"),
            ));
        }
    }
    Ok(())
}

fn check_knobs(req: &GenerationRequest, caps: &ModelCaps) -> Result<(), ApiError> {
    let k = &caps.knobs;
    let s = &req.sampling;
    let refuse = |param: &str| {
        ApiError::invalid_param(
            param,
            format!("`{param}` is not supported by model `{}`", caps.id),
        )
    };
    if req.seed.is_some() && !k.seed {
        return Err(refuse("seed"));
    }
    if req
        .negative_prompt
        .as_deref()
        .is_some_and(|n| !n.trim().is_empty())
        && !k.negative
    {
        return Err(refuse("negative_prompt"));
    }
    if let Some(steps) = s.steps {
        if !k.steps {
            return Err(if caps.family == Family::H3 {
                ApiError::unsupported(GapId::PerRequestSteps).with_param("num_inference_steps")
            } else {
                refuse("num_inference_steps")
            });
        }
        if steps == 0 {
            return Err(ApiError::invalid_param(
                "num_inference_steps",
                "num_inference_steps must be positive",
            ));
        }
    }
    let finite_nonneg = |v: f32| v.is_finite() && v >= 0.0;
    if let Some(g) = s.guidance {
        if !k.guidance {
            return Err(refuse("guidance_scale"));
        }
        if !finite_nonneg(g) {
            return Err(ApiError::invalid_param(
                "guidance_scale",
                "guidance_scale must be a non-negative number",
            ));
        }
    }
    if let Some(g) = s.guidance_2 {
        if !k.guidance_2 {
            return Err(refuse("guidance_scale_2"));
        }
        if !finite_nonneg(g) {
            return Err(ApiError::invalid_param(
                "guidance_scale_2",
                "guidance_scale_2 must be a non-negative number",
            ));
        }
    }
    if let Some(f) = s.flow_shift {
        if !k.flow_shift {
            return Err(refuse("flow_shift"));
        }
        if !(f.is_finite() && f > 0.0) {
            return Err(ApiError::invalid_param(
                "flow_shift",
                "flow_shift must be positive",
            ));
        }
    }
    if let Some(b) = s.boundary_ratio {
        if !k.guidance_2 {
            return Err(refuse("boundary_ratio"));
        }
        if !(0.0..=1.0).contains(&b) {
            return Err(ApiError::invalid_param(
                "boundary_ratio",
                "boundary_ratio must be within 0..=1",
            ));
        }
    }
    Ok(())
}

fn plan_audio(req: &GenerationRequest, caps: &ModelCaps) -> Result<AudioPlan, ApiError> {
    if let Some(a) = &req.audio_in {
        match a.role {
            AudioRole::TargetSoundtrack => {
                return Err(if caps.family == Family::H3 {
                    ApiError::unsupported(GapId::H3TargetAudio)
                } else {
                    ApiError::invalid_param(
                        "audio_url",
                        format!("model `{}` takes no input audio", caps.id),
                    )
                });
            }
            AudioRole::Drive if req.task != Task::A2V => {
                return Err(ApiError::invalid_param(
                    "audio_url",
                    "driving audio is only valid for audio-to-video",
                ));
            }
            AudioRole::Drive => {}
        }
    }
    Ok(match (req.audio_out, &caps.audio) {
        (AudioOut::ModelDefault, Some(a)) if !a.via_sidecar => AudioPlan::Native {
            rate: a.native_rate,
            channels: a.channels,
        },
        (AudioOut::ModelDefault, _) => AudioPlan::None,
        (AudioOut::Silent, Some(a)) if !a.via_sidecar => AudioPlan::Drop,
        (AudioOut::Silent, _) => AudioPlan::None,
        (AudioOut::Sidecar, Some(a)) if a.via_sidecar => AudioPlan::Sidecar,
        (AudioOut::Sidecar, _) => {
            return Err(ApiError::invalid_param(
                "generate_audio",
                format!("model `{}` has no audio sidecar", caps.id),
            ))
        }
    })
}
