//! `ModelCaps`, `FrameGrid`, `CanvasCaps`, `KnobCaps`, `StreamCaps` (design §3.2).
//!
//! What one resident model can do. The engine service builds these (fake or
//! from loaded configs); `negotiate()` checks requests against them; the
//! native `/fv/v1/capabilities` route serializes them.

use std::collections::BTreeSet;

use fastvideo_models::h3::config as h3;
use serde::{Deserialize, Serialize};

use crate::request::{Family, ModelId, Task};

/// Capabilities of one servable model.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ModelCaps {
    pub id: ModelId,
    pub family: Family,
    /// e.g. `["fasth3"]`: FastVideo `/v1/models` ids.
    pub served_names: Vec<String>,
    pub tasks: BTreeSet<Task>,
    /// `None` => video-only.
    pub audio: Option<AudioCaps>,
    pub fps: FpsCaps,
    pub frames: FrameGrid,
    pub canvas: CanvasCaps,
    /// H3: 9 img / 3 vid / 3 aud / 12 total.
    pub refs: RefLimits,
    pub stream: Option<StreamCaps>,
    /// Which `SamplingOverrides` are honoured per request.
    pub knobs: KnobCaps,
    pub resident: bool,
    /// Serving tier (design §0.3): `Max` = the family's highest-quality
    /// configuration, `Turbo` = its fastest configuration that passes the
    /// quality gate. `None` = an untiered model (addressed only by id/alias).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tier: Option<Tier>,
    /// The resolved internal recipe, e.g. `"fasth3-4step-vsa"`. Copied onto
    /// [`crate::ResolvedJob::recipe`] so responses can echo it in metadata.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub recipe: Option<String>,
}

impl ModelCaps {
    /// Tag this model with a tier and the internal recipe it runs.
    pub fn with_tier(mut self, tier: Tier, recipe: impl Into<String>) -> Self {
        self.tier = Some(tier);
        self.recipe = Some(recipe.into());
        self
    }
    /// Whether `name` addresses this model (its id or a served name).
    pub fn answers_to(&self, name: &str) -> bool {
        self.id.0 == name || self.served_names.iter().any(|n| n == name)
    }
    /// Whether the model produces audio natively (not via a sidecar).
    pub fn has_native_audio(&self) -> bool {
        matches!(
            self.audio,
            Some(AudioCaps {
                via_sidecar: false,
                ..
            })
        )
    }
    pub fn supports(&self, task: Task) -> bool {
        self.tasks.contains(&task)
    }

    /// Reference caps for a resident FastH3-style model: t2v, i2v, keyframes
    /// (fl2va) (plus `Ref2V` when `ref2va` is resident), stereo 32 kHz audio,
    /// 24 fps, the `17n+5` grid for 5..15 s, the 768 short-edge canvas, and
    /// only `seed` honoured. Engine backends may start from this and adjust.
    pub fn h3(id: impl Into<String>, ref2va: bool) -> Self {
        let id = id.into();
        let mut tasks: BTreeSet<Task> = [Task::T2V, Task::I2V, Task::Keyframes]
            .into_iter()
            .collect();
        if ref2va {
            tasks.insert(Task::Ref2V);
        }
        Self {
            served_names: vec![id.clone()],
            id: ModelId(id),
            family: Family::H3,
            tasks,
            audio: Some(AudioCaps {
                native_rate: 32_000,
                channels: h3::H3_AUDIO_CHANNELS as u8,
                via_sidecar: false,
            }),
            fps: FpsCaps::fixed(h3::H3_FPS as u32),
            frames: FrameGrid::h3(),
            canvas: CanvasCaps::h3(),
            refs: RefLimits::h3(),
            stream: Some(StreamCaps::Clip {
                min_s: FrameGrid::h3().min as f32 / h3::H3_FPS as f32,
                max_s: FrameGrid::h3().max as f32 / h3::H3_FPS as f32,
            }),
            knobs: KnobCaps {
                seed: true,
                ..KnobCaps::default()
            },
            resident: true,
            tier: None,
            recipe: None,
        }
    }
}

/// Model tier exposed by the public APIs (design §0.3, §0.6).
///
/// Public names map onto tiers through `ServeConfig.aliases` or
/// [`crate::resolve_tier`]: `h3-max` / `MiniMax-H3-Max` / `ltx-2-5-pro` /
/// `ltx-2-3-pro` -> `Max`; `h3-turbo` / `MiniMax-H3-Turbo` / `ltx-turbo` ->
/// `Turbo`; `h3-draft` / `MiniMax-H3-Draft` / `ltx-draft` -> `Draft`.
///
/// Ordered by quality: `Draft < Turbo < Max` (the derived `Ord` follows the
/// variant order).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Tier {
    /// Faster configurations that do NOT pass the quality gate; previews and
    /// iteration only. Results must be marked draft quality (design §0.6).
    Draft,
    /// Fastest generation that still passes the quality gate.
    Turbo,
    /// Highest quality: full step count, non-lossy attention, full VAE.
    Max,
}

impl Tier {
    pub fn as_str(&self) -> &'static str {
        match self {
            Tier::Draft => "draft",
            Tier::Turbo => "turbo",
            Tier::Max => "max",
        }
    }

    /// Whether results at this tier passed the quality gate (`Draft` did not).
    pub fn passes_quality_gate(&self) -> bool {
        !matches!(self, Tier::Draft)
    }
}

impl std::fmt::Display for Tier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Native audio output.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct AudioCaps {
    /// Hz (H3 32 kHz; LTX from the vocoder config).
    pub native_rate: u32,
    pub channels: u8,
    /// Audio comes from the MMAudio V2A sidecar; the model itself is
    /// video-only. Such a model emits audio only for `AudioOut::Sidecar`.
    pub via_sidecar: bool,
}

/// Frame rates.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct FpsCaps {
    pub allowed: Vec<u32>,
    pub default: u32,
    /// The fps only sets the container rate (Wan): frames are generated the
    /// same regardless.
    pub container_only: bool,
}

impl FpsCaps {
    /// Exactly one allowed rate.
    pub fn fixed(fps: u32) -> Self {
        Self {
            allowed: vec![fps],
            default: fps,
            container_only: false,
        }
    }
    pub fn allows(&self, fps: u32) -> bool {
        self.allowed.contains(&fps)
    }
}

/// Admissible frame counts: `offset + step * k` within `min..=max`.
/// H3 `17n+5`, LTX `8k+1`, Wan `4k+1`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct FrameGrid {
    pub step: u32,
    pub offset: u32,
    pub min: u32,
    pub max: u32,
    /// Frame count for `Length::ModelDefault`. **Addition to design §3.2**
    /// (the caps had no default length; see the WP-01 notes in design §8).
    pub default: u32,
}

impl FrameGrid {
    /// A grid; `min`, `max` and `default` should be on it.
    pub fn new(step: u32, offset: u32, min: u32, max: u32, default: u32) -> Self {
        Self {
            step,
            offset,
            min,
            max,
            default,
        }
    }

    /// H3 `17n+5` over 4..15 s at 24 fps (107..=362), default 5 s (124, the
    /// FastVideo / MiniMax default; 4 s is admitted for MiniMax-H3). Built from
    /// `fastvideo_models::h3::config` (`align_num_frames` and constants).
    pub fn h3() -> Self {
        let min = h3::align_num_frames(h3::H3_MIN_DURATION_S * h3::H3_FPS) as u32;
        let max = h3::align_num_frames(h3::H3_MAX_DURATION_S * h3::H3_FPS) as u32;
        let default =
            h3::align_num_frames(h3::H3_FASTVIDEO_MIN_DURATION_S * h3::H3_FPS) as u32;
        Self {
            step: h3::H3_FRAMES_PER_CHUNK as u32,
            offset: h3::H3_LATENTS_PER_CHUNK as u32,
            min,
            max,
            default,
        }
    }

    /// Whether `n = offset + step * k` for some `k >= 0` (ignoring `min`/`max`).
    pub fn on_grid(&self, n: u32) -> bool {
        if n < self.offset {
            return false;
        }
        if self.step == 0 {
            return n == self.offset;
        }
        (n - self.offset) % self.step == 0
    }

    /// The smallest grid value `>= n` (ignoring `min`/`max`).
    pub fn next_on_grid(&self, n: u32) -> Option<u32> {
        if n <= self.offset {
            return Some(self.offset);
        }
        if self.step == 0 {
            return None;
        }
        let k = (n - self.offset).div_ceil(self.step);
        k.checked_mul(self.step)?.checked_add(self.offset)
    }

    /// The smallest admissible count `>= n`: on the grid and within
    /// `min..=max`. `None` when that value is below `min` (i.e. `n < min`
    /// after snapping) or above `max`.
    pub fn align_up(&self, n: u32) -> Option<u32> {
        let m = self.next_on_grid(n)?;
        (self.min..=self.max).contains(&m).then_some(m)
    }

    /// The largest admissible count `<= n`: on the grid, capped at `max`.
    /// `None` when that is below `min`.
    pub fn align_down(&self, n: u32) -> Option<u32> {
        let n = n.min(self.max);
        if n < self.offset {
            return None;
        }
        let m = match (n - self.offset).checked_div(self.step) {
            Some(k) => self.offset + k * self.step,
            None => self.offset,
        };
        (m >= self.min).then_some(m)
    }

    /// On the grid and within `min..=max`.
    pub fn contains(&self, n: u32) -> bool {
        self.on_grid(n) && (self.min..=self.max).contains(&n)
    }
}

/// Output canvas constraints.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CanvasCaps {
    /// Side multiple: 32 (LTX two-stage: 64).
    pub multiple: u32,
    /// Maximum pixels at the largest short-edge tier. H3: 768*1344.
    pub max_area: u64,
    /// Allowed `width / height` range. H3: 1:4 .. 4:1 = `(0.25, 4.0)`.
    pub aspect: (f32, f32),
    /// Short-edge tiers the model is validated at; the **first** entry is the
    /// default tier (used for `CanvasSpec::ModelDefault`).
    pub short_edges: Vec<u32>,
    /// LTX: 1080 -> 1088 then crop (ltx §3.1). Exact canvases off the multiple
    /// are generated padded up and cropped back in post.
    pub pad_and_crop: bool,
    /// An opt-in tier above the trained pixel budget (H3 1080P), also listed
    /// in `short_edges`. `max_area` and [`CanvasCaps::area_at`] of the other
    /// tiers ignore it. `None` on every other model.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hd: Option<HdTier>,
}

/// An opt-in canvas tier with its own pixel budget ([`CanvasCaps::hd`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct HdTier {
    /// The delivered short edge (H3: 1080).
    pub short_edge: u32,
    /// Largest generation canvas area (H3: 1088*1920).
    pub max_area: u64,
    /// Longest clip at this tier, in frames (`None`: the model's frame
    /// grid). H3 1080P: 5 s (124 frames) by default.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_frames: Option<u32>,
    /// The longer cap the experimental feature flag [`FLAG_H3_1080P_LONG`]
    /// allows, while that flag is off (`None` once it is on, or when no
    /// flag lifts the cap). Refusals name it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub experimental_max_frames: Option<u32>,
}

/// The experimental feature flag that lifts the H3 1080P tier's clip length
/// from 5 s to 10 s (owner decision 2026-09-29; docs/serve/console.md §7).
pub const FLAG_H3_1080P_LONG: &str = "h3_1080p_long";

/// H3 1080P clip cap without the flag (seconds).
pub const H3_1080P_MAX_S: usize = 5;
/// H3 1080P clip cap with [`FLAG_H3_1080P_LONG`] on (seconds).
pub const H3_1080P_LONG_MAX_S: usize = 10;

impl HdTier {
    /// H3 1080P: short edge 1080, generated at 1088 on the short side and
    /// centre-cropped back; at most 1088x1920 pixels
    /// (docs/serve/h3-1080p-and-upscaler.md). Clips up to 5 s; 10 s with
    /// the [`FLAG_H3_1080P_LONG`] feature flag ([`HdTier::with_long`]).
    pub fn h3_1080p() -> Self {
        Self {
            short_edge: h3::H3_SHORT_EDGE_1080P as u32,
            max_area: h3::H3_MAX_PIXELS_1080P as u64,
            max_frames: Some(h3::align_num_frames(H3_1080P_MAX_S * h3::H3_FPS) as u32),
            experimental_max_frames: Some(h3::align_num_frames(H3_1080P_LONG_MAX_S * h3::H3_FPS) as u32),
        }
    }

    /// The tier with the [`FLAG_H3_1080P_LONG`] flag applied: `on` allows
    /// up to 10 s, off keeps 5 s. Idempotent, so a gateway can re-apply it
    /// to caps a worker already flagged.
    pub fn with_long(self, on: bool) -> Self {
        let short = h3::align_num_frames(H3_1080P_MAX_S * h3::H3_FPS) as u32;
        let long = h3::align_num_frames(H3_1080P_LONG_MAX_S * h3::H3_FPS) as u32;
        if on {
            Self { max_frames: Some(long), experimental_max_frames: None, ..self }
        } else {
            Self { max_frames: Some(short), experimental_max_frames: Some(long), ..self }
        }
    }
}

/// Applies the experimental feature flags to a model's caps (what the
/// server negotiates against and the console's forms are built from).
/// `enabled(name)` answers whether a flag is on. Today one flag:
/// [`FLAG_H3_1080P_LONG`] on the H3 1080P tier.
pub fn apply_feature_flags(caps: &mut ModelCaps, enabled: &dyn Fn(&str) -> bool) {
    if caps.family == Family::H3 {
        if let Some(t) = caps.canvas.hd {
            caps.canvas.hd = Some(t.with_long(enabled(FLAG_H3_1080P_LONG)));
        }
    }
}

impl CanvasCaps {
    /// H3: multiple 32, `768*1344`, 1:4..4:1, the 768 tier.
    pub fn h3() -> Self {
        Self {
            multiple: h3::H3_CANVAS_MULTIPLE as u32,
            max_area: h3::H3_MAX_PIXELS as u64,
            aspect: (0.25, 4.0),
            short_edges: vec![h3::H3_SHORT_EDGE as u32],
            pad_and_crop: false,
            hd: None,
        }
    }

    /// Adds the H3 1080P tier ([`HdTier::h3_1080p`]) after the other tiers
    /// (the first tier stays the default).
    pub fn with_h3_1080p(mut self) -> Self {
        let t = HdTier::h3_1080p();
        if !self.short_edges.contains(&t.short_edge) {
            self.short_edges.push(t.short_edge);
        }
        self.hd = Some(t);
        self
    }

    /// Whether `short_edge` is the opt-in [`CanvasCaps::hd`] tier.
    pub fn is_hd(&self, short_edge: u32) -> bool {
        self.hd.is_some_and(|t| t.short_edge == short_edge)
    }

    /// Whether `width / height` lies within [`CanvasCaps::aspect`].
    pub fn aspect_ok(&self, width: u32, height: u32) -> bool {
        if width == 0 || height == 0 {
            return false;
        }
        let r = width as f64 / height as f64;
        let (lo, hi) = (self.aspect.0 as f64, self.aspect.1 as f64);
        r >= lo - 1e-6 && r <= hi + 1e-6
    }

    /// The pixel budget at `short_edge`: `max_area` scaled by
    /// `(short_edge / largest tier)^2`. For H3 768 this is `768*1344`; for 480
    /// it admits 832x480 at 16:9 (fal §6). The [`CanvasCaps::hd`] tier has its
    /// own budget and is not "the largest tier" for the others.
    pub fn area_at(&self, short_edge: u32) -> u64 {
        if let Some(t) = self.hd.filter(|t| t.short_edge == short_edge) {
            return t.max_area;
        }
        let top = self
            .short_edges
            .iter()
            .copied()
            .filter(|&s| !self.is_hd(s))
            .max()
            .unwrap_or(short_edge)
            .max(1);
        if short_edge >= top {
            return self.max_area;
        }
        let s = short_edge as f64 / top as f64;
        (self.max_area as f64 * s * s).floor() as u64
    }
}

/// Reference-input limits for `Ref2V`.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct RefLimits {
    pub images: u32,
    pub videos: u32,
    pub audio: u32,
    pub total: u32,
}

impl RefLimits {
    /// H3 ref2va: 9 images / 3 videos / 3 audio / 12 total.
    pub fn h3() -> Self {
        Self {
            images: 9,
            videos: 3,
            audio: 3,
            total: 12,
        }
    }
    /// LTX-2.5 reference-to-video (the Ingredients IC-LoRA): one reference
    /// image, the reference sheet (`ic_lora.py` takes one reference per
    /// conditioning; the sheet carries every subject in its panels).
    pub fn ltx_ingredients() -> Self {
        Self {
            images: 1,
            videos: 0,
            audio: 0,
            total: 1,
        }
    }
    /// No references at all.
    pub fn none() -> Self {
        Self::default()
    }
}

/// Which request knobs a model honours per request. A knob the caps do not
/// honour is refused, never dropped (design §3.2 rule 7).
/// `SamplingOverrides::boundary_ratio` is honoured exactly when `guidance_2` is
/// (both only exist for two-expert Wan 2.2 MoE).
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct KnobCaps {
    pub seed: bool,
    pub negative: bool,
    pub steps: bool,
    pub guidance: bool,
    pub guidance_2: bool,
    pub flow_shift: bool,
    /// `SamplingOverrides::reference_strength` and `reference_lora_strength`
    /// (LTX-2.5 reference-to-video only).
    #[serde(default)]
    pub reference_strength: bool,
}

impl KnobCaps {
    /// Every sampling knob honoured (the reference strengths are a
    /// reference-to-video model's own and stay off).
    pub fn all() -> Self {
        Self {
            seed: true,
            negative: true,
            steps: true,
            guidance: true,
            guidance_2: true,
            flow_shift: true,
            reference_strength: false,
        }
    }
}

/// Streaming shape of a model.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StreamCaps {
    /// SF-Wan block rollout.
    Causal { block_frames: u32, target_fps: u32 },
    /// Clip-queue playout (H3, LTX, FastWan): clip length bounds in seconds.
    Clip { min_s: f32, max_s: f32 },
}
