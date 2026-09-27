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

/// Model tier exposed by the public APIs (design §0.3).
///
/// Public names map onto tiers through `ServeConfig.aliases` or
/// [`crate::resolve_tier`]: `h3-max` / `MiniMax-H3-Max` / `ltx-2-5-pro` /
/// `ltx-2-3-pro` -> `Max`; `h3-turbo` / `MiniMax-H3-Turbo` / `ltx-turbo` ->
/// `Turbo`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Tier {
    /// Highest quality: full step count, non-lossy attention, full VAE.
    Max,
    /// Fastest generation that still passes the quality gate.
    Turbo,
}

impl Tier {
    pub fn as_str(&self) -> &'static str {
        match self {
            Tier::Max => "max",
            Tier::Turbo => "turbo",
        }
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

    /// H3 `17n+5` over 5..15 s at 24 fps (124..=362), default 5 s. Built from
    /// `fastvideo_models::h3::config` (`align_num_frames` and constants).
    pub fn h3() -> Self {
        let min = h3::align_num_frames(h3::H3_MIN_DURATION_S * h3::H3_FPS) as u32;
        let max = h3::align_num_frames(h3::H3_MAX_DURATION_S * h3::H3_FPS) as u32;
        Self {
            step: h3::H3_FRAMES_PER_CHUNK as u32,
            offset: h3::H3_LATENTS_PER_CHUNK as u32,
            min,
            max,
            default: min,
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
        }
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
    /// it admits 832x480 at 16:9 (fal §6).
    pub fn area_at(&self, short_edge: u32) -> u64 {
        let top = self
            .short_edges
            .iter()
            .copied()
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
}

impl KnobCaps {
    /// Every knob honoured.
    pub fn all() -> Self {
        Self {
            seed: true,
            negative: true,
            steps: true,
            guidance: true,
            guidance_2: true,
            flow_shift: true,
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
