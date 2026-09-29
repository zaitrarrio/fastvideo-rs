//! Normalized `GenerationRequest`, `Task`, `CanvasSpec`, `TimingSpec`, media
//! refs (design §3.1).
//!
//! Every adapter's `SubmitEndpoint::normalize` produces one
//! [`GenerationRequest`]; `negotiate()` turns it into a `ResolvedJob`.

use std::fmt;
use std::str::FromStr;

use serde::{Deserialize, Serialize};

use crate::error::ApiError;
use crate::job::CallbackSpec;

/// Engine model id after alias resolution.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ModelId(pub String);

impl ModelId {
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for ModelId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl From<&str> for ModelId {
    fn from(s: &str) -> Self {
        Self(s.to_owned())
    }
}

/// Which external (or native) API a request or job belongs to.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProtocolId {
    /// FastVideo `/v1/videos` family (design §4.1).
    #[serde(rename = "openai_videos")]
    OpenAiVideos,
    /// FastWan `/generate` shape (design §4.2).
    #[serde(rename = "fastwan")]
    FastWan,
    /// MiniMax V2 (design §4.3).
    #[serde(rename = "minimax_v2")]
    MiniMaxV2,
    /// fal queue and sync (design §4.4).
    Fal,
    /// fal `minimax/h3-max/director` WMA session (design §5.6).
    FalDirector,
    /// LTX `/v1/*` sync (design §4.5).
    LtxV1,
    /// LTX `/v2/*` async (design §4.5).
    LtxV2,
    /// Reactor local runtime (design §5.7).
    Reactor,
    /// Native `/fv/v1/*`.
    Native,
}

impl ProtocolId {
    pub const ALL: [ProtocolId; 9] = [
        ProtocolId::OpenAiVideos,
        ProtocolId::FastWan,
        ProtocolId::MiniMaxV2,
        ProtocolId::Fal,
        ProtocolId::FalDirector,
        ProtocolId::LtxV1,
        ProtocolId::LtxV2,
        ProtocolId::Reactor,
        ProtocolId::Native,
    ];

    /// Stable snake_case name (the serde name), for logs, metrics and manifests.
    pub fn as_str(&self) -> &'static str {
        match self {
            ProtocolId::OpenAiVideos => "openai_videos",
            ProtocolId::FastWan => "fastwan",
            ProtocolId::MiniMaxV2 => "minimax_v2",
            ProtocolId::Fal => "fal",
            ProtocolId::FalDirector => "fal_director",
            ProtocolId::LtxV1 => "ltx_v1",
            ProtocolId::LtxV2 => "ltx_v2",
            ProtocolId::Reactor => "reactor",
            ProtocolId::Native => "native",
        }
    }

    /// Default job retention (design §3.4): MiniMax 7 d; LTX, fal and
    /// FastVideo 24 h (FastVideo: until DELETE, capped at 24 h). Everything
    /// else 24 h. Configurable by the server.
    pub fn default_retention(&self) -> std::time::Duration {
        const HOUR: u64 = 3600;
        match self {
            ProtocolId::MiniMaxV2 => std::time::Duration::from_secs(7 * 24 * HOUR),
            _ => std::time::Duration::from_secs(24 * HOUR),
        }
    }
}

impl fmt::Display for ProtocolId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Model family; selects family-specific negotiation rules.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Family {
    /// H3, FastH3, Sol-H3.
    H3,
    /// LTX-2, 2.3, 2.5.
    Ltx2,
    /// Wan, FastWan, SF-Wan causal, TI2V-5B.
    Wan,
    /// MMAudio (V2A sidecar).
    MmAudio,
    /// Transport test models (the duplex loopback echo): no weights.
    Loopback,
}

/// What the client asked for, before capability checks.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Task {
    /// Text only.
    T2V,
    /// Exactly one first-frame image.
    I2V,
    /// Last-only or first+last (H3 fl2va, LTX `last_frame_uri`).
    Keyframes,
    /// H3 ref2va: ordered image/video/audio references.
    Ref2V,
    /// Audio drives output (LTX audio-to-video): a driving `audio_in`
    /// (`AudioRole::Drive`), an optional first-frame image (and a last frame
    /// with it), the prompt required without an image.
    A2V,
    /// LTX extend: continue a source video (`GenerationRequest::edit`,
    /// [`EditOp::Extend`]) at its end or its start.
    Extend,
    /// LTX retake: regenerate a time window of a source video
    /// (`GenerationRequest::edit`, [`EditOp::Retake`]), video, audio or both.
    Retake,
    /// LTX edit endpoint. Unsupported today.
    V2V,
}

impl Task {
    /// The LTX edit endpoints no engine path serves (`GapId::LtxEndpoint`).
    /// Audio-to-video, retake and extend are served by LTX-2.5; other LTX
    /// models answer the same gap for them (`negotiate` rule 2).
    pub fn is_edit_endpoint(&self) -> bool {
        matches!(self, Task::V2V)
    }

    /// Retake and extend: the task edits a source video.
    pub fn edits_video(&self) -> bool {
        matches!(self, Task::Retake | Task::Extend)
    }
}

/// The normalized request every adapter produces.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GenerationRequest {
    pub protocol: ProtocolId,
    /// As sent (alias); resolved via `ServeConfig.aliases` ([`crate::resolve_model`]).
    pub model: String,
    pub task: Task,
    pub prompt: String,
    pub negative_prompt: Option<String>,
    /// `None` -> the server draws one, recorded on the Job.
    pub seed: Option<u64>,
    pub canvas: CanvasSpec,
    pub timing: TimingSpec,
    pub keyframes: Vec<Keyframe>,
    /// Order preserved exactly as sent.
    pub references: Vec<Reference>,
    pub audio_in: Option<AudioInput>,
    pub audio_out: AudioOut,
    pub sampling: SamplingOverrides,
    pub output: OutputOptions,
    /// Retake / extend: the source video and the edit.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub edit: Option<VideoEdit>,
    /// Fields accepted but ignored (e.g. `"prompt_expansion_mode"`); logged and
    /// counted only. Not deserialized (a `&'static str` cannot be); a
    /// deserialized request has this empty.
    #[serde(default, skip_deserializing)]
    pub accepted_noop: Vec<&'static str>,
    /// Where status changes are delivered (fal `fal_webhook`, MiniMax
    /// `callback_url`); copied onto `Job::callback`. **Addition to design
    /// §3.1** (see the WP-01 notes in design §8).
    #[serde(default)]
    pub callback: Option<CallbackSpec>,
}

impl GenerationRequest {
    /// A text-to-video request with every other field at its model default.
    pub fn text(protocol: ProtocolId, model: impl Into<String>, prompt: impl Into<String>) -> Self {
        Self {
            protocol,
            model: model.into(),
            task: Task::T2V,
            prompt: prompt.into(),
            negative_prompt: None,
            seed: None,
            canvas: CanvasSpec::ModelDefault,
            timing: TimingSpec::default(),
            keyframes: Vec::new(),
            references: Vec::new(),
            audio_in: None,
            audio_out: AudioOut::ModelDefault,
            sampling: SamplingOverrides::default(),
            output: OutputOptions::default(),
            edit: None,
            accepted_noop: Vec::new(),
            callback: None,
        }
    }

    /// Every media ref the request carries, in order: keyframes, references,
    /// `audio_in`, then the edit's source video.
    pub fn media_refs(&self) -> impl Iterator<Item = &MediaRef> {
        self.keyframes
            .iter()
            .map(|k| &k.image)
            .chain(self.references.iter().map(|r| &r.media))
            .chain(self.audio_in.iter().map(|a| &a.media))
            .chain(self.edit.iter().map(|e| &e.video))
    }

    /// Records an accepted-but-ignored field once.
    pub fn note_noop(&mut self, field: &'static str) {
        if !self.accepted_noop.contains(&field) {
            self.accepted_noop.push(field);
        }
    }
}

/// How the output canvas is chosen.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CanvasSpec {
    /// FastVideo size/width/height, FastWan, LTX `"WxH"`.
    Exact {
        width: u32,
        height: u32,
    },
    /// fal/MiniMax ratio + 480P/768P.
    Aspect {
        ratio: Ratio,
        short_edge: u32,
    },
    /// fal i2v, MiniMax `"adaptive"`: the first image's aspect.
    FollowImage {
        short_edge: u32,
    },
    /// A landscape size, transposed when the first image is portrait (the
    /// LTX API audio-to-video default: "Portrait image → 1080x1920,
    /// landscape → 1920x1080; no image → 1920x1080").
    Oriented {
        width: u32,
        height: u32,
    },
    ModelDefault,
}

/// An aspect ratio `w:h`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Ratio {
    pub w: u32,
    pub h: u32,
}

impl Ratio {
    pub const R16_9: Ratio = Ratio { w: 16, h: 9 };
    pub const R9_16: Ratio = Ratio { w: 9, h: 16 };
    pub const R1_1: Ratio = Ratio { w: 1, h: 1 };
    pub const R4_3: Ratio = Ratio { w: 4, h: 3 };
    pub const R3_4: Ratio = Ratio { w: 3, h: 4 };

    pub fn new(w: u32, h: u32) -> Self {
        Self { w, h }
    }
    /// `w / h`.
    pub fn value(&self) -> f64 {
        self.w as f64 / self.h as f64
    }
}

impl fmt::Display for Ratio {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}", self.w, self.h)
    }
}

impl FromStr for Ratio {
    type Err = ApiError;
    /// Parses `"16:9"` (both sides positive integers).
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let bad = || {
            ApiError::invalid_param(
                "aspect_ratio",
                format!("invalid aspect ratio `{s}`; expected W:H"),
            )
        };
        let (w, h) = s.trim().split_once(':').ok_or_else(bad)?;
        let w: u32 = w.trim().parse().map_err(|_| bad())?;
        let h: u32 = h.trim().parse().map_err(|_| bad())?;
        if w == 0 || h == 0 {
            return Err(bad());
        }
        Ok(Self { w, h })
    }
}

/// Output length and frame rate.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TimingSpec {
    pub length: Length,
    /// `None` -> the model's default fps.
    pub fps: Option<u32>,
}

impl Default for TimingSpec {
    fn default() -> Self {
        Self {
            length: Length::ModelDefault,
            fps: None,
        }
    }
}

/// Output length.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Length {
    /// MiniMax/fal/LTX duration, FastVideo seconds.
    Seconds {
        value: f64,
        snap: Snap,
    },
    /// FastVideo num_frames (Exact), FastWan num_frames.
    Frames {
        value: u32,
        snap: Snap,
    },
    ModelDefault,
    /// LTX `duration: null` -> `Unsupported(LtxAutoDuration)`.
    Auto,
}

/// How a length lands on the model's frame grid.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Snap {
    /// Round up to the next grid point.
    AlignUp,
    /// Must already be on the grid (FastVideo explicit num_frames).
    Exact,
    /// Round up to the next grid point, or down to the longest the model
    /// makes when that is past its ceiling (an API whose listed durations
    /// can exceed a model's frame grid at some rates: LTX 20 s at 25 fps is
    /// 501 frames, the grid stops at 481). Below the floor, the floor.
    Nearest,
}

/// Opaque id of a file uploaded through `PUT /uploads/{token}`
/// (e.g. the `<token>` of `ltx://uploads/<token>`).
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct UploadId(pub String);

impl fmt::Display for UploadId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// Where a media input comes from; staged by serve-kit ingestion.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MediaRef {
    Http(url::Url),
    /// The full `data:<mime>;base64,...` URI.
    DataUri(String),
    Upload(UploadId),
    /// `mm_file://...`, OpenAI `file_id`: always `Unsupported(ProviderFiles)`.
    ProviderFile(String),
}

impl MediaRef {
    /// Classifies a client-sent string: `data:` URIs, `http(s)://` URLs,
    /// `ltx://uploads/<token>` uploads, and `mm_file://` provider files.
    /// Anything else is `InvalidRequest` naming `param`.
    pub fn parse(s: &str, param: &str) -> Result<Self, ApiError> {
        let t = s.trim();
        if t.starts_with("data:") {
            return Ok(MediaRef::DataUri(t.to_owned()));
        }
        if let Some(tok) = t.strip_prefix("ltx://uploads/") {
            if tok.is_empty() {
                return Err(ApiError::invalid_param(param, "empty upload token"));
            }
            return Ok(MediaRef::Upload(UploadId(tok.to_owned())));
        }
        if t.starts_with("mm_file://") {
            return Ok(MediaRef::ProviderFile(t.to_owned()));
        }
        match url::Url::parse(t) {
            Ok(u) if matches!(u.scheme(), "http" | "https") => Ok(MediaRef::Http(u)),
            _ => Err(ApiError::invalid_param(
                param,
                format!("`{param}` is not a valid URL or data URI"),
            )),
        }
    }
}

/// Kind of a media input.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MediaKind {
    Image,
    Video,
    Audio,
}

/// Which end of the clip a keyframe pins.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Anchor {
    First,
    Last,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Keyframe {
    pub at: Anchor,
    pub image: MediaRef,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Reference {
    pub kind: MediaKind,
    pub media: MediaRef,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct AudioInput {
    pub media: MediaRef,
    pub role: AudioRole,
    /// The adapter's own ceiling on the audio's length in seconds, below the
    /// protocol-wide one (audio-to-video: fal and the LTX API cap `pro` at
    /// 10 s, `negotiate` otherwise takes up to 20 s).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_s: Option<u32>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AudioRole {
    /// fal `target_audio_url`, director `audio_url`.
    TargetSoundtrack,
    /// LTX A2V.
    Drive,
    /// LTX retake (`replace_video`): the new audio of the retaken window,
    /// spliced into the source's audio, which then stays clean conditioning.
    Dub,
}

/// What audio the output should carry.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AudioOut {
    /// Native audio if the model has it, none otherwise.
    #[default]
    ModelDefault,
    /// LTX `generate_audio=false`.
    Silent,
    /// MMAudio V2A for a video-only model.
    Sidecar,
}

/// Per-request sampling knobs; `negotiate()` refuses any the caps don't honour.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct SamplingOverrides {
    pub steps: Option<u32>,
    pub guidance: Option<f32>,
    pub guidance_2: Option<f32>,
    pub flow_shift: Option<f64>,
    pub boundary_ratio: Option<f32>,
    /// Reference-to-video conditioning strength in `[0, 1]` (LTX IC-LoRA:
    /// the reference tokens' denoise mask is `1 − s`; 1 keeps them clean).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reference_strength: Option<f32>,
    /// Reference-to-video LoRA strength in `[0, 2]` (LTX IC-LoRA: the stage-1
    /// fuse strength; fal `ingredient_strength`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reference_lora_strength: Option<f32>,
}

impl SamplingOverrides {
    pub fn is_empty(&self) -> bool {
        self == &Self::default()
    }
}

/// A retake or extend of a source video (LTX `/retake`, `/extend`; fal
/// `fal-ai/ltx-2.3/{retake,extend}-video`).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct VideoEdit {
    /// The source video.
    pub video: MediaRef,
    pub op: EditOp,
}

/// What an edit does to its source.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum EditOp {
    /// Regenerate `[start_s, start_s + duration_s)` (clamped to the video).
    Retake {
        start_s: f64,
        duration_s: f64,
        mode: RetakeMode,
    },
    /// Add `duration_s` seconds at `at`, continuing from up to `context_s`
    /// seconds of the source (`None`: as much as fits).
    Extend {
        duration_s: f64,
        at: ExtendAt,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        context_s: Option<f64>,
    },
}

/// LTX `retake` `mode` / fal `retake_mode`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RetakeMode {
    ReplaceAudio,
    ReplaceVideo,
    #[default]
    ReplaceAudioAndVideo,
}

impl RetakeMode {
    pub const ALL: [RetakeMode; 3] = [RetakeMode::ReplaceAudio, RetakeMode::ReplaceVideo, RetakeMode::ReplaceAudioAndVideo];

    pub fn as_str(&self) -> &'static str {
        match self {
            RetakeMode::ReplaceAudio => "replace_audio",
            RetakeMode::ReplaceVideo => "replace_video",
            RetakeMode::ReplaceAudioAndVideo => "replace_audio_and_video",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|m| m.as_str() == s)
    }

    /// `(regenerate_video, regenerate_audio)` (LTX-Desktop `_resolve_retake_mode`).
    pub fn regenerates(&self) -> (bool, bool) {
        match self {
            RetakeMode::ReplaceAudio => (false, true),
            RetakeMode::ReplaceVideo => (true, false),
            RetakeMode::ReplaceAudioAndVideo => (true, true),
        }
    }
}

/// LTX `extend` `mode` / fal `mode`: where the new frames go.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExtendAt {
    Start,
    #[default]
    End,
}

impl ExtendAt {
    pub fn as_str(&self) -> &'static str {
        match self {
            ExtendAt::Start => "start",
            ExtendAt::End => "end",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "start" => Some(ExtendAt::Start),
            "end" => Some(ExtendAt::End),
            _ => None,
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct OutputOptions {
    /// fal `sync_mode`: return the video as a data URI.
    pub inline_data_uri: bool,
}
