//! Normalized protocol model shared by every fv-serve adapter (design §3).
//!
//! - [`request`]: the normalized [`GenerationRequest`] every adapter produces.
//! - [`caps`]: [`ModelCaps`] and the frame-grid / canvas helpers.
//! - [`negotiate`](mod@negotiate): [`negotiate()`] turns request + caps +
//!   staged inputs into a [`ResolvedJob`], refusing engine gaps with
//!   [`GapId`]s.
//! - [`error`]: the [`ApiError`] model every adapter renders.
//! - [`job`]: the [`Job`] state machine and the [`JobStore`] trait.
//! - [`http`]: framework-free [`HttpReply`] and the [`BatchProtocol`],
//!   [`SubmitEndpoint`], [`JobView`] traits.
//! - [`stream`]: [`TrackSet`], [`StreamProtocol`], session types.
//! - [`av`]: raw [`RgbFrame`] / [`Pcm`] buffers.
//!
//! No tokio runtime and no axum: only `tokio::sync` for `JobStore::watch`.
//!
//! Owned by WP-01 (docs/serve/design.md §8).

pub mod av;
pub mod caps;
pub mod error;
pub mod http;
pub mod job;
pub mod negotiate;
pub mod request;
pub mod stream;

pub use av::{Pcm, RgbFrame};
pub use caps::{
    AudioCaps, CanvasCaps, FpsCaps, FrameGrid, KnobCaps, ModelCaps, RefLimits, StreamCaps,
};
pub use error::{ApiError, ErrorKind, GapId};
pub use http::{
    BatchProtocol, ErrorCtx, HttpReply, JobView, NormalizeCtx, ReplyBody, SseEvent, SseFollow,
    SseSpec, SubmitEndpoint, UrlSigner, ViewCtx,
};
pub use job::{
    Artifact, ArtifactId, ArtifactLocation, CallbackSpec, Job, JobId, JobMetrics, JobSnapshot,
    JobState, JobStatus, JobStore, JobUpdate, KeyId, ListQuery, LogLevel, LogLine, Page, SortOrder,
    StoreError, TransitionError,
};
pub use negotiate::{
    canvas_for_aspect, draw_seed, negotiate, precheck, resolve_canvas, resolve_frames,
    resolve_model, AudioPlan, MediaProbe, PostProcess, ResolvedCanvas, ResolvedJob, StagedInputs,
    StagedMedia,
};
pub use request::{
    Anchor, AudioInput, AudioOut, AudioRole, CanvasSpec, Family, GenerationRequest, Keyframe,
    Length, MediaKind, MediaRef, ModelId, OutputOptions, ProtocolId, Ratio, Reference,
    SamplingOverrides, Snap, Task, TimingSpec, UploadId,
};
pub use stream::{
    AudioTrack, Continuity, EndReason, SessionSpec, SessionState, StreamProtocol, TrackSet,
    VideoTrack, WIRE_AUDIO_RATE,
};
