//! Normalized protocol model shared by every fv-serve adapter (design §3).
//!
//! - [`request`]: the normalized [`GenerationRequest`] every adapter produces.
//! - [`caps`]: [`ModelCaps`], the draft/turbo/max [`Tier`]s, and the frame-grid /
//!   canvas helpers.
//! - [`negotiate`](mod@negotiate): [`negotiate()`] turns request + caps +
//!   staged inputs into a [`ResolvedJob`], refusing engine gaps with
//!   [`GapId`]s.
//! - [`error`]: the [`ApiError`] model every adapter renders.
//! - [`job`]: the [`Job`] state machine and the [`JobStore`] trait.
//! - [`http`]: framework-free [`HttpReply`] and the [`BatchProtocol`],
//!   [`SubmitEndpoint`], [`JobView`] traits.
//! - [`stream`]: [`TrackSet`], [`StreamProtocol`], session types.
//! - [`av`]: raw [`RgbFrame`] / [`Pcm`] buffers.
//! - [`ingest`]: duplex streaming: [`DuplexCaps`], input tracks, decoded
//!   [`InputFrame`] / [`InputAudio`], the session [`SessionContext`].
//!
//! No tokio runtime and no axum: only `tokio::sync` for `JobStore::watch`.
//!
//! Owned by WP-01 (docs/serve/design.md §8).

pub mod av;
pub mod caps;
pub mod error;
pub mod http;
pub mod ingest;
pub mod job;
pub mod negotiate;
pub mod request;
pub mod stream;

pub use av::{Pcm, RgbFrame};
pub use caps::{
    apply_feature_flags, AudioCaps, CanvasCaps, FpsCaps, FrameGrid, HdTier, KnobCaps, ModelCaps, RefLimits, StreamCaps, Tier,
    FLAG_H3_1080P_LONG, H3_1080P_LONG_MAX_S, H3_1080P_MAX_S,
};
pub use error::{ApiError, ErrorKind, GapId};
pub use http::{
    BatchProtocol, ErrorCtx, HttpReply, JobView, NormalizeCtx, ReplyBody, SseEvent, SseFollow,
    SseSpec, SubmitEndpoint, UrlSigner, ViewCtx,
};
pub use ingest::{
    AudioInputCaps, DuplexCaps, DuplexSpec, InputAudio, InputCaps, InputFrame, InputVideoCodec, SessionContext,
    VideoInputCaps, CONTEXT_MAX_CHARS,
};
pub use job::{
    Artifact, ArtifactId, ArtifactLocation, CallbackSpec, Job, JobId, JobMetrics, JobSnapshot,
    JobState, JobStatus, JobStore, JobUpdate, KeyId, ListQuery, LogLevel, LogLine, Page, SortOrder,
    StoreError, TransitionError,
};
pub use negotiate::{
    canvas_for_aspect, draw_seed, A2V_AUDIO_MAX_S, A2V_AUDIO_MIN_S, A2V_AUDIO_RATES, effective_canvas, h3_1080p_canvas, is_hd_canvas, negotiate, negotiate_noted, precheck,
    resolve_canvas, resolve_canvas_noted, resolve_frames,
    resolve_model, resolve_tier, route_task, AudioPlan, MediaProbe, PostProcess, ResolvedCanvas, ResolvedJob,
    StagedInputs, StagedMedia,
};
pub use request::{
    Anchor, AudioInput, AudioOut, AudioRole, CanvasSpec, Family, GenerationRequest, Keyframe,
    Length, MediaKind, MediaRef, ModelId, OutputOptions, ProtocolId, Ratio, Reference,
    SamplingOverrides, Snap, Task, TimingSpec, UploadId,
};
pub use stream::{
    AudioTrack, CausalLimits, Continuity, EndReason, SessionSpec, SessionState, StreamProtocol, TrackSet,
    VideoTrack, WIRE_AUDIO_RATE,
};
