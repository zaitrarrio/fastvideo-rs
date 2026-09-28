//! `ApiError`, `ErrorKind`, `GapId` (design §3.3).
//!
//! One error model for every adapter. Each adapter renders an [`ApiError`]
//! into its own envelope and status table through
//! `BatchProtocol::render_error` (design §4.6); [`ErrorKind::http_status`] is
//! only the canonical (native `/fv/v1/*`) status, used when an adapter has no
//! row of its own.

use serde::{Deserialize, Serialize};

/// What went wrong, independent of any wire format.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ErrorKind {
    /// Malformed or out-of-range input (a validation error).
    InvalidRequest,
    /// A well-formed request the engine cannot serve yet (or ever); see [`GapId`].
    Unsupported(GapId),
    Unauthorized,
    Forbidden,
    NotFound,
    /// Cancel/delete of a job that already finished.
    AlreadyCompleted,
    /// The resource is in a state that forbids the operation (e.g. MiniMax
    /// `DELETE` on a running task, Reactor session not READY).
    Conflict,
    PayloadTooLarge,
    UnsupportedMedia,
    ContentFiltered,
    RateLimited,
    QueueFull,
    /// Models are still loading (warm start).
    Loading,
    Timeout,
    Cancelled,
    /// The engine failed while generating.
    EngineFailed,
    Internal,
}

impl ErrorKind {
    /// The canonical HTTP status (native API). Adapters own their own tables
    /// (design §4.1-§4.6) and may differ, e.g. fal answers 422 for
    /// `InvalidRequest` and MiniMax 529 for `Loading`.
    pub fn http_status(&self) -> u16 {
        match self {
            ErrorKind::InvalidRequest | ErrorKind::Unsupported(_) => 400,
            ErrorKind::Unauthorized => 401,
            ErrorKind::Forbidden => 403,
            ErrorKind::NotFound => 404,
            ErrorKind::AlreadyCompleted | ErrorKind::Conflict | ErrorKind::Cancelled => 409,
            ErrorKind::PayloadTooLarge => 413,
            ErrorKind::UnsupportedMedia => 415,
            ErrorKind::ContentFiltered => 422,
            ErrorKind::RateLimited | ErrorKind::QueueFull => 429,
            ErrorKind::Loading => 503,
            ErrorKind::Timeout => 504,
            ErrorKind::EngineFailed | ErrorKind::Internal => 500,
        }
    }

    /// Stable snake_case name (the serde name; `Unsupported` is `"unsupported"`).
    pub fn code(&self) -> &'static str {
        match self {
            ErrorKind::InvalidRequest => "invalid_request",
            ErrorKind::Unsupported(_) => "unsupported",
            ErrorKind::Unauthorized => "unauthorized",
            ErrorKind::Forbidden => "forbidden",
            ErrorKind::NotFound => "not_found",
            ErrorKind::AlreadyCompleted => "already_completed",
            ErrorKind::Conflict => "conflict",
            ErrorKind::PayloadTooLarge => "payload_too_large",
            ErrorKind::UnsupportedMedia => "unsupported_media",
            ErrorKind::ContentFiltered => "content_filtered",
            ErrorKind::RateLimited => "rate_limited",
            ErrorKind::QueueFull => "queue_full",
            ErrorKind::Loading => "loading",
            ErrorKind::Timeout => "timeout",
            ErrorKind::Cancelled => "cancelled",
            ErrorKind::EngineFailed => "engine_failed",
            ErrorKind::Internal => "internal",
        }
    }

    /// 4xx: the client's request (or its state) is at fault.
    pub fn is_client_error(&self) -> bool {
        (400..500).contains(&self.http_status())
    }

    /// Whether a client may retry the same request later and expect success.
    pub fn is_retryable(&self) -> bool {
        matches!(
            self,
            ErrorKind::RateLimited | ErrorKind::QueueFull | ErrorKind::Loading | ErrorKind::Timeout
        )
    }
}

/// The error every protocol function returns.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, thiserror::Error)]
#[error("{message}")]
pub struct ApiError {
    pub kind: ErrorKind,
    pub message: String,
    /// The offending request field, in the adapter's own naming when known.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub param: Option<String>,
    /// `Retry-After` seconds for retryable kinds.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub retry_after_s: Option<u32>,
}

impl ApiError {
    pub fn new(kind: ErrorKind, message: impl Into<String>) -> Self {
        Self {
            kind,
            message: message.into(),
            param: None,
            retry_after_s: None,
        }
    }
    /// Sets [`ApiError::param`].
    pub fn with_param(mut self, param: impl Into<String>) -> Self {
        self.param = Some(param.into());
        self
    }
    /// Sets [`ApiError::retry_after_s`].
    pub fn with_retry_after(mut self, seconds: u32) -> Self {
        self.retry_after_s = Some(seconds);
        self
    }

    pub fn invalid(message: impl Into<String>) -> Self {
        Self::new(ErrorKind::InvalidRequest, message)
    }
    /// `InvalidRequest` naming the field.
    pub fn invalid_param(param: impl Into<String>, message: impl Into<String>) -> Self {
        Self::invalid(message).with_param(param)
    }
    /// `Unsupported(gap)` with the gap's default message.
    pub fn unsupported(gap: GapId) -> Self {
        Self::new(ErrorKind::Unsupported(gap), gap.default_message())
    }
    /// `Unsupported(gap)` with a custom message.
    pub fn unsupported_msg(gap: GapId, message: impl Into<String>) -> Self {
        Self::new(ErrorKind::Unsupported(gap), message)
    }
    pub fn unauthorized(message: impl Into<String>) -> Self {
        Self::new(ErrorKind::Unauthorized, message)
    }
    pub fn forbidden(message: impl Into<String>) -> Self {
        Self::new(ErrorKind::Forbidden, message)
    }
    pub fn not_found(message: impl Into<String>) -> Self {
        Self::new(ErrorKind::NotFound, message)
    }
    pub fn already_completed(message: impl Into<String>) -> Self {
        Self::new(ErrorKind::AlreadyCompleted, message)
    }
    pub fn conflict(message: impl Into<String>) -> Self {
        Self::new(ErrorKind::Conflict, message)
    }
    pub fn payload_too_large(message: impl Into<String>) -> Self {
        Self::new(ErrorKind::PayloadTooLarge, message)
    }
    pub fn unsupported_media(message: impl Into<String>) -> Self {
        Self::new(ErrorKind::UnsupportedMedia, message)
    }
    pub fn content_filtered(message: impl Into<String>) -> Self {
        Self::new(ErrorKind::ContentFiltered, message)
    }
    pub fn rate_limited(message: impl Into<String>) -> Self {
        Self::new(ErrorKind::RateLimited, message)
    }
    pub fn queue_full(message: impl Into<String>) -> Self {
        Self::new(ErrorKind::QueueFull, message)
    }
    /// `Loading` with `Retry-After: 1` (design §6.3).
    pub fn loading(message: impl Into<String>) -> Self {
        Self::new(ErrorKind::Loading, message).with_retry_after(1)
    }
    pub fn timeout(message: impl Into<String>) -> Self {
        Self::new(ErrorKind::Timeout, message)
    }
    pub fn cancelled(message: impl Into<String>) -> Self {
        Self::new(ErrorKind::Cancelled, message)
    }
    pub fn engine_failed(message: impl Into<String>) -> Self {
        Self::new(ErrorKind::EngineFailed, message)
    }
    pub fn internal(message: impl Into<String>) -> Self {
        Self::new(ErrorKind::Internal, message)
    }

    /// Shorthand for `self.kind.http_status()`.
    pub fn http_status(&self) -> u16 {
        self.kind.http_status()
    }
    /// The gap, when this is `Unsupported`.
    pub fn gap(&self) -> Option<GapId> {
        match self.kind {
            ErrorKind::Unsupported(g) => Some(g),
            _ => None,
        }
    }
}

/// Engine gaps that currently answer 4xx; each maps to a work package in
/// design §8 (or is permanent). See the table in design §4.7.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GapId {
    /// E3: duration 4 / 107 frames.
    H3FourSeconds,
    /// Permanent (no upscaler).
    #[serde(rename = "h3_resolution_2k")]
    H3Resolution2K,
    /// 1080P on a model without the opt-in native H3 1080P tier (h3-draft,
    /// Ref2V, or a GPU below the tier's memory plan). The wire name is
    /// historical (fal hosts 1080P as a latent refinement).
    #[serde(rename = "h3_refine_1080p")]
    H3Refine1080P,
    /// E10: `target_audio_url` / director `audio_url`.
    H3TargetAudio,
    /// Config: ref2va DiT not resident (E11).
    H3Ref2vaNotLoaded,
    /// E9: `last_frame_uri`.
    LtxKeyframes,
    /// E5.
    #[serde(rename = "ltx25_i2v")]
    Ltx25I2V,
    /// `duration: null` (needs a duration head).
    LtxAutoDuration,
    /// `camera_motion`.
    LtxCameraMotion,
    /// E4: 25/48/50 until validated.
    LtxFps,
    /// A2V / retake / extend / HDR / reframe.
    LtxEndpoint,
    /// `mm_file://`, OpenAI `file_id`.
    ProviderFiles,
    /// H3 steps fixed by recipe.
    PerRequestSteps,
    /// Any LoRA other than the startup adapter.
    Lora,
}

impl GapId {
    pub const ALL: [GapId; 14] = [
        GapId::H3FourSeconds,
        GapId::H3Resolution2K,
        GapId::H3Refine1080P,
        GapId::H3TargetAudio,
        GapId::H3Ref2vaNotLoaded,
        GapId::LtxKeyframes,
        GapId::Ltx25I2V,
        GapId::LtxAutoDuration,
        GapId::LtxCameraMotion,
        GapId::LtxFps,
        GapId::LtxEndpoint,
        GapId::ProviderFiles,
        GapId::PerRequestSteps,
        GapId::Lora,
    ];

    /// Stable snake_case name (the serde name).
    pub fn code(&self) -> &'static str {
        match self {
            GapId::H3FourSeconds => "h3_four_seconds",
            GapId::H3Resolution2K => "h3_resolution_2k",
            GapId::H3Refine1080P => "h3_refine_1080p",
            GapId::H3TargetAudio => "h3_target_audio",
            GapId::H3Ref2vaNotLoaded => "h3_ref2va_not_loaded",
            GapId::LtxKeyframes => "ltx_keyframes",
            GapId::Ltx25I2V => "ltx25_i2v",
            GapId::LtxAutoDuration => "ltx_auto_duration",
            GapId::LtxCameraMotion => "ltx_camera_motion",
            GapId::LtxFps => "ltx_fps",
            GapId::LtxEndpoint => "ltx_endpoint",
            GapId::ProviderFiles => "provider_files",
            GapId::PerRequestSteps => "per_request_steps",
            GapId::Lora => "lora",
        }
    }

    /// The design §8 package that closes the gap; `None` when permanent or
    /// not planned.
    pub fn work_package(&self) -> Option<&'static str> {
        match self {
            GapId::H3FourSeconds => Some("E3"),
            GapId::H3TargetAudio => Some("E10"),
            GapId::H3Ref2vaNotLoaded => Some("E11"),
            GapId::LtxKeyframes => Some("E9"),
            GapId::Ltx25I2V => Some("E5"),
            GapId::LtxFps => Some("E4"),
            GapId::H3Resolution2K
            | GapId::H3Refine1080P
            | GapId::LtxAutoDuration
            | GapId::LtxCameraMotion
            | GapId::LtxEndpoint
            | GapId::ProviderFiles
            | GapId::PerRequestSteps
            | GapId::Lora => None,
        }
    }

    /// Whether no package is planned to close this gap.
    pub fn is_permanent(&self) -> bool {
        self.work_package().is_none()
    }

    /// The request field that usually triggers the gap (generic naming).
    pub fn param(&self) -> Option<&'static str> {
        match self {
            GapId::H3FourSeconds | GapId::LtxAutoDuration => Some("duration"),
            GapId::H3Resolution2K | GapId::H3Refine1080P => Some("resolution"),
            GapId::H3TargetAudio => Some("target_audio_url"),
            GapId::H3Ref2vaNotLoaded => Some("references"),
            GapId::LtxKeyframes => Some("last_frame_uri"),
            GapId::Ltx25I2V => Some("image_uri"),
            GapId::LtxCameraMotion => Some("camera_motion"),
            GapId::LtxFps => Some("fps"),
            GapId::LtxEndpoint => None,
            GapId::ProviderFiles => Some("url"),
            GapId::PerRequestSteps => Some("num_inference_steps"),
            GapId::Lora => Some("lora"),
        }
    }

    /// A client-facing sentence explaining the gap.
    pub fn default_message(&self) -> &'static str {
        match self {
            GapId::H3FourSeconds => {
                "a 4 second duration is not supported by this server yet; use 5 to 15 seconds"
            }
            GapId::H3Resolution2K => "2K resolution is not supported by this server",
            GapId::H3Refine1080P => "1080P resolution is not supported by this server",
            GapId::H3TargetAudio => {
                "target / conditioning audio is not supported by this server yet"
            }
            GapId::H3Ref2vaNotLoaded => "reference-to-video is not enabled on this server",
            GapId::LtxKeyframes => "last-frame keyframes are not supported by this server yet",
            GapId::Ltx25I2V => "image-to-video is not supported for this model yet",
            GapId::LtxAutoDuration => {
                "automatic duration is not supported; send an explicit duration"
            }
            GapId::LtxCameraMotion => "camera_motion is not supported by this server",
            GapId::LtxFps => "this frame rate is not supported by this server yet; use 24",
            GapId::LtxEndpoint => "endpoint not available for the account",
            GapId::ProviderFiles => {
                "provider file references are not supported; send a URL or data URI"
            }
            GapId::PerRequestSteps => {
                "num_inference_steps is fixed by the model recipe and cannot be set per request"
            }
            GapId::Lora => "only the LoRA adapter loaded at startup is available",
        }
    }
}

impl std::fmt::Display for GapId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.code())
    }
}
