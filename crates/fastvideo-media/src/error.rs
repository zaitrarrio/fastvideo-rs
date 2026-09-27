//! The one error type of this crate.

use std::path::PathBuf;

pub type Result<T, E = MediaError> = std::result::Result<T, E>;

#[derive(Debug, thiserror::Error)]
pub enum MediaError {
    /// A caller-supplied value is out of range (bad fps, odd canvas, ...).
    #[error("invalid argument: {0}")]
    InvalidArgument(String),
    /// A backend or feature that this build or host does not have.
    #[error("unsupported: {0}")]
    Unsupported(String),
    /// An external tool (ffmpeg, ffprobe, mkfifo) is missing or failed.
    #[error("{tool}: {message}")]
    Tool { tool: String, message: String },
    #[error("encode: {0}")]
    Encode(String),
    #[error("decode: {0}")]
    Decode(String),
    /// A container or bitstream could not be parsed.
    #[error("parse {path:?}: {message}")]
    Parse { path: Option<PathBuf>, message: String },
    #[error("resample: {0}")]
    Resample(String),
    #[error(transparent)]
    Io(#[from] std::io::Error),
}

impl MediaError {
    pub(crate) fn invalid(msg: impl Into<String>) -> Self {
        Self::InvalidArgument(msg.into())
    }

    pub(crate) fn tool(tool: impl Into<String>, message: impl Into<String>) -> Self {
        Self::Tool { tool: tool.into(), message: message.into() }
    }

    pub(crate) fn parse(path: Option<&std::path::Path>, message: impl Into<String>) -> Self {
        Self::Parse { path: path.map(|p| p.to_path_buf()), message: message.into() }
    }
}
