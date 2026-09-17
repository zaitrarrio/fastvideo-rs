//! Errors for the Burn Wan graph.

use thiserror::Error;

#[derive(Debug, Error)]
pub enum BurnError {
    #[error("{0}")]
    Msg(String),
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Image(#[from] image::ImageError),
    #[error(transparent)]
    Loader(#[from] fastvideo_loader::LoaderError),
}

impl BurnError {
    pub fn msg(s: impl Into<String>) -> Self {
        Self::Msg(s.into())
    }
}

pub type Result<T> = std::result::Result<T, BurnError>;
