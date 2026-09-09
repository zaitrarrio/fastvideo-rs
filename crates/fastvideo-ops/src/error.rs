use thiserror::Error;

#[derive(Debug, Error)]
pub enum OpsError {
    #[error("{op} is not implemented for the {backend} backend")]
    NotImplemented {
        backend: &'static str,
        op: &'static str,
    },
    #[error("shape mismatch: {0}")]
    Shape(String),
    #[error("invalid dtype: {0}")]
    DType(String),
    #[error("{0}")]
    Message(String),
}

impl OpsError {
    pub fn not_implemented(backend: &'static str, op: &'static str) -> Self {
        Self::NotImplemented { backend, op }
    }
}
