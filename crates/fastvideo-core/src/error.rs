use thiserror::Error;

#[derive(Debug, Error)]
pub enum FastVideoError {
    #[error("unknown model id `{0}` — no Wan/FastWan registry match")]
    UnknownModel(String),
    #[error("{component} is not implemented yet ({detail})")]
    NotImplemented { component: String, detail: String },
    #[error("{0}")]
    Message(String),
}

pub type Result<T> = std::result::Result<T, FastVideoError>;
