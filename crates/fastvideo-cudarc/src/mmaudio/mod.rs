//! MMAudio T2A/V2A. Spec: docs/ports/mmaudio.md.

pub mod pipeline;
pub mod transformer;

pub use pipeline::{MmAudioPipeline, MmAudioRequest};
pub use transformer::MmAudioTransformer;
