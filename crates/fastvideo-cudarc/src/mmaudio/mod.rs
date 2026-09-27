//! MMAudio V2A / T2A (hkchengrex/MMAudio `large_44k_v2`). Spec: docs/ports/mmaudio.md.

pub mod bigvgan;
pub mod clip;
mod layers;
pub mod pipeline;
#[cfg(test)]
mod reference_tests;
pub mod sidecar;
pub mod synchformer;
pub mod transformer;
pub mod vae;

pub use pipeline::{MmAudioOutput, MmAudioPipeline, MmAudioRequest, VideoFrames};
pub use synchformer::Synchformer;
pub use transformer::MmAudioTransformer;
