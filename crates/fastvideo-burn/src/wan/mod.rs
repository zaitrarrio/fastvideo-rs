//! Native Burn Wan T2V graph (`ndarray` CPU or CUDA via `--features cuda`).

pub mod nn;
pub mod pipeline;
pub mod transformer;
pub mod umt5;
pub mod vae;
pub mod weights;

pub use pipeline::{GenerateConfig, WanPipeline};
pub use transformer::WanTransformer3D;
pub use umt5::Umt5Encoder;
pub use vae::AutoencoderKlWan;
