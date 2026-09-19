//! Flux2 cudarc generate path: DiT + 2D VAE + flow-match Euler.

pub mod pipeline;
pub mod transformer;
pub mod vae;

pub use pipeline::{Flux2Pipeline, GenerateConfig};
pub use transformer::Flux2Transformer2D;
pub use vae::AutoencoderKlFlux2;
