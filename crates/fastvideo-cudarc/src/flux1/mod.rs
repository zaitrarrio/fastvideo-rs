//! FLUX.1 cudarc generate path: DiT + SD3 VAE + CLIP-L/T5 + flow-match Euler.

pub mod pipeline;
pub mod text;
pub mod transformer;

pub use pipeline::{Flux1Pipeline, GenerateConfig};
pub use text::Flux1TextEncoder;
pub use transformer::Flux1Transformer2D;
