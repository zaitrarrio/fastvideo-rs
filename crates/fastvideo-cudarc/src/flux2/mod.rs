//! Flux2 cudarc generate path: DiT + 2D VAE + Qwen3/Mistral3 text + flow-match Euler.

pub mod pipeline;
pub mod text;
pub mod transformer;
pub mod vae;

pub use pipeline::{Flux2Pipeline, GenerateConfig};
pub use text::{Flux2TextEncoder, Qwen3Encoder};
pub use transformer::{Flux2Transformer2D, RopeHostStats};
pub use vae::AutoencoderKlFlux2;
