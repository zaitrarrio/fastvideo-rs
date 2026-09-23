//! SD 3.5 T2I. Spec: docs/ports/sd35.md.

pub mod pipeline;
pub mod transformer;

pub use pipeline::{Sd35Pipeline, Sd35Request};
pub use transformer::Sd35Transformer;
