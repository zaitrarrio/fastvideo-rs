//! Cosmos3-Super T2V (64B Mixture-of-Transformers). Spec: docs/ports/cosmos3.md.
//!
//! Single GPU: the 31B generation tower stays resident (62 GB bf16) and the
//! text tower is run once per prompt into a K/V cache, streamed layer by
//! layer from the checkpoint unless `FASTVIDEO_COSMOS3_UND=resident` (B200).

pub mod pipeline;
pub mod transformer;

pub use pipeline::{Cosmos3Pipeline, Cosmos3Request};
pub use transformer::{Cosmos3Transformer, GenCache, UndCache};
