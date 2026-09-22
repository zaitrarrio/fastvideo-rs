//! LingBot-Video. Spec: docs/ports/lingbot.md.

pub mod pipeline;
pub mod text;
pub mod transformer;

pub use pipeline::{LingBotPipeline, LingBotRequest};
pub use transformer::LingBotTransformer;