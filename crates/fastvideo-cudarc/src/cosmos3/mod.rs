//! Cosmos3-Super T2V scaffold. Spec: docs/ports/cosmos3.md.
//!
//! Multi-GPU sequence parallel is out of scope. Single-GPU NVFP4/FP8 is the
//! intended 96 GB path (`FASTVIDEO_NVFP4=1`, `FASTVIDEO_NVFP4_COSMOS_STEPS`).

pub mod transformer;

pub use transformer::Cosmos3Transformer;
