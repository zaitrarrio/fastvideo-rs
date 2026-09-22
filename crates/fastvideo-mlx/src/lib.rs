//! Apple Silicon MLX runtime scaffold for FastMetal-QAD / FastH3 MLX.
//!
//! # Requirements
//!
//! - **macOS 14+** on Apple Silicon (`aarch64-apple-darwin`).
//! - Unified memory: **16 GB+** for FastMetal 1.3B/5B; **36 GB+** for 14B / FastH3.
//! - Upstream FastVideo MLX packs (INT8 DiT + TAEHV), not CUDA cudarc graphs.
//!
//! This crate compiles on any host. Device execution is gated behind
//! `cfg(all(target_os = "macos", target_arch = "aarch64"))` and the optional
//! `mlx` feature (bindings land when `mlx-rs` / FastVideo MLX API is wired).
//!
//! Spec: docs/ports/fastmetal-mlx.md.

pub mod config;
pub mod generate;

pub use config::{FastH3MlxPreset, FastMetalPreset, MlxModelSpec};
pub use generate::{MlxGenerateRequest, MlxGenerateScaffold};
