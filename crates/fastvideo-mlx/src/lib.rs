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
//! `mlx` feature.
//!
//! # Metal / mlx-rs gate
//!
//! [`mlx-rs`](https://crates.io/crates/mlx-rs) **0.25+** is declared as a
//! **target-specific optional** dependency (Apple Silicon only). On this
//! workspace's x86_64/Linux CI the dep is skipped. On an aarch64 Mac:
//!
//! ```text
//! cargo check -p fastvideo-mlx --features mlx
//! ```
//!
//! Linked APIs (from mlx-rs docs, not invented): `Array::from_slice`,
//! `ops::zeros`, `Device::gpu` / `Device::set_default`. DiT/TAEHV graphs are
//! still scaffold-only after the gate opens.
//!
//! Spec: docs/ports/fastmetal-mlx.md.

pub mod config;
pub mod generate;
pub mod metal;

pub use config::{FastH3MlxPreset, FastMetalPreset, MlxModelSpec};
pub use generate::{MlxGenerateRequest, MlxGenerateScaffold};
pub use metal::{MetalGate, MlxArray, MlxArrayStub, MlxDeviceKind};
