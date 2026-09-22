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
//! [`mlx-rs`](https://crates.io/crates/mlx-rs) (v0.25+) provides real Metal
//! bindings but **only builds on Apple Silicon**. This workspace intentionally
//! does **not** depend on `mlx-rs` in `Cargo.toml` so `cargo check` works on
//! x86_64 / Linux CI. On an aarch64 Mac, enable with:
//!
//! ```text
//! cargo check -p fastvideo-mlx --features mlx
//! ```
//!
//! and add a target-specific dependency locally:
//!
//! ```toml
//! [target.'cfg(all(target_os = "macos", target_arch = "aarch64"))'.dependencies]
//! mlx-rs = { version = "0.25", optional = true, default-features = false, features = ["metal"] }
//! ```
//!
//! Spec: docs/ports/fastmetal-mlx.md.

pub mod config;
pub mod generate;
pub mod metal;

pub use config::{FastH3MlxPreset, FastMetalPreset, MlxModelSpec};
pub use generate::{MlxGenerateRequest, MlxGenerateScaffold};
pub use metal::{MetalGate, MlxArrayStub, MlxDeviceKind};
