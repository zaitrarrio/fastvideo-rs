//! FastVideo `/v1/videos*`, `/v1/models*`, `/v1/model_info`, and the FastWan
//! `/generate`, `/status`, `/video` routes (design §4.1-4.2, WP-06).
//!
//! Owned by WP-06 (docs/serve/design.md §8). Scaffolded by WP-00.

pub mod videos;
pub mod models;
pub mod fastwan;
pub mod error;
