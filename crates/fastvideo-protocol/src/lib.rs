//! Normalized protocol model shared by every fv-serve adapter (design §3).
//!
//! No tokio runtime and no axum: only `tokio::sync` for `JobStore::watch`.
//!
//! Owned by WP-01 (docs/serve/design.md §8). Scaffolded by WP-00.

pub mod request;
pub mod caps;
pub mod negotiate;
pub mod error;
pub mod job;
pub mod http;
pub mod stream;
pub mod av;
