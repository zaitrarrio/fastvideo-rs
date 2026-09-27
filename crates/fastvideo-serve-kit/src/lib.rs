//! Shared axum glue for every fv-serve adapter (design §2.1, WP-05).
//!
//! Outbound HTTP (media fetch, callbacks, webhooks) is behind `fetch`.
//!
//! Owned by WP-05 (docs/serve/design.md §8). Scaffolded by WP-00.

pub mod ctx;
pub mod auth;
pub mod store;
pub mod artifacts;
pub mod uploads;
pub mod ingest;
pub mod callback;
pub mod sse;
pub mod handlers;
