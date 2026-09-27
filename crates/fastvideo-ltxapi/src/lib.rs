//! LTX API `/v2/*` async jobs, `/v1/*` sync, `/v1/upload` (design §4.5, WP-08).
//!
//! Owned by WP-08 (docs/serve/design.md §8). Scaffolded by WP-00.

pub mod v2;
pub mod v1;
pub mod upload;
pub mod stubs;
pub mod error;
