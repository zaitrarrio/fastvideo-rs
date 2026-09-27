//! Reactor local runtime contract (design §5.7, WP-13).
//!
//! Owned by WP-13 (docs/serve/design.md §8). Scaffolded by WP-00.

pub mod wire;
pub mod session;
pub mod signalling;
pub mod gateway;
pub mod commands;
pub mod schema;
