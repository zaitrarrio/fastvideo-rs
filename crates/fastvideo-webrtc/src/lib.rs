//! WebRTC transport for fv-serve (design §5.8, WP-04).
//!
//! The str0m host is behind `str0m`; the WHIP publisher is behind `whip`.
//!
//! Owned by WP-04 (docs/serve/design.md §8). Scaffolded by WP-00.

pub mod host;
pub mod sdp;
pub mod channel;
pub mod writer;
pub mod whip;
