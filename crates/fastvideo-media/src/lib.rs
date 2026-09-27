//! Media plumbing for fv-serve (design §5.1, §5.9, WP-03).
//!
//! The in-process encoders sit behind the `openh264` and `opus` features.
//!
//! Owned by WP-03 (docs/serve/design.md §8). Scaffolded by WP-00.

pub mod pacer;
pub mod resample;
pub mod opus;
pub mod video;
pub mod mp4;
pub mod probe;
pub mod crossfade;
pub mod sink;
