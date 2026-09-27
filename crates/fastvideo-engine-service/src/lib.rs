//! Engine service: warm model pool, one executor thread per GPU, scheduler,
//! cancellation, progress events and the streaming cores (design §3.6, §5).
//!
//! `FakeBackend` is always built; `CudaBackend` needs the `cuda` feature.
//!
//! Owned by WP-02 (docs/serve/design.md §8). Scaffolded by WP-00.

pub mod service;
pub mod executor;
pub mod scheduler;
pub mod caps;
pub mod pool;
pub mod cancel;
pub mod fake;
pub mod stream;
#[cfg(feature = "cuda")]
pub mod cuda;
