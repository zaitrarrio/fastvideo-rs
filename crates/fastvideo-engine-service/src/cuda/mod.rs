//! `CudaBackend` over the fastvideo-cudarc pipelines (WP-11). Built only with `cuda`.
//!
//! Scaffold stub (WP-00): the owning work package fills this in.

// Links the CUDA backend crate so `--features cuda` type-checks against it.
use fastvideo_cudarc as _;

pub mod h3;
pub mod ltx2;
pub mod wan;
pub mod caps;
pub mod causal;
