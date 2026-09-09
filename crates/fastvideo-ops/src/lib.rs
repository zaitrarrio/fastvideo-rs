//! Tensor ops trait shared by Burn, Candle, and Luminal adapters.
//!
//! Model code in `fastvideo-models` is generic over [`TensorBackend`]. Pipeline
//! orchestration in `fastvideo-core` stays backend-agnostic.

pub mod backend;
pub mod device;
pub mod dtype;
pub mod error;
pub mod host;

pub use backend::TensorBackend;
pub use device::Device;
pub use dtype::DType;
pub use error::OpsError;
pub use host::{HostBackend, HostTensor};
