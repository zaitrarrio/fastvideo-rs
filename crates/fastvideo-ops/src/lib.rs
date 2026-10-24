//! Shared numeric helpers: the FP8 E4M3 conversions (`fp8`) the runtime
//! (`fastvideo-cudarc`) and the model crate (`fastvideo-models`) both check
//! against.
//!
//! The backend-agnostic `TensorBackend` trait and its host reference
//! (`HostBackend`, `HostTensor`) were retired in plan B6
//! (`docs/adr/0017-loader-stages.md`): no model ran through them; every
//! pipeline reaches the device through `fastvideo-cudarc`'s runtime.

pub mod fp8;
