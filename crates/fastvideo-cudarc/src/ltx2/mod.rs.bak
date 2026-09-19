//! LTX-2 device graph: Gemma-3 text encoder + connectors, the dual-stream
//! audio+video DiT, video VAE, audio VAE and vocoder. Shares
//! tensor/ops/nn/attention with [`crate::wan`]. See docs/ports/ltx2.md.
//!
//! The port adds no CUDA kernel. Everything here is graph code over ops the
//! Wan backend already has, plus two host-side pieces that are input
//! preparation rather than compute: the rotary tables
//! (`fastvideo_models::ltx2::rope`) and the checkpoint key view ([`keys`]).
//! Device tensors are f32; bf16 lives only inside [`crate::wan::nn::Linear`].

pub mod attention;
pub mod keys;
pub mod text;

use crate::wan::tensor::{CudaTensor, Result, TensorError};

pub(crate) fn msg(s: impl Into<String>) -> TensorError {
    TensorError::Message(s.into())
}

/// A `[width]` weight of ones, kept on the device. Every block norm in LTX-2
/// is an RMSNorm *without* a learned weight; the backend's RMSNorm kernel
/// takes one, so the weightless norm is that kernel with this.
pub(crate) fn ones(width: usize) -> Result<CudaTensor> {
    let mut w = CudaTensor::ones(&[width]);
    w.pin_device()?;
    Ok(w)
}

/// A host vector as a device-resident tensor (tables, statistics).
pub(crate) fn pinned(data: Vec<f32>, shape: Vec<usize>) -> Result<CudaTensor> {
    let mut t = CudaTensor::from_vec(data, shape)?;
    t.pin_device()?;
    Ok(t)
}
