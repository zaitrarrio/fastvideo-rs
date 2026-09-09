//! Luminal adapter.
//!
//! Luminal is a static graph compiler. The denoising loop stays in Rust:
//! compile one DiT step graph and one VAE decode graph, then execute them
//! per timestep. The crates.io `luminal` 0.2 API is stale relative to
//! github.com/luminal-ai/luminal; Phase 6 will pin a git revision.

use fastvideo_ops::{Device, DType, OpsError, TensorBackend};

#[derive(Debug, Clone)]
pub struct LuminalTensor {
    pub shape: Vec<usize>,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct LuminalBackend;

impl TensorBackend for LuminalBackend {
    type Tensor = LuminalTensor;
    type Device = Device;

    fn name() -> &'static str {
        "luminal"
    }

    fn map_device(device: &Device) -> Result<Self::Device, OpsError> {
        Ok(device.clone())
    }

    fn zeros(
        _shape: &[usize],
        _dtype: DType,
        _device: &Self::Device,
    ) -> Result<Self::Tensor, OpsError> {
        Err(OpsError::not_implemented("luminal", "zeros"))
    }

    fn from_f32(
        _data: &[f32],
        _shape: &[usize],
        _device: &Self::Device,
    ) -> Result<Self::Tensor, OpsError> {
        Err(OpsError::not_implemented("luminal", "from_f32"))
    }

    fn to_f32(_tensor: &Self::Tensor) -> Result<Vec<f32>, OpsError> {
        Err(OpsError::not_implemented("luminal", "to_f32"))
    }

    fn shape(tensor: &Self::Tensor) -> Vec<usize> {
        tensor.shape.clone()
    }

    fn dtype(_tensor: &Self::Tensor) -> DType {
        DType::F32
    }

    fn add(_a: &Self::Tensor, _b: &Self::Tensor) -> Result<Self::Tensor, OpsError> {
        Err(OpsError::not_implemented("luminal", "add"))
    }

    fn mul(_a: &Self::Tensor, _b: &Self::Tensor) -> Result<Self::Tensor, OpsError> {
        Err(OpsError::not_implemented("luminal", "mul"))
    }

    fn mul_scalar(_a: &Self::Tensor, _scale: f32) -> Result<Self::Tensor, OpsError> {
        Err(OpsError::not_implemented("luminal", "mul_scalar"))
    }

    fn matmul(_a: &Self::Tensor, _b: &Self::Tensor) -> Result<Self::Tensor, OpsError> {
        Err(OpsError::not_implemented("luminal", "matmul"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reports_luminal_name() {
        assert_eq!(LuminalBackend::name(), "luminal");
    }
}
