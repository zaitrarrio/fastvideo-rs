//! Burn adapter.
//!
//! Burn 0.21 tensors are rank-generic (`Tensor<B, D>`), so this crate keeps a
//! dynamic-rank placeholder until Phase 1 wraps Flex tensors. Enable a real
//! Flex mapping there rather than depending on deprecated `burn-candle`.

use fastvideo_ops::{Device, DType, OpsError, TensorBackend};

#[derive(Debug, Clone)]
pub struct BurnTensor {
    pub shape: Vec<usize>,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct BurnBackend;

impl TensorBackend for BurnBackend {
    type Tensor = BurnTensor;
    type Device = Device;

    fn name() -> &'static str {
        "burn"
    }

    fn map_device(device: &Device) -> Result<Self::Device, OpsError> {
        Ok(device.clone())
    }

    fn zeros(
        shape: &[usize],
        _dtype: DType,
        _device: &Self::Device,
    ) -> Result<Self::Tensor, OpsError> {
        let _ = shape;
        Err(OpsError::not_implemented("burn", "zeros"))
    }

    fn from_f32(
        _data: &[f32],
        _shape: &[usize],
        _device: &Self::Device,
    ) -> Result<Self::Tensor, OpsError> {
        Err(OpsError::not_implemented("burn", "from_f32"))
    }

    fn to_f32(_tensor: &Self::Tensor) -> Result<Vec<f32>, OpsError> {
        Err(OpsError::not_implemented("burn", "to_f32"))
    }

    fn shape(tensor: &Self::Tensor) -> Vec<usize> {
        tensor.shape.clone()
    }

    fn dtype(_tensor: &Self::Tensor) -> DType {
        DType::F32
    }

    fn add(_a: &Self::Tensor, _b: &Self::Tensor) -> Result<Self::Tensor, OpsError> {
        Err(OpsError::not_implemented("burn", "add"))
    }

    fn mul(_a: &Self::Tensor, _b: &Self::Tensor) -> Result<Self::Tensor, OpsError> {
        Err(OpsError::not_implemented("burn", "mul"))
    }

    fn mul_scalar(_a: &Self::Tensor, _scale: f32) -> Result<Self::Tensor, OpsError> {
        Err(OpsError::not_implemented("burn", "mul_scalar"))
    }

    fn matmul(_a: &Self::Tensor, _b: &Self::Tensor) -> Result<Self::Tensor, OpsError> {
        Err(OpsError::not_implemented("burn", "matmul"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reports_burn_name() {
        assert_eq!(BurnBackend::name(), "burn");
    }
}
