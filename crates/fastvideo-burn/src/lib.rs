//! Burn adapter.
//!
//! Burn 0.21 tensors are rank-generic (`Tensor<B, D>`). Until Flex mapping
//! lands, this backend executes the same f32 reference kernels as `HostBackend`
//! so UniPC / CFG / SDPA generate is testable rather than `NotImplemented`.

use fastvideo_ops::{Device, DType, HostBackend, HostTensor, OpsError, TensorBackend};

#[derive(Debug, Clone, Copy, Default)]
pub struct BurnBackend;

impl TensorBackend for BurnBackend {
    type Tensor = HostTensor;
    type Device = Device;

    fn name() -> &'static str {
        "burn"
    }

    fn map_device(device: &Device) -> Result<Self::Device, OpsError> {
        HostBackend::map_device(device)
    }

    fn zeros(shape: &[usize], dtype: DType, device: &Self::Device) -> Result<Self::Tensor, OpsError> {
        HostBackend::zeros(shape, dtype, device)
    }

    fn from_f32(data: &[f32], shape: &[usize], device: &Self::Device) -> Result<Self::Tensor, OpsError> {
        HostBackend::from_f32(data, shape, device)
    }

    fn to_f32(tensor: &Self::Tensor) -> Result<Vec<f32>, OpsError> {
        HostBackend::to_f32(tensor)
    }

    fn shape(tensor: &Self::Tensor) -> Vec<usize> {
        HostBackend::shape(tensor)
    }

    fn dtype(tensor: &Self::Tensor) -> DType {
        HostBackend::dtype(tensor)
    }

    fn add(a: &Self::Tensor, b: &Self::Tensor) -> Result<Self::Tensor, OpsError> {
        HostBackend::add(a, b)
    }

    fn mul(a: &Self::Tensor, b: &Self::Tensor) -> Result<Self::Tensor, OpsError> {
        HostBackend::mul(a, b)
    }

    fn mul_scalar(a: &Self::Tensor, scale: f32) -> Result<Self::Tensor, OpsError> {
        HostBackend::mul_scalar(a, scale)
    }

    fn matmul(a: &Self::Tensor, b: &Self::Tensor) -> Result<Self::Tensor, OpsError> {
        HostBackend::matmul(a, b)
    }

    fn silu(a: &Self::Tensor) -> Result<Self::Tensor, OpsError> {
        HostBackend::silu(a)
    }

    fn gelu(a: &Self::Tensor) -> Result<Self::Tensor, OpsError> {
        HostBackend::gelu(a)
    }

    fn softmax(a: &Self::Tensor, dim: usize) -> Result<Self::Tensor, OpsError> {
        HostBackend::softmax(a, dim)
    }

    fn rms_norm(a: &Self::Tensor, weight: &Self::Tensor, eps: f32) -> Result<Self::Tensor, OpsError> {
        HostBackend::rms_norm(a, weight, eps)
    }

    fn scaled_dot_product_attention(
        query: &Self::Tensor,
        key: &Self::Tensor,
        value: &Self::Tensor,
        scale: Option<f32>,
    ) -> Result<Self::Tensor, OpsError> {
        HostBackend::scaled_dot_product_attention(query, key, value, scale)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reports_burn_name() {
        assert_eq!(BurnBackend::name(), "burn");
    }

    #[test]
    fn matmul_matches_host() {
        let device = Device::cpu();
        let a = BurnBackend::from_f32(&[1.0, 2.0, 3.0, 4.0], &[2, 2], &device).unwrap();
        let i = BurnBackend::from_f32(&[1.0, 0.0, 0.0, 1.0], &[2, 2], &device).unwrap();
        let out = BurnBackend::matmul(&a, &i).unwrap();
        assert_eq!(BurnBackend::to_f32(&out).unwrap(), vec![1.0, 2.0, 3.0, 4.0]);
    }
}
