use crate::device::Device;
use crate::dtype::DType;
use crate::error::OpsError;

/// Backend-agnostic tensor surface used by Wan DiT / VAE / UMT5.
///
/// Burn, Candle, and Luminal each implement this trait. Luminal should record
/// a single DiT step (and a VAE decode) as a compiled graph and execute that
/// graph from the Rust denoising loop — do not unroll UniPC/DMD into one graph.
pub trait TensorBackend: Sized + Send + Sync + 'static {
    type Tensor: Clone + Send + Sync;
    type Device: Clone + Send + Sync;

    fn name() -> &'static str;
    fn map_device(device: &Device) -> Result<Self::Device, OpsError>;

    fn zeros(
        shape: &[usize],
        dtype: DType,
        device: &Self::Device,
    ) -> Result<Self::Tensor, OpsError>;

    fn from_f32(
        data: &[f32],
        shape: &[usize],
        device: &Self::Device,
    ) -> Result<Self::Tensor, OpsError>;

    fn to_f32(tensor: &Self::Tensor) -> Result<Vec<f32>, OpsError>;
    fn shape(tensor: &Self::Tensor) -> Vec<usize>;
    fn dtype(tensor: &Self::Tensor) -> DType;

    fn add(a: &Self::Tensor, b: &Self::Tensor) -> Result<Self::Tensor, OpsError>;
    fn mul(a: &Self::Tensor, b: &Self::Tensor) -> Result<Self::Tensor, OpsError>;
    fn mul_scalar(a: &Self::Tensor, scale: f32) -> Result<Self::Tensor, OpsError>;
    fn matmul(a: &Self::Tensor, b: &Self::Tensor) -> Result<Self::Tensor, OpsError>;

    fn silu(a: &Self::Tensor) -> Result<Self::Tensor, OpsError> {
        let _ = a;
        Err(OpsError::not_implemented(Self::name(), "silu"))
    }

    fn gelu(a: &Self::Tensor) -> Result<Self::Tensor, OpsError> {
        let _ = a;
        Err(OpsError::not_implemented(Self::name(), "gelu"))
    }

    fn softmax(a: &Self::Tensor, dim: usize) -> Result<Self::Tensor, OpsError> {
        let _ = (a, dim);
        Err(OpsError::not_implemented(Self::name(), "softmax"))
    }

    fn rms_norm(
        a: &Self::Tensor,
        weight: &Self::Tensor,
        eps: f32,
    ) -> Result<Self::Tensor, OpsError> {
        let _ = (a, weight, eps);
        Err(OpsError::not_implemented(Self::name(), "rms_norm"))
    }

    fn layer_norm(
        a: &Self::Tensor,
        weight: &Self::Tensor,
        bias: Option<&Self::Tensor>,
        eps: f32,
    ) -> Result<Self::Tensor, OpsError> {
        let _ = (a, weight, bias, eps);
        Err(OpsError::not_implemented(Self::name(), "layer_norm"))
    }

    fn conv2d(
        input: &Self::Tensor,
        weight: &Self::Tensor,
        bias: Option<&Self::Tensor>,
        stride: [usize; 2],
        padding: [usize; 2],
    ) -> Result<Self::Tensor, OpsError> {
        let _ = (input, weight, bias, stride, padding);
        Err(OpsError::not_implemented(Self::name(), "conv2d"))
    }

    fn conv3d(
        input: &Self::Tensor,
        weight: &Self::Tensor,
        bias: Option<&Self::Tensor>,
        stride: [usize; 3],
        padding: [usize; 3],
    ) -> Result<Self::Tensor, OpsError> {
        let _ = (input, weight, bias, stride, padding);
        Err(OpsError::not_implemented(Self::name(), "conv3d"))
    }

    fn scaled_dot_product_attention(
        query: &Self::Tensor,
        key: &Self::Tensor,
        value: &Self::Tensor,
        scale: Option<f32>,
    ) -> Result<Self::Tensor, OpsError> {
        let _ = (query, key, value, scale);
        Err(OpsError::not_implemented(Self::name(), "sdpa"))
    }

    fn rope_nd(
        query: &Self::Tensor,
        key: &Self::Tensor,
        freqs: &Self::Tensor,
    ) -> Result<(Self::Tensor, Self::Tensor), OpsError> {
        let _ = (query, key, freqs);
        Err(OpsError::not_implemented(Self::name(), "rope_nd"))
    }

    fn random_normal(
        shape: &[usize],
        mean: f32,
        std: f32,
        seed: u64,
        device: &Self::Device,
    ) -> Result<Self::Tensor, OpsError> {
        let _ = (shape, mean, std, seed, device);
        Err(OpsError::not_implemented(Self::name(), "random_normal"))
    }
}
