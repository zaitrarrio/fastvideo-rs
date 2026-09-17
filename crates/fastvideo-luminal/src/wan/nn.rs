//! Layer primitives matching Candle `fastvideo_models::nn` for the host Wan graph.

use super::tensor::{NdTensor, Result, TensorError};

#[derive(Debug, Clone)]
pub struct Linear {
    pub weight: NdTensor, // [out, in]
    pub bias: Option<NdTensor>,
}

impl Linear {
    pub fn zeros(in_dim: usize, out_dim: usize, bias: bool) -> Self {
        Self {
            weight: NdTensor::zeros(&[out_dim, in_dim]),
            bias: if bias {
                Some(NdTensor::zeros(&[out_dim]))
            } else {
                None
            },
        }
    }

    pub fn load(
        map: &super::weights::WeightMap,
        prefix: &str,
        in_dim: usize,
        out_dim: usize,
        has_bias: bool,
    ) -> Result<Self> {
        let w_key = super::weights::join_key(prefix, "weight");
        let weight = super::weights::nd_tensor_shaped(map, &w_key, &[out_dim, in_dim])?;
        let bias = if has_bias {
            let b_key = super::weights::join_key(prefix, "bias");
            Some(super::weights::nd_tensor_shaped(map, &b_key, &[out_dim])?)
        } else {
            None
        };
        Ok(Self { weight, bias })
    }

    pub fn forward(&self, xs: &NdTensor) -> Result<NdTensor> {
        // weight is [out, in]; matmul xs[..., in] @ weight^T
        let w_t = self.weight.transpose(0, 1)?;
        let mut out = match xs.rank() {
            2 => xs.matmul(&w_t)?,
            3 => {
                let (b, s, i) = (xs.shape[0], xs.shape[1], xs.shape[2]);
                let flat = xs.reshape(vec![b * s, i])?;
                flat.matmul(&w_t)?.reshape(vec![b, s, self.weight.shape[0]])?
            }
            4 => {
                let (b1, b2, s, i) = (xs.shape[0], xs.shape[1], xs.shape[2], xs.shape[3]);
                let flat = xs.reshape(vec![b1 * b2 * s, i])?;
                flat.matmul(&w_t)?
                    .reshape(vec![b1, b2, s, self.weight.shape[0]])?
            }
            _ => {
                return Err(TensorError::Message(format!(
                    "linear unsupported rank {}",
                    xs.rank()
                )));
            }
        };
        if let Some(bias) = &self.bias {
            let mut bshape = vec![1; out.rank()];
            bshape[out.rank() - 1] = bias.shape[0];
            let b = bias.reshape(bshape)?;
            out = out.add(&b)?;
        }
        Ok(out)
    }
}

pub fn silu(xs: &NdTensor) -> NdTensor {
    xs.silu()
}

pub fn gelu_tanh(xs: &NdTensor) -> NdTensor {
    xs.gelu_tanh()
}

pub fn rms_norm(xs: &NdTensor, weight: &NdTensor, eps: f32) -> Result<NdTensor> {
    xs.rms_norm(weight, eps)
}

pub fn layer_norm(
    xs: &NdTensor,
    eps: f32,
    weight: Option<&NdTensor>,
    bias: Option<&NdTensor>,
) -> Result<NdTensor> {
    xs.layer_norm(eps, weight, bias)
}

pub fn softmax(xs: &NdTensor, dim: isize) -> Result<NdTensor> {
    xs.softmax(dim)
}

pub fn sinusoidal_timesteps(timesteps: &NdTensor, dim: usize) -> Result<NdTensor> {
    let half = dim / 2;
    let n = timesteps.data.len();
    let mut out = vec![0.0f32; n * dim];
    for (ti, &t) in timesteps.data.iter().enumerate() {
        for i in 0..half {
            let freq = (-(10000f32.ln()) * (i as f32) / half as f32).exp();
            let arg = t * freq;
            out[ti * dim + i] = arg.cos();
            out[ti * dim + half + i] = arg.sin();
        }
    }
    NdTensor::from_vec(out, vec![n, dim])
}

/// Scaled dot-product attention. q/k/v: [B, H, S, D]
pub fn scaled_dot_product_attention(
    q: &NdTensor,
    k: &NdTensor,
    v: &NdTensor,
    scale: Option<f32>,
) -> Result<NdTensor> {
    if q.rank() != 4 || k.rank() != 4 || v.rank() != 4 {
        return Err(TensorError::Message("sdpa expects BHSD".into()));
    }
    let d = q.shape[3] as f32;
    let scale = scale.unwrap_or(1.0 / d.sqrt());
    let k_t = k.transpose(2, 3)?;
    let mut scores = q.matmul(&k_t)?;
    scores = scores.mul_scalar(scale);
    let attn = scores.softmax(-1)?;
    attn.matmul(v)
}

pub fn conv2d(
    xs: &NdTensor,
    kernel: &NdTensor,
    padding: usize,
    stride: usize,
) -> Result<NdTensor> {
    xs.conv2d(kernel, None, padding, stride)
}
