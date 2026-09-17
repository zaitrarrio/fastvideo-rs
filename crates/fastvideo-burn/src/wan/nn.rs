//! Burn primitives matching Diffusers / Candle `fastvideo_models::nn` layout.

use burn::prelude::*;
use burn::tensor::activation::{silu as burn_silu, softmax};
use burn::tensor::module::conv2d;
use burn::tensor::ops::ConvOptions;

use crate::error::{BurnError, Result};

#[cfg(feature = "cuda")]
pub type B = burn::backend::Cuda<f32, i32>;
#[cfg(not(feature = "cuda"))]
pub type B = burn::backend::NdArray<f32>;

pub type Device = <B as Backend>::Device;

/// Resolve Burn device from CLI/`LoadOptions` spec (`cpu`, `cuda`, `cuda:0`).
pub fn resolve_device(spec: &str) -> Result<Device> {
    let s = spec.trim().to_ascii_lowercase();
    #[cfg(feature = "cuda")]
    {
        if s == "cpu" {
            return Err(BurnError::msg(
                "this binary was built with burn cuda; use --device cuda (ndarray CPU needs a non-cuda build)",
            ));
        }
        if s == "cuda" || s == "cuda:0" {
            return Ok(burn::backend::cuda::CudaDevice::new(0));
        }
        if let Some(rest) = s.strip_prefix("cuda:") {
            let idx: usize = rest.parse().map_err(|_| {
                BurnError::msg(format!("invalid cuda device `{spec}`"))
            })?;
            return Ok(burn::backend::cuda::CudaDevice::new(idx));
        }
        return Err(BurnError::msg(format!(
            "unknown device `{spec}` (expected cuda or cuda:N)"
        )));
    }
    #[cfg(not(feature = "cuda"))]
    {
        if s.starts_with("cuda") {
            return Err(BurnError::msg(
                "CUDA requested but fastvideo-burn was built without `--features cuda`",
            ));
        }
        let _ = s;
        Ok(Default::default())
    }
}

pub fn default_device() -> Device {
    resolve_device(if cfg!(feature = "cuda") { "cuda" } else { "cpu" })
        .unwrap_or_else(|_| Default::default())
}

/// Diffusers-style Linear: weight `[out, in]`, optional bias `[out]`.
#[derive(Debug, Clone)]
pub struct Linear {
    pub weight: Tensor<B, 2>,
    pub bias: Option<Tensor<B, 1>>,
}

impl Linear {
    pub fn zeros(in_dim: usize, out_dim: usize, device: &Device) -> Self {
        Self {
            weight: Tensor::zeros([out_dim, in_dim], device),
            bias: Some(Tensor::zeros([out_dim], device)),
        }
    }

    pub fn zeros_no_bias(in_dim: usize, out_dim: usize, device: &Device) -> Self {
        Self {
            weight: Tensor::zeros([out_dim, in_dim], device),
            bias: None,
        }
    }

    /// Diffusers Linear: `{prefix}.weight` `[out, in]`, optional `{prefix}.bias`.
    pub fn load(
        map: &super::weights::WeightMap,
        prefix: &str,
        in_dim: usize,
        out_dim: usize,
        has_bias: bool,
        device: &Device,
    ) -> Result<Self> {
        let w_key = super::weights::join_key(prefix, "weight");
        let weight = super::weights::tensor2_shaped(map, &w_key, [out_dim, in_dim], device)?;
        let bias = if has_bias {
            let b_key = super::weights::join_key(prefix, "bias");
            Some(super::weights::tensor1_shaped(map, &b_key, out_dim, device)?)
        } else {
            None
        };
        Ok(Self { weight, bias })
    }

    pub fn forward_2(&self, xs: Tensor<B, 2>) -> Tensor<B, 2> {
        // y = x @ W^T + b
        let mut out = xs.matmul(self.weight.clone().transpose());
        if let Some(bias) = &self.bias {
            out = out + bias.clone().unsqueeze::<2>();
        }
        out
    }

    pub fn forward_3(&self, xs: Tensor<B, 3>) -> Tensor<B, 3> {
        let [b, s, i] = xs.dims();
        let flat = xs.reshape([b * s, i]);
        let out = self.forward_2(flat);
        let o = out.dims()[1];
        out.reshape([b, s, o])
    }
}

pub fn silu<const D: usize>(xs: Tensor<B, D>) -> Tensor<B, D> {
    burn_silu(xs)
}

pub fn gelu_tanh<const D: usize>(xs: Tensor<B, D>) -> Tensor<B, D> {
    // 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
    let x3 = xs.clone().powf_scalar(3.0);
    let inner = xs.clone() + x3.mul_scalar(0.044715);
    let tanh = inner
        .mul_scalar((2.0f32 / std::f32::consts::PI).sqrt())
        .tanh();
    xs.mul_scalar(0.5) * (tanh.add_scalar(1.0))
}

pub fn rms_norm<const D: usize>(
    xs: Tensor<B, D>,
    weight: Tensor<B, 1>,
    eps: f32,
) -> Tensor<B, D> {
    let mean_sq = xs.clone().powf_scalar(2.0).mean_dim(D - 1);
    let rms = (mean_sq.add_scalar(eps)).sqrt();
    let y = xs / rms;
    let mut w_shape = [1; D];
    w_shape[D - 1] = weight.dims()[0];
    y * weight.reshape(w_shape)
}

pub fn layer_norm<const D: usize>(
    xs: Tensor<B, D>,
    eps: f32,
    weight: Option<&Tensor<B, 1>>,
    bias: Option<&Tensor<B, 1>>,
) -> Tensor<B, D> {
    let mean = xs.clone().mean_dim(D - 1);
    let centered = xs - mean;
    let var = centered.clone().powf_scalar(2.0).mean_dim(D - 1);
    let mut y = centered / (var.add_scalar(eps)).sqrt();
    if let Some(w) = weight {
        let mut w_shape = [1; D];
        w_shape[D - 1] = w.dims()[0];
        y = y * w.clone().reshape(w_shape);
    }
    if let Some(b) = bias {
        let mut b_shape = [1; D];
        b_shape[D - 1] = b.dims()[0];
        y = y + b.clone().reshape(b_shape);
    }
    y
}

/// q/k/v: `[B, heads, seq, dim]` — full SDPA (tiny graphs only).
pub fn sdpa(
    q: Tensor<B, 4>,
    k: Tensor<B, 4>,
    v: Tensor<B, 4>,
    mask: Option<Tensor<B, 4>>,
) -> Tensor<B, 4> {
    let dim = q.dims()[3] as f32;
    let scale = 1.0 / dim.sqrt();
    let mut attn = q.matmul(k.swap_dims(2, 3)).mul_scalar(scale);
    if let Some(m) = mask {
        attn = attn + m;
    }
    let attn = softmax(attn, 3);
    attn.matmul(v)
}

pub fn sinusoidal_timesteps(timesteps: Tensor<B, 1>, dim: usize, device: &Device) -> Tensor<B, 2> {
    let half = dim / 2;
    let n = timesteps.dims()[0];
    let exponent: Vec<f32> = (0..half)
        .map(|i| (-(10_000f32.ln()) * (i as f32) / half as f32).exp())
        .collect();
    let freqs = Tensor::<B, 1>::from_floats(exponent.as_slice(), device).reshape([1, half]);
    let args = timesteps.reshape([n, 1]) * freqs;
    Tensor::cat(vec![args.clone().cos(), args.sin()], 1)
}

pub fn conv2d_nhwc(
    xs: Tensor<B, 4>,
    kernel: Tensor<B, 4>,
    bias: Option<Tensor<B, 1>>,
    padding: usize,
    stride: usize,
) -> Tensor<B, 4> {
    let opts = ConvOptions::new([stride, stride], [padding, padding], [1, 1], 1);
    conv2d(xs, kernel, bias, opts)
}

/// Pad along an arbitrary axis of a rank-5 tensor (leading zeros on the left).
pub fn pad_dim5(xs: Tensor<B, 5>, dim: usize, left: usize, right: usize) -> Tensor<B, 5> {
    if left == 0 && right == 0 {
        return xs;
    }
    let od = xs.dims();
    let mut nd = od;
    nd[dim] += left + right;
    let device = xs.device();
    let padded = Tensor::<B, 5>::zeros(nd, &device);
    let mut dest = [
        0..nd[0],
        0..nd[1],
        0..nd[2],
        0..nd[3],
        0..nd[4],
    ];
    for i in 0..5 {
        if i == dim {
            dest[i] = left..(left + od[i]);
        } else {
            dest[i] = 0..od[i];
        }
    }
    padded.slice_assign(dest, xs)
}

pub fn to_vec_f32<const D: usize>(xs: Tensor<B, D>) -> Result<Vec<f32>> {
    xs.into_data()
        .to_vec::<f32>()
        .map_err(|e| BurnError::msg(format!("tensor to_vec failed: {e:?}")))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn silu_gelu_rms_match_numpy() {
        let device = Default::default();
        let x = Tensor::<B, 1>::from_floats([-2.0f32, -0.5, 0.0, 0.25, 1.5], &device);
        let silu_y = to_vec_f32(silu(x.clone())).unwrap();
        let gelu_y = to_vec_f32(gelu_tanh(x.clone())).unwrap();
        let w = Tensor::<B, 1>::from_floats([1.1f32, 0.9, 1.0, 0.8, 1.2], &device);
        let rms_y = to_vec_f32(rms_norm(x, w, 1e-6)).unwrap();
        let silu_exp = [-0.23840584, -0.18877033, 0.0, 0.14054413, 1.2263617];
        let gelu_exp = [-0.045402306, -0.15428599, 0.0, 0.14967535, 1.3995716];
        let rms_exp = [-1.9203167, -0.39279205, 0.0, 0.17457425, 1.5711682];
        for i in 0..5 {
            assert!((silu_y[i] - silu_exp[i]).abs() < 1e-5);
            assert!((gelu_y[i] - gelu_exp[i]).abs() < 1e-5);
            assert!((rms_y[i] - rms_exp[i]).abs() < 1e-4);
        }
    }
}
