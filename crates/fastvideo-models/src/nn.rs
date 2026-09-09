//! Candle primitives matching Diffusers layer names.

use candle_core::{DType, Device, Result, Tensor, D};
use candle_nn::VarBuilder;

#[derive(Debug, Clone)]
pub struct Linear {
    weight: Tensor,
    bias: Option<Tensor>,
}

impl Linear {
    pub fn load(in_dim: usize, out_dim: usize, vb: VarBuilder) -> Result<Self> {
        let weight = vb.get((out_dim, in_dim), "weight")?;
        let bias = vb.get(out_dim, "bias").ok();
        Ok(Self { weight, bias })
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let w = self.weight.to_dtype(xs.dtype())?.t()?;
        let mut out = match *xs.dims() {
            [b1, b2, _, _] => {
                let w = w.broadcast_left((b1, b2))?;
                xs.matmul(&w)?
            }
            [b, _, _] => {
                let w = w.broadcast_left(b)?;
                xs.matmul(&w)?
            }
            _ => xs.matmul(&w)?,
        };
        if let Some(bias) = &self.bias {
            out = out.broadcast_add(&bias.to_dtype(xs.dtype())?)?;
        }
        Ok(out)
    }
}

pub fn silu(xs: &Tensor) -> Result<Tensor> {
    xs * candle_nn::ops::sigmoid(xs)?
}

/// GELU tanh approximation used by Diffusers `gelu-approximate` / `gelu_pytorch_tanh`.
pub fn gelu_tanh(xs: &Tensor) -> Result<Tensor> {
    let inner = ((xs.powf(3.0)? * 0.044715)? + xs)?;
    let tanh = (inner * (2.0 / std::f64::consts::PI).sqrt())?.tanh()?;
    (xs * 0.5)? * (1.0 + tanh)?
}

pub fn rms_norm(xs: &Tensor, weight: &Tensor, eps: f64) -> Result<Tensor> {
    let x = xs.to_dtype(DType::F32)?;
    let mean_sq = x.sqr()?.mean_keepdim(D::Minus1)?;
    let rms = (mean_sq + eps)?.sqrt()?;
    let y = x.broadcast_div(&rms)?;
    y.to_dtype(xs.dtype())?.broadcast_mul(&weight.to_dtype(xs.dtype())?)
}

/// LayerNorm over the last dim. `affine=false` skips weight/bias.
pub fn layer_norm(xs: &Tensor, eps: f64, weight: Option<&Tensor>, bias: Option<&Tensor>) -> Result<Tensor> {
    let x = xs.to_dtype(DType::F32)?;
    let mean = x.mean_keepdim(D::Minus1)?;
    let centered = x.broadcast_sub(&mean)?;
    let var = centered.sqr()?.mean_keepdim(D::Minus1)?;
    let mut y = centered.broadcast_div(&(var + eps)?.sqrt()?)?;
    y = y.to_dtype(xs.dtype())?;
    if let Some(w) = weight {
        y = y.broadcast_mul(&w.to_dtype(xs.dtype())?)?;
    }
    if let Some(b) = bias {
        y = y.broadcast_add(&b.to_dtype(xs.dtype())?)?;
    }
    Ok(y)
}

pub fn sinusoidal_timesteps(timesteps: &Tensor, dim: usize, device: &Device) -> Result<Tensor> {
    // Diffusers Timesteps: flip_sin_to_cos=True, downscale_freq_shift=0.
    let half = dim / 2;
    let timesteps = timesteps.to_dtype(DType::F32)?.flatten_all()?;
    let n = timesteps.dims1()?;
    let exponent: Vec<f32> = (0..half)
        .map(|i| (-(10000f32.ln()) * (i as f32) / half as f32).exp())
        .collect();
    let freqs = Tensor::from_vec(exponent, (half,), device)?;
    let args = timesteps
        .reshape((n, 1))?
        .broadcast_mul(&freqs.reshape((1, half))?)?;
    let cos = args.cos()?;
    let sin = args.sin()?;
    Tensor::cat(&[&cos, &sin], 1)
}

pub fn conv2d(xs: &Tensor, kernel: &Tensor, padding: usize, stride: usize) -> Result<Tensor> {
    xs.conv2d(kernel, padding, stride, 1, 1)
}

pub fn scaled_dot_product_attention(q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
    // q/k/v: [B, heads, seq, dim]
    let dim = q.dim(D::Minus1)? as f64;
    let scale = 1.0 / dim.sqrt();
    let attn = (q.matmul(&k.transpose(D::Minus1, D::Minus2)?)? * scale)?;
    let attn = candle_nn::ops::softmax_last_dim(&attn)?;
    attn.matmul(v)
}
