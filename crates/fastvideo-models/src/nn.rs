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
    let dtype = xs.dtype();
    let x = xs.to_dtype(DType::F32)?;
    (x.clone() * candle_nn::ops::sigmoid(&x)?)?.to_dtype(dtype)
}

/// GELU tanh approximation used by Diffusers `gelu-approximate` / `gelu_pytorch_tanh`.
pub fn gelu_tanh(xs: &Tensor) -> Result<Tensor> {
    let dtype = xs.dtype();
    let xs = xs.to_dtype(DType::F32)?;
    let inner = ((xs.powf(3.0)? * 0.044715)? + &xs)?;
    let tanh = (inner * (2.0 / std::f64::consts::PI).sqrt())?.tanh()?;
    ((&xs * 0.5)? * (1.0 + tanh)?)?.to_dtype(dtype)
}

/// Exact GELU (`erf`) used by OpenCLIP ViT-H (`hidden_act: gelu`).
pub fn gelu(xs: &Tensor) -> Result<Tensor> {
    let dtype = xs.dtype();
    xs.to_dtype(DType::F32)?.gelu()?.to_dtype(dtype)
}

pub fn rms_norm(xs: &Tensor, weight: &Tensor, eps: f64) -> Result<Tensor> {
    let x = xs.to_dtype(DType::F32)?;
    let mean_sq = x.sqr()?.mean_keepdim(D::Minus1)?;
    let rms = (mean_sq + eps)?.sqrt()?;
    let y = x.broadcast_div(&rms)?;
    y.to_dtype(xs.dtype())?.broadcast_mul(&weight.to_dtype(xs.dtype())?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn silu_gelu_rms_match_numpy() {
        let device = Device::Cpu;
        let x = Tensor::from_vec(vec![-2.0f32, -0.5, 0.0, 0.25, 1.5], (5,), &device).unwrap();
        let silu_y = silu(&x).unwrap().to_vec1::<f32>().unwrap();
        let gelu_y = gelu_tanh(&x).unwrap().to_vec1::<f32>().unwrap();
        let w = Tensor::from_vec(vec![1.1f32, 0.9, 1.0, 0.8, 1.2], (5,), &device).unwrap();
        let rms_y = rms_norm(&x, &w, 1e-6).unwrap().to_vec1::<f32>().unwrap();
        let silu_exp = [-0.23840584, -0.18877033, 0.0, 0.14054413, 1.2263617];
        let gelu_exp = [-0.045402306, -0.15428599, 0.0, 0.14967535, 1.3995716];
        let rms_exp = [-1.9203167, -0.39279205, 0.0, 0.17457425, 1.5711682];
        for i in 0..5 {
            assert!((silu_y[i] - silu_exp[i]).abs() < 1e-6);
            assert!((gelu_y[i] - gelu_exp[i]).abs() < 1e-6);
            assert!((rms_y[i] - rms_exp[i]).abs() < 1e-5);
        }
    }

    #[test]
    fn sinusoidal_timestep_matches_diffusers() {
        let device = Device::Cpu;
        let t = Tensor::from_vec(vec![500f32], (1,), &device).unwrap();
        let emb = sinusoidal_timesteps(&t, 256, &device)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert!((emb[0] + 0.8838493).abs() < 1e-5, "cos0={}", emb[0]);
        assert!((emb[1] - 0.9459426).abs() < 1e-5, "emb1={}", emb[1]);
        assert!((emb[128] + 0.4677718).abs() < 1e-5, "emb128={}", emb[128]);
    }
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

pub fn scaled_dot_product_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: Option<&Tensor>,
) -> Result<Tensor> {
    // q/k/v: [B, heads, seq, dim]. CUDA BF16 cannot multiply an F32/F64 scale.
    let dtype = q.dtype();
    let q = q.to_dtype(DType::F32)?;
    let k = k.to_dtype(DType::F32)?;
    let v = v.to_dtype(DType::F32)?;
    let dim = q.dim(D::Minus1)? as f64;
    let scale = 1.0 / dim.sqrt();
    let mut attn = (q.matmul(&k.transpose(D::Minus1, D::Minus2)?)? * scale)?;
    if let Some(mask) = mask {
        attn = attn.broadcast_add(&mask.to_dtype(DType::F32)?)?;
    }
    let attn = candle_nn::ops::softmax_last_dim(&attn)?;
    attn.matmul(&v)?.to_dtype(dtype)
}
