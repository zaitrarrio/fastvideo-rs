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

/// OpenAI CLIP `quick_gelu`: `x * sigmoid(1.702 x)`.
pub fn quick_gelu(xs: &Tensor) -> Result<Tensor> {
    let dtype = xs.dtype();
    let x = xs.to_dtype(DType::F32)?;
    let s = candle_nn::ops::sigmoid(&(&x * 1.702)?)?;
    (x * s)?.to_dtype(dtype)
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

    #[test]
    fn sdpa_query_chunks_match_full() {
        let device = Device::Cpu;
        let q = Tensor::arange(0f32, (1 * 2 * 80 * 8) as f32, &device)
            .unwrap()
            .reshape((1, 2, 80, 8))
            .unwrap();
        let k = (&q * 0.01).unwrap();
        let v = (&q * 0.02).unwrap();
        let chunked = scaled_dot_product_attention(&q, &k, &v, None).unwrap();
        let full = super::sdpa_qk(&q, &k, &v, None, 1.0 / 8f64.sqrt()).unwrap();
        let a = chunked.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let b = full.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(a.len(), b.len());
        for (x, y) in a.iter().zip(b.iter()) {
            assert!((x - y).abs() < 1e-5, "{x} vs {y}");
        }
    }

    #[test]
    fn conv2d_tiles_match_full() {
        let device = Device::Cpu;
        let xs = Tensor::arange(0f32, (1 * 3 * 80 * 96) as f32, &device)
            .unwrap()
            .reshape((1, 3, 80, 96))
            .unwrap();
        let k = Tensor::arange(0f32, (4 * 3 * 3 * 3) as f32, &device)
            .unwrap()
            .reshape((4, 3, 3, 3))
            .unwrap()
            .affine(0.01, 0.0)
            .unwrap();
        for (pad, stride) in [(0usize, 1usize), (1, 1), (1, 2)] {
            let full = xs.conv2d(&k, pad, stride, 1, 1).unwrap();
            let tiled = super::conv2d_tiled(&xs, &k, pad, stride).unwrap();
            assert_eq!(full.dims(), tiled.dims(), "pad={pad} stride={stride}");
            let a = full.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            let b = tiled.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
                assert!((x - y).abs() < 1e-4, "i={i} {x} vs {y} pad={pad} stride={stride}");
            }
        }
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

/// Height/width of each output tile. Keeps CUDA im2col workspaces off the
/// 24GB card at 480p (full-frame 3x3 im2col is multiple GB).
const CONV2D_TILE: usize = 32;

pub fn conv2d(xs: &Tensor, kernel: &Tensor, padding: usize, stride: usize) -> Result<Tensor> {
    let (_, _, h, w) = xs.dims4()?;
    if h.max(w) <= CONV2D_TILE * 2 {
        return xs.conv2d(kernel, padding, stride, 1, 1);
    }
    conv2d_tiled(xs, kernel, padding, stride)
}

fn conv2d_tiled(xs: &Tensor, kernel: &Tensor, padding: usize, stride: usize) -> Result<Tensor> {
    let stride = stride.max(1);
    let (_oc, _ic, kh, kw) = kernel.dims4()?;
    let mut x = xs.clone();
    if padding > 0 {
        x = x.pad_with_zeros(2, padding, padding)?;
        x = x.pad_with_zeros(3, padding, padding)?;
    }
    let (_b, _c, h, w) = x.dims4()?;
    if h < kh || w < kw {
        candle_core::bail!("conv2d tile: input {h}x{w} smaller than kernel {kh}x{kw}");
    }
    let oh = (h - kh) / stride + 1;
    let ow = (w - kw) / stride + 1;
    let dtype = xs.dtype();
    let kernel = kernel.to_dtype(dtype)?;
    let mut rows = Vec::new();
    let mut oy = 0usize;
    while oy < oh {
        let th = CONV2D_TILE.min(oh - oy);
        let mut cols = Vec::new();
        let mut ox = 0usize;
        while ox < ow {
            let tw = CONV2D_TILE.min(ow - ox);
            let in_y = oy * stride;
            let in_x = ox * stride;
            let in_h = (th - 1) * stride + kh;
            let in_w = (tw - 1) * stride + kw;
            let tile = x
                .narrow(2, in_y, in_h)?
                .narrow(3, in_x, in_w)?
                .contiguous()?;
            let y = if dtype == DType::F32 {
                tile.conv2d(&kernel, 0, stride, 1, 1)?
            } else {
                tile.to_dtype(DType::F32)?
                    .conv2d(&kernel.to_dtype(DType::F32)?, 0, stride, 1, 1)?
                    .to_dtype(dtype)?
            };
            cols.push(y);
            ox += tw;
        }
        rows.push(if cols.len() == 1 {
            cols.pop().unwrap()
        } else {
            Tensor::cat(&cols, 3)?
        });
        oy += th;
    }
    let out = if rows.len() == 1 {
        rows.pop().unwrap()
    } else {
        Tensor::cat(&rows, 2)?
    };
    Ok(out)
}

/// Query-chunked SDPA so we never allocate a full `[seq, seq]` score matrix.
/// Chunking is exact (softmax is over keys, independent per query).
const SDPA_QUERY_CHUNK: usize = 64;

pub fn scaled_dot_product_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: Option<&Tensor>,
) -> Result<Tensor> {
    // q/k/v: [B, heads, seq, dim]
    let dtype = q.dtype();
    let (_b, _h, sq, dim) = q.dims4()?;
    let compute = match q.device() {
        Device::Cpu => DType::F32,
        _ => dtype,
    };
    let q = q.to_dtype(compute)?.contiguous()?;
    let k = k.to_dtype(compute)?.contiguous()?;
    let v = v.to_dtype(compute)?.contiguous()?;
    let scale = 1.0 / (dim as f64).sqrt();
    let out = if sq <= SDPA_QUERY_CHUNK {
        sdpa_qk(&q, &k, &v, mask, scale)?
    } else {
        let mut chunks = Vec::new();
        let mut start = 0;
        while start < sq {
            let len = SDPA_QUERY_CHUNK.min(sq - start);
            let qc = q.narrow(2, start, len)?;
            let mask_c = match mask {
                Some(m) => Some(m.narrow(m.dims().len() - 2, start, len)?),
                None => None,
            };
            chunks.push(sdpa_qk(&qc, &k, &v, mask_c.as_ref(), scale)?);
            start += len;
        }
        Tensor::cat(&chunks, 2)?
    };
    out.to_dtype(dtype)
}

fn sdpa_qk(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: Option<&Tensor>,
    scale: f64,
) -> Result<Tensor> {
    // QK stays in `q`'s dtype (BF16 on CUDA). Softmax is F32 on the query chunk only.
    let mut attn = q.matmul(&k.transpose(D::Minus1, D::Minus2)?)?;
    attn = attn.to_dtype(DType::F32)?.affine(scale, 0.0)?;
    if let Some(mask) = mask {
        attn = attn.broadcast_add(&mask.to_dtype(DType::F32)?)?;
    }
    let attn = candle_nn::ops::softmax_last_dim(&attn)?;
    attn.to_dtype(v.dtype())?.matmul(v)
}
