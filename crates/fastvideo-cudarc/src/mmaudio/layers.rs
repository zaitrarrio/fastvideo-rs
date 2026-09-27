//! Small building blocks shared by the MMAudio modules (all device ops).

use crate::wan::nn::Linear;
use crate::wan::tensor::{CudaTensor, Result, TensorError};
use crate::wan::weights::{cuda_tensor_shaped, WeightMap};

pub(crate) fn msg(s: impl Into<String>) -> TensorError {
    TensorError::Message(s.into())
}

/// A weight kept on the device only.
pub(crate) fn pinned(data: Vec<f32>, shape: Vec<usize>) -> Result<CudaTensor> {
    let mut t = CudaTensor::from_vec(data, shape)?;
    t.pin_device()?;
    Ok(t)
}

pub(crate) fn host_values(map: &WeightMap, key: &str, shape: &[usize]) -> Result<Vec<f32>> {
    Ok(cuda_tensor_shaped(map, key, shape)?.host_cow()?.into_owned())
}

pub(crate) fn weight(map: &WeightMap, key: &str, shape: &[usize]) -> Result<CudaTensor> {
    pinned(host_values(map, key, shape)?, shape.to_vec())
}

pub(crate) fn linear(
    map: &WeightMap,
    prefix: &str,
    in_dim: usize,
    out_dim: usize,
    bias: bool,
) -> Result<Linear> {
    Linear::load(map, prefix, in_dim, out_dim, bias)
}

/// `Linear` from explicit host values (`w`: `[out, in]`).
pub(crate) fn linear_from(w: Vec<f32>, b: Option<Vec<f32>>, out_dim: usize, in_dim: usize) -> Result<Linear> {
    let wt = CudaTensor::from_vec(w, vec![out_dim, in_dim])?;
    let bt = match b {
        Some(b) => Some(CudaTensor::from_vec(b, vec![out_dim])?),
        None => None,
    };
    Linear::from_tensors(wt, bt)
}

/// `nn.Conv1d` (or `ChannelLastConv1d`) weight `[out, in, k]`, padding `k / 2`
/// unless given.
pub(crate) struct Conv1d {
    pub weight: CudaTensor,
    pub bias: Option<CudaTensor>,
    pub pad: usize,
}

impl Conv1d {
    pub fn load(map: &WeightMap, prefix: &str, cin: usize, cout: usize, k: usize, bias: bool) -> Result<Self> {
        Ok(Self {
            weight: weight(map, &format!("{prefix}.weight"), &[cout, cin, k])?,
            bias: if bias {
                Some(weight(map, &format!("{prefix}.bias"), &[cout])?)
            } else {
                None
            },
            pad: k / 2,
        })
    }

    pub fn from_host(w: Vec<f32>, b: Option<Vec<f32>>, cout: usize, cin: usize, k: usize) -> Result<Self> {
        Ok(Self {
            weight: pinned(w, vec![cout, cin, k])?,
            bias: match b {
                Some(b) => Some(pinned(b, vec![cout])?),
                None => None,
            },
            pad: k / 2,
        })
    }

    /// `[B, C, L]` to `[B, C_out, L]`.
    pub fn forward(&self, x: &CudaTensor) -> Result<CudaTensor> {
        x.conv1d(&self.weight, self.bias.as_ref(), self.pad, 1, 1, 1)
    }

    /// `ChannelLastConv1d`: `[B, N, C]` to `[B, N, C_out]`.
    pub fn forward_cl(&self, x: &CudaTensor) -> Result<CudaTensor> {
        self.forward(&x.transpose(1, 2)?)?.transpose(1, 2)
    }
}

/// `silu(x) * gate` for two same-shape tensors.
pub(crate) fn swiglu(a: &CudaTensor, b: &CudaTensor) -> Result<CudaTensor> {
    a.silu().mul(b)
}

/// `MLP` (SwiGLU of three bias-free linears).
pub(crate) struct Mlp {
    w1: Linear,
    w2: Linear,
    w3: Linear,
}

impl Mlp {
    pub fn load(map: &WeightMap, prefix: &str, dim: usize, hidden: usize) -> Result<Self> {
        Ok(Self {
            w1: linear(map, &format!("{prefix}.w1"), dim, hidden, false)?,
            w2: linear(map, &format!("{prefix}.w2"), hidden, dim, false)?,
            w3: linear(map, &format!("{prefix}.w3"), dim, hidden, false)?,
        })
    }

    pub fn forward(&self, x: &CudaTensor) -> Result<CudaTensor> {
        self.w2.forward(&swiglu(&self.w1.forward(x)?, &self.w3.forward(x)?)?)
    }
}

/// `ConvMLP` (the same with channel-last convolutions).
pub(crate) struct ConvMlp {
    w1: Conv1d,
    w2: Conv1d,
    w3: Conv1d,
}

impl ConvMlp {
    pub fn load(map: &WeightMap, prefix: &str, dim: usize, hidden: usize, k: usize) -> Result<Self> {
        Ok(Self {
            w1: Conv1d::load(map, &format!("{prefix}.w1"), dim, hidden, k, false)?,
            w2: Conv1d::load(map, &format!("{prefix}.w2"), hidden, dim, k, false)?,
            w3: Conv1d::load(map, &format!("{prefix}.w3"), dim, hidden, k, false)?,
        })
    }

    /// Channel-last `[B, N, D]`.
    pub fn forward_cl(&self, x: &CudaTensor) -> Result<CudaTensor> {
        let xt = x.transpose(1, 2)?;
        let h = swiglu(&self.w1.forward(&xt)?, &self.w3.forward(&xt)?)?;
        self.w2.forward(&h)?.transpose(1, 2)
    }
}

/// `x * (1 + scale) + shift`.
pub(crate) fn modulate(x: &CudaTensor, shift: &CudaTensor, scale: &CudaTensor) -> Result<CudaTensor> {
    x.mul(&scale.try_add_scalar(1.0)?)?.add(shift)
}

/// Split the last dim into `n` equal chunks.
pub(crate) fn chunk_last(x: &CudaTensor, n: usize) -> Result<Vec<CudaTensor>> {
    let d = *x.shape.last().ok_or_else(|| msg("chunk of a scalar"))?;
    if d % n != 0 {
        return Err(msg(format!("chunk {n} of {:?}", x.shape)));
    }
    let axis = x.rank() - 1;
    (0..n).map(|i| x.narrow(axis, i * d / n, d / n)).collect()
}

/// `x * sigmoid(1.702 x)` (open_clip QuickGELU) as `silu(1.702 x) / 1.702`.
pub(crate) fn quick_gelu(x: &CudaTensor) -> Result<CudaTensor> {
    x.try_mul_scalar(1.702)?.silu().try_mul_scalar(1.0 / 1.702)
}

/// Mean over axis 1 of `[B, N, D]` as a GEMM with a `1/N` row: `[B, 1, D]`.
pub(crate) fn mean_seq(x: &CudaTensor) -> Result<CudaTensor> {
    let [b, n, d] = x.shape[..] else {
        return Err(msg(format!("mean_seq expects [B, N, D], got {:?}", x.shape)));
    };
    let ones = CudaTensor::from_vec(vec![1.0 / n as f32; n], vec![1, 1, n])?;
    let mut rows = Vec::with_capacity(b);
    for i in 0..b {
        let xi = x.narrow(0, i, 1)?;
        rows.push(ones.matmul(&xi)?);
    }
    let refs: Vec<&CudaTensor> = rows.iter().collect();
    CudaTensor::cat(&refs, 0)?.reshape(vec![b, 1, d])
}

/// Rows of `x` (`[R, D]` view) gathered by `idx`.
pub(crate) fn gather_rows(x: &CudaTensor, idx: &[usize]) -> Result<CudaTensor> {
    let d = *x.shape.last().ok_or_else(|| msg("gather of a scalar"))?;
    x.reshape(vec![x.numel() / d, d])?.index_select_rows(idx)
}

/// Half-split RoPE tables `[S, dim]` (cos, sin) for the interleaved angles
/// `angles[s * dim/2 + i]` after the head channels were reordered to
/// `(0, 2, 4, …, 1, 3, 5, …)`.
pub(crate) fn rope_tables(angles: &[f32], len: usize, dim: usize) -> Result<(CudaTensor, CudaTensor)> {
    let half = dim / 2;
    let mut cos = vec![0.0f32; len * dim];
    let mut sin = vec![0.0f32; len * dim];
    for s in 0..len {
        for i in 0..dim {
            let a = angles[s * half + i % half];
            cos[s * dim + i] = a.cos();
            sin[s * dim + i] = a.sin();
        }
    }
    Ok((
        CudaTensor::from_vec(cos, vec![len, dim])?,
        CudaTensor::from_vec(sin, vec![len, dim])?,
    ))
}

/// The head-channel order that turns interleaved-pair RoPE into half-split:
/// new channel `p` reads old channel `perm[p]`.
pub(crate) fn interleave_to_half(dim: usize) -> Vec<usize> {
    let half = dim / 2;
    (0..dim)
        .map(|p| if p < half { 2 * p } else { 2 * (p - half) + 1 })
        .collect()
}
