use crate::backend::TensorBackend;
use crate::device::Device;
use crate::dtype::DType;
use crate::error::OpsError;

/// Reference CPU backend used by unit tests and scheduler bring-up.
#[derive(Debug, Clone)]
pub struct HostTensor {
    pub data: Vec<f32>,
    pub shape: Vec<usize>,
}

fn numel(shape: &[usize]) -> usize {
    shape.iter().product()
}

fn same_shape(a: &[usize], b: &[usize]) -> Result<(), OpsError> {
    if a == b {
        Ok(())
    } else {
        Err(OpsError::Shape(format!("{a:?} vs {b:?}")))
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct HostBackend;

impl TensorBackend for HostBackend {
    type Tensor = HostTensor;
    type Device = Device;

    fn name() -> &'static str {
        "host"
    }

    fn map_device(device: &Device) -> Result<Self::Device, OpsError> {
        match device {
            Device::Cpu => Ok(Device::Cpu),
            other => Err(OpsError::Message(format!(
                "host backend only supports CPU, got {other:?}"
            ))),
        }
    }

    fn zeros(
        shape: &[usize],
        dtype: DType,
        _device: &Self::Device,
    ) -> Result<Self::Tensor, OpsError> {
        if dtype != DType::F32 {
            return Err(OpsError::DType("host backend is f32-only".into()));
        }
        Ok(HostTensor {
            data: vec![0.0; numel(shape)],
            shape: shape.to_vec(),
        })
    }

    fn from_f32(
        data: &[f32],
        shape: &[usize],
        _device: &Self::Device,
    ) -> Result<Self::Tensor, OpsError> {
        if data.len() != numel(shape) {
            return Err(OpsError::Shape(format!(
                "data len {} does not match shape {:?}",
                data.len(),
                shape
            )));
        }
        Ok(HostTensor {
            data: data.to_vec(),
            shape: shape.to_vec(),
        })
    }

    fn to_f32(tensor: &Self::Tensor) -> Result<Vec<f32>, OpsError> {
        Ok(tensor.data.clone())
    }

    fn shape(tensor: &Self::Tensor) -> Vec<usize> {
        tensor.shape.clone()
    }

    fn dtype(_tensor: &Self::Tensor) -> DType {
        DType::F32
    }

    fn add(a: &Self::Tensor, b: &Self::Tensor) -> Result<Self::Tensor, OpsError> {
        same_shape(&a.shape, &b.shape)?;
        Ok(HostTensor {
            data: a.data.iter().zip(&b.data).map(|(x, y)| x + y).collect(),
            shape: a.shape.clone(),
        })
    }

    fn mul(a: &Self::Tensor, b: &Self::Tensor) -> Result<Self::Tensor, OpsError> {
        same_shape(&a.shape, &b.shape)?;
        Ok(HostTensor {
            data: a.data.iter().zip(&b.data).map(|(x, y)| x * y).collect(),
            shape: a.shape.clone(),
        })
    }

    fn mul_scalar(a: &Self::Tensor, scale: f32) -> Result<Self::Tensor, OpsError> {
        Ok(HostTensor {
            data: a.data.iter().map(|x| x * scale).collect(),
            shape: a.shape.clone(),
        })
    }

    fn matmul(a: &Self::Tensor, b: &Self::Tensor) -> Result<Self::Tensor, OpsError> {
        if a.shape.len() != 2 || b.shape.len() != 2 {
            return Err(OpsError::Shape("host matmul expects rank-2 tensors".into()));
        }
        let (m, k) = (a.shape[0], a.shape[1]);
        let (k2, n) = (b.shape[0], b.shape[1]);
        if k != k2 {
            return Err(OpsError::Shape(format!("inner dim {k} vs {k2}")));
        }
        let mut out = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0.0;
                for t in 0..k {
                    acc += a.data[i * k + t] * b.data[t * n + j];
                }
                out[i * n + j] = acc;
            }
        }
        Ok(HostTensor {
            data: out,
            shape: vec![m, n],
        })
    }

    fn random_normal(
        shape: &[usize],
        mean: f32,
        std: f32,
        seed: u64,
        _device: &Self::Device,
    ) -> Result<Self::Tensor, OpsError> {
        // Box-Muller with a tiny LCG so tests are deterministic without rand.
        let mut state = seed | 1;
        let mut data = Vec::with_capacity(numel(shape));
        while data.len() < numel(shape) {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let u1 = ((state >> 33) as f32 / (1u32 << 31) as f32).clamp(1e-7, 1.0);
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let u2 = (state >> 33) as f32 / (1u32 << 31) as f32;
            let mag = (-2.0 * u1.ln()).sqrt();
            let z0 = mag * (2.0 * std::f32::consts::PI * u2).cos();
            data.push(mean + std * z0);
        }
        data.truncate(numel(shape));
        Ok(HostTensor {
            data,
            shape: shape.to_vec(),
        })
    }

    fn silu(a: &Self::Tensor) -> Result<Self::Tensor, OpsError> {
        Ok(HostTensor {
            data: a
                .data
                .iter()
                .map(|x| x / (1.0 + (-x).exp()))
                .collect(),
            shape: a.shape.clone(),
        })
    }

    fn gelu(a: &Self::Tensor) -> Result<Self::Tensor, OpsError> {
        // tanh approximation (Diffusers gelu-approximate)
        let c = (2.0 / std::f32::consts::PI).sqrt();
        Ok(HostTensor {
            data: a
                .data
                .iter()
                .map(|x| {
                    let inner = c * (x + 0.044715 * x.powi(3));
                    0.5 * x * (1.0 + inner.tanh())
                })
                .collect(),
            shape: a.shape.clone(),
        })
    }

    fn softmax(a: &Self::Tensor, dim: usize) -> Result<Self::Tensor, OpsError> {
        if dim + 1 != a.shape.len() {
            return Err(OpsError::Shape(
                "host softmax currently implements the last dimension only".into(),
            ));
        }
        let last = *a.shape.last().unwrap_or(&1);
        let rows = numel(&a.shape) / last;
        let mut data = a.data.clone();
        for r in 0..rows {
            let row = &mut data[r * last..(r + 1) * last];
            let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            for v in row.iter_mut() {
                *v = (*v - max).exp();
                sum += *v;
            }
            for v in row.iter_mut() {
                *v /= sum;
            }
        }
        Ok(HostTensor {
            data,
            shape: a.shape.clone(),
        })
    }

    fn rms_norm(
        a: &Self::Tensor,
        weight: &Self::Tensor,
        eps: f32,
    ) -> Result<Self::Tensor, OpsError> {
        let last = *a.shape.last().unwrap_or(&1);
        if weight.data.len() != last {
            return Err(OpsError::Shape("rms_norm weight dim".into()));
        }
        let rows = numel(&a.shape) / last;
        let mut data = vec![0.0f32; a.data.len()];
        for r in 0..rows {
            let row = &a.data[r * last..(r + 1) * last];
            let mean_sq: f32 = row.iter().map(|x| x * x).sum::<f32>() / last as f32;
            let inv = (mean_sq + eps).sqrt().recip();
            for (i, x) in row.iter().enumerate() {
                data[r * last + i] = x * inv * weight.data[i];
            }
        }
        Ok(HostTensor {
            data,
            shape: a.shape.clone(),
        })
    }

    fn layer_norm(
        a: &Self::Tensor,
        weight: &Self::Tensor,
        bias: Option<&Self::Tensor>,
        eps: f32,
    ) -> Result<Self::Tensor, OpsError> {
        let last = *a.shape.last().unwrap_or(&1);
        let rows = numel(&a.shape) / last;
        let mut data = vec![0.0f32; a.data.len()];
        for r in 0..rows {
            let row = &a.data[r * last..(r + 1) * last];
            let mean = row.iter().sum::<f32>() / last as f32;
            let var = row.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / last as f32;
            let inv = (var + eps).sqrt().recip();
            for (i, x) in row.iter().enumerate() {
                let mut y = (x - mean) * inv;
                if weight.data.len() == last {
                    y *= weight.data[i];
                }
                if let Some(b) = bias {
                    if b.data.len() == last {
                        y += b.data[i];
                    }
                }
                data[r * last + i] = y;
            }
        }
        Ok(HostTensor {
            data,
            shape: a.shape.clone(),
        })
    }

    fn scaled_dot_product_attention(
        query: &Self::Tensor,
        key: &Self::Tensor,
        value: &Self::Tensor,
        scale: Option<f32>,
    ) -> Result<Self::Tensor, OpsError> {
        if query.shape.len() != 4 || key.shape != query.shape || value.shape != query.shape {
            return Err(OpsError::Shape("host sdpa expects [B,H,S,D]".into()));
        }
        let (b, h, s, d) = (
            query.shape[0],
            query.shape[1],
            query.shape[2],
            query.shape[3],
        );
        let scale = scale.unwrap_or(1.0 / (d as f32).sqrt());
        let mut out = vec![0.0f32; b * h * s * d];
        for bi in 0..b {
            for hi in 0..h {
                let q = |si: usize, di: usize| query.data[((bi * h + hi) * s + si) * d + di];
                let k = |si: usize, di: usize| key.data[((bi * h + hi) * s + si) * d + di];
                let v = |si: usize, di: usize| value.data[((bi * h + hi) * s + si) * d + di];
                let mut attn = vec![0.0f32; s * s];
                for i in 0..s {
                    let mut max = f32::NEG_INFINITY;
                    for j in 0..s {
                        let mut dot = 0.0;
                        for di in 0..d {
                            dot += q(i, di) * k(j, di);
                        }
                        let sij = dot * scale;
                        attn[i * s + j] = sij;
                        max = max.max(sij);
                    }
                    let mut sum = 0.0;
                    for j in 0..s {
                        attn[i * s + j] = (attn[i * s + j] - max).exp();
                        sum += attn[i * s + j];
                    }
                    for j in 0..s {
                        attn[i * s + j] /= sum;
                    }
                    for di in 0..d {
                        let mut acc = 0.0;
                        for j in 0..s {
                            acc += attn[i * s + j] * v(j, di);
                        }
                        out[((bi * h + hi) * s + i) * d + di] = acc;
                    }
                }
            }
        }
        Ok(HostTensor {
            data: out,
            shape: query.shape.clone(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matmul_identity() {
        let device = Device::cpu();
        let a = HostBackend::from_f32(&[1.0, 2.0, 3.0, 4.0], &[2, 2], &device).unwrap();
        let i = HostBackend::from_f32(&[1.0, 0.0, 0.0, 1.0], &[2, 2], &device).unwrap();
        let out = HostBackend::matmul(&a, &i).unwrap();
        assert_eq!(out.data, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn silu_matches_numpy() {
        let device = Device::cpu();
        let a = HostBackend::from_f32(&[-2.0, -0.5, 0.0, 0.25, 1.5], &[5], &device).unwrap();
        let y = HostBackend::silu(&a).unwrap();
        let exp = [-0.23840584, -0.18877033, 0.0, 0.14054413, 1.2263617];
        for (g, e) in y.data.iter().zip(exp) {
            assert!((g - e).abs() < 1e-6);
        }
    }

    #[test]
    fn sdpa_identity_value() {
        let device = Device::cpu();
        // One head, seq=2, dim=2. Q=K so softmax is well-defined.
        let q = HostBackend::from_f32(&[1.0, 0.0, 0.0, 1.0], &[1, 1, 2, 2], &device).unwrap();
        let v = HostBackend::from_f32(&[1.0, 2.0, 3.0, 4.0], &[1, 1, 2, 2], &device).unwrap();
        let out = HostBackend::scaled_dot_product_attention(&q, &q, &v, None).unwrap();
        assert_eq!(out.shape, vec![1, 1, 2, 2]);
        assert_eq!(out.data.len(), 4);
    }
}
