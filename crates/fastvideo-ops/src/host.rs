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
}
