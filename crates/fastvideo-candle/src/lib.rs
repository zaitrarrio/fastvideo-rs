//! Candle adapter. Phase 0 implements alloc / arithmetic / matmul on CPU.

use candle_core::{DType as CandleDType, Device as CandleDevice, Tensor};
use fastvideo_ops::{Device, DType, OpsError, TensorBackend};

#[derive(Debug, Clone, Copy, Default)]
pub struct CandleBackend;

fn map_err(err: candle_core::Error) -> OpsError {
    OpsError::Message(err.to_string())
}

fn to_candle_dtype(dtype: DType) -> Result<CandleDType, OpsError> {
    match dtype {
        DType::F32 => Ok(CandleDType::F32),
        DType::F16 => Ok(CandleDType::F16),
        DType::BF16 => Ok(CandleDType::BF16),
        DType::I32 => Ok(CandleDType::I32),
        DType::U32 => Ok(CandleDType::U32),
    }
}

fn from_candle_dtype(dtype: CandleDType) -> DType {
    match dtype {
        CandleDType::F32 => DType::F32,
        CandleDType::F16 => DType::F16,
        CandleDType::BF16 => DType::BF16,
        CandleDType::I32 => DType::I32,
        CandleDType::U32 => DType::U32,
        _ => DType::F32,
    }
}

impl TensorBackend for CandleBackend {
    type Tensor = Tensor;
    type Device = CandleDevice;

    fn name() -> &'static str {
        "candle"
    }

    fn map_device(device: &Device) -> Result<Self::Device, OpsError> {
        match device {
            Device::Cpu => Ok(CandleDevice::Cpu),
            Device::Cuda { index } => {
                #[cfg(feature = "cuda")]
                {
                    CandleDevice::new_cuda(*index).map_err(map_err)
                }
                #[cfg(not(feature = "cuda"))]
                {
                    let _ = index;
                    Err(OpsError::Message(
                        "rebuild with --features cuda for CUDA devices".into(),
                    ))
                }
            }
            other => Err(OpsError::Message(format!(
                "candle adapter does not map {other:?} yet"
            ))),
        }
    }

    fn zeros(
        shape: &[usize],
        dtype: DType,
        device: &Self::Device,
    ) -> Result<Self::Tensor, OpsError> {
        Tensor::zeros(shape, to_candle_dtype(dtype)?, device).map_err(map_err)
    }

    fn from_f32(
        data: &[f32],
        shape: &[usize],
        device: &Self::Device,
    ) -> Result<Self::Tensor, OpsError> {
        Tensor::from_slice(data, shape, device).map_err(map_err)
    }

    fn to_f32(tensor: &Self::Tensor) -> Result<Vec<f32>, OpsError> {
        tensor
            .flatten_all()
            .map_err(map_err)?
            .to_vec1::<f32>()
            .map_err(map_err)
    }

    fn shape(tensor: &Self::Tensor) -> Vec<usize> {
        tensor.dims().to_vec()
    }

    fn dtype(tensor: &Self::Tensor) -> DType {
        from_candle_dtype(tensor.dtype())
    }

    fn add(a: &Self::Tensor, b: &Self::Tensor) -> Result<Self::Tensor, OpsError> {
        (a + b).map_err(map_err)
    }

    fn mul(a: &Self::Tensor, b: &Self::Tensor) -> Result<Self::Tensor, OpsError> {
        (a * b).map_err(map_err)
    }

    fn mul_scalar(a: &Self::Tensor, scale: f32) -> Result<Self::Tensor, OpsError> {
        (a * scale as f64).map_err(map_err)
    }

    fn matmul(a: &Self::Tensor, b: &Self::Tensor) -> Result<Self::Tensor, OpsError> {
        a.matmul(b).map_err(map_err)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zeros_and_add() {
        let device = CandleBackend::map_device(&Device::cpu()).unwrap();
        let a = CandleBackend::from_f32(&[1.0, 2.0, 3.0], &[3], &device).unwrap();
        let b = CandleBackend::zeros(&[3], DType::F32, &device).unwrap();
        let out = CandleBackend::add(&a, &b).unwrap();
        assert_eq!(CandleBackend::to_f32(&out).unwrap(), vec![1.0, 2.0, 3.0]);
    }
}
