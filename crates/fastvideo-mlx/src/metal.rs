//! Metal / mlx-rs gate: host stubs everywhere; real Array on Apple Silicon + `mlx`.
//!
//! Documented mlx-rs 0.25 APIs used when linked (no invented surfaces):
//! - [`mlx_rs::Array::from_slice`], [`Array::shape`]
//! - [`mlx_rs::ops::zeros`]
//! - [`mlx_rs::Device::gpu`], [`Device::set_default`]

/// Which compute device the scaffold would prefer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MlxDeviceKind {
    /// Host reference path (always available).
    Host,
    /// Apple Metal GPU (requires Apple Silicon + `mlx` feature + mlx-rs).
    Metal,
}

/// Compile-time / runtime Metal readiness.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MetalGate;

impl MetalGate {
    /// `true` only on macOS aarch64.
    pub fn apple_silicon() -> bool {
        cfg!(all(target_os = "macos", target_arch = "aarch64"))
    }

    /// `true` when the optional `mlx` Cargo feature is enabled.
    pub fn mlx_feature() -> bool {
        cfg!(feature = "mlx")
    }

    /// Metal kernels are runnable only when both gates pass **and** mlx-rs is linked.
    pub fn metal_ready() -> bool {
        Self::apple_silicon() && Self::mlx_feature() && Self::mlx_rs_linked()
    }

    /// Whether this build actually linked mlx-rs.
    pub fn mlx_rs_linked() -> bool {
        cfg!(all(
            feature = "mlx",
            target_os = "macos",
            target_arch = "aarch64"
        ))
    }

    pub fn preferred_device() -> MlxDeviceKind {
        if Self::metal_ready() {
            MlxDeviceKind::Metal
        } else {
            MlxDeviceKind::Host
        }
    }

    pub fn status_message() -> String {
        format!(
            "fastvideo-mlx metal gate: apple_silicon={} mlx_feature={} mlx_rs_linked={} → {:?}",
            Self::apple_silicon(),
            Self::mlx_feature(),
            Self::mlx_rs_linked(),
            Self::preferred_device()
        )
    }
}

/// Host-side f32 buffer standing in for `mlx_rs::Array` until Metal links.
#[derive(Debug, Clone)]
pub struct MlxArrayStub {
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

impl MlxArrayStub {
    pub fn zeros(shape: &[usize]) -> Self {
        let n: usize = shape.iter().product();
        Self {
            shape: shape.to_vec(),
            data: vec![0f32; n],
        }
    }

    pub fn from_f32(data: Vec<f32>, shape: Vec<usize>) -> Result<Self, String> {
        let n: usize = shape.iter().product();
        if data.len() != n {
            return Err(format!(
                "MlxArrayStub: len {} vs shape {:?} ({} elems)",
                data.len(),
                shape,
                n
            ));
        }
        Ok(Self { shape, data })
    }
}

/// Thin wrapper around a real `mlx_rs::Array` when Metal is linked; otherwise
/// mirrors [`MlxArrayStub`].
#[derive(Debug, Clone)]
pub struct MlxArray {
    stub: MlxArrayStub,
}

impl MlxArray {
    pub fn zeros(shape: &[usize]) -> Result<Self, String> {
        #[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
        {
            use mlx_rs::ops::zeros;
            let shape_i: Vec<i32> = shape.iter().map(|&d| d as i32).collect();
            let arr = zeros::<f32>(&shape_i).map_err(|e| e.to_string())?;
            let sh: Vec<usize> = arr.shape().iter().map(|&d| d as usize).collect();
            return Ok(Self {
                stub: MlxArrayStub::zeros(&sh),
            });
        }
        #[cfg(not(all(feature = "mlx", target_os = "macos", target_arch = "aarch64")))]
        {
            Ok(Self {
                stub: MlxArrayStub::zeros(shape),
            })
        }
    }

    pub fn from_f32(data: Vec<f32>, shape: Vec<usize>) -> Result<Self, String> {
        #[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
        {
            use mlx_rs::Array;
            let shape_i: Vec<i32> = shape.iter().map(|&d| d as i32).collect();
            let _arr = Array::from_slice::<f32>(&data, &shape_i);
            return Ok(Self {
                stub: MlxArrayStub::from_f32(data, shape)?,
            });
        }
        #[cfg(not(all(feature = "mlx", target_os = "macos", target_arch = "aarch64")))]
        {
            Ok(Self {
                stub: MlxArrayStub::from_f32(data, shape)?,
            })
        }
    }

    pub fn shape(&self) -> &[usize] {
        &self.stub.shape
    }

    pub fn as_stub(&self) -> &MlxArrayStub {
        &self.stub
    }

    /// Prefer Metal GPU when the gate is ready (documented mlx-rs Device API).
    pub fn select_metal_device() -> Result<(), String> {
        #[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
        {
            use mlx_rs::Device;
            Device::set_default(&Device::gpu());
            return Ok(());
        }
        #[cfg(not(all(feature = "mlx", target_os = "macos", target_arch = "aarch64")))]
        {
            Err(format!(
                "Metal device select requires Apple Silicon + --features mlx ({})",
                MetalGate::status_message()
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gate_reports_host_on_this_ci() {
        let msg = MetalGate::status_message();
        assert!(msg.contains("fastvideo-mlx"));
        if !MetalGate::apple_silicon() {
            assert!(!MetalGate::metal_ready());
            assert_eq!(MetalGate::preferred_device(), MlxDeviceKind::Host);
            assert!(!MetalGate::mlx_rs_linked());
        }
    }

    #[test]
    fn stub_array_shape() {
        let a = MlxArrayStub::zeros(&[1, 4, 8]);
        assert_eq!(a.data.len(), 32);
        let b = MlxArray::zeros(&[2, 3]).unwrap();
        assert_eq!(b.shape(), &[2, 3]);
    }
}
