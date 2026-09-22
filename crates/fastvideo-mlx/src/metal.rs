//! Metal / mlx-rs gate: host stubs that compile everywhere.
//!
//! Real `mlx_rs::Array` / `Device` types are only available behind
//! `feature = "mlx"` **and** `aarch64-apple-darwin`. Until that target+feature
//! combo is enabled in Cargo.toml, these stubs document the intended surface
//! without inventing mlx-rs APIs.

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

    /// Whether this build actually linked mlx-rs (always false until target dep lands).
    pub fn mlx_rs_linked() -> bool {
        // Deliberately false: Cargo.toml does not depend on mlx-rs so CI on
        // x86_64/Linux stays green. Flip when adding the target-specific dep.
        false
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gate_reports_host_on_this_ci() {
        let msg = MetalGate::status_message();
        assert!(msg.contains("fastvideo-mlx"));
        // This workspace CI host is x86_64 Darwin — Metal must not claim ready.
        if !MetalGate::apple_silicon() {
            assert!(!MetalGate::metal_ready());
            assert_eq!(MetalGate::preferred_device(), MlxDeviceKind::Host);
        }
    }

    #[test]
    fn stub_array_shape() {
        let a = MlxArrayStub::zeros(&[1, 4, 8]);
        assert_eq!(a.data.len(), 32);
    }
}
