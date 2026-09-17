//! Device-resident weights and residency policy for the cudarc Wan graph.
//!
//! Residency is **on by default** when a global CUDA device is live. Set
//! `FASTVIDEO_RESIDENT=0` (or `false`) to force the host-upload-per-op path.
//!
//! [`CachedLinear`] pins weight/bias on device and runs GEMM without re-uploading
//! weights. Prefer [`super::nn::Linear`], which pins on `load` and keeps
//! activations device-resident when [`CudaTensor`] dual-storage is active.

use super::tensor::{CudaTensor, Result, TensorError};

#[cfg(feature = "cuda")]
use super::device;

use super::envflag::CachedBool;

static RESIDENT_CACHE: CachedBool = CachedBool::new();

/// Whether device residency is enabled (default **true**; `FASTVIDEO_RESIDENT=0` disables).
///
/// Cached after first read: this is called from nearly every `CudaTensor` op
/// (`add`, `mul`, `matmul`, `Linear::forward`, …), so a raw `std::env::var`
/// per call would add real overhead (a process-wide lock + allocation) across
/// a generate run with thousands of ops. The env var is only ever consulted
/// at process start in practice; set it before the first op if scripting.
pub fn residency_enabled() -> bool {
    RESIDENT_CACHE.get_or_init(|| super::envflag::bool_flag("FASTVIDEO_RESIDENT", true))
}

/// Host-mirrored linear with optional CUDA-resident weight/bias.
///
/// Prefer [`super::nn::Linear`] for new code; this remains for explicit pin tests.
pub struct CachedLinear {
    pub weight: CudaTensor, // [out, in] host mirror
    pub bias: Option<CudaTensor>,
    #[cfg(feature = "cuda")]
    weight_dev: Option<cudarc::driver::CudaSlice<f32>>,
    #[cfg(feature = "cuda")]
    bias_dev: Option<cudarc::driver::CudaSlice<f32>>,
    out_dim: usize,
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    in_dim: usize,
}

impl CachedLinear {
    pub fn from_tensors(weight: CudaTensor, bias: Option<CudaTensor>) -> Result<Self> {
        if weight.rank() != 2 {
            return Err(TensorError::Message("CachedLinear weight must be 2D".into()));
        }
        let out_dim = weight.shape[0];
        let in_dim = weight.shape[1];
        let mut s = Self {
            weight,
            bias,
            #[cfg(feature = "cuda")]
            weight_dev: None,
            #[cfg(feature = "cuda")]
            bias_dev: None,
            out_dim,
            in_dim,
        };
        s.try_pin();
        Ok(s)
    }

    pub fn try_pin(&mut self) {
        #[cfg(feature = "cuda")]
        {
            if !residency_enabled() {
                return;
            }
            let Some(dev) = device::global_device() else {
                return;
            };
            let _ = self.weight.ensure_host();
            if let Ok(slice) = dev.stream.memcpy_stod(&self.weight.data) {
                self.weight_dev = Some(slice);
            }
            if let Some(bias) = &mut self.bias {
                let _ = bias.ensure_host();
                if let Ok(slice) = dev.stream.memcpy_stod(&bias.data) {
                    self.bias_dev = Some(slice);
                }
            }
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = residency_enabled();
        }
    }

    /// `(…, in) @ weight^T + bias` using resident weights when pinned.
    pub fn forward(&self, xs: &CudaTensor) -> Result<CudaTensor> {
        #[cfg(feature = "cuda")]
        {
            if let (Some(w_dev), Some(dev)) = (&self.weight_dev, device::global_device()) {
                return self.forward_resident(xs, &dev, w_dev);
            }
        }
        let w_t = self.weight.transpose(0, 1)?;
        let mut out = match xs.rank() {
            2 => xs.matmul(&w_t)?,
            3 => {
                let (b, s, i) = (xs.shape[0], xs.shape[1], xs.shape[2]);
                let flat = xs.reshape(vec![b * s, i])?;
                flat.matmul(&w_t)?.reshape(vec![b, s, self.out_dim])?
            }
            _ => {
                return Err(TensorError::Message(format!(
                    "CachedLinear unsupported rank {}",
                    xs.rank()
                )));
            }
        };
        if let Some(bias) = &self.bias {
            let mut bshape = vec![1; out.rank()];
            bshape[out.rank() - 1] = bias.shape[0];
            let b = bias.reshape(bshape)?;
            out = out.add(&b)?;
        }
        Ok(out)
    }

    #[cfg(feature = "cuda")]
    fn forward_resident(
        &self,
        xs: &CudaTensor,
        dev: &std::sync::Arc<device::DeviceContext>,
        w_dev: &cudarc::driver::CudaSlice<f32>,
    ) -> Result<CudaTensor> {
        let (m, k, n, out_shape) = match xs.rank() {
            2 => {
                let m = xs.shape[0];
                (m, self.in_dim, self.out_dim, vec![m, self.out_dim])
            }
            3 => {
                let (b, s) = (xs.shape[0], xs.shape[1]);
                (
                    b * s,
                    self.in_dim,
                    self.out_dim,
                    vec![b, s, self.out_dim],
                )
            }
            _ => {
                return Err(TensorError::Message(
                    "CachedLinear resident path needs rank 2 or 3".into(),
                ));
            }
        };
        let mut x = xs.clone();
        x.ensure_device()?;
        let x_dev = x.device_slice().ok_or_else(|| {
            TensorError::Message("CachedLinear: activation not on device".into())
        })?;
        if x_dev.len() != m * k {
            return Err(TensorError::Message(
                "CachedLinear activation size mismatch".into(),
            ));
        }
        let mut c_dev = dev
            .stream
            .alloc_zeros::<f32>(m * n)
            .map_err(|e| TensorError::Message(e.to_string()))?;
        device::matmul_linear_wt_device(x_dev, w_dev, &mut c_dev, m, k, n)
            .map_err(|e| TensorError::Message(e.to_string()))?;
        if let Some(bias_dev) = &self.bias_dev {
            super::ops::add_bias_last_inplace(&mut c_dev, bias_dev)?;
        } else if let Some(bias) = &self.bias {
            let mut host = dev
                .stream
                .memcpy_dtov(&c_dev)
                .map_err(|e| TensorError::Message(e.to_string()))?;
            let bias_host = bias.host_cow()?;
            for row in 0..m {
                for j in 0..n {
                    host[row * n + j] += bias_host[j];
                }
            }
            return CudaTensor::from_vec(host, out_shape);
        }
        CudaTensor::from_device_slice(c_dev, out_shape)
    }
}

/// Convert host f32 weights to BF16 bytes (for BF16 GEMM / storage).
pub fn f32_weights_to_bf16(values: &[f32]) -> Vec<u8> {
    super::ops::f32_to_bf16_bytes(values)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cached_linear_matches_host_matmul() {
        let w = CudaTensor::from_vec(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]).unwrap();
        let lin = CachedLinear::from_tensors(w, None).unwrap();
        let x = CudaTensor::from_vec(vec![2.0, 3.0], vec![1, 2]).unwrap();
        let y = lin.forward(&x).unwrap();
        let mut y = y;
        y.ensure_host().unwrap();
        assert_eq!(y.data, vec![2.0, 3.0]);
    }

    #[test]
    fn residency_default_on() {
        // Unset → enabled. Do not rely on ambient env in CI; just check the parser.
        let prev = std::env::var("FASTVIDEO_RESIDENT").ok();
        std::env::remove_var("FASTVIDEO_RESIDENT");
        RESIDENT_CACHE.reset();
        assert!(residency_enabled());
        std::env::set_var("FASTVIDEO_RESIDENT", "0");
        RESIDENT_CACHE.reset();
        assert!(!residency_enabled());
        std::env::set_var("FASTVIDEO_RESIDENT", "false");
        RESIDENT_CACHE.reset();
        assert!(!residency_enabled());
        match prev {
            Some(v) => std::env::set_var("FASTVIDEO_RESIDENT", v),
            None => std::env::remove_var("FASTVIDEO_RESIDENT"),
        }
        RESIDENT_CACHE.reset();
    }
}
