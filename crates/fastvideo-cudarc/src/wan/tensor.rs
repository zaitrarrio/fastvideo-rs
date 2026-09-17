//! Owned f32 N-D tensors for the cudarc CUDA Wan backend.
//!
//! With `--features cuda` and a live global device, hot ops (matmul, same-shape
//! elementwise, last-axis softmax/rms_norm, NCHW conv2d) run on GPU via
//! [`super::ops`] / [`super::device`].
//!
//! **Dual storage:** optional device-resident buffer plus a host mirror.
//! Residency is on by default (`FASTVIDEO_RESIDENT=0` disables). Hot ops keep
//! results on device; call [`CudaTensor::ensure_host`] before reading `.data`
//! (UniPC step, PNG write, TeaCache). Broadcast ops stay on host.

use std::borrow::Cow;

use thiserror::Error;

#[derive(Debug, Error)]
pub enum TensorError {
    #[error("{0}")]
    Message(String),
}

pub type Result<T> = std::result::Result<T, TensorError>;

/// Per-op device-vs-host dispatch counters, keyed by a fixed small set of
/// hot-op names (not a `HashMap` — this is on the per-call path, so it's a
/// few plain atomics, not a locked map). Read via [`device_path_stats`],
/// reset via [`reset_device_path_stats`] (test/bench use). Purely additive
/// bookkeeping: recording never changes control flow, unlike
/// [`strict_device_check`].
#[derive(Debug, Default)]
struct DevicePathCounters {
    layer_norm: (std::sync::atomic::AtomicU64, std::sync::atomic::AtomicU64),
    modulate: (std::sync::atomic::AtomicU64, std::sync::atomic::AtomicU64),
    gate_mul: (std::sync::atomic::AtomicU64, std::sync::atomic::AtomicU64),
    attention: (std::sync::atomic::AtomicU64, std::sync::atomic::AtomicU64),
}

static DEVICE_PATH_COUNTERS: DevicePathCounters = DevicePathCounters {
    layer_norm: (
        std::sync::atomic::AtomicU64::new(0),
        std::sync::atomic::AtomicU64::new(0),
    ),
    modulate: (
        std::sync::atomic::AtomicU64::new(0),
        std::sync::atomic::AtomicU64::new(0),
    ),
    gate_mul: (
        std::sync::atomic::AtomicU64::new(0),
        std::sync::atomic::AtomicU64::new(0),
    ),
    attention: (
        std::sync::atomic::AtomicU64::new(0),
        std::sync::atomic::AtomicU64::new(0),
    ),
};

fn counter_for(op: &str) -> Option<&'static (std::sync::atomic::AtomicU64, std::sync::atomic::AtomicU64)> {
    match op {
        "layer_norm" => Some(&DEVICE_PATH_COUNTERS.layer_norm),
        "modulate" => Some(&DEVICE_PATH_COUNTERS.modulate),
        "gate_mul" => Some(&DEVICE_PATH_COUNTERS.gate_mul),
        "attention" => Some(&DEVICE_PATH_COUNTERS.attention),
        _ => None,
    }
}

pub(crate) fn record_device_hit(op: &str) {
    if let Some((hits, _)) = counter_for(op) {
        hits.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
}

fn record_host_fallback(op: &str) {
    if let Some((_, misses)) = counter_for(op) {
        misses.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
}

/// `(op, device_hits, host_fallbacks)` for every tracked hot op, as of now.
/// Call after a real (ideally GPU) `generate()` run to audit dispatch —
/// unlike [`strict_device_check`] this never changes behavior, so it's safe
/// to enable unconditionally; the counters themselves are always updated
/// regardless of `FASTVIDEO_STRICT_DEVICE`.
pub fn device_path_stats() -> Vec<(&'static str, u64, u64)> {
    use std::sync::atomic::Ordering::Relaxed;
    [
        ("layer_norm", &DEVICE_PATH_COUNTERS.layer_norm),
        ("modulate", &DEVICE_PATH_COUNTERS.modulate),
        ("gate_mul", &DEVICE_PATH_COUNTERS.gate_mul),
        ("attention", &DEVICE_PATH_COUNTERS.attention),
    ]
    .into_iter()
    .map(|(name, (hits, misses))| (name, hits.load(Relaxed), misses.load(Relaxed)))
    .collect()
}

/// Test/bench-only: zero every counter so a fresh run's stats aren't mixed
/// with a previous one's (counters are process-global statics).
#[cfg(any(test, feature = "cuda"))]
pub fn reset_device_path_stats() {
    use std::sync::atomic::Ordering::Relaxed;
    for (hits, misses) in [
        &DEVICE_PATH_COUNTERS.layer_norm,
        &DEVICE_PATH_COUNTERS.modulate,
        &DEVICE_PATH_COUNTERS.gate_mul,
        &DEVICE_PATH_COUNTERS.attention,
    ] {
        hits.store(0, Relaxed);
        misses.store(0, Relaxed);
    }
}

/// Returns `Err` when `FASTVIDEO_STRICT_DEVICE=1`, residency is enabled, and
/// a CUDA device is actually live, but `op` is about to fall back to a host
/// computation anyway.
///
/// This is the direct answer to "how do we know a GPU code path didn't
/// silently run on CPU": every device op in this crate is structured as
/// *try the device kernel, and if it returns `None` for any reason (shape
/// guard, allocation failure, residency toggled off, no device), silently
/// compute on host instead* — which is the right default for portability
/// (CPU-only builds, dev-time host tests) but means a real GPU run where
/// something is subtly broken still produces correct output, just slower,
/// with nothing in the logs to say so. Call this right before the host-path
/// computation begins, and only for ops that *have* a device kernel — set
/// `FASTVIDEO_STRICT_DEVICE=1` on a real GPU run (bench or CI smoke test) and
/// a silent fallback becomes a loud, specific error naming the op and shapes
/// involved, instead of an unexplained slowdown.
///
/// Do **not** call this for ops with no device kernel at all (`div`,
/// `mean_keepdim`, `sqrt`, non-trailing-axis `narrow`/`chunk`/`cat`, sparse
/// block-attention) — those are known, accepted gaps, not per-call
/// regressions, and strict mode would just always fire there with zero
/// diagnostic value.
///
/// Always records to [`device_path_stats`] regardless of strict mode, so the
/// counters stay meaningful even on a host-only run where strict mode can
/// never fire (no device to be strict about).
static STRICT_DEVICE_CACHE: super::envflag::CachedBool = super::envflag::CachedBool::new();

pub(crate) fn strict_device_check(op: &str, detail: impl std::fmt::Display) -> Result<()> {
    let strict = STRICT_DEVICE_CACHE.get_or_init(|| {
        super::envflag::bool_flag("FASTVIDEO_STRICT_DEVICE", false)
    });

    let device_live = super::device::has_live_device();
    if device_live && super::resident::residency_enabled() {
        record_host_fallback(op);
    }
    if !strict || !device_live || !super::resident::residency_enabled() {
        return Ok(());
    }
    Err(TensorError::Message(format!(
        "FASTVIDEO_STRICT_DEVICE=1: '{op}' fell back to host compute with a live CUDA device \
         and residency enabled ({detail}). This op has a device kernel — falling back here \
         means the device path failed or was bypassed unexpectedly; treat this as a bug to fix, \
         not a performance note. Unset FASTVIDEO_STRICT_DEVICE to run anyway."
    )))
}

/// Shareable device buffer (cudarc `CudaSlice<f32>`).
#[cfg(feature = "cuda")]
#[derive(Clone)]
pub struct DeviceBuffer {
    pub(crate) slice: std::sync::Arc<cudarc::driver::CudaSlice<f32>>,
}

#[cfg(feature = "cuda")]
impl std::fmt::Debug for DeviceBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DeviceBuffer")
            .field("len", &self.slice.len())
            .finish()
    }
}

#[derive(Debug)]
pub struct CudaTensor {
    /// Host mirror. May be stale when device is fresher — call [`Self::ensure_host`].
    pub data: Vec<f32>,
    pub shape: Vec<usize>,
    #[cfg(feature = "cuda")]
    device: Option<DeviceBuffer>,
    #[cfg(feature = "cuda")]
    device_fresh: bool,
}

#[cfg(feature = "cuda")]
impl Clone for CudaTensor {
    fn clone(&self) -> Self {
        Self {
            data: self.data.clone(),
            shape: self.shape.clone(),
            device: self.device.clone(),
            device_fresh: self.device_fresh,
        }
    }
}

#[cfg(not(feature = "cuda"))]
impl Clone for CudaTensor {
    fn clone(&self) -> Self {
        Self {
            data: self.data.clone(),
            shape: self.shape.clone(),
        }
    }
}

fn numel(shape: &[usize]) -> usize {
    shape.iter().product()
}

fn check_numel(data: &[f32], shape: &[usize]) -> Result<()> {
    if data.len() != numel(shape) {
        Err(TensorError::Message(format!(
            "data len {} != shape {:?}",
            data.len(),
            shape
        )))
    } else {
        Ok(())
    }
}

impl CudaTensor {
    pub fn new(data: Vec<f32>, shape: Vec<usize>) -> Result<Self> {
        check_numel(&data, &shape)?;
        Ok(Self {
            data,
            shape,
            #[cfg(feature = "cuda")]
            device: None,
            #[cfg(feature = "cuda")]
            device_fresh: false,
        })
    }

    pub fn zeros(shape: &[usize]) -> Self {
        Self {
            data: vec![0.0; numel(shape)],
            shape: shape.to_vec(),
            #[cfg(feature = "cuda")]
            device: None,
            #[cfg(feature = "cuda")]
            device_fresh: false,
        }
    }

    pub fn ones(shape: &[usize]) -> Self {
        Self {
            data: vec![1.0; numel(shape)],
            shape: shape.to_vec(),
            #[cfg(feature = "cuda")]
            device: None,
            #[cfg(feature = "cuda")]
            device_fresh: false,
        }
    }

    pub fn from_vec(data: Vec<f32>, shape: Vec<usize>) -> Result<Self> {
        Self::new(data, shape)
    }

    pub fn from_slice(data: &[f32], shape: &[usize]) -> Result<Self> {
        Self::new(data.to_vec(), shape.to_vec())
    }

    /// Wrap an existing device buffer; host mirror is zeros (stale until ensure_host).
    #[cfg(feature = "cuda")]
    pub fn from_device_slice(
        slice: cudarc::driver::CudaSlice<f32>,
        shape: Vec<usize>,
    ) -> Result<Self> {
        let n = numel(&shape);
        if slice.len() != n {
            return Err(TensorError::Message(format!(
                "device len {} != shape {:?}",
                slice.len(),
                shape
            )));
        }
        Ok(Self {
            data: vec![0.0; n],
            shape,
            device: Some(DeviceBuffer {
                slice: std::sync::Arc::new(slice),
            }),
            device_fresh: true,
        })
    }

    pub fn ensure_host(&mut self) -> Result<()> {
        #[cfg(feature = "cuda")]
        {
            if !self.device_fresh {
                return Ok(());
            }
            let Some(buf) = &self.device else {
                self.device_fresh = false;
                return Ok(());
            };
            let Some(dev) = super::device::global_device() else {
                return Err(TensorError::Message(
                    "ensure_host: device_fresh but no global CUDA device".into(),
                ));
            };
            self.data = dev
                .stream
                .memcpy_dtov(buf.slice.as_ref())
                .map_err(|e| TensorError::Message(e.to_string()))?;
            self.device_fresh = false;
        }
        Ok(())
    }

    pub fn host_cow(&self) -> Result<Cow<'_, [f32]>> {
        #[cfg(feature = "cuda")]
        {
            if self.device_fresh {
                let Some(buf) = &self.device else {
                    return Ok(Cow::Borrowed(self.data.as_slice()));
                };
                let Some(dev) = super::device::global_device() else {
                    return Err(TensorError::Message(
                        "host_cow: device_fresh but no global CUDA device".into(),
                    ));
                };
                let v = dev
                    .stream
                    .memcpy_dtov(buf.slice.as_ref())
                    .map_err(|e| TensorError::Message(e.to_string()))?;
                return Ok(Cow::Owned(v));
            }
        }
        Ok(Cow::Borrowed(self.data.as_slice()))
    }

    pub fn ensure_device(&mut self) -> Result<()> {
        #[cfg(feature = "cuda")]
        {
            if !super::resident::residency_enabled() {
                return Ok(());
            }
            let Some(dev) = super::device::global_device() else {
                return Ok(());
            };
            if self.device_fresh && self.device.is_some() {
                return Ok(());
            }
            let slice = dev
                .stream
                .memcpy_stod(&self.data)
                .map_err(|e| TensorError::Message(e.to_string()))?;
            self.device = Some(DeviceBuffer {
                slice: std::sync::Arc::new(slice),
            });
            self.device_fresh = true;
        }
        Ok(())
    }

    pub fn to_host(&mut self) -> Result<()> {
        self.ensure_host()
    }

    #[cfg(feature = "cuda")]
    pub fn device_slice(&self) -> Option<&cudarc::driver::CudaSlice<f32>> {
        self.device.as_ref().map(|b| b.slice.as_ref())
    }

    #[cfg(feature = "cuda")]
    pub fn is_device_fresh(&self) -> bool {
        self.device_fresh
    }

    pub fn pin_device(&mut self) -> Result<()> {
        self.ensure_device()
    }

    pub(crate) fn host_only(data: Vec<f32>, shape: Vec<usize>) -> Self {
        Self {
            data,
            shape,
            #[cfg(feature = "cuda")]
            device: None,
            #[cfg(feature = "cuda")]
            device_fresh: false,
        }
    }

    pub fn rank(&self) -> usize {
        self.shape.len()
    }

    pub fn dim(&self, axis: usize) -> Result<usize> {
        self.shape
            .get(axis)
            .copied()
            .ok_or_else(|| TensorError::Message(format!("axis {axis} out of range {:?}", self.shape)))
    }

    pub fn reshape(&self, shape: Vec<usize>) -> Result<CudaTensor> {
        if numel(&shape) != numel(&self.shape) {
            return Err(TensorError::Message(format!(
                "data len {} != shape {:?}",
                numel(&self.shape),
                shape
            )));
        }
        #[cfg(feature = "cuda")]
        if self.device_fresh {
            return Ok(Self {
                data: self.data.clone(),
                shape,
                device: self.device.clone(),
                device_fresh: true,
            });
        }
        Ok(Self::host_only(self.data.clone(), shape))
    }

    pub fn reshape_owned(mut self, shape: Vec<usize>) -> Result<CudaTensor> {
        if numel(&shape) != numel(&self.shape) {
            return Err(TensorError::Message(format!(
                "data len {} != shape {:?}",
                numel(&self.shape),
                shape
            )));
        }
        #[cfg(feature = "cuda")]
        if self.device_fresh {
            self.shape = shape;
            return Ok(self);
        }
        self.shape = shape;
        Ok(self)
    }

    /// Resolve negative axes like Candle (`-1` = last).
    fn axis(&self, axis: isize) -> Result<usize> {
        let rank = self.rank() as isize;
        let a = if axis < 0 { rank + axis } else { axis };
        if a < 0 || a as usize >= self.rank() {
            Err(TensorError::Message(format!(
                "axis {axis} invalid for shape {:?}",
                self.shape
            )))
        } else {
            Ok(a as usize)
        }
    }

    pub fn permute(&self, dims: &[usize]) -> Result<CudaTensor> {
        if dims.len() != self.rank() {
            return Err(TensorError::Message("permute rank mismatch".into()));
        }
        let mut seen = vec![false; self.rank()];
        for &d in dims {
            if d >= self.rank() || seen[d] {
                return Err(TensorError::Message("invalid permute".into()));
            }
            seen[d] = true;
        }
        let out_shape: Vec<usize> = dims.iter().map(|&d| self.shape[d]).collect();

        #[cfg(feature = "cuda")]
        if super::resident::residency_enabled() && self.device_fresh {
            if let Some(out) = self.permute_device(dims, &out_shape)? {
                return Ok(out);
            }
        }

        let host = self.host_cow()?;
        let mut out = vec![0.0; host.len()];
        let in_strides = strides(&self.shape);
        let out_strides = strides(&out_shape);
        for out_idx in 0..out.len() {
            let out_coord = unravel(out_idx, &out_strides);
            let mut in_coord = vec![0usize; self.rank()];
            for (o, &d) in dims.iter().enumerate() {
                in_coord[d] = out_coord[o];
            }
            let in_idx = ravel(&in_coord, &in_strides);
            out[out_idx] = host[in_idx];
        }
        let mut t = Self::host_only(out, out_shape);
        let _ = t.ensure_device();
        Ok(t)
    }

    #[cfg(feature = "cuda")]
    fn permute_device(&self, dims: &[usize], out_shape: &[usize]) -> Result<Option<CudaTensor>> {
        let Some(dev) = super::device::global_device() else {
            return Ok(None);
        };
        let Some(src) = self.device_slice() else {
            return Ok(None);
        };
        // Pad to 4D for the NVRTC kernel.
        let mut in4 = [1usize; 4];
        let mut dims4 = [0usize, 1, 2, 3];
        let r = self.rank();
        if r > 4 {
            return Ok(None);
        }
        for i in 0..r {
            in4[4 - r + i] = self.shape[i];
        }
        // Map dims into the padded 4D space: leading axes are identity.
        for i in 0..(4 - r) {
            dims4[i] = i;
        }
        for (o, &d) in dims.iter().enumerate() {
            dims4[4 - r + o] = 4 - r + d;
        }
        let out_dev = super::device::permute_4d_device(src, in4, dims4)
            .map_err(|e| TensorError::Message(e.to_string()))?;
        let _ = dev;
        Ok(Some(Self::from_device_slice(out_dev, out_shape.to_vec())?))
    }

    pub fn transpose(&self, dim0: usize, dim1: usize) -> Result<CudaTensor> {
        let mut dims: Vec<usize> = (0..self.rank()).collect();
        dims.swap(dim0, dim1);
        self.permute(&dims)
    }

    pub fn narrow(&self, dim: usize, start: usize, len: usize) -> Result<CudaTensor> {
        let d = self.dim(dim)?;
        if start + len > d {
            return Err(TensorError::Message(format!(
                "narrow dim={dim} start={start} len={len} of {d}"
            )));
        }
        let mut out_shape = self.shape.clone();
        out_shape[dim] = len;
        // Device-resident dense split: trailing dims are contiguous in both src & dst.
        if dim == self.rank() - 1 {
            let trailing = len;
            let outer: usize = self.shape[..self.rank() - 1].iter().product();
            if outer == 0 {
                let mut t = Self::host_only(vec![], out_shape);
                let _ = t.ensure_device();
                return Ok(t);
            }
            let in_stride = self.shape[self.rank() - 1];
            let out_stride = len;
            let in_offset = start;
            let out_offset = 0;
            let _ = (trailing, in_stride, out_stride, in_offset, out_offset);
            #[cfg(feature = "cuda")]
            {
            let mut src = self.clone();
            src.ensure_device().ok();
            if let (Some(dev), Some(in_dev)) =
                (super::device::global_device(), src.device_slice())
            {
                if src.is_device_fresh() {
                        let mut out_dev = dev
                            .stream
                            .alloc_zeros::<f32>(outer * len)
                            .map_err(|e| TensorError::Message(e.to_string()))?;
                        if let Some(()) = super::ops::block_copy_device(
                            in_dev,
                            &mut out_dev,
                            outer,
                            trailing,
                            in_stride,
                            out_stride,
                            in_offset,
                            out_offset,
                        ) {
                            return Self::from_device_slice(out_dev, out_shape);
                        }
                    }
                }
            }
        }
        let host = self.host_cow()?;
        let in_strides = strides(&self.shape);
        let out_strides = strides(&out_shape);
        let mut out = vec![0.0; numel(&out_shape)];
        for out_idx in 0..out.len() {
            let mut coord = unravel(out_idx, &out_strides);
            coord[dim] += start;
            out[out_idx] = host[ravel(&coord, &in_strides)];
        }
        let mut t = Self::host_only(out, out_shape);
        let _ = t.ensure_device();
        Ok(t)
    }

    pub fn squeeze(&self, dim: usize) -> Result<CudaTensor> {
        if self.dim(dim)? != 1 {
            return Err(TensorError::Message(format!(
                "squeeze expected size 1 at {dim}, got {}",
                self.dim(dim)?
            )));
        }
        let mut shape = self.shape.clone();
        shape.remove(dim);
        #[cfg(feature = "cuda")]
        if self.device_fresh {
            return Ok(Self {
                data: self.data.clone(),
                shape,
                device: self.device.clone(),
                device_fresh: true,
            });
        }
        Ok(Self::host_only(self.data.clone(), shape))
    }

    pub fn unsqueeze(&self, dim: usize) -> Result<CudaTensor> {
        if dim > self.rank() {
            return Err(TensorError::Message("unsqueeze out of range".into()));
        }
        let mut shape = self.shape.clone();
        shape.insert(dim, 1);
        #[cfg(feature = "cuda")]
        if self.device_fresh {
            return Ok(Self {
                data: self.data.clone(),
                shape,
                device: self.device.clone(),
                device_fresh: true,
            });
        }
        Ok(Self::host_only(self.data.clone(), shape))
    }

    pub fn cat(tensors: &[&CudaTensor], dim: usize) -> Result<CudaTensor> {
        if tensors.is_empty() {
            return Err(TensorError::Message("cat empty".into()));
        }
        let rank = tensors[0].rank();
        let mut out_shape = tensors[0].shape.clone();
        let mut cat_len = 0usize;
        for t in tensors {
            if t.rank() != rank {
                return Err(TensorError::Message("cat rank mismatch".into()));
            }
            for (i, (&a, &b)) in out_shape.iter().zip(t.shape.iter()).enumerate() {
                if i != dim && a != b {
                    return Err(TensorError::Message("cat shape mismatch".into()));
                }
            }
            cat_len += t.shape[dim];
        }
        out_shape[dim] = cat_len;
        // Device path: cat on trailing (last) dim when every input is device-fresh.
        // Gather raw `*const u64` device pointers and shapes; run the gather kernel.
        if dim == rank - 1 {
            let outer: usize = tensors[0].shape[..rank - 1].iter().product();
            let inner = out_shape[rank - 1];
            if outer == 0 || inner == 0 {
                let mut t = Self::host_only(vec![], out_shape);
                let _ = t.ensure_device();
                return Ok(t);
            }
            #[cfg(feature = "cuda")]
            {
                if let Some(dev) = super::device::global_device() {
                    use cudarc::driver::DevicePtr;
                    let mut owned: Vec<cudarc::driver::CudaSlice<f32>> = Vec::with_capacity(tensors.len());
                    let mut ok = true;
                    for t in tensors {
                        let mut tt = (*t).clone();
                        tt.ensure_device().ok();
                        if !tt.is_device_fresh() {
                            ok = false;
                            break;
                        }
                        match tt.device_slice() {
                            Some(d) => owned.push(d.clone()),
                            None => {
                                ok = false;
                                break;
                            }
                        }
                    }
                    if ok {
                        if let Ok(mut out_dev) =
                            dev.stream.alloc_zeros::<f32>(outer * inner)
                        {
                            let mut offset = 0usize;
                            let mut all_ok = true;
                            for (i, t) in tensors.iter().enumerate() {
                                let in_len = t.shape[dim];
                                let in_stride = in_len;
                                let out_stride = inner;
                                if super::ops::block_copy_device(
                                    &owned[i],
                                    &mut out_dev,
                                    outer,
                                    in_len,
                                    in_stride,
                                    out_stride,
                                    0,
                                    offset,
                                )
                                .is_none()
                                {
                                    all_ok = false;
                                    break;
                                }
                                offset += in_len;
                            }
                            if all_ok {
                                return Self::from_device_slice(out_dev, out_shape);
                            }
                        }
                    }
                }
            }
        }
        let mut out = vec![0.0; numel(&out_shape)];
        let out_strides = strides(&out_shape);
        let mut offset = 0usize;
        for t in tensors {
            let host = t.host_cow()?;
            let in_strides = strides(&t.shape);
            for in_idx in 0..host.len() {
                let mut coord = unravel(in_idx, &in_strides);
                coord[dim] += offset;
                out[ravel(&coord, &out_strides)] = host[in_idx];
            }
            offset += t.shape[dim];
        }
        let mut t = Self::host_only(out, out_shape);
        let _ = t.ensure_device();
        Ok(t)
    }

    pub fn pad_zeros(&self, dim: usize, left: usize, right: usize) -> Result<CudaTensor> {
        let mut out_shape = self.shape.clone();
        out_shape[dim] += left + right;
        // Device path only when padding the trailing (last) dim.
        if dim == self.rank() - 1 {
            let outer: usize = self.shape[..self.rank() - 1].iter().product();
            let inner = self.shape[self.rank() - 1];
            if outer == 0 || inner == 0 {
                let mut t = Self::host_only(vec![0.0; numel(&out_shape)], out_shape);
                let _ = t.ensure_device();
                return Ok(t);
            }
            let in_stride = inner;
            let out_stride = out_shape[self.rank() - 1];
            let in_offset = 0;
            let out_offset = left;
            let _ = (in_stride, out_stride, in_offset, out_offset);
            #[cfg(feature = "cuda")]
            {
            let mut src = self.clone();
            src.ensure_device().ok();
            if let (Some(dev), Some(in_dev)) =
                (super::device::global_device(), src.device_slice())
            {
                if src.is_device_fresh() {
                        let mut out_dev = dev
                            .stream
                            .alloc_zeros::<f32>(outer * out_stride)
                            .map_err(|e| TensorError::Message(e.to_string()))?;
                        if let Some(()) = super::ops::block_copy_device(
                            in_dev,
                            &mut out_dev,
                            outer,
                            inner,
                            in_stride,
                            out_stride,
                            in_offset,
                            out_offset,
                        ) {
                            return Self::from_device_slice(out_dev, out_shape);
                        }
                    }
                }
            }
        }
        let host = self.host_cow()?;
        let mut out = vec![0.0; numel(&out_shape)];
        let in_strides = strides(&self.shape);
        let out_strides = strides(&out_shape);
        for in_idx in 0..host.len() {
            let mut coord = unravel(in_idx, &in_strides);
            coord[dim] += left;
            out[ravel(&coord, &out_strides)] = host[in_idx];
        }
        let mut t = Self::host_only(out, out_shape);
        let _ = t.ensure_device();
        Ok(t)
    }

    pub fn add(&self, other: &CudaTensor) -> Result<CudaTensor> {
        #[cfg(feature = "cuda")]
        if self.shape == other.shape {
            if super::resident::residency_enabled() {
                if let (Some(a), Some(b)) = (self.device_slice(), other.device_slice()) {
                    if self.device_fresh && other.device_fresh {
                        if let Some(out) =
                            super::ops::elem_binary_device(a, b, super::ops::ElemBinary::Add)
                        {
                            return Self::from_device_slice(out, self.shape.clone());
                        }
                    }
                }
            }
            let a = self.host_cow()?;
            let b = other.host_cow()?;
            if let Some(data) =
                super::ops::try_elem_binary(&a, &b, super::ops::ElemBinary::Add)
            {
                let mut t = Self::host_only(data, self.shape.clone());
                let _ = t.ensure_device();
                return Ok(t);
            }
        }
        broadcast_bin(self, other, |a, b| a + b)
    }

    pub fn sub(&self, other: &CudaTensor) -> Result<CudaTensor> {
        #[cfg(feature = "cuda")]
        if self.shape == other.shape {
            if super::resident::residency_enabled() {
                if let (Some(a), Some(b)) = (self.device_slice(), other.device_slice()) {
                    if self.device_fresh && other.device_fresh {
                        if let Some(out) =
                            super::ops::elem_binary_device(a, b, super::ops::ElemBinary::Sub)
                        {
                            return Self::from_device_slice(out, self.shape.clone());
                        }
                    }
                }
            }
            let a = self.host_cow()?;
            let b = other.host_cow()?;
            if let Some(data) =
                super::ops::try_elem_binary(&a, &b, super::ops::ElemBinary::Sub)
            {
                let mut t = Self::host_only(data, self.shape.clone());
                let _ = t.ensure_device();
                return Ok(t);
            }
        }
        broadcast_bin(self, other, |a, b| a - b)
    }

    pub fn mul(&self, other: &CudaTensor) -> Result<CudaTensor> {
        #[cfg(feature = "cuda")]
        if self.shape == other.shape {
            if super::resident::residency_enabled() {
                if let (Some(a), Some(b)) = (self.device_slice(), other.device_slice()) {
                    if self.device_fresh && other.device_fresh {
                        if let Some(out) =
                            super::ops::elem_binary_device(a, b, super::ops::ElemBinary::Mul)
                        {
                            return Self::from_device_slice(out, self.shape.clone());
                        }
                    }
                }
            }
            let a = self.host_cow()?;
            let b = other.host_cow()?;
            if let Some(data) =
                super::ops::try_elem_binary(&a, &b, super::ops::ElemBinary::Mul)
            {
                let mut t = Self::host_only(data, self.shape.clone());
                let _ = t.ensure_device();
                return Ok(t);
            }
        }
        broadcast_bin(self, other, |a, b| a * b)
    }

    pub fn div(&self, other: &CudaTensor) -> Result<CudaTensor> {
        broadcast_bin(self, other, |a, b| a / b)
    }

    pub fn add_scalar(&self, s: f32) -> CudaTensor {
        #[cfg(feature = "cuda")]
        {
            if super::resident::residency_enabled() && self.device_fresh {
                if let Some(a) = self.device_slice() {
                    if let Some(out) = super::ops::add_scalar_device(a, s) {
                        if let Ok(t) = Self::from_device_slice(out, self.shape.clone()) {
                            return t;
                        }
                    }
                }
            }
            if let Ok(host) = self.host_cow() {
                if let Some(data) = super::ops::try_add_scalar(&host, s) {
                    let mut t = Self::host_only(data, self.shape.clone());
                    let _ = t.ensure_device();
                    return t;
                }
            }
        }
        let host = self.host_cow().unwrap_or(Cow::Borrowed(self.data.as_slice()));
        let mut t = Self::host_only(host.iter().map(|x| x + s).collect(), self.shape.clone());
        let _ = t.ensure_device();
        t
    }

    pub fn mul_scalar(&self, s: f32) -> CudaTensor {
        #[cfg(feature = "cuda")]
        {
            if super::resident::residency_enabled() && self.device_fresh {
                if let Some(a) = self.device_slice() {
                    if let Some(out) = super::ops::mul_scalar_device(a, s) {
                        if let Ok(t) = Self::from_device_slice(out, self.shape.clone()) {
                            return t;
                        }
                    }
                }
            }
            if let Ok(host) = self.host_cow() {
                if let Some(data) = super::ops::try_mul_scalar(&host, s) {
                    let mut t = Self::host_only(data, self.shape.clone());
                    let _ = t.ensure_device();
                    return t;
                }
            }
        }
        let host = self.host_cow().unwrap_or(Cow::Borrowed(self.data.as_slice()));
        let mut t = Self::host_only(host.iter().map(|x| x * s).collect(), self.shape.clone());
        let _ = t.ensure_device();
        t
    }

    pub fn clamp(&self, min: f32, max: f32) -> CudaTensor {
        #[cfg(feature = "cuda")]
        {
            if super::resident::residency_enabled() && self.device_fresh {
                if let Some(a) = self.device_slice() {
                    if let Some(out) = super::ops::clamp_device(a, min, max) {
                        if let Ok(t) = Self::from_device_slice(out, self.shape.clone()) {
                            return t;
                        }
                    }
                }
            }
            if let Ok(host) = self.host_cow() {
                if let Some(data) = super::ops::try_clamp(&host, min, max) {
                    let mut t = Self::host_only(data, self.shape.clone());
                    let _ = t.ensure_device();
                    return t;
                }
            }
        }
        let host = self.host_cow().unwrap_or(Cow::Borrowed(self.data.as_slice()));
        let mut t = Self::host_only(
            host.iter().map(|x| x.clamp(min, max)).collect(),
            self.shape.clone(),
        );
        let _ = t.ensure_device();
        t
    }

    pub fn sqrt(&self) -> CudaTensor {
        let host = self.host_cow().unwrap_or(Cow::Borrowed(self.data.as_slice()));
        let mut t = Self::host_only(host.iter().map(|x| x.sqrt()).collect(), self.shape.clone());
        let _ = t.ensure_device();
        t
    }

    pub fn sqr(&self) -> CudaTensor {
        let host = self.host_cow().unwrap_or(Cow::Borrowed(self.data.as_slice()));
        let mut t = Self::host_only(host.iter().map(|x| x * x).collect(), self.shape.clone());
        let _ = t.ensure_device();
        t
    }

    pub fn mean_keepdim(&self, dim: isize) -> Result<CudaTensor> {
        let axis = self.axis(dim)?;
        let mut out_shape = self.shape.clone();
        out_shape[axis] = 1;
        let mut out = vec![0.0; numel(&out_shape)];
        let counts = vec![0usize; out.len()];
        let mut counts = counts;
        let in_strides = strides(&self.shape);
        let out_strides = strides(&out_shape);
        let host = self.host_cow()?;
        for in_idx in 0..host.len() {
            let mut coord = unravel(in_idx, &in_strides);
            coord[axis] = 0;
            let oi = ravel(&coord, &out_strides);
            out[oi] += host[in_idx];
            counts[oi] += 1;
        }
        for (o, c) in out.iter_mut().zip(counts) {
            *o /= c as f32;
        }
        Ok(Self::host_only(out, out_shape))
    }

    pub fn matmul(&self, other: &CudaTensor) -> Result<CudaTensor> {
        // Support (…, M, K) @ (…, K, N) with matching batch dims.
        if self.rank() < 2 || other.rank() < 2 {
            return Err(TensorError::Message("matmul needs rank >= 2".into()));
        }
        let m = self.shape[self.rank() - 2];
        let k = self.shape[self.rank() - 1];
        let k2 = other.shape[other.rank() - 2];
        let n = other.shape[other.rank() - 1];
        if k != k2 {
            return Err(TensorError::Message(format!("matmul inner {k} vs {k2}")));
        }
        let a_batch = &self.shape[..self.rank() - 2];
        let b_batch = &other.shape[..other.rank() - 2];
        let batch_shape = broadcast_shapes(a_batch, b_batch)?;
        let batch = numel(&batch_shape);

        // Fast path: device-resident GEMM (single or batched tiles).
        #[cfg(feature = "cuda")]
        if super::resident::residency_enabled() {
            if let (Some(a), Some(b)) = (self.device_slice(), other.device_slice()) {
                if self.device_fresh && other.device_fresh {
                    if let Some(dev) = super::device::global_device() {
                        if a_batch == b_batch && a.len() == batch * m * k && b.len() == batch * k * n
                        {
                            let mut c = dev
                                .stream
                                .alloc_zeros::<f32>(batch * m * n)
                                .map_err(|e| TensorError::Message(e.to_string()))?;
                            if batch == 1 {
                                super::device::matmul_2d_f32_device(a, b, &mut c, m, k, n)
                                    .map_err(|e| TensorError::Message(e.to_string()))?;
                            } else {
                                super::device::matmul_2d_strided_batched(
                                    a, b, &mut c, batch, m, k, n,
                                )
                                .map_err(|e| TensorError::Message(e.to_string()))?;
                            }
                            let mut out_shape = batch_shape.clone();
                            out_shape.push(m);
                            out_shape.push(n);
                            return Self::from_device_slice(c, out_shape);
                        }
                    }
                }
            }
        }

        let a_host = self.host_cow()?;
        let b_host = other.host_cow()?;
        let mut out = vec![0.0; batch * m * n];
        for bi in 0..batch {
            let a_bi = batch_index(bi, &batch_shape, a_batch);
            let b_bi = batch_index(bi, &batch_shape, b_batch);
            let a_off = a_bi * m * k;
            let b_off = b_bi * k * n;
            let o_off = bi * m * n;
            let a_slice = &a_host[a_off..a_off + m * k];
            let b_slice = &b_host[b_off..b_off + k * n];
            let o_slice = &mut out[o_off..o_off + m * n];
            if !matmul_slice(a_slice, b_slice, o_slice, m, k, n) {
                for i in 0..m {
                    for j in 0..n {
                        let mut acc = 0.0;
                        for t in 0..k {
                            acc += a_slice[i * k + t] * b_slice[t * n + j];
                        }
                        o_slice[i * n + j] = acc;
                    }
                }
            }
        }
        let mut out_shape = batch_shape;
        out_shape.push(m);
        out_shape.push(n);
        let mut t = Self::host_only(out, out_shape);
        let _ = t.ensure_device();
        Ok(t)
    }

    pub fn softmax(&self, dim: isize) -> Result<CudaTensor> {
        let axis = self.axis(dim)?;
        #[cfg(feature = "cuda")]
        if axis + 1 == self.rank() {
            let width = self.shape[axis];
            if super::resident::residency_enabled() && self.device_fresh {
                if let Some(a) = self.device_slice() {
                    if let Some(out) = super::ops::softmax_last_device(a, width) {
                        return Self::from_device_slice(out, self.shape.clone());
                    }
                }
            }
            let host = self.host_cow()?;
            if let Some(data) = super::ops::try_softmax_last(&host, width) {
                let mut t = Self::host_only(data, self.shape.clone());
                let _ = t.ensure_device();
                return Ok(t);
            }
        }
        let axis_len = self.shape[axis];
        let out_shape = self.shape.clone();
        let host = self.host_cow()?;
        let mut result = vec![0.0; host.len()];
        let in_strides = strides(&self.shape);
        let mut seen = vec![false; host.len()];
        for idx in 0..host.len() {
            if seen[idx] {
                continue;
            }
            let mut coord = unravel(idx, &in_strides);
            let mut vals = Vec::with_capacity(axis_len);
            let mut indices = Vec::with_capacity(axis_len);
            for a in 0..axis_len {
                coord[axis] = a;
                let i = ravel(&coord, &in_strides);
                vals.push(host[i]);
                indices.push(i);
                seen[i] = true;
            }
            let max = vals.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f32> = vals.iter().map(|v| (v - max).exp()).collect();
            let sum: f32 = exps.iter().sum();
            for (i, e) in indices.into_iter().zip(exps) {
                result[i] = e / sum;
            }
        }
        let mut t = Self::host_only(result, out_shape);
        let _ = t.ensure_device();
        Ok(t)
    }

    pub fn silu(&self) -> CudaTensor {
        #[cfg(feature = "cuda")]
        {
            if super::resident::residency_enabled() {
                if let Some(a) = self.device_slice() {
                    if self.device_fresh {
                        if let Some(out) = super::ops::unary_device(a, super::ops::ElemUnary::Silu)
                        {
                            if let Ok(t) = Self::from_device_slice(out, self.shape.clone()) {
                                return t;
                            }
                        }
                    }
                }
            }
            if let Ok(host) = self.host_cow() {
                if let Some(data) = super::ops::try_unary(&host, super::ops::ElemUnary::Silu) {
                    return Self::host_only(data, self.shape.clone());
                }
            }
        }
        let host = self.host_cow().unwrap_or(Cow::Borrowed(self.data.as_slice()));
        Self::host_only(
            host.iter().map(|&x| x / (1.0 + (-x).exp())).collect(),
            self.shape.clone(),
        )
    }

    pub fn gelu_tanh(&self) -> CudaTensor {
        #[cfg(feature = "cuda")]
        {
            if super::resident::residency_enabled() {
                if let Some(a) = self.device_slice() {
                    if self.device_fresh {
                        if let Some(out) =
                            super::ops::unary_device(a, super::ops::ElemUnary::GeluTanh)
                        {
                            if let Ok(t) = Self::from_device_slice(out, self.shape.clone()) {
                                return t;
                            }
                        }
                    }
                }
            }
            if let Ok(host) = self.host_cow() {
                if let Some(data) = super::ops::try_unary(&host, super::ops::ElemUnary::GeluTanh)
                {
                    return Self::host_only(data, self.shape.clone());
                }
            }
        }
        let c = (2.0 / std::f32::consts::PI).sqrt();
        let host = self.host_cow().unwrap_or(Cow::Borrowed(self.data.as_slice()));
        Self::host_only(
            host.iter()
                .map(|&x| {
                    let inner = c * (x + 0.044715 * x * x * x);
                    0.5 * x * (1.0 + inner.tanh())
                })
                .collect(),
            self.shape.clone(),
        )
    }

    pub fn rms_norm(&self, weight: &CudaTensor, eps: f32) -> Result<CudaTensor> {
        // Normalize over last dim.
        let axis = self.rank() - 1;
        let axis_len = self.shape[axis];
        let w_len = weight.shape.last().copied().unwrap_or(weight.data.len());
        if w_len != axis_len {
            return Err(TensorError::Message("rms_norm weight size".into()));
        }
        #[cfg(feature = "cuda")]
        if super::resident::residency_enabled() && self.device_fresh {
            let mut w = weight.clone();
            w.ensure_device()?;
            if let (Some(a), Some(wv)) = (self.device_slice(), w.device_slice()) {
                if let Some(out) = super::ops::rms_norm_last_device(a, wv, eps) {
                    return Self::from_device_slice(out, self.shape.clone());
                }
            }
        }
        let host = self.host_cow()?;
        let w_host = weight.host_cow()?;
        #[cfg(feature = "cuda")]
        if let Some(data) = super::ops::try_rms_norm_last(&host, &w_host, eps) {
            let mut t = Self::host_only(data, self.shape.clone());
            let _ = t.ensure_device();
            return Ok(t);
        }
        let mut out = vec![0.0; host.len()];
        let outer = numel(&self.shape) / axis_len;
        for o in 0..outer {
            let base = o * axis_len;
            let mut mean_sq = 0.0;
            for a in 0..axis_len {
                let v = host[base + a];
                mean_sq += v * v;
            }
            mean_sq /= axis_len as f32;
            let inv = 1.0 / (mean_sq + eps).sqrt();
            for a in 0..axis_len {
                out[base + a] = host[base + a] * inv * w_host[a];
            }
        }
        let mut t = Self::host_only(out, self.shape.clone());
        let _ = t.ensure_device();
        Ok(t)
    }

    pub fn layer_norm(
        &self,
        eps: f32,
        weight: Option<&CudaTensor>,
        bias: Option<&CudaTensor>,
    ) -> Result<CudaTensor> {
        let axis = self.rank() - 1;
        let axis_len = self.shape[axis];
        #[cfg(feature = "cuda")]
        if super::resident::residency_enabled() && self.device_fresh {
            // Device path only when weight/bias (if given) are both present
            // and device-uploadable; falls through to host on any mismatch.
            let mut w_dev = None;
            let mut w_hold;
            let mut b_dev = None;
            let mut b_hold;
            if let (Some(w), Some(b)) = (weight, bias) {
                w_hold = w.clone();
                w_hold.ensure_device()?;
                b_hold = b.clone();
                b_hold.ensure_device()?;
                w_dev = w_hold.device_slice();
                b_dev = b_hold.device_slice();
            }
            if let Some(a) = self.device_slice() {
                if let Some(out) =
                    super::ops::layer_norm_last_device(a, w_dev, b_dev, axis_len, eps)
                {
                    record_device_hit("layer_norm");
                    return Self::from_device_slice(out, self.shape.clone());
                }
            }
        }
        let host = self.host_cow()?;
        let w_host = weight.map(|w| w.host_cow()).transpose()?;
        let b_host = bias.map(|b| b.host_cow()).transpose()?;
        #[cfg(feature = "cuda")]
        if let Some(data) = super::ops::try_layer_norm_last(
            &host,
            w_host.as_deref(),
            b_host.as_deref(),
            axis_len,
            eps,
        ) {
            record_device_hit("layer_norm");
            let mut t = Self::host_only(data, self.shape.clone());
            let _ = t.ensure_device();
            return Ok(t);
        }
        strict_device_check(
            "layer_norm",
            format_args!("shape={:?} axis_len={axis_len}", self.shape),
        )?;
        let mut out = vec![0.0; host.len()];
        let outer = numel(&self.shape) / axis_len;
        for o in 0..outer {
            let base = o * axis_len;
            let mut mean = 0.0;
            for a in 0..axis_len {
                mean += host[base + a];
            }
            mean /= axis_len as f32;
            let mut var = 0.0;
            for a in 0..axis_len {
                let d = host[base + a] - mean;
                var += d * d;
            }
            var /= axis_len as f32;
            let inv = 1.0 / (var + eps).sqrt();
            for a in 0..axis_len {
                let mut y = (host[base + a] - mean) * inv;
                if let Some(ref w) = w_host {
                    y *= w[a];
                }
                if let Some(ref b) = b_host {
                    y += b[a];
                }
                out[base + a] = y;
            }
        }
        let mut t = Self::host_only(out, self.shape.clone());
        let _ = t.ensure_device();
        Ok(t)
    }

    /// Fused AdaLN modulate: `out[b,l,d] = self[b,l,d] * (1 + scale[b,0,d]) +
    /// shift[b,0,d]`. `self` is `[B, L, dim]`; `scale`/`shift` are `[B, 1,
    /// dim]` (broadcast over `L`). Replaces
    /// `self.mul(&scale.add_scalar(1.0))?.add(shift)?`, which — because the
    /// shapes differ — used to fall through to the fully generic
    /// `broadcast_bin` host path (per-element `Vec` coordinate allocation,
    /// forced device↔host sync) on every DiT block. This has a single
    /// device kernel launch on the resident path, and a direct-indexed
    /// (no per-element heap allocation) host fallback.
    pub fn modulate(&self, scale: &CudaTensor, shift: &CudaTensor) -> Result<CudaTensor> {
        let (batch, seq, dim) = self.modulate_broadcast_shape(scale, shift, "modulate")?;
        #[cfg(feature = "cuda")]
        if super::resident::residency_enabled() && self.device_fresh {
            let mut sc = scale.clone();
            let mut sh = shift.clone();
            sc.ensure_device()?;
            sh.ensure_device()?;
            if let (Some(x), Some(s), Some(h)) =
                (self.device_slice(), sc.device_slice(), sh.device_slice())
            {
                if let Some(out) =
                    super::ops::modulate_scale_shift_device(x, s, h, batch, seq, dim)
                {
                    record_device_hit("modulate");
                    return Self::from_device_slice(out, self.shape.clone());
                }
            }
        }
        let x = self.host_cow()?;
        let sc = scale.host_cow()?;
        let sh = shift.host_cow()?;
        #[cfg(feature = "cuda")]
        if let Some(data) = super::ops::try_modulate_scale_shift(&x, &sc, &sh, batch, seq, dim) {
            record_device_hit("modulate");
            let mut t = Self::host_only(data, self.shape.clone());
            let _ = t.ensure_device();
            return Ok(t);
        }
        strict_device_check("modulate", format_args!("batch={batch} seq={seq} dim={dim}"))?;
        let mut out = vec![0.0; x.len()];
        for b in 0..batch {
            for l in 0..seq {
                let row = (b * seq + l) * dim;
                let bd = b * dim;
                for d in 0..dim {
                    out[row + d] = x[row + d] * (1.0 + sc[bd + d]) + sh[bd + d];
                }
            }
        }
        let mut t = Self::host_only(out, self.shape.clone());
        let _ = t.ensure_device();
        Ok(t)
    }

    /// Fused AdaLN gated multiply: `out[b,l,d] = self[b,l,d] * gate[b,0,d]`.
    /// Same broadcast shape as [`Self::modulate`]; replaces
    /// `self.mul(gate)` for the AdaLN gated-residual add (also a
    /// `broadcast_bin` host round trip before this).
    pub fn gate_mul(&self, gate: &CudaTensor) -> Result<CudaTensor> {
        // Reuse the same shape validation with shift==gate (unused shape-wise).
        let (batch, seq, dim) = self.modulate_broadcast_shape(gate, gate, "gate_mul")?;
        #[cfg(feature = "cuda")]
        if super::resident::residency_enabled() && self.device_fresh {
            let mut g = gate.clone();
            g.ensure_device()?;
            if let (Some(x), Some(gd)) = (self.device_slice(), g.device_slice()) {
                if let Some(out) = super::ops::broadcast_mul_last_device(x, gd, batch, seq, dim) {
                    record_device_hit("gate_mul");
                    return Self::from_device_slice(out, self.shape.clone());
                }
            }
        }
        let x = self.host_cow()?;
        let g = gate.host_cow()?;
        #[cfg(feature = "cuda")]
        if let Some(data) = super::ops::try_broadcast_mul_last(&x, &g, batch, seq, dim) {
            record_device_hit("gate_mul");
            let mut t = Self::host_only(data, self.shape.clone());
            let _ = t.ensure_device();
            return Ok(t);
        }
        strict_device_check("gate_mul", format_args!("batch={batch} seq={seq} dim={dim}"))?;
        let mut out = vec![0.0; x.len()];
        for b in 0..batch {
            for l in 0..seq {
                let row = (b * seq + l) * dim;
                let bd = b * dim;
                for d in 0..dim {
                    out[row + d] = x[row + d] * g[bd + d];
                }
            }
        }
        let mut t = Self::host_only(out, self.shape.clone());
        let _ = t.ensure_device();
        Ok(t)
    }

    /// Shared shape check for [`Self::modulate`]/[`Self::gate_mul`]: `self`
    /// is `[batch, seq, dim]`, `a`/`b` are `[batch, 1, dim]`.
    fn modulate_broadcast_shape(
        &self,
        a: &CudaTensor,
        b: &CudaTensor,
        who: &str,
    ) -> Result<(usize, usize, usize)> {
        if self.rank() != 3 || a.rank() != 3 || b.rank() != 3 {
            return Err(TensorError::Message(format!("{who}: expected rank-3 tensors")));
        }
        let (batch, seq, dim) = (self.shape[0], self.shape[1], self.shape[2]);
        if a.shape != [batch, 1, dim] || b.shape != [batch, 1, dim] {
            return Err(TensorError::Message(format!(
                "{who}: shape mismatch, self={:?} a={:?} b={:?} (expected a/b=[{batch},1,{dim}])",
                self.shape, a.shape, b.shape
            )));
        }
        Ok((batch, seq, dim))
    }

    /// NCHW conv2d with square/rectangular kernel, padding, stride, dilation=1, groups=1.
    pub fn conv2d(
        &self,
        weight: &CudaTensor,
        bias: Option<&CudaTensor>,
        padding: usize,
        stride: usize,
    ) -> Result<CudaTensor> {
        // self: [N, C_in, H, W], weight: [C_out, C_in, Kh, Kw]
        if self.rank() != 4 || weight.rank() != 4 {
            return Err(TensorError::Message("conv2d expects NCHW + OIHW".into()));
        }
        let (n, c_in, h, w) = (self.shape[0], self.shape[1], self.shape[2], self.shape[3]);
        let (c_out, ic, kh, kw) = (weight.shape[0], weight.shape[1], weight.shape[2], weight.shape[3]);
        if c_in != ic {
            return Err(TensorError::Message("conv2d channel mismatch".into()));
        }
        let stride = stride.max(1);
        let out_h = (h + 2 * padding - kh) / stride + 1;
        let out_w = (w + 2 * padding - kw) / stride + 1;
        // Host views: a device-fresh tensor's `data` mirror is stale (zeros)
        // until downloaded, so never read `.data` directly here.
        let x_host = self.host_cow()?;
        let w_host = weight.host_cow()?;
        let b_host = match bias {
            Some(b) => Some(b.host_cow()?),
            None => None,
        };

        #[cfg(feature = "cuda")]
        if let Some(mut out) = super::ops::try_conv2d(
            &x_host,
            &w_host,
            n,
            c_in,
            h,
            w,
            c_out,
            kh,
            kw,
            padding,
            stride,
        ) {
            if let Some(b) = &b_host {
                for ni in 0..n {
                    for oc in 0..c_out {
                        let add = b[oc];
                        for oh in 0..out_h {
                            for ow in 0..out_w {
                                out[((ni * c_out + oc) * out_h + oh) * out_w + ow] += add;
                            }
                        }
                    }
                }
            }
            return Ok(Self::host_only(out, vec![n, c_out, out_h, out_w]));
        }

        let mut out = vec![0.0; n * c_out * out_h * out_w];
        for ni in 0..n {
            for oc in 0..c_out {
                for oh in 0..out_h {
                    for ow in 0..out_w {
                        let mut acc = 0.0;
                        for ic in 0..c_in {
                            for kh_i in 0..kh {
                                for kw_i in 0..kw {
                                    let ih = oh * stride + kh_i;
                                    let iw = ow * stride + kw_i;
                                    if ih < padding || iw < padding {
                                        continue;
                                    }
                                    let ih = ih - padding;
                                    let iw = iw - padding;
                                    if ih >= h || iw >= w {
                                        continue;
                                    }
                                    let xv = x_host[((ni * c_in + ic) * h + ih) * w + iw];
                                    let wv = w_host[((oc * c_in + ic) * kh + kh_i) * kw + kw_i];
                                    acc += xv * wv;
                                }
                            }
                        }
                        if let Some(b) = &b_host {
                            acc += b[oc];
                        }
                        out[((ni * c_out + oc) * out_h + oh) * out_w + ow] = acc;
                    }
                }
            }
        }
        Ok(Self::host_only(out, vec![n, c_out, out_h, out_w]))
    }

    pub fn upsample_nearest2d(&self, out_h: usize, out_w: usize) -> Result<CudaTensor> {
        if self.rank() != 4 {
            return Err(TensorError::Message("upsample_nearest2d NCHW".into()));
        }
        let (n, c, h, w) = (self.shape[0], self.shape[1], self.shape[2], self.shape[3]);
        let host = self.host_cow()?;
        let mut out = vec![0.0; n * c * out_h * out_w];
        for ni in 0..n {
            for ci in 0..c {
                for oh in 0..out_h {
                    for ow in 0..out_w {
                        let ih = oh * h / out_h;
                        let iw = ow * w / out_w;
                        out[((ni * c + ci) * out_h + oh) * out_w + ow] =
                            host[((ni * c + ci) * h + ih) * w + iw];
                    }
                }
            }
        }
        Ok(Self::host_only(out, vec![n, c, out_h, out_w]))
    }

    pub fn chunk(&self, chunks: usize, dim: usize) -> Result<Vec<CudaTensor>> {
        let d = self.dim(dim)?;
        if d % chunks != 0 {
            return Err(TensorError::Message("chunk size not divisible".into()));
        }
        let size = d / chunks;
        let mut out = Vec::with_capacity(chunks);
        for i in 0..chunks {
            out.push(self.narrow(dim, i * size, size)?);
        }
        Ok(out)
    }

    pub fn flatten_from(&self, dim: usize) -> Result<CudaTensor> {
        let mut shape = self.shape[..dim].to_vec();
        shape.push(numel(&self.shape[dim..]));
        self.reshape(shape)
    }

    pub fn index_select_rows(&self, indices: &[usize]) -> Result<CudaTensor> {
        // self: [V, D], gather rows
        if self.rank() != 2 {
            return Err(TensorError::Message("index_select_rows expects 2D".into()));
        }
        let d = self.shape[1];
        let host = self.host_cow()?;
        let mut data = Vec::with_capacity(indices.len() * d);
        for &i in indices {
            if i >= self.shape[0] {
                return Err(TensorError::Message("index OOB".into()));
            }
            data.extend_from_slice(&host[i * d..(i + 1) * d]);
        }
        Ok(Self::host_only(data, vec![indices.len(), d]))
    }
}

fn strides(shape: &[usize]) -> Vec<usize> {
    let mut s = vec![1; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        s[i] = s[i + 1] * shape[i + 1];
    }
    s
}

fn unravel(mut idx: usize, strides: &[usize]) -> Vec<usize> {
    let mut coord = vec![0; strides.len()];
    for i in 0..strides.len() {
        coord[i] = idx / strides[i];
        idx %= strides[i];
    }
    coord
}

fn ravel(coord: &[usize], strides: &[usize]) -> usize {
    coord.iter().zip(strides).map(|(c, s)| c * s).sum()
}

fn broadcast_shapes(a: &[usize], b: &[usize]) -> Result<Vec<usize>> {
    let rank = a.len().max(b.len());
    let mut out = vec![1; rank];
    for i in 0..rank {
        let da = if i < rank - a.len() {
            1
        } else {
            a[i - (rank - a.len())]
        };
        let db = if i < rank - b.len() {
            1
        } else {
            b[i - (rank - b.len())]
        };
        if da == db || da == 1 || db == 1 {
            out[i] = da.max(db);
        } else {
            return Err(TensorError::Message(format!(
                "broadcast {a:?} vs {b:?}"
            )));
        }
    }
    Ok(out)
}

fn broadcast_bin(a: &CudaTensor, b: &CudaTensor, op: impl Fn(f32, f32) -> f32) -> Result<CudaTensor> {
    let shape = broadcast_shapes(&a.shape, &b.shape)?;
    let a_host = a.host_cow()?;
    let b_host = b.host_cow()?;
    let a_strides = strides(&a.shape);
    let b_strides = strides(&b.shape);
    let out_strides = strides(&shape);
    let mut data = vec![0.0; numel(&shape)];
    for i in 0..data.len() {
        let coord = unravel(i, &out_strides);
        let mut a_coord = vec![0usize; a.rank()];
        let mut b_coord = vec![0usize; b.rank()];
        for (c_i, &c) in coord.iter().enumerate() {
            let a_axis = c_i as isize - (shape.len() - a.rank()) as isize;
            let b_axis = c_i as isize - (shape.len() - b.rank()) as isize;
            if a_axis >= 0 {
                let aa = a_axis as usize;
                a_coord[aa] = if a.shape[aa] == 1 { 0 } else { c };
            }
            if b_axis >= 0 {
                let bb = b_axis as usize;
                b_coord[bb] = if b.shape[bb] == 1 { 0 } else { c };
            }
        }
        let av = if a.rank() == 0 {
            a_host[0]
        } else {
            a_host[ravel(&a_coord, &a_strides)]
        };
        let bv = if b.rank() == 0 {
            b_host[0]
        } else {
            b_host[ravel(&b_coord, &b_strides)]
        };
        data[i] = op(av, bv);
    }
    Ok({
        let mut t = CudaTensor::host_only(data, shape);
        let _ = t.ensure_device();
        t
    })
}

/// Prefer cuBLAS for a single `(m,k)@(k,n)` tile when a global CUDA context is set.
fn matmul_slice(a: &[f32], b: &[f32], out: &mut [f32], m: usize, k: usize, n: usize) -> bool {
    #[cfg(feature = "cuda")]
    {
        if let Ok(data) = super::device::matmul_2d_f32(a, b, m, k, n) {
            if data.len() == out.len() {
                out.copy_from_slice(&data);
                return true;
            }
        }
    }
    let _ = (a, b, out, m, k, n);
    false
}

fn batch_index(flat: usize, full: &[usize], subset: &[usize]) -> usize {
    if subset.is_empty() {
        return 0;
    }
    let full_strides = strides(full);
    let sub_strides = strides(subset);
    let coord = unravel(flat, &full_strides);
    let offset = full.len() - subset.len();
    let mut sub_coord = vec![0usize; subset.len()];
    for i in 0..subset.len() {
        sub_coord[i] = if subset[i] == 1 { 0 } else { coord[offset + i] };
    }
    ravel(&sub_coord, &sub_strides)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matmul_2d() {
        let a = CudaTensor::from_vec(vec![1., 2., 3., 4.], vec![2, 2]).unwrap();
        let b = CudaTensor::from_vec(vec![5., 6., 7., 8.], vec![2, 2]).unwrap();
        let c = a.matmul(&b).unwrap();
        assert_eq!(c.data, vec![19., 22., 43., 50.]);
    }

    /// Strict mode must be a true no-op with no live CUDA device — that's
    /// the default state on any non-GPU machine (this sandbox, plain `cargo
    /// test`, a Mac dev box) and it must never spuriously fail a host-only
    /// run just because `FASTVIDEO_STRICT_DEVICE=1` was set. This is the
    /// half of strict mode's contract that's actually testable without a
    /// GPU; the other half (it *does* fire when a device is live but an op
    /// falls back anyway) can only be exercised on real hardware — see
    /// `strict_device_check`'s doc comment for the intended workflow.
    #[test]
    fn strict_device_check_is_noop_without_a_live_device() {
        assert!(!super::super::device::has_live_device());
        let prev = std::env::var("FASTVIDEO_STRICT_DEVICE").ok();
        std::env::set_var("FASTVIDEO_STRICT_DEVICE", "1");
        STRICT_DEVICE_CACHE.reset();
        let result = strict_device_check("layer_norm", "test: no device present");
        assert!(
            result.is_ok(),
            "strict mode fired with no live device: {result:?}"
        );
        match prev {
            Some(v) => std::env::set_var("FASTVIDEO_STRICT_DEVICE", v),
            None => std::env::remove_var("FASTVIDEO_STRICT_DEVICE"),
        }
        STRICT_DEVICE_CACHE.reset();
    }

    #[test]
    fn strict_device_check_off_by_default() {
        let prev = std::env::var("FASTVIDEO_STRICT_DEVICE").ok();
        std::env::remove_var("FASTVIDEO_STRICT_DEVICE");
        STRICT_DEVICE_CACHE.reset();
        assert!(strict_device_check("modulate", "default state").is_ok());
        match prev {
            Some(v) => std::env::set_var("FASTVIDEO_STRICT_DEVICE", v),
            None => {}
        }
        STRICT_DEVICE_CACHE.reset();
    }

    #[test]
    fn device_path_stats_record_and_reset() {
        reset_device_path_stats();
        record_device_hit("layer_norm");
        record_device_hit("layer_norm");
        record_host_fallback("layer_norm");
        let stats = device_path_stats();
        let (name, hits, misses) = stats
            .iter()
            .find(|(n, _, _)| *n == "layer_norm")
            .expect("layer_norm tracked");
        assert_eq!(*name, "layer_norm");
        assert_eq!(*hits, 2);
        assert_eq!(*misses, 1);
        reset_device_path_stats();
        let stats = device_path_stats();
        assert!(stats.iter().all(|(_, h, m)| *h == 0 && *m == 0));
    }

    /// On this host-only build (no `cuda` feature), `layer_norm`/`modulate`/
    /// `gate_mul`'s device branches don't exist at all (compiled out), so
    /// there's no "device was live but we fell back anyway" to record —
    /// counters correctly stay at zero rather than reporting a misleading
    /// fallback for a device that was never tried. The thing this test
    /// actually proves: running these ops with `FASTVIDEO_STRICT_DEVICE=1`
    /// set never errors here, because `strict_device_check` correctly
    /// recognizes there's no live device to be strict about — see
    /// `strict_device_check_is_noop_without_a_live_device` for that
    /// assertion in isolation, and `strict_device_check`'s doc comment for
    /// why the "does it actually fire" half needs a real GPU to test.
    #[test]
    fn hot_ops_run_without_strict_device_tripping() {
        let prev = std::env::var("FASTVIDEO_STRICT_DEVICE").ok();
        std::env::set_var("FASTVIDEO_STRICT_DEVICE", "1");
        STRICT_DEVICE_CACHE.reset();
        reset_device_path_stats();

        let x = CudaTensor::from_vec((0..24).map(|i| i as f32).collect(), vec![2, 3, 4]).unwrap();
        let w = CudaTensor::from_vec(vec![1.0; 4], vec![4]).unwrap();
        let b = CudaTensor::from_vec(vec![0.0; 4], vec![4]).unwrap();
        x.layer_norm(1e-5, Some(&w), Some(&b)).unwrap();
        let scale = CudaTensor::from_vec(vec![0.1; 8], vec![2, 1, 4]).unwrap();
        let shift = CudaTensor::from_vec(vec![0.0; 8], vec![2, 1, 4]).unwrap();
        x.modulate(&scale, &shift).unwrap();
        x.gate_mul(&scale).unwrap();

        if !super::super::device::has_live_device() {
            assert!(
                device_path_stats().iter().all(|(_, h, m)| *h == 0 && *m == 0),
                "no live device: nothing should be recorded either way"
            );
        }

        match prev {
            Some(v) => std::env::set_var("FASTVIDEO_STRICT_DEVICE", v),
            None => std::env::remove_var("FASTVIDEO_STRICT_DEVICE"),
        }
        STRICT_DEVICE_CACHE.reset();
    }

    #[test]
    fn gelu_silu_smoke() {
        let x = CudaTensor::from_vec(vec![-2.0, 0.0, 1.5], vec![3]).unwrap();
        let s = x.silu();
        assert!((s.data[2] - 1.2263617).abs() < 1e-5);
    }

    #[test]
    fn layer_norm_matches_naive_reference() {
        // [batch=2, seq=3, dim=4], with affine weight/bias.
        let x = CudaTensor::from_vec(
            (0..24).map(|i| (i as f32) * 0.1 - 1.0).collect(),
            vec![2, 3, 4],
        )
        .unwrap();
        let w = CudaTensor::from_vec(vec![1.0, 2.0, 0.5, 1.5], vec![4]).unwrap();
        let b = CudaTensor::from_vec(vec![0.1, -0.1, 0.2, -0.2], vec![4]).unwrap();
        let eps = 1e-5;
        let got = x.layer_norm(eps, Some(&w), Some(&b)).unwrap();

        let mut expect = vec![0.0f32; 24];
        for row in 0..6 {
            let base = row * 4;
            let vals = &x.data[base..base + 4];
            let mean = vals.iter().sum::<f32>() / 4.0;
            let var = vals.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / 4.0;
            let inv = 1.0 / (var + eps).sqrt();
            for d in 0..4 {
                expect[base + d] = (vals[d] - mean) * inv * w.data[d] + b.data[d];
            }
        }
        for (a, e) in got.data.iter().zip(expect.iter()) {
            assert!((a - e).abs() < 1e-4, "{a} vs {e}");
        }
    }

    #[test]
    fn layer_norm_unaffine_matches_naive_reference() {
        let x = CudaTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 1, 4]).unwrap();
        let got = x.layer_norm(1e-5, None, None).unwrap();
        let mean = 2.5f32;
        let var = ((1.5f32).powi(2) + (0.5f32).powi(2) + (0.5f32).powi(2) + (1.5f32).powi(2)) / 4.0;
        let inv = 1.0 / (var + 1e-5).sqrt();
        let expect: Vec<f32> = [1.0, 2.0, 3.0, 4.0]
            .iter()
            .map(|v| (v - mean) * inv)
            .collect();
        for (a, e) in got.data.iter().zip(expect.iter()) {
            assert!((a - e).abs() < 1e-4, "{a} vs {e}");
        }
    }

    #[test]
    fn modulate_matches_naive_broadcast() {
        // self: [batch=2, seq=3, dim=2]; scale/shift: [batch=2, 1, dim=2].
        let x = CudaTensor::from_vec(
            (0..12).map(|i| i as f32).collect(),
            vec![2, 3, 2],
        )
        .unwrap();
        let scale = CudaTensor::from_vec(vec![0.5, -0.5, 1.0, 2.0], vec![2, 1, 2]).unwrap();
        let shift = CudaTensor::from_vec(vec![1.0, -1.0, 0.0, 0.5], vec![2, 1, 2]).unwrap();
        let got = x.modulate(&scale, &shift).unwrap();

        // Reference via the old (slow) broadcast path: mul(scale+1).add(shift).
        let expect = x
            .mul(&scale.add_scalar(1.0))
            .unwrap()
            .add(&shift)
            .unwrap();
        for (a, e) in got.data.iter().zip(expect.data.iter()) {
            assert!((a - e).abs() < 1e-5, "{a} vs {e}");
        }
    }

    #[test]
    fn gate_mul_matches_naive_broadcast() {
        let x = CudaTensor::from_vec((0..12).map(|i| i as f32).collect(), vec![2, 3, 2]).unwrap();
        let gate = CudaTensor::from_vec(vec![2.0, 3.0, 0.5, -1.0], vec![2, 1, 2]).unwrap();
        let got = x.gate_mul(&gate).unwrap();
        let expect = x.mul(&gate).unwrap();
        for (a, e) in got.data.iter().zip(expect.data.iter()) {
            assert!((a - e).abs() < 1e-5, "{a} vs {e}");
        }
    }
}
