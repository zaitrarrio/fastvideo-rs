//! Owned f32 N-D tensors for the cudarc CUDA Wan backend.
//!
//! **Storage.** A tensor holds a host buffer, a device buffer, or both:
//! - `data` is valid only while `host_valid`; device results leave it empty, so
//!   cloning or reshaping a device tensor copies nothing on the host.
//! - `device` is always current when present; mutating the host copy through
//!   [`CudaTensor::host_mut`] drops it.
//! - [`CudaTensor::pin_device`] uploads and frees the host copy (weights).
//!
//! **Dispatch.** With a live CUDA device and residency on, every op runs on the
//! device: host-only inputs are uploaded (counted as transfers in
//! [`super::stats`]) and results stay on device until [`CudaTensor::host_cow`]
//! or [`CudaTensor::ensure_host`] reads them. Without a device the same ops run
//! the plain-Rust twins in [`super::ops::host`], which are the CPU reference.
//! A host computation with a device live is a recorded fallback.

use std::borrow::Cow;

use thiserror::Error;

use super::ops::{host, BcastOp};
use super::stats;

#[derive(Debug, Error)]
pub enum TensorError {
    #[error("{0}")]
    Message(String),
}

pub type Result<T> = std::result::Result<T, TensorError>;

fn msg(s: impl Into<String>) -> TensorError {
    TensorError::Message(s.into())
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
        f.debug_struct("DeviceBuffer").field("len", &self.slice.len()).finish()
    }
}

/// A device view of a tensor for one op: borrowed when the tensor lives on
/// the device, or a temporary upload of a host-only tensor.
#[cfg(feature = "cuda")]
pub(crate) enum DevRef<'a> {
    Borrowed(&'a cudarc::driver::CudaSlice<f32>),
    Owned(cudarc::driver::CudaSlice<f32>),
}

#[cfg(feature = "cuda")]
impl std::ops::Deref for DevRef<'_> {
    type Target = cudarc::driver::CudaSlice<f32>;
    fn deref(&self) -> &Self::Target {
        match self {
            DevRef::Borrowed(s) => s,
            DevRef::Owned(s) => s,
        }
    }
}

#[derive(Debug)]
pub struct CudaTensor {
    /// Host copy; valid only while `host_valid` (empty for device results).
    pub data: Vec<f32>,
    pub shape: Vec<usize>,
    #[cfg(feature = "cuda")]
    device: Option<DeviceBuffer>,
    #[cfg(feature = "cuda")]
    host_valid: bool,
}

impl Clone for CudaTensor {
    fn clone(&self) -> Self {
        #[cfg(feature = "cuda")]
        {
            if let Some(dev) = &self.device {
                return Self::device_only(dev.clone(), self.shape.clone());
            }
        }
        Self::host_only(self.data.clone(), self.shape.clone())
    }
}

fn numel(shape: &[usize]) -> usize {
    shape.iter().product()
}

fn strides(shape: &[usize]) -> Vec<usize> {
    let mut s = vec![1; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        s[i] = s[i + 1] * shape[i + 1];
    }
    s
}

/// Broadcast `small` against `big` when `small`'s non-singleton dims form one
/// contiguous block matching `big`: then `small[(i / inner) % period]` pairs
/// with `big[i]`. Returns `(inner, period)`.
fn repeat_plan(big: &[usize], small: &[usize]) -> Option<(usize, usize)> {
    if small.len() > big.len() {
        return None;
    }
    let pad = big.len() - small.len();
    let padded: Vec<usize> = std::iter::repeat_n(1, pad).chain(small.iter().copied()).collect();
    let non_one: Vec<usize> = (0..padded.len()).filter(|&i| padded[i] != 1).collect();
    let (lo, hi) = match (non_one.first(), non_one.last()) {
        (Some(&lo), Some(&hi)) => (lo, hi + 1),
        _ => return Some((1, 1)),
    };
    if (lo..hi).any(|i| padded[i] != big[i]) {
        return None;
    }
    Some((numel(&big[hi..]), numel(&big[lo..hi])))
}

fn broadcast_shapes(a: &[usize], b: &[usize]) -> Result<Vec<usize>> {
    let rank = a.len().max(b.len());
    let mut out = vec![1; rank];
    for (i, o) in out.iter_mut().enumerate() {
        let da = if i < rank - a.len() { 1 } else { a[i - (rank - a.len())] };
        let db = if i < rank - b.len() { 1 } else { b[i - (rank - b.len())] };
        if da == db || da == 1 || db == 1 {
            *o = da.max(db);
        } else {
            return Err(msg(format!("broadcast {a:?} vs {b:?}")));
        }
    }
    Ok(out)
}

impl CudaTensor {
    pub fn new(data: Vec<f32>, shape: Vec<usize>) -> Result<Self> {
        if data.len() != numel(&shape) {
            return Err(msg(format!("data len {} != shape {:?}", data.len(), shape)));
        }
        Ok(Self::host_only(data, shape))
    }

    pub fn zeros(shape: &[usize]) -> Self {
        Self::host_only(vec![0.0; numel(shape)], shape.to_vec())
    }

    pub fn ones(shape: &[usize]) -> Self {
        Self::host_only(vec![1.0; numel(shape)], shape.to_vec())
    }

    pub fn from_vec(data: Vec<f32>, shape: Vec<usize>) -> Result<Self> {
        Self::new(data, shape)
    }

    pub fn from_slice(data: &[f32], shape: &[usize]) -> Result<Self> {
        Self::new(data.to_vec(), shape.to_vec())
    }

    pub(crate) fn host_only(data: Vec<f32>, shape: Vec<usize>) -> Self {
        Self {
            data,
            shape,
            #[cfg(feature = "cuda")]
            device: None,
            #[cfg(feature = "cuda")]
            host_valid: true,
        }
    }

    #[cfg(feature = "cuda")]
    fn device_only(device: DeviceBuffer, shape: Vec<usize>) -> Self {
        Self {
            data: Vec::new(),
            shape,
            device: Some(device),
            host_valid: false,
        }
    }

    /// Wrap a device buffer (no host copy).
    #[cfg(feature = "cuda")]
    pub fn from_device_slice(slice: cudarc::driver::CudaSlice<f32>, shape: Vec<usize>) -> Result<Self> {
        if slice.len() != numel(&shape) {
            return Err(msg(format!("device len {} != shape {:?}", slice.len(), shape)));
        }
        Ok(Self::device_only(DeviceBuffer { slice: std::sync::Arc::new(slice) }, shape))
    }

    #[cfg(feature = "cuda")]
    pub fn device_slice(&self) -> Option<&cudarc::driver::CudaSlice<f32>> {
        self.device.as_ref().map(|b| b.slice.as_ref())
    }

    #[cfg(feature = "cuda")]
    pub fn is_device_fresh(&self) -> bool {
        self.device.is_some()
    }

    fn has_host(&self) -> bool {
        #[cfg(feature = "cuda")]
        {
            self.host_valid
        }
        #[cfg(not(feature = "cuda"))]
        {
            true
        }
    }

    /// Host values, downloading (without caching) when the tensor lives on device.
    pub fn host_cow(&self) -> Result<Cow<'_, [f32]>> {
        #[cfg(feature = "cuda")]
        if !self.host_valid {
            let buf = self.device.as_ref().ok_or_else(|| msg("tensor has neither host nor device data"))?;
            let dev = super::device::global_device()
                .ok_or_else(|| msg("host_cow: device tensor but no global CUDA device"))?;
            let v = dev.stream.memcpy_dtov(buf.slice.as_ref()).map_err(|e| msg(e.to_string()))?;
            stats::record_d2h(v.len());
            return Ok(Cow::Owned(v));
        }
        Ok(Cow::Borrowed(self.data.as_slice()))
    }

    /// Make `data` valid (download once); the device copy stays current.
    pub fn ensure_host(&mut self) -> Result<()> {
        if !self.has_host() {
            self.data = self.host_cow()?.into_owned();
            #[cfg(feature = "cuda")]
            {
                self.host_valid = true;
            }
        }
        Ok(())
    }

    pub fn to_host(&mut self) -> Result<()> {
        self.ensure_host()
    }

    /// Mutable host values; invalidates the device copy.
    pub fn host_mut(&mut self) -> Result<&mut Vec<f32>> {
        self.ensure_host()?;
        #[cfg(feature = "cuda")]
        {
            self.device = None;
        }
        Ok(&mut self.data)
    }

    /// Upload when a device is expected and the tensor is host-only.
    pub fn ensure_device(&mut self) -> Result<()> {
        #[cfg(feature = "cuda")]
        if self.device.is_none() && stats::device_expected() {
            let dev = super::device::global_device().ok_or_else(|| msg("no global CUDA device"))?;
            let slice = dev.stream.memcpy_stod(&self.data).map_err(|e| msg(e.to_string()))?;
            stats::record_h2d(self.data.len());
            self.device = Some(DeviceBuffer { slice: std::sync::Arc::new(slice) });
        }
        Ok(())
    }

    /// Owned [`Self::ensure_device`].
    pub fn to_device(mut self) -> Result<Self> {
        self.ensure_device()?;
        Ok(self)
    }

    /// Keep this tensor on the device only (weights): upload, then free the
    /// host copy. A no-op on CPU runs.
    pub fn pin_device(&mut self) -> Result<()> {
        self.ensure_device()?;
        #[cfg(feature = "cuda")]
        if self.device.is_some() {
            self.data = Vec::new();
            self.host_valid = false;
        }
        Ok(())
    }

    /// Device buffer for an op, uploading a host-only tensor temporarily.
    /// `None` when ops should run on host.
    #[cfg(feature = "cuda")]
    pub(crate) fn dev(&self) -> Result<Option<DevRef<'_>>> {
        if let Some(buf) = &self.device {
            return Ok(Some(DevRef::Borrowed(buf.slice.as_ref())));
        }
        if !stats::device_expected() {
            return Ok(None);
        }
        let dev = super::device::global_device().ok_or_else(|| msg("no global CUDA device"))?;
        let slice = dev.stream.memcpy_stod(&self.data).map_err(|e| msg(e.to_string()))?;
        stats::record_h2d(self.data.len());
        Ok(Some(DevRef::Owned(slice)))
    }

    /// Wrap an op result: device buffer, or host data.
    #[cfg(feature = "cuda")]
    pub(crate) fn from_dev_result(slice: cudarc::driver::CudaSlice<f32>, shape: Vec<usize>) -> Result<Self> {
        Self::from_device_slice(slice, shape)
    }

    pub fn rank(&self) -> usize {
        self.shape.len()
    }

    pub fn numel(&self) -> usize {
        numel(&self.shape)
    }

    pub fn dim(&self, axis: usize) -> Result<usize> {
        self.shape
            .get(axis)
            .copied()
            .ok_or_else(|| msg(format!("axis {axis} out of range {:?}", self.shape)))
    }

    fn axis(&self, axis: isize) -> Result<usize> {
        let rank = self.rank() as isize;
        let a = if axis < 0 { rank + axis } else { axis };
        if a < 0 || a >= rank {
            Err(msg(format!("axis {axis} invalid for shape {:?}", self.shape)))
        } else {
            Ok(a as usize)
        }
    }

    /// Same storage, new shape (no copy).
    fn with_shape(&self, shape: Vec<usize>) -> Self {
        let mut t = self.clone();
        t.shape = shape;
        t
    }

    pub fn reshape(&self, shape: Vec<usize>) -> Result<CudaTensor> {
        if numel(&shape) != self.numel() {
            return Err(msg(format!("reshape {:?} -> {:?}", self.shape, shape)));
        }
        Ok(self.with_shape(shape))
    }

    pub fn reshape_owned(mut self, shape: Vec<usize>) -> Result<CudaTensor> {
        if numel(&shape) != self.numel() {
            return Err(msg(format!("reshape {:?} -> {:?}", self.shape, shape)));
        }
        self.shape = shape;
        Ok(self)
    }

    pub fn squeeze(&self, dim: usize) -> Result<CudaTensor> {
        if self.dim(dim)? != 1 {
            return Err(msg(format!("squeeze expected size 1 at {dim}, got {:?}", self.shape)));
        }
        let mut shape = self.shape.clone();
        shape.remove(dim);
        Ok(self.with_shape(shape))
    }

    pub fn unsqueeze(&self, dim: usize) -> Result<CudaTensor> {
        if dim > self.rank() {
            return Err(msg("unsqueeze out of range"));
        }
        let mut shape = self.shape.clone();
        shape.insert(dim, 1);
        Ok(self.with_shape(shape))
    }

    pub fn flatten_from(&self, dim: usize) -> Result<CudaTensor> {
        let mut shape = self.shape[..dim].to_vec();
        shape.push(numel(&self.shape[dim..]));
        self.reshape(shape)
    }

    pub fn permute(&self, dims: &[usize]) -> Result<CudaTensor> {
        let rank = self.rank();
        let mut seen = vec![false; rank];
        if dims.len() != rank || dims.iter().any(|&d| d >= rank || std::mem::replace(&mut seen[d], true)) {
            return Err(msg(format!("invalid permute {dims:?} for {:?}", self.shape)));
        }
        let out_shape: Vec<usize> = dims.iter().map(|&d| self.shape[d]).collect();
        if dims.iter().enumerate().all(|(i, &d)| i == d) {
            return Ok(self.clone());
        }
        // Moving only singleton axes is a reshape.
        let non_one: Vec<usize> = dims.iter().copied().filter(|&d| self.shape[d] != 1).collect();
        if non_one.windows(2).all(|w| w[0] < w[1]) {
            return Ok(self.with_shape(out_shape));
        }
        #[cfg(feature = "cuda")]
        if rank <= 6 {
            if let Some(src) = self.dev()? {
                let out = super::ops::gather_nd_device(&src, &self.shape, dims)?;
                return Self::from_dev_result(out, out_shape);
            }
        }
        stats::host_fallback("permute", format_args!("{:?} {dims:?}", self.shape))?;
        Ok(Self::host_only(host::permute(&self.host_cow()?, &self.shape, dims), out_shape))
    }

    pub fn transpose(&self, dim0: usize, dim1: usize) -> Result<CudaTensor> {
        let mut dims: Vec<usize> = (0..self.rank()).collect();
        dims.swap(dim0, dim1);
        self.permute(&dims)
    }

    /// Block geometry for copies along `dim`: `(outer, inner)`.
    fn blocks(&self, dim: usize) -> (usize, usize) {
        (numel(&self.shape[..dim]), numel(&self.shape[dim + 1..]))
    }

    pub fn narrow(&self, dim: usize, start: usize, len: usize) -> Result<CudaTensor> {
        let d = self.dim(dim)?;
        if start + len > d {
            return Err(msg(format!("narrow dim={dim} start={start} len={len} of {:?}", self.shape)));
        }
        if start == 0 && len == d {
            return Ok(self.clone());
        }
        let mut out_shape = self.shape.clone();
        out_shape[dim] = len;
        let (outer, inner) = self.blocks(dim);
        #[cfg(feature = "cuda")]
        if let Some(src) = self.dev()? {
            let mut out = super::ops::alloc(numel(&out_shape).max(1))?;
            super::ops::block_copy_device(&src, &mut out, outer, len * inner, d * inner, len * inner, start * inner, 0)?;
            return Self::from_dev_result(out, out_shape);
        }
        stats::host_fallback("narrow", format_args!("{:?} dim={dim}", self.shape))?;
        let src = self.host_cow()?;
        let mut out = Vec::with_capacity(numel(&out_shape));
        for o in 0..outer {
            let base = o * d * inner + start * inner;
            out.extend_from_slice(&src[base..base + len * inner]);
        }
        Ok(Self::host_only(out, out_shape))
    }

    pub fn chunk(&self, chunks: usize, dim: usize) -> Result<Vec<CudaTensor>> {
        let d = self.dim(dim)?;
        if chunks == 0 || d % chunks != 0 {
            return Err(msg("chunk size not divisible"));
        }
        let size = d / chunks;
        (0..chunks).map(|i| self.narrow(dim, i * size, size)).collect()
    }

    pub fn cat(tensors: &[&CudaTensor], dim: usize) -> Result<CudaTensor> {
        let first = tensors.first().ok_or_else(|| msg("cat empty"))?;
        let rank = first.rank();
        let mut out_shape = first.shape.clone();
        let mut cat_len = 0usize;
        for t in tensors {
            if t.rank() != rank || t.shape.iter().zip(&first.shape).enumerate().any(|(i, (a, b))| i != dim && a != b) {
                return Err(msg(format!("cat shape mismatch at dim {dim}: {:?} vs {:?}", t.shape, first.shape)));
            }
            cat_len += t.shape[dim];
        }
        out_shape[dim] = cat_len;
        if tensors.len() == 1 {
            return Ok((*first).clone());
        }
        let (outer, inner) = first.blocks(dim);
        #[cfg(feature = "cuda")]
        if stats::device_expected() {
            let mut out = super::ops::alloc(numel(&out_shape).max(1))?;
            let mut offset = 0usize;
            for t in tensors {
                let len = t.shape[dim] * inner;
                if let Some(src) = t.dev()? {
                    super::ops::block_copy_device(&src, &mut out, outer, len, len, cat_len * inner, 0, offset)?;
                }
                offset += len;
            }
            return Self::from_dev_result(out, out_shape);
        }
        let hosts = tensors.iter().map(|t| t.host_cow()).collect::<Result<Vec<_>>>()?;
        let mut out = Vec::with_capacity(numel(&out_shape));
        for o in 0..outer {
            for (t, h) in tensors.iter().zip(&hosts) {
                let len = t.shape[dim] * inner;
                out.extend_from_slice(&h[o * len..(o + 1) * len]);
            }
        }
        Ok(Self::host_only(out, out_shape))
    }

    pub fn pad_zeros(&self, dim: usize, left: usize, right: usize) -> Result<CudaTensor> {
        if left == 0 && right == 0 {
            return Ok(self.clone());
        }
        let d = self.dim(dim)?;
        let mut out_shape = self.shape.clone();
        out_shape[dim] = d + left + right;
        let (outer, inner) = self.blocks(dim);
        #[cfg(feature = "cuda")]
        if let Some(src) = self.dev()? {
            let mut out = super::ops::fill_device(numel(&out_shape).max(1), 0.0)?;
            super::ops::block_copy_device(&src, &mut out, outer, d * inner, d * inner, out_shape[dim] * inner, 0, left * inner)?;
            return Self::from_dev_result(out, out_shape);
        }
        let src = self.host_cow()?;
        let mut out = vec![0.0f32; numel(&out_shape)];
        for o in 0..outer {
            let dst = o * out_shape[dim] * inner + left * inner;
            out[dst..dst + d * inner].copy_from_slice(&src[o * d * inner..(o + 1) * d * inner]);
        }
        Ok(Self::host_only(out, out_shape))
    }

    fn binary(&self, other: &CudaTensor, op: BcastOp) -> Result<CudaTensor> {
        if self.shape == other.shape && matches!(op, BcastOp::Add | BcastOp::Sub | BcastOp::Mul) {
            #[cfg(feature = "cuda")]
            if let (Some(a), Some(b)) = (self.dev()?, other.dev()?) {
                let kind = match op {
                    BcastOp::Add => super::ops::ElemBinary::Add,
                    BcastOp::Sub => super::ops::ElemBinary::Sub,
                    _ => super::ops::ElemBinary::Mul,
                };
                return Self::from_dev_result(super::ops::elem_binary_device(&a, &b, kind)?, self.shape.clone());
            }
            let (a, b) = (self.host_cow()?, other.host_cow()?);
            return Ok(Self::host_only(host::map2(&a, &b, |x, y| op.apply(x, y)), self.shape.clone()));
        }
        let out_shape = broadcast_shapes(&self.shape, &other.shape)?;
        // `big` carries the full output shape; ops are swapped when it is `other`.
        let plan = if self.shape == out_shape {
            repeat_plan(&self.shape, &other.shape).map(|p| (self, other, op, p))
        } else if other.shape == out_shape {
            let swapped = match op {
                BcastOp::Sub => BcastOp::RSub,
                BcastOp::Div => BcastOp::RDiv,
                o => o,
            };
            repeat_plan(&other.shape, &self.shape).map(|p| (other, self, swapped, p))
        } else {
            None
        };
        if let Some((big, small, op, (inner, period))) = plan {
            #[cfg(feature = "cuda")]
            if let (Some(a), Some(b)) = (big.dev()?, small.dev()?) {
                let out = super::ops::bcast_binary_device(&a, &b, inner, period, op)?;
                return Self::from_dev_result(out, out_shape);
            }
            let (a, b) = (big.host_cow()?, small.host_cow()?);
            return Ok(Self::host_only(host::bcast_binary(&a, &b, inner, period, op), out_shape));
        }
        stats::host_fallback("broadcast", format_args!("{:?} {op:?} {:?}", self.shape, other.shape))?;
        let (a, b) = (self.host_cow()?, other.host_cow()?);
        let (sa, sb, so) = (strides(&self.shape), strides(&other.shape), strides(&out_shape));
        let (pa, pb) = (out_shape.len() - self.rank(), out_shape.len() - other.rank());
        let mut out = vec![0.0f32; numel(&out_shape)];
        for (i, o) in out.iter_mut().enumerate() {
            let (mut ia, mut ib, mut rem) = (0usize, 0usize, i);
            for k in 0..out_shape.len() {
                let c = rem / so[k];
                rem %= so[k];
                if k >= pa && self.shape[k - pa] != 1 {
                    ia += c * sa[k - pa];
                }
                if k >= pb && other.shape[k - pb] != 1 {
                    ib += c * sb[k - pb];
                }
            }
            *o = op.apply(a[ia], b[ib]);
        }
        Ok(Self::host_only(out, out_shape))
    }

    pub fn add(&self, other: &CudaTensor) -> Result<CudaTensor> {
        self.binary(other, BcastOp::Add)
    }

    pub fn sub(&self, other: &CudaTensor) -> Result<CudaTensor> {
        self.binary(other, BcastOp::Sub)
    }

    pub fn mul(&self, other: &CudaTensor) -> Result<CudaTensor> {
        self.binary(other, BcastOp::Mul)
    }

    pub fn div(&self, other: &CudaTensor) -> Result<CudaTensor> {
        self.binary(other, BcastOp::Div)
    }

    /// `Σ coef·tensor` over same-shaped tensors (sampler updates, CFG).
    pub fn lincomb(terms: &[(f32, &CudaTensor)]) -> Result<CudaTensor> {
        let (_, first) = terms.first().ok_or_else(|| msg("lincomb: no terms"))?;
        if terms.iter().any(|(_, t)| t.shape != first.shape) {
            return Err(msg("lincomb: shape mismatch"));
        }
        #[cfg(feature = "cuda")]
        if stats::device_expected() {
            let refs = terms.iter().map(|(_, t)| t.dev()).collect::<Result<Vec<_>>>()?;
            if refs.iter().all(Option::is_some) {
                let slices: Vec<(f32, &cudarc::driver::CudaSlice<f32>)> =
                    terms.iter().zip(&refs).map(|((c, _), r)| (*c, &**r.as_ref().unwrap())).collect();
                return Self::from_dev_result(super::ops::lincomb_device(&slices)?, first.shape.clone());
            }
        }
        let hosts = terms.iter().map(|(_, t)| t.host_cow()).collect::<Result<Vec<_>>>()?;
        let pairs: Vec<(f32, &[f32])> = terms.iter().zip(&hosts).map(|((c, _), h)| (*c, &h[..])).collect();
        Ok(Self::host_only(host::lincomb(&pairs), first.shape.clone()))
    }

    fn unary_op(
        &self,
        #[cfg(feature = "cuda")] device: impl FnOnce(&cudarc::driver::CudaSlice<f32>) -> Result<cudarc::driver::CudaSlice<f32>>,
        #[cfg(not(feature = "cuda"))] _device: impl FnOnce(&()) -> Result<()>,
        host_fn: impl Fn(f32) -> f32 + Sync,
    ) -> Result<CudaTensor> {
        #[cfg(feature = "cuda")]
        if let Some(a) = self.dev()? {
            return Self::from_dev_result(device(&a)?, self.shape.clone());
        }
        Ok(Self::host_only(host::map1(&self.host_cow()?, host_fn), self.shape.clone()))
    }

    pub fn add_scalar(&self, s: f32) -> CudaTensor {
        self.try_add_scalar(s).expect("add_scalar")
    }

    pub fn try_add_scalar(&self, s: f32) -> Result<CudaTensor> {
        self.unary_op(
            #[cfg(feature = "cuda")]
            |a| super::ops::add_scalar_device(a, s),
            #[cfg(not(feature = "cuda"))]
            |_| Ok(()),
            move |x| x + s,
        )
    }

    pub fn mul_scalar(&self, s: f32) -> CudaTensor {
        self.try_mul_scalar(s).expect("mul_scalar")
    }

    pub fn try_mul_scalar(&self, s: f32) -> Result<CudaTensor> {
        self.unary_op(
            #[cfg(feature = "cuda")]
            |a| super::ops::mul_scalar_device(a, s),
            #[cfg(not(feature = "cuda"))]
            |_| Ok(()),
            move |x| x * s,
        )
    }

    pub fn clamp(&self, lo: f32, hi: f32) -> CudaTensor {
        self.unary_op(
            #[cfg(feature = "cuda")]
            |a| super::ops::clamp_device(a, lo, hi),
            #[cfg(not(feature = "cuda"))]
            |_| Ok(()),
            move |x| x.clamp(lo, hi),
        )
        .expect("clamp")
    }

    pub fn silu(&self) -> CudaTensor {
        self.unary_op(
            #[cfg(feature = "cuda")]
            |a| super::ops::unary_device(a, super::ops::ElemUnary::Silu),
            #[cfg(not(feature = "cuda"))]
            |_| Ok(()),
            host::silu,
        )
        .expect("silu")
    }

    /// Exact GELU (erf). `gelu_tanh` is the approximation; a checkpoint means
    /// one or the other and they differ by ~1e-3.
    pub fn gelu_erf(&self) -> CudaTensor {
        self.unary_op(
            #[cfg(feature = "cuda")]
            |a| super::ops::unary_device(a, super::ops::ElemUnary::GeluErf),
            #[cfg(not(feature = "cuda"))]
            |_| Ok(()),
            host::gelu_erf,
        )
        .expect("gelu_erf")
    }

    pub fn leaky_relu(&self, slope: f32) -> CudaTensor {
        self.unary_op(
            #[cfg(feature = "cuda")]
            |a| super::ops::leaky_relu_device(a, slope),
            #[cfg(not(feature = "cuda"))]
            |_| Ok(()),
            move |x| host::leaky_relu(x, slope),
        )
        .expect("leaky_relu")
    }

    /// Snake / SnakeBeta over `[N, C, L]`: `x + inv_beta[c] * sin^2(alpha[c] x)`.
    /// `alpha` and `inv_beta` are `[C]`, already out of log space and with the
    /// `1 / (beta + eps)` taken, so every Snake variant is this one call.
    pub fn snake_beta(&self, alpha: &CudaTensor, inv_beta: &CudaTensor) -> Result<CudaTensor> {
        let [_, c, l] = self.shape[..] else {
            return Err(msg(format!("snake_beta expects [N, C, L], got {:?}", self.shape)));
        };
        if alpha.numel() != c || inv_beta.numel() != c {
            return Err(msg(format!("snake_beta: {} alphas for {c} channels", alpha.numel())));
        }
        #[cfg(feature = "cuda")]
        if let (Some(a), Some(al), Some(ib)) = (self.dev()?, alpha.dev()?, inv_beta.dev()?) {
            return Self::from_dev_result(super::ops::snake_beta_device(&a, &al, &ib, c, l)?, self.shape.clone());
        }
        Ok(Self::host_only(
            host::snake_beta(&self.host_cow()?, &alpha.host_cow()?, &inv_beta.host_cow()?, c, l),
            self.shape.clone(),
        ))
    }

    /// rotate_half rotary embedding (HF Llama / Qwen / Gemma convention) on
    /// `[B, H, S, D]` with explicit `[S, R]` cos/sin tables; channels `[R, D)`
    /// pass through. Positions are whatever built the tables.
    pub fn rope_half(&self, cos: &CudaTensor, sin: &CudaTensor) -> Result<CudaTensor> {
        let [_, _, s, d] = self.shape[..] else {
            return Err(msg(format!("rope_half expects [B, H, S, D], got {:?}", self.shape)));
        };
        let r = match cos.shape[..] {
            [cs, r] if cs == s && cos.shape == sin.shape && r % 2 == 0 && r > 0 && r <= d => r,
            _ => return Err(msg(format!("rope_half tables {:?}/{:?} for S={s} D={d}", cos.shape, sin.shape))),
        };
        #[cfg(feature = "cuda")]
        if let (Some(x), Some(c), Some(sn)) = (self.dev()?, cos.dev()?, sin.dev()?) {
            return Self::from_dev_result(super::ops::rope_half_device(&x, &c, &sn, s, d, r)?, self.shape.clone());
        }
        Ok(Self::host_only(
            host::rope_half(&self.host_cow()?, &cos.host_cow()?, &sin.host_cow()?, s, d, r),
            self.shape.clone(),
        ))
    }

    /// Grouped-query attention's key/value expansion: `[B, Hkv, S, D]` →
    /// `[B, Hkv * rep, S, D]`, each kv head repeated `rep` times in place.
    pub fn repeat_kv(&self, rep: usize) -> Result<CudaTensor> {
        let [b, hkv, s, d] = self.shape[..] else {
            return Err(msg(format!("repeat_kv expects [B, Hkv, S, D], got {:?}", self.shape)));
        };
        if rep == 1 {
            return Ok(self.clone());
        }
        if rep == 0 {
            return Err(msg("repeat_kv by zero"));
        }
        let shape = vec![b, hkv * rep, s, d];
        #[cfg(feature = "cuda")]
        if let Some(x) = self.dev()? {
            return Self::from_dev_result(super::ops::repeat_kv_device(&x, hkv, rep, s * d)?, shape);
        }
        Ok(Self::host_only(host::repeat_kv(&self.host_cow()?, hkv, rep, s * d), shape))
    }

    pub fn gelu_tanh(&self) -> CudaTensor {
        self.unary_op(
            #[cfg(feature = "cuda")]
            |a| super::ops::unary_device(a, super::ops::ElemUnary::GeluTanh),
            #[cfg(not(feature = "cuda"))]
            |_| Ok(()),
            host::gelu_tanh,
        )
        .expect("gelu_tanh")
    }

    /// Host-only elementwise math with no device kernel (not on any hot path).
    fn host_unary(&self, op: &'static str, f: impl Fn(f32) -> f32 + Sync) -> Result<CudaTensor> {
        stats::host_fallback(op, format_args!("{:?}", self.shape))?;
        Ok(Self::host_only(host::map1(&self.host_cow()?, f), self.shape.clone()))
    }

    pub fn sqrt(&self) -> Result<CudaTensor> {
        self.host_unary("sqrt", f32::sqrt)
    }

    pub fn sqr(&self) -> Result<CudaTensor> {
        self.host_unary("sqr", |x| x * x)
    }

    pub fn mean_keepdim(&self, dim: isize) -> Result<CudaTensor> {
        let axis = self.axis(dim)?;
        stats::host_fallback("mean_keepdim", format_args!("{:?}", self.shape))?;
        let (outer, inner) = self.blocks(axis);
        let d = self.shape[axis];
        let src = self.host_cow()?;
        let mut out = vec![0.0f32; outer * inner];
        for o in 0..outer {
            for a in 0..d {
                for i in 0..inner {
                    out[o * inner + i] += src[(o * d + a) * inner + i];
                }
            }
        }
        out.iter_mut().for_each(|v| *v /= d as f32);
        let mut shape = self.shape.clone();
        shape[axis] = 1;
        Ok(Self::host_only(out, shape))
    }

    pub fn matmul(&self, other: &CudaTensor) -> Result<CudaTensor> {
        if self.rank() < 2 || other.rank() < 2 {
            return Err(msg("matmul needs rank >= 2"));
        }
        let (m, k) = (self.shape[self.rank() - 2], self.shape[self.rank() - 1]);
        let (k2, n) = (other.shape[other.rank() - 2], other.shape[other.rank() - 1]);
        if k != k2 {
            return Err(msg(format!("matmul inner {k} vs {k2}")));
        }
        let a_batch = &self.shape[..self.rank() - 2];
        let b_batch = &other.shape[..other.rank() - 2];
        let batch_shape = broadcast_shapes(a_batch, b_batch)?;
        let batch = numel(&batch_shape);
        let mut out_shape = batch_shape.clone();
        out_shape.extend([m, n]);
        #[cfg(feature = "cuda")]
        if numel(a_batch) == batch && numel(b_batch) == batch {
            if let (Some(a), Some(b)) = (self.dev()?, other.dev()?) {
                let mut c = super::ops::alloc((batch * m * n).max(1))?;
                super::device::matmul_2d_strided_batched(&a, &b, &mut c, batch, m, k, n).map_err(|e| msg(e.to_string()))?;
                return Self::from_dev_result(c, out_shape);
            }
        }
        stats::host_fallback("matmul", format_args!("{:?} @ {:?}", self.shape, other.shape))?;
        let (a, b) = (self.host_cow()?, other.host_cow()?);
        let (a_n, b_n) = (numel(a_batch), numel(b_batch));
        let mut out = vec![0.0f32; batch * m * n];
        use rayon::prelude::*;
        out.par_chunks_mut((m * n).max(1)).enumerate().for_each(|(bi, o)| {
            let (ai, bj) = (if a_n == 1 { 0 } else { bi }, if b_n == 1 { 0 } else { bi });
            let (a, b) = (&a[ai * m * k..][..m * k], &b[bj * k * n..][..k * n]);
            for i in 0..m {
                let row = &mut o[i * n..(i + 1) * n];
                for t in 0..k {
                    let av = a[i * k + t];
                    for (r, &bv) in row.iter_mut().zip(&b[t * n..(t + 1) * n]) {
                        *r += av * bv;
                    }
                }
            }
        });
        Ok(Self::host_only(out, out_shape))
    }

    pub fn softmax(&self, dim: isize) -> Result<CudaTensor> {
        let axis = self.axis(dim)?;
        if axis + 1 != self.rank() {
            let mut perm: Vec<usize> = (0..self.rank()).collect();
            perm.swap(axis, self.rank() - 1);
            return self.permute(&perm)?.softmax(-1)?.permute(&perm);
        }
        let width = self.shape[axis];
        #[cfg(feature = "cuda")]
        if let Some(a) = self.dev()? {
            return Self::from_dev_result(super::ops::softmax_last_device(&a, width)?, self.shape.clone());
        }
        Ok(Self::host_only(host::softmax_last(&self.host_cow()?, width), self.shape.clone()))
    }

    /// RMS norm over the last dim.
    pub fn rms_norm(&self, weight: &CudaTensor, eps: f32) -> Result<CudaTensor> {
        let width = *self.shape.last().ok_or_else(|| msg("rms_norm on scalar"))?;
        if weight.numel() != width {
            return Err(msg("rms_norm weight size"));
        }
        #[cfg(feature = "cuda")]
        if let (Some(a), Some(w)) = (self.dev()?, weight.dev()?) {
            return Self::from_dev_result(super::ops::rms_norm_last_device(&a, &w, eps)?, self.shape.clone());
        }
        Ok(Self::host_only(host::rms_norm_last(&self.host_cow()?, &weight.host_cow()?, eps), self.shape.clone()))
    }

    pub fn layer_norm(&self, eps: f32, weight: Option<&CudaTensor>, bias: Option<&CudaTensor>) -> Result<CudaTensor> {
        let width = *self.shape.last().ok_or_else(|| msg("layer_norm on scalar"))?;
        let affine = match (weight, bias) {
            (Some(w), Some(b)) => Some((w, b)),
            (None, None) => None,
            _ => return Err(msg("layer_norm needs both weight and bias, or neither")),
        };
        #[cfg(feature = "cuda")]
        if let Some(a) = self.dev()? {
            let out = match affine {
                Some((w, b)) => {
                    let (w, b) = (w.dev()?.ok_or_else(|| msg("weight"))?, b.dev()?.ok_or_else(|| msg("bias"))?);
                    super::ops::layer_norm_last_device(&a, Some((&w, &b)), width, eps)?
                }
                None => super::ops::layer_norm_last_device(&a, None, width, eps)?,
            };
            return Self::from_dev_result(out, self.shape.clone());
        }
        let x = self.host_cow()?;
        let out = match affine {
            Some((w, b)) => host::layer_norm_last(&x, width, Some((&w.host_cow()?, &b.host_cow()?)), eps),
            None => host::layer_norm_last(&x, width, None, eps),
        };
        Ok(Self::host_only(out, self.shape.clone()))
    }

    /// Add `bias` along `dim` (channel bias for NC… tensors, last dim for linears).
    pub fn add_bias(mut self, bias: &CudaTensor, dim: usize) -> Result<CudaTensor> {
        let c = self.dim(dim)?;
        if bias.numel() != c {
            return Err(msg(format!("bias len {} != dim {dim} of {:?}", bias.numel(), self.shape)));
        }
        let inner = numel(&self.shape[dim + 1..]);
        #[cfg(feature = "cuda")]
        if let Some(DeviceBuffer { slice }) = self.device.as_mut() {
            if let Some(buf) = std::sync::Arc::get_mut(slice) {
                let b = bias.dev()?.ok_or_else(|| msg("bias upload"))?;
                super::ops::add_bias_inplace_device(buf, &b, inner)?;
                return Ok(self);
            }
        }
        #[cfg(feature = "cuda")]
        if self.device.is_some() {
            return self.add(&bias.reshape(bias_shape(self.rank(), dim, c))?);
        }
        let b = bias.host_cow()?.into_owned();
        let data = self.host_mut()?;
        for (i, v) in data.iter_mut().enumerate() {
            *v += b[(i / inner) % c];
        }
        Ok(self)
    }

    /// Cross-correlation of NCHW `self` with OIHW `weight`.
    /// 1-D convolution, PyTorch `Conv1d` semantics. `self`: `[N, C, L]`,
    /// `weight`: `[C_out, C / groups, K]`. On the device this is a cuDNN conv2d
    /// over a unit height, so dilation and groups cost nothing extra.
    pub fn conv1d(
        &self,
        weight: &CudaTensor,
        bias: Option<&CudaTensor>,
        padding: usize,
        stride: usize,
        dilation: usize,
        groups: usize,
    ) -> Result<CudaTensor> {
        let ([n, c, l], [oc, cg, k]) = (self.dims3("conv1d input")?, weight.dims3("conv1d weight")?);
        if stride == 0 || dilation == 0 || groups == 0 || k == 0 || c % groups != 0 || oc % groups != 0 || cg * groups != c {
            return Err(msg(format!("conv1d: x={:?} w={:?} groups={groups}", self.shape, weight.shape)));
        }
        if l + 2 * padding < dilation * (k - 1) + 1 {
            return Err(msg(format!("conv1d: kernel reach exceeds input: x={:?} w={:?} pad={padding} dilation={dilation}", self.shape, weight.shape)));
        }
        #[cfg(feature = "cuda")]
        if let (Some(x), Some(w)) = (self.dev()?, weight.dev()?) {
            let (y, ys) = super::conv::cudnn_conv_ext(&x, &[n, c, 1, l], &w, &[oc, cg, 1, k], &[0, padding], &[1, stride], &[1, dilation], groups)
                .map_err(|e| msg(e.to_string()))?;
            let y = Self::from_dev_result(y, vec![n, oc, ys[3]])?;
            return match bias {
                Some(b) => y.add_bias(b, 1),
                None => Ok(y),
            };
        }
        let (y, lo) = host::conv1d(&self.host_cow()?, (n, c, l), &weight.host_cow()?, (oc, k), padding, stride, dilation, groups);
        let y = Self::host_only(y, vec![n, oc, lo]);
        match bias {
            Some(b) => y.add_bias(b, 1),
            None => Ok(y),
        }
    }

    /// 1-D transposed convolution, PyTorch `ConvTranspose1d` semantics. `self`:
    /// `[N, C, L]`, `weight`: `[C, C_out / groups, K]`; output length
    /// `(L - 1) * stride - 2 * padding + dilation * (K - 1) + output_padding + 1`.
    #[allow(clippy::too_many_arguments)]
    pub fn conv_transpose1d(
        &self,
        weight: &CudaTensor,
        bias: Option<&CudaTensor>,
        padding: usize,
        stride: usize,
        dilation: usize,
        groups: usize,
        output_padding: usize,
    ) -> Result<CudaTensor> {
        let ([n, c, l], [wc, og, k]) = (self.dims3("conv_transpose1d input")?, weight.dims3("conv_transpose1d weight")?);
        if stride == 0 || dilation == 0 || groups == 0 || k == 0 || l == 0 || wc != c || c % groups != 0 {
            return Err(msg(format!("conv_transpose1d: x={:?} w={:?} groups={groups}", self.shape, weight.shape)));
        }
        if (l - 1) * stride + dilation * (k - 1) + output_padding + 1 <= 2 * padding {
            return Err(msg(format!("conv_transpose1d: padding {padding} leaves no output for x={:?} w={:?}", self.shape, weight.shape)));
        }
        let oc = og * groups;
        #[cfg(feature = "cuda")]
        if let (Some(x), Some(w)) = (self.dev()?, weight.dev()?) {
            let (y, ys) = super::conv::cudnn_conv_transpose(
                &x, &[n, c, 1, l], &w, &[c, og, 1, k], &[0, padding], &[1, stride], &[1, dilation], groups, &[0, output_padding],
            )
            .map_err(|e| msg(e.to_string()))?;
            let y = Self::from_dev_result(y, vec![n, oc, ys[3]])?;
            return match bias {
                Some(b) => y.add_bias(b, 1),
                None => Ok(y),
            };
        }
        let (y, lo) = host::conv_transpose1d(&self.host_cow()?, (n, c, l), &weight.host_cow()?, (og, k), padding, stride, dilation, groups, output_padding);
        let y = Self::host_only(y, vec![n, oc, lo]);
        match bias {
            Some(b) => y.add_bias(b, 1),
            None => Ok(y),
        }
    }

    fn dims3(&self, what: &str) -> Result<[usize; 3]> {
        match self.shape[..] {
            [a, b, c] => Ok([a, b, c]),
            _ => Err(msg(format!("{what} must be rank 3, got {:?}", self.shape))),
        }
    }

    pub fn conv2d(&self, weight: &CudaTensor, bias: Option<&CudaTensor>, padding: usize, stride: usize) -> Result<CudaTensor> {
        self.conv_nd(weight, bias, &[padding, padding], &[stride.max(1), stride.max(1)])
    }

    /// Cross-correlation of NCDHW `self` with OIDHW `weight`, symmetric
    /// zero padding `pad` per spatial axis (causal time padding is the caller's).
    pub fn conv3d(&self, weight: &CudaTensor, bias: Option<&CudaTensor>, pad: [usize; 3], stride: [usize; 3]) -> Result<CudaTensor> {
        self.conv_nd(weight, bias, &pad, &stride)
    }

    fn conv_nd(&self, weight: &CudaTensor, bias: Option<&CudaTensor>, pad: &[usize], stride: &[usize]) -> Result<CudaTensor> {
        let spatial = self.rank().saturating_sub(2);
        if !(2..=3).contains(&spatial) || weight.rank() != self.rank() || weight.shape[1] != self.shape[1] {
            return Err(msg(format!("conv shapes: x={:?} w={:?}", self.shape, weight.shape)));
        }
        #[cfg(feature = "cuda")]
        let one_by_one = weight.shape[2..].iter().all(|&k| k == 1) && pad.iter().all(|&p| p == 0) && stride.iter().all(|&s| s == 1);
        let (n, c, oc) = (self.shape[0], self.shape[1], weight.shape[0]);
        #[cfg(feature = "cuda")]
        if let (Some(x), Some(w)) = (self.dev()?, weight.dev()?) {
            let (y, y_shape) = if one_by_one {
                let s = numel(&self.shape[2..]);
                let mut y = super::ops::alloc((n * oc * s).max(1))?;
                super::device::matmul_shared_left(&w, &x, &mut y, n, oc, c, s).map_err(|e| msg(e.to_string()))?;
                let mut shape = self.shape.clone();
                shape[1] = oc;
                (y, shape)
            } else if spatial == 3 {
                super::conv::conv3d(&x, &self.shape, &w, &weight.shape, [pad[0], pad[1], pad[2]], [stride[0], stride[1], stride[2]])
                    .map_err(|e| msg(e.to_string()))?
            } else {
                super::conv::cudnn_conv(&x, &self.shape, &w, &weight.shape, pad, stride).map_err(|e| msg(e.to_string()))?
            };
            let y = Self::from_dev_result(y, y_shape)?;
            return match bias {
                Some(b) => y.add_bias(b, 1),
                None => Ok(y),
            };
        }
        let x = self.host_cow()?;
        let w = weight.host_cow()?;
        let (data, shape) = if spatial == 2 {
            let (h, wd) = (self.shape[2], self.shape[3]);
            let (kh, kw) = (weight.shape[2], weight.shape[3]);
            let (y, oh, ow) = host::conv2d(&x, &w, n, c, h, wd, oc, kh, kw, [pad[0], pad[1]], [stride[0], stride[1]]);
            (y, vec![n, oc, oh, ow])
        } else {
            // Reference 3-D conv: pad time explicitly, unfold, conv2d, permute back.
            let padded = if pad[0] > 0 { self.pad_zeros(2, pad[0], pad[0])? } else { self.clone() };
            let px = padded.host_cow()?;
            let (t, h, wd) = (padded.shape[2], padded.shape[3], padded.shape[4]);
            let (kt, kh, kw) = (weight.shape[2], weight.shape[3], weight.shape[4]);
            let (unfolded, ot) = host::temporal_unfold(&px, n, c, t, h, wd, kt, stride[0]);
            let (y, oh, ow) = host::conv2d(&unfolded, &w, n * ot, c * kt, h, wd, oc, kh, kw, [pad[1], pad[2]], [stride[1], stride[2]]);
            (host::permute(&y, &[n, ot, oc, oh, ow], &[0, 2, 1, 3, 4]), vec![n, oc, ot, oh, ow])
        };
        let y = Self::host_only(data, shape);
        match bias {
            Some(b) => y.add_bias(b, 1),
            None => Ok(y),
        }
    }

    pub fn upsample_nearest2d(&self, out_h: usize, out_w: usize) -> Result<CudaTensor> {
        if self.rank() != 4 {
            return Err(msg("upsample_nearest2d NCHW"));
        }
        let (n, c, h, w) = (self.shape[0], self.shape[1], self.shape[2], self.shape[3]);
        if out_h % h != 0 || out_w % w != 0 {
            return Err(msg(format!("upsample_nearest2d needs integer factors: {h}x{w} -> {out_h}x{out_w}")));
        }
        let (fy, fx) = (out_h / h, out_w / w);
        let out_shape = vec![n, c, out_h, out_w];
        #[cfg(feature = "cuda")]
        if let Some(x) = self.dev()? {
            return Self::from_dev_result(super::ops::upsample_nearest_device(&x, n * c, h, w, fy, fx)?, out_shape);
        }
        Ok(Self::host_only(host::upsample_nearest(&self.host_cow()?, n * c, h, w, fy, fx), out_shape))
    }

    /// Embedding lookup for a table kept in host memory: gather the rows on
    /// the host and upload only them. This is input preparation (a prompt's
    /// few hundred rows out of a 250k-row UMT5 vocabulary), so a 4 GB table
    /// never has to occupy device memory. A device table uses the kernel.
    pub fn embedding_rows(&self, indices: &[usize]) -> Result<CudaTensor> {
        #[cfg(feature = "cuda")]
        if self.device.is_some() {
            return self.index_select_rows(indices);
        }
        if self.rank() != 2 {
            return Err(msg("embedding_rows expects [V, D]"));
        }
        let (v, d) = (self.shape[0], self.shape[1]);
        let mut data = Vec::with_capacity(indices.len() * d);
        for &i in indices {
            if i >= v {
                return Err(msg(format!("index {i} out of range {v}")));
            }
            data.extend_from_slice(&self.data[i * d..(i + 1) * d]);
        }
        Self::host_only(data, vec![indices.len(), d]).to_device()
    }

    /// Rows of a `[V, D]` table.
    pub fn index_select_rows(&self, indices: &[usize]) -> Result<CudaTensor> {
        if self.rank() != 2 {
            return Err(msg("index_select_rows expects 2D"));
        }
        let (v, d) = (self.shape[0], self.shape[1]);
        if let Some(&bad) = indices.iter().find(|&&i| i >= v) {
            return Err(msg(format!("index {bad} out of range {v}")));
        }
        let out_shape = vec![indices.len(), d];
        #[cfg(feature = "cuda")]
        if let Some(table) = self.dev()? {
            let idx: Vec<u32> = indices.iter().map(|&i| i as u32).collect();
            return Self::from_dev_result(super::ops::index_select_rows_device(&table, d, &idx)?, out_shape);
        }
        let host = self.host_cow()?;
        let mut data = Vec::with_capacity(indices.len() * d);
        for &i in indices {
            data.extend_from_slice(&host[i * d..(i + 1) * d]);
        }
        Ok(Self::host_only(data, out_shape))
    }
}

#[cfg(feature = "cuda")]
fn bias_shape(rank: usize, dim: usize, c: usize) -> Vec<usize> {
    let mut s = vec![1; rank];
    s[dim] = c;
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    fn t(data: Vec<f32>, shape: &[usize]) -> CudaTensor {
        CudaTensor::from_vec(data, shape.to_vec()).unwrap()
    }

    fn seq(n: usize) -> Vec<f32> {
        (0..n).map(|i| i as f32 * 0.5 - 3.0).collect()
    }

    #[test]
    fn matmul_2d() {
        let a = t(vec![1., 2., 3., 4.], &[2, 2]);
        let b = t(vec![5., 6., 7., 8.], &[2, 2]);
        assert_eq!(a.matmul(&b).unwrap().data, vec![19., 22., 43., 50.]);
    }

    #[test]
    fn repeat_plan_shapes() {
        assert_eq!(repeat_plan(&[2, 6, 4], &[1, 6, 4]), Some((1, 24)));
        assert_eq!(repeat_plan(&[2, 3, 5, 7], &[1, 3, 1, 1]), Some((35, 3)));
        assert_eq!(repeat_plan(&[2, 3, 5], &[5]), Some((1, 5)));
        assert_eq!(repeat_plan(&[2, 3, 5], &[1]), Some((1, 1)));
        assert_eq!(repeat_plan(&[2, 3, 5, 7], &[1, 3, 1, 7]), None);
    }

    #[test]
    fn broadcast_matches_generic() {
        let a = t(seq(2 * 3 * 4), &[2, 3, 4]);
        let b = t(vec![1.0, -2.0, 0.5], &[1, 3, 1]);
        let fast = a.add(&b).unwrap();
        let rsub = b.sub(&a).unwrap();
        for i in 0..24 {
            let bi = (i / 4) % 3;
            assert_eq!(fast.data[i], a.data[i] + b.data[bi]);
            assert_eq!(rsub.data[i], b.data[bi] - a.data[i]);
        }
        // Non-contiguous broadcast goes through the generic path.
        let c = t(vec![1.0, 2.0], &[2, 1, 1]);
        let d = t(vec![10.0, 20.0, 30.0, 40.0], &[1, 1, 4]);
        let g = c.mul(&d).unwrap();
        assert_eq!(g.shape, vec![2, 1, 4]);
        assert_eq!(g.data, vec![10.0, 20.0, 30.0, 40.0, 20.0, 40.0, 60.0, 80.0]);
    }

    #[test]
    fn narrow_cat_pad_any_dim_roundtrip() {
        let x = t(seq(2 * 3 * 4 * 5), &[2, 3, 4, 5]);
        for dim in 0..4 {
            let d = x.shape[dim];
            let a = x.narrow(dim, 0, 1).unwrap();
            let b = x.narrow(dim, 1, d - 1).unwrap();
            let back = CudaTensor::cat(&[&a, &b], dim).unwrap();
            assert_eq!(back.data, x.data, "dim {dim}");
            let p = x.pad_zeros(dim, 2, 1).unwrap();
            assert_eq!(p.narrow(dim, 2, d).unwrap().data, x.data);
            assert!(p.narrow(dim, 0, 2).unwrap().data.iter().all(|&v| v == 0.0));
        }
    }

    #[test]
    fn permute_matches_index_math() {
        let x = t(seq(2 * 3 * 4 * 5 * 2), &[2, 3, 4, 5, 2]);
        let perm = [0, 2, 1, 4, 3];
        let y = x.permute(&perm).unwrap();
        assert_eq!(y.shape, vec![2, 4, 3, 2, 5]);
        let s = strides(&x.shape);
        let so = strides(&y.shape);
        for (i, &v) in y.data.iter().enumerate() {
            let mut rem = i;
            let mut src = 0;
            for k in 0..5 {
                src += (rem / so[k]) * s[perm[k]];
                rem %= so[k];
            }
            assert_eq!(v, x.data[src]);
        }
    }

    #[test]
    fn lincomb_matches_formula() {
        let a = t(vec![1.0, 2.0], &[2]);
        let b = t(vec![3.0, -1.0], &[2]);
        let c = t(vec![0.5, 0.5], &[2]);
        let d = t(vec![2.0, 4.0], &[2]);
        let y = CudaTensor::lincomb(&[(2.0, &a), (-1.0, &b), (4.0, &c), (0.25, &d)]).unwrap();
        assert_eq!(y.data, vec![2.0 - 3.0 + 2.0 + 0.5, 4.0 + 1.0 + 2.0 + 1.0]);
    }

    #[test]
    fn conv3d_reference_matches_naive() {
        let (n, c, tt, h, w, oc) = (1usize, 2usize, 4usize, 5usize, 6usize, 3usize);
        let x = t(seq(n * c * tt * h * w), &[n, c, tt, h, w]);
        let wt = t((0..oc * c * 27).map(|i| ((i * 7) % 11) as f32 * 0.1 - 0.5).collect(), &[oc, c, 3, 3, 3]);
        let bias = t(vec![0.1, -0.2, 0.3], &[oc]);
        let y = x.conv3d(&wt, Some(&bias), [0, 1, 1], [1, 1, 1]).unwrap();
        assert_eq!(y.shape, vec![n, oc, tt - 2, h, w]);
        for o in 0..oc {
            for ot in 0..tt - 2 {
                for yy in 0..h {
                    for xx in 0..w {
                        let mut acc = bias.data[o];
                        for ci in 0..c {
                            for dt in 0..3 {
                                for dy in 0..3 {
                                    for dx in 0..3 {
                                        let (iy, ix) = (yy + dy, xx + dx);
                                        if iy < 1 || ix < 1 || iy > h || ix > w {
                                            continue;
                                        }
                                        acc += x.data[((ci * tt + ot + dt) * h + iy - 1) * w + ix - 1]
                                            * wt.data[((o * c + ci) * 3 + dt) * 9 + dy * 3 + dx];
                                    }
                                }
                            }
                        }
                        let got = y.data[((o * (tt - 2) + ot) * h + yy) * w + xx];
                        assert!((got - acc).abs() < 1e-4, "{got} vs {acc}");
                    }
                }
            }
        }
    }

    #[test]
    fn host_mut_and_pin_without_device() {
        let mut x = t(vec![1.0, 2.0], &[2]);
        x.pin_device().unwrap();
        x.host_mut().unwrap()[0] = 5.0;
        assert_eq!(x.host_cow().unwrap().as_ref(), &[5.0, 2.0]);
    }

    #[test]
    fn softmax_non_last_axis() {
        let x = t(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let y = x.softmax(0).unwrap();
        let e = (2.0f32).exp();
        let p = 1.0 / (1.0 + e);
        assert!((y.data[0] - p).abs() < 1e-6 && (y.data[2] - (1.0 - p)).abs() < 1e-6);
    }
}

#[cfg(test)]
mod encoder_op_tests {
    use super::*;

    #[test]
    fn erf_matches_known_values() {
        for (x, want) in [
            (0.0, 0.0),
            (0.1, 0.112_462_916_018_284_9),
            (0.5, 0.520_499_877_813_046_5),
            (1.0, 0.842_700_792_949_714_9),
            (2.0, 0.995_322_265_018_952_7),
            (3.5, 0.999_999_256_901_627_7),
            (5.0, 0.999_999_999_998_462_5),
        ] {
            assert!((host::erf(x) - want).abs() < 1e-14, "erf({x}) = {}", host::erf(x));
            assert!((host::erf(-x) + want).abs() < 1e-14, "erf is odd");
        }
        // Phi(1) * 1: the value every GELU table quotes.
        assert!((f64::from(host::gelu_erf(1.0)) - 0.841_344_746_068_542_9).abs() < 1e-7);
        // And it is not the tanh approximation.
        assert!((host::gelu_erf(1.0) - host::gelu_tanh(1.0)).abs() > 1e-5);
    }

    /// HF `apply_rotary_pos_emb`: x * cos + rotate_half(x) * sin, tables built
    /// as cat(freqs, freqs). Written out per element, independently of the op.
    #[test]
    fn rope_half_matches_the_hf_formula_and_passes_the_tail_through() {
        let (b, h, s, d, r) = (2usize, 3usize, 5usize, 8usize, 6usize);
        let x: Vec<f32> = (0..b * h * s * d).map(|i| ((i * 7 % 23) as f32 - 11.0) / 5.0).collect();
        let half = r / 2;
        let (mut cos, mut sin) = (vec![0f32; s * r], vec![0f32; s * r]);
        for p in 0..s {
            for k in 0..half {
                let ang = (p as f32 + 2.0) * 10000f32.powf(-(2.0 * k as f32) / r as f32);
                for j in [k, k + half] {
                    cos[p * r + j] = ang.cos();
                    sin[p * r + j] = ang.sin();
                }
            }
        }
        let xt = CudaTensor::from_vec(x.clone(), vec![b, h, s, d]).unwrap();
        let out = xt
            .rope_half(
                &CudaTensor::from_vec(cos.clone(), vec![s, r]).unwrap(),
                &CudaTensor::from_vec(sin.clone(), vec![s, r]).unwrap(),
            )
            .unwrap();
        let got = out.host_cow().unwrap();
        for bi in 0..b * h {
            for p in 0..s {
                let base = (bi * s + p) * d;
                for j in 0..d {
                    let want = if j >= r {
                        x[base + j]
                    } else {
                        let rot = if j < half { -x[base + j + half] } else { x[base + j - half] };
                        x[base + j] * cos[p * r + j] + rot * sin[p * r + j]
                    };
                    assert!((got[base + j] - want).abs() < 1e-6, "b{bi} p{p} j{j}");
                }
            }
        }
        // A rotation preserves the norm of the rotated channels.
        let n0: f32 = x[..r].iter().map(|v| v * v).sum();
        let n1: f32 = got[..r].iter().map(|v| v * v).sum();
        assert!((n0 - n1).abs() < 1e-4);
        assert!(xt.rope_half(&CudaTensor::zeros(&[s, 5]), &CudaTensor::zeros(&[s, 5])).is_err(), "odd R");
    }

    #[test]
    fn repeat_kv_is_repeat_interleave_on_heads() {
        let (b, hkv, s, d, rep) = (2usize, 2usize, 3usize, 2usize, 4usize);
        let x: Vec<f32> = (0..b * hkv * s * d).map(|i| i as f32).collect();
        let out = CudaTensor::from_vec(x.clone(), vec![b, hkv, s, d]).unwrap().repeat_kv(rep).unwrap();
        assert_eq!(out.shape, vec![b, hkv * rep, s, d]);
        let got = out.host_cow().unwrap();
        for bi in 0..b {
            for ho in 0..hkv * rep {
                for k in 0..s * d {
                    let want = x[(bi * hkv + ho / rep) * s * d + k];
                    assert_eq!(got[(bi * hkv * rep + ho) * s * d + k], want);
                }
            }
        }
    }

    #[test]
    fn snake_and_leaky_relu() {
        let (n, c, l) = (2usize, 3usize, 4usize);
        let x: Vec<f32> = (0..n * c * l).map(|i| (i as f32 - 10.0) / 4.0).collect();
        let alpha = [0.5f32, 1.0, 2.0];
        let inv_beta = [2.0f32, 1.0, 0.25];
        let out = CudaTensor::from_vec(x.clone(), vec![n, c, l])
            .unwrap()
            .snake_beta(
                &CudaTensor::from_vec(alpha.to_vec(), vec![c]).unwrap(),
                &CudaTensor::from_vec(inv_beta.to_vec(), vec![c]).unwrap(),
            )
            .unwrap();
        for (i, (&g, &v)) in out.host_cow().unwrap().iter().zip(&x).enumerate() {
            let ch = (i / l) % c;
            let want = v + inv_beta[ch] * (alpha[ch] * v).sin().powi(2);
            assert!((g - want).abs() < 1e-6);
        }
        let lr = CudaTensor::from_vec(vec![-2.0, 0.0, 3.0], vec![3]).unwrap().leaky_relu(0.1);
        assert_eq!(&*lr.host_cow().unwrap(), &[-0.2, 0.0, 3.0]);
    }
}

#[cfg(test)]
mod conv1d_tests {
    use super::*;

    fn seq(n: usize, mul: usize, modulo: usize) -> Vec<f32> {
        (0..n).map(|i| ((i * mul) % modulo) as f32 / modulo as f32 - 0.4).collect()
    }

    fn dot(a: &[f32], b: &[f32]) -> f64 {
        a.iter().zip(b).map(|(x, y)| f64::from(*x) * f64::from(*y)).sum()
    }

    /// PyTorch's documented example shapes, worked by hand.
    #[test]
    fn small_cases_by_hand() {
        let x = CudaTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0], vec![1, 1, 5]).unwrap();
        let w = CudaTensor::from_vec(vec![1.0, 0.0, -1.0], vec![1, 1, 3]).unwrap();
        // Same padding, dilation 2: reach 5, pad 2.
        let y = x.conv1d(&w, None, 2, 1, 2, 1).unwrap();
        assert_eq!(y.shape, vec![1, 1, 5]);
        assert_eq!(&*y.host_cow().unwrap(), &[-3.0, -4.0, -4.0, 2.0, 3.0]);
        // Stride 2, no padding.
        let y = x.conv1d(&w, None, 0, 2, 1, 1).unwrap();
        assert_eq!(&*y.host_cow().unwrap(), &[-2.0, -2.0]);

        // Transposed, stride 2, kernel [1, 1]: every sample held for two.
        let x3 = CudaTensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 1, 3]).unwrap();
        let w2 = CudaTensor::from_vec(vec![1.0, 1.0], vec![1, 1, 2]).unwrap();
        let bias = CudaTensor::from_vec(vec![0.5], vec![1]).unwrap();
        let y = x3.conv_transpose1d(&w2, Some(&bias), 0, 2, 1, 1, 0).unwrap();
        assert_eq!(&*y.host_cow().unwrap(), &[1.5, 1.5, 2.5, 2.5, 3.5, 3.5]);
        // output_padding appends a position no input reaches: bias only.
        let y = x3.conv_transpose1d(&w2, Some(&bias), 0, 2, 1, 1, 1).unwrap();
        assert_eq!(y.shape, vec![1, 1, 7]);
        assert_eq!(y.host_cow().unwrap()[6], 0.5);
        // HiFi-GAN style upsampler: kernel 4, stride 2, padding 1 doubles the length.
        let w4 = CudaTensor::from_vec(vec![1.0; 4], vec![1, 1, 4]).unwrap();
        assert_eq!(x3.conv_transpose1d(&w4, None, 1, 2, 1, 1, 0).unwrap().shape, vec![1, 1, 6]);
    }

    /// A transposed convolution is the adjoint of the convolution with the same
    /// weight: <conv(x), y> = <x, conv_T(y)>. Holds for every stride, dilation,
    /// padding and grouping, and neither side is written in terms of the other.
    #[test]
    fn transpose_is_the_adjoint_of_conv() {
        for (c, oc, k, pad, stride, dil, groups, l) in [
            (4usize, 6usize, 3usize, 1usize, 1usize, 1usize, 1usize, 9usize),
            (4, 6, 4, 1, 2, 1, 2, 10),
            (6, 6, 5, 4, 1, 2, 6, 11),
            (3, 9, 7, 3, 3, 1, 3, 13),
        ] {
            let lo = (l + 2 * pad - dil * (k - 1) - 1) / stride + 1;
            // Pick the output_padding that makes conv_T map length lo back to l.
            let out_pad = l - ((lo - 1) * stride + dil * (k - 1) + 1 - 2 * pad);
            let x = CudaTensor::from_vec(seq(2 * c * l, 7, 31), vec![2, c, l]).unwrap();
            let w = CudaTensor::from_vec(seq(oc * (c / groups) * k, 11, 29), vec![oc, c / groups, k]).unwrap();
            let y = CudaTensor::from_vec(seq(2 * oc * lo, 13, 37), vec![2, oc, lo]).unwrap();
            let ax = x.conv1d(&w, None, pad, stride, dil, groups).unwrap();
            assert_eq!(ax.shape, vec![2, oc, lo]);
            let aty = y.conv_transpose1d(&w, None, pad, stride, dil, groups, out_pad).unwrap();
            assert_eq!(aty.shape, vec![2, c, l]);
            let lhs = dot(&ax.host_cow().unwrap(), &y.host_cow().unwrap());
            let rhs = dot(&x.host_cow().unwrap(), &aty.host_cow().unwrap());
            assert!((lhs - rhs).abs() < 1e-4 * lhs.abs().max(1.0), "c={c} oc={oc} k={k} g={groups}: {lhs} vs {rhs}");
        }
    }

    #[test]
    fn bad_shapes_are_errors() {
        let x = CudaTensor::zeros(&[1, 4, 8]);
        assert!(x.conv1d(&CudaTensor::zeros(&[6, 3, 3]), None, 1, 1, 1, 1).is_err(), "channel mismatch");
        assert!(x.conv1d(&CudaTensor::zeros(&[6, 2, 3]), None, 1, 1, 1, 4).is_err(), "groups do not divide");
        assert!(x.conv1d(&CudaTensor::zeros(&[6, 4, 3]), None, 0, 1, 8, 1).is_err(), "reach exceeds input");
        assert!(x.conv_transpose1d(&CudaTensor::zeros(&[3, 2, 3]), None, 0, 1, 1, 1, 0).is_err());
    }
}
