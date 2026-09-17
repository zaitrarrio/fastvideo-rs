//! Owned f32 N-D tensors for the Luminal Wan tiny graph.
//!
//! crates.io `luminal` 0.2 uses const-generic shapes; variable T/H/W Wan graphs
//! run on these host tensors until a Graph pin is available.

use thiserror::Error;

#[derive(Debug, Error)]
pub enum TensorError {
    #[error("{0}")]
    Message(String),
}

pub type Result<T> = std::result::Result<T, TensorError>;

#[derive(Debug, Clone)]
pub struct NdTensor {
    pub data: Vec<f32>,
    pub shape: Vec<usize>,
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

impl NdTensor {
    pub fn new(data: Vec<f32>, shape: Vec<usize>) -> Result<Self> {
        check_numel(&data, &shape)?;
        Ok(Self { data, shape })
    }

    pub fn zeros(shape: &[usize]) -> Self {
        Self {
            data: vec![0.0; numel(shape)],
            shape: shape.to_vec(),
        }
    }

    pub fn ones(shape: &[usize]) -> Self {
        Self {
            data: vec![1.0; numel(shape)],
            shape: shape.to_vec(),
        }
    }

    pub fn from_vec(data: Vec<f32>, shape: Vec<usize>) -> Result<Self> {
        Self::new(data, shape)
    }

    pub fn from_slice(data: &[f32], shape: &[usize]) -> Result<Self> {
        Self::new(data.to_vec(), shape.to_vec())
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

    pub fn reshape(&self, shape: Vec<usize>) -> Result<NdTensor> {
        check_numel(&self.data, &shape)?;
        Ok(NdTensor {
            data: self.data.clone(),
            shape,
        })
    }

    pub fn reshape_owned(self, shape: Vec<usize>) -> Result<NdTensor> {
        check_numel(&self.data, &shape)?;
        Ok(NdTensor {
            data: self.data,
            shape,
        })
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

    pub fn permute(&self, dims: &[usize]) -> Result<NdTensor> {
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
        let mut out = vec![0.0; self.data.len()];
        let in_strides = strides(&self.shape);
        let out_strides = strides(&out_shape);
        for out_idx in 0..out.len() {
            let out_coord = unravel(out_idx, &out_strides);
            let mut in_coord = vec![0usize; self.rank()];
            for (o, &d) in dims.iter().enumerate() {
                in_coord[d] = out_coord[o];
            }
            let in_idx = ravel(&in_coord, &in_strides);
            out[out_idx] = self.data[in_idx];
        }
        Ok(NdTensor {
            data: out,
            shape: out_shape,
        })
    }

    pub fn transpose(&self, dim0: usize, dim1: usize) -> Result<NdTensor> {
        let mut dims: Vec<usize> = (0..self.rank()).collect();
        dims.swap(dim0, dim1);
        self.permute(&dims)
    }

    pub fn narrow(&self, dim: usize, start: usize, len: usize) -> Result<NdTensor> {
        let d = self.dim(dim)?;
        if start + len > d {
            return Err(TensorError::Message(format!(
                "narrow dim={dim} start={start} len={len} of {d}"
            )));
        }
        let mut out_shape = self.shape.clone();
        out_shape[dim] = len;
        let in_strides = strides(&self.shape);
        let out_strides = strides(&out_shape);
        let mut out = vec![0.0; numel(&out_shape)];
        for out_idx in 0..out.len() {
            let mut coord = unravel(out_idx, &out_strides);
            coord[dim] += start;
            out[out_idx] = self.data[ravel(&coord, &in_strides)];
        }
        Ok(NdTensor {
            data: out,
            shape: out_shape,
        })
    }

    pub fn squeeze(&self, dim: usize) -> Result<NdTensor> {
        if self.dim(dim)? != 1 {
            return Err(TensorError::Message(format!(
                "squeeze expected size 1 at {dim}, got {}",
                self.dim(dim)?
            )));
        }
        let mut shape = self.shape.clone();
        shape.remove(dim);
        Ok(NdTensor {
            data: self.data.clone(),
            shape,
        })
    }

    pub fn unsqueeze(&self, dim: usize) -> Result<NdTensor> {
        if dim > self.rank() {
            return Err(TensorError::Message("unsqueeze out of range".into()));
        }
        let mut shape = self.shape.clone();
        shape.insert(dim, 1);
        Ok(NdTensor {
            data: self.data.clone(),
            shape,
        })
    }

    pub fn cat(tensors: &[&NdTensor], dim: usize) -> Result<NdTensor> {
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
        let mut out = vec![0.0; numel(&out_shape)];
        let out_strides = strides(&out_shape);
        let mut offset = 0usize;
        for t in tensors {
            let in_strides = strides(&t.shape);
            for in_idx in 0..t.data.len() {
                let mut coord = unravel(in_idx, &in_strides);
                coord[dim] += offset;
                out[ravel(&coord, &out_strides)] = t.data[in_idx];
            }
            offset += t.shape[dim];
        }
        Ok(NdTensor {
            data: out,
            shape: out_shape,
        })
    }

    pub fn pad_zeros(&self, dim: usize, left: usize, right: usize) -> Result<NdTensor> {
        let mut out_shape = self.shape.clone();
        out_shape[dim] += left + right;
        let mut out = vec![0.0; numel(&out_shape)];
        let in_strides = strides(&self.shape);
        let out_strides = strides(&out_shape);
        for in_idx in 0..self.data.len() {
            let mut coord = unravel(in_idx, &in_strides);
            coord[dim] += left;
            out[ravel(&coord, &out_strides)] = self.data[in_idx];
        }
        Ok(NdTensor {
            data: out,
            shape: out_shape,
        })
    }

    pub fn add(&self, other: &NdTensor) -> Result<NdTensor> {
        broadcast_bin(self, other, |a, b| a + b)
    }

    pub fn sub(&self, other: &NdTensor) -> Result<NdTensor> {
        broadcast_bin(self, other, |a, b| a - b)
    }

    pub fn mul(&self, other: &NdTensor) -> Result<NdTensor> {
        broadcast_bin(self, other, |a, b| a * b)
    }

    pub fn div(&self, other: &NdTensor) -> Result<NdTensor> {
        broadcast_bin(self, other, |a, b| a / b)
    }

    pub fn add_scalar(&self, s: f32) -> NdTensor {
        NdTensor {
            data: self.data.iter().map(|x| x + s).collect(),
            shape: self.shape.clone(),
        }
    }

    pub fn mul_scalar(&self, s: f32) -> NdTensor {
        NdTensor {
            data: self.data.iter().map(|x| x * s).collect(),
            shape: self.shape.clone(),
        }
    }

    pub fn clamp(&self, min: f32, max: f32) -> NdTensor {
        NdTensor {
            data: self.data.iter().map(|x| x.clamp(min, max)).collect(),
            shape: self.shape.clone(),
        }
    }

    pub fn sqrt(&self) -> NdTensor {
        NdTensor {
            data: self.data.iter().map(|x| x.sqrt()).collect(),
            shape: self.shape.clone(),
        }
    }

    pub fn sqr(&self) -> NdTensor {
        NdTensor {
            data: self.data.iter().map(|x| x * x).collect(),
            shape: self.shape.clone(),
        }
    }

    pub fn mean_keepdim(&self, dim: isize) -> Result<NdTensor> {
        let axis = self.axis(dim)?;
        let mut out_shape = self.shape.clone();
        out_shape[axis] = 1;
        let mut out = vec![0.0; numel(&out_shape)];
        let counts = vec![0usize; out.len()];
        let mut counts = counts;
        let in_strides = strides(&self.shape);
        let out_strides = strides(&out_shape);
        for in_idx in 0..self.data.len() {
            let mut coord = unravel(in_idx, &in_strides);
            coord[axis] = 0;
            let oi = ravel(&coord, &out_strides);
            out[oi] += self.data[in_idx];
            counts[oi] += 1;
        }
        for (o, c) in out.iter_mut().zip(counts) {
            *o /= c as f32;
        }
        Ok(NdTensor {
            data: out,
            shape: out_shape,
        })
    }

    pub fn matmul(&self, other: &NdTensor) -> Result<NdTensor> {
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
        let mut out = vec![0.0; batch * m * n];
        for bi in 0..batch {
            let a_bi = batch_index(bi, &batch_shape, a_batch);
            let b_bi = batch_index(bi, &batch_shape, b_batch);
            let a_off = a_bi * m * k;
            let b_off = b_bi * k * n;
            let o_off = bi * m * n;
            for i in 0..m {
                for j in 0..n {
                    let mut acc = 0.0;
                    for t in 0..k {
                        acc += self.data[a_off + i * k + t] * other.data[b_off + t * n + j];
                    }
                    out[o_off + i * n + j] = acc;
                }
            }
        }
        let mut out_shape = batch_shape;
        out_shape.push(m);
        out_shape.push(n);
        Ok(NdTensor {
            data: out,
            shape: out_shape,
        })
    }

    pub fn softmax(&self, dim: isize) -> Result<NdTensor> {
        let axis = self.axis(dim)?;
        let axis_len = self.shape[axis];
        let out_shape = self.shape.clone();
        let mut result = vec![0.0; self.data.len()];
        let in_strides = strides(&self.shape);
        let mut seen = vec![false; self.data.len()];
        for idx in 0..self.data.len() {
            if seen[idx] {
                continue;
            }
            let mut coord = unravel(idx, &in_strides);
            let mut vals = Vec::with_capacity(axis_len);
            let mut indices = Vec::with_capacity(axis_len);
            for a in 0..axis_len {
                coord[axis] = a;
                let i = ravel(&coord, &in_strides);
                vals.push(self.data[i]);
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
        Ok(NdTensor {
            data: result,
            shape: out_shape,
        })
    }

    pub fn silu(&self) -> NdTensor {
        NdTensor {
            data: self
                .data
                .iter()
                .map(|&x| x / (1.0 + (-x).exp()))
                .collect(),
            shape: self.shape.clone(),
        }
    }

    pub fn gelu_tanh(&self) -> NdTensor {
        let c = (2.0 / std::f32::consts::PI).sqrt();
        NdTensor {
            data: self
                .data
                .iter()
                .map(|&x| {
                    let inner = c * (x + 0.044715 * x * x * x);
                    0.5 * x * (1.0 + inner.tanh())
                })
                .collect(),
            shape: self.shape.clone(),
        }
    }

    pub fn rms_norm(&self, weight: &NdTensor, eps: f32) -> Result<NdTensor> {
        // Normalize over last dim.
        let axis = self.rank() - 1;
        let axis_len = self.shape[axis];
        if weight.data.len() != axis_len {
            return Err(TensorError::Message("rms_norm weight size".into()));
        }
        let mut out = vec![0.0; self.data.len()];
        let strides = strides(&self.shape);
        let outer = numel(&self.shape) / axis_len;
        for o in 0..outer {
            let base = o * axis_len;
            // Contiguous last-dim assumption when strides make last dim contiguous.
            let mut mean_sq = 0.0;
            for a in 0..axis_len {
                let v = self.data[base + a];
                mean_sq += v * v;
            }
            mean_sq /= axis_len as f32;
            let inv = 1.0 / (mean_sq + eps).sqrt();
            for a in 0..axis_len {
                out[base + a] = self.data[base + a] * inv * weight.data[a];
            }
        }
        let _ = strides;
        Ok(NdTensor {
            data: out,
            shape: self.shape.clone(),
        })
    }

    pub fn layer_norm(
        &self,
        eps: f32,
        weight: Option<&NdTensor>,
        bias: Option<&NdTensor>,
    ) -> Result<NdTensor> {
        let axis = self.rank() - 1;
        let axis_len = self.shape[axis];
        let mut out = vec![0.0; self.data.len()];
        let outer = numel(&self.shape) / axis_len;
        for o in 0..outer {
            let base = o * axis_len;
            let mut mean = 0.0;
            for a in 0..axis_len {
                mean += self.data[base + a];
            }
            mean /= axis_len as f32;
            let mut var = 0.0;
            for a in 0..axis_len {
                let d = self.data[base + a] - mean;
                var += d * d;
            }
            var /= axis_len as f32;
            let inv = 1.0 / (var + eps).sqrt();
            for a in 0..axis_len {
                let mut y = (self.data[base + a] - mean) * inv;
                if let Some(w) = weight {
                    y *= w.data[a];
                }
                if let Some(b) = bias {
                    y += b.data[a];
                }
                out[base + a] = y;
            }
        }
        Ok(NdTensor {
            data: out,
            shape: self.shape.clone(),
        })
    }

    /// NCHW conv2d with square/rectangular kernel, padding, stride, dilation=1, groups=1.
    pub fn conv2d(
        &self,
        weight: &NdTensor,
        bias: Option<&NdTensor>,
        padding: usize,
        stride: usize,
    ) -> Result<NdTensor> {
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
                                    let xv = self.data
                                        [((ni * c_in + ic) * h + ih) * w + iw];
                                    let wv = weight.data
                                        [((oc * c_in + ic) * kh + kh_i) * kw + kw_i];
                                    acc += xv * wv;
                                }
                            }
                        }
                        if let Some(b) = bias {
                            acc += b.data[oc];
                        }
                        out[((ni * c_out + oc) * out_h + oh) * out_w + ow] = acc;
                    }
                }
            }
        }
        Ok(NdTensor {
            data: out,
            shape: vec![n, c_out, out_h, out_w],
        })
    }

    pub fn upsample_nearest2d(&self, out_h: usize, out_w: usize) -> Result<NdTensor> {
        if self.rank() != 4 {
            return Err(TensorError::Message("upsample_nearest2d NCHW".into()));
        }
        let (n, c, h, w) = (self.shape[0], self.shape[1], self.shape[2], self.shape[3]);
        let mut out = vec![0.0; n * c * out_h * out_w];
        for ni in 0..n {
            for ci in 0..c {
                for oh in 0..out_h {
                    for ow in 0..out_w {
                        let ih = oh * h / out_h;
                        let iw = ow * w / out_w;
                        out[((ni * c + ci) * out_h + oh) * out_w + ow] =
                            self.data[((ni * c + ci) * h + ih) * w + iw];
                    }
                }
            }
        }
        Ok(NdTensor {
            data: out,
            shape: vec![n, c, out_h, out_w],
        })
    }

    pub fn chunk(&self, chunks: usize, dim: usize) -> Result<Vec<NdTensor>> {
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

    pub fn flatten_from(&self, dim: usize) -> Result<NdTensor> {
        let mut shape = self.shape[..dim].to_vec();
        shape.push(numel(&self.shape[dim..]));
        self.reshape(shape)
    }

    pub fn index_select_rows(&self, indices: &[usize]) -> Result<NdTensor> {
        // self: [V, D], gather rows
        if self.rank() != 2 {
            return Err(TensorError::Message("index_select_rows expects 2D".into()));
        }
        let d = self.shape[1];
        let mut data = Vec::with_capacity(indices.len() * d);
        for &i in indices {
            if i >= self.shape[0] {
                return Err(TensorError::Message("index OOB".into()));
            }
            data.extend_from_slice(&self.data[i * d..(i + 1) * d]);
        }
        Ok(NdTensor {
            data,
            shape: vec![indices.len(), d],
        })
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

fn broadcast_bin(a: &NdTensor, b: &NdTensor, op: impl Fn(f32, f32) -> f32) -> Result<NdTensor> {
    let shape = broadcast_shapes(&a.shape, &b.shape)?;
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
            a.data[0]
        } else {
            a.data[ravel(&a_coord, &a_strides)]
        };
        let bv = if b.rank() == 0 {
            b.data[0]
        } else {
            b.data[ravel(&b_coord, &b_strides)]
        };
        data[i] = op(av, bv);
    }
    Ok(NdTensor { data, shape })
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
        let a = NdTensor::from_vec(vec![1., 2., 3., 4.], vec![2, 2]).unwrap();
        let b = NdTensor::from_vec(vec![5., 6., 7., 8.], vec![2, 2]).unwrap();
        let c = a.matmul(&b).unwrap();
        assert_eq!(c.data, vec![19., 22., 43., 50.]);
    }

    #[test]
    fn gelu_silu_smoke() {
        let x = NdTensor::from_vec(vec![-2.0, 0.0, 1.5], vec![3]).unwrap();
        let s = x.silu();
        assert!((s.data[2] - 1.2263617).abs() < 1e-5);
    }
}
