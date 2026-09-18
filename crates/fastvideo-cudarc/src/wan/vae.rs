//! Wan VAE decode with Diffusers/Wan feat-cache (`CACHE_T=2`, Rep sentinel).

use fastvideo_models::wan::WanVaeConfig;

use super::tensor::{CudaTensor, Result, TensorError};
use super::weights::{self, WeightMap};

const CACHE_T: usize = 2;

/// Weights live on the device (a no-op on CPU runs). An upload failure here is
/// fatal: there is no host path to continue on.
fn pinned(mut t: CudaTensor) -> CudaTensor {
    t.pin_device().expect("upload VAE weight to device");
    t
}

/// A `[c, 1, 1(, 1)]` gamma as a pinned `[c]` vector.
fn gamma(map: &WeightMap, key: &str, shape: &[usize]) -> Result<CudaTensor> {
    Ok(pinned(weights::cuda_tensor_shaped(map, key, shape)?.reshape(vec![shape[0]])?))
}

const LATENTS_MEAN: [f32; 16] = [
    -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508, 0.4134, -0.0715, 0.5517,
    -0.3632, -0.1922, -0.9497, 0.2503, -0.2921,
];
const LATENTS_STD: [f32; 16] = [
    2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743, 3.2687, 2.1526, 2.8652, 1.5579,
    1.6382, 1.1253, 2.8251, 1.9160,
];

#[derive(Clone)]
enum CacheSlot {
    Empty,
    Rep,
    Tensor(CudaTensor),
}

struct FeatCache {
    slots: Vec<CacheSlot>,
    idx: usize,
}

impl FeatCache {
    fn new() -> Self {
        Self {
            slots: Vec::new(),
            idx: 0,
        }
    }

    fn begin_pass(&mut self) {
        self.idx = 0;
    }

    fn reserve(&mut self) -> usize {
        let i = self.idx;
        if i >= self.slots.len() {
            self.slots.push(CacheSlot::Empty);
        }
        self.idx += 1;
        i
    }
}

fn last_frames(x: &CudaTensor, n: usize) -> Result<CudaTensor> {
    let t = x.dim(2)?;
    let take = t.min(n);
    x.narrow(2, t - take, take)
}

#[derive(Debug, Clone)]
struct CausalConv3d {
    weight: CudaTensor, // [out, in, kt, kh, kw]
    bias: CudaTensor,
    stride: [usize; 3],
    pad: [usize; 3],
}

impl CausalConv3d {
    fn zeros(in_c: usize, out_c: usize, kernel: [usize; 3], stride: [usize; 3], pad: [usize; 3]) -> Self {
        Self {
            weight: pinned(CudaTensor::zeros(&[out_c, in_c, kernel[0], kernel[1], kernel[2]])),
            bias: pinned(CudaTensor::zeros(&[out_c])),
            stride,
            pad,
        }
    }

    fn load(
        map: &WeightMap,
        prefix: &str,
        in_c: usize,
        out_c: usize,
        kernel: [usize; 3],
        stride: [usize; 3],
        pad: [usize; 3],
    ) -> Result<Self> {
        Ok(Self {
            weight: pinned(weights::cuda_tensor_shaped(
                map,
                &weights::join_key(prefix, "weight"),
                &[out_c, in_c, kernel[0], kernel[1], kernel[2]],
            )?),
            bias: pinned(weights::cuda_tensor_shaped(map, &weights::join_key(prefix, "bias"), &[out_c])?),
            stride,
            pad,
        })
    }

    fn forward(&self, xs: &CudaTensor) -> Result<CudaTensor> {
        self.forward_with_cache(xs, None)
    }

    /// Causal time padding (`2*pad_t` frames in front, filled from the feat
    /// cache first), symmetric spatial padding inside the conv.
    fn forward_with_cache(&self, xs: &CudaTensor, cache_x: Option<&CudaTensor>) -> Result<CudaTensor> {
        let mut x = xs.clone();
        let mut pad_t = 2 * self.pad[0];
        if let Some(c) = cache_x {
            if pad_t > 0 {
                x = CudaTensor::cat(&[c, &x], 2)?;
                pad_t = pad_t.saturating_sub(c.dim(2)?);
            }
        }
        if pad_t > 0 {
            x = x.pad_zeros(2, pad_t, 0)?;
        }
        x.conv3d(&self.weight, Some(&self.bias), [0, self.pad[1], self.pad[2]], self.stride)
    }
}

fn conv_cached(
    conv: &CausalConv3d,
    x: &CudaTensor,
    cache: Option<&mut FeatCache>,
) -> Result<CudaTensor> {
    let Some(cache) = cache else {
        return conv.forward(x);
    };
    let i = cache.reserve();
    let prev = match cache.slots.get(i) {
        Some(CacheSlot::Tensor(t)) => Some(t.clone()),
        _ => None,
    };
    let mut cache_x = last_frames(x, CACHE_T)?;
    if cache_x.dim(2)? < CACHE_T {
        if let Some(prev) = &prev {
            let last = prev.narrow(2, prev.dim(2)? - 1, 1)?;
            cache_x = CudaTensor::cat(&[&last, &cache_x], 2)?;
        }
    }
    let y = conv.forward_with_cache(x, prev.as_ref())?;
    cache.slots[i] = CacheSlot::Tensor(cache_x);
    Ok(y)
}

fn double_time(x: CudaTensor) -> Result<CudaTensor> {
    let (b, c2, t, h, w) = (x.shape[0], x.shape[1], x.shape[2], x.shape[3], x.shape[4]);
    let c = c2 / 2;
    x.reshape(vec![b, 2, c, t, h, w])?
        .permute(&[0, 2, 3, 1, 4, 5])?
        .reshape(vec![b, c, t * 2, h, w])
}

fn rms_video(xs: &CudaTensor, gamma: &CudaTensor) -> Result<CudaTensor> {
    // Channel-first RMS over dim 1 (`F.normalize * sqrt(C) * gamma`).
    xs.rms_norm_channels(gamma, 1e-12)
}

/// `rms_video` with SiLU folded in, which is how the decoder always uses it.
fn rms_silu_video(xs: &CudaTensor, gamma: &CudaTensor) -> Result<CudaTensor> {
    xs.rms_norm_channels_act(gamma, 1e-12, true)
}

fn silu_video(xs: &CudaTensor) -> CudaTensor {
    xs.silu()
}

#[derive(Debug, Clone)]
struct ResidualBlock {
    norm1: CudaTensor,
    conv1: CausalConv3d,
    norm2: CudaTensor,
    conv2: CausalConv3d,
    shortcut: Option<CausalConv3d>,
}

impl ResidualBlock {
    fn zeros(in_dim: usize, out_dim: usize) -> Self {
        let shortcut = if in_dim != out_dim {
            Some(CausalConv3d::zeros(
                in_dim,
                out_dim,
                [1, 1, 1],
                [1, 1, 1],
                [0, 0, 0],
            ))
        } else {
            None
        };
        Self {
            norm1: pinned(CudaTensor::ones(&[in_dim])),
            conv1: CausalConv3d::zeros(in_dim, out_dim, [3, 3, 3], [1, 1, 1], [1, 1, 1]),
            norm2: pinned(CudaTensor::ones(&[out_dim])),
            conv2: CausalConv3d::zeros(out_dim, out_dim, [3, 3, 3], [1, 1, 1], [1, 1, 1]),
            shortcut,
        }
    }

    fn load(map: &WeightMap, prefix: &str, in_dim: usize, out_dim: usize) -> Result<Self> {
        let shortcut = if in_dim != out_dim {
            Some(CausalConv3d::load(
                map,
                &weights::join_key(prefix, "conv_shortcut"),
                in_dim,
                out_dim,
                [1, 1, 1],
                [1, 1, 1],
                [0, 0, 0],
            )?)
        } else {
            None
        };
        Ok(Self {
            norm1: gamma(map, &weights::join_key(prefix, "norm1.gamma"), &[in_dim, 1, 1, 1])?,
            conv1: CausalConv3d::load(
                map,
                &weights::join_key(prefix, "conv1"),
                in_dim,
                out_dim,
                [3, 3, 3],
                [1, 1, 1],
                [1, 1, 1],
            )?,
            norm2: gamma(map, &weights::join_key(prefix, "norm2.gamma"), &[out_dim, 1, 1, 1])?,
            conv2: CausalConv3d::load(
                map,
                &weights::join_key(prefix, "conv2"),
                out_dim,
                out_dim,
                [3, 3, 3],
                [1, 1, 1],
                [1, 1, 1],
            )?,
            shortcut,
        })
    }

    fn forward(&self, xs: &CudaTensor, mut cache: Option<&mut FeatCache>) -> Result<CudaTensor> {
        let mut x = rms_silu_video(xs, &self.norm1)?;
        x = conv_cached(&self.conv1, &x, cache.as_deref_mut())?;
        x = rms_silu_video(&x, &self.norm2)?;
        x = conv_cached(&self.conv2, &x, cache.as_deref_mut())?;
        match &self.shortcut {
            Some(sc) => Ok(sc.forward(xs)?.add(&x)?),
            None => xs.add(&x),
        }
    }
}

#[derive(Debug, Clone)]
struct AttentionBlock {
    norm: CudaTensor,
    qkv: CudaTensor, // [3c, c, 1, 1]
    qkv_bias: CudaTensor,
    proj: CudaTensor,
    proj_bias: CudaTensor,
}

impl AttentionBlock {
    fn zeros(dim: usize) -> Self {
        Self {
            norm: pinned(CudaTensor::ones(&[dim])),
            qkv: pinned(CudaTensor::zeros(&[dim * 3, dim, 1, 1])),
            qkv_bias: pinned(CudaTensor::zeros(&[dim * 3])),
            proj: pinned(CudaTensor::zeros(&[dim, dim, 1, 1])),
            proj_bias: pinned(CudaTensor::zeros(&[dim])),
        }
    }

    fn load(map: &WeightMap, prefix: &str, dim: usize) -> Result<Self> {
        let key = |name: &str| weights::join_key(prefix, name);
        Ok(Self {
            norm: gamma(map, &key("norm.gamma"), &[dim, 1, 1])?,
            qkv: pinned(weights::cuda_tensor_shaped(map, &key("to_qkv.weight"), &[dim * 3, dim, 1, 1])?),
            qkv_bias: pinned(weights::cuda_tensor_shaped(map, &key("to_qkv.bias"), &[dim * 3])?),
            proj: pinned(weights::cuda_tensor_shaped(map, &key("proj.weight"), &[dim, dim, 1, 1])?),
            proj_bias: pinned(weights::cuda_tensor_shaped(map, &key("proj.bias"), &[dim])?),
        })
    }

    /// Per-frame single-head attention over `h*w` tokens, identity residual.
    /// Accepts `[b, c, t, h, w]` or `[b, c, h, w]`.
    fn forward(&self, xs: &CudaTensor) -> Result<CudaTensor> {
        let (b, c) = (xs.shape[0], xs.shape[1]);
        let (t, h, w) = match xs.shape[..] {
            [_, _, t, h, w] => (t, h, w),
            [_, _, h, w] => (1, h, w),
            _ => return Err(TensorError::Message(format!("attention block input {:?}", xs.shape))),
        };
        let frames = if xs.rank() == 5 {
            xs.permute(&[0, 2, 1, 3, 4])?.reshape(vec![b * t, c, h, w])?
        } else {
            xs.clone()
        };
        let x = rms_video(&frames, &self.norm)?;
        let hw = h * w;
        // [bt, 3c, h, w] → [bt, hw, 3c]; q/k/v are its column blocks.
        let qkv = x
            .conv2d(&self.qkv, Some(&self.qkv_bias), 0, 1)?
            .reshape(vec![b * t, c * 3, hw])?
            .permute(&[0, 2, 1])?;
        let q = qkv.split_heads_bhsd(0, 1, c)?;
        let k = qkv.split_heads_bhsd(c, 1, c)?;
        let v = qkv.split_heads_bhsd(2 * c, 1, c)?;
        let attn = super::nn::scaled_dot_product_attention(&q, &k, &v, None)?
            .reshape(vec![b * t, hw, c])?
            .permute(&[0, 2, 1])?
            .reshape(vec![b * t, c, h, w])?;
        let y = attn.conv2d(&self.proj, Some(&self.proj_bias), 0, 1)?;
        let y = if xs.rank() == 5 {
            y.reshape(vec![b, t, c, h, w])?.permute(&[0, 2, 1, 3, 4])?
        } else {
            y
        };
        xs.add(&y)
    }
}

#[derive(Debug, Clone, Copy)]
enum ResampleMode {
    Upsample2d,
    Upsample3d,
}

#[derive(Debug, Clone)]
struct Resample {
    mode: ResampleMode,
    conv_w: CudaTensor,
    conv_b: CudaTensor,
    time_conv: Option<CausalConv3d>,
}

impl Resample {
    fn zeros(dim: usize, mode: ResampleMode) -> Self {
        let out = dim / 2;
        let time_conv = match mode {
            ResampleMode::Upsample2d => None,
            ResampleMode::Upsample3d => Some(CausalConv3d::zeros(
                dim,
                dim * 2,
                [3, 1, 1],
                [1, 1, 1],
                [1, 0, 0],
            )),
        };
        Self {
            mode,
            conv_w: pinned(CudaTensor::zeros(&[out, dim, 3, 3])),
            conv_b: pinned(CudaTensor::zeros(&[out])),
            time_conv,
        }
    }

    fn load(map: &WeightMap, prefix: &str, dim: usize, mode: ResampleMode) -> Result<Self> {
        let out = dim / 2;
        let time_conv = match mode {
            ResampleMode::Upsample2d => None,
            ResampleMode::Upsample3d => Some(CausalConv3d::load(
                map,
                &weights::join_key(prefix, "time_conv"),
                dim,
                dim * 2,
                [3, 1, 1],
                [1, 1, 1],
                [1, 0, 0],
            )?),
        };
        Ok(Self {
            mode,
            conv_w: pinned(weights::cuda_tensor_shaped(map, &weights::join_key(prefix, "resample.1.weight"), &[out, dim, 3, 3])?),
            conv_b: pinned(weights::cuda_tensor_shaped(map, &weights::join_key(prefix, "resample.1.bias"), &[out])?),
            time_conv,
        })
    }

    fn forward(&self, xs: &CudaTensor, cache: Option<&mut FeatCache>) -> Result<CudaTensor> {
        let mut x = xs.clone();
        if matches!(self.mode, ResampleMode::Upsample3d) {
            if let Some(tc) = &self.time_conv {
                if let Some(cache) = cache {
                    let i = cache.reserve();
                    let slot = cache.slots.get(i).cloned().unwrap_or(CacheSlot::Empty);
                    match slot {
                        CacheSlot::Empty => {
                            cache.slots[i] = CacheSlot::Rep;
                        }
                        CacheSlot::Rep | CacheSlot::Tensor(_) => {
                            let mut cache_x = last_frames(&x, CACHE_T)?;
                            if cache_x.dim(2)? < CACHE_T {
                                if let CacheSlot::Tensor(prev) = &slot {
                                    let last = prev.narrow(2, prev.dim(2)? - 1, 1)?;
                                    cache_x = CudaTensor::cat(&[&last, &cache_x], 2)?;
                                }
                            }
                            let cache_arg = match &slot {
                                CacheSlot::Tensor(t) => Some(t.clone()),
                                _ => None,
                            };
                            x = tc.forward_with_cache(&x, cache_arg.as_ref())?;
                            cache.slots[i] = CacheSlot::Tensor(cache_x);
                            x = double_time(x)?;
                        }
                    }
                } else {
                    x = tc.forward(&x)?;
                    x = double_time(x)?;
                }
            }
        }
        let (b, c, t, h, w) = (x.shape[0], x.shape[1], x.shape[2], x.shape[3], x.shape[4]);
        let x2 = x.permute(&[0, 2, 1, 3, 4])?.reshape(vec![b * t, c, h, w])?;
        let up = x2.upsample_nearest2d(h * 2, w * 2)?;
        let y = up.conv2d(&self.conv_w, Some(&self.conv_b), 1, 1)?;
        let oc = y.shape[1];
        y.reshape(vec![b, t, oc, h * 2, w * 2])?
            .permute(&[0, 2, 1, 3, 4])
    }
}

#[derive(Debug, Clone)]
struct UpBlock {
    resnets: Vec<ResidualBlock>,
    upsample: Option<Resample>,
}

impl UpBlock {
    fn zeros(
        in_dim: usize,
        out_dim: usize,
        n_res: usize,
        upsample: Option<ResampleMode>,
    ) -> Self {
        let mut resnets = Vec::new();
        let mut current = in_dim;
        for _ in 0..=n_res {
            resnets.push(ResidualBlock::zeros(current, out_dim));
            current = out_dim;
        }
        let upsample = upsample.map(|mode| Resample::zeros(out_dim, mode));
        Self { resnets, upsample }
    }

    fn load(
        map: &WeightMap,
        prefix: &str,
        in_dim: usize,
        out_dim: usize,
        n_res: usize,
        upsample: Option<ResampleMode>,
    ) -> Result<Self> {
        let mut resnets = Vec::new();
        let mut current = in_dim;
        for i in 0..=n_res {
            resnets.push(ResidualBlock::load(
                map,
                &weights::join_key(prefix, &format!("resnets.{i}")),
                current,
                out_dim,
            )?);
            current = out_dim;
        }
        let upsample = match upsample {
            Some(mode) => Some(Resample::load(
                map,
                &weights::join_key(prefix, "upsamplers.0"),
                out_dim,
                mode,
            )?),
            None => None,
        };
        Ok(Self { resnets, upsample })
    }

    fn forward(&self, mut xs: CudaTensor, mut cache: Option<&mut FeatCache>) -> Result<CudaTensor> {
        for r in &self.resnets {
            xs = r.forward(&xs, cache.as_deref_mut())?;
        }
        if let Some(up) = &self.upsample {
            xs = up.forward(&xs, cache.as_deref_mut())?;
        }
        Ok(xs)
    }
}

#[derive(Debug, Clone)]
pub struct WanDecoder {
    conv_in: CausalConv3d,
    mid_res0: ResidualBlock,
    mid_attn: AttentionBlock,
    mid_res1: ResidualBlock,
    up_blocks: Vec<UpBlock>,
    norm_out: CudaTensor,
    conv_out: CausalConv3d,
}

impl WanDecoder {
    pub fn zeros(cfg: &WanVaeConfig) -> Self {
        let mut dims: Vec<usize> = vec![cfg.base_dim * *cfg.dim_mult.last().unwrap()];
        for u in cfg.dim_mult.iter().rev() {
            dims.push(cfg.base_dim * *u);
        }
        let conv_in = CausalConv3d::zeros(cfg.z_dim, dims[0], [3, 3, 3], [1, 1, 1], [1, 1, 1]);
        let mid_res0 = ResidualBlock::zeros(dims[0], dims[0]);
        let mid_attn = AttentionBlock::zeros(dims[0]);
        let mid_res1 = ResidualBlock::zeros(dims[0], dims[0]);
        let mut up_blocks = Vec::new();
        for i in 0..dims.len() - 1 {
            let mut in_dim = dims[i];
            let out_dim = dims[i + 1];
            if i > 0 {
                in_dim /= 2;
            }
            let up_flag = i != cfg.dim_mult.len() - 1;
            let mode = if !up_flag {
                None
            } else if cfg.temporal_upsample.get(i).copied().unwrap_or(false) {
                Some(ResampleMode::Upsample3d)
            } else {
                Some(ResampleMode::Upsample2d)
            };
            up_blocks.push(UpBlock::zeros(in_dim, out_dim, cfg.num_res_blocks, mode));
        }
        let out_dim = *dims.last().unwrap();
        Self {
            conv_in,
            mid_res0,
            mid_attn,
            mid_res1,
            up_blocks,
            norm_out: pinned(CudaTensor::ones(&[out_dim])),
            conv_out: CausalConv3d::zeros(out_dim, 3, [3, 3, 3], [1, 1, 1], [1, 1, 1]),
        }
    }

    pub fn load(map: &WeightMap, prefix: &str, cfg: &WanVaeConfig) -> Result<Self> {
        let mut dims: Vec<usize> = vec![cfg.base_dim * *cfg.dim_mult.last().unwrap()];
        for u in cfg.dim_mult.iter().rev() {
            dims.push(cfg.base_dim * *u);
        }
        let conv_in = CausalConv3d::load(
            map,
            &weights::join_key(prefix, "conv_in"),
            cfg.z_dim,
            dims[0],
            [3, 3, 3],
            [1, 1, 1],
            [1, 1, 1],
        )?;
        let mid_res0 = ResidualBlock::load(
            map,
            &weights::join_key(prefix, "mid_block.resnets.0"),
            dims[0],
            dims[0],
        )?;
        let mid_attn = AttentionBlock::load(
            map,
            &weights::join_key(prefix, "mid_block.attentions.0"),
            dims[0],
        )?;
        let mid_res1 = ResidualBlock::load(
            map,
            &weights::join_key(prefix, "mid_block.resnets.1"),
            dims[0],
            dims[0],
        )?;
        let mut up_blocks = Vec::new();
        for i in 0..dims.len() - 1 {
            let mut in_dim = dims[i];
            let out_dim = dims[i + 1];
            if i > 0 {
                in_dim /= 2;
            }
            let up_flag = i != cfg.dim_mult.len() - 1;
            let mode = if !up_flag {
                None
            } else if cfg.temporal_upsample.get(i).copied().unwrap_or(false) {
                Some(ResampleMode::Upsample3d)
            } else {
                Some(ResampleMode::Upsample2d)
            };
            up_blocks.push(UpBlock::load(
                map,
                &weights::join_key(prefix, &format!("up_blocks.{i}")),
                in_dim,
                out_dim,
                cfg.num_res_blocks,
                mode,
            )?);
        }
        let out_dim = *dims.last().unwrap();
        Ok(Self {
            conv_in,
            mid_res0,
            mid_attn,
            mid_res1,
            up_blocks,
            norm_out: gamma(map, &weights::join_key(prefix, "norm_out.gamma"), &[out_dim, 1, 1, 1])?,
            conv_out: CausalConv3d::load(
                map,
                &weights::join_key(prefix, "conv_out"),
                out_dim,
                3,
                [3, 3, 3],
                [1, 1, 1],
                [1, 1, 1],
            )?,
        })
    }

    fn forward(&self, zs: &CudaTensor, mut cache: Option<&mut FeatCache>) -> Result<CudaTensor> {
        let mut x = conv_cached(&self.conv_in, zs, cache.as_deref_mut())?;
        x = self.mid_res0.forward(&x, cache.as_deref_mut())?;
        x = self.mid_attn.forward(&x)?;
        x = self.mid_res1.forward(&x, cache.as_deref_mut())?;
        for up in &self.up_blocks {
            x = up.forward(x, cache.as_deref_mut())?;
        }
        x = rms_silu_video(&x, &self.norm_out)?;
        x = conv_cached(&self.conv_out, &x, cache.as_deref_mut())?;
        Ok(x.clamp(-1.0, 1.0))
    }
}

#[derive(Debug, Clone)]
struct EncoderDownsample {
    spatial_w: CudaTensor,
    spatial_b: CudaTensor,
    time_conv: Option<CausalConv3d>,
}

impl EncoderDownsample {
    fn load_2d(map: &WeightMap, prefix: &str, channels: usize) -> Result<Self> {
        Ok(Self {
            spatial_w: pinned(weights::cuda_tensor_shaped(
                map,
                &weights::join_key(prefix, "resample.1.weight"),
                &[channels, channels, 3, 3],
            )?),
            spatial_b: pinned(weights::cuda_tensor_shaped(
                map,
                &weights::join_key(prefix, "resample.1.bias"),
                &[channels],
            )?),
            time_conv: None,
        })
    }

    fn load_3d(map: &WeightMap, prefix: &str, channels: usize) -> Result<Self> {
        Ok(Self {
            spatial_w: pinned(weights::cuda_tensor_shaped(
                map,
                &weights::join_key(prefix, "resample.1.weight"),
                &[channels, channels, 3, 3],
            )?),
            spatial_b: pinned(weights::cuda_tensor_shaped(
                map,
                &weights::join_key(prefix, "resample.1.bias"),
                &[channels],
            )?),
            time_conv: Some(CausalConv3d::load(
                map,
                &weights::join_key(prefix, "time_conv"),
                channels,
                channels,
                [3, 1, 1],
                [2, 1, 1],
                [1, 0, 0],
            )?),
        })
    }

    fn forward(&self, xs: &CudaTensor) -> Result<CudaTensor> {
        let mut x = xs.clone();
        if let Some(tc) = &self.time_conv {
            x = tc.forward(&x)?;
        }
        let (b, c, t, h, w) = (x.shape[0], x.shape[1], x.shape[2], x.shape[3], x.shape[4]);
        let x2 = x.permute(&[0, 2, 1, 3, 4])?.reshape(vec![b * t, c, h, w])?;
        let y = x2.conv2d(&self.spatial_w, Some(&self.spatial_b), 1, 2)?;
        let (_n, oc, hh, ww) = (y.shape[0], y.shape[1], y.shape[2], y.shape[3]);
        y.reshape(vec![b, t, oc, hh, ww])?.permute(&[0, 2, 1, 3, 4])
    }
}

#[derive(Debug, Clone)]
enum EncoderStage {
    Res(ResidualBlock),
    Down(EncoderDownsample),
}

#[derive(Debug, Clone)]
struct WanEncoder {
    conv_in: CausalConv3d,
    blocks: Vec<EncoderStage>,
    mid_res0: ResidualBlock,
    mid_attn: AttentionBlock,
    mid_res1: ResidualBlock,
    norm_out: CudaTensor,
    conv_out: CausalConv3d,
    quant: CausalConv3d,
}

impl WanEncoder {
    fn load_wan_2_1(map: &WeightMap) -> Result<Self> {
        let d = "encoder";
        let mut blocks = Vec::new();
        blocks.push(EncoderStage::Res(ResidualBlock::load(
            map,
            &format!("{d}.down_blocks.0"),
            96,
            96,
        )?));
        blocks.push(EncoderStage::Res(ResidualBlock::load(
            map,
            &format!("{d}.down_blocks.1"),
            96,
            96,
        )?));
        blocks.push(EncoderStage::Down(EncoderDownsample::load_2d(
            map,
            &format!("{d}.down_blocks.2"),
            96,
        )?));
        blocks.push(EncoderStage::Res(ResidualBlock::load(
            map,
            &format!("{d}.down_blocks.3"),
            96,
            192,
        )?));
        blocks.push(EncoderStage::Res(ResidualBlock::load(
            map,
            &format!("{d}.down_blocks.4"),
            192,
            192,
        )?));
        blocks.push(EncoderStage::Down(EncoderDownsample::load_3d(
            map,
            &format!("{d}.down_blocks.5"),
            192,
        )?));
        blocks.push(EncoderStage::Res(ResidualBlock::load(
            map,
            &format!("{d}.down_blocks.6"),
            192,
            384,
        )?));
        blocks.push(EncoderStage::Res(ResidualBlock::load(
            map,
            &format!("{d}.down_blocks.7"),
            384,
            384,
        )?));
        blocks.push(EncoderStage::Down(EncoderDownsample::load_3d(
            map,
            &format!("{d}.down_blocks.8"),
            384,
        )?));
        blocks.push(EncoderStage::Res(ResidualBlock::load(
            map,
            &format!("{d}.down_blocks.9"),
            384,
            384,
        )?));
        blocks.push(EncoderStage::Res(ResidualBlock::load(
            map,
            &format!("{d}.down_blocks.10"),
            384,
            384,
        )?));
        let mid = format!("{d}.mid_block");
        Ok(Self {
            conv_in: CausalConv3d::load(map, &format!("{d}.conv_in"), 3, 96, [3, 3, 3], [1, 1, 1], [1, 1, 1])?,
            blocks,
            mid_res0: ResidualBlock::load(map, &format!("{mid}.resnets.0"), 384, 384)?,
            mid_attn: AttentionBlock::load(map, &format!("{mid}.attentions.0"), 384)?,
            mid_res1: ResidualBlock::load(map, &format!("{mid}.resnets.1"), 384, 384)?,
            norm_out: gamma(map, &format!("{d}.norm_out.gamma"), &[384, 1, 1, 1])?,
            conv_out: CausalConv3d::load(
                map,
                &format!("{d}.conv_out"),
                384,
                32,
                [3, 3, 3],
                [1, 1, 1],
                [1, 1, 1],
            )?,
            quant: CausalConv3d::load(map, "quant_conv", 32, 32, [1, 1, 1], [1, 1, 1], [0, 0, 0])?,
        })
    }

    fn forward(&self, xs: &CudaTensor) -> Result<CudaTensor> {
        let mut x = self.conv_in.forward(xs)?;
        for block in &self.blocks {
            x = match block {
                EncoderStage::Res(r) => r.forward(&x, None)?,
                EncoderStage::Down(d) => d.forward(&x)?,
            };
        }
        x = self.mid_res0.forward(&x, None)?;
        x = self.mid_attn.forward(&x)?;
        x = self.mid_res1.forward(&x, None)?;
        x = rms_silu_video(&x, &self.norm_out)?;
        x = self.conv_out.forward(&x)?;
        x = self.quant.forward(&x)?;
        let chunks = x.chunk(2, 1)?;
        Ok(chunks[0].clone())
    }
}

#[derive(Debug, Clone)]
pub struct AutoencoderKlWan {
    pub cfg: WanVaeConfig,
    post_quant: CausalConv3d,
    decoder: WanDecoder,
    encoder: Option<WanEncoder>,
}

impl AutoencoderKlWan {
    pub fn zeros(cfg: WanVaeConfig) -> Self {
        Self {
            post_quant: CausalConv3d::zeros(cfg.z_dim, cfg.z_dim, [1, 1, 1], [1, 1, 1], [0, 0, 0]),
            decoder: WanDecoder::zeros(&cfg),
            encoder: None,
            cfg,
        }
    }

    pub fn load(cfg: WanVaeConfig, map: &WeightMap) -> Result<Self> {
        Self::from_map(cfg, map)
    }

    pub fn from_map(cfg: WanVaeConfig, map: &WeightMap) -> Result<Self> {
        let encoder = if cfg.load_encoder {
            match WanEncoder::load_wan_2_1(map) {
                Ok(enc) => Some(enc),
                Err(e) => {
                    eprintln!("warn: VAE encoder not loaded ({e}); I2V encode unavailable");
                    None
                }
            }
        } else {
            None
        };
        Ok(Self {
            post_quant: CausalConv3d::load(
                map,
                "post_quant_conv",
                cfg.z_dim,
                cfg.z_dim,
                [1, 1, 1],
                [1, 1, 1],
                [0, 0, 0],
            )?,
            decoder: WanDecoder::load(map, "decoder", &cfg)?,
            encoder,
            cfg,
        })
    }

    pub fn scale_latents(&self, latents: &CudaTensor) -> Result<CudaTensor> {
        let n = self.cfg.z_dim.min(16);
        let mean = CudaTensor::from_vec(LATENTS_MEAN[..n].to_vec(), vec![1, n, 1, 1, 1])?;
        let std = CudaTensor::from_vec(LATENTS_STD[..n].to_vec(), vec![1, n, 1, 1, 1])?;
        latents.mul(&std)?.add(&mean)
    }

    pub fn normalize_latents(&self, latents: &CudaTensor) -> Result<CudaTensor> {
        let n = self.cfg.z_dim.min(16);
        let mean = CudaTensor::from_vec(LATENTS_MEAN[..n].to_vec(), vec![1, n, 1, 1, 1])?;
        let std = CudaTensor::from_vec(LATENTS_STD[..n].to_vec(), vec![1, n, 1, 1, 1])?;
        latents.sub(&mean)?.div(&std)
    }

    pub fn encode_video(&self, video: &CudaTensor) -> Result<CudaTensor> {
        let enc = self
            .encoder
            .as_ref()
            .ok_or_else(|| TensorError::Message("VAE encoder not loaded".into()))?;
        enc.forward(video)
    }

    pub fn decode(&self, latents: &CudaTensor) -> Result<CudaTensor> {
        let z = self.post_quant.forward(latents)?;
        let t = z.dim(2)?;
        let mut cache = FeatCache::new();
        let mut frames = Vec::with_capacity(t);
        for i in 0..t {
            cache.begin_pass();
            frames.push(self.decoder.forward(&z.narrow(2, i, 1)?, Some(&mut cache))?);
        }
        let refs: Vec<&CudaTensor> = frames.iter().collect();
        CudaTensor::cat(&refs, 2)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fastvideo_models::wan::WanVaeConfig;

    #[test]
    fn rms_video_4d_and_5d_shapes() {
        // 5D path (B, C, T, H, W) — primary decode input.
        let xs5 = CudaTensor::zeros(&[1, 4, 3, 2, 2]);
        let gamma = CudaTensor::ones(&[4]);
        let out5 = rms_video(&xs5, &gamma).expect("5D rms_video");
        assert_eq!(out5.shape, vec![1, 4, 3, 2, 2]);

        // 4D path (B, C, H, W) — used by tiny unit tests / mid-block inputs.
        let xs4 = CudaTensor::zeros(&[1, 4, 2, 2]);
        let out4 = rms_video(&xs4, &gamma).expect("4D rms_video");
        assert_eq!(out4.shape, vec![1, 4, 2, 2]);

        // 3D path: even smaller degenerate rank should not panic.
        let xs3 = CudaTensor::zeros(&[1, 4, 2]);
        let out3 = rms_video(&xs3, &gamma).expect("3D rms_video");
        assert_eq!(out3.shape, vec![1, 4, 2]);
    }

    #[test]
    fn attention_block_4d_and_5d() {
        // Tiny VAE config so we don't blow up memory.
        let cfg = WanVaeConfig {
            base_dim: 8,
            z_dim: 4,
            dim_mult: vec![1, 2, 2],
            num_res_blocks: 1,
            temporal_upsample: vec![true, true],
            load_encoder: false,
        };
        let vae = AutoencoderKlWan::zeros(cfg);

        // 5D decode — normal latent tensor.
        let z5 = CudaTensor::zeros(&[1, 4, 3, 2, 2]);
        let _ = vae.decode(&z5).expect("5D decode");

        // 4D latent (B, C, H, W): exercises AttentionBlock.forward 4D-safe path.
        let attn = AttentionBlock::zeros(8);
        let x4 = CudaTensor::zeros(&[1, 8, 2, 2]);
        let _ = attn.forward(&x4).expect("4D attention");
    }
}
