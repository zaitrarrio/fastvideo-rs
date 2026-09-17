//! Wan VAE decode with Diffusers/Wan feat-cache (`CACHE_T=2`, Rep sentinel).

use fastvideo_models::wan::WanVaeConfig;

use super::nn;
use super::tensor::{NdTensor, Result, TensorError};
use super::weights::{self, WeightMap};

const CACHE_T: usize = 2;

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
    Tensor(NdTensor),
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

fn last_frames(x: &NdTensor, n: usize) -> Result<NdTensor> {
    let t = x.dim(2)?;
    let take = t.min(n);
    x.narrow(2, t - take, take)
}

#[derive(Debug, Clone)]
struct CausalConv3d {
    weight: NdTensor, // [out, in, kt, kh, kw]
    bias: NdTensor,
    stride: [usize; 3],
    pad: [usize; 3],
}

impl CausalConv3d {
    fn zeros(in_c: usize, out_c: usize, kernel: [usize; 3], stride: [usize; 3], pad: [usize; 3]) -> Self {
        Self {
            weight: NdTensor::zeros(&[out_c, in_c, kernel[0], kernel[1], kernel[2]]),
            bias: NdTensor::zeros(&[out_c]),
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
            weight: weights::nd_tensor_shaped(
                map,
                &weights::join_key(prefix, "weight"),
                &[out_c, in_c, kernel[0], kernel[1], kernel[2]],
            )?,
            bias: weights::nd_tensor_shaped(map, &weights::join_key(prefix, "bias"), &[out_c])?,
            stride,
            pad,
        })
    }

    fn forward(&self, xs: &NdTensor) -> Result<NdTensor> {
        self.forward_with_cache(xs, None)
    }

    fn forward_with_cache(&self, xs: &NdTensor, cache_x: Option<&NdTensor>) -> Result<NdTensor> {
        let b = xs.shape[0];
        let (out_c, in_c, kt, kh, kw) = (
            self.weight.shape[0],
            self.weight.shape[1],
            self.weight.shape[2],
            self.weight.shape[3],
            self.weight.shape[4],
        );
        let mut x = xs.clone();
        let mut pad_t = 2 * self.pad[0];
        if let Some(c) = cache_x {
            if pad_t > 0 {
                x = NdTensor::cat(&[c, &x], 2)?;
                pad_t = pad_t.saturating_sub(c.dim(2)?);
            }
        }
        if self.pad[2] > 0 {
            x = x.pad_zeros(4, self.pad[2], self.pad[2])?;
        }
        if self.pad[1] > 0 {
            x = x.pad_zeros(3, self.pad[1], self.pad[1])?;
        }
        if pad_t > 0 {
            x = x.pad_zeros(2, pad_t, 0)?;
        }
        let (_b, _c, t_p, h_p, w_p) = (x.shape[0], x.shape[1], x.shape[2], x.shape[3], x.shape[4]);
        if kt == 1 && kh == 1 && kw == 1 && self.stride == [1, 1, 1] {
            let x = x
                .permute(&[0, 2, 3, 4, 1])?
                .reshape(vec![b * t_p * h_p * w_p, in_c])?;
            let w = self.weight.reshape(vec![out_c, in_c])?.transpose(0, 1)?;
            let y = x.matmul(&w)?;
            let y = y.add(&self.bias.reshape(vec![1, out_c])?)?;
            return y
                .reshape(vec![b, t_p, h_p, w_p, out_c])?
                .permute(&[0, 4, 1, 2, 3]);
        }
        let w2 = self
            .weight
            .reshape(vec![out_c, in_c * kt, kh, kw])?;
        let mut frames = Vec::new();
        let mut ti = 0usize;
        let t_stride = self.stride[0].max(1);
        while ti + kt <= t_p {
            let window = x.narrow(2, ti, kt)?.reshape(vec![b, in_c * kt, h_p, w_p])?;
            let y = nn::conv2d(&window, &w2, 0, self.stride[1])?;
            let bias = self.bias.reshape(vec![1, out_c, 1, 1])?;
            let y = y.add(&bias)?.unsqueeze(2)?;
            frames.push(y);
            ti += t_stride;
        }
        if frames.is_empty() {
            return Err(TensorError::Message("empty causal conv3d output".into()));
        }
        let refs: Vec<&NdTensor> = frames.iter().collect();
        NdTensor::cat(&refs, 2)
    }
}

fn conv_cached(
    conv: &CausalConv3d,
    x: &NdTensor,
    cache: Option<&mut FeatCache>,
) -> Result<NdTensor> {
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
            cache_x = NdTensor::cat(&[&last, &cache_x], 2)?;
        }
    }
    let y = conv.forward_with_cache(x, prev.as_ref())?;
    cache.slots[i] = CacheSlot::Tensor(cache_x);
    Ok(y)
}

fn double_time(x: NdTensor) -> Result<NdTensor> {
    let (b, c2, t, h, w) = (x.shape[0], x.shape[1], x.shape[2], x.shape[3], x.shape[4]);
    let c = c2 / 2;
    x.reshape(vec![b, 2, c, t, h, w])?
        .permute(&[0, 2, 3, 1, 4, 5])?
        .reshape(vec![b, c, t * 2, h, w])
}

fn rms_video(xs: &NdTensor, gamma: &NdTensor) -> Result<NdTensor> {
    // Channel-first RMS over dim=1.
    let (b, c, t, h, w) = (
        xs.shape[0],
        xs.shape[1],
        xs.shape[2],
        xs.shape[3],
        xs.shape[4],
    );
    let mut out = vec![0.0f32; xs.data.len()];
    let spatial = t * h * w;
    for bi in 0..b {
        for s in 0..spatial {
            let mut mean_sq = 0.0;
            for ci in 0..c {
                let v = xs.data[((bi * c + ci) * spatial) + s];
                mean_sq += v * v;
            }
            mean_sq /= c as f32;
            let inv = 1.0 / (mean_sq + 1e-12).sqrt();
            for ci in 0..c {
                let idx = ((bi * c + ci) * spatial) + s;
                let g = gamma.data[ci];
                out[idx] = xs.data[idx] * inv * g;
            }
        }
    }
    Ok(NdTensor {
        data: out,
        shape: xs.shape.clone(),
    })
}

fn silu_video(xs: &NdTensor) -> NdTensor {
    xs.silu()
}

#[derive(Debug, Clone)]
struct ResidualBlock {
    norm1: NdTensor,
    conv1: CausalConv3d,
    norm2: NdTensor,
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
            norm1: NdTensor::ones(&[in_dim]),
            conv1: CausalConv3d::zeros(in_dim, out_dim, [3, 3, 3], [1, 1, 1], [1, 1, 1]),
            norm2: NdTensor::ones(&[out_dim]),
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
            norm1: weights::nd_tensor_shaped(
                map,
                &weights::join_key(prefix, "norm1.gamma"),
                &[in_dim, 1, 1, 1],
            )?
            .reshape(vec![in_dim])?,
            conv1: CausalConv3d::load(
                map,
                &weights::join_key(prefix, "conv1"),
                in_dim,
                out_dim,
                [3, 3, 3],
                [1, 1, 1],
                [1, 1, 1],
            )?,
            norm2: weights::nd_tensor_shaped(
                map,
                &weights::join_key(prefix, "norm2.gamma"),
                &[out_dim, 1, 1, 1],
            )?
            .reshape(vec![out_dim])?,
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

    fn forward(&self, xs: &NdTensor, mut cache: Option<&mut FeatCache>) -> Result<NdTensor> {
        let mut x = rms_video(xs, &self.norm1)?;
        x = silu_video(&x);
        x = conv_cached(&self.conv1, &x, cache.as_deref_mut())?;
        x = rms_video(&x, &self.norm2)?;
        x = silu_video(&x);
        x = conv_cached(&self.conv2, &x, cache.as_deref_mut())?;
        match &self.shortcut {
            Some(sc) => Ok(sc.forward(xs)?.add(&x)?),
            None => xs.add(&x),
        }
    }
}

#[derive(Debug, Clone)]
struct AttentionBlock {
    norm: NdTensor,
    qkv: NdTensor, // [3c, c, 1, 1]
    qkv_bias: NdTensor,
    proj: NdTensor,
    proj_bias: NdTensor,
}

impl AttentionBlock {
    fn zeros(dim: usize) -> Self {
        Self {
            norm: NdTensor::ones(&[dim]),
            qkv: NdTensor::zeros(&[dim * 3, dim, 1, 1]),
            qkv_bias: NdTensor::zeros(&[dim * 3]),
            proj: NdTensor::zeros(&[dim, dim, 1, 1]),
            proj_bias: NdTensor::zeros(&[dim]),
        }
    }

    fn load(map: &WeightMap, prefix: &str, dim: usize) -> Result<Self> {
        Ok(Self {
            norm: weights::nd_tensor_shaped(
                map,
                &weights::join_key(prefix, "norm.gamma"),
                &[dim, 1, 1],
            )?
            .reshape(vec![dim])?,
            qkv: weights::nd_tensor_shaped(
                map,
                &weights::join_key(prefix, "to_qkv.weight"),
                &[dim * 3, dim, 1, 1],
            )?,
            qkv_bias: weights::nd_tensor_shaped(
                map,
                &weights::join_key(prefix, "to_qkv.bias"),
                &[dim * 3],
            )?,
            proj: weights::nd_tensor_shaped(
                map,
                &weights::join_key(prefix, "proj.weight"),
                &[dim, dim, 1, 1],
            )?,
            proj_bias: weights::nd_tensor_shaped(
                map,
                &weights::join_key(prefix, "proj.bias"),
                &[dim],
            )?,
        })
    }

    fn forward(&self, xs: &NdTensor) -> Result<NdTensor> {
        let (b, c, t, h, w) = (
            xs.shape[0],
            xs.shape[1],
            xs.shape[2],
            xs.shape[3],
            xs.shape[4],
        );
        let mut x = xs
            .permute(&[0, 2, 1, 3, 4])?
            .reshape(vec![b * t, c, h, w])?;
        // RMS over channels
        let gamma = self.norm.reshape(vec![1, c, 1, 1])?;
        let mut normed = vec![0.0f32; x.data.len()];
        let hw = h * w;
        for n in 0..(b * t) {
            for s in 0..hw {
                let mut mean_sq = 0.0;
                for ci in 0..c {
                    let v = x.data[(n * c + ci) * hw + s];
                    mean_sq += v * v;
                }
                mean_sq /= c as f32;
                let inv = 1.0 / (mean_sq + 1e-12).sqrt();
                for ci in 0..c {
                    let idx = (n * c + ci) * hw + s;
                    normed[idx] = x.data[idx] * inv * gamma.data[ci];
                }
            }
        }
        x = NdTensor {
            data: normed,
            shape: x.shape.clone(),
        };
        let qkv = nn::conv2d(&x, &self.qkv, 0, 1)?.add(
            &self.qkv_bias.reshape(vec![1, c * 3, 1, 1])?,
        )?;
        let qkv = qkv
            .reshape(vec![b * t, 1, c * 3, hw])?
            .permute(&[0, 1, 3, 2])?;
        let chunks = qkv.chunk(3, 3)?;
        let attn = nn::scaled_dot_product_attention(&chunks[0], &chunks[1], &chunks[2], None)?;
        let attn = attn
            .squeeze(1)?
            .permute(&[0, 2, 1])?
            .reshape(vec![b * t, c, h, w])?;
        let y = nn::conv2d(&attn, &self.proj, 0, 1)?
            .add(&self.proj_bias.reshape(vec![1, c, 1, 1])?)?;
        let y = y
            .reshape(vec![b, t, c, h, w])?
            .permute(&[0, 2, 1, 3, 4])?;
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
    conv_w: NdTensor,
    conv_b: NdTensor,
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
            conv_w: NdTensor::zeros(&[out, dim, 3, 3]),
            conv_b: NdTensor::zeros(&[out]),
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
            conv_w: weights::nd_tensor_shaped(
                map,
                &weights::join_key(prefix, "resample.1.weight"),
                &[out, dim, 3, 3],
            )?,
            conv_b: weights::nd_tensor_shaped(
                map,
                &weights::join_key(prefix, "resample.1.bias"),
                &[out],
            )?,
            time_conv,
        })
    }

    fn forward(&self, xs: &NdTensor, cache: Option<&mut FeatCache>) -> Result<NdTensor> {
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
                                    cache_x = NdTensor::cat(&[&last, &cache_x], 2)?;
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
        let bias = self.conv_b.reshape(vec![1, self.conv_w.shape[0], 1, 1])?;
        let x2 = x.permute(&[0, 2, 1, 3, 4])?.reshape(vec![b * t, c, h, w])?;
        let up = x2.upsample_nearest2d(h * 2, w * 2)?;
        let y = nn::conv2d(&up, &self.conv_w, 1, 1)?.add(&bias)?;
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

    fn forward(&self, mut xs: NdTensor, mut cache: Option<&mut FeatCache>) -> Result<NdTensor> {
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
    norm_out: NdTensor,
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
            norm_out: NdTensor::ones(&[out_dim]),
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
            norm_out: weights::nd_tensor_shaped(
                map,
                &weights::join_key(prefix, "norm_out.gamma"),
                &[out_dim, 1, 1, 1],
            )?
            .reshape(vec![out_dim])?,
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

    fn forward(&self, zs: &NdTensor, mut cache: Option<&mut FeatCache>) -> Result<NdTensor> {
        let mut x = conv_cached(&self.conv_in, zs, cache.as_deref_mut())?;
        x = self.mid_res0.forward(&x, cache.as_deref_mut())?;
        x = self.mid_attn.forward(&x)?;
        x = self.mid_res1.forward(&x, cache.as_deref_mut())?;
        for up in &self.up_blocks {
            x = up.forward(x, cache.as_deref_mut())?;
        }
        x = rms_video(&x, &self.norm_out)?;
        x = silu_video(&x);
        x = conv_cached(&self.conv_out, &x, cache.as_deref_mut())?;
        Ok(x.clamp(-1.0, 1.0))
    }
}

#[derive(Debug, Clone)]
pub struct AutoencoderKlWan {
    pub cfg: WanVaeConfig,
    post_quant: CausalConv3d,
    decoder: WanDecoder,
}

impl AutoencoderKlWan {
    pub fn zeros(cfg: WanVaeConfig) -> Self {
        Self {
            post_quant: CausalConv3d::zeros(cfg.z_dim, cfg.z_dim, [1, 1, 1], [1, 1, 1], [0, 0, 0]),
            decoder: WanDecoder::zeros(&cfg),
            cfg,
        }
    }

    pub fn load(cfg: WanVaeConfig, map: &WeightMap) -> Result<Self> {
        Self::from_map(cfg, map)
    }

    pub fn from_map(cfg: WanVaeConfig, map: &WeightMap) -> Result<Self> {
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
            cfg,
        })
    }

    pub fn scale_latents(&self, latents: &NdTensor) -> Result<NdTensor> {
        let n = self.cfg.z_dim.min(16);
        let mean = NdTensor::from_vec(LATENTS_MEAN[..n].to_vec(), vec![1, n, 1, 1, 1])?;
        let std = NdTensor::from_vec(LATENTS_STD[..n].to_vec(), vec![1, n, 1, 1, 1])?;
        latents.mul(&std)?.add(&mean)
    }

    pub fn decode(&self, latents: &NdTensor) -> Result<NdTensor> {
        let z = self.post_quant.forward(latents)?;
        let t = z.dim(2)?;
        let mut cache = FeatCache::new();
        let mut frames = Vec::with_capacity(t);
        for i in 0..t {
            cache.begin_pass();
            frames.push(self.decoder.forward(&z.narrow(2, i, 1)?, Some(&mut cache))?);
        }
        let refs: Vec<&NdTensor> = frames.iter().collect();
        NdTensor::cat(&refs, 2)
    }
}
