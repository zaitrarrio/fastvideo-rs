//! Wan VAE decode with Diffusers/Wan feat-cache protocol (Burn ndarray, zeros init).

use burn::prelude::*;
use burn::tensor::module::interpolate;
use burn::tensor::ops::{InterpolateMode, InterpolateOptions};

use fastvideo_models::WanVaeConfig;

use super::nn::{conv2d_nhwc, pad_dim5, sdpa, silu, B, Device};
use super::weights::{self, WeightMap};
use crate::error::Result;

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
    Tensor(Tensor<B, 5>),
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

fn last_frames(x: &Tensor<B, 5>, n: usize) -> Tensor<B, 5> {
    let t = x.dims()[2];
    let take = t.min(n);
    x.clone().narrow(2, t - take, take)
}

fn double_time(x: Tensor<B, 5>) -> Tensor<B, 5> {
    let [b, c2, t, h, w] = x.dims();
    let c = c2 / 2;
    x.reshape([b, 2, c, t, h, w])
        .permute([0, 2, 3, 1, 4, 5])
        .reshape([b, c, t * 2, h, w])
}

#[derive(Debug, Clone)]
struct CausalConv3d {
    weight: Tensor<B, 5>,
    bias: Tensor<B, 1>,
    stride: [usize; 3],
    pad: [usize; 3],
}

impl CausalConv3d {
    fn zeros(
        in_c: usize,
        out_c: usize,
        kernel: [usize; 3],
        stride: [usize; 3],
        pad: [usize; 3],
        device: &Device,
    ) -> Self {
        Self {
            weight: Tensor::zeros([out_c, in_c, kernel[0], kernel[1], kernel[2]], device),
            bias: Tensor::zeros([out_c], device),
            stride,
            pad,
        }
    }

    fn zeros_k(
        in_c: usize,
        out_c: usize,
        k: usize,
        stride: [usize; 3],
        pad: [usize; 3],
        device: &Device,
    ) -> Self {
        Self::zeros(in_c, out_c, [k, k, k], stride, pad, device)
    }

    fn load(
        map: &WeightMap,
        prefix: &str,
        in_c: usize,
        out_c: usize,
        kernel: [usize; 3],
        stride: [usize; 3],
        pad: [usize; 3],
        device: &Device,
    ) -> Result<Self> {
        Ok(Self {
            weight: weights::tensor5_shaped(
                map,
                &weights::join_key(prefix, "weight"),
                [out_c, in_c, kernel[0], kernel[1], kernel[2]],
                device,
            )?,
            bias: weights::tensor1_shaped(map, &weights::join_key(prefix, "bias"), out_c, device)?,
            stride,
            pad,
        })
    }

    fn load_k(
        map: &WeightMap,
        prefix: &str,
        in_c: usize,
        out_c: usize,
        k: usize,
        stride: [usize; 3],
        pad: [usize; 3],
        device: &Device,
    ) -> Result<Self> {
        Self::load(map, prefix, in_c, out_c, [k, k, k], stride, pad, device)
    }

    fn forward(&self, xs: Tensor<B, 5>) -> Tensor<B, 5> {
        self.forward_with_cache(xs, None)
    }

    fn forward_with_cache(&self, xs: Tensor<B, 5>, cache_x: Option<&Tensor<B, 5>>) -> Tensor<B, 5> {
        let [b, _c, _t, _h, _w] = xs.dims();
        let [out_c, in_c, kt, kh, kw] = self.weight.dims();
        let mut x = xs;
        let mut pad_t = 2 * self.pad[0];
        if let Some(c) = cache_x {
            if pad_t > 0 {
                x = Tensor::cat(vec![c.clone(), x], 2);
                pad_t = pad_t.saturating_sub(c.dims()[2]);
            }
        }
        if self.pad[2] > 0 {
            x = pad_dim5(x, 4, self.pad[2], self.pad[2]);
        }
        if self.pad[1] > 0 {
            x = pad_dim5(x, 3, self.pad[1], self.pad[1]);
        }
        if pad_t > 0 {
            x = pad_dim5(x, 2, pad_t, 0);
        }
        let [_b, _c, t_p, h_p, w_p] = x.dims();
        if kt == 1 && kh == 1 && kw == 1 && self.stride == [1, 1, 1] {
            let x = x
                .permute([0, 2, 3, 4, 1])
                .reshape([b * t_p * h_p * w_p, in_c]);
            let w = self.weight.clone().reshape([out_c, in_c]).transpose();
            let y = x.matmul(w) + self.bias.clone().unsqueeze::<2>();
            return y
                .reshape([b, t_p, h_p, w_p, out_c])
                .permute([0, 4, 1, 2, 3]);
        }
        let w2 = self
            .weight
            .clone()
            .reshape([out_c, in_c * kt, kh, kw]);
        let mut frames = Vec::new();
        let mut ti = 0usize;
        let t_stride = self.stride[0].max(1);
        while ti + kt <= t_p {
            let window = x.clone().narrow(2, ti, kt).reshape([b, in_c * kt, h_p, w_p]);
            let y = conv2d_nhwc(
                window,
                w2.clone(),
                Some(self.bias.clone()),
                0,
                self.stride[1],
            );
            frames.push(y.unsqueeze_dim::<5>(2));
            ti += t_stride;
        }
        assert!(!frames.is_empty(), "empty causal conv3d output");
        Tensor::cat(frames, 2)
    }
}

fn conv_cached(
    conv: &CausalConv3d,
    x: Tensor<B, 5>,
    cache: Option<&mut FeatCache>,
) -> Tensor<B, 5> {
    let Some(cache) = cache else {
        return conv.forward(x);
    };
    let i = cache.reserve();
    let prev = match cache.slots.get(i) {
        Some(CacheSlot::Tensor(t)) => Some(t.clone()),
        _ => None,
    };
    let mut cache_x = last_frames(&x, CACHE_T);
    if cache_x.dims()[2] < CACHE_T {
        if let Some(prev) = &prev {
            let last = prev.clone().narrow(2, prev.dims()[2] - 1, 1);
            cache_x = Tensor::cat(vec![last, cache_x], 2);
        }
    }
    let y = conv.forward_with_cache(x, prev.as_ref());
    cache.slots[i] = CacheSlot::Tensor(cache_x);
    y
}

fn rms_video(xs: Tensor<B, 5>, gamma: &Tensor<B, 4>) -> Tensor<B, 5> {
    let var = xs.clone().powf_scalar(2.0).mean_dim(1);
    let y = xs / (var.add_scalar(1e-12)).sqrt();
    let g = gamma.clone().reshape([1, gamma.dims()[0], 1, 1, 1]);
    y * g
}

#[derive(Debug, Clone)]
struct ResidualBlock {
    norm1: Tensor<B, 4>,
    conv1: CausalConv3d,
    norm2: Tensor<B, 4>,
    conv2: CausalConv3d,
    shortcut: Option<CausalConv3d>,
}

impl ResidualBlock {
    fn zeros(in_dim: usize, out_dim: usize, device: &Device) -> Self {
        let shortcut = if in_dim != out_dim {
            Some(CausalConv3d::zeros_k(
                in_dim,
                out_dim,
                1,
                [1, 1, 1],
                [0, 0, 0],
                device,
            ))
        } else {
            None
        };
        Self {
            norm1: Tensor::zeros([in_dim, 1, 1, 1], device),
            conv1: CausalConv3d::zeros_k(in_dim, out_dim, 3, [1, 1, 1], [1, 1, 1], device),
            norm2: Tensor::zeros([out_dim, 1, 1, 1], device),
            conv2: CausalConv3d::zeros_k(out_dim, out_dim, 3, [1, 1, 1], [1, 1, 1], device),
            shortcut,
        }
    }

    fn load(map: &WeightMap, prefix: &str, in_dim: usize, out_dim: usize, device: &Device) -> Result<Self> {
        let shortcut = if in_dim != out_dim {
            Some(CausalConv3d::load_k(
                map,
                &weights::join_key(prefix, "conv_shortcut"),
                in_dim,
                out_dim,
                1,
                [1, 1, 1],
                [0, 0, 0],
                device,
            )?)
        } else {
            None
        };
        Ok(Self {
            norm1: weights::tensor4_shaped(
                map,
                &weights::join_key(prefix, "norm1.gamma"),
                [in_dim, 1, 1, 1],
                device,
            )?,
            conv1: CausalConv3d::load_k(
                map,
                &weights::join_key(prefix, "conv1"),
                in_dim,
                out_dim,
                3,
                [1, 1, 1],
                [1, 1, 1],
                device,
            )?,
            norm2: weights::tensor4_shaped(
                map,
                &weights::join_key(prefix, "norm2.gamma"),
                [out_dim, 1, 1, 1],
                device,
            )?,
            conv2: CausalConv3d::load_k(
                map,
                &weights::join_key(prefix, "conv2"),
                out_dim,
                out_dim,
                3,
                [1, 1, 1],
                [1, 1, 1],
                device,
            )?,
            shortcut,
        })
    }

    fn forward(&self, xs: Tensor<B, 5>, mut cache: Option<&mut FeatCache>) -> Tensor<B, 5> {
        let mut x = rms_video(xs.clone(), &self.norm1);
        x = silu(x);
        x = conv_cached(&self.conv1, x, cache.as_deref_mut());
        x = rms_video(x, &self.norm2);
        x = silu(x);
        x = conv_cached(&self.conv2, x, cache.as_deref_mut());
        match &self.shortcut {
            Some(sc) => sc.forward(xs) + x,
            None => xs + x,
        }
    }
}

#[derive(Debug, Clone)]
struct AttentionBlock {
    norm: Tensor<B, 3>,
    qkv: Tensor<B, 4>,
    qkv_bias: Tensor<B, 1>,
    proj: Tensor<B, 4>,
    proj_bias: Tensor<B, 1>,
}

impl AttentionBlock {
    fn zeros(dim: usize, device: &Device) -> Self {
        Self {
            norm: Tensor::zeros([dim, 1, 1], device),
            qkv: Tensor::zeros([dim * 3, dim, 1, 1], device),
            qkv_bias: Tensor::zeros([dim * 3], device),
            proj: Tensor::zeros([dim, dim, 1, 1], device),
            proj_bias: Tensor::zeros([dim], device),
        }
    }

    fn load(map: &WeightMap, prefix: &str, dim: usize, device: &Device) -> Result<Self> {
        Ok(Self {
            norm: weights::tensor3_shaped(
                map,
                &weights::join_key(prefix, "norm.gamma"),
                [dim, 1, 1],
                device,
            )?,
            qkv: weights::tensor4_shaped(
                map,
                &weights::join_key(prefix, "to_qkv.weight"),
                [dim * 3, dim, 1, 1],
                device,
            )?,
            qkv_bias: weights::tensor1_shaped(
                map,
                &weights::join_key(prefix, "to_qkv.bias"),
                dim * 3,
                device,
            )?,
            proj: weights::tensor4_shaped(
                map,
                &weights::join_key(prefix, "proj.weight"),
                [dim, dim, 1, 1],
                device,
            )?,
            proj_bias: weights::tensor1_shaped(
                map,
                &weights::join_key(prefix, "proj.bias"),
                dim,
                device,
            )?,
        })
    }

    fn forward(&self, xs: Tensor<B, 5>) -> Tensor<B, 5> {
        let [b, c, t, h, w] = xs.dims();
        let mut x = xs.clone().swap_dims(1, 2).reshape([b * t, c, h, w]);
        let gamma = self.norm.clone().reshape([1, c, 1, 1]);
        let var = x.clone().powf_scalar(2.0).mean_dim(1);
        x = (x / (var.add_scalar(1e-12)).sqrt()) * gamma;
        let qkv = conv2d_nhwc(x, self.qkv.clone(), Some(self.qkv_bias.clone()), 0, 1);
        let hw = h * w;
        let qkv = qkv.reshape([b * t, 1, c * 3, hw]).permute([0, 1, 3, 2]);
        let q = qkv.clone().narrow(3, 0, c);
        let k = qkv.clone().narrow(3, c, c);
        let v = qkv.narrow(3, 2 * c, c);
        let attn = sdpa(q, k, v, None);
        let attn = attn.squeeze::<3>(1).permute([0, 2, 1]).reshape([b * t, c, h, w]);
        let y = conv2d_nhwc(attn, self.proj.clone(), Some(self.proj_bias.clone()), 0, 1);
        let y = y.reshape([b, t, c, h, w]).permute([0, 2, 1, 3, 4]);
        xs + y
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
    conv_w: Option<Tensor<B, 4>>,
    conv_b: Option<Tensor<B, 1>>,
    time_conv: Option<CausalConv3d>,
}

impl Resample {
    fn zeros(dim: usize, mode: ResampleMode, device: &Device) -> Self {
        let out = dim / 2;
        let (conv_w, conv_b, time_conv) = match mode {
            ResampleMode::Upsample2d => (
                Some(Tensor::zeros([out, dim, 3, 3], device)),
                Some(Tensor::zeros([out], device)),
                None,
            ),
            ResampleMode::Upsample3d => (
                Some(Tensor::zeros([out, dim, 3, 3], device)),
                Some(Tensor::zeros([out], device)),
                Some(CausalConv3d::zeros(
                    dim,
                    dim * 2,
                    [3, 1, 1],
                    [1, 1, 1],
                    [1, 0, 0],
                    device,
                )),
            ),
        };
        Self {
            mode,
            conv_w,
            conv_b,
            time_conv,
        }
    }

    fn load(map: &WeightMap, prefix: &str, dim: usize, mode: ResampleMode, device: &Device) -> Result<Self> {
        let out = dim / 2;
        let (conv_w, conv_b, time_conv) = match mode {
            ResampleMode::Upsample2d => (
                Some(weights::tensor4_shaped(
                    map,
                    &weights::join_key(prefix, "resample.1.weight"),
                    [out, dim, 3, 3],
                    device,
                )?),
                Some(weights::tensor1_shaped(
                    map,
                    &weights::join_key(prefix, "resample.1.bias"),
                    out,
                    device,
                )?),
                None,
            ),
            ResampleMode::Upsample3d => (
                Some(weights::tensor4_shaped(
                    map,
                    &weights::join_key(prefix, "resample.1.weight"),
                    [out, dim, 3, 3],
                    device,
                )?),
                Some(weights::tensor1_shaped(
                    map,
                    &weights::join_key(prefix, "resample.1.bias"),
                    out,
                    device,
                )?),
                Some(CausalConv3d::load(
                    map,
                    &weights::join_key(prefix, "time_conv"),
                    dim,
                    dim * 2,
                    [3, 1, 1],
                    [1, 1, 1],
                    [1, 0, 0],
                    device,
                )?),
            ),
        };
        Ok(Self {
            mode,
            conv_w,
            conv_b,
            time_conv,
        })
    }

    fn forward(&self, xs: Tensor<B, 5>, cache: Option<&mut FeatCache>) -> Tensor<B, 5> {
        let mut x = xs;
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
                            let mut cache_x = last_frames(&x, CACHE_T);
                            if cache_x.dims()[2] < CACHE_T {
                                if let CacheSlot::Tensor(prev) = &slot {
                                    let last = prev.clone().narrow(2, prev.dims()[2] - 1, 1);
                                    cache_x = Tensor::cat(vec![last, cache_x], 2);
                                }
                            }
                            let cache_arg = match &slot {
                                CacheSlot::Tensor(t) => Some(t.clone()),
                                _ => None,
                            };
                            x = tc.forward_with_cache(x, cache_arg.as_ref());
                            cache.slots[i] = CacheSlot::Tensor(cache_x);
                            x = double_time(x);
                        }
                    }
                } else {
                    x = tc.forward(x);
                    x = double_time(x);
                }
            }
        }
        if let (Some(w_conv), Some(bias)) = (&self.conv_w, &self.conv_b) {
            let [b, c, t, h, w] = x.dims();
            let x2 = x.swap_dims(1, 2).reshape([b * t, c, h, w]);
            let up = interpolate(
                x2,
                [h * 2, w * 2],
                InterpolateOptions::new(InterpolateMode::Nearest),
            );
            let y = conv2d_nhwc(up, w_conv.clone(), Some(bias.clone()), 1, 1);
            let oc = y.dims()[1];
            x = y
                .reshape([b, t, oc, h * 2, w * 2])
                .permute([0, 2, 1, 3, 4]);
        }
        x
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
        device: &Device,
    ) -> Self {
        let mut resnets = Vec::new();
        let mut current = in_dim;
        for _ in 0..=n_res {
            resnets.push(ResidualBlock::zeros(current, out_dim, device));
            current = out_dim;
        }
        let upsample = upsample.map(|mode| Resample::zeros(out_dim, mode, device));
        Self { resnets, upsample }
    }

    fn load(
        map: &WeightMap,
        prefix: &str,
        in_dim: usize,
        out_dim: usize,
        n_res: usize,
        upsample: Option<ResampleMode>,
        device: &Device,
    ) -> Result<Self> {
        let mut resnets = Vec::new();
        let mut current = in_dim;
        for i in 0..=n_res {
            resnets.push(ResidualBlock::load(
                map,
                &weights::join_key(prefix, &format!("resnets.{i}")),
                current,
                out_dim,
                device,
            )?);
            current = out_dim;
        }
        let upsample = match upsample {
            Some(mode) => Some(Resample::load(
                map,
                &weights::join_key(prefix, "upsamplers.0"),
                out_dim,
                mode,
                device,
            )?),
            None => None,
        };
        Ok(Self { resnets, upsample })
    }

    fn forward(&self, mut xs: Tensor<B, 5>, mut cache: Option<&mut FeatCache>) -> Tensor<B, 5> {
        for r in &self.resnets {
            xs = r.forward(xs, cache.as_deref_mut());
        }
        if let Some(up) = &self.upsample {
            xs = up.forward(xs, cache.as_deref_mut());
        }
        xs
    }
}

#[derive(Debug, Clone)]
struct WanDecoder {
    conv_in: CausalConv3d,
    mid_res0: ResidualBlock,
    mid_attn: AttentionBlock,
    mid_res1: ResidualBlock,
    up_blocks: Vec<UpBlock>,
    norm_out: Tensor<B, 4>,
    conv_out: CausalConv3d,
}

impl WanDecoder {
    fn zeros(cfg: &WanVaeConfig, device: &Device) -> Self {
        let mut dims: Vec<usize> = vec![cfg.base_dim * *cfg.dim_mult.last().unwrap()];
        for u in cfg.dim_mult.iter().rev() {
            dims.push(cfg.base_dim * *u);
        }
        let conv_in =
            CausalConv3d::zeros_k(cfg.z_dim, dims[0], 3, [1, 1, 1], [1, 1, 1], device);
        let mid_res0 = ResidualBlock::zeros(dims[0], dims[0], device);
        let mid_attn = AttentionBlock::zeros(dims[0], device);
        let mid_res1 = ResidualBlock::zeros(dims[0], dims[0], device);
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
            up_blocks.push(UpBlock::zeros(
                in_dim,
                out_dim,
                cfg.num_res_blocks,
                mode,
                device,
            ));
        }
        let out_dim = *dims.last().unwrap();
        Self {
            conv_in,
            mid_res0,
            mid_attn,
            mid_res1,
            up_blocks,
            norm_out: Tensor::zeros([out_dim, 1, 1, 1], device),
            conv_out: CausalConv3d::zeros_k(out_dim, 3, 3, [1, 1, 1], [1, 1, 1], device),
        }
    }

    fn load(map: &WeightMap, prefix: &str, cfg: &WanVaeConfig, device: &Device) -> Result<Self> {
        let mut dims: Vec<usize> = vec![cfg.base_dim * *cfg.dim_mult.last().unwrap()];
        for u in cfg.dim_mult.iter().rev() {
            dims.push(cfg.base_dim * *u);
        }
        let conv_in = CausalConv3d::load_k(
            map,
            &weights::join_key(prefix, "conv_in"),
            cfg.z_dim,
            dims[0],
            3,
            [1, 1, 1],
            [1, 1, 1],
            device,
        )?;
        let mid_res0 = ResidualBlock::load(
            map,
            &weights::join_key(prefix, "mid_block.resnets.0"),
            dims[0],
            dims[0],
            device,
        )?;
        let mid_attn = AttentionBlock::load(
            map,
            &weights::join_key(prefix, "mid_block.attentions.0"),
            dims[0],
            device,
        )?;
        let mid_res1 = ResidualBlock::load(
            map,
            &weights::join_key(prefix, "mid_block.resnets.1"),
            dims[0],
            dims[0],
            device,
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
                device,
            )?);
        }
        let out_dim = *dims.last().unwrap();
        Ok(Self {
            conv_in,
            mid_res0,
            mid_attn,
            mid_res1,
            up_blocks,
            norm_out: weights::tensor4_shaped(
                map,
                &weights::join_key(prefix, "norm_out.gamma"),
                [out_dim, 1, 1, 1],
                device,
            )?,
            conv_out: CausalConv3d::load_k(
                map,
                &weights::join_key(prefix, "conv_out"),
                out_dim,
                3,
                3,
                [1, 1, 1],
                [1, 1, 1],
                device,
            )?,
        })
    }

    fn forward(&self, zs: Tensor<B, 5>, mut cache: Option<&mut FeatCache>) -> Tensor<B, 5> {
        let mut x = conv_cached(&self.conv_in, zs, cache.as_deref_mut());
        x = self.mid_res0.forward(x, cache.as_deref_mut());
        x = self.mid_attn.forward(x);
        x = self.mid_res1.forward(x, cache.as_deref_mut());
        for up in &self.up_blocks {
            x = up.forward(x, cache.as_deref_mut());
        }
        x = rms_video(x, &self.norm_out);
        x = silu(x);
        x = conv_cached(&self.conv_out, x, cache.as_deref_mut());
        x.clamp(-1.0, 1.0)
    }
}

#[derive(Debug, Clone)]
pub struct AutoencoderKlWan {
    pub cfg: WanVaeConfig,
    post_quant: CausalConv3d,
    decoder: WanDecoder,
}

impl AutoencoderKlWan {
    pub fn zeros(cfg: WanVaeConfig, device: &Device) -> Self {
        Self {
            post_quant: CausalConv3d::zeros_k(
                cfg.z_dim,
                cfg.z_dim,
                1,
                [1, 1, 1],
                [0, 0, 0],
                device,
            ),
            decoder: WanDecoder::zeros(&cfg, device),
            cfg,
        }
    }

    pub fn load(cfg: WanVaeConfig, map: &WeightMap, device: &Device) -> Result<Self> {
        Self::from_map(cfg, map, device)
    }

    pub fn from_map(cfg: WanVaeConfig, map: &WeightMap, device: &Device) -> Result<Self> {
        Ok(Self {
            post_quant: CausalConv3d::load_k(
                map,
                "post_quant_conv",
                cfg.z_dim,
                cfg.z_dim,
                1,
                [1, 1, 1],
                [0, 0, 0],
                device,
            )?,
            decoder: WanDecoder::load(map, "decoder", &cfg, device)?,
            cfg,
        })
    }

    pub fn scale_latents(&self, latents: Tensor<B, 5>) -> Tensor<B, 5> {
        let n = self.cfg.z_dim.min(16);
        let device = latents.device();
        let mean = Tensor::<B, 1>::from_floats(&LATENTS_MEAN[..n], &device)
            .reshape([1, n, 1, 1, 1]);
        let std = Tensor::<B, 1>::from_floats(&LATENTS_STD[..n], &device).reshape([1, n, 1, 1, 1]);
        latents * std + mean
    }

    pub fn decode(&self, latents: Tensor<B, 5>) -> Tensor<B, 5> {
        let z = self.post_quant.forward(latents);
        let t = z.dims()[2];
        let mut cache = FeatCache::new();
        let mut frames = Vec::with_capacity(t);
        for i in 0..t {
            cache.begin_pass();
            frames.push(self.decoder.forward(z.clone().narrow(2, i, 1), Some(&mut cache)));
        }
        Tensor::cat(frames, 2)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tiny_vae_decode_shape() {
        let device = Default::default();
        let cfg = WanVaeConfig::tiny();
        let vae = AutoencoderKlWan::zeros(cfg, &device);
        let z = Tensor::<B, 5>::zeros([1, 4, 2, 4, 4], &device);
        let out = vae.decode(z);
        assert_eq!(out.dims(), [1, 3, 2, 8, 8]);
    }

    #[test]
    fn feat_cache_decode_is_4n_plus_1() {
        let device = Default::default();
        let cfg = WanVaeConfig {
            base_dim: 8,
            z_dim: 4,
            dim_mult: vec![1, 2, 2],
            num_res_blocks: 1,
            temporal_upsample: vec![true, true],
            load_encoder: false,
        };
        let vae = AutoencoderKlWan::zeros(cfg, &device);
        let z = Tensor::<B, 5>::zeros([1, 4, 3, 2, 2], &device);
        let out = vae.decode(z);
        assert_eq!(
            out.dims(),
            [1, 3, 9, 8, 8],
            "two upsample3d stages: 3 latents → 9 RGB"
        );
    }

    #[test]
    fn feat_cache_single_latent_skips_time_double() {
        let device = Default::default();
        let cfg = WanVaeConfig {
            base_dim: 8,
            z_dim: 4,
            dim_mult: vec![1, 2],
            num_res_blocks: 1,
            temporal_upsample: vec![true],
            load_encoder: false,
        };
        let vae = AutoencoderKlWan::zeros(cfg, &device);
        let z = Tensor::<B, 5>::zeros([1, 4, 1, 2, 2], &device);
        let out = vae.decode(z);
        assert_eq!(out.dims()[2], 1, "first chunk Rep skips upsample3d time conv");
    }
}
