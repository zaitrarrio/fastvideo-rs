//! Wan 2.1 VAE with Diffusers/Wan feat-cache decode.
//!
//! Decode walks one latent frame at a time and keeps `CACHE_T=2` activations at
//! each causal conv. Upsample3d uses the `"Rep"` first-chunk sentinel so time
//! doubling is skipped on latent 0, yielding `4n+1` RGB frames. Combined with
//! tiled `conv2d` and BF16 weights on CUDA, peak activation memory is O(1) in
//! clip duration.

use candle_core::{DType, Device, Result, Tensor, D};
use candle_nn::VarBuilder;

use crate::nn;

const CACHE_T: usize = 2;

const LATENTS_MEAN: [f32; 16] = [
    -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508, 0.4134, -0.0715, 0.5517,
    -0.3632, -0.1922, -0.9497, 0.2503, -0.2921,
];
const LATENTS_STD: [f32; 16] = [
    2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743, 3.2687, 2.1526, 2.8652, 1.5579,
    1.6382, 1.1253, 2.8251, 1.9160,
];

#[derive(Debug, Clone)]
pub struct WanVaeConfig {
    pub base_dim: usize,
    pub z_dim: usize,
    pub dim_mult: Vec<usize>,
    pub num_res_blocks: usize,
    pub temporal_upsample: Vec<bool>,
    pub load_encoder: bool,
}

impl WanVaeConfig {
    pub fn wan_2_1() -> Self {
        Self {
            base_dim: 96,
            z_dim: 16,
            dim_mult: vec![1, 2, 4, 4],
            num_res_blocks: 2,
            temporal_upsample: vec![true, true, false],
            load_encoder: true,
        }
    }

    pub fn tiny() -> Self {
        Self {
            base_dim: 8,
            z_dim: 4,
            dim_mult: vec![1, 2],
            num_res_blocks: 1,
            temporal_upsample: vec![false],
            load_encoder: false,
        }
    }
}

#[derive(Clone)]
enum CacheSlot {
    Empty,
    Rep,
    Tensor(Tensor),
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

fn last_frames(x: &Tensor, n: usize) -> Result<Tensor> {
    let t = x.dim(2)?;
    let take = t.min(n);
    x.narrow(2, t - take, take)?.contiguous()
}

fn conv_cached(conv: &CausalConv3d, x: &Tensor, cache: Option<&mut FeatCache>) -> Result<Tensor> {
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
            cache_x = Tensor::cat(&[last, cache_x], 2)?;
        }
    }
    let y = conv.forward_with_cache(x, prev.as_ref())?;
    cache.slots[i] = CacheSlot::Tensor(cache_x);
    Ok(y)
}

fn double_time(x: Tensor) -> Result<Tensor> {
    let (b, c2, t, h, w) = x.dims5()?;
    let c = c2 / 2;
    x.reshape((b, 2, c, t, h, w))?
        .permute((0, 2, 3, 1, 4, 5))?
        .contiguous()?
        .reshape((b, c, t * 2, h, w))
}

#[derive(Debug, Clone)]
struct CausalConv3d {
    weight: Tensor,
    bias: Tensor,
    stride: [usize; 3],
    pad: [usize; 3],
}

impl CausalConv3d {
    fn load(in_c: usize, out_c: usize, k: usize, stride: [usize; 3], pad: [usize; 3], vb: VarBuilder) -> Result<Self> {
        Self::load_k(in_c, out_c, [k, k, k], stride, pad, vb)
    }

    fn load_k(
        in_c: usize,
        out_c: usize,
        kernel: [usize; 3],
        stride: [usize; 3],
        pad: [usize; 3],
        vb: VarBuilder,
    ) -> Result<Self> {
        Ok(Self {
            weight: vb.get((out_c, in_c, kernel[0], kernel[1], kernel[2]), "weight")?,
            bias: vb.get(out_c, "bias")?,
            stride,
            pad,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.forward_with_cache(xs, None)
    }

    fn forward_with_cache(&self, xs: &Tensor, cache_x: Option<&Tensor>) -> Result<Tensor> {
        let (b, _c, _t, _h, _w) = xs.dims5()?;
        let (out_c, in_c, kt, kh, kw) = self.weight.dims5()?;
        let mut x = xs.clone();
        let mut pad_t = 2 * self.pad[0];
        if let Some(c) = cache_x {
            if pad_t > 0 {
                x = Tensor::cat(&[c, &x], 2)?;
                pad_t = pad_t.saturating_sub(c.dim(2)?);
            }
        }
        if self.pad[2] > 0 {
            x = x.pad_with_zeros(4, self.pad[2], self.pad[2])?;
        }
        if self.pad[1] > 0 {
            x = x.pad_with_zeros(3, self.pad[1], self.pad[1])?;
        }
        if pad_t > 0 {
            x = x.pad_with_zeros(2, pad_t, 0)?;
        }
        let (_b, _c, t_p, h_p, w_p) = x.dims5()?;
        if kt == 1 && kh == 1 && kw == 1 && self.stride == [1, 1, 1] {
            let x = x
                .permute((0, 2, 3, 4, 1))?
                .contiguous()?
                .reshape((b * t_p * h_p * w_p, in_c))?;
            let w = self.weight.reshape((out_c, in_c))?.t()?.to_dtype(xs.dtype())?;
            let y = x.matmul(&w)?;
            let y = y.broadcast_add(&self.bias.to_dtype(xs.dtype())?)?;
            return y
                .reshape((b, t_p, h_p, w_p, out_c))?
                .permute((0, 4, 1, 2, 3))?
                .contiguous();
        }
        let w2 = self
            .weight
            .to_dtype(xs.dtype())?
            .reshape((out_c, in_c * kt, kh, kw))?;
        let mut frames = Vec::new();
        let mut ti = 0usize;
        let t_stride = self.stride[0].max(1);
        while ti + kt <= t_p {
            let window = x.narrow(2, ti, kt)?;
            let window = window.contiguous()?.reshape((b, in_c * kt, h_p, w_p))?;
            let y = nn::conv2d(&window, &w2, 0, self.stride[1])?;
            let y = y.broadcast_add(
                &self.bias.to_dtype(xs.dtype())?.reshape((1, out_c, 1, 1))?,
            )?;
            frames.push(y.unsqueeze(2)?);
            ti += t_stride;
        }
        if frames.is_empty() {
            candle_core::bail!("empty causal conv3d output");
        }
        Tensor::cat(&frames, 2)
    }
}

/// Channel-first RMS over dim=1, matching Wan `F.normalize * sqrt(C)`.
fn rms_video(xs: &Tensor, gamma: &Tensor) -> Result<Tensor> {
    if xs.elem_count() > 1_000_000 && xs.dim(2)? > 1 {
        return map_time(xs, |frame| rms_video_one(frame, gamma));
    }
    rms_video_one(xs, gamma)
}

fn rms_video_one(xs: &Tensor, gamma: &Tensor) -> Result<Tensor> {
    let x = xs.to_dtype(DType::F32)?;
    let var = x.sqr()?.mean_keepdim(1)?;
    let y = x.broadcast_div(&(var + 1e-12)?.sqrt()?)?;
    y.to_dtype(xs.dtype())?
        .broadcast_mul(&gamma.to_dtype(xs.dtype())?)
}

fn silu_video(xs: &Tensor) -> Result<Tensor> {
    if xs.elem_count() > 1_000_000 && xs.dim(2)? > 1 {
        return map_time(xs, nn::silu);
    }
    nn::silu(xs)
}

fn map_time(xs: &Tensor, mut f: impl FnMut(&Tensor) -> Result<Tensor>) -> Result<Tensor> {
    let t = xs.dim(2)?;
    let mut parts = Vec::with_capacity(t);
    for i in 0..t {
        parts.push(f(&xs.narrow(2, i, 1)?)?);
    }
    Tensor::cat(&parts, 2)
}

#[derive(Debug, Clone)]
struct ResidualBlock {
    norm1: Tensor,
    conv1: CausalConv3d,
    norm2: Tensor,
    conv2: CausalConv3d,
    shortcut: Option<CausalConv3d>,
}

impl ResidualBlock {
    fn load(in_dim: usize, out_dim: usize, vb: VarBuilder) -> Result<Self> {
        let shortcut = if in_dim != out_dim {
            Some(CausalConv3d::load(
                in_dim,
                out_dim,
                1,
                [1, 1, 1],
                [0, 0, 0],
                vb.pp("conv_shortcut"),
            )?)
        } else {
            None
        };
        Ok(Self {
            norm1: vb.pp("norm1").get((in_dim, 1, 1, 1), "gamma")?,
            conv1: CausalConv3d::load(in_dim, out_dim, 3, [1, 1, 1], [1, 1, 1], vb.pp("conv1"))?,
            norm2: vb.pp("norm2").get((out_dim, 1, 1, 1), "gamma")?,
            conv2: CausalConv3d::load(out_dim, out_dim, 3, [1, 1, 1], [1, 1, 1], vb.pp("conv2"))?,
            shortcut,
        })
    }

    fn forward(&self, xs: &Tensor, mut cache: Option<&mut FeatCache>) -> Result<Tensor> {
        let mut x = rms_video(xs, &self.norm1)?;
        x = silu_video(&x)?;
        x = conv_cached(&self.conv1, &x, cache.as_deref_mut())?;
        x = rms_video(&x, &self.norm2)?;
        x = silu_video(&x)?;
        x = conv_cached(&self.conv2, &x, cache.as_deref_mut())?;
        match &self.shortcut {
            Some(sc) => sc.forward(xs)? + x,
            None => xs + x,
        }
    }
}

#[derive(Debug, Clone)]
struct AttentionBlock {
    norm: Tensor,
    qkv: Tensor,
    qkv_bias: Tensor,
    proj: Tensor,
    proj_bias: Tensor,
}

impl AttentionBlock {
    fn load(dim: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            norm: vb.pp("norm").get((dim, 1, 1), "gamma")?,
            qkv: vb.pp("to_qkv").get((dim * 3, dim, 1, 1), "weight")?,
            qkv_bias: vb.pp("to_qkv").get(dim * 3, "bias")?,
            proj: vb.pp("proj").get((dim, dim, 1, 1), "weight")?,
            proj_bias: vb.pp("proj").get(dim, "bias")?,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (b, c, t, h, w) = xs.dims5()?;
        let mut x = xs.transpose(1, 2)?.contiguous()?.reshape((b * t, c, h, w))?;
        let gamma = self.norm.reshape((1, c, 1, 1))?;
        let xf = x.to_dtype(DType::F32)?;
        let var = xf.sqr()?.mean_keepdim(1)?;
        x = xf
            .broadcast_div(&(var + 1e-12)?.sqrt()?)?
            .to_dtype(xs.dtype())?
            .broadcast_mul(&gamma.to_dtype(xs.dtype())?)?;
        let qkv = nn::conv2d(&x, &self.qkv.to_dtype(xs.dtype())?, 0, 1)?.broadcast_add(
            &self
                .qkv_bias
                .to_dtype(xs.dtype())?
                .reshape((1, c * 3, 1, 1))?,
        )?;
        let hw = h * w;
        let qkv = qkv
            .reshape((b * t, 1, c * 3, hw))?
            .permute((0, 1, 3, 2))?;
        let chunks = qkv.chunk(3, D::Minus1)?;
        let attn = nn::scaled_dot_product_attention(&chunks[0], &chunks[1], &chunks[2], None)?;
        let attn = attn
            .squeeze(1)?
            .permute((0, 2, 1))?
            .reshape((b * t, c, h, w))?;
        let y = nn::conv2d(&attn, &self.proj.to_dtype(xs.dtype())?, 0, 1)?.broadcast_add(
            &self.proj_bias.to_dtype(xs.dtype())?.reshape((1, c, 1, 1))?,
        )?;
        let y = y.reshape((b, t, c, h, w))?.permute((0, 2, 1, 3, 4))?;
        xs + y
    }
}

#[derive(Debug, Clone)]
struct Resample {
    mode: ResampleMode,
    conv: Option<(Tensor, Tensor)>,
    time_conv: Option<CausalConv3d>,
}

#[derive(Debug, Clone, Copy)]
enum ResampleMode {
    Upsample2d,
    Upsample3d,
}

impl Resample {
    fn load(dim: usize, mode: ResampleMode, vb: VarBuilder) -> Result<Self> {
        let (conv, time_conv) = match mode {
            ResampleMode::Upsample2d => {
                let out = dim / 2;
                (
                    Some((
                        vb.pp("resample").pp("1").get((out, dim, 3, 3), "weight")?,
                        vb.pp("resample").pp("1").get(out, "bias")?,
                    )),
                    None,
                )
            }
            ResampleMode::Upsample3d => {
                let out = dim / 2;
                (
                    Some((
                        vb.pp("resample").pp("1").get((out, dim, 3, 3), "weight")?,
                        vb.pp("resample").pp("1").get(out, "bias")?,
                    )),
                    Some(CausalConv3d::load_k(
                        dim,
                        dim * 2,
                        [3, 1, 1],
                        [1, 1, 1],
                        [1, 0, 0],
                        vb.pp("time_conv"),
                    )?),
                )
            }
        };
        Ok(Self {
            mode,
            conv,
            time_conv,
        })
    }

    fn forward(&self, xs: &Tensor, cache: Option<&mut FeatCache>) -> Result<Tensor> {
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
                                    cache_x = Tensor::cat(&[last, cache_x], 2)?;
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
        if let Some((w_conv, bias)) = &self.conv {
            let (b, c, t, h, w) = x.dims5()?;
            let w_conv = w_conv.to_dtype(xs.dtype())?;
            let bias = bias
                .to_dtype(xs.dtype())?
                .reshape((1, w_conv.dim(0)?, 1, 1))?;
            let up_elems = b.saturating_mul(t).saturating_mul(c).saturating_mul(h * 2).saturating_mul(w * 2);
            if t > 1 && up_elems > 1_000_000 {
                let mut frames = Vec::with_capacity(t);
                for i in 0..t {
                    let frame = x.narrow(2, i, 1)?.squeeze(2)?.contiguous()?;
                    let up = frame.upsample_nearest2d(h * 2, w * 2)?;
                    let y = nn::conv2d(&up, &w_conv, 1, 1)?.broadcast_add(&bias)?;
                    frames.push(y.unsqueeze(2)?);
                }
                x = Tensor::cat(&frames, 2)?;
            } else {
                let x2 = x.transpose(1, 2)?.contiguous()?.reshape((b * t, c, h, w))?;
                let up = x2.upsample_nearest2d(h * 2, w * 2)?;
                let y = nn::conv2d(&up, &w_conv, 1, 1)?.broadcast_add(&bias)?;
                let oc = y.dim(1)?;
                x = y
                    .reshape((b, t, oc, h * 2, w * 2))?
                    .permute((0, 2, 1, 3, 4))?;
            }
        }
        Ok(x)
    }
}

#[derive(Debug, Clone)]
struct UpBlock {
    resnets: Vec<ResidualBlock>,
    upsample: Option<Resample>,
}

impl UpBlock {
    fn load(
        in_dim: usize,
        out_dim: usize,
        n_res: usize,
        upsample: Option<ResampleMode>,
        vb: VarBuilder,
    ) -> Result<Self> {
        let mut resnets = Vec::new();
        let mut current = in_dim;
        for i in 0..=n_res {
            resnets.push(ResidualBlock::load(
                current,
                out_dim,
                vb.pp("resnets").pp(&i.to_string()),
            )?);
            current = out_dim;
        }
        let upsample = match upsample {
            Some(mode) => Some(Resample::load(out_dim, mode, vb.pp("upsamplers").pp("0"))?),
            None => None,
        };
        Ok(Self { resnets, upsample })
    }

    fn forward(&self, mut xs: Tensor, mut cache: Option<&mut FeatCache>) -> Result<Tensor> {
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
    norm_out: Tensor,
    conv_out: CausalConv3d,
}

impl WanDecoder {
    pub fn load(cfg: &WanVaeConfig, vb: VarBuilder) -> Result<Self> {
        let mut dims: Vec<usize> = vec![cfg.base_dim * *cfg.dim_mult.last().unwrap()];
        for u in cfg.dim_mult.iter().rev() {
            dims.push(cfg.base_dim * *u);
        }
        let conv_in = CausalConv3d::load(cfg.z_dim, dims[0], 3, [1, 1, 1], [1, 1, 1], vb.pp("conv_in"))?;
        let mid = vb.pp("mid_block");
        let mid_res0 = ResidualBlock::load(dims[0], dims[0], mid.pp("resnets").pp("0"))?;
        let mid_attn = AttentionBlock::load(dims[0], mid.pp("attentions").pp("0"))?;
        let mid_res1 = ResidualBlock::load(dims[0], dims[0], mid.pp("resnets").pp("1"))?;
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
                in_dim,
                out_dim,
                cfg.num_res_blocks,
                mode,
                vb.pp("up_blocks").pp(&i.to_string()),
            )?);
        }
        let out_dim = *dims.last().unwrap();
        Ok(Self {
            conv_in,
            mid_res0,
            mid_attn,
            mid_res1,
            up_blocks,
            norm_out: vb.pp("norm_out").get((out_dim, 1, 1, 1), "gamma")?,
            conv_out: CausalConv3d::load(out_dim, 3, 3, [1, 1, 1], [1, 1, 1], vb.pp("conv_out"))?,
        })
    }

    fn forward(&self, zs: &Tensor, mut cache: Option<&mut FeatCache>) -> Result<Tensor> {
        let mut x = conv_cached(&self.conv_in, zs, cache.as_deref_mut())?;
        x = self.mid_res0.forward(&x, cache.as_deref_mut())?;
        x = self.mid_attn.forward(&x)?;
        x = self.mid_res1.forward(&x, cache.as_deref_mut())?;
        for up in &self.up_blocks {
            x = up.forward(x, cache.as_deref_mut())?;
        }
        x = rms_video(&x, &self.norm_out)?;
        x = silu_video(&x)?;
        x = conv_cached(&self.conv_out, &x, cache.as_deref_mut())?;
        x.clamp(-1.0, 1.0)
    }
}

#[derive(Debug, Clone)]
struct EncoderDownsample {
    spatial_w: Tensor,
    spatial_b: Tensor,
    time_conv: Option<CausalConv3d>,
}

impl EncoderDownsample {
    fn load_2d(channels: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            spatial_w: vb.pp("resample").pp("1").get((channels, channels, 3, 3), "weight")?,
            spatial_b: vb.pp("resample").pp("1").get(channels, "bias")?,
            time_conv: None,
        })
    }

    fn load_3d(channels: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            spatial_w: vb.pp("resample").pp("1").get((channels, channels, 3, 3), "weight")?,
            spatial_b: vb.pp("resample").pp("1").get(channels, "bias")?,
            time_conv: Some(CausalConv3d::load_k(
                channels,
                channels,
                [3, 1, 1],
                [2, 1, 1],
                [1, 0, 0],
                vb.pp("time_conv"),
            )?),
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let mut x = xs.clone();
        if let Some(tc) = &self.time_conv {
            x = tc.forward(&x)?;
        }
        let (b, c, t, h, w) = x.dims5()?;
        let x2 = x.transpose(1, 2)?.contiguous()?.reshape((b * t, c, h, w))?;
        let y = nn::conv2d(&x2, &self.spatial_w.to_dtype(xs.dtype())?, 1, 2)?.broadcast_add(
            &self.spatial_b.to_dtype(xs.dtype())?.reshape((1, c, 1, 1))?,
        )?;
        let (_, oc, hh, ww) = y.dims4()?;
        y.reshape((b, t, oc, hh, ww))?.permute((0, 2, 1, 3, 4))
    }
}

#[derive(Debug, Clone)]
struct WanEncoder {
    conv_in: CausalConv3d,
    blocks: Vec<EncoderStage>,
    mid_res0: ResidualBlock,
    mid_attn: AttentionBlock,
    mid_res1: ResidualBlock,
    norm_out: Tensor,
    conv_out: CausalConv3d,
    quant: CausalConv3d,
}

#[derive(Debug, Clone)]
enum EncoderStage {
    Res(ResidualBlock),
    Down(EncoderDownsample),
}

impl WanEncoder {
    fn load_wan_2_1(vb: VarBuilder) -> Result<Self> {
        let d = vb.pp("encoder");
        let mut blocks = Vec::new();
        blocks.push(EncoderStage::Res(ResidualBlock::load(96, 96, d.pp("down_blocks").pp("0"))?));
        blocks.push(EncoderStage::Res(ResidualBlock::load(96, 96, d.pp("down_blocks").pp("1"))?));
        blocks.push(EncoderStage::Down(EncoderDownsample::load_2d(
            96,
            d.pp("down_blocks").pp("2"),
        )?));
        blocks.push(EncoderStage::Res(ResidualBlock::load(96, 192, d.pp("down_blocks").pp("3"))?));
        blocks.push(EncoderStage::Res(ResidualBlock::load(192, 192, d.pp("down_blocks").pp("4"))?));
        blocks.push(EncoderStage::Down(EncoderDownsample::load_3d(
            192,
            d.pp("down_blocks").pp("5"),
        )?));
        blocks.push(EncoderStage::Res(ResidualBlock::load(192, 384, d.pp("down_blocks").pp("6"))?));
        blocks.push(EncoderStage::Res(ResidualBlock::load(384, 384, d.pp("down_blocks").pp("7"))?));
        blocks.push(EncoderStage::Down(EncoderDownsample::load_3d(
            384,
            d.pp("down_blocks").pp("8"),
        )?));
        blocks.push(EncoderStage::Res(ResidualBlock::load(384, 384, d.pp("down_blocks").pp("9"))?));
        blocks.push(EncoderStage::Res(ResidualBlock::load(384, 384, d.pp("down_blocks").pp("10"))?));
        let mid = d.pp("mid_block");
        Ok(Self {
            conv_in: CausalConv3d::load(3, 96, 3, [1, 1, 1], [1, 1, 1], d.pp("conv_in"))?,
            blocks,
            mid_res0: ResidualBlock::load(384, 384, mid.pp("resnets").pp("0"))?,
            mid_attn: AttentionBlock::load(384, mid.pp("attentions").pp("0"))?,
            mid_res1: ResidualBlock::load(384, 384, mid.pp("resnets").pp("1"))?,
            norm_out: d.pp("norm_out").get((384, 1, 1, 1), "gamma")?,
            conv_out: CausalConv3d::load(384, 32, 3, [1, 1, 1], [1, 1, 1], d.pp("conv_out"))?,
            quant: CausalConv3d::load(32, 32, 1, [1, 1, 1], [0, 0, 0], vb.pp("quant_conv"))?,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
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
        x = rms_video(&x, &self.norm_out)?;
        x = silu_video(&x)?;
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
    pub fn load(cfg: WanVaeConfig, vb: VarBuilder) -> Result<Self> {
        let encoder = if cfg.load_encoder {
            Some(WanEncoder::load_wan_2_1(vb.clone())?)
        } else {
            None
        };
        Ok(Self {
            post_quant: CausalConv3d::load(
                cfg.z_dim,
                cfg.z_dim,
                1,
                [1, 1, 1],
                [0, 0, 0],
                vb.pp("post_quant_conv"),
            )?,
            decoder: WanDecoder::load(&cfg, vb.pp("decoder"))?,
            encoder,
            cfg,
        })
    }

    pub fn device(&self) -> &Device {
        self.post_quant.weight.device()
    }

    pub fn scale_latents(&self, latents: &Tensor) -> Result<Tensor> {
        let device = latents.device();
        let n = self.cfg.z_dim.min(16);
        let mean = Tensor::from_slice(&LATENTS_MEAN[..n], (1, n, 1, 1, 1), device)?
            .to_dtype(latents.dtype())?;
        let std = Tensor::from_slice(&LATENTS_STD[..n], (1, n, 1, 1, 1), device)?
            .to_dtype(latents.dtype())?;
        latents.broadcast_mul(&std)?.broadcast_add(&mean)
    }

    pub fn normalize_latents(&self, latents: &Tensor) -> Result<Tensor> {
        let device = latents.device();
        let n = self.cfg.z_dim.min(16);
        let mean = Tensor::from_slice(&LATENTS_MEAN[..n], (1, n, 1, 1, 1), device)?
            .to_dtype(latents.dtype())?;
        let std = Tensor::from_slice(&LATENTS_STD[..n], (1, n, 1, 1, 1), device)?
            .to_dtype(latents.dtype())?;
        (latents.broadcast_sub(&mean)?).broadcast_div(&std)
    }

    pub fn encode_video(&self, video: &Tensor) -> Result<Tensor> {
        let enc = self
            .encoder
            .as_ref()
            .ok_or_else(|| candle_core::Error::Msg("VAE encoder not loaded".into()))?;
        enc.forward(video)
    }

    pub fn decode(&self, latents: &Tensor) -> Result<Tensor> {
        let z = self.post_quant.forward(latents)?;
        let t = z.dim(2)?;
        let mut cache = FeatCache::new();
        let mut frames = Vec::with_capacity(t);
        for i in 0..t {
            cache.begin_pass();
            if t > 4 {
                eprintln!("vae feat-cache decode latent {}/{t}", i + 1);
            }
            frames.push(self.decoder.forward(&z.narrow(2, i, 1)?, Some(&mut cache))?);
        }
        Tensor::cat(&frames, 2)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};

    #[test]
    fn tiny_vae_decode_shape() {
        let device = Device::Cpu;
        let cfg = WanVaeConfig::tiny();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let vae = AutoencoderKlWan::load(cfg, vb).unwrap();
        let z = Tensor::zeros((1, 4, 2, 4, 4), DType::F32, &device).unwrap();
        let out = vae.decode(&z).unwrap();
        assert_eq!(out.dim(0).unwrap(), 1);
        assert_eq!(out.dim(1).unwrap(), 3);
        assert_eq!(out.dim(2).unwrap(), 2);
        assert_eq!(out.dim(3).unwrap(), 8);
        assert_eq!(out.dim(4).unwrap(), 8);
    }

    #[test]
    fn feat_cache_decode_is_4n_plus_1() {
        let device = Device::Cpu;
        let cfg = WanVaeConfig {
            base_dim: 8,
            z_dim: 4,
            dim_mult: vec![1, 2, 2],
            num_res_blocks: 1,
            temporal_upsample: vec![true, true],
            load_encoder: false,
        };
        let vb = VarBuilder::zeros(DType::F32, &device);
        let vae = AutoencoderKlWan::load(cfg, vb).unwrap();
        let z = Tensor::zeros((1, 4, 3, 2, 2), DType::F32, &device).unwrap();
        let out = vae.decode(&z).unwrap();
        assert_eq!(out.dims(), &[1, 3, 9, 8, 8], "two upsample3d stages: 3 latents → 9 RGB");
    }

    #[test]
    fn feat_cache_single_latent_skips_time_double() {
        let device = Device::Cpu;
        let cfg = WanVaeConfig {
            base_dim: 8,
            z_dim: 4,
            dim_mult: vec![1, 2],
            num_res_blocks: 1,
            temporal_upsample: vec![true],
            load_encoder: false,
        };
        let vb = VarBuilder::zeros(DType::F32, &device);
        let vae = AutoencoderKlWan::load(cfg, vb).unwrap();
        let z = Tensor::zeros((1, 4, 1, 2, 2), DType::F32, &device).unwrap();
        let out = vae.decode(&z).unwrap();
        assert_eq!(out.dim(2).unwrap(), 1, "first chunk Rep skips upsample3d time conv");
    }
}
