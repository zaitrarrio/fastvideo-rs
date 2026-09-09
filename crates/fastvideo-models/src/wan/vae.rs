//! Wan 2.1 VAE decoder (full-sequence causal conv3d, no streaming cache).

use candle_core::{DType, Result, Tensor, D};
use candle_nn::VarBuilder;

use crate::nn;

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
}

impl WanVaeConfig {
    pub fn wan_2_1() -> Self {
        Self {
            base_dim: 96,
            z_dim: 16,
            dim_mult: vec![1, 2, 4, 4],
            num_res_blocks: 2,
            temporal_upsample: vec![true, true, false],
        }
    }

    pub fn tiny() -> Self {
        Self {
            base_dim: 8,
            z_dim: 4,
            dim_mult: vec![1, 2],
            num_res_blocks: 1,
            temporal_upsample: vec![false],
        }
    }
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
        let (b, _c, _t, _h, _w) = xs.dims5()?;
        let (out_c, in_c, kt, kh, kw) = self.weight.dims5()?;
        let mut x = xs.clone();
        if self.pad[2] > 0 {
            x = x.pad_with_zeros(4, self.pad[2], self.pad[2])?;
        }
        if self.pad[1] > 0 {
            x = x.pad_with_zeros(3, self.pad[1], self.pad[1])?;
        }
        if self.pad[0] > 0 {
            x = x.pad_with_zeros(2, 2 * self.pad[0], 0)?;
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
        Tensor::cat(&frames, 2)
    }
}

/// Channel-first RMS over dim=1, matching Wan `F.normalize * sqrt(C)`.
fn rms_video(xs: &Tensor, gamma: &Tensor) -> Result<Tensor> {
    let x = xs.to_dtype(DType::F32)?;
    let var = x.sqr()?.mean_keepdim(1)?;
    let y = x.broadcast_div(&(var + 1e-12)?.sqrt()?)?;
    y.to_dtype(xs.dtype())?
        .broadcast_mul(&gamma.to_dtype(xs.dtype())?)
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

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let h = match &self.shortcut {
            Some(sc) => sc.forward(xs)?,
            None => xs.clone(),
        };
        let mut x = rms_video(xs, &self.norm1)?;
        x = nn::silu(&x)?;
        x = self.conv1.forward(&x)?;
        x = rms_video(&x, &self.norm2)?;
        x = nn::silu(&x)?;
        x = self.conv2.forward(&x)?;
        h + x
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
        let attn = nn::scaled_dot_product_attention(&chunks[0], &chunks[1], &chunks[2])?;
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

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let mut x = xs.clone();
        if matches!(self.mode, ResampleMode::Upsample3d) {
            if let Some(tc) = &self.time_conv {
                x = tc.forward(&x)?;
                let (b, c2, t, h, w) = x.dims5()?;
                let c = c2 / 2;
                x = x
                    .reshape((b, 2, c, t, h, w))?
                    .permute((0, 2, 3, 1, 4, 5))?
                    .contiguous()?
                    .reshape((b, c, t * 2, h, w))?;
            }
        }
        if let Some((w_conv, bias)) = &self.conv {
            let (b, c, t, h, w) = x.dims5()?;
            let x2 = x.transpose(1, 2)?.contiguous()?.reshape((b * t, c, h, w))?;
            let up = x2.upsample_nearest2d(h * 2, w * 2)?;
            let y = nn::conv2d(&up, &w_conv.to_dtype(xs.dtype())?, 1, 1)?.broadcast_add(
                &bias
                    .to_dtype(xs.dtype())?
                    .reshape((1, w_conv.dim(0)?, 1, 1))?,
            )?;
            let oc = y.dim(1)?;
            x = y
                .reshape((b, t, oc, h * 2, w * 2))?
                .permute((0, 2, 1, 3, 4))?;
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

    fn forward(&self, mut xs: Tensor) -> Result<Tensor> {
        for r in &self.resnets {
            xs = r.forward(&xs)?;
        }
        if let Some(up) = &self.upsample {
            xs = up.forward(&xs)?;
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

    pub fn forward(&self, zs: &Tensor) -> Result<Tensor> {
        let mut x = self.conv_in.forward(zs)?;
        x = self.mid_res0.forward(&x)?;
        x = self.mid_attn.forward(&x)?;
        x = self.mid_res1.forward(&x)?;
        for up in &self.up_blocks {
            x = up.forward(x)?;
        }
        x = rms_video(&x, &self.norm_out)?;
        x = nn::silu(&x)?;
        x = self.conv_out.forward(&x)?;
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
    pub fn load(cfg: WanVaeConfig, vb: VarBuilder) -> Result<Self> {
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
            cfg,
        })
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

    pub fn decode(&self, latents: &Tensor) -> Result<Tensor> {
        let z = self.post_quant.forward(latents)?;
        self.decoder.forward(&z)
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
}
