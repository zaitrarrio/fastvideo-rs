//! Flux2 2D VAE (AutoencoderKLFlux2) decode path.
//!
//! Tiny graph is a short conv stack for CI smoke. Full decode loads Diffusers
//! `decoder.*` / `post_quant_conv` keys and walks ResNet + upsample blocks.

use candle_core::{DType, Result, Tensor};
use candle_nn::VarBuilder;

use crate::nn;

use super::config::Flux2VaeConfig;

struct Conv2d {
    weight: Tensor,
    bias: Option<Tensor>,
    stride: usize,
    padding: usize,
}

impl Conv2d {
    fn load(vb: VarBuilder, cin: usize, cout: usize, k: usize, stride: usize, padding: usize) -> Result<Self> {
        Ok(Self {
            weight: vb.get((cout, cin, k, k), "weight")?,
            bias: vb.get(cout, "bias").ok(),
            stride,
            padding,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let mut y = nn::conv2d(xs, &self.weight.to_dtype(xs.dtype())?, self.padding, self.stride)?;
        if let Some(bias) = &self.bias {
            y = y.broadcast_add(&bias.to_dtype(xs.dtype())?.reshape((1, bias.dims1()?, 1, 1))?)?;
        }
        Ok(y)
    }
}

struct GroupNorm {
    weight: Tensor,
    bias: Tensor,
    groups: usize,
    eps: f64,
}

impl GroupNorm {
    fn load(vb: VarBuilder, channels: usize, groups: usize, eps: f64) -> Result<Self> {
        Ok(Self {
            weight: vb.get(channels, "weight")?,
            bias: vb.get(channels, "bias")?,
            groups,
            eps,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (b, c, h, w) = xs.dims4()?;
        let g = self.groups.min(c).max(1);
        let x = xs.to_dtype(DType::F32)?.reshape((b, g, c / g, h, w))?;
        let mean = x.mean_keepdim(2)?.mean_keepdim(3)?.mean_keepdim(4)?;
        let centered = x.broadcast_sub(&mean)?;
        let var = centered.sqr()?.mean_keepdim(2)?.mean_keepdim(3)?.mean_keepdim(4)?;
        let y = centered.broadcast_div(&(var + self.eps)?.sqrt()?)?;
        let y = y.reshape((b, c, h, w))?.to_dtype(xs.dtype())?;
        let w = self.weight.to_dtype(xs.dtype())?.reshape((1, c, 1, 1))?;
        let b = self.bias.to_dtype(xs.dtype())?.reshape((1, c, 1, 1))?;
        y.broadcast_mul(&w)?.broadcast_add(&b)
    }
}

struct ResnetBlock2d {
    norm1: GroupNorm,
    conv1: Conv2d,
    norm2: GroupNorm,
    conv2: Conv2d,
    skip: Option<Conv2d>,
}

impl ResnetBlock2d {
    fn load(vb: VarBuilder, cin: usize, cout: usize, groups: usize) -> Result<Self> {
        Ok(Self {
            norm1: GroupNorm::load(vb.pp("norm1"), cin, groups, 1e-6)?,
            conv1: Conv2d::load(vb.pp("conv1"), cin, cout, 3, 1, 1)?,
            norm2: GroupNorm::load(vb.pp("norm2"), cout, groups, 1e-6)?,
            conv2: Conv2d::load(vb.pp("conv2"), cout, cout, 3, 1, 1)?,
            skip: if cin != cout {
                Some(Conv2d::load(vb.pp("conv_shortcut"), cin, cout, 1, 1, 0)?)
            } else {
                None
            },
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let h = self.conv1.forward(&nn::silu(&self.norm1.forward(xs)?)?)?;
        let h = self.conv2.forward(&nn::silu(&self.norm2.forward(&h)?)?)?;
        let skip = match &self.skip {
            Some(c) => c.forward(xs)?,
            None => xs.clone(),
        };
        skip + h
    }
}

/// Decode-only AutoencoderKLFlux2.
pub struct AutoencoderKlFlux2 {
    pub cfg: Flux2VaeConfig,
    post_quant: Option<Conv2d>,
    conv_in: Conv2d,
    mid: Option<ResnetBlock2d>,
    ups: Vec<(Option<ResnetBlock2d>, Option<Conv2d>)>,
    conv_norm_out: Option<GroupNorm>,
    conv_out: Conv2d,
    tiny: bool,
}

impl AutoencoderKlFlux2 {
    pub fn load(cfg: Flux2VaeConfig, vb: VarBuilder) -> Result<Self> {
        let tiny = cfg.block_out_channels.len() <= 2;
        if tiny {
            return Self::load_tiny(cfg, vb);
        }
        let z = cfg.latent_channels;
        let c0 = *cfg.block_out_channels.last().unwrap_or(&512);
        let groups = cfg.norm_num_groups.min(c0).max(1);
        let post_quant = Conv2d::load(vb.pp("post_quant_conv"), z, z, 1, 1, 0).ok();
        let conv_in = Conv2d::load(vb.pp("decoder").pp("conv_in"), z, c0, 3, 1, 1)?;
        let mid = ResnetBlock2d::load(vb.pp("decoder").pp("mid_block").pp("resnets").pp(0), c0, c0, groups).ok();
        let mut ups = Vec::new();
        let chans = cfg.block_out_channels.clone();
        for (i, &ch) in chans.iter().rev().enumerate() {
            let block = vb.pp("decoder").pp("up_blocks").pp(i);
            let res = ResnetBlock2d::load(block.pp("resnets").pp(0), ch, ch, groups.min(ch).max(1)).ok();
            let up = Conv2d::load(block.pp("upsamplers").pp(0).pp("conv"), ch, ch, 3, 1, 1).ok();
            ups.push((res, up));
        }
        let conv_norm_out = GroupNorm::load(vb.pp("decoder").pp("conv_norm_out"), chans[0], groups.min(chans[0]).max(1), 1e-6).ok();
        let conv_out = Conv2d::load(vb.pp("decoder").pp("conv_out"), chans[0], cfg.out_channels, 3, 1, 1)?;
        Ok(Self {
            cfg,
            post_quant,
            conv_in,
            mid,
            ups,
            conv_norm_out,
            conv_out,
            tiny: false,
        })
    }

    fn load_tiny(cfg: Flux2VaeConfig, vb: VarBuilder) -> Result<Self> {
        let z = cfg.latent_channels;
        let mid = cfg.block_out_channels[0];
        Ok(Self {
            post_quant: None,
            conv_in: Conv2d::load(vb.pp("decoder").pp("conv_in"), z, mid, 3, 1, 1)?,
            mid: None,
            ups: Vec::new(),
            conv_norm_out: None,
            conv_out: Conv2d::load(vb.pp("decoder").pp("conv_out"), mid, cfg.out_channels, 3, 1, 1)?,
            cfg,
            tiny: true,
        })
    }

    pub fn decode(&self, latents: &Tensor) -> Result<Tensor> {
        let mut x = latents.clone();
        if x.rank() == 5 {
            let (b, c, t, h, w) = x.dims5()?;
            if t != 1 {
                candle_core::bail!("Flux2 VAE is image-only, got T={t}");
            }
            x = x.reshape((b, c, h, w))?;
        }
        x = (x * (1.0 / self.cfg.scaling_factor as f64))?;
        if let Some(pq) = &self.post_quant {
            x = pq.forward(&x)?;
        }
        x = self.conv_in.forward(&x)?;
        if let Some(mid) = &self.mid {
            x = mid.forward(&x)?;
        }
        if self.tiny {
            x = upsample_nearest2(&x)?;
            x = upsample_nearest2(&x)?;
        } else {
            for (res, up) in &self.ups {
                if let Some(r) = res {
                    x = r.forward(&x)?;
                }
                x = upsample_nearest2(&x)?;
                if let Some(u) = up {
                    x = u.forward(&x)?;
                }
            }
        }
        if let Some(n) = &self.conv_norm_out {
            x = nn::silu(&n.forward(&x)?)?;
        } else {
            x = nn::silu(&x)?;
        }
        x = self.conv_out.forward(&x)?;
        let (b, c, h, w) = x.dims4()?;
        x.reshape((b, c, 1, h, w))
    }
}

fn upsample_nearest2(xs: &Tensor) -> Result<Tensor> {
    let (b, c, h, w) = xs.dims4()?;
    xs.reshape((b, c, h, 1, w, 1))?
        .expand((b, c, h, 2, w, 2))?
        .reshape((b, c, h * 2, w * 2))
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn tiny_decode_writes_rgb() {
        let device = Device::Cpu;
        let cfg = Flux2VaeConfig::tiny();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let vae = AutoencoderKlFlux2::load(cfg.clone(), vb).unwrap();
        let z = Tensor::zeros((1, cfg.latent_channels, 1, 2, 2), DType::F32, &device).unwrap();
        let out = vae.decode(&z).unwrap();
        assert_eq!(out.dims()[0], 1);
        assert_eq!(out.dims()[1], 3);
        assert_eq!(out.dims()[2], 1);
        assert_eq!(out.dims()[3], 8);
        assert_eq!(out.dims()[4], 8);
    }
}
