//! Flux2 2D VAE (AutoencoderKLFlux2) decode path.
//!
//! Tiny graph is a short conv stack for CI smoke. Full decode walks the
//! Diffusers `Decoder`: `post_quant_conv` → `conv_in` → mid ResNets +
//! optional spatial attention → up blocks (`layers_per_block + 1` ResNets,
//! nearest-2× + 3×3 conv except the last block) → `conv_norm_out` / `conv_out`.

use candle_core::{DType, Result, Tensor};
use candle_nn::VarBuilder;

use crate::nn::{self, Linear};

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
        group_norm(xs, &self.weight, &self.bias, self.groups, self.eps)
    }
}

/// GroupNorm over `(C/G, H, W)` — same reduction Diffusers uses on the VAE.
pub fn group_norm(xs: &Tensor, weight: &Tensor, bias: &Tensor, groups: usize, eps: f64) -> Result<Tensor> {
    let (b, c, h, w) = xs.dims4()?;
    let g = groups.min(c).max(1);
    let x = xs.to_dtype(DType::F32)?.reshape((b, g, c / g, h, w))?;
    let mean = x.mean_keepdim(2)?.mean_keepdim(3)?.mean_keepdim(4)?;
    let centered = x.broadcast_sub(&mean)?;
    let var = centered.sqr()?.mean_keepdim(2)?.mean_keepdim(3)?.mean_keepdim(4)?;
    let y = centered.broadcast_div(&(var + eps)?.sqrt()?)?;
    let y = y.reshape((b, c, h, w))?.to_dtype(xs.dtype())?;
    let w = weight.to_dtype(xs.dtype())?.reshape((1, c, 1, 1))?;
    let b = bias.to_dtype(xs.dtype())?.reshape((1, c, 1, 1))?;
    y.broadcast_mul(&w)?.broadcast_add(&b)
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
            norm1: GroupNorm::load(vb.pp("norm1"), cin, groups.min(cin).max(1), 1e-6)?,
            conv1: Conv2d::load(vb.pp("conv1"), cin, cout, 3, 1, 1)?,
            norm2: GroupNorm::load(vb.pp("norm2"), cout, groups.min(cout).max(1), 1e-6)?,
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

/// Diffusers VAE mid-block `Attention` (single-head spatial, residual).
struct SpatialAttention {
    norm: GroupNorm,
    to_q: Linear,
    to_k: Linear,
    to_v: Linear,
    to_out: Linear,
}

impl SpatialAttention {
    fn load(vb: VarBuilder, channels: usize, groups: usize) -> Result<Self> {
        Ok(Self {
            norm: GroupNorm::load(vb.pp("group_norm"), channels, groups.min(channels).max(1), 1e-6)?,
            to_q: Linear::load(channels, channels, vb.pp("to_q"))?,
            to_k: Linear::load(channels, channels, vb.pp("to_k"))?,
            to_v: Linear::load(channels, channels, vb.pp("to_v"))?,
            to_out: Linear::load(channels, channels, vb.pp("to_out").pp(0))?,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (b, c, h, w) = xs.dims4()?;
        let n = self.norm.forward(xs)?;
        let seq = n
            .reshape((b, c, h * w))?
            .transpose(1, 2)?
            .contiguous()?;
        let q = self.to_q.forward(&seq)?.reshape((b, 1, h * w, c))?;
        let k = self.to_k.forward(&seq)?.reshape((b, 1, h * w, c))?;
        let v = self.to_v.forward(&seq)?.reshape((b, 1, h * w, c))?;
        let attn = nn::scaled_dot_product_attention(&q, &k, &v, None)?;
        let out = attn.reshape((b, h * w, c))?;
        let out = self
            .to_out
            .forward(&out)?
            .transpose(1, 2)?
            .reshape((b, c, h, w))?;
        xs + out
    }
}

struct MidBlock {
    resnets: Vec<ResnetBlock2d>,
    attn: Option<SpatialAttention>,
}

impl MidBlock {
    fn load(vb: VarBuilder, channels: usize, groups: usize) -> Result<Self> {
        let r0 = ResnetBlock2d::load(vb.pp("resnets").pp(0), channels, channels, groups)?;
        let attn = SpatialAttention::load(vb.pp("attentions").pp(0), channels, groups).ok();
        let r1 = ResnetBlock2d::load(vb.pp("resnets").pp(1), channels, channels, groups).ok();
        let mut resnets = vec![r0];
        if let Some(r) = r1 {
            resnets.push(r);
        }
        Ok(Self { resnets, attn })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let mut x = self.resnets[0].forward(xs)?;
        if let Some(attn) = &self.attn {
            x = attn.forward(&x)?;
        }
        for res in self.resnets.iter().skip(1) {
            x = res.forward(&x)?;
        }
        Ok(x)
    }
}

struct UpBlock {
    resnets: Vec<ResnetBlock2d>,
    upsample: Option<Conv2d>,
}

impl UpBlock {
    fn load(
        vb: VarBuilder,
        cin: usize,
        cout: usize,
        n_res: usize,
        groups: usize,
        add_upsample: bool,
    ) -> Result<Self> {
        let mut resnets = Vec::with_capacity(n_res);
        let mut current = cin;
        for i in 0..n_res {
            resnets.push(ResnetBlock2d::load(vb.pp("resnets").pp(i), current, cout, groups)?);
            current = cout;
        }
        let upsample = if add_upsample {
            Conv2d::load(vb.pp("upsamplers").pp(0).pp("conv"), cout, cout, 3, 1, 1).ok()
        } else {
            None
        };
        Ok(Self { resnets, upsample })
    }

    fn forward(&self, xs: &Tensor, upsample: bool) -> Result<Tensor> {
        let mut x = xs.clone();
        for res in &self.resnets {
            x = res.forward(&x)?;
        }
        if upsample {
            x = upsample_nearest2(&x)?;
            if let Some(u) = &self.upsample {
                x = u.forward(&x)?;
            }
        }
        Ok(x)
    }
}

/// Decode-only AutoencoderKLFlux2.
pub struct AutoencoderKlFlux2 {
    pub cfg: Flux2VaeConfig,
    post_quant: Option<Conv2d>,
    conv_in: Conv2d,
    mid: Option<MidBlock>,
    ups: Vec<UpBlock>,
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
        let chans = cfg.block_out_channels.clone();
        let c0 = *chans.last().unwrap_or(&512);
        let groups = cfg.norm_num_groups;
        let post_quant = Conv2d::load(vb.pp("post_quant_conv"), z, z, 1, 1, 0).ok();
        let conv_in = Conv2d::load(vb.pp("decoder").pp("conv_in"), z, c0, 3, 1, 1)?;
        let mid = MidBlock::load(vb.pp("decoder").pp("mid_block"), c0, groups.min(c0).max(1)).ok();
        let n_res = cfg.layers_per_block + 1;
        let mut ups = Vec::new();
        let reversed: Vec<usize> = chans.iter().rev().copied().collect();
        let mut prev = c0;
        for (i, &ch) in reversed.iter().enumerate() {
            let add_up = i + 1 != reversed.len();
            let block = vb.pp("decoder").pp("up_blocks").pp(i);
            ups.push(UpBlock::load(block, prev, ch, n_res, groups, add_up)?);
            prev = ch;
        }
        let last_ch = *chans.first().unwrap_or(&c0);
        let conv_norm_out =
            GroupNorm::load(vb.pp("decoder").pp("conv_norm_out"), last_ch, groups.min(last_ch).max(1), 1e-6).ok();
        let conv_out = Conv2d::load(vb.pp("decoder").pp("conv_out"), last_ch, cfg.out_channels, 3, 1, 1)?;
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
        if self.cfg.shift_factor != 0.0 {
            x = (x + self.cfg.shift_factor as f64)?;
        }
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
            let n = self.ups.len();
            for (i, up) in self.ups.iter().enumerate() {
                x = up.forward(&x, i + 1 != n)?;
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

    #[test]
    fn small_full_path_spatial_matches_compression() {
        let device = Device::Cpu;
        let cfg = Flux2VaeConfig::small();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let vae = AutoencoderKlFlux2::load(cfg.clone(), vb).unwrap();
        assert!(!vae.tiny);
        let z = Tensor::zeros((1, cfg.latent_channels, 1, 2, 2), DType::F32, &device).unwrap();
        let out = vae.decode(&z).unwrap();
        assert_eq!(out.dims()[3], 8);
        assert_eq!(out.dims()[4], 8);
        assert_eq!(out.dims()[1], 3);
    }
}
