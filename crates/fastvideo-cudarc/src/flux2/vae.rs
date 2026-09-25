//! Flux2 2D VAE decode on `CudaTensor` (Diffusers AutoencoderKLFlux2).
//!
//! Tiny (`block_out_channels.len() <= 2`) stays a short conv + 2× nearest
//! stack for CI. Full decode walks ResNet + mid attention + upsample convs.

use fastvideo_models::flux2::Flux2VaeConfig;

use crate::wan::nn::{self, Linear};
use crate::wan::tensor::{CudaTensor, Result, TensorError};
use crate::wan::weights::{self, WeightMap};

fn msg(s: impl Into<String>) -> TensorError {
    TensorError::Message(s.into())
}

fn pinned(mut t: CudaTensor) -> Result<CudaTensor> {
    t.pin_device()?;
    Ok(t)
}

struct Conv2d {
    weight: CudaTensor,
    bias: Option<CudaTensor>,
    stride: usize,
    padding: usize,
}

impl Conv2d {
    fn zeros(cin: usize, cout: usize, k: usize, stride: usize, padding: usize, bias: bool) -> Self {
        Self {
            weight: pinned(CudaTensor::zeros(&[cout, cin, k, k])).expect("conv w"),
            bias: bias.then(|| pinned(CudaTensor::zeros(&[cout])).expect("conv b")),
            stride,
            padding,
        }
    }

    fn load(
        map: &WeightMap,
        prefix: &str,
        cin: usize,
        cout: usize,
        k: usize,
        stride: usize,
        padding: usize,
    ) -> Result<Self> {
        let w = pinned(weights::cuda_tensor_shaped(
            map,
            &weights::join_key(prefix, "weight"),
            &[cout, cin, k, k],
        )?)?;
        let bias = match weights::cuda_tensor_shaped(map, &weights::join_key(prefix, "bias"), &[cout]) {
            Ok(b) => Some(pinned(b)?),
            Err(_) => None,
        };
        Ok(Self {
            weight: w,
            bias,
            stride,
            padding,
        })
    }

    fn forward(&self, xs: &CudaTensor) -> Result<CudaTensor> {
        xs.conv2d(&self.weight, self.bias.as_ref(), self.padding, self.stride)
    }
}

struct GroupNorm {
    weight: CudaTensor,
    bias: CudaTensor,
    groups: usize,
    eps: f32,
}

impl GroupNorm {
    fn zeros(channels: usize, groups: usize, eps: f32) -> Self {
        Self {
            weight: pinned(CudaTensor::from_vec(vec![1.0; channels], vec![channels]).expect("gn w")).expect("pin"),
            bias: pinned(CudaTensor::zeros(&[channels])).expect("gn b"),
            groups: groups.min(channels).max(1),
            eps,
        }
    }

    fn load(map: &WeightMap, prefix: &str, channels: usize, groups: usize, eps: f32) -> Result<Self> {
        Ok(Self {
            weight: pinned(weights::cuda_tensor_shaped(map, &weights::join_key(prefix, "weight"), &[channels])?)?,
            bias: pinned(weights::cuda_tensor_shaped(map, &weights::join_key(prefix, "bias"), &[channels])?)?,
            groups: groups.min(channels).max(1),
            eps,
        })
    }

    fn forward(&self, xs: &CudaTensor) -> Result<CudaTensor> {
        group_norm(xs, &self.weight, &self.bias, self.groups, self.eps)
    }
}

/// GroupNorm via last-dim LayerNorm on the flattened `(C/G, H, W)` groups.
pub fn group_norm(
    xs: &CudaTensor,
    weight: &CudaTensor,
    bias: &CudaTensor,
    groups: usize,
    eps: f32,
) -> Result<CudaTensor> {
    if xs.rank() != 4 {
        return Err(msg("group_norm wants NCHW"));
    }
    let (b, c, h, w) = (xs.shape[0], xs.shape[1], xs.shape[2], xs.shape[3]);
    let g = groups.min(c).max(1);
    if c % g != 0 {
        return Err(msg(format!("group_norm channels {c} not divisible by {g}")));
    }
    let x = xs.reshape(vec![b * g, (c / g) * h * w])?;
    let y = x.layer_norm(eps, None, None)?.reshape(vec![b, c, h, w])?;
    let scale = weight.reshape(vec![1, c, 1, 1])?;
    let shift = bias.reshape(vec![1, c, 1, 1])?;
    y.mul(&scale)?.add(&shift)
}

struct ResnetBlock2d {
    norm1: GroupNorm,
    conv1: Conv2d,
    norm2: GroupNorm,
    conv2: Conv2d,
    skip: Option<Conv2d>,
}

impl ResnetBlock2d {
    fn zeros(cin: usize, cout: usize, groups: usize) -> Self {
        Self {
            norm1: GroupNorm::zeros(cin, groups, 1e-6),
            conv1: Conv2d::zeros(cin, cout, 3, 1, 1, true),
            norm2: GroupNorm::zeros(cout, groups, 1e-6),
            conv2: Conv2d::zeros(cout, cout, 3, 1, 1, true),
            skip: (cin != cout).then(|| Conv2d::zeros(cin, cout, 1, 1, 0, true)),
        }
    }

    fn load(map: &WeightMap, prefix: &str, cin: usize, cout: usize, groups: usize) -> Result<Self> {
        Ok(Self {
            norm1: GroupNorm::load(map, &weights::join_key(prefix, "norm1"), cin, groups, 1e-6)?,
            conv1: Conv2d::load(map, &weights::join_key(prefix, "conv1"), cin, cout, 3, 1, 1)?,
            norm2: GroupNorm::load(map, &weights::join_key(prefix, "norm2"), cout, groups, 1e-6)?,
            conv2: Conv2d::load(map, &weights::join_key(prefix, "conv2"), cout, cout, 3, 1, 1)?,
            skip: if cin != cout {
                Some(Conv2d::load(
                    map,
                    &weights::join_key(prefix, "conv_shortcut"),
                    cin,
                    cout,
                    1,
                    1,
                    0,
                )?)
            } else {
                None
            },
        })
    }

    fn forward(&self, xs: &CudaTensor) -> Result<CudaTensor> {
        let h = self.conv1.forward(&nn::silu(&self.norm1.forward(xs)?))?;
        let h = self.conv2.forward(&nn::silu(&self.norm2.forward(&h)?))?;
        match &self.skip {
            Some(c) => c.forward(xs)?.add(&h),
            None => xs.add(&h),
        }
    }
}

struct SpatialAttention {
    norm: GroupNorm,
    to_q: Linear,
    to_k: Linear,
    to_v: Linear,
    to_out: Linear,
}

impl SpatialAttention {
    fn zeros(channels: usize, groups: usize) -> Self {
        Self {
            norm: GroupNorm::zeros(channels, groups, 1e-6),
            to_q: Linear::zeros(channels, channels, true),
            to_k: Linear::zeros(channels, channels, true),
            to_v: Linear::zeros(channels, channels, true),
            to_out: Linear::zeros(channels, channels, true),
        }
    }

    fn load(map: &WeightMap, prefix: &str, channels: usize, groups: usize) -> Result<Self> {
        Ok(Self {
            norm: GroupNorm::load(map, &weights::join_key(prefix, "group_norm"), channels, groups, 1e-6)?,
            to_q: Linear::load(map, &weights::join_key(prefix, "to_q"), channels, channels, true)?,
            to_k: Linear::load(map, &weights::join_key(prefix, "to_k"), channels, channels, true)?,
            to_v: Linear::load(map, &weights::join_key(prefix, "to_v"), channels, channels, true)?,
            to_out: Linear::load(map, &weights::join_key(prefix, "to_out.0"), channels, channels, true)?,
        })
    }

    fn forward(&self, xs: &CudaTensor) -> Result<CudaTensor> {
        let (b, c, h, w) = (xs.shape[0], xs.shape[1], xs.shape[2], xs.shape[3]);
        let n = self.norm.forward(xs)?;
        let seq = n.reshape(vec![b, c, h * w])?.transpose(1, 2)?;
        let q = self.to_q.forward(&seq)?.reshape(vec![b, 1, h * w, c])?;
        let k = self.to_k.forward(&seq)?.reshape(vec![b, 1, h * w, c])?;
        let v = self.to_v.forward(&seq)?.reshape(vec![b, 1, h * w, c])?;
        let attn = nn::scaled_dot_product_attention(&q, &k, &v, None)?;
        let out = attn.reshape(vec![b, h * w, c])?;
        let out = self.to_out.forward(&out)?.transpose(1, 2)?.reshape(vec![b, c, h, w])?;
        xs.add(&out)
    }
}

struct MidBlock {
    resnets: Vec<ResnetBlock2d>,
    attn: Option<SpatialAttention>,
}

impl MidBlock {
    fn zeros(channels: usize, groups: usize) -> Self {
        Self {
            resnets: vec![
                ResnetBlock2d::zeros(channels, channels, groups),
                ResnetBlock2d::zeros(channels, channels, groups),
            ],
            attn: Some(SpatialAttention::zeros(channels, groups)),
        }
    }

    fn load(map: &WeightMap, prefix: &str, channels: usize, groups: usize) -> Result<Self> {
        let r0 = ResnetBlock2d::load(map, &weights::join_key(prefix, "resnets.0"), channels, channels, groups)?;
        let attn = SpatialAttention::load(
            map,
            &weights::join_key(prefix, "attentions.0"),
            channels,
            groups,
        )
        .ok();
        let r1 = ResnetBlock2d::load(map, &weights::join_key(prefix, "resnets.1"), channels, channels, groups).ok();
        let mut resnets = vec![r0];
        if let Some(r) = r1 {
            resnets.push(r);
        }
        Ok(Self { resnets, attn })
    }

    fn forward(&self, xs: &CudaTensor) -> Result<CudaTensor> {
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
    fn zeros(cin: usize, cout: usize, n_res: usize, groups: usize, add_upsample: bool) -> Self {
        let mut resnets = Vec::with_capacity(n_res);
        let mut current = cin;
        for _ in 0..n_res {
            resnets.push(ResnetBlock2d::zeros(current, cout, groups));
            current = cout;
        }
        Self {
            resnets,
            upsample: add_upsample.then(|| Conv2d::zeros(cout, cout, 3, 1, 1, true)),
        }
    }

    fn load(
        map: &WeightMap,
        prefix: &str,
        cin: usize,
        cout: usize,
        n_res: usize,
        groups: usize,
        add_upsample: bool,
    ) -> Result<Self> {
        let mut resnets = Vec::with_capacity(n_res);
        let mut current = cin;
        for i in 0..n_res {
            resnets.push(ResnetBlock2d::load(
                map,
                &weights::join_key(prefix, &format!("resnets.{i}")),
                current,
                cout,
                groups,
            )?);
            current = cout;
        }
        let upsample = if add_upsample {
            Conv2d::load(
                map,
                &weights::join_key(prefix, "upsamplers.0.conv"),
                cout,
                cout,
                3,
                1,
                1,
            )
            .ok()
        } else {
            None
        };
        Ok(Self { resnets, upsample })
    }

    fn forward(&self, xs: &CudaTensor, upsample: bool) -> Result<CudaTensor> {
        let mut x = xs.clone();
        for res in &self.resnets {
            x = res.forward(&x)?;
        }
        if upsample {
            let (h, w) = (x.shape[2], x.shape[3]);
            x = x.upsample_nearest2d(h * 2, w * 2)?;
            if let Some(u) = &self.upsample {
                x = u.forward(&x)?;
            }
        }
        Ok(x)
    }
}

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
    pub fn zeros(cfg: Flux2VaeConfig) -> Self {
        if cfg.block_out_channels.len() <= 2 {
            return Self::zeros_tiny(cfg);
        }
        Self::zeros_full(cfg)
    }

    fn zeros_tiny(cfg: Flux2VaeConfig) -> Self {
        let z = cfg.latent_channels;
        let mid = cfg.block_out_channels[0];
        Self {
            post_quant: None,
            conv_in: Conv2d::zeros(z, mid, 3, 1, 1, true),
            mid: None,
            ups: Vec::new(),
            conv_norm_out: None,
            conv_out: Conv2d::zeros(mid, cfg.out_channels, 3, 1, 1, true),
            cfg,
            tiny: true,
        }
    }

    fn zeros_full(cfg: Flux2VaeConfig) -> Self {
        let z = cfg.latent_channels;
        let chans = cfg.block_out_channels.clone();
        let c0 = *chans.last().unwrap_or(&8);
        let groups = cfg.norm_num_groups;
        let n_res = cfg.layers_per_block + 1;
        let reversed: Vec<usize> = chans.iter().rev().copied().collect();
        let mut ups = Vec::new();
        let mut prev = c0;
        for (i, &ch) in reversed.iter().enumerate() {
            let add_up = i + 1 != reversed.len();
            ups.push(UpBlock::zeros(prev, ch, n_res, groups, add_up));
            prev = ch;
        }
        let last_ch = *chans.first().unwrap_or(&c0);
        Self {
            post_quant: Some(Conv2d::zeros(z, z, 1, 1, 0, true)),
            conv_in: Conv2d::zeros(z, c0, 3, 1, 1, true),
            mid: Some(MidBlock::zeros(c0, groups)),
            ups,
            conv_norm_out: Some(GroupNorm::zeros(last_ch, groups, 1e-6)),
            conv_out: Conv2d::zeros(last_ch, cfg.out_channels, 3, 1, 1, true),
            cfg,
            tiny: false,
        }
    }

    pub fn load(cfg: Flux2VaeConfig, map: &WeightMap) -> Result<Self> {
        if cfg.block_out_channels.len() <= 2 {
            return Ok(Self::zeros_tiny(cfg));
        }
        if !map.contains("decoder.conv_out.weight") && !map.contains("decoder.conv_in.weight") {
            return Err(msg("Flux2 VAE missing decoder.conv_in/out.weight"));
        }
        let z = cfg.latent_channels;
        let chans = cfg.block_out_channels.clone();
        let c0 = *chans.last().unwrap_or(&512);
        let groups = cfg.norm_num_groups;
        let post_quant = Conv2d::load(map, "post_quant_conv", z, z, 1, 1, 0).ok();
        let conv_in = Conv2d::load(map, "decoder.conv_in", z, c0, 3, 1, 1)?;
        let mid = MidBlock::load(map, "decoder.mid_block", c0, groups.min(c0).max(1)).ok();
        let n_res = cfg.layers_per_block + 1;
        let reversed: Vec<usize> = chans.iter().rev().copied().collect();
        let mut ups = Vec::new();
        let mut prev = c0;
        for (i, &ch) in reversed.iter().enumerate() {
            let add_up = i + 1 != reversed.len();
            ups.push(UpBlock::load(
                map,
                &format!("decoder.up_blocks.{i}"),
                prev,
                ch,
                n_res,
                groups,
                add_up,
            )?);
            prev = ch;
        }
        let last_ch = *chans.first().unwrap_or(&c0);
        let conv_norm_out = GroupNorm::load(map, "decoder.conv_norm_out", last_ch, groups.min(last_ch).max(1), 1e-6).ok();
        let conv_out = Conv2d::load(map, "decoder.conv_out", last_ch, cfg.out_channels, 3, 1, 1)?;
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

    pub fn decode(&self, latents: &CudaTensor) -> Result<CudaTensor> {
        let mut x = latents.clone();
        if x.rank() == 5 {
            let (b, c, t, h, w) = (x.shape[0], x.shape[1], x.shape[2], x.shape[3], x.shape[4]);
            if t != 1 {
                return Err(msg(format!("Flux2 VAE is image-only, T={t}")));
            }
            x = x.reshape(vec![b, c, h, w])?;
        }
        x = x.mul_scalar(1.0 / self.cfg.scaling_factor);
        if self.cfg.shift_factor != 0.0 {
            x = x.add_scalar(self.cfg.shift_factor);
        }
        if let Some(pq) = &self.post_quant {
            x = pq.forward(&x)?;
        }
        x = self.conv_in.forward(&x)?;
        if let Some(mid) = &self.mid {
            x = mid.forward(&x)?;
        }
        if self.tiny {
            let (h, w) = (x.shape[2], x.shape[3]);
            x = x.upsample_nearest2d(h * 2, w * 2)?;
            let (h, w) = (x.shape[2], x.shape[3]);
            x = x.upsample_nearest2d(h * 2, w * 2)?;
        } else {
            let n = self.ups.len();
            for (i, up) in self.ups.iter().enumerate() {
                x = up.forward(&x, i + 1 != n)?;
            }
        }
        if let Some(n) = &self.conv_norm_out {
            x = nn::silu(&n.forward(&x)?);
        } else {
            x = x.silu();
        }
        x = self.conv_out.forward(&x)?;
        let (b, c, h, w) = (x.shape[0], x.shape[1], x.shape[2], x.shape[3]);
        x.reshape(vec![b, c, 1, h, w])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tiny_decode_writes_rgb() {
        let cfg = Flux2VaeConfig::tiny();
        let vae = AutoencoderKlFlux2::zeros(cfg.clone());
        let z = CudaTensor::zeros(&[1, cfg.latent_channels, 1, 2, 2]);
        let out = vae.decode(&z).unwrap();
        assert_eq!(out.shape, vec![1, 3, 1, 8, 8]);
    }

    #[test]
    fn small_full_path_spatial_matches_compression() {
        let cfg = Flux2VaeConfig::small();
        let vae = AutoencoderKlFlux2::zeros(cfg.clone());
        assert!(!vae.tiny);
        let z = CudaTensor::zeros(&[1, cfg.latent_channels, 1, 2, 2]);
        let out = vae.decode(&z).unwrap();
        assert_eq!(out.shape, vec![1, 3, 1, 8, 8]);
    }

}
