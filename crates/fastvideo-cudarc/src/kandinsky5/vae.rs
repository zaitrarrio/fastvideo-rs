//! Classic HunyuanVideo VAE (16-ch) used by Kandinsky 5.
//!
//! Diffusers `AutoencoderKLHunyuanVideo`: GroupNorm ResNets, nearest causal
//! upsample, mid-block causal attention. Spec: docs/ports/kandinsky5.md.

use crate::wan::nn;
use crate::wan::ops::PadMode;
use crate::wan::tensor::{CudaTensor, Result, TensorError};
use crate::wan::weights::{self, WeightMap};

fn msg(s: impl Into<String>) -> TensorError {
    TensorError::Message(s.into())
}

fn pinned(mut t: CudaTensor) -> Result<CudaTensor> {
    t.pin_device()?;
    Ok(t)
}

/// Latent → pixel frame counts for the causal VAE.
pub fn pixel_frames(latent_frames: usize, temporal_compression: usize) -> usize {
    if latent_frames == 0 {
        return 0;
    }
    1 + (latent_frames - 1) * temporal_compression
}

pub fn latent_frames(pixel_frames: usize, temporal_compression: usize) -> usize {
    if pixel_frames == 0 {
        return 0;
    }
    1 + (pixel_frames - 1).div_ceil(temporal_compression)
}

pub fn latent_hw(pixel: usize, spatial_compression: usize) -> usize {
    pixel.div_ceil(spatial_compression)
}

#[derive(Debug, Clone)]
pub struct HunyuanVideo16VaeConfig {
    pub out_channels: usize,
    pub latent_channels: usize,
    pub block_out_channels: [usize; 4],
    pub layers_per_block: usize,
    pub norm_num_groups: usize,
    pub spatial_compression_ratio: usize,
    pub temporal_compression_ratio: usize,
    pub scaling_factor: f32,
    pub mid_block_add_attention: bool,
}

impl HunyuanVideo16VaeConfig {
    pub fn default_hunyuan() -> Self {
        Self {
            out_channels: 3,
            latent_channels: 16,
            block_out_channels: [128, 256, 512, 512],
            layers_per_block: 2,
            norm_num_groups: 32,
            spatial_compression_ratio: 8,
            temporal_compression_ratio: 4,
            scaling_factor: 0.476986,
            mid_block_add_attention: true,
        }
    }
}

/// Causal 3-D conv: replicate-pad then unpadded `conv3d`.
struct CausalConv3d {
    weight: CudaTensor, // [out, in, kt, kh, kw]
    bias: CudaTensor,
    pad_t: usize,
    pad_h: usize,
    pad_w: usize,
}

impl CausalConv3d {
    fn zeros(in_c: usize, out_c: usize, k: usize) -> Result<Self> {
        Ok(Self {
            weight: pinned(CudaTensor::zeros(&[out_c, in_c, k, k, k]))?,
            bias: pinned(CudaTensor::zeros(&[out_c]))?,
            pad_t: k.saturating_sub(1),
            pad_h: k / 2,
            pad_w: k / 2,
        })
    }

    fn load(map: &WeightMap, prefix: &str, in_c: usize, out_c: usize, k: usize) -> Result<Self> {
        Ok(Self {
            weight: pinned(weights::cuda_tensor_shaped(
                map,
                &weights::join_key(prefix, "conv.weight"),
                &[out_c, in_c, k, k, k],
            )?)?,
            bias: pinned(weights::cuda_tensor_shaped(
                map,
                &weights::join_key(prefix, "conv.bias"),
                &[out_c],
            )?)?,
            pad_t: k.saturating_sub(1),
            pad_h: k / 2,
            pad_w: k / 2,
        })
    }

    fn forward(&self, x: &CudaTensor) -> Result<CudaTensor> {
        let mut h = x.clone();
        if self.pad_w > 0 {
            h = h.pad(4, self.pad_w, self.pad_w, PadMode::Replicate)?;
        }
        if self.pad_h > 0 {
            h = h.pad(3, self.pad_h, self.pad_h, PadMode::Replicate)?;
        }
        if self.pad_t > 0 {
            h = h.pad(2, self.pad_t, 0, PadMode::Replicate)?;
        }
        h.conv3d(&self.weight, Some(&self.bias), [0, 0, 0], [1, 1, 1])
    }
}

struct Conv1x1 {
    weight: CudaTensor,
    bias: CudaTensor,
}

impl Conv1x1 {
    fn zeros(in_c: usize, out_c: usize) -> Result<Self> {
        Ok(Self {
            weight: pinned(CudaTensor::zeros(&[out_c, in_c, 1, 1, 1]))?,
            bias: pinned(CudaTensor::zeros(&[out_c]))?,
        })
    }

    fn load(map: &WeightMap, prefix: &str, in_c: usize, out_c: usize) -> Result<Self> {
        Ok(Self {
            weight: pinned(weights::cuda_tensor_shaped(
                map,
                &weights::join_key(prefix, "weight"),
                &[out_c, in_c, 1, 1, 1],
            )?)?,
            bias: pinned(weights::cuda_tensor_shaped(
                map,
                &weights::join_key(prefix, "bias"),
                &[out_c],
            )?)?,
        })
    }

    fn forward(&self, x: &CudaTensor) -> Result<CudaTensor> {
        x.conv3d(&self.weight, Some(&self.bias), [0, 0, 0], [1, 1, 1])
    }
}

struct GroupNormAffine {
    weight: CudaTensor,
    bias: CudaTensor,
    groups: usize,
    eps: f32,
}

impl GroupNormAffine {
    fn zeros(c: usize, groups: usize) -> Result<Self> {
        Ok(Self {
            weight: pinned(CudaTensor::ones(&[c]))?,
            bias: pinned(CudaTensor::zeros(&[c]))?,
            groups,
            eps: 1e-6,
        })
    }

    fn load(map: &WeightMap, prefix: &str, c: usize, groups: usize) -> Result<Self> {
        Ok(Self {
            weight: pinned(weights::cuda_tensor_shaped(
                map,
                &weights::join_key(prefix, "weight"),
                &[c],
            )?)?,
            bias: pinned(weights::cuda_tensor_shaped(
                map,
                &weights::join_key(prefix, "bias"),
                &[c],
            )?)?,
            groups,
            eps: 1e-6,
        })
    }

    fn forward(&self, x: &CudaTensor, silu: bool) -> Result<CudaTensor> {
        x.group_norm(self.groups, &self.weight, &self.bias, self.eps, silu)
    }
}

struct ResnetBlock {
    norm1: GroupNormAffine,
    conv1: CausalConv3d,
    norm2: GroupNormAffine,
    conv2: CausalConv3d,
    shortcut: Option<CausalConv3d>,
}

impl ResnetBlock {
    fn zeros(in_c: usize, out_c: usize, groups: usize) -> Result<Self> {
        Ok(Self {
            norm1: GroupNormAffine::zeros(in_c, groups.min(in_c).max(1))?,
            conv1: CausalConv3d::zeros(in_c, out_c, 3)?,
            norm2: GroupNormAffine::zeros(out_c, groups.min(out_c).max(1))?,
            conv2: CausalConv3d::zeros(out_c, out_c, 3)?,
            shortcut: if in_c != out_c {
                Some(CausalConv3d::zeros(in_c, out_c, 1)?)
            } else {
                None
            },
        })
    }

    fn load(map: &WeightMap, prefix: &str, in_c: usize, out_c: usize, groups: usize) -> Result<Self> {
        let key = |n: &str| weights::join_key(prefix, n);
        Ok(Self {
            norm1: GroupNormAffine::load(map, &key("norm1"), in_c, groups.min(in_c).max(1))?,
            conv1: CausalConv3d::load(map, &key("conv1"), in_c, out_c, 3)?,
            norm2: GroupNormAffine::load(map, &key("norm2"), out_c, groups.min(out_c).max(1))?,
            conv2: CausalConv3d::load(map, &key("conv2"), out_c, out_c, 3)?,
            shortcut: if in_c != out_c {
                Some(CausalConv3d::load(map, &key("conv_shortcut"), in_c, out_c, 1)?)
            } else {
                None
            },
        })
    }

    fn forward(&self, x: &CudaTensor) -> Result<CudaTensor> {
        let mut h = self.norm1.forward(x, true)?;
        h = self.conv1.forward(&h)?;
        h = self.norm2.forward(&h, true)?;
        h = self.conv2.forward(&h)?;
        let residual = match &self.shortcut {
            Some(s) => s.forward(x)?,
            None => x.clone(),
        };
        h.add(&residual)
    }
}

/// Diffusers mid-block Attention (`_from_deprecated_attn_block`).
struct AttnBlock {
    group_norm: GroupNormAffine,
    to_q: nn::Linear,
    to_k: nn::Linear,
    to_v: nn::Linear,
    to_out: nn::Linear,
    channels: usize,
}

impl AttnBlock {
    fn zeros(c: usize, groups: usize) -> Result<Self> {
        let w = |rows: usize| -> Result<nn::Linear> {
            nn::Linear::from_tensors(CudaTensor::zeros(&[rows, c]), Some(CudaTensor::zeros(&[rows])))
        };
        Ok(Self {
            group_norm: GroupNormAffine::zeros(c, groups.min(c).max(1))?,
            to_q: w(c)?,
            to_k: w(c)?,
            to_v: w(c)?,
            to_out: w(c)?,
            channels: c,
        })
    }

    fn load(map: &WeightMap, prefix: &str, c: usize, groups: usize) -> Result<Self> {
        let key = |n: &str| weights::join_key(prefix, n);
        Ok(Self {
            group_norm: GroupNormAffine::load(map, &key("group_norm"), c, groups.min(c).max(1))?,
            to_q: nn::Linear::load(map, &key("to_q"), c, c, true)?,
            to_k: nn::Linear::load(map, &key("to_k"), c, c, true)?,
            to_v: nn::Linear::load(map, &key("to_v"), c, c, true)?,
            to_out: nn::Linear::load(map, &key("to_out.0"), c, c, true)?,
            channels: c,
        })
    }

    fn forward(&self, x: &CudaTensor) -> Result<CudaTensor> {
        let [b, c, t, h, w] = match x.shape[..] {
            [b, c, t, h, w] => [b, c, t, h, w],
            _ => return Err(msg("attn: want 5D")),
        };
        if c != self.channels {
            return Err(msg(format!("attn channels {c} vs {}", self.channels)));
        }
        let n_hw = h * w;
        let seq = t * n_hw;
        // [B,C,T,H,W] → [B, S, C]
        let flat = x
            .permute(&[0, 2, 3, 4, 1])?
            .reshape(vec![b, seq, c])?;
        let residual = flat.clone();
        // group_norm over channels: [B,C,S]
        let n = flat.permute(&[0, 2, 1])?;
        let n = self.group_norm.forward(&n, false)?;
        let n = n.permute(&[0, 2, 1])?;
        let q = self.to_q.forward(&n)?;
        let k = self.to_k.forward(&n)?;
        let v = self.to_v.forward(&n)?;
        // single head [B,1,S,C]
        let to_bhsd = |t: CudaTensor| -> Result<CudaTensor> { t.unsqueeze(1) };
        let mask = causal_frame_mask(t, n_hw)?;
        let attn = nn::scaled_dot_product_attention_masked(
            &to_bhsd(q)?,
            &to_bhsd(k)?,
            &to_bhsd(v)?,
            None,
            Some(&mask),
        )?;
        let out = attn.squeeze(1)?;
        let out = self.to_out.forward(&out)?.add(&residual)?;
        out.reshape(vec![b, t, h, w, c])?.permute(&[0, 4, 1, 2, 3])
    }
}

fn causal_frame_mask(n_frame: usize, n_hw: usize) -> Result<CudaTensor> {
    let seq = n_frame * n_hw;
    let mut m = vec![f32::NEG_INFINITY; seq * seq];
    for i in 0..seq {
        let i_frame = i / n_hw;
        let end = (i_frame + 1) * n_hw;
        for j in 0..end {
            m[i * seq + j] = 0.0;
        }
    }
    Ok(CudaTensor::from_vec(m, vec![1, 1, seq, seq])?)
}

struct MidBlock {
    resnets: Vec<ResnetBlock>,
    attns: Vec<Option<AttnBlock>>,
}

impl MidBlock {
    fn zeros(c: usize, groups: usize, add_attn: bool) -> Result<Self> {
        Ok(Self {
            resnets: vec![
                ResnetBlock::zeros(c, c, groups)?,
                ResnetBlock::zeros(c, c, groups)?,
            ],
            attns: vec![if add_attn {
                Some(AttnBlock::zeros(c, groups)?)
            } else {
                None
            }],
        })
    }

    fn load(map: &WeightMap, prefix: &str, c: usize, groups: usize, add_attn: bool) -> Result<Self> {
        Ok(Self {
            resnets: vec![
                ResnetBlock::load(map, &format!("{prefix}.resnets.0"), c, c, groups)?,
                ResnetBlock::load(map, &format!("{prefix}.resnets.1"), c, c, groups)?,
            ],
            attns: vec![if add_attn {
                Some(AttnBlock::load(
                    map,
                    &format!("{prefix}.attentions.0"),
                    c,
                    groups,
                )?)
            } else {
                None
            }],
        })
    }

    fn forward(&self, x: &CudaTensor) -> Result<CudaTensor> {
        let mut h = self.resnets[0].forward(x)?;
        for (attn, res) in self.attns.iter().zip(self.resnets.iter().skip(1)) {
            if let Some(a) = attn {
                h = a.forward(&h)?;
            }
            h = res.forward(&h)?;
        }
        Ok(h)
    }
}

/// Nearest upsample on `[B,C,T,H,W]` with integer factors `(ft,fh,fw)`.
fn upsample_nearest5d(x: &CudaTensor, ft: usize, fh: usize, fw: usize) -> Result<CudaTensor> {
    let [b, c, t, h, w] = match x.shape[..] {
        [b, c, t, h, w] => [b, c, t, h, w],
        _ => return Err(msg("upsample_nearest5d: want 5D")),
    };
    if ft == 1 && fh == 1 && fw == 1 {
        return Ok(x.clone());
    }
    let out_t = t * ft;
    let out_h = h * fh;
    let out_w = w * fw;
    let host = x.host_cow()?;
    let mut out = vec![0f32; b * c * out_t * out_h * out_w];
    for bi in 0..b {
        for ci in 0..c {
            for ti in 0..out_t {
                let st = ti / ft;
                for yi in 0..out_h {
                    let sy = yi / fh;
                    for xi in 0..out_w {
                        let sx = xi / fw;
                        let src = (((bi * c + ci) * t + st) * h + sy) * w + sx;
                        let dst = (((bi * c + ci) * out_t + ti) * out_h + yi) * out_w + xi;
                        out[dst] = host[src];
                    }
                }
            }
        }
    }
    Ok(CudaTensor::from_vec(out, vec![b, c, out_t, out_h, out_w])?.to_device()?)
}

struct Upsample {
    conv: CausalConv3d,
    scale_t: usize,
    scale_h: usize,
    scale_w: usize,
}

impl Upsample {
    fn zeros(c: usize, scale: (usize, usize, usize)) -> Result<Self> {
        Ok(Self {
            conv: CausalConv3d::zeros(c, c, 3)?,
            scale_t: scale.0,
            scale_h: scale.1,
            scale_w: scale.2,
        })
    }

    fn load(map: &WeightMap, prefix: &str, c: usize, scale: (usize, usize, usize)) -> Result<Self> {
        Ok(Self {
            conv: CausalConv3d::load(map, &weights::join_key(prefix, "conv"), c, c, 3)?,
            scale_t: scale.0,
            scale_h: scale.1,
            scale_w: scale.2,
        })
    }

    fn forward(&self, x: &CudaTensor) -> Result<CudaTensor> {
        let [b, c, t, h, w] = match x.shape[..] {
            [b, c, t, h, w] => [b, c, t, h, w],
            _ => return Err(msg("upsample: want 5D")),
        };
        let _ = (b, c, h, w);
        // First frame: spatial-only upsample; remaining frames: full (T,H,W).
        let first = x.narrow(2, 0, 1)?;
        let first = upsample_nearest5d(&first, 1, self.scale_h, self.scale_w)?;
        let h = if t > 1 {
            let rest = x.narrow(2, 1, t - 1)?;
            let rest = upsample_nearest5d(&rest, self.scale_t, self.scale_h, self.scale_w)?;
            CudaTensor::cat(&[&first, &rest], 2)?
        } else {
            first
        };
        self.conv.forward(&h)
    }
}

struct UpBlock {
    resnets: Vec<ResnetBlock>,
    upsample: Option<Upsample>,
}

impl UpBlock {
    fn zeros(
        in_c: usize,
        out_c: usize,
        num_layers: usize,
        groups: usize,
        scale: Option<(usize, usize, usize)>,
    ) -> Result<Self> {
        let mut resnets = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            let ic = if i == 0 { in_c } else { out_c };
            resnets.push(ResnetBlock::zeros(ic, out_c, groups)?);
        }
        let upsample = match scale {
            Some(s) if s != (1, 1, 1) => Some(Upsample::zeros(out_c, s)?),
            _ => None,
        };
        Ok(Self { resnets, upsample })
    }

    fn load(
        map: &WeightMap,
        prefix: &str,
        in_c: usize,
        out_c: usize,
        num_layers: usize,
        groups: usize,
        scale: Option<(usize, usize, usize)>,
    ) -> Result<Self> {
        let mut resnets = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            let ic = if i == 0 { in_c } else { out_c };
            resnets.push(ResnetBlock::load(
                map,
                &format!("{prefix}.resnets.{i}"),
                ic,
                out_c,
                groups,
            )?);
        }
        let upsample = match scale {
            Some(s) if s != (1, 1, 1) => Some(Upsample::load(
                map,
                &format!("{prefix}.upsamplers.0"),
                out_c,
                s,
            )?),
            _ => None,
        };
        Ok(Self { resnets, upsample })
    }

    fn forward(&self, x: &CudaTensor) -> Result<CudaTensor> {
        let mut h = x.clone();
        for r in &self.resnets {
            h = r.forward(&h)?;
        }
        if let Some(up) = &self.upsample {
            h = up.forward(&h)?;
        }
        Ok(h)
    }
}

/// Decoder half of `AutoencoderKLHunyuanVideo` (classic 16-ch).
pub struct HunyuanVideo16Vae {
    pub cfg: HunyuanVideo16VaeConfig,
    post_quant: Conv1x1,
    conv_in: CausalConv3d,
    mid: MidBlock,
    up_blocks: Vec<UpBlock>,
    norm_out: GroupNormAffine,
    conv_out: CausalConv3d,
    ready: bool,
}

impl HunyuanVideo16Vae {
    pub fn from_config(cfg: HunyuanVideo16VaeConfig) -> Self {
        Self {
            ready: false,
            post_quant: Conv1x1::zeros(1, 1).expect("placeholder"),
            conv_in: CausalConv3d::zeros(1, 1, 1).expect("placeholder"),
            mid: MidBlock::zeros(1, 1, false).expect("placeholder"),
            up_blocks: Vec::new(),
            norm_out: GroupNormAffine::zeros(1, 1).expect("placeholder"),
            conv_out: CausalConv3d::zeros(1, 1, 1).expect("placeholder"),
            cfg,
        }
    }

    /// Tiny decoder for host unit tests (full 8×/4× compression, small channels).
    pub fn zeros_tiny() -> Result<Self> {
        let cfg = HunyuanVideo16VaeConfig {
            out_channels: 3,
            latent_channels: 4,
            block_out_channels: [8, 8, 8, 8],
            layers_per_block: 1,
            norm_num_groups: 4,
            spatial_compression_ratio: 8,
            temporal_compression_ratio: 4,
            scaling_factor: 1.0,
            mid_block_add_attention: true,
        };
        Self::build(&cfg, None)
    }

    pub fn load(map: &WeightMap, cfg: HunyuanVideo16VaeConfig) -> Result<Self> {
        Self::build(&cfg, Some(map))
    }

    pub fn scaling_factor(&self) -> f32 {
        self.cfg.scaling_factor
    }

    fn build(cfg: &HunyuanVideo16VaeConfig, map: Option<&WeightMap>) -> Result<Self> {
        let dec: Vec<usize> = cfg.block_out_channels.iter().copied().rev().collect();
        let latent = cfg.latent_channels;
        let layers = cfg.layers_per_block + 1;
        let groups = cfg.norm_num_groups;
        let log_s = (cfg.spatial_compression_ratio as f64).log2() as usize;
        let log_t = (cfg.temporal_compression_ratio as f64).log2() as usize;
        let n_blocks = dec.len();

        let post_quant = match map {
            Some(m) => Conv1x1::load(m, "post_quant_conv", latent, latent)?,
            None => Conv1x1::zeros(latent, latent)?,
        };
        let conv_in = match map {
            Some(m) => CausalConv3d::load(m, "decoder.conv_in", latent, dec[0], 3)?,
            None => CausalConv3d::zeros(latent, dec[0], 3)?,
        };
        let mid = match map {
            Some(m) => MidBlock::load(
                m,
                "decoder.mid_block",
                dec[0],
                groups,
                cfg.mid_block_add_attention,
            )?,
            None => MidBlock::zeros(dec[0], groups, cfg.mid_block_add_attention)?,
        };

        let mut up_blocks = Vec::new();
        let mut prev = dec[0];
        for i in 0..n_blocks {
            let out_c = dec[i];
            let is_final = i + 1 == n_blocks;
            let add_spatial = i < log_s;
            let add_time = i >= n_blocks.saturating_sub(1 + log_t) && !is_final;
            let scale = if add_spatial || add_time {
                Some((
                    if add_time { 2 } else { 1 },
                    if add_spatial { 2 } else { 1 },
                    if add_spatial { 2 } else { 1 },
                ))
            } else {
                None
            };
            let block = match map {
                Some(m) => UpBlock::load(
                    m,
                    &format!("decoder.up_blocks.{i}"),
                    prev,
                    out_c,
                    layers,
                    groups,
                    scale,
                )?,
                None => UpBlock::zeros(prev, out_c, layers, groups, scale)?,
            };
            prev = out_c;
            up_blocks.push(block);
        }

        let last = *dec.last().unwrap();
        let (norm_out, conv_out) = match map {
            Some(m) => (
                GroupNormAffine::load(m, "decoder.conv_norm_out", last, groups.min(last).max(1))?,
                CausalConv3d::load(m, "decoder.conv_out", last, cfg.out_channels, 3)?,
            ),
            None => (
                GroupNormAffine::zeros(last, groups.min(last).max(1))?,
                CausalConv3d::zeros(last, cfg.out_channels, 3)?,
            ),
        };

        Ok(Self {
            cfg: cfg.clone(),
            post_quant,
            conv_in,
            mid,
            up_blocks,
            norm_out,
            conv_out,
            ready: true,
        })
    }

    /// `latents` `[B, C, T, H, W]` → RGB `[B, 3, Tf, Hf, Wf]`.
    pub fn decode(&self, latents: &CudaTensor) -> Result<CudaTensor> {
        if !self.ready || self.up_blocks.is_empty() {
            return Err(msg(
                "HunyuanVideo-16 VAE: call load() or zeros_tiny() before decode",
            ));
        }
        let [b, c, t, h, w] = match latents.shape[..] {
            [b, c, t, h, w] => [b, c, t, h, w],
            _ => {
                return Err(msg(format!(
                    "hunyuan16 vae: latents {:?} want [B,C,T,H,W]",
                    latents.shape
                )))
            }
        };
        if c != self.cfg.latent_channels {
            return Err(msg(format!(
                "hunyuan16 vae: channels {c} vs {}",
                self.cfg.latent_channels
            )));
        }
        let _ = (b, t, h, w);
        let mut hs = self.post_quant.forward(latents)?;
        hs = self.conv_in.forward(&hs)?;
        hs = self.mid.forward(&hs)?;
        for block in &self.up_blocks {
            hs = block.forward(&hs)?;
        }
        hs = self.norm_out.forward(&hs, true)?;
        self.conv_out.forward(&hs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults() {
        let c = HunyuanVideo16VaeConfig::default_hunyuan();
        assert_eq!(c.latent_channels, 16);
        assert_eq!(c.spatial_compression_ratio, 8);
        assert_eq!(c.temporal_compression_ratio, 4);
    }

    #[test]
    fn frame_math() {
        assert_eq!(latent_frames(121, 4), 31);
        assert_eq!(pixel_frames(31, 4), 121);
        assert_eq!(latent_hw(512, 8), 64);
    }

    #[test]
    fn tiny_decode_shapes() {
        let vae = HunyuanVideo16Vae::zeros_tiny().unwrap();
        // T=2,H=1,W=1 → temporal×4 spatial×8 → T=5, H=8, W=8
        let lat = CudaTensor::zeros(&[1, 4, 2, 1, 1]);
        let out = vae.decode(&lat).unwrap();
        assert_eq!(
            out.shape,
            vec![1, 3, pixel_frames(2, 4), 8, 8]
        );
    }
}
