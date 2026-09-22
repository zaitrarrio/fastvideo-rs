//! HunyuanVideo 1.5 causal video VAE decode (`AutoencoderKLHunyuanVideo15`).
//!
//! Diffusers layout: DCAE-style temporal/spatial upsample, channel-first
//! L2×√C RMS, mid-block causal attention. Spec: docs/ports/hunyuan15.md.

use fastvideo_models::hunyuan15::Hunyuan15VaeConfig;

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

/// Pointwise 1×1×1 conv (attn / shortcut).
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

/// Diffusers `HunyuanVideo15RMS_norm`: L2-normalize over channels × √C × γ.
fn rms_l2_channels(x: &CudaTensor, gamma: &CudaTensor) -> Result<CudaTensor> {
    let [n, c, t, h, w] = match x.shape[..] {
        [n, c, t, h, w] => [n, c, t, h, w],
        _ => return Err(msg(format!("hy15 rms expects [B,C,T,H,W], got {:?}", x.shape))),
    };
    if gamma.numel() != c {
        return Err(msg(format!("hy15 rms gamma {:?} for C={c}", gamma.shape)));
    }
    let spatial = t * h * w;
    let host = x.host_cow()?;
    let g = gamma.host_cow()?;
    let scale = (c as f32).sqrt();
    let mut out = vec![0f32; host.len()];
    for bi in 0..n {
        for s in 0..spatial {
            let mut sq = 0f32;
            for ci in 0..c {
                let v = host[(bi * c + ci) * spatial + s];
                sq += v * v;
            }
            let inv = 1.0 / sq.max(1e-12).sqrt();
            for ci in 0..c {
                let idx = (bi * c + ci) * spatial + s;
                out[idx] = host[idx] * inv * scale * g[ci];
            }
        }
    }
    Ok(CudaTensor::from_vec(out, x.shape.clone())?.to_device()?)
}

fn load_gamma(map: &WeightMap, key: &str, c: usize) -> Result<CudaTensor> {
    // Diffusers stores `[C,1,1,1]` or `[C]`.
    if let Ok(t) = weights::cuda_tensor_shaped(map, key, &[c, 1, 1, 1]) {
        return pinned(t.reshape(vec![c])?);
    }
    pinned(weights::cuda_tensor_shaped(map, key, &[c])?)
}

fn repeat_interleave_channels(x: &CudaTensor, repeats: usize) -> Result<CudaTensor> {
    if repeats <= 1 {
        return Ok(x.clone());
    }
    let [b, c, t, h, w] = match x.shape[..] {
        [b, c, t, h, w] => [b, c, t, h, w],
        _ => return Err(msg("repeat_interleave_channels: want 5D")),
    };
    let host = x.host_cow()?;
    let spatial = t * h * w;
    let mut out = vec![0f32; b * c * repeats * spatial];
    for bi in 0..b {
        for ci in 0..c {
            let src = &host[(bi * c + ci) * spatial..][..spatial];
            for r in 0..repeats {
                let dst = ((bi * c * repeats + ci * repeats + r) * spatial)..;
                out[dst][..spatial].copy_from_slice(src);
            }
        }
    }
    Ok(CudaTensor::from_vec(out, vec![b, c * repeats, t, h, w])?.to_device()?)
}

/// `(B, r1*r2*r3*C, F, H, W)` → `(B, C, r1*F, r2*H, r3*W)`.
fn dcae_upsample_rearrange(x: &CudaTensor, r1: usize, r2: usize, r3: usize) -> Result<CudaTensor> {
    let [b, packed, f, h, w] = match x.shape[..] {
        [b, packed, f, h, w] => [b, packed, f, h, w],
        _ => return Err(msg("dcae upsample: want 5D")),
    };
    let factor = r1 * r2 * r3;
    if packed % factor != 0 {
        return Err(msg(format!("dcae upsample: C={packed} not divisible by {factor}")));
    }
    let c = packed / factor;
    // view (b,r1,r2,r3,c,f,h,w) → permute (0,4,5,1,6,2,7,3) → (b,c,f*r1,h*r2,w*r3)
    let v = x.reshape(vec![b, r1, r2, r3, c, f, h, w])?;
    let p = v.permute(&[0, 4, 5, 1, 6, 2, 7, 3])?;
    p.reshape(vec![b, c, f * r1, h * r2, w * r3])
}

struct ResnetBlock {
    norm1: CudaTensor,
    conv1: CausalConv3d,
    norm2: CudaTensor,
    conv2: CausalConv3d,
    shortcut: Option<Conv1x1>,
}

impl ResnetBlock {
    fn zeros(in_c: usize, out_c: usize) -> Result<Self> {
        Ok(Self {
            norm1: pinned(CudaTensor::ones(&[in_c]))?,
            conv1: CausalConv3d::zeros(in_c, out_c, 3)?,
            norm2: pinned(CudaTensor::ones(&[out_c]))?,
            conv2: CausalConv3d::zeros(out_c, out_c, 3)?,
            shortcut: if in_c != out_c {
                Some(Conv1x1::zeros(in_c, out_c)?)
            } else {
                None
            },
        })
    }

    fn load(map: &WeightMap, prefix: &str, in_c: usize, out_c: usize) -> Result<Self> {
        let key = |n: &str| weights::join_key(prefix, n);
        Ok(Self {
            norm1: load_gamma(map, &key("norm1.gamma"), in_c)?,
            conv1: CausalConv3d::load(map, &key("conv1"), in_c, out_c, 3)?,
            norm2: load_gamma(map, &key("norm2.gamma"), out_c)?,
            conv2: CausalConv3d::load(map, &key("conv2"), out_c, out_c, 3)?,
            shortcut: if in_c != out_c {
                Some(Conv1x1::load(map, &key("conv_shortcut"), in_c, out_c)?)
            } else {
                None
            },
        })
    }

    fn forward(&self, x: &CudaTensor) -> Result<CudaTensor> {
        let mut h = rms_l2_channels(x, &self.norm1)?.silu();
        h = self.conv1.forward(&h)?;
        h = rms_l2_channels(&h, &self.norm2)?.silu();
        h = self.conv2.forward(&h)?;
        let residual = match &self.shortcut {
            Some(s) => s.forward(x)?,
            None => x.clone(),
        };
        h.add(&residual)
    }
}

struct AttnBlock {
    norm: CudaTensor,
    to_q: Conv1x1,
    to_k: Conv1x1,
    to_v: Conv1x1,
    proj_out: Conv1x1,
}

impl AttnBlock {
    fn zeros(c: usize) -> Result<Self> {
        Ok(Self {
            norm: pinned(CudaTensor::ones(&[c]))?,
            to_q: Conv1x1::zeros(c, c)?,
            to_k: Conv1x1::zeros(c, c)?,
            to_v: Conv1x1::zeros(c, c)?,
            proj_out: Conv1x1::zeros(c, c)?,
        })
    }

    fn load(map: &WeightMap, prefix: &str, c: usize) -> Result<Self> {
        let key = |n: &str| weights::join_key(prefix, n);
        Ok(Self {
            norm: load_gamma(map, &key("norm.gamma"), c)?,
            to_q: Conv1x1::load(map, &key("to_q"), c, c)?,
            to_k: Conv1x1::load(map, &key("to_k"), c, c)?,
            to_v: Conv1x1::load(map, &key("to_v"), c, c)?,
            proj_out: Conv1x1::load(map, &key("proj_out"), c, c)?,
        })
    }

    fn forward(&self, x: &CudaTensor) -> Result<CudaTensor> {
        let [b, c, t, h, w] = match x.shape[..] {
            [b, c, t, h, w] => [b, c, t, h, w],
            _ => return Err(msg("attn: want 5D")),
        };
        let n_hw = h * w;
        let seq = t * n_hw;
        let n = rms_l2_channels(x, &self.norm)?;
        let q = self.to_q.forward(&n)?;
        let k = self.to_k.forward(&n)?;
        let v = self.to_v.forward(&n)?;
        // [B,C,THW] → [B,1,THW,C]
        let to_bhsd = |t: CudaTensor| -> Result<CudaTensor> {
            t.reshape(vec![b, c, seq])?
                .permute(&[0, 2, 1])?
                .unsqueeze(1)
        };
        let q = to_bhsd(q)?;
        let k = to_bhsd(k)?;
        let v = to_bhsd(v)?;
        let mask = causal_frame_mask(t, n_hw)?;
        let attn = nn::scaled_dot_product_attention_masked(&q, &k, &v, None, Some(&mask))?;
        let out = attn
            .squeeze(1)?
            .reshape(vec![b, t, h, w, c])?
            .permute(&[0, 4, 1, 2, 3])?;
        self.proj_out.forward(&out)?.add(x)
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
    // BHSD mask broadcast: [1, 1, S, S]
    Ok(CudaTensor::from_vec(m, vec![1, 1, seq, seq])?)
}

struct MidBlock {
    resnets: Vec<ResnetBlock>,
    attns: Vec<AttnBlock>,
}

impl MidBlock {
    fn zeros(c: usize) -> Result<Self> {
        Ok(Self {
            resnets: vec![ResnetBlock::zeros(c, c)?, ResnetBlock::zeros(c, c)?],
            attns: vec![AttnBlock::zeros(c)?],
        })
    }

    fn load(map: &WeightMap, prefix: &str, c: usize) -> Result<Self> {
        Ok(Self {
            resnets: vec![
                ResnetBlock::load(map, &format!("{prefix}.resnets.0"), c, c)?,
                ResnetBlock::load(map, &format!("{prefix}.resnets.1"), c, c)?,
            ],
            attns: vec![AttnBlock::load(map, &format!("{prefix}.attentions.0"), c)?],
        })
    }

    fn forward(&self, x: &CudaTensor) -> Result<CudaTensor> {
        let mut h = self.resnets[0].forward(x)?;
        for (attn, res) in self.attns.iter().zip(self.resnets.iter().skip(1)) {
            h = attn.forward(&h)?;
            h = res.forward(&h)?;
        }
        Ok(h)
    }
}

struct Upsample {
    conv: CausalConv3d,
    add_temporal: bool,
    repeats: usize,
}

impl Upsample {
    fn zeros(in_c: usize, out_c: usize, add_temporal: bool) -> Result<Self> {
        let factor = if add_temporal { 8 } else { 4 };
        Ok(Self {
            conv: CausalConv3d::zeros(in_c, out_c * factor, 3)?,
            add_temporal,
            repeats: factor * out_c / in_c,
        })
    }

    fn load(map: &WeightMap, prefix: &str, in_c: usize, out_c: usize, add_temporal: bool) -> Result<Self> {
        let factor = if add_temporal { 8 } else { 4 };
        Ok(Self {
            conv: CausalConv3d::load(map, prefix, in_c, out_c * factor, 3)?,
            add_temporal,
            repeats: factor * out_c / in_c,
        })
    }

    fn forward(&self, x: &CudaTensor) -> Result<CudaTensor> {
        let r1 = if self.add_temporal { 2 } else { 1 };
        let h = self.conv.forward(x)?;
        if self.add_temporal {
            let h_first = h.narrow(2, 0, 1)?;
            let mut h_first = dcae_upsample_rearrange(&h_first, 1, 2, 2)?;
            let half = h_first.shape[1] / 2;
            h_first = h_first.narrow(1, 0, half)?;
            let h_next = h.narrow(2, 1, h.shape[2] - 1)?;
            let h_next = dcae_upsample_rearrange(&h_next, r1, 2, 2)?;
            let h = CudaTensor::cat(&[&h_first, &h_next], 2)?;

            let x_first = x.narrow(2, 0, 1)?;
            let x_first = dcae_upsample_rearrange(&x_first, 1, 2, 2)?;
            let x_first = repeat_interleave_channels(&x_first, self.repeats / 2)?;
            let x_next = x.narrow(2, 1, x.shape[2] - 1)?;
            let x_next = dcae_upsample_rearrange(&x_next, r1, 2, 2)?;
            let x_next = repeat_interleave_channels(&x_next, self.repeats)?;
            let shortcut = CudaTensor::cat(&[&x_first, &x_next], 2)?;
            h.add(&shortcut)
        } else {
            let h = dcae_upsample_rearrange(&h, r1, 2, 2)?;
            let shortcut = repeat_interleave_channels(x, self.repeats)?;
            let shortcut = dcae_upsample_rearrange(&shortcut, r1, 2, 2)?;
            h.add(&shortcut)
        }
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
        upsample_out: Option<(usize, bool)>,
    ) -> Result<Self> {
        let mut resnets = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            let ic = if i == 0 { in_c } else { out_c };
            resnets.push(ResnetBlock::zeros(ic, out_c)?);
        }
        let upsample = match upsample_out {
            Some((uc, temporal)) => Some(Upsample::zeros(out_c, uc, temporal)?),
            None => None,
        };
        Ok(Self { resnets, upsample })
    }

    fn load(
        map: &WeightMap,
        prefix: &str,
        in_c: usize,
        out_c: usize,
        num_layers: usize,
        upsample_out: Option<(usize, bool)>,
    ) -> Result<Self> {
        let mut resnets = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            let ic = if i == 0 { in_c } else { out_c };
            resnets.push(ResnetBlock::load(
                map,
                &format!("{prefix}.resnets.{i}"),
                ic,
                out_c,
            )?);
        }
        let upsample = match upsample_out {
            Some((uc, temporal)) => Some(Upsample::load(
                map,
                &format!("{prefix}.upsamplers.0"),
                out_c,
                uc,
                temporal,
            )?),
            None => None,
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

/// Decoder half of `AutoencoderKLHunyuanVideo15`.
pub struct Hunyuan15Vae {
    pub cfg: Hunyuan15VaeConfig,
    conv_in: CausalConv3d,
    mid: MidBlock,
    up_blocks: Vec<UpBlock>,
    norm_out: CudaTensor,
    conv_out: CausalConv3d,
    repeat: usize,
}

impl Hunyuan15Vae {
    pub fn from_config(cfg: Hunyuan15VaeConfig) -> Self {
        // Soft handle when weights are missing — decode will still refuse until
        // `zeros` / `load` builds the graph.
        Self {
            repeat: 0,
            conv_in: CausalConv3d::zeros(1, 1, 1).expect("placeholder"),
            mid: MidBlock::zeros(1).expect("placeholder"),
            up_blocks: Vec::new(),
            norm_out: CudaTensor::ones(&[1]),
            conv_out: CausalConv3d::zeros(1, 1, 1).expect("placeholder"),
            cfg,
        }
    }

    /// Tiny decoder for host unit tests (decoder channels `(16,8,8,8,8)`).
    pub fn zeros_tiny() -> Result<Self> {
        let cfg = Hunyuan15VaeConfig {
            in_channels: 3,
            out_channels: 3,
            latent_channels: 4,
            // Encoder-order channels; decoder uses the reverse.
            block_out_channels: [8, 8, 8, 8, 16],
            layers_per_block: 1,
            spatial_compression_ratio: 4,
            temporal_compression_ratio: 2,
            scaling_factor: 1.0,
        };
        Self::build(&cfg, None)
    }

    pub fn load(map: &WeightMap, cfg: Hunyuan15VaeConfig) -> Result<Self> {
        Self::build(&cfg, Some(map))
    }

    pub fn scaling_factor(&self) -> f32 {
        self.cfg.scaling_factor
    }

    fn build(cfg: &Hunyuan15VaeConfig, map: Option<&WeightMap>) -> Result<Self> {
        let dec: Vec<usize> = cfg.block_out_channels.iter().copied().rev().collect();
        let latent = cfg.latent_channels;
        let layers = cfg.layers_per_block + 1;
        let log_s = (cfg.spatial_compression_ratio as f64).log2() as usize;
        let log_t = (cfg.temporal_compression_ratio as f64).log2() as usize;
        let repeat = dec[0] / latent;

        let conv_in = match map {
            Some(m) => CausalConv3d::load(m, "decoder.conv_in", latent, dec[0], 3)?,
            None => CausalConv3d::zeros(latent, dec[0], 3)?,
        };
        let mid = match map {
            Some(m) => MidBlock::load(m, "decoder.mid_block", dec[0])?,
            None => MidBlock::zeros(dec[0])?,
        };

        let mut up_blocks = Vec::new();
        let mut input_c = dec[0];
        for i in 0..dec.len() {
            let out_c = dec[i];
            let add_spatial = i < log_s;
            let add_temporal = i < log_t;
            let upsample_out = if add_spatial || add_temporal {
                let uc = if i + 1 < dec.len() { dec[i + 1] } else { out_c };
                Some((uc, add_temporal))
            } else {
                None
            };
            let block = match map {
                Some(m) => UpBlock::load(
                    m,
                    &format!("decoder.up_blocks.{i}"),
                    input_c,
                    out_c,
                    layers,
                    upsample_out,
                )?,
                None => UpBlock::zeros(input_c, out_c, layers, upsample_out)?,
            };
            input_c = upsample_out.map(|(uc, _)| uc).unwrap_or(out_c);
            up_blocks.push(block);
        }

        let last = *dec.last().unwrap();
        let (norm_out, conv_out) = match map {
            Some(m) => (
                load_gamma(m, "decoder.norm_out.gamma", last)?,
                CausalConv3d::load(m, "decoder.conv_out", last, cfg.out_channels, 3)?,
            ),
            None => (
                pinned(CudaTensor::ones(&[last]))?,
                CausalConv3d::zeros(last, cfg.out_channels, 3)?,
            ),
        };

        Ok(Self {
            cfg: cfg.clone(),
            conv_in,
            mid,
            up_blocks,
            norm_out,
            conv_out,
            repeat,
        })
    }

    /// `latents` `[B, C, T, H, W]` → RGB `[B, 3, Tf, Hf, Wf]`.
    pub fn decode(&self, latents: &CudaTensor) -> Result<CudaTensor> {
        if self.up_blocks.is_empty() {
            return Err(msg(
                "HunyuanVideo 1.5 VAE: call load() or zeros_tiny() before decode",
            ));
        }
        let [b, c, t, h, w] = match latents.shape[..] {
            [b, c, t, h, w] => [b, c, t, h, w],
            _ => {
                return Err(msg(format!(
                    "hy15 vae: latents {:?} want [B,C,T,H,W]",
                    latents.shape
                )))
            }
        };
        if c != self.cfg.latent_channels {
            return Err(msg(format!(
                "hy15 vae: latent channels {c} vs {}",
                self.cfg.latent_channels
            )));
        }
        let _ = (b, t, h, w);
        let mut hs = self.conv_in.forward(latents)?;
        let skip = repeat_interleave_channels(latents, self.repeat)?;
        hs = hs.add(&skip)?;
        hs = self.mid.forward(&hs)?;
        for block in &self.up_blocks {
            hs = block.forward(&hs)?;
        }
        hs = rms_l2_channels(&hs, &self.norm_out)?.silu();
        self.conv_out.forward(&hs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frame_math() {
        let cfg = Hunyuan15VaeConfig::fasthunyuan15();
        assert_eq!(latent_frames(121, cfg.temporal_compression_ratio), 31);
        assert_eq!(pixel_frames(31, cfg.temporal_compression_ratio), 121);
        assert_eq!(latent_hw(480, cfg.spatial_compression_ratio), 30);
    }

    #[test]
    fn tiny_decode_shapes() {
        let vae = Hunyuan15Vae::zeros_tiny().unwrap();
        // latent 4ch, T=2,H=2,W=2 → temporal×2 spatial×4 → T=3, H=8, W=8
        let lat = CudaTensor::zeros(&[1, 4, 2, 2, 2]);
        let out = vae.decode(&lat).unwrap();
        assert_eq!(out.shape[0], 1);
        assert_eq!(out.shape[1], 3);
        assert_eq!(out.shape[2], pixel_frames(2, 2));
        assert_eq!(out.shape[3], 2 * 4);
        assert_eq!(out.shape[4], 2 * 4);
    }
}
