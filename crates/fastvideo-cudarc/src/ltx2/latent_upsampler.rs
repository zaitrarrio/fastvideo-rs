//! `LTX2LatentUpsamplerModel`: spatial ×2 on **de-normalized** video latents.
//!
//! Architecture (LTX-2.5 pack, `use_rational_resampler=false`):
//! Conv3d stem → GroupNorm(32) → SiLU → 4 ResBlocks (3-D) → per-frame
//! Conv2d→PixelShuffle(2) → 4 ResBlocks → Conv3d head. Audio is not touched.
//! See docs/ports/ltx25.md §two-stage and diffusers `latent_upsampler.py`.

use fastvideo_models::ltx2::config::Ltx2LatentUpsamplerConfig;

use crate::wan::ops::PadMode;
use crate::wan::tensor::{CudaTensor, Result};
use crate::wan::weights::{cuda_tensor_shaped, WeightMap};

use super::msg;

struct Conv3d {
    weight: CudaTensor,
    bias: CudaTensor,
}

impl Conv3d {
    fn load(map: &WeightMap, prefix: &str, cin: usize, cout: usize) -> Result<Self> {
        let mut weight =
            cuda_tensor_shaped(map, &format!("{prefix}.weight"), &[cout, cin, 3, 3, 3])?;
        let mut bias = cuda_tensor_shaped(map, &format!("{prefix}.bias"), &[cout])?;
        weight.pin_device()?;
        bias.pin_device()?;
        Ok(Self { weight, bias })
    }

    fn forward(&self, x: &CudaTensor) -> Result<CudaTensor> {
        // Same padding for k=3: one zero on each side of F/H/W.
        x.pad(2, 1, 1, PadMode::Zeros)?
            .pad(3, 1, 1, PadMode::Zeros)?
            .pad(4, 1, 1, PadMode::Zeros)?
            .conv3d(&self.weight, Some(&self.bias), [0, 0, 0], [1, 1, 1])
    }
}

struct Conv2d {
    weight: CudaTensor,
    bias: CudaTensor,
}

impl Conv2d {
    fn load(map: &WeightMap, prefix: &str, cin: usize, cout: usize) -> Result<Self> {
        let mut weight = cuda_tensor_shaped(map, &format!("{prefix}.weight"), &[cout, cin, 3, 3])?;
        let mut bias = cuda_tensor_shaped(map, &format!("{prefix}.bias"), &[cout])?;
        weight.pin_device()?;
        bias.pin_device()?;
        Ok(Self { weight, bias })
    }

    fn forward(&self, x: &CudaTensor) -> Result<CudaTensor> {
        x.conv2d(&self.weight, Some(&self.bias), 1, 1)
    }
}

struct GroupNorm {
    weight: CudaTensor,
    bias: CudaTensor,
    groups: usize,
}

impl GroupNorm {
    fn load(map: &WeightMap, prefix: &str, channels: usize) -> Result<Self> {
        let mut weight = cuda_tensor_shaped(map, &format!("{prefix}.weight"), &[channels])?;
        let mut bias = cuda_tensor_shaped(map, &format!("{prefix}.bias"), &[channels])?;
        weight.pin_device()?;
        bias.pin_device()?;
        Ok(Self {
            weight,
            bias,
            groups: 32,
        })
    }

    fn forward(&self, x: &CudaTensor) -> Result<CudaTensor> {
        x.group_norm(self.groups, &self.weight, &self.bias, 1e-6, false)
    }
}

/// `conv → GroupNorm → SiLU → conv → GroupNorm → SiLU(x + residual)`.
struct ResBlock {
    conv1: Conv3d,
    norm1: GroupNorm,
    conv2: Conv3d,
    norm2: GroupNorm,
}

impl ResBlock {
    fn load(map: &WeightMap, prefix: &str, channels: usize) -> Result<Self> {
        Ok(Self {
            conv1: Conv3d::load(map, &format!("{prefix}.conv1"), channels, channels)?,
            norm1: GroupNorm::load(map, &format!("{prefix}.norm1"), channels)?,
            conv2: Conv3d::load(map, &format!("{prefix}.conv2"), channels, channels)?,
            norm2: GroupNorm::load(map, &format!("{prefix}.norm2"), channels)?,
        })
    }

    fn forward(&self, x: &CudaTensor) -> Result<CudaTensor> {
        let h = self.norm1.forward(&self.conv1.forward(x)?)?.silu();
        let h = self.norm2.forward(&self.conv2.forward(&h)?)?;
        Ok(h.add(x)?.silu())
    }
}

/// Spatial PixelShuffle: `[N, 4C, H, W] → [N, C, 2H, 2W]`.
fn pixel_shuffle2(x: &CudaTensor) -> Result<CudaTensor> {
    let [n, c4, h, w] = match x.shape[..] {
        [n, c4, h, w] => [n, c4, h, w],
        _ => {
            return Err(msg(format!(
                "pixel_shuffle2 expects [N, C, H, W], got {:?}",
                x.shape
            )))
        }
    };
    if !c4.is_multiple_of(4) {
        return Err(msg(format!(
            "pixel_shuffle2: channel count {c4} is not 4·C"
        )));
    }
    let c = c4 / 4;
    // [N, C, 2, 2, H, W] → [N, C, H, 2, W, 2] → [N, C, 2H, 2W].
    x.reshape(vec![n, c, 2, 2, h, w])?
        .permute(&[0, 1, 4, 2, 5, 3])?
        .reshape(vec![n, c, h * 2, w * 2])
}

pub struct LatentUpsampler {
    cfg: Ltx2LatentUpsamplerConfig,
    initial_conv: Conv3d,
    initial_norm: GroupNorm,
    res_blocks: Vec<ResBlock>,
    /// `upsampler.0` Conv2d mid → 4·mid (PixelShuffle has no weights).
    upsample_conv: Conv2d,
    post_blocks: Vec<ResBlock>,
    final_conv: Conv3d,
}

impl LatentUpsampler {
    /// `map` is the diffusers `latent_upsampler/` folder.
    pub fn load(map: &WeightMap, cfg: &Ltx2LatentUpsamplerConfig) -> Result<Self> {
        if cfg.dims != 3 || !cfg.spatial_upsample || cfg.temporal_upsample {
            return Err(msg(
                "ltx2 latent upsampler: only dims=3 spatial-only is supported",
            ));
        }
        if cfg.use_rational_resampler {
            return Err(msg("ltx2 latent upsampler: rational resampler is not supported (2.5 pack uses PixelShuffle)"));
        }
        let mid = cfg.mid_channels;
        let n = cfg.num_blocks_per_stage;
        let res_blocks = (0..n)
            .map(|i| ResBlock::load(map, &format!("res_blocks.{i}"), mid))
            .collect::<Result<Vec<_>>>()?;
        let post_blocks = (0..n)
            .map(|i| ResBlock::load(map, &format!("post_upsample_res_blocks.{i}"), mid))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            initial_conv: Conv3d::load(map, "initial_conv", cfg.in_channels, mid)?,
            initial_norm: GroupNorm::load(map, "initial_norm", mid)?,
            res_blocks,
            upsample_conv: Conv2d::load(map, "upsampler.0", mid, 4 * mid)?,
            post_blocks,
            final_conv: Conv3d::load(map, "final_conv", mid, cfg.in_channels)?,
            cfg: cfg.clone(),
        })
    }

    /// De-normalized video `[B, C, F, H, W]` → `[B, C, F, 2H, 2W]`.
    pub fn forward(&self, x: &CudaTensor) -> Result<CudaTensor> {
        let [b, c, f, h, w] = match x.shape[..] {
            [b, c, f, h, w] => [b, c, f, h, w],
            _ => {
                return Err(msg(format!(
                    "latent upsampler expects [B, C, F, H, W], got {:?}",
                    x.shape
                )))
            }
        };
        if c != self.cfg.in_channels || f == 0 || h == 0 || w == 0 {
            return Err(msg(format!(
                "latent upsampler expects [{}, {}, F, H, W], got {:?}",
                "B", self.cfg.in_channels, x.shape
            )));
        }
        let mut y = self
            .initial_norm
            .forward(&self.initial_conv.forward(x)?)?
            .silu();
        for block in &self.res_blocks {
            y = block.forward(&y)?;
        }
        // [B, C, F, H, W] → [B·F, C, H, W] for the 2-D upsample.
        let mid = y.shape[1];
        let flat = y
            .permute(&[0, 2, 1, 3, 4])?
            .reshape(vec![b * f, mid, h, w])?;
        let up = pixel_shuffle2(&self.upsample_conv.forward(&flat)?)?;
        let (h2, w2) = (up.shape[2], up.shape[3]);
        let mut y = up
            .reshape(vec![b, f, mid, h2, w2])?
            .permute(&[0, 2, 1, 3, 4])?;
        for block in &self.post_blocks {
            y = block.forward(&y)?;
        }
        let out = self.final_conv.forward(&y)?;
        if out.shape != [b, c, f, h * 2, w * 2] {
            return Err(msg(format!(
                "latent upsampler produced {:?} from {:?}",
                out.shape, x.shape
            )));
        }
        Ok(out)
    }

    pub fn config(&self) -> &Ltx2LatentUpsamplerConfig {
        &self.cfg
    }
}

#[cfg(test)]
mod tests {
    use super::super::attention::tests::weights;
    use super::*;

    #[test]
    fn spatial_upsample_doubles_hw() {
        // Tiny mid width (GroupNorm needs C % 32 == 0) — geometry + GroupNorm path only.
        let cfg = Ltx2LatentUpsamplerConfig {
            in_channels: 4,
            mid_channels: 32,
            num_blocks_per_stage: 1,
            ..Ltx2LatentUpsamplerConfig::ltx2_5_22b()
        };
        let up = LatentUpsampler::load(&weights(), &cfg).expect("load");
        let x = CudaTensor::zeros(&[1, 4, 2, 4, 6]);
        let y = up.forward(&x).expect("forward");
        assert_eq!(y.shape, vec![1, 4, 2, 8, 12]);
        assert!(y.host_cow().unwrap().iter().all(|v| v.is_finite()));
    }

    #[test]
    fn rational_resampler_is_refused() {
        let cfg = Ltx2LatentUpsamplerConfig {
            use_rational_resampler: true,
            ..Ltx2LatentUpsamplerConfig::ltx2_5_22b()
        };
        assert!(LatentUpsampler::load(&weights(), &cfg).is_err());
    }
}
