//! Wan 2.1 VAE encoder, chunked with the causal feature cache exactly as
//! Diffusers' `AutoencoderKLWan._encode` runs it: frame 0 alone, then four
//! frames per pass, every causal conv fed the previous pass's last frames,
//! and the `downsample3d` time conv skipped on the first pass (its stride-2
//! conv then runs over the cached last frame plus the new ones). The spatial
//! downsample is `ZeroPad2d((0, 1, 0, 1))` + a stride-2 3×3 conv
//! ([`super::vae22::Downsample`], the same `WanResample` code).
//!
//! The memory of a pass is bounded by four frames, which is what makes a
//! 1088×1920×121 clip (the LingBot refiner input) encodable; the uncached
//! whole-clip encoder in [`super::vae`] would hold every frame's 96-channel
//! activations at once.

use super::tensor::{CudaTensor, Result, TensorError};
use super::vae::{
    conv_cached, gamma, rms_silu_video, AttentionBlock, CausalConv3d, FeatCache, ResidualBlock,
};
use super::vae22::Downsample;
use super::weights::WeightMap;

fn msg(s: impl Into<String>) -> TensorError {
    TensorError::Message(s.into())
}

#[derive(Debug, Clone)]
enum Stage {
    Res(ResidualBlock),
    Down(Downsample),
}

/// `WanEncoder3d` (base 96, mults 1/2/4/4, 2 res blocks, temporal
/// downsample `[false, true, true]`, z 16) plus `quant_conv`.
#[derive(Debug, Clone)]
pub struct Wan21ChunkedEncoder {
    conv_in: CausalConv3d,
    stages: Vec<Stage>,
    mid_res0: ResidualBlock,
    mid_attn: AttentionBlock,
    mid_res1: ResidualBlock,
    norm_out: CudaTensor,
    conv_out: CausalConv3d,
    quant: CausalConv3d,
    z_dim: usize,
}

/// `(in, out)` widths of the 8 residual blocks and where the downsamples sit.
fn layout(base: usize) -> Vec<(usize, usize, Option<bool>)> {
    // (in, out, Some(temporal)) for a downsample entry; None for a resblock.
    let dims = [base, base, 2 * base, 4 * base, 4 * base];
    let temporal = [false, true, true];
    let mut out = Vec::new();
    for i in 0..4 {
        let (mut cin, cout) = (dims[i], dims[i + 1]);
        for _ in 0..2 {
            out.push((cin, cout, None));
            cin = cout;
        }
        if i < 3 {
            out.push((cout, cout, Some(temporal[i])));
        }
    }
    out
}

impl Wan21ChunkedEncoder {
    pub fn load(map: &WeightMap) -> Result<Self> {
        Self::build(Some(map), 96, 16)
    }

    /// Zero-weight encoder of base width `base` (unit tests).
    pub fn zeros(base: usize, z_dim: usize) -> Result<Self> {
        Self::build(None, base, z_dim)
    }

    fn build(map: Option<&WeightMap>, base: usize, z_dim: usize) -> Result<Self> {
        let top = 4 * base;
        let conv = |key: &str, i: usize, o: usize, k: [usize; 3], p: [usize; 3]| -> Result<CausalConv3d> {
            match map {
                Some(m) => CausalConv3d::load(m, key, i, o, k, [1, 1, 1], p),
                None => Ok(CausalConv3d::zeros(i, o, k, [1, 1, 1], p)),
            }
        };
        let res = |key: &str, i: usize, o: usize| -> Result<ResidualBlock> {
            match map {
                Some(m) => ResidualBlock::load(m, key, i, o),
                None => Ok(ResidualBlock::zeros(i, o)),
            }
        };
        let mut stages = Vec::new();
        for (j, (i, o, down)) in layout(base).into_iter().enumerate() {
            let key = format!("encoder.down_blocks.{j}");
            stages.push(match down {
                None => Stage::Res(res(&key, i, o)?),
                Some(t) => Stage::Down(match map {
                    Some(m) => Downsample::load(m, &key, o, t)?,
                    None => Downsample::zeros(o, t),
                }),
            });
        }
        let (mid_attn, norm_out) = match map {
            Some(m) => (
                AttentionBlock::load(m, "encoder.mid_block.attentions.0", top)?,
                gamma(m, "encoder.norm_out.gamma", &[top, 1, 1, 1])?,
            ),
            None => (AttentionBlock::zeros(top), CudaTensor::ones(&[top, 1, 1, 1])),
        };
        Ok(Self {
            conv_in: conv("encoder.conv_in", 3, base, [3, 3, 3], [1, 1, 1])?,
            stages,
            mid_res0: res("encoder.mid_block.resnets.0", top, top)?,
            mid_attn,
            mid_res1: res("encoder.mid_block.resnets.1", top, top)?,
            norm_out,
            conv_out: conv("encoder.conv_out", top, 2 * z_dim, [3, 3, 3], [1, 1, 1])?,
            quant: conv("quant_conv", 2 * z_dim, 2 * z_dim, [1, 1, 1], [0, 0, 0])?,
            z_dim,
        })
    }

    fn forward_chunk(&self, xs: &CudaTensor, cache: &mut FeatCache) -> Result<CudaTensor> {
        let mut x = conv_cached(&self.conv_in, xs, Some(cache))?;
        for s in &self.stages {
            x = match s {
                Stage::Res(r) => r.forward(&x, Some(cache))?,
                Stage::Down(d) => d.forward(&x, Some(cache))?,
            };
        }
        x = self.mid_res0.forward(&x, Some(cache))?;
        x = self.mid_attn.forward(&x)?;
        x = self.mid_res1.forward(&x, Some(cache))?;
        x = rms_silu_video(&x, &self.norm_out)?;
        conv_cached(&self.conv_out, &x, Some(cache))
    }

    /// `[1, 3, F, H, W]` in `[-1, 1]` (F = 1 + 4k) → the posterior mean
    /// `[1, z, 1 + (F-1)/4, H/8, W/8]` in VAE space, f32 activations.
    pub fn encode_mean(&self, video: &CudaTensor) -> Result<CudaTensor> {
        super::tensor::with_bf16_act(false, || {
            let x = video.to_f32_act()?;
            let t = x.dim(2)?;
            if t == 0 {
                return Err(msg("wan21 encode: no frames"));
            }
            let iters = 1 + (t - 1) / 4;
            let mut cache = FeatCache::new();
            let mut outs = Vec::with_capacity(iters);
            for i in 0..iters {
                cache.begin_pass();
                let chunk = if i == 0 {
                    x.narrow(2, 0, 1)?
                } else {
                    let start = 1 + 4 * (i - 1);
                    x.narrow(2, start, 4.min(t - start))?
                };
                outs.push(self.forward_chunk(&chunk, &mut cache)?);
            }
            let refs: Vec<&CudaTensor> = outs.iter().collect();
            let enc = self.quant.forward(&CudaTensor::cat(&refs, 2)?)?;
            enc.narrow(1, 0, self.z_dim)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layout_matches_diffusers_down_blocks() {
        let l = layout(96);
        assert_eq!(l.len(), 11);
        assert_eq!(l[2], (96, 96, Some(false)));
        assert_eq!(l[3], (96, 192, None));
        assert_eq!(l[5], (192, 192, Some(true)));
        assert_eq!(l[8], (384, 384, Some(true)));
        assert_eq!(l[10], (384, 384, None));
    }

    #[test]
    fn chunked_encode_shapes() {
        let enc = Wan21ChunkedEncoder::zeros(4, 3).unwrap();
        let v = CudaTensor::zeros(&[1, 3, 9, 16, 24]);
        let z = enc.encode_mean(&v).unwrap();
        assert_eq!(z.shape, vec![1, 3, 3, 2, 3]);
    }
}
