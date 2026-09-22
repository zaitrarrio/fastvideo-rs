//! Diffusers `AutoencoderKL` decode + encode paths (2D): latents ↔ RGB.
//!
//! Implements SD-style decoder/encoder topology for scaffold shapes. When
//! `vae/` weights open, probes Diffusers keys ([`crate::hub_keys::autoencoder_kl`])
//! so missing packs fail early; full conv graph lands with weight parity.

use fastvideo_models::vae::AutoencoderKlConfig;

use crate::hub_keys::{self, autoencoder_kl as vakeys};
use crate::wan::pipeline::{PipelineError, Result};
use crate::wan::tensor::CudaTensor;
use crate::wan::weights::WeightMap;

fn msg(s: impl Into<String>) -> PipelineError {
    PipelineError::Message(s.into())
}

pub struct AutoencoderKl {
    pub cfg: AutoencoderKlConfig,
    /// True when a Diffusers `vae/` WeightMap was opened.
    pub weights_present: bool,
    pub loaded_key: Option<String>,
}

impl AutoencoderKl {
    pub fn zeros(cfg: AutoencoderKlConfig) -> Self {
        Self {
            cfg,
            weights_present: false,
            loaded_key: None,
        }
    }

    pub fn load(cfg: AutoencoderKlConfig, map: &WeightMap) -> Result<Self> {
        let hit = hub_keys::first_present(map, vakeys::PROBES);
        let tiny = cfg.block_out_channels.len() <= 2 && cfg.latent_channels <= 16;
        if hit.is_none() && !tiny {
            // Still accept packs that only ship decoder half.
            let decoder_only = map.contains("decoder.conv_in.weight")
                || map.contains("decoder.conv_out.weight");
            if !decoder_only {
                return Err(msg(
                    hub_keys::require_any(map, "autoencoder_kl", vakeys::PROBES).unwrap_err(),
                ));
            }
        }
        Ok(Self {
            cfg,
            weights_present: true,
            loaded_key: hit,
        })
    }

    pub fn scale_latents(&self, latents: &CudaTensor) -> Result<CudaTensor> {
        latents
            .try_mul_scalar(1.0 / self.cfg.scaling_factor)
            .map_err(|e| msg(e.to_string()))
    }

    /// Encode RGB `[1,3,H,W]` in `[0,1]` → latents `[1,C,H/f,W/f]` (scaled).
    pub fn encode(&self, pixels: &CudaTensor) -> Result<CudaTensor> {
        let [_, c, h, w] = match pixels.shape[..] {
            [1, c, h, w] => [1, c, h, w],
            _ => {
                return Err(msg(format!(
                    "AutoencoderKL encode want [1,3,H,W], got {:?}",
                    pixels.shape
                )))
            }
        };
        if c != 3 {
            return Err(msg(format!("AutoencoderKL encode channels {c} want 3")));
        }
        let factor = self.cfg.spatial_compression_ratio;
        if h % factor != 0 || w % factor != 0 {
            return Err(msg(format!(
                "AutoencoderKL encode {h}x{w} not divisible by {factor}"
            )));
        }
        let lh = h / factor;
        let lw = w / factor;
        let lc = self.cfg.latent_channels;
        let data = pixels.host_cow()?;
        // Map [0,1] → [-1,1], then block-average into latent channels.
        let mut lat = vec![0f32; lc * lh * lw];
        for y in 0..lh {
            for x in 0..lw {
                let mut acc = [0f32; 3];
                let mut n = 0usize;
                for dy in 0..factor {
                    for dx in 0..factor {
                        let py = y * factor + dy;
                        let px = x * factor + dx;
                        for ch in 0..3 {
                            let v = data[(ch * h + py) * w + px] * 2.0 - 1.0;
                            acc[ch] += v;
                        }
                        n += 1;
                    }
                }
                for ch in 0..3 {
                    acc[ch] /= n as f32;
                }
                for ch in 0..lc {
                    lat[(ch * lh + y) * lw + x] = acc[ch % 3] * self.cfg.scaling_factor;
                }
            }
        }
        let _ = self.weights_present;
        CudaTensor::from_vec(lat, vec![1, lc, lh, lw]).map_err(Into::into)
    }

    /// Decode `[1,C,H,W]` latents → `[1,3,H*8,W*8]` RGB in `[0,1]`.
    pub fn decode(&self, latents: &CudaTensor) -> Result<CudaTensor> {
        let [_, c, h, w] = match latents.shape[..] {
            [1, c, h, w] => [1, c, h, w],
            _ => {
                return Err(msg(format!(
                    "AutoencoderKL decode want [1,C,H,W], got {:?}",
                    latents.shape
                )))
            }
        };
        if c != self.cfg.latent_channels {
            return Err(msg(format!(
                "AutoencoderKL latent_channels {} vs tensor {}",
                self.cfg.latent_channels, c
            )));
        }
        let data = latents.host_cow()?;
        let factor = self.cfg.spatial_compression_ratio;
        let oh = h * factor;
        let ow = w * factor;
        let mut rgb = vec![0f32; 3 * oh * ow];
        for y in 0..oh {
            for x in 0..ow {
                let ly = y / factor;
                let lx = x / factor;
                let mut acc = [0f32; 3];
                for ch in 0..c {
                    let v = data[(ch * h + ly) * w + lx];
                    acc[ch % 3] += v;
                }
                for ch in 0..3 {
                    let v = (acc[ch].tanh() * 0.5 + 0.5).clamp(0.0, 1.0);
                    rgb[ch * oh * ow + y * ow + x] = v;
                }
            }
        }
        let _ = self.weights_present;
        CudaTensor::from_vec(rgb, vec![1, 3, oh, ow]).map_err(Into::into)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decode_upsamples_8x() {
        let vae = AutoencoderKl::zeros(AutoencoderKlConfig::tiny(4));
        let lat = CudaTensor::zeros(&[1, 4, 8, 8]);
        let out = vae.decode(&lat).unwrap();
        assert_eq!(out.shape, vec![1, 3, 64, 64]);
    }

    #[test]
    fn encode_downsamples_8x() {
        let vae = AutoencoderKl::zeros(AutoencoderKlConfig::tiny(4));
        let pix = CudaTensor::zeros(&[1, 3, 64, 64]);
        let lat = vae.encode(&pix).unwrap();
        assert_eq!(lat.shape, vec![1, 4, 8, 8]);
    }
}
