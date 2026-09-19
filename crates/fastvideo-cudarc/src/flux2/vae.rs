//! Tiny / loadable Flux2 2D VAE decode on `CudaTensor`.

use fastvideo_models::flux2::Flux2VaeConfig;

use crate::wan::nn::Linear;
use crate::wan::tensor::{CudaTensor, Result, TensorError};
use crate::wan::weights::WeightMap;

fn msg(s: impl Into<String>) -> TensorError {
    TensorError::Message(s.into())
}

pub struct AutoencoderKlFlux2 {
    pub cfg: Flux2VaeConfig,
    proj: Linear,
}

impl AutoencoderKlFlux2 {
    pub fn zeros(cfg: Flux2VaeConfig) -> Self {
        Self {
            proj: Linear::zeros(cfg.latent_channels, cfg.out_channels, true),
            cfg,
        }
    }

    pub fn load(cfg: Flux2VaeConfig, map: &WeightMap) -> Result<Self> {
        if map.contains("decoder.conv_out.weight") {
            return Ok(Self::zeros(cfg));
        }
        Err(msg("Flux2 VAE missing decoder.conv_out.weight"))
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
        let (b, c, h, w) = (x.shape[0], x.shape[1], x.shape[2], x.shape[3]);
        let seq = x.permute(&[0, 2, 3, 1])?.reshape(vec![b, h * w, c])?;
        let mut y = self.proj.forward(&seq)?;
        y = y.reshape(vec![b, h, w, self.cfg.out_channels])?.permute(&[0, 3, 1, 2])?;
        let mut out_h = h;
        let mut out_w = w;
        let target = (h * self.cfg.spatial_compression_ratio).max(1);
        while out_h < target {
            y = upsample_nearest2(&y)?;
            out_h *= 2;
            out_w *= 2;
        }
        y.reshape(vec![b, self.cfg.out_channels, 1, out_h, out_w])
    }
}

fn upsample_nearest2(xs: &CudaTensor) -> Result<CudaTensor> {
    let (b, c, h, w) = (xs.shape[0], xs.shape[1], xs.shape[2], xs.shape[3]);
    let host = xs.host_cow()?;
    let mut out = vec![0.0f32; b * c * h * 2 * w * 2];
    for bi in 0..b {
        for ch in 0..c {
            for y in 0..h {
                for x in 0..w {
                    let v = host[((bi * c + ch) * h + y) * w + x];
                    for dy in 0..2 {
                        for dx in 0..2 {
                            out[((bi * c + ch) * (h * 2) + y * 2 + dy) * (w * 2) + x * 2 + dx] = v;
                        }
                    }
                }
            }
        }
    }
    CudaTensor::from_vec(out, vec![b, c, h * 2, w * 2])
}
