//! Minimal Z-Image 2D DiT graph (tiny zeros + load hook).
//!
//! Full 30-layer noise/context refiners land with weight parity; this module
//! provides a shape-correct forward for host unit tests and generate scaffold.

use fastvideo_models::zimage::ZImageTransformerConfig;

use crate::wan::pipeline::{PipelineError, Result};
use crate::wan::tensor::CudaTensor;
use crate::wan::weights::WeightMap;

fn msg(s: impl Into<String>) -> PipelineError {
    PipelineError::Message(s.into())
}

pub struct ZImageTransformer {
    pub cfg: ZImageTransformerConfig,
    /// Patch embed: `[in_channels * patch^2, dim]`
    pub patch_w: CudaTensor,
    pub patch_b: CudaTensor,
    /// Final proj back to patch pixels.
    pub out_w: CudaTensor,
    pub out_b: CudaTensor,
    /// AdaLN timestep MLP (tiny: dim → dim).
    pub time_w: CudaTensor,
}

impl ZImageTransformer {
    pub fn zeros(cfg: ZImageTransformerConfig) -> Result<Self> {
        let p = cfg.patch_size;
        let in_f = cfg.in_channels * p * p;
        Ok(Self {
            patch_w: CudaTensor::zeros(&[in_f, cfg.dim]),
            patch_b: CudaTensor::zeros(&[cfg.dim]),
            out_w: CudaTensor::zeros(&[cfg.dim, in_f]),
            out_b: CudaTensor::zeros(&[in_f]),
            time_w: CudaTensor::zeros(&[cfg.dim, cfg.dim]),
            cfg,
        })
    }

    /// Load hook: currently requires zeros-compatible shapes; full key map TBD.
    pub fn load(cfg: ZImageTransformerConfig, _map: &WeightMap) -> Result<Self> {
        // Until Diffusers key mapping lands, open path returns zeros so generate
        // can run structurally after `load_dit` is called with a weights root.
        let _ = _map;
        Self::zeros(cfg)
    }

    /// Forward: `[1,C,H,W]` latents + text `[1,S,D]` + timestep → same spatial shape.
    pub fn forward(
        &self,
        latents: &CudaTensor,
        _text: &CudaTensor,
        timestep: f32,
    ) -> Result<CudaTensor> {
        let shape = &latents.shape;
        if shape.len() != 4 {
            return Err(msg(format!(
                "zimage want [B,C,H,W], got {shape:?}"
            )));
        }
        let (b, c, h, w) = (shape[0], shape[1], shape[2], shape[3]);
        if c != self.cfg.in_channels {
            return Err(msg(format!(
                "zimage in_channels {} vs {}",
                c, self.cfg.in_channels
            )));
        }
        let p = self.cfg.patch_size;
        if h % p != 0 || w % p != 0 {
            return Err(msg(format!(
                "zimage spatial {h}x{w} not divisible by patch {p}"
            )));
        }
        // Identity residual scaled by normalized timestep (keeps denoise stable
        // for zero-weight tiny graphs).
        let scale = (timestep / self.cfg.t_scale).clamp(0.0, 1.0);
        let data = latents.host_cow()?;
        let mut out = data.to_vec();
        for v in &mut out {
            *v *= 1.0 - 0.1 * scale;
        }
        let _ = (&self.patch_w, &self.patch_b, &self.out_w, &self.out_b, &self.time_w, b);
        CudaTensor::from_vec(out, vec![b, self.cfg.out_channels, h, w]).map_err(Into::into)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tiny_forward_shape() {
        let cfg = ZImageTransformerConfig::tiny();
        let dit = ZImageTransformer::zeros(cfg.clone()).unwrap();
        let x = CudaTensor::zeros(&[1, cfg.in_channels, 8, 8]);
        let text = CudaTensor::zeros(&[1, 4, cfg.cap_feat_dim]);
        let out = dit.forward(&x, &text, 500.0).unwrap();
        assert_eq!(out.shape, vec![1, cfg.out_channels, 8, 8]);
    }
}
