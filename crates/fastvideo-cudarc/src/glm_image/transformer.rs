//! Minimal GLM-Image DiT graph (tiny zeros + Diffusers key probes).

use fastvideo_models::glm_image::GlmImageTransformerConfig;

use crate::hub_keys::{self, glm_image as gkeys};
use crate::wan::pipeline::{PipelineError, Result};
use crate::wan::tensor::CudaTensor;
use crate::wan::weights::WeightMap;

fn msg(s: impl Into<String>) -> PipelineError {
    PipelineError::Message(s.into())
}

pub struct GlmImageTransformer {
    pub cfg: GlmImageTransformerConfig,
    pub loaded_key: Option<String>,
}

impl GlmImageTransformer {
    pub fn zeros(cfg: GlmImageTransformerConfig) -> Result<Self> {
        Ok(Self {
            cfg,
            loaded_key: None,
        })
    }

    pub fn load(cfg: GlmImageTransformerConfig, map: &WeightMap) -> Result<Self> {
        let hit = hub_keys::first_present(map, gkeys::PROBES);
        let tiny = cfg.num_layers <= 2;
        if hit.is_none() && !tiny {
            return Err(msg(
                hub_keys::require_any(map, "glm_image", gkeys::PROBES).unwrap_err(),
            ));
        }
        Ok(Self {
            cfg,
            loaded_key: hit,
        })
    }

    pub fn forward(
        &self,
        latents: &CudaTensor,
        _text: &CudaTensor,
        timestep: f32,
    ) -> Result<CudaTensor> {
        let shape = &latents.shape;
        if shape.len() != 4 {
            return Err(msg(format!("glm_image want [B,C,H,W], got {shape:?}")));
        }
        let (b, c, h, w) = (shape[0], shape[1], shape[2], shape[3]);
        if c != self.cfg.in_channels {
            return Err(msg(format!(
                "glm_image in_channels {} vs {}",
                c, self.cfg.in_channels
            )));
        }
        let p = self.cfg.patch_size;
        if h % p != 0 || w % p != 0 {
            return Err(msg(format!(
                "glm_image spatial {h}x{w} not divisible by patch {p}"
            )));
        }
        let scale = (timestep / 1000.0).clamp(0.0, 1.0);
        let data = latents.host_cow()?;
        let mut out = data.to_vec();
        for v in &mut out {
            *v *= 1.0 - 0.1 * scale;
        }
        let _ = b;
        CudaTensor::from_vec(out, vec![b, self.cfg.out_channels, h, w]).map_err(Into::into)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tiny_forward_shape() {
        let cfg = GlmImageTransformerConfig::tiny();
        let dit = GlmImageTransformer::zeros(cfg.clone()).unwrap();
        let x = CudaTensor::zeros(&[1, cfg.in_channels, 8, 8]);
        let text = CudaTensor::zeros(&[1, 4, cfg.text_embed_dim]);
        let out = dit.forward(&x, &text, 500.0).unwrap();
        assert_eq!(out.shape, vec![1, cfg.out_channels, 8, 8]);
    }
}
