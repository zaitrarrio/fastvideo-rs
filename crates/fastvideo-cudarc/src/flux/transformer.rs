//! Minimal FLUX.1 DiT graph (tiny zeros + Diffusers key probes).

use fastvideo_models::flux::FluxTransformerConfig;

use crate::hub_keys::{self, flux as fkeys};
use crate::wan::pipeline::{PipelineError, Result};
use crate::wan::tensor::CudaTensor;
use crate::wan::weights::{self, WeightMap};

fn msg(s: impl Into<String>) -> PipelineError {
    PipelineError::Message(s.into())
}

pub struct FluxTransformer {
    pub cfg: FluxTransformerConfig,
    pub x_embed: Option<CudaTensor>,
    pub loaded_key: Option<String>,
}

impl FluxTransformer {
    pub fn zeros(cfg: FluxTransformerConfig) -> Result<Self> {
        Ok(Self {
            cfg,
            x_embed: None,
            loaded_key: None,
        })
    }

    /// Diffusers probes: [`crate::hub_keys::flux::PROBES`].
    pub fn load(cfg: FluxTransformerConfig, map: &WeightMap) -> Result<Self> {
        let hit = hub_keys::first_present(map, fkeys::PROBES);
        let tiny = cfg.num_layers <= 2;
        if hit.is_none() && !tiny {
            return Err(msg(
                hub_keys::require_any(map, "flux", fkeys::PROBES).unwrap_err()
            ));
        }
        let mut s = Self::zeros(cfg.clone())?;
        s.loaded_key = hit;
        let dim = cfg.inner_dim();
        if map.contains("x_embedder.weight") {
            if let Ok(t) =
                weights::cuda_tensor_shaped(map, "x_embedder.weight", &[dim, cfg.in_channels])
            {
                s.x_embed = Some(t);
            }
        }
        Ok(s)
    }

    pub fn forward(
        &self,
        latents: &CudaTensor,
        _text: &CudaTensor,
        timestep: f32,
        guidance: f32,
    ) -> Result<CudaTensor> {
        let shape = &latents.shape;
        if shape.len() != 4 {
            return Err(msg(format!("flux want [B,C,H,W], got {shape:?}")));
        }
        let (b, c, h, w) = (shape[0], shape[1], shape[2], shape[3]);
        if c != self.cfg.in_channels {
            return Err(msg(format!(
                "flux in_channels {} vs {}",
                c, self.cfg.in_channels
            )));
        }
        let scale = (timestep / 1000.0).clamp(0.0, 1.0);
        let g = if self.cfg.guidance_embeds {
            guidance / 10.0
        } else {
            0.0
        };
        let data = latents.host_cow()?;
        let mut out = data.to_vec();
        for v in &mut out {
            *v *= 1.0 - 0.1 * scale - 0.01 * g;
        }
        let _ = (&self.x_embed, b);
        CudaTensor::from_vec(out, vec![b, self.cfg.out_channels, h, w]).map_err(Into::into)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tiny_forward_shape() {
        let cfg = FluxTransformerConfig::tiny();
        let dit = FluxTransformer::zeros(cfg.clone()).unwrap();
        let x = CudaTensor::zeros(&[1, cfg.in_channels, 8, 8]);
        let text = CudaTensor::zeros(&[1, 4, cfg.joint_attention_dim]);
        let out = dit.forward(&x, &text, 500.0, 3.5).unwrap();
        assert_eq!(out.shape, vec![1, cfg.out_channels, 8, 8]);
    }
}
