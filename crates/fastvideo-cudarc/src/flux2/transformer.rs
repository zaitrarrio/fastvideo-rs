//! Minimal FLUX.2 DiT graph (tiny zeros + load hook).

use fastvideo_models::flux2::Flux2TransformerConfig;

use crate::wan::pipeline::{PipelineError, Result};
use crate::wan::tensor::CudaTensor;
use crate::wan::weights::WeightMap;

fn msg(s: impl Into<String>) -> PipelineError {
    PipelineError::Message(s.into())
}

pub struct Flux2Transformer {
    pub cfg: Flux2TransformerConfig,
}

impl Flux2Transformer {
    pub fn zeros(cfg: Flux2TransformerConfig) -> Result<Self> {
        Ok(Self { cfg })
    }

    pub fn load(cfg: Flux2TransformerConfig, _map: &WeightMap) -> Result<Self> {
        let _ = _map;
        Self::zeros(cfg)
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
            return Err(msg(format!("flux2 want [B,C,H,W], got {shape:?}")));
        }
        let (b, c, h, w) = (shape[0], shape[1], shape[2], shape[3]);
        if c != self.cfg.in_channels {
            return Err(msg(format!("flux2 in_channels {} vs {}", c, self.cfg.in_channels)));
        }
        let scale = (timestep / 1000.0).clamp(0.0, 1.0);
        let g = if self.cfg.guidance_embeds { guidance / 10.0 } else { 0.0 };
        let data = latents.host_cow()?;
        let mut out = data.to_vec();
        for v in &mut out {
            *v *= 1.0 - 0.1 * scale - 0.01 * g;
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
        let cfg = Flux2TransformerConfig::tiny();
        let dit = Flux2Transformer::zeros(cfg.clone()).unwrap();
        let x = CudaTensor::zeros(&[1, cfg.in_channels, 8, 8]);
        let text = CudaTensor::zeros(&[1, 4, cfg.joint_attention_dim]);
        let out = dit.forward(&x, &text, 500.0, 1.0).unwrap();
        assert_eq!(out.shape, vec![1, cfg.out_channels, 8, 8]);
    }
}
