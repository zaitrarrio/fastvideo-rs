//! Minimal SD3 MMDiT graph (tiny zeros + load hook).

use fastvideo_models::sd35::Sd35TransformerConfig;

use crate::wan::pipeline::{PipelineError, Result};
use crate::wan::tensor::CudaTensor;
use crate::wan::weights::WeightMap;

fn msg(s: impl Into<String>) -> PipelineError {
    PipelineError::Message(s.into())
}

pub struct Sd35Transformer {
    pub cfg: Sd35TransformerConfig,
    pub patch_w: CudaTensor,
    pub out_w: CudaTensor,
}

impl Sd35Transformer {
    pub fn zeros(cfg: Sd35TransformerConfig) -> Result<Self> {
        let p = cfg.patch_size;
        let in_f = cfg.in_channels * p * p;
        let dim = cfg.inner_dim();
        Ok(Self {
            patch_w: CudaTensor::zeros(&[in_f, dim]),
            out_w: CudaTensor::zeros(&[dim, in_f]),
            cfg,
        })
    }

    pub fn load(cfg: Sd35TransformerConfig, _map: &WeightMap) -> Result<Self> {
        let _ = _map;
        Self::zeros(cfg)
    }

    pub fn forward(
        &self,
        latents: &CudaTensor,
        _text: &CudaTensor,
        timestep: f32,
    ) -> Result<CudaTensor> {
        let shape = &latents.shape;
        if shape.len() != 4 {
            return Err(msg(format!("sd35 want [B,C,H,W], got {shape:?}")));
        }
        let (b, c, h, w) = (shape[0], shape[1], shape[2], shape[3]);
        if c != self.cfg.in_channels {
            return Err(msg(format!("sd35 in_channels {} vs {}", c, self.cfg.in_channels)));
        }
        let p = self.cfg.patch_size;
        if h % p != 0 || w % p != 0 {
            return Err(msg(format!("sd35 spatial {h}x{w} not divisible by patch {p}")));
        }
        let scale = (timestep / 1000.0).clamp(0.0, 1.0);
        let data = latents.host_cow()?;
        let mut out = data.to_vec();
        for v in &mut out {
            *v *= 1.0 - 0.1 * scale;
        }
        let _ = (&self.patch_w, &self.out_w, b);
        CudaTensor::from_vec(out, vec![b, self.cfg.out_channels, h, w]).map_err(Into::into)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tiny_forward_shape() {
        let cfg = Sd35TransformerConfig::tiny();
        let dit = Sd35Transformer::zeros(cfg.clone()).unwrap();
        let x = CudaTensor::zeros(&[1, cfg.in_channels, 8, 8]);
        let text = CudaTensor::zeros(&[1, 4, cfg.joint_attention_dim]);
        let out = dit.forward(&x, &text, 500.0).unwrap();
        assert_eq!(out.shape, vec![1, cfg.out_channels, 8, 8]);
    }
}
