//! Minimal MMAudio DiT graph (tiny zeros + load hook).

use fastvideo_models::mmaudio::MmAudioDiTConfig;

use crate::wan::pipeline::{PipelineError, Result};
use crate::wan::tensor::CudaTensor;
use crate::wan::weights::WeightMap;

fn msg(s: impl Into<String>) -> PipelineError {
    PipelineError::Message(s.into())
}

pub struct MmAudioTransformer {
    pub cfg: MmAudioDiTConfig,
}

impl MmAudioTransformer {
    pub fn zeros(cfg: MmAudioDiTConfig) -> Result<Self> {
        Ok(Self { cfg })
    }

    pub fn load(cfg: MmAudioDiTConfig, _map: &WeightMap) -> Result<Self> {
        let _ = _map;
        Self::zeros(cfg)
    }

    pub fn forward(
        &self,
        latents: &CudaTensor,
        _text: &CudaTensor,
        _visual: Option<&CudaTensor>,
        timestep: f32,
    ) -> Result<CudaTensor> {
        let shape = &latents.shape;
        if shape.len() != 3 {
            return Err(msg(format!("mmaudio want [B,C,T], got {shape:?}")));
        }
        let (b, c, t) = (shape[0], shape[1], shape[2]);
        if c != self.cfg.in_channels {
            return Err(msg(format!(
                "mmaudio in_channels {} vs {}",
                c, self.cfg.in_channels
            )));
        }
        let scale = (timestep / 1000.0).clamp(0.0, 1.0);
        let data = latents.host_cow()?;
        let mut out = data.to_vec();
        for v in &mut out {
            *v *= 1.0 - 0.1 * scale;
        }
        let _ = b;
        CudaTensor::from_vec(out, vec![b, self.cfg.out_channels, t]).map_err(Into::into)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tiny_forward_shape() {
        let cfg = MmAudioDiTConfig::tiny();
        let dit = MmAudioTransformer::zeros(cfg.clone()).unwrap();
        let x = CudaTensor::zeros(&[1, cfg.in_channels, cfg.sample_size]);
        let text = CudaTensor::zeros(&[1, 4, cfg.text_dim]);
        let out = dit.forward(&x, &text, None, 500.0).unwrap();
        assert_eq!(out.shape, vec![1, cfg.out_channels, cfg.sample_size]);
    }
}
