//! Minimal StableAudioDiT graph (tiny zeros + Diffusers key probes).

use fastvideo_models::stable_audio::StableAudioDiTConfig;

use crate::hub_keys::{self, stable_audio as sakeys};
use crate::wan::pipeline::{PipelineError, Result};
use crate::wan::tensor::CudaTensor;
use crate::wan::weights::WeightMap;

fn msg(s: impl Into<String>) -> PipelineError {
    PipelineError::Message(s.into())
}

pub struct StableAudioTransformer {
    pub cfg: StableAudioDiTConfig,
    pub loaded_key: Option<String>,
}

impl StableAudioTransformer {
    pub fn zeros(cfg: StableAudioDiTConfig) -> Result<Self> {
        Ok(Self {
            cfg,
            loaded_key: None,
        })
    }

    pub fn load(cfg: StableAudioDiTConfig, map: &WeightMap) -> Result<Self> {
        let hit = hub_keys::first_present(map, sakeys::PROBES);
        let tiny = cfg.num_layers <= 2;
        if hit.is_none() && !tiny {
            return Err(msg(hub_keys::require_any(
                map,
                "stable_audio",
                sakeys::PROBES,
            )
            .unwrap_err()));
        }
        Ok(Self {
            cfg,
            loaded_key: hit,
        })
    }

    /// Latents `[1,C,T]` + text `[1,S,D]` → same shape.
    pub fn forward(
        &self,
        latents: &CudaTensor,
        _text: &CudaTensor,
        timestep: f32,
    ) -> Result<CudaTensor> {
        let shape = &latents.shape;
        if shape.len() != 3 {
            return Err(msg(format!("stable_audio want [B,C,T], got {shape:?}")));
        }
        let (b, c, t) = (shape[0], shape[1], shape[2]);
        if c != self.cfg.in_channels {
            return Err(msg(format!(
                "stable_audio in_channels {} vs {}",
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
        let cfg = StableAudioDiTConfig::tiny();
        let dit = StableAudioTransformer::zeros(cfg.clone()).unwrap();
        let x = CudaTensor::zeros(&[1, cfg.in_channels, cfg.sample_size]);
        let text = CudaTensor::zeros(&[1, 4, cfg.cross_attention_dim]);
        let out = dit.forward(&x, &text, 500.0).unwrap();
        assert_eq!(out.shape, vec![1, cfg.out_channels, cfg.sample_size]);
    }
}
