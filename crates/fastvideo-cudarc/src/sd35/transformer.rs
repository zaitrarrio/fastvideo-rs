//! Minimal SD3 MMDiT graph (tiny zeros + Diffusers key probes).

use fastvideo_models::sd35::Sd35TransformerConfig;

use crate::hub_keys::{self, sd35 as sdkeys};
use crate::wan::pipeline::{PipelineError, Result};
use crate::wan::tensor::CudaTensor;
use crate::wan::weights::{self, WeightMap};

fn msg(s: impl Into<String>) -> PipelineError {
    PipelineError::Message(s.into())
}

pub struct Sd35Transformer {
    pub cfg: Sd35TransformerConfig,
    pub patch_w: CudaTensor,
    pub out_w: CudaTensor,
    pub loaded_key: Option<String>,
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
            loaded_key: None,
        })
    }

    /// Diffusers probes: [`crate::hub_keys::sd35::PROBES`].
    pub fn load(cfg: Sd35TransformerConfig, map: &WeightMap) -> Result<Self> {
        let hit = hub_keys::first_present(map, sdkeys::PROBES);
        let tiny = cfg.num_layers <= 2;
        if hit.is_none() && !tiny {
            return Err(msg(
                hub_keys::require_any(map, "sd35", sdkeys::PROBES).unwrap_err()
            ));
        }
        let mut s = Self::zeros(cfg.clone())?;
        s.loaded_key = hit;
        let p = cfg.patch_size;
        let in_f = cfg.in_channels * p * p;
        let dim = cfg.inner_dim();
        // PatchEmbed proj is Conv2d [dim, in_c, p, p] — fold to [in_f, dim] if needed.
        if map.contains("pos_embed.proj.weight") {
            if let Ok(t) = weights::cuda_tensor_shaped(
                map,
                "pos_embed.proj.weight",
                &[dim, cfg.in_channels, p, p],
            ) {
                let host = t.host_cow()?;
                let mut flat = vec![0f32; in_f * dim];
                // [dim, in_c, p, p] → treat as [dim, in_f] then transpose → [in_f, dim]
                for o in 0..dim {
                    for i in 0..in_f {
                        flat[i * dim + o] = host[o * in_f + i];
                    }
                }
                s.patch_w = CudaTensor::from_vec(flat, vec![in_f, dim])?;
            }
        }
        if map.contains("proj_out.weight") {
            if let Ok(t) = weights::cuda_tensor_shaped(map, "proj_out.weight", &[in_f, dim]) {
                s.out_w = t;
            }
        }
        Ok(s)
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
            return Err(msg(format!(
                "sd35 in_channels {} vs {}",
                c, self.cfg.in_channels
            )));
        }
        let p = self.cfg.patch_size;
        if h % p != 0 || w % p != 0 {
            return Err(msg(format!(
                "sd35 spatial {h}x{w} not divisible by patch {p}"
            )));
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
