//! Minimal Z-Image 2D DiT graph (tiny zeros + Diffusers key probes).
//!
//! Full 30-layer noise/context refiners land with weight parity; this module
//! provides a shape-correct forward for host unit tests and generate scaffold.

use fastvideo_models::zimage::ZImageTransformerConfig;

use crate::hub_keys::{self, zimage as zkeys};
use crate::wan::pipeline::{PipelineError, Result};
use crate::wan::tensor::CudaTensor;
use crate::wan::weights::{self, WeightMap};

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
    /// First Diffusers probe key that matched (if any).
    pub loaded_key: Option<String>,
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
            loaded_key: None,
        })
    }

    /// Load against a Diffusers `transformer/` map.
    ///
    /// Expected probes (any one required for full packs): see
    /// [`crate::hub_keys::zimage::PROBES`]. Tiny configs may fall back to zeros
    /// when probes miss; full packs error.
    pub fn load(cfg: ZImageTransformerConfig, map: &WeightMap) -> Result<Self> {
        let hit = hub_keys::first_present(map, zkeys::PROBES);
        let tiny = cfg.n_layers <= 2 || cfg.dim < 1000;
        if hit.is_none() && !tiny {
            return Err(msg(
                hub_keys::require_any(map, "zimage", zkeys::PROBES).unwrap_err()
            ));
        }
        let mut s = Self::zeros(cfg.clone())?;
        s.loaded_key = hit;
        // Best-effort: load final / patch linears when shapes match Diffusers.
        let p = cfg.patch_size;
        let in_f = cfg.in_channels * p * p;
        for key in [
            "final_layer.linear.weight",
            "proj_out.weight",
            "all_x_embedder.weight",
            "x_embedder.weight",
        ] {
            if map.contains(key) {
                if let Ok(t) = weights::cuda_tensor_shaped(map, key, &[in_f, cfg.dim]) {
                    if key.contains("embed") {
                        s.patch_w = t;
                    } else {
                        // [out, in] vs [dim, in_f]
                        if let Ok(t2) = weights::cuda_tensor_shaped(map, key, &[cfg.dim, in_f]) {
                            s.out_w = t2;
                        } else {
                            let _ = t;
                        }
                    }
                } else if let Ok(t) = weights::cuda_tensor_shaped(map, key, &[cfg.dim, in_f]) {
                    s.out_w = t;
                }
            }
        }
        Ok(s)
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
            return Err(msg(format!("zimage want [B,C,H,W], got {shape:?}")));
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
        let scale = (timestep / self.cfg.t_scale).clamp(0.0, 1.0);
        let data = latents.host_cow()?;
        let mut out = data.to_vec();
        for v in &mut out {
            *v *= 1.0 - 0.1 * scale;
        }
        let _ = (
            &self.patch_w,
            &self.patch_b,
            &self.out_w,
            &self.out_b,
            &self.time_w,
            b,
        );
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
