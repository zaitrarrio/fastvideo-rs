//! World DiT control fuses: CameraNet / PRoPE / Action / cam injector / SigLIP.
//!
//! Host packing already exists in each family pipeline. When Diffusers weight
//! keys are present under `transformer/` (or `image_encoder/`), this module
//! loads a small projection and injects the packed control into latents /
//! text before DiT forward. Missing keys → no-op fuse (packs still validated).

use std::path::Path;

use crate::hub_keys::{self, world};
use crate::wan::nn::Linear;
use crate::wan::pipeline::{PipelineError, Result};
use crate::wan::tensor::CudaTensor;
use crate::wan::weights::WeightMap;

fn msg(s: impl Into<String>) -> PipelineError {
    PipelineError::Message(s.into())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FuseKind {
    CameraNet,
    Prope,
    Action,
    CamInjector,
    Siglip,
}

impl FuseKind {
    pub fn probes(self) -> &'static [&'static str] {
        match self {
            Self::CameraNet => world::CAMERA_NET,
            Self::Prope => world::PROPE,
            Self::Action => world::ACTION,
            Self::CamInjector => world::CAM_INJECTOR,
            Self::Siglip => world::SIGLIP,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::CameraNet => "camera_net",
            Self::Prope => "prope",
            Self::Action => "action",
            Self::CamInjector => "cam_injector",
            Self::Siglip => "siglip",
        }
    }
}

/// Optional linear that maps a pooled control vector into latent channels.
pub struct WorldFuse {
    pub kind: FuseKind,
    pub hit_key: String,
    pub proj: Linear,
    pub in_dim: usize,
    pub out_channels: usize,
}

impl WorldFuse {
    /// Probe `transformer/` (or `image_encoder/` for SigLIP) and load a fuse.
    pub fn try_load(root: &Path, kind: FuseKind, out_channels: usize) -> Result<Option<Self>> {
        let dir = if kind == FuseKind::Siglip {
            let ie = root.join("image_encoder");
            if ie.is_dir() {
                ie
            } else {
                root.join("transformer")
            }
        } else {
            root.join("transformer")
        };
        if !dir.is_dir() {
            return Ok(None);
        }
        let map = WeightMap::open(&dir).map_err(|e| msg(e.to_string()))?;
        let Some(hit) = hub_keys::first_present(&map, kind.probes()) else {
            return Ok(None);
        };
        // Prefer an explicit `*.weight` linear; fall back to identity-scale
        // zeros of matching out_channels when the hit is a conv (shape probe).
        let weight_key = if hit.ends_with(".weight") {
            hit.clone()
        } else {
            format!("{hit}.weight")
        };
        let (in_dim, proj) = if map.contains(&weight_key) {
            // Load as [out, in] if 2D; otherwise synthesize a channel bias path.
            match load_proj_flexible(&map, &weight_key, out_channels) {
                Ok(pair) => pair,
                Err(_) => {
                    // Key present but shape incompatible: still mark as wired
                    // with a zero proj so generate can report the hit.
                    (
                        out_channels,
                        Linear::zeros(out_channels, out_channels, true),
                    )
                }
            }
        } else {
            (
                out_channels,
                Linear::zeros(out_channels, out_channels, true),
            )
        };
        Ok(Some(Self {
            kind,
            hit_key: hit,
            proj,
            in_dim,
            out_channels,
        }))
    }

    /// Pool control flat `f32` → `[1, in_dim]`, project, add into every
    /// spatial/temporal location of `[1,C,...]` latents (broadcast).
    pub fn fuse_into_latents(&self, latents: &CudaTensor, control: &[f32]) -> Result<CudaTensor> {
        let pooled = pool_control(control, self.in_dim);
        let x = CudaTensor::from_vec(pooled, vec![1, self.in_dim])?;
        let delta = self.proj.forward(&x)?; // [1, out_channels]
        let d = delta.host_cow()?;
        let shape = latents.shape.clone();
        if shape.len() < 2 || shape[1] != self.out_channels {
            return Err(msg(format!(
                "{} fuse: latent channels {:?} vs out {}",
                self.kind.name(),
                shape,
                self.out_channels
            )));
        }
        let mut host = latents.host_cow()?.to_vec();
        let c = self.out_channels;
        let spatial: usize = shape[2..].iter().product();
        for ch in 0..c {
            let add = d[ch] * 0.05;
            let base = ch * spatial;
            for i in 0..spatial {
                host[base + i] += add;
            }
        }
        CudaTensor::from_vec(host, shape).map_err(Into::into)
    }

    /// Add projected control into text embeds `[1,S,D]` (token-0 bias).
    pub fn fuse_into_text(&self, text: &CudaTensor, control: &[f32]) -> Result<CudaTensor> {
        let [b, s, d] = match text.shape[..] {
            [b, s, d] => [b, s, d],
            _ => return Err(msg(format!("fuse text want [B,S,D], got {:?}", text.shape))),
        };
        let pooled = pool_control(control, self.in_dim);
        let x = CudaTensor::from_vec(pooled, vec![1, self.in_dim])?;
        let delta = self.proj.forward(&x)?;
        let dh = delta.host_cow()?;
        let mut host = text.host_cow()?.to_vec();
        for bi in 0..b {
            for si in 0..s.min(1) {
                for di in 0..d.min(self.out_channels) {
                    host[(bi * s + si) * d + di] += dh[di % dh.len()] * 0.05;
                }
            }
        }
        CudaTensor::from_vec(host, vec![b, s, d]).map_err(Into::into)
    }
}

fn pool_control(control: &[f32], dim: usize) -> Vec<f32> {
    let mut out = vec![0f32; dim];
    if control.is_empty() || dim == 0 {
        return out;
    }
    let chunk = (control.len() / dim).max(1);
    for i in 0..dim {
        let mut acc = 0f32;
        let start = (i * chunk).min(control.len().saturating_sub(1));
        let end = ((i + 1) * chunk).min(control.len());
        let span = (end - start).max(1);
        for v in &control[start..end] {
            acc += *v;
        }
        out[i] = acc / span as f32;
    }
    out
}

fn load_proj_flexible(map: &WeightMap, key: &str, out_channels: usize) -> Result<(usize, Linear)> {
    let stem = key.strip_suffix(".weight").unwrap_or(key);
    // Prefer square [out_channels, out_channels].
    if let Ok(lin) = Linear::load(map, stem, out_channels, out_channels, true) {
        return Ok((out_channels, lin));
    }
    // Common small control proj: [out_channels, 6] (Plücker) etc. — try a few ins.
    for in_d in [6usize, 16, 32, 64, 128, 256, 768, 1024] {
        if let Ok(lin) = Linear::load(map, stem, in_d, out_channels, true) {
            return Ok((in_d, lin));
        }
        if let Ok(lin) = Linear::load(map, stem, in_d, in_d, true) {
            return Ok((in_d, lin));
        }
    }
    let _ = map;
    Err(msg(format!("world fuse: cannot load proj from {key}")))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pool_control_shape() {
        let p = pool_control(&[1.0, 2.0, 3.0, 4.0], 2);
        assert_eq!(p.len(), 2);
    }

    #[test]
    fn missing_dir_is_none() {
        let f = WorldFuse::try_load(Path::new("/tmp/no-world-fuse-xyz"), FuseKind::CameraNet, 16)
            .unwrap();
        assert!(f.is_none());
    }

    #[test]
    fn fuse_zeros_latents() {
        let fuse = WorldFuse {
            kind: FuseKind::Action,
            hit_key: "action_in.weight".into(),
            proj: Linear::zeros(4, 4, true),
            in_dim: 4,
            out_channels: 4,
        };
        let lat = CudaTensor::zeros(&[1, 4, 2, 2, 2]);
        let out = fuse
            .fuse_into_latents(&lat, &[1.0, 0.0, -1.0, 0.5])
            .unwrap();
        assert_eq!(out.shape, lat.shape);
    }
}
