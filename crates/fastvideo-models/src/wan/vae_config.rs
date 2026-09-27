//! Wan VAE architecture constants (host-only; device graph lives in cudarc).
//!
//! Two variants of Diffusers `AutoencoderKLWan`:
//!
//! * Wan 2.1 (`is_residual = false`): 16 latent channels, 8× spatial, 4×
//!   temporal, base dim 96 in both halves.
//! * Wan 2.2 TI2V-5B (`is_residual = true`): 48 latent channels, 16× spatial
//!   (the 8× network between a 2× `patchify` / `unpatchify`, 12 channels in
//!   and out), residual down / up blocks (`AvgDown3D` / `DupUp3D` shortcuts),
//!   encoder base dim 160 and decoder base dim 256.

use std::path::Path;

/// Wan 2.1 `latents_mean` (Diffusers `AutoencoderKLWan` default).
pub const WAN21_LATENTS_MEAN: [f32; 16] = [
    -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508, 0.4134, -0.0715, 0.5517,
    -0.3632, -0.1922, -0.9497, 0.2503, -0.2921,
];
/// Wan 2.1 `latents_std`.
pub const WAN21_LATENTS_STD: [f32; 16] = [
    2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743, 3.2687, 2.1526, 2.8652, 1.5579,
    1.6382, 1.1253, 2.8251, 1.9160,
];
/// Wan 2.2 TI2V-5B `vae/config.json` `latents_mean` (Wan-AI/Wan2.2-TI2V-5B-Diffusers).
pub const WAN22_LATENTS_MEAN: [f32; 48] = [
    -0.2289, -0.0052, -0.1323, -0.2339, -0.2799, 0.0174, 0.1838, 0.1557, -0.1382, 0.0542, 0.2813,
    0.0891, 0.157, -0.0098, 0.0375, -0.1825, -0.2246, -0.1207, -0.0698, 0.5109, 0.2665, -0.2108,
    -0.2158, 0.2502, -0.2055, -0.0322, 0.1109, 0.1567, -0.0729, 0.0899, -0.2799, -0.123, -0.0313,
    -0.1649, 0.0117, 0.0723, -0.2839, -0.2083, -0.052, 0.3748, 0.0152, 0.1957, 0.1433, -0.2944,
    0.3573, -0.0548, -0.1681, -0.0667,
];
/// Wan 2.2 TI2V-5B `vae/config.json` `latents_std`.
pub const WAN22_LATENTS_STD: [f32; 48] = [
    0.4765, 1.0364, 0.4514, 1.1677, 0.5313, 0.499, 0.4818, 0.5013, 0.8158, 1.0344, 0.5894, 1.0901,
    0.6885, 0.6165, 0.8454, 0.4978, 0.5759, 0.3523, 0.7135, 0.6804, 0.5833, 1.4146, 0.8986, 0.5659,
    0.7069, 0.5338, 0.4889, 0.4917, 0.4069, 0.4999, 0.6866, 0.4093, 0.5709, 0.6065, 0.6415, 0.4944,
    0.5726, 1.2042, 0.5458, 1.6887, 0.3971, 1.06, 0.3943, 0.5537, 0.5444, 0.4089, 0.7468, 0.7744,
];

#[derive(Debug, Clone)]
pub struct WanVaeConfig {
    /// Encoder base width (`base_dim`).
    pub base_dim: usize,
    pub z_dim: usize,
    pub dim_mult: Vec<usize>,
    pub num_res_blocks: usize,
    /// Decoder temporal upsampling per up block: `temperal_downsample[::-1]`.
    pub temporal_upsample: Vec<bool>,
    pub load_encoder: bool,
    /// Decoder base width (`decoder_base_dim`, `None` in the config = `base_dim`).
    pub decoder_base_dim: usize,
    /// Wan 2.2 residual down / up blocks (`is_residual`).
    pub is_residual: bool,
    /// `patch_size` (1 = no patchify; Wan 2.2: 2).
    pub patch_size: usize,
    /// Per-channel `latents_mean` / `latents_std` (length `z_dim`).
    pub latents_mean: Vec<f32>,
    pub latents_std: Vec<f32>,
}

impl WanVaeConfig {
    pub fn wan_2_1() -> Self {
        Self {
            base_dim: 96,
            z_dim: 16,
            dim_mult: vec![1, 2, 4, 4],
            num_res_blocks: 2,
            temporal_upsample: vec![true, true, false],
            load_encoder: true,
            decoder_base_dim: 96,
            is_residual: false,
            patch_size: 1,
            latents_mean: WAN21_LATENTS_MEAN.to_vec(),
            latents_std: WAN21_LATENTS_STD.to_vec(),
        }
    }

    /// Wan 2.2 TI2V-5B (`Wan-AI/Wan2.2-TI2V-5B-Diffusers` `vae/config.json`).
    pub fn wan_2_2() -> Self {
        Self {
            base_dim: 160,
            z_dim: 48,
            dim_mult: vec![1, 2, 4, 4],
            num_res_blocks: 2,
            temporal_upsample: vec![true, true, false],
            load_encoder: true,
            decoder_base_dim: 256,
            is_residual: true,
            patch_size: 2,
            latents_mean: WAN22_LATENTS_MEAN.to_vec(),
            latents_std: WAN22_LATENTS_STD.to_vec(),
        }
    }

    pub fn tiny() -> Self {
        Self {
            base_dim: 8,
            z_dim: 4,
            dim_mult: vec![1, 2],
            num_res_blocks: 1,
            temporal_upsample: vec![false],
            load_encoder: false,
            decoder_base_dim: 8,
            is_residual: false,
            patch_size: 1,
            latents_mean: vec![0.0; 4],
            latents_std: vec![1.0; 4],
        }
    }

    /// Pixels per latent cell: 8 through the network, times the patchify.
    pub fn spatial_compression(&self) -> usize {
        let downs = self.dim_mult.len().saturating_sub(1);
        (1usize << downs) * self.patch_size.max(1)
    }

    /// Frames per latent frame (after the first): 2 per temporal stage.
    pub fn temporal_compression(&self) -> usize {
        1usize << self.temporal_upsample.iter().filter(|&&t| t).count()
    }

    /// Channels the encoder reads and the decoder writes: RGB times the
    /// patchify's `p²` (Wan 2.2: 12).
    pub fn io_channels(&self) -> usize {
        3 * self.patch_size.max(1) * self.patch_size.max(1)
    }

    /// Parse Diffusers `vae/config.json`. Unknown keys are ignored; missing
    /// ones take the Diffusers `AutoencoderKLWan` defaults (the Wan 2.1 VAE).
    pub fn from_json_str(text: &str) -> Result<Self, String> {
        let v: serde_json::Value =
            serde_json::from_str(text).map_err(|e| format!("vae config.json: {e}"))?;
        let usize_of = |k: &str, d: usize| -> Result<usize, String> {
            match v.get(k) {
                None | Some(serde_json::Value::Null) => Ok(d),
                Some(x) => x
                    .as_u64()
                    .map(|n| n as usize)
                    .ok_or_else(|| format!("vae config.json: {k} is not an integer")),
            }
        };
        let floats = |k: &str, d: &[f32]| -> Result<Vec<f32>, String> {
            match v.get(k) {
                None | Some(serde_json::Value::Null) => Ok(d.to_vec()),
                Some(serde_json::Value::Array(a)) => a
                    .iter()
                    .map(|x| {
                        x.as_f64()
                            .map(|f| f as f32)
                            .ok_or_else(|| format!("vae config.json: {k} holds a non-number"))
                    })
                    .collect(),
                Some(_) => Err(format!("vae config.json: {k} is not a list")),
            }
        };
        let base_dim = usize_of("base_dim", 96)?;
        let z_dim = usize_of("z_dim", 16)?;
        let dim_mult: Vec<usize> = match v.get("dim_mult") {
            Some(serde_json::Value::Array(a)) => a
                .iter()
                .map(|x| x.as_u64().map(|n| n as usize))
                .collect::<Option<_>>()
                .ok_or("vae config.json: dim_mult holds a non-integer")?,
            _ => vec![1, 2, 4, 4],
        };
        let down: Vec<bool> = match v.get("temperal_downsample") {
            Some(serde_json::Value::Array(a)) => a
                .iter()
                .map(|x| x.as_bool())
                .collect::<Option<_>>()
                .ok_or("vae config.json: temperal_downsample holds a non-bool")?,
            _ => vec![false, true, true],
        };
        let patch_size = usize_of("patch_size", 1)?.max(1);
        let in_channels = usize_of("in_channels", 3)?;
        let out_channels = usize_of("out_channels", 3)?;
        let io = 3 * patch_size * patch_size;
        if in_channels != io || out_channels != io {
            return Err(format!(
                "vae config.json: in/out channels {in_channels}/{out_channels} with patch {patch_size} (expected {io})"
            ));
        }
        let (dm, ds) = if z_dim == 48 {
            (&WAN22_LATENTS_MEAN[..], &WAN22_LATENTS_STD[..])
        } else {
            (&WAN21_LATENTS_MEAN[..], &WAN21_LATENTS_STD[..])
        };
        let latents_mean = floats("latents_mean", dm)?;
        let latents_std = floats("latents_std", ds)?;
        if latents_mean.len() != z_dim || latents_std.len() != z_dim {
            return Err(format!(
                "vae config.json: latents_mean/std have {}/{} entries for z_dim {z_dim}",
                latents_mean.len(),
                latents_std.len()
            ));
        }
        Ok(Self {
            base_dim,
            z_dim,
            temporal_upsample: down.iter().rev().copied().collect(),
            dim_mult,
            num_res_blocks: usize_of("num_res_blocks", 2)?,
            load_encoder: true,
            decoder_base_dim: usize_of("decoder_base_dim", base_dim)?,
            is_residual: v
                .get("is_residual")
                .and_then(serde_json::Value::as_bool)
                .unwrap_or(false),
            patch_size,
            latents_mean,
            latents_std,
        })
    }

    /// [`Self::from_json_str`] on `<vae dir>/config.json`.
    pub fn from_dir(dir: &Path) -> Result<Self, String> {
        let path = dir.join("config.json");
        let text =
            std::fs::read_to_string(&path).map_err(|e| format!("{}: {e}", path.display()))?;
        Self::from_json_str(&text)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `Wan-AI/Wan2.2-TI2V-5B-Diffusers` `vae/config.json` (trimmed of the
    /// stats, which then come from the built-in table).
    const WAN22_JSON: &str = r#"{
      "_class_name": "AutoencoderKLWan", "attn_scales": [], "base_dim": 160,
      "clip_output": false, "decoder_base_dim": 256, "dim_mult": [1, 2, 4, 4],
      "dropout": 0.0, "in_channels": 12, "is_residual": true, "num_res_blocks": 2,
      "out_channels": 12, "patch_size": 2, "scale_factor_spatial": 16,
      "scale_factor_temporal": 4, "temperal_downsample": [false, true, true], "z_dim": 48
    }"#;

    #[test]
    fn wan22_geometry() {
        let cfg = WanVaeConfig::from_json_str(WAN22_JSON).unwrap();
        assert_eq!(cfg.z_dim, 48);
        assert_eq!(cfg.decoder_base_dim, 256);
        assert!(cfg.is_residual);
        assert_eq!(cfg.patch_size, 2);
        assert_eq!(cfg.spatial_compression(), 16);
        assert_eq!(cfg.temporal_compression(), 4);
        assert_eq!(cfg.io_channels(), 12);
        assert_eq!(cfg.temporal_upsample, vec![true, true, false]);
        assert_eq!(cfg.latents_mean.len(), 48);
        assert_eq!(cfg.latents_std[0], 0.4765);
        let built = WanVaeConfig::wan_2_2();
        assert_eq!(built.latents_mean, cfg.latents_mean);
        assert_eq!(built.base_dim, cfg.base_dim);
    }

    #[test]
    fn wan21_defaults() {
        let cfg = WanVaeConfig::from_json_str(r#"{"z_dim": 16, "base_dim": 96}"#).unwrap();
        assert_eq!(cfg.spatial_compression(), 8);
        assert_eq!(cfg.temporal_compression(), 4);
        assert!(!cfg.is_residual);
        assert_eq!(cfg.decoder_base_dim, 96);
        assert_eq!(cfg.latents_mean, WAN21_LATENTS_MEAN.to_vec());
        assert!(WanVaeConfig::from_json_str(r#"{"z_dim": 48}"#).is_ok());
        assert!(WanVaeConfig::from_json_str(r#"{"z_dim": 48, "latents_mean": [0.0]}"#).is_err());
    }
}
