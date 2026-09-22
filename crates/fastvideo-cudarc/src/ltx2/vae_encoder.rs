//! LTX-2 video VAE **encoder** path (Diffusers `encoder.*` keys).
//!
//! Full causal ResNet/downsample graph matches [`super::vae::VideoDecoder`]
//! complexity. This module loads `encoder.conv_in` (+ stats) when present and
//! runs patchify → optional conv center-tap → latent channel fold with
//! Diffusers normalize (`latents_mean` / `latents_std`). Missing encoder keys
//! → `try_load` returns `None` so I2V can fall back to the spatial stub.

use std::path::Path;

use fastvideo_models::ltx2::config::Ltx2VideoVaeConfig;
use image::RgbImage;

use crate::hub_keys::{self, ltx2_vae_encoder as ekeys};
use crate::wan::pipeline::{PipelineError, Result};
use crate::wan::tensor::CudaTensor;
use crate::wan::weights::{cuda_tensor_shaped, WeightMap};

fn msg(s: impl Into<String>) -> PipelineError {
    PipelineError::Message(s.into())
}

/// Minimal encoder: patchify RGB still → `encoder.conv_in` → channel project.
pub struct VideoEncoder {
    pub cfg: Ltx2VideoVaeConfig,
    pub loaded_key: String,
    /// Optional learned conv weight `[cout, cin, 3, 3, 3]` (host).
    conv_in_w: Option<Vec<f32>>,
    conv_in_shape: Option<[usize; 5]>,
    latents_mean: Vec<f32>,
    latents_std: Vec<f32>,
}

impl VideoEncoder {
    pub fn try_load(map: &WeightMap, cfg: &Ltx2VideoVaeConfig) -> Result<Option<Self>> {
        let Some(hit) = hub_keys::first_present(map, ekeys::PROBES) else {
            return Ok(None);
        };
        let z = cfg.latent_channels;
        let mean = if map.contains("latents_mean") {
            cuda_tensor_shaped(map, "latents_mean", &[z])?
                .host_cow()?
                .to_vec()
        } else {
            vec![0f32; z]
        };
        let std = if map.contains("latents_std") {
            cuda_tensor_shaped(map, "latents_std", &[z])?
                .host_cow()?
                .to_vec()
        } else {
            vec![1f32; z]
        };
        let mut conv_in_w = None;
        let mut conv_in_shape = None;
        let cin = 3 * cfg.patch_size * cfg.patch_size;
        let cout = cfg.block_out_channels.first().copied().unwrap_or(128);
        for key in ["encoder.conv_in.conv.weight", "encoder.conv_in.weight"] {
            if map.contains(key) {
                if let Ok(t) = cuda_tensor_shaped(map, key, &[cout, cin, 3, 3, 3]) {
                    conv_in_w = Some(t.host_cow()?.to_vec());
                    conv_in_shape = Some([cout, cin, 3, 3, 3]);
                    break;
                }
            }
        }
        Ok(Some(Self {
            cfg: cfg.clone(),
            loaded_key: hit,
            conv_in_w,
            conv_in_shape,
            latents_mean: mean,
            latents_std: std,
        }))
    }

    pub fn load(map: &WeightMap, cfg: &Ltx2VideoVaeConfig) -> Result<Self> {
        Self::try_load(map, cfg)?.ok_or_else(|| {
            msg(hub_keys::require_any(map, "ltx2_vae_encoder", ekeys::PROBES).unwrap_err())
        })
    }

    /// Encode a first-frame RGB still into `[1, C, 1, H, W]` DiT-normalized latents.
    pub fn encode_first_frame(
        &self,
        path: &Path,
        pixel_height: usize,
        pixel_width: usize,
    ) -> Result<CudaTensor> {
        let img = image::open(path)
            .map_err(|e| msg(format!("ltx2 encode open {}: {e}", path.display())))?
            .into_rgb8();
        let img = image::imageops::resize(
            &img,
            pixel_width as u32,
            pixel_height as u32,
            image::imageops::FilterType::Lanczos3,
        );
        self.encode_rgb8(&img, pixel_height, pixel_width)
    }

    pub fn encode_rgb8(
        &self,
        img: &RgbImage,
        pixel_height: usize,
        pixel_width: usize,
    ) -> Result<CudaTensor> {
        let spat = 32usize; // LTX-2 spatial compression
        let lh = pixel_height / spat;
        let lw = pixel_width / spat;
        if lh == 0 || lw == 0 {
            return Err(msg(format!(
                "ltx2 encode: {pixel_height}x{pixel_width} too small"
            )));
        }
        let z = self.cfg.latent_channels;
        let p = self.cfg.patch_size.max(1);
        let ph = (pixel_height / p).max(1);
        let pw = (pixel_width / p).max(1);
        let cin = 3 * p * p;
        let mut patched = vec![0f32; cin * ph * pw];
        for y in 0..ph {
            for x in 0..pw {
                for dy in 0..p {
                    for dx in 0..p {
                        let py = (y * p + dy).min(pixel_height - 1);
                        let px = (x * p + dx).min(pixel_width - 1);
                        let pix = img.get_pixel(px as u32, py as u32);
                        for ch in 0..3 {
                            let v = f32::from(pix[ch]) / 127.5 - 1.0;
                            let cidx = (ch * p + dy) * p + dx;
                            patched[(cidx * ph + y) * pw + x] = v;
                        }
                    }
                }
            }
        }
        let mut feat = patched;
        let mut feat_c = cin;
        let fh = ph;
        let fw = pw;
        if let (Some(w), Some(shape)) = (&self.conv_in_w, self.conv_in_shape) {
            let [cout, cin_w, _, _, _] = shape;
            if cin_w == cin {
                let mut out = vec![0f32; cout * ph * pw];
                for oc in 0..cout {
                    for y in 0..ph {
                        for x in 0..pw {
                            let mut acc = 0f32;
                            for ic in 0..cin {
                                let widx = (((oc * cin + ic) * 3 + 1) * 3 + 1) * 3 + 1;
                                acc += w[widx] * feat[(ic * ph + y) * pw + x];
                            }
                            out[(oc * ph + y) * pw + x] = acc;
                        }
                    }
                }
                feat = out;
                feat_c = cout;
            }
        }
        let mut lat = vec![0f32; z * lh * lw];
        for y in 0..lh {
            for x in 0..lw {
                let y0 = y * fh / lh;
                let x0 = x * fw / lw;
                for ch in 0..z {
                    let src_c = ch % feat_c;
                    let v = feat[(src_c * fh + y0.min(fh - 1)) * fw + x0.min(fw - 1)];
                    let mean = self.latents_mean.get(ch).copied().unwrap_or(0.0);
                    let std = self.latents_std.get(ch).copied().unwrap_or(1.0).max(1e-6);
                    lat[(ch * lh + y) * lw + x] =
                        (v - mean) / std * self.cfg.scaling_factor as f32;
                }
            }
        }
        CudaTensor::from_vec(lat, vec![1, z, 1, lh, lw]).map_err(Into::into)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{Rgb, RgbImage};

    #[test]
    fn probes_documented() {
        assert!(ekeys::PROBES.iter().any(|k| k.contains("encoder")));
    }

    #[test]
    fn encode_rgb_without_conv_weights() {
        let cfg = Ltx2VideoVaeConfig::ltx2_19b();
        let enc = VideoEncoder {
            cfg: cfg.clone(),
            loaded_key: "encoder.conv_in.conv.weight".into(),
            conv_in_w: None,
            conv_in_shape: None,
            latents_mean: vec![0f32; cfg.latent_channels],
            latents_std: vec![1f32; cfg.latent_channels],
        };
        let mut img = RgbImage::new(64, 64);
        for p in img.pixels_mut() {
            *p = Rgb([10, 20, 30]);
        }
        let lat = enc.encode_rgb8(&img, 64, 64).unwrap();
        assert_eq!(lat.shape, vec![1, cfg.latent_channels, 1, 2, 2]);
    }
}
