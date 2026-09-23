//! LTX-2 I2V first-frame VAE encode.
//!
//! Prefers [`super::vae_encoder::VideoEncoder`] when Diffusers `encoder.*`
//! keys are present under `vae/` (full ResNet/downsample stack when
//! `down_blocks` load). Otherwise uses a spatial downsample stub so
//! `--image` still conditions frame 0 instead of refusing.

use std::path::Path;

use fastvideo_models::ltx2::config::Ltx2VideoVaeConfig;

use crate::wan::pipeline::{PipelineError, Result};
use crate::wan::tensor::CudaTensor;
use crate::wan::weights::WeightMap;

use super::vae_encoder::VideoEncoder;

fn msg(s: impl Into<String>) -> PipelineError {
    PipelineError::Message(s.into())
}

/// Encode a first-frame image into a single latent frame `[1, C, 1, H, W]`.
///
/// When `vae_dir` contains encoder keys, runs the real (partial) Diffusers
/// encoder path; otherwise falls back to spatial stub.
pub fn encode_first_frame(
    path: &Path,
    pixel_height: usize,
    pixel_width: usize,
    latent_channels: usize,
    spatial_compression: usize,
    vae_dir: Option<&Path>,
) -> Result<CudaTensor> {
    if let Some(dir) = vae_dir {
        if dir.is_dir() {
            let map = WeightMap::open(dir).map_err(|e| msg(e.to_string()))?;
            let mut cfg = Ltx2VideoVaeConfig::ltx2_19b();
            cfg.latent_channels = latent_channels;
            if let Some(enc) = VideoEncoder::try_load(&map, &cfg)? {
                return enc.encode_first_frame(path, pixel_height, pixel_width);
            }
        }
    }
    encode_first_frame_stub(
        path,
        pixel_height,
        pixel_width,
        latent_channels,
        spatial_compression,
    )
}

/// Spatial downsample stub (no `encoder.*` weights).
pub fn encode_first_frame_stub(
    path: &Path,
    pixel_height: usize,
    pixel_width: usize,
    latent_channels: usize,
    spatial_compression: usize,
) -> Result<CudaTensor> {
    let img = image::open(path)
        .map_err(|e| msg(format!("ltx2 i2v open {}: {e}", path.display())))?
        .into_rgb8();
    let img = image::imageops::resize(
        &img,
        pixel_width as u32,
        pixel_height as u32,
        image::imageops::FilterType::Lanczos3,
    );
    let lh = pixel_height / spatial_compression;
    let lw = pixel_width / spatial_compression;
    if lh == 0 || lw == 0 {
        return Err(msg(format!(
            "ltx2 i2v: {pixel_height}x{pixel_width} too small for compression {spatial_compression}"
        )));
    }
    let mut lat = vec![0f32; latent_channels * lh * lw];
    for y in 0..lh {
        for x in 0..lw {
            let mut acc = [0f32; 3];
            let y0 = y * spatial_compression;
            let x0 = x * spatial_compression;
            let mut n = 0usize;
            for dy in 0..spatial_compression {
                for dx in 0..spatial_compression {
                    let py = (y0 + dy).min(pixel_height - 1);
                    let px = (x0 + dx).min(pixel_width - 1);
                    let p = img.get_pixel(px as u32, py as u32);
                    for ch in 0..3 {
                        acc[ch] += f32::from(p[ch]) / 127.5 - 1.0;
                    }
                    n += 1;
                }
            }
            for ch in 0..3 {
                acc[ch] /= n as f32;
            }
            for ch in 0..latent_channels {
                lat[(ch * lh + y) * lw + x] = acc[ch % 3] * 0.5;
            }
        }
    }
    CudaTensor::from_vec(lat, vec![1, latent_channels, 1, lh, lw]).map_err(Into::into)
}

/// Overwrite frame 0 of packed video tokens with the encoded still.
pub fn condition_first_frame(
    packed_video: &CudaTensor,
    grid: [usize; 3],
    first_frame: &CudaTensor,
) -> Result<CudaTensor> {
    use super::transformer::{pack_video, unpack_video};

    let [f, h, w] = grid;
    let video = unpack_video(packed_video, grid).map_err(|e| msg(e.to_string()))?;
    let [_, c, ef, eh, ew] = match first_frame.shape[..] {
        [1, c, ef, eh, ew] => [1, c, ef, eh, ew],
        _ => {
            return Err(msg(format!(
                "ltx2 i2v cond shape {:?} want [1,C,1,H,W]",
                first_frame.shape
            )))
        }
    };
    if ef != 1 || eh != h || ew != w {
        return Err(msg(format!(
            "ltx2 i2v cond grid [{ef},{eh},{ew}] vs want [1,{h},{w}]"
        )));
    }
    if c != video.shape[1] {
        return Err(msg(format!(
            "ltx2 i2v channels {c} vs video {}",
            video.shape[1]
        )));
    }
    let mut host = video.host_cow()?.to_vec();
    let cond = first_frame.host_cow()?;
    for ch in 0..c {
        for y in 0..h {
            for x in 0..w {
                let dst = ((ch * f) * h + y) * w + x;
                let src = (ch * h + y) * w + x;
                host[dst] = cond[src];
            }
        }
    }
    let video = CudaTensor::from_vec(host, vec![1, c, f, h, w])?;
    pack_video(&video).map_err(|e| msg(e.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{Rgb, RgbImage};

    #[test]
    fn stub_encode_shape() {
        let dir = std::env::temp_dir().join("ltx2-i2v-stub.png");
        let mut img = RgbImage::new(64, 64);
        for p in img.pixels_mut() {
            *p = Rgb([40, 80, 120]);
        }
        img.save(&dir).unwrap();
        let lat = encode_first_frame_stub(&dir, 64, 64, 128, 32).unwrap();
        assert_eq!(lat.shape, vec![1, 128, 1, 2, 2]);
        let _ = std::fs::remove_file(&dir);
    }
}
