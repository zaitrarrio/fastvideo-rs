//! Qwen2/3-VL image preprocessing for MiniMax-H3: smart-resize, CLIP normalize,
//! and temporal-patch flatten matching HF `Qwen2VLImageProcessor` (patch 16 /
//! merge 2 for H3).

use super::config::H3VisionConfig;

/// OpenAI CLIP mean / std (HF `OPENAI_CLIP_*`).
pub const CLIP_MEAN: [f64; 3] = [0.481_454_66, 0.457_827_5, 0.408_210_73];
pub const CLIP_STD: [f64; 3] = [0.268_629_54, 0.261_302_58, 0.275_777_11];

/// Smart-resize so H/W are divisible by `factor` and area ∈ `[min, max]` pixels.
pub fn smart_resize(
    height: usize,
    width: usize,
    factor: usize,
    min_pixels: usize,
    max_pixels: usize,
) -> Result<(usize, usize), String> {
    if height == 0 || width == 0 || factor == 0 {
        return Err(format!(
            "smart_resize: bad size {height}x{width} factor {factor}"
        ));
    }
    if height.max(width) as f64 / height.min(width) as f64 > 200.0 {
        return Err(format!(
            "absolute aspect ratio must be smaller than 200, got {}x{}",
            width, height
        ));
    }
    let mut h_bar = ((height as f64 / factor as f64).round() as usize) * factor;
    let mut w_bar = ((width as f64 / factor as f64).round() as usize) * factor;
    if h_bar * w_bar > max_pixels {
        let beta = ((height * width) as f64 / max_pixels as f64).sqrt();
        h_bar = ((height as f64 / beta / factor as f64).floor() as usize * factor).max(factor);
        w_bar = ((width as f64 / beta / factor as f64).floor() as usize * factor).max(factor);
    } else if h_bar * w_bar < min_pixels {
        let beta = (min_pixels as f64 / (height * width) as f64).sqrt();
        h_bar = ((height as f64 * beta / factor as f64).ceil() as usize) * factor;
        w_bar = ((width as f64 * beta / factor as f64).ceil() as usize) * factor;
    }
    Ok((h_bar, w_bar))
}

/// One image's Qwen grid after preprocess: `grid = (1, H/patch, W/patch)`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VisionGrid {
    pub temporal: usize,
    pub height: usize,
    pub width: usize,
}

impl VisionGrid {
    pub fn num_patches(&self) -> usize {
        self.temporal * self.height * self.width
    }

    /// Pad tokens after spatial merge: `prod(grid) / merge²`.
    pub fn num_tokens(&self, merge: usize) -> Result<usize, String> {
        let area = merge * merge;
        if area == 0 || self.height % merge != 0 || self.width % merge != 0 {
            return Err(format!("grid {self:?} not divisible by merge {merge}"));
        }
        Ok(self.num_patches() / area)
    }
}

/// Processed pixels ready for the vision Conv3d patch embed.
///
/// Layout matches HF: `[num_patches, C * temporal_patch * patch * patch]` with
/// patches ordered in merge-block major order, and each spatial patch duplicated
/// across `temporal_patch_size` (images use T=1 then expand).
#[derive(Debug, Clone)]
pub struct PreparedVisionImage {
    pub pixels: Vec<f32>,
    pub grid: VisionGrid,
    pub patch_dim: usize,
}

/// Resize RGB u8 HWC → CLIP-normalized patch tokens for one image.
pub fn prepare_vision_image(
    rgb: &[u8],
    height: usize,
    width: usize,
    cfg: &H3VisionConfig,
) -> Result<PreparedVisionImage, String> {
    if rgb.len() != height * width * 3 {
        return Err(format!(
            "vision image: {} bytes for {height}x{width}",
            rgb.len()
        ));
    }
    let factor = cfg.patch_size * cfg.spatial_merge_size;
    let (out_h, out_w) = smart_resize(height, width, factor, cfg.min_pixels, cfg.max_pixels)?;
    let resized = resize_rgb_bilinear(rgb, height, width, out_h, out_w);
    let grid_h = out_h / cfg.patch_size;
    let grid_w = out_w / cfg.patch_size;
    let merge = cfg.spatial_merge_size;
    let patch_dim = 3 * cfg.temporal_patch_size * cfg.patch_size * cfg.patch_size;
    let mut pixels = Vec::with_capacity(grid_h * grid_w * patch_dim);

    // Normalize to CHW f32, then patchify like HF (merge-block order).
    let mut chw = vec![0f32; 3 * out_h * out_w];
    for y in 0..out_h {
        for x in 0..out_w {
            for c in 0..3 {
                let v01 = resized[(y * out_w + x) * 3 + c] as f32 / 255.0;
                chw[c * out_h * out_w + y * out_w + x] =
                    (v01 - CLIP_MEAN[c] as f32) / CLIP_STD[c] as f32;
            }
        }
    }

    let gh_m = grid_h / merge;
    let gw_m = grid_w / merge;
    let ps = cfg.patch_size;
    for bh in 0..gh_m {
        for bw in 0..gw_m {
            for mh in 0..merge {
                for mw in 0..merge {
                    let gh = bh * merge + mh;
                    let gw = bw * merge + mw;
                    // temporal_patch copies of the same spatial patch (image T=1).
                    for _t in 0..cfg.temporal_patch_size {
                        for c in 0..3 {
                            for py in 0..ps {
                                for px in 0..ps {
                                    let y = gh * ps + py;
                                    let x = gw * ps + px;
                                    pixels.push(chw[c * out_h * out_w + y * out_w + x]);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    let grid = VisionGrid {
        temporal: 1,
        height: grid_h,
        width: grid_w,
    };
    if pixels.len() != grid.num_patches() * patch_dim {
        return Err(format!(
            "vision patchify: {} values for grid {:?} dim {patch_dim}",
            pixels.len(),
            grid
        ));
    }
    Ok(PreparedVisionImage {
        pixels,
        grid,
        patch_dim,
    })
}

/// Prepare a video clip: `frames` is concatenated HWC RGB u8, `num_frames` at
/// native `height`×`width`. Frame count must be a multiple of
/// `temporal_patch_size` (pad by repeating the last frame before calling).
pub fn prepare_vision_video(
    frames: &[u8],
    num_frames: usize,
    height: usize,
    width: usize,
    cfg: &H3VisionConfig,
) -> Result<PreparedVisionImage, String> {
    let tt = cfg.temporal_patch_size;
    if num_frames == 0 || num_frames % tt != 0 {
        return Err(format!(
            "vision video: {num_frames} frames not a multiple of temporal_patch={tt}"
        ));
    }
    let frame_bytes = height * width * 3;
    if frames.len() != num_frames * frame_bytes {
        return Err(format!(
            "vision video: {} bytes for {num_frames}×{height}×{width}",
            frames.len()
        ));
    }
    let factor = cfg.patch_size * cfg.spatial_merge_size;
    let (out_h, out_w) = smart_resize(height, width, factor, cfg.min_pixels, cfg.max_pixels)?;
    let grid_h = out_h / cfg.patch_size;
    let grid_w = out_w / cfg.patch_size;
    let merge = cfg.spatial_merge_size;
    let patch_dim = 3 * tt * cfg.patch_size * cfg.patch_size;
    let temporal_groups = num_frames / tt;
    let mut pixels = Vec::with_capacity(temporal_groups * grid_h * grid_w * patch_dim);
    let ps = cfg.patch_size;
    let gh_m = grid_h / merge;
    let gw_m = grid_w / merge;

    // Resize every frame once.
    let mut resized: Vec<Vec<f32>> = Vec::with_capacity(num_frames);
    for f in 0..num_frames {
        let rgb = &frames[f * frame_bytes..(f + 1) * frame_bytes];
        let u8s = resize_rgb_bilinear(rgb, height, width, out_h, out_w);
        let mut chw = vec![0f32; 3 * out_h * out_w];
        for y in 0..out_h {
            for x in 0..out_w {
                for c in 0..3 {
                    let v01 = u8s[(y * out_w + x) * 3 + c] as f32 / 255.0;
                    chw[c * out_h * out_w + y * out_w + x] =
                        (v01 - CLIP_MEAN[c] as f32) / CLIP_STD[c] as f32;
                }
            }
        }
        resized.push(chw);
    }

    for tg in 0..temporal_groups {
        for bh in 0..gh_m {
            for bw in 0..gw_m {
                for mh in 0..merge {
                    for mw in 0..merge {
                        let gh = bh * merge + mh;
                        let gw = bw * merge + mw;
                        for t in 0..tt {
                            let chw = &resized[tg * tt + t];
                            for c in 0..3 {
                                for py in 0..ps {
                                    for px in 0..ps {
                                        let y = gh * ps + py;
                                        let x = gw * ps + px;
                                        pixels.push(chw[c * out_h * out_w + y * out_w + x]);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    let grid = VisionGrid {
        temporal: temporal_groups,
        height: grid_h,
        width: grid_w,
    };
    if pixels.len() != grid.num_patches() * patch_dim {
        return Err(format!(
            "vision video patchify: {} values for grid {:?} dim {patch_dim}",
            pixels.len(),
            grid
        ));
    }
    Ok(PreparedVisionImage {
        pixels,
        grid,
        patch_dim,
    })
}

fn resize_rgb_bilinear(rgb: &[u8], sh: usize, sw: usize, dh: usize, dw: usize) -> Vec<u8> {
    if sh == dh && sw == dw {
        return rgb.to_vec();
    }
    let mut out = vec![0u8; dh * dw * 3];
    let y_scale = (sh as f64) / (dh as f64);
    let x_scale = (sw as f64) / (dw as f64);
    for y in 0..dh {
        let sy = (y as f64 + 0.5) * y_scale - 0.5;
        let y0 = sy.floor().max(0.0) as usize;
        let y1 = (y0 + 1).min(sh - 1);
        let fy = (sy - y0 as f64).clamp(0.0, 1.0);
        for x in 0..dw {
            let sx = (x as f64 + 0.5) * x_scale - 0.5;
            let x0 = sx.floor().max(0.0) as usize;
            let x1 = (x0 + 1).min(sw - 1);
            let fx = (sx - x0 as f64).clamp(0.0, 1.0);
            for c in 0..3 {
                let p = |yy: usize, xx: usize| rgb[(yy * sw + xx) * 3 + c] as f64;
                let top = p(y0, x0) * (1.0 - fx) + p(y0, x1) * fx;
                let bot = p(y1, x0) * (1.0 - fx) + p(y1, x1) * fx;
                out[(y * dw + x) * 3 + c] =
                    (top * (1.0 - fy) + bot * fy).round().clamp(0.0, 255.0) as u8;
            }
        }
    }
    out
}

/// Sample 2-fps Qwen video presentation frames and block timestamps
/// (FastVideo `sample_reference_video_frames`).
pub fn sample_qwen_video_frames(
    num_frames_24fps: usize,
    sample_fps: f64,
    temporal_patch: usize,
) -> Result<(Vec<usize>, Vec<f64>), String> {
    if num_frames_24fps == 0 || sample_fps <= 0.0 || temporal_patch == 0 {
        return Err("qwen video sample: need frames, positive fps, temporal_patch".into());
    }
    let stride = 24.0_f64 / sample_fps;
    let mut indices = Vec::new();
    let mut cursor = 0.0_f64;
    while cursor.round() < num_frames_24fps as f64 {
        let idx = cursor.round() as usize;
        if indices.last().copied() != Some(idx) {
            indices.push(idx.min(num_frames_24fps - 1));
        }
        cursor += stride;
    }
    if indices.is_empty() {
        indices.push(0);
    }
    let mut timestamps: Vec<f64> = (0..indices.len()).map(|i| i as f64 / sample_fps).collect();
    while timestamps.len() % temporal_patch != 0 {
        timestamps.push(*timestamps.last().unwrap());
        indices.push(*indices.last().unwrap());
    }
    let block_timestamps: Vec<f64> = timestamps
        .chunks_exact(temporal_patch)
        .map(|c| (c[0] + c[temporal_patch - 1]) / 2.0)
        .collect();
    Ok((indices, block_timestamps))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn smart_resize_snaps_to_factor() {
        let cfg = H3VisionConfig::fasth3_8step();
        let factor = cfg.patch_size * cfg.spatial_merge_size;
        let (h, w) = smart_resize(480, 640, factor, cfg.min_pixels, cfg.max_pixels).unwrap();
        assert_eq!(h % factor, 0);
        assert_eq!(w % factor, 0);
    }

    #[test]
    fn prepare_square_image_token_count() {
        let cfg = H3VisionConfig::fasth3_8step();
        let (h, w) = (64, 64);
        let rgb = vec![128u8; h * w * 3];
        let prep = prepare_vision_image(&rgb, h, w, &cfg).unwrap();
        let tokens = prep.grid.num_tokens(cfg.spatial_merge_size).unwrap();
        assert!(tokens > 0);
        assert_eq!(prep.pixels.len(), prep.grid.num_patches() * prep.patch_dim);
    }

    #[test]
    fn qwen_video_sample_pads_temporal_patch() {
        let (idx, ts) = sample_qwen_video_frames(48, 2.0, 2).unwrap();
        assert!(!idx.is_empty());
        assert_eq!(ts.len() % 2, 0);
        assert!(!ts.is_empty());
    }
}
