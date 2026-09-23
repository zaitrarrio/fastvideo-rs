//! Reference-free video sanity metrics and a contact sheet for human review.
//!
//! These catch the failure modes that shape/dimension checks miss: NaNs,
//! flat or saturated frames (dead weights, wrong latent scaling), frozen video
//! (temporal path broken), per-frame noise (no temporal coherence), and
//! brightness flashes (feat-cache/chunk boundary bugs).

use std::path::Path;

use anyhow::{bail, Result};
use serde::Serialize;

#[derive(Debug, Clone, Serialize)]
pub struct VideoStats {
    pub frames: usize,
    pub height: usize,
    pub width: usize,
    pub non_finite: usize,
    /// Mean / min over frames of the per-frame pixel std, in 0..255 units.
    pub frame_std_mean: f64,
    pub frame_std_min: f64,
    /// Fraction of samples at the clamp rails (|x| >= 0.99).
    pub clipped_fraction: f64,
    /// Mean absolute difference between consecutive frames, 0..255 units.
    pub temporal_mad_mean: f64,
    pub temporal_mad_max: f64,
    /// Largest jump in mean brightness between consecutive frames, 0..255.
    pub luma_jump_max: f64,
    pub frame_mean: Vec<f64>,
}

#[derive(Debug, Clone, Copy, Serialize)]
pub struct QualityGates {
    pub frame_std_min: f64,
    pub clipped_fraction_max: f64,
    pub temporal_mad_min: f64,
    pub temporal_mad_max: f64,
    pub luma_jump_max: f64,
}

impl Default for QualityGates {
    fn default() -> Self {
        Self {
            frame_std_min: 3.0,
            clipped_fraction_max: 0.25,
            temporal_mad_min: 0.2,
            temporal_mad_max: 45.0,
            luma_jump_max: 40.0,
        }
    }
}

impl VideoStats {
    /// Names of every failed gate (empty = pass).
    pub fn failures(&self, g: &QualityGates) -> Vec<String> {
        let mut f = Vec::new();
        if self.non_finite > 0 {
            f.push(format!("non_finite={}", self.non_finite));
        }
        if self.frame_std_min < g.frame_std_min {
            f.push(format!(
                "flat frame (std {:.2} < {})",
                self.frame_std_min, g.frame_std_min
            ));
        }
        if self.clipped_fraction > g.clipped_fraction_max {
            f.push(format!(
                "saturated ({:.3} > {})",
                self.clipped_fraction, g.clipped_fraction_max
            ));
        }
        if self.frames > 1 && self.temporal_mad_mean < g.temporal_mad_min {
            f.push(format!(
                "frozen (mad {:.3} < {})",
                self.temporal_mad_mean, g.temporal_mad_min
            ));
        }
        if self.temporal_mad_mean > g.temporal_mad_max {
            f.push(format!(
                "incoherent (mad {:.2} > {})",
                self.temporal_mad_mean, g.temporal_mad_max
            ));
        }
        if self.luma_jump_max > g.luma_jump_max {
            f.push(format!(
                "flash (luma jump {:.2} > {})",
                self.luma_jump_max, g.luma_jump_max
            ));
        }
        f
    }
}

/// `video` is `[1, 3, F, H, W]` in `[-1, 1]` (C-order).
pub fn video_stats(video: &[f32], shape: &[usize]) -> Result<VideoStats> {
    if shape.len() != 5 || shape[0] != 1 || shape[1] < 3 {
        bail!("expected [1,3,F,H,W] video, got {shape:?}");
    }
    let (c, f, h, w) = (shape[1], shape[2], shape[3], shape[4]);
    let plane = h * w;
    let px = |ch: usize, t: usize, i: usize| video[(ch * f + t) * plane + i];
    let to255 = |v: f32| (f64::from(v) + 1.0) * 127.5;

    let mut non_finite = 0usize;
    let mut clipped = 0usize;
    let mut frame_mean = Vec::with_capacity(f);
    let mut frame_std = Vec::with_capacity(f);
    for t in 0..f {
        let (mut sum, mut sq) = (0.0f64, 0.0f64);
        for ch in 0..3 {
            for i in 0..plane {
                let v = px(ch, t, i);
                if !v.is_finite() {
                    non_finite += 1;
                    continue;
                }
                if v.abs() >= 0.99 {
                    clipped += 1;
                }
                let v = to255(v.clamp(-1.0, 1.0));
                sum += v;
                sq += v * v;
            }
        }
        let n = (3 * plane) as f64;
        let mean = sum / n;
        frame_mean.push(mean);
        frame_std.push((sq / n - mean * mean).max(0.0).sqrt());
    }
    let mut mads = Vec::new();
    let mut luma_jump_max = 0.0f64;
    for t in 1..f {
        let mut acc = 0.0f64;
        for ch in 0..3 {
            for i in 0..plane {
                let (a, b) = (px(ch, t, i), px(ch, t - 1, i));
                if a.is_finite() && b.is_finite() {
                    acc += (to255(a.clamp(-1.0, 1.0)) - to255(b.clamp(-1.0, 1.0))).abs();
                }
            }
        }
        mads.push(acc / (3 * plane) as f64);
        luma_jump_max = luma_jump_max.max((frame_mean[t] - frame_mean[t - 1]).abs());
    }
    let _ = c;
    let avg = |v: &[f64]| {
        if v.is_empty() {
            0.0
        } else {
            v.iter().sum::<f64>() / v.len() as f64
        }
    };
    Ok(VideoStats {
        frames: f,
        height: h,
        width: w,
        non_finite,
        frame_std_mean: avg(&frame_std),
        frame_std_min: frame_std.iter().copied().fold(f64::INFINITY, f64::min),
        clipped_fraction: clipped as f64 / (3 * plane * f).max(1) as f64,
        temporal_mad_mean: avg(&mads),
        temporal_mad_max: mads.iter().copied().fold(0.0, f64::max),
        luma_jump_max,
        frame_mean,
    })
}

/// 2×4 grid of evenly spaced frames, each downscaled by an integer stride.
pub fn contact_sheet(video: &[f32], shape: &[usize], path: &Path) -> Result<()> {
    let (f, h, w) = (shape[2], shape[3], shape[4]);
    let stride = (w / 208).max(1);
    let (th, tw) = (h / stride, w / stride);
    let cols = 4usize;
    let tiles = 8usize.min(f);
    let rows = tiles.div_ceil(cols);
    let mut img = image::RgbImage::new((tw * cols) as u32, (th * rows) as u32);
    let plane = h * w;
    for k in 0..tiles {
        let t = if tiles == 1 {
            0
        } else {
            k * (f - 1) / (tiles - 1)
        };
        for y in 0..th {
            for x in 0..tw {
                let i = (y * stride) * w + x * stride;
                let mut rgb = [0u8; 3];
                for (ch, out) in rgb.iter_mut().enumerate() {
                    let v = video[(ch * f + t) * plane + i];
                    *out = if v.is_finite() {
                        ((v + 1.0) * 127.5).clamp(0.0, 255.0) as u8
                    } else {
                        255
                    };
                }
                img.put_pixel(
                    (x + (k % cols) * tw) as u32,
                    (y + (k / cols) * th) as u32,
                    image::Rgb(rgb),
                );
            }
        }
    }
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    img.save(path)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn video(
        f: usize,
        h: usize,
        w: usize,
        px: impl Fn(usize, usize, usize, usize) -> f32,
    ) -> Vec<f32> {
        let mut v = vec![0.0; 3 * f * h * w];
        for ch in 0..3 {
            for t in 0..f {
                for y in 0..h {
                    for x in 0..w {
                        v[((ch * f + t) * h + y) * w + x] = px(ch, t, y, x);
                    }
                }
            }
        }
        v
    }

    #[test]
    fn flat_video_fails_std_and_frozen() {
        let v = video(4, 8, 8, |_, _, _, _| 0.0);
        let s = video_stats(&v, &[1, 3, 4, 8, 8]).unwrap();
        let fails = s.failures(&QualityGates::default());
        assert!(fails.iter().any(|f| f.starts_with("flat")));
        assert!(fails.iter().any(|f| f.starts_with("frozen")));
    }

    #[test]
    fn moving_gradient_passes() {
        let v = video(6, 16, 16, |ch, t, y, x| {
            (((x + 2 * t) % 16) as f32 / 8.0 - 1.0) * 0.8 + (y as f32 / 64.0) - ch as f32 * 0.05
        });
        let s = video_stats(&v, &[1, 3, 6, 16, 16]).unwrap();
        assert!(s.failures(&QualityGates::default()).is_empty(), "{s:?}");
    }

    #[test]
    fn nan_frame_fails() {
        let v = video(2, 4, 4, |_, t, _, _| if t == 1 { f32::NAN } else { 0.5 });
        let s = video_stats(&v, &[1, 3, 2, 4, 4]).unwrap();
        assert!(s
            .failures(&QualityGates::default())
            .iter()
            .any(|f| f.starts_with("non_finite")));
    }
}
