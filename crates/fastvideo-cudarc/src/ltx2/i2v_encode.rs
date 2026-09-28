//! LTX-2 image conditioning: first-frame I2V and keyframes.
//!
//! The reference is `ltx_pipelines` (Lightricks/LTX-2 `fd4ded7`):
//!
//! * **Preprocess** (`utils/media_io/decode.py:load_image_and_preprocess`):
//!   decode to sRGB RGB (EXIF rotations 3/6/8 applied, alpha dropped), re-encode
//!   as one H.264 frame at the checkpoint's CRF and decode it back
//!   (`preprocess`: libx264 `veryfast`, yuv420p, even crop; CRF 33 for 2.0-2.3,
//!   18 from 2.4 on, `constants.py:DEFAULT_IMAGE_CRF / LTX_2_4_IMAGE_CRF`), then
//!   `resize_and_center_crop` (bilinear, `align_corners=False`, no antialias, to
//!   the smallest size covering the target, then a centered crop) and
//!   `x / 127.5 − 1` in bf16.
//! * **Encode** with the video VAE encoder ([`super::vae_encoder`]): one latent
//!   frame per image, at each stage's resolution (half size for stage 1 of the
//!   two-stage pipeline, full size for stage 2).
//! * **Condition** (`helpers.py:combined_image_conditionings`): an image at
//!   frame 0 replaces the latent tokens of latent frame 0
//!   (`VideoConditionByLatentIndex`); an image at any other pixel frame `k` is
//!   *appended* as extra tokens whose rotary positions sit at pixel frame
//!   `[k, k + 1)` (`VideoConditionByKeyframeIndex`). Either way those tokens
//!   get the clean latent and the denoise mask `1 − strength`.
//! * **Sample** with the mask: the noiser draws over every token and
//!   `lerp(clean, noised, mask)` (`noisers.py`); each forward sees per-token
//!   timesteps `mask · σ` ([`super::transformer::VideoTimestepSegment`]); `x0 =
//!   x − mask·σ·v` and `x0 ← x0·mask + clean·(1 − mask)` before every update,
//!   and again after an ancestral step's noise (`samplers.py:post_process_latent`).
//!   The appended tokens are dropped after the stage (`clear_conditioning`).
//! * **IC-LoRA reference** (`iclora_utils.py:append_ic_lora_reference_video_conditionings`,
//!   `VideoConditionByReferenceLatent`): a reference *video* (the Ingredients
//!   LoRA's reference sheet looped into a static clip) is decoded frame by
//!   frame, `resize_and_center_crop`ped to the stage size over the LoRA's
//!   `reference_downscale_factor` (no CRF re-encode), VAE-encoded and appended
//!   after the image conditionings as clean tokens (mask `1 − strength`) with
//!   their own causal-fixed positions ([`fastvideo_models::ltx2::rope::ReferenceBlock`]).
//!   A static clip of identical frames encodes to one latent frame repeated
//!   (every causal conv and space-to-depth sees identical frames), so
//!   [`ReferenceTokens::static_clip`] encodes the still once and repeats it.

use std::path::{Path, PathBuf};
use std::process::Command;

use fastvideo_models::ltx2::config::Ltx2ModelVersion;
use image::RgbImage;

use crate::wan::pipeline::{PipelineError, Result};
use crate::wan::tensor::CudaTensor;

use super::transformer::VideoTimestepSegment;

fn msg(s: impl Into<String>) -> PipelineError {
    PipelineError::Message(s.into())
}

/// One conditioning image (`ImageConditioningInput`: path, pixel frame index,
/// strength, CRF).
#[derive(Debug, Clone, PartialEq)]
pub struct ConditioningImage {
    pub path: PathBuf,
    /// Pixel frame the image pins: 0 for I2V, `num_frames − 1` for a last frame.
    pub frame_idx: usize,
    /// 1.0 = the frame is kept clean; 0.0 = no conditioning.
    pub strength: f32,
    /// H.264 CRF of the re-encode (`None` = the checkpoint's; 0 = none).
    pub crf: Option<u32>,
}

impl ConditioningImage {
    pub fn first_frame(path: impl Into<PathBuf>) -> Self {
        Self {
            path: path.into(),
            frame_idx: 0,
            strength: 1.0,
            crf: None,
        }
    }
}

/// The CRF an image conditioning is re-encoded at for a checkpoint line
/// (`detect_params(...).default_image_crf`: 18 from 2.4 on, else 33).
pub fn default_crf(version: Ltx2ModelVersion) -> u32 {
    match version {
        Ltx2ModelVersion::V25 => 18,
        _ => 33,
    }
}

/// Decode `path` the way `decode_image` does: RGB, EXIF rotations applied
/// (orientations 3, 6 and 8 only, as the reference), alpha dropped. The ICC
/// profile conversion to sRGB is not applied (most inputs are sRGB already).
pub fn decode_image(path: &Path) -> Result<RgbImage> {
    use image::ImageDecoder;
    let reader = image::ImageReader::open(path)
        .map_err(|e| msg(format!("ltx2 image {}: {e}", path.display())))?
        .with_guessed_format()
        .map_err(|e| msg(format!("ltx2 image {}: {e}", path.display())))?;
    let mut decoder = reader
        .into_decoder()
        .map_err(|e| msg(format!("ltx2 image {}: {e}", path.display())))?;
    let orientation = decoder.orientation().ok();
    let img = image::DynamicImage::from_decoder(decoder)
        .map_err(|e| msg(format!("ltx2 image {}: {e}", path.display())))?;
    let img = match orientation {
        Some(image::metadata::Orientation::Rotate90) => img.rotate90(),
        Some(image::metadata::Orientation::Rotate180) => img.rotate180(),
        Some(image::metadata::Orientation::Rotate270) => img.rotate270(),
        _ => img,
    };
    Ok(img.into_rgb8())
}

/// `preprocess(image, crf)`: one libx264 frame at `crf` (`veryfast`, yuv420p,
/// cropped to even dimensions) decoded back to RGB, through the `ffmpeg` on
/// the PATH (`FASTVIDEO_FFMPEG` overrides). CRF 0 returns the image as is.
pub fn recompress(img: &RgbImage, crf: u32) -> Result<RgbImage> {
    if crf == 0 || img.width() < 2 || img.height() < 2 {
        return Ok(img.clone());
    }
    let (w, h) = (img.width() / 2 * 2, img.height() / 2 * 2);
    let cropped = image::imageops::crop_imm(img, 0, 0, w, h).to_image();
    let ffmpeg = std::env::var("FASTVIDEO_FFMPEG").unwrap_or_else(|_| "ffmpeg".into());
    let tmp = std::env::temp_dir().join(format!(
        "ltx2-crf-{}-{}.mp4",
        std::process::id(),
        uuid_like()
    ));
    let encode = || -> Result<()> {
        use std::io::Write;
        let mut child = Command::new(&ffmpeg)
            .args(["-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo"])
            .args(["-pix_fmt", "rgb24", "-s", &format!("{w}x{h}"), "-r", "1", "-i", "-"])
            .args(["-sws_flags", "bilinear", "-frames:v", "1", "-c:v", "libx264"])
            .args(["-preset", "veryfast", "-crf", &crf.to_string(), "-pix_fmt", "yuv420p"])
            .args(["-f", "mp4"])
            .arg(&tmp)
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::piped())
            .spawn()
            .map_err(|e| msg(format!("ltx2 image CRF re-encode: {ffmpeg}: {e}")))?;
        child
            .stdin
            .take()
            .expect("stdin")
            .write_all(cropped.as_raw())
            .map_err(|e| msg(format!("ltx2 image CRF re-encode: {e}")))?;
        let out = child
            .wait_with_output()
            .map_err(|e| msg(format!("ltx2 image CRF re-encode: {e}")))?;
        if !out.status.success() {
            return Err(msg(format!(
                "ltx2 image CRF re-encode failed: {}",
                String::from_utf8_lossy(&out.stderr).trim()
            )));
        }
        Ok(())
    };
    let result = encode().and_then(|()| {
        let out = Command::new(&ffmpeg)
            .args(["-hide_banner", "-loglevel", "error", "-i"])
            .arg(&tmp)
            .args(["-frames:v", "1", "-sws_flags", "bilinear", "-f", "rawvideo"])
            .args(["-pix_fmt", "rgb24", "-"])
            .output()
            .map_err(|e| msg(format!("ltx2 image CRF decode: {e}")))?;
        if !out.status.success() || out.stdout.len() != (w * h * 3) as usize {
            return Err(msg(format!(
                "ltx2 image CRF decode: {} bytes for {w}x{h} ({})",
                out.stdout.len(),
                String::from_utf8_lossy(&out.stderr).trim()
            )));
        }
        RgbImage::from_raw(w, h, out.stdout).ok_or_else(|| msg("ltx2 image CRF decode: size"))
    });
    let _ = std::fs::remove_file(&tmp);
    result
}

fn uuid_like() -> u64 {
    use std::sync::atomic::{AtomicU64, Ordering};
    static N: AtomicU64 = AtomicU64::new(0);
    let t = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0);
    t ^ N.fetch_add(1, Ordering::Relaxed).wrapping_mul(0x9e37_79b9_7f4a_7c15)
}

/// `torch.nn.functional.interpolate(mode="bilinear", align_corners=False)`
/// source coordinates for one axis: `(i0, i1, λ)` per output index.
fn bilinear_axis(src: usize, dst: usize) -> Vec<(usize, usize, f32)> {
    // ATen: scale = src / dst in the accumulate type (float), source index
    // `scale · (i + 0.5) − 0.5` clamped at 0.
    let scale = src as f32 / dst as f32;
    (0..dst)
        .map(|i| {
            let s = (scale * (i as f32 + 0.5) - 0.5).max(0.0);
            let i0 = (s as usize).min(src - 1);
            let i1 = if i0 < src - 1 { i0 + 1 } else { i0 };
            (i0, i1, s - i0 as f32)
        })
        .collect()
}

/// `resize_and_center_crop` + `normalize_images`: `[3, height, width]`
/// row-major in `[−1, 1]`, rounded to bf16 as the reference casts it.
pub fn image_pixels(img: &RgbImage, height: usize, width: usize) -> Vec<f32> {
    let (sh, sw) = (img.height() as usize, img.width() as usize);
    let scale = f64::max(height as f64 / sh as f64, width as f64 / sw as f64);
    let (nh, nw) = (
        (sh as f64 * scale).ceil() as usize,
        (sw as f64 * scale).ceil() as usize,
    );
    let (top, left) = ((nh - height) / 2, (nw - width) / 2);
    let ys = bilinear_axis(sh, nh);
    let xs = bilinear_axis(sw, nw);
    let raw = img.as_raw();
    let px = |y: usize, x: usize, c: usize| f32::from(raw[(y * sw + x) * 3 + c]);
    let mut out = vec![0f32; 3 * height * width];
    for oy in 0..height {
        let (y0, y1, ly) = ys[oy + top];
        for ox in 0..width {
            let (x0, x1, lx) = xs[ox + left];
            for c in 0..3 {
                // ATen's upsample_bilinear2d: h0lambda·(w0lambda·p00 + w1lambda·p01)
                // + h1lambda·(w0lambda·p10 + w1lambda·p11).
                let top_row = (1.0 - lx) * px(y0, x0, c) + lx * px(y0, x1, c);
                let bottom = (1.0 - lx) * px(y1, x0, c) + lx * px(y1, x1, c);
                let v = (1.0 - ly) * top_row + ly * bottom;
                out[(c * height + oy) * width + ox] =
                    fastvideo_models::ltx2::schedule::bf16_round(v / 127.5 - 1.0);
            }
        }
    }
    out
}

/// The IC-LoRA reference of a stage (stage 1 of `ICLoraPipeline` only).
pub struct ReferenceTokens {
    /// The reference's latent grid `[F, H, W]`.
    pub grid: [usize; 3],
    /// `reference_downscale_factor` of the LoRA.
    pub downscale: usize,
    /// `VideoConditionByReferenceLatent.strength`: 1 keeps it clean.
    pub strength: f32,
    /// Packed `[1, F·H·W, C]` in the state's dtype.
    pub clean: CudaTensor,
}

impl ReferenceTokens {
    /// A static clip of `frames` latent frames of one still: `frame` is the
    /// still's packed latent `[1, H·W, C]`, repeated frame-major.
    pub fn static_clip(
        frame: &CudaTensor,
        frames: usize,
        hw: [usize; 2],
        downscale: usize,
        strength: f32,
    ) -> Result<Self> {
        if frame.shape.len() != 3 || frame.shape[1] != hw[0] * hw[1] || frames == 0 {
            return Err(msg(format!(
                "ltx2 reference latent {:?} for a {}x{} grid, {frames} frames",
                frame.shape, hw[0], hw[1]
            )));
        }
        let parts: Vec<&CudaTensor> = std::iter::repeat_n(frame, frames).collect();
        let clean = if frames == 1 {
            frame.clone()
        } else {
            CudaTensor::cat(&parts, 1)?
        };
        Ok(Self {
            grid: [frames, hw[0], hw[1]],
            downscale,
            strength,
            clean,
        })
    }

    pub fn tokens(&self) -> usize {
        self.grid.iter().product()
    }

    pub fn block(&self) -> fastvideo_models::ltx2::rope::ReferenceBlock {
        fastvideo_models::ltx2::rope::ReferenceBlock {
            grid: self.grid,
            downscale: self.downscale,
        }
    }
}

/// One conditioned run of video rows of a stage.
pub struct CondSegment {
    pub start: usize,
    pub len: usize,
    /// Denoise mask `1 − strength`.
    pub mask: f32,
    /// The clean tokens `[1, len, C]` in the state's dtype.
    pub clean: CudaTensor,
}

/// The image conditioning of one denoise stage.
pub struct StageConditioning {
    /// Rows of the stage's latent grid (`F·H·W`); appended keyframe tokens
    /// follow them.
    pub grid_tokens: usize,
    /// Appended keyframe token blocks: the pixel frame each one pins.
    pub extra_frames: Vec<usize>,
    /// `H·W` of the stage's grid: the size of one frame's token block.
    pub frame_tokens: usize,
    /// Sorted, disjoint.
    pub segments: Vec<CondSegment>,
    /// The IC-LoRA reference block after the keyframe blocks, if any.
    pub reference: Option<fastvideo_models::ltx2::rope::ReferenceBlock>,
}

impl StageConditioning {
    /// `latents[i]` is image `i`'s packed latent `[1, H·W, C]` at this stage's
    /// grid (`[F, H, W]`). Frame 0 replaces latent frame 0; frame `k > 0` is
    /// appended. A later image at frame 0 overrides an earlier one.
    pub fn new(
        grid: [usize; 3],
        images: &[ConditioningImage],
        latents: Vec<CudaTensor>,
        num_frames: usize,
    ) -> Result<Self> {
        let [f, h, w] = grid;
        let hw = h * w;
        let mut first: Option<CondSegment> = None;
        let mut extra = Vec::new();
        let mut extra_frames = Vec::new();
        for (img, lat) in images.iter().zip(latents) {
            if lat.shape.len() != 3 || lat.shape[1] != hw {
                return Err(msg(format!(
                    "ltx2 conditioning latent {:?} for a {h}x{w} grid",
                    lat.shape
                )));
            }
            if img.frame_idx >= num_frames {
                return Err(msg(format!(
                    "ltx2 conditioning frame {} is past the clip ({num_frames} frames)",
                    img.frame_idx
                )));
            }
            if !(0.0..=1.0).contains(&img.strength) {
                return Err(msg(format!(
                    "ltx2 conditioning strength {} is outside [0, 1]",
                    img.strength
                )));
            }
            if img.frame_idx == 0 {
                first = Some(CondSegment {
                    start: 0,
                    len: hw,
                    mask: 1.0 - img.strength,
                    clean: lat,
                });
            } else {
                extra.push((img.strength, lat));
                extra_frames.push(img.frame_idx);
            }
        }
        let grid_tokens = f * hw;
        let mut segments: Vec<CondSegment> = first.into_iter().collect();
        for (k, (strength, lat)) in extra.into_iter().enumerate() {
            segments.push(CondSegment {
                start: grid_tokens + k * hw,
                len: hw,
                mask: 1.0 - strength,
                clean: lat,
            });
        }
        Ok(Self {
            grid_tokens,
            extra_frames,
            frame_tokens: hw,
            segments,
            reference: None,
        })
    }

    /// Append the IC-LoRA reference after every other block (`ic_lora.py`
    /// `_create_conditionings`: image conditionings first, references last).
    pub fn with_reference(mut self, r: ReferenceTokens) -> Result<Self> {
        if self.reference.is_some() {
            return Err(msg("ltx2 conditioning: one reference per stage"));
        }
        if !(0.0..=1.0).contains(&r.strength) {
            return Err(msg(format!(
                "ltx2 reference strength {} is outside [0, 1]",
                r.strength
            )));
        }
        if r.clean.shape.len() != 3 || r.clean.shape[1] != r.tokens() {
            return Err(msg(format!(
                "ltx2 reference latent {:?} for a {:?} grid",
                r.clean.shape, r.grid
            )));
        }
        let start = self.total_tokens();
        self.reference = Some(r.block());
        self.segments.push(CondSegment {
            start,
            len: r.tokens(),
            mask: 1.0 - r.strength,
            clean: r.clean,
        });
        Ok(self)
    }

    /// Rows appended after the grid (keyframes and the reference).
    pub fn appended_tokens(&self) -> usize {
        self.extra_frames.len() * self.frame_tokens + self.reference.map_or(0, |r| r.tokens())
    }

    /// All rows: the grid, the appended keyframe tokens and the reference.
    pub fn total_tokens(&self) -> usize {
        self.grid_tokens + self.appended_tokens()
    }

    /// The per-token timestep runs for the transformer (rows with mask 1 are
    /// left out: they run at `σ`).
    pub fn timestep_segments(&self) -> Vec<VideoTimestepSegment> {
        self.segments
            .iter()
            .filter(|s| s.mask != 1.0)
            .map(|s| VideoTimestepSegment {
                start: s.start,
                len: s.len,
                mask: s.mask,
            })
            .collect()
    }

    /// `x` with each segment's rows replaced by `f(rows, segment)`.
    fn splice(
        &self,
        x: &CudaTensor,
        mut f: impl FnMut(&CudaTensor, &CondSegment) -> Result<CudaTensor>,
    ) -> Result<CudaTensor> {
        let rows = x.shape[1];
        let mut parts = Vec::with_capacity(2 * self.segments.len() + 1);
        let mut at = 0usize;
        for s in &self.segments {
            if s.start < at || s.start + s.len > rows {
                return Err(msg(format!(
                    "ltx2 conditioning rows {}..{} outside {rows}",
                    s.start,
                    s.start + s.len
                )));
            }
            if s.start > at {
                parts.push(x.narrow(1, at, s.start - at)?);
            }
            parts.push(f(&x.narrow(1, s.start, s.len)?, s)?);
            at = s.start + s.len;
        }
        if parts.is_empty() {
            return Ok(x.clone());
        }
        if at < rows {
            parts.push(x.narrow(1, at, rows - at)?);
        }
        let refs: Vec<&CudaTensor> = parts.iter().collect();
        Ok(CudaTensor::cat(&refs, 1)?)
    }

    /// `torch.lerp(clean, noised, mask)` on the conditioned rows (`GaussianNoiser`).
    pub fn apply_initial(&self, noised: &CudaTensor) -> Result<CudaTensor> {
        self.splice(noised, |x, s| lerp(&s.clean, x, s.mask))
    }

    /// `X0Model`: `x − (mask·σ)·v` on the conditioned rows, `x − σ·v` elsewhere.
    pub fn x0(&self, x: &CudaTensor, v: &CudaTensor, sigma: f32) -> Result<CudaTensor> {
        let plain = CudaTensor::lincomb(&[(1.0, x), (-sigma, v)])?;
        self.splice(&plain, |_, s| {
            let xs = x.narrow(1, s.start, s.len)?;
            let vs = v.narrow(1, s.start, s.len)?;
            Ok(CudaTensor::lincomb(&[(1.0, &xs), (-(s.mask * sigma), &vs)])?)
        })
    }

    /// `post_process_latent`: `x·mask + clean·(1 − mask)` on the conditioned rows.
    pub fn post(&self, x: &CudaTensor) -> Result<CudaTensor> {
        self.splice(x, |xs, s| {
            if s.mask == 0.0 {
                Ok(s.clean.clone())
            } else {
                Ok(CudaTensor::lincomb(&[(s.mask, xs), (1.0 - s.mask, &s.clean)])?)
            }
        })
    }

    /// `clear_conditioning`: the grid rows only.
    pub fn clear(&self, x: &CudaTensor) -> Result<CudaTensor> {
        if x.shape[1] == self.grid_tokens {
            return Ok(x.clone());
        }
        Ok(x.narrow(1, 0, self.grid_tokens)?)
    }
}

/// `torch.lerp(a, b, w)` for a scalar weight: ATen evaluates `a + w·(b − a)`
/// below 0.5 and `b − (b − a)·(1 − w)` from 0.5 on.
fn lerp(a: &CudaTensor, b: &CudaTensor, w: f32) -> Result<CudaTensor> {
    if w == 0.0 {
        return Ok(a.clone());
    }
    if w == 1.0 {
        return Ok(b.clone());
    }
    let diff = CudaTensor::lincomb(&[(1.0, b), (-1.0, a)])?;
    Ok(if w < 0.5 {
        CudaTensor::lincomb(&[(1.0, a), (w, &diff)])?
    } else {
        CudaTensor::lincomb(&[(1.0, b), (-(1.0 - w), &diff)])?
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn t(v: Vec<f32>, shape: Vec<usize>) -> CudaTensor {
        CudaTensor::from_vec(v, shape).unwrap()
    }

    #[test]
    fn bilinear_matches_aten_half_pixel_centers() {
        // 4 → 2: sources 0.5 and 2.5.
        let a = bilinear_axis(4, 2);
        assert_eq!(a[0], (0, 1, 0.5));
        assert_eq!(a[1], (2, 3, 0.5));
        // 2 → 4: 0.25·i − 0.25 clamped: -0.25→0, 0.25, 0.75, 1.25→(1,1).
        let b = bilinear_axis(2, 4);
        assert_eq!(b[0], (0, 1, 0.0));
        assert_eq!(b[1], (0, 1, 0.25));
        assert_eq!(b[3].0, 1);
        assert_eq!(b[3].1, 1);
    }

    #[test]
    fn pixels_cover_then_center_crop() {
        // 4x2 (w x h) image into 2x2: scale 1 → 4x2 → crop columns 1..3.
        let mut img = RgbImage::new(4, 2);
        for (x, y, p) in img.enumerate_pixels_mut() {
            *p = image::Rgb([(x * 50) as u8, (y * 100) as u8, 0]);
        }
        let px = image_pixels(&img, 2, 2);
        assert_eq!(px.len(), 12);
        let r = |x: u32| fastvideo_models::ltx2::schedule::bf16_round((x * 50) as f32 / 127.5 - 1.0);
        assert_eq!(px[0], r(1));
        assert_eq!(px[1], r(2));
    }

    #[test]
    fn first_frame_and_keyframe_rows() {
        let lat = |v: f32| t(vec![v; 2 * 3], vec![1, 2, 3]);
        let imgs = vec![
            ConditioningImage::first_frame("a.png"),
            ConditioningImage {
                path: "b.png".into(),
                frame_idx: 16,
                strength: 0.5,
                crf: None,
            },
        ];
        let c = StageConditioning::new([3, 1, 2], &imgs, vec![lat(7.0), lat(9.0)], 17).unwrap();
        assert_eq!(c.grid_tokens, 6);
        assert_eq!(c.total_tokens(), 8);
        assert_eq!(c.extra_frames, vec![16]);
        let segs = c.timestep_segments();
        assert_eq!(segs.len(), 2);
        assert_eq!((segs[0].start, segs[0].len, segs[0].mask), (0, 2, 0.0));
        assert_eq!((segs[1].start, segs[1].len, segs[1].mask), (6, 2, 0.5));

        let noise = t(vec![1.0; 8 * 3], vec![1, 8, 3]);
        let x = c.apply_initial(&noise).unwrap().host_cow().unwrap().into_owned();
        assert_eq!(&x[..6], &[7.0; 6]);
        assert_eq!(&x[6..18], &[1.0; 12]);
        // lerp(9, 1, 0.5) = 1 − (1 − 9)·0.5 = 5
        assert_eq!(&x[18..], &[5.0; 6]);

        let v = t(vec![2.0; 8 * 3], vec![1, 8, 3]);
        let xs = t(vec![3.0; 8 * 3], vec![1, 8, 3]);
        let x0 = c.x0(&xs, &v, 0.5).unwrap().host_cow().unwrap().into_owned();
        assert_eq!(&x0[..6], &[3.0; 6]); // mask 0: timestep 0
        assert_eq!(&x0[6..18], &[2.0; 12]); // 3 − 0.5·2
        assert_eq!(&x0[18..], &[2.5; 6]); // 3 − 0.25·2
        let p = c.post(&t(x0, vec![1, 8, 3])).unwrap().host_cow().unwrap().into_owned();
        assert_eq!(&p[..6], &[7.0; 6]);
        assert_eq!(&p[18..], &[0.5 * 2.5 + 0.5 * 9.0; 6]);
        assert_eq!(c.clear(&noise).unwrap().shape, vec![1, 6, 3]);
    }

    #[test]
    fn reference_rows_follow_the_keyframes_and_stay_clean() {
        let lat = |v: f32, n: usize| t(vec![v; n * 3], vec![1, n, 3]);
        let imgs = vec![ConditioningImage {
            path: "b.png".into(),
            frame_idx: 16,
            strength: 1.0,
            crf: None,
        }];
        // Grid [3, 1, 2]: 6 rows; one keyframe block (2); a static reference
        // clip of 3 latent frames of the same 1x2 still (6).
        let r = ReferenceTokens::static_clip(&lat(4.0, 2), 3, [1, 2], 1, 1.0).unwrap();
        assert_eq!(r.grid, [3, 1, 2]);
        let c = StageConditioning::new([3, 1, 2], &imgs, vec![lat(9.0, 2)], 17)
            .unwrap()
            .with_reference(r)
            .unwrap();
        assert_eq!(c.total_tokens(), 14);
        assert_eq!(c.appended_tokens(), 8);
        let segs = c.timestep_segments();
        assert_eq!((segs[1].start, segs[1].len, segs[1].mask), (8, 6, 0.0));
        let noise = t(vec![1.0; 14 * 3], vec![1, 14, 3]);
        let x = c.apply_initial(&noise).unwrap().host_cow().unwrap().into_owned();
        assert_eq!(&x[..18], &[1.0; 18]);
        assert_eq!(&x[18..24], &[9.0; 6]);
        assert_eq!(&x[24..], &[4.0; 18]);
        let v = t(vec![2.0; 14 * 3], vec![1, 14, 3]);
        let x0 = c.x0(&noise, &v, 0.5).unwrap().host_cow().unwrap().into_owned();
        assert_eq!(&x0[24..], &[1.0; 18]); // mask 0: timestep 0, x0 = x
        let p = c.post(&t(x0, vec![1, 14, 3])).unwrap().host_cow().unwrap().into_owned();
        assert_eq!(&p[24..], &[4.0; 18]);
        assert_eq!(c.clear(&noise).unwrap().shape, vec![1, 6, 3]);
        // A second reference, or one outside [0, 1], is refused.
        let r2 = ReferenceTokens::static_clip(&lat(4.0, 2), 1, [1, 2], 1, 1.0).unwrap();
        assert!(c.with_reference(r2).is_err());
        let bad = ReferenceTokens::static_clip(&lat(4.0, 2), 1, [1, 2], 1, 1.5).unwrap();
        assert!(StageConditioning::new([3, 1, 2], &[], vec![], 17)
            .unwrap()
            .with_reference(bad)
            .is_err());
    }
}
