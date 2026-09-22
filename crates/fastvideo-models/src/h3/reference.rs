//! MiniMax-H3 Ref2VA reference media: limits, canvas math, and prepared
//! geometry (FastVideo `pipelines/basic/minimax_h3/reference.py`).
//!
//! Video/audio decode stays at the call site (cudarc / CLI); this module is the
//! host contract the packed layout and encoders consume.

use super::config::{
    resolve_canvas_size, H3_AUDIO_CHANNELS, H3_FRAMES_PER_CHUNK, H3_FPS, H3_LATENTS_PER_CHUNK,
};

/// Released short-edge for reference images (`MINIMAX_H3_REFERENCE_IMAGE_SHORT_EDGE`).
pub const REFERENCE_IMAGE_SHORT_EDGE: usize = 2048;
/// Canvas multiple (VAE spatial + patch).
pub const CANVAS_MULTIPLE: usize = 32;

pub const MAX_REFERENCE_IMAGES: usize = 9;
pub const MAX_REFERENCE_VIDEOS: usize = 3;
pub const MAX_REFERENCE_AUDIOS: usize = 3;
pub const MAX_REFERENCES: usize = 12;

/// Ordered Ref2VA input kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReferenceKind {
    Image,
    Video,
    Audio,
}

impl ReferenceKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Image => "image",
            Self::Video => "video",
            Self::Audio => "audio",
        }
    }
}

/// One ordered reference before encode (path + kind).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct H3ReferenceSpec {
    pub path: std::path::PathBuf,
    pub kind: ReferenceKind,
}

/// Geometry after image prep / VAE encode planning (one latent frame for images).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PreparedImageRef {
    /// Pixel canvas after 2048-short-edge resize (multiples of 32).
    pub height: usize,
    pub width: usize,
    /// Latent spatial size (`height / spatial_compression`, same for width).
    pub latent_height: usize,
    pub latent_width: usize,
}

impl PreparedImageRef {
    pub fn from_pixel_size(height: usize, width: usize, spatial_compression: usize) -> Result<Self, String> {
        if height == 0 || width == 0 || spatial_compression == 0 {
            return Err(format!("bad image ref size {width}x{height} / {spatial_compression}"));
        }
        if height % spatial_compression != 0 || width % spatial_compression != 0 {
            return Err(format!(
                "image canvas {width}x{height} not divisible by VAE ratio {spatial_compression}"
            ));
        }
        Ok(Self {
            height,
            width,
            latent_height: height / spatial_compression,
            latent_width: width / spatial_compression,
        })
    }

    pub fn rows_per_frame(&self, patch: [usize; 3]) -> Result<usize, String> {
        let [pt, ph, pw] = patch;
        if pt != 1 {
            return Err(format!("Ref2VA image refs need temporal patch 1, got {pt}"));
        }
        if self.latent_height % ph != 0 || self.latent_width % pw != 0 {
            return Err(format!(
                "image latent {}x{} not divisible by patch {patch:?}",
                self.latent_height, self.latent_width
            ));
        }
        Ok((self.latent_height / ph) * (self.latent_width / pw))
    }
}

/// One prepared Ref2VA medium with latent geometry resolved (post-encode plan).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PreparedReference {
    Image(PreparedImageRef),
    /// Visual latents; optional soundtrack audio latents (stereo rows = 2 * Na).
    Video {
        num_latent_frames: usize,
        latent_height: usize,
        latent_width: usize,
        num_audio_latents: usize,
    },
    Audio {
        num_audio_latents: usize,
    },
}

impl PreparedReference {
    pub fn kind(&self) -> ReferenceKind {
        match self {
            Self::Image(_) => ReferenceKind::Image,
            Self::Video { .. } => ReferenceKind::Video,
            Self::Audio { .. } => ReferenceKind::Audio,
        }
    }

    pub fn has_audio(&self) -> bool {
        match self {
            Self::Image(_) => false,
            Self::Video {
                num_audio_latents, ..
            } => *num_audio_latents > 0,
            Self::Audio { .. } => true,
        }
    }

    pub fn num_audio_latents(&self) -> usize {
        match self {
            Self::Image(_) => 0,
            Self::Video {
                num_audio_latents, ..
            }
            | Self::Audio { num_audio_latents } => *num_audio_latents,
        }
    }

    pub fn video_rows(&self, patch: [usize; 3]) -> Result<usize, String> {
        let [pt, ph, pw] = patch;
        match self {
            Self::Audio { .. } => Ok(0),
            Self::Image(img) => img.rows_per_frame(patch),
            Self::Video {
                num_latent_frames,
                latent_height,
                latent_width,
                ..
            } => {
                if *num_latent_frames == 0
                    || *latent_height == 0
                    || *latent_width == 0
                    || num_latent_frames % pt != 0
                    || latent_height % ph != 0
                    || latent_width % pw != 0
                {
                    return Err(format!(
                        "video ref geometry {num_latent_frames}x{latent_height}x{latent_width} not divisible by {patch:?}"
                    ));
                }
                Ok((num_latent_frames / pt) * (latent_height / ph) * (latent_width / pw))
            }
        }
    }

    pub fn audio_rows(&self) -> usize {
        self.num_audio_latents() * H3_AUDIO_CHANNELS
    }
}

/// Ordered segment of the Ref2VA condition span (for packing projected rows).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RefSegment {
    /// Consume `rows` from the condition-video stream.
    Video { rows: usize },
    /// Consume `rows` from the condition-audio stream.
    Audio { rows: usize },
}

/// Resolve the released 2048-short-edge reference-image canvas.
pub fn resolve_reference_image_size(width: usize, height: usize) -> Result<(usize, usize), String> {
    if width == 0 || height == 0 {
        return Err(format!("reference image must have a positive size, got {width}x{height}"));
    }
    if width > 4 * height || height > 4 * width {
        return Err(format!("reference image must be within 1:4 and 4:1, got {width}x{height}"));
    }
    let scale = REFERENCE_IMAGE_SHORT_EDGE as f64 / (width.min(height) as f64);
    let multiple = CANVAS_MULTIPLE as f64;
    let out_h = ((height as f64 * scale / multiple).round() * multiple).max(multiple) as usize;
    let out_w = ((width as f64 * scale / multiple).round() * multiple).max(multiple) as usize;
    Ok((out_h, out_w))
}

/// Infer Ref2VA kind from a path extension (CLI `--ref` convenience).
pub fn infer_reference_kind(path: &std::path::Path) -> ReferenceKind {
    match path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_ascii_lowercase()
        .as_str()
    {
        "mp4" | "mov" | "webm" | "mkv" | "avi" | "m4v" | "mpeg" | "mpg" => ReferenceKind::Video,
        "wav" | "mp3" | "flac" | "ogg" | "m4a" | "aac" | "opus" => ReferenceKind::Audio,
        _ => ReferenceKind::Image,
    }
}

/// Trim video references to the causal VAE's complete chunk geometry
/// (`trim_reference_num_frames`): `((n - 5) // 17).max(1) * 17 + 5`.
pub fn trim_reference_num_frames(num_frames: usize) -> Result<usize, String> {
    if num_frames < 1 {
        return Err(format!("a reference video must have at least one frame, got {num_frames}"));
    }
    let chunks = (num_frames.saturating_sub(H3_LATENTS_PER_CHUNK)) / H3_FRAMES_PER_CHUNK;
    Ok(chunks.max(1) * H3_FRAMES_PER_CHUNK + H3_LATENTS_PER_CHUNK)
}

/// Nearest-resample reference frames onto H3's fixed 24 fps timeline
/// (FastVideo `resample_reference_frames`: `np.repeat` with slot diffs).
/// `frames` is `T * H * W * 3` uint8 (channel-last).
pub fn resample_reference_frames(
    frames: &[u8],
    num_frames: usize,
    height: usize,
    width: usize,
    fps: f64,
) -> Result<(Vec<u8>, usize), String> {
    let frame_bytes = height.checked_mul(width).and_then(|n| n.checked_mul(3)).ok_or_else(|| {
        format!("reference video frame size overflow {height}x{width}")
    })?;
    if num_frames == 0 || frames.len() != num_frames * frame_bytes {
        return Err(format!(
            "reference video must be non-empty RGB frames, got {num_frames} frames / {} bytes",
            frames.len()
        ));
    }
    if fps <= 0.0 {
        return Err(format!("a reference video must have a positive frame rate, got {fps}"));
    }
    if (fps - H3_FPS as f64).abs() < 1e-9 {
        return Ok((frames.to_vec(), num_frames));
    }
    let scale = H3_FPS as f64 / fps;
    let mut slots = Vec::with_capacity(num_frames + 1);
    for i in 0..num_frames {
        slots.push(((i as f64) * scale + 0.5).floor() as i64);
    }
    let append = ((num_frames as f64) * scale + 0.5).floor() as i64;
    slots.push(append);
    let out_len: usize = slots.windows(2).map(|w| (w[1] - w[0]).max(0) as usize).sum();
    if out_len == 0 {
        return Err("resampling a reference video produced zero frames".into());
    }
    let mut out = Vec::with_capacity(out_len * frame_bytes);
    for i in 0..num_frames {
        let repeats = (slots[i + 1] - slots[i]).max(0) as usize;
        let start = i * frame_bytes;
        for _ in 0..repeats {
            out.extend_from_slice(&frames[start..start + frame_bytes]);
        }
    }
    Ok((out, out_len))
}

/// Trim to `num_frames` and resolve a 768-short-edge canvas for the clip's
/// aspect ratio. Returns `(frames_hwc, out_h, out_w)` — caller still resizes.
pub fn plan_reference_video_canvas(
    src_height: usize,
    src_width: usize,
    num_frames: usize,
    keep_frames: usize,
) -> Result<(usize, usize, usize), String> {
    if src_height == 0 || src_width == 0 || num_frames == 0 {
        return Err(format!(
            "a reference video must be non-empty RGB frames, got {num_frames}x{src_height}x{src_width}"
        ));
    }
    let keep = keep_frames.min(num_frames).max(1);
    let (out_h, out_w) = resolve_canvas_size(src_width as f64, src_height as f64)?;
    Ok((keep, out_h, out_w))
}

/// Validate ordered Ref2VA specs against per-modality and total caps.
pub fn validate_references(refs: &[H3ReferenceSpec]) -> Result<(), String> {
    if refs.is_empty() {
        return Err("Ref2VA requires at least one reference".into());
    }
    let mut images = 0usize;
    let mut videos = 0usize;
    let mut audios = 0usize;
    for r in refs {
        match r.kind {
            ReferenceKind::Image => images += 1,
            ReferenceKind::Video => videos += 1,
            ReferenceKind::Audio => audios += 1,
        }
    }
    if images > MAX_REFERENCE_IMAGES {
        return Err(format!("at most {MAX_REFERENCE_IMAGES} image references"));
    }
    if videos > MAX_REFERENCE_VIDEOS {
        return Err(format!("at most {MAX_REFERENCE_VIDEOS} video references"));
    }
    if audios > MAX_REFERENCE_AUDIOS {
        return Err(format!("at most {MAX_REFERENCE_AUDIOS} audio references"));
    }
    if refs.len() > MAX_REFERENCES {
        return Err(format!("at most {MAX_REFERENCES} references total"));
    }
    if audios == refs.len() {
        return Err("audio references must be paired with at least one image or video".into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn short_edge_2048_square() {
        let (h, w) = resolve_reference_image_size(1024, 1024).unwrap();
        assert_eq!((h, w), (2048, 2048));
    }

    #[test]
    fn short_edge_wide() {
        let (h, w) = resolve_reference_image_size(1920, 1080).unwrap();
        assert_eq!(h.min(w), 2048);
        assert_eq!(h % 32, 0);
        assert_eq!(w % 32, 0);
    }

    #[test]
    fn trim_reference_snaps_to_17n_plus_5() {
        assert_eq!(trim_reference_num_frames(1).unwrap(), 22);
        assert_eq!(trim_reference_num_frames(22).unwrap(), 22);
        assert_eq!(trim_reference_num_frames(50).unwrap(), 39);
        assert_eq!(trim_reference_num_frames(125).unwrap(), 124);
    }

    #[test]
    fn resample_doubles_at_half_rate() {
        let (h, w, t) = (2, 2, 3);
        let mut frames = vec![0u8; t * h * w * 3];
        for i in 0..t {
            frames[i * h * w * 3] = i as u8 + 1;
        }
        let (out, n) = resample_reference_frames(&frames, t, h, w, 12.0).unwrap();
        assert_eq!(n, 6);
        assert_eq!(out[0], 1);
        assert_eq!(out[h * w * 3], 1);
        assert_eq!(out[2 * h * w * 3], 2);
    }

    #[test]
    fn rejects_audio_only() {
        let refs = vec![H3ReferenceSpec {
            path: "a.wav".into(),
            kind: ReferenceKind::Audio,
        }];
        assert!(validate_references(&refs).is_err());
    }

    #[test]
    fn accepts_image_and_audio() {
        let refs = vec![
            H3ReferenceSpec {
                path: "a.png".into(),
                kind: ReferenceKind::Image,
            },
            H3ReferenceSpec {
                path: "b.wav".into(),
                kind: ReferenceKind::Audio,
            },
        ];
        validate_references(&refs).unwrap();
    }
}
