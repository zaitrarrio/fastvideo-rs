//! MiniMax-H3 Ref2VA reference media: limits, canvas math, and prepared
//! geometry (FastVideo `pipelines/basic/minimax_h3/reference.py`).
//!
//! Video/audio decode stays at the call site (cudarc / CLI); this module is the
//! host contract the packed layout and encoders consume.

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
