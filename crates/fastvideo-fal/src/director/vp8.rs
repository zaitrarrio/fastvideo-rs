//! Intra-only VP8, the fallback for offers without H.264 (open-source
//! Chromium builds, which is what headless test browsers are) when ffmpeg
//! has no `libvpx` (otherwise `fastvideo_media::vp8` encodes inter-frame
//! VP8): every frame is a keyframe, encoded in process by libwebp (a lossy
//! WebP image *is* one VP8 key frame in a RIFF container). Heavier on
//! bitrate than an inter-frame encoder, but it needs no external tool and
//! every WebRTC stack decodes VP8. H.264 stays the codec of every offer that
//! has it (design §5.9); the Reactor runtime has the same fallback.

use bytes::Bytes;
use fastvideo_protocol::RgbFrame;

/// The VP8 frame inside a simple-format lossy WebP file
/// (`RIFF....WEBPVP8 <len><frame>`).
pub fn webp_to_vp8(webp: &[u8]) -> Result<&[u8], String> {
    if webp.len() < 20 || &webp[0..4] != b"RIFF" || &webp[8..12] != b"WEBP" {
        return Err("not a WebP file".into());
    }
    if &webp[12..16] != b"VP8 " {
        return Err(format!("unexpected WebP chunk {:?} (want lossy `VP8 `)", String::from_utf8_lossy(&webp[12..16])));
    }
    let len = u32::from_le_bytes([webp[16], webp[17], webp[18], webp[19]]) as usize;
    webp.get(20..20 + len).ok_or_else(|| "truncated VP8 chunk".to_string())
}

/// Whether a VP8 frame is a key frame (RFC 6386 §9.1: bit 0 clear).
pub fn is_keyframe(frame: &[u8]) -> bool {
    frame.first().is_some_and(|b| b & 1 == 0)
}

/// libwebp quality (0-100) for a target bitrate at `w·h·fps`: intra-only
/// frames need roughly 0.1-0.4 bit/pixel at a useful quality.
pub fn quality_for(bitrate_bps: u32, w: u32, h: u32, fps: u32) -> f32 {
    let bpp = f64::from(bitrate_bps) / (f64::from(w.max(1)) * f64::from(h.max(1)) * f64::from(fps.max(1)));
    (bpp * 250.0).clamp(20.0, 80.0) as f32
}

/// Intra-only VP8 encoder (the libwebp part needs feature `director`).
pub struct Vp8Encoder {
    dims: (u32, u32),
    #[cfg_attr(not(feature = "director"), allow(dead_code))]
    quality: f32,
}

impl Vp8Encoder {
    pub fn new(w: u32, h: u32, quality: f32) -> Result<Self, String> {
        if w == 0 || h == 0 || w > 16383 || h > 16383 {
            return Err(format!("VP8 cannot carry {w}x{h}"));
        }
        Ok(Self { dims: (w, h), quality })
    }

    pub fn dims(&self) -> (u32, u32) {
        self.dims
    }

    /// One key frame.
    #[cfg(feature = "director")]
    pub fn encode(&mut self, f: &RgbFrame) -> Result<Bytes, String> {
        if (f.width, f.height) != self.dims {
            return Err(format!("frame {}x{} does not match the encoder {}x{}", f.width, f.height, self.dims.0, self.dims.1));
        }
        let mut cfg = webp::WebPConfig::new().map_err(|_| "libwebp config".to_string())?;
        cfg.lossless = 0;
        cfg.quality = self.quality;
        // Fastest method: real-time pacing matters more than size.
        cfg.method = 0;
        let mem = webp::Encoder::from_rgb(&f.data, f.width, f.height)
            .encode_advanced(&cfg)
            .map_err(|e| format!("libwebp: {e:?}"))?;
        Ok(Bytes::copy_from_slice(webp_to_vp8(&mem)?))
    }

    #[cfg(not(feature = "director"))]
    pub fn encode(&mut self, _f: &RgbFrame) -> Result<Bytes, String> {
        Err("built without the `director` feature of fastvideo-fal".into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn container_and_quality() {
        let mut w = b"RIFF\0\0\0\0WEBPVP8 ".to_vec();
        w.extend_from_slice(&3u32.to_le_bytes());
        w.extend_from_slice(&[0x10, 2, 3]);
        assert_eq!(webp_to_vp8(&w).unwrap(), &[0x10, 2, 3]);
        assert!(webp_to_vp8(b"RIFF\0\0\0\0WEBPVP8L\0\0\0\0").is_err());
        assert!(is_keyframe(&[0x10]) && !is_keyframe(&[0x11]));
        assert!((20.0..=80.0).contains(&quality_for(2_500_000, 832, 480, 24)));
    }

    #[cfg(feature = "director")]
    #[test]
    fn encodes_a_key_frame() {
        let f = RgbFrame::black(64, 48, 0);
        let mut e = Vp8Encoder::new(64, 48, 50.0).unwrap();
        let v = e.encode(&f).unwrap();
        assert!(is_keyframe(&v));
        // Key frame start code (RFC 6386 §9.1).
        assert_eq!(&v[3..6], &[0x9d, 0x01, 0x2a]);
    }
}
