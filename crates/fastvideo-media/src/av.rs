//! Raw A/V buffers (design §3.5).
//!
//! [`RgbFrame`] and [`Pcm`] are `fastvideo_protocol`'s types, re-exported so
//! the engine, the pacer and the encoders all pass the same buffers.
//! [`AvCheck`] adds the shape checks the media code needs, returning this
//! crate's error type, plus PCM byte conversions.

pub use fastvideo_protocol::{Pcm, RgbFrame};

use crate::error::{MediaError, Result};

/// Shape validation for raw buffers.
pub trait AvCheck {
    /// `Ok` when the buffer is well formed.
    fn check(&self) -> Result<()>;
}

impl AvCheck for RgbFrame {
    fn check(&self) -> Result<()> {
        if self.width == 0 || self.height == 0 {
            return Err(MediaError::invalid("frame has a zero dimension"));
        }
        let want = RgbFrame::byte_len(self.width, self.height);
        if self.data.len() != want {
            return Err(MediaError::invalid(format!(
                "frame {}x{} needs {want} RGB24 bytes, got {}",
                self.width,
                self.height,
                self.data.len()
            )));
        }
        Ok(())
    }
}

impl AvCheck for Pcm {
    fn check(&self) -> Result<()> {
        if self.rate == 0 || self.channels == 0 {
            return Err(MediaError::invalid("pcm needs a nonzero rate and channel count"));
        }
        if self.samples.len() % self.channels as usize != 0 {
            return Err(MediaError::invalid(format!(
                "{} interleaved samples do not divide into {} channels",
                self.samples.len(),
                self.channels
            )));
        }
        Ok(())
    }
}

/// Build a checked frame from raw RGB24 bytes.
pub fn frame(width: u32, height: u32, data: impl Into<bytes::Bytes>, index: u64) -> Result<RgbFrame> {
    let f = RgbFrame { width, height, data: data.into(), index };
    f.check()?;
    Ok(f)
}

/// Build checked PCM.
pub fn pcm(rate: u32, channels: u8, samples: Vec<f32>) -> Result<Pcm> {
    let p = Pcm::new(rate, channels, samples);
    p.check()?;
    Ok(p)
}

/// Convert interleaved f32 samples to little-endian s16 bytes (the ffmpeg
/// `s16le` and WebRTC PCM format), clamped and rounded.
pub fn f32_to_s16le(samples: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(samples.len() * 2);
    for &s in samples {
        let v = (s.clamp(-1.0, 1.0) * 32767.0).round() as i16;
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

/// Little-endian f32 bytes (ffmpeg `f32le`).
pub fn f32_to_f32le(samples: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(samples.len() * 4);
    for &s in samples {
        out.extend_from_slice(&s.to_le_bytes());
    }
    out
}

/// Parse little-endian f32 bytes (a trailing partial sample is ignored).
pub fn f32le_to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frame_validation() {
        assert!(frame(2, 2, vec![0u8; 12], 0).is_ok());
        assert!(frame(2, 2, vec![0u8; 11], 0).is_err());
        assert!(frame(0, 2, Vec::<u8>::new(), 0).is_err());
        assert_eq!(RgbFrame::black(4, 2, 7).data.len(), 24);
        assert!(RgbFrame::black(4, 2, 7).check().is_ok());
    }

    #[test]
    fn pcm_frames_and_s16() {
        let p = pcm(48_000, 2, vec![0.0, 1.0, -1.0, 2.0]).unwrap();
        assert_eq!(p.frames(), 2);
        let b = f32_to_s16le(&p.samples);
        assert_eq!(i16::from_le_bytes([b[2], b[3]]), 32767);
        assert_eq!(i16::from_le_bytes([b[4], b[5]]), -32767);
        assert_eq!(i16::from_le_bytes([b[6], b[7]]), 32767); // clamped
        assert!(pcm(48_000, 2, vec![0.0; 3]).is_err());
        assert!(pcm(0, 2, vec![]).is_err());
        assert_eq!(f32le_to_f32(&f32_to_f32le(&[0.5, -0.25])), vec![0.5, -0.25]);
    }
}
