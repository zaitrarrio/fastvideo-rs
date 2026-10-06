//! Raw A/V buffers: `RgbFrame`, `Pcm` (design §3.5).
//!
//! Cheap to clone (`Bytes` / `Arc`), so the engine can hand the same buffers
//! to the pacer, a recorder and tests.

use std::sync::Arc;

use crate::error::ApiError;

/// One RGB24 video frame (row-major, no padding).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RgbFrame {
    pub width: u32,
    pub height: u32,
    /// RGB24, `width * height * 3` bytes.
    pub data: bytes::Bytes,
    /// Frame index within its clip or session.
    pub index: u64,
}

impl RgbFrame {
    /// Checks the buffer length.
    pub fn new(width: u32, height: u32, data: bytes::Bytes, index: u64) -> Result<Self, ApiError> {
        let want = Self::byte_len(width, height);
        if data.len() != want {
            return Err(ApiError::internal(format!(
                "RGB24 frame {width}x{height} needs {want} bytes, got {}",
                data.len()
            )));
        }
        Ok(Self {
            width,
            height,
            data,
            index,
        })
    }
    /// A frame filled with one colour.
    pub fn solid(width: u32, height: u32, rgb: [u8; 3], index: u64) -> Self {
        let n = (width as usize) * (height as usize);
        let mut v = Vec::with_capacity(n * 3);
        for _ in 0..n {
            v.extend_from_slice(&rgb);
        }
        Self {
            width,
            height,
            data: v.into(),
            index,
        }
    }
    /// A black frame (Reactor start-of-connection / flush).
    pub fn black(width: u32, height: u32, index: u64) -> Self {
        Self::solid(width, height, [0, 0, 0], index)
    }
    pub fn byte_len(width: u32, height: u32) -> usize {
        width as usize * height as usize * 3
    }
    /// The centre `(w, h)` of the frame (pad-and-crop canvases: LTX
    /// generates 1920x1088 for 1920x1080). A no-op when the frame already is
    /// that size or is smaller on either side.
    pub fn crop_center(&self, w: u32, h: u32) -> RgbFrame {
        if (self.width, self.height) == (w, h) || w > self.width || h > self.height {
            return self.clone();
        }
        let x0 = ((self.width - w) / 2) as usize;
        let y0 = ((self.height - h) / 2) as usize;
        let stride = self.width as usize * 3;
        let mut out = Vec::with_capacity(Self::byte_len(w, h));
        for y in 0..h as usize {
            let row = (y0 + y) * stride + x0 * 3;
            out.extend_from_slice(&self.data[row..row + w as usize * 3]);
        }
        RgbFrame { width: w, height: h, data: out.into(), index: self.index }
    }
    /// The pixel at `(x, y)`.
    pub fn pixel(&self, x: u32, y: u32) -> Option<[u8; 3]> {
        if x >= self.width || y >= self.height {
            return None;
        }
        let i = (y as usize * self.width as usize + x as usize) * 3;
        self.data.get(i..i + 3).map(|p| [p[0], p[1], p[2]])
    }
}

/// Interleaved f32 PCM.
#[derive(Clone, Debug, PartialEq)]
pub struct Pcm {
    pub rate: u32,
    pub channels: u8,
    /// Interleaved samples, `frames * channels` long.
    pub samples: Arc<[f32]>,
}

impl Pcm {
    pub fn new(rate: u32, channels: u8, samples: impl Into<Arc<[f32]>>) -> Self {
        Self {
            rate,
            channels,
            samples: samples.into(),
        }
    }
    /// `frames` sample frames of silence.
    pub fn silence(rate: u32, channels: u8, frames: usize) -> Self {
        Self::new(rate, channels, vec![0.0; frames * channels as usize])
    }
    /// Sample frames (samples per channel).
    pub fn frames(&self) -> usize {
        if self.channels == 0 {
            0
        } else {
            self.samples.len() / self.channels as usize
        }
    }
    pub fn duration_s(&self) -> f64 {
        if self.rate == 0 {
            0.0
        } else {
            self.frames() as f64 / self.rate as f64
        }
    }
    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }
    /// Mean downmix to mono (Reactor, design §5.3).
    pub fn to_mono(&self) -> Pcm {
        let c = self.channels as usize;
        if c <= 1 {
            return self.clone();
        }
        let mono: Vec<f32> = self
            .samples
            .chunks_exact(c)
            .map(|f| f.iter().sum::<f32>() / c as f32)
            .collect();
        Pcm::new(self.rate, 1, mono)
    }
    /// Duplicates a mono signal to `channels`.
    pub fn upmix(&self, channels: u8) -> Pcm {
        if self.channels != 1 || channels <= 1 {
            return self.clone();
        }
        let c = channels as usize;
        let mut v = Vec::with_capacity(self.samples.len() * c);
        for &s in self.samples.iter() {
            v.extend(std::iter::repeat_n(s, c));
        }
        Pcm::new(self.rate, channels, v)
    }
}
