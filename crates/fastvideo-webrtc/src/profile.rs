//! Encode profiles per sink (design §0 decision 2, §5.8, risk R3).
//!
//! WHIP to Cloudflare needs H.264 Constrained Baseline level 3.1, which by the
//! book cannot carry 1344x768, so those streams are scaled to fit 1280x720.
//! MediaMTX (self-hosted relay) and peer WebRTC take native resolution at
//! level 4.0. The media layer asks the sink for its [`EncodeProfile`] and
//! scales and encodes to it; the WebRTC layer advertises the same level in SDP.

use serde::{Deserialize, Serialize};

/// H.264 levels we encode at.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum H264Level {
    /// Level 3.1 (`level_idc` 31): max 3600 macroblocks per frame.
    #[serde(rename = "3.1")]
    L3_1,
    /// Level 4.0 (`level_idc` 40): max 8192 macroblocks per frame.
    #[serde(rename = "4.0")]
    L4_0,
}

impl H264Level {
    /// `level_idc` as written in SPS and in `profile-level-id`.
    pub fn idc(self) -> u8 {
        match self {
            H264Level::L3_1 => 31,
            H264Level::L4_0 => 40,
        }
    }

    /// MaxFS (H.264 Table A-1), in 16x16 macroblocks.
    pub fn max_frame_mbs(self) -> u32 {
        match self {
            H264Level::L3_1 => 3600,
            H264Level::L4_0 => 8192,
        }
    }

    /// MaxMBPS (H.264 Table A-1), macroblocks per second.
    pub fn max_mbs_per_sec(self) -> u32 {
        match self {
            H264Level::L3_1 => 108_000,
            H264Level::L4_0 => 245_760,
        }
    }

    /// `profile-level-id` for Constrained Baseline (`42e0xx`).
    pub fn constrained_baseline_plid(self) -> u32 {
        0x42e000 | self.idc() as u32
    }

    /// `profile-level-id` for Baseline (`4200xx`).
    pub fn baseline_plid(self) -> u32 {
        0x420000 | self.idc() as u32
    }

    /// Whether `width`x`height` at `fps` is within this level.
    pub fn fits(self, width: u32, height: u32, fps: u32) -> bool {
        let mbs = width.div_ceil(16) * height.div_ceil(16);
        mbs <= self.max_frame_mbs() && mbs * fps <= self.max_mbs_per_sec()
    }
}

/// What a sink accepts: an optional bounding box and the H.264 level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct EncodeProfile {
    /// Output must fit inside `(width, height)`; `None`: native resolution.
    pub max_size: Option<(u32, u32)>,
    pub h264_level: H264Level,
}

impl EncodeProfile {
    /// Cloudflare WHIP: fit 1280x720, level 3.1.
    pub const CLOUDFLARE: EncodeProfile = EncodeProfile {
        max_size: Some((1280, 720)),
        h264_level: H264Level::L3_1,
    };
    /// MediaMTX WHIP and peer WebRTC: native resolution, level 4.0.
    pub const NATIVE: EncodeProfile = EncodeProfile {
        max_size: None,
        h264_level: H264Level::L4_0,
    };

    /// The encode size for a `width`x`height` source: scaled down (never up)
    /// to fit `max_size` with the aspect ratio kept, rounded down to even
    /// dimensions (4:2:0). 1344x768 under `CLOUDFLARE` gives 1260x720.
    pub fn output_size(&self, width: u32, height: u32) -> (u32, u32) {
        let (w, h) = match self.max_size {
            Some((mw, mh)) if width > mw || height > mh => {
                // Compare mw/width against mh/height without floats.
                if (mw as u64) * (height as u64) <= (mh as u64) * (width as u64) {
                    (mw, ((height as u64 * mw as u64) / width as u64) as u32)
                } else {
                    (((width as u64 * mh as u64) / height as u64) as u32, mh)
                }
            }
            _ => (width, height),
        };
        ((w & !1).max(2), (h & !1).max(2))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cloudflare_scales_h3_to_720p_and_fits_level_3_1() {
        let p = EncodeProfile::CLOUDFLARE;
        let (w, h) = p.output_size(1344, 768);
        assert_eq!((w, h), (1260, 720));
        assert!(p.h264_level.fits(w, h, 24) && p.h264_level.fits(w, h, 30));
        // Native H3 does not fit 3.1 (risk R3) but fits 4.0.
        assert!(!H264Level::L3_1.fits(1344, 768, 24));
        assert!(H264Level::L4_0.fits(1344, 768, 30));
        // Already small, 16:9, portrait: never upscaled, aspect kept.
        assert_eq!(p.output_size(832, 480), (832, 480));
        assert_eq!(p.output_size(1920, 1080), (1280, 720));
        assert_eq!(p.output_size(768, 1344), (410, 720));
    }

    #[test]
    fn native_keeps_resolution() {
        assert_eq!(EncodeProfile::NATIVE.output_size(1344, 768), (1344, 768));
        assert_eq!(EncodeProfile::NATIVE.output_size(1281, 721), (1280, 720));
        assert_eq!(H264Level::L4_0.constrained_baseline_plid(), 0x42e028);
        assert_eq!(H264Level::L3_1.constrained_baseline_plid(), 0x42e01f);
        assert_eq!(H264Level::L3_1.baseline_plid(), 0x42001f);
    }
}
