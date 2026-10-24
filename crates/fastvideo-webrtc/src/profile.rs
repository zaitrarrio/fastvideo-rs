//! Encode profiles per sink (design §0 decision 2, §5.8, risk R3).
//!
//! Every sink takes the model's **native** resolution and fps
//! (docs/serve/stream-outputs.md). What differs is the H.264 level we
//! **offer** in SDP:
//!
//! - Cloudflare Stream's WHIP ingest documents H.264 Constrained Baseline
//!   `profile-level-id=42e01f` (level 3.1) in the offer, so that is what we
//!   offer. The bitstream itself carries the level its resolution needs
//!   (1344x768@24 is level 3.2, 1920x1080@30 level 4.0): what browsers do
//!   (libwebrtc offers `42e01f` and sends 1080p), and Cloudflare publishes
//!   no resolution cap (its own FFmpeg example is 1920x1080@30). The old
//!   1280x720 cap here assumed the offered level bounds the stream; it was
//!   never a Cloudflare requirement.
//! - MediaMTX (self-hosted relay) and peer WebRTC: level 4.0 offered.
//!
//! The media layer encodes the native canvas at the smallest level that
//! fits, never below the offered one
//! (`fastvideo_media::video::H264Config::for_publish`).

use serde::{Deserialize, Serialize};

/// H.264 levels we offer.
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

/// What a sink is offered: the H.264 level in SDP. The resolution is
/// always native.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct EncodeProfile {
    /// The level of the offered `profile-level-id`. The encoder never goes
    /// below it, and above it when the canvas needs more.
    pub h264_level: H264Level,
}

impl EncodeProfile {
    /// Cloudflare WHIP: `42e01f` offered, native resolution.
    pub const CLOUDFLARE: EncodeProfile = EncodeProfile {
        h264_level: H264Level::L3_1,
    };
    /// MediaMTX WHIP and peer WebRTC: `42e028` offered, native resolution.
    pub const NATIVE: EncodeProfile = EncodeProfile {
        h264_level: H264Level::L4_0,
    };

    /// The encode size for a `width`x`height` source: native, rounded down
    /// to even dimensions (4:2:0).
    pub fn output_size(&self, width: u32, height: u32) -> (u32, u32) {
        ((width & !1).max(2), (height & !1).max(2))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cloudflare_offers_3_1_and_keeps_native_resolution() {
        let p = EncodeProfile::CLOUDFLARE;
        assert_eq!(p.h264_level.constrained_baseline_plid(), 0x42e01f);
        for (w, h) in [
            (1344, 768),
            (1920, 1080),
            (768, 1344),
            (832, 480),
            (1280, 704),
        ] {
            assert_eq!(p.output_size(w, h), (w, h));
        }
        // The offered level does not bound the stream: native H3 needs 3.2.
        assert!(!H264Level::L3_1.fits(1344, 768, 24));
        assert!(H264Level::L4_0.fits(1344, 768, 30));
        assert!(H264Level::L4_0.fits(1920, 1080, 30));
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
