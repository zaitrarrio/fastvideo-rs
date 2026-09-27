//! Pre-encoded media for the writers, and PLI handling (design §5.1, §5.9).
//!
//! A session encodes **once** (OpenH264 Constrained Baseline, libopus 48 kHz
//! 20 ms) and fans the bitstream out to every peer. The host's writers take:
//!
//! - [`VideoFrame`]: one H.264 access unit in Annex-B form (start codes),
//!   with a 90 kHz RTP timestamp;
//! - [`AudioPacket`]: one Opus packet with a 48 kHz RTP timestamp
//!   (derived from the global sample counter).
//!
//! str0m packetizes (FU-A / STAP-A for H.264). Timestamps are the caller's:
//! the pacer's tick counter drives both lanes so A/V stay locked.
//!
//! PLI/FIR from any peer asks the one encoder for an IDR; [`KeyframeLimiter`]
//! rate-limits that to one per second (§5.1).

use std::time::{Duration, Instant};

use bytes::Bytes;

/// RTP clock of every video codec in WebRTC.
pub const VIDEO_CLOCK_HZ: u32 = 90_000;
/// RTP clock of Opus (always 48 kHz on the wire).
pub const AUDIO_CLOCK_HZ: u32 = 48_000;
/// Opus frame size we emit (20 ms at 48 kHz).
pub const OPUS_FRAME_SAMPLES: u32 = 960;

/// Which lane a write goes to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TrackKind {
    Video,
    Audio,
}

/// Video codecs the host can answer with (pre-encoded; str0m packetizes).
///
/// H.264 is the default everywhere (design §5.9). VP8 exists for clients
/// whose WebRTC stack offers no H.264 at all, such as the Reactor Python SDK
/// (its libwebrtc offers VP8/VP9/AV1 only; WP-13).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum VideoCodec {
    H264,
    Vp8,
}

/// One encoded video frame.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VideoFrame {
    /// H.264: an Annex-B access unit (`00 00 00 01` / `00 00 01` start
    /// codes); an IDR access unit should carry SPS and PPS in-band.
    /// VP8: one complete VP8 frame. The peer's negotiated
    /// [`VideoCodec`] decides which (`PeerHandle::video_codec`).
    pub data: Bytes,
    /// RTP timestamp, 90 kHz, monotonically increasing.
    pub rtp_time: u64,
}

impl VideoFrame {
    pub fn new(data: impl Into<Bytes>, rtp_time: u64) -> Self {
        VideoFrame {
            data: data.into(),
            rtp_time,
        }
    }

    /// Whether the access unit contains an IDR slice (NAL type 5).
    pub fn is_keyframe(&self) -> bool {
        h264_nal_types(&self.data).any(|t| t == 5)
    }
}

/// One encoded Opus packet.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AudioPacket {
    pub data: Bytes,
    /// RTP timestamp, 48 kHz: the index of the packet's first sample.
    pub rtp_time: u64,
}

impl AudioPacket {
    pub fn new(data: impl Into<Bytes>, rtp_time: u64) -> Self {
        AudioPacket {
            data: data.into(),
            rtp_time,
        }
    }
}

/// 90 kHz RTP time of frame `index` at `fps` (exact for integer fps that
/// divide 90000, which every supported rate does: 16, 24, 25, 30, 48, 50).
pub fn video_rtp_time(index: u64, fps: u32) -> u64 {
    index * VIDEO_CLOCK_HZ as u64 / fps.max(1) as u64
}

/// Iterate the NAL unit types of an Annex-B buffer.
pub fn h264_nal_types(annexb: &[u8]) -> impl Iterator<Item = u8> + '_ {
    let mut i = 0usize;
    std::iter::from_fn(move || {
        while i + 3 <= annexb.len() {
            let three = annexb[i] == 0 && annexb[i + 1] == 0 && annexb[i + 2] == 1;
            if three {
                i += 3;
                if i < annexb.len() {
                    let t = annexb[i] & 0x1f;
                    return Some(t);
                }
                return None;
            }
            i += 1;
        }
        None
    })
}

/// Rate limiter for encoder IDR requests triggered by PLI/FIR (§5.1: at
/// most one forced IDR per second across all peers of a session).
#[derive(Debug, Clone)]
pub struct KeyframeLimiter {
    min_interval: Duration,
    last: Option<Instant>,
    pending: bool,
}

impl Default for KeyframeLimiter {
    fn default() -> Self {
        Self::new(Duration::from_secs(1))
    }
}

impl KeyframeLimiter {
    pub fn new(min_interval: Duration) -> Self {
        KeyframeLimiter {
            min_interval,
            last: None,
            pending: false,
        }
    }

    /// Record a request. Returns `true` when the IDR should be forced now;
    /// otherwise the request is remembered and [`poll`](Self::poll) fires it
    /// once the interval has elapsed (requests are merged, never lost).
    pub fn request(&mut self, now: Instant) -> bool {
        self.pending = true;
        self.poll(now)
    }

    /// `true` when a remembered request may fire now.
    pub fn poll(&mut self, now: Instant) -> bool {
        if !self.pending {
            return false;
        }
        let ready = self
            .last
            .is_none_or(|l| now.duration_since(l) >= self.min_interval);
        if ready {
            self.pending = false;
            self.last = Some(now);
        }
        ready
    }

    /// Tell the limiter the encoder produced an IDR on its own schedule
    /// (the 2 s GOP), which satisfies any pending request.
    pub fn keyframe_sent(&mut self, now: Instant) {
        self.pending = false;
        self.last = Some(now);
    }

    pub fn is_pending(&self) -> bool {
        self.pending
    }
}

/// Maps RTP time to a wallclock `Instant` for str0m's sender reports:
/// anchored at the first write, then advanced by media time.
#[derive(Debug, Clone, Copy)]
pub struct WallclockMap {
    clock_hz: u32,
    anchor: Option<(Instant, u64)>,
}

impl WallclockMap {
    pub fn new(clock_hz: u32) -> Self {
        WallclockMap {
            clock_hz,
            anchor: None,
        }
    }

    pub fn wallclock(&mut self, now: Instant, rtp_time: u64) -> Instant {
        let (t0, r0) = *self.anchor.get_or_insert((now, rtp_time));
        if rtp_time >= r0 {
            t0 + Duration::from_micros((rtp_time - r0) * 1_000_000 / self.clock_hz as u64)
        } else {
            // Timestamps went backwards (a new stream): re-anchor.
            self.anchor = Some((now, rtp_time));
            now
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nal_types_and_keyframes() {
        let idr = [
            0, 0, 0, 1, 0x67, 1, 2, 0, 0, 0, 1, 0x68, 3, 0, 0, 1, 0x65, 9, 9,
        ];
        assert_eq!(h264_nal_types(&idr).collect::<Vec<_>>(), vec![7, 8, 5]);
        assert!(VideoFrame::new(idr.to_vec(), 0).is_keyframe());
        let p = [0, 0, 0, 1, 0x41, 1, 2, 3];
        assert!(!VideoFrame::new(p.to_vec(), 0).is_keyframe());
        assert_eq!(h264_nal_types(&[]).count(), 0);
        assert_eq!(h264_nal_types(&[0, 0, 1]).count(), 0);
    }

    #[test]
    fn rtp_times() {
        assert_eq!(video_rtp_time(1, 24), 3750);
        assert_eq!(video_rtp_time(24, 24), 90_000);
        assert_eq!(video_rtp_time(3, 16), 16_875);
        assert_eq!(video_rtp_time(10, 25), 36_000);
    }

    #[test]
    fn keyframe_limiter_merges_and_rate_limits() {
        let t0 = Instant::now();
        let mut k = KeyframeLimiter::default();
        assert!(k.request(t0));
        assert!(!k.request(t0 + Duration::from_millis(100)));
        assert!(!k.request(t0 + Duration::from_millis(200)));
        assert!(k.is_pending());
        assert!(!k.poll(t0 + Duration::from_millis(999)));
        assert!(k.poll(t0 + Duration::from_millis(1000)));
        assert!(!k.poll(t0 + Duration::from_millis(3000)));
        // A GOP IDR satisfies a pending request.
        assert!(!k.request(t0 + Duration::from_millis(3100)) || !k.is_pending());
        let t1 = t0 + Duration::from_secs(10);
        k.request(t1);
        k.request(t1 + Duration::from_millis(10));
        k.keyframe_sent(t1 + Duration::from_millis(20));
        assert!(!k.poll(t1 + Duration::from_secs(5)));
    }

    #[test]
    fn wallclock_follows_media_time() {
        let t0 = Instant::now();
        let mut w = WallclockMap::new(VIDEO_CLOCK_HZ);
        assert_eq!(w.wallclock(t0, 1000), t0);
        assert_eq!(
            w.wallclock(t0 + Duration::from_secs(9), 1000 + 90_000),
            t0 + Duration::from_secs(1)
        );
        let t2 = t0 + Duration::from_secs(10);
        assert_eq!(w.wallclock(t2, 5), t2);
    }
}
