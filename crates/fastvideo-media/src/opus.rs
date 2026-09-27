//! Opus framing and encoding (design §5.3, §5.9).
//!
//! - [`OpusFramer`] (always built) cuts interleaved 48 kHz PCM into fixed
//!   Opus frames and stamps each with an RTP timestamp taken from the global
//!   sample counter, so audio time can never drift from the pacer tick count.
//! - [`OpusEncoder`] (feature `opus`, libopus via `audiopus`) encodes them.
//! - [`toc`] reads a packet's TOC byte, which the tests use to prove the wire
//!   format (mono/stereo, frame duration) without a decoder.
//!
//! Presets:
//!
//! | Preset | Channels | Frame | Bitrate | Why |
//! |---|---|---|---|---|
//! | [`OpusConfig::reactor`] | mono | 10 ms (480 samples) | 64 kb/s | Reactor RT pushes 48 kHz mono in 10 ms frames (reactor §4.5) |
//! | [`OpusConfig::wma`] / [`OpusConfig::whip`] | stereo | 20 ms (960 samples) | 96 kb/s (or `audio_bitrate`) | §5.3, §5.9 |
//!
//! At 24 fps one pacer tick is 2000 samples, i.e. 4⅙ 10 ms frames or 2⅙ 20 ms
//! frames; the framer carries the remainder, so the packet stream is
//! continuous and exactly one sample-clock long.

use bytes::Bytes;

use crate::clock::AudioRtpClock;
use crate::error::{MediaError, Result};
use crate::lockstep::WIRE_RATE;

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct OpusConfig {
    pub channels: u8,
    /// 10 or 20.
    pub frame_ms: u8,
    pub bitrate_bps: u32,
}

impl OpusConfig {
    /// Reactor local runtime: 48 kHz mono, 10 ms frames.
    pub fn reactor() -> Self {
        Self { channels: 1, frame_ms: 10, bitrate_bps: 64_000 }
    }

    /// fal director (WMA): stereo, 20 ms, `audio_bitrate` 96/128/192k (default 96k).
    pub fn wma(bitrate_bps: Option<u32>) -> Self {
        Self { channels: 2, frame_ms: 20, bitrate_bps: bitrate_bps.unwrap_or(96_000) }
    }

    /// WHIP publish: stereo, 20 ms, 96k.
    pub fn whip() -> Self {
        Self::wma(None)
    }

    /// Samples per channel in one frame (480 for 10 ms, 960 for 20 ms).
    pub fn frame_samples(&self) -> usize {
        WIRE_RATE as usize * self.frame_ms as usize / 1000
    }

    pub fn validate(&self) -> Result<()> {
        if !matches!(self.channels, 1 | 2) {
            return Err(MediaError::invalid("opus channels must be 1 or 2"));
        }
        if !matches!(self.frame_ms, 10 | 20) {
            return Err(MediaError::invalid("opus frame_ms must be 10 or 20"));
        }
        if !(6_000..=510_000).contains(&self.bitrate_bps) {
            return Err(MediaError::invalid("opus bitrate must be within 6..510 kb/s"));
        }
        Ok(())
    }
}

/// One PCM frame ready for the encoder.
#[derive(Debug, Clone, PartialEq)]
pub struct PcmFrame {
    /// Interleaved, exactly `frame_samples · channels` long.
    pub samples: Vec<f32>,
    pub rtp_ts: u32,
    /// Sample-counter position of the frame's first sample.
    pub offset: u64,
}

/// Cuts a continuous PCM stream into Opus-sized frames.
#[derive(Debug, Clone)]
pub struct OpusFramer {
    cfg: OpusConfig,
    buf: Vec<f32>,
    clock: AudioRtpClock,
}

impl OpusFramer {
    pub fn new(cfg: OpusConfig, rtp_base: u32) -> Result<Self> {
        cfg.validate()?;
        Ok(Self { cfg, buf: Vec::new(), clock: AudioRtpClock::new(rtp_base) })
    }

    pub fn config(&self) -> &OpusConfig {
        &self.cfg
    }

    /// Append interleaved 48 kHz PCM with the configured channel count.
    pub fn push(&mut self, samples: &[f32]) -> Result<()> {
        if samples.len() % self.cfg.channels as usize != 0 {
            return Err(MediaError::invalid("pcm is not a whole number of sample frames"));
        }
        self.buf.extend_from_slice(samples);
        Ok(())
    }

    /// The next complete frame, if one is buffered.
    pub fn next_frame(&mut self) -> Option<PcmFrame> {
        let n = self.cfg.frame_samples() * self.cfg.channels as usize;
        if self.buf.len() < n {
            return None;
        }
        let samples: Vec<f32> = self.buf.drain(..n).collect();
        let offset = self.clock.samples();
        let rtp_ts = self.clock.advance(self.cfg.frame_samples() as u64);
        Some(PcmFrame { samples, rtp_ts, offset })
    }

    /// Samples per channel waiting for a full frame.
    pub fn pending(&self) -> usize {
        self.buf.len() / self.cfg.channels as usize
    }

    /// Total samples per channel framed so far.
    pub fn framed_samples(&self) -> u64 {
        self.clock.samples()
    }
}

/// One encoded Opus packet.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OpusPacket {
    pub data: Bytes,
    pub rtp_ts: u32,
    /// Duration in samples per channel at 48 kHz.
    pub samples: u32,
}

/// Opus TOC byte facts (RFC 6716 §3.1).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Toc {
    pub config: u8,
    pub stereo: bool,
    /// Frames in the packet.
    pub frames: u8,
    /// Duration of one frame in samples at 48 kHz.
    pub frame_samples: u32,
}

impl Toc {
    pub fn packet_samples(&self) -> u32 {
        self.frame_samples * u32::from(self.frames)
    }
}

/// Parse a packet's TOC byte.
pub fn toc(packet: &[u8]) -> Result<Toc> {
    let b = *packet.first().ok_or_else(|| MediaError::parse(None, "empty opus packet"))?;
    let config = b >> 3;
    let stereo = b & 0x04 != 0;
    // Frame durations in units of 2.5 ms (120 samples at 48 kHz).
    let units: u32 = match config {
        0..=11 => [4, 8, 16, 24][(config % 4) as usize],
        12..=15 => [4, 8][(config % 2) as usize],
        _ => [1, 2, 4, 8][(config % 4) as usize],
    };
    let frames = match b & 0x03 {
        0 => 1,
        1 | 2 => 2,
        _ => packet.get(1).map(|c| c & 0x3f).ok_or_else(|| MediaError::parse(None, "opus code-3 packet too short"))?,
    };
    Ok(Toc { config, stereo, frames, frame_samples: units * 120 })
}

#[cfg(feature = "opus")]
pub use enc::{OpusDecoder, OpusEncoder};

#[cfg(feature = "opus")]
mod enc {
    use super::*;
    use audiopus::coder::{Decoder, Encoder};
    use audiopus::{packet::Packet, Application, Bitrate, Channels, MutSignals, SampleRate};

    fn channels(n: u8) -> Channels {
        if n == 2 { Channels::Stereo } else { Channels::Mono }
    }

    /// libopus encoder with a built-in framer.
    pub struct OpusEncoder {
        enc: Encoder,
        framer: OpusFramer,
        out: Vec<u8>,
    }

    impl OpusEncoder {
        pub fn new(cfg: OpusConfig, rtp_base: u32) -> Result<Self> {
            let framer = OpusFramer::new(cfg, rtp_base)?;
            let mut enc = Encoder::new(SampleRate::Hz48000, channels(cfg.channels), Application::Audio)
                .map_err(|e| MediaError::Encode(format!("opus init: {e}")))?;
            enc.set_bitrate(Bitrate::BitsPerSecond(cfg.bitrate_bps as i32))
                .map_err(|e| MediaError::Encode(format!("opus bitrate: {e}")))?;
            // Keep the declared channel layout on the wire (no automatic
            // stereo-to-mono decisions for correlated input).
            enc.set_force_channels(channels(cfg.channels))
                .map_err(|e| MediaError::Encode(format!("opus channels: {e}")))?;
            Ok(Self { enc, framer, out: vec![0u8; 4000] })
        }

        pub fn config(&self) -> &OpusConfig {
            self.framer.config()
        }

        /// Encode exactly one frame of interleaved samples.
        pub fn encode_frame(&mut self, samples: &[f32]) -> Result<Bytes> {
            let cfg = *self.framer.config();
            if samples.len() != cfg.frame_samples() * cfg.channels as usize {
                return Err(MediaError::invalid("opus frame has the wrong length"));
            }
            let n = self.enc.encode_float(samples, &mut self.out).map_err(|e| MediaError::Encode(format!("opus: {e}")))?;
            Ok(Bytes::copy_from_slice(&self.out[..n]))
        }

        /// Push PCM and encode every complete frame.
        pub fn push(&mut self, samples: &[f32]) -> Result<Vec<OpusPacket>> {
            self.framer.push(samples)?;
            let mut out = Vec::new();
            while let Some(f) = self.framer.next_frame() {
                let data = self.encode_frame(&f.samples)?;
                out.push(OpusPacket { data, rtp_ts: f.rtp_ts, samples: self.framer.config().frame_samples() as u32 });
            }
            Ok(out)
        }

        pub fn framed_samples(&self) -> u64 {
            self.framer.framed_samples()
        }
    }

    /// libopus decoder (tests and loopback checks).
    pub struct OpusDecoder {
        dec: Decoder,
        channels: u8,
        buf: Vec<f32>,
    }

    impl OpusDecoder {
        pub fn new(ch: u8) -> Result<Self> {
            let dec = Decoder::new(SampleRate::Hz48000, channels(ch)).map_err(|e| MediaError::Decode(e.to_string()))?;
            Ok(Self { dec, channels: ch, buf: vec![0.0; 5760 * 2] })
        }

        /// Decode one packet to interleaved f32.
        pub fn decode(&mut self, packet: &[u8]) -> Result<Vec<f32>> {
            let p = Packet::try_from(packet).map_err(|e| MediaError::Decode(e.to_string()))?;
            let sig = MutSignals::try_from(&mut self.buf[..]).map_err(|e| MediaError::Decode(e.to_string()))?;
            let n = self.dec.decode_float(Some(p), sig, false).map_err(|e| MediaError::Decode(e.to_string()))?;
            Ok(self.buf[..n * self.channels as usize].to_vec())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn presets() {
        assert_eq!(OpusConfig::reactor().frame_samples(), 480);
        assert_eq!(OpusConfig::reactor().channels, 1);
        assert_eq!(OpusConfig::wma(None).frame_samples(), 960);
        assert_eq!(OpusConfig::wma(Some(192_000)).bitrate_bps, 192_000);
        assert!(OpusConfig { channels: 3, ..OpusConfig::whip() }.validate().is_err());
        assert!(OpusConfig { frame_ms: 5, ..OpusConfig::whip() }.validate().is_err());
    }

    #[test]
    fn framer_carries_remainders_and_stamps_from_the_sample_counter() {
        // 24 fps ticks of 2000 samples into 10 ms (480) frames, mono.
        let mut f = OpusFramer::new(OpusConfig::reactor(), 1000).unwrap();
        let mut frames = Vec::new();
        for _ in 0..24 * 60 {
            f.push(&[0.0; 2000]).unwrap();
            while let Some(fr) = f.next_frame() {
                frames.push(fr);
            }
        }
        // 60 s of audio is exactly 6000 10 ms frames, nothing left over.
        assert_eq!(frames.len(), 6000);
        assert_eq!(f.pending(), 0);
        assert_eq!(f.framed_samples(), 2_880_000);
        for (i, fr) in frames.iter().enumerate() {
            assert_eq!(fr.samples.len(), 480);
            assert_eq!(fr.offset, i as u64 * 480);
            assert_eq!(fr.rtp_ts, 1000 + i as u32 * 480);
        }
        // Stereo 20 ms from 16 fps ticks (3000 samples each).
        let mut s = OpusFramer::new(OpusConfig::whip(), 0).unwrap();
        let mut n = 0;
        for _ in 0..16 * 12 {
            s.push(&[0.0; 6000]).unwrap();
            while s.next_frame().is_some() {
                n += 1;
            }
        }
        assert_eq!(n, 600); // 12 s / 20 ms
        assert!(s.push(&[0.0; 3]).is_err());
    }

    #[test]
    fn toc_parsing() {
        // config 30 = CELT FB 10 ms, mono, code 0.
        let t = toc(&[30 << 3]).unwrap();
        assert_eq!((t.config, t.stereo, t.frames, t.frame_samples), (30, false, 1, 480));
        // config 31 = CELT FB 20 ms, stereo.
        let t = toc(&[(31 << 3) | 0x04]).unwrap();
        assert!(t.stereo);
        assert_eq!(t.packet_samples(), 960);
        // SILK WB 20 ms (config 9), code 3 with 3 frames.
        let t = toc(&[(9 << 3) | 0x03, 3]).unwrap();
        assert_eq!(t.packet_samples(), 2880);
        // Hybrid FB 10 ms (config 14).
        assert_eq!(toc(&[14 << 3]).unwrap().frame_samples, 480);
        assert!(toc(&[]).is_err());
    }

    #[cfg(feature = "opus")]
    #[test]
    fn reactor_format_mono_10ms_round_trip() {
        let mut e = OpusEncoder::new(OpusConfig::reactor(), 0).unwrap();
        let mut d = OpusDecoder::new(1).unwrap();
        // 1 s of a 440 Hz tone in 24 fps ticks.
        let tone: Vec<f32> =
            (0..48_000).map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 48_000.0).sin() * 0.3).collect();
        let mut pkts = Vec::new();
        for tick in tone.chunks(2000) {
            pkts.extend(e.push(tick).unwrap());
        }
        assert_eq!(pkts.len(), 100);
        let mut decoded = Vec::new();
        for (i, p) in pkts.iter().enumerate() {
            let t = toc(&p.data).unwrap();
            assert!(!t.stereo, "reactor audio must be mono");
            assert_eq!(t.packet_samples(), 480, "reactor packets must be 10 ms");
            assert_eq!(p.rtp_ts, i as u32 * 480);
            let pcm = d.decode(&p.data).unwrap();
            assert_eq!(pcm.len(), 480);
            decoded.extend(pcm);
        }
        assert_eq!(decoded.len(), 48_000);
        // Energy survives (lossy codec, allow for lookahead delay).
        let rms = |v: &[f32]| (v.iter().map(|x| x * x).sum::<f32>() / v.len() as f32).sqrt();
        let r_in = rms(&tone[4800..]);
        let r_out = rms(&decoded[4800..]);
        assert!((r_out / r_in - 1.0).abs() < 0.15, "rms in {r_in} out {r_out}");
    }

    #[cfg(feature = "opus")]
    #[test]
    fn wma_format_stereo_20ms() {
        let mut e = OpusEncoder::new(OpusConfig::wma(Some(128_000)), 77).unwrap();
        let pkts = e.push(&vec![0.1f32; 960 * 2 * 5]).unwrap();
        assert_eq!(pkts.len(), 5);
        for p in &pkts {
            let t = toc(&p.data).unwrap();
            assert!(t.stereo);
            assert_eq!(t.packet_samples(), 960);
        }
        assert_eq!(pkts[4].rtp_ts, 77 + 4 * 960);
        assert!(e.encode_frame(&[0.0; 10]).is_err());
    }
}
