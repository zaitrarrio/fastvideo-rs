//! Warm ffmpeg decode pipe for client video (VP8, H.264 → RGB24) and
//! bitstream probes (design §5.11).
//!
//! A [`VideoDecoder`] is an ffmpeg subprocess: compressed frames on stdin
//! (VP8 in IVF records, H.264 as Annex-B access units), RGB24 pictures of a
//! fixed size on stdout. Every picture is scaled to the model's input size
//! (fit inside, padded black), so the output is independent of what the
//! client sends and a mid-stream resolution change needs no restart.
//!
//! Latency (30 fps input):
//!
//! - H.264: ~1 ms per frame with ffmpeg 6.1. The h264 parser only knows an
//!   access unit is complete when the next one starts, so every access unit
//!   is followed by an access unit delimiter (AUD), which ends it at once.
//!   ffmpeg 5.1 (Debian bookworm) still holds one frame.
//! - VP8: one frame (the IVF demuxer hands frame N to the decoder when
//!   frame N+1's record arrives).
//!
//! Flags: `-probesize 32 -analyzeduration 0 -flags low_delay -threads 1`
//! (frame threading would add a frame per thread), `-fps_mode passthrough`
//! (one picture out per decoded frame, no duplication). `-fflags nobuffer`
//! breaks both demuxers and is not used.
//!
//! **Warm processes.** Starting ffmpeg costs 50 ms to several seconds on a
//! loaded host, as for the pipe encoders ([`crate::pipe`]), and the first
//! client frames must not wait for it. A session owns a [`DecoderPool`]:
//! it pre-starts one decoder process per profile (codec and output size;
//! for VP8 already fed the IVF file header) when the session opens, before
//! any client frame, and a decoder created from the pool adopts that
//! process. After it adopts one, the pool starts the next, which is the
//! decoder's **warm spare**: [`VideoDecoder::restart`] (the process died,
//! or a clean decoder is wanted) swaps it in without an ffmpeg start.
//! `FV_DECODER_SPARE=0` turns pools off (every start is cold).
//!
//! The first picture waits for the second input frame (ffmpeg's stream
//! probe reads two frames); from then on the latencies above hold.
//!
//! **Timestamps.** VP8 and Constrained Baseline H.264 have no reordering,
//! so pictures come out in input order. Each input's pts is queued and
//! popped per picture. Callers feed a decoder only from a keyframe on (and
//! again from the next keyframe after a gap), so the decoder does not drop
//! frames it was given; a restart clears the queue. The queue is bounded
//! (a decoder that swallowed frames would otherwise grow it forever).

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use fastvideo_protocol::{InputVideoCodec, RgbFrame};
use serde::Serialize;

use crate::error::{MediaError, Result};
use crate::pipe::{Framer, Proc};
use crate::{h264, tools};

/// Decoder settings: the codec in, the picture size out.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VideoDecoderConfig {
    pub codec: InputVideoCodec,
    /// Output picture size (the model's input size).
    pub width: u32,
    pub height: u32,
}

impl VideoDecoderConfig {
    pub fn validate(&self) -> Result<()> {
        if self.width < 2 || self.height < 2 || self.width % 2 != 0 || self.height % 2 != 0 || self.width > 8192 || self.height > 8192 {
            return Err(MediaError::invalid(format!("decode size {}x{} (even sides 2..8192)", self.width, self.height)));
        }
        Ok(())
    }

    pub fn frame_len(&self) -> usize {
        self.width as usize * self.height as usize * 3
    }

    /// ffmpeg arguments (after the quiet flags).
    pub fn ffmpeg_args(&self) -> Vec<String> {
        let (w, h) = (self.width, self.height);
        let demux = match self.codec {
            InputVideoCodec::Vp8 => "ivf",
            InputVideoCodec::H264 => "h264",
        };
        let vf = format!(
            "scale={w}:{h}:force_original_aspect_ratio=decrease:flags=bilinear,pad={w}:{h}:(ow-iw)/2:(oh-ih)/2,setsar=1"
        );
        [
            "-probesize", "32", "-analyzeduration", "0", "-flags", "low_delay", "-threads", "1", "-f", demux, "-i",
            "pipe:0", "-an", "-vf", &vf, "-fps_mode", "passthrough", "-pix_fmt", "rgb24", "-f", "rawvideo", "pipe:1",
        ]
        .map(String::from)
        .to_vec()
    }
}

/// The 32-byte IVF file header: VP8, microsecond timestamps.
pub fn ivf_header(width: u32, height: u32) -> [u8; 32] {
    let mut h = [0u8; 32];
    h[0..4].copy_from_slice(b"DKIF");
    h[4..6].copy_from_slice(&0u16.to_le_bytes());
    h[6..8].copy_from_slice(&32u16.to_le_bytes());
    h[8..12].copy_from_slice(b"VP80");
    h[12..14].copy_from_slice(&(width.min(0xffff) as u16).to_le_bytes());
    h[14..16].copy_from_slice(&(height.min(0xffff) as u16).to_le_bytes());
    // Time base scale/rate = 1/1_000_000: pts in microseconds.
    h[16..20].copy_from_slice(&1_000_000u32.to_le_bytes());
    h[20..24].copy_from_slice(&1u32.to_le_bytes());
    h
}

/// One IVF frame record (12-byte header + payload).
pub fn ivf_record(frame: &[u8], pts_us: u64) -> Vec<u8> {
    let mut v = Vec::with_capacity(12 + frame.len());
    v.extend_from_slice(&(frame.len() as u32).to_le_bytes());
    v.extend_from_slice(&pts_us.to_le_bytes());
    v.extend_from_slice(frame);
    v
}

/// An access unit delimiter (primary_pic_type 7: any slice type).
pub const H264_AUD: [u8; 6] = [0, 0, 0, 1, 0x09, 0xf0];

/// An H.264 access unit for the decode pipe: leading AUDs removed, one AUD
/// appended (it ends the access unit for the parser at once).
pub fn h264_unit(au: &[u8]) -> Vec<u8> {
    let mut v = Vec::with_capacity(au.len() + H264_AUD.len());
    let mut rest = au;
    loop {
        let sc = if rest.starts_with(&[0, 0, 0, 1]) {
            4
        } else if rest.starts_with(&[0, 0, 1]) {
            3
        } else {
            break;
        };
        if rest.get(sc).is_some_and(|b| b & 0x1f == h264::nal::AUD) {
            // Skip to the next start code.
            let body = &rest[sc..];
            let next = body.windows(3).position(|w| w == [0, 0, 1]).map(|i| {
                // Include the leading zero of a 4-byte start code.
                if i > 0 && body[i - 1] == 0 {
                    sc + i - 1
                } else {
                    sc + i
                }
            });
            match next {
                Some(n) => rest = &rest[n..],
                None => {
                    rest = &[];
                    break;
                }
            }
        } else {
            break;
        }
    }
    v.extend_from_slice(rest);
    v.extend_from_slice(&H264_AUD);
    v
}

/// The picture size a VP8 keyframe declares (RFC 6386 §9.1), `None` for an
/// interframe or a truncated/invalid frame.
pub fn vp8_keyframe_size(frame: &[u8]) -> Option<(u32, u32)> {
    if frame.len() < 10 || frame[0] & 1 != 0 || frame[3..6] != [0x9d, 0x01, 0x2a] {
        return None;
    }
    let w = u32::from(u16::from_le_bytes([frame[6], frame[7]]) & 0x3fff);
    let h = u32::from(u16::from_le_bytes([frame[8], frame[9]]) & 0x3fff);
    (w > 0 && h > 0).then_some((w, h))
}

/// The picture size of a compressed keyframe: VP8's frame header, H.264's
/// SPS. `None` when the frame declares none (an interframe).
pub fn keyframe_size(codec: InputVideoCodec, frame: &[u8]) -> Option<(u32, u32)> {
    match codec {
        InputVideoCodec::Vp8 => vp8_keyframe_size(frame),
        InputVideoCodec::H264 => h264::find_sps(frame).map(|s| (s.width, s.height)),
    }
}

/// Whether a compressed frame is a keyframe (VP8 key frame; H.264 IDR).
pub fn is_keyframe(codec: InputVideoCodec, frame: &[u8]) -> bool {
    match codec {
        InputVideoCodec::Vp8 => crate::vp8::is_keyframe(frame),
        InputVideoCodec::H264 => h264::is_idr(frame),
    }
}

/// A decoded picture and its input's pts.
#[derive(Debug, Clone, PartialEq)]
pub struct DecodedPicture {
    pub frame: RgbFrame,
    pub pts_us: u64,
}

/// Decoder counters.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
pub struct DecodeStats {
    pub frames_in: u64,
    pub bytes_in: u64,
    pub frames_out: u64,
    pub restarts: u64,
    /// Restarts served by the warm spare (no ffmpeg start).
    pub warm_restarts: u64,
    pub write_errors: u64,
    /// Timestamps dropped because the decoder swallowed frames.
    pub pts_trimmed: u64,
}

/// Most inputs in flight without a picture before the oldest timestamps
/// are dropped (a burst fed faster than ffmpeg decodes stays well below).
const MAX_IN_FLIGHT: usize = 256;

fn spares_enabled() -> bool {
    !std::env::var("FV_DECODER_SPARE").is_ok_and(|v| v.trim() == "0")
}

/// A started decoder process, with the IVF header already written for VP8.
fn spawn_decoder(cfg: &VideoDecoderConfig) -> Result<Proc> {
    let mut p = Proc::spawn(&cfg.ffmpeg_args(), Framer::Raw(cfg.frame_len()), 0)?;
    if cfg.codec == InputVideoCodec::Vp8 {
        p.write(&ivf_header(cfg.width, cfg.height))?;
    }
    Ok(p)
}

/// A session's warm decoder processes: at most one per profile (codec and
/// output size), started on a background thread. Clones share it; the
/// processes are killed when the last clone goes.
#[derive(Clone)]
pub struct DecoderPool(Arc<PoolInner>);

struct PoolInner {
    enabled: bool,
    warm: Mutex<Vec<(VideoDecoderConfig, Proc)>>,
    pending: Mutex<Vec<VideoDecoderConfig>>,
}

impl std::fmt::Debug for DecoderPool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DecoderPool").field("enabled", &self.0.enabled).field("warm", &self.warm()).finish()
    }
}

impl DecoderPool {
    pub fn new(enabled: bool) -> Self {
        Self(Arc::new(PoolInner { enabled, warm: Mutex::new(Vec::new()), pending: Mutex::new(Vec::new()) }))
    }

    /// One session's pool; disabled with `FV_DECODER_SPARE=0`.
    pub fn per_session() -> Self {
        Self::new(spares_enabled())
    }

    pub fn enabled(&self) -> bool {
        self.0.enabled
    }

    /// The profiles with a warm process now.
    pub fn warm(&self) -> Vec<VideoDecoderConfig> {
        lock(&self.0.warm).iter().map(|(c, _)| *c).collect()
    }

    /// Starts a process for `cfg` in the background (unless one is warm or
    /// starting).
    pub fn prewarm(&self, cfg: VideoDecoderConfig) {
        if !self.0.enabled || cfg.validate().is_err() {
            return;
        }
        {
            let mut pending = lock(&self.0.pending);
            if pending.contains(&cfg) || lock(&self.0.warm).iter().any(|(c, _)| *c == cfg) {
                return;
            }
            pending.push(cfg);
        }
        let weak = Arc::downgrade(&self.0);
        let r = std::thread::Builder::new().name("ffmpeg-decode-prewarm".into()).spawn(move || {
            let p = spawn_decoder(&cfg);
            let Some(inner) = weak.upgrade() else { return };
            lock(&inner.pending).retain(|c| *c != cfg);
            match p {
                Ok(p) => lock(&inner.warm).push((cfg, p)),
                Err(e) => tracing::warn!(error = %e, "cannot pre-start a video decoder"),
            }
        });
        if r.is_err() {
            lock(&self.0.pending).retain(|c| *c != cfg);
        }
    }

    /// The warm process for `cfg`, if one is up.
    fn take(&self, cfg: &VideoDecoderConfig) -> Option<Proc> {
        let mut w = lock(&self.0.warm);
        let i = w.iter().position(|(c, _)| c == cfg)?;
        let (_, mut p) = w.swap_remove(i);
        p.alive().then_some(p)
    }
}

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|p| p.into_inner())
}

/// An ffmpeg decode pipe with a warm spare. Blocking: run it on its own
/// thread (writes wait while ffmpeg's stdin pipe is full).
pub struct VideoDecoder {
    cfg: VideoDecoderConfig,
    cur: Option<Proc>,
    pool: DecoderPool,
    pending: VecDeque<u64>,
    index: u64,
    stats: DecodeStats,
}

impl std::fmt::Debug for VideoDecoder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VideoDecoder").field("cfg", &self.cfg).field("stats", &self.stats).finish()
    }
}

impl VideoDecoder {
    /// Starts ffmpeg, with a private pool for its warm spare (unless
    /// `FV_DECODER_SPARE=0`).
    pub fn new(cfg: VideoDecoderConfig) -> Result<Self> {
        Self::with_pool(cfg, DecoderPool::per_session())
    }

    /// With (`true`) or without a warm spare.
    pub fn with_spare(cfg: VideoDecoderConfig, spare: bool) -> Result<Self> {
        Self::with_pool(cfg, DecoderPool::new(spare))
    }

    /// Adopts the pool's warm process for `cfg` (else starts ffmpeg cold);
    /// the pool then starts this decoder's spare.
    pub fn with_pool(cfg: VideoDecoderConfig, pool: DecoderPool) -> Result<Self> {
        cfg.validate()?;
        let t0 = Instant::now();
        let (cur, warm) = match pool.take(&cfg) {
            Some(p) => (p, true),
            None => (spawn_decoder(&cfg)?, false),
        };
        tracing::debug!(codec = cfg.codec.as_str(), warm, ms = t0.elapsed().as_millis() as u64, "video decoder started");
        let me = Self { cfg, cur: Some(cur), pool, pending: VecDeque::new(), index: 0, stats: DecodeStats::default() };
        me.pool.prewarm(cfg);
        Ok(me)
    }

    /// Whether the next restart would find a warm process.
    pub fn has_warm_spare(&self) -> bool {
        self.pool.warm().contains(&self.cfg)
    }

    pub fn config(&self) -> &VideoDecoderConfig {
        &self.cfg
    }

    pub fn stats(&self) -> &DecodeStats {
        &self.stats
    }

    /// Id of the current ffmpeg process.
    pub fn pid(&self) -> Option<u32> {
        self.cur.as_ref().map(Proc::id)
    }

    /// Replaces the current process (the warm spare when there is one).
    /// Pictures still inside the old process are lost.
    pub fn restart(&mut self) -> Result<()> {
        self.cur = None;
        self.pending.clear();
        self.stats.restarts += 1;
        let t0 = Instant::now();
        let spare = self.pool.take(&self.cfg);
        let warm = spare.is_some();
        self.cur = Some(match spare {
            Some(p) => p,
            None => spawn_decoder(&self.cfg)?,
        });
        if warm {
            self.stats.warm_restarts += 1;
        }
        tracing::info!(codec = self.cfg.codec.as_str(), warm, ms = t0.elapsed().as_millis() as u64, "video decoder restarted");
        self.pool.prewarm(self.cfg);
        Ok(())
    }

    /// Feeds one compressed frame (VP8 frame / H.264 access unit). A write
    /// error (ffmpeg died) restarts the decoder and returns the error: the
    /// caller must resume from a keyframe.
    pub fn push(&mut self, frame: &[u8], pts_us: u64) -> Result<()> {
        let data = match self.cfg.codec {
            InputVideoCodec::Vp8 => ivf_record(frame, pts_us),
            InputVideoCodec::H264 => h264_unit(frame),
        };
        let cur = self.cur.as_mut().ok_or_else(|| MediaError::Decode("decoder finished".into()))?;
        match cur.write(&data) {
            Ok(()) => {
                self.stats.frames_in += 1;
                self.stats.bytes_in += frame.len() as u64;
                self.pending.push_back(pts_us);
                if self.pending.len() > MAX_IN_FLIGHT {
                    let excess = self.pending.len() - MAX_IN_FLIGHT;
                    self.pending.drain(..excess);
                    self.stats.pts_trimmed += excess as u64;
                }
                Ok(())
            }
            Err(e) => {
                self.stats.write_errors += 1;
                self.restart()?;
                Err(e)
            }
        }
    }

    fn picture(&mut self, data: Vec<u8>) -> Option<DecodedPicture> {
        let pts_us = self.pending.pop_front().unwrap_or_default();
        let frame = RgbFrame::new(self.cfg.width, self.cfg.height, data.into(), self.index).ok()?;
        self.index += 1;
        self.stats.frames_out += 1;
        Some(DecodedPicture { frame, pts_us })
    }

    /// Pictures decoded so far (never blocks). A process that exited is
    /// replaced (its error is returned once).
    pub fn poll(&mut self) -> Result<Vec<DecodedPicture>> {
        let Some(cur) = self.cur.as_mut() else { return Ok(Vec::new()) };
        let (frames, ended) = cur.try_take()?;
        let mut out = Vec::with_capacity(frames.len());
        for f in frames {
            if let Some(p) = self.picture(f) {
                out.push(p);
            }
        }
        if ended {
            self.restart()?;
            if out.is_empty() {
                return Err(MediaError::tool("ffmpeg", "the video decoder exited"));
            }
        }
        Ok(out)
    }

    /// Waits up to `timeout` for at least one picture.
    pub fn poll_wait(&mut self, timeout: std::time::Duration) -> Result<Vec<DecodedPicture>> {
        let deadline = Instant::now() + timeout;
        loop {
            let out = self.poll()?;
            if !out.is_empty() || Instant::now() >= deadline {
                return Ok(out);
            }
            std::thread::sleep(std::time::Duration::from_millis(2));
        }
    }

    /// EOF: every picture still in the decoder.
    pub fn finish(&mut self) -> Result<Vec<DecodedPicture>> {
        let Some(cur) = self.cur.take() else { return Ok(Vec::new()) };
        let frames = cur.close("decode")?;
        Ok(frames.into_iter().filter_map(|f| self.picture(f)).collect())
    }
}

/// Whether ffmpeg is on this host (decoder tests skip without it).
pub fn decoder_available() -> bool {
    tools::ffmpeg_available()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ivf_framing() {
        let h = ivf_header(640, 360);
        assert_eq!(&h[0..4], b"DKIF");
        assert_eq!(u16::from_le_bytes([h[12], h[13]]), 640);
        assert_eq!(u32::from_le_bytes([h[16], h[17], h[18], h[19]]), 1_000_000);
        let r = ivf_record(&[1, 2, 3], 42);
        assert_eq!(r.len(), 15);
        assert_eq!(u32::from_le_bytes([r[0], r[1], r[2], r[3]]), 3);
        assert_eq!(u64::from_le_bytes(r[4..12].try_into().unwrap()), 42);
        // Our own IVF reader takes it back.
        let mut buf = h.to_vec();
        buf.extend(r);
        let mut done = false;
        assert_eq!(crate::vp8::take_ivf_frames(&mut buf, &mut done).unwrap(), vec![vec![1, 2, 3]]);
    }

    #[test]
    fn h264_units_end_with_one_aud() {
        let idr = [0u8, 0, 0, 1, 0x65, 0xaa];
        assert_eq!(h264_unit(&idr), [&idr[..], &H264_AUD[..]].concat());
        // A leading AUD (3- or 4-byte start code) is removed.
        let with_aud = [&H264_AUD[..], &idr[..]].concat();
        assert_eq!(h264_unit(&with_aud), [&idr[..], &H264_AUD[..]].concat());
        let short = [0u8, 0, 1, 0x09, 0xf0, 0, 0, 1, 0x41, 7];
        assert_eq!(h264_unit(&short), [&[0u8, 0, 1, 0x41, 7][..], &H264_AUD[..]].concat());
        assert_eq!(h264_unit(&H264_AUD), H264_AUD.to_vec());
    }

    #[test]
    fn vp8_keyframe_sizes() {
        // Frame tag (keyframe, show), start code, 640x360 with scale bits set.
        let mut k = vec![0x10, 0x02, 0x00, 0x9d, 0x01, 0x2a];
        k.extend_from_slice(&(640u16 | 0x4000).to_le_bytes());
        k.extend_from_slice(&360u16.to_le_bytes());
        assert_eq!(vp8_keyframe_size(&k), Some((640, 360)));
        assert_eq!(keyframe_size(InputVideoCodec::Vp8, &k), Some((640, 360)));
        assert!(is_keyframe(InputVideoCodec::Vp8, &k));
        let mut inter = k.clone();
        inter[0] |= 1;
        assert_eq!(vp8_keyframe_size(&inter), None);
        assert_eq!(vp8_keyframe_size(&k[..8]), None);
        let mut bad = k.clone();
        bad[4] = 0;
        assert_eq!(vp8_keyframe_size(&bad), None);
    }

    #[test]
    fn args_scale_and_pad_to_the_model_size() {
        let c = VideoDecoderConfig { codec: InputVideoCodec::H264, width: 64, height: 36 };
        let a = c.ffmpeg_args().join(" ");
        assert!(a.contains("-f h264 -i pipe:0"), "{a}");
        assert!(a.contains("scale=64:36:force_original_aspect_ratio=decrease"), "{a}");
        assert!(a.ends_with("-pix_fmt rgb24 -f rawvideo pipe:1"), "{a}");
        assert_eq!(c.frame_len(), 64 * 36 * 3);
        assert!(VideoDecoderConfig { width: 63, ..c }.validate().is_err());
    }
}
