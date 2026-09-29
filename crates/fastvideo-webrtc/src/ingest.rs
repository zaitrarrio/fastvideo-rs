//! Client media into a session's input rings (design §5.11).
//!
//! An [`Ingest`] takes the [`InboundMedia`] of one publishing peer (the
//! bounded queue of [`crate::host::Peer::take_inbound`]) and turns it into
//! decoded [`InputFrame`]s and [`InputAudio`] in the session's
//! [`InputBuffers`]:
//!
//! ```text
//! str0m depacketizer ─► inbound queue (64, drop newest, bitrate cap)
//!   ─► Ingest::push (gate: published tracks, mids)  ─► decode thread queue (32, drop)
//!   ─► keyframe gate (PLI on loss, ≤ 1/s) ─► size check on keyframes
//!   ─► VideoDecoder (ffmpeg, warm spare) ─► fps cap ─► video ring (drop oldest)
//!   ─► Opus decode (libopus) ─► mono/stereo mix ─► audio ring (drop oldest)
//! ```
//!
//! Limits (from the model's [`InputCaps`](fastvideo_protocol::InputCaps)
//! and the session): a keyframe larger than `max_width`x`max_height` is
//! refused (its frames are dropped until an acceptable keyframe arrives);
//! pictures above `max_fps` are decoded (the reference chain needs them)
//! but not buffered; input stops after `max_seconds` from the first frame.
//! Every drop is counted in [`IngestStats`].

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{sync_channel, Receiver, RecvTimeoutError, SyncSender, TrySendError};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use fastvideo_media::decode::{keyframe_size, DecoderPool, VideoDecoder, VideoDecoderConfig};
use fastvideo_media::ring::InputBuffers;
use fastvideo_protocol::{AudioInputCaps, InputFrame, InputVideoCodec, VideoInputCaps};
use serde::Serialize;

use crate::host::{InboundMedia, PeerHandle};
use crate::writer::{TrackKind, VideoCodec};

/// What one ingest accepts.
#[derive(Debug, Clone)]
pub struct IngestConfig {
    pub video: Option<VideoInputCaps>,
    pub audio: Option<AudioInputCaps>,
    /// Input stops this long after the first frame (the session limit).
    pub max_seconds: Option<u32>,
    /// Frames queued for the decode thread before new ones are dropped.
    pub queue: usize,
    /// The session's warm decoders ([`IngestConfig::prewarm`] at session
    /// start); `None`: a private pool (the first frame waits for ffmpeg).
    pub pool: Option<DecoderPool>,
}

impl IngestConfig {
    pub fn new(video: Option<VideoInputCaps>, audio: Option<AudioInputCaps>) -> Self {
        Self { video, audio, max_seconds: None, queue: 32, pool: None }
    }

    /// The decoder profiles a client may need: every accepted codec at the
    /// model's input size.
    pub fn decoder_configs(&self) -> Vec<VideoDecoderConfig> {
        self.video
            .iter()
            .flat_map(|v| v.codecs.iter().map(|c| VideoDecoderConfig { codec: *c, width: v.width, height: v.height }))
            .collect()
    }

    /// Pre-starts a decoder for every accepted codec in `pool` (session
    /// start: before the first client frame) and uses it.
    pub fn prewarm(mut self, pool: DecoderPool) -> Self {
        for c in self.decoder_configs() {
            pool.prewarm(c);
        }
        self.pool = Some(pool);
        self
    }
}

/// Ingest counters (exposed in stream status and Reactor `state_update`).
#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub struct IngestStats {
    /// The codec the client sends.
    pub codec: Option<String>,
    /// The picture size of the last accepted keyframe.
    pub source_size: Option<(u32, u32)>,
    pub video_frames_in: u64,
    pub video_frames_decoded: u64,
    pub video_frames_buffered: u64,
    /// Frames that arrived while no track was published (Reactor).
    pub dropped_unpublished: u64,
    /// The decode thread was behind (its queue was full).
    pub dropped_behind: u64,
    /// Frames between a loss and the next keyframe.
    pub dropped_waiting_keyframe: u64,
    /// Frames of refused keyframes (too large, unreadable).
    pub dropped_rejected: u64,
    /// Decoded pictures above `max_fps`.
    pub dropped_fps: u64,
    pub keyframes: u64,
    pub keyframe_requests: u64,
    pub decoder_restarts: u64,
    pub decode_errors: u64,
    pub audio_packets_in: u64,
    pub audio_chunks_buffered: u64,
    pub audio_decode_errors: u64,
    /// The last refusal, for the client (`input 1920x1080 exceeds 1280x720`).
    pub rejected: Option<String>,
    /// Why input ended (`duration limit`, `closed`).
    pub ended: Option<String>,
}

enum Msg {
    Media(InboundMedia),
    Close,
}

struct Shared {
    stats: Mutex<IngestStats>,
    video_on: AtomicBool,
    audio_on: AtomicBool,
    video_mid: Mutex<Option<String>>,
    audio_mid: Mutex<Option<String>>,
}

impl Shared {
    fn stats(&self) -> std::sync::MutexGuard<'_, IngestStats> {
        self.stats.lock().unwrap_or_else(|p| p.into_inner())
    }
}

/// One publisher's decode pipeline. Dropping it (or [`Ingest::close`])
/// stops the thread; the rings stay open for another publisher.
pub struct Ingest {
    tx: SyncSender<Msg>,
    shared: Arc<Shared>,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl std::fmt::Debug for Ingest {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Ingest").field("stats", &self.stats()).finish()
    }
}

impl Ingest {
    /// Starts the decode thread. `peer` receives the keyframe requests.
    /// Both tracks start enabled; a Reactor gateway disables them until
    /// `PublishTrack`.
    pub fn start(cfg: IngestConfig, buffers: Arc<InputBuffers>, peer: PeerHandle) -> std::io::Result<Self> {
        let (tx, rx) = sync_channel(cfg.queue.max(1));
        let shared = Arc::new(Shared {
            stats: Mutex::new(IngestStats::default()),
            video_on: AtomicBool::new(cfg.video.is_some()),
            audio_on: AtomicBool::new(cfg.audio.is_some()),
            video_mid: Mutex::new(None),
            audio_mid: Mutex::new(None),
        });
        let sh = shared.clone();
        let thread = std::thread::Builder::new()
            .name("fv-ingest".into())
            .spawn(move || Worker::new(cfg, buffers, peer, sh).run(rx))?;
        Ok(Self { tx, shared, thread: Some(thread) })
    }

    /// Accept (or drop) a track's media: the Reactor publisher slot.
    pub fn set_enabled(&self, kind: TrackKind, on: bool) {
        match kind {
            TrackKind::Video => self.shared.video_on.store(on, Ordering::Relaxed),
            TrackKind::Audio => self.shared.audio_on.store(on, Ordering::Relaxed),
        }
    }

    pub fn enabled(&self, kind: TrackKind) -> bool {
        match kind {
            TrackKind::Video => self.shared.video_on.load(Ordering::Relaxed),
            TrackKind::Audio => self.shared.audio_on.load(Ordering::Relaxed),
        }
    }

    /// Only media on `mid` counts for `kind` (`None`: any m-line of it).
    pub fn set_mid(&self, kind: TrackKind, mid: Option<String>) {
        let m = match kind {
            TrackKind::Video => &self.shared.video_mid,
            TrackKind::Audio => &self.shared.audio_mid,
        };
        *m.lock().unwrap_or_else(|p| p.into_inner()) = mid;
    }

    /// Hands one inbound frame to the decode thread. Never blocks: a full
    /// queue drops it (the decoder then waits for a keyframe).
    pub fn push(&self, m: InboundMedia) {
        let (on, mid) = match m.kind {
            TrackKind::Video => (&self.shared.video_on, &self.shared.video_mid),
            TrackKind::Audio => (&self.shared.audio_on, &self.shared.audio_mid),
        };
        let want = mid.lock().unwrap_or_else(|p| p.into_inner()).clone();
        if want.is_some_and(|w| w != m.mid) {
            return;
        }
        if !on.load(Ordering::Relaxed) {
            self.shared.stats().dropped_unpublished += 1;
            return;
        }
        match self.tx.try_send(Msg::Media(m)) {
            Ok(()) => {}
            Err(TrySendError::Full(_)) => self.shared.stats().dropped_behind += 1,
            Err(TrySendError::Disconnected(_)) => {}
        }
    }

    pub fn stats(&self) -> IngestStats {
        self.shared.stats().clone()
    }

    /// Stops the decode thread (pictures still in ffmpeg are flushed into
    /// the rings first).
    pub fn close(&mut self) {
        let _ = self.tx.send(Msg::Close);
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}

impl Drop for Ingest {
    fn drop(&mut self) {
        let _ = self.tx.try_send(Msg::Close);
    }
}

/// RTP timestamps (extended by str0m) to microseconds from the first.
#[derive(Debug, Default)]
struct Clock {
    first: Option<u64>,
}

impl Clock {
    fn us(&mut self, rtp: u64, hz: u64) -> u64 {
        let first = *self.first.get_or_insert(rtp);
        rtp.saturating_sub(first).saturating_mul(1_000_000) / hz
    }
}

struct Worker {
    cfg: IngestConfig,
    buffers: Arc<InputBuffers>,
    peer: PeerHandle,
    sh: Arc<Shared>,
    decoder: Option<VideoDecoder>,
    codec: Option<InputVideoCodec>,
    waiting_keyframe: bool,
    last_pli: Option<Instant>,
    video_clock: Clock,
    #[cfg_attr(not(feature = "opus"), allow(dead_code))]
    audio_clock: Clock,
    last_buffered_pts: Option<u64>,
    first_media: Option<Instant>,
    source: Option<(u32, u32)>,
    #[cfg(feature = "opus")]
    opus: Option<fastvideo_media::opus::OpusDecoder>,
}

/// PLIs at most this often.
const PLI_INTERVAL: Duration = Duration::from_secs(1);

fn to_input_codec(c: VideoCodec) -> InputVideoCodec {
    match c {
        VideoCodec::Vp8 => InputVideoCodec::Vp8,
        VideoCodec::H264 => InputVideoCodec::H264,
    }
}

impl Worker {
    fn new(cfg: IngestConfig, buffers: Arc<InputBuffers>, peer: PeerHandle, sh: Arc<Shared>) -> Self {
        Self {
            cfg,
            buffers,
            peer,
            sh,
            decoder: None,
            codec: None,
            waiting_keyframe: true,
            last_pli: None,
            video_clock: Clock::default(),
            audio_clock: Clock::default(),
            last_buffered_pts: None,
            first_media: None,
            source: None,
            #[cfg(feature = "opus")]
            opus: None,
        }
    }

    fn run(mut self, rx: Receiver<Msg>) {
        loop {
            let msg = rx.recv_timeout(Duration::from_millis(10));
            match msg {
                Ok(Msg::Media(m)) => {
                    if self.expired() {
                        continue;
                    }
                    match m.kind {
                        TrackKind::Video => self.video(m),
                        TrackKind::Audio => self.audio(m),
                    }
                }
                Ok(Msg::Close) | Err(RecvTimeoutError::Disconnected) => break,
                Err(RecvTimeoutError::Timeout) => {}
            }
            self.drain_decoder();
        }
        if let Some(mut d) = self.decoder.take() {
            if let Ok(pics) = d.finish() {
                for p in pics {
                    self.buffer_picture(p.frame, p.pts_us);
                }
            }
        }
        let mut st = self.sh.stats();
        st.ended.get_or_insert_with(|| "closed".into());
    }

    /// Past the session's input limit: stop taking media.
    fn expired(&mut self) -> bool {
        let first = *self.first_media.get_or_insert_with(Instant::now);
        let Some(max) = self.cfg.max_seconds else { return false };
        if first.elapsed() < Duration::from_secs(u64::from(max)) {
            return false;
        }
        let mut st = self.sh.stats();
        if st.ended.is_none() {
            st.ended = Some(format!("input duration limit ({max} s)"));
            tracing::info!(max, "ingest reached its duration limit");
        }
        true
    }

    fn request_keyframe(&mut self, mid: &str) {
        if self.last_pli.is_some_and(|t| t.elapsed() < PLI_INTERVAL) {
            return;
        }
        self.last_pli = Some(Instant::now());
        if self.peer.request_keyframe(mid).is_ok() {
            self.sh.stats().keyframe_requests += 1;
        }
    }

    fn video(&mut self, m: InboundMedia) {
        let Some(caps) = self.cfg.video.clone() else { return };
        self.sh.stats().video_frames_in += 1;
        let Some(codec) = m.codec.map(to_input_codec) else { return };
        if !caps.codecs.contains(&codec) {
            self.reject(format!("codec {} is not accepted", codec.as_str()));
            return;
        }
        if self.codec != Some(codec) {
            // A new codec (or the first frame): a fresh decoder.
            self.codec = Some(codec);
            self.decoder = None;
            self.waiting_keyframe = true;
            self.sh.stats().codec = Some(codec.as_str().to_owned());
        }
        if !m.contiguous {
            self.waiting_keyframe = true;
        }
        if self.waiting_keyframe && !m.keyframe {
            self.sh.stats().dropped_waiting_keyframe += 1;
            self.request_keyframe(&m.mid);
            return;
        }
        if m.keyframe {
            match keyframe_size(codec, &m.data) {
                Some((w, h)) if caps.allows_size(w, h) => {
                    if self.source != Some((w, h)) {
                        tracing::info!(codec = codec.as_str(), width = w, height = h, "ingest: client video");
                    }
                    self.source = Some((w, h));
                    let mut st = self.sh.stats();
                    st.source_size = Some((w, h));
                    st.keyframes += 1;
                    st.rejected = None;
                }
                Some((w, h)) => {
                    self.waiting_keyframe = true;
                    self.reject(format!(
                        "input {w}x{h} exceeds the {}x{} limit",
                        caps.max_width.max(caps.max_height),
                        caps.max_width.min(caps.max_height)
                    ));
                    return;
                }
                // H.264 IDRs normally carry their SPS; without one, trust
                // the previous keyframe's size.
                None if self.source.is_some() => {
                    self.sh.stats().keyframes += 1;
                }
                None => {
                    self.waiting_keyframe = true;
                    self.reject("keyframe without a readable picture size".into());
                    self.request_keyframe(&m.mid);
                    return;
                }
            }
            self.waiting_keyframe = false;
        }
        let pts = self.video_clock.us(m.rtp_time, 90_000);
        if self.decoder.is_none() {
            let dcfg = VideoDecoderConfig { codec, width: caps.width, height: caps.height };
            let pool = self.cfg.pool.clone().unwrap_or_else(DecoderPool::per_session);
            match VideoDecoder::with_pool(dcfg, pool) {
                Ok(d) => self.decoder = Some(d),
                Err(e) => {
                    tracing::warn!(error = %e, "ingest: cannot start the video decoder");
                    self.sh.stats().decode_errors += 1;
                    self.waiting_keyframe = true;
                    return;
                }
            }
        }
        let Some(dec) = self.decoder.as_mut() else { return };
        if let Err(e) = dec.push(&m.data, pts) {
            tracing::debug!(error = %e, "ingest: decoder write failed");
            let mut st = self.sh.stats();
            st.decode_errors += 1;
            st.decoder_restarts = dec.stats().restarts;
            drop(st);
            self.waiting_keyframe = true;
            self.request_keyframe(&m.mid);
        }
    }

    fn reject(&mut self, why: String) {
        let mut st = self.sh.stats();
        st.dropped_rejected += 1;
        if st.rejected.as_deref() != Some(why.as_str()) {
            tracing::info!(reason = %why, "ingest: client video refused");
            st.rejected = Some(why);
        }
    }

    fn drain_decoder(&mut self) {
        let Some(dec) = self.decoder.as_mut() else { return };
        let pics = match dec.poll() {
            Ok(p) => p,
            Err(e) => {
                tracing::debug!(error = %e, "ingest: decoder exited");
                let restarts = dec.stats().restarts;
                let mut st = self.sh.stats();
                st.decode_errors += 1;
                st.decoder_restarts = restarts;
                drop(st);
                self.waiting_keyframe = true;
                return;
            }
        };
        for p in pics {
            self.buffer_picture(p.frame, p.pts_us);
        }
    }

    fn buffer_picture(&mut self, frame: fastvideo_protocol::RgbFrame, pts_us: u64) {
        let Some(caps) = &self.cfg.video else { return };
        self.sh.stats().video_frames_decoded += 1;
        // The fps cap, with 20 % slack for capture jitter.
        let min_gap = 800_000 / u64::from(caps.max_fps.max(1));
        if self.last_buffered_pts.is_some_and(|last| pts_us.saturating_sub(last) < min_gap && pts_us >= last) {
            self.sh.stats().dropped_fps += 1;
            return;
        }
        self.last_buffered_pts = Some(pts_us);
        let source = self.source.unwrap_or((frame.width, frame.height));
        if self.buffers.video.push(pts_us, InputFrame { frame, pts_us, source }) {
            self.sh.stats().video_frames_buffered += 1;
        }
    }

    #[cfg(feature = "opus")]
    fn audio(&mut self, m: InboundMedia) {
        let Some(caps) = self.cfg.audio.clone() else { return };
        self.sh.stats().audio_packets_in += 1;
        if self.opus.is_none() {
            // Decode stereo: a mono Opus stream comes out duplicated.
            match fastvideo_media::opus::OpusDecoder::new(2) {
                Ok(d) => self.opus = Some(d),
                Err(e) => {
                    tracing::warn!(error = %e, "ingest: no Opus decoder");
                    self.sh.stats().audio_decode_errors += 1;
                    return;
                }
            }
        }
        let pts = self.audio_clock.us(m.rtp_time, 48_000);
        let Some(dec) = self.opus.as_mut() else { return };
        let samples = match dec.decode(&m.data) {
            Ok(s) => s,
            Err(_) => {
                self.sh.stats().audio_decode_errors += 1;
                return;
            }
        };
        let pcm = fastvideo_protocol::Pcm::new(48_000, 2, samples);
        let pcm = if caps.channels == 1 { pcm.to_mono() } else { pcm };
        if self.buffers.audio.push(pts, fastvideo_protocol::InputAudio { pcm, pts_us: pts }) {
            self.sh.stats().audio_chunks_buffered += 1;
        }
    }

    #[cfg(not(feature = "opus"))]
    fn audio(&mut self, _m: InboundMedia) {
        let mut st = self.sh.stats();
        st.audio_packets_in += 1;
        st.audio_decode_errors += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rtp_clock_counts_from_the_first_timestamp() {
        let mut c = Clock::default();
        assert_eq!(c.us(90_000 * 5, 90_000), 0);
        assert_eq!(c.us(90_000 * 5 + 3000, 90_000), 33_333);
        assert_eq!(c.us(10, 90_000), 0, "never negative");
        let mut a = Clock::default();
        a.us(480, 48_000);
        assert_eq!(a.us(480 + 960, 48_000), 20_000);
    }
}
