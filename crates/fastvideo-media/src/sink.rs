//! RTMP / HLS / file sinks: one ffmpeg process with two inputs
//! (streaming-refs §4.2, design §5.3).
//!
//! - Video: RGB24 frames on ffmpeg's stdin.
//! - Audio: s16le PCM on a **second pipe**. The reference client passes an
//!   inherited file descriptor; this crate forbids `unsafe` (which fd
//!   inheritance needs), so the second pipe is a loopback TCP socket that
//!   ffmpeg connects to (`-i tcp://127.0.0.1:<port>`). Same semantics, and a
//!   dead ffmpeg surfaces as a write error instead of a hang.
//! - Each pipe has its own writer thread fed by a bounded drop-oldest queue,
//!   so a slow or dead ffmpeg never blocks the pacer.
//! - **There is always an audio track** (platforms reject video-only FLV):
//!   a video-only session gets ffmpeg's `anullsrc` silence instead of the
//!   second pipe.
//! - Video is NVENC (`h264_nvenc`) with the same GOP/profile arguments as the
//!   NVENC pipe encoder; audio is AAC 128k at 48 kHz (§5.3).
//! - A dead ffmpeg is restarted lazily (at most once per 2 s).

use std::io::Write;
use std::net::{TcpListener, TcpStream};
use std::path::PathBuf;
use std::process::{Child, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use bytes::Bytes;

use crate::av::{f32_to_s16le, RgbFrame};
use crate::error::{MediaError, Result};
use crate::mp4::AudioTarget;
use crate::queue::DropOldest;
use crate::tools;
use crate::video::{FfmpegH264, H264Config};

/// HLS playlist file name inside the output directory.
pub const HLS_PLAYLIST: &str = "stream.m3u8";

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SinkTarget {
    /// FLV to an RTMP ingest; 2 s keyframes.
    Rtmp { url: String },
    /// Rolling HLS in `dir`: 1 s keyframes and segments, `window` segments.
    Hls { dir: PathBuf, segment_s: u32, window: u32 },
    /// A local MP4 (recording, and tests).
    File { path: PathBuf },
}

/// PCM the caller feeds per tick (wire audio from the pacer).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SinkAudioIn {
    pub rate: u32,
    pub channels: u8,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SinkConfig {
    pub target: SinkTarget,
    pub width: u32,
    pub height: u32,
    pub fps: u32,
    pub video_bitrate_bps: u32,
    /// `None` = video-only session: the sink synthesizes a silent track.
    pub audio_in: Option<SinkAudioIn>,
    pub aac: AudioTarget,
    /// Ticks buffered per pipe before the oldest is dropped.
    pub queue_ticks: usize,
    /// NVENC in production.
    pub encoder: FfmpegH264,
}

impl SinkConfig {
    pub fn new(target: SinkTarget, width: u32, height: u32, fps: u32, audio_in: Option<SinkAudioIn>) -> Self {
        Self {
            target,
            width,
            height,
            fps,
            video_bitrate_bps: crate::video::default_bitrate(width, height),
            audio_in,
            aac: AudioTarget::BROADCAST,
            queue_ticks: (fps as usize) * 2,
            encoder: FfmpegH264::Nvenc,
        }
    }

    fn gop_seconds(&self) -> f32 {
        match &self.target {
            SinkTarget::Hls { segment_s, .. } => (*segment_s).max(1) as f32,
            _ => 2.0,
        }
    }

    /// The full ffmpeg argument list (after the binary) for an audio port.
    pub fn ffmpeg_args(&self, audio_port: Option<u16>) -> Result<Vec<String>> {
        let mut a: Vec<String> = ["-hide_banner", "-loglevel", "error", "-nostats", "-y"].map(String::from).to_vec();
        a.extend(["-f", "rawvideo", "-pix_fmt", "rgb24", "-s"].map(String::from));
        a.push(format!("{}x{}", self.width, self.height));
        a.extend(["-r".into(), self.fps.to_string(), "-i".into(), "pipe:0".into()]);
        match (self.audio_in, audio_port) {
            (Some(ai), Some(port)) => {
                a.extend(["-f", "s16le", "-ar"].map(String::from));
                a.extend([ai.rate.to_string(), "-ac".into(), ai.channels.to_string(), "-i".into()]);
                a.push(format!("tcp://127.0.0.1:{port}"));
            }
            _ => {
                a.extend(["-f", "lavfi", "-i"].map(String::from));
                let layout = if self.aac.channels == 1 { "mono" } else { "stereo" };
                a.push(format!("anullsrc=channel_layout={layout}:sample_rate={}", self.aac.rate));
                a.push("-shortest".into());
            }
        }
        a.extend(["-map", "0:v", "-map", "1:a"].map(String::from));
        let mut h = H264Config::new(self.width, self.height, self.fps);
        h.bitrate_bps = self.video_bitrate_bps;
        h.gop_seconds = self.gop_seconds();
        a.extend(self.encoder.stream_args(&h)?);
        a.extend(["-c:a", "aac", "-b:a"].map(String::from));
        a.extend([self.aac.bitrate_bps.to_string(), "-ar".into(), self.aac.rate.to_string()]);
        a.extend(["-ac".into(), self.aac.channels.to_string()]);
        match &self.target {
            SinkTarget::Rtmp { url } => {
                a.extend(["-f".into(), "flv".into(), url.clone()]);
            }
            SinkTarget::Hls { dir, segment_s, window } => {
                a.extend(["-f".into(), "hls".into(), "-hls_time".into(), segment_s.to_string()]);
                a.extend(["-hls_list_size".into(), window.to_string(), "-hls_flags".into()]);
                a.push("delete_segments+temp_file+independent_segments+omit_endlist".into());
                a.extend(["-hls_start_number_source", "epoch", "-hls_segment_filename"].map(String::from));
                a.push(dir.join("segment-%d.ts").to_string_lossy().into_owned());
                a.push(dir.join(HLS_PLAYLIST).to_string_lossy().into_owned());
            }
            SinkTarget::File { path } => {
                a.extend(["-movflags".into(), "+faststart".into(), path.to_string_lossy().into_owned()]);
            }
        }
        Ok(a)
    }
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct SinkStats {
    pub frames_sent: u64,
    pub audio_ticks_sent: u64,
    pub dropped_video: u64,
    pub dropped_audio: u64,
    pub restarts: u64,
}

struct Running {
    child: Child,
    video_q: Arc<DropOldest<Bytes>>,
    audio_q: Option<Arc<DropOldest<Bytes>>>,
    threads: Vec<std::thread::JoinHandle<()>>,
    stop: Arc<AtomicBool>,
}

/// An ffmpeg RTMP/HLS/file sink fed one tick at a time.
pub struct FfmpegSink {
    cfg: SinkConfig,
    run: Option<Running>,
    started_at: Instant,
    stats: SinkStats,
}

impl FfmpegSink {
    pub fn start(cfg: SinkConfig) -> Result<Self> {
        if cfg.width % 2 != 0 || cfg.height % 2 != 0 || cfg.fps == 0 {
            return Err(MediaError::invalid("sink needs even dimensions and a positive fps"));
        }
        if let SinkTarget::Hls { dir, .. } = &cfg.target {
            std::fs::create_dir_all(dir)?;
        }
        let run = Self::spawn(&cfg)?;
        Ok(Self { cfg, run: Some(run), started_at: Instant::now(), stats: SinkStats::default() })
    }

    fn spawn(cfg: &SinkConfig) -> Result<Running> {
        if let SinkTarget::Hls { dir, .. } = &cfg.target {
            // A dead predecessor's playlist points at segments nobody deletes.
            if let Ok(rd) = std::fs::read_dir(dir) {
                for e in rd.flatten() {
                    let _ = std::fs::remove_file(e.path());
                }
            }
        }
        let listener = match cfg.audio_in {
            Some(_) => Some(TcpListener::bind(("127.0.0.1", 0))?),
            None => None,
        };
        let port = listener.as_ref().map(|l| l.local_addr().map(|a| a.port())).transpose()?;
        let mut cmd = std::process::Command::new(tools::ffmpeg_bin());
        cmd.args(cfg.ffmpeg_args(port)?).stdin(Stdio::piped()).stdout(Stdio::null()).stderr(Stdio::null());
        let mut child = cmd.spawn().map_err(|e| MediaError::tool("ffmpeg", format!("not available: {e}")))?;
        let mut stdin = child.stdin.take().ok_or_else(|| MediaError::tool("ffmpeg", "no stdin"))?;
        let stop = Arc::new(AtomicBool::new(false));
        let cap = cfg.queue_ticks.max(1);
        let video_q = Arc::new(DropOldest::<Bytes>::new(cap));
        let mut threads = Vec::new();
        {
            let q = video_q.clone();
            let stop = stop.clone();
            threads.push(std::thread::Builder::new().name("sink-video".into()).spawn(move || {
                while let Some(b) = next(&q, &stop) {
                    if stdin.write_all(&b).is_err() {
                        break;
                    }
                }
            })?);
        }
        let audio_q = match listener {
            Some(l) => {
                let q = Arc::new(DropOldest::<Bytes>::new(cap));
                let q2 = q.clone();
                let stop2 = stop.clone();
                threads.push(std::thread::Builder::new().name("sink-audio".into()).spawn(move || {
                    let Some(mut s) = accept(&l, &stop2, Duration::from_secs(15)) else { return };
                    while let Some(b) = next(&q2, &stop2) {
                        if s.write_all(&b).is_err() {
                            break;
                        }
                    }
                    let _ = s.shutdown(std::net::Shutdown::Write);
                })?);
                Some(q)
            }
            None => None,
        };
        Ok(Running { child, video_q, audio_q, threads, stop })
    }

    /// Whether ffmpeg is alive; restarts it (at most every 2 s) when not.
    fn ensure_running(&mut self) -> bool {
        let alive = match self.run.as_mut() {
            Some(r) => matches!(r.child.try_wait(), Ok(None)),
            None => false,
        };
        if alive {
            return true;
        }
        if self.started_at.elapsed() < Duration::from_secs(2) {
            return false;
        }
        if let Some(r) = self.run.take() {
            self.retire(r);
        }
        self.started_at = Instant::now();
        match Self::spawn(&self.cfg) {
            Ok(r) => {
                self.stats.restarts += 1;
                tracing::warn!(restarts = self.stats.restarts, "sink: ffmpeg restarted");
                self.run = Some(r);
                true
            }
            Err(e) => {
                tracing::warn!("sink: ffmpeg restart failed: {e}");
                false
            }
        }
    }

    fn retire(&mut self, mut r: Running) {
        r.stop.store(true, Ordering::SeqCst);
        r.video_q.close();
        if let Some(q) = &r.audio_q {
            q.close();
        }
        self.stats.dropped_video += r.video_q.dropped_total();
        self.stats.dropped_audio += r.audio_q.as_ref().map(|q| q.dropped_total()).unwrap_or(0);
        let _ = r.child.kill();
        let _ = r.child.wait();
        for t in r.threads {
            let _ = t.join();
        }
    }

    /// Feed one tick: a frame and (for an A/V session) exactly its audio.
    pub fn send(&mut self, frame: &RgbFrame, audio: Option<&[f32]>) -> Result<()> {
        if frame.width != self.cfg.width || frame.height != self.cfg.height {
            return Err(MediaError::invalid("frame size does not match the sink"));
        }
        if !self.ensure_running() {
            self.stats.dropped_video += 1;
            return Ok(());
        }
        let r = self.run.as_ref().expect("running");
        r.video_q.push(frame.data.clone());
        self.stats.frames_sent += 1;
        if let (Some(q), Some(a)) = (&r.audio_q, audio) {
            q.push(Bytes::from(f32_to_s16le(a)));
            self.stats.audio_ticks_sent += 1;
        }
        Ok(())
    }

    /// Close the pipes and wait for ffmpeg to finish writing (trailer,
    /// playlist). Returns the stats.
    pub fn finish(mut self, timeout: Duration) -> Result<SinkStats> {
        let Some(mut r) = self.run.take() else { return Ok(self.stats) };
        r.video_q.close();
        if let Some(q) = &r.audio_q {
            q.close();
        }
        for t in r.threads.drain(..) {
            let _ = t.join();
        }
        let deadline = Instant::now() + timeout;
        let status = loop {
            if let Some(s) = r.child.try_wait()? {
                break Some(s);
            }
            if Instant::now() >= deadline {
                let _ = r.child.kill();
                let _ = r.child.wait();
                break None;
            }
            std::thread::sleep(Duration::from_millis(10));
        };
        self.stats.dropped_video += r.video_q.dropped_total();
        self.stats.dropped_audio += r.audio_q.as_ref().map(|q| q.dropped_total()).unwrap_or(0);
        match status {
            Some(s) if s.success() => Ok(self.stats),
            Some(s) => Err(MediaError::tool("ffmpeg", format!("sink exited with {s}"))),
            None => Err(MediaError::tool("ffmpeg", "sink did not exit in time")),
        }
    }

    pub fn stats(&self) -> SinkStats {
        self.stats
    }
}

impl Drop for FfmpegSink {
    fn drop(&mut self) {
        if let Some(r) = self.run.take() {
            self.retire(r);
        }
    }
}

fn next(q: &DropOldest<Bytes>, stop: &AtomicBool) -> Option<Bytes> {
    loop {
        if stop.load(Ordering::SeqCst) {
            return None;
        }
        if let Some(b) = q.pop_timeout(Duration::from_millis(100)) {
            return Some(b);
        }
        if q.is_closed() && q.is_empty() {
            return None;
        }
    }
}

fn accept(l: &TcpListener, stop: &AtomicBool, timeout: Duration) -> Option<TcpStream> {
    l.set_nonblocking(true).ok()?;
    let deadline = Instant::now() + timeout;
    while Instant::now() < deadline && !stop.load(Ordering::SeqCst) {
        match l.accept() {
            Ok((s, _)) => {
                s.set_nonblocking(false).ok()?;
                let _ = s.set_nodelay(true);
                return Some(s);
            }
            Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => std::thread::sleep(Duration::from_millis(5)),
            Err(_) => return None,
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn video_only_gets_a_silent_track() {
        let c = SinkConfig::new(SinkTarget::Rtmp { url: "rtmp://x/live/k".into() }, 1344, 768, 24, None);
        let a = c.ffmpeg_args(None).unwrap().join(" ");
        assert!(a.contains("-f lavfi -i anullsrc=channel_layout=stereo:sample_rate=48000"), "{a}");
        assert!(a.contains("-map 0:v -map 1:a"));
        assert!(a.contains("-c:a aac -b:a 128000 -ar 48000 -ac 2"));
        assert!(a.contains("-c:v h264_nvenc"));
        assert!(a.contains("-g 48 -bf 0 -forced-idr 1 -no-scenecut 1"));
        assert!(a.contains("-profile:v baseline -level:v 3.2"));
        assert!(!a.contains("libx264"));
        assert!(a.ends_with("-f flv rtmp://x/live/k"));
    }

    #[test]
    fn av_uses_the_second_pipe_and_hls_uses_1s_gops() {
        let c = SinkConfig::new(
            SinkTarget::Hls { dir: PathBuf::from("/tmp/h"), segment_s: 1, window: 6 },
            832,
            480,
            16,
            Some(SinkAudioIn { rate: 48_000, channels: 1 }),
        );
        let a = c.ffmpeg_args(Some(5555)).unwrap().join(" ");
        assert!(a.contains("-f s16le -ar 48000 -ac 1 -i tcp://127.0.0.1:5555"), "{a}");
        assert!(!a.contains("anullsrc"));
        assert!(a.contains("-g 16 -bf 0"));
        assert!(a.contains("-hls_time 1 -hls_list_size 6"));
        assert!(a.contains("-hls_start_number_source epoch"));
        assert!(a.ends_with("/tmp/h/stream.m3u8"));
    }
}
