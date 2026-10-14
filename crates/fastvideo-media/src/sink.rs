//! RTMP(S) / SRT / HLS / file sinks: one ffmpeg process with two inputs
//! (streaming-refs §4.2, design §5.3, docs/serve/stream-outputs.md).
//!
//! - Live targets (`rtmp://`, `rtmps://` as FLV; `srt://` as MPEG-TS) meet
//!   Cloudflare Stream Live's input rules: H.264 + AAC-LC 48 kHz stereo
//!   (ADTS in MPEG-TS), CBR video clamped to 2-13 Mb/s, a closed 2 s GOP
//!   (Constrained Baseline, IDR only, no B-frames, no scene-cut keyframes).
//! - The URL of a live target carries credentials (an RTMP stream key in
//!   the path, an SRT passphrase in the query): they never reach a log or
//!   an error. ffmpeg's stderr lines are scrubbed of every secret part
//!   ([`SinkTarget::secrets`]) before they are kept, and `Debug` prints
//!   [`SinkTarget::redacted`].
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

use std::collections::VecDeque;
use std::io::{BufRead, Write};
use std::net::{TcpListener, TcpStream};
use std::path::PathBuf;
use std::process::{Child, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
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

#[derive(Clone, PartialEq, Eq)]
pub enum SinkTarget {
    /// FLV to an RTMP ingest (`rtmp://` or `rtmps://`, the stream key in the
    /// path); 2 s keyframes.
    Rtmp { url: String },
    /// MPEG-TS over SRT (`srt://host:port?streamid=…&passphrase=…`, see
    /// [`srt_url`]); 2 s keyframes.
    Srt { url: String },
    /// Rolling HLS in `dir`: 1 s keyframes and segments, `window` segments.
    Hls { dir: PathBuf, segment_s: u32, window: u32 },
    /// A local MP4 (recording, and tests).
    File { path: PathBuf },
}

impl std::fmt::Debug for SinkTarget {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SinkTarget::Rtmp { .. } => write!(f, "Rtmp({})", self.redacted()),
            SinkTarget::Srt { .. } => write!(f, "Srt({})", self.redacted()),
            SinkTarget::Hls { dir, segment_s, window } => write!(f, "Hls({}, {segment_s}s x {window})", dir.display()),
            SinkTarget::File { path } => write!(f, "File({})", path.display()),
        }
    }
}

impl SinkTarget {
    /// Whether this is a live network target (RTMP(S), SRT).
    pub fn is_live(&self) -> bool {
        matches!(self, SinkTarget::Rtmp { .. } | SinkTarget::Srt { .. })
    }

    /// The URL without credentials: scheme, host and port, plus the path of
    /// an RTMP URL with its last segment (the stream key) replaced. No
    /// userinfo, no query.
    pub fn redacted(&self) -> String {
        match self {
            SinkTarget::Rtmp { url } => {
                let (origin, path) = split_origin(url);
                let path = path.split(['?', '#']).next().unwrap_or("");
                let keep = match path.trim_end_matches('/').rsplit_once('/') {
                    Some((head, _key)) => format!("{head}/<key>"),
                    None => String::new(),
                };
                format!("{origin}{keep}")
            }
            SinkTarget::Srt { url } => split_origin(url).0,
            SinkTarget::Hls { dir, .. } => dir.display().to_string(),
            SinkTarget::File { path } => path.display().to_string(),
        }
    }

    /// The substrings that must never be printed: the whole URL, its
    /// userinfo, an RTMP stream key, the query values (raw and decoded).
    pub fn secrets(&self) -> Vec<String> {
        let mut out = Vec::new();
        let url = match self {
            SinkTarget::Rtmp { url } | SinkTarget::Srt { url } => url,
            _ => return out,
        };
        out.push(url.clone());
        let after_scheme = url.split_once("://").map_or(url.as_str(), |(_, r)| r);
        let end = after_scheme.find(['/', '?', '#']).unwrap_or(after_scheme.len());
        let (authority, rest) = after_scheme.split_at(end);
        if let Some((userinfo, _)) = authority.rsplit_once('@') {
            out.extend(userinfo.split(':').map(str::to_owned));
        }
        let (path, query) = match rest.split_once('?') {
            Some((p, q)) => (p, q),
            None => (rest, ""),
        };
        if matches!(self, SinkTarget::Rtmp { .. }) {
            if let Some(key) = path.trim_end_matches('/').rsplit('/').next() {
                out.push(key.to_owned());
            }
        }
        for kv in query.split('#').next().unwrap_or("").split('&') {
            if let Some((_, v)) = kv.split_once('=') {
                out.push(v.to_owned());
                out.push(percent_decode(v));
            }
        }
        // Short values (`caller`, `1316`) are settings, not secrets.
        out.retain(|x| x.len() >= 6 && x != "caller");
        out.sort_by_key(|x| std::cmp::Reverse(x.len()));
        out.dedup();
        out
    }
}

/// `scheme://host:port` (no userinfo) and the rest after the authority.
fn split_origin(url: &str) -> (String, String) {
    let Some((scheme, rest)) = url.split_once("://") else { return ("<url>".into(), String::new()) };
    let end = rest.find(['/', '?', '#']).unwrap_or(rest.len());
    let authority = &rest[..end];
    let host = authority.rsplit_once('@').map_or(authority, |(_, h)| h);
    (format!("{scheme}://{host}"), rest[end..].to_owned())
}

/// Replaces every secret in `line` with `<redacted>` (longest first).
pub fn scrub(line: &str, secrets: &[String]) -> String {
    let mut out = line.to_owned();
    for s in secrets {
        if !s.is_empty() && out.contains(s.as_str()) {
            out = out.replace(s.as_str(), "<redacted>");
        }
    }
    out
}

fn percent_decode(s: &str) -> String {
    let b = s.as_bytes();
    let mut out = Vec::with_capacity(b.len());
    let mut i = 0;
    while i < b.len() {
        if b[i] == b'%' && i + 2 < b.len() {
            if let Ok(v) = u8::from_str_radix(&s[i + 1..i + 3], 16) {
                out.push(v);
                i += 3;
                continue;
            }
        }
        out.push(b[i]);
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

/// Whether `v` can go into an ffmpeg `srt://` query as is: ffmpeg's libsrt
/// takes `streamid` and `passphrase` verbatim (no percent-decoding), so a
/// value must not contain the query's own delimiters.
pub fn srt_value_ok(v: &str) -> bool {
    !v.is_empty() && !v.contains(['&', '#', '?', '%']) && !v.chars().any(char::is_whitespace)
}

/// Percent-encodes a query value (RFC 3986 unreserved characters kept).
pub fn percent_encode(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for b in s.bytes() {
        if b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_' | b'.' | b'~') {
            out.push(b as char);
        } else {
            out.push_str(&format!("%{b:02X}"));
        }
    }
    out
}

/// An ffmpeg `srt://` URL in caller mode: `base` (`srt://host:port`; a
/// query it has is kept), `streamid` and `passphrase` added verbatim
/// (ffmpeg does not percent-decode them; check them with
/// [`srt_value_ok`]), `latency` (ffmpeg's libsrt takes microseconds) and
/// the MPEG-TS payload size (7 x 188).
pub fn srt_url(base: &str, stream_id: Option<&str>, passphrase: Option<&str>, latency_ms: u32) -> String {
    let has = |k: &str| base.split(['?', '&']).skip(1).any(|kv| kv.split('=').next() == Some(k));
    let mut adds: Vec<(&str, String)> = Vec::new();
    if let Some(id) = stream_id.filter(|v| !v.is_empty()) {
        if !has("streamid") {
            adds.push(("streamid", id.to_owned()));
        }
    }
    if let Some(p) = passphrase.filter(|v| !v.is_empty()) {
        if !has("passphrase") {
            adds.push(("passphrase", p.to_owned()));
        }
    }
    if !has("latency") {
        adds.push(("latency", (u64::from(latency_ms) * 1000).to_string()));
    }
    if !has("mode") {
        adds.push(("mode", "caller".into()));
    }
    if !has("pkt_size") {
        adds.push(("pkt_size", "1316".into()));
    }
    let mut u = base.trim_end_matches(['?', '&']).to_owned();
    let mut sep = if u.contains('?') { '&' } else { '?' };
    for (k, v) in adds {
        u.push(sep);
        u.push_str(k);
        u.push('=');
        u.push_str(&v);
        sep = '&';
    }
    u
}

/// Video bitrate bounds for live targets (Cloudflare Stream Live takes
/// 2-14 Mb/s combined; AAC adds 128 kb/s).
pub const LIVE_VIDEO_BPS: (u32, u32) = (2_000_000, 13_000_000);

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
    /// RTMP only: the `tcUrl` ffmpeg announces (`-rtmp_tcurl`) when the URL
    /// it connects to is not the one the server knows itself by: fastvideo
    /// connects ffmpeg to a pinned IP (plain RTMP) or to its own TLS relay on
    /// loopback (RTMPS). No credentials (the key is in the URL's path).
    pub rtmp_tc_url: Option<String>,
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
            rtmp_tc_url: None,
        }
    }

    fn gop_seconds(&self) -> f32 {
        match &self.target {
            SinkTarget::Hls { segment_s, .. } => (*segment_s).max(1) as f32,
            _ => 2.0,
        }
    }

    /// The video bitrate ffmpeg is given: live targets are clamped to
    /// [`LIVE_VIDEO_BPS`].
    pub fn effective_video_bps(&self) -> u32 {
        if self.target.is_live() {
            self.video_bitrate_bps.clamp(LIVE_VIDEO_BPS.0, LIVE_VIDEO_BPS.1)
        } else {
            self.video_bitrate_bps
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
        h.bitrate_bps = self.effective_video_bps();
        h.gop_seconds = self.gop_seconds();
        let mut v = self.encoder.stream_args(&h)?;
        if self.target.is_live() {
            // CBR with a filler-padded HRD for libx264 (NVENC's `-rc cbr`
            // already pads), and the closed-GOP flag for both.
            for x in v.iter_mut() {
                if x == "aud=1:repeat-headers=1" {
                    *x = "aud=1:repeat-headers=1:nal-hrd=cbr:force-cfr=1".into();
                }
            }
            v.extend(["-flags", "+cgop"].map(String::from));
        }
        a.extend(v);
        a.extend(["-c:a", "aac", "-profile:a", "aac_low", "-b:a"].map(String::from));
        a.extend([self.aac.bitrate_bps.to_string(), "-ar".into(), self.aac.rate.to_string()]);
        a.extend(["-ac".into(), self.aac.channels.to_string()]);
        match &self.target {
            SinkTarget::Rtmp { url } => {
                a.extend(["-flush_packets", "1", "-f", "flv", "-flvflags", "no_duration_filesize"].map(String::from));
                if let Some(tc) = &self.rtmp_tc_url {
                    a.extend(["-rtmp_tcurl".to_owned(), tc.clone()]);
                }
                a.push(url.clone());
            }
            SinkTarget::Srt { url } => {
                a.extend(["-flush_packets", "1", "-f", "mpegts", "-mpegts_flags", "+resend_headers"].map(String::from));
                a.push(url.clone());
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
    /// The last lines ffmpeg wrote to stderr (`-loglevel error`: its
    /// reason for failing), for the error of a failed [`FfmpegSink::finish`].
    stderr: Arc<Mutex<VecDeque<String>>>,
    stderr_reader: Option<std::thread::JoinHandle<()>>,
}

/// Lines of ffmpeg's stderr kept for an error message.
const STDERR_LINES: usize = 8;

/// An ffmpeg RTMP/HLS/file sink fed one tick at a time.
pub struct FfmpegSink {
    cfg: SinkConfig,
    run: Option<Running>,
    started_at: Instant,
    stats: SinkStats,
    /// Why the last ffmpeg exited, until [`FfmpegSink::exit_error`] takes it
    /// (recorded when a restart reaps it, so a quick restart cannot hide it).
    last_exit: Option<String>,
}

/// Why `r`'s ffmpeg exited, once it has: its exit status and its last
/// stderr lines (scrubbed), read after the stderr reader reached EOF.
fn exit_reason(r: &mut Running) -> Option<String> {
    let status = match r.child.try_wait() {
        Ok(Some(s)) => s,
        Ok(None) => return None,
        Err(e) => return Some(format!("ffmpeg: {e}")),
    };
    if let Some(t) = r.stderr_reader.take() {
        let _ = t.join();
    }
    let t = r.stderr.lock().unwrap_or_else(|p| p.into_inner());
    Some(if t.is_empty() {
        format!("ffmpeg exited ({status}) without an error message")
    } else {
        format!("ffmpeg exited ({status}): {}", t.iter().cloned().collect::<Vec<_>>().join(" | "))
    })
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
        Ok(Self { cfg, run: Some(run), started_at: Instant::now(), stats: SinkStats::default(), last_exit: None })
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
        cmd.args(cfg.ffmpeg_args(port)?).stdin(Stdio::piped()).stdout(Stdio::null()).stderr(Stdio::piped());
        let mut child = cmd.spawn().map_err(|e| MediaError::tool("ffmpeg", format!("not available: {e}")))?;
        let mut stdin = child.stdin.take().ok_or_else(|| MediaError::tool("ffmpeg", "no stdin"))?;
        let stderr = Arc::new(Mutex::new(VecDeque::new()));
        let stderr_reader = match child.stderr.take() {
            Some(pipe) => {
                let tail = stderr.clone();
                let secrets = cfg.target.secrets();
                Some(std::thread::Builder::new().name("sink-stderr".into()).spawn(move || {
                    for line in std::io::BufReader::new(pipe).lines().map_while(|l| l.ok()) {
                        let line = scrub(&line, &secrets);
                        let mut t = tail.lock().unwrap_or_else(|p| p.into_inner());
                        if t.len() == STDERR_LINES {
                            t.pop_front();
                        }
                        t.push_back(line);
                    }
                })?)
            }
            None => None,
        };
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
        Ok(Running { child, video_q, audio_q, threads, stop, stderr, stderr_reader })
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
        if let Some(mut r) = self.run.take() {
            if let Some(m) = exit_reason(&mut r) {
                self.last_exit = Some(m);
            }
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
        if let Some(t) = r.stderr_reader {
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
        // ffmpeg has exited: its stderr is at EOF.
        if let Some(t) = r.stderr_reader.take() {
            let _ = t.join();
        }
        match status {
            Some(s) if s.success() => Ok(self.stats),
            Some(s) => {
                let tail = r.stderr.lock().unwrap_or_else(|p| p.into_inner()).iter().cloned().collect::<Vec<_>>().join("\n");
                Err(MediaError::tool("ffmpeg", format!("sink exited with {s}: {tail}")))
            }
            None => Err(MediaError::tool("ffmpeg", "sink did not exit in time")),
        }
    }

    pub fn stats(&self) -> SinkStats {
        self.stats
    }

    /// ffmpeg's last stderr lines (scrubbed of credentials), e.g. why a
    /// live target refused the connection.
    pub fn last_error(&self) -> Option<String> {
        let r = self.run.as_ref()?;
        let t = r.stderr.lock().unwrap_or_else(|p| p.into_inner());
        (!t.is_empty()).then(|| t.iter().cloned().collect::<Vec<_>>().join(" | "))
    }

    /// Why ffmpeg exited: the current one if it is dead, else the last one
    /// a restart reaped (once). Its exit status and last stderr lines
    /// (scrubbed), read after the stderr reader reached EOF (reading them
    /// while the reader still drained the pipe gave an empty message).
    /// `None` when nothing exited since the last call.
    pub fn exit_error(&mut self) -> Option<String> {
        if let Some(m) = self.run.as_mut().and_then(exit_reason) {
            self.last_exit = Some(m);
        }
        self.last_exit.take()
    }

    /// Whether ffmpeg is running now (a dead one restarts on the next `send`).
    pub fn is_running(&mut self) -> bool {
        self.run.as_mut().is_some_and(|r| matches!(r.child.try_wait(), Ok(None)))
    }

    /// The target, for status (print [`SinkTarget::redacted`]).
    pub fn target(&self) -> &SinkTarget {
        &self.cfg.target
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
        assert!(a.contains("-c:a aac -profile:a aac_low -b:a 128000 -ar 48000 -ac 2"));
        assert!(a.contains("-c:v h264_nvenc"));
        assert!(a.contains("-rc cbr -b:v 6000000 -maxrate 6000000"), "{a}");
        assert!(a.contains("-g 48 -bf 0 -forced-idr 1 -no-scenecut 1"));
        assert!(a.contains("-profile:v baseline -level:v 3.2"));
        assert!(a.contains("-flags +cgop"));
        assert!(!a.contains("libx264"));
        assert!(a.ends_with("-f flv -flvflags no_duration_filesize rtmp://x/live/k"), "{a}");
    }

    #[test]
    fn rtmp_tc_url_names_the_real_server() {
        let mut c = SinkConfig::new(SinkTarget::Rtmp { url: "rtmp://127.0.0.1:40001/live/k".into() }, 64, 64, 16, None);
        assert!(!c.ffmpeg_args(None).unwrap().iter().any(|a| a == "-rtmp_tcurl"));
        c.rtmp_tc_url = Some("rtmps://live.example.com:443/live".into());
        let a = c.ffmpeg_args(None).unwrap().join(" ");
        assert!(a.ends_with("-rtmp_tcurl rtmps://live.example.com:443/live rtmp://127.0.0.1:40001/live/k"), "{a}");
    }

    #[test]
    fn srt_is_mpegts_with_cloudflare_settings() {
        let url = srt_url("srt://live.example.com:778", Some("publish:abc123"), Some("pass/word-123"), 1000);
        assert_eq!(
            url,
            "srt://live.example.com:778?streamid=publish:abc123&passphrase=pass/word-123&latency=1000000&mode=caller&pkt_size=1316"
        );
        assert!(srt_value_ok("publish:live/a") && !srt_value_ok("a&b") && !srt_value_ok("a b") && !srt_value_ok("50%"));
        // An existing query is kept and not doubled.
        assert_eq!(srt_url("srt://h:9000?latency=200000", None, None, 1000), "srt://h:9000?latency=200000&mode=caller&pkt_size=1316");
        let mut c = SinkConfig::new(SinkTarget::Srt { url: url.clone() }, 832, 480, 16, Some(SinkAudioIn { rate: 48_000, channels: 1 }));
        c.encoder = FfmpegH264::Libx264CpuTest;
        c.video_bitrate_bps = 500_000; // clamped up to Cloudflare's minimum
        let a = c.ffmpeg_args(Some(1)).unwrap().join(" ");
        assert!(a.contains("-b:v 2000000 -maxrate 2000000"), "{a}");
        assert!(a.contains("nal-hrd=cbr"), "{a}");
        assert!(a.contains("-g 32 -keyint_min 32 -sc_threshold 0"), "{a}");
        assert!(a.contains("-c:a aac -profile:a aac_low -b:a 128000 -ar 48000 -ac 2"), "{a}");
        assert!(a.ends_with(&format!("-f mpegts -mpegts_flags +resend_headers {url}")), "{a}");
        c.video_bitrate_bps = 40_000_000;
        assert_eq!(c.effective_video_bps(), 13_000_000);
    }

    #[test]
    fn credentials_never_print() {
        let r = SinkTarget::Rtmp { url: "rtmps://live.example.com:443/live/0123456789abcdefk".into() };
        assert_eq!(r.redacted(), "rtmps://live.example.com:443/live/<key>");
        assert!(!format!("{r:?}").contains("0123456789abcdefk"));
        let line = "rtmps://live.example.com:443/live/0123456789abcdefk: Input/output error";
        assert_eq!(scrub(line, &r.secrets()), "<redacted>: Input/output error");
        assert_eq!(scrub("key 0123456789abcdefk refused", &r.secrets()), "key <redacted> refused");
        let s = SinkTarget::Srt { url: srt_url("srt://user:pw1234@h:778", Some("streamid-xyz"), Some("secret-pass"), 500) };
        assert_eq!(s.redacted(), "srt://h:778");
        let dbg = format!("{s:?}");
        for x in ["streamid-xyz", "secret", "pw1234"] {
            assert!(!dbg.contains(x), "{dbg}");
        }
        let sec = s.secrets();
        assert!(sec.contains(&"secret-pass".to_owned()), "{sec:?}");
        assert_eq!(scrub("Connection to srt://h:778 (secret-pass) failed", &sec), "Connection to srt://h:778 (<redacted>) failed");
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
