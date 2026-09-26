//! Decoded frames on their way to disk: an ffmpeg rgb24 feed for
//! `output.mp4` and `frame-NNN.png` files.
//!
//! The two outputs have different jobs. The mp4 is the product, so it is fed
//! as frames arrive and a timed decode includes it (as sol-engine's
//! `video_vae_seconds` includes its x264 encode). The PNGs exist for
//! `compare-clips`; by default ([`PngMode::Deferred`]) their frames are kept
//! in host RAM while the decode runs and encoded only after ffmpeg has exited,
//! so PNG compression and the disk never push back into the decode. Each PNG
//! is synced to disk before [`VideoWriter::finish`] returns: no dirty page
//! cache from one clip is still being written back while the next one runs.
//!
//! `FASTVIDEO_PNG`: `deferred` (default), `inline` (encode while the decode
//! runs, the writer before deferral) or `off` (no PNGs).
//! `FASTVIDEO_PNG_BUFFER_GIB`: the deferral's host-RAM cap (default half of
//! the available memory, cgroup limit included); frames beyond it are encoded
//! inline. `FASTVIDEO_PNG_SYNC=0`: skip the per-file sync.
//! `FASTVIDEO_WRITER_TRACE=1`: log one line per batch (arrival time, ffmpeg
//! blocking, PNGs in flight, bytes buffered).

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::Instant;

use super::pipeline::{PipelineError, Result};

fn msg(s: impl Into<String>) -> PipelineError {
    PipelineError::Message(s.into())
}

/// When `frame-NNN.png` files are written.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PngMode {
    /// Buffered in host RAM (up to a budget) and encoded after the mp4.
    Deferred,
    /// Encoded while the decode runs, at most [`PNG_IN_FLIGHT`] at a time.
    Inline,
    /// None written.
    Off,
}

impl PngMode {
    /// `FASTVIDEO_PNG`, default [`PngMode::Deferred`].
    pub fn from_env() -> Self {
        match std::env::var("FASTVIDEO_PNG")
            .unwrap_or_default()
            .trim()
            .to_ascii_lowercase()
            .as_str()
        {
            "inline" => Self::Inline,
            "off" | "0" | "none" => Self::Off,
            _ => Self::Deferred,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Deferred => "deferred",
            Self::Inline => "inline",
            Self::Off => "off",
        }
    }
}

/// PNG encodes in flight at once while frames are still arriving.
pub const PNG_IN_FLIGHT: usize = 48;

/// Writers spawned and not yet finished or dropped, process-wide.
static ACTIVE: AtomicUsize = AtomicUsize::new(0);

/// Video writers that are still running (0 between clips: `finish` joins the
/// ffmpeg feed, ffmpeg and every PNG encode and sync before it returns).
pub fn active_writers() -> usize {
    ACTIVE.load(Ordering::SeqCst)
}

/// Where a writer's time went.
#[derive(Debug, Clone, Default)]
pub struct WriterStats {
    pub png_mode: &'static str,
    pub frames: usize,
    /// Worker time blocked writing rgb24 into ffmpeg's stdin.
    pub ffmpeg_feed_s: f64,
    /// [`VideoWriter::finish_video`]: from the last frame handed over to
    /// ffmpeg's exit (the mp4's tail).
    pub video_tail_s: f64,
    /// PNGs encoded while frames were still arriving (inline mode, or over
    /// the deferral budget).
    pub png_inline: usize,
    /// PNGs encoded (and synced) after the mp4, in [`VideoWriter::finish`].
    pub png_deferred: usize,
    /// Wall time of that deferred PNG phase.
    pub png_s: f64,
    pub png_bytes: u64,
    pub buffered_peak_bytes: usize,
    pub buffer_budget_bytes: usize,
    /// Writers already active when this one was spawned (should be 0).
    pub active_at_spawn: usize,
}

/// One chunk of packed frames on its way to disk.
struct FrameBatch {
    offset: usize,
    h: usize,
    w: usize,
    rgb: Vec<u8>,
}

/// Frames held back for the deferred PNG phase.
struct Held {
    offset: usize,
    h: usize,
    w: usize,
    rgb: Arc<Vec<u8>>,
}

/// What the feed thread hands back once ffmpeg has exited.
struct Fed {
    mp4: Option<String>,
    held: Vec<Held>,
    written: Vec<(usize, Result<(String, u64)>)>,
    stats: WriterStats,
}

/// Writes frames as they are decoded: raw rgb24 straight into an ffmpeg
/// process (when `fps > 0` and `mp4` is set) and `frame-NNN.png` files as
/// [`PngMode`] says. See the module docs.
pub struct VideoWriter {
    tx: Option<std::sync::mpsc::SyncSender<FrameBatch>>,
    worker: Option<std::thread::JoinHandle<Result<Fed>>>,
    fed: Option<Fed>,
    dir: Option<PathBuf>,
    sync: bool,
    active: bool,
    stats: WriterStats,
}

impl VideoWriter {
    /// `fps` and `mp4` decide whether an `output.mp4` is produced next to the
    /// frames; ffmpeg is spawned lazily on the first batch, once the frame
    /// size is known. PNGs as `FASTVIDEO_PNG` says.
    pub fn spawn(dir: &Path, fps: u32, mp4: bool) -> Result<Self> {
        Self::spawn_with(dir, fps, mp4, None, PngMode::from_env())
    }

    /// As [`Self::spawn`], muxing `audio` (a WAV already on disk) into the
    /// mp4 as AAC. The audio-video models decode their audio first, which
    /// takes milliseconds, so the track exists before the first frame does
    /// and ffmpeg can still be fed frames as they decode.
    pub fn spawn_with_audio(dir: &Path, fps: u32, mp4: bool, audio: Option<&Path>) -> Result<Self> {
        Self::spawn_with(dir, fps, mp4, audio, PngMode::from_env())
    }

    pub fn spawn_with(
        dir: &Path,
        fps: u32,
        mp4: bool,
        audio: Option<&Path>,
        png: PngMode,
    ) -> Result<Self> {
        let budget = match png {
            PngMode::Deferred => buffer_budget(),
            _ => 0,
        };
        Self::spawn_budget(dir, fps, mp4, audio, png, budget)
    }

    /// [`Self::spawn_with`] with an explicit deferral budget in bytes.
    fn spawn_budget(
        dir: &Path,
        fps: u32,
        mp4: bool,
        audio: Option<&Path>,
        png: PngMode,
        budget: usize,
    ) -> Result<Self> {
        std::fs::create_dir_all(dir).map_err(|e| msg(e.to_string()))?;
        if let Some(a) = audio {
            if !a.is_file() {
                return Err(msg(format!("audio track {} does not exist", a.display())));
            }
        }
        let cfg = FeedConfig {
            dir: dir.to_path_buf(),
            fps,
            mp4,
            audio: audio.map(Path::to_path_buf),
            png,
            budget,
            trace: super::envflag::bool_flag("FASTVIDEO_WRITER_TRACE", false),
        };
        Self::start(
            "fv-video-writer",
            Some(dir.to_path_buf()),
            png,
            budget,
            move |rx| feed(rx, &cfg),
        )
    }

    /// A writer that accepts frames and drops them (decode benchmarks).
    pub fn spawn_discard() -> Result<Self> {
        Self::start("fv-video-discard", None, PngMode::Off, 0, |rx| {
            let mut frames = 0;
            for b in rx {
                let b: FrameBatch = b;
                frames += b.rgb.len() / (b.h * b.w * 3).max(1);
            }
            Ok(Fed {
                mp4: None,
                held: Vec::new(),
                written: Vec::new(),
                stats: WriterStats {
                    frames,
                    ..WriterStats::default()
                },
            })
        })
    }

    fn start(
        name: &str,
        dir: Option<PathBuf>,
        png: PngMode,
        budget: usize,
        body: impl FnOnce(std::sync::mpsc::Receiver<FrameBatch>) -> Result<Fed> + Send + 'static,
    ) -> Result<Self> {
        // Two chunks of backlog between the caller and the feed thread.
        let (tx, rx) = std::sync::mpsc::sync_channel::<FrameBatch>(2);
        let worker = std::thread::Builder::new()
            .name(name.into())
            .spawn(move || body(rx))
            .map_err(|e| msg(e.to_string()))?;
        let active_at_spawn = ACTIVE.fetch_add(1, Ordering::SeqCst);
        if active_at_spawn > 0 {
            super::log::info(format_args!(
                "video writer: {active_at_spawn} earlier writer(s) still active at spawn"
            ));
        }
        Ok(Self {
            tx: Some(tx),
            worker: Some(worker),
            fed: None,
            dir,
            sync: super::envflag::bool_flag("FASTVIDEO_PNG_SYNC", true),
            active: true,
            stats: WriterStats {
                png_mode: png.as_str(),
                buffer_budget_bytes: budget,
                active_at_spawn,
                ..WriterStats::default()
            },
        })
    }

    /// Queue `[frames, h, w, 3]` bytes starting at frame `offset`. Batches must
    /// arrive in frame order; the mp4 is written in the order pushed.
    pub fn push(&mut self, offset: usize, h: usize, w: usize, rgb: Vec<u8>) -> Result<()> {
        let tx = self
            .tx
            .as_ref()
            .ok_or_else(|| msg("video writer already finished"))?;
        if tx.send(FrameBatch { offset, h, w, rgb }).is_err() {
            // The worker is gone; its error is the one worth reporting.
            return match self.finish_video() {
                Ok(_) => Err(msg("video writer stopped early")),
                Err(e) => Err(e),
            };
        }
        Ok(())
    }

    /// Wait until every queued frame is in ffmpeg and ffmpeg has exited (the
    /// mp4 is complete); return its path, if one was requested. Deferred PNG
    /// frames stay in memory until [`Self::finish`].
    pub fn finish_video(&mut self) -> Result<Option<String>> {
        if let Some(fed) = &self.fed {
            return Ok(fed.mp4.clone());
        }
        let timer = Instant::now();
        drop(self.tx.take());
        let fed = match self.worker.take() {
            Some(h) => h.join().map_err(|_| msg("video writer thread panicked"))?,
            None => Err(msg("video writer already finished")),
        };
        let fed = match fed {
            Ok(f) => f,
            Err(e) => {
                self.release();
                return Err(e);
            }
        };
        let tail = timer.elapsed().as_secs_f64();
        let (mode, budget, active) = (
            self.stats.png_mode,
            self.stats.buffer_budget_bytes,
            self.stats.active_at_spawn,
        );
        self.stats = WriterStats {
            png_mode: mode,
            buffer_budget_bytes: budget,
            active_at_spawn: active,
            video_tail_s: tail,
            ..fed.stats.clone()
        };
        let mp4 = fed.mp4.clone();
        self.fed = Some(fed);
        Ok(mp4)
    }

    /// [`Self::finish_video`], then write every deferred PNG (synced to disk)
    /// and return the PNG paths in frame order plus the mp4 path. Nothing of
    /// this writer runs after it returns.
    pub fn finish(&mut self) -> Result<(Vec<String>, Option<String>)> {
        self.finish_video()?;
        let fed = self
            .fed
            .take()
            .ok_or_else(|| msg("video writer already finished"))?;
        let Fed {
            mp4,
            held,
            mut written,
            ..
        } = fed;
        let timer = Instant::now();
        let dir = self.dir.clone();
        let sync = self.sync;
        if let Some(dir) = dir.as_deref() {
            use rayon::prelude::*;
            let jobs: Vec<(&Held, usize)> = held
                .iter()
                .flat_map(|b| (0..b.rgb.len() / (b.h * b.w * 3)).map(move |i| (b, i)))
                .collect();
            let done: Vec<(usize, Result<(String, u64)>)> = jobs
                .par_iter()
                .map(|&(b, i)| {
                    let n = b.h * b.w * 3;
                    let frame = &b.rgb[i * n..(i + 1) * n];
                    (
                        b.offset + i,
                        write_png(dir, b.offset + i, b.w, b.h, frame, sync),
                    )
                })
                .collect();
            self.stats.png_deferred = done.len();
            written.extend(done);
        }
        drop(held);
        self.stats.png_s = timer.elapsed().as_secs_f64();
        self.release();
        written.sort_by_key(|(i, _)| *i);
        let mut paths = Vec::with_capacity(written.len());
        for (_, r) in written {
            let (p, bytes) = r?;
            self.stats.png_bytes += bytes;
            paths.push(p);
        }
        let s = &self.stats;
        if s.frames > 0 {
            super::log::info(format_args!(
                "video writer: {} frames, png {} ({} inline, {} after the mp4 in {:.2}s, {:.2} GiB{}), ffmpeg feed blocked {:.2}s, mp4 tail {:.2}s, buffered peak {:.2} GiB of {:.2}",
                s.frames,
                s.png_mode,
                s.png_inline,
                s.png_deferred,
                s.png_s,
                s.png_bytes as f64 / f64::from(1u32 << 30),
                if sync && s.png_deferred > 0 { ", synced" } else { "" },
                s.ffmpeg_feed_s,
                s.video_tail_s,
                s.buffered_peak_bytes as f64 / f64::from(1u32 << 30),
                s.buffer_budget_bytes as f64 / f64::from(1u32 << 30),
            ));
        }
        Ok((paths, mp4))
    }

    /// Where this writer's time went (complete after [`Self::finish`]).
    pub fn stats(&self) -> &WriterStats {
        &self.stats
    }

    fn release(&mut self) {
        if std::mem::take(&mut self.active) {
            ACTIVE.fetch_sub(1, Ordering::SeqCst);
        }
    }
}

impl Drop for VideoWriter {
    fn drop(&mut self) {
        drop(self.tx.take());
        if let Some(h) = self.worker.take() {
            let _ = h.join();
        }
        self.release();
    }
}

struct FeedConfig {
    dir: PathBuf,
    fps: u32,
    mp4: bool,
    audio: Option<PathBuf>,
    png: PngMode,
    budget: usize,
    trace: bool,
}

/// The feed thread: rgb24 into ffmpeg in order, PNGs encoded inline or held
/// for later, then ffmpeg closed and waited for.
fn feed(rx: std::sync::mpsc::Receiver<FrameBatch>, cfg: &FeedConfig) -> Result<Fed> {
    use std::io::Write as _;

    let started = Instant::now();
    let out = cfg.dir.join("output.mp4");
    let mut ffmpeg: Option<std::process::Child> = None;
    let mut stats = WriterStats::default();
    let mut held: Vec<Held> = Vec::new();
    let mut held_bytes = 0usize;
    // Inline PNG encodes run on the rayon pool across batches, at most
    // PNG_IN_FLIGHT frames at once; this thread feeds ffmpeg meanwhile.
    let slots = Arc::new((Mutex::new(0usize), Condvar::new()));
    let written: Mutex<Vec<(usize, Result<(String, u64)>)>> = Mutex::new(Vec::new());
    let dir = cfg.dir.as_path();
    let fed: Result<()> = rayon::in_place_scope(|s| {
        for batch in rx {
            let arrived = started.elapsed().as_secs_f64();
            let FrameBatch { offset, h, w, rgb } = batch;
            let frame_bytes = h * w * 3;
            if frame_bytes == 0 || rgb.len() % frame_bytes != 0 {
                return Err(msg(format!(
                    "frame batch of {} bytes is not whole {w}x{h} RGB frames",
                    rgb.len()
                )));
            }
            let n = rgb.len() / frame_bytes;
            stats.frames += n;
            if cfg.mp4 && cfg.fps > 0 && ffmpeg.is_none() {
                ffmpeg = Some(spawn_ffmpeg_rgb(&out, w, h, cfg.fps, cfg.audio.as_deref())?);
            }
            let rgb = Arc::new(rgb);
            let inline = match cfg.png {
                PngMode::Off => false,
                PngMode::Inline => true,
                PngMode::Deferred => held_bytes + rgb.len() > cfg.budget,
            };
            if inline {
                stats.png_inline += n;
                for i in 0..n {
                    {
                        let (k, cv) = &*slots;
                        let mut k = k.lock().expect("png slots");
                        while *k >= PNG_IN_FLIGHT {
                            k = cv.wait(k).expect("png slots");
                        }
                        *k += 1;
                    }
                    let (rgb, slots, written) = (rgb.clone(), slots.clone(), &written);
                    s.spawn(move |_| {
                        let frame = &rgb[i * frame_bytes..(i + 1) * frame_bytes];
                        let r = write_png(dir, offset + i, w, h, frame, false);
                        written.lock().expect("png results").push((offset + i, r));
                        let (k, cv) = &*slots;
                        *k.lock().expect("png slots") -= 1;
                        cv.notify_one();
                    });
                }
            } else if cfg.png == PngMode::Deferred {
                held_bytes += rgb.len();
                stats.buffered_peak_bytes = stats.buffered_peak_bytes.max(held_bytes);
                held.push(Held {
                    offset,
                    h,
                    w,
                    rgb: rgb.clone(),
                });
            }
            let mut blocked = 0.0;
            if let Some(child) = ffmpeg.as_mut() {
                let stdin = child
                    .stdin
                    .as_mut()
                    .ok_or_else(|| msg("ffmpeg stdin closed"))?;
                let t = Instant::now();
                stdin
                    .write_all(&rgb)
                    .map_err(|e| msg(format!("ffmpeg stdin: {e}")))?;
                blocked = t.elapsed().as_secs_f64();
                stats.ffmpeg_feed_s += blocked;
            }
            if cfg.trace {
                let in_flight = *slots.0.lock().expect("png slots");
                super::log::info(format_args!(
                    "video writer trace: t {arrived:.2}s frames {offset}+{n} ffmpeg {blocked:.2}s png in flight {in_flight} held {:.2} GiB",
                    held_bytes as f64 / f64::from(1u32 << 30)
                ));
            }
        }
        Ok(())
    });
    fed?;
    let mp4 = match ffmpeg {
        Some(mut child) => {
            drop(child.stdin.take());
            let status = child.wait().map_err(|e| msg(format!("ffmpeg: {e}")))?;
            if !status.success() {
                return Err(msg(format!("ffmpeg failed with {status}")));
            }
            Some(out.to_string_lossy().into_owned())
        }
        None => None,
    };
    Ok(Fed {
        mp4,
        held,
        written: written.into_inner().expect("png results"),
        stats,
    })
}

/// `dir/frame-NNN.png` (the image crate's default PNG settings, so every
/// mode writes the same bytes), optionally synced; returns path and size.
fn write_png(
    dir: &Path,
    index: usize,
    w: usize,
    h: usize,
    frame: &[u8],
    sync: bool,
) -> Result<(String, u64)> {
    use image::ImageEncoder as _;
    use std::io::Write as _;
    let path = dir.join(format!("frame-{index:03}.png"));
    let err = |e: &dyn std::fmt::Display| msg(format!("{}: {e}", path.display()));
    if frame.len() != w * h * 3 {
        return Err(msg("rgb buffer size mismatch"));
    }
    let file = std::fs::File::create(&path).map_err(|e| err(&e))?;
    let mut out = std::io::BufWriter::with_capacity(1 << 20, file);
    image::codecs::png::PngEncoder::new(&mut out)
        .write_image(frame, w as u32, h as u32, image::ExtendedColorType::Rgb8)
        .map_err(|e| err(&e))?;
    out.flush().map_err(|e| err(&e))?;
    let file = out.into_inner().map_err(|e| err(&e.error().to_string()))?;
    if sync {
        file.sync_data().map_err(|e| err(&e))?;
    }
    let bytes = file.metadata().map(|m| m.len()).unwrap_or(0);
    Ok((path.to_string_lossy().into_owned(), bytes))
}

/// Host RAM the deferred PNG frames may hold: `FASTVIDEO_PNG_BUFFER_GIB`, else
/// half of what is available (the smaller of `MemAvailable` and the cgroup's
/// limit less its anonymous memory), else 8 GiB.
fn buffer_budget() -> usize {
    const GIB: f64 = (1u64 << 30) as f64;
    if let Some(g) = std::env::var("FASTVIDEO_PNG_BUFFER_GIB")
        .ok()
        .and_then(|v| v.trim().parse::<f64>().ok())
    {
        return (g.max(0.0) * GIB) as usize;
    }
    match available_memory() {
        Some(b) => (b / 2) as usize,
        None => (8.0 * GIB) as usize,
    }
}

/// Bytes of host memory this process could still take.
pub fn available_memory() -> Option<u64> {
    let read = |p: &str| std::fs::read_to_string(p).ok();
    let meminfo = read("/proc/meminfo").and_then(|s| {
        s.lines()
            .find(|l| l.starts_with("MemAvailable:"))
            .and_then(|l| l.split_whitespace().nth(1))
            .and_then(|kb| kb.parse::<u64>().ok())
            .map(|kb| kb * 1024)
    });
    // cgroup v2: limit less anonymous memory (page cache is reclaimable).
    let cgroup = read("/sys/fs/cgroup/memory.max")
        .and_then(|s| s.trim().parse::<u64>().ok())
        .map(|max| {
            let anon = read("/sys/fs/cgroup/memory.stat")
                .and_then(|s| {
                    s.lines()
                        .find(|l| l.starts_with("anon "))
                        .and_then(|l| l.split_whitespace().nth(1))
                        .and_then(|v| v.parse::<u64>().ok())
                })
                .unwrap_or(0);
            max.saturating_sub(anon)
        });
    match (meminfo, cgroup) {
        (Some(a), Some(b)) => Some(a.min(b)),
        (a, b) => a.or(b),
    }
}

fn spawn_ffmpeg_rgb(
    out: &Path,
    w: usize,
    h: usize,
    fps: u32,
    audio: Option<&Path>,
) -> Result<std::process::Child> {
    let mut cmd = Command::new("ffmpeg");
    cmd.args([
        "-y",
        "-loglevel",
        "error",
        "-nostats",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
    ])
    .arg(format!("{w}x{h}"))
    .args(["-framerate", &fps.to_string(), "-i", "-"]);
    if let Some(a) = audio {
        // Input 1. Mapped explicitly so the track is never dropped silently,
        // and the mp4 ends with the shorter stream: the two decoders round
        // their lengths differently by a few milliseconds.
        cmd.arg("-i").arg(a).args([
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-shortest",
        ]);
    }
    // The reference's H.264 settings (ltx_pipelines media_io/encode.py
    // `encode_video`: libx264, crf 19, preset veryfast, yuv420p); x264's
    // default preset keeps a 4K clip's encode far behind the decode.
    cmd.args([
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "19", "-pix_fmt", "yuv420p",
    ])
    .arg(out)
    .stdin(std::process::Stdio::piped())
    .spawn()
    .map_err(|e| msg(format!("ffmpeg not available: {e}")))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp(tag: &str) -> PathBuf {
        std::env::temp_dir().join(format!("fv-writer-{tag}-{}", std::process::id()))
    }

    /// Every PNG mode writes the same files (deferral changes when, not what),
    /// and a finished writer leaves nothing running.
    #[test]
    fn png_modes_write_the_same_frames() {
        let (h, w) = (5usize, 7usize);
        let frames: Vec<u8> = (0..4 * h * w * 3).map(|i| (i * 37 % 251) as u8).collect();
        let mut outputs = Vec::new();
        for mode in [PngMode::Inline, PngMode::Deferred] {
            let dir = tmp(mode.as_str());
            let _ = std::fs::remove_dir_all(&dir);
            let mut writer =
                VideoWriter::spawn_budget(&dir, 0, false, None, mode, 1 << 20).unwrap();
            writer
                .push(0, h, w, frames[..3 * h * w * 3].to_vec())
                .unwrap();
            writer
                .push(3, h, w, frames[3 * h * w * 3..].to_vec())
                .unwrap();
            assert!(writer.finish_video().unwrap().is_none());
            let (paths, _) = writer.finish().unwrap();
            assert_eq!(paths.len(), 4);
            let s = writer.stats();
            assert_eq!(s.frames, 4);
            match mode {
                PngMode::Inline => assert_eq!((s.png_inline, s.png_deferred), (4, 0)),
                _ => assert_eq!((s.png_inline, s.png_deferred), (0, 4)),
            }
            outputs.push(
                paths
                    .iter()
                    .map(|p| std::fs::read(p).unwrap())
                    .collect::<Vec<_>>(),
            );
            let _ = std::fs::remove_dir_all(&dir);
        }
        assert_eq!(outputs[0], outputs[1]);
    }

    /// Over the deferral budget, frames are encoded inline instead of held.
    #[test]
    fn deferral_spills_over_its_budget() {
        let (h, w) = (4usize, 4usize);
        let dir = tmp("spill");
        let _ = std::fs::remove_dir_all(&dir);
        let mut writer =
            VideoWriter::spawn_budget(&dir, 0, false, None, PngMode::Deferred, 0).unwrap();
        writer.push(0, h, w, vec![9u8; 2 * h * w * 3]).unwrap();
        let (paths, _) = writer.finish().unwrap();
        assert_eq!(paths.len(), 2);
        assert_eq!(writer.stats().png_inline, 2);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn png_off_writes_no_frames() {
        let dir = tmp("off");
        let _ = std::fs::remove_dir_all(&dir);
        let mut writer = VideoWriter::spawn_with(&dir, 0, false, None, PngMode::Off).unwrap();
        writer.push(0, 2, 2, vec![0u8; 12]).unwrap();
        let (paths, mp4) = writer.finish().unwrap();
        assert!(paths.is_empty() && mp4.is_none());
        assert_eq!(writer.stats().frames, 1);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
