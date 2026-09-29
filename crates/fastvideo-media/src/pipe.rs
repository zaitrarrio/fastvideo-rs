//! ffmpeg pipe-encoder processes and the session's warm spare (design §5.1,
//! §5.10).
//!
//! The H.264 ([`crate::video::PipeEncoder`]: NVENC, libx264) and VP8
//! ([`crate::vp8::Vp8Encoder`]: libvpx) pipe encoders have no per-frame
//! keyframe control, so a forced keyframe restarts the ffmpeg process. A
//! cold ffmpeg start takes 1-7 s on a loaded host (exec, ~200 shared
//! libraries, encoder init), and video freezes for that long.
//!
//! A streaming session therefore owns a [`SparePool`]: at most one **warm
//! spare**, an ffmpeg process for one encoder profile (the full argument
//! list: codec, size, fps, bitrate, level), started ahead of need and
//! pre-initialised with one black primer frame whose output is discarded;
//! its next frame is a forced keyframe (`-force_key_frames expr:eq(n,1)`).
//!
//! - **Session start**: the session may pre-warm the pool for the profile it
//!   expects ([`SparePool::prewarm_with`], off the media thread); the first
//!   encoder of that profile adopts the spare instead of starting ffmpeg.
//! - **Keyframe restart**: the old process gets EOF, so it flushes the frames
//!   it still holds (libvpx holds its last frame until more input arrives);
//!   they are returned, in order, before the new process's first frame,
//!   without the encoder thread waiting for them. The spare is swapped in
//!   (no process start on the encoder thread), or ffmpeg starts cold when
//!   there is none for this profile.
//! - **Refill**: once the current process has output its first frame, the
//!   encoder asks for a spare of its profile, so a spare never competes
//!   with the start of the process that serves the session.
//! - A keyframe asked for while the current process has not output its
//!   first frame (always a keyframe) needs no restart; pooled encoders skip
//!   it.
//!
//! Bounds: one spare per pool (a refill for another profile replaces it);
//! pools belong to streaming sessions only (batch encoders have none, and
//! behave as before); the spare is killed when the last handle to the pool
//! goes (session end). `FV_ENCODER_SPARE=0` disables pools.
//!
//! Every restart is measured from the restart to the new process's first
//! output: `fv_encoder_restart_duration_seconds{codec,spare}` and
//! `fv_encoder_restarts_total{codec,spare}`, with `spare` = `warm` (the
//! spare was primed), `warming` (it was still starting) or `cold`, plus one
//! log line per restart (and per encoder start from a spare).

use std::collections::VecDeque;
use std::io::{Read, Write};
use std::process::{Child, ChildStdin, Stdio};
use std::sync::{mpsc, Arc, Condvar, Mutex, OnceLock, Weak};
use std::time::{Duration, Instant};

use crate::error::{MediaError, Result};
use crate::h264;
use crate::tools;

/// How the ffmpeg output is split into frames.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Framer {
    /// H.264 Annex-B with an AUD at the start of every access unit.
    AnnexB,
    /// IVF (one VP8 frame per record).
    Ivf,
}

/// A process whose EOF went unanswered this long is killed (its remaining
/// output is lost), so a hung ffmpeg cannot hold the new one's frames back.
const DRAIN_TIMEOUT: Duration = Duration::from_secs(15);

/// How long taking from a pool waits for a pre-warm in progress (a codec
/// probe and a process start).
const PREWARM_WAIT: Duration = Duration::from_secs(15);

/// Where a process came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SpareKind {
    /// The spare had started and taken its primer frame.
    Warm,
    /// The spare was still starting; the restart waited for it.
    Warming,
    /// No spare: ffmpeg started on the spot.
    Cold,
}

impl SpareKind {
    pub fn as_str(self) -> &'static str {
        match self {
            SpareKind::Warm => "warm",
            SpareKind::Warming => "warming",
            SpareKind::Cold => "cold",
        }
    }
}

/// Process starts and keyframe restarts of one pipe encoder.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct RestartStats {
    /// How the encoder's first process started.
    pub start: Option<SpareKind>,
    pub restarts: u64,
    pub warm: u64,
    pub warming: u64,
    pub cold: u64,
    /// Keyframe requests that needed no restart: the current process had
    /// not output its first frame, a keyframe, yet (pooled encoders).
    pub skipped: u64,
    /// Restart to the new process's first frame, per completed restart.
    pub latencies: Vec<(SpareKind, Duration)>,
}

impl RestartStats {
    fn count(&mut self, k: SpareKind) {
        self.restarts += 1;
        match k {
            SpareKind::Warm => self.warm += 1,
            SpareKind::Warming => self.warming += 1,
            SpareKind::Cold => self.cold += 1,
        }
    }
}

/// What a spare is started with: one encoder profile.
#[derive(Debug, Clone, PartialEq)]
pub struct SpareSpec {
    /// Metrics / log label (`h264_nvenc`, `libx264`, `libvpx`).
    pub(crate) codec: &'static str,
    /// The encoder's ffmpeg arguments (the profile; without the primer).
    pub(crate) args: Vec<String>,
    pub(crate) framer: Framer,
    /// Bytes of one input frame (the primer).
    pub(crate) frame_len: usize,
}

impl SpareSpec {
    /// The profile of an H.264 pipe encoder.
    pub fn h264(codec: crate::video::FfmpegH264, cfg: &crate::video::H264Config) -> Result<Self> {
        cfg.validate()?;
        Ok(Self {
            codec: crate::video::pipe_label(codec),
            args: crate::video::pipe_encoder_args(cfg, codec)?,
            framer: Framer::AnnexB,
            frame_len: cfg.input_width as usize * cfg.input_height as usize * 3,
        })
    }

    /// The profile of a libvpx VP8 encoder.
    pub fn vp8(cfg: &crate::vp8::Vp8Config) -> Result<Self> {
        cfg.validate()?;
        Ok(Self {
            codec: "libvpx",
            args: crate::vp8::ffmpeg_args(cfg),
            framer: Framer::Ivf,
            frame_len: cfg.width as usize * cfg.height as usize * 3,
        })
    }
}

/// A streaming session's warm spare: at most one pre-started, primed ffmpeg
/// encoder process, shared by the session's pipe encoders. Clones share it;
/// the spare is killed when the last clone is dropped.
#[derive(Clone)]
pub struct SparePool(Arc<PoolShared>);

struct PoolShared {
    enabled: bool,
    st: Mutex<PoolState>,
    cv: Condvar,
}

#[derive(Default)]
struct PoolState {
    spare: Option<Spare>,
    /// Pre-warms in progress (a take waits for them).
    pending: u32,
}

impl std::fmt::Debug for SparePool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let st = self.lock();
        f.debug_struct("SparePool")
            .field("enabled", &self.0.enabled)
            .field("spare", &st.spare.as_ref().map(|s| (s.spec.codec, s.proc.child.id())))
            .field("pending", &st.pending)
            .finish()
    }
}

impl SparePool {
    /// A pool that keeps a spare (`enabled`) or never does.
    pub fn new(enabled: bool) -> Self {
        Self(Arc::new(PoolShared { enabled, st: Mutex::new(PoolState::default()), cv: Condvar::new() }))
    }

    /// One session's pool; disabled with `FV_ENCODER_SPARE=0`.
    pub fn per_session() -> Self {
        Self::new(!std::env::var("FV_ENCODER_SPARE").is_ok_and(|v| v.trim() == "0"))
    }

    pub fn enabled(&self) -> bool {
        self.0.enabled
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, PoolState> {
        self.0.st.lock().unwrap_or_else(|p| p.into_inner())
    }

    /// Start a spare for `spec` now, replacing one of another profile.
    pub fn prewarm(&self, spec: SpareSpec) {
        if !self.0.enabled {
            return;
        }
        let mut st = self.lock();
        if st.spare.as_ref().is_some_and(|s| s.spec.args == spec.args) {
            return;
        }
        st.spare = None;
        match Spare::start(spec) {
            Ok(s) => st.spare = Some(s),
            Err(e) => tracing::warn!(error = %e, "cannot start the encoder spare"),
        }
    }

    /// Pre-warm on a background thread: `spec` (which may probe codecs)
    /// says what to start, if anything. Encoders created meanwhile wait for
    /// it (at most 15 s) rather than start ffmpeg a second time.
    pub fn prewarm_with(&self, spec: impl FnOnce() -> Option<SpareSpec> + Send + 'static) {
        if !self.0.enabled {
            return;
        }
        self.lock().pending += 1;
        let weak: Weak<PoolShared> = Arc::downgrade(&self.0);
        let spawned = std::thread::Builder::new().name("ffmpeg-spare-prewarm".into()).spawn(move || {
            let spec = spec();
            let Some(pool) = weak.upgrade().map(SparePool) else { return };
            if let Some(spec) = spec {
                pool.prewarm(spec);
            }
            pool.lock().pending -= 1;
            pool.0.cv.notify_all();
        });
        if spawned.is_err() {
            self.lock().pending -= 1;
        }
    }

    /// The spare's process id and whether it is primed.
    pub fn spare_state(&self) -> Option<(u32, bool)> {
        let mut st = self.lock();
        let s = st.spare.as_mut()?;
        Some((s.proc.child.id(), s.is_primed()))
    }

    /// The spare for `spec`'s profile, if there is one (after waiting for a
    /// pre-warm in progress). A spare of another profile stays.
    fn take(&self, spec: &SpareSpec) -> Option<Spare> {
        if !self.0.enabled {
            return None;
        }
        let deadline = Instant::now() + PREWARM_WAIT;
        let mut st = self.lock();
        while st.pending > 0 {
            let now = Instant::now();
            if now >= deadline {
                break;
            }
            st = self.0.cv.wait_timeout(st, deadline - now).map(|r| r.0).unwrap_or_else(|p| p.into_inner().0);
        }
        if st.spare.as_ref().is_some_and(|s| s.spec.args == spec.args) {
            st.spare.take()
        } else {
            None
        }
    }
}

/// One ffmpeg encoder process: frames on stdin, framed output from a reader
/// thread.
pub(crate) struct Proc {
    child: Child,
    stdin: Option<ChildStdin>,
    rx: mpsc::Receiver<Result<Vec<u8>>>,
    reader: Option<std::thread::JoinHandle<()>>,
    /// When the first (non-skipped) frame came out.
    first_out: Arc<OnceLock<Instant>>,
    eof_at: Option<Instant>,
}

impl Proc {
    /// Starts ffmpeg with `args` (after the common quiet flags); the reader
    /// discards the first `skip` frames (a primer's output).
    pub(crate) fn spawn(args: &[String], framer: Framer, skip: usize) -> Result<Self> {
        let mut cmd = tools::ffmpeg_command();
        cmd.args(args).stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::null());
        let mut child = cmd.spawn().map_err(|e| MediaError::tool("ffmpeg", format!("not available: {e}")))?;
        let stdin = child.stdin.take();
        let stdout = child.stdout.take().ok_or_else(|| MediaError::tool("ffmpeg", "no stdout"))?;
        let (tx, rx) = mpsc::channel();
        let first_out = Arc::new(OnceLock::new());
        let first = first_out.clone();
        let name = match framer {
            Framer::AnnexB => "h264-ffmpeg-out",
            Framer::Ivf => "vp8-ffmpeg-out",
        };
        let reader =
            std::thread::Builder::new().name(name.into()).spawn(move || read_frames(stdout, framer, skip, &tx, &first));
        let reader = match reader {
            Ok(r) => r,
            Err(e) => {
                let _ = child.kill();
                let _ = child.wait();
                return Err(e.into());
            }
        };
        Ok(Self { child, stdin, rx, reader: Some(reader), first_out, eof_at: None })
    }

    pub(crate) fn write(&mut self, data: &[u8]) -> Result<()> {
        let stdin = self.stdin.as_mut().ok_or_else(|| MediaError::Encode("encoder finished".into()))?;
        stdin.write_all(data).map_err(|e| MediaError::tool("ffmpeg", format!("stdin: {e}")))
    }

    /// Close stdin: ffmpeg encodes what it has and exits.
    fn eof(&mut self) {
        drop(self.stdin.take());
        self.eof_at.get_or_insert_with(Instant::now);
    }

    /// The frames ready now, and whether the output ended.
    fn try_take(&mut self) -> Result<(Vec<Vec<u8>>, bool)> {
        let mut out = Vec::new();
        loop {
            match self.rx.try_recv() {
                Ok(f) => out.push(f?),
                Err(mpsc::TryRecvError::Empty) => return Ok((out, false)),
                Err(mpsc::TryRecvError::Disconnected) => return Ok((out, true)),
            }
        }
    }

    /// After the output ended: reap the process.
    fn reap(mut self, what: &str) -> Result<()> {
        if let Some(r) = self.reader.take() {
            let _ = r.join();
        }
        let status = self.child.wait()?;
        if !status.success() {
            return Err(MediaError::tool("ffmpeg", format!("{what} encode exited with {status}")));
        }
        Ok(())
    }

    /// Close stdin, drain every frame, reap the process.
    fn close(mut self, what: &str) -> Result<Vec<Vec<u8>>> {
        self.eof();
        let frames: Vec<Vec<u8>> = self.rx.iter().collect::<Result<_>>()?;
        self.reap(what)?;
        Ok(frames)
    }

    fn alive(&mut self) -> bool {
        matches!(self.child.try_wait(), Ok(None))
    }
}

impl Drop for Proc {
    fn drop(&mut self) {
        drop(self.stdin.take());
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn read_frames(
    mut stdout: impl Read,
    framer: Framer,
    mut skip: usize,
    tx: &mpsc::Sender<Result<Vec<u8>>>,
    first: &OnceLock<Instant>,
) {
    let mut buf = Vec::new();
    let mut header_done = false;
    let mut chunk = vec![0u8; 1 << 16];
    let emit = |f: Vec<u8>, skip: &mut usize| -> bool {
        if *skip > 0 {
            *skip -= 1;
            return true;
        }
        let _ = first.set(Instant::now());
        tx.send(Ok(f)).is_ok()
    };
    loop {
        match stdout.read(&mut chunk) {
            Ok(0) | Err(_) => break,
            Ok(n) => {
                buf.extend_from_slice(&chunk[..n]);
                let frames = match framer {
                    Framer::AnnexB => Ok(take_complete_aus(&mut buf)),
                    Framer::Ivf => crate::vp8::take_ivf_frames(&mut buf, &mut header_done),
                };
                match frames {
                    Ok(frames) => {
                        for f in frames {
                            if !emit(f, &mut skip) {
                                return;
                            }
                        }
                    }
                    Err(e) => {
                        let _ = tx.send(Err(e));
                        return;
                    }
                }
            }
        }
    }
    // The last access unit has no AUD after it.
    if framer == Framer::AnnexB && !buf.is_empty() {
        emit(buf, &mut skip);
    }
}

/// Pop every access unit that is followed by the next AUD from `buf`.
pub(crate) fn take_complete_aus(buf: &mut Vec<u8>) -> Vec<Vec<u8>> {
    let mut starts = Vec::new();
    let mut i = 0;
    while i + 4 <= buf.len() {
        if buf[i] == 0 && buf[i + 1] == 0 && buf[i + 2] == 1 && buf[i + 3] & 0x1f == h264::nal::AUD {
            // Include the leading zero of a 4-byte start code.
            let s = if i > 0 && buf[i - 1] == 0 { i - 1 } else { i };
            starts.push(s);
            i += 4;
        } else {
            i += 1;
        }
    }
    let mut out = Vec::new();
    if starts.len() < 2 {
        return out;
    }
    let last = *starts.last().expect("len >= 2");
    for w in starts.windows(2) {
        out.push(buf[w[0]..w[1]].to_vec());
    }
    buf.drain(..last);
    out
}

/// `args` with the primer's keyframe: frame 0 is the discarded primer,
/// frame 1 (the first real one) a forced keyframe. `args` ends with the
/// output (`-f <fmt> pipe:1`); the option goes before it.
pub(crate) fn primed_args(args: &[String]) -> Vec<String> {
    debug_assert!(args.len() >= 3 && args[args.len() - 1] == "pipe:1" && args[args.len() - 3] == "-f");
    let at = args.len().saturating_sub(3);
    let mut a = args[..at].to_vec();
    a.extend(["-force_key_frames".into(), "expr:eq(n,1)".into()]);
    a.extend_from_slice(&args[at..]);
    a
}

/// A warm spare: started, fed its primer on a helper thread.
struct Spare {
    spec: SpareSpec,
    proc: Proc,
    primer: mpsc::Receiver<std::io::Result<ChildStdin>>,
    /// The primer's outcome, once known.
    primed: Option<std::io::Result<ChildStdin>>,
    started: Instant,
}

impl Spare {
    fn start(spec: SpareSpec) -> Result<Self> {
        let mut proc = Proc::spawn(&primed_args(&spec.args), spec.framer, 1)?;
        let mut stdin = proc.stdin.take().ok_or_else(|| MediaError::tool("ffmpeg", "no stdin"))?;
        let (tx, primer) = mpsc::channel();
        let frame_len = spec.frame_len;
        std::thread::Builder::new().name("ffmpeg-spare-primer".into()).spawn(move || {
            // Blocks until ffmpeg reads it: done means started. Killing the
            // spare fails the write.
            let black = vec![0u8; frame_len];
            let _ = tx.send(stdin.write_all(&black).map(|()| stdin));
        })?;
        Ok(Self { spec, proc, primer, primed: None, started: Instant::now() })
    }

    /// Whether the primer went in (without waiting).
    fn is_primed(&mut self) -> bool {
        if self.primed.is_none() {
            self.primed = self.primer.try_recv().ok();
        }
        matches!(self.primed, Some(Ok(_)))
    }

    /// The primed process, waiting for its primer when it is still starting.
    fn into_proc(mut self) -> Result<(Proc, SpareKind)> {
        self.is_primed();
        let kind = if self.primed.is_some() { SpareKind::Warm } else { SpareKind::Warming };
        let Spare { mut proc, primer, primed, .. } = self;
        let primed = match primed {
            Some(r) => r,
            None => primer.recv().map_err(|_| MediaError::tool("ffmpeg", "spare primer thread died"))?,
        };
        let stdin = primed.map_err(|e| MediaError::tool("ffmpeg", format!("spare primer: {e}")))?;
        if !proc.alive() {
            return Err(MediaError::tool("ffmpeg", "spare exited"));
        }
        proc.stdin = Some(stdin);
        Ok((proc, kind))
    }
}

/// The process side of a pipe encoder: the current ffmpeg, processes
/// flushing after a restart, and the session's spare pool.
pub(crate) struct PipeProcs {
    spec: SpareSpec,
    cur: Option<Proc>,
    /// Old processes after EOF, oldest first; their frames go out before
    /// `cur`'s.
    draining: VecDeque<Proc>,
    pool: Option<SparePool>,
    /// The restart whose first frame is awaited.
    restart: Option<(Instant, SpareKind)>,
    /// Ask the pool for a spare once the current process has output its
    /// first frame (once per process start: two encoders of different
    /// profiles must not replace each other's spare on every frame).
    refill: bool,
    stats: RestartStats,
}

impl PipeProcs {
    /// Starts the first process: the pool's spare for this profile when it
    /// has one, else ffmpeg cold.
    pub(crate) fn new(spec: SpareSpec, pool: Option<SparePool>) -> Result<Self> {
        let pool = pool.filter(SparePool::enabled);
        let refill = pool.is_some();
        let mut me = Self {
            spec,
            cur: None,
            draining: VecDeque::new(),
            pool,
            restart: None,
            refill,
            stats: RestartStats::default(),
        };
        let t0 = Instant::now();
        let (proc, kind) = me.next_proc()?;
        if kind != SpareKind::Cold {
            tracing::info!(
                codec = me.spec.codec,
                spare = kind.as_str(),
                wait_ms = t0.elapsed().as_millis() as u64,
                "encoder started from the session's warm spare"
            );
        }
        me.cur = Some(proc);
        me.stats.start = Some(kind);
        Ok(me)
    }

    /// The pool's spare for this profile, else a cold start.
    fn next_proc(&mut self) -> Result<(Proc, SpareKind)> {
        if let Some(spare) = self.pool.as_ref().and_then(|p| p.take(&self.spec)) {
            let age = spare.started.elapsed();
            match spare.into_proc() {
                Ok(p) => return Ok(p),
                Err(e) => {
                    tracing::warn!(codec = self.spec.codec, error = %e, ?age, "encoder spare unusable; starting ffmpeg cold")
                }
            }
        }
        Ok((Proc::spawn(&self.spec.args, self.spec.framer, 0)?, SpareKind::Cold))
    }

    pub(crate) fn stats(&self) -> &RestartStats {
        &self.stats
    }

    /// Ids of the encoder's ffmpeg processes alive now (current, flushing).
    pub(crate) fn pids(&self) -> Vec<u32> {
        self.cur.iter().chain(self.draining.iter()).map(|p| p.child.id()).collect()
    }

    pub(crate) fn write(&mut self, data: &[u8]) -> Result<()> {
        self.cur.as_mut().ok_or_else(|| MediaError::Encode("encoder finished".into()))?.write(data)
    }

    /// Replace the current process: its next frame is a keyframe.
    pub(crate) fn restart(&mut self) -> Result<()> {
        if self.pool.is_some() && self.cur.as_ref().is_some_and(|c| c.first_out.get().is_none()) {
            // Its first output, still to come, is a keyframe.
            self.stats.skipped += 1;
            tracing::debug!(codec = self.spec.codec, "keyframe restart skipped: the first keyframe is still on its way");
            return Ok(());
        }
        let t0 = Instant::now();
        if let Some(mut old) = self.cur.take() {
            old.eof();
            self.draining.push_back(old);
        }
        let (proc, kind) = self.next_proc()?;
        self.cur = Some(proc);
        self.stats.count(kind);
        self.restart = Some((t0, kind));
        self.refill = self.pool.is_some();
        Ok(())
    }

    /// Frames ready now, in order: flushing processes first.
    pub(crate) fn ready(&mut self) -> Result<Vec<Vec<u8>>> {
        let mut out = Vec::new();
        while let Some(d) = self.draining.front_mut() {
            let (frames, done) = d.try_take()?;
            out.extend(frames);
            if done {
                let d = self.draining.pop_front().expect("front");
                d.reap(self.spec.codec)?;
            } else if d.eof_at.is_some_and(|t| t.elapsed() > DRAIN_TIMEOUT) {
                tracing::warn!(codec = self.spec.codec, "ffmpeg did not exit after EOF; killed");
                self.draining.pop_front();
            } else {
                return Ok(out);
            }
        }
        if let Some(c) = self.cur.as_mut() {
            out.extend(c.try_take()?.0);
        }
        self.after_output();
        Ok(out)
    }

    /// Records a finished restart and asks the pool for a spare of this
    /// profile once the current process has output its first frame.
    fn after_output(&mut self) {
        let Some(first) = self.cur.as_ref().and_then(|c| c.first_out.get().copied()) else { return };
        if std::mem::take(&mut self.refill) {
            if let Some(p) = &self.pool {
                p.prewarm(self.spec.clone());
            }
        }
        let Some((t0, kind)) = self.restart.take() else { return };
        let codec = self.spec.codec;
        let latency = first.saturating_duration_since(t0);
        self.stats.latencies.push((kind, latency));
        metrics::histogram!("fv_encoder_restart_duration_seconds", "codec" => codec, "spare" => kind.as_str())
            .record(latency.as_secs_f64());
        metrics::counter!("fv_encoder_restarts_total", "codec" => codec, "spare" => kind.as_str()).increment(1);
        tracing::info!(
            codec,
            spare = kind.as_str(),
            latency_ms = latency.as_millis() as u64,
            restarts = self.stats.restarts,
            "encoder keyframe restart: first frame out"
        );
    }

    /// EOF to every process; all remaining frames in order.
    pub(crate) fn finish(&mut self) -> Result<Vec<Vec<u8>>> {
        let mut out = Vec::new();
        while let Some(d) = self.draining.pop_front() {
            out.extend(d.close(self.spec.codec)?);
        }
        if let Some(c) = self.cur.take() {
            let first = c.first_out.clone();
            out.extend(c.close(self.spec.codec)?);
            if let (Some((t0, kind)), Some(t)) = (self.restart.take(), first.get()) {
                self.stats.latencies.push((kind, t.saturating_duration_since(t0)));
            }
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn primer_keyframe_goes_before_the_output() {
        let a: Vec<String> = ["-i", "pipe:0", "-c:v", "libvpx", "-f", "ivf", "pipe:1"].map(String::from).to_vec();
        assert_eq!(primed_args(&a).join(" "), "-i pipe:0 -c:v libvpx -force_key_frames expr:eq(n,1) -f ivf pipe:1");
    }

    #[test]
    fn reader_skips_the_primer_output() {
        let aud = |b: u8| vec![0, 0, 0, 1, 0x09, 0xf0, 0, 0, 1, b];
        let mut s = Vec::new();
        for b in [0x65, 0x41, 0x41] {
            s.extend(aud(b));
        }
        let (tx, rx) = mpsc::channel();
        let first = OnceLock::new();
        read_frames(&s[..], Framer::AnnexB, 1, &tx, &first);
        drop(tx);
        let got: Vec<Vec<u8>> = rx.iter().map(|r| r.unwrap()).collect();
        assert_eq!(got, vec![aud(0x41), aud(0x41)]);
        assert!(first.get().is_some());
    }

    #[test]
    fn a_disabled_pool_never_prewarms_or_waits() {
        let p = SparePool::new(false);
        p.prewarm_with(|| panic!("not called"));
        let spec = SpareSpec { codec: "libvpx", args: vec![], framer: Framer::Ivf, frame_len: 3 };
        p.prewarm(spec.clone());
        assert!(p.spare_state().is_none());
        assert!(p.take(&spec).is_none());
    }

    #[test]
    fn a_take_waits_for_a_prewarm_in_progress() {
        let p = SparePool::new(true);
        let (tx, rx) = mpsc::channel::<()>();
        p.prewarm_with(move || {
            let _ = rx.recv();
            None
        });
        let spec = SpareSpec { codec: "libvpx", args: vec![], framer: Framer::Ivf, frame_len: 3 };
        let t0 = Instant::now();
        let h = {
            let p = p.clone();
            let spec = spec.clone();
            std::thread::spawn(move || p.take(&spec).is_none())
        };
        std::thread::sleep(Duration::from_millis(100));
        tx.send(()).unwrap();
        assert!(h.join().unwrap());
        assert!(t0.elapsed() >= Duration::from_millis(100));
    }
}
