//! The script avatar (docs/serve/research-avatar-v2v.md P0-3): a photo and
//! a script in, speech and video streamed out window by window, over a
//! [`ClipSession`] of an image-to-video model with native audio (LTX-2.5).
//!
//! - **Plan** ([`plan`]): the script is cut at sentence ends (then clauses,
//!   then words) into windows of at most `window_s` of speech at the
//!   requested words per minute; each window lasts its speech plus a short
//!   breath, snapped up onto the model's frame grid. The take lasts the
//!   script's derived length (`words · 60 / wpm`) or the requested
//!   duration, clamped to 4..300 s: a longer duration is filled with idle
//!   windows (the person listens, silent), a shorter one cuts the script.
//! - **Speech**: the model speaks the window's lines itself (joint audio and
//!   video from the prompt, `… says: "<lines>"`), as Reactor's `ltx` model
//!   does ("no separate text-to-speech step"). The seed is the same for every
//!   window, which keeps the voice close. With a driving voice file
//!   ([`AvatarTake::voice`]) the windows are audio-to-video instead: each
//!   window's slice of the file is held clean as the audio conditioning and
//!   the output carries it unchanged.
//! - **Continuity**: window 0 starts from the photo (image-to-video, frame
//!   0); window `k` starts from window `k−1`'s last frame (the uncropped
//!   frame, written as a PNG), and its first frame, a copy of that anchor,
//!   is dropped at playout with its audio. Voice windows cut the file so
//!   that the dropped frame's audio is the anchor's: the file plays without
//!   a gap or a repeat. Window edges get a 20 ms audio crossfade.
//! - **Pipelining**: builds are sequential (each needs the previous last
//!   frame) and run while earlier windows play, at most `lookahead` built
//!   windows ahead. A window that is not ready when the previous one ends
//!   holds the last frame with silent audio (the clip pacer's underrun), and
//!   the wait is counted (`stalls`, `stalled_s`).
//! - **Output**: 3-frame lockstep [`MediaItem`] slices on a re-anchoring
//!   metronome, for [`spawn_clip_pacer`](super::pace::spawn_clip_pacer);
//!   frames are centre-cropped to the delivered canvas when the model
//!   generates a padded one (LTX two-stage: 640x352 is generated at
//!   640x384).
//!
//! The player never blocks the executor: builds are `Priority::Stream`
//! jobs, media preparation runs on the blocking pool.

use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use fastvideo_media::crossfade::apply_clip_fades;
use fastvideo_media::lockstep::{frame_sample_offset, prepare_clip_audio, EMIT_FRAMES, WIRE_RATE};
use fastvideo_protocol::{ApiError, FrameGrid, RgbFrame, SessionSpec, Task};
use serde::Serialize;
use tokio::sync::{mpsc, oneshot, watch};
use tokio::task::JoinHandle;
use uuid::Uuid;

use crate::backend::ClipOutput;
use crate::cancel::CancelToken;

use super::clip::{ClipBuild, ClipSession};
use super::pace::{MediaItem, MediaSlice, PlayOutcome};

/// Reactor `ltx`: `set_wpm` range and default.
pub const WPM_MIN: u32 = 80;
pub const WPM_MAX: u32 = 220;
pub const WPM_DEFAULT: u32 = 140;
/// Reactor `ltx`: a take lasts 4..300 s.
pub const TAKE_MIN_S: f64 = 4.0;
pub const TAKE_MAX_S: f64 = 300.0;
/// Reactor `ltx`: `set_script` limit.
pub const SCRIPT_MAX_CHARS: usize = 10_000;
/// Reactor `ltx`: `set_prompt` limit.
pub const SCENE_MAX_CHARS: usize = 800;
/// Default window length (seconds of speech per generation).
pub const DEFAULT_WINDOW_S: f64 = 10.0;
/// Silence after a window's lines, so a sentence is not cut at the edge.
pub const BREATH_S: f64 = 0.5;
/// The scene when `set_prompt` is empty (Reactor: "straight-to-camera").
pub const DEFAULT_SCENE: &str = "A single person speaks directly to the camera, head and shoulders in frame, \
steady camera, soft natural light, clear voice.";

/// Words in a script (whitespace-separated).
pub fn word_count(script: &str) -> usize {
    script.split_whitespace().count()
}

/// `words · 60 / wpm`, to 0.1 s.
pub fn derived_seconds(words: usize, wpm: u32) -> f64 {
    (words as f64 * 60.0 / f64::from(wpm.max(1)) * 10.0).round() / 10.0
}

/// The take's length: the requested duration (`> 0`), else the derived one,
/// clamped to [`TAKE_MIN_S`]..[`TAKE_MAX_S`].
pub fn effective_seconds(derived: f64, duration: f64) -> f64 {
    let d = if duration > 0.0 { duration } else { derived };
    d.clamp(TAKE_MIN_S, TAKE_MAX_S)
}

/// What a window says.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum WindowKind {
    /// Lines of the script, spoken by the model.
    Speech,
    /// A slice of the driving voice file (audio-to-video).
    Voice,
    /// Filler after the script: the person listens, silent.
    Idle,
}

/// One generation window.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct AvatarWindow {
    pub index: u32,
    pub kind: WindowKind,
    /// The script lines of this window (empty for idle).
    pub text: String,
    pub prompt: String,
    /// Generated frames (on the model grid).
    pub frames: u32,
    /// Leading frames dropped at playout (1 after window 0: the anchor copy).
    pub trim: u32,
    /// Take frame at which this window's first played frame lands.
    pub start_frame: u32,
    /// Voice windows: `(start_s, seconds)` of the driving file.
    pub audio: Option<(f64, f64)>,
}

impl AvatarWindow {
    /// Frames this window plays.
    pub fn played(&self) -> u32 {
        self.frames.saturating_sub(self.trim)
    }
}

/// A planned take.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct AvatarPlan {
    pub words: usize,
    pub derived_seconds: f64,
    pub effective_seconds: f64,
    pub fps: u32,
    /// Frames the take plays (`round(effective · fps)`); the last window is
    /// cut there.
    pub total_frames: u32,
    pub windows: Vec<AvatarWindow>,
}

/// Planner input.
#[derive(Clone, Debug, PartialEq)]
pub struct PlanInput<'a> {
    pub script: &'a str,
    /// Scene / delivery prompt (`set_prompt`); empty: [`DEFAULT_SCENE`].
    pub scene: &'a str,
    pub wpm: u32,
    /// Requested take length; `0`: derived from the script (or the voice).
    pub duration_s: f64,
    /// Longest window.
    pub window_s: f64,
    pub fps: u32,
    pub grid: FrameGrid,
    /// Length of the driving voice file, when the take is voice-driven.
    pub voice_s: Option<f64>,
}

/// Sentences: cut after `.`, `!`, `?` or `…` followed by whitespace, and at
/// line breaks.
pub fn sentences(script: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    let chars: Vec<char> = script.chars().collect();
    for (i, &c) in chars.iter().enumerate() {
        if c == '\n' {
            push_trimmed(&mut out, &mut cur);
            continue;
        }
        cur.push(c);
        let end = matches!(c, '.' | '!' | '?' | '…');
        let next_ws = chars.get(i + 1).is_none_or(|n| n.is_whitespace());
        if end && next_ws {
            push_trimmed(&mut out, &mut cur);
        }
    }
    push_trimmed(&mut out, &mut cur);
    out
}

fn push_trimmed(out: &mut Vec<String>, cur: &mut String) {
    let t = cur.split_whitespace().collect::<Vec<_>>().join(" ");
    if !t.is_empty() {
        out.push(t);
    }
    cur.clear();
}

/// Pieces of at most `cap` words: whole sentences packed greedily; a
/// sentence longer than `cap` is cut at clause marks (`,` `;` `:` `—`),
/// then between words.
pub fn pack(script: &str, cap: usize) -> Vec<String> {
    let cap = cap.max(1);
    let mut units: Vec<String> = Vec::new();
    for s in sentences(script) {
        if word_count(&s) <= cap {
            units.push(s);
            continue;
        }
        let mut clause = String::new();
        let mut clauses = Vec::new();
        for w in s.split_whitespace() {
            if !clause.is_empty() {
                clause.push(' ');
            }
            clause.push_str(w);
            if w.ends_with([',', ';', ':', '—']) {
                clauses.push(std::mem::take(&mut clause));
            }
        }
        if !clause.is_empty() {
            clauses.push(clause);
        }
        for c in clauses {
            let words: Vec<&str> = c.split_whitespace().collect();
            for piece in words.chunks(cap) {
                units.push(piece.join(" "));
            }
        }
    }
    let mut out: Vec<String> = Vec::new();
    let mut cur = String::new();
    for u in units {
        if !cur.is_empty() && word_count(&cur) + word_count(&u) > cap {
            out.push(std::mem::take(&mut cur));
        }
        if !cur.is_empty() {
            cur.push(' ');
        }
        cur.push_str(&u);
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    out
}

/// The prompt of a window: the scene, then what the person does.
pub fn window_prompt(scene: &str, kind: WindowKind, text: &str) -> String {
    let scene = scene.trim();
    let scene = if scene.is_empty() { DEFAULT_SCENE } else { scene };
    let scene = if scene.ends_with(['.', '!', '?']) { scene.to_owned() } else { format!("{scene}.") };
    match kind {
        WindowKind::Speech => format!("{scene} The person says: \"{}\"", text.replace('"', "'")),
        WindowKind::Voice if text.is_empty() => format!("{scene} The person is speaking."),
        WindowKind::Voice => format!("{scene} The person says: \"{}\"", text.replace('"', "'")),
        WindowKind::Idle => format!(
            "{scene} The person pauses and listens quietly, mouth closed, with small natural movements; no speech."
        ),
    }
}

/// Frames for `played` played frames plus `trim`, on the grid (clamped to
/// its range).
fn grid_frames(grid: &FrameGrid, played: u32, trim: u32) -> u32 {
    let want = played.max(1) + trim;
    grid.align_up(want).unwrap_or(if want > grid.max { grid.max } else { grid.min })
}

/// Plans a take (pure).
pub fn plan(p: &PlanInput<'_>) -> AvatarPlan {
    let fps = p.fps.max(1);
    let ffps = f64::from(fps);
    let words = word_count(p.script);
    let wpm = p.wpm.clamp(WPM_MIN, WPM_MAX);
    let derived = match p.voice_s {
        Some(v) => (v * 10.0).round() / 10.0,
        None => derived_seconds(words, wpm),
    };
    let effective = effective_seconds(derived, p.duration_s);
    let total_frames = (effective * ffps).round() as u32;
    let window_s = p.window_s.clamp(1.0, f64::from(p.grid.max) / ffps);
    let win_frames = ((window_s * ffps).round() as u32).max(1);
    let mut windows: Vec<AvatarWindow> = Vec::new();
    let mut cum = 0u32;
    let push = |windows: &mut Vec<AvatarWindow>,
                cum: &mut u32,
                kind: WindowKind,
                text: String,
                played: u32,
                audio: Option<(f64, f64)>| {
        let trim = u32::from(!windows.is_empty());
        let frames = grid_frames(&p.grid, played, trim);
        let w = AvatarWindow {
            index: windows.len() as u32,
            kind,
            prompt: window_prompt(p.scene, kind, &text),
            text,
            frames,
            trim,
            start_frame: *cum,
            audio,
        };
        *cum += w.played();
        windows.push(w);
    };
    match p.voice_s {
        None => {
            let cap = ((window_s * f64::from(wpm) / 60.0).floor() as usize).max(1);
            for chunk in pack(p.script, cap) {
                if cum >= total_frames {
                    break;
                }
                let speech = word_count(&chunk) as f64 * 60.0 / f64::from(wpm) + BREATH_S;
                let played = ((speech * ffps).round() as u32).min(win_frames + (BREATH_S * ffps) as u32);
                push(&mut windows, &mut cum, WindowKind::Speech, chunk, played, None);
            }
        }
        Some(voice) => {
            let audio_frames = ((voice.min(effective)) * ffps).round() as u32;
            let script_words: Vec<&str> = p.script.split_whitespace().collect();
            while cum < audio_frames {
                let played = (audio_frames - cum).min(win_frames);
                // The script's words in proportion to the window's share of
                // the voice (a transcript hint for the prompt, optional).
                let (a, b) = (cum as usize, (cum + played) as usize);
                let n = script_words.len();
                let (wa, wb) = (a * n / audio_frames as usize, b * n / audio_frames as usize);
                let text = script_words[wa..wb.min(n)].join(" ");
                let trim = u32::from(!windows.is_empty());
                let frames = grid_frames(&p.grid, played, trim);
                // The dropped anchor frame's audio is the previous window's
                // last frame: start one frame early.
                let start = (f64::from(cum) - f64::from(trim)) / ffps;
                push(&mut windows, &mut cum, WindowKind::Voice, text, played, Some((start.max(0.0), f64::from(frames) / ffps)));
            }
        }
    }
    while cum < total_frames {
        let played = (total_frames - cum).min(win_frames);
        push(&mut windows, &mut cum, WindowKind::Idle, String::new(), played, None);
    }
    AvatarPlan { words, derived_seconds: derived, effective_seconds: effective, fps, total_frames, windows }
}

/// Centre-crops an RGB24 frame to `(w, h)` (no-op when it already is, or is
/// smaller): [`RgbFrame::crop_center`].
pub fn crop_center(f: &RgbFrame, w: u32, h: u32) -> RgbFrame {
    f.crop_center(w, h)
}

/// Player settings.
#[derive(Clone, Debug, PartialEq)]
pub struct AvatarConfig {
    /// Audio crossfade at window edges.
    pub crossfade_ms: u16,
    /// Frames per emitted slice (3).
    pub emit_frames: u32,
    /// Built windows allowed to wait behind the playing one.
    pub lookahead: usize,
    /// Anchor PNGs and voice slices; `None`: `$TMPDIR/fv-sessions/<id>-avatar`.
    pub session_dir: Option<PathBuf>,
    /// Delivered canvas; frames are centre-cropped to it (`None`: as built).
    pub delivered: Option<(u32, u32)>,
    /// Playout speed (1.0 = real time). Tests only.
    pub speed: f64,
    pub event_depth: usize,
    pub media_depth: usize,
}

impl Default for AvatarConfig {
    fn default() -> Self {
        Self {
            crossfade_ms: 20,
            emit_frames: EMIT_FRAMES,
            lookahead: 2,
            session_dir: None,
            delivered: None,
            speed: 1.0,
            event_depth: 256,
            media_depth: 8,
        }
    }
}

/// One take to play.
#[derive(Clone, Debug, PartialEq)]
pub struct AvatarTake {
    pub plan: AvatarPlan,
    /// The portrait (window 0's first frame).
    pub image: PathBuf,
    pub seed: u64,
    /// Driving voice file (voice windows).
    pub voice: Option<PathBuf>,
}

/// Timing of one built window.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct WindowReport {
    pub index: u32,
    pub total_windows: u32,
    pub kind: WindowKind,
    pub frames: u32,
    /// Seconds this window plays.
    pub seconds: f64,
    /// Submit to frames ready (queue wait included).
    pub build_s: f64,
    /// `build_s / seconds` (below 1: faster than real time).
    pub rtf: f64,
}

/// What the player reports.
#[derive(Clone, Debug, PartialEq)]
pub enum AvatarEvent {
    Started { seconds: f64, total_windows: u32, width: u32, height: u32 },
    WindowBuilt(WindowReport),
    /// A window's first frame went out. `since_start_s`: from `start`;
    /// `stalled_s`: how long playout waited for it.
    WindowStarted { index: u32, total_windows: u32, seconds_sent: f64, since_start_s: f64, stalled_s: f64 },
    /// A window was streamed to its end.
    WindowProgress { index: u32, total_windows: u32, seconds_sent: f64, total_seconds: f64 },
    Paused { seconds_sent: f64 },
    Resumed { seconds_sent: f64 },
    Stopped { seconds_sent: f64 },
    Complete { seconds_sent: f64 },
    Failed { reason: String, seconds_sent: f64 },
}

/// The player's state (for `state_update`).
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct AvatarStatus {
    pub generating: bool,
    pub paused: bool,
    pub finished: bool,
    /// The playing window, else the next one to play.
    pub window_index: u32,
    pub total_windows: u32,
    pub windows_built: u32,
    pub seconds_sent: f64,
    pub total_seconds: f64,
    /// Windows that were not ready when the previous one ended.
    pub stalls: u32,
    pub stalled_s: f64,
    /// `start` to the first frame of window 0.
    pub first_frame_s: Option<f64>,
}

/// The player's outputs. Drain both.
#[derive(Debug)]
pub struct AvatarOutputs {
    pub events: mpsc::Receiver<AvatarEvent>,
    pub media: mpsc::Receiver<MediaItem>,
}

enum Msg {
    Start(Box<AvatarTake>, oneshot::Sender<Result<(), String>>),
    Pause(oneshot::Sender<Result<(), String>>),
    Resume(oneshot::Sender<Result<(), String>>),
    Stop(oneshot::Sender<Result<(), String>>),
    Audience(bool),
    Close(oneshot::Sender<()>),
}

/// Handle to a running avatar player. Dropping every handle stops it and
/// closes the session.
#[derive(Clone, Debug)]
pub struct AvatarPlayer {
    tx: mpsc::Sender<Msg>,
    state: watch::Receiver<AvatarStatus>,
    task: Arc<std::sync::Mutex<Option<JoinHandle<()>>>>,
}

fn gone() -> String {
    "the avatar session has ended".to_owned()
}

impl AvatarPlayer {
    /// Starts the player over `session` (needs a tokio runtime). The model
    /// must build image-to-video clips.
    pub fn start(session: ClipSession, cfg: AvatarConfig) -> Result<(Self, AvatarOutputs), ApiError> {
        if !session.caps().supports(Task::I2V) {
            return Err(ApiError::invalid_param(
                "model",
                format!("`{}` cannot build image-to-video clips: no script avatar", session.caps().id),
            ));
        }
        let spec = session.spec().clone();
        let channels = spec.tracks.audio.as_ref().map(|a| a.channels);
        if channels.is_some() {
            spec.tracks.samples_per_frame()?;
        }
        let (ev_tx, ev_rx) = mpsc::channel(cfg.event_depth.max(1));
        let (media_tx, media_rx) = mpsc::channel(cfg.media_depth.max(1));
        let (msg_tx, msg_rx) = mpsc::channel(32);
        let (build_tx, build_rx) = mpsc::channel(4);
        let (session_dir, own_dir) = match &cfg.session_dir {
            Some(d) => (d.clone(), false),
            None => (std::env::temp_dir().join("fv-sessions").join(format!("{}-avatar", session.id())), true),
        };
        let mut p = Player {
            fps: spec.fps.max(1),
            channels,
            spec,
            session,
            cfg,
            take: None,
            events: ev_tx,
            media: media_tx,
            build_tx,
            session_dir,
            own_dir,
            audience: false,
            status: AvatarStatus::default(),
        };
        let (state_tx, state_rx) = watch::channel(AvatarStatus::default());
        let task = tokio::spawn(async move { p.run(msg_rx, build_rx, state_tx).await });
        Ok((
            Self { tx: msg_tx, state: state_rx, task: Arc::new(std::sync::Mutex::new(Some(task))) },
            AvatarOutputs { events: ev_rx, media: media_rx },
        ))
    }

    async fn ask(&self, f: impl FnOnce(oneshot::Sender<Result<(), String>>) -> Msg) -> Result<(), String> {
        let (tx, rx) = oneshot::channel();
        self.tx.send(f(tx)).await.map_err(|_| gone())?;
        rx.await.map_err(|_| gone())?
    }

    /// Starts a take (refused while one is generating or playing).
    pub async fn start_take(&self, take: AvatarTake) -> Result<(), String> {
        self.ask(|tx| Msg::Start(Box::new(take), tx)).await
    }

    /// Holds playout (last frame, silent audio) and dispatches no new build.
    pub async fn pause(&self) -> Result<(), String> {
        self.ask(Msg::Pause).await
    }

    pub async fn resume(&self) -> Result<(), String> {
        self.ask(Msg::Resume).await
    }

    /// Ends the take: the build in flight is cancelled, nothing more plays.
    pub async fn stop(&self) -> Result<(), String> {
        self.ask(Msg::Stop).await
    }

    /// Without an audience nothing is built and playout waits (like a pause).
    pub async fn set_audience(&self, present: bool) {
        let _ = self.tx.send(Msg::Audience(present)).await;
    }

    pub fn status(&self) -> AvatarStatus {
        self.state.borrow().clone()
    }

    pub fn watch_status(&self) -> watch::Receiver<AvatarStatus> {
        self.state.clone()
    }

    pub fn is_closed(&self) -> bool {
        self.tx.is_closed()
    }

    /// Stops the player and closes the session (frees the executor).
    pub async fn close(&self) {
        let (tx, rx) = oneshot::channel();
        if self.tx.send(Msg::Close(tx)).await.is_ok() {
            let _ = rx.await;
        }
        let t = self.task.lock().ok().and_then(|mut g| g.take());
        if let Some(t) = t {
            let _ = t.await;
        }
    }
}

struct Built {
    index: u32,
    frames: Vec<RgbFrame>,
    /// Wire audio (faded), exactly the played frames' samples.
    audio: Option<Vec<f32>>,
}

struct BuildDone {
    take: Uuid,
    index: u32,
    result: Result<(Built, PathBuf, f64), ApiError>,
}

struct Playing {
    built: Built,
    next: u32,
    clock_start: Option<Instant>,
}

struct Take {
    id: Uuid,
    plan: AvatarPlan,
    image: PathBuf,
    seed: u64,
    voice: Option<PathBuf>,
    started: Instant,
    next_build: u32,
    building: Option<CancelToken>,
    anchor: Option<PathBuf>,
    ready: VecDeque<Built>,
    playing: Option<Playing>,
    next_play: u32,
    frames_sent: u32,
    paused: bool,
    /// When playout started waiting for the next window.
    waiting_since: Option<Instant>,
    done: bool,
}

struct Player {
    session: ClipSession,
    spec: SessionSpec,
    cfg: AvatarConfig,
    fps: u32,
    channels: Option<u8>,
    take: Option<Take>,
    events: mpsc::Sender<AvatarEvent>,
    media: mpsc::Sender<MediaItem>,
    build_tx: mpsc::Sender<BuildDone>,
    session_dir: PathBuf,
    own_dir: bool,
    audience: bool,
    status: AvatarStatus,
}

fn round2(x: f64) -> f64 {
    (x * 100.0).round() / 100.0
}

async fn sleep_until(d: Option<tokio::time::Instant>) {
    match d {
        Some(d) => tokio::time::sleep_until(d).await,
        None => std::future::pending().await,
    }
}

/// Cuts `[start, start + len)` of `src` into a 48 kHz stereo WAV, padded
/// with silence past the file's end.
pub fn cut_voice(src: &Path, start: f64, len: f64, dst: &Path) -> Result<(), ApiError> {
    if let Some(d) = dst.parent() {
        std::fs::create_dir_all(d).map_err(|e| ApiError::internal(format!("creating {}: {e}", d.display())))?;
    }
    let out = std::process::Command::new("ffmpeg")
        .args(["-v", "error", "-nostdin", "-y", "-ss", &format!("{start:.6}"), "-i"])
        .arg(src)
        .args(["-vn", "-ac", "2", "-ar", "48000", "-af", "apad", "-t", &format!("{len:.6}"), "-c:a", "pcm_s16le"])
        .arg(dst)
        .output()
        .map_err(|e| ApiError::internal(format!("ffmpeg not available: {e}")))?;
    if !out.status.success() {
        return Err(ApiError::invalid_param(
            "voice_audio",
            format!("cutting the voice audio failed: {}", String::from_utf8_lossy(&out.stderr).trim()),
        ));
    }
    Ok(())
}

impl Player {
    fn seconds_sent(&self) -> f64 {
        self.take.as_ref().map_or(0.0, |t| round2(f64::from(t.frames_sent) / f64::from(self.fps)))
    }

    async fn send(&self, ev: AvatarEvent) {
        let _ = self.events.send(ev).await;
    }

    async fn media(&self, item: MediaItem) {
        let _ = self.media.send(item).await;
    }

    fn snapshot(&self) -> AvatarStatus {
        let mut s = self.status.clone();
        match &self.take {
            Some(t) => {
                s.generating = !t.done;
                s.paused = t.paused;
                s.total_windows = t.plan.windows.len() as u32;
                s.window_index = t.playing.as_ref().map_or(t.next_play, |p| p.built.index);
                s.seconds_sent = round2(f64::from(t.frames_sent) / f64::from(self.fps));
                s.total_seconds = t.plan.effective_seconds;
            }
            None => {
                s.generating = false;
                s.paused = false;
            }
        }
        s
    }

    async fn run(
        &mut self,
        mut msgs: mpsc::Receiver<Msg>,
        mut builds: mpsc::Receiver<BuildDone>,
        state: watch::Sender<AvatarStatus>,
    ) {
        let mut closer = None;
        loop {
            self.pump_build().await;
            self.advance().await;
            state.send_if_modified(|s| {
                let now = self.snapshot();
                if *s != now {
                    *s = now;
                    true
                } else {
                    false
                }
            });
            let deadline = self.slice_deadline();
            tokio::select! {
                biased;
                m = msgs.recv() => match m {
                    Some(Msg::Start(t, r)) => { let x = self.start(*t).await; let _ = r.send(x); }
                    Some(Msg::Pause(r)) => { let x = self.pause(true).await; let _ = r.send(x); }
                    Some(Msg::Resume(r)) => { let x = self.pause(false).await; let _ = r.send(x); }
                    Some(Msg::Stop(r)) => { let x = self.stop().await; let _ = r.send(x); }
                    Some(Msg::Audience(b)) => self.audience = b,
                    Some(Msg::Close(tx)) => { tracing::info!(session = %self.session.id(), "avatar player: close"); closer = Some(tx); break; }
                    None => break,
                },
                Some(done) = builds.recv() => self.apply_build(done).await,
                _ = sleep_until(deadline), if deadline.is_some() => self.emit_slice().await,
            }
        }
        if let Some(t) = self.take.take() {
            if let Some(c) = t.building {
                c.cancel();
            }
            if t.playing.is_some() {
                self.media(MediaItem::Clear).await;
            }
        }
        if self.own_dir {
            let _ = std::fs::remove_dir_all(&self.session_dir);
        }
        self.session.end();
        tracing::info!(session = %self.session.id(), "avatar player: session released");
        if let Some(tx) = closer {
            let _ = tx.send(());
        }
    }

    async fn start(&mut self, take: AvatarTake) -> Result<(), String> {
        if self.take.as_ref().is_some_and(|t| !t.done) {
            return Err("a take is already running; stop it first".into());
        }
        if take.plan.windows.is_empty() {
            return Err("the take has no windows".into());
        }
        if !take.image.is_file() {
            return Err("the avatar image is missing".into());
        }
        if take.voice.is_some() && !self.session.caps().supports(Task::A2V) {
            return Err(format!("`{}` cannot take a driving voice (no audio-to-video)", self.session.caps().id));
        }
        let (w, h) = self.cfg.delivered.unwrap_or(self.spec.canvas);
        let n = take.plan.windows.len() as u32;
        let seconds = take.plan.effective_seconds;
        self.status = AvatarStatus::default();
        self.take = Some(Take {
            id: Uuid::new_v4(),
            plan: take.plan,
            image: take.image,
            seed: take.seed,
            voice: take.voice,
            started: Instant::now(),
            next_build: 0,
            building: None,
            anchor: None,
            ready: VecDeque::new(),
            playing: None,
            next_play: 0,
            frames_sent: 0,
            paused: false,
            waiting_since: Some(Instant::now()),
            done: false,
        });
        tracing::info!(session = %self.session.id(), windows = n, seconds, "avatar take started");
        self.send(AvatarEvent::Started { seconds, total_windows: n, width: w, height: h }).await;
        Ok(())
    }

    async fn pause(&mut self, paused: bool) -> Result<(), String> {
        let sent = self.seconds_sent();
        let Some(t) = self.take.as_mut().filter(|t| !t.done) else {
            return Err(if paused { "nothing is generating" } else { "nothing is paused" }.into());
        };
        if t.paused == paused {
            return Err(if paused { "already paused" } else { "not paused" }.into());
        }
        t.paused = paused;
        if paused {
            if let Some(p) = t.playing.as_mut() {
                p.clock_start = None;
            }
            self.send(AvatarEvent::Paused { seconds_sent: sent }).await;
        } else {
            self.send(AvatarEvent::Resumed { seconds_sent: sent }).await;
        }
        Ok(())
    }

    async fn stop(&mut self) -> Result<(), String> {
        let sent = self.seconds_sent();
        let Some(t) = self.take.as_mut().filter(|t| !t.done) else {
            return Err("nothing is generating".into());
        };
        if let Some(c) = t.building.take() {
            c.cancel();
        }
        let was_playing = t.playing.take().is_some();
        t.ready.clear();
        t.done = true;
        if was_playing {
            self.media(MediaItem::Clear).await;
        }
        self.send(AvatarEvent::Stopped { seconds_sent: sent }).await;
        Ok(())
    }

    /// Dispatches the next build when allowed.
    async fn pump_build(&mut self) {
        let lookahead = self.cfg.lookahead.max(1);
        let Some(t) = self.take.as_mut() else { return };
        if t.done || t.paused || !self.audience || t.building.is_some() || t.ready.len() >= lookahead {
            return;
        }
        if t.next_build as usize >= t.plan.windows.len() {
            return;
        }
        let w = t.plan.windows[t.next_build as usize].clone();
        let first_frame = if w.index == 0 { t.image.clone() } else {
            match &t.anchor {
                Some(a) => a.clone(),
                None => return,
            }
        };
        let session_id = self.session.id();
        let voice_src = match (&t.voice, w.audio) {
            (Some(v), Some(seg)) => Some((v.clone(), seg)),
            _ => None,
        };
        let voice_path = self.session_dir.join(format!("voice-{}-{}.wav", t.id, w.index));
        let anchor_path = self.session_dir.join(format!("anchor-{}-{}.png", t.id, w.index));
        // Cut the voice slice synchronously (small, ffmpeg, milliseconds).
        let audio_drive = match voice_src {
            Some((src, (start, len))) => match cut_voice(&src, start, len, &voice_path) {
                Ok(()) => Some(voice_path),
                Err(e) => {
                    let tx = self.build_tx.clone();
                    let (take, index) = (t.id, w.index);
                    t.building = Some(CancelToken::new());
                    tokio::spawn(async move {
                        let _ = tx.send(BuildDone { take, index, result: Err(e) }).await;
                    });
                    return;
                }
            },
            None => None,
        };
        let b = ClipBuild {
            prompt: w.prompt.clone(),
            negative_prompt: None,
            seed: Some(t.seed),
            seconds: None,
            frames: Some(w.frames),
            canvas: Some(self.spec.canvas),
            first_frame: Some(first_frame),
            last_frame: None,
            audio_drive,
        };
        let job = match self.session.resolve(&b) {
            Ok(_) => b,
            Err(e) => {
                let tx = self.build_tx.clone();
                let (take, index) = (t.id, w.index);
                t.building = Some(CancelToken::new());
                tokio::spawn(async move {
                    let _ = tx.send(BuildDone { take, index, result: Err(e) }).await;
                });
                return;
            }
        };
        let handle = match self.session.build(job).await {
            Ok(h) => h,
            Err(e) => {
                let t = self.take.as_mut().expect("take");
                let tx = self.build_tx.clone();
                let (take, index) = (t.id, w.index);
                t.building = Some(CancelToken::new());
                tokio::spawn(async move {
                    let _ = tx.send(BuildDone { take, index, result: Err(e) }).await;
                });
                return;
            }
        };
        let t = self.take.as_mut().expect("take");
        t.building = Some(handle.cancel.clone());
        t.next_build += 1;
        let (take, index) = (t.id, w.index);
        let last = w.index as usize + 1 == t.plan.windows.len();
        let tx = self.build_tx.clone();
        let fps = self.fps;
        let channels = self.channels;
        let delivered = self.cfg.delivered;
        let fade = self.cfg.crossfade_ms;
        tracing::info!(session = %session_id, window = index, frames = w.frames, "avatar window build submitted");
        tokio::spawn(async move {
            let t0 = Instant::now();
            let result = async {
                let out = handle.wait().await?;
                let build_s = t0.elapsed().as_secs_f64();
                let built = tokio::task::spawn_blocking(move || {
                    prepare_window(out, &w, fps, channels, delivered, fade, last, &anchor_path).map(|b| (b, anchor_path))
                })
                .await
                .map_err(|e| ApiError::internal(format!("preparing the window: {e}")))??;
                Ok::<_, ApiError>((built.0, built.1, build_s))
            }
            .await;
            let _ = tx.send(BuildDone { take, index, result }).await;
        });
    }

    async fn apply_build(&mut self, done: BuildDone) {
        let sent = self.seconds_sent();
        let fps = f64::from(self.fps);
        let Some(t) = self.take.as_mut() else { return };
        if t.id != done.take || t.done {
            return;
        }
        t.building = None;
        match done.result {
            Err(e) => {
                tracing::warn!(session = %self.session.id(), index = done.index, error = %e, "avatar window failed");
                if let Some(p) = t.playing.take() {
                    drop(p);
                    self.media(MediaItem::Clear).await;
                }
                let t = self.take.as_mut().expect("take");
                t.done = true;
                t.ready.clear();
                self.send(AvatarEvent::Failed { reason: e.message, seconds_sent: sent }).await;
            }
            Ok((built, anchor, build_s)) => {
                let w = &t.plan.windows[built.index as usize];
                let seconds = f64::from(w.played()) / fps;
                let report = WindowReport {
                    index: built.index,
                    total_windows: t.plan.windows.len() as u32,
                    kind: w.kind,
                    frames: w.frames,
                    seconds: round2(seconds),
                    build_s: round2(build_s),
                    rtf: round2(build_s / seconds.max(1e-6)),
                };
                if let Some(old) = t.anchor.replace(anchor) {
                    let _ = std::fs::remove_file(old);
                }
                t.ready.push_back(built);
                self.status.windows_built += 1;
                tracing::info!(
                    session = %self.session.id(),
                    window = report.index,
                    build_s = report.build_s,
                    rtf = report.rtf,
                    "avatar window built"
                );
                self.send(AvatarEvent::WindowBuilt(report)).await;
            }
        }
    }

    /// Starts the next ready window when nothing plays.
    async fn advance(&mut self) {
        let fps = f64::from(self.fps);
        let Some(t) = self.take.as_mut() else { return };
        if t.done || t.paused || !self.audience || t.playing.is_some() {
            return;
        }
        let Some(front) = t.ready.front() else { return };
        if front.index != t.next_play {
            return;
        }
        let built = t.ready.pop_front().expect("front");
        let index = built.index;
        let stalled = t.waiting_since.take().map_or(0.0, |s| s.elapsed().as_secs_f64());
        let since_start = t.started.elapsed().as_secs_f64();
        let total = t.plan.windows.len() as u32;
        let sent = round2(f64::from(t.frames_sent) / fps);
        t.playing = Some(Playing { built, next: 0, clock_start: None });
        if index == 0 {
            self.status.first_frame_s = Some(round2(since_start));
        } else if stalled > 0.05 {
            self.status.stalls += 1;
            self.status.stalled_s = round2(self.status.stalled_s + stalled);
        }
        self.send(AvatarEvent::WindowStarted {
            index,
            total_windows: total,
            seconds_sent: sent,
            since_start_s: round2(since_start),
            stalled_s: round2(stalled),
        })
        .await;
    }

    fn slice_deadline(&mut self) -> Option<tokio::time::Instant> {
        let fps = f64::from(self.fps) * self.cfg.speed.max(1e-3);
        let t = self.take.as_mut()?;
        if t.paused || !self.audience {
            return None;
        }
        let p = t.playing.as_mut()?;
        let now = Instant::now();
        let content = Duration::from_secs_f64(f64::from(p.next) / fps);
        let start = *p.clock_start.get_or_insert(now);
        let due = (start + content).max(now);
        p.clock_start = Some(due - content);
        Some(tokio::time::Instant::from_std(due))
    }

    async fn emit_slice(&mut self) {
        let emit = self.cfg.emit_frames.max(1);
        let fps = self.fps;
        let ch = self.channels.unwrap_or(1) as usize;
        let Some(t) = self.take.as_mut() else { return };
        let left = t.plan.total_frames.saturating_sub(t.frames_sent);
        let Some(p) = t.playing.as_mut() else { return };
        let total = (p.built.frames.len() as u32).min(p.next + left);
        let lo = p.next;
        let hi = (lo + emit).min(total);
        let frames: Vec<RgbFrame> = p.built.frames[lo as usize..hi as usize].to_vec();
        let audio = p.built.audio.as_ref().map(|a| {
            let s0 = frame_sample_offset(u64::from(lo), fps, WIRE_RATE) as usize * ch;
            let s1 = frame_sample_offset(u64::from(hi), fps, WIRE_RATE) as usize * ch;
            a[s0.min(a.len())..s1.min(a.len())].to_vec()
        });
        p.next = hi;
        t.frames_sent += hi - lo;
        let window_done = hi >= total;
        let index = p.built.index;
        let item = MediaItem::Slice(MediaSlice { clip_id: t.id, first_frame: lo, frames, audio });
        let _ = self.media.send(item).await;
        if window_done {
            self.finish_window(index).await;
        }
    }

    async fn finish_window(&mut self, index: u32) {
        let fps = f64::from(self.fps);
        let Some(t) = self.take.as_mut() else { return };
        t.playing = None;
        t.next_play = index + 1;
        let total_windows = t.plan.windows.len() as u32;
        let sent = round2(f64::from(t.frames_sent) / fps);
        let total_seconds = t.plan.effective_seconds;
        let complete = t.frames_sent >= t.plan.total_frames || t.next_play >= total_windows;
        if complete {
            t.done = true;
            if let Some(c) = t.building.take() {
                c.cancel();
            }
            t.ready.clear();
        } else {
            t.waiting_since = Some(Instant::now());
        }
        let id = t.id;
        self.send(AvatarEvent::WindowProgress { index, total_windows, seconds_sent: sent, total_seconds }).await;
        if complete {
            self.media(MediaItem::ClipEnd { clip_id: id, outcome: PlayOutcome::Finished, armed: false }).await;
            self.status.finished = true;
            tracing::info!(session = %self.session.id(), seconds = sent, "avatar take complete");
            self.send(AvatarEvent::Complete { seconds_sent: sent }).await;
        }
    }
}

/// Executor output → a playable window (blocking pool): the anchor PNG
/// (the uncropped last frame), frames cropped to the delivered canvas with
/// the leading anchor copy dropped, and 48 kHz wire audio of exactly the
/// played frames, faded at the edges.
#[allow(clippy::too_many_arguments)]
fn prepare_window(
    out: ClipOutput,
    w: &AvatarWindow,
    fps: u32,
    channels: Option<u8>,
    delivered: Option<(u32, u32)>,
    fade_ms: u16,
    last: bool,
    anchor: &Path,
) -> Result<Built, ApiError> {
    let frames = out.frames.unwrap_or_default();
    let Some(tail) = frames.last() else {
        return Err(ApiError::engine_failed("the window build returned no frames"));
    };
    super::player::write_png(tail, anchor)?;
    let n = frames.len() as u32;
    let trim = w.trim.min(n.saturating_sub(1));
    let audio = match channels {
        Some(ch) => {
            let pcm = prepare_clip_audio(out.audio.as_ref(), n, fps, WIRE_RATE, ch)
                .map_err(|e| ApiError::internal(format!("window audio: {e}")))?;
            let s0 = frame_sample_offset(u64::from(trim), fps, WIRE_RATE) as usize * ch as usize;
            let kept = fastvideo_protocol::Pcm::new(WIRE_RATE, ch, pcm.samples[s0.min(pcm.samples.len())..].to_vec());
            let faded = if fade_ms > 0 { apply_clip_fades(&kept, fade_ms, w.index > 0, !last) } else { kept };
            Some(faded.samples.to_vec())
        }
        None => None,
    };
    let frames: Vec<RgbFrame> = frames
        .into_iter()
        .skip(trim as usize)
        .map(|f| match delivered {
            Some((dw, dh)) => crop_center(&f, dw, dh),
            None => f,
        })
        .collect();
    Ok(Built { index: w.index, frames, audio })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn grid() -> FrameGrid {
        FrameGrid::new(8, 1, 9, 481, 121)
    }

    fn input<'a>(script: &'a str, duration_s: f64) -> PlanInput<'a> {
        PlanInput { script, scene: "", wpm: 140, duration_s, window_s: 10.0, fps: 24, grid: grid(), voice_s: None }
    }

    #[test]
    fn sentences_and_packing() {
        let s = sentences("Hello there. How are you?\nFine, thanks! Version 2.5 is out");
        assert_eq!(s, ["Hello there.", "How are you?", "Fine, thanks!", "Version 2.5 is out"]);
        assert_eq!(pack("One two. Three four five. Six.", 3), ["One two.", "Three four five.", "Six."]);
        assert_eq!(pack("One two. Three.", 3), ["One two. Three."]);
        // A long sentence: clauses, then words.
        assert_eq!(pack("a b c d, e f g h i j", 4), ["a b c d,", "e f g h", "i j"]);
    }

    #[test]
    fn durations() {
        assert_eq!(word_count("  a b\tc \n d "), 4);
        assert_eq!(derived_seconds(140, 140), 60.0);
        assert_eq!(derived_seconds(7, 140), 3.0);
        assert_eq!(effective_seconds(3.0, 0.0), TAKE_MIN_S);
        assert_eq!(effective_seconds(30.0, 0.0), 30.0);
        assert_eq!(effective_seconds(30.0, 12.0), 12.0);
        assert_eq!(effective_seconds(900.0, 0.0), TAKE_MAX_S);
    }

    #[test]
    fn plan_covers_the_take_on_the_grid() {
        let script = "Welcome to the show. Today we talk about streaming video models, and how they keep a voice \
            steady across windows. Each window is ten seconds or less. The next one starts from the last frame. \
            That is all for now, thank you for watching.";
        let p = plan(&input(script, 0.0));
        assert_eq!(p.words, word_count(script));
        assert!(p.windows.len() >= 2, "{p:?}");
        assert_eq!(p.total_frames, (p.effective_seconds * 24.0).round() as u32);
        let mut at = 0;
        for (i, w) in p.windows.iter().enumerate() {
            assert!(grid().contains(w.frames), "{w:?}");
            assert_eq!(w.trim, u32::from(i > 0));
            assert_eq!(w.start_frame, at);
            assert!(w.played() as f64 / 24.0 <= 10.0 + BREATH_S + 0.4, "{w:?}");
            at += w.played();
        }
        assert!(at >= p.total_frames);
        let spoken: Vec<&str> = p.windows.iter().filter(|w| w.kind == WindowKind::Speech).map(|w| w.text.as_str()).collect();
        assert_eq!(spoken.join(" ").split_whitespace().count(), p.words);
        assert!(p.windows[0].prompt.contains("The person says: \"Welcome to the show."), "{}", p.windows[0].prompt);
    }

    #[test]
    fn longer_duration_adds_idle_windows_and_shorter_cuts() {
        let p = plan(&input("Hi there, nice to meet you.", 25.0));
        assert_eq!(p.effective_seconds, 25.0);
        assert_eq!(p.windows[0].kind, WindowKind::Speech);
        assert!(p.windows[1..].iter().all(|w| w.kind == WindowKind::Idle));
        assert!(p.windows.iter().map(AvatarWindow::played).sum::<u32>() >= 600);
        let long = "word ".repeat(200);
        let p = plan(&input(&long, 8.0));
        assert_eq!(p.total_frames, 192);
        assert!(p.windows.iter().map(AvatarWindow::played).sum::<u32>() >= 192);
        assert!(p.windows.len() <= 2);
    }

    #[test]
    fn voice_windows_cut_the_file_contiguously() {
        let mut i = input("one two three four five six seven eight", 0.0);
        i.voice_s = Some(23.0);
        let p = plan(&i);
        assert_eq!(p.effective_seconds, 23.0);
        let voice: Vec<&AvatarWindow> = p.windows.iter().filter(|w| w.kind == WindowKind::Voice).collect();
        assert_eq!(voice.len(), 3);
        for w in &voice {
            let (start, len) = w.audio.unwrap();
            // The first played frame's audio is at the window's take position.
            let played_at = start + f64::from(w.trim) / 24.0;
            assert!((played_at - f64::from(w.start_frame) / 24.0).abs() < 1e-9, "{w:?}");
            assert!((len - f64::from(w.frames) / 24.0).abs() < 1e-9);
        }
        assert!(p.windows[0].prompt.contains("one two"));
    }

    #[test]
    fn prompts() {
        assert!(window_prompt("", WindowKind::Idle, "").starts_with(DEFAULT_SCENE));
        assert_eq!(window_prompt("A chef in a kitchen", WindowKind::Speech, "Say \"hi\""), "A chef in a kitchen. The person says: \"Say 'hi'\"");
    }

    #[test]
    fn crop_takes_the_centre() {
        let mut data = Vec::new();
        for y in 0..4u8 {
            for x in 0..4u8 {
                data.extend_from_slice(&[x, y, 0]);
            }
        }
        let f = RgbFrame { width: 4, height: 4, data: data.into(), index: 7 };
        let c = crop_center(&f, 2, 2);
        assert_eq!((c.width, c.height, c.index), (2, 2, 7));
        assert_eq!(&c.data[..], &[1, 1, 0, 2, 1, 0, 1, 2, 0, 2, 2, 0]);
        assert_eq!(crop_center(&f, 4, 4), f);
    }
}
