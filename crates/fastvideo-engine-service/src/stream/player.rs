//! `ClipPlayer`: the fast-h3 queue-and-playout contract over a
//! [`ClipSession`] (design §5.5, WP-12).
//!
//! A port of infinite-livestream's `fast-h3/fasth3.py` semantics onto the
//! engine service:
//!
//! - **Queues.** `enqueue` adds to the generation queue (cap 20); builds
//!   consume it front first, one at a time, as `Priority::Stream` jobs on the
//!   session's executor; a finished build crosses into the playout queue
//!   (cap 10) with `clip_generated`. A build is submitted only while the
//!   playout queue has room (the submit-time reservation) and only while an
//!   audience is present ([`ClipPlayer::set_audience`]). A build whose entry
//!   was popped (or reset away) is cancelled and its result discarded.
//! - **Playout.** Nothing plays until `play`, or autoplay (a standing
//!   `play`). The playing clip is in neither queue. It is emitted as
//!   3-frame lockstep slices ([`MediaItem::Slice`]: frames plus exactly
//!   `round(hi·spf) − round(lo·spf)` samples of 48 kHz audio) on a
//!   re-anchoring metronome that never bursts to catch up.
//! - **Refusals are broadcast, never raised**: a refused command emits
//!   `command_error{command,reason}` and its reply is bodyless (`None`).
//!   Broadcasts go out on [`ClipOutputs::events`] **before** the reply
//!   returns, in fast-h3's order.
//! - **Continuity** ([`Continuity`] in the [`SessionSpec`]): `HardCut`
//!   (fast-h3 parity), `Crossfade{ms}` (raised-cosine edges, sample count
//!   unchanged) and `AnchorLastFrame{crossfade_ms}` (each build without its
//!   own first frame starts from the previous build's last frame, written as
//!   a PNG to the session directory; builds are sequential anyway).
//!
//! The player is a tokio task; it never blocks the executor. Its outputs are
//! bounded: events ([`ClipPlayerConfig::event_depth`], consumers must drain
//! them) and media ([`ClipPlayerConfig::media_depth`] slices; the emitter
//! waits on a full channel). Feed the media to
//! [`spawn_clip_pacer`](super::pace::spawn_clip_pacer) for per-tick A/V.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use fastvideo_media::crossfade::apply_clip_fades;
use fastvideo_media::lockstep::{frame_sample_offset, prepare_clip_audio, EMIT_FRAMES, WIRE_RATE};
use fastvideo_protocol::{
    canvas_for_aspect, ApiError, Continuity, JobMetrics, ModelCaps, RgbFrame, SessionSpec,
    StreamCaps, Task,
};
use serde::{Deserialize, Serialize};
use tokio::sync::{mpsc, oneshot, watch};
use tokio::task::JoinHandle;
use uuid::Uuid;

use crate::backend::{ClipOutput, SessionId};
use crate::cancel::CancelToken;

use super::clip::{ClipBuild, ClipSession};
use super::pace::{MediaItem, MediaSlice, PlayOutcome};
use super::queue::{
    BuiltClip, ClipEntry, ClipInfo, ClipQueue, QueueName, GENERATION_CAPACITY, PLAYOUT_CAPACITY,
};
use super::rules::valid_commands;

/// `enqueue.prompt` limit (fast-h3 `MAX_PROMPT_CHARS`).
pub const MAX_PROMPT_CHARS: usize = 800;
/// `enqueue.metadata` limit (fast-h3 `MAX_METADATA_CHARS`).
pub const MAX_METADATA_CHARS: usize = 2000;
/// The aspect ratios fast-h3 offers for `set_canvas`.
pub const ASPECT_CHOICES: [&str; 4] = ["16:9", "1:1", "9:16", "4:3"];

/// A clip-session command (design §5.5; fast-h3 verbatim plus the director's
/// `Chunk`).
///
/// **Deviation from design §5.5:** clip ids are the wire strings (blank or
/// malformed ids are refused inside with fast-h3's reasons), `SetCanvas`
/// takes the aspect label (`"16:9"`), and `Chunk` gains `first_image`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", content = "data", rename_all = "snake_case")]
pub enum ClipCommand {
    Enqueue {
        prompt: String,
        #[serde(default)]
        metadata: String,
        #[serde(default)]
        seed: Option<u64>,
        #[serde(default)]
        seconds: Option<f64>,
        #[serde(default)]
        position: Option<u32>,
    },
    /// Blank `clip_id`: the playout front.
    Play {
        #[serde(default)]
        clip_id: String,
    },
    Pop {
        #[serde(default)]
        clip_id: String,
    },
    Move {
        #[serde(default)]
        clip_id: String,
        #[serde(default)]
        position: u32,
    },
    Stop,
    Reset,
    SetClipSeconds(f64),
    SetSeed(u64),
    SetAutoplay(bool),
    SetCanvas(String),
    GetQueue,
    GetState,
    /// fal director chunk: enqueued at the back of the generation queue with
    /// `prompt_version`; `BuildStarted` marks its dispatch (`prompt_applied`).
    Chunk {
        prompt_version: u64,
        prompt: String,
        #[serde(default)]
        first_image: Option<PathBuf>,
        #[serde(default)]
        end_image: Option<PathBuf>,
        seconds: f64,
    },
}

impl ClipCommand {
    /// The wire command name.
    pub fn name(&self) -> &'static str {
        match self {
            ClipCommand::Enqueue { .. } => "enqueue",
            ClipCommand::Play { .. } => "play",
            ClipCommand::Pop { .. } => "pop",
            ClipCommand::Move { .. } => "move",
            ClipCommand::Stop => "stop",
            ClipCommand::Reset => "reset",
            ClipCommand::SetClipSeconds(_) => "set_clip_seconds",
            ClipCommand::SetSeed(_) => "set_seed",
            ClipCommand::SetAutoplay(_) => "set_autoplay",
            ClipCommand::SetCanvas(_) => "set_canvas",
            ClipCommand::GetQueue => "get_queue",
            ClipCommand::GetState => "get_state",
            ClipCommand::Chunk { .. } => "chunk",
        }
    }
}

/// Throughput honesty (design §5.5): what one build took.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct BuildReport {
    /// Submit → built (wall seconds, including queueing on the executor).
    pub build_s: f64,
    /// The clip's playout length.
    pub clip_s: f64,
    pub metrics: JobMetrics,
}

impl BuildReport {
    /// `build_s / clip_s`: below 1.0 keeps up with real time.
    pub fn rtf(&self) -> f64 {
        if self.clip_s > 0.0 {
            self.build_s / self.clip_s
        } else {
            0.0
        }
    }
}

/// `state_update` (fast-h3 `StateUpdate`, field for field).
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ClipState {
    pub clip_seconds: f64,
    pub clip_seconds_min: f64,
    pub clip_seconds_max: f64,
    pub seed: u64,
    pub autoplay: bool,
    pub aspect: String,
    pub width: u32,
    pub height: u32,
    pub playing: bool,
    pub playing_clip_id: Option<Uuid>,
    pub generation_queued: u32,
    pub generation_capacity: u32,
    pub playout_queued: u32,
    pub playout_capacity: u32,
    pub clips_played: u64,
    pub seconds_sent: f64,
    pub valid_commands: Vec<String>,
}

/// Everything a clip session tells its clients: the fast-h3 messages
/// (serialized as `{"type": "<snake_case>", "data": {…}}`) plus two native
/// extensions ([`ClipEvent::is_fasth3`] is false for those).
#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(tag = "type", content = "data", rename_all = "snake_case")]
pub enum ClipEvent {
    ClipQueued {
        clip: ClipInfo,
    },
    ClipGenerated {
        clip: ClipInfo,
        #[serde(skip)]
        build: BuildReport,
    },
    ClipMoved {
        clip: ClipInfo,
        queue: QueueName,
        position: u32,
    },
    ClipStarted {
        clip: ClipInfo,
    },
    ClipFinished {
        clip: ClipInfo,
        seconds_sent: f64,
    },
    ClipStopped {
        clip: ClipInfo,
        seconds_sent: f64,
    },
    ClipPopped {
        clip: ClipInfo,
    },
    ClipFailed {
        clip: ClipInfo,
        reason: String,
    },
    ClipLengthAccepted {
        clip_seconds: f64,
        frames: u32,
    },
    SeedAccepted {
        seed: u64,
    },
    AutoplayAccepted {
        enabled: bool,
    },
    CanvasAccepted {
        aspect: String,
        width: u32,
        height: u32,
    },
    SessionReset {
        cleared_clips: u32,
        was_playing: bool,
    },
    StateUpdate(ClipState),
    QueueUpdate {
        generation: Vec<ClipInfo>,
        playout: Vec<ClipInfo>,
    },
    CommandError {
        command: String,
        reason: String,
    },
    /// Native: a build was dispatched to the executor (fal director
    /// `prompt_applied` for a `Chunk`).
    BuildStarted {
        clip: ClipInfo,
        prompt_version: Option<u64>,
    },
    /// Native: a clip finished with nothing armed while work was still
    /// pending (autoplay on): playout ran dry and the stream holds (fal
    /// director `deadline_missed`).
    Starved {
        after: ClipInfo,
        pending: u32,
    },
}

impl ClipEvent {
    /// The message `type` (snake_case).
    pub fn type_name(&self) -> &'static str {
        match self {
            ClipEvent::ClipQueued { .. } => "clip_queued",
            ClipEvent::ClipGenerated { .. } => "clip_generated",
            ClipEvent::ClipMoved { .. } => "clip_moved",
            ClipEvent::ClipStarted { .. } => "clip_started",
            ClipEvent::ClipFinished { .. } => "clip_finished",
            ClipEvent::ClipStopped { .. } => "clip_stopped",
            ClipEvent::ClipPopped { .. } => "clip_popped",
            ClipEvent::ClipFailed { .. } => "clip_failed",
            ClipEvent::ClipLengthAccepted { .. } => "clip_length_accepted",
            ClipEvent::SeedAccepted { .. } => "seed_accepted",
            ClipEvent::AutoplayAccepted { .. } => "autoplay_accepted",
            ClipEvent::CanvasAccepted { .. } => "canvas_accepted",
            ClipEvent::SessionReset { .. } => "session_reset",
            ClipEvent::StateUpdate(_) => "state_update",
            ClipEvent::QueueUpdate { .. } => "queue_update",
            ClipEvent::CommandError { .. } => "command_error",
            ClipEvent::BuildStarted { .. } => "build_started",
            ClipEvent::Starved { .. } => "starved",
        }
    }

    /// Whether this is one of fast-h3's wire messages.
    pub fn is_fasth3(&self) -> bool {
        !matches!(self, ClipEvent::BuildStarted { .. } | ClipEvent::Starved { .. })
    }

    /// The `data` payload as JSON (what a protocol puts on the wire).
    pub fn data(&self) -> serde_json::Value {
        match serde_json::to_value(self) {
            Ok(serde_json::Value::Object(mut m)) => m.remove("data").unwrap_or(serde_json::Value::Null),
            _ => serde_json::Value::Null,
        }
    }
}

/// Player settings.
#[derive(Clone, Debug, PartialEq)]
pub struct ClipPlayerConfig {
    pub generation_capacity: usize,
    pub playout_capacity: usize,
    /// Initial (and post-`reset`) autoplay. fast-h3: off; fal director: on.
    pub autoplay: bool,
    /// Default clip length; `None`: the model's default frame count.
    pub clip_seconds: Option<f64>,
    /// Default aspect label for `state_update.aspect`; `None`: derived from
    /// the session canvas.
    pub aspect: Option<String>,
    /// Where `AnchorLastFrame` writes last-frame PNGs; `None`:
    /// `$TMPDIR/fv-sessions/<session id>` (removed at close).
    pub session_dir: Option<PathBuf>,
    /// Frames per emitted slice (3).
    pub emit_frames: u32,
    pub event_depth: usize,
    pub media_depth: usize,
    /// Whether an audience is present at start ([`ClipPlayer::set_audience`]).
    pub audience: bool,
    /// Playout speed (1.0 = real time). Tests only.
    pub speed: f64,
}

impl Default for ClipPlayerConfig {
    fn default() -> Self {
        Self {
            generation_capacity: GENERATION_CAPACITY,
            playout_capacity: PLAYOUT_CAPACITY,
            autoplay: false,
            clip_seconds: None,
            aspect: None,
            session_dir: None,
            emit_frames: EMIT_FRAMES,
            event_depth: 256,
            media_depth: 8,
            audience: true,
            speed: 1.0,
        }
    }
}

/// The player's output streams. Drain both.
#[derive(Debug)]
pub struct ClipOutputs {
    pub events: mpsc::Receiver<ClipEvent>,
    pub media: mpsc::Receiver<MediaItem>,
}

enum Msg {
    Command(ClipCommand, oneshot::Sender<Option<ClipEvent>>),
    Audience(bool),
    Close(oneshot::Sender<()>),
}

/// Handle to a running clip player. Dropping every handle stops the player
/// and closes the session.
#[derive(Debug, Clone)]
pub struct ClipPlayer {
    id: SessionId,
    tx: mpsc::Sender<Msg>,
    state: watch::Receiver<ClipState>,
    task: Arc<std::sync::Mutex<Option<JoinHandle<()>>>>,
}

impl std::fmt::Debug for Msg {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Msg::Command(c, _) => write!(f, "Command({})", c.name()),
            Msg::Audience(b) => write!(f, "Audience({b})"),
            Msg::Close(_) => write!(f, "Close"),
        }
    }
}

fn gone() -> ApiError {
    ApiError::internal("the clip session has ended")
}

impl ClipPlayer {
    /// Starts the player task over `session` (needs a tokio runtime).
    /// Fails for `AnchorLastFrame` on a model that cannot build I2V clips.
    pub fn start(session: ClipSession, cfg: ClipPlayerConfig) -> Result<(Self, ClipOutputs), ApiError> {
        let spec = session.spec().clone();
        let caps = session.caps().clone();
        if matches!(spec.continuity, Continuity::AnchorLastFrame { .. }) && !caps.supports(Task::I2V) {
            return Err(ApiError::invalid_param(
                "continuity",
                format!("`{}` cannot anchor clips on the last frame (no image-to-video)", caps.id),
            ));
        }
        let id = session.id();
        let (ev_tx, ev_rx) = mpsc::channel(cfg.event_depth.max(1));
        let (media_tx, media_rx) = mpsc::channel(cfg.media_depth.max(1));
        let (msg_tx, msg_rx) = mpsc::channel(64);
        let (build_tx, build_rx) = mpsc::channel(4);
        let mut p = Player::new(session, spec, caps, cfg, ev_tx, media_tx, build_tx)?;
        let (state_tx, state_rx) = watch::channel(p.snapshot());
        let task = tokio::spawn(async move { p.run(msg_rx, build_rx, state_tx).await });
        Ok((
            Self {
                id,
                tx: msg_tx,
                state: state_rx,
                task: Arc::new(std::sync::Mutex::new(Some(task))),
            },
            ClipOutputs {
                events: ev_rx,
                media: media_rx,
            },
        ))
    }

    pub fn id(&self) -> SessionId {
        self.id
    }

    /// Runs one command. `Ok(Some(reply))` is the correlated reply,
    /// `Ok(None)` a bodyless ack (including every refusal, which is
    /// broadcast as `command_error`). Broadcasts caused by the command are on
    /// the event stream before this returns.
    pub async fn command(&self, cmd: ClipCommand) -> Result<Option<ClipEvent>, ApiError> {
        let (tx, rx) = oneshot::channel();
        self.tx.send(Msg::Command(cmd, tx)).await.map_err(|_| gone())?;
        rx.await.map_err(|_| gone())
    }

    /// Whether any client is watching. Without an audience no build is
    /// submitted and a playing clip is cut quietly (`Orphaned`, §5.2).
    pub async fn set_audience(&self, present: bool) -> Result<(), ApiError> {
        self.tx.send(Msg::Audience(present)).await.map_err(|_| gone())
    }

    /// The latest `state_update` snapshot.
    pub fn state(&self) -> ClipState {
        self.state.borrow().clone()
    }

    pub fn watch_state(&self) -> watch::Receiver<ClipState> {
        self.state.clone()
    }

    pub fn is_closed(&self) -> bool {
        self.tx.is_closed()
    }

    /// Stops the player: cancels the build in flight, drops the queues and
    /// closes the session (freeing the executor). Idempotent.
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

struct Playing {
    entry: ClipEntry,
    /// Faded wire audio (interleaved).
    audio: Option<Vec<f32>>,
    next: u32,
    clock_start: Option<Instant>,
}

struct InFlight {
    seq: u64,
    clip_id: Uuid,
    cancel: CancelToken,
}

struct BuildDone {
    seq: u64,
    clip_id: Uuid,
    result: Result<(BuiltClip, Option<PathBuf>), ApiError>,
}

struct Defaults {
    clip_frames: u32,
    seed: u64,
    aspect: String,
    canvas: (u32, u32),
    autoplay: bool,
}

struct Player {
    session: ClipSession,
    spec: SessionSpec,
    caps: ModelCaps,
    cfg: ClipPlayerConfig,
    fps: u32,
    channels: Option<u8>,
    gen: ClipQueue,
    playout: ClipQueue,
    play_request: Option<ClipEntry>,
    playing: Option<Playing>,
    stop_playout: bool,
    build: Option<InFlight>,
    build_seq: u64,
    clip_frames: u32,
    seed: u64,
    aspect: String,
    canvas: (u32, u32),
    autoplay: bool,
    defaults: Defaults,
    clips_played: u64,
    frames_sent: u64,
    anchor: Option<PathBuf>,
    audience: bool,
    session_dir: PathBuf,
    own_dir: bool,
    events: mpsc::Sender<ClipEvent>,
    media: mpsc::Sender<MediaItem>,
    build_tx: mpsc::Sender<BuildDone>,
    /// Published clip-length range (rounded inward to ms).
    range_s: (f64, f64),
    /// Legal frame range for the stream (on the grid).
    range_frames: (u32, u32),
}

fn round3(x: f64) -> f64 {
    (x * 1000.0).round() / 1000.0
}

fn gcd(a: u32, b: u32) -> u32 {
    if b == 0 {
        a
    } else {
        gcd(b, a % b)
    }
}

fn parse_aspect(a: &str) -> Option<f64> {
    let (w, h) = a.trim().split_once(':')?;
    let (w, h): (f64, f64) = (w.trim().parse().ok()?, h.trim().parse().ok()?);
    (w > 0.0 && h > 0.0 && w.is_finite() && h.is_finite()).then(|| w / h)
}

fn aspect_label(w: u32, h: u32) -> String {
    let r = w as f64 / h.max(1) as f64;
    for c in ["16:9", "9:16", "1:1", "4:3", "3:4"] {
        if let Some(cr) = parse_aspect(c) {
            if (r / cr - 1.0).abs() < 0.03 {
                return c.to_owned();
            }
        }
    }
    let g = gcd(w, h).max(1);
    format!("{}:{}", w / g, h / g)
}

impl Player {
    fn new(
        session: ClipSession,
        spec: SessionSpec,
        caps: ModelCaps,
        cfg: ClipPlayerConfig,
        events: mpsc::Sender<ClipEvent>,
        media: mpsc::Sender<MediaItem>,
        build_tx: mpsc::Sender<BuildDone>,
    ) -> Result<Self, ApiError> {
        let fps = spec.fps.max(1);
        let grid = &caps.frames;
        let (min_s, max_s) = match caps.stream {
            Some(StreamCaps::Clip { min_s, max_s }) => (min_s as f64, max_s as f64),
            _ => (grid.min as f64 / fps as f64, grid.max as f64 / fps as f64),
        };
        let lo_n = ((min_s * fps as f64) - 1e-3).ceil().max(grid.min as f64) as u32;
        let hi_n = ((max_s * fps as f64) + 1e-3).floor().min(grid.max as f64) as u32;
        let lo = grid.align_up(lo_n).unwrap_or(grid.min);
        let hi = if grid.step == 0 || hi_n < grid.offset {
            grid.max
        } else {
            grid.offset + grid.step * ((hi_n - grid.offset) / grid.step)
        }
        .max(lo);
        let range_frames = (lo, hi);
        let range_s = (
            (lo as f64 / fps as f64 * 1000.0).ceil() / 1000.0,
            (hi as f64 / fps as f64 * 1000.0).floor() / 1000.0,
        );
        let clip_frames = match cfg.clip_seconds {
            Some(s) => snap_frames(s, fps, grid, range_frames, range_s).map_err(|r| ApiError::invalid_param("clip_seconds", r))?,
            None => grid.default.clamp(lo, hi),
        };
        let seed = spec.seed.unwrap_or(1000);
        let aspect = cfg
            .aspect
            .clone()
            .unwrap_or_else(|| aspect_label(spec.canvas.0, spec.canvas.1));
        let (session_dir, own_dir) = match &cfg.session_dir {
            Some(d) => (d.clone(), false),
            None => (
                std::env::temp_dir().join("fv-sessions").join(session.id().to_string()),
                true,
            ),
        };
        let channels = spec.tracks.audio.as_ref().map(|a| a.channels);
        if channels.is_some() {
            spec.tracks.samples_per_frame()?;
        }
        Ok(Self {
            defaults: Defaults {
                clip_frames,
                seed,
                aspect: aspect.clone(),
                canvas: spec.canvas,
                autoplay: cfg.autoplay,
            },
            gen: ClipQueue::new(cfg.generation_capacity),
            playout: ClipQueue::new(cfg.playout_capacity),
            play_request: None,
            playing: None,
            stop_playout: false,
            build: None,
            build_seq: 0,
            clip_frames,
            seed,
            canvas: spec.canvas,
            aspect,
            autoplay: cfg.autoplay,
            clips_played: 0,
            frames_sent: 0,
            anchor: None,
            audience: cfg.audience,
            session_dir,
            own_dir,
            events,
            media,
            build_tx,
            range_s,
            range_frames,
            fps,
            channels,
            session,
            spec,
            caps,
            cfg,
        })
    }

    // ------------------------------------------------------------ snapshot

    fn current(&self) -> Option<&ClipEntry> {
        self.playing
            .as_ref()
            .map(|p| &p.entry)
            .or(self.play_request.as_ref())
    }

    fn seconds_sent(&self) -> f64 {
        (self.frames_sent as f64 / self.fps as f64 * 100.0).round() / 100.0
    }

    fn snapshot(&self) -> ClipState {
        let cur = self.current();
        ClipState {
            clip_seconds: round3(self.clip_frames as f64 / self.fps as f64),
            clip_seconds_min: self.range_s.0,
            clip_seconds_max: self.range_s.1,
            seed: self.seed,
            autoplay: self.autoplay,
            aspect: self.aspect.clone(),
            width: self.canvas.0,
            height: self.canvas.1,
            playing: cur.is_some(),
            playing_clip_id: cur.map(|e| e.clip_id),
            generation_queued: self.gen.len() as u32,
            generation_capacity: self.gen.capacity() as u32,
            playout_queued: self.playout.len() as u32,
            playout_capacity: self.playout.capacity() as u32,
            clips_played: self.clips_played,
            seconds_sent: self.seconds_sent(),
            valid_commands: valid_commands(
                cur.is_some(),
                self.gen.len(),
                self.gen.capacity(),
                self.playout.len(),
            )
            .into_iter()
            .map(str::to_owned)
            .collect(),
        }
    }

    fn queue_update(&self) -> ClipEvent {
        ClipEvent::QueueUpdate {
            generation: self.gen.snapshot(),
            playout: self.playout.snapshot(),
        }
    }

    async fn send(&self, ev: ClipEvent) {
        let _ = self.events.send(ev).await;
    }

    async fn send_state(&self) {
        self.send(ClipEvent::StateUpdate(self.snapshot())).await;
    }

    async fn send_queue(&self) {
        self.send(self.queue_update()).await;
    }

    async fn refuse(&self, command: &str, reason: impl Into<String>) -> Option<ClipEvent> {
        let reason = reason.into();
        tracing::info!(session = %self.session.id(), command, reason = %reason, "command refused");
        self.send(ClipEvent::CommandError {
            command: command.to_owned(),
            reason,
        })
        .await;
        None
    }

    async fn media(&self, item: MediaItem) {
        let _ = self.media.send(item).await;
    }

    // ------------------------------------------------------------ run loop

    async fn run(
        &mut self,
        mut msgs: mpsc::Receiver<Msg>,
        mut builds: mpsc::Receiver<BuildDone>,
        state: watch::Sender<ClipState>,
    ) {
        let mut closer: Option<oneshot::Sender<()>> = None;
        loop {
            self.pump_builds().await;
            self.autoplay_arm().await;
            self.advance_playout().await;
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
                    Some(Msg::Command(c, reply)) => {
                        let r = self.handle(c).await;
                        let _ = reply.send(r);
                    }
                    Some(Msg::Audience(b)) => self.set_audience(b).await,
                    Some(Msg::Close(tx)) => {
                        closer = Some(tx);
                        break;
                    }
                    None => break,
                },
                Some(done) = builds.recv() => self.apply_build(done).await,
                _ = sleep_until(deadline), if deadline.is_some() => self.emit_slice().await,
            }
        }
        if let Some(b) = self.build.take() {
            b.cancel.cancel();
        }
        if self.playing.is_some() {
            self.media(MediaItem::Clear).await;
        }
        self.gen.clear();
        self.playout.clear();
        if self.own_dir {
            let _ = std::fs::remove_dir_all(&self.session_dir);
        }
        self.session.end();
        if let Some(tx) = closer {
            let _ = tx.send(());
        }
    }

    async fn set_audience(&mut self, present: bool) {
        self.audience = present;
        if !present && self.playing.is_some() {
            self.finish_play(PlayOutcome::Gone).await;
        }
    }

    // ------------------------------------------------------------ builds

    /// Applies nothing itself; submits the front non-building entry when the
    /// playout queue has room (the reservation) and an audience is present.
    async fn pump_builds(&mut self) {
        loop {
            if self.build.is_some() || !self.audience || self.playout.is_full() {
                return;
            }
            let anchor_mode = matches!(self.spec.continuity, Continuity::AnchorLastFrame { .. });
            let anchor = self.anchor.clone();
            let canvas = self.canvas;
            let Some(entry) = self.gen.next_to_build() else {
                return;
            };
            entry.building = true;
            let first_frame = entry
                .first_frame
                .clone()
                .or(if anchor_mode { anchor } else { None });
            let b = ClipBuild {
                prompt: entry.prompt.clone(),
                negative_prompt: None,
                seed: Some(entry.seed),
                seconds: None,
                frames: Some(entry.frames),
                canvas: Some(canvas),
                first_frame,
                last_frame: entry.last_frame.clone(),
                audio_drive: None,
            };
            let info = entry.info();
            let prompt_version = entry.prompt_version;
            let clip_id = entry.clip_id;
            match self.session.build(b).await {
                Ok(handle) => {
                    self.build_seq += 1;
                    let seq = self.build_seq;
                    self.build = Some(InFlight {
                        seq,
                        clip_id,
                        cancel: handle.cancel.clone(),
                    });
                    tracing::info!(session = %self.session.id(), clip = %clip_id, frames = info.frames, "clip build submitted");
                    let tx = self.build_tx.clone();
                    let fps = self.fps;
                    let channels = self.channels;
                    let anchor_path = anchor_mode.then(|| self.session_dir.join(format!("anchor-{clip_id}.png")));
                    tokio::spawn(async move {
                        let t0 = Instant::now();
                        let result = match handle.wait().await {
                            Ok(out) => tokio::task::spawn_blocking(move || {
                                prepare_build(out, fps, channels, anchor_path)
                            })
                            .await
                            .unwrap_or_else(|e| Err(ApiError::internal(format!("preparing the clip: {e}")))),
                            Err(e) => Err(e),
                        }
                        .map(|(mut b, a)| {
                            b.build_s = t0.elapsed().as_secs_f64();
                            (b, a)
                        });
                        let _ = tx.send(BuildDone { seq, clip_id, result }).await;
                    });
                    self.send(ClipEvent::BuildStarted {
                        clip: info,
                        prompt_version,
                    })
                    .await;
                    return;
                }
                Err(e) => {
                    // Refused at submit: fail the clip, move on to the next.
                    if let Some(entry) = self.gen.remove(clip_id) {
                        self.send(ClipEvent::ClipFailed {
                            clip: entry.info(),
                            reason: e.message.clone(),
                        })
                        .await;
                        self.send_queue().await;
                        self.send_state().await;
                    }
                }
            }
        }
    }

    async fn apply_build(&mut self, done: BuildDone) {
        match &self.build {
            Some(b) if b.seq == done.seq => self.build = None,
            // A build cancelled by `pop`/`reset`: nothing to land on.
            _ => return,
        }
        let Some(entry) = self.gen.get_mut(done.clip_id) else {
            return;
        };
        entry.building = false;
        match done.result {
            Err(e) => {
                let entry = self.gen.remove(done.clip_id).expect("entry present");
                tracing::warn!(session = %self.session.id(), clip = %done.clip_id, error = %e, "clip build failed");
                self.send(ClipEvent::ClipFailed {
                    clip: entry.info(),
                    reason: e.message,
                })
                .await;
                self.send_queue().await;
                self.send_state().await;
            }
            Ok((built, anchor)) => {
                if let Some(a) = anchor {
                    if let Some(old) = self.anchor.replace(a) {
                        let _ = std::fs::remove_file(old);
                    }
                }
                let mut entry = self.gen.remove(done.clip_id).expect("entry present");
                let report = BuildReport {
                    build_s: built.build_s,
                    clip_s: entry.seconds(),
                    metrics: built.metrics.clone(),
                };
                entry.frames = built.frames.len() as u32;
                entry.built = Some(built);
                tracing::info!(
                    session = %self.session.id(),
                    clip = %done.clip_id,
                    build_s = report.build_s,
                    rtf = report.rtf(),
                    "clip generated"
                );
                let info = entry.info();
                // The submit-time reservation guarantees room: only builds
                // add here, and play/pop only shrink it.
                if self.playout.add(entry, None).is_err() {
                    tracing::error!("playout queue overflow despite the reservation");
                    return;
                }
                self.send(ClipEvent::ClipGenerated { clip: info, build: report }).await;
                self.send_queue().await;
                self.send_state().await;
            }
        }
    }

    // ------------------------------------------------------------ playout

    async fn autoplay_arm(&mut self) {
        if !self.autoplay || self.play_request.is_some() || self.playing.is_some() || !self.audience {
            return;
        }
        if let Some(e) = self.playout.pop_front() {
            self.play_request = Some(e);
            self.send_queue().await;
            self.send_state().await;
        }
    }

    /// Starts an armed clip, or cuts the playing one on `stop`.
    async fn advance_playout(&mut self) {
        if self.stop_playout && self.playing.is_some() {
            self.finish_play(PlayOutcome::Stopped).await;
        }
        if self.playing.is_some() || !self.audience {
            return;
        }
        let Some(entry) = self.play_request.take() else {
            return;
        };
        if self.stop_playout {
            // Cut between arming and starting.
            self.stop_playout = false;
            self.clips_played += 1;
            self.send(ClipEvent::ClipStopped {
                clip: entry.info(),
                seconds_sent: self.seconds_sent(),
            })
            .await;
            self.send_state().await;
            return;
        }
        let fade = match self.spec.continuity {
            Continuity::HardCut => None,
            Continuity::Crossfade { ms } | Continuity::AnchorLastFrame { crossfade_ms: ms } => Some(ms),
        };
        let audio = entry.built.as_ref().and_then(|b| b.audio.as_ref()).map(|pcm| match fade {
            Some(ms) if ms > 0 => apply_clip_fades(pcm, ms, self.clips_played > 0, true).samples.to_vec(),
            _ => pcm.samples.to_vec(),
        });
        let info = entry.info();
        self.playing = Some(Playing {
            entry,
            audio,
            next: 0,
            clock_start: None,
        });
        self.send(ClipEvent::ClipStarted { clip: info }).await;
        self.send_state().await;
    }

    /// When the next slice is due (re-anchoring: never in the past, so a
    /// stall shifts the schedule instead of bursting).
    fn slice_deadline(&mut self) -> Option<tokio::time::Instant> {
        let fps = self.fps as f64 * self.cfg.speed.max(1e-3);
        let p = self.playing.as_mut()?;
        let now = Instant::now();
        let content = Duration::from_secs_f64(p.next as f64 / fps);
        let start = *p.clock_start.get_or_insert(now);
        let due = (start + content).max(now);
        p.clock_start = Some(due - content);
        Some(tokio::time::Instant::from_std(due))
    }

    async fn emit_slice(&mut self) {
        let emit = self.cfg.emit_frames.max(1);
        let fps = self.fps;
        let ch = self.channels.unwrap_or(1) as usize;
        let Some(p) = self.playing.as_mut() else {
            return;
        };
        let Some(built) = p.entry.built.as_ref() else {
            return;
        };
        let total = built.frames.len() as u32;
        let lo = p.next;
        let hi = (lo + emit).min(total);
        let frames: Vec<RgbFrame> = built.frames[lo as usize..hi as usize].to_vec();
        let audio = p.audio.as_ref().map(|a| {
            let s0 = frame_sample_offset(lo as u64, fps, WIRE_RATE) as usize * ch;
            let s1 = frame_sample_offset(hi as u64, fps, WIRE_RATE) as usize * ch;
            a[s0.min(a.len())..s1.min(a.len())].to_vec()
        });
        p.next = hi;
        let clip_id = p.entry.clip_id;
        self.frames_sent += (hi - lo) as u64;
        self.media(MediaItem::Slice(MediaSlice {
            clip_id,
            first_frame: lo,
            frames,
            audio,
        }))
        .await;
        if hi >= total {
            self.finish_play(PlayOutcome::Finished).await;
        }
    }

    async fn finish_play(&mut self, outcome: PlayOutcome) {
        let Some(p) = self.playing.take() else {
            return;
        };
        self.stop_playout = false;
        let armed = self.play_request.is_some()
            || (self.autoplay && self.audience && self.playout.head().is_some());
        if outcome != PlayOutcome::Finished {
            self.media(MediaItem::Clear).await;
        }
        self.media(MediaItem::ClipEnd {
            clip_id: p.entry.clip_id,
            outcome,
            armed,
        })
        .await;
        self.clips_played += 1;
        let clip = p.entry.info();
        match outcome {
            PlayOutcome::Finished => {
                self.send(ClipEvent::ClipFinished {
                    clip: clip.clone(),
                    seconds_sent: self.seconds_sent(),
                })
                .await;
                let pending = self.gen.len() as u32;
                if !armed && self.autoplay && (pending > 0 || self.build.is_some()) {
                    self.send(ClipEvent::Starved { after: clip, pending }).await;
                }
            }
            PlayOutcome::Stopped => {
                self.send(ClipEvent::ClipStopped {
                    clip,
                    seconds_sent: self.seconds_sent(),
                })
                .await;
            }
            PlayOutcome::Gone => {}
        }
        self.send_state().await;
    }

    // ------------------------------------------------------------ commands

    fn snap(&self, seconds: f64) -> Result<u32, String> {
        snap_frames(seconds, self.fps, &self.caps.frames, self.range_frames, self.range_s)
    }

    fn cancel_build_for(&mut self, id: Uuid) {
        if self.build.as_ref().is_some_and(|b| b.clip_id == id) {
            if let Some(b) = self.build.take() {
                b.cancel.cancel();
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    async fn enqueue(
        &mut self,
        command: &str,
        prompt: &str,
        metadata: String,
        seed: Option<u64>,
        seconds: Option<f64>,
        position: Option<u32>,
        prompt_version: Option<u64>,
        first_frame: Option<PathBuf>,
        last_frame: Option<PathBuf>,
    ) -> Option<ClipEvent> {
        let prompt = prompt.trim();
        if prompt.is_empty() {
            return self.refuse(command, "The prompt is empty; a clip needs one.").await;
        }
        if prompt.chars().count() > MAX_PROMPT_CHARS {
            return self
                .refuse(command, format!("The prompt is longer than {MAX_PROMPT_CHARS} characters."))
                .await;
        }
        if metadata.chars().count() > MAX_METADATA_CHARS {
            return self
                .refuse(command, format!("The metadata is longer than {MAX_METADATA_CHARS} characters."))
                .await;
        }
        if self.gen.is_full() {
            return self
                .refuse(
                    command,
                    format!(
                        "The generation queue is full ({} clips); `pop` one or wait for a build to finish.",
                        self.gen.capacity()
                    ),
                )
                .await;
        }
        let frames = match seconds {
            Some(s) => match self.snap(s) {
                Ok(f) => f,
                Err(r) => return self.refuse(command, r).await,
            },
            None => self.clip_frames,
        };
        let seed = seed.unwrap_or_else(|| {
            let s = self.seed;
            self.seed = self.seed.wrapping_add(1);
            s
        });
        let mut entry = ClipEntry::new(prompt, metadata, frames, self.fps, seed);
        entry.prompt_version = prompt_version;
        entry.first_frame = first_frame;
        entry.last_frame = last_frame;
        let info = entry.info();
        let _ = self.gen.add(entry, position.map(|p| p as usize));
        self.send_queue().await;
        self.send_state().await;
        Some(ClipEvent::ClipQueued { clip: info })
    }

    async fn handle(&mut self, cmd: ClipCommand) -> Option<ClipEvent> {
        let name = cmd.name();
        match cmd {
            ClipCommand::Enqueue {
                prompt,
                metadata,
                seed,
                seconds,
                position,
            } => {
                self.enqueue(name, &prompt, metadata, seed, seconds, position, None, None, None)
                    .await
            }
            ClipCommand::Chunk {
                prompt_version,
                prompt,
                first_image,
                end_image,
                seconds,
            } => {
                self.enqueue(
                    name,
                    &prompt,
                    String::new(),
                    None,
                    Some(seconds),
                    None,
                    Some(prompt_version),
                    first_image,
                    end_image,
                )
                .await
            }
            ClipCommand::Play { clip_id } => {
                if self.current().is_some() {
                    return self.refuse(name, "A clip is already playing; send `stop` first.").await;
                }
                let raw = clip_id.trim();
                let entry = if raw.is_empty() {
                    match self.playout.pop_front() {
                        Some(e) => e,
                        None => {
                            return self
                                .refuse(
                                    name,
                                    "The playout queue is empty; `enqueue` a clip and wait for `clip_generated`.",
                                )
                                .await
                        }
                    }
                } else {
                    let id = Uuid::parse_str(raw).ok();
                    match id.and_then(|id| self.playout.remove(id)) {
                        Some(e) => e,
                        None if id.is_some_and(|id| self.gen.contains(id)) => {
                            return self
                                .refuse(
                                    name,
                                    "That clip is still generating; `clip_generated` will announce it entering the playout queue.",
                                )
                                .await
                        }
                        None => return self.refuse(name, format!("No queued clip has id '{raw}'.")).await,
                    }
                };
                self.play_request = Some(entry);
                self.send_queue().await;
                self.send_state().await;
                None
            }
            ClipCommand::Pop { clip_id } => {
                let raw = clip_id.trim();
                let id = Uuid::parse_str(raw).ok();
                let entry = id.and_then(|id| self.gen.remove(id).or_else(|| self.playout.remove(id)));
                let Some(entry) = entry else {
                    let reason = if raw.is_empty() {
                        "Pass the `clip_id` of the queued clip to remove.".to_owned()
                    } else {
                        format!("No queued clip has id '{raw}'.")
                    };
                    return self.refuse(name, reason).await;
                };
                self.cancel_build_for(entry.clip_id);
                self.send_queue().await;
                self.send_state().await;
                Some(ClipEvent::ClipPopped { clip: entry.info() })
            }
            ClipCommand::Move { clip_id, position } => {
                let raw = clip_id.trim();
                let id = Uuid::parse_str(raw).ok();
                let moved = id.and_then(|id| {
                    if let Some(i) = self.gen.move_to(id, position as usize) {
                        Some((QueueName::Generation, i, self.gen.get(id).map(ClipEntry::info)))
                    } else {
                        self.playout
                            .move_to(id, position as usize)
                            .map(|i| (QueueName::Playout, i, self.playout.get(id).map(ClipEntry::info)))
                    }
                });
                let Some((queue, i, Some(clip))) = moved else {
                    let reason = if raw.is_empty() {
                        "Pass the `clip_id` of the queued clip to move.".to_owned()
                    } else {
                        format!("No queued clip has id '{raw}'.")
                    };
                    return self.refuse(name, reason).await;
                };
                self.send_queue().await;
                Some(ClipEvent::ClipMoved {
                    clip,
                    queue,
                    position: i as u32,
                })
            }
            ClipCommand::Stop => {
                if self.current().is_none() {
                    return self.refuse(name, "No clip is playing.").await;
                }
                self.stop_playout = true;
                None
            }
            ClipCommand::Reset => {
                let was_playing = self.current().is_some();
                let mut cleared = (self.gen.clear() + self.playout.clear()) as u32;
                if self.play_request.take().is_some() {
                    cleared += 1;
                }
                if self.playing.is_some() {
                    self.stop_playout = true;
                }
                if let Some(b) = self.build.take() {
                    b.cancel.cancel();
                }
                self.clip_frames = self.defaults.clip_frames;
                self.seed = self.defaults.seed;
                self.aspect = self.defaults.aspect.clone();
                self.canvas = self.defaults.canvas;
                self.autoplay = self.defaults.autoplay;
                if let Some(a) = self.anchor.take() {
                    let _ = std::fs::remove_file(a);
                }
                self.media(MediaItem::Clear).await;
                self.send_queue().await;
                self.send_state().await;
                Some(ClipEvent::SessionReset {
                    cleared_clips: cleared,
                    was_playing,
                })
            }
            ClipCommand::SetClipSeconds(s) => match self.snap(s) {
                Ok(f) => {
                    self.clip_frames = f;
                    self.send_state().await;
                    Some(ClipEvent::ClipLengthAccepted {
                        clip_seconds: round3(f as f64 / self.fps as f64),
                        frames: f,
                    })
                }
                Err(r) => self.refuse(name, r).await,
            },
            ClipCommand::SetSeed(s) => {
                self.seed = s;
                self.send_state().await;
                Some(ClipEvent::SeedAccepted { seed: s })
            }
            ClipCommand::SetAutoplay(b) => {
                self.autoplay = b;
                self.send_state().await;
                Some(ClipEvent::AutoplayAccepted { enabled: b })
            }
            ClipCommand::SetCanvas(aspect) => {
                if self.current().is_some() || !self.gen.is_empty() || !self.playout.is_empty() {
                    return self
                        .refuse(
                            name,
                            "The canvas is fixed while clips are queued or playing; `reset` or play the queue out first.",
                        )
                        .await;
                }
                let Some(ratio) = parse_aspect(&aspect) else {
                    return self
                        .refuse(
                            name,
                            format!("unknown aspect '{aspect}'; choose one of {ASPECT_CHOICES:?}"),
                        )
                        .await;
                };
                let (lo, hi) = self.caps.canvas.aspect;
                if !(lo as f64..=hi as f64).contains(&ratio) {
                    return self
                        .refuse(name, format!("aspect ratios run from {lo} to {hi}, got {aspect}"))
                        .await;
                }
                let short = self.defaults.canvas.0.min(self.defaults.canvas.1);
                let (w, h) = canvas_for_aspect(&self.caps.canvas, ratio, short);
                self.canvas = (w, h);
                self.aspect = aspect.trim().to_owned();
                self.send_state().await;
                Some(ClipEvent::CanvasAccepted {
                    aspect: self.aspect.clone(),
                    width: w,
                    height: h,
                })
            }
            ClipCommand::GetQueue => Some(self.queue_update()),
            ClipCommand::GetState => Some(ClipEvent::StateUpdate(self.snapshot())),
        }
    }
}

async fn sleep_until(d: Option<tokio::time::Instant>) {
    match d {
        Some(d) => tokio::time::sleep_until(d).await,
        None => std::future::pending().await,
    }
}

/// Snaps a requested clip length to a generatable frame count: round to
/// frames, up onto the grid, then clamp into the stream's range (fast-h3
/// `frames_for_seconds`). Lengths outside the published range are refused.
fn snap_frames(
    seconds: f64,
    fps: u32,
    grid: &fastvideo_protocol::FrameGrid,
    (lo, hi): (u32, u32),
    (min_s, max_s): (f64, f64),
) -> Result<u32, String> {
    if !seconds.is_finite() || seconds < min_s - 1e-9 || seconds > max_s + 1e-9 {
        return Err(format!(
            "The clip length must be between {min_s} and {max_s} seconds, got {seconds}."
        ));
    }
    let n = ((seconds * fps as f64).round() as u32).max(1);
    let f = grid.next_on_grid(n).unwrap_or(hi);
    Ok(f.clamp(lo, hi))
}

/// Executor output → a playable clip (runs on the blocking pool): wire
/// audio (48 kHz, lockstep length) and, with `anchor`, the last frame as a
/// PNG.
fn prepare_build(
    out: ClipOutput,
    fps: u32,
    channels: Option<u8>,
    anchor: Option<PathBuf>,
) -> Result<(BuiltClip, Option<PathBuf>), ApiError> {
    let frames = out.frames.unwrap_or_default();
    if frames.is_empty() {
        return Err(ApiError::engine_failed("the build returned no frames"));
    }
    let audio = match channels {
        Some(ch) => Some(
            prepare_clip_audio(out.audio.as_ref(), frames.len() as u32, fps, WIRE_RATE, ch)
                .map_err(|e| ApiError::internal(format!("clip audio: {e}")))?,
        ),
        None => None,
    };
    let anchor = match anchor {
        Some(path) => {
            let last = frames.last().expect("non-empty");
            write_png(last, &path)?;
            Some(path)
        }
        None => None,
    };
    Ok((
        BuiltClip {
            frames: Arc::new(frames),
            audio,
            build_s: 0.0,
            metrics: out.metrics,
        },
        anchor,
    ))
}

/// Writes an RGB24 frame as a PNG (the `AnchorLastFrame` keyframe).
pub fn write_png(f: &RgbFrame, path: &std::path::Path) -> Result<(), ApiError> {
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)
            .map_err(|e| ApiError::internal(format!("creating {}: {e}", dir.display())))?;
    }
    let img = image::RgbImage::from_raw(f.width, f.height, f.data.to_vec())
        .ok_or_else(|| ApiError::internal("anchor frame has the wrong size"))?;
    img.save(path)
        .map_err(|e| ApiError::internal(format!("writing {}: {e}", path.display())))
}

#[cfg(test)]
mod tests {
    use super::*;
    use fastvideo_protocol::FrameGrid;

    #[test]
    fn snapping_matches_fasth3_clip_plan() {
        // H3: 17n+5 at 24 fps, published 5.167..14.375 s (124..345 frames).
        let g = FrameGrid::new(17, 5, 124, 345, 124);
        let r = (124, 345);
        let s = (5.167, 14.375);
        assert_eq!(snap_frames(5.167, 24, &g, r, s).unwrap(), 124);
        assert_eq!(snap_frames(10.0, 24, &g, r, s).unwrap(), 243);
        assert_eq!(snap_frames(14.375, 24, &g, r, s).unwrap(), 345);
        assert!(snap_frames(15.0, 24, &g, r, s).is_err());
        assert!(snap_frames(5.0, 24, &g, r, s).is_err());
        assert!(snap_frames(f64::NAN, 24, &g, r, s).is_err());
    }

    #[test]
    fn aspect_labels() {
        assert_eq!(aspect_label(1344, 768), "16:9");
        assert_eq!(aspect_label(832, 480), "16:9");
        assert_eq!(aspect_label(768, 768), "1:1");
        assert_eq!(aspect_label(640, 200), "16:5");
        assert_eq!(parse_aspect("9:16"), Some(9.0 / 16.0));
        assert_eq!(parse_aspect("x"), None);
    }

    #[test]
    fn events_serialize_as_fasth3_messages() {
        let e = ClipEvent::CommandError {
            command: "play".into(),
            reason: "No clip is playing.".into(),
        };
        assert_eq!(e.type_name(), "command_error");
        let v = serde_json::to_value(&e).unwrap();
        assert_eq!(v["type"], "command_error");
        assert_eq!(v["data"]["command"], "play");
        let g = ClipEvent::ClipGenerated {
            clip: ClipEntry::new("p", "", 124, 24, 1).info(),
            build: BuildReport::default(),
        };
        let d = g.data();
        assert!(d.get("build").is_none());
        assert!(d.get("clip").is_some());
        let st = ClipEvent::StateUpdate(ClipState::default());
        assert_eq!(st.data()["playing_clip_id"], serde_json::Value::Null);
        let c: ClipCommand = serde_json::from_value(serde_json::json!({"type": "enqueue", "data": {"prompt": "x"}})).unwrap();
        assert_eq!(c.name(), "enqueue");
    }
}
