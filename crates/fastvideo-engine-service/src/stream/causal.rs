//! `CausalSession`: SF-Wan block rollout under an exclusive executor lease
//! (design §5.4).
//!
//! WP-02 provides the engine side: the lease, the per-turn block loop on the
//! executor, prompt/seed/pause/reset controls applied at block boundaries,
//! and a bounded block channel (depth `EngineConfig::causal_depth`, 4) on
//! which the executor blocks when the consumer falls behind.
//!
//! WP-15 adds, on top:
//!
//! - [`CausalControl`]: a cloneable control handle (prompt, pause, seed,
//!   reset, stats, TTFF) so protocol handlers can steer a session whose
//!   block stream a pacer owns ([`super::pace::spawn_causal_pacer`]);
//! - the Reactor causal command set ([`CausalCommand`] → [`CausalReply`]:
//!   `set_prompt`, `set_paused`, `set_seed`, `reset`, `get_state`, answered
//!   with `state_update{prompt,paused,seed,block_index,unique_fps}` or
//!   `command_error{command,reason}`);
//! - TTFF phases ([`Ttff`]: `load`, `first_block`, `transport`).
//!
//! Prompt changes land at the next block boundary: the executor snapshots
//! the prompt when it starts a block ([`BlockInput`]), so a block is never
//! generated under two prompts. A seed change takes effect at the next
//! `reset` (the rollout draws its noise per reset).

use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

use fastvideo_protocol::{draw_seed, ApiError, ModelCaps, Pcm, RgbFrame, SessionSpec};
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;

use crate::backend::{BlockInput, BlockStats, CausalSpec, SessionId};
use crate::cancel::{lock, CancelToken};
use crate::service::Shared;

/// One generated block, in order.
#[derive(Clone, Debug, PartialEq)]
pub struct CausalBlock {
    /// 0-based since open or the last reset.
    pub index: u64,
    /// The prompt version this block was generated with.
    pub prompt_version: u64,
    /// Whether the KV cache was reset before this block.
    pub reset: bool,
    pub frames: Vec<RgbFrame>,
    pub audio: Option<Pcm>,
    pub stats: BlockStats,
}

/// Session counters (`state_update{block_index, …}`).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct CausalStats {
    pub blocks: u64,
    pub frames: u64,
    pub last_block_ms: f64,
    /// Blocks generated but dropped because the session closed while the
    /// channel was full.
    pub dropped: u64,
    /// Next block's index (`state_update.block_index`).
    pub block_index: u64,
    /// Unique frames per second of playout, as the pacer reports it.
    pub unique_fps: f64,
    /// The pacer's current playout rate.
    pub effective_fps: f64,
}

/// Time to first frame, by phase (design §5.4 metrics), in milliseconds.
/// A phase is `None` until it has happened.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Ttff {
    /// Session open → the backend's `causal_open` returned (waiting for the
    /// executor, prompt encode, cache setup).
    pub load_ms: Option<f64>,
    /// Backend open → the first block delivered.
    pub first_block_ms: Option<f64>,
    /// First block → the first frame on the wire (reported by the transport,
    /// [`CausalControl::mark_first_frame_sent`]).
    pub transport_ms: Option<f64>,
    /// Session open → first frame on the wire.
    pub total_ms: Option<f64>,
}

/// The Reactor causal command set (design §5.7), protocol-agnostic.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", content = "data", rename_all = "snake_case")]
pub enum CausalCommand {
    SetPrompt { prompt: String },
    SetPaused { paused: bool },
    SetSeed { seed: u64 },
    Reset,
    GetState,
}

impl CausalCommand {
    /// The wire command name.
    pub fn name(&self) -> &'static str {
        match self {
            CausalCommand::SetPrompt { .. } => "set_prompt",
            CausalCommand::SetPaused { .. } => "set_paused",
            CausalCommand::SetSeed { .. } => "set_seed",
            CausalCommand::Reset => "reset",
            CausalCommand::GetState => "get_state",
        }
    }
}

/// `state_update` of a causal session.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct CausalState {
    pub prompt: String,
    pub paused: bool,
    pub seed: u64,
    pub block_index: u64,
    pub unique_fps: f64,
}

/// The answer to a [`CausalCommand`]: the new state (broadcast it as
/// `state_update`) or a refusal (broadcast `command_error`, reply bodyless).
#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(tag = "type", content = "data", rename_all = "snake_case")]
pub enum CausalReply {
    StateUpdate(CausalState),
    CommandError { command: String, reason: String },
}

/// Longest accepted prompt (the fast-h3 `enqueue` limit, reused).
pub const MAX_CAUSAL_PROMPT_CHARS: usize = 800;

#[derive(Debug)]
struct Ctl {
    prompt: Option<String>,
    prompt_version: u64,
    seed: u64,
    paused: bool,
    reset_pending: bool,
    closed: bool,
    opened: bool,
    next_block: u64,
    stats: CausalStats,
    opened_at: Option<Instant>,
    first_block_at: Option<Instant>,
    first_frame_at: Option<Instant>,
}

/// State shared by the session handle and the executor.
#[derive(Debug)]
pub(crate) struct CausalShared {
    pub id: SessionId,
    pub cancel: CancelToken,
    created: Instant,
    spec: CausalSpec,
    ctl: Mutex<Ctl>,
    cv: Condvar,
    /// Dropped on close so the consumer's `next_block` ends.
    tx: Mutex<Option<mpsc::Sender<Result<CausalBlock, ApiError>>>>,
}

impl CausalShared {
    pub fn is_closed(&self) -> bool {
        lock(&self.ctl).closed
    }

    pub fn is_opened(&self) -> bool {
        lock(&self.ctl).opened
    }

    pub fn set_opened(&self) {
        let mut c = lock(&self.ctl);
        c.opened = true;
        c.opened_at.get_or_insert_with(Instant::now);
    }

    pub fn backend_spec(&self) -> CausalSpec {
        let c = lock(&self.ctl);
        CausalSpec {
            prompt: c.prompt.clone().unwrap_or_default(),
            seed: c.seed,
            ..self.spec.clone()
        }
    }

    /// The next block's input, or `None` (after parking up to `park`) while
    /// paused, without a prompt, or closed.
    pub fn next_input(&self, park: Duration) -> Option<BlockInput> {
        let mut c = lock(&self.ctl);
        if c.closed {
            return None;
        }
        if c.paused || c.prompt.is_none() {
            let _ = self.cv.wait_timeout(c, park);
            return None;
        }
        let reset = std::mem::take(&mut c.reset_pending);
        if reset {
            c.next_block = 0;
        }
        Some(BlockInput {
            block_index: c.next_block,
            prompt: c.prompt.clone().unwrap_or_default(),
            prompt_version: c.prompt_version,
            seed: c.seed,
            reset,
        })
    }

    /// Hands a finished block to the consumer, blocking (interruptibly)
    /// while the channel is full.
    pub fn deliver(&self, input: BlockInput, frames: Vec<RgbFrame>, audio: Option<Pcm>, stats: BlockStats) {
        let mut c = lock(&self.ctl);
        c.next_block = input.block_index + 1;
        c.first_block_at.get_or_insert_with(Instant::now);
        c.stats.blocks += 1;
        c.stats.frames += frames.len() as u64;
        c.stats.last_block_ms = stats.block_ms;
        let mut block = Ok(CausalBlock {
            index: input.block_index,
            prompt_version: input.prompt_version,
            reset: input.reset,
            frames,
            audio,
            stats,
        });
        let Some(tx) = lock(&self.tx).clone() else {
            c.stats.dropped += 1;
            return;
        };
        loop {
            if c.closed {
                c.stats.dropped += 1;
                return;
            }
            match tx.try_send(block) {
                Ok(()) => return,
                Err(mpsc::error::TrySendError::Full(b)) => {
                    block = b;
                    c = self
                        .cv
                        .wait_timeout(c, Duration::from_millis(10))
                        .unwrap_or_else(|p| p.into_inner())
                        .0;
                }
                Err(mpsc::error::TrySendError::Closed(_)) => {
                    c.closed = true;
                    return;
                }
            }
        }
    }

    /// Reports a backend error to the consumer and closes.
    pub fn fail(&self, e: ApiError) {
        let mut c = lock(&self.ctl);
        if let Some(tx) = lock(&self.tx).take() {
            let _ = tx.try_send(Err(e));
        }
        c.closed = true;
        self.cv.notify_all();
    }

    /// Stops the loop: interrupts an in-flight block and wakes a parked or
    /// blocked executor.
    pub fn close(&self) {
        lock(&self.ctl).closed = true;
        lock(&self.tx).take();
        self.cancel.cancel();
        self.cv.notify_all();
    }

    fn update(&self, f: impl FnOnce(&mut Ctl)) {
        f(&mut lock(&self.ctl));
        self.cv.notify_all();
    }
}

fn ms(a: Instant, b: Instant) -> f64 {
    b.saturating_duration_since(a).as_secs_f64() * 1e3
}

/// A cloneable handle that steers a causal session (design §5.4, §5.7).
///
/// Every clone controls the same session. [`close`](Self::close) ends it for
/// everyone; dropping a control does not.
#[derive(Clone)]
pub struct CausalControl {
    shared: Arc<CausalShared>,
    engine: Arc<Shared>,
}

impl std::fmt::Debug for CausalControl {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CausalControl").field("id", &self.shared.id).finish()
    }
}

impl CausalControl {
    pub fn id(&self) -> SessionId {
        self.shared.id
    }

    /// Sets the prompt from the next block on; returns its version (1, 2, …).
    /// Generation starts with the first prompt.
    pub fn set_prompt(&self, prompt: impl Into<String>) -> u64 {
        let mut v = 0;
        let p = prompt.into();
        self.shared.update(|c| {
            c.prompt = Some(p);
            c.prompt_version += 1;
            v = c.prompt_version;
        });
        v
    }

    /// Pauses (or resumes) generation at the next block boundary.
    pub fn set_paused(&self, paused: bool) {
        self.shared.update(|c| c.paused = paused);
    }

    /// Seed for the following blocks (applies from the next reset).
    pub fn set_seed(&self, seed: u64) {
        self.shared.update(|c| c.seed = seed);
    }

    /// Clears the KV cache; the next block is block 0 again.
    pub fn reset(&self) {
        self.shared.update(|c| c.reset_pending = true);
    }

    pub fn stats(&self) -> CausalStats {
        let c = lock(&self.shared.ctl);
        let mut s = c.stats.clone();
        s.block_index = c.next_block;
        s
    }

    /// The `state_update` snapshot.
    pub fn state(&self) -> CausalState {
        let c = lock(&self.shared.ctl);
        CausalState {
            prompt: c.prompt.clone().unwrap_or_default(),
            paused: c.paused,
            seed: c.seed,
            block_index: c.next_block,
            unique_fps: (c.stats.unique_fps * 100.0).round() / 100.0,
        }
    }

    /// Applies one Reactor causal command. Refusals never raise: they come
    /// back as [`CausalReply::CommandError`] (fast-h3 convention).
    pub fn apply(&self, cmd: CausalCommand) -> CausalReply {
        let refuse = |reason: &str| CausalReply::CommandError {
            command: cmd.name().to_owned(),
            reason: reason.to_owned(),
        };
        if self.is_closed() {
            return refuse("The session has ended.");
        }
        match &cmd {
            CausalCommand::SetPrompt { prompt } => {
                let p = prompt.trim();
                if p.is_empty() {
                    return refuse("The prompt is empty.");
                }
                if p.chars().count() > MAX_CAUSAL_PROMPT_CHARS {
                    return refuse("The prompt is longer than 800 characters.");
                }
                self.set_prompt(p);
            }
            CausalCommand::SetPaused { paused } => self.set_paused(*paused),
            CausalCommand::SetSeed { seed } => self.set_seed(*seed),
            CausalCommand::Reset => self.reset(),
            CausalCommand::GetState => {}
        }
        CausalReply::StateUpdate(self.state())
    }

    /// The pacer's playout numbers (`unique_fps` in `state_update`).
    pub fn report_playout(&self, unique_fps: f64, effective_fps: f64) {
        let mut c = lock(&self.shared.ctl);
        c.stats.unique_fps = unique_fps;
        c.stats.effective_fps = effective_fps;
    }

    /// The transport sent its first frame (ends the TTFF `transport` phase).
    pub fn mark_first_frame_sent(&self) {
        lock(&self.shared.ctl).first_frame_at.get_or_insert_with(Instant::now);
    }

    /// TTFF so far.
    pub fn ttff(&self) -> Ttff {
        let c = lock(&self.shared.ctl);
        let t0 = self.shared.created;
        Ttff {
            load_ms: c.opened_at.map(|o| ms(t0, o)),
            first_block_ms: match (c.opened_at, c.first_block_at) {
                (Some(o), Some(b)) => Some(ms(o, b)),
                _ => None,
            },
            transport_ms: match (c.first_block_at, c.first_frame_at) {
                (Some(b), Some(f)) => Some(ms(b, f)),
                _ => None,
            },
            total_ms: c.first_frame_at.map(|f| ms(t0, f)),
        }
    }

    pub fn is_closed(&self) -> bool {
        self.shared.is_closed()
    }

    /// Ends the session and releases the lease (idempotent).
    pub fn close(&self) {
        self.shared.close();
        self.engine.close_causal_session(self.shared.id);
    }
}

/// A live causal session (design §5.4). Dropping it closes the session and
/// frees the executor for batch work.
pub struct CausalSession {
    control: CausalControl,
    rx: mpsc::Receiver<Result<CausalBlock, ApiError>>,
    spec: SessionSpec,
    caps: Option<ModelCaps>,
    executor: usize,
    closed: bool,
}

impl std::fmt::Debug for CausalSession {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CausalSession")
            .field("id", &self.control.shared.id)
            .field("model", &self.spec.model)
            .field("executor", &self.executor)
            .finish()
    }
}

impl CausalSession {
    pub(crate) fn new(
        engine: Arc<Shared>,
        id: SessionId,
        spec: SessionSpec,
    ) -> (Self, Arc<CausalShared>) {
        let (tx, rx) = mpsc::channel(engine.cfg.causal_depth.max(1));
        let seed = spec.seed.unwrap_or_else(draw_seed);
        let shared = Arc::new(CausalShared {
            id,
            cancel: CancelToken::new(),
            created: Instant::now(),
            spec: CausalSpec {
                model: spec.model.clone(),
                width: spec.canvas.0,
                height: spec.canvas.1,
                fps: spec.fps,
                seed,
                prompt: String::new(),
            },
            ctl: Mutex::new(Ctl {
                prompt: None,
                prompt_version: 0,
                seed,
                paused: false,
                reset_pending: false,
                closed: false,
                opened: false,
                next_block: 0,
                stats: CausalStats::default(),
                opened_at: None,
                first_block_at: None,
                first_frame_at: None,
            }),
            cv: Condvar::new(),
            tx: Mutex::new(Some(tx)),
        });
        (
            Self {
                control: CausalControl {
                    shared: shared.clone(),
                    engine,
                },
                rx,
                caps: None,
                spec,
                executor: 0,
                closed: false,
            },
            shared,
        )
    }

    pub fn id(&self) -> SessionId {
        self.control.shared.id
    }

    pub fn spec(&self) -> &SessionSpec {
        &self.spec
    }

    pub(crate) fn attach(&mut self, executor: usize, caps: ModelCaps) {
        self.executor = executor;
        self.caps = Some(caps);
    }

    pub fn caps(&self) -> &ModelCaps {
        // Set by `attach` before the session is handed out.
        self.caps.as_ref().expect("admitted causal session has caps")
    }

    /// The executor holding the lease.
    pub fn executor(&self) -> usize {
        self.executor
    }

    /// A control handle for this session (clone freely).
    pub fn control(&self) -> CausalControl {
        self.control.clone()
    }

    /// The next block; `None` once the session has ended. A backend error
    /// arrives once as `Some(Err(_))`, then the session ends.
    pub async fn next_block(&mut self) -> Option<Result<CausalBlock, ApiError>> {
        self.rx.recv().await
    }

    /// See [`CausalControl::set_prompt`].
    pub fn set_prompt(&self, prompt: impl Into<String>) -> u64 {
        self.control.set_prompt(prompt)
    }

    /// See [`CausalControl::set_paused`].
    pub fn set_paused(&self, paused: bool) {
        self.control.set_paused(paused);
    }

    /// See [`CausalControl::set_seed`].
    pub fn set_seed(&self, seed: u64) {
        self.control.set_seed(seed);
    }

    /// See [`CausalControl::reset`].
    pub fn reset(&self) {
        self.control.reset();
    }

    /// See [`CausalControl::apply`].
    pub fn apply(&self, cmd: CausalCommand) -> CausalReply {
        self.control.apply(cmd)
    }

    pub fn stats(&self) -> CausalStats {
        self.control.stats()
    }

    pub fn ttff(&self) -> Ttff {
        self.control.ttff()
    }

    /// Ends the session and releases the lease.
    pub fn close(mut self) {
        self.do_close();
    }

    fn do_close(&mut self) {
        if !self.closed {
            self.closed = true;
            self.control.close();
        }
    }
}

impl Drop for CausalSession {
    fn drop(&mut self) {
        self.do_close();
    }
}
