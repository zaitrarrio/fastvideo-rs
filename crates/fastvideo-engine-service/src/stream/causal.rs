//! `CausalSession`: SF-Wan block rollout under an exclusive executor lease
//! (design §5.4).
//!
//! WP-02 provides the engine side: the lease, the per-turn block loop on the
//! executor, prompt/seed/pause/reset controls applied at block boundaries,
//! and a bounded block channel (depth `EngineConfig::causal_depth`, 4) on
//! which the executor blocks when the consumer falls behind. WP-15 builds the
//! pacer, the Reactor causal command set and the CUDA block backend on top.

use std::sync::{Arc, Condvar, Mutex};
use std::time::Duration;

use fastvideo_protocol::{
    draw_seed, ApiError, ModelCaps, Pcm, RgbFrame, SessionSpec,
};
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
}

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
}

/// State shared by the session handle and the executor.
#[derive(Debug)]
pub(crate) struct CausalShared {
    pub id: SessionId,
    pub cancel: CancelToken,
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
        lock(&self.ctl).opened = true;
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

/// A live causal session (design §5.4). Dropping it closes the session and
/// frees the executor for batch work.
pub struct CausalSession {
    shared: Arc<CausalShared>,
    engine: Arc<Shared>,
    rx: mpsc::Receiver<Result<CausalBlock, ApiError>>,
    spec: SessionSpec,
    caps: Option<ModelCaps>,
    executor: usize,
    closed: bool,
}

impl std::fmt::Debug for CausalSession {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CausalSession")
            .field("id", &self.shared.id)
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
            }),
            cv: Condvar::new(),
            tx: Mutex::new(Some(tx)),
        });
        (
            Self {
                shared: shared.clone(),
                engine,
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
        self.shared.id
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

    /// The next block; `None` once the session has ended. A backend error
    /// arrives once as `Some(Err(_))`, then the session ends.
    pub async fn next_block(&mut self) -> Option<Result<CausalBlock, ApiError>> {
        self.rx.recv().await
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

    /// Seed for the following blocks.
    pub fn set_seed(&self, seed: u64) {
        self.shared.update(|c| c.seed = seed);
    }

    /// Clears the KV cache; the next block is block 0 again.
    pub fn reset(&self) {
        self.shared.update(|c| c.reset_pending = true);
    }

    pub fn stats(&self) -> CausalStats {
        lock(&self.shared.ctl).stats.clone()
    }

    /// Ends the session and releases the lease.
    pub fn close(mut self) {
        self.do_close();
    }

    fn do_close(&mut self) {
        if !self.closed {
            self.closed = true;
            self.shared.close();
            self.engine.close_causal_session(self.shared.id);
        }
    }
}

impl Drop for CausalSession {
    fn drop(&mut self) {
        self.do_close();
    }
}
