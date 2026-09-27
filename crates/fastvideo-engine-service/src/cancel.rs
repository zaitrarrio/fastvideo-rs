//! `CancelToken`, `StepControl` and the step-hook seam for the CUDA
//! pipelines (design §3.6; package E1 wires the real observers).
//!
//! A [`CancelToken`] is shared by the front-end (DELETE / cancel routes, the
//! `JobHandle`) and the executor thread. Cancelling a **queued** job removes
//! it from the scheduler at once (through an on-cancel callback the service
//! registers); cancelling a **running** job trips the flag, and the backend
//! observes it at its next denoise step through [`StepControl::step`], which
//! returns `Err(Cancelled)` so the pipeline unwinds.
//!
//! [`StepHook`] is the engine-side trait the CUDA backend (WP-11) adapts to the
//! pipeline observers:
//!
//! - Wan `StepObserver = dyn FnMut(&DenoiseStep) -> Result<()>`;
//! - LTX `StepObserver = &mut dyn FnMut(usize, &CudaTensor, &CudaTensor, f64) -> Result<()>`;
//! - H3: the observer E1 adds.
//!
//! Each adapter closure calls `hook.on_step(step, total)` and turns an `Err`
//! into the pipeline's error type; the backend maps that error back to
//! [`ApiError`] of kind `Cancelled` (see [`is_cancel`]).

use std::fmt;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use fastvideo_protocol::{ApiError, ErrorKind, LogLine};
use time::OffsetDateTime;

type Callback = Box<dyn FnOnce() + Send>;

struct Inner {
    flag: AtomicBool,
    callbacks: Mutex<Vec<Callback>>,
    notify: tokio::sync::Notify,
}

/// Cooperative cancellation shared between front-end and executor. Cheap to
/// clone; every clone observes the same flag.
#[derive(Clone)]
pub struct CancelToken(Arc<Inner>);

impl Default for CancelToken {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Debug for CancelToken {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CancelToken")
            .field("cancelled", &self.is_cancelled())
            .finish()
    }
}

impl CancelToken {
    pub fn new() -> Self {
        Self(Arc::new(Inner {
            flag: AtomicBool::new(false),
            callbacks: Mutex::new(Vec::new()),
            notify: tokio::sync::Notify::new(),
        }))
    }

    /// Trips the token. Idempotent: callbacks run once, on the first call,
    /// on the calling thread. Never call it while holding a lock a callback
    /// might take.
    pub fn cancel(&self) {
        if self.0.flag.swap(true, Ordering::SeqCst) {
            return;
        }
        let cbs = std::mem::take(&mut *lock(&self.0.callbacks));
        for cb in cbs {
            cb();
        }
        self.0.notify.notify_waiters();
    }

    pub fn is_cancelled(&self) -> bool {
        self.0.flag.load(Ordering::SeqCst)
    }

    /// `Err(Cancelled)` once tripped.
    pub fn check(&self) -> Result<(), ApiError> {
        if self.is_cancelled() {
            Err(cancelled_error())
        } else {
            Ok(())
        }
    }

    /// Runs `f` when the token is cancelled (at once if it already is).
    pub fn on_cancel(&self, f: impl FnOnce() + Send + 'static) {
        {
            let mut cbs = lock(&self.0.callbacks);
            if !self.is_cancelled() {
                cbs.push(Box::new(f));
                return;
            }
        }
        f();
    }

    /// Resolves once the token is cancelled.
    pub async fn cancelled(&self) {
        loop {
            let n = self.0.notify.notified();
            if self.is_cancelled() {
                return;
            }
            n.await;
        }
    }

    /// Whether two handles share one token.
    pub fn same_as(&self, other: &CancelToken) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}

/// The error a cancelled step returns.
pub fn cancelled_error() -> ApiError {
    ApiError::cancelled("cancelled")
}

/// Whether `e` is a cancellation (as opposed to a failure).
pub fn is_cancel(e: &ApiError) -> bool {
    e.kind == ErrorKind::Cancelled
}

/// Per-step callbacks a pipeline observer drives (the E1 seam). Implemented
/// by [`StepControl`]; the CUDA backend adapts it to each pipeline's
/// observer signature.
pub trait StepHook {
    /// Called after denoise step `step` of `total` (1-based). `Err` means
    /// stop now: the observer must return an error so the pipeline unwinds.
    fn on_step(&self, step: u32, total: u32) -> Result<(), ApiError>;
    /// Entering a named stage (`text_encode`, `denoise`, `decode`, `mux`, …).
    fn on_stage(&self, name: &'static str);
    /// Whether the job has been cancelled (for loops between steps).
    fn cancelled(&self) -> bool;
}

/// What a step reports upward (to the job's `EngineEvent` stream).
#[derive(Clone, Debug, PartialEq)]
pub enum StepEvent {
    Stage(&'static str),
    Progress { step: u32, total: u32 },
    Log(LogLine),
}

/// Where a generation's output goes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum OutputMode {
    /// Batch: the backend writes an MP4 under this directory
    /// (`ClipOutput::mp4`); frames pushed to the sink are dropped.
    File { dir: std::path::PathBuf },
    /// Streaming builds: frames and PCM are collected in memory
    /// (`ClipOutput::frames` / `audio`); no file.
    Frames,
}

type Emit = Arc<dyn Fn(StepEvent) + Send + Sync>;

/// Handed to `EngineBackend::generate` / `causal_block`.
///
/// **Deviation from design §3.6:** the design lists
/// `progress: Box<dyn FnMut(u32, u32) + Send>` next to `cancel`, but
/// `generate` takes `&StepControl`, through which an `FnMut` cannot be
/// called. Progress, stages and logs therefore go through `&self` methods
/// backed by a shared `Fn` emitter, and the output mode travels here too.
pub struct StepControl {
    pub cancel: CancelToken,
    pub mode: OutputMode,
    emit: Emit,
}

impl fmt::Debug for StepControl {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("StepControl")
            .field("cancel", &self.cancel)
            .field("mode", &self.mode)
            .finish()
    }
}

impl StepControl {
    pub fn new(
        cancel: CancelToken,
        mode: OutputMode,
        emit: impl Fn(StepEvent) + Send + Sync + 'static,
    ) -> Self {
        Self {
            cancel,
            mode,
            emit: Arc::new(emit),
        }
    }

    /// A control that reports nothing (tests, warmup).
    pub fn detached(cancel: CancelToken, mode: OutputMode) -> Self {
        Self::new(cancel, mode, |_| {})
    }

    /// Reports step `step/total`, then fails with `Cancelled` if the token
    /// is tripped.
    pub fn step(&self, step: u32, total: u32) -> Result<(), ApiError> {
        (self.emit)(StepEvent::Progress { step, total });
        self.cancel.check()
    }

    pub fn stage(&self, name: &'static str) {
        (self.emit)(StepEvent::Stage(name));
    }

    pub fn log(&self, message: impl Into<String>) {
        (self.emit)(StepEvent::Log(LogLine::info(
            message,
            OffsetDateTime::now_utc(),
        )));
    }

    /// `Err(Cancelled)` once tripped, without reporting progress.
    pub fn check(&self) -> Result<(), ApiError> {
        self.cancel.check()
    }

    /// A `FnMut(step, total)` observer (0-based `usize` steps, as the cudarc
    /// observers count them) for plugging straight into a pipeline.
    pub fn observer(&self) -> impl FnMut(usize, usize) -> Result<(), ApiError> + '_ {
        move |step, total| self.step(step as u32 + 1, total as u32)
    }
}

impl StepHook for StepControl {
    fn on_step(&self, step: u32, total: u32) -> Result<(), ApiError> {
        self.step(step, total)
    }
    fn on_stage(&self, name: &'static str) {
        self.stage(name)
    }
    fn cancelled(&self) -> bool {
        self.cancel.is_cancelled()
    }
}

pub(crate) fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|p| p.into_inner())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicU32;

    #[test]
    fn callbacks_run_once() {
        let t = CancelToken::new();
        let n = Arc::new(AtomicU32::new(0));
        let n2 = n.clone();
        t.on_cancel(move || {
            n2.fetch_add(1, Ordering::SeqCst);
        });
        assert!(t.check().is_ok());
        t.cancel();
        t.cancel();
        assert_eq!(n.load(Ordering::SeqCst), 1);
        assert!(is_cancel(&t.check().unwrap_err()));
        // Registered after the fact: runs at once.
        let n3 = n.clone();
        t.on_cancel(move || {
            n3.fetch_add(1, Ordering::SeqCst);
        });
        assert_eq!(n.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn step_control_reports_then_checks() {
        let t = CancelToken::new();
        let seen = Arc::new(Mutex::new(Vec::new()));
        let s2 = seen.clone();
        let ctl = StepControl::new(t.clone(), OutputMode::Frames, move |e| {
            lock(&s2).push(e);
        });
        ctl.stage("denoise");
        assert!(ctl.step(1, 4).is_ok());
        t.cancel();
        let err = ctl.step(2, 4).unwrap_err();
        assert!(is_cancel(&err));
        let mut obs = ctl.observer();
        assert!(obs(2, 4).is_err());
        let seen = lock(&seen).clone();
        assert_eq!(seen[0], StepEvent::Stage("denoise"));
        assert_eq!(seen[1], StepEvent::Progress { step: 1, total: 4 });
        assert_eq!(seen[2], StepEvent::Progress { step: 2, total: 4 });
        assert_eq!(seen[3], StepEvent::Progress { step: 3, total: 4 });
    }
}
