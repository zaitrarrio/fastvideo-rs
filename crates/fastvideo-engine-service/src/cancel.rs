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
use std::time::{Duration, Instant};

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

/// A running job's wall-clock deadline (`[engine] job_deadline_s`).
///
/// Cooperative, like cancellation: [`StepControl::step`] and
/// [`StepControl::check`] compare the clock with it, and the first call past
/// it trips the job's [`CancelToken`] (so a pipeline that only polls the
/// token stops too) and returns `Err(Timeout)`. The executor turns the
/// job's result into `Failed(Timeout)` once [`Deadline::tripped`] holds,
/// whatever error the backend unwound with. A kernel that never reaches a
/// step is not caught here (the serve-side stale-job sweep is).
#[derive(Clone)]
pub struct Deadline {
    started: Instant,
    limit: Duration,
    tripped: Arc<AtomicBool>,
}

impl fmt::Debug for Deadline {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Deadline")
            .field("limit", &self.limit)
            .field("elapsed", &self.started.elapsed())
            .field("tripped", &self.tripped())
            .finish()
    }
}

impl Deadline {
    /// `limit` of wall clock from now.
    pub fn after(limit: Duration) -> Self {
        Self {
            started: Instant::now(),
            limit,
            tripped: Arc::new(AtomicBool::new(false)),
        }
    }

    pub fn limit(&self) -> Duration {
        self.limit
    }

    /// Whether the deadline has passed; records the first time it is seen
    /// passed (every later call reports `true` too).
    pub fn expired(&self) -> bool {
        if self.tripped() {
            return true;
        }
        if self.started.elapsed() >= self.limit {
            self.tripped.store(true, Ordering::SeqCst);
            return true;
        }
        false
    }

    /// Whether a check already saw the deadline passed.
    pub fn tripped(&self) -> bool {
        self.tripped.load(Ordering::SeqCst)
    }

    /// The `Timeout` error for this deadline.
    pub fn error(&self) -> ApiError {
        ApiError::timeout(format!(
            "job exceeded its deadline of {}s ({:.1}s elapsed): cancelled at the next step",
            self.limit.as_secs_f64(),
            self.started.elapsed().as_secs_f64()
        ))
    }
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
    /// A traced run's stage/step marks (docs/serve/tracing.md).
    timeline: Option<Mutex<fastvideo_trace::Timeline>>,
    /// The job's wall-clock deadline, enforced at every step and check.
    deadline: Option<Deadline>,
}

impl fmt::Debug for StepControl {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("StepControl")
            .field("cancel", &self.cancel)
            .field("mode", &self.mode)
            .field("deadline", &self.deadline)
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
            timeline: None,
            deadline: None,
        }
    }

    /// Marks every stage and step of this run on `t` (a traced job).
    pub fn with_timeline(mut self, t: fastvideo_trace::Timeline) -> Self {
        self.timeline = Some(Mutex::new(t));
        self
    }

    /// Fails the run with `Timeout` at the first step or check past `d`
    /// (tripping the cancel token at the same time).
    pub fn with_deadline(mut self, d: Deadline) -> Self {
        self.deadline = Some(d);
        self
    }

    /// The run's deadline, if it has one.
    pub fn deadline(&self) -> Option<&Deadline> {
        self.deadline.as_ref()
    }

    /// The deadline first (so a job past it fails with `Timeout`, not
    /// `Cancelled`), then the token.
    fn enforce(&self) -> Result<(), ApiError> {
        if let Some(d) = &self.deadline {
            if d.expired() {
                self.cancel.cancel();
                return Err(d.error());
            }
        }
        self.cancel.check()
    }

    /// Whether this run is traced.
    pub fn traced(&self) -> bool {
        self.timeline.is_some()
    }

    /// This run's trace, if it is traced.
    pub fn trace(&self) -> Option<fastvideo_trace::Trace> {
        self.timeline.as_ref().map(|t| lock(t).trace())
    }

    #[inline]
    fn timeline(&self, f: impl FnOnce(&mut fastvideo_trace::Timeline)) {
        if let Some(t) = &self.timeline {
            f(&mut lock(t));
        }
    }

    /// A named boundary on the run's timeline only (no engine event): e.g.
    /// `encode` after the last decoded frame.
    pub fn mark(&self, name: &'static str) {
        self.timeline(|t| t.mark(name));
    }

    /// Ends the run's timeline and hands it to the trace drain.
    pub fn finish_timeline(&mut self) {
        if let Some(t) = self.timeline.take() {
            t.into_inner().unwrap_or_else(|p| p.into_inner()).finish();
        }
    }

    /// A control that reports nothing (tests, warmup).
    pub fn detached(cancel: CancelToken, mode: OutputMode) -> Self {
        Self::new(cancel, mode, |_| {})
    }

    /// Reports step `step/total`, then fails with `Timeout` past the
    /// deadline or `Cancelled` if the token is tripped.
    pub fn step(&self, step: u32, total: u32) -> Result<(), ApiError> {
        self.timeline(|t| t.step(step, total));
        (self.emit)(StepEvent::Progress { step, total });
        self.enforce()
    }

    pub fn stage(&self, name: &'static str) {
        self.timeline(|t| t.stage(name));
        (self.emit)(StepEvent::Stage(name));
    }

    pub fn log(&self, message: impl Into<String>) {
        (self.emit)(StepEvent::Log(LogLine::info(
            message,
            OffsetDateTime::now_utc(),
        )));
    }

    /// `Err(Timeout)` past the deadline, `Err(Cancelled)` once tripped,
    /// without reporting progress.
    pub fn check(&self) -> Result<(), ApiError> {
        self.enforce()
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
        self.cancel.is_cancelled() || self.deadline.as_ref().is_some_and(Deadline::expired)
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

    #[test]
    fn deadline_trips_the_token_and_fails_with_timeout() {
        let t = CancelToken::new();
        let ctl = StepControl::detached(t.clone(), OutputMode::Frames)
            .with_deadline(Deadline::after(Duration::ZERO));
        assert!(!t.is_cancelled());
        let err = ctl.step(1, 4).unwrap_err();
        assert_eq!(err.kind, ErrorKind::Timeout, "{err}");
        assert!(err.message.contains("deadline of 0s"), "{}", err.message);
        assert!(
            t.is_cancelled(),
            "the token is tripped for pipelines that only poll it"
        );
        assert!(ctl.deadline().unwrap().tripped());
        assert!(StepHook::cancelled(&ctl));
        // Still Timeout, not Cancelled, on every later check.
        assert_eq!(ctl.check().unwrap_err().kind, ErrorKind::Timeout);
    }

    #[test]
    fn far_deadline_does_not_interfere() {
        let t = CancelToken::new();
        let ctl = StepControl::detached(t.clone(), OutputMode::Frames)
            .with_deadline(Deadline::after(Duration::from_secs(3600)));
        assert!(ctl.step(1, 2).is_ok());
        assert!(ctl.check().is_ok());
        assert!(!ctl.deadline().unwrap().tripped());
        t.cancel();
        assert!(
            is_cancel(&ctl.check().unwrap_err()),
            "a user cancel stays Cancelled"
        );
    }
}
