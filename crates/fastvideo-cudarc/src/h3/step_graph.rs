//! The H3 denoise loop's step timing (docs/perf/h3-denoise-graphs.md).
//!
//! **Lazy step clock.** The loop used to synchronize the device after every
//! step only to time it. That drained the queue at each step boundary: the
//! next step's host preparation (its bf16 AdaLN table, ~50 ms of host work
//! before its first block) then ran with the GPU idle. [`StepClock`] records a
//! CUDA event after each step instead and reads the events lazily: a step's
//! log line is printed once its event has completed (polled, never waited
//! for, at the next step boundary), and the device is synchronized once,
//! after the last step. Per-step seconds are then GPU time between the
//! events. No kernel, buffer or launch order changes, so the output bits do
//! not either.
//!
//! `FASTVIDEO_H3_STEP_SYNC=1` restores the per-step synchronize (and
//! host-clock step seconds). `FASTVIDEO_GPU_TRACE` implies it: the CUPTI
//! trace window of one step needs the queue drained around that step.

use std::time::Instant;

use crate::wan::pipeline::{PipelineError, Result};

fn msg(s: impl Into<String>) -> PipelineError {
    PipelineError::Message(s.into())
}

/// `FASTVIDEO_H3_STEP_SYNC=1` (default off), or a GPU trace: synchronize
/// after every denoise step.
pub fn step_sync() -> bool {
    crate::wan::envflag::bool_flag("FASTVIDEO_H3_STEP_SYNC", false)
        || crate::wan::gpu_trace::enabled()
        || !cfg!(feature = "cuda")
}

/// Per-step timing of one denoise ([`step_sync`] or lazy CUDA events).
pub struct StepClock {
    steps: usize,
    sync: bool,
    last: Instant,
    step_s: Vec<f64>,
    /// Lazy mode: the start event, then one per queued step.
    #[cfg(feature = "cuda")]
    events: Vec<cudarc::driver::CudaEvent>,
}

impl StepClock {
    /// Before the first step is queued.
    pub fn start(steps: usize) -> Result<Self> {
        let sync = step_sync();
        let mut clock = Self {
            steps,
            sync,
            last: Instant::now(),
            step_s: Vec::with_capacity(steps),
            #[cfg(feature = "cuda")]
            events: Vec::with_capacity(steps + 1),
        };
        if sync {
            // FASTVIDEO_GPU_TRACE: one step's device activity (a no-op when off).
            crate::wan::gpu_trace::pass_begin("h3");
            crate::wan::gpu_trace::step_begin();
        } else {
            clock.record()?;
        }
        Ok(clock)
    }

    /// Whether this clock synchronizes per step.
    pub fn is_sync(&self) -> bool {
        self.sync
    }

    #[cfg(feature = "cuda")]
    fn record(&mut self) -> Result<()> {
        let dev = crate::wan::device::global_device()
            .ok_or_else(|| msg("h3 step clock: no CUDA device"))?;
        let event = dev
            .stream
            .record_event(Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT))
            .map_err(|e| msg(format!("h3 step clock: {e}")))?;
        self.events.push(event);
        Ok(())
    }

    #[cfg(not(feature = "cuda"))]
    fn record(&mut self) -> Result<()> {
        Ok(())
    }

    /// Step `step`'s work is queued (on this thread's device stream).
    pub fn step_queued(&mut self, step: usize) -> Result<()> {
        if self.sync {
            // The trace window closes before the step-timing sync.
            crate::wan::gpu_trace::step_before_sync();
            crate::wan::device::synchronize().map_err(|e| msg(e.to_string()))?;
            let s = self.last.elapsed().as_secs_f64();
            self.step_s.push(s);
            crate::wan::gpu_trace::step_end(s);
            self.log(step, s);
            if step + 1 < self.steps {
                crate::wan::gpu_trace::step_begin();
            }
            self.last = Instant::now();
            return Ok(());
        }
        self.record()?;
        self.collect(false)
    }

    fn log(&self, step: usize, s: f64) {
        crate::wan::log::info(format_args!("h3 step {}/{}: {s:.1}s", step + 1, self.steps));
    }

    /// Lazy mode: time and log every step whose event has completed (all of
    /// them with `wait`, after a synchronize).
    #[cfg(feature = "cuda")]
    fn collect(&mut self, wait: bool) -> Result<()> {
        while self.step_s.len() + 1 < self.events.len() {
            let i = self.step_s.len();
            if !wait && !self.events[i + 1].is_complete() {
                break;
            }
            let ms = self.events[i]
                .elapsed_ms(&self.events[i + 1])
                .map_err(|e| msg(format!("h3 step clock: {e}")))?;
            let s = f64::from(ms) * 1e-3;
            self.step_s.push(s);
            self.log(i, s);
        }
        Ok(())
    }

    #[cfg(not(feature = "cuda"))]
    fn collect(&mut self, _wait: bool) -> Result<()> {
        Ok(())
    }

    /// After the last step is queued: wait for the device (lazy mode) and
    /// return the per-step seconds.
    pub fn finish(mut self) -> Result<Vec<f64>> {
        if !self.sync {
            crate::wan::device::synchronize().map_err(|e| msg(e.to_string()))?;
            self.collect(true)?;
        }
        Ok(self.step_s)
    }
}
