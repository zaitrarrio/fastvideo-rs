//! Preallocated timing events on the compute stream, for request tracing
//! (docs/serve/tracing.md "GPU timing").
//!
//! [`EventMarks::new`] creates every event up front (before a run). During
//! the run [`EventMarks::record`] only enqueues `cuEventRecord` on the
//! global device's stream: no allocation, no synchronisation, the host
//! never waits. After the run (the host has already waited for its last
//! frames, so every event is complete) [`EventMarks::elapsed_ns`] reads
//! event-to-event times, on whichever thread resolves them (the trace
//! drain). Without a device (CPU builds) nothing is recorded.

/// `n` timing events on the global device's compute stream.
pub struct EventMarks {
    #[cfg(feature = "cuda")]
    inner: Option<(
        std::sync::Arc<cudarc::driver::CudaStream>,
        Vec<cudarc::driver::CudaEvent>,
    )>,
}

impl EventMarks {
    /// `None` without a device or when the events cannot be created.
    pub fn new(n: usize) -> Option<Self> {
        #[cfg(feature = "cuda")]
        {
            let dev = crate::wan::device::global_device()?;
            let mut events = Vec::with_capacity(n);
            for _ in 0..n {
                events.push(
                    dev.ctx
                        .new_event(Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT))
                        .ok()?,
                );
            }
            Some(Self {
                inner: Some((dev.stream.clone(), events)),
            })
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = n;
            None
        }
    }

    pub fn len(&self) -> usize {
        #[cfg(feature = "cuda")]
        {
            self.inner.as_ref().map_or(0, |(_, e)| e.len())
        }
        #[cfg(not(feature = "cuda"))]
        {
            0
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Enqueues mark `i` on the compute stream (asynchronous).
    pub fn record(&self, i: usize) -> bool {
        #[cfg(feature = "cuda")]
        {
            match &self.inner {
                Some((stream, events)) => events.get(i).is_some_and(|e| e.record(stream).is_ok()),
                None => false,
            }
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = i;
            false
        }
    }

    /// Device time from mark `from` to mark `to`, ns. Waits for both
    /// events: call after the run.
    pub fn elapsed_ns(&self, from: usize, to: usize) -> Option<u64> {
        #[cfg(feature = "cuda")]
        {
            let (_, events) = self.inner.as_ref()?;
            let ms = events.get(from)?.elapsed_ms(events.get(to)?).ok()?;
            Some((f64::from(ms.max(0.0)) * 1e6) as u64)
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = (from, to);
            None
        }
    }
}
