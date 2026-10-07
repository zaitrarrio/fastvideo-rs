//! End-to-end request tracing (docs/serve/tracing.md).
//!
//! One trace id per request, from the click (or the bench client) through
//! the edge, the pod's HTTP front, the protocol adapter, the job queue, the
//! engine and its GPU stages, the post encoder, the upload and the job
//! store, to the answer the client polls.
//!
//! **The hot path only reads a clock and pushes a fixed-size [`Rec`]** into
//! a bounded, lock-free channel ([`Recorder::emit`]): no formatting, no
//! allocation, no I/O, no lock, no device synchronisation. When the channel
//! is full the record is dropped and counted; nothing ever blocks. A
//! background thread (started by the first traced request, so an untraced
//! process never starts it) drains the channel, turns monotonic times into
//! wall-clock times, keeps the last traces in memory for
//! `GET /fv/v1/traces/{id}`, writes JSON lines to `FV_TRACE_FILE` and hands
//! each event to a sink (fv-serve's log shipper).
//!
//! GPU work is timed with device events recorded on the compute stream
//! ([`Timeline`] over a [`MarkPool`]) and resolved on the drain thread after
//! the work is done ([`Recorder::defer`]); recording a mark never waits.
//!
//! Tracing is opt-in per request ([`policy`]): untraced requests carry
//! `None` and every call site is behind an `if let Some(trace)`.

mod id;
pub mod policy;
mod recorder;
mod timeline;

pub use id::{new_span_id, Span, Trace, TraceId};
pub use policy::{decide, mode, set_mode, Mode, OPT_IN_HEADER, TIME_HEADER, TRACEPARENT};
pub use recorder::{
    global, ingest, now_ns, recent, set_host, set_sink, snapshot, started, wall_ns, Clock, Comp,
    Event, Parked, Rec, Recorder, Stats, TraceDump,
};
pub use timeline::{HostMarks, MarkPool, Timeline};

tokio::task_local! {
    static CURRENT: Trace;
}

/// The trace of the request this task serves (set by the HTTP layer with
/// [`scope`]); `None` for untraced requests and outside a request.
pub fn current() -> Option<Trace> {
    CURRENT.try_with(|t| *t).ok()
}

/// Runs `f` with `t` as [`current`].
pub async fn scope<F: std::future::Future>(t: Trace, f: F) -> F::Output {
    CURRENT.scope(t, f).await
}

/// `t`'s `traceparent` for a job record (`Job::trace`), from [`current`].
pub fn current_traceparent() -> Option<String> {
    current().map(|t| t.traceparent(t.parent))
}
