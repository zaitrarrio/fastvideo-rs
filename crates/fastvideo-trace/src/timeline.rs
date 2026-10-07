//! Stage and step timing of one generation, on the host and on the device.
//!
//! The engine marks every stage boundary and every finished denoise step
//! ([`Timeline::stage`], [`Timeline::step`]). A mark reads the host clock
//! and records a device mark ([`MarkPool::record`], a CUDA event on the
//! compute stream: asynchronous, no synchronisation) into storage allocated
//! before the run. [`Timeline::finish`] hands the marks to the drain thread,
//! which waits for nothing (the run is over: the events are complete) and
//! turns them into one host span and one device span per segment.
//!
//! **Placing device spans on the host clock.** Device times are relative
//! (event to event). Mark *i* cannot complete before the host enqueued it,
//! so the device timeline starts at `h0 + lag` with
//! `lag = max(0, max_i(h_i − h0 − g_i))` (`h_i`: host time of mark *i*,
//! `g_i`: device time from mark 0 to mark *i*): the earliest placement that
//! is consistent with every mark. The end mark is taken after the host has
//! waited for the device, so it pins `lag` to within that wait's latency.

use crate::id::Trace;
use crate::recorder::{global, now_ns, Clock, Comp, Rec};

/// Preallocated device marks (CUDA events on the compute stream; the fake
/// engine's are scripted).
pub trait MarkPool: Send {
    /// How many marks it holds.
    fn capacity(&self) -> usize;
    /// Records mark `i` on the device's work queue. Must not wait or allocate.
    fn record(&mut self, i: usize) -> bool;
    /// Device time from mark `from` to mark `to`, ns, once both completed.
    /// Runs on the drain thread after the run.
    fn elapsed_ns(&self, from: usize, to: usize) -> Option<u64>;
}

/// A mark pool without a device: no device spans (host spans only).
pub struct HostMarks;

impl MarkPool for HostMarks {
    fn capacity(&self) -> usize {
        usize::MAX
    }
    fn record(&mut self, _: usize) -> bool {
        false
    }
    fn elapsed_ns(&self, _: usize, _: usize) -> Option<u64> {
        None
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    /// A stage begins (`text`, `denoise`, `video_decode`, …).
    Stage,
    /// Step `arg` of the current stage begins (the previous one finished).
    Step,
    /// The last step of the stage finished; what follows until the next stage.
    Tail,
    /// A named boundary inside the run (`encode`).
    Mark,
    /// The run is over.
    End,
}

#[derive(Clone, Copy, Debug)]
struct Label {
    kind: Kind,
    stage: &'static str,
    arg: i64,
    host_ns: u64,
    /// Whether the device mark was recorded.
    device: bool,
}

/// One generation's marks. Every method is allocation-free; the label
/// storage is reserved by [`Timeline::new`].
pub struct Timeline {
    trace: Trace,
    pool: Box<dyn MarkPool>,
    labels: Vec<Label>,
    cap: usize,
    stage: &'static str,
    overflow: u32,
}

impl std::fmt::Debug for Timeline {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Timeline").field("trace", &self.trace.id).field("marks", &self.labels.len()).finish()
    }
}

impl Timeline {
    /// Room for `cap` marks (stage boundaries + steps + a few).
    pub fn new(trace: Trace, pool: Box<dyn MarkPool>, cap: usize) -> Self {
        let cap = cap.min(pool.capacity()).max(2);
        Self { trace, pool, labels: Vec::with_capacity(cap), cap, stage: "", overflow: 0 }
    }

    fn push(&mut self, kind: Kind, stage: &'static str, arg: i64) {
        // Keep the last slot for the end mark.
        let room = if kind == Kind::End { self.cap } else { self.cap - 1 };
        if self.labels.len() >= room {
            self.overflow += 1;
            return;
        }
        let host_ns = now_ns();
        let device = self.pool.record(self.labels.len());
        self.labels.push(Label { kind, stage, arg, host_ns, device });
    }

    /// The run's trace.
    pub fn trace(&self) -> Trace {
        self.trace
    }

    /// Stage `name` begins.
    pub fn stage(&mut self, name: &'static str) {
        self.stage = name;
        self.push(Kind::Stage, name, 0);
    }

    /// `done` of `total` steps of the current stage are finished.
    pub fn step(&mut self, done: u32, total: u32) {
        if done < total {
            self.push(Kind::Step, self.stage, i64::from(done) + 1);
        } else {
            self.push(Kind::Tail, self.stage, 0);
        }
    }

    /// A named boundary (a segment `name` begins).
    pub fn mark(&mut self, name: &'static str) {
        self.push(Kind::Mark, name, 0);
    }

    /// Ends the run and resolves the marks on the drain thread.
    pub fn finish(mut self) {
        self.push(Kind::End, "", 0);
        global().defer(move || self.resolve());
    }

    /// The segments as records (host spans, device spans when the pool has
    /// them, an overflow count). Pure; [`Timeline::finish`] runs it off the
    /// hot path.
    pub fn resolve(self) -> Vec<Rec> {
        let n = self.labels.len();
        let mut out = Vec::with_capacity(2 * n + 1);
        let id = self.trace.id;
        let name_of = |i: usize| -> (&'static str, &'static str, i64) {
            let l = self.labels[i];
            match l.kind {
                // A stage whose next mark is its step 2 is its step 1.
                Kind::Stage if self.labels.get(i + 1).is_some_and(|x| x.kind == Kind::Step && x.stage == l.stage && x.arg == 2) => (l.stage, "step", 1),
                Kind::Stage | Kind::Mark => ("", l.stage, 0),
                Kind::Step => (l.stage, "step", l.arg),
                Kind::Tail => (l.stage, "tail", 0),
                Kind::End => ("", "end", 0),
            }
        };
        for i in 0..n.saturating_sub(1) {
            let (stage, name, arg) = name_of(i);
            let (a, b) = (self.labels[i].host_ns, self.labels[i + 1].host_ns);
            out.push(Rec { trace: id, comp: Comp::Engine, clock: Clock::Host, name, stage, t_ns: a, dur_ns: b.saturating_sub(a), arg });
        }
        // Device spans: every mark recorded, elapsed times available.
        let g: Option<Vec<u64>> = if n >= 2 && self.labels.iter().all(|l| l.device) {
            (0..n).map(|i| if i == 0 { Some(0) } else { self.pool.elapsed_ns(0, i) }).collect()
        } else {
            None
        };
        if let Some(g) = g {
            let h0 = self.labels[0].host_ns;
            let lag = (0..n).map(|i| self.labels[i].host_ns.saturating_sub(h0).saturating_sub(g[i])).max().unwrap_or(0);
            for i in 0..n - 1 {
                let (stage, name, arg) = name_of(i);
                out.push(Rec {
                    trace: id,
                    comp: Comp::Gpu,
                    clock: Clock::Device,
                    name,
                    stage,
                    t_ns: h0 + lag + g[i],
                    dur_ns: g[i + 1].saturating_sub(g[i]),
                    arg,
                });
            }
        }
        if self.overflow > 0 {
            let t = self.labels.last().map_or(0, |l| l.host_ns);
            out.push(Rec::point(id, Comp::Engine, "timeline.overflow", t, i64::from(self.overflow)));
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Device time = 2 ms per mark index, recorded calls counted.
    struct Scripted {
        recorded: std::sync::Arc<std::sync::atomic::AtomicUsize>,
    }
    impl MarkPool for Scripted {
        fn capacity(&self) -> usize {
            64
        }
        fn record(&mut self, _: usize) -> bool {
            self.recorded.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            true
        }
        fn elapsed_ns(&self, from: usize, to: usize) -> Option<u64> {
            Some((to - from) as u64 * 2_000_000)
        }
    }

    #[test]
    fn segments_are_named_and_placed() {
        let recorded = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let mut t = Timeline::new(Trace::new_root(), Box::new(Scripted { recorded: recorded.clone() }), 16);
        t.stage("text");
        t.stage("denoise");
        for k in 1..=3 {
            t.step(k, 3);
        }
        t.stage("video_decode");
        t.mark("encode");
        t.push(Kind::End, "", 0);
        assert_eq!(recorded.load(std::sync::atomic::Ordering::Relaxed), 8);
        let recs = t.resolve();
        let gpu: Vec<_> = recs.iter().filter(|r| r.comp == Comp::Gpu).map(|r| (r.stage, r.name, r.arg, r.dur_ns)).collect();
        assert_eq!(
            gpu,
            vec![
                ("", "text", 0, 2_000_000),
                ("denoise", "step", 1, 2_000_000),
                ("denoise", "step", 2, 2_000_000),
                ("denoise", "step", 3, 2_000_000),
                ("denoise", "tail", 0, 2_000_000),
                ("", "video_decode", 0, 2_000_000),
                ("", "encode", 0, 2_000_000),
            ]
        );
        let host = recs.iter().filter(|r| r.comp == Comp::Engine).count();
        assert_eq!(host, 7);
        // Device placement never precedes the host enqueue of a mark.
        let gpu0 = recs.iter().find(|r| r.comp == Comp::Gpu).unwrap();
        let host0 = recs.iter().find(|r| r.comp == Comp::Engine).unwrap();
        assert!(gpu0.t_ns >= host0.t_ns);
    }

    #[test]
    fn overflow_is_counted_and_the_end_mark_kept() {
        let recorded = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let mut t = Timeline::new(Trace::new_root(), Box::new(Scripted { recorded }), 4);
        t.stage("denoise");
        for k in 1..=10 {
            t.step(k, 10);
        }
        t.push(Kind::End, "", 0);
        assert_eq!(t.labels.len(), 4);
        assert_eq!(t.labels.last().unwrap().kind, Kind::End);
        let recs = t.resolve();
        let o = recs.iter().find(|r| r.name == "timeline.overflow").unwrap();
        assert_eq!(o.arg, 8);
    }

    #[test]
    fn host_marks_give_host_spans_only() {
        let mut t = Timeline::new(Trace::new_root(), Box::new(HostMarks), 8);
        t.stage("text");
        t.stage("denoise");
        t.push(Kind::End, "", 0);
        let recs = t.resolve();
        assert!(recs.iter().all(|r| r.comp == Comp::Engine));
        assert_eq!(recs.len(), 2);
    }
}
