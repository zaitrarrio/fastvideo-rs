//! The recorder: fixed-size records into a bounded channel, drained,
//! formatted and stored by a background thread.

use std::collections::{HashMap, VecDeque};
use std::io::Write as _;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{sync_channel, Receiver, SyncSender, TrySendError};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

use crate::id::TraceId;

/// Which part of the path an event belongs to.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Comp {
    /// The pod's HTTP layer (one span per traced request it answers).
    Http,
    /// A protocol adapter: parse, validate, model resolution, ingestion.
    Adapter,
    /// The job store (insert, terminal write, lookups).
    Store,
    /// A front behind the edge: the hop to the family object.
    Front,
    /// A worker behind the edge: offers from the family object.
    Worker,
    /// The engine's queue.
    Queue,
    /// The engine on the host (stage boundaries as the host saw them).
    Engine,
    /// The device (CUDA events on the compute stream).
    Gpu,
    /// Post-processing: MP4 finish, remux.
    Post,
    /// Output upload (artifact store, R2).
    Upload,
}

impl Comp {
    pub fn name(self) -> &'static str {
        match self {
            Comp::Http => "http",
            Comp::Adapter => "adapter",
            Comp::Store => "store",
            Comp::Front => "front",
            Comp::Worker => "worker",
            Comp::Queue => "queue",
            Comp::Engine => "engine",
            Comp::Gpu => "gpu",
            Comp::Post => "post",
            Comp::Upload => "upload",
        }
    }
}

/// Which clock a record's times come from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Clock {
    /// This process's monotonic clock ([`now_ns`]).
    Host,
    /// Device event times, placed on the host clock when resolved
    /// (docs/serve/tracing.md "GPU timing").
    Device,
}

/// One fixed-size record: what the hot path pushes. `name` and `stage` are
/// static strings, so nothing is formatted or allocated until the drain.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Rec {
    pub trace: TraceId,
    pub comp: Comp,
    pub clock: Clock,
    pub name: &'static str,
    /// Optional prefix of `name` (`denoise` + `step` → `denoise.step`).
    pub stage: &'static str,
    /// Start, ns on [`now_ns`]'s clock.
    pub t_ns: u64,
    /// 0 for a point event.
    pub dur_ns: u64,
    pub arg: i64,
}

impl Rec {
    pub fn point(trace: TraceId, comp: Comp, name: &'static str, t_ns: u64, arg: i64) -> Self {
        Self { trace, comp, clock: Clock::Host, name, stage: "", t_ns, dur_ns: 0, arg }
    }
}

/// A formatted event: what the drain stores, writes and ships, and what
/// other hosts (the edge, the browser) post in.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Event {
    /// Trace id, 32 hex.
    pub trace: String,
    /// The host whose clock `t_wall_ns` is on (`pod:<id>`, `edge`, `client`).
    #[serde(default)]
    pub host: String,
    pub comp: String,
    pub name: String,
    /// `host` or `gpu`.
    #[serde(default = "host_clock")]
    pub clock: String,
    /// Start, ns since the Unix epoch on `host`'s clock.
    pub t_wall_ns: i64,
    /// Start on the host's monotonic clock (pod events).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub t_mono_ns: Option<u64>,
    #[serde(default)]
    pub dur_ns: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub arg: Option<i64>,
    /// Free-form details (clock-sync samples, URLs, sizes).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub attrs: Option<serde_json::Value>,
}

fn host_clock() -> String {
    "host".into()
}

/// Counters of a recorder.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Stats {
    /// Records accepted into the channel.
    pub sent: u64,
    /// Records dropped because the channel was full (never blocks).
    pub dropped: u64,
    /// Channel capacity.
    pub capacity: u64,
}

/// What `GET /fv/v1/traces/{id}` answers.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TraceDump {
    pub trace: String,
    pub host: String,
    pub events: Vec<Event>,
    /// Events not kept because the trace hit its cap.
    pub truncated: u64,
    /// The recorder's counters since the process started.
    pub stats: Stats,
}

/// What the channel carries.
pub enum Msg {
    Rec(Rec),
    /// Work for the drain thread (resolving device events) that yields records.
    Defer(Box<dyn FnOnce() -> Vec<Rec> + Send>),
    /// Formatted events from elsewhere (the edge, the browser).
    Events(Vec<Event>),
    /// Answered once everything before it is drained.
    Flush(SyncSender<()>),
}

/// A bounded, non-blocking event recorder.
pub struct Recorder {
    tx: SyncSender<Msg>,
    sent: AtomicU64,
    dropped: AtomicU64,
    capacity: u64,
}

/// The receiving end of a recorder nobody drains (tests).
pub struct Parked(#[allow(dead_code)] Receiver<Msg>);

impl Recorder {
    fn channel(capacity: usize) -> (Self, Receiver<Msg>) {
        let (tx, rx) = sync_channel(capacity.max(1));
        (Self { tx, sent: AtomicU64::new(0), dropped: AtomicU64::new(0), capacity: capacity.max(1) as u64 }, rx)
    }

    /// A recorder whose channel is never drained: what `emit` does when the
    /// drain falls behind (tests).
    pub fn parked(capacity: usize) -> (Self, Parked) {
        let (r, rx) = Self::channel(capacity);
        (r, Parked(rx))
    }

    /// Pushes `r`; when the channel is full (or gone) drops it and counts
    /// the drop. Never blocks, never allocates.
    #[inline]
    pub fn emit(&self, r: Rec) -> bool {
        self.push(Msg::Rec(r))
    }

    /// Runs `f` on the drain thread, off the caller's path; its records are
    /// stored like emitted ones. One allocation (the box) per call: for once
    /// per job, not per step.
    pub fn defer(&self, f: impl FnOnce() -> Vec<Rec> + Send + 'static) -> bool {
        self.push(Msg::Defer(Box::new(f)))
    }

    #[inline]
    fn push(&self, m: Msg) -> bool {
        match self.tx.try_send(m) {
            Ok(()) => {
                self.sent.fetch_add(1, Ordering::Relaxed);
                true
            }
            Err(TrySendError::Full(_) | TrySendError::Disconnected(_)) => {
                self.dropped.fetch_add(1, Ordering::Relaxed);
                false
            }
        }
    }

    pub fn stats(&self) -> Stats {
        Stats { sent: self.sent.load(Ordering::Relaxed), dropped: self.dropped.load(Ordering::Relaxed), capacity: self.capacity }
    }

    /// Waits (up to `timeout`) until everything sent before this call is
    /// drained. Blocks: not for the hot path (tests, the trace route).
    pub fn flush(&self, timeout: Duration) -> bool {
        let (tx, rx) = sync_channel(1);
        let deadline = Instant::now() + timeout;
        let mut m = Msg::Flush(tx);
        loop {
            match self.tx.try_send(m) {
                Ok(()) => break,
                Err(TrySendError::Full(back)) if Instant::now() < deadline => {
                    m = back;
                    std::thread::sleep(Duration::from_millis(1));
                }
                Err(_) => return false,
            }
        }
        rx.recv_timeout(deadline.saturating_duration_since(Instant::now())).is_ok()
    }
}

// ---- clocks -----------------------------------------------------------------

fn anchor() -> &'static (Instant, i64) {
    static A: OnceLock<(Instant, i64)> = OnceLock::new();
    A.get_or_init(|| {
        let wall = SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_nanos() as i64).unwrap_or(0);
        (Instant::now(), wall)
    })
}

/// Monotonic ns since this process's trace anchor.
#[inline]
pub fn now_ns() -> u64 {
    let a = anchor();
    Instant::now().saturating_duration_since(a.0).as_nanos() as u64
}

/// [`now_ns`] → ns since the Unix epoch (the anchor's wall time plus the
/// monotonic offset: no wall-clock steps inside a process).
pub fn wall_ns(t_ns: u64) -> i64 {
    anchor().1 + t_ns as i64
}

// ---- the global recorder and its store --------------------------------------

const DEFAULT_CAPACITY: usize = 65_536;
const MAX_TRACES: usize = 512;
const MAX_EVENTS: usize = 8192;

struct TraceBuf {
    events: Vec<Event>,
    truncated: u64,
}

#[derive(Default)]
struct Store {
    traces: HashMap<String, TraceBuf>,
    order: VecDeque<String>,
}

impl Store {
    fn push(&mut self, e: Event) {
        if !self.traces.contains_key(&e.trace) {
            if self.order.len() >= MAX_TRACES {
                if let Some(old) = self.order.pop_front() {
                    self.traces.remove(&old);
                }
            }
            self.order.push_back(e.trace.clone());
            self.traces.insert(e.trace.clone(), TraceBuf { events: Vec::new(), truncated: 0 });
        }
        let b = self.traces.get_mut(&e.trace).expect("inserted above");
        if b.events.len() >= MAX_EVENTS {
            b.truncated += 1;
        } else {
            b.events.push(e);
        }
    }
}

fn store() -> &'static Mutex<Store> {
    static S: OnceLock<Mutex<Store>> = OnceLock::new();
    S.get_or_init(Mutex::default)
}

fn lock_store() -> std::sync::MutexGuard<'static, Store> {
    store().lock().unwrap_or_else(|p| p.into_inner())
}

static HOST: OnceLock<String> = OnceLock::new();
type Sink = Box<dyn Fn(&Event) + Send + Sync>;
static SINK: OnceLock<Sink> = OnceLock::new();
static G: OnceLock<Recorder> = OnceLock::new();

/// Names this process in its events (`pod:<id>`). First call wins.
pub fn set_host(name: impl Into<String>) {
    let _ = HOST.set(name.into());
}

fn host() -> &'static str {
    HOST.get().map_or("local", String::as_str)
}

/// Every drained event also goes to `f` (on the drain thread). First call wins.
pub fn set_sink(f: impl Fn(&Event) + Send + Sync + 'static) {
    let _ = SINK.set(Box::new(f));
}

/// Whether the global recorder exists (false in a process that traced nothing).
pub fn started() -> bool {
    G.get().is_some()
}

/// The global recorder; the first call starts its drain thread
/// (`FV_TRACE_BUFFER` records, default 65 536; `FV_TRACE_FILE`: JSON lines).
pub fn global() -> &'static Recorder {
    G.get_or_init(|| {
        let cap = std::env::var("FV_TRACE_BUFFER").ok().and_then(|v| v.parse().ok()).unwrap_or(DEFAULT_CAPACITY);
        let (r, rx) = Recorder::channel(cap);
        let file = std::env::var("FV_TRACE_FILE").ok().filter(|p| !p.is_empty());
        let spawned = std::thread::Builder::new().name("fv-trace-drain".into()).spawn(move || drain(rx, file));
        if let Err(e) = spawned {
            eprintln!("fv-trace: cannot start the drain thread ({e}); trace events are dropped");
        }
        r
    })
}

/// Formats one record (on the drain thread).
fn format(r: &Rec) -> Event {
    let name = if r.stage.is_empty() { r.name.to_owned() } else { format!("{}.{}", r.stage, r.name) };
    Event {
        trace: r.trace.hex(),
        host: host().to_owned(),
        comp: r.comp.name().to_owned(),
        name,
        clock: match r.clock {
            Clock::Host => "host".into(),
            Clock::Device => "gpu".into(),
        },
        t_wall_ns: wall_ns(r.t_ns),
        t_mono_ns: Some(r.t_ns),
        dur_ns: r.dur_ns,
        arg: (r.arg != 0).then_some(r.arg),
        attrs: None,
    }
}

fn drain(rx: Receiver<Msg>, file: Option<String>) {
    let mut out = file.and_then(|p| match std::fs::OpenOptions::new().create(true).append(true).open(&p) {
        Ok(f) => Some(std::io::BufWriter::new(f)),
        Err(e) => {
            eprintln!("fv-trace: FV_TRACE_FILE {p}: {e}");
            None
        }
    });
    let mut batch: Vec<Event> = Vec::with_capacity(256);
    let mut flushes: Vec<SyncSender<()>> = Vec::new();
    while let Ok(first) = rx.recv() {
        let mut next = Some(first);
        while let Some(m) = next.take() {
            match m {
                Msg::Rec(r) => batch.push(format(&r)),
                Msg::Defer(f) => batch.extend(f().iter().map(format)),
                Msg::Events(v) => batch.extend(v),
                Msg::Flush(tx) => flushes.push(tx),
            }
            if batch.len() < 4096 {
                next = rx.try_recv().ok();
            }
        }
        if let Some(w) = out.as_mut() {
            for e in &batch {
                if let Ok(line) = serde_json::to_string(e) {
                    let _ = writeln!(w, "{line}");
                }
            }
            let _ = w.flush();
        }
        if let Some(sink) = SINK.get() {
            for e in &batch {
                sink(e);
            }
        }
        {
            let mut s = lock_store();
            for e in batch.drain(..) {
                s.push(e);
            }
        }
        for tx in flushes.drain(..) {
            let _ = tx.try_send(());
        }
    }
}

/// Stores events from another host (the edge's, the browser's), through
/// the drain (non-blocking; dropped and counted when the channel is full).
pub fn ingest(events: Vec<Event>) -> bool {
    global().push(Msg::Events(events))
}

/// Everything recorded for `id` (after the drain caught up, up to `wait`).
pub fn snapshot(id: &str, wait: Duration) -> Option<TraceDump> {
    let id = id.trim().to_ascii_lowercase();
    if started() && !wait.is_zero() {
        global().flush(wait);
    }
    let s = lock_store();
    let b = s.traces.get(&id)?;
    let mut events = b.events.clone();
    events.sort_by_key(|e| (e.t_wall_ns, e.dur_ns));
    Some(TraceDump { trace: id, host: host().to_owned(), events, truncated: b.truncated, stats: G.get().map(Recorder::stats).unwrap_or_default() })
}

/// The ids of the traces in memory, newest last.
pub fn recent() -> Vec<String> {
    lock_store().order.iter().cloned().collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::id::Trace;

    #[test]
    fn a_full_channel_drops_and_counts_without_blocking() {
        let (r, _parked) = Recorder::parked(8);
        let id = TraceId::random();
        let t0 = Instant::now();
        for i in 0..100_000 {
            r.emit(Rec::point(id, Comp::Engine, "x", i, 0));
        }
        let took = t0.elapsed();
        let s = r.stats();
        assert_eq!(s.sent, 8);
        assert_eq!(s.dropped, 100_000 - 8);
        // 100k pushes into a full channel: microseconds each at worst, never a wait.
        assert!(took < Duration::from_secs(2), "{took:?}");
    }

    #[test]
    fn a_closed_channel_drops_too() {
        let (r, parked) = Recorder::parked(4);
        drop(parked);
        assert!(!r.emit(Rec::point(TraceId::random(), Comp::Http, "x", 0, 0)));
        assert_eq!(r.stats().dropped, 1);
    }

    #[test]
    fn the_drain_formats_and_stores() {
        let t = Trace::new_root();
        t.point(Comp::Http, "recv", 7);
        {
            let mut s = t.span(Comp::Adapter, "parse");
            s.arg(3);
        }
        let ok = global().defer(move || vec![Rec { stage: "denoise", name: "step", clock: Clock::Device, dur_ns: 5, ..Rec::point(t.id, Comp::Gpu, "", 1, 2) }]);
        assert!(ok);
        ingest(vec![Event {
            trace: t.id.hex(),
            host: "client".into(),
            comp: "client".into(),
            name: "click".into(),
            clock: "host".into(),
            t_wall_ns: 1,
            t_mono_ns: None,
            dur_ns: 0,
            arg: None,
            attrs: None,
        }]);
        let d = snapshot(&t.id.hex(), Duration::from_secs(5)).expect("trace stored");
        let names: Vec<_> = d.events.iter().map(|e| (e.comp.as_str(), e.name.as_str(), e.clock.as_str())).collect();
        assert!(names.contains(&("http", "recv", "host")), "{names:?}");
        assert!(names.contains(&("adapter", "parse", "host")), "{names:?}");
        assert!(names.contains(&("gpu", "denoise.step", "gpu")), "{names:?}");
        assert!(names.contains(&("client", "click", "host")), "{names:?}");
        let parse = d.events.iter().find(|e| e.name == "parse").unwrap();
        assert_eq!(parse.arg, Some(3));
        assert!(parse.t_wall_ns > 1_600_000_000_000_000_000, "wall clock ns");
        assert!(recent().contains(&t.id.hex()));
    }
}
