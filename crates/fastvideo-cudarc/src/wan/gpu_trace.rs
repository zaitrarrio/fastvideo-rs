//! `FASTVIDEO_GPU_TRACE=1`: an unperturbed GPU activity trace of one warm
//! denoise step, through CUPTI's activity API.
//!
//! The question it answers is whether the device ever waits on the host inside
//! a step. [`super::stats::phase`] cannot: it synchronizes per phase, which is
//! itself the perturbation. CUPTI records kernel / memcpy / memset start and end
//! on the GPU clock with no synchronization, so the union of those intervals is
//! the time the GPU was busy, and everything else inside the step's window is
//! idle time a CUDA graph replay could recover.
//!
//! Only one step per pass is traced (index `FASTVIDEO_GPU_TRACE_STEP`, default
//! 1: the second step, past any first-step warm-up). The step-timing
//! synchronize is excluded: the synchronization kind is switched off before it,
//! and the window is built from device activity only. With the flag off every
//! hook is a cached-bool check.
//!
//! The analysis half ([`analyze`]) is plain host code, unit-tested without a
//! device; the CUPTI half only exists with the `cuda` feature.

use std::collections::BTreeMap;
use std::sync::Mutex;

use serde_json::{json, Value};

/// One device activity, timestamps in ns (CUPTI's normalized clock).
#[derive(Debug, Clone, PartialEq)]
pub struct Activity {
    pub start: u64,
    pub end: u64,
    pub kind: ActivityKind,
    /// Kernel name (shortened), or `memcpy HtoD`, `memset`, ...
    pub name: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActivityKind {
    Kernel,
    Memcpy { dir: &'static str, bytes: u64 },
    Memset { bytes: u64 },
}

/// Everything collected for one traced step.
#[derive(Debug, Clone, Default)]
pub struct TraceInput {
    pub activities: Vec<Activity>,
    pub syncs: u64,
    pub dropped: u64,
    /// Host wall time of the step (as the pipeline's step timer reports it).
    pub wall_s: f64,
    /// CUPTI timestamps (same clock as the activities) of the step's first
    /// launch and of the moment the host finished enqueueing it.
    pub host_begin_ns: Option<u64>,
    pub host_enqueued_ns: Option<u64>,
    /// `launch!` count over the step (NVRTC kernels only; cuBLAS is extra).
    pub host_launches: u64,
}

/// A merged busy interval: `[start, end)` and the indices (into the sorted
/// activity list) of the activity that opened it and the one that ended last.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Busy {
    pub start: u64,
    pub end: u64,
    pub first: usize,
    pub last: usize,
}

/// Union of `[start, end)` intervals. `items` must be sorted by start.
/// Touching intervals (`next.start == end`) merge.
pub fn union(items: &[(u64, u64)]) -> Vec<Busy> {
    let mut out: Vec<Busy> = Vec::new();
    for (i, &(s, e)) in items.iter().enumerate() {
        let e = e.max(s);
        match out.last_mut() {
            Some(b) if s <= b.end => {
                if e > b.end {
                    b.end = e;
                    b.last = i;
                }
            }
            _ => out.push(Busy {
                start: s,
                end: e,
                first: i,
                last: i,
            }),
        }
    }
    out
}

/// Gap between consecutive busy intervals: `(len_ns, before_idx, after_idx, at_ns)`.
pub fn gaps(busy: &[Busy]) -> Vec<(u64, usize, usize, u64)> {
    busy.windows(2)
        .map(|w| (w[1].start - w[0].end, w[0].last, w[1].first, w[0].end))
        .filter(|g| g.0 > 0)
        .collect()
}

pub const GAP_BUCKETS: [(&str, u64); 4] = [
    ("<5us", 5_000),
    ("5-50us", 50_000),
    ("50-500us", 500_000),
    (">500us", u64::MAX),
];

/// `(count, total ns)` of gaps per [`GAP_BUCKETS`] bucket.
pub fn histogram(gap_ns: impl IntoIterator<Item = u64>) -> [(u64, u64); 4] {
    let mut h = [(0u64, 0u64); 4];
    for g in gap_ns {
        let b = GAP_BUCKETS
            .iter()
            .position(|&(_, hi)| g < hi)
            .unwrap_or(GAP_BUCKETS.len() - 1);
        h[b].0 += 1;
        h[b].1 += g;
    }
    h
}

/// The `k` largest by `key`, largest first; ties keep input order.
pub fn top_k<T: Clone>(items: &[T], k: usize, key: impl Fn(&T) -> u64) -> Vec<T> {
    let mut idx: Vec<usize> = (0..items.len()).collect();
    idx.sort_by(|&a, &b| key(&items[b]).cmp(&key(&items[a])).then(a.cmp(&b)));
    idx.into_iter().take(k).map(|i| items[i].clone()).collect()
}

/// Where a kernel's time goes, for aiming fusion / bf16 work.
pub fn category(name: &str) -> &'static str {
    // `bcast` (broadcast elementwise) must not read as a dtype cast.
    let n = name.to_ascii_lowercase().replace("bcast", "bcst");
    let any = |keys: &[&str]| keys.iter().any(|k| n.contains(k));
    // Attention first: cuDNN / flash kernels can carry an `sm90_`-style tag.
    if any(&[
        "attn", "flash", "fmha", "sdpa", "vsa_", "sol_", "pisa", "softmax",
    ]) {
        "attention"
    } else if any(&[
        // cuDNN convolution engines (before `gemm`: `implicit_gemm` fprop
        // kernels are convolutions).
        "fprop", "convolve", "winograd", "dgrad", "conv2d", "conv3d", "fft2d", "fft3d",
    ]) {
        "conv"
    } else if any(&[
        "gemm", "gemv", "cutlass", "xmma", "nvjet", "cublas", "matmul", "splitk", "wgmma",
        "tensorop", "s16816", "sm80_", "sm90_", "sm100_", "sm120_", "ampere_", "hopper",
    ]) {
        "gemm"
    } else if any(&[
        "cast",
        "convert",
        "quantiz",
        "dequant",
        "e4m3",
        "fp8",
        "mxfp8",
        "nvfp4",
        "w8a8",
        "_bf16_f32",
        "_f32_bf16",
        "amax",
    ]) {
        "cast"
    } else if any(&[
        "copy",
        "gather",
        "scatter",
        "split_heads",
        "merge_heads",
        "qkv_heads",
        "pad_axis",
        "repeat",
        "permute",
        "transpose",
        "index_",
        "unfold",
        "concat",
        "fill",
    ]) {
        "layout"
    } else if any(&[
        "norm", "mod", "adaln", "rope", "rotary", "gate", "swiglu", "gelu", "silu", "elem", "bcst",
        "binary", "unary", "add", "mul", "sub", "lincomb", "scalar", "residual", "bias", "sigmoid",
        "tanh", "snake", "clamp", "abs", "act", "scale",
    ]) {
        "norm_modulate_elementwise"
    } else {
        "other"
    }
}

/// Short, readable kernel name: Itanium-mangled names (`_Z...`) become the
/// qualified function name plus its first template argument, which for
/// cuBLAS / CUTLASS kernels is what names the tile config. Anything else is
/// kept, capped in length.
pub fn short_name(raw: &str) -> String {
    const CAP: usize = 120;
    let s = raw.strip_prefix("void ").unwrap_or(raw);
    let out = demangle_head(s).unwrap_or_else(|| s.to_string());
    if out.len() > CAP {
        let mut cut = CAP;
        while !out.is_char_boundary(cut) {
            cut -= 1;
        }
        format!("{}…", &out[..cut])
    } else {
        out
    }
}

struct Demangler<'a> {
    b: &'a [u8],
    i: usize,
    subs: Vec<String>,
}

impl Demangler<'_> {
    fn peek(&self) -> Option<u8> {
        self.b.get(self.i).copied()
    }

    fn ident(&mut self) -> Option<String> {
        let start = self.i;
        while self.peek()?.is_ascii_digit() {
            self.i += 1;
        }
        let n: usize = std::str::from_utf8(&self.b[start..self.i])
            .ok()?
            .parse()
            .ok()?;
        let s = std::str::from_utf8(self.b.get(self.i..self.i + n)?).ok()?;
        self.i += n;
        Some(s.to_string())
    }

    /// `S_`, `S0_`, `St`, ...: best-effort, from the prefixes seen so far.
    fn substitution(&mut self) -> Option<String> {
        self.i += 1; // 'S'
        match self.peek()? {
            b't' => {
                self.i += 1;
                Some("std".into())
            }
            b'_' => {
                self.i += 1;
                Some(self.subs.first().cloned().unwrap_or_else(|| "?".into()))
            }
            _ => {
                let start = self.i;
                while self.peek()? != b'_' {
                    self.i += 1;
                }
                let id = std::str::from_utf8(&self.b[start..self.i]).ok()?;
                self.i += 1;
                let n = usize::from_str_radix(id, 36).ok()? + 1;
                Some(self.subs.get(n).cloned().unwrap_or_else(|| "?".into()))
            }
        }
    }

    /// First template argument only; the rest of the list is not walked.
    fn first_template_arg(&mut self) -> Option<String> {
        self.i += 1; // 'I'
        match self.peek()? {
            b'N' | b'0'..=b'9' | b'S' => self.name(),
            _ => Some("…".into()),
        }
    }

    fn name(&mut self) -> Option<String> {
        match self.peek()? {
            b'N' => {
                self.i += 1;
                while matches!(self.peek()?, b'r' | b'V' | b'K') {
                    self.i += 1;
                }
                let mut parts: Vec<String> = Vec::new();
                loop {
                    match self.peek()? {
                        b'E' => {
                            self.i += 1;
                            break;
                        }
                        b'0'..=b'9' => {
                            parts.push(self.ident()?);
                            self.subs.push(parts.join("::"));
                        }
                        b'S' => {
                            let s = self.substitution()?;
                            parts.push(s);
                        }
                        b'I' => {
                            let arg = self.first_template_arg()?;
                            let last = parts.last_mut()?;
                            last.push('<');
                            last.push_str(&arg);
                            last.push('>');
                            break;
                        }
                        b'L' => self.i += 1,
                        _ => break,
                    }
                }
                (!parts.is_empty()).then(|| parts.join("::"))
            }
            b'S' => self.substitution(),
            b'L' => {
                self.i += 1;
                self.name()
            }
            b'0'..=b'9' => {
                let mut id = self.ident()?;
                self.subs.push(id.clone());
                if self.peek() == Some(b'I') {
                    let arg = self.first_template_arg()?;
                    id = format!("{id}<{arg}>");
                }
                Some(id)
            }
            _ => None,
        }
    }
}

fn demangle_head(s: &str) -> Option<String> {
    let rest = s.strip_prefix("_Z")?;
    Demangler {
        b: rest.as_bytes(),
        i: 0,
        subs: Vec::new(),
    }
    .name()
}

const TOP_KERNELS: usize = 25;
const TOP_GAPS: usize = 10;

/// The `h3/gpu_trace` JSON from one step's records.
pub fn analyze(label: &str, step: usize, input: &TraceInput) -> Value {
    let mut acts = input.activities.clone();
    acts.sort_by_key(|a| (a.start, a.end));
    let s = |ns: u64| ns as f64 * 1e-9;
    if acts.is_empty() {
        return json!({
            "pass": label, "step": step, "wall_s": input.wall_s,
            "error": "no device activity recorded",
            "sync_count": input.syncs, "dropped_records": input.dropped,
        });
    }
    let first = acts.iter().map(|a| a.start).min().unwrap_or(0);
    let last = acts.iter().map(|a| a.end).max().unwrap_or(first);
    let window = last.saturating_sub(first);
    let spans: Vec<(u64, u64)> = acts.iter().map(|a| (a.start, a.end)).collect();
    let busy = union(&spans);
    let busy_ns: u64 = busy.iter().map(|b| b.end - b.start).sum();
    let idle_ns = window.saturating_sub(busy_ns);
    let gap_list = gaps(&busy);
    let hist = histogram(gap_list.iter().map(|g| g.0));

    let largest: Vec<Value> = top_k(&gap_list, TOP_GAPS, |g| g.0)
        .into_iter()
        .map(|(len, before, after, at)| {
            json!({
                "gap_us": len as f64 * 1e-3,
                "at_s": s(at - first),
                "before": acts[before].name,
                "after": acts[after].name,
            })
        })
        .collect();

    let mut kernels: BTreeMap<&str, (u64, u64)> = BTreeMap::new();
    let mut memcpy: BTreeMap<&str, (u64, u64)> = BTreeMap::new();
    let (mut memset_n, mut memset_b) = (0u64, 0u64);
    for a in &acts {
        match a.kind {
            ActivityKind::Kernel => {
                let e = kernels.entry(a.name.as_str()).or_default();
                e.0 += 1;
                e.1 += a.end.saturating_sub(a.start);
            }
            ActivityKind::Memcpy { dir, bytes } => {
                let e = memcpy.entry(dir).or_default();
                e.0 += 1;
                e.1 += bytes;
            }
            ActivityKind::Memset { bytes } => {
                memset_n += 1;
                memset_b += bytes;
            }
        }
    }
    let kernel_count: u64 = kernels.values().map(|v| v.0).sum();
    let kernel_ns: u64 = kernels.values().map(|v| v.1).sum();
    let share = |ns: u64| {
        if kernel_ns > 0 {
            ns as f64 / kernel_ns as f64
        } else {
            0.0
        }
    };
    let rows: Vec<(&str, u64, u64)> = kernels.iter().map(|(n, v)| (*n, v.0, v.1)).collect();
    let top_kernels: Vec<Value> = top_k(&rows, TOP_KERNELS, |r| r.2)
        .into_iter()
        .map(|(n, c, ns)| {
            json!({"name": n, "category": category(n), "calls": c, "total_s": s(ns), "share": share(ns)})
        })
        .collect();
    let mut cats: BTreeMap<&str, (u64, u64)> = BTreeMap::new();
    for (n, c, ns) in &rows {
        let e = cats.entry(category(n)).or_default();
        e.0 += c;
        e.1 += ns;
    }
    let cat_rows: Vec<(&str, u64, u64)> = cats.iter().map(|(n, v)| (*n, v.0, v.1)).collect();
    let by_category: Vec<Value> = top_k(&cat_rows, cat_rows.len(), |r| r.2)
        .into_iter()
        .map(|(n, c, ns)| json!({"category": n, "calls": c, "total_s": s(ns), "share": share(ns)}))
        .collect();
    let memcpy_json: serde_json::Map<String, Value> = memcpy
        .iter()
        .map(|(d, (n, b))| (d.to_string(), json!({"count": n, "bytes": b})))
        .collect();
    let host = match (input.host_begin_ns, input.host_enqueued_ns) {
        (Some(b), Some(e)) => json!({
            // Host time to enqueue the whole step.
            "enqueue_s": s(e.saturating_sub(b)),
            // First launch call to the first activity on the device.
            "first_activity_lag_s": first as f64 * 1e-9 - b as f64 * 1e-9,
            // How long the device kept running after the host had enqueued
            // everything: large means the host was ahead (not launch-bound).
            "device_tail_s": last as f64 * 1e-9 - e as f64 * 1e-9,
        }),
        _ => Value::Null,
    };
    json!({
        "pass": label,
        "step": step,
        "wall_s": input.wall_s,
        "window_s": s(window),
        "busy_s": s(busy_ns),
        "idle_s": s(idle_ns),
        "idle_frac": if window > 0 { idle_ns as f64 / window as f64 } else { 0.0 },
        "kernel_count": kernel_count,
        "kernel_s": s(kernel_ns),
        "host_launches": input.host_launches,
        "memcpy": memcpy_json,
        "memset": {"count": memset_n, "bytes": memset_b},
        "sync_count": input.syncs,
        "dropped_records": input.dropped,
        "gap_count": gap_list.len(),
        "gap_histogram": GAP_BUCKETS.iter().zip(hist.iter()).map(|((name, _), (n, ns))| {
            json!({"bucket": name, "count": n, "idle_s": s(*ns)})
        }).collect::<Vec<_>>(),
        "largest_gaps": largest,
        "top_kernels": top_kernels,
        "by_category": by_category,
        "host": host,
    })
}

// ---- step hooks -------------------------------------------------------------

/// The flag. Off: every hook below returns after this check.
pub fn enabled() -> bool {
    static ON: super::envflag::CachedBool = super::envflag::CachedBool::new();
    ON.get_or_init(|| super::envflag::bool_flag("FASTVIDEO_GPU_TRACE", false))
}

fn target_step() -> usize {
    super::envflag::usize_flag("FASTVIDEO_GPU_TRACE_STEP", 1)
}

#[derive(Default)]
struct PassState {
    label: &'static str,
    next_step: usize,
    /// Step being traced, while it runs.
    active: Option<usize>,
    done: bool,
    host_begin_ns: Option<u64>,
    host_enqueued_ns: Option<u64>,
    launches_before: u64,
}

static PASS: Mutex<Option<PassState>> = Mutex::new(None);
static LAST: Mutex<Option<Value>> = Mutex::new(None);

/// Start of a generate: the step counter restarts, one step will be traced.
pub fn pass_begin(label: &'static str) {
    if !enabled() {
        return;
    }
    if let Err(e) = backend::init() {
        eprintln!("[INFO] {label}/gpu_trace disabled: {e}");
        return;
    }
    *PASS.lock().expect("gpu_trace lock") = Some(PassState {
        label,
        ..PassState::default()
    });
}

/// Before the first launch of a step.
pub fn step_begin() {
    if !enabled() {
        return;
    }
    let mut g = PASS.lock().expect("gpu_trace lock");
    let Some(p) = g.as_mut() else { return };
    let step = p.next_step;
    p.next_step += 1;
    if p.done || p.active.is_some() || step != target_step() {
        return;
    }
    match backend::start() {
        Ok(()) => {
            p.active = Some(step);
            p.host_begin_ns = backend::now();
            p.launches_before = super::stats::snapshot().launches;
        }
        Err(e) => {
            p.done = true;
            eprintln!("[INFO] {}/gpu_trace could not start: {e}", p.label);
        }
    }
}

/// After the step's last launch, before the step-timing synchronize: that
/// synchronize is not part of what is measured.
pub fn step_before_sync() {
    if !enabled() {
        return;
    }
    let mut g = PASS.lock().expect("gpu_trace lock");
    if let Some(p) = g.as_mut().filter(|p| p.active.is_some()) {
        p.host_enqueued_ns = backend::now();
        backend::stop_syncs();
    }
}

/// After the step-timing synchronize, with the step's wall time. Collects,
/// analyzes and logs the traced step.
pub fn step_end(wall_s: f64) {
    if !enabled() {
        return;
    }
    let mut g = PASS.lock().expect("gpu_trace lock");
    let Some(p) = g.as_mut() else { return };
    let Some(step) = p.active.take() else { return };
    p.done = true;
    let launches = super::stats::snapshot()
        .launches
        .saturating_sub(p.launches_before);
    let value = match backend::finish() {
        Ok((activities, syncs, dropped)) => analyze(
            p.label,
            step,
            &TraceInput {
                activities,
                syncs,
                dropped,
                wall_s,
                host_begin_ns: p.host_begin_ns,
                host_enqueued_ns: p.host_enqueued_ns,
                host_launches: launches,
            },
        ),
        Err(e) => json!({"pass": p.label, "step": step, "wall_s": wall_s, "error": e}),
    };
    // Printed regardless of FASTVIDEO_LOG: this line is the measurement.
    eprintln!("[INFO] {}/gpu_trace {value}", p.label);
    *LAST.lock().expect("gpu_trace lock") = Some(value);
}

/// The most recent step report (for the gpucheck summary).
pub fn last_report() -> Option<Value> {
    LAST.lock().expect("gpu_trace lock").clone()
}

pub fn reset_report() {
    *LAST.lock().expect("gpu_trace lock") = None;
    *LAST_WINDOW.lock().expect("gpu_trace lock") = None;
}

// ---- whole-phase windows ----------------------------------------------------

/// `FASTVIDEO_GPU_TRACE_DECODE=1`: trace a whole video decode (every kernel,
/// copy and gap from the first launch to the last) instead of a denoise
/// step. Independent of `FASTVIDEO_GPU_TRACE`; the two should not both be on
/// for the same process (they share CUPTI's activity buffers).
pub fn decode_enabled() -> bool {
    static ON: super::envflag::CachedBool = super::envflag::CachedBool::new();
    ON.get_or_init(|| super::envflag::bool_flag("FASTVIDEO_GPU_TRACE_DECODE", false))
}

struct WindowState {
    label: &'static str,
    host_begin_ns: Option<u64>,
    launches_before: u64,
}

static WINDOW: Mutex<Option<WindowState>> = Mutex::new(None);
static LAST_WINDOW: Mutex<Option<Value>> = Mutex::new(None);

/// Start tracing a phase (`label`, e.g. `ltx2/decode`). A no-op unless
/// [`decode_enabled`]; a second begin before the end is ignored.
pub fn window_begin(label: &'static str) {
    if !decode_enabled() {
        return;
    }
    let mut g = WINDOW.lock().expect("gpu_trace lock");
    if g.is_some() {
        return;
    }
    if let Err(e) = backend::init().and_then(|()| backend::start()) {
        eprintln!("[INFO] {label}/gpu_trace disabled: {e}");
        return;
    }
    *g = Some(WindowState {
        label,
        host_begin_ns: backend::now(),
        launches_before: super::stats::snapshot().launches,
    });
}

/// End the phase begun by [`window_begin`], after the caller synchronized
/// (`wall_s` is its host time). Analyzes, logs (`[INFO] <label>/gpu_trace`)
/// and keeps the report for [`last_window_report`].
pub fn window_end(wall_s: f64) {
    if !decode_enabled() {
        return;
    }
    let Some(w) = WINDOW.lock().expect("gpu_trace lock").take() else {
        return;
    };
    let enqueued = backend::now();
    backend::stop_syncs();
    let launches = super::stats::snapshot()
        .launches
        .saturating_sub(w.launches_before);
    let value = match backend::finish() {
        Ok((activities, syncs, dropped)) => analyze(
            w.label,
            0,
            &TraceInput {
                activities,
                syncs,
                dropped,
                wall_s,
                host_begin_ns: w.host_begin_ns,
                host_enqueued_ns: enqueued,
                host_launches: launches,
            },
        ),
        Err(e) => json!({"pass": w.label, "wall_s": wall_s, "error": e}),
    };
    eprintln!("[INFO] {}/gpu_trace {value}", w.label);
    *LAST_WINDOW.lock().expect("gpu_trace lock") = Some(value);
}

/// The most recent phase report ([`window_end`]).
pub fn last_window_report() -> Option<Value> {
    LAST_WINDOW.lock().expect("gpu_trace lock").clone()
}

#[cfg(not(feature = "cuda"))]
mod backend {
    use super::Activity;

    pub fn init() -> Result<(), String> {
        Err("built without the cuda feature".into())
    }
    pub fn start() -> Result<(), String> {
        Err("built without the cuda feature".into())
    }
    pub fn now() -> Option<u64> {
        None
    }
    pub fn stop_syncs() {}
    pub fn finish() -> Result<(Vec<Activity>, u64, u64), String> {
        Err("built without the cuda feature".into())
    }
}

#[cfg(feature = "cuda")]
mod backend {
    //! CUPTI activity buffers. Records are parsed in the buffer-complete
    //! callback (CUPTI may call it on its own thread) into [`RECORDS`].

    use super::{short_name, Activity, ActivityKind};
    use cudarc::cupti::{
        result::{activity, get_timestamp, subscribe, CuptiError},
        sys::{self, CUptiResult, CUpti_ActivityKind as K},
    };
    use std::alloc::{alloc, dealloc, Layout};
    use std::ffi::CStr;
    use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
    use std::sync::{Mutex, OnceLock};

    const BUF_SIZE: usize = 8 * 1024 * 1024;
    const BUF_ALIGN: usize = 8;
    const KINDS: [K; 4] = [
        K::CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL,
        K::CUPTI_ACTIVITY_KIND_MEMCPY,
        K::CUPTI_ACTIVITY_KIND_MEMSET,
        K::CUPTI_ACTIVITY_KIND_SYNCHRONIZATION,
    ];

    static RECORDS: Mutex<Vec<Activity>> = Mutex::new(Vec::new());
    static SYNCS: AtomicU64 = AtomicU64::new(0);
    static DROPPED: AtomicU64 = AtomicU64::new(0);

    // The leading fields of the kernel / memcpy / memset / synchronization
    // records. Every record version since CUDA 11 shares these prefixes; the
    // asserts pin them to the bindings so a layout change fails the build.
    #[allow(dead_code)]
    #[repr(C)]
    struct KernelHead {
        kind: u32,
        cache_config: u8,
        shared_memory_config: u8,
        registers_per_thread: u16,
        pgc_requested: u32,
        pgc_executed: u32,
        start: u64,
        end: u64,
        completed: u64,
        device_id: u32,
        context_id: u32,
        stream_id: u32,
        grid: [i32; 3],
        block: [i32; 3],
        static_smem: i32,
        dynamic_smem: i32,
        local_per_thread: u32,
        local_total: u32,
        correlation_id: u32,
        grid_id: i64,
        name: *const std::ffi::c_char,
    }
    #[allow(dead_code)]
    #[repr(C)]
    struct MemcpyHead {
        kind: u32,
        copy_kind: u8,
        src_kind: u8,
        dst_kind: u8,
        flags: u8,
        bytes: u64,
        start: u64,
        end: u64,
    }
    #[allow(dead_code)]
    #[repr(C)]
    struct MemsetHead {
        kind: u32,
        value: u32,
        bytes: u64,
        start: u64,
        end: u64,
    }
    use std::mem::offset_of;
    const _: () = {
        assert!(offset_of!(KernelHead, start) == offset_of!(sys::CUpti_ActivityKernel4, start));
        assert!(offset_of!(KernelHead, end) == offset_of!(sys::CUpti_ActivityKernel4, end));
        assert!(offset_of!(KernelHead, name) == offset_of!(sys::CUpti_ActivityKernel4, name));
        assert!(offset_of!(MemcpyHead, bytes) == offset_of!(sys::CUpti_ActivityMemcpy, bytes));
        assert!(offset_of!(MemcpyHead, start) == offset_of!(sys::CUpti_ActivityMemcpy, start));
        assert!(offset_of!(MemcpyHead, end) == offset_of!(sys::CUpti_ActivityMemcpy, end));
        assert!(offset_of!(MemsetHead, bytes) == offset_of!(sys::CUpti_ActivityMemset, bytes));
        assert!(offset_of!(MemsetHead, start) == offset_of!(sys::CUpti_ActivityMemset, start));
        assert!(offset_of!(MemsetHead, end) == offset_of!(sys::CUpti_ActivityMemset, end));
    };

    fn copy_dir(kind: u8) -> &'static str {
        match kind {
            1 => "HtoD",
            2 => "DtoH",
            8 => "DtoD",
            9 => "HtoH",
            10 => "PtoP",
            3..=7 => "array",
            _ => "unknown",
        }
    }

    fn cupti_err(what: &str, e: CuptiError) -> String {
        format!("{what}: {:?}", e.0)
    }

    /// CUPTI calls into the loaded library; a missing symbol would panic
    /// inside cudarc, so every entry point is caught and reported instead.
    fn guarded<T>(what: &str, f: impl FnOnce() -> Result<T, String>) -> Result<T, String> {
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(f))
            .unwrap_or_else(|_| Err(format!("{what}: libcupti call panicked")))
    }

    unsafe extern "C" fn buffer_requested(buf: *mut *mut u8, size: *mut usize, max: *mut usize) {
        let layout = Layout::from_size_align(BUF_SIZE, BUF_ALIGN).expect("layout");
        unsafe {
            // Freed in `buffer_completed`, which CUPTI calls for every buffer.
            let ptr = alloc(layout);
            *buf = ptr;
            *size = if ptr.is_null() { 0 } else { BUF_SIZE };
            *max = 0;
        }
    }

    unsafe extern "C" fn buffer_completed(
        ctx: cudarc::driver::sys::CUcontext,
        stream: u32,
        buf: *mut u8,
        _size: usize,
        valid: usize,
    ) {
        if buf.is_null() {
            return;
        }
        let mut parsed = Vec::new();
        let mut rec: *mut sys::CUpti_Activity = std::ptr::null_mut();
        loop {
            match unsafe { activity::get_next_record(buf, valid, &mut rec) } {
                Ok(()) => {
                    // SAFETY: CUPTI hands back a record inside `buf`; the kind is
                    // read as a raw u32 so an unknown kind is not an invalid enum.
                    unsafe { parse(rec.cast::<u8>(), &mut parsed) };
                }
                Err(CuptiError(CUptiResult::CUPTI_ERROR_MAX_LIMIT_REACHED)) => break,
                Err(_) => break,
            }
        }
        let mut dropped = 0usize;
        if unsafe { activity::get_num_dropped_records(ctx, stream, &mut dropped) }.is_ok() {
            DROPPED.fetch_add(dropped as u64, Relaxed);
        }
        if let Ok(mut g) = RECORDS.lock() {
            g.extend(parsed);
        }
        let layout = Layout::from_size_align(BUF_SIZE, BUF_ALIGN).expect("layout");
        unsafe { dealloc(buf, layout) };
    }

    unsafe fn parse(p: *const u8, out: &mut Vec<Activity>) {
        let kind = unsafe { p.cast::<u32>().read_unaligned() };
        if kind == K::CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL as u32
            || kind == K::CUPTI_ACTIVITY_KIND_KERNEL as u32
        {
            let h = unsafe { p.cast::<KernelHead>().read_unaligned() };
            let name = if h.name.is_null() {
                "?".to_string()
            } else {
                short_name(&unsafe { CStr::from_ptr(h.name) }.to_string_lossy())
            };
            out.push(Activity {
                start: h.start,
                end: h.end,
                kind: ActivityKind::Kernel,
                name,
            });
        } else if kind == K::CUPTI_ACTIVITY_KIND_MEMCPY as u32 {
            let h = unsafe { p.cast::<MemcpyHead>().read_unaligned() };
            let dir = copy_dir(h.copy_kind);
            out.push(Activity {
                start: h.start,
                end: h.end,
                kind: ActivityKind::Memcpy {
                    dir,
                    bytes: h.bytes,
                },
                name: format!("memcpy {dir}"),
            });
        } else if kind == K::CUPTI_ACTIVITY_KIND_MEMSET as u32 {
            let h = unsafe { p.cast::<MemsetHead>().read_unaligned() };
            out.push(Activity {
                start: h.start,
                end: h.end,
                kind: ActivityKind::Memset { bytes: h.bytes },
                name: "memset".into(),
            });
        } else if kind == K::CUPTI_ACTIVITY_KIND_SYNCHRONIZATION as u32 {
            SYNCS.fetch_add(1, Relaxed);
        }
    }

    /// cudarc's loader panics when no libcupti is found; that panic is caught
    /// here, with the default hook silenced so it is not printed as a crash.
    fn cupti_loadable() -> bool {
        let hook = std::panic::take_hook();
        std::panic::set_hook(Box::new(|_| {}));
        let ok = std::panic::catch_unwind(|| {
            // The first CUPTI call loads the library; a timestamp read has no
            // side effects. SAFETY: `t` outlives the call.
            let mut t = 0u64;
            let _ = unsafe { get_timestamp(&mut t) };
        })
        .is_ok();
        std::panic::set_hook(hook);
        ok
    }

    static INIT: OnceLock<Result<(), String>> = OnceLock::new();

    pub fn init() -> Result<(), String> {
        INIT.get_or_init(|| {
            guarded("cupti init", || {
                if !cupti_loadable() {
                    return Err(
                        "libcupti not found (install cuda-cupti-13-4 and put its lib \
                         directory on LD_LIBRARY_PATH; the runtime image ships it)"
                            .into(),
                    );
                }
                let mut handle: sys::CUpti_SubscriberHandle = std::ptr::null_mut();
                unsafe { subscribe(&mut handle, None, std::ptr::null_mut()) }.map_err(|e| {
                    cupti_err("cuptiSubscribe (another CUPTI client, e.g. nsys?)", e)
                })?;
                activity::register_callbacks(Some(buffer_requested), Some(buffer_completed))
                    .map_err(|e| cupti_err("cuptiActivityRegisterCallbacks", e))
            })
        })
        .clone()
    }

    pub fn start() -> Result<(), String> {
        init()?;
        guarded("cupti start", || {
            RECORDS.lock().map_err(|e| e.to_string())?.clear();
            SYNCS.store(0, Relaxed);
            DROPPED.store(0, Relaxed);
            for k in KINDS {
                activity::enable(k).map_err(|e| cupti_err(&format!("enable {k:?}"), e))?;
            }
            Ok(())
        })
    }

    pub fn now() -> Option<u64> {
        let mut t = 0u64;
        guarded("cupti timestamp", || {
            unsafe { get_timestamp(&mut t) }.map_err(|e| cupti_err("cuptiGetTimestamp", e))
        })
        .ok()
        .map(|()| t)
    }

    pub fn stop_syncs() {
        let _ = guarded("cupti disable sync", || {
            activity::disable(K::CUPTI_ACTIVITY_KIND_SYNCHRONIZATION)
                .map_err(|e| cupti_err("disable sync", e))
        });
    }

    pub fn finish() -> Result<(Vec<Activity>, u64, u64), String> {
        guarded("cupti finish", || {
            let mut first_err = None;
            for k in KINDS {
                if let Err(e) = activity::disable(k) {
                    first_err.get_or_insert(cupti_err(&format!("disable {k:?}"), e));
                }
            }
            // SAFETY: plain value transmute between two u32 C enums, as in
            // cudarc's CUPTI example.
            let forced = unsafe {
                std::mem::transmute::<sys::CUpti_ActivityFlag, u32>(
                    sys::CUpti_ActivityFlag::CUPTI_ACTIVITY_FLAG_FLUSH_FORCED,
                )
            };
            activity::flush_all(forced).map_err(|e| cupti_err("cuptiActivityFlushAll", e))?;
            if let Some(e) = first_err {
                return Err(e);
            }
            let acts = std::mem::take(&mut *RECORDS.lock().map_err(|e| e.to_string())?);
            Ok((acts, SYNCS.load(Relaxed), DROPPED.load(Relaxed)))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn k(start: u64, end: u64, name: &str) -> Activity {
        Activity {
            start,
            end,
            kind: ActivityKind::Kernel,
            name: name.into(),
        }
    }

    #[test]
    fn union_merges_overlap_and_touching() {
        let u = union(&[(0, 10), (5, 12), (12, 20), (25, 30), (26, 28)]);
        assert_eq!(
            u.iter().map(|b| (b.start, b.end)).collect::<Vec<_>>(),
            vec![(0, 20), (25, 30)]
        );
        // The interval that ended the first busy run is index 2 ((12, 20)).
        assert_eq!((u[0].first, u[0].last), (0, 2));
        assert_eq!((u[1].first, u[1].last), (3, 3));
        assert!(union(&[]).is_empty());
    }

    #[test]
    fn contained_interval_does_not_move_last() {
        let u = union(&[(0, 100), (10, 20)]);
        assert_eq!(u.len(), 1);
        assert_eq!(u[0].last, 0);
    }

    #[test]
    fn gaps_name_neighbours() {
        let u = union(&[(0, 10), (15, 20), (20, 22), (1022, 1030)]);
        let g = gaps(&u);
        assert_eq!(g, vec![(5, 0, 1, 10), (1000, 2, 3, 22)]);
    }

    #[test]
    fn histogram_buckets() {
        let h = histogram([
            1_000, 4_999, 5_000, 49_999, 50_000, 499_999, 500_000, 9_000_000,
        ]);
        assert_eq!(h[0], (2, 5_999));
        assert_eq!(h[1], (2, 54_999));
        assert_eq!(h[2], (2, 549_999));
        assert_eq!(h[3], (2, 9_500_000));
    }

    #[test]
    fn top_k_is_stable_and_bounded() {
        let v = vec![3u64, 9, 1, 9, 5];
        assert_eq!(top_k(&v, 3, |x| *x), vec![9, 9, 5]);
        assert_eq!(top_k(&v, 10, |x| *x).len(), 5);
        let pairs = vec![("a", 2u64), ("b", 2), ("c", 1)];
        assert_eq!(top_k(&pairs, 2, |p| p.1), vec![("a", 2), ("b", 2)]);
    }

    #[test]
    fn analyze_reports_busy_idle_and_gaps() {
        let mut acts = vec![
            k(1_000, 2_000, "gemm_a"),
            k(1_500, 3_000, "h3_norm_mod"),
            k(103_000, 104_000, "vsa_fused_attn"),
            Activity {
                start: 104_000,
                end: 105_000,
                kind: ActivityKind::Memcpy {
                    dir: "DtoH",
                    bytes: 8,
                },
                name: "memcpy DtoH".into(),
            },
            k(105_010, 106_000, "gemm_a"),
        ];
        acts.reverse(); // analysis must not depend on input order
        let v = analyze(
            "h3",
            1,
            &TraceInput {
                activities: acts,
                syncs: 2,
                wall_s: 1.0,
                ..TraceInput::default()
            },
        );
        let f = |k: &str| v[k].as_f64().unwrap();
        assert!((f("window_s") - 105_000e-9).abs() < 1e-15);
        assert!((f("busy_s") - 4_990e-9).abs() < 1e-15);
        assert!((f("idle_s") - 100_010e-9).abs() < 1e-15);
        assert_eq!(v["kernel_count"], 4);
        assert_eq!(v["memcpy"]["DtoH"]["count"], 1);
        assert_eq!(v["memcpy"]["DtoH"]["bytes"], 8);
        assert_eq!(v["sync_count"], 2);
        assert_eq!(v["gap_count"], 2);
        assert_eq!(v["gap_histogram"][0]["count"], 1); // 10 ns
        assert_eq!(v["gap_histogram"][2]["count"], 1); // 100 us
        assert_eq!(v["largest_gaps"][0]["before"], "h3_norm_mod");
        assert_eq!(v["largest_gaps"][0]["after"], "vsa_fused_attn");
        assert_eq!(v["top_kernels"][0]["name"], "gemm_a");
        assert_eq!(v["top_kernels"][0]["calls"], 2);
        assert_eq!(v["top_kernels"][0]["category"], "gemm");
        assert_eq!(v["by_category"][0]["category"], "gemm");
    }

    #[test]
    fn analyze_empty_is_an_error_row() {
        let v = analyze("h3", 1, &TraceInput::default());
        assert!(v["error"].is_string());
    }

    #[test]
    fn short_names() {
        assert_eq!(short_name("h3_norm_mod"), "h3_norm_mod");
        assert_eq!(short_name("_Z8vsa_topkPKfPii"), "vsa_topk");
        let arg = "cutlass_80_tensorop_bf16_s16816gemm_bf16_64x64";
        let mangled = format!("_ZN7cutlass7Kernel2I{}{arg}EEvNT_6ParamsE", arg.len());
        assert_eq!(short_name(&mangled), format!("cutlass::Kernel2<{arg}>"));
        // Template argument through a substitution (`NS_...E` = cutlass::...).
        assert_eq!(
            short_name("_ZN7cutlass7Kernel2INS_4gemm6kernel4GemmEEEvNT_6ParamsE"),
            "cutlass::Kernel2<cutlass::gemm::kernel::Gemm>"
        );
        assert_eq!(
            short_name("void cutlass::Kernel2<foo>(bar)"),
            "cutlass::Kernel2<foo>(bar)"
        );
        assert!(short_name(&"x".repeat(500)).chars().count() <= 121);
        // Not a valid mangling: kept.
        assert_eq!(short_name("_Zgarbage"), "_Zgarbage");
    }

    #[test]
    fn categories() {
        assert_eq!(category("cutlass::Kernel2<cutlass_80_tensorop>"), "gemm");
        assert_eq!(category("nvjet_sm120_tst_128x256"), "gemm");
        assert_eq!(category("nvfp4_w4a4_gemm"), "gemm");
        assert_eq!(category("vsa_fused_attn"), "attention");
        assert_eq!(category("flash_attn_f32"), "attention");
        assert_eq!(category("cast_f32_bf16"), "cast");
        assert_eq!(category("bcast_binary"), "norm_modulate_elementwise");
        assert_eq!(category("h3_norm_mod"), "norm_modulate_elementwise");
        assert_eq!(category("h3v_swiglu_bf16"), "norm_modulate_elementwise");
        assert_eq!(category("split_heads_bhsd"), "layout");
        assert_eq!(category("pack_rgb_u8"), "other");
        assert_eq!(
            category("sm90_xmma_fprop_implicit_gemm_bf16bf16_bf16f32_f32_ndhwc"),
            "conv"
        );
        assert_eq!(category("implicit_convolveNd_sgemm"), "conv");
        assert_eq!(category("ltxv_norm_silu"), "norm_modulate_elementwise");
    }
}
