//! Cold-load read-ahead for [`crate::LazyStore`] (E12).
//!
//! A lazy store maps its shards and lets the consumer page tensors in by
//! touching them. On a local NVMe that is fine; on a network volume
//! (`/runpod-volume`, `/workspace`) every page fault is a small synchronous
//! request, and a 40 GB DiT loads at a few hundred MB/s while the link could
//! carry several GB/s. Here a pool of threads reads the tensors a loader is
//! about to ask for with large `pread`s, in the order the loader will ask for
//! them, so the pages are in the page cache by the time the mapping is
//! touched. The consumer's code path is unchanged — it still reads the same
//! mapping — so the loaded tensors are the same bytes by construction.
//!
//! The pool keeps at most a window of prefetched-but-not-yet-consumed bytes
//! (`FASTVIDEO_PREFETCH_WINDOW_GB`, default the smaller of 24 GB and 40% of
//! `MemAvailable`), so a checkpoint larger than host RAM is streamed rather
//! than read ahead into eviction. A tensor counts as consumed the first time a
//! view of it is taken. A store that is dropped removes its queued reads and
//! releases its share of the window.
//!
//! Requests from several stores queue in submission order: a pipeline that
//! registers its text encoder, then its DiT, then its VAE keeps the volume busy
//! across component boundaries (the DiT's first shards are read while the
//! encoder's last layers are still being uploaded).
//!
//! `FASTVIDEO_PREFETCH=0` turns it off; `FASTVIDEO_PREFETCH_THREADS` (16) and
//! `FASTVIDEO_PREFETCH_CHUNK_MB` (16) size the reads.

use std::collections::VecDeque;
use std::fs::File;
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicI64, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, OnceLock};
use std::time::{Duration, Instant};

/// Pool configuration, read once from the environment.
#[derive(Debug, Clone, Copy)]
pub struct PrefetchConfig {
    pub enabled: bool,
    pub threads: usize,
    pub chunk: usize,
    pub window: u64,
}

fn env_usize(name: &str) -> Option<usize> {
    std::env::var(name).ok().and_then(|v| v.trim().parse().ok())
}

/// `MemAvailable` from `/proc/meminfo`, in bytes.
pub fn mem_available() -> Option<u64> {
    let text = std::fs::read_to_string("/proc/meminfo").ok()?;
    text.lines()
        .find_map(|l| l.strip_prefix("MemAvailable:"))
        .and_then(|v| v.trim().trim_end_matches("kB").trim().parse::<u64>().ok())
        .map(|kb| kb * 1024)
}

impl PrefetchConfig {
    pub fn from_env() -> Self {
        let enabled = std::env::var("FASTVIDEO_PREFETCH").map_or(true, |v| v.trim() != "0");
        let threads = env_usize("FASTVIDEO_PREFETCH_THREADS")
            .unwrap_or(16)
            .clamp(1, 256);
        let chunk = env_usize("FASTVIDEO_PREFETCH_CHUNK_MB")
            .unwrap_or(16)
            .clamp(1, 1024)
            << 20;
        let window = match std::env::var("FASTVIDEO_PREFETCH_WINDOW_GB")
            .ok()
            .and_then(|v| v.trim().parse::<f64>().ok())
        {
            Some(gb) => (gb * 1e9) as u64,
            None => {
                let cap = 24_000_000_000u64;
                mem_available().map_or(cap, |a| cap.min(a / 10 * 4))
            }
        }
        .max(chunk as u64 * 2);
        Self {
            enabled,
            threads,
            chunk,
            window,
        }
    }

    pub fn get() -> Self {
        static CFG: OnceLock<PrefetchConfig> = OnceLock::new();
        *CFG.get_or_init(Self::from_env)
    }
}

/// Per-store bookkeeping shared by the store and its queued reads.
#[derive(Debug)]
pub(crate) struct StoreShared {
    id: u64,
    /// Bytes prefetched and not yet consumed, per tensor (entry index).
    prefetched: Vec<AtomicU64>,
    consumed: Vec<AtomicBool>,
    dropped: AtomicBool,
}

impl StoreShared {
    /// The consumer took a view of entry `i`. `true` the first time.
    pub(crate) fn consume(&self, i: usize) -> bool {
        if self.consumed[i].load(Ordering::Relaxed) || self.consumed[i].swap(true, Ordering::AcqRel)
        {
            return false;
        }
        let x = self.prefetched[i].swap(0, Ordering::AcqRel);
        if x > 0 {
            pool().outstanding.fetch_sub(x as i64, Ordering::AcqRel);
            pool().cv.notify_all();
        }
        true
    }

    pub(crate) fn release(&self) {
        self.dropped.store(true, Ordering::Release);
        if !pool().started.load(Ordering::Acquire) {
            return;
        }
        let mut freed = 0u64;
        for p in &self.prefetched {
            freed += p.swap(0, Ordering::AcqRel);
        }
        let pool = pool();
        pool.queue
            .lock()
            .expect("prefetch queue")
            .retain(|c| c.store.id != self.id);
        if freed > 0 {
            pool.outstanding.fetch_sub(freed as i64, Ordering::AcqRel);
        }
        pool.cv.notify_all();
    }
}

struct Chunk {
    file: Arc<File>,
    offset: u64,
    len: usize,
    /// `(entry index, bytes of that entry inside this chunk)`.
    keys: Vec<(u32, u64)>,
    store: Arc<StoreShared>,
}

#[derive(Default)]
struct Stats {
    read_bytes: AtomicU64,
    read_ns: AtomicU64,
    chunks: AtomicU64,
    skipped_bytes: AtomicU64,
    window_wait_ns: AtomicU64,
    /// ns since [`epoch`] of the first read's start and the last read's end.
    first_ns: AtomicU64,
    last_ns: AtomicU64,
    queued_bytes: AtomicU64,
}

struct Pool {
    queue: Mutex<VecDeque<Chunk>>,
    cv: Condvar,
    outstanding: AtomicI64,
    started: AtomicBool,
    stats: Stats,
    next_id: AtomicU64,
}

fn epoch() -> Instant {
    static E: OnceLock<Instant> = OnceLock::new();
    *E.get_or_init(Instant::now)
}

fn pool() -> &'static Pool {
    static POOL: OnceLock<Pool> = OnceLock::new();
    POOL.get_or_init(|| Pool {
        queue: Mutex::new(VecDeque::new()),
        cv: Condvar::new(),
        outstanding: AtomicI64::new(0),
        started: AtomicBool::new(false),
        stats: Stats {
            first_ns: AtomicU64::new(u64::MAX),
            ..Default::default()
        },
        next_id: AtomicU64::new(1),
    })
}

fn start_workers(cfg: PrefetchConfig) {
    let p = pool();
    if p.started.swap(true, Ordering::AcqRel) {
        return;
    }
    let _ = epoch();
    for t in 0..cfg.threads {
        let _ = std::thread::Builder::new()
            .name(format!("fv-prefetch-{t}"))
            .spawn(move || worker(cfg));
    }
}

fn worker(cfg: PrefetchConfig) {
    let p = pool();
    let mut buf = vec![0u8; cfg.chunk];
    loop {
        let chunk = {
            let mut q = p.queue.lock().expect("prefetch queue");
            loop {
                if q.is_empty() {
                    q = p.cv.wait(q).expect("prefetch queue");
                    continue;
                }
                if p.outstanding.load(Ordering::Acquire) > cfg.window as i64 {
                    let t = Instant::now();
                    q =
                        p.cv.wait_timeout(q, Duration::from_millis(20))
                            .expect("prefetch queue")
                            .0;
                    p.stats
                        .window_wait_ns
                        .fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
                    continue;
                }
                break q.pop_front().expect("non-empty");
            }
        };
        let store = &chunk.store;
        if store.dropped.load(Ordering::Acquire)
            || chunk
                .keys
                .iter()
                .all(|(k, _)| store.consumed[*k as usize].load(Ordering::Acquire))
        {
            p.stats
                .skipped_bytes
                .fetch_add(chunk.len as u64, Ordering::Relaxed);
            continue;
        }
        // Book the bytes before the read so the window holds while it runs.
        for &(k, b) in &chunk.keys {
            let k = k as usize;
            if !store.consumed[k].load(Ordering::Acquire) {
                store.prefetched[k].fetch_add(b, Ordering::AcqRel);
                p.outstanding.fetch_add(b as i64, Ordering::AcqRel);
                if store.consumed[k].load(Ordering::Acquire) {
                    let x = store.prefetched[k].swap(0, Ordering::AcqRel);
                    p.outstanding.fetch_sub(x as i64, Ordering::AcqRel);
                }
            }
        }
        let t0 = Instant::now();
        let start_ns = t0.duration_since(epoch()).as_nanos() as u64;
        p.stats.first_ns.fetch_min(start_ns, Ordering::Relaxed);
        let mut done = 0usize;
        while done < chunk.len {
            match chunk
                .file
                .read_at(&mut buf[..chunk.len - done], chunk.offset + done as u64)
            {
                Ok(0) => break,
                Ok(n) => done += n,
                Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {}
                Err(_) => break,
            }
        }
        let end = Instant::now();
        p.stats
            .read_ns
            .fetch_add((end - t0).as_nanos() as u64, Ordering::Relaxed);
        p.stats.read_bytes.fetch_add(done as u64, Ordering::Relaxed);
        p.stats.chunks.fetch_add(1, Ordering::Relaxed);
        p.stats.last_ns.fetch_max(
            end.duration_since(epoch()).as_nanos() as u64,
            Ordering::Relaxed,
        );
    }
}

/// One tensor to read: entry index, file, `[start, end)` in the file.
pub(crate) struct Range {
    pub entry: usize,
    pub file: usize,
    pub start: usize,
    pub end: usize,
}

/// A store's bookkeeping, before any read is queued.
pub(crate) fn new_shared(entries: usize) -> Arc<StoreShared> {
    Arc::new(StoreShared {
        id: pool().next_id.fetch_add(1, Ordering::Relaxed),
        prefetched: (0..entries).map(|_| AtomicU64::new(0)).collect(),
        consumed: (0..entries).map(|_| AtomicBool::new(false)).collect(),
        dropped: AtomicBool::new(false),
    })
}

/// Queue `ranges` (in the order given) for `store`. `paths[i]` is file `i`.
/// Returns the bytes queued. Ranges whose tensors were already consumed are
/// skipped; neighbouring ranges in one file are merged into reads of up to the
/// configured chunk size.
pub(crate) fn submit(store: &Arc<StoreShared>, paths: &[PathBuf], ranges: &[Range]) -> u64 {
    let cfg = PrefetchConfig::get();
    if !cfg.enabled || ranges.is_empty() {
        return 0;
    }
    let mut files: Vec<Option<Arc<File>>> = vec![None; paths.len()];
    let mut chunks: Vec<Chunk> = Vec::new();
    let mut total = 0u64;
    for r in ranges {
        if store.consumed[r.entry].load(Ordering::Acquire) || r.end <= r.start {
            continue;
        }
        let file = match &files[r.file] {
            Some(f) => f.clone(),
            None => match File::open(&paths[r.file]) {
                Ok(f) => {
                    let f = Arc::new(f);
                    files[r.file] = Some(f.clone());
                    f
                }
                Err(_) => continue,
            },
        };
        let mut at = r.start;
        while at < r.end {
            let n = (r.end - at).min(cfg.chunk);
            let merged = chunks.last_mut().is_some_and(|c| {
                let contiguous = Arc::ptr_eq(&c.file, &file)
                    && c.offset + c.len as u64 == at as u64
                    && c.len + n <= cfg.chunk;
                if contiguous {
                    c.len += n;
                    match c.keys.last_mut() {
                        Some((k, b)) if *k as usize == r.entry => *b += n as u64,
                        _ => c.keys.push((r.entry as u32, n as u64)),
                    }
                }
                contiguous
            });
            if !merged {
                chunks.push(Chunk {
                    file: file.clone(),
                    offset: at as u64,
                    len: n,
                    keys: vec![(r.entry as u32, n as u64)],
                    store: store.clone(),
                });
            }
            at += n;
            total += n as u64;
        }
    }
    if chunks.is_empty() {
        return 0;
    }
    start_workers(cfg);
    let p = pool();
    p.stats.queued_bytes.fetch_add(total, Ordering::Relaxed);
    p.queue.lock().expect("prefetch queue").extend(chunks);
    p.cv.notify_all();
    total
}

/// Totals since process start (or the last [`PrefetchStats::delta`] base).
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct PrefetchStats {
    pub queued_bytes: u64,
    pub read_bytes: u64,
    pub skipped_bytes: u64,
    pub chunks: u64,
    /// Sum of per-thread read time.
    pub read_busy_s: f64,
    /// First read start to last read end.
    pub read_wall_s: f64,
    pub window_wait_s: f64,
    /// Bytes of distinct tensors any lazy store handed out.
    pub viewed_bytes: u64,
}

pub(crate) static VIEWED_BYTES: AtomicU64 = AtomicU64::new(0);

impl PrefetchStats {
    pub fn now() -> Self {
        let s = &pool().stats;
        let first = s.first_ns.load(Ordering::Relaxed);
        let last = s.last_ns.load(Ordering::Relaxed);
        Self {
            queued_bytes: s.queued_bytes.load(Ordering::Relaxed),
            read_bytes: s.read_bytes.load(Ordering::Relaxed),
            skipped_bytes: s.skipped_bytes.load(Ordering::Relaxed),
            chunks: s.chunks.load(Ordering::Relaxed),
            read_busy_s: s.read_ns.load(Ordering::Relaxed) as f64 * 1e-9,
            read_wall_s: if first == u64::MAX || last < first {
                0.0
            } else {
                (last - first) as f64 * 1e-9
            },
            window_wait_s: s.window_wait_ns.load(Ordering::Relaxed) as f64 * 1e-9,
            viewed_bytes: VIEWED_BYTES.load(Ordering::Relaxed),
        }
    }

    /// Counters accumulated since `base` (the wall span is not additive and
    /// stays the process-wide one).
    pub fn since(&self, base: &Self) -> Self {
        Self {
            queued_bytes: self.queued_bytes - base.queued_bytes,
            read_bytes: self.read_bytes - base.read_bytes,
            skipped_bytes: self.skipped_bytes - base.skipped_bytes,
            chunks: self.chunks - base.chunks,
            read_busy_s: self.read_busy_s - base.read_busy_s,
            read_wall_s: self.read_wall_s,
            window_wait_s: self.window_wait_s - base.window_wait_s,
            viewed_bytes: self.viewed_bytes - base.viewed_bytes,
        }
    }

    /// One JSON object for a `load/io` log line. `wall_s` is the caller's
    /// elapsed time for the phase this covers.
    pub fn json(&self, wall_s: f64) -> String {
        let gb = |b: u64| b as f64 / 1e9;
        format!(
            "{{\"wall_s\":{wall_s:.2},\"viewed_gb\":{:.2},\"prefetch_read_gb\":{:.2},\"prefetch_skipped_gb\":{:.2},\"prefetch_busy_s\":{:.1},\"viewed_gbps\":{:.2},\"window_wait_s\":{:.1},\"prefetch\":{}}}",
            gb(self.viewed_bytes),
            gb(self.read_bytes),
            gb(self.skipped_bytes),
            self.read_busy_s,
            if wall_s > 0.0 { gb(self.viewed_bytes) / wall_s } else { 0.0 },
            self.window_wait_s,
            PrefetchConfig::get().enabled,
        )
    }
}

/// Drop `path`'s pages (a file, or every file under a directory) from the
/// page cache, so the next read comes from the volume: what a cold-start
/// measurement needs on a box whose cache is warm from an earlier run.
/// Returns the bytes covered. Pages a live mapping holds are not dropped.
pub fn evict_page_cache(path: &Path) -> std::io::Result<u64> {
    let meta = std::fs::metadata(path)?;
    if meta.is_dir() {
        let mut total = 0;
        for e in std::fs::read_dir(path)? {
            total += evict_page_cache(&e?.path()).unwrap_or(0);
        }
        return Ok(total);
    }
    let f = File::open(path)?;
    // SAFETY: plain syscall on an open descriptor.
    let rc = unsafe {
        use std::os::unix::io::AsRawFd;
        libc::posix_fadvise(f.as_raw_fd(), 0, 0, libc::POSIX_FADV_DONTNEED)
    };
    if rc != 0 {
        return Err(std::io::Error::from_raw_os_error(rc));
    }
    Ok(meta.len())
}

/// Raw read throughput: every file in `files` read once, `threads` workers
/// taking `chunk`-byte reads in file order. Returns `(bytes, seconds)`.
/// `limit` caps the bytes read (0 = all).
pub fn read_bench(files: &[PathBuf], threads: usize, chunk: usize, limit: u64) -> (u64, f64) {
    let mut jobs: Vec<(usize, u64, usize)> = Vec::new();
    let mut handles = Vec::new();
    let mut total = 0u64;
    'outer: for (i, p) in files.iter().enumerate() {
        let Ok(meta) = std::fs::metadata(p) else {
            continue;
        };
        let mut at = 0u64;
        while at < meta.len() {
            if limit > 0 && total >= limit {
                break 'outer;
            }
            let n = (meta.len() - at).min(chunk as u64) as usize;
            jobs.push((i, at, n));
            at += n as u64;
            total += n as u64;
        }
    }
    let open: Vec<Option<Arc<File>>> = files
        .iter()
        .map(|p| File::open(p).ok().map(Arc::new))
        .collect();
    let jobs = Arc::new(Mutex::new(VecDeque::from(jobs)));
    let open = Arc::new(open);
    let read = Arc::new(AtomicU64::new(0));
    let t0 = Instant::now();
    for _ in 0..threads.max(1) {
        let (jobs, open, read) = (jobs.clone(), open.clone(), read.clone());
        handles.push(std::thread::spawn(move || {
            let mut buf = vec![0u8; chunk];
            loop {
                let Some((i, off, n)) = jobs.lock().expect("jobs").pop_front() else {
                    return;
                };
                let Some(f) = open[i].as_ref() else { continue };
                let mut done = 0;
                while done < n {
                    match f.read_at(&mut buf[..n - done], off + done as u64) {
                        Ok(0) | Err(_) => break,
                        Ok(k) => done += k,
                    }
                }
                read.fetch_add(done as u64, Ordering::Relaxed);
            }
        }));
    }
    for h in handles {
        let _ = h.join();
    }
    (read.load(Ordering::Relaxed), t0.elapsed().as_secs_f64())
}
