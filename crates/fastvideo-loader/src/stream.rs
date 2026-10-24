//! Loading while the weights are still arriving (docs/ops/r2-weights.md §11).
//!
//! `scripts/gpu/r2-fetch.py --stream` writes a tree in place and, for every
//! file, a progress map `<file>.fvparts` *before* the file itself exists. A
//! part's flag is set only after the part's bytes are on disk and matched the
//! manifest's SHA-256; the map is removed once the whole file is verified. Here
//! a reader waits for the parts covering a byte range before it touches them:
//! [`crate::LazyStore`] for its headers and every view, the read-ahead pool
//! ([`crate::prefetch`]) before every read, and the eager paths
//! ([`crate::load_raw_tensors`], [`crate::pth`]) for the whole file. A file
//! with no map (a volume, the Hub, a tree already complete) has no gate: one
//! `stat` at open, nothing per read.
//!
//! Map layout (little-endian): `0..8` magic `FVPARTS1`, `8..16` file size,
//! `16..24` part size, `24..32` parts, `32..40` want (part + 1 a reader is
//! blocked on, 0 none; written here, read by the fetcher, which moves that
//! file's remaining GETs to the front), `40` state (0 fetching, 1 done, 2
//! failed), `48..` one byte per part (1 = verified).
//!
//! A wait fails, naming the file and the part, when the map says the fetch
//! failed, when the fetcher's status file (`FASTVIDEO_STREAMING_FETCH_STATUS`)
//! says failed or has not been written for `FASTVIDEO_STREAMING_FETCH_STALE_S`
//! (120 s; the fetcher writes it every 5 s), or after
//! `FASTVIDEO_STREAMING_FETCH_TIMEOUT_S` (3600 s). It never hands out a byte
//! of a part that is not flagged.

use std::fs::{File, OpenOptions};
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Once};
use std::time::{Duration, Instant, SystemTime};

use crate::LoaderError;

/// The progress map beside a file being fetched.
pub const MAP_SUFFIX: &str = ".fvparts";
const MAGIC: &[u8; 8] = b"FVPARTS1";
const HEADER: usize = 48;
const WANT: u64 = 32;
const STATE: usize = 40;
const STATE_DONE: u8 = 1;
const STATE_FAILED: u8 = 2;

/// Nanoseconds readers spent blocked on parts not yet fetched (all threads).
pub(crate) static FETCH_WAIT_NS: AtomicU64 = AtomicU64::new(0);

/// How a gate decides the fetch is dead.
#[derive(Debug, Clone)]
pub struct GateOptions {
    /// The fetcher's `--status` JSON: `"state": "failed"`, or not written for `stale`, ends a wait.
    pub status: Option<PathBuf>,
    pub stale: Duration,
    /// The longest a single wait may take.
    pub timeout: Duration,
}

fn env_secs(name: &str, default: f64) -> Duration {
    let s = fastvideo_models::settings::env_var(name)
        .ok()
        .and_then(|v| v.trim().parse::<f64>().ok())
        .filter(|v| v.is_finite() && *v > 0.0)
        .unwrap_or(default);
    Duration::from_secs_f64(s)
}

impl GateOptions {
    pub fn from_env() -> Self {
        Self {
            status: fastvideo_models::settings::env_var_os("FASTVIDEO_STREAMING_FETCH_STATUS")
                .filter(|v| !v.is_empty())
                .map(PathBuf::from),
            stale: env_secs("FASTVIDEO_STREAMING_FETCH_STALE_S", 120.0),
            timeout: env_secs("FASTVIDEO_STREAMING_FETCH_TIMEOUT_S", 3600.0),
        }
    }
}

/// `FASTVIDEO_STREAMING_FETCH=1`: the boot started this process while the
/// weights are still arriving (the read-ahead takes fewer threads then).
pub fn streaming_fetch_env() -> bool {
    streaming_fetch_from(&fastvideo_models::settings::Live)
}

/// [`streaming_fetch_env`] with the setting from `settings` (a load's
/// snapshot).
pub fn streaming_fetch_from(settings: &dyn fastvideo_models::settings::Source) -> bool {
    settings
        .env_var("FASTVIDEO_STREAMING_FETCH")
        .is_ok_and(|v| v.trim() == "1")
}

/// The progress map of one file being fetched.
#[derive(Debug)]
pub struct PartsGate {
    data: PathBuf,
    map: File,
    writable: bool,
    size: u64,
    part: u64,
    done: Vec<AtomicBool>,
    all: AtomicBool,
    want: AtomicU64,
    opts: GateOptions,
}

fn map_path(data: &Path) -> PathBuf {
    let mut s = data.as_os_str().to_os_string();
    s.push(MAP_SUFFIX);
    PathBuf::from(s)
}

fn u64_at(b: &[u8], at: usize) -> u64 {
    u64::from_le_bytes(b[at..at + 8].try_into().expect("8 bytes"))
}

impl PartsGate {
    /// The gate of `data`, or `None` when it has no progress map (not being fetched).
    pub fn open(data: &Path) -> Result<Option<Arc<Self>>, LoaderError> {
        Self::open_with(data, GateOptions::from_env())
    }

    pub fn open_with(data: &Path, opts: GateOptions) -> Result<Option<Arc<Self>>, LoaderError> {
        let mp = map_path(data);
        let err = |what: String| LoaderError::Message(format!("{}: {what}", mp.display()));
        let (map, writable) = match OpenOptions::new().read(true).write(true).open(&mp) {
            Ok(f) => (f, true),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(e) if e.kind() == std::io::ErrorKind::PermissionDenied => match File::open(&mp) {
                Ok(f) => (f, false),
                Err(e) => return Err(err(e.to_string())),
            },
            Err(e) => return Err(err(e.to_string())),
        };
        let mut head = [0u8; HEADER];
        map.read_exact_at(&mut head, 0)
            .map_err(|e| err(format!("progress map header: {e}")))?;
        if &head[..8] != MAGIC {
            return Err(err("not a progress map (bad magic)".into()));
        }
        let (size, part, n) = (u64_at(&head, 8), u64_at(&head, 16), u64_at(&head, 24));
        if part == 0 || n != size.div_ceil(part) || n > 1 << 24 {
            return Err(err(format!(
                "progress map: {n} parts of {part} bytes for {size} bytes"
            )));
        }
        static ANNOUNCE: Once = Once::new();
        ANNOUNCE.call_once(|| {
            eprintln!(
                "fastvideo-loader: streaming fetch (fv-streaming-fetch/1): reads wait on the progress maps of files still arriving"
            )
        });
        let g = Self {
            data: data.to_path_buf(),
            map,
            writable,
            size,
            part,
            done: (0..n).map(|_| AtomicBool::new(false)).collect(),
            all: AtomicBool::new(n == 0),
            want: AtomicU64::new(0),
            opts,
        };
        g.refresh().map_err(|e| err(e.to_string()))?;
        Ok(Some(Arc::new(g)))
    }

    /// The file's size per its map.
    pub fn size(&self) -> u64 {
        self.size
    }

    /// Every part is verified (as last seen).
    pub fn complete(&self) -> bool {
        self.all.load(Ordering::Acquire)
    }

    /// Reads the map again; returns its state byte.
    fn refresh(&self) -> std::io::Result<u8> {
        let n = self.done.len();
        let mut buf = vec![0u8; HEADER + n];
        self.map.read_exact_at(&mut buf, 0)?;
        let state = buf[STATE];
        let mut all = true;
        for (i, flag) in buf[HEADER..].iter().enumerate() {
            if *flag == 1 {
                self.done[i].store(true, Ordering::Release);
            } else {
                all = false;
            }
        }
        // `done` only after every part's flag: the fetcher sets it after the last one.
        if all || state == STATE_DONE {
            self.all.store(true, Ordering::Release);
        }
        Ok(state)
    }

    fn parts(&self, start: u64, end: u64) -> Option<(usize, usize)> {
        let end = end.min(self.size);
        if start >= end {
            return None;
        }
        Some((
            (start / self.part) as usize,
            ((end - 1) / self.part) as usize,
        ))
    }

    fn first_missing(&self, a: usize, b: usize) -> Option<usize> {
        (a..=b).find(|&i| !self.done[i].load(Ordering::Acquire))
    }

    fn dead(&self, i: usize, why: impl std::fmt::Display) -> LoaderError {
        LoaderError::Message(format!(
            "{}: part {i} of {} ({} bytes each) is not fetched and {why}: not loading unverified bytes",
            self.data.display(),
            self.done.len(),
            self.part
        ))
    }

    /// The fetcher's status file: `Err` when it says failed or went stale.
    fn fetcher_alive(&self) -> Result<(), String> {
        let Some(p) = &self.opts.status else {
            return Ok(());
        };
        let Ok(meta) = std::fs::metadata(p) else {
            return Ok(()); // not written yet (the boot starts us right after the fetcher's first status)
        };
        if let Ok(text) = std::fs::read_to_string(p) {
            if let Ok(v) = serde_json::from_str::<serde_json::Value>(&text) {
                if v.get("state").and_then(|s| s.as_str()) == Some("failed") {
                    let e = v.get("error").and_then(|s| s.as_str()).unwrap_or("?");
                    return Err(format!("the fetch failed ({}: {e})", p.display()));
                }
            }
        }
        let age = meta
            .modified()
            .ok()
            .and_then(|m| SystemTime::now().duration_since(m).ok())
            .unwrap_or_default();
        if age > self.opts.stale {
            return Err(format!(
                "the fetcher stopped ({} not written for {:.0}s)",
                p.display(),
                age.as_secs_f64()
            ));
        }
        Ok(())
    }

    /// Blocks until every part covering `[start, end)` is verified.
    pub fn wait(&self, start: u64, end: u64) -> Result<(), LoaderError> {
        if self.all.load(Ordering::Acquire) {
            return Ok(());
        }
        let Some((a, b)) = self.parts(start, end) else {
            return Ok(());
        };
        if self.first_missing(a, b).is_none() {
            return Ok(());
        }
        let t0 = Instant::now();
        let mut nap = Duration::from_millis(1);
        let mut checked = t0;
        let result = loop {
            let state = match self.refresh() {
                Ok(s) => s,
                Err(e) => break Err(self.dead(a, format!("its progress map is unreadable ({e})"))),
            };
            let missing = if self.all.load(Ordering::Acquire) {
                None
            } else {
                self.first_missing(a, b)
            };
            let Some(i) = missing else {
                break Ok(());
            };
            if state == STATE_FAILED {
                break Err(self.dead(i, "the fetch failed"));
            }
            // Ask the fetcher for this part (and the rest of the file) first.
            let w = i as u64 + 1;
            if self.writable && self.want.swap(w, Ordering::AcqRel) != w {
                let _ = self.map.write_at(&w.to_le_bytes(), WANT);
            }
            if checked.elapsed() >= Duration::from_secs(1) {
                checked = Instant::now();
                if let Err(why) = self.fetcher_alive() {
                    break Err(self.dead(i, why));
                }
            }
            if t0.elapsed() > self.opts.timeout {
                break Err(self.dead(
                    i,
                    format!("did not arrive in {:.0}s", self.opts.timeout.as_secs_f64()),
                ));
            }
            std::thread::sleep(nap);
            nap = (nap * 2).min(Duration::from_millis(25));
        };
        FETCH_WAIT_NS.fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
        result
    }

    /// Blocks until the whole file is verified.
    pub fn wait_all(&self) -> Result<(), LoaderError> {
        self.wait(0, self.size)
    }
}

/// Blocks until `[start, end)` of `path` is verified, when `path` is being
/// fetched; returns at once otherwise.
pub fn wait_range(path: &Path, start: u64, end: u64) -> Result<(), LoaderError> {
    match PartsGate::open(path)? {
        Some(g) => g.wait(start, end),
        None => Ok(()),
    }
}

/// Blocks until all of `path` is verified, when it is being fetched (the
/// eager loaders, which read every byte).
pub fn wait_file(path: &Path) -> Result<(), LoaderError> {
    match PartsGate::open(path)? {
        Some(g) => g.wait_all(),
        None => Ok(()),
    }
}

/// Seconds readers have spent blocked on parts not fetched yet (all threads).
pub fn fetch_wait_s() -> f64 {
    FETCH_WAIT_NS.load(Ordering::Relaxed) as f64 * 1e-9
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use std::io::Write;

    fn sha256_hex(data: &[u8]) -> String {
        use sha2::{Digest, Sha256};
        Sha256::digest(data)
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect()
    }

    pub(crate) fn tmp(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("fv-stream-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    /// A safetensors file's bytes: `tensors` is `(name, dtype, shape, bytes)`.
    pub(crate) fn st_bytes(tensors: &[(&str, &str, Vec<usize>, Vec<u8>)]) -> Vec<u8> {
        let mut header = serde_json::Map::new();
        let mut data = Vec::new();
        for (name, dtype, shape, bytes) in tensors {
            let start = data.len();
            data.extend_from_slice(bytes);
            header.insert(
                (*name).to_string(),
                serde_json::json!({"dtype": dtype, "shape": shape, "data_offsets": [start, data.len()]}),
            );
        }
        let h = serde_json::to_vec(&header).unwrap();
        let mut out = (h.len() as u64).to_le_bytes().to_vec();
        out.extend_from_slice(&h);
        out.extend_from_slice(&data);
        out
    }

    fn pattern(n: usize, seed: u32) -> Vec<u8> {
        let mut x = seed.wrapping_mul(2654435761).wrapping_add(1);
        (0..n)
            .map(|_| {
                x ^= x << 13;
                x ^= x >> 17;
                x ^= x << 5;
                (x >> 8) as u8
            })
            .collect()
    }

    /// Twelve bf16 tensors over ~10 parts of 4 KiB: views cross part edges.
    pub(crate) fn sample() -> Vec<u8> {
        let ts: Vec<(String, Vec<u8>)> = (0..12)
            .map(|i| {
                (
                    format!("blocks.{i}.w"),
                    pattern(2 * (1000 + 333 * i), i as u32 + 7),
                )
            })
            .collect();
        let refs: Vec<(&str, &str, Vec<usize>, Vec<u8>)> = ts
            .iter()
            .map(|(n, b)| (n.as_str(), "BF16", vec![b.len() / 2], b.clone()))
            .collect();
        st_bytes(&refs)
    }

    pub(crate) const PART: u64 = 4096;

    /// What r2-fetch.py --stream leaves before its first part: the map (all pending), then the file at full size.
    pub(crate) fn start_fetch(path: &Path, size: u64, part: u64) {
        let n = size.div_ceil(part);
        let mut head = MAGIC.to_vec();
        for v in [size, part, n, 0u64, 0u64] {
            head.extend_from_slice(&v.to_le_bytes());
        }
        head.extend(std::iter::repeat_n(0u8, n as usize));
        std::fs::write(map_path(path), head).unwrap();
        let f = File::create(path).unwrap();
        f.set_len(size).unwrap();
    }

    /// The fetcher's side of one part: its bytes, then (only if `verified`) its flag.
    pub(crate) fn put_part(
        path: &Path,
        data: &[u8],
        i: usize,
        part: u64,
        bytes: &[u8],
        verified: bool,
    ) {
        let f = OpenOptions::new().write(true).open(path).unwrap();
        let a = i * part as usize;
        f.write_all_at(bytes, a as u64).unwrap();
        if verified {
            assert_eq!(bytes, &data[a..(a + part as usize).min(data.len())]);
            let m = OpenOptions::new().write(true).open(map_path(path)).unwrap();
            m.write_all_at(&[1], HEADER as u64 + i as u64).unwrap();
        }
    }

    pub(crate) fn set_state(path: &Path, state: u8) {
        let m = OpenOptions::new().write(true).open(map_path(path)).unwrap();
        m.write_all_at(&[state], STATE as u64).unwrap();
    }

    /// The whole file verified: `done`, then the map goes (as finish_file does).
    pub(crate) fn finish(path: &Path) {
        set_state(path, STATE_DONE);
        std::fs::remove_file(map_path(path)).unwrap();
    }

    pub(crate) fn want(path: &Path) -> u64 {
        let m = File::open(map_path(path)).unwrap();
        let mut b = [0u8; 8];
        m.read_exact_at(&mut b, WANT).unwrap();
        u64::from_le_bytes(b)
    }

    fn opts(timeout_ms: u64) -> GateOptions {
        GateOptions {
            status: None,
            stale: Duration::from_secs(120),
            timeout: Duration::from_millis(timeout_ms),
        }
    }

    /// Every tensor of a store, hashed in key order (what a loader uploads).
    pub(crate) fn digest(store: &crate::LazyStore) -> String {
        let mut keys: Vec<&str> = store.keys().collect();
        keys.sort_by(|a, b| crate::natural_cmp(a, b));
        let mut all = Vec::new();
        for k in keys {
            all.extend_from_slice(k.as_bytes());
            all.extend_from_slice(store.view(k).unwrap().bytes);
        }
        sha256_hex(&all)
    }

    #[test]
    fn no_map_no_gate() {
        let d = tmp("nomap");
        let p = d.join("m.safetensors");
        std::fs::write(&p, sample()).unwrap();
        assert!(PartsGate::open(&p).unwrap().is_none());
        wait_range(&p, 0, 10).unwrap();
        wait_file(&p).unwrap();
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn a_wait_returns_once_its_parts_are_flagged_and_asks_for_the_first_missing_one() {
        let d = tmp("wait");
        let p = d.join("m.bin");
        let data = pattern(5 * PART as usize + 10, 3);
        start_fetch(&p, data.len() as u64, PART);
        let g = PartsGate::open_with(&p, opts(10_000)).unwrap().unwrap();
        put_part(&p, &data, 0, PART, &data[..PART as usize], true);
        g.wait(0, PART).unwrap();
        let (p2, data2) = (p.clone(), data.clone());
        let writer = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(150));
            let w = want(&p2);
            for i in [3usize, 2, 1] {
                let a = i * PART as usize;
                put_part(&p2, &data2, i, PART, &data2[a..a + PART as usize], true);
                std::thread::sleep(Duration::from_millis(30));
            }
            w
        });
        g.wait(PART + 5, 3 * PART + 1).unwrap();
        assert_eq!(
            writer.join().unwrap(),
            2,
            "the reader asks for part 1 (want = part + 1)"
        );
        let mut got = vec![0u8; 3 * PART as usize];
        File::open(&p)
            .unwrap()
            .read_exact_at(&mut got, PART)
            .unwrap();
        assert_eq!(&got[..], &data[PART as usize..4 * PART as usize]);
        assert!(!g.complete());
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn a_failed_fetch_ends_the_wait_naming_the_file_and_part() {
        let d = tmp("failed");
        let p = d.join("m.bin");
        let data = pattern(3 * PART as usize, 4);
        start_fetch(&p, data.len() as u64, PART);
        let g = PartsGate::open_with(&p, opts(30_000)).unwrap().unwrap();
        let p2 = p.clone();
        std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(100));
            set_state(&p2, STATE_FAILED);
        });
        let t = Instant::now();
        let e = g.wait(0, 10).unwrap_err().to_string();
        assert!(t.elapsed() < Duration::from_secs(5));
        assert!(
            e.contains("m.bin") && e.contains("part 0") && e.contains("fetch failed"),
            "{e}"
        );
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn a_stale_or_failed_status_file_and_the_timeout_end_the_wait() {
        let d = tmp("status");
        let p = d.join("m.bin");
        start_fetch(&p, 2 * PART, PART);
        let status = d.join("r2-fetch.json");
        std::fs::write(&status, r#"{"state": "running"}"#).unwrap();
        let old = SystemTime::now() - Duration::from_secs(600);
        File::options()
            .write(true)
            .open(&status)
            .unwrap()
            .set_modified(old)
            .unwrap();
        let mut o = opts(60_000);
        o.status = Some(status.clone());
        let g = PartsGate::open_with(&p, o.clone()).unwrap().unwrap();
        let e = g.wait(PART, PART + 1).unwrap_err().to_string();
        assert!(e.contains("part 1") && e.contains("fetcher stopped"), "{e}");

        std::fs::write(
            &status,
            r#"{"state": "failed", "error": "part 7: gave up"}"#,
        )
        .unwrap();
        let e = g.wait(0, 1).unwrap_err().to_string();
        assert!(e.contains("fetch failed") && e.contains("gave up"), "{e}");

        let g = PartsGate::open_with(&p, opts(200)).unwrap().unwrap();
        let e = g.wait(0, 1).unwrap_err().to_string();
        assert!(e.contains("did not arrive"), "{e}");
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn a_finished_file_is_complete_through_an_open_map_and_without_one() {
        let d = tmp("finish");
        let p = d.join("m.bin");
        let data = pattern(2 * PART as usize + 1, 5);
        start_fetch(&p, data.len() as u64, PART);
        let g = PartsGate::open_with(&p, opts(10_000)).unwrap().unwrap();
        for i in 0..3 {
            let a = i * PART as usize;
            put_part(
                &p,
                &data,
                i,
                PART,
                &data[a..(a + PART as usize).min(data.len())],
                true,
            );
        }
        finish(&p);
        g.wait_all().unwrap();
        assert!(g.complete());
        assert!(PartsGate::open(&p).unwrap().is_none());
        let _ = std::fs::remove_dir_all(&d);
    }

    /// A deterministic shuffle of `0..n` (out-of-order arrival, as 64 parallel GETs land).
    fn shuffled(n: usize, seed: u32) -> Vec<usize> {
        let mut v: Vec<usize> = (0..n).collect();
        let r = pattern(4 * n, seed);
        for i in (1..n).rev() {
            let x = u32::from_le_bytes(r[4 * i..4 * i + 4].try_into().unwrap()) as usize;
            v.swap(i, x % (i + 1));
        }
        v
    }

    /// The fetcher, slowly: parts in `order`, `bad` parts first written wrong (as a corrupt GET would leave them,
    /// unflagged) and fixed later, and after `die_after` parts the fetch dies (the map says failed).
    fn fetcher(
        path: PathBuf,
        data: Vec<u8>,
        order: Vec<usize>,
        bad: Vec<usize>,
        die_after: Option<usize>,
    ) -> std::thread::JoinHandle<()> {
        std::thread::spawn(move || {
            let part = PART as usize;
            for (k, &i) in order.iter().enumerate() {
                if die_after == Some(k) {
                    set_state(&path, STATE_FAILED);
                    return;
                }
                let a = i * part;
                let good = &data[a..(a + part).min(data.len())];
                if bad.contains(&i) {
                    let wrong: Vec<u8> = good.iter().map(|b| b ^ 0x5a).collect();
                    put_part(&path, &data, i, PART, &wrong, false);
                    std::thread::sleep(Duration::from_millis(40));
                }
                put_part(&path, &data, i, PART, good, true);
                std::thread::sleep(Duration::from_millis(3));
            }
            finish(&path);
        })
    }

    #[test]
    fn a_store_loads_while_its_file_arrives_and_matches_a_sequential_load() {
        let data = sample();
        let n = data.len().div_ceil(PART as usize);
        assert!(n >= 8, "the sample spans {n} parts");
        let whole = tmp("seq");
        std::fs::write(whole.join("m.safetensors"), &data).unwrap();
        let expect = digest(&crate::LazyStore::open(&whole).unwrap());
        for (case, prefetch) in [("views", false), ("prefetch", true)] {
            let d = tmp(&format!("arrive-{case}"));
            let p = d.join("m.safetensors");
            start_fetch(&p, data.len() as u64, PART);
            // views: the header part last but one (the store's open waits for it); prefetch: the header first, so
            // the read-ahead is queued while most parts are still missing
            let mut order = shuffled(n, 9);
            order.retain(|&i| i != 0);
            order.insert(if prefetch { 0 } else { n - 2 }, 0);
            let w = fetcher(p.clone(), data.clone(), order, vec![3, n - 1], None);
            let store = crate::LazyStore::open(&d).unwrap();
            if prefetch {
                // the read-ahead must wait too: memory mode would otherwise keep zeros or the corrupt bytes
                store.prefetch_all();
            }
            assert_eq!(
                digest(&store),
                expect,
                "{case}: byte-identical to a sequential load"
            );
            w.join().unwrap();
            assert!(PartsGate::open(&p).unwrap().is_none());
            let _ = std::fs::remove_dir_all(&d);
        }
        let _ = std::fs::remove_dir_all(&whole);
    }

    #[test]
    fn a_dead_fetch_is_an_error_never_zeros() {
        let data = sample();
        let n = data.len().div_ceil(PART as usize);
        let d = tmp("dead");
        let p = d.join("m.safetensors");
        start_fetch(&p, data.len() as u64, PART);
        let w = fetcher(
            p.clone(),
            data.clone(),
            (0..n).collect(),
            vec![],
            Some(n / 2),
        );
        let store = crate::LazyStore::open(&d).unwrap();
        store.prefetch_all();
        w.join().unwrap();
        let whole = tmp("dead-ref");
        std::fs::write(whole.join("m.safetensors"), &data).unwrap();
        let reference = crate::LazyStore::open(&whole).unwrap();
        let (mut ok, mut failed) = (0, 0);
        for k in reference.keys() {
            match store.view(k) {
                Ok(v) => {
                    assert_eq!(v.bytes, reference.view(k).unwrap().bytes, "{k}");
                    ok += 1;
                }
                Err(e) => {
                    let e = e.to_string();
                    assert!(
                        e.contains(k) && e.contains("m.safetensors") && e.contains("fetch failed"),
                        "{e}"
                    );
                    failed += 1;
                }
            }
        }
        assert!(ok > 0 && failed > 0, "{ok} loaded, {failed} refused");
        let _ = std::fs::remove_dir_all(&d);
        let _ = std::fs::remove_dir_all(&whole);
    }

    #[test]
    fn the_eager_loader_waits_for_the_whole_file() {
        let data = sample();
        let n = data.len().div_ceil(PART as usize);
        let whole = tmp("eager-ref");
        std::fs::write(whole.join("m.safetensors"), &data).unwrap();
        let expect = crate::load_raw_tensors_native(&whole).unwrap();
        let d = tmp("eager");
        let p = d.join("m.safetensors");
        start_fetch(&p, data.len() as u64, PART);
        let w = fetcher(p.clone(), data.clone(), shuffled(n, 4), vec![1], None);
        let got = crate::load_raw_tensors_native(&d).unwrap();
        w.join().unwrap();
        assert_eq!(got.len(), expect.len());
        for (k, t) in &expect {
            assert_eq!(got[k].data, t.data, "{k}");
        }
        let _ = std::fs::remove_dir_all(&d);
        let _ = std::fs::remove_dir_all(&whole);
    }

    #[test]
    fn a_bad_map_is_an_error_not_a_free_pass() {
        let d = tmp("badmap");
        let p = d.join("m.bin");
        std::fs::write(&p, b"x").unwrap();
        let mut f = File::create(map_path(&p)).unwrap();
        f.write_all(&[0u8; HEADER]).unwrap();
        assert!(PartsGate::open(&p)
            .unwrap_err()
            .to_string()
            .contains("bad magic"));
        let _ = std::fs::remove_dir_all(&d);
    }
}
