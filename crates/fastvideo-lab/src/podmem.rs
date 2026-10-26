//! The pod's own memory, from its memory cgroup (docs/adr/0019-fv-lab.md
//! §16.10): what `fv-lab run` adds to `box.txt` and to each arm's
//! `summary.json` and `live.log`, as `scripts/gpu/upstream/pod.sh` does for
//! the upstream pods (`cg_dir`, `cg_mem`, docs/oracle.md "Upstream cells:
//! host RAM").
//!
//! `free` shows the host's RAM, not the pod's share of it. A process the
//! kernel OOM-kills in the pod's cgroup dies with SIGKILL (exit 137) and
//! prints nothing; the cgroup's `oom_kill` count is the evidence, and the
//! limit is what an arm can use before it.
//!
//! - [`MemCgroup::find`]: the cgroup of this process from `/proc/self/cgroup`
//!   (v2's `0::<path>`, v1's `memory` controller line) under the cgroup
//!   mount, else the mount's root (a private cgroup namespace);
//! - [`MemCgroup::limit_bytes`] (v2 `memory.max`, v1 `memory.limit_in_bytes`),
//!   [`MemCgroup::used_bytes`] (process memory: v2 `anon` + `shmem`, v1
//!   `total_rss` + `total_shmem` of `memory.stat`, without the page cache a
//!   weight-heavy arm fills up to the limit at no risk), [`MemCgroup::peak_bytes`]
//!   (the cgroup's peak since the pod started, page cache included: v2
//!   `memory.peak`, v1 `memory.max_usage_in_bytes`) and
//!   [`MemCgroup::oom_kills`] (v2 `memory.events`, v1 `memory.oom_control`);
//! - [`Sampler`]: an arm's peak process memory, sampled while it runs.
//!
//! Nothing here fails a run: what cannot be read is unknown (`None`, `null`).

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::Duration;

/// Where the kernel exposes the cgroup tree, and this process's line in it.
pub const CGROUP_ROOT: &str = "/sys/fs/cgroup";
pub const PROC_SELF_CGROUP: &str = "/proc/self/cgroup";

/// A limit at or over this is "no limit" (v1 writes about 2^63; `pod.sh`'s
/// `mib_or_null` reads the same bound).
const NO_LIMIT: u64 = 1 << 50;

/// One memory cgroup: its version and its directory.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemCgroup {
    /// cgroup v2 (the unified hierarchy) or v1 (the `memory` controller).
    pub v2: bool,
    pub dir: PathBuf,
}

/// `root` joined with a cgroup path (`/`, `/a/b`, `/a/b/`).
fn under(root: &Path, p: &str) -> PathBuf {
    let rel = p.trim_matches('/');
    if rel.is_empty() {
        root.to_path_buf()
    } else {
        root.join(rel)
    }
}

/// The value of `key` in a flat-keyed cgroup file (`memory.stat`,
/// `memory.events`, `memory.oom_control`: `<key> <number>` per line).
fn keyed(text: &str, key: &str) -> Option<u64> {
    text.lines().find_map(|l| {
        let mut w = l.split_whitespace();
        (w.next() == Some(key))
            .then(|| w.next().and_then(|v| v.parse().ok()))
            .flatten()
    })
}

fn read(p: &Path) -> Option<String> {
    fs::read_to_string(p).ok()
}

impl MemCgroup {
    /// The memory cgroup of the process whose `/proc/<pid>/cgroup` is
    /// `proc_cgroup`, under the cgroup mount `root` (`pod.sh`'s `cg_dir`):
    /// v2 at `0::<path>` when `memory.current` is there, else the mount's
    /// root when it has one; then v1's `memory` controller line under
    /// `<root>/memory`, else `<root>/memory` itself. `None`: neither.
    pub fn find(proc_cgroup: &str, root: &Path) -> Option<MemCgroup> {
        let v2_path = proc_cgroup.lines().find_map(|l| {
            let mut f = l.splitn(3, ':');
            match (f.next(), f.next(), f.next()) {
                (Some("0"), Some(""), Some(p)) if !p.is_empty() => Some(p.to_owned()),
                _ => None,
            }
        });
        if let Some(p) = v2_path {
            let dir = under(root, &p);
            if dir.join("memory.current").is_file() {
                return Some(MemCgroup { v2: true, dir });
            }
        }
        if root.join("memory.current").is_file() {
            return Some(MemCgroup {
                v2: true,
                dir: root.to_path_buf(),
            });
        }
        let v1_root = root.join("memory");
        let v1_path = proc_cgroup.lines().find_map(|l| {
            let mut f = l.splitn(3, ':');
            match (f.next(), f.next(), f.next()) {
                (Some(_), Some(c), Some(p)) if c.split(',').any(|c| c == "memory") => {
                    Some(p.to_owned())
                }
                _ => None,
            }
        });
        if let Some(p) = v1_path {
            let dir = under(&v1_root, &p);
            if dir.join("memory.usage_in_bytes").is_file() {
                return Some(MemCgroup { v2: false, dir });
            }
        }
        v1_root
            .join("memory.usage_in_bytes")
            .is_file()
            .then_some(MemCgroup {
                v2: false,
                dir: v1_root,
            })
    }

    /// [`MemCgroup::find`] for this process: `proc_cgroup` read from the
    /// file at that path, under `root`.
    pub fn detect(proc_cgroup: &Path, root: &Path) -> Option<MemCgroup> {
        MemCgroup::find(&read(proc_cgroup).unwrap_or_default(), root)
    }

    /// `v2` or `v1`.
    pub fn version(&self) -> &'static str {
        if self.v2 {
            "v2"
        } else {
            "v1"
        }
    }

    /// The pod's RAM limit (bytes); `None` for none (`max`, v1's huge
    /// number) or unreadable.
    pub fn limit_bytes(&self) -> Option<u64> {
        let f = if self.v2 {
            "memory.max"
        } else {
            "memory.limit_in_bytes"
        };
        read(&self.dir.join(f))
            .and_then(|s| s.trim().parse::<u64>().ok())
            .filter(|b| *b < NO_LIMIT)
    }

    /// Process memory now (bytes): anonymous plus shared memory, without the
    /// page cache.
    pub fn used_bytes(&self) -> Option<u64> {
        let stat = read(&self.dir.join("memory.stat"))?;
        let (a, b) = if self.v2 {
            ("anon", "shmem")
        } else {
            ("total_rss", "total_shmem")
        };
        match (keyed(&stat, a), keyed(&stat, b)) {
            (None, None) => None,
            (x, y) => Some(x.unwrap_or(0) + y.unwrap_or(0)),
        }
    }

    /// The cgroup's peak since the pod started (bytes), page cache included:
    /// cumulative over every earlier arm, not one arm's.
    pub fn peak_bytes(&self) -> Option<u64> {
        let f = if self.v2 {
            "memory.peak"
        } else {
            "memory.max_usage_in_bytes"
        };
        read(&self.dir.join(f)).and_then(|s| s.trim().parse().ok())
    }

    /// Processes the kernel OOM-killed in this cgroup so far.
    pub fn oom_kills(&self) -> Option<u64> {
        let f = if self.v2 {
            "memory.events"
        } else {
            "memory.oom_control"
        };
        read(&self.dir.join(f)).and_then(|s| keyed(&s, "oom_kill"))
    }
}

/// Bytes to whole MiB.
pub fn mib(bytes: Option<u64>) -> Option<u64> {
    bytes.map(|b| b >> 20)
}

/// A JSON number or `null`.
pub fn json_num(v: Option<u64>) -> String {
    v.map_or_else(|| "null".to_owned(), |v| v.to_string())
}

/// `MemTotal` of `/proc/meminfo` text, in GiB (the host's RAM, `free -g`).
pub fn host_ram_gib(meminfo: &str) -> Option<u64> {
    keyed(meminfo, "MemTotal:").map(|kib| kib >> 20)
}

/// The `box.txt` lines of a pod (`pod.sh info_box`): its CPUs and the
/// host's RAM, then its cgroup, RAM limit and OOM kills so far.
pub fn box_lines(cg: Option<&MemCgroup>, cpus: Option<usize>, meminfo: &str) -> String {
    let cpus = cpus.map_or_else(|| "?".to_owned(), |n| n.to_string());
    let host = host_ram_gib(meminfo).map_or_else(|| "?".to_owned(), |g| g.to_string());
    let cgroup = match cg {
        Some(c) => format!(
            "cgroup={} {} pod_ram_limit_mib={} oom_kills={}",
            c.version(),
            c.dir.display(),
            json_num(mib(c.limit_bytes())),
            c.oom_kills().map_or_else(String::new, |n| n.to_string())
        ),
        None => "cgroup=none  pod_ram_limit_mib=null oom_kills=".to_owned(),
    };
    format!("cpus={cpus} mem={host}G\n{cgroup}\n")
}

/// `ram <used> of <limit> MiB` for `live.log`, leaving out what is unknown
/// (`pod.sh`'s `ram_txt`).
pub fn ram_text(used_mib: Option<u64>, limit_mib: Option<u64>) -> String {
    match (used_mib, limit_mib) {
        (Some(u), Some(l)) => format!("ram {u} MiB of {l} MiB"),
        (Some(u), None) => format!("ram {u} MiB"),
        (None, Some(l)) => format!("ram ? of {l} MiB"),
        (None, None) => "ram unknown (no memory cgroup)".to_owned(),
    }
}

/// One arm's memory, read around it.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ArmMem {
    /// The peak process memory sampled during the arm (MiB, no page cache):
    /// the whole cgroup's anon + shmem, so the runner and anything running
    /// beside the arm count too.
    pub peak_ram_mib: Option<u64>,
    /// The RAM limit of the runner's own cgroup (MiB); an ancestor's tighter
    /// limit is not read.
    pub ram_limit_mib: Option<u64>,
    /// The cgroup's exact peak since the pod started (MiB, page cache
    /// included; cumulative). `None` on cgroup v2 before Linux 5.19, which
    /// has no `memory.peak`.
    pub pod_peak_mib_cumulative: Option<u64>,
    /// Processes the kernel OOM-killed in the pod during the arm.
    pub oom_kills: Option<u64>,
}

impl ArmMem {
    /// The `summary.json` fields, each `"key":value,` (`null` when unknown),
    /// to go before `"ended"`.
    pub fn json_fields(&self) -> String {
        format!(
            "\"peak_ram_mib\":{},\"ram_limit_mib\":{},\"pod_peak_mib_cumulative\":{},\"oom_kills\":{},",
            json_num(self.peak_ram_mib),
            json_num(self.ram_limit_mib),
            json_num(self.pod_peak_mib_cumulative),
            json_num(self.oom_kills)
        )
    }
}

/// An arm's peak process memory, sampled every `period` on a thread from
/// [`Sampler::start`] until [`Sampler::finish`], with one sample at each end.
/// Dropped without `finish` (a panic, an early return), it still stops and
/// joins its thread.
pub struct Sampler {
    cg: MemCgroup,
    oom0: Option<u64>,
    stop: Arc<AtomicBool>,
    /// The peak in bytes plus one (0: no sample read).
    peak: Arc<AtomicU64>,
    thread: Option<JoinHandle<()>>,
}

impl Sampler {
    pub fn start(cg: &MemCgroup, period: Duration) -> Sampler {
        let oom0 = cg.oom_kills();
        let stop = Arc::new(AtomicBool::new(false));
        let peak = Arc::new(AtomicU64::new(0));
        let sample = |cg: &MemCgroup, peak: &AtomicU64| {
            if let Some(b) = cg.used_bytes() {
                peak.fetch_max(b.saturating_add(1), Ordering::Relaxed);
            }
        };
        sample(cg, &peak);
        let thread = {
            let (cg, stop, peak) = (cg.clone(), Arc::clone(&stop), Arc::clone(&peak));
            std::thread::Builder::new()
                .name("fv-lab-mem".into())
                .spawn(move || {
                    while !stop.load(Ordering::Relaxed) {
                        std::thread::park_timeout(period);
                        sample(&cg, &peak);
                    }
                })
                .ok()
        };
        Sampler {
            cg: cg.clone(),
            oom0,
            stop,
            peak,
            thread,
        }
    }

    /// Stops the thread and waits for it (once: the handle is taken out).
    fn stop_thread(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(t) = self.thread.take() {
            t.thread().unpark();
            let _ = t.join();
        }
    }

    /// Stops the sampling; the arm's memory.
    pub fn finish(mut self) -> ArmMem {
        self.stop_thread();
        if let Some(b) = self.cg.used_bytes() {
            self.peak.fetch_max(b.saturating_add(1), Ordering::Relaxed);
        }
        let peak = self.peak.load(Ordering::Relaxed);
        ArmMem {
            peak_ram_mib: mib(peak.checked_sub(1)),
            ram_limit_mib: mib(self.cg.limit_bytes()),
            pod_peak_mib_cumulative: mib(self.cg.peak_bytes()),
            oom_kills: match (self.oom0, self.cg.oom_kills()) {
                (Some(a), Some(b)) => Some(b.saturating_sub(a)),
                _ => None,
            },
        }
    }
}

impl Drop for Sampler {
    fn drop(&mut self) {
        self.stop_thread();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("fv-lab-podmem-{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&d);
        fs::create_dir_all(&d).unwrap();
        d
    }

    fn put(dir: &Path, f: &str, text: &str) {
        fs::create_dir_all(dir).unwrap();
        fs::write(dir.join(f), text).unwrap();
    }

    #[test]
    fn finds_the_v2_cgroup_of_the_process_else_the_mount_root() {
        let root = tmp("v2");
        let pod = root.join("kubepods/pod1");
        put(&pod, "memory.current", "1\n");
        let cg = MemCgroup::find("0::/kubepods/pod1\n", &root).unwrap();
        assert_eq!(cg, MemCgroup { v2: true, dir: pod });
        // A private namespace: `0::/`, the files at the mount's root.
        put(&root, "memory.current", "1\n");
        assert_eq!(MemCgroup::find("0::/\n", &root).unwrap().dir, root.clone());
        assert_eq!(MemCgroup::find("0::/gone\n", &root).unwrap().dir, root);
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn finds_the_v1_memory_controller_and_nothing_without_files() {
        let root = tmp("v1");
        assert_eq!(MemCgroup::find("0::/x\n", &root), None);
        let dir = root.join("memory/docker/abc");
        put(&dir, "memory.usage_in_bytes", "1\n");
        let text = "12:pids:/docker/abc\n4:cpu,memory:/docker/abc\n";
        assert_eq!(
            MemCgroup::find(text, &root),
            Some(MemCgroup { v2: false, dir })
        );
        put(&root.join("memory"), "memory.usage_in_bytes", "1\n");
        assert_eq!(
            MemCgroup::find("4:memory:/elsewhere\n", &root).map(|c| c.dir),
            Some(root.join("memory"))
        );
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn reads_limit_process_memory_peak_and_oom_kills() {
        let root = tmp("read");
        let v2 = MemCgroup {
            v2: true,
            dir: root.join("v2"),
        };
        put(&v2.dir, "memory.max", "206158430208\n");
        put(
            &v2.dir,
            "memory.stat",
            "anon 3145728\nfile 999999999\nshmem 1048576\nshmem_thp 7\n",
        );
        put(&v2.dir, "memory.peak", "10485760\n");
        put(
            &v2.dir,
            "memory.events",
            "low 0\nhigh 0\nmax 4\noom 2\noom_kill 1\n",
        );
        assert_eq!(mib(v2.limit_bytes()), Some(196_608));
        assert_eq!(v2.used_bytes(), Some(4 << 20));
        assert_eq!(mib(v2.peak_bytes()), Some(10));
        assert_eq!(v2.oom_kills(), Some(1));
        put(&v2.dir, "memory.max", "max\n");
        assert_eq!(v2.limit_bytes(), None);

        let v1 = MemCgroup {
            v2: false,
            dir: root.join("v1"),
        };
        put(&v1.dir, "memory.limit_in_bytes", "9223372036854771712\n");
        put(
            &v1.dir,
            "memory.stat",
            "rss 1\ntotal_rss 2097152\ntotal_shmem 0\ntotal_cache 5\n",
        );
        put(&v1.dir, "memory.max_usage_in_bytes", "3145728\n");
        put(
            &v1.dir,
            "memory.oom_control",
            "oom_kill_disable 0\nunder_oom 0\noom_kill 3\n",
        );
        assert_eq!(v1.limit_bytes(), None, "v1's no-limit number");
        assert_eq!(v1.used_bytes(), Some(2 << 20));
        assert_eq!(mib(v1.peak_bytes()), Some(3));
        assert_eq!(v1.oom_kills(), Some(3));

        let gone = MemCgroup {
            v2: true,
            dir: root.join("none"),
        };
        assert_eq!(
            (gone.limit_bytes(), gone.used_bytes(), gone.oom_kills()),
            (None, None, None)
        );
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn box_lines_and_ram_text() {
        let root = tmp("box");
        let cg = MemCgroup {
            v2: true,
            dir: root.clone(),
        };
        put(&root, "memory.max", "206158430208\n");
        put(&root, "memory.events", "oom_kill 0\n");
        assert_eq!(
            box_lines(Some(&cg), Some(24), "MemTotal:       2113290240 kB\n"),
            format!(
                "cpus=24 mem=2015G\ncgroup=v2 {} pod_ram_limit_mib=196608 oom_kills=0\n",
                root.display()
            )
        );
        assert_eq!(
            box_lines(None, None, ""),
            "cpus=? mem=?G\ncgroup=none  pod_ram_limit_mib=null oom_kills=\n"
        );
        assert_eq!(ram_text(Some(1), Some(2)), "ram 1 MiB of 2 MiB");
        assert_eq!(ram_text(None, Some(2)), "ram ? of 2 MiB");
        assert_eq!(ram_text(Some(1), None), "ram 1 MiB");
        assert_eq!(ram_text(None, None), "ram unknown (no memory cgroup)");
        let _ = fs::remove_dir_all(&root);
    }

    /// A sampler dropped without `finish` stops and joins its thread (its
    /// clone of the shared peak is gone once the thread has ended).
    #[test]
    fn a_dropped_sampler_stops_its_thread() {
        let root = tmp("drop");
        put(&root, "memory.stat", "anon 1048576\n");
        let cg = MemCgroup {
            v2: true,
            dir: root.clone(),
        };
        let s = Sampler::start(&cg, Duration::from_secs(3600));
        let (stop, peak) = (Arc::clone(&s.stop), Arc::clone(&s.peak));
        assert_eq!(
            Arc::strong_count(&peak),
            3,
            "the sampler, its thread and this test"
        );
        drop(s);
        assert!(stop.load(Ordering::Relaxed));
        assert_eq!(Arc::strong_count(&peak), 1, "the thread has ended");
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn the_sampler_keeps_the_peak_and_counts_kills_during_the_arm() {
        let root = tmp("sampler");
        let cg = MemCgroup {
            v2: true,
            dir: root.clone(),
        };
        put(&root, "memory.max", "1073741824\n");
        put(&root, "memory.peak", "2147483648\n");
        put(&root, "memory.events", "oom_kill 2\n");
        put(&root, "memory.stat", "anon 1048576\nshmem 0\n");
        let s = Sampler::start(&cg, Duration::from_millis(1));
        put(&root, "memory.stat", "anon 734003200\nshmem 0\n");
        std::thread::sleep(Duration::from_millis(50));
        put(&root, "memory.stat", "anon 1048576\nshmem 0\n");
        put(&root, "memory.events", "oom_kill 3\n");
        let m = s.finish();
        assert_eq!(
            m,
            ArmMem {
                peak_ram_mib: Some(700),
                ram_limit_mib: Some(1024),
                pod_peak_mib_cumulative: Some(2048),
                oom_kills: Some(1),
            }
        );
        assert_eq!(
            m.json_fields(),
            "\"peak_ram_mib\":700,\"ram_limit_mib\":1024,\"pod_peak_mib_cumulative\":2048,\"oom_kills\":1,"
        );
        assert_eq!(
            ArmMem::default().json_fields(),
            "\"peak_ram_mib\":null,\"ram_limit_mib\":null,\"pod_peak_mib_cumulative\":null,\"oom_kills\":null,"
        );
        let _ = fs::remove_dir_all(&root);
    }
}
