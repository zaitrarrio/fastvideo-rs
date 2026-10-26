//! `fv-lab run` against fakes (docs/adr/0019-fv-lab.md §5): a fake
//! fv-gpucheck, verify-weights.sh, nvidia-smi and timeout in a temp root, so
//! the runner's files, logs, exit codes, gates, compares and promotion gates
//! are checked without a GPU. The layout is runpod-matrix.sh's.

mod common;

use std::path::{Path, PathBuf};
use std::time::Duration;

use fastvideo_lab::plan::{CellFile, Lab};
use fastvideo_lab::run::{RunConfig, Runner};

/// A `#!/bin/sh` fake tool, written so no exec of it can race a fork
/// (`common::write_exe`).
fn script(path: &Path, body: &str) {
    common::write_exe(path, &format!("#!/bin/sh\n{body}"));
}

struct Fake {
    dir: PathBuf,
}

impl Drop for Fake {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

fn fake(name: &str) -> Fake {
    let dir = std::env::temp_dir().join(format!("fv-lab-runner-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let root = dir.join("root");
    script(
        &root.join("scripts/gpu/verify-weights.sh"),
        "echo \"verify $1 in $FV_WEIGHTS\"\n[ \"$1\" = missing ] && { echo \"INCOMPLETE $1\"; exit 1; }\nexit 0\n",
    );
    std::fs::write(root.join("scripts/gpu/cuda-13.pins"), "# none\n").unwrap();
    std::fs::write(root.join("scripts/gpu/gate-policy.toml"), "# policy\n").unwrap();
    script(
        &dir.join("bin/fv-gpucheck"),
        r#"if [ "$1" = --out ]; then
  out="$2"; tag="$4"; sub="$5"
  case "$sub" in
    compare-clips) echo "{\"psnr_mean\":40.5,\"x\":1}" >&2; echo "$*" > "$out/compare-clips-$tag.json" ;;
    gate) echo "gate verdict: pass (speedup 1.2x)" >&2; echo "$*" > "$out/gate-$tag.json" ;;
  esac
  exit 0
fi
echo "args: $*"
echo "aux=$FV_AUX_DIR"
echo "peak vram 123 MiB" >&2
mkdir -p frames && : > frames/frame-000.png
echo '{"total_s":1}' > benchmark.json
[ -n "$FAIL" ] && { echo "boom" >&2; exit 3; }
exit 0
"#,
    );
    script(
        &dir.join("bin/nvidia-smi"),
        "case \"$1\" in\n  --query-gpu=memory.used) echo 0 ;;\n  -L) echo 'GPU 0: Fake' ;;\n  *) echo 'Fake, 1 MiB' ;;\nesac\n",
    );
    script(
        &dir.join("bin/timeout"),
        "echo \"$3\" > cap.txt\nshift 3\nexec \"$@\"\n",
    );
    std::fs::create_dir_all(dir.join("scratch/fv-libs")).unwrap();
    std::fs::write(dir.join("scratch/fv-libs/libcudnn.so"), "").unwrap();
    std::fs::create_dir_all(dir.join("work/weights")).unwrap();
    Fake { dir }
}

const CELL: &str = r#"
schema = 1
name = "t"
description = "fake"

[workload]
model = "h3"
seed = 1

[[arm]]
name = "a"
gate = "ok"
trees = ["w"]
argv = ["gen", "--weights", "{weights}/w", "--seed", "{seed}", "--clip-dir", "{runs}/{arm}/frames"]

[[arm]]
name = "b"
gate = "ok"
trees = ["w"]
env = ["FAIL=1"]
argv = ["gen", "--weights", "{weights}/w"]

[[arm]]
name = "c"
gate = "missing"
trees = ["w"]
argv = ["gen"]

[[compare]]
baseline = "a"
candidate = "b"
args = ["--off-identity"]

[[gate]]
baseline = "a"
candidate = "b"
kind = "exact"
"#;

fn config(f: &Fake) -> RunConfig {
    let mut c = RunConfig::new("fam", &f.dir.join("root"));
    c.work = f.dir.join("work");
    c.scratch = f.dir.join("scratch");
    c.tag = Some("t1".into());
    c.bin = f.dir.join("bin/fv-gpucheck");
    c.nvidia_smi = f.dir.join("bin/nvidia-smi").display().to_string();
    c.timeout = f.dir.join("bin/timeout").display().to_string();
    c.cell_timeout_s = 77;
    c.gpu_poll = Duration::from_millis(1);
    c
}

fn read(p: &Path) -> String {
    std::fs::read_to_string(p).unwrap_or_else(|e| panic!("{}: {e}", p.display()))
}

#[test]
fn writes_what_runpod_matrix_sh_writes() {
    let f = fake("files");
    let cell = CellFile::parse(CELL, "t.toml").unwrap();
    let cfg = config(&f);
    let steps = Lab::steps(&[&cell], &cfg.bin, &cfg.paths()).unwrap();
    let mut r = Runner::new(cfg);
    r.run(&steps).unwrap();
    let runs = f.dir.join("scratch/runs/fam/t1");
    assert_eq!(r.runs(), runs);

    let a = read(&runs.join("a/summary.json"));
    assert!(
        a.starts_with("{\"cell\":\"a\",\"family\":\"fam\",\"exit\":0,\"seconds\":"),
        "{a}"
    );
    assert!(a.ends_with("Z\"}\n"), "{a}");
    let out = read(&runs.join("a/stdout.log"));
    assert!(
        out.contains(&format!(
            "--weights {}/work/weights/w --seed 1",
            f.dir.display()
        )),
        "{out}"
    );
    assert!(
        out.contains(&format!("aux={}/work/weights/auxiliary", f.dir.display())),
        "{out}"
    );
    assert_eq!(read(&runs.join("a/cap.txt")), "77\n");
    for file in [
        "stderr.log",
        "nvidia-smi-start.txt",
        "nvidia-smi-end.txt",
        "benchmark.json",
        "frames/frame-000.png",
    ] {
        assert!(runs.join("a").join(file).is_file(), "a/{file}");
    }
    assert!(read(&runs.join("a.weights.log")).contains("verify ok in"));
    // The byte-identity record: the empty frame-000.png's SHA-256.
    assert_eq!(
        read(&runs.join("a/frames.sha256.json")),
        "{\"arm\":\"a\",\"frames\":[\n{\"path\":\"frame-000.png\",\"sha256\":\"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855\"}\n],\"count\":1}\n"
    );
    assert!(read(&runs.join("b/frames.sha256.json")).contains("\"count\":1}"));
    assert!(!runs.join("c/frames.sha256.json").exists());

    assert!(read(&runs.join("b/summary.json")).contains("\"exit\":3,"));
    assert_eq!(
        read(&runs.join("c/summary.json")),
        "{\"cell\":\"c\",\"family\":\"fam\",\"exit\":null,\"skipped\":\"weights incomplete\"}\n"
    );
    assert!(!runs.join("c/stdout.log").exists());

    let cmp = read(&runs.join("compare/compare-clips-a--b.json"));
    assert!(
        cmp.contains("compare-clips --baseline") && cmp.ends_with("--off-identity\n"),
        "{cmp}"
    );
    let gate = read(&runs.join("gate/gate-a--b.json"));
    assert!(gate.contains("--kind exact --compare"), "{gate}");
    assert!(gate.contains("compare-clips-a--b.json"), "{gate}");
    assert!(read(&runs.join("done.json")).starts_with("{\"family\":\"fam\",\"ended\":\""));

    let log = read(&runs.join("live.log"));
    for want in [
        "[fam] matrix start image=unknown",
        "GPU 0: Fake",
        "[fam] ▶ a  ",
        "(wall cap 77s; 0=none)",
        "[fam] ok a  ",
        "stderr.log:peak vram 123 MiB",
        "[fam] FAIL b  ",
        "exit=3",
        "boom",
        "[fam] SKIP c: weights incomplete",
        "INCOMPLETE missing",
        "[fam] compare a--b exit=0 \"psnr_mean\":40.5",
        "[fam] gate a--b exit=0 gate verdict: pass (speedup 1.2x)",
        "[fam] matrix done",
    ] {
        assert!(log.contains(want), "live.log lacks {want:?}:\n{log}");
    }
}

#[test]
fn fv_cells_and_missing_frames_and_benchmarks() {
    let f = fake("subset");
    let cell = CellFile::parse(CELL, "t.toml").unwrap();
    let mut cfg = config(&f);
    cfg.only = Some(vec!["a".into()]);
    let steps = Lab::steps(&[&cell], &cfg.bin, &cfg.paths()).unwrap();
    Runner::new(cfg).run(&steps).unwrap();
    let runs = f.dir.join("scratch/runs/fam/t1");
    assert!(runs.join("a/summary.json").is_file());
    assert!(!runs.join("b").exists() && !runs.join("c").exists());
    assert_eq!(
        read(&runs.join("compare/compare-clips-a--b.json")),
        "{\"stage\":\"compare-clips-a--b\",\"status\":\"skipped\",\"reason\":\"frames missing\"}\n"
    );
    let log = read(&runs.join("live.log"));
    assert!(log.contains("[fam] skip b (not in FV_CELLS)"));
    assert!(log.contains("[fam] skip compare a--b (frames missing)"));
    assert!(log.contains("[fam] skip gate a--b (benchmark.json missing)"));
}

#[test]
fn a_failing_nvidia_smi_is_fatal() {
    // The script runs `nvidia-smi -L | tee` under set -e -o pipefail.
    let f = fake("smi");
    script(
        &f.dir.join("bin/nvidia-smi"),
        "case \"$1\" in\n  -L) echo 'NVIDIA-SMI has failed' >&2; exit 9 ;;\n  *) echo 0 ;;\nesac\n",
    );
    let cell = CellFile::parse(CELL, "t.toml").unwrap();
    let cfg = config(&f);
    let steps = Lab::steps(&[&cell], &cfg.bin, &cfg.paths()).unwrap();
    let e = Runner::new(cfg).run(&steps).unwrap_err().to_string();
    assert!(e.contains("nvidia-smi -L failed (exit 9)"), "{e}");
    let runs = f.dir.join("scratch/runs/fam/t1");
    assert!(!runs.join("a").exists(), "no arm ran");
    assert!(!runs.join("done.json").exists());
}

#[test]
fn a_missing_binary_or_library_is_fatal() {
    let f = fake("fatal");
    let cell = CellFile::parse(CELL, "t.toml").unwrap();
    let mut cfg = config(&f);
    cfg.bin = f.dir.join("bin/nope");
    let steps = Lab::steps(&[&cell], &cfg.bin, &cfg.paths()).unwrap();
    let e = Runner::new(cfg).run(&steps).unwrap_err().to_string();
    assert!(e.contains("fv-gpucheck missing"), "{e}");

    let g = fake("nolib");
    std::fs::remove_file(g.dir.join("scratch/fv-libs/libcudnn.so")).unwrap();
    let cfg = config(&g);
    let steps = Lab::steps(&[&cell], &cfg.bin, &cfg.paths()).unwrap();
    let r = Runner::new(cfg).run(&steps);
    // No CUDA library on this host: the script's FATAL (unless the pins'
    // paths exist here, which a CPU build host does not have).
    if !Path::new("/lib/x86_64-linux-gnu/libcudnn.so.9").exists() {
        let e = r.unwrap_err().to_string();
        assert!(e.contains("libcudnn.so not linked"), "{e}");
    }
}

const PAIRS: &str = r#"
schema = 1
name = "p"
description = "baseline and tae arms"

[workload]
model = "h3"

[[arm]]
name = "x-base"
gate = "ok"
trees = ["w"]
bin = "baseline"
env = []
argv = ["gen", "--clip-dir", "{runs}/{arm}/frames"]

[[arm]]
name = "x-pr"
gate = "ok"
trees = ["w"]
env = []
argv = ["gen", "--clip-dir", "{runs}/{arm}/frames"]

[[arm]]
name = "x-tae"
gate = "ok"
trees = ["w"]
tae = "t1.safetensors"
argv = ["gen", "--tae", "{tae}/t1.safetensors"]

[[arm]]
name = "y-tae"
gate = "ok"
trees = ["w"]
tae = "t2.safetensors"
argv = ["gen", "--tae", "{tae}/t2.safetensors"]

[[compare]]
baseline = "x-base"
candidate = "x-pr"
args = ["--off-identity"]
gate = { kind = "exact" }
"#;

/// fetch-baseline.sh puts a binary that says so into <dest>; fetch-tae.sh
/// writes only t1 (t2 stays missing) and logs each call.
fn fetchers(f: &Fake, baseline_ok: bool) {
    let root = f.dir.join("root/scripts/gpu");
    let body = if baseline_ok {
        format!(
            "mkdir -p \"$2\" && cp {} \"$2/fv-gpucheck\" && echo \"base-$1\" > \"$2/fv-gpucheck.build-id\" && echo \"baseline ok: $1 into $2\"\n",
            f.dir.join("bin/fv-gpucheck").display()
        )
    } else {
        "echo \"fetch-baseline: no amd64 manifest for sha-$1\" >&2\nexit 1\n".to_owned()
    };
    script(&root.join("fetch-baseline.sh"), &body);
    script(
        &root.join("fetch-tae.sh"),
        "echo \"fetch-tae into $1\"\nmkdir -p \"$1\" && : > \"$1/t1.safetensors\"\n",
    );
}

#[test]
fn baseline_arms_and_tae_arms() {
    let f = fake("pairs");
    fetchers(&f, true);
    let cell = CellFile::parse(PAIRS, "p.toml").unwrap();
    let mut cfg = config(&f);
    cfg.baseline_sha = Some("7654321".into());
    let steps = Lab::steps(&[&cell], &cfg.bin, &cfg.paths()).unwrap();
    Runner::new(cfg).run(&steps).unwrap();
    let runs = f.dir.join("scratch/runs/fam/t1");
    let base = f.dir.join("scratch/baseline-7654321/fv-gpucheck");
    assert!(base.is_file(), "fetched into <scratch>/baseline-<sha>");
    assert!(read(&runs.join("baseline-fetch.log")).contains("baseline ok: 7654321"));
    // Both arms of the pair ran (the base one from the fetched copy), the
    // paired gate right after the compare.
    for arm in ["x-base", "x-pr", "x-tae"] {
        assert!(
            read(&runs.join(arm).join("summary.json")).contains("\"exit\":0,"),
            "{arm}"
        );
    }
    assert!(read(&runs.join("x-tae/stdout.log")).contains("/scratch/tae/t1.safetensors"));
    assert_eq!(
        read(&runs.join("y-tae/summary.json")),
        "{\"cell\":\"y-tae\",\"family\":\"fam\",\"exit\":null,\"skipped\":\"tae weights missing\"}\n"
    );
    // fetch-tae.sh ran once, for the first missing file only.
    assert_eq!(
        read(&runs.join("tae-fetch.log"))
            .matches("fetch-tae into")
            .count(),
        1
    );
    assert!(read(&runs.join("gate/gate-x-base--x-pr.json")).contains("--kind exact"));
    let log = read(&runs.join("live.log"));
    for want in [
        "[fam] baseline baseline ok: 7654321 into ",
        "[fam] baseline build-id base-7654321, image build-id unknown",
        "[fam] SKIP y-tae: ",
        "t2.safetensors missing (see tae-fetch.log)",
        "[fam] gate x-base--x-pr exit=0",
    ] {
        assert!(log.contains(want), "live.log lacks {want:?}:\n{log}");
    }
    let compare = log.find("[fam] compare x-base--x-pr").unwrap();
    let gate = log.find("[fam] gate x-base--x-pr").unwrap();
    assert!(compare < gate);
}

#[test]
fn an_unavailable_baseline_is_fatal_before_any_arm() {
    let f = fake("nobase");
    fetchers(&f, false);
    let cell = CellFile::parse(PAIRS, "p.toml").unwrap();
    let mut cfg = config(&f);
    cfg.baseline_sha = Some("7654321".into());
    let steps = Lab::steps(&[&cell], &cfg.bin, &cfg.paths()).unwrap();
    let e = Runner::new(cfg).run(&steps).unwrap_err().to_string();
    assert!(e.contains("baseline binary sha-7654321 unavailable"), "{e}");
    let runs = f.dir.join("scratch/runs/fam/t1");
    assert!(!runs.join("x-base").exists() && !runs.join("done.json").exists());
    let log = read(&runs.join("live.log"));
    assert!(log.contains("FATAL: baseline binary sha-7654321 unavailable (baseline-fetch.log)"));
    assert!(log.contains("no amd64 manifest for sha-7654321"), "{log}");
}

#[test]
fn a_baseline_that_is_this_build_is_fatal() {
    let f = fake("selfbase");
    fetchers(&f, true);
    // The image's build-id equals the one the fetched baseline carries.
    std::fs::write(f.dir.join("bin/fv-gpucheck.build-id"), "base-7654321\n").unwrap();
    let cell = CellFile::parse(PAIRS, "p.toml").unwrap();
    let mut cfg = config(&f);
    cfg.baseline_sha = Some("7654321".into());
    let steps = Lab::steps(&[&cell], &cfg.bin, &cfg.paths()).unwrap();
    let e = Runner::new(cfg).run(&steps).unwrap_err().to_string();
    assert!(e.contains("is this build (base-7654321)"), "{e}");
    let runs = f.dir.join("scratch/runs/fam/t1");
    assert!(!runs.join("x-base").exists());
}

/// The pod's memory (ADR-0019 §16.10) on a fake cgroup v2: `box.txt` gets the
/// RAM limit and the OOM kills at start (after the start command's lines),
/// each arm's `summary.json` its memory fields before `ended`, and an arm
/// the kernel OOM-killed says so in `live.log`.
#[test]
fn the_pods_ram_limit_and_an_arms_oom_kill_are_recorded() {
    let f = fake("mem");
    let cg = f.dir.join("cgroup");
    let events = cg.join("pod/memory.events");
    std::fs::create_dir_all(cg.join("pod")).unwrap();
    for (file, text) in [
        ("memory.current", "1\n"),
        ("memory.max", "206158430208\n"),
        ("memory.peak", "104857600\n"),
        ("memory.stat", "anon 52428800\nfile 1\nshmem 1048576\n"),
        ("memory.events", "oom 0\noom_kill 0\n"),
    ] {
        std::fs::write(cg.join("pod").join(file), text).unwrap();
    }
    std::fs::write(f.dir.join("proc-cgroup"), "0::/pod\n").unwrap();
    std::fs::write(f.dir.join("meminfo"), "MemTotal:       2113290240 kB\n").unwrap();
    // `oom` is killed as the kernel kills a loader over the limit: SIGKILL,
    // nothing printed, the cgroup's oom_kill counted.
    let cell = CellFile::parse(
        &format!(
            r#"
schema = 1
name = "t"
description = "fake"

[workload]
model = "h3"
seed = 1

[[arm]]
name = "ok"
gate = "ok"
trees = ["w"]
argv = ["gen", "--seed", "{{seed}}"]

[[arm]]
name = "oom"
gate = "ok"
trees = ["w"]
env = ["OOM_EVENTS={}"]
argv = ["gen"]
"#,
            events.display()
        ),
        "t.toml",
    )
    .unwrap();
    let gpucheck = f.dir.join("bin/fv-gpucheck");
    let body = read(&gpucheck);
    common::write_exe(
        &gpucheck,
        &body.replacen(
            "#!/bin/sh\n",
            "#!/bin/sh\nif [ -n \"$OOM_EVENTS\" ]; then printf 'oom 1\\noom_kill 1\\n' > \"$OOM_EVENTS\"; kill -9 $$; fi\n",
            1,
        ),
    );
    let mut cfg = config(&f);
    cfg.proc_cgroup = f.dir.join("proc-cgroup");
    cfg.cgroup_root = cg.clone();
    cfg.meminfo = f.dir.join("meminfo");
    cfg.mem_poll = Duration::from_millis(5);
    let runs = f.dir.join("scratch/runs/fam/t1");
    std::fs::create_dir_all(&runs).unwrap();
    std::fs::write(runs.join("box.txt"), "image_build_id=x\nFake, 1 MiB\n").unwrap();
    let steps = Lab::steps(&[&cell], &cfg.bin, &cfg.paths()).unwrap();
    Runner::new(cfg).run(&steps).unwrap();

    assert_eq!(
        read(&runs.join("box.txt")),
        format!(
            "image_build_id=x\nFake, 1 MiB\ncpus={} mem=2015G\ncgroup=v2 {} pod_ram_limit_mib=196608 oom_kills=0\n",
            std::thread::available_parallelism().map_or(0, |n| n.get()),
            cg.join("pod").display()
        )
    );
    let ok = read(&runs.join("ok/summary.json"));
    assert!(
        ok.starts_with("{\"cell\":\"ok\",\"family\":\"fam\",\"exit\":0,\"seconds\":")
            && ok.contains(",\"peak_ram_mib\":51,\"ram_limit_mib\":196608,\"pod_peak_mib_cumulative\":100,\"oom_kills\":0,\"ended\":\"")
            && ok.ends_with("Z\"}\n"),
        "{ok}"
    );
    let v: serde_json::Value = serde_json::from_str(&ok).unwrap();
    assert_eq!(v["exit"], 0);
    let oom: serde_json::Value =
        serde_json::from_str(&read(&runs.join("oom/summary.json"))).unwrap();
    assert_eq!(
        (oom["exit"].clone(), oom["oom_kills"].clone()),
        (137.into(), 1.into())
    );
    let log = read(&runs.join("live.log"));
    for want in [
        format!("[fam] pod memory: cgroup v2 {}, limit 196608 MiB, oom_kills 0", cg.join("pod").display()),
        "[fam] ram ok: ram 51 MiB of 196608 MiB, oom_kills 0".to_owned(),
        "[fam] FAIL oom  ".to_owned(),
        "exit=137".to_owned(),
        "[fam] ram oom: ram 51 MiB of 196608 MiB, oom_kills 1".to_owned(),
        "[fam] OOM oom: the kernel OOM-killed 1 process(es) in this pod during the arm (host RAM, not GPU".to_owned(),
    ] {
        assert!(log.contains(&want), "live.log lacks {want:?}:\n{log}");
    }
    assert!(!log.contains("OOM ok:"), "{log}");
}

/// Without a memory cgroup an arm's summary is as before (no memory fields)
/// and `box.txt` says so.
#[test]
fn without_a_memory_cgroup_the_summary_is_unchanged() {
    let f = fake("nomem");
    let cell = CellFile::parse(CELL, "t.toml").unwrap();
    let mut cfg = config(&f);
    cfg.proc_cgroup = f.dir.join("none");
    cfg.cgroup_root = f.dir.join("none");
    cfg.only = Some(vec!["a".into()]);
    let steps = Lab::steps(&[&cell], &cfg.bin, &cfg.paths()).unwrap();
    Runner::new(cfg).run(&steps).unwrap();
    let runs = f.dir.join("scratch/runs/fam/t1");
    let a = read(&runs.join("a/summary.json"));
    assert!(!a.contains("ram"), "{a}");
    assert!(
        a.starts_with("{\"cell\":\"a\",\"family\":\"fam\",\"exit\":0,\"seconds\":")
            && a.contains(",\"ended\":\""),
        "{a}"
    );
    assert!(
        read(&runs.join("box.txt")).contains("cgroup=none  pod_ram_limit_mib=null oom_kills=\n")
    );
    assert!(read(&runs.join("live.log")).contains("[fam] pod memory: no memory cgroup found"));
}
