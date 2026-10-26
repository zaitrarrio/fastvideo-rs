//! A `weights = "volume+r2"` run (docs/adr/0019-fv-lab.md §14) against fakes:
//!
//! - a fake `python3` standing in for `r2-weights.py plan` and `r2-fetch.py`
//!   (what the runner runs, in which order, with which environment; the
//!   merged root, the drops, the skips, where the key goes);
//! - a fake R2 (`scripts/gpu/tests/fake_r2.py`, the S3 stand-in the R2 tools'
//!   own tests use) with the real `r2-weights.py` and `r2-fetch.py`: a tree
//!   published there is planned, fetched part by part and read by the arm.
//!   Skipped without python3, as `fastvideo-deploy`'s `boot_r2` test.

use std::collections::BTreeMap;
use std::io::{BufRead as _, BufReader};
use std::os::unix::fs::PermissionsExt as _;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::Duration;

use fastvideo_lab::plan::{CellFile, Lab};
use fastvideo_lab::run::{dry_run_lines, RunConfig, Runner};
use fastvideo_lab::weights::{parse_r2_table, R2Key, R2Setup, TreeSources};

const KEY_ID: &str = "fake-access-key";
const KEY_SECRET: &str = "fake-secret";

fn script(path: &Path, body: &str) {
    if let Some(d) = path.parent() {
        std::fs::create_dir_all(d).unwrap();
    }
    std::fs::write(path, format!("#!/bin/sh\n{body}")).unwrap();
    std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o755)).unwrap();
}

fn read(p: &Path) -> String {
    std::fs::read_to_string(p).unwrap_or_else(|e| panic!("{}: {e}", p.display()))
}

struct Fake {
    dir: PathBuf,
}

impl Drop for Fake {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

/// The runner's world: a root with a fake verify-weights.sh (a tree passes
/// when `$FV_WEIGHTS/<tree>/.complete` exists), a fake fv-gpucheck that
/// prints its arguments and whether it sees the R2 key, nvidia-smi, timeout,
/// and the volume at `work/weights` with the tree `w`.
fn fake(name: &str) -> Fake {
    let dir = std::env::temp_dir().join(format!("fv-lab-r2-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let root = dir.join("root");
    script(
        &root.join("scripts/gpu/verify-weights.sh"),
        "echo \"verify $1 in $FV_WEIGHTS\"\nfor t in $(echo \"$1\" | tr : ' '); do\n  [ -f \"$FV_WEIGHTS/$t/.complete\" ] || { echo \"INCOMPLETE $t\"; exit 1; }\ndone\nexit 0\n",
    );
    std::fs::write(root.join("scripts/gpu/cuda-13.pins"), "# none\n").unwrap();
    script(
        &dir.join("bin/fv-gpucheck"),
        "echo \"args: $*\"\necho \"key=${FV_WEIGHTS_R2_ACCESS_KEY_ID:-none} secret=${FV_WEIGHTS_R2_SECRET_ACCESS_KEY:-none}\"\nexit 0\n",
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
    let w = dir.join("work/weights/w");
    std::fs::create_dir_all(&w).unwrap();
    std::fs::write(w.join(".complete"), "").unwrap();
    Fake { dir }
}

fn config(f: &Fake, r2: R2Setup) -> RunConfig {
    let mut c = RunConfig::new("fam", &f.dir.join("root"));
    c.work = f.dir.join("work");
    c.scratch = f.dir.join("scratch");
    c.tag = Some("t1".into());
    c.bin = f.dir.join("bin/fv-gpucheck");
    c.nvidia_smi = f.dir.join("bin/nvidia-smi").display().to_string();
    c.timeout = f.dir.join("bin/timeout").display().to_string();
    c.cell_timeout_s = 77;
    c.gpu_poll = Duration::from_millis(1);
    c.r2 = Some(r2);
    c
}

fn key() -> Option<R2Key> {
    Some(R2Key {
        id: KEY_ID.into(),
        secret: KEY_SECRET.into(),
    })
}

/// Every file under `dir` (recursive, symlinks not followed).
fn files(dir: &Path, out: &mut Vec<PathBuf>) {
    for e in std::fs::read_dir(dir).into_iter().flatten().flatten() {
        let p = e.path();
        match e.file_type() {
            Ok(t) if t.is_dir() => files(&p, out),
            Ok(t) if t.is_file() => out.push(p),
            _ => {}
        }
    }
}

// ------------------------------------------------------------ fake python

const TABLE: &str = "big\trev1\t1\t3000000000\tsha\thub\nsmall\trev2\t1\t1000000000\tsha\tdir\nbad\trev3\t1\t5\tsha\tdir\n";

const CELL: &str = r#"
schema = 1
name = "c"
description = "volume + r2"

[workload]
model = "engine"
seed = 1

[[arm]]
name = "vol"
gate = "w"
trees = ["w"]
argv = ["gen", "--weights-root", "{weights}", "--seed", "{seed}"]

[[arm]]
name = "one"
gate = "w:big"
trees = ["w", "big"]
argv = ["gen", "--weights-root", "{weights}"]

[[arm]]
name = "again"
gate = "big"
trees = ["big/sub/file.safetensors"]
argv = ["gen", "--weights-root", "{weights}"]

[[arm]]
name = "two"
gate = "small"
trees = ["small"]
argv = ["gen", "--weights-root", "{weights}"]

[[arm]]
name = "broken"
gate = "bad"
trees = ["bad"]
argv = ["gen", "--weights-root", "{weights}"]

[[arm]]
name = "nowhere"
gate = "gone"
trees = ["gone"]
argv = ["gen", "--weights-root", "{weights}"]
"#;

/// A fake `python3`: `-c` is the runner's python check (exit 1 until the
/// fake apt-get installed it, when `nopython` exists); `r2-weights.py plan`
/// writes its `dest@rev` arguments as the plan; `r2-fetch.py` makes each
/// planned tree complete under `--root` (and fails on `bad`). Every call is
/// recorded with the key it saw.
fn fake_python(f: &Fake) -> String {
    let rec = f.dir.join("python.log");
    let gate = f.dir.join("nopython");
    let py = f.dir.join("bin/python3");
    script(
        &py,
        &format!(
            r#"rec={rec}
a="${{1:-}}"; echo "${{a##*/}} key=${{FV_WEIGHTS_R2_ACCESS_KEY_ID:-none}} secret=${{FV_WEIGHTS_R2_SECRET_ACCESS_KEY:-none}} $*" >> "$rec"
case "$1" in
  -c) [ -f {gate} ] && exit 1; exit 0 ;;
  */r2-weights.py)
    shift; [ "$1" = plan ] || exit 2; shift
    out=""; specs=""
    while [ $# -gt 0 ]; do
      case "$1" in --ttl) shift ;; --out) out="$2"; shift ;; *) specs="$specs $1" ;; esac
      shift
    done
    for s in $specs; do echo "$s"; done > "$out"
    echo "plan: $(echo $specs | wc -w) tree(s) -> $out" >&2 ;;
  */r2-fetch.py)
    shift; root=""; plan=""; status=""
    while [ $# -gt 0 ]; do
      case "$1" in --root) root="$2"; shift ;; --plan) plan="$2"; shift ;; --status) status="$2"; shift ;; esac
      shift
    done
    echo '{{"state":"done"}}' > "$status"
    while read -r s; do
      d="${{s%@*}}"; r="${{s#*@}}"
      if [ "$d" = bad ]; then mkdir -p "$root/.r2-partial/bad@x"; echo "r2-fetch: FAILED: part 0 of bad"; exit 1; fi
      mkdir -p "$root/$d/sub"; : > "$root/$d/.complete"; echo "$r" > "$root/$d/.fv-revision"; : > "$root/$d/sub/file.safetensors"
    done < "$plan"
    echo "weights ready: $(wc -l < "$plan") tree(s), 1.00 GB fetched in 1.0s (1000 MB/s); retries 0" >&2 ;;
esac
"#,
            rec = rec.display(),
            gate = gate.display()
        ),
    );
    script(
        &f.dir.join("bin/apt-get"),
        &format!(
            "echo \"apt-get $*\" >> {}\n[ \"$5\" = install ] && rm -f {}\nexit 0\n",
            rec.display(),
            gate.display()
        ),
    );
    py.display().to_string()
}

fn setup(f: &Fake, key: Option<R2Key>) -> R2Setup {
    let python = fake_python(f);
    let sources = TreeSources {
        volume: vec!["w".into()],
        r2: parse_r2_table(TABLE).unwrap(),
    };
    let mut r = R2Setup::new(sources, &f.dir.join("scratch"), key);
    r.python = python;
    r.apt_get = f.dir.join("bin/apt-get").display().to_string();
    r
}

#[test]
fn volume_trees_are_linked_r2_trees_fetched_per_arm_and_the_key_reaches_the_plan_only() {
    let f = fake("orchestration");
    let cell = CellFile::parse(CELL, "c.toml").unwrap();
    let cfg = config(&f, setup(&f, key()));
    let root = f.dir.join("scratch/weights");
    assert_eq!(cfg.paths().weights, root);
    let steps = Lab::steps(&[&cell], &cfg.bin, &cfg.paths()).unwrap();
    // The dry run names every step of the plan, and runs nothing.
    let dry = dry_run_lines(&cfg, &steps).join("\n");
    let wants = [
        format!(
            "[fam] dry-run weights {}: 1 volume trees linked from {}/work/weights; 3 R2 trees (4.00 GB) fetched before the arms that read them, at most 3.00 GB on the container disk at once",
            root.display(),
            f.dir.display()
        ),
        format!("[fam] dry-run r2 one: fetch big@rev1 (3.00 GB) into {}", root.display()),
        "[fam] dry-run r2 two: drop big (3.00 GB); fetch small@rev2 (1.00 GB)".to_owned(),
        "[fam] dry-run r2 broken: drop small (1.00 GB); fetch bad@rev3 (0.00 GB)".to_owned(),
    ];
    for want in &wants {
        assert!(dry.contains(want), "{want}\n{dry}");
    }
    assert!(!dry.contains("dry-run r2 again") && !dry.contains("dry-run r2 nowhere"));
    assert!(!f.dir.join("python.log").exists());

    let mut r = Runner::new(cfg);
    r.run(&steps).unwrap();
    let runs = f.dir.join("scratch/runs/fam/t1");

    // The merged root: the volume's tree as a symlink, the fetched trees as
    // directories; `big` was dropped before `small` was fetched.
    let link = std::fs::symlink_metadata(root.join("w")).unwrap();
    assert!(link.file_type().is_symlink());
    assert_eq!(
        std::fs::read_link(root.join("w")).unwrap(),
        f.dir.join("work/weights/w")
    );
    assert!(
        f.dir.join("work/weights/w/.complete").is_file(),
        "the volume is untouched"
    );
    // `broken` dropped `small` before its own fetch failed.
    assert!(!root.join("big").exists() && !root.join("small").exists());
    assert!(
        !root.join(".r2-partial").exists(),
        "a failed fetch's staging is gone"
    );

    let summary = |a: &str| read(&runs.join(a).join("summary.json"));
    for a in ["vol", "one", "again", "two"] {
        assert!(summary(a).contains("\"exit\":0,"), "{a}: {}", summary(a));
    }
    assert_eq!(
        summary("broken"),
        "{\"cell\":\"broken\",\"family\":\"fam\",\"exit\":null,\"skipped\":\"r2 fetch failed\"}\n"
    );
    assert!(summary("nowhere").contains("\"skipped\":\"weights incomplete\""));
    let out = read(&runs.join("one/stdout.log"));
    assert!(
        out.contains(&format!("args: gen --weights-root {}", root.display())),
        "{out}"
    );
    assert!(
        out.contains("key=none secret=none"),
        "an arm never sees the key: {out}"
    );
    assert!(read(&runs.join("one.weights.log")).contains(&format!("in {}", root.display())));
    assert_eq!(
        read(&runs.join("one/r2-fetch.json")),
        "{\"state\":\"done\"}\n"
    );
    assert!(read(&runs.join("one/r2-fetch.log")).contains("weights ready: 1 tree(s)"));

    // The calls, in order: the check, then per fetching arm the plan (with
    // the key) and the fetch (without it), under the arm's cap.
    let calls = read(&f.dir.join("python.log"));
    let lines: Vec<&str> = calls.lines().collect();
    assert_eq!(lines.len(), 7, "{calls}");
    assert!(lines[0].starts_with("-c key=none"), "{calls}");
    assert!(lines[1].starts_with(&format!("r2-weights.py key={KEY_ID} secret={KEY_SECRET} ")));
    assert!(
        lines[1].contains(" plan big@rev1 --ttl 677 --out "),
        "{calls}"
    );
    assert!(
        lines[2].starts_with("r2-fetch.py key=none secret=none "),
        "{calls}"
    );
    assert!(lines[3].contains(" plan small@rev2 "), "{calls}");
    assert!(lines[5].contains(" plan bad@rev3 "), "{calls}");
    assert_eq!(read(&f.dir.join("scratch/r2/cap.txt")), "77\n");
    // The plan (presigned URLs on a pod) never outlives its fetch, and the
    // key is in no file of the run directory.
    assert!(!f.dir.join("scratch/r2/plan.json").exists());
    let mut all = Vec::new();
    files(&runs, &mut all);
    for p in &all {
        let t = std::fs::read_to_string(p).unwrap_or_default();
        assert!(
            !t.contains(KEY_SECRET) && !t.contains(KEY_ID),
            "{}",
            p.display()
        );
    }
    let log = read(&runs.join("live.log"));
    for want in [
        "r2 one: fetch big@rev1 (3.00 GB) into",
        "r2 one: weights ready: 1 tree(s)",
        "r2 two: drop big (3.00 GB)",
        "SKIP broken: r2 fetch failed: r2-fetch.py: r2-fetch: FAILED: part 0 of bad",
        "SKIP nowhere: weights incomplete",
    ] {
        assert!(log.contains(want), "{want}\n{log}");
    }
    assert!(
        !log.contains("r2 again"),
        "a tree already fetched is read again: {log}"
    );
}

#[test]
fn without_a_key_r2_arms_are_skipped_and_volume_arms_run() {
    let f = fake("nokey");
    let cell = CellFile::parse(CELL, "c.toml").unwrap();
    let cfg = config(&f, setup(&f, None));
    let steps = Lab::steps(&[&cell], &cfg.bin, &cfg.paths()).unwrap();
    let mut r = Runner::new(cfg);
    r.run(&steps).unwrap();
    let runs = f.dir.join("scratch/runs/fam/t1");
    assert!(read(&runs.join("vol/summary.json")).contains("\"exit\":0,"));
    for a in ["one", "two", "broken"] {
        assert!(
            read(&runs.join(a).join("summary.json")).contains("\"skipped\":\"r2 fetch failed\""),
            "{a}"
        );
    }
    // "again" reads big, which never came: its own fetch is refused too.
    assert!(read(&runs.join("again/summary.json")).contains("r2 fetch failed"));
    let log = read(&runs.join("live.log"));
    assert!(log.contains("WARN: no R2 key in the environment (FV_WEIGHTS_R2_ACCESS_KEY_ID / FV_WEIGHTS_R2_SECRET_ACCESS_KEY, from the Runpod secrets fv_r2_access_key_id / fv_r2_secret_access_key)"), "{log}");
    assert!(
        !f.dir.join("python.log").exists(),
        "nothing ran without a key"
    );
}

#[test]
fn a_missing_python3_is_installed_once_with_apt() {
    let f = fake("apt");
    let cell = CellFile::parse(CELL, "c.toml").unwrap();
    let cfg = config(&f, setup(&f, key()));
    std::fs::write(f.dir.join("nopython"), "").unwrap();
    let steps = Lab::steps(&[&cell], &cfg.bin, &cfg.paths()).unwrap();
    let mut r = Runner::new(cfg);
    r.run(&steps).unwrap();
    let calls = read(&f.dir.join("python.log"));
    assert_eq!(calls.matches("apt-get ").count(), 2, "{calls}");
    assert!(calls.contains("update -qq"), "{calls}");
    assert!(
        calls.contains("install -y -qq --no-install-recommends python3"),
        "{calls}"
    );
    assert_eq!(
        calls.matches("-c key=").count(),
        2,
        "checked before and after: {calls}"
    );
    let runs = f.dir.join("scratch/runs/fam/t1");
    assert!(read(&runs.join("two/summary.json")).contains("\"exit\":0,"));
    assert!(read(&runs.join("live.log")).contains("r2: no python3 in the image"));
}

#[test]
fn an_r2_tree_holding_a_volume_tree_is_fatal_before_any_arm() {
    let f = fake("fatal");
    let mut r2 = setup(&f, key());
    r2.sources.volume = vec!["big/inner".into(), "w".into()];
    let cell = CellFile::parse(CELL, "c.toml").unwrap();
    let mut cfg = config(&f, r2);
    cfg.only = Some(vec!["two".into()]);
    let steps = Lab::steps(&[&cell], &cfg.bin, &cfg.paths()).unwrap();
    // Only `two` selected: its tree is fine.
    assert!(!dry_run_lines(&cfg, &steps).join("\n").contains("FATAL"));
    cfg.only = None;
    let dry = dry_run_lines(&cfg, &steps).join("\n");
    assert!(
        dry.contains("FATAL: arm one: the R2 tree big holds the volume tree big/inner"),
        "{dry}"
    );
    let mut r = Runner::new(cfg);
    assert!(r.run(&steps).is_err());
    assert!(!f.dir.join("scratch/runs/fam/t1/vol").exists());
}

// --------------------------------------------------------------- fake R2

/// The fake R2 with one published tree: `fake_r2.serve()` and the real
/// publisher (`r2-weights.py`'s `Publisher`, small parts), in a python3
/// process that prints the endpoint and the tree's bytes, then serves until
/// its stdin closes.
struct FakeR2 {
    child: std::process::Child,
    base: String,
    bytes: u64,
}

impl Drop for FakeR2 {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn repo() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .unwrap()
}

fn fake_r2(src: &Path) -> FakeR2 {
    let gpu = repo().join("scripts/gpu");
    let driver = format!(
        r#"
import importlib.util, os, sys
sys.path.insert(0, {tests:?})
import fake_r2
spec = importlib.util.spec_from_file_location("r2_weights", {w:?})
W = importlib.util.module_from_spec(spec)
spec.loader.exec_module(W)
srv, st, base = fake_r2.serve()
s3 = W.S3(base, "fv-weights", fake_r2.ACCESS, fake_r2.SECRET)
args = type("A", (), dict(prefix="", files=2, window=2, dry_run=False))()
pub = W.Publisher(s3, args)
pub.part_size = 64 << 10
pub.publish("demo", "r1", {{"kind": "dir", "path": {src:?}}}, W.dir_sources("demo", {src:?}))
m = W.load_manifest(s3, "", "demo", "r1")
print(base, m["total_bytes"], flush=True)
sys.stdin.read()
"#,
        tests = gpu.join("tests").display().to_string(),
        w = gpu.join("r2-weights.py").display().to_string(),
        src = src.display().to_string(),
    );
    let mut child = Command::new("python3")
        .args(["-I", "-c", &driver])
        .env("NO_PROXY", "127.0.0.1,localhost")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .expect("python3");
    let mut line = String::new();
    BufReader::new(child.stdout.take().expect("stdout"))
        .read_line(&mut line)
        .expect("the fake R2's endpoint");
    let mut w = line.split_whitespace();
    let base = w.next().expect("endpoint").to_owned();
    let bytes = w.next().and_then(|b| b.parse().ok()).expect("bytes");
    FakeR2 { child, base, bytes }
}

#[test]
fn a_tree_published_in_a_fake_r2_is_planned_fetched_and_read_by_its_arm() {
    let have_python = Command::new("python3")
        .arg("--version")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .is_ok_and(|s| s.success());
    if !have_python {
        eprintln!("skipped: no python3 on PATH");
        return;
    }
    let f = fake("fake-r2");
    // The tree: a config, a shard of several 64 KiB parts, an empty file.
    let src = f.dir.join("src");
    std::fs::create_dir_all(src.join("transformer")).unwrap();
    std::fs::write(src.join("config.json"), b"{\"a\": 1}").unwrap();
    let shard: Vec<u8> = (0..300_000u32).map(|i| (i * 7 % 251) as u8).collect();
    std::fs::write(src.join("transformer/model.safetensors"), &shard).unwrap();
    std::fs::write(src.join("transformer/empty.bin"), b"").unwrap();
    let r2 = fake_r2(&src);
    // The real tools, from a root whose scripts/gpu holds them (they find
    // their TSVs beside their real path) and the fake gate.
    let gpu = f.dir.join("root/scripts/gpu");
    for t in ["r2-weights.py", "r2-fetch.py"] {
        std::os::unix::fs::symlink(repo().join("scripts/gpu").join(t), gpu.join(t)).unwrap();
    }
    // python3 with the fake R2's endpoint and no proxy.
    let py = f.dir.join("bin/py");
    script(
        &py,
        &format!(
            "exec env -u HTTPS_PROXY -u https_proxy -u HTTP_PROXY -u http_proxy NO_PROXY=127.0.0.1,localhost FV_WEIGHTS_R2_ENDPOINT={} python3 \"$@\"\n",
            r2.base
        ),
    );
    let table = format!("demo\tr1\t3\t{}\tsha\tdir\n", r2.bytes);
    let mut setup = R2Setup::new(
        TreeSources {
            volume: vec!["w".into()],
            r2: parse_r2_table(&table).unwrap(),
        },
        &f.dir.join("scratch"),
        Some(R2Key {
            id: "fake-access-key".into(),
            secret: "fake-secret".into(),
        }),
    );
    setup.python = py.display().to_string();
    let cell = CellFile::parse(
        r#"
schema = 1
name = "c"
description = "fake R2"

[workload]
model = "engine"

[[arm]]
name = "demo-arm"
gate = "w:demo"
trees = ["w", "demo"]
argv = ["gen", "--weights-root", "{weights}"]
"#,
        "c.toml",
    )
    .unwrap();
    let cfg = config(&f, setup);
    let steps = Lab::steps(&[&cell], &cfg.bin, &cfg.paths()).unwrap();
    let mut r = Runner::new(cfg);
    r.run(&steps).unwrap();
    let runs = f.dir.join("scratch/runs/fam/t1");
    let fetch_log = std::fs::read_to_string(runs.join("demo-arm/r2-fetch.log")).unwrap_or_default();
    let summary = read(&runs.join("demo-arm/summary.json"));
    assert!(summary.contains("\"exit\":0,"), "{summary}\n{fetch_log}");
    let tree = f.dir.join("scratch/weights/demo");
    assert_eq!(
        std::fs::read(tree.join("transformer/model.safetensors")).unwrap(),
        shard
    );
    assert_eq!(read(&tree.join("config.json")), "{\"a\": 1}");
    assert_eq!(read(&tree.join(".fv-revision")).trim(), "r1");
    assert!(tree.join(".complete").is_file());
    let status: BTreeMap<String, serde_json::Value> =
        serde_json::from_str(&read(&runs.join("demo-arm/r2-fetch.json"))).unwrap();
    assert_eq!(status["state"], "done", "{status:?}");
    assert!(fetch_log.contains("weights ready"), "{fetch_log}");
    assert!(
        !fetch_log.contains("X-Amz-Signature"),
        "no presigned URL in the log"
    );
    assert!(!f.dir.join("scratch/r2/plan.json").exists());
    assert!(read(&runs.join("live.log")).contains("r2 demo-arm: weights ready"));
}
