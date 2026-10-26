//! D1d's second slice (docs/adr/0019-fv-lab.md §15) against the script it
//! moves: `runpod-matrix.sh det-short`, `determinism` and `oracle` (copied
//! with what they source into a fake root) and `fv-lab run` of the committed
//! tiers, on one fake box: a fake fv-gpucheck that writes
//! frames (per prompt-set run, with a warm-up directory), audio, stage digests
//! and tensor dumps and answers `compare-dumps`, a fake nvidia-smi and
//! verify-weights.sh (the trees the EU volume lacks fail), and an upstream
//! reference served under a `file://` URL (curl reads it as it reads the
//! upstream pod's HTTP server).
//!
//! Both runners must write the same hashes (`<arm>/hashes.json`,
//! `<arm>/hashes.txt`, `<runs>/hashes.txt`), the same `determinism.txt`, the
//! same stage digests and prompt-set files, keep the same frames, record the
//! same exits and skips with byte-identical frames records, echo the same
//! `compare-dumps` lines and remove the dumps and the reference after an
//! oracle target. `fv-lab gate` must derive the reports again from either run
//! directory (`--check`: no difference) and decide the typed verdicts.
//!
//! `det-short` and `determinism` are retired (ADR-0019 §15.10: their GPU
//! checks passed on `f93a961`, their branches are deleted from
//! `scripts/gpu/runpod-matrix.sh`). Their script side is frozen as
//! `tests/verdict.rs` froze `b4a`'s: `fixtures/det-script/` holds the script
//! as it last ran them, at main `0f63c56b` (the commit before the deletion):
//! `runpod-matrix.sh` cut to its shared frame, the `determinism` and
//! `det-short` branches and the fallback branch, every kept line unchanged
//! (`git show 0f63c56b:scripts/gpu/runpod-matrix.sh | sed -n
//! '1,454p;2114,2327p;3093,3119p'`), and that commit's `b4-verdict.sh`,
//! `cuda-13.pins` and `fv-lab-retired.sh` whole. So a change to fv-lab's
//! `[repeats]` step is still held to the script's bytes, and a change to
//! `scripts/gpu/` no longer moves the reference. `oracle`'s branch is live
//! (it stays whole, §15.10) and is copied from this checkout. The frozen
//! `determinism` branch's oracle prelude (`FV_ORACLE_URL` set: `bash
//! runpod-matrix.sh oracle`) would hit "unknown family" in the cut copy,
//! which has no `oracle` branch; no test sets `FV_ORACLE_URL` for it.

mod common;

use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Duration;

use fastvideo_lab::plan::Lab;
use fastvideo_lab::run::{gate_selection, Decision, RunConfig, Runner};

const TAG: &str = "t1";

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// A `#!/bin/sh` fake tool, written so no exec of it can race a fork
/// (`common::write_exe`).
fn script(path: &Path, body: &str) {
    common::write_exe(path, &format!("#!/bin/sh\n{body}"));
}

/// fv-gpucheck on the fake box. `compare-dumps --baseline A --candidate B`:
/// one line per `*.bin` of A, equal or not. A generation: its dumps under
/// `FASTVIDEO_DUMP_DIR`, stage digests with `FASTVIDEO_DIGEST=1`, and frames
/// (`frame-000`, `001`, `072`, `073`) in each prompt-set run directory (or the
/// clip directory) and a `warmup/`, `audio.wav` for LTX. The bytes depend on
/// the timed pick (`FASTVIDEO_SDPA_AUTO`) and f32 activations, so the timed
/// control differs; the mode word `flaky` makes `h3-r3` and `ltx-t2v-c` differ.
const GPUCHECK: &str = r#"case " $* " in *" compare-clips "*)
  # `--out D --tag T compare-clips --baseline B --candidate C`: a report of
  # both directories' PNG bytes (ADR-0019 §21: sfwan13's frame compares).
  out=""; tag=""; b=""; c=""; prev=""
  for a in "$@"; do
    case "$prev" in --out) out="$a" ;; --tag) tag="$a" ;; --baseline) b="$a" ;; --candidate) c="$a" ;; esac
    prev="$a"
  done
  sb="$(cat "$b"/*.png | cksum | cut -d' ' -f1)"; sc="$(cat "$c"/*.png | cksum | cut -d' ' -f1)"
  printf '{"stage":"compare-clips-%s","baseline":%s,"candidate":%s}\n' "$tag" "$sb" "$sc" >"$out/compare-clips-$tag.json"
  echo "\"psnr_mean\":$sb.$sc," >&2
  exit 0 ;;
esac
if [ "$1" = compare-dumps ]; then
  for f in "$3"/*.bin; do
    n="$(basename "$f" .bin)"
    if cmp -s "$f" "$5/$n.bin"; then echo "compare-dumps $n rel_l2 0.000e0 cosine 1.000000" >&2
    else echo "compare-dumps $n rel_l2 1.000e-3 cosine 0.999990" >&2; fi
  done
  echo "[INFO] compared" >&2
  exit 0
fi
mode="$(cat @MODE@ 2>/dev/null)"
clip=""; prompts=""; prev=""
for a in "$@"; do
  case "$prev" in --clip-dir | --clip) clip="$a" ;; --prompts) prompts="$a" ;; esac
  prev="$a"
done
salt="${FASTVIDEO_SDPA_AUTO:-rule}"
[ -n "${FASTVIDEO_BF16_ACT:-}" ] && salt="$salt-f32"
case " $mode " in *" flaky "*) case "$clip" in *h3-r3* | *ltx-t2v-c/*) salt="$salt-flaky" ;; esac ;; esac
audio=0
case " $* " in *" ltx2 "*) audio=1 ;; esac
if [ -n "${FASTVIDEO_DUMP_DIR:-}" ]; then
  mkdir -p "$FASTVIDEO_DUMP_DIR"
  printf 'latent-%s' "$salt" >"$FASTVIDEO_DUMP_DIR/latent.bin"
  printf 'noise' >"$FASTVIDEO_DUMP_DIR/noise.bin"
fi
[ "${FASTVIDEO_DIGEST:-}" = 1 ] && echo "[INFO] [fastvideo] stage denoise digest sha=$salt" >&2
run() {
  mkdir -p "$1"
  for i in 000 001 072 073; do printf '%s-%s' "$salt" "$i" >"$1/frame-$i.png"; done
  [ "$audio" = 1 ] && printf 'wav-%s' "$salt" >"$1/audio.wav"
  return 0
}
if [ -n "$prompts" ]; then
  for n in $(grep -o '"name": "[^"]*"' "$prompts" | sed 's/.*: "//; s/"$//'); do run "$clip/$n"; done
else
  run "$clip"
fi
run "$clip/warmup"
# An LTX clip's video (the clip-metric targets read it, ADR-0019 §20).
if [ "$audio" = 1 ] && [ -z "$prompts" ] && [ -n "$clip" ]; then
  printf 'mp4-%s-%s%s' "$salt" "${FASTVIDEO_INJECT_COND:-i}" "${FASTVIDEO_INJECT_PIXELS:-i}" >"$clip/output.mp4"
fi
# A Wan clip's video (`wan gen --clip-dir` writes output.mp4; sfwan13, §21).
case " $* " in *" wan gen "*) [ -n "$clip" ] && printf 'mp4-%s-%s%s' "$salt" "${FASTVIDEO_WAN_CAUSAL_AR:-a}" "${FASTVIDEO_WAN_CAUSAL_FLASH:-f}" >"$clip/output.mp4" ;; esac
echo "peak vram 1 MiB" >&2
exit 0
"#;

struct Fake {
    dir: PathBuf,
}

impl Drop for Fake {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

impl Fake {
    fn root(&self) -> PathBuf {
        self.dir.join("root")
    }
    fn bin(&self) -> PathBuf {
        self.dir.join("bin")
    }
    fn up(&self) -> PathBuf {
        self.dir.join("upstream")
    }
}

/// runpod-matrix.sh and what it sources, from this checkout for `oracle`
/// (its branch is live). `b4-verdict.sh` went with the B4 branches
/// (ADR-0019 §18), so the live script no longer sources it.
const SOURCED: [&str; 3] = ["runpod-matrix.sh", "cuda-13.pins", "fv-lab-retired.sh"];

/// The same from `fixtures/det-script/` (frozen at `0f63c56b`) for the
/// retired `det-short` and `determinism`: that copy still sources its
/// `b4-verdict.sh`.
const SOURCED_FROZEN: [&str; 4] = [
    "runpod-matrix.sh",
    "b4-verdict.sh",
    "cuda-13.pins",
    "fv-lab-retired.sh",
];

/// Where a family's script comes from, and its files: the frozen copy for
/// the retired ones.
fn script_source(family: &str) -> (PathBuf, &'static [&'static str]) {
    match family {
        "det-short" | "determinism" => (
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/det-script"),
            &SOURCED_FROZEN,
        ),
        _ => (repo_root().join("scripts/gpu"), &SOURCED),
    }
}

fn fake(name: &str, mode: &str) -> Fake {
    let dir = std::env::temp_dir().join(format!("fv-lab-d1d-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let f = Fake { dir };
    let gpu = f.root().join("scripts/gpu");
    std::fs::create_dir_all(&gpu).unwrap();
    // `name` starts with the family (det-short…, determinism…, oracle…).
    let family = ["det-short", "determinism", "oracle"]
        .into_iter()
        .find(|fam| name.starts_with(fam))
        .expect("a fake is named after its family");
    let (src, files) = script_source(family);
    for s in files {
        std::fs::copy(src.join(s), gpu.join(s)).unwrap();
    }
    std::fs::write(f.dir.join("mode"), mode).unwrap();
    // The EU volume lacks h3-8step and fastwan22-ti2v-5b.
    script(
        &gpu.join("verify-weights.sh"),
        &format!(
            "echo \"verify $1\"\ncase \"$1\" in fasth3-8step | fastwan22-ti2v-5b) exit 1 ;; esac\ngrep -qw noh3 {m} && [ \"$1\" = fasth3-4step-vsa ] && exit 1\ngrep -qw \"no-$1\" {m} && exit 1\nexit 0\n",
            m = f.dir.join("mode").display()
        ),
    );
    std::fs::write(gpu.join("gate-policy.toml"), "# policy\n").unwrap();
    script(
        &f.bin().join("fv-gpucheck"),
        &GPUCHECK.replace("@MODE@", &f.dir.join("mode").display().to_string()),
    );
    script(
        &f.bin().join("nvidia-smi"),
        "case \"$1\" in\n  --query-gpu=memory.used) echo 0 ;;\n  -L) echo 'GPU 0: Fake' ;;\n  *) echo 'Fake, 1 MiB' ;;\nesac\n",
    );
    for side in ["script", "lab"] {
        let libs = f.dir.join(side).join("fv-libs");
        std::fs::create_dir_all(&libs).unwrap();
        for l in ["libcudnn.so", "libcupti.so"] {
            std::fs::write(libs.join(l), "").unwrap();
        }
    }
    std::fs::create_dir_all(f.dir.join("work/weights")).unwrap();
    // The upstream pod: oracle-fasth3-4step-vsa finished and served; the
    // other targets never finish.
    let t = f.up().join("oracle-fasth3-4step-vsa");
    std::fs::create_dir_all(t.join("dump")).unwrap();
    std::fs::write(t.join("dump/latent.bin"), "latent-rule").unwrap();
    std::fs::write(t.join("dump/noise.bin"), "noise").unwrap();
    std::fs::write(t.join("dump/extra.bin"), "x").unwrap();
    let st = Command::new("tar")
        .arg("-cf")
        .arg(t.join("oracle-dump.tar"))
        .arg("-C")
        .arg(&t)
        .arg("dump")
        .status()
        .expect("tar");
    assert!(st.success());
    std::fs::remove_dir_all(t.join("dump")).unwrap();
    std::fs::write(t.join("ORACLE_DONE"), "").unwrap();
    f
}

fn url(f: &Fake) -> String {
    format!("file://{}", f.up().display())
}

const TARGETS: &str = "fasth3-4step-vsa ltx25-512p";

fn script_run(f: &Fake, family: &str) -> PathBuf {
    let path = format!(
        "{}:{}",
        f.bin().display(),
        std::env::var("PATH").unwrap_or_default()
    );
    let mut c = Command::new("bash");
    c.env_clear();
    // (`determinism` runs the oracle first when FV_ORACLE_URL is set: that
    // prelude is the oracle tier's, so only the oracle run gets the URL.)
    if family == "oracle" {
        c.env("FV_ORACLE_URL", url(f))
            .env("FV_ORACLE_WAIT_S", "0")
            .env("FV_ORACLE_TARGETS", TARGETS);
    }
    let o = c
        .arg(f.root().join("scripts/gpu/runpod-matrix.sh"))
        .arg(family)
        .env("PATH", path)
        .env("HOME", &f.dir)
        .env("LC_ALL", "C")
        .env("FV_WORK", f.dir.join("work"))
        .env("FV_SCRATCH", f.dir.join("script"))
        .env("FV_RUN_TAG", TAG)
        .env("FV_GPUCHECK", f.bin().join("fv-gpucheck"))
        .env("FV_GEN_TIMEOUT_S", "60")
        .output()
        .expect("bash runs");
    assert!(
        o.status.success(),
        "runpod-matrix.sh {family}: {}\n{}\n{}",
        o.status,
        String::from_utf8_lossy(&o.stdout),
        String::from_utf8_lossy(&o.stderr)
    );
    f.dir.join("script/runs").join(family).join(TAG)
}

fn lab() -> Lab {
    Lab::load(&repo_root()).expect("lab/ parses")
}

fn config(f: &Fake, family: &str) -> RunConfig {
    let mut c = RunConfig::new(family, &f.root());
    c.work = f.dir.join("work");
    c.scratch = f.dir.join("lab");
    c.tag = Some(TAG.into());
    c.bin = f.bin().join("fv-gpucheck");
    c.nvidia_smi = f.bin().join("nvidia-smi").display().to_string();
    c.cell_timeout_s = 60;
    c.gpu_poll = Duration::from_millis(1);
    c.oracle_urls = vec![url(f)];
    c.oracle_wait_s = 0;
    c.oracle_poll = Duration::from_millis(1);
    c.oracle_retry = Duration::from_millis(1);
    // The fake upstream is a file:// URL (a run reads http(s) only).
    c.allow_file_urls = true;
    c
}

/// The arms of the oracle cells of [`TARGETS`] (the script's
/// `FV_ORACLE_TARGETS` as fv-lab's `--cells`).
fn oracle_arms(lab: &Lab) -> Vec<String> {
    TARGETS
        .split(' ')
        .flat_map(|t| lab.cells[&format!("oracle-{t}")].arms.iter())
        .map(|a| a.name.clone())
        .collect()
}

fn lab_run(f: &Fake, family: &str, only: Option<Vec<String>>) -> PathBuf {
    let lab = lab();
    let (_, cells) = lab.resolve(family, None).expect("tier");
    let mut cfg = config(f, family);
    cfg.only = only;
    let steps = Lab::steps(&cells, &cfg.bin, &cfg.paths()).expect("render");
    Runner::new(cfg).run(&steps).expect("fv-lab run");
    f.dir.join("lab/runs").join(family).join(TAG)
}

/// Every file below `dir`, relative, sorted.
fn files(dir: &Path) -> Vec<String> {
    fn walk(base: &Path, d: &Path, out: &mut Vec<String>) {
        for e in std::fs::read_dir(d).into_iter().flatten().flatten() {
            let p = e.path();
            if p.is_dir() {
                walk(base, &p, out);
            } else {
                out.push(p.strip_prefix(base).unwrap().display().to_string());
            }
        }
    }
    let mut v = Vec::new();
    walk(dir, dir, &mut v);
    v.sort();
    v
}

fn read(p: &Path) -> Option<Vec<u8>> {
    std::fs::read(p).ok()
}

/// A run directory's scratch disk (`<side>/runs/<family>/<tag>`): the
/// paths a report names differ by it alone between the two sides.
fn side(runs: &Path) -> String {
    runs.ancestors()
        .nth(3)
        .map(|p| p.display().to_string())
        .unwrap_or_default()
}

/// `text` with the run directory `runs` (its tag differs between the sides)
/// read as `<runs>`, then its scratch disk as `<scratch>`.
fn norm_paths(runs: &Path, text: &str) -> String {
    text.replace(&runs.display().to_string(), "<runs>")
        .replace(&side(runs), "<scratch>")
}

/// The same bytes in both runs for every file that is a pure function of the
/// outputs (not logs, timings or dates), and the same set of such files (a
/// path into either side's scratch disk read as `<scratch>`).
fn assert_same_outputs(a: &Path, b: &Path) {
    let keep = |f: &String| {
        let name = f.rsplit('/').next().unwrap_or(f);
        !(name.ends_with(".log")
            || name.starts_with("nvidia-smi-")
            || name == "summary.json"
            || name == "done.json"
            || name.ends_with(".arms.json")
            || name.ends_with(".weights.log")
            // The pod's box (the start command writes it; fv-lab adds the
            // pod's memory cgroup, ADR-0019 §16.10): no output of the cells.
            || name == "box.txt")
    };
    let (fa, fb): (Vec<String>, Vec<String>) = (
        files(a).into_iter().filter(keep).collect(),
        files(b).into_iter().filter(keep).collect(),
    );
    let only_a: Vec<&String> = fa.iter().filter(|f| !fb.contains(f)).collect();
    let only_b: Vec<&String> = fb.iter().filter(|f| !fa.contains(f)).collect();
    assert!(
        only_a.is_empty() && only_b.is_empty(),
        "the same files: only in {}: {only_a:?}; only in {}: {only_b:?}",
        a.display(),
        b.display()
    );
    for f in &fa {
        assert_eq!(
            read(&a.join(f)).map(|v| norm_paths(a, &String::from_utf8_lossy(&v))),
            read(&b.join(f)).map(|v| norm_paths(b, &String::from_utf8_lossy(&v))),
            "{f}"
        );
    }
    // Exits and skips.
    for f in files(a).iter().filter(|f| f.ends_with("/summary.json")) {
        let j = |p: &Path| {
            let v: serde_json::Value =
                serde_json::from_slice(&read(p).unwrap_or_default()).unwrap_or_default();
            (v.get("exit").cloned(), v.get("skipped").cloned())
        };
        assert_eq!(j(&a.join(f)), j(&b.join(f)), "{f}");
    }
    assert_eq!(
        files(a)
            .iter()
            .filter(|f| f.ends_with("/summary.json"))
            .count(),
        files(b)
            .iter()
            .filter(|f| f.ends_with("/summary.json"))
            .count()
    );
}

/// The lines of `live.log` without a `[HH:MM:SS] ` stamp (`tee`'d reports).
fn raw_lines(runs: &Path) -> Vec<String> {
    String::from_utf8_lossy(&read(&runs.join("live.log")).unwrap_or_default())
        .lines()
        .filter(|l| {
            let b = l.as_bytes();
            let stamp = b.len() > 11
                && b[0] == b'['
                && b[3] == b':'
                && b[6] == b':'
                && b[9] == b']'
                && [1, 2, 4, 5, 7, 8].iter().all(|&i| b[i].is_ascii_digit());
            !stamp
        })
        .filter(|l| !l.starts_with("GPU 0") && !l.starts_with("Fake") && !l.starts_with("verify "))
        .map(str::to_owned)
        .collect()
}

/// `fv-lab gate <runs> --check`: the outcomes and the differences.
fn regate(
    f: &Fake,
    family: &str,
    runs: &Path,
    out: &Path,
) -> (Vec<(String, Decision)>, Vec<String>) {
    let lab = lab();
    let (_, cells) = lab.resolve(family, None).expect("tier");
    let arms: Vec<String> = cells
        .iter()
        .flat_map(|c| &c.arms)
        .map(|a| a.name.clone())
        .collect();
    let mut cfg = config(f, family);
    cfg.runs_dir = Some(runs.to_path_buf());
    cfg.only = Some(gate_selection(family, &arms, runs, None).expect("selection"));
    let steps = Lab::steps(&cells, &cfg.bin, &cfg.paths()).expect("render");
    let mut r = Runner::offline(cfg, out);
    let o = r.regate(&steps).expect("regate").to_vec();
    (
        o.into_iter().map(|o| (o.what, o.decision)).collect(),
        r.differences(&steps),
    )
}

/// The frame logs `matrix start image=<build id>` from a fixed path, the
/// runtime image's, which the frozen cut cannot redirect: on a host that has
/// the file, its id reads as the goldens' `unknown` (as in tests/verdict.rs).
fn host_build_id() -> Option<String> {
    std::fs::read_to_string("/opt/fastvideo-rs/target/release/fv-gpucheck.build-id")
        .ok()
        .map(|s| s.trim().to_owned())
        .filter(|s| !s.is_empty())
}

/// The frozen script is the one the frozen goldens came from: its dry run of
/// `det-short` and `determinism`, under `fv-lab-goldens.sh`'s inputs, is
/// `fixtures/<family>-dry-run.txt` line for line (which `tests/dry_run.rs`
/// holds fv-lab's dry run to), and its `FV_DRY_RUN_GATES` records are the
/// first four columns of `fixtures/<family>-gates.tsv` (the fifth, a cell's
/// needs, is `verify-weights.sh --list`'s, added by the goldens script).
#[test]
fn the_frozen_script_dry_runs_as_the_frozen_goldens() {
    for family in ["det-short", "determinism"] {
        let f = fake(&format!("{family}-frozen-dry"), "");
        let (work, scratch) = (f.dir.join("dry-work"), f.dir.join("dry-scratch"));
        std::fs::create_dir_all(work.join("weights/auxiliary/tae")).unwrap();
        std::fs::write(work.join("weights/auxiliary/tae/.complete"), "").unwrap();
        let libs = scratch.join("fv-libs");
        std::fs::create_dir_all(&libs).unwrap();
        for l in ["libcudnn.so", "libcupti.so"] {
            std::fs::write(libs.join(l), "").unwrap();
        }
        let gates = f.dir.join("dry-gates.tsv");
        let o = Command::new("bash")
            .arg(f.root().join("scripts/gpu/runpod-matrix.sh"))
            .arg(family)
            .env_clear()
            .env("PATH", std::env::var("PATH").unwrap_or_default())
            .env("HOME", &f.dir)
            .env("LC_ALL", "C")
            .env("FV_DRY_RUN", "1")
            .env("FV_DRY_RUN_GATES", &gates)
            .env("FV_WORK", &work)
            .env("FV_SCRATCH", &scratch)
            .env("FV_RUN_TAG", "1234567-10241200")
            .env("FV_BASELINE_SHA", "7654321")
            .output()
            .expect("bash runs");
        assert!(
            o.status.success(),
            "{family}: {}",
            String::from_utf8_lossy(&o.stderr)
        );
        let (s, w, r) = (
            scratch.display().to_string(),
            work.display().to_string(),
            f.root().display().to_string(),
        );
        let tag = format!("[{family}] ");
        let ours: Vec<String> = String::from_utf8_lossy(&o.stdout)
            .lines()
            .map(|l| l.get(11..).unwrap_or(""))
            .filter(|l| l.contains(&tag))
            .map(|l| {
                let l = l
                    .replace(&s, "/fvscratch")
                    .replace(&w, "/workspace")
                    .replace(&r, "/opt/fastvideo-rs");
                match host_build_id() {
                    Some(id) => l.replace(
                        &format!("matrix start image={id}"),
                        "matrix start image=unknown",
                    ),
                    None => l,
                }
            })
            .collect();
        let golden = std::fs::read_to_string(
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join(format!("fixtures/{family}-dry-run.txt")),
        )
        .expect("frozen golden");
        assert_eq!(ours, golden.lines().collect::<Vec<_>>(), "{family}");
        let first4 = |t: &str| -> Vec<String> {
            t.lines()
                .map(|l| l.split('\t').take(4).collect::<Vec<_>>().join("\t"))
                .collect()
        };
        let ours = std::fs::read_to_string(&gates)
            .expect("FV_DRY_RUN_GATES written")
            .replace(&s, "/fvscratch")
            .replace(&w, "/workspace");
        let golden = std::fs::read_to_string(
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(format!("fixtures/{family}-gates.tsv")),
        )
        .expect("frozen gates");
        assert!(!golden.is_empty(), "{family}: gates");
        assert_eq!(first4(&ours), first4(&golden), "{family}: gates");
    }
}

#[test]
fn det_short_hashes_and_report_as_the_script() {
    for mode in ["", "flaky"] {
        let f = fake(&format!("det-short{mode}"), mode);
        let s = script_run(&f, "det-short");
        let l = lab_run(&f, "det-short", None);
        assert_same_outputs(&s, &l);
        let det = String::from_utf8_lossy(&read(&l.join("determinism.txt")).unwrap()).into_owned();
        assert!(det.contains("h3-timed DIFFERENT"), "{det}");
        assert!(
            det.contains("wan-r1 DIFFERENT frames none audio none"),
            "{det}"
        );
        assert_eq!(det.contains("h3-r3 DIFFERENT"), mode == "flaky", "{det}");
        // The kept frame and the report lines in live.log, as the script tees them.
        assert!(l.join("h3-r1/frames/frame-000.png").is_file());
        assert!(!l.join("h3-r1/frames/frame-001.png").exists());
        assert_eq!(raw_lines(&s), raw_lines(&l));
        for (runs, side) in [(&s, "script"), (&l, "lab")] {
            let (outcomes, diffs) =
                regate(&f, "det-short", runs, &f.dir.join(format!("{side}.regate")));
            assert!(diffs.is_empty(), "{side}: {diffs:?}");
            let want = |cell: &str, d: Decision| (format!("repeats {cell}"), d);
            let h3 = if mode == "flaky" {
                Decision::Fail
            } else {
                Decision::Pass
            };
            assert_eq!(
                outcomes,
                [
                    want("det-short-h3", h3),
                    want("det-short-wan", Decision::Skipped),
                    want("det-short-ltx", Decision::Pass),
                    want("det-short-ltx-sage", Decision::Pass),
                ],
                "{side}"
            );
        }
    }
}

#[test]
fn determinism_hashes_runs_and_reports_as_the_script() {
    // `noh3`: the h3 model's arms skip on their weights, so its reference
    // arm has no run1 hashes (the script ended the family there before §15).
    for mode in ["", "flaky", "noh3"] {
        let f = fake(&format!("determinism{mode}"), mode);
        let s = script_run(&f, "determinism");
        let l = lab_run(&f, "determinism", None);
        assert_same_outputs(&s, &l);
        for p in ["t2v-ab.json", "i2v-c.json", "h3-ab.json"] {
            assert!(l.join(p).is_file(), "{p}");
        }
        let det = String::from_utf8_lossy(&read(&l.join("determinism.txt")).unwrap()).into_owned();
        assert!(det.contains("ltx-t2v-ab/run2 frames IDENTICAL"), "{det}");
        assert!(det.contains("ltx-t2v-timed/run1 frames DIFFERENT"), "{det}");
        assert_eq!(
            det.contains("ltx-t2v-c/run3 frames DIFFERENT"),
            mode == "flaky"
        );
        assert_eq!(
            det.contains("h3-ab: no run1 hashes\n"),
            mode == "noh3",
            "{det}"
        );
        // Three keyframes kept per run, the digests read from stderr.
        assert!(l.join("ltx-t2v-ab/frames/run1/frame-072.png").is_file());
        assert!(!l.join("ltx-t2v-ab/frames/run1/frame-001.png").exists());
        assert!(
            l.join("ltx-t2v-ab/frames/warmup/frame-001.png").is_file(),
            "warm-up untouched"
        );
        assert_eq!(
            read(&l.join("ltx-t2v-ab/digests.txt")).unwrap(),
            b"sha=rule\n".to_vec()
        );
        assert_eq!(raw_lines(&s), raw_lines(&l));
        for (runs, side) in [(&s, "script"), (&l, "lab")] {
            let (outcomes, diffs) = regate(
                &f,
                "determinism",
                runs,
                &f.dir.join(format!("{side}.regate")),
            );
            assert!(diffs.is_empty(), "{side}: {diffs:?}");
            let t2v = if mode == "flaky" {
                Decision::Fail
            } else {
                Decision::Pass
            };
            let h3 = if mode == "noh3" {
                Decision::Skipped
            } else {
                Decision::Pass
            };
            assert_eq!(
                outcomes.iter().map(|(_, d)| *d).collect::<Vec<_>>(),
                [t2v, Decision::Pass, h3],
                "{side}: {outcomes:?}"
            );
        }
    }
}

#[test]
fn oracle_fetches_compares_dumps_and_cleans_up_as_the_script() {
    let f = fake("oracle", "");
    let s = script_run(&f, "oracle");
    let lab = lab();
    let l = lab_run(&f, "oracle", Some(oracle_arms(&lab)));
    assert_same_outputs(&s, &l);
    for runs in [&s, &l] {
        let sum = |a: &str| {
            String::from_utf8_lossy(&read(&runs.join(a).join("summary.json")).unwrap_or_default())
                .into_owned()
        };
        assert!(sum("oracle-fasth3-4step-vsa").contains("\"exit\":0"));
        assert!(sum("oracle-fasth3-4step-vsa-bf16-vs-f32").contains("\"exit\":0"));
        assert!(sum("oracle-ltx25-512p").contains("reference dump unavailable"));
        assert!(!runs.join("oracle-ltx25-512p-diff").exists());
        // The dumps (hundreds of MB on a pod) are gone after the target.
        assert!(!runs.join("oracle-fasth3-4step-vsa-dump").exists());
        assert!(!runs.join("oracle-fasth3-4step-vsa-dump-f32").exists());
    }
    for side in ["script", "lab"] {
        assert!(
            !f.dir
                .join(side)
                .join("oracle-ref/fasth3-4step-vsa")
                .exists(),
            "{side}"
        );
    }
    // The compare-dumps lines echoed into live.log, as oracle_diff tees them.
    let echoed = |runs: &Path| -> Vec<String> {
        raw_lines(runs)
            .into_iter()
            .filter(|l| l.starts_with("[oracle-"))
            .collect()
    };
    assert_eq!(echoed(&s), echoed(&l));
    assert!(echoed(&l).contains(
        &"[oracle-fasth3-4step-vsa-f32-diff] compare-dumps latent rel_l2 1.000e-3 cosine 0.999990"
            .to_owned()
    ));
    assert!(!echoed(&l).iter().any(|e| e.contains("[INFO] compared")));
    // One reference failed: no ORACLE_REFS_DONE on either side.
    assert!(!s.join("ORACLE_REFS_DONE").exists() && !l.join("ORACLE_REFS_DONE").exists());
    // With every selected reference fetched, fv-lab publishes it.
    let only: Vec<String> = lab.cells["oracle-fasth3-4step-vsa"]
        .arms
        .iter()
        .map(|a| a.name.clone())
        .collect();
    let mut cfg = config(&f, "oracle");
    cfg.scratch = f.dir.join("lab2");
    cfg.only = Some(only);
    let libs = cfg.scratch.join("fv-libs");
    std::fs::create_dir_all(&libs).unwrap();
    std::fs::write(libs.join("libcudnn.so"), "").unwrap();
    std::fs::write(libs.join("libcupti.so"), "").unwrap();
    let (_, cells) = lab.resolve("oracle", None).unwrap();
    let steps = Lab::steps(&cells, &cfg.bin, &cfg.paths()).unwrap();
    Runner::new(cfg).run(&steps).unwrap();
    let done = f
        .dir
        .join("lab2/runs/oracle")
        .join(TAG)
        .join("ORACLE_REFS_DONE");
    assert!(done.is_file(), "ORACLE_REFS_DONE");
}

/// A run reads http(s) references only (review of #202): without the tests'
/// escape the file:// upstream is refused and the target skipped, nothing
/// fetched.
#[test]
fn a_file_reference_url_is_refused_outside_the_tests() {
    let f = fake("oracle-file", "");
    let lab = lab();
    let mut cfg = config(&f, "oracle");
    cfg.allow_file_urls = false;
    cfg.only = Some(oracle_arms(&lab));
    let (_, cells) = lab.resolve("oracle", None).unwrap();
    let steps = Lab::steps(&cells, &cfg.bin, &cfg.paths()).unwrap();
    Runner::new(cfg).run(&steps).unwrap();
    let runs = f.dir.join("lab/runs/oracle").join(TAG);
    let sum = String::from_utf8_lossy(
        &read(&runs.join("oracle-fasth3-4step-vsa/summary.json")).unwrap_or_default(),
    )
    .into_owned();
    assert!(sum.contains("reference dump unavailable"), "{sum}");
    assert!(!f.dir.join("lab/oracle-ref/fasth3-4step-vsa/dump").exists());
    let log = String::from_utf8_lossy(&read(&runs.join("live.log")).unwrap()).into_owned();
    assert!(log.contains("is not an http(s):// URL"), "{log}");
}

// ------------------------------------------------------------------ clip metrics

/// The clip-metric targets the fakes run (ADR-0019 §20), all ten: every metric
/// kind (pinned frames, the clip against the reference's with a required
/// file, probes, kept frames), a reference without its `ref.mp4`
/// (`retake-a`), a target that echoes its report twice (`a2v-i2v`), and one
/// whose arms skip on their weights (`a2v-guided`: its clips are missing).
/// The script side's run tag in the clip-metric test (fv-lab's is [`TAG`]).
const SCRIPT_TAG: &str = "t2-script";

const CLIP_TARGETS: &str = "ltx25-i2v ltx25-kf ltx25-ref2v ltx25-a2v ltx25-a2v-i2v ltx25-retake ltx25-retake-v ltx25-retake-a ltx25-extend ltx25-a2v-guided";

/// ffmpeg on the fake box: the comparison's filter (`ssim` / `psnr`, the end
/// of the graph) and a checksum of its inputs' bytes and of the graph, as the
/// last line holding `Parsed_<filter>`, behind a `] `.
const FFMPEG: &str = r#"graph=""; frames=""; prev=""; ins=""; last=""
for a in "$@"; do
  case "$prev" in -lavfi) graph="$a" ;; -i) ins="$ins $a" ;; esac
  [ "$a" = "-frames:v" ] && frames=1
  prev="$a"; last="$a"
done
# A decode (`-i <clip> -start_number 0 <dir>/frame-%03d.png`, ADR-0019 §21):
# two frames from the clip's bytes; a missing clip fails as ffmpeg does.
case "$last" in *.png)
  for f in $ins; do [ -f "$f" ] || exit 1; done
  for i in 000 001; do printf 'png-%s-%s' "$(cat $ins | cksum | cut -d' ' -f1)" "$i" >"${last%frame-%03d.png}frame-$i.png"; done
  exit 0 ;;
esac
m="${graph##*]}"
sum="$(for f in $ins; do cat "$f" 2>/dev/null || echo none; done | cksum | cut -d' ' -f1)"
g="$(printf '%s %s' "$graph" "$frames" | cksum | cut -d' ' -f1)"
echo "[Parsed_${m}_1 @ 0x1] early" >&2
echo "[Parsed_${m}_2 @ 0x55] ${m^^} all:$sum graph:$g" >&2
echo "[out#0/null @ 0x9] video:1kB" >&2
"#;

/// ffprobe on the fake box: two stream lines from the clip's size.
const FFPROBE: &str = r#"f=""; for a in "$@"; do f="$a"; done
n="$(wc -c <"$f" | tr -d ' ')"
echo "stream|codec_type=video|codec_name=h264|nb_frames=$n"
echo "stream|codec_type=audio|codec_name=aac|sample_rate=44100|channels=2"
echo "unused stderr" >&2
"#;

/// The fake box of [`fake`], plus ffmpeg / ffprobe, the repository's
/// fixtures (the images and the source clip the metrics read), the script's
/// `PATH` with the fakes first (it puts `/usr/local/bin` before the caller's
/// `PATH`; a real ffmpeg there would shadow them), and the clip-metric
/// targets' references, `ref.mp4` in each dump but `retake-a`'s.
fn fake_clips(name: &str) -> Fake {
    // The guided target's weight gate fails: its arms skip, its clips are missing.
    let f = fake(name, "no-ltx25-a2v-guided");
    for (tool, body) in [("ffmpeg", FFMPEG), ("ffprobe", FFPROBE)] {
        common::write_exe(&f.bin().join(tool), &format!("#!/bin/bash\n{body}"));
    }
    let fx = f.root().join("scripts/gpu/fixtures");
    std::fs::create_dir_all(&fx).unwrap();
    for e in std::fs::read_dir(repo_root().join("scripts/gpu/fixtures")).unwrap() {
        let e = e.unwrap();
        if e.path().is_file() {
            std::fs::copy(e.path(), fx.join(e.file_name())).unwrap();
        }
    }
    let m = f.root().join("scripts/gpu/runpod-matrix.sh");
    let text = std::fs::read_to_string(&m).unwrap();
    let line = "export PATH=\"/opt/fastvideo-rs/target/release:/usr/local/bin:/usr/local/cuda/bin:${PATH:-}\"";
    assert_eq!(text.matches(line).count(), 1, "the script's PATH line");
    std::fs::write(
        &m,
        text.replace(
            line,
            &format!(
                "export PATH=\"{}:/opt/fastvideo-rs/target/release:/usr/local/bin:/usr/local/cuda/bin:${{PATH:-}}\"",
                f.bin().display()
            ),
        ),
    )
    .unwrap();
    for t in CLIP_TARGETS.split(' ') {
        let d = f.up().join(format!("oracle-{t}"));
        std::fs::create_dir_all(d.join("dump")).unwrap();
        std::fs::write(d.join("dump/latent.bin"), "latent-rule").unwrap();
        std::fs::write(d.join("dump/noise.bin"), "noise").unwrap();
        if t != "ltx25-retake-a" {
            std::fs::write(d.join("dump/ref.mp4"), format!("ref-{t}")).unwrap();
        }
        let st = Command::new("tar")
            .arg("-cf")
            .arg(d.join("oracle-dump.tar"))
            .arg("-C")
            .arg(&d)
            .arg("dump")
            .status()
            .expect("tar");
        assert!(st.success());
        std::fs::remove_dir_all(d.join("dump")).unwrap();
        std::fs::write(d.join("ORACLE_DONE"), "").unwrap();
    }
    f
}

#[test]
fn clip_metrics_report_as_the_script() {
    let f = fake_clips("oracle-clips");
    let path = format!(
        "{}:{}",
        f.bin().display(),
        std::env::var("PATH").unwrap_or_default()
    );
    let o = Command::new("bash")
        .arg(f.root().join("scripts/gpu/runpod-matrix.sh"))
        .arg("oracle")
        .env_clear()
        .env("FV_ORACLE_URL", url(&f))
        .env("FV_ORACLE_WAIT_S", "0")
        .env("FV_ORACLE_TARGETS", CLIP_TARGETS)
        .env("PATH", path)
        .env("HOME", &f.dir)
        .env("LC_ALL", "C")
        .env("FV_WORK", f.dir.join("work"))
        .env("FV_SCRATCH", f.dir.join("script"))
        // Another tag than fv-lab's: the `missing` lines name the run directory.
        .env("FV_RUN_TAG", SCRIPT_TAG)
        .env("FV_GPUCHECK", f.bin().join("fv-gpucheck"))
        .env("FV_GEN_TIMEOUT_S", "60")
        .output()
        .expect("bash runs");
    assert!(
        o.status.success(),
        "runpod-matrix.sh oracle: {}\n{}\n{}",
        o.status,
        String::from_utf8_lossy(&o.stdout),
        String::from_utf8_lossy(&o.stderr)
    );
    let s = f.dir.join("script/runs/oracle").join(SCRIPT_TAG);
    let lab = lab();
    let only: Vec<String> = CLIP_TARGETS
        .split(' ')
        .flat_map(|t| lab.cells[&format!("oracle-{t}")].arms.iter())
        .map(|a| a.name.clone())
        .collect();
    let (_, cells) = lab
        .resolve_selected("oracle", None, Some(&only))
        .expect("tier");
    let mut cfg = config(&f, "oracle");
    cfg.only = Some(only);
    cfg.ffmpeg = f.bin().join("ffmpeg").display().to_string();
    cfg.ffprobe = f.bin().join("ffprobe").display().to_string();
    let steps = Lab::steps(&cells, &cfg.bin, &cfg.paths()).expect("render");
    Runner::new(cfg).run(&steps).expect("fv-lab run");
    let l = f.dir.join("lab/runs/oracle").join(TAG);
    // fv-lab publishes ORACLE_REFS_DONE once every reference is fetched; the
    // script's watcher can miss it in a run of several targets (§15.5).
    assert!(l.join("ORACLE_REFS_DONE").is_file());
    let _ = std::fs::remove_file(s.join("ORACLE_REFS_DONE"));
    std::fs::remove_file(l.join("ORACLE_REFS_DONE")).unwrap();
    // metrics.txt, the clip copies, the arms' records: the same bytes.
    assert_same_outputs(&s, &l);
    let norm = |runs: &Path| -> Vec<String> {
        raw_lines(runs)
            .into_iter()
            .map(|x| norm_paths(runs, &x))
            .collect()
    };
    // The reports echoed into live.log (and the compare-dumps lines), as tee'd.
    assert_eq!(norm(&s), norm(&l));
    let metrics = |t: &str| -> String {
        String::from_utf8_lossy(
            &read(&l.join(format!("oracle-{t}-frames/metrics.txt"))).unwrap_or_default(),
        )
        .into_owned()
    };
    let kf = metrics("ltx25-kf");
    for want in [
        "ours frame 0 ssim: SSIM all:",
        "ownimg frame 120 psnr: PSNR all:",
        "reference frame 120 ssim: SSIM all:",
    ] {
        assert!(kf.contains(want), "{want}\n{kf}");
    }
    assert_eq!(kf.lines().count(), 12, "{kf}");
    let r2v = metrics("ltx25-ref2v");
    assert!(
        r2v.contains("ownimg vs reference clip psnr: PSNR all:"),
        "{r2v}"
    );
    assert!(l.join("oracle-ltx25-ref2v-frames/reference.mp4").is_file());
    let retake = metrics("ltx25-retake");
    for want in [
        "ownenc: stream|codec_type=video|codec_name=h264|nb_frames=",
        "reference kept frames vs source psnr: PSNR all:",
        "ours vs reference clip ssim: SSIM all:",
    ] {
        assert!(retake.contains(want), "{want}\n{retake}");
    }
    // No reference clip: one line for it, and no clip comparison.
    let ra = metrics("ltx25-retake-a");
    assert!(
        ra.contains("reference missing ") && ra.contains("ltx25-retake-a/dump/ref.mp4"),
        "{ra}"
    );
    assert!(!ra.contains("vs reference clip"), "{ra}");
    // The guided target's arms skipped: its two clips are missing (in the run
    // directory, whose tag differs between the sides), the reference probed.
    let guided = metrics("ltx25-a2v-guided");
    assert!(
        guided.contains("ours missing ") && guided.contains("ownaud missing "),
        "{guided}"
    );
    assert!(guided.contains("reference: stream|"), "{guided}");
    assert!(!guided.contains("vs reference clip"), "{guided}");
    for t in CLIP_TARGETS.split(' ') {
        assert!(
            l.join(format!("oracle-{t}-frames/metrics.txt")).is_file(),
            "{t}"
        );
    }
    // a2v-i2v echoes its report after each part: the probe lines twice.
    let echoed = norm(&l)
        .into_iter()
        .filter(|x| x.starts_with("[ltx25-a2v-i2v-frames] ours: stream|codec_type=video"))
        .count();
    assert_eq!(echoed, 2);
    // The dumps, ownaud's too, and the references are gone after a target.
    for runs in [&s, &l] {
        assert!(!runs.join("oracle-ltx25-a2v-i2v-dump-ownaud").exists());
        assert!(!runs.join("oracle-ltx25-retake-dump-ownenc").exists());
    }
}

// ------------------------------------------------------------------ sfwan13

/// `sfwan13` (ADR-0019 §21) on both runners: the live `oracle` branch with
/// `FV_ORACLE_TARGETS=sfwan13` and `fv-lab run oracle-r2` (its weights
/// read as a volume's here: the R2 staging is `tests/r2_runner.rs`'s), on
/// one fake box with a fake ffmpeg (a decode writes frames from the clip's
/// bytes) and a fake `compare-clips` (a report of both directories' bytes).
/// The same records, `compare-dumps` lines, decoded frames and compare
/// reports, in the same order; the dumps and the reference (its decoded
/// frames with it) gone after the target. Then without the reference's
/// `ref.mp4`: no reference frames on either side, both compares against
/// them "frames missing", the codec floor still compared.
#[test]
fn sfwan13_decodes_compares_and_cleans_up_as_the_script() {
    for with_ref_mp4 in [true, false] {
        let f = fake(
            if with_ref_mp4 {
                "oracle-sfwan13"
            } else {
                "oracle-sfwan13-noref"
            },
            "",
        );
        for (tool, body) in [("ffmpeg", FFMPEG), ("ffprobe", FFPROBE)] {
            common::write_exe(&f.bin().join(tool), &format!("#!/bin/bash\n{body}"));
        }
        // The script puts /usr/local/bin before the caller's PATH: the fakes first.
        let m = f.root().join("scripts/gpu/runpod-matrix.sh");
        let text = std::fs::read_to_string(&m).unwrap();
        let line = "export PATH=\"/opt/fastvideo-rs/target/release:/usr/local/bin:/usr/local/cuda/bin:${PATH:-}\"";
        assert_eq!(text.matches(line).count(), 1, "the script's PATH line");
        std::fs::write(
            &m,
            text.replace(
                line,
                &format!(
                    "export PATH=\"{}:/opt/fastvideo-rs/target/release:/usr/local/bin:/usr/local/cuda/bin:${{PATH:-}}\"",
                    f.bin().display()
                ),
            ),
        )
        .unwrap();
        let d = f.up().join("oracle-sfwan13");
        std::fs::create_dir_all(d.join("dump")).unwrap();
        std::fs::write(d.join("dump/latent.bin"), "latent-rule").unwrap();
        std::fs::write(d.join("dump/noise.bin"), "noise").unwrap();
        if with_ref_mp4 {
            std::fs::write(d.join("dump/ref.mp4"), "ref-sfwan13").unwrap();
        }
        let st = Command::new("tar")
            .arg("-cf")
            .arg(d.join("oracle-dump.tar"))
            .arg("-C")
            .arg(&d)
            .arg("dump")
            .status()
            .expect("tar");
        assert!(st.success());
        std::fs::remove_dir_all(d.join("dump")).unwrap();
        std::fs::write(d.join("ORACLE_DONE"), "").unwrap();
        let path = format!(
            "{}:{}",
            f.bin().display(),
            std::env::var("PATH").unwrap_or_default()
        );
        let o = Command::new("bash")
            .arg(&m)
            .arg("oracle")
            .env_clear()
            .env("FV_ORACLE_URL", url(&f))
            .env("FV_ORACLE_WAIT_S", "0")
            .env("FV_ORACLE_TARGETS", "sfwan13")
            .env("PATH", path)
            .env("HOME", &f.dir)
            .env("LC_ALL", "C")
            .env("FV_WORK", f.dir.join("work"))
            .env("FV_SCRATCH", f.dir.join("script"))
            .env("FV_RUN_TAG", TAG)
            .env("FV_GPUCHECK", f.bin().join("fv-gpucheck"))
            .env("FV_GEN_TIMEOUT_S", "60")
            .output()
            .expect("bash runs");
        assert!(
            o.status.success(),
            "runpod-matrix.sh oracle: {}\n{}\n{}",
            o.status,
            String::from_utf8_lossy(&o.stdout),
            String::from_utf8_lossy(&o.stderr)
        );
        let s = f.dir.join("script/runs/oracle").join(TAG);
        let lab = lab();
        let (_, cells) = lab.resolve("oracle-r2", None).expect("tier");
        let mut cfg = config(&f, "oracle-r2");
        cfg.ffmpeg = f.bin().join("ffmpeg").display().to_string();
        cfg.ffprobe = f.bin().join("ffprobe").display().to_string();
        let steps = Lab::steps(&cells, &cfg.bin, &cfg.paths()).expect("render");
        Runner::new(cfg).run(&steps).expect("fv-lab run");
        let l = f.dir.join("lab/runs/oracle-r2").join(TAG);
        // fv-lab publishes ORACLE_REFS_DONE once its one reference is
        // fetched; the script's watcher, given FV_ORACLE_WAIT_S=0 here, does
        // not wait to (§15.5).
        assert!(l.join("ORACLE_REFS_DONE").is_file());
        for runs in [&s, &l] {
            let _ = std::fs::remove_file(runs.join("ORACLE_REFS_DONE"));
        }
        assert_same_outputs(&s, &l);
        let norm = |runs: &Path| -> Vec<String> {
            raw_lines(runs)
                .into_iter()
                .map(|x| norm_paths(runs, &x))
                .collect()
        };
        assert_eq!(norm(&s), norm(&l));
        // The compares' log lines, in the script's order between the arms.
        let order = |runs: &Path| -> Vec<String> {
            let log = String::from_utf8_lossy(&read(&runs.join("live.log")).unwrap_or_default())
                .into_owned();
            log.lines()
                .filter_map(|x| x.split_once("] ").map(|(_, r)| r))
                .filter_map(|r| r.split_once("] ").map_or(Some(r), |(_, r)| Some(r)))
                .filter(|r| {
                    r.starts_with("ok oracle-")
                        || r.starts_with("compare oracle-")
                        || r.starts_with("skip compare oracle-")
                })
                .map(|r| {
                    let r = r.split("  ").next().unwrap_or(r);
                    let r = r.split(" exit=").next().unwrap_or(r);
                    r.split(" (frames missing)").next().unwrap_or(r).to_owned()
                })
                .collect()
        };
        assert_eq!(order(&s), order(&l));
        let o = order(&l);
        let at = |w: &str| {
            o.iter()
                .position(|x| x.ends_with(w))
                .unwrap_or_else(|| panic!("{w}: {o:?}"))
        };
        assert!(at("oracle-sfwan13-bf16-vs-f32") < at("oracle-sfwan13-frames-vs-ref"));
        assert!(at("oracle-sfwan13-codec-floor") < at("oracle-sfwan13-legacy"));
        assert!(at("oracle-sfwan13-legacy-diff") < at("oracle-sfwan13-legacy-frames-vs-ref"));
        assert!(at("oracle-sfwan13-legacy-frames-vs-ref") < at("oracle-sfwan13-composed"));
        for runs in [&s, &l] {
            let sum = |a: &str| {
                String::from_utf8_lossy(
                    &read(&runs.join(a).join("summary.json")).unwrap_or_default(),
                )
                .into_owned()
            };
            for a in [
                "oracle-sfwan13",
                "oracle-sfwan13-f32",
                "oracle-sfwan13-legacy",
                "oracle-sfwan13-composed",
                "oracle-sfwan13-flash-vs-composed",
            ] {
                assert!(sum(a).contains("\"exit\":0"), "{a}: {}", sum(a));
            }
            // Our mp4 decoded beside our frames; the codec floor compared.
            assert!(runs
                .join("oracle-sfwan13/mp4frames/frame-001.png")
                .is_file());
            let rep = |t: &str| {
                String::from_utf8_lossy(
                    &read(&runs.join(format!("compare/compare-clips-{t}.json")))
                        .unwrap_or_default(),
                )
                .into_owned()
            };
            assert!(rep("oracle-sfwan13-codec-floor").contains("\"baseline\":"));
            for t in [
                "oracle-sfwan13-frames-vs-ref",
                "oracle-sfwan13-legacy-frames-vs-ref",
            ] {
                if with_ref_mp4 {
                    assert!(rep(t).contains("\"candidate\":"), "{t}: {}", rep(t));
                } else {
                    assert!(rep(t).contains("frames missing"), "{t}: {}", rep(t));
                }
            }
            // The four dumps are gone after the target.
            for d in ["", "-f32", "-legacy", "-composed"] {
                assert!(
                    !runs.join(format!("oracle-sfwan13-dump{d}")).exists(),
                    "{d}"
                );
            }
        }
        // The reference, its decoded frames with it, is gone on both sides.
        for side in ["script", "lab"] {
            assert!(
                !f.dir.join(side).join("oracle-ref/sfwan13").exists(),
                "{side}"
            );
        }
    }
}
