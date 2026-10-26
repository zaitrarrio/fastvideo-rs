//! D1d's second slice (docs/adr/0019-fv-lab.md §15) against the script it
//! moves: `runpod-matrix.sh det-short`, `determinism` and `oracle` (their live
//! branches, copied with what they source into a fake root) and `fv-lab run`
//! of the committed tiers, on one fake box: a fake fv-gpucheck that writes
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

use std::os::unix::fs::PermissionsExt as _;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Duration;

use fastvideo_lab::plan::Lab;
use fastvideo_lab::run::{gate_selection, Decision, RunConfig, Runner};

const TAG: &str = "t1";

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn script(path: &Path, body: &str) {
    if let Some(d) = path.parent() {
        std::fs::create_dir_all(d).unwrap();
    }
    std::fs::write(path, format!("#!/bin/sh\n{body}")).unwrap();
    std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o755)).unwrap();
}

/// fv-gpucheck on the fake box. `compare-dumps --baseline A --candidate B`:
/// one line per `*.bin` of A, equal or not. A generation: its dumps under
/// `FASTVIDEO_DUMP_DIR`, stage digests with `FASTVIDEO_DIGEST=1`, and frames
/// (`frame-000`, `001`, `072`, `073`) in each prompt-set run directory (or the
/// clip directory) and a `warmup/`, `audio.wav` for LTX. The bytes depend on
/// the timed pick (`FASTVIDEO_SDPA_AUTO`) and f32 activations, so the timed
/// control differs; the mode word `flaky` makes `h3-r3` and `ltx-t2v-c` differ.
const GPUCHECK: &str = r#"if [ "$1" = compare-dumps ]; then
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

/// What runpod-matrix.sh sources, from this checkout (the branches are live).
const SOURCED: [&str; 4] = [
    "runpod-matrix.sh",
    "b4-verdict.sh",
    "cuda-13.pins",
    "fv-lab-retired.sh",
];

fn fake(name: &str, mode: &str) -> Fake {
    let dir = std::env::temp_dir().join(format!("fv-lab-d1d-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let f = Fake { dir };
    let gpu = f.root().join("scripts/gpu");
    std::fs::create_dir_all(&gpu).unwrap();
    for s in SOURCED {
        std::fs::copy(repo_root().join("scripts/gpu").join(s), gpu.join(s)).unwrap();
    }
    std::fs::write(f.dir.join("mode"), mode).unwrap();
    // The EU volume lacks h3-8step and fastwan22-ti2v-5b.
    script(
        &gpu.join("verify-weights.sh"),
        &format!(
            "echo \"verify $1\"\ncase \"$1\" in fasth3-8step | fastwan22-ti2v-5b) exit 1 ;; esac\ngrep -qw noh3 {} && [ \"$1\" = fasth3-4step-vsa ] && exit 1\nexit 0\n",
            f.dir.join("mode").display()
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

/// The same bytes in both runs for every file that is a pure function of the
/// outputs (not logs, timings or dates), and the same set of such files.
fn assert_same_outputs(a: &Path, b: &Path) {
    let keep = |f: &String| {
        let name = f.rsplit('/').next().unwrap_or(f);
        !(name.ends_with(".log")
            || name.starts_with("nvidia-smi-")
            || name == "summary.json"
            || name == "done.json"
            || name.ends_with(".arms.json")
            || name.ends_with(".weights.log"))
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
            read(&a.join(f)).map(|v| String::from_utf8_lossy(&v).into_owned()),
            read(&b.join(f)).map(|v| String::from_utf8_lossy(&v).into_owned()),
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
