//! The verdict step (docs/adr/0019-fv-lab.md §11, D1d) against the script it
//! replaced: `runpod-matrix.sh b4a` (its last real branch, copied into a fake
//! root) and `fv-lab run b4a` (the real `lab/cells/b4a-*.toml`) on the same
//! fake box: a fake fv-gpucheck whose arms log step-cache evidence, a fake
//! nvidia-smi, verify-weights.sh and fetch-baseline.sh. Both runners must
//! write the same verdict files, evidence files and evidence diffs, byte for
//! byte, and the same verdict lines in `live.log`; and `fv-lab gate` must
//! derive the same verdicts again from either run directory alone, with the
//! selection it infers itself (`run::gate_selection`).
//!
//! The script side is frozen (ADR-0019 §11.6: `b4a` retired, its branch
//! deleted from `scripts/gpu/runpod-matrix.sh`). `fixtures/b4a-script/` holds
//! it as it last ran, at main `8b18d89f` (the commit before the deletion):
//! `runpod-matrix.sh` is that script cut to its shared frame, the `b4a`
//! branch and the fallback branch, every kept line unchanged (`git show
//! 8b18d89f:scripts/gpu/runpod-matrix.sh | sed -n '1,446p;1618,1743p;3197,3223p'`),
//! and `b4-verdict.sh` is that commit's file whole. What they source besides
//! (`cuda-13.pins`, `fv-lab-retired.sh`) is data, copied as committed. So a
//! change to fv-lab's verdict step is still held to the script's bytes, and
//! a change to the live scripts no longer moves the reference.
//!
//! The fake box's default makes one cell pass, one fail on its evidence
//! only, and one fail on everything (exit, reuse, identity, evidence). Its
//! `edge` mode (a word in `<fake>/mode`) gives an arm an empty stderr.log and
//! the other CRLF lines, bytes that are not UTF-8 and a NUL; `nocmp` writes no
//! compare report for one cell; `noweights` skips another cell's arms on
//! their weight gate.

use std::os::unix::fs::PermissionsExt as _;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Duration;

use fastvideo_lab::plan::{CellFile, Lab, Step};
use fastvideo_lab::run::{gate_selection, selection_file, Decision, RunConfig, Runner};

const BASELINE_SHA: &str = "7654321";
const TAG: &str = "t1";
const CELLS: [&str; 3] = [
    "b4a-h3-fullopt-env",
    "b4a-h3-fullopt-prof",
    "b4a-h3-4step-tea",
];

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

/// fv-gpucheck on the fake box (`@MODE@`: the mode file). `gen`: frames
/// under `--clip-dir`, step-cache lines on stderr (with the log's bracketed
/// prefixes), per cell (the recipe argument) and arm (`$0` under
/// `baseline-<sha>/` is the baseline): `env` identical arms that reuse;
/// `prof` the same frames but one decision differs; `tea` the PR arm exits
/// 3, reuses nothing, other frames. `compare-clips`: an off-identity report
/// (pretty-printed, as the real one) from the two first frames' bytes.
const GPUCHECK: &str = r#"mode="$(cat @MODE@ 2>/dev/null)"
has() { case " $mode " in *" $1 "*) return 0 ;; esac; return 1; }
if [ "$1" = --out ]; then
  out="$2"; tag="$4"; sub="$5"; shift 5
  case "$sub" in
    compare-clips)
      if has nocmp; then case "$tag" in *prof*) exit 1 ;; esac; fi
      ok=false
      cmp -s "$2/frame-000.png" "$4/frame-000.png" && ok=true
      st=fail; [ "$ok" = true ] && st=pass
      printf '{\n  "checks": [\n    {\n      "name": "off_identity",\n      "ok": %s\n    },\n    {\n      "name": "off_identity_frame_count",\n      "ok": true\n    }\n  ],\n  "status": "%s"\n}\n' "$ok" "$st" >"$out/compare-clips-$tag.json"
      echo '{"psnr_mean":99.0,"x":1}' >&2 ;;
    gate)
      echo "gate verdict: pass" >&2
      echo '{"stage":"gate","status":"pass","elapsed_s":1}' >"$out/gate-$tag.json" ;;
  esac
  exit 0
fi
base=0
case "$0" in *baseline-*) base=1 ;; esac
clip=""; prev=""
for a in "$@"; do [ "$prev" = --clip-dir ] && clip="$a"; prev="$a"; done
case "$*" in
  *sol-h3-rtx*) cell=env ;;
  *rtx5090_fullopt*) cell=prof ;;
  *) cell=tea ;;
esac
mkdir -p "$clip"
if has edge && [ "$cell" = env ]; then
  printf '%s' "$cell" >"$clip/frame-000.png"
  if [ "$base" = 1 ]; then exit 0; fi
  printf '[INFO] [fastvideo] h3 teacache step 0: compute (warmup)\r\n' >&2
  printf '[INFO] [fastvideo] h3 teacache step 1: REUSE (\377\376 not utf-8)\r\n' >&2
  printf '[x] h3 teacache step 2: REUSE (a \000 nul)\n' >&2
  printf 'h3 teacache: no newline at the end' >&2
  exit 0
fi
echo "[2026-10-25T10:00:0${base}Z INFO fastvideo] h3 sol teacache: rtx 0.10 retain 5 cooldown 1" >&2
echo "[INFO] loading weights" >&2
echo "[INFO] [loader] fuse step cache: on" >&2
echo "[INFO] [fastvideo] h3 teacache step 0: compute (warmup; acc 0.0000)" >&2
if [ "$cell" = tea ] && [ "$base" = 0 ]; then
  echo "[INFO] [fastvideo] h3 teacache step 1: compute (above; acc 9.0)" >&2
  printf 'pr' >"$clip/frame-000.png"
  exit 3
fi
echo "[INFO] [fastvideo] h3 teacache step 1: REUSE (below_threshold; acc 0.1282)" >&2
if [ "$cell" = prof ] && [ "$base" = 1 ]; then
  echo "[INFO] [fastvideo] h3 teacache step 2: compute (cooldown; acc 0.0000)" >&2
else
  echo "[INFO] [fastvideo] h3 teacache step 2: REUSE (below_threshold; acc 0.2881)" >&2
fi
echo "peak vram 123 MiB" >&2
printf '%s' "$cell" >"$clip/frame-000.png"
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
    fn mode(&self, words: &str) {
        std::fs::write(self.dir.join("mode"), words).unwrap();
    }
}

fn fake(name: &str) -> Fake {
    let dir = std::env::temp_dir().join(format!("fv-lab-verdict-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let f = Fake { dir };
    let gpu = f.root().join("scripts/gpu");
    std::fs::create_dir_all(&gpu).unwrap();
    // The script as it last ran b4a (frozen), and what it sources.
    let frozen = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/b4a-script");
    for s in ["runpod-matrix.sh", "b4-verdict.sh"] {
        std::fs::copy(frozen.join(s), gpu.join(s)).unwrap();
    }
    for s in ["cuda-13.pins", "fv-lab-retired.sh"] {
        std::fs::copy(repo_root().join("scripts/gpu").join(s), gpu.join(s)).unwrap();
    }
    script(
        &gpu.join("verify-weights.sh"),
        &format!(
            "echo \"verify $1\"\ngrep -qw noweights {} 2>/dev/null && [ \"$1\" = fasth3-4step-vsa ] && exit 1\nexit 0\n",
            f.dir.join("mode").display()
        ),
    );
    script(
        &gpu.join("fetch-baseline.sh"),
        &format!(
            "mkdir -p \"$2\" && cp {} \"$2/fv-gpucheck\" && echo \"fv-gpucheck sha-$1 -> $2\"\n",
            f.bin().join("fv-gpucheck").display()
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
        std::fs::write(libs.join("libcudnn.so"), "").unwrap();
    }
    std::fs::create_dir_all(f.dir.join("work/weights")).unwrap();
    f
}

/// The frozen `runpod-matrix.sh b4a` on the fake box (`FV_B4A_CELLS`,
/// `FV_CELLS`, the locale): its run directory.
fn script_run_with(
    f: &Fake,
    b4a_cells: Option<&str>,
    fv_cells: Option<&str>,
    locale: &str,
) -> PathBuf {
    let path = format!(
        "{}:{}",
        f.bin().display(),
        std::env::var("PATH").unwrap_or_default()
    );
    let mut c = Command::new("bash");
    c.arg(f.root().join("scripts/gpu/runpod-matrix.sh"))
        .arg("b4a")
        .env_clear()
        .env("PATH", path)
        .env("HOME", &f.dir)
        .env("LANG", locale)
        .env("LC_ALL", locale)
        .env("FV_WORK", f.dir.join("work"))
        .env("FV_SCRATCH", f.dir.join("script"))
        .env("FV_RUN_TAG", TAG)
        .env("FV_BASELINE_SHA", BASELINE_SHA)
        .env("FV_GPUCHECK", f.bin().join("fv-gpucheck"))
        .env("FV_GEN_TIMEOUT_S", "60");
    if let Some(cells) = b4a_cells {
        c.env("FV_B4A_CELLS", cells);
    }
    if let Some(cells) = fv_cells {
        c.env("FV_CELLS", cells);
    }
    let o = c.output().expect("bash runs");
    assert!(
        o.status.success(),
        "runpod-matrix.sh b4a: {}\n{}\n{}",
        o.status,
        String::from_utf8_lossy(&o.stdout),
        String::from_utf8_lossy(&o.stderr)
    );
    f.dir.join("script/runs/b4a").join(TAG)
}

fn script_run(f: &Fake, b4a_cells: Option<&str>) -> PathBuf {
    script_run_with(f, b4a_cells, None, "C")
}

fn config(f: &Fake) -> RunConfig {
    let mut c = RunConfig::new("b4a", &f.root());
    c.work = f.dir.join("work");
    c.scratch = f.dir.join("lab");
    c.tag = Some(TAG.into());
    c.bin = f.bin().join("fv-gpucheck");
    c.nvidia_smi = f.bin().join("nvidia-smi").display().to_string();
    c.cell_timeout_s = 60;
    c.gpu_poll = Duration::from_millis(1);
    c.baseline_sha = Some(BASELINE_SHA.into());
    c
}

fn lab() -> Lab {
    Lab::load(&repo_root()).expect("lab/ parses")
}

fn b4a_cells(lab: &Lab) -> Vec<&CellFile> {
    lab.resolve("b4a", None).expect("the b4a tier").1
}

fn b4a_arms(lab: &Lab) -> Vec<String> {
    b4a_cells(lab)
        .iter()
        .flat_map(|c| &c.arms)
        .map(|a| a.name.clone())
        .collect()
}

fn arms(names: &[&str]) -> Vec<String> {
    names.iter().map(|s| (*s).to_owned()).collect()
}

/// `fv-lab run b4a` (the committed cells) on the fake box: its run directory.
fn lab_run(f: &Fake, only: Option<Vec<String>>) -> PathBuf {
    let lab = lab();
    let mut cfg = config(f);
    cfg.only = only;
    let steps = Lab::steps(&b4a_cells(&lab), &cfg.bin, &cfg.paths()).expect("render");
    let mut r = Runner::new(cfg);
    r.run(&steps).expect("fv-lab run b4a");
    f.dir.join("lab/runs/b4a").join(TAG)
}

/// `fv-lab gate <runs> --out <out> [--cells …]` as the binary does it: the
/// selection `gate_selection` infers, the run evaluated again into `out`.
/// The outcomes and the differences `--check` reports.
fn regate(
    f: &Fake,
    runs: &Path,
    out: &Path,
    cells: Option<Vec<String>>,
) -> Result<(Vec<Decision>, Vec<String>), String> {
    let lab = lab();
    let mut cfg = config(f);
    cfg.runs_dir = Some(runs.to_path_buf());
    cfg.only = Some(gate_selection("b4a", &b4a_arms(&lab), runs, cells)?);
    let steps = Lab::steps(&b4a_cells(&lab), &cfg.bin, &cfg.paths()).expect("render");
    let mut r = Runner::offline(cfg, out);
    let outcomes = r.regate(&steps).map_err(|e| e.to_string())?.to_vec();
    Ok((
        outcomes.iter().map(|o| o.decision).collect(),
        r.differences(&steps),
    ))
}

fn text(p: &Path) -> String {
    String::from_utf8_lossy(&std::fs::read(p).unwrap_or_else(|e| panic!("{}: {e}", p.display())))
        .into_owned()
}

/// The verdict step's lines of a `live.log`, without the time stamp.
fn verdict_lines(runs: &Path) -> Vec<String> {
    text(&runs.join("live.log"))
        .lines()
        .filter_map(|l| l.get(11..))
        .filter(|l| l.starts_with("[b4a] [b4a-") || l.starts_with("[b4a] b4a "))
        .map(str::to_owned)
        .collect()
}

/// Every file the verdict step may write, relative to the run directory.
fn verdict_files(cells: &[&str]) -> Vec<String> {
    let mut v = vec!["b4a.verdict.json".to_owned()];
    for c in cells {
        v.push(format!("b4a-{c}.verdict.json"));
        v.push(format!("{c}.evidence.diff"));
        for arm in ["base", "pr"] {
            v.push(format!("{c}-{arm}/cache-evidence.log"));
        }
    }
    v
}

/// The same files present in both, byte for byte; `want` of them must exist.
fn assert_same(a: &Path, b: &Path, files: &[String], want: usize) {
    let mut present = 0;
    for f in files {
        let (x, y) = (std::fs::read(a.join(f)).ok(), std::fs::read(b.join(f)).ok());
        assert_eq!(
            x.as_ref().map(|v| String::from_utf8_lossy(v).into_owned()),
            y.as_ref().map(|v| String::from_utf8_lossy(v).into_owned()),
            "{f}: {} vs {}",
            a.display(),
            b.display()
        );
        assert_eq!(x, y, "{f}: bytes");
        present += usize::from(x.is_some());
    }
    assert_eq!(present, want, "files present in both");
}

/// The frozen script is the one the frozen goldens came from: its dry run,
/// under `fv-lab-goldens.sh`'s inputs, is `fixtures/b4a-dry-run.txt` line for
/// line (which `tests/dry_run.rs` holds fv-lab's dry run to).
#[test]
fn the_frozen_script_dry_runs_as_the_frozen_golden() {
    let f = fake("frozen-dry");
    let (work, scratch) = (f.dir.join("dry-work"), f.dir.join("dry-scratch"));
    std::fs::create_dir_all(work.join("weights/auxiliary/tae")).unwrap();
    std::fs::write(work.join("weights/auxiliary/tae/.complete"), "").unwrap();
    std::fs::create_dir_all(&scratch).unwrap();
    let o = Command::new("bash")
        .arg(f.root().join("scripts/gpu/runpod-matrix.sh"))
        .arg("b4a")
        .env_clear()
        .env("PATH", std::env::var("PATH").unwrap_or_default())
        .env("HOME", &f.dir)
        .env("LC_ALL", "C")
        .env("FV_DRY_RUN", "1")
        .env("FV_WORK", &work)
        .env("FV_SCRATCH", &scratch)
        .env("FV_RUN_TAG", "1234567-10241200")
        .env("FV_BASELINE_SHA", BASELINE_SHA)
        .output()
        .expect("bash runs");
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    let (s, w, r) = (
        scratch.display().to_string(),
        work.display().to_string(),
        f.root().display().to_string(),
    );
    let ours: Vec<String> = String::from_utf8_lossy(&o.stdout)
        .lines()
        .map(|l| l.get(11..).unwrap_or(""))
        .filter(|l| l.contains("[b4a] "))
        .map(|l| {
            l.replace(&s, "/fvscratch")
                .replace(&w, "/workspace")
                .replace(&r, "/opt/fastvideo-rs")
        })
        .collect();
    let golden = text(&PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/b4a-dry-run.txt"));
    assert_eq!(ours, golden.lines().collect::<Vec<_>>());
}

#[test]
fn fv_lab_writes_the_scripts_verdicts_byte_for_byte() {
    let f = fake("all");
    let theirs = script_run(&f, None);
    let ours = lab_run(&f, None);
    let files = verdict_files(&CELLS);
    assert_same(&ours, &theirs, &files, files.len());
    assert_eq!(verdict_lines(&ours), verdict_lines(&theirs));

    // What the fake box made of each cell.
    assert_eq!(
        text(&ours.join("b4a-b4a-h3-fullopt-env.verdict.json")),
        "{\"cell\":\"b4a-h3-fullopt-env\",\"verdict\":\"pass\",\"why\":\"\"}\n"
    );
    assert_eq!(
        text(&ours.join("b4a-b4a-h3-fullopt-prof.verdict.json")),
        "{\"cell\":\"b4a-h3-fullopt-prof\",\"verdict\":\"fail\",\"why\":\"evidence-differs\"}\n"
    );
    assert_eq!(
        text(&ours.join("b4a-b4a-h3-4step-tea.verdict.json")),
        "{\"cell\":\"b4a-h3-4step-tea\",\"verdict\":\"fail\",\"why\":\"pr:exit pr:no-reuse not-identical evidence-differs\"}\n"
    );
    assert_eq!(
        text(&ours.join("b4a.verdict.json")),
        "{\"family\":\"b4a\",\"verdict\":\"fail\",\"cells\":\"b4a-h3-fullopt-env b4a-h3-fullopt-prof b4a-h3-4step-tea\",\"why\":\"b4a-h3-fullopt-prof:fail b4a-h3-4step-tea:fail\"}\n"
    );
    // The evidence without the bracketed prefixes; the noise line dropped.
    let ev = text(&ours.join("b4a-h3-fullopt-env-base/cache-evidence.log"));
    assert!(ev.starts_with("h3 sol teacache: rtx 0.10"), "{ev}");
    assert!(
        ev.contains("\nfuse step cache: on\nh3 teacache step 0:"),
        "{ev}"
    );
    assert!(!ev.contains("loading weights"), "{ev}");
    assert!(text(&ours.join("b4a-h3-fullopt-env.evidence.diff")).is_empty());
    assert!(!text(&ours.join("b4a-h3-fullopt-prof.evidence.diff")).is_empty());
    assert!(verdict_lines(&ours)
        .contains(&"[b4a] [b4a-h3-4step-tea-pr] teacache reused forwards: 0".to_owned()));
    // The selection record: every arm, in run order.
    assert_eq!(
        text(&ours.join(selection_file("b4a"))),
        format!(
            "{}\n",
            serde_json::json!({"family": "b4a", "arms": b4a_arms(&lab())})
        )
    );

    // The frame, compare and summary records agree too (fv-lab-compare-runs.sh's view).
    for c in CELLS {
        let r = format!("compare/compare-clips-{c}-base--{c}-pr.json");
        assert_eq!(text(&ours.join(&r)), text(&theirs.join(&r)), "{r}");
        for arm in ["base", "pr"] {
            let d = format!("{c}-{arm}/frames.sha256.json");
            assert_eq!(text(&ours.join(&d)), text(&theirs.join(&d)), "{d}");
        }
    }

    // `fv-lab gate` derives the same verdicts from either run directory
    // alone, with the selection it infers: the record on fv-lab's side, all
    // six arms present on the script's. `--check` finds the run's files equal.
    for (name, runs) in [("ours", &ours), ("theirs", &theirs)] {
        let out = f.dir.join(format!("regate-{name}"));
        let (decisions, diffs) = regate(&f, runs, &out, None).expect(name);
        assert_eq!(
            decisions,
            [
                Decision::Pass,
                Decision::Fail,
                Decision::Fail,
                Decision::Fail
            ],
            "{name}"
        );
        assert!(diffs.is_empty(), "{name}: {diffs:?}");
        assert_same(&out, &theirs, &files, files.len());
        // Offline nothing ran and nothing was removed.
        assert!(runs.join("compare").is_dir() && runs.join("b4a-h3-4step-tea-pr").is_dir());
        assert!(!out.join("live.log").exists());
    }
}

/// F6, and the cases the default leaves out: an empty stderr.log, CRLF lines,
/// bytes that are not UTF-8 and a NUL (the script in a UTF-8 locale, its grep
/// and sed pinned to bytes), a missing compare report, and arms skipped on
/// their weight gate. Script and fv-lab alike, and `fv-lab gate` too.
#[test]
fn edge_cases_match_the_script_in_a_utf8_locale() {
    let f = fake("edge");
    f.mode("edge nocmp noweights");
    let theirs = script_run_with(&f, None, None, "C.UTF-8");
    let ours = lab_run(&f, None);
    let files = verdict_files(&CELLS);
    // The 4-step arms never ran: no evidence, no diff.
    assert_same(&ours, &theirs, &files, files.len() - 3);
    assert_eq!(verdict_lines(&ours), verdict_lines(&theirs));
    assert_eq!(
        text(&ours.join("b4a-b4a-h3-fullopt-env.verdict.json")),
        "{\"cell\":\"b4a-h3-fullopt-env\",\"verdict\":\"fail\",\"why\":\"base:no-reuse evidence-differs\"}\n"
    );
    let ev = std::fs::read(ours.join("b4a-h3-fullopt-env-pr/cache-evidence.log")).unwrap();
    assert!(
        ev.windows(3).any(|w| w == b"\xff\xfe ") && ev.contains(&0) && ev.contains(&b'\r'),
        "{ev:?}"
    );
    assert!(ev.ends_with(b"h3 teacache: no newline at the end\n"));
    assert!(
        std::fs::read(ours.join("b4a-h3-fullopt-env-base/cache-evidence.log"))
            .unwrap()
            .is_empty()
    );
    assert_eq!(
        text(&ours.join("b4a-b4a-h3-fullopt-prof.verdict.json")),
        "{\"cell\":\"b4a-h3-fullopt-prof\",\"verdict\":\"fail\",\"why\":\"not-identical evidence-differs\"}\n"
    );
    assert_eq!(
        text(&ours.join("b4a-b4a-h3-4step-tea.verdict.json")),
        "{\"cell\":\"b4a-h3-4step-tea\",\"verdict\":\"fail\",\"why\":\"base:exit base:no-run pr:exit pr:no-run not-identical\"}\n"
    );

    // F5: a reused --out holding an earlier evaluation's evidence for the
    // arms that never ran: removed, not diffed.
    let out = f.dir.join("regate-edge");
    for arm in ["b4a-h3-4step-tea-base", "b4a-h3-4step-tea-pr"] {
        std::fs::create_dir_all(out.join(arm)).unwrap();
        std::fs::write(out.join(arm).join("cache-evidence.log"), arm).unwrap();
    }
    std::fs::write(out.join("b4a-h3-4step-tea.evidence.diff"), "stale").unwrap();
    let (_, diffs) = regate(&f, &theirs, &out, None).expect("regate");
    assert!(diffs.is_empty(), "{diffs:?}");
    assert_same(&out, &theirs, &files, files.len() - 3);
}

#[test]
fn a_subset_of_cells_is_the_scripts_fv_b4a_cells() {
    // The coordinator's GPU gate runs one pair: FV_B4A_CELLS on the script
    // side, its two arms in FV_CELLS on fv-lab's. The other cells run nothing
    // (no compare report, no verdict) and the family verdict names the one.
    let f = fake("subset");
    let theirs = script_run(&f, Some("b4a-h3-fullopt-env"));
    let ours = lab_run(
        &f,
        Some(arms(&["b4a-h3-fullopt-env-base", "b4a-h3-fullopt-env-pr"])),
    );
    let files = verdict_files(&["b4a-h3-fullopt-env"]);
    assert_same(&ours, &theirs, &files, files.len());
    assert_eq!(
        text(&ours.join("b4a.verdict.json")),
        "{\"family\":\"b4a\",\"verdict\":\"pass\",\"cells\":\"b4a-h3-fullopt-env\",\"why\":\"\"}\n"
    );
    for runs in [&ours, &theirs] {
        let mut listed: Vec<String> = std::fs::read_dir(runs.join("compare"))
            .unwrap()
            .flatten()
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .filter(|n| n.ends_with(".json"))
            .collect();
        listed.sort();
        assert_eq!(
            listed,
            ["compare-clips-b4a-h3-fullopt-env-base--b4a-h3-fullopt-env-pr.json"],
            "{}",
            runs.display()
        );
        assert!(!runs.join("b4a-b4a-h3-4step-tea.verdict.json").exists());
        assert!(!runs.join("b4a-h3-4step-tea-base").exists());
    }
    // fv-lab's run records its selection; the script's holds 2 of 6 arms and
    // no record, so `fv-lab gate` needs --cells there and refuses to guess.
    let (d, diffs) = regate(&f, &ours, &f.dir.join("re-ours"), None).expect("record");
    assert_eq!(d, [Decision::Pass, Decision::Pass]);
    assert!(diffs.is_empty(), "{diffs:?}");
    let e = regate(&f, &theirs, &f.dir.join("re-theirs"), None).unwrap_err();
    assert!(
        e.contains("holds 2 of the 6 arms") && e.contains("--cells"),
        "{e}"
    );
    let pair = arms(&["b4a-h3-fullopt-env-base", "b4a-h3-fullopt-env-pr"]);
    let (d, diffs) = regate(&f, &theirs, &f.dir.join("re-theirs"), Some(pair)).expect("cells");
    assert_eq!(d, [Decision::Pass, Decision::Pass]);
    assert!(diffs.is_empty(), "{diffs:?}");
}

#[test]
fn a_one_arm_selection_and_its_skipped_compare_match_the_script() {
    // Only the PR arm of one pair: the base arm did not run, its compare is
    // a "frames missing" skip, the verdict fails on both.
    let f = fake("one-arm");
    let theirs = script_run_with(
        &f,
        Some("b4a-h3-fullopt-env"),
        Some("b4a-h3-fullopt-env-pr"),
        "C",
    );
    let ours = lab_run(&f, Some(arms(&["b4a-h3-fullopt-env-pr"])));
    let files = verdict_files(&["b4a-h3-fullopt-env"]);
    // The base arm wrote no evidence, so no diff either.
    assert_same(&ours, &theirs, &files, files.len() - 2);
    let r = "compare/compare-clips-b4a-h3-fullopt-env-base--b4a-h3-fullopt-env-pr.json";
    assert_eq!(text(&ours.join(r)), text(&theirs.join(r)));
    assert!(text(&ours.join(r)).contains("\"status\":\"skipped\""));
    assert_eq!(
        text(&ours.join("b4a-b4a-h3-fullopt-env.verdict.json")),
        "{\"cell\":\"b4a-h3-fullopt-env\",\"verdict\":\"fail\",\"why\":\"base:exit base:no-run not-identical\"}\n"
    );
}

/// F1: a run that stopped after its first cell. Its selection record names
/// all six arms, so `fv-lab gate` fails the two cells that never finished;
/// without the record it refuses to guess, and with --cells it says the same.
#[test]
fn an_interrupted_run_fails_its_missing_cells() {
    let f = fake("interrupted");
    let runs = lab_run(&f, None);
    for c in &CELLS[1..] {
        for arm in ["base", "pr"] {
            std::fs::remove_dir_all(runs.join(format!("{c}-{arm}"))).unwrap();
        }
        let _ = std::fs::remove_file(runs.join(format!("b4a-{c}.verdict.json")));
        let _ = std::fs::remove_file(runs.join(format!("{c}.evidence.diff")));
        let _ =
            std::fs::remove_file(runs.join(format!("compare/compare-clips-{c}-base--{c}-pr.json")));
    }
    let _ = std::fs::remove_file(runs.join("b4a.verdict.json"));
    let out = f.dir.join("re");
    let (d, _) = regate(&f, &runs, &out, None).expect("record");
    assert_eq!(
        d,
        [
            Decision::Pass,
            Decision::Fail,
            Decision::Fail,
            Decision::Fail
        ]
    );
    assert_eq!(
        text(&out.join("b4a.verdict.json")),
        "{\"family\":\"b4a\",\"verdict\":\"fail\",\"cells\":\"b4a-h3-fullopt-env b4a-h3-fullopt-prof b4a-h3-4step-tea\",\"why\":\"b4a-h3-fullopt-prof:fail b4a-h3-4step-tea:fail\"}\n"
    );
    assert_eq!(
        text(&out.join("b4a-b4a-h3-fullopt-prof.verdict.json")),
        "{\"cell\":\"b4a-h3-fullopt-prof\",\"verdict\":\"fail\",\"why\":\"base:exit base:no-run pr:exit pr:no-run not-identical\"}\n"
    );
    std::fs::remove_file(runs.join(selection_file("b4a"))).unwrap();
    let e = regate(&f, &runs, &out, None).unwrap_err();
    assert!(e.contains("holds 2 of the 6 arms"), "{e}");
    let (d, _) = regate(&f, &runs, &out, Some(b4a_arms(&lab()))).expect("cells");
    assert_eq!(d.iter().filter(|d| **d == Decision::Fail).count(), 3);
    // Unknown arms are refused, not ignored.
    let e = regate(&f, &runs, &out, Some(arms(&["b4a-x"]))).unwrap_err();
    assert!(e.contains("`b4a-x` is not an arm"), "{e}");
}

/// F2: the run directory is never written: an `--out` equal to it or inside
/// it is refused; and `--check` reports a run file that is not what fv-lab
/// derives.
#[test]
fn gate_never_writes_into_the_run_and_check_finds_a_changed_file() {
    let f = fake("out");
    let runs = lab_run(
        &f,
        Some(arms(&["b4a-h3-fullopt-env-base", "b4a-h3-fullopt-env-pr"])),
    );
    let before = text(&runs.join("b4a.verdict.json"));
    for out in [runs.clone(), runs.join("sub")] {
        let e = regate(&f, &runs, &out, None).unwrap_err();
        assert!(e.contains("is the run directory or inside it"), "{e}");
    }
    assert_eq!(text(&runs.join("b4a.verdict.json")), before);
    assert!(!runs.join("sub").exists(), "nothing created in the run");
    std::fs::write(
        runs.join("b4a-h3-fullopt-env-pr/cache-evidence.log"),
        "edited\n",
    )
    .unwrap();
    let (_, diffs) = regate(&f, &runs, &f.dir.join("re"), None).expect("regate");
    assert_eq!(
        diffs,
        ["b4a-h3-fullopt-env-pr/cache-evidence.log: differs (diff the two files)"]
    );
}

#[test]
fn a_stale_verdict_never_stands_in_for_this_run() {
    // A verdict file left in the run directory by an earlier run is removed
    // before the first arm, as the script's `rm -f` does; so is a stale
    // compare report and arm directory of a verdict cell.
    let f = fake("stale");
    let runs = f.dir.join("lab/runs/b4a").join(TAG);
    std::fs::create_dir_all(runs.join("compare")).unwrap();
    std::fs::create_dir_all(runs.join("b4a-h3-fullopt-env-pr")).unwrap();
    std::fs::write(
        runs.join("b4a-h3-fullopt-env-pr/summary.json"),
        "{\"exit\":0}",
    )
    .unwrap();
    std::fs::write(
        runs.join("b4a-b4a-h3-gone.verdict.json"),
        "{\"verdict\":\"pass\"}",
    )
    .unwrap();
    // Only the base arm is selected: the PR arm's stale directory must not
    // pass for a run of it.
    lab_run(&f, Some(arms(&["b4a-h3-fullopt-env-base"])));
    assert!(!runs.join("b4a-b4a-h3-gone.verdict.json").exists());
    assert!(!runs.join("b4a-h3-fullopt-env-pr").exists());
    assert_eq!(
        text(&runs.join("b4a-b4a-h3-fullopt-env.verdict.json")),
        "{\"cell\":\"b4a-h3-fullopt-env\",\"verdict\":\"fail\",\"why\":\"pr:exit pr:no-run not-identical\"}\n"
    );
}

#[test]
fn the_dry_run_holds_the_verdict_steps_in_the_scripts_order() {
    let lab = lab();
    let cells = b4a_cells(&lab);
    let mut cfg = RunConfig::new("b4a", Path::new("/opt/fastvideo-rs"));
    cfg.scratch = "/fvscratch".into();
    cfg.tag = Some(TAG.into());
    cfg.baseline_sha = Some(BASELINE_SHA.into());
    let steps = Lab::steps(&cells, &cfg.bin, &cfg.paths()).unwrap();
    assert_eq!(steps.len(), 4, "three verdict cells and the family verdict");
    assert!(
        matches!(&steps[3], Step::FamilyVerdict { family, cells } if family == "b4a" && cells.len() == 3)
    );
    cfg.only = Some(vec!["b4a-h3-4step-tea-pr".into()]);
    let lines = fastvideo_lab::run::dry_run_lines(&cfg, &steps);
    let r = "/fvscratch/runs/b4a/t1";
    let want_prefix = [
        "[b4a] skip b4a-h3-fullopt-env (no arm in FV_CELLS)".to_owned(),
        "[b4a] skip b4a-h3-fullopt-prof (no arm in FV_CELLS)".to_owned(),
        format!("[b4a] dry-run b4a-h3-4step-tea-base: rm -rf {r}/b4a-h3-4step-tea-base"),
        "[b4a] skip b4a-h3-4step-tea-base (not in FV_CELLS)".to_owned(),
        format!("[b4a] dry-run b4a-h3-4step-tea-pr: rm -rf {r}/b4a-h3-4step-tea-pr"),
    ];
    assert_eq!(lines[1..6], want_prefix);
    assert_eq!(
        lines[lines.len() - 2],
        format!("[b4a] dry-run b4a: family verdict b4_family_verdict b4a b4a-h3-4step-tea -> {r}/b4a.verdict.json (a named cell without a passing verdict fails)")
    );
}

/// A promotion gate re-evaluated offline (`fv-lab gate`): `fv-gpucheck gate`
/// against the finished run's arms and compare reports, the report written
/// into `--out`, the run directory untouched. F3: a gate whose arms did not
/// both run is skipped (and counted), one whose arms both ran but lack a
/// benchmark.json fails.
#[test]
fn gates_are_evaluated_again_offline() {
    const CELL: &str = r#"
schema = 1
name = "g"
description = "a gate"

[workload]
model = "h3"

[[arm]]
name = "a"
gate = "ok"
trees = ["w"]
argv = ["gen"]

[[arm]]
name = "b"
gate = "ok"
trees = ["w"]
argv = ["gen"]

[[arm]]
name = "c"
gate = "ok"
trees = ["w"]
argv = ["gen"]

[[arm]]
name = "d"
gate = "ok"
trees = ["w"]
argv = ["gen"]

[[gate]]
baseline = "a"
candidate = "b"
kind = "exact"

[[gate]]
baseline = "a"
candidate = "c"

[[gate]]
baseline = "a"
candidate = "d"
"#;
    let f = fake("gate");
    let runs = f.dir.join("finished");
    let summary = |arm: &str, body: &str| {
        std::fs::create_dir_all(runs.join(arm)).unwrap();
        std::fs::write(runs.join(arm).join("summary.json"), body).unwrap();
    };
    for arm in ["a", "b"] {
        summary(arm, "{\"cell\":\"x\",\"exit\":0}");
        std::fs::write(runs.join(arm).join("benchmark.json"), "{}").unwrap();
    }
    // c was skipped on its weights; d crashed before its benchmark.json.
    summary(
        "c",
        "{\"cell\":\"c\",\"exit\":null,\"skipped\":\"weights incomplete\"}",
    );
    summary("d", "{\"cell\":\"d\",\"exit\":3,\"seconds\":4}");
    std::fs::create_dir_all(runs.join("compare")).unwrap();
    std::fs::write(runs.join("compare/compare-clips-a--b.json"), "{}").unwrap();
    let cell = CellFile::parse(CELL, "g.toml").unwrap();
    let mut cfg = config(&f);
    cfg.runs_dir = Some(runs.clone());
    let steps = Lab::steps(&[&cell], &cfg.bin, &cfg.paths()).unwrap();
    let out = f.dir.join("out");
    let mut r = Runner::offline(cfg, &out);
    let outcomes = r.regate(&steps).unwrap().to_vec();
    let got: Vec<(&str, Decision)> = outcomes
        .iter()
        .map(|o| (o.what.as_str(), o.decision))
        .collect();
    assert_eq!(
        got,
        [
            ("gate a--b", Decision::Pass),
            ("gate a--c", Decision::Skipped),
            ("gate a--d", Decision::Fail),
        ]
    );
    assert_eq!(
        text(&out.join("gate/gate-a--b.json")),
        "{\"stage\":\"gate\",\"status\":\"pass\",\"elapsed_s\":1}\n"
    );
    assert!(
        !runs.join("gate").exists(),
        "the run directory is only read"
    );
    assert!(!runs.join("a/stdout.log").exists(), "no arm ran");
    // --check: the run has no gate report, the derived one is a difference.
    assert_eq!(
        r.differences(&steps),
        ["gate/gate-a--b.json: not in the run"]
    );
}
