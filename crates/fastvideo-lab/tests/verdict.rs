//! The verdict step (docs/adr/0019-fv-lab.md §11, D1d) against the script it
//! replaces: `runpod-matrix.sh b4a` (its real branch, copied into a fake
//! root) and `fv-lab run b4a` (the real `lab/cells/b4a-*.toml`) on the same
//! fake box: a fake fv-gpucheck whose arms log step-cache evidence, a fake
//! nvidia-smi, verify-weights.sh and fetch-baseline.sh. Both runners must
//! write the same verdict files, evidence files and evidence diffs, byte for
//! byte, and the same verdict lines in `live.log`; and `fv-lab gate` must
//! derive the same verdicts again from either run directory alone.
//!
//! The fake box makes one cell pass, one fail on its evidence only, and one
//! fail on everything (exit, reuse, identity, evidence).

use std::os::unix::fs::PermissionsExt as _;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Duration;

use fastvideo_lab::plan::{CellFile, Lab, Step};
use fastvideo_lab::run::{RunConfig, Runner};

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

/// fv-gpucheck on the fake box. `gen`: frames under `--clip-dir`, step-cache
/// lines on stderr (with the log's bracketed prefixes), per cell (the recipe
/// argument) and arm (`$0` under `baseline-<sha>/` is the baseline):
/// `env` identical arms that reuse; `prof` the same frames but one decision
/// differs; `tea` the PR arm exits 3, reuses nothing, other frames.
/// `compare-clips`: an off-identity report (pretty-printed, as the real one)
/// from the two first frames' bytes.
const GPUCHECK: &str = r#"if [ "$1" = --out ]; then
  out="$2"; tag="$4"; sub="$5"; shift 5
  case "$sub" in
    compare-clips)
      ok=false
      cmp -s "$2/frame-000.png" "$4/frame-000.png" && ok=true
      st=fail; [ "$ok" = true ] && st=pass
      printf '{\n  "checks": [\n    {\n      "name": "off_identity",\n      "ok": %s\n    },\n    {\n      "name": "off_identity_frame_count",\n      "ok": true\n    }\n  ],\n  "status": "%s"\n}\n' "$ok" "$st" >"$out/compare-clips-$tag.json"
      echo '{"psnr_mean":99.0,"x":1}' >&2 ;;
    gate)
      echo "gate verdict: pass" >&2
      echo '{"verdict":"pass"}' >"$out/gate-$tag.json" ;;
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
}

fn fake(name: &str) -> Fake {
    let dir = std::env::temp_dir().join(format!("fv-lab-verdict-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let f = Fake { dir };
    let gpu = f.root().join("scripts/gpu");
    std::fs::create_dir_all(&gpu).unwrap();
    // The script and what it sources, as committed.
    for s in [
        "runpod-matrix.sh",
        "cuda-13.pins",
        "b4-verdict.sh",
        "fv-lab-retired.sh",
    ] {
        std::fs::copy(repo_root().join("scripts/gpu").join(s), gpu.join(s)).unwrap();
    }
    script(
        &gpu.join("verify-weights.sh"),
        "echo \"verify $1\"\nexit 0\n",
    );
    script(
        &gpu.join("fetch-baseline.sh"),
        &format!(
            "mkdir -p \"$2\" && cp {} \"$2/fv-gpucheck\" && echo \"fv-gpucheck sha-$1 -> $2\"\n",
            f.bin().join("fv-gpucheck").display()
        ),
    );
    std::fs::write(gpu.join("gate-policy.toml"), "# policy\n").unwrap();
    script(&f.bin().join("fv-gpucheck"), GPUCHECK);
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

/// `runpod-matrix.sh b4a` on the fake box: its run directory.
fn script_run(f: &Fake, b4a_cells: Option<&str>) -> PathBuf {
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
        .env("LC_ALL", "C")
        .env("FV_WORK", f.dir.join("work"))
        .env("FV_SCRATCH", f.dir.join("script"))
        .env("FV_RUN_TAG", TAG)
        .env("FV_BASELINE_SHA", BASELINE_SHA)
        .env("FV_GPUCHECK", f.bin().join("fv-gpucheck"))
        .env("FV_GEN_TIMEOUT_S", "60");
    if let Some(cells) = b4a_cells {
        c.env("FV_B4A_CELLS", cells);
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

fn b4a_cells(lab: &Lab) -> Vec<&CellFile> {
    lab.resolve("b4a", None).expect("the b4a tier").1
}

/// `fv-lab run b4a` (the committed cells) on the fake box: its run directory.
fn lab_run(f: &Fake, only: Option<Vec<String>>) -> PathBuf {
    let lab = Lab::load(&repo_root()).expect("lab/ parses");
    let mut cfg = config(f);
    cfg.only = only;
    let steps = Lab::steps(&b4a_cells(&lab), &cfg.bin, &cfg.paths()).expect("render");
    let mut r = Runner::new(cfg);
    r.run(&steps).expect("fv-lab run b4a");
    f.dir.join("lab/runs/b4a").join(TAG)
}

fn bytes(p: &Path) -> Vec<u8> {
    std::fs::read(p).unwrap_or_else(|e| panic!("{}: {e}", p.display()))
}

fn text(p: &Path) -> String {
    String::from_utf8(bytes(p)).unwrap()
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

/// Every file the verdict step writes, relative to the run directory.
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

fn assert_same(a: &Path, b: &Path, files: &[String]) {
    for f in files {
        assert_eq!(
            text(&a.join(f)),
            text(&b.join(f)),
            "{f}: {} vs {}",
            a.display(),
            b.display()
        );
    }
}

#[test]
fn fv_lab_writes_the_scripts_verdicts_byte_for_byte() {
    let f = fake("all");
    let theirs = script_run(&f, None);
    let ours = lab_run(&f, None);
    let files = verdict_files(&CELLS);
    assert_same(&ours, &theirs, &files);
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

    // The frame, compare and summary records agree too (fv-lab-compare-runs.sh's view).
    for c in CELLS {
        let r = format!("compare/compare-clips-{c}-base--{c}-pr.json");
        assert_eq!(text(&ours.join(&r)), text(&theirs.join(&r)), "{r}");
        for arm in ["base", "pr"] {
            let d = format!("{c}-{arm}/frames.sha256.json");
            assert_eq!(
                text(&ours.join(&d)).replace(&f.dir.display().to_string(), ""),
                text(&theirs.join(&d)).replace(&f.dir.display().to_string(), ""),
                "{d}"
            );
        }
    }

    // `fv-lab gate` derives the same verdicts from either run directory alone.
    let lab = Lab::load(&repo_root()).expect("lab/ parses");
    for (name, runs) in [("ours", &ours), ("theirs", &theirs)] {
        let out = f.dir.join(format!("regate-{name}"));
        let mut cfg = config(&f);
        cfg.runs_dir = Some(runs.clone());
        cfg.only = Some(
            b4a_cells(&lab)
                .iter()
                .flat_map(|c| &c.arms)
                .map(|a| a.name.clone())
                .filter(|a| runs.join(a).join("summary.json").is_file())
                .collect(),
        );
        let steps = Lab::steps(&b4a_cells(&lab), &cfg.bin, &cfg.paths()).expect("render");
        let mut r = Runner::offline(cfg, &out);
        let outcomes = r.regate(&steps).to_vec();
        assert_eq!(
            outcomes.iter().map(|o| o.pass).collect::<Vec<_>>(),
            [true, false, false, false],
            "{name}: {outcomes:?}"
        );
        assert_same(&out, &theirs, &files);
        // Offline nothing ran and nothing was removed.
        assert!(runs.join("compare").is_dir() && runs.join("b4a-h3-4step-tea-pr").is_dir());
        assert!(!out.join("live.log").exists());
    }
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
        Some(vec![
            "b4a-h3-fullopt-env-base".into(),
            "b4a-h3-fullopt-env-pr".into(),
        ]),
    );
    assert_same(&ours, &theirs, &verdict_files(&["b4a-h3-fullopt-env"]));
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
    lab_run(&f, Some(vec!["b4a-h3-fullopt-env-base".into()]));
    assert!(!runs.join("b4a-b4a-h3-gone.verdict.json").exists());
    assert!(!runs.join("b4a-h3-fullopt-env-pr").exists());
    assert_eq!(
        text(&runs.join("b4a-b4a-h3-fullopt-env.verdict.json")),
        "{\"cell\":\"b4a-h3-fullopt-env\",\"verdict\":\"fail\",\"why\":\"pr:exit pr:no-run not-identical\"}\n"
    );
}

#[test]
fn the_dry_run_holds_the_verdict_steps_in_the_scripts_order() {
    let lab = Lab::load(&repo_root()).expect("lab/ parses");
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
/// into `--out`, the run directory untouched.
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

[[gate]]
baseline = "a"
candidate = "b"
kind = "exact"

[[gate]]
baseline = "a"
candidate = "c"
"#;
    let f = fake("gate");
    let runs = f.dir.join("finished");
    for arm in ["a", "b"] {
        std::fs::create_dir_all(runs.join(arm)).unwrap();
        std::fs::write(runs.join(arm).join("benchmark.json"), "{}").unwrap();
    }
    std::fs::create_dir_all(runs.join("compare")).unwrap();
    std::fs::write(runs.join("compare/compare-clips-a--b.json"), "{}").unwrap();
    let cell = CellFile::parse(CELL, "g.toml").unwrap();
    let mut cfg = config(&f);
    cfg.runs_dir = Some(runs.clone());
    let steps = Lab::steps(&[&cell], &cfg.bin, &cfg.paths()).unwrap();
    let out = f.dir.join("out");
    let mut r = Runner::offline(cfg, &out);
    let outcomes = r.regate(&steps).to_vec();
    // a--c has no benchmark.json for c: nothing to gate, not a failure.
    assert_eq!(outcomes.len(), 1, "{outcomes:?}");
    assert_eq!(outcomes[0].what, "gate a--b");
    assert!(outcomes[0].pass);
    assert_eq!(
        text(&out.join("gate/gate-a--b.json")),
        "{\"verdict\":\"pass\"}\n"
    );
    assert!(
        !runs.join("gate").exists(),
        "the run directory is only read"
    );
    assert!(!runs.join("a/stdout.log").exists(), "no arm ran");
}
