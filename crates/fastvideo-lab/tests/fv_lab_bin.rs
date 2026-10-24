//! The `fv-lab` binary (feature `cli`): `run --dry-run` prints the script's
//! dry run, `check` reads lab/, `pod up` without `--yes` prints the plan and
//! makes no request, `schema` prints the committed schemas.
#![cfg(feature = "cli")]

use std::path::PathBuf;
use std::process::Command;

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn fv_lab() -> Command {
    let mut c = Command::new(env!("CARGO_BIN_EXE_fv-lab"));
    c.current_dir(repo_root())
        .env_remove("RUNPOD_API_KEY")
        .env_remove("FV_CELLS")
        .env_remove("FV_WORK")
        .env_remove("FV_SCRATCH")
        .env_remove("FV_RUN_TAG")
        .env_remove("FV_GPUCHECK")
        .env_remove("FV_GEN_TIMEOUT_S")
        .env_remove("FV_BASELINE_SHA")
        .env_remove("FV_AUX_DIR")
        // The goldens' `$TAE` (no complete volume copy): the container disk.
        .env("FV_TAE_DIR", "/fvscratch/tae");
    c
}

fn stdout(c: &mut Command) -> String {
    let o = c.output().expect("fv-lab runs");
    assert!(
        o.status.success(),
        "{c:?}: {}\n{}",
        o.status,
        String::from_utf8_lossy(&o.stderr)
    );
    String::from_utf8_lossy(&o.stdout).into_owned()
}

#[test]
fn dry_run_prints_the_scripts_lines() {
    let out = stdout(fv_lab().args([
        "run",
        "rtx6000",
        "--dry-run",
        "--work",
        "/workspace",
        "--scratch",
        "/fvscratch",
        "--tag",
        "1234567-10241200",
    ]));
    let stripped: Vec<&str> = out.lines().map(|l| l.get(11..).unwrap_or("")).collect();
    let fixture = std::fs::read_to_string(
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/rtx6000-dry-run.txt"),
    )
    .unwrap();
    assert_eq!(stripped, fixture.lines().collect::<Vec<_>>());
    assert!(out.lines().all(|l| l.as_bytes().get(9) == Some(&b']')));
}

#[test]
fn a_baseline_tier_needs_the_baseline_commit() {
    let args = [
        "run",
        "b6",
        "--dry-run",
        "--work",
        "/workspace",
        "--scratch",
        "/fvscratch",
        "--tag",
        "1234567-10241200",
    ];
    let no = fv_lab().args(args).output().unwrap();
    assert!(!no.status.success());
    assert!(String::from_utf8_lossy(&no.stderr).contains("--baseline-sha"));
    let out = stdout(fv_lab().args(args).args(["--baseline-sha", "7654321"]));
    // `{root}` is the root fv-lab found (the image's /opt/fastvideo-rs on a
    // pod; this checkout here), as the goldens script writes it.
    let root = std::fs::canonicalize(repo_root()).expect("repo root");
    let out = out.replace(&root.display().to_string(), "/opt/fastvideo-rs");
    let stripped: Vec<&str> = out.lines().map(|l| l.get(11..).unwrap_or("")).collect();
    let fixture = std::fs::read_to_string(
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/b6-dry-run.txt"),
    )
    .unwrap();
    assert_eq!(stripped, fixture.lines().collect::<Vec<_>>());
    // The pod: refused without it, FV_BASELINE_SHA in the run's environment with it.
    let up = [
        "pod",
        "up",
        "--sha",
        "1234567",
        "--family",
        "b6",
        "--tag",
        "1234567-10241200",
    ];
    assert!(!fv_lab().args(up).output().unwrap().status.success());
    let plan = stdout(fv_lab().args(up).args(["--baseline-sha", "7654321"]));
    let v: serde_json::Value = serde_json::from_str(&plan).expect("json plan");
    let cmd = v["requests"][0]["body"]["dockerStartCmd"][2]
        .as_str()
        .unwrap_or("");
    assert!(
        cmd.contains("env FV_BASELINE_SHA=7654321 FV_WORK=/workspace"),
        "{cmd}"
    );
    assert!(cmd.contains("fv-lab run b6 >"), "{cmd}");
}

#[test]
fn dry_run_of_one_cell_and_a_subset() {
    let out = stdout(fv_lab().args([
        "run",
        "rtx6000-ltx25",
        "--dry-run",
        "--cells",
        "ltx25-two-stage",
    ]));
    assert!(out.contains("[rtx6000] dry-run ltx25-two-stage: "));
    assert_eq!(out.matches("dry-run").count(), 1);
    let out = stdout(fv_lab().args([
        "run",
        "rtx6000",
        "--dry-run",
        "--cells",
        "sol-h3,sol-h3-rtx",
    ]));
    assert_eq!(out.matches("] dry-run ").count(), 2);
    assert_eq!(out.matches("(not in FV_CELLS)").count(), 5);
}

#[test]
fn check_and_schema() {
    let out = stdout(fv_lab().arg("check"));
    assert!(out.contains("rtx6000: 2 cell(s), 7 arm(s)"), "{out}");
    assert!(out.contains("b6: 8 cell(s), 16 arm(s)"), "{out}");
    let s = stdout(fv_lab().args(["schema", "cell"]));
    let committed = std::fs::read_to_string(
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("schemas/cell.v1.json"),
    )
    .unwrap();
    assert_eq!(s, committed);
}

#[test]
fn pod_up_without_yes_is_the_plan_and_needs_no_key() {
    let out = stdout(fv_lab().args([
        "pod",
        "up",
        "--sha",
        "1234567",
        "--family",
        "rtx6000",
        "--tag",
        "1234567-10241200",
    ]));
    let v: serde_json::Value = serde_json::from_str(&out).expect("json plan");
    assert_eq!(v["launch"]["name"], "fv-lab-rtx6000-1234567-10241200");
    assert_eq!(v["launch"]["guards"]["cap_s"], 14400);
    assert_eq!(v["launch"]["guards"]["ready_timeout_s"], 600);
    assert_eq!(v["requests"][0]["method"], "POST");
    assert_eq!(v["requests"][0]["body"]["networkVolumeId"], "bz0ofkfx45");
    let bad = fv_lab()
        .args(["pod", "up", "--sha", "nothex", "--family", "rtx6000"])
        .output()
        .unwrap();
    assert!(!bad.status.success());
    let nokey = fv_lab()
        .args([
            "pod", "up", "--sha", "1234567", "--family", "rtx6000", "--yes",
        ])
        .output()
        .unwrap();
    assert!(!nokey.status.success());
    assert!(String::from_utf8_lossy(&nokey.stderr).contains("RUNPOD_API_KEY"));
}
