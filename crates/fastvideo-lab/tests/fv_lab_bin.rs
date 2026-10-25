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
        .env_remove("FV_RUNS_DIR")
        .env_remove("FV_GPUCHECK")
        .env_remove("FV_GEN_TIMEOUT_S")
        .env_remove("FV_BASELINE_SHA")
        .env_remove("FV_AUX_DIR")
        // The goldens' `$TAE` (the volume's complete auxiliary/tae copy).
        .env("FV_TAE_DIR", "/workspace/weights/auxiliary/tae");
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

/// `pod up --wall-cap-s` lowers the tier's cap (runpod-http.sh maps
/// FV_POD_CAP_S to it): the guard's deadline and the pod's own cap watchdog;
/// 0 and anything above the tier's 14400 s are refused.
#[test]
fn wall_cap_can_only_be_lowered() {
    let up = [
        "pod",
        "up",
        "--sha",
        "1234567",
        "--family",
        "rtx6000",
        "--tag",
        "1234567-10241200",
    ];
    let plan = stdout(fv_lab().args(up).args(["--wall-cap-s", "1800"]));
    let v: serde_json::Value = serde_json::from_str(&plan).expect("json plan");
    assert_eq!(v["launch"]["guards"]["cap_s"], 1800, "{plan}");
    let cmd = v["requests"][0]["body"]["dockerStartCmd"][2]
        .as_str()
        .unwrap_or("");
    assert!(cmd.contains("sleep 1800\n"), "{cmd}");
    assert!(!cmd.contains("sleep 14400"), "{cmd}");
    let default = stdout(fv_lab().args(up));
    assert!(default.contains("sleep 14400"), "the tier's cap by default");
    for bad in ["0", "14401"] {
        let o = fv_lab()
            .args(up)
            .args(["--wall-cap-s", bad])
            .output()
            .unwrap();
        assert!(!o.status.success(), "--wall-cap-s {bad}");
        assert!(
            String::from_utf8_lossy(&o.stderr).contains("can only be lowered"),
            "{bad}"
        );
    }
}

/// `--runs-dir` / `FV_RUNS_DIR` replace the run directory (`{runs}`): how
/// runpod-matrix.sh `headline` runs a retired tier's cells into its own.
#[test]
fn runs_dir_replaces_the_run_directory() {
    let args = [
        "run",
        "rtx6000",
        "--dry-run",
        "--cells",
        "sol-h3",
        "--scratch",
        "/fvscratch",
        "--tag",
        "1234567-10241200",
    ];
    let def = stdout(fv_lab().args(args));
    assert!(
        def.contains("/fvscratch/runs/rtx6000/1234567-10241200/sol-h3/frames"),
        "{def}"
    );
    let flag = stdout(
        fv_lab()
            .args(args)
            .args(["--runs-dir", "/fvscratch/runs/headline/t"]),
    );
    let env = stdout(
        fv_lab()
            .args(args)
            .env("FV_RUNS_DIR", "/fvscratch/runs/headline/t"),
    );
    for out in [&flag, &env] {
        assert!(
            out.contains("--clip-dir /fvscratch/runs/headline/t/sol-h3/frames"),
            "{out}"
        );
        assert!(!out.contains("/runs/rtx6000/"), "{out}");
    }
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
    // Refused: the run's own commit as its baseline, a bad or empty
    // FV_BASELINE_SHA in --extra-env, and the commit given twice.
    for (extra, want) in [
        (vec!["--baseline-sha", "1234567"], "own commit"),
        (vec!["--extra-env", "FV_BASELINE_SHA=1234567"], "own commit"),
        (
            vec!["--extra-env", "FV_BASELINE_SHA="],
            "FV_BASELINE_SHA ``",
        ),
        (
            vec!["--extra-env", "FV_BASELINE_SHA=main"],
            "7 lower-case hex",
        ),
        (vec!["--baseline-sha", "xyz"], "--baseline-sha `xyz`"),
        (
            vec![
                "--baseline-sha",
                "7654321",
                "--extra-env",
                "FV_BASELINE_SHA=7654321",
            ],
            "twice",
        ),
    ] {
        let o = fv_lab().args(up).args(&extra).output().unwrap();
        assert!(!o.status.success(), "{extra:?}");
        let err = String::from_utf8_lossy(&o.stderr);
        assert!(err.contains(want), "{extra:?}: {err}");
    }
    let o = fv_lab()
        .args(args)
        .args(["--baseline-sha", "main"])
        .output()
        .unwrap();
    assert!(String::from_utf8_lossy(&o.stderr).contains("--baseline-sha / FV_BASELINE_SHA `main`"));
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

#[test]
fn pod_down_from_the_ledger_needs_no_record() {
    // An interrupted `pod up` leaves only its ledger line: `pod down
    // --from-ledger` reads it (no pod record); nothing to delete needs no key.
    let state = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("fv-lab-down-ledger");
    let _ = std::fs::remove_dir_all(&state);
    std::fs::create_dir_all(&state).unwrap();
    let out = stdout(
        fv_lab()
            .args(["pod", "down", "--from-ledger", "--state-dir"])
            .arg(&state),
    );
    assert!(out.contains("no pod to delete"), "{out}");
    std::fs::write(
        state.join("ledger.tsv"),
        "2026-10-24T21:00:00Z\trunpod-pod-created abc123 fv-lab-gold-x gpu=G image=I\n\
         2026-10-24T21:00:09Z\trunpod-pod-created def456 fv-lab-gold-y gpu=G image=I\n\
         2026-10-24T21:01:00Z\trunpod-pod-deleted def456 down\n",
    )
    .unwrap();
    // One pod left to delete: it gets as far as the key (none here), and fails.
    let o = fv_lab()
        .args(["pod", "down", "--from-ledger", "--state-dir"])
        .arg(&state)
        .output()
        .unwrap();
    assert!(!o.status.success());
    assert!(String::from_utf8_lossy(&o.stderr).contains("RUNPOD_API_KEY"));
    // Neither an id nor --from-ledger, or both: refused.
    assert!(!fv_lab()
        .args(["pod", "down"])
        .output()
        .unwrap()
        .status
        .success());
    assert!(!fv_lab()
        .args(["pod", "down", "abc123", "--from-ledger"])
        .output()
        .unwrap()
        .status
        .success());
}

#[test]
fn pod_up_cells_must_be_arms_of_the_tier() {
    let ok = fv_lab()
        .args([
            "pod", "up", "--sha", "1234567", "--family", "gold", "--cells", "sol-h3",
        ])
        .output()
        .unwrap();
    assert!(
        ok.status.success(),
        "{}",
        String::from_utf8_lossy(&ok.stderr)
    );
    for bad in ["sol-h3x", "x"] {
        let o = fv_lab()
            .args([
                "pod", "up", "--sha", "1234567", "--family", "gold", "--cells", bad,
            ])
            .output()
            .unwrap();
        assert!(!o.status.success(), "{bad}");
        assert!(String::from_utf8_lossy(&o.stderr).contains("is not an arm of tier `gold`"));
    }
}

/// `fv-lab run b4a --dry-run` (D1d): the script's dry run, verdict lines
/// included (the arm directories removed, the verdict's description, the
/// family verdict). `{root}` is this checkout here, `/opt/fastvideo-rs` in
/// the fixture (the goldens' image root).
#[test]
fn b4a_dry_run_prints_the_scripts_verdict_lines() {
    let root = repo_root().canonicalize().unwrap().display().to_string();
    let out = stdout(fv_lab().args([
        "run",
        "b4a",
        "--dry-run",
        "--baseline-sha",
        "7654321",
        "--work",
        "/workspace",
        "--scratch",
        "/fvscratch",
        "--tag",
        "1234567-10241200",
    ]));
    let ours: Vec<String> = out
        .lines()
        .map(|l| {
            l.get(11..)
                .unwrap_or("")
                .replace(&root, "/opt/fastvideo-rs")
        })
        .collect();
    let fixture = std::fs::read_to_string(
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/b4a-dry-run.txt"),
    )
    .unwrap();
    assert_eq!(ours, fixture.lines().collect::<Vec<_>>());
    assert!(ours
        .iter()
        .any(|l| l.contains("family verdict b4_family_verdict b4a")));
}

/// A finished b4a run directory, as a fetch leaves it, under `<dir>/b4a/<tag>`.
fn b4a_run(dir: &std::path::Path, pr_reuses: bool) -> PathBuf {
    let runs = dir.join("b4a/1234567-10241200");
    let cell = "b4a-h3-fullopt-env";
    for arm in ["base", "pr"] {
        let d = runs.join(format!("{cell}-{arm}"));
        std::fs::create_dir_all(&d).unwrap();
        std::fs::write(
            d.join("summary.json"),
            format!("{{\"cell\":\"{cell}-{arm}\",\"family\":\"b4a\",\"exit\":0,\"seconds\":9,\"ended\":\"x\"}}\n"),
        )
        .unwrap();
        let step = if arm == "pr" && !pr_reuses {
            "compute"
        } else {
            "REUSE"
        };
        std::fs::write(
            d.join("stderr.log"),
            format!("[INFO] h3 teacache: threshold 0.1\n[INFO] [fastvideo] h3 teacache step 1: {step} (x)\nother\n"),
        )
        .unwrap();
    }
    std::fs::create_dir_all(runs.join("compare")).unwrap();
    std::fs::write(
        runs.join(format!("compare/compare-clips-{cell}-base--{cell}-pr.json")),
        "{\n  \"checks\": [\n    {\n      \"name\": \"off_identity\",\n      \"ok\": true\n    },\n    {\n      \"name\": \"off_identity_frame_count\",\n      \"ok\": true\n    }\n  ],\n  \"status\": \"pass\"\n}\n",
    )
    .unwrap();
    runs
}

/// `fv-lab gate <run dir>` (D1d): the verdicts of a fetched run evaluated
/// again offline, the tier from the directory's parent, the cells the run
/// holds; exit 0 on a pass, 1 on a failure, and the files the pod writes.
#[test]
fn gate_evaluates_a_finished_runs_verdicts_offline() {
    let tmp = std::env::temp_dir().join(format!("fv-lab-bin-gate-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&tmp);
    let runs = b4a_run(&tmp.join("pass"), true);
    let out = stdout(fv_lab().arg("gate").arg(&runs));
    assert!(out.contains("b4a b4a-h3-fullopt-env: pass"), "{out}");
    assert!(
        out.contains("fv-lab gate b4a: 2 evaluated, 0 failed"),
        "{out}"
    );
    assert_eq!(
        std::fs::read_to_string(runs.join("b4a.verdict.json")).unwrap(),
        "{\"family\":\"b4a\",\"verdict\":\"pass\",\"cells\":\"b4a-h3-fullopt-env\",\"why\":\"\"}\n"
    );
    assert_eq!(
        std::fs::read_to_string(runs.join("b4a-h3-fullopt-env-pr/cache-evidence.log")).unwrap(),
        "h3 teacache: threshold 0.1\nh3 teacache step 1: REUSE (x)\n"
    );

    // The PR arm reused nothing: its evidence differs too. Into --out.
    let runs = b4a_run(&tmp.join("fail"), false);
    let elsewhere = tmp.join("out");
    let o = fv_lab()
        .arg("gate")
        .arg(&runs)
        .arg("--out")
        .arg(&elsewhere)
        .output()
        .unwrap();
    assert_eq!(
        o.status.code(),
        Some(1),
        "{}",
        String::from_utf8_lossy(&o.stderr)
    );
    assert_eq!(
        std::fs::read_to_string(elsewhere.join("b4a-b4a-h3-fullopt-env.verdict.json")).unwrap(),
        "{\"cell\":\"b4a-h3-fullopt-env\",\"verdict\":\"fail\",\"why\":\"pr:no-reuse evidence-differs\"}\n"
    );
    assert!(
        !runs.join("b4a.verdict.json").exists(),
        "--out leaves the run alone"
    );

    // Not under a tier's name: --family says which.
    let odd = tmp.join("odd");
    std::fs::create_dir_all(&odd).unwrap();
    let o = fv_lab().arg("gate").arg(&odd).output().unwrap();
    assert!(!o.status.success());
    assert!(String::from_utf8_lossy(&o.stderr).contains("pass --family"));
    // A run of a tier with nothing to evaluate fails: no evidence.
    let o = fv_lab()
        .args(["gate", "--family", "b4a"])
        .arg(&odd)
        .output()
        .unwrap();
    assert_eq!(o.status.code(), Some(1));
    assert!(String::from_utf8_lossy(&o.stdout).contains("nothing to evaluate"));
    let _ = std::fs::remove_dir_all(&tmp);
}
