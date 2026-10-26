//! The `fv-lab` binary (feature `cli`): `run --dry-run` prints the script's
//! dry run, `check` reads lab/, `pod up` without `--yes` prints the plan and
//! makes no request, `schema` prints the committed schemas.
#![cfg(feature = "cli")]

mod common;

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

/// The `gold` tier (`weights = "volume+r2"`, ADR-0019 §16) through the
/// binary: its R2 plan comes from the repository's `volumes.sh` and
/// `weights-r2.tsv`, and the dry run is its golden, with or without an R2
/// key in the environment (never printed).
#[test]
fn gold_dry_run_prints_its_r2_plan() {
    let fixture = std::fs::read_to_string(
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/gold-dry-run.txt"),
    )
    .unwrap();
    for key in [None, Some("dry-run-secret-value")] {
        let mut c = fv_lab();
        c.args([
            "run",
            "gold",
            "--all",
            "--dry-run",
            "--work",
            "/workspace",
            "--scratch",
            "/fvscratch",
            "--tag",
            "1234567-10241200",
        ]);
        match key {
            Some(k) => c
                .env("FV_WEIGHTS_R2_ACCESS_KEY_ID", k)
                .env("FV_WEIGHTS_R2_SECRET_ACCESS_KEY", k),
            None => c
                .env_remove("FV_WEIGHTS_R2_ACCESS_KEY_ID")
                .env_remove("FV_WEIGHTS_R2_SECRET_ACCESS_KEY"),
        };
        let o = c.output().expect("fv-lab runs");
        assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
        let out = String::from_utf8_lossy(&o.stdout);
        let stripped: Vec<&str> = out.lines().map(|l| l.get(11..).unwrap_or("")).collect();
        assert_eq!(stripped, fixture.lines().collect::<Vec<_>>());
        assert!(!out.contains("dry-run-secret-value"));
        assert!(!String::from_utf8_lossy(&o.stderr).contains("dry-run-secret-value"));
    }
}

/// A `volume+r2` run never puts its weights root on the volume (ADR-0019
/// §16.3): without `--scratch` the scratch disk is the work directory, and a
/// scratch under the work directory or the volume mount is refused too; exit
/// 2, before anything is written.
#[test]
fn a_volume_r2_root_on_the_volume_is_refused() {
    for extra in [
        &["--work", "/tmp/fv-lab-w"][..],
        &["--work", "/tmp/fv-lab-w", "--scratch", "/tmp/fv-lab-w/s"][..],
        &["--work", "/tmp/fv-lab-w", "--scratch", "/workspace/s"][..],
    ] {
        let o = fv_lab()
            .args(["run", "gold", "--all", "--dry-run"])
            .args(extra)
            .output()
            .expect("fv-lab runs");
        assert_eq!(o.status.code(), Some(2), "{extra:?}");
        let err = String::from_utf8_lossy(&o.stderr);
        assert!(
            err.contains("FATAL") && err.contains("--scratch / FV_SCRATCH"),
            "{err}"
        );
        assert!(o.stdout.is_empty(), "{extra:?}");
    }
}

/// The restored modes (ADR-0019 §12) through the binary: each tier's dry run
/// is the script's in that mode at 5987e6a4, the family named by the tier;
/// a gen cell of `ltxvae-gen` runs in that tier, a shared cell in `ltxvae`
/// (the first tier listing it) unless `--family` says otherwise.
#[test]
fn restored_modes_dry_run_as_the_script_did() {
    for r in common::restored() {
        let out = stdout(fv_lab().args([
            "run",
            &r.tier,
            "--dry-run",
            "--work",
            "/workspace",
            "--scratch",
            "/fvscratch",
            "--tag",
            "1234567-10241200",
        ]));
        let ours: Vec<&str> = out.lines().map(|l| l.get(11..).unwrap_or("")).collect();
        let fixture = std::fs::read_to_string(
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join(format!("fixtures/{}-dry-run.txt", r.tier)),
        )
        .unwrap();
        let theirs: Vec<String> = fixture.lines().map(|l| r.dry_run_line(l)).collect();
        assert_eq!(ours, theirs, "{}", r.tier);
    }
    let out = stdout(fv_lab().args(["run", "ltxvae-gen-4k5s", "--dry-run"]));
    assert!(
        out.contains("[ltxvae-gen] dry-run ltx25-4k5s-gen: "),
        "{out}"
    );
    let out = stdout(fv_lab().args(["run", "ltxvae-4k5s", "--dry-run"]));
    assert!(out.contains("[ltxvae] dry-run ltxvae-4k5s-fast: "), "{out}");
    let out = stdout(fv_lab().args(["run", "ltxvae-4k5s", "--family", "ltxvae-gen", "--dry-run"]));
    assert!(
        out.contains("[ltxvae-gen] dry-run ltxvae-4k5s-fast: "),
        "{out}"
    );
    let out = stdout(fv_lab().arg("check"));
    assert!(
        out.contains("precision-nvfp4: 2 cell(s), 4 arm(s)"),
        "{out}"
    );
    assert!(out.contains("ltxvae-gen: 6 cell(s), 16 arm(s)"), "{out}");
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
    // The gold tier's plan (ADR-0019 §16): the R2 key as Runpod secret
    // references in the pod's env, and the tier's 300 GB disk.
    let plan = String::from_utf8_lossy(&ok.stdout);
    for want in [
        "\"FV_WEIGHTS_R2_ACCESS_KEY_ID\": \"{{ RUNPOD_SECRET_fv_r2_access_key_id }}\"",
        "\"FV_WEIGHTS_R2_SECRET_ACCESS_KEY\": \"{{ RUNPOD_SECRET_fv_r2_secret_access_key }}\"",
        "\"containerDiskInGb\": 300",
    ] {
        assert!(plan.contains(want), "{want}\n{plan}");
    }
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

/// What no tier reads is refused, not ignored (ADR-0019 §12, `knobs`): a
/// restored mode's setting in `fv-lab run`'s environment or in `pod up
/// --extra-env` names the tier, every other mode knob is refused, and so is
/// what the start command sets after `--extra-env`.
#[test]
fn mode_knobs_are_refused_naming_the_restored_tier() {
    let fails = |c: &mut Command, want: &str| {
        let o = c.output().expect("fv-lab runs");
        let err = String::from_utf8_lossy(&o.stderr).into_owned();
        assert!(!o.status.success(), "{c:?} succeeded");
        assert!(err.contains(want), "{c:?}: {err}");
    };
    fails(
        fv_lab()
            .args(["run", "precision", "--dry-run"])
            .env("FV_PRECISION_ARM", "ltx-nvfp4"),
        "FV_PRECISION_ARM=ltx-nvfp4 was a mode of runpod-matrix.sh precision; it is the fv-lab tier precision-nvfp4 now",
    );
    fails(
        fv_lab()
            .args(["run", "ltxvae", "--dry-run"])
            .env("FV_VAE_TRACE", "0")
            .env("FV_VAE_GEN", "1"),
        "refused, fv-lab would not read them: FV_VAE_GEN=1 was a mode of runpod-matrix.sh ltxvae",
    );
    fails(
        fv_lab()
            .args(["run", "rtx6000", "--dry-run"])
            .env("FV_TECH_CELLS", "x"),
        "FV_TECH_CELLS: no tier reads it",
    );
    let up = |extra: &str| {
        let mut c = fv_lab();
        c.args([
            "pod",
            "up",
            "--sha",
            "1234567",
            "--family",
            "precision",
            "--tag",
            "1234567-10241200",
            "--extra-env",
            extra,
        ]);
        c
    };
    fails(
        &mut up("FV_PRECISION_ARM=ltx-nvfp4"),
        "--extra-env FV_PRECISION_ARM=ltx-nvfp4 was a mode of runpod-matrix.sh precision; it is the fv-lab tier precision-nvfp4 now",
    );
    fails(&mut up("FV_NVFP4_WORKLOADS=4k5s"), "narrow them by arm");
    fails(
        &mut up("FV_WORK=/x"),
        "--extra-env FV_WORK: the pod's start command sets it",
    );
    // Unread values pass: FV_PRECISION_ARM is refused as a knob on any value,
    // and a plain setting goes through to the plan.
    fails(
        &mut up("FV_PRECISION_ARM=all"),
        "FV_PRECISION_ARM: no tier reads it",
    );
    let out = stdout(&mut up("FASTVIDEO_TEST_X=1"));
    assert!(out.contains("FASTVIDEO_TEST_X=1"), "{out}");
    // The environment of a plain run is fine (FV_WORK and friends are its inputs).
    stdout(
        fv_lab()
            .args(["run", "precision-nvfp4", "--dry-run"])
            .env("FV_PRECISION_ARM", ""),
    );
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

/// A finished b4a run directory, as a fetch leaves it, under `<dir>/b4a/<tag>`,
/// with fv-lab's selection record of the one pair when `record`.
fn b4a_run(dir: &std::path::Path, pr_reuses: bool, record: bool) -> PathBuf {
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
    if record {
        std::fs::write(
            runs.join("b4a.arms.json"),
            format!("{{\"family\":\"b4a\",\"arms\":[\"{cell}-base\",\"{cell}-pr\"]}}\n"),
        )
        .unwrap();
    }
    runs
}

fn out_err(c: &mut Command) -> (Option<i32>, String, String) {
    let o = c.output().unwrap();
    (
        o.status.code(),
        String::from_utf8_lossy(&o.stdout).into_owned(),
        String::from_utf8_lossy(&o.stderr).into_owned(),
    )
}

/// `fv-lab gate <run dir>` (D1d): the verdicts of a fetched run evaluated
/// again offline, the tier from the directory's parent, the arms from the
/// run's selection record (or --cells), written beside the run
/// (`<run>.regate`), never into it; exit 0 on a pass, 1 on a failure.
#[test]
fn gate_evaluates_a_finished_runs_verdicts_offline() {
    let tmp = std::env::temp_dir().join(format!("fv-lab-bin-gate-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&tmp);
    let runs = b4a_run(&tmp.join("pass"), true, true);
    let out = stdout(fv_lab().arg("gate").arg(&runs));
    assert!(out.contains("b4a b4a-h3-fullopt-env: pass"), "{out}");
    assert!(
        out.contains("fv-lab gate b4a: 2 evaluated, 0 failed, 0 skipped"),
        "{out}"
    );
    let sibling = runs.with_file_name("1234567-10241200.regate");
    assert_eq!(
        std::fs::read_to_string(sibling.join("b4a.verdict.json")).unwrap(),
        "{\"family\":\"b4a\",\"verdict\":\"pass\",\"cells\":\"b4a-h3-fullopt-env\",\"why\":\"\"}\n"
    );
    assert_eq!(
        std::fs::read_to_string(sibling.join("b4a-h3-fullopt-env-pr/cache-evidence.log")).unwrap(),
        "h3 teacache: threshold 0.1\nh3 teacache step 1: REUSE (x)\n"
    );
    assert!(
        !runs.join("b4a.verdict.json").exists(),
        "the run is only read"
    );
    assert!(!runs
        .join("b4a-h3-fullopt-env-pr/cache-evidence.log")
        .exists());

    // --check: the run holds none of the derived files: each a difference.
    let (code, out, _) = out_err(fv_lab().arg("gate").arg(&runs).arg("--check"));
    assert_eq!(code, Some(1), "{out}");
    assert!(
        out.contains("DIFF b4a.verdict.json: not in the run"),
        "{out}"
    );
    // Copied into the run, they are what fv-lab derives: --check passes.
    for f in [
        "b4a.verdict.json",
        "b4a-b4a-h3-fullopt-env.verdict.json",
        "b4a-h3-fullopt-env.evidence.diff",
        "b4a-h3-fullopt-env-base/cache-evidence.log",
        "b4a-h3-fullopt-env-pr/cache-evidence.log",
    ] {
        std::fs::copy(sibling.join(f), runs.join(f)).unwrap();
    }
    let (code, out, err) = out_err(fv_lab().arg("gate").arg(&runs).arg("--check"));
    assert_eq!(code, Some(0), "{out}{err}");
    assert!(out.contains("0 difference(s)"), "{out}");
    // Never into the run itself.
    let (code, _, err) = out_err(fv_lab().arg("gate").arg(&runs).arg("--out").arg(&runs));
    assert_ne!(code, Some(0));
    assert!(err.contains("is the run directory or inside it"), "{err}");

    // The PR arm reused nothing: its evidence differs too. Into --out.
    let runs = b4a_run(&tmp.join("fail"), false, true);
    let elsewhere = tmp.join("out");
    let (code, _, err) = out_err(fv_lab().arg("gate").arg(&runs).arg("--out").arg(&elsewhere));
    assert_eq!(code, Some(1), "{err}");
    assert_eq!(
        std::fs::read_to_string(elsewhere.join("b4a-b4a-h3-fullopt-env.verdict.json")).unwrap(),
        "{\"cell\":\"b4a-h3-fullopt-env\",\"verdict\":\"fail\",\"why\":\"pr:no-reuse evidence-differs\"}\n"
    );

    // No selection record and 2 of the 6 arms (a script run, or one cut
    // short): no guess; --cells names the selection.
    let runs = b4a_run(&tmp.join("norecord"), true, false);
    let (code, _, err) = out_err(fv_lab().arg("gate").arg(&runs));
    assert_ne!(code, Some(0));
    assert!(
        err.contains("holds 2 of the 6 arms") && err.contains("--cells"),
        "{err}"
    );
    let out = stdout(
        fv_lab()
            .arg("gate")
            .arg(&runs)
            .args(["--cells", "b4a-h3-fullopt-env-base,b4a-h3-fullopt-env-pr"]),
    );
    assert!(out.contains("b4a family: pass"), "{out}");

    // Not under a tier's name: --family says which.
    let odd = tmp.join("odd");
    std::fs::create_dir_all(&odd).unwrap();
    let (code, _, err) = out_err(fv_lab().arg("gate").arg(&odd));
    assert_ne!(code, Some(0));
    assert!(err.contains("pass --family"), "{err}");
    let _ = std::fs::remove_dir_all(&tmp);
}

/// F8: `FV_B4A_CELLS`, which fv-lab would ignore (and run every b4a cell),
/// is refused by `pod up --extra-env` and by `run`.
#[test]
fn fv_b4a_cells_is_refused_not_ignored() {
    let (code, _, err) = out_err(fv_lab().args([
        "pod",
        "up",
        "--sha",
        "1234567",
        "--family",
        "b4a",
        "--baseline-sha",
        "7654321",
        "--extra-env",
        "FV_B4A_CELLS=b4a-h3-4step-tea",
    ]));
    assert_ne!(code, Some(0));
    assert!(err.contains("not FV_B4A_CELLS"), "{err}");
    let (code, _, err) = out_err(
        fv_lab()
            .args(["run", "b4a", "--dry-run", "--baseline-sha", "7654321"])
            .env("FV_B4A_CELLS", "b4a-h3-4step-tea"),
    );
    assert_ne!(code, Some(0));
    assert!(err.contains("not FV_B4A_CELLS"), "{err}");
}

#[test]
fn report_summary_and_table_print_the_scripts_output() {
    let fx = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/report");
    let read =
        |p: PathBuf| std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("{}: {e}", p.display()));
    // The tier from the run directory's parent; the retired branch's lines.
    let out = stdout(fv_lab().args([
        "report",
        "summary",
        "artifacts/runpod/trace/b0d0bb8-10251650",
    ]));
    assert_eq!(out, read(fx.join("summary/trace--b0d0bb8-10251650.txt")));
    let out = stdout(fv_lab().args([
        "report",
        "summary",
        "--kind",
        "h3-graphs",
        "artifacts/perf/h3-denoise-graphs/run-2026-10-14",
    ]));
    assert_eq!(out, read(fx.join("h3-graphs.txt")));
    // --json writes the record as the script's json.dump does.
    let dir = std::env::temp_dir().join(format!("fv-lab-bin-report-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let json = dir.join("out.json");
    let out = stdout(
        fv_lab()
            .args(["report", "summary", "--kind", "h3-tier-options", "--json"])
            .arg(&json)
            .arg("artifacts/perf/h3-nvfp4-tier-options"),
    );
    assert_eq!(out, read(fx.join("h3-tier-options.md")));
    assert_eq!(read(json.clone()), read(fx.join("h3-tier-options.json")));
    // timings: printed and written (here to --out).
    let md = dir.join("timings.md");
    let out = stdout(
        fv_lab()
            .args(["report", "table", "--kind", "timings", "--out"])
            .arg(&md)
            .arg(fx.join("timings-run")),
    );
    assert_eq!(out, read(fx.join("timings.md")));
    assert_eq!(read(md), out);
    // sol-bench: the table, and results.json where --json says.
    let res = dir.join("results.json");
    let out = stdout(
        fv_lab()
            .args(["report", "table", "--kind", "sol-bench", "--json"])
            .arg(&res)
            .args([
                "artifacts/runpod/sol-bench/2026-10-06/a1-h3-ltx",
                "artifacts/runpod/sol-bench/2026-10-06/a2-wan",
                "artifacts/runpod/sol-bench/2026-10-06/a3-h3-bf16",
            ]),
    );
    assert_eq!(out, read(fx.join("sol-bench--2026-10-06.md")));
    assert_eq!(
        read(res),
        read(fx.join("sol-bench--2026-10-06.results.json"))
    );
    // Options of another kind are refused, not ignored; an unknown tier too.
    for args in [
        vec![
            "report",
            "summary",
            "--json",
            "x.json",
            "artifacts/runpod/trace/b0d0bb8-10251650",
        ],
        vec![
            "report",
            "summary",
            "--kind",
            "hd",
            "--family",
            "hd",
            "artifacts/runpod/hd/2cd1ba0-09290122",
        ],
        vec![
            "report",
            "summary",
            "--family",
            "nosuchtier",
            "artifacts/runpod/trace/b0d0bb8-10251650",
        ],
        vec![
            "report",
            "table",
            "--kind",
            "timings",
            "--json",
            "x.json",
            "artifacts/runpod/hd/2cd1ba0-09290122",
        ],
    ] {
        let (code, _, err) = out_err(fv_lab().args(&args));
        assert_eq!(code, Some(1), "{args:?}: {err}");
    }
    let _ = std::fs::remove_dir_all(&dir);
}

/// `report tiers --runs` with several runs of one tier and commit (ADR-0019
/// §16.7: a gold run split with `--cells`): repeated or comma separated, one
/// table naming both, `--check` finds both again; an empty `--runs`, runs of
/// two tiers, or an arm that passed in both, are refused.
#[test]
fn report_tiers_renders_several_runs_as_one() {
    let fx = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/tiers");
    let run = fx.join("runs/gold/1234567-1024120000-abcd");
    let root = std::env::temp_dir().join(format!("fv-lab-bin-tiers-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    let copy = |arm: &str, to: &str| {
        let dest = root.join(to);
        std::fs::create_dir_all(&dest).unwrap();
        std::fs::copy(run.join("pod.json"), dest.join("pod.json")).unwrap();
        copy_tree(&run.join(arm), &dest.join(arm));
    };
    let (a, b) = (
        "artifacts/runpod/gold/1234567-1026080000-aaaa",
        "artifacts/runpod/gold/1234567-1026200000-bbbb",
    );
    for arm in ["demo-h3-turbo", "demo-h3-max"] {
        copy(arm, a);
    }
    for arm in ["demo-ltx-turbo", "demo-ltx-max", "extra-arm"] {
        copy(arm, b);
    }
    let tiers = |extra: &[&str]| {
        let mut c = fv_lab();
        c.args(["report", "tiers", "--root"])
            .arg(&root)
            .arg("--gold")
            .arg(fx.join("gold"))
            .args(extra);
        c
    };
    let comma = stdout(&mut tiers(&["--stdout", "--runs", &format!("{a},{b}")]));
    let repeated = stdout(&mut tiers(&["--stdout", "--runs", a, "--runs", b]));
    assert_eq!(comma, repeated);
    assert!(
        comma.contains(&format!("Run set: `{a}`, `{b}`\n")),
        "{comma}"
    );
    assert!(comma.contains("2 runs rendered as one run set"), "{comma}");
    assert!(comma
        .lines()
        .any(|l| l.starts_with("| `demo-ltx-turbo` |") && l.contains("| measured |")));
    // Written, then checked against the runs its header names.
    let out = root.join("tiers.md");
    let out_s = out.to_str().unwrap();
    stdout(&mut tiers(&["--out", out_s, "--runs", a, "--runs", b]));
    assert_eq!(std::fs::read_to_string(&out).unwrap(), comma);
    stdout(&mut tiers(&["--check", "--out", out_s]));
    // `--runs` naming no directory is refused, not rendered as no run set.
    for empty in ["", ",", " , "] {
        let (code, _, err) = out_err(&mut tiers(&["--out", out_s, "--runs", empty]));
        assert_eq!(code, Some(1), "{empty:?}: {err}");
        assert!(err.contains("names no run directory"), "{err}");
    }
    assert_eq!(std::fs::read_to_string(&out).unwrap(), comma);
    // Refused: another tier; an arm that passed in both runs.
    let other = "artifacts/runpod/rtx6000/1234567-1026200000-cccc";
    copy("extra-arm", other);
    let (code, _, err) = out_err(&mut tiers(&["--stdout", "--runs", &format!("{a},{other}")]));
    assert_eq!(code, Some(1), "{err}");
    assert!(err.contains("different tiers"), "{err}");
    copy("demo-h3-turbo", b);
    let (code, _, err) = out_err(&mut tiers(&["--stdout", "--runs", a, "--runs", b]));
    assert_eq!(code, Some(1), "{err}");
    assert!(
        err.contains("arm `demo-h3-turbo` passed in more than one run"),
        "{err}"
    );
    let _ = std::fs::remove_dir_all(&root);
}

fn copy_tree(from: &std::path::Path, to: &std::path::Path) {
    std::fs::create_dir_all(to).unwrap();
    for e in std::fs::read_dir(from).unwrap() {
        let e = e.unwrap();
        if e.path().is_dir() {
            copy_tree(&e.path(), &to.join(e.file_name()));
        } else {
            std::fs::copy(e.path(), to.join(e.file_name())).unwrap();
        }
    }
}

#[test]
fn the_tree_fetch_is_checked_before_any_pod_exists() {
    // A skip pattern fv-lab cannot read, and a tree fetch without --wait
    // (nothing would fetch it), fail before any request.
    for extra in [
        vec!["--wait", "--fetch-tree", "--fetch-skip", "fr[a]mes"],
        vec!["--fetch-tree"],
    ] {
        let (code, _, err) = out_err(
            fv_lab()
                .args(["pod", "up", "--sha", "1234567", "--family", "rtx6000"])
                .args(&extra),
        );
        assert_eq!(code, Some(1), "{extra:?}: {err}");
        assert!(err.contains("fetch"), "{err}");
    }
    // --fetch-skip needs --fetch-tree (clap).
    let (code, _, _) = out_err(fv_lab().args([
        "pod",
        "up",
        "--sha",
        "1234567",
        "--family",
        "rtx6000",
        "--fetch-skip",
        "/frames/",
    ]));
    assert_eq!(code, Some(2));
    // A readable pattern passes to the plan (no --yes: no request).
    stdout(fv_lab().args([
        "pod",
        "up",
        "--sha",
        "1234567",
        "--family",
        "rtx6000",
        "--wait",
        "--fetch-tree",
        "--fetch-skip",
        "/frames/|\\.mp4$",
        "--fetch-max-mb",
        "100",
    ]));
}

/// The oracle tier (ADR-0019 §15) reads upstream references: `pod up` refuses
/// it without their URL or without `--cells` (a reference the upstream pods
/// never make is waited for, FV_ORACLE_WAIT_S, on a GPU pod), and passes the
/// URL to the run; the D1d families' script knobs are refused, not ignored.
#[test]
fn the_oracle_pod_needs_its_references_and_the_d1d_knobs_are_refused() {
    let up = [
        "pod",
        "up",
        "--sha",
        "1234567",
        "--family",
        "oracle",
        "--tag",
        "1234567-10241200",
    ];
    let cells = [
        "--cells",
        "oracle-fasth3-4step-vsa,oracle-fasth3-4step-vsa-diff",
    ];
    let url = [
        "--extra-env",
        "FV_ORACLE_URL=https://abc-8000.proxy.runpod.net/upstream/1234567-10260100",
    ];
    for args in [vec![], cells.to_vec(), url.to_vec()] {
        let (code, _, err) = out_err(fv_lab().args(up).args(&args));
        assert_ne!(code, Some(0), "{args:?}");
        assert!(err.contains("FV_ORACLE_URL=<base URL>"), "{args:?}: {err}");
    }
    let plan = stdout(fv_lab().args(up).args(cells).args(url));
    let v: serde_json::Value = serde_json::from_str(&plan).expect("json plan");
    let cmd = v["requests"][0]["body"]["dockerStartCmd"][2]
        .as_str()
        .unwrap_or("");
    assert!(cmd.contains("env FV_ORACLE_URL=https://abc-8000.proxy.runpod.net/upstream/1234567-10260100 FV_WORK=/workspace"), "{cmd}");
    assert!(cmd.contains("FV_GEN_TIMEOUT_S=5400"), "{cmd}");
    let (code, _, err) = out_err(
        fv_lab()
            .args(up)
            .args(cells)
            .args(url)
            .args(["--extra-env", "FV_ORACLE_TARGETS=fasth3-4step-vsa"]),
    );
    assert_ne!(code, Some(0));
    assert!(err.contains("fv-lab selects by arm"), "{err}");
    let (code, _, err) = out_err(
        fv_lab()
            .args(["run", "det-short", "--dry-run"])
            .env("FV_DET_REPEATS", "5"),
    );
    assert_ne!(code, Some(0));
    assert!(err.contains("FV_DET_REPEATS: no tier reads it"), "{err}");
    // http(s) references only (review of #202): a file:// URL or one that
    // would read as a curl option is refused, by pod up and by run.
    for bad in ["FV_ORACLE_URL=file:///etc", "FV_ORACLE_URL=-o/tmp/x"] {
        let (code, _, err) = out_err(fv_lab().args(up).args(cells).args(["--extra-env", bad]));
        assert_ne!(code, Some(0), "{bad}");
        assert!(err.contains("is not an http(s):// URL"), "{bad}: {err}");
    }
    let (code, _, err) = out_err(
        fv_lab()
            .args(["run", "oracle", "--dry-run"])
            .env("FV_ORACLE_URL", "file:///etc"),
    );
    assert_ne!(code, Some(0));
    assert!(err.contains("is not an http(s):// URL"), "{err}");
}

/// `fv-lab oracle` without `--yes` (ADR-0019 §19): the plan of both sides,
/// no request and no key; its refusals exit non-zero before anything exists.
#[test]
fn oracle_without_yes_is_the_plan_and_needs_no_key() {
    const UP: &str = "1234567890abcdef1234567890abcdef12345678";
    let cells = "oracle-fasth3-4step-vsa,oracle-fasth3-4step-vsa-diff,oracle-fasth3-4step-vsa-f32,oracle-fasth3-4step-vsa-f32-diff,oracle-fasth3-4step-vsa-bf16-vs-f32";
    let out = stdout(fv_lab().args([
        "oracle",
        "--sha",
        "1234567",
        "--up-sha",
        UP,
        "--cells",
        cells,
        "--tag",
        "1234567-10241200",
    ]));
    let v: serde_json::Value = serde_json::from_str(&out).expect("json plan");
    let up = &v["upstream"];
    assert_eq!(up.as_array().map(Vec::len), Some(1));
    assert_eq!(up[0]["stack"], "fastvideo");
    assert_eq!(
        up[0]["launch"]["image"],
        "ghcr.io/zaitrarrio/fastvideo-rs-upstream-fastvideo:latest"
    );
    let body = &up[0]["requests"][0]["body"];
    assert_eq!(body["minRAMPerGPU"], 192);
    assert_eq!(body["networkVolumeId"], "bz0ofkfx45");
    assert_eq!(body["dataCenterIds"], serde_json::json!(["EU-RO-1"]));
    assert_eq!(up[0]["launch"]["guards"]["ready_timeout_s"], 600);
    let cmd = body["dockerStartCmd"][2].as_str().unwrap();
    assert!(cmd.contains("UP_STEPS=\"info:box weights:h3-diffusers oracle:fasth3-4step-vsa\""));
    assert!(cmd.contains(&format!("origin {UP} ")));
    let rt = &v["runtime"];
    assert_eq!(rt["tag"], "1234567-10241200");
    assert_eq!(rt["launch"]["name"], "fv-lab-oracle-1234567-10241200");
    let argv: Vec<&str> = rt["command"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|x| x.as_str())
        .collect();
    assert_eq!(&argv[..3], ["fv-lab", "pod", "up"]);
    assert!(argv.contains(&"--wait") && argv.contains(&"--yes"));
    assert!(argv.iter().any(|a| a.starts_with(
        "FV_ORACLE_URL=https://upstream-pod-8000.proxy.runpod.net/upstream/1234567-"
    )));
    assert_eq!(v["stock_wait_s"], 2100);
    assert_eq!(v["refs_wait_s"], 10500);
    for bad in [
        vec!["--cells", "nope"],
        vec!["--cells", cells, "--extra-env", "FV_ORACLE_URL=https://x"],
        vec!["--cells", cells, "--upstream-image-tag", "main"],
        vec!["--cells", cells, "--wall-cap-s", "20000"],
        vec!["--family", "rtx6000", "--cells", "sol-h3"],
    ] {
        let o = fv_lab()
            .args(["oracle", "--sha", "1234567", "--up-sha", UP])
            .args(&bad)
            .output()
            .unwrap();
        assert!(!o.status.success(), "{bad:?}");
    }
    let nokey = fv_lab()
        .args([
            "oracle", "--sha", "1234567", "--up-sha", UP, "--cells", cells, "--yes",
        ])
        .output()
        .unwrap();
    assert!(!nokey.status.success());
    assert!(String::from_utf8_lossy(&nokey.stderr).contains("RUNPOD_API_KEY"));
    // `check` reads lab/upstream.toml too.
    let out = stdout(fv_lab().arg("check"));
    assert!(out.contains("upstream fastvideo: 5 target(s)"), "{out}");
    assert!(out.contains("upstream sol-ltx25: 4 target(s)"), "{out}");
}

/// The runtime image's smoke step runs `fv-lab check` (and a dry run) in
/// `/opt/fastvideo-rs`, where `docker/gpucheck.Dockerfile` copies only a
/// subset of the repository. The same subset, read from the Dockerfile's
/// `COPY` lines of the stage that ships fv-lab, in a root of its own: what
/// `fv-lab` reads there must be among them (`lab/upstream.toml` was not, at
/// first, ADR-0019 §19).
#[test]
fn the_runtime_images_lab_subset_passes_check() {
    let df = std::fs::read_to_string(repo_root().join("docker/gpucheck.Dockerfile")).unwrap();
    let stages: Vec<&str> = df.split("\nFROM ").collect();
    let stage = stages
        .iter()
        .find(|s| s.contains("COPY --from=lab-build /out/fv-lab"))
        .expect("the stage that ships fv-lab");
    let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("fv-lab-image-root");
    let _ = std::fs::remove_dir_all(&root);
    fn copy(from: &std::path::Path, to: &std::path::Path) {
        if from.is_dir() {
            std::fs::create_dir_all(to).unwrap();
            for e in std::fs::read_dir(from).unwrap().flatten() {
                copy(&e.path(), &to.join(e.file_name()));
            }
        } else {
            std::fs::create_dir_all(to.parent().unwrap()).unwrap();
            std::fs::copy(from, to).unwrap_or_else(|e| panic!("{}: {e}", from.display()));
        }
    }
    let mut lab = 0;
    for line in stage.lines() {
        let w: Vec<&str> = line.split_whitespace().collect();
        if w.len() != 3 || w[0] != "COPY" || w[1].starts_with("--") {
            continue;
        }
        let Some(dst) = w[2].strip_prefix("/opt/fastvideo-rs/") else {
            continue;
        };
        lab += usize::from(w[1].starts_with("lab/"));
        copy(&repo_root().join(w[1]), &root.join(dst));
    }
    assert!(lab >= 3, "lab/tiers.toml, lab/cells, lab/upstream.toml");
    let out = stdout(fv_lab().args(["check", "--root"]).arg(&root));
    assert!(out.contains("rtx6000: 2 cell(s), 7 arm(s)"), "{out}");
    assert!(out.contains("upstream fastvideo: 5 target(s)"), "{out}");
    let dry = stdout(
        fv_lab()
            .args([
                "run",
                "rtx6000",
                "--dry-run",
                "--work",
                "/workspace",
                "--scratch",
                "/fvscratch",
                "--tag",
                "smoke",
                "--root",
            ])
            .arg(&root),
    );
    assert!(dry.contains("dry-run sol-h3: "), "{dry}");
    let _ = std::fs::remove_dir_all(&root);
}
