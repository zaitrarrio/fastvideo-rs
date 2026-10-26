//! `fv-lab report tiers` (docs/adr/0023-lab-report-and-ci-bridge.md §1):
//!
//! - the golden: a small fixture run set (`fixtures/tiers/runs/gold/…`: a
//!   serve-engine arm, a CLI arm at another canvas, a skipped arm, a failed
//!   arm, an arm of no gold variant, and a variant with no arm) and fixture
//!   gold files render to `fixtures/tiers/tiers.md` byte for byte;
//! - the same run set split in two runs of one tier and commit (ADR-0019
//!   §16.7, `--runs a,b`) renders the same rows, each from its own run;
//! - the committed `docs/perf/tiers.md` is what the gold files and the run
//!   set its header names render to (`fv-lab report tiers --check`).

use std::path::{Path, PathBuf};

use fastvideo_lab::report::{first_difference, render, run_set_of, RunSet, Runs, TIERS_MD};
use fastvideo_lab::{load_dir, Gold};

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

const FIXTURE_RUNS: &str = "crates/fastvideo-lab/fixtures/tiers/runs/gold/1234567-1024120000-abcd";

fn fixture_golds() -> Vec<Gold> {
    load_dir(&repo_root().join("crates/fastvideo-lab/fixtures/tiers/gold")).expect("fixture golds")
}

fn assert_same(committed: &str, rendered: &str, what: &str) {
    if let Some((line, a, b)) = first_difference(committed, rendered) {
        panic!("{what} differs at line {line}:\n  committed: {a}\n  rendered:  {b}");
    }
}

#[test]
fn fixture_run_set_renders_its_golden() {
    let set = RunSet::load(&repo_root(), FIXTURE_RUNS).expect("fixture run set");
    assert_eq!(
        (set.tier.as_str(), set.tag.as_str()),
        ("gold", "1234567-1024120000-abcd")
    );
    assert_eq!(set.arms.len(), 5);
    let text = render(&fixture_golds(), Some(&Runs::from(set.clone())));
    let golden =
        std::fs::read_to_string(repo_root().join("crates/fastvideo-lab/fixtures/tiers/tiers.md"))
            .expect("fixtures/tiers/tiers.md");
    assert_same(&golden, &text, "fixtures/tiers/tiers.md");
    // The rendering names its run set, so --check can find it again.
    assert_eq!(run_set_of(&text).unwrap(), vec![FIXTURE_RUNS.to_owned()]);
    // Deterministic.
    assert_eq!(text, render(&fixture_golds(), Some(&Runs::from(set))));
}

#[test]
fn fixture_rows_say_what_was_and_was_not_measured() {
    let set = RunSet::load(&repo_root(), FIXTURE_RUNS).expect("fixture run set");
    let text = render(&fixture_golds(), Some(&Runs::from(set)));
    let row = |v: &str| {
        text.lines()
            .find(|l| l.starts_with(&format!("| `{v}` |")))
            .unwrap_or_else(|| panic!("no row {v}"))
            .to_owned()
    };
    let h3 = row("demo-h3-turbo");
    assert!(h3.contains("| measured |"), "{h3}");
    assert!(
        h3.contains("832x480x124, 4 steps | same | 9.88 job"),
        "{h3}"
    );
    // $/clip = the pod's 2.49 $/hr x the job's 9.876 s.
    assert!(
        h3.contains(&format!("{:.5}", 2.49 * 9.876 / 3600.0)),
        "{h3}"
    );
    assert!(
        h3.contains("2026-10-14 `abcdef1`, 10 d before the run set"),
        "{h3}"
    );
    let ltx = row("demo-ltx-turbo");
    assert!(ltx.contains("differs (gold 1280x768x121)"), "{ltx}");
    assert!(row("demo-h3-max").contains("not measured (skipped: weights incomplete)"));
    assert!(row("demo-ltx-max").contains("not measured (failed: exit 101)"));
    assert!(row("demo-wan-stream").contains("not measured (no arm in the run set)"));
    assert!(text.contains("not shown): `extra-arm`."));
}

#[test]
fn a_same_named_arm_of_another_tier_is_not_the_variant() {
    // The same run set as another tier's: only the engine arm whose report
    // ran the variant (`context.model.id`) still counts; the CLI arm does not.
    let mut set = RunSet::load(&repo_root(), FIXTURE_RUNS).expect("fixture run set");
    set.tier = "rtx6000".into();
    let text = render(&fixture_golds(), Some(&Runs::from(set)));
    let row = |v: &str| {
        text.lines()
            .find(|l| l.starts_with(&format!("| `{v}` |")))
            .unwrap_or_else(|| panic!("no row {v}"))
            .to_owned()
    };
    assert!(row("demo-h3-turbo").contains("| measured |"));
    assert!(row("demo-ltx-turbo")
        .contains("not measured (arm `demo-ltx-turbo` of tier `rtx6000` is not its gold cell)"));
    // The engine's decode is reported through the MP4: video_decode + encode.
    assert!(row("demo-h3-turbo").contains("| 6.50 | 3.01 |"));
}

#[test]
fn without_a_run_set_nothing_is_measured() {
    let text = render(&fixture_golds(), None);
    assert!(run_set_of(&text).unwrap().is_empty());
    let rows: Vec<&str> = text.lines().filter(|l| l.starts_with("| `")).collect();
    assert_eq!(rows.len(), 5);
    assert!(rows.iter().all(|r| r.contains("not measured (no run set)")));
}

#[test]
fn committed_tiers_md_is_current() {
    let root = repo_root();
    let committed = std::fs::read_to_string(root.join(TIERS_MD)).expect("docs/perf/tiers.md");
    let labels = run_set_of(&committed).expect("a Run set line");
    let set = (!labels.is_empty()).then(|| {
        Runs::new(
            labels
                .iter()
                .map(|l| RunSet::load(&root, l).expect("a run the table names"))
                .collect(),
        )
        .expect("the runs the table names are one run set")
    });
    let golds = load_dir(&root.join("lab/gold")).expect("gold files");
    assert_same(
        &committed,
        &render(&golds, set.as_ref()),
        "docs/perf/tiers.md (fv-lab report tiers)",
    );
}

/// Copies the fixture run's arms `arms` into `<root>/<label>`, with its
/// `pod.json` at `usd_per_hr` (the fixture's is 2.49).
fn split_run(root: &Path, label: &str, arms: &[&str], usd_per_hr: f64) {
    let from = repo_root().join(FIXTURE_RUNS);
    let to = root.join(label);
    for arm in arms {
        copy_dir(&from.join(arm), &to.join(arm));
    }
    let pod = std::fs::read_to_string(from.join("pod.json")).expect("pod.json");
    let pod = pod.replace(
        "\"usd_per_hr\": 2.49",
        &format!("\"usd_per_hr\": {usd_per_hr}"),
    );
    std::fs::write(to.join("pod.json"), pod).expect("write pod.json");
}

fn copy_dir(from: &Path, to: &Path) {
    std::fs::create_dir_all(to).expect("mkdir");
    for e in std::fs::read_dir(from).expect("read_dir") {
        let e = e.expect("entry");
        let p = e.path();
        if p.is_dir() {
            copy_dir(&p, &to.join(e.file_name()));
        } else {
            std::fs::copy(&p, to.join(e.file_name())).expect("copy");
        }
    }
}

const SPLIT_A: &str = "artifacts/runpod/gold/1234567-1026080000-aaaa";
const SPLIT_B: &str = "artifacts/runpod/gold/1234567-1026200000-bbbb";

#[test]
fn a_split_run_set_renders_as_one() {
    let root = std::env::temp_dir().join(format!("fv-lab-tiers-split-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    // The fixture's arms in two runs at the same price: the rows are the
    // single run's, byte for byte; `demo-h3-max` is skipped in both.
    split_run(&root, SPLIT_A, &["demo-h3-turbo", "demo-h3-max"], 2.49);
    split_run(
        &root,
        SPLIT_B,
        &["demo-ltx-turbo", "demo-ltx-max", "demo-h3-max", "extra-arm"],
        2.49,
    );
    let load = |l: &str| RunSet::load(&root, l).expect("split run");
    let single = render(
        &fixture_golds(),
        Some(&Runs::from(
            RunSet::load(&repo_root(), FIXTURE_RUNS).expect("fixture run set"),
        )),
    );
    let split = Runs::new(vec![load(SPLIT_A), load(SPLIT_B)]).expect("one run set");
    let text = render(&fixture_golds(), Some(&split));
    let tail = |t: &str| -> Vec<String> {
        t.lines()
            .skip_while(|l| !l.starts_with("## "))
            .map(str::to_owned)
            .collect()
    };
    assert_eq!(tail(&text), tail(&single));
    assert_eq!(
        run_set_of(&text).unwrap(),
        vec![SPLIT_A.to_owned(), SPLIT_B.to_owned()]
    );
    assert!(text.contains(&format!(
        "- `{SPLIT_A}`: tag `1234567-1026080000-aaaa`, 2 arms, ended 2026-10-24; pod GPU NVIDIA RTX PRO 6000 Blackwell Server Edition, $2.49/hr.\n"
    )), "{text}");

    // Each row reads its own run's pod: the second run at another price.
    let _ = std::fs::remove_dir_all(root.join(SPLIT_B));
    split_run(
        &root,
        SPLIT_B,
        &["demo-ltx-turbo", "demo-ltx-max", "demo-h3-max", "extra-arm"],
        3.6,
    );
    let split = Runs::new(vec![load(SPLIT_A), load(SPLIT_B)]).expect("one run set");
    let text = render(&fixture_golds(), Some(&split));
    let row = |v: &str| {
        text.lines()
            .find(|l| l.starts_with(&format!("| `{v}` |")))
            .unwrap_or_else(|| panic!("no row {v}"))
            .to_owned()
    };
    let h3 = row("demo-h3-turbo");
    assert!(
        h3.contains(&format!("{:.5}", 2.49 * 9.876 / 3600.0)),
        "{h3}"
    );
    let ltx = row("demo-ltx-turbo");
    let ltx_single = single
        .lines()
        .find(|l| l.starts_with("| `demo-ltx-turbo` |"))
        .expect("single ltx row");
    assert_ne!(ltx, ltx_single, "the second run's price is its own");
    assert!(text.contains("$3.60/hr.\n"), "{text}");

    // An arm that ran in both runs is refused.
    copy_dir(
        &repo_root().join(FIXTURE_RUNS).join("demo-h3-turbo"),
        &root.join(SPLIT_B).join("demo-h3-turbo"),
    );
    let e = Runs::new(vec![load(SPLIT_A), load(SPLIT_B)]).expect_err("refused");
    assert!(
        e.contains("arm `demo-h3-turbo` ran in more than one run"),
        "{e}"
    );
    let _ = std::fs::remove_dir_all(&root);
}
