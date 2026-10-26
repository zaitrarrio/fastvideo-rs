//! `fv-lab report summary` and `report table` against what the scripts they
//! replace print (docs/adr/0019-fv-lab.md §13). The fixtures under
//! `fixtures/report/` are those scripts' output on committed run directories
//! (`scripts/gpu/tests/fv-lab-report-goldens.sh`); every one is compared here,
//! byte for byte.

use std::path::{Path, PathBuf};
use std::process::Command;

use fastvideo_lab::{experiments, summary, table};

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn fixtures() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/report")
}

fn read(p: &Path) -> Vec<u8> {
    std::fs::read(p).unwrap_or_else(|e| panic!("{}: {e}", p.display()))
}

fn text(p: &Path) -> String {
    String::from_utf8(read(p)).unwrap_or_else(|e| panic!("{}: {e}", p.display()))
}

/// The first line where two texts differ, for a readable failure.
fn assert_same(what: &str, ours: &[u8], theirs: &[u8]) {
    if ours == theirs {
        return;
    }
    let o = String::from_utf8_lossy(ours);
    let t = String::from_utf8_lossy(theirs);
    let (ol, tl): (Vec<&str>, Vec<&str>) = (o.split('\n').collect(), t.split('\n').collect());
    for i in 0..ol.len().max(tl.len()) {
        let (a, b) = (ol.get(i), tl.get(i));
        if a != b {
            panic!(
                "{what}: line {} differs\n  fv-lab: {:?}\n  script: {:?}",
                i + 1,
                a,
                b
            );
        }
    }
    panic!("{what}: differs (bytes)");
}

fn scratch(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("fv-lab-report-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

#[test]
fn tier_summaries_are_the_retired_branches_live_log_lines() {
    let dir = fixtures().join("summary");
    let mut n = 0;
    let mut families = std::collections::BTreeSet::new();
    for e in std::fs::read_dir(&dir).unwrap() {
        let p = e.unwrap().path();
        let name = p.file_name().unwrap().to_string_lossy().into_owned();
        let stem = name.strip_suffix(".txt").expect("<family>--<tag>.txt");
        let (family, tag) = stem.split_once("--").expect("<family>--<tag>");
        let run = root().join("artifacts/runpod").join(family).join(tag);
        assert_same(&name, &summary::tier_summary(&run, family), &read(&p));
        families.insert(family.to_owned());
        n += 1;
    }
    assert!(n >= 30, "{n} fixtures");
    // Every family whose branch printed summary lines has a fixture.
    for f in summary::SUMMARY_TIERS {
        assert!(families.contains(f), "no summary fixture for {f}");
    }
    // A tier whose branch printed none prints nothing.
    assert!(summary::tier_summary(
        &root().join("artifacts/runpod/rtx6000/492e28a-10241144"),
        "rtx6000"
    )
    .is_empty());
}

#[test]
fn hd_is_hd_report_without_the_sharpness() {
    let mut n = 0;
    for e in std::fs::read_dir(fixtures()).unwrap() {
        let p = e.unwrap().path();
        let name = p.file_name().unwrap().to_string_lossy().into_owned();
        let Some(tag) = name
            .strip_prefix("hd--")
            .and_then(|s| s.strip_suffix(".txt"))
        else {
            continue;
        };
        let run = root().join("artifacts/runpod/hd").join(tag);
        let ours = summary::hd_summary(&run, None).unwrap();
        assert_same(&name, ours.as_bytes(), &read(&p));
        n += 1;
    }
    assert!(n >= 5);
}

#[test]
fn hd_lists_keyframes_it_cannot_measure_and_creates_the_crop_dir() {
    let d = scratch("hd");
    std::fs::create_dir_all(d.join("max-768p/keyframes/h3-demo")).unwrap();
    std::fs::write(d.join("max-768p/keyframes/h3-demo/0001.png"), b"png").unwrap();
    std::fs::create_dir_all(d.join("max-768p/keyframes/empty")).unwrap();
    let out = summary::hd_summary(&d, Some(&d.join("crops"))).unwrap();
    assert!(out.contains(&format!(
        "{:<28} {:<22} not computed: fv-lab reads no PNG (scripts/gpu/hd_report.py)\n",
        "max-768p", "h3-demo"
    )));
    assert!(!out.contains("empty"));
    assert!(out.ends_with(&format!("## crops -> {}\n", d.join("crops").display())));
    assert!(d.join("crops").is_dir());
    let _ = std::fs::remove_dir_all(&d);
}

#[test]
fn h3_graphs_is_its_summary_script() {
    let run = root().join("artifacts/perf/h3-denoise-graphs/run-2026-10-14");
    let s = experiments::h3_graphs(&run).unwrap();
    assert_same(
        "h3-graphs",
        s.text.as_bytes(),
        &read(&fixtures().join("h3-graphs.txt")),
    );
    // The script's own output, committed beside the run, is the same text.
    assert_eq!(s.text, text(&run.join("summary.txt")));
    assert!(s.json.is_none());
}

#[test]
fn h3_tier_options_is_its_summary_script_with_its_record() {
    let run = root().join("artifacts/perf/h3-nvfp4-tier-options");
    let s = experiments::h3_tier_options(&run).unwrap();
    assert_same(
        "h3-tier-options.md",
        s.text.as_bytes(),
        &read(&fixtures().join("h3-tier-options.md")),
    );
    assert_same(
        "h3-tier-options.json",
        s.json_text().unwrap().as_bytes(),
        &read(&fixtures().join("h3-tier-options.json")),
    );
}

/// An archived experiment in the layout its summary reads (the goldens
/// script's `layout`, the one mapping).
fn layout(what: &str) -> PathBuf {
    let d = scratch(what);
    let st = Command::new("bash")
        .arg(root().join("scripts/gpu/tests/fv-lab-report-goldens.sh"))
        .args(["layout", what])
        .arg(&d)
        .status()
        .unwrap();
    assert!(st.success(), "layout {what}");
    d
}

#[test]
fn h3_trim_nvfp4_is_its_summary_script_with_its_record() {
    let d = layout("trim");
    let s = experiments::h3_trim_nvfp4(&d).unwrap();
    assert_same(
        "h3-trim-nvfp4.md",
        s.text.as_bytes(),
        &read(&fixtures().join("h3-trim-nvfp4.md")),
    );
    assert_same(
        "h3-trim-nvfp4.json",
        s.json_text().unwrap().as_bytes(),
        &read(&fixtures().join("h3-trim-nvfp4.json")),
    );
    let _ = std::fs::remove_dir_all(&d);
}

#[test]
fn fasth3_nvfp4_is_its_summary_script_with_its_record() {
    let d = layout("fasth3");
    let s = experiments::fasth3_nvfp4(&d).unwrap();
    assert_same(
        "fasth3-nvfp4.md",
        s.text.as_bytes(),
        &read(&fixtures().join("fasth3-nvfp4.md")),
    );
    assert_same(
        "fasth3-nvfp4.json",
        s.json_text().unwrap().as_bytes(),
        &read(&fixtures().join("fasth3-nvfp4.json")),
    );
    let _ = std::fs::remove_dir_all(&d);
}

fn subdirs(d: &Path) -> Vec<PathBuf> {
    let mut v: Vec<PathBuf> = std::fs::read_dir(d)
        .unwrap()
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.is_dir())
        .collect();
    v.sort();
    v
}

#[test]
fn sol_bench_is_sol_bench_table_with_its_results_json() {
    for set in ["2026-10-06", "2026-10-06-phaseB", "2026-10-06-phaseB2"] {
        let base = root().join("artifacts/runpod/sol-bench").join(set);
        let t = table::sol_bench(&subdirs(&base)).unwrap();
        assert_same(
            set,
            t.table.as_bytes(),
            &read(&fixtures().join(format!("sol-bench--{set}.md"))),
        );
        assert_same(
            &format!("{set} results.json"),
            t.results_json.as_bytes(),
            &read(&fixtures().join(format!("sol-bench--{set}.results.json"))),
        );
        assert_eq!(t.results_path, base.join("results.json"));
    }
    // Every published row, the LingBot and Cosmos3 timings, -s15, -rs3, a
    // missing argument, a later run of a cell, a trailing slash.
    let c = fixtures().join("sol-bench-cells");
    let t = table::sol_bench(&[
        c.join("a"),
        c.join("missing"),
        PathBuf::from(format!("{}/b/", c.display())),
    ])
    .unwrap();
    assert_same(
        "cells",
        t.table.as_bytes(),
        &read(&fixtures().join("sol-bench--cells.md")),
    );
    assert_same(
        "cells results.json",
        t.results_json.as_bytes(),
        &read(&fixtures().join("sol-bench--cells.results.json")),
    );
    assert!(table::sol_bench(&[c.join("missing")]).is_err());
}

#[test]
fn timings_is_timings_sh() {
    let t = table::timings(&fixtures().join("timings-run")).unwrap();
    assert_same(
        "timings.md",
        t.as_bytes(),
        &read(&fixtures().join("timings.md")),
    );
    let empty = scratch("timings");
    assert!(table::timings(&empty)
        .unwrap_err()
        .starts_with("no stage reports in "));
    let _ = std::fs::remove_dir_all(&empty);
}

/// Every verbatim block of the frozen summary script is held to its source by
/// the goldens script; here: the blocks are there and name a source.
#[test]
fn the_summary_script_marks_its_frozen_lines() {
    let t = text(&fixtures().join("summary-script.sh"));
    let blocks = t
        .lines()
        .filter(|l| l.starts_with("# --- verbatim "))
        .count();
    assert_eq!(blocks, t.lines().filter(|l| *l == "# --- end").count());
    assert_eq!(blocks, 10);
}
