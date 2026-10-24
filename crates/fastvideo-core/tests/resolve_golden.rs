//! Golden: what every model id the registry knows resolves to (plan B2).
//!
//! `tests/golden/resolve-inputs.txt` is a fixed list of inputs: every Hub id,
//! its lowercase and bare-name forms, every preset (snake and kebab case),
//! every detector token and a few unknown strings. `tests/golden/resolve.tsv`
//! records, for each, the family, the preset, the workloads and the
//! family-specific line, then the resolved Wan sampling settings per preset.
//! The file was taken on the tree before B2 and must not change with it.
//!
//! `FV_GOLDEN_BLESS=1` writes the current output to
//! `$CARGO_TARGET_TMPDIR/b2-goldens/resolve.tsv` instead of comparing.

use std::fmt::Write as _;
use std::path::PathBuf;

use fastvideo_core::family::{ltx2, wan};
use fastvideo_core::{resolve, sampling::pipeline_defaults, sampling_from_definition};

fn golden_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("golden")
}

fn render() -> String {
    let inputs = std::fs::read_to_string(golden_dir().join("resolve-inputs.txt"))
        .unwrap_or_else(|e| panic!("read resolve-inputs.txt: {e}"));
    let mut out = String::from("# input\tfamily\tpreset\tworkloads\tdetail\n");
    let mut wan_presets: Vec<&str> = Vec::new();
    for input in inputs.lines().filter(|l| !l.is_empty()) {
        match resolve(input) {
            Err(e) => {
                let _ = writeln!(out, "{input}\t-\t-\t-\terror: {e}");
            }
            Ok(r) => {
                let workloads: Vec<&str> = r.workloads().iter().map(|w| w.as_str()).collect();
                let detail = if r.family.name() == wan::NAME {
                    let d = wan::definition(r.preset())
                        .unwrap_or_else(|| panic!("no Wan definition for {}", r.preset()));
                    if !wan_presets.contains(&d.preset) {
                        wan_presets.push(d.preset);
                    }
                    format!(
                        "pipeline_config={} sampling={}",
                        d.pipeline_config,
                        d.sampling.as_str()
                    )
                } else {
                    let line = if r.family.name() == ltx2::NAME {
                        ltx2::line(r.preset())
                    } else {
                        None
                    };
                    format!("ltx_line={line:?}")
                };
                let _ = writeln!(
                    out,
                    "{input}\t{}\t{}\t{}\t{detail}",
                    r.family.name(),
                    r.preset(),
                    workloads.join(",")
                );
            }
        }
    }
    out.push_str("\n# wan preset\tsampling\tpipeline\n");
    for preset in wan_presets {
        let def = fastvideo_core::WAN_MODEL_DEFINITIONS
            .iter()
            .find(|d| d.preset == preset)
            .unwrap_or_else(|| panic!("preset {preset} not in the Wan table"));
        let _ = writeln!(
            out,
            "{preset}\t{:?}\t{:?}",
            sampling_from_definition(def),
            pipeline_defaults(def)
        );
    }
    out
}

#[test]
fn resolution_matches_golden() {
    let got = render();
    if std::env::var("FV_GOLDEN_BLESS").ok().as_deref() == Some("1") {
        let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("b2-goldens");
        std::fs::create_dir_all(&dir).unwrap_or_else(|e| panic!("{e}"));
        std::fs::write(dir.join("resolve.tsv"), &got).unwrap_or_else(|e| panic!("{e}"));
        eprintln!("blessed {}", dir.join("resolve.tsv").display());
        return;
    }
    let want = std::fs::read_to_string(golden_dir().join("resolve.tsv"))
        .unwrap_or_else(|e| panic!("read resolve.tsv: {e}"));
    if got != want {
        for (i, (g, w)) in got.lines().zip(want.lines()).enumerate() {
            if g != w {
                panic!("resolve.tsv line {}:\n got: {g}\nwant: {w}", i + 1);
            }
        }
        panic!(
            "resolve.tsv: {} lines, golden {}",
            got.lines().count(),
            want.lines().count()
        );
    }
}
