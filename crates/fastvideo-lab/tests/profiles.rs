//! Plan D2 acceptance: the comments of `profiles/**/*.toml` carry no
//! measurement-looking numbers (seconds, ms, dB, GiB, %, speed-ups, LPIPS /
//! PSNR / sharpness / jitter values, prices). The numbers live in
//! `lab/gold/<variant>.toml` and `docs/perf`; a profile comment points there.

use std::path::{Path, PathBuf};

use fastvideo_lab::measurement_claims;

fn profiles_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../profiles")
        .canonicalize()
        .expect("profiles dir")
}

fn toml_files(dir: &Path, out: &mut Vec<PathBuf>) {
    for entry in std::fs::read_dir(dir).unwrap_or_else(|e| panic!("{}: {e}", dir.display())) {
        let p = entry.expect("entry").path();
        if p.is_dir() {
            toml_files(&p, out);
        } else if p.extension().is_some_and(|x| x == "toml") {
            out.push(p);
        }
    }
}

#[test]
fn profile_comments_carry_no_measurements() {
    let mut files = Vec::new();
    toml_files(&profiles_dir(), &mut files);
    files.sort();
    assert!(
        files.len() >= 40,
        "expected the shipped profiles, found {}",
        files.len()
    );
    let mut offending = Vec::new();
    for p in &files {
        let text = std::fs::read_to_string(p).expect("read profile");
        for c in measurement_claims(&text) {
            offending.push(format!(
                "{}:{}: `{}` in `#{}`",
                p.strip_prefix(profiles_dir().parent().unwrap())
                    .unwrap_or(p)
                    .display(),
                c.line,
                c.text,
                c.comment
            ));
        }
    }
    assert!(
        offending.is_empty(),
        "profile comments carry measurements; move them to lab/gold/<variant>.toml or docs/perf and point there:\n{}",
        offending.join("\n")
    );
}
