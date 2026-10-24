//! Goldens of the `fastvideo` CLI (plan B2): the `--help` text of every
//! command, `list-models`, what `generate` and `schedule` do for every model
//! id and preset on a CPU build, and tiny CPU generations hashed frame by
//! frame. The files under `tests/golden/` were taken on the tree before B2;
//! the refactor must leave them byte-identical.
//!
//! `FV_GOLDEN_BLESS=1` writes the current output to
//! `$CARGO_TARGET_TMPDIR/b2-goldens/` instead of comparing.

use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::process::Command;

use sha2::{Digest, Sha256};

const BIN: &str = env!("CARGO_BIN_EXE_fastvideo");

fn golden_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("golden")
}

fn scratch(name: &str) -> PathBuf {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join("cli-golden")
        .join(name);
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap_or_else(|e| panic!("mkdir {}: {e}", dir.display()));
    dir
}

struct Run {
    code: i32,
    stdout: String,
    stderr: String,
}

/// The binary in a clean environment: no `FASTVIDEO_*`, an empty `HOME` (no
/// Hugging Face cache), no terminal width.
fn run(args: &[&str], home: &Path) -> Run {
    let out = Command::new(BIN)
        .args(args)
        .env_clear()
        .env("PATH", std::env::var_os("PATH").unwrap_or_default())
        .env("HOME", home)
        .output()
        .unwrap_or_else(|e| panic!("spawn {BIN}: {e}"));
    Run {
        code: out.status.code().unwrap_or(-1),
        stdout: String::from_utf8_lossy(&out.stdout).into_owned(),
        stderr: String::from_utf8_lossy(&out.stderr).into_owned(),
    }
}

fn last_line(s: &str) -> &str {
    s.lines().rev().find(|l| !l.trim().is_empty()).unwrap_or("")
}

fn compare(name: &str, got: &str) {
    if std::env::var("FV_GOLDEN_BLESS").ok().as_deref() == Some("1") {
        let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("b2-goldens");
        std::fs::create_dir_all(&dir).unwrap_or_else(|e| panic!("{e}"));
        std::fs::write(dir.join(name), got).unwrap_or_else(|e| panic!("{e}"));
        eprintln!("blessed {}", dir.join(name).display());
        return;
    }
    let want = std::fs::read_to_string(golden_dir().join(name))
        .unwrap_or_else(|e| panic!("read golden {name}: {e}"));
    if got != want {
        for (i, (g, w)) in got.lines().zip(want.lines()).enumerate() {
            if g != w {
                panic!("{name} line {}:\n got: {g}\nwant: {w}", i + 1);
            }
        }
        panic!(
            "{name}: {} lines, golden {}",
            got.lines().count(),
            want.lines().count()
        );
    }
}

/// Every Hub id `list-models` prints, in order, with its preset.
fn registered(home: &Path) -> Vec<(String, String)> {
    let r = run(&["list-models"], home);
    assert_eq!(r.code, 0, "list-models: {}", r.stderr);
    r.stdout
        .lines()
        .map(|l| {
            let mut cols = l.split('\t');
            let id = cols.next().unwrap_or_default().to_string();
            let preset = cols
                .find_map(|c| c.strip_prefix("preset="))
                .unwrap_or_default()
                .to_string();
            (id, preset)
        })
        .collect()
}

fn section(out: &mut String, title: &str, r: &Run, full_stderr: bool) {
    let _ = writeln!(out, "### {title}\nexit={}", r.code);
    out.push_str("--- stdout\n");
    out.push_str(&r.stdout);
    if r.code != 0 || full_stderr {
        let _ = writeln!(out, "--- stderr (last line)\n{}", last_line(&r.stderr));
    }
    out.push('\n');
}

#[test]
fn help_matches_golden() {
    let home = scratch("help-home");
    let mut out = String::new();
    for args in [
        &["--help"][..],
        &["help"],
        &["list-models", "--help"],
        &["generate", "--help"],
        &["bench", "--help"],
        &["schedule", "--help"],
        &["help", "generate"],
        &[],
        &["generate"],
        &["no-such-command"],
    ] {
        let r = run(args, &home);
        let _ = writeln!(out, "### fastvideo {}\nexit={}", args.join(" "), r.code);
        out.push_str("--- stdout\n");
        out.push_str(&r.stdout);
        out.push_str("--- stderr\n");
        out.push_str(&r.stderr);
        out.push('\n');
    }
    compare("help.txt", &out);
}

#[test]
fn list_models_matches_golden() {
    let home = scratch("list-home");
    let r = run(&["list-models"], &home);
    assert_eq!(r.code, 0);
    // The plan's count: 83 Hub ids (82 distinct; LongCat's id backs two presets).
    assert_eq!(r.stdout.lines().count(), 83, "{}", r.stdout);
    compare("list-models.txt", &r.stdout);
}

/// `generate` without weights and `schedule` for every Hub id and every
/// preset name: the resolution as the CLI reports it, and the error each
/// family gives on a CPU build.
#[test]
fn generate_and_schedule_per_model_match_golden() {
    let home = scratch("models-home");
    let out_dir = scratch("models-out");
    let mut inputs: Vec<String> = Vec::new();
    for (id, preset) in registered(&home) {
        for s in [id, preset.clone(), preset.replace('_', "-")] {
            if !inputs.contains(&s) {
                inputs.push(s);
            }
        }
    }
    let mut out = String::new();
    for model in &inputs {
        let o = out_dir.join("o");
        let o = o.to_string_lossy();
        let r = run(&["generate", "--model", model, "--output", &o], &home);
        section(&mut out, &format!("generate --model {model}"), &r, false);
        let r = run(&["schedule", "--model", model], &home);
        section(&mut out, &format!("schedule --model {model}"), &r, false);
    }
    compare(
        "models.txt",
        &out.replace(&*out_dir.to_string_lossy(), "<OUT>"),
    );
}

/// Tiny CPU generations: one per preset (`--tiny` is the zero-weight graph,
/// Wan only; the other families refuse it), frames hashed.
#[test]
fn tiny_generations_match_golden() {
    let home = scratch("tiny-home");
    let mut seen: Vec<String> = Vec::new();
    let mut out = String::new();
    for (id, preset) in registered(&home) {
        if seen.contains(&preset) {
            continue;
        }
        seen.push(preset.clone());
        let dir = scratch(&format!("tiny-{preset}"));
        let d = dir.to_string_lossy().into_owned();
        let args = [
            "generate", "--model", &id, "--tiny", "--device", "cpu", "--height", "64", "--width",
            "64", "--frames", "5", "--steps", "2", "--seed", "7", "--output", &d,
        ];
        let r = run(&args, &home);
        let mut frames: Vec<PathBuf> = std::fs::read_dir(&dir)
            .map(|rd| {
                rd.flatten()
                    .map(|e| e.path())
                    .filter(|p| p.extension().and_then(|e| e.to_str()) == Some("png"))
                    .collect()
            })
            .unwrap_or_default();
        frames.sort();
        let _ = writeln!(out, "### {preset} ({id})\nexit={}", r.code);
        out.push_str("--- stdout\n");
        out.push_str(&r.stdout.replace(&d, "<OUT>"));
        if r.code != 0 {
            let _ = writeln!(out, "--- stderr (last line)\n{}", last_line(&r.stderr));
        }
        let _ = writeln!(out, "--- frames {}", frames.len());
        for f in &frames {
            let bytes = std::fs::read(f).unwrap_or_else(|e| panic!("read {}: {e}", f.display()));
            let name = f.file_name().map(|n| n.to_string_lossy().into_owned());
            let _ = writeln!(
                out,
                "{}  {}",
                hex(&Sha256::digest(&bytes)),
                name.unwrap_or_default()
            );
        }
        out.push('\n');
    }
    compare("tiny.txt", &out);
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().fold(String::new(), |mut s, b| {
        let _ = write!(s, "{b:02x}");
        s
    })
}
