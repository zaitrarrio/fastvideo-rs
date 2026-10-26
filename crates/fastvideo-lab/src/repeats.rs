//! Frame-hash equality across repeats (docs/adr/0019-fv-lab.md §15): the
//! hashes and the report the `det-short` and `determinism` families of
//! `scripts/gpu/runpod-matrix.sh` write, ported byte for byte, and the typed
//! verdict the script left to a reader of `determinism.txt`.
//!
//! - [`RepeatHash::Arm`] (`det-short`'s `sh_hash`): per arm, every
//!   `frame-*.png` under the arm directory but `*/warmup/*` (`find`, sorted by
//!   path bytes, concatenated) and every `audio.wav` alike;
//!   `<arm>/hashes.json` and a line of `<runs>/hashes.txt`.
//! - [`RepeatHash::Run`] (`determinism`'s `det_hash`): per run directory
//!   `frames/<run>/` but `warmup`, its `frame-*.png` (a glob, name order) and
//!   its `audio.wav`; `<arm>/hashes.txt`, and `<arm>/digests.txt` (the stage
//!   digests of `stderr.log`).
//!
//! Then `<runs>/determinism.txt`: every hash against the reference's,
//! `IDENTICAL` or `DIFFERENT`, in the script's words.

use std::fs;
use std::path::{Path, PathBuf};

use sha2::{Digest as _, Sha256};

use crate::plan::{line_matches, log_lines, RepeatHash, RepeatsCell};

fn hex(d: &[u8]) -> String {
    d.iter().map(|x| format!("{x:02x}")).collect()
}

/// `cat <files> | sha256sum`: the files that read (a directory or a dangling
/// link writes an error to cat's stderr and nothing to the pipe), in order.
fn cat_sha256(files: &[PathBuf]) -> String {
    let mut h = Sha256::new();
    for f in files {
        if let Ok(b) = fs::read(f) {
            h.update(&b);
        }
    }
    hex(&h.finalize())
}

/// `find -name 'frame-*.png'`: the glob on the file name.
fn is_frame_name(name: &str) -> bool {
    name.len() >= "frame-.png".len() && name.starts_with("frame-") && name.ends_with(".png")
}

/// Every entry below `dir` (not `dir` itself), as `find -P` lists them: a
/// symbolic link is listed, never followed.
fn walk(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(rd) = fs::read_dir(dir) else { return };
    for e in rd.flatten() {
        let p = e.path();
        let is_dir = e.file_type().is_ok_and(|t| t.is_dir());
        out.push(p.clone());
        if is_dir {
            walk(&p, out);
        }
    }
}

fn name_of(p: &Path) -> String {
    p.file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default()
}

/// `! -path '*/warmup/*'`.
fn under_warmup(p: &Path) -> bool {
    p.to_string_lossy().contains("/warmup/")
}

/// `sort -z` in the C locale: by the path's bytes.
fn sort_paths(v: &mut [PathBuf]) {
    use std::os::unix::ffi::OsStrExt as _;
    v.sort_by(|a, b| a.as_os_str().as_bytes().cmp(b.as_os_str().as_bytes()));
}

/// `sh_hash`'s numbers for one arm directory: the frame count, the frames'
/// and the audio's SHA-256 (`none` without any).
pub fn arm_hashes(cell: &Path) -> (usize, String, String) {
    let mut all = Vec::new();
    walk(cell, &mut all);
    let mut frames: Vec<PathBuf> = all
        .iter()
        .filter(|p| is_frame_name(&name_of(p)) && !under_warmup(p))
        .cloned()
        .collect();
    let mut audio: Vec<PathBuf> = all
        .iter()
        .filter(|p| name_of(p) == "audio.wav" && !under_warmup(p))
        .cloned()
        .collect();
    sort_paths(&mut frames);
    sort_paths(&mut audio);
    let n = frames.len();
    let fh = if n == 0 {
        "none".to_owned()
    } else {
        cat_sha256(&frames)
    };
    let ah = if audio.is_empty() {
        "none".to_owned()
    } else {
        cat_sha256(&audio)
    };
    (n, fh, ah)
}

/// `<arm>/hashes.json` (with the newline `write_json` adds).
pub fn hashes_json(arm: &str, n: usize, fh: &str, ah: &str) -> String {
    format!(
        "{{\"cell\":\"{arm}\",\"frames\":{n},\"frames_sha256\":\"{fh}\",\"audio_sha256\":\"{ah}\"}}\n"
    )
}

/// One line of `<runs>/hashes.txt`.
pub fn hashes_line(arm: &str, n: usize, fh: &str, ah: &str) -> String {
    format!("{arm} frames={n} frames_sha256={fh} audio_sha256={ah}\n")
}

/// The line of `<runs>/hashes.txt` a `hashes.json` holds (`fv-lab gate`
/// rebuilds the file from what both fetches take).
pub fn line_from_json(text: &str) -> Option<String> {
    let v: serde_json::Value = serde_json::from_str(text).ok()?;
    let s = |k: &str| v.get(k).and_then(|x| x.as_str()).map(str::to_owned);
    let n = v.get("frames")?.as_u64()?;
    Some(hashes_line(
        &s("cell")?,
        usize::try_from(n).ok()?,
        &s("frames_sha256")?,
        &s("audio_sha256")?,
    ))
}

/// `find <dir> -name 'frame-*.png' ! -name <keep>… -delete`: every frame
/// file (not a directory) below `dir` but those named in `keep`.
pub fn prune(dir: &Path, keep: &[String]) {
    let mut all = Vec::new();
    walk(dir, &mut all);
    for p in all {
        let name = name_of(&p);
        let is_dir = fs::symlink_metadata(&p).is_ok_and(|m| m.is_dir());
        if is_frame_name(&name) && !keep.contains(&name) && !is_dir {
            let _ = fs::remove_file(&p);
        }
    }
}

/// The entries of `dir` a glob sees (no dot names), by name bytes.
fn glob_names(dir: &Path) -> Vec<String> {
    let mut v: Vec<String> = fs::read_dir(dir)
        .map(|rd| {
            rd.flatten()
                .filter_map(|e| e.file_name().into_string().ok())
                .filter(|n| !n.starts_with('.'))
                .collect()
        })
        .unwrap_or_default();
    v.sort();
    v
}

/// `det_hash`'s `<arm>/hashes.txt` for one arm directory: per run
/// directory of `frames/` (a `*/` glob: directories and links to them, name
/// order) but `warmup`, `<run> frames <count> <sha256>` and, with an
/// `audio.wav`, `<run> audio <sha256>`; and the run directories, which the
/// caller prunes.
pub fn run_hashes(cell: &Path) -> (String, Vec<PathBuf>) {
    let frames = cell.join("frames");
    let mut out = String::new();
    let mut runs = Vec::new();
    for run in glob_names(&frames) {
        let d = frames.join(&run);
        if !d.is_dir() || run == "warmup" {
            continue;
        }
        let names: Vec<String> = glob_names(&d)
            .into_iter()
            .filter(|n| is_frame_name(n) && !n.contains('\n'))
            .collect();
        let files: Vec<PathBuf> = names.iter().map(|n| d.join(n)).collect();
        out.push_str(&format!(
            "{run} frames {} {}\n",
            names.len(),
            cat_sha256(&files)
        ));
        let wav = d.join("audio.wav");
        if wav.is_file() {
            out.push_str(&format!("{run} audio {}\n", cat_sha256(&[wav])));
        }
        runs.push(d);
    }
    (out, runs)
}

/// `grep -h ' digest ' stderr.log | sed 's/^.*digest //'`: each stage-digest
/// line from its last `digest ` on.
pub fn digests(stderr: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    for l in log_lines(stderr) {
        if !line_matches(l, " digest ") {
            continue;
        }
        let pat = b"digest ";
        let last = l
            .windows(pat.len())
            .rposition(|w| w == pat)
            .map_or(0, |i| i + pat.len());
        out.extend_from_slice(&l[last..]);
        out.push(b'\n');
    }
    out
}

/// A cell's typed verdict: `None` undecided (its reference arm did not run).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CellVerdict {
    pub cell: String,
    pub pass: Option<bool>,
    pub why: String,
}

/// Bash's `$(…)` of several lines: joined with newlines.
fn joined(v: &[String]) -> String {
    v.join("\n")
}

/// `sed 's/.*<key>=\([^ ]*\).*/\1/'`: the value after the last `<key>=`, or
/// the line as it was when it has none.
fn sed_value(line: &str, key: &str) -> String {
    let pat = format!("{key}=");
    match line.rfind(&pat) {
        Some(i) => {
            let rest = &line[i + pat.len()..];
            rest[..rest.find(' ').unwrap_or(rest.len())].to_owned()
        }
        None => line.to_owned(),
    }
}

/// `read -r a b c d`: the first three whitespace fields and the rest.
fn read4(line: &str) -> (String, String, String, String) {
    let mut rest = line.trim_matches([' ', '\t']);
    let mut f = Vec::new();
    for _ in 0..3 {
        let end = rest.find([' ', '\t']).unwrap_or(rest.len());
        f.push(rest[..end].to_owned());
        rest = rest[end..].trim_start_matches([' ', '\t']);
    }
    (
        f[0].clone(),
        f[1].clone(),
        f[2].clone(),
        rest.trim_end_matches([' ', '\t']).to_owned(),
    )
}

/// `det-short`'s report over `<runs>/hashes.txt` (`hashes`) for the cells
/// that ran, and each cell's verdict. `ran(arm)`: the arm recorded an exit.
pub fn report_arm(
    hashes: &str,
    cells: &[&RepeatsCell],
    ran: &dyn Fn(&str) -> bool,
) -> (String, Vec<CellVerdict>) {
    let lines: Vec<&str> = hashes.lines().collect();
    let first = |l: &str| l.split(' ').next().unwrap_or("").to_owned();
    let mut out = String::new();
    let mut verdicts = Vec::new();
    for c in cells {
        let r = &c.repeats;
        let reference = r.reference_arm();
        let ref_lines: Vec<&str> = lines
            .iter()
            .copied()
            .filter(|l| l.starts_with(&format!("{reference} ")))
            .collect();
        let ref_fh = joined(
            &ref_lines
                .iter()
                .map(|l| sed_value(l, "frames_sha256"))
                .collect::<Vec<_>>(),
        );
        let ref_ah = joined(
            &ref_lines
                .iter()
                .map(|l| sed_value(l, "audio_sha256"))
                .collect::<Vec<_>>(),
        );
        let mut why = String::new();
        if ref_fh.is_empty() {
            why.push_str(&format!(" {reference}:missing"));
        } else {
            if ref_fh == "none" {
                why.push_str(&format!(" {reference}:no-frames"));
            }
            let mut seen = Vec::new();
            for l in lines
                .iter()
                .filter(|l| c.arms.contains(&first(l)) && l.contains(' '))
            {
                let (cell, _fr, fh, ah) = read4(l);
                let fh = fh.strip_prefix("frames_sha256=").unwrap_or(&fh).to_owned();
                let ah = ah.strip_prefix("audio_sha256=").unwrap_or(&ah).to_owned();
                let same = fh != "none" && fh == ref_fh && ah == ref_ah;
                let v = if same { "IDENTICAL" } else { "DIFFERENT" };
                out.push_str(&format!("{cell} {v} frames {fh} audio {ah}\n"));
                if r.judged.contains(&cell) && !same && cell != reference {
                    why.push_str(&format!(" {cell}:different"));
                }
                seen.push(cell);
            }
            for a in &r.judged {
                if !seen.contains(a) && a != reference {
                    why.push_str(&format!(" {a}:missing"));
                }
            }
        }
        verdicts.push(CellVerdict {
            cell: c.cell.clone(),
            pass: ran(reference).then_some(why.is_empty()),
            why: why.trim_start().to_owned(),
        });
    }
    (out, verdicts)
}

/// `determinism`'s report over each arm's `hashes.txt` (`read(arm)`) for the
/// cells that ran, and each cell's verdict.
pub fn report_run(
    read: &dyn Fn(&str) -> Option<String>,
    cells: &[&RepeatsCell],
    ran: &dyn Fn(&str) -> bool,
) -> (String, Vec<CellVerdict>) {
    let mut out = String::new();
    let mut verdicts = Vec::new();
    for c in cells {
        let r = &c.repeats;
        let reference = r.reference_arm();
        let run = r.reference_run().unwrap_or("run1");
        let ref_text = read(reference).unwrap_or_default();
        let field = |kind: &str, n: usize| {
            joined(
                &ref_text
                    .lines()
                    .filter(|l| l.starts_with(&format!("{run} {kind}")))
                    .map(|l| l.split_whitespace().nth(n).unwrap_or("").to_owned())
                    .collect::<Vec<_>>(),
            )
        };
        let (want_f, want_a) = (field("frames", 3), field("audio", 2));
        let ref_count = field("frames", 2);
        let mut why = String::new();
        // The script's line for a reference arm that ran (a hashes.txt)
        // without hashes of the reference run (skipped, crashed).
        if read(reference).is_some() && want_f.is_empty() {
            out.push_str(&format!("{reference}: no {run} hashes\n"));
        }
        if want_f.is_empty() || ref_count == "0" {
            why.push_str(&format!(" {reference}/{run}:no-frames"));
        }
        for arm in &c.arms {
            let Some(text) = read(arm) else {
                if r.judged.contains(arm) {
                    why.push_str(&format!(" {arm}:no-runs"));
                }
                continue;
            };
            let mut n = 0;
            for l in text.lines() {
                let (run_l, kind, a, b) = read4(l);
                let (want, mut got) = if kind == "audio" {
                    (&want_a, a.clone())
                } else {
                    (&want_f, b.clone())
                };
                if kind == "frames" && a == "0" {
                    got = "none".to_owned();
                }
                let same = !want.is_empty() && got == *want;
                let v = if same { "IDENTICAL" } else { "DIFFERENT" };
                out.push_str(&format!("{arm}/{run_l} {kind} {v} {got}\n"));
                if r.judged.contains(arm) && !same {
                    why.push_str(&format!(" {arm}/{run_l}:{kind}-different"));
                }
                n += 1;
            }
            if n == 0 && r.judged.contains(arm) {
                why.push_str(&format!(" {arm}:no-runs"));
            }
        }
        verdicts.push(CellVerdict {
            cell: c.cell.clone(),
            pass: ran(reference).then_some(why.is_empty()),
            why: why.trim_start().to_owned(),
        });
    }
    (out, verdicts)
}

/// Which report a [`RepeatHash`] writes besides `determinism.txt`.
pub fn top_files(hash: RepeatHash) -> &'static [&'static str] {
    match hash {
        RepeatHash::Arm => &["hashes.txt", "determinism.txt"],
        RepeatHash::Run => &["determinism.txt"],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn read_and_sed_split_as_bash() {
        assert_eq!(
            read4("run1 frames 145 abc"),
            ("run1".into(), "frames".into(), "145".into(), "abc".into())
        );
        assert_eq!(
            read4("run1 audio xyz"),
            ("run1".into(), "audio".into(), "xyz".into(), String::new())
        );
        assert_eq!(
            sed_value("a frames_sha256=x audio_sha256=y", "frames_sha256"),
            "x"
        );
        assert_eq!(sed_value("nothing here", "frames_sha256"), "nothing here");
        assert_eq!(
            digests(b"x digest a=1\nno\n[t] digest y digest z=2\n"),
            b"a=1\nz=2\n"
        );
        assert!(is_frame_name("frame-.png") && !is_frame_name("frame.png"));
    }
}
