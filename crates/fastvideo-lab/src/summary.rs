//! `fv-lab report summary <run dir>`: what a run's summary scripts printed,
//! read from the files the run directory holds (docs/adr/0019-fv-lab.md §13).
//!
//! - [`tier_summary`] (`--kind tier`, the default): the lines the
//!   `runpod-matrix.sh` branches of the retired families copied into
//!   `live.log` after their arms (§9.4, §10 item 4: b6's load-stage lines,
//!   the techniques / b6 / b4a compare fields and arm timings, h3arms' reuse
//!   counts and gate verdicts, trace's `/gpu_trace` line, ltxfps' PASS / FAIL
//!   lines, ltxvae's decode lines). Each was a `grep` of a file both runners
//!   write and both fetches take (`stderr.log`, `compare/*.json`,
//!   `gate/*.json`); this is that `grep`, on bytes, in the script's order.
//!   The lines carry no `[HH:MM:SS] [family] ` prefix (the script's `log`
//!   added one to some of them).
//! - [`hd_summary`] (`--kind hd`): `scripts/gpu/hd_report.py` without its
//!   keyframe sharpness (numpy and Pillow over PNGs; fv-lab reads no PNG).
//!
//! The experiment summaries (`--kind h3-graphs|h3-tier-options|h3-trim-nvfp4|
//! fasth3-nvfp4`) are [`crate::experiments`].

use std::path::{Path, PathBuf};

use crate::pyfmt::{self, dumps, head_chars, ljust, tuple_repr, Py};

/// The tiers whose script branch printed summary lines, and the families the
/// lines came from (a restored tier prints its mode's family's lines).
pub const SUMMARY_TIERS: [&str; 8] = [
    "techniques",
    "b6",
    "b4a",
    "h3arms",
    "trace",
    "ltxfps",
    "ltxvae",
    "ltxvae-gen",
];

/// `b6_cells`' default (runpod-matrix.sh at 5987e6a4, the `b6)` branch).
const B6_CELLS: [&str; 8] = [
    "b6-solh3-tree",
    "b6-solh3-merge",
    "b6-4stepvsa-merge",
    "b6-h3-streamed",
    "b6-h3-ref2va",
    "b6-ltx25-ref2v",
    "b6-ltx25-a2v-guided",
    "b6-ltx25-offload-cpu",
];

/// b6's load-stage patterns (`grep -h 'h3 dit\|…'`).
const B6_STAGE: [&str; 12] = [
    "h3 dit",
    "h3 text encoder: auto",
    "h3 dit offload",
    "h3 i2v encoder",
    "h3 sol-h3",
    "h3 fasth3-preview",
    "h3 lora merge",
    "ltx2 dit offload",
    "ltx2 offload",
    "ltx2 ic-lora",
    "ltx2 lora",
    "load/io",
];

/// The `trace)` branch's cells, in its loop's order.
const TRACE_CELLS: [&str; 5] = [
    "fasth3-4step-vsa-768p",
    "fasth3-8step-768p",
    "fasth3-4step-dense-768p",
    "ltx25-4k5s-sol",
    "ltx25-1080p20s-sol",
];

/// The lines `grep` reads from a file: split at `\n`, a last line without one
/// still a line; a missing file has none (`2>/dev/null`).
fn lines_of(path: &Path) -> Vec<Vec<u8>> {
    let Ok(b) = std::fs::read(path) else {
        return Vec::new();
    };
    let mut out: Vec<Vec<u8>> = b.split(|c| *c == b'\n').map(<[u8]>::to_vec).collect();
    if out.last().is_some_and(Vec::is_empty) {
        out.pop();
    }
    out
}

fn contains(hay: &[u8], needle: &[u8]) -> bool {
    needle.is_empty() || hay.windows(needle.len()).any(|w| w == needle)
}

fn any_of(line: &[u8], pats: &[&str]) -> bool {
    pats.iter().any(|p| contains(line, p.as_bytes()))
}

/// `cut -c1-<n>` (GNU cut: bytes).
fn cut(line: &[u8], n: usize) -> &[u8] {
    &line[..line.len().min(n)]
}

fn prefixed(out: &mut Vec<u8>, prefix: &str, line: &[u8]) {
    out.extend_from_slice(prefix.as_bytes());
    out.extend_from_slice(line);
    out.push(b'\n');
}

/// The names of a directory's entries, in byte order (a C-locale glob or
/// `ls`); empty when it cannot be read.
fn entries(dir: &Path) -> Vec<String> {
    let mut v: Vec<String> = std::fs::read_dir(dir)
        .map(|rd| {
            rd.flatten()
                .map(|e| e.file_name().to_string_lossy().into_owned())
                .collect()
        })
        .unwrap_or_default();
    v.sort();
    v
}

/// `grep -oE '"(status|off_identity|max_abs_diff_uint8)": *[^,}]*'` over a
/// file: each match, in order.
fn compare_field_matches(path: &Path) -> Vec<Vec<u8>> {
    const KEYS: [&str; 3] = [
        "\"status\":",
        "\"off_identity\":",
        "\"max_abs_diff_uint8\":",
    ];
    let mut out = Vec::new();
    for line in lines_of(path) {
        let mut i = 0;
        while i < line.len() {
            let Some(k) = KEYS.iter().find(|k| line[i..].starts_with(k.as_bytes())) else {
                i += 1;
                continue;
            };
            let mut j = i + k.len();
            while line.get(j) == Some(&b' ') {
                j += 1;
            }
            while j < line.len() && line[j] != b',' && line[j] != b'}' {
                j += 1;
            }
            out.push(line[i..j].to_vec());
            i = j.max(i + 1);
        }
    }
    out
}

/// `for f in "$RUNS"/compare/compare-clips-<prefix>*.json; do [[ -f "$f" ]] &&
/// log "$(basename "$f"): $(grep -oE … "$f" | tr '\n' ' ')"; done`.
fn compare_fields(run: &Path, prefix: &str, out: &mut Vec<u8>) {
    let dir = run.join("compare");
    for name in entries(&dir) {
        let p = dir.join(&name);
        if !(name.starts_with(prefix) && name.ends_with(".json") && p.is_file()) {
            continue;
        }
        out.extend_from_slice(name.as_bytes());
        out.extend_from_slice(b": ");
        for m in compare_field_matches(&p) {
            out.extend_from_slice(&m);
            out.push(b' ');
        }
        out.push(b'\n');
    }
}

/// `grep -ho '"denoise_s":[0-9.]*\|"total_s":[0-9.]*' stderr.log | tail -2 |
/// tr '\n' ' '`.
fn arm_times(path: &Path) -> Vec<u8> {
    const KEYS: [&str; 2] = ["\"denoise_s\":", "\"total_s\":"];
    let mut found: Vec<Vec<u8>> = Vec::new();
    for line in lines_of(path) {
        let mut i = 0;
        while i < line.len() {
            let Some(k) = KEYS.iter().find(|k| line[i..].starts_with(k.as_bytes())) else {
                i += 1;
                continue;
            };
            let mut j = i + k.len();
            while j < line.len() && (line[j].is_ascii_digit() || line[j] == b'.') {
                j += 1;
            }
            found.push(line[i..j].to_vec());
            i = j;
        }
    }
    let mut out = Vec::new();
    for m in &found[found.len().saturating_sub(2)..] {
        out.extend_from_slice(m);
        out.push(b' ');
    }
    out
}

/// The summary lines of a run of `tier`, from its files.
pub fn tier_summary(run: &Path, tier: &str) -> Vec<u8> {
    let mut out = Vec::new();
    match tier {
        "techniques" => {
            compare_fields(run, "compare-clips-", &mut out);
            // `for c in "$RUNS"/*-base "$RUNS"/*-env "$RUNS"/*-prof`: three globs.
            let names = entries(run);
            for suffix in ["-base", "-env", "-prof"] {
                for name in names.iter().filter(|n| n.ends_with(suffix)) {
                    let d = run.join(name);
                    if !d.is_dir() {
                        continue;
                    }
                    out.extend_from_slice(name.as_bytes());
                    out.push(b' ');
                    out.extend_from_slice(&arm_times(&d.join("stderr.log")));
                    out.push(b'\n');
                }
            }
        }
        "b6" => {
            for cell in B6_CELLS {
                for arm in ["base", "pr"] {
                    let a = format!("{cell}-{arm}");
                    for line in lines_of(&run.join(&a).join("stderr.log")) {
                        if any_of(&line, &B6_STAGE) {
                            prefixed(&mut out, &format!("[{a}] "), &line);
                        }
                    }
                }
            }
            compare_fields(run, "compare-clips-b6-", &mut out);
        }
        "b4a" => compare_fields(run, "compare-clips-b4a-", &mut out),
        "h3arms" => {
            for res in ["768p", "480p"] {
                for arm in ["sol", "tea", "soltea"] {
                    let a = format!("fasth3-8step-{res}-{arm}");
                    let n = lines_of(&run.join(&a).join("stderr.log"))
                        .iter()
                        .filter(|l| contains(l, b"h3 teacache step") && contains(l, b"REUSE"))
                        .count();
                    out.extend_from_slice(
                        format!("[{a}] teacache reused steps (all passes): {n}\n").as_bytes(),
                    );
                }
            }
            let dir = run.join("gate");
            for name in entries(&dir) {
                let p = dir.join(&name);
                if !(name.starts_with("gate-") && name.ends_with(".json") && p.is_file()) {
                    continue;
                }
                out.extend_from_slice(format!("{name}: ").as_bytes());
                if let Some(m) = first_verdict(&p) {
                    out.extend_from_slice(&m);
                }
                out.push(b'\n');
            }
        }
        "trace" => {
            for cell in TRACE_CELLS {
                let last = lines_of(&run.join(cell).join("stderr.log"))
                    .into_iter()
                    .rfind(|l| contains(l, b"/gpu_trace "));
                if let Some(l) = last {
                    prefixed(&mut out, &format!("[{cell}] "), cut(&l, 400));
                }
            }
        }
        "ltxfps" => {
            for cell in entries(run)
                .into_iter()
                .filter(|n| n.starts_with("ltx25-1080p") || n.starts_with("fasth3-8step-sink"))
            {
                for line in lines_of(&run.join(&cell).join("stderr.log")) {
                    if any_of(&line, &["PASS", "FAIL"])
                        && any_of(&line, &["sink", "mp4_frames", "no_wav", "gen.frames"])
                    {
                        prefixed(&mut out, &format!("[{cell}] "), cut(&line, 300));
                    }
                }
            }
        }
        "ltxvae" | "ltxvae-gen" => {
            for cell in entries(run)
                .into_iter()
                .filter(|n| n.starts_with("ltxvae") || ltx25_gen(n))
            {
                let hits: Vec<Vec<u8>> = lines_of(&run.join(&cell).join("stderr.log"))
                    .into_iter()
                    .filter(|l| any_of(l, &["vae-bench", "vae_check", "video decode"]))
                    .collect();
                for line in &hits[hits.len().saturating_sub(3)..] {
                    prefixed(&mut out, &format!("[{cell}] "), cut(line, 400));
                }
            }
        }
        _ => {}
    }
    out
}

/// `grep -o '"verdict": "[a-z]*"' | head -1`.
fn first_verdict(path: &Path) -> Option<Vec<u8>> {
    const K: &[u8] = b"\"verdict\": \"";
    for line in lines_of(path) {
        let mut i = 0;
        while i + K.len() <= line.len() {
            if line[i..].starts_with(K) {
                let mut j = i + K.len();
                while j < line.len() && line[j].is_ascii_lowercase() {
                    j += 1;
                }
                if line.get(j) == Some(&b'"') {
                    return Some(line[i..=j].to_vec());
                }
            }
            i += 1;
        }
    }
    None
}

/// ERE `^ltx25-.*-gen`: starts with `ltx25-`, and `-gen` somewhere after.
fn ltx25_gen(name: &str) -> bool {
    name.strip_prefix("ltx25-")
        .is_some_and(|rest| rest.contains("-gen"))
}

// ------------------------------------------------------------------ hd

/// `hd_report.py`'s cells, in its order.
const HD_CELLS: [&str; 8] = [
    "turbo-768p",
    "turbo-1080p",
    "turbo-768p-v",
    "turbo-1080p-v",
    "max-768p",
    "max-1080p",
    "turbo-1080p-10s",
    "max-1080p-10s",
];

/// The `(codec_type, duration, nb_frames)` of each stream in ffprobe's JSON
/// (`-show_entries stream=… -of json`), as `hd_report.py` prints them.
pub fn hd_streams(ffprobe_json: &str) -> Result<String, String> {
    let text = if ffprobe_json.is_empty() {
        "{}"
    } else {
        ffprobe_json
    };
    let v = pyfmt::loads(text)?;
    // `json.loads(out or "{}").get("streams", [])`: a non-object raises, and
    // so does a `streams` that is not a list of objects.
    let Py::Dict(_) = &v else {
        return Err("ffprobe: the output is not a JSON object".into());
    };
    let streams = match v.get("streams") {
        None => &[][..],
        Some(Py::List(items)) => items.as_slice(),
        Some(other) => return Err(format!("ffprobe: streams is {}", pyfmt::repr(other))),
    };
    let rows: Vec<String> = streams
        .iter()
        .map(|s| match s {
            Py::Dict(_) => Ok(tuple_repr(&[
                s.at("codec_type").clone(),
                s.at("duration").clone(),
                s.at("nb_frames").clone(),
            ])),
            other => Err(format!("ffprobe: a stream is {}", pyfmt::repr(other))),
        })
        .collect::<Result<_, String>>()?;
    Ok(format!("[{}]", rows.join(", ")))
}

/// Files matching `<dir>/*` (not hidden) that are directories or files, as
/// Python's `glob` lists them.
fn glob_names(dir: &Path) -> Vec<String> {
    entries(dir)
        .into_iter()
        .filter(|n| !n.starts_with('.'))
        .collect()
}

/// `hd_report.py <run>` without the keyframe sharpness: each keyframe set is
/// listed `not computed` (docs/adr/0019-fv-lab.md §13.4). `ffprobe` (the
/// program to run, `ffprobe` on `PATH` from the CLI) is a process, as in the
/// script. The script's `--jpg-out DIR` made the directory and printed a
/// `## crops` header and wrote no crop; fv-lab has no such option.
pub fn hd_summary(run: &Path, ffprobe: &std::ffi::OsStr) -> Result<String, String> {
    let mut out = String::from("## cells\n");
    for cell in HD_CELLS {
        let p = run.join(cell).join("benchmark.json");
        if !p.exists() {
            out.push_str(&format!("{cell} missing\n"));
            continue;
        }
        let b = pyfmt::load(&p)?;
        out.push_str(&format!("{cell} {}\n\n", head_chars(&dumps(&b), 3000)));
    }
    out.push_str("## compare reports\n");
    let cdir = run.join("compare");
    for name in glob_names(&cdir) {
        if !(name.starts_with("compare-clips-") && name.ends_with(".json")) {
            continue;
        }
        let d = pyfmt::load(&cdir.join(&name))?;
        out.push_str(&format!("{name} {}\n\n", head_chars(&dumps(&d), 1500)));
    }
    out.push_str("## keyframe sharpness (grayscale, at 1920x1088 / 1088x1920)\n");
    for cell in entries(run) {
        let kd = run.join(&cell).join("keyframes");
        if !kd.is_dir() {
            continue;
        }
        for p in entries(&kd) {
            let has_png = glob_names(&kd.join(&p)).iter().any(|f| f.ends_with(".png"));
            if has_png {
                out.push_str(&format!(
                    "{} {} not computed: fv-lab reads no PNG (scripts/gpu/hd_report.py)\n",
                    ljust(&cell, 28),
                    ljust(&p, 22)
                ));
            }
        }
    }
    out.push_str("## streams\n");
    let mut mp4s: Vec<PathBuf> = Vec::new();
    for cell in glob_names(run) {
        let frames = run.join(&cell).join("frames");
        if !frames.is_dir() {
            continue;
        }
        for sub in glob_names(&frames) {
            let d = frames.join(&sub);
            if !d.is_dir() {
                continue;
            }
            for f in glob_names(&d) {
                if f.ends_with(".mp4") {
                    mp4s.push(PathBuf::from(&cell).join("frames").join(&sub).join(f));
                }
            }
        }
    }
    mp4s.sort_by(|a, b| a.to_string_lossy().cmp(&b.to_string_lossy()));
    for rel in mp4s {
        let o = std::process::Command::new(ffprobe)
            .args([
                "-v",
                "error",
                "-show_entries",
                "stream=codec_type,duration,nb_frames,width,height",
                "-of",
                "json",
            ])
            .arg(run.join(&rel))
            .output()
            .map_err(|e| format!("ffprobe: {e}"))?;
        let streams = hd_streams(&String::from_utf8_lossy(&o.stdout))?;
        out.push_str(&format!("{} {streams}\n", rel.display()));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compare_fields_match_the_scripts_grep() {
        let d = std::env::temp_dir().join(format!("fv-lab-summary-{}", std::process::id()));
        std::fs::create_dir_all(d.join("compare")).unwrap();
        std::fs::write(
            d.join("compare/compare-clips-a--b.json"),
            "{\"status\":\"skipped\",\"reason\":\"x\"}\n",
        )
        .unwrap();
        std::fs::write(
            d.join("compare/compare-clips-c--d.json"),
            "{\n  \"status\": \"ok\",\n  \"off_identity\": {\"max_abs_diff_uint8\": 0}\n}",
        )
        .unwrap();
        let mut out = Vec::new();
        compare_fields(&d, "compare-clips-", &mut out);
        assert_eq!(
            String::from_utf8(out).unwrap(),
            "compare-clips-a--b.json: \"status\":\"skipped\" \n\
             compare-clips-c--d.json: \"status\": \"ok\" \"off_identity\": {\"max_abs_diff_uint8\": 0 \n"
        );
        std::fs::remove_dir_all(&d).unwrap();
    }

    #[test]
    fn hd_streams_print_as_python_tuples() {
        let j = r#"{"streams":[{"codec_type":"video","width":1920,"duration":"5.041667","nb_frames":"121"},{"codec_type":"audio","duration":"5.04"}]}"#;
        assert_eq!(
            hd_streams(j).unwrap(),
            "[('video', '5.041667', '121'), ('audio', '5.04', None)]"
        );
        assert_eq!(hd_streams("").unwrap(), "[]");
        for bad in [
            "[]",
            r#"{"streams": {"a": 1}}"#,
            r#"{"streams": [1]}"#,
            r#"{"streams": null}"#,
            "{",
        ] {
            assert!(hd_streams(bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn ltx25_gen_is_the_scripts_ere() {
        assert!(ltx25_gen("ltx25-4k5s-gen"));
        assert!(ltx25_gen("ltx25-4k5s-gen-streaming"));
        assert!(!ltx25_gen("ltx25-gen"));
        assert!(!ltx25_gen("ltxvae-4k5s-fast"));
    }
}
