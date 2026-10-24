//! The settings layer's acceptance tests over the workspace sources
//! (`docs/adr/0010-settings-layer.md`):
//!
//! 1. no `std::env::var("FASTVIDEO_...")` / `var_os` outside this module;
//! 2. in the inference crates, no `std::env::var` / `var_os` whose argument
//!    is not a string literal (a `const` name, a lookup closure's parameter),
//!    except the reviewed sites in [`INDIRECT_ALLOWED`], none of which reads
//!    a `FASTVIDEO_*` setting;
//! 3. every `FASTVIDEO_*` name in a string literal is a row of
//!    [`super::SETTINGS`], and every row is named somewhere;
//! 4. a typed getter (`bool_flag("FASTVIDEO_X", default)`, ...) or a reader
//!    call with a literal name agrees with its row: the parse, the literal
//!    default, and the reader (layered getters on `Layered` rows,
//!    `env_var` on `Env` rows);
//! 5. `docs/scope.md`'s generated region is [`super::render_markdown`].

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use super::table::{lookup, DefaultValue, Parse, Reader, SETTINGS};

/// This module: the only place a `FASTVIDEO_*` name may be read directly.
const SETTINGS_DIR: &str = "crates/fastvideo-models/src/techniques/settings/";

/// Crates whose every `std::env::var` argument must be a literal (scan 2).
const INFERENCE_CRATES: &[&str] = &[
    "crates/fastvideo-models/",
    "crates/fastvideo-cudarc/",
    "crates/fastvideo-core/",
    "crates/fastvideo-loader/",
    "crates/fastvideo-gpucheck/",
    "crates/fastvideo-engine-service/",
    "crates/fastvideo-cli/",
    "crates/fastvideo-ops/",
];

/// `(file, argument)`: reviewed non-literal `std::env::var(_os)` sites in the
/// inference crates, with why each is not a settings read.
const INDIRECT_ALLOWED: &[(&str, &str, &str)] = &[
    (
        "crates/fastvideo-engine-service/src/cuda/families.rs",
        "FAMILIES_DIR_ENV",
        "FV_FAMILIES_DIR (an FV_* deployment override of the embedded family files), not a FASTVIDEO_* setting",
    ),
    (
        "crates/fastvideo-engine-service/src/cuda/backend.rs",
        "k",
        "install_process_plan compares the environment with the plan before it sets it",
    ),
    (
        "crates/fastvideo-engine-service/src/cuda/wan.rs",
        "key",
        "EnvGuard saves the variable it sets around a load, to restore it",
    ),
    (
        "crates/fastvideo-engine-service/src/cuda/caps.rs",
        "WARMUP_ENV",
        "FV_WARMUP, not a FASTVIDEO_* name",
    ),
    (
        "crates/fastvideo-engine-service/src/cuda/causal.rs",
        "k",
        "the FV_LONGLIVE_* flags, not FASTVIDEO_* names",
    ),
    (
        "crates/fastvideo-cudarc/tests/determinism.rs",
        "CHILD_ENV",
        "FV_DETERMINISM_CHILD, not a FASTVIDEO_* name",
    ),
    (
        "crates/fastvideo-cudarc/src/wan/transformer.rs",
        "fallback",
        "the legacy WAN22_* alias after the FASTVIDEO_* name",
    ),
    (
        "crates/fastvideo-cudarc/src/wan/sol_cache.rs",
        "fallback",
        "the legacy WAN22_* alias after the FASTVIDEO_* name",
    ),
    (
        "crates/fastvideo-cudarc/src/h3/pipeline.rs",
        "FASTH3_LORA_STRENGTH_ENV",
        "FASTH3_LORA_STRENGTH, not a FASTVIDEO_* name",
    ),
    (
        "crates/fastvideo-cudarc/src/h3/pipeline.rs",
        "TURBO_LORA_STRENGTH_ENV",
        "FV_H3_TURBO_LORA_STRENGTH, not a FASTVIDEO_* name",
    ),
    (
        "crates/fastvideo-cudarc/src/ltx2/text_cache.rs",
        "k",
        "XDG_CACHE_HOME and HOME",
    ),
];

/// `(file, name)`: reviewed `env_var` reads of a `Layered` row. Tests that
/// save and restore a variable around `set_var`, and two readers that have
/// always read the environment only and keep doing so (the routing is
/// byte-identical, it does not promote them to the layered reader).
const ENV_READS_OF_LAYERED: &[(&str, &str, &str)] = &[
    (
        "crates/fastvideo-core/src/generator.rs",
        "FASTVIDEO_VSA",
        "the `fastvideo` CLI's VSA-id check: `1` / `true` from the environment only",
    ),
    (
        "crates/fastvideo-gpucheck/src/wan_stage.rs",
        "FASTVIDEO_WAN_COND_CACHE",
        "benchmark.json records the environment's value",
    ),
    (
        "crates/fastvideo-core/src/lib.rs",
        "FASTVIDEO_VSA",
        "test: save / restore",
    ),
    (
        "crates/fastvideo-core/src/lib.rs",
        "FASTVIDEO_SDPA",
        "test: save / restore",
    ),
    (
        "crates/fastvideo-cudarc/src/rt/resident.rs",
        "FASTVIDEO_RESIDENT",
        "test: save / restore",
    ),
    (
        "crates/fastvideo-cudarc/src/wan/vae.rs",
        "FASTVIDEO_VAE_CHUNK",
        "test: save / restore",
    ),
];

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn rust_files(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    let mut entries: Vec<_> = entries.filter_map(Result::ok).map(|e| e.path()).collect();
    entries.sort();
    for p in entries {
        let name = p.file_name().and_then(|n| n.to_str()).unwrap_or("");
        if p.is_dir() {
            if name != "target" && !name.starts_with('.') {
                rust_files(&p, out);
            }
        } else if name.ends_with(".rs") {
            out.push(p);
        }
    }
}

/// A source file with its comments blanked (newlines kept) and the byte
/// spans of its string literals' contents.
struct Source {
    rel: String,
    code: String,
    strings: Vec<(usize, usize)>,
}

impl Source {
    fn line_of(&self, at: usize) -> usize {
        self.code[..at].bytes().filter(|&b| b == b'\n').count() + 1
    }
}

/// Blank `//` and `/* */` comments; record string literal contents. Handles
/// escapes, raw strings (`r"..."`, `r#"..."#`), byte strings and char
/// literals versus lifetimes, which is all the workspace uses.
fn lex(text: &str) -> (String, Vec<(usize, usize)>) {
    let b = text.as_bytes();
    let mut out = b.to_vec();
    let mut strings = Vec::new();
    let mut i = 0;
    let blank = |out: &mut Vec<u8>, from: usize, to: usize| {
        for c in &mut out[from..to] {
            if *c != b'\n' {
                *c = b' ';
            }
        }
    };
    while i < b.len() {
        match b[i] {
            b'/' if b.get(i + 1) == Some(&b'/') => {
                let end = b[i..]
                    .iter()
                    .position(|&c| c == b'\n')
                    .map_or(b.len(), |n| i + n);
                blank(&mut out, i, end);
                i = end;
            }
            b'/' if b.get(i + 1) == Some(&b'*') => {
                let mut depth = 0usize;
                let start = i;
                while i < b.len() {
                    if b[i] == b'/' && b.get(i + 1) == Some(&b'*') {
                        depth += 1;
                        i += 2;
                    } else if b[i] == b'*' && b.get(i + 1) == Some(&b'/') {
                        depth -= 1;
                        i += 2;
                        if depth == 0 {
                            break;
                        }
                    } else {
                        i += 1;
                    }
                }
                blank(&mut out, start, i);
            }
            b'r' if (b.get(i + 1) == Some(&b'"') || b.get(i + 1) == Some(&b'#'))
                && (i == 0 || !(b[i - 1].is_ascii_alphanumeric() || b[i - 1] == b'_')) =>
            {
                let mut j = i + 1;
                let mut hashes = 0;
                while b.get(j) == Some(&b'#') {
                    hashes += 1;
                    j += 1;
                }
                if b.get(j) != Some(&b'"') {
                    i += 1;
                    continue;
                }
                let start = j + 1;
                let mut k = start;
                loop {
                    if k >= b.len() {
                        break;
                    }
                    if b[k] == b'"'
                        && b[k + 1..]
                            .iter()
                            .take(hashes)
                            .filter(|&&c| c == b'#')
                            .count()
                            == hashes
                    {
                        break;
                    }
                    k += 1;
                }
                strings.push((start, k.min(b.len())));
                i = (k + 1 + hashes).min(b.len());
            }
            b'"' => {
                let start = i + 1;
                let mut k = start;
                while k < b.len() && b[k] != b'"' {
                    if b[k] == b'\\' {
                        k += 1;
                    }
                    k += 1;
                }
                strings.push((start, k.min(b.len())));
                i = k + 1;
            }
            b'\'' => {
                // A char literal ('x', '\n', '\u{..}') or a lifetime ('a).
                if b.get(i + 1) == Some(&b'\\') {
                    let mut k = i + 3;
                    while k < b.len() && b[k] != b'\'' {
                        k += 1;
                    }
                    i = k + 1;
                } else if b.get(i + 2) == Some(&b'\'') {
                    i += 3;
                } else {
                    // Multi-byte char literal or lifetime.
                    let ch_len = text[i + 1..].chars().next().map_or(1, char::len_utf8);
                    if b.get(i + 1 + ch_len) == Some(&b'\'') {
                        i += 2 + ch_len;
                    } else {
                        i += 1;
                    }
                }
            }
            _ => i += 1,
        }
    }
    (String::from_utf8_lossy(&out).into_owned(), strings)
}

fn sources() -> Vec<Source> {
    let root = workspace_root();
    let mut files = Vec::new();
    rust_files(&root.join("crates"), &mut files);
    assert!(
        files.len() > 100,
        "found only {} Rust files under {}",
        files.len(),
        root.display()
    );
    files
        .into_iter()
        .map(|p| {
            let text =
                std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("{}: {e}", p.display()));
            let rel = p
                .strip_prefix(&root)
                .map(|r| r.to_string_lossy().replace('\\', "/"))
                .unwrap_or_else(|_| p.to_string_lossy().into_owned());
            let (code, strings) = lex(&text);
            Source { rel, code, strings }
        })
        .collect()
}

/// `(byte offset of the call, argument text up to the first `,` or `)`)` of
/// every `needle(` call in the code.
fn calls<'a>(code: &'a str, needle: &str) -> Vec<(usize, &'a str)> {
    let mut v = Vec::new();
    let mut from = 0;
    while let Some(n) = code[from..].find(needle) {
        let at = from + n;
        from = at + needle.len();
        let prev = code[..at].chars().next_back();
        if prev.is_some_and(|c| c.is_ascii_alphanumeric() || c == '_') {
            continue;
        }
        let rest = &code[from..];
        let end = rest.find([',', ')']).unwrap_or(rest.len());
        v.push((at, rest[..end].trim()));
    }
    v
}

/// The `FASTVIDEO_*` name a string literal argument starts with.
fn literal_name(arg: &str) -> Option<&str> {
    let s = arg.strip_prefix('"')?;
    let end = s
        .find(|c: char| !(c.is_ascii_uppercase() || c.is_ascii_digit() || c == '_'))
        .unwrap_or(s.len());
    let name = &s[..end];
    name.starts_with("FASTVIDEO_").then_some(name)
}

#[test]
fn no_direct_fastvideo_reads_outside_the_settings_module() {
    let mut bad = Vec::new();
    for src in sources() {
        if src.rel.starts_with(SETTINGS_DIR) {
            continue;
        }
        for needle in ["env::var(", "env::var_os("] {
            for (at, arg) in calls(&src.code, needle) {
                if arg.starts_with("\"FASTVIDEO_") {
                    bad.push(format!("{}:{}: {needle}{arg}", src.rel, src.line_of(at)));
                }
            }
        }
    }
    assert!(
        bad.is_empty(),
        "read these through fastvideo_models::settings (env_var / var), not std::env:\n{}",
        bad.join("\n")
    );
}

#[test]
fn no_indirect_env_reads_in_the_inference_crates() {
    let mut bad = Vec::new();
    for src in sources() {
        if src.rel.starts_with(SETTINGS_DIR)
            || !INFERENCE_CRATES.iter().any(|c| src.rel.starts_with(c))
        {
            continue;
        }
        for needle in [
            "std::env::var(",
            "std::env::var_os(",
            "env::var(",
            "env::var_os(",
        ] {
            for (at, arg) in calls(&src.code, needle) {
                // `env::var(` also matches inside `std::env::var(`; count once.
                if needle.starts_with("env::") && src.code[..at].ends_with("std::") {
                    continue;
                }
                if arg.starts_with('"') {
                    continue; // a literal: scan 1 covers FASTVIDEO_* literals
                }
                if INDIRECT_ALLOWED
                    .iter()
                    .any(|(f, a, _)| *f == src.rel && *a == arg)
                {
                    continue;
                }
                bad.push(format!("{}:{}: {needle}{arg})", src.rel, src.line_of(at)));
            }
        }
    }
    assert!(
        bad.is_empty(),
        "non-literal environment reads in the inference crates (a FASTVIDEO_* name must be \
         read through fastvideo_models::settings; anything else needs a reviewed entry in \
         INDIRECT_ALLOWED):\n{}",
        bad.join("\n")
    );
}

/// Names in string literals: the test scaffolding's own names are exempt.
fn exempt(name: &str) -> bool {
    name.starts_with("FASTVIDEO_TEST_")
}

#[test]
fn every_named_setting_is_a_row_and_every_row_is_named() {
    let mut bad = Vec::new();
    let mut named = BTreeSet::new();
    for src in sources() {
        // The table and its tests name every row; `mod.rs` names the
        // profile variable it reads.
        if src.rel.starts_with(SETTINGS_DIR) && !src.rel.ends_with("/mod.rs") {
            continue;
        }
        for &(a, z) in &src.strings {
            let lit = &src.code.as_bytes()[a..z];
            let lit = String::from_utf8_lossy(lit);
            let mut from = 0;
            while let Some(n) = lit[from..].find("FASTVIDEO_") {
                let at = from + n;
                let prev = lit[..at].chars().next_back();
                let tail = &lit[at..];
                let end = tail
                    .find(|c: char| !(c.is_ascii_uppercase() || c.is_ascii_digit() || c == '_'))
                    .unwrap_or(tail.len());
                let name = &tail[..end];
                from = at + end.max(1);
                if prev.is_some_and(|c| c.is_ascii_alphanumeric() || c == '_')
                    || name.ends_with('_')
                {
                    continue; // part of a longer identifier, or a prefix match
                }
                if exempt(name) {
                    continue;
                }
                named.insert(name.to_string());
                if lookup(name).is_none() {
                    bad.push(format!("{}:{}: {name}", src.rel, src.line_of(a)));
                }
            }
        }
    }
    assert!(
        bad.is_empty(),
        "FASTVIDEO_* names that are not rows of techniques::settings::SETTINGS (add the row):\n{}",
        bad.join("\n")
    );
    let unnamed: Vec<_> = SETTINGS
        .iter()
        .filter(|s| !named.contains(s.name))
        .map(|s| s.name)
        .collect();
    assert!(
        unnamed.is_empty(),
        "rows no source names (stale?): {unnamed:?}"
    );
}

/// The parse a typed getter implies.
const TYPED: &[(&str, Parse)] = &[
    ("bool_flag(", Parse::Flag),
    ("usize_flag(", Parse::Usize),
    ("f32_flag(", Parse::F32),
    ("f64_flag(", Parse::F64),
    ("string_flag(", Parse::Keyword),
];

/// The literal default argument of a typed getter call at `at`, if literal.
fn second_arg(code: &str, at: usize) -> Option<String> {
    let rest = &code[at..];
    let open = rest.find('(')?;
    let comma = rest[open..].find(',')? + open;
    let close = rest[comma..].find(')')? + comma;
    let arg = rest[comma + 1..close].trim();
    let lit = arg.trim_matches('"');
    let is_literal = arg == "true"
        || arg == "false"
        || (arg.starts_with('"') && arg.ends_with('"') && arg.len() >= 2)
        || arg.replace('_', "").parse::<f64>().is_ok();
    is_literal.then(|| match arg {
        "true" => "1".to_string(),
        "false" => "0".to_string(),
        _ => lit.replace('_', ""),
    })
}

#[test]
fn literal_reads_agree_with_their_rows() {
    let mut bad = Vec::new();
    for src in sources() {
        if src.rel.starts_with(SETTINGS_DIR) {
            continue;
        }
        let at_line = |at: usize| format!("{}:{}", src.rel, src.line_of(at));
        for &(needle, parse) in TYPED {
            for (at, arg) in calls(&src.code, needle) {
                let Some(name) = literal_name(arg) else {
                    continue;
                };
                if exempt(name) {
                    continue;
                }
                let Some(row) = lookup(name) else { continue }; // reported by the names test
                if row.parse != parse {
                    bad.push(format!(
                        "{}: {name}: {needle}..) but the row says {:?}",
                        at_line(at),
                        row.parse
                    ));
                }
                if row.reader != Reader::Layered {
                    bad.push(format!(
                        "{}: {name}: a layered getter on a {:?} row",
                        at_line(at),
                        row.reader
                    ));
                }
                if let (DefaultValue::Value(v), Some(d)) = (row.default, second_arg(&src.code, at))
                {
                    let same = match parse {
                        Parse::Usize | Parse::F32 | Parse::F64 => {
                            v.parse::<f64>().ok() == d.parse::<f64>().ok()
                        }
                        _ => v == d,
                    };
                    if !same {
                        bad.push(format!(
                            "{}: {name}: default {d:?}, the row says {v:?}",
                            at_line(at)
                        ));
                    }
                }
            }
        }
        for needle in ["settings::var(", "settings::is_set("] {
            for (at, arg) in calls(&src.code, needle) {
                if let Some(row) = literal_name(arg).and_then(lookup) {
                    if row.reader != Reader::Layered {
                        bad.push(format!(
                            "{}: {}: {needle}..) on a {:?} row",
                            at_line(at),
                            row.name,
                            row.reader
                        ));
                    }
                }
            }
        }
        for needle in ["settings::env_var(", "settings::env_var_os("] {
            for (at, arg) in calls(&src.code, needle) {
                if let Some(row) = literal_name(arg).and_then(lookup) {
                    let reviewed = ENV_READS_OF_LAYERED
                        .iter()
                        .any(|(f, n, _)| *f == src.rel && *n == row.name);
                    if row.reader != Reader::Env && !reviewed {
                        bad.push(format!(
                            "{}: {}: {needle}..) on a {:?} row",
                            at_line(at),
                            row.name,
                            row.reader
                        ));
                    }
                }
            }
        }
    }
    assert!(
        bad.is_empty(),
        "reads that disagree with their rows:\n{}",
        bad.join("\n")
    );
}

#[test]
fn scope_doc_table_is_generated_from_the_rows() {
    let path = workspace_root().join("docs/scope.md");
    let doc = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let want = super::render_markdown();
    let begin = doc.find(super::DOC_BEGIN);
    let end = doc
        .find(super::DOC_END)
        .map(|e| e + super::DOC_END.len() + 1);
    let have = match (begin, end) {
        (Some(b), Some(e)) if e <= doc.len() && b < e => &doc[b..e],
        _ => "",
    };
    assert!(
        have == want,
        "docs/scope.md's settings table is stale; replace the region between the markers with:\n{want}"
    );
}

#[test]
fn the_lexer_blanks_comments_and_finds_strings() {
    let (code, strings) =
        lex("let a = \"x // y\"; // FASTVIDEO_GONE\nlet b = r#\"q\"#; /* c */ let c = 'z'; fn f<'a>() {}");
    assert!(!code.contains("FASTVIDEO_GONE"));
    assert!(!code.contains("/* c */"));
    let lits: Vec<&str> = strings.iter().map(|&(a, z)| &code[a..z]).collect();
    assert_eq!(lits, vec!["x // y", "q"]);
}
