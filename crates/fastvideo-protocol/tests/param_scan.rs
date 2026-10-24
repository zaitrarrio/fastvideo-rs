//! Plan E2 (ADR-0012): no adapter keeps a request-parameter limit of its
//! own. Every limit or allowed set lives in
//! `fastvideo_protocol::params::rules` (or comes from the model's caps), and
//! the adapters validate with those constants.
//!
//! This scans the request-validation sources of every adapter (the code
//! before its `#[cfg(test)]` module) for numeric limit literals:
//!
//! - a `const` whose initializer holds a number;
//! - a range with a literal bound of 2 or more (`1..=200`, `6..=max`);
//! - a comparison with a literal of 2 or more (`len() <= 64`, `> 15`).
//!
//! `0` and `1` pass (positivity, non-emptiness, `1..` lower bounds); so do
//! the entries of [`ALLOWED`], each a non-parameter number with its reason.

use std::path::{Path, PathBuf};

/// The request-validation sources, relative to the workspace root.
const SCOPE: &[&str] = &[
    "crates/fastvideo-openai-videos/src/videos.rs",
    "crates/fastvideo-openai-videos/src/fastwan.rs",
    "crates/fastvideo-openai-videos/src/models.rs",
    "crates/fastvideo-openai-videos/src/params.rs",
    "crates/fastvideo-minimax/src/create.rs",
    "crates/fastvideo-minimax/src/lib.rs",
    "crates/fastvideo-minimax/src/query.rs",
    "crates/fastvideo-minimax/src/params.rs",
    "crates/fastvideo-ltxapi/src/body.rs",
    "crates/fastvideo-ltxapi/src/request.rs",
    "crates/fastvideo-ltxapi/src/models.rs",
    "crates/fastvideo-ltxapi/src/params.rs",
    "crates/fastvideo-fal/src/schema.rs",
    "crates/fastvideo-fal/src/schema/ltx.rs",
    "crates/fastvideo-fal/src/schema/wan.rs",
    "crates/fastvideo-fal/src/schema/a2v.rs",
    "crates/fastvideo-fal/src/schema/edit.rs",
    "crates/fastvideo-fal/src/schema/ingredient.rs",
    "crates/fastvideo-fal/src/catalog.rs",
    "crates/fastvideo-fal/src/params.rs",
    "crates/fastvideo-fal/src/director/messages.rs",
    "crates/fastvideo-reactor/src/commands.rs",
    "crates/fastvideo-reactor/src/http.rs",
    "crates/fastvideo-reactor/src/uploads.rs",
    "crates/fastvideo-reactor/src/session.rs",
    "crates/fastvideo-serve/src/native.rs",
    "crates/fastvideo-serve/src/params.rs",
];

/// `(file, text on the line, reason)`: numbers that are not request limits.
const ALLOWED: &[(&str, &str, &str)] = &[
    (
        "fastvideo-minimax/src/lib.rs",
        "const MB: u64",
        "a byte unit for the media ingestion sizes (service limits, not request parameters)",
    ),
    (
        "fastvideo-minimax/src/create.rs",
        "const MIB: usize",
        "a byte unit formatting the body-size error",
    ),
    (
        "fastvideo-fal/src/schema.rs",
        "f.abs() < 9.0e15",
        "the largest integral float read as an integer (JSON precision)",
    ),
    (
        "fastvideo-reactor/src/commands.rs",
        "f.abs() < 9.0e15",
        "the largest integral float read as an integer (JSON precision)",
    ),
    (
        "fastvideo-reactor/src/session.rs",
        "Duration::from_secs(10)",
        "the driver's close timeout (a service setting)",
    ),
    (
        "fastvideo-reactor/src/uploads.rs",
        "const RESOLVE_TIMEOUT: Duration",
        "how long a command waits for an upload's bytes (a service setting)",
    ),
    (
        "fastvideo-openai-videos/src/videos.rs",
        "const URL_TTL: Duration",
        "the signed-URL lifetime (a service setting)",
    ),
];

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// The line without its `//` comment and with string literals emptied.
fn code_of(line: &str) -> String {
    let mut out = String::new();
    let mut chars = line.chars().peekable();
    let mut in_str = false;
    while let Some(c) = chars.next() {
        if in_str {
            match c {
                '\\' => {
                    chars.next();
                }
                '"' => {
                    in_str = false;
                    out.push('"');
                }
                _ => {}
            }
            continue;
        }
        match c {
            '"' => {
                in_str = true;
                out.push('"');
            }
            '/' if chars.peek() == Some(&'/') => break,
            _ => out.push(c),
        }
    }
    out
}

/// The numeric literal starting at byte `i` of `s`, when one starts there
/// (not inside an identifier): its text.
fn literal_at(s: &str, i: usize) -> Option<&str> {
    let b = s.as_bytes();
    if !b.get(i)?.is_ascii_digit() {
        return None;
    }
    if i > 0 && (b[i - 1].is_ascii_alphanumeric() || b[i - 1] == b'_' || b[i - 1] == b'.') {
        return None;
    }
    let mut j = i;
    while j < b.len() {
        let c = b[j];
        let dot_then_digit = c == b'.' && b.get(j + 1).is_some_and(u8::is_ascii_digit);
        if c.is_ascii_digit() || c == b'_' || dot_then_digit || c == b'e' {
            j += 1;
        } else {
            break;
        }
    }
    Some(&s[i..j])
}

/// A literal that can be a limit: an integer of 2 or more, or a float other
/// than 0 and 1.
fn is_limit(lit: &str) -> bool {
    let t: String = lit.chars().filter(|c| *c != '_').collect();
    if t.contains('.') || t.contains('e') {
        return t.parse::<f64>().is_ok_and(|f| f != 0.0 && f != 1.0);
    }
    t.parse::<u128>().is_ok_and(|n| n >= 2)
}

/// Why `code` holds a limit literal, if it does.
fn finding(code: &str) -> Option<&'static str> {
    let t = code.trim_start();
    let is_const = [
        "const ",
        "pub const ",
        "pub(crate) const ",
        "pub(super) const ",
    ]
    .iter()
    .any(|p| t.starts_with(p));
    if is_const {
        if let Some((_, init)) = t.split_once('=') {
            if (0..init.len()).any(|i| literal_at(init, i).is_some_and(is_limit)) {
                return Some("a const with a numeric initializer");
            }
        }
    }
    let b = code.as_bytes();
    for i in 0..b.len() {
        // Ranges: `<lit>..`, `..<lit>`, `..=<lit>`.
        if b[i] == b'.' && b.get(i + 1) == Some(&b'.') {
            let mut k = i;
            while k > 0 && (b[k - 1].is_ascii_alphanumeric() || b[k - 1] == b'_') {
                k -= 1;
            }
            let before = literal_at(code, k).filter(|l| k + l.len() == i);
            let mut j = i + 2;
            if b.get(j) == Some(&b'=') {
                j += 1;
            }
            while b.get(j) == Some(&b' ') {
                j += 1;
            }
            let after = literal_at(code, j);
            if before.is_some_and(is_limit) || after.is_some_and(is_limit) {
                return Some("a range with a literal bound");
            }
        }
        // Comparisons: `<`, `>`, `<=`, `>=` (not `=>`, `->`, `<<`, `>>`).
        if matches!(b[i], b'<' | b'>') {
            let prev = if i > 0 { b[i - 1] } else { b' ' };
            if matches!(prev, b'=' | b'-' | b'<' | b'>') || b.get(i + 1) == Some(&b[i]) {
                continue;
            }
            let mut j = i + 1;
            if b.get(j) == Some(&b'=') {
                j += 1;
            }
            while b.get(j) == Some(&b' ') {
                j += 1;
            }
            if literal_at(code, j).is_some_and(is_limit) {
                return Some("a comparison with a literal");
            }
        }
    }
    None
}

#[test]
fn adapters_keep_no_numeric_limits_of_their_own() {
    let root = root();
    let mut found = Vec::new();
    let mut used = vec![false; ALLOWED.len()];
    for rel in SCOPE {
        let path = root.join(rel);
        let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{rel}: {e}"));
        for (n, line) in text.lines().enumerate() {
            if line.starts_with("#[cfg(test)]") {
                break;
            }
            let Some(why) = finding(&code_of(line)) else {
                continue;
            };
            let allowed = ALLOWED
                .iter()
                .position(|(f, t, _)| rel.ends_with(f) && line.contains(t));
            match allowed {
                Some(i) => used[i] = true,
                None => found.push(format!("{rel}:{}: {why}: {}", n + 1, line.trim())),
            }
        }
    }
    assert!(
        found.is_empty(),
        "request limits belong in fastvideo_protocol::params::rules (or the caps):\n{}",
        found.join("\n")
    );
    let stale: Vec<&str> = ALLOWED
        .iter()
        .zip(&used)
        .filter(|(_, u)| !**u)
        .map(|((_, t, _), _)| *t)
        .collect();
    assert!(
        stale.is_empty(),
        "allow-list entries that match nothing: {stale:?}"
    );
}

#[test]
fn the_scan_finds_what_it_should() {
    for (code, hit) in [
        ("pub const MAX: usize = 5000;", true),
        ("if !(1..=200).contains(&s) {", true),
        ("(6..=max).step_by(2)", true),
        ("if id.len() <= 64 {", true),
        ("if v > 15 {", true),
        ("Some(f) if f.abs() < 9.0e15 =>", true),
        ("pub const N: i64 = rules::X.min;", false),
        ("pub const ALL: [Kind; 4] = [A, B, C, D];", false),
        ("pub const DEFAULT: u32 = 0;", false),
        ("for i in 0..n {", false),
        ("if !(1..=cap).contains(n) {", false),
        ("Resolution::P480 => 480,", false),
        ("let v: Vec<u32> = Vec::new();", false),
        ("if n == 0 || n > MAX {", false),
        ("x = \"1..=200\";", false),
        ("y // 1..=200", false),
    ] {
        assert_eq!(finding(&code_of(code)).is_some(), hit, "{code}");
    }
}
