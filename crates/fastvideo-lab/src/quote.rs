//! Bash's `printf %q` (bash 5.2, `LC_ALL=C`), so `fv-lab run --dry-run`
//! prints a command line exactly as `runpod-matrix.sh`'s `FV_DRY_RUN=1` does
//! (`log "dry-run $name: $(printf '%q ' "$@")"`), and the two can be compared
//! byte for byte (`tests/dry_run.rs`).
//!
//! The rules, from bash's `sh_backslash_quote` / `ansic_quote` and checked
//! against bash 5.2.21: the empty string is `''`; a string with a byte outside
//! printable ASCII is ANSI-C quoted (`$'…'`, `\a \b \E \f \n \r \t \v \\ \'`,
//! octal `\ooo` for the rest); otherwise each shell-special character gets a
//! backslash, and so do a leading `#` and a `~` at the start or after `=` or
//! `:`.

/// The printable ASCII characters `printf %q` escapes with a backslash.
const SPECIAL: &[u8] = b" !\"$&'()*,;<>?[\\]^`{|}";

/// One argument as `printf %q` writes it.
pub fn shell_quote(s: &str) -> String {
    if s.is_empty() {
        return "''".to_owned();
    }
    let bytes = s.as_bytes();
    if bytes.iter().any(|b| !(0x20..0x7f).contains(b)) {
        return ansi_c(bytes);
    }
    let mut out = String::with_capacity(s.len() + 8);
    for (i, &b) in bytes.iter().enumerate() {
        let prev = i.checked_sub(1).map(|j| bytes[j]);
        let escape = SPECIAL.contains(&b)
            || (b == b'#' && i == 0)
            || (b == b'~' && (i == 0 || matches!(prev, Some(b'=') | Some(b':'))));
        if escape {
            out.push('\\');
        }
        out.push(char::from(b));
    }
    out
}

fn ansi_c(bytes: &[u8]) -> String {
    let mut out = String::from("$'");
    for &b in bytes {
        match b {
            0x07 => out.push_str("\\a"),
            0x08 => out.push_str("\\b"),
            0x0c => out.push_str("\\f"),
            b'\n' => out.push_str("\\n"),
            b'\r' => out.push_str("\\r"),
            b'\t' => out.push_str("\\t"),
            0x0b => out.push_str("\\v"),
            0x1b => out.push_str("\\E"),
            b'\\' => out.push_str("\\\\"),
            b'\'' => out.push_str("\\'"),
            0x20..=0x7e => out.push(char::from(b)),
            _ => out.push_str(&format!("\\{b:03o}")),
        }
    }
    out.push('\'');
    out
}

/// `printf '%q ' "$@"`: every argument quoted and followed by one space.
pub fn printf_q_line<S: AsRef<str>>(args: &[S]) -> String {
    let mut out = String::new();
    for a in args {
        out.push_str(&shell_quote(a.as_ref()));
        out.push(' ');
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_bash_5_2() {
        // Each pair was printed by `LC_ALL=C bash -c 'printf %q "$0"'`.
        for (raw, want) in [
            ("", "''"),
            ("plain-word_1.2/x", "plain-word_1.2/x"),
            ("a b", "a\\ b"),
            ("a,b", "a\\,b"),
            ("<d>Hi</d>", "\\<d\\>Hi\\</d\\>"),
            ("a'b", "a\\'b"),
            ("#a", "\\#a"),
            ("a#", "a#"),
            ("~a", "\\~a"),
            ("a~", "a~"),
            ("=~", "=\\~"),
            ("a=~b", "a=\\~b"),
            (":~", ":\\~"),
            ("x=1", "x=1"),
            ("\u{1b}", "$'\\E'"),
            ("\u{1}", "$'\\001'"),
            ("a'b\u{1}", "$'a\\'b\\001'"),
            ("\u{7}\u{8}\u{c}\n\r\u{b}", "$'\\a\\b\\f\\n\\r\\v'"),
            ("\\\u{7f}", "$'\\\\\\177'"),
            ("x\u{1}\"y", "$'x\\001\"y'"),
            ("é", "$'\\303\\251'"),
            ("a\tb", "$'a\\tb'"),
        ] {
            assert_eq!(shell_quote(raw), want, "{raw:?}");
        }
    }

    #[test]
    fn every_printable_special_is_escaped() {
        for b in 0x20u8..0x7f {
            let c = char::from(b);
            let q = shell_quote(&format!("a{c}b"));
            let escaped = q == format!("a\\{c}b");
            assert_eq!(escaped, SPECIAL.contains(&b), "{c:?} -> {q}");
        }
    }

    #[test]
    fn line_has_a_trailing_space_per_argument() {
        assert_eq!(printf_q_line(&["a", "b c"]), "a b\\ c ");
        assert_eq!(printf_q_line::<&str>(&[]), "");
    }
}
