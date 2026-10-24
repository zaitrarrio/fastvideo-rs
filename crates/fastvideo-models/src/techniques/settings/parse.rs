//! The shared parse rules of `FASTVIDEO_*` values, as pure functions of the
//! raw value (`None` when the setting is unset).
//!
//! These are the one implementation of the rules the runtime's typed getters
//! apply (`fastvideo_cudarc::rt::envflag::{bool_flag, usize_flag, f32_flag,
//! f64_flag, string_flag}` call them on [`super::var`]), so the declarative
//! table ([`super::table`]) and the off-identity tests check the very code
//! the readers run.

/// A boolean flag.
///
/// With `default = true` (a flag that is on unless turned off): `0`, `false`
/// or `off` (trimmed, any case) turn it off; anything else, unset included,
/// leaves it on. With `default = false`: `1`, `true` or `on` (trimmed, any
/// case) turn it on; anything else, unset included, leaves it off.
pub fn flag(raw: Option<&str>, default: bool) -> bool {
    match raw {
        Some(v) => {
            let v = v.trim();
            if default {
                !(v == "0" || v.eq_ignore_ascii_case("false") || v.eq_ignore_ascii_case("off"))
            } else {
                v == "1" || v.eq_ignore_ascii_case("true") || v.eq_ignore_ascii_case("on")
            }
        }
        None => default,
    }
}

/// A `usize`: the value as written (not trimmed) parsed as a decimal
/// `usize`; `default` when unset or unparsable.
pub fn usize_or(raw: Option<&str>, default: usize) -> usize {
    raw.and_then(|s| s.parse::<usize>().ok()).unwrap_or(default)
}

/// An `f32`: the value as written (not trimmed); `default` when unset or
/// unparsable.
pub fn f32_or(raw: Option<&str>, default: f32) -> f32 {
    raw.and_then(|s| s.parse::<f32>().ok()).unwrap_or(default)
}

/// An `f64`: the trimmed value; `default` when unset or unparsable.
pub fn f64_or(raw: Option<&str>, default: f64) -> f64 {
    raw.and_then(|v| v.trim().parse().ok()).unwrap_or(default)
}

/// A keyword: the value (or `default` when unset) lower-cased with
/// `to_ascii_lowercase`, not trimmed. An empty value stays empty.
pub fn lower_or(raw: Option<&str>, default: &str) -> String {
    raw.unwrap_or(default).to_ascii_lowercase()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The truthy / falsy table of [`flag`], both polarities.
    #[test]
    fn flag_truth_table() {
        // (raw, default-on result, default-off result)
        let cases: &[(Option<&str>, bool, bool)] = &[
            (None, true, false),
            (Some(""), true, false),
            (Some("0"), false, false),
            (Some(" 0 "), false, false),
            (Some("false"), false, false),
            (Some("FALSE"), false, false),
            (Some("off"), false, false),
            (Some("Off"), false, false),
            (Some("no"), true, false),
            (Some("1"), true, true),
            (Some(" 1\n"), true, true),
            (Some("true"), true, true),
            (Some("TRUE"), true, true),
            (Some("on"), true, true),
            (Some("ON"), true, true),
            (Some("yes"), true, false),
            (Some("2"), true, false),
            (Some("anything"), true, false),
        ];
        for &(raw, on, off) in cases {
            assert_eq!(flag(raw, true), on, "default-on flag of {raw:?}");
            assert_eq!(flag(raw, false), off, "default-off flag of {raw:?}");
        }
    }

    #[test]
    fn number_tables() {
        assert_eq!(usize_or(None, 7), 7);
        assert_eq!(usize_or(Some("12"), 7), 12);
        assert_eq!(usize_or(Some(" 12"), 7), 7, "usize values are not trimmed");
        assert_eq!(usize_or(Some("-1"), 7), 7);
        assert_eq!(usize_or(Some("1.5"), 7), 7);
        assert_eq!(usize_or(Some(""), 7), 7);
        assert_eq!(f32_or(None, 0.1), 0.1);
        assert_eq!(f32_or(Some("0.25"), 0.1), 0.25);
        assert_eq!(
            f32_or(Some(" 0.25"), 0.1),
            0.1,
            "f32 values are not trimmed"
        );
        assert_eq!(f32_or(Some("x"), 0.1), 0.1);
        assert_eq!(f64_or(None, 0.8), 0.8);
        assert_eq!(f64_or(Some(" 0.9 "), 0.8), 0.9, "f64 values are trimmed");
        assert_eq!(f64_or(Some("nope"), 0.8), 0.8);
    }

    #[test]
    fn keyword_table() {
        assert_eq!(lower_or(None, "auto"), "auto");
        assert_eq!(lower_or(Some("NVRTC"), "auto"), "nvrtc");
        assert_eq!(
            lower_or(Some(" Rule "), "auto"),
            " rule ",
            "keywords are not trimmed"
        );
        assert_eq!(
            lower_or(Some(""), "auto"),
            "",
            "an empty value is not the default"
        );
    }
}
