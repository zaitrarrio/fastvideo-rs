//! Canonical JSON (ADR-0007): the RFC 8785 (JCS) subset this crate needs,
//! implemented here rather than through a JCS crate.
//!
//! - Object members are sorted by the UTF-16 code units of their keys (JCS
//!   §3.2.3). For ASCII keys this is byte order.
//! - No whitespace; `null`, `true`, `false` as literals.
//! - Strings are escaped as `JSON.stringify` does (JCS §3.2.2.2): `"`, `\`,
//!   `\b` `\t` `\n` `\f` `\r`, other control characters as `\u00xx` with
//!   lowercase hex; everything else, including non-ASCII, literally.
//! - Numbers: **integers only**. A float anywhere is
//!   [`CanonicalError::Float`]; manifests never carry one (model.toml may:
//!   it is not digested through this module). Integers are written as plain
//!   decimal digits, which agrees with JCS (ES6 `Number::toString`) for
//!   magnitudes up to 2^53; [`crate::manifest::Manifest::validate`] bounds
//!   every manifest integer by [`MAX_SAFE_INTEGER`] so a valid manifest is
//!   exactly JCS-conformant.

use serde_json::Value;

/// 2^53: the largest integer every JCS implementation serialises exactly.
pub const MAX_SAFE_INTEGER: u64 = 1 << 53;

/// Why a JSON value has no canonical form here.
#[derive(thiserror::Error, Debug, Clone, PartialEq, Eq)]
pub enum CanonicalError {
    #[error("floating-point number at `{path}`: canonical manifests carry integers only")]
    Float { path: String },
}

/// The canonical bytes of `value`.
pub fn to_canonical_bytes(value: &Value) -> Result<Vec<u8>, CanonicalError> {
    let mut out = Vec::new();
    write_value(value, &mut out, &mut String::from("$"))?;
    Ok(out)
}

/// The canonical form as a `String` (always valid UTF-8: JSON text).
pub fn to_canonical_string(value: &Value) -> Result<String, CanonicalError> {
    let bytes = to_canonical_bytes(value)?;
    // Built from `str` pieces and ASCII punctuation: always UTF-8.
    Ok(String::from_utf8_lossy(&bytes).into_owned())
}

fn write_value(value: &Value, out: &mut Vec<u8>, path: &mut String) -> Result<(), CanonicalError> {
    match value {
        Value::Null => out.extend_from_slice(b"null"),
        Value::Bool(true) => out.extend_from_slice(b"true"),
        Value::Bool(false) => out.extend_from_slice(b"false"),
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                out.extend_from_slice(i.to_string().as_bytes());
            } else if let Some(u) = n.as_u64() {
                out.extend_from_slice(u.to_string().as_bytes());
            } else {
                return Err(CanonicalError::Float { path: path.clone() });
            }
        }
        Value::String(s) => write_string(s, out),
        Value::Array(items) => {
            out.push(b'[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push(b',');
                }
                let len = path.len();
                path.push_str(&format!("[{i}]"));
                write_value(item, out, path)?;
                path.truncate(len);
            }
            out.push(b']');
        }
        Value::Object(map) => {
            let mut keys: Vec<&String> = map.keys().collect();
            keys.sort_by(|a, b| utf16_cmp(a, b));
            out.push(b'{');
            for (i, key) in keys.into_iter().enumerate() {
                if i > 0 {
                    out.push(b',');
                }
                write_string(key, out);
                out.push(b':');
                let len = path.len();
                path.push('.');
                path.push_str(key);
                if let Some(v) = map.get(key) {
                    write_value(v, out, path)?;
                }
                path.truncate(len);
            }
            out.push(b'}');
        }
    }
    Ok(())
}

/// JCS key order: by UTF-16 code units.
fn utf16_cmp(a: &str, b: &str) -> std::cmp::Ordering {
    a.encode_utf16().cmp(b.encode_utf16())
}

/// `JSON.stringify` string escaping.
pub fn write_string(s: &str, out: &mut Vec<u8>) {
    out.push(b'"');
    for c in s.chars() {
        match c {
            '"' => out.extend_from_slice(b"\\\""),
            '\\' => out.extend_from_slice(b"\\\\"),
            '\u{8}' => out.extend_from_slice(b"\\b"),
            '\t' => out.extend_from_slice(b"\\t"),
            '\n' => out.extend_from_slice(b"\\n"),
            '\u{c}' => out.extend_from_slice(b"\\f"),
            '\r' => out.extend_from_slice(b"\\r"),
            c if (c as u32) < 0x20 => {
                out.extend_from_slice(format!("\\u{:04x}", c as u32).as_bytes());
            }
            c => {
                let mut buf = [0u8; 4];
                out.extend_from_slice(c.encode_utf8(&mut buf).as_bytes());
            }
        }
    }
    out.push(b'"');
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn sorts_keys_and_strips_whitespace() {
        let v: Value =
            serde_json::from_str(r#"{ "b": [1, 2], "a": {"z": null, "y": true} }"#).unwrap();
        assert_eq!(
            to_canonical_string(&v).unwrap(),
            r#"{"a":{"y":true,"z":null},"b":[1,2]}"#
        );
    }

    #[test]
    fn rejects_floats_with_a_path() {
        let v = json!({"files": [{"size": 1.5}]});
        assert_eq!(
            to_canonical_bytes(&v).unwrap_err(),
            CanonicalError::Float {
                path: "$.files[0].size".to_string()
            }
        );
    }

    #[test]
    fn escapes_like_json_stringify() {
        let v = json!("a\"b\\c\n\u{1}é/");
        assert_eq!(
            to_canonical_string(&v).unwrap(),
            "\"a\\\"b\\\\c\\n\\u0001é/\""
        );
    }

    #[test]
    fn utf16_order_differs_from_code_point_order_for_astral_keys() {
        // U+FF61 (one UTF-16 unit 0xFF61) sorts after U+1F600 (surrogates D83D DE00).
        let v = json!({"\u{1F600}": 1, "\u{FF61}": 2});
        assert_eq!(
            to_canonical_string(&v).unwrap(),
            "{\"\u{1F600}\":1,\"\u{FF61}\":2}"
        );
    }
}
