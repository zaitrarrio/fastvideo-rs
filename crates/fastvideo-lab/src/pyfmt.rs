//! What the replaced reports printed, byte for byte: Python's `json` module
//! (order-preserving objects, `json.dumps` with and without `indent`), Python's
//! `repr` / `str` of floats and strings and its `format()` of numbers, and
//! jq's number output (docs/adr/0019-fv-lab.md §13).
//!
//! `serde_json` sorts object keys (the workspace builds it without
//! `preserve_order`) and spells floats as ryu does (`1e-5`), where Python
//! keeps the file's key order and writes `1e-05`. The reports echo JSON they
//! read (`json.dumps(benchmark)[:3000]`, `results.json`), so they need a
//! value that keeps the order and an int / float distinction, and the
//! printers below. No dependency: a small JSON reader, Python's float `repr`
//! from Rust's shortest round-trip digits, and `{:.Nf}` / `{:.Ne}` with
//! Python's spelling of the special values and of the exponent.

use std::fmt::Write as _;

/// A JSON value as Python's `json.loads` returns it: objects keep their key
/// order (a repeated key keeps its first position and takes the last value,
/// as a `dict` does), integers stay integers.
#[derive(Debug, Clone, PartialEq)]
pub enum Py {
    None,
    Bool(bool),
    /// The integer's digits, canonical (no `-0`); any size.
    Int(String),
    Float(f64),
    Str(String),
    List(Vec<Py>),
    Dict(Vec<(String, Py)>),
}

impl Py {
    /// `d.get(k)` (`None` for a missing key or a non-dict).
    pub fn get(&self, k: &str) -> Option<&Py> {
        match self {
            Py::Dict(items) => items.iter().find(|(key, _)| key == k).map(|(_, v)| v),
            _ => None,
        }
    }

    /// `d.get(k)` with Python's `None` for a missing key.
    pub fn at(&self, k: &str) -> &Py {
        self.get(k).unwrap_or(&Py::None)
    }

    /// `d[k] = v`: an existing key keeps its position.
    pub fn set(&mut self, k: &str, v: Py) {
        if let Py::Dict(items) = self {
            if let Some(slot) = items.iter_mut().find(|(key, _)| key == k) {
                slot.1 = v;
            } else {
                items.push((k.to_owned(), v));
            }
        }
    }

    /// An empty `dict`.
    pub fn dict() -> Py {
        Py::Dict(Vec::new())
    }

    /// Python truthiness.
    pub fn truthy(&self) -> bool {
        match self {
            Py::None => false,
            Py::Bool(b) => *b,
            Py::Int(s) => s != "0",
            Py::Float(f) => *f != 0.0,
            Py::Str(s) => !s.is_empty(),
            Py::List(v) => !v.is_empty(),
            Py::Dict(v) => !v.is_empty(),
        }
    }

    /// `a or b`.
    pub fn or(&self, b: Py) -> Py {
        if self.truthy() {
            self.clone()
        } else {
            b
        }
    }

    pub fn is_none(&self) -> bool {
        matches!(self, Py::None)
    }

    /// The number as an `f64` (ints and bools as Python converts them).
    pub fn num(&self) -> Option<f64> {
        match self {
            Py::Int(s) => s.parse::<f64>().ok(),
            Py::Float(f) => Some(*f),
            Py::Bool(b) => Some(f64::from(u8::from(*b))),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Py::Str(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_list(&self) -> &[Py] {
        match self {
            Py::List(v) => v,
            _ => &[],
        }
    }

    pub fn items(&self) -> &[(String, Py)] {
        match self {
            Py::Dict(v) => v,
            _ => &[],
        }
    }

    pub fn int(i: i64) -> Py {
        Py::Int(i.to_string())
    }

    /// A Python number from an `f64` result that Python computed in floats.
    pub fn float(f: f64) -> Py {
        Py::Float(f)
    }

    fn as_i64(&self) -> Option<i64> {
        match self {
            Py::Int(s) => s.parse().ok(),
            Py::Bool(b) => Some(i64::from(*b)),
            _ => None,
        }
    }

    /// `a + b` on numbers: int + int stays int, else float.
    pub fn add(&self, b: &Py) -> Option<Py> {
        match (self.as_i64(), b.as_i64()) {
            (Some(x), Some(y)) => x.checked_add(y).map(Py::int),
            _ => Some(Py::Float(self.num()? + b.num()?)),
        }
    }

    /// `a - b`.
    pub fn sub(&self, b: &Py) -> Option<Py> {
        match (self.as_i64(), b.as_i64()) {
            (Some(x), Some(y)) => x.checked_sub(y).map(Py::int),
            _ => Some(Py::Float(self.num()? - b.num()?)),
        }
    }

    /// `a * b`.
    pub fn mul(&self, b: &Py) -> Option<Py> {
        match (self.as_i64(), b.as_i64()) {
            (Some(x), Some(y)) => x.checked_mul(y).map(Py::int),
            _ => Some(Py::Float(self.num()? * b.num()?)),
        }
    }

    /// `a / b` (true division: always a float).
    pub fn div(&self, b: &Py) -> Option<Py> {
        let d = b.num()?;
        if d == 0.0 {
            return None;
        }
        Some(Py::Float(self.num()? / d))
    }
}

/// Python's `str()` of a value (`format(v, "")`, what an f-string field
/// without a spec prints).
pub fn py_str(v: &Py) -> String {
    match v {
        Py::Str(s) => s.clone(),
        other => repr(other),
    }
}

/// Python's `repr()` of a value.
pub fn repr(v: &Py) -> String {
    match v {
        Py::None => "None".into(),
        Py::Bool(true) => "True".into(),
        Py::Bool(false) => "False".into(),
        Py::Int(s) => s.clone(),
        Py::Float(f) => float_repr(*f),
        Py::Str(s) => str_repr(s),
        Py::List(items) => {
            let parts: Vec<String> = items.iter().map(repr).collect();
            format!("[{}]", parts.join(", "))
        }
        Py::Dict(items) => {
            let parts: Vec<String> = items
                .iter()
                .map(|(k, v)| format!("{}: {}", str_repr(k), repr(v)))
                .collect();
            format!("{{{}}}", parts.join(", "))
        }
    }
}

/// `repr()` of a tuple of values.
pub fn tuple_repr(items: &[Py]) -> String {
    let parts: Vec<String> = items.iter().map(repr).collect();
    if parts.len() == 1 {
        format!("({},)", parts[0])
    } else {
        format!("({})", parts.join(", "))
    }
}

/// Python's `repr()` of a `str`: single quotes unless the text holds a single
/// quote and no double quote; backslash escapes for the quote, `\\`, `\t`,
/// `\n`, `\r` and other control characters. Printable non-ASCII characters
/// are kept as they are (Python's `str.isprintable` for the common cases).
pub fn str_repr(s: &str) -> String {
    let q = if s.contains('\'') && !s.contains('"') {
        '"'
    } else {
        '\''
    };
    let mut out = String::with_capacity(s.len() + 2);
    out.push(q);
    for c in s.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '\t' => out.push_str("\\t"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            c if c == q => {
                out.push('\\');
                out.push(c);
            }
            c if (c as u32) < 0x20 || c as u32 == 0x7f => {
                let _ = write!(out, "\\x{:02x}", c as u32);
            }
            c if (0x80..0xa0).contains(&(c as u32)) => {
                let _ = write!(out, "\\x{:02x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out.push(q);
    out
}

/// The shortest round-trip digits of a finite, non-zero `|x|` and its decimal
/// exponent (`d.ddd × 10^exp`), from Rust's `{:e}` (which is shortest).
fn digits_exp(x: f64) -> (String, i32) {
    let s = format!("{:e}", x.abs());
    let (mant, exp) = s.split_once('e').unwrap_or((s.as_str(), "0"));
    let digits: String = mant.chars().filter(|c| *c != '.').collect();
    (digits, exp.parse().unwrap_or(0))
}

/// Python's `repr(float)`: the shortest round-trip digits, fixed notation for
/// a decimal exponent in `-4..16`, else `d.ddde±XX`; `nan`, `inf`, `-inf`.
pub fn float_repr(x: f64) -> String {
    if x.is_nan() {
        return "nan".into();
    }
    if x.is_infinite() {
        return if x > 0.0 { "inf" } else { "-inf" }.into();
    }
    let sign = if x.is_sign_negative() { "-" } else { "" };
    if x == 0.0 {
        return format!("{sign}0.0");
    }
    let (digits, exp) = digits_exp(x);
    if (-4..16).contains(&exp) {
        format!("{sign}{}", fixed_from_digits(&digits, exp, true))
    } else {
        format!("{sign}{}", sci_from_digits(&digits, exp, 2))
    }
}

/// `digits × 10^(exp - len + 1)` in fixed notation; `point_zero`: an integral
/// value ends in `.0` (Python) rather than nothing (jq).
fn fixed_from_digits(digits: &str, exp: i32, point_zero: bool) -> String {
    let n = digits.len() as i32;
    if exp < 0 {
        return format!("0.{}{digits}", "0".repeat((-exp - 1) as usize));
    }
    let int_len = exp + 1;
    if n <= int_len {
        let int = format!("{digits}{}", "0".repeat((int_len - n) as usize));
        if point_zero {
            format!("{int}.0")
        } else {
            int
        }
    } else {
        let (a, b) = digits.split_at(int_len as usize);
        format!("{a}.{b}")
    }
}

/// `d.ddd e±XX` with at least `min_exp_digits` exponent digits.
fn sci_from_digits(digits: &str, exp: i32, min_exp_digits: usize) -> String {
    let (first, rest) = digits.split_at(1);
    let mant = if rest.is_empty() {
        first.to_owned()
    } else {
        format!("{first}.{rest}")
    };
    let sign = if exp < 0 { '-' } else { '+' };
    format!(
        "{mant}e{sign}{:0width$}",
        exp.unsigned_abs(),
        width = min_exp_digits
    )
}

/// Python's `format(x, ".{prec}f")` (`f"{x:.2f}"`): correctly rounded, `nan`,
/// `inf`, `-inf`.
pub fn fixed(x: f64, prec: usize) -> String {
    if x.is_nan() {
        return "nan".into();
    }
    if x.is_infinite() {
        return if x > 0.0 { "inf" } else { "-inf" }.into();
    }
    format!("{x:.prec$}")
}

/// Python's `format(x, ".{prec}e")` (`f"{x:.2e}"`): `1.23e-05`, never `1.23e-5`.
pub fn exp(x: f64, prec: usize) -> String {
    if x.is_nan() {
        return "nan".into();
    }
    if x.is_infinite() {
        return if x > 0.0 { "inf" } else { "-inf" }.into();
    }
    let s = format!("{x:.prec$e}");
    let (mant, e) = s.split_once('e').unwrap_or((s.as_str(), "0"));
    let e: i32 = e.parse().unwrap_or(0);
    let sign = if e < 0 { '-' } else { '+' };
    format!("{mant}e{sign}{:02}", e.unsigned_abs())
}

/// Python's `format(v, ".{prec}f")` of a number value; `None` for a value
/// that is not one (Python would raise).
pub fn fixed_py(v: &Py, prec: usize) -> Option<String> {
    match v {
        Py::Int(_) | Py::Float(_) | Py::Bool(_) => v.num().map(|x| fixed(x, prec)),
        _ => None,
    }
}

/// Right-align in `width` (Python's default for numbers, `{x:7.2f}`).
pub fn rjust(s: &str, width: usize) -> String {
    let n = s.chars().count();
    if n >= width {
        s.to_owned()
    } else {
        format!("{}{s}", " ".repeat(width - n))
    }
}

/// Left-align in `width` (Python's default for strings, `{s:28s}`).
pub fn ljust(s: &str, width: usize) -> String {
    let n = s.chars().count();
    if n >= width {
        s.to_owned()
    } else {
        format!("{s}{}", " ".repeat(width - n))
    }
}

/// The first `n` characters (`s[:n]`).
pub fn head_chars(s: &str, n: usize) -> String {
    s.chars().take(n).collect()
}

/// The last `n` characters (`s[-n:]`).
pub fn tail_chars(s: &str, n: usize) -> String {
    let len = s.chars().count();
    s.chars().skip(len.saturating_sub(n)).collect()
}

/// `statistics.median` of numbers (as `f64`; the middle of an odd count, the
/// mean of the two middle values of an even one). `None` when empty.
pub fn median(xs: &[f64]) -> Option<f64> {
    if xs.is_empty() {
        return None;
    }
    let mut v = xs.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = v.len();
    Some(if n % 2 == 1 {
        v[n / 2]
    } else {
        (v[n / 2 - 1] + v[n / 2]) / 2.0
    })
}

/// `statistics.median` of Python numbers, keeping an int middle value an int
/// (an odd count of ints returns the int itself).
pub fn median_py(xs: &[Py]) -> Option<Py> {
    if xs.is_empty() {
        return None;
    }
    let mut v: Vec<&Py> = xs.iter().collect();
    v.sort_by(|a, b| {
        a.num()
            .unwrap_or(f64::NAN)
            .partial_cmp(&b.num().unwrap_or(f64::NAN))
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let n = v.len();
    if n % 2 == 1 {
        Some(v[n / 2].clone())
    } else {
        v[n / 2 - 1].add(v[n / 2])?.div(&Py::int(2))
    }
}

// ------------------------------------------------------------------ json

/// `json.dumps(v)`: `, ` and `: ` separators, `ensure_ascii`, `NaN` /
/// `Infinity` for the special floats.
pub fn dumps(v: &Py) -> String {
    let mut out = String::new();
    write_json(&mut out, v, None, 0);
    out
}

/// `json.dumps(v, indent=n)` (and `json.dump`): one item per line, `,` and
/// `: ` separators, no trailing newline.
pub fn dumps_indent(v: &Py, indent: usize) -> String {
    let mut out = String::new();
    write_json(&mut out, v, Some(indent), 0);
    out
}

fn write_json(out: &mut String, v: &Py, indent: Option<usize>, level: usize) {
    match v {
        Py::None => out.push_str("null"),
        Py::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        Py::Int(s) => out.push_str(s),
        Py::Float(f) => out.push_str(&json_float(*f)),
        Py::Str(s) => json_str(out, s),
        Py::List(items) => {
            if items.is_empty() {
                out.push_str("[]");
                return;
            }
            out.push('[');
            for (i, item) in items.iter().enumerate() {
                sep(out, indent, level + 1, i == 0);
                write_json(out, item, indent, level + 1);
            }
            close(out, indent, level);
            out.push(']');
        }
        Py::Dict(items) => {
            if items.is_empty() {
                out.push_str("{}");
                return;
            }
            out.push('{');
            for (i, (k, item)) in items.iter().enumerate() {
                sep(out, indent, level + 1, i == 0);
                json_str(out, k);
                out.push_str(": ");
                write_json(out, item, indent, level + 1);
            }
            close(out, indent, level);
            out.push('}');
        }
    }
}

fn sep(out: &mut String, indent: Option<usize>, level: usize, first: bool) {
    match indent {
        None => {
            if !first {
                out.push_str(", ");
            }
        }
        Some(n) => {
            if !first {
                out.push(',');
            }
            out.push('\n');
            out.push_str(&" ".repeat(n * level));
        }
    }
}

fn close(out: &mut String, indent: Option<usize>, level: usize) {
    if let Some(n) = indent {
        out.push('\n');
        out.push_str(&" ".repeat(n * level));
    }
}

/// `float.__repr__` as `json` writes it (`NaN`, `Infinity`, `-Infinity`).
fn json_float(f: f64) -> String {
    if f.is_nan() {
        "NaN".into()
    } else if f.is_infinite() {
        if f > 0.0 { "Infinity" } else { "-Infinity" }.into()
    } else {
        float_repr(f)
    }
}

/// A JSON string with `ensure_ascii` (`\uXXXX`, surrogate pairs above the BMP).
fn json_str(out: &mut String, s: &str) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{08}' => out.push_str("\\b"),
            '\u{0c}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 || (c as u32) > 0x7e => {
                let mut buf = [0u16; 2];
                for unit in c.encode_utf16(&mut buf) {
                    let _ = write!(out, "\\u{unit:04x}");
                }
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

/// `json.loads`: JSON plus Python's `NaN`, `Infinity`, `-Infinity`.
pub fn loads(text: &str) -> Result<Py, String> {
    let mut p = Parser {
        b: text.as_bytes(),
        i: 0,
    };
    p.ws();
    let v = p.value()?;
    p.ws();
    if p.i != p.b.len() {
        return Err(format!("extra data at byte {}", p.i));
    }
    Ok(v)
}

/// Read a JSON file (`json.load(open(p))`).
pub fn load(path: &std::path::Path) -> Result<Py, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    loads(&text).map_err(|e| format!("{}: {e}", path.display()))
}

struct Parser<'a> {
    b: &'a [u8],
    i: usize,
}

impl Parser<'_> {
    fn ws(&mut self) {
        while self.i < self.b.len() && matches!(self.b[self.i], b' ' | b'\t' | b'\n' | b'\r') {
            self.i += 1;
        }
    }

    fn eat(&mut self, lit: &str) -> bool {
        if self.b[self.i..].starts_with(lit.as_bytes()) {
            self.i += lit.len();
            true
        } else {
            false
        }
    }

    fn value(&mut self) -> Result<Py, String> {
        let Some(&c) = self.b.get(self.i) else {
            return Err("unexpected end of JSON".into());
        };
        match c {
            b'{' => self.object(),
            b'[' => self.array(),
            b'"' => self.string().map(Py::Str),
            b't' if self.eat("true") => Ok(Py::Bool(true)),
            b'f' if self.eat("false") => Ok(Py::Bool(false)),
            b'n' if self.eat("null") => Ok(Py::None),
            b'N' if self.eat("NaN") => Ok(Py::Float(f64::NAN)),
            b'I' if self.eat("Infinity") => Ok(Py::Float(f64::INFINITY)),
            b'-' if self.b[self.i..].starts_with(b"-Infinity") => {
                self.i += 9;
                Ok(Py::Float(f64::NEG_INFINITY))
            }
            b'-' | b'0'..=b'9' => self.number(),
            _ => Err(format!("unexpected byte at {}", self.i)),
        }
    }

    fn object(&mut self) -> Result<Py, String> {
        self.i += 1;
        let mut d = Py::dict();
        self.ws();
        if self.eat("}") {
            return Ok(d);
        }
        loop {
            self.ws();
            if self.b.get(self.i) != Some(&b'"') {
                return Err(format!("expected a key at byte {}", self.i));
            }
            let k = self.string()?;
            self.ws();
            if !self.eat(":") {
                return Err(format!("expected ':' at byte {}", self.i));
            }
            self.ws();
            let v = self.value()?;
            d.set(&k, v);
            self.ws();
            if self.eat(",") {
                continue;
            }
            if self.eat("}") {
                return Ok(d);
            }
            return Err(format!("expected ',' or '}}' at byte {}", self.i));
        }
    }

    fn array(&mut self) -> Result<Py, String> {
        self.i += 1;
        let mut v = Vec::new();
        self.ws();
        if self.eat("]") {
            return Ok(Py::List(v));
        }
        loop {
            self.ws();
            v.push(self.value()?);
            self.ws();
            if self.eat(",") {
                continue;
            }
            if self.eat("]") {
                return Ok(Py::List(v));
            }
            return Err(format!("expected ',' or ']' at byte {}", self.i));
        }
    }

    fn hex4(&mut self) -> Result<u32, String> {
        let h = self
            .b
            .get(self.i..self.i + 4)
            .and_then(|s| std::str::from_utf8(s).ok())
            .and_then(|s| u32::from_str_radix(s, 16).ok())
            .ok_or_else(|| format!("bad \\u escape at byte {}", self.i))?;
        self.i += 4;
        Ok(h)
    }

    fn string(&mut self) -> Result<String, String> {
        self.i += 1;
        let mut out: Vec<u8> = Vec::new();
        loop {
            let Some(&c) = self.b.get(self.i) else {
                return Err("unterminated string".into());
            };
            self.i += 1;
            match c {
                b'"' => break,
                b'\\' => {
                    let Some(&e) = self.b.get(self.i) else {
                        return Err("unterminated escape".into());
                    };
                    self.i += 1;
                    let ch = match e {
                        b'"' => '"',
                        b'\\' => '\\',
                        b'/' => '/',
                        b'b' => '\u{08}',
                        b'f' => '\u{0c}',
                        b'n' => '\n',
                        b'r' => '\r',
                        b't' => '\t',
                        b'u' => {
                            let hi = self.hex4()?;
                            let code = if (0xd800..0xdc00).contains(&hi)
                                && self.b[self.i..].starts_with(b"\\u")
                            {
                                let save = self.i;
                                self.i += 2;
                                let lo = self.hex4()?;
                                if (0xdc00..0xe000).contains(&lo) {
                                    0x10000 + ((hi - 0xd800) << 10) + (lo - 0xdc00)
                                } else {
                                    self.i = save;
                                    hi
                                }
                            } else {
                                hi
                            };
                            // A lone surrogate: Python keeps it in a `str`; a
                            // Rust `String` cannot hold one, so it is refused.
                            char::from_u32(code)
                                .ok_or_else(|| format!("a lone surrogate \\u{code:04x}"))?
                        }
                        _ => return Err(format!("bad escape at byte {}", self.i)),
                    };
                    let mut buf = [0u8; 4];
                    out.extend_from_slice(ch.encode_utf8(&mut buf).as_bytes());
                }
                // `json.loads` (strict) refuses a raw control character.
                c if c < 0x20 => {
                    return Err(format!(
                        "a control character in a string at byte {}",
                        self.i - 1
                    ))
                }
                c => out.push(c),
            }
        }
        String::from_utf8(out).map_err(|e| e.to_string())
    }

    /// JSON's number grammar, as `json.loads` reads it:
    /// `-?(0|[1-9][0-9]*)(\.[0-9]+)?([eE][+-]?[0-9]+)?`; an int without a
    /// fraction or exponent, else a float.
    fn number(&mut self) -> Result<Py, String> {
        let start = self.i;
        let digits = |p: &mut Self| {
            let s = p.i;
            while p.b.get(p.i).is_some_and(u8::is_ascii_digit) {
                p.i += 1;
            }
            p.i - s
        };
        if self.b.get(self.i) == Some(&b'-') {
            self.i += 1;
        }
        let int_start = self.i;
        let n = digits(self);
        if n == 0 || (n > 1 && self.b[int_start] == b'0') {
            return Err(format!("bad number at byte {start}"));
        }
        let mut float = false;
        if self.b.get(self.i) == Some(&b'.') {
            self.i += 1;
            if digits(self) == 0 {
                return Err(format!("bad number at byte {start}"));
            }
            float = true;
        }
        if matches!(self.b.get(self.i), Some(b'e' | b'E')) {
            self.i += 1;
            if matches!(self.b.get(self.i), Some(b'+' | b'-')) {
                self.i += 1;
            }
            if digits(self) == 0 {
                return Err(format!("bad number at byte {start}"));
            }
            float = true;
        }
        let s = std::str::from_utf8(&self.b[start..self.i]).map_err(|e| e.to_string())?;
        if float {
            return s
                .parse::<f64>()
                .map(Py::Float)
                .map_err(|_| format!("bad number `{s}`"));
        }
        Ok(Py::Int(if s == "-0" {
            "0".to_owned()
        } else {
            s.to_owned()
        }))
    }
}

// ------------------------------------------------------------------ jq

/// jq's spelling of a number (`jvp_dtoa_fmt`): shortest round-trip digits,
/// fixed notation unless the decimal point is 4 or more places left of the
/// first digit or more than 15 places right of the last, then `de±XX`;
/// integral values without `.0`.
///
/// jq 1.6 spells every number so; jq 1.7 keeps a number read from its input
/// as the input spelled it (`1.500`) until arithmetic touches it. fv-lab
/// prints the canonical form (docs/adr/0019-fv-lab.md §13).
pub fn jq_num(x: f64) -> String {
    if x.is_nan() {
        return "null".into();
    }
    if x.is_infinite() {
        return if x > 0.0 {
            "1.7976931348623157e+308"
        } else {
            "-1.7976931348623157e+308"
        }
        .into();
    }
    let sign = if x.is_sign_negative() { "-" } else { "" };
    if x == 0.0 {
        return format!("{sign}0");
    }
    let (digits, exp) = digits_exp(x);
    let decpt = exp + 1;
    let n = digits.len() as i32;
    if decpt <= -4 || decpt > n + 15 {
        format!("{sign}{}", sci_from_digits(&digits, exp, 2))
    } else {
        format!("{sign}{}", fixed_from_digits(&digits, exp, false))
    }
}

/// jq's `tojson` / string interpolation of a value (compact, keys in order).
pub fn jq_text(v: &Py) -> String {
    match v {
        Py::Str(s) => s.clone(),
        other => jq_json(other),
    }
}

/// jq's compact JSON of a value.
pub fn jq_json(v: &Py) -> String {
    match v {
        Py::None => "null".into(),
        Py::Bool(b) => b.to_string(),
        Py::Int(_) | Py::Float(_) => jq_num(v.num().unwrap_or(f64::NAN)),
        Py::Str(s) => {
            let mut out = String::new();
            jq_str(&mut out, s);
            out
        }
        Py::List(items) => {
            let parts: Vec<String> = items.iter().map(jq_json).collect();
            format!("[{}]", parts.join(","))
        }
        Py::Dict(items) => {
            let parts: Vec<String> = items
                .iter()
                .map(|(k, v)| {
                    let mut key = String::new();
                    jq_str(&mut key, k);
                    format!("{key}:{}", jq_json(v))
                })
                .collect();
            format!("{{{}}}", parts.join(","))
        }
    }
}

fn jq_str(out: &mut String, s: &str) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{08}' => out.push_str("\\b"),
            '\u{0c}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 || c as u32 == 0x7f => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn float_repr_is_pythons() {
        for (x, want) in [
            (1.0, "1.0"),
            (0.1, "0.1"),
            (1e-5, "1e-05"),
            (0.0001, "0.0001"),
            (1e16, "1e+16"),
            (1e15, "1000000000000000.0"),
            (123.456, "123.456"),
            (-0.0, "-0.0"),
            (1.5e-7, "1.5e-07"),
            (2.5e300, "2.5e+300"),
            (0.30000000000000004, "0.30000000000000004"),
            (1234567890123456.7, "1234567890123456.8"),
        ] {
            assert_eq!(float_repr(x), want, "{x}");
        }
        assert_eq!(float_repr(f64::NAN), "nan");
        assert_eq!(fixed(f64::NAN, 2), "nan");
        assert_eq!(fixed(-f64::INFINITY, 2), "-inf");
    }

    #[test]
    fn fixed_and_exp_are_pythons() {
        assert_eq!(fixed(2.675, 2), "2.67"); // the binary value is below the tie
        assert_eq!(fixed(0.125, 2), "0.12"); // an exact tie rounds to even
        assert_eq!(fixed(0.375, 2), "0.38");
        assert_eq!(fixed(2.5, 0), "2");
        assert_eq!(fixed(3.5, 0), "4");
        assert_eq!(fixed(1e22, 1), "10000000000000000000000.0");
        assert_eq!(exp(1.234e-5, 2), "1.23e-05");
        assert_eq!(exp(0.0, 2), "0.00e+00");
        assert_eq!(exp(4.76e-1, 2), "4.76e-01");
        assert_eq!(exp(1.5e123, 1), "1.5e+123");
    }

    #[test]
    fn json_round_trips_as_python_writes_it() {
        // A `\u` escape and a raw non-ASCII character read alike; both are written escaped.
        let v =
            loads("{\"b\": 1, \"a\": [1.0, 1e-05, \"x\\u00e9\", null, true], \"b\": 2, \"c\": {}}")
                .unwrap();
        assert_eq!(
            v,
            loads("{\"b\": 2, \"a\": [1.0, 1e-05, \"x\u{e9}\", null, true], \"c\": {}}").unwrap()
        );
        assert_eq!(
            dumps(&v),
            "{\"b\": 2, \"a\": [1.0, 1e-05, \"x\\u00e9\", null, true], \"c\": {}}"
        );
        assert_eq!(
            dumps_indent(&v, 1),
            "{\n \"b\": 2,\n \"a\": [\n  1.0,\n  1e-05,\n  \"x\\u00e9\",\n  null,\n  true\n ],\n \"c\": {}\n}"
        );
        assert_eq!(
            repr(&v.at("a").clone()),
            "[1.0, 1e-05, 'x\u{e9}', None, True]"
        );
        assert_eq!(loads("-0").unwrap(), Py::Int("0".into()));
        assert_eq!(
            dumps(&loads("[NaN, -Infinity]").unwrap()),
            "[NaN, -Infinity]"
        );
        assert!(loads("{\"a\": 1} x").is_err());
        // What `json.loads` refuses is refused.
        for bad in [
            "01",
            "-01",
            "1.",
            ".5",
            "1e",
            "1e+",
            "-",
            "\"a\u{1}b\"",
            "\"\\ud800\"",
            "\"\\udc00x\"",
        ] {
            assert!(loads(bad).is_err(), "{bad:?}");
        }
        assert_eq!(loads("-0.0").unwrap(), Py::Float(-0.0));
        assert_eq!(loads("1E+2").unwrap(), Py::Float(100.0));
        assert_eq!(
            loads("\"\\ud83d\\ude00\"").unwrap(),
            Py::Str("\u{1f600}".into())
        );
    }

    #[test]
    fn python_arithmetic_keeps_ints() {
        let (a, b) = (Py::int(3), Py::int(4));
        assert_eq!(a.add(&b), Some(Py::int(7)));
        assert_eq!(a.div(&b), Some(Py::Float(0.75)));
        assert_eq!(a.add(&Py::Float(0.5)), Some(Py::Float(3.5)));
        assert_eq!(
            median_py(&[Py::int(3), Py::int(1), Py::int(2)]),
            Some(Py::int(2))
        );
        assert_eq!(median_py(&[Py::int(1), Py::int(2)]), Some(Py::Float(1.5)));
        assert_eq!(median(&[4.0, 1.0, 3.0, 2.0]), Some(2.5));
    }

    #[test]
    fn str_repr_quotes_as_python() {
        assert_eq!(str_repr("a'b"), "\"a'b\"");
        assert_eq!(str_repr("a'b\""), "'a\\'b\"'");
        assert_eq!(str_repr("t\tn\n"), "'t\\tn\\n'");
        assert_eq!(
            tuple_repr(&[Py::Str("v2".into()), Py::Str("h3-demo".into())]),
            "('v2', 'h3-demo')"
        );
    }

    #[test]
    fn jq_numbers_are_canonical() {
        for (x, want) in [
            (1e-5, "1e-05"),
            (0.0001, "0.0001"),
            (0.00012345, "0.00012345"),
            (1e15, "1000000000000000"),
            (1e16, "1e+16"),
            (123456789012345678.0, "123456789012345680"),
            (12345678901234567000.0, "12345678901234567000"),
            (1.25e18, "1.25e+18"),
            (3.0, "3"),
            (2.5e-7, "2.5e-07"),
            (-12.5, "-12.5"),
            (-0.0, "-0"),
            (0.0, "0"),
        ] {
            assert_eq!(jq_num(x), want, "{x}");
        }
        let v = loads(r#"{"a": [1, "x"], "b": null}"#).unwrap();
        assert_eq!(jq_json(&v), r#"{"a":[1,"x"],"b":null}"#);
    }
}
