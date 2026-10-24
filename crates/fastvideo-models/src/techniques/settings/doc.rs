//! `docs/scope.md`'s settings table, rendered from [`super::SETTINGS`].
//!
//! The committed doc carries the rendered table between [`DOC_BEGIN`] and
//! [`DOC_END`]; a test fails when the two differ and prints the table to
//! paste.

use super::table::{DefaultValue, Parse, Reader, Setting, SETTINGS};

/// The line that opens the generated region of `docs/scope.md`.
pub const DOC_BEGIN: &str =
    "<!-- BEGIN GENERATED: FASTVIDEO_* settings (crates/fastvideo-models/src/techniques/settings/table.rs; do not edit) -->";

/// The line that closes it.
pub const DOC_END: &str = "<!-- END GENERATED: FASTVIDEO_* settings -->";

fn cell(s: &str) -> String {
    s.replace('|', "\\|").replace('\n', " ")
}

fn reader(s: &Setting) -> &'static str {
    match s.reader {
        Reader::Layered => "env, then profile",
        Reader::Env => "env",
        Reader::Reserved => "reserved (not read)",
        Reader::Retired => "retired (not read)",
    }
}

fn kind(s: &Setting) -> String {
    let base = match s.parse {
        Parse::Flag => "flag",
        Parse::Usize => "usize",
        Parse::F32 => "f32",
        Parse::F64 => "f64",
        Parse::Keyword => "keyword",
        Parse::Text => "text",
    };
    if s.rule.is_empty() {
        base.to_string()
    } else {
        format!("{base}: {}", s.rule)
    }
}

fn default(s: &Setting) -> String {
    match s.default {
        DefaultValue::Unset => "unset".to_string(),
        DefaultValue::Value("") => "empty".to_string(),
        DefaultValue::Value(v) => format!("`{v}`"),
        DefaultValue::Computed(how) => how.to_string(),
    }
}

/// The generated region, markers included, newline-terminated.
pub fn render_markdown() -> String {
    let mut out = String::new();
    out.push_str(DOC_BEGIN);
    out.push_str("\n\n");
    out.push_str(
        "| Variable | Read by | Type and parse | Default | Read | Family | Seam | Effect |\n",
    );
    out.push_str("|---|---|---|---|---|---|---|---|\n");
    for s in SETTINGS {
        let row = [
            format!("`{}`", s.name),
            reader(s).to_string(),
            kind(s),
            default(s),
            s.timing.label().to_string(),
            s.family.label().to_string(),
            format!("`{}`", s.seam),
            s.doc.to_string(),
        ];
        out.push('|');
        for c in row {
            out.push(' ');
            out.push_str(&cell(&c));
            out.push_str(" |");
        }
        out.push('\n');
    }
    out.push('\n');
    out.push_str(DOC_END);
    out.push('\n');
    out
}
