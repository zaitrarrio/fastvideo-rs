//! The profile-comment measurement lint (plan D2 acceptance: "comments in
//! `profiles/` no longer carry numbers").
//!
//! A technique profile (`profiles/<model>/<name>.toml`) says *what* is on and
//! *why*; the numbers that justify it live in the variant's gold file
//! (`lab/gold/<variant>.toml`) or in `docs/perf`, where they carry a date, a
//! GPU and a source. A comment that quotes a measurement goes stale the day
//! after the next run, so the lint flags, inside comments only:
//!
//! - a number followed by a measurement unit: `19.04 s`, `6.3 ms`, `51 dB`,
//!   `2.3 GiB`, `38 GB`, `15.2 fps`, `-25.6 %`;
//! - a speed-up: a number immediately followed by `x` and nothing
//!   alphanumeric after it (`1.53x`; `832x480` is a size, not a speed-up);
//! - a metric name followed by a number: `LPIPS 0.510`, `PSNR 12.7`,
//!   `sharpness 0.99`, `jitter 0.43`, `SSIM 0.98`;
//! - a price: `$2.09`.
//!
//! Settings (`tau 1.0`, `sparsity 0.925`, `shifts 12 / 3`, `4 forwards`),
//! card classes (`a 96 GB card`), dates, commit hashes, `sm_120`, `768p`
//! and TOML values outside comments are not measurements and pass.

/// One measurement-looking number found in a comment.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Claim {
    /// 1-based line number.
    pub line: usize,
    /// The matched text (`19.04 s`, `LPIPS 0.510`).
    pub text: String,
    /// The whole comment the match is in, trimmed.
    pub comment: String,
}

const UNITS: [&str; 12] = [
    "s", "ms", "sec", "secs", "seconds", "dB", "GiB", "GB", "MiB", "MB", "fps", "TB",
];
const MEMORY_UNITS: [&str; 4] = ["GiB", "GB", "MiB", "MB"];
const METRICS: [&str; 7] = [
    "lpips",
    "psnr",
    "ssim",
    "sharpness",
    "jitter",
    "speedup",
    "rel-l2",
];

/// The comment part of one TOML line (after a `#` that is not inside a
/// string), without the `#`; `None` when the line has no comment.
pub fn comment_of(line: &str) -> Option<&str> {
    let mut in_str = false;
    let mut escaped = false;
    for (i, c) in line.char_indices() {
        if in_str {
            if escaped {
                escaped = false;
            } else if c == '\\' {
                escaped = true;
            } else if c == '"' {
                in_str = false;
            }
            continue;
        }
        match c {
            '"' => in_str = true,
            '#' => return Some(&line[i + 1..]),
            _ => {}
        }
    }
    None
}

fn is_word(c: char) -> bool {
    c.is_alphanumeric() || c == '_'
}

/// The ASCII-alphabetic run starting at `chars[i]`, and the index after it.
fn alpha_run(chars: &[char], i: usize) -> (String, usize) {
    let mut j = i;
    while j < chars.len() && chars[j].is_ascii_alphabetic() {
        j += 1;
    }
    (chars[i..j].iter().collect(), j)
}

/// The word (letters, digits, `-`) that ends at `end` (exclusive), skipping
/// trailing spaces, `:` and `=`.
fn word_before(chars: &[char], end: usize) -> String {
    let mut j = end;
    while j > 0 && (chars[j - 1] == ' ' || chars[j - 1] == ':' || chars[j - 1] == '=') {
        j -= 1;
    }
    let stop = j;
    while j > 0 && (chars[j - 1].is_ascii_alphanumeric() || chars[j - 1] == '-') {
        j -= 1;
    }
    chars[j..stop].iter().collect()
}

/// Measurement-looking numbers in one comment, as the matched text.
pub fn claims_in_comment(comment: &str) -> Vec<String> {
    let chars: Vec<char> = comment.chars().collect();
    let mut out = Vec::new();
    let mut i = 0;
    while i < chars.len() {
        if !chars[i].is_ascii_digit() {
            i += 1;
            continue;
        }
        // A number must not continue a word (`sm_120`, `b299214`) or a range
        // or date (the `10` of `2026-10-14`, the `2` of `0-2`); `+2.3` and
        // `-25.6` are numbers.
        let prev = if i > 0 { Some(chars[i - 1]) } else { None };
        let in_range = i > 1 && chars[i - 1] == '-' && chars[i - 2].is_ascii_digit();
        if in_range || prev.is_some_and(|p| is_word(p) || p == '.') {
            while i < chars.len() && (chars[i].is_ascii_digit() || chars[i] == '.') {
                i += 1;
            }
            continue;
        }
        let start = i;
        while i < chars.len() && chars[i].is_ascii_digit() {
            i += 1;
        }
        if i + 1 < chars.len() && chars[i] == '.' && chars[i + 1].is_ascii_digit() {
            i += 1;
            while i < chars.len() && chars[i].is_ascii_digit() {
                i += 1;
            }
        }
        let number: String = chars[start..i].iter().collect();
        // A date (`2026-10-14`) or a range (`0-2`, `48-49`): not a measurement.
        if i < chars.len()
            && chars[i] == '-'
            && i + 1 < chars.len()
            && chars[i + 1].is_ascii_digit()
        {
            continue;
        }
        // A price.
        let mut k = start;
        while k > 0 && chars[k - 1] == ' ' {
            k -= 1;
        }
        if k > 0 && chars[k - 1] == '$' {
            out.push(format!("${number}"));
            continue;
        }
        // A unit after the number (at most one space between).
        let mut j = i;
        if j < chars.len() && chars[j] == ' ' {
            j += 1;
        }
        if j < chars.len() && chars[j] == '%' {
            out.push(format!("{number} %"));
            continue;
        }
        let (run, after) = alpha_run(&chars, j);
        let followed_by_word = after < chars.len() && is_word(chars[after]);
        if !run.is_empty() && !followed_by_word {
            if run == "x" && j == i {
                out.push(format!("{number}x"));
                continue;
            }
            if UNITS.contains(&run.as_str()) {
                // A card class (`a 96 GB card`, `80 GB-class`) names hardware,
                // not a measurement.
                let rest: String = chars[after..].iter().collect();
                let card = MEMORY_UNITS.contains(&run.as_str())
                    && (rest.starts_with(" card") || rest.starts_with("-class"));
                if !card {
                    out.push(format!("{number} {run}"));
                }
                continue;
            }
        }
        // A metric name before the number.
        let w = word_before(&chars, start).to_ascii_lowercase();
        if METRICS.contains(&w.as_str()) {
            out.push(format!("{w} {number}"));
        }
    }
    out
}

/// Every measurement-looking number in the comments of a TOML text.
pub fn measurement_claims(toml_text: &str) -> Vec<Claim> {
    let mut out = Vec::new();
    for (idx, line) in toml_text.lines().enumerate() {
        let Some(comment) = comment_of(line) else {
            continue;
        };
        for text in claims_in_comment(comment) {
            out.push(Claim {
                line: idx + 1,
                text,
                comment: comment.trim().to_owned(),
            });
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn texts(comment: &str) -> Vec<String> {
        claims_in_comment(comment)
    }

    #[test]
    fn finds_measurements() {
        assert_eq!(
            texts(" Measured (RTX PRO 6000, b299214, 5 prompts, medians): 768p denoise 19.04 s (1.04x), LPIPS 0.510, PSNR 12.7 dB, sharpness 0.99 (min 0.96)"),
            ["19.04 s", "1.04x", "lpips 0.510", "12.7 dB", "sharpness 0.99"]
        );
        assert_eq!(texts(" +2.3 GiB beside the bf16 weights"), ["2.3 GiB"]);
        assert_eq!(
            texts(" keeps the 32 GiB plan under 30 GiB"),
            ["32 GiB", "30 GiB"]
        );
        assert_eq!(
            texts(" -25.6 % video VAE, 15.2 fps, $2.09/hr"),
            ["25.6 %", "15.2 fps", "$2.09"]
        );
        assert_eq!(
            texts(" jitter 0.43; SSIM: 0.98; rel-L2 3e-3"),
            ["jitter 0.43", "ssim 0.98", "rel-l2 3"]
        );
        assert_eq!(texts(" (1344x768, 5 s, seed 0)"), ["5 s"]);
    }

    #[test]
    fn ignores_settings_dates_and_sizes() {
        for ok in [
            " tau 1.0 / 1.25 / 1.5 on forwards 1-3",
            " sparsity 0.925 everywhere (recipe 0.9), MXFP8",
            " shifts 12 / 3, dense attention, 4 forwards (6 allowed)",
            " alpha 8 / rank 128; 50-point grid, 49 forwards",
            " default on since 2026-10-14, owner decision; b299214; sm_120",
            " 832x480 and 1344x768, 1080p, 124 frames, steps 0-2 and 6-7",
            " blocks 0-1 / 48-49 at 0.8; threshold 0.10, retain 5, cooldown 1",
            " the recipe's (0.8, 8); FASTVIDEO_BF16_ACT=1",
            " 4-step, 8-step, a 5-second clip",
            " the ltx25-*-sol cells on a 96 GB card; an 80 GB-class card",
        ] {
            assert_eq!(texts(ok), Vec::<String>::new(), "{ok:?}");
        }
    }

    #[test]
    fn comments_only() {
        assert_eq!(
            comment_of("tau = 1.0                 # SOL_ATTN_TAU"),
            Some(" SOL_ATTN_TAU")
        );
        assert_eq!(comment_of("description = \"a # b\""), None);
        assert_eq!(comment_of("description = \"a \\\" # b\" # c"), Some(" c"));
        assert_eq!(comment_of("sparsity = 0.95"), None);
        let text = "# Measured: denoise 19.04 s\nsparsity = 0.95 # the recipe's 0.9\n# LPIPS 0.5\n";
        let claims = measurement_claims(text);
        assert_eq!(claims.len(), 2);
        assert_eq!((claims[0].line, claims[0].text.as_str()), (1, "19.04 s"));
        assert_eq!((claims[1].line, claims[1].text.as_str()), (3, "lpips 0.5"));
    }
}
