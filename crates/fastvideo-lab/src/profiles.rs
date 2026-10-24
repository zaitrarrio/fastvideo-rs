//! The profile-comment measurement lint (plan D2 acceptance: "comments in
//! `profiles/` no longer carry numbers").
//!
//! A technique profile (`profiles/<model>/<name>.toml`) says *what* is on and
//! *why*; the numbers that justify it live in the variant's gold file
//! (`lab/gold/<variant>.toml`) or in `docs/perf`, where they carry a date, a
//! GPU and a source. A comment that quotes a measurement goes stale the day
//! after the next run, so the lint flags, inside comments only:
//!
//! - a number or a range followed by a measurement unit: `19.04 s`,
//!   `0.49-0.55 s`, `6.3 ms`, `4 min`, `2 h`, `51 dB`, `2.3 GiB`, `38 GB`,
//!   `1.2 TiB`, `15.2 fps`, `3.1 TFLOPS`, `-25.6 %`;
//! - a speed-up: `1.53x`, `1.53X`, `1.53×`, `1.53 X`, `1.53 ×` and, with a
//!   decimal point, `1.53 x` (`832x480` is a size, not a speed-up);
//! - a metric name up to three words before a number: `LPIPS 0.510`,
//!   `PSNR: 12.7`, `LPIPS vs base 0.685`, `rel-L2 3e-3`;
//! - a price: `$2.09`.
//!
//! Settings (`tau 1.0`, `sparsity 0.925`, `shifts 12 / 3`, `4 forwards`),
//! scale factors of a module (`4x upsampler`, `2x spatial`), clip lengths
//! (`a 5 s clip`, `10 s of video`), card classes (`a 96 GB card`), dates,
//! ranges without a unit (`steps 0-2`), commit hashes, `sm_120`, `768p` and
//! TOML values outside comments are not measurements and pass.

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

const UNITS: [&str; 26] = [
    "s", "ms", "us", "\u{b5}s", "ns", "sec", "secs", "seconds", "min", "mins", "h", "hr", "hrs",
    "dB", "KiB", "MiB", "GiB", "TiB", "KB", "MB", "GB", "TB", "fps", "GFLOPS", "TFLOPS", "PFLOPS",
];
const TIME_UNITS: [&str; 4] = ["s", "sec", "secs", "seconds"];
const MEMORY_UNITS: [&str; 8] = ["KiB", "MiB", "GiB", "TiB", "KB", "MB", "GB", "TB"];
const METRICS: [&str; 7] = [
    "lpips",
    "psnr",
    "ssim",
    "sharpness",
    "jitter",
    "speedup",
    "rel-l2",
];
/// What follows `<n>x` when it is a module's scale factor, not a speed-up.
const SCALE_NOUNS: [&str; 14] = [
    "upsampler",
    "upsampling",
    "upscale",
    "upscaler",
    "upscaling",
    "downsample",
    "downsampling",
    "downscale",
    "compression",
    "spatial",
    "temporal",
    "latent",
    "super-resolution",
    "tiles",
];
/// What follows `<n> s` when it is a clip length, not a measured time.
const CLIP_NOUNS: [&str; 5] = [" clip", " clips", " of video", " video", " rollout"];
/// How many words back a metric name may stand from its number.
const METRIC_WINDOW: usize = 3;

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

/// The alphabetic run starting at `chars[i]`, and the index after it.
fn alpha_run(chars: &[char], i: usize) -> (String, usize) {
    let mut j = i;
    while j < chars.len() && chars[j].is_alphabetic() {
        j += 1;
    }
    (chars[i..j].iter().collect(), j)
}

/// The word starting at `chars[i]` (letters, digits, `-`), lowercase.
fn word_at(chars: &[char], i: usize) -> String {
    let mut j = i;
    while j < chars.len() && (chars[j].is_alphanumeric() || chars[j] == '-') {
        j += 1;
    }
    chars[i..j].iter().collect::<String>().to_lowercase()
}

/// A metric name among the [`METRIC_WINDOW`] words before `end`, without
/// crossing `,`, `;`, a bracket or the end of a sentence.
fn metric_before(chars: &[char], end: usize) -> Option<&'static str> {
    let mut j = end;
    for _ in 0..METRIC_WINDOW {
        while j > 0 && matches!(chars[j - 1], ' ' | ':' | '=' | '/') {
            j -= 1;
        }
        let stop = j;
        while j > 0 && (chars[j - 1].is_alphanumeric() || chars[j - 1] == '-') {
            j -= 1;
        }
        if j == stop {
            return None;
        }
        let w: String = chars[j..stop].iter().collect::<String>().to_lowercase();
        if let Some(m) = METRICS.iter().find(|m| **m == w) {
            return Some(m);
        }
    }
    None
}

/// The end of the number starting at `i`: digits, an optional fraction and
/// an optional exponent (`3e-3`, `1.5e4`).
fn number_end(chars: &[char], mut i: usize) -> usize {
    while i < chars.len() && chars[i].is_ascii_digit() {
        i += 1;
    }
    if i + 1 < chars.len() && chars[i] == '.' && chars[i + 1].is_ascii_digit() {
        i += 1;
        while i < chars.len() && chars[i].is_ascii_digit() {
            i += 1;
        }
    }
    if i < chars.len() && (chars[i] == 'e' || chars[i] == 'E') {
        let mut k = i + 1;
        if k < chars.len() && (chars[k] == '-' || chars[k] == '+') {
            k += 1;
        }
        if k < chars.len() && chars[k].is_ascii_digit() {
            while k < chars.len() && chars[k].is_ascii_digit() {
                k += 1;
            }
            let ends_word = k >= chars.len() || !is_word(chars[k]);
            if ends_word {
                return k;
            }
        }
    }
    i
}

/// The unit, speed-up or percent after a number ending at `i`, as the text
/// to report after the number, or `None`.
fn suffix_claim(chars: &[char], i: usize, number: &str) -> Option<String> {
    let spaced = i < chars.len() && chars[i] == ' ';
    let j = if spaced { i + 1 } else { i };
    if j >= chars.len() {
        return None;
    }
    if chars[j] == '%' {
        return Some(" %".to_owned());
    }
    let rest_after = |k: usize| -> String { chars[k.min(chars.len())..].iter().collect() };
    let scale_factor = |k: usize| -> bool {
        let rest = rest_after(k);
        rest.strip_prefix(' ')
            .map(|r| {
                let w = word_at(&r.chars().collect::<Vec<_>>(), 0);
                SCALE_NOUNS.contains(&w.as_str())
            })
            .unwrap_or(false)
    };
    if chars[j] == '\u{d7}' {
        let after = j + 1;
        if after < chars.len() && is_word(chars[after]) || scale_factor(after) {
            return None;
        }
        return Some("x".to_owned());
    }
    let (run, after) = alpha_run(chars, j);
    if run.is_empty() || (after < chars.len() && is_word(chars[after])) {
        return None;
    }
    if run == "x" || run == "X" {
        // `1.5x`, `1.5X`, `1.5 X`; a spaced lowercase `x` only with a decimal
        // point (`4 x 3` is a size).
        let speedup = !spaced || run == "X" || number.contains('.');
        if speedup && !scale_factor(after) {
            return Some("x".to_owned());
        }
        return None;
    }
    if UNITS.contains(&run.as_str()) {
        let rest = rest_after(after);
        // A card class (`a 96 GB card`, `80 GB-class`) names hardware.
        if MEMORY_UNITS.contains(&run.as_str())
            && (rest.starts_with(" card") || rest.starts_with("-class"))
        {
            return None;
        }
        // A clip length (`a 5 s clip`, `10 s of video`) is the workload.
        if TIME_UNITS.contains(&run.as_str()) && CLIP_NOUNS.iter().any(|n| rest.starts_with(n)) {
            return None;
        }
        return Some(format!(" {run}"));
    }
    None
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
        // A number must not continue a word (`sm_120`, `b299214`, `h3`);
        // `+2.3` and `-25.6` are numbers.
        let prev = if i > 0 { Some(chars[i - 1]) } else { None };
        if prev.is_some_and(|p| is_word(p) || p == '.') {
            while i < chars.len() && (chars[i].is_ascii_alphanumeric() || chars[i] == '.') {
                i += 1;
            }
            continue;
        }
        let start = i;
        i = number_end(&chars, i);
        // A range or a date: `0-2`, `48-49`, `0.49-0.55`, `2026-10-14`. It is
        // a measurement only with a unit or a metric name.
        while i + 1 < chars.len() && chars[i] == '-' && chars[i + 1].is_ascii_digit() {
            i = number_end(&chars, i + 1);
        }
        let number: String = chars[start..i].iter().collect();
        // A price.
        let mut k = start;
        while k > 0 && chars[k - 1] == ' ' {
            k -= 1;
        }
        if k > 0 && chars[k - 1] == '$' {
            out.push(format!("${number}"));
            continue;
        }
        if let Some(suffix) = suffix_claim(&chars, i, &number) {
            out.push(format!("{number}{suffix}"));
            continue;
        }
        if let Some(m) = metric_before(&chars, start) {
            out.push(format!("{m} {number}"));
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
            ["jitter 0.43", "ssim 0.98", "rel-l2 3e-3"]
        );
        assert_eq!(texts(" (1344x768, 5 s, seed 0)"), ["5 s"]);
    }

    #[test]
    fn finds_ranges_with_a_unit() {
        assert_eq!(texts(" upsample 0.49-0.55 s per run"), ["0.49-0.55 s"]);
        assert_eq!(texts(" load 117-139 s"), ["117-139 s"]);
        assert_eq!(texts(" peak 26-33 GB each"), ["26-33 GB"]);
        assert_eq!(texts(" saves 10-12 % end to end"), ["10-12 %"]);
        assert_eq!(texts(" LPIPS 0.11-0.20"), ["lpips 0.11-0.20"]);
    }

    #[test]
    fn finds_every_speedup_spelling() {
        assert_eq!(texts(" 1.53x"), ["1.53x"]);
        assert_eq!(texts(" 1.53X faster"), ["1.53x"]);
        assert_eq!(texts(" 1.53\u{d7} faster"), ["1.53x"]);
        assert_eq!(texts(" 1.53 X faster"), ["1.53x"]);
        assert_eq!(texts(" 1.53 \u{d7} faster"), ["1.53x"]);
        assert_eq!(texts(" 1.53 x faster"), ["1.53x"]);
        assert_eq!(texts(" 2x."), ["2x"]);
    }

    #[test]
    fn finds_more_units() {
        assert_eq!(texts(" took 4 min"), ["4 min"]);
        assert_eq!(texts(" a 2 h run"), ["2 h"]);
        assert_eq!(
            texts(" 35 us per call, 12 \u{b5}s per tile"),
            ["35 us", "12 \u{b5}s"]
        );
        assert_eq!(texts(" 512 KiB, 1.2 TiB"), ["512 KiB", "1.2 TiB"]);
        assert_eq!(texts(" 3.1 TFLOPS"), ["3.1 TFLOPS"]);
    }

    #[test]
    fn finds_metrics_a_few_words_away() {
        assert_eq!(texts(" LPIPS vs base 0.685"), ["lpips 0.685"]);
        assert_eq!(texts(" PSNR / SSIM: 39.4"), ["ssim 39.4"]);
        // Not across a comma, and not further than three words.
        assert_eq!(texts(" LPIPS, then tau 1.0"), Vec::<String>::new());
        assert_eq!(
            texts(" LPIPS is not a gate here at tau 1.0"),
            Vec::<String>::new()
        );
    }

    #[test]
    fn ignores_scale_factors_and_clip_lengths() {
        for ok in [
            " the 2x spatial latent upsampler and its 4x upsampler",
            " a 2x compression, 8x temporal",
            " a 5 s clip, 10 s of video, 20 s rollout",
            " 4 x 3 grid of shots",
        ] {
            assert_eq!(texts(ok), Vec::<String>::new(), "{ok:?}");
        }
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
