//! Token counting: the DiT text encoder's limit (exact, with the model's
//! own tokenizer when fv-serve has it) and the LLM context budget (an
//! estimate, calibrated by the LLM server's `/tokenize` when it has one).
//!
//! - [`TokenCounter`]: one model's tokenizer. fv-serve builds them from the
//!   checkpoints' `tokenizer/tokenizer.json` (H3 with its `<d>` markers);
//!   `count` answers `None` while the tokenizer is not available (weights
//!   still streaming), and the limit falls back to characters.
//! - [`PromptLimit`]: a family's DiT prompt limit: tokens with a counter,
//!   characters without.
//! - [`estimate_tokens`]: the cheap LLM-side estimate (characters / 3.5).

use std::sync::Arc;

/// Characters per token of the LLM-side estimate (English prose on Qwen3,
/// Llama 3 and GPT tokenizers is about 3.5 to 4.5).
pub const CHARS_PER_TOKEN: f64 = 3.5;

/// Tokens of `s` at `chars_per_token` (rounded up; 0 for "").
pub fn estimate_tokens_at(s: &str, chars_per_token: f64) -> usize {
    let n = s.chars().count();
    if n == 0 {
        return 0;
    }
    (n as f64 / chars_per_token.max(0.5)).ceil() as usize
}

/// [`estimate_tokens_at`] with [`CHARS_PER_TOKEN`].
pub fn estimate_tokens(s: &str) -> usize {
    estimate_tokens_at(s, CHARS_PER_TOKEN)
}

/// One text encoder's tokenizer.
pub trait TokenCounter: Send + Sync + 'static {
    /// Tokens the encoder sees for `text` (special tokens it adds
    /// included); `None` when the tokenizer is not available right now.
    fn count(&self, text: &str) -> Option<usize>;
    /// For logs and `session_info` (a path or a name).
    fn describe(&self) -> String;
}

/// How a prompt was measured.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Unit {
    Tokens,
    Chars,
}

impl Unit {
    pub fn as_str(self) -> &'static str {
        match self {
            Unit::Tokens => "tokens",
            Unit::Chars => "chars",
        }
    }
}

/// A family's DiT prompt limit: `max_tokens` by the real tokenizer when
/// there is one, else `max_chars` characters.
#[derive(Clone)]
pub struct PromptLimit {
    pub counter: Option<Arc<dyn TokenCounter>>,
    pub max_tokens: usize,
    pub max_chars: usize,
}

impl std::fmt::Debug for PromptLimit {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PromptLimit")
            .field("counter", &self.counter.as_ref().map(|c| c.describe()))
            .field("max_tokens", &self.max_tokens)
            .field("max_chars", &self.max_chars)
            .finish()
    }
}

impl PartialEq for PromptLimit {
    fn eq(&self, o: &Self) -> bool {
        self.max_tokens == o.max_tokens
            && self.max_chars == o.max_chars
            && self.counter.as_ref().map(|c| c.describe()) == o.counter.as_ref().map(|c| c.describe())
    }
}

impl PromptLimit {
    /// Characters only (no tokenizer).
    pub fn chars(max_chars: usize) -> Self {
        Self { counter: None, max_tokens: 0, max_chars }
    }

    /// `(size, limit, unit)` of `s`.
    pub fn measure(&self, s: &str) -> (usize, usize, Unit) {
        if let Some(n) = self.counter.as_ref().and_then(|c| c.count(s)) {
            return (n, self.max_tokens, Unit::Tokens);
        }
        (s.chars().count(), self.max_chars, Unit::Chars)
    }

    pub fn fits(&self, s: &str) -> bool {
        let (n, max, _) = self.measure(s);
        n <= max
    }

    /// The unit in force now.
    pub fn unit(&self) -> Unit {
        self.measure("").2
    }
}

/// Cuts `s` to the longest prefix that `fits`, at a boundary: the last
/// sentence end (when that keeps at least half), else the last word; never
/// inside a `<d>…</d>` pair (a pair is kept whole or dropped). Used only as
/// the last resort, after the priority trims. Empty when nothing fits.
pub fn cut_to_fit(s: &str, fits: &dyn Fn(&str) -> bool) -> String {
    if fits(s) {
        return s.to_owned();
    }
    // Byte offsets inside a dialogue pair are not boundaries.
    let mut inside = vec![false; s.len() + 1];
    let mut i = 0;
    while let Some(o) = s[i..].find("<d>") {
        let start = i + o;
        let end = s[start..].find("</d>").map_or(s.len(), |e| start + e + 4);
        for x in inside.iter_mut().take(end).skip(start + 1) {
            *x = true;
        }
        i = end;
        if i >= s.len() {
            break;
        }
    }
    let mut words: Vec<usize> = Vec::new();
    let mut sentences: Vec<usize> = Vec::new();
    for (b, c) in s.char_indices() {
        if inside[b] {
            continue;
        }
        if c == ' ' || c == '\n' {
            words.push(b);
            let prev = s[..b].trim_end();
            if prev.ends_with(['.', '!', '?']) || prev.ends_with("</d>") {
                sentences.push(prev.len());
            }
        }
    }
    let ok = |b: usize| fits(s[..b].trim_end());
    // The longest fitting candidate by binary search (shorter prefixes of
    // the same text never take more tokens).
    let best = |cands: &[usize]| -> Option<usize> {
        let (mut lo, mut hi) = (0usize, cands.len());
        while lo < hi {
            let mid = (lo + hi) / 2;
            if ok(cands[mid]) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        lo.checked_sub(1).map(|k| cands[k])
    };
    let word = best(&words);
    let sentence = best(&sentences).filter(|&b| word.is_none_or(|w| b * 2 >= w));
    match sentence.or(word) {
        Some(b) => s[..b].trim_end().to_owned(),
        None => String::new(),
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// Whitespace words, `<d>` and `</d>` one token each (a stand-in for a
    /// real tokenizer in tests).
    pub struct Words;
    impl TokenCounter for Words {
        fn count(&self, text: &str) -> Option<usize> {
            Some(text.replace("<d>", " <d> ").replace("</d>", " </d> ").split_whitespace().count())
        }
        fn describe(&self) -> String {
            "words".into()
        }
    }

    #[test]
    fn estimate_and_limits() {
        assert_eq!(estimate_tokens(""), 0);
        assert_eq!(estimate_tokens("abcdefg"), 2);
        assert_eq!(estimate_tokens(&"x".repeat(50_000)), 14_286);
        let l = PromptLimit { counter: Some(Arc::new(Words)), max_tokens: 3, max_chars: 5 };
        assert_eq!(l.measure("a b c"), (3, 3, Unit::Tokens));
        assert!(!l.fits("a b c d"));
        assert_eq!(PromptLimit::chars(5).measure("a b c"), (5, 5, Unit::Chars));
    }

    #[test]
    fn cut_never_splits_a_dialogue_pair() {
        let fits = |n: usize| move |s: &str| Words.count(s).unwrap() <= n;
        let s = "Mara turns. Mara says <d>[English] Light the lamp now.</d> Waves crash.";
        // Inside the pair: the cut goes back to before it.
        assert_eq!(cut_to_fit(s, &fits(6)), "Mara turns.");
        assert_eq!(cut_to_fit(s, &fits(4)), "Mara turns.");
        // The whole pair fits: kept whole.
        assert_eq!(cut_to_fit(s, &fits(11)), "Mara turns. Mara says <d>[English] Light the lamp now.</d>");
        assert_eq!(cut_to_fit(s, &fits(100)), s);
        assert_eq!(cut_to_fit("one two three four", &fits(3)), "one two three");
        assert_eq!(cut_to_fit("one", &fits(0)), "");
    }
}
