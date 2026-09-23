//! Prompt tokenization for MiniMax-H3, which is deliberately *not* what a
//! Qwen chat model does: the prompt is encoded verbatim, with no chat
//! template, no BOS/EOS and no padding (`tokenizer(prompt,
//! add_special_tokens=False)`, diffusers `encoders.py:192`, FastVideo
//! `minimax_h3_conditioning.py:187`).
//!
//! The one subtlety is the seven dialogue / lyrics markers. They are listed in
//! `tokenizer_config.json` under `additional_special_tokens` but are absent
//! from `tokenizer.json`, so transformers appends them to the vocabulary when
//! it loads the fast tokenizer, in the listed order, at the first free ids
//! (151669..=151675, right after `</think>`). A port that reads
//! `tokenizer.json` alone would split `<d>` into `<`, `d`, `>` and condition
//! every dialogue prompt on different rows of the embedding table. We add them
//! the same way the reference does; `h3_oracle.py` records the reference's ids
//! in `meta.added_special_token_ids` and the `h3 text` stage compares.

use std::path::Path;

use tokenizers::{AddedToken, Tokenizer};

/// `additional_special_tokens` of the checkpoint's `tokenizer_config.json`
/// that `tokenizer.json` does not define, in the order transformers adds them.
pub const H3_ADDED_SPECIAL_TOKENS: [&str; 7] = [
    "<d>",
    "</d>",
    "<|cutoff|>",
    "<|lyrics_start|>",
    "<|lyrics_end|>",
    "<|caption_start|>",
    "<|caption_end|>",
];

/// Id of `<d>` in the reference tokenizer; the other six follow consecutively.
pub const H3_FIRST_ADDED_ID: u32 = 151_669;

/// Rows of `embed_tokens`; any id at or past this cannot be embedded.
pub const H3_VOCAB_ROWS: u32 = 151_936;

/// FastVideo's `text_len`: neither inference path truncates, so a longer
/// prompt is rejected rather than silently cut (docs/ports/h3.md, section b).
pub const H3_MAX_PROMPT_TOKENS: usize = 1024;

pub struct H3Tokenizer {
    inner: Tokenizer,
    added: Vec<(String, u32)>,
}

impl H3Tokenizer {
    /// Load `tokenizer.json` and append the H3 marker tokens it lacks.
    pub fn from_file(tokenizer_json: &Path) -> Result<Self, String> {
        let inner = Tokenizer::from_file(tokenizer_json)
            .map_err(|e| format!("load {}: {e}", tokenizer_json.display()))?;
        Self::from_tokenizer(inner)
    }

    /// As [`Self::from_file`], from the JSON bytes (tests).
    pub fn from_bytes(json: &[u8]) -> Result<Self, String> {
        let inner = Tokenizer::from_bytes(json).map_err(|e| format!("parse tokenizer: {e}"))?;
        Self::from_tokenizer(inner)
    }

    fn from_tokenizer(mut inner: Tokenizer) -> Result<Self, String> {
        // One call, in order: ids are handed out consecutively from the
        // current vocabulary size, exactly as transformers' `add_tokens` does
        // through this same library. Tokens already present keep their id.
        let tokens: Vec<AddedToken> = H3_ADDED_SPECIAL_TOKENS
            .iter()
            .map(|t| AddedToken::from(*t, true))
            .collect();
        inner.add_special_tokens(&tokens);
        let added = H3_ADDED_SPECIAL_TOKENS
            .iter()
            .map(|t| {
                inner
                    .token_to_id(t)
                    .map(|id| ((*t).to_string(), id))
                    .ok_or_else(|| format!("tokenizer refused the added token {t}"))
            })
            .collect::<Result<Vec<_>, String>>()?;
        Ok(Self { inner, added })
    }

    /// `(token, id)` for the seven markers, as this tokenizer numbers them.
    pub fn added_special_token_ids(&self) -> &[(String, u32)] {
        &self.added
    }

    /// Whether the markers landed on the ids the reference tokenizer gives
    /// them (`151669..=151675`). False for a `tokenizer.json` with a different
    /// vocabulary, which would address the wrong embedding rows.
    pub fn added_ids_match_reference(&self) -> bool {
        self.added
            .iter()
            .zip(H3_FIRST_ADDED_ID..)
            .all(|((_, id), want)| *id == want)
    }

    /// Token ids of `prompt`, verbatim: no template, no special tokens.
    pub fn encode(&self, prompt: &str) -> Result<Vec<u32>, String> {
        let encoding = self
            .inner
            .encode(prompt, false)
            .map_err(|e| format!("tokenize: {e}"))?;
        let ids = encoding.get_ids().to_vec();
        if ids.is_empty() {
            return Err("the prompt tokenizes to nothing; H3 needs at least one text row".into());
        }
        if ids.len() > H3_MAX_PROMPT_TOKENS {
            return Err(format!(
                "the prompt is {} tokens; H3 conditions on at most {H3_MAX_PROMPT_TOKENS} and does not truncate",
                ids.len()
            ));
        }
        if let Some(bad) = ids.iter().find(|&&id| id >= H3_VOCAB_ROWS) {
            return Err(format!(
                "token id {bad} is outside the {H3_VOCAB_ROWS}-row embedding table"
            ));
        }
        Ok(ids)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A word-level tokenizer with a post-processor that *would* wrap the
    /// sequence in special tokens if asked to.
    const TINY: &str = r#"{
        "version": "1.0",
        "truncation": null,
        "padding": null,
        "added_tokens": [{"id": 0, "content": "[CLS]", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true}],
        "normalizer": null,
        "pre_tokenizer": {"type": "Whitespace"},
        "post_processor": {
            "type": "TemplateProcessing",
            "single": [{"SpecialToken": {"id": "[CLS]", "type_id": 0}}, {"Sequence": {"id": "A", "type_id": 0}}],
            "pair": [{"Sequence": {"id": "A", "type_id": 0}}, {"Sequence": {"id": "B", "type_id": 1}}],
            "special_tokens": {"[CLS]": {"id": "[CLS]", "ids": [0], "tokens": ["[CLS]"]}}
        },
        "decoder": null,
        "model": {"type": "WordLevel", "vocab": {"[CLS]": 0, "[UNK]": 1, "a": 2, "dog": 3, "says": 4, "d": 5, "<": 6, ">": 7, "hi": 8}, "unk_token": "[UNK]"}
    }"#;

    #[test]
    fn the_prompt_is_encoded_without_template_tokens() {
        let t = H3Tokenizer::from_bytes(TINY.as_bytes()).unwrap();
        assert_eq!(
            t.encode("a dog says hi").unwrap(),
            vec![2, 3, 4, 8],
            "no [CLS] in front"
        );
    }

    #[test]
    fn dialogue_markers_are_single_tokens_appended_after_the_vocabulary() {
        let t = H3Tokenizer::from_bytes(TINY.as_bytes()).unwrap();
        let ids: Vec<u32> = t
            .added_special_token_ids()
            .iter()
            .map(|(_, id)| *id)
            .collect();
        assert_eq!(
            ids,
            (9..16).collect::<Vec<u32>>(),
            "consecutive, in the listed order"
        );
        assert_eq!(t.added_special_token_ids()[1].0, "</d>");
        assert_eq!(
            t.encode("a dog says <d>hi</d>").unwrap(),
            vec![2, 3, 4, 9, 8, 10]
        );
        assert!(
            !t.added_ids_match_reference(),
            "a toy vocabulary does not end at 151668"
        );
    }

    #[test]
    fn an_empty_prompt_is_an_error() {
        let t = H3Tokenizer::from_bytes(TINY.as_bytes()).unwrap();
        assert!(t.encode("").is_err());
    }
}
