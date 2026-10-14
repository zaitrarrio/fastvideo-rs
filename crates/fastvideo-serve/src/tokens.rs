//! The text encoders' tokenizers for the prompt upsampler's limits
//! (docs/serve/prompt-upsampling.md §Limits): the upsampler measures each
//! family's DiT prompt in that encoder's tokens.
//!
//! The tokenizer of a family is `[upsampler.tokenizers] <family>` when set,
//! else `<weights>/tokenizer/tokenizer.json` of the first `[[models]]`
//! entry of that family (for an H3 model, of its text tree: `text_weights`
//! when the entry sets it, and for FastH3 Trim, whose tree has no tokenizer,
//! the `h3-base` tree beside it, as the engine catalog reads it). It is loaded on first use and, while the file is
//! missing (weights still streaming from R2), looked for again at most
//! every 30 s; meanwhile the limit is in characters. H3 uses
//! [`H3Tokenizer`] (the `<d>` markers as single tokens, no template);
//! Wan (umT5) and LTX (Gemma 3) count with the special tokens their
//! encoders add.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use fastvideo_models::h3::tokenizer::H3Tokenizer;
use fastvideo_upsampler::{rewrite::PromptFamily, TokenCounter};

use crate::config::Config;

enum Loaded {
    H3(H3Tokenizer),
    Hf(tokenizers::Tokenizer),
}

struct State {
    loaded: Option<Loaded>,
    tried: Option<Instant>,
}

/// One family's tokenizer, loaded lazily from `path`.
pub struct FileCounter {
    path: PathBuf,
    h3: bool,
    state: Mutex<State>,
}

impl FileCounter {
    pub fn new(path: PathBuf, family: PromptFamily) -> Self {
        Self { path, h3: family == PromptFamily::H3, state: Mutex::new(State { loaded: None, tried: None }) }
    }

    fn load(&self) -> Result<Loaded, String> {
        if self.h3 {
            H3Tokenizer::from_file(&self.path).map(Loaded::H3)
        } else {
            tokenizers::Tokenizer::from_file(&self.path).map(Loaded::Hf).map_err(|e| format!("load {}: {e}", self.path.display()))
        }
    }
}

const RETRY: Duration = Duration::from_secs(30);

impl TokenCounter for FileCounter {
    fn count(&self, text: &str) -> Option<usize> {
        let mut st = self.state.lock().ok()?;
        if st.loaded.is_none() && st.tried.is_none_or(|t| t.elapsed() >= RETRY) {
            st.tried = Some(Instant::now());
            if self.path.is_file() {
                match self.load() {
                    Ok(l) => {
                        tracing::info!(path = %self.path.display(), "upsampler: text-encoder tokenizer loaded; prompt limits are in tokens");
                        st.loaded = Some(l);
                    }
                    Err(e) => tracing::warn!(error = %e, "upsampler: the tokenizer cannot be loaded; prompt limits stay in characters"),
                }
            }
        }
        match st.loaded.as_ref()? {
            Loaded::H3(t) => t.count(text).ok(),
            Loaded::Hf(t) => t.encode(text, true).ok().map(|e| e.get_ids().len()),
        }
    }

    fn describe(&self) -> String {
        self.path.display().to_string()
    }
}

/// `tokenizer/tokenizer.json` under a model's weights.
fn under(weights: &str) -> PathBuf {
    Path::new(weights).join("tokenizer").join("tokenizer.json")
}

/// The tree that holds a model's text-encoder tokenizer. H3: the entry's
/// `text_weights` when set; FastH3 Trim (`fasth3-trim-8step`, recipe
/// `h3-trim` / `trim-8step`) ships no tokenizer and the engine reads it from
/// `h3-base` beside its tree (caps.rs `catalog`). Else the model's weights.
fn text_tree(m: &crate::config::ModelCfg, fam: PromptFamily) -> Option<PathBuf> {
    let weights = m.weights.as_deref()?;
    if fam == PromptFamily::H3 {
        if let Some(t) = m.extra.get("text_weights").and_then(|v| v.as_str()).filter(|t| !t.trim().is_empty()) {
            return Some(PathBuf::from(t));
        }
        let w = Path::new(weights.trim_end_matches('/'));
        let trim = w.file_name().and_then(|n| n.to_str()).is_some_and(|n| n.starts_with("fasth3-trim"))
            || m.recipe.as_deref().is_some_and(|r| matches!(r, "h3-trim" | "trim-8step" | "fasth3-trim-8step"));
        if trim {
            if let Some(parent) = w.parent() {
                return Some(parent.join("h3-base"));
            }
        }
    }
    Some(PathBuf::from(weights))
}

/// The family of a `[[models]]` entry (`h3`, `wan`, `ltx`/`ltx2`).
fn family_of(f: &str) -> Option<PromptFamily> {
    match f.trim().to_ascii_lowercase().as_str() {
        "h3" | "minimax_h3" => Some(PromptFamily::H3),
        "wan" => Some(PromptFamily::Wan),
        "ltx" | "ltx2" => Some(PromptFamily::Ltx),
        _ => None,
    }
}

/// The counters `[upsampler]` and `[[models]]` point at.
pub fn counters(c: &Config) -> fastvideo_upsampler::Counters {
    let explicit = c.upsampler.as_ref().map(|u| u.tokenizers.clone()).unwrap_or_default();
    let path = |fam: PromptFamily, set: &Option<String>| -> Option<PathBuf> {
        if let Some(p) = set.as_deref().map(str::trim).filter(|p| !p.is_empty()) {
            return Some(PathBuf::from(p));
        }
        c.models
            .iter()
            .find(|m| family_of(&m.family) == Some(fam))
            .and_then(|m| text_tree(m, fam))
            .map(|t| under(&t.to_string_lossy()))
    };
    let make = |fam: PromptFamily, set: &Option<String>| -> Option<Arc<dyn TokenCounter>> {
        path(fam, set).map(|p| Arc::new(FileCounter::new(p, fam)) as Arc<dyn TokenCounter>)
    };
    fastvideo_upsampler::Counters {
        h3: make(PromptFamily::H3, &explicit.h3),
        wan: make(PromptFamily::Wan, &explicit.wan),
        ltx: make(PromptFamily::Ltx, &explicit.ltx),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A word-level vocabulary ending at id 8: the H3 markers are appended
    /// after it, so `<d>` and `</d>` count one token each.
    const TINY: &str = r#"{
        "version": "1.0", "truncation": null, "padding": null, "added_tokens": [],
        "normalizer": null, "pre_tokenizer": {"type": "Whitespace"}, "post_processor": null, "decoder": null,
        "model": {"type": "WordLevel", "vocab": {"[UNK]": 0, "mara": 1, "says": 2, "hi": 3, "<": 4, "d": 5, ">": 6, "/": 7, "[": 8}, "unk_token": "[UNK]"}
    }"#;

    #[test]
    fn h3_counts_dialogue_markers_as_one_token_and_waits_for_the_file() {
        let dir = std::env::temp_dir().join(format!("fv-tok-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(dir.join("tokenizer")).unwrap();
        let path = dir.join("tokenizer").join("tokenizer.json");
        let c = FileCounter::new(path.clone(), PromptFamily::H3);
        assert_eq!(c.count("mara says hi"), None, "no file yet: characters");
        std::fs::write(&path, TINY).unwrap();
        // Looked for again only after the retry interval.
        assert_eq!(c.count("mara says hi"), None);
        c.state.lock().unwrap().tried = None;
        assert_eq!(c.count("mara says hi"), Some(3));
        assert_eq!(c.count("mara says <d>hi</d>"), Some(5));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn counters_follow_config_and_models() {
        let mut c = Config::default();
        c.models.push(toml::from_str("id = \"fasth3\"\nfamily = \"h3\"\nweights = \"/w/FastH3\"").unwrap());
        let k = counters(&c);
        assert_eq!(k.h3.map(|x| x.describe()).as_deref(), Some("/w/FastH3/tokenizer/tokenizer.json"));
        assert!(k.wan.is_none() && k.ltx.is_none());
        let mut u = crate::config::UpsamplerCfg::default();
        u.tokenizers.h3 = Some("/t/h3.json".into());
        u.tokenizers.wan = Some("/t/wan.json".into());
        c.upsampler = Some(u);
        let k = counters(&c);
        assert_eq!(k.h3.map(|x| x.describe()).as_deref(), Some("/t/h3.json"));
        assert_eq!(k.wan.map(|x| x.describe()).as_deref(), Some("/t/wan.json"));
    }

    /// FastH3 Trim's tree has no tokenizer: its prompts are measured with
    /// h3-base's (the engine's text tree), or the entry's `text_weights`.
    #[test]
    fn trim_counts_with_the_h3_base_tokenizer() {
        let mut c = Config::default();
        c.models.push(
            toml::from_str("id = \"fasth3-trim-8step\"\nfamily = \"h3\"\nrecipe = \"h3-trim\"\nweights = \"/w/fasth3-trim-8step\"").unwrap(),
        );
        let k = counters(&c);
        assert_eq!(k.h3.map(|x| x.describe()).as_deref(), Some("/w/h3-base/tokenizer/tokenizer.json"));
        c.models[0] = toml::from_str("id = \"t\"\nfamily = \"h3\"\nweights = \"/w/fasth3-trim-8step/\"").unwrap();
        let k = counters(&c);
        assert_eq!(k.h3.map(|x| x.describe()).as_deref(), Some("/w/h3-base/tokenizer/tokenizer.json"));
        c.models[0] =
            toml::from_str("id = \"t\"\nfamily = \"h3\"\nweights = \"/w/fasth3-trim-8step\"\ntext_weights = \"/v/h3-base\"").unwrap();
        let k = counters(&c);
        assert_eq!(k.h3.map(|x| x.describe()).as_deref(), Some("/v/h3-base/tokenizer/tokenizer.json"));
    }
}
