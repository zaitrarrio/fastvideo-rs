//! The text encoders' tokenizers for the prompt upsampler's limits
//! (docs/serve/prompt-upsampling.md §Limits): the upsampler measures each
//! family's DiT prompt in that encoder's tokens.
//!
//! The tokenizer of a family is `[upsampler.tokenizers] <family>` when set,
//! else `<weights>/tokenizer/tokenizer.json` of the first `[[models]]`
//! entry of that family. It is loaded on first use and, while the file is
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
        c.models.iter().find(|m| family_of(&m.family) == Some(fam)).and_then(|m| m.weights.as_deref()).map(under)
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
}
