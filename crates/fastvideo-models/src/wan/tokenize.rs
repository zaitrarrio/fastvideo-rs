//! Host-side UMT5 prompt tokenization (tokenizers crate; no tensor backend).

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

/// `tokenizer.json` parsed once per path per process: parsing the UMT5
/// vocabulary costs ~0.4 s, which a warm request would otherwise pay twice
/// (prompt and negative) every time.
fn load(path: &str) -> Result<Arc<tokenizers::Tokenizer>, String> {
    static CACHE: OnceLock<Mutex<HashMap<String, Arc<tokenizers::Tokenizer>>>> = OnceLock::new();
    let cache = CACHE.get_or_init(Default::default);
    if let Some(t) = cache.lock().expect("tokenizer cache").get(path) {
        return Ok(t.clone());
    }
    let t = Arc::new(
        tokenizers::Tokenizer::from_file(path)
            .map_err(|e| format!("tokenizer load failed: {e}"))?,
    );
    cache
        .lock()
        .expect("tokenizer cache")
        .insert(path.to_string(), t.clone());
    Ok(t)
}

pub fn tokenize_prompt(
    path: &str,
    text: &str,
    max_len: usize,
) -> Result<(Vec<u32>, usize), String> {
    let tokenizer = load(path)?;
    let encoding = tokenizer
        .encode(text, true)
        .map_err(|e| format!("tokenize failed: {e}"))?;
    let mut ids = encoding.get_ids().to_vec();
    if ids.len() > max_len {
        ids.truncate(max_len);
    }
    let len = ids.len().max(1);
    Ok((ids, len))
}
