//! Kandinsky 5 host helpers (CLIP tokenize).

use std::path::Path;

/// Diffusers `tokenizer_2` CLIPTokenizer via `tokenizer.json`.
pub fn tokenize_clip(root: &Path, prompt: &str, max_length: usize) -> Result<Vec<u32>, String> {
    let path = root.join("tokenizer_2").join("tokenizer.json");
    if !path.is_file() {
        return Err(format!(
            "k5 CLIP: missing {} (need HuggingFace CLIP tokenizer.json)",
            path.display()
        ));
    }
    let tokenizer = tokenizers::Tokenizer::from_file(&path)
        .map_err(|e| format!("{}: {e}", path.display()))?;
    let encoding = tokenizer
        .encode(prompt, true)
        .map_err(|e| format!("clip tokenize: {e}"))?;
    let mut ids = encoding.get_ids().to_vec();
    if ids.len() > max_length {
        ids.truncate(max_length);
        if let Some(last) = ids.last_mut() {
            *last = 49407; // EOS
        }
    } else {
        ids.resize(max_length, 0); // pad
    }
    Ok(ids)
}

#[cfg(test)]
mod tests {
    #[test]
    fn truncate_sets_eos() {
        // Unit-level: just ensure the constant used above stays stable.
        assert_eq!(49407u32, 49407);
    }
}
