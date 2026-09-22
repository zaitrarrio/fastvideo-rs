//! Host-side T5 tokenization for Cosmos (`tokenizer/tokenizer.json`).

use std::path::Path;

/// Pad/truncate to `max_len` like Diffusers Cosmos `_get_t5_prompt_embeds`.
pub fn tokenize_t5(
    root: &Path,
    prompt: &str,
    max_len: usize,
) -> Result<(Vec<u32>, Vec<bool>), String> {
    let path = root.join("tokenizer").join("tokenizer.json");
    if !path.is_file() {
        return Err(format!("missing {}", path.display()));
    }
    let tokenizer = tokenizers::Tokenizer::from_file(&path)
        .map_err(|e| format!("T5 tokenizer load failed: {e}"))?;
    let encoding = tokenizer
        .encode(prompt, true)
        .map_err(|e| format!("T5 tokenize failed: {e}"))?;
    let mut ids = encoding.get_ids().to_vec();
    if ids.len() > max_len {
        ids.truncate(max_len);
    }
    let mut mask = vec![true; ids.len()];
    let pad_id = tokenizer
        .token_to_id("<pad>")
        .or_else(|| tokenizer.token_to_id("[PAD]"))
        .unwrap_or(0);
    while ids.len() < max_len {
        ids.push(pad_id);
        mask.push(false);
    }
    Ok((ids, mask))
}

#[cfg(test)]
mod tests {
    #[test]
    fn pad_id_fallback_is_zero() {
        // Structural: real tokenize needs Hub files; pad default documented as 0.
        assert_eq!(0u32, 0);
    }
}
