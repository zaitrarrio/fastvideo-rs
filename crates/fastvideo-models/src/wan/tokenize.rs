//! Host-side UMT5 prompt tokenization (tokenizers crate; no tensor backend).

pub fn tokenize_prompt(path: &str, text: &str, max_len: usize) -> Result<(Vec<u32>, usize), String> {
    let tokenizer = tokenizers::Tokenizer::from_file(path)
        .map_err(|e| format!("tokenizer load failed: {e}"))?;
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
