//! HunyuanVideo 1.5 dual-text host helpers (Qwen chat template + ByT5 glyphs).

use std::path::Path;

/// Diffusers `PROMPT_TEMPLATE_ENCODE_START_IDX` — crop this many leading tokens
/// from the Qwen chat-templated sequence before feeding the DiT.
pub const QWEN_CROP_START: usize = 108;

/// `num_hidden_layers_to_skip` in Diffusers `_get_mllm_prompt_embeds` (tap
/// `hidden_states[-(skip + 1)]` ≡ our tap `num_layers - skip`).
pub const QWEN_LAYERS_TO_SKIP: usize = 2;

/// System message Diffusers prepends via the Qwen chat template.
pub const QWEN_SYSTEM_MESSAGE: &str =
    "You are a helpful assistant. Describe the video by detailing the following aspects: \
1. The main content and theme of the video. \
2. The color, shape, size, texture, quantity, text, and spatial relationships of the objects. \
3. Actions, events, behaviors temporal relationships, physical movement changes of the objects. \
4. background environment, light, style and atmosphere. \
5. camera angles, movements, and transitions used in the video.";

/// Mid-layer tap for a `num_layers`-deep Qwen tower (`hidden_states[-3]`).
pub fn qwen_hidden_tap(num_layers: usize) -> usize {
    num_layers.saturating_sub(QWEN_LAYERS_TO_SKIP)
}

/// Diffusers `extract_glyph_texts`: quoted spans → `"Text \"…\". "` or `None`.
pub fn extract_glyph_texts(prompt: &str) -> Option<String> {
    let mut found = Vec::new();
    let bytes = prompt.as_bytes();
    let mut i = 0usize;
    while i < bytes.len() {
        let open = match bytes[i] {
            b'"' => Some((i + 1, b'"')),
            // U+201C LEFT DOUBLE QUOTATION MARK
            0xe2 if i + 2 < bytes.len() && bytes[i + 1] == 0x80 && bytes[i + 2] == 0x9c => {
                Some((i + 3, 0)) // closed by U+201D below
            }
            _ => None,
        };
        let Some((start, ascii_close)) = open else {
            i += 1;
            continue;
        };
        if ascii_close == b'"' {
            if let Some(end) = prompt[start..].find('"') {
                found.push(prompt[start..start + end].to_string());
                i = start + end + 1;
                continue;
            }
        } else {
            // Find UTF-8 U+201D (e2 80 9d).
            let rest = &prompt[start..];
            if let Some(rel) = rest.find('\u{201d}') {
                found.push(rest[..rel].to_string());
                i = start + rel + '\u{201d}'.len_utf8();
                continue;
            }
        }
        i += 1;
    }
    if found.is_empty() {
        return None;
    }
    // Diffusers: dedupe only when more than one match.
    if found.len() > 1 {
        let mut uniq = Vec::new();
        for t in found {
            if !uniq.contains(&t) {
                uniq.push(t);
            }
        }
        found = uniq;
    }
    let joined = found
        .iter()
        .map(|t| format!("Text \"{t}\""))
        .collect::<Vec<_>>()
        .join(". ");
    Some(format!("{joined}. "))
}

/// User turn body Diffusers builds before `apply_chat_template`.
pub fn format_user_prompt(prompt: &str) -> String {
    format!("{}\n{}", QWEN_SYSTEM_MESSAGE, prompt)
}

/// Tokenize with Diffusers `tokenizer/tokenizer.json` (Qwen2).
///
/// Uses a plain encode of [`format_user_prompt`] when the JSON chat template
/// is unavailable through the tokenizers crate API; packs that ship
/// `tokenizer_config.json` chat templates should still produce a sequence
/// long enough that [`QWEN_CROP_START`] is meaningful after a real
/// `apply_chat_template` path is wired.
pub fn tokenize_qwen(root: &Path, prompt: &str, max_length: usize) -> Result<Vec<u32>, String> {
    let path = root.join("tokenizer").join("tokenizer.json");
    let tokenizer =
        tokenizers::Tokenizer::from_file(&path).map_err(|e| format!("{}: {e}", path.display()))?;
    let body = format_user_prompt(prompt);
    let encoding = tokenizer
        .encode(body.as_str(), true)
        .map_err(|e| format!("hy15 qwen tokenize: {e}"))?;
    let mut ids = encoding.get_ids().to_vec();
    if ids.len() > max_length {
        ids.truncate(max_length);
    }
    Ok(ids)
}

/// Diffusers `ByT5Tokenizer`: UTF-8 bytes + 3 (0=pad, 1=eos, 2=unk).
pub fn tokenize_byt5(text: &str, max_length: usize) -> Vec<u32> {
    let mut ids: Vec<u32> = text.bytes().map(|b| u32::from(b) + 3).collect();
    ids.push(1); // eos
    if ids.len() > max_length {
        ids.truncate(max_length);
        if let Some(last) = ids.last_mut() {
            *last = 1;
        }
    } else {
        ids.resize(max_length, 0); // pad
    }
    ids
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn glyph_none_without_quotes() {
        assert!(extract_glyph_texts("a cat runs").is_none());
    }

    #[test]
    fn glyph_ascii_quotes() {
        let g = extract_glyph_texts(r#"sign says "OPEN""#).unwrap();
        assert!(g.contains("Text \"OPEN\""));
        assert!(g.ends_with(". "));
    }

    #[test]
    fn tap_is_layers_minus_skip() {
        assert_eq!(qwen_hidden_tap(28), 26);
    }

    #[test]
    fn byt5_bytes_offset() {
        let ids = tokenize_byt5("A", 8);
        assert_eq!(ids[0], u32::from(b'A') + 3);
        assert_eq!(ids[1], 1); // eos
        assert_eq!(ids[2], 0); // pad
        assert_eq!(ids.len(), 8);
    }
}
