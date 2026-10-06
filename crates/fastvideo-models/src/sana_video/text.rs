//! SANA-Video prompt handling (Diffusers `SanaVideoPipeline.encode_prompt`).
//!
//! The positive prompt is prefixed with the "complex human instruction"
//! (CHI) and tokenized right-padded to `len(tokenize(CHI)) + max_seq - 2`;
//! the negative prompt gets no CHI and pads to `max_seq`. Gemma-2's last
//! hidden state is then cut to `select_index = [0] + range(-max_seq + 1, 0)`
//! (BOS plus the last `max_seq - 1` positions), which keeps the end of the
//! instruction, the user prompt and the padding. Padding positions are
//! masked out of the DiT's cross-attention, so the port encodes only the
//! real tokens and drops the masked ones (exact up to the reference's
//! `-10000` bias, which underflows to a zero weight).

use std::path::Path;

/// Diffusers default `max_sequence_length`.
pub const SANA_MAX_SEQUENCE_LENGTH: usize = 300;

/// `complex_human_instruction`, joined with `"\n"` by the pipeline.
pub const COMPLEX_HUMAN_INSTRUCTION: [&str; 8] = [
    "Given a user prompt, generate an 'Enhanced prompt' that provides detailed visual descriptions suitable for video generation. Evaluate the level of detail in the user prompt:",
    "- If the prompt is simple, focus on adding specifics about colors, shapes, sizes, textures, motion, and temporal relationships to create vivid and dynamic scenes.",
    "- If the prompt is already detailed, refine and enhance the existing details slightly without overcomplicating.",
    "Here are examples of how to transform or refine prompts:",
    "- User Prompt: A cat sleeping -> Enhanced: A small, fluffy white cat slowly settling into a curled position, peacefully falling asleep on a warm sunny windowsill, with gentle sunlight filtering through surrounding pots of blooming red flowers.",
    "- User Prompt: A busy city street -> Enhanced: A bustling city street scene at dusk, featuring glowing street lamps gradually lighting up, a diverse crowd of people in colorful clothing walking past, and a double-decker bus smoothly passing by towering glass skyscrapers.",
    "Please generate only the enhanced description for the prompt below and avoid including any additional commentary or evaluations:",
    "User Prompt: ",
];

/// The CHI text the pipeline prepends to the prompt.
pub fn chi_prompt() -> String {
    COMPLEX_HUMAN_INSTRUCTION.join("\n")
}

/// The model card's motion conditioning: `prompt + " motion score: {m}."`.
/// The pipeline itself does not add it; callers opt in.
pub fn with_motion_score(prompt: &str, motion_score: Option<u32>) -> String {
    match motion_score {
        Some(m) => format!("{prompt} motion score: {m}."),
        None => prompt.to_string(),
    }
}

/// Positions of the padded sequence the DiT sees, in order.
pub fn select_index(max_sequence_length: usize, padded_len: usize) -> Vec<usize> {
    let tail = max_sequence_length.saturating_sub(1).min(padded_len.saturating_sub(1));
    std::iter::once(0)
        .chain(padded_len - tail..padded_len)
        .collect()
}

/// How one prompt maps onto the DiT's text tokens.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PromptLayout {
    /// Token ids of the real (unpadded) sequence, BOS first.
    pub ids: Vec<u32>,
    /// `max_length` the reference pads to.
    pub padded_len: usize,
    /// Positions (into `ids`) the DiT attends to, in order. Masked padding
    /// positions are dropped.
    pub kept: Vec<usize>,
}

impl PromptLayout {
    /// `ids` is the tokenizer output with special tokens and no padding;
    /// truncated to `padded_len` like `truncation=True`.
    pub fn new(mut ids: Vec<u32>, padded_len: usize, max_sequence_length: usize) -> Self {
        ids.truncate(padded_len);
        let kept = select_index(max_sequence_length, padded_len)
            .into_iter()
            .filter(|&p| p < ids.len())
            .collect();
        Self {
            ids,
            padded_len,
            kept,
        }
    }
}

/// `max_length_all` for a positive prompt: `len(tokenize(chi)) + max_seq - 2`.
pub fn chi_padded_len(chi_tokens_with_bos: usize, max_sequence_length: usize) -> usize {
    chi_tokens_with_bos + max_sequence_length - 2
}

fn load(root: &Path) -> Result<tokenizers::Tokenizer, String> {
    let path = root.join("tokenizer").join("tokenizer.json");
    tokenizers::Tokenizer::from_file(&path).map_err(|e| format!("{}: {e}", path.display()))
}

fn encode(tok: &tokenizers::Tokenizer, text: &str) -> Result<Vec<u32>, String> {
    Ok(tok
        .encode(text, true)
        .map_err(|e| format!("gemma tokenize: {e}"))?
        .get_ids()
        .to_vec())
}

/// Positive and negative prompt layouts from `root/tokenizer/tokenizer.json`.
pub fn tokenize_pair(
    root: &Path,
    prompt: &str,
    negative: &str,
    max_sequence_length: usize,
) -> Result<(PromptLayout, PromptLayout), String> {
    let tok = load(root)?;
    let chi = chi_prompt();
    let chi_len = encode(&tok, &chi)?.len();
    let pos = encode(&tok, &format!("{chi}{prompt}"))?;
    let neg = encode(&tok, negative)?;
    Ok((
        PromptLayout::new(
            pos,
            chi_padded_len(chi_len, max_sequence_length),
            max_sequence_length,
        ),
        PromptLayout::new(neg, max_sequence_length, max_sequence_length),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn select_index_is_bos_plus_the_tail() {
        assert_eq!(select_index(4, 10), vec![0, 7, 8, 9]);
        // Negative prompt: padded to max_seq, so every position is kept.
        assert_eq!(select_index(300, 300), (0..300).collect::<Vec<_>>());
    }

    #[test]
    fn layout_drops_padding_and_keeps_the_prompt_tail() {
        // 6 CHI tokens (with BOS), max_seq 5 → padded 9; prompt adds 2 tokens.
        let ids: Vec<u32> = (100..108).collect(); // 8 real tokens
        let l = PromptLayout::new(ids, chi_padded_len(6, 5), 5);
        assert_eq!(l.padded_len, 9);
        // select [0, 5, 6, 7, 8]; position 8 is padding.
        assert_eq!(l.kept, vec![0, 5, 6, 7]);
        // Empty negative: BOS only.
        let n = PromptLayout::new(vec![2], 5, 5);
        assert_eq!(n.kept, vec![0]);
    }

    #[test]
    fn truncates_like_the_tokenizer() {
        let l = PromptLayout::new((0..20).collect(), 9, 5);
        assert_eq!(l.ids.len(), 9);
        assert_eq!(l.kept, vec![0, 5, 6, 7, 8]);
    }

    #[test]
    fn chi_and_motion_text() {
        assert!(chi_prompt().ends_with("\nUser Prompt: "));
        assert_eq!(chi_prompt().matches('\n').count(), 7);
        assert_eq!(with_motion_score("a cat", Some(30)), "a cat motion score: 30.");
        assert_eq!(with_motion_score("a cat", None), "a cat");
    }
}
