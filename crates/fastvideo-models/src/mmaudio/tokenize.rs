//! open_clip `SimpleTokenizer` semantics for MMAudio's text features.
//!
//! `open_clip.get_tokenizer('ViT-H-14-378-quickgelu')` is the default BPE
//! tokenizer: `<start_of_text>` (49406), the BPE of the lowercased,
//! whitespace-collapsed text, `<end_of_text>` (49407), truncated to 77 with the
//! last id forced to EOT, and **padded with 0** (HF's CLIP tokenizer pads with
//! EOT). MMAudio feeds all 77 positions (`patch_clip` returns every hidden
//! state), so the pad id matters. The BPE itself comes from the DFN5B repo's
//! `tokenizer.json` (the same vocab/merges).

use std::path::Path;

pub const SOT: u32 = 49406;
pub const EOT: u32 = 49407;
pub const CONTEXT: usize = 77;

/// `whitespace_clean(basic_clean(text)).lower()` minus ftfy (not needed for
/// the ASCII/UTF-8 prompts we pass).
pub fn clean(text: &str) -> String {
    text.split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .trim()
        .to_lowercase()
}

/// Pack BPE ids as open_clip does.
pub fn pack(bpe: &[u32], context: usize) -> Vec<u32> {
    let mut ids = Vec::with_capacity(context);
    ids.push(SOT);
    ids.extend_from_slice(bpe);
    ids.push(EOT);
    if ids.len() > context {
        ids.truncate(context);
        ids[context - 1] = EOT;
    }
    ids.resize(context, 0);
    ids
}

pub fn tokenize(tokenizer_json: &Path, text: &str) -> Result<Vec<u32>, String> {
    let tok = tokenizers::Tokenizer::from_file(tokenizer_json)
        .map_err(|e| format!("{}: {e}", tokenizer_json.display()))?;
    let cleaned = clean(text);
    let enc = tok
        .encode(cleaned.as_str(), false)
        .map_err(|e| format!("open_clip tokenize: {e}"))?;
    let bpe: Vec<u32> = enc
        .get_ids()
        .iter()
        .copied()
        .filter(|&i| i != SOT && i != EOT)
        .collect();
    Ok(pack(&bpe, CONTEXT))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pack_pads_with_zero_and_forces_eot() {
        let p = pack(&[1, 2, 3], 8);
        assert_eq!(p, vec![SOT, 1, 2, 3, EOT, 0, 0, 0]);
        let long: Vec<u32> = (1..20).collect();
        let p = pack(&long, 8);
        assert_eq!(p.len(), 8);
        assert_eq!(p[0], SOT);
        assert_eq!(p[7], EOT);
        assert_eq!(pack(&[], 4), vec![SOT, EOT, 0, 0]);
    }

    #[test]
    fn clean_collapses_whitespace() {
        assert_eq!(clean("  A Fox\n runs\t"), "a fox runs");
    }
}
