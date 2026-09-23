//! MiniMax-H3 Qwen3-VL prompt presentation: ordered labels + vision pad
//! blocks for FL2VA / Ref2VA (FastVideo `minimax_h3_conditioning.py`).
//!
//! Token *ids* are produced here; vision *features* that replace image/video
//! pad embeddings live in the cudarc vision tower.

use super::config::{H3TextEncoderConfig, TAG_TEXT, TAG_VIDEO};
use super::tokenizer::H3Tokenizer;

/// One prepared visual block's pad count (after spatial merge).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VisionTokenCount {
    pub pads: usize,
}

/// Ordered Ref2VA medium for presentation (geometry already prepared).
#[derive(Debug, Clone)]
pub enum PresentationRef {
    Image {
        token_count: usize,
    },
    /// `token_count` is the total merged vision pads (`T*H*W / merge²`).
    Video {
        token_count: usize,
        block_timestamps: Vec<f64>,
    },
    Audio,
}

/// Tokenized multimodal presentation ready for the language model.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TextPresentation {
    pub token_ids: Vec<u32>,
    /// Per-token DiT AdaLN tags for the text span (`TAG_TEXT` / `TAG_VIDEO`).
    pub token_tags: Vec<u8>,
}

impl TextPresentation {
    pub fn len(&self) -> usize {
        self.token_ids.len()
    }

    pub fn is_empty(&self) -> bool {
        self.token_ids.is_empty()
    }
}

fn emit_text(
    tokenizer: &H3Tokenizer,
    text: &str,
    ids: &mut Vec<u32>,
    tags: &mut Vec<u8>,
) -> Result<(), String> {
    let piece = tokenizer.encode(text)?;
    tags.extend(std::iter::repeat_n(TAG_TEXT, piece.len()));
    ids.extend(piece);
    Ok(())
}

fn emit_vision(
    cfg: &H3TextEncoderConfig,
    pad_token: u32,
    count: usize,
    ids: &mut Vec<u32>,
    tags: &mut Vec<u8>,
) -> Result<(), String> {
    if count == 0 {
        return Err("a vision block needs at least one pad token".into());
    }
    ids.push(cfg.vision_start_token_id);
    ids.extend(std::iter::repeat_n(pad_token, count));
    ids.push(cfg.vision_end_token_id);
    tags.extend(std::iter::repeat_n(TAG_VIDEO, count + 2));
    Ok(())
}

/// FL2VA: `<Picture i>: ` + vision block per keyframe, then the prompt.
pub fn build_fl2va_presentation(
    tokenizer: &H3Tokenizer,
    cfg: &H3TextEncoderConfig,
    prompt: &str,
    image_token_counts: &[usize],
) -> Result<TextPresentation, String> {
    let mut token_ids = Vec::new();
    let mut token_tags = Vec::new();
    for (i, &count) in image_token_counts.iter().enumerate() {
        emit_text(
            tokenizer,
            &format!("<Picture {}>: ", i + 1),
            &mut token_ids,
            &mut token_tags,
        )?;
        emit_vision(
            cfg,
            cfg.image_token_id,
            count,
            &mut token_ids,
            &mut token_tags,
        )?;
    }
    emit_text(tokenizer, prompt, &mut token_ids, &mut token_tags)?;
    if token_ids.is_empty() {
        return Err("FL2VA presentation tokenizes to nothing".into());
    }
    Ok(TextPresentation {
        token_ids,
        token_tags,
    })
}

/// Ref2VA: ordered `<Audio|Picture|Video n>: ` labels, vision blocks, prompt.
pub fn build_ref2va_presentation(
    tokenizer: &H3Tokenizer,
    cfg: &H3TextEncoderConfig,
    prompt: &str,
    references: &[PresentationRef],
) -> Result<TextPresentation, String> {
    let mut token_ids = Vec::new();
    let mut token_tags = Vec::new();
    let mut counts = [0usize; 3]; // image, video, audio
    for reference in references {
        match reference {
            PresentationRef::Audio => {
                counts[2] += 1;
                emit_text(
                    tokenizer,
                    &format!("<Audio {}>: ", counts[2]),
                    &mut token_ids,
                    &mut token_tags,
                )?;
            }
            PresentationRef::Image { token_count } => {
                counts[0] += 1;
                emit_text(
                    tokenizer,
                    &format!("<Picture {}>: ", counts[0]),
                    &mut token_ids,
                    &mut token_tags,
                )?;
                emit_vision(
                    cfg,
                    cfg.image_token_id,
                    *token_count,
                    &mut token_ids,
                    &mut token_tags,
                )?;
            }
            PresentationRef::Video {
                token_count,
                block_timestamps,
            } => {
                counts[1] += 1;
                emit_text(
                    tokenizer,
                    &format!("<Video {}>: ", counts[1]),
                    &mut token_ids,
                    &mut token_tags,
                )?;
                for &ts in block_timestamps {
                    emit_text(
                        tokenizer,
                        &format!("<{ts:.1} seconds>"),
                        &mut token_ids,
                        &mut token_tags,
                    )?;
                }
                emit_vision(
                    cfg,
                    cfg.video_token_id,
                    *token_count,
                    &mut token_ids,
                    &mut token_tags,
                )?;
            }
        }
    }
    emit_text(tokenizer, prompt, &mut token_ids, &mut token_tags)?;
    if token_ids.is_empty() {
        return Err("Ref2VA presentation tokenizes to nothing".into());
    }
    Ok(TextPresentation {
        token_ids,
        token_tags,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::h3::tokenizer::H3Tokenizer;

    fn toy_tokenizer() -> H3Tokenizer {
        // Minimal vocab covering presentation punctuation and words.
        let json = r#"{
            "version": "1.0",
            "truncation": null,
            "padding": null,
            "added_tokens": [],
            "normalizer": null,
            "pre_tokenizer": {"type": "Whitespace"},
            "post_processor": null,
            "decoder": null,
            "model": {
                "type": "WordLevel",
                "vocab": {
                    "[UNK]": 0, "a": 1, "dog": 2, "Picture": 3, "Video": 4, "Audio": 5,
                    "1": 6, "2": 7, ":": 8, "<": 9, ">": 10, "seconds": 11, "0.0": 12
                },
                "unk_token": "[UNK]"
            }
        }"#;
        H3Tokenizer::from_bytes(json.as_bytes()).unwrap()
    }

    #[test]
    fn fl2va_labels_then_prompt() {
        let tok = toy_tokenizer();
        let cfg = H3TextEncoderConfig::fasth3_8step();
        let p = build_fl2va_presentation(&tok, &cfg, "a dog", &[3]).unwrap();
        assert!(p.token_ids.contains(&cfg.vision_start_token_id));
        assert!(p.token_ids.contains(&cfg.image_token_id));
        assert!(p.token_ids.contains(&cfg.vision_end_token_id));
        let pads = p
            .token_ids
            .iter()
            .filter(|&&id| id == cfg.image_token_id)
            .count();
        assert_eq!(pads, 3);
        assert_eq!(p.token_tags.len(), p.token_ids.len());
        assert!(p.token_tags.iter().any(|&t| t == TAG_VIDEO));
        assert!(p.token_tags.iter().any(|&t| t == TAG_TEXT));
    }

    #[test]
    fn ref2va_orders_audio_picture_video() {
        let tok = toy_tokenizer();
        let cfg = H3TextEncoderConfig::fasth3_8step();
        let refs = [
            PresentationRef::Audio,
            PresentationRef::Image { token_count: 2 },
            PresentationRef::Video {
                token_count: 8,
                block_timestamps: vec![0.0, 0.5],
            },
        ];
        let p = build_ref2va_presentation(&tok, &cfg, "a dog", &refs).unwrap();
        let video_pads = p
            .token_ids
            .iter()
            .filter(|&&id| id == cfg.video_token_id)
            .count();
        assert_eq!(video_pads, 8);
        assert_eq!(p.len(), p.token_tags.len());
    }
}
