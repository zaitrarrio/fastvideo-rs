//! Cosmos3 prompt → token ids (`Cosmos3OmniPipeline.tokenize_prompt`).
//!
//! The 64B transformer embeds raw token ids itself (no separate text
//! encoder). For video the pipeline appends a duration and a resolution
//! sentence to the prompt (their negations to the negative prompt), wraps
//! it in the Qwen chat template with the video system prompt and a
//! generation turn, and appends `<|im_end|>` and `<|vision_start|>`.

/// Re-exported so callers without a direct `tokenizers` dependency can hold one.
pub use tokenizers::Tokenizer;

pub const SYSTEM_PROMPT_VIDEO: &str =
    "You are a helpful assistant who will generate videos from a give prompt.";
pub const SYSTEM_PROMPT_IMAGE: &str =
    "You are a helpful assistant who will generate images from a give prompt.";

/// `<|im_end|>` (the tokenizer's eos) and `<|vision_start|>`.
pub const EOS_TOKEN_ID: u32 = 151_645;
pub const VISION_START_TOKEN_ID: u32 = 151_652;

/// `_append`: strip trailing periods, then `"{base}. {addition}"`.
fn append(base: &str, addition: &str) -> String {
    let base = base.trim_end_matches('.');
    if base.is_empty() {
        addition.to_string()
    } else {
        format!("{base}. {addition}")
    }
}

/// Python's `"{:.1f}"` for the duration (round-half-even on the binary value).
fn fmt1(x: f64) -> String {
    format!("{x:.1}")
}

/// The prompt text with the duration / resolution templates applied.
pub fn templated(
    text: &str,
    negative: bool,
    num_frames: usize,
    height: usize,
    width: usize,
    fps: f64,
) -> String {
    let image = num_frames == 1;
    let mut t = text.to_string();
    if !image {
        let d = fmt1(num_frames as f64 / fps);
        let f = format!("{fps:.0}");
        let s = if negative {
            format!("The video is not {d} seconds long and is not of {f} FPS.")
        } else {
            format!("The video is {d} seconds long and is of {f} FPS.")
        };
        t = append(&t, &s);
    }
    let kind = if image { "image" } else { "video" };
    let s = if negative {
        format!("This {kind} is not of {height}x{width} resolution.")
    } else {
        format!("This {kind} is of {height}x{width} resolution.")
    };
    append(&t, &s)
}

/// `apply_chat_template(system + user, add_generation_prompt=True)`.
pub fn chat(text: &str, image: bool) -> String {
    let system = if image {
        SYSTEM_PROMPT_IMAGE
    } else {
        SYSTEM_PROMPT_VIDEO
    };
    format!(
        "<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n"
    )
}

/// Token ids of one (positive or negative) prompt, ready for the transformer.
pub fn token_ids(
    tokenizer: &tokenizers::Tokenizer,
    text: &str,
    negative: bool,
    num_frames: usize,
    height: usize,
    width: usize,
    fps: f64,
) -> Result<Vec<u32>, String> {
    let body = chat(&templated(text, negative, num_frames, height, width, fps), num_frames == 1);
    let enc = tokenizer
        .encode(body.as_str(), false)
        .map_err(|e| format!("cosmos3 tokenize: {e}"))?;
    let mut ids = enc.get_ids().to_vec();
    ids.push(EOS_TOKEN_ID);
    ids.push(VISION_START_TOKEN_ID);
    Ok(ids)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn official_templates() {
        let p = templated("A lake at sunrise.", false, 189, 720, 1280, 24.0);
        assert_eq!(
            p,
            "A lake at sunrise. The video is 7.9 seconds long and is of 24 FPS. \
             This video is of 720x1280 resolution."
        );
        let n = templated("", true, 189, 720, 1280, 24.0);
        assert_eq!(
            n,
            "The video is not 7.9 seconds long and is not of 24 FPS. \
             This video is not of 720x1280 resolution."
        );
        let img = templated("x", false, 1, 512, 512, 24.0);
        assert_eq!(img, "x. This image is of 512x512 resolution.");
    }

    #[test]
    fn chat_wraps_system_and_generation_turn() {
        let c = chat("hi", false);
        assert!(c.starts_with("<|im_start|>system\nYou are a helpful assistant who will generate videos"));
        assert!(c.ends_with("<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n"));
    }
}
