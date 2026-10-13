//! One-shot prompt rewrites for batch jobs (fal `prompt_expansion_mode`,
//! `enable_prompt_expansion`, MiniMax `prompt_optimizer`, …): one LLM call
//! that returns the rewritten prompt as plain text, with a system prompt
//! per model family written in the spirit of that family's prompt guide,
//! then cleaned and held under the family's text-encoder budget.

use std::sync::Arc;
use std::time::Duration;

use crate::chat::{hedged, ChatBackend, ChatRequest, UpsampleError};

/// Which prompt guide a model follows.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PromptFamily {
    /// Wan 2.x, FastWan, SF-Wan, LongLive (umT5-XXL, 512 tokens).
    Wan,
    /// LTX-2 / 2.3 / 2.5 (Gemma 3, 1024 tokens).
    Ltx,
    /// H3 (Qwen3-VL text tower, 1024 tokens, refused beyond).
    H3,
    /// Anything else.
    Generic,
}

impl PromptFamily {
    pub fn as_str(self) -> &'static str {
        match self {
            PromptFamily::Wan => "wan",
            PromptFamily::Ltx => "ltx",
            PromptFamily::H3 => "h3",
            PromptFamily::Generic => "other",
        }
    }
    /// Whether the family generates sound with the picture.
    pub fn has_audio(self) -> bool {
        matches!(self, PromptFamily::Ltx | PromptFamily::H3)
    }
}

/// How much to rewrite: fal's `balanced` (and H3 base's `fast`) or `quality`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Effort {
    Balanced,
    Quality,
}

impl Effort {
    pub fn as_str(self) -> &'static str {
        match self {
            Effort::Balanced => "balanced",
            Effort::Quality => "quality",
        }
    }
    /// Target length in words.
    fn words(self, family: PromptFamily) -> (u32, u32) {
        match (self, family) {
            (Effort::Balanced, PromptFamily::Wan) => (60, 100),
            (Effort::Quality, PromptFamily::Wan) => (100, 160),
            (Effort::Balanced, _) => (80, 130),
            (Effort::Quality, _) => (140, 200),
        }
    }
}

/// One rewrite's settings.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RewriteSpec {
    pub family: PromptFamily,
    pub effort: Effort,
    /// The video starts from (or is conditioned on) an image the LLM does
    /// not see: describe motion and camera, not a new scene.
    pub image: bool,
    /// The family's prompt budget in characters.
    pub max_chars: usize,
}

// Short and concrete on purpose: the intended local model is small (a 4B
// instruct model), and those follow short rules best.
const COMMON_RULES: &str = "Rules:
- Keep the user's subjects, intent, quoted text and any style or camera instruction.
- Add concrete visual detail only. One continuous shot, present tense.
- One English paragraph. No title, list, markdown or comments.
- Reply with the prompt only.";

fn family_guide(family: PromptFamily) -> &'static str {
    match family {
        PromptFamily::Wan => "Rewrite the user's request as a video prompt.
Order: subject and action, appearance, setting, style, shot and camera motion, lighting.
Stress motion: what moves, where, how fast. This model makes no sound: no audio, no dialogue.",
        PromptFamily::Ltx => "Rewrite the user's request as a video prompt, the way a cinematographer describes a shot.
Start directly with the main action. Then movements, appearance, background, camera angle and movement, lighting and color.
End with the ambience and sound effects. Spoken words go in double quotes after the speaker.",
        PromptFamily::H3 => "Rewrite the user's request as a video prompt.
Order: subject and appearance, scene, action, camera language (shot size, angle, push in, pan, tracking), lighting and mood.
Then the ambience, music and sound effects. Quote spoken lines verbatim after the speaker.",
        PromptFamily::Generic => "Rewrite the user's request as a detailed video prompt.
Describe the subject, action, setting, camera, lighting and mood.",
    }
}

/// The system prompt for one rewrite.
pub fn system_prompt(spec: &RewriteSpec) -> String {
    let RewriteSpec { family, effort, image, max_chars } = *spec;
    let (lo, hi) = effort.words(family);
    let image = if image {
        "\nThe first frame is an image you cannot see: do not describe its look; describe the motion and camera from it."
    } else {
        ""
    };
    format!(
        "{}{image}\n\n{COMMON_RULES}\n- {lo} to {hi} words, at most {max_chars} characters.",
        family_guide(family)
    )
}

/// The rewrite request (plain text, temperature 0.6).
pub fn request(spec: &RewriteSpec, prompt: &str) -> ChatRequest {
    ChatRequest { system: system_prompt(spec), user: prompt.to_owned(), temperature: 0.6, response_format: None }
}

/// Cleans an LLM answer into a prompt: drops code fences, a leading
/// `Prompt:`-style label and wrapping quotes, joins lines into one
/// paragraph, and cuts it to `max_chars` at the last sentence end (else
/// word) that fits. `None` when nothing is left.
pub fn clean(answer: &str, max_chars: usize) -> Option<String> {
    let mut s = crate::chat::strip_think(answer).trim();
    if s.starts_with("```") {
        s = s.trim_start_matches('`');
        // A language tag on the fence line.
        if let Some((first, rest)) = s.split_once('\n') {
            if !first.trim().contains(' ') {
                s = rest;
            }
        }
        s = s.trim_end().trim_end_matches('`');
    }
    let mut s = s.trim();
    for label in ["rewritten prompt:", "enhanced prompt:", "expanded prompt:", "prompt:"] {
        if s.get(..label.len()).is_some_and(|h| h.eq_ignore_ascii_case(label)) {
            s = s[label.len()..].trim_start();
            break;
        }
    }
    let s = s.trim();
    let s = match (s.chars().next(), s.chars().last()) {
        (Some(a), Some(b)) if s.chars().count() >= 2 && matches!((a, b), ('"', '"') | ('“', '”') | ('\'', '\'')) => {
            let inner = &s[a.len_utf8()..s.len() - b.len_utf8()];
            // Only a wrapper: no other quote of that kind inside.
            if inner.contains(a) || inner.contains(b) { s } else { inner }
        }
        _ => s,
    };
    let joined = s.split_whitespace().collect::<Vec<_>>().join(" ");
    if joined.is_empty() {
        return None;
    }
    Some(cut(&joined, max_chars))
}

/// `s` within `max` characters: whole when it fits, else up to the last
/// sentence end that fits (when that keeps at least half), else the last
/// whole word.
pub fn cut(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        return s.to_owned();
    }
    let end = s.char_indices().nth(max).map_or(s.len(), |(i, _)| i);
    let head = &s[..end];
    let sentence = head.rfind(['.', '!', '?']).map(|i| i + 1);
    if let Some(i) = sentence.filter(|&i| i >= head.len() / 2) {
        return head[..i].trim_end().to_owned();
    }
    match head.rfind(' ') {
        Some(i) if i > 0 => head[..i].trim_end().to_owned(),
        _ => head.to_owned(),
    }
}

/// One rewrite: hedged and timed out like the director's calls.
pub async fn rewrite(backend: &Arc<dyn ChatBackend>, hedge: bool, timeout: Duration, spec: &RewriteSpec, prompt: &str) -> Result<String, UpsampleError> {
    let req = request(spec, prompt);
    let max_chars = spec.max_chars;
    let attempts = if hedge { 2 } else { 1 };
    hedged(attempts, timeout, || async {
        let answer = backend.complete(&req).await?;
        clean(&answer, max_chars).ok_or(UpsampleError::NoContent)
    })
    .await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn system_prompts_follow_the_family() {
        let spec = |family, effort, image, max_chars| RewriteSpec { family, effort, image, max_chars };
        let wan = system_prompt(&spec(PromptFamily::Wan, Effort::Balanced, false, 1800));
        assert!(wan.contains("makes no sound") && wan.contains("60 to 100 words") && wan.contains("1800 characters"));
        let ltx = system_prompt(&spec(PromptFamily::Ltx, Effort::Quality, true, 3600));
        assert!(ltx.contains("Start directly with the main action") && ltx.contains("140 to 200 words") && ltx.contains("image you cannot see"));
        let h3 = system_prompt(&spec(PromptFamily::H3, Effort::Balanced, false, 3200));
        assert!(h3.contains("camera language") && h3.contains("verbatim") && !h3.contains("image you cannot see"));
        let r = request(&spec(PromptFamily::H3, Effort::Balanced, false, 3200), "a cat");
        assert_eq!((r.user.as_str(), r.response_format.is_none()), ("a cat", true));
    }

    #[test]
    fn clean_strips_wrappers() {
        assert_eq!(clean("  A cat walks.  ", 100).as_deref(), Some("A cat walks."));
        assert_eq!(clean("\"A cat walks.\"", 100).as_deref(), Some("A cat walks."));
        assert_eq!(clean("Prompt: A cat\nwalks.", 100).as_deref(), Some("A cat walks."));
        assert_eq!(clean("```text\nA cat walks.\n```", 100).as_deref(), Some("A cat walks."));
        assert_eq!(clean("```\nA cat walks.\n```", 100).as_deref(), Some("A cat walks."));
        // Inner quotes keep the wrapper (it is dialogue, not a wrapper).
        assert_eq!(clean("\"Go,\" she says, \"now.\"", 100).as_deref(), Some("\"Go,\" she says, \"now.\""));
        assert_eq!(clean("   ", 100), None);
        assert_eq!(clean("\"\"", 100), None);
    }

    #[test]
    fn cut_prefers_sentences_then_words() {
        let s = "One two three. Four five six. Seven eight nine.";
        assert_eq!(cut(s, 100), s);
        assert_eq!(cut(s, 32), "One two three. Four five six.");
        // Too little of the head is a whole sentence: the last word.
        assert_eq!(cut("Alpha beta gamma delta. Epsilon", 12), "Alpha beta");
        assert_eq!(cut("Supercalifragilistic", 5), "Super");
        assert!(cut(&"word ".repeat(1000), 1800).chars().count() <= 1800);
    }
}
