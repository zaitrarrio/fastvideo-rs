//! One-shot prompt rewrites for batch jobs (fal `prompt_expansion_mode`,
//! `enable_prompt_expansion`, MiniMax `prompt_optimizer`, …): one LLM call
//! that returns the rewritten prompt as plain text, with a system prompt
//! per model family written in the spirit of that family's prompt guide,
//! then cleaned and held under the family's text-encoder budget.

use std::sync::Arc;
use std::time::Duration;

use crate::chat::{hedged, ChatBackend, ChatRequest, UpsampleError};
use crate::tokens::{cut_to_fit, PromptLimit};

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
            (Effort::Balanced, PromptFamily::H3) => (60, 110),
            (Effort::Quality, PromptFamily::H3) => (120, 220),
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
        PromptFamily::H3 => "Rewrite the user's request as a video prompt for MiniMax H3 in its structured format, three labelled sections:
integrated_multimodal_description: the setting, each subject with a concrete look, the action through the shot in present tense, every spoken line inline as Name says <d>[English] the line</d>, then camera language (shot size, angle, push in, pan, tracking), lighting and style.
overall_soundscape: ambience and sound effects.
non_diegetic_music: the score, or None.",
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
    if family == PromptFamily::H3 {
        let shots = match effort {
            Effort::Balanced => "Keep it short: one paragraph of description.",
            Effort::Quality => "Be rich and specific. When the request has distinct shots, write the description as [Shot 1], [Shot 2] … blocks, each a few sentences; otherwise one paragraph.",
        };
        return format!(
            "{}{image}\n{shots}\n\n{H3_RULES}\n- {lo} to {hi} words, at most {max_chars} characters.",
            family_guide(family)
        );
    }
    format!(
        "{}{image}\n\n{COMMON_RULES}\n- {lo} to {hi} words, at most {max_chars} characters.",
        family_guide(family)
    )
}

const H3_RULES: &str = "Rules:
- Keep the user's subjects, intent, quoted text and any style or camera instruction.
- Add concrete visual detail only. Present tense.
- English, the three labels exactly as shown, each section on its own line. No markdown or comments.
- Reply with the prompt only.";

/// H3's section labels (Context-IR, docs/ports/h3-ref2v.md §3).
pub const H3_DESCRIPTION: &str = "integrated_multimodal_description:";
pub const H3_SOUNDSCAPE: &str = "overall_soundscape:";
pub const H3_MUSIC: &str = "non_diegetic_music:";

/// The three sections of an H3 prompt (`[Shot N]` blocks stay in the
/// description, each on its own line).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct H3Sections {
    pub description: String,
    pub soundscape: Option<String>,
    pub music: Option<String>,
}

fn squash(s: &str) -> String {
    s.lines()
        .map(|l| l.split_whitespace().collect::<Vec<_>>().join(" "))
        .filter(|l| !l.is_empty())
        .collect::<Vec<_>>()
        .join("\n")
}

impl H3Sections {
    /// Splits an answer at the labels; text without labels is all
    /// description.
    pub fn parse(s: &str) -> Self {
        let find = |l: &str| s.find(l);
        let (d, sd, m) = (find(H3_DESCRIPTION), find(H3_SOUNDSCAPE), find(H3_MUSIC));
        let next_after = |start: usize| [d, sd, m].into_iter().flatten().filter(|&e| e > start).min().unwrap_or(s.len());
        let desc = match d {
            Some(i) => &s[i + H3_DESCRIPTION.len()..next_after(i)],
            None => &s[..[sd, m].into_iter().flatten().min().unwrap_or(s.len())],
        };
        let section = |o: Option<usize>, label: &str| {
            o.map(|i| squash(&s[i + label.len()..next_after(i)]).replace('\n', " ")).filter(|t| !t.is_empty())
        };
        H3Sections { description: squash(desc), soundscape: section(sd, H3_SOUNDSCAPE), music: section(m, H3_MUSIC) }
    }

    pub fn render(&self) -> String {
        let mut v = vec![format!("{H3_DESCRIPTION} {}", self.description)];
        if let Some(x) = &self.soundscape {
            v.push(format!("{H3_SOUNDSCAPE} {x}"));
        }
        if let Some(x) = &self.music {
            v.push(format!("{H3_MUSIC} {x}"));
        }
        v.join("\n")
    }

    /// Within `limit` by priority: the music becomes `None.`, then the
    /// description is cut at a sentence (else word) boundary, never inside
    /// `<d>…</d>`; the soundscape is kept.
    pub fn fit(mut self, limit: &PromptLimit) -> String {
        let p = self.render();
        if limit.fits(&p) {
            return p;
        }
        if self.music.as_deref().is_some_and(|m| !m.trim_end_matches('.').eq_ignore_ascii_case("none")) {
            self.music = Some("None.".into());
            let p = self.render();
            if limit.fits(&p) {
                return p;
            }
        }
        let desc = self.description.clone();
        let fits = |d: &str| limit.fits(&H3Sections { description: d.to_owned(), ..self.clone() }.render());
        self.description = cut_to_fit(&desc, &fits);
        let p = self.render();
        if limit.fits(&p) {
            p
        } else {
            cut_to_fit(&p, &|s: &str| limit.fits(s))
        }
    }
}

/// The H3 cleaning: the Context-IR sections kept on their own lines (an
/// answer without labels becomes the description), within `limit`.
pub fn clean_h3(answer: &str, limit: &PromptLimit) -> Option<String> {
    let s = strip_wrappers(answer);
    let sec = H3Sections::parse(&s);
    if sec.description.trim().is_empty() {
        return None;
    }
    Some(sec.fit(limit))
}

/// Code fences, a leading `Prompt:` label and wrapping quotes removed
/// (line breaks kept).
fn strip_wrappers(answer: &str) -> String {
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
    match (s.chars().next(), s.chars().last()) {
        (Some(a), Some(b)) if s.chars().count() >= 2 && matches!((a, b), ('"', '"') | ('“', '”') | ('\'', '\'')) => {
            let inner = &s[a.len_utf8()..s.len() - b.len_utf8()];
            // Only a wrapper: no other quote of that kind inside.
            if inner.contains(a) || inner.contains(b) { s.to_owned() } else { inner.to_owned() }
        }
        _ => s.to_owned(),
    }
}

/// The rewrite request (plain text, temperature 0.6).
pub fn request(spec: &RewriteSpec, prompt: &str) -> ChatRequest {
    ChatRequest { system: system_prompt(spec), user: prompt.to_owned(), temperature: 0.6, response_format: None, images: Vec::new() }
}

/// Cleans an LLM answer into a prompt: drops code fences, a leading
/// `Prompt:`-style label and wrapping quotes, joins lines into one
/// paragraph, and cuts it to `max_chars` at the last sentence end (else
/// word) that fits. `None` when nothing is left.
pub fn clean(answer: &str, max_chars: usize) -> Option<String> {
    let s = strip_wrappers(answer);
    let joined = s.split_whitespace().collect::<Vec<_>>().join(" ");
    if joined.is_empty() {
        return None;
    }
    Some(cut(&joined, max_chars))
}

/// [`clean`] within a [`PromptLimit`] (tokens when the family's tokenizer
/// is available); the H3 family keeps its sections ([`clean_h3`]).
pub fn clean_for(family: PromptFamily, answer: &str, limit: &PromptLimit) -> Option<String> {
    if family == PromptFamily::H3 {
        return clean_h3(answer, limit);
    }
    let s = clean(answer, usize::MAX)?;
    if limit.fits(&s) {
        return Some(s);
    }
    let c = cut(&s, limit.max_chars);
    Some(if limit.fits(&c) { c } else { cut_to_fit(&s, &|x: &str| limit.fits(x)) })
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
/// `limit` is the family's prompt limit (tokens with its tokenizer).
pub async fn rewrite(backend: &Arc<dyn ChatBackend>, hedge: bool, timeout: Duration, spec: &RewriteSpec, limit: &PromptLimit, prompt: &str) -> Result<String, UpsampleError> {
    let req = request(spec, prompt);
    let family = spec.family;
    let attempts = if hedge { 2 } else { 1 };
    hedged(attempts, timeout, || async {
        let answer = backend.complete(&req).await?;
        clean_for(family, &answer, limit).ok_or(UpsampleError::NoContent)
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
        assert!(h3.contains("camera language") && h3.contains("<d>[English]") && !h3.contains("image you cannot see"));
        assert!(h3.contains("integrated_multimodal_description:") && h3.contains("overall_soundscape:") && h3.contains("non_diegetic_music:"));
        assert!(h3.contains("one paragraph") && !h3.contains("[Shot 1]") && h3.contains("60 to 110 words"));
        let q = system_prompt(&spec(PromptFamily::H3, Effort::Quality, false, 3200));
        assert!(q.contains("[Shot 1]") && q.contains("120 to 220 words"));
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

    #[test]
    fn h3_answers_keep_their_sections() {
        let lim = PromptLimit::chars(10_000);
        let a = "```\nintegrated_multimodal_description: A keeper climbs.\n  Mara says <d>[English] Light it.</d>\noverall_soundscape: Wind.\nnon_diegetic_music: None.\n```";
        assert_eq!(
            clean_h3(a, &lim).unwrap(),
            "integrated_multimodal_description: A keeper climbs.\nMara says <d>[English] Light it.</d>\noverall_soundscape: Wind.\nnon_diegetic_music: None."
        );
        // No labels: all description.
        assert_eq!(clean_h3("A cat walks.", &lim).unwrap(), "integrated_multimodal_description: A cat walks.");
        // [Shot N] blocks stay on their lines.
        let q = "integrated_multimodal_description:\n[Shot 1] A wide shot.\n[Shot 2] A close-up.\noverall_soundscape: Rain.\nnon_diegetic_music: Strings.";
        assert_eq!(
            clean_h3(q, &lim).unwrap(),
            "integrated_multimodal_description: [Shot 1] A wide shot.\n[Shot 2] A close-up.\noverall_soundscape: Rain.\nnon_diegetic_music: Strings."
        );
        assert_eq!(clean_h3("  ", &lim), None);
    }

    /// Over the limit (tokens by the tokenizer): the music goes to None,
    /// then the description is cut, never inside <d>…</d>, the soundscape
    /// kept.
    #[test]
    fn h3_answers_fit_the_token_limit() {
        use crate::tokens::tests::Words;
        use crate::tokens::TokenCounter;
        let a = "integrated_multimodal_description: A keeper climbs the stairs. Mara says <d>[English] Light the lamp now please.</d> The beam sweeps the sea.\noverall_soundscape: Wind and rain.\nnon_diegetic_music: A low cello drone.";
        let words = |n| PromptLimit { counter: Some(std::sync::Arc::new(Words)), max_tokens: n, max_chars: 1 };
        let full = clean_h3(a, &words(1000)).unwrap();
        let n = Words.count(&full).unwrap();
        let p = clean_h3(a, &words(n - 1)).unwrap();
        assert!(p.ends_with("non_diegetic_music: None.") && p.contains("The beam sweeps the sea."), "{p}");
        for k in 2..(n - 12) {
            let p = clean_h3(a, &words(n - k)).unwrap();
            assert!(Words.count(&p).unwrap() <= n - k, "{p}");
            assert_eq!(p.matches("<d>").count(), p.matches("</d>").count(), "{p}");
            assert!(p.contains("overall_soundscape: Wind and rain."), "{p}");
        }
    }
}
